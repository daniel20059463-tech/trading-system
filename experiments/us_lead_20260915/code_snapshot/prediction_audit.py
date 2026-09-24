"""離線逐筆稽核與不可覆寫盤前留底。只使用標準函式庫，不連網、不推播。"""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from functools import lru_cache

HERE = Path(__file__).resolve().parent
TZ = timezone(timedelta(hours=8))


def number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def prices(root, ticker):
    path = Path(root) / "data" / "raw" / f"{ticker}.csv"
    if not path.exists():
        return {}
    stat = path.stat()
    return _read_prices(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=128)
def _read_prices(path, modified_ns, size):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {r["Date"][:10]: number(r.get("Close")) for r in csv.DictReader(f)}


def require_pre_market(now=None):
    now = now or datetime.now(TZ)
    if now.tzinfo is None:
        raise ValueError("預測時間必須包含時區")
    now = now.astimezone(TZ)
    if now.weekday() >= 5 or now.time() >= time(9):
        raise ValueError("已非盤前時段，拒絕產生／覆寫今日盤前預測")
    return now


def freeze(predictions, kind="point", root=HERE, now=None):
    now = require_pre_market(now)
    if kind not in ("point", "interval"):
        raise ValueError(kind)
    root = Path(root)
    directory = root / "predictions"
    directory.mkdir(exist_ok=True)
    target = now.date().isoformat()
    records = []
    for original in predictions:
        p = dict(original)
        asof = p.get("data_as_of")
        if not asof or asof >= target:
            raise ValueError(f"{p.get('ticker')} 缺少正確資料截止日")
        p.update(target_date=target, generated_at=now.isoformat(), horizon_sessions=1)
        records.append(p)
    if not records:
        raise ValueError("無預測可留底")
    fingerprints = {}
    for ticker in {r["ticker"] for r in records}:
        for file in sorted((root / "models" / "checkpoints").glob(f"{ticker}_*")):
            if file.is_file():
                fingerprints[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    payload = dict(schema_version=1, kind=kind, target_date=target,
                   generated_at=now.isoformat(), model_sha256=fingerprints, records=records)
    # x 模式：重跑不得更換已留底的預測。先完整序列化，避免型別錯誤留下半檔。
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    path = directory / f"{target}_{kind}.json"
    with path.open("x", encoding="utf-8") as f:
        f.write(content)
    return path


def score(p, target, root=HERE, trusted=False):
    ticker = p.get("ticker", "")
    history = prices(root, ticker)
    prior = sorted(d for d in history if d < target)
    base_date = prior[-1] if prior else None
    base = number(p.get("last_close"))
    actual = history.get(target)
    reasons = [] if trusted else ["legacy_timestamp_unverified"]
    if trusted and not p.get("data_as_of"):
        reasons.append("missing_data_as_of")
    if p.get("data_as_of") and p["data_as_of"] != base_date:
        reasons.append("data_date_mismatch")
    if base is None or base <= 0 or not base_date:
        reasons.append("missing_base")
    elif history[base_date] is None or abs(base - history[base_date]) > max(0.02, base * 0.0001):
        reasons.append("base_price_mismatch")
    if actual is None or actual <= 0:
        reasons.append("missing_actual")
    predicted = number(p.get("predicted_change_pct", p.get("q50_pct")))
    if predicted is None:
        reasons.append("missing_prediction")
    aligned = not any(r != "legacy_timestamp_unverified" for r in reasons)
    actual_pct = (actual / base - 1) * 100 if aligned else None
    row = dict(date=target, ticker=ticker, generated_at=p.get("generated_at"),
               data_as_of=p.get("data_as_of"), expected_base_date=base_date,
               base_close=base, predicted_close=p.get("predicted_close", p.get("q50_price")),
               actual_close=actual, predicted_pct=predicted, actual_pct=actual_pct,
               eligible=not reasons, status=";".join(reasons) or "verified",
               direction_correct=None, abs_error_pp=None, baseline_abs_error_pp=None)
    if aligned:
        sign = lambda x: (x > 1e-9) - (x < -1e-9)
        row.update(direction_correct=sign(predicted) == sign(actual_pct),
                   abs_error_pp=abs(predicted - actual_pct), baseline_abs_error_pp=abs(actual_pct))
    for label, low, high in (("50", "q25_pct", "q75_pct"), ("80", "q10_pct", "q90_pct"),
                             ("80s", "q10s_pct", "q90s_pct")):
        lo, hi = number(p.get(low)), number(p.get(high))
        valid = lo is not None and hi is not None and lo <= hi
        row[f"lo{label}_pct"], row[f"hi{label}_pct"] = lo, hi
        row[f"width{label}_pp"] = hi - lo if valid else None
        row[f"covered{label}"] = lo <= actual_pct <= hi if valid and aligned else None
    return row


def saved_predictions(target, root=HERE):
    """供盤後舊介面使用，只交付通過時間／基準價格檢查的盤前原值。"""
    path = Path(root) / "predictions" / f"{target}_point.json"
    if not path.exists():
        return [], {}
    payload = read_json(path)
    out, actual = [], {}
    for p in payload["records"]:
        row = score(p, target, root, trusted=valid_time(p, target))
        if row["eligible"]:
            out.append(p)
            actual[p["ticker"].split(".")[0]] = row["actual_close"]
    return out, actual


def valid_time(p, target):
    try:
        stamp = require_pre_market(datetime.fromisoformat(p["generated_at"]))
        return stamp.date().isoformat() == target == p["target_date"] and p["horizon_sessions"] == 1
    except (KeyError, ValueError, TypeError):
        return False


def valid_daily_candidate(point, target):
    candidate = point.get("daily_retrain", {})
    if not valid_time(point, target):
        return False
    try:
        if not (candidate["ticker"] == point["ticker"] and candidate["target_date"] == target and
                candidate["trained_target_end"] == candidate["data_as_of"] < target):
            return False
        available = candidate.get("context_available_at")
        return not available or datetime.fromisoformat(available) < datetime.fromisoformat(point["generated_at"])
    except (KeyError, ValueError, TypeError):
        return False


def collect(root=HERE, wiki=None):
    root = Path(root)
    rows, issues = [], []
    ledger_days = set()
    for path in sorted((root / "predictions").glob("*.json")):
        try:
            payload = read_json(path)
            target, kind = payload["target_date"], payload["kind"]
            if kind == "point":
                ledger_days.add(target)
            for p in payload["records"]:
                row = score(p, target, root, valid_time(p, target))
                row.update(kind=kind, source=str(path.relative_to(root)))
                rows.append(row)
                if kind == "point" and isinstance(p.get("daily_retrain"), dict):
                    candidate = dict(p["daily_retrain"], generated_at=p.get("generated_at"))
                    trusted = valid_daily_candidate(p, target)
                    daily_row = score(candidate, target, root, trusted)
                    daily_row.update(kind="daily_retrain", source=str(path.relative_to(root)),
                                     daily_protocol=candidate.get("protocol"),
                                     train_target_end=candidate.get("trained_target_end"))
                    rows.append(daily_row)
                if kind == "point" and isinstance(p.get("shadow_candidate"), dict):
                    shadow = dict(p["shadow_candidate"])
                    shadow.update(target_date=target, generated_at=p.get("generated_at"),
                                  horizon_sessions=p.get("horizon_sessions"),
                                  last_close=p.get("last_close"))
                    shadow_row = score(shadow, target, root, valid_time(p, target))
                    shadow_row.update(kind="shadow_candidate", source=str(path.relative_to(root)),
                                      shadow_version=shadow.get("version"))
                    rows.append(shadow_row)
        except (ValueError, KeyError, TypeError) as e:
            issues.append(f"{path.name}: {type(e).__name__}")
    for path in sorted((root / "news").glob("*_pre_market.json")):
        target = path.name[:10]
        if target in ledger_days:
            continue
        try:
            payload = read_json(path)
            post_path = root / "news" / f"{target}_post_market.json"
            post = read_json(post_path).get("ticker_performance", {}) if post_path.exists() else {}
            for code, record in payload.items():
                p = record.get("ml_prediction") if isinstance(record, dict) else None
                if not p:
                    continue
                row = score(p, target, root)
                log = root / "logs" / f"{target}_pre.log"
                starts = re.findall(r"\[" + target + r" (\d{4})\] start session=pre",
                                    log.read_text(encoding="utf-8-sig", errors="replace")) if log.exists() else []
                row["pre_log_start_times"] = ";".join(starts)
                row["late_pre_run_observed"] = any(t >= "0900" for t in starts)
                perf = post.get(code, {})
                row.update(kind="point", source=str(path.relative_to(root)),
                           post_record_predicted=perf.get("predicted"),
                           post_record_actual=perf.get("actual", perf.get("actual_price")))
                row["post_prediction_matches"] = (abs(number(perf.get("predicted")) - number(p.get("predicted_close"))) < 0.011
                    if number(perf.get("predicted")) is not None and number(p.get("predicted_close")) is not None else None)
                rows.append(row)
        except (ValueError, KeyError, TypeError) as e:
            issues.append(f"{path.name}: {type(e).__name__}")
    # 舊 Wiki 區間保留原始價格寬度；缺少可追溯的模型基準／目標日，不推算涵蓋率。
    if wiki is None and root.resolve() == HERE:
        import config as cfg
        wiki = cfg.WIKI_DIR
    if wiki:
        for path in sorted(Path(wiki).glob("????-??-??.md")):
            target = path.stem
            if (root / "predictions" / f"{target}_interval.json").exists():
                continue
            content = path.read_text(encoding="utf-8-sig", errors="replace")
            sections = content.split("## 隔日區間預測")[1:]
            stamp = re.search(r"建立時間：([^\n]+)", content)
            for revision, section in enumerate(sections):
                for line in section.split("\n## ")[0].splitlines():
                    cells = [s.strip() for s in line.strip().strip("|").split("|")]
                    if len(cells) < 7 or not re.fullmatch(r"\d+\.TW(?:O)?", cells[0]):
                        continue
                    p = dict(ticker=cells[0], q50_pct=number(cells[3].replace("%", "")))
                    row = score(p, target, root)
                    row.update(kind="interval", source=str(path), status="legacy_interval_target_unverified",
                               wiki_created_at=stamp[1].strip() if stamp else None, wiki_revision=revision)
                    for label, cell in zip(("50", "80", "80s"), cells[4:7]):
                        bounds = cell.split("~")
                        if len(bounds) == 2:
                            lo, hi = map(number, bounds)
                            row[f"lo{label}_price"], row[f"hi{label}_price"] = lo, hi
                            row[f"width{label}_price"] = hi - lo if lo is not None and hi is not None and hi >= lo else None
                    rows.append(row)
    # 日期涵蓋 CSV 已有交易資料的完整範圍；盤前缺席也必須出現在逐股對照。
    present = {(r["date"], r["ticker"]) for r in rows if r["kind"] == "point"}
    if rows:
        first = min(r["date"] for r in rows)
        # 結束日依已到日期的行情，而非最後一份預測；否則最新整天缺席會消失。
        last = datetime.now(TZ).date().isoformat()
        for ticker in sorted({r["ticker"] for r in rows}):
            for target in prices(root, ticker):
                if first <= target <= last and (target, ticker) not in present:
                    row = score({"ticker": ticker}, target, root)
                    row.update(kind="point", source="missing", status="missing_pre_prediction")
                    rows.append(row)
    return rows, issues


def summarize(rows):
    metrics = {}
    for key in ("direction_correct", "abs_error_pp", "baseline_abs_error_pp", "width50_pp", "covered50",
                "width80_pp", "covered80", "width80s_pp", "covered80s"):
        values = [r[key] for r in rows if r.get(key) is not None]
        metrics[key] = {"n": len(values), "mean": sum(values) / len(values) if values else None}
    return metrics


def update(root=HERE, output=None):
    root = Path(root)
    output = Path(output) if output else root / "reports"
    output.mkdir(exist_ok=True, parents=True)
    rows, issues = collect(root)
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else ["date"]
    with (output / "prediction_audit.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["date"], r["ticker"], r["kind"])))
    latest = max((r["date"] for r in rows), default=datetime.now(TZ).date().isoformat())
    cutoff = (datetime.fromisoformat(latest) - timedelta(days=28)).date().isoformat()
    recent = [r for r in rows if r["date"] >= cutoff]
    result = dict(latest_target=latest, recent_since=cutoff, rows=len(rows), recent_rows=len(recent),
                  statuses=dict(Counter(r["status"] for r in recent)), issues=issues,
                  verified={kind: summarize([r for r in recent if r["eligible"] and r["kind"] == kind])
                            for kind in ("point", "interval", "shadow_candidate", "daily_retrain")},
                  legacy_aligned_diagnostic=summarize([r for r in recent if r["kind"] == "point" and not r["eligible"]]),
                  recent_post_prediction_mismatches=sum(r.get("post_prediction_matches") is False for r in recent))
    sources = {root / "data" / "raw" / f"{r['ticker']}.csv" for r in rows}
    sources.update(Path(r["source"]) if Path(r["source"]).is_absolute() else root / r["source"]
                   for r in rows if r["source"] != "missing")
    manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources) if p.is_file()}
    (output / "prediction_audit_sources.json").write_text(json.dumps(
        dict(generated_at=datetime.now(TZ).isoformat(), sha256=manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "prediction_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 每日預測稽核", "", f"資料截至 {latest}；近期範圍 {cutoff} 起。共 {len(rows)} 筆，近期 {len(recent)} 筆。",
             "", "原始逐股逐日對照見 prediction_audit.csv；缺失保留空白，每項指標各自列有效樣本數。",
             "歷史新聞 JSON 沒有可驗證的發布時間，且可能遭重跑覆寫，全部列為未驗證，不能充當實盤成績。",
             "方向以漲／平／跌三類判定；幅度誤差為百分點，並列猜不變的幅度誤差。50%/80% 是名目涵蓋率，不是保證。",
             "σ5/σ20 都預測下一交易日；5/20 只代表波動估計回看天數。舊 Wiki 區間原始價格與價差另存 CSV；缺少可靠目標日和基準，不推算百分點寬度／涵蓋率，不以現在模型補算。", "",
             "| 分組／指標 | 平均 | 有效樣本 |", "|---|---:|---:|"]
    for group, stats in [("已驗證點預測", result["verified"]["point"]), ("已驗證區間", result["verified"]["interval"]),
                         ("探索性影子候選（未來資料）", result["verified"]["shadow_candidate"]),
                         ("每日重訓候選（未來資料）", result["verified"]["daily_retrain"]),
                         ("舊檔可配對診斷（非實盤績效）", result["legacy_aligned_diagnostic"])]:
        for key, v in stats.items():
            names = dict(direction_correct="方向準確率", abs_error_pp="幅度絕對誤差（百分點）",
                         baseline_abs_error_pp="猜不變誤差（百分點）", width50_pp="50%區間寬度（百分點）",
                         width80_pp="80%區間寬度σ20（百分點）", width80s_pp="80%區間寬度σ5（百分點）",
                         covered50="50%區間涵蓋率", covered80="80%區間涵蓋率σ20", covered80s="80%區間涵蓋率σ5")
            percentage = key == "direction_correct" or key.startswith("covered")
            val = "缺失" if v["mean"] is None else (f"{v['mean']:.2%}" if percentage else f"{v['mean']:.4f}")
            lines.append(f"| {group} / {names[key]} | {val} | {v['n']} |")
    lines += ["", f"近期盤後與盤前預測值不同：{result['recent_post_prediction_mismatches']} 筆。", "", "排除原因："]
    lines += [f"- {reason}：{n}" for reason, n in result["statuses"].items()]
    lines += ["", "未驗證成績不得用於模型晉升。每日更新代表增加可稽核資料，並不保證每天進步。"]
    (output / "prediction_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    print(json.dumps(update(output=args.output), ensure_ascii=False, indent=2))
