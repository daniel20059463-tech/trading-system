"""08:30 盤前快照完成後，以美股核心訊號重新校準當日區間。"""
from __future__ import annotations

import json
import math
import hashlib
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TZ = ZoneInfo("Asia/Taipei")
SEALED = ROOT / "experiments" / "us_lead_20260915" / "predictions.csv"


def _radius(errors: list[float], coverage: float) -> float:
    values = np.sort(np.asarray(errors, dtype=float))
    if not len(values):
        raise ValueError("沒有可校準的歷史誤差")
    rank = min(max(int(math.ceil((len(values) + 1) * coverage)), 1), len(values))
    return float(values[rank - 1])


def _digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def render(payload: dict) -> str:
    lines = ["## 08:30後美股條件式區間", "",
             "中心值來自美股核心數值影子；50%/80%區間由歷史逐日誤差校準。",
             "本組為研究觀察，不覆寫08:30正式預測。", "",
             "| 股票 | 美股條件中心 | 50%區間 | 80%區間 |",
             "|---|---:|---:|---:|"]
    for row in payload["records"]:
        lines.append(f"| {row['ticker']} {row['name']} | {row['center_pct']:+.2f}% | "
                     f"{row['q25_pct']:+.2f}%~{row['q75_pct']:+.2f}% | "
                     f"{row['q10_pct']:+.2f}%~{row['q90_pct']:+.2f}% |")
    return "\n".join(lines)


def settle_day(day: str | None = None, now: datetime | None = None,
               root: Path = ROOT) -> dict:
    """收盤後用正式日線結算，並與08:30原始點預測公平比較。"""
    now = now or datetime.now(TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=TZ)
    now = now.astimezone(TZ)
    day = day or now.date().isoformat()
    if day != now.date().isoformat() or now.time() < time(15, 0):
        return {"n": 0, "reason": "only_after_same_day_close"}
    interval_path = Path(root) / "predictions" / f"{day}_us_context_interval.json"
    point_path = Path(root) / "predictions" / f"{day}_point.json"
    if not interval_path.exists() or not point_path.exists():
        return {"n": 0, "reason": "missing_frozen_inputs"}
    directory = Path(root) / "live_evidence" / "us_context_interval" / "settlements"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{day}.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        return {"n": len(saved["rows"]), "reason": "already_settled", "summary": saved["summary"]}

    intervals = json.loads(interval_path.read_text(encoding="utf-8"))["records"]
    formal = {row["ticker"]: row for row in json.loads(point_path.read_text(encoding="utf-8"))["records"]}
    rows = []
    sign = lambda value: (value > 1e-9) - (value < -1e-9)
    grade = lambda direction_ok, error: ("準確" if direction_ok and error <= 1.0 + 1e-9 else
                                         "方向對、幅度偏差" if direction_ok else "方向錯")
    for item in intervals:
        ticker = item["ticker"]
        raw = pd.read_csv(Path(root) / "data" / "raw" / f"{ticker}.csv")
        actual_row = raw[raw["Date"] == day]
        if len(actual_row) != 1:
            return {"n": 0, "reason": f"missing_actual_{ticker}"}
        actual_close = float(actual_row.iloc[0]["Close"])
        base = float(item["last_close"])
        actual_pct = 100 * (actual_close / base - 1)
        context_pct = float(item["center_pct"])
        formal_pct = float(formal[ticker]["predicted_change_pct"])
        formal_error = abs(formal_pct - actual_pct)
        context_error = abs(context_pct - actual_pct)
        formal_ok = sign(formal_pct) == sign(actual_pct)
        context_ok = sign(context_pct) == sign(actual_pct)
        rows.append({
            "date": day, "ticker": ticker, "name": item.get("name", ""),
            "prior_close": base, "actual_close": actual_close,
            "actual_change_price": actual_close - base,
            "actual_pct": actual_pct, "formal_pct": formal_pct,
            "context_pct": context_pct,
            "formal_predicted_close": base * (1 + formal_pct / 100),
            "context_predicted_close": base * (1 + context_pct / 100),
            "actual_direction": "UP" if actual_pct > 0 else "DOWN" if actual_pct < 0 else "FLAT",
            "formal_abs_error_pp": formal_error, "context_abs_error_pp": context_error,
            "formal_direction_correct": formal_ok, "context_direction_correct": context_ok,
            "formal_result": grade(formal_ok, formal_error),
            "context_result": grade(context_ok, context_error),
            "context_better": context_error < formal_error,
            "covered50": item["q25_pct"] - 1e-9 <= actual_pct <= item["q75_pct"] + 1e-9,
            "covered80": item["q10_pct"] - 1e-9 <= actual_pct <= item["q90_pct"] + 1e-9,
        })
    summary = {
        "n": len(rows),
        "formal_direction_accuracy": sum(row["formal_direction_correct"] for row in rows) / len(rows),
        "context_direction_accuracy": sum(row["context_direction_correct"] for row in rows) / len(rows),
        "formal_mae_pp": sum(row["formal_abs_error_pp"] for row in rows) / len(rows),
        "context_mae_pp": sum(row["context_abs_error_pp"] for row in rows) / len(rows),
        "context_better_count": sum(row["context_better"] for row in rows),
        "coverage50": sum(row["covered50"] for row in rows) / len(rows),
        "coverage80": sum(row["covered80"] for row in rows) / len(rows),
    }
    payload = {"date": day, "settled_at": now.isoformat(), "summary": summary, "rows": rows}
    payload["accuracy_rule"] = "準確=方向正確且幅度絕對誤差≤1.0個百分點；否則分為方向對、幅度偏差／方向錯"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    report = Path(root) / "reports" / "us_context_interval_live.json"
    settlements = [json.loads(file.read_text(encoding="utf-8"))
                   for file in sorted(directory.glob("????-??-??.json"))]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"days": len(settlements), "settlements": settlements},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    daily_report = Path(root) / "reports" / f"{day}_prediction_accuracy.md"
    lines = [f"# {day} 預測準確度", "",
             f"- 原模型：方向 {summary['formal_direction_accuracy']:.0%}，MAE {summary['formal_mae_pp']:.3f}pp",
             f"- 加入美股：方向 {summary['context_direction_accuracy']:.0%}，MAE {summary['context_mae_pp']:.3f}pp",
             f"- 美股版較準：{summary['context_better_count']}/{summary['n']} 檔", "",
             "| 股票 | 前收 | 收盤 | 實際漲跌 | 原預測 | 原結果 | 美股後預測 | 美股後結果 |",
             "|---|---:|---:|---:|---:|---|---:|---|"]
    for row in rows:
        lines.append(f"| {row['ticker']} {row['name']} | {row['prior_close']:.2f} | "
                     f"{row['actual_close']:.2f} | {row['actual_pct']:+.2f}% | "
                     f"{row['formal_pct']:+.2f}% | {row['formal_result']} | "
                     f"{row['context_pct']:+.2f}% | {row['context_result']} |")
    lines += ["", "_判定規則：方向正確且幅度絕對誤差≤1.0pp=準確。完整數值仍保留，不只看二元標籤。_", ""]
    daily_report.write_text("\n".join(lines), encoding="utf-8")
    return {"n": len(rows), "summary": summary, "path": str(path)}


def build(day: str | None = None, now: datetime | None = None,
          root: Path = ROOT) -> dict:
    now = now or datetime.now(TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=TZ)
    now = now.astimezone(TZ)
    day = day or now.date().isoformat()
    if day != now.date().isoformat() or not (time(8, 33) <= now.time() < time(9, 0)):
        raise RuntimeError("盤前條件式區間只能在當日08:33至09:00建立")

    point_path = Path(root) / "predictions" / f"{day}_point.json"
    if not point_path.exists():
        raise FileNotFoundError("尚未完成08:30盤前點預測")
    output = Path(root) / "predictions" / f"{day}_us_context_interval.json"
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))

    history = pd.read_csv(SEALED)
    live_path = Path(root) / "reports" / "us_lead_shadow_live.json"
    live = json.loads(live_path.read_text(encoding="utf-8")).get("rows", []) if live_path.exists() else []
    point = json.loads(point_path.read_text(encoding="utf-8"))
    records = []
    for item in point["records"]:
        candidate = item.get("us_lead_shadow")
        if not candidate or candidate.get("target_date") != day:
            raise ValueError(f"{item.get('ticker')} 缺少當日美股數值影子")
        ticker = item["ticker"]
        rows = history[(history["ticker"] == ticker) &
                       (history["target"] == "close_to_close") &
                       (history["method"] == "price_us") &
                       (history["date"] < day)].sort_values("date")
        errors = rows["abs_error_pp"].dropna().astype(float).tolist()
        errors.extend(float(row["abs_error_pp"]) for row in live
                      if row.get("ticker") == ticker and row.get("method") == "price_us"
                      and row.get("date", day) < day and row.get("abs_error_pp") is not None)
        errors = errors[-126:]
        if len(errors) < 60:
            raise ValueError(f"{ticker} 美股區間校準樣本不足")
        center = float(candidate["predictions_pct"]["close_to_close_price_us"])
        r50, r80 = _radius(errors, .50), _radius(errors, .80)
        lo50, hi50 = max(-10.0, center - r50), min(10.0, center + r50)
        lo80, hi80 = max(-10.0, center - r80), min(10.0, center + r80)
        base = float(item["last_close"])
        records.append({
            "ticker": ticker, "name": item.get("name", ""), "target_date": day,
            "data_as_of": item["data_as_of"], "generated_at": now.isoformat(),
            "last_close": base, "center_pct": round(center, 4),
            "q25_pct": round(lo50, 4), "q75_pct": round(hi50, 4),
            "q10_pct": round(lo80, 4), "q90_pct": round(hi80, 4),
            "q25_price": round(base * (1 + lo50 / 100), 2),
            "q75_price": round(base * (1 + hi50 / 100), 2),
            "q10_price": round(base * (1 + lo80 / 100), 2),
            "q90_price": round(base * (1 + hi80 / 100), 2),
            "calibration_n": len(errors), "trained_target_end": candidate["trained_target_end"],
            "us_session": candidate["us_session"],
            "status": "shadow_context_interval_not_formal_prediction",
        })
    us_snapshot = Path(root) / "live_evidence" / "us_analogs" / f"{day}.json"
    news_snapshot = Path(root) / "news" / f"{day}_pre_market.json"
    payload = {
        "schema_version": 1, "kind": "us_context_interval", "target_date": day,
        "generated_at": now.isoformat(), "source_point": str(point_path),
        "method": "price_us center plus trailing conformal absolute-error radii",
        "news_policy": "snapshot recorded; no numeric shift until live validation passes",
        "source_sha256": {"point": _digest(point_path), "us_snapshot": _digest(us_snapshot),
                          "news_snapshot": _digest(news_snapshot)},
        "records": records,
    }
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(content)
    return payload


def main() -> None:
    payload = build()
    report = ROOT / "reports" / f"{payload['target_date']}_us_context_interval.md"
    if not report.exists():
        report.parent.mkdir(parents=True, exist_ok=True)
        note = render(payload)
        with report.open("x", encoding="utf-8") as handle:
            handle.write(note + "\n")
        try:
            from agents.wiki_agent import append_intraday_note
            append_intraday_note(payload["target_date"], note, "08:30後條件區間")
        except Exception as error:
            print(f"  Wiki 追加失敗：{error}")
    try:
        from push_intervals import push_context_to_discord
        if push_context_to_discord(payload, ROOT):
            print("  最終單一區間已推播到 Discord")
    except Exception as error:
        print(f"  Discord 單一區間推播失敗：{error}")
    print(f"美股盤前條件式區間已留底：{len(payload['records'])} 支")
    for row in payload["records"]:
        print(f"  {row['ticker']} 中心 {row['center_pct']:+.2f}% | "
              f"50% {row['q25_pct']:+.2f}%~{row['q75_pct']:+.2f}% | "
              f"80% {row['q10_pct']:+.2f}%~{row['q90_pct']:+.2f}%")


if __name__ == "__main__":
    main()
