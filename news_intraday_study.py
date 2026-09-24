"""盤前新聞對「開盤到收盤」漲幅的逐日回放研究。

只接受有日誌證明在台股 09:00 前完成的盤前摘要；模型在每個日期只用更早日期。
歷史 JSON 可被覆寫，因此結果一律標為 reconstructed / exploratory，不能冒充實盤。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments" / "news_intraday_20260916"
INPUTS = OUT / "inputs"
MIN_PRIOR_DAYS = 20
MIN_TRAIN_ROWS = 150
RIDGE_ALPHA = 100.0
MAX_ADJUSTMENT_PP = 1.5


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _pre_market_saved_before_open(log_path: Path, day: str) -> tuple[bool, str | None]:
    if not log_path.exists():
        return False, None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    stamps = re.findall(
        rf"({re.escape(day)}\s+(\d{{2}}):(\d{{2}}):\d{{2}},\d+).*盤前分析已存至",
        text,
    )
    if len(stamps) != 1:
        return False, None
    stamp, hour, minute = stamps[0]
    return (int(hour), int(minute)) < (9, 0), stamp


def audit_and_snapshot_inputs() -> pd.DataFrame:
    (INPUTS / "news").mkdir(parents=True, exist_ok=True)
    (INPUTS / "logs").mkdir(parents=True, exist_ok=True)
    rows = []
    for news_path in sorted((ROOT / "news").glob("*_pre_market.json")):
        day = news_path.name[:10]
        log_path = ROOT / "logs" / f"{day}_pre.log"
        ok, saved_at = _pre_market_saved_before_open(log_path, day)
        reason = "one logged save before 09:00" if ok else "missing/ambiguous/not-before-09:00 log"
        rows.append({"date": day, "eligible": ok, "saved_at": saved_at, "reason": reason})
        if ok:
            shutil.copy2(news_path, INPUTS / "news" / news_path.name)
            shutil.copy2(log_path, INPUTS / "logs" / log_path.name)

    source_predictions = ROOT / "experiments" / "us_lead_20260915" / "predictions.csv"
    shutil.copy2(source_predictions, INPUTS / "us_lead_predictions.csv")
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT / "availability_audit.csv", index=False, encoding="utf-8-sig")
    manifest = []
    for path in sorted(INPUTS.rglob("*")):
        if path.is_file():
            manifest.append({
                "path": str(path.relative_to(OUT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    (OUT / "inputs_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def write_locked_plan() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {
        "locked_at": datetime.now().astimezone().isoformat(),
        "status": "reconstructed exploratory replay; historical JSON files were mutable",
        "question": "Does a pre-09:00 news correction improve same-day open-to-close magnitude?",
        "primary_comparison": "price_plus_news versus price, paired stock/date rows",
        "target": "100 * (Close / Open - 1)",
        "availability": "exactly one log line proving pre-market JSON save before 09:00",
        "base": "price intraday prediction from sealed us_lead_20260915 replay",
        "news_model": {
            "type": "pooled expanding Ridge residual correction",
            "alpha": RIDGE_ALPHA,
            "minimum_prior_dates": MIN_PRIOR_DAYS,
            "minimum_rows": MIN_TRAIN_ROWS,
            "scaling": "past rows only",
            "max_adjustment_pp": MAX_ADJUSTMENT_PP,
            "features": [
                "base_pred", "sentiment_score", "abs_sentiment", "confidence",
                "log1p_news_count", "buy", "sell", "hold", "base_x_sentiment",
                "ticker one-hot",
            ],
        },
        "metrics": ["direction_accuracy", "MAE percentage points"],
        "promotion_rule": "none; live shadow collection required before formal use",
    }
    (OUT / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")


def _news_rows() -> pd.DataFrame:
    rows = []
    for path in sorted((INPUTS / "news").glob("*_pre_market.json")):
        day = path.name[:10]
        obj = json.loads(path.read_text(encoding="utf-8"))
        for code, item in obj.items():
            if code == "market" or not isinstance(item, dict):
                continue
            ticker = str(item.get("ml_prediction", {}).get("ticker", ""))
            if not ticker:
                ticker = next((t for t in [f"{code}.TW", f"{code}.TWO"]
                               if (ROOT / "experiments" / "us_lead_20260915" / "inputs" / "data" / "raw" / f"{t}.csv").exists()), "")
            signal = str(item.get("strategy_signal", "hold")).lower()
            score = float(item.get("sentiment_score", 0.0) or 0.0)
            rows.append({
                "date": day,
                "ticker": ticker,
                "sentiment_score": score,
                "abs_sentiment": abs(score),
                "confidence": float(item.get("confidence", 0.0) or 0.0),
                "log1p_news_count": math.log1p(float(item.get("news_count", 0) or 0)),
                "buy": float(signal == "buy"),
                "sell": float(signal == "sell"),
                "hold": float(signal in {"hold", "watch"}),
            })
    return pd.DataFrame(rows)


def _design(df: pd.DataFrame, ticker_levels: list[str]) -> np.ndarray:
    numeric = df[[
        "base_pred", "sentiment_score", "abs_sentiment", "confidence",
        "log1p_news_count", "buy", "sell", "hold", "base_x_sentiment",
    ]].to_numpy(float)
    one_hot = np.column_stack([(df["ticker"] == t).to_numpy(float) for t in ticker_levels])
    return np.column_stack([numeric, one_hot])


def run_replay() -> tuple[pd.DataFrame, pd.DataFrame]:
    news = _news_rows()
    pred = pd.read_csv(INPUTS / "us_lead_predictions.csv")
    base = pred[(pred["target"] == "intraday") & (pred["method"] == "price")].copy()
    base = base.rename(columns={"predicted_pct": "base_pred", "actual_pct": "actual"})
    merged = news.merge(base[["date", "ticker", "base_pred", "actual", "train_end"]],
                        on=["date", "ticker"], how="inner", validate="one_to_one")
    merged["base_x_sentiment"] = merged["base_pred"] * merged["sentiment_score"]
    merged = merged.sort_values(["date", "ticker"]).reset_index(drop=True)
    levels = sorted(merged["ticker"].unique())
    results = []
    dates = sorted(merged["date"].unique())
    for day in dates:
        prior_days = [d for d in dates if d < day]
        train = merged[merged["date"] < day]
        test = merged[merged["date"] == day]
        if len(prior_days) < MIN_PRIOR_DAYS or len(train) < MIN_TRAIN_ROWS:
            continue
        x_train = _design(train, levels)
        x_test = _design(test, levels)
        scaler = StandardScaler().fit(x_train)
        model = Ridge(alpha=RIDGE_ALPHA).fit(
            scaler.transform(x_train), (train["actual"] - train["base_pred"]).to_numpy(float)
        )
        adjustment = np.clip(model.predict(scaler.transform(x_test)),
                             -MAX_ADJUSTMENT_PP, MAX_ADJUSTMENT_PP)
        for (_, row), adj in zip(test.iterrows(), adjustment):
            final = float(row["base_pred"] + adj)
            actual = float(row["actual"])
            results.append({
                "date": day, "ticker": row["ticker"], "train_end": max(prior_days),
                "training_days": len(prior_days), "training_rows": len(train),
                "base_pred": row["base_pred"], "news_adjustment": adj,
                "news_pred": final, "actual": actual,
                "base_abs_error": abs(float(row["base_pred"]) - actual),
                "news_abs_error": abs(final - actual),
                "base_correct": bool(np.sign(float(row["base_pred"])) == np.sign(actual)),
                "news_correct": bool(np.sign(final) == np.sign(actual)),
                "sentiment_score": row["sentiment_score"],
            })
    out = pd.DataFrame(results)
    if out.empty:
        out.to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")
        summary = pd.DataFrame(columns=["window", "method", "n", "days", "accuracy", "mae"])
        summary.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
        return out, summary
    summaries = []
    for label, sub in [("all_scored", out), ("last20_dates", out[out["date"].isin(sorted(out["date"].unique())[-20:])])]:
        if sub.empty:
            continue
        summaries.extend([
            {"window": label, "method": "price", "n": len(sub), "days": sub["date"].nunique(),
             "accuracy": sub["base_correct"].mean(), "mae": sub["base_abs_error"].mean()},
            {"window": label, "method": "price_plus_news", "n": len(sub), "days": sub["date"].nunique(),
             "accuracy": sub["news_correct"].mean(), "mae": sub["news_abs_error"].mean()},
        ])
    summary = pd.DataFrame(summaries)
    out.to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    return out, summary


def write_report(audit: pd.DataFrame, replay: pd.DataFrame, summary: pd.DataFrame) -> Path:
    eligible = audit[audit["eligible"]]
    first = replay["date"].min() if not replay.empty else "N/A"
    last = replay["date"].max() if not replay.empty else "N/A"
    lines = [
        "# 盤前新聞對盤中漲幅的逐日驗證（2026-09-16）", "",
        "## 結論", "",
    ]
    if not summary.empty:
        all_s = summary[summary["window"] == "all_scored"].set_index("method")
        b, n = all_s.loc["price"], all_s.loc["price_plus_news"]
        lines += [
            f"- 配對樣本 {int(b['n'])} 筆、{int(b['days'])} 個交易日（{first}～{last}）。",
            f"- 價格基準：方向 {b['accuracy']:.1%}，MAE {b['mae']:.4f} 個百分點。",
            f"- 加入新聞修正：方向 {n['accuracy']:.1%}，MAE {n['mae']:.4f} 個百分點。",
            f"- 差異：方向 {(n['accuracy']-b['accuracy']):+.1%}，MAE {(n['mae']-b['mae']):+.4f} 個百分點（負值才是改善）。",
        ]
    else:
        lines += [
            "- 嚴格可用資料不足，未產生模型分數：規格要求至少 20 個更早交易日，現有可結算盤前日只有 10 天。",
            "- 不能據此宣稱新聞能改善盤中漲幅；需從現在開始累積不可變的實盤影子樣本。",
        ]
    lines += [
        "", "## 資料限制", "",
        f"- 找到 {len(audit)} 份盤前檔，其中 {len(eligible)} 份有日誌證明在 09:00 前完成。",
        "- 舊 JSON 並非不可變快照，無法證明之後從未被重跑覆寫；所以本報告是重建回放，不是實盤成績。",
        "- 先前新聞模型預測的是隔日收盤方向；本次才把標籤改成同日開盤到收盤。",
        "- 所有修正每天只用更早日期，幅度限制在 ±1.5 個百分點；未依結果挑參數。",
        "", "## 決策", "",
        "- 歷史重建結果只決定是否值得收集 live shadow，不直接改正式預測。",
        "- 從下一次盤前開始應保存含完成時間的不可變新聞快照，盤後以 Open→Close 結算。",
        "- 原本的強新聞直接翻多規則與此標籤不一致，應停止改寫正式點預測，改列為影子訊號。",
    ]
    report = OUT / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    write_locked_plan()
    audit = audit_and_snapshot_inputs()
    replay, summary = run_replay()
    report = write_report(audit, replay, summary)
    print(summary.to_string(index=False))
    print(f"report={report}")


if __name__ == "__main__":
    main()
