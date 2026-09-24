"""將歷史盤前預測與正式收盤價整併成單一、可稽核的逐檔記錄。"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

import config as cfg

ROOT = Path(__file__).resolve().parent


def _prices(root: Path, ticker: str) -> dict[str, float]:
    path = root / "data" / "raw" / f"{ticker}.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, usecols=["Date", "Close"])
    return {str(row.Date)[:10]: float(row.Close) for row in frame.itertuples()
            if math.isfinite(float(row.Close)) and float(row.Close) > 0}


def _grade(ok: bool | None, error: float | None) -> str | None:
    if ok is None or error is None:
        return None
    if not ok:
        return "方向錯"
    return "準確" if error <= 1.0 + 1e-9 else "方向對、幅度偏差"


def _legacy_records(root: Path) -> dict[tuple[str, str], dict]:
    records = {}
    for path in sorted((root / "news").glob("????-??-??_pre_market.json")):
        day = path.name[:10]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in payload.values():
            if not isinstance(item, dict) or not isinstance(item.get("ml_prediction"), dict):
                continue
            prediction = dict(item["ml_prediction"])
            ticker = prediction.get("ticker")
            if ticker in cfg.ALL_STOCKS:
                records[(day, ticker)] = {"source": "legacy_news_json", "source_path": str(path),
                                          "prediction": prediction, "generated_at": None}
    return records


def _point_records(root: Path) -> dict[tuple[str, str], dict]:
    records = {}
    for path in sorted((root / "predictions").glob("????-??-??_point.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        day = payload["target_date"]
        for prediction in payload.get("records", []):
            records[(day, prediction["ticker"])] = {
                "source": "immutable_point", "source_path": str(path),
                "prediction": prediction,
                "generated_at": prediction.get("generated_at", payload.get("generated_at")),
            }
    return records


def rebuild(root: Path = ROOT) -> dict:
    root = Path(root)
    records = _legacy_records(root)
    records.update(_point_records(root))  # 同日以不可覆寫留底為準
    price_cache = {ticker: _prices(root, ticker) for ticker in cfg.ALL_STOCKS}
    rows = []
    for (day, ticker), source in sorted(records.items()):
        prediction = source["prediction"]
        history = price_cache[ticker]
        prior_days = sorted(date for date in history if date < day and pd.Timestamp(date).weekday() < 5)
        prior_day = prior_days[-1] if prior_days else None
        prior_close = history.get(prior_day) if prior_day else None
        actual_close = history.get(day)
        stored_base = prediction.get("last_close")
        predicted_pct = prediction.get("predicted_change_pct")
        reasons = []
        if source["source"] == "legacy_news_json":
            reasons.append("legacy_timestamp_unverified")
        else:
            generated = source.get("generated_at") or ""
            if not generated.startswith(day + "T08:"):
                reasons.append("not_verified_pre_market_time")
        if actual_close is None:
            reasons.append("missing_actual")
        if prior_close is None:
            reasons.append("missing_prior_close")
        if stored_base is None or prior_close is None or abs(float(stored_base) - prior_close) > max(.02, prior_close * .0001):
            reasons.append("base_price_mismatch")
        data_as_of = prediction.get("data_as_of")
        if source["source"] == "immutable_point" and data_as_of != prior_day:
            reasons.append("data_date_mismatch")
        market_actual_available = actual_close is not None and prior_close is not None
        actual_pct = 100 * (actual_close / prior_close - 1) if market_actual_available else None
        aligned = (market_actual_available and predicted_pct is not None
                   and "base_price_mismatch" not in reasons and "data_date_mismatch" not in reasons)
        predicted_pct = float(predicted_pct) if predicted_pct is not None else None
        sign = lambda value: (value > 1e-9) - (value < -1e-9)
        direction_ok = sign(predicted_pct) == sign(actual_pct) if aligned else None
        error = abs(predicted_pct - actual_pct) if aligned else None
        trusted = source["source"] == "immutable_point" and not reasons
        rows.append({
            "date": day, "ticker": ticker, "name": cfg.ALL_STOCKS[ticker],
            "source": source["source"], "source_path": source["source_path"],
            "generated_at": source.get("generated_at"), "official_eligible": trusted,
            "status": "verified" if trusted else ";".join(reasons) or "diagnostic_only",
            "prior_date": prior_day, "prior_close": prior_close,
            "actual_close": actual_close,
            "actual_change_price": actual_close - prior_close if market_actual_available else None,
            "actual_pct": actual_pct, "predicted_pct": predicted_pct,
            "predicted_close": prediction.get("predicted_close"),
            "direction_correct": direction_ok, "abs_error_pp": error,
            "accuracy_result": _grade(direction_ok, error),
            "has_us_shadow": bool(prediction.get("us_lead_shadow")),
        })
    frame = pd.DataFrame(rows)
    output = root / "reports" / "historical_prediction_records.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")

    trusted = frame[frame["official_eligible"] == True]
    diagnostic = frame[(frame["official_eligible"] == False) & frame["abs_error_pp"].notna()]
    def metrics(part):
        return {"rows": int(len(part)), "days": int(part["date"].nunique()),
                "direction_accuracy": float(part["direction_correct"].mean()) if len(part) else None,
                "mae_pp": float(part["abs_error_pp"].mean()) if len(part) else None,
                "accurate_count": int((part["accuracy_result"] == "準確").sum())}
    summary = {"total_rows": len(frame), "dates": int(frame["date"].nunique()),
               "date_start": frame["date"].min(), "date_end": frame["date"].max(),
               "official": metrics(trusted), "legacy_diagnostic": metrics(diagnostic),
               "missing_actual": int(frame["actual_close"].isna().sum()),
               "invalid_or_unverified": int((frame["official_eligible"] == False).sum())}
    (root / "reports" / "historical_prediction_records.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    lines = ["# 歷史預測與收盤對照", "",
             f"日期：{summary['date_start']} 至 {summary['date_end']}；{summary['dates']} 天，{summary['total_rows']} 筆。", "",
             f"- 正式可計分：{summary['official']['days']} 天/{summary['official']['rows']} 筆；"
             f"方向 {summary['official']['direction_accuracy']:.1%}，MAE {summary['official']['mae_pp']:.3f}pp",
             f"- 舊檔診斷：{summary['legacy_diagnostic']['days']} 天/{summary['legacy_diagnostic']['rows']} 筆；"
             f"不列入正式績效", "",
             "舊新記錄皆保留前收、收盤、實際漲跌幅、預測幅度、方向、誤差及排除原因。",
             "CSV 為完整逐筆明細。", ""]
    (root / "reports" / "historical_prediction_records.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(rebuild(), ensure_ascii=False, indent=2))
