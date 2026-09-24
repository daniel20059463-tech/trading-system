"""同一份盤前美股快照供十檔數值影子預測；正式模型保持不變。"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import config as cfg
import us_lead_study as study
import walkforward_training as wf
from prediction_audit import score

ROOT = Path(__file__).resolve().parent
SEALED = ROOT / "experiments" / "us_lead_20260915" / "aligned.csv"
TZ = ZoneInfo("Asia/Taipei")
TARGETS = ("close_to_close", "gap", "intraday")


def _source_digest() -> str:
    return hashlib.sha256(SEALED.read_bytes()).hexdigest()


def make_candidates(snapshot: dict, day: str, now: datetime | None = None,
                    root: Path = ROOT) -> dict:
    now = now or datetime.now(TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=TZ)
    now = now.astimezone(TZ)
    if day != now.date().isoformat() or now.hour >= 9:
        raise RuntimeError("美股數值影子只能在當日09:00前產生")
    if snapshot.get("date") != day:
        raise ValueError("美股快照日期與目標日期不同")
    metadata = snapshot["metadata"]
    us_features = metadata["us_features"]
    required = set(study.US)
    if not required.issubset(us_features) or not all(np.isfinite(us_features[key]) for key in required):
        raise ValueError("美股快照缺少數值模型所需的完整欄位")
    history = pd.read_csv(SEALED)
    live_report = Path(root) / "reports" / "us_lead_shadow_live.json"
    live_rows = json.loads(live_report.read_text(encoding="utf-8"))["rows"] if live_report.exists() else []
    outputs = {}
    with threadpool_limits(limits=1):
        for ticker in cfg.ALL_STOCKS:
            frame = wf.frame_for(ticker, root, next_target=day, now=now)
            row = frame.iloc[-1].copy()
            asof = row["asof"].date().isoformat()
            if asof != metadata["previous_tw_date"]:
                raise ValueError(f"{ticker} 台股資料截止日與美股快照不一致")
            for key, value in us_features.items():
                row[key] = value
            training = history[(history["ticker"] == ticker) & (history["date"] < day)].copy()
            extra = [record for record in live_rows
                     if record["ticker"] == ticker and record["method"] == "price_us"
                     and record["date"] < day and record["date"] > training["date"].max()
                     and "feature_values" in record and "actual_targets" in record]
            if extra:
                live_frame = pd.DataFrame([{"date": record["date"], "ticker": ticker,
                                            **record["feature_values"], **record["actual_targets"]}
                                           for record in extra])
                training = pd.concat([training, live_frame], ignore_index=True).sort_values("date")
            if len(training) < 60:
                raise ValueError(f"{ticker} 美股歷史訓練樣本不足")
            predictions = {}
            for target in TARGETS:
                for method in ("price", "price_us"):
                    model = study.train(training, method, ticker, target)
                    columns = study.columns(method, ticker)
                    scaled = model.predict(pd.DataFrame([row])[columns].to_numpy(float))[0]
                    predictions[f"{target}_{method}"] = float(scaled * row["vol20"])
            outputs[ticker] = {
                "ticker": ticker, "target_date": day, "data_as_of": asof,
                "last_close": float(row["base_close"]), "trained_target_end": str(training["date"].max()),
                "training_n": int(len(training)), "predictions_pct": predictions,
                "feature_values": {key: float(row[key]) for key in
                                   dict.fromkeys(wf.BASE + study.US + ["vol20"])},
                "us_session": metadata["us_session"], "source_sha256": _source_digest(),
                "status": "shadow_only_not_formal_prediction",
            }
    return outputs


def attach_candidates(predictions: list[dict], snapshot: dict, day: str | None = None,
                      now: datetime | None = None, root: Path = ROOT) -> int:
    day = day or datetime.now(TZ).date().isoformat()
    candidates = make_candidates(snapshot, day, now=now, root=root)
    attached = 0
    for point in predictions:
        ticker = point.get("ticker")
        if ticker in candidates:
            point["us_lead_shadow"] = candidates[ticker]
            attached += 1
    return attached


def settle_day(day: str | None = None, root: Path = ROOT,
               now: datetime | None = None) -> dict:
    """只讀不可覆寫正式留底；當日結果存在才結算。"""
    now = now or datetime.now(TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=TZ)
    now = now.astimezone(TZ)
    day = day or now.date().isoformat()
    if day != now.date().isoformat() or now.hour < 15:
        return {"n": 0, "reason": "only_after_same_day_close"}
    ledger = root / "predictions" / f"{day}_point.json"
    if not ledger.exists():
        return {"n": 0, "reason": "no_frozen_prediction"}
    report = root / "reports" / "us_lead_shadow_live.json"
    settlement = root / "live_evidence" / "us_lead_shadow" / "settlements" / f"{day}.json"
    if settlement.exists():
        saved = json.loads(settlement.read_text(encoding="utf-8"))
        current = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {"rows": []}
        if not any(item["date"] == day for item in current["rows"]):
            current["rows"].extend(saved["rows"])
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"n": len(saved["rows"]), "reason": "already_settled", "path": str(settlement)}
    existing = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {"rows": []}
    if any(item["date"] == day for item in existing["rows"]):
        return {"n": 0, "reason": "legacy_settlement_exists"}
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    settled = []
    for point in payload["records"]:
        candidate = point.get("us_lead_shadow")
        if not candidate:
            continue
        if candidate["target_date"] != day or candidate["trained_target_end"] >= day:
            continue
        for method in ("price", "price_us"):
            prediction = candidate["predictions_pct"][f"close_to_close_{method}"]
            item = {"ticker": point["ticker"], "data_as_of": candidate["data_as_of"],
                    "last_close": candidate["last_close"], "predicted_change_pct": prediction}
            result = score(item, day, root=root, trusted=True)
            if result["eligible"]:
                entry = {"date": day, "ticker": point["ticker"], "method": method,
                                "predicted_pct": prediction, "actual_pct": result["actual_pct"],
                                "abs_error_pp": result["abs_error_pp"],
                                "direction_correct": result["direction_correct"]}
                if method == "price_us":
                    raw_path = root / "data" / "raw" / f"{point['ticker']}.csv"
                    raw = pd.read_csv(raw_path)
                    today = raw[raw["Date"] == day]
                    if len(today) != 1:
                        return {"n": 0, "reason": "missing_open_price"}
                    open_price = float(today.iloc[0]["Open"])
                    if not np.isfinite(open_price) or open_price <= 0:
                        return {"n": 0, "reason": "invalid_open_price"}
                    actual_close = float(result["actual_close"])
                    prior_close = float(candidate["last_close"])
                    entry["feature_values"] = candidate["feature_values"]
                    entry["actual_targets"] = {
                        "close_to_close": result["actual_pct"],
                        "gap": 100 * (open_price / prior_close - 1),
                        "intraday": 100 * (actual_close / open_price - 1),
                    }
                settled.append(entry)
    if len(settled) != 2 * len(cfg.ALL_STOCKS):
        return {"n": 0, "reason": "incomplete_actuals_or_candidates"}
    settlement.parent.mkdir(parents=True, exist_ok=True)
    with settlement.open("x", encoding="utf-8") as handle:
        json.dump({"date": day, "rows": settled}, handle, ensure_ascii=False, indent=2)
    existing["rows"].extend(settled)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"n": len(settled), "days": len({item["date"] for item in existing["rows"]}),
            "path": str(report)}
