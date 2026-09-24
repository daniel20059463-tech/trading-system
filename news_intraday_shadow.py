"""盤前新聞的同日 Open→Close 影子預測與不可變實盤留底。"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import config as cfg

ROOT = Path(__file__).resolve().parent
EVIDENCE_DIR = ROOT / "live_evidence" / "news_intraday"
TZ = ZoneInfo("Asia/Taipei")
MIN_LIVE_DAYS = 20
MIN_LIVE_ROWS = 150


def _code(ticker: str) -> str:
    return str(ticker).replace(".TWO", "").replace(".TW", "")


def _features(item: dict, ticker: str, levels: list[str]) -> list[float]:
    score = float(item.get("sentiment_score", 0.0) or 0.0)
    confidence = float(item.get("confidence", 0.0) or 0.0)
    signal = str(item.get("strategy_signal", "hold")).lower()
    return [
        score, abs(score), confidence,
        math.log1p(float(item.get("news_count", 0) or 0)),
        float(signal == "buy"), float(signal == "sell"),
        float(signal in {"hold", "watch"}),
        *[float(ticker == level) for level in levels],
    ]


def _settled_rows(directory: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(directory.glob("????-??-??.json")):
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        settlement_path = directory / "settlements" / path.name
        if not settlement_path.exists():
            continue
        settlement = json.loads(settlement_path.read_text(encoding="utf-8"))
        outcomes = settlement.get("tickers", {})
        for ticker, item in snapshot.get("tickers", {}).items():
            outcome = outcomes.get(ticker)
            if not outcome:
                continue
            rows.append({"date": snapshot["date"], "ticker": ticker, **item, **outcome})
    return pd.DataFrame(rows)


def _make_candidates(news_analysis: dict, directory: Path) -> tuple[dict, dict]:
    levels = sorted(cfg.ALL_STOCKS)
    history = _settled_rows(directory)
    live_days = history["date"].nunique() if not history.empty else 0
    can_train = live_days >= MIN_LIVE_DAYS and len(history) >= MIN_LIVE_ROWS
    scaler = model = None
    if can_train:
        x = np.asarray([row["features"] for _, row in history.iterrows()], dtype=float)
        y = history["actual_intraday_pct"].to_numpy(float)
        scaler = StandardScaler().fit(x)
        model = Ridge(alpha=100.0).fit(scaler.transform(x), y)

    candidates = {}
    for ticker in levels:
        item = news_analysis.get(_code(ticker), {})
        features = _features(item, ticker, levels)
        if model is not None:
            pred = float(np.clip(model.predict(scaler.transform([features]))[0], -1.5, 1.5))
            status = "live_trained_shadow"
        else:
            score = float(item.get("sentiment_score", 0.0) or 0.0)
            confidence = float(item.get("confidence", 0.0) or 0.0)
            pred = float(np.clip(0.5 * score * confidence, -0.4, 0.4))
            status = "warmup_shadow"
        candidates[ticker] = {
            "target": "same_day_open_to_close_pct",
            "predicted_intraday_pct": round(pred, 4),
            "sentiment_score": float(item.get("sentiment_score", 0.0) or 0.0),
            "features": features,
            "status": status,
            "formal_strategy_input": False,
        }
    meta = {"training_days": int(live_days), "training_rows": int(len(history)),
            "minimum_days": MIN_LIVE_DAYS, "minimum_rows": MIN_LIVE_ROWS}
    return candidates, meta


def capture_pre_market(news_analysis: dict, now: datetime | None = None,
                       directory: Path | None = None,
                       source_articles: dict | None = None) -> dict:
    """在 09:00 前以 exclusive-create 保存；同日重跑只讀第一份。"""
    now = now or datetime.now(TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=TZ)
    now = now.astimezone(TZ)
    if (now.hour, now.minute) >= (9, 0):
        raise RuntimeError("新聞影子快照只能在台股 09:00 前建立")
    directory = directory or EVIDENCE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{now.date().isoformat()}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    candidates, training = _make_candidates(news_analysis, directory)
    payload = {
        "schema_version": 1,
        "date": now.date().isoformat(),
        "captured_at": now.isoformat(),
        "cutoff": "09:00 Asia/Taipei",
        "target": "100 * (Close / Open - 1)",
        "evidence_status": "live_shadow",
        "training": training,
        "source_articles": source_articles or {},
        "tickers": candidates,
    }
    with path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def attach_to_strategies(strategies: list[dict], snapshot: dict) -> list[dict]:
    by_ticker = snapshot.get("tickers", {})
    for strategy in strategies:
        ticker = strategy.get("ticker", "")
        candidate = by_ticker.get(ticker)
        if candidate:
            strategy["intraday_news_shadow"] = {
                k: v for k, v in candidate.items() if k != "features"
            }
    return strategies


def settle_day(day: str | None = None, directory: Path | None = None,
               now: datetime | None = None) -> dict:
    """盤後用當日 Open/Close 結算；另存結果，不修改盤前快照。"""
    now = now or datetime.now(TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=TZ)
    now = now.astimezone(TZ)
    day = day or now.date().isoformat()
    if day != now.date().isoformat() or (now.hour, now.minute) < (13, 40):
        raise RuntimeError("新聞影子只能在當日收盤後結算")
    directory = directory or EVIDENCE_DIR
    path = directory / f"{day}.json"
    if not path.exists():
        return {"settled": 0, "reason": "no_snapshot"}
    settlement_dir = directory / "settlements"
    settlement_dir.mkdir(parents=True, exist_ok=True)
    settlement_path = settlement_dir / f"{day}.json"
    if settlement_path.exists():
        existing = json.loads(settlement_path.read_text(encoding="utf-8"))
        return {"settled": len(existing.get("tickers", {})), "path": str(settlement_path),
                "reason": "already_settled"}
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    outcomes = {}
    for ticker, item in snapshot.get("tickers", {}).items():
        raw = Path(cfg.RAW_DATA_DIR) / f"{ticker}.csv"
        if not raw.exists():
            return {"settled": 0, "reason": f"missing_stock_{ticker}"}
        df = pd.read_csv(raw)
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        row = df[pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d") == day]
        if len(row) != 1:
            return {"settled": 0, "reason": f"incomplete_stock_{ticker}"}
        open_price = float(row.iloc[-1]["Open"])
        close_price = float(row.iloc[-1]["Close"])
        if not all(math.isfinite(value) and value > 0 for value in (open_price, close_price)):
            return {"settled": 0, "reason": f"invalid_stock_{ticker}"}
        actual = 100.0 * (close_price / open_price - 1.0)
        pred = float(item["predicted_intraday_pct"])
        outcomes[ticker] = {
            "actual_intraday_pct": round(actual, 4),
            "absolute_error_pp": round(abs(pred - actual), 4),
            "zero_baseline_error_pp": round(abs(actual), 4),
            "direction_correct": bool(np.sign(pred) == np.sign(actual)),
        }
    if len(outcomes) == len(cfg.ALL_STOCKS):
        settlement = {"date": day, "snapshot": path.name,
                      "settled_at": now.isoformat(), "tickers": outcomes}
        with settlement_path.open("x", encoding="utf-8") as f:
            json.dump(settlement, f, ensure_ascii=False, indent=2)
    return {"settled": len(outcomes), "path": str(settlement_path)}


def evidence_report(directory: Path | None = None) -> dict:
    history = _settled_rows(directory or EVIDENCE_DIR)
    if history.empty:
        return {"days": 0, "rows": 0, "status": "collecting"}
    accuracy = float(history["direction_correct"].mean())
    mae = float(history["absolute_error_pp"].mean())
    days = int(history["date"].nunique())
    by_day = []
    for day, group in history.groupby("date", sort=True):
        day_accuracy = float(group["direction_correct"].mean())
        day_mae = float(group["absolute_error_pp"].mean())
        baseline_mae = float(group["zero_baseline_error_pp"].mean())
        by_day.append({
            "date": day, "rows": int(len(group)), "accuracy": day_accuracy,
            "mae": day_mae, "zero_baseline_mae": baseline_mae,
            "good_day": bool(day_accuracy >= 0.60 and day_mae <= baseline_mae),
        })
    status = "ready_for_review" if days >= MIN_LIVE_DAYS and len(history) >= MIN_LIVE_ROWS else "collecting"
    return {"days": days, "rows": int(len(history)), "accuracy": accuracy,
            "mae": mae, "status": status,
            "good_days": [row["date"] for row in by_day if row["good_day"]],
            "by_day": by_day}
