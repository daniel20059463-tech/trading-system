"""逐日先前案例回放：檢查相似行情能否勝過簡單基準。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from market_analogs import FEATURES, LABELS, OUT, find_analogs


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(OUT / "us_to_tw_daily.csv")
    stocks = pd.read_csv(OUT / "us_to_tw_stocks.csv")
    records = []
    for _, day in daily.iloc[60:].iterrows():
        query = {feature: day[feature] for feature in FEATURES}
        query["new_us_sessions"] = day["new_us_sessions"]
        try:
            reference = find_analogs(query, daily, stocks, before_date=day["date"], k=8)
        except ValueError:
            continue
        similar_dates = [case["date"] for case in reference["cases"]]
        history = stocks[stocks["date"].isin(similar_dates)]
        today = stocks[stocks["date"].eq(day["date"])]
        for _, row in today.iterrows():
            ticker = row["ticker"]
            past = history[history["ticker"].eq(ticker)]
            if len(past) != len(similar_dates):
                continue
            for label in LABELS:
                actual = float(row[label])
                prediction = float(past[label].mean())
                records.append({"date": day["date"], "ticker": ticker, "target": label,
                                "analog_pred": prediction, "actual": actual,
                                "analog_abs_error": abs(prediction - actual),
                                "zero_abs_error": abs(actual),
                                "analog_correct": np.sign(prediction) == np.sign(actual),
                                "case_count": len(similar_dates)})
    pred = pd.DataFrame(records)
    rows = []
    dates = sorted(pred["date"].unique())
    for window, frame in [("all", pred), ("last120", pred[pred["date"].isin(dates[-120:])]),
                          ("last60", pred[pred["date"].isin(dates[-60:])])]:
        for label, subset in frame.groupby("target"):
            rows.extend([
                {"window": window, "target": label, "method": "analog_8_mean", "n": len(subset),
                 "days": subset["date"].nunique(), "mae": subset["analog_abs_error"].mean(),
                 "accuracy": subset["analog_correct"].mean()},
                {"window": window, "target": label, "method": "zero", "n": len(subset),
                 "days": subset["date"].nunique(), "mae": subset["zero_abs_error"].mean(),
                 "accuracy": np.nan},
            ])
    summary = pd.DataFrame(rows)
    pred.to_csv(OUT / "analog_replay.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "analog_replay_summary.csv", index=False, encoding="utf-8-sig")
    return pred, summary


if __name__ == "__main__":
    _, result = evaluate()
    print(result.to_string(index=False))
