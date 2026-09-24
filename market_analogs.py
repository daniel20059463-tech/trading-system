"""美股已完成交易日 -> 下一個台股開盤日的歷史對照與相似行情查詢。

基礎資料沿用 us_lead_20260915 的已校驗時區對齊與封存輸入；
相似案例只可來自查詢日之前，並只用查詢時已知的美股欄位比較。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "experiments" / "us_lead_20260915"
OUT = ROOT / "experiments" / "market_analogs_20260916"
FEATURES = ["sox_last", "nasdaq_last", "sp500_last", "on_last", "vsh_last"]
LABELS = ["gap", "intraday", "close_to_close"]


def build_library(source: Path = SOURCE, out: Path = OUT) -> tuple[pd.DataFrame, pd.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    aligned = pd.read_csv(source / "aligned.csv")
    aligned = aligned[aligned["alignment_status"].eq("ok")].copy()
    # 同一天必須有十檔完整台股資料及相同美股訊號，避免缺股改變日平均。
    grouped = aligned.groupby("date", sort=True)
    complete_dates = [d for d, g in grouped if g["ticker"].nunique() == 10]
    aligned = aligned[aligned["date"].isin(complete_dates)].copy()
    cols = ["date", "ticker", "previous_tw_date", "us_session", "us_close_taipei",
            "us_available_taipei", "cutoff_taipei", "new_us_sessions", "repeated_session",
            *FEATURES, "sox_since_tw", "nasdaq_since_tw", "sp500_since_tw",
            "on_since_tw", "vsh_since_tw", "Open", "Close", *LABELS]
    stocks = aligned[cols].sort_values(["date", "ticker"]).reset_index(drop=True)
    assert stocks[[*FEATURES, *LABELS]].notna().all().all()
    assert (pd.to_datetime(stocks["us_available_taipei"], utc=True)
            < pd.to_datetime(stocks["cutoff_taipei"], utc=True)).all()
    assert (stocks["date"] > stocks["us_session"]).all()

    daily = stocks.groupby("date", sort=True).agg(
        previous_tw_date=("previous_tw_date", "first"),
        us_session=("us_session", "first"),
        us_available_taipei=("us_available_taipei", "first"),
        new_us_sessions=("new_us_sessions", "first"),
        repeated_session=("repeated_session", "first"),
        **{c: (c, "first") for c in FEATURES},
        gap_mean=("gap", "mean"), intraday_mean=("intraday", "mean"),
        close_to_close_mean=("close_to_close", "mean"),
        gap_up_share=("gap", lambda s: float((s > 0).mean())),
        intraday_up_share=("intraday", lambda s: float((s > 0).mean())),
        close_to_close_up_share=("close_to_close", lambda s: float((s > 0).mean())),
        stock_count=("ticker", "nunique"),
    ).reset_index()
    for date, group in stocks.groupby("date"):
        row = daily[daily["date"].eq(date)].iloc[0]
        for feature in FEATURES:
            assert np.allclose(group[feature], row[feature])
    daily.to_csv(out / "us_to_tw_daily.csv", index=False, encoding="utf-8-sig")
    stocks.to_csv(out / "us_to_tw_stocks.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "source": str(source / "aligned.csv"),
        "source_status": "2026-09-15 sealed exploratory historical alignment",
        "first_tw_date": daily["date"].min(), "last_tw_date": daily["date"].max(),
        "days": len(daily), "stocks_per_day": 10,
        "us_features": FEATURES,
        "tw_outcomes": {"gap": "Open / prior Close - 1", "intraday": "Close / Open - 1",
                        "close_to_close": "Close / prior Close - 1"},
        "cutoff": "08:30 Asia/Taipei; US market close + 60 minutes availability",
        "caveat": "US/TW prices are vendor historical adjusted data; analogs are descriptive, not validated forecasts",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return daily, stocks


def find_analogs(query: dict, daily: pd.DataFrame, stocks: pd.DataFrame,
                 before_date: str, ticker: str | None = None, k: int = 8) -> dict:
    """固定欄位與距離尺度；不讀歷史台股結果來挑相似日。"""
    missing = [f for f in FEATURES if f not in query or not np.isfinite(float(query[f]))]
    if missing:
        raise ValueError(f"缺少美股行情欄位：{missing}")
    pool = daily[daily["date"] < before_date].copy()
    if "new_us_sessions" in query:
        pool = pool[pool["new_us_sessions"].eq(int(query["new_us_sessions"]))]
    if pool.empty:
        raise ValueError("查詢日前沒有符合交易日條件的歷史案例")
    # 每個變數依歷史分布縮放；費半與三大指數的影響力保持可見，
    # 同業個股較雜，權重設為一半。尺度只取候選池，無未來資料。
    scales = pool[FEATURES].std(ddof=0).clip(lower=0.5)
    weights = np.array([1.0, 1.0, 1.0, 0.5, 0.5])
    delta = (pool[FEATURES].to_numpy(float) - np.array([float(query[f]) for f in FEATURES]))
    pool["distance"] = np.sqrt(np.mean(((delta / scales.to_numpy()) * weights) ** 2, axis=1))
    nearest = pool.sort_values(["distance", "date"]).head(k)
    case_rows = []
    for _, row in nearest.iterrows():
        case = {key: (float(row[key]) if key in FEATURES or key.endswith("_mean") or key == "distance"
                      else str(row[key]))
                for key in ["date", "us_session", "distance", *FEATURES,
                            "gap_mean", "intraday_mean", "close_to_close_mean"]}
        if ticker:
            actual = stocks[(stocks["date"] == row["date"]) & (stocks["ticker"] == ticker)]
            if actual.empty:
                raise ValueError(f"資料沒有股票 {ticker}")
            case["ticker"] = ticker
            case.update({label: float(actual.iloc[0][label]) for label in LABELS})
        case_rows.append(case)
    metrics = {}
    for label in LABELS:
        values = np.array([row[label if ticker else label + "_mean"] for row in case_rows], dtype=float)
        metrics[label] = {"mean_pct": float(values.mean()), "median_pct": float(np.median(values)),
                          "up_share": float((values > 0).mean()), "range_min": float(values.min()),
                          "range_max": float(values.max())}
    return {"query_date": before_date, "ticker": ticker or "TEN_STOCK_EQUAL_WEIGHT",
            "candidate_days": len(pool), "case_count": len(case_rows),
            "us_input": {f: float(query[f]) for f in FEATURES},
            "new_us_sessions": query.get("new_us_sessions"),
            "cases": case_rows, "summary": metrics,
            "note": "歷史相似案例是描述性參考；不等於未來報酬率或經驗證的交易訊號"}


def render_markdown(result: dict) -> str:
    lines = [f"# 美股行情與隔日台股相似案例：{result['query_date']}", "",
             f"對象：{result['ticker']}；候選歷史日 {result['candidate_days']} 天；列出最相似 {result['case_count']} 天。", "",
             "查詢時美股變動：" + "、".join(f"{f} {result['us_input'][f]:+.2f}%" for f in FEATURES), "",
             "| 台股結果 | 相似日平均 | 中位數 | 上漲比例 | 範圍 |",
             "|---|---:|---:|---:|---:|"]
    names = {"gap": "開盤跳空", "intraday": "開盤至收盤", "close_to_close": "前收至今收"}
    for label, values in result["summary"].items():
        lines.append(f"| {names[label]} | {values['mean_pct']:+.2f}% | {values['median_pct']:+.2f}% | "
                     f"{values['up_share']:.0%} | {values['range_min']:+.2f}%～{values['range_max']:+.2f}% |")
    lines += ["", "| 台股日 | 對應美股日 | 距離 | 費半 | 那指 | 標普 | ON | VSH | 開盤跳空 | 盤中 | 收盤 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["cases"]:
        value = lambda label: row[label if "ticker" in row else label + "_mean"]
        lines.append(f"| {row['date']} | {row['us_session']} | {row['distance']:.2f} | "
                     + " | ".join(f"{row[f]:+.2f}%" for f in FEATURES) + " | "
                     + " | ".join(f"{value(label):+.2f}%" for label in LABELS) + " |")
    lines += ["", result["note"], "美股可用時間按收盤後一小時與台北 08:30 截止確認；查詢日當天及其後不進入相似案例。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="查詢的台股日期；資料庫內日期會用該日美股行情，並只找更早案例")
    parser.add_argument("--ticker", help="例如 2327.TW；省略則為十檔等權平均")
    parser.add_argument("--k", type=int, default=8)
    for feature in FEATURES:
        parser.add_argument("--" + feature.replace("_last", ""), type=float,
                            help="新日期盤前查詢時，輸入最後已完成美股交易日的漲跌幅百分點")
    parser.add_argument("--new-us-sessions", type=int, default=None)
    args = parser.parse_args()
    daily, stocks = build_library()
    manual = {f: getattr(args, f.replace("_last", "")) for f in FEATURES}
    if any(v is not None for v in manual.values()):
        if not args.date:
            parser.error("手動輸入美股行情時必須指定 --date")
        if any(v is None for v in manual.values()):
            parser.error("手動查詢需要五項美股變動值")
        query = manual
        if args.new_us_sessions is not None:
            query["new_us_sessions"] = args.new_us_sessions
        day = args.date
    else:
        day = args.date or daily["date"].max()
        row = daily[daily["date"] == day]
        if row.empty:
            parser.error("該日期不在封存資料內；請提供五項美股變動值")
        query = {f: float(row.iloc[0][f]) for f in FEATURES}
        query["new_us_sessions"] = int(row.iloc[0]["new_us_sessions"])
    result = find_analogs(query, daily, stocks, before_date=day, ticker=args.ticker, k=args.k)
    stem = day + ("_" + args.ticker.replace(".", "_") if args.ticker else "_all")
    (OUT / f"{stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / f"{stem}.md").write_text(render_markdown(result), encoding="utf-8")
    print(render_markdown(result))


if __name__ == "__main__":
    main()
