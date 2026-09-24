"""08:30 自動建立美股→台股相似行情快照，僅供策略分析參考。"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import config as cfg
from market_analogs import FEATURES, OUT, find_analogs, render_markdown

TZ = ZoneInfo("Asia/Taipei")
SNAPSHOTS = Path(__file__).resolve().parent / "live_evidence" / "us_analogs"
SYMBOLS = {"sox_last": "^SOX", "nasdaq_last": "^IXIC", "sp500_last": "^GSPC",
           "on_last": "ON", "vsh_last": "VSH"}


def load_reference_library(snapshot_dir: Path = SNAPSHOTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """封存歷史加上已獨立盤後結算的實盤日期。"""
    daily = pd.read_csv(OUT / "us_to_tw_daily.csv")
    stocks = pd.read_csv(OUT / "us_to_tw_stocks.csv")
    settlement_dir = snapshot_dir / "settlements"
    daily_extra, stock_extra = [], []
    for path in sorted(settlement_dir.glob("????-??-??.json")) if settlement_dir.exists() else []:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if len(payload.get("stocks", {})) != len(cfg.ALL_STOCKS):
            continue
        day = payload["date"]
        if day in set(daily["date"]):
            continue
        snapshot_path = snapshot_dir / f"{day}.json"
        if not snapshot_path.exists():
            continue
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        input_us = snapshot["aggregate"]["us_input"]
        metadata = snapshot["metadata"]
        per_stock = payload["stocks"]
        daily_extra.append({
            "date": day, "previous_tw_date": metadata["previous_tw_date"],
            "us_session": metadata["us_session"],
            "us_available_taipei": metadata["us_available_taipei"],
            "new_us_sessions": snapshot["aggregate"]["new_us_sessions"],
            "repeated_session": snapshot["aggregate"]["new_us_sessions"] == 0,
            **input_us,
            **{label + "_mean": float(sum(v[label] for v in per_stock.values()) / len(per_stock))
               for label in ["gap", "intraday", "close_to_close"]},
            **{label + "_up_share": float(sum(v[label] > 0 for v in per_stock.values()) / len(per_stock))
               for label in ["gap", "intraday", "close_to_close"]},
            "stock_count": len(per_stock),
        })
        for ticker, values in per_stock.items():
            stock_extra.append({"date": day, "ticker": ticker, **values})
    if daily_extra:
        daily = pd.concat([daily, pd.DataFrame(daily_extra)], ignore_index=True).sort_values("date")
        stocks = pd.concat([stocks, pd.DataFrame(stock_extra)], ignore_index=True).sort_values(["date", "ticker"])
    return daily, stocks


def _previous_tw_date(day: str) -> str:
    dates = []
    for ticker in cfg.ALL_STOCKS:
        path = Path(cfg.RAW_DATA_DIR) / f"{ticker}.csv"
        if not path.exists():
            raise ValueError(f"缺少台股資料：{ticker}")
        df = pd.read_csv(path, usecols=["Date"])
        parsed = pd.to_datetime(df["Date"], errors="coerce")
        prior = df.loc[(df["Date"] < day) & (parsed.dt.weekday < 5), "Date"]
        if prior.empty:
            raise ValueError(f"缺少前一個台股交易日：{ticker}")
        dates.append(prior.max())
    if len(set(dates)) != 1:
        raise ValueError("十檔台股資料的前一交易日不一致")
    return dates[0]


def _live_us_query(day: str, previous_tw_date: str, history_getter=None) -> tuple[dict, dict]:
    """使用紐約交易所實際排程核對五項資料的最新日與前一美股日。"""
    import yfinance as yf
    from us_lead_study import schedule

    provider_is_default = history_getter is None
    history_getter = history_getter or (lambda symbol: yf.Ticker(symbol).history(
        period="20d", auto_adjust=True))
    calendar = schedule((pd.Timestamp(day) - pd.Timedelta(days=35)).strftime("%Y-%m-%d"), day)
    cutoff = pd.Timestamp(f"{day}T08:30:00", tz="Asia/Taipei").tz_convert("UTC")
    eligible = calendar[calendar["available_at"] < cutoff]
    if len(eligible) < 2:
        raise ValueError("查詢日前沒有足夠的已完成美股交易日")
    latest, previous = eligible.iloc[-1], eligible.iloc[-2]
    prior_tw_close = pd.Timestamp(f"{previous_tw_date}T13:30:00", tz="Asia/Taipei").tz_convert("UTC")
    anchors = eligible[eligible["market_close"] <= prior_tw_close]
    if anchors.empty:
        raise ValueError("缺少前次台股收盤前的美股基準交易日")
    anchor = anchors.iloc[-1]
    query = {"new_us_sessions": int((eligible["market_close"] > prior_tw_close).sum())}
    metadata = {"us_session": latest["session"],
                "us_available_taipei": latest["available_at"].tz_convert("Asia/Taipei").isoformat(),
                "previous_tw_date": previous_tw_date, "source": "Yahoo Finance adjusted daily close",
                "symbols": SYMBOLS, "us_age_hours": (cutoff - latest["market_close"]).total_seconds() / 3600,
                "anchor_us_session": anchor["session"]}
    missing_symbols = []
    us_features = {"new_us_sessions": query["new_us_sessions"],
                   "us_age_hours": metadata["us_age_hours"]}
    for feature, symbol in SYMBOLS.items():
        last_error = None
        for attempt in range(3):
            try:
                hist = history_getter(symbol)
                if hist is None or hist.empty or "Close" not in hist:
                    raise ValueError(f"美股資料缺漏：{symbol}")
                dates = pd.to_datetime(hist.index).strftime("%Y-%m-%d")
                closes = pd.Series(pd.to_numeric(hist["Close"], errors="coerce").to_numpy(), index=dates)
                required = (latest["session"], previous["session"], anchor["session"])
                if any(session not in closes.index for session in required):
                    raise ValueError(f"美股資料過期或缺少基準收盤：{symbol}")
                current = float(closes.loc[latest["session"]])
                earlier = float(closes.loc[previous["session"]])
                anchor_close = float(closes.loc[anchor["session"]])
                if not all(math.isfinite(value) and value > 0 for value in (current, earlier, anchor_close)):
                    raise ValueError(f"美股收盤價無效：{symbol}")
                query[feature] = (current / earlier - 1.0) * 100.0
                us_features[feature] = query[feature]
                us_features[feature.replace("_last", "_since_tw")] = (current / anchor_close - 1.0) * 100.0
                break
            except Exception as error:
                last_error = error
                if attempt < 2 and provider_is_default:
                    time.sleep(0.5)
        else:
            # 三大指數是數值影子的核心；ON/VSH 只用於五維相似案例。
            # 個股報價延遲時保留核心訊號，不再拖垮整組美股資料。
            if feature in ("sox_last", "nasdaq_last", "sp500_last"):
                raise ValueError(f"{symbol} 連續三次無法取得有效的同日行情：{last_error}")
            missing_symbols.append(symbol)
    metadata["missing_symbols"] = missing_symbols
    metadata["us_features"] = us_features
    return query, metadata


def capture_pre_market(day: str | None = None, now: datetime | None = None,
                       snapshot_dir: Path = SNAPSHOTS, history_getter=None,
                       previous_tw_date: str | None = None,
                       daily: pd.DataFrame | None = None,
                       stocks: pd.DataFrame | None = None) -> dict:
    now = now or datetime.now(TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=TZ)
    now = now.astimezone(TZ)
    day = day or now.date().isoformat()
    if day != now.date().isoformat() or (now.hour, now.minute) >= (9, 0):
        raise RuntimeError("相似行情快照只能在當日台股09:00前建立")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / f"{day}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    prior = previous_tw_date or _previous_tw_date(day)
    prior_stamp = pd.Timestamp(prior)
    if prior_stamp.weekday() >= 5 or prior >= day:
        raise ValueError(f"前一台股交易日無效：{prior}")
    query, metadata = _live_us_query(day, prior, history_getter=history_getter)
    if daily is None or stocks is None:
        reference_daily, reference_stocks = load_reference_library(snapshot_dir)
        daily = daily if daily is not None else reference_daily
        stocks = stocks if stocks is not None else reference_stocks
    full_analog = all(feature in query for feature in FEATURES)
    aggregate = find_analogs(query, daily, stocks, before_date=day, k=8) if full_analog else None
    by_ticker = ({ticker: find_analogs(query, daily, stocks, before_date=day, ticker=ticker, k=8)
                  for ticker in cfg.ALL_STOCKS} if full_analog else {})
    payload = {"date": day, "captured_at": now.isoformat(),
               "status": "descriptive_reference" if full_analog else "core_indices_only",
               "cutoff": "08:30 Asia/Taipei", "metadata": metadata,
               "aggregate": aggregate, "by_ticker": by_ticker}
    with path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    if aggregate:
        (snapshot_dir / f"{day}.md").write_text(render_markdown(aggregate), encoding="utf-8")
    return payload


def settle_day(day: str | None = None, now: datetime | None = None,
               snapshot_dir: Path = SNAPSHOTS) -> dict:
    """收盤後把十檔 Open/Close 結果另存；盤前快照保持原樣。"""
    now = now or datetime.now(TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=TZ)
    now = now.astimezone(TZ)
    day = day or now.date().isoformat()
    if day != now.date().isoformat() or (now.hour, now.minute) < (13, 40):
        raise RuntimeError("美股相似行情只能在當日收盤後結算")
    snapshot_path = snapshot_dir / f"{day}.json"
    if not snapshot_path.exists():
        return {"settled": 0, "reason": "no_pre_market_snapshot"}
    settlement_dir = snapshot_dir / "settlements"
    settlement_dir.mkdir(parents=True, exist_ok=True)
    path = settlement_dir / f"{day}.json"
    if path.exists():
        return {"settled": len(json.loads(path.read_text(encoding="utf-8"))["stocks"]),
                "reason": "already_settled", "path": str(path)}
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not snapshot.get("aggregate"):
        return {"settled": 0, "reason": "analog_unavailable_missing_peer_quotes"}
    prior = snapshot["metadata"]["previous_tw_date"]
    results = {}
    for ticker in cfg.ALL_STOCKS:
        csv_path = Path(cfg.RAW_DATA_DIR) / f"{ticker}.csv"
        if not csv_path.exists():
            return {"settled": 0, "reason": f"missing_stock_{ticker}"}
        df = pd.read_csv(csv_path, usecols=["Date", "Open", "Close"])
        today_row = df[df["Date"] == day]
        previous_row = df[df["Date"] == prior]
        if len(today_row) != 1 or len(previous_row) != 1:
            return {"settled": 0, "reason": f"incomplete_stock_{ticker}"}
        open_price = float(today_row.iloc[0]["Open"])
        close_price = float(today_row.iloc[0]["Close"])
        previous_close = float(previous_row.iloc[0]["Close"])
        if not all(math.isfinite(value) and value > 0
                   for value in (open_price, close_price, previous_close)):
            return {"settled": 0, "reason": f"invalid_stock_{ticker}"}
        results[ticker] = {
            "gap": 100 * (open_price / previous_close - 1),
            "intraday": 100 * (close_price / open_price - 1),
            "close_to_close": 100 * (close_price / previous_close - 1),
        }
    payload = {"date": day, "settled_at": now.isoformat(), "stocks": results}
    with path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return {"settled": len(results), "path": str(path)}


def prompt_context(snapshot: dict) -> str:
    aggregate = snapshot["aggregate"]
    metadata = snapshot["metadata"]
    if not aggregate:
        features = metadata["us_features"]
        return (f"【美股核心指數｜美股 {metadata['us_session']} → 台股 {snapshot['date']}】\n"
                f"費半 {features['sox_last']:+.2f}%、那指 {features['nasdaq_last']:+.2f}%、"
                f"標普 {features['sp500_last']:+.2f}%。個股同業報價延遲，本日不用五維相似案例。")
    lines = [f"【美股歷史相似行情｜美股 {metadata['us_session']} → 台股 {snapshot['date']}】",
             "以下只描述過去8個相似日，不是經驗證的預測；不得據此自動翻轉方向或提高倉位。"]
    for label, name in [("gap", "開盤跳空"), ("intraday", "開盤至收盤"),
                        ("close_to_close", "前收至今收")]:
        item = aggregate["summary"][label]
        lines.append(f"十檔等權{name}：歷史均值 {item['mean_pct']:+.2f}%、上漲比例 {item['up_share']:.0%}")
    lines.append("個股盤中歷史均值（只供情境比較）：")
    for ticker, result in snapshot["by_ticker"].items():
        item = result["summary"]["intraday"]
        lines.append(f"{ticker}: {item['mean_pct']:+.2f}%，上漲比例 {item['up_share']:.0%}")
    return "\n".join(lines)
