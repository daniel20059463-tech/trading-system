# fetch_stocks.py - 下載台灣股票資料並計算技術指標

import os
import sys
from datetime import datetime, time as _dtime
import numpy as np
import pandas as pd
import yfinance as yf

# 將專案根目錄加入 path，讓 config 可被匯入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ALL_STOCKS, START_DATE, END_DATE, RAW_DATA_DIR,
    PASSIVE_STOCKS, MACRO_INDICES, PASSIVE_PEERS, POWER_PEERS,
)

# 同業/宏觀指標的 pct_change 快取（避免每支股票重複下載）
_PEER_CACHE: dict = {}


def _get_peer_pct(ticker: str) -> "pd.Series":
    """下載並快取單一同業/宏觀指標的日漲跌幅（%），以日期為索引。"""
    if ticker in _PEER_CACHE:
        return _PEER_CACHE[ticker]
    try:
        raw = yf.Ticker(ticker).history(
            start=START_DATE, end=END_DATE, auto_adjust=True,
        )
        if raw.empty:
            _PEER_CACHE[ticker] = pd.Series(dtype=float)
        else:
            close = raw["Close"]
            close.index = close.index.tz_localize(None)   # 去除時區，便於對齊
            _PEER_CACHE[ticker] = close.pct_change() * 100
    except Exception as e:
        print(f"  [同業下載失敗] {ticker}: {e}")
        _PEER_CACHE[ticker] = pd.Series(dtype=float)
    return _PEER_CACHE[ticker]


def _sector_of(ticker: str) -> str:
    """判斷股票屬於被動還是功率元件。"""
    return "passive" if ticker in PASSIVE_STOCKS else "power"


def add_peer_features(df: pd.DataFrame, sector: str) -> pd.DataFrame:
    """將同業與宏觀指標的漲跌幅對齊到個股交易日，新增為特徵欄位。

    跨國交易日不同（日/美/台假日各異），用 reindex + 前向填補對齊；
    無法填補的開頭缺值補 0（視為當日無變動）。
    """
    peers = dict(MACRO_INDICES)
    peers.update(PASSIVE_PEERS if sector == "passive" else POWER_PEERS)

    for peer_ticker, prefix in peers.items():
        pct = _get_peer_pct(peer_ticker)
        col = f"{prefix}_pct"
        if pct.empty:
            df[col] = 0.0
            continue
        aligned = pct.reindex(df.index, method="ffill")
        df[col] = aligned.fillna(0.0).values
    return df


# ── 技術指標計算工具 ────────────────────────────────────────────────────────────

def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """用 EWM 方式計算 RSI 動量指標。"""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    alpha = 1 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calc_macd(close: pd.Series):
    """計算 MACD、Signal 與 Histogram。"""
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal
    return macd, signal, hist


def _calc_bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    """計算布林通道上軌、中軌、下軌。"""
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def _calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """用 EMA14 計算平均真實波幅（ATR）。"""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr


# ── 主要功能函式 ────────────────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """在 OHLCV DataFrame 上計算並附加所有技術指標欄位。"""
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # 移動平均（趨勢/動量）
    df["MA5"]  = close.rolling(5).mean()
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()

    # RSI（動量指標）
    df["RSI"] = _calc_rsi(close, period=14)

    # MACD（趨勢）
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = _calc_macd(close)

    # Bollinger Band（波動率/均值回歸）
    df["BB_upper"], df["BB_mid"], df["BB_lower"] = _calc_bollinger(close)

    # 量價關係
    df["Volume_MA20"]   = volume.rolling(20).mean()
    df["Volume_ratio"]  = volume / df["Volume_MA20"]

    # ATR（波動率）
    df["ATR"] = _calc_atr(high, low, close, period=14)

    # 日漲跌幅（%）
    df["pct_change"] = close.pct_change() * 100

    return df


def _sanitize_daily_rows(raw: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """只保留台股已完成的週一至週五日線。

    Yahoo 偶爾在週末後的週一盤前回傳標成週日的異常 bar。這種資料
    不能當成前一交易日收盤；盤前也不允許任何今日／未來 bar。
    """
    now = now or datetime.now()
    clean = raw.copy()
    dates = pd.DatetimeIndex(clean.index)
    clean = clean[dates.weekday < 5]
    if now.time() < _dtime(14, 0):
        clean_dates = pd.DatetimeIndex(clean.index)
        clean = clean[clean_dates.date < now.date()]
    return clean


def validate_pre_market_data(tickers=None, today=None) -> str:
    """盤前十檔必須共用同一個有效平日收盤日，否則停止發布。"""
    tickers = list(tickers or ALL_STOCKS)
    today = today or datetime.now().date()
    latest = {}
    for ticker in tickers:
        path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        if frame.empty:
            raise ValueError(f"{ticker} 無可用日線")
        stamp = pd.Timestamp(frame.index[-1]).date()
        close = float(frame["Close"].iloc[-1])
        if stamp.weekday() >= 5 or stamp >= today:
            raise ValueError(f"{ticker} 最新日線日期無效：{stamp}")
        if not np.isfinite(close) or close <= 0:
            raise ValueError(f"{ticker} 最新收盤價無效")
        latest[ticker] = stamp
    dates = set(latest.values())
    if len(dates) != 1:
        details = ", ".join(f"{ticker}={day}" for ticker, day in latest.items())
        raise ValueError(f"十檔股價資料截止日不一致：{details}")
    day = dates.pop()
    if (today - day).days > 7:
        raise ValueError(f"盤前股價資料過期：{day}")
    return day.isoformat()


def fetch_stock(ticker: str) -> pd.DataFrame:
    """從 yfinance 下載單支股票資料並計算技術指標，儲存為 CSV。"""
    print(f"[下載] {ticker} ({ALL_STOCKS.get(ticker, '')})")
    raw = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        print(f"  警告：{ticker} 無資料，跳過。")
        return pd.DataFrame()

    # yfinance 有時回傳 MultiIndex columns，展平為單層
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = _sanitize_daily_rows(raw)
    if raw.empty:
        raise ValueError(f"{ticker} 剔除週末／未完成日線後無資料")

    df = raw.copy()
    df = add_features(df)
    df = add_peer_features(df, _sector_of(ticker))   # 同業/宏觀指標漲跌幅
    # 註：籌碼面（_add_chips）2026-06 實驗無效已退回，故不再注入。fetch_chips.py 保留備用。
    df.dropna(inplace=True)

    save_path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
    df.to_csv(save_path)
    print(f"  已儲存 {len(df)} 筆 -> {save_path}")
    return df


def load_stock(ticker: str) -> pd.DataFrame:
    """從已存檔案讀取單支股票的特徵資料。"""
    path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 {path}，請先執行 fetch_stocks.py 下載資料。")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def get_all_stocks() -> dict:
    """回傳所有已下載股票的 DataFrame 字典（ticker -> DataFrame）。"""
    result = {}
    for ticker in ALL_STOCKS:
        try:
            result[ticker] = load_stock(ticker)
        except FileNotFoundError as e:
            print(f"  跳過 {ticker}：{e}")
    return result


def get_today_closing_prices() -> dict:
    """用 yfinance 抓取今日（或最近交易日）收盤價，回傳 {純代碼: price} 字典。

    盤後 15:30 後呼叫可取得當日真實收盤價；
    若當日尚未收盤則回傳最新成交價（intraday last price）。
    """
    from datetime import date, timedelta
    import logging
    logger = logging.getLogger(__name__)

    today = date.today().isoformat()
    start = (date.today() - timedelta(days=5)).isoformat()   # 往前5天保證抓到最近交易日

    result = {}
    for ticker, name in ALL_STOCKS.items():
        code = ticker.replace(".TWO", "").replace(".TW", "")
        try:
            hist = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
                result[code] = round(price, 2)
                logger.info(f"{ticker}（{name}）：{price:.2f}")
            else:
                logger.warning(f"{ticker} 無歷史資料")
        except Exception as e:
            logger.warning(f"{ticker} 收盤價抓取失敗：{e}")

    return result


# ── 主程式 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"開始下載 {len(ALL_STOCKS)} 支股票資料（{START_DATE} ~ {END_DATE}）\n")
    success, fail = 0, 0
    for ticker in ALL_STOCKS:
        try:
            df = fetch_stock(ticker)
            if not df.empty:
                success += 1
            else:
                fail += 1
        except Exception as exc:
            print(f"  錯誤 [{ticker}]: {exc}")
            fail += 1
    print(f"\n完成：成功 {success} 支，失敗 {fail} 支。")
