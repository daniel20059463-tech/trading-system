"""
籌碼面資料抓取：三大法人買賣超（外資/投信/自營），台股大型股最強的預測訊號之一。

資料來源：FinMind 開放 API（免 token，dataset=TaiwanStockInstitutionalInvestorsBuySell）。
法人類別：
  Foreign_Investor      外資（最具影響力，台股大型股主推手）
  Foreign_Dealer_Self   外資自營
  Investment_Trust      投信（國內基金，常有作帳/動能行為）
  Dealer_self           自營商自行買賣
  Dealer_Hedging        自營商避險

特徵設計（全部正規化，避免不同股票量級差異 + 維持穩態）：
  foreign_net_ratio  外資當日淨買 / 當日成交量   （|值|<=1，買超為正）
  trust_net_ratio    投信當日淨買 / 當日成交量
  dealer_net_ratio   自營當日淨買 / 當日成交量
  foreign_net_ma5    外資近5日累積淨買 / 近5日累積量（捕捉連續買超動能）

⚠️ 法人資料約於每日盤後 15:00 後更新；當日盤前預測時用的是「昨日」法人，已足夠領先。
"""
import sys
import os
import time
import logging

import requests
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"
DATASET = "TaiwanStockInstitutionalInvestorsBuySell"

FOREIGN_TYPES = {"Foreign_Investor", "Foreign_Dealer_Self"}
TRUST_TYPES   = {"Investment_Trust"}
DEALER_TYPES  = {"Dealer_self", "Dealer_Hedging"}


def _fetch_raw(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """抓取單支股票的法人買賣原始資料（含限速重試）。失敗回傳空 DataFrame。

    FinMind 免 token 會限速，批次抓多支時部分會回空或非 200；
    對「非 200 或空資料」都重試並指數退避，因 10 支都是流動性高、必有資料的股票。
    """
    backoff = 6
    for attempt in range(5):
        try:
            r = requests.get(FINMIND_URL, params={
                "dataset":    DATASET,
                "data_id":    stock_code,
                "start_date": start_date,
                "end_date":   end_date,
            }, timeout=30)
            j = r.json()
            if j.get("status") == 200 and j.get("data"):
                return pd.DataFrame(j["data"])
            # 非200或空資料 → 多半是限速，退避重試
            logger.info(f"FinMind {stock_code} 第{attempt+1}次空/異常(status={j.get('status')})，{backoff}s後重試")
        except Exception as e:
            logger.info(f"FinMind {stock_code} 第{attempt+1}次例外：{e}，{backoff}s後重試")
        time.sleep(backoff)
        backoff = min(backoff * 2, 40)
    logger.warning(f"FinMind {stock_code} 重試耗盡，回傳空")
    return pd.DataFrame()


def get_chip_daily_net(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """回傳以日期(date)為索引的法人每日淨買張數 DataFrame：
    欄位 foreign_net / trust_net / dealer_net（單位：股）。"""
    raw = _fetch_raw(stock_code, start_date, end_date)
    if raw.empty:
        return pd.DataFrame()

    raw["net"] = raw["buy"] - raw["sell"]

    def _net_of(types):
        sub = raw[raw["name"].isin(types)]
        return sub.groupby("date")["net"].sum()

    out = pd.DataFrame({
        "foreign_net": _net_of(FOREIGN_TYPES),
        "trust_net":   _net_of(TRUST_TYPES),
        "dealer_net":  _net_of(DEALER_TYPES),
    }).fillna(0.0)
    out.index = pd.to_datetime(out.index)
    return out


def add_chip_features(df: pd.DataFrame, ticker: str,
                      start_date: str, end_date: str) -> pd.DataFrame:
    """在個股 DataFrame（含 Volume）上附加正規化籌碼面特徵。

    對齊：法人資料以日期 mapping 到 df 交易日，缺值補 0（視為當日無法人進出）。
    正規化：淨買股數 / 當日成交量，clip 到 [-1, 1]。
    """
    stock_code = ticker.replace(".TWO", "").replace(".TW", "")
    chip_cols = ["foreign_net_ratio", "trust_net_ratio", "dealer_net_ratio", "foreign_net_ma5"]

    chips = get_chip_daily_net(stock_code, start_date, end_date)
    if chips.empty:
        logger.warning(f"{ticker} 無法人資料，籌碼特徵補 0")
        for c in chip_cols:
            df[c] = 0.0
        return df

    # 以日期(date)為鍵 mapping，避開時區問題
    fmap = {d.date(): v for d, v in chips["foreign_net"].items()}
    tmap = {d.date(): v for d, v in chips["trust_net"].items()}
    dmap = {d.date(): v for d, v in chips["dealer_net"].items()}

    idx_dates = [d.date() for d in df.index]
    vol = df["Volume"].astype(float)
    vol = vol.where(vol != 0, np.nan)   # 零量日設 NaN，避免除以零（用 np.nan 不用 pd.NA）

    foreign = pd.Series([fmap.get(d, 0.0) for d in idx_dates], index=df.index, dtype=float)
    trust   = pd.Series([tmap.get(d, 0.0) for d in idx_dates], index=df.index, dtype=float)
    dealer  = pd.Series([dmap.get(d, 0.0) for d in idx_dates], index=df.index, dtype=float)

    df["foreign_net_ratio"] = (foreign / vol).fillna(0.0).clip(-1, 1)
    df["trust_net_ratio"]   = (trust   / vol).fillna(0.0).clip(-1, 1)
    df["dealer_net_ratio"]  = (dealer  / vol).fillna(0.0).clip(-1, 1)

    # 外資近5日累積淨買 / 近5日累積量（連續買超動能）
    foreign_ma5 = foreign.rolling(5, min_periods=1).sum()
    vol_ma5     = df["Volume"].astype(float).rolling(5, min_periods=1).sum()
    vol_ma5     = vol_ma5.where(vol_ma5 != 0, np.nan)
    df["foreign_net_ma5"] = (foreign_ma5 / vol_ma5).fillna(0.0).clip(-1, 1)

    return df


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO)
    # 自我測試：抓國巨近期法人並印出
    net = get_chip_daily_net("2327", "2026-06-01", "2026-06-11")
    print("國巨 2327 法人每日淨買（張）：")
    print((net / 1000).round(0).tail(10))
