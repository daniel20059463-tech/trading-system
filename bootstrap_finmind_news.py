"""
用 FinMind 歷史新聞 bootstrap 關鍵字學習表：逐日抓新聞 → 隔日漲跌 → 累積關鍵字勝率。
FinMind TaiwanStockNews 一次只給一天（不可帶 end_date），故逐(股,日)查詢。
增量存檔 + 冪等，可中斷後重跑續抓。

用法：
  python bootstrap_finmind_news.py --test          # 小測：國巨近1月
  python bootstrap_finmind_news.py --months 6      # 正式：近6個月全部股票
"""
import os, sys, time, json, argparse
from datetime import date

import requests
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from news_keyword_learner import KEYWORDS, extract_keywords, STORE, _load, _save

URL = "https://api.finmindtrade.com/api/v4/data"

def _load_tokens() -> list:
    """手動讀 .env 全部 FINMIND_TOKEN（同名多行 dotenv 會蓋掉，故自己解析）。"""
    toks = []
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if line.startswith("FINMIND_TOKEN") and "=" in line:
                v = line.split("=", 1)[1].strip()
                if len(v) > 50:
                    toks.append(v)
    return toks


TOKENS = _load_tokens()
DELAY = 0.05 if TOKENS else 0.25      # 有 token 可加快
_TI = [0]                             # 目前用第幾個 token（額度爆就輪下一個）


def _fetch_news(stock_id: str, d: str):
    """抓單股單日新聞標題。多 token 輪替：某 token 被限速就換下一個。"""
    params = {"dataset": "TaiwanStockNews", "data_id": stock_id, "start_date": d}
    backoff = 5
    tries = max(len(TOKENS) * 2, 4)
    for attempt in range(tries):
        if TOKENS:
            params["token"] = TOKENS[_TI[0] % len(TOKENS)]
        try:
            r = requests.get(URL, params=params, timeout=20)
            j = r.json()
            if j.get("status") == 200:
                return " ".join(str(a.get("title", "")) for a in j.get("data", []))
            _TI[0] += 1                # 限速 → 換 token
            if TOKENS and (attempt + 1) % len(TOKENS) == 0:
                time.sleep(backoff); backoff = min(backoff * 2, 30)
        except Exception:
            _TI[0] += 1
            time.sleep(1)
    return None


def _next_day_dir(df, dates, i):
    """新聞日 i → 隔一交易日漲跌（+1/-1），無則 None。"""
    if i + 1 >= len(dates):
        return None
    return 1 if df["Close"].iloc[i + 1] > df["Close"].iloc[i] else -1


def run(months: int, test: bool = False):
    store = _load()
    seen = store.setdefault("_processed", [])
    tickers = ["2327.TW"] if test else list(cfg.ALL_STOCKS.keys())

    n_recorded = n_withnews = n_req = 0
    for tk in tickers:
        code = tk.replace(".TWO", "").replace(".TW", "")
        df = pd.read_csv(os.path.join(cfg.RAW_DATA_DIR, f"{tk}.csv"), index_col=0, parse_dates=True)
        dates = [str(x.date()) for x in df.index]
        # 取近 months 個月、但排除 6/02 之後（那段用盤前 JSON 較完整）
        cutoff_lo = (pd.Timestamp.today() - pd.DateOffset(months=months)).strftime("%Y-%m-%d")
        idxs = [i for i, d in enumerate(dates) if cutoff_lo <= d < "2026-06-02"]
        if test:
            idxs = idxs[-22:]   # 近一個月

        for i in idxs:
            d = dates[i]
            key = f"fm|{d}|{code}"
            if key in seen:
                continue
            text = _fetch_news(code, d); n_req += 1
            seen.append(key)
            if text and any(k in text for k in KEYWORDS):
                direction = _next_day_dir(df, dates, i)
                if direction is not None:
                    n_withnews += 1
                    for kw in extract_keywords(text):
                        e = store.setdefault(kw, {"count": 0, "up": 0, "stocks": []})
                        e["count"] += 1
                        if direction > 0:
                            e["up"] += 1
                        if code not in e["stocks"]:
                            e["stocks"].append(code)
                    n_recorded += 1
            time.sleep(DELAY)
            if n_req % 50 == 0:
                _save(store)
                print(f"  進度：{tk} 已查 {n_req} 天，有效新聞日 {n_withnews}")
    _save(store)
    print(f"\n完成：查詢 {n_req} 天，有關鍵字的新聞日 {n_withnews}，已累積進關鍵字表")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--test", action="store_true")
    p.add_argument("--months", type=int, default=6)
    a = p.parse_args()
    run(a.months, a.test)
