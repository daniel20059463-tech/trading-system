"""
建立 ML 用的「每日新聞特徵」（防資料洩漏版）。

流程：
  1. 逐 (股,日) 抓 FinMind 新聞標題 → 存每日關鍵字到 news_daily.json（快取，可續抓）
     （重用 bootstrap_finmind_news 的多 token 輪替 _fetch_news）
  2. 只用「訓練期」資料算每個關鍵字的 up-rate 偏離基準 → 權重（防洩漏）
  3. 每日 news_score = 當天出現關鍵字的權重總和 → 輸出 news_feature.csv

用法：
  python build_news_feature.py --fetch            # 抓+快取每日關鍵字（明天額度重置後跑）
  python build_news_feature.py --build --split 0.8  # 用快取算防洩漏特徵 → news_feature.csv
"""
import os, sys, json, argparse
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
import pandas as pd

from bootstrap_finmind_news import _fetch_news, TOKENS, DELAY
from news_keyword_learner import KEYWORDS, extract_keywords

CACHE = os.path.join(cfg.WIKI_DIR, "news_daily.json")   # {"date|code": ["關鍵字",...]}
FEATURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "news_feature.csv")


def _load_cache() -> dict:
    if os.path.exists(CACHE):
        return json.load(open(CACHE, encoding="utf-8"))
    return {}


def _save_cache(c: dict):
    os.makedirs(cfg.WIKI_DIR, exist_ok=True)
    json.dump(c, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)


def fetch(months: int = 24):
    """逐 (股,日) 抓新聞、抽關鍵字、存快取（可續抓，已抓的跳過）。"""
    import time
    cache = _load_cache()
    tickers = list(cfg.ALL_STOCKS.keys())
    lo = (pd.Timestamp.today() - pd.DateOffset(months=months)).strftime("%Y-%m-%d")
    n_new = 0
    for tk in tickers:
        code = tk.replace(".TWO", "").replace(".TW", "")
        df = pd.read_csv(os.path.join(cfg.RAW_DATA_DIR, f"{tk}.csv"), index_col=0, parse_dates=True)
        for ts in df.index:
            d = str(ts.date())
            if d < lo:
                continue
            ck = f"{d}|{code}"
            if ck in cache:           # 已抓過，跳過（續抓）
                continue
            text = _fetch_news(code, d)
            cache[ck] = sorted(extract_keywords(text)) if text else []
            n_new += 1
            time.sleep(DELAY)
            if n_new % 100 == 0:
                _save_cache(cache)
                print(f"  已抓 {n_new} 筆新（{tk} {d}）")
    _save_cache(cache)
    print(f"完成：新增 {n_new} 筆，快取共 {len(cache)} 筆 → {CACHE}")


def build(split: float = 0.8):
    """用快取建防洩漏特徵：只用訓練期算關鍵字權重，套到全期算 news_score。"""
    cache = _load_cache()
    if not cache:
        print("快取為空，請先 --fetch")
        return

    # 收集每股每日方向 + 關鍵字，依日期排序決定訓練/測試切點
    rows = []   # (date, code, keywords, next_dir)
    for tk in list(cfg.ALL_STOCKS.keys()):
        code = tk.replace(".TWO", "").replace(".TW", "")
        df = pd.read_csv(os.path.join(cfg.RAW_DATA_DIR, f"{tk}.csv"), index_col=0, parse_dates=True)
        closes = df["Close"].values
        dates = [str(x.date()) for x in df.index]
        for i in range(len(dates) - 1):
            ck = f"{dates[i]}|{code}"
            if ck not in cache:
                continue
            nxt = 1 if closes[i + 1] > closes[i] else -1
            rows.append((dates[i], code, cache[ck], nxt))

    rows.sort(key=lambda r: r[0])
    cut = int(len(rows) * split)
    train = rows[:cut]

    # 只用訓練期算每關鍵字 up-rate 與基準
    kw_up, kw_n = {}, {}
    tot_up = tot_n = 0
    for _, _, kws, nxt in train:
        for kw in kws:
            kw_n[kw] = kw_n.get(kw, 0) + 1
            kw_up[kw] = kw_up.get(kw, 0) + (1 if nxt > 0 else 0)
        tot_n += 1
        tot_up += (1 if nxt > 0 else 0)
    base = tot_up / tot_n if tot_n else 0.5
    # 權重 = 偏離基準（樣本<10 的關鍵字權重設 0，避免雜訊）
    weight = {kw: (kw_up[kw] / kw_n[kw] - base) if kw_n[kw] >= 10 else 0.0 for kw in kw_n}

    # 套到全期算每日 news_score
    out = []
    for d, code, kws, _ in rows:
        score = sum(weight.get(kw, 0.0) for kw in kws)
        out.append({"date": d, "code": code, "news_score": round(score, 4), "n_kw": len(kws)})
    fdf = pd.DataFrame(out)
    os.makedirs(os.path.dirname(FEATURE), exist_ok=True)
    fdf.to_csv(FEATURE, index=False, encoding="utf-8-sig")
    nz = (fdf["news_score"] != 0).mean() * 100
    print(f"基準 {base:.0%}｜訓練期樣本 {tot_n}｜特徵 {len(fdf)} 筆，非零 {nz:.0f}%")
    print(f"已輸出 → {FEATURE}")
    print("權重前 5 偏多:", sorted(weight.items(), key=lambda x: -x[1])[:5])
    print("權重前 5 偏空:", sorted(weight.items(), key=lambda x: x[1])[:5])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--fetch", action="store_true")
    p.add_argument("--build", action="store_true")
    p.add_argument("--months", type=int, default=24)
    p.add_argument("--split", type=float, default=0.8)
    a = p.parse_args()
    if a.fetch:
        print(f"Token 數：{len(TOKENS)}（明天額度重置後抓最快）")
        fetch(a.months)
    if a.build:
        build(a.split)
