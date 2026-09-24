"""
新聞關鍵字學習：記錄「某股某日新聞關鍵字 → 當日漲跌」，累積出每個關鍵字的隔日勝率。
不需歷史 533 天，從現有資料 bootstrap、之後每日盤後持續累積。完全可解釋。

核心：keyword_impact.json 累積 {關鍵字: {count, up_count, 出現股票}}
      up_rate = up_count / count → 越高越偏多、越低越偏空。

用法：
  python news_keyword_learner.py --bootstrap   # 用現有新聞 JSON 初始化
  python news_keyword_learner.py               # 印關鍵字影響報告
  （盤後流程會自動呼叫 record_day 累積）
"""
import os
import sys
import re
import json
import glob
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

STORE = os.path.join(cfg.WIKI_DIR, "keyword_impact.json")
NEWS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "news")

# 台股常見關鍵字（不預先標多空，由資料學）
KEYWORDS = [
    "漲價", "漲停", "跌停", "訂單", "滿單", "缺貨", "急殺", "重挫", "創高", "創新高",
    "財報", "季報", "營收", "年增", "月增", "法說", "目標價", "看好", "看壞", "利多",
    "利空", "外資", "投信", "買超", "賣超", "AI", "MLCC", "受惠", "題材", "回檔",
    "回溫", "強勢", "飆", "套牢", "配息", "併購", "收購", "擴產", "供不應求", "拉貨",
    "庫存", "報價", "出貨", "車用", "高階", "需求",
]


def _load() -> dict:
    if os.path.exists(STORE):
        with open(STORE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(d: dict):
    os.makedirs(cfg.WIKI_DIR, exist_ok=True)
    with open(STORE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def extract_keywords(text: str) -> set:
    """從新聞文字找出出現的關鍵字。"""
    return {k for k in KEYWORDS if k in text}


def _direction_on(ticker: str, date_str: str):
    """回傳該股在 date_str 當日的漲跌方向（+1漲/-1跌），無資料回 None。"""
    import pandas as pd
    path = os.path.join(cfg.RAW_DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = df.index.map(lambda x: str(x.date()))
    if date_str not in df.index:
        return None
    i = df.index.get_loc(date_str)
    if isinstance(i, slice) or i == 0:
        return None
    today = df["Close"].iloc[i]
    prev = df["Close"].iloc[i - 1]
    return 1 if today > prev else -1


def record_day(store: dict, code: str, date_str: str, news_text: str) -> int:
    """記錄某股某日：新聞關鍵字 → 當日漲跌。回傳記錄的關鍵字數。
    冪等：同一 (日期,股票) 只計一次，避免重複累積。"""
    seen = store.setdefault("_processed", [])
    key = f"{date_str}|{code}"
    if key in seen:
        return 0
    full = next((k for k in cfg.ALL_STOCKS if k.startswith(code + ".")), None)
    if not full:
        return 0
    direction = _direction_on(full, date_str)
    if direction is None:
        return 0
    kws = extract_keywords(news_text)
    seen.append(key)
    for kw in kws:
        e = store.setdefault(kw, {"count": 0, "up": 0, "stocks": []})
        e["count"] += 1
        if direction > 0:
            e["up"] += 1
        if code not in e["stocks"]:
            e["stocks"].append(code)
    return len(kws)


def bootstrap() -> dict:
    """用現有盤前新聞 JSON 初始化關鍵字影響表。"""
    store = _load()
    n_days = 0
    for jf in sorted(glob.glob(os.path.join(NEWS_DIR, "*_pre_market.json"))):
        date_str = os.path.basename(jf)[:10]
        try:
            data = json.load(open(jf, encoding="utf-8"))
        except Exception:
            continue
        any_rec = False
        for code, info in data.items():
            if code == "market" or not isinstance(info, dict):
                continue
            # 組合該股新聞文字：催化劑 + 理由
            parts = info.get("key_catalysts", []) or info.get("catalysts", [])
            text = " ".join(parts) + " " + str(info.get("strategy_reason", "")) + " " + str(info.get("reason", ""))
            if record_day(store, code, date_str, text):
                any_rec = True
        if any_rec:
            n_days += 1
    _save(store)
    print(f"已從 {n_days} 個交易日的新聞 bootstrap")
    return store


def record_today(date_str: str | None = None) -> int:
    """盤後呼叫：把今日盤前新聞的關鍵字 + 今日漲跌記進累積表（冪等）。"""
    date_str = date_str or date.today().isoformat()
    jf = os.path.join(NEWS_DIR, f"{date_str}_pre_market.json")
    if not os.path.exists(jf):
        return 0
    store = _load()
    try:
        data = json.load(open(jf, encoding="utf-8"))
    except Exception:
        return 0
    n = 0
    for code, info in data.items():
        if code == "market" or not isinstance(info, dict):
            continue
        parts = info.get("key_catalysts", []) or info.get("catalysts", [])
        text = " ".join(parts) + " " + str(info.get("strategy_reason", "")) + " " + str(info.get("reason", ""))
        if record_day(store, code, date_str, text):
            n += 1
    _save(store)
    return n


def report(min_count: int = 3) -> str:
    """產出關鍵字影響報告（依隔日上漲率排序）。"""
    store = _load()
    rows = []
    for kw, e in store.items():
        if kw.startswith("_") or not isinstance(e, dict):
            continue
        if e["count"] < min_count:
            continue
        rate = e["up"] / e["count"] * 100
        rows.append((kw, e["count"], rate, len(e["stocks"])))
    rows.sort(key=lambda x: -x[2])

    lines = [f"# 新聞關鍵字影響報告　{date.today().isoformat()}\n",
             f"（樣本數 ≥ {min_count} 才列入；上漲率=該關鍵字出現後當日上漲的比例）\n",
             "| 關鍵字 | 出現次數 | 當日上漲率 | 傾向 |",
             "|--------|---------|-----------|------|"]
    for kw, cnt, rate, nstk in rows:
        bias = "🔴 偏多" if rate >= 60 else ("🔵 偏空" if rate <= 40 else "⚪ 中性")
        lines.append(f"| {kw} | {cnt} | {rate:.0f}% | {bias} |")
    if not rows:
        lines.append("| （樣本不足，需累積更多天）| | | |")
    return "\n".join(lines)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--min", type=int, default=3)
    p.add_argument("--save", action="store_true")
    args = p.parse_args()

    if args.bootstrap:
        bootstrap()
    rep = report(min_count=args.min)
    print("\n" + rep)
    if args.save:
        path = os.path.join(cfg.WIKI_DIR, "keyword_impact_report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(rep)
        print(f"\n已存：{path}")


if __name__ == "__main__":
    main()
