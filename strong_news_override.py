"""強利多方向修正訊號 —— ⚠️ 初步訊號（Fable 稽核 F2 降級）。

證據現況（誠實標示）：
  • in-sample 73~77%，但 out-of-sample 僅 15 樣本 11 勝：p=0.059 vs 銅板、
    p=0.120 vs 基準55% → 統計上不顯著。
  • 回測對齊（日T新聞→T+1方向）≠ live 對齊（早上LLM摘要→當日方向），
    且評分文字常含「股價勁揚」等昨日走勢描述 → 訊號疑似動能回聲。
  → 結論：值得用但別當鐵律。每次觸發記 log（strong_news_log.json），
    累積 30+ 筆 live 樣本後跑 report() 驗收，不顯著就退場。

用法：
  python strong_news_override.py        # 印 live 觸發驗收報告
"""
import os, sys, json
from datetime import date, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STRONG_THRESHOLD = 0.20     # 門檻（全樣本選點，屬 in-sample——驗收時一併檢視）
HIST_UP_RATE = 73           # 歷史隔日上漲率（初步值，見上方證據現況）

def _log_path():
    import config as cfg
    return os.path.join(cfg.WIKI_DIR, "strong_news_log.json")


def _log_trigger(code: str, date_str: str, score: float, context: str):
    """記錄一次觸發（冪等：同 日期|股票 只留最新）。供 live 驗收累積樣本。"""
    try:
        p = _log_path()
        log = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        log[f"{date_str}|{code}"] = {"score": round(score, 3), "context": context,
                                     "ts": datetime.now().strftime("%m-%d %H:%M")}
        json.dump(log, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    except Exception:
        pass


def strong_score(code: str, date_str: str = None) -> float:
    """回傳該股今日新聞偏多分數（無則 0）。"""
    try:
        from news_aux_judge import today_score
        return today_score(code, date_str)
    except Exception:
        return 0.0


def is_strong_bullish(code: str, date_str: str = None) -> bool:
    return strong_score(code, date_str) >= STRONG_THRESHOLD


def adjust_direction(code: str, predicted_pct: float, date_str: str = None):
    """強利多且模型猜跌/持平 → 方向翻多。回傳 (調整後pct, 是否觸發, 說明)。
    調整後給溫和正值（強利多日歷史中位約 +1.7%），只翻方向不誇大幅度。
    每次觸發自動記 log 供驗收。"""
    sc = strong_score(code, date_str)
    if sc >= STRONG_THRESHOLD and predicted_pct <= 0.5:
        d = date_str or date.today().isoformat()
        _log_trigger(code, d, sc, f"模型{predicted_pct:+.2f}%→翻多+1.5%")
        return 1.5, True, f"🔥強利多({sc:.2f}) 歷史{HIST_UP_RATE}%漲(初步)"
    return predicted_pct, False, ""


def strong_list(date_str: str = None) -> list:
    """回傳今日所有強利多個股 [(code, name, score), ...]。"""
    import config as cfg
    out = []
    for tk in cfg.ALL_STOCKS:
        code = tk.replace(".TWO", "").replace(".TW", "")
        sc = strong_score(code, date_str)
        if sc >= STRONG_THRESHOLD:
            out.append((code, cfg.ALL_STOCKS.get(tk, ""), sc))
    return sorted(out, key=lambda x: -x[2])


def report() -> str:
    """live 觸發驗收：對照每筆觸發的當日實際漲跌（與 live 用法同一對齊）。"""
    import config as cfg
    import pandas as pd
    p = _log_path()
    if not os.path.exists(p):
        return "尚無觸發紀錄"
    log = json.load(open(p, encoding="utf-8"))
    rows, up = [], 0
    for key, v in sorted(log.items()):
        d, code = key.split("|")
        full = next((k for k in cfg.ALL_STOCKS if k.startswith(code + ".")), None)
        if not full:
            continue
        df = pd.read_csv(os.path.join(cfg.RAW_DATA_DIR, f"{full}.csv"),
                         index_col=0, parse_dates=True)
        idx = [str(x.date()) for x in df.index]
        if d not in idx:
            rows.append(f"{d} {code} score={v['score']} → 待收盤"); continue
        i = idx.index(d)
        if i == 0:
            continue
        won = df["Close"].iloc[i] > df["Close"].iloc[i - 1]
        up += won
        rows.append(f"{d} {code} score={v['score']} → {'✅漲' if won else '❌跌'}")
    n = sum(1 for r in rows if "→ ✅" in r or "→ ❌" in r)
    head = (f"強利多 live 驗收：{n} 筆已結算，上漲 {up} 筆"
            f"（{up/n*100:.0f}%）｜目標樣本 30+ 筆再下結論\n" if n else "尚無已結算樣本\n")
    return head + "\n".join(rows)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(report())
