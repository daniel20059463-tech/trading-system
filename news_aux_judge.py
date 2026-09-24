"""新聞關鍵字「輔助判斷」——擺在模型預測旁邊的獨立人看訊號，不混進 ML。

依據 2 年 FinMind 新聞學到的關鍵字隔日勝率（keyword_impact.json），
把今日某股新聞裡的關鍵字換算成「偏多/偏空」傾向，並透明列出是哪些字、各偏多偏空多少。

注意：這是「弱訊號、稀疏、落後」的參考用判斷（曾測過當 ML 特徵無提升），
僅供人對照，不取代模型，不誇大確定性。
"""
import os, sys, json
from datetime import date
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from news_keyword_learner import extract_keywords, _load

MIN_COUNT = 15        # 樣本太少的關鍵字不採用
POS, NEG = 0.10, -0.10  # 偏多/偏空門檻（分數=各關鍵字偏離基準之總和）


def _weights_and_base():
    """從 keyword_impact.json 算基準上漲率 + 每個關鍵字的偏離基準權重。"""
    s = _load()
    kws = {k: e for k, e in s.items() if not k.startswith("_") and isinstance(e, dict) and e.get("count")}
    common = [(k, e) for k, e in kws.items() if e["count"] >= 150]
    base = (sum(e["up"] for _, e in common) / sum(e["count"] for _, e in common)) if common else 0.5
    weight = {k: (e["up"] / e["count"] - base) for k, e in kws.items() if e["count"] >= MIN_COUNT}
    return weight, base


_W, _BASE = None, None


def judge(news_text: str) -> dict:
    """回傳 {lean, score, contrib:[(kw,dev)], n_kw}。dev 為該關鍵字隔日漲率偏離基準。"""
    global _W, _BASE
    if _W is None:
        _W, _BASE = _weights_and_base()
    kws = extract_keywords(news_text or "")
    contrib = sorted([(k, _W[k]) for k in kws if k in _W], key=lambda x: -abs(x[1]))
    score = sum(d for _, d in contrib)
    lean = "偏多" if score >= POS else ("偏空" if score <= NEG else "中性")
    return {"lean": lean, "score": score, "contrib": contrib, "n_kw": len(contrib), "base": _BASE}


def today_score(code: str, date_str: str = None) -> float:
    """今日某股的新聞偏多分數（給區間不對稱撐高用）。讀盤前新聞 JSON，無則回 0。"""
    date_str = date_str or date.today().isoformat()
    jf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "news", f"{date_str}_pre_market.json")
    if not os.path.exists(jf):
        return 0.0
    try:
        data = json.load(open(jf, encoding="utf-8"))
    except Exception:
        return 0.0
    info = data.get(code)
    if not isinstance(info, dict):
        return 0.0
    return judge(text_from_analysis(info)).get("score", 0.0)


def text_from_analysis(info: dict) -> str:
    """從新聞分析/盤前 JSON 的單股結構取出可判斷的文字。"""
    if not isinstance(info, dict):
        return ""
    parts = info.get("key_catalysts", []) or info.get("catalysts", []) or []
    return " ".join(map(str, parts)) + " " + str(info.get("strategy_reason", "")) + " " + str(info.get("reason", ""))


def format_line(code: str, name: str, r: dict) -> str:
    """一行人看的輔助判斷，含前幾個關鍵字與偏多偏空。"""
    icon = {"偏多": "🔴", "偏空": "🔵", "中性": "⚪"}[r["lean"]]
    if not r["contrib"]:
        return f"{icon} {code} {name}：中性（今日新聞無已學關鍵字）"
    tops = "、".join(f"{k}{d*100:+.0f}%" for k, d in r["contrib"][:3])
    return f"{icon} {code} {name}：{r['lean']}（{tops}）"


def judge_all(news_analysis: dict) -> list:
    """對所有股票產生輔助判斷行。回傳 [(code, line, result), ...]。"""
    out = []
    for code, info in (news_analysis or {}).items():
        if code == "market" or not isinstance(info, dict):
            continue
        full = next((k for k in cfg.ALL_STOCKS if k.startswith(code + ".")), None)
        name = cfg.ALL_STOCKS.get(full, "") if full else ""
        r = judge(text_from_analysis(info))
        out.append((code, format_line(code, name, r), r))
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    import json, glob
    # 用最近一份盤前新聞 JSON 示範
    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "news", "*_pre_market.json")))
    if not files:
        print("找不到盤前新聞 JSON，改用範例：")
        for t in ["國巨漲價訂單滿手，外資買超", "華新科財報營收年增", "立隆電法人賣超跌停"]:
            r = judge(t)
            print(f"  「{t}」→ {r['lean']}（分數{r['score']:+.2f}）{r['contrib'][:3]}")
    else:
        data = json.load(open(files[-1], encoding="utf-8"))
        print(f"=== 新聞關鍵字輔助判斷（{os.path.basename(files[-1])[:10]}）===")
        print(f"（基準上漲率 {judge('')['base']:.0%}；參考用弱訊號，不取代模型）\n")
        for code, line, r in judge_all(data):
            print(line)
