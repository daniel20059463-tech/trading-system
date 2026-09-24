"""
新聞情緒分析模組：使用 OpenAI API 分析台股新聞並產出策略訊號。
"""

import sys
import os
import json
import logging
from datetime import datetime

from openai import OpenAI, OpenAIError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SECTOR_INFO = {
    "passive": ["2327", "2492", "3026", "2472", "3357"],
    "power":   ["5425", "2481", "3675", "8255", "8261"],
}

DEFAULT_NEUTRAL = {
    "sentiment": "neutral",
    "sentiment_score": 0.0,
    "key_catalysts": [],
    "risk_factors": ["無法取得 AI 分析，請人工確認"],
    "strategy_signal": "hold",
    "strategy_reason": "API 分析失敗，維持觀望",
    "confidence": 0.0,
}

SYSTEM_PROMPT = (
    "你是一位專業的台灣股市分析師，專精於被動元件（MLCC、電阻、電容）與功率元件（MOSFET、整流器、二極體）產業。"
    "你熟悉台灣電子供應鏈的上下游關係，了解國際科技大廠（Apple、NVIDIA、Tesla 等）的零組件採購對台灣元件廠的影響。"
    "你擅長從新聞中萃取對股價有實質影響的關鍵催化劑，並區分系統性訊號與短期雜訊。"
    "分析時請保持客觀，避免過度樂觀或悲觀。"
    "所有輸出必須以嚴格的 JSON 格式回應，不得包含 Markdown 程式碼區塊或額外說明文字。"
)


def _call_openai(user_prompt: str, max_tokens: int = 1024) -> str:
    """呼叫 OpenAI Chat Completions API，回傳原始文字。"""
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def _parse_json(raw: str, fallback: dict) -> dict:
    """清理 markdown 包裝後解析 JSON，失敗時回傳 fallback。"""
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失敗：{e}\n原始內容：{raw[:200]}")
        return fallback.copy()


def analyze_news_batch(
    news_list: list[dict],
    ticker: str,
    ticker_name: str,
    session_type: str = "pre_market",
) -> dict:
    """分析單一股票的新聞批次，回傳情緒與策略訊號 JSON。"""
    if not news_list:
        logger.warning(f"{ticker} 無新聞可分析，回傳預設中立值")
        return DEFAULT_NEUTRAL.copy()

    if ticker in SECTOR_INFO["passive"]:
        sector = "被動元件（MLCC / 電阻 / 電容）"
    elif ticker in SECTOR_INFO["power"]:
        sector = "功率元件（MOSFET / 整流器 / 二極體）"
    else:
        sector = "電子元件"

    session_label = {
        "pre_market":  "盤前（今日開盤前分析）",
        "intraday":    "盤中（即時更新）",
        "post_market": "盤後（收盤總結）",
    }.get(session_type, session_type)

    news_lines = []
    for i, article in enumerate(news_list[:20], 1):
        ts  = f"[{article['timestamp']}] " if article.get("timestamp") else ""
        src = f"({article.get('source', '?')})"
        news_lines.append(f"{i}. {ts}{article['title']} {src}")

    user_prompt = f"""
分析時段：{session_label}
股票代碼：{ticker}.TW
公司名稱：{ticker_name}
產業分類：{sector}
分析時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}

以下是最新相關新聞（共 {len(news_list)} 篇）：

{chr(10).join(news_lines)}

請根據上述新聞，輸出以下 JSON 格式的分析結果（不要加 markdown 包裝）：
{{
  "sentiment": "bullish 或 bearish 或 neutral",
  "sentiment_score": -1.0 到 1.0 的浮點數（負值看空，正值看多）,
  "key_catalysts": ["最重要的新聞催化劑（最多3條）"],
  "risk_factors": ["主要風險因子（最多3條）"],
  "strategy_signal": "buy 或 sell 或 hold 或 watch",
  "strategy_reason": "中文說明，結合產業背景，60字以內",
  "confidence": 0.0 到 1.0 的浮點數（對此分析的信心程度）
}}
"""

    try:
        raw = _call_openai(user_prompt, max_tokens=1024)
        result = _parse_json(raw, DEFAULT_NEUTRAL)

        for key in DEFAULT_NEUTRAL:
            if key not in result:
                result[key] = DEFAULT_NEUTRAL[key]

        logger.info(
            f"{ticker} {session_type} 分析完成：{result['sentiment']} "
            f"(score={result['sentiment_score']:.2f}, signal={result['strategy_signal']})"
        )
        return result

    except OpenAIError as e:
        logger.error(f"{ticker} OpenAI API 錯誤：{e}")
        return DEFAULT_NEUTRAL.copy()
    except Exception as e:
        logger.error(f"{ticker} 分析時發生未預期錯誤：{e}")
        return DEFAULT_NEUTRAL.copy()


def analyze_market_sentiment(market_news_list: list[dict]) -> dict:
    """分析大盤新聞的整體情緒，回傳市場情緒摘要。"""
    if not market_news_list:
        return {"market_sentiment": "neutral", "score": 0.0, "summary": "無大盤新聞可供分析"}

    news_lines = [
        f"{i}. {('[' + a['timestamp'] + '] ') if a.get('timestamp') else ''}{a['title']}"
        for i, a in enumerate(market_news_list[:15], 1)
    ]

    user_prompt = f"""
分析時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}
以下是台股大盤相關新聞（共 {len(market_news_list)} 篇）：

{chr(10).join(news_lines)}

請輸出以下 JSON 格式（不要加 markdown 包裝）：
{{
  "market_sentiment": "bullish 或 bearish 或 neutral",
  "score": -1.0 到 1.0 的浮點數,
  "summary": "中文摘要，說明當前市場主要關注焦點，80字以內",
  "key_themes": ["市場主題1", "市場主題2"],
  "macro_risk": "高、中、低"
}}
"""

    fallback = {"market_sentiment": "neutral", "score": 0.0, "summary": "分析失敗"}
    try:
        raw = _call_openai(user_prompt, max_tokens=512)
        result = _parse_json(raw, fallback)
        logger.info(
            f"大盤情緒分析：{result.get('market_sentiment', 'N/A')} "
            f"(score={result.get('score', 0):.2f})"
        )
        return result
    except OpenAIError as e:
        logger.error(f"大盤新聞 OpenAI API 錯誤：{e}")
        return fallback
    except Exception as e:
        logger.error(f"大盤新聞分析時發生未預期錯誤：{e}")
        return fallback


if __name__ == "__main__":
    print("=== 測試新聞分析模組 ===")

    sample_news = [
        {"title": "國巨Q2營收創歷史新高，MLCC需求回溫超預期", "url": "", "timestamp": "2026-06-01", "source": "yahoo"},
        {"title": "被動元件庫存去化完成，法人預期下半年景氣復甦",  "url": "", "timestamp": "2026-06-01", "source": "moneydj"},
        {"title": "美中科技戰升溫，電子元件出口管制影響評估",      "url": "", "timestamp": "2026-06-02", "source": "yahoo"},
    ]

    print("\n分析 2327（國巨）盤前新聞...")
    result = analyze_news_batch(sample_news, "2327", "國巨", "pre_market")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n分析大盤情緒...")
    market_result = analyze_market_sentiment([
        {"title": "台股加權指數收漲150點，外資連續買超", "url": "", "timestamp": "2026-06-02", "source": "yahoo_market"},
        {"title": "Fed 維持利率不變，科技股全面上漲",    "url": "", "timestamp": "2026-06-02", "source": "yahoo_market"},
    ])
    print(json.dumps(market_result, ensure_ascii=False, indent=2))

    print("\n=== 測試完成 ===")
