"""
策略 Agent：結合 ML 預測、新聞分析與歷史 Wiki 知識，使用 Reflexion+CLIN 框架產出操作建議。
"""

import sys
import os
import json
import logging
from datetime import datetime

from openai import OpenAI, OpenAIError

# 確保可以從專案根目錄執行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.wiki_agent import get_wiki_context_for_agent
from agents.rule_manager import (
    load_rules, save_rules, format_rules_for_prompt, apply_reflexion_updates
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 目標股票清單
TICKER_INFO = {
    "2327.TW":  {"name": "國巨",  "sector": "被動元件", "sub": "MLCC"},
    "2492.TW":  {"name": "華新科", "sector": "被動元件", "sub": "MLCC"},
    "3026.TW":  {"name": "禾伸堂", "sector": "被動元件", "sub": "電阻/電容"},
    "2472.TW":  {"name": "立隆電", "sector": "被動元件", "sub": "鋁質電容"},
    "3357.TWO": {"name": "臺慶科", "sector": "被動元件", "sub": "電容器"},
    "5425.TWO": {"name": "台半",  "sector": "功率元件", "sub": "整流器"},
    "2481.TW":  {"name": "強茂",  "sector": "功率元件", "sub": "二極體"},
    "3675.TWO": {"name": "德微",  "sector": "功率元件", "sub": "MOSFET"},
    "8255.TWO": {"name": "朋程",  "sector": "功率元件", "sub": "整流模組"},
    "8261.TW":  {"name": "富鼎",  "sector": "功率元件", "sub": "功率IC"},
}

STRATEGY_SYSTEM_PROMPT = """你是一位資深台灣股市量化交易策略師，專精被動元件與功率元件產業。

你的分析框架基於以下原則：
1. **奧卡姆剃刀約束**：不為單日隨機波動制定特殊規則。只有在多日一致的系統性訊號下才建議操作。
2. **CLIN 因果律**：優先套用已驗證的產業因果律（如「美國科技大廠財報超預期 → MLCC 需求回升」），避免過度擬合。
3. **Reflexion 反思**：結合歷史 Wiki 記錄，判斷今日訊號是否與過去規律一致。
4. **風險優先**：設定清晰的停損點，倉位大小與信心度掛鉤。

輸出格式為嚴格 JSON 陣列，不得包含 Markdown 或額外說明文字。
每支股票一個物件，共10個物件。"""


class StrategyAgent:
    """結合多源資訊產出最終策略建議的主策略 Agent，支援 Reflexion 自我反思機制。"""

    def __init__(self):
        """初始化 OpenAI client，載入 Wiki 知識與 CLIN 規則庫。"""
        self.client = OpenAI()
        self.wiki_context = get_wiki_context_for_agent()
        self.rules = load_rules()
        self.today_str = datetime.now().strftime("%Y-%m-%d")
        active_count = sum(1 for r in self.rules["global_rules"] if r["status"] == "active")
        logger.info(f"StrategyAgent 初始化完成，規則庫：{active_count} 條活躍規則")

    def generate_strategy(
        self,
        predictions: list[dict],
        news_analysis: dict,
        market_sentiment: dict,
        wiki_context: str | None = None,
    ) -> list[dict]:
        """融合 ML 預測、新聞分析、大盤情緒與歷史知識，產出每支股票的操作建議清單。"""
        if wiki_context is None:
            wiki_context = self.wiki_context

        # 組合 ML 預測摘要
        pred_lines = []
        for p in predictions:
            ticker = p.get("ticker", "")
            name = p.get("name", "")
            direction = p.get("direction", p.get("predicted_direction", "N/A"))
            magnitude = p.get("magnitude", p.get("predicted_change_pct", "N/A"))
            if isinstance(magnitude, float):
                magnitude = f"{magnitude:+.2f}%"
            confidence = p.get("confidence", p.get("model_confidence", "N/A"))
            if isinstance(confidence, float):
                confidence = f"{confidence:.0%}"
            us_shadow = p.get("us_lead_shadow", {}).get("predictions_pct", {}).get(
                "close_to_close_price_us")
            us_text = (f"，美股核心數值影子 {us_shadow:+.2f}%（實驗中）"
                       if isinstance(us_shadow, (int, float)) else "")
            pred_lines.append(
                f"  - {ticker} ({name}): {direction} {magnitude}，信心度 {confidence}{us_text}")

        pred_text = "\n".join(pred_lines) if pred_lines else "  （無 ML 預測資料）"

        # 組合新聞情緒摘要
        news_lines = []
        for ticker, analysis in news_analysis.items():
            if ticker == "market":
                continue
            name = analysis.get("ticker_name", ticker)
            sentiment = analysis.get("sentiment", "neutral")
            score = analysis.get("sentiment_score", 0.0)
            signal = analysis.get("strategy_signal", "hold")
            reason = analysis.get("strategy_reason", "")[:40]
            news_lines.append(f"  - {ticker} ({name}): {sentiment} (score={score:+.2f}), 訊號={signal}, {reason}")

        news_text = "\n".join(news_lines) if news_lines else "  （無新聞分析資料）"

        # 大盤情緒
        mkt_sentiment = market_sentiment.get("market_sentiment", "neutral")
        mkt_score = market_sentiment.get("score", 0.0)
        mkt_summary = market_sentiment.get("summary", "")

        # 股票資訊清單
        ticker_list = "\n".join(
            f"  - {ticker}: {info['name']} ({info['sector']} / {info['sub']})"
            for ticker, info in TICKER_INFO.items()
        )

        rules_text = format_rules_for_prompt(self.rules)

        # 國際同業監控（異常偵測 + 財報日曆 + 同業新聞）
        try:
            from agents.peer_monitor import build_peer_alert
            peer_alert_text = build_peer_alert()
            if peer_alert_text:
                logger.info("同業監控偵測到異動，已注入策略 prompt")
        except Exception as e:
            logger.debug(f"同業監控失敗：{e}")
            peer_alert_text = ""

        # 國際盤前快照（昨夜美股，台股開盤前最強領先指標）
        try:
            from agents.global_snapshot import build_global_alert
            global_alert_text = build_global_alert()
            if global_alert_text:
                logger.info("國際盤前快照已注入策略 prompt")
        except Exception as e:
            logger.debug(f"國際快照失敗：{e}")
            global_alert_text = ""

        user_prompt = f"""
分析日期：{self.today_str}
分析時間：{datetime.now().strftime('%H:%M')}

【大盤情緒】
情緒：{mkt_sentiment}（評分：{mkt_score:+.2f}）
摘要：{mkt_summary}

【ML 模型預測結果】
{pred_text}

【新聞情緒分析】
{news_text}
{global_alert_text}
{rules_text}
{peer_alert_text}

【歷史 Wiki 知識（近3日）】
{wiki_context[:1500]}

【待分析股票清單】
{ticker_list}

請根據以上資訊，為每支股票產出操作建議。
注意：
- 套用奧卡姆剃刀，若訊號不明確，優先建議 hold 或 watch
- 「美股核心數值影子」只作為個股方向衝突時的弱參考；尚未通過實盤門檻前，不得單獨翻轉正式方向或放大倉位。
- **國際盤前快照優先：** 若「🌍 國際盤前快照」判定 🔴 高風險（昨夜美股科技股重挫），
  今日所有電子族群一律降低倉位：position_size 最多到 small，且傾向 hold/watch；
  即使個股新聞偏多，也只能在「明確逢低買進且嚴設停損」前提下小量布局，禁止 large 倉位。
  若判定 🟠 偏空，position_size 最多到 medium。系統性賣壓凌駕個股基本面。
- 必須說明套用了哪條 CLIN 因果律（若無適用，填 "無適用因果律"）
- entry_price_hint / stop_loss_hint / take_profit_hint 若無法估計填 null

輸出嚴格 JSON 陣列（10個物件，與股票清單順序對應）：
[
  {{
    "ticker": "2327.TW",
    "name": "國巨",
    "action": "buy 或 sell 或 hold 或 watch",
    "position_size": "large 或 medium 或 small 或 none",
    "entry_price_hint": 數字或 null,
    "stop_loss_hint": 數字或 null,
    "take_profit_hint": 數字或 null,
    "reason": "中文理由，結合預測+新聞+歷史，60字以內",
    "risk_level": "high 或 medium 或 low",
    "clin_rule_applied": "套用的因果律名稱或 無適用因果律"
  }},
  ...
]
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": STRATEGY_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
            )

            raw_text = response.choices[0].message.content.strip()

            if raw_text.startswith("```"):
                lines = raw_text.split("\n")
                raw_text = "\n".join(lines[1:-1])

            strategies = json.loads(raw_text)

            if not isinstance(strategies, list):
                raise ValueError("回應不是 JSON 陣列")

            logger.info(f"策略生成完成，共 {len(strategies)} 支股票")
            for s in strategies:
                logger.info(
                    f"  {s.get('ticker')} {s.get('name')}: "
                    f"{s.get('action')} / {s.get('position_size')} / risk={s.get('risk_level')}"
                )

            return strategies

        except json.JSONDecodeError as e:
            logger.error(f"策略 JSON 解析失敗：{e}")
            return self._default_strategies()
        except OpenAIError as e:
            logger.error(f"OpenAI API 錯誤：{e}")
            return self._default_strategies()
        except Exception as e:
            logger.error(f"策略生成時發生未預期錯誤：{e}")
            return self._default_strategies()

    def reflexion_update(
        self,
        strategy_decisions: list[dict],
        actual_outcomes: dict,
        wiki_context: str | None = None,
    ) -> str:
        """盤後 Reflexion：AI 反思今日預測偏誤，判斷系統性錯誤或隨機雜訊，提出 CLIN 更新建議。"""
        if wiki_context is None:
            wiki_context = self.wiki_context

        # 組合今日結果對比（以 actual_outcomes 為主要資料來源）
        from config import ALL_STOCKS as _STOCKS
        _name_map = {k.replace(".TWO","").replace(".TW",""): v for k, v in _STOCKS.items()}
        decision_map = {
            d.get("ticker","").replace(".TWO","").replace(".TW",""): d
            for d in strategy_decisions
        }
        outcome_lines = []
        for ticker_key, actual in actual_outcomes.items():
            name = _name_map.get(ticker_key, ticker_key)
            actual_price = actual.get("actual_price", "N/A")
            predicted    = actual.get("predicted_price", "N/A")
            mape         = actual.get("mape", "N/A")
            dir_ok       = actual.get("direction_correct")
            dir_str      = "方向正確" if dir_ok is True else ("方向錯誤" if dir_ok is False else "")
            pct_str = ""
            if isinstance(actual_price, float) and isinstance(predicted, float):
                pct = (actual_price - predicted) / predicted * 100
                pct_str = f", 偏差={pct:+.2f}%"
            d = decision_map.get(ticker_key, {})
            action_str = f", 策略建議={d['action']}" if d.get("action") else ""
            outcome_lines.append(
                f"  - {ticker_key} {name}: 預測={predicted}, 實際={actual_price}, "
                f"MAPE={mape}%{pct_str}{action_str} {dir_str}"
            )

        outcome_text = "\n".join(outcome_lines) if outcome_lines else "  （無實際結果資料）"

        rules_text = format_rules_for_prompt(self.rules)

        reflexion_prompt = f"""
今日日期：{self.today_str}

【今日策略決策 vs 實際結果】
{outcome_text}

{rules_text}

【近期歷史 Wiki 記錄】
{wiki_context[:1200]}

請用中文條列式回答以下四點，每點不超過 60 字，不要輸出 JSON 或程式碼：

1. **偏誤識別**：哪些股票預測出現偏差？偏差方向為何？
2. **系統性 vs 隨機性**：這些偏差是可重複的系統性錯誤，還是單日隨機雜訊？
3. **規則有效性**：今日結果對現有因果律有何支持或挑戰？
4. **明日關注點**：明日盤前應特別觀察哪些訊號或指標？
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=600,
                messages=[
                    {"role": "system", "content": STRATEGY_SYSTEM_PROMPT},
                    {"role": "user",   "content": reflexion_prompt},
                ],
            )

            reflexion_report = response.choices[0].message.content.strip()
            logger.info("Reflexion 分析完成")
            return reflexion_report

        except OpenAIError as e:
            logger.error(f"Reflexion OpenAI API 錯誤：{e}")
            return "Reflexion 分析失敗（API 錯誤），請人工複查今日交易記錄。"
        except Exception as e:
            logger.error(f"Reflexion 時發生未預期錯誤：{e}")
            return f"Reflexion 分析失敗：{e}"

    def _default_strategies(self) -> list[dict]:
        """API 失敗時回傳所有股票的預設 hold 策略。"""
        return [
            {
                "ticker": ticker,
                "name": info["name"],
                "action": "hold",
                "position_size": "none",
                "entry_price_hint": None,
                "stop_loss_hint": None,
                "take_profit_hint": None,
                "reason": "API 分析失敗，維持觀望",
                "risk_level": "medium",
                "clin_rule_applied": "無適用因果律",
            }
            for ticker, info in TICKER_INFO.items()
        ]


if __name__ == "__main__":
    print("=== 測試 StrategyAgent ===")
    agent = StrategyAgent()

    sample_predictions = [
        {"ticker": "2327.TW", "name": "國巨", "direction": "up", "magnitude": "+1.2%", "confidence": 0.72},
        {"ticker": "2492.TW", "name": "華新科", "direction": "neutral", "magnitude": "+0.1%", "confidence": 0.55},
        {"ticker": "5425.TW", "name": "台半", "direction": "up", "magnitude": "+0.8%", "confidence": 0.65},
    ]

    sample_news_analysis = {
        "2327": {
            "ticker_name": "國巨",
            "sentiment": "bullish",
            "sentiment_score": 0.65,
            "strategy_signal": "buy",
            "strategy_reason": "MLCC需求回溫，法人持續買超",
        },
        "2492": {
            "ticker_name": "華新科",
            "sentiment": "neutral",
            "sentiment_score": 0.1,
            "strategy_signal": "hold",
            "strategy_reason": "無明顯催化劑",
        },
    }

    sample_market_sentiment = {
        "market_sentiment": "bullish",
        "score": 0.45,
        "summary": "外資買超，科技股表現強勢",
    }

    print("\n生成策略建議...")
    strategies = agent.generate_strategy(
        predictions=sample_predictions,
        news_analysis=sample_news_analysis,
        market_sentiment=sample_market_sentiment,
    )

    print(f"\n共 {len(strategies)} 支股票策略：")
    for s in strategies[:3]:
        print(f"  {s['ticker']} {s['name']}: {s['action']} ({s['position_size']}) - {s['reason'][:40]}")

    print("\n=== 測試完成 ===")
