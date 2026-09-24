"""
Wiki 知識庫讀寫模組：以 Karpathy 風格的 Markdown 記錄每日交易操作與決策。
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# 確保可以從專案根目錄執行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import WIKI_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 確保目錄存在
os.makedirs(WIKI_DIR, exist_ok=True)


def write_daily_record(
    date_str: str,
    predictions: list[dict],
    news_analysis: dict,
    strategy_decisions: dict,
    market_summary: str,
) -> str:
    """將當日完整交易記錄寫入 Markdown 知識庫檔案，回傳寫入路徑。"""
    output_path = os.path.join(WIKI_DIR, f"{date_str}.md")

    # ── 大盤摘要 ──────────────────────────────────────────────
    market_section = f"## 大盤摘要\n\n{market_summary}\n"

    # ── 預測結果 ──────────────────────────────────────────────
    pred_lines = ["## 預測結果（按股票）\n"]
    if predictions:
        pred_lines.append("| 股票代碼 | 公司名稱 | 原始方向 | 原始幅度 | 證據狀態 |")
        pred_lines.append("|---------|---------|---------|---------|---------|")
        for p in predictions:
            ticker = p.get("ticker", "N/A")
            name = p.get("name", "")
            direction = p.get("predicted_direction", p.get("direction", "N/A"))
            magnitude = p.get("predicted_change_pct", p.get("magnitude", "N/A"))
            if isinstance(magnitude, float):
                magnitude = f"{magnitude:+.2f}%"
            evidence = "已通過" if p.get("evidence_status") == "approved" else "影子觀察／樣本不足"
            pred_lines.append(f"| {ticker} | {name} | {direction} | {magnitude} | {evidence} |")
    else:
        pred_lines.append("_本日無 ML 預測資料_")
    pred_section = "\n".join(pred_lines) + "\n"

    # ── 新聞分析 ──────────────────────────────────────────────
    news_lines = ["## 新聞分析（盤前/中/後）\n"]
    if news_analysis:
        for ticker, analysis in news_analysis.items():
            if ticker == "market":
                continue
            ticker_name = analysis.get("ticker_name", ticker)
            sentiment = analysis.get("sentiment", "neutral")
            score = analysis.get("sentiment_score", 0.0)
            signal = analysis.get("strategy_signal", "hold")
            reason = analysis.get("strategy_reason", "")
            catalysts = analysis.get("key_catalysts", [])
            risks = analysis.get("risk_factors", [])
            session = analysis.get("session_type", "")

            news_lines.append(f"### {ticker}.TW {ticker_name} ({session})")
            news_lines.append(f"- **情緒**：{sentiment}（評分：{score:+.2f}）")
            news_lines.append(f"- **訊號**：{signal}")
            news_lines.append(f"- **理由**：{reason}")
            if catalysts:
                news_lines.append(f"- **關鍵催化劑**：")
                for c in catalysts:
                    news_lines.append(f"  - {c}")
            if risks:
                news_lines.append(f"- **風險因子**：")
                for r in risks:
                    news_lines.append(f"  - {r}")
            news_lines.append("")
    else:
        news_lines.append("_本日無新聞分析資料_")
    news_section = "\n".join(news_lines) + "\n"

    # ── 策略決策 ──────────────────────────────────────────────
    strat_lines = ["## 策略決策\n"]
    if strategy_decisions:
        strat_lines.append("| 股票 | 操作 | 倉位 | 進場價 | 停損 | 停利 | 風險 | 理由 |")
        strat_lines.append("|-----|------|------|-------|------|------|------|------|")
        for ticker, decision in strategy_decisions.items():
            action = decision.get("action", "hold")
            size = decision.get("position_size", "none")
            entry = decision.get("entry_price_hint", "-")
            sl = decision.get("stop_loss_hint", "-")
            tp = decision.get("take_profit_hint", "-")
            risk = decision.get("risk_level", "medium")
            reason = decision.get("reason", "")[:30]
            name = decision.get("name", "")
            strat_lines.append(
                f"| {ticker} {name} | {action} | {size} | {entry} | {sl} | {tp} | {risk} | {reason} |"
            )
    else:
        strat_lines.append("_本日無策略決策資料_")
    strat_section = "\n".join(strat_lines) + "\n"

    # ── 自我反思（Reflexion）──────────────────────────────────
    reflexion_section = (
        "## 自我反思（Reflexion）\n\n"
        "_待盤後填入：分析預測偏誤是系統性錯誤還是隨機雜訊。_\n"
    )

    # ── 明日關注點 ────────────────────────────────────────────
    tomorrow_section = (
        "## 明日關注點\n\n"
        "_待盤後填入：根據今日分析提煉的明日重點觀察。_\n"
    )

    # ── 組合完整 Markdown ─────────────────────────────────────
    content = f"""# 交易記錄 {date_str}

> 建立時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{market_section}
{pred_section}
{news_section}
{strat_section}
{reflexion_section}
{tomorrow_section}
"""

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Wiki 記錄已寫入：{output_path}")
    except OSError as e:
        logger.error(f"寫入 Wiki 失敗：{e}")
        raise

    return output_path


def read_recent_wiki(days: int = 3) -> str:
    """讀取最近 N 天的 Wiki 記錄，回傳以 '---' 分隔的合併 Markdown 字串。"""
    today = datetime.now().date()
    sections = []

    for offset in range(days):
        target_date = today - timedelta(days=offset)
        date_str = target_date.strftime("%Y-%m-%d")
        wiki_path = os.path.join(WIKI_DIR, f"{date_str}.md")

        if os.path.exists(wiki_path):
            try:
                with open(wiki_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                sections.append(content)
                logger.debug(f"讀取 Wiki：{wiki_path}")
            except OSError as e:
                logger.warning(f"無法讀取 Wiki {date_str}：{e}")
        else:
            logger.debug(f"Wiki 不存在，跳過：{wiki_path}")

    if not sections:
        return "_近期無 Wiki 記錄_"

    return "\n\n---\n\n".join(sections)


def get_wiki_context_for_agent() -> str:
    """讀取近3日 Wiki 並格式化為 Agent 可用的 context 字串。"""
    recent_content = read_recent_wiki(days=3)

    context = f"""=== 歷史交易知識庫（近3日記錄）===
建立時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}

{recent_content}

=== 知識庫結束 ===
"""
    return context


def append_intraday_note(date_str: str, note: str, session_type: str = "intraday") -> None:
    """在當日 Wiki 末尾追加盤中或盤後備注。"""
    wiki_path = os.path.join(WIKI_DIR, f"{date_str}.md")

    session_label = {
        "intraday": "盤中更新",
        "post_market": "盤後記錄",
    }.get(session_type, session_type)

    timestamp = datetime.now().strftime("%H:%M:%S")
    append_content = f"\n## {session_label}（{timestamp}）\n\n{note}\n"

    try:
        # 若檔案不存在則建立
        mode = "a" if os.path.exists(wiki_path) else "w"
        with open(wiki_path, mode, encoding="utf-8") as f:
            if mode == "w":
                f.write(f"# 交易記錄 {date_str}\n\n> 建立時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(append_content)
        logger.info(f"已追加 {session_label} 備注至：{wiki_path}")
    except OSError as e:
        logger.error(f"追加 Wiki 備注失敗：{e}")
        raise


if __name__ == "__main__":
    print("=== 測試 Wiki Agent ===")
    today_str = datetime.now().strftime("%Y-%m-%d")

    # 測試寫入
    sample_predictions = [
        {"ticker": "2327.TW", "name": "國巨", "direction": "up", "magnitude": "+1.5%", "confidence": 0.72},
        {"ticker": "2492.TW", "name": "華新科", "direction": "neutral", "magnitude": "+0.1%", "confidence": 0.55},
    ]
    sample_news_analysis = {
        "2327": {
            "ticker_name": "國巨",
            "sentiment": "bullish",
            "sentiment_score": 0.65,
            "strategy_signal": "buy",
            "strategy_reason": "MLCC需求回溫，法人持續買超",
            "key_catalysts": ["Q2營收創高"],
            "risk_factors": ["美中貿易風險"],
            "session_type": "pre_market",
        }
    }
    sample_strategy = {
        "2327": {
            "name": "國巨",
            "action": "buy",
            "position_size": "medium",
            "entry_price_hint": 380.0,
            "stop_loss_hint": 370.0,
            "take_profit_hint": 400.0,
            "risk_level": "medium",
            "reason": "MLCC需求回溫，技術面突破整理區",
        }
    }

    path = write_daily_record(
        today_str,
        sample_predictions,
        sample_news_analysis,
        sample_strategy,
        "台股大盤今日偏多，外資連續買超，電子權值股表現強勢。",
    )
    print(f"寫入成功：{path}")

    # 測試追加備注
    append_intraday_note(today_str, "盤中觀察：2327 突破前高，成交量放大。", "intraday")

    # 測試讀取
    context = get_wiki_context_for_agent()
    print(f"\nWiki Context 長度：{len(context)} 字元")
    print(context[:500])

    print("\n=== 測試完成 ===")
