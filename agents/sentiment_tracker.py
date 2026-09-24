"""
情緒連續性追蹤器：記錄個股持續新聞情緒，僅供事後觀察。

運作邏輯：
  每日盤前/中/後新聞分析完成後，呼叫 update() 記錄每股情緒分數。
  當某股連續 N 天 sentiment_score 絕對值 >= threshold（同方向），
  可標記連續訊號；在實盤證據足夠之前，不送進正式買賣策略。
"""

import sys
import os
import json
import logging
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WIKI_DIR

HISTORY_PATH  = os.path.join(WIKI_DIR, "sentiment_history.json")
OVERRIDE_THRESHOLD   = 0.7    # sentiment_score 絕對值超過此值視為強訊號
CONSECUTIVE_DAYS     = 2      # 連續幾天觸發覆蓋
MAX_HISTORY_DAYS     = 10     # 每支股票最多保留幾天記錄

logger = logging.getLogger(__name__)


# ── 讀寫歷史 ──────────────────────────────────────────────────────────────────

def load_history() -> dict:
    """載入情緒歷史記錄。"""
    if not os.path.exists(HISTORY_PATH):
        return {}
    try:
        with open(HISTORY_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_history(history: dict) -> None:
    """儲存情緒歷史記錄。"""
    os.makedirs(WIKI_DIR, exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# ── 更新與查詢 ────────────────────────────────────────────────────────────────

def update(ticker: str, date_str: str, score: float, signal: str, session: str = "pre_market") -> None:
    """記錄某股某日的情緒分數（只保留最後 N 天）。"""
    history = load_history()
    if ticker not in history:
        history[ticker] = []

    # 同日同 session 更新而非新增
    existing = next((r for r in history[ticker]
                     if r["date"] == date_str and r["session"] == session), None)
    if existing:
        existing["score"]  = score
        existing["signal"] = signal
    else:
        history[ticker].append({
            "date":    date_str,
            "session": session,
            "score":   score,
            "signal":  signal,
        })

    # 只保留最近 MAX_HISTORY_DAYS 天（每天取 pre_market 分數為代表）
    history[ticker] = sorted(history[ticker], key=lambda x: x["date"])[-MAX_HISTORY_DAYS * 3:]
    save_history(history)


def get_override_flags(date_str: str | None = None) -> dict:
    """分析所有股票，回傳需要新聞覆蓋的股票清單。

    Returns:
        {
          "2481": {
            "trigger": True,
            "direction": "bullish",
            "consecutive_days": 3,
            "avg_score": 0.82,
            "override_message": "...",
          },
          ...
        }
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    history = load_history()
    flags = {}

    for ticker, records in history.items():
        # 只看 pre_market 記錄，按日期排序
        pre_records = sorted(
            [r for r in records if r["session"] == "pre_market"],
            key=lambda x: x["date"],
            reverse=True,  # 最新在前
        )

        if len(pre_records) < CONSECUTIVE_DAYS:
            continue

        # 取最近 N 天
        recent = pre_records[:CONSECUTIVE_DAYS]
        scores = [r["score"] for r in recent]

        # 判斷是否連續強訊號（同方向且絕對值均超過閾值）
        all_strong   = all(abs(s) >= OVERRIDE_THRESHOLD for s in scores)
        all_bullish  = all(s > 0 for s in scores)
        all_bearish  = all(s < 0 for s in scores)
        same_direction = all_bullish or all_bearish

        if all_strong and same_direction:
            direction  = "bullish" if all_bullish else "bearish"
            avg_score  = sum(scores) / len(scores)
            trend_verb = "持續看多" if direction == "bullish" else "持續看空"

            flags[ticker] = {
                "trigger":           True,
                "direction":         direction,
                "consecutive_days":  CONSECUTIVE_DAYS,
                "avg_score":         round(avg_score, 3),
                "override_message":  (
                    f"⚡ 新聞覆蓋模式已觸發（{ticker}）：\n"
                    f"   連續 {CONSECUTIVE_DAYS} 日{trend_verb}，"
                    f"平均 sentiment_score={avg_score:+.2f}\n"
                    f"   ML 預測可能低估趨勢，請以新聞訊號為主要依據，"
                    f"適當放大倉位或追漲/殺跌。"
                ),
            }
            logger.warning(
                f"[覆蓋觸發] {ticker} 連續 {CONSECUTIVE_DAYS} 日 {direction} "
                f"(avg={avg_score:+.2f})"
            )

    return flags


def format_override_prompt(flags: dict) -> str:
    """將覆蓋旗標格式化為可注入 strategy_agent prompt 的文字。"""
    if not flags:
        return ""

    lines = ["\n【⚡ 新聞優先覆蓋警示】"]
    for ticker, info in flags.items():
        lines.append(info["override_message"])
    lines.append(
        "\n覆蓋規則：以上股票的 ML 預測可能因題材爆發而嚴重低估，"
        "策略應以新聞情緒方向為主，允許更積極的倉位建議。"
        "但仍須通過奧卡姆剃刀檢驗：確認多日訊號一致才執行，避免單日雜訊誤判。"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    # 模擬 3 天強訊號測試
    for day, score in [("2026-05-31", 0.75), ("2026-06-01", 0.82), ("2026-06-02", 0.88)]:
        update("2481", day, score, "buy", "pre_market")

    flags = get_override_flags("2026-06-02")
    print("觸發覆蓋的股票：", list(flags.keys()))
    print(format_override_prompt(flags))
