"""
同業監控：盤前偵測國際同業價格異動與財報日曆，產生 strategy prompt 警示。

方法一：同業價格異常偵測 — 村田/TDK/ON/VSH 單日漲跌幅超過閾值即警示
方法三：財報日曆提醒 — 同業財報週前提示，讓策略提前考量
"""

import sys
import os
import logging
from datetime import date, datetime

import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MACRO_INDICES, PASSIVE_PEERS, POWER_PEERS, PEER_NAMES,
    PEER_ANOMALY_THRESHOLD, PEER_EARNINGS_MONTHS,
    PASSIVE_PEER_KEYWORDS, POWER_PEER_KEYWORDS,
)

logger = logging.getLogger(__name__)


# ── 方法一：同業價格異常偵測 ──────────────────────────────────────────────────

def detect_peer_anomalies() -> dict:
    """抓取同業最新單日漲跌幅，回傳超過閾值的異動清單。

    Returns:
        {
          "passive": [{"ticker","name","pct","direction"}],
          "power":   [...],
          "macro":   [...],
        }
    """
    result = {"passive": [], "power": [], "macro": []}

    groups = {
        "passive": PASSIVE_PEERS,
        "power":   POWER_PEERS,
        "macro":   MACRO_INDICES,
    }

    for group, peers in groups.items():
        for ticker in peers:
            try:
                hist = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
                if len(hist) < 2:
                    continue
                prev  = float(hist["Close"].iloc[-2])
                last  = float(hist["Close"].iloc[-1])
                pct   = (last - prev) / prev * 100
                if abs(pct) >= PEER_ANOMALY_THRESHOLD:
                    result[group].append({
                        "ticker":    ticker,
                        "name":      PEER_NAMES.get(ticker, ticker),
                        "pct":       round(pct, 2),
                        "direction": "up" if pct > 0 else "down",
                    })
                    logger.warning(f"[同業異動] {PEER_NAMES.get(ticker, ticker)} {pct:+.2f}%")
            except Exception as e:
                logger.debug(f"同業異常偵測失敗 {ticker}：{e}")

    return result


# ── 方法三：財報日曆提醒 ──────────────────────────────────────────────────────

def get_earnings_reminders(today: date | None = None) -> list:
    """檢查同業是否進入財報週（每季首月下旬 20-31 日），回傳提醒清單。"""
    if today is None:
        today = date.today()

    reminders = []
    if today.month in set(sum(PEER_EARNINGS_MONTHS.values(), [])):
        # 財報通常落在月底，當月 18 日後開始提醒
        if today.day >= 18:
            for ticker, months in PEER_EARNINGS_MONTHS.items():
                if today.month in months:
                    reminders.append({
                        "ticker": ticker,
                        "name":   PEER_NAMES.get(ticker, ticker),
                        "window": f"{today.month}月下旬",
                    })
    return reminders


# ── 格式化為 prompt 文字 ──────────────────────────────────────────────────────

def _fetch_peer_news() -> dict:
    """抓取被動/功率同業關鍵字新聞（方法二），失敗時回傳空清單。"""
    try:
        from news.scraper import scrape_peer_news
        return {
            "passive": scrape_peer_news(PASSIVE_PEER_KEYWORDS, max_per_keyword=2),
            "power":   scrape_peer_news(POWER_PEER_KEYWORDS,   max_per_keyword=2),
        }
    except Exception as e:
        logger.debug(f"同業新聞抓取失敗：{e}")
        return {"passive": [], "power": []}


def build_peer_alert(today: date | None = None, include_news: bool = True) -> str:
    """整合異常偵測 + 財報日曆 + 同業新聞，產生注入 strategy prompt 的警示文字。"""
    anomalies = detect_peer_anomalies()
    earnings  = get_earnings_reminders(today)
    peer_news = _fetch_peer_news() if include_news else {"passive": [], "power": []}

    has_anomaly = any(anomalies[g] for g in anomalies)
    has_news    = any(peer_news[g] for g in peer_news)
    if not has_anomaly and not earnings and not has_news:
        return ""

    lines = ["\n【🌐 國際同業監控】"]

    if anomalies["passive"]:
        lines.append("■ 被動元件同業異動（影響國巨/華新科/禾伸堂/立隆電/臺慶科）：")
        for a in anomalies["passive"]:
            arrow = "▲" if a["direction"] == "up" else "▼"
            lines.append(f"   {arrow} {a['name']} {a['pct']:+.2f}%")

    if anomalies["power"]:
        lines.append("■ 功率元件同業異動（影響台半/強茂/德微/朋程/富鼎）：")
        for a in anomalies["power"]:
            arrow = "▲" if a["direction"] == "up" else "▼"
            lines.append(f"   {arrow} {a['name']} {a['pct']:+.2f}%")

    if anomalies["macro"]:
        lines.append("■ 宏觀指標異動：")
        for a in anomalies["macro"]:
            arrow = "▲" if a["direction"] == "up" else "▼"
            lines.append(f"   {arrow} {a['name']} {a['pct']:+.2f}%")

    if peer_news["passive"]:
        lines.append("■ 被動元件同業新聞：")
        for n in peer_news["passive"][:4]:
            lines.append(f"   • {n['title'][:50]}")

    if peer_news["power"]:
        lines.append("■ 功率元件同業新聞：")
        for n in peer_news["power"][:4]:
            lines.append(f"   • {n['title'][:50]}")

    if earnings:
        lines.append("■ 同業財報週提醒：")
        for e in earnings:
            lines.append(f"   📅 {e['name']} 預計 {e['window']} 發布財報")

    lines.append(
        "\n判讀提示：同業大漲不必然等於台廠跟漲（產業鏈位置、報價傳導有時間差）。"
        "請結合個股新聞情緒綜合判斷，避免單純跟隨同業價格。"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=== 同業異常偵測 ===")
    anomalies = detect_peer_anomalies()
    for group, items in anomalies.items():
        print(f"  {group}: {items}")

    print("\n=== 財報日曆（測試 2026-05-20）===")
    print(get_earnings_reminders(date(2026, 5, 20)))

    print("\n=== 完整警示文字 ===")
    print(build_peer_alert() or "（今日無異動）")
