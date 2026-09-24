"""
國際盤前快照：抓昨夜美股收盤，注入盤前 strategy prompt。

台股 08:30 開盤前，美股已收盤——昨夜美股表現是台股當日最強的領先指標。
今天（2026-06-08）台股暴跌的直接原因，就是上週五美股科技股崩盤
（費半 -10%、NVIDIA -6%、台積電/鴻海 ADR -6%），這個快照能在盤前預警。
"""

import sys
import os
import logging

import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GLOBAL_INDICES, GLOBAL_STOCKS, GLOBAL_ALERT_THRESHOLD

logger = logging.getLogger(__name__)


def _last_change(ticker: str):
    """回傳 (最後收盤日, 隔日漲跌幅%)，抓不到回 (None, None)。"""
    try:
        h = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
        # 去除 NaN 收盤後再算，避免 yfinance 偶發 NaN 導致漲跌幅變 nan
        closes = h["Close"].dropna()
        if len(closes) >= 2:
            pct = (closes.iloc[-1] / closes.iloc[-2] - 1) * 100
            import math
            if not math.isnan(pct):
                return closes.index[-1].date(), round(float(pct), 2)
    except Exception as e:
        logger.debug(f"國際快照抓取失敗 {ticker}：{e}")
    return None, None


def _expected_us_date(as_of=None):
    """按 NYSE/Nasdaq 實際交易日與 08:30 可用時間找最近美股收盤。"""
    from datetime import date, timedelta
    import pandas as pd
    from us_lead_study import schedule

    day = as_of or date.today()
    cutoff = pd.Timestamp(f"{day.isoformat()}T08:30:00", tz="Asia/Taipei").tz_convert("UTC")
    calendar = schedule((day - timedelta(days=15)).isoformat(), day.isoformat())
    eligible = calendar[calendar["available_at"] < cutoff]
    if eligible.empty:
        return None
    return date.fromisoformat(eligible.iloc[-1]["session"])


def _stale_days(data_date, as_of=None) -> int:
    """資料日期落後預期美股交易日幾個工作日（0=即時）。"""
    from datetime import date
    if data_date is None:
        return 99
    expected = _expected_us_date(as_of=as_of)
    if expected is None:
        return 99
    if data_date >= expected:
        return 0
    # 假日/提早休市依真實交易日計算，不能把 Labor Day 當成過期一天。
    from us_lead_study import schedule
    return int(sum(data_date < date.fromisoformat(session) <= expected
                   for session in schedule(data_date.isoformat(), expected.isoformat())["session"]))


def get_global_snapshot() -> dict:
    """抓取昨夜美股指數與重點個股收盤變化，每項標註資料日期與新鮮度。"""
    snapshot = {"indices": [], "stocks": []}

    for ticker, name in GLOBAL_INDICES.items():
        d, pct = _last_change(ticker)
        if pct is not None:
            snapshot["indices"].append({
                "ticker": ticker, "name": name, "pct": pct,
                "date": d, "stale_days": _stale_days(d),
            })

    for ticker, name in GLOBAL_STOCKS.items():
        d, pct = _last_change(ticker)
        if pct is not None:
            snapshot["stocks"].append({
                "ticker": ticker, "name": name, "pct": pct,
                "date": d, "stale_days": _stale_days(d),
            })

    return snapshot


def _risk_verdict(snapshot: dict) -> str:
    """判斷風險基調。只採用『即時（stale_days=0）』訊號，優先台積電ADR與費半。"""
    def fresh(items, ticker):
        return next((x["pct"] for x in items
                     if x["ticker"] == ticker and x["stale_days"] == 0), None)

    tsm  = fresh(snapshot["stocks"], "TSM")     # 台積電 ADR（最相關且通常即時）
    sox  = fresh(snapshot["indices"], "^SOX")   # 費半（指數，常落後1天）
    ixic = fresh(snapshot["indices"], "^IXIC")

    # 加權：台積電 ADR 0.5、費半 0.3、那斯達克 0.2，只用即時且存在者
    weights = [("tsm", tsm, 0.5), ("sox", sox, 0.3), ("ixic", ixic, 0.2)]
    used = [(v, w) for _, v, w in weights if v is not None]
    if not used:
        return "⚪ 中性：無即時美股訊號（資料延遲），今日以個股與盤面為主"

    avg = sum(v * w for v, w in used) / sum(w for _, w in used)

    if avg <= -3:
        return "🔴 高風險：昨夜美股科技股重挫，台股電子今日高機率承壓，建議防禦、降低倉位"
    elif avg <= -1:
        return "🟠 偏空：昨夜美股科技股走弱，台股電子今日偏空，謹慎追高"
    elif avg >= 3:
        return "🟢 偏多：昨夜美股科技股大漲，台股電子今日有望跟漲"
    elif avg >= 1:
        return "🟢 偏多：昨夜美股科技股收紅，台股電子情緒偏正面"
    else:
        return "⚪ 中性：昨夜美股變動有限，今日以個股與盤面為主"


def build_global_alert() -> str:
    """產生注入盤前 strategy prompt 的國際快照文字。"""
    snap = get_global_snapshot()
    if not snap["indices"] and not snap["stocks"]:
        return ""

    lines = [f"\n【🌍 國際盤前快照｜預期美股交易日 {_expected_us_date()}】"]

    def fmt(item):
        tag = f"({item['date']})" if item["stale_days"] == 0 else \
              f"({item['date']} ⚠️落後{item['stale_days']}日)"
        return f"{item['name']} {item['pct']:+.2f}% {tag}"

    if snap["indices"]:
        lines.append("■ 美股指數：" + "  ".join(fmt(i) for i in snap["indices"]))

    if snap["stocks"]:
        notable = [s for s in snap["stocks"] if abs(s["pct"]) >= GLOBAL_ALERT_THRESHOLD]
        if notable:
            lines.append("■ 重點個股：" + "  ".join(fmt(s) for s in notable))

    # 新鮮度總結
    all_items = snap["indices"] + snap["stocks"]
    stale = [x for x in all_items if x["stale_days"] > 0]
    if stale:
        lines.append(f"⚠️ 資料新鮮度：{len(all_items)-len(stale)}/{len(all_items)} 即時；"
                     f"指數常落後1日，風險判定優先採用即時的台積電ADR與費半。")

    lines.append(_risk_verdict(snap))
    lines.append(
        "判讀提示：台股 08:30 開盤前美股已收盤，昨夜美股是今日台股最強領先指標。"
        "台積電/鴻海 ADR 連動性最高且通常即時；費半（SOX）指數常落後1日。"
    )
    return "\n".join(lines)


def record_snapshot_to_wiki(date_str: str, alert_text: str | None = None) -> str:
    """把當日國際快照（含正確美股日期與新鮮度）留底到 snapshot_history.md，供日後驗證。

    解決「無法重建當時用哪天美股」的問題——每天的快照原文都留檔。
    """
    import os
    from config import WIKI_DIR
    if alert_text is None:
        alert_text = build_global_alert()
    path = os.path.join(WIKI_DIR, "snapshot_history.md")
    header = f"\n\n## {date_str}（預期美股交易日 {_expected_us_date()}）\n"
    body = alert_text.strip() if alert_text else "（無國際快照資料）"
    try:
        os.makedirs(WIKI_DIR, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(header + body + "\n")
        logger.info(f"國際快照已留底：{path}")
    except Exception as e:
        logger.warning(f"快照留底失敗：{e}")
    return path


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(build_global_alert())
