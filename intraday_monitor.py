"""
盤中風險監控 + Discord 推播：每次執行抓一次盤中價，比對風險價位，觸發才推播。
由 Windows 排程器在交易時段每 10 分鐘執行一次（每次=一輪檢查，用狀態檔避免重複推播）。

觸發條件（都是「規則觸發的風險警示」，非買賣建議）：
  1. 跌破當日「80% 區間下緣」（分位數模型的風險地板）→ 已超出模型最壞預期
  2. 盤中跌幅 ≤ 急殺門檻（預設 -4%）→ 急殺警示
  3. （選用）跌破你在 positions.json 設定的停損價

報價來源：證交所 MIS 即時 API（mis.twse.com.tw），近即時（延遲約數十秒），
          免費、不需券商帳戶。yfinance 不提供台股盤中資料，故不採用。
⚠️ 此程式只在價位碰到你設定的風險線時通知你，由你自行決定，不代下單、不喊買賣點。

設定：
  在 .env 加一行：DISCORD_WEBHOOK_URL=你的webhook網址
用法：
  python intraday_monitor.py --test        # 測試推播是否會收到
  python intraday_monitor.py --announce     # 推播今日各股風險地板（盤前用）
  python intraday_monitor.py                # 跑一輪檢查（排程每10分鐘呼叫）
"""
import os
import sys
import json
import logging
from datetime import date, datetime

import requests
import yfinance as yf
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

load_dotenv()
WEBHOOK = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CRASH_PCT   = -4.0    # 盤中跌幅低於此值 → 急殺警示（%）
STATE_DIR   = os.path.join(cfg.WIKI_DIR, "intraday_state")
os.makedirs(STATE_DIR, exist_ok=True)


# ── Discord 推播 ──────────────────────────────────────────────────────────────

def send_discord(content: str) -> bool:
    """推播訊息到 Discord webhook。"""
    if not WEBHOOK:
        logger.warning("未設定 DISCORD_WEBHOOK_URL，跳過推播")
        return False
    try:
        r = requests.post(WEBHOOK, json={"content": content}, timeout=15)
        return r.status_code in (200, 204)
    except Exception as e:
        logger.warning(f"Discord 推播失敗：{e}")
        return False


# ── 風險地板（每日快取）──────────────────────────────────────────────────────

def _floors_path() -> str:
    return os.path.join(STATE_DIR, f"floors_{date.today().isoformat()}.json")


def todays_risk_floors() -> dict:
    """回傳各股今日風險地板 {code: {name, q10, q50, prev_close}}，當日只算一次並快取。"""
    path = _floors_path()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    from quantile_forecast import predict_quantile
    floors = {}
    for tk in cfg.ALL_STOCKS:
        code = tk.replace(".TWO", "").replace(".TW", "")
        try:
            q = predict_quantile(tk)
            floors[code] = {"name": q["name"], "ticker": tk,
                            "q10": q["q10_price"], "q50": q["q50_price"],
                            "prev_close": q["last_close"]}
        except Exception as e:
            logger.debug(f"{tk} 風險地板計算失敗：{e}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(floors, f, ensure_ascii=False, indent=2)
    return floors


# ── 盤中報價（證交所 MIS 即時 API；yfinance 不提供台股盤中資料）──────────────

MIS_URL = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp"
_QUOTE_CACHE = None   # 每個排程進程只抓一次（一輪檢查共用）


def _pick_price(s: dict):
    """從 MIS 回應取現價：最近成交價優先，無則用最佳買賣價中點，再退而求其次。"""
    z = s.get("z")
    if z and z not in ("-", ""):
        return float(z)
    b = (s.get("b") or "").split("_")[0]
    a = (s.get("a") or "").split("_")[0]
    try:
        if b and a and b not in ("-", "") and a not in ("-", ""):
            return (float(b) + float(a)) / 2
    except ValueError:
        pass
    for k in ("o", "h", "l", "y"):   # 開/高/低/昨收 最後手段
        v = s.get(k)
        if v and v not in ("-", ""):
            return float(v)
    return None


def _fetch_all_quotes() -> dict:
    """一次抓全部股票的證交所即時報價，回傳 {純代碼: 現價}。"""
    global _QUOTE_CACHE
    if _QUOTE_CACHE is not None:
        return _QUOTE_CACHE
    ex_ch = "|".join(
        ("otc_" if tk.endswith(".TWO") else "tse_") +
        tk.replace(".TWO", "").replace(".TW", "") + ".tw"
        for tk in cfg.ALL_STOCKS
    )
    out = {}
    try:
        r = requests.get(MIS_URL, params={"ex_ch": ex_ch, "json": "1", "delay": "0"},
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        for s in r.json().get("msgArray", []):
            price = _pick_price(s)
            if price is not None:
                out[s.get("c")] = price
    except Exception as e:
        logger.warning(f"MIS 即時報價失敗：{e}")
    _QUOTE_CACHE = out
    return out


def get_quote(ticker: str, prev_close: float):
    """回傳 (現價, 盤中漲跌幅%)。用證交所 MIS 即時報價，失敗回 (None, None)。"""
    code = ticker.replace(".TWO", "").replace(".TW", "")
    price = _fetch_all_quotes().get(code)
    if price is None:
        return None, None
    pct = (price - prev_close) / prev_close * 100 if prev_close else None
    return price, pct


# ── 觸發狀態（避免同一警示重複推播）──────────────────────────────────────────

def _state_path() -> str:
    return os.path.join(STATE_DIR, f"fired_{date.today().isoformat()}.json")


def _load_state() -> dict:
    p = _state_path()
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_state(st: dict):
    with open(_state_path(), "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)


def _load_positions() -> dict:
    """選用：讀 positions.json 取得持股停損價 {code: {stop_loss: 價}}。"""
    p = os.path.join(cfg.WIKI_DIR, "positions.json")
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


# ── 主檢查 ────────────────────────────────────────────────────────────────────

def check_and_alert() -> int:
    """跑一輪檢查，觸發的風險推播到 Discord，回傳本輪新觸發數。"""
    floors = todays_risk_floors()
    state = _load_state()
    positions = _load_positions()
    now = datetime.now().strftime("%H:%M")
    new_alerts = 0

    for code, info in floors.items():
        price, pct = get_quote(info["ticker"], info["prev_close"])
        if price is None:
            continue
        fired = set(state.get(code, []))

        # 1) 跌破 80% 區間下緣（風險地板）
        if price < info["q10"] and "floor" not in fired:
            send_discord(
                f"🔴 **{code} {info['name']}** 跌破風險地板\n"
                f"現價 **{price:.1f}**（{pct:+.1f}%）< 80%區間下緣 {info['q10']:.1f}\n"
                f"→ 已超出模型最壞預期，建議檢視是否減碼/認錯（{now}，證交所即時報價）"
            )
            fired.add("floor"); new_alerts += 1

        # 2) 盤中急殺
        if pct is not None and pct <= CRASH_PCT and "crash" not in fired:
            send_discord(
                f"⚠️ **{code} {info['name']}** 盤中急殺 **{pct:+.1f}%**\n"
                f"現價 {price:.1f}（昨收 {info['prev_close']:.1f}）（{now}，證交所即時報價）"
            )
            fired.add("crash"); new_alerts += 1

        # 3) 自訂停損（若有設定持股）
        sl = positions.get(code, {}).get("stop_loss")
        if sl and price <= sl and "stop" not in fired:
            send_discord(
                f"🛑 **{code} {info['name']}** 觸及你的停損價 {sl}\n"
                f"現價 {price:.1f}（{now}，證交所即時報價）"
            )
            fired.add("stop"); new_alerts += 1

        if fired:
            state[code] = sorted(fired)

    _save_state(state)
    if new_alerts:
        logger.info(f"本輪推播 {new_alerts} 則風險警示")
    else:
        logger.info("本輪無觸發")
    return new_alerts


def announce_floors():
    """盤前推播今日各股風險地板一覽。"""
    floors = todays_risk_floors()
    lines = [f"📊 **{date.today().isoformat()} 今日風險地板**（跌破=超出模型最壞預期）"]
    for code, i in floors.items():
        lines.append(f"・{code} {i['name']}：地板 {i['q10']:.1f}（昨收 {i['prev_close']:.1f}）")
    lines.append("_盤中跌破會自動通知你；報價延遲約15分_")
    send_discord("\n".join(lines))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if "--test" in sys.argv:
        ok = send_discord("✅ 盤中風險監控測試訊息：你會在這個頻道收到風險警示。")
        print("測試推播：", "成功，去 Discord 看有沒有收到" if ok else "失敗，檢查 .env 的 DISCORD_WEBHOOK_URL")
    elif "--announce" in sys.argv:
        announce_floors()
        print("已推播今日風險地板")
    else:
        check_and_alert()
