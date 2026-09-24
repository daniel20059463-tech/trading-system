"""抓 GDELT 全球新聞的每日「聲量+語氣」，供國際新聞→台股訊號測試。
GDELT 龜毛：限速(5秒1次)、偶爾 timeout、JSON 要容錯。故大間隔 + 重試 + 穩健解析。

輸出 D:\\trading-wiki\\gdelt_intl_news.json：{keyword: {date: {vol, tone}}}
用法：python fetch_gdelt_news.py
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
import requests

URL = "https://api.gdeltproject.org/api/v2/doc/doc"
OUT = os.path.join(cfg.WIKI_DIR, "gdelt_intl_news.json")
# 與這些台股相關的國際題材關鍵字
KEYWORDS = ["MLCC capacitor", "semiconductor shortage", "chip demand", "passive components"]
START, END = "20240601000000", "20260601000000"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def _fetch_series(kw: str, mode: str):
    """單一 mode（timelinevol 或 timelinetone）的每日時間線 → {date: value}。"""
    for attempt in range(8):
        try:
            r = requests.get(URL, headers=HEADERS, timeout=90, params={
                "query": kw, "mode": mode,
                "startdatetime": START, "enddatetime": END, "format": "json"})
            txt = r.text.strip().lstrip("﻿")
            if not txt or txt[0] not in "{[":          # 429/錯誤頁
                time.sleep(18); continue
            j = json.loads(txt)
            out = {}
            for tl in j.get("timeline", []):
                for pt in tl.get("data", []):
                    out[pt["date"][:8]] = pt["value"]
            if out:
                return out
            time.sleep(12)
        except Exception as e:
            print(f"      重試({attempt+1}) {mode}: {str(e)[:40]}"); time.sleep(18)
    return None


def _fetch_one(kw: str):
    """聲量(vol)+語氣(tone)分兩次抓，合併成 {date: {vol, tone}}。"""
    vol = _fetch_series(kw, "timelinevol")
    time.sleep(8)
    tone = _fetch_series(kw, "timelinetone")
    if not vol and not tone:
        return None
    out = {}
    for d, v in (vol or {}).items():
        out[d] = {"vol": v}
    for d, t in (tone or {}).items():
        out.setdefault(d, {})["tone"] = t
    return out


def run():
    data = {}
    if os.path.exists(OUT):
        data = json.load(open(OUT, encoding="utf-8"))
    for kw in KEYWORDS:
        if kw in data and len(data[kw]) > 100:
            print(f"  已有 {kw}（{len(data[kw])}天），跳過"); continue
        print(f"  抓取「{kw}」..."); sys.stdout.flush()
        res = _fetch_one(kw)
        if res:
            data[kw] = res
            json.dump(data, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"    ✅ {kw}: {len(res)} 天 {min(res)}~{max(res)}")
        else:
            print(f"    ❌ {kw}: 抓不到")
        time.sleep(8)          # 遵守限速
    print(f"\n完成，存到 {OUT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    run()
