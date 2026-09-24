"""美股 → 台股 連動/逆勢 觀察報告（近2年）。
條件在美股(費半SOX)隔夜漲跌，逐股分析：跟隨 vs 逆勢、逆勢幅度、近期是否脫鉤。
輸出 Markdown 報告到 Obsidian（WIKI_DIR/us_tw_observe.md），可手動或排程定期更新。

用法：python us_tw_observe.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, yfinance as yf
import config as cfg

PASSIVE = {"2327", "2492", "3026", "2472", "3357"}
REPORT = os.path.join(cfg.WIKI_DIR, "us_tw_observe.md")


def _norm(s):
    s = pd.to_datetime(s)
    try:
        s = s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return s.dt.normalize().astype("datetime64[ns]")


def _sox():
    c = yf.download("^SOX", start="2023-06-01", progress=False)["Close"].squeeze()
    d = pd.DataFrame({"us_date": c.pct_change().index, "sox": (c.pct_change() * 100).values}).dropna()
    d["us_date"] = _norm(d["us_date"])
    return d.sort_values("us_date")


def _aligned(tk, sxdf):
    df = pd.read_csv(os.path.join(cfg.RAW_DATA_DIR, f"{tk}.csv"), index_col=0, parse_dates=True)
    tw = pd.DataFrame({"tw_date": _norm(pd.Series(df.index)),
                       "tw_pct": df["Close"].pct_change().values * 100}).dropna().sort_values("tw_date")
    return pd.merge_asof(tw, sxdf, left_on="tw_date", right_on="us_date",
                         direction="backward", allow_exact_matches=False).dropna()


def build_report() -> str:
    sxdf = _sox()
    start2y = (pd.Timestamp.today() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    rows, decouple = [], []
    for tk in cfg.ALL_STOCKS:
        full = _aligned(tk, sxdf)
        m = full[full["tw_date"] >= start2y]
        up, dn = m[m["sox"] > 0], m[m["sox"] < 0]
        if len(up) < 20 or len(dn) < 20:
            continue
        nm = cfg.ALL_STOCKS.get(tk, "")
        seg = "被動" if tk.replace(".TWO", "").replace(".TW", "") in PASSIVE else "功率"
        rows.append({
            "nm": nm, "seg": seg,
            "follow_up": (up["tw_pct"] > 0).mean() * 100,
            "contra_dn": (up["tw_pct"] < 0).mean() * 100,
            "follow_dn": (dn["tw_pct"] < 0).mean() * 100,
            "contra_up": (dn["tw_pct"] > 0).mean() * 100,
            "up_mag": dn[dn["tw_pct"] > 0]["tw_pct"].mean(),
            "dn_mag": up[up["tw_pct"] < 0]["tw_pct"].mean(),
            "follow": ((up["tw_pct"] > 0).sum() + (dn["tw_pct"] < 0).sum()) / (len(up) + len(dn)) * 100,
        })
        c_all = np.corrcoef(full["sox"], full["tw_pct"])[0, 1]
        c_rec = np.corrcoef(full.tail(60)["sox"], full.tail(60)["tw_pct"])[0, 1]
        decouple.append((nm, seg, c_all, c_rec))

    L = [f"# 美股 → 台股 連動／逆勢 觀察報告",
         f"\n更新：{pd.Timestamp.today().strftime('%Y-%m-%d %H:%M')}｜條件訊號：費半 SOX 隔夜｜期間：近 2 年\n",
         "> 看「跟隨率」與「逆勢率」。跟隨率越接近 50% 代表越不甩美股、越獨立。\n",
         "## 1. 美股漲/跌「之後」台股怎麼走\n",
         "| 股 | 類 | 美漲→跟漲 | 美漲→逆勢跌 | 美跌→跟跌 | 美跌→逆勢漲 | 整體跟隨率 |",
         "|----|----|----------|------------|----------|------------|-----------|"]
    for r in rows:
        L.append(f"| {r['nm']} | {r['seg']} | {r['follow_up']:.0f}% | {r['contra_dn']:.0f}% "
                 f"| {r['follow_dn']:.0f}% | {r['contra_up']:.0f}% | {r['follow']:.0f}% |")

    abest = sorted(rows, key=lambda x: -x["contra_up"])[:3]
    wbest = sorted(rows, key=lambda x: -x["contra_dn"])[:3]
    L += ["\n## 2. 逆勢領先股\n",
          "**🔵 逆勢抗跌（美股跌、它反而漲）：**"]
    for r in abest:
        L.append(f"- **{r['nm']}**：美股跌時 {r['contra_up']:.0f}% 仍上漲，逆勢漲均 +{r['up_mag']:.2f}%")
    L.append("\n**🔴 逆勢走弱（美股漲、它反而跌）：**")
    for r in wbest:
        L.append(f"- **{r['nm']}**：美股漲時 {r['contra_dn']:.0f}% 反而跌，逆勢跌均 {r['dn_mag']:.2f}%")

    L += ["\n## 3. 脫鉤監控（最強訊號相關性：全期 vs 近60日）\n",
          "| 股 | 類 | 全期相關 | 近60日 | 狀態 |",
          "|----|----|---------|--------|------|"]
    for nm, seg, c_all, c_rec in decouple:
        tag = "🔴 脫鉤" if c_rec < c_all - 0.15 else ("🟡 轉弱" if c_rec < c_all - 0.05 else "✅ 穩定")
        L.append(f"| {nm} | {seg} | {c_all:.2f} | {c_rec:.2f} | {tag} |")

    L += ["\n## 4. 怎麼用",
          "- **跟隨率僅 ~55%**：美股方向對這幾支是「弱訊號」，不能單靠美股推台股。",
          "- **🔴脫鉤** 的股票：當下被本土外資籌碼主導，**美股參考無效**，看籌碼。",
          "- **逆勢抗跌股**：大盤/美股殺時相對抗跌；**逆勢走弱股**：美股漲也未必跟上。",
          "- 「脫鉤」本身是警訊：平常跟美股的突然不跟，代表外資正在主導。",
          "\n_純觀察分析，非投資建議。美股訊號已證實不適合當 ML 特徵（被同業稀釋），故獨立為人看的觀察報告。_"]
    return "\n".join(L)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rep = build_report()
    os.makedirs(cfg.WIKI_DIR, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(rep)
    print(f"✅ 已更新觀察報告：{REPORT}")
    print(f"   共 {rep.count(chr(10))} 行")


if __name__ == "__main__":
    main()
