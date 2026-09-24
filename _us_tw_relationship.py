"""分析工具：每支台股各自被哪些美股訊號影響、影響多深。
美股訊號(隔夜) → 台股當日，逐股算相關性與敏感度(beta)，並比較「全期 vs 近60日」看是否脫鉤。
純觀察分析，不進模型。"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, yfinance as yf
import config as cfg

US = {"費半SOX": "^SOX", "Nasdaq": "^IXIC", "ON功率": "ON", "Vishay": "VSH"}
PASSIVE = {"2327", "2492", "3026", "2472", "3357"}


def _norm(s):
    s = pd.to_datetime(s)
    try:
        s = s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return s.dt.normalize().astype("datetime64[ns]")


# 抓各美股訊號隔夜漲跌
usdfs = {}
for nm, sym in US.items():
    c = yf.download(sym, start="2023-01-01", progress=False)["Close"].squeeze()
    d = pd.DataFrame({"us_date": c.pct_change().index, nm: (c.pct_change() * 100).values}).dropna()
    d["us_date"] = _norm(d["us_date"])
    usdfs[nm] = d.sort_values("us_date")
print(f"美股訊號：{list(US.keys())}\n")


def aligned(tk):
    df = pd.read_csv(f"data/raw/{tk}.csv", index_col=0, parse_dates=True)
    tw = pd.DataFrame({"tw_date": _norm(pd.Series(df.index)),
                       "tw_pct": df["Close"].pct_change().values * 100}).dropna().sort_values("tw_date")
    for nm, d in usdfs.items():
        tw = pd.merge_asof(tw, d, left_on="tw_date", right_on="us_date",
                           direction="backward", allow_exact_matches=False).drop(columns="us_date")
    return tw.dropna()


print("=== 逐股：與各美股訊號的相關性（隔夜美股→台股當日）===")
hdr = f"{'股':<8}" + "".join(f"{nm:>9}" for nm in US) + f"{'最強連動':>11}{'敏感度β':>9}"
print(hdr); print("-" * len(hdr))
recent_decouple = []
for tk in cfg.ALL_STOCKS:
    m = aligned(tk)
    corrs = {nm: np.corrcoef(m[nm], m["tw_pct"])[0, 1] for nm in US}
    best = max(corrs, key=corrs.get)
    beta = np.polyfit(m[best], m["tw_pct"], 1)[0]   # 台股對最強訊號的敏感度
    nm_tw = cfg.ALL_STOCKS.get(tk, "")
    seg = "被動" if tk.replace(".TWO", "").replace(".TW", "") in PASSIVE else "功率"
    row = f"{nm_tw:<7}" + "".join(f"{corrs[nm]:>9.2f}" for nm in US) + f"{best:>11}{beta:>9.2f}"
    print(row)
    # 近60日 vs 全期，看脫鉤
    r = m.tail(60)
    c_all = np.corrcoef(m[best], m["tw_pct"])[0, 1]
    c_rec = np.corrcoef(r[best], r["tw_pct"])[0, 1]
    recent_decouple.append((nm_tw, seg, c_all, c_rec))

print("\n=== 是否脫鉤：最強訊號相關性 全期 vs 近60日 ===")
print(f"{'股':<8}{'類':>5}{'全期相關':>9}{'近60日':>9}  狀態")
print("-" * 40)
for nm_tw, seg, c_all, c_rec in recent_decouple:
    tag = "🔴脫鉤" if c_rec < c_all - 0.15 else ("🟡轉弱" if c_rec < c_all - 0.05 else "✅穩定")
    print(f"{nm_tw:<7}{seg:>5}{c_all:>9.2f}{c_rec:>9.2f}  {tag}")
