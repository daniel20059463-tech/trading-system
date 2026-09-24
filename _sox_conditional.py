"""條件分析：美股(費半SOX)當天漲跌「之後」台股隔天怎麼走。
不做黑箱特徵，直接看條件關係——尤其極端日(美股大漲/大跌)台股的反應，
看訊號是否藏在尾部。並檢查最近的崩盤日美股是否領先。"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, yfinance as yf
import config as cfg


def _norm(s):
    s = pd.to_datetime(s)
    try:
        s = s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return s.dt.normalize().astype("datetime64[ns]")


_sx = yf.download("^SOX", start="2023-01-01", progress=False)["Close"].squeeze()
_sxdf = pd.DataFrame({"us_date": _sx.pct_change().index, "sox": (_sx.pct_change() * 100).values}).dropna()
_sxdf["us_date"] = _norm(_sxdf["us_date"])
_sxdf = _sxdf.sort_values("us_date")

# 匯整所有台股：每筆 (台股當日漲跌, 對應隔夜SOX)
rows = []
for tk in cfg.ALL_STOCKS:
    df = pd.read_csv(f"data/raw/{tk}.csv", index_col=0, parse_dates=True)
    tw = pd.DataFrame({"tw_date": _norm(pd.Series(df.index)),
                       "tw_pct": df["Close"].pct_change().values * 100}).dropna()
    m = pd.merge_asof(tw.sort_values("tw_date"), _sxdf, left_on="tw_date", right_on="us_date",
                      direction="backward", allow_exact_matches=False).dropna()
    m["ticker"] = tk
    rows.append(m)
allm = pd.concat(rows, ignore_index=True)

# 分桶：美股隔夜漲跌
bins = [-100, -2, -1, -0.5, 0.5, 1, 2, 100]
labels = ["大跌<-2%", "跌-2~-1%", "微跌-1~-0.5%", "持平±0.5%", "微漲0.5~1%", "漲1~2%", "大漲>2%"]
allm["bucket"] = pd.cut(allm["sox"], bins=bins, labels=labels)

print("=== 美股隔夜漲跌 → 台股隔天反應（全10股匯整）===")
print(f"{'美股隔夜':<14}{'樣本':>6}{'台股隔天均漲跌':>13}{'台股隔天上漲比例':>15}")
print("-" * 50)
for lab in labels:
    g = allm[allm["bucket"] == lab]
    if len(g) == 0:
        continue
    print(f"{lab:<13}{len(g):>6}{g['tw_pct'].mean():>+12.2f}%{(g['tw_pct']>0).mean()*100:>14.0f}%")

# 尾部對比：美股大跌 vs 大漲，台股隔天跟隨度
big_dn = allm[allm["sox"] < -2]
big_up = allm[allm["sox"] > 2]
print("-" * 50)
print(f"尾部：美股大跌(<-2%)後 台股隔天下跌比例 {(big_dn['tw_pct']<0).mean()*100:.0f}%（均 {big_dn['tw_pct'].mean():+.2f}%）")
print(f"尾部：美股大漲(>2%)後 台股隔天上漲比例 {(big_up['tw_pct']>0).mean()*100:.0f}%（均 {big_up['tw_pct'].mean():+.2f}%）")

# 最近交易日：美股隔夜 vs 台股當日（看美股是否領先崩盤）
print("\n=== 最近交易日：美股隔夜 → 台股當日（均10股）===")
print(f"{'台股日':<12}{'前晚美股SOX':>12}{'台股當日均':>12}")
print("-" * 38)
recent = allm.groupby("tw_date").agg(sox=("sox", "first"), tw=("tw_pct", "mean")).reset_index()
for r in recent.sort_values("tw_date").tail(6).itertuples():
    print(f"{str(r.tw_date.date()):<12}{r.sox:>+11.2f}%{r.tw:>+11.2f}%")
