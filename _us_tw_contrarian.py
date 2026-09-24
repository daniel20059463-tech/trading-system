"""逆勢分析（近2年）：美股(費半SOX)收盤後，這些台股是跟著走還是逆勢走？
條件在美股隔夜漲/跌，逐股算：跟隨機率 vs 逆勢機率 + 逆勢時的平均幅度。
回答：哪幾支有明顯逆勢傾向（美股跌它還漲 / 美股漲它反跌）。"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, yfinance as yf
import config as cfg

PASSIVE = {"2327", "2492", "3026", "2472", "3357"}
START = (pd.Timestamp.today() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")


def _norm(s):
    s = pd.to_datetime(s)
    try:
        s = s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return s.dt.normalize().astype("datetime64[ns]")


_sx = yf.download("^SOX", start="2023-06-01", progress=False)["Close"].squeeze()
_sxdf = pd.DataFrame({"us_date": _sx.pct_change().index, "sox": (_sx.pct_change() * 100).values}).dropna()
_sxdf["us_date"] = _norm(_sxdf["us_date"]); _sxdf = _sxdf.sort_values("us_date")
print(f"條件訊號：費半SOX 隔夜｜分析期間：近2年（{START} 起）\n")


def aligned(tk):
    df = pd.read_csv(f"data/raw/{tk}.csv", index_col=0, parse_dates=True)
    tw = pd.DataFrame({"tw_date": _norm(pd.Series(df.index)),
                       "tw_pct": df["Close"].pct_change().values * 100}).dropna().sort_values("tw_date")
    m = pd.merge_asof(tw, _sxdf, left_on="tw_date", right_on="us_date",
                      direction="backward", allow_exact_matches=False).dropna()
    return m[m["tw_date"] >= START]


print("=== 美股漲/跌「之後」台股怎麼走（近2年，條件SOX隔夜）===")
print(f"{'股':<7}{'類':>4}│{'美股漲日→台股':>16}│{'美股跌日→台股':>16}│{'逆勢傾向':>9}")
print(f"{'':<7}{'':>4}│{'跟漲':>8}{'逆勢跌':>8}│{'跟跌':>8}{'逆勢漲':>8}│")
print("-" * 62)
ranks = []
for tk in cfg.ALL_STOCKS:
    m = aligned(tk)
    up = m[m["sox"] > 0]; dn = m[m["sox"] < 0]
    follow_up = (up["tw_pct"] > 0).mean() * 100          # 美漲→台漲（跟）
    contra_dn = (up["tw_pct"] < 0).mean() * 100          # 美漲→台跌（逆勢走弱）
    follow_dn = (dn["tw_pct"] < 0).mean() * 100          # 美跌→台跌（跟）
    contra_up = (dn["tw_pct"] > 0).mean() * 100          # 美跌→台漲（逆勢抗跌）
    contra_score = (contra_dn + contra_up) / 2            # 整體逆勢傾向
    nm = cfg.ALL_STOCKS.get(tk, ""); seg = "被動" if tk.replace(".TWO","").replace(".TW","") in PASSIVE else "功率"
    print(f"{nm:<6}{seg:>4}│{follow_up:>7.0f}%{contra_dn:>7.0f}%│{follow_dn:>7.0f}%{contra_up:>7.0f}%│{contra_score:>8.0f}%")
    ranks.append((nm, seg, contra_up, contra_dn, contra_score,
                  dn[dn["tw_pct"] > 0]["tw_pct"].mean(), up[up["tw_pct"] < 0]["tw_pct"].mean()))

print("-" * 62)
print("\n=== 逆勢「抗跌」最強（美股跌、它反而漲的機率）===")
for nm, seg, cu, cd, cs, mu, md in sorted(ranks, key=lambda x: -x[2])[:4]:
    print(f"  {nm}({seg})：美股跌時 {cu:.0f}% 仍上漲，逆勢上漲均 +{mu:.2f}%")
print("\n=== 逆勢「走弱」最強（美股漲、它反而跌的機率）===")
for nm, seg, cu, cd, cs, mu, md in sorted(ranks, key=lambda x: -x[3])[:4]:
    print(f"  {nm}({seg})：美股漲時 {cd:.0f}% 反而下跌，逆勢下跌均 {md:.2f}%")
