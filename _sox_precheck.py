"""前置檢查：隔夜美股(費半SOX)漲跌 與 台股當日漲跌 的相關性。
對齊：台股第 T 日 ← 最近一個「早於 T」的美股收盤(隔夜)。
相關性夠才值得做完整訓練實驗。"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, yfinance as yf
import config as cfg

# 1. 抓美股半導體指數（^SOX 費半；備援 SOXX / SMH）
sox = None
for sym in ["^SOX", "SOXX", "SMH"]:
    try:
        df = yf.download(sym, start="2023-01-01", progress=False)
        if len(df) > 100:
            sox = df["Close"].copy(); used = sym; break
    except Exception as e:
        print(f"{sym} 失敗: {e}")
if sox is None:
    print("❌ 美股半導體指數都抓不到"); sys.exit(1)
sox = sox.squeeze()
sox_pct = sox.pct_change() * 100
def _norm(s):
    s = pd.to_datetime(s)
    try:
        s = s.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return s.dt.normalize().astype("datetime64[ns]")

sox_df = pd.DataFrame({"us_date": sox_pct.index, "sox_overnight": sox_pct.values}).dropna()
sox_df["us_date"] = _norm(sox_df["us_date"])
print(f"美股來源: {used}，{len(sox_df)} 個交易日，{sox_df['us_date'].min().date()}~{sox_df['us_date'].max().date()}")

# 2. 對每支台股：對齊隔夜SOX → 算台股當日漲跌的相關性
print(f"\n{'股':<8}{'相關係數':>9}{'大跌日命中':>12}")
print("-" * 32)
corrs = []
big_hit = []   # 美股隔夜跌時，台股當日也跌的比例
for tk in cfg.ALL_STOCKS:
    df = pd.read_csv(f"data/raw/{tk}.csv", index_col=0, parse_dates=True)
    tw = pd.DataFrame({"tw_date": df.index, "tw_pct": df["Close"].pct_change().values * 100}).dropna()
    tw["tw_date"] = _norm(tw["tw_date"])
    # merge_asof：台股T日 ← 嚴格早於T的最近美股日
    m = pd.merge_asof(tw.sort_values("tw_date"), sox_df.sort_values("us_date"),
                      left_on="tw_date", right_on="us_date",
                      direction="backward", allow_exact_matches=False).dropna()
    c = np.corrcoef(m["sox_overnight"], m["tw_pct"])[0, 1]
    corrs.append(c)
    # 美股隔夜大跌(<-1.5%)時，台股當日跌的比例
    big = m[m["sox_overnight"] < -1.5]
    hit = (big["tw_pct"] < 0).mean() * 100 if len(big) else np.nan
    big_hit.append(hit)
    nm = cfg.ALL_STOCKS.get(tk, "")
    print(f"{nm:<7}{c:>9.2f}{hit:>11.0f}%")
print("-" * 32)
print(f"{'平均':<8}{np.nanmean(corrs):>9.2f}{np.nanmean(big_hit):>11.0f}%")
print()
mc = np.nanmean(corrs)
print(f"判讀：平均相關 {mc:.2f}")
if mc >= 0.25:
    print("  → 相關夠強，值得做完整訓練實驗 ✅")
elif mc >= 0.12:
    print("  → 弱相關，可能有點用也可能失敗，值得一試 ⚪")
else:
    print("  → 幾乎無相關，大概率失敗，先別浪費訓練時間 ❌")
print(f"  美股隔夜大跌時台股跟跌比例 {np.nanmean(big_hit):.0f}%（>65% 代表能抓 risk-off 日）")
