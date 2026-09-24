"""驗證用戶觀察：每日 80% 區間寬度是否越來越大？來源是什麼？
1) 從每日 Wiki 的「隔日區間預測」表解析歷史區間寬度（當日實際發布的數字）
2) 對照 20 日波動率 σ 的走勢（CQR 的寬度引擎）
"""
import sys, os, re, glob, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import config as cfg

# 1) 解析每日 wiki 的區間表：| ticker | name | p_up | q50 | a~b | c~d |
pat = re.compile(r"\|\s*(\d{4}\.\w+)\s*\|[^|]+\|[^|]+\|[^|]+\|\s*([\d.]+)~([\d.]+)\s*\|\s*([\d.]+)~([\d.]+)\s*\|")
rows = []
for f in sorted(glob.glob(os.path.join(cfg.WIKI_DIR, "2026-*.md"))):
    d = os.path.basename(f)[:10]
    txt = open(f, encoding="utf-8", errors="ignore").read()
    if "隔日區間預測" not in txt:
        continue
    w80s, w50s = [], []
    for m in pat.finditer(txt):
        lo50, hi50, lo80, hi80 = map(float, m.groups()[1:])
        mid = (lo80 + hi80) / 2
        w80s.append((hi80 - lo80) / mid * 100)
        w50s.append((hi50 - lo50) / mid * 100)
    if w80s:
        rows.append((d, np.mean(w50s), np.mean(w80s), len(w80s)))

print("每日「實際發布」的平均區間寬度（從 Wiki 解析）：")
print(f"{'日期':<12}{'50%寬':>8}{'80%寬':>8}{'支數':>5}")
print("-" * 35)
for d, w50, w80, n in rows:
    bar = "█" * int(w80 / 1.5)
    print(f"{d:<12}{w50:>7.1f}%{w80:>7.1f}%{n:>5}  {bar}")

# 2) σ20（CQR 寬度引擎）近 20 個交易日的平均走勢
print("\n10 股平均 20 日波動率 σ（區間寬度的引擎）：")
sig_all = {}
for tk in cfg.ALL_STOCKS:
    df = pd.read_csv(f"data/raw/{tk}.csv", index_col=0, parse_dates=True)
    sig = (df["Close"].pct_change() * 100).rolling(20, min_periods=5).std()
    for dt, v in sig.tail(22).items():
        sig_all.setdefault(str(dt.date()), []).append(v)
print(f"{'日期':<12}{'σ20':>7}")
print("-" * 22)
for d in sorted(sig_all):
    m = np.nanmean(sig_all[d])
    print(f"{d:<12}{m:>6.2f}%  {'█' * int(m * 4)}")
