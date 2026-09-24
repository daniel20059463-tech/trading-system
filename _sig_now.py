import sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, numpy as np, config as cfg

print("今天的 σ5 vs σ20（哪個大，換視窗區間就往哪邊走）：")
h, a, b = "股", "σ5(近5日)", "σ20(近20日)"
print(f"{h:<8}{a:>10}{b:>11}")
print("-" * 32)
w5s, w20s = [], []
for tk in cfg.ALL_STOCKS:
    df = pd.read_csv(f"data/raw/{tk}.csv", index_col=0, parse_dates=True)
    r = df["Close"].pct_change() * 100
    s5, s20 = r.tail(5).std(), r.tail(20).std()
    w5s.append(s5); w20s.append(s20)
    print(f"{cfg.ALL_STOCKS.get(tk,''):<7}{s5:>9.2f}%{s20:>10.2f}%")
print("-" * 32)
m = "平均"
print(f"{m:<7}{np.mean(w5s):>9.2f}%{np.mean(w20s):>10.2f}%")
