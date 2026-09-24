"""一次性：新量尺(holdout)誠實基準 vs 舊量尺(full)對照。"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import config as cfg
from backtest import _predict_testset_ensemble
from eval_harness import _direction_accuracy

hdr_a, hdr_b, hdr_c, hdr_d = "股", "舊量尺(full)", "新量尺(holdout)", "樣本"
print(f"{hdr_a:<8}{hdr_b:>11}{hdr_c:>14}{hdr_d:>6}")
print("-" * 42)
olds, news = [], []
for tk in cfg.ALL_STOCKS:
    try:
        f = _predict_testset_ensemble(tk, segment="full")
        h = _predict_testset_ensemble(tk, segment="holdout")
        af, ah = _direction_accuracy(f), _direction_accuracy(h)
        olds.append(af); news.append(ah)
        nm = cfg.ALL_STOCKS.get(tk, "")
        print(f"{nm:<7}{af:>10.1f}%{ah:>13.1f}%{len(h):>6}")
    except Exception as e:
        print(f"{tk}: {str(e)[:40]}")
print("-" * 42)
lab = "平均"
print(f"{lab:<7}{np.mean(olds):>10.1f}%{np.mean(news):>13.1f}%")
