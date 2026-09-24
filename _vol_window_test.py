"""實測：σ 視窗 5日 vs 10日 vs 20日 對區間的影響。
同一批模型預測（qs 不變），只換 σ 視窗並各自重新校準 conformal Q（公平比較）。
指標：整體覆蓋率 / 平均寬度 / 暴衝日(|實際|>5%)覆蓋率（σ5 的死穴檢查）。
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import torch
import config as cfg
import quantile_forecast as qf

WINDOWS = [5, 10, 20]
ALPHA = 0.2          # 80% 區間
LO_Q, HI_Q = 0.1, 0.9
lo_i, hi_i = qf.Q_IDX[LO_Q], qf.Q_IDX[HI_Q]

# 1) 每支股票：跑一次模型，快取 (qs, y, 各視窗σ)
cache = {}
for tk in cfg.ALL_STOCKS:
    try:
        model, scaler, ckpt = qf._load_quantile(tk)
        df = qf._load_feature_df(tk, ckpt["feature_names"])
        ds = scaler.transform(df.values.astype(np.float32))
        w = cfg.WINDOW_SIZE
        split = int(len(ds) * cfg.TRAIN_RATIO)
        end = len(ds) - w
        sigmas = {}
        for vw in WINDOWS:
            s = (df["pct_change"].rolling(vw, min_periods=3).std()
                 .bfill().clip(lower=qf.VOL_FLOOR).values)
            sigmas[vw] = s
        recs = []
        for i in range(split, end):
            qs = qf._predict_scaled_window(model, scaler, ckpt, ds[i:i+w])
            y = float(df["pct_change"].iloc[i + w])
            recs.append((qs, y, {vw: sigmas[vw][i + w] for vw in WINDOWS}))
        cache[tk] = recs
    except Exception as e:
        print(f"[跳過] {tk}: {str(e)[:50]}")

# 2) 各視窗：cal 校準 Q → test 段量 覆蓋/寬度/暴衝日覆蓋
print(f"{'σ視窗':<8}{'覆蓋率':>8}{'平均寬':>8}{'最窄寬':>8}{'暴衝日覆蓋':>10}{'暴衝樣本':>8}")
print("-" * 52)
for vw in WINDOWS:
    covs, widths, minws, big_in, big_n = [], [], [], 0, 0
    for tk, recs in cache.items():
        n = len(recs)
        cal, test = recs[: n // 2], recs[n // 2:]
        scores = sorted(max(qs[lo_i] - y, y - qs[hi_i]) / s[vw] for qs, y, s in cal)
        m = len(scores)
        rank = min(max(int(np.ceil((m + 1) * (1 - ALPHA))), 1), m)
        Q = scores[rank - 1]
        tw = []
        for qs, y, s in test:
            lo = qs[lo_i] - Q * s[vw]
            hi = qs[hi_i] + Q * s[vw]
            # 與生產一致：夾在 ±10%
            lo, hi = max(lo, -10.0), min(hi, 10.0)
            inside = lo <= y <= hi
            covs.append(inside)
            tw.append(hi - lo)
            if abs(y) > 5:
                big_n += 1
                big_in += inside
        widths += tw
        minws.append(min(tw))
    print(f"{vw}日{'':<5}{np.mean(covs)*100:>7.0f}%{np.mean(widths):>7.1f}%"
          f"{np.mean(minws):>7.1f}%{big_in/big_n*100 if big_n else float('nan'):>9.0f}%{big_n:>8}")
print("\n判讀：覆蓋率越接近80%越誠實；平均寬越小越好；")
print("但「暴衝日覆蓋」若明顯掉，代表縮窄是用『漏接大行情』換來的假緊。")
