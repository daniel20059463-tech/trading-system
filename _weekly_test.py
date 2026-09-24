"""驗證「一週(5交易日)展望」產品可不可以誠實端出：
中心 = 日中位數×5（漂移持續假設）；寬度 = 日分位數半寬×√5，再保形校準(σ20·√5)。
指標：週區間覆蓋率 / 平均寬 / 週方向準確率（若≈50%就老實說方向不可信）。
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import config as cfg
import quantile_forecast as qf

lo_i, hi_i, mid_i = qf.Q_IDX[0.1], qf.Q_IDX[0.9], qf.Q_IDX[0.5]
H = 5          # 一週 = 5 交易日
ALPHA = 0.2
SQ = np.sqrt(H)

covs, widths, dir_ok, dir_n = [], [], 0, 0
base_up = []
for tk in cfg.ALL_STOCKS:
    try:
        model, scaler, ckpt = qf._load_quantile(tk)
        df = qf._load_feature_df(tk, ckpt["feature_names"])
        ds = scaler.transform(df.values.astype(np.float32))
        w = cfg.WINDOW_SIZE
        split = int(len(ds) * cfg.TRAIN_RATIO)
        end = len(ds) - w - H            # 要能看到未來5日
        s20 = qf._vol_series(df, 20).values
        pct = df["pct_change"].values
        recs = []
        for i in range(split, end):
            qs = qf._predict_scaled_window(model, scaler, ckpt, ds[i:i+w])
            y5 = float(np.sum(pct[i + w: i + w + H]))     # 未來5日累積
            recs.append((qs, y5, s20[i + w]))
        n = len(recs)
        cal, test = recs[: n // 2], recs[n // 2:]
        scores = []
        for qs, y5, s in cal:
            c = qs[mid_i] * H
            lo = c + (qs[lo_i] - qs[mid_i]) * SQ
            hi = c + (qs[hi_i] - qs[mid_i]) * SQ
            scores.append(max(lo - y5, y5 - hi) / (s * SQ))
        scores = sorted(scores)
        m = len(scores)
        rank = min(max(int(np.ceil((m + 1) * (1 - ALPHA))), 1), m)
        Q = scores[rank - 1]
        for qs, y5, s in test:
            c = qs[mid_i] * H
            lo = c + (qs[lo_i] - qs[mid_i]) * SQ - Q * s * SQ
            hi = c + (qs[hi_i] - qs[mid_i]) * SQ + Q * s * SQ
            covs.append(lo <= y5 <= hi)
            widths.append(hi - lo)
            if abs(c) > 0.5:              # 中心有明確方向才算方向判斷
                dir_n += 1
                dir_ok += (c > 0) == (y5 > 0)
            base_up.append(y5 > 0)
    except Exception as e:
        print(f"[跳過] {tk}: {str(e)[:50]}")

print(f"週區間 覆蓋率 {np.mean(covs)*100:.0f}%（目標80%）  平均寬 {np.mean(widths):.1f}%")
print(f"週方向準確率（中心|漂移|>0.5%時）: {dir_ok}/{dir_n} = {dir_ok/dir_n*100 if dir_n else 0:.0f}%")
print(f"週基準上漲率: {np.mean(base_up)*100:.0f}%")
print("\n判讀：覆蓋~80%→區間可端出；方向≈基準→老實標示「方向僅供參考」")
