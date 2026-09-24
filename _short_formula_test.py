"""驗證兩種「短線區間」公式的覆蓋率與縮放效果：
A) 加法（現行）：raw ± Q·σ5 —— 已知縮不進原始分位數寬度
B) 乘法+保形：繞 q50 按 ρ=σ5/σ20 縮放原始寬度，再 ± Q·σ5 校準
指標：覆蓋率 / 平均寬 / 「冷卻日(ρ<0.7)」平均寬（短線的存在意義就在這裡）。
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import config as cfg
import quantile_forecast as qf

lo_i, hi_i, mid_i = qf.Q_IDX[0.1], qf.Q_IDX[0.9], qf.Q_IDX[0.5]
ALPHA = 0.2

cache = {}
for tk in cfg.ALL_STOCKS:
    try:
        model, scaler, ckpt = qf._load_quantile(tk)
        df = qf._load_feature_df(tk, ckpt["feature_names"])
        ds = scaler.transform(df.values.astype(np.float32))
        w = cfg.WINDOW_SIZE
        split = int(len(ds) * cfg.TRAIN_RATIO)
        end = len(ds) - w
        s5 = qf._vol_series(df, 5).values
        s20 = qf._vol_series(df, 20).values
        recs = []
        for i in range(split, end):
            qs = qf._predict_scaled_window(model, scaler, ckpt, ds[i:i+w])
            y = float(df["pct_change"].iloc[i + w])
            recs.append((qs, y, s5[i + w], s20[i + w]))
        cache[tk] = recs
    except Exception as e:
        print(f"[跳過] {tk}: {str(e)[:50]}")


def bounds_A(qs, s5):        # 加法（未含Q項，Q校準時加）
    return qs[lo_i], qs[hi_i]

def bounds_B(qs, s5, s20):   # 乘法：繞 q50 按 ρ 縮放
    rho = min(max(s5 / s20, 0.3), 2.0)
    mid = qs[mid_i]
    return mid + (qs[lo_i] - mid) * rho, mid + (qs[hi_i] - mid) * rho


for label in ["A加法(現行)", "B乘法縮放"]:
    covs, widths, calm_w, calm_n = [], [], [], 0
    for tk, recs in cache.items():
        n = len(recs)
        cal, test = recs[: n // 2], recs[n // 2:]
        scores = []
        for qs, y, s5, s20 in cal:
            lo, hi = bounds_A(qs, s5) if label.startswith("A") else bounds_B(qs, s5, s20)
            scores.append(max(lo - y, y - hi) / s5)
        scores = sorted(scores)
        m = len(scores)
        rank = min(max(int(np.ceil((m + 1) * (1 - ALPHA))), 1), m)
        Q = scores[rank - 1]
        for qs, y, s5, s20 in test:
            lo, hi = bounds_A(qs, s5) if label.startswith("A") else bounds_B(qs, s5, s20)
            lo, hi = max(lo - Q * s5, -10.0), min(hi + Q * s5, 10.0)
            covs.append(lo <= y <= hi)
            wd = hi - lo
            widths.append(wd)
            if s5 / s20 < 0.7:
                calm_w.append(wd); calm_n += 1
    print(f"{label}: 覆蓋率 {np.mean(covs)*100:.0f}%  平均寬 {np.mean(widths):.1f}%  "
          f"冷卻日平均寬 {np.mean(calm_w):.1f}%（{calm_n}天）")
print("\n判讀：B 若覆蓋率仍~72% 且冷卻日明顯更窄 → 值得換；否則維持 A 並如實告知差異小。")
