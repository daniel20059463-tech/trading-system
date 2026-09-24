"""
分位數迴歸（Quantile Regression）：預測隔日漲跌幅的「信賴區間」，而非單一點。

數學原理 — Pinball Loss（分位數損失）：
  對分位數 q（如 0.1, 0.5, 0.9），令誤差 e = y - ŷ：
      L_q(e) = max( q·e , (q-1)·e )
  • 當 q=0.5 時退化為 MAE（預測中位數）。
  • 當 q=0.9 時，低估（e>0）罰 0.9、高估（e<0）罰 0.1
    → 逼模型把 ŷ 往高推，使約 90% 實際值落在預測之下。
  同時學 q10/q50/q90 三條線，即得到「80% 信賴區間 [q10, q90]」。

數學原理 2 — 波動率正規化保形分位數迴歸（Normalized CQR, Romano 2019）：
  純分位數迴歸的區間在「樣本外」常失準（本專案實測覆蓋率僅 ~61%，遠低於 80%）。
  金融資料波動率會隨時間漂移（volatility clustering），固定寬度修正補不夠，
  故用「當期波動率 σ」縮放修正量，讓區間隨波動自適應：
      1. 令 σ_i = 第 i 日近 20 日報酬標準差（波動率尺度）
      2. 校準集每筆算正規化保形分數 E_i = max( q_lo − y_i , y_i − q_hi ) / σ_i
      3. 取 {E_i} 的 ceil((n+1)(1−α))/n 經驗分位數 Q（無單位倍數）
      4. 預測時修正區間 = [ q_lo − Q·σ_now , q_hi + Q·σ_now ]
  當前波動大 → σ_now 大 → 區間自動撐寬；波動小 → 區間收窄，使覆蓋率動態校準到目標。

用途：
  輸出「明天國巨大概率落在 -1.2% ~ +3.1%（80% 信賴區間），中位數 +0.9%」，
  讓策略依「區間寬窄」判斷把握度：區間窄=高把握，區間寬=高不確定性宜降倉。

⚠️ 區間反映模型對「歷史波動模式」的不確定性，非保證；黑天鵝事件仍可能突破。

用法：
  python quantile_forecast.py --train            # 訓練全部股票分位數模型
  python quantile_forecast.py --train --ticker 2327.TW
  python quantile_forecast.py                     # 用現有模型輸出今日區間預測
"""
import sys
import os
import argparse
import pickle
from datetime import date

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from models.lstm_model import get_model
from models.train import prepare_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)   # 80% 風險區間 + 50% 最可能區間 + 中位數
Q_IDX = {q: i for i, q in enumerate(QUANTILES)}   # 分位數 -> 輸出索引
MODEL_TYPE = "lstm"                # 分位數用 LSTM backbone（區間預測不需 ensemble 複雜度）


# ── Pinball Loss ──────────────────────────────────────────────────────────────

class QuantileLoss(nn.Module):
    """多分位數 Pinball Loss。pred shape (batch, n_quantiles)，target shape (batch, 1)。"""

    def __init__(self, quantiles=QUANTILES):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, pred, target):
        losses = []
        for i, q in enumerate(self.quantiles):
            e = target[:, 0] - pred[:, i]
            losses.append(torch.max(q * e, (q - 1) * e))
        return torch.stack(losses, dim=1).mean()


# ── 訓練 ──────────────────────────────────────────────────────────────────────

def train_quantile(ticker: str) -> dict:
    """訓練單支股票的分位數模型，輸出 q10/q50/q90，存成 {ticker}_quantile_best.pt。"""
    print("注意：目前驗證段重複用於選模與校準，以下涵蓋率僅為診斷，不可據此晉升。")
    name = cfg.ALL_STOCKS.get(ticker, "")
    print(f"\n{'='*60}")
    print(f"[分位數訓練] {ticker} {name} | 分位數 {QUANTILES}")
    print(f"{'='*60}")

    X_train, y_train, X_test, y_test, scaler, feature_names, target_col = prepare_dataset(ticker)
    input_size = X_train.shape[2]
    print(f"  特徵數：{input_size} | 訓練樣本：{len(X_train)} | 測試樣本：{len(X_test)}")

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False)

    model     = get_model(MODEL_TYPE, input_size, cfg, output_size=len(QUANTILES)).to(DEVICE)
    criterion = QuantileLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           factor=0.5, patience=cfg.LR_PATIENCE)

    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_quantile_best.pt")
    best_val = float("inf")

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_sum += criterion(model(xb), yb).item() * len(xb)
        val_loss = val_sum / len(test_ds)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state":   model.state_dict(),
                "input_size":    input_size,
                "quantiles":     QUANTILES,
                "feature_names": feature_names,
                "target_col":    target_col,
                "val_loss":      val_loss,
            }, ckpt_path)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{cfg.EPOCHS}  val_pinball={val_loss:.6f}"
                  f"  lr={optimizer.param_groups[0]['lr']:.2e}")

    # ── Normalized CQR 校準（同時校準 80% 風險區間與 50% 最可能區間）──
    # 80% 區間 = [q10, q90]，alpha=0.2
    q80_holdout = _compute_conformal_q(ticker, 0.1, 0.9, alpha=0.2, on="cal")
    raw80  = _coverage(ticker, 0.0,         on="test", lo_q=0.1, hi_q=0.9)
    cov80  = _coverage(ticker, q80_holdout, on="test", lo_q=0.1, hi_q=0.9)
    # 50% 區間 = [q25, q75]，alpha=0.5
    q50_holdout = _compute_conformal_q(ticker, 0.25, 0.75, alpha=0.5, on="cal")
    cov50  = _coverage(ticker, q50_holdout, on="test", lo_q=0.25, hi_q=0.75)
    # 部署用 Q：完整測試集校準
    conformal_q80 = _compute_conformal_q(ticker, 0.1, 0.9, alpha=0.2, on="full")
    conformal_q50 = _compute_conformal_q(ticker, 0.25, 0.75, alpha=0.5, on="full")
    # 短線（乘法縮放公式）獨立校準：見 _compute_conformal_q_short 的實測依據
    conformal_q80_s5 = _compute_conformal_q_short(ticker, alpha=0.2, on="full")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    ckpt["conformal_q80"] = conformal_q80
    ckpt["conformal_q50"] = conformal_q50
    ckpt["conformal_q80_s5"] = conformal_q80_s5
    torch.save(ckpt, ckpt_path)

    print(f"  最佳 val_pinball：{best_val:.6f}")
    print(f"  80% 區間覆蓋率：校準前 {raw80:.0%} → 校準後 {cov80:.0%}（理想 80%）")
    print(f"  50% 區間覆蓋率：校準後 {cov50:.0%}（理想 50%）")
    return {"ticker": ticker, "val_loss": round(best_val, 6),
            "conformal_q80": round(conformal_q80, 4),
            "conformal_q50": round(conformal_q50, 4),
            "cov80": round(cov80, 4), "cov50": round(cov50, 4)}


VOL_WINDOW       = 20  # 波段：波動率估計視窗（日）
VOL_WINDOW_SHORT = 5   # 短線：近5日體溫（覆蓋率實測與20日同為72%，各自獨立校準）
VOL_FLOOR  = 0.3       # σ 下限（%），避免低波動期區間塌縮


def _vol_series(df: pd.DataFrame, window: int = VOL_WINDOW) -> pd.Series:
    """近 window 日報酬標準差作為波動率尺度 σ（含下限保護）。"""
    sigma = df["pct_change"].rolling(window, min_periods=3).std()
    return sigma.fillna(VOL_FLOOR).clip(lower=VOL_FLOOR)


def _compute_conformal_q(ticker: str, lo_q: float, hi_q: float,
                         alpha: float, on: str = "cal",
                         vol_window: int = VOL_WINDOW) -> float:
    """計算 [lo_q, hi_q] 分位數區間的「波動率正規化」修正量 Q（無單位倍數）。

    on="cal"  → 校準集（測試前半），僅供歷史診斷（同段曾用於 epoch 選擇，非獨立測試）
    on="full" → 完整驗證段（曾用於 epoch 選擇），供部署給「明天」用

    E_i = max(q_lo - y_i, y_i - q_hi) / σ_i，
    Q = {E_i} 的 ceil((n+1)(1-alpha))/n 經驗分位數。
    """
    lo_i, hi_i = Q_IDX[lo_q], Q_IDX[hi_q]
    model, scaler, ckpt = _load_quantile(ticker)
    df = _load_feature_df(ticker, ckpt["feature_names"])
    data_scaled = scaler.transform(df.values.astype(np.float32))
    sigma = _vol_series(df, vol_window).values
    w = cfg.WINDOW_SIZE
    split = int(len(data_scaled) * cfg.TRAIN_RATIO)
    end = len(data_scaled) - w
    hi = (split + (end - split) // 2) if on == "cal" else end

    scores = []
    for i in range(split, hi):
        qs = _predict_scaled_window(model, scaler, ckpt, data_scaled[i:i+w])
        y = float(df["pct_change"].iloc[i + w])
        s = sigma[i + w - 1]
        scores.append(max(qs[lo_i] - y, y - qs[hi_i]) / s)
    if not scores:
        return 0.0
    scores = np.sort(scores)
    n = len(scores)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    rank = min(max(rank, 1), n)               # 夾在 [1, n]
    return float(scores[rank - 1])


RHO_MIN, RHO_MAX = 0.3, 2.0     # 短線縮放比 ρ=σ5/σ20 的夾限（防塌陷/防爆寬）


def _compute_conformal_q_short(ticker: str, alpha: float = 0.2, on: str = "full") -> float:
    """短線（乘法縮放）公式的保形校準 Q。

    短線區間 = q50 ± (原始半寬 × ρ) ± Q·σ5，ρ=clip(σ5/σ20, 0.3, 2.0)。
    實測 vs 加法式：覆蓋率 72%→78%（更貼近80%目標）、冷卻日平均寬 12.6%→8.9%。
    """
    model, scaler, ckpt = _load_quantile(ticker)
    df = _load_feature_df(ticker, ckpt["feature_names"])
    data_scaled = scaler.transform(df.values.astype(np.float32))
    s5 = _vol_series(df, VOL_WINDOW_SHORT).values
    s20 = _vol_series(df, VOL_WINDOW).values
    lo_i, hi_i, mid_i = Q_IDX[0.1], Q_IDX[0.9], Q_IDX[0.5]
    w = cfg.WINDOW_SIZE
    split = int(len(data_scaled) * cfg.TRAIN_RATIO)
    end = len(data_scaled) - w
    hi = (split + (end - split) // 2) if on == "cal" else end

    scores = []
    for i in range(split, hi):
        qs = _predict_scaled_window(model, scaler, ckpt, data_scaled[i:i+w])
        y = float(df["pct_change"].iloc[i + w])
        rho = min(max(s5[i + w - 1] / s20[i + w - 1], RHO_MIN), RHO_MAX)
        lo_b = qs[mid_i] + (qs[lo_i] - qs[mid_i]) * rho
        hi_b = qs[mid_i] + (qs[hi_i] - qs[mid_i]) * rho
        scores.append(max(lo_b - y, y - hi_b) / s5[i + w - 1])
    if not scores:
        return 0.0
    scores = np.sort(scores)
    n = len(scores)
    rank = min(max(int(np.ceil((n + 1) * (1 - alpha))), 1), n)
    return float(scores[rank - 1])


# ── 預測 ──────────────────────────────────────────────────────────────────────

def _load_quantile(ticker: str):
    """載入分位數模型與共用 scaler。"""
    ckpt_path   = os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_quantile_best.pt")
    scaler_path = os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_scaler.pkl")
    if not (os.path.exists(ckpt_path) and os.path.exists(scaler_path)):
        raise FileNotFoundError(f"{ticker} 缺少分位數模型或 scaler")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = get_model(MODEL_TYPE, ckpt["input_size"], cfg, output_size=len(ckpt["quantiles"])).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler, ckpt


def _load_feature_df(ticker: str, feature_names: list) -> pd.DataFrame:
    """讀取 CSV 並取出模型特徵欄位（dropna）。"""
    df = pd.read_csv(os.path.join(cfg.RAW_DATA_DIR, f"{ticker}.csv"),
                     index_col=0, parse_dates=True)
    avail = [c for c in feature_names if c in df.columns]
    return df[avail].dropna()


def _predict_scaled_window(model, scaler, ckpt, window_scaled):
    """對單一視窗預測各分位數（scaled 空間），回傳排序後（不交叉）的 pct 值陣列。"""
    target_col = ckpt["target_col"]
    x = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x).cpu().numpy()[0]          # (n_quantiles,) scaled
    # 分位數不交叉修正：排序確保 q10 <= q25 <= q50 <= q75 <= q90
    out = np.sort(out)
    pcts = []
    for v in out:
        dummy = np.zeros((1, window_scaled.shape[1]), dtype=np.float32)
        dummy[0, target_col] = v
        pcts.append(float(scaler.inverse_transform(dummy)[0, target_col]))
    return pcts   # 依 QUANTILES 順序的 pct 值


def _p_up(q10, q25, q50, q75, q90) -> float:
    """由分位數內插 F(0)，回傳明日上漲機率 P(漲)=1−F(0)（%）。"""
    pts = [(q10, 0.1), (q25, 0.25), (q50, 0.5), (q75, 0.75), (q90, 0.9)]
    if 0 < pts[0][0]:          # 嚴格不等：q10==0 時應走內插得 F0=0.10（Fable 稽核 F4）
        F0 = 0.05
    elif 0 > pts[-1][0]:
        F0 = 0.95
    else:
        F0 = 0.5
        for (xa, pa), (xb, pb) in zip(pts, pts[1:]):
            if xa <= 0 <= xb and xb > xa:
                F0 = pa + (0 - xa) / (xb - xa) * (pb - pa)
                break
    F0 = min(max(F0, 0.01), 0.99)
    return round((1 - F0) * 100, 1)


def predict_quantile(ticker: str, news_bull=None) -> dict:
    """輸出今日對隔日的區間預測：50% 最可能區間 + 80% 風險區間 + 上漲機率。

    news_bull：今日新聞偏多分數。利多時把上界往 +10% 拉（資料：利多日90百分位+9.9%，
    32% 噴>+5%）。中心/下界不動，維持誠實。None=自動讀今日盤前新聞。"""
    model, scaler, ckpt = _load_quantile(ticker)
    cq80 = ckpt.get("conformal_q80", 0.0)
    cq50 = ckpt.get("conformal_q50", 0.0)

    df = _load_feature_df(ticker, ckpt["feature_names"])
    window = scaler.transform(df.values.astype(np.float32))[-cfg.WINDOW_SIZE:]
    qs = _predict_scaled_window(model, scaler, ckpt, window)
    q10, q25, q50, q75, q90 = qs
    q10_raw, q50_raw, q90_raw = q10, q50, q90   # 短線區間共用同一模型原始輸出

    # Normalized CQR：依當前波動率撐寬。80% 用 cq80、50% 用 cq50
    sig = float(_vol_series(df).iloc[-1])
    q10, q90 = q10 - cq80 * sig, q90 + cq80 * sig
    q25, q75 = q25 - cq50 * sig, q75 + cq50 * sig
    # 不交叉保護
    q25, q75 = max(q25, q10), min(q75, q90)
    # 漲跌停夾限：台股單日 close 對前收最多 ±10%，超出為物理不可能，夾掉避免區間虛胖
    # （不影響覆蓋率：實際漲跌也不可能超出 ±10%，只是修掉不可能的那段）
    LIMIT = 10.0
    q10, q25, q50, q75, q90 = [max(-LIMIT, min(LIMIT, x)) for x in (q10, q25, q50, q75, q90)]

    # 短線區間（乘法縮放 + 獨立保形校準）：原始寬度繞 q50 按 ρ=σ5/σ20 縮放，
    # 再 ± Q·σ5。實測覆蓋率 78%（優於加法式72%）、冷卻日平均寬 8.9%（vs 12.6%）。
    # 純波動率讀數（不做新聞撐寬）。⚠️ 短線窄 ≠ 安全：平靜5日後仍可能被暴衝打穿。
    sig5 = float(_vol_series(df, VOL_WINDOW_SHORT).iloc[-1])
    cq80s = ckpt.get("conformal_q80_s5", 0.0)
    rho = min(max(sig5 / sig, RHO_MIN), RHO_MAX) if sig else 1.0
    q10s = max(q50_raw + (q10_raw - q50_raw) * rho - cq80s * sig5, -LIMIT)
    q90s = min(q50_raw + (q90_raw - q50_raw) * rho + cq80s * sig5, LIMIT)
    temp = "❄️冷卻中" if rho < 0.7 else ("♨️加溫中" if rho > 1.3 else "")

    # 漲機率先用「未含新聞」的分位數算（Fable 稽核 F4：避免新聞先撐寬區間、
    # 再混機率的雙重計入——同一訊號只允許影響 p_up 一次）
    p_up_raw = _p_up(q10, q25, q50, q75, q90)

    # 新聞感知不對稱：中利多(0.15+)起把上界往 +10% 拉，0.30 拉滿
    # （資料：利多日90百分位+9.9%、32%噴>+5%；0.15=尾部分析的利多桶邊界）。
    # 只動上界(q75/q90)，中心/下界不動 → 維持誠實，不亂喊漲。
    if news_bull is None:
        try:
            from news_aux_judge import today_score
            news_bull = today_score(ticker.replace(".TWO", "").replace(".TW", ""))
        except Exception:
            news_bull = 0.0
    if news_bull and news_bull >= 0.15:
        frac = max(0.0, min((news_bull - 0.15) / 0.15, 1.0))   # 0.15→0、0.30→滿
        q90 = q90 + frac * (LIMIT - q90)
        q75 = min(q75 + 0.5 * frac * (LIMIT - q75), q90)

    # 強利多(0.20+，初步訊號)：p_up 往歷史 73% 靠，但取 max 不往下拉——
    # 模型自身已更樂觀時，新聞不應反而降低機率（Fable 稽核 F4）
    p_up = p_up_raw
    if news_bull and news_bull >= 0.20:
        p_up = max(p_up_raw, 0.4 * p_up_raw + 0.6 * 73.0)

    last = float(df["Close"].iloc[-1])

    return {
        "data_as_of": df.index[-1].date().isoformat(),
        "horizon_sessions": 1,
        "calibration_status": "legacy_unverified",
        "ticker":     ticker,
        "name":       cfg.ALL_STOCKS.get(ticker, ""),
        "last_close": round(last, 2),
        "p_up":       round(p_up, 1),
        "q50_pct":    round(q50, 3),
        "q50_price":  round(last * (1 + q50 / 100), 2),
        # 50% 最可能區間
        "q25_pct":    round(q25, 3), "q75_pct": round(q75, 3),
        "q25_price":  round(last * (1 + q25 / 100), 2),
        "q75_price":  round(last * (1 + q75 / 100), 2),
        "width50":    round(q75 - q25, 2),
        # 80% 風險區間（波段，σ20）
        "q10_pct":    round(q10, 3), "q90_pct": round(q90, 3),
        "q10_price":  round(last * (1 + q10 / 100), 2),
        "q90_price":  round(last * (1 + q90 / 100), 2),
        "width80":    round(q90 - q10, 2),
        # 短線 80% 區間（σ5 近5日體溫，獨立校準；純波動率讀數）
        "q10s_pct":   round(q10s, 3), "q90s_pct": round(q90s, 3),
        "q10s_price": round(last * (1 + q10s / 100), 2),
        "q90s_price": round(last * (1 + q90s / 100), 2),
        "width80s":   round(q90s - q10s, 2),
        "vol_temp":   temp,           # ❄️冷卻中 / ♨️加溫中 / ""（σ5/σ20 背離標記）
    }


def _coverage(ticker: str, conformal_q: float = 0.0, on: str = "test",
              lo_q: float = 0.1, hi_q: float = 0.9) -> float:
    """區間覆蓋率：實際漲跌幅落在 [q_lo−Q·σ, q_hi+Q·σ] 的比例。

    on="cal"  → 校準集（測試前半）
    on="test" → 測試後半（曾參與 epoch 選擇，不能稱為獨立測試）
    on="all"  → 整個測試集
    """
    lo_i, hi_i = Q_IDX[lo_q], Q_IDX[hi_q]
    model, scaler, ckpt = _load_quantile(ticker)
    df = _load_feature_df(ticker, ckpt["feature_names"])
    data_scaled = scaler.transform(df.values.astype(np.float32))
    sigma = _vol_series(df).values
    w = cfg.WINDOW_SIZE
    split = int(len(data_scaled) * cfg.TRAIN_RATIO)
    end = len(data_scaled) - w
    cal_end = split + (end - split) // 2
    if on == "cal":
        lo, hi = split, cal_end
    elif on == "test":
        lo, hi = cal_end, end
    else:
        lo, hi = split, end

    inside, total = 0, 0
    for i in range(lo, hi):
        qs = _predict_scaled_window(model, scaler, ckpt, data_scaled[i:i+w])
        actual = float(df["pct_change"].iloc[i + w])
        s = sigma[i + w - 1]
        if (qs[lo_i] - conformal_q * s) <= actual <= (qs[hi_i] + conformal_q * s):
            inside += 1
        total += 1
    return inside / total if total else float("nan")


# ── Wiki 格式化 ───────────────────────────────────────────────────────────────

def build_interval_lines(results: list) -> list:
    """產生寫入 Wiki 的區間預測 markdown（含上漲機率 + 50%/80% 雙區間）。"""
    lines = ["\n## 隔日區間預測（分位數迴歸 + 波動率正規化 CQR）\n",
             "| 股票 | 名稱 | 上漲機率 | 中位數 | 50%最可能區間(價) | 下一日80%區間σ20(價) | 下一日80%敏感度σ5(價) | 溫度 |",
             "|------|------|----------|--------|-------------------|-----------------|-----------------|------|"]
    for r in sorted(results, key=lambda x: -x["p_up"]):
        pu = r["p_up"]
        bias = "▲偏漲" if pu >= 55 else ("▼偏跌" if pu <= 45 else "—中性")
        lines.append(
            f"| {r['ticker']} | {r['name']} | {pu:.0f}% {bias} "
            f"| {r['q50_pct']:+.2f}% "
            f"| {r['q25_price']}~{r['q75_price']} "
            f"| {r['q10_price']}~{r['q90_price']} "
            f"| {r.get('q10s_price','—')}~{r.get('q90s_price','—')} "
            f"| {r.get('vol_temp','') or '—'} |"
        )
    lines.append("\n_所有區間目標都是下一交易日，σ20/σ5 是波動估計回看天數；名目涵蓋率尚待盤前留底驗證；"
                 "短線窄≠安全，平靜後仍可能被暴衝打穿。溫度=σ5/σ20背離標記。_")
    return lines


# ── 主程式 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="分位數迴歸區間預測")
    parser.add_argument("--train", action="store_true", help="訓練分位數模型")
    parser.add_argument("--ticker", default=None, help="只處理指定股票")
    args = parser.parse_args()

    tickers = [args.ticker] if args.ticker else list(cfg.ALL_STOCKS.keys())

    if args.train:
        print(f"裝置：{DEVICE} | 分位數模型訓練（{len(tickers)} 支）")
        summary = []
        for tk in tickers:
            try:
                summary.append(train_quantile(tk))
            except Exception as e:
                print(f"  [錯誤] {tk}: {e}")
        print("\n── 訓練摘要（80%區間理想≈80%、50%區間理想≈50%）──")
        for s in summary:
            print(f"  {s['ticker']:<10} val={s['val_loss']:.5f}  "
                  f"80%覆蓋 {s['cov80']:.0%}  50%覆蓋 {s['cov50']:.0%}")
    else:
        print("今日隔日區間預測（依上漲機率排序）：\n")
        results = []
        for tk in tickers:
            try:
                results.append(predict_quantile(tk))
            except Exception as e:
                print(f"  [跳過] {tk}: {e}")
        for r in sorted(results, key=lambda x: -x["p_up"]):
            print(f"  {r['ticker']:<10} {r['name']:<5} "
                  f"P(漲) {r['p_up']:>5.1f}%  中位數 {r['q50_pct']:+.2f}%  "
                  f"50%區間 {r['q25_price']}~{r['q75_price']}  "
                  f"80%區間 {r['q10_price']}~{r['q90_price']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
