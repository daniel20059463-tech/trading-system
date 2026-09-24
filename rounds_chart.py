"""
多模型對比折線圖（深色風格，對應 .md 框架的「各輪預測 vs 實際」圖）。

三條預測線 = LSTM / Transformer / Ensemble 三個模型
黃色分隔線 = Discovery 探索區（訓練 80%）| Generalization 泛化區（測試 20%）

用法：
  python rounds_chart.py --ticker 2327.TW
  python rounds_chart.py                 # 全部股票
"""
import sys
import os
import argparse
import pickle
from datetime import date

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from models.lstm_model import get_model
from visualize import _setup_font, CHART_DIR

DEVICE = torch.device("cpu")
FONT_NAME = _setup_font()


def _predict_fullseries(ticker: str, model_type: str):
    """對全序列每個視窗預測收盤價，回傳 (dates, actual_close, pred_close)。"""
    ckpt_path   = os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_{model_type}_best.pt")
    scaler_path = os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_scaler.pkl")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = get_model(model_type, ckpt["input_size"], cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    target_col    = ckpt.get("target_col", 0)
    feature_names = ckpt.get("feature_names", [])

    df = pd.read_csv(os.path.join(cfg.RAW_DATA_DIR, f"{ticker}.csv"),
                     index_col=0, parse_dates=True)
    avail = [c for c in feature_names if c in df.columns]
    df = df[avail].dropna()
    data = df.values.astype(np.float32)
    scaled = scaler.transform(data)

    w = cfg.WINDOW_SIZE
    dates, actual, pred = [], [], []
    for i in range(len(scaled) - w):
        x = torch.tensor(scaled[i:i+w], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p = model(x).cpu().numpy()[0, 0]
        dummy = np.zeros((1, scaled.shape[1]), dtype=np.float32)
        dummy[0, target_col] = p
        pred_pct = float(scaler.inverse_transform(dummy)[0, target_col])

        prev_close = float(df["Close"].iloc[i + w - 1])
        pred_close = prev_close * (1 + pred_pct / 100)
        actual_close = float(df["Close"].iloc[i + w])

        dates.append(df.index[i + w])
        actual.append(actual_close)
        pred.append(pred_close)

    return np.array(dates), np.array(actual), np.array(pred)


def _ensemble_pred(ticker: str):
    """Ensemble 全序列預測（動態權重）。"""
    d, a, lstm_p = _predict_fullseries(ticker, "lstm")
    try:
        _, _, tf_p = _predict_fullseries(ticker, "transformer")
    except FileNotFoundError:
        return d, a, lstm_p
    lstm_val = torch.load(os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_lstm_best.pt"),
                          map_location=DEVICE)["val_loss"]
    tf_val   = torch.load(os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_transformer_best.pt"),
                          map_location=DEVICE)["val_loss"]
    total = lstm_val + tf_val
    lw, tw = tf_val / total, lstm_val / total
    return d, a, lstm_p * lw + tf_p * tw


def plot_rounds_chart(ticker: str, n_windows: int = 40, save: bool = True) -> str:
    """畫深色風格的多模型對比圖：實際 + LSTM/Transformer/Ensemble + 訓練測試分隔線。"""
    dates, actual, lstm_pred = _predict_fullseries(ticker, "lstm")
    try:
        _, _, tf_pred = _predict_fullseries(ticker, "transformer")
        has_tf = True
    except FileNotFoundError:
        tf_pred, has_tf = None, False
    _, _, ens_pred = _ensemble_pred(ticker)

    total = len(actual)
    # 降採樣到 ~n_windows 個點（保留首尾）
    stride = max(1, total // n_windows)
    idx = list(range(0, total, stride))
    if idx[-1] != total - 1:
        idx.append(total - 1)

    x = np.arange(len(idx))
    a   = actual[idx]
    lp  = lstm_pred[idx]
    tp  = tf_pred[idx] if has_tf else None
    ep  = ens_pred[idx]

    # 訓練/測試分隔（80%）在降採樣後的位置
    split_full = int(total * cfg.TRAIN_RATIO)
    split_x = np.searchsorted(np.array(idx), split_full)

    name = cfg.ALL_STOCKS.get(ticker, ticker)

    # ── 深色風格 ──
    BG = "#16213e"
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.plot(x, a, color="#FFFFFF", linewidth=2.2, marker="o", markersize=3,
            label="實際價格" if FONT_NAME else "Actual", zorder=5)
    ax.plot(x, lp, color="#FF5252", linewidth=1.3, linestyle="--", marker="o", markersize=2.5,
            alpha=0.85, label="LSTM 預測" if FONT_NAME else "LSTM", zorder=3)
    if has_tf:
        ax.plot(x, tp, color="#40C4FF", linewidth=1.3, linestyle="--", marker="s", markersize=2.5,
                alpha=0.85, label="Transformer 預測" if FONT_NAME else "Transformer", zorder=3)
    ax.plot(x, ep, color="#69F0AE", linewidth=1.3, linestyle="--", marker="^", markersize=2.5,
            alpha=0.85, label="Ensemble 預測" if FONT_NAME else "Ensemble", zorder=4)

    # 訓練/測試分隔線
    ax.axvline(split_x, color="#FFD740", linestyle="--", linewidth=1.5, alpha=0.9)
    ymin, ymax = ax.get_ylim()
    ax.text(split_x - 0.5, ymin + (ymax - ymin) * 0.05,
            "Discovery 探索區" if FONT_NAME else "Train",
            color="#FFD740", fontsize=9, ha="right", va="bottom")
    ax.text(split_x + 0.5, ymin + (ymax - ymin) * 0.05,
            "Generalization 泛化區" if FONT_NAME else "Test",
            color="#FFD740", fontsize=9, ha="left", va="bottom")

    # 標注實際價格數值（每隔幾點）
    label_stride = max(1, len(idx) // 12)
    for k in range(0, len(idx), label_stride):
        ax.annotate(f"{a[k]:.0f}", (x[k], a[k]),
                    textcoords="offset points", xytext=(0, 7),
                    fontsize=7, color="#E0E0E0", ha="center")

    title = f"{ticker} {name} — 各模型預測 vs 實際" if FONT_NAME else f"{ticker} models vs actual"
    ax.set_title(title, fontsize=13, fontweight="bold", color="#FFFFFF", pad=12)
    ax.set_xlabel(f"Window W1 to W{len(idx)}", color="#B0BEC5", fontsize=9)
    ax.set_ylabel("價格" if FONT_NAME else "Price", color="#B0BEC5", fontsize=9)
    ax.tick_params(colors="#B0BEC5", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#37474F")
    ax.grid(True, alpha=0.15, linestyle=":", color="#FFFFFF")

    legend = ax.legend(fontsize=9, loc="upper left", framealpha=0.3)
    legend.get_frame().set_facecolor("#0D1B2A")
    for text in legend.get_texts():
        text.set_color("#FFFFFF")

    plt.tight_layout()
    out_path = ""
    if save:
        os.makedirs(CHART_DIR, exist_ok=True)
        out_path = os.path.join(CHART_DIR, f"{date.today().isoformat()}_{ticker}_rounds.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="多模型對比折線圖（深色風格）")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--windows", type=int, default=40)
    args = parser.parse_args()

    tickers = [args.ticker] if args.ticker else list(cfg.ALL_STOCKS.keys())
    for tk in tickers:
        try:
            path = plot_rounds_chart(tk, n_windows=args.windows)
            print(f"{tk}: {path}")
        except Exception as e:
            print(f"[跳過] {tk}: {e}")
