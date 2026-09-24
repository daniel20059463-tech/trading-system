"""
未來多日遞迴預測：用「預測值當輸入」逐日往後推 N 天。

⚠️ 警告：遞迴預測誤差會逐日累積放大（第5天可能 MAPE >15%），
僅供趨勢方向參考，不可當作精確價位。

遞迴流程每一步：
  1. 用目前特徵視窗預測隔日 pct_change
  2. 由 pct_change 推算隔日 Close
  3. 附加合成 OHLCV 列，重算所有技術指標
  4. 同業/宏觀特徵設為 0（未來未知，中性假設）
  5. 滑動視窗前進，重複

用法：
  python forecast.py --ticker 2327.TW --days 5
  python forecast.py --days 5            # 全部股票，輸出圖表
"""
import sys
import os
import argparse
import pickle
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from models.lstm_model import get_model
from data.fetch_stocks import add_features
from visualize import _setup_font, CHART_DIR

DEVICE = torch.device("cpu")
FONT_NAME = _setup_font()


def _load(ticker: str, model_type: str):
    """載入模型、scaler、特徵名稱、目標欄位。"""
    ckpt_path   = os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_{model_type}_best.pt")
    scaler_path = os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_scaler.pkl")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = get_model(model_type, ckpt["input_size"], cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler, ckpt.get("feature_names", []), ckpt.get("target_col", 0)


def _next_business_day(d):
    """回傳下一個工作日（跳過週末）。"""
    nd = d + timedelta(days=1)
    while nd.weekday() >= 5:   # 5=六, 6=日
        nd += timedelta(days=1)
    return nd


def _predict_one_step(models: dict, feature_names, target_col, scaler, feat_df):
    """用一組模型（單一或 ensemble）對特徵視窗預測隔日 pct_change。"""
    window = feat_df[feature_names].dropna().values[-cfg.WINDOW_SIZE:].astype(np.float32)
    scaled = scaler.transform(window)
    x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

    preds = []
    weights = []
    for mt, (model, val_loss) in models.items():
        with torch.no_grad():
            p = model(x).cpu().numpy()[0, 0]
        dummy = np.zeros((1, scaled.shape[1]), dtype=np.float32)
        dummy[0, target_col] = p
        pct = float(scaler.inverse_transform(dummy)[0, target_col])
        preds.append(pct)
        weights.append(1.0 / (val_loss + 1e-9))   # val_loss 越低權重越高

    w = np.array(weights) / sum(weights)
    return float(np.dot(preds, w))


def forecast_future(ticker: str, days: int = 5, model_type: str = "ensemble") -> pd.DataFrame:
    """遞迴預測未來 N 天，回傳 DataFrame(date, predicted_close, predicted_pct, cum_pct)。"""
    # 載入模型（ensemble 載兩個）
    models = {}
    if model_type == "ensemble":
        for mt in ("lstm", "transformer"):
            try:
                model, scaler, feature_names, target_col = _load(ticker, mt)
                val = torch.load(os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_{mt}_best.pt"),
                                 map_location=DEVICE)["val_loss"]
                models[mt] = (model, val)
                _scaler, _fn, _tc = scaler, feature_names, target_col
            except FileNotFoundError:
                continue
        if not models:
            raise FileNotFoundError(f"{ticker} 無可用模型")
        scaler, feature_names, target_col = _scaler, _fn, _tc
    else:
        model, scaler, feature_names, target_col = _load(ticker, model_type)
        val = torch.load(os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_{model_type}_best.pt"),
                         map_location=DEVICE)["val_loss"]
        models[model_type] = (model, val)

    # 載入原始 OHLCV
    df = pd.read_csv(os.path.join(cfg.RAW_DATA_DIR, f"{ticker}.csv"),
                     index_col=0, parse_dates=True)

    # 真實同業特徵（依日期保留），未來合成列補 0
    peer_cols = [c for c in feature_names if c.endswith("_pct")]   # usdtwd_pct/twii_pct/...
    peer_hist = {c: df[c].copy() for c in peer_cols if c in df.columns}

    # 工作用 OHLCV（重算指標的基底）
    ohlcv = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    recent_volume = float(ohlcv["Volume"].tail(20).mean())

    results = []
    cum = 1.0
    for step in range(days):
        feat = add_features(ohlcv.copy())
        # 還原同業特徵：歷史用真實值，合成日補 0
        for c in peer_cols:
            if c in peer_hist:
                feat[c] = peer_hist[c].reindex(feat.index).fillna(0.0)
            else:
                feat[c] = 0.0

        pred_pct = _predict_one_step(models, feature_names, target_col, scaler, feat)

        last_close = float(ohlcv["Close"].iloc[-1])
        new_close  = last_close * (1 + pred_pct / 100)
        next_date  = _next_business_day(ohlcv.index[-1].date())
        cum *= (1 + pred_pct / 100)

        results.append({
            "date":            pd.Timestamp(next_date),
            "predicted_close": round(new_close, 2),
            "predicted_pct":   round(pred_pct, 4),
            "cum_pct":         round((cum - 1) * 100, 4),
        })

        # 附加合成列（OHLC 設為預測收盤，量用近期均量）
        ohlcv.loc[pd.Timestamp(next_date)] = {
            "Open": new_close, "High": new_close,
            "Low": new_close, "Close": new_close, "Volume": recent_volume,
        }

    return pd.DataFrame(results)


def plot_forecast(ticker: str, days: int = 5, model_type: str = "ensemble",
                  history_days: int = 30, save: bool = True) -> str:
    """畫出歷史收盤 + 未來預測折線（含誤差錐）。"""
    fc = forecast_future(ticker, days, model_type)
    df = pd.read_csv(os.path.join(cfg.RAW_DATA_DIR, f"{ticker}.csv"),
                     index_col=0, parse_dates=True)
    hist = df["Close"].tail(history_days)

    # 歷史日波動率，建誤差錐（隨機漫步 ±std*sqrt(t)）
    daily_std = float(df["pct_change"].tail(60).std())

    name = cfg.ALL_STOCKS.get(ticker, ticker)
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(hist.index, hist.values, color="#1976D2", linewidth=1.8,
            label="歷史收盤" if FONT_NAME else "History")

    # 接點：歷史最後一天 → 預測
    fc_dates = [hist.index[-1]] + list(fc["date"])
    fc_close = [hist.values[-1]] + list(fc["predicted_close"])
    ax.plot(fc_dates, fc_close, color="#E53935", linewidth=1.8,
            linestyle="--", marker="o", markersize=4,
            label=f"預測未來{days}天" if FONT_NAME else f"Forecast {days}d")

    # 誤差錐
    band_dates = list(fc["date"])
    upper, lower = [], []
    for k, row in enumerate(fc.itertuples(), start=1):
        band = row.predicted_close * (daily_std / 100) * np.sqrt(k)
        upper.append(row.predicted_close + band)
        lower.append(row.predicted_close - band)
    ax.fill_between(band_dates, lower, upper, color="#E53935", alpha=0.12,
                    label="誤差範圍 (±1σ)" if FONT_NAME else "Uncertainty")

    title = f"{ticker} {name} — 未來 {days} 日預測" if FONT_NAME else f"{ticker} {days}d forecast"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25, linestyle=":")

    # 標注每日預測漲跌幅
    for row in fc.itertuples():
        ax.annotate(f"{row.predicted_pct:+.1f}%",
                    (row.date, row.predicted_close),
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=7, color="#E53935", ha="center")

    plt.tight_layout()
    out_path = ""
    if save:
        os.makedirs(CHART_DIR, exist_ok=True)
        out_path = os.path.join(CHART_DIR, f"{date.today().isoformat()}_{ticker}_forecast.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="未來多日遞迴預測")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--model", default="ensemble", choices=["lstm", "transformer", "ensemble"])
    args = parser.parse_args()

    tickers = [args.ticker] if args.ticker else list(cfg.ALL_STOCKS.keys())
    for tk in tickers:
        try:
            fc = forecast_future(tk, args.days, args.model)
            name = cfg.ALL_STOCKS.get(tk, "")
            print(f"\n{tk} {name} 未來 {args.days} 日預測：")
            for row in fc.itertuples():
                print(f"  {row.date.date()}  收盤 {row.predicted_close:>8.2f}  "
                      f"({row.predicted_pct:+.2f}%)  累積 {row.cum_pct:+.2f}%")
            path = plot_forecast(tk, args.days, args.model)
            if path:
                print(f"  圖表：{path}")
        except Exception as e:
            print(f"  [跳過] {tk}: {e}")
