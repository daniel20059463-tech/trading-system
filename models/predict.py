# predict.py - 載入訓練好的模型對股票進行下一日收盤價預測

import os
import sys
import pickle

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from models.lstm_model import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 載入模型與 Scaler ──────────────────────────────────────────────────────────

def load_model_and_scaler(ticker: str, model_type: str = "lstm"):
    """從檢查點目錄載入指定股票的最佳模型與 MinMaxScaler。"""
    ckpt_path   = os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_{model_type}_best.pt")
    scaler_path = os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_scaler.pkl")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到模型檢查點：{ckpt_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"找不到 scaler 檔案：{scaler_path}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model = get_model(
        model_type = checkpoint.get("model_type", model_type),
        input_size = checkpoint["input_size"],
        config     = cfg,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    feature_names = checkpoint.get("feature_names", None)
    target_col    = checkpoint.get("target_col", 0)
    return model, scaler, feature_names, target_col


# ── 單支股票預測 ───────────────────────────────────────────────────────────────

def predict_next_day(ticker: str, model_type: str = "lstm") -> dict:
    """用最後 window_size 筆資料推論下一交易日收盤價，回傳含方向與信心度的預測字典。

    預測流程：
    1. 模型輸出為正規化後的 pct_change（漲跌幅）
    2. Inverse transform 還原為實際漲跌幅 %
    3. predicted_close = last_close * (1 + pct / 100)
    """
    model, scaler, feature_names, target_col = load_model_and_scaler(ticker, model_type)

    csv_path = os.path.join(cfg.RAW_DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到原始資料：{csv_path}")

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if feature_names is not None:
        available = [c for c in feature_names if c in df.columns]
    else:
        available = [c for c in df.columns if c != "Open"]
    df = df[available].dropna()

    if len(df) < cfg.WINDOW_SIZE:
        raise ValueError(f"{ticker} 資料不足 {cfg.WINDOW_SIZE} 筆，無法預測。")

    window = df.values[-cfg.WINDOW_SIZE:].astype(np.float32)
    window_scaled = scaler.transform(window)

    x = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_scaled = model(x).cpu().numpy().reshape(1, 1)

    # Inverse transform：還原 target_col 對應的原始值
    dummy = np.zeros((1, window_scaled.shape[1]), dtype=np.float32)
    dummy[0, target_col] = pred_scaled[0, 0]
    pred_value = float(scaler.inverse_transform(dummy)[0, target_col])

    last_close = float(df["Close"].iloc[-1])

    # target_col 為 pct_change 時，直接用漲跌幅推算價格
    feature_names_list = feature_names or list(df.columns)
    target_name = feature_names_list[target_col] if target_col < len(feature_names_list) else "Close"

    if target_name == "pct_change":
        predicted_change_pct = pred_value
        pred_price = last_close * (1 + predicted_change_pct / 100)
    else:
        pred_price = pred_value
        predicted_change_pct = ((pred_price - last_close) / last_close) * 100

    direction = "UP" if predicted_change_pct >= 0 else "DOWN"

    # 信心度：預測幅度 / 歷史平均波動（上限 1.0）
    if "pct_change" in df.columns:
        avg_volatility = float(df["pct_change"].abs().mean())
    else:
        avg_volatility = 1.0

    confidence = min(abs(predicted_change_pct) / avg_volatility, 1.0) if avg_volatility > 0 else 0.0

    # prev_close：前一日收盤價，供盤後方向準確率計算使用
    prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else last_close

    return {
        "ticker":               ticker,
        "name":                 cfg.ALL_STOCKS.get(ticker, ""),
        "data_as_of":          df.index[-1].date().isoformat(),
        "predicted_close":      round(float(pred_price), 2),
        "last_close":           round(last_close, 2),
        "prev_close":           round(prev_close, 2),
        "predicted_change_pct": round(float(predicted_change_pct), 4),
        "direction":            direction,
        "confidence":           round(confidence, 4),
        "model_type":           model_type,
    }


# ── Ensemble 預測（LSTM + Transformer 加權平均）────────────────────────────────

def predict_ensemble(ticker: str, lstm_weight: float = 0.5) -> dict:
    """LSTM 與 Transformer 預測的加權平均 Ensemble。

    自動偵測哪個模型 val_loss 較低，給予較高權重（若其中一個不存在則 fallback 到單模型）。
    """
    transformer_weight = 1.0 - lstm_weight

    # 嘗試載入兩個模型
    try:
        lstm_ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_lstm_best.pt")
        tf_ckpt_path   = os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_transformer_best.pt")
        lstm_val  = torch.load(lstm_ckpt_path, map_location=DEVICE)["val_loss"]
        tf_val    = torch.load(tf_ckpt_path,   map_location=DEVICE)["val_loss"]
        # 動態權重：val_loss 越低，權重越高
        total = lstm_val + tf_val
        lstm_weight      = round(tf_val / total, 3)   # 對方 loss 高 → 我方 weight 高
        transformer_weight = round(lstm_val / total, 3)
        have_both = True
    except FileNotFoundError:
        have_both = False

    if not have_both:
        # 只有其中一個模型
        for mt in ("lstm", "transformer"):
            try:
                return predict_next_day(ticker, model_type=mt)
            except FileNotFoundError:
                continue
        raise FileNotFoundError(f"{ticker} 無可用模型")

    lstm_result = predict_next_day(ticker, model_type="lstm")
    tf_result   = predict_next_day(ticker, model_type="transformer")

    # 在 pct_change 空間做加權平均，再還原成價格
    lstm_pct = lstm_result["predicted_change_pct"]
    tf_pct   = tf_result["predicted_change_pct"]
    ensemble_pct = lstm_pct * lstm_weight + tf_pct * transformer_weight

    last_close = lstm_result["last_close"]
    ensemble_price = round(last_close * (1 + ensemble_pct / 100), 2)
    direction = "UP" if ensemble_pct >= 0 else "DOWN"

    avg_volatility = abs(lstm_pct + tf_pct) / 2 or 1.0
    confidence = min(abs(ensemble_pct) / avg_volatility, 1.0)

    return {
        "ticker":               ticker,
        "name":                 cfg.ALL_STOCKS.get(ticker, ""),
        "data_as_of":          lstm_result["data_as_of"],
        "predicted_close":      ensemble_price,
        "last_close":           last_close,
        "prev_close":           lstm_result["prev_close"],
        "predicted_change_pct": round(float(ensemble_pct), 4),
        "direction":            direction,
        "confidence":           round(confidence, 4),
        "model_type":           "ensemble",
        "lstm_weight":          lstm_weight,
        "transformer_weight":   transformer_weight,
        "lstm_pct":             lstm_pct,
        "transformer_pct":      tf_pct,
    }


# ── 盤後驗證用：預測 CSV 最後一日（用其前一日資料）─────────────────────────────

def predict_verification() -> list:
    """盤後驗證：對每支股票，用「倒數第2天為止」的資料預測「最後一天」，
    回傳與該日實際收盤對齊的預測，供盤後正確計算 MAPE 與方向準確率。

    解決時間錯位：predict_all 預測的是「未來下一日」，無法與今日實際對齊；
    本函式預測的是「CSV 最後一日」（即今日），可直接與今日實際比對。
    """
    results = []
    for ticker in cfg.ALL_STOCKS:
        try:
            # 載入兩模型權重（ensemble）
            models = {}
            for mt in ("lstm", "transformer"):
                try:
                    model, scaler, feature_names, target_col = load_model_and_scaler(ticker, mt)
                    val = torch.load(
                        os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_{mt}_best.pt"),
                        map_location=DEVICE,
                    )["val_loss"]
                    models[mt] = (model, val)
                    _fn, _tc = feature_names, target_col
                    _scaler = scaler
                except FileNotFoundError:
                    continue
            if not models:
                continue

            df = pd.read_csv(os.path.join(cfg.RAW_DATA_DIR, f"{ticker}.csv"),
                             index_col=0, parse_dates=True)
            avail = [c for c in _fn if c in df.columns]
            df = df[avail].dropna()
            if len(df) < cfg.WINDOW_SIZE + 1:
                continue

            # 用「倒數第2天為止」的視窗預測最後一天
            window = df.values[-cfg.WINDOW_SIZE - 1:-1].astype(np.float32)
            window_scaled = _scaler.transform(window)
            x = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            preds, weights = [], []
            for mt, (model, val) in models.items():
                with torch.no_grad():
                    p = model(x).cpu().numpy()[0, 0]
                dummy = np.zeros((1, window_scaled.shape[1]), dtype=np.float32)
                dummy[0, _tc] = p
                pct = float(_scaler.inverse_transform(dummy)[0, _tc])
                preds.append(pct)
                weights.append(1.0 / (val + 1e-9))
            w = np.array(weights) / sum(weights)
            ens_pct = float(np.dot(preds, w))

            prev_close = float(df["Close"].iloc[-2])   # 倒數第2天（預測基準）
            actual     = float(df["Close"].iloc[-1])   # 最後一天（今日實際）
            pred_price = prev_close * (1 + ens_pct / 100)

            results.append({
                "ticker":          ticker,
                "name":            cfg.ALL_STOCKS.get(ticker, ""),
                "predicted_close": round(pred_price, 2),
                "predicted_price": round(pred_price, 2),
                "last_close":      round(prev_close, 2),   # 預測基準=昨日
                "prev_close":      round(prev_close, 2),
                "direction":       "UP" if ens_pct >= 0 else "DOWN",
                "actual_close":    round(actual, 2),
                "model_type":      "ensemble",
            })
        except Exception as exc:
            print(f"  [驗證跳過] {ticker}: {exc}")
    return results


# ── 批次預測所有股票 ───────────────────────────────────────────────────────────

def predict_all(model_type: str = "ensemble") -> list:
    """對 ALL_STOCKS 中的所有股票執行下一日收盤價預測，回傳預測結果列表。"""
    results = []
    for ticker in cfg.ALL_STOCKS:
        try:
            if model_type == "ensemble":
                result = predict_ensemble(ticker)
            else:
                result = predict_next_day(ticker, model_type=model_type)
            results.append(result)
            sign = "+" if result["predicted_change_pct"] >= 0 else ""
            model_tag = result.get("model_type", model_type)
            print(
                f"  {ticker:<12} {result['name']:<6}"
                f"  預測: {result['predicted_close']:>8.2f}"
                f"  ({sign}{result['predicted_change_pct']:.2f}%)"
                f"  方向: {result['direction']:<4}"
                f"  信心: {result['confidence']:.2f}"
                f"  [{model_tag}]"
            )
        except Exception as exc:
            print(f"  [跳過] {ticker}: {exc}")
    return results


# ── 主程式 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="台灣股票下一日收盤價預測")
    parser.add_argument(
        "--model",
        type=str,
        default="lstm",
        choices=["lstm", "transformer"],
        help="選擇模型架構：lstm（預設）或 transformer",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="只預測指定股票，不指定則預測全部",
    )
    args = parser.parse_args()

    print(f"裝置：{DEVICE} | 模型：{args.model.upper()}\n")
    print(f"{'Ticker':<10} {'名稱':<6}  {'預測收盤':>10}  {'漲跌幅':>10}  {'方向':<4}  {'信心':>6}")
    print("-" * 65)

    if args.ticker:
        try:
            result = predict_next_day(args.ticker, model_type=args.model)
            sign = "+" if result["predicted_change_pct"] >= 0 else ""
            print(
                f"  {result['ticker']:<10} {result['name']:<6}"
                f"  預測: {result['predicted_close']:>8.2f}"
                f"  ({sign}{result['predicted_change_pct']:.2f}%)"
                f"  方向: {result['direction']:<4}"
                f"  信心: {result['confidence']:.2f}"
            )
        except Exception as exc:
            print(f"錯誤：{exc}")
    else:
        results = predict_all(model_type=args.model)
        print(f"\n共預測 {len(results)} 支股票。")
        up   = sum(1 for r in results if r["direction"] == "UP")
        down = len(results) - up
        print(f"看漲：{up} 支  看跌：{down} 支")
