# train.py - 台灣股票預測模型訓練腳本

import os
import sys
import argparse
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from models.lstm_model import get_model

# ── 常數 ───────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "Close",
    "MA5", "MA20", "MA60",
    "RSI",
    "MACD", "MACD_signal", "MACD_hist",
    "BB_upper", "BB_mid", "BB_lower",
    "Volume_MA20", "Volume_ratio",
    "ATR",
    "pct_change",
    # 同業與宏觀指標漲跌幅（每支股票只會有其產業對應的欄位，filter 自動挑選）
    "usdtwd_pct", "twii_pct",          # 共用宏觀
    "murata_pct", "tdk_pct",           # 被動元件同業
    "onsemi_pct", "vishay_pct",        # 功率元件同業
    # 註：三大法人買賣超（籌碼面）曾於 2026-06 實驗，測試集方向準確率 56.2%→55.0%
    #     無提升反小幅變差（外資對中小型功率股是雜訊），已退回。模組 fetch_chips.py 保留備用。
    # 註：新聞關鍵字分數（news_score）曾於 2026-06 實驗（2年 FinMind 新聞、防洩漏權重），
    #     全測試集方向 -0.2%、連「有新聞當天」也 -1.3%，無提升已退回。
    #     原因：新聞訊號真實但稀疏(22%天)且落後於價格——模型已從價量動能間接吃到。
    #     關鍵字表（keyword_impact_report.md）保留作「人看的」決策輔助，非 ML 特徵。
    #     實驗腳本 _test_news_feature.py、特徵 build_news_feature.py 保留備查。
    # 註：隔夜美股費半 SOX 曾於 2026-06 實驗，全測試集 +0.5%、大動作日 +0.6%（皆雜訊內），
    #     未過門檻已退回。原因：模型已有 onsemi_pct/vishay_pct（美股半導體同業），
    #     SOX 訊息大部分重複；且崩盤/反轉日仍抓不到。實驗腳本 _test_sox_feature.py 保留備查。
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── 資料集準備 ─────────────────────────────────────────────────────────────────

def prepare_dataset(ticker: str):
    """讀取股票 CSV，正規化、滑動視窗切割，預測目標為下一日 pct_change（漲跌幅）。

    修正：
    - scaler 只 fit 訓練集，避免測試期資料洩漏到正規化統計量
    - 預測目標改為 pct_change（穩態序列），再由 predict.py 還原成價格

    已知輕微邊界重疊（Fable 稽核 F5）：滑動視窗在 train/test 交界處共用約
    WINDOW*0.2 行資料（滑窗通病），影響極小；嚴格作法是交界留 WINDOW 天緩衝。
    """
    csv_path = os.path.join(cfg.RAW_DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 {csv_path}，請先執行 data/fetch_stocks.py。")

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    available = [c for c in FEATURE_COLS if c in df.columns]
    df = df[available].dropna()

    feature_names = available
    data = df.values.astype(np.float32)

    # 決定預測目標欄位：優先用 pct_change，否則 fallback 到 Close
    if "pct_change" in feature_names:
        target_col = feature_names.index("pct_change")
    else:
        target_col = feature_names.index("Close")

    # 按時間順序切分資料點（先切再 fit scaler，防止洩漏）
    data_split = int(len(data) * cfg.TRAIN_RATIO)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[:data_split])           # 只用訓練集統計量
    data_scaled = scaler.transform(data)    # 整體 transform（含測試集）

    # 建立滑動視窗樣本
    X, y = [], []
    w = cfg.WINDOW_SIZE
    for i in range(len(data_scaled) - w):
        X.append(data_scaled[i : i + w])
        y.append(data_scaled[i + w, target_col])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    # 按時間順序切分樣本（不 shuffle）
    split = int(len(X) * cfg.TRAIN_RATIO)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, y_train, X_test, y_test, scaler, feature_names, target_col


# ── 訓練函式 ───────────────────────────────────────────────────────────────────

def train_model(ticker: str, model_type: str = "lstm",
                direction_weight: float = 0.0, select_metric: str = "loss"):
    """訓練指定股票的預測模型，每 10 epoch 印出損失，並儲存最佳模型與 scaler。

    Args:
        direction_weight: 方向感知 loss 權重。>0 時在 MSE 外加上「猜錯漲跌方向」的
                          hinge 懲罰，讓模型不只顧價格幅度、也顧漲跌方向。
        select_metric: 最佳 checkpoint 選擇依據。
                       "loss"=驗證 MSE 最低（預設，向後相容）；
                       "direction"=驗證方向準確率最高（MSE 為平手時的次要依據）。
    """
    print(f"\n{'='*60}")
    print(f"[訓練] {ticker} ({cfg.ALL_STOCKS.get(ticker, '')}) | 模型：{model_type.upper()}")
    if direction_weight > 0 or select_metric == "direction":
        print(f"  方向感知模式：dir_weight={direction_weight} | 選擇依據={select_metric}")
    print(f"{'='*60}")

    X_train, y_train, X_test, y_test, scaler, feature_names, target_col = prepare_dataset(ticker)

    input_size = X_train.shape[2]
    target_name = feature_names[target_col]
    print(f"  特徵數：{input_size} | 訓練樣本：{len(X_train)} | 測試樣本：{len(X_test)}")
    print(f"  預測目標：{target_name}（欄位 {target_col}）")

    # 漲跌方向門檻：scaled 空間中 pct_change=0 對應的值（MinMaxScaler 下即 min_[target_col]）
    # 預測值 > 門檻 = 看漲，< 門檻 = 看跌。
    dir_thr = float(scaler.min_[target_col])

    # DataLoader
    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train)
    )
    # F1 修正（Fable 稽核）：方向選擇模式只用測試段「前半 cal」挑 checkpoint，
    # 後半 holdout 留給 eval_harness 閘門誠實評估——否則挑選與評估同一段，
    # 200 epoch 挑最好再自評 = 選擇偏誤，成績虛高。
    if select_metric == "direction":
        n_cal = max(len(X_test) // 2, 1)
        val_ds = TensorDataset(torch.tensor(X_test[:n_cal]), torch.tensor(y_test[:n_cal]))
    else:
        val_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False)

    # 模型、損失、優化器
    model     = get_model(model_type, input_size, cfg).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=getattr(cfg, "LR_PATIENCE", 20)
    )

    # 檢查點路徑
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    best_ckpt    = os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_{model_type}_best.pt")
    scaler_path  = os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_scaler.pkl")

    def _direction_penalty(pred, target):
        """漲跌方向 hinge 懲罰：預測落在門檻錯誤一側時，依錯誤幅度給罰。"""
        d_target = torch.sign(target - dir_thr)          # +1 漲 / -1 跌
        wrong_side = torch.relu(-(pred - dir_thr) * d_target)
        return wrong_side.mean()

    best_val_loss   = float("inf")
    best_val_dir    = -1.0
    train_losses, val_losses = [], []

    for epoch in range(1, cfg.EPOCHS + 1):
        # --- 訓練 ---
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            if direction_weight > 0:
                loss = loss + direction_weight * _direction_penalty(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        train_loss = epoch_loss / len(train_ds)
        train_losses.append(train_loss)

        # --- 驗證（同時算 MSE 與方向準確率）---
        model.eval()
        val_loss_sum = 0.0
        dir_correct, dir_total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_loss_sum += criterion(pred, yb).item() * len(xb)
                pred_up = (pred > dir_thr)
                act_up  = (yb > dir_thr)
                dir_correct += (pred_up == act_up).sum().item()
                dir_total   += yb.numel()
        val_loss = val_loss_sum / len(val_ds)
        val_dir  = dir_correct / dir_total if dir_total else 0.0
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        # 儲存最佳模型：依 select_metric 決定主要判準
        if select_metric == "direction":
            # 方向準確率為主，MSE 為平手時次要依據
            improved = (val_dir > best_val_dir) or \
                       (val_dir == best_val_dir and val_loss < best_val_loss)
        else:
            improved = val_loss < best_val_loss

        if improved:
            best_val_loss = min(best_val_loss, val_loss)
            best_val_dir  = max(best_val_dir, val_dir)
            torch.save(
                {
                    "epoch":         epoch,
                    "model_state":   model.state_dict(),
                    "input_size":    input_size,
                    "model_type":    model_type,
                    "val_loss":      val_loss,
                    "val_dir_acc":   round(val_dir, 4),
                    "feature_names": feature_names,
                    "target_col":    target_col,
                },
                best_ckpt,
            )

        # 每 10 epoch 印出進度
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:>3}/{cfg.EPOCHS}"
                f"  train_loss={train_loss:.6f}"
                f"  val_loss={val_loss:.6f}"
                f"  val_dir={val_dir:.1%}"
                f"  lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    # 儲存 scaler
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n  最佳 val_loss：{best_val_loss:.6f} | 最佳 val_dir：{best_val_dir:.1%}")
    print(f"  模型存至：{best_ckpt}")
    print(f"  Scaler 存至：{scaler_path}")

    return train_losses, val_losses


# ── 主程式 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="訓練台灣股票預測模型")
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
        help="只訓練指定股票，不指定則訓練全部",
    )
    args = parser.parse_args()

    tickers = [args.ticker] if args.ticker else list(cfg.ALL_STOCKS.keys())
    print(f"裝置：{DEVICE} | 模型：{args.model.upper()} | 股票數：{len(tickers)}")

    results = {}
    for ticker in tickers:
        try:
            train_losses, val_losses = train_model(ticker, model_type=args.model)
            results[ticker] = {
                "final_train_loss": train_losses[-1],
                "final_val_loss":   val_losses[-1],
            }
        except Exception as exc:
            print(f"  [錯誤] {ticker}: {exc}")

    print("\n\n── 訓練摘要 ──")
    for t, r in results.items():
        print(
            f"  {t:<10} train={r['final_train_loss']:.6f}  val={r['final_val_loss']:.6f}"
        )
