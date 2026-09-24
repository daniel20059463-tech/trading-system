"""
過擬合監控：比較每支股票訓練集 vs 測試集 MAPE，自動偵測並警告過擬合。

對應 .md 框架原則：
  「若 Discovery MAPE 持續下降，但 Generalization MAPE 反而飆升，
   代表預測規則已嚴重過擬合，必須立刻回溯並簡化全域規則。」

判定標準（差距 = 測試MAPE - 訓練MAPE）：
  < 2%    健康
  2~4%    輕微（觀察）
  > 4%    ⚠️ 過擬合警告（建議簡化模型/減特徵）

用法：
  python overfit_monitor.py                 # 全部股票，LSTM
  python overfit_monitor.py --model transformer
  python overfit_monitor.py --save          # 結果寫入 Wiki
"""
import sys
import os
import argparse
import pickle
from datetime import date

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from models.lstm_model import get_model

DEVICE = torch.device("cpu")
WARN_GAP   = 4.0   # 差距超過此值判定過擬合警告（%）
MILD_GAP   = 2.0   # 輕微過擬合門檻（%）


def _mape_range(model, scaler, df, data, target_col, lo, hi) -> float:
    """計算 [lo, hi) 視窗區間的還原價格 MAPE。"""
    w = cfg.WINDOW_SIZE
    errs = []
    for i in range(lo, hi):
        x = torch.tensor(data[i:i+w], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p = model(x).cpu().numpy()[0, 0]
        dummy = np.zeros((1, data.shape[1]), dtype=np.float32)
        dummy[0, target_col] = p
        pred_pct = float(scaler.inverse_transform(dummy)[0, target_col])
        prev = float(df["Close"].iloc[i + w - 1])
        pred = prev * (1 + pred_pct / 100)
        act  = float(df["Close"].iloc[i + w])
        errs.append(abs(act - pred) / act * 100)
    return float(np.mean(errs)) if errs else float("nan")


def check_ticker(ticker: str, model_type: str = "lstm") -> dict:
    """回傳單支股票的訓練/測試 MAPE 與過擬合判定。"""
    ckpt_path   = os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_{model_type}_best.pt")
    scaler_path = os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_scaler.pkl")
    csv_path    = os.path.join(cfg.RAW_DATA_DIR, f"{ticker}.csv")
    if not all(os.path.exists(p) for p in [ckpt_path, scaler_path, csv_path]):
        raise FileNotFoundError(f"{ticker} 缺少模型或資料")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = get_model(model_type, ckpt["input_size"], cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    target_col    = ckpt.get("target_col", 0)
    feature_names = ckpt.get("feature_names", [])

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df[[c for c in feature_names if c in df.columns]].dropna()
    data = scaler.transform(df.values.astype(np.float32))

    w     = cfg.WINDOW_SIZE
    split = int(len(data) * cfg.TRAIN_RATIO)

    train_mape = _mape_range(model, scaler, df, data, target_col, 0, split - w)
    test_mape  = _mape_range(model, scaler, df, data, target_col, split, len(data) - w)
    gap = test_mape - train_mape

    if gap > WARN_GAP:
        verdict = "warning"
    elif gap > MILD_GAP:
        verdict = "mild"
    else:
        verdict = "healthy"

    return {
        "ticker":     ticker,
        "name":       cfg.ALL_STOCKS.get(ticker, ""),
        "train_mape": round(train_mape, 2),
        "test_mape":  round(test_mape, 2),
        "gap":        round(gap, 2),
        "verdict":    verdict,
    }


def check_all(model_type: str = "lstm") -> pd.DataFrame:
    """對所有股票執行過擬合檢測。"""
    rows = []
    for ticker in cfg.ALL_STOCKS:
        try:
            rows.append(check_ticker(ticker, model_type))
        except Exception as e:
            print(f"  [跳過] {ticker}: {e}")
    return pd.DataFrame(rows)


_VERDICT_LABEL = {"healthy": "健康", "mild": "輕微", "warning": "⚠️ 過擬合"}


def print_report(df: pd.DataFrame, model_type: str):
    """印出過擬合檢測報告。"""
    if df.empty:
        print("無檢測結果")
        return
    print(f"\n{'='*60}")
    print(f"  過擬合監控 | 模型：{model_type}")
    print(f"  判定：差距<2% 健康 | 2~4% 輕微 | >4% 警告")
    print(f"{'='*60}")
    print(f"{'股票':<11}{'名稱':<7}{'訓練MAPE':>9}{'測試MAPE':>9}{'差距':>8}  判定")
    print("-" * 60)
    for _, r in df.iterrows():
        print(f"{r['ticker']:<11}{r['name']:<7}{r['train_mape']:>8.2f}%"
              f"{r['test_mape']:>8.2f}%{r['gap']:>+7.2f}%  {_VERDICT_LABEL[r['verdict']]}")
    print("-" * 60)
    n_warn = int((df["verdict"] == "warning").sum())
    n_mild = int((df["verdict"] == "mild").sum())
    print(f"健康 {len(df)-n_warn-n_mild} | 輕微 {n_mild} | 警告 {n_warn}")
    if n_warn:
        warn_list = df[df["verdict"] == "warning"]["ticker"].tolist()
        print(f"⚠️ 需關注（建議簡化模型或減特徵）：{', '.join(warn_list)}")
    print(f"{'='*60}\n")


def to_wiki_lines(df: pd.DataFrame) -> list:
    """產生寫入 Wiki 的 markdown 行。"""
    if df.empty:
        return []
    lines = ["\n## 過擬合監控\n",
             "| 股票 | 名稱 | 訓練MAPE | 測試MAPE | 差距 | 判定 |",
             "|------|------|---------|---------|------|------|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['ticker']} | {r['name']} | {r['train_mape']}% "
                     f"| {r['test_mape']}% | {r['gap']:+.2f}% | {_VERDICT_LABEL[r['verdict']]} |")
    n_warn = int((df["verdict"] == "warning").sum())
    lines.append(f"\n**過擬合警告：{n_warn}/{len(df)} 支**")
    if n_warn:
        warn_list = df[df["verdict"] == "warning"]["ticker"].tolist()
        lines.append(f"⚠️ 需關注：{', '.join(warn_list)}（建議簡化模型或減特徵）")
    return lines


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="過擬合監控")
    parser.add_argument("--model", default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--save", action="store_true", help="結果寫入 Wiki")
    args = parser.parse_args()

    df = check_all(args.model)
    print_report(df, args.model)
    if args.save and not df.empty:
        path = os.path.join(cfg.WIKI_DIR, f"overfit_{date.today().isoformat()}_{args.model}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join([f"# 過擬合監控 {date.today().isoformat()}"] + to_wiki_lines(df)))
        print(f"報告已存：{path}")
