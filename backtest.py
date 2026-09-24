"""
回測框架：在測試集（後 20%）上模擬交易，驗證預測策略歷史上是否賺錢。

策略邏輯（long-only，符合台股散戶現實）：
  - 每日收盤模型預測隔日漲跌幅
  - 預測漲幅 > 進場門檻 → 隔日持有（賺/賠隔日實際漲跌）
  - 否則空手（報酬 0）
  - 每次進出場扣交易成本

輸出指標：
  - 策略累積報酬 vs 買進持有（Buy & Hold）
  - 勝率、交易次數
  - 最大回撤（MDD）
  - 年化夏普比率

用法：
  python backtest.py                      # 全部股票，ensemble 模型
  python backtest.py --model lstm
  python backtest.py --ticker 2327.TW
  python backtest.py --threshold 0.5      # 進場門檻 %
  python backtest.py --cost 0.4           # 單次來回交易成本 %
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
from models.train import FEATURE_COLS

DEVICE = torch.device("cpu")

# 台股交易成本：買進手續費 0.1425% + 賣出手續費 0.1425% + 證交稅 0.3% ≈ 0.585%
DEFAULT_COST_PCT  = 0.585
DEFAULT_THRESHOLD = 0.3   # 預測漲幅超過此值才進場（%）
TRADING_DAYS_YEAR = 252


# ── 測試集預測 ────────────────────────────────────────────────────────────────

def _predict_testset(ticker: str, model_type: str, segment: str = "full"):
    """重跑模型於測試集，回傳 DataFrame(date, predicted_pct, actual_pct)。

    segment："full"=整個測試段（回測用）；
             "cal"=測試段前半（訓練時挑 checkpoint 用的那半）；
             "holdout"=測試段後半（挑選過程沒看過 → 閘門誠實評估用，防選擇偏誤）。"""
    csv_path    = os.path.join(cfg.RAW_DATA_DIR, f"{ticker}.csv")
    ckpt_path   = os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_{model_type}_best.pt")
    scaler_path = os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_scaler.pkl")

    if not all(os.path.exists(p) for p in [csv_path, ckpt_path, scaler_path]):
        raise FileNotFoundError(f"{ticker} 缺少 {model_type} 模型或資料")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = get_model(model_type, ckpt["input_size"], cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    target_col    = ckpt.get("target_col", 0)
    feature_names = ckpt.get("feature_names", [])

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    avail = [c for c in feature_names if c in df.columns]
    df = df[avail].dropna()
    data = df.values.astype(np.float32)
    data_scaled = scaler.transform(data)

    w     = cfg.WINDOW_SIZE
    split = int(len(data_scaled) * cfg.TRAIN_RATIO)
    end   = len(data_scaled) - w
    cal_end = split + (end - split) // 2      # 與 quantile_forecast 同一切法
    if segment == "cal":
        lo, hi = split, cal_end
    elif segment == "holdout":
        lo, hi = cal_end, end
    else:
        lo, hi = split, end

    rows = []
    for i in range(lo, hi):
        x = torch.tensor(data_scaled[i:i+w], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_scaled = model(x).cpu().numpy()[0, 0]
        dummy = np.zeros((1, data_scaled.shape[1]), dtype=np.float32)
        dummy[0, target_col] = pred_scaled
        pred_pct = float(scaler.inverse_transform(dummy)[0, target_col])

        actual_pct = float(df["pct_change"].iloc[i + w])
        dt = df.index[i + w]
        rows.append({"date": dt, "predicted_pct": pred_pct, "actual_pct": actual_pct})

    return pd.DataFrame(rows)


def _predict_testset_ensemble(ticker: str, segment: str = "full"):
    """Ensemble：LSTM 與 Transformer 測試集預測動態加權平均。"""
    lstm_df = _predict_testset(ticker, "lstm", segment)
    try:
        tf_df = _predict_testset(ticker, "transformer", segment)
    except FileNotFoundError:
        return lstm_df

    # 動態權重（val_loss 越低權重越高）
    lstm_val = torch.load(os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_lstm_best.pt"),
                          map_location=DEVICE)["val_loss"]
    tf_val   = torch.load(os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_transformer_best.pt"),
                          map_location=DEVICE)["val_loss"]
    total = lstm_val + tf_val
    lw, tw = tf_val / total, lstm_val / total

    merged = lstm_df.merge(tf_df, on="date", suffixes=("_l", "_t"))
    merged["predicted_pct"] = merged["predicted_pct_l"] * lw + merged["predicted_pct_t"] * tw
    merged["actual_pct"]    = merged["actual_pct_l"]
    return merged[["date", "predicted_pct", "actual_pct"]]


# ── 回測引擎 ──────────────────────────────────────────────────────────────────

def backtest_ticker(ticker: str, model_type: str = "ensemble",
                    threshold: float = DEFAULT_THRESHOLD,
                    cost_pct: float = DEFAULT_COST_PCT) -> dict:
    """對單支股票執行回測，回傳績效指標。"""
    if model_type == "ensemble":
        pred_df = _predict_testset_ensemble(ticker)
    else:
        pred_df = _predict_testset(ticker, model_type)

    if pred_df.empty:
        return {}

    # 策略訊號：預測漲幅超過門檻才進場
    pred_df = pred_df.copy()
    pred_df["signal"]   = (pred_df["predicted_pct"] > threshold).astype(int)
    # 進出場變化扣交易成本（持倉變動的日子）
    pred_df["position_change"] = pred_df["signal"].diff().abs().fillna(pred_df["signal"])
    # 策略日報酬 = 持有時的實際漲跌 - 交易成本
    pred_df["strategy_ret"] = (
        pred_df["signal"] * pred_df["actual_pct"]
        - pred_df["position_change"] * cost_pct
    )

    # 累積報酬（複利）
    pred_df["strategy_cum"]  = (1 + pred_df["strategy_ret"] / 100).cumprod()
    pred_df["buyhold_cum"]   = (1 + pred_df["actual_pct"]   / 100).cumprod()

    # ── 指標計算 ──
    n_days = len(pred_df)
    trades = pred_df[pred_df["signal"] == 1]
    n_trades = int(trades.shape[0])
    wins = int((trades["actual_pct"] > 0).sum())
    win_rate = wins / n_trades if n_trades else 0.0

    strategy_total = (pred_df["strategy_cum"].iloc[-1] - 1) * 100
    buyhold_total  = (pred_df["buyhold_cum"].iloc[-1]  - 1) * 100

    # 最大回撤
    cum = pred_df["strategy_cum"].values
    running_max = np.maximum.accumulate(cum)
    drawdown = (cum - running_max) / running_max
    max_dd = float(drawdown.min() * 100)

    # 年化夏普（日報酬）
    daily_ret = pred_df["strategy_ret"].values / 100
    if daily_ret.std() > 1e-9:
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS_YEAR))
    else:
        sharpe = 0.0

    return {
        "ticker":         ticker,
        "name":           cfg.ALL_STOCKS.get(ticker, ""),
        "test_days":      n_days,
        "n_trades":       n_trades,
        "win_rate":       round(win_rate * 100, 1),
        "strategy_return": round(strategy_total, 2),
        "buyhold_return":  round(buyhold_total, 2),
        "excess_return":   round(strategy_total - buyhold_total, 2),
        "max_drawdown":    round(max_dd, 2),
        "sharpe":          round(sharpe, 2),
    }


def backtest_all(model_type: str = "ensemble",
                 threshold: float = DEFAULT_THRESHOLD,
                 cost_pct: float = DEFAULT_COST_PCT) -> pd.DataFrame:
    """對所有股票回測並輸出彙總表。"""
    results = []
    for ticker in cfg.ALL_STOCKS:
        try:
            r = backtest_ticker(ticker, model_type, threshold, cost_pct)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  [跳過] {ticker}: {e}")
    return pd.DataFrame(results)


def print_report(df: pd.DataFrame, model_type: str, threshold: float, cost_pct: float):
    """印出回測彙總報告。"""
    if df.empty:
        print("無回測結果")
        return

    print(f"\n{'='*78}")
    print(f"  回測報告 | 模型：{model_type} | 進場門檻：{threshold}% | 交易成本：{cost_pct}%")
    print(f"{'='*78}")
    header = (f"{'股票':<11}{'名稱':<7}{'交易數':>6}{'勝率':>8}"
             f"{'策略報酬':>10}{'買持報酬':>10}{'超額':>9}{'最大回撤':>10}{'夏普':>7}")
    print(header)
    print("-" * 78)

    for _, r in df.iterrows():
        print(f"{r['ticker']:<11}{r['name']:<7}{r['n_trades']:>6}"
              f"{r['win_rate']:>7.1f}%{r['strategy_return']:>9.1f}%"
              f"{r['buyhold_return']:>9.1f}%{r['excess_return']:>8.1f}%"
              f"{r['max_drawdown']:>9.1f}%{r['sharpe']:>7.2f}")

    print("-" * 78)
    # 彙總
    avg_win   = df["win_rate"].mean()
    avg_strat = df["strategy_return"].mean()
    avg_bh    = df["buyhold_return"].mean()
    avg_excess= df["excess_return"].mean()
    avg_dd    = df["max_drawdown"].mean()
    avg_sharpe= df["sharpe"].mean()
    n_beat    = int((df["excess_return"] > 0).sum())

    print(f"{'平均':<11}{'':<7}{'':>6}{avg_win:>7.1f}%{avg_strat:>9.1f}%"
          f"{avg_bh:>9.1f}%{avg_excess:>8.1f}%{avg_dd:>9.1f}%{avg_sharpe:>7.2f}")
    print(f"\n策略打敗買進持有：{n_beat}/{len(df)} 支")
    print(f"平均勝率：{avg_win:.1f}%  平均夏普：{avg_sharpe:.2f}")
    print(f"{'='*78}\n")


def save_report(df: pd.DataFrame, model_type: str):
    """將回測結果存到 Wiki。"""
    if df.empty:
        return
    path = os.path.join(cfg.WIKI_DIR, f"backtest_{date.today().isoformat()}_{model_type}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"回測明細已存：{path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="預測策略回測")
    parser.add_argument("--model", default="ensemble", choices=["lstm", "transformer", "ensemble"])
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--cost", type=float, default=DEFAULT_COST_PCT)
    parser.add_argument("--save", action="store_true", help="存回測明細到 Wiki")
    args = parser.parse_args()

    if args.ticker:
        r = backtest_ticker(args.ticker, args.model, args.threshold, args.cost)
        print_report(pd.DataFrame([r]), args.model, args.threshold, args.cost)
    else:
        df = backtest_all(args.model, args.threshold, args.cost)
        print_report(df, args.model, args.threshold, args.cost)
        if args.save:
            save_report(df, args.model)
