"""實驗：5 日後方向預測 LSTM。目標=未來5日累積報酬，量測測試集方向命中率。
不動生產程式，純研究比較。"""
import sys, os, pickle
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import config as cfg
from models.lstm_model import get_model
from models.train import FEATURE_COLS

DEVICE = torch.device("cpu")
HORIZON = 5
W = cfg.WINDOW_SIZE


def run_stock(ticker):
    df = pd.read_csv(f"data/raw/{ticker}.csv", index_col=0, parse_dates=True)
    feats = [c for c in FEATURE_COLS if c in df.columns]
    df = df[feats].dropna()
    data = df.values.astype(np.float32)
    close = df["Close"].values

    # 目標：未來 HORIZON 日累積報酬（%）
    fwd = np.full(len(close), np.nan, dtype=np.float32)
    for i in range(len(close) - HORIZON):
        fwd[i] = (close[i + HORIZON] / close[i] - 1) * 100

    split = int(len(data) * cfg.TRAIN_RATIO)
    scaler = MinMaxScaler().fit(data[:split])
    ds = scaler.transform(data)

    X, y, yreal = [], [], []
    for i in range(len(ds) - W - HORIZON):
        X.append(ds[i:i+W]); y.append(fwd[i+W]); yreal.append(fwd[i+W])
    X = np.array(X, np.float32); y = np.array(y, np.float32).reshape(-1, 1)
    yreal = np.array(yreal, np.float32)
    # 目標也標準化（穩定訓練）
    ymin, ymax = y[:int(len(y)*cfg.TRAIN_RATIO)].min(), y[:int(len(y)*cfg.TRAIN_RATIO)].max()
    ys = (y - ymin) / (ymax - ymin + 1e-9)

    sp = int(len(X) * cfg.TRAIN_RATIO)
    Xtr, ytr = X[:sp], ys[:sp]
    Xte = X[sp:]; yte_real = yreal[sp:]

    model = get_model("lstm", X.shape[2], cfg).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    crit = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                        batch_size=cfg.BATCH_SIZE, shuffle=True)
    for ep in range(80):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte)).cpu().numpy().ravel()
    pred_real = pred * (ymax - ymin) + ymin     # 還原
    # 方向命中（5日漲跌）
    acc = (np.sign(pred_real) == np.sign(yte_real)).mean() * 100
    base = max((yte_real > 0).mean(), (yte_real < 0).mean()) * 100
    return acc, base


print("5 日後方向預測（LSTM）測試集命中率：")
print(f"{'股票':<10}{'5日方向命中':>10}{'全押漲基準':>10}")
print("-" * 32)
accs, bases = [], []
for tk in cfg.ALL_STOCKS:
    try:
        a, b = run_stock(tk); accs.append(a); bases.append(b)
        print(f"{tk:<10}{a:>9.1f}%{b:>9.1f}%")
    except Exception as e:
        print(f"{tk}: {e}")
print("-" * 32)
print(f"{'平均':<10}{np.mean(accs):>9.1f}%{np.mean(bases):>9.1f}%")
print(f"\n對照：現行 1 日模型方向準確率 56.8%")
print(f"判定：5日命中率若明顯 > 56.8% 且 > 全押漲基準，才算真贏")
