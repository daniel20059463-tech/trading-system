"""
回填驗證：用正確的歷史美股日期 + 正確的 as-of 預測，
重算每個交易日的「國際盤 → 台股預測 vs 實際」，補上之前因 bug 失真的記錄。

- 美股：用 yfinance 歷史資料，挑「該台股交易日前最後一個美股交易日」的真實漲跌
- 預測：用「該日前一交易日為止」的資料預測該日（as-of，時間對齊）
- 實際：CSV 該日真實收盤
"""
import sys
import os
import pickle
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from models.lstm_model import get_model

DEVICE = torch.device("cpu")

# 美股關鍵領先指標
US_TICKERS = {"^SOX": "費半", "TSM": "台積電ADR", "^IXIC": "那斯達克"}


def _us_change_for(taiwan_day: pd.Timestamp, us_hist: dict) -> dict:
    """取『台股交易日前最後一個美股交易日』的美股漲跌。"""
    out = {}
    for t, name in US_TICKERS.items():
        s = us_hist[t]
        # 美股日期 < 台股當日（台股早上開盤時，美股最近收盤是前一日）
        prior = s[s.index.date < taiwan_day.date()]
        if len(prior) >= 2:
            pct = (prior.iloc[-1] / prior.iloc[-2] - 1) * 100
            out[name] = (round(float(pct), 2), prior.index[-1].date())
    return out


def _predict_asof(ticker: str, upto_idx: int):
    """用 df 到 upto_idx（含）為止的資料，預測 upto_idx+1 天，回傳 ensemble 漲跌幅。"""
    models = {}
    for mt in ("lstm", "transformer"):
        p = os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_{mt}_best.pt")
        if not os.path.exists(p):
            continue
        ck = torch.load(p, map_location=DEVICE)
        m = get_model(mt, ck["input_size"], cfg).to(DEVICE)
        m.load_state_dict(ck["model_state"]); m.eval()
        models[mt] = (m, ck["val_loss"], ck.get("target_col", 0), ck.get("feature_names", []))
    if not models:
        return None
    _, _, tc, fn = next(iter(models.values()))
    sc = pickle.load(open(os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_scaler.pkl"), "rb"))

    df = pd.read_csv(os.path.join(cfg.RAW_DATA_DIR, f"{ticker}.csv"), index_col=0, parse_dates=True)
    df = df[[c for c in fn if c in df.columns]].dropna()
    w = cfg.WINDOW_SIZE
    if upto_idx + 1 < w:
        return None
    window = df.values[upto_idx + 1 - w: upto_idx + 1].astype(np.float32)
    scaled = sc.transform(window)
    x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
    preds, wts = [], []
    for mt, (m, val, _tc, _fn) in models.items():
        with torch.no_grad():
            p = m(x).numpy()[0, 0]
        d = np.zeros((1, scaled.shape[1]), dtype=np.float32); d[0, tc] = p
        preds.append(float(sc.inverse_transform(d)[0, tc])); wts.append(1.0 / (val + 1e-9))
    wts = np.array(wts) / sum(wts)
    return float(np.dot(preds, wts))


def backfill(start="2026-06-05", end="2026-06-10"):
    """回填區間內每個交易日的驗證記錄。"""
    # 預抓美股歷史（一次）
    us_hist = {}
    for t in US_TICKERS:
        us_hist[t] = yf.Ticker(t).history(start="2026-05-25", end="2026-06-12",
                                          auto_adjust=True)["Close"].dropna()
        us_hist[t].index = us_hist[t].index.tz_localize(None)

    ref = pd.read_csv(os.path.join(cfg.RAW_DATA_DIR, "2327.TW.csv"), index_col=0, parse_dates=True).dropna()
    days = [d for d in ref.index if start <= d.date().isoformat() <= end]

    print(f"{'='*70}")
    print(f"  回填驗證 {start} ~ {end}")
    print(f"{'='*70}\n")

    for day in days:
        # 美股（正確歷史日期）
        us = _us_change_for(day, us_hist)
        us_str = "  ".join(f"{nm} {v:+.1f}%({d})" for nm, (v, d) in us.items())

        # 全股預測 vs 實際
        mapes, dir_ok, n = [], 0, 0
        rows = []
        for tk, name in cfg.ALL_STOCKS.items():
            df = pd.read_csv(os.path.join(cfg.RAW_DATA_DIR, f"{tk}.csv"), index_col=0, parse_dates=True).dropna()
            if day not in df.index:
                continue
            idx = df.index.get_loc(day)
            if idx < 1:
                continue
            pct = _predict_asof(tk, idx - 1)   # 用前一日為止預測 day
            if pct is None:
                continue
            prev = float(df["Close"].iloc[idx - 1]); actual = float(df["Close"].iloc[idx])
            pred = prev * (1 + pct / 100)
            mape = abs(actual - pred) / actual * 100; mapes.append(mape)
            pdir = pct >= 0; adir = actual > prev
            ok = (pdir == adir); dir_ok += ok; n += 1
            rows.append((tk, name, pred, actual, mape, "✓" if ok else "✗"))

        print(f"── {day.date()} ──")
        print(f"  🌍 昨夜美股：{us_str}")
        if mapes:
            print(f"  📊 預測績效：平均MAPE {np.mean(mapes):.2f}%  方向 {dir_ok}/{n}={dir_ok/n*100:.0f}%")
        print()

    print(f"{'='*70}")
    print("回填完成（美股已用正確歷史日期，預測已時間對齊）")
    print(f"{'='*70}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-06-05")
    ap.add_argument("--end", default="2026-06-10")
    args = ap.parse_args()
    backfill(args.start, args.end)
