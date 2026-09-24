"""
視覺化腳本：盤後產生兩張圖表
  圖1：預測 vs 實際收盤價折線圖（測試集 + 每日累積記錄）
  圖2：各股票 MAPE 熱力圖

用法：
  python visualize.py               # 產生今日圖表
  python visualize.py --days 60     # 顯示近 60 日
  python visualize.py --show        # 產生後直接開啟預覽
"""
import sys
import os
import argparse
from datetime import datetime, date

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ALL_STOCKS, WIKI_DIR

LOG_PATH  = os.path.join(WIKI_DIR, "prediction_log.csv")
CHART_DIR = os.path.join(WIKI_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

STOCK_NAMES = {k.replace(".TWO","").replace(".TW",""): v for k, v in ALL_STOCKS.items()}


# ── 舊圖清理 ──────────────────────────────────────────────────────────────────
def cleanup_old_charts(keep_days: int = 7) -> int:
    """刪除 charts 目錄中超過最近 keep_days 個日期的圖檔，回傳刪除張數。

    依檔名前 10 碼（YYYY-MM-DD）分組，只保留最新的 keep_days 個日期。
    """
    import re
    if not os.path.isdir(CHART_DIR):
        return 0
    date_re = re.compile(r"^(\d{4}-\d{2}-\d{2})_")
    by_date = {}
    for fn in os.listdir(CHART_DIR):
        m = date_re.match(fn)
        if m:
            by_date.setdefault(m.group(1), []).append(fn)

    dates_sorted = sorted(by_date.keys(), reverse=True)
    to_remove_dates = dates_sorted[keep_days:]
    removed = 0
    for d in to_remove_dates:
        for fn in by_date[d]:
            try:
                os.remove(os.path.join(CHART_DIR, fn))
                removed += 1
            except OSError:
                pass
    return removed


# ── 中文字體設定 ──────────────────────────────────────────────────────────────
def _setup_font():
    candidates = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei",
                  "PingFang TC", "Noto Sans CJK TC", "Arial Unicode MS"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font] + plt.rcParams.get("font.sans-serif", [])
            plt.rcParams["axes.unicode_minus"] = False
            return font
    # fallback：全部改英文標籤，至少不亂碼
    plt.rcParams["axes.unicode_minus"] = False
    return None

FONT_NAME = _setup_font()


# ── 讀寫每日預測記錄 ──────────────────────────────────────────────────────────

def append_log(date_str: str, predictions: list, actual_prices: dict) -> None:
    """追加今日預測與實際收盤價到 prediction_log.csv。"""
    rows = []
    for p in predictions:
        code = p.get("ticker","").replace(".TWO","").replace(".TW","")
        actual = actual_prices.get(code)
        if actual is None:
            continue
        predicted  = p.get("predicted_close")
        prev_close = p.get("prev_close", p.get("last_close"))
        mape = round(abs(actual - predicted) / actual * 100, 4) if predicted else None
        dir_ok = None
        if prev_close and predicted and actual:
            dir_ok = int((predicted > prev_close) == (actual > prev_close))
        rows.append({"date": date_str, "ticker": code,
                     "name": STOCK_NAMES.get(code, code),
                     "predicted": predicted, "actual": actual,
                     "prev_close": prev_close, "mape": mape,
                     "direction_correct": dir_ok, "source": "live"})
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if os.path.exists(LOG_PATH):
        df_old = pd.read_csv(LOG_PATH)
        df_old = df_old[df_old["date"] != date_str]
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")


def _build_testset_log() -> pd.DataFrame:
    """從訓練好的模型重算測試集預測，補足歷史折線資料。"""
    import torch, pickle
    from config import CHECKPOINT_DIR, RAW_DATA_DIR, WINDOW_SIZE, TRAIN_RATIO
    from models.lstm_model import get_model
    import config as cfg

    DEVICE = torch.device("cpu")
    rows = []

    for ticker, name in ALL_STOCKS.items():
        code = ticker.replace(".TWO","").replace(".TW","")
        ckpt_path   = os.path.join(CHECKPOINT_DIR, f"{ticker}_lstm_best.pt")
        scaler_path = os.path.join(CHECKPOINT_DIR, f"{ticker}_scaler.pkl")
        csv_path    = os.path.join(RAW_DATA_DIR,   f"{ticker}.csv")
        if not all(os.path.exists(p) for p in [ckpt_path, scaler_path, csv_path]):
            continue
        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model = get_model("lstm", ckpt["input_size"], cfg).to(DEVICE)
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

            target_col   = ckpt.get("target_col", 0)
            feature_names = ckpt.get("feature_names", [])

            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            avail = [c for c in feature_names if c in df.columns]
            df = df[avail].dropna()
            data = df.values.astype(np.float32)
            data_scaled = scaler.transform(data)

            split = int(len(data_scaled) * TRAIN_RATIO)
            # 只跑測試集的 windows
            for i in range(split, len(data_scaled) - WINDOW_SIZE):
                x = torch.tensor(data_scaled[i:i+WINDOW_SIZE],
                                 dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    pred_scaled = model(x).cpu().numpy()[0, 0]

                dummy = np.zeros((1, data_scaled.shape[1]), dtype=np.float32)
                dummy[0, target_col] = pred_scaled
                pred_val = float(scaler.inverse_transform(dummy)[0, target_col])

                actual_close = float(df["Close"].iloc[i + WINDOW_SIZE])
                prev_close   = float(df["Close"].iloc[i + WINDOW_SIZE - 1])

                if target_col != 0 and "pct_change" in feature_names:
                    pred_price = prev_close * (1 + pred_val / 100)
                else:
                    pred_price = pred_val

                mape = round(abs(actual_close - pred_price) / actual_close * 100, 4)
                dt   = df.index[i + WINDOW_SIZE]
                rows.append({"date": dt.strftime("%Y-%m-%d"), "ticker": code,
                             "name": name, "predicted": round(pred_price, 2),
                             "actual": actual_close, "prev_close": prev_close,
                             "mape": mape,
                             "direction_correct": int((pred_price > prev_close) == (actual_close > prev_close)),
                             "source": "testset"})
        except Exception as e:
            print(f"  [跳過 {ticker}] {e}")

    return pd.DataFrame(rows)


def load_log(days: int = 60) -> pd.DataFrame:
    """讀取記錄：若 live 資料不足 5 天，自動補測試集歷史資料。"""
    live_df = pd.DataFrame()
    if os.path.exists(LOG_PATH):
        live_df = pd.read_csv(LOG_PATH, parse_dates=["date"])

    # live 資料天數
    live_days = live_df["date"].nunique() if not live_df.empty else 0

    if live_days < 5:
        print("  live 記錄不足 5 天，載入測試集歷史預測補充資料...")
        test_df = _build_testset_log()
        if not test_df.empty:
            test_df["date"] = pd.to_datetime(test_df["date"])
        df = pd.concat([test_df, live_df], ignore_index=True) if not test_df.empty else live_df
    else:
        df = live_df

    if df.empty:
        return df

    df["ticker"] = df["ticker"].astype(str)
    df = df.sort_values("date")
    if days > 0:
        cutoff = df["date"].max() - pd.Timedelta(days=days)
        df = df[df["date"] >= cutoff]
    return df.drop_duplicates(subset=["date","ticker"])


# ── 圖1：折線圖 ───────────────────────────────────────────────────────────────

def plot_line_chart(df: pd.DataFrame, save_path: str, show: bool = False) -> str:
    """預測 vs 實際收盤價折線圖，每股一個子圖。"""
    if df.empty:
        return save_path

    tickers = sorted(df["ticker"].unique())
    n    = len(tickers)
    cols = 2
    rows = (n + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3.8))
    fig.suptitle("預測 vs 實際收盤價" if FONT_NAME else "Predicted vs Actual Close",
                 fontsize=14, fontweight="bold", y=1.005)
    axes_flat = axes.flatten() if n > 1 else [axes]

    for i, ticker in enumerate(tickers):
        ax   = axes_flat[i]
        sub  = df[df["ticker"] == ticker].sort_values("date")
        name = STOCK_NAMES.get(ticker, ticker)
        label_actual = "實際" if FONT_NAME else "Actual"
        label_pred   = "預測" if FONT_NAME else "Predicted"

        ax.plot(sub["date"], sub["actual"],
                label=label_actual, color="#1976D2", linewidth=1.8, zorder=3)
        ax.fill_between(sub["date"], sub["actual"],
                        alpha=0.08, color="#1976D2")
        ax.plot(sub["date"], sub["predicted"],
                label=label_pred, color="#E53935", linewidth=1.4,
                linestyle="--", zorder=2)

        title = f"{ticker}  {name}" if FONT_NAME else ticker
        ax.set_title(title, fontsize=9.5, pad=4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        n_ticks = min(8, len(sub))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=n_ticks))
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.7)
        ax.grid(True, alpha=0.25, linestyle=":")

        avg_mape = sub["mape"].mean()
        if not pd.isna(avg_mape):
            mape_label = f"avg MAPE {avg_mape:.1f}%"
            ax.text(0.99, 0.04, mape_label, transform=ax.transAxes,
                    ha="right", fontsize=7.5, color="#555",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8))

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if show:
        os.startfile(save_path)
    return save_path


# ── 圖2：MAPE 熱力圖 ─────────────────────────────────────────────────────────

def plot_mape_heatmap(df: pd.DataFrame, save_path: str, show: bool = False) -> str:
    """MAPE 熱力圖（股票 × 日期），每格顯示數值，越綠越準。"""
    if df.empty:
        return save_path

    pivot = df.pivot_table(index="ticker", columns="date", values="mape", aggfunc="mean")

    # 最多顯示 30 天（太多欄位會擠）
    if len(pivot.columns) > 30:
        pivot = pivot.iloc[:, -30:]

    pivot.columns = [d.strftime("%m/%d") if hasattr(d, "strftime")
                     else str(d)[-5:] for d in pivot.columns]

    if FONT_NAME:
        row_labels = [f"{t}  {STOCK_NAMES.get(t,'')}" for t in pivot.index]
    else:
        row_labels = list(pivot.index)
    pivot.index = row_labels

    cell_w = max(0.65, 10 / max(len(pivot.columns), 1))
    fig_w  = max(10, len(pivot.columns) * cell_w + 3)
    fig_h  = max(5, len(pivot) * 0.8 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn_r",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        linecolor="#ccc",
        cbar_kws={"label": "MAPE (%)", "shrink": 0.8},
        vmin=0, vmax=15,
        annot_kws={"size": 8, "weight": "bold"},
    )

    title = "MAPE 熱力圖（越綠越準確）" if FONT_NAME else "MAPE Heatmap (greener = more accurate)"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("日期" if FONT_NAME else "Date", fontsize=9, labelpad=6)
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8.5, rotation=0)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if show:
        os.startfile(save_path)
    return save_path


# ── 主入口 ────────────────────────────────────────────────────────────────────

def generate_charts(days: int = 60, show: bool = False) -> tuple:
    """讀取記錄並產生兩張圖，回傳 (折線圖路徑, 熱力圖路徑)。"""
    df = load_log(days=days)
    if df.empty:
        print("  無資料，無法產生圖表")
        return "", ""

    today      = date.today().strftime("%Y-%m-%d")
    line_path  = os.path.join(CHART_DIR, f"{today}_linechart.png")
    heat_path  = os.path.join(CHART_DIR, f"{today}_heatmap.png")

    plot_line_chart(df, line_path, show=show)
    plot_mape_heatmap(df, heat_path, show=show)

    return line_path, heat_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="產生預測視覺化圖表")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    print(f"字體：{FONT_NAME or '(fallback，標籤改英文)'}")
    print(f"產生近 {args.days} 日圖表...")
    l, h = generate_charts(days=args.days, show=args.show)
    if l:
        print(f"折線圖：{l}")
        print(f"熱力圖：{h}")
        print("完成")
