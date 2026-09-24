"""
Regime 漂移監控（Evidently）：偵測「近期市場狀態是否偏離模型訓練時的分布」，
在暴衝/變盤盤勢自動警示「模型可信度下降」，補足純技術面模型的盲區。

設計要點（針對金融時間序列調整，避免 i.i.d. 假設造成天天誤報）：
  1. 只監控「平穩特徵」（報酬率/震盪指標/相對量/同業漲跌幅）——這些不會因
     股價長期趨勢而漂移，只在真正 regime 改變（波動率、動能結構）時才動。
     絕對價格特徵（Close/MA/布林）會隨股價趨勢必然漂移，排除。
  2. 門檻自我校準：用訓練期內部多個視窗的漂移率算出「正常上限」
     （mean + k·std），近期漂移率超過才判定異常——與 CQR 同精神，
     讓警報依每支股票自身的雜訊水準調整，而非武斷固定值。

⚠️ 這是監控工具，不改模型、不需重訓。只回報「現在能不能信模型」。

用法：
  python drift_monitor.py --calibrate     # 校準各股正常漂移門檻（一次/月）
  python drift_monitor.py                  # 檢查今日 regime 並印報告
"""
import sys
import os
import json
import warnings
import argparse
from datetime import date

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

from evidently import Report
from evidently.presets import DataDriftPreset

# 平穩特徵（報酬/震盪/相對量，不受價格長期趨勢影響）
REGIME_FEATURES = [
    "pct_change", "RSI", "Volume_ratio", "MACD_hist",
    "usdtwd_pct", "twii_pct",
    "murata_pct", "tdk_pct", "onsemi_pct", "vishay_pct",
]

RECENT_WINDOW = 20          # 近期視窗（交易日）
CALIB_K       = 1.5         # 門檻 = 訓練期漂移率 mean + K·std
THRESHOLDS_PATH = os.path.join(cfg.WIKI_DIR, "drift_thresholds.json")


def _feats(df: pd.DataFrame) -> list:
    return [c for c in REGIME_FEATURES if c in df.columns]


def _drift_share(reference: pd.DataFrame, current: pd.DataFrame) -> float:
    """回傳 current 相對 reference 的漂移特徵比例（0~1）。"""
    snap = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)
    for m in snap.dict()["metrics"]:
        v = m.get("value")
        if isinstance(v, dict) and "share" in v:
            return float(v["share"])
    return float("nan")


def _load_df(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(cfg.RAW_DATA_DIR, f"{ticker}.csv"),
                     index_col=0, parse_dates=True)
    return df[_feats(df)].dropna()


# ── 校準（一次/月）────────────────────────────────────────────────────────────

def calibrate_ticker(ticker: str, n_windows: int = 20) -> dict:
    """用訓練期內部多個視窗算正常漂移率分布，回傳門檻。"""
    df = _load_df(ticker)
    split = int(len(df) * cfg.TRAIN_RATIO)
    ref = df.iloc[:split]
    w = RECENT_WINDOW

    shares = []
    # 在訓練期內均勻取樣 n_windows 個視窗（與整體訓練分布比）
    starts = np.linspace(w, split - w, n_windows).astype(int)
    for s in starts:
        shares.append(_drift_share(ref, df.iloc[s:s + w]))
    shares = [x for x in shares if not np.isnan(x)]
    mean, std = float(np.mean(shares)), float(np.std(shares))
    threshold = min(1.0, mean + CALIB_K * std)
    return {"ticker": ticker, "normal_mean": round(mean, 3),
            "normal_std": round(std, 3), "threshold": round(threshold, 3)}


def calibrate_all() -> dict:
    """校準全部股票並存檔。"""
    out = {}
    for tk in cfg.ALL_STOCKS:
        try:
            r = calibrate_ticker(tk)
            out[tk] = r
            print(f"  {tk:<10} 正常漂移 {r['normal_mean']:.0%}±{r['normal_std']:.0%} "
                  f"→ 門檻 {r['threshold']:.0%}")
        except Exception as e:
            print(f"  [跳過] {tk}: {e}")
    with open(THRESHOLDS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n門檻已存：{THRESHOLDS_PATH}")
    return out


# ── 每日檢查 ──────────────────────────────────────────────────────────────────

def _load_thresholds() -> dict:
    if os.path.exists(THRESHOLDS_PATH):
        with open(THRESHOLDS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def check_regime() -> dict:
    """檢查每支股票近期 regime 漂移，回傳整體判定。"""
    thresholds = _load_thresholds()
    rows = []
    for tk in cfg.ALL_STOCKS:
        try:
            df = _load_df(tk)
            split = int(len(df) * cfg.TRAIN_RATIO)
            ref = df.iloc[:split]
            share = _drift_share(ref, df.iloc[-RECENT_WINDOW:])
            thr = thresholds.get(tk, {}).get("threshold", 0.7)
            rows.append({"ticker": tk, "name": cfg.ALL_STOCKS[tk],
                         "share": round(share, 3), "threshold": thr,
                         "abnormal": share > thr})
        except Exception as e:
            print(f"  [跳過] {tk}: {e}")
    n_abn = sum(r["abnormal"] for r in rows)
    # 過半股票 regime 異常 → 整體市場狀態異常
    overall = "abnormal" if n_abn >= len(rows) / 2 else ("watch" if n_abn >= 3 else "normal")
    return {"rows": rows, "n_abnormal": n_abn, "total": len(rows), "overall": overall}


def build_wiki_lines(result: dict) -> list:
    """產生寫入 Wiki 的 regime 監控 markdown。"""
    verdict_map = {
        "normal":   "✅ 市場狀態正常，模型可信度正常",
        "watch":    "🟡 部分個股 regime 偏移，留意模型偏誤可能升高",
        "abnormal": "🔴 市場狀態異常（多數個股偏離訓練分布），模型可信度下降，建議降低對點預測的依賴",
    }
    lines = ["\n## Regime 漂移監控（Evidently）\n",
             f"**{verdict_map[result['overall']]}**",
             f"（{result['n_abnormal']}/{result['total']} 支偏離正常分布）\n",
             "| 股票 | 名稱 | 近期漂移率 | 正常門檻 | 狀態 |",
             "|------|------|-----------|---------|------|"]
    for r in sorted(result["rows"], key=lambda x: -x["share"]):
        mark = "🔴 異常" if r["abnormal"] else "✅ 正常"
        lines.append(f"| {r['ticker']} | {r['name']} | {r['share']:.0%} "
                     f"| {r['threshold']:.0%} | {mark} |")
    lines.append("\n_只監控平穩特徵（報酬/震盪/同業）；門檻為各股訓練期正常漂移率自我校準。_")
    return lines


# ── HTML 報告 + 靜態預覽圖 ────────────────────────────────────────────────────

def save_html_report(ticker: str) -> str:
    """生成 Evidently 互動 HTML 漂移報告（瀏覽器開啟）。"""
    from evidently.presets import DataSummaryPreset
    df = _load_df(ticker)
    split = int(len(df) * cfg.TRAIN_RATIO)
    snap = Report([DataDriftPreset(), DataSummaryPreset()]).run(
        current_data=df.iloc[-RECENT_WINDOW:], reference_data=df.iloc[:split])
    out_dir = os.path.join(cfg.WIKI_DIR, "drift_reports")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{date.today().isoformat()}_{ticker}_drift.html")
    snap.save_html(path)
    return path


def plot_drift_preview(ticker: str, save: bool = True) -> str:
    """靜態漂移預覽圖：每個平穩特徵的『訓練期 vs 近期』分布對比（matplotlib）。

    重現 Evidently 漂移報告的核心視覺；紅標=該特徵分布顯著偏移（KS 檢定 p<0.05）。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats
    from visualize import _setup_font
    _setup_font()

    df = _load_df(ticker)
    split = int(len(df) * cfg.TRAIN_RATIO)
    ref, cur = df.iloc[:split], df.iloc[-RECENT_WINDOW:]
    feats = _feats(df)

    ncol = 3
    nrow = (len(feats) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.6 * nrow))
    axes = axes.flatten()
    name = cfg.ALL_STOCKS.get(ticker, ticker)
    n_drift = 0

    for i, col in enumerate(feats):
        ax = axes[i]
        r, c = ref[col].dropna(), cur[col].dropna()
        ks_p = stats.ks_2samp(r, c).pvalue
        drifted = ks_p < 0.05
        n_drift += drifted
        ax.hist(r, bins=30, density=True, alpha=0.55, color="#1976D2", label="訓練期")
        ax.hist(c, bins=15, density=True, alpha=0.65, color="#E53935", label="近20日")
        title_c = "#C62828" if drifted else "#37474F"
        ax.set_title(f"{col} {'[漂移]' if drifted else ''}", fontsize=9, color=title_c,
                     fontweight="bold" if drifted else "normal")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(feats), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{ticker} {name} — Regime 漂移：訓練期 vs 近20日分布"
                 f"（{n_drift}/{len(feats)} 特徵偏移）", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out = ""
    if save:
        from visualize import CHART_DIR
        os.makedirs(CHART_DIR, exist_ok=True)
        out = os.path.join(CHART_DIR, f"{date.today().isoformat()}_{ticker}_drift.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ── 主程式 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Regime 漂移監控（Evidently）")
    parser.add_argument("--calibrate", action="store_true", help="校準各股正常漂移門檻")
    args = parser.parse_args()

    if args.calibrate:
        print("校準各股正常漂移門檻（訓練期內部視窗）...")
        calibrate_all()
        return

    if not os.path.exists(THRESHOLDS_PATH):
        print("尚未校準，先執行 python drift_monitor.py --calibrate")
        return

    result = check_regime()
    print("\n".join(build_wiki_lines(result)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
