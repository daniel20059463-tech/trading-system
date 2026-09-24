"""
模型品質閘門 Harness：在測試集（後 20%）量測每支股票的「漲跌方向準確率」，
以 50%（丟硬幣 baseline）為及格線。低於 50% 代表模型比亂猜還差，必須重訓。

設計原則：
  • 純本地 PyTorch 評估與重訓，完全不呼叫任何 LLM API（零 token 花費）。
  • 重訓使用方向感知 loss（train.py 的 direction_weight），直接優化漲跌方向，
    而非只看價格幅度的 MSE。
  • 不退步保護（don't-regress）：重訓後方向準確率沒進步就還原舊模型，
    確保每次重訓只會變好或持平，不會把能用的模型弄壞。

用法：
  python eval_harness.py                  # 只評估，印出 + 寫入 Wiki
  python eval_harness.py --retrain        # 評估後自動重訓所有低於 50% 的股票
  python eval_harness.py --gate 55        # 自訂及格線（預設 50）
  python eval_harness.py --no-save        # 不寫入 Wiki
"""
import sys
import os
import shutil
import argparse
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from backtest import _predict_testset, _predict_testset_ensemble

GATE_DEFAULT = 50.0          # 及格線：方向準確率須 >= 此值（%）
DIRECTION_WEIGHT = 0.3       # 重訓時的方向感知 loss 權重
RETRAIN_MODELS = ("lstm", "transformer")


# ── 評估 ──────────────────────────────────────────────────────────────────────

def _direction_accuracy(df: pd.DataFrame) -> float:
    """由 (predicted_pct, actual_pct) 計算漲跌方向準確率（%）。"""
    if df.empty:
        return float("nan")
    pred_up = df["predicted_pct"] > 0
    act_up = df["actual_pct"] > 0
    return float((pred_up == act_up).mean() * 100)


def evaluate_ticker(ticker: str) -> dict:
    """評估單支股票 LSTM / Transformer / Ensemble 的方向準確率。

    F1 修正（Fable 稽核）：只評「測試段後半 holdout」——訓練挑 checkpoint 用的是
    前半（cal），若在同一段上報成績，等於 200 個 epoch 裡挑最好的再自評，
    數字必然虛高（曾出現 48.1→61.5 的假進步）。holdout 沒被挑選過程看過，才誠實。"""
    out = {"ticker": ticker, "name": cfg.ALL_STOCKS.get(ticker, "")}
    for mt in ("lstm", "transformer"):
        try:
            out[mt] = round(_direction_accuracy(_predict_testset(ticker, mt, segment="holdout")), 1)
        except Exception:
            out[mt] = None
    try:
        ens_df = _predict_testset_ensemble(ticker, segment="holdout")
        out["ensemble"] = round(_direction_accuracy(ens_df), 1)
        out["samples"] = len(ens_df)
    except Exception as e:
        out["ensemble"] = None
        out["samples"] = 0
        out["error"] = str(e)
    return out


def evaluate_all(gate: float = GATE_DEFAULT) -> pd.DataFrame:
    """評估所有股票，回傳含及格判定的 DataFrame（以 ensemble 為準）。"""
    rows = []
    for ticker in cfg.ALL_STOCKS:
        r = evaluate_ticker(ticker)
        ens = r.get("ensemble")
        r["verdict"] = "pass" if (ens is not None and ens >= gate) else "fail"
        rows.append(r)
    return pd.DataFrame(rows)


# ── 重訓（不退步保護）──────────────────────────────────────────────────────────

def _ckpt_paths(ticker: str) -> list:
    """回傳該股票所有 checkpoint 檔路徑（lstm/transformer）。"""
    return [os.path.join(cfg.CHECKPOINT_DIR, f"{ticker}_{mt}_best.pt")
            for mt in RETRAIN_MODELS]


def retrain_ticker(ticker: str, gate: float = GATE_DEFAULT) -> dict:
    """以方向感知 loss 重訓單支股票，含不退步保護。

    流程：備份舊模型 → 量測舊方向準確率 → 重訓 → 量測新準確率 →
          沒進步則還原舊模型。
    """
    from models.train import train_model

    name = cfg.ALL_STOCKS.get(ticker, "")
    print(f"\n{'='*60}")
    print(f"[重訓] {ticker} {name} | 方向感知 loss（dir_weight={DIRECTION_WEIGHT}）")
    print(f"{'='*60}")

    before = evaluate_ticker(ticker).get("ensemble")
    print(f"  重訓前 ensemble 方向準確率：{before}%")

    # 備份舊 checkpoint
    backups = {}
    for p in _ckpt_paths(ticker):
        if os.path.exists(p):
            bak = p + ".bak"
            shutil.copy2(p, bak)
            backups[p] = bak

    # 重訓（方向感知 + 以驗證方向準確率選最佳 checkpoint）
    for mt in RETRAIN_MODELS:
        try:
            train_model(ticker, model_type=mt,
                        direction_weight=DIRECTION_WEIGHT, select_metric="direction")
        except Exception as e:
            print(f"  [{mt}] 重訓失敗：{e}")

    after = evaluate_ticker(ticker).get("ensemble")
    print(f"  重訓後 ensemble 方向準確率：{after}%")

    # 不退步保護
    improved = (after is not None and before is not None and after >= before)
    if improved:
        for bak in backups.values():
            try:
                os.remove(bak)
            except OSError:
                pass
        status = "kept_new"
        print(f"  ✅ 採用新模型（{before}% → {after}%）")
    else:
        for p, bak in backups.items():
            shutil.copy2(bak, p)
            os.remove(bak)
        status = "reverted"
        after = before
        print(f"  ↩️ 新模型未進步，已還原舊模型（維持 {before}%）")

    return {
        "ticker": ticker, "name": name,
        "before": before, "after": after,
        "passed": (after is not None and after >= gate),
        "status": status,
    }


# ── 報告 ──────────────────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame, gate: float):
    """印出評估報告。"""
    print(f"\n{'='*64}")
    print(f"  模型品質閘門 | 測試集漲跌方向準確率（及格線 {gate:.0f}%）")
    print(f"{'='*64}")
    print(f"{'股票':<11}{'名稱':<7}{'LSTM':>7}{'TF':>7}{'Ensemble':>10}  判定")
    print("-" * 64)
    for _, r in df.iterrows():
        ls = f"{r['lstm']:.0f}%" if r.get("lstm") is not None else "—"
        tf = f"{r['transformer']:.0f}%" if r.get("transformer") is not None else "—"
        en = f"{r['ensemble']:.1f}%" if r.get("ensemble") is not None else "—"
        mark = "✅" if r["verdict"] == "pass" else "❌ 需重訓"
        print(f"{r['ticker']:<11}{r['name']:<7}{ls:>7}{tf:>7}{en:>10}  {mark}")
    print("-" * 64)
    ens_vals = [r for r in df["ensemble"] if r is not None]
    n_fail = int((df["verdict"] == "fail").sum())
    print(f"平均 ensemble 方向準確率：{np.mean(ens_vals):.1f}%"
          if ens_vals else "無有效結果")
    print(f"及格 {len(df)-n_fail}/{len(df)} 支 | 需重訓 {n_fail} 支")
    if n_fail:
        fails = df[df["verdict"] == "fail"]["ticker"].tolist()
        print(f"❌ 低於 {gate:.0f}%：{', '.join(fails)}")
    print(f"{'='*64}\n")


def to_wiki_lines(df: pd.DataFrame, gate: float, retrain_log: list = None) -> list:
    """產生寫入 Wiki 的 markdown 行。"""
    lines = [
        f"# 模型品質閘門 {date.today().isoformat()}",
        f"\n測試集漲跌方向準確率（及格線 {gate:.0f}% = 丟硬幣 baseline）\n",
        "| 股票 | 名稱 | LSTM | Transformer | Ensemble | 判定 |",
        "|------|------|------|-------------|----------|------|",
    ]
    for _, r in df.iterrows():
        ls = f"{r['lstm']:.0f}%" if r.get("lstm") is not None else "—"
        tf = f"{r['transformer']:.0f}%" if r.get("transformer") is not None else "—"
        en = f"{r['ensemble']:.1f}%" if r.get("ensemble") is not None else "—"
        mark = "✅ 及格" if r["verdict"] == "pass" else "❌ 需重訓"
        lines.append(f"| {r['ticker']} | {r['name']} | {ls} | {tf} | {en} | {mark} |")
    ens_vals = [r for r in df["ensemble"] if r is not None]
    n_fail = int((df["verdict"] == "fail").sum())
    lines.append(f"\n**平均方向準確率：{np.mean(ens_vals):.1f}% | "
                 f"及格 {len(df)-n_fail}/{len(df)} 支**")
    if retrain_log:
        lines.append("\n## 重訓記錄（方向感知 loss，不退步保護）\n")
        lines.append("| 股票 | 重訓前 | 重訓後 | 結果 |")
        lines.append("|------|--------|--------|------|")
        for r in retrain_log:
            st = {"kept_new": "✅ 採用新模型", "reverted": "↩️ 還原舊模型"}.get(r["status"], r["status"])
            lines.append(f"| {r['ticker']} {r['name']} | {r['before']}% | {r['after']}% | {st} |")
    return lines


def save_to_wiki(df: pd.DataFrame, gate: float, retrain_log: list = None) -> str:
    """寫入 Wiki。"""
    path = os.path.join(cfg.WIKI_DIR, f"eval_harness_{date.today().isoformat()}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(to_wiki_lines(df, gate, retrain_log)))
    return path


# ── 主程式 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="模型品質閘門 Harness（方向準確率）")
    parser.add_argument("--retrain", action="store_true", help="評估後自動重訓低於及格線的股票")
    parser.add_argument("--gate", type=float, default=GATE_DEFAULT, help="及格線（%），預設 50")
    parser.add_argument("--no-save", action="store_true", help="不寫入 Wiki")
    args = parser.parse_args()

    print("評估中（純本地 PyTorch，零 LLM token）...")
    df = evaluate_all(gate=args.gate)
    print_report(df, args.gate)

    retrain_log = None
    if args.retrain:
        fails = df[df["verdict"] == "fail"]["ticker"].tolist()
        if not fails:
            print("所有股票皆及格，無需重訓。")
        else:
            print(f"開始重訓 {len(fails)} 支低於 {args.gate:.0f}% 的股票...")
            retrain_log = [retrain_ticker(tk, gate=args.gate) for tk in fails]
            # 重訓後重新評估
            print("\n重訓後重新評估：")
            df = evaluate_all(gate=args.gate)
            print_report(df, args.gate)

    if not args.no_save:
        path = save_to_wiki(df, args.gate, retrain_log)
        print(f"報告已寫入：{path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
