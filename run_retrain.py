"""
每月自動重訓腳本：抓取最新資料並重新訓練所有股票模型。
目前 Windows 工作排程器每月 28 日 20:00 執行。輸出候選版本，禁止自動覆蓋正式模型。
"""
import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="每月模型重訓腳本")
    parser.add_argument("--model",  default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--dry-run", action="store_true", help="只下載資料，不重訓")
    args = parser.parse_args()

    start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"  月度模型重訓  {start_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    # Step 1：抓取最新資料（更新 END_DATE 到今天）
    print("[1/3] 更新股票歷史資料...")
    try:
        import config as cfg
        import yfinance as yf
        import pandas as pd
        from data.fetch_stocks import fetch_stock

        today = datetime.now().strftime("%Y-%m-%d")
        cfg.END_DATE = today

        success, fail = 0, 0
        for ticker in cfg.ALL_STOCKS:
            try:
                df = fetch_stock(ticker)
                if not df.empty:
                    success += 1
                else:
                    fail += 1
            except Exception as e:
                print(f"  [跳過] {ticker}: {e}")
                fail += 1

        print(f"  資料更新完成：成功 {success} 支，失敗 {fail} 支")
    except Exception as e:
        print(f"  資料更新失敗：{e}")
        return

    if args.dry_run:
        print("\n--dry-run 模式，跳過訓練")
        return

    # 候選目錄隔離模型及共用 scaler，保留完整正式版本。
    import shutil
    from pathlib import Path
    production = Path(cfg.CHECKPOINT_DIR)
    candidate = production.parent / "candidates" / start_time.strftime("%Y%m%dT%H%M%S")
    shutil.copytree(production, candidate)
    cfg.CHECKPOINT_DIR = str(candidate)
    (candidate / "CANDIDATE_ONLY.txt").write_text(
        "未晉升：舊驗證段曾用於 epoch 選擇與校準，不屬獨立測試。禁止以此績效晉升。\n"
        "需固定 train/validation/calibration/test 日期與獨立 scaler；新舊盤前影子留底比較後人工審查。\n", encoding="utf-8")
    print(f"候選模型目錄：{candidate}（正式模型保留）")

    # Step 2：重新訓練所有模型
    print(f"\n[2/3] 重新訓練模型（{args.model.upper()}）...")
    from models.train import train_model

    results = {}
    for ticker, name in cfg.ALL_STOCKS.items():
        print(f"\n  訓練 {ticker} ({name})...")
        try:
            train_losses, val_losses = train_model(ticker, model_type=args.model)
            results[ticker] = {"status": "success", "val_loss": f"{val_losses[-1]:.6f}"}
        except Exception as e:
            results[ticker] = {"status": "failed", "error": str(e)}
            print(f"  失敗：{e}")

    # Step 2b：重訓分位數模型（區間預測，波動率正規化 CQR）
    print(f"\n[2b/3] 重新訓練分位數區間模型...")
    try:
        from quantile_forecast import train_quantile
        q_covs = []
        for ticker, name in cfg.ALL_STOCKS.items():
            try:
                qr = train_quantile(ticker)
                q_covs.append(qr["cov80"])
            except Exception as e:
                print(f"  分位數重訓失敗（{ticker}）：{e}")
        if q_covs:
            import numpy as _np
            print(f"  分位數模型平均80%覆蓋率：{_np.mean(q_covs):.1%}（理想 80%）")
    except Exception as e:
        print(f"  分位數重訓整體失敗：{e}")

    # Step 2c：重新校準 Regime 漂移門檻（訓練分布已更新）
    print(f"\n[2c/3] 重新校準 Regime 漂移門檻（Evidently）...")
    try:
        from drift_monitor import calibrate_all
        print("候選模式：略過正式漂移門檻更新")
    except Exception as e:
        print(f"  漂移門檻校準失敗：{e}")

    # Step 3：將重訓記錄寫入 Wiki
    print("\n[3/3] 寫入重訓記錄到 Wiki...")
    try:
        from config import WIKI_DIR
        log_path = os.path.join(WIKI_DIR, f"retrain_{datetime.now().strftime('%Y-%m')}.md")
        lines = [
            f"# 月度重訓記錄 {datetime.now().strftime('%Y-%m')}",
            f"\n執行時間：{start_time.strftime('%Y-%m-%d %H:%M')} — {datetime.now().strftime('%H:%M')}",
            f"\n## 訓練結果\n",
            "| 股票 | 名稱 | 狀態 | Val Loss |",
            "|------|------|------|---------|",
        ]
        for ticker, r in results.items():
            name = cfg.ALL_STOCKS.get(ticker, "")
            status = r["status"]
            detail = r.get("val_loss", r.get("error", ""))
            lines.append(f"| {ticker} | {name} | {status} | {detail} |")

        # 重訓後自動回測，驗證策略相對 Buy & Hold 表現
        try:
            from backtest import backtest_all
            bt = backtest_all(model_type="ensemble")
            if not bt.empty:
                n_beat = int((bt["excess_return"] > 0).sum())
                lines.append("\n## 回測驗證（vs 買進持有）\n")
                lines.append("| 股票 | 勝率 | 策略報酬 | 買持報酬 | 超額 | 夏普 |")
                lines.append("|------|------|---------|---------|------|------|")
                for _, r in bt.iterrows():
                    lines.append(
                        f"| {r['ticker']} | {r['win_rate']}% | {r['strategy_return']}% "
                        f"| {r['buyhold_return']}% | {r['excess_return']}% | {r['sharpe']} |"
                    )
                lines.append(
                    f"\n**打敗買進持有：{n_beat}/{len(bt)} 支** | "
                    f"平均勝率 {bt['win_rate'].mean():.1f}% | "
                    f"平均夏普 {bt['sharpe'].mean():.2f}"
                )
                print(f"  回測完成：打敗買進持有 {n_beat}/{len(bt)} 支")
        except Exception as e:
            print(f"  回測失敗：{e}")

        # 重訓後過擬合監控（訓練 vs 測試 MAPE 差距）
        try:
            from overfit_monitor import check_all, to_wiki_lines
            of = check_all(model_type="lstm")
            if not of.empty:
                lines += to_wiki_lines(of)
                n_warn = int((of["verdict"] == "warning").sum())
                print(f"  過擬合監控：{n_warn} 支警告（差距>4%）")
                if n_warn:
                    warn = of[of["verdict"] == "warning"]["ticker"].tolist()
                    print(f"  ⚠️ 需關注：{', '.join(warn)}")
        except Exception as e:
            print(f"  過擬合監控失敗：{e}")

        # 模型品質閘門：方向準確率 < 50% 者，用方向感知 loss 重訓（不退步保護）
        try:
            from eval_harness import evaluate_all, retrain_ticker, to_wiki_lines as harness_lines
            print("\n  模型品質閘門：檢查方向準確率（及格線 50%）...")
            ev = evaluate_all(gate=50.0)
            fails = ev[ev["verdict"] == "fail"]["ticker"].tolist()
            retrain_log = None
            if fails:
                print(f"  ❌ 低於 50% 需重訓：{', '.join(fails)}")
                retrain_log = [retrain_ticker(tk, gate=50.0) for tk in fails]
                ev = evaluate_all(gate=50.0)   # 重訓後重新評估
            else:
                print("  ✅ 全部及格，無需方向重訓")
            lines.append("")
            lines += harness_lines(ev, 50.0, retrain_log)
            avg = ev["ensemble"].dropna().mean()
            n_pass = int((ev["verdict"] == "pass").sum())
            print(f"  方向閘門完成：及格 {n_pass}/{len(ev)} 支，平均 {avg:.1f}%")
        except Exception as e:
            print(f"  模型品質閘門失敗：{e}")

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  記錄已寫入：{log_path}")
    except Exception as e:
        print(f"  Wiki 寫入失敗：{e}")

    elapsed = (datetime.now() - start_time).seconds // 60
    print(f"\n{'='*60}")
    print(f"重訓完成，耗時約 {elapsed} 分鐘")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
