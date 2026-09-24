"""
模型訓練腳本
用法：
  python run_train.py                    # 訓練所有股票（LSTM）
  python run_train.py --model transformer
  python run_train.py --ticker 2327.TW   # 只訓練指定股票
  python run_train.py --fetch            # 先抓資料再訓練
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="股票預測模型訓練腳本")
    parser.add_argument("--model", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--ticker", type=str, default=None, help="指定單一股票，例如 2327.TW")
    parser.add_argument("--fetch", action="store_true", help="訓練前先重新抓取資料")
    parser.add_argument("--epochs", type=int, default=None, help="覆寫訓練 epoch 數")
    args = parser.parse_args()

    if args.fetch:
        print("=== 抓取最新股票資料 ===")
        from data.fetch_stocks import download_all_stocks
        download_all_stocks()

    from config import ALL_STOCKS
    from models.train import train_model

    if args.epochs:
        import config
        config.EPOCHS = args.epochs

    tickers = [args.ticker] if args.ticker else list(ALL_STOCKS.keys())

    print(f"\n=== 開始訓練（模型：{args.model}，共 {len(tickers)} 支股票）===\n")

    results = {}
    for ticker in tickers:
        name = ALL_STOCKS.get(ticker, ticker)
        print(f"\n--- 訓練 {ticker} ({name}) ---")
        try:
            train_losses, val_losses = train_model(ticker, model_type=args.model)
            final_val = val_losses[-1] if val_losses else float("inf")
            results[ticker] = {"status": "success", "final_val_loss": f"{final_val:.6f}"}
            print(f"  完成 | 最終 Val Loss: {final_val:.6f}")
        except Exception as e:
            results[ticker] = {"status": "failed", "error": str(e)}
            print(f"  失敗：{e}")

    print(f"\n{'='*60}")
    print("訓練結果摘要：")
    for ticker, result in results.items():
        name = ALL_STOCKS.get(ticker, ticker)
        status = result["status"]
        detail = result.get("final_val_loss", result.get("error", ""))
        print(f"  {ticker:12s} ({name:6s})  {status:8s}  {detail}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
