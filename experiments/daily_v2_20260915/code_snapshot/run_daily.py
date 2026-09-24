"""
每日自動化執行腳本
用法：
  python run_daily.py pre      # 盤前分析（08:30 前）
  python run_daily.py intraday # 盤中更新
  python run_daily.py post     # 盤後檢討
  python run_daily.py all      # 完整流程（測試用）
"""
import sys
import os
import argparse
from datetime import datetime
import pytz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TWN_TZ = pytz.timezone("Asia/Taipei")


def get_taiwan_time():
    return datetime.now(TWN_TZ)


def _refresh_price_data():
    """盤前先更新股價資料到最新交易日，確保 ML 預測不用過期資料。"""
    print("更新股價資料到最新交易日...")
    try:
        from data.fetch_stocks import fetch_stock
        from config import ALL_STOCKS
        ok = 0
        for ticker in ALL_STOCKS:
            try:
                df = fetch_stock(ticker)
                if not df.empty:
                    ok += 1
            except Exception as e:
                print(f"  [跳過] {ticker}: {e}")
        print(f"  資料更新完成：{ok}/{len(ALL_STOCKS)} 支")
    except Exception as e:
        print(f"  資料更新失敗（將使用現有資料）：{e}")


def run_pre_market():
    print(f"\n{'='*60}")
    print(f"  盤前分析  {get_taiwan_time().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")
    from prediction_audit import require_pre_market
    require_pre_market()
    _refresh_price_data()   # 先更新資料，再預測
    from daily_retrain import train_latest
    try:
        train_latest()  # 用最後已收盤資料完成每日版本；相同輸入會跳過。
    except Exception as error:
        print(f"每日候選重訓失敗：{error}；繼續正式盤前分析，候選另行檢查日期。")
    from agents.orchestrator import TradingOrchestrator
    orchestrator = TradingOrchestrator()
    return orchestrator.run_pre_market()


def run_intraday():
    print(f"\n{'='*60}")
    print(f"  盤中更新  {get_taiwan_time().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")
    from agents.orchestrator import TradingOrchestrator
    orchestrator = TradingOrchestrator()
    return orchestrator.run_intraday()


def run_post_market(actual_prices=None):
    print(f"\n{'='*60}")
    print(f"  盤後檢討  {get_taiwan_time().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    # 未手動傳入價格時，自動從 yfinance 抓當日收盤價
    if actual_prices is None:
        print("自動抓取今日收盤價...")
        from data.fetch_stocks import get_today_closing_prices
        actual_prices = get_today_closing_prices()
        if actual_prices:
            print(f"取得 {len(actual_prices)} 支股票收盤價：")
            for code, price in actual_prices.items():
                print(f"  {code}: {price}")
        else:
            print("  無法取得收盤價，Reflexion 將跳過績效計算")

    from agents.orchestrator import TradingOrchestrator
    orchestrator = TradingOrchestrator()
    try:
        return orchestrator.run_post_market(actual_prices=actual_prices)
    finally:
        from prediction_audit import update
        update()
        from daily_retrain import train_latest
        train_latest()


def check_session():
    """根據目前台灣時間自動判斷應執行哪個 session"""
    now = get_taiwan_time()
    hour = now.hour
    minute = now.minute
    total_minutes = hour * 60 + minute

    # 08:00-09:00 盤前
    if 480 <= total_minutes < 540:
        return "pre"
    # 09:00-13:30 盤中
    elif 540 <= total_minutes < 810:
        return "intraday"
    # 13:30 之後 盤後
    elif total_minutes >= 810:
        return "post"
    else:
        return "pre"


def main():
    parser = argparse.ArgumentParser(description="股票交易系統每日執行腳本")
    parser.add_argument(
        "session",
        nargs="?",
        choices=["pre", "intraday", "post", "all", "auto", "audit"],
        default="auto",
        help="執行哪個時段（auto=自動依時間判斷）",
    )
    parser.add_argument(
        "--prices",
        nargs="*",
        help="盤後使用：實際收盤價 格式: 2327.TW:450 2492.TW:280 ...",
    )
    args = parser.parse_args()

    actual_prices = None
    if args.prices:
        actual_prices = {}
        for item in args.prices:
            ticker, price = item.split(":")
            actual_prices[ticker] = float(price)

    session = args.session
    if session == "auto":
        session = check_session()
        print(f"自動判斷時段：{session}")

    if session == "audit":
        from prediction_audit import update
        update()
    elif session == "pre":
        run_pre_market()
    elif session == "intraday":
        run_intraday()
    elif session == "post":
        run_post_market(actual_prices=actual_prices)
    elif session == "all":
        # 完整流程（測試用）
        session_data = run_pre_market()
        run_intraday()
        run_post_market(actual_prices=actual_prices)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
