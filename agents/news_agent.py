"""
新聞 Agent：負責盤前、盤中、盤後三時段的新聞爬取與 AI 分析。
"""

import sys
import os
import json
import logging
from datetime import datetime
import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

# 確保可以從專案根目錄執行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news.scraper import fetch_all_news, get_market_news
from news.analyzer import analyze_news_batch, analyze_market_sentiment
from agents.wiki_agent import append_intraday_note
from agents.sentiment_tracker import update as tracker_update

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from config import ALL_STOCKS, NEWS_DIR

# 純代碼（去除 .TW/.TWO）對應名稱，供新聞爬蟲使用
TARGET_TICKERS = {
    k.replace(".TWO", "").replace(".TW", ""): v
    for k, v in ALL_STOCKS.items()
}

os.makedirs(NEWS_DIR, exist_ok=True)


class NewsAgent:
    """盤前/中/後三時段的新聞分析 Agent，整合爬蟲與 AI 情緒分析。"""

    def __init__(self):
        """初始化基本設定。"""
        self.tickers = list(TARGET_TICKERS.keys())
        self.ticker_names = TARGET_TICKERS
        self.news_dir = NEWS_DIR
        self.today_str = datetime.now().strftime("%Y-%m-%d")
        self._cached_news: dict = {}
        logger.info(f"NewsAgent 初始化完成，追蹤 {len(self.tickers)} 支股票")

    def run_pre_market(self, predictions: list[dict] | None = None) -> dict:
        """盤前執行（建議 08:30 前）：抓取新聞並結合 ML 預測產出今日策略建議。"""
        logger.info("=== 盤前新聞分析開始 ===")
        self.today_str = datetime.now().strftime("%Y-%m-%d")

        # 1. 抓取所有股票新聞 + 大盤新聞
        logger.info("抓取個股新聞中...")
        all_news = fetch_all_news(self.tickers)
        self._cached_news = all_news

        # 2. 對每支股票分析
        analysis_results = {}
        for ticker in self.tickers:
            ticker_name = self.ticker_names.get(ticker, ticker)
            news_list = all_news.get(ticker, [])

            logger.info(f"分析 {ticker}（{ticker_name}），共 {len(news_list)} 篇新聞...")
            result = analyze_news_batch(
                news_list=news_list,
                ticker=ticker,
                ticker_name=ticker_name,
                session_type="pre_market",
            )
            result["ticker_name"] = ticker_name
            result["session_type"] = "pre_market"
            result["news_count"] = len(news_list)

            # 更新情緒連續性追蹤器
            try:
                tracker_update(
                    ticker   = ticker,
                    date_str = self.today_str,
                    score    = result.get("sentiment_score", 0.0),
                    signal   = result.get("strategy_signal", "hold"),
                    session  = "pre_market",
                )
            except Exception as e:
                logger.debug(f"sentiment_tracker 更新失敗（{ticker}）：{e}")

            # 若有 ML 預測，附加到分析結果
            if predictions:
                pred = next(
                    (p for p in predictions if ticker in str(p.get("ticker", ""))),
                    None,
                )
                if pred:
                    result["ml_prediction"] = pred

            analysis_results[ticker] = result

        # 3. 大盤情緒
        market_news = all_news.get("market", [])
        market_analysis = analyze_market_sentiment(market_news)
        analysis_results["market"] = market_analysis

        # 4. 存入 JSON
        output_path = os.path.join(self.news_dir, f"{self.today_str}_pre_market.json")
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
            logger.info(f"盤前分析已存至：{output_path}")
        except OSError as e:
            logger.warning(f"存檔失敗：{e}")

        logger.info("=== 盤前新聞分析完成 ===")
        return analysis_results

    def run_intraday(self, current_prices: dict | None = None) -> dict:
        """盤中執行：更新新聞並偵測重大訊號變化，必要時更新策略。"""
        logger.info("=== 盤中新聞更新開始 ===")
        current_time = datetime.now().strftime("%H:%M")

        # 重新抓取新聞（只取最新幾篇）
        new_analysis = {}
        significant_changes = []

        for ticker in self.tickers:
            ticker_name = self.ticker_names.get(ticker, ticker)
            from news.scraper import scrape_stock_news
            import time
            fresh_news = scrape_stock_news(ticker, max_articles=5)
            time.sleep(0.5)

            if not fresh_news:
                continue

            result = analyze_news_batch(
                news_list=fresh_news,
                ticker=ticker,
                ticker_name=ticker_name,
                session_type="intraday",
            )
            result["ticker_name"] = ticker_name
            result["session_type"] = "intraday"
            result["current_time"] = current_time

            # 加入即時價格（若有）
            if current_prices and ticker in current_prices:
                result["current_price"] = current_prices[ticker]

            new_analysis[ticker] = result

            # 偵測重大訊號：sentiment_score 絕對值超過 0.7
            score = result.get("sentiment_score", 0.0)
            if abs(score) >= 0.7:
                significant_changes.append({
                    "ticker": ticker,
                    "name": ticker_name,
                    "signal": result.get("strategy_signal"),
                    "score": score,
                    "reason": result.get("strategy_reason", ""),
                })

        if significant_changes:
            logger.warning(f"偵測到 {len(significant_changes)} 個重大訊號變化！")
            for change in significant_changes:
                logger.warning(
                    f"  [{change['ticker']}] {change['name']}: {change['signal']} (score={change['score']:+.2f})"
                )

        # 追加 Wiki 備注
        if new_analysis or significant_changes:
            note_lines = [f"盤中更新時間：{current_time}\n"]
            if significant_changes:
                note_lines.append("**重大訊號：**\n")
                for c in significant_changes:
                    note_lines.append(
                        f"- {c['ticker']} {c['name']}: {c['signal']} — {c['reason']}\n"
                    )
            note = "".join(note_lines)
            append_intraday_note(self.today_str, note, "intraday")

        logger.info("=== 盤中新聞更新完成 ===")
        return new_analysis

    def run_post_market(
        self,
        actual_prices: dict | None = None,
        predictions: list[dict] | None = None,
    ) -> dict:
        """盤後執行：比較預測 vs 實際結果，計算績效並寫入 Reflexion。"""
        logger.info("=== 盤後新聞分析與績效評估開始 ===")
        performance_report = {
            "date": self.today_str,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ticker_performance": {},
            "summary": {},
        }

        # 計算 MAPE、方向準確率與 baseline 比較
        if actual_prices and predictions:
            mape_list = []
            baseline_mape_list = []   # 猜不變（prev_close）的 MAPE
            direction_correct = 0
            total = 0

            for pred in predictions:
                ticker = pred.get("ticker", "").replace(".TWO", "").replace(".TW", "")
                if ticker not in actual_prices:
                    continue

                actual = actual_prices[ticker]
                predicted_price = pred.get("predicted_price") or pred.get("predicted_close")
                if not predicted_price or not actual:
                    continue

                mape = abs(actual - predicted_price) / actual * 100
                mape_list.append(mape)

                # baseline：猜不變（predicted = prev_close）
                base_price = pred.get("last_close") or pred.get("prev_close")
                if base_price:
                    baseline_mape_list.append(abs(actual - base_price) / actual * 100)

                    # 方向準確率：基準必須用 last_close（預測當下的最新收盤），
                    # 與 predict.py 計算 direction 的基準一致，否則方向判定會錯亂。
                    sign = lambda x: (x > 1e-9) - (x < -1e-9)
                    pred_pct = pred.get("predicted_change_pct")
                    if pred_pct is None:
                        pred_pct = (predicted_price / base_price - 1) * 100
                    actual_pct = (actual / base_price - 1) * 100
                    dir_ok = sign(pred_pct) == sign(actual_pct)
                    if dir_ok:
                        direction_correct += 1
                    total += 1
                else:
                    dir_ok = None

                performance_report["ticker_performance"][ticker] = {
                    "predicted": predicted_price,
                    "actual": actual,
                    "prev_close": base_price,
                    "mape": round(mape, 2),
                    "abs_error_pp": abs(pred_pct - actual_pct) if base_price else None,
                    "prediction_source": "immutable_pre_market",
                    "baseline_mape": round(abs(actual - base_price) / actual * 100, 2) if base_price else None,
                    "direction_correct": dir_ok,
                }

            if mape_list:
                avg_mape = sum(mape_list) / len(mape_list)
                avg_baseline = sum(baseline_mape_list) / len(baseline_mape_list) if baseline_mape_list else None
                dir_accuracy = direction_correct / total if total > 0 else None
                beat_baseline = (avg_mape < avg_baseline) if avg_baseline else None
                performance_report["summary"] = {
                    "avg_mape": round(avg_mape, 2),
                    "baseline_mape": round(avg_baseline, 2) if avg_baseline else None,
                    "beat_baseline": beat_baseline,
                    "direction_accuracy": round(dir_accuracy, 3) if dir_accuracy is not None else None,
                    "sample_count": len(mape_list),
                }
                beat_str = f"{'✅ 勝過' if beat_baseline else '❌ 未勝'} baseline({avg_baseline:.2f}%)" if avg_baseline else ""
                logger.info(
                    f"績效摘要：MAPE={avg_mape:.2f}% {beat_str}，方向準確率={dir_accuracy:.1%}"
                    if dir_accuracy is not None else f"績效摘要：MAPE={avg_mape:.2f}% {beat_str}"
                )

        # 盤後新聞分析
        post_analysis = {}
        for ticker in self.tickers[:5]:  # 盤後只分析前5支，避免 API 過度使用
            ticker_name = self.ticker_names.get(ticker, ticker)
            from news.scraper import scrape_stock_news
            import time
            news = scrape_stock_news(ticker, max_articles=5)
            time.sleep(0.5)

            result = analyze_news_batch(
                news_list=news,
                ticker=ticker,
                ticker_name=ticker_name,
                session_type="post_market",
            )
            result["ticker_name"] = ticker_name
            result["session_type"] = "post_market"
            post_analysis[ticker] = result

        performance_report["post_market_analysis"] = post_analysis

        # 生成 Reflexion 摘要並寫入 Wiki
        reflexion_lines = [
            "### 績效統計\n",
        ]
        if performance_report["summary"]:
            s = performance_report["summary"]
            avg_mape = s.get('avg_mape', 'N/A')
            baseline = s.get('baseline_mape')
            beat = s.get('beat_baseline')
            beat_str = f"  {'✅ 勝過' if beat else '❌ 未勝'} baseline {baseline}%" if baseline else ""
            dir_acc = s.get('direction_accuracy')
            dir_str = f"{dir_acc:.1%}" if isinstance(dir_acc, float) else str(dir_acc)
            reflexion_lines.append(f"- 平均 MAPE：{avg_mape}%{beat_str}\n")
            if baseline:
                reflexion_lines.append(f"- Baseline（猜不變）MAPE：{baseline}%\n")
            reflexion_lines.append(f"- 方向準確率：{dir_str}\n")
            reflexion_lines.append(f"- 樣本數：{s.get('sample_count', 0)}\n")

        reflexion_lines.append("\n### 個股偏誤分析\n")
        for ticker, perf in performance_report["ticker_performance"].items():
            name = self.ticker_names.get(ticker, ticker)
            dir_icon = "✓" if perf.get("direction_correct") else "✗"
            bl = perf.get("baseline_mape")
            bl_str = f"  baseline={bl}%" if bl else ""
            reflexion_lines.append(
                f"- {ticker} {name}：預測={perf['predicted']}，實際={perf['actual']}，"
                f"MAPE={perf['mape']}%{bl_str} {dir_icon}\n"
            )

        reflexion_note = "".join(reflexion_lines)
        append_intraday_note(self.today_str, reflexion_note, "post_market")

        # 存檔
        output_path = os.path.join(self.news_dir, f"{self.today_str}_post_market.json")
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(performance_report, f, ensure_ascii=False, indent=2)
            logger.info(f"盤後報告已存至：{output_path}")
        except OSError as e:
            logger.warning(f"盤後報告存檔失敗：{e}")

        logger.info("=== 盤後分析完成 ===")
        return performance_report

    def _get_session_context(
        self,
        wiki_context: str,
        predictions: list[dict] | None,
        session_type: str,
    ) -> str:
        """組合包含 wiki 知識、今日預測和時段資訊的完整 system prompt。"""
        session_label = {
            "pre_market": "盤前分析（08:30 前）",
            "intraday": "盤中即時更新",
            "post_market": "盤後檢討與反思",
        }.get(session_type, session_type)

        pred_summary = ""
        if predictions:
            pred_lines = []
            for p in predictions[:10]:
                ticker = p.get("ticker", "")
                name = p.get("name", "")
                direction = p.get("direction", p.get("predicted_direction", "N/A"))
                confidence = p.get("confidence", p.get("model_confidence", "N/A"))
                if isinstance(confidence, float):
                    confidence = f"{confidence:.0%}"
                pred_lines.append(f"  - {ticker} {name}: {direction}（信心度 {confidence}）")
            pred_summary = "\n今日 ML 預測：\n" + "\n".join(pred_lines)

        context = (
            f"當前時段：{session_label}\n"
            f"分析時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"{pred_summary}\n\n"
            f"近期歷史記錄：\n{wiki_context[:3000]}"  # 限制長度避免超出 token
        )
        return context


if __name__ == "__main__":
    print("=== 測試 NewsAgent ===")
    agent = NewsAgent()

    # 測試盤前分析（使用模擬預測）
    sample_predictions = [
        {"ticker": "2327.TW", "name": "國巨", "direction": "up", "confidence": 0.70},
    ]

    print("\n執行盤前分析...")
    results = agent.run_pre_market(predictions=sample_predictions)

    print(f"\n分析完成，共 {len(results)} 個結果")
    for ticker, analysis in list(results.items())[:3]:
        print(f"\n{ticker}: sentiment={analysis.get('sentiment')}, signal={analysis.get('strategy_signal')}")

    print("\n=== 測試完成 ===")
