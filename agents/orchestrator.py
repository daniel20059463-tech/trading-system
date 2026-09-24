"""
主控 Orchestrator：協調 NewsAgent、StrategyAgent 與 WikiAgent 的完整交易流程。
"""

import sys
import os
import json
import logging
from datetime import datetime
import numpy as np

# 確保可以從專案根目錄執行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.news_agent import NewsAgent
from agents.strategy_agent import StrategyAgent
from agents.rule_agent import RuleAgent
from agents.wiki_agent import (
    write_daily_record,
    get_wiki_context_for_agent,
    append_intraday_note,
    read_recent_wiki,
)
from config import ALL_STOCKS as TICKER_NAMES
from visualize import append_log, generate_charts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TradingOrchestrator:
    """主控 Agent：協調所有子 Agent 完成盤前、盤中、盤後三時段的完整交易分析流程。"""

    def __init__(self):
        """初始化所有子 Agent 並載入 ML 預測模組。"""
        logger.info("初始化 TradingOrchestrator...")
        self.news_agent = NewsAgent()
        self.strategy_agent = StrategyAgent()
        self.rule_agent = RuleAgent()
        self.today_str = datetime.now().strftime("%Y-%m-%d")

        # 嘗試載入 ML 預測模組
        try:
            from models.predict import predict_all
            self.predict_all = predict_all
            logger.info("ML 預測模組載入成功")
        except ImportError as e:
            logger.warning(f"ML 預測模組載入失敗（{e}），將使用空預測")
            self.predict_all = lambda: []

        # 儲存本次 session 的資料，供跨方法使用
        self._session_data: dict = {}

        logger.info("TradingOrchestrator 初始化完成")

    @staticmethod
    def _to_serializable(obj):
        """遞迴將 numpy 型別轉為 Python 原生型別，確保 JSON 可序列化。"""
        if isinstance(obj, dict):
            return {k: TradingOrchestrator._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [TradingOrchestrator._to_serializable(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def run_pre_market(self) -> dict:
        """盤前分析流程（建議 08:30 前執行）：預測→新聞→策略→Wiki。"""
        self.today_str = datetime.now().strftime("%Y-%m-%d")
        print(f"\n{'='*60}")
        print(f"=== 盤前分析 {self.today_str} ===")
        print(f"{'='*60}")
        print(f"開始時間：{datetime.now().strftime('%H:%M:%S')}\n")

        from prediction_audit import require_pre_market
        require_pre_market()

        # Step 1：ML 預測
        print("[1/5] 執行 ML 預測...")
        try:
            predictions = self.predict_all()
            print(f"      ML 預測完成，共 {len(predictions)} 支股票")
        except Exception as e:
            logger.warning(f"ML 預測失敗：{e}")
            predictions = []
            print("      ML 預測失敗，使用空預測繼續")

        # Step 2：讀取近3日 Wiki context
        print("[2/5] 讀取歷史 Wiki 知識...")
        wiki_context = get_wiki_context_for_agent()
        print(f"      Wiki context 長度：{len(wiki_context)} 字元")

        # Step 3：新聞抓取與分析
        print("[3/5] 抓取並分析新聞（此步驟需要幾分鐘）...")
        try:
            news_analysis = self.news_agent.run_pre_market(predictions=predictions)
            market_sentiment = news_analysis.get("market", {})
            print(f"      新聞分析完成，大盤情緒：{market_sentiment.get('market_sentiment', 'N/A')}")
        except Exception as e:
            logger.error(f"新聞分析失敗：{e}")
            news_analysis = {}
            market_sentiment = {"market_sentiment": "neutral", "score": 0.0, "summary": "新聞分析失敗"}
            print("      新聞分析失敗，使用中立預設值")

        # 新聞只建立同日 Open→Close 的不可變影子預測。舊規則的標籤是 T→T+1，
        # 不得再用它改寫今天的正式點預測。
        news_intraday_snapshot = None
        try:
            from news_intraday_shadow import capture_pre_market
            news_intraday_snapshot = capture_pre_market(
                self._to_serializable(news_analysis),
                source_articles=self._to_serializable(self.news_agent._cached_news),
            )
            tr = news_intraday_snapshot.get("training", {})
            print(f"      新聞盤中影子：已留底（live {tr.get('training_days', 0)} 天，尚不改正式預測）")
        except Exception as e:
            logger.warning(f"新聞盤中影子留底失敗：{e}")

        # 美股相似行情：只讀已收盤且同日的五項美股資料，留底供策略參考。
        market_analog_snapshot = None
        try:
            from market_analog_live import capture_pre_market as capture_market_analogs
            market_analog_snapshot = capture_market_analogs(day=self.today_str)
            if market_analog_snapshot.get("aggregate"):
                print(f"      美股相似行情：已留底 {market_analog_snapshot['aggregate']['case_count']} 個歷史案例")
            else:
                missing = ", ".join(market_analog_snapshot["metadata"].get("missing_symbols", []))
                print(f"      美股核心指數：已留底；同業 {missing} 延遲，相似案例暫停")
        except Exception as e:
            logger.warning(f"美股相似行情略過（缺漏/過期/非盤前）：{e}")

        # 標示可信實盤證據狀態並附探索性影子值；正式方向／幅度維持原行為。
        from model_governance import apply_publication_gate
        from rigorous_evaluation import predict_shadow
        for p in predictions:
            shadow = predict_shadow(p.get("ticker", ""))
            if shadow:
                p["shadow_candidate"] = shadow
        if market_analog_snapshot:
            try:
                from us_lead_shadow import attach_candidates as attach_us_lead_shadow
                attached = attach_us_lead_shadow(predictions, market_analog_snapshot,
                                                 day=self.today_str)
                print(f"      美股數值影子：{attached} 支已留待同日盤後驗證（正式預測維持原值）")
            except Exception as e:
                logger.warning(f"美股數值影子略過：{e}")
        evidence = apply_publication_gate(predictions)
        if not evidence["approved"]:
            print(f"      模型證據閘門：尚未通過實盤證據（{evidence['reason']}）；正式原始輸出維持")

        # 固定實際發布的最終點預測；重跑或超過開盤一律拒絕覆寫。
        from prediction_audit import freeze

        # Step 4：策略生成
        print("[4/5] 生成操作策略...")
        try:
            strategies = self.strategy_agent.generate_strategy(
                predictions=predictions,
                news_analysis=news_analysis,
                market_sentiment=market_sentiment,
                wiki_context=wiki_context,
            )
            if news_intraday_snapshot:
                from news_intraday_shadow import attach_to_strategies
                attach_to_strategies(strategies, news_intraday_snapshot)
            if market_analog_snapshot:
                for strategy in strategies:
                    ticker = strategy.get("ticker", "")
                    case = market_analog_snapshot["by_ticker"].get(ticker)
                    if case:
                        strategy["us_market_analog_reference"] = {
                            "us_session": market_analog_snapshot["metadata"]["us_session"],
                            "case_count": case["case_count"],
                            "intraday_history": case["summary"]["intraday"],
                            "status": "descriptive_reference",
                        }
            print(f"      策略生成完成，共 {len(strategies)} 支股票")
        except Exception as e:
            logger.error(f"策略生成失敗：{e}")
            strategies = []
            print("      策略生成失敗")

        # 策略完成後才固定上下文及最終預測，讓候選只讀當時已完成的新聞/策略。
        try:
            from daily_retrain import snapshot_context, attach_candidates
            try:
                snapshot_context(self._to_serializable(news_analysis), self._to_serializable(strategies))
            except FileExistsError:
                logger.info("今日上下文已留底，使用原始快照")
            decision = attach_candidates(predictions)
            print(f"      每日重訓比較：{decision['reason']}")
        except Exception as e:
            logger.warning(f"每日重訓候選/上下文留底失敗：{e}")
        freeze(self._to_serializable(predictions))

        # Step 5：寫入 Wiki
        print("[5/5] 寫入 Wiki 知識庫...")
        market_summary_text = (
            f"{market_sentiment.get('market_sentiment', 'neutral')} - "
            f"{market_sentiment.get('summary', '無摘要')}"
        )
        strategy_dict = {
            s.get("ticker", "").replace(".TWO", "").replace(".TW", ""): s
            for s in strategies
        }
        try:
            wiki_path = write_daily_record(
                date_str=self.today_str,
                predictions=self._to_serializable(predictions),
                news_analysis=self._to_serializable(news_analysis),
                strategy_decisions=self._to_serializable(strategy_dict),
                market_summary=market_summary_text,
            )
            print(f"      Wiki 已寫入：{wiki_path}")
        except Exception as e:
            logger.error(f"Wiki 寫入失敗：{e}")

        if market_analog_snapshot:
            try:
                from market_analog_live import prompt_context
                append_intraday_note(self.today_str, prompt_context(market_analog_snapshot), "pre_market")
            except Exception as e:
                logger.warning(f"美股相似行情 Wiki 摘要失敗：{e}")

        # 國際快照留底（含正確美股日期+新鮮度），供日後驗證用了哪天美股
        try:
            from agents.global_snapshot import record_snapshot_to_wiki
            record_snapshot_to_wiki(self.today_str)
            print("      國際快照已留底")
        except Exception as e:
            logger.debug(f"快照留底失敗：{e}")

        # 隔日區間預測（分位數迴歸 + 波動率正規化 CQR，純本地零 token）
        try:
            from quantile_forecast import predict_quantile, build_interval_lines
            q_results = []
            for ticker in TICKER_NAMES:
                try:
                    q_results.append(predict_quantile(ticker))
                except Exception as qe:
                    logger.debug(f"區間預測失敗（{ticker}）：{qe}")
            if q_results:
                apply_publication_gate(q_results)
                freeze(self._to_serializable(q_results), kind="interval")
                append_intraday_note(self.today_str,
                                     "\n".join(build_interval_lines(q_results)), "pre_market")
                n_high = sum(1 for r in q_results if r.get("width50", 99) < 3)
                print(f"      隔日區間預測完成：{len(q_results)} 支（50%名目區間較窄 {n_high} 支）")
        except Exception as e:
            logger.debug(f"區間預測整體失敗：{e}")

        # Regime 漂移監控（Evidently）：近期市場狀態是否偏離模型訓練分布
        try:
            from drift_monitor import check_regime, build_wiki_lines, plot_drift_preview, save_html_report
            reg = check_regime()
            append_intraday_note(self.today_str, "\n".join(build_wiki_lines(reg)), "pre_market")
            icon = {"normal": "✅ 正常", "watch": "🟡 留意", "abnormal": "🔴 異常"}[reg["overall"]]
            print(f"      Regime 漂移監控：{icon}（{reg['n_abnormal']}/{reg['total']} 支偏離）")
            if reg["overall"] == "abnormal":
                print("      ⚠️ 市場狀態異常，今日點預測可信度下降，建議參考區間而非點位")
            # 對漂移最嚴重的個股生成視覺化（靜態預覽圖 + Evidently 互動 HTML）
            if reg["rows"]:
                worst = max(reg["rows"], key=lambda r: r["share"])["ticker"]
                try:
                    plot_drift_preview(worst)
                    save_html_report(worst)
                    print(f"      漂移視覺化已生成（{worst}）：charts/*_drift.png + drift_reports/*.html")
                except Exception as ve:
                    logger.debug(f"漂移視覺化失敗：{ve}")
        except Exception as e:
            logger.debug(f"Regime 漂移監控失敗：{e}")

        # 08:30 只更新風險地板；最終單一區間由08:36美股條件流程推播。
        try:
            from intraday_monitor import todays_risk_floors, send_discord, WEBHOOK
            if WEBHOOK:
                regime_tip = ""
                try:
                    regime_tip = {"normal": "", "watch": "\n🟡 提醒：部分個股偏離訓練分布，留意",
                                  "abnormal": "\n🔴 警告：市場狀態異常，今日點預測可信度低，建議降倉"}.get(reg["overall"], "")
                except Exception:
                    pass
                todays_risk_floors()   # 僅計算+快取地板供盤中跌破偵測，不推 Discord 訊息
                if regime_tip:
                    send_discord(regime_tip.strip())
        except Exception as e:
            logger.debug(f"盤前風險地板推播失敗：{e}")

        # 展示策略表格
        print(f"\n{'─'*60}")
        print("今日策略建議：")
        print(f"{'─'*60}")
        self.display_strategy_table(strategies)

        # 新聞關鍵字輔助判斷（人看的弱訊號，獨立於模型；透明列出關鍵字與偏多偏空）
        try:
            from news_aux_judge import judge_all
            aux = judge_all(news_analysis)
            if aux:
                print(f"\n{'─'*60}")
                print("📰 新聞關鍵字輔助判斷（參考用弱訊號，不取代模型）：")
                print(f"{'─'*60}")
                for _c, line, _r in aux:
                    print("  " + line)
                note = "## 📰 新聞關鍵字輔助判斷（2年新聞學習；參考用，非預測）\n" + \
                       "\n".join(f"- {line}" for _c, line, _r in aux)
                append_intraday_note(self.today_str, note, "pre_market")
        except Exception as e:
            logger.debug(f"新聞關鍵字輔助判斷失敗：{e}")

        # 儲存 session 資料
        self._session_data = {
            "date": self.today_str,
            "predictions": predictions,
            "news_analysis": news_analysis,
            "market_sentiment": market_sentiment,
            "strategies": strategies,
            "market_analog_snapshot": market_analog_snapshot,
            "wiki_context": wiki_context,
        }

        print(f"\n{'='*60}")
        print(f"盤前分析完成：{datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}\n")

        return self._session_data

    def run_intraday(self) -> dict:
        """盤中更新流程：抓取最新新聞，偵測重大訊號變化並追加 Wiki。"""
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'='*60}")
        print(f"=== 盤中更新 {current_time} ===")
        print(f"{'='*60}")

        try:
            intraday_analysis = self.news_agent.run_intraday()

            # 偵測重大訊號變化（sentiment_score 絕對值 >= 0.7）
            alerts = []
            for ticker, analysis in intraday_analysis.items():
                score = analysis.get("sentiment_score", 0.0)
                if abs(score) >= 0.7:
                    name = analysis.get("ticker_name", ticker)
                    signal = analysis.get("strategy_signal", "hold")
                    reason = analysis.get("strategy_reason", "")
                    alerts.append((ticker, name, signal, score, reason))

            if alerts:
                print(f"\n【重大訊號警示】（共 {len(alerts)} 個）")
                for ticker, name, signal, score, reason in alerts:
                    direction = "看多" if score > 0 else "看空"
                    print(f"  ⚡ {ticker} {name}: {signal} ({direction} {score:+.2f}) — {reason}")
            else:
                print("\n  無重大訊號變化，維持盤前策略")

            # 追加 Wiki
            note = (
                f"盤中更新（{current_time}）\n"
                + (f"重大訊號：{len(alerts)} 個" if alerts else "無重大訊號變化")
            )
            append_intraday_note(self.today_str, note, "intraday")

            print(f"\n盤中更新完成：{current_time}")
            return intraday_analysis

        except Exception as e:
            logger.error(f"盤中更新失敗：{e}")
            print(f"  盤中更新失敗：{e}")
            return {}

    def run_post_market(self, actual_prices: dict | None = None) -> dict:
        """盤後檢討流程：計算績效、Reflexion 反思並更新 Wiki。"""
        self.today_str = datetime.now().strftime("%Y-%m-%d")
        print(f"\n{'='*60}")
        print(f"=== 盤後檢討 {self.today_str} ===")
        print(f"{'='*60}")
        print(f"開始時間：{datetime.now().strftime('%H:%M:%S')}\n")

        strategies  = self._session_data.get("strategies", [])

        # 盤後先刷新 CSV，確保 predict_verification 用到今日（已收盤）資料
        try:
            from data.fetch_stocks import fetch_stock
            from config import ALL_STOCKS as _all
            for _tk in _all:
                try:
                    fetch_stock(_tk)
                except Exception:
                    pass
            logger.info("盤後股價資料刷新完成")
            try:
                from news_intraday_shadow import settle_day, evidence_report
                settled = settle_day(self.today_str)
                evidence_now = evidence_report()
                logger.info(
                    "新聞盤中影子結算：%s 支；live %s 天/%s 筆（%s）",
                    settled.get("settled", 0), evidence_now.get("days", 0),
                    evidence_now.get("rows", 0), evidence_now.get("status", "collecting")
                )
            except Exception as shadow_error:
                logger.warning(f"新聞盤中影子結算失敗：{shadow_error}")
        except Exception as _e:
            logger.warning(f"盤後資料刷新失敗（使用現有資料）：{_e}")

        try:
            from market_analog_live import settle_day as settle_market_analogs
            analog_result = settle_market_analogs(self.today_str)
            logger.info("美股相似行情結算：%s 支（%s）",
                        analog_result.get("settled", 0), analog_result.get("reason", "complete"))
        except Exception as analog_error:
            logger.warning(f"美股相似行情結算失敗：{analog_error}")
        try:
            from us_lead_shadow import settle_day as settle_us_lead_shadow
            us_result = settle_us_lead_shadow(self.today_str)
            logger.info("美股數值影子結算：%s 筆（%s）",
                        us_result.get("n", 0), us_result.get("reason", "complete"))
        except Exception as us_error:
            logger.warning(f"美股數值影子結算失敗：{us_error}")
        try:
            from premarket_context_interval import settle_day as settle_context_interval
            context_result = settle_context_interval(self.today_str)
            summary = context_result.get("summary", {})
            logger.info("美股條件式區間結算：%s 支；原模型方向 %.1f%%、美股後 %.1f%%、MAE %.3f→%.3f",
                        context_result.get("n", 0),
                        100 * summary.get("formal_direction_accuracy", 0),
                        100 * summary.get("context_direction_accuracy", 0),
                        summary.get("formal_mae_pp", 0), summary.get("context_mae_pp", 0))
        except Exception as context_error:
            logger.warning(f"美股條件式區間結算失敗：{context_error}")

        # 禁止盤後重新推論冒充盤前成績。實際值按明確交易日由已刷新 CSV 配對。
        from prediction_audit import saved_predictions, update as update_audit
        predictions, actual_prices = saved_predictions(self.today_str)
        update_audit()
        try:
            from historical_prediction_records import rebuild as rebuild_history
            history_summary = rebuild_history()
            logger.info("歷史逐檔對照已更新：%s 天/%s 筆",
                        history_summary.get("dates", 0), history_summary.get("total_rows", 0))
        except Exception as history_error:
            logger.warning(f"歷史逐檔對照更新失敗：{history_error}")
        if not predictions:
            logger.warning("今日沒有可驗證盤前留底，績效留空；不得以重算預測替代")

        # Step 1：盤後新聞分析 + 績效計算
        print("[1/3] 執行盤後新聞分析與績效計算...")
        try:
            performance_report = self.news_agent.run_post_market(
                actual_prices=actual_prices,
                predictions=predictions,
            )
            summary = performance_report.get("summary", {})
            if summary:
                avg_mape = summary.get("avg_mape", "N/A")
                dir_acc = summary.get("direction_accuracy", "N/A")
                if isinstance(dir_acc, float):
                    dir_acc = f"{dir_acc:.1%}"
                print(f"      MAPE：{avg_mape}%，方向準確率：{dir_acc}")
            else:
                print("      無實際價格資料，跳過績效計算")
        except Exception as e:
            logger.error(f"盤後分析失敗：{e}")
            performance_report = {}
            print(f"      盤後分析失敗：{e}")

        # Step 2：Reflexion 反思
        print("[2/3] 執行 Reflexion 自我反思...")
        try:
            actual_outcomes = {}
            if actual_prices:
                ticker_perf = performance_report.get("ticker_performance", {})
                for ticker, price in actual_prices.items():
                    perf = ticker_perf.get(ticker, {})
                    actual_outcomes[ticker] = {
                        "actual_price": price,
                        "predicted_price": perf.get("predicted"),
                        "mape": perf.get("mape"),
                        "direction_correct": perf.get("direction_correct"),
                    }

            reflexion_report = self.strategy_agent.reflexion_update(
                strategy_decisions=strategies,
                actual_outcomes=actual_outcomes,
                wiki_context=self._session_data.get("wiki_context", ""),
            )
            print("      Reflexion 分析完成")
        except Exception as e:
            logger.error(f"Reflexion 失敗：{e}")
            reflexion_report = f"Reflexion 分析失敗：{e}"
            print(f"      Reflexion 失敗：{e}")

        # Step 3：Rule Agent — LATS 雙假說規則學習
        print("[3/4] Rule Agent 執行 CLIN 規則學習...")
        rule_result = {"rule_updates": [], "analysis_report": ""}
        try:
            rule_result = self.rule_agent.analyze(
                performance=performance_report,
                news_analysis=self._session_data.get("news_analysis", {}),
                wiki_context=self._session_data.get("wiki_context", ""),
            )
            n_updates = len(rule_result.get("rule_updates", []))
            print(f"      規則學習完成，執行了 {n_updates} 條操作")
        except Exception as e:
            logger.error(f"Rule Agent 失敗：{e}")
            print(f"      Rule Agent 失敗：{e}")

        # Step 3b：規則庫健檢（防 R004~R013 垃圾規則 bug 復發，純 Python 零 token）
        try:
            from agents.rule_manager import load_rules, health_check
            hc = health_check(load_rules())
            if hc["healthy"]:
                print(f"      規則庫健檢：✅ 健康（{hc['active_count']} 條活躍規則）")
            else:
                print(f"      規則庫健檢：⚠️ {len(hc['warnings'])} 項警示")
                hc_lines = ["## 規則庫健檢警示\n",
                            f"- 活躍規則數：{hc['active_count']} 條"]
                for w in hc["warnings"]:
                    print(f"        ⚠️ {w}")
                    hc_lines.append(f"- ⚠️ {w}")
                hc_lines.append("\n建議：執行 `python -m agents.rule_manager` 檢視並手動清理")
                append_intraday_note(self.today_str, "\n".join(hc_lines), "post_market")
        except Exception as e:
            logger.debug(f"規則庫健檢失敗：{e}")

        # Step 4：更新 Wiki
        print("[4/5] 更新 Wiki 知識庫...")
        try:
            wiki_note = (
                f"## Reflexion 自我反思\n\n{reflexion_report}\n\n"
                f"## Rule Agent 規則學習\n\n{rule_result.get('analysis_report','')}\n"
            )
            append_intraday_note(self.today_str, wiki_note, "post_market")
            print("      Wiki 已更新")
        except Exception as e:
            logger.error(f"Wiki 更新失敗：{e}")

        # Step 4b：模型品質閘門（每日輕量檢查：只評估方向準確率，不重訓）
        print("[4b/5] 模型品質閘門（方向準確率，及格線 50%）...")
        try:
            from eval_harness import evaluate_all
            ev = evaluate_all(gate=50.0)
            avg = ev["ensemble"].dropna().mean()
            fails = ev[ev["verdict"] == "fail"]
            n_pass = int((ev["verdict"] == "pass").sum())
            print(f"      及格 {n_pass}/{len(ev)} 支，平均方向準確率 {avg:.1f}%")

            gate_lines = [
                "## 模型品質閘門（測試集方向準確率）\n",
                f"- 平均方向準確率：{avg:.1f}%（baseline=50% 丟硬幣）",
                f"- 及格：{n_pass}/{len(ev)} 支",
            ]
            if not fails.empty:
                fail_list = ", ".join(
                    f"{r['ticker']} {r['name']}({r['ensemble']}%)"
                    for _, r in fails.iterrows()
                )
                print(f"      ⚠️ 低於 50% 需重訓：{fail_list}")
                gate_lines.append(
                    f"- ⚠️ **低於 50% 需重訓**：{fail_list}\n"
                    f"  執行 `python eval_harness.py --retrain` 以方向感知 loss 修復"
                )
            else:
                gate_lines.append("- ✅ 歷史檢查完成；是否優於基準須另看盤前留底評估")
            append_intraday_note(self.today_str, "\n".join(gate_lines), "post_market")
        except Exception as e:
            logger.error(f"模型品質閘門失敗：{e}")
            print(f"      模型品質閘門失敗：{e}")

        # Step 4c：新聞關鍵字學習（累積「今日新聞關鍵字→今日漲跌」，可解釋、長期學）
        try:
            from news_keyword_learner import record_today
            n = record_today(self.today_str)
            if n:
                print(f"      新聞關鍵字學習：已累積今日 {n} 支的關鍵字→漲跌")
        except Exception as e:
            logger.debug(f"新聞關鍵字學習失敗：{e}")

        # Step 5：視覺化圖表
        print("[5/5] 產生視覺化圖表...")
        try:
            append_log(self.today_str, predictions, actual_prices or {})
            line_path, heat_path = generate_charts(days=30)
            if line_path:
                print(f"      折線圖：{line_path}")
                print(f"      熱力圖：{heat_path}")
        except Exception as e:
            logger.error(f"視覺化失敗：{e}")
            print(f"      視覺化失敗：{e}")

        # 績效追蹤圖：每日方向命中率趨勢 + regime異常標記（誠實成績單，進 Obsidian）
        try:
            from performance_tracker import render as render_perf
            render_perf()
            print("      績效追蹤圖已更新：績效追蹤.md")
        except Exception as e:
            logger.debug(f"績效追蹤圖失敗：{e}")

        # Step 5b：各股深色多模型對比圖
        try:
            from rounds_chart import plot_rounds_chart
            n_ok = 0
            for ticker in TICKER_NAMES:
                try:
                    plot_rounds_chart(ticker)
                    n_ok += 1
                except Exception as e:
                    logger.debug(f"對比圖失敗（{ticker}）：{e}")
            print(f"      多模型對比圖：{n_ok}/{len(TICKER_NAMES)} 張")
        except Exception as e:
            logger.error(f"多模型對比圖失敗：{e}")

        # Step 5c：清理舊圖（只保留最近 7 天）
        try:
            from visualize import cleanup_old_charts
            removed = cleanup_old_charts(keep_days=7)
            if removed:
                print(f"      清理舊圖：刪除 {removed} 張（保留最近 7 天）")
        except Exception as e:
            logger.debug(f"清理舊圖失敗：{e}")

        # 顯示績效摘要
        print(f"\n{'─'*60}")
        print("盤後績效摘要：")
        print(f"{'─'*60}")
        if performance_report.get("summary"):
            s = performance_report["summary"]
            avg_mape = s.get('avg_mape', 'N/A')
            baseline = s.get('baseline_mape')
            beat = s.get('beat_baseline')
            if baseline:
                beat_str = f"  {'✅ 勝過' if beat else '❌ 未勝'} baseline({baseline}%)"
                print(f"  平均 MAPE：{avg_mape}%{beat_str}")
                print(f"  Baseline（猜不變）MAPE：{baseline}%")
            else:
                print(f"  平均 MAPE：{avg_mape}%")
            dir_acc = s.get('direction_accuracy', 'N/A')
            if isinstance(dir_acc, float):
                dir_acc = f"{dir_acc:.1%}"
            print(f"  方向準確率：{dir_acc}")
            print(f"  樣本數：{s.get('sample_count', 0)}")

        if performance_report.get("ticker_performance"):
            print("\n  個股績效：")
            for ticker, perf in performance_report["ticker_performance"].items():
                name = TICKER_NAMES.get(f"{ticker}.TW", TICKER_NAMES.get(f"{ticker}.TWO", ticker))
                dir_ok = "✓" if perf.get("direction_correct") else "✗" if perf.get("direction_correct") is False else "-"
                bl = perf.get("baseline_mape")
                bl_str = f" (baseline={bl}%)" if bl else ""
                print(
                    f"    {ticker} {name:<4}: "
                    f"預測={perf.get('predicted', 'N/A')}, "
                    f"實際={perf.get('actual', 'N/A')}, "
                    f"MAPE={perf.get('mape', 'N/A')}%{bl_str} {dir_ok}"
                )

        print(f"\n{'─'*60}")
        print("Reflexion 摘要：")
        print(f"{'─'*60}")
        print(reflexion_report[:500] + ("..." if len(reflexion_report) > 500 else ""))

        print(f"\n{'='*60}")
        print(f"盤後檢討完成：{datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}\n")

        return {
            "performance_report": performance_report,
            "reflexion_report": reflexion_report,
        }

    def display_strategy_table(self, strategies: list[dict]) -> None:
        """以格式化表格印出每支股票的操作策略建議。"""
        if not strategies:
            print("  （無策略資料）")
            return

        # 表頭
        header = f"{'股票':<10} {'名稱':<6} {'操作':<6} {'倉位':<8} {'進場價':>8} {'停損':>8} {'停利':>8} {'風險':<6} 理由"
        separator = "─" * 90
        print(header)
        print(separator)

        action_color = {
            "buy": "▲",
            "sell": "▼",
            "hold": "─",
            "watch": "◎",
        }

        for s in strategies:
            ticker = s.get("ticker", "N/A")
            name = s.get("name", "")[:4]
            action = s.get("action", "hold")
            size = s.get("position_size", "none")
            entry = s.get("entry_price_hint")
            sl = s.get("stop_loss_hint")
            tp = s.get("take_profit_hint")
            risk = s.get("risk_level", "medium")
            reason = (s.get("reason") or "")[:28]

            icon = action_color.get(action, " ")
            entry_str = f"{entry:.1f}" if isinstance(entry, (int, float)) else "-"
            sl_str = f"{sl:.1f}" if isinstance(sl, (int, float)) else "-"
            tp_str = f"{tp:.1f}" if isinstance(tp, (int, float)) else "-"

            print(
                f"{ticker:<10} {name:<6} {icon}{action:<5} {size:<8} "
                f"{entry_str:>8} {sl_str:>8} {tp_str:>8} {risk:<6} {reason}"
            )

        print(separator)

        # 統計
        action_counts = {}
        for s in strategies:
            a = s.get("action", "hold")
            action_counts[a] = action_counts.get(a, 0) + 1
        summary_parts = [f"{k}={v}" for k, v in sorted(action_counts.items())]
        print(f"  操作分布：{', '.join(summary_parts)}")


if __name__ == "__main__":
    print("=== 測試 TradingOrchestrator ===\n")
    orchestrator = TradingOrchestrator()

    print("執行盤前分析流程...")
    session = orchestrator.run_pre_market()

    print(f"\n盤前 session_data keys: {list(session.keys())}")

    # 模擬盤中更新
    print("\n模擬盤中更新...")
    orchestrator.run_intraday()

    # 模擬盤後（使用假實際價格）
    print("\n模擬盤後檢討（使用假實際價格）...")
    fake_actual_prices = {
        "2327": 382.0,
        "2492": 156.5,
        "5425": 78.3,
    }
    orchestrator.run_post_market(actual_prices=fake_actual_prices)

    print("\n=== 完整流程測試完成 ===")
