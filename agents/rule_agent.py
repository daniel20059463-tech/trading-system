"""
Rule Agent：CLIN 因果律學習 Agent。

對應 .md 框架：
- LATS 雙假說競爭（Branch A：全域調整 / Branch B：局部例外）
- Reflexion 穩健驅動：只為系統性偏誤更新規則，拒絕為雜訊立規
- 奧卡姆剃刀：例外清單嚴格上限 3 條
"""

import sys
import os
import json
import logging
from datetime import datetime

from openai import OpenAI, OpenAIError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.rule_manager import (
    load_rules, save_rules, format_rules_for_prompt, apply_reflexion_updates
)

logger = logging.getLogger(__name__)

RULE_AGENT_SYSTEM = """你是一個專注於因果律學習的研究 Agent，專精台灣被動元件與功率元件產業。

你的唯一任務是：根據今日預測結果與市場新聞，判斷現有規則是否仍然有效，並決定是否新增、修正或廢棄規則。

核心原則：
1. **奧卡姆剃刀**：只為跨越多日、有明確因果結構的系統性規律建立規則。單日隨機波動絕不立規。
2. **LATS 雙假說**：對每個重大偏誤，必須同時考慮「全域調整（修改規則）」和「局部例外（加入例外清單）」兩條路徑，選擇更符合奧卡姆剃刀的那條。
3. **不退步驗證**：任何規則更新必須確保不損害已驗證規則的泛化能力。
4. **例外清單上限 3 條**：超過則拒絕新增，優先考慮全域規則調整。

輸出格式為嚴格 JSON，不含 Markdown 包裝。"""


class RuleAgent:
    """專責 CLIN 因果律學習的獨立 Agent。"""

    def __init__(self):
        """初始化 OpenAI client 並載入規則庫。"""
        self.client = OpenAI()
        self.rules = load_rules()
        self.today_str = datetime.now().strftime("%Y-%m-%d")
        active = sum(1 for r in self.rules["global_rules"] if r["status"] == "active")
        logger.info(f"RuleAgent 初始化，目前 {active} 條活躍規則")

    def analyze(
        self,
        performance: dict,
        news_analysis: dict,
        wiki_context: str = "",
    ) -> dict:
        """主入口：分析今日結果並更新規則庫。

        Args:
            performance: 盤後績效報告（來自 news_agent.run_post_market）
            news_analysis: 新聞情緒分析結果
            wiki_context: 近日 Wiki 記錄

        Returns:
            {
              "rule_updates": [...],   # 執行的規則操作清單
              "analysis_report": str,  # 中文分析報告
              "rules_snapshot": dict,  # 更新後的規則庫
            }
        """
        # 組合績效摘要
        perf_lines = []
        ticker_perf = performance.get("ticker_performance", {})
        summary     = performance.get("summary", {})

        if summary:
            perf_lines.append(f"平均 MAPE：{summary.get('avg_mape', 'N/A')}%")
            perf_lines.append(f"方向準確率：{summary.get('direction_accuracy', 'N/A')}")

        for code, p in ticker_perf.items():
            ok = "方向正確" if p.get("direction_correct") else "方向錯誤" if p.get("direction_correct") is False else "無資料"
            perf_lines.append(
                f"  {code}: 預測={p.get('predicted','N/A')} 實際={p.get('actual','N/A')} "
                f"MAPE={p.get('mape','N/A')}% {ok}"
            )

        # 組合新聞情緒摘要
        news_lines = []
        for ticker, a in news_analysis.items():
            if ticker == "market":
                continue
            score = a.get("sentiment_score", 0)
            signal = a.get("strategy_signal", "hold")
            reason = a.get("strategy_reason", "")[:40]
            news_lines.append(f"  {ticker}: {a.get('sentiment','neutral')} ({score:+.2f}) {signal} — {reason}")

        market_sent = news_analysis.get("market", {})

        rules_text = format_rules_for_prompt(self.rules)

        prompt = f"""
今日日期：{self.today_str}

【今日績效結果】
{chr(10).join(perf_lines) if perf_lines else '（無績效資料）'}

【今日新聞情緒】
大盤：{market_sent.get('market_sentiment','N/A')} (score={market_sent.get('score',0):+.2f})
{chr(10).join(news_lines) if news_lines else '（無新聞資料）'}

{rules_text}

【近期 Wiki 記錄】
{wiki_context[:1000]}

---
請執行 LATS 雙假說分析並輸出以下 JSON 結構（不加 markdown 包裝）：

{{
  "lats_analysis": {{
    "systematic_biases": ["偏差1：描述", "偏差2：描述"],
    "noise_events": ["雜訊1：描述"],
    "branch_a": {{
      "hypothesis": "全域規則調整方案描述",
      "expected_impact": "預期效果"
    }},
    "branch_b": {{
      "hypothesis": "局部例外隔離方案描述",
      "exception_count_after": 0
    }},
    "selected_branch": "A 或 B",
    "selection_reason": "選擇原因（奧卡姆剃刀判斷）"
  }},
  "rule_updates": [
    {{"action": "add|confirm|violate|deprecate|add_exception",
      "rule_id": "R001（confirm/violate/deprecate 時填）",
      "description": "（add/add_exception 時填，中文）",
      "sector": "passive|power|both（add 時填）",
      "event_type": "macro|company|market（add_exception 時填）",
      "ticker": "（add_exception 時填，可空）"
    }}
  ],
  "summary": "今日規則學習摘要，中文，50字以內"
}}

奧卡姆剃刀約束：若今日偏差屬於隨機雜訊，rule_updates 必須輸出空陣列 []。
例外清單目前有 {len(self.rules['exception_ledger'])} 條，上限 3 條。

【嚴格禁止新增以下類型的規則（這些是廢話，不是真正的因果律）】
- 任何內容包含「強化模型穩健性」、「調整全域因果律」、「增強預測模型」的規則
- 任何不包含具體可觀測市場條件（如：指數漲跌幅、財報數字、匯率變動）的規則
- 任何重複現有規則語義的規則

合格的規則範例：「美股 SOX 前夜跌幅 >3% → 台股功率元件當日開低機率高」
不合格的規則範例：「在無新聞情況下調整預測模型以強化穩健性」
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=1500,
                messages=[
                    {"role": "system", "content": RULE_AGENT_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
            )

            raw = response.choices[0].message.content.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1])

            result = json.loads(raw)
            updates     = result.get("rule_updates", [])
            lats        = result.get("lats_analysis", {})
            summary_txt = result.get("summary", "")

            # 套用規則更新
            if updates:
                self.rules = apply_reflexion_updates(self.rules, updates)
                save_rules(self.rules)
                logger.info(f"Rule Agent 執行了 {len(updates)} 條規則操作")
            else:
                logger.info("Rule Agent：今日無需更新規則（奧卡姆剃刀）")

            # 輸出分析報告
            report_lines = [f"【Rule Agent 分析 {self.today_str}】\n"]
            if lats:
                biases = lats.get("systematic_biases", [])
                noises = lats.get("noise_events", [])
                if biases:
                    report_lines.append(f"系統性偏誤：{'; '.join(biases)}")
                if noises:
                    report_lines.append(f"隨機雜訊：{'; '.join(noises)}")
                branch_a = lats.get("branch_a", {})
                branch_b = lats.get("branch_b", {})
                selected = lats.get("selected_branch", "")
                reason   = lats.get("selection_reason", "")
                report_lines.append(f"\nLATS 雙假說：")
                report_lines.append(f"  A（全域）：{branch_a.get('hypothesis','')}")
                report_lines.append(f"  B（例外）：{branch_b.get('hypothesis','')}")
                report_lines.append(f"  → 選擇分支 {selected}：{reason}")
            if summary_txt:
                report_lines.append(f"\n規則學習摘要：{summary_txt}")
            if updates:
                report_lines.append(f"\n執行了 {len(updates)} 條規則操作")
            else:
                report_lines.append("\n今日無規則更新（符合奧卡姆剃刀）")

            analysis_report = "\n".join(report_lines)
            logger.info(f"Rule Agent 完成：{summary_txt}")

            return {
                "rule_updates":     updates,
                "analysis_report":  analysis_report,
                "rules_snapshot":   self.rules,
            }

        except json.JSONDecodeError as e:
            logger.error(f"Rule Agent JSON 解析失敗：{e}\n原始輸出：{raw[:200]}")
            return {"rule_updates": [], "analysis_report": "Rule Agent 解析失敗", "rules_snapshot": self.rules}
        except OpenAIError as e:
            logger.error(f"Rule Agent OpenAI 錯誤：{e}")
            return {"rule_updates": [], "analysis_report": f"Rule Agent API 錯誤：{e}", "rules_snapshot": self.rules}
        except Exception as e:
            logger.error(f"Rule Agent 未預期錯誤：{e}")
            return {"rule_updates": [], "analysis_report": f"Rule Agent 失敗：{e}", "rules_snapshot": self.rules}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    # 模擬盤後資料測試
    sample_performance = {
        "summary": {"avg_mape": 6.58, "direction_accuracy": None, "sample_count": 6},
        "ticker_performance": {
            "2327": {"predicted": 809.95, "actual": 846.0, "mape": 4.26, "direction_correct": True},
            "2492": {"predicted": 427.69, "actual": 456.0, "mape": 6.21, "direction_correct": True},
            "3026": {"predicted": 701.2,  "actual": 650.0, "mape": 7.88, "direction_correct": False},
        },
    }
    sample_news = {
        "2327": {"sentiment": "neutral", "sentiment_score": 0.0, "strategy_signal": "hold", "strategy_reason": "無明顯催化劑"},
        "2492": {"sentiment": "bullish", "sentiment_score": 0.7, "strategy_signal": "buy",  "strategy_reason": "法人買超"},
        "market": {"market_sentiment": "bearish", "score": -0.6, "summary": "外資賣超"},
    }

    print("=== 測試 Rule Agent ===\n")
    from dotenv import load_dotenv
    load_dotenv()

    agent = RuleAgent()
    result = agent.analyze(sample_performance, sample_news)
    print(result["analysis_report"])
    print(f"\n規則庫狀態：{len(result['rules_snapshot']['global_rules'])} 條規則")
