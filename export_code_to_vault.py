"""
把專案有用到的程式碼匯出成 Obsidian 可讀的 markdown（程式碼區塊 + 語法高亮）。
原始碼仍以 trading-system 為唯一真實來源；此處是「快照」，改了 code 重跑本檔即可同步。

用法：python export_code_to_vault.py
輸出：D:\trading-wiki\程式碼\*.md  + 程式碼\_程式總覽.md
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

SRC = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(cfg.WIKI_DIR, "程式碼")
os.makedirs(OUT, exist_ok=True)

# 分類 -> [(相對路徑, 一句說明)]
CATEGORIES = {
    "設定": [
        ("config.py", "全域設定：股票清單、特徵、日期、路徑、訓練參數"),
    ],
    "資料層": [
        ("data/fetch_stocks.py", "下載股價 + 計算 19 維特徵（技術指標 + 同業/宏觀）"),
        ("data/fetch_chips.py", "三大法人籌碼抓取（實驗後退回，保留備用）"),
    ],
    "模型層": [
        ("models/lstm_model.py", "LSTM 與 Transformer 模型定義"),
        ("models/train.py", "訓練腳本：時序切分、方向感知 loss、最佳模型選優"),
        ("models/predict.py", "預測：點預測 + 盤後時間對齊驗證"),
    ],
    "預測與評估": [
        ("quantile_forecast.py", "分位數迴歸 + 波動率正規化 CQR：50%/80% 區間 + 上漲機率"),
        ("eval_harness.py", "模型品質閘門：方向準確率 50% 基準，自動重訓 fail 股"),
        ("backtest.py", "回測框架：vs 買進持有、勝率、夏普、最大回撤"),
        ("overfit_monitor.py", "過擬合監控：訓練 vs 測試 MAPE 差距"),
        ("forecast.py", "未來多日遞迴預測（誤差錐）"),
    ],
    "Agent 層": [
        ("agents/orchestrator.py", "主控：協調盤前/中/後完整流程"),
        ("agents/news_agent.py", "新聞情緒分析 + 績效計算（含 baseline 比較）"),
        ("agents/strategy_agent.py", "策略生成 + Reflexion 反思"),
        ("agents/rule_agent.py", "CLIN 因果律學習（LATS 雙假說）"),
        ("agents/rule_manager.py", "規則庫讀寫 + 健檢（防垃圾規則）"),
        ("agents/global_snapshot.py", "國際盤前快照（昨夜美股領先指標）"),
        ("agents/peer_monitor.py", "國際同業監控 + 財報日曆"),
        ("agents/sentiment_tracker.py", "情緒連續性追蹤（新聞覆蓋觸發）"),
        ("agents/wiki_agent.py", "Wiki 知識庫讀寫"),
    ],
    "監控與推播": [
        ("drift_monitor.py", "Evidently Regime 漂移監控 + 視覺化"),
        ("intraday_monitor.py", "盤中風險監控 + Discord 推播"),
    ],
    "視覺化": [
        ("visualize.py", "折線圖 + 熱力圖 + 預測記錄"),
        ("rounds_chart.py", "深色多模型對比圖"),
    ],
    "排程入口": [
        ("run_daily.py", "每日執行：盤前/盤中/盤後"),
        ("run_retrain.py", "月度重訓 + 品質閘門 + 漂移門檻校準"),
        ("make_ppt.py", "簡報生成（給教授）"),
    ],
    "啟動腳本 (PowerShell)": [
        ("launch.ps1", "盤前/中/後排程啟動器"),
        ("health_check.ps1", "每日健康檢查"),
        ("intraday_launch.ps1", "盤中監控啟動器"),
    ],
}

LANG = {".py": "python", ".ps1": "powershell"}


def _safe(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


def export():
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    index = [f"# 📂 程式碼總覽\n",
             f"> 快照時間：{stamp}　|　真實來源：`trading-system/`，改了 code 重跑 `export_code_to_vault.py` 同步\n",
             "> 在 Obsidian 中點下方連結看各檔原始碼（已語法高亮）\n"]
    n_ok = 0
    for cat, files in CATEGORIES.items():
        index.append(f"\n## {cat}\n")
        for rel, desc in files:
            src_path = os.path.join(SRC, rel)
            note_name = f"{_safe(rel)}"
            if not os.path.exists(src_path):
                index.append(f"- ~~{rel}~~（檔案不存在，跳過）")
                continue
            with open(src_path, encoding="utf-8") as f:
                code = f.read()
            lang = LANG.get(os.path.splitext(rel)[1], "")
            lines = code.count("\n") + 1
            md = (f"# {rel}\n\n"
                  f"> {desc}\n>\n"
                  f"> 路徑：`trading-system/{rel}`　|　{lines} 行　|　快照 {stamp}\n\n"
                  f"```{lang}\n{code}\n```\n")
            with open(os.path.join(OUT, f"{note_name}.md"), "w", encoding="utf-8") as f:
                f.write(md)
            index.append(f"- [[{note_name}]] — {desc}（{lines} 行）")
            n_ok += 1

    with open(os.path.join(OUT, "_程式總覽.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(index))
    print(f"已匯出 {n_ok} 支程式 -> {OUT}")
    print(f"總覽：{os.path.join(OUT, '_程式總覽.md')}")
    return n_ok


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    export()
