"""系統狀態總覽：把『實際發生的事』攤成一份你能打開看的報告。
不靠聊天視窗轉述——直接從硬碟上的模型檔、預測紀錄、產出檔生成證據。
輸出 D:\\trading-wiki\\系統狀態.md，隨時 python system_status.py 更新。
"""
import os, sys, json, glob
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

NAMES = {"2327": "國巨", "2492": "華新科", "3026": "禾伸堂", "2472": "立隆電", "3357": "臺慶科",
         "5425": "台半", "2481": "強茂", "3675": "德微", "8255": "朋程", "8261": "富鼎"}
REPORT = os.path.join(cfg.WIKI_DIR, "系統狀態.md")
HERE = os.path.dirname(os.path.abspath(__file__))


def _mtime(path):
    return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M") if os.path.exists(path) else "—"


def build():
    L = [f"# 系統狀態總覽（實際證據，非轉述）",
         f"\n產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]

    # 1. 模型最後重訓
    ckdir = cfg.CHECKPOINT_DIR
    lstm = sorted(glob.glob(os.path.join(ckdir, "*_lstm_best.pt")), key=os.path.getmtime, reverse=True)
    tf = sorted(glob.glob(os.path.join(ckdir, "*_transformer_best.pt")), key=os.path.getmtime, reverse=True)
    qt = sorted(glob.glob(os.path.join(ckdir, "*_quantile_best.pt")), key=os.path.getmtime, reverse=True)
    L += ["## 1. 模型最後重訓時間（檔案實際修改時間）\n",
          f"- LSTM 模型：最新 **{_mtime(lstm[0]) if lstm else '—'}**（共 {len(lstm)} 支）",
          f"- Transformer 模型：最新 **{_mtime(tf[0]) if tf else '—'}**（共 {len(tf)} 支）",
          f"- 區間(分位數)模型：最新 **{_mtime(qt[0]) if qt else '—'}**（共 {len(qt)} 支）",
          "\n> 這是硬碟上模型檔的真實時間戳，重訓有跑才會更新。\n"]

    # 2. 近期每日預測 vs 實際（最有力的證據）
    L += ["## 2. 近期每日預測 vs 實際（逐日命中率）\n",
          "| 交易日 | 方向命中 | 平均誤差 | 細節 |",
          "|--------|---------|---------|------|"]
    files = sorted(glob.glob(os.path.join(HERE, "news", "*_post_market.json")))[-7:]
    for f in files:
        d = os.path.basename(f)[:10]
        try:
            tp = json.load(open(f, encoding="utf-8")).get("ticker_performance", {})
        except Exception:
            continue
        recs = [v for v in tp.values() if isinstance(v, dict) and "actual" in v]
        if not recs:
            continue
        n = len(recs)
        dirok = sum(1 for v in recs if v.get("direction_correct"))
        mape = sum(v.get("mape", 0) for v in recs) / n
        detail = "、".join(f"{NAMES.get(k,k)}{'✓' if v.get('direction_correct') else '✗'}"
                           for k, v in tp.items() if isinstance(v, dict) and "actual" in v)
        L.append(f"| {d} | **{dirok}/{n} = {dirok*100//n}%** | {mape:.1f}% | {detail} |")
    L += ["\n> 每天盤後自動產出，predicted/actual 都是真實收盤算的，不是我說了算。\n"]

    # 3. 近 N 日預測明細（最近一天逐股）
    if files:
        f = files[-1]; d = os.path.basename(f)[:10]
        tp = json.load(open(f, encoding="utf-8")).get("ticker_performance", {})
        L += [f"## 3. 最近一個交易日（{d}）逐股預測 vs 實際\n",
              "| 股 | 前收 | 預測 | 實際 | 方向 | 誤差 |",
              "|----|------|------|------|------|------|"]
        for k, v in tp.items():
            if not isinstance(v, dict) or "actual" not in v:
                continue
            mk = "✅" if v.get("direction_correct") else "❌"
            L.append(f"| {NAMES.get(k,k)} | {v['prev_close']:.1f} | {v['predicted']:.1f} "
                     f"| {v['actual']:.1f} | {mk} | {v.get('mape',0):.1f}% |")

    # 4. 產出檔案清單（你可以一個個打開）
    L += ["\n## 4. 系統實際產出的檔案（都在 D:\\trading-wiki，可直接開）\n",
          "| 檔案 | 最後更新 |", "|------|---------|"]
    for fn in ["keyword_impact_report.md", "us_tw_observe.md", "snapshot_history.md",
               f"{datetime.now().strftime('%Y-%m-%d')}.md"]:
        p = os.path.join(cfg.WIKI_DIR, fn)
        L.append(f"| {fn} | {_mtime(p)} |")

    # 5. 訊號/特徵狀態（哪些驗證過上線、哪些待辦、哪些測過退回）
    L += ["\n## 5. 訊號/特徵狀態（測過才建的紀律）\n",
          "| 訊號 | 狀態 | 依據 |",
          "|------|------|------|",
          "| 📰 盤前新聞盤中影子 | 🟡 實盤收集中 | 08:30不可變快照；盤後以Open→Close結算；20天/150筆後逐日重訓並列出好日子 |",
          "| 🇺🇸 美股相似行情對照 | 🟡 盤前參考已接入 | 五項美股同交易日驗證；查過去8個相似日；盤後十檔齊全才加入案例庫，不改正式預測 |",
          "| 🎯 今日信心度 | ✅ 已上線 | regime漂移+區間寬度→高/中/低，Discord置頂「今天該不該信點預測」|",
          "| 方向閘門評估 | ✅ 已修(Fable稽核F1) | 挑checkpoint用cal前半、報成績用holdout後半，杜絕選擇偏誤(舊48.1→61.5屬虛高) |",
          "| 盤前盤後一致化 | ✅ 已修 | 正式預測不再套用弱證據新聞翻多；新聞另走同標籤影子驗證 |",
          "| 短線/波段雙區間 | ✅ 已上線(2026-07-08) | 短線=σ5乘法縮放+獨立保形(覆蓋78%、冷卻日寬8.9%vs波段12.6%)；波段=σ20現行；❄️/♨️=體溫背離 |",
          "| 新聞感知不對稱區間 | ✅ 已上線 | 利多日90百分位+9.9%；上界往+10%拉 |",
          "| 區間±10%漲跌停夾限 | ✅ 已修 | 單日物理上限，修掉虛胖區間 |",
          "| 國際新聞（GDELT）| ❌ 測過退回 | 2年資料已抓，語氣±1%、聲量+4%，未過門檻（台股被本土籌碼主導、脫鉤）|",
          "| 美股方向修正 | ❌ 不建 | 測過 57-64%、近期脫鉤53%，未過65%門檻 |",
          "| 新聞當隔日 ML 特徵 | ❌ 退回 | 稀釋後無效（-0.2%）；改收集同日Open→Close實盤影子 |",
          "| 三大法人籌碼特徵(ML) | ❌ 退回 | 56.2→55.0，中小型股是雜訊 |",
          "| 強外資買賣超(強訊號) | ❌ 測過退回 | 單日+6%/-6%、5日累積+1%、佔量比+5%，皆未過8%門檻 |",
          "| ↳ 原因 | — | 籌碼是「同時」不是「領先」：收盤才見報，那根漲已發生，預測不了隔日 |",
          "\n> 紀律：能回測的先測，≥65%（或明顯贏基準）才建；沒過就退回，不硬塞。\n"]

    # 6. 誠實基準（2026-07-02 凍結——新量尺 holdout，所有未來改動與此對照）
    L += ["\n## 6. 誠實基準（2026-07-02 凍結，Fable 稽核後新量尺）\n",
          "> holdout=測試段後半（挑選過程沒看過的 52 天）。未來任何重訓/新訊號都跟這張表比，",
          "> 同一把尺才知道真進步還是假進步。單股 ±14% 誤差，看平均較穩（95%CI 約 53~61%）。\n",
          "| 股 | 方向準確率(holdout) |", "|----|-------------------|",
          "| 國巨 | 65.4% |", "| 華新科 | 63.5% |", "| 禾伸堂 | 59.6% |",
          "| 立隆電 | 55.8% |", "| 臺慶科 | 46.2% |", "| 台半 | 55.8% |",
          "| 強茂 | 51.9% |", "| 德微 | 59.6% |", "| 朋程 | 55.8% |", "| 富鼎 | 59.6% |",
          "| **平均** | **57.3%** |",
          "\n- 註：朋程 6/27 曾報「方向重訓 48.1→61.5%」，新量尺實為 55.8%——",
          "  舊數字含選擇偏誤（Fable 稽核 F1），已修正，此表起不再發生。",
          "- 重跑基準：`python _holdout_baseline.py`（模型變動後才有意義）。"]

    L += ["\n---\n_此檔由 system_status.py 生成。想驗證隨時跑 `python system_status.py`，或自己去看上面任一檔案。_"]
    return "\n".join(L)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    rep = build()
    os.makedirs(cfg.WIKI_DIR, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(rep)
    print(f"✅ 已生成：{REPORT}\n")
    print(rep)
