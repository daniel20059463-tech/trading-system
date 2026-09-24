"""
生成專案簡報（給教授）：研究目標 / 如何研究 / 策略 / ML訓練 / 成果。
用法：python make_ppt.py
輸出：D:\trading-wiki\半導體類股預測系統_簡報.pptx
"""
import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

sys.stdout.reconfigure(encoding="utf-8")

CHART_DIR = r"D:\trading-wiki\charts"
OUT_PATH  = r"D:\trading-wiki\半導體類股預測系統_簡報.pptx"
FONT      = "Microsoft JhengHei"

# 配色
NAVY   = RGBColor(0x1A, 0x2B, 0x4A)
BLUE   = RGBColor(0x21, 0x96, 0xF3)
TEAL   = RGBColor(0x00, 0x96, 0x88)
GREY   = RGBColor(0x37, 0x47, 0x4F)
LIGHT  = RGBColor(0xEC, 0xF2, 0xF9)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
RED    = RGBColor(0xC6, 0x28, 0x28)
GREEN  = RGBColor(0x2E, 0x7D, 0x32)

prs = Presentation()
prs.slide_width  = Inches(13.333)   # 16:9
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height


def _set(run, size, color=GREY, bold=False, italic=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.italic = italic


def _box(slide, l, t, w, h):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    return tf


def _rect(slide, l, t, w, h, color):
    from pptx.enum.shapes import MSO_SHAPE
    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    sp.fill.solid(); sp.fill.fore_color.rgb = color
    sp.line.fill.background()
    return sp


def header(slide, title, num=None):
    """標題列：左上色塊 + 標題。"""
    _rect(slide, 0, 0, SW, Inches(1.05), NAVY)
    _rect(slide, 0, Inches(1.05), SW, Pt(4), BLUE)
    tf = _box(slide, Inches(0.5), Inches(0.18), Inches(12), Inches(0.8))
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = title; _set(r, 26, WHITE, bold=True)
    if num:
        tn = _box(slide, Inches(11.8), Inches(0.18), Inches(1.2), Inches(0.8))
        tn.vertical_anchor = MSO_ANCHOR.MIDDLE
        pn = tn.paragraphs[0]; pn.alignment = PP_ALIGN.RIGHT
        rn = pn.add_run(); rn.text = num; _set(rn, 14, BLUE, bold=True)


def bullets(slide, items, top=1.45, left=0.7, width=12.0, size=18, gap=8):
    """items: list of (text, level, color)。level 0/1/2。"""
    tf = _box(slide, Inches(left), Inches(top), Inches(width), Inches(5.6))
    first = True
    for it in items:
        text = it[0]; level = it[1] if len(it) > 1 else 0
        color = it[2] if len(it) > 2 else GREY
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        prefix = ["● ", "– ", "· "][min(level, 2)]
        bullet_color = [BLUE, TEAL, GREY][min(level, 2)]
        rb = p.add_run(); rb.text = prefix
        _set(rb, size - level * 2, bullet_color, bold=(level == 0))
        rt = p.add_run(); rt.text = text
        _set(rt, size - level * 2, color, bold=(level == 0))


def section_divider(num, zh, en):
    s = prs.slides.add_slide(BLANK)
    _rect(s, 0, 0, SW, SH, NAVY)
    _rect(s, Inches(0.7), Inches(3.0), Inches(0.18), Inches(1.5), BLUE)
    tf = _box(s, Inches(1.1), Inches(2.7), Inches(11), Inches(2.2))
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = f"{num}"; _set(r, 22, BLUE, bold=True)
    p2 = tf.add_paragraph()
    r2 = p2.add_run(); r2.text = zh; _set(r2, 40, WHITE, bold=True)
    p3 = tf.add_paragraph()
    r3 = p3.add_run(); r3.text = en; _set(r3, 18, RGBColor(0x90, 0xA4, 0xBC))
    return s


def flow_box(slide, x, y, w, h, title, sub, fill, tcolor=WHITE):
    """流程圖方塊：圓角矩形 + 標題(粗) + 副標。x,y,w,h 單位吋。"""
    from pptx.enum.shapes import MSO_SHAPE
    sp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(x), Inches(y), Inches(w), Inches(h))
    sp.fill.solid(); sp.fill.fore_color.rgb = fill
    sp.line.color.rgb = WHITE; sp.line.width = Pt(1)
    tf = sp.text_frame; tf.word_wrap = True
    tf.margin_top = Pt(2); tf.margin_bottom = Pt(2)
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = title; _set(r, 13, tcolor, bold=True)
    if sub:
        p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run(); r2.text = sub; _set(r2, 9.5, tcolor)
    return sp


def flow_arrow(slide, x, y, w, h, color=BLUE, shape="right"):
    from pptx.enum.shapes import MSO_SHAPE
    kind = {"right": MSO_SHAPE.RIGHT_ARROW, "down": MSO_SHAPE.DOWN_ARROW,
            "left": MSO_SHAPE.LEFT_ARROW, "up": MSO_SHAPE.UP_ARROW}[shape]
    sp = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.fill.solid(); sp.fill.fore_color.rgb = color
    sp.line.fill.background()
    return sp


def architecture_slide():
    s = prs.slides.add_slide(BLANK)
    header(s, "系統架構流程圖", "2")

    # 上層：三大領先指標輸入
    inp = [("🌍 國際盤前快照", "昨夜美股/ADR", BLUE),
           ("🌐 國際同業監控", "村田/ON 異動", BLUE),
           ("📰 新聞情緒", "LLM 分析", BLUE)]
    ix = [1.1, 5.55, 10.0]
    for (t, sub, c), x in zip(inp, ix):
        flow_box(s, x, 1.45, 2.3, 0.72, t, sub, c)
    # 匯入箭頭（往下）
    flow_arrow(s, 6.45, 2.25, 0.45, 0.45, TEAL, "down")

    # 主流程：5 階段（y=3.0）
    stages = [
        ("資料抓取", "yfinance 3年日線", GREY),
        ("特徵工程", "19維 技術+關聯", GREY),
        ("雙模型", "LSTM + Transformer", NAVY),
        ("動態集成", "+ 分位數區間", NAVY),
        ("預測輸出", "方向 / 區間 / P(漲)", TEAL),
    ]
    bx, bw, by, bh = 0.55, 2.0, 3.0, 1.05
    gap = 0.5
    for i, (t, sub, c) in enumerate(stages):
        x = bx + i * (bw + gap)
        flow_box(s, x, by, bw, bh, t, sub, c)
        if i < len(stages) - 1:
            flow_arrow(s, x + bw + 0.03, by + 0.32, gap - 0.06, 0.42, BLUE, "right")

    # 下層：回饋迴圈（盤後驗收 → 自我進化 → 回饋重訓）
    flow_arrow(s, 11.55, 4.15, 0.45, 0.5, RED, "down")  # 輸出往下到驗收
    fb = [("盤後驗收", "MAPE/方向/baseline", RED),
          ("自我進化", "Reflexion / Rule", RED),
          ("回饋重訓", "不退步保護", GREEN)]
    fx = [9.45, 6.45, 3.45]
    for (t, sub, c), x in zip(fb, fx):
        flow_box(s, x, 4.75, 2.3, 0.72, t, sub, c)
    # 驗收 -> 進化 -> 重訓（向左箭頭）
    flow_arrow(s, 9.15, 4.95, 0.35, 0.42, GREY, "left")
    flow_arrow(s, 6.15, 4.95, 0.35, 0.42, GREY, "left")
    # 重訓回饋到模型（往上箭頭，連回主流程）
    flow_arrow(s, 2.4, 4.1, 0.42, 0.6, GREEN, "up")
    lbl = _box(s, Inches(1.0), Inches(4.78), Inches(2.3), Inches(0.66))
    lbl.vertical_anchor = MSO_ANCHOR.MIDDLE
    pl = lbl.paragraphs[0]; pl.alignment = PP_ALIGN.CENTER
    rl = pl.add_run(); rl.text = "月度回饋"; _set(rl, 10, GREEN, italic=True)
    # 可信度閘門 橫幅
    flow_box(s, 0.55, 6.0, 12.2, 0.7,
             "可信度閘門：Evidently Regime 漂移監控  ·  方向準確率 50% 閘門  ·  過擬合監控  ·  區間覆蓋率校驗",
             "", LIGHT, tcolor=NAVY)

    # 小註
    cap = _box(s, Inches(0.55), Inches(6.78), Inches(12.2), Inches(0.5))
    pc = cap.paragraphs[0]; pc.alignment = PP_ALIGN.CENTER
    rc = pc.add_run()
    rc.text = "資料→特徵→雙模型→集成→輸出，每日盤後驗收回饋進化；底層閘門全程監控「能不能信」"
    _set(rc, 12, GREY, italic=True)
    return s


def image_slide(title, num, img, caption, img_w=8.6):
    s = prs.slides.add_slide(BLANK)
    header(s, title, num)
    if os.path.exists(img):
        w = Inches(img_w)
        pic = s.shapes.add_picture(img, 0, Inches(1.5), width=w)
        # 置中
        pic.left = int((SW - pic.width) / 2)
        cap = _box(s, Inches(0.7), Inches(6.7), Inches(12), Inches(0.6))
        pc = cap.paragraphs[0]; pc.alignment = PP_ALIGN.CENTER
        rc = pc.add_run(); rc.text = caption; _set(rc, 13, GREY, italic=True)
    else:
        bullets(s, [(f"（圖檔不存在：{img}）", 0, RED)])
    return s


# ══════════════════════════════════════════════════════════════════
# 封面
# ══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
_rect(s, 0, 0, SW, SH, NAVY)
_rect(s, 0, Inches(4.3), SW, Pt(3), BLUE)
tf = _box(s, Inches(0.9), Inches(2.0), Inches(11.5), Inches(2.3))
p = tf.paragraphs[0]
r = p.add_run(); r.text = "台灣半導體類股"; _set(r, 40, WHITE, bold=True)
p2 = tf.add_paragraph()
r2 = p2.add_run(); r2.text = "自我進化交易預測系統"; _set(r2, 40, BLUE, bold=True)
tf2 = _box(s, Inches(0.9), Inches(4.55), Inches(11.5), Inches(1.6))
for txt, sz, col in [
    ("LSTM + Transformer 集成 × 分位數迴歸 × 自我進化規則學習", 18, RGBColor(0xC5, 0xD3, 0xE5)),
    ("結合美股領先指標、新聞情緒、籌碼漂移監控的多層預測框架", 16, RGBColor(0x90, 0xA4, 0xBC)),
]:
    pp = tf2.add_paragraph(); rr = pp.add_run(); rr.text = txt; _set(rr, sz, col)
tf3 = _box(s, Inches(0.9), Inches(6.4), Inches(11.5), Inches(0.6))
pp = tf3.paragraphs[0]; rr = pp.add_run()
rr.text = "研究報告　|　2026"; _set(rr, 14, RGBColor(0x90, 0xA4, 0xBC))

# 目錄
s = prs.slides.add_slide(BLANK)
header(s, "簡報大綱  Outline")
bullets(s, [
    ("一、研究目標　Research Goals", 0),
    ("二、研究方法　Methodology — 四層領先指標鏈、資料與驗證", 0),
    ("三、策略　Strategy — 集成模型與自我進化框架", 0),
    ("四、機器學習如何訓練　ML Training — 目標、損失、不確定性量化", 0),
    ("五、成果　Results — 預測績效、科學方法示範、可信度監控", 0),
    ("六、限制與未來工作　Limitations & Future Work", 0),
], top=1.7, size=22, gap=14)

# ══════════════════════════════════════════════════════════════════
# 一、研究目標
# ══════════════════════════════════════════════════════════════════
section_divider("PART 1", "研究目標", "Research Goals")

s = prs.slides.add_slide(BLANK)
header(s, "研究目標", "1")
bullets(s, [
    ("核心目標：預測台灣半導體 10 支個股的「隔日走勢」", 0, NAVY),
    ("被動元件 5 支：國巨、華新科、禾伸堂、立隆電、臺慶科", 1),
    ("功率元件 5 支：台半、強茂、德微、朋程、富鼎", 1),
    ("三個層次的研究問題：", 0, NAVY),
    ("① 方向：明日漲或跌？（比丟硬幣 50% 準）", 1),
    ("② 價位：收盤大約落在哪？（點預測 + 區間）", 1),
    ("③ 可信度：今天這個預測「能不能信」？", 1),
    ("研究定位：不追求「神準」，而是建立一個能", 0, NAVY),
    ("自我驗證、自我進化、且誠實知道自身極限的預測系統", 1, TEAL),
], top=1.4, size=18)

# ══════════════════════════════════════════════════════════════════
# 二、研究方法
# ══════════════════════════════════════════════════════════════════
section_divider("PART 2", "研究方法", "Methodology")

s = prs.slides.add_slide(BLANK)
header(s, "研究方法（一）：四層領先指標鏈", "2")
bullets(s, [
    ("台股 08:30 開盤前，依時間領先性逐層收斂判斷：", 0, NAVY),
    ("🌍 第一層 — 國際盤前快照", 0, BLUE),
    ("昨夜美股（費半 SOX、台積電/鴻海 ADR、那斯達克）是最強領先指標", 1),
    ("🌐 第二層 — 國際同業監控", 0, BLUE),
    ("村田/TDK（被動）、ON/Vishay（功率）價格異動與財報日曆", 1),
    ("📰 第三層 — 新聞情緒分析", 0, BLUE),
    ("LLM 分析個股新聞，連續強訊號觸發「新聞優先覆蓋」", 1),
    ("📊 第四層 — 機器學習預測", 0, BLUE),
    ("LSTM + Transformer 集成，整合上述訊號做最終預測", 1),
], top=1.3, size=17)

s = prs.slides.add_slide(BLANK)
header(s, "研究方法（二）：資料與驗證", "2")
bullets(s, [
    ("資料來源與規模", 0, NAVY),
    ("近 3 年日線（每股約 668 個交易日），yfinance 自動更新", 1),
    ("19 維特徵：技術指標 + 同業/宏觀漲跌幅", 1),
    ("技術面：MA、RSI、MACD、布林通道、ATR、量價比、報酬率", 2),
    ("關聯面：美元台幣、台股加權、國際同業漲跌幅", 2),
    ("時序交叉驗證（嚴格防資料洩漏）", 0, NAVY),
    ("80% 訓練 / 20% 測試，依時間順序切分、不洗牌", 1),
    ("正規化（MinMaxScaler）只用訓練集統計量 fit", 1),
    ("自我進化框架：Reflexion（反思）/ CLIN（因果律）/ LATS（雙假說）", 0, NAVY),
    ("每日盤後反思預測偏誤是「系統性錯誤」還是「隨機雜訊」", 1),
], top=1.3, size=17)

architecture_slide()

# ══════════════════════════════════════════════════════════════════
# 三、策略
# ══════════════════════════════════════════════════════════════════
section_divider("PART 3", "策略", "Strategy")

s = prs.slides.add_slide(BLANK)
header(s, "策略（一）：集成模型架構", "3")
bullets(s, [
    ("雙模型集成（Ensemble）", 0, NAVY),
    ("LSTM：擅長捕捉時序記憶與趨勢", 1),
    ("Transformer：自注意力捕捉跨日關聯", 1),
    ("動態加權：驗證損失越低的模型權重越高", 1),
    ("多 Agent 協作", 0, NAVY),
    ("NewsAgent：盤前/中/後三時段新聞情緒分析", 1),
    ("StrategyAgent：整合 ML + 新聞 + 規則產出操作建議", 1),
    ("RuleAgent：學習因果律（如「美股財報超預期→MLCC需求回升」）", 1),
    ("風控原則：奧卡姆剃刀", 0, NAVY),
    ("只為跨多日的系統性規律立規，單日雜訊絕不立規", 1, TEAL),
], top=1.3, size=17)

s = prs.slides.add_slide(BLANK)
header(s, "策略（二）：自我進化機制", "3")
bullets(s, [
    ("每日盤後自動執行的學習迴圈：", 0, NAVY),
    ("1. 績效評估：計算 MAPE、方向準確率，並與 baseline 比較", 1),
    ("2. Reflexion 反思：辨識偏誤屬系統性或隨機性", 1),
    ("3. Rule Agent（LATS 雙假說）：", 1),
    ("分支 A 全域規則調整 vs 分支 B 局部例外，擇優", 2),
    ("4. 規則庫健檢：自動偵測規則暴增/語義重複（防退化）", 1),
    ("月度自動重訓：抓最新資料重訓，並通過品質閘門驗收", 0, NAVY),
    ("關鍵設計：每一步都有「不退步」保護", 0, NAVY),
    ("規則更新、模型重訓都須確認不損害既有泛化能力才採用", 1, TEAL),
], top=1.3, size=17)

# ══════════════════════════════════════════════════════════════════
# 四、機器學習如何訓練
# ══════════════════════════════════════════════════════════════════
section_divider("PART 4", "機器學習如何訓練", "Machine Learning Training")

s = prs.slides.add_slide(BLANK)
header(s, "ML 訓練（一）：目標與損失設計", "4")
bullets(s, [
    ("預測目標：漲跌幅 pct_change，而非絕對價格", 0, NAVY),
    ("理由：價格是非平穩序列，直接預測會「趨勢漂移」", 1),
    ("漲跌幅近似平穩，再由 pct 還原回價格", 1),
    ("損失函數：MSE + 方向感知 Hinge 損失", 0, NAVY),
    ("純 MSE 只顧價格幅度，不管漲跌方向", 1),
    ("加方向懲罰：預測漲跌方向猜錯 → 額外受罰", 1),
    ("以「驗證集方向準確率」選最佳模型，而非只看 MSE", 1),
    ("訓練設定", 0, NAVY),
    ("30 日滑動視窗、Adam、ReduceLROnPlateau、梯度裁剪、早停式選優", 1),
], top=1.3, size=17)

s = prs.slides.add_slide(BLANK)
header(s, "ML 訓練（二）：不確定性量化", "4")
bullets(s, [
    ("分位數迴歸（Quantile Regression）— 給區間而非單點", 0, NAVY),
    ("Pinball Loss 同時學 q10/q25/q50/q75/q90 五條線", 1),
    ("輸出「明日大概率落在 X~Y」的 50% 與 80% 區間", 1),
    ("保形校準（CQR）+ 波動率正規化 — 讓區間「可信」", 0, NAVY),
    ("純分位數迴歸樣本外覆蓋率僅 ~61%，名不副實", 1),
    ("以校準集修正區間寬度，並依當期波動率自適應縮放", 1),
    ("校準後 80% 區間實測覆蓋率回升至 ~77%", 1, TEAL),
    ("品質控管三道閘門", 0, NAVY),
    ("過擬合監控（訓練vs測試MAPE差距）、方向準確率 50% 閘門、覆蓋率檢驗", 1),
], top=1.3, size=17)

# ══════════════════════════════════════════════════════════════════
# 五、成果
# ══════════════════════════════════════════════════════════════════
section_divider("PART 5", "成果", "Results")

s = prs.slides.add_slide(BLANK)
header(s, "成果（一）：核心績效指標", "5")
bullets(s, [
    ("方向準確率：平均 56.8%（10 支全數 > 50% 丟硬幣基準）", 0, GREEN),
    ("範圍 51.9% ~ 60.6%；穩定贏過隨機，具正期望值", 1),
    ("價格誤差 MAPE：約 2.6%，勝過「猜不變」naive baseline", 0, GREEN),
    ("區間覆蓋率：80% 區間實測 ~77%、50% 區間 ~46%（校準後）", 0, GREEN),
    ("方法論意義", 0, NAVY),
    ("日線方向預測理論天花板約 60%（市場接近效率）", 1),
    ("系統穩定逼近此上限，且能量化「對每個預測的把握度」", 1, TEAL),
], top=1.4, size=18)

image_slide("成果（二）：MAPE 熱力圖（越綠越準）", "5",
            os.path.join(CHART_DIR, "2026-06-12_heatmap.png"),
            "每格＝某股某日預測誤差%；可一眼看出哪天哪支失準（右側偏黃對應大盤暴漲日）")

image_slide("成果（三）：預測 vs 實際收盤", "5",
            os.path.join(CHART_DIR, "2026-06-12_linechart.png"),
            "藍＝實際、紅虛線＝預測；10 支個股逐日對照，角落標註各股 MAPE", img_w=8.0)

s = prs.slides.add_slide(BLANK)
header(s, "成果（四）：科學方法示範 — 籌碼面實驗", "5")
bullets(s, [
    ("假設：加入「三大法人買賣超」籌碼特徵應能提升準確率", 0, NAVY),
    ("實作：FinMind 抓 3 年法人資料，4 個正規化特徵，重訓全部模型", 0, NAVY),
    ("驗證（held-out 測試集嚴格比較）：", 0, NAVY),
    ("方向準確率 56.2% → 55.0%　無提升，反而小幅變差", 1, RED),
    ("解讀：外資籌碼對中小型功率股是雜訊，且增加過擬合風險", 1),
    ("決策：完全退回，回到 56.8%", 0, NAVY),
    ("意義：baseline 閘門擋住了「自我欺騙」", 0, TEAL),
    ("沒有 held-out 驗證的人會把這 4 個特徵留著、誤以為變強了", 1, TEAL),
], top=1.35, size=17)

image_slide("成果（五）：可信度監控 — Regime 漂移偵測", "5",
            os.path.join(CHART_DIR, "2026-06-13_2327.TW_drift.png"),
            "Evidently：訓練期(藍) vs 近期(紅)特徵分布；錯開＝市場進入異常 regime，模型可信度下降")

# ══════════════════════════════════════════════════════════════════
# 六、限制與未來
# ══════════════════════════════════════════════════════════════════
section_divider("PART 6", "限制與未來工作", "Limitations & Future Work")

s = prs.slides.add_slide(BLANK)
header(s, "限制與未來工作", "6")
bullets(s, [
    ("誠實的限制", 0, NAVY),
    ("日線方向預測天花板約 60%，本系統已接近，難再大幅提升", 1),
    ("非平穩性：市場 regime 漂移會使區間在暴衝盤被突破", 1),
    ("純技術面對「外資主導的無差別暴漲」有結構性盲區", 1),
    ("未來工作", 0, NAVY),
    ("選擇性出手（Abstention）：只在高把握日預測，提升整體勝率", 1),
    ("改預測 5 日趨勢（訊噪比更高）", 1),
    ("Regime 漂移預警與模型自動切換", 1),
    ("核心貢獻：一個會說「我何時不準」的系統，", 0, TEAL),
    ("比一個假裝精確的系統，對風險控制更有價值", 1, TEAL),
], top=1.3, size=17)

# 結尾
s = prs.slides.add_slide(BLANK)
_rect(s, 0, 0, SW, SH, NAVY)
tf = _box(s, Inches(0.9), Inches(2.8), Inches(11.5), Inches(2))
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "謝謝指教"; _set(r, 40, WHITE, bold=True)
p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
r2 = p2.add_run()
r2.text = "方向比價位重要 · 區間比點位可信 · 知道何時不準最重要"
_set(r2, 18, BLUE)

try:
    prs.save(OUT_PATH)
    saved = OUT_PATH
except PermissionError:
    saved = OUT_PATH.replace(".pptx", "_new.pptx")
    prs.save(saved)
    print("（原檔開啟中無法覆寫，已另存新檔）")
print(f"簡報已生成：{saved}")
print(f"總頁數：{len(prs.slides._sldIdLst)}")
