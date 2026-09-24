"""生成 LLM Wiki 系統簡報。用法：python -m llm_wiki.make_wiki_ppt"""
import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

sys.stdout.reconfigure(encoding="utf-8")
OUT = r"D:\trading-wiki\LLM_Wiki_簡報.pptx"
FONT = "Microsoft JhengHei"
NAVY = RGBColor(0x1A, 0x2B, 0x4A); BLUE = RGBColor(0x21, 0x96, 0xF3)
TEAL = RGBColor(0x00, 0x96, 0x88); GREY = RGBColor(0x37, 0x47, 0x4F)
WHITE = RGBColor(0xFF, 0xFF, 0xFF); GREEN = RGBColor(0x2E, 0x7D, 0x32)
LIGHT = RGBColor(0xEC, 0xF2, 0xF9); AMBER = RGBColor(0xE6, 0x7E, 0x00)

prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]; SW, SH = prs.slide_width, prs.slide_height


def _set(r, sz, c=GREY, b=False, i=False):
    r.font.name = FONT; r.font.size = Pt(sz); r.font.color.rgb = c; r.font.bold = b; r.font.italic = i


def _box(s, l, t, w, h):
    tb = s.shapes.add_textbox(l, t, w, h); tb.text_frame.word_wrap = True; return tb.text_frame


def _rect(s, l, t, w, h, c):
    sp = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    sp.fill.solid(); sp.fill.fore_color.rgb = c; sp.line.fill.background(); return sp


def header(s, title, num=None):
    _rect(s, 0, 0, SW, Inches(1.05), NAVY); _rect(s, 0, Inches(1.05), SW, Pt(4), BLUE)
    tf = _box(s, Inches(0.5), Inches(0.18), Inches(12), Inches(0.7)); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    r = tf.paragraphs[0].add_run(); r.text = title; _set(r, 25, WHITE, b=True)
    if num:
        tn = _box(s, Inches(11.8), Inches(0.2), Inches(1.2), Inches(0.7)); tn.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tn.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
        rr = p.add_run(); rr.text = num; _set(rr, 13, BLUE, b=True)


def bullets(s, items, top=1.45, left=0.7, width=12.0, size=18, gap=8):
    tf = _box(s, Inches(left), Inches(top), Inches(width), Inches(5.6)); first = True
    for it in items:
        text, lvl = it[0], (it[1] if len(it) > 1 else 0); col = it[2] if len(it) > 2 else GREY
        p = tf.paragraphs[0] if first else tf.add_paragraph(); first = False; p.space_after = Pt(gap)
        pre = ["● ", "– ", "· "][min(lvl, 2)]; bc = [BLUE, TEAL, GREY][min(lvl, 2)]
        rb = p.add_run(); rb.text = pre; _set(rb, size - lvl * 2, bc, b=(lvl == 0))
        rt = p.add_run(); rt.text = text; _set(rt, size - lvl * 2, col, b=(lvl == 0))


def code_box(s, lines, top=1.5, left=0.8, width=11.7, size=12):
    h = 0.34 * len(lines) + 0.3
    bg = _rect(s, Inches(left), Inches(top), Inches(width), Inches(h), RGBColor(0x0D, 0x1B, 0x2A))
    tf = bg.text_frame; tf.word_wrap = True
    tf.margin_left = Pt(10); tf.margin_top = Pt(6)
    first = True
    for ln in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
        r = p.add_run(); r.text = ln
        r.font.name = "Consolas"; r.font.size = Pt(size)
        r.font.color.rgb = RGBColor(0x9C, 0xDC, 0xFE)
    return bg


def image_slide(title, num, img, caption, img_w=8.4):
    s = prs.slides.add_slide(BLANK); header(s, title, num)
    if os.path.exists(img):
        pic = s.shapes.add_picture(img, 0, Inches(1.45), width=Inches(img_w))
        pic.left = int((SW - pic.width) / 2)
        cap = _box(s, Inches(0.6), Inches(6.75), Inches(12.1), Inches(0.6))
        pc = cap.paragraphs[0]; pc.alignment = PP_ALIGN.CENTER
        rc = pc.add_run(); rc.text = caption; _set(rc, 13, GREY, i=True)
    else:
        bullets(s, [(f"（圖檔不存在：{img}）", 0, GREY)])
    return s


# 封面
s = prs.slides.add_slide(BLANK); _rect(s, 0, 0, SW, SH, NAVY); _rect(s, 0, Inches(4.2), SW, Pt(3), BLUE)
tf = _box(s, Inches(0.9), Inches(2.1), Inches(11.5), Inches(2))
r = tf.paragraphs[0].add_run(); r.text = "LLM Wiki"; _set(r, 48, WHITE, b=True)
p = tf.add_paragraph(); r = p.add_run(); r.text = "Karpathy 風格的 LLM 時代知識庫系統"; _set(r, 26, BLUE, b=True)
tf2 = _box(s, Inches(0.9), Inches(4.4), Inches(11.5), Inches(1.5))
for t, sz in [("一切以 markdown 儲存 · LLM 協助注入/連結/查詢/維護 · Obsidian 視覺化", 17),
              ("建構於本專案既有的 trading-wiki 知識庫之上", 15)]:
    pp = tf2.add_paragraph(); rr = pp.add_run(); rr.text = t; _set(rr, sz, RGBColor(0xC5, 0xD3, 0xE5))

# Karpathy 理念
s = prs.slides.add_slide(BLANK); header(s, "Karpathy 的想法：LLM 時代的知識庫", "理念")
bullets(s, [
    ("一切以純文字 / markdown 儲存", 0, NAVY),
    ("LLM 友善、可 grep、永不過時、不被任何 App 綁定", 1),
    ("LLM 是知識庫的「協作引擎」，貫穿四個動作：", 0, NAVY),
    ("注入(Ingest)：把任何來源轉成乾淨 MD 收進來", 1),
    ("連結(Link)：自動建立筆記間的關聯", 1),
    ("查詢(Query)：用自然語言問，LLM 從庫裡綜合答案", 1),
    ("維護(Lint)：保持知識庫健康、不長雜草", 1),
    ("Obsidian 負責視覺化與雙向連結（關聯圖）", 0, NAVY),
    ("核心：你的知識是「資產」，要能累積、連結、被 LLM 善用", 0, TEAL),
], size=18)

# 六大元件對應
s = prs.slides.add_slide(BLANK); header(s, "六大元件 × 對應本專案（既有為主，缺者補上）", "架構")
rows = [
    ("1 想法落地", "建在既有 trading-wiki KB（25 篇 MD）", "✅ 沿用"),
    ("2 markitdown", "各格式 → MD（PDF/Word/PPT/Excel/圖片）", "🆕 新增"),
    ("3 Data Ingest", "轉檔 + frontmatter + LLM摘要/自動連結", "🆕 新增"),
    ("4 Obsidian", "vault + 知識庫 + 關聯圖", "✅ 既有"),
    ("5 Query", "TF-IDF 檢索 + LLM 綜合（RAG）", "🆕 新增"),
    ("6 Linting", "壞連結/孤兒/缺frontmatter/重複", "🆕 擴充自規則健檢"),
]
y = 1.5
_rect(s, Inches(0.6), Inches(y), Inches(2.6), Inches(0.5), NAVY)
_rect(s, Inches(3.3), Inches(y), Inches(6.8), Inches(0.5), NAVY)
_rect(s, Inches(10.2), Inches(y), Inches(2.5), Inches(0.5), NAVY)
for txt, x, w in [("元件", 0.6, 2.6), ("做什麼", 3.3, 6.8), ("狀態", 10.2, 2.5)]:
    tf = _box(s, Inches(x), Inches(y + 0.04), Inches(w), Inches(0.45)); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    pp = tf.paragraphs[0]; pp.alignment = PP_ALIGN.CENTER; rr = pp.add_run(); rr.text = txt; _set(rr, 14, WHITE, b=True)
y += 0.5
for name, what, stat in rows:
    bg = LIGHT if (rows.index((name, what, stat)) % 2 == 0) else WHITE
    _rect(s, Inches(0.6), Inches(y), Inches(12.1), Inches(0.72), bg)
    for txt, x, w, c, sz in [(name, 0.7, 2.5, NAVY, 14), (what, 3.4, 6.7, GREY, 13),
                             (stat, 10.2, 2.4, GREEN if "✅" in stat else AMBER, 13)]:
        tf = _box(s, Inches(x), Inches(y + 0.02), Inches(w), Inches(0.68)); tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        rr = tf.paragraphs[0].add_run(); rr.text = txt; _set(rr, sz, c, b=("元件" in str(name)))
    y += 0.72

# 元件2+3 Ingest
s = prs.slides.add_slide(BLANK); header(s, "Data Ingest — 知識注入（含 markitdown）", "3")
bullets(s, [
    ("一行指令，把任何檔案/網址收進知識庫：", 0, NAVY),
], top=1.3, size=17)
code_box(s, ["python -m llm_wiki ingest report.pdf --tags 研究,被動元件"], top=1.85)
bullets(s, [
    ("自動化流程：", 0, NAVY),
    ("markitdown 轉成乾淨 markdown（PDF/Word/PPT/Excel/圖片OCR）", 1),
    ("LLM 生成 TL;DR 摘要 + 自動標籤", 1),
    ("LLM 從既有筆記中挑相關的，自動建立 [[雙向連結]]", 1),
    ("補 frontmatter（標題/來源/日期/標籤）後存進 ingested/", 1),
    ("實測：本專案 PPTX 簡報 → MD（3989 字）+ 自動連結到「關鍵實驗」筆記 ✅", 0, TEAL),
], top=2.6, size=16)

# 元件5 Query
s = prs.slides.add_slide(BLANK); header(s, "Query — 知識查詢（RAG）", "5")
bullets(s, [("自然語言問，系統從整個知識庫綜合出帶引用的答案：", 0, NAVY)], top=1.3, size=17)
code_box(s, ['python -m llm_wiki query "籌碼面實驗的結論是什麼？"'], top=1.85)
bullets(s, [
    ("流程：TF-IDF 檢索（char n-gram，支援中文）→ 取最相關 5 篇 → LLM 綜合", 0, NAVY),
    ("實測回答（節錄）：", 0, NAVY),
    ("「外資籌碼對中小型功率股是雜訊…準確率 56.2%→55.0% 沒提升反變差，", 1, GREY),
    ("  完全退回 56.8% [[6 關鍵實驗與誠實發現]]」", 1, GREY),
    ("答案直接引用來源筆記，可回溯、不憑空捏造", 0, TEAL),
], top=2.6, size=16)

# 元件6 Lint
s = prs.slides.add_slide(BLANK); header(s, "Linting — 知識庫維護", "6")
bullets(s, [
    ("延續本專案既有「規則庫健檢」，擴及整個 Obsidian 庫：", 0, NAVY),
    ("🔗 壞連結　[[X]] 指向不存在的筆記", 1),
    ("🏝️ 孤兒筆記　關聯圖上的孤點（沒進/出連結）", 1),
    ("📋 缺 frontmatter（只查 ingested 筆記）", 1),
    ("👯 重複標題　/　♊ 近重複內容（char n-gram 相似度）", 1),
], top=1.3, size=17)
code_box(s, ["python -m llm_wiki lint --save"], top=3.6)
bullets(s, [
    ("實測本庫：24 篇，僅 2 個孤兒筆記，0 壞連結 → 健康 ✅", 0, TEAL),
    ("報告自動寫成 lint_report.md，在 Obsidian 裡可點連結直接去修", 1),
], top=4.15, size=16)

# 元件4 Obsidian 視覺化（關聯圖）
import glob as _glob
_g = sorted(_glob.glob(r"D:\trading-wiki\charts\*_kb_graph.png"))
image_slide("Obsidian — 知識視覺化（關聯圖）", "4",
            _g[-1] if _g else "",
            "由知識庫真實 [[連結]] 生成的關聯圖（Obsidian 深色風格）；節點大小＝連結數")

# CLI + Obsidian
s = prs.slides.add_slide(BLANK); header(s, "統一 CLI + Obsidian 視覺化", "整合")
code_box(s, [
    "python -m llm_wiki ingest <檔案或網址> [--tags a,b]   # 注入",
    "python -m llm_wiki query   \"你的問題\"  [--k 5]          # 查詢",
    "python -m llm_wiki lint --save                          # 維護",
    "python -m llm_wiki status                               # 概況",
], top=1.5, size=13)
bullets(s, [
    ("Obsidian 視覺化：Open folder as vault → D:\\trading-wiki", 0, NAVY),
    ("注入的文件、自動連結、關聯圖、查詢來源，全在同一個庫裡互通", 1),
    ("知識庫現況：25 篇 MD · 1 篇已注入文件 · 84 張圖表", 0, TEAL),
], top=3.4, size=17)

# 結尾
s = prs.slides.add_slide(BLANK); _rect(s, 0, 0, SW, SH, NAVY)
tf = _box(s, Inches(0.9), Inches(2.6), Inches(11.5), Inches(2.2))
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER; r = p.add_run(); r.text = "知識是會累積的資產"; _set(r, 36, WHITE, b=True)
p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
r2 = p2.add_run(); r2.text = "markitdown 注入 · LLM 連結與查詢 · Obsidian 視覺化 · Lint 維護"; _set(r2, 17, BLUE)
p3 = tf.add_paragraph(); p3.alignment = PP_ALIGN.CENTER
r3 = p3.add_run(); r3.text = "全部建在你既有的知識庫之上，純 markdown、可長久累積"; _set(r3, 15, RGBColor(0x90, 0xA4, 0xBC))

try:
    prs.save(OUT); saved = OUT
except PermissionError:
    saved = OUT.replace(".pptx", "_new.pptx"); prs.save(saved); print("（原檔開啟中，另存新檔）")
print(f"簡報已生成：{saved}")
print(f"總頁數：{len(prs.slides._sldIdLst)}")
