"""技術版簡報（圖表導向）：預測窗口/特徵/CLIN防呆/評估真實性/混淆矩陣。"""
import os, sys, glob
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

sys.stdout.reconfigure(encoding="utf-8")
OUT = r"D:\trading-wiki\半導體預測系統_技術簡報.pptx"
FONT = "Microsoft JhengHei"; CH = r"D:\trading-wiki\charts"
NAVY=RGBColor(0x1A,0x2B,0x4A); BLUE=RGBColor(0x21,0x96,0xF3); TEAL=RGBColor(0x00,0x96,0x88)
GREY=RGBColor(0x37,0x47,0x4F); WHITE=RGBColor(0xFF,0xFF,0xFF); GREEN=RGBColor(0x2E,0x7D,0x32)
RED=RGBColor(0xC6,0x28,0x28); AMBER=RGBColor(0xE6,0x7E,0x00); LIGHT=RGBColor(0xEC,0xF2,0xF9)

prs=Presentation(); prs.slide_width=Inches(13.333); prs.slide_height=Inches(7.5)
BLANK=prs.slide_layouts[6]; SW,SH=prs.slide_width,prs.slide_height

def _set(r,sz,c=GREY,b=False,i=False):
    r.font.name=FONT; r.font.size=Pt(sz); r.font.color.rgb=c; r.font.bold=b; r.font.italic=i
def _box(s,l,t,w,h):
    tb=s.shapes.add_textbox(l,t,w,h); tb.text_frame.word_wrap=True; return tb.text_frame
def _rect(s,l,t,w,h,c):
    sp=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,l,t,w,h); sp.fill.solid()
    sp.fill.fore_color.rgb=c; sp.line.fill.background(); return sp
def header(s,title,num=None):
    _rect(s,0,0,SW,Inches(1.05),NAVY); _rect(s,0,Inches(1.05),SW,Pt(4),BLUE)
    tf=_box(s,Inches(0.5),Inches(0.18),Inches(12),Inches(0.7)); tf.vertical_anchor=MSO_ANCHOR.MIDDLE
    r=tf.paragraphs[0].add_run(); r.text=title; _set(r,24,WHITE,b=True)
    if num:
        tn=_box(s,Inches(11.7),Inches(0.2),Inches(1.3),Inches(0.7)); tn.vertical_anchor=MSO_ANCHOR.MIDDLE
        p=tn.paragraphs[0]; p.alignment=PP_ALIGN.RIGHT; rr=p.add_run(); rr.text=num; _set(rr,13,BLUE,b=True)
def _chart(name):
    g=sorted(glob.glob(os.path.join(CH,f"*_{name}.png")))
    return g[-1] if g else ""
def chart_slide(title,num,blts,name,caption,h=4.05):
    s=prs.slides.add_slide(BLANK); header(s,title,num)
    # 上方重點
    tf=_box(s,Inches(0.6),Inches(1.2),Inches(12.1),Inches(1.0)); first=True
    for t,col in blts:
        p=tf.paragraphs[0] if first else tf.add_paragraph(); first=False; p.space_after=Pt(3)
        rb=p.add_run(); rb.text="● "; _set(rb,14,BLUE,b=True)
        rt=p.add_run(); rt.text=t; _set(rt,14,col,b=False)
    # 大圖置中
    img=_chart(name)
    if img and os.path.exists(img):
        pic=s.shapes.add_picture(img,0,Inches(2.25),height=Inches(h)); pic.left=int((SW-pic.width)/2)
    cap=_box(s,Inches(0.6),Inches(6.55),Inches(12.1),Inches(0.6))
    pc=cap.paragraphs[0]; pc.alignment=PP_ALIGN.CENTER; rc=pc.add_run(); rc.text=caption; _set(rc,12.5,GREY,i=True)
    return s
def bullets(s,items,top=1.4,size=18,gap=8):
    tf=_box(s,Inches(0.7),Inches(top),Inches(12.0),Inches(5.7)); first=True
    for it in items:
        text,lvl=it[0],(it[1] if len(it)>1 else 0); col=it[2] if len(it)>2 else GREY
        p=tf.paragraphs[0] if first else tf.add_paragraph(); first=False; p.space_after=Pt(gap)
        pre=["● ","– ","· "][min(lvl,2)]; bc=[BLUE,TEAL,GREY][min(lvl,2)]
        rb=p.add_run(); rb.text=pre; _set(rb,size-lvl*2,bc,b=(lvl==0))
        rt=p.add_run(); rt.text=text; _set(rt,size-lvl*2,col,b=(lvl==0))

# 封面
s=prs.slides.add_slide(BLANK); _rect(s,0,0,SW,SH,NAVY); _rect(s,0,Inches(4.2),SW,Pt(3),BLUE)
tf=_box(s,Inches(0.9),Inches(2.2),Inches(11.5),Inches(2))
r=tf.paragraphs[0].add_run(); r.text="半導體類股預測系統"; _set(r,40,WHITE,b=True)
p=tf.add_paragraph(); r=p.add_run(); r.text="技術細節與評估真實性"; _set(r,28,BLUE,b=True)
tf2=_box(s,Inches(0.9),Inches(4.45),Inches(11.5),Inches(1))
rr=tf2.paragraphs[0].add_run()
rr.text="預測窗口 · 特徵工程 · CLIN 防呆 · 評估真實性 · 混淆矩陣（全部真實數據圖表）"
_set(rr,17,RGBColor(0xC5,0xD3,0xE5))

# 大綱
s=prs.slides.add_slide(BLANK); header(s,"大綱")
bullets(s,[("一、預測窗口　Prediction Window",0),("二、特徵值內容　Feature Engineering（19 維）",0),
    ("三、CLIN 防呆機制　不退步回溯",0),("四、ML 評估真實性　Evaluation Rigor",0),
    ("五、混淆矩陣　Confusion Matrix",0),
    ("＊ 所有圖表均來自 held-out 測試集（模型未看過的資料）",0,TEAL)],top=1.7,size=22,gap=15)

# 1 預測窗口（圖）
chart_slide("一、預測窗口","1",
    [("30 日窗口 → 預測隔 1 日漲跌幅（非絕對價，避免趨勢漂移）",NAVY),
     ("時序 80% 訓練 / 20% 測試；曾測 5 日窗口反而更差，故採 1 日",NAVY)],
    "window","1 日方向 56.8% vs 5 日方向 44.7%（比丟硬幣還差）→ 1 日為最佳設定")

# 2 特徵（圖）
chart_slide("二、特徵值內容（19 維）","2",
    [("15 維技術面（均線/RSI/MACD/布林/ATR/量價/報酬率）＋ 4 維關聯面（匯率/大盤/同業）",NAVY),
     ("正規化只用訓練集 fit；下圖為各特徵被模型使用的重要度",NAVY)],
    "feat","布林通道、均線、大盤指數、MACD 最被倚重；籌碼面實驗加入後 56.2%→55% 已移除")

# 3 CLIN（圖）
chart_slide("三、CLIN 防呆機制（不退步回溯）","3",
    [("核心：分多輪訓練，某輪沒比前一輪好 → 退回上一版、重新訓練（絕不退步）",RED),
     ("監控訓練 vs 泛化 MAPE：泛化遠大於訓練=過擬合，立刻回溯簡化",NAVY)],
    "overfit","實作：eval_harness 重訓沒進步就還原舊模型；overfit_monitor 監控泛化退步")

# 4 ML 評估真實性（圖）
chart_slide("四、ML 評估真實性","4",
    [("測試集是「未來」資料、不洗牌、scaler 只用訓練集 fit（嚴防洩漏）",NAVY),
     ("一切對照 baseline；敢於否定——三個假設經驗證全退回",NAVY)],
    "refuted","籌碼面 55% / 5日窗口 44.7%，無一勝過 56.8% 基準 → 全部誠實退回")

# 5 混淆矩陣（圖）
chart_slide("五、混淆矩陣（測試集 1040 筆）","5",
    [("準確率 56.7%，但模型 90% 預測漲——看跌召回僅 13.5%",RED),
     ("非亂猜：預測值隨輸入變化且與實際正相關 +0.1~0.28，是學到「順勢」",NAVY)],
    "confusion","headline 準確率會騙人；混淆矩陣才看得到模型真實行為（高度看漲偏向）")

# 6 總結（圖）
chart_slide("總結：三種預測準度（測試集）","",
    [("同一模型，問「漲跌」只能 57%，問「落在哪範圍」有 78%——差在問題難度",TEAL)],
    "accuracy","方向 56.6% / 價格 MAPE 3.5% / 80%區間覆蓋 78% / 50%區間 48%（校準準確）")

# 結尾
s=prs.slides.add_slide(BLANK); _rect(s,0,0,SW,SH,NAVY)
tf=_box(s,Inches(0.9),Inches(2.8),Inches(11.5),Inches(2))
p=tf.paragraphs[0]; p.alignment=PP_ALIGN.CENTER; r=p.add_run()
r.text="評估的真實性，比準確率本身更重要"; _set(r,32,WHITE,b=True)
p2=tf.add_paragraph(); p2.alignment=PP_ALIGN.CENTER; r2=p2.add_run()
r2.text="held-out 驗證 · baseline 對照 · 混淆矩陣看穿偏差 · 敢於否定無效假設"; _set(r2,16,BLUE)

try:
    prs.save(OUT); saved=OUT
except PermissionError:
    saved=OUT.replace(".pptx","_new.pptx"); prs.save(saved); print("（原檔開啟中，另存新檔）")
print(f"簡報已生成：{saved}　總頁數：{len(prs.slides._sldIdLst)}")
