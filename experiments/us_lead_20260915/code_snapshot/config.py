# config.py - 台灣股票預測系統全域設定

import os

# ── 股票清單 ──────────────────────────────────────────────────────────────────

PASSIVE_STOCKS = {
    "2327.TW":  "國巨",
    "2492.TW":  "華新科",
    "3026.TW":  "禾伸堂",
    "2472.TW":  "立隆電",
    "3357.TWO": "臺慶科",
}

POWER_STOCKS = {
    "5425.TWO": "台半",
    "2481.TW":  "強茂",
    "3675.TWO": "德微",
    "8255.TWO": "朋程",
    "8261.TW":  "富鼎",
}

ALL_STOCKS = {**PASSIVE_STOCKS, **POWER_STOCKS}

CATEGORIES = {
    "passive": PASSIVE_STOCKS,
    "power": POWER_STOCKS,
}

# ── 同業與宏觀指標（作為 ML 特徵 + 異常偵測）───────────────────────────────────
# 共用宏觀指標（所有股票都加）
MACRO_INDICES = {
    "USDTWD=X": "usdtwd",   # 美元/台幣匯率（出口報價影響）
    "^TWII":    "twii",     # 台股加權指數（系統性市場情緒）
}

# 被動元件同業（只加進被動元件股票特徵）
PASSIVE_PEERS = {
    "6981.T": "murata",    # 村田製作所（MLCC 龍頭，直接競爭）
    "6762.T": "tdk",       # TDK（被動元件同業）
}

# 功率元件同業（只加進功率元件股票特徵）
POWER_PEERS = {
    "ON":  "onsemi",       # ON Semiconductor（功率半導體龍頭）
    "VSH": "vishay",       # Vishay（電阻/二極體同業）
}

# 完整對照（代碼 -> 中文名，供異常偵測與新聞顯示）
PEER_NAMES = {
    "6981.T": "村田製作所", "6762.T": "TDK",
    "ON": "ON Semiconductor", "VSH": "Vishay",
    "USDTWD=X": "美元台幣", "^TWII": "台股加權",
}

# 同業新聞關鍵字（納入 Google News 搜尋）
PASSIVE_PEER_KEYWORDS = ["Murata MLCC", "被動元件 庫存", "TDK 電容"]
POWER_PEER_KEYWORDS    = ["ON Semiconductor MOSFET", "功率半導體 缺貨", "車用功率元件"]

# 同業價格異常偵測閾值（單日漲跌幅絕對值，%）
PEER_ANOMALY_THRESHOLD = 3.0

# ── 國際盤前快照（昨夜美股收盤，台股開盤前最強領先指標）─────────────────────────
GLOBAL_INDICES = {
    "^IXIC": "那斯達克",
    "^SOX":  "費城半導體",
    "^GSPC": "標普500",
}
GLOBAL_STOCKS = {
    "NVDA":  "NVIDIA",
    "TSM":   "台積電ADR",
    "HNHPF": "鴻海ADR",
    "AAPL":  "Apple",
}
# 國際快照異動閾值：絕對值超過此值列入警示（%）
GLOBAL_ALERT_THRESHOLD = 1.5

# ── 同業財報日曆 ───────────────────────────────────────────────────────────────
# 美股同業財報通常落在每季首月下旬（1/4/7/10 月）；日股落在每季中旬
# 格式：代碼 -> 典型財報月份清單（系統會在該月 20-31 日內提醒）
PEER_EARNINGS_MONTHS = {
    "ON":     [2, 5, 8, 11],   # ON Semi 通常季後首月發布
    "VSH":    [2, 5, 8, 11],
    "6981.T": [2, 5, 8, 11],   # 村田（日本會計年度，約季中）
    "6762.T": [2, 5, 8, 11],
}

# ── 資料日期範圍 ───────────────────────────────────────────────────────────────

from datetime import date, timedelta

# 抓最近 3 年資料，END_DATE 動態取今天（yfinance end 為排他，+1 天含今日）
START_DATE = (date.today() - timedelta(days=365 * 3)).isoformat()
END_DATE   = (date.today() + timedelta(days=1)).isoformat()

# ── 模型訓練參數 ──────────────────────────────────────────────────────────────

WINDOW_SIZE    = 30       # 時序滑動視窗大小（天）
TRAIN_RATIO    = 0.8      # 訓練集比例
BATCH_SIZE     = 32       # 每批次樣本數
EPOCHS         = 200      # 最大訓練回合數
LEARNING_RATE  = 0.001    # 初始學習率
HIDDEN_SIZE    = 128      # LSTM 隱藏層維度
NUM_LAYERS     = 2        # LSTM 層數
DROPOUT        = 0.2      # Dropout 比例
LR_PATIENCE    = 20       # ReduceLROnPlateau 等待 epoch 數

# ── 路徑設定 ──────────────────────────────────────────────────────────────────

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR  = os.path.join(_BASE_DIR, "data")
MODEL_DIR = os.path.join(_BASE_DIR, "models")
NEWS_DIR  = os.path.join(_BASE_DIR, "news")
WIKI_DIR  = r"D:\trading-wiki"

# 原始資料與模型檢查點子目錄
RAW_DATA_DIR   = os.path.join(DATA_DIR, "raw")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

# 自動建立必要目錄
for _d in [RAW_DATA_DIR, CHECKPOINT_DIR, NEWS_DIR]:
    os.makedirs(_d, exist_ok=True)
