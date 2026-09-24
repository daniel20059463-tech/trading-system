"""
LLM Wiki — 依 Andrej Karpathy「LLM 時代知識庫」想法，建在本專案既有的
markdown 知識庫（D:\\trading-wiki + Obsidian）之上。

核心理念（Karpathy）：
  - 一切以純文字 / markdown 儲存（LLM 友善、可 grep、永不過時、不被廠商綁定）
  - LLM 協助「注入(ingest) → 連結(link) → 查詢(query) → 維護(lint)」
  - Obsidian 負責視覺化與雙向連結

六大元件：
  1. 想法落地     — 建在既有 trading-wiki KB 上（不重造）
  2. markitdown   — 各種格式 → MD（ingest 內）
  3. Data Ingest  — ingest.py
  4. Obsidian     — 既有 vault（視覺化）
  5. Query        — query.py（TF-IDF 檢索 + LLM 綜合）
  6. Linting      — lint.py（壞連結/孤兒/缺 frontmatter/重複）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

# 知識庫根目錄（沿用專案既有 wiki）
KB_ROOT = cfg.WIKI_DIR
INGEST_DIR = os.path.join(KB_ROOT, "ingested")
