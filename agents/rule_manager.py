"""
CLIN 因果律暫存器：管理全域因果律與情境化例外清單。

對應 .md 框架的：
- Global Invariant Rules：跨時間段穩健成立的市場規律
- Episodic Exception Ledger：有明確外部事件佐證的例外，上限 3 條
"""

import sys
import os
import json
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WIKI_DIR

logger = logging.getLogger(__name__)

RULES_PATH      = os.path.join(WIKI_DIR, "rules.json")
MAX_EXCEPTIONS  = 3
MIN_CONFIDENCE  = 0.3   # 低於此值自動標記為 deprecated

EMPTY_RULES = {
    "version": "1.0",
    "last_updated": "",
    "global_rules": [],
    "exception_ledger": [],
}


# ── 讀寫 ──────────────────────────────────────────────────────────────────────

def load_rules() -> dict:
    """從 rules.json 載入規則庫，不存在時回傳空結構。"""
    if not os.path.exists(RULES_PATH):
        return EMPTY_RULES.copy()
    try:
        with open(RULES_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"rules.json 讀取失敗：{e}，使用空規則庫")
        return EMPTY_RULES.copy()


def save_rules(rules: dict) -> None:
    """將規則庫存回 rules.json。"""
    os.makedirs(WIKI_DIR, exist_ok=True)
    rules["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    with open(RULES_PATH, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)
    logger.info(f"規則庫已儲存：{RULES_PATH}")


# ── 規則操作 ──────────────────────────────────────────────────────────────────

def add_global_rule(rules: dict, description: str, sector: str = "both") -> dict:
    """新增一條全域因果律（初始信心度 0.5）。"""
    existing = [r["description"] for r in rules["global_rules"]]
    if description in existing:
        logger.info(f"規則已存在，跳過：{description[:40]}")
        return rules

    rule_id = f"R{len(rules['global_rules']) + 1:03d}"
    rules["global_rules"].append({
        "id":            rule_id,
        "description":   description,
        "sector":        sector,
        "confirmations": 1,
        "violations":    0,
        "confidence":    0.5,
        "created":       datetime.now().strftime("%Y-%m-%d"),
        "last_updated":  datetime.now().strftime("%Y-%m-%d"),
        "status":        "active",
    })
    logger.info(f"新增規則 {rule_id}：{description[:50]}")
    return rules


def confirm_rule(rules: dict, rule_id: str) -> dict:
    """規則被確認（預測方向正確），提升信心度。"""
    for r in rules["global_rules"]:
        if r["id"] == rule_id and r["status"] == "active":
            r["confirmations"] += 1
            total = r["confirmations"] + r["violations"]
            r["confidence"] = round(r["confirmations"] / total, 3)
            r["last_updated"] = datetime.now().strftime("%Y-%m-%d")
            logger.info(f"規則 {rule_id} 確認，信心度：{r['confidence']}")
    return rules


def violate_rule(rules: dict, rule_id: str) -> dict:
    """規則被違反（預測方向錯誤），降低信心度。"""
    for r in rules["global_rules"]:
        if r["id"] == rule_id and r["status"] == "active":
            r["violations"] += 1
            total = r["confirmations"] + r["violations"]
            r["confidence"] = round(r["confirmations"] / total, 3)
            r["last_updated"] = datetime.now().strftime("%Y-%m-%d")
            if r["confidence"] < MIN_CONFIDENCE:
                r["status"] = "deprecated"
                logger.warning(f"規則 {rule_id} 信心度過低（{r['confidence']}），已標記廢棄")
            else:
                logger.info(f"規則 {rule_id} 違反，信心度：{r['confidence']}")
    return rules


def add_exception(rules: dict, description: str, event_type: str, ticker: str = "") -> dict:
    """新增情境化例外（超過上限 3 條時拒絕新增）。"""
    active_exceptions = [e for e in rules["exception_ledger"]]
    if len(active_exceptions) >= MAX_EXCEPTIONS:
        logger.warning(f"例外清單已達上限 {MAX_EXCEPTIONS} 條，拒絕新增：{description[:40]}")
        return rules

    exc_id = f"E{len(rules['exception_ledger']) + 1:03d}"
    rules["exception_ledger"].append({
        "id":          exc_id,
        "description": description,
        "event_type":  event_type,
        "ticker":      ticker,
        "date":        datetime.now().strftime("%Y-%m-%d"),
    })
    logger.info(f"新增例外 {exc_id}：{description[:50]}")
    return rules


def deprecate_rule(rules: dict, rule_id: str) -> dict:
    """手動廢棄一條規則。"""
    for r in rules["global_rules"]:
        if r["id"] == rule_id:
            r["status"] = "deprecated"
            r["last_updated"] = datetime.now().strftime("%Y-%m-%d")
            logger.info(f"規則 {rule_id} 已廢棄")
    return rules


# ── 規則庫健檢（純 Python，零 token，防 bug 復發）─────────────────────────────

RULE_COUNT_WARN  = 12     # 活躍規則數超過此值即警示（可能在為雜訊立規）
DUP_JACCARD_WARN = 0.6    # 兩規則詞彙 Jaccard 相似度超過此值視為語義重複
GENERIC_PHRASES  = [
    "強化", "穩健", "調整全域", "增強預測", "提高模型", "因應", "隨機波動",
    # 「市場波動會影響預測準確性、需關注」類空泛 meta 規則（無可操作方向）
    "影響預測", "密切關注", "可能影響", "明顯波動", "顯著波動", "需注意", "需關注",
]


def _tokenize(text: str) -> set:
    """以 2-gram 字元集合表示一條規則描述，供相似度比較（中文不需斷詞）。"""
    t = "".join(ch for ch in text if ch.strip())
    return {t[i:i+2] for i in range(len(t) - 1)} or {t}


def _jaccard(a: set, b: set) -> float:
    """兩集合 Jaccard 相似度 = |交集| / |聯集|。"""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def health_check(rules: dict) -> dict:
    """檢查規則庫健康度，回傳 {warnings:[...], active_count, duplicates, generic}。

    偵測三種退化徵兆（對應之前 R004~R013 垃圾規則 bug）：
      1. 活躍規則數暴增（疑似為雜訊立規）
      2. 語義重複規則（描述高度相似）
      3. 空泛規則（含「強化穩健性」等無可觀測市場條件的字眼）
    """
    active = [r for r in rules["global_rules"] if r.get("status") == "active"]
    warnings = []

    # 1. 規則數暴增
    if len(active) > RULE_COUNT_WARN:
        warnings.append(f"活躍規則數 {len(active)} 條 > {RULE_COUNT_WARN}，疑似為雜訊立規")

    # 2. 語義重複（兩兩比對）
    duplicates = []
    toks = [(r["id"], _tokenize(r["description"])) for r in active]
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            sim = _jaccard(toks[i][1], toks[j][1])
            if sim >= DUP_JACCARD_WARN:
                duplicates.append((toks[i][0], toks[j][0], round(sim, 2)))
    if duplicates:
        shown = ", ".join(f"{a}≈{b}({s})" for a, b, s in duplicates[:5])
        more = f" 等共 {len(duplicates)} 組" if len(duplicates) > 5 else ""
        warnings.append(f"語義重複規則：{shown}{more}")

    # 3. 空泛規則
    generic = [r["id"] for r in active
               if sum(p in r["description"] for p in GENERIC_PHRASES) >= 2]
    if generic:
        warnings.append(f"空泛規則（含『強化穩健性』類字眼）：{', '.join(generic)}")

    return {
        "active_count": len(active),
        "warnings":     warnings,
        "duplicates":   duplicates,
        "generic":      generic,
        "healthy":      not warnings,
    }


# ── Prompt 格式化 ─────────────────────────────────────────────────────────────

def format_rules_for_prompt(rules: dict) -> str:
    """將規則庫格式化為可注入 AI prompt 的文字。"""
    active_rules = [r for r in rules["global_rules"] if r["status"] == "active"]
    exceptions   = rules["exception_ledger"]

    lines = ["【CLIN 因果律暫存器】\n"]

    lines.append("■ 全域因果律（Global Invariant Rules）")
    if active_rules:
        for r in sorted(active_rules, key=lambda x: -x["confidence"]):
            conf_bar = "#" * int(r["confidence"] * 5) + "-" * (5 - int(r["confidence"] * 5))
            lines.append(
                f"  [{r['id']}] {r['description']}\n"
                f"       信心度：{r['confidence']:.0%} {conf_bar}  "
                f"確認 {r['confirmations']} 次 / 違反 {r['violations']} 次"
            )
    else:
        lines.append("  （尚無規則，請從今日分析中提煉）")

    lines.append(f"\n■ 情境化例外清單（上限 {MAX_EXCEPTIONS} 條，目前 {len(exceptions)} 條）")
    if exceptions:
        for e in exceptions:
            lines.append(f"  [{e['id']}] [{e['date']}] {e['description']}")
    else:
        lines.append("  （無例外記錄）")

    return "\n".join(lines)


# ── 從 Reflexion 輸出解析規則更新 ────────────────────────────────────────────

def apply_reflexion_updates(rules: dict, updates: list[dict]) -> dict:
    """將 Reflexion AI 輸出的規則更新指令套用到規則庫。

    updates 格式（AI 輸出的 JSON 陣列）：
    [
      {"action": "add",     "description": "...", "sector": "passive"},
      {"action": "confirm", "rule_id": "R001"},
      {"action": "violate", "rule_id": "R002"},
      {"action": "deprecate","rule_id": "R003"},
      {"action": "add_exception", "description": "...", "event_type": "macro", "ticker": "2327"},
    ]
    """
    for u in updates:
        action = u.get("action", "")
        if action == "add":
            rules = add_global_rule(rules, u.get("description", ""), u.get("sector", "both"))
        elif action == "confirm":
            rules = confirm_rule(rules, u.get("rule_id", ""))
        elif action == "violate":
            rules = violate_rule(rules, u.get("rule_id", ""))
        elif action == "deprecate":
            rules = deprecate_rule(rules, u.get("rule_id", ""))
        elif action == "add_exception":
            rules = add_exception(
                rules,
                u.get("description", ""),
                u.get("event_type", "other"),
                u.get("ticker", ""),
            )
        else:
            logger.warning(f"未知規則操作：{action}")
    return rules


if __name__ == "__main__":
    rules = load_rules()
    rules = add_global_rule(rules, "美國科技大廠財報超預期 → 被動元件 MLCC 需求回升，國巨/華新科看漲", "passive")
    rules = add_global_rule(rules, "台幣升值超過 1% → 被動元件出口報價壓力，短線偏空", "passive")
    rules = add_global_rule(rules, "大盤 RSI>70 且成交量萎縮 → 功率元件短期回調機率高", "power")
    save_rules(rules)
    print(format_rules_for_prompt(rules))
