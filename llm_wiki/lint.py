"""
Linting（KB 維護）：掃描整個 markdown 知識庫，找出健康問題並產出報告。
精神延續本專案既有的 rule_manager.health_check，但擴及整個 Obsidian 庫。

檢查項目：
  1. 壞連結    [[X]] 指向不存在的筆記
  2. 壞嵌入    ![[img.png]] 指向不存在的檔案
  3. 孤兒筆記  沒有任何進/出連結（在 Obsidian 關聯圖是孤點）
  4. 缺 frontmatter
  5. 過短筆記  內文太少（疑似空殼）
  6. 重複標題  不同資料夾出現同名筆記
  7. 近重複內容（char n-gram 相似度 > 門檻）

用法：
  python -m llm_wiki.lint              # 印報告
  python -m llm_wiki.lint --save        # 同時寫入 KB/lint_report.md
"""
import os
import re
import sys
import glob
from datetime import date
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_wiki import KB_ROOT, INGEST_DIR

LINK_RE = re.compile(r"(!?)\[\[([^\]]+)\]\]")
DUP_SIM = 0.85          # 近重複內容門檻
MIN_BODY = 50           # 過短筆記字數門檻
IMG_EXT = (".png", ".jpg", ".jpeg", ".gif", ".pdf", ".svg", ".webp")
# 日誌型檔案（每日記錄、月度健康/驗收）共用模板，近重複檢查時排除，避免雜訊
LOG_PAT = re.compile(r"^\d{4}-\d{2}|health_|eval_harness_|backfill_|retrain_|lint_report")


def _all_files() -> dict:
    """回傳 {basename(無副檔名): 路徑} 與 {完整檔名: 路徑}。"""
    note_names, file_names = {}, {}
    for p in glob.glob(os.path.join(KB_ROOT, "**", "*.*"), recursive=True):
        fn = os.path.basename(p)
        file_names[fn] = p
        if fn.endswith(".md"):
            note_names[os.path.splitext(fn)[0]] = p
    return note_names, file_names


def _parse_links(text: str):
    """回傳 (note_links, embed_files)。去除 |alias 與 #heading。"""
    notes, embeds = [], []
    for is_embed, target in LINK_RE.findall(text):
        t = target.split("|")[0].split("#")[0].strip()
        if not t:
            continue
        if is_embed or t.lower().endswith(IMG_EXT):
            embeds.append(t)
        else:
            notes.append(t)
    return notes, embeds


def lint() -> dict:
    """執行全庫掃描，回傳問題字典。"""
    note_names, file_names = _all_files()
    md_paths = [p for p in note_names.values()]

    broken_links, broken_embeds = [], []
    missing_fm, tiny = [], []
    outlinks = defaultdict(set)
    inlinked = set()
    contents = {}

    for name, path in note_names.items():
        with open(path, encoding="utf-8") as f:
            text = f.read()
        contents[name] = text
        # frontmatter：只對 ingested 筆記要求（手寫筆記沒有是正常的）
        if os.path.abspath(INGEST_DIR) in os.path.abspath(path) and not text.lstrip().startswith("---"):
            missing_fm.append(name)
        # body 長度
        body = re.sub(r"^---.*?---", "", text, flags=re.DOTALL).strip()
        if len(body) < MIN_BODY:
            tiny.append(name)
        # 連結
        notes, embeds = _parse_links(text)
        for n in notes:
            outlinks[name].add(n)
            if n in note_names:
                inlinked.add(n)
            else:
                broken_links.append((name, n))
        for e in embeds:
            base = os.path.basename(e)
            if base not in file_names and e not in file_names:
                broken_embeds.append((name, e))

    # 孤兒：無出連結 且 無被連結
    orphans = [n for n in note_names
               if not outlinks[n] and n not in inlinked]

    # 重複標題（同 basename 多檔）
    dup_titles = defaultdict(list)
    for p in glob.glob(os.path.join(KB_ROOT, "**", "*.md"), recursive=True):
        dup_titles[os.path.splitext(os.path.basename(p))[0]].append(p)
    duplicates = {k: v for k, v in dup_titles.items() if len(v) > 1}

    # 近重複內容（char n-gram）
    near_dups = []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        # 排除日誌型檔案（共用模板會誤判近重複）
        names = [n for n in contents if not LOG_PAT.match(n)]
        if len(names) >= 2:
            vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=1)
            mat = vec.fit_transform([contents[n] for n in names])
            sim = cosine_similarity(mat)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    if sim[i, j] >= DUP_SIM:
                        near_dups.append((names[i], names[j], round(float(sim[i, j]), 2)))
    except Exception:
        pass

    return {
        "total_notes": len(note_names),
        "broken_links": broken_links,
        "broken_embeds": broken_embeds,
        "orphans": orphans,
        "missing_frontmatter": missing_fm,
        "tiny": tiny,
        "duplicates": duplicates,
        "near_dups": near_dups,
    }


def format_report(r: dict) -> str:
    lines = [f"# KB Lint 報告　{date.today().isoformat()}\n",
             f"掃描筆記數：**{r['total_notes']}**\n"]

    def section(title, items, fmt):
        lines.append(f"\n## {title}（{len(items)}）")
        if not items:
            lines.append("- ✅ 無")
        else:
            for it in items[:30]:
                lines.append(f"- {fmt(it)}")
            if len(items) > 30:
                lines.append(f"- …等共 {len(items)} 項")

    section("🔗 壞連結", r["broken_links"], lambda x: f"`{x[0]}` → 找不到 [[{x[1]}]]")
    section("🖼️ 壞嵌入", r["broken_embeds"], lambda x: f"`{x[0]}` → 找不到檔案 {x[1]}")
    section("🏝️ 孤兒筆記（關聯圖孤點）", r["orphans"], lambda x: f"[[{x}]]")
    section("📋 缺 frontmatter", r["missing_frontmatter"], lambda x: f"[[{x}]]")
    section("📭 過短筆記", r["tiny"], lambda x: f"[[{x}]]")
    section("👯 重複標題", list(r["duplicates"].items()),
            lambda x: f"`{x[0]}` 出現 {len(x[1])} 次")
    section("♊ 近重複內容", r["near_dups"], lambda x: f"[[{x[0]}]] ≈ [[{x[1]}]]（{x[2]}）")

    n_issues = (len(r["broken_links"]) + len(r["broken_embeds"]) + len(r["orphans"])
                + len(r["missing_frontmatter"]) + len(r["tiny"])
                + len(r["duplicates"]) + len(r["near_dups"]))
    lines.insert(2, f"問題總數：**{n_issues}**\n")
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="KB Linting")
    parser.add_argument("--save", action="store_true", help="寫入 KB/lint_report.md")
    args = parser.parse_args()

    r = lint()
    report = format_report(r)
    print(report)
    if args.save:
        path = os.path.join(KB_ROOT, "lint_report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n已寫入：{path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
