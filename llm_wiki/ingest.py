"""
Data Ingest（知識注入）：把任意格式的檔案/網址，用 markitdown 轉成 markdown，
補上 frontmatter（標題/來源/日期/標籤），可選 LLM 生成摘要與建議連結，存進 KB。

支援格式（markitdown）：PDF / Word / PowerPoint / Excel / HTML / 圖片(OCR) / CSV / 純文字…

用法：
  python -m llm_wiki.ingest <檔案或網址> [--tags a,b] [--no-llm]
  python -m llm_wiki.ingest report.pdf --tags 研究,被動元件
"""
import os
import re
import sys
import glob
import datetime

from markitdown import MarkItDown

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_wiki import KB_ROOT, INGEST_DIR


def _slugify(name: str) -> str:
    base = os.path.splitext(os.path.basename(name))[0]
    base = re.sub(r"[\\/:*?\"<>|]", "_", base).strip()
    return base or "untitled"


def _convert(source: str) -> tuple[str, str]:
    """用 markitdown 轉換來源，回傳 (markdown 文字, 推測標題)。"""
    md = MarkItDown()
    result = md.convert(source)
    text = getattr(result, "text_content", None) or getattr(result, "markdown", "") or ""
    title = getattr(result, "title", None) or _slugify(source)
    return text, title


def _existing_titles() -> list[str]:
    """KB 內所有 note 名（供 LLM 建議連結）。"""
    names = []
    for p in glob.glob(os.path.join(KB_ROOT, "**", "*.md"), recursive=True):
        names.append(os.path.splitext(os.path.basename(p))[0])
    return names


def _llm_enrich(text: str, title: str) -> dict:
    """可選：用 LLM 生成摘要、標籤、建議連結。失敗或無金鑰時回傳空。"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            return {}
        from openai import OpenAI
        client = OpenAI()
        existing = _existing_titles()[:60]
        prompt = (
            f"以下是一份要存進知識庫的文件（標題：{title}）。\n"
            f"知識庫現有筆記名稱（可從中挑選相關的做雙向連結）：\n{existing}\n\n"
            f"文件內容（前 3000 字）：\n{text[:3000]}\n\n"
            "請輸出嚴格 JSON（不加 markdown 包裝）：\n"
            '{"summary":"3 句中文摘要","tags":["3-5個標籤"],'
            '"links":["從上面現有筆記名挑最多3個相關的，無則空陣列"]}'
        )
        r = client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = r.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1])
        import json
        return json.loads(raw)
    except Exception as e:
        print(f"  [LLM 增強略過：{e}]")
        return {}


def ingest(source: str, tags: list[str] | None = None, use_llm: bool = True) -> str:
    """注入單一檔案/網址，回傳寫入的 md 路徑。"""
    os.makedirs(INGEST_DIR, exist_ok=True)
    print(f"[Ingest] {source}")
    text, title = _convert(source)
    if not text.strip():
        raise ValueError("markitdown 轉換結果為空")
    print(f"  轉換完成：{len(text)} 字")

    enrich = _llm_enrich(text, title) if use_llm else {}
    all_tags = list(dict.fromkeys((tags or []) + enrich.get("tags", [])))
    summary = enrich.get("summary", "")
    links = enrich.get("links", [])

    # frontmatter
    today = datetime.date.today().isoformat()
    fm = ["---", f"title: {title}", f"source: {source}",
          f"ingested: {today}", "type: ingested"]
    if all_tags:
        fm.append("tags: [" + ", ".join(all_tags) + "]")
    fm.append("---\n")

    body = [f"# {title}\n"]
    if summary:
        body.append(f"> **TL;DR**：{summary}\n")
    if links:
        body.append("**相關**：" + " ".join(f"[[{l}]]" for l in links) + "\n")
    body.append("---\n")
    body.append(text)

    slug = _slugify(title if title and not title.startswith("http") else source)
    out_path = os.path.join(INGEST_DIR, f"{slug}.md")
    # 同名加序號避免覆蓋
    i = 1
    while os.path.exists(out_path):
        out_path = os.path.join(INGEST_DIR, f"{slug}_{i}.md")
        i += 1

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(fm) + "\n".join(body))
    print(f"  已注入：{out_path}")
    if all_tags:
        print(f"  標籤：{all_tags}")
    if links:
        print(f"  自動連結：{links}")
    return out_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Data Ingest（markitdown → KB）")
    parser.add_argument("source", help="檔案路徑或網址")
    parser.add_argument("--tags", default="", help="逗號分隔標籤")
    parser.add_argument("--no-llm", action="store_true", help="不使用 LLM 增強")
    args = parser.parse_args()
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    ingest(args.source, tags=tags, use_llm=not args.no_llm)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
