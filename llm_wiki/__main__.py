"""
LLM Wiki 統一 CLI。

  python -m llm_wiki ingest <檔案或網址> [--tags a,b] [--no-llm]
  python -m llm_wiki query "你的問題" [--k 5] [--no-llm]
  python -m llm_wiki lint [--save]
  python -m llm_wiki status                # KB 概況
"""
import os
import sys
import glob
import argparse

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_wiki import KB_ROOT, INGEST_DIR


def cmd_status():
    mds = glob.glob(os.path.join(KB_ROOT, "**", "*.md"), recursive=True)
    ingested = glob.glob(os.path.join(INGEST_DIR, "*.md"))
    charts = glob.glob(os.path.join(KB_ROOT, "charts", "*.png"))
    print(f"知識庫根目錄：{KB_ROOT}")
    print(f"  markdown 筆記：{len(mds)}")
    print(f"  已注入文件（ingested/）：{len(ingested)}")
    print(f"  圖表（charts/）：{len(charts)}")
    print(f"\nObsidian 視覺化：Open folder as vault → {KB_ROOT}")


def main():
    p = argparse.ArgumentParser(prog="llm_wiki", description="LLM Wiki（Karpathy 風格知識庫）")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="注入檔案/網址（markitdown→MD）")
    pi.add_argument("source"); pi.add_argument("--tags", default="")
    pi.add_argument("--no-llm", action="store_true")

    pq = sub.add_parser("query", help="知識查詢（檢索+LLM綜合）")
    pq.add_argument("question"); pq.add_argument("--k", type=int, default=5)
    pq.add_argument("--no-llm", action="store_true")

    pl = sub.add_parser("lint", help="KB 維護掃描")
    pl.add_argument("--save", action="store_true")

    sub.add_parser("status", help="KB 概況")

    args = p.parse_args()
    if args.cmd == "ingest":
        from llm_wiki.ingest import ingest
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        ingest(args.source, tags=tags, use_llm=not args.no_llm)
    elif args.cmd == "query":
        from llm_wiki.query import answer
        res = answer(args.question, k=args.k, use_llm=not args.no_llm)
        if res["answer"]:
            print("\n" + "=" * 60 + "\n回答：\n" + "=" * 60)
            print(res["answer"])
        print("\n" + "-" * 60 + "\n來源筆記（相似度）：")
        for h in res["sources"]:
            print(f"  [{h['score']}] {h['name']}")
    elif args.cmd == "lint":
        from llm_wiki.lint import lint, format_report
        report = format_report(lint())
        print(report)
        if args.save:
            path = os.path.join(KB_ROOT, "lint_report.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\n已寫入：{path}")
    elif args.cmd == "status":
        cmd_status()


if __name__ == "__main__":
    main()
