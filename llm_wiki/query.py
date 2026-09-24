"""
Query（知識查詢）：對整個 markdown 知識庫做 TF-IDF 檢索，取最相關的筆記，
再用 LLM 綜合出帶引用的答案（RAG）。中文用 char n-gram 向量化，無需 embedding 服務。

用法：
  python -m llm_wiki.query "分位數迴歸怎麼校準？"
  python -m llm_wiki.query "籌碼面實驗結論" --k 5 --no-llm   # 只檢索不綜合
"""
import os
import re
import sys
import glob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_wiki import KB_ROOT


def _load_corpus() -> list[dict]:
    """載入 KB 所有 markdown（跳過圖表/暫存資料夾）。"""
    docs = []
    for p in glob.glob(os.path.join(KB_ROOT, "**", "*.md"), recursive=True):
        if os.sep + "intraday_state" + os.sep in p:
            continue
        try:
            with open(p, encoding="utf-8") as f:
                text = f.read()
        except Exception:
            continue
        if text.strip():
            docs.append({"path": p,
                         "name": os.path.splitext(os.path.basename(p))[0],
                         "text": text})
    return docs


def retrieve(question: str, k: int = 5) -> list[dict]:
    """回傳與問題最相關的前 k 篇筆記（含相似度與摘錄）。"""
    docs = _load_corpus()
    if not docs:
        return []
    corpus = [d["text"] for d in docs] + [question]
    # char n-gram：對中文有效，不需斷詞
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), min_df=1)
    mat = vec.fit_transform(corpus)
    sims = cosine_similarity(mat[-1], mat[:-1])[0]
    order = sims.argsort()[::-1][:k]
    out = []
    for i in order:
        if sims[i] <= 0:
            continue
        out.append({"name": docs[i]["name"], "path": docs[i]["path"],
                    "score": round(float(sims[i]), 3),
                    "excerpt": _excerpt(docs[i]["text"], question)})
    return out


def _excerpt(text: str, question: str, width: int = 400) -> str:
    """取與問題關鍵字最接近的一段文字當摘錄。"""
    body = re.sub(r"^---.*?---", "", text, flags=re.DOTALL).strip()
    # 找問題中任一 2-gram 第一次出現的位置
    grams = [question[i:i+2] for i in range(len(question) - 1)]
    pos = -1
    for g in grams:
        pos = body.find(g)
        if pos >= 0:
            break
    if pos < 0:
        return body[:width]
    start = max(0, pos - width // 3)
    return body[start:start + width]


def answer(question: str, k: int = 5, use_llm: bool = True) -> dict:
    """檢索 + LLM 綜合，回傳 {answer, sources}。"""
    hits = retrieve(question, k)
    if not hits:
        return {"answer": "知識庫中找不到相關內容。", "sources": []}
    if not use_llm:
        return {"answer": None, "sources": hits}

    try:
        from dotenv import load_dotenv
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            return {"answer": "（未設定 OPENAI_API_KEY，僅回傳檢索結果）", "sources": hits}
        from openai import OpenAI
        client = OpenAI()
        context = "\n\n".join(
            f"【{h['name']}】\n{h['excerpt']}" for h in hits
        )
        prompt = (
            "你是知識庫問答助手。只根據下列知識庫摘錄回答問題，"
            "不要編造；若摘錄不足以回答就明說。回答用繁體中文，"
            "並在用到的地方標註來源筆記名 [[筆記名]]。\n\n"
            f"知識庫摘錄：\n{context}\n\n問題：{question}"
        )
        r = client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
        )
        return {"answer": r.choices[0].message.content.strip(), "sources": hits}
    except Exception as e:
        return {"answer": f"（LLM 綜合失敗：{e}，僅回傳檢索結果）", "sources": hits}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Query（KB 檢索 + LLM 綜合）")
    parser.add_argument("question", help="你的問題")
    parser.add_argument("--k", type=int, default=5, help="檢索筆數")
    parser.add_argument("--no-llm", action="store_true", help="只檢索不綜合")
    args = parser.parse_args()

    res = answer(args.question, k=args.k, use_llm=not args.no_llm)
    if res["answer"]:
        print("\n" + "=" * 60)
        print("回答：")
        print("=" * 60)
        print(res["answer"])
    print("\n" + "-" * 60)
    print("來源筆記（相似度）：")
    for h in res["sources"]:
        print(f"  [{h['score']}] {h['name']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
