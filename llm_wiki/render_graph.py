"""
把知識庫真實的 [[連結]] 渲染成 Obsidian 風格的關聯圖（深色背景）。
這是真實資料的視覺化，可放進簡報；非 Obsidian app 截圖（GUI 無法程式截取）。

用法：python -m llm_wiki.render_graph
輸出：D:\\trading-wiki\\charts\\<date>_kb_graph.png
"""
import os
import re
import sys
import glob
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_wiki import KB_ROOT
from visualize import _setup_font, CHART_DIR

_setup_font()
LINK_RE = re.compile(r"(!?)\[\[([^\]]+)\]\]")
# 日誌型節點（每日記錄、月報）排除，讓圖聚焦在知識結構
LOG_PAT = re.compile(r"^\d{4}-\d{2}|health_|eval_harness_|backfill_|retrain_|"
                     r"lint_report|snapshot_history")


def _build_graph() -> nx.Graph:
    G = nx.Graph()
    names = {}
    for p in glob.glob(os.path.join(KB_ROOT, "**", "*.md"), recursive=True):
        names[os.path.splitext(os.path.basename(p))[0]] = p
    for name, p in names.items():
        text = open(p, encoding="utf-8").read()
        for is_embed, target in LINK_RE.findall(text):
            if is_embed:
                continue
            t = target.split("|")[0].split("#")[0].strip()
            if t in names and t != name:
                G.add_edge(name, t)
    return G


def render():
    G = _build_graph()
    # 排除日誌型節點（每日記錄等），聚焦知識結構
    G.remove_nodes_from([n for n in list(G.nodes()) if LOG_PAT.match(n)])
    # 只留有連結的節點（Obsidian 關聯圖的「連成一團」部分）
    G.remove_nodes_from([n for n, d in dict(G.degree()).items() if d == 0])
    if G.number_of_nodes() == 0:
        print("無連結可畫")
        return ""

    BG = "#1A1A1F"
    fig, ax = plt.subplots(figsize=(12, 8.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    try:
        pos = nx.kamada_kawai_layout(G)   # 小圖布局較均勻好看
    except Exception:
        pos = nx.spring_layout(G, k=2.5, iterations=300, seed=7)
    deg = dict(G.degree())
    sizes = [300 + deg[n] * 320 for n in G.nodes()]

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#4A5568", width=1.3, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes,
                           node_color="#7C9CE0", edgecolors="#A9C2F0", linewidths=1.5, alpha=0.95)
    # 標籤
    for n, (x, y) in pos.items():
        ax.text(x, y, n, fontsize=8.5, color="#E8EAF0", ha="center", va="center",
                fontweight="bold", zorder=5)

    ax.set_title("知識庫關聯圖（Obsidian 風格）— 由真實 [[連結]] 生成",
                 color="#E8EAF0", fontsize=14, fontweight="bold", pad=14)
    ax.axis("off")
    plt.tight_layout()

    os.makedirs(CHART_DIR, exist_ok=True)
    out = os.path.join(CHART_DIR, f"{date.today().isoformat()}_kb_graph.png")
    fig.savefig(out, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"節點 {G.number_of_nodes()} · 連結 {G.number_of_edges()}")
    print(f"已輸出：{out}")
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    render()
