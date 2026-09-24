"""今日信心度：綜合 regime 漂移 + 區間寬度，一句話告訴你「今天該不該信點預測」。

風險管理用：模型隔日方向天花板 ~56%，真正價值在「知道何時別信自己」。
- regime 異常 → 信心低（點預測不可信，看區間+降倉）
- 區間很寬 → 信心中（不確定性高）
- 正常+區間收斂 → 信心高（點位可參考）
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


def assess(q_results=None) -> dict:
    """回傳 {level, icon, regime, avg_width, note}。q_results 可傳入已算好的區間避免重算。"""
    widths = []
    if q_results:
        for r in q_results:
            try:
                widths.append((r["q90_price"] - r["q10_price"]) / r["last_close"] * 100)
            except Exception:
                pass
    else:
        from quantile_forecast import predict_quantile
        for tk in cfg.ALL_STOCKS:
            try:
                r = predict_quantile(tk)
                widths.append((r["q90_price"] - r["q10_price"]) / r["last_close"] * 100)
            except Exception:
                pass
    avg_w = sum(widths) / len(widths) if widths else 99.0

    overall = "normal"
    try:
        from drift_monitor import check_regime
        overall = check_regime().get("overall", "normal")
    except Exception:
        pass

    if overall == "abnormal":
        return {"level": "低", "icon": "🔴", "regime": overall, "avg_width": avg_w,
                "note": "市場異常，點預測別信，只看區間、降倉"}
    if overall == "watch" or avg_w >= 14:
        return {"level": "中", "icon": "🟡", "regime": overall, "avg_width": avg_w,
                "note": "部分不確定，以區間為主、點位僅參考"}
    return {"level": "高", "icon": "🟢", "regime": overall, "avg_width": avg_w,
            "note": "狀態正常、區間收斂，點位可參考"}


def line(a: dict) -> str:
    return (f"{a['icon']} **今日信心度：{a['level']}** — {a['note']}"
            f"（80%區間平均 {a['avg_width']:.0f}%）")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(line(assess()))
