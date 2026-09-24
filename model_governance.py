"""把未通過證據閘門的方向預測標成影子觀察。"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIN_LIVE_POINTS = 100
MIN_DIRECTION_ACCURACY = 0.52
MIN_MAE_IMPROVEMENT = 0.02


def status(root=HERE):
    path = Path(root) / "reports" / "prediction_audit.json"
    if not path.exists():
        return {"approved": False, "reason": "尚無實盤稽核報告", "n": 0}
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    point = report.get("verified", {}).get("point", {})
    direction = point.get("direction_correct", {})
    model_mae = point.get("abs_error_pp", {})
    baseline = point.get("baseline_abs_error_pp", {})
    n = min(direction.get("n", 0), model_mae.get("n", 0), baseline.get("n", 0))
    accuracy, mae, base = direction.get("mean"), model_mae.get("mean"), baseline.get("mean")
    approved = bool(n >= MIN_LIVE_POINTS and accuracy is not None and accuracy >= MIN_DIRECTION_ACCURACY
                    and mae is not None and base and mae <= base * (1 - MIN_MAE_IMPROVEMENT))
    reason = ("通過實盤證據閘門" if approved else
              f"可信樣本 {n}/{MIN_LIVE_POINTS}；方向 {accuracy:.1%}，模型MAE {mae:.3f}、猜不變 {base:.3f}"
              if n else f"可信樣本 {n}/{MIN_LIVE_POINTS}")
    return {"approved": approved, "reason": reason, "n": n,
            "direction_accuracy": accuracy, "mae_pp": mae, "baseline_mae_pp": base}


def apply_publication_gate(predictions, root=HERE):
    evidence = status(root)
    for prediction in predictions:
        prediction["raw_direction"] = prediction.get("direction")
        prediction["evidence_status"] = "approved" if evidence["approved"] else "shadow_unverified"
        prediction["evidence_reason"] = evidence["reason"]
        # 僅加標示；不改正式方向或幅度。治理規則本身也須累積驗證後才可改發布行為。
    return evidence
