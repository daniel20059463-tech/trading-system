import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

import adoption_state as adoption
from prediction_audit import TZ


def pair(day, ticker="A.TW", new=.5, old=1.0, zero=1.2, pid=None):
    return {"date": day, "ticker": ticker, "candidate_protocol": adoption.CANDIDATE_PROTOCOL,
            "candidate_artifact_sha256": "a" * 64, "candidate_model_path": "models/candidate/A.TW.pkl",
            "adoption_id": pid,
            "new_mae": new, "old_mae": old, "zero_mae": zero,
            "new_correct": True, "old_correct": False, "new_covered80": True,
            "new_width80": 2.0, "actual_pct": 1.0, "predicted_pct": .5}


def eligible(root, now):
    baseline = root / "models/checkpoints/A.TW.pt"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    if not baseline.exists():
        baseline.write_bytes(b"old")
    evidence = {"decision": {"approved": True}, "sample_days": ["2026-09-20"],
                "target_data_end": "2026-09-20", "pair_count": 200,
                "pair_sha256": "b" * 64, "prediction_sha256": {},
                "affected_tickers": ["A.TW"],
                "versions": {"production_model_sha256": {"A.TW.pt": adoption._file_sha(baseline)},
                             "candidate_manifest": "models/candidate/manifest.json",
                             "candidate_manifest_sha256": "d" * 64, "rule_sha256": {}}}
    return adoption.transition(root, "eligible_for_review", "門檻通過", evidence, now)


def promoted(root, now):
    pending = eligible(root, now)
    with patch.object(adoption, "review", side_effect=lambda *args: adoption.status(root)):
        return adoption.promote(root, pending["event_sha256"], now)


def test_event_chain_rejects_tampering_and_invalid_transition(tmp_path):
    stamp = datetime(2026, 9, 21, 17, tzinfo=TZ)
    event = eligible(tmp_path, stamp)
    assert adoption.status(tmp_path)["state"] == "eligible_for_review"
    with pytest.raises(ValueError, match="不允許"):
        adoption.transition(tmp_path, "retained", "跳級", {}, stamp)
    path = tmp_path / "live_evidence/adoption/events/000001.json"
    altered = json.loads(path.read_text(encoding="utf-8"))
    altered["reason"] = "竄改"
    path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="事件鏈損壞"):
        adoption.status(tmp_path)
    assert event["event_sha256"] != adoption._sha({k: v for k, v in altered.items() if k != "event_sha256"})


def test_promotion_requires_reviewed_hash_and_starts_next_day(tmp_path):
    stamp = datetime(2026, 9, 21, 17, tzinfo=TZ)
    pending = eligible(tmp_path, stamp)
    with pytest.raises(ValueError, match="完整 event_sha256"):
        adoption.promote(tmp_path, now=stamp)
    with pytest.raises(ValueError, match="已變更"):
        adoption.promote(tmp_path, "old-hash", stamp)
    with patch.object(adoption, "review", side_effect=lambda *args: adoption.status(tmp_path)):
        event = adoption.promote(tmp_path, pending["event_sha256"], stamp)
    assert event["to"] == "promoted"
    assert event["evidence"]["monitor_starts_after"] == "2026-09-21"
    assert event["evidence"]["promotion_id"]


def test_review_seals_scored_pairs_and_refreshes_when_evidence_changes(tmp_path):
    stamp = datetime(2026, 9, 21, 17, tzinfo=TZ)
    days = [f"2026-08-{i + 1:02d}" for i in range(20)]
    for day in days:
        path = tmp_path / "predictions" / f"{day}_point.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"target_date":"' + day + '"}', encoding="utf-8")
    rows = [pair(day, ticker=f"{j}.TW") for day in days for j in range(10)]
    decision = {"approved": True, "n": 200, "days": 20, "reason": "通過"}
    versions = {"production_model_sha256": {"old.pt": "a" * 64},
                "candidate_manifest": "m.json", "candidate_manifest_sha256": "b" * 64,
                "rule_sha256": {}}
    with patch("daily_retrain.live_pairs", return_value=rows), \
         patch("daily_retrain.promotion_decision", return_value=decision), \
         patch.object(adoption, "_versions", return_value=versions):
        # 此測試聚焦事件刷新；真實檔案鏈由下方的獨立測試覆蓋。
        with patch.object(adoption, "_review_evidence", side_effect=lambda root, pairs, decision:
                   {"decision": decision, "sample_days": days, "target_data_end": days[-1],
                    "pair_count": len(pairs), "pair_sha256": adoption._sha(pairs),
                    "pairs": list(pairs), "prediction_sha256": {},
                    "affected_tickers": sorted({p["ticker"] for p in pairs}), "versions": versions}):
            first = adoption.review(tmp_path, stamp)
            assert first["state"] == "eligible_for_review"
            assert first["last_event"]["evidence"]["pairs"] == rows
            assert adoption.review(tmp_path, stamp)["event_count"] == 1
            rows[0] = dict(rows[0], new_mae=.4)
            second = adoption.review(tmp_path, stamp)
    assert second["event_count"] == 2
    with pytest.raises(ValueError, match="已變更"):
        adoption.promote(tmp_path, first["last_event"]["event_sha256"], stamp)


def test_review_rejects_changed_candidate_or_formal_model(tmp_path):
    day = "2026-09-20"
    model = tmp_path / "models/candidate/A.TW.pkl"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"candidate")
    artifact_sha = adoption._file_sha(model)
    baseline_sha = "b" * 64
    prediction = tmp_path / "predictions/2026-09-20_point.json"
    prediction.parent.mkdir()
    prediction.write_text(json.dumps({"model_sha256": {"A.TW.pt": baseline_sha},
        "records": [{"ticker": "A.TW", "daily_retrain": {
            "model_path": "models/candidate/A.TW.pkl", "artifact_sha256": artifact_sha,
            "protocol": adoption.CANDIDATE_PROTOCOL, "predicted_change_pct": .5}}]}), encoding="utf-8")
    row = pair(day)
    row["candidate_artifact_sha256"] = artifact_sha
    versions = {"production_model_sha256": {"A.TW.pt": baseline_sha}}
    with patch.object(adoption, "_versions", return_value=versions), \
         patch.dict("config.ALL_STOCKS", {"A.TW": "A"}, clear=True):
        assert adoption._review_evidence(tmp_path, [row], {"n": 1})["pair_count"] == 1
        model.write_bytes(b"changed")
        with pytest.raises(ValueError, match="候選模型來源無法驗證"):
            adoption._review_evidence(tmp_path, [row], {"n": 1})


def test_monitor_uses_only_post_promotion_rows_and_rolls_back(tmp_path):
    stamp = datetime(2026, 9, 21, 17, tzinfo=TZ)
    event = promoted(tmp_path, stamp)
    pid = event["evidence"]["promotion_id"]
    rows = [pair(f"2026-10-{i + 1:02d}", ticker=f"{j}.TW", new=1.5, old=1.0,
                 zero=1.1, pid=pid) for i in range(10) for j in range(10)]
    with patch.object(adoption, "sealed_pairs", return_value=rows):
        result = adoption.monitor(tmp_path, stamp.replace(day=28))
    assert result["state"] == "rolled_back"
    metrics = result["last_event"]["evidence"]["monitoring"]["metrics"]
    assert metrics["n"] == 100 and metrics["days"] == 10
    assert adoption.status(tmp_path)["promotion"]["evidence"]["promotion_id"] == pid


def test_rollback_restores_exact_previous_model_bytes(tmp_path):
    stamp = datetime(2026, 9, 21, 17, tzinfo=TZ)
    promoted(tmp_path, stamp)
    baseline = tmp_path / "models/checkpoints/A.TW.pt"
    baseline.write_bytes(b"changed")
    event = adoption.rollback(tmp_path, "模型檔意外變動", stamp)
    assert event["to"] == "rolled_back"
    assert baseline.read_bytes() == b"old"
    assert event["evidence"]["restored_files"] == ["A.TW.pt"]


def test_restart_requires_new_candidate_protocol(tmp_path):
    stamp = datetime(2026, 9, 21, 17, tzinfo=TZ)
    promoted(tmp_path, stamp)
    adoption.rollback(tmp_path, "退化", stamp)
    with pytest.raises(ValueError, match="不同候選協定"):
        adoption.restart(tmp_path, stamp)
    with patch.object(adoption, "CANDIDATE_PROTOCOL", "daily_v3"), \
         patch.object(adoption, "_versions", return_value={"candidate_manifest": "new.json"}):
        event = adoption.restart(tmp_path, stamp)
        assert event["to"] == "shadow"
        assert adoption.status(tmp_path)["promotion"] is None


def test_monitor_retains_only_after_independent_window(tmp_path):
    stamp = datetime(2026, 9, 21, 17, tzinfo=TZ)
    pid = promoted(tmp_path, stamp)["evidence"]["promotion_id"]
    rows = [pair(f"2026-10-{i + 1:02d}", ticker=f"{j}.TW", pid=pid)
            for i in range(20) for j in range(10)]
    with patch.object(adoption, "sealed_pairs", return_value=rows[:10]):
        assert adoption.monitor(tmp_path, datetime(2026, 10, 1, 17, tzinfo=TZ))["state"] == "monitored"
    with patch.object(adoption, "sealed_pairs", return_value=rows):
        assert adoption.monitor(tmp_path, datetime(2026, 10, 20, 17, tzinfo=TZ))["state"] == "retained"


def test_direction_regression_rolls_back_even_if_mae_wins(tmp_path):
    stamp = datetime(2026, 9, 21, 17, tzinfo=TZ)
    pid = promoted(tmp_path, stamp)["evidence"]["promotion_id"]
    rows = [dict(pair(f"2026-10-{i + 1:02d}", ticker=f"{j}.TW", pid=pid),
                 new_correct=False, old_correct=True)
            for i in range(10) for j in range(10)]
    with patch.object(adoption, "sealed_pairs", return_value=rows):
        result = adoption.monitor(tmp_path, datetime(2026, 10, 10, 17, tzinfo=TZ))
    assert result["state"] == "rolled_back"


def test_post_market_settlement_is_immutable_and_monitor_checks_source(tmp_path):
    stamp = datetime(2026, 9, 21, 17, tzinfo=TZ)
    pid = promoted(tmp_path, stamp)["evidence"]["promotion_id"]
    day = "2026-09-22"
    prediction = tmp_path / "predictions/2026-09-22_point.json"
    prediction.parent.mkdir(parents=True)
    prediction.write_text('{"records":[{"ticker":"A.TW"}]}', encoding="utf-8")
    with patch("daily_retrain.live_pairs", return_value=[pair(day, pid=pid)]):
        sealed = adoption.seal_post_market(tmp_path, datetime(2026, 9, 22, 15, 30, tzinfo=TZ))
        assert adoption.seal_post_market(tmp_path, datetime(2026, 9, 22, 17, tzinfo=TZ)) == sealed
    assert len(adoption.sealed_pairs(tmp_path, adoption.status(tmp_path)["promotion"])) == 1
    prediction.write_text('{"records":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="盤前預測已變動"):
        adoption.sealed_pairs(tmp_path, adoption.status(tmp_path)["promotion"])


def test_route_fail_closed_on_missing_artifact_and_preserves_baseline(tmp_path):
    stamp = datetime(2026, 9, 21, 17, tzinfo=TZ)
    promoted(tmp_path, stamp)
    prediction = {"ticker": "A.TW", "name": "A", "data_as_of": "2026-09-21",
                  "last_close": 100.0, "predicted_change_pct": -1.0, "predicted_close": 99.0,
                  "direction": "DOWN"}
    with patch.dict("config.ALL_STOCKS", {"A.TW": "A"}, clear=True):
        result = adoption.route([prediction], tmp_path, datetime(2026, 9, 22, 8, 30, tzinfo=TZ))
    assert result["state"] == "rolled_back"
    assert prediction["predicted_change_pct"] == -1.0
    assert "adoption" not in prediction


def test_route_records_artifact_and_baseline_versions(tmp_path):
    stamp = datetime(2026, 9, 21, 17, tzinfo=TZ)
    baseline = tmp_path / "models/checkpoints/A.TW.pt"
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes(b"old")
    model = tmp_path / "models/candidate/A.TW.pkl"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"new")
    pending = eligible(tmp_path, stamp)
    with patch.object(adoption, "review", side_effect=lambda *args: adoption.status(tmp_path)):
        promotion = adoption.promote(tmp_path, pending["event_sha256"], stamp)
    target = datetime(2026, 9, 22, 8, 30, tzinfo=TZ)
    candidate = {"ticker": "A.TW", "protocol": adoption.CANDIDATE_PROTOCOL,
                 "target_date": "2026-09-22", "trained_target_end": "2026-09-21",
                 "data_as_of": "2026-09-21", "last_close": 100.0,
                 "predicted_change_pct": 2.0, "predicted_close": 102.0,
                 "direction": "UP", "model_path": "models/candidate/A.TW.pkl",
                 "artifact_sha256": adoption._file_sha(model), "context_available_at": None}
    point = {"ticker": "A.TW", "name": "A", "data_as_of": "2026-09-21",
             "last_close": 100.0, "predicted_change_pct": -1.0, "predicted_close": 99.0,
             "direction": "DOWN", "confidence": .8, "daily_retrain": candidate}
    with patch.dict("config.ALL_STOCKS", {"A.TW": "A"}, clear=True):
        result = adoption.route([point], tmp_path, target)
    assert result["adopted"]
    assert point["predicted_change_pct"] == 2.0
    assert point["production_before_retrain"]["predicted_change_pct"] == -1.0
    assert point["adoption"]["promotion_id"] == promotion["evidence"]["promotion_id"]
    assert point["adoption"]["artifact_sha256"] == adoption._file_sha(model)
    assert "confidence" not in point
