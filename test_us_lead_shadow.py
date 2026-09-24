import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import us_lead_shadow as shadow
import us_lead_study as study
import walkforward_training as wf


def _fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(shadow.cfg, "ALL_STOCKS", {"2327.TW": "國巨"})
    dates = pd.bdate_range("2026-06-01", periods=60).strftime("%Y-%m-%d")
    columns = dict.fromkeys(wf.BASE + study.US + study.PEERS, 0.0)
    training = pd.DataFrame([{"date": day, "ticker": "2327.TW", **columns,
                              "vol20": 1.0, "close_to_close": 0.1,
                              "gap": 0.1, "intraday": 0.0} for day in dates])
    sealed = tmp_path / "aligned.csv"
    training.to_csv(sealed, index=False)
    monkeypatch.setattr(shadow, "SEALED", sealed)

    def fake_frame(ticker, root, next_target, now):
        return pd.DataFrame([{"date": pd.Timestamp(next_target),
                              "asof": pd.Timestamp("2026-09-16"),
                              "base_close": 100.0, "vol20": 1.0, **columns}])

    monkeypatch.setattr(shadow.wf, "frame_for", fake_frame)
    us_features = {key: 0.0 for key in study.US + study.PEERS}
    us_features["new_us_sessions"] = 1
    snapshot = {"date": "2026-09-17", "metadata": {"previous_tw_date": "2026-09-16",
                "us_session": "2026-09-16", "us_features": us_features}}
    return snapshot


def test_us_shadow_uses_same_snapshot_and_only_prior_training(tmp_path, monkeypatch):
    snapshot = _fixture(tmp_path, monkeypatch)
    now = datetime(2026, 9, 17, 8, 35, tzinfo=ZoneInfo("Asia/Taipei"))
    result = shadow.make_candidates(snapshot, "2026-09-17", now=now, root=tmp_path)
    candidate = result["2327.TW"]
    assert candidate["trained_target_end"] < "2026-09-17"
    assert candidate["us_session"] == "2026-09-16"
    assert candidate["status"] == "shadow_only_not_formal_prediction"
    assert "close_to_close_price_us" in candidate["predictions_pct"]


def test_us_shadow_settlement_keeps_formal_prediction_unchanged(tmp_path, monkeypatch):
    snapshot = _fixture(tmp_path, monkeypatch)
    now = datetime(2026, 9, 17, 8, 35, tzinfo=ZoneInfo("Asia/Taipei"))
    candidate = shadow.make_candidates(snapshot, "2026-09-17", now=now, root=tmp_path)["2327.TW"]
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "2327.TW.csv").write_text(
        "Date,Open,Close\n2026-09-16,100,100\n2026-09-17,101,102\n", encoding="utf-8"
    )
    ledger_dir = tmp_path / "predictions"
    ledger_dir.mkdir()
    point = {"ticker": "2327.TW", "data_as_of": "2026-09-16",
             "last_close": 100.0, "predicted_change_pct": -0.5,
             "us_lead_shadow": candidate}
    ledger = ledger_dir / "2026-09-17_point.json"
    ledger.write_text(json.dumps({"records": [point]}), encoding="utf-8")
    before = ledger.read_bytes()
    close_time = datetime(2026, 9, 17, 15, 30, tzinfo=ZoneInfo("Asia/Taipei"))
    assert shadow.settle_day("2026-09-17", root=tmp_path, now=now)["reason"] == "only_after_same_day_close"
    result = shadow.settle_day("2026-09-17", root=tmp_path, now=close_time)
    assert result["n"] == 2
    assert ledger.read_bytes() == before
    settlement = tmp_path / "live_evidence" / "us_lead_shadow" / "settlements" / "2026-09-17.json"
    settled_before = settlement.read_bytes()
    assert shadow.settle_day("2026-09-17", root=tmp_path, now=close_time)["reason"] == "already_settled"
    assert settlement.read_bytes() == settled_before
    report = json.loads((tmp_path / "reports" / "us_lead_shadow_live.json").read_text())
    us_row = next(row for row in report["rows"] if row["method"] == "price_us")
    assert us_row["actual_targets"]["close_to_close"] == pytest.approx(2.0)
    assert us_row["actual_targets"]["gap"] == pytest.approx(1.0)
