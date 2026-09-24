import json

import pandas as pd
import pytest

import historical_prediction_records as history


def test_backfill_separates_verified_and_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(history.cfg, "ALL_STOCKS", {"A.TW": "A"})
    (tmp_path / "data/raw").mkdir(parents=True)
    pd.DataFrame({"Date": ["2026-09-21", "2026-09-22"], "Close": [100, 103]}).to_csv(
        tmp_path / "data/raw/A.TW.csv", index=False)
    (tmp_path / "news").mkdir()
    (tmp_path / "news/2026-09-21_pre_market.json").write_text(json.dumps({
        "A": {"ml_prediction": {"ticker": "A.TW", "last_close": 99,
                                  "predicted_change_pct": 1, "predicted_close": 99.99}}}), encoding="utf-8")
    (tmp_path / "predictions").mkdir()
    point = {"target_date": "2026-09-22", "generated_at": "2026-09-22T08:30:00+08:00",
             "records": [{"ticker": "A.TW", "data_as_of": "2026-09-21", "last_close": 100,
                          "predicted_change_pct": 2, "predicted_close": 102}]}
    (tmp_path / "predictions/2026-09-22_point.json").write_text(json.dumps(point), encoding="utf-8")
    result = history.rebuild(tmp_path)
    assert result["total_rows"] == 2
    assert result["official"]["rows"] == 1
    assert result["official"]["direction_accuracy"] == 1.0
    frame = pd.read_csv(tmp_path / "reports/historical_prediction_records.csv")
    verified = frame[frame["official_eligible"] == True].iloc[0]
    assert verified["prior_close"] == 100
    assert verified["actual_close"] == 103
    assert verified["actual_pct"] == pytest.approx(3)
    assert verified["accuracy_result"] == "準確"
