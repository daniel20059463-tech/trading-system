import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import premarket_context_interval as context
from push_intervals import build_context_message


def test_discord_context_message_contains_only_one_final_interval():
    payload = {
        "target_date": "2026-09-22",
        "records": [{
            "ticker": "A.TW",
            "name": "A",
            "last_close": 100.0,
            "center_pct": 2.0,
            "q25_price": 99.0,
            "q75_price": 104.0,
        }],
    }

    message = build_context_message(payload)

    assert "預估收盤：99.00～104.00｜中心 102.00" in message
    assert message.count("預估收盤：") == 1
    for hidden_label in ("50%區間", "80%區間", "σ5", "σ20"):
        assert hidden_label not in message


def test_builds_immutable_interval_from_only_prior_errors(tmp_path, monkeypatch):
    rows = [{"date": f"2026-06-{(i % 28) + 1:02d}", "ticker": "A.TW",
             "target": "close_to_close", "method": "price_us", "abs_error_pp": 1.0}
            for i in range(60)]
    sealed = tmp_path / "predictions.csv"
    pd.DataFrame(rows).to_csv(sealed, index=False)
    monkeypatch.setattr(context, "SEALED", sealed)
    (tmp_path / "predictions").mkdir()
    point = {"records": [{"ticker": "A.TW", "name": "A", "data_as_of": "2026-09-21",
                           "last_close": 100.0, "predicted_change_pct": -1.0, "us_lead_shadow": {
                               "target_date": "2026-09-22", "trained_target_end": "2026-09-21",
                               "us_session": "2026-09-21",
                               "predictions_pct": {"close_to_close_price_us": 2.0}}}]}
    (tmp_path / "predictions/2026-09-22_point.json").write_text(json.dumps(point), encoding="utf-8")
    now = datetime(2026, 9, 22, 8, 36, tzinfo=ZoneInfo("Asia/Taipei"))
    first = context.build(now=now, root=tmp_path)
    assert first["records"][0]["q25_pct"] == 1.0
    assert first["records"][0]["q90_pct"] == 3.0
    before = (tmp_path / "predictions/2026-09-22_us_context_interval.json").read_bytes()
    assert context.build(now=now, root=tmp_path) == first
    assert (tmp_path / "predictions/2026-09-22_us_context_interval.json").read_bytes() == before

    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    (raw / "A.TW.csv").write_text(
        "Date,Open,Close\n2026-09-21,100,100\n2026-09-22,101,103\n", encoding="utf-8")
    close_time = datetime(2026, 9, 22, 15, 30, tzinfo=ZoneInfo("Asia/Taipei"))
    settled = context.settle_day(now=close_time, root=tmp_path)
    assert settled["summary"]["context_direction_accuracy"] == 1.0
    assert settled["summary"]["context_mae_pp"] == pytest.approx(1.0)
    assert settled["summary"]["coverage50"] == 1.0
    saved = json.loads((tmp_path / "live_evidence/us_context_interval/settlements/2026-09-22.json").read_text(encoding="utf-8"))
    assert saved["rows"][0]["prior_close"] == 100.0
    assert saved["rows"][0]["actual_close"] == 103.0
    assert saved["rows"][0]["actual_pct"] == pytest.approx(3.0)
    assert saved["rows"][0]["context_result"] == "準確"
    assert (tmp_path / "reports/2026-09-22_prediction_accuracy.md").exists()
    assert context.settle_day(now=close_time, root=tmp_path)["reason"] == "already_settled"
