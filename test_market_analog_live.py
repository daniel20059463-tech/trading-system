import json
import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import market_analog_live as live
from market_analog_live import _live_us_query, capture_pre_market, prompt_context


def _history(symbol, stale=None):
    if symbol == stale:
        dates = ["2026-09-11", "2026-09-14"]
    else:
        dates = ["2026-09-14", "2026-09-15"]
    return pd.DataFrame({"Close": [100.0, 101.0]}, index=pd.to_datetime(dates))


def test_live_query_requires_all_five_fresh_us_sessions():
    query, meta = _live_us_query("2026-09-16", "2026-09-15", _history)
    assert meta["us_session"] == "2026-09-15"
    assert query["new_us_sessions"] == 1
    assert query["sox_last"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="過期"):
        _live_us_query("2026-09-16", "2026-09-15",
                       lambda symbol: _history(symbol, stale="^SOX"))


def test_live_query_retries_transient_invalid_close():
    calls = {"ON": 0}

    def transient(symbol):
        frame = _history(symbol)
        if symbol == "ON":
            calls["ON"] += 1
            if calls["ON"] == 1:
                frame.loc[frame.index[-1], "Close"] = 0.0
        return frame

    query, _ = _live_us_query("2026-09-16", "2026-09-15", transient)
    assert query["on_last"] == pytest.approx(1.0)
    assert calls["ON"] == 2


def test_live_query_keeps_core_indices_when_peer_quote_is_missing():
    query, meta = _live_us_query(
        "2026-09-16", "2026-09-15",
        lambda symbol: _history(symbol, stale=symbol) if symbol == "ON" else _history(symbol))
    assert query["sox_last"] == pytest.approx(1.0)
    assert "on_last" not in query
    assert meta["missing_symbols"] == ["ON"]


def test_capture_is_immutable_and_uses_only_past_cases(tmp_path):
    now = datetime(2026, 9, 16, 8, 35, tzinfo=ZoneInfo("Asia/Taipei"))
    first = capture_pre_market(now=now, snapshot_dir=tmp_path,
                               history_getter=_history, previous_tw_date="2026-09-15")
    second = capture_pre_market(now=now, snapshot_dir=tmp_path,
                                history_getter=lambda _: None,
                                previous_tw_date="2026-09-15")
    assert first == second
    assert first["metadata"]["us_session"] == "2026-09-15"
    assert all(case["date"] < "2026-09-16" for case in first["aggregate"]["cases"])
    assert len(first["by_ticker"]) == 10
    assert "描述過去8個相似日" in prompt_context(first)
    assert json.loads((tmp_path / "2026-09-16.json").read_text(encoding="utf-8")) == first


def test_capture_rejects_after_open(tmp_path):
    now = datetime(2026, 9, 16, 9, 0, tzinfo=ZoneInfo("Asia/Taipei"))
    with pytest.raises(RuntimeError):
        capture_pre_market(now=now, snapshot_dir=tmp_path,
                           history_getter=_history, previous_tw_date="2026-09-15")


def test_capture_rejects_weekend_as_previous_tw_session(tmp_path):
    now = datetime(2026, 9, 21, 8, 35, tzinfo=ZoneInfo("Asia/Taipei"))
    with pytest.raises(ValueError, match="前一台股交易日無效"):
        capture_pre_market(now=now, snapshot_dir=tmp_path,
                           history_getter=_history, previous_tw_date="2026-09-20")


def test_settlement_adds_new_case_without_changing_pre_market(tmp_path, monkeypatch):
    before_open = datetime(2026, 9, 16, 8, 35, tzinfo=ZoneInfo("Asia/Taipei"))
    capture_pre_market(now=before_open, snapshot_dir=tmp_path,
                       history_getter=_history, previous_tw_date="2026-09-15")
    snapshot_path = tmp_path / "2026-09-16.json"
    digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    raw = tmp_path / "raw"
    raw.mkdir()
    for ticker in live.cfg.ALL_STOCKS:
        (raw / f"{ticker}.csv").write_text(
            "Date,Open,Close\n2026-09-15,100,100\n2026-09-16,101,102\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(live.cfg, "RAW_DATA_DIR", str(raw))
    after_close = datetime(2026, 9, 16, 15, 0, tzinfo=ZoneInfo("Asia/Taipei"))
    result = live.settle_day(now=after_close, snapshot_dir=tmp_path)
    assert result["settled"] == 10
    assert hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == digest
    daily, stocks = live.load_reference_library(tmp_path)
    assert len(daily[daily["date"] == "2026-09-16"]) == 1
    assert len(stocks[stocks["date"] == "2026-09-16"]) == 10
    assert daily[daily["date"] == "2026-09-16"].iloc[0]["gap_mean"] == pytest.approx(1.0)
