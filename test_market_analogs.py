import pandas as pd
import pytest

from market_analogs import FEATURES, find_analogs


def _frames():
    daily = pd.DataFrame([
        {"date": "2026-09-14", "us_session": "2026-09-11", "new_us_sessions": 1,
         **dict.fromkeys(FEATURES, 0.0), "gap_mean": 1.0, "intraday_mean": -1.0,
         "close_to_close_mean": 0.0},
        {"date": "2026-09-15", "us_session": "2026-09-14", "new_us_sessions": 1,
         **dict.fromkeys(FEATURES, 0.1), "gap_mean": -2.0, "intraday_mean": 2.0,
         "close_to_close_mean": 0.0},
        {"date": "2026-09-16", "us_session": "2026-09-15", "new_us_sessions": 1,
         **dict.fromkeys(FEATURES, 0.1), "gap_mean": 9.0, "intraday_mean": 9.0,
         "close_to_close_mean": 9.0},
    ])
    stocks = pd.DataFrame([
        {"date": "2026-09-14", "ticker": "2327.TW", "gap": 1.0,
         "intraday": -1.0, "close_to_close": 0.0},
        {"date": "2026-09-15", "ticker": "2327.TW", "gap": -2.0,
         "intraday": 2.0, "close_to_close": 0.0},
    ])
    return daily, stocks


def test_analogs_never_include_query_date_or_later():
    daily, stocks = _frames()
    result = find_analogs(dict.fromkeys(FEATURES, 0.1), daily, stocks,
                          before_date="2026-09-16", ticker="2327.TW", k=2)
    assert result["candidate_days"] == 2
    assert all(case["date"] < "2026-09-16" for case in result["cases"])
    assert result["cases"][0]["date"] == "2026-09-15"
    assert result["cases"][0]["intraday"] == 2.0


def test_missing_us_input_is_rejected():
    daily, stocks = _frames()
    with pytest.raises(ValueError, match="缺少美股"):
        find_analogs({"sox_last": 1.0}, daily, stocks, before_date="2026-09-16")


def test_session_filter_keeps_comparable_calendar_days():
    daily, stocks = _frames()
    daily.loc[daily["date"] == "2026-09-15", "new_us_sessions"] = 0
    query = {**dict.fromkeys(FEATURES, 0.1), "new_us_sessions": 1}
    result = find_analogs(query, daily, stocks, before_date="2026-09-16", k=5)
    assert [case["date"] for case in result["cases"]] == ["2026-09-14"]
