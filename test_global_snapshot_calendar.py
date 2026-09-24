from datetime import date

from agents.global_snapshot import _expected_us_date, _stale_days


def test_us_labor_day_is_not_counted_as_missing_session():
    # 2026-09-07 為美國 Labor Day；台股 09-08 盤前最近完成的美股是 09-04。
    assert _expected_us_date(date(2026, 9, 8)) == date(2026, 9, 4)
    assert _stale_days(date(2026, 9, 4), as_of=date(2026, 9, 8)) == 0


def test_regular_day_requires_latest_completed_us_session():
    assert _expected_us_date(date(2026, 9, 16)) == date(2026, 9, 15)
    assert _stale_days(date(2026, 9, 14), as_of=date(2026, 9, 16)) == 1
