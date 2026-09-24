from datetime import datetime, date

import pandas as pd
import pytest

from data import fetch_stocks as stocks


def test_pre_market_removes_weekend_and_today_rows():
    raw = pd.DataFrame(
        {"Close": [100.0, 999.0, 888.0]},
        index=pd.to_datetime(["2026-09-18", "2026-09-20", "2026-09-21"]),
    )
    clean = stocks._sanitize_daily_rows(raw, datetime(2026, 9, 21, 8, 30))
    assert list(clean.index.strftime("%Y-%m-%d")) == ["2026-09-18"]


def test_pre_market_validation_rejects_mixed_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(stocks, "RAW_DATA_DIR", str(tmp_path))
    pd.DataFrame({"Date": ["2026-09-18"], "Close": [100]}).to_csv(
        tmp_path / "A.TW.csv", index=False)
    pd.DataFrame({"Date": ["2026-09-17"], "Close": [100]}).to_csv(
        tmp_path / "B.TW.csv", index=False)
    with pytest.raises(ValueError, match="截止日不一致"):
        stocks.validate_pre_market_data(["A.TW", "B.TW"], date(2026, 9, 21))
