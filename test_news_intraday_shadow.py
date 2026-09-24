import json
import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import news_intraday_shadow as shadow
from news_intraday_shadow import attach_to_strategies, capture_pre_market


def _news(score=0.8):
    return {"2327": {"sentiment_score": score, "confidence": 0.75,
                     "news_count": 10, "strategy_signal": "buy"}}


def test_snapshot_is_immutable_on_same_day(tmp_path):
    now = datetime(2026, 9, 17, 8, 32, tzinfo=ZoneInfo("Asia/Taipei"))
    first = capture_pre_market(_news(0.8), now=now, directory=tmp_path)
    second = capture_pre_market(_news(-0.8), now=now, directory=tmp_path)
    assert first == second
    saved = json.loads((tmp_path / "2026-09-17.json").read_text(encoding="utf-8"))
    assert saved["tickers"]["2327.TW"]["sentiment_score"] == 0.8


def test_snapshot_rejects_after_open(tmp_path):
    now = datetime(2026, 9, 17, 9, 0, tzinfo=ZoneInfo("Asia/Taipei"))
    with pytest.raises(RuntimeError):
        capture_pre_market(_news(), now=now, directory=tmp_path)


def test_attaches_shadow_without_changing_action(tmp_path):
    now = datetime(2026, 9, 17, 8, 32, tzinfo=ZoneInfo("Asia/Taipei"))
    snap = capture_pre_market(_news(), now=now, directory=tmp_path)
    strategies = [{"ticker": "2327.TW", "action": "hold"}]
    attach_to_strategies(strategies, snap)
    assert strategies[0]["action"] == "hold"
    assert strategies[0]["intraday_news_shadow"]["formal_strategy_input"] is False


def test_settlement_does_not_modify_pre_market_snapshot(tmp_path, monkeypatch):
    now = datetime(2026, 9, 17, 8, 32, tzinfo=ZoneInfo("Asia/Taipei"))
    capture_pre_market(_news(), now=now, directory=tmp_path)
    snapshot = tmp_path / "2026-09-17.json"
    before = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    raw = tmp_path / "raw"
    raw.mkdir()
    for ticker in shadow.cfg.ALL_STOCKS:
        (raw / f"{ticker}.csv").write_text(
            "Date,Open,Close\n2026-09-17,100,102\n", encoding="utf-8"
        )
    monkeypatch.setattr(shadow.cfg, "RAW_DATA_DIR", str(raw))
    after_close = datetime(2026, 9, 17, 15, 0, tzinfo=ZoneInfo("Asia/Taipei"))
    result = shadow.settle_day("2026-09-17", directory=tmp_path, now=after_close)
    assert result["settled"] == 10
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == before
    assert (tmp_path / "settlements" / "2026-09-17.json").exists()
