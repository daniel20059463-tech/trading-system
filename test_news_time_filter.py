from datetime import datetime, timezone

from news.scraper import fresh_articles, parse_published_at


def test_preserves_full_rss_timestamp_and_filters_future_old_unknown():
    now = datetime(2026, 9, 17, 0, 30, tzinfo=timezone.utc)  # 台北 08:30
    articles = [
        {"title": "past", "timestamp": "Thu, 17 Sep 2026 00:20:00 GMT"},
        {"title": "future", "timestamp": "Thu, 17 Sep 2026 02:00:00 GMT"},
        {"title": "old", "timestamp": "Tue, 08 Sep 2026 00:20:00 GMT"},
        {"title": "unknown", "timestamp": "2026-09-17"},
    ]
    assert [item["title"] for item in fresh_articles(articles, now=now)] == ["past"]


def test_parses_iso_and_unix_publish_times():
    expected = datetime(2026, 9, 17, 0, 0, tzinfo=timezone.utc)
    assert parse_published_at("2026-09-17T08:00:00+08:00") == expected
    assert parse_published_at(int(expected.timestamp())) == expected
