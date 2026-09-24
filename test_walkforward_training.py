import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd

import walkforward_training as wf


class WalkforwardTests(unittest.TestCase):
    def frame(self):
        rng = np.random.default_rng(5)
        result = pd.DataFrame(rng.normal(size=(66, len(wf.FEATURES['ridge_full']))), columns=wf.FEATURES['ridge_full'])
        result['date'] = pd.bdate_range('2025-01-02', periods=len(result))
        result['asof'] = result['date'] - pd.Timedelta(days=1)
        result['actual_pct'] = rng.normal(size=len(result))
        result['ticker'] = 'TEST.TW'
        result['vol20'] = 2.0
        result['context_available_at'] = None
        result['news_available'] = 0
        return result

    def test_future_labels_cannot_change_earlier_predictions_or_intervals(self):
        frame = self.frame()
        original = wf.replay_ticker(frame, ['ridge_full', 'zero', 'yesterday_full'])
        changed = frame.copy()
        changed.loc[63:, 'actual_pct'] = 500
        altered = wf.replay_ticker(changed, ['ridge_full', 'zero', 'yesterday_full'])
        for old, new in zip(original, altered):
            if old['date'] <= str(frame.loc[63, 'date'].date()):
                self.assertAlmostEqual(old['predicted_pct'], new['predicted_pct'])
                self.assertEqual(old['lo80'], new['lo80'])
                self.assertLess(old['train_target_end'], old['date'])

    def test_news_available_time_and_expiration(self):
        event = dict(ticker='TEST.TW', available_at='2025-01-02T17:00:00+08:00',
                     news={'sentiment_score': .8, 'strategy_signal': 'buy', 'key_catalysts': ['漲價']})
        before, _ = wf.context_values([event], 'TEST.TW', '2025-01-02T08:30:00+08:00')
        after, _ = wf.context_values([event], 'TEST.TW', '2025-01-03T08:30:00+08:00')
        expired, _ = wf.context_values([event], 'TEST.TW', '2025-01-06T08:30:00+08:00')
        self.assertEqual(before['news_available'], 0)
        self.assertEqual(after['kw_漲價'], 1)
        self.assertEqual(after['news_signal'], 1)
        self.assertEqual(expired['news_available'], 0)

    def test_interval_uses_only_enough_completed_errors(self):
        self.assertIsNone(wf.conformal_radius([1] * 29, .8))
        self.assertEqual(wf.conformal_radius([1] * 30, .8), 1)
        row = self.frame().iloc[-1]
        a = wf.metrics_row(row, 'ridge_full', 1, [2] * 60, row['asof'], 60)
        self.assertEqual(a['lo80'], -3)
        self.assertEqual(a['hi80'], 5)

    def test_legacy_pre_requires_log_and_never_uses_filename_as_release_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'news').mkdir()
            (root / 'logs').mkdir()
            (root / 'news/2025-01-02_pre_market.json').write_text(
                json.dumps({'2327': {'sentiment_score': .8}}), encoding='utf-8')
            self.assertEqual(wf.context_events(root)[0], [])
            (root / 'logs/2025-01-02_pre.log').write_text(
                '2025-01-02 08:30:00,000 [INFO] first\n2025-01-03 17:30:00,000 [INFO] rerun', encoding='utf-8')
            events, _ = wf.context_events(root)
            before, _ = wf.context_values(events, '2327.TW', '2025-01-03T08:30:00+08:00')
            after, _ = wf.context_values(events, '2327.TW', '2025-01-04T08:30:00+08:00')
            self.assertEqual(before['news_available'], 0)
            self.assertEqual(after['news_sentiment'], .8)

    def test_price_features_and_target_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data/raw'
            path.mkdir(parents=True)
            dates = pd.bdate_range('2025-01-02', periods=100)
            close = np.arange(100, 200, dtype=float)
            data = pd.DataFrame(dict(Date=dates, Close=close, High=close+1, Low=close-1,
                Volume=100, twii_pct=1, usdtwd_pct=2, murata_pct=3, tdk_pct=4))
            data.to_csv(path / '2327.TW.csv', index=False)
            frame = wf.frame_for('2327.TW', directory, events=[])
            first = frame.iloc[0]
            index = list(dates).index(first['asof'])
            self.assertEqual(first['date'], dates[index+1])
            self.assertAlmostEqual(first['actual_pct'], (close[index+1]/close[index]-1)*100)
            data.loc[70:, 'Close'] *= 5
            data.to_csv(path / '2327.TW.csv', index=False)
            changed = wf.frame_for('2327.TW', directory, events=[])
            np.testing.assert_allclose(frame.iloc[:40][wf.BASE + wf.STRATEGY], changed.iloc[:40][wf.BASE + wf.STRATEGY])


if __name__ == '__main__':
    unittest.main()
