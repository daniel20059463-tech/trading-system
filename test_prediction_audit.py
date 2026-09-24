import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from prediction_audit import freeze, score, saved_predictions, update, summarize, TZ
from push_intervals import build_message, chunks


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'data/raw').mkdir(parents=True)
        (self.root / 'data/raw/TEST.TW.csv').write_text(
            'Date,Close\n2026-09-03,100\n2026-09-04,102\n', encoding='utf-8')
        self.p = dict(ticker='TEST.TW', name='測試', last_close=100, data_as_of='2026-09-03',
                      predicted_close=101, predicted_change_pct=1, p_up=60,
                      q50_pct=1, q25_pct=0, q75_pct=2, q10_pct=-2, q90_pct=4,
                      q10s_pct=-1, q90s_pct=3)
        self.now = datetime(2026, 9, 4, 8, 30, tzinfo=TZ)

    def tearDown(self):
        self.tmp.cleanup()

    def test_freeze_roundtrip_and_no_overwrite(self):
        path = freeze([self.p], root=self.root, now=self.now)
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            freeze([dict(self.p, predicted_change_pct=99)], root=self.root, now=self.now)
        self.assertEqual(original, path.read_bytes())
        predictions, actual = saved_predictions('2026-09-04', self.root)
        self.assertEqual(predictions[0]['predicted_close'], 101)
        self.assertEqual(actual, {'TEST': 102})
        stats = update(self.root)['verified']['point']
        self.assertAlmostEqual(stats['abs_error_pp']['mean'], 1)

    def test_late_and_future_data_rejected(self):
        with self.assertRaises(ValueError):
            freeze([self.p], root=self.root, now=self.now.replace(hour=17))
        with self.assertRaises(ValueError):
            freeze([dict(self.p, data_as_of='2026-09-04')], root=self.root, now=self.now)

    def test_weekend_vendor_bar_cannot_be_frozen(self):
        (self.root / 'data/raw/TEST.TW.csv').write_text(
            'Date,Close\n2026-09-18,100\n2026-09-20,999\n', encoding='utf-8')
        monday = datetime(2026, 9, 21, 8, 30, tzinfo=TZ)
        bad = dict(self.p, data_as_of='2026-09-20', last_close=999)
        with self.assertRaisesRegex(ValueError, '週末'):
            freeze([bad], root=self.root, now=monday)

    def test_mismatch_missing_and_flat(self):
        row = score(dict(self.p, last_close=102), '2026-09-04', self.root, True)
        self.assertFalse(row['eligible'])
        self.assertIsNone(row['abs_error_pp'])
        row = score(self.p, '2026-09-07', self.root, True)
        self.assertIsNone(row['direction_correct'])
        self.assertEqual(summarize([row])['direction_correct']['n'], 0)
        (self.root / 'data/raw/TEST.TW.csv').write_text('Date,Close\n2026-09-03,100\n2026-09-04,100\n')
        self.assertFalse(score(dict(self.p, predicted_change_pct=-1), '2026-09-04', self.root, True)['direction_correct'])
        self.assertTrue(score(dict(self.p, predicted_change_pct=0), '2026-09-04', self.root, True)['direction_correct'])

    def test_interval_width_coverage_and_legacy(self):
        row = score(self.p, '2026-09-04', self.root, True)
        self.assertEqual(row['width80_pp'], 6)
        self.assertTrue(row['covered80'])
        legacy = score(self.p, '2026-09-04', self.root)
        self.assertFalse(legacy['eligible'])
        invalid = score(dict(self.p, q10_pct=5, q90_pct=-5), '2026-09-04', self.root, True)
        self.assertIsNone(invalid['covered80'])

    def test_offline_message(self):
        message = build_message([self.p])
        self.assertIn('單一預估區間', message)
        self.assertIn('預估收盤區間', message)
        self.assertNotIn('80%區間', message)
        self.assertTrue(all(len(c) <= 1900 for c in chunks(message * 20)))

    def test_missing_prediction_dates_visible(self):
        (self.root / 'news').mkdir()
        (self.root / 'news/2026-09-03_pre_market.json').write_text(json.dumps({'TEST': {'ml_prediction': self.p}}))
        (self.root / 'news/2026-09-07_pre_market.json').write_text(json.dumps({'TEST': {'ml_prediction': self.p}}))
        result = update(self.root)
        self.assertEqual(result['rows'], 3)
        self.assertEqual(result['statuses']['missing_pre_prediction'], 1)

    def test_latest_day_without_any_prediction_is_visible(self):
        freeze([self.p], root=self.root, now=self.now)
        with (self.root / 'data/raw/TEST.TW.csv').open('a') as f:
            f.write('2026-09-07,103\n')
        result = update(self.root)
        self.assertEqual(result['latest_target'], '2026-09-07')
        self.assertEqual(result['statuses']['missing_pre_prediction'], 1)
        self.assertEqual(result['verified']['point']['direction_correct']['n'], 1)

    def test_tampered_time_excluded_and_actual_price_schema(self):
        path = freeze([self.p], root=self.root, now=self.now)
        payload = json.loads(path.read_text(encoding='utf-8'))
        payload['records'][0]['generated_at'] = '2026-09-04T17:00:00+08:00'
        path.write_text(json.dumps(payload))
        self.assertEqual(saved_predictions('2026-09-04', self.root), ([], {}))
        (self.root / 'news').mkdir()
        (self.root / 'news/2026-09-03_pre_market.json').write_text(json.dumps({'TEST': {'ml_prediction': self.p}}))
        (self.root / 'news/2026-09-03_post_market.json').write_text(json.dumps({'ticker_performance': {'TEST': {'actual_price': 100}}}))
        from prediction_audit import collect
        row = next(r for r in collect(self.root)[0] if r['date'] == '2026-09-03')
        self.assertEqual(row['post_record_actual'], 100)
        self.assertIsNone(row['post_prediction_matches'])

    def test_nested_shadow_candidate_is_scored_separately(self):
        point = dict(self.p, shadow_candidate={
            'version': 'v1', 'ticker': 'TEST.TW', 'data_as_of': '2026-09-03',
            'last_close': 100, 'predicted_change_pct': 2})
        freeze([point], root=self.root, now=self.now)
        result = update(self.root)
        shadow = result['verified']['shadow_candidate']
        self.assertEqual(shadow['direction_correct']['n'], 1)
        self.assertAlmostEqual(shadow['abs_error_pp']['mean'], 0)


if __name__ == '__main__':
    unittest.main()
