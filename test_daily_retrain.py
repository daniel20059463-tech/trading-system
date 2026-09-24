import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import daily_retrain as dr
from prediction_audit import TZ, freeze, read_json, collect
import test_walkforward_training as fixtures


class DailyRetrainTests(unittest.TestCase):
    def test_gate_pass_is_review_recommendation_and_preserves_formal_forecast(self):
        point = dict(ticker='TEST.TW', predicted_change_pct=-2, confidence=.8)
        candidate = dict(ticker='TEST.TW', predicted_change_pct=1)
        with patch.object(dr, 'live_pairs', return_value=[]), patch.object(dr, 'promotion_decision', return_value={'approved':True}), patch.object(dr, 'candidate_for', return_value=candidate):
            decision = dr.attach_candidates([point], now=datetime(2026,9,16,8,30,tzinfo=TZ))
        self.assertEqual(point['predicted_change_pct'], -2)
        self.assertEqual(point['confidence'], .8)
        self.assertTrue(decision['review_required'])
        self.assertEqual(point['daily_retrain']['predicted_change_pct'], 1)

    def test_similarity_alone_never_promotes(self):
        pair = dict(date='2026-09-10', ticker='TEST.TW', new_mae=.1, old_mae=.1,
                    zero_mae=.2, new_correct=True, old_correct=True, new_covered80=True,
                    new_width80=2, actual_pct=1, predicted_pct=.9)
        self.assertFalse(dr.promotion_decision([pair] * 200)['approved'])

    def test_many_correlated_rows_do_not_replace_twenty_days(self):
        pairs = []
        for i in range(200):
            pairs.append(dict(date=f'2026-09-{1+i%10:02}', ticker='TEST.TW', new_mae=.1,
                old_mae=.5, zero_mae=1, new_correct=True, old_correct=False,
                new_covered80=(i%5 != 0), new_width80=2, actual_pct=1, predicted_pct=.9))
        self.assertFalse(dr.promotion_decision(pairs)['approved'])

    def test_acceptance_requires_error_direction_and_coverage(self):
        pairs = [dict(date=f'2026-08-{1+i%20:02}', ticker=f'{i//20}.TW', new_mae=.3, old_mae=.6,
                      zero_mae=.7, new_correct=i%3!=0, old_correct=i%2==0,
                      new_covered80=i%5!=0) for i in range(200)]
        self.assertTrue(dr.promotion_decision(pairs)['approved'])
        for row in pairs:
            row['new_mae'] = .8
        self.assertFalse(dr.promotion_decision(pairs)['approved'])

    def test_one_exceptional_day_cannot_promote_noisy_candidate(self):
        pairs = []
        for day in range(20):
            new = .99 if day < 12 else 1.03 if day < 19 else 0.0
            for ticker in range(10):
                pairs.append(dict(date=f'2026-08-{day + 1:02d}', ticker=f'{ticker}.TW',
                    new_mae=new, old_mae=1.0, zero_mae=1.0,
                    new_correct=True, old_correct=False,
                    new_covered80=(ticker % 5 != 0)))
        decision = dr.promotion_decision(pairs)
        self.assertFalse(decision['approved'])
        self.assertGreaterEqual(decision['day_win_rate'], .6)
        self.assertLessEqual(decision['leave_one_day_out_worst_advantage_pp'], 0)

    def test_incomplete_day_cannot_distort_review_window(self):
        pairs = [dict(date=f'2026-08-{day + 1:02d}', ticker=f'{ticker}.TW',
                      new_mae=.3, old_mae=.7, zero_mae=.8,
                      new_correct=True, old_correct=False, new_covered80=(ticker % 5 != 0))
                 for day in range(20) for ticker in range(10)]
        pairs.extend(dict(date='2026-08-21', ticker=f'{ticker}.TW',
                          new_mae=10, old_mae=.1, zero_mae=.1,
                          new_correct=False, old_correct=True, new_covered80=False)
                     for ticker in range(5))
        decision = dr.promotion_decision(pairs)
        self.assertTrue(decision['approved'])
        self.assertEqual(decision['n'], 200)
        self.assertEqual(decision['complete_days'], 20)

    def test_snapshot_is_immutable_and_late_snapshot_rejected(self):
        now = datetime(2026, 9, 16, 8, 30, tzinfo=TZ)
        with tempfile.TemporaryDirectory() as directory:
            path = dr.snapshot_context({'2327': {'sentiment_score': .5}}, [{'ticker': '2327.TW', 'action': 'buy'}], directory, now)
            saved = read_json(path)
            self.assertEqual(saved['records'][0]['strategy']['action'], 'buy')
            with self.assertRaises(FileExistsError):
                dr.snapshot_context({}, [], directory, now)
            with self.assertRaises(ValueError):
                dr.snapshot_context({}, [], directory, now.replace(hour=17))

    def test_training_idempotence_and_future_target_guard(self):
        # 用隔離資料測訓練 -> 保存 -> 載入；確認相同輸入不改寫模型。
        frame = fixtures.WalkforwardTests().frame()
        frame['base_close'] = 100.
        now = datetime(2025, 5, 1, 17, tzinfo=TZ)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'data/raw'
            raw.mkdir(parents=True)
            (raw / 'TEST.TW.csv').write_text('test source hash')
            with patch.dict(dr.cfg.ALL_STOCKS, {'TEST.TW': '測試'}, clear=True), patch.object(dr.wf, 'frame_for', return_value=frame):
                manifest = dr.train_latest(root, now)
                state = read_json(root / 'models/daily_model_state.json')
                model_path = root / state['latest'] / 'TEST.TW.pkl'
                before = model_path.read_bytes()
                dr.train_latest(root, now)
                self.assertEqual(before, model_path.read_bytes())
                self.assertEqual(manifest['models']['TEST.TW']['n'], len(frame))
                with self.assertRaises(ValueError):
                    dr.candidate_for('TEST.TW', manifest['trained_through'], root)
                target = (pd.Timestamp(manifest['trained_through']) + pd.Timedelta(days=1)).date().isoformat()
                future = frame.iloc[[-1]].copy()
                future['asof'] = pd.Timestamp(manifest['trained_through'])
                future['date'] = pd.Timestamp(target)
                future['context_available_at'] = np.nan
                with patch.object(dr.wf, 'frame_for', return_value=future):
                    prediction = dr.candidate_for('TEST.TW', target, root)
                self.assertTrue(np.isfinite(prediction['predicted_change_pct']))
                self.assertIsNone(prediction['context_available_at'])
                json.dumps(prediction, allow_nan=False)
                self.assertEqual(prediction['training_n'], len(frame))
                model_path.write_bytes(before + b'corrupt')
                with self.assertRaises(ValueError):
                    dr.candidate_for('TEST.TW', target, root)

    def test_frozen_pair_scoring_excludes_future_context_and_wrong_ticker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data/raw').mkdir(parents=True)
            (root / 'data/raw/TEST.TW.csv').write_text('Date,Close\n2026-09-03,100\n2026-09-04,102\n')
            candidate = dict(ticker='TEST.TW', protocol=dr.wf.VERSION, trained_target_end='2026-09-03',
                data_as_of='2026-09-03', target_date='2026-09-04', last_close=100,
                predicted_change_pct=1.5, q10_pct=-1, q90_pct=3, vol20=1,
                context_available_at='2026-09-04T08:20:00+08:00')
            point = dict(candidate, predicted_change_pct=-1, daily_retrain=candidate)
            stamp = datetime(2026, 9, 4, 8, 30, tzinfo=TZ)
            freeze([point], root=root, now=stamp)
            pairs = dr.live_pairs(root)
            self.assertEqual(len(pairs), 1)
            self.assertAlmostEqual(pairs[0]['new_mae'], .5)
            self.assertAlmostEqual(pairs[0]['old_mae'], 3)
            rows, _ = collect(root, wiki=root / 'no_wiki')
            self.assertEqual(sum(r['kind']=='daily_retrain' and r['eligible'] for r in rows), 1)
            saved = read_json(root / 'predictions/2026-09-04_point.json')['records'][0]
            saved['daily_retrain']['context_available_at'] = '2026-09-04T17:00:00+08:00'
            with patch.object(dr, 'read_json', return_value={'target_date': '2026-09-04', 'records': [saved]}):
                self.assertEqual(dr.live_pairs(root), [])
            saved['daily_retrain']['context_available_at'] = None
            saved['daily_retrain']['ticker'] = 'WRONG.TW'
            with patch.object(dr, 'read_json', return_value={'target_date': '2026-09-04', 'records': [saved]}):
                self.assertEqual(dr.live_pairs(root), [])


if __name__ == '__main__':
    unittest.main()
