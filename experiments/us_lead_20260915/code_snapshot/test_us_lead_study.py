import unittest
import numpy as np
import pandas as pd

import us_lead_study as us


class USLeadTests(unittest.TestCase):
    def calendar(self,start,end):
        cal=us.schedule(start,end)
        for key in us.SYMBOLS:
            cal[key+'_close']=100+np.arange(len(cal),dtype=float)
            cal[key+'_last']=cal[key+'_close'].pct_change(fill_method=None)*100
        return cal

    def test_labor_day_is_not_stale_and_has_no_new_session(self):
        cal=self.calendar('2026-09-01','2026-09-09')
        row=us.align_days([('2026-09-08','2026-09-07')],cal).iloc[0]
        self.assertEqual(row.us_session,'2026-09-04')
        self.assertEqual(row.alignment_status,'ok')
        self.assertEqual(row.new_us_sessions,0)
        self.assertEqual(row.sox_since_tw,0)

    def test_dst_and_early_close_are_real_times(self):
        cal=self.calendar('2026-03-02','2026-03-11')
        a=us.align_days([('2026-03-09','2026-03-06'),('2026-03-10','2026-03-09')],cal)
        self.assertEqual(pd.Timestamp(a.iloc[0].us_close_taipei).hour,5)
        self.assertEqual(pd.Timestamp(a.iloc[1].us_close_taipei).hour,4)
        early=self.calendar('2024-11-20','2024-12-03')
        row=us.align_days([('2024-12-02','2024-11-29')],early).iloc[0]
        self.assertEqual(row.us_session,'2024-11-29')
        self.assertEqual(pd.Timestamp(row.us_close_taipei).hour,2)

    def test_missing_expected_session_is_not_filled_by_older_return(self):
        cal=self.calendar('2026-09-01','2026-09-09')
        cal.loc[cal.session=='2026-09-08','sox_close']=np.nan
        cal['sox_last']=cal.sox_close.pct_change(fill_method=None)*100
        row=us.align_days([('2026-09-09','2026-09-08')],cal).iloc[0]
        self.assertEqual(row.alignment_status,'missing_expected_sox')

    def test_future_us_prices_cannot_change_prior_alignment(self):
        cal=self.calendar('2026-09-01','2026-09-11')
        before=us.align_days([('2026-09-09','2026-09-08')],cal)
        for key in us.SYMBOLS:
            cal.loc[cal.session>='2026-09-09',key+'_close']=999999
            cal[key+'_last']=cal[key+'_close'].pct_change(fill_method=None)*100
        after=us.align_days([('2026-09-09','2026-09-08')],cal)
        pd.testing.assert_frame_equal(before,after)

    def test_taiwan_holiday_accumulates_completed_us_sessions(self):
        cal=self.calendar('2026-02-05','2026-02-25')
        row=us.align_days([('2026-02-23','2026-02-11')],cal).iloc[0]
        self.assertGreater(row.new_us_sessions,1)
        self.assertGreater(row.sox_since_tw,row.sox_last)
        self.assertLess(pd.Timestamp(row.us_available_taipei),pd.Timestamp(row.cutoff_taipei))

    def test_train_scaler_only_sees_prefix(self):
        rng=np.random.default_rng(17)
        data=pd.DataFrame(rng.normal(size=(62,len(us.US))),columns=us.US)
        data['vol20']=1.
        data['close_to_close']=rng.normal(size=62)
        model=us.train(data.iloc[:60],'us_only','2327.TW','close_to_close')
        np.testing.assert_allclose(model[0].mean_,data.iloc[:60][us.US].mean().values)
        old=model.predict(data.iloc[[60]][us.US].to_numpy(float))[0]
        data.loc[60:,'close_to_close']=1e9
        changed=us.train(data.iloc[:60],'us_only','2327.TW','close_to_close')
        self.assertAlmostEqual(old,changed.predict(data.iloc[[60]][us.US].to_numpy(float))[0])


if __name__=='__main__':
    unittest.main()
