"""固定60日歷史分位数基準，與逐日誤差校準、真實區間留底做同日比較。"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config as cfg
import walkforward_training as wf
from prediction_audit import collect


def empirical_interval(past, coverage):
    if len(past) < 30:
        return None
    values = np.asarray(past[-60:])
    alpha = 1 - coverage
    return float(np.quantile(values, alpha / 2, method='lower')), float(np.quantile(values, 1 - alpha / 2, method='higher'))


def metric(day, ticker, method, coverage, low, high, actual):
    return dict(date=day, ticker=ticker, method=method, coverage_target=coverage,
                low=low, high=high, actual=actual, covered=low <= actual <= high,
                width=high-low, interval_score=high-low+2/(1-coverage)*max(low-actual, actual-high, 0))


def summarize(frame):
    return [dict(method=m, coverage_target=float(c), n=len(group), days=int(group.date.nunique()),
                 coverage=float(group.covered.mean()), width=float(group.width.mean()),
                 interval_score=float(group.interval_score.mean()))
            for (m,c), group in frame.groupby(['method', 'coverage_target'])]


def run():
    root = wf.HERE
    experiment = root / 'experiments' / wf.VERSION
    out = experiment / 'interval_diagnostic'
    sources = [root / 'quantile_forecast.py'] + sorted((root / 'models/checkpoints').glob('*quantile_best.pt'))
    wf.write_json(out / 'plan.json', dict(locked_at=datetime.now(wf.TZ).isoformat(),
        interval_levels=[.5,.8], history_window=60, minimum_history=30,
        methods=['empirical_returns_60', 'ridge_full', 'ridge_price', 'zero', 'verified_original'],
        tuning=False, status='exploratory historical replay; formal forecasts unchanged',
        source_sha256={str(p.relative_to(root)):wf.digest(p) for p in sources},
        replay_sha256=wf.digest(experiment/'all_predictions.csv')), exclusive=True)
    replay = pd.read_csv(experiment / 'all_predictions.csv')
    rows=[]
    for ticker in cfg.ALL_STOCKS:
        frame = wf.frame_for(ticker, experiment / 'inputs', events=[])
        past=[]
        for row in frame.to_dict('records'):
            day=str(row['date'].date())
            for coverage in (.5,.8):
                interval=empirical_interval(past, coverage)
                if interval and day >= str(replay.date.min()):
                    rows.append(metric(day,ticker,'empirical_returns_60',coverage,*interval,row['actual_pct']))
            past.append(row['actual_pct'])
    for row in replay[replay.method.isin(['ridge_full','ridge_price','zero'])].to_dict('records'):
        for coverage,label in ((.5,'50'),(.8,'80')):
            if np.isfinite(row['lo'+label]):
                rows.append(metric(row['date'],row['ticker'],row['method'],coverage,row['lo'+label],row['hi'+label],row['actual_pct']))
    audited,_=collect(root)
    verified=[r for r in audited if r['kind']=='interval' and r['eligible']]
    for row in verified:
        for coverage,label in ((.5,'50'),(.8,'80')):
            rows.append(metric(row['date'],row['ticker'],'verified_original',coverage,row['lo'+label+'_pct'],row['hi'+label+'_pct'],row['actual_pct']))
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'daily.csv',index=False,encoding='utf-8-sig')
    common_keys={(r['date'],r['ticker']) for r in verified}
    matched=frame[[key in common_keys for key in zip(frame.date,frame.ticker)]]
    summary=dict(recent=summarize(frame[(frame.date>='2026-07-15')&(frame.method!='verified_original')]),
                 matched_verified=summarize(matched),
                 known_limitation='current adjusted returns; legacy models not independently OOS; only 18 genuine dated interval pairs')
    by_ticker=[]
    for ticker,group in frame[frame.date>='2026-07-15'].groupby('ticker'):
        by_ticker.extend([dict(ticker=ticker,**row) for row in summarize(group)])
    pd.DataFrame(by_ticker).to_csv(out/'by_ticker.csv',index=False,encoding='utf-8-sig')
    # 只拆解今日正式模型的寬度組成，不回推為過去實盤。
    from quantile_forecast import _load_quantile,_load_feature_df,_predict_scaled_window,_vol_series
    decomposition=[]
    torch.set_num_threads(1)
    for ticker in cfg.ALL_STOCKS:
        model,scaler,checkpoint=_load_quantile(ticker)
        features=_load_feature_df(ticker,checkpoint['feature_names'])
        values=scaler.transform(features.values.astype(np.float32))[-cfg.WINDOW_SIZE:]
        q10,q25,q50,q75,q90=_predict_scaled_window(model,scaler,checkpoint,values)
        sigma=float(_vol_series(features).iloc[-1])
        correction=float(checkpoint.get('conformal_q80',0))
        low,high=q10-correction*sigma,q90+correction*sigma
        decomposition.append(dict(ticker=ticker,data_as_of=str(features.index[-1].date()),
            raw_width=q90-q10,sigma20=sigma,stored_correction=correction,
            calibration_added_width=2*correction*sigma,unclipped_width=high-low,
            displayed_width_before_news=min(10,high)-max(-10,low),
            scaler_class=type(scaler).__name__))
    summary['today_width_decomposition']=decomposition
    wf.write_json(out/'summary.json',summary)
    lines=['# 區間校準補充比較','','固定60日歷史分位數；校準僅用當時已完成資料。所有歷史比較均為探索回放。','']
    for label,key in [('最近兩月450組','recent'),('同18組真實區間留底','matched_verified')]:
        lines += ['## '+label,'','| 方法 | 目標涵蓋 | n | 實際涵蓋 | 寬度pp | 區間分數（越小越好） |','|---|---:|---:|---:|---:|---:|']
        for item in summary[key]:
            lines.append(f"| {item['method']} | {item['coverage_target']:.0%} | {item['n']} | {item['coverage']:.1%} | {item['width']:.3f} | {item['interval_score']:.3f} |")
    lines += ['','## 今天正式區間寬度拆解','','寬度 = 原始模型寬度 + 2 × 固定校準係數 × 當前20日波動；再套原系統顯示夾限，利多還可能把上界往+10%推。此處沒有套新聞。',
              '','| 股票 | 原始寬度pp | 校準加寬pp | 夾限前pp | 新聞前顯示pp |','|---|---:|---:|---:|---:|']
    for item in decomposition:
        lines.append(f"| {item['ticker']} | {item['raw_width']:.3f} | {item['calibration_added_width']:.3f} | {item['unclipped_width']:.3f} | {item['displayed_width_before_news']:.3f} |")
    lines += ['','尺度核對：原模型分位數已經由scaler反轉為百分點；CQR先把殘差除以波動，保存無單位係數，預測時再乘一次波動。沒有發現此路徑重複乘100或重複inverse_transform。',
              '寬度來源是原始分位數跨度、固定校準係數在目前波動下的調整，以及新聞上界擴張。現存校準段曾參與模型選擇，係數也不隨新增誤差每日更新；mtime不能证明訓練截止。',
              '新候選改以最近60個逐日誤差校準，但是否改善要同時看涵蓋、寬度和區間分數。未硬縮區間，未改正式分位數或正式方向。18組只涵蓋2日，不足以決定替換。']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(summary)


if __name__=='__main__':
    run()
