"""釐清純美股對照及交易日區塊不確定性；不改主實驗或正式流程。"""
import json
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import us_lead_study as us


def block_interval(values,seed=20260916,block=5,reps=3000):
    values=np.asarray(values,float)
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,len(values),size=(reps,int(np.ceil(len(values)/block))))
    indices=((starts[:,:,None]+np.arange(block))%len(values)).reshape(reps,-1)[:,:len(values)]
    means=values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means,[.025,.975])]


def run():
    out=us.OUT/'definition_and_uncertainty'
    plan=out/'plan.json'
    if not plan.exists():
        us.wf.write_json(plan,dict(locked_at=datetime.now(us.wf.TZ).isoformat(),
            reason='Original us_only features were US-only but output scale used prior Taiwan volatility; add literal US-only raw-target control.',
            primary_comparison_unchanged=True,after_viewing_primary_results=True,
            raw_control='same eight US/calendar features, alpha100, expanding min60, half-life126; raw three targets; no Taiwan volatility input/scale',
            uncertainty='paired daily mean across all ten stocks; circular blocks5 trading days;3000resamples;seed20260916;approximate95percent descriptive interval',
            historical_status='supplementary exploratory diagnostic, not model selection or independent evidence'),exclusive=True)
    frame=pd.read_csv(us.OUT/'aligned.csv')
    result=[]
    path=out/'us_only_raw_predictions.csv'
    if path.exists():
        pure=pd.read_csv(path)
    else:
        with threadpool_limits(limits=1):
            for ticker,data in frame.groupby('ticker',sort=False):
                data=data.reset_index(drop=True)
                x=data[us.US].to_numpy(float)
                y=data[us.TARGETS].to_numpy(float)
                for i in range(60,len(data)):
                    model=make_pipeline(StandardScaler(),Ridge(alpha=100))
                    weights=np.exp2(-np.arange(i-1,-1,-1)/126)
                    model.fit(x[:i],y[:i],ridge__sample_weight=weights)
                    forecast=model.predict(x[i:i+1])[0]
                    for index,target in enumerate(us.TARGETS):
                        actual=float(y[i,index])
                        prediction=float(forecast[index])
                        sign=lambda v:0 if abs(v)<1e-9 else (1 if v>0 else -1)
                        result.append(dict(ticker=ticker,date=data.date.iloc[i],method='us_only_raw',target=target,
                            train_end=data.date.iloc[i-1],predicted_pct=prediction,actual_pct=actual,
                            correct=sign(prediction)==sign(actual),abs_error_pp=abs(prediction-actual)))
                print('pure US',ticker,flush=True)
        pure=pd.DataFrame(result)
        pure.to_csv(path,index=False,encoding='utf-8-sig')
    pred=pd.read_csv(us.OUT/'predictions.csv')
    dates=sorted(pred.date.unique())
    pure_summary=[]
    intervals=[]
    for window,days in [('all',dates),('after_validation',dates[100:]),('last120',dates[-120:]),('last60',dates[-60:]),('last20',dates[-20:])]:
        for target in us.TARGETS:
            g=pure[(pure.date.isin(days))&(pure.target==target)]
            pure_summary.append(dict(window=window,target=target,**us.metrics(g)))
        if window not in ['after_validation','last60','last20']: continue
        selected=pred[(pred.date.isin(days))&(pred.target=='close_to_close')]
        errors=selected.pivot(index=['date','ticker'],columns='method',values='abs_error_pp')
        for reference in ['price','zero']:
            diff=(errors[reference]-errors.price_us).groupby(level='date').mean()
            intervals.append(dict(window=window,reference=reference,days=len(diff),
                mae_reduction_pp=float(diff.mean()),approx95=block_interval(diff.values)))
    us.wf.write_json(out/'summary.json',dict(pure_us=pure_summary,paired_uncertainty=intervals))
    lines=['# 純美股定義與誤差不確定性補充','',
        '主實驗中us_only僅使用美股／日曆特徵，但預測幅度仍乘台股前20日波動；因此更精確的名稱是「美股特徵＋台股波動尺度」。'
        '本補充把目標改為原始報酬，不用台股波動縮放，才是嚴格的純美股對照。參數及日期固定，沒有擇優更換主比較。'
        '此補充於看過主實驗後提出，先留plan再計算，明列為探索性定義查核。','',
        '| 純美股、最近60日目標 | 方向 | MAE百分點 |','|---|---:|---:|']
    for row in pure_summary:
        if row['window']=='last60':
            lines.append(f"| {row['target']} | {row['accuracy']:.1%} | {row['mae']:.4f} |")
    lines += ['','## 同交易日配對誤差的不確定性','',
        '先將同日十檔的誤差改善取平均，再用5個交易日循環區塊、固定seed、3000次重抽樣，'
        '保留同日股票相關與短期日期相依。下列為近似描述區間；選擇過的股票及歷史資料限制仍存在，不能保證未來結果。', '',
        '| 窗口／比較基準 | 平均MAE改善pp | 近似95%區間pp |','|---|---:|---:|']
    for row in intervals:
        lo,hi=row['approx95']
        lines.append(f"| {row['window']}／{row['reference']} | {row['mae_reduction_pp']:+.4f} | {lo:+.4f}～{hi:+.4f} |")
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    report=us.HERE.parent/'半導體預測系統/13 美股領先台股逐日驗證 2026-09-16.md'
    text=report.read_text(encoding='utf-8').replace('| 只用美股 |','| 美股特徵＋台股波動尺度 |')
    marker='## 純美股定義與交易日配對不確定性（補充）'
    if marker not in text:
        raw=next(r for r in pure_summary if r['window']=='last60' and r['target']=='close_to_close')
        paired=next(r for r in intervals if r['window']=='last60' and r['reference']=='price')
        lo,hi=paired['approx95']
        text+='\n\n'+marker+'\n\n'
        text+=f"原表的美股特徵版本仍以台股前20日波動還原幅度，不能稱完全只用美股。按字面重新核對、去掉台股波動後，純美股最近60日全日方向{raw['accuracy']:.1%}、MAE{raw['mae']:.4f}百分點。此為定義修正的補充比較，主模型與參數不變。\n\n"
        text+=f"價格＋美股相對原價格候選的平均MAE改善{paired['mae_reduction_pp']:.4f}百分點；以交易日聚類、5日區塊的近似95%區間為{lo:+.4f}～{hi:+.4f}。這是歷史樣本的不確定性描述，非未來績效保證。\n\n"
        text+='[補充表與方法](../trading-system/experiments/us_lead_20260915/definition_and_uncertainty/report.md)。\n'
        report.write_text(text,encoding='utf-8')
    shutil.copy2(__file__,out/'code.py')
    print(json.dumps(dict(pure_recent=[r for r in pure_summary if r['window']=='last60'],intervals=intervals),ensure_ascii=False,indent=2))


if __name__=='__main__': run()
