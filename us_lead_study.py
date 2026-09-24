"""美股已完成時段 -> 台股下一開盤日：隔離、可重跑的探索研究。"""
import argparse
import json
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / '.research_deps/us_alignment'))
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import config as cfg
import walkforward_training as wf
from prediction_audit import collect

OUT = HERE / 'experiments/us_lead_20260915'
SYMBOLS = {'sox':'^SOX','nasdaq':'^IXIC','sp500':'^GSPC','on':'ON','vsh':'VSH'}
US = [key+'_'+kind for key in ['sox','nasdaq','sp500'] for kind in ['last','since_tw']]+['new_us_sessions','us_age_hours']
PEERS = [key+'_'+kind for key in ['on','vsh'] for kind in ['last','since_tw']]
METHODS = ['zero','sox_copy','price','us_only','price_us','price_us_peers']
TARGETS = ['close_to_close','gap','intraday']
SOURCES = {'hours':'https://www.nyse.com/trade/hours-calendars',
 'labor_day':'https://www.nasdaqtrader.com/TraderNews.aspx?id=ETS2026-47',
 'taiwan_hours':'https://www.twse.com.tw/en/products/system/trading.html',
 'calendar_implementation':'https://pandas-market-calendars.readthedocs.io/en/latest/usage.html',
 'price_provider':'https://finance.yahoo.com/'}


def schedule(start='2023-11-01',end='2026-09-16'):
    result=mcal.get_calendar('NYSE').schedule(start,end)
    nasdaq=mcal.get_calendar('NASDAQ').schedule(start,end)
    pd.testing.assert_frame_equal(result[['market_open','market_close']],nasdaq[['market_open','market_close']])
    result=result.reset_index(names='session')
    result['session']=result['session'].dt.strftime('%Y-%m-%d')
    result['available_at']=result['market_close']+pd.Timedelta(hours=1)
    return result


def prepare(out=OUT):
    import yfinance as yf
    out=Path(out)
    plan_path=out/'plan.json'
    if not plan_path.exists():
        wf.write_json(plan_path,dict(locked_at=datetime.now(wf.TZ).isoformat(),version='us_lead_v1',
            tickers=list(cfg.ALL_STOCKS),us_symbols=SYMBOLS,methods=METHODS,targets=TARGETS,
            price_features=wf.BASE,us_features=US,peer_features=PEERS,
            training='expanding; min60; age half-life126; Ridge alpha100; scaler past only; target/prior_vol20',
            validation='first100 forecast days designated chronological validation; no tuning; remaining dates exploratory replay',
            reporting_windows=['all','after_validation','last120','last60','last20'],
            primary='price_us vs price on same stock/date close-to-close; others fixed secondary diagnostics',
            cutoff='08:30 Asia/Taipei; US regular close +60min conservative availability; actual US holiday/early-close calendar',
            missing='exclude entire stock/date if latest expected US session or its previous scheduled close missing; no zero fill',
            no_news=True,no_production_change=True,historical_status='exploratory, Taiwan history through2026-09-15 already examined',
            sources=SOURCES),exclusive=True)
    raw=out/'inputs/data/raw'
    raw.mkdir(parents=True,exist_ok=True)
    for ticker in cfg.ALL_STOCKS:
        destination=raw/f'{ticker}.csv'
        if not destination.exists():
            shutil.copy2(HERE/'data/raw'/destination.name,destination)
    history=out/'inputs/snapshot_history.md'
    if not history.exists():
        shutil.copy2(Path(cfg.WIKI_DIR)/'snapshot_history.md',history)
    ledger=out/'inputs/predictions'
    ledger.mkdir(exist_ok=True)
    for source in (HERE/'predictions').glob('*.json'):
        if not (ledger/source.name).exists():
            shutil.copy2(source,ledger/source.name)
    cal=schedule()
    cal.to_csv(out/'inputs/us_schedule.csv',index=False)
    usdir=out/'inputs/us'
    usdir.mkdir(exist_ok=True)
    retrieval=[]
    for key,symbol in SYMBOLS.items():
        path=usdir/f'{key}.csv'
        if not path.exists():
            prices=yf.download(symbol,start='2023-11-01',end='2026-09-16',auto_adjust=True,
                               progress=False,threads=False,timeout=25)
            if isinstance(prices.columns,pd.MultiIndex):
                prices.columns=prices.columns.get_level_values(0)
            if prices.empty:
                raise ValueError(f'{symbol} 無可用行情；已完成輸入保留，可再prepare')
            prices.index=pd.to_datetime(prices.index).tz_localize(None).normalize()
            prices.index.name='Date'
            prices.to_csv(path,encoding='utf-8-sig')
        prices=pd.read_csv(path)
        retrieval.append(dict(symbol=symbol,rows=len(prices),first=prices.Date.iloc[0],last=prices.Date.iloc[-1],
                              retrieved_at=datetime.now(wf.TZ).isoformat(),sha256=wf.digest(path)))
        print(symbol,len(prices),prices.Date.iloc[-1],flush=True)
    wf.write_json(out/'inputs_manifest.json',dict(retrieval=retrieval,
        sha256={str(p.relative_to(out/'inputs')):wf.digest(p) for p in (out/'inputs').rglob('*') if p.is_file()}))


def load_us(out):
    cal=pd.read_csv(out/'inputs/us_schedule.csv')
    for key in ['market_open','market_close','available_at']:
        cal[key]=pd.to_datetime(cal[key],utc=True)
    for key in SYMBOLS:
        raw=pd.read_csv(out/f'inputs/us/{key}.csv')
        raw['Date']=raw.Date.str[:10]
        values=raw.set_index('Date').Close
        cal[key+'_close']=cal.session.map(values)
        # 前一個「排定交易時段」缺資料時，不以更早一列冒充昨日變動。
        cal[key+'_last']=cal[key+'_close'].pct_change(fill_method=None)*100
    return cal


def align_days(days,cal):
    rows=[]
    for day,prior in days:
        cutoff=pd.Timestamp(day+'T08:30:00',tz='Asia/Taipei').tz_convert('UTC')
        tw_prior_close=pd.Timestamp(prior+'T13:30:00',tz='Asia/Taipei').tz_convert('UTC')
        eligible=cal[cal.available_at < cutoff]
        if eligible.empty:
            rows.append(dict(date=day,alignment_status='no_completed_session'))
            continue
        latest=eligible.iloc[-1]
        anchors=eligible[eligible.market_close <= tw_prior_close]
        if anchors.empty:
            rows.append(dict(date=day,alignment_status='no_anchor_before_previous_tw_close'))
            continue
        anchor=anchors.iloc[-1]
        new=eligible[eligible.market_close>tw_prior_close]
        row=dict(date=day,us_session=latest.session,us_close_utc=latest.market_close.isoformat(),
                 us_close_taipei=latest.market_close.tz_convert('Asia/Taipei').isoformat(),
                 us_available_taipei=latest.available_at.tz_convert('Asia/Taipei').isoformat(),
                 cutoff_taipei=cutoff.tz_convert('Asia/Taipei').isoformat(),
                 previous_tw_date=prior,new_us_sessions=len(new),us_age_hours=(cutoff-latest.market_close).total_seconds()/3600,
                 repeated_session=len(new)==0,alignment_status='ok')
        for key in SYMBOLS:
            row[key+'_last']=latest[key+'_last']
            row[key+'_since_tw']=(latest[key+'_close']/anchor[key+'_close']-1)*100
            if not np.isfinite(row[key+'_last']) or not np.isfinite(row[key+'_since_tw']):
                row['alignment_status']='missing_expected_'+key
        rows.append(row)
    return pd.DataFrame(rows)


def make_frames(out):
    cal=load_us(out)
    frames=[]
    exclusions=[]
    for ticker in cfg.ALL_STOCKS:
        raw=wf.read_prices(out/'inputs',ticker)
        raw['date']=raw.Date.dt.strftime('%Y-%m-%d')
        raw['previous_date']=raw.Date.shift(1).dt.strftime('%Y-%m-%d')
        raw['gap']=(raw.Open/raw.Close.shift(1)-1)*100
        raw['intraday']=(raw.Close/raw.Open-1)*100
        raw['close_to_close']=(raw.Close/raw.Close.shift(1)-1)*100
        assert np.allclose((1+raw.gap.iloc[1:]/100)*(1+raw.intraday.iloc[1:]/100)-1,raw.close_to_close.iloc[1:]/100)
        aligned=align_days(list(zip(raw.date.iloc[1:],raw.previous_date.iloc[1:])),cal)
        full=raw.merge(aligned,on='date',how='left')
        full['ticker']=ticker
        feature=wf.frame_for(ticker,out/'inputs',events=[])
        feature['date']=feature.date.dt.strftime('%Y-%m-%d')
        data=feature[['date']+wf.BASE].merge(full,on='date')
        data['valid_open']=np.isfinite(data.Open)&(data.Open>0)
        valid=(data.alignment_status=='ok')&data.valid_open
        exclusions.extend(data.loc[~valid,['date','ticker','alignment_status','valid_open']].to_dict('records'))
        frames.append(data.loc[valid].copy())
    result=pd.concat(frames,ignore_index=True)
    wf.write_json(out/'alignment_exclusions.json',exclusions)
    result.to_csv(out/'aligned.csv',index=False,encoding='utf-8-sig')
    return result


def columns(method,ticker):
    if method=='price': return wf.BASE
    if method=='us_only': return US
    if method=='price_us': return wf.BASE+US
    return wf.BASE+US+(PEERS if ticker in cfg.POWER_STOCKS else [])


def train(frame,method,ticker,target):
    model=make_pipeline(StandardScaler(),Ridge(alpha=100))
    weights=np.exp2(-np.arange(len(frame)-1,-1,-1)/126)
    model.fit(frame[columns(method,ticker)].to_numpy(float),frame[target].to_numpy(float)/frame.vol20.to_numpy(float),
              ridge__sample_weight=weights)
    return model


def replay_ticker(frame,ticker):
    rows=[]
    for i in range(60,len(frame)):
        past,row=frame.iloc[:i],frame.iloc[i]
        assert past.date.max()<row.date
        for target in TARGETS:
            forecasts={'zero':0.,'sox_copy':float(row.sox_last) if row.new_us_sessions else 0.}
            for method in METHODS[2:]:
                model=train(past,method,ticker,target)
                forecasts[method]=float(model.predict(row[columns(method,ticker)].to_numpy(float).reshape(1,-1))[0]*row.vol20)
            for method,prediction in forecasts.items():
                actual=float(row[target])
                sign=lambda x: 0 if abs(x)<1e-9 else (1 if x>0 else -1)
                rows.append(dict(date=row.date,ticker=ticker,target=target,method=method,
                    train_end=past.date.max(),training_n=i,phase='validation' if i<160 else 'exploratory_evaluation',
                    predicted_pct=prediction,actual_pct=actual,abs_error_pp=abs(prediction-actual),
                    correct=sign(prediction)==sign(actual),us_session=row.us_session))
        if (i-60)%200==0: print(ticker,row.date,flush=True)
    return rows


def metrics(frame):
    return dict(n=len(frame),days=int(frame.date.nunique()),accuracy=float(frame.correct.mean()),mae=float(frame.abs_error_pp.mean()))


def evaluate(out=OUT):
    out=Path(out)
    manifest=wf.read_json(out/'inputs_manifest.json')
    for relative,expected in manifest['sha256'].items():
        assert wf.digest(out/'inputs'/relative)==expected,relative
    start=perf_counter()
    frame=make_frames(out)
    predictions=[]
    with threadpool_limits(limits=1):
        for ticker,data in frame.groupby('ticker',sort=False):
            path=out/f'{ticker}_replay.csv'
            if path.exists():
                rows=pd.read_csv(path).to_dict('records')
            else:
                rows=replay_ticker(data.reset_index(drop=True),ticker)
                pd.DataFrame(rows).to_csv(path,index=False,encoding='utf-8-sig')
            predictions.extend(rows)
            # 保存固定加美股候選，是否較好不影響保存規則；不接生產。
            artifact=out/'isolated_models'/f'{ticker}_price_us.pkl'
            if not artifact.exists():
                artifact.parent.mkdir(exist_ok=True)
                bundle=dict(model=train(data,'price_us',ticker,'close_to_close'),features=columns('price_us',ticker),
                            trained_through=data.date.max(),purpose='research_only_not_independently_validated')
                with artifact.open('xb') as handle: pickle.dump(bundle,handle)
    pred=pd.DataFrame(predictions)
    pred.to_csv(out/'predictions.csv',index=False,encoding='utf-8-sig')
    dates=sorted(pred.date.unique())
    windows={'all':pred,'after_validation':pred[pred.phase=='exploratory_evaluation']}
    windows.update({f'last{n}':pred[pred.date.isin(dates[-n:])] for n in [120,60,20]})
    summary=[]
    for window,data in windows.items():
        for (target,method),group in data.groupby(['target','method']):
            summary.append(dict(window=window,target=target,method=method,**metrics(group)))
    by_ticker=[]
    for (ticker,target,method),group in windows['last60'].groupby(['ticker','target','method']):
        by_ticker.append(dict(ticker=ticker,target=target,method=method,**metrics(group)))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(by_ticker).to_csv(out/'by_ticker_last60.csv',index=False,encoding='utf-8-sig')
    conditional=[]
    for window,n in [('all',len(frame.date.unique())),('last60',60),('last20',20)]:
        subset=frame[frame.date.isin(sorted(frame.date.unique())[-n:])]
        for ticker,data in [('ALL',subset)]+list(subset.groupby('ticker')):
            for key in ['sox','nasdaq','sp500']:
                for label,mask in [('up',data[key+'_last']>0),('down',data[key+'_last']<0),
                                   ('big_up',data[key+'_last']>=2),('big_down',data[key+'_last']<=-2)]:
                    g=data[mask & (data.new_us_sessions>0)]
                    for target in TARGETS:
                        conditional.append(dict(window=window,ticker=ticker,index=key,bucket=label,target=target,n=len(g),
                            days=int(g.date.nunique()),follow=float((np.sign(g[key+'_last'])==np.sign(g[target])).mean()) if len(g) else None,
                            mean_tw=float(g[target].mean()) if len(g) else None,
                            correlation=float(g[key+'_last'].corr(g[target])) if len(g)>2 and g[target].std()>0 else None))
    pd.DataFrame(conditional).to_csv(out/'conditional.csv',index=False,encoding='utf-8-sig')
    audit,_=collect(out/'inputs',wiki=out/'inputs/no_wiki')
    verified=pd.DataFrame([row for row in audit if row['kind']=='point' and row['eligible']])
    latest=frame[frame.date.isin(sorted(frame.date.unique())[-8:])].copy()
    if len(verified):
        latest=latest.merge(verified[['date','ticker','predicted_pct','abs_error_pp']].rename(columns={'predicted_pct':'original_prediction','abs_error_pp':'original_error'}),on=['date','ticker'],how='left')
    latest.to_csv(out/'recent_days.csv',index=False,encoding='utf-8-sig')
    wf.write_json(out/'run_validation.json',dict(completed_at=datetime.now(wf.TZ).isoformat(),runtime_seconds=perf_counter()-start,
        rows=len(pred),aligned_rows=len(frame),first=frame.date.min(),last=frame.date.max(),
        forecast_first=dates[0],forecast_last=dates[-1],input_hashes=len(manifest['sha256']),
        no_future_training=bool((pred.train_end<pred.date).all()),source_sha256=wf.digest(__file__)))
    print(pd.DataFrame(summary).query("target=='close_to_close' and window=='last60'").to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','evaluate'])
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args()
    (prepare if args.mode=='prepare' else evaluate)(args.out)
