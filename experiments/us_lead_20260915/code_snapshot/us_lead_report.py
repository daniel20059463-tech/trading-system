"""核對固定美股領先實驗並產生繁體中文報告；不再訓練或選模。"""
import json
import shutil
from datetime import datetime
from importlib.metadata import version

import numpy as np
import pandas as pd

import us_lead_study as study
import walkforward_training as wf


def pct(value): return f'{value:.1%}'


def main():
    out=study.OUT
    frame=pd.read_csv(out/'aligned.csv')
    pred=pd.read_csv(out/'predictions.csv')
    summary=pd.read_csv(out/'summary.csv')
    recent=pd.read_csv(out/'recent_days.csv')
    conditional=pd.read_csv(out/'conditional.csv')
    run=wf.read_json(out/'run_validation.json')
    inputs=wf.read_json(out/'inputs_manifest.json')
    assert wf.digest(study.__file__)==run['source_sha256']
    for path,expected in inputs['sha256'].items():
        assert wf.digest(out/'inputs'/path)==expected
    assert not pred.duplicated(['date','ticker','method','target']).any()
    assert (pred.train_end<pred.date).all()
    assert (pd.to_datetime(frame.us_available_taipei,utc=True)<pd.to_datetime(frame.cutoff_taipei,utc=True)).all()
    assert (pred.groupby(['date','ticker','target']).actual_pct.nunique()==1).all()
    assert np.isfinite(pred[['predicted_pct','actual_pct','abs_error_pp']]).all().all()
    assert (pred.groupby(['date','ticker','target']).method.nunique()==len(study.METHODS)).all()
    old=pd.read_csv(study.HERE/'experiments/daily_v2_20260915/all_predictions.csv',low_memory=False)
    original=old[old.method=='ridge_price'].set_index(['date','ticker']).predicted_pct.sort_index()
    now=pred[(pred.target=='close_to_close')&(pred.method=='price')].set_index(['date','ticker']).predicted_pct.sort_index()
    np.testing.assert_allclose(original.values,now.values,atol=1e-10,rtol=1e-10)
    artifacts={path.name:wf.digest(path) for path in (out/'isolated_models').glob('*.pkl')}
    wf.write_json(out/'candidate_manifest.json',dict(protocol='us_lead_v1',method='price_us',
        trained_through='2026-09-15',training_n_per_stock=647,files=artifacts,
        inference='pipeline prediction times current prior20-day return standard deviation (minimum0.3pp)',
        purpose='All ten fixed candidates retained; research-only, not promoted or attached to daily runs'))
    archive=out/'code_snapshot'
    archive.mkdir(exist_ok=True)
    code={}
    for name in ['us_lead_study.py','us_lead_report.py','test_us_lead_study.py','walkforward_training.py','prediction_audit.py','config.py']:
        source=study.HERE/name
        destination=archive/name
        if destination.exists(): assert wf.digest(destination)==wf.digest(source)
        else: shutil.copy2(source,destination)
        code[name]=wf.digest(source)
    wf.write_json(out/'final_validation.json',dict(verified_at=datetime.now(wf.TZ).isoformat(),
        rows=len(pred),paired_methods=True,price_control_matches_previous_replay=True,
        no_future_training_or_us_availability=True,input_hashes=len(inputs['sha256']),
        files=code,packages={p:version(p) for p in ['yfinance','pandas','numpy','scikit-learn','pandas_market_calendars','exchange_calendars']}))
    names={'price':'原價格候選','price_us':'價格＋美股','us_only':'只用美股','sox_copy':'直接沿用SOX漲跌幅',
           'price_us_peers':'價格＋美股＋功率同業','zero':'預測不變'}
    tnames={'close_to_close':'前收→收盤','gap':'前收→開盤','intraday':'開盤→收盤'}
    def get(window,target,method):
        return summary[(summary.window==window)&(summary.target==target)&(summary.method==method)].iloc[0]
    base,new=get('last60','close_to_close','price'),get('last60','close_to_close','price_us')
    dates=sorted(pred.date.unique())
    lines=['# 美股領先台股：時間對齊與逐日預測實驗（資料截至2026-09-15）','',
        f'有值得保留的美股領先訊號。最近60個台股交易日（{dates[-60]}～{dates[-1]}），'
        f'加入美股後，十檔個股全日方向命中率由{base.accuracy:.1%}升至{new.accuracy:.1%}，'
        f'漲跌幅MAE由{base.mae:.4f}降至{new.mae:.4f}百分點，降幅{1-new.mae/base.mae:.1%}。'
        '改善主要在開盤跳空；盤中走勢沒有改善。這是已探索歷史的回放，尚非獨立實盤驗證。','',
        '## 為什麼重做，而不是沿用舊結論','',
        '原有美台觀察用日期asof，只找「日期更早」的美股，沒有以交易所收盤時間判定是否已知。'
        'global_snapshot的預期交易日只排除週末：9/8快照把9/4美股判成落後，因為誤把9/7當交易日；'
        'Nasdaq官方公告9/7是Labor Day休市。這一日9/4其實已是最新時段，但它在9/7台股開盤前也已被知道，因此9/8沒有新的美股隔夜時段。','',
        '舊SOX神經網路實驗先把SOX對齊台股T日，再把特徵放进截至T−1的視窗去預測T；'
        '因而少用最新一晚的訊號。舊結果47.9%→48.4%只適用於那個實作，不能說美股訊號已被證明無效。'
        '舊觀察中把相關減弱直接歸因於外資主導，也沒有足夠因果證據。上述正式程式本輪均未更動。','',
        '## 資料與時間規則','',
        '- 原系統十檔全部保留：國巨、華新科、禾伸堂、立隆電、臺慶科、台半、強茂、德微、朋程、富鼎。',
        '- 台股封存668列／檔，2023-12-13～2026-09-15；20日價格特徵暖身後647列／檔，首次可訓練目標2024-01-12。',
        '- 美股使用SOX、NASDAQ、S&P500；ON與Vishay僅作原系統已列為功率同業的固定次要比較，不事後挑最相關個股。每個美股序列封存719列。',
        '- 台股預測截止為當日08:30（Asia/Taipei）。NYSE／NASDAQ日曆提供每個美股時段的UTC收盤，涵蓋夏令時間、休市及提早收盤。取收盤後60分鐘已過、且早於截止的最新時段。',
        '- 正常美股收盤換算台灣時間：夏令04:00、冬令05:00；研究保守可得時間分別05:00、06:00。這個加60分鐘是假設，不是證明Yahoo當時一定已送達。',
        '- 同時保存最近單一美股時段變化、前一次台股收盤後的累積美股變化、新美股時段數與經過小時數，處理台股長假與美股休市。',
        '- 台股日期只取封存OHLC實際存在的交易觀測，不把週一至週五一律當交易日。對應最新美股時段或其前一期價格缺失就排除，不以前一列或0補值。',
        '- 這份輸入沒有額外美股對齊缺失；647個台股日期中22日沒有新美股時段。下載檔中的美股9/15資料在台股9/15開盤前尚未完成，未用於該日預測。','',
        '## 固定的模型比較','',
        '先保存plan.json，再抓取／封存資料與執行。所有模型每天只用更早的已完成答案，最少60筆，Ridge正則化固定100，'
        '訓練權重每126個交易日減半；特徵標準化只配適過去資料，目標以預測前20日波動正規化。沒有新聞、沒有新調參，也沒有依結果刪除股票。','',
        '原價格候選沿用上一輪相同的10個欄位（其中已有保守延遲的同業／匯率）；本輪其全日預測已與上一輪逐筆核對一致。'
        '「價格＋美股」新增三指數各自的最近一時段及台股休市間累積變動，再加時段數與新鮮度，共8欄。'
        '功率同業版只對五檔功率股再加ON／VSH；被動元件股維持相同模型。','',
        '回放2024-04-19～2026-09-15共587日、5,870组股票／日期。首次100個預測日（4/19～2024/9/11）標為時間順序驗證段，'
        '之後487日為探索評估段；固定參數並未在驗證段搜尋。六方法、三個目標合計105,660筆預測。所有截至9/15的歷史都已被本專案查看，任何區段均不宣稱全新的未見測試。','',
        '| 最近60日全日預測 | 方向命中率 | MAE（百分點） |','|---|---:|---:|']
    for method in study.METHODS:
        row=get('last60','close_to_close',method)
        lines.append(f"| {names[method]} | {'—' if method=='zero' else pct(row.accuracy)} | {row.mae:.4f} |")
    lines += ['','直接SOX基準把最近時段漲跌幅套給台股；沒有新美股時段時預測0。這不是可下單的策略。'
              '「原價格候選」是同一套Ridge對照，並非正式LSTM／Transformer實盤模型。正式模型沒有被替換。','',
              '| 不同全日窗口 | 股票／日期數 | 原價格方向→加美股 | 原價格MAE→加美股 | 不變MAE |',
              '|---|---:|---:|---:|---:|']
    for window,label in [('all','全部587日'),('after_validation','驗證後487日'),('last120','最近120日'),('last60','最近60日'),('last20','最近20日')]:
        a,b,z=[get(window,'close_to_close',method) for method in ['price','price_us','zero']]
        lines.append(f'| {label} | {a.n} | {a.accuracy:.1%}→{b.accuracy:.1%} | {a.mae:.4f}→{b.mae:.4f} | {z.mae:.4f} |')
    lines += ['','## 訊號主要反映在哪一段','',
              '| 最近60日預測目標 | 原價格方向→加美股 | 原價格MAE→加美股 |','|---|---:|---:|']
    for target in study.TARGETS:
        a,b=[get('last60',target,method) for method in ['price','price_us']]
        lines.append(f'| {tnames[target]} | {a.accuracy:.1%}→{b.accuracy:.1%} | {a.mae:.4f}→{b.mae:.4f} |')
    lines += ['','三個目標都在08:30預測，沒有把當日開盤價當作輸入。跳空=開盤/前收−1；盤中=收盤/開盤−1；'
              '全日=收盤/前收−1。全日與兩段的關係是(1+跳空)×(1+盤中)−1，不能把百分比直接相加。','',
              '最近60日、只看有新美股時段的資料，SOX與跳空的相關約0.591，與盤中約−0.026，與全日約0.325。'
              '這與模型結果一致：資訊多在開盤價反映。相關不等於因果；也不能在看到隔夜美股後用昨日台股收盤價成交，吃到已發生的跳空。'
              '本輪沒有估算交易成本或策略獲利。','',
              '## 哪些個股較明顯','',
              '同樣最近60日，每檔60筆；全部列出，避免只列成功股票。','',
              '| 股票 | 原價格方向 | 加美股方向 | 原價格MAE | 加美股MAE | 不變MAE |','|---|---:|---:|---:|---:|---:|']
    table=pd.read_csv(out/'by_ticker_last60.csv')
    for ticker,name in study.cfg.ALL_STOCKS.items():
        g=table[(table.ticker==ticker)&(table.target=='close_to_close')].set_index('method')
        a,b,z=g.loc['price'],g.loc['price_us'],g.loc['zero']
        lines.append(f'| {name}（{ticker}） | {a.accuracy:.1%} | {b.accuracy:.1%} | {a.mae:.3f} | {b.mae:.3f} | {z.mae:.3f} |')
    lines += ['','國巨的方向改善較明顯；朋程、臺慶科、台半也有較大方向增幅。這是本次觀察結果，不是按結果改成只預測這幾檔。'
              '額外ON／VSH沒有比三指數版本更好，完整次要方法成績仍保留於summary.csv。','',
              '## 美股漲跌與大波動的條件觀察','',
              '只統計前一次台股收盤之後確有新美股時段的日子；美股大漲／大跌門檻固定為±2%。'
              '同一天十檔並不獨立，因此同時列日期數。以下「跟隨」指同方向，持平不算跟隨。','',
              '| 最近60日SOX條件 | 日期數／股票筆數 | 全日同向率 | 跳空同向率 | 盤中同向率 | 全日平均報酬 |',
              '|---|---:|---:|---:|---:|---:|']
    for label,display in [('up','上漲'),('down','下跌'),('big_up','大漲≥2%'),('big_down','大跌≤−2%')]:
        g=conditional[(conditional.window=='last60')&(conditional.ticker=='ALL')&(conditional['index']=='sox')&(conditional.bucket==label)].set_index('target')
        a=g.loc['close_to_close']
        lines.append(f"| {display} | {int(a.days)}／{int(a.n)} | {a.follow:.1%} | {g.loc['gap','follow']:.1%} | {g.loc['intraday','follow']:.1%} | {a.mean_tw:+.2f}% |")
    lines += ['','這些桶互有重疊（大漲也屬上漲），不能相加當作更多樣本。三個指數、所有股票與全期／60日／20日的條件表皆在conditional.csv。','',
              '## 最近八個有效交易日逐日對照','',
              '台股數值為原十檔等權平均，不是加權指數；各股完整80列在recent_days.csv。這次未另更換研究標的或拿大盤替代十檔。','',
              '| 台股日 | 美股交易日 | 台灣可得時間（假設） | SOX | Nasdaq | S&P500 | 新時段數 | 台股跳空／盤中／全日 |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    for day,g in recent.groupby('date'):
        row=g.iloc[0]
        available=pd.Timestamp(row.us_available_taipei).strftime('%m/%d %H:%M')
        lines.append(f'| {day} | {row.us_session} | {available} | {row.sox_last:+.2f}% | {row.nasdaq_last:+.2f}% | {row.sp500_last:+.2f}% | {int(row.new_us_sessions)} | {g.gap.mean():+.2f}%／{g.intraday.mean():+.2f}%／{g.close_to_close.mean():+.2f}% |')
    lines += ['','9/8沿用9/4是美股休市造成，不能把那次+3.37%當成9/7晚上新發生的訊號。9/14美股雖漲，十檔台股平均卻低開；'
              '9/15 SOX大跌，但平均開盤仍略高，反例都有保留。','',
              '### 原模型可信留底的對照','',
              '近八日只有9/8、9/9存在原始盤前留底且可核對基準價格；德微兩日因價格修訂不符排除，所以共18筆。'
              '其餘日期或資料缺失不回填成原模型績效。候選是歷史回算，不能與這18筆混稱為已發布的實盤。','',
              '| 日／股票 | 當日實際漲跌 | 原盤前預測 | 原誤差（百分點） |','|---|---:|---:|---:|']
    for row in recent[recent.original_prediction.notna()].sort_values(['date','ticker']).itertuples():
        lines.append(f'| {row.date}／{study.cfg.ALL_STOCKS[row.ticker]} | {row.close_to_close:+.2f}% | {row.original_prediction:+.2f}% | {row.original_error:.3f} |')
    lines += ['','## 限制、成果保存與下一步','',
              '- 原行情是目前下載的還原OHLC，不是當時每天不可變的資料版本；美股歷史下載也可能有後來修订。收盤+60分只保守安排時間，不能保證資料商當時沒有延遲。',
              '- 同日股票相關、窗口互相重疊；600筆不是600個獨立交易日。這輪沒有宣稱統計顯著，也沒有把近期改善直接外推成未來命中率。',
              '- 存在原股池的選擇／存活偏誤。此結論僅對這十檔及這段資料；美股領先不代表它單獨造成台股變化。',
              '- 固定價格＋美股的十個候選均已保存於isolated_models，沒有因結果挑掉差股，沒有接進每日正式或影子流程。原每日重訓、留底、稽核與推播行為維持。',
              '- 後續較值得驗證的是盤前跳空／當日收盤的獨立美股候選；盤中方向需另找訊號。先用未看過日期的真實美股可得時間留底，再與同一次正式預測配對。','',
              '## 重現與驗證','',
              f"運算記錄：{run['completed_at']}完成，耗時{run['runtime_seconds']:.1f}秒。六個新增測試覆蓋美國勞動節、夏令切換、提早收盤、缺資料、台股长假及未來資料隔離。",
              'final_validation.json另核對21個封存輸入、105,660筆日期/配對/有限數值；原價格候選與前輪逐筆一致。'
              'code_snapshot、套件版本和candidate_manifest.json供追溯。新依賴位於.research_deps/us_alignment，沒有安裝到正式環境或改排程。','',
              '離線查核／接續：`python us_lead_study.py evaluate --out experiments/us_lead_20260915`（已有股票結果會讀取快取，不連網）。'
              '若要從頭重新擬合，將plan.json、inputs_manifest.json及inputs資料夾複製到新的實驗目錄，再將--out指向該目錄，便不會沿用已完成的預測檔。', '',
              '主要檔案：aligned.csv（時間與特徵）、predictions.csv（所有預測）、summary.csv（固定窗口）、by_ticker_last60.csv、'
              'conditional.csv、recent_days.csv、alignment_exclusions.json、isolated_models。','',
              '## 來源','',
              '- [NYSE交易時段與休市](https://www.nyse.com/trade/hours-calendars)：美東正常及提早收盤規則。',
              '- [Nasdaq 2026勞動節公告](https://www.nasdaqtrader.com/TraderNews.aspx?id=ETS2026-47)：9/7休市。',
              '- [TWSE交易制度](https://www.twse.com.tw/en/products/system/trading.html)：一般交易09:00～13:30。',
              '- [交易日曆套件文件](https://pandas-market-calendars.readthedocs.io/en/latest/usage.html)：歷史時段與UTC時間日曆。',
              '- [Yahoo Finance](https://finance.yahoo.com/)：透過既有yfinance取得五個美股序列，原始下載與SHA256均封存；並非交易所官方逐筆資料。','',
              '相關筆記：[[12 每日重訓與逐日回放 2026-09-15]]。']
    report=study.HERE.parent/'半導體預測系統/13 美股領先台股逐日驗證 2026-09-16.md'
    report.write_text('\n'.join(lines),encoding='utf-8')
    print(report)


if __name__=='__main__': main()
