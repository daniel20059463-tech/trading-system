"""檢核封存回放及今日模型，產生給使用者的研究報告。不連網、不推播。"""
import json
import shutil
from datetime import datetime
from importlib.metadata import version

import numpy as np
import pandas as pd

import config as cfg
import walkforward_training as wf


def finalize():
    root = wf.HERE
    out = root / 'experiments' / wf.VERSION
    summary = wf.read_json(out / 'summary.json')
    plan = wf.read_json(out / 'plan.json')
    frame = pd.read_csv(out / 'all_predictions.csv')
    state = wf.read_json(root / 'models/daily_model_state.json')
    manifest = wf.read_json(root / state['latest'] / 'manifest.json')
    preview = wf.read_json(root / 'reports/daily_retrain_preview.json')
    for relative, expected in plan['input_sha256'].items():
        assert wf.digest(out / 'inputs' / relative) == expected, relative
    assert not frame.duplicated(['ticker', 'date', 'method']).any()
    assert (frame.train_target_end < frame.date).all()
    assert (frame['asof'] < frame.date).all()
    assert np.isfinite(frame[['actual_pct', 'predicted_pct', 'abs_error_pp']]).all().all()
    targets = frame.groupby(['ticker', 'date']).actual_pct.agg(['min', 'max'])
    assert ((targets['max'] - targets['min']).abs() < 1e-12).all()  # CSV往返最多1個浮點末位
    assert set(manifest['models']) == set(cfg.ALL_STOCKS)
    assert manifest['trained_through'] == summary['end'] == preview['records'][0]['trained_target_end']
    for ticker, artifact in manifest['models'].items():
        assert wf.digest(root / state['latest'] / f'{ticker}.pkl') == artifact['sha256']
        assert artifact['calibration_n'] == 60
    for prediction in preview['records']:
        assert prediction['trained_target_end'] < prediction['target_date']
        assert prediction['q10_pct'] <= prediction['q25_pct'] <= prediction['predicted_change_pct'] <= prediction['q75_pct'] <= prediction['q90_pct']
    # 用同份行情確認全部價格/策略對照沒有被第二階段改成別段資料。
    base = pd.read_csv(root / 'experiments/daily_v1_20260915/all_predictions.csv')
    controls = ['ridge_price', 'ridge_strategy', 'zero', 'history_median']
    keys = ['ticker', 'date', 'method']
    pd.testing.assert_frame_equal(
        frame[frame.method.isin(controls)].set_index(keys)[['actual_pct', 'predicted_pct']].sort_index(),
        base[base.method.isin(controls)].set_index(keys)[['actual_pct', 'predicted_pct']].sort_index())
    code_files = ['walkforward_training.py', 'complete_context_replay.py', 'daily_retrain.py',
                  'prediction_audit.py', 'run_daily.py', 'launch.ps1', 'agents/orchestrator.py',
                  'test_walkforward_training.py', 'test_daily_retrain.py', 'finalize_daily_retrain.py',
                  'interval_calibration_replay.py', 'rigorous_evaluation.py', 'benchmark_daily_retrain.py']
    code_hashes = {}
    for relative in code_files:
        source, destination = root / relative, out / 'code_snapshot' / relative
        if destination.exists() and wf.digest(source) != wf.digest(destination):
            raise ValueError('程式封存已有不同版本，請建立新的封存。')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        code_hashes[relative] = wf.digest(destination)
    validation = dict(generated_at=datetime.now(wf.TZ).isoformat(), rows=len(frame),
                      target_days=int(frame.date.nunique()), input_hashes_checked=len(plan['input_sha256']),
                      no_future_training_targets=True, controls_identical=True, models_checked=10,
                      code_sha256=code_hashes, packages={p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'threadpoolctl']})
    wf.write_json(out / 'validation.json', validation)

    recent = frame[(frame.method == 'ridge_full') & (frame.date >= summary['recent_start'])]
    comparison = pd.read_csv(out / 'new_vs_yesterday.csv')
    methods = [('ridge_full', '每日重訓：價格＋策略＋新聞'), ('ridge_strategy', '每日重訓：價格＋策略'),
               ('ridge_price', '每日重訓：價格'), ('yesterday_full', '前一日版本預測同一天'),
               ('frozen_full', '最初60筆訓練後固定版本'), ('zero', '預測價格不變（誤差基準）')]
    lines = [
        '# 每日重訓與逐日回放（2026-09-15）', '',
        '已完成十檔股票從最早可用行情開始的逐日回放，並用截至9/15收盤的資料訓練、保存今天的十個模型。'
        '最近兩個月的完整新聞候選方向命中率50.2%，漲跌幅MAE為4.3065百分點；尚未優於價格不變基準的4.0774。'
        '每日重訓流程已接入，但目前不能聲稱找到可以持續的高精準預測。', '',
        '## 資料從哪一天開始', '',
        '- 行情：2023-12-13～2026-09-15，每檔668列。第一筆起即參與滾動特徵；20筆歷史不足時不虛構指標。',
        '- 第一個可訓練目標是2024-01-12；先累積60筆已知答案，第一個可評分預測日為2024-04-19。只憑一天無法訓練這類模型，因此暖身期明列，沒有冒充第二天已能精準預測。',
        '- 每日只用該日之前已收盤的目標重訓，再预测下一個資料中實際存在的交易日；預測記錄之後才加入答案和區間校準。沒有把休市日當成有結果的交易日。',
        '- 正式回放587個交易日、十檔共5,870組股票／日期，八種方法合計46,960列。近期採2026-07-15～9/15，共45日、450組股票／日期。',
        '- 今天各檔模型用647筆有效已完成目標，訓練截止9/15；近60個逐日誤差用於區間。完整價格歷史都保留，舊樣本權重每126個交易日減半。', '',
        '## 新聞、策略如何納入', '',
        '固定Ridge模型同時使用價格動能、波動、成交量、大盤、延遲一個台股交易日的匯率與海外同業；'
        '策略特徵是均線趨勢、均值回歸、前20日突破和量價訊號。新聞加入情緒、買賣建議及12個固定詞：'
        '漲價、缺貨、訂單、營收、財報、外資、買超、賣超、庫存、需求、AI、利空。這是新增的每日統計模型，原LSTM／Transformer沒有在這次被當成歷史逐日重訓結果。', '',
        '63份盤後新聞只涵蓋五檔；另補58份能核對執行時間的盤前新聞，合計895筆事件，涵蓋十檔。'
        '舊盤前內容保守延至檔名日期結束，且不早於日誌的最後執行時間；盤後內容只在記錄時間以後使用。'
        '新聞有效期72小時，缺資料用旗標表示。6/2、6/16兩份盤前檔缺日誌而排除。', '',
        f'近期450組有{int(recent.news_available.sum())}組可用新聞。全回放共645組／68個日期可用新聞；更早期間不能捏造當時消息。'
        '不使用今天累積算出的關鍵字勝率、不用沒有可靠原始發布時間的新聞日期快取。', '',
        '歷史完整LLM策略未有相同標準的不可變時間留底，因此本次歷史比較使用可重建的價格策略和新聞買賣建議。'
        '從下一次實際盤前開始，完整策略的買／賣／持有及新聞會先保存在context_snapshots，再提供給候選預測；這些新策略特徵需要累積已完成結果後才有可學習的效果。', '',
        '## 同股票、同日期的比較', '',
        '| 最近兩月方法 | 方向命中 | 漲跌幅MAE（百分點） | 80%區間實際涵蓋 | 80%區間平均寬度（百分點） |',
        '|---|---:|---:|---:|---:|']
    for key, label in methods:
        item = summary['recent'][key]
        direction = '—' if key == 'zero' else f"{item['direction_accuracy']:.1%}"
        lines.append(f"| {label} | {direction} | {item['mae_pp']:.4f} | {item['coverage80']:.1%} | {item['width80_pp']:.2f} |")
    lines += ['', '方向按漲／跌／不變三類判定；預測不變不冒充猜漲跌的50%基準。MAE為漲跌幅差的絕對值，'
              '例如預測+1%、實際−2%，誤差是3百分點。最初固定版本是本次Ridge對照，並非目前正式LSTM模型。', '',
              '加上價格策略後MAE僅減少0.0011百分點；再加新聞，方向由47.6%升至50.2%，但MAE增加0.1925百分點。'
              '在所有有新聞的645組配對樣本中，新聞也使MAE增加0.5750百分點。這表示目前的新聞表示法和權重沒有改善幅度預測，不能因為有新聞就強制採信。', '',
              '額外樹模型沿用第一階段五檔盤後新聞資料，近期方向47.3%、MAE4.3561，僅作不完整新聞對照；'
              '第二階段只補齊新聞、重算相關Ridge版本，沒有依結果更改參數或挑出最佳模型。', '',
              '80%區間是用先前誤差校準的目標涵蓋率，不是方向成功機率。完整新聞版近期涵蓋89.1%，平均寬17.88百分點，'
              '整段历史涵蓋79.95%；不能把區間很寬造成的高涵蓋率宣稱為精準。', '',
              '## 哪幾天相對好', '',
              '以下都是目標日收盤後才知道的歷史回放成績，每天十檔。回放前設定的「方向≥70%、MAE≤1百分點且優於不變」三條件，'
              '最近兩月沒有一天同時通過。以下是方向較好且誤差優於不變的日期，幅度仍有誤差。', '',
              '| 預測目標日 | 方向命中 | MAE（百分點） | 不變MAE | 當日有可用新聞的股票數 |',
              '|---|---:|---:|---:|---:|']
    good = summary['best_recent_diagnostic'][:4]
    for day in good:
        count = int(recent[recent.date == day['date']].news_available.sum())
        lines.append(f"| {day['date']} | {day['accuracy']:.0%} | {day['mae']:.3f} | {day['mae_zero']:.3f} | {count}/10 |")
    lines += ['', '8/4、8/11當時沒有符合時效的新聞，所以這兩天的成功不能歸因於當日新聞；模型仍可能保有先前學到的新聞係數。', '',
              '| 上一天相對好的日期 | 隔日 | 沿用該模型：方向／MAE | 隔日重新訓練：方向／MAE |',
              '|---|---|---:|---:|']
    carry = []
    for day in good:
        next_day = min(frame[frame.date > day['date']].date)
        old = frame[(frame.date == next_day) & (frame.method == 'yesterday_full')]
        new = frame[(frame.date == next_day) & (frame.method == 'ridge_full')]
        carry.append(dict(good_date=day['date'], next_date=next_day, old_accuracy=float(old.direction_correct.mean()),
                          old_mae=float(old.abs_error_pp.mean()), new_accuracy=float(new.direction_correct.mean()), new_mae=float(new.abs_error_pp.mean())))
        lines.append(f"| {day['date']} | {next_day} | {old.direction_correct.mean():.0%}／{old.abs_error_pp.mean():.3f} | {new.direction_correct.mean():.0%}／{new.abs_error_pp.mean():.3f} |")
    wf.write_json(out / 'good_day_continuation.json', carry)
    lines += ['', '因此「昨天準，就一直沿用」目前沒有穩定支持。7/31也出現方向30%、MAE10.522；9/4為10%、MAE6.545；失敗日同樣保留。', '',
              f"新舊版本對同一天預測差≤0.25百分點的比例，全期為{summary['similarity_rate']:.1%}、近期為{(recent.vs_yesterday_prediction_pp <= .25).mean():.1%}。"
              f"以先前20日誤差决定翌日選新或舊版本，近期MAE為{comparison[comparison.date >= summary['recent_start']].selected_mae.mean():.4f}，也未改善。"
              '相似代表輸出穩定，不代表預測正確。', '',
              '整段587日有28日通過三條件，最新一次是2026-02-04（80%、0.995百分點）；它們都早於這批新聞留存，不能作為新聞模型有效的證据。日期列於附錄。', '',
              '## 每天接下來怎麼執行', '',
              '1. 盤後刷新行情、以不可覆寫盤前原值評分，再重訓十檔模型；每版保留截止日、輸入及模型雜湊、樣本數。相同輸入再次執行會驗證後跳過。',
              '2. 次日盤前可補做訓練，並在新聞和策略完成後保存當時上下文；新模型、前一日版本及正式預測做可核對的比較。',
              '3. 新模型先與正式模型並排留底；超過開盤不可補造盤前預測。盤後才把今日答案加進下個模型。預測失敗一樣留存，不刪失敗樣本。',
              '4. 提出替換建議須在最近最多40個實際觀察日中累積至少20日、200個配對樣本，方向≥52%且不差於舊模型；MAE須比舊模型及不變基準至少改善2%；至少100組80%區間，實際涵蓋75%～95%。通過也先提供可審查的建議，不以相似或單一好日子自動採用。', '',
              '目前新版本實際盤前配對樣本是0，尚未自動換用。這次完成的是歷史回放及今天訓練，不能把回算的5,870筆當成原先發布的實盤證據。'
              '既有排程執行pre／post時會帶入新流程；電腦需在排程時開機、可連網，新聞與策略也須在9點前完成。', '',
              '## 今天已保存的模型與預覽', '',
              f"版本 `{wf.VERSION}`；模型目錄 `{state['latest']}`；十檔各647筆訓練、60筆誤差校準。",
              f"以下是{preview['generated_at']}計算的9/16候選預覽；明日盤前新消息可能改變輸出，這份預覽不計為明日實際盤前留底。", '',
              '| 股票 | 次日預測漲跌幅 | 80%區間 |', '|---|---:|---:|']
    for prediction in preview['records']:
        lines.append(f"| {prediction['name']}（{prediction['ticker']}） | {prediction['predicted_change_pct']:+.2f}% | {prediction['q10_pct']:+.2f}%～{prediction['q90_pct']:+.2f}% |")
    lines += ['', '## 後續討論的依據', '',
              '優先改善幅度誤差與新聞對預測的增量效果：目前方向略升，幅度反而變差，兩項應分開檢驗。'
              '下一個版本可比較新聞的時間衰減、事件類型及強度限制，但要先固定規則，再拿未看過的未來日期驗證，不能反覆調到這45天漂亮。', '',
              '也不能直接挑低波動日來宣稱有效：近期盤前20日波動低於3%的只有23筆，方向26.1%、MAE1.982；'
              '高波動427筆方向51.5%、MAE4.432。需要按預測前已知條件及同日基準比較，而非事後選出上漲日或成功日。', '',
              '## 盤前留底與區間補充', '',
              '9/11、9/14、9/15沒有新增可驗證盤前留底；原有9/8、9/9每種預測20筆中，18筆通過基準價格核對。排除的是德微（3675.TWO）兩天：當時基準價282及274與現有還原行情不一致；不是按成績好壞選樣。此前探索影子和本次每日重訓候選皆無新增可信實盤樣本。', '',
              '18筆正式80%區間實際涵蓋94.4%，平均寬13.337百分點。另已固定60日歷史分位數基準，與新候選及正式留底逐股、逐日配對，完整涵蓋／寬度／區間分數見：', '',
              '[區間校準補充報告](../trading-system/experiments/daily_v2_20260915/interval_diagnostic/report.md)。', '',
              '區間補充的60日歷史分位數方案在看過主要回放結果後才提出，並在計算這個新增基準之前保存plan.json；屬第二階段探索。'
              '最初每日重訓方案與第二階段完整新聞方案各在對應計算前留存計畫，未調參；最終程式雜湊和validation.json則是結果完成後的封存，不冒充事前登記。', '',
              '## 可重現性與限制', '',
              '行情、新聞、日誌及輸入SHA256已封存在實驗目錄；逐筆CSV含訓練截止、樣本數、預測、實際、誤差、新聞可用時間及區間。'
              'validation.json記錄輸入核對、模型雜湊、套件版本與程式封存；code_snapshot保存實驗及每日流程程式。', '',
              '行情為今天取得的還原價格，不是當時每天的原始價格版本。舊新聞JSON與日誌不是原本不可變的新聞發布證明，'
              '其時間限制屬保守重建；既有LLM摘要也未經逐篇原文驗證。歷史策略版本未完整留存。這些限制意味本報告是探索性歷史回放，不能稱作完全獨立的實盤或樣本外成績。', '',
              '## 附錄：全期通過三條件的日期', '',
              '、'.join(day['date'] for day in summary['good_days_all']), '',
              '檔案：`experiments/daily_v2_20260915/all_predictions.csv`、`daily_quality.csv`、`recent_by_ticker.csv`、'
              '`new_vs_yesterday.csv`、`summary.json`、`good_day_continuation.json`。', '',
              '相關筆記：[[10 預測稽核與每日驗證]]、[[11 模型改善實驗 2026-09-10]]。']
    report = root.parent / '半導體預測系統' / '12 每日重訓與逐日回放 2026-09-15.md'
    report.write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(report=str(report), validation=validation), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    finalize()
