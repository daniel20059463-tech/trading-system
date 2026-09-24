"""隔離目錄量測新增每日訓練與候選推論；不發布預測。"""
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from time import perf_counter

import daily_retrain as dr


def run():
    source = dr.HERE
    now = datetime.now(dr.TZ)
    target = '2026-09-16'
    with tempfile.TemporaryDirectory(prefix='hiu_benchmark_') as directory:
        root = Path(directory)
        # 只複製本次工作相關輸入；計時不含檔案準備，包含完整特徵建構及模型保存。
        for folder, pattern in [('data/raw','*.csv'),('news','*_market.json'),('logs','*_pre.log'),
                                ('context_snapshots','*.json'),('experiments/'+dr.wf.VERSION,'*_replay.csv')]:
            for path in (source/folder).glob(pattern):
                destination = root/folder/path.name
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,destination)
        start=perf_counter()
        manifest=dr.train_latest(root,now)
        trained=perf_counter()
        values=[dr.candidate_for(ticker,target,root,now) for ticker in dr.cfg.ALL_STOCKS]
        predicted=perf_counter()
        result=dict(measured_at=now.isoformat(),training_seconds=trained-start,
                    inference_seconds=predicted-trained,total_seconds=predicted-start,
                    models=len(manifest['models']),predictions=len(values),training_n=647,
                    training_scope='fresh ten models including features, calibration reads, hashing and artifact writes',
                    excluded='price downloads, news/LLM/strategy calls, original production models; no guarantee of full pre-market completion')
        dr.wf.write_json(source/'reports/daily_retrain_timing.json',result)
        print(result)


if __name__=='__main__':
    run()
