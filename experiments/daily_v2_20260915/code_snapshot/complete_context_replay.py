"""補齊五檔缺失新聞後的回放。固定原參數，價格/策略對照沿用同份封存行情。"""
import json
import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

import config as cfg
import walkforward_training as wf


def run(resume=False):
    base = wf.HERE / "experiments" / "daily_v1_20260915"
    out = wf.HERE / "experiments" / wf.VERSION
    if (out / "plan.json").exists() and not resume:
        raise FileExistsError("完整新聞實驗已存在；--resume 只接續相同封存輸入。")
    if resume:
        plan = wf.read_json(out / "plan.json")
        for relative, expected in plan["input_sha256"].items():
            if wf.digest(out / "inputs" / relative) != expected:
                raise ValueError("封存資料已改變，不能接續。")
        return finish(base, out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = out / "inputs"
    sources = list((base / "inputs" / "data" / "raw").glob("*.csv"))
    for path in sources:
        destination = inputs / "data" / "raw" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    files = (list((wf.HERE / "news").glob("*_market.json")) + list((wf.HERE / "logs").glob("*_pre.log")) +
             list((wf.HERE / "context_snapshots").glob("*.json")))
    for path in files:
        destination = inputs / path.relative_to(wf.HERE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    events, diagnostics = wf.context_events(inputs)
    wf.write_json(out / "plan.json", {**wf.read_json(base / "plan.json"), "version": wf.VERSION,
                  "locked_at": datetime.now(wf.TZ).isoformat(), "reason": "Post archive covers only five tickers; supplement time-checked pre archives with conservative next-day availability.",
                  "parameters_changed": False, "candidate_selected_from_results": False,
                  "new_computation": ["ridge_full", "frozen_full", "yesterday_full"],
                  "inherited_price_controls": "daily_v1_20260915 identical frozen price inputs",
                  "tree_limitation": "tree_full_post_only retains original five-stock post context",
                  "context_diagnostics": diagnostics,
                  "input_sha256": {str(p.relative_to(inputs)): wf.digest(p) for p in inputs.rglob("*") if p.is_file()}}, exclusive=True)
    return finish(base, out)


def finish(base, out):
    inputs = out / "inputs"
    events = wf.context_events(inputs)[0]
    all_rows = []
    with threadpool_limits(limits=1):
        for ticker in cfg.ALL_STOCKS:
            completed = out / f"{ticker}_replay.csv"
            if completed.exists():
                all_rows.append(pd.read_csv(completed))
                continue
            if not (base / f"{ticker}_replay.csv").exists():
                raise FileNotFoundError(f"第一階段尚在執行 {ticker}；已完成部分保留，稍後 --resume。")
            frame = wf.frame_for(ticker, inputs, events)
            rows = pd.DataFrame(wf.replay_ticker(frame, ["ridge_full", "frozen_full", "yesterday_full"]))
            inherited = pd.read_csv(base / f"{ticker}_replay.csv")
            inherited = inherited[~inherited["method"].isin(["ridge_full", "frozen_full", "yesterday_full"])].copy()
            inherited["method"] = inherited["method"].replace({"tree_full": "tree_full_post_only"})
            metadata = {str(r['date'].date()): r for r in frame.to_dict('records')}
            inherited['news_available'] = [int(metadata[d]['news_available']) for d in inherited['date']]
            inherited['context_available_at'] = [metadata[d]['context_available_at'] for d in inherited['date']]
            combined = pd.concat([rows, inherited], ignore_index=True)
            combined.to_csv(out / f"{ticker}_replay.csv", index=False, encoding="utf-8-sig")
            all_rows.append(combined)
    frame = pd.concat(all_rows, ignore_index=True)
    frame.to_csv(out / "all_predictions.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(base / "warmup.json", out / "warmup.json")
    result = wf.report(frame, out)
    print(json.dumps({"recent": result['recent'], "good_days_recent": result['good_days_recent']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.resume)
