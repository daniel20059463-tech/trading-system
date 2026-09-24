"""每日低成本重訓、不可覆寫版本與盤前並排預測。不連網、不推播。"""
import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import config as cfg
import walkforward_training as wf
from adoption_state import REVIEW_GATE
from prediction_audit import TZ, read_json, require_pre_market, score, valid_daily_candidate

HERE = Path(__file__).resolve().parent


def snapshot_context(news, strategies, root=HERE, now=None):
    """在實際完成分析時記錄，絕不回填過去時間。"""
    now = require_pre_market(now)
    strategy_map = {p.get("ticker", "").split(".")[0]: p for p in strategies}
    records = []
    for ticker in cfg.ALL_STOCKS:
        code = ticker.split(".")[0]
        info = news.get(code, news.get(ticker, {}))
        advice = strategy_map.get(code, {})
        records.append({"ticker": ticker,
                        "news": {key: info[key] for key in ("sentiment_score", "strategy_signal", "strategy_reason",
                                  "key_catalysts", "risk_factors", "news_count") if key in info},
                        "strategy": {key: advice[key] for key in ("action", "position_size", "reason") if key in advice}})
    path = Path(root) / "context_snapshots" / f"{now.date().isoformat()}_pre.json"
    wf.write_json(path, {"generated_at": now.isoformat(), "schema_version": 1, "records": records}, exclusive=True)
    return path


def live_pairs(root=HERE):
    """只比較同一次盤前保存、同股同目標日的新舊預測。"""
    root = Path(root)
    pairs = []
    for path in sorted((root / "predictions").glob("*_point.json")):
        payload = read_json(path)
        day = payload["target_date"]
        for point in payload["records"]:
            candidate = point.get("daily_retrain")
            if not candidate or candidate.get("protocol") != wf.VERSION or not valid_daily_candidate(point, day):
                continue
            candidate = dict(candidate)
            if candidate.get("trained_target_end", day) >= day:
                continue
            current = score(candidate, day, root, trusted=True)
            old = score(point.get("production_before_retrain", point), day, root, trusted=True)
            if not (current["eligible"] and old["eligible"]):
                continue
            pairs.append({"date": day, "ticker": point["ticker"],
                          "candidate_protocol": candidate.get("protocol"),
                          "candidate_artifact_sha256": candidate.get("artifact_sha256"),
                          "candidate_model_path": candidate.get("model_path"),
                          "adoption_id": point.get("adoption", {}).get("promotion_id"),
                          "new_mae": current["abs_error_pp"], "old_mae": old["abs_error_pp"],
                          "zero_mae": current["baseline_abs_error_pp"],
                          "new_correct": current["direction_correct"], "old_correct": old["direction_correct"],
                          "new_covered80": current["covered80"], "new_width80": current["width80_pp"],
                          "prior_close": current["base_close"], "actual_close": current["actual_close"],
                          "actual_pct": current["actual_pct"], "predicted_pct": current["predicted_pct"]})
    return pairs


def promotion_decision(pairs):
    # 股票同日相關，不把200筆當200個獨立交易日。
    if not pairs:
        return {"approved": False, "n": 0, "days": 0, "reason": "尚無新版本的盤前並排實盤樣本"}
    frame = pd.DataFrame(pairs)
    days = sorted(frame["date"].unique())[-REVIEW_GATE["lookback_days"]:]
    frame = frame[frame["date"].isin(days)]
    interval = frame[frame["new_covered80"].notna()]
    n, distinct = len(frame), frame["date"].nunique()
    complete_days = int((frame.groupby("date")["ticker"].nunique() >=
                         REVIEW_GATE["min_tickers_per_day"]).sum())
    coverage = float(interval["new_covered80"].astype(float).mean()) if len(interval) else None
    grouped = frame.groupby("date")[["new_mae", "old_mae", "zero_mae"]].mean()
    advantages = grouped[["old_mae", "zero_mae"]].min(axis=1) - grouped["new_mae"]
    day_win_rate = float((advantages > 0).mean())
    leave_one_out = float(min(advantages.drop(day).mean() for day in advantages.index)) if len(advantages) > 1 else None
    approved = bool(n >= REVIEW_GATE["min_pairs"] and distinct >= REVIEW_GATE["min_days"] and
                    complete_days >= REVIEW_GATE["min_complete_days"] and
                    frame["new_correct"].mean() >= max(REVIEW_GATE["min_direction"], frame["old_correct"].mean()) and
                    frame["new_mae"].mean() <= (1 - REVIEW_GATE["min_mae_improvement"]) * min(
                        frame["old_mae"].mean(), frame["zero_mae"].mean()) and
                    day_win_rate >= REVIEW_GATE["min_day_win_rate"] and
                    leave_one_out > REVIEW_GATE["min_leave_one_day_out_advantage_pp"] and
                    len(interval) >= REVIEW_GATE["min_interval_pairs"] and
                    REVIEW_GATE["coverage80_min"] <= coverage <= REVIEW_GATE["coverage80_max"])
    return {"approved": approved, "n": n, "days": int(distinct), "complete_days": complete_days,
            "new_mae": float(frame["new_mae"].mean()), "old_mae": float(frame["old_mae"].mean()),
            "zero_mae": float(frame["zero_mae"].mean()), "coverage80": coverage,
            "day_win_rate": day_win_rate, "leave_one_day_out_worst_advantage_pp": leave_one_out,
            "new_direction_accuracy": float(frame["new_correct"].mean()),
            "reason": "達到並排實盤採用門檻" if approved else "尚未同時達到交易日／配對數、逐日一致性、方向、MAE與涵蓋率門檻"}


def evaluate_live(root=HERE):
    pairs = live_pairs(root)
    decision = promotion_decision(pairs)
    wf.write_json(Path(root) / "reports" / "daily_retrain_live.json", {"decision": decision, "pairs": pairs})
    return decision


def calibration_errors(ticker, through, root=HERE):
    path = Path(root) / "experiments" / wf.VERSION / f"{ticker}_replay.csv"
    samples = {}
    if path.exists():
        frame = pd.read_csv(path)
        selected = frame[(frame["method"] == "ridge_full") & (frame["date"] <= through)]
        for row in selected.to_dict("records"):
            samples[row["date"]] = row["abs_error_pp"] / row["vol20"]
    # 新的真實留底逐日取代同日回放（若有）；其餘保留校準來源的限制說明。
    root = Path(root)
    for path in sorted((root / "predictions").glob("*_point.json")):
        payload = read_json(path)
        day = payload["target_date"]
        if day > through:
            continue
        for point in payload["records"]:
            candidate = point.get("daily_retrain", {})
            if point["ticker"] != ticker or candidate.get("protocol") != wf.VERSION or not valid_daily_candidate(point, day):
                continue
            metric = score(candidate, day, root, True)
            if metric["eligible"] and candidate.get("vol20", 0) > 0:
                samples[day] = metric["abs_error_pp"] / candidate["vol20"]
    return [samples[day] for day in sorted(samples)][-wf.CAL_WINDOW:]


def train_latest(root=HERE, now=None):
    root = Path(root)
    now = now or datetime.now(TZ)
    events = wf.context_events(root)[0]
    events = [e for e in events if wf.timestamp(e["available_at"]) <= wf.timestamp(now)]
    frames = {}
    for ticker in cfg.ALL_STOCKS:
        frame = wf.frame_for(ticker, root, events)
        cutoff = pd.Timestamp(now.date())
        if now.hour < 14:
            frame = frame[frame["date"] < cutoff]
        else:
            frame = frame[frame["date"] <= cutoff]
        frames[ticker] = frame[frame["actual_pct"].notna()]
    endings = {str(frame["date"].max().date()) for frame in frames.values()}
    if len(endings) != 1:
        raise ValueError("十檔資料最後日期不一致，拒絕混用新舊行情")
    end = endings.pop()
    state_path = root / "models" / "daily_model_state.json"
    if state_path.exists() and read_json(state_path)["trained_through"] > end:
        raise ValueError("行情較已保存模型更舊，拒絕退回舊訓練截止日")
    sources = {ticker: wf.digest(root / "data" / "raw" / f"{ticker}.csv") for ticker in frames}
    import hashlib
    calibration = {ticker: calibration_errors(ticker, end, root) for ticker in frames}
    fingerprint = hashlib.sha256(json.dumps({"prices": sources, "events": events, "calibration": calibration}, sort_keys=True,
                                             ensure_ascii=False).encode()).hexdigest()[:12]
    relative = Path("models") / "daily_candidates" / wf.VERSION / f"{end}_{fingerprint}"
    destination = root / relative
    manifest_path = destination / "manifest.json"
    recovered = sorted(destination.parent.glob(destination.name + "_retry_*/manifest.json"))
    if not manifest_path.exists() and recovered:
        manifest_path = recovered[-1]
        destination = manifest_path.parent
        relative = destination.relative_to(root)
    if manifest_path.exists():
        saved = read_json(manifest_path)
        for ticker, model in saved["models"].items():
            if wf.digest(destination / f"{ticker}.pkl") != model["sha256"]:
                raise ValueError("已保存模型雜湊不符，拒絕採用")
        print(f"每日模型已存在：{end}（相同輸入不重複訓練）")
        wf.write_json(root / "models" / "daily_model_state.json", {"latest": str(relative), "trained_through": end})
        evaluate_live(root)
        return saved
    destination.mkdir(parents=True, exist_ok=True)
    # 上次若在全部完成前中斷，保留半成品另建版本，絕不覆寫已寫出的檔案。
    if any(destination.iterdir()):
        relative = relative.with_name(relative.name + "_retry_" + now.strftime("%H%M%S%f"))
        destination = root / relative
        destination.mkdir(parents=True, exist_ok=False)
        manifest_path = destination / "manifest.json"
    manifest = {"protocol": wf.VERSION, "trained_through": end, "generated_at": now.isoformat(),
                "input_fingerprint": fingerprint,
                "source_sha256": sources, "primary_method": "ridge_full", "models": {},
                "calibration_provenance": "older errors reconstructed; future timestamped predictions replace same dates"}
    with threadpool_limits(limits=1):
        for ticker, frame in frames.items():
            if len(frame) < wf.MIN_TRAIN:
                raise ValueError(f"{ticker} 訓練樣本不足")
            model = wf.fit_model(frame, "ridge_full")
            errors = calibration[ticker]
            bundle = {"model": model, "method": "ridge_full", "protocol": wf.VERSION,
                      "features": wf.FEATURES["ridge_full"], "trained_target_end": end, "n": len(frame),
                      "calibration_errors": errors, "source_sha256": sources[ticker]}
            path = destination / f"{ticker}.pkl"
            with path.open("xb") as f:
                pickle.dump(bundle, f)
            manifest["models"][ticker] = {"sha256": wf.digest(path), "n": len(frame), "calibration_n": len(errors)}
    wf.write_json(manifest_path, manifest, exclusive=True)
    wf.write_json(root / "models" / "daily_model_state.json", {"latest": str(relative), "trained_through": end})
    evaluate_live(root)
    print(f"每日重訓完成：十檔已訓練至 {end}，版本 {fingerprint}")
    return manifest


def candidate_for(ticker, target, root=HERE, now=None):
    root = Path(root)
    state_path = root / "models" / "daily_model_state.json"
    if not state_path.exists():
        return None
    state = read_json(state_path)
    path = root / state["latest"] / f"{ticker}.pkl"
    manifest = read_json(path.parent / "manifest.json")
    if manifest["protocol"] != wf.VERSION or wf.digest(path) != manifest["models"][ticker]["sha256"]:
        raise ValueError("候選版本或模型雜湊不符")
    with path.open("rb") as f:
        bundle = pickle.load(f)
    if bundle["trained_target_end"] >= target:
        raise ValueError("模型已看過目標交易日，不得回填預測")
    frame = wf.frame_for(ticker, root, next_target=target, now=now)
    row = frame.iloc[-1]
    if str(row["date"].date()) != target or str(row["asof"].date()) != bundle["trained_target_end"]:
        raise ValueError("模型訓練截止與最新股價不一致")
    predicted = wf.predict(bundle["model"], row, "ridge_full")
    result = {"ticker": ticker, "name": cfg.ALL_STOCKS[ticker], "protocol": wf.VERSION,
              "data_as_of": str(row["asof"].date()), "target_date": target,
              "trained_target_end": bundle["trained_target_end"], "training_n": bundle["n"],
              "last_close": float(row["base_close"]), "predicted_change_pct": predicted,
              "predicted_close": float(row["base_close"] * (1 + predicted / 100)),
              "direction": "UP" if predicted > 0 else "DOWN" if predicted < 0 else "FLAT",
              "artifact_sha256": wf.digest(path), "model_path": str(path.relative_to(root)),
              "vol20": float(row["vol20"]), "news_available": int(row["news_available"]),
              "context_available_at": None if pd.isna(row["context_available_at"]) else str(row["context_available_at"])}
    for coverage, low_key, high_key in ((.5, "q25_pct", "q75_pct"), (.8, "q10_pct", "q90_pct")):
        radius = wf.conformal_radius(bundle["calibration_errors"], coverage)
        result[low_key] = float(predicted - radius * row["vol20"]) if radius is not None else None
        result[high_key] = float(predicted + radius * row["vol20"]) if radius is not None else None
    manifests = sorted((root / "models" / "daily_candidates" / wf.VERSION).glob("*/manifest.json"))
    past = [p for p in manifests if read_json(p)["trained_through"] < bundle["trained_target_end"]]
    if past:
        old_manifest = read_json(past[-1])
        if wf.digest(past[-1].parent / f"{ticker}.pkl") != old_manifest["models"][ticker]["sha256"]:
            raise ValueError("前一日模型雜湊不符")
        with (past[-1].parent / f"{ticker}.pkl").open("rb") as f:
            prior = pickle.load(f)
        previous = wf.predict(prior["model"], row, "ridge_full")
        result.update(previous_model_predicted_pct=previous, previous_model_training_end=prior["trained_target_end"],
                      prediction_difference_pp=abs(predicted - previous), similar_within_quarter_pp=abs(predicted - previous) <= .25)
    return result


def attach_candidates(predictions, root=HERE, now=None):
    now = require_pre_market(now)
    decision = promotion_decision(live_pairs(root))
    decision.update(publication="shadow_only", review_required=True)
    for point in predictions:
        candidate = candidate_for(point["ticker"], now.date().isoformat(), root, now)
        if candidate is None:
            continue
        candidate["adoption_decision"] = decision
        # 門檻通過只產生可審查的替換建議；本輪授權是先驗證並討論後续。
        point["daily_retrain"] = candidate
    return decision


def preview(target, root=HERE, now=None):
    """盤後查看下一日候選；不冒充下一日九點前已發布的預測。"""
    now = now or datetime.now(TZ)
    if target <= now.date().isoformat():
        raise ValueError("預覽日期必須晚於今天")
    with threadpool_limits(limits=1):
        records = [candidate_for(ticker, target, root, now) for ticker in cfg.ALL_STOCKS]
    result = {"generated_at": now.isoformat(), "target_date": target,
              "status": "after_close_preview_not_immutable_pre_market_prediction",
              "records": records, "decision": evaluate_live(root)}
    wf.write_json(Path(root) / "reports" / "daily_retrain_preview.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "evaluate", "preview"], nargs="?", default="train")
    parser.add_argument("--target")
    args = parser.parse_args()
    if args.mode == "train":
        train_latest()
    elif args.mode == "preview":
        if not args.target:
            parser.error("preview 需要 --target YYYY-MM-DD")
        preview(args.target)
    else:
        print(json.dumps(evaluate_live(), ensure_ascii=False, indent=2))
