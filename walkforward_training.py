"""逐日重訓回放及每日候選。日期 t 的目標絕不進入日期 t 的訓練或區間校準。"""
import argparse
import hashlib
import json
import math
import pickle
import re
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import config as cfg
from prediction_audit import TZ, number, read_json, require_pre_market

HERE = Path(__file__).resolve().parent
VERSION = "daily_v2_20260915"
MIN_TRAIN = 60
MIN_CAL = 30
CAL_WINDOW = 60
HALFLIFE = 126
BASE = ["ret1", "ret5", "ret20", "vol20", "range_pct", "volume_ratio", "twii",
        "fx_lag", "peer1_lag", "peer2_lag"]
STRATEGY = ["trend", "mean_reversion", "breakout", "volume_trend"]
WORDS = ["漲價", "缺貨", "訂單", "營收", "財報", "外資", "買超", "賣超", "庫存", "需求", "AI", "利空"]
CONTEXT = ["news_available", "news_sentiment", "news_signal", "strategy_available", "strategy_action"] + ["kw_" + w for w in WORDS]
FEATURES = {"ridge_price": BASE, "ridge_strategy": BASE + STRATEGY,
            "ridge_full": BASE + STRATEGY + CONTEXT, "tree_full": BASE + STRATEGY + CONTEXT}
METHODS = ["zero", "history_median", "frozen_full", "yesterday_full"] + list(FEATURES)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data, exclusive=False):
    content = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as f:
        f.write(content)


@lru_cache(maxsize=10000)
def timestamp(value):
    try:
        result = pd.Timestamp(value)
        return result.tz_localize(TZ) if result.tzinfo is None else result.tz_convert(TZ)
    except (ValueError, TypeError):
        return None


def context_events(root=HERE, include_legacy_pre=True):
    """舊 post JSON 只作其時間戳之後的歷史重建；不讀累積關鍵字勝率。"""
    root = Path(root)
    events, diagnostics = [], {"post_files": 0, "legacy_pre_files": 0, "context_snapshots": 0, "skipped": []}
    mapping = {t.split(".")[0]: t for t in cfg.ALL_STOCKS}
    for path in sorted((root / "news").glob("*_post_market.json")):
        data = read_json(path)
        stamp = timestamp(data.get("timestamp"))
        if stamp is None or str(stamp.date()) != path.name[:10]:
            diagnostics["skipped"].append(path.name)
            continue
        diagnostics["post_files"] += 1
        for code, info in data.get("post_market_analysis", {}).items():
            ticker = mapping.get(code, code)
            if ticker not in cfg.ALL_STOCKS or not isinstance(info, dict):
                continue
            events.append({"ticker": ticker, "available_at": stamp.isoformat(),
                           "event_time": stamp.isoformat(),
                           "news": info, "strategy": {}, "source": str(path.relative_to(root)),
                           "provenance": "legacy_timestamp_reconstruction"})
    if include_legacy_pre:
        for path in sorted((root / "news").glob("*_pre_market.json")):
            day = path.name[:10]
            log = root / "logs" / f"{day}_pre.log"
            if not log.exists():
                diagnostics["skipped"].append(path.name + ":no_log")
                continue
            content = log.read_text(encoding="utf-8-sig", errors="replace")
            stamps = re.findall(r"(?m)(?:^|python.exe : )(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?:,\d+)? \[(?:INFO|WARNING|ERROR)\]", content)
            if not stamps:
                diagnostics["skipped"].append(path.name + ":no_timestamp")
                continue
            logged = max(timestamp(s) for s in stamps)
            # 不確定此舊檔是否即當日首份發布；延至其日末且不早於最晚執行時間。
            available = max(logged, timestamp(day + "T23:59:59+08:00"))
            diagnostics["legacy_pre_files"] += 1
            for code, info in read_json(path).items():
                ticker = mapping.get(code, code)
                if ticker not in cfg.ALL_STOCKS or not isinstance(info, dict):
                    continue
                events.append({"ticker": ticker, "available_at": available.isoformat(), "event_time": logged.isoformat(),
                               "news": info, "strategy": {}, "source": str(path.relative_to(root)),
                               "provenance": "legacy_log_reconstruction_delayed_to_next_day"})
    for path in sorted((root / "context_snapshots").glob("*.json")):
        data = read_json(path)
        stamp = timestamp(data.get("generated_at"))
        if stamp is None:
            diagnostics["skipped"].append(path.name)
            continue
        diagnostics["context_snapshots"] += 1
        for item in data.get("records", []):
            events.append({**item, "available_at": stamp.isoformat(), "source": str(path.relative_to(root)),
                           "provenance": "timestamped_snapshot"})
    return events, diagnostics


def context_values(events, ticker, cutoff):
    cutoff = timestamp(cutoff)
    candidates = [e for e in events if e["ticker"] == ticker and
                  cutoff - pd.Timedelta(days=3) <= timestamp(e["available_at"]) < cutoff]
    values = {f: 0.0 for f in CONTEXT}
    if not candidates:
        return values, None
    event = max(candidates, key=lambda e: timestamp(e.get("event_time", e["available_at"])))
    news, strategy = event.get("news", {}), event.get("strategy", {})
    text = " ".join(str(news.get(k, "")) for k in ("key_catalysts", "risk_factors", "strategy_reason"))
    values["news_available"] = float(bool(news))
    values["news_sentiment"] = number(news.get("sentiment_score")) or 0.0
    values["news_signal"] = {"buy": 1, "sell": -1}.get(news.get("strategy_signal"), 0)
    values["strategy_available"] = float(bool(strategy))
    values["strategy_action"] = {"buy": 1, "sell": -1}.get(strategy.get("action"), 0)
    for word in WORDS:
        values["kw_" + word] = float(word in text)
    return values, event["available_at"]


def read_prices(root, ticker):
    frame = pd.read_csv(Path(root) / "data" / "raw" / f"{ticker}.csv", parse_dates=["Date"])
    frame = frame.sort_values("Date").reset_index(drop=True)
    if frame["Date"].duplicated().any() or (frame["Close"] <= 0).any() or not np.isfinite(frame["Close"]).all():
        raise ValueError(f"{ticker} 日期重複或價格無效")
    return frame


def frame_for(ticker, root=HERE, events=None, next_target=None, now=None):
    df = read_prices(root, ticker)
    if next_target:
        df = df[df["Date"] < next_target].reset_index(drop=True)
    if events is None:
        events = context_events(root)[0]
    events = [e for e in events if e["ticker"] == ticker]
    close, volume = df["Close"], df["Volume"]
    returns = close.pct_change(fill_method=None) * 100
    x = pd.DataFrame(index=df.index)
    x["ret1"], x["ret5"], x["ret20"] = returns, close.pct_change(5) * 100, close.pct_change(20) * 100
    x["vol20"] = returns.rolling(20, min_periods=20).std().clip(lower=.3)
    x["range_pct"] = (df["High"] - df["Low"]) / close * 100
    x["volume_ratio"] = volume / volume.rolling(20).mean()
    x["twii"] = df["twii_pct"]
    x["fx_lag"] = df["usdtwd_pct"].shift(1)
    peers = ("murata_pct", "tdk_pct") if ticker in cfg.PASSIVE_STOCKS else ("onsemi_pct", "vishay_pct")
    x["peer1_lag"], x["peer2_lag"] = df[peers[0]].shift(1), df[peers[1]].shift(1)
    # 策略訊號完全由當時已知價格重建，不使用今天更新的規則庫。
    mean20 = close.rolling(20).mean()
    x["trend"] = np.sign(close.rolling(5).mean() - mean20)
    x["mean_reversion"] = -(close - mean20) / close.rolling(20).std().replace(0, np.nan)
    x["breakout"] = (close > close.shift(1).rolling(20).max()).astype(float) - (close < close.shift(1).rolling(20).min()).astype(float)
    x["volume_trend"] = np.sign(returns) * (x["volume_ratio"] > 1.5).astype(float)
    x["date"] = df["Date"].shift(-1)
    if next_target:
        x.loc[x.index[-1], "date"] = pd.Timestamp(next_target)
    x["asof"] = df["Date"]
    x["base_close"] = close
    x["actual_close"] = close.shift(-1)
    x["actual_pct"] = returns.shift(-1)
    x["ticker"] = ticker
    context, times = [], []
    for day in x["date"]:
        if pd.isna(day):
            context.append({f: 0.0 for f in CONTEXT})
            times.append(None)
            continue
        cutoff = timestamp(str(day.date()) + "T09:00:00+08:00")
        if now is not None and day.date() >= now.date():
            cutoff = min(cutoff, timestamp(now))
        values, available = context_values(events, ticker, cutoff)
        context.append(values)
        times.append(available)
    for feature in CONTEXT:
        x[feature] = [v[feature] for v in context]
    x["context_available_at"] = times
    # 特徵暖身或資料缺漏直接列為不可預測，不補未來值。
    finite = np.isfinite(x[BASE + STRATEGY].to_numpy(dtype=float)).all(axis=1)
    return x[finite & x["date"].notna()].reset_index(drop=True)


def fit_model(training, method):
    columns = FEATURES[method]
    x = training[columns].to_numpy(dtype=float)
    y = training["actual_pct"].to_numpy(dtype=float) / training["vol20"].to_numpy(dtype=float)
    weights = np.exp2(-np.arange(len(training) - 1, -1, -1) / HALFLIFE)
    if method.startswith("ridge"):
        model = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
        model.fit(x, y, ridge__sample_weight=weights)
    else:
        model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=60, max_leaf_nodes=7,
                                             min_samples_leaf=30, l2_regularization=10.0,
                                             early_stopping=False, random_state=17)
        model.fit(x, y, sample_weight=weights)
    return model


def predict(model, row, method):
    return float(model.predict(row[FEATURES[method]].to_numpy(dtype=float).reshape(1, -1))[0] * row["vol20"])


def conformal_radius(errors, coverage):
    if len(errors) < MIN_CAL:
        return None
    values = sorted(errors[-CAL_WINDOW:])
    rank = math.ceil((len(values) + 1) * coverage)
    return float(values[rank - 1]) if rank <= len(values) else None


def metrics_row(row, method, predicted, errors, train_end, train_n):
    actual = float(row["actual_pct"])
    sign = lambda value: 0 if abs(value) < 1e-9 else (1 if value > 0 else -1)
    result = {"date": str(row["date"].date()), "ticker": row["ticker"], "method": method,
              "asof": str(row["asof"].date()), "train_target_end": str(train_end.date()), "train_n": train_n,
              "predicted_pct": predicted, "actual_pct": actual, "abs_error_pp": abs(actual - predicted),
              "direction_correct": sign(actual) == sign(predicted), "signal": sign(predicted),
              "news_available": int(row["news_available"]), "context_available_at": row["context_available_at"],
              "vol20": float(row["vol20"]), "trend": float(row["trend"])}
    for coverage in (.5, .8):
        label = str(int(coverage * 100))
        radius = conformal_radius(errors, coverage)
        low = predicted - radius * row["vol20"] if radius is not None else None
        high = predicted + radius * row["vol20"] if radius is not None else None
        score = (high - low + 2 / (1 - coverage) * max(low - actual, actual - high, 0)) if low is not None else None
        result.update({f"lo{label}": low, f"hi{label}": high,
                       f"width{label}": high - low if low is not None else None,
                       f"covered{label}": low <= actual <= high if low is not None else None,
                       f"interval_score{label}": score})
    return result


def replay_ticker(frame, methods=METHODS):
    records, errors, previous, frozen = [], {m: [] for m in methods}, None, None
    for i in range(MIN_TRAIN, len(frame)):
        row, training = frame.iloc[i], frame.iloc[:i]
        assert training["date"].max() < row["date"]
        if not np.isfinite(row["actual_pct"]):
            continue
        models = {m: fit_model(training, m) for m in FEATURES if m in methods or m == "ridge_full"}
        if frozen is None:
            frozen = models["ridge_full"]
        forecasts = {m: predict(model, row, m) for m, model in models.items()}
        forecasts.update(zero=0.0, history_median=float(training["actual_pct"].median()),
                         frozen_full=predict(frozen, row, "ridge_full"),
                         yesterday_full=predict(previous or models["ridge_full"], row, "ridge_full"))
        for method in methods:
            rec = metrics_row(row, method, forecasts[method], errors[method], training["date"].max(), i)
            if method == "frozen_full":
                rec["train_target_end"] = str(frame.iloc[MIN_TRAIN-1]["date"].date())
                rec["train_n"] = MIN_TRAIN
            if method == "yesterday_full" and i > MIN_TRAIN:
                rec["train_target_end"] = str(frame.iloc[i-2]["date"].date())
                rec["train_n"] = i - 1
            rec["vs_yesterday_prediction_pp"] = abs(forecasts["ridge_full"] - forecasts["yesterday_full"])
            records.append(rec)
        # 必須先記錄該日預測和區間，再把該日結果加進明天的校準。
        for method in methods:
            errors[method].append(abs(row["actual_pct"] - forecasts[method]) / row["vol20"])
        previous = models["ridge_full"]
        if (i - MIN_TRAIN) % 100 == 0:
            print(f"{row['ticker']} 已回放至 {row['date'].date()}（{i-MIN_TRAIN+1}日）", flush=True)
    return records


def aggregate(frame):
    values = {"n": len(frame), "days": int(frame["date"].nunique()),
              "direction_accuracy": float(frame["direction_correct"].mean()),
              "mae_pp": float(frame["abs_error_pp"].mean()),
              "signal_rate": float((frame["signal"] != 0).mean())}
    for coverage in (50, 80):
        available = frame[frame[f"covered{coverage}"].notna()]
        values.update({f"interval_n{coverage}": len(available),
                       f"coverage{coverage}": float(available[f"covered{coverage}"].astype(float).mean()) if len(available) else None,
                       f"width{coverage}_pp": float(available[f"width{coverage}"].mean()) if len(available) else None,
                       f"interval_score{coverage}": float(available[f"interval_score{coverage}"].mean()) if len(available) else None})
    return values


def run_replay(root=HERE, output=None):
    root = Path(root)
    output = Path(output) if output else root / "experiments" / VERSION
    if (output / "plan.json").exists():
        raise FileExistsError("實驗已鎖定；不可把新試驗覆寫成同一結果。另指定 --output 以重現。")
    output.mkdir(parents=True, exist_ok=True)
    events, diagnostics = context_events(root)
    # 封存此次行情與文字輸入。重現時不依賴被每日刷新覆蓋的原始CSV。
    snapshot = output / "inputs"
    import shutil
    files = (list((root / "data" / "raw").glob("*.csv")) + list((root / "news").glob("*_market.json")) +
             list((root / "logs").glob("*_pre.log")) + list((root / "context_snapshots").glob("*.json")))
    for path in files:
        target = snapshot / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write_json(output / "plan.json", {"version": VERSION, "locked_at": datetime.now(TZ).isoformat(),
               "minimum_training_targets": MIN_TRAIN, "history": "expanding, age weights half-life 126 sessions",
               "methods": METHODS, "features": FEATURES, "calibration": "last60 past prequential errors; min30",
               "similarity_pp": .25, "good_day": "ridge_full direction>=70%, MAE<=1pp, beats zero on same ten stocks",
               "primary_method": "ridge_full", "no_hyperparameter_selection": True,
               "historical_status": "reconstruction/exploratory; older periods already examined in prior work",
               "context_diagnostics": diagnostics, "input_sha256": {str(p.relative_to(root)): digest(p) for p in files}}, exclusive=True)
    all_rows, warmup = [], []
    with threadpool_limits(limits=1):
        for ticker in cfg.ALL_STOCKS:
            frame = frame_for(ticker, snapshot, events)
            warmup.append({"ticker": ticker, "raw_first_date": str(read_prices(snapshot, ticker)["Date"].min().date()),
                           "first_training_target": str(frame.iloc[0]["date"].date()),
                           "first_forecast_target": str(frame.iloc[MIN_TRAIN]["date"].date()), "warmup_targets": MIN_TRAIN})
            rows = replay_ticker(frame)
            pd.DataFrame(rows).to_csv(output / f"{ticker}_replay.csv", index=False, encoding="utf-8-sig")
            all_rows.extend(rows)
    frame = pd.DataFrame(all_rows)
    frame.to_csv(output / "all_predictions.csv", index=False, encoding="utf-8-sig")
    write_json(output / "warmup.json", warmup)
    result = report(frame, output)
    print(json.dumps(result["recent"], ensure_ascii=False, indent=2), flush=True)
    return result


def report(frame, output):
    output = Path(output)
    end = max(frame["date"])
    recent_start = str((pd.Timestamp(end) - pd.DateOffset(months=2)).date())
    recent = frame[frame["date"] >= recent_start]
    result = {"start": min(frame["date"]), "end": end, "recent_start": recent_start,
              "all": {m: aggregate(g) for m, g in frame.groupby("method")},
              "recent": {m: aggregate(g) for m, g in recent.groupby("method")},
              "context_present": {m: aggregate(g) for m, g in frame[frame["news_available"] == 1].groupby("method")}}
    daily = frame.groupby(["date", "method"]).agg(n=("ticker", "size"), accuracy=("direction_correct", "mean"), mae=("abs_error_pp", "mean")).reset_index()
    primary = daily[daily["method"] == "ridge_full"].merge(daily[daily["method"] == "zero"][["date", "mae"]], on="date", suffixes=("", "_zero"))
    primary["good"] = (primary["n"] == len(cfg.ALL_STOCKS)) & (primary["accuracy"] >= .7) & (primary["mae"] <= 1) & (primary["mae"] < primary["mae_zero"])
    primary.to_csv(output / "daily_quality.csv", index=False, encoding="utf-8-sig")
    result["good_days_recent"] = primary[(primary["date"] >= recent_start) & primary["good"]].to_dict("records")
    result["good_days_all"] = primary[primary["good"]].to_dict("records")
    result["best_recent_diagnostic"] = primary[primary["date"] >= recent_start].sort_values(["accuracy", "mae"], ascending=[False, True]).head(8).to_dict("records")
    result["worst_recent"] = primary[primary["date"] >= recent_start].nlargest(5, "mae").to_dict("records")
    primary_rows = recent[recent["method"] == "ridge_full"].copy()
    primary_rows["regime"] = np.where(primary_rows["vol20"] >= 3, "prior_vol20_at_least_3pp", "prior_vol20_below_3pp")
    result["known_before_prediction_conditions"] = {
        name: aggregate(group) for name, group in primary_rows.groupby("regime")}
    recent_days = primary[primary["date"] >= recent_start].copy()
    after_good = primary["good"].shift(1, fill_value=False)
    followers = primary[after_good & (primary["date"] >= recent_start)]
    result["day_after_good_day"] = {"days": len(followers),
                                    "accuracy": float(followers["accuracy"].mean()) if len(followers) else None,
                                    "mae_pp": float(followers["mae"].mean()) if len(followers) else None}
    # 新增因素必須在同股票同日期配對，避免新聞少的日期被誤算成增益。
    result["ablation_effects"] = {}
    for sample_name, sample in (("recent_all", recent), ("news_present", frame[frame["news_available"] == 1])):
        comparison = sample.pivot(index=["date", "ticker"], columns="method", values="abs_error_pp")
        result["ablation_effects"][sample_name] = {
            "n": len(comparison),
            "strategy_mae_reduction_pp": float((comparison["ridge_price"] - comparison["ridge_strategy"]).mean()),
            "news_mae_reduction_pp": float((comparison["ridge_strategy"] - comparison["ridge_full"]).mean())}
    ticker_rows = [{"ticker": t, "method": m, **aggregate(g)} for (t, m), g in recent.groupby(["ticker", "method"])]
    pd.DataFrame(ticker_rows).to_csv(output / "recent_by_ticker.csv", index=False, encoding="utf-8-sig")
    # 用前20個日期可知的誤差比較決定翌日用新模型或昨天版本，不能用当天答案挑。
    comparisons = primary[["date", "mae", "accuracy"]].merge(daily[daily["method"] == "yesterday_full"][["date", "mae", "accuracy"]], on="date", suffixes=("_new", "_old"))
    comparisons["historical_advantage_pp"] = (comparisons["mae_old"] - comparisons["mae_new"]).shift(1).rolling(20, min_periods=20).mean()
    comparisons["use_new"] = comparisons["historical_advantage_pp"] > 0
    comparisons["selected_mae"] = np.where(comparisons["use_new"], comparisons["mae_new"], comparisons["mae_old"])
    comparisons.to_csv(output / "new_vs_yesterday.csv", index=False, encoding="utf-8-sig")
    result["past_only_selection_mae"] = float(comparisons["selected_mae"].mean())
    result["similarity_rate"] = float((frame[frame["method"] == "ridge_full"]["vs_yesterday_prediction_pp"] <= .25).mean())
    write_json(output / "summary.json", result)
    lines = ["# 逐日重訓回放", "", f"評估 {result['start']}～{end}；近期 {recent_start} 起。",
             "每個交易日只使用更早已收盤目標重訓；校準只用更早逐日誤差。回放屬歷史重建，不能當原始實盤成績。", "",
             "| 最近兩月方法 | 樣本 | 方向（三類） | MAE百分點 | 80%涵蓋 | 80%寬度 | 80%區間分數 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for method, stats in result["recent"].items():
        lines.append(f"| {method} | {stats['n']} | {stats['direction_accuracy']:.1%} | {stats['mae_pp']:.4f} | {stats['coverage80']:.1%} | {stats['width80_pp']:.3f} | {stats['interval_score80']:.3f} |")
    lines += ["", "## 達到事先定義的好日子", "", "標準：同十檔，方向至少70%、MAE不超過1百分點，並優於猜不變。"]
    if not result["good_days_recent"]:
        lines.append("近期沒有日期同時達標；不能把相對較好的日子說成高精準。")
    for row in result["good_days_recent"]:
        lines.append(f"- {row['date']}：方向 {row['accuracy']:.0%}，MAE {row['mae']:.3f}，猜不變 {row['mae_zero']:.3f}。")
    lines += ["", "## 近期相對較好的日期（事後診斷）"]
    for row in result["best_recent_diagnostic"]:
        lines.append(f"- {row['date']}：方向 {row['accuracy']:.0%}，MAE {row['mae']:.3f}，猜不變 {row['mae_zero']:.3f}。")
    lines += ["", "## 失敗日期"]
    for row in result["worst_recent"]:
        lines.append(f"- {row['date']}：方向 {row['accuracy']:.0%}，MAE {row['mae']:.3f}，猜不變 {row['mae_zero']:.3f}。")
    lines += ["", "新聞採記錄時間之後才可用的盤後內容，最多保留72小時；無時間戳的舊日曆關鍵字快取不採用。缺新聞以available旗標表示，並非中性新聞。",
              "策略包含當時價格可重建訊號、新聞買賣建議；新完整策略在context_snapshots留底後方可加入。",
              "每天重訓不代表每天改善。好日子是事後描述，不能拿當天答案決定當天模型。",
              "價格為本次封存的還原行情，非當時原始資料版本；旧新聞時間戳亦非不可變留底，結果僅為歷史重建。"]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["replay"])
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_replay(args.root, args.output)
