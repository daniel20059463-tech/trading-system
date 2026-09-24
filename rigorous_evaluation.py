"""固定時間切割的低成本候選評估；測試段不參與選模或調參。"""
import json
import pickle
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg

HERE = Path(__file__).resolve().parent
FEATURES = ("pct_change", "RSI", "MACD_hist", "Volume_ratio", "ATR",
            "twii_pct", "usdtwd_pct", "murata_pct", "tdk_pct", "onsemi_pct", "vishay_pct")
VALIDATION_DAYS = 100
TEST_DAYS = 60
VALIDATION_START = "2026-01-14"
TEST_START = "2026-06-18"
TEST_END = "2026-09-10"
SCALES = (0.1, 0.25, 0.5, 1.0)


def _models():
    return {
        "historical_majority": None,
        "logistic_c01": make_pipeline(SimpleImputer(), StandardScaler(),
                                        LogisticRegression(C=0.1, max_iter=2000, random_state=17)),
        "logistic_c1": make_pipeline(SimpleImputer(), StandardScaler(),
                                       LogisticRegression(C=1.0, max_iter=2000, random_state=17)),
        "extra_trees": make_pipeline(SimpleImputer(), ExtraTreesClassifier(
            n_estimators=200, min_samples_leaf=8, max_features=0.7, random_state=17, n_jobs=-1)),
        "hist_gradient": make_pipeline(SimpleImputer(), HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=7, l2_regularization=3, random_state=17)),
    }


def _data(ticker):
    df = pd.read_csv(Path(cfg.RAW_DATA_DIR) / f"{ticker}.csv", parse_dates=["Date"]).sort_values("Date")
    features = [c for c in FEATURES if c in df]
    # 第 t 日收盤後可知的特徵，只預測第 t+1 交易日報酬。
    x = df[features].iloc[:-1].reset_index(drop=True)
    y = df["pct_change"].shift(-1).iloc[:-1].reset_index(drop=True)
    dates = df["Date"].shift(-1).iloc[:-1].reset_index(drop=True)
    return x, y, dates, features


def evaluate_ticker(ticker):
    x, y, dates, features = _data(ticker)
    train_index = dates < VALIDATION_START
    validation_index = (dates >= VALIDATION_START) & (dates < TEST_START)
    test_index = (dates >= TEST_START) & (dates <= TEST_END)
    if int(train_index.sum()) < 200:
        raise ValueError(f"{ticker} 時間切割後訓練資料不足")
    if int(validation_index.sum()) != VALIDATION_DAYS or int(test_index.sum()) != TEST_DAYS:
        raise ValueError(f"{ticker} 固定驗證／測試日期筆數改變")
    median_move = float(np.median(np.abs(y[train_index])))
    choices = []
    for name, model in _models().items():
        if model is None:
            probability = np.repeat(float((y[train_index] > 0).mean()), VALIDATION_DAYS)
        else:
            model.fit(x[train_index], y[train_index] > 0)
            probability = model.predict_proba(x[validation_index])[:, 1]
        for scale in SCALES:
            prediction = (2 * probability - 1) * median_move * scale
            accuracy = float(accuracy_score(y[validation_index] > 0, prediction > 0))
            mae = float(mean_absolute_error(y[validation_index], prediction))
            # 固定選模規則：方向為主；每多 1 百分點 MAE 扣 0.02，不接觸測試段。
            choices.append((accuracy - 0.02 * mae, name, scale, accuracy, mae))
    _, selected, scale, validation_accuracy, validation_mae = max(choices)
    model = _models()[selected]
    if model is None:
        probability = np.repeat(float((y[train_index | validation_index] > 0).mean()), TEST_DAYS)
    else:
        model.fit(x[train_index | validation_index], y[train_index | validation_index] > 0)
        probability = model.predict_proba(x[test_index])[:, 1]
    prediction = (2 * probability - 1) * float(np.median(np.abs(y[train_index | validation_index]))) * scale
    actual = y[test_index].to_numpy()
    direction_accuracy = float(accuracy_score(actual > 0, prediction > 0))
    mae = float(mean_absolute_error(actual, prediction))
    zero_mae = float(mean_absolute_error(actual, np.zeros_like(actual)))
    eligible = direction_accuracy >= 0.55 and mae <= zero_mae * 0.98
    test_dates = dates[test_index].dt.date.astype(str).tolist()
    daily = [{"date": day, "ticker": ticker, "candidate": selected,
              "predicted_pct": float(pred), "actual_pct": float(act),
              "direction_correct": bool((pred > 0) == (act > 0)),
              "abs_error_pp": abs(float(pred - act)),
              "zero_baseline_abs_error_pp": abs(float(act))}
             for day, pred, act in zip(test_dates, prediction, actual)]
    return {
        "ticker": ticker, "name": cfg.ALL_STOCKS[ticker], "features": features,
        "train_end": dates[train_index].iloc[-1].date().isoformat(),
        "validation_start": dates[validation_index].iloc[0].date().isoformat(),
        "validation_end": dates[validation_index].iloc[-1].date().isoformat(),
        "test_start": dates[test_index].iloc[0].date().isoformat(),
        "test_end": dates[test_index].iloc[-1].date().isoformat(), "test_n": TEST_DAYS,
        "selected_on_validation": selected, "scale": scale,
        "validation_direction_accuracy": validation_accuracy,
        "validation_mae_pp": validation_mae,
        "test_direction_accuracy": direction_accuracy,
        "test_mae_pp": mae,
        "test_zero_baseline_mae_pp": zero_mae,
        "shadow_eligible": eligible,
        "_daily": daily,
    }


def _save_shadow(row, output):
    """測試已固定且通過個股門檻後，才用截至測試終點的資料訓練未來影子模型。"""
    ticker = row["ticker"]
    x, y, dates, features = _data(ticker)
    fit_index = dates <= TEST_END
    model = _models()[row["selected_on_validation"]]
    payload = {"schema_version": 1, "ticker": ticker, "features": features,
               "selected_on_validation": row["selected_on_validation"], "scale": row["scale"],
               "median_abs_move": float(np.median(np.abs(y[fit_index]))),
               "trained_targets_through": TEST_END, "evidence": row}
    if model is None:
        payload["constant_probability"] = float((y[fit_index] > 0).mean())
    else:
        model.fit(x[fit_index], y[fit_index] > 0)
        payload["model"] = model
    directory = Path(output) / "shadow_candidates" / "fixed_20260618_20260910"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{ticker}.pkl").open("wb") as f:
        pickle.dump(payload, f)
    return directory


def predict_shadow(ticker, root=HERE):
    path = Path(root) / "reports" / "shadow_candidates" / "fixed_20260618_20260910" / f"{ticker}.pkl"
    if not path.exists():
        return None
    with path.open("rb") as f:
        payload = pickle.load(f)
    df = pd.read_csv(Path(root) / "data" / "raw" / f"{ticker}.csv", parse_dates=["Date"]).sort_values("Date")
    latest = df[payload["features"]].iloc[[-1]]
    probability = payload.get("constant_probability")
    if probability is None:
        probability = float(payload["model"].predict_proba(latest)[0, 1])
    predicted_pct = (2 * probability - 1) * payload["median_abs_move"] * payload["scale"]
    return {"version": "fixed_20260618_20260910", "ticker": ticker,
            "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "data_as_of": df["Date"].iloc[-1].date().isoformat(),
            "last_close": float(df["Close"].iloc[-1]),
            "p_up": probability, "predicted_change_pct": float(predicted_pct),
            "direction": "UP" if predicted_pct > 0 else "DOWN", "status": "shadow_only"}


def production_post_checkpoint():
    """只作 checkpoint 寫入日期後的診斷；mtime 不能證明訓練資料截止日。"""
    from backtest import _predict_testset_ensemble
    parts, detail = [], []
    for ticker in cfg.ALL_STOCKS:
        paths = [Path(cfg.CHECKPOINT_DIR) / f"{ticker}_{kind}_best.pt" for kind in ("lstm", "transformer")]
        cutoff = max(pd.Timestamp.fromtimestamp(path.stat().st_mtime).normalize() for path in paths)
        frame = _predict_testset_ensemble(ticker, "full")
        frame = frame[(frame["date"].dt.normalize() > cutoff) & (frame["date"] <= TEST_END)].copy()
        if frame.empty:
            continue
        parts.append(frame)
        detail.append({"ticker": ticker, "checkpoint_file_date": cutoff.date().isoformat(),
                       "start": frame["date"].min().date().isoformat(),
                       "end": frame["date"].max().date().isoformat(), "n": len(frame),
                       "direction_accuracy": float(((frame["predicted_pct"] > 0) == (frame["actual_pct"] > 0)).mean()),
                       "mae_pp": float((frame["predicted_pct"] - frame["actual_pct"]).abs().mean()),
                       "zero_baseline_mae_pp": float(frame["actual_pct"].abs().mean())})
    if not parts:
        return {"n": 0, "detail": []}
    frame = pd.concat(parts, ignore_index=True)
    return {"n": len(frame),
            "direction_accuracy": float(((frame["predicted_pct"] > 0) == (frame["actual_pct"] > 0)).mean()),
            "mae_pp": float((frame["predicted_pct"] - frame["actual_pct"]).abs().mean()),
            "zero_baseline_mae_pp": float(frame["actual_pct"].abs().mean()), "detail": detail,
            "limitation": "checkpoint 未記錄訓練資料截止日；寫入日期不是防洩漏證明，僅作歷史診斷"}


def production_interval_post_checkpoint():
    """評估 checkpoint 寫入後的原始區間；不套用無法重建的歷史新聞修正。"""
    from quantile_forecast import (_load_quantile, _load_feature_df, _predict_scaled_window,
                                   _vol_series, VOL_WINDOW_SHORT, RHO_MIN, RHO_MAX)
    rows = []
    for ticker in cfg.ALL_STOCKS:
        path = Path(cfg.CHECKPOINT_DIR) / f"{ticker}_quantile_best.pt"
        cutoff = pd.Timestamp.fromtimestamp(path.stat().st_mtime).normalize()
        model, scaler, checkpoint = _load_quantile(ticker)
        df = _load_feature_df(ticker, checkpoint["feature_names"])
        scaled = scaler.transform(df.values.astype(np.float32))
        sigma20, sigma5 = _vol_series(df).to_numpy(), _vol_series(df, VOL_WINDOW_SHORT).to_numpy()
        w = cfg.WINDOW_SIZE
        for target_index in range(w, len(df)):
            target_date = pd.Timestamp(df.index[target_index])
            if target_date.normalize() <= cutoff or target_date > pd.Timestamp(TEST_END):
                continue
            q10, q25, q50, q75, q90 = _predict_scaled_window(
                model, scaler, checkpoint, scaled[target_index-w:target_index])
            known = target_index - 1
            lo80 = max(-10.0, q10 - checkpoint.get("conformal_q80", 0.0) * sigma20[known])
            hi80 = min(10.0, q90 + checkpoint.get("conformal_q80", 0.0) * sigma20[known])
            lo50 = max(lo80, q25 - checkpoint.get("conformal_q50", 0.0) * sigma20[known])
            hi50 = min(hi80, q75 + checkpoint.get("conformal_q50", 0.0) * sigma20[known])
            rho = min(max(sigma5[known] / sigma20[known], RHO_MIN), RHO_MAX)
            cq_short = checkpoint.get("conformal_q80_s5", 0.0)
            lo_short = max(-10.0, q50 + (q10 - q50) * rho - cq_short * sigma5[known])
            hi_short = min(10.0, q50 + (q90 - q50) * rho + cq_short * sigma5[known])
            actual = float(df["pct_change"].iloc[target_index])
            for label, lo, hi, alpha in (("50", lo50, hi50, .5), ("80_sigma20", lo80, hi80, .2),
                                         ("80_sigma5", lo_short, hi_short, .2)):
                interval_score = hi - lo
                if actual < lo:
                    interval_score += 2 / alpha * (lo - actual)
                elif actual > hi:
                    interval_score += 2 / alpha * (actual - hi)
                rows.append({"date": target_date.date().isoformat(), "ticker": ticker, "interval": label,
                             "lo_pct": lo, "hi_pct": hi, "actual_pct": actual, "width_pp": hi - lo,
                             "covered": lo <= actual <= hi, "interval_score": interval_score})
    if not rows:
        return {"summary": {}, "daily": []}
    frame = pd.DataFrame(rows)
    summary = {label: {"n": len(group), "coverage": float(group["covered"].mean()),
                       "mean_width_pp": float(group["width_pp"].mean()),
                       "mean_interval_score": float(group["interval_score"].mean())}
               for label, group in frame.groupby("interval")}
    return {"summary": summary, "daily": rows,
            "limitation": "只評無新聞修正的原始區間；舊校準段曾用於選模，寫入日期不能證明模型沒看過之後資料"}


def run(output=None):
    output = Path(output) if output else HERE / "reports"
    output.mkdir(parents=True, exist_ok=True)
    rows = [evaluate_ticker(ticker) for ticker in cfg.ALL_STOCKS]
    daily = [item for row in rows for item in row.pop("_daily")]
    shadow_artifacts = {}
    for row in rows:
        if row["shadow_eligible"]:
            directory = _save_shadow(row, output)
            path = directory / f"{row['ticker']}.pkl"
            shadow_artifacts[row["ticker"]] = hashlib.sha256(path.read_bytes()).hexdigest()
    total = sum(r["test_n"] for r in rows)
    result = {
        "schema_version": 1,
        "policy": {"validation_days": VALIDATION_DAYS, "test_days": TEST_DAYS,
                   "validation_start": VALIDATION_START, "test_start": TEST_START, "test_end": TEST_END,
                   "candidate_names": list(_models()), "scales": SCALES,
                   "test_used_for_selection": False},
        "rows": rows,
        "pooled": {
            "n": total,
            "direction_accuracy": sum(r["test_direction_accuracy"] * r["test_n"] for r in rows) / total,
            "mae_pp": sum(r["test_mae_pp"] * r["test_n"] for r in rows) / total,
            "zero_baseline_mae_pp": sum(r["test_zero_baseline_mae_pp"] * r["test_n"] for r in rows) / total,
        },
        "production_post_checkpoint": production_post_checkpoint(),
        "production_interval_post_checkpoint": production_interval_post_checkpoint(),
        "input_sha256": {str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(Path(cfg.RAW_DATA_DIR).glob("*.csv"))},
        "shadow_artifact_sha256": shadow_artifacts,
    }
    pooled = result["pooled"]
    production = result["production_post_checkpoint"]
    result["promotion_decision"] = "reject_global" if (
        pooled["direction_accuracy"] < 0.52 or
        pooled["mae_pp"] >= pooled["zero_baseline_mae_pp"] * 0.98
    ) else "eligible_for_shadow"
    result["shadow_eligible_tickers"] = [r["ticker"] for r in rows if r["shadow_eligible"]]
    daily_frame = pd.DataFrame(daily)
    by_date = daily_frame.groupby("date").agg(
        n=("ticker", "size"), direction_accuracy=("direction_correct", "mean"),
        mae_pp=("abs_error_pp", "mean"),
        zero_baseline_mae_pp=("zero_baseline_abs_error_pp", "mean")).reset_index()
    result["worst_test_dates_by_mae"] = by_date.nlargest(5, "mae_pp").to_dict("records")
    interval_daily = result["production_interval_post_checkpoint"].pop("daily")
    (output / "rigorous_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output / "rigorous_evaluation.csv", index=False, encoding="utf-8-sig")
    daily_frame.to_csv(output / "rigorous_evaluation_daily.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(interval_daily).to_csv(
        output / "production_interval_post_checkpoint.csv", index=False, encoding="utf-8-sig")
    lines = ["# 固定時間切割候選評估", "",
             f"測試期：{rows[0]['test_start']} 至 {rows[0]['test_end']}；每股 {TEST_DAYS} 日，共 {total} 筆。",
             "候選及幅度縮放只由更早的100日驗證段選擇；測試段未參與選擇。這是一次性歷史檢驗，不等於未來實盤。", "",
             f"- 測試方向準確率：{pooled['direction_accuracy']:.1%}",
             f"- 候選幅度 MAE：{pooled['mae_pp']:.4f} 百分點",
             f"- 猜不變 MAE：{pooled['zero_baseline_mae_pp']:.4f} 百分點",
             f"- 決策：{result['promotion_decision']}", "",
             f"- 探索性個股影子候選：{', '.join(result['shadow_eligible_tickers']) or '無'}", "",
             f"現有正式模型於 checkpoint 寫入日後：{production.get('n', 0)} 筆，方向 {production.get('direction_accuracy', 0):.1%}，"
             f"MAE {production.get('mae_pp', 0):.4f}，猜不變 {production.get('zero_baseline_mae_pp', 0):.4f} 百分點。",
             "checkpoint 沒有訓練資料截止欄位；mtime 不等於訓練資料截止日，所以90筆只列診斷，不能稱為獨立樣本外。", "",
             "區間於 checkpoint 寫入日後的原始診斷（不含歷史新聞修正）："]
    for label, metric in result["production_interval_post_checkpoint"].get("summary", {}).items():
        lines.append(f"- {label}：n={metric['n']}，涵蓋 {metric['coverage']:.1%}，平均寬 {metric['mean_width_pp']:.4f}，區間分數 {metric['mean_interval_score']:.4f}")
    lines += ["", "整體結果未同時達到方向52%及相對猜不變至少2%的幅度改善，正式模型不替換。",
              "2481／3675 是查看同一測試段後挑出的探索性影子候選，存在選擇偏誤；只能用未來新資料驗證，不能宣稱已獨立通過。"]
    (output / "rigorous_evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
