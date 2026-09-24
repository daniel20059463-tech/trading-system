"""受治理的正式預測政策：不可覆寫事件、人工採用、獨立監測及回滾。"""
import argparse
import hashlib
import json
import os
import shutil
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from prediction_audit import HERE, TZ, read_json, valid_daily_candidate

POLICY_VERSION = "daily_retrain_adoption_v1"
CANDIDATE_PROTOCOL = "daily_v2_20260915"
REVIEW_GATE = {
    "lookback_days": 40, "min_days": 20, "min_complete_days": 20,
    "min_tickers_per_day": 10, "min_pairs": 200,
    "min_direction": 0.52, "min_mae_improvement": 0.02,
    "min_day_win_rate": 0.60, "min_leave_one_day_out_advantage_pp": 0.0,
    "min_interval_pairs": 100, "coverage80_min": 0.75, "coverage80_max": 0.95,
}
MONITOR_GATE = {
    "start": "first_trading_day_after_promotion", "min_rollback_days": 10,
    "min_rollback_pairs": 100, "rollback_mae_ratio": 1.02,
    "rollback_direction_gap": 0.05, "min_day_loss_rate": 0.60,
    "retain_days": 20, "retain_pairs": 200, "retain_mae_ratio": 0.98,
    "retain_day_win_rate": 0.60,
}
ALLOWED = {
    "shadow": {"eligible_for_review"},
    "eligible_for_review": {"shadow", "eligible_for_review", "promoted"},
    "promoted": {"monitored", "rolled_back"},
    "monitored": {"retained", "rolled_back"},
    "retained": {"rolled_back"},
    "rolled_back": {"shadow"},
}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dir(root):
    return Path(root) / "live_evidence" / "adoption"


def events(root=HERE):
    directory = _dir(root) / "events"
    previous, result = None, []
    for index, path in enumerate(sorted(directory.glob("*.json")), 1):
        if path.name != f"{index:06d}.json":
            raise ValueError("採用事件序號不連續")
        event = read_json(path)
        digest = event.get("event_sha256")
        body = {k: v for k, v in event.items() if k != "event_sha256"}
        if body.get("sequence") != index or body.get("previous_sha256") != previous or _sha(body) != digest:
            raise ValueError(f"採用事件鏈損壞：{path.name}")
        if body.get("from") != (result[-1]["to"] if result else "shadow"):
            raise ValueError(f"非法採用狀態來源：{path.name}")
        if body.get("to") not in ALLOWED[body["from"]]:
            raise ValueError(f"非法採用狀態轉換：{path.name}")
        previous = digest
        result.append(event)
    return result


def status(root=HERE):
    history = events(root)
    last = history[-1] if history else None
    promotion = (next((e for e in reversed(history) if e["to"] == "promoted"), None)
                 if last and last["to"] in {"promoted", "monitored", "retained", "rolled_back"} else None)
    return {"state": last["to"] if last else "shadow", "last_event": last,
            "promotion": promotion, "event_count": len(history)}


@contextmanager
def _lock(root):
    directory = _dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".lock"
    fd = None
    for _ in range(50):
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            break
        except FileExistsError:
            time.sleep(.1)
    if fd is None:
        raise RuntimeError("採用帳本正由另一流程更新；為避免競爭，拒絕轉換")
    try:
        yield
    finally:
        os.close(fd)
        path.unlink()


def transition(root, target, reason, evidence, now=None, expected_previous_sha256=None):
    """每筆事件獨立、不可覆寫；回傳已固定的事件。"""
    with _lock(root):
        current = status(root)
        source = current["state"]
        if (expected_previous_sha256 is not None and
                (current["last_event"]["event_sha256"] if current["last_event"] else None) != expected_previous_sha256):
            raise ValueError("採用事件已由另一流程更新，拒絕使用過期狀態")
        if target not in ALLOWED[source]:
            raise ValueError(f"不允許 {source} → {target}")
        event = {"schema_version": 1, "sequence": current["event_count"] + 1,
                 "at": (now or datetime.now(TZ)).astimezone(TZ).isoformat(),
                 "from": source, "to": target, "reason": reason,
                 "policy_version": POLICY_VERSION, "candidate_protocol": CANDIDATE_PROTOCOL,
                 "review_gate": REVIEW_GATE, "monitor_gate": MONITOR_GATE,
                 "previous_sha256": current["last_event"]["event_sha256"] if current["last_event"] else None,
                 "evidence": evidence}
        event["event_sha256"] = _sha(event)
        path = _dir(root) / "events" / f'{event["sequence"]:06d}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(event, ensure_ascii=False, indent=2, allow_nan=False)
        with path.open("x", encoding="utf-8") as out:
            out.write(content)
        return event


def _versions(root):
    root = Path(root)
    checkpoints = root / "models" / "checkpoints"
    baseline = {p.name: _file_sha(p) for p in sorted(checkpoints.glob("*")) if p.is_file()}
    if not baseline:
        raise ValueError("缺少可回復的原正式模型檔")
    rule_files = [root / "wiki" / "rules.json", root / "agents" / "strategy_agent.py",
                  root / "agents" / "rule_manager.py"]
    rules = {str(p.relative_to(root)): _file_sha(p) for p in rule_files if p.is_file()}
    production_code = [root / "models" / "predict.py", root / "models" / "lstm_model.py",
                       root / "config.py"]
    candidate_code = [root / "daily_retrain.py", root / "walkforward_training.py",
                      root / "adoption_state.py"]
    production_code_hashes = {str(p.relative_to(root)): _file_sha(p) for p in production_code if p.is_file()}
    candidate_code_hashes = {str(p.relative_to(root)): _file_sha(p) for p in candidate_code if p.is_file()}
    model_state = root / "models" / "daily_model_state.json"
    if not model_state.exists():
        raise ValueError("缺少候選模型狀態")
    manifest = root / read_json(model_state)["latest"] / "manifest.json"
    if not manifest.exists():
        raise ValueError("缺少候選模型清單")
    saved = read_json(manifest)
    if saved.get("protocol") != CANDIDATE_PROTOCOL:
        raise ValueError("候選模型協定不一致")
    for ticker, item in saved["models"].items():
        model = manifest.parent / f"{ticker}.pkl"
        if not model.exists() or _file_sha(model) != item["sha256"]:
            raise ValueError(f"候選模型雜湊不符：{ticker}")
    return {"candidate_manifest": str(manifest.relative_to(root)),
            "candidate_manifest_sha256": _file_sha(manifest),
            "production_model_sha256": baseline, "rule_sha256": rules,
            "production_code_sha256": production_code_hashes,
            "candidate_code_sha256": candidate_code_hashes}


def _snapshot_baseline(root, promotion_id, versions):
    root = Path(root)
    archive = root / "models" / "adoption_baselines" / promotion_id
    archive.mkdir(parents=True, exist_ok=False)
    expected = versions["production_model_sha256"]
    for name, digest in expected.items():
        source = root / "models" / "checkpoints" / name
        if not source.is_file() or _file_sha(source) != digest:
            raise ValueError(f"正式模型封存前已變動：{name}")
        with source.open("rb") as src, (archive / name).open("xb") as dst:
            shutil.copyfileobj(src, dst)
        if _file_sha(archive / name) != digest:
            raise ValueError(f"正式模型封存驗證失敗：{name}")
    manifest = archive / "manifest.json"
    with manifest.open("x", encoding="utf-8") as out:
        json.dump({"promotion_id": promotion_id, "production_model_sha256": expected}, out,
                  ensure_ascii=False, indent=2)
    return {"path": str(manifest.relative_to(root)), "sha256": _file_sha(manifest)}


def _restore_baseline(root, promotion):
    root = Path(root)
    versions = promotion["evidence"]["versions"]
    archive_info = promotion["evidence"]["baseline_archive"]
    manifest = root / archive_info["path"]
    if not manifest.is_file() or _file_sha(manifest) != archive_info["sha256"]:
        raise ValueError("原正式模型封存清單損壞，無法安全回滾")
    expected = versions["production_model_sha256"]
    if read_json(manifest)["production_model_sha256"] != expected:
        raise ValueError("原正式模型封存版本不符")
    # 全部先驗證，然後才開始回復，避免半套未驗證檔案進入正式目錄。
    for name, digest in expected.items():
        source = manifest.parent / name
        if not source.is_file() or _file_sha(source) != digest:
            raise ValueError(f"原正式模型封存檔損壞：{name}")
    restored = []
    for name, digest in expected.items():
        target = root / "models" / "checkpoints" / name
        if target.is_file() and _file_sha(target) == digest:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".restore-" + uuid4().hex + ".tmp")
        try:
            shutil.copyfile(manifest.parent / name, temporary)
            if _file_sha(temporary) != digest:
                raise ValueError(f"原正式模型回復副本不符：{name}")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        restored.append(name)
    return restored


def _review_evidence(root, pairs, decision):
    root = Path(root)
    window_days = sorted({p["date"] for p in pairs})[-REVIEW_GATE["lookback_days"]:]
    counts = {day: len({p["ticker"] for p in pairs if p["date"] == day}) for day in window_days}
    days = [day for day in window_days if counts[day] >= REVIEW_GATE["min_tickers_per_day"]]
    selected = [p for p in pairs if p["date"] in days and p.get("candidate_protocol") == CANDIDATE_PROTOCOL]
    if len(selected) != decision["n"]:
        raise ValueError("候選配對與門檻計分樣本不一致")
    prediction_files = {}
    versions = _versions(root)
    by_day = {}
    for item in selected:
        by_day.setdefault(item["date"], []).append(item)
    for day in days:
        path = root / "predictions" / f"{day}_point.json"
        if not path.exists():
            raise ValueError(f"缺少不可覆寫盤前來源：{day}")
        frozen = read_json(path)
        if {p["ticker"] for p in frozen["records"]} != set(__import__("config").ALL_STOCKS):
            raise ValueError(f"待審日的正式股票集合不完整：{day}")
        if frozen.get("model_sha256") != versions["production_model_sha256"]:
            raise ValueError(f"歷史正式模型版本與目前回退版本不同：{day}")
        points = {p["ticker"]: p for p in frozen["records"]}
        for row in by_day[day]:
            point = points.get(row["ticker"])
            candidate = point.get("daily_retrain", {}) if point else {}
            model_path = row.get("candidate_model_path")
            artifact_sha = row.get("candidate_artifact_sha256")
            if (not model_path or not artifact_sha or candidate.get("model_path") != model_path or
                    candidate.get("artifact_sha256") != artifact_sha or
                    candidate.get("protocol") != CANDIDATE_PROTOCOL or
                    candidate.get("predicted_change_pct") != row.get("predicted_pct")):
                raise ValueError(f"候選留底與計分配對不一致：{day} {row['ticker']}")
            model = (root / model_path).resolve()
            if not model.is_relative_to(root.resolve()) or not model.is_file() or _file_sha(model) != artifact_sha:
                raise ValueError(f"候選模型來源無法驗證：{day} {row['ticker']}")
        prediction_files[day] = _file_sha(path)
    return {"decision": decision, "sample_days": days, "target_data_end": days[-1],
            "pair_count": len(selected), "pair_sha256": _sha(selected), "pairs": selected,
            "prediction_sha256": prediction_files,
            "affected_tickers": sorted({p["ticker"] for p in selected}),
            "versions": versions}


def review(root=HERE, now=None):
    from daily_retrain import live_pairs, promotion_decision
    current = status(root)
    if current["state"] not in {"shadow", "eligible_for_review"}:
        return current
    pairs = live_pairs(root)
    decision = promotion_decision(pairs)
    if not decision["approved"]:
        if current["state"] == "eligible_for_review":
            transition(root, "shadow", "門檻已不再成立，撤銷待審", {"decision": decision}, now)
        return status(root)
    evidence = _review_evidence(root, pairs, decision)
    if current["state"] == "eligible_for_review" and current["last_event"]["evidence"] == evidence:
        return current
    transition(root, "eligible_for_review", "固定通過門檻的盤前並排實證，等待明確採用", evidence, now)
    return status(root)


def promote(root=HERE, expected_event_hash=None, now=None):
    """只有明確命令可晉升；必須指定使用者剛檢視的待審事件雜湊。"""
    current = status(root)
    if current["state"] != "eligible_for_review" or not expected_event_hash:
        raise ValueError("必須先有待審事件，並提供其完整 event_sha256")
    if current["last_event"]["event_sha256"] != expected_event_hash:
        raise ValueError("待審事件已變更，拒絕使用舊審核結果")
    fresh = review(root, now)
    if fresh["state"] != "eligible_for_review" or fresh["last_event"]["event_sha256"] != expected_event_hash:
        raise ValueError("實盤證據已變更或門檻失效，請重新審核")
    evidence = dict(current["last_event"]["evidence"])
    evidence["promotion_id"] = uuid4().hex
    evidence["monitor_starts_after"] = (now or datetime.now(TZ)).astimezone(TZ).date().isoformat()
    evidence["decision_policy"] = "每日新候選模型；所有股票同批驗證，失敗時回原正式模型"
    evidence["baseline_archive"] = _snapshot_baseline(root, evidence["promotion_id"], evidence["versions"])
    return transition(root, "promoted", "明確審核採用；下一交易日起進入獨立監測", evidence, now,
                      expected_previous_sha256=expected_event_hash)


def seal_post_market(root=HERE, now=None):
    """只封存當日且已發布的並排成績；重跑不得改寫舊結算。"""
    from daily_retrain import live_pairs
    now = (now or datetime.now(TZ)).astimezone(TZ)
    if now.hour < 14:
        raise ValueError("尚未收盤，不得結算採用政策")
    current = status(root)
    if current["state"] not in {"promoted", "monitored", "retained"}:
        return None
    promotion = current["promotion"]
    day = now.date().isoformat()
    if day <= promotion["evidence"]["monitor_starts_after"]:
        return None
    path = _dir(root) / "settlements" / f"{day}.json"
    if path.exists():
        return read_json(path)
    pid = promotion["evidence"]["promotion_id"]
    pairs = [p for p in live_pairs(root) if p["date"] == day and p.get("adoption_id") == pid]
    if not pairs:
        return None
    prediction = Path(root) / "predictions" / f"{day}_point.json"
    if not prediction.exists() or len(pairs) != len(read_json(prediction)["records"]):
        raise ValueError("採用後並排成績不完整，拒絕封存")
    body = {"schema_version": 1, "date": day, "sealed_at": now.isoformat(),
            "promotion_id": pid, "prediction_sha256": _file_sha(prediction),
            "affected_tickers": sorted({p["ticker"] for p in pairs}), "pairs": pairs}
    body["settlement_sha256"] = _sha(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8") as out:
        out.write(content)
    return body


def sealed_pairs(root, promotion):
    pid = promotion["evidence"]["promotion_id"]
    start = promotion["evidence"]["monitor_starts_after"]
    result = []
    for path in sorted((_dir(root) / "settlements").glob("*.json")):
        payload = read_json(path)
        digest = payload.get("settlement_sha256")
        body = {k: v for k, v in payload.items() if k != "settlement_sha256"}
        if path.name != f'{body.get("date")}.json' or _sha(body) != digest:
            raise ValueError(f"採用後結算已變動：{path.name}")
        if body["promotion_id"] != pid or body["date"] <= start:
            continue
        prediction = Path(root) / "predictions" / f'{body["date"]}_point.json'
        if not prediction.exists() or _file_sha(prediction) != body["prediction_sha256"]:
            raise ValueError(f"採用後盤前預測已變動：{body['date']}")
        if any(p.get("date") != body["date"] or p.get("adoption_id") != pid or
               p.get("candidate_protocol") != CANDIDATE_PROTOCOL for p in body["pairs"]):
            raise ValueError(f"採用後結算版本不符：{body['date']}")
        result.extend(body["pairs"])
    return result


def _metrics(pairs):
    if not pairs:
        return {"days": 0, "n": 0}
    n = len(pairs)
    mean = lambda key: sum(float(p[key]) for p in pairs) / n
    by_day = {}
    for row in pairs:
        by_day.setdefault(row["date"], []).append(row)
    day_wins = sum(sum(p["new_mae"] for p in rows) < min(
        sum(p["old_mae"] for p in rows), sum(p["zero_mae"] for p in rows))
        for rows in by_day.values())
    return {"days": len({p["date"] for p in pairs}), "n": n,
            "sample_days": sorted({p["date"] for p in pairs}),
            "day_win_rate": day_wins / len(by_day), "day_loss_rate": 1 - day_wins / len(by_day),
            "candidate_mae": mean("new_mae"), "production_baseline_mae": mean("old_mae"),
            "unchanged_baseline_mae": mean("zero_mae"),
            "candidate_direction": mean("new_correct"), "production_baseline_direction": mean("old_correct")}


def monitor(root=HERE, now=None):
    current = status(root)
    if current["state"] not in {"promoted", "monitored", "retained"}:
        return current
    promotion = current["promotion"]
    pairs = sealed_pairs(root, promotion)
    metrics = _metrics(pairs)
    if not pairs:
        return current
    evidence = {"promotion_id": promotion["evidence"]["promotion_id"], "metrics": metrics,
                "pair_sha256": _sha(pairs), "sample_days": metrics["sample_days"],
                "affected_tickers": sorted({p["ticker"] for p in pairs}),
                "versions": promotion["evidence"]["versions"]}
    worse_mae = metrics["candidate_mae"] > MONITOR_GATE["rollback_mae_ratio"] * min(
        metrics["production_baseline_mae"], metrics["unchanged_baseline_mae"])
    worse_direction = metrics["candidate_direction"] < (metrics["production_baseline_direction"] -
                                                      MONITOR_GATE["rollback_direction_gap"])
    if (metrics["days"] >= MONITOR_GATE["min_rollback_days"] and
            metrics["n"] >= MONITOR_GATE["min_rollback_pairs"] and
            ((worse_mae and metrics["day_loss_rate"] >= MONITOR_GATE["min_day_loss_rate"]) or
             worse_direction)):
        rollback(root, "獨立監測低於基準，恢復原正式模型", now, evidence)
    elif current["state"] == "promoted":
        transition(root, "monitored", "第一個採用後實盤樣本已結算", evidence, now)
    elif (current["state"] == "monitored" and metrics["days"] >= MONITOR_GATE["retain_days"] and
          metrics["n"] >= MONITOR_GATE["retain_pairs"] and
          metrics["day_win_rate"] >= MONITOR_GATE["retain_day_win_rate"] and
          metrics["candidate_mae"] <= MONITOR_GATE["retain_mae_ratio"] * min(
              metrics["production_baseline_mae"], metrics["unchanged_baseline_mae"]) and
          metrics["candidate_direction"] >= metrics["production_baseline_direction"]):
        transition(root, "retained", "獨立監測窗持續勝過基準", evidence, now)
    return status(root)


def rollback(root=HERE, reason="人工要求回復", now=None, monitoring_evidence=None):
    current = status(root)
    if current["state"] not in {"promoted", "monitored", "retained"}:
        raise ValueError("沒有已採用政策可回滾")
    promotion = current["promotion"]
    restored = _restore_baseline(root, promotion)
    evidence = {"promotion_id": promotion["evidence"]["promotion_id"],
                "restore_versions": promotion["evidence"]["versions"],
                "restored_files": restored}
    if monitoring_evidence:
        evidence["monitoring"] = monitoring_evidence
    return transition(root, "rolled_back", reason, evidence, now)


def restart(root=HERE, now=None):
    """上一輪回滾後，只有全新候選協定及完整模型版本才能重新進入 shadow。"""
    current = status(root)
    if current["state"] != "rolled_back":
        raise ValueError("只有已回滾政策可開始新一輪影子驗證")
    previous = current["promotion"]
    if CANDIDATE_PROTOCOL == previous["candidate_protocol"]:
        raise ValueError("新一輪必須使用不同候選協定版本，禁止重用已回滾政策")
    versions = _versions(root)
    return transition(root, "shadow", "新候選版本開始獨立影子驗證",
                      {"previous_promotion_id": previous["evidence"]["promotion_id"],
                       "new_versions": versions}, now,
                      expected_previous_sha256=current["last_event"]["event_sha256"])


def route(predictions, root=HERE, now=None):
    """同批模型與資料全數有效才切換；任何缺漏回滾並保留原預測。"""
    now = now or datetime.now(TZ)
    current = status(root)
    if current["state"] not in {"promoted", "monitored", "retained"}:
        return {"adopted": False, "state": current["state"]}
    promotion = current["promotion"]
    pid = promotion["evidence"]["promotion_id"]
    try:
        if now.date().isoformat() <= promotion["evidence"]["monitor_starts_after"]:
            return {"adopted": False, "state": current["state"], "reason": "採用從下一交易日開始"}
        if not predictions or {p["ticker"] for p in predictions} != set(__import__("config").ALL_STOCKS):
            raise ValueError("正式股票集合不完整")
        expected_models = promotion["evidence"]["versions"]["production_model_sha256"]
        for name, digest in expected_models.items():
            path = Path(root) / "models" / "checkpoints" / name
            if not path.exists() or _file_sha(path) != digest:
                raise ValueError("原正式模型版本已變動，無法安全回滾")
        for point in predictions:
            candidate = point.get("daily_retrain")
            if not isinstance(candidate, dict) or not valid_daily_candidate(
                    dict(point, generated_at=now.isoformat(), target_date=now.date().isoformat(), horizon_sessions=1),
                    now.date().isoformat()):
                raise ValueError(f'{point["ticker"]} 缺少有效盤前候選')
            if candidate.get("protocol") != CANDIDATE_PROTOCOL or candidate.get("last_close") != point.get("last_close"):
                raise ValueError(f'{point["ticker"]} 候選版本／基準價不符')
            if not {"predicted_change_pct", "predicted_close", "direction", "model_path", "artifact_sha256"} <= candidate.keys():
                raise ValueError(f'{point["ticker"]} 候選正式輸出不完整')
            model = Path(root) / candidate["model_path"]
            if not model.exists() or _file_sha(model) != candidate["artifact_sha256"]:
                raise ValueError(f'{point["ticker"]} 候選模型雜湊不符')
        current_rules = _versions_rules(root)
        current_production_code = _code_versions(root, ("models/predict.py", "models/lstm_model.py", "config.py"))
        current_candidate_code = _code_versions(root, ("daily_retrain.py", "walkforward_training.py", "adoption_state.py"))
        fields = ("predicted_change_pct", "predicted_close", "direction", "q25_pct", "q75_pct", "q10_pct", "q90_pct")
        for point in predictions:
            candidate = point["daily_retrain"]
            point["production_before_retrain"] = {key: point[key] for key in
                ("ticker", "name", "data_as_of", "last_close", "predicted_change_pct", "predicted_close", "direction",
                 "confidence", "model_type")
                if key in point}
            for stale in ("confidence", "model_confidence", "p_up", "lstm_weight", "transformer_weight",
                          "lstm_pct", "transformer_pct", "predicted_direction"):
                point.pop(stale, None)
            point.update({key: candidate[key] for key in fields if key in candidate})
            point["model_type"] = CANDIDATE_PROTOCOL
            point["magnitude"] = candidate["predicted_change_pct"]
            point["evidence_status"] = "promoted_monitored"
            point["adoption"] = {"promotion_id": pid, "policy_version": POLICY_VERSION,
                                 "candidate_protocol": CANDIDATE_PROTOCOL,
                                 "artifact_sha256": candidate["artifact_sha256"],
                                 "baseline_model_sha256": expected_models,
                                 "rule_sha256": current_rules,
                                 "production_code_sha256": current_production_code,
                                 "candidate_code_sha256": current_candidate_code,
                                 "promotion_event_sha256": promotion["event_sha256"]}
        return {"adopted": True, "state": current["state"], "promotion_id": pid}
    except (KeyError, ValueError, TypeError) as error:
        rollback(root, f"盤前候選／回退版本校驗失敗：{error}", now)
        return {"adopted": False, "state": "rolled_back", "reason": str(error),
                "halt_publication": "原正式模型版本已變動" in str(error)}


def _versions_rules(root):
    root = Path(root)
    paths = [root / "wiki" / "rules.json", root / "agents" / "strategy_agent.py",
             root / "agents" / "rule_manager.py"]
    return {str(p.relative_to(root)): _file_sha(p) for p in paths if p.is_file()}


def _code_versions(root, paths):
    root = Path(root)
    return {name: _file_sha(root / name) for name in paths if (root / name).is_file()}


def save_strategies(strategies, predictions, root=HERE, now=None):
    adopted = [p for p in predictions if p.get("adoption")]
    if not adopted:
        return None
    if {s.get("ticker") for s in strategies} != {p["ticker"] for p in adopted}:
        raise ValueError("正式策略與採用股票集合不一致")
    now = now or datetime.now(TZ)
    path = Path(root) / "live_evidence" / "adoption" / "strategies" / f"{now.date().isoformat()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": now.isoformat(), "target_date": now.date().isoformat(),
               "promotion_id": adopted[0]["adoption"]["promotion_id"],
               "prediction_sha256": _file_sha(Path(root) / "predictions" / f"{now.date().isoformat()}_point.json"),
               "strategies": strategies}
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    with path.open("x", encoding="utf-8") as out:
        out.write(content)
    return path


def main():
    parser = argparse.ArgumentParser(description="每日候選採用狀態與可稽核轉換")
    parser.add_argument("action", choices=["status", "review", "promote", "monitor", "rollback", "restart"])
    parser.add_argument("--expected-event-hash")
    parser.add_argument("--reason", default="人工要求回復")
    args = parser.parse_args()
    actions = {"status": status, "review": review, "monitor": monitor}
    if args.action == "promote":
        result = promote(expected_event_hash=args.expected_event_hash)
    elif args.action == "rollback":
        result = rollback(reason=args.reason)
    elif args.action == "restart":
        result = restart()
    else:
        result = actions[args.action]()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
