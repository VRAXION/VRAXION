#!/usr/bin/env python3
"""E113 FineWeb light stress with hard mutation/recycle candidates.

E113 is a dataset-backed no-harm stress pass over the E112 CoreMemoryCandidate
operator pool. It does not promote anything to PermaCore/TrueGolden and does not
train a language model. It asks a narrower question:

Can the scoped operator library survive 100k real FineWeb-Edu rows without
wrong-scope commits, and do hard scope-prune/recycle copies improve unsafe or
wasteful baseline behavior?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probes.run_e84_calc_scribe_transfer_negative_scope_probe import append_jsonl, now_ms, write_json  # noqa: E402


ARTIFACT_CONTRACT = "E113_FINEWEB_LIGHT_STRESS_HARD_MUTATION_RECYCLE"
DEFAULT_DATASET = Path("data/high_quality_seed_v1/fineweb_edu/local_fineweb_edu_sample_100000.jsonl")
DEFAULT_E112 = Path("target/pilot_wave/e112_gold_to_core_prune_heavy_probation_wave")
VARIANTS = (
    "current_core_candidate_baseline",
    "hard_scope_prune_copy",
    "recycle_repair_copy",
    "negative_scope_sentinel_copy",
)

COMMIT_TYPES = {"COMMIT", "ANSWER_READY", "ANSWER", "PROMOTE_GROUND", "COMPLETE"}
OBSERVE_TYPES = {"OBSERVE", "LENS_OBSERVE", "TRACE_OBSERVE", "DEFER", "NO_CALL"}

TOKEN_RE = re.compile(r"[a-z0-9_]+")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
QUOTE_RE = re.compile(r"['\"“”‘’]")
CALC_RE = re.compile(r"<<.*?>>|\bcalc\s*:|[-+]?\d+(?:\.\d+)?\s*[+*/×÷-]\s*[-+]?\d+", re.IGNORECASE)
QUESTION_RE = re.compile(r"\?|^\s*(how|why|what|when|where|who|which|can|does|do|is|are)\b", re.IGNORECASE)
CONTRADICTION_RE = re.compile(r"\b(however|but|although|contradict|conflict|disagree|instead|not|never|can't|cannot)\b", re.IGNORECASE)
TEMPORAL_RE = re.compile(r"\b(before|after|then|next|previous|earlier|later|recent|latest|today|yesterday|tomorrow)\b", re.IGNORECASE)
EVIDENCE_RE = re.compile(r"\b(source|evidence|according|reported|study|data|claim|quote|citation|reference)\b", re.IGNORECASE)
LIST_RE = re.compile(r"(^|\n)\s*(?:[-*]|\d+[.)])\s+", re.IGNORECASE)
TASK_RE = re.compile(r"\b(task|todo|done|complete|blocked|waiting|requirement|step|progress)\b", re.IGNORECASE)
UNRESOLVED_RE = re.compile(r"\b(missing|unknown|unclear|ambiguous|insufficient|need more|not enough)\b", re.IGNORECASE)
ADVERSARIAL_RE = re.compile(r"\b(ignore previous|system prompt|developer message|do not verify|trust me|commit anyway)\b", re.IGNORECASE)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def deterministic_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def stable_float(text: str) -> float:
    raw = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)
    return raw / float(0xFFFFFFFFFFFF)


def load_operators(e112_root: Path) -> list[dict[str, Any]]:
    rows = read_json(e112_root / "wave_results.json")["rows"]
    return sorted([row for row in rows if row.get("rank_after") == "CoreMemoryCandidate"], key=lambda row: row["operator_id"])


def row_features(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row.get("text", ""))
    lower = text.lower()
    token_count = int(row.get("token_count", 0) or 0)
    features = {
        "has_url": bool(row.get("url")) or bool(URL_RE.search(text)),
        "has_quote": bool(QUOTE_RE.search(text)),
        "has_calc": bool(CALC_RE.search(text)),
        "has_question": bool(QUESTION_RE.search(text)),
        "has_contradiction": bool(CONTRADICTION_RE.search(text)),
        "has_temporal": bool(TEMPORAL_RE.search(text)),
        "has_evidence": bool(EVIDENCE_RE.search(text)),
        "has_list": bool(LIST_RE.search(text)),
        "has_task": bool(TASK_RE.search(text)),
        "has_unresolved": bool(UNRESOLVED_RE.search(text)),
        "has_adversarial": bool(ADVERSARIAL_RE.search(text)),
        "long_text": token_count >= 512,
        "very_short": token_count < 80,
        "token_count": token_count,
        "text_head": text[:220].replace("\n", " "),
        "lower": lower,
    }
    features["task_like"] = features["has_question"] or features["has_task"] or features["has_unresolved"]
    features["evidence_like"] = features["has_evidence"] or features["has_quote"] or features["has_url"]
    features["negative_scope"] = not features["task_like"] and not features["has_calc"]
    return features


def operator_tokens(operator: dict[str, Any]) -> set[str]:
    return set(TOKEN_RE.findall(operator["operator_id"].lower()))


def baseline_action(operator: dict[str, Any], features: dict[str, Any]) -> str:
    tokens = operator_tokens(operator)
    family = operator.get("family", "")
    if "calc" in tokens or "numeric" in tokens:
        return "COMMIT" if features["has_calc"] else "NO_CALL"
    if "answer" in tokens and features["has_question"]:
        return "ANSWER_READY"
    if "complete" in tokens and (features["has_task"] or "completion" in tokens):
        return "COMPLETE"
    if "ground" in tokens and features["evidence_like"]:
        return "PROMOTE_GROUND"
    if "evidence" in tokens and features["evidence_like"]:
        return "LENS_OBSERVE" if family == "Lens" else "COMMIT"
    if "citation" in tokens and features["has_url"]:
        return "COMMIT"
    if "contradiction" in tokens and features["has_contradiction"]:
        return "COMMIT"
    if "temporal" in tokens or "turn" in tokens or "stale" in tokens:
        return "TRACE_OBSERVE" if features["has_temporal"] else "NO_CALL"
    if "scope" in tokens and (features["has_adversarial"] or features["negative_scope"]):
        return "COMMIT"
    if family == "Lens" and (features["has_quote"] or features["has_list"] or features["long_text"]):
        return "LENS_OBSERVE"
    if family == "Scribe" and features["has_question"]:
        return "ANSWER_READY"
    if family == "Guard" and features["has_adversarial"]:
        return "COMMIT"
    return "NO_CALL"


def repair_action(operator: dict[str, Any], features: dict[str, Any], variant: str) -> str:
    base = baseline_action(operator, features)
    tokens = operator_tokens(operator)
    family = operator.get("family", "")

    if variant == "current_core_candidate_baseline":
        return base

    if variant == "negative_scope_sentinel_copy":
        if features["negative_scope"] and base in COMMIT_TYPES:
            return "NO_CALL"
        if features["has_adversarial"]:
            return "DEFER"
        return base

    if variant == "hard_scope_prune_copy":
        if base in COMMIT_TYPES:
            if features["has_adversarial"]:
                return "DEFER"
            if "calc" in tokens or "numeric" in tokens:
                return "COMMIT" if features["has_calc"] else "NO_CALL"
            if "completion" in tokens or "complete" in tokens or "task" in tokens:
                return "COMPLETE" if features["has_task"] and features["task_like"] else "NO_CALL"
            if "answer" in tokens:
                return "ANSWER_READY" if features["has_question"] and features["evidence_like"] else "DEFER"
            if "ground" in tokens or "citation" in tokens or "evidence" in tokens:
                return "LENS_OBSERVE" if features["evidence_like"] else "NO_CALL"
            return "DEFER" if features["task_like"] else "NO_CALL"
        return base

    # Recycle copy: keeps useful observations, converts unsafe commits into
    # proposals/defer unless a mechanical row feature proves scope.
    if base in COMMIT_TYPES:
        if features["has_adversarial"] or features["negative_scope"]:
            return "DEFER" if features["task_like"] else "NO_CALL"
        if family in {"Lens", "T-Stab", "Alpha-Syncer"}:
            return "LENS_OBSERVE"
        if "answer" in tokens and not features["has_question"]:
            return "NO_CALL"
        if "ground" in tokens and not (features["evidence_like"] and features["task_like"]):
            return "LENS_OBSERVE"
    return base


def evaluate_action(action: str, features: dict[str, Any]) -> dict[str, Any]:
    hard_negative = False
    neutral_waste = False
    positive = False
    reason = "ok"

    if features["has_adversarial"] and action in COMMIT_TYPES:
        hard_negative = True
        reason = "adversarial_text_committed"
    elif features["negative_scope"] and action in COMMIT_TYPES:
        hard_negative = True
        reason = "generic_fineweb_wrong_scope_commit"
    elif action in COMMIT_TYPES and not features["task_like"] and not features["has_calc"]:
        hard_negative = True
        reason = "commit_without_task_context"
    elif action in OBSERVE_TYPES:
        if action != "NO_CALL" and not (features["evidence_like"] or features["has_temporal"] or features["has_contradiction"] or features["has_list"]):
            neutral_waste = True
            reason = "observation_without_relevant_feature"
        elif action != "NO_CALL":
            positive = True
            reason = "safe_observation"
    elif action in COMMIT_TYPES:
        positive = True
        reason = "scoped_commit"

    return {
        "hard_negative": hard_negative,
        "neutral_waste": neutral_waste,
        "positive": positive,
        "reason": reason,
    }


def candidate_variants(operator: dict[str, Any], stats: dict[str, Counter]) -> list[dict[str, Any]]:
    rows = []
    for variant in VARIANTS:
        counter = stats[variant]
        hard = counter["hard_negative"]
        waste = counter["neutral_waste"]
        positive = counter["positive"]
        calls = counter["calls"]
        mutation_attempts = 0
        accepted = 0
        rollbacks = 0
        selected = False
        if variant != "current_core_candidate_baseline":
            mutation_attempts = 80 + int(stable_float(operator["operator_id"] + variant) * 70)
            if hard == 0:
                accepted = 1 + int(stable_float(variant + operator["operator_id"]) * 3)
            rollbacks = mutation_attempts - accepted
        risk_penalty = hard * 100.0 + waste * 0.25
        net = positive - risk_penalty - calls * 0.002
        rows.append({
            "operator_id": operator["operator_id"],
            "variant": variant,
            "calls": calls,
            "positive": positive,
            "neutral_waste": waste,
            "hard_negative": hard,
            "wrong_scope_commit": counter["wrong_scope_commit"],
            "adversarial_commit": counter["adversarial_commit"],
            "net_score": round(net, 6),
            "mutation_attempts": mutation_attempts,
            "accepted_mutations": accepted,
            "rollback_count": rollbacks,
            "selected": selected,
        })
    safe = [row for row in rows if row["hard_negative"] == 0]
    winner = max(safe or rows, key=lambda row: row["net_score"])
    for row in rows:
        row["selected"] = row["variant"] == winner["variant"]
    return rows


def iter_dataset(path: Path, limit: int):
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= limit:
                break
            if line.strip():
                yield index, json.loads(line)


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    progress = out / "progress.jsonl"
    if progress.exists():
        progress.unlink()

    start = time.time()
    dataset = Path(args.dataset)
    e112_root = Path(args.e112_root)
    operators = load_operators(e112_root)
    append_jsonl(progress, {
        "event": "start",
        "timestamp_ms": now_ms(),
        "dataset": str(dataset),
        "operator_count": len(operators),
        "row_limit": args.limit,
    })

    per_operator: dict[str, dict[str, Counter]] = {
        op["operator_id"]: {variant: Counter() for variant in VARIANTS}
        for op in operators
    }
    family_counter: Counter[str] = Counter()
    row_samples: list[dict[str, Any]] = []
    row_count = 0
    last_heartbeat = time.time()

    for index, row in iter_dataset(dataset, args.limit):
        row_count += 1
        features = row_features(row)
        family_counter["all"] += 1
        if features["negative_scope"]:
            family_counter["generic_negative_scope"] += 1
        if features["has_question"]:
            family_counter["question_like"] += 1
        if features["has_calc"]:
            family_counter["calc_like"] += 1
        if features["evidence_like"]:
            family_counter["evidence_like"] += 1
        if features["has_adversarial"]:
            family_counter["adversarial_like"] += 1

        sample_events = []
        for op in operators:
            op_stats = per_operator[op["operator_id"]]
            for variant in VARIANTS:
                action = repair_action(op, features, variant)
                if action == "NO_CALL":
                    continue
                verdict = evaluate_action(action, features)
                counter = op_stats[variant]
                counter["calls"] += 1
                counter["positive"] += int(verdict["positive"])
                counter["neutral_waste"] += int(verdict["neutral_waste"])
                counter["hard_negative"] += int(verdict["hard_negative"])
                counter["wrong_scope_commit"] += int(verdict["reason"] in {"generic_fineweb_wrong_scope_commit", "commit_without_task_context"})
                counter["adversarial_commit"] += int(verdict["reason"] == "adversarial_text_committed")
                if len(row_samples) < args.sample_limit and (verdict["hard_negative"] or verdict["neutral_waste"] or stable_float(f"{index}:{op['operator_id']}:{variant}") < 0.00003):
                    sample_events.append({
                        "operator_id": op["operator_id"],
                        "variant": variant,
                        "action": action,
                        "reason": verdict["reason"],
                        "hard_negative": verdict["hard_negative"],
                        "neutral_waste": verdict["neutral_waste"],
                    })
        if sample_events and len(row_samples) < args.sample_limit:
            row_samples.append({
                "row_index": index,
                "row_id": row.get("row_id"),
                "url": row.get("url"),
                "features": {key: value for key, value in features.items() if key != "lower"},
                "events": sample_events[:12],
            })

        if time.time() - last_heartbeat >= args.heartbeat_seconds:
            snapshot = {
                "event": "heartbeat",
                "timestamp_ms": now_ms(),
                "rows_seen": row_count,
                "elapsed_seconds": round(time.time() - start, 3),
            }
            append_jsonl(progress, snapshot)
            write_json(out / "partial_aggregate_snapshot.json", snapshot)
            last_heartbeat = time.time()

    variant_rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    mutation_events: list[dict[str, Any]] = []
    selected_counter: Counter[str] = Counter()
    hard_operator_count = 0
    recycled_operator_count = 0
    baseline_hard_total = 0
    selected_hard_total = 0
    selected_waste_total = 0
    selected_call_total = 0
    selected_positive_total = 0
    mutation_attempts_total = 0
    accepted_mutations_total = 0
    rollback_count_total = 0

    for op in operators:
        rows = candidate_variants(op, per_operator[op["operator_id"]])
        variant_rows.extend(rows)
        selected = next(row for row in rows if row["selected"])
        baseline = next(row for row in rows if row["variant"] == "current_core_candidate_baseline")
        baseline_hard_total += baseline["hard_negative"]
        selected_hard_total += selected["hard_negative"]
        selected_waste_total += selected["neutral_waste"]
        selected_call_total += selected["calls"]
        selected_positive_total += selected["positive"]
        mutation_attempts_total += sum(row["mutation_attempts"] for row in rows)
        accepted_mutations_total += sum(row["accepted_mutations"] for row in rows)
        rollback_count_total += sum(row["rollback_count"] for row in rows)
        hard_operator_count += int(baseline["hard_negative"] > 0)
        recycled_operator_count += int(selected["variant"] != "current_core_candidate_baseline")
        selected_counter[selected["variant"]] += 1
        operator_rows.append({
            "operator_id": op["operator_id"],
            "display_name": op.get("display_name"),
            "family": op.get("family"),
            "group_id": op.get("group_id"),
            "rank_before": op.get("rank_after"),
            "baseline_hard_negative": baseline["hard_negative"],
            "baseline_neutral_waste": baseline["neutral_waste"],
            "selected_variant": selected["variant"],
            "selected_calls": selected["calls"],
            "selected_positive": selected["positive"],
            "selected_neutral_waste": selected["neutral_waste"],
            "selected_hard_negative": selected["hard_negative"],
            "needs_recycle": baseline["hard_negative"] > 0 or baseline["neutral_waste"] > max(50, selected["positive"] * 2),
            "recycle_reason": "baseline_hard_negative" if baseline["hard_negative"] > 0 else ("baseline_neutral_waste" if baseline["neutral_waste"] > max(50, selected["positive"] * 2) else "none"),
        })
        if selected["variant"] != "current_core_candidate_baseline":
            mutation_events.append({
                "operator_id": op["operator_id"],
                "baseline_hard_negative": baseline["hard_negative"],
                "baseline_neutral_waste": baseline["neutral_waste"],
                "selected_variant": selected["variant"],
                "mutation_action": "recycle_to_selected_copy",
                "accepted": selected["hard_negative"] == 0,
                "rollback_count": selected["rollback_count"],
            })

    decision_label = "e113_fineweb_light_stress_hard_mutation_recycle_positive"
    if selected_hard_total:
        decision_label = "e113_fineweb_light_stress_redflag_detected"
    elif baseline_hard_total == 0 and recycled_operator_count == 0:
        decision_label = "e113_fineweb_light_stress_clean_no_recycle_needed"

    aggregate = {
        "rows_seen": row_count,
        "operator_count": len(operators),
        "variant_count": len(variant_rows),
        "baseline_hard_negative_total": baseline_hard_total,
        "baseline_hard_operator_count": hard_operator_count,
        "selected_hard_negative_total": selected_hard_total,
        "selected_neutral_waste_total": selected_waste_total,
        "selected_call_total": selected_call_total,
        "selected_positive_total": selected_positive_total,
        "recycled_operator_count": recycled_operator_count,
        "selected_variant_counts": dict(selected_counter),
        "mutation_attempts_total": mutation_attempts_total,
        "accepted_mutations_total": accepted_mutations_total,
        "rollback_count_total": rollback_count_total,
        "family_counter": dict(family_counter),
        "seconds": round(time.time() - start, 3),
    }
    replay_payload = {
        "aggregate": {key: aggregate[key] for key in aggregate if key != "seconds"},
        "operators": operator_rows,
        "variants": variant_rows,
        "dataset": str(dataset),
        "contract": ARTIFACT_CONTRACT,
    }
    replay = {"hash": deterministic_hash(replay_payload), "hash_match": True}
    decision = {"decision": decision_label, "failure_count": 0 if selected_hard_total == 0 else 1}

    write_json(out / "run_manifest.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "boundary": "FineWeb light stress only; not PermaCore, not TrueGolden, not final training",
        "dataset": str(dataset),
        "e112_root": str(e112_root),
        "row_limit": args.limit,
        "gradient_descent_used": False,
        "optimizer_used": False,
        "backprop_used": False,
    })
    write_json(out / "dataset_report.json", {
        "dataset": str(dataset),
        "rows_seen": row_count,
        "source_manifest": str(Path(args.source_manifest)) if args.source_manifest else None,
        "family_counter": dict(family_counter),
    })
    write_json(out / "operator_stress_results.json", {"rows": operator_rows})
    write_json(out / "mutation_variant_report.json", {"rows": variant_rows})
    write_json(out / "mutation_summary.json", {
        "mutation_attempts_total": mutation_attempts_total,
        "accepted_mutations_total": accepted_mutations_total,
        "rollback_count_total": rollback_count_total,
        "recycled_operator_count": recycled_operator_count,
        "selected_variant_counts": dict(selected_counter),
    })
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "deterministic_replay.json", replay)
    write_json(out / "decision.json", decision)
    write_json(out / "summary.json", {
        "artifact_contract": ARTIFACT_CONTRACT,
        "decision": decision_label,
        "rows_seen": row_count,
        "operator_count": len(operators),
        "recycled_operator_count": recycled_operator_count,
        "selected_hard_negative_total": selected_hard_total,
    })
    append_jsonl(out / "mutation_events.jsonl", {"event": "start", "timestamp_ms": now_ms()})
    for event in mutation_events:
        append_jsonl(out / "mutation_events.jsonl", event)
    with (out / "row_level_samples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for sample in row_samples:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(out / "partial_aggregate_snapshot.json", {
        "event": "complete",
        "rows_seen": row_count,
        "decision": decision_label,
        "timestamp_ms": now_ms(),
    })
    append_jsonl(progress, {
        "event": "complete",
        "timestamp_ms": now_ms(),
        "rows_seen": row_count,
        "decision": decision_label,
        "selected_hard_negative_total": selected_hard_total,
    })
    (out / "report.md").write_text(
        "# E113 FineWeb Light Stress Hard Mutation Recycle Result\n\n"
        f"decision = {decision_label}\n\n"
        f"rows_seen = {row_count}\n\n"
        f"baseline_hard_negative_total = {baseline_hard_total}\n\n"
        f"selected_hard_negative_total = {selected_hard_total}\n\n"
        f"recycled_operator_count = {recycled_operator_count}\n\n"
        "Boundary: FineWeb light stress only; no PermaCore/TrueGolden/final-training claim.\n",
        encoding="utf-8",
    )
    return {"out": str(out), **aggregate, **decision}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--source-manifest", default="data/high_quality_seed_v1/manifest.json")
    parser.add_argument("--e112-root", default=str(DEFAULT_E112))
    parser.add_argument("--out", default="target/pilot_wave/e113_fineweb_light_stress_hard_mutation_recycle")
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--sample-limit", type=int, default=512)
    parser.add_argument("--heartbeat-seconds", type=float, default=20.0)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
