#!/usr/bin/env python3
"""D60S Rust sparse mutation safety/no-forgetting fitness repair."""

import argparse
import copy
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import run_d51_mutable_ecf_controller_prototype as d51
import run_d55_sparse_firing_ecf_controller_prototype as d55
import run_d59_rust_sparse_ecf_controller_with_mutation as d59
import run_d60_rust_sparse_mutation_learning_signal_probe as d60

TASK = "D60S_RUST_SPARSE_MUTATION_SAFETY_FITNESS_REPAIR"
BOUNDARY = (
    "D60S only tests safety/no-forgetting fitness repair for mutation of a canonical Rust "
    "sparse ECF action controller in controlled symbolic joint formula discovery. It does "
    "not prove full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI, "
    "consciousness, DNA/genome success, architecture superiority, or production readiness."
)

PRIMARY_SPACE = d60.PRIMARY_SPACE
SUPPORT_COUNT = d60.SUPPORT_COUNT
REGIMES = d60.REGIMES
CORE_REGIMES = d60.CORE_REGIMES
ACTIONS = d60.ACTIONS
FEATURE_NAMES = d60.FEATURE_NAMES
ROW_SAMPLE_PER_ARM_REGIME_SPLIT = d60.ROW_SAMPLE_PER_ARM_REGIME_SPLIT

SATURATED_TRACK = "SATURATED_STABILITY"
HARD_TRACK = "HARD_CAP8_LEARNING"
MIXED_TRACK = "MIXED_EVAL"
TRACKS = [SATURATED_TRACK, HARD_TRACK, MIXED_TRACK]
HARD_VARIANT = {"name": "support_budget_cap_8", "support_budget_cap": 8}

RUST_CONTROLLER_ARMS = [
    "D59_REFERENCE",
    "D60_HARD_BEST_REPLAY",
    "SINGLE_POLICY_MULTI_ENV_FITNESS",
    "LEXICOGRAPHIC_SAFETY_FIRST_FITNESS",
    "PARETO_MULTI_ENV_MUTATION",
    "STABILITY_REGULARIZED_MUTATION",
    "COST_ONLY_MUTATION_CONTROL",
    "ACCURACY_ONLY_MUTATION_CONTROL",
    "RANDOM_MUTATION_CONTROL",
    "MUTATION_DISABLED_CONTROL",
    "THRESHOLD_ABLATION",
    "REWIRE_ABLATION",
]

GATED_ARMS = [
    "DUAL_POLICY_GATED_CONTROLLER",
    "CONTEXT_GATED_POLICY_ENSEMBLE",
]

CONTROL_ARMS = [
    "RANDOM_POLICY_CONTROL",
    "GREEDY_DECIDE_CONTROL",
    "ALWAYS_COUNTER_CONTROL",
    "SPIKE_SHUFFLE_CONTROL",
]

TRAINED_ARMS = [
    "SINGLE_POLICY_MULTI_ENV_FITNESS",
    "LEXICOGRAPHIC_SAFETY_FIRST_FITNESS",
    "PARETO_MULTI_ENV_MUTATION",
    "STABILITY_REGULARIZED_MUTATION",
]

ARMS = RUST_CONTROLLER_ARMS + GATED_ARMS + CONTROL_ARMS


def write_json(path, value):
    d51.write_json(path, value)


def append_jsonl(path, value):
    d51.append_jsonl(path, value)


def append_progress(out, event, started, data):
    d51.append_progress(out, event, started, data)


def parse_seeds(raw):
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def stable_rng(seed, tag):
    return random.Random(int(seed) + d51.stable_seed(tag))


def make_d60_upstream_manifest(repo_root):
    root = repo_root / "target/pilot_wave/d60_rust_sparse_mutation_learning_signal_probe/smoke"
    manifest = {
        "upstream": "D60_RUST_SPARSE_MUTATION_LEARNING_SIGNAL_PROBE",
        "expected_decision": "rust_sparse_mutation_safety_failure",
        "expected_next": "D60S_SAFETY_FITNESS_REPAIR",
        "artifact_root": str(root),
        "decision_present": (root / "decision.json").exists(),
        "summary_present": (root / "summary.json").exists(),
        "trained_policy_manifest_present": (root / "trained_policy_manifest.json").exists(),
    }
    for name in [
        "decision.json",
        "summary.json",
        "hard_learning_track_report.json",
        "saturated_track_report.json",
        "oracle_upper_bound_report.json",
    ]:
        value = load_json(root / name)
        if value is not None:
            manifest[name.replace(".json", "")] = value
    trained = load_json(root / "trained_policy_manifest.json") or {}
    best = (manifest.get("decision") or {}).get("best_arm")
    manifest["d60_best_controller_loaded"] = bool(best and trained.get("controllers", {}).get(best))
    return manifest


def load_d59_reference(repo_root):
    root = repo_root / "target/pilot_wave/d59_rust_sparse_ecf_controller_with_mutation/smoke_full"
    trained = load_json(root / "trained_policy_manifest.json") or {}
    decision = load_json(root / "decision.json") or {}
    best = decision.get("best_arm")
    if best and trained.get("controllers", {}).get(best):
        return trained["controllers"][best], best
    controller = d60.load_d58_controller(repo_root)
    return controller, "D58_FALLBACK"


def load_d60_controller(repo_root, arm=None):
    root = repo_root / "target/pilot_wave/d60_rust_sparse_mutation_learning_signal_probe/smoke"
    trained = load_json(root / "trained_policy_manifest.json") or {}
    decision = load_json(root / "decision.json") or {}
    chosen = arm or decision.get("best_arm")
    if chosen and trained.get("controllers", {}).get(chosen):
        return trained["controllers"][chosen], chosen
    return None, None


def clone_track(pack, track, source_track=None):
    out = copy.deepcopy(pack)
    out["track"] = track
    out["mixed_source_track"] = source_track or track
    out["difficulty_variant"] = HARD_VARIANT["name"] if source_track == HARD_TRACK or track == HARD_TRACK else "d58_d59_replay"
    return out


def make_track_packs(base_packs):
    saturated = [clone_track(pack, SATURATED_TRACK) for pack in base_packs]
    hard = [clone_track(d60.harden_pack(pack, HARD_VARIANT), HARD_TRACK) for pack in base_packs]
    mixed = []
    for idx, pack in enumerate(base_packs):
        if idx % 2 == 0:
            mixed.append(clone_track(pack, MIXED_TRACK, SATURATED_TRACK))
        else:
            mixed.append(clone_track(d60.harden_pack(pack, HARD_VARIANT), MIXED_TRACK, HARD_TRACK))
    return saturated, hard, mixed


def output_from_action(pack, arm, action, trace=None):
    row = d60.output_from_action(pack, arm, action, trace)
    row["track"] = pack.get("track", SATURATED_TRACK)
    row["mixed_source_track"] = pack.get("mixed_source_track", row["track"])
    row["context_gate_used"] = False
    row["context_gate_basis"] = None
    row["gate_selected_policy"] = None
    return row


def clone_row_for_arm(row, arm):
    out = copy.deepcopy(row)
    out["arm"] = arm
    return out


def summarize_outputs(outputs):
    by_arm = defaultdict(list)
    by_arm_core = defaultdict(list)
    by_arm_regime = defaultdict(list)
    by_action = defaultdict(Counter)
    by_error = defaultdict(Counter)
    by_seed_core = defaultdict(list)
    rust_counts = defaultdict(Counter)
    for row in outputs:
        arm = row["arm"]
        by_arm[arm].append(row)
        by_arm_regime[(arm, row["support_regime"])].append(row)
        by_action[arm][row["selected_action"]] += 1
        by_error[(arm, row["support_regime"])][row["error_type"]] += 1
        rust_counts[arm]["rows"] += 1
        if row["rust_network_path_invoked"]:
            rust_counts[arm]["rust_rows"] += 1
        if row["python_fallback_used"]:
            rust_counts[arm]["python_fallback_rows"] += 1
        if row["support_regime"] in CORE_REGIMES:
            by_arm_core[arm].append(row)
            by_seed_core[(arm, row["seed"])].append(row)
    return {
        "by_arm": {arm: d51.summarize(by_arm[arm]) for arm in ARMS if arm in by_arm},
        "by_arm_core": {arm: d51.summarize(by_arm_core[arm]) for arm in ARMS if arm in by_arm_core},
        "by_arm_and_regime": {
            arm: {regime: d51.summarize(by_arm_regime[(arm, regime)]) for regime in REGIMES if (arm, regime) in by_arm_regime}
            for arm in ARMS
        },
        "action_distribution": {arm: {action: by_action[arm][action] for action in ACTIONS} for arm in ARMS},
        "error_taxonomy": {
            arm: {regime: dict(by_error[(arm, regime)]) for regime in REGIMES if (arm, regime) in by_error}
            for arm in ARMS
        },
        "by_seed_core": {
            arm: {str(seed): d51.summarize(rows) for (a, seed), rows in by_seed_core.items() if a == arm}
            for arm in ARMS
        },
        "rust_usage": {arm: dict(counts) for arm, counts in rust_counts.items()},
    }


def candidate_rows(candidate_id, packs, action_records, arm_name):
    rows = []
    for idx, pack in enumerate(packs):
        action_record = action_records[candidate_id][idx]
        rows.append(output_from_action(pack, arm_name, action_record["action"], d59.rust_trace(action_record)))
    return rows


def effective_accuracy(rows):
    return d51.mean(
        [
            1.0
            if row["correct"] or (row["support_regime"] == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT" and row["abstained"])
            else 0.0
            for row in rows
        ]
    )


def cost_adjusted(row):
    return row["exact_joint_accuracy"] - 0.002 * row["average_total_support_used"] - 2.0 * row["false_confidence_rate"]


def score_multi_env(sat_rows, hard_rows, objective, saturated_floor):
    sat = d51.summarize(sat_rows)
    hard = d51.summarize(hard_rows)
    sat_eff = effective_accuracy(sat_rows)
    hard_eff = effective_accuracy(hard_rows)
    hard_corr = d51.mean([1.0 if row["correct"] else 0.0 for row in hard_rows if row["support_regime"] == "CORRELATED_ECHO_SUPPORT"])
    hard_adv = d51.mean([1.0 if row["correct"] else 0.0 for row in hard_rows if row["support_regime"] == "ADVERSARIAL_DISTRACTOR_SUPPORT"])
    sat_indist = d51.mean([1.0 if row["abstained"] else 0.0 for row in sat_rows if row["support_regime"] == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"])
    hard_indist = d51.mean([1.0 if row["abstained"] else 0.0 for row in hard_rows if row["support_regime"] == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"])
    support = 0.5 * (sat["average_total_support_used"] + hard["average_total_support_used"])
    false_conf = max(sat["false_confidence_rate"], hard["false_confidence_rate"])
    sat_shortfall = max(0.0, saturated_floor - sat["exact_joint_accuracy"])
    hard_stress = min(hard_corr, hard_adv) if hard_corr and hard_adv else hard["exact_joint_accuracy"]
    if objective == "lexicographic":
        if sat_shortfall > 0:
            score = sat["exact_joint_accuracy"] - 10.0 * sat_shortfall - 2.0 * false_conf
        else:
            score = 1.0 + hard["exact_joint_accuracy"] + 0.25 * hard_stress - 0.002 * support - 2.0 * false_conf
    elif objective == "pareto":
        score = (
            0.35 * sat["exact_joint_accuracy"]
            + 0.35 * hard["exact_joint_accuracy"]
            + 0.15 * hard_stress
            + 0.10 * min(sat_indist, hard_indist)
            - 0.006 * support
            - 4.0 * false_conf
            - 3.0 * sat_shortfall
        )
    elif objective == "stability_regularized":
        score = hard["exact_joint_accuracy"] + 0.25 * hard_stress - 0.003 * support - 6.0 * sat_shortfall - 3.0 * false_conf
    else:
        score = 0.5 * sat_eff + 0.5 * hard_eff + 0.2 * hard_stress - 0.003 * support - 3.0 * false_conf - 2.0 * sat_shortfall
    return score, {
        "saturated": sat,
        "hard": hard,
        "saturated_effective_accuracy": sat_eff,
        "hard_effective_accuracy": hard_eff,
        "hard_correlated": hard_corr,
        "hard_adversarial": hard_adv,
        "saturated_indistinguishable_abstain": sat_indist,
        "hard_indistinguishable_abstain": hard_indist,
        "saturated_shortfall": sat_shortfall,
        "support": support,
        "false_confidence": false_conf,
    }


def mutate_for_arm(controller, rng, arm):
    if arm == "LEXICOGRAPHIC_SAFETY_FIRST_FITNESS":
        return d60.structured_gate_mutation(controller, rng)
    if arm == "PARETO_MULTI_ENV_MUTATION":
        if rng.random() < 0.5:
            return d60.structured_gate_mutation(controller, rng)
    return d59.mutate_controller(controller, rng)


def stratified_subset(packs, limit, offset=0):
    return d59.stratified_subset(packs, limit, offset)


def train_multi_env_controller(base_controller, sat_packs, hard_packs, arm, objective, args, out, repo_root, started, saturated_floor):
    rng = stable_rng(60_500, f"{arm}_{objective}")
    current = copy.deepcopy(base_controller)
    best = copy.deepcopy(base_controller)
    sat_train = stratified_subset(sat_packs, args.mutation_train_packs, 0)
    hard_train = stratified_subset(hard_packs, args.mutation_train_packs, 1)
    sat_val = stratified_subset(sat_packs, args.mutation_validation_packs, 3)
    hard_val = stratified_subset(hard_packs, args.mutation_validation_packs, 5)
    init_sat_actions, _ = d59.run_rust_multi_bridge(out, repo_root, {"baseline": best}, sat_val, "safety_validation", f"{arm}_sat_initial", started)
    init_hard_actions, _ = d59.run_rust_multi_bridge(out, repo_root, {"baseline": best}, hard_val, "safety_validation", f"{arm}_hard_initial", started)
    best_score, best_metrics = score_multi_env(
        candidate_rows("baseline", sat_val, init_sat_actions, "candidate"),
        candidate_rows("baseline", hard_val, init_hard_actions, "candidate"),
        objective,
        saturated_floor,
    )
    history = []
    mutation_counts = Counter()
    accepted_counts = Counter()
    rejected_counts = Counter()
    accepted_total = 0
    rejected_total = 0
    last_write = 0.0
    for gen in range(args.generations):
        candidates = {}
        candidate_mutations = {}
        for idx in range(args.population):
            candidate, mutation = mutate_for_arm(current, rng, arm)
            candidate_id = f"{arm}_g{gen:04d}_c{idx:03d}"
            candidates[candidate_id] = candidate
            candidate_mutations[candidate_id] = mutation
            mutation_counts[mutation["type"]] += 1
        sat_subset = stratified_subset(sat_train, args.mutation_train_packs, gen)
        hard_subset = stratified_subset(hard_train, args.mutation_train_packs, gen + 1)
        sat_actions, _ = d59.run_rust_multi_bridge(out, repo_root, candidates, sat_subset, "safety_mutation_train", f"{arm}_sat_gen_{gen:04d}", started)
        hard_actions, _ = d59.run_rust_multi_bridge(out, repo_root, candidates, hard_subset, "safety_mutation_train", f"{arm}_hard_gen_{gen:04d}", started)
        scored = []
        for candidate_id in candidates:
            score, metrics = score_multi_env(
                candidate_rows(candidate_id, sat_subset, sat_actions, "candidate"),
                candidate_rows(candidate_id, hard_subset, hard_actions, "candidate"),
                objective,
                saturated_floor,
            )
            scored.append((score, candidate_id, metrics))
        scored.sort(key=lambda item: (item[0], -item[2]["support"]), reverse=True)
        top_score, top_id, top_metrics = scored[0]
        top_controller = candidates[top_id]
        sat_val_actions, _ = d59.run_rust_multi_bridge(out, repo_root, {top_id: top_controller}, sat_val, "safety_validation", f"{arm}_sat_gen_{gen:04d}", started)
        hard_val_actions, _ = d59.run_rust_multi_bridge(out, repo_root, {top_id: top_controller}, hard_val, "safety_validation", f"{arm}_hard_gen_{gen:04d}", started)
        val_score, val_metrics = score_multi_env(
            candidate_rows(top_id, sat_val, sat_val_actions, "candidate"),
            candidate_rows(top_id, hard_val, hard_val_actions, "candidate"),
            objective,
            saturated_floor,
        )
        before_score = best_score
        mutation_type = candidate_mutations[top_id]["type"]
        accepted = val_score >= best_score - args.accept_epsilon
        if accepted:
            current = copy.deepcopy(top_controller)
            best = copy.deepcopy(top_controller)
            best_score = val_score
            best_metrics = val_metrics
            accepted_counts[mutation_type] += 1
            accepted_total += 1
        else:
            rejected_counts[mutation_type] += 1
            rejected_total += 1
        for _score, cid, _metrics in scored[1:]:
            rejected_counts[candidate_mutations[cid]["type"]] += 1
            rejected_total += 1
        record = {
            "generation": gen,
            "arm": arm,
            "objective": objective,
            "top_candidate": top_id,
            "top_train_score": top_score,
            "top_validation_score": val_score,
            "best_score_before": before_score,
            "best_score_after": best_score,
            "accepted": accepted,
            "mutation": candidate_mutations[top_id],
            "top_train_metrics": top_metrics,
            "top_validation_metrics": val_metrics,
        }
        history.append(record)
        append_jsonl(out / f"mutation_history_{arm}.jsonl", record)
        now = time.time()
        if now - last_write >= args.heartbeat_sec or (gen + 1) % max(1, args.heartbeat_generations) == 0 or gen == args.generations - 1:
            last_write = now
            write_json(
                out / f"partial_mutation_{arm}.json",
                {
                    "generation": gen,
                    "best_score": best_score,
                    "best_metrics": best_metrics,
                    "accepted_total": accepted_total,
                    "rejected_total": rejected_total,
                    "accepted_counts": dict(accepted_counts),
                    "rejected_counts": dict(rejected_counts),
                },
            )
            append_progress(out, "mutation_progress", started, {"arm": arm, "objective": objective, "generation": gen, "best_score": best_score, "accepted_total": accepted_total})
    report = {
        "arm": arm,
        "objective": objective,
        "initial_validation_score": history[0]["best_score_before"] if history else best_score,
        "final_validation_score": best_score,
        "best_validation_metrics": best_metrics,
        "mutation_counts": dict(mutation_counts),
        "accepted_mutation_counts": dict(accepted_counts),
        "rejected_mutation_counts": dict(rejected_counts),
        "accepted_total": accepted_total,
        "rejected_total": rejected_total,
        "generations": args.generations,
        "population": args.population,
    }
    return best, report, history


def record_batch(batch, outputs, sample_counts, path):
    for row in batch:
        outputs.append(row)
        key = (row["track"], row["arm"], row["support_regime"])
        if sample_counts[key] < ROW_SAMPLE_PER_ARM_REGIME_SPLIT:
            append_jsonl(path, row)
            sample_counts[key] += 1


def gated_choice(pack, rust_actions, idx, arm):
    source = pack.get("mixed_source_track", pack.get("track", SATURATED_TRACK))
    if source == HARD_TRACK:
        chosen_arm = "D60_HARD_BEST_REPLAY"
        basis = "support_budget_cap8_context"
    else:
        chosen_arm = "D59_REFERENCE"
        basis = "saturated_or_full_support_context"
    action_record = rust_actions[chosen_arm][idx]
    row = output_from_action(pack, arm, action_record["action"], d59.rust_trace(action_record))
    row["context_gate_used"] = True
    row["context_gate_basis"] = basis
    row["gate_selected_policy"] = chosen_arm
    return row


def evaluate_pack_all_arms(pack, idx, rust_actions):
    rows = []
    for arm in RUST_CONTROLLER_ARMS:
        action_record = rust_actions[arm][idx]
        rows.append(output_from_action(pack, arm, action_record["action"], d59.rust_trace(action_record)))
    for arm in GATED_ARMS:
        rows.append(gated_choice(pack, rust_actions, idx, arm))
    canonical_action = rust_actions["D59_REFERENCE"][idx]
    shuffle = d55.spike_shuffle_mapping()
    rows.append(output_from_action(pack, "SPIKE_SHUFFLE_CONTROL", shuffle[canonical_action["action"]], d59.rust_trace(canonical_action)))
    rows.append(output_from_action(pack, "RANDOM_POLICY_CONTROL", d51.random_policy_action(f"D60S:{pack['track']}:{pack['row_id']}"), d59.disabled_rust_trace()))
    rows.append(output_from_action(pack, "GREEDY_DECIDE_CONTROL", "DECIDE", d59.disabled_rust_trace()))
    rows.append(output_from_action(pack, "ALWAYS_COUNTER_CONTROL", "REQUEST_JOINT_COUNTER", d59.disabled_rust_trace()))
    return rows


def write_partial_eval(out, split, track, outputs, completed, started):
    recent = outputs[-min(len(outputs), 2000):]
    partial = summarize_outputs(recent)
    write_json(
        out / f"partial_{split}_{track}_metrics_snapshot.json",
        {
            "split": split,
            "track": track,
            "completed_outputs": completed,
            "elapsed_sec": time.time() - started,
            "gated_recent": partial["by_arm_core"].get("CONTEXT_GATED_POLICY_ENSEMBLE", {}),
        },
    )
    append_progress(out, "eval_progress", started, {"split": split, "track": track, "completed_outputs": completed})


def evaluate_track(packs, controllers, out_path, out, split, track, started, heartbeat_sec, repo_root):
    rust_actions, rust_report = d59.run_rust_multi_bridge(out, repo_root, controllers, packs, split, f"{track}_final_eval", started)
    outputs = []
    sample_counts = Counter()
    completed = 0
    total = len(packs) * len(ARMS)
    last = 0.0
    for idx, pack in enumerate(packs):
        batch = evaluate_pack_all_arms(pack, idx, rust_actions)
        record_batch(batch, outputs, sample_counts, out_path)
        completed += len(batch)
        now = time.time()
        if now - last >= heartbeat_sec or completed >= total:
            last = now
            write_partial_eval(out, split, track, outputs, completed, started)
    return outputs, rust_report


def by_track_metrics(outputs):
    tracks = defaultdict(list)
    for row in outputs:
        tracks[row["track"]].append(row)
    return {track: summarize_outputs(rows) for track, rows in tracks.items()}


def arm_track_summary(metrics_by_track, arm, d58_hard_replay_exact, d59_exact):
    out = {}
    for track, metrics in metrics_by_track.items():
        core = metrics["by_arm_core"][arm]
        out[track] = {
            "exact": core["exact_joint_accuracy"],
            "support": core["average_total_support_used"],
            "false_confidence": core["false_confidence_rate"],
            "corr": metrics["by_arm_and_regime"][arm]["CORRELATED_ECHO_SUPPORT"]["accuracy"],
            "adv": metrics["by_arm_and_regime"][arm]["ADVERSARIAL_DISTRACTOR_SUPPORT"]["accuracy"],
            "external": metrics["by_arm_and_regime"][arm]["EXTERNAL_TEST_REQUIRED_SUPPORT"]["accuracy"],
            "abstain": metrics["by_arm_and_regime"][arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["abstain_rate"],
        }
    out["hard_gain_vs_D58"] = out[HARD_TRACK]["exact"] - d58_hard_replay_exact
    out["saturated_regression_vs_D59"] = out[SATURATED_TRACK]["exact"] - d59_exact
    return out


def arm_passes(summary, saturated_floor):
    return (
        summary[SATURATED_TRACK]["exact"] >= saturated_floor
        and summary[HARD_TRACK]["exact"] >= 0.99
        and summary[MIXED_TRACK]["exact"] >= 0.99
        and summary["hard_gain_vs_D58"] >= 0.30
        and max(summary[SATURATED_TRACK]["false_confidence"], summary[HARD_TRACK]["false_confidence"], summary[MIXED_TRACK]["false_confidence"]) <= 0.01
        and min(summary[SATURATED_TRACK]["abstain"], summary[HARD_TRACK]["abstain"], summary[MIXED_TRACK]["abstain"]) >= 0.99
    )


def make_decision(metrics_by_track, d60_summary, failed_jobs, mutation_reports):
    d59_exact = float(d60_summary["d59_reference_exact"])
    d58_hard_replay_exact = float(d60_summary["d58_hard_replay_exact"])
    saturated_floor = d59_exact - 0.002
    candidates = TRAINED_ARMS + GATED_ARMS
    summaries = {arm: arm_track_summary(metrics_by_track, arm, d58_hard_replay_exact, d59_exact) for arm in candidates}
    passing = [arm for arm in candidates if arm_passes(summaries[arm], saturated_floor)]
    if failed_jobs:
        best = max(candidates, key=lambda arm: (summaries[arm][HARD_TRACK]["exact"], summaries[arm][SATURATED_TRACK]["exact"]))
        return {
            "decision": "safety_fitness_repair_not_confirmed",
            "verdict": "D60S_SAFETY_FITNESS_REPAIR_NOT_CONFIRMED",
            "next": "D60R_NO_FORGETTING_REPAIR",
            "best_arm": best,
            "boundary": BOUNDARY,
        }, summaries
    if passing:
        best = max(
            passing,
            key=lambda arm: (
                summaries[arm][MIXED_TRACK]["exact"],
                summaries[arm][HARD_TRACK]["exact"],
                summaries[arm][SATURATED_TRACK]["exact"],
                -summaries[arm][MIXED_TRACK]["support"],
            ),
        )
        if best in GATED_ARMS:
            return {
                "decision": "gated_policy_required_for_no_forgetting",
                "verdict": "D60S_GATED_POLICY_REQUIRED_FOR_NO_FORGETTING",
                "next": "D61_GATED_RUST_SPARSE_MUTATION_SCALE_CONFIRM",
                "best_arm": best,
                "boundary": BOUNDARY,
            }, summaries
        return {
            "decision": "rust_sparse_mutation_safety_fitness_repaired",
            "verdict": "D60S_RUST_SPARSE_MUTATION_SAFETY_FITNESS_REPAIRED",
            "next": "D61_RUST_SPARSE_MUTATION_SCALE_CONFIRM",
            "best_arm": best,
            "boundary": BOUNDARY,
        }, summaries
    best_hard = max(candidates, key=lambda arm: summaries[arm][HARD_TRACK]["exact"])
    if summaries[best_hard][HARD_TRACK]["exact"] - d58_hard_replay_exact >= 0.30:
        return {
            "decision": "safety_fitness_repair_not_confirmed",
            "verdict": "D60S_SAFETY_FITNESS_REPAIR_NOT_CONFIRMED",
            "next": "D60R_NO_FORGETTING_REPAIR",
            "best_arm": best_hard,
            "boundary": BOUNDARY,
        }, summaries
    return {
        "decision": "learning_signal_lost_under_safety_fitness",
        "verdict": "D60S_LEARNING_SIGNAL_LOST_UNDER_SAFETY_FITNESS",
        "next": "D60L_LEARNING_SIGNAL_REPAIR",
        "best_arm": best_hard,
        "boundary": BOUNDARY,
    }, summaries


def make_reports(out, aggregate, decision, arm_summaries):
    metrics = aggregate["test_metrics_by_track"]
    reports = {
        "d60_upstream_manifest.json": aggregate["d60_upstream_manifest"],
        "fitness_definition_report.json": {
            "single_policy_multi_env": "joint score over saturated and hard packs",
            "safety_first": "saturated floor must pass before hard gain meaningfully counts",
            "pareto_multi_env": "weighted score over saturated exact, hard exact, hard stress, support, false confidence",
            "stability_regularized": "hard exact plus explicit saturated shortfall penalty",
            "dual_policy": "non-truth context gate between D59 reference and D60 hard policy",
        },
        "multi_environment_eval_report.json": arm_summaries,
        "saturated_stability_report.json": {arm: arm_summaries[arm][SATURATED_TRACK] for arm in arm_summaries},
        "hard_learning_report.json": {arm: arm_summaries[arm][HARD_TRACK] for arm in arm_summaries},
        "mixed_eval_report.json": {arm: arm_summaries[arm][MIXED_TRACK] for arm in arm_summaries},
        "policy_gate_report.json": {
            "gated_arms": GATED_ARMS,
            "gate_uses_truth_labels": False,
            "gate_inputs": ["track", "mixed_source_track", "support_budget_cap8_context"],
            "routing": {
                "saturated_or_full_support_context": "D59_REFERENCE",
                "support_budget_cap8_context": "D60_HARD_BEST_REPLAY",
            },
        },
        "no_forgetting_report.json": {
            arm: {
                "saturated_exact": arm_summaries[arm][SATURATED_TRACK]["exact"],
                "saturated_regression_vs_D59": arm_summaries[arm]["saturated_regression_vs_D59"],
                "passes_floor": arm_summaries[arm][SATURATED_TRACK]["exact"] >= aggregate["saturated_floor"],
            }
            for arm in arm_summaries
        },
        "pareto_frontier_report.json": {
            arm: {
                "saturated_exact": arm_summaries[arm][SATURATED_TRACK]["exact"],
                "hard_exact": arm_summaries[arm][HARD_TRACK]["exact"],
                "mixed_exact": arm_summaries[arm][MIXED_TRACK]["exact"],
                "support": arm_summaries[arm][MIXED_TRACK]["support"],
                "false_confidence": max(
                    arm_summaries[arm][SATURATED_TRACK]["false_confidence"],
                    arm_summaries[arm][HARD_TRACK]["false_confidence"],
                    arm_summaries[arm][MIXED_TRACK]["false_confidence"],
                ),
            }
            for arm in sorted(arm_summaries, key=lambda item: (-arm_summaries[item][MIXED_TRACK]["exact"], arm_summaries[item][MIXED_TRACK]["support"]))
        },
        "mutation_causality_report.json": {
            "mutation_reports": aggregate["mutation_reports"],
            "mutation_path_exercised": any(report.get("accepted_total", 0) + report.get("rejected_total", 0) > 0 for report in aggregate["mutation_reports"].values()),
            "accepted_mutation_total": sum(report.get("accepted_total", 0) for report in aggregate["mutation_reports"].values()),
            "rejected_mutation_total": sum(report.get("rejected_total", 0) for report in aggregate["mutation_reports"].values()),
            "claim": "D60S distinguishes single-policy mutation from non-truth context-gated policy reuse.",
            "formula_solver_learning_used": False,
            "truth_hidden_from_controller_inputs": True,
        },
        "support_cost_frontier_report.json": {
            track: {
                arm: {
                    "exact": metrics[track]["by_arm_core"][arm]["exact_joint_accuracy"],
                    "support": metrics[track]["by_arm_core"][arm]["average_total_support_used"],
                    "cost_adjusted": cost_adjusted(metrics[track]["by_arm_core"][arm]),
                }
                for arm in ARMS
            }
            for track in TRACKS
        },
        "safety_constraint_report.json": {
            "saturated_floor": aggregate["saturated_floor"],
            "fallback_rows": aggregate["fallback_rows"],
            "rust_path_invoked": aggregate["rust_path_invoked"],
            "false_confidence_by_track": {
                track: {arm: metrics[track]["by_arm_core"][arm]["false_confidence_rate"] for arm in ARMS}
                for track in TRACKS
            },
        },
        "rust_invocation_report.json": aggregate["rust_invocation_report"],
        "ablation_report.json": {
            arm: {
                track: {
                    "exact": metrics[track]["by_arm_core"][arm]["exact_joint_accuracy"],
                    "support": metrics[track]["by_arm_core"][arm]["average_total_support_used"],
                }
                for track in TRACKS
            }
            for arm in ["RANDOM_POLICY_CONTROL", "GREEDY_DECIDE_CONTROL", "SPIKE_SHUFFLE_CONTROL", "THRESHOLD_ABLATION", "REWIRE_ABLATION"]
        },
    }
    for name, value in reports.items():
        write_json(out / name, value)
    return reports


def write_report(out, decision, arm_summaries):
    lines = [
        "# D60S Rust Sparse Mutation Safety Fitness Repair Result",
        "",
        "Decision:",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"verdict = {decision.get('verdict')}",
        f"next = {decision.get('next')}",
        f"best_arm = {decision.get('best_arm')}",
        "```",
        "",
        "Boundary:",
        "",
        "```text",
        BOUNDARY,
        "```",
        "",
        "| arm | sat exact | hard exact | mixed exact | hard gain | sat regression |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, summary in sorted(arm_summaries.items()):
        lines.append(
            f"| {arm} | {summary[SATURATED_TRACK]['exact']:.6f} | {summary[HARD_TRACK]['exact']:.6f} | "
            f"{summary[MIXED_TRACK]['exact']:.6f} | {summary['hard_gain_vs_D58']:.6f} | "
            f"{summary['saturated_regression_vs_D59']:.6f} |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_summary(aggregate, decision, arm_summaries):
    best = decision.get("best_arm")
    return {
        "task": TASK,
        "decision": decision["decision"],
        "verdict": decision.get("verdict"),
        "next": decision.get("next"),
        "best_arm": best,
        "best_summary": arm_summaries.get(best),
        "saturated_floor": aggregate["saturated_floor"],
        "rust_path_invoked": aggregate["rust_path_invoked"],
        "fallback_rows": aggregate["fallback_rows"],
        "failed_jobs": aggregate["failed_jobs"],
        "boundary": BOUNDARY,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="11501,11502,11503,11504,11505")
    parser.add_argument("--train-rows-per-seed", type=int, default=800)
    parser.add_argument("--test-rows-per-seed", type=int, default=800)
    parser.add_argument("--ood-rows-per-seed", type=int, default=800)
    parser.add_argument("--generations", type=int, default=160)
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="50-75")
    parser.add_argument("--heartbeat-sec", type=float, default=20.0)
    parser.add_argument("--heartbeat-generations", type=int, default=8)
    parser.add_argument("--mutation-train-packs", type=int, default=384)
    parser.add_argument("--mutation-validation-packs", type=int, default=384)
    parser.add_argument("--accept-epsilon", type=float, default=0.0005)
    parser.add_argument("--scale-mode", default="full")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    seeds = parse_seeds(args.seeds)
    repo_root = Path(__file__).resolve().parents[2]
    failed_jobs = []
    write_json(out / "queue.json", {"task": TASK, "args": vars(args), "seeds": seeds, "boundary": BOUNDARY})
    append_progress(out, "started", started, {"seeds": seeds, "generations": args.generations, "population": args.population})
    write_json(out / "compute_probe.json", {"cpu_count": os.cpu_count(), "workers": d51.worker_count_from_arg(args.workers), "cpu_target": args.cpu_target})

    d60_manifest = make_d60_upstream_manifest(repo_root)
    write_json(out / "d60_upstream_manifest.json", d60_manifest)
    d59_ref, d59_ref_name = load_d59_reference(repo_root)
    d60_hard, d60_hard_name = load_d60_controller(repo_root)
    d60_accuracy, _ = load_d60_controller(repo_root, "ACCURACY_ONLY_MUTATION")
    d60_cost, _ = load_d60_controller(repo_root, "COST_ONLY_MUTATION_CONTROL")
    if d59_ref is None:
        failed_jobs.append("missing_d59_reference")
    if d60_hard is None:
        failed_jobs.append("missing_d60_hard_best")
        d60_hard = copy.deepcopy(d59_ref)
    if d60_accuracy is None:
        d60_accuracy = copy.deepcopy(d60_hard)
    if d60_cost is None:
        d60_cost = copy.deepcopy(d60_hard)

    d60_summary_artifact = d60_manifest.get("summary") or {}
    d59_exact = 0.9994
    d60_hard_exact = float((d60_summary_artifact.get("hard_key_metrics") or {}).get("exact_joint_accuracy", 0.9957))
    d60_learning_gain = float((d60_summary_artifact.get("learning_gain") or {}).get("exact_gain", 0.39065))
    d58_hard_replay_exact = d60_hard_exact - d60_learning_gain
    saturated_floor = d59_exact - 0.002

    bundle = d55.d49_bundle()
    write_json(
        out / "dataset_manifest.json",
        {
            "task": "controlled_symbolic_joint_formula_discovery",
            "primary_space": PRIMARY_SPACE,
            "support_count": SUPPORT_COUNT,
            "regimes": REGIMES,
            "core_regimes": CORE_REGIMES,
            "tracks": TRACKS,
            "hard_variant": HARD_VARIANT,
            "feature_names": FEATURE_NAMES,
            "truth_hidden_from_controller_inputs": True,
            "controller_only_not_formula_solver": True,
            "formula_solver_learning_used": False,
        },
    )

    train_rows = d51.make_rows_with_progress(seeds, args.train_rows_per_seed, "train", bundle, out, started, args.heartbeat_sec)
    test_rows = d51.make_rows_with_progress(seeds, args.test_rows_per_seed, "test", bundle, out, started, args.heartbeat_sec)
    ood_rows = d51.make_rows_with_progress(seeds, args.ood_rows_per_seed, "ood", bundle, out, started, args.heartbeat_sec)
    train_base = d51.build_packs(train_rows, bundle, out, started, args.heartbeat_sec, args.workers, "train")
    test_base = d51.build_packs(test_rows, bundle, out, started, args.heartbeat_sec, args.workers, "test")
    ood_base = d51.build_packs(ood_rows, bundle, out, started, args.heartbeat_sec, args.workers, "ood")
    train_sat, train_hard, train_mixed = make_track_packs(train_base)
    test_sat, test_hard, test_mixed = make_track_packs(test_base)
    ood_sat, ood_hard, ood_mixed = make_track_packs(ood_base)
    append_progress(out, "packs_built", started, {"train": len(train_base), "test": len(test_base), "ood": len(ood_base)})

    controllers = {
        "D59_REFERENCE": copy.deepcopy(d59_ref),
        "D60_HARD_BEST_REPLAY": copy.deepcopy(d60_hard),
        "COST_ONLY_MUTATION_CONTROL": copy.deepcopy(d60_cost),
        "ACCURACY_ONLY_MUTATION_CONTROL": copy.deepcopy(d60_accuracy),
        "MUTATION_DISABLED_CONTROL": copy.deepcopy(d59_ref),
        "RANDOM_MUTATION_CONTROL": d59.random_mutated_controller(d59_ref, stable_rng(60_999, "d60s_random_mutation"), max(2, args.generations // 8)),
        "THRESHOLD_ABLATION": d55.make_threshold_ablation(d59_ref),
        "REWIRE_ABLATION": d55.make_rewire_ablation(d59_ref),
    }
    objective_map = {
        "SINGLE_POLICY_MULTI_ENV_FITNESS": "single",
        "LEXICOGRAPHIC_SAFETY_FIRST_FITNESS": "lexicographic",
        "PARETO_MULTI_ENV_MUTATION": "pareto",
        "STABILITY_REGULARIZED_MUTATION": "stability_regularized",
    }
    mutation_reports = {}
    mutation_histories = {}
    for arm, objective in objective_map.items():
        controller, report, history = train_multi_env_controller(d59_ref, train_sat, train_hard, arm, objective, args, out, repo_root, started, saturated_floor)
        controllers[arm] = controller
        mutation_reports[arm] = report
        mutation_histories[arm] = history[-10:]

    row_test = out / "row_outputs_test.jsonl"
    row_ood = out / "row_outputs_ood.jsonl"
    for path in [row_test, row_ood]:
        if path.exists():
            path.unlink()
    test_outputs = []
    ood_outputs = []
    rust_reports = {}
    for track, packs in [(SATURATED_TRACK, test_sat), (HARD_TRACK, test_hard), (MIXED_TRACK, test_mixed)]:
        outputs, report = evaluate_track(packs, controllers, row_test, out, "test", track, started, args.heartbeat_sec, repo_root)
        test_outputs.extend(outputs)
        rust_reports[f"test_{track}"] = report
    for track, packs in [(SATURATED_TRACK, ood_sat), (HARD_TRACK, ood_hard), (MIXED_TRACK, ood_mixed)]:
        outputs, report = evaluate_track(packs, controllers, row_ood, out, "ood", track, started, args.heartbeat_sec, repo_root)
        ood_outputs.extend(outputs)
        rust_reports[f"ood_{track}"] = report

    test_metrics_by_track = by_track_metrics(test_outputs)
    ood_metrics_by_track = by_track_metrics(ood_outputs)
    d60_summary = {
        "d59_reference_exact": d59_exact,
        "d58_hard_replay_exact": d58_hard_replay_exact,
        "d60_hard_best_exact": d60_hard_exact,
        "d60_hard_gain_vs_d58": d60_learning_gain,
        "d59_reference_loaded": d59_ref_name,
        "d60_hard_loaded": d60_hard_name,
    }
    decision, arm_summaries = make_decision(test_metrics_by_track, d60_summary, failed_jobs, mutation_reports)
    fallback_rows = 0
    rust_rows = 0
    for metrics in test_metrics_by_track.values():
        for arm, counts in metrics["rust_usage"].items():
            if arm in RUST_CONTROLLER_ARMS + GATED_ARMS:
                fallback_rows += counts.get("python_fallback_rows", 0)
                rust_rows += counts.get("rust_rows", 0)

    aggregate = {
        "task": TASK,
        "failed_jobs": failed_jobs,
        "d60_upstream_manifest": d60_manifest,
        "d60_summary": d60_summary,
        "saturated_floor": saturated_floor,
        "test_metrics_by_track": test_metrics_by_track,
        "ood_metrics_by_track": ood_metrics_by_track,
        "arm_summaries": arm_summaries,
        "mutation_reports": mutation_reports,
        "mutation_history_tail": mutation_histories,
        "rust_invocation_report": rust_reports,
        "rust_path_invoked": rust_rows > 0,
        "fallback_rows": fallback_rows,
        "boundary": BOUNDARY,
        "decision": decision,
    }
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    reports = make_reports(out, aggregate, decision, arm_summaries)
    write_json(out / "summary.json", make_summary(aggregate, decision, arm_summaries))
    write_json(out / "trained_policy_manifest.json", {"controllers": controllers, "mutation_reports": mutation_reports, "decision": decision})
    write_report(out, decision, arm_summaries)
    append_progress(out, "completed", started, {"decision": decision["decision"], "reports": sorted(reports.keys())})
    print(json.dumps(make_summary(aggregate, decision, arm_summaries), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
