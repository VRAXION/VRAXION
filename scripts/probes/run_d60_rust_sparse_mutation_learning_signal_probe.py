#!/usr/bin/env python3
"""D60 Rust sparse mutation learning-signal probe.

This probe deliberately separates two questions:

1. Saturated stability: does the D59/D58 Rust sparse controller path remain
   stable on the already-near-ceiling task?
2. Hard learning signal: when the same controller is evaluated under a tighter,
   still-solvable support budget, can mutation/selection find a better action
   controller than replay?

The Rust sparse network chooses ECF actions only. The symbolic formula solver is
fixed, and truth labels are not exposed to controller features.
"""

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
import run_d57_canonical_rust_sparse_path_bridge as d57
import run_d59_rust_sparse_ecf_controller_with_mutation as d59

PRIMARY_SPACE = d59.PRIMARY_SPACE
SUPPORT_COUNT = d59.SUPPORT_COUNT
REGIMES = d59.REGIMES
CORE_REGIMES = d59.CORE_REGIMES
ACTIONS = d59.ACTIONS
FEATURE_NAMES = d59.FEATURE_NAMES
ROW_SAMPLE_PER_ARM_REGIME_SPLIT = d59.ROW_SAMPLE_PER_ARM_REGIME_SPLIT

TASK = "D60_RUST_SPARSE_MUTATION_LEARNING_SIGNAL_PROBE"
BOUNDARY = (
    "D60 only tests learning signal for mutation and selection of a canonical Rust sparse "
    "ECF action controller on controlled symbolic joint formula discovery. It does not prove "
    "full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI, consciousness, "
    "DNA/genome success, architecture superiority, or production readiness."
)

SATURATED_TRACK = "SATURATED_STABILITY"
HARD_TRACK = "HARD_NON_SATURATED_LEARNING"

RUST_POLICY_ARMS = [
    "D58_REPLAY_REFERENCE",
    "D59_BEST_REPLAY",
    "MUTATION_DISABLED_CONTROL",
    "RANDOM_MUTATION_CONTROL",
    "COST_ONLY_MUTATION_CONTROL",
    "ACCURACY_ONLY_MUTATION",
    "SUPPORT_COST_TARGETED_MUTATION",
    "HARD_STRESS_MUTATION",
    "MULTI_OBJECTIVE_PARETO_MUTATION",
    "LARGE_STEP_MUTATION",
    "STRUCTURED_GATE_MUTATION",
    "NOVELTY_DIVERSITY_MUTATION",
    "THRESHOLD_ABLATION",
    "REWIRE_ABLATION",
]

CONTROL_ARMS = [
    "RANDOM_POLICY_CONTROL",
    "GREEDY_DECIDE_CONTROL",
    "ALWAYS_COUNTER_CONTROL",
    "SPIKE_SHUFFLE_CONTROL",
]

MUTATED_ARMS = [
    "COST_ONLY_MUTATION_CONTROL",
    "ACCURACY_ONLY_MUTATION",
    "SUPPORT_COST_TARGETED_MUTATION",
    "HARD_STRESS_MUTATION",
    "MULTI_OBJECTIVE_PARETO_MUTATION",
    "LARGE_STEP_MUTATION",
    "STRUCTURED_GATE_MUTATION",
    "NOVELTY_DIVERSITY_MUTATION",
]

ARMS = RUST_POLICY_ARMS + CONTROL_ARMS

HARD_VARIANTS = [
    {
        "name": "support_budget_cap_5",
        "support_budget_cap": 5,
        "description": "Only original support is affordable; expected to be too hard for robust repair.",
    },
    {
        "name": "support_budget_cap_6",
        "support_budget_cap": 6,
        "description": "Very tight budget; included as requested but usually below useful oracle bound.",
    },
    {
        "name": "support_budget_cap_8",
        "support_budget_cap": 8,
        "description": "Allows top1-vs-top2 counter-support but blocks full joint counter-support.",
    },
    {
        "name": "support_budget_cap_9",
        "support_budget_cap": 9,
        "description": "Allows top1-vs-top2 and external test, blocks full joint counter-support.",
    },
    {
        "name": "higher_counter_support_cost_cap_9",
        "support_budget_cap": 9,
        "counter_cost_multiplier": 1.35,
        "description": "Same observable task as cap9, with stronger fitness pressure against counter-support.",
    },
]


def write_json(path, value):
    d51.write_json(path, value)


def append_jsonl(path, value):
    d51.append_jsonl(path, value)


def append_progress(out, event, started, data):
    d51.append_progress(out, event, started, data)


def parse_seeds(raw):
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def load_json_if_present(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def stable_rng(seed, tag):
    return random.Random(int(seed) + d51.stable_seed(tag))


def make_d59_upstream_manifest(repo_root):
    root = repo_root / "target/pilot_wave/d59_rust_sparse_ecf_controller_with_mutation/smoke_full"
    manifest = {
        "upstream": "D59_RUST_SPARSE_ECF_CONTROLLER_WITH_MUTATION",
        "expected_decision": "rust_sparse_mutation_path_confirmed_no_gain",
        "expected_next": "D60_RUST_SPARSE_MUTATION_FITNESS_REPAIR",
        "reinterpreted_next": TASK,
        "artifact_root": str(root),
        "decision_present": (root / "decision.json").exists(),
        "summary_present": (root / "summary.json").exists(),
        "trained_policy_manifest_present": (root / "trained_policy_manifest.json").exists(),
    }
    for name in [
        "decision.json",
        "summary.json",
        "rust_invocation_report.json",
        "rust_mutation_representation_report.json",
        "before_after_mutation_report.json",
    ]:
        value = load_json_if_present(root / name)
        if value is not None:
            manifest[name.replace(".json", "")] = value
    trained = load_json_if_present(root / "trained_policy_manifest.json")
    best = (manifest.get("decision") or {}).get("best_arm")
    manifest["d59_best_controller_loaded"] = bool(trained and best and trained.get("controllers", {}).get(best))
    return manifest


def load_d59_best_controller(repo_root):
    root = repo_root / "target/pilot_wave/d59_rust_sparse_ecf_controller_with_mutation/smoke_full"
    decision = load_json_if_present(root / "decision.json") or {}
    trained = load_json_if_present(root / "trained_policy_manifest.json") or {}
    best = decision.get("best_arm")
    if best and trained.get("controllers", {}).get(best):
        return trained["controllers"][best], best
    return None, None


def load_d58_controller(repo_root):
    controller = d59.load_d58_controller(repo_root)
    if controller is None:
        controller = d57.load_d56_best_controller(repo_root)
    return controller


def row_accuracy(row):
    return 1.0 if row.get("correct") else 0.0


def clone_with_track(pack, track, variant_name):
    out = copy.deepcopy(pack)
    out["track"] = track
    out["difficulty_variant"] = variant_name
    return out


def budget_blocked_result(pack, action, cap):
    """Use the real DECIDE outcome when requested support exceeds the budget."""
    blocked = copy.deepcopy(pack["actions"]["DECIDE"])
    blocked["hard_budget_violation"] = True
    blocked["blocked_action"] = action
    blocked["error_type"] = "budget_exceeded_then_" + str(blocked.get("error_type", "unknown"))
    blocked["counter_support_used"] = max(0, cap - blocked["original_support_used"])
    blocked["cell_counter_support_used"] = 0
    blocked["operator_counter_support_used"] = 0
    blocked["joint_counter_support_used"] = blocked["counter_support_used"]
    blocked["random_counter_support_used"] = 0
    blocked["external_test_used"] = 0
    blocked["total_support_used"] = max(blocked["original_support_used"], cap)
    return blocked


def harden_pack(pack, variant):
    cap = int(variant["support_budget_cap"])
    out = clone_with_track(pack, HARD_TRACK, variant["name"])
    actions = {}
    for action in ACTIONS:
        result = copy.deepcopy(pack["actions"][action])
        result["hard_budget_violation"] = False
        result["blocked_action"] = None
        if result["total_support_used"] > cap:
            result = budget_blocked_result(pack, action, cap)
        result["hard_support_budget_cap"] = cap
        result["hard_counter_cost_multiplier"] = float(variant.get("counter_cost_multiplier", 1.0))
        actions[action] = result
    out["actions"] = actions
    out["feature_map"] = copy.deepcopy(pack["feature_map"])
    out["feature_map"]["hard_support_budget_cap_norm"] = cap / 12.0
    out["feature_map"]["hard_counter_cost_multiplier"] = float(variant.get("counter_cost_multiplier", 1.0))
    return out


def saturate_pack(pack):
    return clone_with_track(pack, SATURATED_TRACK, "d58_d59_replay")


def oracle_upper_bound_for_packs(packs):
    rows = []
    for pack in packs:
        best = max(
            pack["actions"].values(),
            key=lambda row: (
                row_accuracy(row),
                1.0 if row.get("abstained") and row["support_regime"] == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT" else 0.0,
                -row["total_support_used"],
            ),
        )
        rows.append(best)
    metrics = d51.summarize(rows)
    effective = d51.mean(
        [
            1.0
            if row["correct"] or (row["support_regime"] == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT" and row["abstained"])
            else 0.0
            for row in rows
        ]
    )
    return {
        "oracle_accuracy": metrics["accuracy"],
        "oracle_effective_accuracy": effective,
        "oracle_exact_joint": metrics["exact_joint_accuracy"],
        "oracle_support": metrics["average_total_support_used"],
        "oracle_false_confidence": metrics["false_confidence_rate"],
        "oracle_indistinguishable_abstain": d51.mean(
            [1.0 if row["abstained"] else 0.0 for row in rows if row["support_regime"] == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]
        ),
        "rows": len(rows),
    }


def output_from_action(pack, arm, action, trace=None):
    row = d59.output_from_action(pack, arm, action, trace)
    row["track"] = pack.get("track", SATURATED_TRACK)
    row["difficulty_variant"] = pack.get("difficulty_variant", "d58_d59_replay")
    row["hard_budget_violation"] = bool(row.get("hard_budget_violation", False))
    row["hard_support_budget_cap"] = row.get("hard_support_budget_cap")
    row["hard_counter_cost_multiplier"] = row.get("hard_counter_cost_multiplier", 1.0)
    return row


def summarize_outputs(outputs):
    by_arm = defaultdict(list)
    by_arm_core = defaultdict(list)
    by_arm_regime = defaultdict(list)
    by_action = defaultdict(Counter)
    by_error = defaultdict(Counter)
    by_seed_core = defaultdict(list)
    by_seed_regime = defaultdict(list)
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
        by_seed_regime[(arm, row["seed"], row["support_regime"])].append(row)
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
        "by_seed_regime": {
            arm: {
                str(seed): {
                    regime: d51.summarize(by_seed_regime[(arm, seed, regime)])
                    for regime in REGIMES
                    if (arm, seed, regime) in by_seed_regime
                }
                for seed in sorted({seed for (a, seed, _regime) in by_seed_regime if a == arm})
            }
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


def baseline_metrics_for_variant(packs, controller, repo_root, out, started, tag):
    subset = packs[: min(len(packs), 512)]
    actions, _report = d59.run_rust_multi_bridge(out, repo_root, {"baseline": controller}, subset, "difficulty_probe", tag, started)
    rows = candidate_rows("baseline", subset, actions, "D58_REPLAY_REFERENCE")
    return summarize_outputs(rows)["by_arm_core"].get("D58_REPLAY_REFERENCE", d51.summarize(rows))


def choose_hard_variant(base_packs, controller, repo_root, out, started):
    report = {}
    selected = None
    for variant in HARD_VARIANTS:
        hardened = [harden_pack(pack, variant) for pack in base_packs]
        oracle = oracle_upper_bound_for_packs(hardened)
        baseline = baseline_metrics_for_variant(hardened, controller, repo_root, out, started, variant["name"])
        entry = {
            "variant": variant,
            "oracle": oracle,
            "baseline_probe": baseline,
            "valid_oracle": oracle["oracle_effective_accuracy"] >= 0.95 and oracle["oracle_false_confidence"] <= 0.02,
            "baseline_non_saturated": 0.55 <= baseline["exact_joint_accuracy"] <= 0.95,
        }
        report[variant["name"]] = entry
        if selected is None and entry["valid_oracle"] and entry["baseline_non_saturated"]:
            selected = variant
    if selected is None:
        for variant in HARD_VARIANTS:
            if report[variant["name"]]["valid_oracle"]:
                selected = variant
                break
    return selected, report


def support_penalty(row):
    mult = float(row.get("hard_counter_cost_multiplier", 1.0))
    return row["total_support_used"] + (mult - 1.0) * row["counter_support_used"]


def summarize_cost_adjusted(rows):
    metrics = d51.summarize(rows)
    exact = metrics["exact_joint_accuracy"]
    false_conf = metrics["false_confidence_rate"]
    support = d51.mean([support_penalty(row) for row in rows])
    return exact - 0.002 * support - 2.0 * false_conf


def score_rows(rows, objective):
    metrics = d51.summarize(rows)
    exact = metrics["exact_joint_accuracy"]
    support = d51.mean([support_penalty(row) for row in rows])
    false_conf = metrics["false_confidence_rate"]
    corr_rows = [row for row in rows if row["support_regime"] == "CORRELATED_ECHO_SUPPORT"]
    adv_rows = [row for row in rows if row["support_regime"] == "ADVERSARIAL_DISTRACTOR_SUPPORT"]
    ext_rows = [row for row in rows if row["support_regime"] == "EXTERNAL_TEST_REQUIRED_SUPPORT"]
    indist_rows = [row for row in rows if row["support_regime"] == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]
    corr = d51.mean([row_accuracy(row) for row in corr_rows]) if corr_rows else exact
    adv = d51.mean([row_accuracy(row) for row in adv_rows]) if adv_rows else exact
    external = d51.mean([row_accuracy(row) for row in ext_rows]) if ext_rows else 1.0
    abstain = d51.mean([1.0 if row["abstained"] else 0.0 for row in indist_rows]) if indist_rows else 1.0
    budget_violation = d51.mean([1.0 if row.get("hard_budget_violation") else 0.0 for row in rows])
    metrics["cost_adjusted"] = summarize_cost_adjusted(rows)
    metrics["hard_budget_violation_rate"] = budget_violation
    metrics["correlated_echo_accuracy"] = corr
    metrics["adversarial_distractor_accuracy"] = adv
    metrics["external_test_required_accuracy"] = external
    metrics["indistinguishable_abstain_rate"] = abstain
    safety_guard = 0.30 * abstain + 0.10 * external - 2.0 * false_conf
    stress = min(corr, adv)
    if objective == "accuracy_only":
        score = exact + 0.25 * stress + safety_guard - 0.0005 * support
    elif objective == "cost_only":
        score = exact - 0.010 * support - 0.50 * budget_violation + safety_guard
    elif objective == "support_cost_targeted":
        score = exact + 0.35 * stress - 0.006 * support - 0.80 * budget_violation + safety_guard
    elif objective == "hard_stress":
        score = 0.45 * exact + 0.55 * stress - 0.004 * support - 0.50 * budget_violation + safety_guard
    elif objective == "pareto":
        score = exact + 0.20 * stress + 0.30 * metrics["cost_adjusted"] - 0.40 * budget_violation + safety_guard
    elif objective == "novelty":
        action_counts = Counter(row["selected_action"] for row in rows)
        action_diversity = len([count for count in action_counts.values() if count]) / float(len(ACTIONS))
        score = exact + 0.20 * stress + 0.02 * action_diversity - 0.004 * support - 0.50 * budget_violation + safety_guard
    else:
        score = exact + 0.20 * stress - 0.003 * support - 0.50 * budget_violation + safety_guard
    return score, metrics


def stratified_subset(packs, limit, offset=0):
    return d59.stratified_subset(packs, limit, offset)


def structured_gate_mutation(controller, rng):
    out = copy.deepcopy(controller)
    candidates = [gate for gate in out["gates"] if gate.get("feature") in {"dominant_cluster_fraction", "inverse_margin", "entropy_norm"}]
    gate = rng.choice(candidates or out["gates"])
    before = copy.deepcopy(gate)
    roll = rng.random()
    if roll < 0.45:
        gate["action"] = "REQUEST_COUNTER_TOP1_TOP2"
        gate["priority"] = max(45, min(75, int(gate.get("priority", 40))))
        mutation_type = "structured_counter_readout"
    elif roll < 0.75:
        gate["threshold"] = max(0, min(17, int(gate["threshold"]) + rng.choice([-2, -1, 1])))
        gate["stored_threshold"] = d55.stored_threshold(gate["threshold"])
        mutation_type = "structured_threshold"
    else:
        gate["weight"] = max(-32, min(48, int(gate["weight"]) + rng.choice([-8, -4, 4, 8])))
        gate["polarity"] = 1 if int(gate["weight"]) >= 0 else -1
        mutation_type = "structured_weight"
    out["mutation_history"] = out.get("mutation_history", []) + [mutation_type]
    out["_last_mutation"] = {"type": mutation_type, "gate_id": gate.get("gate_id"), "before": before, "after": copy.deepcopy(gate)}
    return out, out["_last_mutation"]


def mutate_for_arm(controller, rng, arm):
    if arm == "LARGE_STEP_MUTATION":
        out = copy.deepcopy(controller)
        trace = []
        for _ in range(3):
            out, mutation = d59.mutate_controller(out, rng)
            trace.append(mutation)
        out["_last_mutation"] = {"type": "large_step", "steps": trace}
        return out, out["_last_mutation"]
    if arm in {"STRUCTURED_GATE_MUTATION", "SUPPORT_COST_TARGETED_MUTATION"}:
        return structured_gate_mutation(controller, rng)
    return d59.mutate_controller(controller, rng)


def train_mutation_controller(base_controller, packs, validation_packs, arm, objective, args, out, repo_root, started):
    rng = stable_rng(60_000, f"{arm}_{objective}")
    current = copy.deepcopy(base_controller)
    best = copy.deepcopy(base_controller)
    train_subset = stratified_subset(packs, args.mutation_train_packs, 0)
    val_subset = stratified_subset(validation_packs, args.mutation_validation_packs, 5)
    init_actions, _init_report = d59.run_rust_multi_bridge(out, repo_root, {"baseline": best}, val_subset, "hard_mutation_validation", f"{arm}_initial", started)
    init_rows = candidate_rows("baseline", val_subset, init_actions, "candidate")
    best_score, best_metrics = score_rows(init_rows, objective)
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
        subset = stratified_subset(train_subset, args.mutation_train_packs, gen)
        actions, _report = d59.run_rust_multi_bridge(out, repo_root, candidates, subset, "hard_mutation_train", f"{arm}_gen_{gen:04d}", started)
        scored = []
        for candidate_id in candidates:
            rows = candidate_rows(candidate_id, subset, actions, "candidate")
            score, metrics = score_rows(rows, objective)
            scored.append((score, candidate_id, metrics))
        scored.sort(key=lambda item: (item[0], -item[2]["average_total_support_used"]), reverse=True)
        top_score, top_id, top_metrics = scored[0]
        top_controller = candidates[top_id]
        val_actions, _val_report = d59.run_rust_multi_bridge(out, repo_root, {top_id: top_controller}, val_subset, "hard_mutation_validation", f"{arm}_gen_{gen:04d}", started)
        val_rows = candidate_rows(top_id, val_subset, val_actions, "candidate")
        val_score, val_metrics = score_rows(val_rows, objective)
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
        "train_subset_rows": len(train_subset),
        "validation_subset_rows": len(val_subset),
    }
    return best, report, history


def record_batch(batch, outputs, sample_counts, path):
    for row in batch:
        outputs.append(row)
        key = (row["track"], row["arm"], row["support_regime"])
        if sample_counts[key] < ROW_SAMPLE_PER_ARM_REGIME_SPLIT:
            append_jsonl(path, row)
            sample_counts[key] += 1


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
            "best_recent": partial["by_arm_core"].get("SUPPORT_COST_TARGETED_MUTATION", {}),
        },
    )
    append_progress(out, "eval_progress", started, {"split": split, "track": track, "completed_outputs": completed})


def evaluate_pack_all_arms(pack, idx, rust_actions):
    rows = []
    for arm in RUST_POLICY_ARMS:
        action_record = rust_actions[arm][idx]
        rows.append(output_from_action(pack, arm, action_record["action"], d59.rust_trace(action_record)))
    canonical_action = rust_actions["D59_BEST_REPLAY"][idx]
    shuffle = d55.spike_shuffle_mapping()
    rows.append(output_from_action(pack, "SPIKE_SHUFFLE_CONTROL", shuffle[canonical_action["action"]], d59.rust_trace(canonical_action)))
    rows.append(output_from_action(pack, "RANDOM_POLICY_CONTROL", d51.random_policy_action(f"D60:{pack['track']}:{pack['row_id']}"), d59.disabled_rust_trace()))
    rows.append(output_from_action(pack, "GREEDY_DECIDE_CONTROL", "DECIDE", d59.disabled_rust_trace()))
    rows.append(output_from_action(pack, "ALWAYS_COUNTER_CONTROL", "REQUEST_JOINT_COUNTER", d59.disabled_rust_trace()))
    return rows


def evaluate_packs(packs, controllers, out_path, out, split, track, started, heartbeat_sec, repo_root):
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


def regime_accuracy(metrics, arm, regime):
    return metrics["by_arm_and_regime"][arm][regime]["accuracy"]


def cost_adjusted(row):
    return row["exact_joint_accuracy"] - 0.002 * row["average_total_support_used"] - 2.0 * row["false_confidence_rate"]


def best_mutated_arm(metrics):
    core = metrics["by_arm_core"]
    regimes = metrics["by_arm_and_regime"]
    return max(
        [arm for arm in MUTATED_ARMS if arm in core],
        key=lambda arm: (
            regimes[arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["abstain_rate"],
            min(
                regimes[arm]["CORRELATED_ECHO_SUPPORT"]["accuracy"],
                regimes[arm]["ADVERSARIAL_DISTRACTOR_SUPPORT"]["accuracy"],
            ),
            cost_adjusted(core[arm]),
            core[arm]["exact_joint_accuracy"],
            -core[arm]["average_total_support_used"],
        ),
    )


def track_rust_usage_ok(metrics):
    usage = metrics["rust_usage"]
    return all(usage.get(arm, {}).get("rust_rows", 0) > 0 and usage.get(arm, {}).get("python_fallback_rows", 0) == 0 for arm in RUST_POLICY_ARMS)


def learning_gain_report(hard_metrics):
    core = hard_metrics["by_arm_core"]
    best = best_mutated_arm(hard_metrics)
    base = core["D58_REPLAY_REFERENCE"]
    row = core[best]
    exact_gain = row["exact_joint_accuracy"] - base["exact_joint_accuracy"]
    cost_gain = cost_adjusted(row) - cost_adjusted(base)
    support_delta = row["average_total_support_used"] - base["average_total_support_used"]
    same_accuracy = row["exact_joint_accuracy"] >= base["exact_joint_accuracy"] - 0.002
    learning_success = exact_gain >= 0.03 or cost_gain >= 0.02 or (support_delta <= -0.25 and same_accuracy)
    return {
        "best_arm": best,
        "baseline_arm": "D58_REPLAY_REFERENCE",
        "exact_gain": exact_gain,
        "cost_adjusted_gain": cost_gain,
        "support_delta": support_delta,
        "same_accuracy_for_support_gain": same_accuracy,
        "learning_success": learning_success,
        "best": row,
        "baseline": base,
    }


def stability_report(sat_metrics, d59_summary):
    best = best_mutated_arm(sat_metrics)
    row = sat_metrics["by_arm_core"][best]
    d58_exact = float((d59_summary.get("key_metrics") or {}).get("exact_joint_accuracy", 0.9994))
    indist = sat_metrics["by_arm_and_regime"][best]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]
    stable = row["exact_joint_accuracy"] >= d58_exact - 0.002 and indist["false_confidence_rate"] <= 0.01 and track_rust_usage_ok(sat_metrics)
    return {
        "best_arm": best,
        "d59_reference_exact": d58_exact,
        "exact_joint_accuracy": row["exact_joint_accuracy"],
        "false_confidence": indist["false_confidence_rate"],
        "indistinguishable_abstain": indist["abstain_rate"],
        "rust_usage_ok": track_rust_usage_ok(sat_metrics),
        "stable": stable,
    }


def mutation_path_exercised(reports):
    return any(report.get("accepted_total", 0) > 0 for report in reports.values())


def make_decision(sat_report, gain_report, oracle_report, failed_jobs, mutation_reports):
    selected = oracle_report.get("selected_variant", {})
    oracle_ok = bool(selected.get("valid_oracle", False))
    best = gain_report["best_arm"]
    if failed_jobs:
        return {
            "decision": "rust_sparse_mutation_safety_failure",
            "verdict": "D60_RUST_SPARSE_MUTATION_SAFETY_FAILURE",
            "next": "D60S_SAFETY_FITNESS_REPAIR",
            "best_arm": best,
            "boundary": BOUNDARY,
        }
    if not oracle_ok:
        return {
            "decision": "d60_hard_task_invalid",
            "verdict": "D60_HARD_TASK_INVALID",
            "next": "D60H_HARD_TASK_REDESIGN",
            "best_arm": best,
            "boundary": BOUNDARY,
        }
    best_safety = gain_report["best"]
    hard_safety = best_safety["false_confidence_rate"] <= 0.01 and mutation_path_exercised(mutation_reports)
    if not sat_report["stable"] or not hard_safety:
        return {
            "decision": "rust_sparse_mutation_safety_failure",
            "verdict": "D60_RUST_SPARSE_MUTATION_SAFETY_FAILURE",
            "next": "D60S_SAFETY_FITNESS_REPAIR",
            "best_arm": best,
            "boundary": BOUNDARY,
        }
    if gain_report["learning_success"]:
        return {
            "decision": "rust_sparse_mutation_learning_signal_confirmed",
            "verdict": "D60_RUST_SPARSE_MUTATION_LEARNING_SIGNAL_CONFIRMED",
            "next": "D61_RUST_SPARSE_MUTATION_SCALE_CONFIRM",
            "best_arm": best,
            "boundary": BOUNDARY,
        }
    return {
        "decision": "rust_sparse_mutation_path_confirmed_no_learning_signal",
        "verdict": "D60_RUST_SPARSE_MUTATION_PATH_CONFIRMED_NO_LEARNING_SIGNAL",
        "next": "D60C_MUTATION_SEARCH_SPACE_REPAIR",
        "best_arm": best,
        "boundary": BOUNDARY,
    }


def make_reports(out, aggregate, decision):
    sat = aggregate["test_metrics_by_track"][SATURATED_TRACK]
    hard = aggregate["test_metrics_by_track"][HARD_TRACK]
    hard_core = hard["by_arm_core"]
    reports = {
        "d59_upstream_manifest.json": aggregate["d59_upstream_manifest"],
        "task_difficulty_report.json": aggregate["task_difficulty_report"],
        "oracle_upper_bound_report.json": aggregate["oracle_upper_bound_report"],
        "saturated_track_report.json": aggregate["saturated_track_report"],
        "hard_learning_track_report.json": aggregate["hard_learning_track_report"],
        "mutation_causality_report.json": {
            "claim": "D60 attributes learning only when mutated Rust controllers improve hard-track metrics over D58 replay under identical hard packs.",
            "best_arm": decision.get("best_arm"),
            "gain_report": aggregate["learning_gain_report"],
            "mutation_path_exercised": mutation_path_exercised(aggregate["mutation_reports"]),
            "no_formula_solver_learning": True,
            "truth_hidden_from_controller_inputs": True,
        },
        "accepted_mutation_delta_report.json": {
            arm: {
                "initial_validation_score": report["initial_validation_score"],
                "final_validation_score": report["final_validation_score"],
                "delta_validation_score": report["final_validation_score"] - report["initial_validation_score"],
                "accepted_total": report["accepted_total"],
                "accepted_mutation_counts": report["accepted_mutation_counts"],
            }
            for arm, report in aggregate["mutation_reports"].items()
        },
        "pareto_frontier_report.json": {
            arm: {
                "exact_joint_accuracy": hard_core[arm]["exact_joint_accuracy"],
                "support": hard_core[arm]["average_total_support_used"],
                "false_confidence": hard_core[arm]["false_confidence_rate"],
                "cost_adjusted": cost_adjusted(hard_core[arm]),
            }
            for arm in sorted(hard_core, key=lambda item: (-cost_adjusted(hard_core[item]), hard_core[item]["average_total_support_used"]))
        },
        "support_cost_frontier_report.json": {
            "saturated": {
                arm: {
                    "exact_joint_accuracy": sat["by_arm_core"][arm]["exact_joint_accuracy"],
                    "support": sat["by_arm_core"][arm]["average_total_support_used"],
                    "cost_adjusted": cost_adjusted(sat["by_arm_core"][arm]),
                }
                for arm in sat["by_arm_core"]
            },
            "hard": {
                arm: {
                    "exact_joint_accuracy": hard_core[arm]["exact_joint_accuracy"],
                    "support": hard_core[arm]["average_total_support_used"],
                    "cost_adjusted": cost_adjusted(hard_core[arm]),
                }
                for arm in hard_core
            },
        },
        "safety_constraint_report.json": {
            "saturated_stable": aggregate["saturated_track_report"]["stable"],
            "hard_false_confidence_by_arm": {
                arm: hard["by_arm_and_regime"][arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["false_confidence_rate"]
                for arm in ARMS
            },
            "hard_abstain_by_arm": {
                arm: hard["by_arm_and_regime"][arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["abstain_rate"]
                for arm in ARMS
            },
            "rust_usage": {"saturated": sat["rust_usage"], "hard": hard["rust_usage"]},
        },
        "fitness_definition_report.json": {
            "accuracy_only": "exact + stress bonus + safety guard - tiny support penalty",
            "cost_only": "exact - stronger support penalty - budget violation penalty + safety guard",
            "support_cost_targeted": "exact + stress bonus - support/budget penalties + safety guard",
            "hard_stress": "explicitly weights correlated/adversarial stress",
            "pareto": "combines exact, stress, and cost-adjusted score",
            "novelty": "adds small action-diversity reward while preserving safety guard",
        },
    }
    for name, value in reports.items():
        write_json(out / name, value)
    return reports


def write_report(out, aggregate, decision):
    hard = aggregate["test_metrics_by_track"][HARD_TRACK]
    sat = aggregate["test_metrics_by_track"][SATURATED_TRACK]
    lines = [
        "# D60 Rust Sparse Mutation Learning Signal Probe Result",
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
        "Track summary:",
        "",
        "| track | arm | exact | corr | adv | support | false_conf |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for track, metrics in [(SATURATED_TRACK, sat), (HARD_TRACK, hard)]:
        arm = decision.get("best_arm") if track == HARD_TRACK else aggregate["saturated_track_report"]["best_arm"]
        core = metrics["by_arm_core"][arm]
        lines.append(
            f"| {track} | {arm} | {core['exact_joint_accuracy']:.6f} | "
            f"{regime_accuracy(metrics, arm, 'CORRELATED_ECHO_SUPPORT'):.6f} | "
            f"{regime_accuracy(metrics, arm, 'ADVERSARIAL_DISTRACTOR_SUPPORT'):.6f} | "
            f"{core['average_total_support_used']:.3f} | {core['false_confidence_rate']:.6f} |"
        )
    gain = aggregate["learning_gain_report"]
    lines += [
        "",
        "Learning gain over D58 replay on hard track:",
        "",
        "```json",
        json.dumps(
            {
                "best_arm": gain["best_arm"],
                "exact_gain": gain["exact_gain"],
                "cost_adjusted_gain": gain["cost_adjusted_gain"],
                "support_delta": gain["support_delta"],
                "learning_success": gain["learning_success"],
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "Selected hard variant:",
        "",
        "```json",
        json.dumps(aggregate["oracle_upper_bound_report"].get("selected_variant"), indent=2, sort_keys=True),
        "```",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_summary(aggregate, decision):
    hard = aggregate["test_metrics_by_track"][HARD_TRACK]
    best = decision.get("best_arm") or aggregate["learning_gain_report"]["best_arm"]
    core = hard["by_arm_core"][best]
    return {
        "task": TASK,
        "decision": decision["decision"],
        "verdict": decision.get("verdict"),
        "next": decision.get("next"),
        "best_arm": best,
        "hard_variant": aggregate["oracle_upper_bound_report"].get("selected_variant_name"),
        "hard_key_metrics": {
            "exact_joint_accuracy": core["exact_joint_accuracy"],
            "correlated_echo": regime_accuracy(hard, best, "CORRELATED_ECHO_SUPPORT"),
            "adversarial_distractor": regime_accuracy(hard, best, "ADVERSARIAL_DISTRACTOR_SUPPORT"),
            "support": core["average_total_support_used"],
            "false_confidence": core["false_confidence_rate"],
        },
        "learning_gain": {
            "exact_gain": aggregate["learning_gain_report"]["exact_gain"],
            "cost_adjusted_gain": aggregate["learning_gain_report"]["cost_adjusted_gain"],
            "support_delta": aggregate["learning_gain_report"]["support_delta"],
        },
        "saturated_stable": aggregate["saturated_track_report"]["stable"],
        "rust_path_invoked": aggregate["rust_path_invoked"],
        "fallback_rows": aggregate["fallback_rows"],
        "boundary": BOUNDARY,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="11401,11402,11403,11404,11405")
    parser.add_argument("--train-rows-per-seed", type=int, default=800)
    parser.add_argument("--test-rows-per-seed", type=int, default=800)
    parser.add_argument("--ood-rows-per-seed", type=int, default=800)
    parser.add_argument("--generations", type=int, default=180)
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="50-75")
    parser.add_argument("--heartbeat-sec", type=float, default=20.0)
    parser.add_argument("--heartbeat-generations", type=int, default=8)
    parser.add_argument("--mutation-train-packs", type=int, default=384)
    parser.add_argument("--mutation-validation-packs", type=int, default=384)
    parser.add_argument("--difficulty-probe-packs", type=int, default=512)
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

    d59_manifest = make_d59_upstream_manifest(repo_root)
    write_json(out / "d59_upstream_manifest.json", d59_manifest)
    base_controller = load_d58_controller(repo_root)
    if base_controller is None:
        failed_jobs.append("missing_d58_or_d56_controller")
    d59_best, d59_best_name = load_d59_best_controller(repo_root)
    if d59_best is None:
        d59_best = copy.deepcopy(base_controller)
        d59_best_name = "D58_FALLBACK"
        failed_jobs.append("missing_d59_best_controller")

    bundle = d55.d49_bundle()
    write_json(
        out / "dataset_manifest.json",
        {
            "task": "controlled_symbolic_joint_formula_discovery",
            "primary_space": PRIMARY_SPACE,
            "support_count": SUPPORT_COUNT,
            "regimes": REGIMES,
            "core_regimes": CORE_REGIMES,
            "feature_names": FEATURE_NAMES,
            "truth_hidden_from_controller_inputs": True,
            "controller_only_not_formula_solver": True,
            "formula_solver_learning_used": False,
            "tracks": [SATURATED_TRACK, HARD_TRACK],
            "hard_variants": HARD_VARIANTS,
        },
    )

    train_rows = d51.make_rows_with_progress(seeds, args.train_rows_per_seed, "train", bundle, out, started, args.heartbeat_sec)
    test_rows = d51.make_rows_with_progress(seeds, args.test_rows_per_seed, "test", bundle, out, started, args.heartbeat_sec)
    ood_rows = d51.make_rows_with_progress(seeds, args.ood_rows_per_seed, "ood", bundle, out, started, args.heartbeat_sec)
    train_base = d51.build_packs(train_rows, bundle, out, started, args.heartbeat_sec, args.workers, "train")
    test_base = d51.build_packs(test_rows, bundle, out, started, args.heartbeat_sec, args.workers, "test")
    ood_base = d51.build_packs(ood_rows, bundle, out, started, args.heartbeat_sec, args.workers, "ood")
    append_progress(out, "base_packs_built", started, {"train": len(train_base), "test": len(test_base), "ood": len(ood_base)})

    selected_variant, difficulty_report = choose_hard_variant(train_base[: min(len(train_base), args.difficulty_probe_packs)], base_controller, repo_root, out, started)
    if selected_variant is None:
        selected_variant = HARD_VARIANTS[-1]
        failed_jobs.append("no_valid_hard_variant_found")
    write_json(out / "task_difficulty_report.json", difficulty_report)

    train_saturated = [saturate_pack(pack) for pack in train_base]
    test_saturated = [saturate_pack(pack) for pack in test_base]
    ood_saturated = [saturate_pack(pack) for pack in ood_base]
    train_hard = [harden_pack(pack, selected_variant) for pack in train_base]
    test_hard = [harden_pack(pack, selected_variant) for pack in test_base]
    ood_hard = [harden_pack(pack, selected_variant) for pack in ood_base]

    oracle_report = {
        "variants": difficulty_report,
        "selected_variant_name": selected_variant["name"],
        "selected_variant": difficulty_report.get(selected_variant["name"], {}),
        "test_oracle": oracle_upper_bound_for_packs(test_hard),
        "ood_oracle": oracle_upper_bound_for_packs(ood_hard),
    }
    write_json(out / "oracle_upper_bound_report.json", oracle_report)

    midpoint = max(1, len(train_hard) // 2)
    mutation_train_packs = train_hard[:midpoint]
    mutation_validation_packs = train_hard[midpoint:]
    append_progress(out, "hard_packs_built", started, {"variant": selected_variant["name"], "train": len(mutation_train_packs), "validation": len(mutation_validation_packs)})

    controllers = {
        "D58_REPLAY_REFERENCE": copy.deepcopy(base_controller),
        "D59_BEST_REPLAY": copy.deepcopy(d59_best),
        "MUTATION_DISABLED_CONTROL": copy.deepcopy(base_controller),
        "RANDOM_MUTATION_CONTROL": d59.random_mutated_controller(base_controller, stable_rng(60_999, "random_mutation_control"), max(2, args.generations // 8)),
        "THRESHOLD_ABLATION": d55.make_threshold_ablation(base_controller),
        "REWIRE_ABLATION": d55.make_rewire_ablation(base_controller),
    }
    objective_map = {
        "COST_ONLY_MUTATION_CONTROL": "cost_only",
        "ACCURACY_ONLY_MUTATION": "accuracy_only",
        "SUPPORT_COST_TARGETED_MUTATION": "support_cost_targeted",
        "HARD_STRESS_MUTATION": "hard_stress",
        "MULTI_OBJECTIVE_PARETO_MUTATION": "pareto",
        "LARGE_STEP_MUTATION": "hard_stress",
        "STRUCTURED_GATE_MUTATION": "support_cost_targeted",
        "NOVELTY_DIVERSITY_MUTATION": "novelty",
    }
    mutation_reports = {}
    mutation_histories = {}
    for arm, objective in objective_map.items():
        controller, report, history = train_mutation_controller(base_controller, mutation_train_packs, mutation_validation_packs, arm, objective, args, out, repo_root, started)
        controllers[arm] = controller
        mutation_reports[arm] = report
        mutation_histories[arm] = history[-10:]

    test_outputs = []
    ood_outputs = []
    row_test = out / "row_outputs_test.jsonl"
    row_ood = out / "row_outputs_ood.jsonl"
    if row_test.exists():
        row_test.unlink()
    if row_ood.exists():
        row_ood.unlink()
    sat_test_outputs, sat_test_rust = evaluate_packs(test_saturated, controllers, row_test, out, "test", SATURATED_TRACK, started, args.heartbeat_sec, repo_root)
    hard_test_outputs, hard_test_rust = evaluate_packs(test_hard, controllers, row_test, out, "test", HARD_TRACK, started, args.heartbeat_sec, repo_root)
    sat_ood_outputs, sat_ood_rust = evaluate_packs(ood_saturated, controllers, row_ood, out, "ood", SATURATED_TRACK, started, args.heartbeat_sec, repo_root)
    hard_ood_outputs, hard_ood_rust = evaluate_packs(ood_hard, controllers, row_ood, out, "ood", HARD_TRACK, started, args.heartbeat_sec, repo_root)
    test_outputs.extend(sat_test_outputs + hard_test_outputs)
    ood_outputs.extend(sat_ood_outputs + hard_ood_outputs)

    test_metrics_by_track = by_track_metrics(test_outputs)
    ood_metrics_by_track = by_track_metrics(ood_outputs)
    d59_summary = d59_manifest.get("summary") or {}
    sat_report = stability_report(test_metrics_by_track[SATURATED_TRACK], d59_summary)
    gain_report = learning_gain_report(test_metrics_by_track[HARD_TRACK])
    decision = make_decision(sat_report, gain_report, oracle_report, failed_jobs, mutation_reports)

    fallback_rows = 0
    rust_rows = 0
    for metrics in test_metrics_by_track.values():
        for arm, counts in metrics["rust_usage"].items():
            fallback_rows += counts.get("python_fallback_rows", 0) if arm in RUST_POLICY_ARMS else 0
            rust_rows += counts.get("rust_rows", 0) if arm in RUST_POLICY_ARMS else 0

    aggregate = {
        "task": TASK,
        "failed_jobs": failed_jobs,
        "d59_upstream_manifest": d59_manifest,
        "d59_best_loaded_arm": d59_best_name,
        "selected_hard_variant": selected_variant,
        "task_difficulty_report": difficulty_report,
        "oracle_upper_bound_report": oracle_report,
        "test_metrics_by_track": test_metrics_by_track,
        "ood_metrics_by_track": ood_metrics_by_track,
        "rust_invocation_report": {
            "test_saturated": sat_test_rust,
            "test_hard": hard_test_rust,
            "ood_saturated": sat_ood_rust,
            "ood_hard": hard_ood_rust,
        },
        "mutation_reports": mutation_reports,
        "mutation_history_tail": mutation_histories,
        "saturated_track_report": sat_report,
        "learning_gain_report": gain_report,
        "hard_learning_track_report": {
            "best_arm": gain_report["best_arm"],
            "baseline_arm": gain_report["baseline_arm"],
            "learning_success": gain_report["learning_success"],
            "exact_gain": gain_report["exact_gain"],
            "cost_adjusted_gain": gain_report["cost_adjusted_gain"],
            "support_delta": gain_report["support_delta"],
            "selected_variant": selected_variant,
        },
        "rust_path_invoked": rust_rows > 0,
        "fallback_rows": fallback_rows,
        "boundary": BOUNDARY,
    }
    aggregate["decision"] = decision
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    reports = make_reports(out, aggregate, decision)
    write_json(out / "summary.json", make_summary(aggregate, decision))
    write_report(out, aggregate, decision)
    write_json(out / "trained_policy_manifest.json", {"controllers": controllers, "mutation_reports": mutation_reports, "decision": decision})
    append_progress(out, "completed", started, {"decision": decision["decision"], "reports": sorted(reports.keys())})
    print(json.dumps(make_summary(aggregate, decision), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
