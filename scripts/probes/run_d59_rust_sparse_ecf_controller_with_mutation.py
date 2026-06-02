#!/usr/bin/env python3
"""D59 Rust sparse ECF controller with mutation."""

import argparse
import copy
import json
import os
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import run_d51_mutable_ecf_controller_prototype as d51
import run_d53_mutable_ecf_integration_with_vraxion_mutation_architecture as d53
import run_d55_sparse_firing_ecf_controller_prototype as d55
import run_d57_canonical_rust_sparse_path_bridge as d57

PRIMARY_SPACE = d55.PRIMARY_SPACE
SUPPORT_COUNT = d55.SUPPORT_COUNT
REGIMES = d55.REGIMES
CORE_REGIMES = d55.CORE_REGIMES
ACTIONS = d55.ACTIONS
FEATURE_NAMES = d55.FEATURE_NAMES
ROW_SAMPLE_PER_ARM_REGIME_SPLIT = d55.ROW_SAMPLE_PER_ARM_REGIME_SPLIT
BRIDGE_CALLS = ["Network::new", "Network::graph_mut().add_edge", "Network::propagate_sparse", "Network::spike_data"]

BOUNDARY = (
    "D59 only tests mutation and selection of a canonical Rust sparse ECF action controller "
    "for controlled symbolic joint formula discovery. It does not prove full VRAXION brain, "
    "raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome success, "
    "architecture superiority, or production readiness."
)

RUST_POLICY_ARMS = [
    "D58_RUST_REPLAY_REFERENCE",
    "RUST_SPARSE_MUTATION_CONTROLLER",
    "RUST_SPARSE_MUTATION_CONTROLLER_COST_WEIGHTED",
    "RUST_SPARSE_MUTATION_CONTROLLER_FALSE_CONFIDENCE_WEIGHTED",
    "MUTATION_DISABLED_CONTROL",
    "RANDOM_MUTATION_CONTROL",
    "RUST_THRESHOLD_ABLATION",
    "RUST_REWIRE_ABLATION",
]

CONTROL_ARMS = [
    "RANDOM_POLICY_CONTROL",
    "GREEDY_DECIDE_CONTROL",
    "ALWAYS_COUNTER_CONTROL",
    "RUST_SPIKE_SHUFFLE_CONTROL",
]

ARMS = RUST_POLICY_ARMS + CONTROL_ARMS


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


def make_d58_upstream_manifest(repo_root):
    root = repo_root / "target/pilot_wave/d58_canonical_rust_sparse_controller_scale_confirm/smoke"
    manifest = {
        "upstream": "D58_CANONICAL_RUST_SPARSE_CONTROLLER_SCALE_CONFIRM",
        "expected_decision": "canonical_rust_sparse_controller_scale_confirmed",
        "expected_next": "D59_RUST_SPARSE_ECF_CONTROLLER_WITH_MUTATION",
        "artifact_root": str(root),
        "decision_present": (root / "decision.json").exists(),
        "summary_present": (root / "summary.json").exists(),
        "trained_policy_manifest_present": (root / "trained_policy_manifest.json").exists(),
    }
    for name in ["decision.json", "summary.json", "rust_path_usage_report.json", "controller_comparison_report.json"]:
        value = load_json_if_present(root / name)
        if value is not None:
            manifest[name.replace(".json", "")] = value
    trained = load_json_if_present(root / "trained_policy_manifest.json")
    manifest["d58_canonical_controller_loaded"] = bool(
        trained and trained.get("controllers", {}).get("CANONICAL_RUST_SPARSE_CONTROLLER")
    )
    return manifest


def load_d58_controller(repo_root):
    trained = load_json_if_present(repo_root / "target/pilot_wave/d58_canonical_rust_sparse_controller_scale_confirm/smoke/trained_policy_manifest.json")
    if not trained:
        return None
    return trained.get("controllers", {}).get("CANONICAL_RUST_SPARSE_CONTROLLER")


def stable_rng(seed, tag):
    return random.Random(int(seed) + d51.stable_seed(tag))


def make_canonical_sparse_path_report(repo_root, invoked):
    audit = d55.canonical_sparse_firing_audit(repo_root)
    audit.update(
        {
            "canonical_rust_network_path_probe_arm": "RUST_SPARSE_MUTATION_CONTROLLER",
            "canonical_rust_network_path_available": audit["canonical_rust_network_audited"],
            "canonical_rust_network_path_invoked": bool(invoked),
            "canonical_rust_probe_status": "invoked_in_d59" if invoked else "not_invoked",
            "rust_bridge_uses_path_dependency": True,
            "rust_bridge_calls": BRIDGE_CALLS,
            "candidate_eval_path": "generated Rust multi-controller harness using instnct-core Network::propagate_sparse",
        }
    )
    return audit


def rust_multi_harness_source():
    return r'''
use instnct_core::{Network, PropagationConfig};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::process;

const ACTIONS: [&str; 6] = [
    "DECIDE",
    "REQUEST_SUPPORT",
    "REQUEST_COUNTER_TOP1_TOP2",
    "REQUEST_JOINT_COUNTER",
    "REQUEST_EXTERNAL_TEST",
    "ABSTAIN",
];

#[derive(Clone, Debug)]
struct Gate {
    feature_idx: usize,
    action_idx: usize,
    stored_threshold: u8,
    weight: i32,
    priority: i32,
    channel: u8,
    polarity: i32,
}

fn action_index(name: &str) -> usize {
    ACTIONS.iter().position(|item| *item == name).unwrap_or(0)
}

fn read_controllers(path: &str) -> BTreeMap<String, Vec<Gate>> {
    let raw = fs::read_to_string(path).unwrap_or_else(|err| {
        eprintln!("failed to read controller file {path}: {err}");
        process::exit(2);
    });
    let mut out: BTreeMap<String, Vec<Gate>> = BTreeMap::new();
    for (line_idx, line) in raw.lines().enumerate() {
        if line_idx == 0 || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() != 8 {
            eprintln!("bad controller line {line_idx}: expected 8 columns, got {}", cols.len());
            process::exit(2);
        }
        out.entry(cols[0].to_string()).or_default().push(Gate {
            feature_idx: cols[1].parse().unwrap(),
            action_idx: action_index(cols[2]),
            stored_threshold: cols[3].parse().unwrap(),
            weight: cols[4].parse().unwrap(),
            priority: cols[5].parse().unwrap(),
            channel: cols[6].parse().unwrap(),
            polarity: cols[7].parse().unwrap(),
        });
    }
    out
}

fn read_rows(path: &str) -> (usize, Vec<(String, Vec<u16>, Vec<i8>)>) {
    let raw = fs::read_to_string(path).unwrap_or_else(|err| {
        eprintln!("failed to read rows file {path}: {err}");
        process::exit(2);
    });
    let mut lines = raw.lines();
    let header = lines.next().unwrap_or("");
    let feature_count = header.split('\t').count().saturating_sub(1);
    let mut rows = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        let row_key = cols[0].to_string();
        let mut indices = Vec::<u16>::new();
        let mut values = Vec::<i8>::new();
        for (idx, raw) in cols.iter().skip(1).enumerate() {
            let val: i8 = raw.parse().unwrap_or(0);
            if val != 0 {
                indices.push(idx as u16);
                values.push(val);
            }
        }
        rows.push((row_key, indices, values));
    }
    (feature_count, rows)
}

fn build_network(feature_count: usize, gates: &[Gate]) -> Network {
    let gate_base = feature_count;
    let marker_base = feature_count + gates.len();
    let mut net = Network::new(feature_count + gates.len() + gates.len());
    net.set_all_threshold(15);
    net.set_all_channel(1);
    for (idx, gate) in gates.iter().enumerate() {
        let gate_idx = gate_base + idx;
        let marker_idx = marker_base + idx;
        net.set_threshold(gate_idx, gate.stored_threshold);
        net.set_channel(gate_idx, gate.channel);
        net.polarity_mut()[gate_idx] = if gate.polarity < 0 { -1 } else { 1 };
        net.graph_mut().add_edge(gate.feature_idx as u16, gate_idx as u16);
        net.graph_mut().add_edge(gate_idx as u16, marker_idx as u16);
    }
    net
}

fn choose_action(fired: &[usize], gates: &[Gate], default_action: usize) -> usize {
    if fired.is_empty() {
        return default_action;
    }
    let mut output_charge = [0i32; 6];
    for &idx in fired {
        let gate = &gates[idx];
        output_charge[gate.action_idx] += gate.weight * gate.polarity;
    }
    let mut priority_idx = fired[0];
    for &idx in fired.iter().skip(1) {
        let a = &gates[idx];
        let b = &gates[priority_idx];
        if (a.priority, a.weight, -(idx as i32)) > (b.priority, b.weight, -(priority_idx as i32)) {
            priority_idx = idx;
        }
    }
    if gates[priority_idx].priority >= 80 {
        return gates[priority_idx].action_idx;
    }
    let mut best = 0usize;
    for idx in 1..ACTIONS.len() {
        if (output_charge[idx], -(idx as i32)) > (output_charge[best], -(best as i32)) {
            best = idx;
        }
    }
    best
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        eprintln!("usage: d59_rust_multi_bridge <controllers.tsv> <rows.tsv> <actions.tsv> <default_action>");
        process::exit(2);
    }
    let controllers = read_controllers(&args[1]);
    let (feature_count, rows) = read_rows(&args[2]);
    let default_action = action_index(&args[4]);
    let config = PropagationConfig {
        ticks_per_token: 3,
        input_duration_ticks: 1,
        decay_interval_ticks: 0,
        use_refractory: false,
    };
    let mut out = String::from("controller_id\trow_key\taction\tfired_gate_count\tspike_update_count\tmarker_charge_sum\n");
    let mut propagate_calls = 0usize;
    for (controller_id, gates) in controllers.iter() {
        let marker_base = feature_count + gates.len();
        let mut net = build_network(feature_count, gates);
        for (row_key, indices, values) in rows.iter() {
            net.reset();
            net.propagate_sparse(indices, values, &config).unwrap_or_else(|err| {
                eprintln!("propagate_sparse failed on {controller_id}/{row_key}: {err}");
                process::exit(3);
            });
            propagate_calls += 1;
            let spike = net.spike_data();
            let mut fired = Vec::<usize>::new();
            let mut marker_charge_sum = 0u32;
            for gate_idx in 0..gates.len() {
                let marker_charge = spike[marker_base + gate_idx].charge as u32;
                marker_charge_sum += marker_charge;
                if marker_charge > 0 {
                    fired.push(gate_idx);
                }
            }
            let action_idx = choose_action(&fired, gates, default_action);
            out.push_str(&format!(
                "{controller_id}\t{row_key}\t{}\t{}\t{}\t{}\n",
                ACTIONS[action_idx],
                fired.len(),
                gates.len(),
                marker_charge_sum
            ));
        }
    }
    fs::write(&args[3], out)?;
    let mut stdout = io::stdout();
    writeln!(stdout, "controllers_processed={}", controllers.len())?;
    writeln!(stdout, "rows_processed={}", rows.len())?;
    writeln!(stdout, "propagate_sparse_calls={propagate_calls}")?;
    Ok(())
}
'''


def ensure_rust_multi_harness(out, repo_root):
    harness = out / "rust_mutation_harness"
    src = harness / "src"
    src.mkdir(parents=True, exist_ok=True)
    cargo = harness / "Cargo.toml"
    main = src / "main.rs"
    instnct_path = str((repo_root / "instnct-core").resolve()).replace("\\", "/")
    cargo.write_text(
        "\n".join(
            [
                "[package]",
                'name = "d59_rust_sparse_mutation_harness"',
                'version = "0.1.0"',
                'edition = "2021"',
                "",
                "[dependencies]",
                f'instnct-core = {{ path = "{instnct_path}" }}',
                "",
                "[workspace]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    main.write_text(rust_multi_harness_source(), encoding="utf-8")
    return harness


def write_controllers_tsv(path, controllers):
    feature_index = {name: idx for idx, name in enumerate(FEATURE_NAMES)}
    lines = ["controller_id\tfeature_idx\taction\tstored_threshold\tweight\tpriority\tchannel\tpolarity"]
    for controller_id, controller in controllers.items():
        for gate in controller["gates"]:
            lines.append(
                "\t".join(
                    [
                        controller_id,
                        str(feature_index[gate["feature"]]),
                        gate["action"],
                        str(int(gate["stored_threshold"])),
                        str(int(gate["weight"])),
                        str(int(gate["priority"])),
                        str(int(gate.get("channel", 1))),
                        str(int(gate.get("polarity", 1))),
                    ]
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rows_tsv(path, packs):
    lines = ["row_key\t" + "\t".join(FEATURE_NAMES)]
    for idx, pack in enumerate(packs):
        values = [str(int(d55.quant_feature(pack["features"], name))) for name in FEATURE_NAMES]
        lines.append(f"{idx}\t" + "\t".join(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_multi_actions(path):
    actions = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        controller_id, row_key, action, fired_count, spike_count, marker_charge = line.split("\t")
        actions[controller_id][int(row_key)] = {
            "action": action,
            "fired_gate_count": int(fired_count),
            "spike_update_count": int(spike_count),
            "marker_charge_sum": int(marker_charge),
        }
    return {key: dict(value) for key, value in actions.items()}


def run_rust_multi_bridge(out, repo_root, controllers, packs, split, tag, started):
    harness = ensure_rust_multi_harness(out, repo_root)
    work = out / "rust_bridge_inputs" / split / tag
    work.mkdir(parents=True, exist_ok=True)
    controllers_path = work / "controllers.tsv"
    rows_path = work / "rows.tsv"
    actions_path = work / "actions.tsv"
    write_controllers_tsv(controllers_path, controllers)
    write_rows_tsv(rows_path, packs)
    cmd = [
        "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        str(harness / "Cargo.toml"),
        "--",
        str(controllers_path),
        str(rows_path),
        str(actions_path),
        "DECIDE",
    ]
    append_progress(out, "rust_multi_bridge_start", started, {"split": split, "tag": tag, "controllers": len(controllers), "rows": len(packs)})
    proc = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
    report = {
        "split": split,
        "tag": tag,
        "controllers": sorted(controllers.keys()),
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr[-4000:],
        "rows_requested": len(packs),
        "actions_path": str(actions_path),
        "actions_present": actions_path.exists(),
    }
    if proc.returncode != 0:
        write_json(work / "rust_error.json", report)
        raise RuntimeError(f"Rust multi bridge failed for {split}/{tag}: {proc.stderr[-500:]}")
    actions = parse_multi_actions(actions_path)
    report["actions_rows_by_controller"] = {key: len(value) for key, value in actions.items()}
    write_json(work / "rust_invocation.json", report)
    append_progress(out, "rust_multi_bridge_done", started, {"split": split, "tag": tag, "controllers": len(actions)})
    return actions, report


def rust_trace(action_record):
    return {
        "fired_gates": [],
        "output_charge": {},
        "spike_updates": action_record["spike_update_count"],
        "total_input_charge": 0,
        "rust_network_path_invoked": True,
        "rust_propagate_sparse_called": True,
        "python_fallback_used": False,
        "fired_gate_count": action_record["fired_gate_count"],
        "marker_charge_sum": action_record["marker_charge_sum"],
    }


def disabled_rust_trace():
    return {
        "fired_gates": [],
        "output_charge": {},
        "spike_updates": 0,
        "total_input_charge": 0,
        "rust_network_path_invoked": False,
        "rust_propagate_sparse_called": False,
        "python_fallback_used": False,
        "fired_gate_count": 0,
        "marker_charge_sum": 0,
    }


def output_from_action(pack, arm, action, trace=None):
    row = d55.output_from_action(pack, arm, action, trace)
    row["rust_network_path_invoked"] = bool(trace and trace.get("rust_network_path_invoked"))
    row["rust_propagate_sparse_called"] = bool(trace and trace.get("rust_propagate_sparse_called"))
    row["python_fallback_used"] = bool(trace and trace.get("python_fallback_used"))
    row["rust_marker_charge_sum"] = int(trace.get("marker_charge_sum", 0)) if trace else 0
    return row


def mutate_controller(controller, rng):
    out = copy.deepcopy(controller)
    gate = rng.choice(out["gates"])
    before = copy.deepcopy(gate)
    roll = rng.random()
    if roll < 0.24:
        gate["threshold"] = max(0, min(17, int(gate["threshold"]) + rng.choice([-2, -1, 1, 2])))
        gate["stored_threshold"] = d55.stored_threshold(gate["threshold"])
        mutation_type = "gate_threshold"
    elif roll < 0.44:
        gate["weight"] = max(-32, min(48, int(gate["weight"]) + rng.choice([-4, -2, 2, 4])))
        gate["polarity"] = 1 if int(gate["weight"]) >= 0 else -1
        mutation_type = "gate_weight"
    elif roll < 0.61:
        gate["priority"] = max(0, min(120, int(gate["priority"]) + rng.choice([-10, -5, 5, 10])))
        mutation_type = "gate_priority"
    elif roll < 0.76:
        gate["channel"] = rng.randint(1, 8)
        mutation_type = "gate_channel"
    elif roll < 0.90:
        gate["action"] = rng.choice(ACTIONS)
        mutation_type = "gate_action_readout"
    else:
        idx = (ACTIONS.index(gate["action"]) + rng.choice([1, 2, 3])) % len(ACTIONS)
        gate["action"] = ACTIONS[idx]
        mutation_type = "edge_action_rewire"
    out["mutation_history"] = out.get("mutation_history", []) + [mutation_type]
    out["_last_mutation"] = {
        "type": mutation_type,
        "gate_id": gate.get("gate_id"),
        "before": before,
        "after": copy.deepcopy(gate),
    }
    return out, out["_last_mutation"]


def random_mutated_controller(base, rng, steps):
    out = copy.deepcopy(base)
    history = []
    for _ in range(max(1, steps)):
        out, mutation = mutate_controller(out, rng)
        history.append(mutation)
    out["mutation_history"] = [item["type"] for item in history]
    out["_random_mutation_trace"] = history
    return out


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
        rows.append(output_from_action(pack, arm_name, action_record["action"], rust_trace(action_record)))
    return rows


def score_rows(rows, objective):
    metrics = d51.summarize(rows)
    accuracy = metrics["exact_joint_accuracy"]
    support = metrics["average_total_support_used"]
    false_conf = metrics["false_confidence_rate"]
    external = metrics["average_external_test_used"]
    indist_rows = [row for row in rows if row["support_regime"] == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]
    external_rows = [row for row in rows if row["support_regime"] == "EXTERNAL_TEST_REQUIRED_SUPPORT"]
    indist_abstain = d51.mean([1.0 if row["abstained"] else 0.0 for row in indist_rows]) if indist_rows else 1.0
    external_accuracy = d51.mean([1.0 if row["correct"] else 0.0 for row in external_rows]) if external_rows else 1.0
    metrics["indistinguishable_abstain_rate"] = indist_abstain
    metrics["external_test_required_accuracy"] = external_accuracy
    guard = 0.25 * indist_abstain + 0.10 * external_accuracy
    if objective == "cost":
        score = accuracy - 0.006 * support - 0.35 * false_conf + guard
    elif objective == "false_confidence":
        score = accuracy - 0.002 * support - 2.0 * false_conf + guard
    else:
        score = accuracy - 0.001 * support - 0.50 * false_conf + 0.0001 * external + guard
    return score, metrics


def stratified_subset(packs, limit, offset=0):
    if len(packs) <= limit:
        return list(packs)
    groups = defaultdict(list)
    for pack in packs:
        groups[pack["support_regime"]].append(pack)
    selected = []
    regimes = [regime for regime in REGIMES if groups[regime]]
    cursor = offset
    while len(selected) < limit:
        made_progress = False
        for regime in regimes:
            group = groups[regime]
            idx = cursor % len(group)
            selected.append(group[idx])
            made_progress = True
            if len(selected) >= limit:
                break
        cursor += 1
        if not made_progress:
            break
    return selected[:limit]


def train_mutation_controller(base_controller, packs, validation_packs, objective, args, out, repo_root, started):
    rng = stable_rng(59_000, f"d59_{objective}")
    current = copy.deepcopy(base_controller)
    best = copy.deepcopy(base_controller)
    train_subset = stratified_subset(packs, args.mutation_train_packs, 0)
    val_subset = stratified_subset(validation_packs, args.mutation_validation_packs, 3)
    init_actions, init_report = run_rust_multi_bridge(out, repo_root, {"baseline": best}, val_subset, "mutation_validation", f"{objective}_initial", started)
    init_rows = candidate_rows("baseline", val_subset, init_actions, "candidate")
    best_score, best_metrics = score_rows(init_rows, objective)
    history = []
    mutation_counts = Counter()
    accepted_counts = Counter()
    rejected_counts = Counter()
    accepted_total = 0
    rejected_total = 0
    for gen in range(args.generations):
        candidates = {}
        candidate_mutations = {}
        for idx in range(args.population):
            candidate, mutation = mutate_controller(current, rng)
            candidate_id = f"g{gen:04d}_c{idx:03d}"
            candidates[candidate_id] = candidate
            candidate_mutations[candidate_id] = mutation
            mutation_counts[mutation["type"]] += 1
        subset = stratified_subset(train_subset, args.mutation_train_packs, gen)
        actions, report = run_rust_multi_bridge(out, repo_root, candidates, subset, "mutation_train", f"{objective}_gen_{gen:04d}", started)
        scored = []
        for candidate_id in candidates:
            rows = candidate_rows(candidate_id, subset, actions, "candidate")
            score, metrics = score_rows(rows, objective)
            scored.append((score, candidate_id, metrics))
        scored.sort(key=lambda item: (item[0], -item[2]["average_total_support_used"]), reverse=True)
        top_score, top_id, top_metrics = scored[0]
        top_controller = candidates[top_id]
        val_actions, val_report = run_rust_multi_bridge(out, repo_root, {top_id: top_controller}, val_subset, "mutation_validation", f"{objective}_gen_{gen:04d}", started)
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
        append_jsonl(out / f"mutation_history_{objective}.jsonl", record)
        if gen == 0 or (gen + 1) % max(1, args.heartbeat_generations) == 0:
            write_json(
                out / f"partial_mutation_{objective}.json",
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
            append_progress(out, "mutation_progress", started, {"objective": objective, "generation": gen, "best_score": best_score, "accepted_total": accepted_total})
    report = {
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
        key = (row["arm"], row["support_regime"])
        if sample_counts[key] < ROW_SAMPLE_PER_ARM_REGIME_SPLIT:
            append_jsonl(path, row)
            sample_counts[key] += 1


def write_partial_eval(out, split, outputs, completed, started):
    recent = outputs[-min(len(outputs), 2000):]
    partial = summarize_outputs(recent)
    write_json(
        out / f"partial_{split}_metrics_snapshot.json",
        {
            "split": split,
            "completed_outputs": completed,
            "elapsed_sec": time.time() - started,
            "mutation_core_recent": partial["by_arm_core"].get("RUST_SPARSE_MUTATION_CONTROLLER", {}),
        },
    )
    append_progress(out, "eval_progress", started, {"split": split, "completed_outputs": completed})


def evaluate_pack_all_arms(pack, idx, rust_actions):
    rows = []
    for arm in [
        "D58_RUST_REPLAY_REFERENCE",
        "RUST_SPARSE_MUTATION_CONTROLLER",
        "RUST_SPARSE_MUTATION_CONTROLLER_COST_WEIGHTED",
        "RUST_SPARSE_MUTATION_CONTROLLER_FALSE_CONFIDENCE_WEIGHTED",
        "MUTATION_DISABLED_CONTROL",
        "RANDOM_MUTATION_CONTROL",
        "RUST_THRESHOLD_ABLATION",
        "RUST_REWIRE_ABLATION",
    ]:
        action_record = rust_actions[arm][idx]
        rows.append(output_from_action(pack, arm, action_record["action"], rust_trace(action_record)))
    canonical_action = rust_actions["RUST_SPARSE_MUTATION_CONTROLLER"][idx]
    shuffle = d55.spike_shuffle_mapping()
    rows.append(output_from_action(pack, "RUST_SPIKE_SHUFFLE_CONTROL", shuffle[canonical_action["action"]], rust_trace(canonical_action)))
    rows.append(output_from_action(pack, "RANDOM_POLICY_CONTROL", d51.random_policy_action(f"D59:{pack['row_id']}"), disabled_rust_trace()))
    rows.append(output_from_action(pack, "GREEDY_DECIDE_CONTROL", "DECIDE", disabled_rust_trace()))
    rows.append(output_from_action(pack, "ALWAYS_COUNTER_CONTROL", "REQUEST_JOINT_COUNTER", disabled_rust_trace()))
    return rows


def evaluate_packs(packs, controllers, out_path, out, split, started, heartbeat_sec, repo_root):
    if out_path.exists():
        out_path.unlink()
    rust_actions, rust_report = run_rust_multi_bridge(out, repo_root, controllers, packs, split, "final_eval", started)
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
            write_partial_eval(out, split, outputs, completed, started)
    return outputs, rust_report


def regime_accuracy(metrics, arm, regime):
    return metrics["by_arm_and_regime"][arm][regime]["accuracy"]


def best_mutated_arm(metrics):
    core = metrics["by_arm_core"]
    regimes = metrics["by_arm_and_regime"]
    candidates = [
        "RUST_SPARSE_MUTATION_CONTROLLER",
        "RUST_SPARSE_MUTATION_CONTROLLER_COST_WEIGHTED",
        "RUST_SPARSE_MUTATION_CONTROLLER_FALSE_CONFIDENCE_WEIGHTED",
    ]
    return max(
        candidates,
        key=lambda arm: (
            regimes[arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["abstain_rate"],
            min(
                regimes[arm]["CORRELATED_ECHO_SUPPORT"]["accuracy"],
                regimes[arm]["ADVERSARIAL_DISTRACTOR_SUPPORT"]["accuracy"],
                regimes[arm]["EXTERNAL_TEST_REQUIRED_SUPPORT"]["accuracy"],
            ),
            core[arm]["exact_joint_accuracy"],
            -core[arm]["average_total_support_used"],
        ),
    )


def mutation_path_exercised(reports):
    return any(report.get("accepted_total", 0) > 0 for report in reports.values())


def gates_pass(metrics, failed_jobs, rust_usage, mutation_reports):
    if failed_jobs or not mutation_path_exercised(mutation_reports):
        return False
    arm = best_mutated_arm(metrics)
    core = metrics["by_arm_core"]
    regimes = metrics["by_arm_and_regime"]
    row = core[arm]
    corr = regime_accuracy(metrics, arm, "CORRELATED_ECHO_SUPPORT")
    adv = regime_accuracy(metrics, arm, "ADVERSARIAL_DISTRACTOR_SUPPORT")
    external = regime_accuracy(metrics, arm, "EXTERNAL_TEST_REQUIRED_SUPPORT")
    indist = regimes[arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]
    rust_rows = rust_usage.get(arm, {}).get("rust_rows", 0)
    fallback_rows = rust_usage.get(arm, {}).get("python_fallback_rows", 0)
    return (
        rust_rows > 0
        and fallback_rows == 0
        and row["exact_joint_accuracy"] >= 0.995
        and corr >= 0.995
        and adv >= 0.995
        and external >= 0.99
        and indist["abstain_rate"] >= 0.99
        and indist["false_confidence_rate"] <= 0.01
        and row["accuracy"] > core["RANDOM_POLICY_CONTROL"]["accuracy"]
        and row["accuracy"] > core["GREEDY_DECIDE_CONTROL"]["accuracy"]
        and row["accuracy"] > core["RUST_SPIKE_SHUFFLE_CONTROL"]["accuracy"]
        and row["accuracy"] > core["RUST_THRESHOLD_ABLATION"]["accuracy"]
        and row["accuracy"] > core["RUST_REWIRE_ABLATION"]["accuracy"]
        and core["ALWAYS_COUNTER_CONTROL"]["average_total_support_used"] > row["average_total_support_used"]
    )


def cost_adjusted(row):
    return row["exact_joint_accuracy"] - 0.002 * row["average_total_support_used"] - row["false_confidence_rate"]


def make_decision(metrics, failed_jobs, rust_usage, mutation_reports):
    if not mutation_path_exercised(mutation_reports):
        return {
            "decision": "rust_sparse_mutation_path_not_exercised",
            "verdict": "D59_RUST_SPARSE_MUTATION_PATH_NOT_EXERCISED",
            "next": "D59R_RUST_MUTATION_BRIDGE_REPAIR",
            "best_arm": best_mutated_arm(metrics) if metrics.get("by_arm_core") else None,
            "boundary": BOUNDARY,
        }
    arm = best_mutated_arm(metrics)
    fallback_rows = rust_usage.get(arm, {}).get("python_fallback_rows", 0)
    rust_rows = rust_usage.get(arm, {}).get("rust_rows", 0)
    if rust_rows == 0 or fallback_rows > 0:
        return {
            "decision": "rust_sparse_mutation_path_not_exercised",
            "verdict": "D59_RUST_SPARSE_MUTATION_PATH_NOT_CLEAN",
            "next": "D59R_RUST_MUTATION_BRIDGE_REPAIR",
            "best_arm": arm,
            "boundary": BOUNDARY,
        }
    if not gates_pass(metrics, failed_jobs, rust_usage, mutation_reports):
        return {
            "decision": "rust_sparse_mutation_controller_not_confirmed",
            "verdict": "D59_RUST_SPARSE_MUTATION_CONTROLLER_NOT_CONFIRMED",
            "next": "D59_REPAIR",
            "best_arm": arm,
            "boundary": BOUNDARY,
        }
    core = metrics["by_arm_core"]
    improves = cost_adjusted(core[arm]) > cost_adjusted(core["D58_RUST_REPLAY_REFERENCE"]) + 0.0001
    if improves:
        return {
            "decision": "rust_sparse_mutation_controller_positive",
            "verdict": "D59_RUST_SPARSE_MUTATION_CONTROLLER_POSITIVE",
            "next": "D60_RUST_SPARSE_CONTROLLER_COST_OPTIMIZATION",
            "best_arm": arm,
            "boundary": BOUNDARY,
        }
    return {
        "decision": "rust_sparse_mutation_path_confirmed_no_gain",
        "verdict": "D59_RUST_SPARSE_MUTATION_PATH_CONFIRMED_NO_GAIN",
        "next": "D60_RUST_SPARSE_MUTATION_FITNESS_REPAIR",
        "best_arm": arm,
        "boundary": BOUNDARY,
    }


def make_reports(out, aggregate, decision, d58_manifest, mutation_reports, before_after):
    test = aggregate["test_metrics"]
    core = test["by_arm_core"]
    regimes = test["by_arm_and_regime"]
    rust_usage = test["rust_usage"]
    best = decision.get("best_arm") or best_mutated_arm(test)
    reports = {
        "d58_upstream_manifest.json": d58_manifest,
        "rust_mutation_representation_report.json": {
            "representation": "Rust-serializable sparse controller config TSV",
            "mutation_surface": ["gate_threshold", "gate_weight", "gate_priority", "gate_channel", "gate_action_readout", "edge_action_rewire"],
            "candidate_eval_path": "generated Rust multi-controller harness calling Network::propagate_sparse",
            "formula_solver_learning_used": False,
            "controller_only_not_formula_solver": True,
        },
        "rust_invocation_report.json": aggregate["rust_invocation_report"],
        "rust_path_usage_report.json": {
            "rust_propagate_sparse_called": rust_usage.get(best, {}).get("rust_rows", 0) > 0,
            "best_arm": best,
            "best_arm_fallback_rows": rust_usage.get(best, {}).get("python_fallback_rows", 0),
            "by_arm": rust_usage,
        },
        "python_fallback_audit.json": {
            "rust_arms_have_zero_fallback": all(rust_usage.get(arm, {}).get("python_fallback_rows", 0) == 0 for arm in RUST_POLICY_ARMS),
            "fallback_rows_by_arm": {arm: rust_usage.get(arm, {}).get("python_fallback_rows", 0) for arm in ARMS},
        },
        "mutation_acceptance_report.json": mutation_reports,
        "fitness_landscape_report.json": {
            key: {
                "initial_validation_score": value["initial_validation_score"],
                "final_validation_score": value["final_validation_score"],
                "accepted_total": value["accepted_total"],
                "rejected_total": value["rejected_total"],
            }
            for key, value in mutation_reports.items()
        },
        "before_after_mutation_report.json": before_after,
        "support_cost_frontier_report.json": {
            arm: {
                "exact_joint_accuracy": core[arm]["exact_joint_accuracy"],
                "support": core[arm]["average_total_support_used"],
                "counter_support": core[arm]["average_counter_support_used"],
                "external_test": core[arm]["average_external_test_used"],
                "cost_adjusted": cost_adjusted(core[arm]),
            }
            for arm in sorted(ARMS, key=lambda item: (core[item]["average_total_support_used"], -core[item]["exact_joint_accuracy"]))
        },
        "false_confidence_report.json": {
            arm: {
                "core_false_confidence": core[arm]["false_confidence_rate"],
                "indistinguishable_false_confidence": regimes[arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["false_confidence_rate"],
                "indistinguishable_abstain": regimes[arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["abstain_rate"],
            }
            for arm in ARMS
        },
        "regime_breakdown_report.json": regimes,
        "ablation_report.json": {
            arm: {
                "exact_joint_accuracy": core[arm]["exact_joint_accuracy"],
                "correlated_echo": regime_accuracy(test, arm, "CORRELATED_ECHO_SUPPORT"),
                "adversarial_distractor": regime_accuracy(test, arm, "ADVERSARIAL_DISTRACTOR_SUPPORT"),
                "support": core[arm]["average_total_support_used"],
            }
            for arm in ["RUST_SPIKE_SHUFFLE_CONTROL", "RUST_THRESHOLD_ABLATION", "RUST_REWIRE_ABLATION", "RANDOM_POLICY_CONTROL", "GREEDY_DECIDE_CONTROL"]
        },
        "controller_comparison_report.json": {
            arm: {
                "exact_joint_accuracy": core[arm]["exact_joint_accuracy"],
                "correlated_echo": regime_accuracy(test, arm, "CORRELATED_ECHO_SUPPORT"),
                "adversarial_distractor": regime_accuracy(test, arm, "ADVERSARIAL_DISTRACTOR_SUPPORT"),
                "external_test_required": regime_accuracy(test, arm, "EXTERNAL_TEST_REQUIRED_SUPPORT"),
                "indistinguishable_abstain": regimes[arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["abstain_rate"],
                "false_confidence": regimes[arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["false_confidence_rate"],
                "support": core[arm]["average_total_support_used"],
                "counter_support": core[arm]["average_counter_support_used"],
                "rust_network_path_invoked": rust_usage.get(arm, {}).get("rust_rows", 0) > 0,
                "cost_adjusted": cost_adjusted(core[arm]),
            }
            for arm in ARMS
        },
    }
    for name, value in reports.items():
        write_json(out / name, value)
    return reports


def write_report(out, aggregate, decision):
    test = aggregate["test_metrics"]
    core = test["by_arm_core"]
    regimes = test["by_arm_and_regime"]
    rust_usage = test["rust_usage"]
    lines = [
        "# D59 Rust Sparse ECF Controller With Mutation Result",
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
        "Controller comparison:",
        "",
        "| arm | exact | corr | adv | external | support | rust |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        lines.append(
            f"| {arm} | {core[arm]['exact_joint_accuracy']:.5f} | "
            f"{regimes[arm]['CORRELATED_ECHO_SUPPORT']['accuracy']:.5f} | "
            f"{regimes[arm]['ADVERSARIAL_DISTRACTOR_SUPPORT']['accuracy']:.5f} | "
            f"{regimes[arm]['EXTERNAL_TEST_REQUIRED_SUPPORT']['accuracy']:.5f} | "
            f"{core[arm]['average_total_support_used']:.3f} | "
            f"{rust_usage.get(arm, {}).get('rust_rows', 0)} |"
        )
    lines.extend(
        [
            "",
            "Mutation summary:",
            "",
            "```text",
            json.dumps({k: {"accepted_total": v["accepted_total"], "rejected_total": v["rejected_total"]} for k, v in aggregate["mutation_reports"].items()}, sort_keys=True),
            "```",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def make_summary(aggregate, decision):
    test = aggregate["test_metrics"]
    arm = decision.get("best_arm") or best_mutated_arm(test)
    core = test["by_arm_core"][arm]
    regimes = test["by_arm_and_regime"][arm]
    return {
        "task": "D59_RUST_SPARSE_ECF_CONTROLLER_WITH_MUTATION",
        "decision": decision["decision"],
        "verdict": decision.get("verdict"),
        "next": decision.get("next"),
        "best_arm": arm,
        "rust_path_invoked": test["rust_usage"].get(arm, {}).get("rust_rows", 0) > 0,
        "fallback_rows": test["rust_usage"].get(arm, {}).get("python_fallback_rows", 0),
        "mutation_path_exercised": mutation_path_exercised(aggregate["mutation_reports"]),
        "key_metrics": {
            "exact_joint_accuracy": core["exact_joint_accuracy"],
            "correlated_echo": regimes["CORRELATED_ECHO_SUPPORT"]["accuracy"],
            "adversarial_distractor": regimes["ADVERSARIAL_DISTRACTOR_SUPPORT"]["accuracy"],
            "external_test_required": regimes["EXTERNAL_TEST_REQUIRED_SUPPORT"]["accuracy"],
            "indistinguishable_abstain": regimes["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["abstain_rate"],
            "false_confidence": regimes["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["false_confidence_rate"],
            "support": core["average_total_support_used"],
            "cost_adjusted": cost_adjusted(core),
        },
        "boundary": BOUNDARY,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="11301,11302,11303,11304,11305")
    parser.add_argument("--train-rows-per-seed", type=int, default=800)
    parser.add_argument("--test-rows-per-seed", type=int, default=800)
    parser.add_argument("--ood-rows-per-seed", type=int, default=800)
    parser.add_argument("--generations", type=int, default=160)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="50-75")
    parser.add_argument("--heartbeat-sec", type=float, default=20.0)
    parser.add_argument("--heartbeat-generations", type=int, default=10)
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
    write_json(out / "queue.json", {"task": "D59_RUST_SPARSE_ECF_CONTROLLER_WITH_MUTATION", "args": vars(args), "seeds": seeds, "boundary": BOUNDARY})
    append_progress(out, "started", started, {"seeds": seeds, "generations": args.generations, "population": args.population})
    write_json(out / "compute_probe.json", {"cpu_count": os.cpu_count(), "workers": d51.worker_count_from_arg(args.workers), "cpu_target": args.cpu_target})

    d58_manifest = make_d58_upstream_manifest(repo_root)
    write_json(out / "d58_upstream_manifest.json", d58_manifest)
    base_controller = load_d58_controller(repo_root)
    if base_controller is None:
        failed_jobs.append("missing_d58_canonical_controller")
        base_controller = d57.load_d56_best_controller(repo_root)

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
            "train_validation_test_ood_separated": True,
        },
    )

    train_rows = d51.make_rows_with_progress(seeds, args.train_rows_per_seed, "train", bundle, out, started, args.heartbeat_sec)
    test_rows = d51.make_rows_with_progress(seeds, args.test_rows_per_seed, "test", bundle, out, started, args.heartbeat_sec)
    ood_rows = d51.make_rows_with_progress(seeds, args.ood_rows_per_seed, "ood", bundle, out, started, args.heartbeat_sec)
    train_packs = d51.build_packs(train_rows, bundle, out, started, args.heartbeat_sec, args.workers, "train")
    midpoint = max(1, len(train_packs) // 2)
    mutation_train_packs = train_packs[:midpoint]
    mutation_validation_packs = train_packs[midpoint:]
    test_packs = d51.build_packs(test_rows, bundle, out, started, args.heartbeat_sec, args.workers, "test")
    ood_packs = d51.build_packs(ood_rows, bundle, out, started, args.heartbeat_sec, args.workers, "ood")
    append_progress(out, "packs_built", started, {"train": len(mutation_train_packs), "validation": len(mutation_validation_packs), "test": len(test_packs), "ood": len(ood_packs)})

    controllers = {"D58_RUST_REPLAY_REFERENCE": copy.deepcopy(base_controller)}
    mutation_reports = {}
    mutation_histories = {}
    objective_map = {
        "RUST_SPARSE_MUTATION_CONTROLLER": "base",
        "RUST_SPARSE_MUTATION_CONTROLLER_COST_WEIGHTED": "cost",
        "RUST_SPARSE_MUTATION_CONTROLLER_FALSE_CONFIDENCE_WEIGHTED": "false_confidence",
    }
    for arm, objective in objective_map.items():
        controller, report, history = train_mutation_controller(base_controller, mutation_train_packs, mutation_validation_packs, objective, args, out, repo_root, started)
        controllers[arm] = controller
        mutation_reports[arm] = report
        mutation_histories[arm] = history[-10:]
    controllers["MUTATION_DISABLED_CONTROL"] = copy.deepcopy(base_controller)
    controllers["RANDOM_MUTATION_CONTROL"] = random_mutated_controller(base_controller, stable_rng(59_999, "random_mutation_control"), max(2, args.generations // 10))
    controllers["RUST_THRESHOLD_ABLATION"] = d55.make_threshold_ablation(base_controller)
    controllers["RUST_REWIRE_ABLATION"] = d55.make_rewire_ablation(base_controller)

    test_outputs, test_rust_report = evaluate_packs(test_packs, controllers, out / "row_outputs_test.jsonl", out, "test", started, args.heartbeat_sec, repo_root)
    ood_outputs, ood_rust_report = evaluate_packs(ood_packs, controllers, out / "row_outputs_ood.jsonl", out, "ood", started, args.heartbeat_sec, repo_root)
    test_metrics = summarize_outputs(test_outputs)
    ood_metrics = summarize_outputs(ood_outputs)
    canonical_report = make_canonical_sparse_path_report(repo_root, any(test_metrics["rust_usage"].get(arm, {}).get("rust_rows", 0) > 0 for arm in RUST_POLICY_ARMS))
    before_after = {
        arm: {
            "before": test_metrics["by_arm_core"]["D58_RUST_REPLAY_REFERENCE"],
            "after": test_metrics["by_arm_core"][arm],
            "delta_exact": test_metrics["by_arm_core"][arm]["exact_joint_accuracy"] - test_metrics["by_arm_core"]["D58_RUST_REPLAY_REFERENCE"]["exact_joint_accuracy"],
            "delta_support": test_metrics["by_arm_core"][arm]["average_total_support_used"] - test_metrics["by_arm_core"]["D58_RUST_REPLAY_REFERENCE"]["average_total_support_used"],
            "delta_cost_adjusted": cost_adjusted(test_metrics["by_arm_core"][arm]) - cost_adjusted(test_metrics["by_arm_core"]["D58_RUST_REPLAY_REFERENCE"]),
        }
        for arm in objective_map
    }
    aggregate = {
        "task": "D59_RUST_SPARSE_ECF_CONTROLLER_WITH_MUTATION",
        "failed_jobs": failed_jobs,
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
        "rust_invocation_report": {"test": test_rust_report, "ood": ood_rust_report},
        "mutation_reports": mutation_reports,
        "mutation_history_tail": mutation_histories,
        "before_after_mutation": before_after,
        "canonical_sparse_path_report": canonical_report,
        "boundary": BOUNDARY,
    }
    decision = make_decision(test_metrics, failed_jobs, test_metrics["rust_usage"], mutation_reports)
    aggregate["decision"] = decision
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    reports = make_reports(out, aggregate, decision, d58_manifest, mutation_reports, before_after)
    write_json(out / "summary.json", make_summary(aggregate, decision))
    write_report(out, aggregate, decision)
    write_json(out / "trained_policy_manifest.json", {"controllers": controllers, "mutation_reports": mutation_reports, "decision": decision})
    append_progress(out, "completed", started, {"decision": decision["decision"], "reports": sorted(reports.keys())})
    print(json.dumps(make_summary(aggregate, decision), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
