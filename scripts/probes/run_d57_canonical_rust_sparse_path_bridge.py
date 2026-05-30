#!/usr/bin/env python3
"""D57 canonical Rust sparse path bridge."""

import argparse
import copy
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import run_d51_mutable_ecf_controller_prototype as d51
import run_d53_mutable_ecf_integration_with_vraxion_mutation_architecture as d53
import run_d55_sparse_firing_ecf_controller_prototype as d55
import run_d56_sparse_firing_ecf_controller_scale_confirm as d56

PRIMARY_SPACE = d55.PRIMARY_SPACE
SUPPORT_COUNT = d55.SUPPORT_COUNT
REGIMES = d55.REGIMES
CORE_REGIMES = d55.CORE_REGIMES
ACTIONS = d55.ACTIONS
FEATURE_NAMES = d55.FEATURE_NAMES
ROW_SAMPLE_PER_ARM_REGIME_SPLIT = d55.ROW_SAMPLE_PER_ARM_REGIME_SPLIT

BOUNDARY = (
    "D57 only tests a canonical Rust sparse-path bridge for the ECF action controller on "
    "controlled symbolic joint formula discovery. It does not prove full VRAXION brain, raw "
    "visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture "
    "superiority, or production readiness."
)

ARMS = [
    "D56_PYTHON_SPARSE_REPLAY",
    "PYTHON_FALLBACK_REFERENCE",
    "CANONICAL_RUST_SPARSE_NETWORK_BRIDGE",
    "RUST_SPIKE_SHUFFLE_CONTROL",
    "RUST_THRESHOLD_ABLATION",
    "RUST_REWIRE_ABLATION",
    "RUST_PATH_DISABLED_CONTROL",
    "RANDOM_POLICY_CONTROL",
    "GREEDY_DECIDE_CONTROL",
    "ALWAYS_COUNTER_CONTROL",
]

RUST_ARMS = {
    "CANONICAL_RUST_SPARSE_NETWORK_BRIDGE",
    "RUST_THRESHOLD_ABLATION",
    "RUST_REWIRE_ABLATION",
}


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


def make_d56_upstream_manifest(repo_root):
    root = repo_root / "target/pilot_wave/d56_sparse_firing_ecf_controller_scale_confirm/smoke"
    manifest = {
        "upstream": "D56_SPARSE_FIRING_ECF_CONTROLLER_SCALE_CONFIRM",
        "expected_decision": "sparse_firing_controller_scale_confirmed_python_path_only",
        "expected_next": "D57_CANONICAL_RUST_SPARSE_PATH_BRIDGE",
        "artifact_root": str(root),
        "decision_present": (root / "decision.json").exists(),
        "summary_present": (root / "summary.json").exists(),
        "trained_policy_manifest_present": (root / "trained_policy_manifest.json").exists(),
    }
    for name in ["decision.json", "summary.json", "sparse_firing_usage_report.json", "canonical_sparse_path_report.json"]:
        value = load_json_if_present(root / name)
        if value is not None:
            manifest[name.replace(".json", "")] = value
    trained = load_json_if_present(root / "trained_policy_manifest.json")
    manifest["d56_best_sparse_loaded"] = bool(
        trained and trained.get("sparse_controllers", {}).get("REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION")
    )
    return manifest


def load_d56_best_controller(repo_root):
    trained = load_json_if_present(
        repo_root / "target/pilot_wave/d56_sparse_firing_ecf_controller_scale_confirm/smoke/trained_policy_manifest.json"
    )
    if not trained:
        return None
    return trained.get("sparse_controllers", {}).get("REAL_SPARSE_FIRING_CONTROLLER_WITH_MUTATION")


def make_canonical_sparse_path_report(repo_root, invoked):
    audit = d55.canonical_sparse_firing_audit(repo_root)
    audit.update(
        {
            "canonical_rust_network_path_probe_arm": "CANONICAL_RUST_SPARSE_NETWORK_BRIDGE",
            "canonical_rust_network_path_available": audit["canonical_rust_network_audited"],
            "canonical_rust_network_path_invoked": bool(invoked),
            "canonical_rust_probe_status": "invoked_in_d57" if invoked else "not_invoked",
            "rust_bridge_uses_path_dependency": True,
            "rust_bridge_calls": ["Network::new", "Network::graph_mut().add_edge", "Network::propagate_sparse"],
        }
    )
    return audit


def rust_harness_source():
    return r'''
use instnct_core::{Network, PropagationConfig};
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

fn read_gates(path: &str) -> Vec<Gate> {
    let raw = fs::read_to_string(path).unwrap_or_else(|err| {
        eprintln!("failed to read gate file {path}: {err}");
        process::exit(2);
    });
    let mut out = Vec::new();
    for (line_idx, line) in raw.lines().enumerate() {
        if line_idx == 0 || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() != 7 {
            eprintln!("bad gate line {line_idx}: expected 7 columns, got {}", cols.len());
            process::exit(2);
        }
        out.push(Gate {
            feature_idx: cols[0].parse().unwrap(),
            action_idx: action_index(cols[1]),
            stored_threshold: cols[2].parse().unwrap(),
            weight: cols[3].parse().unwrap(),
            priority: cols[4].parse().unwrap(),
            channel: cols[5].parse().unwrap(),
            polarity: cols[6].parse().unwrap(),
        });
    }
    out
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
        eprintln!("usage: d57_rust_bridge <controller.tsv> <rows.tsv> <actions.tsv> <default_action>");
        process::exit(2);
    }
    let controller_path = &args[1];
    let rows_path = &args[2];
    let out_path = &args[3];
    let default_action = action_index(&args[4]);
    let gates = read_gates(controller_path);
    let raw_rows = fs::read_to_string(rows_path)?;
    let mut lines = raw_rows.lines();
    let header = lines.next().unwrap_or("");
    let feature_count = header.split('\t').count().saturating_sub(1);
    let mut net = build_network(feature_count, &gates);
    let config = PropagationConfig {
        ticks_per_token: 3,
        input_duration_ticks: 1,
        decay_interval_ticks: 0,
        use_refractory: false,
    };
    let gate_base = feature_count;
    let marker_base = feature_count + gates.len();
    let mut out = String::from("row_key\taction\tfired_gate_count\tspike_update_count\tmarker_charge_sum\n");
    let mut row_count = 0usize;
    let mut propagate_calls = 0usize;
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        let row_key = cols[0];
        let mut indices = Vec::<u16>::new();
        let mut values = Vec::<i8>::new();
        for (idx, raw) in cols.iter().skip(1).enumerate() {
            let val: i8 = raw.parse().unwrap_or(0);
            if val != 0 {
                indices.push(idx as u16);
                values.push(val);
            }
        }
        net.reset();
        net.propagate_sparse(&indices, &values, &config).unwrap_or_else(|err| {
            eprintln!("propagate_sparse failed on {row_key}: {err}");
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
        let action_idx = choose_action(&fired, &gates, default_action);
        let _ = gate_base; // keeps the neuron layout explicit in compiled code.
        out.push_str(&format!(
            "{row_key}\t{}\t{}\t{}\t{}\n",
            ACTIONS[action_idx],
            fired.len(),
            gates.len(),
            marker_charge_sum
        ));
        row_count += 1;
    }
    fs::write(out_path, out)?;
    let mut stdout = io::stdout();
    writeln!(stdout, "rows_processed={row_count}")?;
    writeln!(stdout, "propagate_sparse_calls={propagate_calls}")?;
    writeln!(stdout, "gate_count={}", gates.len())?;
    Ok(())
}
'''


def ensure_rust_harness(out, repo_root):
    harness = out / "rust_bridge_harness"
    src = harness / "src"
    src.mkdir(parents=True, exist_ok=True)
    cargo = harness / "Cargo.toml"
    main = src / "main.rs"
    instnct_path = str((repo_root / "instnct-core").resolve()).replace("\\", "/")
    cargo.write_text(
        "\n".join(
            [
                "[package]",
                'name = "d57_rust_sparse_bridge_harness"',
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
    main.write_text(rust_harness_source(), encoding="utf-8")
    return harness


def write_controller_tsv(path, controller):
    feature_index = {name: idx for idx, name in enumerate(FEATURE_NAMES)}
    lines = ["feature_idx\taction\tstored_threshold\tweight\tpriority\tchannel\tpolarity"]
    for gate in controller["gates"]:
        lines.append(
            "\t".join(
                [
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


def parse_rust_actions(path):
    raw = path.read_text(encoding="utf-8").splitlines()
    actions = {}
    for line in raw[1:]:
        if not line.strip():
            continue
        row_key, action, fired_count, spike_count, marker_charge = line.split("\t")
        actions[int(row_key)] = {
            "action": action,
            "fired_gate_count": int(fired_count),
            "spike_update_count": int(spike_count),
            "marker_charge_sum": int(marker_charge),
        }
    return actions


def run_rust_bridge(out, repo_root, controller, packs, split, arm, started):
    harness = ensure_rust_harness(out, repo_root)
    work = out / "rust_bridge_inputs" / split / arm
    work.mkdir(parents=True, exist_ok=True)
    controller_path = work / "controller.tsv"
    rows_path = work / "rows.tsv"
    actions_path = work / "actions.tsv"
    write_controller_tsv(controller_path, controller)
    write_rows_tsv(rows_path, packs)
    cmd = [
        "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        str(harness / "Cargo.toml"),
        "--",
        str(controller_path),
        str(rows_path),
        str(actions_path),
        controller.get("default_action", "DECIDE"),
    ]
    append_progress(out, "rust_bridge_start", started, {"split": split, "arm": arm, "rows": len(packs)})
    proc = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
    report = {
        "split": split,
        "arm": arm,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr[-4000:],
        "actions_path": str(actions_path),
        "rows_requested": len(packs),
        "actions_present": actions_path.exists(),
    }
    if proc.returncode != 0:
        write_json(work / "rust_error.json", report)
        raise RuntimeError(f"Rust bridge failed for {split}/{arm}: {proc.stderr[-500:]}")
    actions = parse_rust_actions(actions_path)
    report["actions_rows"] = len(actions)
    write_json(work / "rust_invocation.json", report)
    append_progress(out, "rust_bridge_done", started, {"split": split, "arm": arm, "actions": len(actions)})
    return actions, report


def output_from_action(pack, arm, action, trace=None):
    row = d55.output_from_action(pack, arm, action, trace)
    row["rust_network_path_invoked"] = bool(trace and trace.get("rust_network_path_invoked"))
    row["rust_propagate_sparse_called"] = bool(trace and trace.get("rust_propagate_sparse_called"))
    row["python_fallback_used"] = bool(trace and trace.get("python_fallback_used"))
    row["rust_marker_charge_sum"] = int(trace.get("marker_charge_sum", 0)) if trace else 0
    return row


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


def python_trace(trace):
    trace = copy.deepcopy(trace)
    trace["rust_network_path_invoked"] = False
    trace["rust_propagate_sparse_called"] = False
    trace["python_fallback_used"] = True
    return trace


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


def evaluate_pack_all_arms(pack, index, controllers, rust_actions, sparse_stats):
    rows = []
    py_controller = controllers["D56_PYTHON_SPARSE_REPLAY"]
    py_action, py_sparse_trace = d55.choose_sparse_action(py_controller, pack["features"], sparse_stats["D56_PYTHON_SPARSE_REPLAY"])
    rows.append(output_from_action(pack, "D56_PYTHON_SPARSE_REPLAY", py_action, python_trace(py_sparse_trace)))
    rows.append(output_from_action(pack, "PYTHON_FALLBACK_REFERENCE", py_action, python_trace(py_sparse_trace)))

    rust_action = rust_actions["CANONICAL_RUST_SPARSE_NETWORK_BRIDGE"][index]
    rows.append(
        output_from_action(
            pack,
            "CANONICAL_RUST_SPARSE_NETWORK_BRIDGE",
            rust_action["action"],
            rust_trace(rust_action),
        )
    )

    shuffle = d55.spike_shuffle_mapping()
    rows.append(
        output_from_action(
            pack,
            "RUST_SPIKE_SHUFFLE_CONTROL",
            shuffle[rust_action["action"]],
            rust_trace(rust_action),
        )
    )

    for arm in ["RUST_THRESHOLD_ABLATION", "RUST_REWIRE_ABLATION"]:
        action_record = rust_actions[arm][index]
        rows.append(output_from_action(pack, arm, action_record["action"], rust_trace(action_record)))

    rows.append(output_from_action(pack, "RUST_PATH_DISABLED_CONTROL", "DECIDE", disabled_rust_trace()))
    rows.append(output_from_action(pack, "RANDOM_POLICY_CONTROL", d51.random_policy_action(f"D57:{pack['row_id']}")))
    rows.append(output_from_action(pack, "GREEDY_DECIDE_CONTROL", "DECIDE"))
    rows.append(output_from_action(pack, "ALWAYS_COUNTER_CONTROL", "REQUEST_JOINT_COUNTER"))
    return rows


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


def record_batch(batch, outputs, sample_counts, path):
    for row in batch:
        outputs.append(row)
        key = (row["arm"], row["support_regime"])
        if sample_counts[key] < ROW_SAMPLE_PER_ARM_REGIME_SPLIT:
            append_jsonl(path, row)
            sample_counts[key] += 1


def write_partial_eval(out, split, outputs, completed, started):
    partial = summarize_outputs(outputs)
    write_json(
        out / f"partial_{split}_metrics_snapshot.json",
        {
            "split": split,
            "completed_outputs": completed,
            "elapsed_sec": time.time() - started,
            "rust_bridge_core": partial["by_arm_core"].get("CANONICAL_RUST_SPARSE_NETWORK_BRIDGE", {}),
        },
    )
    append_progress(out, "eval_progress", started, {"split": split, "completed_outputs": completed})


def evaluate_packs(packs, controllers, out_path, out, split, started, heartbeat_sec, repo_root):
    if out_path.exists():
        out_path.unlink()
    rust_actions = {}
    rust_reports = {}
    for arm in ["CANONICAL_RUST_SPARSE_NETWORK_BRIDGE", "RUST_THRESHOLD_ABLATION", "RUST_REWIRE_ABLATION"]:
        rust_actions[arm], rust_reports[arm] = run_rust_bridge(out, repo_root, controllers[arm], packs, split, arm, started)
    outputs = []
    sparse_stats = defaultdict(
        lambda: {
            "calls": 0,
            "spike_update_executed_count": 0,
            "fired_gate_count": 0,
            "total_input_charge": 0,
            "output_charge_sum": 0,
            "action_counts": Counter(),
            "fired_gate_by_feature": Counter(),
            "fired_gate_by_action": Counter(),
        }
    )
    sample_counts = Counter()
    completed = 0
    total = len(packs) * len(ARMS)
    last = 0.0
    for idx, pack in enumerate(packs):
        batch = evaluate_pack_all_arms(pack, idx, controllers, rust_actions, sparse_stats)
        record_batch(batch, outputs, sample_counts, out_path)
        completed += len(batch)
        now = time.time()
        if now - last >= heartbeat_sec or completed >= total:
            last = now
            write_partial_eval(out, split, outputs, completed, started)
    return outputs, d55.normalize_sparse_stats(sparse_stats), rust_reports


def regime_accuracy(metrics, arm, regime):
    return metrics["by_arm_and_regime"][arm][regime]["accuracy"]


def min_seed_metric(metrics, arm, metric):
    rows = [seed_row[metric] for seed_row in metrics["by_seed_core"].get(arm, {}).values()]
    return min(rows) if rows else 0.0


def min_seed_regime_accuracy(metrics, arm, regime):
    values = []
    for seed_rows in metrics["by_seed_regime"].get(arm, {}).values():
        if regime in seed_rows:
            values.append(seed_rows[regime]["accuracy"])
    return min(values) if values else 0.0


def action_parity(outputs):
    grouped = defaultdict(dict)
    for row in outputs:
        if row["arm"] in {"D56_PYTHON_SPARSE_REPLAY", "CANONICAL_RUST_SPARSE_NETWORK_BRIDGE"}:
            key = (row["split"], row["row_id"], row["support_regime"])
            grouped[key][row["arm"]] = row["selected_action"]
    compared = 0
    matched = 0
    mismatches = []
    for key, value in grouped.items():
        if "D56_PYTHON_SPARSE_REPLAY" in value and "CANONICAL_RUST_SPARSE_NETWORK_BRIDGE" in value:
            compared += 1
            if value["D56_PYTHON_SPARSE_REPLAY"] == value["CANONICAL_RUST_SPARSE_NETWORK_BRIDGE"]:
                matched += 1
            elif len(mismatches) < 25:
                mismatches.append({"key": key, "python": value["D56_PYTHON_SPARSE_REPLAY"], "rust": value["CANONICAL_RUST_SPARSE_NETWORK_BRIDGE"]})
    return {
        "compared": compared,
        "matched": matched,
        "parity_rate": matched / compared if compared else 0.0,
        "sample_mismatches": mismatches,
    }


def make_decision(metrics, failed_jobs, rust_usage, parity):
    if failed_jobs:
        return {
            "decision": "canonical_rust_sparse_path_bridge_not_confirmed",
            "verdict": "D57_FAILED_JOBS_PRESENT",
            "next": "D57_REPAIR",
            "boundary": BOUNDARY,
        }
    arm = "CANONICAL_RUST_SPARSE_NETWORK_BRIDGE"
    core = metrics["by_arm_core"]
    regimes = metrics["by_arm_and_regime"]
    row = core[arm]
    corr = regime_accuracy(metrics, arm, "CORRELATED_ECHO_SUPPORT")
    adv = regime_accuracy(metrics, arm, "ADVERSARIAL_DISTRACTOR_SUPPORT")
    external = regime_accuracy(metrics, arm, "EXTERNAL_TEST_REQUIRED_SUPPORT")
    indist = regimes[arm]["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]
    rust_rows = rust_usage.get(arm, {}).get("rust_rows", 0)
    fallback_rows = rust_usage.get(arm, {}).get("python_fallback_rows", 0)
    rust_invoked = rust_rows > 0 and fallback_rows == 0
    gates_pass = (
        rust_invoked
        and row["exact_joint_accuracy"] >= 0.995
        and corr >= 0.995
        and adv >= 0.995
        and external >= 0.99
        and indist["abstain_rate"] >= 0.99
        and indist["false_confidence_rate"] <= 0.01
        and row["average_total_support_used"] <= core["ALWAYS_COUNTER_CONTROL"]["average_total_support_used"]
        and row["accuracy"] > core["RANDOM_POLICY_CONTROL"]["accuracy"]
        and row["accuracy"] > core["GREEDY_DECIDE_CONTROL"]["accuracy"]
        and row["accuracy"] > core["RUST_SPIKE_SHUFFLE_CONTROL"]["accuracy"]
        and row["accuracy"] > core["RUST_THRESHOLD_ABLATION"]["accuracy"]
        and row["accuracy"] > core["RUST_REWIRE_ABLATION"]["accuracy"]
        and min_seed_metric(metrics, arm, "exact_joint_accuracy") >= 0.99
        and min_seed_regime_accuracy(metrics, arm, "CORRELATED_ECHO_SUPPORT") >= 0.99
        and min_seed_regime_accuracy(metrics, arm, "ADVERSARIAL_DISTRACTOR_SUPPORT") >= 0.99
    )
    if gates_pass and parity["parity_rate"] >= 0.995:
        return {
            "decision": "canonical_rust_sparse_path_bridge_positive",
            "verdict": "D57_CANONICAL_RUST_SPARSE_PATH_BRIDGE_POSITIVE",
            "next": "D58_CANONICAL_RUST_SPARSE_CONTROLLER_SCALE_CONFIRM",
            "best_arm": arm,
            "boundary": BOUNDARY,
        }
    if rust_invoked:
        return {
            "decision": "rust_path_invoked_but_behavior_mismatch",
            "verdict": "D57_RUST_PYTHON_SEMANTICS_MISMATCH",
            "next": "D57B_RUST_PYTHON_SEMANTICS_ALIGNMENT",
            "best_arm": arm,
            "boundary": BOUNDARY,
        }
    return {
        "decision": "rust_path_not_actually_invoked",
        "verdict": "D57_BRIDGE_INSTRUMENTATION_FAILURE",
        "next": "D57R_BRIDGE_INSTRUMENTATION_REPAIR",
        "best_arm": arm,
        "boundary": BOUNDARY,
    }


def make_reports(out, aggregate, decision, d56_manifest, canonical_report, controllers):
    test = aggregate["test_metrics"]
    core = test["by_arm_core"]
    regimes = test["by_arm_and_regime"]
    parity = aggregate["python_vs_rust_action_parity"]
    rust_usage = test["rust_usage"]
    reports = {
        "d56_upstream_manifest.json": d56_manifest,
        "rust_bridge_build_report.json": aggregate["rust_bridge_build_report"],
        "rust_bridge_invocation_report.json": aggregate["rust_bridge_invocation_report"],
        "rust_path_usage_report.json": {
            "canonical_rust_network_path_invoked": rust_usage.get("CANONICAL_RUST_SPARSE_NETWORK_BRIDGE", {}).get("rust_rows", 0) > 0,
            "rust_propagate_sparse_called": rust_usage.get("CANONICAL_RUST_SPARSE_NETWORK_BRIDGE", {}).get("rust_rows", 0) > 0,
            "canonical_arm_python_fallback_rows": rust_usage.get("CANONICAL_RUST_SPARSE_NETWORK_BRIDGE", {}).get("python_fallback_rows", 0),
            "by_arm": rust_usage,
        },
        "python_fallback_audit.json": {
            "python_fallback_reference_present": True,
            "canonical_arm_python_fallback_used": rust_usage.get("CANONICAL_RUST_SPARSE_NETWORK_BRIDGE", {}).get("python_fallback_rows", 0) > 0,
            "python_fallback_arms": ["D56_PYTHON_SPARSE_REPLAY", "PYTHON_FALLBACK_REFERENCE"],
        },
        "python_vs_rust_action_parity_report.json": parity,
        "canonical_sparse_path_report.json": canonical_report,
        "action_readout_report.json": {
            arm: {
                "action_distribution": test["action_distribution"].get(arm, {}),
                "rust_network_path_invoked": rust_usage.get(arm, {}).get("rust_rows", 0) > 0,
                "python_fallback_rows": rust_usage.get(arm, {}).get("python_fallback_rows", 0),
            }
            for arm in ARMS
        },
        "firing_dynamics_report.json": {
            "python_sparse_stats": aggregate["python_sparse_stats"],
            "rust_marker_charge_bridge": {
                arm: {
                    "rust_rows": rust_usage.get(arm, {}).get("rust_rows", 0),
                    "rows": rust_usage.get(arm, {}).get("rows", 0),
                }
                for arm in ARMS
            },
        },
        "support_cost_frontier_report.json": {
            arm: {
                "exact_joint_accuracy": core[arm]["exact_joint_accuracy"],
                "support": core[arm]["average_total_support_used"],
                "counter_support": core[arm]["average_counter_support_used"],
                "external_test": core[arm]["average_external_test_used"],
                "cost_adjusted": d53.cost_adjusted(core[arm]),
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
            for arm in [
                "RUST_SPIKE_SHUFFLE_CONTROL",
                "RUST_THRESHOLD_ABLATION",
                "RUST_REWIRE_ABLATION",
                "RUST_PATH_DISABLED_CONTROL",
                "RANDOM_POLICY_CONTROL",
                "GREEDY_DECIDE_CONTROL",
            ]
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
            }
            for arm in ARMS
        },
        "min_seed_gate_report.json": {
            arm: {
                "min_seed_exact_joint": min_seed_metric(test, arm, "exact_joint_accuracy"),
                "min_seed_correlated_echo": min_seed_regime_accuracy(test, arm, "CORRELATED_ECHO_SUPPORT"),
                "min_seed_adversarial_distractor": min_seed_regime_accuracy(test, arm, "ADVERSARIAL_DISTRACTOR_SUPPORT"),
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
    lines = [
        "# D57 Canonical Rust Sparse Path Bridge Result",
        "",
        "Decision:",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"verdict = {decision.get('verdict')}",
        f"next = {decision.get('next')}",
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
    rust_usage = test["rust_usage"]
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
            "Bridge audit:",
            "",
            "```text",
            f"rust_network_path_invoked = {aggregate['canonical_sparse_path_report']['canonical_rust_network_path_invoked']}",
            f"python_vs_rust_action_parity = {aggregate['python_vs_rust_action_parity']['parity_rate']:.5f}",
            f"canonical_arm_python_fallback_rows = {rust_usage.get('CANONICAL_RUST_SPARSE_NETWORK_BRIDGE', {}).get('python_fallback_rows', 0)}",
            "```",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def make_summary(aggregate, decision):
    test = aggregate["test_metrics"]
    arm = "CANONICAL_RUST_SPARSE_NETWORK_BRIDGE"
    core = test["by_arm_core"][arm]
    regimes = test["by_arm_and_regime"][arm]
    return {
        "task": "D57_CANONICAL_RUST_SPARSE_PATH_BRIDGE",
        "decision": decision["decision"],
        "verdict": decision.get("verdict"),
        "next": decision.get("next"),
        "scale_mode": aggregate["scale_mode"],
        "best_arm": arm,
        "rust_network_path_invoked": aggregate["canonical_sparse_path_report"]["canonical_rust_network_path_invoked"],
        "python_fallback_used_for_canonical_arm": aggregate["test_metrics"]["rust_usage"][arm].get("python_fallback_rows", 0) > 0,
        "python_vs_rust_action_parity": aggregate["python_vs_rust_action_parity"]["parity_rate"],
        "key_metrics": {
            "exact_joint_accuracy": core["exact_joint_accuracy"],
            "correlated_echo": regimes["CORRELATED_ECHO_SUPPORT"]["accuracy"],
            "adversarial_distractor": regimes["ADVERSARIAL_DISTRACTOR_SUPPORT"]["accuracy"],
            "external_test_required": regimes["EXTERNAL_TEST_REQUIRED_SUPPORT"]["accuracy"],
            "indistinguishable_abstain": regimes["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["abstain_rate"],
            "false_confidence": regimes["INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT"]["false_confidence_rate"],
            "support": core["average_total_support_used"],
        },
        "boundary": BOUNDARY,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="11101,11102,11103,11104,11105")
    parser.add_argument("--train-rows-per-seed", type=int, default=800)
    parser.add_argument("--test-rows-per-seed", type=int, default=800)
    parser.add_argument("--ood-rows-per-seed", type=int, default=800)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="50-75")
    parser.add_argument("--heartbeat-sec", type=float, default=20.0)
    parser.add_argument("--scale-mode", default="smoke")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    seeds = parse_seeds(args.seeds)
    repo_root = Path(__file__).resolve().parents[2]
    failed_jobs = []
    write_json(out / "queue.json", {"task": "D57_CANONICAL_RUST_SPARSE_PATH_BRIDGE", "args": vars(args), "seeds": seeds, "boundary": BOUNDARY})
    append_progress(out, "started", started, {"seeds": seeds, "scale_mode": args.scale_mode})
    write_json(out / "compute_probe.json", {"cpu_count": os.cpu_count(), "workers": d51.worker_count_from_arg(args.workers), "cpu_target": args.cpu_target})

    d56_manifest = make_d56_upstream_manifest(repo_root)
    write_json(out / "d56_upstream_manifest.json", d56_manifest)
    base_controller = load_d56_best_controller(repo_root)
    if base_controller is None:
        failed_jobs.append("missing_d56_best_sparse_controller")
        base_controller = d56.load_d55_best_controller(repo_root, {})

    controllers = {
        "D56_PYTHON_SPARSE_REPLAY": copy.deepcopy(base_controller),
        "CANONICAL_RUST_SPARSE_NETWORK_BRIDGE": copy.deepcopy(base_controller),
        "RUST_THRESHOLD_ABLATION": d55.make_threshold_ablation(base_controller),
        "RUST_REWIRE_ABLATION": d55.make_rewire_ablation(base_controller),
    }

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
        },
    )

    train_rows = d51.make_rows_with_progress(seeds, args.train_rows_per_seed, "train", bundle, out, started, args.heartbeat_sec)
    test_rows = d51.make_rows_with_progress(seeds, args.test_rows_per_seed, "test", bundle, out, started, args.heartbeat_sec)
    ood_rows = d51.make_rows_with_progress(seeds, args.ood_rows_per_seed, "ood", bundle, out, started, args.heartbeat_sec)
    append_progress(out, "rows_generated", started, {"train": len(train_rows), "test": len(test_rows), "ood": len(ood_rows)})
    train_packs = d51.build_packs(train_rows, bundle, out, started, args.heartbeat_sec, args.workers, "train")
    test_packs = d51.build_packs(test_rows, bundle, out, started, args.heartbeat_sec, args.workers, "test")
    ood_packs = d51.build_packs(ood_rows, bundle, out, started, args.heartbeat_sec, args.workers, "ood")
    append_progress(out, "packs_built", started, {"train": len(train_packs), "test": len(test_packs), "ood": len(ood_packs)})

    rust_build_report = {"harness_path": str(ensure_rust_harness(out, repo_root)), "path_dependency": str(repo_root / "instnct-core")}
    test_outputs, test_sparse_stats, test_rust_reports = evaluate_packs(test_packs, controllers, out / "row_outputs_test.jsonl", out, "test", started, args.heartbeat_sec, repo_root)
    ood_outputs, ood_sparse_stats, ood_rust_reports = evaluate_packs(ood_packs, controllers, out / "row_outputs_ood.jsonl", out, "ood", started, args.heartbeat_sec, repo_root)
    all_outputs = test_outputs + ood_outputs
    test_metrics = summarize_outputs(test_outputs)
    ood_metrics = summarize_outputs(ood_outputs)
    parity = action_parity(all_outputs)
    rust_invoked = test_metrics["rust_usage"].get("CANONICAL_RUST_SPARSE_NETWORK_BRIDGE", {}).get("rust_rows", 0) > 0
    canonical_report = make_canonical_sparse_path_report(repo_root, rust_invoked)

    aggregate = {
        "task": "D57_CANONICAL_RUST_SPARSE_PATH_BRIDGE",
        "scale_mode": args.scale_mode,
        "failed_jobs": failed_jobs,
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
        "python_sparse_stats": {"test": test_sparse_stats, "ood": ood_sparse_stats},
        "rust_bridge_build_report": rust_build_report,
        "rust_bridge_invocation_report": {"test": test_rust_reports, "ood": ood_rust_reports},
        "python_vs_rust_action_parity": parity,
        "canonical_sparse_path_report": canonical_report,
        "boundary": BOUNDARY,
    }
    decision = make_decision(test_metrics, failed_jobs, test_metrics["rust_usage"], parity)
    aggregate["decision"] = decision
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    reports = make_reports(out, aggregate, decision, d56_manifest, canonical_report, controllers)
    write_json(out / "summary.json", make_summary(aggregate, decision))
    write_report(out, aggregate, decision)
    write_json(
        out / "trained_policy_manifest.json",
        {
            "controllers": controllers,
            "rust_bridge_harness": str(out / "rust_bridge_harness"),
            "decision": decision,
        },
    )
    append_progress(out, "completed", started, {"decision": decision["decision"], "reports": sorted(reports.keys())})
    print(json.dumps(make_summary(aggregate, decision), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
