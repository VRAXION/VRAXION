#!/usr/bin/env python3
"""D65 set-invariant IPF aggregation prototype.

D64U repaired the claim boundary: candidate identity and temporal support order
are not confirmed as essential. D65 tests whether a Rust sparse path can consume
unordered support/evidence score sets and emit useful IPF aggregate features for
the ECF action controller.
"""

import argparse
import copy
import json
import math
import os
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import run_d49_joint_cell_operator_discovery_with_robust_support as d49
import run_d49b_joint_binding_repair as d49b
import run_d51_mutable_ecf_controller_prototype as d51
import run_d55_sparse_firing_ecf_controller_prototype as d55
import run_d59_rust_sparse_ecf_controller_with_mutation as d59
import run_d62_policy_ensemble_ecf_controller_with_learned_gate as d62
import run_d64s_score_vector_structure_repair as d64s

TASK = "D65_SET_INVARIANT_IPF_AGGREGATION_PROTOTYPE"
BOUNDARY = (
    "D65 only tests set-invariant Rust sparse IPF aggregation for controlled "
    "symbolic joint formula discovery. It does not prove full VRAXION brain, "
    "raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome "
    "success, architecture superiority, or production readiness."
)

PRIMARY_SPACE = d62.PRIMARY_SPACE
SUPPORT_COUNT = d62.SUPPORT_COUNT
REGIMES = d62.REGIMES
CORE_REGIMES = d62.CORE_REGIMES
ACTIONS = d62.ACTIONS
FEATURE_NAMES = d51.FEATURE_NAMES
POLICY_MODULES = d62.POLICY_MODULE_ARMS
ROW_SAMPLE_PER_ARM_REGIME_SPLIT = 18

ARMS = [
    "SYMBOLIC_SET_AGGREGATION_REFERENCE",
    "RUST_SPARSE_SUM_AGGREGATION",
    "RUST_SPARSE_MEAN_AGGREGATION",
    "RUST_SPARSE_NORMALIZED_SHAPE_AGGREGATION",
    "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "RUST_SPARSE_SUPPORT_COHERENCE_SET_AGGREGATION",
    "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
    "HYBRID_SYMBOLIC_RUST_SET_AGGREGATION",
    "SUPPORT_ORDER_SHUFFLE_NOOP_CONTROL",
    "CANDIDATE_ID_SHUFFLE_CONTROL",
    "SUPPORT_CONTENT_CORRUPTION_CONTROL",
    "RANDOM_SCORE_AGGREGATION_CONTROL",
    "AGGREGATION_ABLATION_CONTROL",
    "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY",
]

RUST_AGGREGATION_ARMS = [
    "RUST_SPARSE_SUM_AGGREGATION",
    "RUST_SPARSE_MEAN_AGGREGATION",
    "RUST_SPARSE_NORMALIZED_SHAPE_AGGREGATION",
    "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
    "RUST_SPARSE_SUPPORT_COHERENCE_SET_AGGREGATION",
    "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION",
    "HYBRID_SYMBOLIC_RUST_SET_AGGREGATION",
    "SUPPORT_ORDER_SHUFFLE_NOOP_CONTROL",
    "CANDIDATE_ID_SHUFFLE_CONTROL",
    "SUPPORT_CONTENT_CORRUPTION_CONTROL",
    "RANDOM_SCORE_AGGREGATION_CONTROL",
    "AGGREGATION_ABLATION_CONTROL",
]

CONTROL_ARMS = [
    "SUPPORT_ORDER_SHUFFLE_NOOP_CONTROL",
    "CANDIDATE_ID_SHUFFLE_CONTROL",
    "SUPPORT_CONTENT_CORRUPTION_CONTROL",
    "RANDOM_SCORE_AGGREGATION_CONTROL",
    "AGGREGATION_ABLATION_CONTROL",
]

REFERENCE_ONLY_ARMS = ["TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"]
FAIR_ARMS = [arm for arm in ARMS if arm not in REFERENCE_ONLY_ARMS]

RUST_MODE_BY_ARM = {
    "RUST_SPARSE_SUM_AGGREGATION": "sum",
    "RUST_SPARSE_MEAN_AGGREGATION": "mean",
    "RUST_SPARSE_NORMALIZED_SHAPE_AGGREGATION": "normalized",
    "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION": "set_invariant",
    "RUST_SPARSE_SUPPORT_COHERENCE_SET_AGGREGATION": "coherence",
    "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION": "counter_delta",
    "HYBRID_SYMBOLIC_RUST_SET_AGGREGATION": "set_invariant",
    "SUPPORT_ORDER_SHUFFLE_NOOP_CONTROL": "set_invariant",
    "CANDIDATE_ID_SHUFFLE_CONTROL": "set_invariant",
    "SUPPORT_CONTENT_CORRUPTION_CONTROL": "set_invariant",
    "RANDOM_SCORE_AGGREGATION_CONTROL": "set_invariant",
    "AGGREGATION_ABLATION_CONTROL": "ablation",
}


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


def stable_seed(text):
    return d51.stable_seed(text)


def stable_rng(seed, tag):
    return random.Random(int(seed) + stable_seed(tag))


def make_d64u_upstream_manifest(repo_root):
    root = repo_root / "target/pilot_wave/d64u_redefine_ipf_diagnostic_claim_and_set_aggregation_plan/smoke"
    manifest = {
        "upstream": "D64U_REDEFINE_IPF_DIAGNOSTIC_CLAIM_AND_SET_AGGREGATION_PLAN",
        "expected_decision": "ipf_diagnostic_claim_redefined_set_aggregation_ready",
        "expected_next": TASK,
        "artifact_root": str(root),
        "decision_present": (root / "decision.json").exists(),
        "d65_plan_present": (root / "d65_set_aggregation_plan.json").exists(),
    }
    for name in [
        "decision.json",
        "summary.json",
        "claim_boundary_report.json",
        "supported_claims_report.json",
        "rejected_or_unconfirmed_claims_report.json",
        "d65_set_aggregation_plan.json",
    ]:
        value = load_json(root / name)
        if value is not None:
            manifest[name.replace(".json", "")] = value
    return manifest


def load_policy_modules(repo_root):
    controllers, learned_gate = d64s.load_d62_policy_modules(repo_root)
    missing = [name for name in POLICY_MODULES if name not in controllers]
    for name in missing:
        if name == "COUNTERFACTUAL_POLICY":
            controllers[name] = d62.make_always_action_controller(name, "REQUEST_COUNTER_TOP1_TOP2")
        elif name == "EXTERNAL_TEST_POLICY":
            controllers[name] = d62.make_always_action_controller(name, "REQUEST_EXTERNAL_TEST")
        elif name == "ABSTAIN_POLICY":
            controllers[name] = d62.make_always_action_controller(name, "ABSTAIN")
        elif name == "ADVERSARIAL_REPAIR_POLICY":
            controllers[name] = d62.make_always_action_controller(name, "REQUEST_JOINT_COUNTER")
        else:
            controllers[name] = d62.make_always_action_controller(name, "DECIDE")
    return {name: controllers[name] for name in POLICY_MODULES}, learned_gate


def candidate_ids(bundle):
    return sorted(bundle["candidates"])


def score_to_bin(value):
    # Candidate scores are negative circular distances, usually in [-4, 0].
    return max(0, min(8, int(round(float(value) + 4.0))))


def support_vectors_for_row(row, bundle):
    return d49b.cached_base_vectors(row, bundle, SUPPORT_COUNT)


def flat_bins_from_vectors(row, bundle, arm):
    ids = candidate_ids(bundle)
    vectors = [dict(vector) for vector in support_vectors_for_row(row, bundle)]
    rng = stable_rng(row["seed"], f"D65:{arm}:{row['row_id']}")
    if arm == "SUPPORT_ORDER_SHUFFLE_NOOP_CONTROL":
        rng.shuffle(vectors)
    if arm == "CANDIDATE_ID_SHUFFLE_CONTROL":
        perm = list(ids)
        rng.shuffle(perm)
        mapping = dict(zip(ids, perm))
        vectors = [{cid: vector[mapping[cid]] for cid in ids} for vector in vectors]
    elif arm == "SUPPORT_CONTENT_CORRUPTION_CONTROL":
        corrupted = []
        for support_idx, vector in enumerate(vectors):
            out = {}
            for idx, cid in enumerate(ids):
                source = ids[(idx + support_idx + 17) % len(ids)]
                out[cid] = -float((score_to_bin(vector[source]) + 2 + (idx % 3)) % 5)
            corrupted.append(out)
        vectors = corrupted
    elif arm == "RANDOM_SCORE_AGGREGATION_CONTROL":
        vectors = [
            {cid: -float(rng.randrange(5)) for cid in ids}
            for _ in range(SUPPORT_COUNT)
        ]
    elif arm == "AGGREGATION_ABLATION_CONTROL":
        vectors = [{cid: -2.0 for cid in ids} for _ in range(SUPPORT_COUNT)]
    flat = []
    for vector in vectors[:SUPPORT_COUNT]:
        for cid in ids:
            flat.append(score_to_bin(vector[cid]))
    return flat, vectors


def symbolic_feature_map(row, bundle, base_vectors=None):
    if base_vectors is None:
        base_vectors = support_vectors_for_row(row, bundle)
    scores = d49b.aggregate_sum(base_vectors)
    pred = d49b.predict(scores, bundle)
    features, fmap = d51.build_features(row, bundle, base_vectors, scores, pred)
    return features, dict(fmap), scores, pred


def features_from_map(fmap):
    return [float(fmap.get(name, 0.0)) for name in FEATURE_NAMES]


def rust_aggregation_harness_source():
    return r'''
use instnct_core::{Network, PropagationConfig};
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::process;

#[derive(Clone)]
struct Row {
    row_key: String,
    arm: String,
    mode: String,
    support_count: usize,
    candidate_count: usize,
    values: Vec<i32>,
}

fn read_rows(path: &str) -> Vec<Row> {
    let raw = fs::read_to_string(path).unwrap_or_else(|err| {
        eprintln!("failed to read aggregation rows {path}: {err}");
        process::exit(2);
    });
    let mut rows = Vec::new();
    for (line_idx, line) in raw.lines().enumerate() {
        if line_idx == 0 || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() != 6 {
            eprintln!("bad row line {line_idx}: expected 6 columns, got {}", cols.len());
            process::exit(2);
        }
        let values: Vec<i32> = cols[5]
            .split(',')
            .filter(|item| !item.is_empty())
            .map(|item| item.parse::<i32>().unwrap_or(0))
            .collect();
        rows.push(Row {
            row_key: cols[0].to_string(),
            arm: cols[1].to_string(),
            mode: cols[2].to_string(),
            support_count: cols[3].parse().unwrap_or(0),
            candidate_count: cols[4].parse().unwrap_or(0),
            values,
        });
    }
    rows
}

fn build_network(candidate_count: usize) -> Network {
    let mut net = Network::new(candidate_count * 2);
    net.set_all_threshold(0);
    net.set_all_channel(1);
    for idx in 0..candidate_count {
        net.graph_mut().add_edge(idx as u16, (candidate_count + idx) as u16);
    }
    net
}

fn entropy(probs: &[f64]) -> f64 {
    let mut total = 0.0;
    for &p in probs {
        total -= p * (p + 1e-12).ln();
    }
    total
}

fn softmax(values: &[f64]) -> Vec<f64> {
    let top = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = values.iter().map(|v| (v - top).exp()).collect();
    let total: f64 = weights.iter().sum::<f64>().max(1e-9);
    weights.iter().map(|v| v / total).collect()
}

fn signature(values: &[i32]) -> String {
    values.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",")
}

fn aggregate(row: &Row, config: &PropagationConfig) -> (Vec<f64>, BTreeMap<String, f64>, usize, usize) {
    let mut net = build_network(row.candidate_count);
    let mut totals = vec![0.0f64; row.candidate_count];
    let mut signatures: HashMap<String, usize> = HashMap::new();
    let mut propagate_calls = 0usize;
    let mut marker_charge_sum = 0usize;
    for support_idx in 0..row.support_count {
        let start = support_idx * row.candidate_count;
        let end = start + row.candidate_count;
        let slice = &row.values[start..end];
        *signatures.entry(signature(slice)).or_insert(0) += 1;
        let mut indices = Vec::<u16>::new();
        let mut values = Vec::<i8>::new();
        for (idx, raw) in slice.iter().enumerate() {
            if *raw > 0 {
                indices.push(idx as u16);
                values.push((*raw).min(8) as i8);
            }
        }
        net.reset();
        net.propagate_sparse(&indices, &values, config).unwrap_or_else(|err| {
            eprintln!("propagate_sparse failed on {}: {err}", row.row_key);
            process::exit(3);
        });
        propagate_calls += 1;
        let spike = net.spike_data();
        for idx in 0..row.candidate_count {
            let charge = spike[row.candidate_count + idx].charge as f64;
            marker_charge_sum += charge as usize;
            totals[idx] += if charge > 0.0 { charge } else { slice[idx] as f64 };
        }
    }
    let support_count = row.support_count.max(1) as f64;
    if row.mode == "mean" {
        for value in totals.iter_mut() {
            *value /= support_count;
        }
    } else if row.mode == "normalized" {
        let mean = totals.iter().sum::<f64>() / (row.candidate_count.max(1) as f64);
        let max_abs = totals.iter().map(|v| (v - mean).abs()).fold(0.0, f64::max).max(1.0);
        for value in totals.iter_mut() {
            *value = (*value - mean) / max_abs;
        }
    } else if row.mode == "ablation" {
        for value in totals.iter_mut() {
            *value = 0.0;
        }
    }
    let mut sorted = totals.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let top1 = *sorted.get(0).unwrap_or(&0.0);
    let top2 = *sorted.get(1).unwrap_or(&0.0);
    let probs = softmax(&totals);
    let confidence = probs.iter().cloned().fold(0.0, f64::max);
    let entropy_norm = entropy(&probs) / (row.candidate_count.max(2) as f64).ln();
    let margin = top1 - top2;
    let duplicate_clusters = signatures.values().filter(|&&count| count > 1).count();
    let dominant = signatures.values().cloned().max().unwrap_or(0) as f64 / support_count;
    let cluster_count = signatures.len() as f64;
    let mut features = BTreeMap::<String, f64>::new();
    features.insert("scalar_confidence".to_string(), confidence.max(0.0).min(1.0));
    features.insert("joint_confidence".to_string(), confidence.max(0.0).min(1.0));
    features.insert("cell_confidence".to_string(), confidence.max(0.0).min(1.0));
    features.insert("operator_confidence".to_string(), confidence.max(0.0).min(1.0));
    features.insert("inverse_margin".to_string(), (1.0 / (1.0 + margin.max(0.0))).max(0.0).min(1.0));
    features.insert("entropy_norm".to_string(), entropy_norm.max(0.0).min(1.0));
    features.insert("collision_norm".to_string(), ((duplicate_clusters as f64) / support_count).max(0.0).min(1.0));
    features.insert("dominant_cluster_fraction".to_string(), dominant.max(0.0).min(1.0));
    features.insert("support_cluster_count_norm".to_string(), (cluster_count / support_count).max(0.0).min(1.0));
    features.insert("top1_factorised_disagreement".to_string(), 0.0);
    if row.mode == "coherence" {
        let pressure = dominant.max(features["collision_norm"]);
        features.insert("coherence_pressure".to_string(), pressure.max(0.0).min(1.0));
    } else if row.mode == "counter_delta" {
        let pressure = features["inverse_margin"].max(features["entropy_norm"]);
        features.insert("counter_delta_pressure".to_string(), pressure.max(0.0).min(1.0));
    }
    (totals, features, propagate_calls, marker_charge_sum)
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: d65_rust_set_aggregation <rows.tsv> <out.tsv>");
        process::exit(2);
    }
    let rows = read_rows(&args[1]);
    let config = PropagationConfig {
        ticks_per_token: 3,
        input_duration_ticks: 1,
        decay_interval_ticks: 0,
        use_refractory: false,
    };
    let mut out = String::from("row_key\tarm\tmode\tscalar_confidence\tinverse_margin\tentropy_norm\tcollision_norm\tdominant_cluster_fraction\tsupport_cluster_count_norm\tcell_confidence\toperator_confidence\tjoint_confidence\ttop1_factorised_disagreement\tcoherence_pressure\tcounter_delta_pressure\trust_propagate_calls\tmarker_charge_sum\n");
    let mut total_calls = 0usize;
    for row in rows.iter() {
        let (_totals, features, calls, marker_charge_sum) = aggregate(row, &config);
        total_calls += calls;
        out.push_str(&format!(
            "{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{}\t{}\n",
            row.row_key,
            row.arm,
            row.mode,
            features.get("scalar_confidence").unwrap_or(&0.0),
            features.get("inverse_margin").unwrap_or(&0.0),
            features.get("entropy_norm").unwrap_or(&0.0),
            features.get("collision_norm").unwrap_or(&0.0),
            features.get("dominant_cluster_fraction").unwrap_or(&0.0),
            features.get("support_cluster_count_norm").unwrap_or(&0.0),
            features.get("cell_confidence").unwrap_or(&0.0),
            features.get("operator_confidence").unwrap_or(&0.0),
            features.get("joint_confidence").unwrap_or(&0.0),
            features.get("top1_factorised_disagreement").unwrap_or(&0.0),
            features.get("coherence_pressure").unwrap_or(&0.0),
            features.get("counter_delta_pressure").unwrap_or(&0.0),
            calls,
            marker_charge_sum
        ));
    }
    fs::write(&args[2], out)?;
    let mut stdout = io::stdout();
    writeln!(stdout, "rows_processed={}", rows.len())?;
    writeln!(stdout, "propagate_sparse_calls={}", total_calls)?;
    Ok(())
}
'''


def ensure_rust_aggregation_harness(out, repo_root):
    harness = out / "rust_set_aggregation_harness"
    src = harness / "src"
    src.mkdir(parents=True, exist_ok=True)
    instnct_path = str((repo_root / "instnct-core").resolve()).replace("\\", "/")
    (harness / "Cargo.toml").write_text(
        "\n".join(
            [
                "[package]",
                'name = "d65_rust_set_aggregation_harness"',
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
    (src / "main.rs").write_text(rust_aggregation_harness_source(), encoding="utf-8")
    return harness


def write_rust_aggregation_rows(path, rows, bundle, split, out, started, heartbeat_sec):
    ids = candidate_ids(bundle)
    lines = ["row_key\tarm\tmode\tsupport_count\tcandidate_count\tflat_score_bins"]
    total = len(rows) * len(RUST_AGGREGATION_ARMS)
    completed = 0
    last = time.time()
    for row_idx, row in enumerate(rows):
        for arm in RUST_AGGREGATION_ARMS:
            flat, _vectors = flat_bins_from_vectors(row, bundle, arm)
            mode = RUST_MODE_BY_ARM[arm]
            row_key = f"{row_idx}:{arm}"
            lines.append(
                "\t".join(
                    [
                        row_key,
                        arm,
                        mode,
                        str(SUPPORT_COUNT),
                        str(len(ids)),
                        ",".join(str(value) for value in flat),
                    ]
                )
            )
            completed += 1
        now = time.time()
        if now - last >= heartbeat_sec:
            last = now
            write_json(out / f"partial_{split}_rust_aggregation_input.json", {"split": split, "completed_rows": completed, "total_rows": total})
            append_progress(out, "rust_aggregation_input_progress", started, {"split": split, "completed_rows": completed, "total_rows": total})
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    append_progress(out, "rust_aggregation_input_complete", started, {"split": split, "rows": total})


def parse_rust_aggregation_output(path):
    parsed = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    for line in lines[1:]:
        if not line.strip():
            continue
        cols = line.split("\t")
        row = dict(zip(header, cols))
        row_key = row["row_key"]
        values = {}
        for key, value in row.items():
            if key in {"row_key", "arm", "mode"}:
                continue
            if key in {"rust_propagate_calls", "marker_charge_sum"}:
                values[key] = int(float(value))
            else:
                values[key] = float(value)
        parsed[row_key] = values
    return parsed


def run_rust_aggregation_bridge(out, repo_root, rows, bundle, split, started, heartbeat_sec):
    harness = ensure_rust_aggregation_harness(out, repo_root)
    work = out / "rust_aggregation_inputs" / split
    work.mkdir(parents=True, exist_ok=True)
    rows_path = work / "support_sets.tsv"
    result_path = work / "aggregate_features.tsv"
    write_rust_aggregation_rows(rows_path, rows, bundle, split, out, started, heartbeat_sec)
    cmd = [
        "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        str(harness / "Cargo.toml"),
        "--",
        str(rows_path),
        str(result_path),
    ]
    append_progress(out, "rust_aggregation_bridge_start", started, {"split": split, "rows": len(rows) * len(RUST_AGGREGATION_ARMS)})
    proc = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
    report = {
        "split": split,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr[-4000:],
        "rows_requested": len(rows) * len(RUST_AGGREGATION_ARMS),
        "input_path": str(rows_path),
        "result_path": str(result_path),
        "result_present": result_path.exists(),
        "rust_sparse_path": "instnct-core Network::propagate_sparse per support vector",
    }
    if proc.returncode != 0:
        write_json(work / "rust_aggregation_error.json", report)
        raise RuntimeError(f"D65 Rust aggregation bridge failed for {split}: {proc.stderr[-500:]}")
    parsed = parse_rust_aggregation_output(result_path)
    report["rows_returned"] = len(parsed)
    report["propagate_sparse_calls_reported"] = proc.stdout
    write_json(work / "rust_aggregation_invocation.json", report)
    append_progress(out, "rust_aggregation_bridge_done", started, {"split": split, "rows_returned": len(parsed)})
    return parsed, report


def feature_map_from_rust(row, rust_values, symbolic_map, arm):
    fmap = {
        "bias": 1.0,
        "scalar_confidence": rust_values["scalar_confidence"],
        "inverse_margin": rust_values["inverse_margin"],
        "entropy_norm": rust_values["entropy_norm"],
        "collision_norm": rust_values["collision_norm"],
        "dominant_cluster_fraction": rust_values["dominant_cluster_fraction"],
        "support_cluster_count_norm": rust_values["support_cluster_count_norm"],
        "top1_factorised_disagreement": rust_values["top1_factorised_disagreement"],
        "cell_confidence": rust_values["cell_confidence"],
        "operator_confidence": rust_values["operator_confidence"],
        "joint_confidence": rust_values["joint_confidence"],
        # Channel availability is observed support-channel metadata. It is not a
        # truth label and is reported separately from Rust aggregate features.
        "internal_unresolvable_indicator": 1.0 if row.get("internal_counter_supports") else 0.0,
        "external_channel_available": 1.0 if row.get("external_counter_supports") else 0.0,
    }
    if arm == "RUST_SPARSE_SUPPORT_COHERENCE_SET_AGGREGATION":
        fmap["runtime_adversarial_pressure_norm"] = max(rust_values["coherence_pressure"], fmap["dominant_cluster_fraction"], fmap["collision_norm"])
    else:
        fmap["runtime_adversarial_pressure_norm"] = max(fmap["dominant_cluster_fraction"], fmap["collision_norm"])
    if arm == "RUST_SPARSE_COUNTERFACTUAL_DELTA_SET_AGGREGATION":
        fmap["counterfactual_pressure_norm"] = max(rust_values["counter_delta_pressure"], fmap["inverse_margin"])
    else:
        fmap["counterfactual_pressure_norm"] = fmap["inverse_margin"]
    fmap["support_budget_pressure_norm"] = 0.0
    fmap["runtime_support_budget_available"] = 0.0
    fmap["hard_support_budget_cap_norm"] = 0.0
    fmap["context_noise_norm"] = 0.0
    if arm == "HYBRID_SYMBOLIC_RUST_SET_AGGREGATION":
        for name in ["cell_confidence", "operator_confidence", "top1_factorised_disagreement"]:
            fmap[name] = float(symbolic_map.get(name, fmap[name]))
    if arm == "RANDOM_SCORE_AGGREGATION_CONTROL":
        fmap["context_noise_norm"] = max(fmap["entropy_norm"], 0.5)
        fmap["counterfactual_pressure_norm"] = max(fmap["counterfactual_pressure_norm"], fmap["context_noise_norm"])
    if arm == "AGGREGATION_ABLATION_CONTROL":
        fmap["scalar_confidence"] = 0.0
        fmap["joint_confidence"] = 0.0
        fmap["cell_confidence"] = 0.0
        fmap["operator_confidence"] = 0.0
        fmap["inverse_margin"] = 0.0
        fmap["entropy_norm"] = 0.0
        fmap["counterfactual_pressure_norm"] = 0.0
        fmap["runtime_adversarial_pressure_norm"] = 0.0
    return fmap


def build_pack(row, bundle, arm, idx, rust_features=None):
    base_vectors = support_vectors_for_row(row, bundle)
    symbolic_features, symbolic_map, scores, pred = symbolic_feature_map(row, bundle, base_vectors)
    if arm == "SYMBOLIC_SET_AGGREGATION_REFERENCE" or arm == "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY":
        fmap = symbolic_map
        features = symbolic_features
        rust_agg_used = False
        rust_values = {}
    else:
        key = f"{idx}:{arm}"
        rust_values = rust_features[key]
        fmap = feature_map_from_rust(row, rust_values, symbolic_map, arm)
        features = features_from_map(fmap)
        rust_agg_used = True
    base = (base_vectors, scores, pred)
    actions = {action: d51.run_action(row, bundle, action, base=base, arm_name=f"ACTION_{action}") for action in ACTIONS}
    return {
        "row_id": row["row_id"],
        "seed": row["seed"],
        "split": row["split"],
        "support_regime": row["support_regime"],
        "features": features,
        "feature_map": fmap,
        "actions": actions,
        "action_compact": {action: d51.compact_outcome(result) for action, result in actions.items()},
        "references": {},
        "track": "D65_SET_AGGREGATION",
        "mixed_source_track": "D65_SET_AGGREGATION",
        "aggregation_arm": arm,
        "rust_aggregation_used": rust_agg_used,
        "rust_aggregate_values": rust_values,
        "rust_aggregation_input_is_support_set": rust_agg_used,
        "python_precomputed_final_aggregate_label_used": False,
        "truth_hidden_from_controller_inputs": True,
    }


def learned_policy(pack, learned_gate):
    features = d62.gate_features(pack)
    for rule in learned_gate["rules"]:
        if features.get(rule["feature"], 0.0) >= rule["threshold"]:
            return rule["policy"], features, "learned_gate_over_set_aggregate_features"
    return learned_gate["default_policy"], features, "learned_gate_default_over_set_aggregate_features"


def truth_leak_policy(pack, rust_actions, idx):
    scored = []
    for policy in POLICY_MODULES:
        record = rust_actions[policy][idx]
        row = d59.output_from_action(pack, f"sentinel_{policy}", record["action"], d59.rust_trace(record))
        effective = row["correct"] or (row["support_regime"] == "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT" and row["abstained"])
        scored.append((1.0 if effective else 0.0, -row["total_support_used"], policy))
    return max(scored)[2]


def output_row(pack, arm, policy, action_record, gate_features, gate_basis, used_truth=False):
    row = d59.output_from_action(pack, arm, action_record["action"], d59.rust_trace(action_record))
    row["aggregation_arm"] = arm
    row["gate_selected_policy"] = policy
    row["gate_basis"] = gate_basis
    row["gate_features"] = gate_features
    row["rust_aggregation_used"] = bool(pack["rust_aggregation_used"])
    row["rust_aggregation_input_is_support_set"] = bool(pack["rust_aggregation_input_is_support_set"])
    row["python_precomputed_final_aggregate_label_used"] = False
    row["rust_aggregate_values"] = pack["rust_aggregate_values"]
    row["fair_arm"] = arm in FAIR_ARMS
    row["control_arm"] = arm in CONTROL_ARMS
    row["reference_only_arm"] = arm in REFERENCE_ONLY_ARMS
    row["gate_used_truth_label"] = bool(used_truth)
    return row


def build_eval_items(rows, bundle, rust_features):
    items = []
    for idx, row in enumerate(rows):
        for arm in ARMS:
            items.append({"arm": arm, "pack": build_pack(row, bundle, arm, idx, rust_features)})
    return items


def record_row(row, outputs, sample_counts, path):
    outputs.append(row)
    key = (row["arm"], row["support_regime"])
    if sample_counts[key] < ROW_SAMPLE_PER_ARM_REGIME_SPLIT:
        append_jsonl(path, row)
        sample_counts[key] += 1


def summarize_outputs(outputs):
    by_arm = defaultdict(list)
    by_arm_core = defaultdict(list)
    by_arm_regime = defaultdict(list)
    by_seed_core = defaultdict(list)
    rust_usage = defaultdict(Counter)
    action_counts = defaultdict(Counter)
    for row in outputs:
        arm = row["arm"]
        by_arm[arm].append(row)
        by_arm_regime[(arm, row["support_regime"])].append(row)
        action_counts[arm][row["selected_action"]] += 1
        rust_usage[arm]["rows"] += 1
        if row["rust_network_path_invoked"]:
            rust_usage[arm]["controller_rust_rows"] += 1
        if row["rust_aggregation_used"]:
            rust_usage[arm]["aggregation_rust_rows"] += 1
        if row["python_fallback_used"]:
            rust_usage[arm]["python_fallback_rows"] += 1
        if row["python_precomputed_final_aggregate_label_used"]:
            rust_usage[arm]["python_precomputed_final_aggregate_label_rows"] += 1
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
        "by_seed_core": {
            arm: {str(seed): d51.summarize(rows) for (a, seed), rows in by_seed_core.items() if a == arm and rows}
            for arm in ARMS
        },
        "action_distribution": {arm: dict(action_counts[arm]) for arm in ARMS},
        "rust_usage": {arm: dict(rust_usage[arm]) for arm in ARMS},
    }


def write_partial(out, split, outputs, completed, started):
    recent = outputs[-min(len(outputs), 5000):]
    write_json(
        out / f"partial_{split}_metrics_snapshot.json",
        {
            "split": split,
            "completed_outputs": completed,
            "elapsed_sec": time.time() - started,
            "recent_metrics": summarize_outputs(recent)["by_arm_core"],
        },
    )
    append_progress(out, "eval_progress", started, {"split": split, "completed_outputs": completed})


def evaluate_split(rows, bundle, policy_controllers, learned_gate, out, split, started, heartbeat_sec, repo_root, row_output_path):
    rust_features, aggregation_report = run_rust_aggregation_bridge(out, repo_root, rows, bundle, split, started, heartbeat_sec)
    items = build_eval_items(rows, bundle, rust_features)
    packs = [item["pack"] for item in items]
    policy_actions, policy_report = d59.run_rust_multi_bridge(out, repo_root, policy_controllers, packs, split, "d65_policy_eval", started)
    outputs = []
    sample_counts = Counter()
    completed = 0
    last = 0.0
    for idx, item in enumerate(items):
        arm = item["arm"]
        pack = item["pack"]
        if arm == "TRUTH_LEAK_SENTINEL_REFERENCE_ONLY":
            policy = truth_leak_policy(pack, policy_actions, idx)
            gate_features = d62.gate_features(pack)
            basis = "reference_only_best_policy_after_truth_scoring"
            used_truth = True
        else:
            policy, gate_features, basis = learned_policy(pack, learned_gate)
            used_truth = False
        action_record = policy_actions[policy][idx]
        row = output_row(pack, arm, policy, action_record, gate_features, basis, used_truth)
        record_row(row, outputs, sample_counts, row_output_path)
        completed += 1
        now = time.time()
        if now - last >= heartbeat_sec or completed >= len(items):
            last = now
            write_partial(out, split, outputs, completed, started)
    return outputs, {"aggregation": aggregation_report, "controller": policy_report}


def pairwise_report(outputs, arm_a, arm_b):
    by_key = {}
    for row in outputs:
        key = (row["split"], row["seed"], row["row_id"], row["support_regime"])
        by_key[(key, row["arm"])] = row
    pairs = []
    for (key, arm), row_a in list(by_key.items()):
        if arm != arm_a:
            continue
        row_b = by_key.get((key, arm_b))
        if row_b:
            pairs.append((row_a, row_b))
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "rows": len(pairs),
        "action_disagreement_rate": d51.mean([1.0 if a["selected_action"] != b["selected_action"] else 0.0 for a, b in pairs]),
        "correctness_disagreement_rate": d51.mean([1.0 if bool(a["correct"]) != bool(b["correct"]) else 0.0 for a, b in pairs]),
        "arm_a_accuracy": d51.mean([1.0 if a["correct"] else 0.0 for a, _b in pairs]),
        "arm_b_accuracy": d51.mean([1.0 if b["correct"] else 0.0 for _a, b in pairs]),
    }


def arm_regime(metrics, arm, regime, field="accuracy"):
    return metrics["by_arm_and_regime"].get(arm, {}).get(regime, {}).get(field, 0.0)


def make_decision(test_metrics, failed_jobs):
    core = test_metrics["by_arm_core"]
    if failed_jobs:
        return {
            "decision": "set_invariant_ipf_aggregation_not_confirmed",
            "verdict": "D65_FAILED_JOBS",
            "next": "D65_REPAIR",
            "best_arm": None,
            "reason": "failed_jobs not empty",
        }
    ref = core["SYMBOLIC_SET_AGGREGATION_REFERENCE"]["exact_joint_accuracy"]
    candidate_arms = [
        arm for arm in RUST_AGGREGATION_ARMS
        if arm not in CONTROL_ARMS
    ]
    best = max(candidate_arms, key=lambda arm: (core[arm]["exact_joint_accuracy"], -core[arm]["average_total_support_used"]))
    pure_rust = [arm for arm in candidate_arms if arm != "HYBRID_SYMBOLIC_RUST_SET_AGGREGATION"]
    best_pure = max(pure_rust, key=lambda arm: (core[arm]["exact_joint_accuracy"], -core[arm]["average_total_support_used"]))
    order_delta = abs(core["RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION"]["exact_joint_accuracy"] - core["SUPPORT_ORDER_SHUFFLE_NOOP_CONTROL"]["exact_joint_accuracy"])
    content_gap = core[best]["exact_joint_accuracy"] - core["SUPPORT_CONTENT_CORRUPTION_CONTROL"]["exact_joint_accuracy"]
    random_gap = core[best]["exact_joint_accuracy"] - core["RANDOM_SCORE_AGGREGATION_CONTROL"]["exact_joint_accuracy"]
    ablation_gap = core[best]["exact_joint_accuracy"] - core["AGGREGATION_ABLATION_CONTROL"]["exact_joint_accuracy"]
    strong = (
        core[best]["exact_joint_accuracy"] >= ref - 0.007
        and arm_regime(test_metrics, best, "CORRELATED_ECHO_SUPPORT") >= 0.95
        and arm_regime(test_metrics, best, "ADVERSARIAL_DISTRACTOR_SUPPORT") >= 0.95
        and arm_regime(test_metrics, best, "EXTERNAL_TEST_REQUIRED_SUPPORT") >= 0.95
        and arm_regime(test_metrics, best, "INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT", "abstain_rate") >= 0.99
        and core[best]["false_confidence_rate"] <= 0.01
        and order_delta <= 0.01
        and content_gap >= 0.02
        and random_gap >= 0.02
        and ablation_gap >= 0.02
    )
    if order_delta > 0.03:
        return {
            "decision": "order_dependence_reappeared",
            "verdict": "D65_ORDER_DEPENDENCE_REAPPEARED",
            "next": "D65O_ORDER_ARTIFACT_REPAIR",
            "best_arm": best,
            "reason": f"support order shuffle delta {order_delta:.6f} exceeded tolerance",
        }
    if strong and best == "HYBRID_SYMBOLIC_RUST_SET_AGGREGATION" and core[best_pure]["exact_joint_accuracy"] < ref - 0.007:
        return {
            "decision": "hybrid_set_aggregation_positive",
            "verdict": "D65_HYBRID_SET_AGGREGATION_POSITIVE",
            "next": "D65B_FULL_RUST_SET_AGGREGATION_REPAIR",
            "best_arm": best,
            "reason": "hybrid passed but pure Rust aggregation did not match the symbolic reference floor",
        }
    if strong:
        return {
            "decision": "set_invariant_ipf_aggregation_confirmed",
            "verdict": "D65_SET_INVARIANT_IPF_AGGREGATION_CONFIRMED",
            "next": "D66_RUST_SPARSE_SUPPORT_SCORING_MIGRATION_PLAN",
            "best_arm": best,
            "reason": "best Rust/hybrid set aggregation matched reference floor and controls were worse",
        }
    return {
        "decision": "set_invariant_ipf_aggregation_not_confirmed",
        "verdict": "D65_SET_INVARIANT_IPF_AGGREGATION_NOT_CONFIRMED",
        "next": "D65_REPAIR",
        "best_arm": best,
        "reason": {
            "reference_exact": ref,
            "best_exact": core[best]["exact_joint_accuracy"],
            "order_delta": order_delta,
            "content_gap": content_gap,
            "random_gap": random_gap,
            "ablation_gap": ablation_gap,
        },
    }


def make_reports(out, aggregate, decision):
    test_metrics = aggregate["test_metrics"]
    outputs = aggregate["_test_outputs_for_reports"]
    reports = {
        "d64u_upstream_manifest.json": aggregate["d64u_upstream_manifest"],
        "set_aggregation_definition_report.json": {
            "input_to_rust_arms": "flat unordered support/evidence score-bin set, support_count x candidate_count",
            "rust_path": "generated Rust harness using instnct-core Network::propagate_sparse per support vector",
            "not_used": [
                "Python-precomputed final aggregate labels",
                "truth joint",
                "truth pair",
                "truth operator",
                "row id lookup",
                "Python hash",
            ],
            "runtime_channel_metadata": [
                "external_channel_available from actual external support channel presence",
                "internal_unresolvable_indicator from constrained internal counter-support channel presence",
            ],
        },
        "order_invariance_report.json": {
            "set_vs_order_shuffle": pairwise_report(outputs, "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION", "SUPPORT_ORDER_SHUFFLE_NOOP_CONTROL"),
            "metrics": {
                "set": test_metrics["by_arm_core"].get("RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION"),
                "order_shuffle": test_metrics["by_arm_core"].get("SUPPORT_ORDER_SHUFFLE_NOOP_CONTROL"),
            },
        },
        "score_shape_aggregation_report.json": {
            arm: test_metrics["by_arm_core"].get(arm)
            for arm in [
                "RUST_SPARSE_SUM_AGGREGATION",
                "RUST_SPARSE_MEAN_AGGREGATION",
                "RUST_SPARSE_NORMALIZED_SHAPE_AGGREGATION",
                "RUST_SPARSE_SET_INVARIANT_IPF_AGGREGATION",
            ]
        },
        "rust_aggregation_mapping_report.json": {
            "arms": {arm: RUST_MODE_BY_ARM.get(arm) for arm in RUST_AGGREGATION_ARMS},
            "candidate_identity_output_used": False,
            "features_emitted": [
                "scalar_confidence",
                "inverse_margin",
                "entropy_norm",
                "collision_norm",
                "dominant_cluster_fraction",
                "support_cluster_count_norm",
                "cell_confidence",
                "operator_confidence",
                "joint_confidence",
            ],
        },
        "aggregation_quality_report.json": {
            "by_arm_core": test_metrics["by_arm_core"],
            "by_arm_and_regime": test_metrics["by_arm_and_regime"],
        },
        "controller_with_set_aggregation_report.json": {
            "action_distribution": test_metrics["action_distribution"],
            "symbolic_reference": test_metrics["by_arm_core"].get("SYMBOLIC_SET_AGGREGATION_REFERENCE"),
            "best_arm": decision.get("best_arm"),
        },
        "support_content_corruption_report.json": {
            "content_corruption": test_metrics["by_arm_core"].get("SUPPORT_CONTENT_CORRUPTION_CONTROL"),
            "random_score": test_metrics["by_arm_core"].get("RANDOM_SCORE_AGGREGATION_CONTROL"),
            "ablation": test_metrics["by_arm_core"].get("AGGREGATION_ABLATION_CONTROL"),
        },
        "truth_leak_audit_report.json": {
            "fair_arms": FAIR_ARMS,
            "reference_only_arms": REFERENCE_ONLY_ARMS,
            "fair_arms_using_truth_label": [],
            "python_precomputed_final_aggregate_label_used_by_fair_arms": False,
            "truth_leak_sentinel": test_metrics["by_arm_core"].get("TRUTH_LEAK_SENTINEL_REFERENCE_ONLY"),
        },
        "rust_invocation_report.json": aggregate["rust_invocation_report"],
    }
    for name, value in reports.items():
        write_json(out / name, value)
    return reports


def write_report(out, decision, metrics):
    rows = [
        "# D65 Set Invariant IPF Aggregation Prototype Result",
        "",
        "Decision:",
        "",
        "```text",
        f"decision = {decision['decision']}",
        f"verdict = {decision['verdict']}",
        f"next = {decision['next']}",
        f"best_arm = {decision['best_arm']}",
        "```",
        "",
        "| arm | exact core | corr | adv | external | abstain | false conf | support |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_core = metrics["by_arm_core"]
    by_regime = metrics["by_arm_and_regime"]
    ordered = sorted(ARMS, key=lambda arm: (-(by_core.get(arm, {}).get("exact_joint_accuracy", 0.0)), arm))
    for arm in ordered:
        core = by_core.get(arm, {})
        regimes = by_regime.get(arm, {})
        rows.append(
            f"| {arm} | {core.get('exact_joint_accuracy', 0.0):.6f} | "
            f"{regimes.get('CORRELATED_ECHO_SUPPORT', {}).get('accuracy', 0.0):.6f} | "
            f"{regimes.get('ADVERSARIAL_DISTRACTOR_SUPPORT', {}).get('accuracy', 0.0):.6f} | "
            f"{regimes.get('EXTERNAL_TEST_REQUIRED_SUPPORT', {}).get('accuracy', 0.0):.6f} | "
            f"{regimes.get('INDISTINGUISHABLE_CORRELATED_FALSE_SUPPORT', {}).get('abstain_rate', 0.0):.6f} | "
            f"{core.get('false_confidence_rate', 0.0):.6f} | "
            f"{core.get('average_total_support_used', 0.0):.4f} |"
        )
    rows += ["", "Boundary:", "", "```text", BOUNDARY, "```"]
    (out / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", default="12301,12302,12303,12304,12305")
    parser.add_argument("--train-rows-per-seed", type=int, default=300)
    parser.add_argument("--test-rows-per-seed", type=int, default=300)
    parser.add_argument("--ood-rows-per-seed", type=int, default=300)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--cpu-target", default="50-75")
    parser.add_argument("--heartbeat-sec", type=float, default=20.0)
    parser.add_argument("--scale-mode", default="scale-lite")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    repo_root = Path(__file__).resolve().parents[2]
    seeds = parse_seeds(args.seeds)
    failed_jobs = []
    write_json(out / "queue.json", {"task": TASK, "args": vars(args), "seeds": seeds, "boundary": BOUNDARY})
    append_progress(out, "started", started, {"seeds": seeds, "scale_mode": args.scale_mode})
    write_json(out / "compute_probe.json", {"cpu_count": os.cpu_count(), "workers": d51.worker_count_from_arg(args.workers), "cpu_target": args.cpu_target})

    d64u_manifest = make_d64u_upstream_manifest(repo_root)
    write_json(out / "d64u_upstream_manifest.json", d64u_manifest)
    policy_controllers, learned_gate = load_policy_modules(repo_root)
    bundle = d55.d49_bundle()
    write_json(
        out / "dataset_manifest.json",
        {
            "task": "controlled_symbolic_joint_formula_discovery",
            "primary_space": PRIMARY_SPACE,
            "support_count": SUPPORT_COUNT,
            "regimes": REGIMES,
            "core_regimes": CORE_REGIMES,
            "arms": ARMS,
            "rust_aggregation_arms": RUST_AGGREGATION_ARMS,
            "truth_hidden_from_controller_inputs": True,
            "rust_arms_receive_support_evidence_set_representation": True,
            "python_precomputed_final_aggregate_label_used": False,
            "formula_solver_learning_used": False,
            "controller_only_not_formula_solver": True,
        },
    )

    train_rows = d51.make_rows_with_progress(seeds, args.train_rows_per_seed, "train", bundle, out, started, args.heartbeat_sec)
    write_json(out / "partial_training_rows_generated.json", {"rows": len(train_rows), "training_used": False})
    test_rows = d51.make_rows_with_progress(seeds, args.test_rows_per_seed, "test", bundle, out, started, args.heartbeat_sec)
    ood_rows = d51.make_rows_with_progress(seeds, args.ood_rows_per_seed, "ood", bundle, out, started, args.heartbeat_sec)

    row_test = out / "row_outputs_test.jsonl"
    row_ood = out / "row_outputs_ood.jsonl"
    for path in [row_test, row_ood]:
        if path.exists():
            path.unlink()

    try:
        test_outputs, test_report = evaluate_split(test_rows, bundle, policy_controllers, learned_gate, out, "test", started, args.heartbeat_sec, repo_root, row_test)
        write_json(out / "partial_test_metrics.json", summarize_outputs(test_outputs))
        ood_outputs, ood_report = evaluate_split(ood_rows, bundle, policy_controllers, learned_gate, out, "ood", started, args.heartbeat_sec, repo_root, row_ood)
        write_json(out / "partial_ood_metrics.json", summarize_outputs(ood_outputs))
    except Exception as exc:
        failed_jobs.append({"error": repr(exc)})
        write_json(out / "error.json", {"error": repr(exc), "failed_jobs": failed_jobs})
        raise

    test_metrics = summarize_outputs(test_outputs)
    ood_metrics = summarize_outputs(ood_outputs)
    decision = make_decision(test_metrics, failed_jobs)
    fallback_rows = 0
    rust_controller_rows = 0
    rust_aggregation_rows = 0
    precomputed_label_rows = 0
    for metrics in [test_metrics, ood_metrics]:
        for counts in metrics["rust_usage"].values():
            fallback_rows += counts.get("python_fallback_rows", 0)
            rust_controller_rows += counts.get("controller_rust_rows", 0)
            rust_aggregation_rows += counts.get("aggregation_rust_rows", 0)
            precomputed_label_rows += counts.get("python_precomputed_final_aggregate_label_rows", 0)

    aggregate = {
        "task": TASK,
        "failed_jobs": failed_jobs,
        "d64u_upstream_manifest": d64u_manifest,
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
        "rust_invocation_report": {"test": test_report, "ood": ood_report},
        "rust_path_invoked": rust_controller_rows > 0 and rust_aggregation_rows > 0,
        "rust_controller_rows": rust_controller_rows,
        "rust_aggregation_rows": rust_aggregation_rows,
        "fallback_rows": fallback_rows,
        "python_precomputed_final_aggregate_label_rows": precomputed_label_rows,
        "decision": decision,
        "boundary": BOUNDARY,
    }
    reports = make_reports(out, {**aggregate, "_test_outputs_for_reports": test_outputs}, decision)
    write_json(out / "aggregate_metrics.json", aggregate)
    write_json(out / "decision.json", decision)
    write_json(
        out / "summary.json",
        {
            "task": TASK,
            "decision": decision["decision"],
            "verdict": decision["verdict"],
            "next": decision["next"],
            "best_arm": decision.get("best_arm"),
            "rust_path_invoked": aggregate["rust_path_invoked"],
            "rust_controller_rows": rust_controller_rows,
            "rust_aggregation_rows": rust_aggregation_rows,
            "fallback_rows": fallback_rows,
            "python_precomputed_final_aggregate_label_rows": precomputed_label_rows,
            "failed_jobs": failed_jobs,
            "artifact_reports": sorted(reports.keys()),
            "boundary": BOUNDARY,
        },
    )
    write_report(out, decision, test_metrics)
    append_progress(out, "completed", started, {"decision": decision["decision"], "failed_jobs": failed_jobs})
    print(json.dumps(load_json(out / "summary.json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
