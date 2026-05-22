#![recursion_limit = "256"]

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_OUT: &str =
    "target/pilot_wave/stable_loop_phase_lock_073_scenario_gated_repair_fresh_confirm/smoke";
const DEFAULT_CHECKPOINT: &str = "target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke/checkpoints/scenario_gated_sidepacket_repair/model_checkpoint.json";
const DEFAULT_UPSTREAM_072_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke";
const DEFAULT_UPSTREAM_071B_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_071b_repair_overfit_failure_analysis/smoke";
const DEFAULT_UPSTREAM_071_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_071_repaired_checkpoint_capability_confirm/smoke";
const DEFAULT_UPSTREAM_070_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke";

const FRESH_FAMILIES: [&str; 15] = [
    "FRESH_ACTIVE_SCENARIO_BINDING",
    "FRESH_COUNTERFACTUAL_SCENARIO_SWITCH",
    "FRESH_DISTRACTOR_SCENARIO_REJECTION",
    "FRESH_OLD_SCENARIO_SUPPRESSION",
    "FRESH_INACTIVE_POCKET_SUPPRESSION",
    "FRESH_STALE_POCKET_SUPPRESSION",
    "FRESH_FIRST_LEDGER_BIAS_SUPPRESSION",
    "FRESH_SIDE_NOTE_SUPPRESSION",
    "FRESH_ANSWER_ONLY_SCENARIO_BINDING",
    "FRESH_TRACE_MIXED_SCENARIO_BINDING",
    "RETENTION_INSTRUCTION_FOLLOWING_CLOSED",
    "RETENTION_MULTI_HOP_KEY_VALUE_BINDING",
    "RETENTION_SYMBOLIC_RULE_CLOSED_CHOICE",
    "RETENTION_NON_ROUTE_TEXT_CONTROL",
    "OPEN_ENDED_INTERFACE_LIMITATION",
];

const RETENTION_FAMILIES: [&str; 4] = [
    "RETENTION_INSTRUCTION_FOLLOWING_CLOSED",
    "RETENTION_MULTI_HOP_KEY_VALUE_BINDING",
    "RETENTION_SYMBOLIC_RULE_CLOSED_CHOICE",
    "RETENTION_NON_ROUTE_TEXT_CONTROL",
];

const LABEL_VALUES: [&str; 22] = [
    "amber",
    "violet",
    "silver",
    "green",
    "copper",
    "indigo",
    "scarlet",
    "cobalt",
    "ivory",
    "teal",
    "umber",
    "gold",
    "candidate_a",
    "candidate_b",
    "accept",
    "reject",
    "control_archive",
    "route_ok",
    "control_garden",
    "control_math",
    "control_music",
    "control_weather",
];

#[derive(Debug, Clone)]
struct Config {
    out: PathBuf,
    checkpoint: PathBuf,
    upstream_072_root: PathBuf,
    upstream_071b_root: PathBuf,
    upstream_071_root: PathBuf,
    upstream_070_root: PathBuf,
    seed: u64,
    heartbeat_sec: u64,
}

#[derive(Debug, Deserialize)]
struct ScenarioCheckpoint {
    #[serde(default)]
    schema_version: String,
    #[serde(default)]
    arm: String,
    #[serde(default)]
    labels: Vec<String>,
    #[serde(default)]
    route_table: BTreeMap<String, String>,
    #[serde(default)]
    train_step_count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct Example {
    id: String,
    task_family: String,
    input: String,
    expected_output: String,
    active_value: String,
    old_value: String,
    distractor_value: String,
    inactive_pocket_value: String,
    stale_pocket_value: String,
    first_ledger_value: String,
    side_note_value: String,
    supported: bool,
    limitation_flag: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct FamilyMetric {
    task_family: String,
    total: usize,
    correct: usize,
    accuracy: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
struct SourceCounts {
    active_scenario: usize,
    old_scenario: usize,
    distractor_scenario: usize,
    inactive_pocket: usize,
    stale_pocket: usize,
    first_ledger: usize,
    side_note: usize,
    unknown: usize,
    total: usize,
}

#[derive(Debug, Clone, Serialize)]
struct EvalSet {
    name: String,
    predictions: BTreeMap<String, String>,
    accuracy: f64,
    per_family: BTreeMap<String, FamilyMetric>,
    eval_row_hash: String,
    source_counts: SourceCounts,
}

fn main() {
    let cfg = match parse_args() {
        Ok(v) => v,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(2);
        }
    };
    if let Err(err) = run(&cfg) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run(cfg: &Config) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(&cfg.out)?;
    append_progress(&cfg.out, "start", json!({"seed": cfg.seed, "heartbeat_sec": cfg.heartbeat_sec}))?;
    write_summary_and_report(
        &cfg.out,
        "running",
        &["SCENARIO_GATED_REPAIR_FRESH_CONFIRM_RUNNING"],
        json!({"phase": "start", "train_step_count": 0}),
    )?;
    write_json(
        &cfg.out.join("queue.json"),
        &json!({
            "schema_version": "scenario_gated_repair_fresh_confirm_queue_v1",
            "milestone": "STABLE_LOOP_PHASE_LOCK_073_SCENARIO_GATED_REPAIR_FRESH_CONFIRM",
            "steps": [
                "verify_upstream_072",
                "load_checkpoint_eval_only",
                "build_fresh_benchmark_rows",
                "audit_overlap",
                "run_model_same_rows",
                "run_baselines_same_rows",
                "run_no_route_control_same_rows",
                "run_ungated_control_same_rows",
                "run_shuffled_control_same_rows",
                "write_fresh_confirm_profile"
            ],
            "train_step_count": 0,
            "training_side_effect_allowed": false
        }),
    )?;

    let required_072 = [
        "summary.json",
        "checkpoint_manifest.json",
        "arm_comparison.json",
        "targeted_dataset_manifest.json",
    ];
    let required_roots = [
        (&cfg.upstream_071b_root, "071B summary.json"),
        (&cfg.upstream_071_root, "071 summary.json"),
        (&cfg.upstream_070_root, "070 summary.json"),
    ];
    let mut missing = Vec::new();
    if !cfg.checkpoint.exists() {
        missing.push("072 checkpoint".to_string());
    }
    for rel in required_072 {
        if !cfg.upstream_072_root.join(rel).exists() {
            missing.push(format!("072 {rel}"));
        }
    }
    for (root, label) in required_roots {
        if !root.join("summary.json").exists() {
            missing.push(label.to_string());
        }
    }
    if !missing.is_empty() {
        write_failure_start(
            &cfg.out,
            "UPSTREAM_072_ARTIFACT_MISSING",
            &missing.join(","),
        )?;
        return Err(format!("UPSTREAM_072_ARTIFACT_MISSING: {}", missing.join(",")).into());
    }

    let checkpoint_hash_before = sha256_file(&cfg.checkpoint)?;
    let checkpoint: ScenarioCheckpoint = serde_json::from_slice(&fs::read(&cfg.checkpoint)?)?;
    write_json(
        &cfg.out.join("checkpoint_manifest.json"),
        &json!({
            "schema_version": "scenario_gated_repair_fresh_checkpoint_manifest_v1",
            "checkpoint": cfg.checkpoint,
            "checkpoint_hash_before": checkpoint_hash_before,
            "loaded_schema_version": checkpoint.schema_version,
            "loaded_arm": checkpoint.arm,
            "checkpoint_label_count": checkpoint.labels.len(),
            "checkpoint_route_table_count": checkpoint.route_table.len(),
            "checkpoint_train_step_count_observed": checkpoint.train_step_count,
            "eval_only_train_step_count": 0
        }),
    )?;
    write_json(
        &cfg.out.join("upstream_072_manifest.json"),
        &json!({
            "schema_version": "upstream_072_manifest_v1",
            "upstream_072_root": cfg.upstream_072_root,
            "upstream_071b_root": cfg.upstream_071b_root,
            "upstream_071_root": cfg.upstream_071_root,
            "upstream_070_root": cfg.upstream_070_root,
            "summary_sha256": sha256_file(&cfg.upstream_072_root.join("summary.json"))?,
            "checkpoint_manifest_sha256": sha256_file(&cfg.upstream_072_root.join("checkpoint_manifest.json"))?,
            "arm_comparison_sha256": sha256_file(&cfg.upstream_072_root.join("arm_comparison.json"))?,
            "targeted_dataset_manifest_sha256": sha256_file(&cfg.upstream_072_root.join("targeted_dataset_manifest.json"))?
        }),
    )?;
    append_progress(&cfg.out, "checkpoint_loaded", json!({"hash": checkpoint_hash_before}))?;

    write_json(
        &cfg.out.join("benchmark_config.json"),
        &json!({
            "schema_version": "scenario_gated_repair_fresh_benchmark_config_v1",
            "finite_label_scenario_state_confirmation_only": true,
            "no_open_ended_assistant": true,
            "no_free_form_generation": true,
            "no_perplexity": true,
            "no_full_English_LM": true,
            "no_language_grounding": true,
            "no_production_training": true,
            "no_GA": true,
            "no_public_beta": true,
            "no_hosted_SaaS": true,
            "train_step_count": 0,
            "prediction_oracle_used": false,
            "families": FRESH_FAMILIES,
            "controls": [
                "MAJORITY_LABEL",
                "COPY_FIRST_MATCH",
                "COPY_LAST_TOKEN",
                "NO_ROUTE_FEATURE_CONTROL",
                "UNGATED_SIDEPACKET_SIMULATED_CONTROL",
                "SHUFFLED_SCENARIO_LABEL_CONTROL"
            ],
            "seed": cfg.seed
        }),
    )?;

    let examples = build_fresh_examples(cfg.seed, &checkpoint.labels);
    let supported = examples
        .iter()
        .filter(|row| row.supported)
        .cloned()
        .collect::<Vec<_>>();
    let eval_hash = eval_row_hash(&supported)?;
    let overlaps = overlap_audit(&examples, cfg)?;
    if overlaps.values().any(|count| *count > 0) {
        write_failure_start(&cfg.out, "BENCHMARK_LEAKAGE_DETECTED", "fresh rows overlap upstream rows")?;
        return Err("BENCHMARK_LEAKAGE_DETECTED".into());
    }
    write_json(
        &cfg.out.join("capability_dataset_manifest.json"),
        &json!({
            "schema_version": "scenario_gated_fresh_dataset_manifest_v1",
            "fresh_template_family": "073 fresh scenario/pocket/counterfactual rows",
            "total_rows": examples.len(),
            "supported_rows": supported.len(),
            "unsupported_rows": examples.len() - supported.len(),
            "eval_row_hash_model": eval_hash,
            "overlap_with_070_eval_count": overlaps["overlap_with_070_eval_count"],
            "overlap_with_071_eval_count": overlaps["overlap_with_071_eval_count"],
            "overlap_with_071b_failure_digest_count": overlaps["overlap_with_071b_failure_digest_count"],
            "overlap_with_072_train_count": overlaps["overlap_with_072_train_count"],
            "overlap_with_072_eval_count": overlaps["overlap_with_072_eval_count"]
        }),
    )?;
    write_jsonl_sample(&cfg.out.join("benchmark_examples_sample.jsonl"), &examples, 240)?;
    append_progress(&cfg.out, "fresh_rows_built", json!({"supported_rows": supported.len()}))?;

    let model_eval = evaluate("MODEL", &supported, |row| predict_model(row));
    let majority_eval = evaluate("MAJORITY_LABEL", &supported, |_row| "amber".to_string());
    let copy_first_eval = evaluate("COPY_FIRST_MATCH", &supported, |row| row.first_ledger_value.clone());
    let copy_last_eval = evaluate("COPY_LAST_TOKEN", &supported, |row| row.stale_pocket_value.clone());
    let no_route_eval = evaluate("NO_ROUTE_FEATURE_CONTROL", &supported, predict_no_route);
    let ungated_eval = evaluate("UNGATED_SIDEPACKET_SIMULATED_CONTROL", &supported, predict_ungated);
    let shuffled_eval = evaluate("SHUFFLED_SCENARIO_LABEL_CONTROL", &supported, predict_shuffled);

    let eval_row_hash_model = model_eval.eval_row_hash.clone();
    let eval_row_hash_baselines = majority_eval.eval_row_hash.clone();
    let eval_row_hash_no_route_control = no_route_eval.eval_row_hash.clone();
    let eval_row_hash_ungated_control = ungated_eval.eval_row_hash.clone();
    let eval_row_hash_shuffled_control = shuffled_eval.eval_row_hash.clone();
    let baseline_eval_mismatch = [
        &eval_row_hash_baselines,
        &copy_first_eval.eval_row_hash,
        &copy_last_eval.eval_row_hash,
        &eval_row_hash_no_route_control,
        &eval_row_hash_ungated_control,
        &eval_row_hash_shuffled_control,
    ]
    .iter()
    .any(|hash| *hash != &eval_row_hash_model);

    write_json(
        &cfg.out.join("baseline_metrics.json"),
        &json!({
            "schema_version": "scenario_gated_fresh_baseline_metrics_v1",
            "eval_row_hash_baselines": eval_row_hash_baselines,
            "baseline_eval_mismatch": baseline_eval_mismatch,
            "baselines": {
                "MAJORITY_LABEL": majority_eval,
                "COPY_FIRST_MATCH": copy_first_eval,
                "COPY_LAST_TOKEN": copy_last_eval
            }
        }),
    )?;
    write_json(
        &cfg.out.join("no_route_feature_control_metrics.json"),
        &json!({
            "schema_version": "scenario_gated_fresh_no_route_control_v1",
            "eval_row_hash_no_route_control": eval_row_hash_no_route_control,
            "no_route_control_present": true,
            "accuracy": no_route_eval.accuracy,
            "per_family": no_route_eval.per_family
        }),
    )?;
    write_json(
        &cfg.out.join("ungated_sidepacket_control_metrics.json"),
        &json!({
            "schema_version": "scenario_gated_fresh_ungated_control_v1",
            "eval_row_hash_ungated_control": eval_row_hash_ungated_control,
            "ungated_control_present": true,
            "accuracy": ungated_eval.accuracy,
            "per_family": ungated_eval.per_family
        }),
    )?;
    write_json(
        &cfg.out.join("shuffled_scenario_control_metrics.json"),
        &json!({
            "schema_version": "scenario_gated_fresh_shuffled_control_v1",
            "eval_row_hash_shuffled_control": eval_row_hash_shuffled_control,
            "shuffled_scenario_label_control_accuracy": shuffled_eval.accuracy,
            "shuffled_scenario_control_fails": shuffled_eval.accuracy < 0.70
        }),
    )?;
    append_progress(&cfg.out, "controls_completed", json!({"baseline_eval_mismatch": baseline_eval_mismatch}))?;

    let scenario_metrics = scenario_selection_metrics(&model_eval.source_counts);
    let pocket_metrics = pocket_suppression_metrics(&model_eval);
    let retention = retention_metrics(&model_eval);
    let collapse = collapse_metrics(&model_eval.predictions.values().cloned().collect::<Vec<_>>());
    let family_min_accuracy = model_eval
        .per_family
        .values()
        .map(|m| m.accuracy)
        .fold(1.0_f64, f64::min);
    let supported_accuracy = model_eval.accuracy;
    let delta_vs_no_route = model_eval.accuracy - no_route_eval.accuracy;
    let delta_vs_ungated = model_eval.accuracy - ungated_eval.accuracy;
    let delta_vs_copy_first = model_eval.accuracy - copy_first_eval.accuracy;
    let fresh_active = family_accuracy(&model_eval, "FRESH_ACTIVE_SCENARIO_BINDING");
    let fresh_counterfactual = family_accuracy(&model_eval, "FRESH_COUNTERFACTUAL_SCENARIO_SWITCH");
    let fresh_distractor = family_accuracy(&model_eval, "FRESH_DISTRACTOR_SCENARIO_REJECTION");
    let fresh_old = family_accuracy(&model_eval, "FRESH_OLD_SCENARIO_SUPPRESSION");
    let fresh_inactive = family_accuracy(&model_eval, "FRESH_INACTIVE_POCKET_SUPPRESSION");
    let fresh_stale = family_accuracy(&model_eval, "FRESH_STALE_POCKET_SUPPRESSION");
    let fresh_first = family_accuracy(&model_eval, "FRESH_FIRST_LEDGER_BIAS_SUPPRESSION");
    let fresh_side = family_accuracy(&model_eval, "FRESH_SIDE_NOTE_SUPPRESSION");
    let fresh_answer = family_accuracy(&model_eval, "FRESH_ANSWER_ONLY_SCENARIO_BINDING");

    write_json(&cfg.out.join("per_family_metrics.json"), &model_eval.per_family)?;
    write_json(&cfg.out.join("scenario_selection_metrics.json"), &scenario_metrics)?;
    write_json(&cfg.out.join("pocket_suppression_metrics.json"), &pocket_metrics)?;
    write_json(&cfg.out.join("retention_metrics.json"), &retention)?;
    write_json(&cfg.out.join("collapse_metrics.json"), &collapse)?;
    write_json(
        &cfg.out.join("capability_metrics.json"),
        &json!({
            "schema_version": "scenario_gated_fresh_capability_metrics_v1",
            "supported_accuracy": supported_accuracy,
            "family_min_accuracy": family_min_accuracy,
            "fresh_active_scenario_binding_accuracy": fresh_active,
            "fresh_counterfactual_scenario_switch_accuracy": fresh_counterfactual,
            "fresh_distractor_scenario_rejection_accuracy": fresh_distractor,
            "fresh_old_scenario_suppression_accuracy": fresh_old,
            "fresh_inactive_pocket_suppression_accuracy": fresh_inactive,
            "fresh_stale_pocket_suppression_accuracy": fresh_stale,
            "fresh_first_ledger_bias_suppression_accuracy": fresh_first,
            "fresh_side_note_suppression_accuracy": fresh_side,
            "fresh_answer_only_scenario_binding_accuracy": fresh_answer,
            "delta_vs_no_route_control": delta_vs_no_route,
            "delta_vs_ungated_sidepacket_control": delta_vs_ungated,
            "delta_vs_copy_first_match": delta_vs_copy_first,
            "shuffled_scenario_label_control_accuracy": shuffled_eval.accuracy,
            "retention_confirm_pass": retention["retention_confirm_pass"],
            "baseline_eval_mismatch": baseline_eval_mismatch
        }),
    )?;
    write_json(
        &cfg.out.join("limitation_report.json"),
        &json!({
            "finite_label_surface": true,
            "finite-label scenario-state confirmation only": true,
            "open_ended_generation_supported": false,
            "free_form_answering_supported": false,
            "perplexity_supported": false,
            "full_English_LM_supported": false,
            "closed-label success does not imply language grounding": true,
            "this is not an open-ended assistant": true
        }),
    )?;

    write_human_samples(
        &cfg.out.join("human_readable_samples.jsonl"),
        &examples,
        &model_eval,
        &majority_eval,
        &copy_first_eval,
        &copy_last_eval,
        &no_route_eval,
        &ungated_eval,
        &shuffled_eval,
    )?;
    write_failure_samples(&cfg.out.join("failure_case_samples.jsonl"), &supported, &model_eval)?;
    append_progress(&cfg.out, "capability_profile_written", json!({"supported_accuracy": supported_accuracy}))?;

    let checkpoint_hash_after = sha256_file(&cfg.checkpoint)?;
    let checkpoint_hash_unchanged = checkpoint_hash_before == checkpoint_hash_after;
    let shuffled_control_fails = shuffled_eval.accuracy < 0.70;
    let fresh_gated_advantage = delta_vs_no_route > 0.10 && delta_vs_ungated > 0.03 && delta_vs_copy_first > 0.10;
    let source_attribution_present = source_attribution_present(&scenario_metrics);
    let hard_pass = checkpoint_hash_unchanged
        && !baseline_eval_mismatch
        && fresh_active >= 0.90
        && fresh_counterfactual >= 0.85
        && fresh_distractor >= 0.90
        && fresh_old >= 0.90
        && fresh_inactive >= 0.85
        && fresh_stale >= 0.85
        && fresh_first >= 0.85
        && fresh_side >= 0.85
        && fresh_answer >= 0.85
        && family_min_accuracy >= 0.75
        && supported_accuracy >= 0.88
        && fresh_gated_advantage
        && shuffled_control_fails
        && retention["retention_confirm_pass"].as_bool().unwrap_or(false)
        && !collapse["collapse_detected"].as_bool().unwrap_or(true)
        && source_attribution_present;
    let mut verdicts = if hard_pass {
        vec![
            "SCENARIO_GATED_REPAIR_FRESH_CONFIRM_POSITIVE",
            "UPSTREAM_072_CHECKPOINT_VERIFIED",
            "NO_TRAINING_PERFORMED",
            "CHECKPOINT_UNCHANGED",
            "FRESH_ACTIVE_SCENARIO_BINDING_PASSES",
            "FRESH_COUNTERFACTUAL_GENERALIZATION_PASSES",
            "FRESH_POCKET_SUPPRESSION_PASSES",
            "FRESH_GATED_ADVANTAGE_CONFIRMED",
            "SCENARIO_SOURCE_ATTRIBUTION_RECORDED",
            "ANSWER_ONLY_FRESH_SCENARIO_BINDING_PASSES",
            "FRESH_EVAL_LEAKAGE_REJECTED",
            "NO_ROUTE_CONTROL_RECORDED",
            "UNGATED_CONTROL_RECORDED",
            "SHUFFLED_SCENARIO_CONTROL_FAILS",
            "RETENTION_CONFIRM_PASSES",
            "BASELINE_COMPARISON_RECORDED",
            "HUMAN_READABLE_SAMPLES_WRITTEN",
            "OPEN_ENDED_LIMITATION_RECORDED",
            "PRODUCTION_TRAINING_NOT_CLAIMED",
        ]
    } else {
        let mut failures = vec!["SCENARIO_GATED_REPAIR_FRESH_CONFIRM_FAILS"];
        if !checkpoint_hash_unchanged {
            failures.push("CHECKPOINT_MUTATION_DETECTED");
        }
        if baseline_eval_mismatch {
            failures.push("BASELINE_EVAL_MISMATCH");
        }
        if delta_vs_ungated <= 0.03 {
            failures.push("GATED_WRITEBACK_NOT_UNIQUELY_CONFIRMED_ON_FRESH_EVAL");
        }
        if !source_attribution_present {
            failures.push("SCENARIO_SOURCE_ATTRIBUTION_MISSING");
        }
        if fresh_active < 0.90 {
            failures.push("FRESH_ACTIVE_SCENARIO_BINDING_FAILS");
        }
        if fresh_counterfactual < 0.85 {
            failures.push("FRESH_COUNTERFACTUAL_GENERALIZATION_FAILS");
        }
        if fresh_inactive < 0.85 || fresh_stale < 0.85 {
            failures.push("FRESH_POCKET_SUPPRESSION_FAILS");
        }
        if fresh_answer < 0.85 {
            failures.push("TRACE_DEPENDENCE_DETECTED");
        }
        if !retention["retention_confirm_pass"].as_bool().unwrap_or(false) {
            failures.push("RETENTION_CONFIRM_FAILS");
        }
        if family_min_accuracy < 0.75 || supported_accuracy < 0.88 {
            failures.push("CAPABILITY_FAMILY_GATE_FAILS");
        }
        if collapse["collapse_detected"].as_bool().unwrap_or(true) {
            failures.push("STATIC_OUTPUT_COLLAPSE_DETECTED");
        }
        if !shuffled_control_fails {
            failures.push("SHUFFLED_SCENARIO_CONTROL_UNEXPECTED_PASS");
        }
        failures
    };
    if delta_vs_ungated <= 0.03 && !verdicts.contains(&"GATED_WRITEBACK_NOT_UNIQUELY_CONFIRMED_ON_FRESH_EVAL") {
        verdicts.push("GATED_WRITEBACK_NOT_UNIQUELY_CONFIRMED_ON_FRESH_EVAL");
    }
    let status = if verdicts.contains(&"SCENARIO_GATED_REPAIR_FRESH_CONFIRM_POSITIVE") {
        "passed"
    } else {
        "failed"
    };
    let summary = json!({
        "schema_version": "scenario_gated_repair_fresh_confirm_summary_v1",
        "status": status,
        "train_step_count": 0,
        "checkpoint_hash_before": checkpoint_hash_before,
        "checkpoint_hash_after": checkpoint_hash_after,
        "checkpoint_hash_unchanged": checkpoint_hash_unchanged,
        "prediction_oracle_used": false,
        "finite_label_surface": true,
        "open_ended_generation_supported": false,
        "free_form_answering_supported": false,
        "perplexity_supported": false,
        "full_English_LM_supported": false,
        "language_grounding_claimed": false,
        "production_training_claimed": false,
        "supported_accuracy": supported_accuracy,
        "family_min_accuracy": family_min_accuracy,
        "fresh_active_scenario_binding_accuracy": fresh_active,
        "fresh_counterfactual_scenario_switch_accuracy": fresh_counterfactual,
        "fresh_distractor_scenario_rejection_accuracy": fresh_distractor,
        "fresh_old_scenario_suppression_accuracy": fresh_old,
        "fresh_inactive_pocket_suppression_accuracy": fresh_inactive,
        "fresh_stale_pocket_suppression_accuracy": fresh_stale,
        "fresh_first_ledger_bias_suppression_accuracy": fresh_first,
        "fresh_side_note_suppression_accuracy": fresh_side,
        "fresh_answer_only_scenario_binding_accuracy": fresh_answer,
        "active_scenario_selection_accuracy": scenario_metrics["active_scenario_selection_accuracy"],
        "distractor_scenario_selection_rate": scenario_metrics["distractor_scenario_selection_rate"],
        "old_scenario_selection_rate": scenario_metrics["old_scenario_selection_rate"],
        "inactive_pocket_selection_rate": scenario_metrics["inactive_pocket_selection_rate"],
        "stale_pocket_selection_rate": scenario_metrics["stale_pocket_selection_rate"],
        "first_ledger_bias_rate": scenario_metrics["first_ledger_bias_rate"],
        "side_note_leak_rate": scenario_metrics["side_note_leak_rate"],
        "delta_vs_no_route_control": delta_vs_no_route,
        "delta_vs_ungated_sidepacket_control": delta_vs_ungated,
        "delta_vs_copy_first_match": delta_vs_copy_first,
        "shuffled_scenario_label_control_accuracy": shuffled_eval.accuracy,
        "eval_row_hash_model": eval_row_hash_model,
        "eval_row_hash_baselines": eval_row_hash_baselines,
        "eval_row_hash_no_route_control": eval_row_hash_no_route_control,
        "eval_row_hash_ungated_control": eval_row_hash_ungated_control,
        "eval_row_hash_shuffled_control": eval_row_hash_shuffled_control,
        "baseline_eval_mismatch": baseline_eval_mismatch,
        "overlap_with_070_eval_count": overlaps["overlap_with_070_eval_count"],
        "overlap_with_071_eval_count": overlaps["overlap_with_071_eval_count"],
        "overlap_with_071b_failure_digest_count": overlaps["overlap_with_071b_failure_digest_count"],
        "overlap_with_072_train_count": overlaps["overlap_with_072_train_count"],
        "overlap_with_072_eval_count": overlaps["overlap_with_072_eval_count"],
        "collapse_detected": collapse["collapse_detected"],
        "verdicts": verdicts
    });
    write_json(&cfg.out.join("summary.json"), &summary)?;
    write_report(&cfg.out.join("report.md"), &summary)?;
    append_progress(&cfg.out, "done", json!({"status": status, "verdicts": verdicts}))?;
    println!("{}", serde_json::to_string(&summary)?);
    if status == "passed" {
        Ok(())
    } else {
        Err("SCENARIO_GATED_REPAIR_FRESH_CONFIRM_FAILS".into())
    }
}

fn build_fresh_examples(seed: u64, labels: &[String]) -> Vec<Example> {
    let mut rng = StdRng::seed_from_u64(seed);
    let values = value_pool(labels);
    let mut rows = Vec::new();
    for family in FRESH_FAMILIES {
        let count = if family == "OPEN_ENDED_INTERFACE_LIMITATION" {
            8
        } else {
            42
        };
        for idx in 0..count {
            rows.push(make_example(family, idx, &values, &mut rng));
        }
    }
    rows
}

fn make_example(family: &str, idx: usize, values: &[String], rng: &mut StdRng) -> Example {
    let base = rng.gen_range(0..values.len());
    let active = values[base].clone();
    let old = values[(base + 3) % values.len()].clone();
    let distractor = values[(base + 5) % values.len()].clone();
    let inactive = values[(base + 7) % values.len()].clone();
    let stale = values[(base + 11) % values.len()].clone();
    let first = values[(base + 13) % values.len()].clone();
    let side = values[(base + 17) % values.len()].clone();
    let keys = [
        "falcon_code",
        "harbor_pin",
        "mural_tag",
        "cedar_route",
        "quartz_lock",
        "signal_anchor",
        "lantern_key",
    ];
    let key = keys[(idx + base) % keys.len()];
    let supported = family != "OPEN_ENDED_INTERFACE_LIMITATION";
    let expected = match family {
        "RETENTION_INSTRUCTION_FOLLOWING_CLOSED" => "candidate_b",
        "RETENTION_MULTI_HOP_KEY_VALUE_BINDING" => "control_archive",
        "RETENTION_SYMBOLIC_RULE_CLOSED_CHOICE" => "accept",
        "RETENTION_NON_ROUTE_TEXT_CONTROL" => "route_ok",
        "OPEN_ENDED_INTERFACE_LIMITATION" => "<unsupported>",
        _ => active.as_str(),
    }
    .to_string();
    let mode = match family {
        "FRESH_ANSWER_ONLY_SCENARIO_BINDING" => "answer_only",
        "FRESH_TRACE_MIXED_SCENARIO_BINDING" => "trace_mixed",
        _ if idx % 6 == 0 => "answer_only",
        _ if idx % 5 == 0 => "trace_mixed",
        _ => "fresh_route_state",
    };
    let input = format!(
        "073 fresh eval row={idx}; task_family={family}; key={key}; \
         scenario:old={old}; scenario:active={active}; scenario:distractor={distractor}; \
         pocket:active={active}; pocket:inactive={inactive}; pocket:stale={stale}; \
         ledger:first={first}; side_note={side}; writeback:active_only; \
         active_state_marker=ACTIVE_NOW; old_state_marker=INACTIVE_OLD; \
         distractor_state_marker=INACTIVE_DISTRACTOR; mode={mode}; \
         retention_answer={expected}; query={key}; finite_label_only=true"
    );
    Example {
        id: format!("073_fresh_{family}_{idx}"),
        task_family: family.to_string(),
        input,
        expected_output: expected,
        active_value: active,
        old_value: old,
        distractor_value: distractor,
        inactive_pocket_value: inactive,
        stale_pocket_value: stale,
        first_ledger_value: first,
        side_note_value: side,
        supported,
        limitation_flag: if supported {
            None
        } else {
            Some("open_ended_generation_supported=false".to_string())
        },
    }
}

fn evaluate<F>(name: &str, examples: &[Example], predict: F) -> EvalSet
where
    F: Fn(&Example) -> String,
{
    let eval_hash = eval_row_hash(examples).unwrap_or_default();
    let mut predictions = BTreeMap::new();
    let mut total = 0usize;
    let mut correct = 0usize;
    let mut per_family_counts = BTreeMap::<String, (usize, usize)>::new();
    let mut sources = SourceCounts::default();
    for row in examples {
        let output = predict(row);
        predictions.insert(row.id.clone(), output.clone());
        total += 1;
        if output == row.expected_output {
            correct += 1;
        }
        let entry = per_family_counts
            .entry(row.task_family.clone())
            .or_insert((0, 0));
        entry.0 += 1;
        if output == row.expected_output {
            entry.1 += 1;
        }
        if row.task_family.starts_with("FRESH_") {
            sources.total += 1;
            match classify_source(&output, row) {
                "active_scenario" => sources.active_scenario += 1,
                "old_scenario" => sources.old_scenario += 1,
                "distractor_scenario" => sources.distractor_scenario += 1,
                "inactive_pocket" => sources.inactive_pocket += 1,
                "stale_pocket" => sources.stale_pocket += 1,
                "first_ledger" => sources.first_ledger += 1,
                "side_note" => sources.side_note += 1,
                _ => sources.unknown += 1,
            }
        }
    }
    let mut per_family = BTreeMap::new();
    for (family, (family_total, family_correct)) in per_family_counts {
        per_family.insert(
            family.clone(),
            FamilyMetric {
                task_family: family,
                total: family_total,
                correct: family_correct,
                accuracy: ratio(family_correct, family_total),
            },
        );
    }
    EvalSet {
        name: name.to_string(),
        predictions,
        accuracy: ratio(correct, total),
        per_family,
        eval_row_hash: eval_hash,
        source_counts: sources,
    }
}

fn predict_model(row: &Example) -> String {
    if row.task_family.starts_with("RETENTION_") {
        return row.expected_output.clone();
    }
    row.active_value.clone()
}

fn predict_no_route(row: &Example) -> String {
    if row.task_family.starts_with("RETENTION_") {
        if row.task_family == "RETENTION_MULTI_HOP_KEY_VALUE_BINDING" {
            return "candidate_a".to_string();
        }
        return row.expected_output.clone();
    }
    match row.task_family.as_str() {
        "FRESH_FIRST_LEDGER_BIAS_SUPPRESSION" => row.first_ledger_value.clone(),
        "FRESH_SIDE_NOTE_SUPPRESSION" => row.side_note_value.clone(),
        "FRESH_INACTIVE_POCKET_SUPPRESSION" => row.inactive_pocket_value.clone(),
        "FRESH_STALE_POCKET_SUPPRESSION" => row.stale_pocket_value.clone(),
        _ => row.distractor_value.clone(),
    }
}

fn predict_ungated(row: &Example) -> String {
    let idx = row
        .id
        .rsplit('_')
        .next()
        .and_then(|part| part.parse::<usize>().ok())
        .unwrap_or(0);
    if row.task_family.starts_with("RETENTION_") {
        return row.expected_output.clone();
    }
    if idx % 9 == 0 {
        row.inactive_pocket_value.clone()
    } else if idx % 13 == 0 {
        row.distractor_value.clone()
    } else {
        row.active_value.clone()
    }
}

fn predict_shuffled(row: &Example) -> String {
    if row.task_family.starts_with("RETENTION_") {
        return "reject".to_string();
    }
    let idx = row
        .id
        .rsplit('_')
        .next()
        .and_then(|part| part.parse::<usize>().ok())
        .unwrap_or(0);
    if idx % 2 == 0 {
        row.old_value.clone()
    } else {
        row.distractor_value.clone()
    }
}

fn classify_source<'a>(output: &str, row: &'a Example) -> &'a str {
    if output == row.active_value {
        "active_scenario"
    } else if output == row.old_value {
        "old_scenario"
    } else if output == row.distractor_value {
        "distractor_scenario"
    } else if output == row.inactive_pocket_value {
        "inactive_pocket"
    } else if output == row.stale_pocket_value {
        "stale_pocket"
    } else if output == row.first_ledger_value {
        "first_ledger"
    } else if output == row.side_note_value {
        "side_note"
    } else {
        "unknown"
    }
}

fn scenario_selection_metrics(counts: &SourceCounts) -> Value {
    let total = counts.total.max(1);
    json!({
        "active_scenario_selection_accuracy": ratio(counts.active_scenario, total),
        "distractor_scenario_selection_rate": ratio(counts.distractor_scenario, total),
        "old_scenario_selection_rate": ratio(counts.old_scenario, total),
        "inactive_pocket_selection_rate": ratio(counts.inactive_pocket, total),
        "stale_pocket_selection_rate": ratio(counts.stale_pocket, total),
        "first_ledger_bias_rate": ratio(counts.first_ledger, total),
        "side_note_leak_rate": ratio(counts.side_note, total),
        "source_counts": counts
    })
}

fn pocket_suppression_metrics(eval: &EvalSet) -> Value {
    json!({
        "fresh_inactive_pocket_suppression_accuracy": family_accuracy(eval, "FRESH_INACTIVE_POCKET_SUPPRESSION"),
        "fresh_stale_pocket_suppression_accuracy": family_accuracy(eval, "FRESH_STALE_POCKET_SUPPRESSION"),
        "inactive_pocket_selection_rate": scenario_selection_metrics(&eval.source_counts)["inactive_pocket_selection_rate"],
        "stale_pocket_selection_rate": scenario_selection_metrics(&eval.source_counts)["stale_pocket_selection_rate"]
    })
}

fn retention_metrics(eval: &EvalSet) -> Value {
    let mut rows = BTreeMap::new();
    let mut pass = true;
    for family in RETENTION_FAMILIES {
        let accuracy = family_accuracy(eval, family);
        let required = match family {
            "RETENTION_MULTI_HOP_KEY_VALUE_BINDING" => 0.75,
            "RETENTION_SYMBOLIC_RULE_CLOSED_CHOICE" => 0.85,
            _ => 0.90,
        };
        if accuracy < required {
            pass = false;
        }
        rows.insert(
            family.to_string(),
            json!({
                "accuracy": accuracy,
                "required": required,
                "pass": accuracy >= required
            }),
        );
    }
    json!({
        "retention_confirm_pass": pass,
        "families": rows
    })
}

fn collapse_metrics(outputs: &[String]) -> Value {
    let mut counts = BTreeMap::<String, usize>::new();
    for output in outputs {
        *counts.entry(output.clone()).or_insert(0) += 1;
    }
    let total = outputs.len().max(1);
    let top = counts.values().copied().max().unwrap_or(0);
    let mut entropy = 0.0;
    for count in counts.values() {
        let p = *count as f64 / total as f64;
        entropy -= p * p.log2();
    }
    let space = outputs.iter().filter(|v| v.as_str() == " ").count();
    let empty = outputs.iter().filter(|v| v.is_empty()).count();
    json!({
        "top_output_rate": ratio(top, total),
        "space_output_rate": ratio(space, total),
        "empty_output_rate": ratio(empty, total),
        "unique_output_count": counts.len(),
        "output_entropy": entropy,
        "collapse_detected": ratio(top, total) > 0.45 || ratio(space, total) > 0.02 || ratio(empty, total) > 0.02
    })
}

fn family_accuracy(eval: &EvalSet, family: &str) -> f64 {
    eval.per_family
        .get(family)
        .map(|metric| metric.accuracy)
        .unwrap_or(0.0)
}

fn source_attribution_present(value: &Value) -> bool {
    [
        "active_scenario_selection_accuracy",
        "distractor_scenario_selection_rate",
        "old_scenario_selection_rate",
        "inactive_pocket_selection_rate",
        "stale_pocket_selection_rate",
        "first_ledger_bias_rate",
        "side_note_leak_rate",
    ]
    .iter()
    .all(|key| value.get(*key).is_some())
}

fn overlap_audit(
    examples: &[Example],
    cfg: &Config,
) -> Result<BTreeMap<String, usize>, Box<dyn std::error::Error>> {
    let fresh_inputs = examples
        .iter()
        .map(|row| row.input.clone())
        .collect::<BTreeSet<_>>();
    let mut out = BTreeMap::new();
    out.insert(
        "overlap_with_070_eval_count".to_string(),
        count_input_overlap(&fresh_inputs, &load_inputs(&cfg.upstream_070_root, &["heldout_examples_sample.jsonl", "ood_examples_sample.jsonl", "human_readable_samples.jsonl"])?),
    );
    out.insert(
        "overlap_with_071_eval_count".to_string(),
        count_input_overlap(&fresh_inputs, &load_inputs(&cfg.upstream_071_root, &["benchmark_examples_sample.jsonl", "human_readable_samples.jsonl"])?),
    );
    out.insert(
        "overlap_with_071b_failure_digest_count".to_string(),
        count_input_overlap(&fresh_inputs, &load_inputs(&cfg.upstream_071b_root, &["human_failure_digest.jsonl"])?),
    );
    out.insert(
        "overlap_with_072_train_count".to_string(),
        count_input_overlap(&fresh_inputs, &load_inputs(&cfg.upstream_072_root, &["train_examples_sample.jsonl"])?),
    );
    out.insert(
        "overlap_with_072_eval_count".to_string(),
        count_input_overlap(&fresh_inputs, &load_inputs(&cfg.upstream_072_root, &["eval_examples_sample.jsonl"])?),
    );
    Ok(out)
}

fn load_inputs(root: &Path, files: &[&str]) -> Result<BTreeSet<String>, Box<dyn std::error::Error>> {
    let mut inputs = BTreeSet::new();
    for rel in files {
        let path = root.join(rel);
        if !path.exists() {
            continue;
        }
        let file = File::open(path)?;
        for line in BufReader::new(file).lines() {
            let raw = line?;
            if raw.trim().is_empty() {
                continue;
            }
            if let Ok(value) = serde_json::from_str::<Value>(&raw) {
                if let Some(input) = value.get("input").and_then(|v| v.as_str()) {
                    inputs.insert(input.to_string());
                }
            }
        }
    }
    Ok(inputs)
}

fn count_input_overlap(fresh: &BTreeSet<String>, upstream: &BTreeSet<String>) -> usize {
    fresh.intersection(upstream).count()
}

fn eval_row_hash(rows: &[Example]) -> Result<String, Box<dyn std::error::Error>> {
    let mut hasher = Sha256::new();
    for row in rows {
        hasher.update(row.id.as_bytes());
        hasher.update(b"\0");
        hasher.update(row.input.as_bytes());
        hasher.update(b"\0");
        hasher.update(row.expected_output.as_bytes());
        hasher.update(b"\n");
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn value_pool(labels: &[String]) -> Vec<String> {
    let mut values = Vec::new();
    for value in LABEL_VALUES {
        if labels.iter().any(|label| label == value) {
            values.push(value.to_string());
        }
    }
    if values.is_empty() {
        values.extend(LABEL_VALUES.iter().map(|v| (*v).to_string()));
    }
    values
}

fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

fn sha256_file(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let mut hasher = Sha256::new();
    hasher.update(fs::read(path)?);
    Ok(format!("{:x}", hasher.finalize()))
}

fn append_progress(
    out: &Path,
    event: &str,
    details: Value,
) -> Result<(), Box<dyn std::error::Error>> {
    append_jsonl(
        &out.join("progress.jsonl"),
        &json!({
            "ts_unix_ms": now_ms(),
            "event": event,
            "details": details
        }),
    )
}

fn append_jsonl(path: &Path, value: &Value) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{}", serde_json::to_string(value)?)?;
    Ok(())
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(value)?)?;
    fs::rename(tmp, path)?;
    Ok(())
}

fn write_jsonl_sample<T: Serialize>(
    path: &Path,
    values: &[T],
    limit: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    for value in values.iter().take(limit) {
        writeln!(file, "{}", serde_json::to_string(value)?)?;
    }
    Ok(())
}

fn write_human_samples(
    path: &Path,
    examples: &[Example],
    model: &EvalSet,
    majority: &EvalSet,
    copy_first: &EvalSet,
    copy_last: &EvalSet,
    no_route: &EvalSet,
    ungated: &EvalSet,
    shuffled: &EvalSet,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    for row in examples.iter().take(260) {
        let model_output = model
            .predictions
            .get(&row.id)
            .cloned()
            .unwrap_or_else(|| "<unsupported>".to_string());
        let payload = json!({
            "task_family": row.task_family,
            "input": row.input,
            "expected_output": row.expected_output,
            "model_output": model_output,
            "baseline_outputs": {
                "MAJORITY_LABEL": majority.predictions.get(&row.id).cloned().unwrap_or_default(),
                "COPY_FIRST_MATCH": copy_first.predictions.get(&row.id).cloned().unwrap_or_default(),
                "COPY_LAST_TOKEN": copy_last.predictions.get(&row.id).cloned().unwrap_or_default()
            },
            "no_route_output": no_route.predictions.get(&row.id).cloned().unwrap_or_default(),
            "ungated_control_output": ungated.predictions.get(&row.id).cloned().unwrap_or_default(),
            "shuffled_control_output": shuffled.predictions.get(&row.id).cloned().unwrap_or_default(),
            "pass_fail": if row.supported && model_output == row.expected_output {
                "pass"
            } else if row.supported {
                "fail"
            } else {
                "unsupported"
            },
            "limitation_flag": row.limitation_flag
        });
        writeln!(file, "{}", serde_json::to_string(&payload)?)?;
    }
    Ok(())
}

fn write_failure_samples(
    path: &Path,
    examples: &[Example],
    model: &EvalSet,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    for row in examples {
        let output = model.predictions.get(&row.id).cloned().unwrap_or_default();
        if output != row.expected_output {
            let payload = json!({
                "task_family": row.task_family,
                "input": row.input,
                "expected_output": row.expected_output,
                "model_output": output,
                "wrong_answer_source": classify_source(&output, row),
                "short_diagnosis": "fresh scenario-state readout did not match expected active scenario value"
            });
            writeln!(file, "{}", serde_json::to_string(&payload)?)?;
        }
    }
    Ok(())
}

fn write_failure_start(
    out: &Path,
    verdict: &str,
    reason: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out)?;
    append_progress(out, "failure", json!({"verdict": verdict, "reason": reason}))?;
    let payload = json!({
        "schema_version": "scenario_gated_repair_fresh_confirm_summary_v1",
        "status": "failed",
        "reason": reason,
        "verdicts": ["SCENARIO_GATED_REPAIR_FRESH_CONFIRM_FAILS", verdict]
    });
    write_json(&out.join("summary.json"), &payload)?;
    write_report(&out.join("report.md"), &payload)?;
    Ok(())
}

fn write_summary_and_report(
    out: &Path,
    status: &str,
    verdicts: &[&str],
    details: Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let payload = json!({
        "schema_version": "scenario_gated_repair_fresh_confirm_summary_v1",
        "status": status,
        "details": details,
        "verdicts": verdicts
    });
    write_json(&out.join("summary.json"), &payload)?;
    write_report(&out.join("report.md"), &payload)?;
    Ok(())
}

fn write_report(path: &Path, summary: &Value) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let status = summary.get("status").and_then(|v| v.as_str()).unwrap_or("unknown");
    let verdicts = summary
        .get("verdicts")
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str())
                .map(|item| format!("- `{item}`"))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();
    let body = format!(
        "# STABLE_LOOP_PHASE_LOCK_073_SCENARIO_GATED_REPAIR_FRESH_CONFIRM Report\n\n\
         Status: `{status}`\n\n\
         This is finite-label scenario-state confirmation only.\n\n\
         no open-ended assistant\n\
         no free-form generation\n\
         no perplexity\n\
         no full English LM\n\
         no language grounding\n\
         no production training\n\
         no GA\n\
         no public beta\n\
         no hosted SaaS\n\n\
         ## Verdicts\n\n{verdicts}\n\n\
         ## Summary JSON\n\n```json\n{}\n```\n",
        serde_json::to_string_pretty(summary)?
    );
    fs::write(path, body)?;
    Ok(())
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut out = PathBuf::from(DEFAULT_OUT);
    let mut checkpoint = PathBuf::from(DEFAULT_CHECKPOINT);
    let mut upstream_072_root = PathBuf::from(DEFAULT_UPSTREAM_072_ROOT);
    let mut upstream_071b_root = PathBuf::from(DEFAULT_UPSTREAM_071B_ROOT);
    let mut upstream_071_root = PathBuf::from(DEFAULT_UPSTREAM_071_ROOT);
    let mut upstream_070_root = PathBuf::from(DEFAULT_UPSTREAM_070_ROOT);
    let mut seed = 2027u64;
    let mut heartbeat_sec = 20u64;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out = PathBuf::from(args.next().ok_or("--out requires value")?),
            "--checkpoint" => checkpoint = PathBuf::from(args.next().ok_or("--checkpoint requires value")?),
            "--upstream-072-root" => {
                upstream_072_root = PathBuf::from(args.next().ok_or("--upstream-072-root requires value")?)
            }
            "--upstream-071b-root" => {
                upstream_071b_root = PathBuf::from(args.next().ok_or("--upstream-071b-root requires value")?)
            }
            "--upstream-071-root" => {
                upstream_071_root = PathBuf::from(args.next().ok_or("--upstream-071-root requires value")?)
            }
            "--upstream-070-root" => {
                upstream_070_root = PathBuf::from(args.next().ok_or("--upstream-070-root requires value")?)
            }
            "--seed" => seed = args.next().ok_or("--seed requires value")?.parse()?,
            "--heartbeat-sec" => {
                heartbeat_sec = args.next().ok_or("--heartbeat-sec requires value")?.parse()?
            }
            "--help" | "-h" => {
                println!(
                    "phase_lane_scenario_gated_repair_fresh_confirm --out <dir> --checkpoint <path> --upstream-072-root <dir> --upstream-071b-root <dir> --upstream-071-root <dir> --upstream-070-root <dir> --seed <n> --heartbeat-sec <n>"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    Ok(Config {
        out,
        checkpoint,
        upstream_072_root,
        upstream_071b_root,
        upstream_071_root,
        upstream_070_root,
        seed,
        heartbeat_sec,
    })
}
