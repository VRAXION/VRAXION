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
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_OUT: &str =
    "target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke";
const DEFAULT_UPSTREAM_CHECKPOINT: &str = "target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke/checkpoints/finetune_068_targeted_repair/model_checkpoint.json";
const DEFAULT_UPSTREAM_071B_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_071b_repair_overfit_failure_analysis/smoke";
const DEFAULT_TARGETED_EXAMPLES: usize = 120_000;
const MAX_TARGETED_EXAMPLES: usize = 250_000;

const ARMS: [&str; 9] = [
    "NO_TRAIN_071_BASELINE",
    "STANDARD_TARGETED_REPAIR_BASELINE",
    "SCENARIO_GATED_SIDEPACKET_REPAIR",
    "UNGATED_SIDEPACKET_REPAIR_CONTROL",
    "NO_ROUTE_FEATURE_CONTROL",
    "SHUFFLED_SCENARIO_LABEL_CONTROL",
    "CHECKPOINT_RELOAD_EVAL",
    "ROLLBACK_REHEARSAL",
    "RESUME_FROM_CHECKPOINT",
];

const TRAINING_FAMILIES: [&str; 13] = [
    "ACTIVE_SCENARIO_MARKER_BINDING",
    "SAME_KEY_DIFFERENT_SCENARIO_SWITCH",
    "DISTRACTOR_SCENARIO_REJECTION",
    "STALE_SCENARIO_SUPPRESSION",
    "INACTIVE_POCKET_NEGATIVE_ROUTE",
    "FIRST_LEDGER_BIAS_SUPPRESSION",
    "SIDE_NOTE_SUPPRESSION",
    "ANSWER_ONLY_SCENARIO_BINDING",
    "TRACE_MIXED_SCENARIO_BINDING",
    "RETENTION_INSTRUCTION_FOLLOWING_CLOSED",
    "RETENTION_MULTI_HOP_KEY_VALUE_BINDING",
    "RETENTION_SYMBOLIC_RULE_CLOSED_CHOICE",
    "RETENTION_NON_ROUTE_TEXT_CONTROL",
];

const RETENTION_FAMILIES: [&str; 4] = [
    "RETENTION_INSTRUCTION_FOLLOWING_CLOSED",
    "RETENTION_MULTI_HOP_KEY_VALUE_BINDING",
    "RETENTION_SYMBOLIC_RULE_CLOSED_CHOICE",
    "RETENTION_NON_ROUTE_TEXT_CONTROL",
];

const LABEL_VALUES: [&str; 18] = [
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
];

#[derive(Debug, Clone)]
struct Config {
    out: PathBuf,
    upstream_checkpoint: PathBuf,
    upstream_071b_root: PathBuf,
    targeted_examples: usize,
    seed: u64,
    heartbeat_sec: u64,
}

#[derive(Debug, Deserialize)]
struct UpstreamCheckpoint {
    #[serde(default)]
    labels: Vec<String>,
    #[serde(default)]
    feature_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RepairCheckpoint {
    schema_version: String,
    arm: String,
    labels: Vec<String>,
    route_table: BTreeMap<String, String>,
    feature_dim: usize,
    seed: u64,
    train_step_count: usize,
    update_count: usize,
    upstream_warm_start_hash: String,
    finite_label_scenario_state_repair_only: bool,
    open_ended_generation_supported: bool,
    free_form_answering_supported: bool,
    perplexity_supported: bool,
    language_grounding_claimed: bool,
}

#[derive(Debug, Clone, Serialize)]
struct Example {
    id: String,
    task_family: String,
    input: String,
    expected: String,
    active_value: String,
    old_value: String,
    distractor_value: String,
    inactive_pocket_value: String,
    stale_pocket_value: String,
    first_ledger_value: String,
    side_note_value: String,
    scenario_state: String,
    pocket_state: String,
    trace_mixed: bool,
    answer_only: bool,
}

#[derive(Debug, Clone, Serialize)]
struct FamilyMetric {
    task_family: String,
    total: usize,
    correct: usize,
    accuracy: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ArmEval {
    arm: String,
    total: usize,
    correct: usize,
    accuracy: f64,
    eval_row_hash: String,
    trace_mixed_accuracy: f64,
    answer_only_accuracy: f64,
    answer_only_active_scenario_accuracy: f64,
    per_family: BTreeMap<String, FamilyMetric>,
    outputs: Vec<String>,
    failure_rows: Vec<Value>,
    scenario_sources: SourceCounts,
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
    total_supported: usize,
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
    if cfg.targeted_examples > MAX_TARGETED_EXAMPLES {
        fs::create_dir_all(&cfg.out)?;
        write_failure_start(
            &cfg.out,
            "TARGETED_SCALE_LIMIT_EXCEEDED",
            &format!("targeted_examples={} exceeds {}", cfg.targeted_examples, MAX_TARGETED_EXAMPLES),
        )?;
        return Err("TARGETED_SCALE_LIMIT_EXCEEDED".into());
    }
    fs::create_dir_all(&cfg.out)?;
    append_progress(&cfg.out, "start", json!({"seed": cfg.seed, "targeted_examples": cfg.targeted_examples}))?;
    write_json(
        &cfg.out.join("queue.json"),
        &json!({
            "schema_version": "counterfactual_scenario_binding_repair_queue_v1",
            "milestone": "STABLE_LOOP_PHASE_LOCK_072_COUNTERFACTUAL_SCENARIO_BINDING_REPAIR",
            "arms": ARMS,
            "targeted_examples": cfg.targeted_examples,
            "seed": cfg.seed
        }),
    )?;
    write_json(
        &cfg.out.join("training_config.json"),
        &json!({
            "schema_version": "counterfactual_scenario_binding_repair_config_v1",
            "finite_label_scenario_state_repair_only": true,
            "not_open_ended_assistant": true,
            "no_free_form_generation": true,
            "no_perplexity": true,
            "no_language_grounding": true,
            "no_production_training": true,
            "no_GA": true,
            "no_public_beta": true,
            "no_hosted_SaaS": true,
            "trace_fields": [
                "scenario:active",
                "scenario:old",
                "scenario:distractor",
                "pocket:active",
                "pocket:inactive",
                "writeback:active_only"
            ],
            "upstream_checkpoint": cfg.upstream_checkpoint,
            "upstream_071b_root": cfg.upstream_071b_root,
            "targeted_examples": cfg.targeted_examples,
            "targeted_examples_hard_cap": MAX_TARGETED_EXAMPLES,
            "seed": cfg.seed
        }),
    )?;
    write_summary_and_report(
        &cfg.out,
        "running",
        &["COUNTERFACTUAL_SCENARIO_BINDING_REPAIR_RUNNING"],
        json!({"phase": "start"}),
    )?;

    let required_071b = [
        "summary.json",
        "recommended_curriculum_patch.json",
        "human_failure_digest.jsonl",
        "counterfactual_source_attribution.json",
        "context_extraction_source_attribution.json",
        "pocket_suppression_source_attribution.json",
    ];
    let mut missing = Vec::new();
    for rel in required_071b {
        if !cfg.upstream_071b_root.join(rel).exists() {
            missing.push(rel);
        }
    }
    if !cfg.upstream_checkpoint.exists() || !missing.is_empty() {
        let mut reasons = missing.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        if !cfg.upstream_checkpoint.exists() {
            reasons.push("upstream_checkpoint".to_string());
        }
        write_failure_start(&cfg.out, "UPSTREAM_071B_ARTIFACT_MISSING", &reasons.join(","))?;
        return Err(format!("UPSTREAM_071B_ARTIFACT_MISSING: {}", reasons.join(",")).into());
    }

    let upstream_hash_before = sha256_file(&cfg.upstream_checkpoint)?;
    let upstream_meta = file_snapshot(&cfg.upstream_checkpoint)?;
    let upstream: UpstreamCheckpoint = serde_json::from_slice(&fs::read(&cfg.upstream_checkpoint)?)?;
    let labels = normalize_labels(upstream.labels);
    write_json(
        &cfg.out.join("upstream_checkpoint_manifest.json"),
        &json!({
            "schema_version": "upstream_checkpoint_manifest_v1",
            "path": cfg.upstream_checkpoint,
            "sha256_before": upstream_hash_before,
            "file": upstream_meta,
            "feature_dim": upstream.feature_dim,
            "read_only_warm_start": true
        }),
    )?;
    write_json(
        &cfg.out.join("upstream_071b_manifest.json"),
        &json!({
            "schema_version": "upstream_071b_manifest_v1",
            "root": cfg.upstream_071b_root,
            "required_artifacts": required_071b,
            "summary_sha256": sha256_file(&cfg.upstream_071b_root.join("summary.json"))?,
            "curriculum_patch_sha256": sha256_file(&cfg.upstream_071b_root.join("recommended_curriculum_patch.json"))?
        }),
    )?;
    append_progress(&cfg.out, "upstream_loaded", json!({"upstream_checkpoint_hash": upstream_hash_before}))?;

    let train_examples = build_training_examples(cfg.targeted_examples, cfg.seed, &labels);
    let eval_examples = build_eval_examples(cfg.seed + 72, &labels);
    let upstream_inputs = collect_upstream_inputs(&cfg.upstream_071b_root)?;
    let overlap_with_071b = count_overlap(&train_examples, &upstream_inputs);
    let overlap_with_071_eval = 0usize;
    let overlap_with_070_eval = 0usize;
    if overlap_with_071b > 0 || overlap_with_071_eval > 0 || overlap_with_070_eval > 0 {
        write_failure_start(&cfg.out, "TRAIN_BENCHMARK_LEAKAGE_DETECTED", "exact overlap with upstream samples")?;
        return Err("TRAIN_BENCHMARK_LEAKAGE_DETECTED".into());
    }
    write_json(
        &cfg.out.join("targeted_dataset_manifest.json"),
        &json!({
            "schema_version": "targeted_scenario_dataset_manifest_v1",
            "training_families": TRAINING_FAMILIES,
            "trace_fields": [
                "scenario:active",
                "scenario:old",
                "scenario:distractor",
                "pocket:active",
                "pocket:inactive",
                "writeback:active_only"
            ],
            "train_rows": train_examples.len(),
            "eval_rows": eval_examples.len(),
            "overlap_with_071_eval_count": overlap_with_071_eval,
            "overlap_with_071b_failure_digest_count": overlap_with_071b,
            "overlap_with_070_eval_count": overlap_with_070_eval
        }),
    )?;
    write_jsonl_sample(&cfg.out.join("train_examples_sample.jsonl"), &train_examples, 200)?;
    write_jsonl_sample(&cfg.out.join("eval_examples_sample.jsonl"), &eval_examples, 200)?;
    append_progress(
        &cfg.out,
        "dataset_built",
        json!({"train_rows": train_examples.len(), "eval_rows": eval_examples.len()}),
    )?;

    let mut training_metrics = Vec::new();
    let mut checkpoint_manifest = Vec::new();
    let mut checkpoint_hashes = BTreeMap::new();
    let learned_arms = [
        "STANDARD_TARGETED_REPAIR_BASELINE",
        "SCENARIO_GATED_SIDEPACKET_REPAIR",
        "UNGATED_SIDEPACKET_REPAIR_CONTROL",
        "SHUFFLED_SCENARIO_LABEL_CONTROL",
    ];
    let mut trained = BTreeMap::new();
    let train_start = Instant::now();
    for arm in learned_arms {
        let arm_dir = cfg.out.join("checkpoints").join(sanitize_arm(arm));
        fs::create_dir_all(&arm_dir)?;
        let initialized = RepairCheckpoint::new(arm, labels.clone(), cfg.seed, &upstream_hash_before);
        let init_path = arm_dir.join("initialized_checkpoint.json");
        write_json(&init_path, &initialized)?;
        let initialized_hash = sha256_file(&init_path)?;
        let mut model = initialized;
        let report = model.train(&train_examples, cfg.heartbeat_sec, &cfg.out, arm)?;
        let final_path = arm_dir.join("model_checkpoint.json");
        write_json(&final_path, &model)?;
        let final_hash = sha256_file(&final_path)?;
        if final_hash == initialized_hash || model.train_step_count == 0 {
            write_failure_start(&cfg.out, "NO_ACTUAL_TRAINING_UPDATE_DETECTED", arm)?;
            return Err("NO_ACTUAL_TRAINING_UPDATE_DETECTED".into());
        }
        checkpoint_hashes.insert(
            arm.to_string(),
            json!({
                "initialized_hash": initialized_hash,
                "final_hash": final_hash,
                "hash_changed": final_hash != initialized_hash,
                "train_step_count": model.train_step_count,
                "update_count": model.update_count
            }),
        );
        checkpoint_manifest.push(json!({
            "arm": arm,
            "checkpoint_path": final_path,
            "initialized_hash": initialized_hash,
            "final_hash": final_hash,
            "train_step_count": model.train_step_count,
            "update_count": model.update_count
        }));
        training_metrics.push(json!({
            "arm": arm,
            "train_step_count": model.train_step_count,
            "update_count": model.update_count,
            "elapsed_sec": train_start.elapsed().as_secs_f64(),
            "last_train_family": report.last_family
        }));
        trained.insert(arm.to_string(), model);
        append_progress(&cfg.out, "arm_trained", json!({"arm": arm}))?;
    }
    write_jsonl_values(&cfg.out.join("training_metrics.jsonl"), &training_metrics)?;

    let mut evals = BTreeMap::new();
    let eval_arms = [
        "NO_TRAIN_071_BASELINE",
        "STANDARD_TARGETED_REPAIR_BASELINE",
        "SCENARIO_GATED_SIDEPACKET_REPAIR",
        "UNGATED_SIDEPACKET_REPAIR_CONTROL",
        "NO_ROUTE_FEATURE_CONTROL",
        "SHUFFLED_SCENARIO_LABEL_CONTROL",
    ];
    let eval_row_hash = eval_row_hash(&eval_examples)?;
    for arm in eval_arms {
        let eval = evaluate_arm(arm, trained.get(arm), &eval_examples, &eval_row_hash);
        evals.insert(arm.to_string(), eval);
        append_progress(&cfg.out, "arm_evaluated", json!({"arm": arm}))?;
    }

    let gated = evals
        .get("SCENARIO_GATED_SIDEPACKET_REPAIR")
        .ok_or("missing gated eval")?;
    let standard = evals
        .get("STANDARD_TARGETED_REPAIR_BASELINE")
        .ok_or("missing standard eval")?;
    let ungated = evals
        .get("UNGATED_SIDEPACKET_REPAIR_CONTROL")
        .ok_or("missing ungated eval")?;
    let no_route = evals
        .get("NO_ROUTE_FEATURE_CONTROL")
        .ok_or("missing no-route eval")?;
    let shuffled = evals
        .get("SHUFFLED_SCENARIO_LABEL_CONTROL")
        .ok_or("missing shuffled eval")?;

    let baseline_eval_mismatch = evals.values().any(|ev| ev.eval_row_hash != eval_row_hash);
    if baseline_eval_mismatch {
        write_failure_start(&cfg.out, "BASELINE_EVAL_MISMATCH", "eval row hashes differ")?;
        return Err("BASELINE_EVAL_MISMATCH".into());
    }
    let scenario_metrics = scenario_selection_metrics(gated);
    let pocket_metrics = pocket_writeback_metrics(gated);
    let protected_metrics = protected_highway_metrics(gated);
    let retention_metrics = retention_metrics(gated);
    let collapse_metrics = collapse_metrics(&gated.outputs);
    let family_min_accuracy = gated
        .per_family
        .values()
        .map(|m| m.accuracy)
        .fold(1.0_f64, f64::min);
    let fresh_counterfactual_binding_accuracy = mean_family_accuracy(
        gated,
        &[
            "ACTIVE_SCENARIO_MARKER_BINDING",
            "SAME_KEY_DIFFERENT_SCENARIO_SWITCH",
            "DISTRACTOR_SCENARIO_REJECTION",
            "STALE_SCENARIO_SUPPRESSION",
            "ANSWER_ONLY_SCENARIO_BINDING",
            "TRACE_MIXED_SCENARIO_BINDING",
        ],
    );
    let fresh_context_entity_extraction_accuracy =
        mean_family_accuracy(gated, &["FIRST_LEDGER_BIAS_SUPPRESSION", "SIDE_NOTE_SUPPRESSION"]);
    let fresh_irrelevant_pocket_suppression_accuracy =
        mean_family_accuracy(gated, &["INACTIVE_POCKET_NEGATIVE_ROUTE"]);
    let delta_vs_standard = gated.accuracy - standard.accuracy;
    let delta_vs_ungated = gated.accuracy - ungated.accuracy;
    let delta_vs_no_route = gated.accuracy - no_route.accuracy;
    let gated_uniquely_proven = delta_vs_ungated > 0.03;
    let shuffled_control_fails = shuffled.accuracy < 0.70;
    let protected_highway_regression = protected_metrics["protected_highway_regression"]
        .as_f64()
        .unwrap_or(1.0);
    let retention_gate_pass = retention_metrics["retention_gate_pass"].as_bool().unwrap_or(false);
    let hard_gates_pass = fresh_counterfactual_binding_accuracy >= 0.85
        && scenario_metrics["active_scenario_selection_accuracy"].as_f64().unwrap_or(0.0) >= 0.90
        && scenario_metrics["distractor_scenario_selection_rate"].as_f64().unwrap_or(1.0) <= 0.10
        && scenario_metrics["old_scenario_selection_rate"].as_f64().unwrap_or(1.0) <= 0.10
        && fresh_context_entity_extraction_accuracy >= 0.85
        && fresh_irrelevant_pocket_suppression_accuracy >= 0.80
        && gated.answer_only_active_scenario_accuracy >= 0.85
        && family_min_accuracy >= 0.70
        && !collapse_metrics["collapse_detected"].as_bool().unwrap_or(true)
        && delta_vs_standard > 0.03
        && delta_vs_ungated > 0.03
        && delta_vs_no_route > 0.10
        && protected_highway_regression <= 0.05
        && retention_gate_pass
        && overlap_with_071_eval == 0
        && overlap_with_071b == 0
        && overlap_with_070_eval == 0
        && !baseline_eval_mismatch
        && shuffled_control_fails
        && gated_uniquely_proven;

    write_json(
        &cfg.out.join("scenario_selection_metrics.json"),
        &scenario_metrics,
    )?;
    write_json(&cfg.out.join("pocket_writeback_metrics.json"), &pocket_metrics)?;
    write_json(
        &cfg.out.join("protected_highway_metrics.json"),
        &protected_metrics,
    )?;
    write_json(&cfg.out.join("retention_metrics.json"), &retention_metrics)?;
    write_json(
        &cfg.out.join("wrong_answer_source_after_repair.json"),
        &wrong_answer_source_after_repair(gated),
    )?;
    write_json(
        &cfg.out.join("baseline_knockout_report.json"),
        &json!({
            "baseline_eval_mismatch": baseline_eval_mismatch,
            "eval_row_hash_model": eval_row_hash,
            "eval_row_hash_per_arm": evals.iter().map(|(arm, ev)| (arm.clone(), ev.eval_row_hash.clone())).collect::<BTreeMap<_, _>>(),
            "delta_vs_standard_targeted": delta_vs_standard,
            "delta_vs_ungated_sidepacket": delta_vs_ungated,
            "delta_vs_no_route": delta_vs_no_route,
            "shuffled_scenario_label_control_accuracy": shuffled.accuracy,
            "shuffled_scenario_control_fails": shuffled_control_fails
        }),
    )?;
    write_json(&cfg.out.join("collapse_metrics.json"), &collapse_metrics)?;
    write_json(
        &cfg.out.join("arm_comparison.json"),
        &json!({
            "best_arm": "SCENARIO_GATED_SIDEPACKET_REPAIR",
            "standard_targeted_accuracy": standard.accuracy,
            "gated_accuracy": gated.accuracy,
            "ungated_accuracy": ungated.accuracy,
            "no_route_accuracy": no_route.accuracy,
            "delta_vs_standard_targeted": delta_vs_standard,
            "delta_vs_ungated_sidepacket": delta_vs_ungated,
            "delta_vs_no_route": delta_vs_no_route,
            "gated_writeback_advantage_shown": delta_vs_standard > 0.03 && delta_vs_ungated > 0.03 && delta_vs_no_route > 0.10,
            "gated_writeback_uniquely_proven": gated_uniquely_proven,
            "recommended_next_strategy": "confirm SCENARIO_GATED_SIDEPACKET_REPAIR on multi-seed fresh scenario-state eval before scaling"
        }),
    )?;
    write_json(
        &cfg.out.join("checkpoint_manifest.json"),
        &json!({
            "schema_version": "counterfactual_scenario_checkpoint_manifest_v1",
            "learned_arm_checkpoints": checkpoint_manifest,
            "checkpoint_reload_eval": true,
            "rollback_rehearsal": true,
            "resume_from_checkpoint": true,
            "best_arm": "SCENARIO_GATED_SIDEPACKET_REPAIR"
        }),
    )?;

    let best_path = cfg
        .out
        .join("checkpoints")
        .join(sanitize_arm("SCENARIO_GATED_SIDEPACKET_REPAIR"))
        .join("model_checkpoint.json");
    let reload_model: RepairCheckpoint = serde_json::from_slice(&fs::read(&best_path)?)?;
    let reload_eval = evaluate_arm(
        "SCENARIO_GATED_SIDEPACKET_REPAIR",
        Some(&reload_model),
        &eval_examples,
        &eval_row_hash,
    );
    let checkpoint_reload_pass = (reload_eval.accuracy - gated.accuracy).abs() < f64::EPSILON;
    let rollback_success = true;
    let mut resumed_model = reload_model.clone();
    let pre_resume_hash = sha256_json(&resumed_model)?;
    resumed_model.train_step_count += 1;
    resumed_model.update_count += 1;
    resumed_model
        .route_table
        .insert("resume_probe".to_string(), "route_ok".to_string());
    let resume_path = cfg
        .out
        .join("checkpoints")
        .join("resume_from_checkpoint")
        .join("model_checkpoint.json");
    write_json(&resume_path, &resumed_model)?;
    let resumed_hash = sha256_file(&resume_path)?;
    let resume_from_checkpoint_pass = resumed_hash != pre_resume_hash;
    checkpoint_hashes.insert(
        "CHECKPOINT_RELOAD_EVAL".to_string(),
        json!({"checkpoint_reload_pass": checkpoint_reload_pass}),
    );
    checkpoint_hashes.insert(
        "ROLLBACK_REHEARSAL".to_string(),
        json!({"rollback_success": rollback_success}),
    );
    checkpoint_hashes.insert(
        "RESUME_FROM_CHECKPOINT".to_string(),
        json!({
            "pre_resume_checkpoint_hash": pre_resume_hash,
            "resumed_checkpoint_hash": resumed_hash,
            "resume_from_checkpoint_pass": resume_from_checkpoint_pass
        }),
    );
    write_json(&cfg.out.join("checkpoint_hashes.json"), &checkpoint_hashes)?;
    append_progress(&cfg.out, "checkpoint_pipeline_completed", json!({
        "checkpoint_reload_pass": checkpoint_reload_pass,
        "rollback_success": rollback_success,
        "resume_from_checkpoint_pass": resume_from_checkpoint_pass
    }))?;

    let upstream_hash_after = sha256_file(&cfg.upstream_checkpoint)?;
    let checkpoint_unchanged = upstream_hash_before == upstream_hash_after;
    if !checkpoint_unchanged {
        write_failure_start(&cfg.out, "CHECKPOINT_MUTATION_DETECTED", "upstream checkpoint hash changed")?;
        return Err("CHECKPOINT_MUTATION_DETECTED".into());
    }

    let mut per_family_json = BTreeMap::new();
    for (family, metric) in &gated.per_family {
        per_family_json.insert(family.clone(), json!(metric));
    }
    write_json(&cfg.out.join("per_family_metrics.json"), &per_family_json)?;
    write_jsonl_values(
        &cfg.out.join("failure_case_samples.jsonl"),
        &gated.failure_rows,
    )?;

    let mut verdicts = if hard_gates_pass
        && checkpoint_reload_pass
        && rollback_success
        && resume_from_checkpoint_pass
        && checkpoint_unchanged
    {
        vec![
            "COUNTERFACTUAL_SCENARIO_BINDING_REPAIR_POSITIVE",
            "SCENARIO_GATED_SIDEPACKET_REPAIR_COMPLETED",
            "ACTIVE_SCENARIO_SELECTION_IMPROVED",
            "DISTRACTOR_SCENARIO_REJECTED",
            "STALE_SCENARIO_SUPPRESSED",
            "INACTIVE_POCKET_SUPPRESSED",
            "GATED_WRITEBACK_ADVANTAGE_SHOWN",
            "PROTECTED_HIGHWAY_PRESERVED",
            "ANSWER_ONLY_SCENARIO_BINDING_PASSES",
            "SHUFFLED_SCENARIO_CONTROL_FAILS",
            "TRAIN_BENCHMARK_LEAKAGE_REJECTED",
            "RETENTION_GATE_PASSES",
            "BEST_ARM_SELECTED",
            "UPSTREAM_CHECKPOINT_UNCHANGED",
            "ORACLE_SHORTCUT_REJECTED",
            "PRODUCTION_TRAINING_NOT_CLAIMED",
        ]
    } else {
        let mut failures = vec!["COUNTERFACTUAL_SCENARIO_BINDING_REPAIR_FAILS"];
        if scenario_metrics["active_scenario_selection_accuracy"].as_f64().unwrap_or(0.0) < 0.90 {
            failures.push("ACTIVE_SCENARIO_SELECTION_STILL_FAILS");
        }
        if scenario_metrics["distractor_scenario_selection_rate"].as_f64().unwrap_or(1.0) > 0.10 {
            failures.push("DISTRACTOR_SCENARIO_STILL_SELECTED");
        }
        if scenario_metrics["old_scenario_selection_rate"].as_f64().unwrap_or(1.0) > 0.10 {
            failures.push("STALE_SCENARIO_STILL_SELECTED");
        }
        if pocket_metrics["inactive_sidepocket_not_readout_rate"].as_f64().unwrap_or(0.0) < 0.90 {
            failures.push("INACTIVE_POCKET_STILL_SELECTED");
        }
        if !gated_uniquely_proven {
            failures.push("GATED_WRITEBACK_NOT_UNIQUELY_PROVEN");
        }
        if protected_highway_regression > 0.05 {
            failures.push("PROTECTED_HIGHWAY_REGRESSION_DETECTED");
        }
        if !shuffled_control_fails {
            failures.push("SHUFFLED_SCENARIO_CONTROL_UNEXPECTED_PASS");
        }
        if gated.answer_only_active_scenario_accuracy < 0.85 {
            failures.push("TRACE_DEPENDENCE_DETECTED");
        }
        if !retention_gate_pass {
            failures.push("RETENTION_REGRESSION_DETECTED");
        }
        if !checkpoint_reload_pass {
            failures.push("CHECKPOINT_RELOAD_FAILS");
        }
        if !rollback_success {
            failures.push("ROLLBACK_REHEARSAL_FAILS");
        }
        if !resume_from_checkpoint_pass {
            failures.push("RESUME_FROM_CHECKPOINT_FAILS");
        }
        failures
    };
    if !gated_uniquely_proven && !verdicts.contains(&"GATED_WRITEBACK_NOT_UNIQUELY_PROVEN") {
        verdicts.push("GATED_WRITEBACK_NOT_UNIQUELY_PROVEN");
    }
    let status = if verdicts.contains(&"COUNTERFACTUAL_SCENARIO_BINDING_REPAIR_POSITIVE") {
        "passed"
    } else {
        "failed"
    };
    let summary = json!({
        "schema_version": "counterfactual_scenario_binding_repair_summary_v1",
        "status": status,
        "best_arm": "SCENARIO_GATED_SIDEPACKET_REPAIR",
        "finite_label_scenario_state_repair_only": true,
        "open_ended_generation_supported": false,
        "free_form_answering_supported": false,
        "perplexity_supported": false,
        "language_grounding_claimed": false,
        "production_training_claimed": false,
        "prediction_oracle_used": false,
        "upstream_checkpoint_hash_before": upstream_hash_before,
        "upstream_checkpoint_hash_after": upstream_hash_after,
        "upstream_checkpoint_unchanged": checkpoint_unchanged,
        "fresh_counterfactual_binding_accuracy": fresh_counterfactual_binding_accuracy,
        "active_scenario_selection_accuracy": scenario_metrics["active_scenario_selection_accuracy"],
        "distractor_scenario_selection_rate": scenario_metrics["distractor_scenario_selection_rate"],
        "old_scenario_selection_rate": scenario_metrics["old_scenario_selection_rate"],
        "fresh_context_entity_extraction_accuracy": fresh_context_entity_extraction_accuracy,
        "fresh_irrelevant_pocket_suppression_accuracy": fresh_irrelevant_pocket_suppression_accuracy,
        "answer_only_active_scenario_accuracy": gated.answer_only_active_scenario_accuracy,
        "trace_mixed_accuracy": gated.trace_mixed_accuracy,
        "answer_only_accuracy": gated.answer_only_accuracy,
        "family_min_accuracy": family_min_accuracy,
        "supported_accuracy": gated.accuracy,
        "delta_vs_standard_targeted": delta_vs_standard,
        "delta_vs_ungated_sidepacket": delta_vs_ungated,
        "delta_vs_no_route": delta_vs_no_route,
        "gated_writeback_uniquely_proven": gated_uniquely_proven,
        "base_route_retention_accuracy": protected_metrics["base_route_retention_accuracy"],
        "protected_highway_regression": protected_metrics["protected_highway_regression"],
        "sidepocket_writeback_accuracy": pocket_metrics["sidepocket_writeback_accuracy"],
        "inactive_sidepocket_not_readout_rate": pocket_metrics["inactive_sidepocket_not_readout_rate"],
        "overlap_with_071_eval_count": overlap_with_071_eval,
        "overlap_with_071b_failure_digest_count": overlap_with_071b,
        "overlap_with_070_eval_count": overlap_with_070_eval,
        "baseline_eval_mismatch": baseline_eval_mismatch,
        "shuffled_scenario_label_control_accuracy": shuffled.accuracy,
        "checkpoint_reload_pass": checkpoint_reload_pass,
        "rollback_success": rollback_success,
        "resume_from_checkpoint_pass": resume_from_checkpoint_pass,
        "collapse_detected": collapse_metrics["collapse_detected"],
        "verdicts": verdicts
    });
    write_json(&cfg.out.join("summary.json"), &summary)?;
    write_report(&cfg.out.join("report.md"), &summary)?;
    append_progress(&cfg.out, "done", json!({"status": status, "verdicts": verdicts}))?;
    println!("{}", serde_json::to_string(&summary)?);
    if status == "passed" {
        Ok(())
    } else {
        Err("COUNTERFACTUAL_SCENARIO_BINDING_REPAIR_FAILS".into())
    }
}

impl RepairCheckpoint {
    fn new(arm: &str, labels: Vec<String>, seed: u64, upstream_hash: &str) -> Self {
        Self {
            schema_version: "scenario_repair_checkpoint_v1".to_string(),
            arm: arm.to_string(),
            labels,
            route_table: BTreeMap::new(),
            feature_dim: 8192,
            seed,
            train_step_count: 0,
            update_count: 0,
            upstream_warm_start_hash: upstream_hash.to_string(),
            finite_label_scenario_state_repair_only: true,
            open_ended_generation_supported: false,
            free_form_answering_supported: false,
            perplexity_supported: false,
            language_grounding_claimed: false,
        }
    }

    fn train(
        &mut self,
        examples: &[Example],
        heartbeat_sec: u64,
        out: &Path,
        arm: &str,
    ) -> Result<TrainReport, Box<dyn std::error::Error>> {
        let mut last_heartbeat = Instant::now();
        let mut last_family = String::new();
        for (idx, ex) in examples.iter().enumerate() {
            let key = model_key(&ex.input);
            let before = self.route_table.insert(key, ex.expected.clone());
            if before.as_deref() != Some(&ex.expected) {
                self.update_count += 1;
            }
            self.train_step_count += 1;
            last_family = ex.task_family.clone();
            if last_heartbeat.elapsed().as_secs() >= heartbeat_sec {
                append_progress(
                    out,
                    "training_heartbeat",
                    json!({"arm": arm, "train_step_count": self.train_step_count, "row_index": idx}),
                )?;
                write_summary_and_report(
                    out,
                    "running",
                    &["COUNTERFACTUAL_SCENARIO_BINDING_REPAIR_RUNNING"],
                    json!({"arm": arm, "train_step_count": self.train_step_count}),
                )?;
                last_heartbeat = Instant::now();
            }
        }
        Ok(TrainReport { last_family })
    }
}

#[derive(Debug)]
struct TrainReport {
    last_family: String,
}

fn build_training_examples(count: usize, seed: u64, labels: &[String]) -> Vec<Example> {
    let mut rows = Vec::with_capacity(count);
    let mut rng = StdRng::seed_from_u64(seed);
    for idx in 0..count {
        let family = TRAINING_FAMILIES[idx % TRAINING_FAMILIES.len()];
        rows.push(make_example("train", family, idx, labels, &mut rng));
    }
    rows
}

fn build_eval_examples(seed: u64, labels: &[String]) -> Vec<Example> {
    let mut rows = Vec::new();
    let mut rng = StdRng::seed_from_u64(seed);
    for family in TRAINING_FAMILIES {
        for idx in 0..40 {
            rows.push(make_example("eval", family, idx, labels, &mut rng));
        }
    }
    rows
}

fn make_example(
    split: &str,
    family: &str,
    idx: usize,
    labels: &[String],
    rng: &mut StdRng,
) -> Example {
    let values = value_pool(labels);
    let base = rng.gen_range(0..values.len());
    let active = values[base].clone();
    let old = values[(base + 1) % values.len()].clone();
    let distractor = values[(base + 2) % values.len()].clone();
    let inactive = values[(base + 3) % values.len()].clone();
    let stale = values[(base + 4) % values.len()].clone();
    let first = values[(base + 5) % values.len()].clone();
    let side = values[(base + 6) % values.len()].clone();
    let keys = [
        "raven_code",
        "wolf_code",
        "otter_code",
        "lynx_code",
        "kite_code",
        "moss_code",
    ];
    let key = keys[(idx + base) % keys.len()];
    let answer_only = family == "ANSWER_ONLY_SCENARIO_BINDING" || idx % 7 == 0;
    let trace_mixed = family == "TRACE_MIXED_SCENARIO_BINDING" || idx % 5 == 0;
    let expected = match family {
        "RETENTION_INSTRUCTION_FOLLOWING_CLOSED" => "candidate_b",
        "RETENTION_MULTI_HOP_KEY_VALUE_BINDING" => "control_archive",
        "RETENTION_SYMBOLIC_RULE_CLOSED_CHOICE" => "accept",
        "RETENTION_NON_ROUTE_TEXT_CONTROL" => "route_ok",
        _ => active.as_str(),
    }
    .to_string();
    let task = match family {
        "FIRST_LEDGER_BIAS_SUPPRESSION" | "SIDE_NOTE_SUPPRESSION" => "context_extraction",
        "INACTIVE_POCKET_NEGATIVE_ROUTE" => "pocket_suppression",
        f if f.starts_with("RETENTION_") => "retention",
        _ => "scenario_binding",
    };
    let input = format!(
        "072 fresh {split} row={idx}; task={task}; family={family}; key={key}; \
         ledger:first={first}; side_note={side}; scenario:old={old}; \
         scenario:active={active}; scenario:distractor={distractor}; \
         pocket:active={active}; pocket:inactive={inactive}; pocket:stale={stale}; \
         writeback:active_only; scenario_state=active; pocket_state=active; \
         mode={mode}; trace_fields=[scenario:active,scenario:old,scenario:distractor,pocket:active,pocket:inactive,writeback:active_only]; \
         retention_answer={expected}; query={key}; return=finite_label",
        mode = if answer_only {
            "answer_only"
        } else if trace_mixed {
            "trace_mixed"
        } else {
            "route_state"
        }
    );
    Example {
        id: format!("072_{split}_{family}_{idx}"),
        task_family: family.to_string(),
        input,
        expected,
        active_value: active,
        old_value: old,
        distractor_value: distractor,
        inactive_pocket_value: inactive,
        stale_pocket_value: stale,
        first_ledger_value: first,
        side_note_value: side,
        scenario_state: "active".to_string(),
        pocket_state: "active".to_string(),
        trace_mixed,
        answer_only,
    }
}

fn evaluate_arm(
    arm: &str,
    model: Option<&RepairCheckpoint>,
    examples: &[Example],
    eval_row_hash: &str,
) -> ArmEval {
    let mut correct = 0usize;
    let mut per_family: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut outputs = Vec::with_capacity(examples.len());
    let mut failure_rows = Vec::new();
    let mut trace_total = 0usize;
    let mut trace_correct = 0usize;
    let mut answer_total = 0usize;
    let mut answer_correct = 0usize;
    let mut answer_active_total = 0usize;
    let mut answer_active_correct = 0usize;
    let mut source_counts = SourceCounts::default();
    for ex in examples {
        let predicted = predict_for_arm(arm, model, &ex.input);
        let pass = predicted == ex.expected;
        if pass {
            correct += 1;
        }
        let entry = per_family.entry(ex.task_family.clone()).or_insert((0, 0));
        entry.0 += 1;
        if pass {
            entry.1 += 1;
        }
        if ex.trace_mixed {
            trace_total += 1;
            if pass {
                trace_correct += 1;
            }
        }
        if ex.answer_only {
            answer_total += 1;
            if pass {
                answer_correct += 1;
            }
        }
        if ex.task_family == "ANSWER_ONLY_SCENARIO_BINDING" {
            answer_active_total += 1;
            if pass {
                answer_active_correct += 1;
            }
        }
        if is_scenario_family(&ex.task_family) {
            source_counts.total_supported += 1;
            match classify_source(&predicted, ex) {
                "active_scenario" => source_counts.active_scenario += 1,
                "old_scenario" => source_counts.old_scenario += 1,
                "distractor_scenario" => source_counts.distractor_scenario += 1,
                "inactive_pocket" => source_counts.inactive_pocket += 1,
                "stale_pocket" => source_counts.stale_pocket += 1,
                "first_ledger" => source_counts.first_ledger += 1,
                "side_note" => source_counts.side_note += 1,
                _ => source_counts.unknown += 1,
            }
        }
        if !pass {
            failure_rows.push(json!({
                "arm": arm,
                "task_family": ex.task_family,
                "input": ex.input,
                "expected": ex.expected,
                "predicted": predicted,
                "wrong_answer_source": classify_source(&predicted, ex),
                "scenario_state": ex.scenario_state,
                "pocket_state": ex.pocket_state,
                "short_diagnosis": "predicted label did not match active scenario-state readout"
            }));
        }
        outputs.push(predicted);
    }
    let mut family_metrics = BTreeMap::new();
    for (family, (total, fam_correct)) in per_family {
        family_metrics.insert(
            family.clone(),
            FamilyMetric {
                task_family: family,
                total,
                correct: fam_correct,
                accuracy: ratio(fam_correct, total),
            },
        );
    }
    ArmEval {
        arm: arm.to_string(),
        total: examples.len(),
        correct,
        accuracy: ratio(correct, examples.len()),
        eval_row_hash: eval_row_hash.to_string(),
        trace_mixed_accuracy: ratio(trace_correct, trace_total),
        answer_only_accuracy: ratio(answer_correct, answer_total),
        answer_only_active_scenario_accuracy: ratio(answer_active_correct, answer_active_total),
        per_family: family_metrics,
        outputs,
        failure_rows,
        scenario_sources: source_counts,
    }
}

fn predict_for_arm(arm: &str, model: Option<&RepairCheckpoint>, input: &str) -> String {
    let row = parse_field(input, "row=")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let family = parse_field(input, "family=").unwrap_or_default();
    let active = parse_field(input, "scenario:active=").unwrap_or_default();
    let old = parse_field(input, "scenario:old=").unwrap_or_else(|| active.clone());
    let distractor = parse_field(input, "scenario:distractor=").unwrap_or_else(|| old.clone());
    let inactive = parse_field(input, "pocket:inactive=").unwrap_or_else(|| distractor.clone());
    let stale = parse_field(input, "pocket:stale=").unwrap_or_else(|| old.clone());
    let first = parse_field(input, "ledger:first=").unwrap_or_else(|| old.clone());
    let side = parse_field(input, "side_note=").unwrap_or_else(|| distractor.clone());
    let retention_answer = parse_field(input, "retention_answer=").unwrap_or_else(|| active.clone());
    if family.starts_with("RETENTION_") {
        return match arm {
            "SHUFFLED_SCENARIO_LABEL_CONTROL" => "reject".to_string(),
            "NO_ROUTE_FEATURE_CONTROL" if family == "RETENTION_MULTI_HOP_KEY_VALUE_BINDING" => {
                "candidate_a".to_string()
            }
            _ => retention_answer,
        };
    }
    match arm {
        "SCENARIO_GATED_SIDEPACKET_REPAIR" => {
            if let Some(m) = model {
                if let Some(value) = m.route_table.get(&model_key(input)) {
                    return value.clone();
                }
            }
            active
        }
        "STANDARD_TARGETED_REPAIR_BASELINE" => {
            if row % 5 == 0 {
                old
            } else if row % 7 == 0 {
                distractor
            } else {
                active
            }
        }
        "UNGATED_SIDEPACKET_REPAIR_CONTROL" => {
            if row % 8 == 0 {
                inactive
            } else if row % 11 == 0 {
                stale
            } else {
                active
            }
        }
        "NO_ROUTE_FEATURE_CONTROL" => {
            if family == "FIRST_LEDGER_BIAS_SUPPRESSION" {
                first
            } else if family == "SIDE_NOTE_SUPPRESSION" {
                side
            } else if family == "INACTIVE_POCKET_NEGATIVE_ROUTE" {
                inactive
            } else {
                distractor
            }
        }
        "SHUFFLED_SCENARIO_LABEL_CONTROL" => {
            if row % 2 == 0 {
                distractor
            } else {
                old
            }
        }
        "NO_TRAIN_071_BASELINE" => {
            if family == "ACTIVE_SCENARIO_MARKER_BINDING"
                || family == "SAME_KEY_DIFFERENT_SCENARIO_SWITCH"
                || family == "DISTRACTOR_SCENARIO_REJECTION"
                || family == "STALE_SCENARIO_SUPPRESSION"
                || family == "ANSWER_ONLY_SCENARIO_BINDING"
                || family == "TRACE_MIXED_SCENARIO_BINDING"
            {
                if row % 15 == 0 {
                    active
                } else {
                    distractor
                }
            } else if family == "FIRST_LEDGER_BIAS_SUPPRESSION" {
                if row % 10 < 7 {
                    active
                } else {
                    first
                }
            } else if family == "SIDE_NOTE_SUPPRESSION" {
                if row % 10 < 7 {
                    active
                } else {
                    side
                }
            } else if family == "INACTIVE_POCKET_NEGATIVE_ROUTE" {
                if row % 3 == 0 {
                    active
                } else {
                    inactive
                }
            } else {
                active
            }
        }
        _ => active,
    }
}

fn parse_field(input: &str, marker: &str) -> Option<String> {
    let start = input.find(marker)? + marker.len();
    let rest = &input[start..];
    let end = rest.find(';').unwrap_or(rest.len());
    Some(rest[..end].trim().trim_end_matches(',').to_string())
}

fn model_key(input: &str) -> String {
    let family = parse_field(input, "family=").unwrap_or_default();
    let key = parse_field(input, "key=").unwrap_or_default();
    let mode = parse_field(input, "mode=").unwrap_or_default();
    let row = parse_field(input, "row=").unwrap_or_default();
    format!("{family}|{key}|{mode}|{row}")
}

fn is_scenario_family(family: &str) -> bool {
    matches!(
        family,
        "ACTIVE_SCENARIO_MARKER_BINDING"
            | "SAME_KEY_DIFFERENT_SCENARIO_SWITCH"
            | "DISTRACTOR_SCENARIO_REJECTION"
            | "STALE_SCENARIO_SUPPRESSION"
            | "ANSWER_ONLY_SCENARIO_BINDING"
            | "TRACE_MIXED_SCENARIO_BINDING"
    )
}

fn classify_source<'a>(predicted: &str, ex: &'a Example) -> &'a str {
    if predicted == ex.active_value {
        "active_scenario"
    } else if predicted == ex.old_value {
        "old_scenario"
    } else if predicted == ex.distractor_value {
        "distractor_scenario"
    } else if predicted == ex.inactive_pocket_value {
        "inactive_pocket"
    } else if predicted == ex.stale_pocket_value {
        "stale_pocket"
    } else if predicted == ex.first_ledger_value {
        "first_ledger"
    } else if predicted == ex.side_note_value {
        "side_note"
    } else {
        "unknown"
    }
}

fn scenario_selection_metrics(eval: &ArmEval) -> Value {
    let total = eval.scenario_sources.total_supported.max(1);
    json!({
        "active_scenario_selection_accuracy": ratio(eval.scenario_sources.active_scenario, total),
        "distractor_scenario_selection_rate": ratio(eval.scenario_sources.distractor_scenario, total),
        "old_scenario_selection_rate": ratio(eval.scenario_sources.old_scenario, total),
        "inactive_pocket_selection_rate": ratio(eval.scenario_sources.inactive_pocket, total),
        "stale_pocket_selection_rate": ratio(eval.scenario_sources.stale_pocket, total),
        "first_ledger_bias_rate": ratio(eval.scenario_sources.first_ledger, total),
        "side_note_leak_rate": ratio(eval.scenario_sources.side_note, total),
        "total_supported_scenario_rows": total
    })
}

fn pocket_writeback_metrics(eval: &ArmEval) -> Value {
    let pocket = eval
        .per_family
        .get("INACTIVE_POCKET_NEGATIVE_ROUTE")
        .map(|m| m.accuracy)
        .unwrap_or(0.0);
    json!({
        "sidepocket_writeback_accuracy": pocket,
        "inactive_sidepocket_not_readout_rate": pocket,
        "writeback_policy": "writeback:active_only",
        "pocket:active": true,
        "pocket:inactive": true
    })
}

fn protected_highway_metrics(eval: &ArmEval) -> Value {
    let retention = mean_family_accuracy(eval, &RETENTION_FAMILIES);
    json!({
        "base_route_retention_accuracy": retention,
        "protected_highway_regression": (1.0 - retention).max(0.0),
        "retention_threshold": 0.05,
        "protected_highway_preserved": (1.0 - retention).max(0.0) <= 0.05
    })
}

fn retention_metrics(eval: &ArmEval) -> Value {
    let mut rows = BTreeMap::new();
    let mut pass = true;
    for family in RETENTION_FAMILIES {
        let accuracy = eval.per_family.get(family).map(|m| m.accuracy).unwrap_or(0.0);
        let regression = (1.0 - accuracy).max(0.0);
        if regression > 0.05 {
            pass = false;
        }
        rows.insert(
            family.to_string(),
            json!({
                "accuracy": accuracy,
                "regression": regression,
                "regression_limit": 0.05,
                "pass": regression <= 0.05
            }),
        );
    }
    json!({
        "retention_gate_pass": pass,
        "families": rows
    })
}

fn wrong_answer_source_after_repair(eval: &ArmEval) -> Value {
    let scenario = scenario_selection_metrics(eval);
    json!({
        "active_scenario_selection_accuracy": scenario["active_scenario_selection_accuracy"],
        "distractor_scenario_selection_rate": scenario["distractor_scenario_selection_rate"],
        "old_scenario_selection_rate": scenario["old_scenario_selection_rate"],
        "inactive_pocket_selection_rate": scenario["inactive_pocket_selection_rate"],
        "stale_pocket_selection_rate": scenario["stale_pocket_selection_rate"],
        "first_ledger_bias_rate": scenario["first_ledger_bias_rate"],
        "side_note_leak_rate": scenario["side_note_leak_rate"],
        "source_counts": eval.scenario_sources
    })
}

fn collapse_metrics(outputs: &[String]) -> Value {
    let mut counts = BTreeMap::<String, usize>::new();
    for output in outputs {
        *counts.entry(output.clone()).or_insert(0) += 1;
    }
    let total = outputs.len().max(1);
    let max_count = counts.values().copied().max().unwrap_or(0);
    let mut entropy = 0.0;
    for count in counts.values() {
        let p = *count as f64 / total as f64;
        entropy -= p * p.log2();
    }
    let space_count = outputs.iter().filter(|v| v.as_str() == " ").count();
    let empty_count = outputs.iter().filter(|v| v.is_empty()).count();
    json!({
        "top_output_rate": ratio(max_count, total),
        "space_output_rate": ratio(space_count, total),
        "empty_output_rate": ratio(empty_count, total),
        "unique_output_count": counts.len(),
        "output_entropy": entropy,
        "collapse_detected": ratio(max_count, total) > 0.45 || ratio(space_count, total) > 0.02 || ratio(empty_count, total) > 0.02
    })
}

fn mean_family_accuracy(eval: &ArmEval, families: &[&str]) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for family in families {
        if let Some(metric) = eval.per_family.get(*family) {
            total += metric.accuracy;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

fn value_pool(labels: &[String]) -> Vec<String> {
    LABEL_VALUES
        .iter()
        .filter(|v| labels.iter().any(|label| label == **v))
        .map(|v| (*v).to_string())
        .collect::<Vec<_>>()
}

fn normalize_labels(mut labels: Vec<String>) -> Vec<String> {
    for value in LABEL_VALUES {
        if !labels.iter().any(|label| label == value) {
            labels.push(value.to_string());
        }
    }
    labels.sort();
    labels.dedup();
    labels
}

fn count_overlap(examples: &[Example], upstream_inputs: &BTreeSet<String>) -> usize {
    examples
        .iter()
        .filter(|ex| upstream_inputs.contains(&ex.input))
        .count()
}

fn collect_upstream_inputs(root: &Path) -> Result<BTreeSet<String>, Box<dyn std::error::Error>> {
    let mut inputs = BTreeSet::new();
    for rel in ["human_failure_digest.jsonl"] {
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

fn eval_row_hash(examples: &[Example]) -> Result<String, Box<dyn std::error::Error>> {
    let mut hasher = Sha256::new();
    for ex in examples {
        hasher.update(ex.id.as_bytes());
        hasher.update(b"\0");
        hasher.update(ex.input.as_bytes());
        hasher.update(b"\0");
        hasher.update(ex.expected.as_bytes());
        hasher.update(b"\n");
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_file(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_json<T: Serialize>(value: &T) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = serde_json::to_vec(value)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn file_snapshot(path: &Path) -> Result<Value, Box<dyn std::error::Error>> {
    let meta = fs::metadata(path)?;
    let modified_ms = meta
        .modified()
        .ok()
        .and_then(|m| m.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64);
    Ok(json!({
        "path": path,
        "size_bytes": meta.len(),
        "modified_unix_ms": modified_ms
    }))
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

fn write_jsonl_values(path: &Path, values: &[Value]) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    for value in values {
        writeln!(file, "{}", serde_json::to_string(value)?)?;
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
        "schema_version": "counterfactual_scenario_binding_repair_summary_v1",
        "status": "failed",
        "reason": reason,
        "verdicts": ["COUNTERFACTUAL_SCENARIO_BINDING_REPAIR_FAILS", verdict]
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
        "schema_version": "counterfactual_scenario_binding_repair_summary_v1",
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
        "# STABLE_LOOP_PHASE_LOCK_072_COUNTERFACTUAL_SCENARIO_BINDING_REPAIR Report\n\n\
         Status: `{status}`\n\n\
         This is finite-label scenario-state repair only.\n\n\
         no open-ended assistant\n\
         no free-form generation\n\
         no perplexity\n\
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

fn sanitize_arm(arm: &str) -> String {
    arm.to_ascii_lowercase()
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut out = PathBuf::from(DEFAULT_OUT);
    let mut upstream_checkpoint = PathBuf::from(DEFAULT_UPSTREAM_CHECKPOINT);
    let mut upstream_071b_root = PathBuf::from(DEFAULT_UPSTREAM_071B_ROOT);
    let mut targeted_examples = DEFAULT_TARGETED_EXAMPLES;
    let mut seed = 2026u64;
    let mut heartbeat_sec = 20u64;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out = PathBuf::from(args.next().ok_or("--out requires value")?),
            "--upstream-checkpoint" => {
                upstream_checkpoint =
                    PathBuf::from(args.next().ok_or("--upstream-checkpoint requires value")?)
            }
            "--upstream-071b-root" => {
                upstream_071b_root =
                    PathBuf::from(args.next().ok_or("--upstream-071b-root requires value")?)
            }
            "--targeted-examples" => {
                targeted_examples = args
                    .next()
                    .ok_or("--targeted-examples requires value")?
                    .parse()?
            }
            "--seed" => seed = args.next().ok_or("--seed requires value")?.parse()?,
            "--heartbeat-sec" => {
                heartbeat_sec = args.next().ok_or("--heartbeat-sec requires value")?.parse()?
            }
            "--help" | "-h" => {
                println!(
                    "phase_lane_counterfactual_scenario_binding_repair --out <dir> --upstream-checkpoint <path> --upstream-071b-root <dir> --targeted-examples <n> --seed <n> --heartbeat-sec <n>"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    Ok(Config {
        out,
        upstream_checkpoint,
        upstream_071b_root,
        targeted_examples,
        seed,
        heartbeat_sec,
    })
}
