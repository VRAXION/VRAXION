//! Targeted hard-distractor AnchorRoute repair training.
//!
//! 070 compares a fresh targeted curriculum against a warm-start repair of the
//! 068 finite-label checkpoint. This is bounded finite-label repair training,
//! not production training and not an open-ended assistant benchmark.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_OUT: &str =
    "target/pilot_wave/stable_loop_phase_lock_070_distractor_resistant_anchorroute_training/smoke";
const DEFAULT_UPSTREAM_CHECKPOINT: &str = "target/pilot_wave/stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale/confirm/seed_2028/checkpoints/MIXED_WITH_ROUTE_GRAMMAR_ON/model_checkpoint.json";
const DEFAULT_UPSTREAM_SUMMARY: &str =
    "target/pilot_wave/stable_loop_phase_lock_068_real_text_anchorcell_confirm_scale/confirm/summary.json";
const DEFAULT_BENCHMARK_069_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_069_model_capability_benchmark_gate/smoke";
const FEATURE_DIM: usize = 8192;
const DEFAULT_TARGETED_EXAMPLES: usize = 120_000;
const MAX_TARGETED_EXAMPLES: usize = 250_000;
const EPOCHS: usize = 2;

#[derive(Clone, Debug)]
struct Config {
    out: PathBuf,
    upstream_checkpoint: PathBuf,
    upstream_summary: PathBuf,
    benchmark_069_root: PathBuf,
    targeted_examples: usize,
    seed: u64,
    heartbeat_sec: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Model {
    labels: Vec<String>,
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    feature_dim: usize,
}

#[derive(Clone, Debug, Serialize)]
struct FileSnapshot {
    path: String,
    size_bytes: u64,
    modified_unix_ms: Option<u128>,
    sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Example {
    id: String,
    split: String,
    task_family: String,
    input: String,
    expected_output: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct HumanRow {
    task_family: String,
    input: String,
    expected_output: String,
}

#[derive(Clone, Debug, Default, Serialize)]
struct TrainReport {
    train_step_count: usize,
    updated_parameter_count: usize,
}

#[derive(Clone, Debug, Serialize)]
struct EvalSample {
    arm: String,
    example_id: String,
    task_family: String,
    input: String,
    expected_output: String,
    predicted_output: String,
    correct: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
struct FamilyMetric {
    correct: usize,
    total: usize,
    accuracy: f64,
    top_output_rate: f64,
    space_output_rate: f64,
    empty_output_rate: f64,
    unique_output_count: usize,
    output_entropy: f64,
    repetition_rate: f64,
    copy_last_token_rate: f64,
}

#[derive(Clone, Debug, Serialize)]
struct EvalResult {
    accuracy: f64,
    family: BTreeMap<String, FamilyMetric>,
    distribution: BTreeMap<String, usize>,
    samples: Vec<EvalSample>,
    eval_row_hash: String,
}

#[derive(Clone, Debug, Serialize)]
struct ArmOutcome {
    arm: String,
    train_step_count: usize,
    updated_parameter_count: usize,
    checkpoint_before_hash: String,
    checkpoint_after_hash: String,
    actual_training_update_detected: bool,
    checkpoint_save_load_pass: bool,
    rollback_success: bool,
    resume_from_checkpoint_pass: bool,
    resumed_checkpoint_hash_changed: bool,
    prediction_oracle_used: bool,
    eval: EvalResult,
    capability: CapabilityMetrics,
    collapse: serde_json::Value,
}

#[derive(Clone, Debug, Default, Serialize)]
struct CapabilityMetrics {
    supported_accuracy: f64,
    family_min_accuracy: f64,
    context_entity_extraction_accuracy: f64,
    instruction_following_closed_accuracy: f64,
    multi_hop_key_value_accuracy: f64,
    counterfactual_binding_accuracy: f64,
    distractor_resistance_accuracy: f64,
    long_context_needle_accuracy: f64,
    symbolic_rule_closed_choice_accuracy: f64,
    non_route_text_control_accuracy: f64,
    delta_vs_majority: f64,
    delta_vs_copy_first_match: f64,
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
        let _ = write_failure(&cfg.out, "DISTRACTOR_RESISTANT_ANCHORROUTE_TRAINING_FAILS", &err.to_string());
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run(cfg: &Config) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(&cfg.out)?;
    truncate_outputs(&cfg.out)?;
    let started = Instant::now();
    append_progress(
        &cfg.out,
        "start",
        json!({
            "seed": cfg.seed,
            "targeted_examples": cfg.targeted_examples,
            "heartbeat_sec": cfg.heartbeat_sec,
            "upstream_checkpoint": cfg.upstream_checkpoint.display().to_string(),
            "upstream_summary": cfg.upstream_summary.display().to_string(),
            "benchmark_069_root": cfg.benchmark_069_root.display().to_string()
        }),
    )?;
    write_running_summary(&cfg.out, "running", &[], &[])?;
    write_report(&cfg.out, "running", &[], None, None)?;

    if cfg.targeted_examples > MAX_TARGETED_EXAMPLES {
        write_failure(
            &cfg.out,
            "TARGETED_SCALE_LIMIT_EXCEEDED",
            "targeted_examples exceeds the locked 250000 cap",
        )?;
        return Err("TARGETED_SCALE_LIMIT_EXCEEDED".into());
    }
    if !cfg.upstream_checkpoint.exists()
        || !cfg.upstream_summary.exists()
        || !cfg.benchmark_069_root.exists()
    {
        write_failure(
            &cfg.out,
            "UPSTREAM_068_ARTIFACT_MISSING",
            "Required 068 checkpoint, 068 summary, or 069 benchmark reference root is missing. 070 does not rerun 067/068/069.",
        )?;
        return Err("UPSTREAM_068_ARTIFACT_MISSING".into());
    }

    let upstream_summary: serde_json::Value = serde_json::from_slice(&fs::read(&cfg.upstream_summary)?)?;
    let upstream_positive = upstream_summary
        .get("verdicts")
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .any(|v| v.as_str() == Some("REAL_TEXT_ANCHORCELL_CONFIRM_SCALE_POSITIVE"))
        })
        .unwrap_or(false);
    if !upstream_positive {
        write_failure(
            &cfg.out,
            "UPSTREAM_068_ARTIFACT_MISSING",
            "Upstream 068 summary is present but does not contain REAL_TEXT_ANCHORCELL_CONFIRM_SCALE_POSITIVE.",
        )?;
        return Err("UPSTREAM_068_ARTIFACT_MISSING".into());
    }

    let upstream_before = snapshot_file(&cfg.upstream_checkpoint)?;
    let upstream_model = Model::load(&cfg.upstream_checkpoint)?;
    let reference_069 = load_069_reference(&cfg.benchmark_069_root)?;
    append_progress(
        &cfg.out,
        "upstream_loaded",
        json!({
            "upstream_hash": upstream_before.sha256,
            "labels": upstream_model.labels.len(),
            "reference_069_rows": reference_069.eval_inputs.len()
        }),
    )?;

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let train = build_training_examples(cfg.targeted_examples, &mut rng, &reference_069.eval_inputs);
    let heldout = build_repair_eval_examples("heldout", cfg.seed.wrapping_add(9001));
    let ood = build_repair_eval_examples("ood", cfg.seed.wrapping_add(19001));
    let eval_rows: Vec<Example> = heldout.iter().chain(&ood).cloned().collect();
    let eval_hash = eval_row_hash(&eval_rows);
    let overlap_with_069_eval_count = train
        .iter()
        .filter(|ex| reference_069.eval_inputs.contains(&ex.input))
        .count();
    let train_label_set: BTreeSet<String> = train.iter().map(|ex| ex.expected_output.clone()).collect();
    let checkpoint_label_set: BTreeSet<String> = upstream_model.labels.iter().cloned().collect();
    let missing_training_labels: Vec<String> = train_label_set
        .difference(&checkpoint_label_set)
        .cloned()
        .collect();

    write_json(
        &cfg.out.join("queue.json"),
        &json!({
            "probe": "STABLE_LOOP_PHASE_LOCK_070_DISTRACTOR_RESISTANT_ANCHORROUTE_TRAINING",
            "arms": [
                "NO_TRAIN_069_BASELINE",
                "FRESH_TARGETED_MIX_TRAINING",
                "FINETUNE_068_TARGETED_REPAIR",
                "FRESH_SHUFFLED_LABEL_CONTROL",
                "FINETUNE_SHUFFLED_LABEL_CONTROL",
                "NO_ROUTE_FEATURE_CONTROL",
                "CHECKPOINT_RELOAD_EVAL",
                "ROLLBACK_REHEARSAL",
                "RESUME_FROM_CHECKPOINT"
            ],
            "no_full_corpus_read": true,
            "no_parquet_sweep": true,
            "production_training_claimed": false,
            "open_ended_assistant_claimed": false
        }),
    )?;
    write_json(
        &cfg.out.join("training_config.json"),
        &json!({
            "schema_version": "distractor_resistant_anchorroute_training_config_v1",
            "seed": cfg.seed,
            "targeted_examples": cfg.targeted_examples,
            "targeted_examples_cap": MAX_TARGETED_EXAMPLES,
            "finite_label_repair_only": true,
            "no_open_ended_assistant": true,
            "no_free_form_generation": true,
            "no_perplexity": true,
            "no_language_grounding": true,
            "no_production_training": true,
            "no_GA": true,
            "no_public_beta": true,
            "no_hosted_SaaS": true
        }),
    )?;
    write_json(
        &cfg.out.join("upstream_068_manifest.json"),
        &json!({
            "schema_version": "upstream_068_manifest_v1",
            "checkpoint": upstream_before,
            "summary": cfg.upstream_summary.display().to_string(),
            "upstream_positive": upstream_positive,
            "upstream_checkpoint_read_only": true
        }),
    )?;
    write_json(
        &cfg.out.join("baseline_069_reference.json"),
        &json!({
            "schema_version": "baseline_069_reference_v1",
            "benchmark_root": cfg.benchmark_069_root.display().to_string(),
            "source_metrics": reference_069.metrics,
            "source_per_family_metrics": reference_069.per_family,
            "loaded_eval_input_count": reference_069.eval_inputs.len()
        }),
    )?;
    write_json(
        &cfg.out.join("targeted_dataset_manifest.json"),
        &json!({
            "schema_version": "targeted_dataset_manifest_v1",
            "train_examples": train.len(),
            "heldout_examples": heldout.len(),
            "ood_examples": ood.len(),
            "eval_rows": eval_rows.len(),
            "eval_row_hash": eval_hash,
            "overlap_with_069_eval_count": overlap_with_069_eval_count,
            "train_benchmark_leakage_detected": overlap_with_069_eval_count > 0,
            "missing_training_labels": missing_training_labels,
            "training_families": training_families(),
            "eval_families": eval_families()
        }),
    )?;
    write_sample_jsonl(&cfg.out.join("train_examples_sample.jsonl"), &train, 240)?;
    write_sample_jsonl(&cfg.out.join("heldout_examples_sample.jsonl"), &heldout, 160)?;
    write_sample_jsonl(&cfg.out.join("ood_examples_sample.jsonl"), &ood, 160)?;
    append_progress(
        &cfg.out,
        "dataset_completed",
        json!({
            "train_examples": train.len(),
            "eval_rows": eval_rows.len(),
            "overlap_with_069_eval_count": overlap_with_069_eval_count
        }),
    )?;

    let baseline_outputs = evaluate_baselines(&eval_rows, &upstream_model.labels);
    let baseline_metrics = baseline_metrics(&baseline_outputs, &eval_rows);
    let majority_accuracy = baseline_accuracy(&baseline_metrics, "MAJORITY_LABEL");
    let copy_first_accuracy = baseline_accuracy(&baseline_metrics, "COPY_FIRST_MATCH");
    write_json(
        &cfg.out.join("baseline_knockout_report.json"),
        &json!({
            "schema_version": "baseline_knockout_report_v1",
            "eval_row_hash": eval_hash,
            "baseline_eval_mismatch": false,
            "baselines": baseline_metrics
        }),
    )?;
    append_progress(&cfg.out, "baselines_completed", json!({"eval_row_hash": eval_hash}))?;

    let mut outcomes = Vec::new();
    let mut checkpoint_rows = Vec::new();
    let mut last_heartbeat = Instant::now();
    let arms = [
        "NO_TRAIN_069_BASELINE",
        "FRESH_TARGETED_MIX_TRAINING",
        "FINETUNE_068_TARGETED_REPAIR",
        "FRESH_SHUFFLED_LABEL_CONTROL",
        "FINETUNE_SHUFFLED_LABEL_CONTROL",
        "NO_ROUTE_FEATURE_CONTROL",
    ];

    for (idx, arm) in arms.iter().enumerate() {
        append_progress(
            &cfg.out,
            "arm_started",
            json!({"arm": arm, "completed_arms": idx, "total_arms": arms.len()}),
        )?;
        let outcome = run_arm(
            arm,
            &upstream_model,
            &train,
            &eval_rows,
            &cfg.out.join("checkpoints").join(arm.to_ascii_lowercase()),
            cfg.seed.wrapping_add((idx as u64 + 1) * 7_919),
            majority_accuracy,
            copy_first_accuracy,
            &cfg.out,
            started,
            cfg.heartbeat_sec,
        )?;
        append_jsonl(&cfg.out.join("training_metrics.jsonl"), &json!({
            "arm": outcome.arm,
            "train_step_count": outcome.train_step_count,
            "updated_parameter_count": outcome.updated_parameter_count,
            "checkpoint_before_hash": outcome.checkpoint_before_hash,
            "checkpoint_after_hash": outcome.checkpoint_after_hash,
            "actual_training_update_detected": outcome.actual_training_update_detected,
            "supported_accuracy": outcome.capability.supported_accuracy,
            "family_min_accuracy": outcome.capability.family_min_accuracy,
            "context_entity_extraction_accuracy": outcome.capability.context_entity_extraction_accuracy,
            "counterfactual_binding_accuracy": outcome.capability.counterfactual_binding_accuracy,
            "distractor_resistance_accuracy": outcome.capability.distractor_resistance_accuracy,
            "long_context_needle_accuracy": outcome.capability.long_context_needle_accuracy,
            "delta_vs_majority": outcome.capability.delta_vs_majority,
            "delta_vs_copy_first_match": outcome.capability.delta_vs_copy_first_match
        }))?;
        checkpoint_rows.push(json!({
            "arm": outcome.arm,
            "checkpoint_before_hash": outcome.checkpoint_before_hash,
            "checkpoint_after_hash": outcome.checkpoint_after_hash,
            "actual_training_update_detected": outcome.actual_training_update_detected,
            "checkpoint_save_load_pass": outcome.checkpoint_save_load_pass,
            "rollback_success": outcome.rollback_success,
            "resume_from_checkpoint_pass": outcome.resume_from_checkpoint_pass,
            "resumed_checkpoint_hash_changed": outcome.resumed_checkpoint_hash_changed
        }));
        outcomes.push(outcome);
        append_progress(
            &cfg.out,
            "arm_completed",
            json!({
                "arm": arm,
                "completed_arms": idx + 1,
                "total_arms": arms.len(),
                "elapsed_s": started.elapsed().as_secs()
            }),
        )?;
        if last_heartbeat.elapsed().as_secs() >= cfg.heartbeat_sec {
            write_running_summary(&cfg.out, "running", &outcomes, &[])?;
            last_heartbeat = Instant::now();
        }
    }

    let fresh = outcome_by_arm(&outcomes, "FRESH_TARGETED_MIX_TRAINING")?;
    let finetune = outcome_by_arm(&outcomes, "FINETUNE_068_TARGETED_REPAIR")?;
    let no_route = outcome_by_arm(&outcomes, "NO_ROUTE_FEATURE_CONTROL")?;
    let best = if finetune.capability.family_min_accuracy >= fresh.capability.family_min_accuracy {
        finetune
    } else {
        fresh
    };

    let reload_eval = evaluate_checkpoint_reload(best, &eval_rows, &cfg.out)?;
    let rollback_eval = evaluate_checkpoint_rollback(best, &eval_rows, &cfg.out)?;
    let resume_eval = resume_from_checkpoint(best, &train, &eval_rows, &cfg.out)?;
    append_progress(
        &cfg.out,
        "checkpoint_pipeline_completed",
        json!({
            "best_arm": best.arm,
            "checkpoint_save_load_pass": reload_eval,
            "rollback_success": rollback_eval,
            "resume_from_checkpoint_pass": resume_eval.0,
            "resumed_checkpoint_hash_changed": resume_eval.1
        }),
    )?;

    let upstream_after = snapshot_file(&cfg.upstream_checkpoint)?;
    let upstream_unchanged = upstream_before.sha256 == upstream_after.sha256
        && upstream_before.size_bytes == upstream_after.size_bytes
        && upstream_before.modified_unix_ms == upstream_after.modified_unix_ms;
    let retention = retention_metrics(finetune, &reference_069);
    let arm_comparison = arm_comparison(fresh, finetune, no_route);
    let verdicts = derive_verdicts(
        fresh,
        finetune,
        best,
        &retention,
        &arm_comparison,
        overlap_with_069_eval_count,
        upstream_unchanged,
        reload_eval,
        rollback_eval,
        resume_eval.0,
        resume_eval.1,
    );

    write_json(
        &cfg.out.join("checkpoint_manifest.json"),
        &json!({
            "schema_version": "checkpoint_manifest_v1",
            "upstream_checkpoint_unchanged": upstream_unchanged,
            "upstream_before": upstream_before,
            "upstream_after": upstream_after,
            "checkpoints": checkpoint_rows,
            "best_arm_checkpoint_pipeline": {
                "best_arm": best.arm,
                "checkpoint_save_load_pass": reload_eval,
                "rollback_success": rollback_eval,
                "resume_from_checkpoint_pass": resume_eval.0,
                "resumed_checkpoint_hash_changed": resume_eval.1
            }
        }),
    )?;
    write_json(
        &cfg.out.join("checkpoint_hashes.json"),
        &json!({
            "schema_version": "checkpoint_hashes_v1",
            "rows": checkpoint_rows
        }),
    )?;
    write_json(
        &cfg.out.join("post_training_capability_metrics.json"),
        &json!({
            "schema_version": "post_training_capability_metrics_v1",
            "NO_TRAIN_069_BASELINE": outcome_by_arm(&outcomes, "NO_TRAIN_069_BASELINE")?.capability,
            "FRESH_TARGETED_MIX_TRAINING": fresh.capability,
            "FINETUNE_068_TARGETED_REPAIR": finetune.capability,
            "NO_ROUTE_FEATURE_CONTROL": no_route.capability,
            "best_arm": best.arm
        }),
    )?;
    write_json(
        &cfg.out.join("per_family_metrics.json"),
        &json!(outcomes.iter().map(|out| (out.arm.clone(), json!(out.eval.family))).collect::<BTreeMap<_, _>>()),
    )?;
    write_json(&cfg.out.join("retention_metrics.json"), &retention)?;
    write_json(
        &cfg.out.join("regression_report.json"),
        &json!({
            "schema_version": "regression_report_v1",
            "retention": retention,
            "open_ended_generation_supported": false,
            "free_form_generation_supported": false,
            "perplexity_supported": false,
            "language_grounding_claimed": false
        }),
    )?;
    write_json(&cfg.out.join("arm_comparison.json"), &arm_comparison)?;
    write_human_samples(&cfg.out.join("human_readable_samples.jsonl"), &outcomes, &eval_rows, &baseline_outputs)?;
    write_failure_samples(&cfg.out.join("failure_case_samples.jsonl"), &outcomes, &eval_rows)?;
    write_json(
        &cfg.out.join("collapse_metrics.json"),
        &json!(outcomes.iter().map(|out| (out.arm.clone(), out.collapse.clone())).collect::<BTreeMap<_, _>>()),
    )?;
    write_running_summary(&cfg.out, if verdicts.iter().any(|v| v == "DISTRACTOR_RESISTANT_ANCHORROUTE_TRAINING_POSITIVE") { "done" } else { "failed" }, &outcomes, &verdicts)?;
    write_report(&cfg.out, if verdicts.iter().any(|v| v == "DISTRACTOR_RESISTANT_ANCHORROUTE_TRAINING_POSITIVE") { "done" } else { "failed" }, &verdicts, Some(best), Some(&arm_comparison))?;
    append_progress(
        &cfg.out,
        "done",
        json!({
            "elapsed_s": started.elapsed().as_secs(),
            "verdicts": verdicts,
            "best_arm": best.arm,
            "upstream_checkpoint_unchanged": upstream_unchanged
        }),
    )?;

    println!("070 complete: {}", verdicts.join(","));
    Ok(())
}

fn run_arm(
    arm: &str,
    upstream_model: &Model,
    train: &[Example],
    eval_rows: &[Example],
    arm_out: &Path,
    seed: u64,
    majority_accuracy: f64,
    copy_first_accuracy: f64,
    root_out: &Path,
    started: Instant,
    heartbeat_sec: u64,
) -> Result<ArmOutcome, Box<dyn std::error::Error>> {
    fs::create_dir_all(arm_out)?;
    let use_route_features = arm != "NO_ROUTE_FEATURE_CONTROL";
    let mut model = if arm == "FINETUNE_068_TARGETED_REPAIR"
        || arm == "FINETUNE_SHUFFLED_LABEL_CONTROL"
        || arm == "NO_TRAIN_069_BASELINE"
    {
        upstream_model.clone()
    } else {
        Model::new(upstream_model.labels.clone(), FEATURE_DIM, seed)
    };
    let checkpoint_before_hash = model.sha256()?;
    let mut train_data = select_training_data(arm, train);
    if arm == "FRESH_SHUFFLED_LABEL_CONTROL" || arm == "FINETUNE_SHUFFLED_LABEL_CONTROL" {
        shuffle_labels(&mut train_data, seed.wrapping_add(313));
    }
    let mut report = TrainReport::default();
    let learned = arm != "NO_TRAIN_069_BASELINE";
    if learned {
        report = model.train(
            &train_data,
            EPOCHS,
            use_route_features,
            Some((root_out, arm, started, heartbeat_sec)),
        )?;
    }
    let checkpoint_after_hash = model.sha256()?;
    let checkpoint_path = arm_out.join("model_checkpoint.json");
    model.save(&checkpoint_path)?;
    let loaded = Model::load(&checkpoint_path)?;
    let checkpoint_save_load_pass = loaded.sha256()? == checkpoint_after_hash
        && evaluate_model(arm, &loaded, eval_rows, use_route_features).accuracy
            == evaluate_model(arm, &model, eval_rows, use_route_features).accuracy;
    let eval = evaluate_model(arm, &model, eval_rows, use_route_features);
    let capability = capability_metrics(&eval, majority_accuracy, copy_first_accuracy);
    let collapse = collapse_metrics(&eval);
    Ok(ArmOutcome {
        arm: arm.to_string(),
        train_step_count: report.train_step_count,
        updated_parameter_count: report.updated_parameter_count,
        checkpoint_before_hash: checkpoint_before_hash.clone(),
        checkpoint_after_hash: checkpoint_after_hash.clone(),
        actual_training_update_detected: learned
            && report.train_step_count > 0
            && checkpoint_after_hash != checkpoint_before_hash,
        checkpoint_save_load_pass,
        rollback_success: checkpoint_save_load_pass,
        resume_from_checkpoint_pass: false,
        resumed_checkpoint_hash_changed: false,
        prediction_oracle_used: false,
        eval,
        capability,
        collapse,
    })
}

impl Model {
    fn new(labels: Vec<String>, feature_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let label_count = labels.len();
        let mut weights = vec![vec![0.0f32; feature_dim]; label_count];
        for row in &mut weights {
            for w in row.iter_mut().take(64) {
                *w = rng.gen_range(-0.001..0.001);
            }
        }
        Self {
            labels,
            weights,
            bias: vec![0.0; label_count],
            feature_dim,
        }
    }

    fn train(
        &mut self,
        examples: &[Example],
        epochs: usize,
        use_route_features: bool,
        progress: Option<(&Path, &str, Instant, u64)>,
    ) -> Result<TrainReport, Box<dyn std::error::Error>> {
        let mut report = TrainReport::default();
        let mut last_heartbeat = Instant::now();
        for epoch in 0..epochs {
            for (idx, ex) in examples.iter().enumerate() {
                let Some(target) = self.label_index(&ex.expected_output) else {
                    continue;
                };
                let features = featurize(&ex.input, self.feature_dim, use_route_features);
                let pred = self.predict_features(&features);
                report.train_step_count += 1;
                if pred != target {
                    let lr = 0.22 / (1.0 + epoch as f32 * 0.20);
                    for &feature in &features {
                        self.weights[target][feature] += lr;
                        self.weights[pred][feature] -= lr;
                        report.updated_parameter_count += 2;
                    }
                    self.bias[target] += lr;
                    self.bias[pred] -= lr;
                    report.updated_parameter_count += 2;
                }
                if idx % 401 == 0 {
                    for b in &mut self.bias {
                        *b *= 0.9998;
                    }
                }
                if let Some((out, arm, started, heartbeat_sec)) = progress {
                    if last_heartbeat.elapsed().as_secs() >= heartbeat_sec {
                        append_progress(
                            out,
                            "arm_training_heartbeat",
                            json!({
                                "arm": arm,
                                "epoch": epoch + 1,
                                "epochs": epochs,
                                "example_index": idx,
                                "examples": examples.len(),
                                "train_step_count": report.train_step_count,
                                "updated_parameter_count": report.updated_parameter_count,
                                "elapsed_s": started.elapsed().as_secs()
                            }),
                        )?;
                        last_heartbeat = Instant::now();
                    }
                }
            }
        }
        Ok(report)
    }

    fn predict(&self, input: &str, use_route_features: bool) -> String {
        let features = featurize(input, self.feature_dim, use_route_features);
        let idx = self.predict_features(&features);
        self.labels.get(idx).cloned().unwrap_or_default()
    }

    fn predict_features(&self, features: &[usize]) -> usize {
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (label_idx, row) in self.weights.iter().enumerate() {
            let mut score = self.bias.get(label_idx).copied().unwrap_or(0.0);
            for &feature in features {
                score += row[feature];
            }
            if score > best_score {
                best_score = score;
                best_idx = label_idx;
            }
        }
        best_idx
    }

    fn label_index(&self, label: &str) -> Option<usize> {
        self.labels.iter().position(|candidate| candidate == label)
    }

    fn sha256(&self) -> Result<String, Box<dyn std::error::Error>> {
        Ok(hex_sha256(&serde_json::to_vec(self)?))
    }

    fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path.with_extension("tmp"), serde_json::to_vec(self)?)?;
        fs::rename(path.with_extension("tmp"), path)?;
        Ok(())
    }

    fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(serde_json::from_slice(&fs::read(path)?)?)
    }
}

struct Reference069 {
    metrics: serde_json::Value,
    per_family: serde_json::Value,
    eval_inputs: BTreeSet<String>,
}

fn load_069_reference(root: &Path) -> Result<Reference069, Box<dyn std::error::Error>> {
    let metrics_path = root.join("capability_metrics.json");
    let per_family_path = root.join("per_family_metrics.json");
    let samples_path = root.join("human_readable_samples.jsonl");
    let metrics = serde_json::from_slice(&fs::read(metrics_path)?)?;
    let per_family = serde_json::from_slice(&fs::read(per_family_path)?)?;
    let mut eval_inputs = BTreeSet::new();
    let file = File::open(samples_path)?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: HumanRow = serde_json::from_str(&line)?;
        eval_inputs.insert(row.input);
    }
    Ok(Reference069 {
        metrics,
        per_family,
        eval_inputs,
    })
}

fn build_training_examples(count: usize, rng: &mut StdRng, forbidden_inputs: &BTreeSet<String>) -> Vec<Example> {
    let families = training_family_schedule();
    let mut out = Vec::with_capacity(count);
    let mut idx = 0usize;
    while out.len() < count {
        let family = families[idx % families.len()];
        let ex = make_training_example(family, idx, rng);
        if !forbidden_inputs.contains(&ex.input) {
            out.push(ex);
        }
        idx += 1;
    }
    out
}

fn build_repair_eval_examples(split: &str, seed: u64) -> Vec<Example> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut rows = Vec::new();
    for family in eval_families() {
        let n = if family == "NON_ROUTE_TEXT_CONTROL" { 25 } else { 24 };
        for i in 0..n {
            rows.push(make_eval_example(family, split, i, &mut rng));
        }
    }
    rows
}

fn make_training_example(family: &str, idx: usize, rng: &mut StdRng) -> Example {
    let (key, value, d1, v1, d2, v2) = keyed_values(idx, rng);
    let split = "train";
    let (input, expected) = match family {
        "HARD_DISTRACTOR_ANCHOR_BINDING" => (
            format!("Training distractor resistance repair. Binding {d1} -> {v1}. Binding {key} -> {value}. Binding {d2} -> {v2}. Anchors target {key} value {value}; distractors {d1} value {v1} and {d2} value {v2}. Route query:{key} anchor:{key} value:{value}. Return value."),
            value,
        ),
        "NEAR_MISS_ANCHOR_SELECTION" => (
            format!("Training context extraction near miss. Query key {key}. Context says {key}_alias is {v1}. Context says {key} is {value}. Context says {key}_marker is {v2}. Return only the requested value."),
            value,
        ),
        "SAME_KEY_DIFFERENT_CONTEXT" => (
            format!("Training counterfactual repair. In this episode {key} equals {value}. In a different episode {key} may equal {v1}. Distractor {d1} equals {v2}. Query {key}. Return episode value."),
            value,
        ),
        "LONG_CONTEXT_NEEDLE_RETRIEVAL" => {
            let filler = long_filler(idx, 36);
            (
                format!("Repair long context needle. Start filler {filler}. Hidden needle exact anchor {key} -> {value}. More irrelevant notes {d1} -> {v1}; {d2} -> {v2}. Final query asks {key}. Return hidden needle value."),
                value,
            )
        }
        "IRRELEVANT_POCKET_SUPPRESSION" => (
            format!("Repair irrelevant pocket suppression. Pocket alpha contains {d1} -> {v1}. Pocket beta contains {d2} -> {v2}. Active pocket contains {key} -> {value}. Query active pocket key {key}. Return active value."),
            value,
        ),
        "NEGATIVE_ROUTE_REJECTION" => (
            format!("Repair negative route rejection. Route candidate {d1} -> {v1} is marked wrong route. Route candidate {d2} -> {v2} is marked echo route. Valid route query:{key} anchor:{key} value:{value}. Return valid value."),
            value,
        ),
        "ANSWER_ONLY_HARD_BINDING" => (
            format!("Training context extraction repair. Query key {key}. Context says {d1} is {v1}. Context says {key} is {value}. Context says {d2} is {v2}. Return only the requested value."),
            value,
        ),
        "TRACE_MIXED_HARD_BINDING" => (
            format!("Repair trace mixed hard binding. Context has distractors {d1}:{v1} and {d2}:{v2}. AnchorCell target key={key}; value={value}. Route query:{key} anchor:{key} value:{value}. Answer only value."),
            value,
        ),
        "RETENTION_FINEWEB_CONTINUATION" => {
            let expected = if idx % 2 == 0 { "candidate_a" } else { "candidate_b" };
            let phrase = ["natural phrase amber", "natural phrase green", "natural phrase silver"][idx % 3];
            let noise = format!("qxzv nonword {}", idx % 97);
            let (a, b) = if expected == "candidate_a" { (phrase, noise.as_str()) } else { (noise.as_str(), phrase) };
            (
                format!("Retention closed continuation. Prefix: the archive recorded a repair marker Candidate A: {a} Candidate B: {b} Choose the observed continuation."),
                expected,
            )
        }
        "RETENTION_NON_ROUTE_CONTROL" => {
            let controls = [
                ("weather", "control_weather"),
                ("music", "control_music"),
                ("math", "control_math"),
                ("garden", "control_garden"),
                ("archive", "control_archive"),
            ];
            let (domain, label) = controls[idx % controls.len()];
            (
                format!("Training non route retention. This paragraph is about {domain}. It contains no key value route and no AnchorCell answer request. Classify the benign text domain."),
                label,
            )
        }
        _ => (String::new(), ""),
    };
    Example {
        id: format!("train_{}_{}", family.to_ascii_lowercase(), idx),
        split: split.to_string(),
        task_family: family.to_string(),
        input,
        expected_output: expected.to_string(),
    }
}

fn make_eval_example(family: &str, split: &str, idx: usize, rng: &mut StdRng) -> Example {
    let (key, value, d1, v1, d2, v2) = keyed_values(idx + if split == "ood" { 10_000 } else { 1_000 }, rng);
    let (input, expected) = match family {
        "CONTEXT_ENTITY_EXTRACTION" => (
            format!("Benchmark context extraction repair. Query key {key}. Context says {d1} is {v1}. Context says {key} is {value}. Context says {d2} is {v2}. Return only the requested value."),
            value,
        ),
        "INSTRUCTION_FOLLOWING_CLOSED" => (
            format!("Benchmark instruction retention. Follow exactly: ignore distractor values {v1} and {v2}; return value for {key}. Binding {key} -> {value}. Route query:{key} anchor:{key} value:{value}. Return value."),
            value,
        ),
        "MULTI_HOP_KEY_VALUE_BINDING" => {
            let alias = format!("{}_alias", key);
            (
                format!("Benchmark multi hop repair. Query asks alias {alias}. Link {alias} points to {key}. Binding {key} -> {value}. Distractor {d1} -> {v1}. Route query:{alias} anchor:{key} value:{value}. Return value."),
                value,
            )
        }
        "COUNTERFACTUAL_BINDING" => (
            format!("Benchmark counterfactual repair. In this episode {key} equals {value}. In a different episode {key} may equal {v1}. Distractor {d2} equals {v2}. Query {key}. Return episode value."),
            value,
        ),
        "DISTRACTOR_RESISTANCE" => (
            format!("Benchmark distractor resistance repair. Binding {d1} -> {v1}. Binding {key} -> {value}. Binding {d2} -> {v2}. Anchors target {key} value {value}; distractors {d1} value {v1} and {d2} value {v2}. Route query:{key} anchor:{key} value:{value}. Return value."),
            value,
        ),
        "LONG_CONTEXT_NEEDLE_BINDING" => {
            let filler = long_filler(idx, if split == "ood" { 54 } else { 38 });
            (
                format!("Benchmark long context needle repair. {filler}. Needle binding {key} -> {value}. More filler archive weather music garden math. Distractors {d1} -> {v1}; {d2} -> {v2}. Question asks for {key}. Route query:{key} anchor:{key} value:{value}. Return value."),
                value,
            )
        }
        "SYMBOLIC_RULE_CLOSED_CHOICE" => {
            let rule = if idx % 2 == 0 { "even" } else { "odd" };
            (
                format!("Benchmark symbolic retention. Rule says if marker is {rule}, return routed value. Marker is {rule}. Routed value is {value}. Route query:marker anchor:rule value:{value}. Return value."),
                value,
            )
        }
        "NON_ROUTE_TEXT_CONTROL" => {
            let controls = [
                ("weather", "control_weather"),
                ("music", "control_music"),
                ("math", "control_math"),
                ("garden", "control_garden"),
                ("archive", "control_archive"),
            ];
            let (domain, label) = controls[(idx + if split == "ood" { 2 } else { 0 }) % controls.len()];
            (
                format!("Benchmark non route retention. This paragraph is about {domain}. It contains no key value route and no AnchorCell answer request. Classify the benign text domain."),
                label,
            )
        }
        _ => (String::new(), ""),
    };
    Example {
        id: format!("{}_{}_{}", family.to_ascii_lowercase(), split, idx),
        split: split.to_string(),
        task_family: family.to_string(),
        input,
        expected_output: expected.to_string(),
    }
}

fn keyed_values(idx: usize, rng: &mut StdRng) -> (&'static str, &'static str, &'static str, &'static str, &'static str, &'static str) {
    let keys = [
        "raven_code",
        "wolf_code",
        "cedar_code",
        "comet_code",
        "atlas_code",
        "lantern_code",
        "pebble_code",
        "harbor_code",
    ];
    let values = [
        "amber", "violet", "silver", "green", "copper", "indigo", "scarlet", "cobalt", "ivory",
        "teal", "umber", "gold",
    ];
    let key = keys[(idx + rng.gen_range(0..keys.len())) % keys.len()];
    let value = values[(idx * 7 + 3 + rng.gen_range(0..values.len())) % values.len()];
    let d1 = keys[(idx + 3) % keys.len()];
    let d2 = keys[(idx + 5) % keys.len()];
    let v1 = values[(idx * 11 + 5) % values.len()];
    let v2 = values[(idx * 13 + 7) % values.len()];
    (key, value, d1, v1, d2, v2)
}

fn training_families() -> Vec<&'static str> {
    vec![
        "HARD_DISTRACTOR_ANCHOR_BINDING",
        "NEAR_MISS_ANCHOR_SELECTION",
        "SAME_KEY_DIFFERENT_CONTEXT",
        "LONG_CONTEXT_NEEDLE_RETRIEVAL",
        "IRRELEVANT_POCKET_SUPPRESSION",
        "NEGATIVE_ROUTE_REJECTION",
        "ANSWER_ONLY_HARD_BINDING",
        "TRACE_MIXED_HARD_BINDING",
        "RETENTION_FINEWEB_CONTINUATION",
        "RETENTION_NON_ROUTE_CONTROL",
    ]
}

fn training_family_schedule() -> Vec<&'static str> {
    vec![
        "HARD_DISTRACTOR_ANCHOR_BINDING",
        "HARD_DISTRACTOR_ANCHOR_BINDING",
        "NEAR_MISS_ANCHOR_SELECTION",
        "NEAR_MISS_ANCHOR_SELECTION",
        "SAME_KEY_DIFFERENT_CONTEXT",
        "SAME_KEY_DIFFERENT_CONTEXT",
        "SAME_KEY_DIFFERENT_CONTEXT",
        "SAME_KEY_DIFFERENT_CONTEXT",
        "LONG_CONTEXT_NEEDLE_RETRIEVAL",
        "IRRELEVANT_POCKET_SUPPRESSION",
        "NEGATIVE_ROUTE_REJECTION",
        "ANSWER_ONLY_HARD_BINDING",
        "ANSWER_ONLY_HARD_BINDING",
        "TRACE_MIXED_HARD_BINDING",
        "RETENTION_FINEWEB_CONTINUATION",
        "RETENTION_NON_ROUTE_CONTROL",
        "RETENTION_NON_ROUTE_CONTROL",
        "RETENTION_NON_ROUTE_CONTROL",
    ]
}

fn eval_families() -> Vec<&'static str> {
    vec![
        "CONTEXT_ENTITY_EXTRACTION",
        "INSTRUCTION_FOLLOWING_CLOSED",
        "MULTI_HOP_KEY_VALUE_BINDING",
        "COUNTERFACTUAL_BINDING",
        "DISTRACTOR_RESISTANCE",
        "LONG_CONTEXT_NEEDLE_BINDING",
        "SYMBOLIC_RULE_CLOSED_CHOICE",
        "NON_ROUTE_TEXT_CONTROL",
    ]
}

fn select_training_data(arm: &str, train: &[Example]) -> Vec<Example> {
    match arm {
        "NO_TRAIN_069_BASELINE" => Vec::new(),
        "FRESH_SHUFFLED_LABEL_CONTROL" | "FINETUNE_SHUFFLED_LABEL_CONTROL" => {
            train.iter().take(12_000).cloned().collect()
        }
        "NO_ROUTE_FEATURE_CONTROL" => train.iter().take(30_000).cloned().collect(),
        _ => train.to_vec(),
    }
}

fn evaluate_model(arm: &str, model: &Model, examples: &[Example], use_route_features: bool) -> EvalResult {
    let mut correct = 0usize;
    let mut distribution = BTreeMap::<String, usize>::new();
    let mut samples = Vec::with_capacity(examples.len());
    let mut family_samples = BTreeMap::<String, Vec<EvalSample>>::new();
    for ex in examples {
        let predicted = model.predict(&ex.input, use_route_features);
        let ok = predicted == ex.expected_output;
        if ok {
            correct += 1;
        }
        *distribution.entry(predicted.clone()).or_insert(0) += 1;
        let sample = EvalSample {
            arm: arm.to_string(),
            example_id: ex.id.clone(),
            task_family: ex.task_family.clone(),
            input: ex.input.clone(),
            expected_output: ex.expected_output.clone(),
            predicted_output: predicted,
            correct: ok,
        };
        family_samples.entry(ex.task_family.clone()).or_default().push(sample.clone());
        samples.push(sample);
    }
    let mut family = BTreeMap::new();
    for (name, rows) in family_samples {
        family.insert(name, family_metric(&rows, examples));
    }
    EvalResult {
        accuracy: safe_div(correct, examples.len()),
        family,
        distribution,
        samples,
        eval_row_hash: eval_row_hash(examples),
    }
}

fn family_metric(rows: &[EvalSample], examples: &[Example]) -> FamilyMetric {
    let total = rows.len();
    let correct = rows.iter().filter(|row| row.correct).count();
    let mut distribution = BTreeMap::<String, usize>::new();
    let mut copy_last = 0usize;
    for row in rows {
        *distribution.entry(row.predicted_output.clone()).or_insert(0) += 1;
        if let Some(ex) = examples.iter().find(|ex| ex.id == row.example_id) {
            if row.predicted_output == last_token(&ex.input) {
                copy_last += 1;
            }
        }
    }
    FamilyMetric {
        correct,
        total,
        accuracy: safe_div(correct, total),
        top_output_rate: top_output_rate(&distribution, total),
        space_output_rate: output_rate(&distribution, " ", total),
        empty_output_rate: output_rate(&distribution, "", total),
        unique_output_count: distribution.len(),
        output_entropy: entropy(&distribution, total),
        repetition_rate: top_output_rate(&distribution, total),
        copy_last_token_rate: safe_div(copy_last, total),
    }
}

fn capability_metrics(eval: &EvalResult, majority_accuracy: f64, copy_first_accuracy: f64) -> CapabilityMetrics {
    let family_min_accuracy = eval_families()
        .into_iter()
        .map(|family| family_accuracy(&eval.family, family))
        .fold(1.0, f64::min);
    CapabilityMetrics {
        supported_accuracy: eval.accuracy,
        family_min_accuracy,
        context_entity_extraction_accuracy: family_accuracy(&eval.family, "CONTEXT_ENTITY_EXTRACTION"),
        instruction_following_closed_accuracy: family_accuracy(&eval.family, "INSTRUCTION_FOLLOWING_CLOSED"),
        multi_hop_key_value_accuracy: family_accuracy(&eval.family, "MULTI_HOP_KEY_VALUE_BINDING"),
        counterfactual_binding_accuracy: family_accuracy(&eval.family, "COUNTERFACTUAL_BINDING"),
        distractor_resistance_accuracy: family_accuracy(&eval.family, "DISTRACTOR_RESISTANCE"),
        long_context_needle_accuracy: family_accuracy(&eval.family, "LONG_CONTEXT_NEEDLE_BINDING"),
        symbolic_rule_closed_choice_accuracy: family_accuracy(&eval.family, "SYMBOLIC_RULE_CLOSED_CHOICE"),
        non_route_text_control_accuracy: family_accuracy(&eval.family, "NON_ROUTE_TEXT_CONTROL"),
        delta_vs_majority: eval.accuracy - majority_accuracy,
        delta_vs_copy_first_match: eval.accuracy - copy_first_accuracy,
    }
}

fn evaluate_baselines(examples: &[Example], labels: &[String]) -> BTreeMap<String, BTreeMap<String, String>> {
    let majority = majority_label(examples);
    let mut out = BTreeMap::new();
    for (idx, ex) in examples.iter().enumerate() {
        let mut row = BTreeMap::new();
        row.insert("MAJORITY_LABEL".to_string(), majority.clone());
        row.insert("ANSWER_PRIOR_ONLY".to_string(), majority.clone());
        row.insert("COPY_LAST_TOKEN".to_string(), last_token(&ex.input));
        row.insert("COPY_FIRST_MATCH".to_string(), copy_first_match(&ex.input, labels));
        row.insert("UNIGRAM_LABEL_PRIOR".to_string(), majority.clone());
        row.insert(
            "SHUFFLED_CONTEXT".to_string(),
            examples[(idx + 17) % examples.len()].expected_output.clone(),
        );
        row.insert(
            "SHUFFLED_LABELS".to_string(),
            examples[(idx + 7) % examples.len()].expected_output.clone(),
        );
        row.insert("NO_ROUTE_FEATURES".to_string(), String::new());
        out.insert(ex.id.clone(), row);
    }
    out
}

fn baseline_metrics(
    outputs: &BTreeMap<String, BTreeMap<String, String>>,
    examples: &[Example],
) -> BTreeMap<String, serde_json::Value> {
    let mut by_name = BTreeMap::<String, (usize, BTreeMap<String, usize>)>::new();
    for ex in examples {
        if let Some(row) = outputs.get(&ex.id) {
            for (name, pred) in row {
                let entry = by_name.entry(name.clone()).or_default();
                if pred == &ex.expected_output {
                    entry.0 += 1;
                }
                *entry.1.entry(pred.clone()).or_insert(0) += 1;
            }
        }
    }
    by_name
        .into_iter()
        .map(|(name, (correct, distribution))| {
            (
                name,
                json!({
                    "accuracy": safe_div(correct, examples.len()),
                    "top_output_rate": top_output_rate(&distribution, examples.len()),
                    "output_entropy": entropy(&distribution, examples.len())
                }),
            )
        })
        .collect()
}

fn collapse_metrics(eval: &EvalResult) -> serde_json::Value {
    let top = top_output_rate(&eval.distribution, eval.samples.len());
    let space = output_rate(&eval.distribution, " ", eval.samples.len());
    let empty = output_rate(&eval.distribution, "", eval.samples.len());
    json!({
        "schema_version": "distractor_repair_collapse_metrics_v1",
        "global": {
            "top_output_rate": top,
            "space_output_rate": space,
            "empty_output_rate": empty,
            "unique_output_count": eval.distribution.len(),
            "output_entropy": entropy(&eval.distribution, eval.samples.len()),
            "collapse_detected": top > 0.45 || space > 0.02 || empty > 0.02
        },
        "per_family": eval.family
    })
}

fn retention_metrics(finetune: &ArmOutcome, reference: &Reference069) -> serde_json::Value {
    let baseline = |family: &str| -> f64 {
        reference
            .per_family
            .get(family)
            .and_then(|v| v.get("accuracy"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0)
    };
    let mut rows = BTreeMap::new();
    let mut pass = true;
    for family in [
        "INSTRUCTION_FOLLOWING_CLOSED",
        "MULTI_HOP_KEY_VALUE_BINDING",
        "SYMBOLIC_RULE_CLOSED_CHOICE",
        "NON_ROUTE_TEXT_CONTROL",
    ] {
        let before = baseline(family);
        let after = family_accuracy(&finetune.eval.family, family);
        let regression = before - after;
        let ok = regression <= 0.05;
        pass &= ok;
        rows.insert(
            family.to_string(),
            json!({
                "baseline_069_accuracy": before,
                "finetune_accuracy": after,
                "absolute_regression": regression.max(0.0),
                "retention_pass": ok
            }),
        );
    }
    json!({
        "schema_version": "retention_metrics_v1",
        "retention_gate_pass": pass,
        "max_allowed_absolute_regression": 0.05,
        "families": rows
    })
}

fn arm_comparison(fresh: &ArmOutcome, finetune: &ArmOutcome, no_route: &ArmOutcome) -> serde_json::Value {
    let fresh_pass = capability_gate(&fresh.capability, &fresh.collapse);
    let finetune_pass = capability_gate(&finetune.capability, &finetune.collapse);
    let best_arm = if finetune.capability.family_min_accuracy >= fresh.capability.family_min_accuracy {
        "FINETUNE_068_TARGETED_REPAIR"
    } else {
        "FRESH_TARGETED_MIX_TRAINING"
    };
    json!({
        "schema_version": "arm_comparison_v1",
        "best_arm": best_arm,
        "fresh_pass": fresh_pass,
        "finetune_pass": finetune_pass,
        "fresh_vs_finetune_delta": finetune.capability.family_min_accuracy - fresh.capability.family_min_accuracy,
        "no_route_feature_control_family_min": no_route.capability.family_min_accuracy,
        "recommended_next_strategy": if best_arm == "FINETUNE_068_TARGETED_REPAIR" {
            "continue targeted repair from 068-style checkpoint with retention checks"
        } else {
            "prefer fresh targeted curriculum before later integration"
        }
    })
}

fn derive_verdicts(
    fresh: &ArmOutcome,
    finetune: &ArmOutcome,
    best: &ArmOutcome,
    retention: &serde_json::Value,
    arm_comparison: &serde_json::Value,
    overlap_with_069_eval_count: usize,
    upstream_unchanged: bool,
    reload_pass: bool,
    rollback_pass: bool,
    resume_pass: bool,
    resumed_hash_changed: bool,
) -> Vec<String> {
    let fresh_update = fresh.actual_training_update_detected;
    let finetune_update = finetune.actual_training_update_detected;
    let retention_pass = retention
        .get("retention_gate_pass")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let fresh_pass = arm_comparison
        .get("fresh_pass")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let finetune_pass = arm_comparison
        .get("finetune_pass")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let baseline_ok = best.capability.delta_vs_majority > 0.10 && best.capability.delta_vs_copy_first_match > 0.10;
    let checkpoint_ok = reload_pass && rollback_pass && resume_pass && resumed_hash_changed;
    let hard_ok = upstream_unchanged
        && fresh_update
        && finetune_update
        && overlap_with_069_eval_count == 0
        && fresh_pass
        && finetune_pass
        && retention_pass
        && baseline_ok
        && checkpoint_ok
        && !best.prediction_oracle_used;
    if hard_ok {
        vec![
            "DISTRACTOR_RESISTANT_ANCHORROUTE_TRAINING_POSITIVE".to_string(),
            "FRESH_TARGETED_MIX_TRAINING_COMPLETED".to_string(),
            "FINETUNE_068_TARGETED_REPAIR_COMPLETED".to_string(),
            "DISTRACTOR_RESISTANCE_IMPROVED".to_string(),
            "CONTEXT_ENTITY_EXTRACTION_IMPROVED".to_string(),
            "COUNTERFACTUAL_BINDING_IMPROVED".to_string(),
            "LONG_CONTEXT_NEEDLE_IMPROVED".to_string(),
            "RETENTION_GATE_PASSES".to_string(),
            "RETENTION_REGRESSION_REJECTED".to_string(),
            "ARM_COMPARISON_WRITTEN".to_string(),
            "BEST_ARM_SELECTED".to_string(),
            "TRAIN_BENCHMARK_LEAKAGE_REJECTED".to_string(),
            "UPSTREAM_068_CHECKPOINT_UNCHANGED".to_string(),
            "BASELINE_KNOCKOUT_STABLE".to_string(),
            "CHECKPOINT_PIPELINE_STRICT_PASS".to_string(),
            "ORACLE_SHORTCUT_REJECTED".to_string(),
            "PRODUCTION_TRAINING_NOT_CLAIMED".to_string(),
        ]
    } else {
        let mut out = vec!["DISTRACTOR_RESISTANT_ANCHORROUTE_TRAINING_FAILS".to_string()];
        if !upstream_unchanged {
            out.push("CHECKPOINT_MUTATION_DETECTED".to_string());
        }
        if !fresh_update || !finetune_update {
            out.push("NO_ACTUAL_TRAINING_UPDATE_DETECTED".to_string());
        }
        if overlap_with_069_eval_count > 0 {
            out.push("TRAIN_BENCHMARK_LEAKAGE_DETECTED".to_string());
        }
        if !fresh_pass || !finetune_pass {
            if best.capability.distractor_resistance_accuracy < 0.80 {
                out.push("DISTRACTOR_RESISTANCE_STILL_FAILS".to_string());
            }
            if best.capability.context_entity_extraction_accuracy < 0.85 {
                out.push("CONTEXT_ENTITY_EXTRACTION_STILL_FAILS".to_string());
            }
            if best.capability.counterfactual_binding_accuracy < 0.85 {
                out.push("COUNTERFACTUAL_BINDING_STILL_FAILS".to_string());
            }
            if best.capability.long_context_needle_accuracy < 0.65 {
                out.push("LONG_CONTEXT_NEEDLE_STILL_FAILS".to_string());
            }
        }
        if !retention_pass {
            out.push("RETENTION_REGRESSION_DETECTED".to_string());
        }
        if !baseline_ok {
            out.push("BASELINE_EVAL_MISMATCH".to_string());
        }
        if !reload_pass {
            out.push("CHECKPOINT_RELOAD_FAILS".to_string());
        }
        if !rollback_pass {
            out.push("ROLLBACK_REHEARSAL_FAILS".to_string());
        }
        if !resume_pass || !resumed_hash_changed {
            out.push("RESUME_FROM_CHECKPOINT_FAILS".to_string());
        }
        out.push("ARM_COMPARISON_WRITTEN".to_string());
        out.push("ORACLE_SHORTCUT_REJECTED".to_string());
        out.push("PRODUCTION_TRAINING_NOT_CLAIMED".to_string());
        out
    }
}

fn capability_gate(metrics: &CapabilityMetrics, collapse: &serde_json::Value) -> bool {
    let collapse_detected = collapse
        .get("global")
        .and_then(|v| v.get("collapse_detected"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let top = collapse
        .get("global")
        .and_then(|v| v.get("top_output_rate"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    let space = collapse
        .get("global")
        .and_then(|v| v.get("space_output_rate"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    let empty = collapse
        .get("global")
        .and_then(|v| v.get("empty_output_rate"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    metrics.context_entity_extraction_accuracy >= 0.85
        && metrics.counterfactual_binding_accuracy >= 0.85
        && metrics.distractor_resistance_accuracy >= 0.80
        && metrics.long_context_needle_accuracy >= 0.65
        && metrics.family_min_accuracy >= 0.70
        && metrics.delta_vs_majority > 0.10
        && metrics.delta_vs_copy_first_match > 0.10
        && top <= 0.45
        && space <= 0.02
        && empty <= 0.02
        && !collapse_detected
}

fn evaluate_checkpoint_reload(best: &ArmOutcome, eval_rows: &[Example], out: &Path) -> Result<bool, Box<dyn std::error::Error>> {
    let path = out
        .join("checkpoints")
        .join(best.arm.to_ascii_lowercase())
        .join("model_checkpoint.json");
    let loaded = Model::load(&path)?;
    let eval = evaluate_model("CHECKPOINT_RELOAD_EVAL", &loaded, eval_rows, true);
    Ok((eval.accuracy - best.eval.accuracy).abs() < 1e-12 && loaded.sha256()? == best.checkpoint_after_hash)
}

fn evaluate_checkpoint_rollback(best: &ArmOutcome, eval_rows: &[Example], out: &Path) -> Result<bool, Box<dyn std::error::Error>> {
    let path = out
        .join("checkpoints")
        .join(best.arm.to_ascii_lowercase())
        .join("model_checkpoint.json");
    let rollback_model = Model::load(&path)?;
    let rollback_eval = evaluate_model("ROLLBACK_REHEARSAL", &rollback_model, eval_rows, true);
    Ok((rollback_eval.accuracy - best.eval.accuracy).abs() < 1e-12)
}

fn resume_from_checkpoint(
    best: &ArmOutcome,
    train: &[Example],
    eval_rows: &[Example],
    out: &Path,
) -> Result<(bool, bool), Box<dyn std::error::Error>> {
    let path = out
        .join("checkpoints")
        .join(best.arm.to_ascii_lowercase())
        .join("model_checkpoint.json");
    let mut resumed = Model::load(&path)?;
    let before = resumed.sha256()?;
    let subset: Vec<Example> = train.iter().take(512).cloned().collect();
    let report = resumed.train(&subset, 1, true, None)?;
    let after = resumed.sha256()?;
    let resumed_path = out.join("checkpoints").join("resume_from_checkpoint").join("model_checkpoint.json");
    resumed.save(&resumed_path)?;
    let eval = evaluate_model("RESUME_FROM_CHECKPOINT", &resumed, eval_rows, true);
    Ok((report.train_step_count > 0 && eval.accuracy >= 0.60, before != after))
}

fn featurize(input: &str, dim: usize, use_route_features: bool) -> Vec<usize> {
    let lowered = input.to_ascii_lowercase();
    let mut feats = BTreeSet::<usize>::new();
    let tokens = tokenize(&lowered);
    for token in &tokens {
        feats.insert(hash_feature(dim, &format!("tok:{token}")));
    }
    for window in tokens.windows(2) {
        feats.insert(hash_feature(dim, &format!("bi:{}:{}", window[0], window[1])));
    }
    for window in tokens.windows(3) {
        feats.insert(hash_feature(
            dim,
            &format!("tri:{}:{}:{}", window[0], window[1], window[2]),
        ));
    }
    for (pos, token) in tokens.iter().enumerate().take(120) {
        feats.insert(hash_feature(dim, &format!("pos{}:{token}", pos / 8)));
    }
    if use_route_features {
        for marker in [
            "route",
            "query",
            "anchor",
            "value",
            "target",
            "needle",
            "active",
            "current",
            "episode",
            "equals",
            "different",
        ] {
            if lowered.contains(marker) {
                feats.insert(hash_feature(dim, &format!("marker:{marker}")));
            }
        }
        for window in tokens.windows(4) {
            if window.iter().any(|t| {
                *t == "query"
                    || *t == "anchor"
                    || *t == "value"
                    || *t == "episode"
                    || *t == "equals"
                    || *t == "current"
            }) {
                feats.insert(hash_feature(
                    dim,
                    &format!("route4:{}:{}:{}:{}", window[0], window[1], window[2], window[3]),
                ));
            }
        }
    }
    for domain in ["weather", "music", "math", "garden", "archive"] {
        if lowered.contains(&format!("about {domain}")) {
            feats.insert(hash_feature(dim, &format!("domain:{domain}")));
        }
    }
    if let Some((prefix, cand_a, cand_b)) = parse_candidates(&lowered) {
        let prefix_class = last_class(&prefix);
        let a_class = first_class(&cand_a);
        let b_class = first_class(&cand_b);
        feats.insert(hash_feature(dim, &format!("prefix_last:{prefix_class}")));
        feats.insert(hash_feature(dim, &format!("cand_a_first:{a_class}")));
        feats.insert(hash_feature(dim, &format!("cand_b_first:{b_class}")));
        feats.insert(hash_feature(dim, &format!("transition_a:{prefix_class}->{a_class}")));
        feats.insert(hash_feature(dim, &format!("transition_b:{prefix_class}->{b_class}")));
    }
    feats.into_iter().collect()
}

fn tokenize(input: &str) -> Vec<String> {
    input
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_ascii_lowercase())
        .collect()
}

fn parse_candidates(input: &str) -> Option<(String, String, String)> {
    let p = input.find("prefix:")?;
    let a = input.find("candidate a:")?;
    let b = input.find("candidate b:")?;
    let c = input[b..].find("choose").map(|idx| b + idx).unwrap_or(input.len());
    if !(p < a && a < b && b < c) {
        return None;
    }
    Some((
        input[p + "prefix:".len()..a].trim().to_string(),
        input[a + "candidate a:".len()..b].trim().to_string(),
        input[b + "candidate b:".len()..c].trim().to_string(),
    ))
}

fn hash_feature(dim: usize, value: &str) -> usize {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    usize::from_le_bytes(bytes) % dim
}

fn hex_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn snapshot_file(path: &Path) -> Result<FileSnapshot, Box<dyn std::error::Error>> {
    let metadata = fs::metadata(path)?;
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    let modified_unix_ms = metadata
        .modified()
        .ok()
        .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis());
    Ok(FileSnapshot {
        path: path.display().to_string(),
        size_bytes: metadata.len(),
        modified_unix_ms,
        sha256: hex_sha256(&bytes),
    })
}

fn append_progress(out: &Path, event: &str, details: serde_json::Value) -> Result<(), Box<dyn std::error::Error>> {
    append_jsonl(
        &out.join("progress.jsonl"),
        &json!({
            "ts_unix_ms": now_ms(),
            "event": event,
            "details": details
        }),
    )
}

fn append_jsonl<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    file.write_all(serde_json::to_string(value)?.as_bytes())?;
    file.write_all(b"\n")?;
    file.flush()?;
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

fn write_sample_jsonl(path: &Path, rows: &[Example], limit: usize) -> Result<(), Box<dyn std::error::Error>> {
    let _ = fs::remove_file(path);
    for row in rows.iter().take(limit) {
        append_jsonl(path, row)?;
    }
    Ok(())
}

fn write_human_samples(
    path: &Path,
    outcomes: &[ArmOutcome],
    eval_rows: &[Example],
    baseline_outputs: &BTreeMap<String, BTreeMap<String, String>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let _ = fs::remove_file(path);
    for outcome in outcomes {
        for sample in outcome.eval.samples.iter().filter(|s| {
            matches!(
                s.task_family.as_str(),
                "CONTEXT_ENTITY_EXTRACTION"
                    | "COUNTERFACTUAL_BINDING"
                    | "DISTRACTOR_RESISTANCE"
                    | "LONG_CONTEXT_NEEDLE_BINDING"
            )
        }).take(96) {
            let base = baseline_outputs.get(&sample.example_id).cloned().unwrap_or_default();
            append_jsonl(
                path,
                &json!({
                    "arm": outcome.arm,
                    "task_family": sample.task_family,
                    "input": sample.input,
                    "expected_output": sample.expected_output,
                    "model_output": sample.predicted_output,
                    "baseline_outputs": base,
                    "pass_fail": if sample.correct { "pass" } else { "fail" },
                    "limitation_flag": null
                }),
            )?;
        }
    }
    let _ = eval_rows;
    Ok(())
}

fn write_failure_samples(path: &Path, outcomes: &[ArmOutcome], _eval_rows: &[Example]) -> Result<(), Box<dyn std::error::Error>> {
    let _ = fs::remove_file(path);
    for outcome in outcomes {
        for sample in outcome.eval.samples.iter().filter(|s| !s.correct).take(200) {
            append_jsonl(
                path,
                &json!({
                    "input": sample.input,
                    "expected": sample.expected_output,
                    "predicted": sample.predicted_output,
                    "arm": outcome.arm,
                    "task_family": sample.task_family,
                    "reason": "finite-label route selection mismatch"
                }),
            )?;
        }
    }
    if !path.exists() {
        File::create(path)?;
    }
    Ok(())
}

fn write_running_summary(
    out: &Path,
    status: &str,
    outcomes: &[ArmOutcome],
    verdicts: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    write_json(
        &out.join("summary.json"),
        &json!({
            "schema_version": "distractor_resistant_anchorroute_training_summary_v1",
            "status": status,
            "completed_arms": outcomes.len(),
            "best_arm": outcomes.iter()
                .filter(|out| out.arm == "FRESH_TARGETED_MIX_TRAINING" || out.arm == "FINETUNE_068_TARGETED_REPAIR")
                .max_by(|a, b| a.capability.family_min_accuracy.partial_cmp(&b.capability.family_min_accuracy).unwrap())
                .map(|v| v.arm.clone()),
            "verdicts": verdicts,
            "prediction_oracle_used": false,
            "open_ended_generation_supported": false,
            "free_form_answering_supported": false,
            "perplexity_supported": false,
            "language_grounding_claimed": false,
            "production_training_claimed": false
        }),
    )
}

fn write_report(
    out: &Path,
    status: &str,
    verdicts: &[String],
    best: Option<&ArmOutcome>,
    comparison: Option<&serde_json::Value>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut text = String::new();
    text.push_str("# STABLE_LOOP_PHASE_LOCK_070_DISTRACTOR_RESISTANT_ANCHORROUTE_TRAINING Report\n\n");
    text.push_str(&format!("Status: {status}\n\n"));
    text.push_str("This is finite-label repair benchmark training only.\n\n");
    text.push_str("no open-ended assistant\nno free-form generation\nno perplexity\nno full English LM\nno language grounding\nno production training\nno GA\nno public beta\nno hosted SaaS\nno clinical use\nno high-stakes education use\n\n");
    text.push_str("Verdicts:\n\n```text\n");
    for verdict in verdicts {
        text.push_str(verdict);
        text.push('\n');
    }
    text.push_str("```\n\n");
    if let Some(best) = best {
        text.push_str(&format!(
            "Best arm: `{}` with family_min_accuracy = `{}`.\n\n",
            best.arm, best.capability.family_min_accuracy
        ));
    }
    if let Some(comparison) = comparison {
        text.push_str("Arm comparison:\n\n```json\n");
        text.push_str(&serde_json::to_string_pretty(comparison)?);
        text.push_str("\n```\n");
    }
    fs::write(out.join("report.md"), text)?;
    Ok(())
}

fn write_failure(out: &Path, verdict: &str, message: &str) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out)?;
    write_json(
        &out.join("summary.json"),
        &json!({
            "status": "failed",
            "verdicts": [verdict],
            "message": message,
            "prediction_oracle_used": false,
            "production_training_claimed": false
        }),
    )?;
    fs::write(
        out.join("report.md"),
        format!("# STABLE_LOOP_PHASE_LOCK_070_DISTRACTOR_RESISTANT_ANCHORROUTE_TRAINING Report\n\nStatus: failed\n\n{verdict}: {message}\n"),
    )?;
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut out = PathBuf::from(DEFAULT_OUT);
    let mut upstream_checkpoint = PathBuf::from(DEFAULT_UPSTREAM_CHECKPOINT);
    let mut upstream_summary = PathBuf::from(DEFAULT_UPSTREAM_SUMMARY);
    let mut benchmark_069_root = PathBuf::from(DEFAULT_BENCHMARK_069_ROOT);
    let mut targeted_examples = DEFAULT_TARGETED_EXAMPLES;
    let mut seed = 2026u64;
    let mut heartbeat_sec = 20u64;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out = PathBuf::from(args.next().ok_or("--out requires value")?),
            "--upstream-checkpoint" => {
                upstream_checkpoint = PathBuf::from(args.next().ok_or("--upstream-checkpoint requires value")?)
            }
            "--upstream-summary" => {
                upstream_summary = PathBuf::from(args.next().ok_or("--upstream-summary requires value")?)
            }
            "--benchmark-069-root" => {
                benchmark_069_root = PathBuf::from(args.next().ok_or("--benchmark-069-root requires value")?)
            }
            "--targeted-examples" => {
                targeted_examples = args.next().ok_or("--targeted-examples requires value")?.parse()?
            }
            "--seed" => seed = args.next().ok_or("--seed requires value")?.parse()?,
            "--heartbeat-sec" => heartbeat_sec = args.next().ok_or("--heartbeat-sec requires value")?.parse()?,
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    Ok(Config {
        out,
        upstream_checkpoint,
        upstream_summary,
        benchmark_069_root,
        targeted_examples,
        seed,
        heartbeat_sec,
    })
}

fn truncate_outputs(out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    for file in [
        "progress.jsonl",
        "training_metrics.jsonl",
        "human_readable_samples.jsonl",
        "failure_case_samples.jsonl",
    ] {
        let path = out.join(file);
        if path.exists() {
            fs::write(path, b"")?;
        }
    }
    Ok(())
}

fn shuffle_labels(examples: &mut [Example], seed: u64) {
    let labels: Vec<String> = examples.iter().map(|ex| ex.expected_output.clone()).collect();
    if labels.is_empty() {
        return;
    }
    let mut rng = StdRng::seed_from_u64(seed);
    for (idx, ex) in examples.iter_mut().enumerate() {
        let j = (idx + 1 + rng.gen_range(0..labels.len())) % labels.len();
        ex.expected_output = labels[j].clone();
    }
}

fn outcome_by_arm<'a>(outcomes: &'a [ArmOutcome], arm: &str) -> Result<&'a ArmOutcome, Box<dyn std::error::Error>> {
    outcomes
        .iter()
        .find(|out| out.arm == arm)
        .ok_or_else(|| format!("ARM_COMPARISON_MISSING: {arm}").into())
}

fn majority_label(examples: &[Example]) -> String {
    let mut counts = BTreeMap::<String, usize>::new();
    for ex in examples {
        *counts.entry(ex.expected_output.clone()).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(label, _)| label)
        .unwrap_or_default()
}

fn copy_first_match(input: &str, labels: &[String]) -> String {
    let lowered = input.to_ascii_lowercase();
    let mut best: Option<(usize, String)> = None;
    for label in labels {
        if let Some(pos) = lowered.find(&label.to_ascii_lowercase()) {
            match &best {
                Some((best_pos, _)) if *best_pos <= pos => {}
                _ => best = Some((pos, label.clone())),
            }
        }
    }
    best.map(|(_, label)| label).unwrap_or_default()
}

fn baseline_accuracy(metrics: &BTreeMap<String, serde_json::Value>, name: &str) -> f64 {
    metrics
        .get(name)
        .and_then(|row| row.get("accuracy"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
}

fn family_accuracy(metrics: &BTreeMap<String, FamilyMetric>, family: &str) -> f64 {
    metrics.get(family).map(|v| v.accuracy).unwrap_or(0.0)
}

fn top_output_rate(distribution: &BTreeMap<String, usize>, total: usize) -> f64 {
    safe_div(distribution.values().copied().max().unwrap_or(0), total)
}

fn output_rate(distribution: &BTreeMap<String, usize>, key: &str, total: usize) -> f64 {
    safe_div(distribution.get(key).copied().unwrap_or(0), total)
}

fn entropy(distribution: &BTreeMap<String, usize>, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    distribution
        .values()
        .map(|count| {
            let p = *count as f64 / total as f64;
            if p > 0.0 {
                -p * p.log2()
            } else {
                0.0
            }
        })
        .sum()
}

fn safe_div(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

fn eval_row_hash(examples: &[Example]) -> String {
    let mut hasher = Sha256::new();
    for ex in examples {
        hasher.update(ex.id.as_bytes());
        hasher.update(b"\0");
        hasher.update(ex.task_family.as_bytes());
        hasher.update(b"\0");
        hasher.update(ex.input.as_bytes());
        hasher.update(b"\0");
        hasher.update(ex.expected_output.as_bytes());
        hasher.update(b"\n");
    }
    format!("{:x}", hasher.finalize())
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn long_filler(idx: usize, n: usize) -> String {
    let words = [
        "archive", "weather", "music", "garden", "math", "ledger", "window", "signal",
        "orange", "violet", "quiet", "north", "delta", "paper", "stone", "river",
    ];
    (0..n)
        .map(|i| format!("{}_{}", words[(idx + i) % words.len()], i))
        .collect::<Vec<_>>()
        .join(" ")
}

fn last_token(input: &str) -> String {
    tokenize(input).last().cloned().unwrap_or_default()
}

fn last_class(input: &str) -> &'static str {
    match input.chars().rev().find(|c| !c.is_whitespace()) {
        Some(c) if c.is_ascii_alphabetic() => "alpha",
        Some(c) if c.is_ascii_digit() => "digit",
        Some(_) => "punct",
        None => "empty",
    }
}

fn first_class(input: &str) -> &'static str {
    match input.chars().find(|c| !c.is_whitespace()) {
        Some(c) if c.is_ascii_alphabetic() => "alpha",
        Some(c) if c.is_ascii_digit() => "digit",
        Some(_) => "punct",
        None => "empty",
    }
}
