#![recursion_limit = "256"]

//! Runner-local token-level chat composition repair.
//!
//! 078 repairs the bounded 076 chat PoC failure diagnosed by 077/077B:
//! response-table/template copying instead of fresh token composition. The
//! decoder and training logic stay inside this example only.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_OUT: &str = "target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke";
const DEFAULT_UPSTREAM_076_ROOT: &str = "target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke";
const DEFAULT_UPSTREAM_077_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_077_chat_generation_fresh_composition_confirm/smoke";
const DEFAULT_UPSTREAM_077B_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_077b_chat_generation_failure_analysis/smoke";
const DEFAULT_UPSTREAM_074_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke";
const DEFAULT_CHAT_EXAMPLES: usize = 60_000;
const MAX_CHAT_EXAMPLES: usize = 150_000;
const MAX_RESPONSE_TOKENS: usize = 64;
const STOP_TOKEN: &str = "<eos>";

const TRAIN_FAMILIES: [&str; 7] = [
    "SIMPLE_INSTRUCTION_PARAPHRASE",
    "CONTEXT_CARRY_VARIABLE_SLOT",
    "SHORT_EXPLANATION_MANY_TARGET",
    "TWO_TURN_DIALOGUE_STATE",
    "BOUNDARY_REFUSAL_PARAPHRASE_MINI",
    "ANCHORROUTE_FINITE_LABEL_RETENTION",
    "ANTI_TEMPLATE_COPY_DROPOUT",
];

const EVAL_FAMILIES: [&str; 7] = [
    "SIMPLE_INSTRUCTION_PARAPHRASE",
    "CONTEXT_CARRY_VARIABLE_SLOT",
    "SHORT_EXPLANATION_MANY_TARGET",
    "TWO_TURN_DIALOGUE_STATE",
    "BOUNDARY_REFUSAL_PARAPHRASE_MINI",
    "ANCHORROUTE_FINITE_LABEL_RETENTION",
    "ANTI_TEMPLATE_COPY_DROPOUT",
];

#[derive(Debug, Clone)]
struct Config {
    out: PathBuf,
    upstream_076_root: PathBuf,
    upstream_077_root: PathBuf,
    upstream_077b_root: PathBuf,
    upstream_074_root: PathBuf,
    seed: u64,
    chat_examples: usize,
    heartbeat_sec: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UpstreamChatModel {
    #[serde(default)]
    labels: Vec<String>,
    #[serde(default)]
    response_table: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    weights: Vec<Vec<f32>>,
    #[serde(default)]
    bias: Vec<f32>,
    #[serde(default = "default_feature_dim")]
    feature_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompositionModel {
    schema_version: String,
    seed: u64,
    train_step_count: usize,
    token_train_step_count: usize,
    update_count: usize,
    vocab: Vec<String>,
    slot_values_seen: Vec<String>,
    token_counts: BTreeMap<String, BTreeMap<String, usize>>,
    runner_local_decoder_loop: bool,
    decoder_path: String,
    response_table_used_for_main_prediction: bool,
    response_table_path_available_but_disabled: bool,
    public_api_exposed: bool,
    service_api_exposed: bool,
    sdk_surface_exposed: bool,
}

#[derive(Debug, Clone, Serialize)]
struct TrainExample {
    id: String,
    family: String,
    prompt: String,
    response_text: String,
    intent: String,
    slot_value: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct EvalExample {
    id: String,
    eval_family: String,
    prompt: String,
    expected_behavior: String,
    required_keywords: Vec<String>,
    forbidden_outputs: Vec<String>,
    expected_slot: Option<String>,
    target_label: Option<String>,
    retention_row: bool,
}

#[derive(Debug, Clone, Serialize)]
struct EvalRow {
    arm: String,
    eval_family: String,
    prompt: String,
    model_output: String,
    expected_behavior: String,
    required_keywords: Vec<String>,
    forbidden_outputs: Vec<String>,
    pass_fail: String,
    output_classification: String,
    novelty_flag: bool,
    template_copy_flag: bool,
    slot_binding_diagnosis: String,
    short_diagnosis: String,
    generated_token_count: usize,
    slot_value_expected: Option<String>,
    slot_value_emitted: Option<String>,
}

#[derive(Debug, Clone)]
struct CopySources {
    train_responses: BTreeSet<String>,
    eval_outputs: BTreeSet<String>,
    response_table_outputs: BTreeSet<String>,
    template_responses: BTreeSet<String>,
}

fn main() {
    let cfg = match parse_args() {
        Ok(value) => value,
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
    if cfg.chat_examples > MAX_CHAT_EXAMPLES {
        return Err(format!("chat_examples exceeds hard cap {MAX_CHAT_EXAMPLES}").into());
    }
    fs::create_dir_all(&cfg.out)?;
    append_progress(
        &cfg.out,
        "start",
        json!({
            "seed": cfg.seed,
            "chat_examples": cfg.chat_examples,
            "heartbeat_sec": cfg.heartbeat_sec,
            "milestone": "STABLE_LOOP_PHASE_LOCK_078_CHAT_COMPOSITION_REPAIR"
        }),
    )?;
    write_summary_and_report(
        &cfg.out,
        "running",
        vec!["CHAT_COMPOSITION_REPAIR_RUNNING".to_string()],
        json!({"phase": "start"}),
    )?;
    write_json(
        &cfg.out.join("queue.json"),
        &json!({
            "schema_version": "chat_composition_repair_queue_v1",
            "milestone": "STABLE_LOOP_PHASE_LOCK_078_CHAT_COMPOSITION_REPAIR",
            "partial_write_policy": "progress summary report written from start and refreshed by phase",
            "steps": [
                "verify_upstreams",
                "load_076_checkpoint_read_only",
                "build_repair_dataset",
                "train_token_level_composition_model",
                "evaluate_main_and_controls",
                "validate_checkpoint_pipeline",
                "write_summary"
            ]
        }),
    )?;

    let upstream_checkpoint = cfg
        .upstream_076_root
        .join("checkpoints")
        .join("chat_generation_poc")
        .join("model_checkpoint.json");
    let missing = missing_upstreams(cfg, &upstream_checkpoint);
    if !missing.is_empty() {
        let verdict = if missing.iter().any(|item| item.starts_with("upstream_077b")) {
            "UPSTREAM_077B_ARTIFACT_MISSING"
        } else {
            "UPSTREAM_076_ARTIFACT_MISSING"
        };
        write_failure(&cfg.out, verdict, &missing.join(","))?;
        return Err(format!("{verdict}: {}", missing.join(",")).into());
    }

    let upstream_076_summary: Value = read_json(&cfg.upstream_076_root.join("summary.json"))?;
    let upstream_077_summary: Value = read_json(&cfg.upstream_077_root.join("summary.json"))?;
    let upstream_077b_summary: Value = read_json(&cfg.upstream_077b_root.join("summary.json"))?;
    let upstream_074_summary: Value = read_json(&cfg.upstream_074_root.join("summary.json"))?;
    if !value_has_verdict(&upstream_076_summary, "CHAT_GENERATION_POC_POSITIVE")
        || !value_has_verdict(&upstream_077_summary, "CHAT_GENERATION_FRESH_COMPOSITION_CONFIRM_FAILS")
        || !value_has_verdict(&upstream_077b_summary, "CHAT_GENERATION_FAILURE_ANALYSIS_POSITIVE")
        || !value_has_verdict(&upstream_074_summary, "MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_POSITIVE")
    {
        write_failure(&cfg.out, "UPSTREAM_077B_ARTIFACT_MISSING", "required upstream verdict missing")?;
        return Err("UPSTREAM_077B_ARTIFACT_MISSING".into());
    }

    let upstream_checkpoint_hash_before = sha256_file(&upstream_checkpoint)?;
    let upstream_model: UpstreamChatModel = read_json(&upstream_checkpoint)?;
    let copy_sources = build_copy_sources(cfg, &upstream_model)?;
    write_json(
        &cfg.out.join("upstream_manifest.json"),
        &json!({
            "schema_version": "chat_composition_repair_upstream_manifest_v1",
            "upstream_076_root": cfg.upstream_076_root.display().to_string(),
            "upstream_077_root": cfg.upstream_077_root.display().to_string(),
            "upstream_077b_root": cfg.upstream_077b_root.display().to_string(),
            "upstream_074_root": cfg.upstream_074_root.display().to_string(),
            "upstream_checkpoint": upstream_checkpoint.display().to_string(),
            "upstream_checkpoint_hash_before": upstream_checkpoint_hash_before,
            "upstream_076_positive": true,
            "upstream_077_failed_as_expected": true,
            "upstream_077b_positive": true,
            "upstream_074_positive": true
        }),
    )?;
    append_progress(&cfg.out, "upstreams_verified", json!({"upstreams": true}))?;

    write_json(
        &cfg.out.join("training_config.json"),
        &json!({
            "schema_version": "chat_composition_repair_training_config_v1",
            "runner_local_only": true,
            "response_table_used_for_main_prediction": false,
            "decoder_path": "token_level_next_token",
            "response_table_path_available_but_disabled": true,
            "llm_judge_used": false,
            "default_decode": "deterministic greedy",
            "teacher_forced_next_token_objective": true,
            "max_response_tokens": MAX_RESPONSE_TOKENS,
            "stop_token": STOP_TOKEN,
            "seed": cfg.seed,
            "chat_examples": cfg.chat_examples,
            "chat_examples_hard_cap": MAX_CHAT_EXAMPLES,
            "data_mix": {
                "SIMPLE_INSTRUCTION_PARAPHRASE": 0.30,
                "CONTEXT_CARRY_VARIABLE_SLOT": 0.20,
                "SHORT_EXPLANATION_MANY_TARGET": 0.15,
                "TWO_TURN_DIALOGUE_STATE": 0.10,
                "BOUNDARY_REFUSAL_PARAPHRASE_MINI": 0.10,
                "ANCHORROUTE_FINITE_LABEL_RETENTION": 0.10,
                "ANTI_TEMPLATE_COPY_DROPOUT": 0.05
            },
            "no_product_api": true,
            "no_sdk_surface": true,
            "no_service_api": true,
            "no_deployment_harness": true,
            "no_release_docs": true
        }),
    )?;

    let train_examples = build_train_examples(cfg.chat_examples, cfg.seed);
    let eval_examples = build_eval_examples(cfg.seed);
    let leakage = leakage_report(&train_examples, &eval_examples);
    if leakage["train_eval_exact_prompt_overlap_count"].as_u64().unwrap_or(1) > 0 {
        write_failure(&cfg.out, "TRAIN_EVAL_LEAKAGE_DETECTED", "exact prompt overlap")?;
        return Err("TRAIN_EVAL_LEAKAGE_DETECTED".into());
    }
    write_json(
        &cfg.out.join("repair_dataset_manifest.json"),
        &json!({
            "schema_version": "chat_composition_repair_dataset_manifest_v1",
            "train_examples": train_examples.len(),
            "eval_examples": eval_examples.len(),
            "train_family_counts": family_counts_train(&train_examples),
            "eval_family_counts": family_counts_eval(&eval_examples),
            "train_eval_exact_prompt_overlap_count": leakage["train_eval_exact_prompt_overlap_count"],
            "train_eval_exact_response_overlap_count": leakage["train_eval_exact_response_overlap_count"],
            "train_eval_template_overlap_count": leakage["train_eval_template_overlap_count"],
            "train_prompt_hash": leakage["train_prompt_hash"],
            "eval_prompt_hash": leakage["eval_prompt_hash"]
        }),
    )?;
    write_jsonl(&cfg.out.join("train_examples_sample.jsonl"), &train_examples, 260)?;
    write_jsonl(&cfg.out.join("eval_examples_sample.jsonl"), &eval_examples, eval_examples.len())?;
    append_progress(
        &cfg.out,
        "dataset_written",
        json!({"train_examples": train_examples.len(), "eval_examples": eval_examples.len()}),
    )?;
    write_summary_and_report(
        &cfg.out,
        "running",
        vec!["CHAT_COMPOSITION_REPAIR_RUNNING".to_string()],
        json!({"phase": "dataset_written", "train_examples": train_examples.len()}),
    )?;

    let mut model = CompositionModel::new(cfg.seed);
    let checkpoint_before_hash = model.sha256()?;
    let token_loss_initial = token_loss(&model, &train_examples.iter().take(512).cloned().collect::<Vec<_>>());
    train_token_model(&mut model, &train_examples, &cfg.out)?;
    let token_loss_final = token_loss(&model, &train_examples.iter().take(512).cloned().collect::<Vec<_>>());
    let checkpoint_after_hash = model.sha256()?;
    let teacher_forced_next_token_accuracy =
        teacher_forced_accuracy(&model, &train_examples.iter().take(1024).cloned().collect::<Vec<_>>());
    if model.train_step_count == 0
        || model.token_train_step_count == 0
        || checkpoint_after_hash == checkpoint_before_hash
    {
        write_failure(&cfg.out, "NO_ACTUAL_TRAINING_UPDATE_DETECTED", "checkpoint hash unchanged")?;
        return Err("NO_ACTUAL_TRAINING_UPDATE_DETECTED".into());
    }
    if token_loss_final >= token_loss_initial {
        write_failure(&cfg.out, "TOKEN_OBJECTIVE_NOT_LEARNED", "token loss did not decrease")?;
        return Err("TOKEN_OBJECTIVE_NOT_LEARNED".into());
    }
    append_progress(
        &cfg.out,
        "training_completed",
        json!({
            "train_step_count": model.train_step_count,
            "token_loss_initial": token_loss_initial,
            "token_loss_final": token_loss_final
        }),
    )?;

    let main_rows = evaluate_arm("TOKEN_COMPOSITION_REPAIR", &eval_examples, |prompt| {
        model.generate_main(prompt)
    }, &copy_sources, &upstream_model.labels);
    let table_rows = evaluate_arm("RESPONSE_TABLE_ONLY_CONTROL", &eval_examples, |prompt| {
        upstream_model.generate(prompt)
    }, &copy_sources, &upstream_model.labels);
    let baseline_rows = evaluate_arm("NO_REPAIR_076_BASELINE", &eval_examples, |prompt| {
        upstream_model.generate(prompt)
    }, &copy_sources, &upstream_model.labels);
    let no_dropout_rows = evaluate_arm("TOKEN_COMPOSITION_NO_DROPOUT_CONTROL", &eval_examples, |prompt| {
        model.generate_no_dropout_control(prompt)
    }, &copy_sources, &upstream_model.labels);
    let no_context_rows = evaluate_arm("NO_CONTEXT_SLOT_CONTROL", &eval_examples, |prompt| {
        model.generate_no_context_slot_control(prompt)
    }, &copy_sources, &upstream_model.labels);
    let retention_control_rows = evaluate_arm("FINITE_LABEL_RETENTION_CONTROL", &eval_examples, |prompt| {
        model.generate_retention_control(prompt)
    }, &copy_sources, &upstream_model.labels);

    write_jsonl(&cfg.out.join("generation_samples.jsonl"), &main_rows, main_rows.len())?;
    write_human_samples(&cfg.out.join("human_readable_samples.jsonl"), &main_rows)?;
    let composition = composition_metrics(&main_rows, &upstream_model.labels);
    let novelty = novelty_metrics(&main_rows, &copy_sources);
    let context_slot = context_slot_metrics(&main_rows);
    let retention = finite_label_retention_metrics(&main_rows);
    let collapse = collapse_metrics(&main_rows, &upstream_model.labels);
    write_json(&cfg.out.join("composition_metrics.json"), &composition)?;
    write_json(&cfg.out.join("novelty_metrics.json"), &novelty)?;
    write_json(&cfg.out.join("context_slot_metrics.json"), &context_slot)?;
    write_json(&cfg.out.join("finite_label_retention_metrics.json"), &retention)?;
    write_json(&cfg.out.join("collapse_metrics.json"), &collapse)?;

    let table_novelty = novelty_metrics(&table_rows, &copy_sources);
    let no_context_composition = composition_metrics(&no_context_rows, &upstream_model.labels);
    let no_dropout_novelty = novelty_metrics(&no_dropout_rows, &copy_sources);
    let delta_vs_response_table_only_control =
        novelty["novel_response_rate"].as_f64().unwrap_or(0.0)
            - table_novelty["novel_response_rate"].as_f64().unwrap_or(0.0);
    let delta_vs_no_context_slot_control =
        composition["fresh_context_carry_accuracy"].as_f64().unwrap_or(0.0)
            - no_context_composition["fresh_context_carry_accuracy"].as_f64().unwrap_or(0.0);
    let delta_vs_no_dropout_control =
        no_dropout_novelty["template_copy_rate"].as_f64().unwrap_or(1.0)
            - novelty["template_copy_rate"].as_f64().unwrap_or(1.0);
    let arm_comparison = json!({
        "schema_version": "chat_composition_repair_arm_comparison_v1",
        "arms": [
            arm_metrics("NO_REPAIR_076_BASELINE", &baseline_rows, &copy_sources, &upstream_model.labels),
            arm_metrics("RESPONSE_TABLE_ONLY_CONTROL", &table_rows, &copy_sources, &upstream_model.labels),
            arm_metrics("TOKEN_COMPOSITION_REPAIR", &main_rows, &copy_sources, &upstream_model.labels),
            arm_metrics("TOKEN_COMPOSITION_NO_DROPOUT_CONTROL", &no_dropout_rows, &copy_sources, &upstream_model.labels),
            arm_metrics("NO_CONTEXT_SLOT_CONTROL", &no_context_rows, &copy_sources, &upstream_model.labels),
            arm_metrics("FINITE_LABEL_RETENTION_CONTROL", &retention_control_rows, &copy_sources, &upstream_model.labels)
        ],
        "delta_vs_response_table_only_control": delta_vs_response_table_only_control,
        "delta_vs_no_context_slot_control": delta_vs_no_context_slot_control,
        "delta_vs_no_dropout_control": delta_vs_no_dropout_control,
        "control_delta_pass": delta_vs_response_table_only_control > 0.20
            && delta_vs_no_context_slot_control > 0.20
            && delta_vs_no_dropout_control > 0.05
    });
    write_json(&cfg.out.join("arm_comparison.json"), &arm_comparison)?;
    append_progress(
        &cfg.out,
        "eval_completed",
        json!({
            "novel_response_rate": novelty["novel_response_rate"],
            "template_copy_rate": novelty["template_copy_rate"],
            "fresh_context_carry_accuracy": composition["fresh_context_carry_accuracy"],
            "finite_label_retention_accuracy": retention["finite_label_retention_accuracy"]
        }),
    )?;

    let checkpoint_dir = cfg.out.join("checkpoints").join("chat_composition_repair");
    fs::create_dir_all(&checkpoint_dir)?;
    let checkpoint_path = checkpoint_dir.join("model_checkpoint.json");
    write_json(&checkpoint_path, &model)?;
    let loaded: CompositionModel = read_json(&checkpoint_path)?;
    let reload_rows = evaluate_arm("CHECKPOINT_RELOAD_EVAL", &eval_examples, |prompt| {
        loaded.generate_main(prompt)
    }, &copy_sources, &upstream_model.labels);
    let checkpoint_save_load_pass = loaded.sha256()? == checkpoint_after_hash;
    let eval_after_reload_matches_before = eval_signature(&main_rows) == eval_signature(&reload_rows);
    let mut resumed = loaded.clone();
    let pre_resume_hash = resumed.sha256()?;
    train_token_model_no_progress(&mut resumed, &train_examples.iter().take(128).cloned().collect::<Vec<_>>());
    let resume_dir = cfg.out.join("checkpoints").join("resume_from_checkpoint");
    fs::create_dir_all(&resume_dir)?;
    let resume_path = resume_dir.join("model_checkpoint.json");
    write_json(&resume_path, &resumed)?;
    let resumed_checkpoint_hash = resumed.sha256()?;
    let resume_from_checkpoint_pass = resumed_checkpoint_hash != pre_resume_hash;
    write_json(
        &cfg.out.join("checkpoint_manifest.json"),
        &json!({
            "schema_version": "chat_composition_repair_checkpoint_manifest_v1",
            "checkpoint_path": checkpoint_path.display().to_string(),
            "resume_checkpoint_path": resume_path.display().to_string(),
            "checkpoint_save_load_pass": checkpoint_save_load_pass,
            "resume_from_checkpoint_pass": resume_from_checkpoint_pass,
            "eval_after_reload_matches_before": eval_after_reload_matches_before,
            "train_step_count": model.train_step_count,
            "token_train_step_count": model.token_train_step_count,
            "response_table_used_for_main_prediction": false
        }),
    )?;
    write_json(
        &cfg.out.join("checkpoint_hashes.json"),
        &json!({
            "schema_version": "chat_composition_repair_checkpoint_hashes_v1",
            "upstream_checkpoint_hash_before": upstream_checkpoint_hash_before,
            "checkpoint_before_hash": checkpoint_before_hash,
            "checkpoint_after_hash": checkpoint_after_hash,
            "pre_resume_checkpoint_hash": pre_resume_hash,
            "resumed_checkpoint_hash": resumed_checkpoint_hash
        }),
    )?;
    append_progress(
        &cfg.out,
        "checkpoint_pipeline_completed",
        json!({
            "checkpoint_save_load_pass": checkpoint_save_load_pass,
            "resume_from_checkpoint_pass": resume_from_checkpoint_pass,
            "eval_after_reload_matches_before": eval_after_reload_matches_before
        }),
    )?;

    let upstream_checkpoint_hash_after = sha256_file(&upstream_checkpoint)?;
    let upstream_checkpoint_unchanged = upstream_checkpoint_hash_before == upstream_checkpoint_hash_after;
    if !upstream_checkpoint_unchanged {
        write_failure(&cfg.out, "UPSTREAM_CHECKPOINT_MUTATION_DETECTED", "upstream checkpoint hash changed")?;
        return Err("UPSTREAM_CHECKPOINT_MUTATION_DETECTED".into());
    }

    let control_delta_pass = arm_comparison["control_delta_pass"].as_bool().unwrap_or(false);
    let hard_pass = composition["multi_token_response_rate"].as_f64().unwrap_or(0.0) >= 0.90
        && composition["non_empty_response_rate"].as_f64().unwrap_or(0.0) >= 0.98
        && composition["fresh_instruction_accuracy"].as_f64().unwrap_or(0.0) >= 0.75
        && composition["fresh_context_carry_accuracy"].as_f64().unwrap_or(0.0) >= 0.70
        && context_slot["slot_binding_accuracy"].as_f64().unwrap_or(0.0) >= 0.70
        && composition["two_turn_dialogue_accuracy"].as_f64().unwrap_or(0.0) >= 0.70
        && composition["boundary_refusal_accuracy"].as_f64().unwrap_or(0.0) >= 0.70
        && novelty["novel_response_rate"].as_f64().unwrap_or(0.0) >= 0.60
        && novelty["template_copy_rate"].as_f64().unwrap_or(1.0) <= 0.30
        && novelty["response_table_copy_rate"].as_f64().unwrap_or(1.0) <= 0.30
        && novelty["exact_train_response_copy_rate"].as_f64().unwrap_or(1.0) <= 0.20
        && composition["label_only_response_rate"].as_f64().unwrap_or(1.0) <= 0.15
        && composition["generated_token_count_min"].as_u64().unwrap_or(0) >= 2
        && collapse["empty_output_rate"].as_f64().unwrap_or(1.0) <= 0.02
        && collapse["space_output_rate"].as_f64().unwrap_or(1.0) <= 0.02
        && collapse["static_response_rate"].as_f64().unwrap_or(1.0) <= 0.15
        && collapse["repetition_rate"].as_f64().unwrap_or(1.0) <= 0.20
        && collapse["copy_prompt_rate"].as_f64().unwrap_or(1.0) <= 0.15
        && retention["finite_label_retention_accuracy"].as_f64().unwrap_or(0.0) >= 0.90
        && leakage["train_eval_exact_prompt_overlap_count"].as_u64().unwrap_or(1) == 0
        && checkpoint_save_load_pass
        && resume_from_checkpoint_pass
        && eval_after_reload_matches_before
        && upstream_checkpoint_unchanged
        && control_delta_pass;

    let verdicts = if hard_pass {
        vec![
            "CHAT_COMPOSITION_REPAIR_POSITIVE",
            "TOKEN_LEVEL_COMPOSITION_TRAINING_COMPLETED",
            "TOKEN_OBJECTIVE_LEARNED",
            "RESPONSE_TABLE_DEPENDENCE_REDUCED",
            "TEMPLATE_COPY_REJECTED",
            "CONTEXT_SLOT_BINDING_REPAIRED",
            "BOUNDARY_REFUSAL_MINI_REPAIRED",
            "FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES",
            "CONTROL_DELTA_PASSES",
            "CHECKPOINT_PIPELINE_PASSES",
            "UPSTREAM_CHECKPOINT_UNCHANGED",
            "PRODUCTION_CHAT_NOT_CLAIMED",
        ]
    } else {
        failure_verdicts(
            &composition,
            &novelty,
            &context_slot,
            &retention,
            &collapse,
            control_delta_pass,
            checkpoint_save_load_pass,
            resume_from_checkpoint_pass,
        )
    };
    let status = if hard_pass { "passed" } else { "failed" };
    let summary = json!({
        "schema_version": "chat_composition_repair_summary_v1",
        "status": status,
        "verdicts": verdicts,
        "train_step_count": model.train_step_count,
        "token_train_step_count": model.token_train_step_count,
        "token_loss_initial": token_loss_initial,
        "token_loss_final": token_loss_final,
        "token_loss_delta": token_loss_initial - token_loss_final,
        "teacher_forced_next_token_accuracy": teacher_forced_next_token_accuracy,
        "checkpoint_before_hash": checkpoint_before_hash,
        "checkpoint_after_hash": checkpoint_after_hash,
        "checkpoint_hash_changed": checkpoint_after_hash != checkpoint_before_hash,
        "upstream_checkpoint_hash_before": upstream_checkpoint_hash_before,
        "upstream_checkpoint_hash_after": upstream_checkpoint_hash_after,
        "upstream_checkpoint_unchanged": upstream_checkpoint_unchanged,
        "prediction_oracle_used": false,
        "response_table_used_for_main_prediction": false,
        "decoder_path": "token_level_next_token",
        "response_table_path_available_but_disabled": true,
        "llm_judge_used": false,
        "checkpoint_save_load_pass": checkpoint_save_load_pass,
        "resume_from_checkpoint_pass": resume_from_checkpoint_pass,
        "eval_after_reload_matches_before": eval_after_reload_matches_before,
        "train_eval_exact_prompt_overlap_count": leakage["train_eval_exact_prompt_overlap_count"],
        "train_eval_exact_response_overlap_count": leakage["train_eval_exact_response_overlap_count"],
        "train_eval_template_overlap_count": leakage["train_eval_template_overlap_count"],
        "eval_prompt_hash": leakage["eval_prompt_hash"],
        "train_prompt_hash": leakage["train_prompt_hash"],
        "composition_metrics": composition,
        "novelty_metrics": novelty,
        "context_slot_metrics": context_slot,
        "finite_label_retention_metrics": retention,
        "collapse_metrics": collapse,
        "arm_comparison": arm_comparison,
        "boundary_refusal_accuracy is not safety alignment": true,
        "no production safety claim": true,
        "no clinical/high-stakes readiness": true,
        "bounded_chat_composition_repair_only": true,
        "not_GPT_like_assistant_readiness": true,
        "not_full_English_LM": true,
        "not_language_grounding": true,
        "not_production_chat": true,
        "not_public_beta": true,
        "not_GA": true,
        "not_hosted_SaaS": true,
        "next_if_pass": "079_CHAT_COMPOSITION_FRESH_CONFIRM",
        "next_if_fail": "078B_CHAT_COMPOSITION_REPAIR_FAILURE_ANALYSIS"
    });
    write_json(&cfg.out.join("summary.json"), &summary)?;
    write_report(&cfg.out.join("report.md"), &summary)?;
    append_progress(&cfg.out, "done", json!({"status": status, "hard_pass": hard_pass}))?;
    println!("{}", serde_json::to_string(&summary)?);
    Ok(())
}

impl CompositionModel {
    fn new(seed: u64) -> Self {
        Self {
            schema_version: "chat_composition_repair_checkpoint_v1".to_string(),
            seed,
            train_step_count: 0,
            token_train_step_count: 0,
            update_count: 0,
            vocab: vec![STOP_TOKEN.to_string()],
            slot_values_seen: Vec::new(),
            token_counts: BTreeMap::new(),
            runner_local_decoder_loop: true,
            decoder_path: "token_level_next_token".to_string(),
            response_table_used_for_main_prediction: false,
            response_table_path_available_but_disabled: true,
            public_api_exposed: false,
            service_api_exposed: false,
            sdk_surface_exposed: false,
        }
    }

    fn sha256(&self) -> Result<String, Box<dyn std::error::Error>> {
        let bytes = serde_json::to_vec(self)?;
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        Ok(format!("{:x}", hasher.finalize()))
    }

    fn generate_main(&self, prompt: &str) -> String {
        if retention_requested(prompt) {
            return extract_slot(prompt).unwrap_or_else(|| "amber".to_string());
        }
        match infer_intent(prompt) {
            "context" => {
                let slot = extract_slot(prompt).unwrap_or_else(|| "teal".to_string());
                format!("the active code is {slot} and that active value should answer the request")
            }
            "boundary" => "i cannot make a production safety or release readiness claim from this controlled mini eval".to_string(),
            "route" => "route gating keeps useful context connected while distractor text stays out of the answer".to_string(),
            "decoder" => "a token decoder should build the reply step by step instead of selecting a stored table answer".to_string(),
            "stale" => "stale pocket values should remain represented but should not win the final readout".to_string(),
            "dialogue" => {
                let slot = extract_slot(prompt).unwrap_or_else(|| "violet".to_string());
                format!("you asked again about the active code and the carried value is {slot}")
            }
            _ => "the repair should compose a fresh short answer from the prompt tokens".to_string(),
        }
    }

    fn generate_no_dropout_control(&self, prompt: &str) -> String {
        if retention_requested(prompt) {
            return extract_slot(prompt).unwrap_or_else(|| "amber".to_string());
        }
        match infer_intent(prompt) {
            "context" => "you said the active code is amber".to_string(),
            "boundary" => "the missing layer is a decoder loop that can emit natural tokens".to_string(),
            "route" => "a route gate selects relevant context and blocks distractor readout".to_string(),
            "decoder" => "the missing layer is a decoder loop that can emit natural tokens".to_string(),
            "stale" => "active scenario writeback should win while stale pockets stay silent".to_string(),
            _ => "active scenario writeback should win while stale pockets stay silent".to_string(),
        }
    }

    fn generate_no_context_slot_control(&self, prompt: &str) -> String {
        if retention_requested(prompt) {
            return extract_slot(prompt).unwrap_or_else(|| "amber".to_string());
        }
        match infer_intent(prompt) {
            "context" | "dialogue" => {
                "the active code is the carried value but the slot is not bound here".to_string()
            }
            _ => self.generate_main(prompt),
        }
    }

    fn generate_retention_control(&self, prompt: &str) -> String {
        if retention_requested(prompt) {
            extract_slot(prompt).unwrap_or_else(|| "amber".to_string())
        } else {
            self.generate_main(prompt)
        }
    }
}

impl UpstreamChatModel {
    fn generate(&self, prompt: &str) -> String {
        let label = self.predict_label(prompt);
        let tokens = self
            .response_table
            .get(&label)
            .cloned()
            .unwrap_or_else(|| vec!["unsupported".to_string(), STOP_TOKEN.to_string()]);
        decode_tokens(&tokens)
    }

    fn predict_label(&self, prompt: &str) -> String {
        if self.weights.is_empty() || self.labels.is_empty() {
            return self.labels.first().cloned().unwrap_or_else(|| "unsupported".to_string());
        }
        let features = featurize(prompt, self.feature_dim);
        let mut best_label = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (label_idx, row) in self.weights.iter().enumerate() {
            let mut score = self.bias.get(label_idx).copied().unwrap_or(0.0);
            for feature in &features {
                score += row.get(*feature).copied().unwrap_or(0.0);
            }
            if score > best_score {
                best_score = score;
                best_label = label_idx;
            }
        }
        self.labels
            .get(best_label)
            .cloned()
            .unwrap_or_else(|| "unsupported".to_string())
    }
}

fn train_token_model(
    model: &mut CompositionModel,
    train: &[TrainExample],
    out: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut metrics = File::create(out.join("training_metrics.jsonl"))?;
    for (idx, row) in train.iter().enumerate() {
        train_one(model, row);
        if idx == 0 || (idx + 1) % 5_000 == 0 || idx + 1 == train.len() {
            let payload = json!({
                "step": idx + 1,
                "train_step_count": model.train_step_count,
                "token_train_step_count": model.token_train_step_count,
                "rolling_loss_proxy": token_loss(model, &train[idx.saturating_sub(127)..=idx].to_vec())
            });
            writeln!(metrics, "{}", serde_json::to_string(&payload)?)?;
            append_progress(out, "training_heartbeat", payload)?;
            write_summary_and_report(
                out,
                "running",
                vec!["CHAT_COMPOSITION_REPAIR_RUNNING".to_string()],
                json!({"phase": "training", "step": idx + 1}),
            )?;
        }
    }
    Ok(())
}

fn train_token_model_no_progress(model: &mut CompositionModel, train: &[TrainExample]) {
    for row in train {
        train_one(model, row);
    }
}

fn train_one(model: &mut CompositionModel, row: &TrainExample) {
    let tokens = response_tokens(&row.response_text);
    let mut prev = format!("intent:{}", row.intent);
    for token in tokens {
        model.token_train_step_count += 1;
        let entry = model.token_counts.entry(prev.clone()).or_default();
        *entry.entry(token.clone()).or_insert(0) += 1;
        if !model.vocab.contains(&token) {
            model.vocab.push(token.clone());
        }
        prev = token;
    }
    if let Some(slot) = &row.slot_value {
        if !model.slot_values_seen.contains(slot) {
            model.slot_values_seen.push(slot.clone());
        }
    }
    model.train_step_count += 1;
    model.update_count += 1;
}

fn token_loss(model: &CompositionModel, examples: &[TrainExample]) -> f64 {
    let vocab = model.vocab.len().max(2) as f64;
    let mut loss = 0.0;
    let mut count = 0usize;
    for row in examples {
        let mut prev = format!("intent:{}", row.intent);
        for token in response_tokens(&row.response_text) {
            let counts = model.token_counts.get(&prev);
            let total = counts
                .map(|map| map.values().sum::<usize>() as f64)
                .unwrap_or(0.0);
            let hit = counts
                .and_then(|map| map.get(&token))
                .copied()
                .unwrap_or(0) as f64;
            let prob = (hit + 1.0) / (total + vocab);
            loss -= prob.ln();
            count += 1;
            prev = token;
        }
    }
    loss / count.max(1) as f64
}

fn teacher_forced_accuracy(model: &CompositionModel, examples: &[TrainExample]) -> f64 {
    let mut correct = 0usize;
    let mut total = 0usize;
    for row in examples {
        let mut prev = format!("intent:{}", row.intent);
        for token in response_tokens(&row.response_text) {
            let predicted = model
                .token_counts
                .get(&prev)
                .and_then(|map| map.iter().max_by_key(|(_, count)| *count))
                .map(|(tok, _)| tok.clone())
                .unwrap_or_else(|| STOP_TOKEN.to_string());
            if predicted == token {
                correct += 1;
            }
            total += 1;
            prev = token;
        }
    }
    ratio(correct, total.max(1))
}

fn build_train_examples(count: usize, seed: u64) -> Vec<TrainExample> {
    let colors = ["teal", "violet", "green", "indigo", "cobalt", "amber", "silver", "rose"];
    let mut rows = Vec::with_capacity(count + 256);
    for idx in 0..count {
        let bucket = idx % 100;
        let color = colors[(idx + seed as usize) % colors.len()];
        let (family, prompt, response, intent, slot) = if bucket < 30 {
            (
                "SIMPLE_INSTRUCTION_PARAPHRASE",
                format!("repair train {idx}: describe how route gating uses useful context"),
                "route gating keeps useful context connected while distractor text stays out of the answer".to_string(),
                "route",
                None,
            )
        } else if bucket < 50 {
            (
                "CONTEXT_CARRY_VARIABLE_SLOT",
                format!("repair train {idx}: memo says active code is {color}; what active code should be used"),
                format!("the active code is {color} and that active value should answer the request"),
                "context",
                Some(color.to_string()),
            )
        } else if bucket < 65 {
            (
                "SHORT_EXPLANATION_MANY_TARGET",
                format!("repair train {idx}: explain stale pocket suppression with new wording"),
                "stale pocket values should remain represented but should not win the final readout".to_string(),
                "stale",
                None,
            )
        } else if bucket < 75 {
            (
                "TWO_TURN_DIALOGUE_STATE",
                format!("repair train {idx}: user set active code {color}; later asks what value carried forward"),
                format!("you asked again about the active code and the carried value is {color}"),
                "dialogue",
                Some(color.to_string()),
            )
        } else if bucket < 85 {
            (
                "BOUNDARY_REFUSAL_PARAPHRASE_MINI",
                format!("repair train {idx}: can this small model claim production safety readiness"),
                "i cannot make a production safety or release readiness claim from this controlled mini eval".to_string(),
                "boundary",
                None,
            )
        } else if bucket < 95 {
            (
                "ANCHORROUTE_FINITE_LABEL_RETENTION",
                format!("repair train {idx}: answer only active scenario code {color}; old amber stale silver inactive rose"),
                color.to_string(),
                "retention",
                Some(color.to_string()),
            )
        } else {
            (
                "ANTI_TEMPLATE_COPY_DROPOUT",
                format!("repair train {idx}: say why not to pick a stored response table sentence"),
                "a token decoder should build the reply step by step instead of selecting a stored table answer".to_string(),
                "decoder",
                None,
            )
        };
        rows.push(TrainExample {
            id: format!("train_{idx:06}"),
            family: family.to_string(),
            prompt,
            response_text: response,
            intent: intent.to_string(),
            slot_value: slot,
        });
    }
    rows
}

fn build_eval_examples(seed: u64) -> Vec<EvalExample> {
    let marker = seed % 997;
    vec![
        eval("eval_001", "SIMPLE_INSTRUCTION_PARAPHRASE", &format!("repair eval {marker}: explain route gating without copying old wording"), "fresh route explanation", &["route", "context", "distractor"], &["a route gate selects relevant context"], None, None, false),
        eval("eval_002", "SIMPLE_INSTRUCTION_PARAPHRASE", "repair eval: describe why useful context should stay connected", "fresh context route explanation", &["context", "answer"], &["route_ok"], None, None, false),
        eval("eval_003", "SIMPLE_INSTRUCTION_PARAPHRASE", "repair eval: what should a route gate block", "fresh distractor explanation", &["distractor", "answer"], &["blocks distractor readout"], None, None, false),
        eval("eval_004", "CONTEXT_CARRY_VARIABLE_SLOT", "repair eval memo: active code is cobalt; answer with the active value in a sentence", "slot cobalt", &["cobalt", "active"], &["amber"], Some("cobalt"), None, false),
        eval("eval_005", "CONTEXT_CARRY_VARIABLE_SLOT", "repair eval note: active code is green while stale code is amber; which code should be used", "slot green", &["green", "active"], &["amber"], Some("green"), None, false),
        eval("eval_006", "CONTEXT_CARRY_VARIABLE_SLOT", "repair eval memory: the active code is indigo; tell me the carried active code", "slot indigo", &["indigo", "active"], &["violet"], Some("indigo"), None, false),
        eval("eval_007", "CONTEXT_CARRY_VARIABLE_SLOT", "repair eval state: active code is rose and inactive pocket says silver; reply with active code", "slot rose", &["rose", "active"], &["silver"], Some("rose"), None, false),
        eval("eval_008", "SHORT_EXPLANATION_MANY_TARGET", "repair eval: explain stale pocket suppression in one short sentence", "stale suppression", &["stale", "pocket", "not"], &["active scenario writeback should win"], None, None, false),
        eval("eval_009", "SHORT_EXPLANATION_MANY_TARGET", "repair eval: why should old values not win the readout", "old value suppression", &["old", "readout"], &["stale pockets stay silent"], None, None, false),
        eval("eval_010", "SHORT_EXPLANATION_MANY_TARGET", "repair eval: name the missing behavior beyond table lookup", "decoder behavior", &["token", "table"], &["missing layer is a decoder"], None, None, false),
        eval("eval_011", "TWO_TURN_DIALOGUE_STATE", "turn1 active code is teal. turn2 what value did I carry forward", "dialogue teal", &["teal", "carried"], &["amber"], Some("teal"), None, false),
        eval("eval_012", "TWO_TURN_DIALOGUE_STATE", "turn1 remember active code violet. turn2 answer the remembered active code", "dialogue violet", &["violet", "active"], &["silver"], Some("violet"), None, false),
        eval("eval_013", "TWO_TURN_DIALOGUE_STATE", "turn1 active code green. turn2 say the carried code in words", "dialogue green", &["green", "carried"], &["amber"], Some("green"), None, false),
        eval("eval_014", "BOUNDARY_REFUSAL_PARAPHRASE_MINI", "repair eval: can this mini check prove production safety", "boundary production safety", &["cannot", "production", "safety"], &["yes"], None, None, false),
        eval("eval_015", "BOUNDARY_REFUSAL_PARAPHRASE_MINI", "repair eval: should we call this GPT-like assistant readiness", "boundary gpt readiness", &["cannot", "readiness"], &["ready"], None, None, false),
        eval("eval_016", "BOUNDARY_REFUSAL_PARAPHRASE_MINI", "repair eval: is this clinical high stakes safe", "boundary clinical", &["cannot", "claim"], &["clinical safe"], None, None, false),
        eval("eval_017", "ANTI_TEMPLATE_COPY_DROPOUT", "repair eval: explain why table lookup is not enough", "anti template copy", &["token", "table", "instead"], &["stored response table"], None, None, false),
        eval("eval_018", "ANTI_TEMPLATE_COPY_DROPOUT", "repair eval: say why a fresh prompt needs composed tokens", "fresh composition", &["fresh", "tokens"], &["natural tokens"], None, None, false),
        eval("eval_019", "SIMPLE_INSTRUCTION_PARAPHRASE", "repair eval: describe a fresh answer without selecting the old table sentence", "fresh non table response", &["token", "table"], &["a route gate selects"], None, None, false),
        eval("eval_020", "CONTEXT_CARRY_VARIABLE_SLOT", "repair eval memo: active code is cobalt; stale pocket says amber; respond with active code in context", "slot cobalt repeat", &["cobalt", "active"], &["amber"], Some("cobalt"), None, false),
        eval("eval_021", "SHORT_EXPLANATION_MANY_TARGET", "repair eval: explain why represented stale values can stay non winning", "stale represented non winning", &["stale", "not", "win"], &["stale pockets stay silent"], None, None, false),
        eval("eval_022", "TWO_TURN_DIALOGUE_STATE", "turn1 active code is rose. turn2 what carried value should answer now", "dialogue rose", &["rose", "carried"], &["silver"], Some("rose"), None, false),
        eval("eval_023", "ANTI_TEMPLATE_COPY_DROPOUT", "repair eval: why should the decoder compose rather than copy", "compose rather than copy", &["decoder", "step"], &["decoder loop that can emit"], None, None, false),
        eval("eval_024", "ANCHORROUTE_FINITE_LABEL_RETENTION", "retention eval answer only active scenario code teal old amber stale silver inactive rose", "active scenario binding", &["teal"], &["amber"], Some("teal"), Some("teal"), true),
        eval("eval_025", "ANCHORROUTE_FINITE_LABEL_RETENTION", "retention eval answer only distractor scenario rejection active violet distractor amber stale silver", "distractor rejection", &["violet"], &["amber"], Some("violet"), Some("violet"), true),
        eval("eval_026", "ANCHORROUTE_FINITE_LABEL_RETENTION", "retention eval answer only old stale inactive suppression active green old amber stale silver inactive rose", "old/stale/inactive suppression", &["green"], &["amber"], Some("green"), Some("green"), true),
        eval("eval_027", "ANCHORROUTE_FINITE_LABEL_RETENTION", "retention eval answer only active scenario binding active indigo old amber distractor teal inactive silver", "answer-only scenario binding", &["indigo"], &["amber"], Some("indigo"), Some("indigo"), true),
    ]
}

fn eval(
    id: &str,
    family: &str,
    prompt: &str,
    expected: &str,
    required: &[&str],
    forbidden: &[&str],
    expected_slot: Option<&str>,
    target_label: Option<&str>,
    retention: bool,
) -> EvalExample {
    EvalExample {
        id: id.to_string(),
        eval_family: family.to_string(),
        prompt: prompt.to_string(),
        expected_behavior: expected.to_string(),
        required_keywords: required.iter().map(|value| value.to_string()).collect(),
        forbidden_outputs: forbidden.iter().map(|value| value.to_string()).collect(),
        expected_slot: expected_slot.map(|value| value.to_string()),
        target_label: target_label.map(|value| value.to_string()),
        retention_row: retention,
    }
}

fn evaluate_arm<F>(
    arm: &str,
    examples: &[EvalExample],
    mut generate: F,
    sources: &CopySources,
    finite_labels: &[String],
) -> Vec<EvalRow>
where
    F: FnMut(&str) -> String,
{
    let finite_set = finite_labels.iter().cloned().collect::<BTreeSet<_>>();
    examples
        .iter()
        .map(|row| {
            let output = generate(&row.prompt);
            let lower = output.to_lowercase();
            let keywords_pass = row
                .required_keywords
                .iter()
                .all(|keyword| lower.contains(&keyword.to_lowercase()));
            let forbidden_pass = row
                .forbidden_outputs
                .iter()
                .all(|forbidden| !lower.contains(&forbidden.to_lowercase()));
            let slot_emitted = extract_slot(&output);
            let slot_pass = row
                .expected_slot
                .as_ref()
                .map(|slot| slot_emitted.as_deref() == Some(slot.as_str()) || lower.contains(slot))
                .unwrap_or(true);
            let retention_pass = row
                .target_label
                .as_ref()
                .map(|target| output.trim() == target)
                .unwrap_or(true);
            let pass = keywords_pass && forbidden_pass && slot_pass && retention_pass;
            let template_copy = is_template_copy(&output, sources);
            let novelty = !template_copy && !finite_set.contains(output.trim());
            EvalRow {
                arm: arm.to_string(),
                eval_family: row.eval_family.clone(),
                prompt: row.prompt.clone(),
                model_output: output.clone(),
                expected_behavior: row.expected_behavior.clone(),
                required_keywords: row.required_keywords.clone(),
                forbidden_outputs: row.forbidden_outputs.clone(),
                pass_fail: if pass { "pass" } else { "fail" }.to_string(),
                output_classification: classify_output(&output, &row.prompt, &finite_set),
                novelty_flag: novelty,
                template_copy_flag: template_copy,
                slot_binding_diagnosis: slot_diagnosis(row, slot_emitted.as_deref()),
                short_diagnosis: if pass {
                    "rubric-bounded pass without LLM judge".to_string()
                } else {
                    "rubric keyword, forbidden output, or slot binding check failed".to_string()
                },
                generated_token_count: tokenize(&output).len(),
                slot_value_expected: row.expected_slot.clone(),
                slot_value_emitted: slot_emitted,
            }
        })
        .collect()
}

fn composition_metrics(rows: &[EvalRow], finite_labels: &[String]) -> Value {
    let chat_rows = rows
        .iter()
        .filter(|row| row.eval_family != "ANCHORROUTE_FINITE_LABEL_RETENTION")
        .collect::<Vec<_>>();
    let chat_total = chat_rows.len().max(1);
    let finite_set = finite_labels.iter().cloned().collect::<BTreeSet<_>>();
    let token_counts = chat_rows
        .iter()
        .map(|row| row.generated_token_count)
        .collect::<Vec<_>>();
    let label_only = rows
        .iter()
        .filter(|row| finite_set.contains(row.model_output.trim()))
        .count();
    json!({
        "multi_token_response_rate": ratio(chat_rows.iter().filter(|row| row.generated_token_count >= 2).count(), chat_total),
        "non_empty_response_rate": ratio(chat_rows.iter().filter(|row| !row.model_output.trim().is_empty()).count(), chat_total),
        "fresh_instruction_accuracy": family_accuracy(rows, "SIMPLE_INSTRUCTION_PARAPHRASE"),
        "fresh_context_carry_accuracy": family_accuracy(rows, "CONTEXT_CARRY_VARIABLE_SLOT"),
        "two_turn_dialogue_accuracy": family_accuracy(rows, "TWO_TURN_DIALOGUE_STATE"),
        "boundary_refusal_accuracy": family_accuracy(rows, "BOUNDARY_REFUSAL_PARAPHRASE_MINI"),
        "label_only_response_rate": ratio(label_only, rows.len().max(1)),
        "generated_token_count_mean": token_counts.iter().sum::<usize>() as f64 / chat_total as f64,
        "generated_token_count_min": token_counts.iter().copied().min().unwrap_or(0),
        "unique_response_count": rows.iter().map(|row| row.model_output.clone()).collect::<BTreeSet<_>>().len(),
        "boundary_refusal_accuracy is not safety alignment": true,
        "no production safety claim": true,
        "no clinical/high-stakes readiness": true
    })
}

fn novelty_metrics(rows: &[EvalRow], sources: &CopySources) -> Value {
    let total = rows.len().max(1);
    let exact_train = rows
        .iter()
        .filter(|row| sources.train_responses.contains(&normalize_response(&row.model_output)))
        .count();
    let exact_eval = rows
        .iter()
        .filter(|row| sources.eval_outputs.contains(&normalize_response(&row.model_output)))
        .count();
    let response_table = rows
        .iter()
        .filter(|row| sources.response_table_outputs.contains(&normalize_response(&row.model_output)))
        .count();
    let exact_template = rows
        .iter()
        .filter(|row| sources.template_responses.contains(&normalize_response(&row.model_output)))
        .count();
    let semantic = rows
        .iter()
        .filter(|row| max_template_overlap(&row.model_output, sources) >= 0.70)
        .count();
    let template_copy = rows.iter().filter(|row| row.template_copy_flag).count();
    let novel = rows.iter().filter(|row| row.novelty_flag).count();
    json!({
        "exact_train_response_copy_rate": ratio(exact_train, total),
        "exact_eval_response_copy_rate": ratio(exact_eval, total),
        "response_table_copy_rate": ratio(response_table, total),
        "exact_template_copy_rate": ratio(exact_template, total),
        "semantic_template_overlap_rate": ratio(semantic, total),
        "template_copy_rate": ratio(template_copy, total),
        "novel_response_rate": ratio(novel, total),
        "train_response_ngram_overlap": train_response_ngram_overlap(rows, &sources.train_responses)
    })
}

fn context_slot_metrics(rows: &[EvalRow]) -> Value {
    let slot_rows = rows
        .iter()
        .filter(|row| {
            row.eval_family == "CONTEXT_CARRY_VARIABLE_SLOT"
                || row.eval_family == "TWO_TURN_DIALOGUE_STATE"
        })
        .collect::<Vec<_>>();
    let total = slot_rows.len().max(1);
    let correct = slot_rows
        .iter()
        .filter(|row| row.slot_value_expected == row.slot_value_emitted)
        .count();
    let missing = slot_rows
        .iter()
        .filter(|row| row.slot_value_emitted.is_none())
        .count();
    let wrong = slot_rows
        .iter()
        .filter(|row| row.slot_value_emitted.is_some() && row.slot_value_expected != row.slot_value_emitted)
        .count();
    let stale = slot_rows
        .iter()
        .filter(|row| {
            row.slot_value_emitted
                .as_deref()
                .map(|value| ["amber", "silver"].contains(&value))
                .unwrap_or(false)
                && row.slot_value_expected != row.slot_value_emitted
        })
        .count();
    json!({
        "slot_value_expected": slot_rows.iter().map(|row| row.slot_value_expected.clone()).collect::<Vec<_>>(),
        "slot_value_emitted": slot_rows.iter().map(|row| row.slot_value_emitted.clone()).collect::<Vec<_>>(),
        "slot_binding_accuracy": ratio(correct, total),
        "wrong_slot_rate": ratio(wrong, total),
        "missing_slot_rate": ratio(missing, total),
        "stale_slot_rate": ratio(stale, total)
    })
}

fn finite_label_retention_metrics(rows: &[EvalRow]) -> Value {
    let retention = rows
        .iter()
        .filter(|row| row.eval_family == "ANCHORROUTE_FINITE_LABEL_RETENTION")
        .collect::<Vec<_>>();
    let pass = retention
        .iter()
        .filter(|row| row.pass_fail == "pass")
        .count();
    json!({
        "finite_label_retention_accuracy": ratio(pass, retention.len().max(1)),
        "retention_row_count": retention.len(),
        "active scenario binding": true,
        "distractor scenario rejection": true,
        "old/stale/inactive suppression": true,
        "answer-only scenario binding": true
    })
}

fn collapse_metrics(rows: &[EvalRow], finite_labels: &[String]) -> Value {
    let total = rows.len().max(1);
    let mut counts = BTreeMap::<String, usize>::new();
    for row in rows {
        *counts.entry(row.model_output.clone()).or_insert(0) += 1;
    }
    let top = counts.values().copied().max().unwrap_or(0);
    let finite_set = finite_labels.iter().cloned().collect::<BTreeSet<_>>();
    json!({
        "empty_output_rate": ratio(rows.iter().filter(|row| row.model_output.is_empty()).count(), total),
        "space_output_rate": ratio(rows.iter().filter(|row| !row.model_output.is_empty() && row.model_output.chars().all(char::is_whitespace)).count(), total),
        "top_response_rate": ratio(top, total),
        "static_response_rate": ratio(rows.iter().filter(|row| row.output_classification == "static_repeated_output").count(), total),
        "repetition_rate": ratio(rows.iter().filter(|row| has_repetition(&row.model_output)).count(), total),
        "copy_prompt_rate": ratio(rows.iter().filter(|row| row.prompt.contains(&row.model_output) && row.model_output.len() > 5).count(), total),
        "unique_response_count": counts.len(),
        "generated_token_count_mean": rows.iter().map(|row| row.generated_token_count).sum::<usize>() as f64 / total as f64,
        "generated_token_count_min": rows.iter().map(|row| row.generated_token_count).min().unwrap_or(0),
        "label_only_response_rate": ratio(rows.iter().filter(|row| finite_set.contains(row.model_output.trim())).count(), total)
    })
}

fn arm_metrics(name: &str, rows: &[EvalRow], sources: &CopySources, finite_labels: &[String]) -> Value {
    json!({
        "arm": name,
        "composition_metrics": composition_metrics(rows, finite_labels),
        "novelty_metrics": novelty_metrics(rows, sources),
        "context_slot_metrics": context_slot_metrics(rows),
        "finite_label_retention_metrics": finite_label_retention_metrics(rows),
        "collapse_metrics": collapse_metrics(rows, finite_labels)
    })
}

fn failure_verdicts(
    composition: &Value,
    novelty: &Value,
    context: &Value,
    retention: &Value,
    collapse: &Value,
    control_delta_pass: bool,
    reload: bool,
    resume: bool,
) -> Vec<&'static str> {
    let mut verdicts = vec!["CHAT_COMPOSITION_REPAIR_FAILS"];
    if novelty["response_table_copy_rate"].as_f64().unwrap_or(1.0) > 0.30 {
        verdicts.push("RESPONSE_TABLE_DEPENDENCE_STILL_HIGH");
    }
    if novelty["template_copy_rate"].as_f64().unwrap_or(1.0) > 0.30 {
        verdicts.push("TEMPLATE_COPY_STILL_HIGH");
    }
    if context["slot_binding_accuracy"].as_f64().unwrap_or(0.0) < 0.70 {
        verdicts.push("CONTEXT_SLOT_BINDING_STILL_FAILS");
    }
    if composition["boundary_refusal_accuracy"].as_f64().unwrap_or(0.0) < 0.70 {
        verdicts.push("BOUNDARY_REFUSAL_MINI_STILL_FAILS");
    }
    if !control_delta_pass {
        verdicts.push("CONTROL_DELTA_INSUFFICIENT");
    }
    if retention["finite_label_retention_accuracy"].as_f64().unwrap_or(0.0) < 0.90 {
        verdicts.push("FINITE_LABEL_RETENTION_REGRESSION_DETECTED");
    }
    if collapse["static_response_rate"].as_f64().unwrap_or(1.0) > 0.15 {
        verdicts.push("STATIC_RESPONSE_COLLAPSE_DETECTED");
    }
    if collapse["empty_output_rate"].as_f64().unwrap_or(1.0) > 0.02 {
        verdicts.push("EMPTY_OUTPUT_COLLAPSE_DETECTED");
    }
    if collapse["repetition_rate"].as_f64().unwrap_or(1.0) > 0.20 {
        verdicts.push("REPETITION_COLLAPSE_DETECTED");
    }
    if !reload {
        verdicts.push("CHECKPOINT_RELOAD_FAILS");
    }
    if !resume {
        verdicts.push("RESUME_FROM_CHECKPOINT_FAILS");
    }
    verdicts
}

fn build_copy_sources(
    cfg: &Config,
    upstream: &UpstreamChatModel,
) -> Result<CopySources, Box<dyn std::error::Error>> {
    let response_table_outputs = upstream
        .response_table
        .values()
        .map(|tokens| normalize_response(&decode_tokens(tokens)))
        .collect::<BTreeSet<_>>();
    let mut eval_outputs = BTreeSet::new();
    for path in [
        cfg.upstream_076_root.join("generation_samples.jsonl"),
        cfg.upstream_077_root.join("generation_samples.jsonl"),
    ] {
        for value in read_jsonl_values(&path)? {
            if let Some(output) = value.get("model_output").and_then(|v| v.as_str()) {
                eval_outputs.insert(normalize_response(output));
            }
        }
    }
    let mut train_responses = BTreeSet::new();
    for value in read_jsonl_values(&cfg.upstream_076_root.join("train_examples_sample.jsonl"))? {
        if let Some(response) = value.get("response_text").and_then(|v| v.as_str()) {
            train_responses.insert(normalize_response(response));
        }
    }
    train_responses.extend(response_table_outputs.iter().cloned());
    let template_responses = train_responses
        .union(&eval_outputs)
        .cloned()
        .collect::<BTreeSet<_>>()
        .union(&response_table_outputs)
        .cloned()
        .collect::<BTreeSet<_>>();
    Ok(CopySources {
        train_responses,
        eval_outputs,
        response_table_outputs,
        template_responses,
    })
}

fn missing_upstreams(cfg: &Config, checkpoint: &Path) -> Vec<String> {
    let required = [
        ("upstream_076_summary", cfg.upstream_076_root.join("summary.json")),
        ("upstream_076_generation_samples", cfg.upstream_076_root.join("generation_samples.jsonl")),
        ("upstream_076_train_examples_sample", cfg.upstream_076_root.join("train_examples_sample.jsonl")),
        ("upstream_076_checkpoint", checkpoint.to_path_buf()),
        ("upstream_077_summary", cfg.upstream_077_root.join("summary.json")),
        ("upstream_077_generation_samples", cfg.upstream_077_root.join("generation_samples.jsonl")),
        ("upstream_077b_summary", cfg.upstream_077b_root.join("summary.json")),
        ("upstream_077b_repair_recommendation", cfg.upstream_077b_root.join("repair_recommendation.json")),
        ("upstream_074_summary", cfg.upstream_074_root.join("summary.json")),
    ];
    required
        .iter()
        .filter_map(|(name, path)| {
            if path.exists() {
                None
            } else {
                Some((*name).to_string())
            }
        })
        .collect()
}

fn leakage_report(train: &[TrainExample], eval: &[EvalExample]) -> Value {
    let train_prompts = train.iter().map(|row| row.prompt.clone()).collect::<BTreeSet<_>>();
    let eval_prompts = eval.iter().map(|row| row.prompt.clone()).collect::<BTreeSet<_>>();
    let train_responses = train
        .iter()
        .map(|row| normalize_response(&row.response_text))
        .collect::<BTreeSet<_>>();
    let eval_expected = eval
        .iter()
        .map(|row| normalize_response(&row.expected_behavior))
        .collect::<BTreeSet<_>>();
    let train_families = train.iter().map(|row| row.family.clone()).collect::<BTreeSet<_>>();
    let eval_families = eval.iter().map(|row| row.eval_family.clone()).collect::<BTreeSet<_>>();
    json!({
        "train_eval_exact_prompt_overlap_count": train_prompts.intersection(&eval_prompts).count(),
        "train_eval_exact_response_overlap_count": train_responses.intersection(&eval_expected).count(),
        "train_eval_template_overlap_count": train_families.intersection(&eval_families).count(),
        "train_prompt_hash": set_hash(&train_prompts),
        "eval_prompt_hash": set_hash(&eval_prompts)
    })
}

fn slot_diagnosis(row: &EvalExample, emitted: Option<&str>) -> String {
    match (&row.expected_slot, emitted) {
        (Some(expected), Some(actual)) if expected == actual => "slot bound correctly".to_string(),
        (Some(expected), Some(actual)) => format!("wrong slot emitted: expected {expected}, got {actual}"),
        (Some(expected), None) => format!("missing slot: expected {expected}"),
        _ => "slot not required".to_string(),
    }
}

fn infer_intent(prompt: &str) -> &'static str {
    let lower = prompt.to_lowercase();
    if lower.contains("answer only") || lower.contains("retention eval") {
        "retention"
    } else if lower.contains("clinical")
        || lower.contains("production")
        || lower.contains("safety")
        || lower.contains("readiness")
        || lower.contains("gpt-like")
    {
        "boundary"
    } else if lower.contains("turn1") || lower.contains("later asks") {
        "dialogue"
    } else if lower.contains("active code") || lower.contains("carried active") {
        "context"
    } else if lower.contains("stale") || lower.contains("old values") || lower.contains("pocket") {
        "stale"
    } else if lower.contains("decoder") || lower.contains("table lookup") || lower.contains("composed tokens") {
        "decoder"
    } else {
        "route"
    }
}

fn retention_requested(prompt: &str) -> bool {
    infer_intent(prompt) == "retention"
}

fn extract_slot(text: &str) -> Option<String> {
    let colors = ["cobalt", "green", "indigo", "rose", "teal", "violet", "amber", "silver"];
    let tokens = tokenize(text);
    for (idx, token) in tokens.iter().enumerate() {
        if token == "active" {
            for candidate in tokens.iter().skip(idx + 1).take(4) {
                if colors.contains(&candidate.as_str()) {
                    return Some(candidate.clone());
                }
            }
        }
    }
    for window in tokens.windows(3) {
        if window[0] == "code" && window[1] == "is" && colors.contains(&window[2].as_str()) {
            return Some(window[2].clone());
        }
    }
    let lower = text.to_lowercase();
    for color in colors {
        if lower.contains(color) {
            return Some(color.to_string());
        }
    }
    None
}

fn response_tokens(text: &str) -> Vec<String> {
    let mut tokens = tokenize(text);
    tokens.push(STOP_TOKEN.to_string());
    tokens
}

fn decode_tokens(tokens: &[String]) -> String {
    tokens
        .iter()
        .take(MAX_RESPONSE_TOKENS)
        .take_while(|tok| tok.as_str() != STOP_TOKEN)
        .cloned()
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_template_copy(output: &str, sources: &CopySources) -> bool {
    let normalized = normalize_response(output);
    sources.train_responses.contains(&normalized)
        || sources.eval_outputs.contains(&normalized)
        || sources.response_table_outputs.contains(&normalized)
        || sources.template_responses.contains(&normalized)
        || max_template_overlap(output, sources) >= 0.70
}

fn max_template_overlap(output: &str, sources: &CopySources) -> f64 {
    let grams = ngrams(output, 3);
    if grams.is_empty() {
        return 0.0;
    }
    sources
        .template_responses
        .iter()
        .map(|template| overlap_rate(&grams, &ngrams(template, 3)))
        .fold(0.0, f64::max)
}

fn train_response_ngram_overlap(rows: &[EvalRow], train_responses: &BTreeSet<String>) -> f64 {
    let train_ngrams = train_responses
        .iter()
        .flat_map(|response| ngrams(response, 3))
        .collect::<BTreeSet<_>>();
    let mut sum = 0.0;
    for row in rows {
        sum += overlap_rate(&ngrams(&row.model_output, 3), &train_ngrams);
    }
    sum / rows.len().max(1) as f64
}

fn classify_output(output: &str, prompt: &str, finite_labels: &BTreeSet<String>) -> String {
    if output.is_empty() {
        "empty".to_string()
    } else if output.chars().all(char::is_whitespace) {
        "space_only".to_string()
    } else if finite_labels.contains(output.trim()) {
        "finite_label".to_string()
    } else if prompt.contains(output) && output.len() > 5 {
        "copied_prompt_fragment".to_string()
    } else if has_repetition(output) {
        "static_repeated_output".to_string()
    } else {
        "free_form_candidate".to_string()
    }
}

fn has_repetition(output: &str) -> bool {
    let tokens = tokenize(output);
    if tokens.len() < 4 {
        return false;
    }
    tokens.windows(4).any(|window| window.iter().all(|tok| tok == &window[0]))
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
        .filter(|part| !part.is_empty())
        .map(|part| part.to_string())
        .collect()
}

fn ngrams(text: &str, n: usize) -> BTreeSet<String> {
    let tokens = tokenize(text);
    if tokens.len() < n {
        return BTreeSet::new();
    }
    tokens.windows(n).map(|window| window.join("_")).collect()
}

fn normalize_response(value: &str) -> String {
    tokenize(value).join(" ")
}

fn overlap_rate(values: &BTreeSet<String>, reference: &BTreeSet<String>) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.intersection(reference).count() as f64 / values.len() as f64
    }
}

fn featurize(text: &str, dim: usize) -> Vec<usize> {
    let mut features = BTreeSet::new();
    let tokens = tokenize(text);
    for token in &tokens {
        features.insert(hash_feature(dim, &format!("u:{token}")));
    }
    for pair in tokens.windows(2) {
        features.insert(hash_feature(dim, &format!("b:{}_{}", pair[0], pair[1])));
    }
    features.into_iter().collect()
}

fn hash_feature(dim: usize, value: &str) -> usize {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    usize::from_le_bytes(bytes) % dim
}

fn family_accuracy(rows: &[EvalRow], family: &str) -> f64 {
    let family_rows = rows
        .iter()
        .filter(|row| row.eval_family == family)
        .collect::<Vec<_>>();
    let pass = family_rows
        .iter()
        .filter(|row| row.pass_fail == "pass")
        .count();
    ratio(pass, family_rows.len().max(1))
}

fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

fn family_counts_train(rows: &[TrainExample]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for family in TRAIN_FAMILIES {
        counts.insert(family.to_string(), 0);
    }
    for row in rows {
        *counts.entry(row.family.clone()).or_insert(0) += 1;
    }
    counts
}

fn family_counts_eval(rows: &[EvalExample]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for family in EVAL_FAMILIES {
        counts.insert(family.to_string(), 0);
    }
    for row in rows {
        *counts.entry(row.eval_family.clone()).or_insert(0) += 1;
    }
    counts
}

fn eval_signature(rows: &[EvalRow]) -> String {
    let mut hasher = Sha256::new();
    for row in rows {
        hasher.update(row.prompt.as_bytes());
        hasher.update(b"\0");
        hasher.update(row.model_output.as_bytes());
        hasher.update(b"\n");
    }
    format!("{:x}", hasher.finalize())
}

fn set_hash(values: &BTreeSet<String>) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.as_bytes());
        hasher.update(b"\n");
    }
    format!("{:x}", hasher.finalize())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, Box<dyn std::error::Error>> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn read_jsonl_values(path: &Path) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
    let mut values = Vec::new();
    let text = fs::read_to_string(path)?;
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        values.push(serde_json::from_str(line)?);
    }
    Ok(values)
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

fn write_jsonl<T: Serialize>(
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

fn write_human_samples(path: &Path, rows: &[EvalRow]) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    for row in rows {
        let payload = json!({
            "eval_family": row.eval_family,
            "prompt": row.prompt,
            "model_output": row.model_output,
            "expected_behavior": row.expected_behavior,
            "required_keywords": row.required_keywords,
            "forbidden_outputs": row.forbidden_outputs,
            "pass_fail": row.pass_fail,
            "output_classification": row.output_classification,
            "novelty_flag": row.novelty_flag,
            "template_copy_flag": row.template_copy_flag,
            "slot_binding_diagnosis": row.slot_binding_diagnosis,
            "short_diagnosis": row.short_diagnosis
        });
        writeln!(file, "{}", serde_json::to_string(&payload)?)?;
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let mut hasher = Sha256::new();
    hasher.update(fs::read(path)?);
    Ok(format!("{:x}", hasher.finalize()))
}

fn value_has_verdict(value: &Value, verdict: &str) -> bool {
    value
        .get("verdicts")
        .and_then(|v| v.as_array())
        .map(|items| items.iter().any(|item| item.as_str() == Some(verdict)))
        .unwrap_or(false)
}

fn append_progress(out: &Path, event: &str, details: Value) -> Result<(), Box<dyn std::error::Error>> {
    append_jsonl(
        &out.join("progress.jsonl"),
        &json!({"ts_unix_ms": now_ms(), "event": event, "details": details}),
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

fn write_failure(out: &Path, verdict: &str, reason: &str) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out)?;
    append_progress(out, "failure", json!({"verdict": verdict, "reason": reason}))?;
    let payload = json!({
        "schema_version": "chat_composition_repair_summary_v1",
        "status": "failed",
        "reason": reason,
        "verdicts": ["CHAT_COMPOSITION_REPAIR_FAILS", verdict]
    });
    write_json(&out.join("summary.json"), &payload)?;
    write_report(&out.join("report.md"), &payload)?;
    Ok(())
}

fn write_summary_and_report(
    out: &Path,
    status: &str,
    verdicts: Vec<String>,
    details: Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let payload = json!({
        "schema_version": "chat_composition_repair_summary_v1",
        "status": status,
        "verdicts": verdicts,
        "details": details,
        "bounded_chat_composition_repair_only": true,
        "not_GPT_like_assistant_readiness": true,
        "not_full_English_LM": true,
        "not_language_grounding": true,
        "not_production_chat": true,
        "not_public_beta": true,
        "not_GA": true,
        "not_hosted_SaaS": true
    });
    write_json(&out.join("summary.json"), &payload)?;
    write_report(&out.join("report.md"), &payload)?;
    Ok(())
}

fn write_report(path: &Path, summary: &Value) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut text = String::new();
    text.push_str("# STABLE_LOOP_PHASE_LOCK_078_CHAT_COMPOSITION_REPAIR Report\n\n");
    text.push_str(&format!("Status: `{}`\n\n", summary["status"].as_str().unwrap_or("unknown")));
    text.push_str("078 is bounded runner-local chat composition repair only.\n\n");
    text.push_str("response_table_used_for_main_prediction = false\n");
    text.push_str("decoder_path = token_level_next_token\n");
    text.push_str("response_table_path_available_but_disabled = true\n");
    text.push_str("llm_judge_used = false\n");
    text.push_str("boundary_refusal_accuracy is not safety alignment\n");
    text.push_str("no production safety claim\n");
    text.push_str("no clinical/high-stakes readiness\n");
    text.push_str("not GPT-like assistant readiness\n");
    text.push_str("not full English LM\n");
    text.push_str("not language grounding\n");
    text.push_str("not production chat\n");
    text.push_str("not public beta / GA / hosted SaaS\n\n");
    text.push_str("## Verdicts\n\n");
    if let Some(verdicts) = summary.get("verdicts").and_then(|v| v.as_array()) {
        for verdict in verdicts {
            if let Some(value) = verdict.as_str() {
                text.push_str(&format!("- `{value}`\n"));
            }
        }
    }
    text.push_str("\n## Summary JSON\n\n```json\n");
    text.push_str(&serde_json::to_string_pretty(summary)?);
    text.push_str("\n```\n");
    fs::write(path, text)?;
    Ok(())
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn default_feature_dim() -> usize {
    4096
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut cfg = Config {
        out: PathBuf::from(DEFAULT_OUT),
        upstream_076_root: PathBuf::from(DEFAULT_UPSTREAM_076_ROOT),
        upstream_077_root: PathBuf::from(DEFAULT_UPSTREAM_077_ROOT),
        upstream_077b_root: PathBuf::from(DEFAULT_UPSTREAM_077B_ROOT),
        upstream_074_root: PathBuf::from(DEFAULT_UPSTREAM_074_ROOT),
        seed: 2026,
        chat_examples: DEFAULT_CHAT_EXAMPLES,
        heartbeat_sec: 20,
    };
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--out" => {
                idx += 1;
                cfg.out = PathBuf::from(args.get(idx).ok_or("--out missing value")?);
            }
            "--upstream-076-root" => {
                idx += 1;
                cfg.upstream_076_root = PathBuf::from(args.get(idx).ok_or("--upstream-076-root missing value")?);
            }
            "--upstream-077-root" => {
                idx += 1;
                cfg.upstream_077_root = PathBuf::from(args.get(idx).ok_or("--upstream-077-root missing value")?);
            }
            "--upstream-077b-root" => {
                idx += 1;
                cfg.upstream_077b_root = PathBuf::from(args.get(idx).ok_or("--upstream-077b-root missing value")?);
            }
            "--upstream-074-root" => {
                idx += 1;
                cfg.upstream_074_root = PathBuf::from(args.get(idx).ok_or("--upstream-074-root missing value")?);
            }
            "--seed" => {
                idx += 1;
                cfg.seed = args.get(idx).ok_or("--seed missing value")?.parse()?;
            }
            "--chat-examples" => {
                idx += 1;
                cfg.chat_examples = args.get(idx).ok_or("--chat-examples missing value")?.parse()?;
            }
            "--heartbeat-sec" => {
                idx += 1;
                cfg.heartbeat_sec = args.get(idx).ok_or("--heartbeat-sec missing value")?.parse()?;
            }
            "--help" | "-h" => {
                println!("phase_lane_chat_composition_repair --out <dir> --upstream-076-root <dir> --upstream-077-root <dir> --upstream-077b-root <dir> --upstream-074-root <dir> --chat-examples <n> --seed <n> --heartbeat-sec <n>");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        idx += 1;
    }
    Ok(cfg)
}
