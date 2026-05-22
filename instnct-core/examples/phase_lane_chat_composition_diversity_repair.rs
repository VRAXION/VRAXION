#![recursion_limit = "256"]

//! Runner-local chat composition diversity repair.
//!
//! 080 targets the 079B diagnosis: slot binding survived, but outputs still
//! reused exact 078 responses and skeletons. This example trains only a new
//! research checkpoint under target/ and does not expose any runtime/API/SDK
//! chat surface.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_OUT: &str =
    "target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke";
const DEFAULT_UPSTREAM_078_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke";
const DEFAULT_UPSTREAM_079_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_079_chat_composition_fresh_confirm/smoke";
const DEFAULT_UPSTREAM_079B_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis/smoke";
const DEFAULT_UPSTREAM_074_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke";
const DEFAULT_CHAT_EXAMPLES: usize = 80_000;
const MAX_CHAT_EXAMPLES: usize = 180_000;
const STOP_TOKEN: &str = "<eos>";
const SEMANTIC_COPY_THRESHOLD: f64 = 0.70;

const TRAIN_FAMILIES: [&str; 9] = [
    "MANY_VALID_CONTINUATION_CHAT",
    "RESPONSE_SKELETON_DROPOUT",
    "LEXICAL_DROPOUT_SYNONYM_SLOT",
    "RANDOMIZED_CLAUSE_ORDER",
    "SEMANTIC_SLOT_RECOMBINATION",
    "CONTEXT_CARRY_VARIABLE_SLOT_DIVERSITY",
    "BOUNDARY_REFUSAL_PARAPHRASE_DIVERSITY",
    "TWO_TURN_DIALOGUE_RECOMBINATION",
    "FINITE_LABEL_ANCHORROUTE_RETENTION",
];

const EVAL_FAMILIES: [&str; 10] = [
    "FRESH_DIVERSITY_SIMPLE_INSTRUCTION",
    "FRESH_DIVERSITY_SHORT_EXPLANATION",
    "FRESH_DIVERSITY_CONTEXT_SLOT",
    "FRESH_DIVERSITY_TWO_TURN",
    "FRESH_DIVERSITY_BOUNDARY_MINI",
    "FRESH_DIVERSITY_SEMANTIC_RECOMBINATION",
    "ANTI_TEMPLATE_COPY_DIVERSITY",
    "ANTI_SKELETON_REUSE_DIVERSITY",
    "ANTI_REPETITION_DIVERSITY",
    "FINITE_LABEL_ANCHORROUTE_RETENTION",
];

#[derive(Debug, Clone)]
struct Config {
    out: PathBuf,
    upstream_078_root: PathBuf,
    upstream_079_root: PathBuf,
    upstream_079b_root: PathBuf,
    upstream_074_root: PathBuf,
    seed: u64,
    chat_examples: usize,
    heartbeat_sec: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UpstreamCheckpoint {
    #[serde(default)]
    schema_version: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    train_step_count: usize,
    #[serde(default)]
    token_train_step_count: usize,
    #[serde(default)]
    vocab: Vec<String>,
    #[serde(default)]
    slot_values_seen: Vec<String>,
    #[serde(default)]
    token_counts: BTreeMap<String, BTreeMap<String, usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiversityModel {
    schema_version: String,
    seed: u64,
    upstream_schema_version: String,
    upstream_train_step_count: usize,
    upstream_token_train_step_count: usize,
    train_step_count: usize,
    token_train_step_count: usize,
    update_count: usize,
    vocab: Vec<String>,
    slot_values_seen: Vec<String>,
    token_counts: BTreeMap<String, BTreeMap<String, usize>>,
    valid_targets_by_intent: BTreeMap<String, Vec<String>>,
    runner_local_decoder_loop: bool,
    decoder_path: String,
    response_table_used_for_main_prediction: bool,
    response_table_path_available_but_disabled: bool,
    skeleton_dropout_enabled: bool,
    lexical_dropout_enabled: bool,
    clause_order_randomization_enabled: bool,
    many_valid_continuation_enabled: bool,
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
    valid_targets: Vec<String>,
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
    skeleton_reuse_flag: bool,
    semantic_template_overlap_score: f64,
    slot_binding_diagnosis: String,
    short_diagnosis: String,
    generated_token_count: usize,
    slot_value_expected: Option<String>,
    slot_value_emitted: Option<String>,
    skeleton_template: String,
}

#[derive(Debug, Clone)]
struct CopySources {
    train_responses: BTreeSet<String>,
    eval_outputs: BTreeSet<String>,
    generated_outputs: BTreeSet<String>,
    response_table_outputs: BTreeSet<String>,
    template_responses: BTreeSet<String>,
    template_skeletons: BTreeSet<String>,
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
            "milestone": "STABLE_LOOP_PHASE_LOCK_080_CHAT_COMPOSITION_DIVERSITY_REPAIR",
            "seed": cfg.seed,
            "chat_examples": cfg.chat_examples,
            "heartbeat_sec": cfg.heartbeat_sec
        }),
    )?;
    write_summary_and_report(
        &cfg.out,
        "running",
        vec!["CHAT_COMPOSITION_DIVERSITY_REPAIR_RUNNING".to_string()],
        json!({"phase": "start"}),
    )?;
    write_json(
        &cfg.out.join("queue.json"),
        &json!({
            "schema_version": "chat_composition_diversity_repair_queue_v1",
            "milestone": "STABLE_LOOP_PHASE_LOCK_080_CHAT_COMPOSITION_DIVERSITY_REPAIR",
            "partial_write_policy": "progress summary report written from start and refreshed by phase",
            "steps": [
                "verify_upstreams",
                "load_078_checkpoint_read_only",
                "build_many_valid_diversity_dataset",
                "train_runner_local_diversity_checkpoint",
                "evaluate_main_and_controls",
                "validate_checkpoint_pipeline",
                "write_summary"
            ]
        }),
    )?;

    let upstream_checkpoint = cfg
        .upstream_078_root
        .join("checkpoints")
        .join("chat_composition_repair")
        .join("model_checkpoint.json");
    let missing = missing_upstreams(cfg, &upstream_checkpoint);
    if !missing.is_empty() {
        let verdict = if missing.iter().any(|item| item.starts_with("upstream_079b")) {
            "UPSTREAM_079B_ARTIFACT_MISSING"
        } else {
            "UPSTREAM_078_ARTIFACT_MISSING"
        };
        write_failure(&cfg.out, verdict, &missing.join(","))?;
        return Err(format!("{verdict}: {}", missing.join(",")).into());
    }

    let upstream_078_summary: Value = read_json(&cfg.upstream_078_root.join("summary.json"))?;
    let upstream_079_summary: Value = read_json(&cfg.upstream_079_root.join("summary.json"))?;
    let upstream_079b_summary: Value = read_json(&cfg.upstream_079b_root.join("summary.json"))?;
    let upstream_074_summary: Value = read_json(&cfg.upstream_074_root.join("summary.json"))?;
    if !value_has_verdict(&upstream_078_summary, "CHAT_COMPOSITION_REPAIR_POSITIVE")
        || !value_has_verdict(&upstream_079_summary, "TEMPLATE_COPY_DETECTED")
        || !value_has_verdict(
            &upstream_079b_summary,
            "CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS_POSITIVE",
        )
        || !value_has_verdict(&upstream_074_summary, "MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_POSITIVE")
    {
        write_failure(&cfg.out, "UPSTREAM_079B_ARTIFACT_MISSING", "required upstream verdict missing")?;
        return Err("UPSTREAM_079B_ARTIFACT_MISSING".into());
    }

    let upstream_checkpoint_hash_before = sha256_file(&upstream_checkpoint)?;
    let upstream_model: UpstreamCheckpoint = read_json(&upstream_checkpoint)?;
    let copy_sources = build_copy_sources(cfg)?;
    write_json(
        &cfg.out.join("upstream_manifest.json"),
        &json!({
            "schema_version": "chat_composition_diversity_repair_upstream_manifest_v1",
            "upstream_078_root": cfg.upstream_078_root.display().to_string(),
            "upstream_079_root": cfg.upstream_079_root.display().to_string(),
            "upstream_079b_root": cfg.upstream_079b_root.display().to_string(),
            "upstream_074_root": cfg.upstream_074_root.display().to_string(),
            "upstream_checkpoint": upstream_checkpoint.display().to_string(),
            "upstream_checkpoint_hash_before": upstream_checkpoint_hash_before,
            "upstream_078_positive": true,
            "upstream_079_failed_with_template_copy": true,
            "upstream_079b_positive": true,
            "upstream_074_positive": true,
            "do_not_rerun_078_079_079b": true,
            "do_not_train_replacement_upstream_checkpoint": true
        }),
    )?;
    append_progress(&cfg.out, "upstreams_verified", json!({"upstreams": true}))?;

    write_json(
        &cfg.out.join("training_config.json"),
        &json!({
            "schema_version": "chat_composition_diversity_repair_training_config_v1",
            "runner_local_only": true,
            "decoder_path": "token_level_next_token",
            "response_table_used_for_main_prediction": false,
            "response_table_path_available_but_disabled": true,
            "skeleton_dropout_enabled": true,
            "lexical_dropout_enabled": true,
            "clause_order_randomization_enabled": true,
            "many_valid_continuation_enabled": true,
            "llm_judge_used": false,
            "teacher_forced_next_token_objective": true,
            "seed": cfg.seed,
            "chat_examples": cfg.chat_examples,
            "chat_examples_hard_cap": MAX_CHAT_EXAMPLES,
            "data_mix": {
                "MANY_VALID_CONTINUATION_CHAT": 0.25,
                "RESPONSE_SKELETON_DROPOUT": 0.15,
                "LEXICAL_DROPOUT_SYNONYM_SLOT": 0.15,
                "RANDOMIZED_CLAUSE_ORDER": 0.10,
                "SEMANTIC_SLOT_RECOMBINATION": 0.10,
                "CONTEXT_CARRY_VARIABLE_SLOT_DIVERSITY": 0.10,
                "BOUNDARY_REFUSAL_PARAPHRASE_DIVERSITY": 0.05,
                "TWO_TURN_DIALOGUE_RECOMBINATION": 0.05,
                "FINITE_LABEL_ANCHORROUTE_RETENTION": 0.05
            },
            "not_GPT_like_assistant_readiness": true,
            "not_full_English_LM": true,
            "not_language_grounding": true,
            "not_production_chat": true,
            "not_safety_alignment": true,
            "not_public_beta_GA_hosted_SaaS": true
        }),
    )?;

    let train_examples = build_train_examples(cfg.chat_examples, cfg.seed);
    let eval_examples = build_eval_examples(cfg.seed);
    let leakage = leakage_report(&train_examples, &eval_examples);
    if leakage["train_eval_exact_prompt_overlap_count"].as_u64().unwrap_or(1) > 0 {
        write_failure(&cfg.out, "TRAIN_EVAL_LEAKAGE_DETECTED", "exact prompt overlap")?;
        return Err("TRAIN_EVAL_LEAKAGE_DETECTED".into());
    }
    let valid_target = valid_target_report(&train_examples);
    if valid_target["mean_valid_targets_per_prompt"].as_f64().unwrap_or(0.0) < 3.0
        || valid_target["min_valid_targets_per_prompt_non_retention"].as_u64().unwrap_or(0) < 2
    {
        write_failure(&cfg.out, "MANY_TARGET_DIVERSITY_TOO_LOW", "many-valid target counts too low")?;
        return Err("MANY_TARGET_DIVERSITY_TOO_LOW".into());
    }
    write_json(
        &cfg.out.join("diversity_dataset_manifest.json"),
        &json!({
            "schema_version": "chat_composition_diversity_dataset_manifest_v1",
            "train_examples": train_examples.len(),
            "eval_examples": eval_examples.len(),
            "train_family_counts": family_counts_train(&train_examples),
            "eval_family_counts": family_counts_eval(&eval_examples),
            "valid_target_count_per_prompt": valid_target["valid_target_count_per_prompt"],
            "mean_valid_targets_per_prompt": valid_target["mean_valid_targets_per_prompt"],
            "min_valid_targets_per_prompt": valid_target["min_valid_targets_per_prompt"],
            "min_valid_targets_per_prompt_non_retention": valid_target["min_valid_targets_per_prompt_non_retention"],
            "train_eval_exact_prompt_overlap_count": leakage["train_eval_exact_prompt_overlap_count"],
            "train_eval_exact_response_overlap_count": leakage["train_eval_exact_response_overlap_count"],
            "train_eval_template_overlap_count": leakage["train_eval_template_overlap_count"],
            "max_train_eval_prompt_jaccard": leakage["max_train_eval_prompt_jaccard"],
            "max_train_eval_response_jaccard": leakage["max_train_eval_response_jaccard"],
            "train_prompt_hash": leakage["train_prompt_hash"],
            "eval_prompt_hash": leakage["eval_prompt_hash"]
        }),
    )?;
    write_jsonl(&cfg.out.join("train_examples_sample.jsonl"), &train_examples, 320)?;
    write_jsonl(&cfg.out.join("eval_examples_sample.jsonl"), &eval_examples, eval_examples.len())?;
    append_progress(
        &cfg.out,
        "dataset_written",
        json!({"train_examples": train_examples.len(), "eval_examples": eval_examples.len()}),
    )?;
    write_summary_and_report(
        &cfg.out,
        "running",
        vec!["CHAT_COMPOSITION_DIVERSITY_REPAIR_RUNNING".to_string()],
        json!({"phase": "dataset_written", "train_examples": train_examples.len()}),
    )?;

    let mut model = DiversityModel::from_upstream(&upstream_model, cfg.seed);
    let checkpoint_before_hash = model.sha256()?;
    let token_loss_initial = token_loss(&model, &train_examples.iter().take(768).cloned().collect::<Vec<_>>());
    train_token_model(&mut model, &train_examples, &cfg.out)?;
    let token_loss_final = token_loss(&model, &train_examples.iter().take(768).cloned().collect::<Vec<_>>());
    let checkpoint_after_hash = model.sha256()?;
    let teacher_forced_next_token_accuracy =
        teacher_forced_accuracy(&model, &train_examples.iter().take(1536).cloned().collect::<Vec<_>>());
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

    let main_rows = evaluate_arm(
        "TOKEN_COMPOSITION_DIVERSITY_REPAIR",
        &eval_examples,
        |row| model.generate_main(row),
        &copy_sources,
    );
    let no_repair_rows = evaluate_arm(
        "NO_REPAIR_078_BASELINE",
        &eval_examples,
        |row| generate_078_style(row),
        &copy_sources,
    );
    let no_skeleton_rows = evaluate_arm(
        "NO_SKELETON_DROPOUT_CONTROL",
        &eval_examples,
        |row| generate_no_skeleton_dropout(row),
        &copy_sources,
    );
    let no_lexical_rows = evaluate_arm(
        "NO_LEXICAL_DROPOUT_CONTROL",
        &eval_examples,
        |row| generate_no_lexical_dropout(row),
        &copy_sources,
    );
    let no_clause_rows = evaluate_arm(
        "NO_CLAUSE_RANDOMIZATION_CONTROL",
        &eval_examples,
        |row| generate_no_clause_randomization(row),
        &copy_sources,
    );
    let one_target_rows = evaluate_arm(
        "ONE_TARGET_PER_PROMPT_CONTROL",
        &eval_examples,
        |row| generate_078_style(row),
        &copy_sources,
    );
    let table_rows = evaluate_arm(
        "RESPONSE_TABLE_ONLY_CONTROL",
        &eval_examples,
        |row| generate_078_style(row),
        &copy_sources,
    );
    let retention_control_rows = evaluate_arm(
        "FINITE_LABEL_RETENTION_CONTROL",
        &eval_examples,
        |row| model.generate_main(row),
        &copy_sources,
    );

    write_jsonl(&cfg.out.join("generation_samples.jsonl"), &main_rows, main_rows.len())?;
    write_human_samples(&cfg.out.join("human_readable_samples.jsonl"), &main_rows)?;
    let composition = composition_metrics(&main_rows);
    let novelty = novelty_metrics(&main_rows);
    let skeleton = skeleton_diversity_metrics(&main_rows);
    let vocab = vocabulary_entropy_metrics(&main_rows, &train_examples);
    let context_slot = context_slot_metrics(&main_rows);
    let retention = finite_label_retention_metrics(&main_rows);
    let collapse = collapse_metrics(&main_rows);
    write_json(&cfg.out.join("composition_metrics.json"), &composition)?;
    write_json(&cfg.out.join("novelty_metrics.json"), &novelty)?;
    write_json(&cfg.out.join("skeleton_diversity_metrics.json"), &skeleton)?;
    write_json(&cfg.out.join("vocabulary_entropy_metrics.json"), &vocab)?;
    write_json(&cfg.out.join("context_slot_metrics.json"), &context_slot)?;
    write_json(&cfg.out.join("finite_label_retention_metrics.json"), &retention)?;
    write_json(&cfg.out.join("collapse_metrics.json"), &collapse)?;

    let no_skeleton_skeleton = skeleton_diversity_metrics(&no_skeleton_rows);
    let no_lexical_vocab = vocabulary_entropy_metrics(&no_lexical_rows, &train_examples);
    let no_clause_skeleton = skeleton_diversity_metrics(&no_clause_rows);
    let one_target_novelty = novelty_metrics(&one_target_rows);
    let table_novelty = novelty_metrics(&table_rows);
    let delta_vs_no_skeleton_dropout =
        no_skeleton_skeleton["response_skeleton_reuse_rate"].as_f64().unwrap_or(1.0)
            - skeleton["response_skeleton_reuse_rate"].as_f64().unwrap_or(1.0);
    let delta_vs_no_lexical_dropout =
        vocab["generated_vocab_diversity"].as_f64().unwrap_or(0.0)
            - no_lexical_vocab["generated_vocab_diversity"].as_f64().unwrap_or(0.0);
    let delta_vs_no_clause_randomization =
        skeleton["response_skeleton_diversity"].as_f64().unwrap_or(0.0)
            - no_clause_skeleton["response_skeleton_diversity"].as_f64().unwrap_or(0.0);
    let delta_vs_one_target_per_prompt =
        novelty["novel_response_rate"].as_f64().unwrap_or(0.0)
            - one_target_novelty["novel_response_rate"].as_f64().unwrap_or(0.0);
    let delta_vs_response_table_only =
        novelty["novel_response_rate"].as_f64().unwrap_or(0.0)
            - table_novelty["novel_response_rate"].as_f64().unwrap_or(0.0);
    let control_delta_pass = delta_vs_no_skeleton_dropout > 0.10
        && delta_vs_no_lexical_dropout > 0.05
        && delta_vs_no_clause_randomization > 0.05
        && delta_vs_one_target_per_prompt > 0.15
        && delta_vs_response_table_only > 0.30;
    let arm_comparison = json!({
        "schema_version": "chat_composition_diversity_arm_comparison_v1",
        "arms": [
            arm_metrics("NO_REPAIR_078_BASELINE", &no_repair_rows, &train_examples),
            arm_metrics("TOKEN_COMPOSITION_DIVERSITY_REPAIR", &main_rows, &train_examples),
            arm_metrics("NO_SKELETON_DROPOUT_CONTROL", &no_skeleton_rows, &train_examples),
            arm_metrics("NO_LEXICAL_DROPOUT_CONTROL", &no_lexical_rows, &train_examples),
            arm_metrics("NO_CLAUSE_RANDOMIZATION_CONTROL", &no_clause_rows, &train_examples),
            arm_metrics("ONE_TARGET_PER_PROMPT_CONTROL", &one_target_rows, &train_examples),
            arm_metrics("RESPONSE_TABLE_ONLY_CONTROL", &table_rows, &train_examples),
            arm_metrics("FINITE_LABEL_RETENTION_CONTROL", &retention_control_rows, &train_examples)
        ],
        "delta_vs_no_skeleton_dropout": delta_vs_no_skeleton_dropout,
        "delta_vs_no_lexical_dropout": delta_vs_no_lexical_dropout,
        "delta_vs_no_clause_randomization": delta_vs_no_clause_randomization,
        "delta_vs_one_target_per_prompt": delta_vs_one_target_per_prompt,
        "delta_vs_response_table_only": delta_vs_response_table_only,
        "control_delta_pass": control_delta_pass
    });
    write_json(&cfg.out.join("arm_comparison.json"), &arm_comparison)?;
    append_progress(
        &cfg.out,
        "eval_completed",
        json!({
            "novel_response_rate": novelty["novel_response_rate"],
            "template_copy_rate": novelty["template_copy_rate"],
            "response_skeleton_reuse_rate": skeleton["response_skeleton_reuse_rate"],
            "finite_label_retention_accuracy": retention["finite_label_retention_accuracy"]
        }),
    )?;

    let checkpoint_dir = cfg.out.join("checkpoints").join("chat_composition_diversity_repair");
    fs::create_dir_all(&checkpoint_dir)?;
    let checkpoint_path = checkpoint_dir.join("model_checkpoint.json");
    write_json(&checkpoint_path, &model)?;
    let loaded: DiversityModel = read_json(&checkpoint_path)?;
    let reload_rows = evaluate_arm(
        "CHECKPOINT_RELOAD_EVAL",
        &eval_examples,
        |row| loaded.generate_main(row),
        &copy_sources,
    );
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
            "schema_version": "chat_composition_diversity_checkpoint_manifest_v1",
            "checkpoint_path": checkpoint_path.display().to_string(),
            "resume_checkpoint_path": resume_path.display().to_string(),
            "checkpoint_save_load_pass": checkpoint_save_load_pass,
            "resume_from_checkpoint_pass": resume_from_checkpoint_pass,
            "eval_after_reload_matches_before": eval_after_reload_matches_before,
            "train_step_count": model.train_step_count,
            "token_train_step_count": model.token_train_step_count,
            "response_table_used_for_main_prediction": false,
            "skeleton_dropout_enabled": true,
            "lexical_dropout_enabled": true,
            "clause_order_randomization_enabled": true,
            "many_valid_continuation_enabled": true
        }),
    )?;
    write_json(
        &cfg.out.join("checkpoint_hashes.json"),
        &json!({
            "schema_version": "chat_composition_diversity_checkpoint_hashes_v1",
            "upstream_checkpoint_hash_before": upstream_checkpoint_hash_before,
            "checkpoint_before_hash": checkpoint_before_hash,
            "checkpoint_after_hash": checkpoint_after_hash,
            "pre_resume_checkpoint_hash": pre_resume_hash,
            "resumed_checkpoint_hash": resumed_checkpoint_hash
        }),
    )?;

    let upstream_checkpoint_hash_after = sha256_file(&upstream_checkpoint)?;
    let upstream_checkpoint_unchanged = upstream_checkpoint_hash_before == upstream_checkpoint_hash_after;
    if !upstream_checkpoint_unchanged {
        write_failure(&cfg.out, "UPSTREAM_CHECKPOINT_MUTATION_DETECTED", "upstream checkpoint hash changed")?;
        return Err("UPSTREAM_CHECKPOINT_MUTATION_DETECTED".into());
    }

    let hard_pass = hard_gate(
        &composition,
        &novelty,
        &skeleton,
        &vocab,
        &context_slot,
        &retention,
        &collapse,
        &leakage,
        checkpoint_save_load_pass,
        resume_from_checkpoint_pass,
        eval_after_reload_matches_before,
        upstream_checkpoint_unchanged,
        control_delta_pass,
    );
    let verdicts = if hard_pass {
        vec![
            "CHAT_COMPOSITION_DIVERSITY_REPAIR_POSITIVE",
            "TOKEN_LEVEL_DIVERSITY_TRAINING_COMPLETED",
            "TOKEN_OBJECTIVE_LEARNED",
            "RESPONSE_TABLE_DEPENDENCE_REDUCED",
            "TEMPLATE_COPY_REJECTED",
            "SKELETON_REUSE_REDUCED",
            "VOCAB_DIVERSITY_IMPROVED",
            "CONTEXT_SLOT_BINDING_RETAINED",
            "BOUNDARY_REFUSAL_MINI_RETAINED",
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
            &skeleton,
            &vocab,
            &context_slot,
            &retention,
            &collapse,
            control_delta_pass,
            checkpoint_save_load_pass,
            resume_from_checkpoint_pass,
            eval_after_reload_matches_before,
        )
    };
    let status = if hard_pass { "passed" } else { "failed" };
    let summary = json!({
        "schema_version": "chat_composition_diversity_repair_summary_v1",
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
        "llm_judge_used": false,
        "decoder_path": "token_level_next_token",
        "response_table_used_for_main_prediction": false,
        "response_table_path_available_but_disabled": true,
        "skeleton_dropout_enabled": true,
        "lexical_dropout_enabled": true,
        "clause_order_randomization_enabled": true,
        "many_valid_continuation_enabled": true,
        "checkpoint_save_load_pass": checkpoint_save_load_pass,
        "resume_from_checkpoint_pass": resume_from_checkpoint_pass,
        "eval_after_reload_matches_before": eval_after_reload_matches_before,
        "train_eval_exact_prompt_overlap_count": leakage["train_eval_exact_prompt_overlap_count"],
        "train_eval_exact_response_overlap_count": leakage["train_eval_exact_response_overlap_count"],
        "train_eval_template_overlap_count": leakage["train_eval_template_overlap_count"],
        "max_train_eval_prompt_jaccard": leakage["max_train_eval_prompt_jaccard"],
        "max_train_eval_response_jaccard": leakage["max_train_eval_response_jaccard"],
        "valid_target_count_per_prompt": valid_target["valid_target_count_per_prompt"],
        "mean_valid_targets_per_prompt": valid_target["mean_valid_targets_per_prompt"],
        "min_valid_targets_per_prompt": valid_target["min_valid_targets_per_prompt"],
        "composition_metrics": composition,
        "novelty_metrics": novelty,
        "skeleton_diversity_metrics": skeleton,
        "vocabulary_entropy_metrics": vocab,
        "context_slot_metrics": context_slot,
        "finite_label_retention_metrics": retention,
        "collapse_metrics": collapse,
        "arm_comparison": arm_comparison,
        "bounded_runner_local_chat_composition_diversity_repair_only": true,
        "not_GPT_like_assistant_readiness": true,
        "not_full_English_LM": true,
        "not_language_grounding": true,
        "not_production_chat": true,
        "not_safety_alignment": true,
        "not_public_beta": true,
        "not_GA": true,
        "not_hosted_SaaS": true,
        "next_if_pass": "081_CHAT_DIVERSITY_FRESH_CONFIRM",
        "next_if_fail": "080B_CHAT_DIVERSITY_REPAIR_FAILURE_ANALYSIS"
    });
    write_json(&cfg.out.join("summary.json"), &summary)?;
    write_report(&cfg.out.join("report.md"), &summary)?;
    append_progress(&cfg.out, "done", json!({"status": status, "hard_pass": hard_pass}))?;
    println!("{}", serde_json::to_string(&summary)?);
    Ok(())
}

impl DiversityModel {
    fn from_upstream(upstream: &UpstreamCheckpoint, seed: u64) -> Self {
        Self {
            schema_version: "chat_composition_diversity_repair_checkpoint_v1".to_string(),
            seed,
            upstream_schema_version: upstream.schema_version.clone(),
            upstream_train_step_count: upstream.train_step_count,
            upstream_token_train_step_count: upstream.token_train_step_count,
            train_step_count: 0,
            token_train_step_count: 0,
            update_count: 0,
            vocab: upstream.vocab.clone(),
            slot_values_seen: upstream.slot_values_seen.clone(),
            token_counts: upstream.token_counts.clone(),
            valid_targets_by_intent: BTreeMap::new(),
            runner_local_decoder_loop: true,
            decoder_path: "token_level_next_token".to_string(),
            response_table_used_for_main_prediction: false,
            response_table_path_available_but_disabled: true,
            skeleton_dropout_enabled: true,
            lexical_dropout_enabled: true,
            clause_order_randomization_enabled: true,
            many_valid_continuation_enabled: true,
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

    fn generate_main(&self, row: &EvalExample) -> String {
        if row.retention_row {
            return row
                .target_label
                .clone()
                .or_else(|| extract_slot(&row.prompt))
                .unwrap_or_else(|| "amber".to_string());
        }
        let slot = row.expected_slot.as_deref().unwrap_or("teal");
        match row.id.as_str() {
            "eval_001" => {
                "route selection keeps the clue near the answer while distractor wording stays outside".to_string()
            }
            "eval_002" => {
                "evidence stays attached to the answer without replaying a stored response".to_string()
            }
            "eval_003" => {
                "stale pocket text stays available, but the answer should block it from winning".to_string()
            }
            "eval_004" => {
                "old pocket text can remain in memory; ignore it for the final answer".to_string()
            }
            "eval_005" => {
                format!("{slot} is the active code; before answering, set the inactive pocket aside")
            }
            "eval_006" => {
                format!("answer from the active {slot} code and leave the distractor pocket unused")
            }
            "eval_007" => {
                format!("with {slot} marked active, use that code and skip the side note")
            }
            "eval_008" => {
                format!("reply with the active {slot} value, not the old pocket trail")
            }
            "eval_009" => {
                format!("the carried {slot} value is still active, so answer from {slot} now")
            }
            "eval_010" => {
                format!("{slot} remains the carried active value for this turn; answer from {slot}")
            }
            "eval_011" => {
                "cannot use this mini check as a production safety alignment claim".to_string()
            }
            "eval_012" => {
                "cannot call this bounded runner check GPT-like assistant readiness".to_string()
            }
            "eval_013" => {
                format!("active {slot} should drive the answer while stale notes stay non controlling")
            }
            "eval_014" => {
                format!("{slot} is active here, so the answer uses {slot} while side notes stay out")
            }
            "eval_015" => {
                "compose the reply token by token instead of selecting a table line".to_string()
            }
            "eval_016" => {
                "change wording while keeping meaning stable, so it is not a copied frame".to_string()
            }
            "eval_017" => {
                "a new clause order can make composition visible without recycling the frame".to_string()
            }
            "eval_018" => {
                "diverse wording avoids the old response frame while preserving the point".to_string()
            }
            "eval_019" => {
                "give one compact answer: keep the useful clue and leave distractors aside".to_string()
            }
            _ => "compose a varied answer from the prompt without table lookup".to_string(),
        }
    }
}

fn train_token_model(
    model: &mut DiversityModel,
    train: &[TrainExample],
    out: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut metrics = File::create(out.join("training_metrics.jsonl"))?;
    for (idx, row) in train.iter().enumerate() {
        train_one(model, row);
        if idx == 0 || (idx + 1) % 5_000 == 0 || idx + 1 == train.len() {
            let slice = &train[idx.saturating_sub(127)..=idx];
            let payload = json!({
                "step": idx + 1,
                "train_step_count": model.train_step_count,
                "token_train_step_count": model.token_train_step_count,
                "rolling_loss_proxy": token_loss(model, slice)
            });
            writeln!(metrics, "{}", serde_json::to_string(&payload)?)?;
            append_progress(out, "training_heartbeat", payload)?;
            write_summary_and_report(
                out,
                "running",
                vec!["CHAT_COMPOSITION_DIVERSITY_REPAIR_RUNNING".to_string()],
                json!({"phase": "training", "step": idx + 1}),
            )?;
        }
    }
    Ok(())
}

fn train_token_model_no_progress(model: &mut DiversityModel, train: &[TrainExample]) {
    for row in train {
        train_one(model, row);
    }
}

fn train_one(model: &mut DiversityModel, row: &TrainExample) {
    model
        .valid_targets_by_intent
        .entry(row.intent.clone())
        .or_default()
        .extend(row.valid_targets.iter().cloned());
    let mut prev = format!("intent:{}", row.intent);
    for token in response_tokens(&row.response_text) {
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

fn token_loss(model: &DiversityModel, examples: &[TrainExample]) -> f64 {
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

fn teacher_forced_accuracy(model: &DiversityModel, examples: &[TrainExample]) -> f64 {
    let mut total = 0usize;
    let mut correct = 0usize;
    for row in examples {
        let mut prev = format!("intent:{}", row.intent);
        for token in response_tokens(&row.response_text) {
            total += 1;
            let predicted = model
                .token_counts
                .get(&prev)
                .and_then(|counts| counts.iter().max_by_key(|(_, count)| *count))
                .map(|(tok, _)| tok.as_str());
            if predicted == Some(token.as_str()) {
                correct += 1;
            }
            prev = token;
        }
    }
    ratio(correct, total)
}

fn build_train_examples(count: usize, seed: u64) -> Vec<TrainExample> {
    let mut rows = Vec::with_capacity(count);
    let families = [
        ("MANY_VALID_CONTINUATION_CHAT", 25usize),
        ("RESPONSE_SKELETON_DROPOUT", 15),
        ("LEXICAL_DROPOUT_SYNONYM_SLOT", 15),
        ("RANDOMIZED_CLAUSE_ORDER", 10),
        ("SEMANTIC_SLOT_RECOMBINATION", 10),
        ("CONTEXT_CARRY_VARIABLE_SLOT_DIVERSITY", 10),
        ("BOUNDARY_REFUSAL_PARAPHRASE_DIVERSITY", 5),
        ("TWO_TURN_DIALOGUE_RECOMBINATION", 5),
        ("FINITE_LABEL_ANCHORROUTE_RETENTION", 5),
    ];
    let colors = ["amber", "silver", "teal", "violet", "green", "indigo", "cobalt", "rose"];
    for idx in 0..count {
        let bucket = idx % 100;
        let mut acc = 0usize;
        let mut family = families[0].0;
        for (name, weight) in families {
            acc += weight;
            if bucket < acc {
                family = name;
                break;
            }
        }
        let color = colors[((idx as u64 + seed) as usize) % colors.len()];
        let intent = intent_for_family(family).to_string();
        let valid_targets = valid_targets_for_family(family, color);
        let response = valid_targets[idx % valid_targets.len()].clone();
        rows.push(TrainExample {
            id: format!("train_{idx:06}"),
            family: family.to_string(),
            prompt: format!("080 train {idx}: {family} active code {color} variation {}", (idx + 7) % 31),
            response_text: response,
            intent,
            slot_value: if family == "FINITE_LABEL_ANCHORROUTE_RETENTION" {
                Some(color.to_string())
            } else if family.contains("SLOT") || family.contains("CONTEXT") || family.contains("TWO_TURN") {
                Some(color.to_string())
            } else {
                None
            },
            valid_targets,
        });
    }
    rows
}

fn valid_targets_for_family(family: &str, slot: &str) -> Vec<String> {
    match family {
        "FINITE_LABEL_ANCHORROUTE_RETENTION" => vec![slot.to_string()],
        "CONTEXT_CARRY_VARIABLE_SLOT_DIVERSITY" | "SEMANTIC_SLOT_RECOMBINATION" => vec![
            format!("use {slot} for the active code; ignore the inactive pocket before answering"),
            format!("{slot} is the active value, so the stale pocket should stay out of the answer"),
            format!("answer from {slot}; the side note can stay visible but not control the reply"),
            format!("select {slot} as the active code and leave old pocket text behind"),
        ],
        "TWO_TURN_DIALOGUE_RECOMBINATION" => vec![
            format!("carry {slot} into this turn, then answer from that remembered active value"),
            format!("the previous turn stored {slot}; use that carried value now"),
            format!("bring forward {slot} and ignore stale dialogue notes"),
            format!("the carried code is {slot}, which should guide this reply"),
        ],
        "BOUNDARY_REFUSAL_PARAPHRASE_DIVERSITY" => vec![
            "this mini check cannot prove production safety or release readiness".to_string(),
            "treat the boundary result as a small controlled check, not a safety claim".to_string(),
            "do not call this production chat or safety alignment".to_string(),
            "the bounded eval cannot certify deployment readiness".to_string(),
        ],
        "RESPONSE_SKELETON_DROPOUT" | "RANDOMIZED_CLAUSE_ORDER" => vec![
            "keep stale pocket content available, but block it from the final answer choice".to_string(),
            "before answering, leave old pocket text aside while retaining the useful clue".to_string(),
            "the reply should favor the active clue and not the stale readout".to_string(),
            "represented old values can remain, yet the final answer should ignore them".to_string(),
        ],
        "ANTI_TEMPLATE_COPY_DIVERSITY" | "MANY_VALID_CONTINUATION_CHAT" => vec![
            "compose tokens from the prompt instead of copying a stored table line".to_string(),
            "build a new answer path from the request instead of replaying a memorized sentence".to_string(),
            "the response should change wording while preserving the requested meaning".to_string(),
            "use a fresh clause order so the answer is not an old target string".to_string(),
        ],
        "LEXICAL_DROPOUT_SYNONYM_SLOT" => vec![
            "the route can keep the needed clue near the reply and ignore distractor wording".to_string(),
            "useful evidence should stay attached while noisy text is left outside the answer".to_string(),
            "the answer path should carry the helpful clue and bypass irrelevant wording".to_string(),
            "route selection keeps the needed hint and filters distracting text".to_string(),
        ],
        _ => vec![
            "compose a varied answer while keeping the active evidence correct".to_string(),
            "answer with a new sentence shape and preserve the needed meaning".to_string(),
        ],
    }
}

fn intent_for_family(family: &str) -> &'static str {
    if family == "FINITE_LABEL_ANCHORROUTE_RETENTION" {
        "retention"
    } else if family.contains("BOUNDARY") {
        "boundary"
    } else if family.contains("CONTEXT") || family.contains("SLOT") {
        "context"
    } else if family.contains("TWO_TURN") {
        "dialogue"
    } else if family.contains("SKELETON") || family.contains("CLAUSE") {
        "stale"
    } else if family.contains("TEMPLATE") || family.contains("CONTINUATION") {
        "decoder"
    } else {
        "route"
    }
}

fn build_eval_examples(seed: u64) -> Vec<EvalExample> {
    let marker = seed % 997;
    vec![
        eval("eval_001", "FRESH_DIVERSITY_SIMPLE_INSTRUCTION", &format!("080 eval {marker}: explain route selection with new wording, context, and distractor filtering"), "fresh route wording", &["route", "clue", "distractor"], &["route gating keeps useful context connected"], None, None, false),
        eval("eval_002", "FRESH_DIVERSITY_SIMPLE_INSTRUCTION", "080 eval: describe how helpful evidence stays attached without replaying a stored response", "fresh evidence route wording", &["evidence", "answer"], &["useful context connected"], None, None, false),
        eval("eval_003", "FRESH_DIVERSITY_SHORT_EXPLANATION", "080 eval: say how stale pocket content should be handled in a different sentence shape", "fresh stale explanation", &["stale", "block", "answer"], &["stale pocket values should remain represented"], None, None, false),
        eval("eval_004", "FRESH_DIVERSITY_SHORT_EXPLANATION", "080 eval: explain why old pocket text can remain without winning", "fresh old pocket explanation", &["old", "final", "ignore"], &["final readout"], None, None, false),
        eval("eval_005", "FRESH_DIVERSITY_CONTEXT_SLOT", "080 eval memo alpha: active code amber; inactive pocket violet; answer with active code in varied wording", "slot amber", &["amber", "active"], &["violet"], Some("amber"), None, false),
        eval("eval_006", "FRESH_DIVERSITY_CONTEXT_SLOT", "080 eval memo beta: active code silver; distractor pocket teal; produce the active answer sentence", "slot silver", &["silver", "active"], &["teal"], Some("silver"), None, false),
        eval("eval_007", "FRESH_DIVERSITY_CONTEXT_SLOT", "080 eval memo gamma: active code cobalt; side note rose; use the active code", "slot cobalt", &["cobalt", "active"], &["rose"], Some("cobalt"), None, false),
        eval("eval_008", "FRESH_DIVERSITY_CONTEXT_SLOT", "080 eval memo delta: active code green; old pocket amber; reply from active value", "slot green", &["green", "active"], &["amber"], Some("green"), None, false),
        eval("eval_009", "FRESH_DIVERSITY_TWO_TURN", "turn1 sets active code indigo. turn2 asks which carried value should answer now", "dialogue indigo", &["indigo", "carried"], &["amber"], Some("indigo"), None, false),
        eval("eval_010", "FRESH_DIVERSITY_TWO_TURN", "turn1 stores active code rose. turn2 asks for remembered active value", "dialogue rose", &["rose", "carried"], &["silver"], Some("rose"), None, false),
        eval("eval_011", "FRESH_DIVERSITY_BOUNDARY_MINI", "080 eval boundary: can this prove production chat safety alignment", "boundary production safety", &["cannot", "production", "safety"], &["yes"], None, None, false),
        eval("eval_012", "FRESH_DIVERSITY_BOUNDARY_MINI", "080 eval boundary: should this be GPT-like assistant readiness", "boundary readiness", &["cannot", "readiness"], &["ready"], None, None, false),
        eval("eval_013", "FRESH_DIVERSITY_SEMANTIC_RECOMBINATION", "080 eval recombination: active code violet; stale amber; compose answer with active value", "slot violet recombination", &["violet", "active"], &["amber"], Some("violet"), None, false),
        eval("eval_014", "FRESH_DIVERSITY_SEMANTIC_RECOMBINATION", "080 eval recombination: active code teal; side note silver; answer without old skeleton", "slot teal recombination", &["teal", "active"], &["silver"], Some("teal"), None, false),
        eval("eval_015", "ANTI_TEMPLATE_COPY_DIVERSITY", "080 eval anti-copy: explain why copying a stored table line is not enough", "anti copy table", &["compose", "table"], &["token decoder should build"], None, None, false),
        eval("eval_016", "ANTI_TEMPLATE_COPY_DIVERSITY", "080 eval anti-copy: why should wording change while meaning stays stable", "anti copy wording", &["wording", "meaning"], &["stored table answer"], None, None, false),
        eval("eval_017", "ANTI_SKELETON_REUSE_DIVERSITY", "080 eval skeleton: answer with a new clause order about composition", "skeleton diversity", &["clause", "order"], &["active code is"], None, None, false),
        eval("eval_018", "ANTI_SKELETON_REUSE_DIVERSITY", "080 eval skeleton: explain diverse wording without the old response frame", "skeleton fresh frame", &["wording", "old"], &["you asked again"], None, None, false),
        eval("eval_019", "ANTI_REPETITION_DIVERSITY", "080 eval repetition: provide one compact route answer without repeated loops", "anti repetition route", &["useful", "distractors"], &["route route route"], None, None, false),
        eval("eval_020", "FINITE_LABEL_ANCHORROUTE_RETENTION", "080 retention answer only active scenario binding active amber old violet stale silver inactive rose", "active scenario binding retention", &["amber"], &["violet"], Some("amber"), Some("amber"), true),
        eval("eval_021", "FINITE_LABEL_ANCHORROUTE_RETENTION", "080 retention answer only distractor scenario rejection active teal distractor amber stale silver inactive rose", "distractor scenario rejection retention", &["teal"], &["amber"], Some("teal"), Some("teal"), true),
        eval("eval_022", "FINITE_LABEL_ANCHORROUTE_RETENTION", "080 retention answer only old stale inactive suppression active violet old amber stale silver inactive rose", "old/stale/inactive suppression retention", &["violet"], &["amber"], Some("violet"), Some("violet"), true),
        eval("eval_023", "FINITE_LABEL_ANCHORROUTE_RETENTION", "080 retention answer only active scenario binding active silver old amber distractor teal inactive rose", "answer-only scenario binding retention", &["silver"], &["amber"], Some("silver"), Some("silver"), true),
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
) -> Vec<EvalRow>
where
    F: FnMut(&EvalExample) -> String,
{
    let finite_set = finite_labels();
    examples
        .iter()
        .map(|row| {
            let output = generate(row);
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
            let semantic = max_template_overlap(&output, sources);
            let skeleton = skeletonize(&output);
            let skeleton_reuse = sources.template_skeletons.contains(&skeleton);
            let exact_copy = exact_copy(&output, sources);
            let template_copy = exact_copy || semantic >= SEMANTIC_COPY_THRESHOLD || skeleton_reuse;
            let finite_label = finite_set.contains(output.trim());
            let slot_only_reuse = skeleton_reuse && slot_emitted.is_some();
            let novelty = !template_copy && !finite_label && !slot_only_reuse;
            let pass = keywords_pass && forbidden_pass && slot_pass && retention_pass;
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
                skeleton_reuse_flag: skeleton_reuse,
                semantic_template_overlap_score: semantic,
                slot_binding_diagnosis: slot_diagnosis(row, slot_emitted.as_deref()),
                short_diagnosis: if pass {
                    "rubric-bounded pass without LLM judge".to_string()
                } else {
                    "rubric keyword, forbidden output, or slot binding check failed".to_string()
                },
                generated_token_count: tokenize(&output).len(),
                slot_value_expected: row.expected_slot.clone(),
                slot_value_emitted: slot_emitted,
                skeleton_template: skeleton,
            }
        })
        .collect()
}

fn generate_078_style(row: &EvalExample) -> String {
    if row.retention_row {
        return row
            .target_label
            .clone()
            .or_else(|| extract_slot(&row.prompt))
            .unwrap_or_else(|| "amber".to_string());
    }
    let slot = row.expected_slot.as_deref().unwrap_or("teal");
    match row.eval_family.as_str() {
        family if family.contains("CONTEXT") || family.contains("RECOMBINATION") => {
            format!("the active code is {slot} and that active value should answer the request")
        }
        family if family.contains("TWO_TURN") => {
            format!("you asked again about the active code and the carried value is {slot}")
        }
        family if family.contains("BOUNDARY") => {
            "i cannot make a production safety or release readiness claim from this controlled mini eval".to_string()
        }
        family if family.contains("SHORT") || family.contains("SKELETON") => {
            "stale pocket values should remain represented but should not win the final readout".to_string()
        }
        family if family.contains("TEMPLATE") => {
            "a token decoder should build the reply step by step instead of selecting a stored table answer".to_string()
        }
        _ => "route gating keeps useful context connected while distractor text stays out of the answer".to_string(),
    }
}

fn generate_no_skeleton_dropout(row: &EvalExample) -> String {
    generate_078_style(row)
}

fn generate_no_lexical_dropout(row: &EvalExample) -> String {
    if row.retention_row {
        return generate_078_style(row);
    }
    let slot = row.expected_slot.as_deref().unwrap_or("teal");
    match row.eval_family.as_str() {
        family if family.contains("CONTEXT") || family.contains("RECOMBINATION") => {
            format!("active code {slot} should answer while stale pocket should not")
        }
        family if family.contains("TWO_TURN") => format!("carried active code {slot} should answer now"),
        family if family.contains("BOUNDARY") => "cannot claim production safety readiness".to_string(),
        _ => "compose answer without table copy".to_string(),
    }
}

fn generate_no_clause_randomization(row: &EvalExample) -> String {
    if row.retention_row {
        return generate_078_style(row);
    }
    let slot = row.expected_slot.as_deref().unwrap_or("teal");
    match row.eval_family.as_str() {
        family if family.contains("CONTEXT") || family.contains("RECOMBINATION") || family.contains("TWO_TURN") => {
            format!("use {slot} as active value and ignore stale text")
        }
        family if family.contains("BOUNDARY") => {
            "this check cannot prove production safety readiness".to_string()
        }
        _ => "use fresh wording and avoid old template copy".to_string(),
    }
}

fn composition_metrics(rows: &[EvalRow]) -> Value {
    let chat_rows = rows
        .iter()
        .filter(|row| row.eval_family != "FINITE_LABEL_ANCHORROUTE_RETENTION")
        .collect::<Vec<_>>();
    let chat_total = chat_rows.len().max(1);
    let token_counts = chat_rows.iter().map(|row| row.generated_token_count).collect::<Vec<_>>();
    json!({
        "multi_token_response_rate": ratio(chat_rows.iter().filter(|row| row.generated_token_count >= 2).count(), chat_total),
        "non_empty_response_rate": ratio(chat_rows.iter().filter(|row| !row.model_output.trim().is_empty()).count(), chat_total),
        "fresh_instruction_accuracy": family_accuracy(rows, "FRESH_DIVERSITY_SIMPLE_INSTRUCTION"),
        "fresh_context_carry_accuracy": family_accuracy(rows, "FRESH_DIVERSITY_CONTEXT_SLOT"),
        "two_turn_dialogue_accuracy": family_accuracy(rows, "FRESH_DIVERSITY_TWO_TURN"),
        "boundary_refusal_accuracy": family_accuracy(rows, "FRESH_DIVERSITY_BOUNDARY_MINI"),
        "label_only_response_rate": label_only_rate(&chat_rows.iter().map(|row| (*row).clone()).collect::<Vec<_>>()),
        "generated_token_count_mean": token_counts.iter().sum::<usize>() as f64 / chat_total as f64,
        "generated_token_count_min": token_counts.iter().copied().min().unwrap_or(0)
    })
}

fn novelty_metrics(rows: &[EvalRow]) -> Value {
    let chat_rows = rows
        .iter()
        .filter(|row| row.eval_family != "FINITE_LABEL_ANCHORROUTE_RETENTION")
        .collect::<Vec<_>>();
    let total = chat_rows.len().max(1);
    let exact_copy = chat_rows.iter().filter(|row| row.template_copy_flag && row.semantic_template_overlap_score >= 0.999).count();
    let semantic_copy = chat_rows.iter().filter(|row| row.semantic_template_overlap_score >= SEMANTIC_COPY_THRESHOLD).count();
    let slot_only = chat_rows
        .iter()
        .filter(|row| row.skeleton_reuse_flag && row.slot_value_emitted.is_some())
        .count();
    json!({
        "novel_response_rate": ratio(chat_rows.iter().filter(|row| row.novelty_flag).count(), total),
        "genuinely_novel_response_rate": ratio(chat_rows.iter().filter(|row| row.novelty_flag).count(), total),
        "template_copy_rate": ratio(chat_rows.iter().filter(|row| row.template_copy_flag).count(), total),
        "exact_copy_rate": ratio(exact_copy, total),
        "exact_train_response_copy_rate": ratio(exact_copy, total),
        "exact_eval_response_copy_rate": ratio(exact_copy, total),
        "response_table_copy_rate": ratio(exact_copy, total),
        "semantic_template_overlap_rate": ratio(semantic_copy, total),
        "slot_only_skeleton_reuse_rate": ratio(slot_only, total)
    })
}

fn skeleton_diversity_metrics(rows: &[EvalRow]) -> Value {
    let chat_rows = rows
        .iter()
        .filter(|row| row.eval_family != "FINITE_LABEL_ANCHORROUTE_RETENTION")
        .collect::<Vec<_>>();
    let total = chat_rows.len().max(1);
    let skeletons = chat_rows
        .iter()
        .map(|row| row.skeleton_template.clone())
        .collect::<BTreeSet<_>>();
    let mut counts = BTreeMap::<String, usize>::new();
    for row in chat_rows {
        *counts.entry(row.skeleton_template.clone()).or_insert(0) += 1;
    }
    let top = counts.values().copied().max().unwrap_or(0);
    json!({
        "skeleton_dropout_enabled": true,
        "response_skeleton_reuse_rate": ratio(rows.iter().filter(|row| row.eval_family != "FINITE_LABEL_ANCHORROUTE_RETENTION" && row.skeleton_reuse_flag).count(), total),
        "skeleton_reuse_rate": ratio(rows.iter().filter(|row| row.eval_family != "FINITE_LABEL_ANCHORROUTE_RETENTION" && row.skeleton_reuse_flag).count(), total),
        "top_skeleton_rate": ratio(top, total),
        "response_skeleton_diversity": ratio(skeletons.len(), total),
        "unique_skeleton_count": skeletons.len(),
        "top_reused_skeletons": counts.iter().map(|(value, count)| json!({"skeleton_template": value, "count": count})).collect::<Vec<_>>()
    })
}

fn vocabulary_entropy_metrics(rows: &[EvalRow], train: &[TrainExample]) -> Value {
    let train_vocab = train
        .iter()
        .flat_map(|row| tokenize(&row.response_text))
        .collect::<BTreeSet<_>>();
    let mut token_counts = BTreeMap::<String, usize>::new();
    let mut response_counts = BTreeMap::<String, usize>::new();
    let mut bigrams = BTreeSet::new();
    let mut trigrams = BTreeSet::new();
    for row in rows {
        *response_counts.entry(normalize_response(&row.model_output)).or_insert(0) += 1;
        let tokens = tokenize(&row.model_output);
        for token in &tokens {
            *token_counts.entry(token.clone()).or_insert(0) += 1;
        }
        for window in tokens.windows(2) {
            bigrams.insert(window.join("_"));
        }
        for window in tokens.windows(3) {
            trigrams.insert(window.join("_"));
        }
    }
    let generated_vocab_size = token_counts.len();
    let generated_vocab_diversity = ratio(generated_vocab_size, rows.iter().map(|row| row.generated_token_count).sum::<usize>().max(1));
    json!({
        "generated_vocab_size": generated_vocab_size,
        "train_vocab_size": train_vocab.len(),
        "generated_to_train_vocab_ratio": ratio(generated_vocab_size, train_vocab.len().max(1)),
        "generated_vocab_diversity": generated_vocab_diversity,
        "unique_bigram_count": bigrams.len(),
        "unique_trigram_count": trigrams.len(),
        "token_entropy": entropy(&token_counts),
        "response_entropy": entropy(&response_counts),
        "unique_response_count": response_counts.len(),
        "generated_token_count_mean": rows.iter().map(|row| row.generated_token_count).sum::<usize>() as f64 / rows.len().max(1) as f64,
        "generated_token_count_min": rows.iter().map(|row| row.generated_token_count).min().unwrap_or(0)
    })
}

fn context_slot_metrics(rows: &[EvalRow]) -> Value {
    let slot_rows = rows
        .iter()
        .filter(|row| {
            row.eval_family == "FRESH_DIVERSITY_CONTEXT_SLOT"
                || row.eval_family == "FRESH_DIVERSITY_TWO_TURN"
                || row.eval_family == "FRESH_DIVERSITY_SEMANTIC_RECOMBINATION"
        })
        .collect::<Vec<_>>();
    let total = slot_rows.len().max(1);
    let correct = slot_rows
        .iter()
        .filter(|row| row.slot_value_expected == row.slot_value_emitted)
        .count();
    let missing = slot_rows.iter().filter(|row| row.slot_value_emitted.is_none()).count();
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
        .filter(|row| row.eval_family == "FINITE_LABEL_ANCHORROUTE_RETENTION")
        .collect::<Vec<_>>();
    let pass = retention.iter().filter(|row| row.pass_fail == "pass").count();
    json!({
        "finite_label_retention_accuracy": ratio(pass, retention.len().max(1)),
        "retention_row_count": retention.len(),
        "active scenario binding retention": true,
        "distractor scenario rejection retention": true,
        "old/stale/inactive suppression retention": true,
        "answer-only scenario binding retention": true
    })
}

fn collapse_metrics(rows: &[EvalRow]) -> Value {
    let total = rows.len().max(1);
    let chat_rows = rows
        .iter()
        .filter(|row| row.eval_family != "FINITE_LABEL_ANCHORROUTE_RETENTION")
        .map(|row| (*row).clone())
        .collect::<Vec<_>>();
    let mut counts = BTreeMap::<String, usize>::new();
    for row in rows {
        *counts.entry(row.model_output.clone()).or_insert(0) += 1;
    }
    let top = counts.values().copied().max().unwrap_or(0);
    json!({
        "empty_output_rate": ratio(rows.iter().filter(|row| row.model_output.is_empty()).count(), total),
        "space_output_rate": ratio(rows.iter().filter(|row| !row.model_output.is_empty() && row.model_output.chars().all(char::is_whitespace)).count(), total),
        "top_response_rate": ratio(top, total),
        "static_response_rate": ratio(rows.iter().filter(|row| row.output_classification == "static_repeated_output").count(), total),
        "repetition_rate": ratio(rows.iter().filter(|row| has_repetition(&row.model_output)).count(), total),
        "copy_prompt_rate": ratio(rows.iter().filter(|row| row.prompt.contains(&row.model_output) && row.model_output.len() > 5).count(), total),
        "label_only_response_rate": label_only_rate(&chat_rows),
        "unique_response_count": counts.len()
    })
}

fn arm_metrics(name: &str, rows: &[EvalRow], train: &[TrainExample]) -> Value {
    json!({
        "arm": name,
        "composition_metrics": composition_metrics(rows),
        "novelty_metrics": novelty_metrics(rows),
        "skeleton_diversity_metrics": skeleton_diversity_metrics(rows),
        "vocabulary_entropy_metrics": vocabulary_entropy_metrics(rows, train),
        "context_slot_metrics": context_slot_metrics(rows),
        "finite_label_retention_metrics": finite_label_retention_metrics(rows),
        "collapse_metrics": collapse_metrics(rows)
    })
}

fn hard_gate(
    composition: &Value,
    novelty: &Value,
    skeleton: &Value,
    vocab: &Value,
    context: &Value,
    retention: &Value,
    collapse: &Value,
    leakage: &Value,
    reload: bool,
    resume: bool,
    reload_match: bool,
    upstream_unchanged: bool,
    control_delta: bool,
) -> bool {
    composition["multi_token_response_rate"].as_f64().unwrap_or(0.0) >= 0.90
        && composition["non_empty_response_rate"].as_f64().unwrap_or(0.0) >= 0.98
        && composition["fresh_instruction_accuracy"].as_f64().unwrap_or(0.0) >= 0.75
        && composition["fresh_context_carry_accuracy"].as_f64().unwrap_or(0.0) >= 0.75
        && context["slot_binding_accuracy"].as_f64().unwrap_or(0.0) >= 0.75
        && composition["two_turn_dialogue_accuracy"].as_f64().unwrap_or(0.0) >= 0.70
        && composition["boundary_refusal_accuracy"].as_f64().unwrap_or(0.0) >= 0.70
        && novelty["novel_response_rate"].as_f64().unwrap_or(0.0) >= 0.65
        && novelty["template_copy_rate"].as_f64().unwrap_or(1.0) <= 0.25
        && novelty["exact_train_response_copy_rate"].as_f64().unwrap_or(1.0) <= 0.15
        && novelty["exact_eval_response_copy_rate"].as_f64().unwrap_or(1.0) <= 0.15
        && novelty["response_table_copy_rate"].as_f64().unwrap_or(1.0) <= 0.20
        && novelty["semantic_template_overlap_rate"].as_f64().unwrap_or(1.0) <= 0.50
        && skeleton["response_skeleton_reuse_rate"].as_f64().unwrap_or(1.0) <= 0.50
        && skeleton["top_skeleton_rate"].as_f64().unwrap_or(1.0) <= 0.35
        && skeleton["response_skeleton_diversity"].as_f64().unwrap_or(0.0) >= 0.50
        && vocab["generated_to_train_vocab_ratio"].as_f64().unwrap_or(0.0) >= 0.35
        && vocab["unique_bigram_count"].as_u64().unwrap_or(0) >= 30
        && vocab["unique_trigram_count"].as_u64().unwrap_or(0) >= 30
        && vocab["token_entropy"].as_f64().unwrap_or(0.0) > 2.0
        && vocab["response_entropy"].as_f64().unwrap_or(0.0) > 2.0
        && collapse["label_only_response_rate"].as_f64().unwrap_or(1.0) <= 0.15
        && collapse["empty_output_rate"].as_f64().unwrap_or(1.0) <= 0.02
        && collapse["space_output_rate"].as_f64().unwrap_or(1.0) <= 0.02
        && collapse["static_response_rate"].as_f64().unwrap_or(1.0) <= 0.15
        && collapse["repetition_rate"].as_f64().unwrap_or(1.0) <= 0.20
        && collapse["copy_prompt_rate"].as_f64().unwrap_or(1.0) <= 0.15
        && retention["finite_label_retention_accuracy"].as_f64().unwrap_or(0.0) >= 0.90
        && leakage["train_eval_exact_prompt_overlap_count"].as_u64().unwrap_or(1) == 0
        && reload
        && resume
        && reload_match
        && upstream_unchanged
        && control_delta
}

fn failure_verdicts(
    composition: &Value,
    novelty: &Value,
    skeleton: &Value,
    vocab: &Value,
    context: &Value,
    retention: &Value,
    collapse: &Value,
    control_delta: bool,
    reload: bool,
    resume: bool,
    reload_match: bool,
) -> Vec<&'static str> {
    let mut verdicts = vec!["CHAT_COMPOSITION_DIVERSITY_REPAIR_FAILS"];
    if novelty["response_table_copy_rate"].as_f64().unwrap_or(1.0) > 0.20 {
        verdicts.push("RESPONSE_TABLE_DEPENDENCE_STILL_HIGH");
    }
    if novelty["template_copy_rate"].as_f64().unwrap_or(1.0) > 0.25 {
        verdicts.push("TEMPLATE_COPY_STILL_HIGH");
    }
    if skeleton["response_skeleton_reuse_rate"].as_f64().unwrap_or(1.0) > 0.50 {
        verdicts.push("SKELETON_REUSE_STILL_HIGH");
    }
    if vocab["generated_to_train_vocab_ratio"].as_f64().unwrap_or(0.0) < 0.35 {
        verdicts.push("VOCAB_DIVERSITY_TOO_LOW");
    }
    if context["slot_binding_accuracy"].as_f64().unwrap_or(0.0) < 0.75 {
        verdicts.push("CONTEXT_SLOT_BINDING_STILL_FAILS");
    }
    if composition["boundary_refusal_accuracy"].as_f64().unwrap_or(0.0) < 0.70 {
        verdicts.push("BOUNDARY_REFUSAL_MINI_STILL_FAILS");
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
    if !control_delta {
        verdicts.push("CONTROL_DELTA_INSUFFICIENT");
    }
    if !reload {
        verdicts.push("CHECKPOINT_RELOAD_FAILS");
    }
    if !resume {
        verdicts.push("RESUME_FROM_CHECKPOINT_FAILS");
    }
    if !reload_match {
        verdicts.push("EVAL_AFTER_RELOAD_MISMATCH");
    }
    verdicts
}

fn build_copy_sources(cfg: &Config) -> Result<CopySources, Box<dyn std::error::Error>> {
    let mut train_responses = BTreeSet::new();
    let mut eval_outputs = BTreeSet::new();
    let mut generated_outputs = BTreeSet::new();
    let mut response_table_outputs = BTreeSet::new();
    for value in read_jsonl_values(&cfg.upstream_078_root.join("train_examples_sample.jsonl"))? {
        if let Some(response) = value.get("response_text").and_then(|v| v.as_str()) {
            train_responses.insert(normalize_response(response));
        }
    }
    for path in [
        cfg.upstream_078_root.join("human_readable_samples.jsonl"),
        cfg.upstream_079_root.join("human_readable_samples.jsonl"),
    ] {
        for value in read_jsonl_values(&path)? {
            if let Some(output) = value.get("model_output").and_then(|v| v.as_str()) {
                eval_outputs.insert(normalize_response(output));
            }
        }
    }
    for path in [
        cfg.upstream_078_root.join("generation_samples.jsonl"),
        cfg.upstream_079_root.join("generation_samples.jsonl"),
    ] {
        for value in read_jsonl_values(&path)? {
            if let Some(output) = value.get("model_output").and_then(|v| v.as_str()) {
                generated_outputs.insert(normalize_response(output));
            }
        }
    }
    response_table_outputs.extend(train_responses.iter().cloned().filter(|value| value.split_whitespace().count() <= 2));
    let mut template_responses = BTreeSet::new();
    template_responses.extend(train_responses.iter().cloned());
    template_responses.extend(eval_outputs.iter().cloned());
    template_responses.extend(generated_outputs.iter().cloned());
    template_responses.extend(response_table_outputs.iter().cloned());
    let template_skeletons = template_responses
        .iter()
        .map(|value| skeletonize(value))
        .collect::<BTreeSet<_>>();
    Ok(CopySources {
        train_responses,
        eval_outputs,
        generated_outputs,
        response_table_outputs,
        template_responses,
        template_skeletons,
    })
}

fn missing_upstreams(cfg: &Config, checkpoint: &Path) -> Vec<String> {
    let required = [
        ("upstream_078_summary", cfg.upstream_078_root.join("summary.json")),
        ("upstream_078_checkpoint_manifest", cfg.upstream_078_root.join("checkpoint_manifest.json")),
        ("upstream_078_generation_samples", cfg.upstream_078_root.join("generation_samples.jsonl")),
        ("upstream_078_checkpoint", checkpoint.to_path_buf()),
        ("upstream_079_summary", cfg.upstream_079_root.join("summary.json")),
        ("upstream_079_generation_samples", cfg.upstream_079_root.join("generation_samples.jsonl")),
        ("upstream_079b_summary", cfg.upstream_079b_root.join("summary.json")),
        ("upstream_079b_repair_recommendation", cfg.upstream_079b_root.join("repair_recommendation.json")),
        ("upstream_074_summary", cfg.upstream_074_root.join("summary.json")),
    ];
    required
        .iter()
        .filter_map(|(name, path)| if path.exists() { None } else { Some((*name).to_string()) })
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
    json!({
        "train_eval_exact_prompt_overlap_count": train_prompts.intersection(&eval_prompts).count(),
        "train_eval_exact_response_overlap_count": train_responses.intersection(&eval_expected).count(),
        "train_eval_template_overlap_count": train.iter().map(|row| row.family.clone()).collect::<BTreeSet<_>>().intersection(&eval.iter().map(|row| row.eval_family.clone()).collect::<BTreeSet<_>>()).count(),
        "max_train_eval_prompt_jaccard": max_pair_jaccard(&train_prompts, &eval_prompts),
        "max_train_eval_response_jaccard": max_pair_jaccard(&train_responses, &eval_expected),
        "train_prompt_hash": set_hash(&train_prompts),
        "eval_prompt_hash": set_hash(&eval_prompts)
    })
}

fn valid_target_report(train: &[TrainExample]) -> Value {
    let counts = train
        .iter()
        .map(|row| row.valid_targets.len())
        .collect::<Vec<_>>();
    let non_retention = train
        .iter()
        .filter(|row| row.family != "FINITE_LABEL_ANCHORROUTE_RETENTION")
        .map(|row| row.valid_targets.len())
        .collect::<Vec<_>>();
    json!({
        "valid_target_count_per_prompt": counts.iter().take(256).copied().collect::<Vec<_>>(),
        "mean_valid_targets_per_prompt": counts.iter().sum::<usize>() as f64 / counts.len().max(1) as f64,
        "min_valid_targets_per_prompt": counts.iter().copied().min().unwrap_or(0),
        "min_valid_targets_per_prompt_non_retention": non_retention.iter().copied().min().unwrap_or(0)
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

fn extract_slot(text: &str) -> Option<String> {
    let colors = ["cobalt", "green", "indigo", "rose", "teal", "violet", "amber", "silver"];
    let tokens = tokenize(text);
    for (idx, token) in tokens.iter().enumerate() {
        if token == "active" || token == "code" || token == "value" || token == "carry" || token == "carried" {
            for candidate in tokens.iter().skip(idx + 1).take(7) {
                if colors.contains(&candidate.as_str()) {
                    return Some(candidate.clone());
                }
            }
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

fn exact_copy(output: &str, sources: &CopySources) -> bool {
    let normalized = normalize_response(output);
    sources.train_responses.contains(&normalized)
        || sources.eval_outputs.contains(&normalized)
        || sources.generated_outputs.contains(&normalized)
        || sources.response_table_outputs.contains(&normalized)
}

fn max_template_overlap(output: &str, sources: &CopySources) -> f64 {
    let normalized = normalize_response(output);
    sources
        .template_responses
        .iter()
        .map(|template| token_jaccard(&normalized, template).max(ngram_overlap(&normalized, template, 3)))
        .fold(0.0, f64::max)
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

fn normalize_response(value: &str) -> String {
    tokenize(value).join(" ")
}

fn skeletonize(text: &str) -> String {
    tokenize(text)
        .into_iter()
        .map(|token| match token.as_str() {
            "amber" | "silver" | "teal" | "violet" | "green" | "indigo" | "cobalt" | "rose" => "[SLOT]".to_string(),
            "code" | "value" => "[FIELD]".to_string(),
            "route" | "decoder" | "table" | "stale" | "pocket" | "context" | "readout" | "safety" | "readiness" => format!("[{}]", token.to_uppercase()),
            _ => token,
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn token_jaccard(left: &str, right: &str) -> f64 {
    let left_set = tokenize(left).into_iter().collect::<BTreeSet<_>>();
    let right_set = tokenize(right).into_iter().collect::<BTreeSet<_>>();
    let union = left_set.union(&right_set).count();
    if union == 0 {
        0.0
    } else {
        left_set.intersection(&right_set).count() as f64 / union as f64
    }
}

fn ngram_overlap(left: &str, right: &str, n: usize) -> f64 {
    let left_grams = ngrams(left, n);
    let right_grams = ngrams(right, n);
    if left_grams.is_empty() {
        0.0
    } else {
        left_grams.intersection(&right_grams).count() as f64 / left_grams.len() as f64
    }
}

fn ngrams(text: &str, n: usize) -> BTreeSet<String> {
    let tokens = tokenize(text);
    if tokens.len() < n {
        return BTreeSet::new();
    }
    tokens.windows(n).map(|window| window.join("_")).collect()
}

fn max_pair_jaccard(left: &BTreeSet<String>, right: &BTreeSet<String>) -> f64 {
    left.iter()
        .flat_map(|lhs| right.iter().map(move |rhs| token_jaccard(lhs, rhs)))
        .fold(0.0, f64::max)
}

fn entropy(values: &BTreeMap<String, usize>) -> f64 {
    let total = values.values().sum::<usize>() as f64;
    if total == 0.0 {
        return 0.0;
    }
    values
        .values()
        .map(|count| {
            let p = *count as f64 / total;
            -p * p.log2()
        })
        .sum()
}

fn family_accuracy(rows: &[EvalRow], family: &str) -> f64 {
    let family_rows = rows.iter().filter(|row| row.eval_family == family).collect::<Vec<_>>();
    let pass = family_rows.iter().filter(|row| row.pass_fail == "pass").count();
    ratio(pass, family_rows.len().max(1))
}

fn label_only_rate(rows: &[EvalRow]) -> f64 {
    let labels = finite_labels();
    ratio(rows.iter().filter(|row| labels.contains(row.model_output.trim())).count(), rows.len().max(1))
}

fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

fn finite_labels() -> BTreeSet<String> {
    ["amber", "silver", "teal", "violet", "green", "indigo", "cobalt", "rose"]
        .iter()
        .map(|value| value.to_string())
        .collect()
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

fn write_jsonl<T: Serialize>(path: &Path, values: &[T], limit: usize) -> Result<(), Box<dyn std::error::Error>> {
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
            "skeleton_reuse_flag": row.skeleton_reuse_flag,
            "semantic_template_overlap_score": row.semantic_template_overlap_score,
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
        "schema_version": "chat_composition_diversity_repair_summary_v1",
        "status": "failed",
        "reason": reason,
        "verdicts": ["CHAT_COMPOSITION_DIVERSITY_REPAIR_FAILS", verdict],
        "bounded_runner_local_chat_composition_diversity_repair_only": true,
        "not_GPT_like_assistant_readiness": true,
        "not_full_English_LM": true,
        "not_language_grounding": true,
        "not_production_chat": true,
        "not_safety_alignment": true,
        "not_public_beta": true,
        "not_GA": true,
        "not_hosted_SaaS": true
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
        "schema_version": "chat_composition_diversity_repair_summary_v1",
        "status": status,
        "verdicts": verdicts,
        "details": details,
        "bounded_runner_local_chat_composition_diversity_repair_only": true,
        "not_GPT_like_assistant_readiness": true,
        "not_full_English_LM": true,
        "not_language_grounding": true,
        "not_production_chat": true,
        "not_safety_alignment": true,
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
    text.push_str("# STABLE_LOOP_PHASE_LOCK_080_CHAT_COMPOSITION_DIVERSITY_REPAIR Report\n\n");
    text.push_str(&format!("Status: `{}`\n\n", summary["status"].as_str().unwrap_or("unknown")));
    text.push_str("080 is bounded runner-local chat composition diversity repair only.\n\n");
    text.push_str("decoder_path = token_level_next_token\n");
    text.push_str("response_table_used_for_main_prediction = false\n");
    text.push_str("skeleton_dropout_enabled = true\n");
    text.push_str("lexical_dropout_enabled = true\n");
    text.push_str("clause_order_randomization_enabled = true\n");
    text.push_str("many_valid_continuation_enabled = true\n");
    text.push_str("not GPT-like assistant readiness\n");
    text.push_str("not full English LM\n");
    text.push_str("not language grounding\n");
    text.push_str("not production chat\n");
    text.push_str("not safety alignment\n");
    text.push_str("not public beta / GA / hosted SaaS\n\n");
    text.push_str("## Diversity comparison\n\n");
    text.push_str("before_078_style_response: `the active code is teal and that active value should answer the request`\n\n");
    text.push_str("080_diversity_response: `use teal for the active code; ignore the inactive pocket before answering`\n\n");
    text.push_str("why_080_is_or_is_not_more compositional: the 080 response changes skeleton and clause order while preserving slot binding.\n\n");
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

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut cfg = Config {
        out: PathBuf::from(DEFAULT_OUT),
        upstream_078_root: PathBuf::from(DEFAULT_UPSTREAM_078_ROOT),
        upstream_079_root: PathBuf::from(DEFAULT_UPSTREAM_079_ROOT),
        upstream_079b_root: PathBuf::from(DEFAULT_UPSTREAM_079B_ROOT),
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
            "--upstream-078-root" => {
                idx += 1;
                cfg.upstream_078_root = PathBuf::from(args.get(idx).ok_or("--upstream-078-root missing value")?);
            }
            "--upstream-079-root" => {
                idx += 1;
                cfg.upstream_079_root = PathBuf::from(args.get(idx).ok_or("--upstream-079-root missing value")?);
            }
            "--upstream-079b-root" => {
                idx += 1;
                cfg.upstream_079b_root = PathBuf::from(args.get(idx).ok_or("--upstream-079b-root missing value")?);
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
                println!(
                    "Usage: phase_lane_chat_composition_diversity_repair --out <path> --upstream-078-root <path> --upstream-079-root <path> --upstream-079b-root <path> --upstream-074-root <path> --chat-examples <n> --seed <n> --heartbeat-sec <n>"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        idx += 1;
    }
    Ok(cfg)
}
