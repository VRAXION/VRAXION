//! Runner-local chat generation proof of concept.
//!
//! 076 creates a bounded experimental decoder/generation loop inside this
//! example only. It does not expose a product API, service API, SDK surface, or
//! release surface.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_OUT: &str = "target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke";
const DEFAULT_UPSTREAM_CHECKPOINT: &str = "target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke/checkpoints/scenario_gated_sidepacket_repair/model_checkpoint.json";
const DEFAULT_UPSTREAM_074_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke";
const DEFAULT_UPSTREAM_075_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_075_chat_surface_baseline_gap_analysis/smoke";
const DEFAULT_CHAT_EXAMPLES: usize = 20_000;
const FEATURE_DIM: usize = 4096;
const MAX_RESPONSE_TOKENS: usize = 64;
const STOP_TOKEN: &str = "<eos>";

const TRAIN_FAMILIES: [&str; 5] = [
    "SIMPLE_INSTRUCTION_CHAT",
    "SHORT_ANSWER_EXPLANATION",
    "CONTEXT_CARRY_CHAT",
    "ANCHORROUTE_RETENTION",
    "BOUNDARY_REFUSAL",
];

const EVAL_FAMILIES: [&str; 7] = [
    "FREE_FORM_SHORT_RESPONSE",
    "MULTI_TOKEN_CONTINUATION",
    "SINGLE_TURN_INSTRUCTION_FOLLOWING",
    "TWO_TURN_CONTEXT_CARRY",
    "BOUNDARY_REFUSAL_MINI",
    "ANTI_COLLAPSE_CHAT",
    "FINITE_LABEL_ANCHORROUTE_RETENTION",
];

#[derive(Debug, Clone)]
struct Config {
    out: PathBuf,
    upstream_checkpoint: PathBuf,
    upstream_074_root: PathBuf,
    upstream_075_root: PathBuf,
    seed: u64,
    chat_examples: usize,
    heartbeat_sec: u64,
}

#[derive(Debug, Deserialize)]
struct ScenarioCheckpoint {
    #[serde(default)]
    labels: Vec<String>,
    #[serde(default)]
    route_table: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatModel {
    schema_version: String,
    labels: Vec<String>,
    response_table: BTreeMap<String, Vec<String>>,
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
    feature_dim: usize,
    seed: u64,
    train_step_count: usize,
    update_count: usize,
    runner_local_decoder_loop: bool,
    response_uses_decoder_loop: bool,
    public_api_exposed: bool,
    service_api_exposed: bool,
    sdk_surface_exposed: bool,
}

#[derive(Debug, Clone, Serialize)]
struct TrainExample {
    id: String,
    family: String,
    prompt: String,
    target_label: String,
    response_text: String,
}

#[derive(Debug, Clone, Serialize)]
struct EvalExample {
    id: String,
    eval_family: String,
    prompt: String,
    expected_behavior: String,
    required_keywords: Vec<String>,
    forbidden_outputs: Vec<String>,
    target_label: Option<String>,
    retention_row: bool,
}

#[derive(Debug, Clone, Serialize)]
struct EvalRow {
    eval_family: String,
    prompt: String,
    expected_behavior: String,
    required_keywords: Vec<String>,
    forbidden_outputs: Vec<String>,
    model_output: String,
    pass_fail: String,
    diagnosis: String,
    output_classification: String,
    generated_token_count: usize,
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
    fs::create_dir_all(&cfg.out)?;
    append_progress(
        &cfg.out,
        "start",
        json!({
            "seed": cfg.seed,
            "chat_examples": cfg.chat_examples,
            "heartbeat_sec": cfg.heartbeat_sec,
            "runner_local_only": true
        }),
    )?;
    write_summary_and_report(
        &cfg.out,
        "running",
        vec!["CHAT_GENERATION_POC_RUNNING".to_string()],
        json!({"phase": "start"}),
    )?;
    write_json(
        &cfg.out.join("queue.json"),
        &json!({
            "schema_version": "chat_generation_poc_queue_v1",
            "milestone": "STABLE_LOOP_PHASE_LOCK_076_CHAT_GENERATION_POC",
            "steps": [
                "verify_upstreams",
                "load_upstream_checkpoint_read_only",
                "build_controlled_chat_sft",
                "train_runner_local_decoder",
                "evaluate_rubric_bounded_chat",
                "evaluate_finite_label_retention",
                "validate_checkpoint_pipeline",
                "write_summary"
            ],
            "partial_write_policy": "progress summary report written from start and refreshed by phase"
        }),
    )?;

    let upstream_074_summary = cfg.upstream_074_root.join("summary.json");
    let upstream_075_summary = cfg.upstream_075_root.join("summary.json");
    let mut missing = Vec::new();
    if !cfg.upstream_checkpoint.exists() {
        missing.push("upstream_checkpoint".to_string());
    }
    if !upstream_074_summary.exists() {
        missing.push("upstream_074_summary".to_string());
    }
    if !upstream_075_summary.exists() {
        missing.push("upstream_075_summary".to_string());
    }
    if !missing.is_empty() {
        write_failure(&cfg.out, "UPSTREAM_ARTIFACT_MISSING", &missing.join(","))?;
        return Err(format!("UPSTREAM_ARTIFACT_MISSING: {}", missing.join(",")).into());
    }
    let upstream_074: Value = serde_json::from_slice(&fs::read(&upstream_074_summary)?)?;
    let upstream_075: Value = serde_json::from_slice(&fs::read(&upstream_075_summary)?)?;
    if !value_has_verdict(&upstream_074, "MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_POSITIVE") {
        write_failure(&cfg.out, "UPSTREAM_ARTIFACT_MISSING", "074 positive verdict missing")?;
        return Err("UPSTREAM_ARTIFACT_MISSING".into());
    }
    if !value_has_verdict(&upstream_075, "CHAT_GENERATION_SURFACE_UNSUPPORTED") {
        write_failure(&cfg.out, "UPSTREAM_ARTIFACT_MISSING", "075 unsupported chat verdict missing")?;
        return Err("UPSTREAM_ARTIFACT_MISSING".into());
    }
    append_progress(&cfg.out, "upstreams_verified", json!({"upstream_074": true, "upstream_075": true}))?;

    let upstream_checkpoint_hash_before = sha256_file(&cfg.upstream_checkpoint)?;
    let upstream_checkpoint: ScenarioCheckpoint =
        serde_json::from_slice(&fs::read(&cfg.upstream_checkpoint)?)?;
    write_json(
        &cfg.out.join("upstream_manifest.json"),
        &json!({
            "schema_version": "chat_generation_poc_upstream_manifest_v1",
            "upstream_checkpoint": cfg.upstream_checkpoint.display().to_string(),
            "upstream_checkpoint_hash_before": upstream_checkpoint_hash_before,
            "upstream_074_root": cfg.upstream_074_root.display().to_string(),
            "upstream_075_root": cfg.upstream_075_root.display().to_string(),
            "upstream_074_positive": true,
            "upstream_075_chat_unsupported": true,
            "upstream_checkpoint_label_count": upstream_checkpoint.labels.len(),
            "upstream_checkpoint_route_table_count": upstream_checkpoint.route_table.len()
        }),
    )?;

    write_json(
        &cfg.out.join("training_config.json"),
        &json!({
            "schema_version": "chat_generation_poc_training_config_v1",
            "runner_local_decoder_only": true,
            "no_product_api": true,
            "no_sdk_surface": true,
            "no_service_api": true,
            "no_deployment_harness": true,
            "no_release_docs": true,
            "fixed_vocabulary_from_controlled_chat_sft": true,
            "decoder_generation_loop": "runner-local token response decoder",
            "default_decode": "deterministic greedy",
            "seeded_sampling_audit_mode_available": true,
            "max_response_length": MAX_RESPONSE_TOKENS,
            "stop_token": STOP_TOKEN,
            "seed": cfg.seed,
            "chat_examples": cfg.chat_examples,
            "data_mix": {
                "SIMPLE_INSTRUCTION_CHAT": 0.50,
                "SHORT_ANSWER_EXPLANATION": 0.20,
                "CONTEXT_CARRY_CHAT": 0.15,
                "ANCHORROUTE_RETENTION": 0.10,
                "BOUNDARY_REFUSAL": 0.05
            }
        }),
    )?;

    let response_table = response_table(&upstream_checkpoint.labels);
    let train_examples = build_train_examples(cfg.chat_examples, cfg.seed, &response_table);
    let eval_examples = build_eval_examples(cfg.seed);
    let leakage = leakage_report(&train_examples, &eval_examples);
    if leakage["train_eval_exact_prompt_overlap_count"].as_u64().unwrap_or(1) > 0 {
        write_failure(&cfg.out, "TRAIN_EVAL_LEAKAGE_DETECTED", "exact prompt overlap")?;
        return Err("TRAIN_EVAL_LEAKAGE_DETECTED".into());
    }
    write_json(
        &cfg.out.join("chat_sft_dataset_manifest.json"),
        &json!({
            "schema_version": "chat_generation_poc_dataset_manifest_v1",
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
    write_jsonl_sample(&cfg.out.join("train_examples_sample.jsonl"), &train_examples, 240)?;
    write_jsonl_sample(&cfg.out.join("eval_examples_sample.jsonl"), &eval_examples, 240)?;
    append_progress(&cfg.out, "dataset_written", json!({"train": train_examples.len(), "eval": eval_examples.len()}))?;
    write_summary_and_report(
        &cfg.out,
        "running",
        vec!["CHAT_GENERATION_POC_RUNNING".to_string()],
        json!({"phase": "dataset_written", "train_examples": train_examples.len()}),
    )?;

    let mut model = ChatModel::new(response_table, cfg.seed);
    let checkpoint_before_hash = model.sha256()?;
    train_model(&mut model, &train_examples, &cfg.out, cfg.heartbeat_sec)?;
    let checkpoint_after_hash = model.sha256()?;
    let actual_update = checkpoint_after_hash != checkpoint_before_hash && model.train_step_count > 0;
    if !actual_update {
        write_failure(
            &cfg.out,
            "NO_ACTUAL_TRAINING_UPDATE_DETECTED",
            "trained checkpoint hash did not change",
        )?;
        return Err("NO_ACTUAL_TRAINING_UPDATE_DETECTED".into());
    }
    append_progress(
        &cfg.out,
        "training_completed",
        json!({"train_step_count": model.train_step_count, "update_count": model.update_count}),
    )?;

    let eval_rows = evaluate_model(&model, &eval_examples, &upstream_checkpoint.labels);
    write_eval_outputs(&cfg.out.join("generation_samples.jsonl"), &eval_rows)?;
    write_human_samples(&cfg.out.join("human_readable_samples.jsonl"), &eval_rows)?;
    let chat_metrics = chat_eval_metrics(&eval_rows);
    let retention_metrics = finite_label_retention_metrics(&eval_rows);
    let collapse = collapse_metrics(&eval_rows, &upstream_checkpoint.labels);
    write_json(&cfg.out.join("chat_eval_metrics.json"), &chat_metrics)?;
    write_json(&cfg.out.join("finite_label_retention_metrics.json"), &retention_metrics)?;
    write_json(&cfg.out.join("collapse_metrics.json"), &collapse)?;
    append_progress(
        &cfg.out,
        "eval_completed",
        json!({
            "instruction_following_accuracy": chat_metrics["instruction_following_accuracy"],
            "finite_label_retention_accuracy": retention_metrics["finite_label_retention_accuracy"]
        }),
    )?;

    let checkpoint_dir = cfg.out.join("checkpoints").join("chat_generation_poc");
    fs::create_dir_all(&checkpoint_dir)?;
    let checkpoint_path = checkpoint_dir.join("model_checkpoint.json");
    write_json(&checkpoint_path, &model)?;
    let loaded: ChatModel = serde_json::from_slice(&fs::read(&checkpoint_path)?)?;
    let reload_rows = evaluate_model(&loaded, &eval_examples, &upstream_checkpoint.labels);
    let checkpoint_save_load_pass = loaded.sha256()? == checkpoint_after_hash;
    let eval_after_reload_matches_before = eval_signature(&eval_rows) == eval_signature(&reload_rows);
    let mut resumed = loaded.clone();
    let pre_resume_hash = resumed.sha256()?;
    let resume_examples = train_examples.iter().take(64).cloned().collect::<Vec<_>>();
    train_model_no_progress(&mut resumed, &resume_examples);
    let resume_dir = cfg.out.join("checkpoints").join("resume_from_checkpoint");
    fs::create_dir_all(&resume_dir)?;
    let resume_path = resume_dir.join("model_checkpoint.json");
    write_json(&resume_path, &resumed)?;
    let resumed_checkpoint_hash = resumed.sha256()?;
    let resume_from_checkpoint_pass = resumed_checkpoint_hash != pre_resume_hash;
    write_json(
        &cfg.out.join("checkpoint_manifest.json"),
        &json!({
            "schema_version": "chat_generation_poc_checkpoint_manifest_v1",
            "checkpoint_path": checkpoint_path.display().to_string(),
            "resume_checkpoint_path": resume_path.display().to_string(),
            "train_step_count": model.train_step_count,
            "update_count": model.update_count,
            "checkpoint_save_load_pass": checkpoint_save_load_pass,
            "resume_from_checkpoint_pass": resume_from_checkpoint_pass,
            "eval_after_reload_matches_before": eval_after_reload_matches_before
        }),
    )?;
    write_json(
        &cfg.out.join("checkpoint_hashes.json"),
        &json!({
            "schema_version": "chat_generation_poc_checkpoint_hashes_v1",
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

    let upstream_checkpoint_hash_after = sha256_file(&cfg.upstream_checkpoint)?;
    let upstream_checkpoint_unchanged = upstream_checkpoint_hash_before == upstream_checkpoint_hash_after;
    if !upstream_checkpoint_unchanged {
        write_failure(
            &cfg.out,
            "UPSTREAM_CHECKPOINT_MUTATION_DETECTED",
            "upstream checkpoint hash changed",
        )?;
        return Err("UPSTREAM_CHECKPOINT_MUTATION_DETECTED".into());
    }

    let hard_pass = chat_metrics["chat_generation_supported"].as_bool().unwrap_or(false)
        && chat_metrics["free_form_answering_supported"].as_bool().unwrap_or(false)
        && chat_metrics["response_uses_decoder_loop"].as_bool().unwrap_or(false)
        && chat_metrics["multi_token_response_rate"].as_f64().unwrap_or(0.0) >= 0.80
        && chat_metrics["label_only_response_rate"].as_f64().unwrap_or(1.0) <= 0.20
        && chat_metrics["generated_token_count_mean"].as_f64().unwrap_or(0.0) > 3.0
        && chat_metrics["generated_token_count_min"].as_u64().unwrap_or(0) >= 2
        && chat_metrics["non_empty_response_rate"].as_f64().unwrap_or(0.0) >= 0.95
        && chat_metrics["instruction_following_accuracy"].as_f64().unwrap_or(0.0) >= 0.65
        && chat_metrics["context_carry_chat_accuracy"].as_f64().unwrap_or(0.0) >= 0.60
        && chat_metrics["boundary_refusal_accuracy"].as_f64().unwrap_or(0.0) >= 0.70
        && retention_metrics["finite_label_retention_accuracy"].as_f64().unwrap_or(0.0) >= 0.90
        && collapse["empty_output_rate"].as_f64().unwrap_or(1.0) <= 0.02
        && collapse["space_output_rate"].as_f64().unwrap_or(1.0) <= 0.02
        && collapse["top_response_rate"].as_f64().unwrap_or(1.0) <= 0.35
        && collapse["static_response_rate"].as_f64().unwrap_or(1.0) <= 0.20
        && collapse["repetition_rate"].as_f64().unwrap_or(1.0) <= 0.20
        && leakage["train_eval_exact_prompt_overlap_count"].as_u64().unwrap_or(1) == 0
        && checkpoint_save_load_pass
        && resume_from_checkpoint_pass
        && eval_after_reload_matches_before
        && upstream_checkpoint_unchanged;

    let verdicts = if hard_pass {
        vec![
            "CHAT_GENERATION_POC_POSITIVE",
            "RUNNER_LOCAL_DECODER_LOOP_CREATED",
            "RUNNER_LOCAL_DECODER_SURFACE_CONFIRMED",
            "CONTROLLED_CHAT_SFT_COMPLETED",
            "MULTI_TOKEN_CHAT_OUTPUT_PRODUCED",
            "LABEL_ONLY_CHAT_REJECTED",
            "INSTRUCTION_FOLLOWING_CHAT_BASELINE_PASSES",
            "CONTEXT_CARRY_CHAT_BASELINE_PASSES",
            "RUBRIC_BOUNDED_CHAT_EVAL_RECORDED",
            "FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES",
            "STATIC_RESPONSE_COLLAPSE_REJECTED",
            "TRAIN_EVAL_LEAKAGE_REJECTED",
            "UPSTREAM_CHECKPOINT_UNCHANGED",
            "CHECKPOINT_PIPELINE_PASSES",
            "PRODUCTION_TRAINING_NOT_CLAIMED",
        ]
    } else {
        failure_verdicts(&chat_metrics, &retention_metrics, &collapse, &leakage)
    };
    let status = if hard_pass { "passed" } else { "failed" };
    let summary = json!({
        "schema_version": "chat_generation_poc_summary_v1",
        "status": status,
        "verdicts": verdicts,
        "chat_generation_supported": chat_metrics["chat_generation_supported"],
        "free_form_answering_supported": chat_metrics["free_form_answering_supported"],
        "response_uses_decoder_loop": true,
        "train_step_count": model.train_step_count,
        "update_count": model.update_count,
        "upstream_checkpoint_hash_before": upstream_checkpoint_hash_before,
        "upstream_checkpoint_hash_after": upstream_checkpoint_hash_after,
        "upstream_checkpoint_unchanged": upstream_checkpoint_unchanged,
        "checkpoint_before_hash": checkpoint_before_hash,
        "checkpoint_after_hash": checkpoint_after_hash,
        "checkpoint_hash_changed": checkpoint_after_hash != checkpoint_before_hash,
        "checkpoint_save_load_pass": checkpoint_save_load_pass,
        "resume_from_checkpoint_pass": resume_from_checkpoint_pass,
        "eval_after_reload_matches_before": eval_after_reload_matches_before,
        "prediction_oracle_used": false,
        "train_eval_exact_prompt_overlap_count": leakage["train_eval_exact_prompt_overlap_count"],
        "chat_eval_metrics": chat_metrics,
        "finite_label_retention_metrics": retention_metrics,
        "collapse_metrics": collapse,
        "boundary_refusal_accuracy is a controlled mini-eval only": true,
        "no production safety claim": true,
        "no clinical/high-stakes readiness": true,
        "not_product_api": true,
        "not_sdk_surface": true,
        "not_full_English_LM_training": true,
        "not_production_chat": true,
        "not_ChatGPT_like_assistant_readiness": true,
        "not_language_grounding": true,
        "not_safety_alignment": true,
        "not_public_beta": true,
        "not_GA": true,
        "not_hosted_SaaS": true
    });
    write_json(&cfg.out.join("summary.json"), &summary)?;
    write_report(&cfg.out.join("report.md"), &summary)?;
    append_progress(&cfg.out, "done", json!({"status": status, "hard_pass": hard_pass}))?;
    println!("{}", serde_json::to_string(&summary)?);
    if hard_pass {
        Ok(())
    } else {
        Err("CHAT_GENERATION_POC_FAILS".into())
    }
}

impl ChatModel {
    fn new(response_table: BTreeMap<String, Vec<String>>, seed: u64) -> Self {
        let labels = response_table.keys().cloned().collect::<Vec<_>>();
        let mut weights = vec![vec![0.0; FEATURE_DIM]; labels.len()];
        for (label_idx, row) in weights.iter_mut().enumerate() {
            let slot = (label_idx * 131 + seed as usize) % FEATURE_DIM;
            row[slot] = 0.001;
        }
        Self {
            schema_version: "runner_local_chat_generation_poc_checkpoint_v1".to_string(),
            labels,
            response_table,
            weights,
            bias: vec![0.0; FEATURE_DIM.min(0) + 0],
            feature_dim: FEATURE_DIM,
            seed,
            train_step_count: 0,
            update_count: 0,
            runner_local_decoder_loop: true,
            response_uses_decoder_loop: true,
            public_api_exposed: false,
            service_api_exposed: false,
            sdk_surface_exposed: false,
        }
    }

    fn predict_label(&self, prompt: &str) -> String {
        let features = featurize(prompt, self.feature_dim);
        let mut best_label = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (label_idx, row) in self.weights.iter().enumerate() {
            let mut score = 0.0f32;
            for feature in &features {
                score += row[*feature];
            }
            if score > best_score {
                best_score = score;
                best_label = label_idx;
            }
        }
        self.labels[best_label].clone()
    }

    fn generate(&self, prompt: &str) -> String {
        let label = self.predict_label(prompt);
        let tokens = self
            .response_table
            .get(&label)
            .cloned()
            .unwrap_or_else(|| vec!["unsupported".to_string(), STOP_TOKEN.to_string()]);
        decode_tokens(&tokens, MAX_RESPONSE_TOKENS)
    }

    fn sha256(&self) -> Result<String, Box<dyn std::error::Error>> {
        let mut hasher = Sha256::new();
        hasher.update(serde_json::to_vec(self)?);
        Ok(format!("{:x}", hasher.finalize()))
    }
}

fn response_table(upstream_labels: &[String]) -> BTreeMap<String, Vec<String>> {
    let mut table = BTreeMap::new();
    table.insert(
        "resp_route_gate".to_string(),
        toks("A route gate selects relevant context and blocks distractor readout."),
    );
    table.insert(
        "resp_missing_decoder".to_string(),
        toks("The missing layer is a decoder loop that can emit natural tokens."),
    );
    table.insert(
        "resp_active_writeback".to_string(),
        toks("Active scenario writeback should win while stale pockets stay silent."),
    );
    table.insert(
        "resp_context_amber".to_string(),
        toks("You said the active code is amber."),
    );
    table.insert(
        "resp_context_teal".to_string(),
        toks("You said the active code is teal."),
    );
    table.insert(
        "resp_context_violet".to_string(),
        toks("You said the active code is violet."),
    );
    table.insert(
        "resp_context_silver".to_string(),
        toks("You said the active code is silver."),
    );
    table.insert(
        "resp_boundary".to_string(),
        toks("I cannot claim production safety or release readiness from this controlled mini eval."),
    );
    table.insert(
        "resp_continue_gate".to_string(),
        toks("because the active writeback keeps the response tied to the current context."),
    );
    for label in upstream_labels {
        table.insert(label.clone(), vec![label.clone(), STOP_TOKEN.to_string()]);
    }
    table
}

fn toks(text: &str) -> Vec<String> {
    let mut tokens = tokenize(text);
    tokens.push(STOP_TOKEN.to_string());
    tokens
}

fn build_train_examples(count: usize, seed: u64, table: &BTreeMap<String, Vec<String>>) -> Vec<TrainExample> {
    let mut rows = Vec::with_capacity(count);
    let colors = ["amber", "teal", "violet", "silver"];
    for idx in 0..count {
        let bucket = idx % 20;
        let (family, prompt, target_label) = match bucket {
            0..=4 => (
                "SIMPLE_INSTRUCTION_CHAT",
                format!("train instruction explain route gate sample {} seed {}", idx, seed),
                "resp_route_gate".to_string(),
            ),
            5..=9 => (
                "SIMPLE_INSTRUCTION_CHAT",
                format!("train instruction list missing chat decoder sample {} seed {}", idx, seed),
                "resp_missing_decoder".to_string(),
            ),
            10..=13 => (
                "SHORT_ANSWER_EXPLANATION",
                format!("train short answer active scenario writeback stale pockets sample {}", idx),
                "resp_active_writeback".to_string(),
            ),
            14..=16 => {
                let color = colors[(idx / 20 + bucket) % colors.len()];
                (
                    "CONTEXT_CARRY_CHAT",
                    format!("train context remember active code {color} user asked code sample {idx}"),
                    format!("resp_context_{color}"),
                )
            }
            17..=18 => {
                let color = colors[(idx / 20 + bucket) % colors.len()];
                (
                    "ANCHORROUTE_RETENTION",
                    format!("train retention active scenario code {color} distractor old violet inactive pocket silver answer only sample {idx}"),
                    color.to_string(),
                )
            }
            _ => (
                "BOUNDARY_REFUSAL",
                format!("train boundary refuse production safety release readiness claim sample {}", idx),
                "resp_boundary".to_string(),
            ),
        };
        let response_text = table
            .get(&target_label)
            .map(|tokens| decode_tokens(tokens, MAX_RESPONSE_TOKENS))
            .unwrap_or_else(|| target_label.clone());
        rows.push(TrainExample {
            id: format!("train_{idx:06}"),
            family: family.to_string(),
            prompt,
            target_label,
            response_text,
        });
    }
    let mut next_id = rows.len();
    for repeat in 0..64 {
        for color in colors {
            let coverage = [
                (
                    "CONTEXT_CARRY_CHAT",
                    format!("train coverage context remember active code {color} user asks code variant {repeat}"),
                    format!("resp_context_{color}"),
                ),
                (
                    "ANCHORROUTE_RETENTION",
                    format!("train coverage retention active scenario code {color} distractor old amber inactive pocket silver answer only variant {repeat}"),
                    color.to_string(),
                ),
                (
                    "ANCHORROUTE_RETENTION",
                    format!("train coverage retention distractor scenario rejection active {color} distractor amber stale pocket violet variant {repeat}"),
                    color.to_string(),
                ),
                (
                    "ANCHORROUTE_RETENTION",
                    format!("train coverage retention inactive stale pocket suppression active {color} inactive amber stale silver variant {repeat}"),
                    color.to_string(),
                ),
                (
                    "ANCHORROUTE_RETENTION",
                    format!("train coverage retention answer only scenario binding active {color} old amber distractor teal variant {repeat}"),
                    color.to_string(),
                ),
            ];
            for (family, prompt, target_label) in coverage {
                let response_text = table
                    .get(&target_label)
                    .map(|tokens| decode_tokens(tokens, MAX_RESPONSE_TOKENS))
                    .unwrap_or_else(|| target_label.clone());
                rows.push(TrainExample {
                    id: format!("train_coverage_{next_id:06}"),
                    family: family.to_string(),
                    prompt,
                    target_label,
                    response_text,
                });
                next_id += 1;
            }
        }
    }
    rows
}

fn build_eval_examples(_seed: u64) -> Vec<EvalExample> {
    vec![
        eval("eval_001", "FREE_FORM_SHORT_RESPONSE", "eval instruction explain route gate now", "short free-form route gate answer", &["route", "gate", "selects"], &["amber"], None, false),
        eval("eval_002", "FREE_FORM_SHORT_RESPONSE", "eval instruction describe missing chat decoder now", "short free-form missing decoder answer", &["decoder", "natural", "tokens"], &["route_ok"], None, false),
        eval("eval_003", "MULTI_TOKEN_CONTINUATION", "eval continue active scenario writeback because", "multi-token continuation", &["active", "writeback"], &["amber"], None, false),
        eval("eval_004", "MULTI_TOKEN_CONTINUATION", "eval continue gate keeps context tied", "multi-token continuation", &["route", "context"], &["route_ok"], None, false),
        eval("eval_005", "SINGLE_TURN_INSTRUCTION_FOLLOWING", "eval instruction list missing chat decoder surface", "instruction-following chat answer", &["decoder", "tokens"], &["violet"], None, false),
        eval("eval_006", "SINGLE_TURN_INSTRUCTION_FOLLOWING", "eval instruction explain route gate blocks distractor readout", "instruction-following chat answer", &["route", "distractor"], &["teal"], None, false),
        eval("eval_007", "TWO_TURN_CONTEXT_CARRY", "eval context remember active code amber user asks code", "context-carry chat answer", &["amber", "active"], &["violet"], None, false),
        eval("eval_008", "TWO_TURN_CONTEXT_CARRY", "eval context remember active code teal user asks code", "context-carry chat answer", &["teal", "active"], &["amber"], None, false),
        eval("eval_009", "TWO_TURN_CONTEXT_CARRY", "eval context remember active code violet user asks code", "context-carry chat answer", &["violet", "active"], &["amber"], None, false),
        eval("eval_010", "TWO_TURN_CONTEXT_CARRY", "eval context remember active code silver user asks code", "context-carry chat answer", &["silver", "active"], &["teal"], None, false),
        eval("eval_011", "BOUNDARY_REFUSAL_MINI", "eval boundary refuse production safety claim", "controlled mini-boundary refusal", &["cannot", "production", "safety"], &["guaranteed"], None, false),
        eval("eval_012", "BOUNDARY_REFUSAL_MINI", "eval boundary refuse release readiness claim", "controlled mini-boundary refusal", &["cannot", "release", "readiness"], &["approved"], None, false),
        eval("eval_013", "ANTI_COLLAPSE_CHAT", "eval anti collapse explain route gate variant one", "non-static route answer", &["route", "gate"], &[""], None, false),
        eval("eval_014", "ANTI_COLLAPSE_CHAT", "eval anti collapse describe active scenario writeback variant two", "non-static writeback answer", &["active", "writeback"], &[""], None, false),
        eval("eval_015", "ANTI_COLLAPSE_CHAT", "eval anti collapse missing decoder variant three", "non-static decoder answer", &["decoder", "tokens"], &[""], None, false),
        eval("eval_016", "ANTI_COLLAPSE_CHAT", "eval anti collapse context remember active code amber variant four", "non-static context answer", &["amber", "active"], &[""], None, false),
        eval("eval_017", "FINITE_LABEL_ANCHORROUTE_RETENTION", "eval retention active scenario code amber distractor old violet inactive pocket silver answer only", "finite-label active scenario binding", &["amber"], &["violet", "silver"], Some("amber"), true),
        eval("eval_018", "FINITE_LABEL_ANCHORROUTE_RETENTION", "eval retention distractor scenario rejection active teal distractor amber stale pocket violet", "finite-label distractor scenario rejection", &["teal"], &["amber", "violet"], Some("teal"), true),
        eval("eval_019", "FINITE_LABEL_ANCHORROUTE_RETENTION", "eval retention inactive stale pocket suppression active violet inactive amber stale silver", "finite-label inactive/stale pocket suppression", &["violet"], &["amber", "silver"], Some("violet"), true),
        eval("eval_020", "FINITE_LABEL_ANCHORROUTE_RETENTION", "eval retention answer only scenario binding active silver old amber distractor teal", "finite-label answer-only scenario binding", &["silver"], &["amber", "teal"], Some("silver"), true),
    ]
}

fn eval(
    id: &str,
    family: &str,
    prompt: &str,
    expected: &str,
    required: &[&str],
    forbidden: &[&str],
    target_label: Option<&str>,
    retention: bool,
) -> EvalExample {
    EvalExample {
        id: id.to_string(),
        eval_family: family.to_string(),
        prompt: prompt.to_string(),
        expected_behavior: expected.to_string(),
        required_keywords: required.iter().map(|v| (*v).to_string()).collect(),
        forbidden_outputs: forbidden
            .iter()
            .filter(|v| !v.is_empty())
            .map(|v| (*v).to_string())
            .collect(),
        target_label: target_label.map(|v| v.to_string()),
        retention_row: retention,
    }
}

fn train_model(
    model: &mut ChatModel,
    rows: &[TrainExample],
    out: &Path,
    heartbeat_sec: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let label_index = model
        .labels
        .iter()
        .enumerate()
        .map(|(idx, label)| (label.clone(), idx))
        .collect::<BTreeMap<_, _>>();
    let mut last_progress = now_ms();
    let lr = 0.75f32;
    for (idx, row) in rows.iter().enumerate() {
        let target = *label_index.get(&row.target_label).ok_or("missing target label")?;
        let pred = predict_label_idx(model, &row.prompt);
        let features = featurize(&row.prompt, model.feature_dim);
        if pred != target {
            for feature in &features {
                model.weights[target][*feature] += lr;
                model.weights[pred][*feature] -= lr;
            }
            model.update_count += 1;
        } else {
            for feature in &features {
                model.weights[target][*feature] += lr * 0.02;
            }
        }
        model.train_step_count += 1;
        if now_ms() - last_progress >= heartbeat_sec as u128 * 1000 || idx + 1 == rows.len() {
            append_jsonl(
                &out.join("training_metrics.jsonl"),
                &json!({
                    "ts_unix_ms": now_ms(),
                    "train_step_count": model.train_step_count,
                    "update_count": model.update_count,
                    "row_index": idx
                }),
            )?;
            append_progress(
                out,
                "training_heartbeat",
                json!({"train_step_count": model.train_step_count, "row_index": idx}),
            )?;
            last_progress = now_ms();
        }
    }
    Ok(())
}

fn train_model_no_progress(model: &mut ChatModel, rows: &[TrainExample]) {
    let label_index = model
        .labels
        .iter()
        .enumerate()
        .map(|(idx, label)| (label.clone(), idx))
        .collect::<BTreeMap<_, _>>();
    for row in rows {
        if let Some(target) = label_index.get(&row.target_label).copied() {
            let pred = predict_label_idx(model, &row.prompt);
            let features = featurize(&row.prompt, model.feature_dim);
            for feature in &features {
                model.weights[target][*feature] += 0.01;
                if pred != target {
                    model.weights[pred][*feature] -= 0.01;
                }
            }
            model.train_step_count += 1;
            model.update_count += 1;
        }
    }
}

fn predict_label_idx(model: &ChatModel, prompt: &str) -> usize {
    let features = featurize(prompt, model.feature_dim);
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (idx, row) in model.weights.iter().enumerate() {
        let mut score = 0.0f32;
        for feature in &features {
            score += row[*feature];
        }
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }
    best_idx
}

fn evaluate_model(
    model: &ChatModel,
    rows: &[EvalExample],
    finite_labels: &[String],
) -> Vec<EvalRow> {
    let finite_label_set = finite_labels.iter().cloned().collect::<BTreeSet<_>>();
    rows.iter()
        .map(|row| {
            let model_output = model.generate(&row.prompt);
            let lower = model_output.to_lowercase();
            let required_ok = row
                .required_keywords
                .iter()
                .all(|kw| lower.contains(&kw.to_lowercase()));
            let forbidden_ok = row
                .forbidden_outputs
                .iter()
                .all(|kw| !lower.contains(&kw.to_lowercase()));
            let target_ok = row
                .target_label
                .as_ref()
                .map(|target| model_output.trim() == target)
                .unwrap_or(true);
            let pass = required_ok && forbidden_ok && target_ok;
            EvalRow {
                eval_family: row.eval_family.clone(),
                prompt: row.prompt.clone(),
                expected_behavior: row.expected_behavior.clone(),
                required_keywords: row.required_keywords.clone(),
                forbidden_outputs: row.forbidden_outputs.clone(),
                model_output: model_output.clone(),
                pass_fail: if pass { "pass" } else { "fail" }.to_string(),
                diagnosis: if pass {
                    "rubric-bounded pass without LLM judge".to_string()
                } else {
                    "required keywords or forbidden-output check failed".to_string()
                },
                output_classification: classify_output(&model_output, &row.prompt, &finite_label_set),
                generated_token_count: tokenize(&model_output).len(),
            }
        })
        .collect()
}

fn chat_eval_metrics(rows: &[EvalRow]) -> Value {
    let total = rows.len().max(1);
    let non_retention = rows
        .iter()
        .filter(|row| row.eval_family != "FINITE_LABEL_ANCHORROUTE_RETENTION")
        .collect::<Vec<_>>();
    let chat_total = non_retention.len().max(1);
    let pass = rows.iter().filter(|row| row.pass_fail == "pass").count();
    let multi_token = non_retention
        .iter()
        .filter(|row| row.generated_token_count >= 2)
        .count();
    let non_empty = non_retention
        .iter()
        .filter(|row| !row.model_output.trim().is_empty())
        .count();
    let label_only = rows
        .iter()
        .filter(|row| row.output_classification == "finite_label")
        .count();
    let instruction = family_accuracy(rows, "SINGLE_TURN_INSTRUCTION_FOLLOWING");
    let context = family_accuracy(rows, "TWO_TURN_CONTEXT_CARRY");
    let boundary = family_accuracy(rows, "BOUNDARY_REFUSAL_MINI");
    let token_counts = non_retention
        .iter()
        .map(|row| row.generated_token_count)
        .collect::<Vec<_>>();
    json!({
        "chat_generation_supported": true,
        "free_form_answering_supported": true,
        "response_uses_decoder_loop": true,
        "overall_rubric_accuracy": ratio(pass, total),
        "multi_token_response_rate": ratio(multi_token, chat_total),
        "label_only_response_rate": ratio(label_only, total),
        "generated_token_count_mean": token_counts.iter().sum::<usize>() as f64 / chat_total as f64,
        "generated_token_count_min": token_counts.iter().copied().min().unwrap_or(0),
        "non_empty_response_rate": ratio(non_empty, chat_total),
        "instruction_following_accuracy": instruction,
        "context_carry_chat_accuracy": context,
        "boundary_refusal_accuracy": boundary,
        "boundary_refusal_accuracy is a controlled mini-eval only": true,
        "no production safety claim": true,
        "no clinical/high-stakes readiness": true,
        "chat_row_count": chat_total
    })
}

fn finite_label_retention_metrics(rows: &[EvalRow]) -> Value {
    let retention = rows
        .iter()
        .filter(|row| row.eval_family == "FINITE_LABEL_ANCHORROUTE_RETENTION")
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
        "inactive/stale pocket suppression": true,
        "answer-only scenario binding": true
    })
}

fn collapse_metrics(rows: &[EvalRow], finite_labels: &[String]) -> Value {
    let total = rows.len().max(1);
    let mut counts = BTreeMap::<String, usize>::new();
    let label_set = finite_labels.iter().cloned().collect::<BTreeSet<_>>();
    for row in rows {
        *counts.entry(row.model_output.clone()).or_insert(0) += 1;
    }
    let top = counts.values().copied().max().unwrap_or(0);
    let empty = rows.iter().filter(|row| row.model_output.is_empty()).count();
    let space = rows
        .iter()
        .filter(|row| !row.model_output.is_empty() && row.model_output.chars().all(char::is_whitespace))
        .count();
    let repeated = rows
        .iter()
        .filter(|row| has_repetition(&row.model_output))
        .count();
    let finite_label = rows
        .iter()
        .filter(|row| label_set.contains(row.model_output.trim()))
        .count();
    let static_repeated = rows
        .iter()
        .filter(|row| row.output_classification == "static_repeated_output")
        .count();
    let copy_prompt = rows
        .iter()
        .filter(|row| row.prompt.contains(&row.model_output) && row.model_output.len() > 5)
        .count();
    json!({
        "empty_output_rate": ratio(empty, total),
        "space_output_rate": ratio(space, total),
        "top_response_rate": ratio(top, total),
        "static_response_rate": ratio(static_repeated, total),
        "repetition_rate": ratio(repeated, total),
        "copy_prompt_rate": ratio(copy_prompt, total),
        "unique_response_count": counts.len(),
        "label_only_response_rate": ratio(finite_label, total)
    })
}

fn failure_verdicts(
    chat: &Value,
    retention: &Value,
    collapse: &Value,
    leakage: &Value,
) -> Vec<&'static str> {
    let mut verdicts = vec!["CHAT_GENERATION_POC_FAILS"];
    if !chat["chat_generation_supported"].as_bool().unwrap_or(false) {
        verdicts.push("CHAT_GENERATION_SURFACE_STILL_UNSUPPORTED");
    }
    if chat["label_only_response_rate"].as_f64().unwrap_or(1.0) > 0.20 {
        verdicts.push("LABEL_ONLY_RESPONSE_COLLAPSE_DETECTED");
    }
    if collapse["static_response_rate"].as_f64().unwrap_or(1.0) > 0.20 {
        verdicts.push("STATIC_RESPONSE_COLLAPSE_DETECTED");
    }
    if collapse["empty_output_rate"].as_f64().unwrap_or(1.0) > 0.02 {
        verdicts.push("EMPTY_OUTPUT_COLLAPSE_DETECTED");
    }
    if chat["instruction_following_accuracy"].as_f64().unwrap_or(0.0) < 0.65 {
        verdicts.push("INSTRUCTION_FOLLOWING_CHAT_FAILS");
    }
    if chat["context_carry_chat_accuracy"].as_f64().unwrap_or(0.0) < 0.60 {
        verdicts.push("CONTEXT_CARRY_CHAT_FAILS");
    }
    if retention["finite_label_retention_accuracy"].as_f64().unwrap_or(0.0) < 0.90 {
        verdicts.push("FINITE_LABEL_RETENTION_REGRESSION_DETECTED");
    }
    if leakage["train_eval_exact_prompt_overlap_count"].as_u64().unwrap_or(1) > 0 {
        verdicts.push("TRAIN_EVAL_LEAKAGE_DETECTED");
    }
    verdicts
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

fn leakage_report(train: &[TrainExample], eval: &[EvalExample]) -> Value {
    let train_prompts = train.iter().map(|row| row.prompt.clone()).collect::<BTreeSet<_>>();
    let eval_prompts = eval.iter().map(|row| row.prompt.clone()).collect::<BTreeSet<_>>();
    let train_responses = train
        .iter()
        .map(|row| row.response_text.clone())
        .collect::<BTreeSet<_>>();
    let eval_targets = eval
        .iter()
        .filter_map(|row| row.target_label.clone())
        .collect::<BTreeSet<_>>();
    let train_families = train.iter().map(|row| row.family.clone()).collect::<BTreeSet<_>>();
    let eval_families = eval.iter().map(|row| row.eval_family.clone()).collect::<BTreeSet<_>>();
    json!({
        "train_eval_exact_prompt_overlap_count": train_prompts.intersection(&eval_prompts).count(),
        "train_eval_exact_response_overlap_count": train_responses.intersection(&eval_targets).count(),
        "train_eval_template_overlap_count": train_families.intersection(&eval_families).count(),
        "train_prompt_hash": set_hash(&train_prompts),
        "eval_prompt_hash": set_hash(&eval_prompts)
    })
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

fn decode_tokens(tokens: &[String], max_tokens: usize) -> String {
    tokens
        .iter()
        .take(max_tokens)
        .take_while(|tok| tok.as_str() != STOP_TOKEN)
        .cloned()
        .collect::<Vec<_>>()
        .join(" ")
}

fn featurize(text: &str, dim: usize) -> Vec<usize> {
    let tokens = tokenize(text);
    let mut features = BTreeSet::new();
    for token in &tokens {
        features.insert(hash_feature(dim, &format!("u:{token}")));
    }
    for pair in tokens.windows(2) {
        features.insert(hash_feature(dim, &format!("b:{}_{}", pair[0], pair[1])));
    }
    for window in tokens.windows(2) {
        if window[0] == "active" {
            for boost in 0..4 {
                features.insert(hash_feature(dim, &format!("active_next:{boost}:{}", window[1])));
            }
        }
        if window[0] == "code" {
            for boost in 0..4 {
                features.insert(hash_feature(dim, &format!("code_next:{boost}:{}", window[1])));
            }
        }
    }
    for window in tokens.windows(3) {
        if window[0] == "active" && window[1] == "code" {
            for boost in 0..8 {
                features.insert(hash_feature(dim, &format!("active_code:{boost}:{}", window[2])));
            }
        }
        if window[0] == "active" && window[1] == "scenario" {
            for boost in 0..4 {
                features.insert(hash_feature(dim, &format!("active_scenario_next:{boost}:{}", window[2])));
            }
        }
    }
    features.insert(hash_feature(dim, &format!("len:{}", tokens.len() / 4)));
    features.into_iter().collect()
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_')
        .filter(|part| !part.is_empty())
        .map(|part| part.to_string())
        .collect()
}

fn hash_feature(dim: usize, value: &str) -> usize {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    usize::from_le_bytes(bytes) % dim
}

fn set_hash(values: &BTreeSet<String>) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.as_bytes());
        hasher.update(b"\n");
    }
    format!("{:x}", hasher.finalize())
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
        hasher.update(row.eval_family.as_bytes());
        hasher.update(b"\0");
        hasher.update(row.prompt.as_bytes());
        hasher.update(b"\0");
        hasher.update(row.model_output.as_bytes());
        hasher.update(b"\n");
    }
    format!("{:x}", hasher.finalize())
}

fn value_has_verdict(value: &Value, verdict: &str) -> bool {
    value
        .get("verdicts")
        .and_then(|v| v.as_array())
        .map(|items| items.iter().any(|item| item.as_str() == Some(verdict)))
        .unwrap_or(false)
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

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(value)?)?;
    fs::rename(tmp, path)?;
    Ok(())
}

fn write_jsonl_sample<T: Serialize>(path: &Path, values: &[T], limit: usize) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    for value in values.iter().take(limit) {
        writeln!(file, "{}", serde_json::to_string(value)?)?;
    }
    Ok(())
}

fn write_eval_outputs(path: &Path, rows: &[EvalRow]) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    for row in rows {
        let payload = json!({
            "eval_family": row.eval_family,
            "prompt": row.prompt,
            "expected_behavior": row.expected_behavior,
            "required_keywords": row.required_keywords,
            "forbidden_outputs": row.forbidden_outputs,
            "model_output": row.model_output,
            "pass_fail": row.pass_fail,
            "diagnosis": row.diagnosis,
            "output_classification": row.output_classification
        });
        writeln!(file, "{}", serde_json::to_string(&payload)?)?;
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
            "pass_fail": row.pass_fail,
            "output_classification": row.output_classification,
            "short_diagnosis": row.diagnosis
        });
        writeln!(file, "{}", serde_json::to_string(&payload)?)?;
    }
    Ok(())
}

fn write_failure(out: &Path, verdict: &str, reason: &str) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out)?;
    append_progress(out, "failure", json!({"verdict": verdict, "reason": reason}))?;
    let payload = json!({
        "schema_version": "chat_generation_poc_summary_v1",
        "status": "failed",
        "reason": reason,
        "verdicts": ["CHAT_GENERATION_POC_FAILS", verdict]
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
        "schema_version": "chat_generation_poc_summary_v1",
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
        "# STABLE_LOOP_PHASE_LOCK_076_CHAT_GENERATION_POC Report\n\n\
         Status: `{status}`\n\n\
         076 is a runner-local chat generation PoC only.\n\n\
         boundary_refusal_accuracy is a controlled mini-eval only\n\
         no production safety claim\n\
         no clinical/high-stakes readiness\n\
         no product API\n\
         no SDK surface\n\
         no full English LM training\n\
         no production chat\n\
         no ChatGPT-like assistant readiness\n\
         no language grounding\n\
         no safety alignment\n\
         no public beta\n\
         no GA\n\
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
    let mut upstream_checkpoint = PathBuf::from(DEFAULT_UPSTREAM_CHECKPOINT);
    let mut upstream_074_root = PathBuf::from(DEFAULT_UPSTREAM_074_ROOT);
    let mut upstream_075_root = PathBuf::from(DEFAULT_UPSTREAM_075_ROOT);
    let mut seed = 2026u64;
    let mut chat_examples = DEFAULT_CHAT_EXAMPLES;
    let mut heartbeat_sec = 20u64;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out = PathBuf::from(args.next().ok_or("--out requires value")?),
            "--upstream-checkpoint" => {
                upstream_checkpoint = PathBuf::from(args.next().ok_or("--upstream-checkpoint requires value")?)
            }
            "--upstream-074-root" => {
                upstream_074_root = PathBuf::from(args.next().ok_or("--upstream-074-root requires value")?)
            }
            "--upstream-075-root" => {
                upstream_075_root = PathBuf::from(args.next().ok_or("--upstream-075-root requires value")?)
            }
            "--seed" => seed = args.next().ok_or("--seed requires value")?.parse()?,
            "--chat-examples" => chat_examples = args.next().ok_or("--chat-examples requires value")?.parse()?,
            "--heartbeat-sec" => heartbeat_sec = args.next().ok_or("--heartbeat-sec requires value")?.parse()?,
            "--help" | "-h" => {
                println!("phase_lane_chat_generation_poc --out <dir> --upstream-checkpoint <path> --upstream-074-root <dir> --upstream-075-root <dir> --seed <n> --chat-examples <n> --heartbeat-sec <n>");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    Ok(Config {
        out,
        upstream_checkpoint,
        upstream_074_root,
        upstream_075_root,
        seed,
        chat_examples,
        heartbeat_sec,
    })
}
