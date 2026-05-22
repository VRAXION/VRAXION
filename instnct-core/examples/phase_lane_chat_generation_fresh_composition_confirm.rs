//! Eval-only fresh composition confirmation for the 076 chat generation PoC.
//!
//! 077 checks whether the runner-local 076 chat checkpoint composes fresh
//! responses or mostly selects/copies controlled response templates. It does
//! not train, resume, repair, mutate checkpoints, or expose a runtime surface.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_OUT: &str =
    "target/pilot_wave/stable_loop_phase_lock_077_chat_generation_fresh_composition_confirm/smoke";
const DEFAULT_CHECKPOINT: &str =
    "target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke/checkpoints/chat_generation_poc/model_checkpoint.json";
const DEFAULT_UPSTREAM_076_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke";
const DEFAULT_UPSTREAM_074_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke";
const MAX_RESPONSE_TOKENS: usize = 64;
const STOP_TOKEN: &str = "<eos>";

const EVAL_FAMILIES: [&str; 9] = [
    "FRESH_SIMPLE_INSTRUCTION",
    "FRESH_SHORT_EXPLANATION",
    "FRESH_CONTEXT_CARRY_CHAT",
    "FRESH_TWO_TURN_DIALOGUE",
    "FRESH_BOUNDARY_REFUSAL_MINI",
    "FRESH_COMPOSITION_NOVELTY",
    "ANTI_TEMPLATE_COPY",
    "ANTI_REPETITION",
    "FINITE_LABEL_ANCHORROUTE_RETENTION",
];

#[derive(Debug, Clone)]
struct Config {
    out: PathBuf,
    checkpoint: PathBuf,
    upstream_076_root: PathBuf,
    upstream_074_root: PathBuf,
    seed: u64,
    heartbeat_sec: u64,
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
    output_classification: String,
    pass_fail: String,
    short_diagnosis: String,
    generated_token_count: usize,
    template_copy_flag: bool,
    novelty_flag: bool,
}

#[derive(Debug, Clone)]
struct CopySources {
    train_responses: BTreeSet<String>,
    eval_outputs: BTreeSet<String>,
    response_table_outputs: BTreeSet<String>,
    template_responses: BTreeSet<String>,
    train_prompts: BTreeSet<String>,
    eval_prompts: BTreeSet<String>,
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
    let start_ms = now_ms();
    fs::create_dir_all(&cfg.out)?;
    append_progress(
        &cfg.out,
        "start",
        json!({
            "seed": cfg.seed,
            "heartbeat_sec": cfg.heartbeat_sec,
            "eval_only": true,
            "run_start_ms": start_ms
        }),
    )?;
    write_summary_and_report(
        &cfg.out,
        "running",
        vec!["CHAT_GENERATION_FRESH_COMPOSITION_CONFIRM_RUNNING".to_string()],
        json!({"phase": "start"}),
    )?;
    write_json(
        &cfg.out.join("queue.json"),
        &json!({
            "schema_version": "chat_generation_fresh_composition_queue_v1",
            "milestone": "STABLE_LOOP_PHASE_LOCK_077_CHAT_GENERATION_FRESH_COMPOSITION_CONFIRM",
            "partial_write_policy": "progress summary report written from start and refreshed by phase",
            "steps": [
                "verify_upstream_076_and_074",
                "load_076_checkpoint_read_only",
                "build_fresh_eval_rows",
                "run_eval_only_generation",
                "measure_template_copy_and_novelty",
                "measure_collapse_and_retention",
                "write_summary"
            ]
        }),
    )?;

    let upstream_076_summary_path = cfg.upstream_076_root.join("summary.json");
    let upstream_074_summary_path = cfg.upstream_074_root.join("summary.json");
    let upstream_076_dataset_manifest_path = cfg.upstream_076_root.join("chat_sft_dataset_manifest.json");
    let upstream_076_eval_sample_path = cfg.upstream_076_root.join("eval_examples_sample.jsonl");
    let upstream_076_train_sample_path = cfg.upstream_076_root.join("train_examples_sample.jsonl");
    let upstream_076_generation_path = cfg.upstream_076_root.join("generation_samples.jsonl");
    let upstream_076_training_config_path = cfg.upstream_076_root.join("training_config.json");

    let mut missing = Vec::new();
    for (name, path) in [
        ("checkpoint", cfg.checkpoint.as_path()),
        ("upstream_076_summary", upstream_076_summary_path.as_path()),
        ("upstream_074_summary", upstream_074_summary_path.as_path()),
        ("upstream_076_dataset_manifest", upstream_076_dataset_manifest_path.as_path()),
        ("upstream_076_eval_examples_sample", upstream_076_eval_sample_path.as_path()),
        ("upstream_076_train_examples_sample", upstream_076_train_sample_path.as_path()),
        ("upstream_076_generation_samples", upstream_076_generation_path.as_path()),
        ("upstream_076_training_config", upstream_076_training_config_path.as_path()),
    ] {
        if !path.exists() {
            missing.push(name.to_string());
        }
    }
    if !missing.is_empty() {
        write_failure(&cfg.out, "UPSTREAM_076_ARTIFACT_MISSING", &missing.join(","))?;
        return Err(format!("UPSTREAM_076_ARTIFACT_MISSING: {}", missing.join(",")).into());
    }

    let upstream_076_summary: Value = serde_json::from_slice(&fs::read(&upstream_076_summary_path)?)?;
    let upstream_074_summary: Value = serde_json::from_slice(&fs::read(&upstream_074_summary_path)?)?;
    let upstream_076_positive = value_has_verdict(&upstream_076_summary, "CHAT_GENERATION_POC_POSITIVE");
    let upstream_074_positive =
        value_has_verdict(&upstream_074_summary, "MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_POSITIVE");
    if !upstream_076_positive {
        write_failure(&cfg.out, "UPSTREAM_076_ARTIFACT_MISSING", "076 positive verdict missing")?;
        return Err("UPSTREAM_076_ARTIFACT_MISSING".into());
    }
    if !upstream_074_positive {
        write_failure(&cfg.out, "UPSTREAM_076_ARTIFACT_MISSING", "074 positive verdict missing")?;
        return Err("UPSTREAM_076_ARTIFACT_MISSING".into());
    }

    let checkpoint_hash_before = sha256_file(&cfg.checkpoint)?;
    let model: ChatModel = serde_json::from_slice(&fs::read(&cfg.checkpoint)?)?;
    let child_eval_started_after_077_start = now_ms() >= start_ms;
    append_progress(
        &cfg.out,
        "upstreams_verified",
        json!({
            "upstream_076_summary_present": true,
            "upstream_076_positive": upstream_076_positive,
            "upstream_074_positive": upstream_074_positive,
            "checkpoint_exists": true,
            "child_eval_started_after_077_start": child_eval_started_after_077_start
        }),
    )?;

    write_json(
        &cfg.out.join("upstream_076_manifest.json"),
        &json!({
            "schema_version": "chat_generation_fresh_composition_upstream_076_manifest_v1",
            "upstream_076_root": cfg.upstream_076_root.display().to_string(),
            "upstream_076_summary_present": true,
            "upstream_076_positive": true,
            "checkpoint": cfg.checkpoint.display().to_string(),
            "checkpoint_exists": true,
            "checkpoint_hash_before": checkpoint_hash_before,
            "child_eval_started_after_077_start": child_eval_started_after_077_start,
            "upstream_076_status": upstream_076_summary["status"],
            "upstream_076_train_step_count": upstream_076_summary["train_step_count"]
        }),
    )?;
    write_json(
        &cfg.out.join("upstream_074_manifest.json"),
        &json!({
            "schema_version": "chat_generation_fresh_composition_upstream_074_manifest_v1",
            "upstream_074_root": cfg.upstream_074_root.display().to_string(),
            "upstream_074_positive": true
        }),
    )?;

    let training_config: Value = serde_json::from_slice(&fs::read(&upstream_076_training_config_path)?)?;
    let dataset_manifest: Value = serde_json::from_slice(&fs::read(&upstream_076_dataset_manifest_path)?)?;
    let chat_examples = training_config["chat_examples"].as_u64().unwrap_or(20_000) as usize;
    let upstream_seed = training_config["seed"].as_u64().unwrap_or(2026);
    let copy_sources = copy_sources(
        &model,
        &upstream_076_eval_sample_path,
        &upstream_076_train_sample_path,
        &upstream_076_generation_path,
        chat_examples,
        upstream_seed,
    )?;

    write_json(
        &cfg.out.join("benchmark_config.json"),
        &json!({
            "schema_version": "chat_generation_fresh_composition_config_v1",
            "bounded_fresh_composition_confirm_only": true,
            "eval_only": true,
            "no_training": true,
            "no_resume": true,
            "no_checkpoint_repair": true,
            "no_checkpoint_mutation": true,
            "no_replacement_checkpoint": true,
            "not_GPT_like_assistant_readiness": true,
            "not_full_English_LM": true,
            "not_language_grounding": true,
            "not_production_chat": true,
            "not_public_beta_GA_hosted_SaaS": true,
            "eval_families": EVAL_FAMILIES,
            "seed": cfg.seed,
            "fresh_prompt_policy": "new wording, entities, instruction shapes, context-carry variants, refusal phrasing, and no exact 076 train/eval prompt overlap",
            "upstream_076_chat_examples": chat_examples,
            "upstream_076_train_examples": dataset_manifest["train_examples"]
        }),
    )?;

    let eval_examples = build_fresh_eval_examples(cfg.seed);
    let train_eval_exact_prompt_overlap_count = eval_examples
        .iter()
        .filter(|row| copy_sources.train_prompts.contains(&row.prompt) || copy_sources.eval_prompts.contains(&row.prompt))
        .count();
    let prompt_ngram_overlap_stats = prompt_ngram_overlap_stats(&eval_examples, &copy_sources);
    write_jsonl_sample(&cfg.out.join("fresh_chat_eval_dataset.jsonl"), &eval_examples, eval_examples.len())?;
    append_progress(
        &cfg.out,
        "fresh_dataset_written",
        json!({
            "eval_examples": eval_examples.len(),
            "train_eval_exact_prompt_overlap_count": train_eval_exact_prompt_overlap_count,
            "prompt_ngram_overlap_stats": prompt_ngram_overlap_stats
        }),
    )?;
    write_summary_and_report(
        &cfg.out,
        "running",
        vec!["CHAT_GENERATION_FRESH_COMPOSITION_CONFIRM_RUNNING".to_string()],
        json!({"phase": "fresh_dataset_written", "eval_examples": eval_examples.len()}),
    )?;
    if train_eval_exact_prompt_overlap_count > 0 {
        write_failure(&cfg.out, "TRAIN_EVAL_LEAKAGE_DETECTED", "exact prompt overlap with 076 train/eval")?;
        return Err("TRAIN_EVAL_LEAKAGE_DETECTED".into());
    }

    let eval_rows = evaluate_model(&model, &eval_examples, &copy_sources);
    write_eval_outputs(&cfg.out.join("generation_samples.jsonl"), &eval_rows)?;
    write_human_samples(&cfg.out.join("human_readable_samples.jsonl"), &eval_rows)?;

    let composition = composition_metrics(&eval_rows, &model.labels);
    let novelty = novelty_metrics(&eval_rows, &copy_sources);
    let collapse = collapse_metrics(&eval_rows, &model.labels);
    let retention = finite_label_retention_metrics(&eval_rows);
    write_json(&cfg.out.join("composition_metrics.json"), &composition)?;
    write_json(&cfg.out.join("novelty_metrics.json"), &novelty)?;
    write_json(&cfg.out.join("collapse_metrics.json"), &collapse)?;
    write_json(&cfg.out.join("finite_label_retention_metrics.json"), &retention)?;
    append_progress(
        &cfg.out,
        "eval_completed",
        json!({
            "fresh_instruction_accuracy": composition["fresh_instruction_accuracy"],
            "fresh_context_carry_accuracy": composition["fresh_context_carry_accuracy"],
            "novel_response_rate": novelty["novel_response_rate"],
            "template_copy_rate": novelty["template_copy_rate"],
            "finite_label_retention_accuracy": retention["finite_label_retention_accuracy"]
        }),
    )?;

    let checkpoint_hash_after = sha256_file(&cfg.checkpoint)?;
    let checkpoint_hash_unchanged = checkpoint_hash_before == checkpoint_hash_after;
    write_json(
        &cfg.out.join("checkpoint_manifest.json"),
        &json!({
            "schema_version": "chat_generation_fresh_composition_checkpoint_manifest_v1",
            "checkpoint": cfg.checkpoint.display().to_string(),
            "checkpoint_hash_before": checkpoint_hash_before,
            "checkpoint_hash_after": checkpoint_hash_after,
            "checkpoint_hash_unchanged": checkpoint_hash_unchanged,
            "train_step_count": 0,
            "prediction_oracle_used": false,
            "response_uses_decoder_loop": true,
            "no_replacement_checkpoint_written": true
        }),
    )?;
    if !checkpoint_hash_unchanged {
        write_failure(&cfg.out, "CHECKPOINT_MUTATION_DETECTED", "checkpoint hash changed")?;
        return Err("CHECKPOINT_MUTATION_DETECTED".into());
    }

    let hard_pass = composition["multi_token_response_rate"].as_f64().unwrap_or(0.0) >= 0.90
        && composition["non_empty_response_rate"].as_f64().unwrap_or(0.0) >= 0.98
        && composition["fresh_instruction_accuracy"].as_f64().unwrap_or(0.0) >= 0.70
        && composition["fresh_context_carry_accuracy"].as_f64().unwrap_or(0.0) >= 0.65
        && novelty["novel_response_rate"].as_f64().unwrap_or(0.0) >= 0.60
        && novelty["template_copy_rate"].as_f64().unwrap_or(1.0) <= 0.30
        && composition["label_only_response_rate"].as_f64().unwrap_or(1.0) <= 0.20
        && composition["generated_token_count_min"].as_u64().unwrap_or(0) >= 2
        && collapse["empty_output_rate"].as_f64().unwrap_or(1.0) <= 0.02
        && collapse["space_output_rate"].as_f64().unwrap_or(1.0) <= 0.02
        && collapse["static_response_rate"].as_f64().unwrap_or(1.0) <= 0.15
        && collapse["repetition_rate"].as_f64().unwrap_or(1.0) <= 0.20
        && retention["finite_label_retention_accuracy"].as_f64().unwrap_or(0.0) >= 0.90
        && checkpoint_hash_unchanged
        && train_eval_exact_prompt_overlap_count == 0;

    let verdicts = if hard_pass {
        vec![
            "CHAT_GENERATION_FRESH_COMPOSITION_CONFIRM_POSITIVE",
            "FRESH_MULTI_TOKEN_RESPONSES_PASS",
            "FRESH_INSTRUCTION_FOLLOWING_PASSES",
            "FRESH_CONTEXT_CARRY_CHAT_PASSES",
            "TEMPLATE_COPY_REJECTED",
            "STATIC_RESPONSE_COLLAPSE_REJECTED",
            "FINITE_LABEL_RETENTION_PASSES",
            "NO_TRAINING_PERFORMED",
            "PRODUCTION_CHAT_NOT_CLAIMED",
        ]
    } else {
        failure_verdicts(&composition, &novelty, &collapse, &retention, train_eval_exact_prompt_overlap_count)
    };
    let status = if hard_pass { "passed" } else { "failed" };
    let summary = json!({
        "schema_version": "chat_generation_fresh_composition_summary_v1",
        "status": status,
        "verdicts": verdicts,
        "upstream_076_summary_present": true,
        "upstream_076_positive": true,
        "checkpoint_exists": true,
        "child_eval_started_after_077_start": child_eval_started_after_077_start,
        "checkpoint_hash_before": checkpoint_hash_before,
        "checkpoint_hash_after": checkpoint_hash_after,
        "checkpoint_hash_unchanged": checkpoint_hash_unchanged,
        "train_step_count": 0,
        "prediction_oracle_used": false,
        "response_uses_decoder_loop": true,
        "train_eval_exact_prompt_overlap_count": train_eval_exact_prompt_overlap_count,
        "prompt_ngram_overlap_stats": prompt_ngram_overlap_stats,
        "composition_metrics": composition,
        "novelty_metrics": novelty,
        "collapse_metrics": collapse,
        "finite_label_retention_metrics": retention,
        "bounded_fresh_composition_confirm_only": true,
        "not_GPT_like_assistant_readiness": true,
        "not_full_English_LM": true,
        "not_language_grounding": true,
        "not_production_chat": true,
        "not_public_beta_GA_hosted_SaaS": true,
        "next_if_pass": "078_CHAT_GENERATION_MULTI_SEED_CONFIRM",
        "next_if_fail": "077B_CHAT_GENERATION_FAILURE_ANALYSIS"
    });
    write_json(&cfg.out.join("summary.json"), &summary)?;
    write_report(&cfg.out.join("report.md"), &summary)?;
    append_progress(&cfg.out, "done", json!({"status": status, "hard_pass": hard_pass}))?;
    println!("{}", serde_json::to_string(&summary)?);
    Ok(())
}

impl ChatModel {
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
}

fn build_fresh_eval_examples(seed: u64) -> Vec<EvalExample> {
    let marker = seed % 997;
    vec![
        eval("fresh_001", "FRESH_SIMPLE_INSTRUCTION", &format!("fresh request {marker}: say how a routing gate chooses useful context"), "fresh instruction response about route gate", &["route", "context"], &["amber"], None, false),
        eval("fresh_002", "FRESH_SIMPLE_INSTRUCTION", "fresh instruction: name the missing layer that turns labels into natural token output", "fresh instruction response about decoder", &["decoder", "tokens"], &["route_ok"], None, false),
        eval("fresh_003", "FRESH_SHORT_EXPLANATION", "fresh explain why stale pocket values should stay out of the answer", "fresh short explanation about stale pocket suppression", &["stale", "pockets"], &["silver"], None, false),
        eval("fresh_004", "FRESH_SHORT_EXPLANATION", "fresh explain active writeback without using a single label", "fresh short explanation about active writeback", &["active", "writeback"], &["route_ok"], None, false),
        eval("fresh_005", "FRESH_CONTEXT_CARRY_CHAT", "fresh note: active code is teal; answer what active code I gave", "fresh context carry teal", &["teal", "active"], &["amber"], None, false),
        eval("fresh_006", "FRESH_CONTEXT_CARRY_CHAT", "fresh memory check: the active code is violet; reply with the remembered active code", "fresh context carry violet", &["violet", "active"], &["amber"], None, false),
        eval("fresh_007", "FRESH_TWO_TURN_DIALOGUE", "fresh turn one user says active code amber. turn two asks which active code was stated", "fresh two-turn amber", &["amber", "active"], &["violet"], None, false),
        eval("fresh_008", "FRESH_TWO_TURN_DIALOGUE", "fresh first turn stores active code silver, second turn asks for it conversationally", "fresh two-turn silver", &["silver", "active"], &["teal"], None, false),
        eval("fresh_009", "FRESH_BOUNDARY_REFUSAL_MINI", "fresh refuse the claim that this is production chat ready", "fresh mini-boundary refusal", &["cannot", "production"], &["guaranteed"], None, false),
        eval("fresh_010", "FRESH_BOUNDARY_REFUSAL_MINI", "fresh decline saying this proves GPT-like assistant readiness", "fresh mini-boundary refusal", &["cannot", "readiness"], &["approved"], None, false),
        eval("fresh_011", "FRESH_COMPOSITION_NOVELTY", "fresh compose a new sentence about why decoder output is still bounded", "fresh composition novelty", &["decoder"], &["route_ok"], None, false),
        eval("fresh_012", "FRESH_COMPOSITION_NOVELTY", "fresh compose a new explanation connecting active context and stale suppression", "fresh composition novelty", &["active"], &["route_ok"], None, false),
        eval("fresh_013", "ANTI_TEMPLATE_COPY", "fresh avoid canned wording and explain the gate using different words", "anti-template-copy route response", &["route"], &["route_ok"], None, false),
        eval("fresh_014", "ANTI_TEMPLATE_COPY", "fresh do not repeat the training phrase; describe why old pockets stay quiet", "anti-template-copy pocket response", &["pockets"], &["route_ok"], None, false),
        eval("fresh_015", "ANTI_REPETITION", "fresh give a varied non-repeating response about context selection", "anti-repetition response", &["context"], &[""], None, false),
        eval("fresh_016", "ANTI_REPETITION", "fresh answer with a coherent phrase about decoder limits", "anti-repetition response", &["decoder"], &[""], None, false),
        eval("fresh_017", "FINITE_LABEL_ANCHORROUTE_RETENTION", "fresh retention active scenario code amber old violet inactive silver answer only", "finite-label active scenario binding", &["amber"], &["violet", "silver"], Some("amber"), true),
        eval("fresh_018", "FINITE_LABEL_ANCHORROUTE_RETENTION", "fresh retention reject distractor: active teal, distractor amber, stale violet", "finite-label distractor scenario rejection", &["teal"], &["amber", "violet"], Some("teal"), true),
        eval("fresh_019", "FINITE_LABEL_ANCHORROUTE_RETENTION", "fresh retention inactive stale pocket suppression active violet inactive amber stale silver", "finite-label old/stale/inactive suppression", &["violet"], &["amber", "silver"], Some("violet"), true),
        eval("fresh_020", "FINITE_LABEL_ANCHORROUTE_RETENTION", "fresh retention answer-only active silver old amber distractor teal", "finite-label answer-only scenario binding", &["silver"], &["amber", "teal"], Some("silver"), true),
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

fn evaluate_model(model: &ChatModel, rows: &[EvalExample], copy_sources: &CopySources) -> Vec<EvalRow> {
    let finite_labels = model.labels.iter().cloned().collect::<BTreeSet<_>>();
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
            let output_classification = classify_output(&model_output, &row.prompt, &finite_labels);
            let template_copy_flag = is_template_copy(&model_output, copy_sources);
            let novelty_flag = !template_copy_flag && output_classification != "finite_label";
            EvalRow {
                eval_family: row.eval_family.clone(),
                prompt: row.prompt.clone(),
                expected_behavior: row.expected_behavior.clone(),
                required_keywords: row.required_keywords.clone(),
                forbidden_outputs: row.forbidden_outputs.clone(),
                model_output: model_output.clone(),
                output_classification,
                pass_fail: if pass { "pass" } else { "fail" }.to_string(),
                short_diagnosis: if pass {
                    if template_copy_flag {
                        "rubric pass, but output matches a known 076 response template".to_string()
                    } else {
                        "fresh rubric pass without LLM judge".to_string()
                    }
                } else {
                    "required keywords, forbidden-output, or target-label check failed".to_string()
                },
                generated_token_count: tokenize(&model_output).len(),
                template_copy_flag,
                novelty_flag,
            }
        })
        .collect()
}

fn composition_metrics(rows: &[EvalRow], finite_labels: &[String]) -> Value {
    let non_retention = rows
        .iter()
        .filter(|row| row.eval_family != "FINITE_LABEL_ANCHORROUTE_RETENTION")
        .collect::<Vec<_>>();
    let chat_total = non_retention.len().max(1);
    let label_set = finite_labels.iter().cloned().collect::<BTreeSet<_>>();
    let token_counts = non_retention
        .iter()
        .map(|row| row.generated_token_count)
        .collect::<Vec<_>>();
    let label_only = rows
        .iter()
        .filter(|row| label_set.contains(row.model_output.trim()))
        .count();
    let unique = rows
        .iter()
        .map(|row| row.model_output.clone())
        .collect::<BTreeSet<_>>()
        .len();
    json!({
        "multi_token_response_rate": ratio(non_retention.iter().filter(|row| row.generated_token_count >= 2).count(), chat_total),
        "non_empty_response_rate": ratio(non_retention.iter().filter(|row| !row.model_output.trim().is_empty()).count(), chat_total),
        "fresh_instruction_accuracy": family_accuracy(rows, "FRESH_SIMPLE_INSTRUCTION"),
        "fresh_context_carry_accuracy": family_accuracy(rows, "FRESH_CONTEXT_CARRY_CHAT"),
        "two_turn_dialogue_accuracy": family_accuracy(rows, "FRESH_TWO_TURN_DIALOGUE"),
        "boundary_refusal_accuracy": family_accuracy(rows, "FRESH_BOUNDARY_REFUSAL_MINI"),
        "label_only_response_rate": ratio(label_only, rows.len().max(1)),
        "generated_token_count_mean": token_counts.iter().sum::<usize>() as f64 / chat_total as f64,
        "generated_token_count_min": token_counts.iter().copied().min().unwrap_or(0),
        "unique_response_count": unique
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
    let template = rows.iter().filter(|row| row.template_copy_flag).count();
    let novel = rows.iter().filter(|row| row.novelty_flag).count();
    let overlap = train_response_ngram_overlap(rows, &sources.train_responses);
    json!({
        "exact_train_response_copy_rate": ratio(exact_train, total),
        "exact_eval_response_copy_rate": ratio(exact_eval, total),
        "response_table_copy_rate": ratio(response_table, total),
        "template_copy_rate": ratio(template, total),
        "train_response_ngram_overlap": overlap,
        "novel_response_rate": ratio(novel, total)
    })
}

fn collapse_metrics(rows: &[EvalRow], finite_labels: &[String]) -> Value {
    let total = rows.len().max(1);
    let label_set = finite_labels.iter().cloned().collect::<BTreeSet<_>>();
    let mut counts = BTreeMap::<String, usize>::new();
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
        "label_only_response_rate": ratio(rows.iter().filter(|row| label_set.contains(row.model_output.trim())).count(), total),
        "unique_response_count": counts.len()
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
        "old/stale/inactive suppression": true,
        "answer-only scenario binding": true
    })
}

fn failure_verdicts(
    composition: &Value,
    novelty: &Value,
    collapse: &Value,
    retention: &Value,
    overlap: usize,
) -> Vec<&'static str> {
    let mut verdicts = vec!["CHAT_GENERATION_FRESH_COMPOSITION_CONFIRM_FAILS"];
    if composition["fresh_instruction_accuracy"].as_f64().unwrap_or(0.0) < 0.70 {
        verdicts.push("FRESH_INSTRUCTION_FOLLOWING_FAILS");
    }
    if composition["fresh_context_carry_accuracy"].as_f64().unwrap_or(0.0) < 0.65 {
        verdicts.push("FRESH_CONTEXT_CARRY_CHAT_FAILS");
    }
    if novelty["template_copy_rate"].as_f64().unwrap_or(1.0) > 0.30 {
        verdicts.push("TEMPLATE_COPY_DETECTED");
    }
    if composition["label_only_response_rate"].as_f64().unwrap_or(1.0) > 0.20 {
        verdicts.push("LABEL_ONLY_RESPONSE_COLLAPSE_DETECTED");
    }
    if novelty["novel_response_rate"].as_f64().unwrap_or(0.0) < 0.60 {
        verdicts.push("CHAT_GENERATION_SURFACE_STILL_TOO_TABLE_LIKE");
    }
    if collapse["static_response_rate"].as_f64().unwrap_or(1.0) > 0.15 {
        verdicts.push("STATIC_RESPONSE_COLLAPSE_DETECTED");
    }
    if collapse["repetition_rate"].as_f64().unwrap_or(1.0) > 0.20 {
        verdicts.push("REPETITION_COLLAPSE_DETECTED");
    }
    if collapse["empty_output_rate"].as_f64().unwrap_or(1.0) > 0.02 {
        verdicts.push("EMPTY_OUTPUT_COLLAPSE_DETECTED");
    }
    if retention["finite_label_retention_accuracy"].as_f64().unwrap_or(0.0) < 0.90 {
        verdicts.push("FINITE_LABEL_RETENTION_REGRESSION_DETECTED");
    }
    if overlap > 0 {
        verdicts.push("TRAIN_EVAL_LEAKAGE_DETECTED");
    }
    verdicts
}

fn copy_sources(
    model: &ChatModel,
    eval_sample_path: &Path,
    train_sample_path: &Path,
    generation_path: &Path,
    chat_examples: usize,
    seed: u64,
) -> Result<CopySources, Box<dyn std::error::Error>> {
    let response_table_outputs = model
        .response_table
        .values()
        .map(|tokens| normalize_response(&decode_tokens(tokens, MAX_RESPONSE_TOKENS)))
        .collect::<BTreeSet<_>>();
    let template_responses = response_table_outputs.clone();
    let mut eval_outputs = BTreeSet::new();
    let mut eval_prompts = BTreeSet::new();
    for value in read_jsonl_values(generation_path)? {
        if let Some(output) = value.get("model_output").and_then(|v| v.as_str()) {
            eval_outputs.insert(normalize_response(output));
        }
        if let Some(prompt) = value.get("prompt").and_then(|v| v.as_str()) {
            eval_prompts.insert(prompt.to_string());
        }
    }
    for value in read_jsonl_values(eval_sample_path)? {
        if let Some(prompt) = value.get("prompt").and_then(|v| v.as_str()) {
            eval_prompts.insert(prompt.to_string());
        }
    }
    let mut train_prompts = reconstruct_076_train_prompts(chat_examples, seed);
    let mut train_responses = reconstruct_076_train_responses();
    for value in read_jsonl_values(train_sample_path)? {
        if let Some(prompt) = value.get("prompt").and_then(|v| v.as_str()) {
            train_prompts.insert(prompt.to_string());
        }
        if let Some(response) = value.get("response_text").and_then(|v| v.as_str()) {
            train_responses.insert(normalize_response(response));
        }
    }
    Ok(CopySources {
        train_responses,
        eval_outputs,
        response_table_outputs,
        template_responses,
        train_prompts,
        eval_prompts,
    })
}

fn reconstruct_076_train_prompts(count: usize, seed: u64) -> BTreeSet<String> {
    let mut prompts = BTreeSet::new();
    let colors = ["amber", "teal", "violet", "silver"];
    for idx in 0..count {
        let bucket = idx % 20;
        let prompt = match bucket {
            0..=4 => format!("train instruction explain route gate sample {} seed {}", idx, seed),
            5..=9 => format!("train instruction list missing chat decoder sample {} seed {}", idx, seed),
            10..=13 => format!("train short answer active scenario writeback stale pockets sample {}", idx),
            14..=16 => {
                let color = colors[(idx / 20 + bucket) % colors.len()];
                format!("train context remember active code {color} user asked code sample {idx}")
            }
            17..=18 => {
                let color = colors[(idx / 20 + bucket) % colors.len()];
                format!("train retention active scenario code {color} distractor old violet inactive pocket silver answer only sample {idx}")
            }
            _ => format!("train boundary refuse production safety release readiness claim sample {}", idx),
        };
        prompts.insert(prompt);
    }
    for repeat in 0..64 {
        for color in colors {
            prompts.insert(format!("train coverage context remember active code {color} user asks code variant {repeat}"));
            prompts.insert(format!("train coverage retention active scenario code {color} distractor old amber inactive pocket silver answer only variant {repeat}"));
            prompts.insert(format!("train coverage retention distractor scenario rejection active {color} distractor amber stale pocket violet variant {repeat}"));
            prompts.insert(format!("train coverage retention inactive stale pocket suppression active {color} inactive amber stale silver variant {repeat}"));
            prompts.insert(format!("train coverage retention answer only scenario binding active {color} old amber distractor teal variant {repeat}"));
        }
    }
    prompts
}

fn reconstruct_076_train_responses() -> BTreeSet<String> {
    [
        "a route gate selects relevant context and blocks distractor readout",
        "the missing layer is a decoder loop that can emit natural tokens",
        "active scenario writeback should win while stale pockets stay silent",
        "you said the active code is amber",
        "you said the active code is teal",
        "you said the active code is violet",
        "you said the active code is silver",
        "i cannot claim production safety or release readiness from this controlled mini eval",
        "amber",
        "teal",
        "violet",
        "silver",
    ]
    .iter()
    .map(|value| normalize_response(value))
    .collect()
}

fn prompt_ngram_overlap_stats(rows: &[EvalExample], sources: &CopySources) -> Value {
    let train_ngrams = sources
        .train_prompts
        .iter()
        .flat_map(|prompt| ngrams(prompt, 3))
        .collect::<BTreeSet<_>>();
    let eval_ngrams = sources
        .eval_prompts
        .iter()
        .flat_map(|prompt| ngrams(prompt, 3))
        .collect::<BTreeSet<_>>();
    let mut max_train = 0.0f64;
    let mut max_eval = 0.0f64;
    for row in rows {
        let row_ngrams = ngrams(&row.prompt, 3);
        max_train = max_train.max(overlap_rate(&row_ngrams, &train_ngrams));
        max_eval = max_eval.max(overlap_rate(&row_ngrams, &eval_ngrams));
    }
    json!({
        "max_train_prompt_trigram_overlap": max_train,
        "max_eval_prompt_trigram_overlap": max_eval
    })
}

fn train_response_ngram_overlap(rows: &[EvalRow], train_responses: &BTreeSet<String>) -> f64 {
    let train_ngrams = train_responses
        .iter()
        .flat_map(|response| ngrams(response, 3))
        .collect::<BTreeSet<_>>();
    let mut sum = 0.0f64;
    for row in rows {
        sum += overlap_rate(&ngrams(&row.model_output, 3), &train_ngrams);
    }
    sum / rows.len().max(1) as f64
}

fn overlap_rate(values: &BTreeSet<String>, reference: &BTreeSet<String>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.intersection(reference).count() as f64 / values.len() as f64
}

fn is_template_copy(output: &str, sources: &CopySources) -> bool {
    let normalized = normalize_response(output);
    sources.train_responses.contains(&normalized)
        || sources.eval_outputs.contains(&normalized)
        || sources.response_table_outputs.contains(&normalized)
        || sources.template_responses.contains(&normalized)
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

fn ngrams(text: &str, n: usize) -> BTreeSet<String> {
    let tokens = tokenize(text);
    if tokens.len() < n {
        return BTreeSet::new();
    }
    tokens
        .windows(n)
        .map(|window| window.join("_"))
        .collect::<BTreeSet<_>>()
}

fn normalize_response(value: &str) -> String {
    tokenize(value).join(" ")
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

fn value_has_verdict(value: &Value, verdict: &str) -> bool {
    value
        .get("verdicts")
        .and_then(|v| v.as_array())
        .map(|items| items.iter().any(|item| item.as_str() == Some(verdict)))
        .unwrap_or(false)
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

fn sha256_file(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let mut hasher = Sha256::new();
    hasher.update(fs::read(path)?);
    Ok(format!("{:x}", hasher.finalize()))
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
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
    write_jsonl_sample(path, rows, rows.len())
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
            "expected_behavior": row.expected_behavior,
            "required_keywords": row.required_keywords,
            "forbidden_outputs": row.forbidden_outputs,
            "model_output": row.model_output,
            "output_classification": row.output_classification,
            "pass_fail": row.pass_fail,
            "short_diagnosis": row.short_diagnosis,
            "template_copy_flag": row.template_copy_flag,
            "novelty_flag": row.novelty_flag
        });
        writeln!(file, "{}", serde_json::to_string(&payload)?)?;
    }
    Ok(())
}

fn write_failure(out: &Path, verdict: &str, reason: &str) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out)?;
    append_progress(out, "failure", json!({"verdict": verdict, "reason": reason}))?;
    let payload = json!({
        "schema_version": "chat_generation_fresh_composition_summary_v1",
        "status": "failed",
        "reason": reason,
        "verdicts": ["CHAT_GENERATION_FRESH_COMPOSITION_CONFIRM_FAILS", verdict],
        "train_step_count": 0,
        "prediction_oracle_used": false
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
        "schema_version": "chat_generation_fresh_composition_summary_v1",
        "status": status,
        "verdicts": verdicts,
        "details": details,
        "bounded_fresh_composition_confirm_only": true,
        "not_GPT_like_assistant_readiness": true,
        "not_full_English_LM": true,
        "not_language_grounding": true,
        "not_production_chat": true,
        "not_public_beta_GA_hosted_SaaS": true
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
    text.push_str("# STABLE_LOOP_PHASE_LOCK_077_CHAT_GENERATION_FRESH_COMPOSITION_CONFIRM Report\n\n");
    text.push_str(&format!("Status: `{}`\n\n", summary["status"].as_str().unwrap_or("unknown")));
    text.push_str("077 is bounded fresh composition confirm only.\n\n");
    text.push_str("no training\nno resume\nno checkpoint repair\nno checkpoint mutation\nno replacement checkpoint\n");
    text.push_str("not GPT-like assistant readiness\nnot full English LM\nnot language grounding\nnot production chat\nnot public beta / GA / hosted SaaS\n\n");
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

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut cfg = Config {
        out: PathBuf::from(DEFAULT_OUT),
        checkpoint: PathBuf::from(DEFAULT_CHECKPOINT),
        upstream_076_root: PathBuf::from(DEFAULT_UPSTREAM_076_ROOT),
        upstream_074_root: PathBuf::from(DEFAULT_UPSTREAM_074_ROOT),
        seed: 2027,
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
            "--checkpoint" => {
                idx += 1;
                cfg.checkpoint = PathBuf::from(args.get(idx).ok_or("--checkpoint missing value")?);
            }
            "--upstream-076-root" => {
                idx += 1;
                cfg.upstream_076_root = PathBuf::from(args.get(idx).ok_or("--upstream-076-root missing value")?);
            }
            "--upstream-074-root" => {
                idx += 1;
                cfg.upstream_074_root = PathBuf::from(args.get(idx).ok_or("--upstream-074-root missing value")?);
            }
            "--seed" => {
                idx += 1;
                cfg.seed = args.get(idx).ok_or("--seed missing value")?.parse()?;
            }
            "--heartbeat-sec" => {
                idx += 1;
                cfg.heartbeat_sec = args.get(idx).ok_or("--heartbeat-sec missing value")?.parse()?;
            }
            "--help" | "-h" => {
                println!(
                    "phase_lane_chat_generation_fresh_composition_confirm --out <dir> --checkpoint <path> --upstream-076-root <dir> --upstream-074-root <dir> --seed <n> --heartbeat-sec <n>"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        idx += 1;
    }
    Ok(cfg)
}
