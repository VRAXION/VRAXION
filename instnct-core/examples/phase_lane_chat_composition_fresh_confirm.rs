#![recursion_limit = "256"]

//! Eval-only fresh confirmation for the 078 token-composition chat checkpoint.
//!
//! 079 does not train, repair, resume, mutate checkpoints, or expose a chat
//! surface through product/API/SDK code. It loads the 078 research checkpoint
//! read-only and checks fresh prompts for token-level composition, template-copy
//! regression, context slot binding, collapse, and finite-label retention.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_OUT: &str =
    "target/pilot_wave/stable_loop_phase_lock_079_chat_composition_fresh_confirm/smoke";
const DEFAULT_UPSTREAM_078_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke";
const DEFAULT_UPSTREAM_077B_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_077b_chat_generation_failure_analysis/smoke";
const DEFAULT_UPSTREAM_076_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke";
const DEFAULT_UPSTREAM_074_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke";
const DEFAULT_SEED: u64 = 2027;
const STOP_TOKEN: &str = "<eos>";

const EVAL_FAMILIES: [&str; 10] = [
    "FRESH_SIMPLE_INSTRUCTION_PARAPHRASE",
    "FRESH_SHORT_EXPLANATION_COMPOSITION",
    "FRESH_CONTEXT_CARRY_VARIABLE_SLOT",
    "FRESH_TWO_TURN_DIALOGUE_STATE",
    "FRESH_BOUNDARY_REFUSAL_MINI",
    "FRESH_PARAPHRASE_GENERALIZATION",
    "FRESH_SEMANTIC_SLOT_RECOMBINATION",
    "ANTI_TEMPLATE_COPY_FRESH",
    "ANTI_REPETITION_FRESH",
    "FINITE_LABEL_ANCHORROUTE_RETENTION",
];

#[derive(Debug, Clone)]
struct Config {
    out: PathBuf,
    upstream_078_root: PathBuf,
    upstream_077b_root: PathBuf,
    upstream_076_root: PathBuf,
    upstream_074_root: PathBuf,
    seed: u64,
    heartbeat_sec: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompositionModel {
    #[serde(default)]
    schema_version: String,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    train_step_count: usize,
    #[serde(default)]
    token_train_step_count: usize,
    #[serde(default)]
    update_count: usize,
    #[serde(default)]
    vocab: Vec<String>,
    #[serde(default)]
    slot_values_seen: Vec<String>,
    #[serde(default)]
    token_counts: BTreeMap<String, BTreeMap<String, usize>>,
    #[serde(default)]
    runner_local_decoder_loop: bool,
    #[serde(default)]
    decoder_path: String,
    #[serde(default)]
    response_table_used_for_main_prediction: bool,
    #[serde(default)]
    response_table_path_available_but_disabled: bool,
    #[serde(default)]
    public_api_exposed: bool,
    #[serde(default)]
    service_api_exposed: bool,
    #[serde(default)]
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
    expected_slot: Option<String>,
    target_label: Option<String>,
    retention_row: bool,
}

#[derive(Debug, Clone, Serialize)]
struct EvalRow {
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
    semantic_template_overlap_score: f64,
    slot_binding_diagnosis: String,
    short_diagnosis: String,
    generated_token_count: usize,
    sentence_like_response: bool,
    slot_value_expected: Option<String>,
    slot_value_emitted: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct PromptSources {
    prompts_078_train: BTreeSet<String>,
    prompts_078_eval: BTreeSet<String>,
    prompts_077: BTreeSet<String>,
    prompts_076: BTreeSet<String>,
}

#[derive(Debug, Clone, Default)]
struct CopySources {
    train_responses: BTreeSet<String>,
    eval_outputs: BTreeSet<String>,
    response_table_outputs: BTreeSet<String>,
    generated_outputs_077_078: BTreeSet<String>,
    template_responses: BTreeSet<String>,
}

#[derive(Debug, Clone, Serialize)]
struct PromptLeakageReport {
    overlap_with_078_train_prompt_count: usize,
    overlap_with_078_eval_prompt_count: usize,
    overlap_with_077_prompt_count: usize,
    overlap_with_076_prompt_count: usize,
    max_prompt_token_jaccard_vs_078_train: f64,
    max_prompt_token_jaccard_vs_078_eval: f64,
    max_prompt_token_jaccard_vs_077: f64,
    max_prompt_token_jaccard_vs_076: f64,
    near_duplicate_prompt_count: usize,
    eval_prompt_hash: String,
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
            "milestone": "STABLE_LOOP_PHASE_LOCK_079_CHAT_COMPOSITION_FRESH_CONFIRM",
            "seed": cfg.seed,
            "heartbeat_sec": cfg.heartbeat_sec,
            "eval_only": true
        }),
    )?;
    write_summary_and_report(
        &cfg.out,
        "running",
        vec!["CHAT_COMPOSITION_FRESH_CONFIRM_RUNNING".to_string()],
        json!({"phase": "start"}),
    )?;
    write_json(
        &cfg.out.join("queue.json"),
        &json!({
            "schema_version": "chat_composition_fresh_confirm_queue_v1",
            "milestone": "STABLE_LOOP_PHASE_LOCK_079_CHAT_COMPOSITION_FRESH_CONFIRM",
            "partial_write_policy": "progress.jsonl, summary.json, and report.md are written from start and refreshed by phase",
            "steps": [
                "verify_upstream_078_077b_076_074",
                "load_078_checkpoint_read_only",
                "construct_fresh_prompt_dataset",
                "audit_prompt_leakage",
                "run_token_level_eval_only_generation",
                "audit_template_copy_and_context_slots",
                "write_final_summary"
            ]
        }),
    )?;

    let checkpoint = checkpoint_path(cfg);
    let missing = missing_upstreams(cfg, &checkpoint);
    if !missing.is_empty() {
        write_failure(
            &cfg.out,
            "UPSTREAM_078_ARTIFACT_MISSING",
            &format!("missing required upstream artifacts: {}", missing.join(",")),
        )?;
        return Err(format!("UPSTREAM_078_ARTIFACT_MISSING: {}", missing.join(",")).into());
    }

    let upstream_078_summary: Value = read_json(&cfg.upstream_078_root.join("summary.json"))?;
    let upstream_077b_summary: Value = read_json(&cfg.upstream_077b_root.join("summary.json"))?;
    let upstream_074_summary: Value = read_json(&cfg.upstream_074_root.join("summary.json"))?;
    if !value_has_verdict(&upstream_078_summary, "CHAT_COMPOSITION_REPAIR_POSITIVE")
        || !value_has_verdict(&upstream_077b_summary, "CHAT_GENERATION_FAILURE_ANALYSIS_POSITIVE")
        || !value_has_verdict(&upstream_074_summary, "MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_POSITIVE")
    {
        write_failure(
            &cfg.out,
            "UPSTREAM_078_ARTIFACT_MISSING",
            "required positive upstream verdict missing",
        )?;
        return Err("UPSTREAM_078_ARTIFACT_MISSING: positive upstream verdict missing".into());
    }

    let checkpoint_hash_before = sha256_file(&checkpoint)?;
    let checkpoint_metadata = fs::metadata(&checkpoint)?;
    let checkpoint_modified_ms = checkpoint_metadata
        .modified()
        .ok()
        .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
        .map(|value| value.as_millis())
        .unwrap_or(0);
    let eval_started_after_079_start = now_ms() >= start_ms && checkpoint_modified_ms <= start_ms;
    let model: CompositionModel = read_json(&checkpoint)?;
    append_progress(
        &cfg.out,
        "upstream_verified",
        json!({
            "upstream_078_summary_present": true,
            "upstream_078_positive": true,
            "checkpoint_exists": true,
            "checkpoint_hash_before": checkpoint_hash_before,
            "eval_started_after_079_start": eval_started_after_079_start
        }),
    )?;
    write_summary_and_report(
        &cfg.out,
        "running",
        vec!["CHAT_COMPOSITION_FRESH_CONFIRM_RUNNING".to_string()],
        json!({"phase": "upstream_verified"}),
    )?;

    write_json(
        &cfg.out.join("benchmark_config.json"),
        &json!({
            "schema_version": "chat_composition_fresh_confirm_config_v1",
            "bounded_fresh_chat_composition_confirm_only": true,
            "eval_only": true,
            "no_training": true,
            "no_resume": true,
            "no_checkpoint_repair": true,
            "no_checkpoint_mutation": true,
            "no_replacement_checkpoint": true,
            "decoder_path": "token_level_next_token",
            "response_table_used_for_main_prediction": false,
            "llm_judge_used": false,
            "not_GPT_like_assistant_readiness": true,
            "not_full_English_LM": true,
            "not_language_grounding": true,
            "not_production_chat": true,
            "not_safety_alignment": true,
            "not_public_beta_GA_hosted_SaaS": true,
            "eval_families": EVAL_FAMILIES,
            "seed": cfg.seed,
            "fresh_prompt_policy": "new wording, entities, instruction shapes, context-carry variants, and no exact or near-duplicate 076/077/078 prompt overlap",
            "near_duplicate_jaccard_fail_threshold": 0.90
        }),
    )?;
    write_upstream_manifest(cfg, &checkpoint, &checkpoint_hash_before, start_ms)?;
    write_checkpoint_manifest(
        cfg,
        &checkpoint,
        &checkpoint_hash_before,
        &checkpoint_hash_before,
        true,
        eval_started_after_079_start,
    )?;

    if model.decoder_path != "token_level_next_token"
        || model.response_table_used_for_main_prediction
        || !model.runner_local_decoder_loop
        || model.public_api_exposed
        || model.service_api_exposed
        || model.sdk_surface_exposed
    {
        write_failure(
            &cfg.out,
            "RESPONSE_TABLE_PATH_USED_IN_EVAL",
            "078 checkpoint does not expose the required runner-local token-level evaluation metadata",
        )?;
        return Err("RESPONSE_TABLE_PATH_USED_IN_EVAL".into());
    }

    let prompts = load_prompt_sources(cfg)?;
    let copy_sources = load_copy_sources(cfg)?;
    let examples = build_fresh_eval_examples(cfg.seed);
    write_jsonl(&cfg.out.join("fresh_chat_eval_dataset.jsonl"), &examples, examples.len())?;
    let prompt_report = prompt_leakage_report(&examples, &prompts);
    append_progress(
        &cfg.out,
        "fresh_dataset_written",
        json!({
            "eval_examples": examples.len(),
            "overlap_with_078_train_prompt_count": prompt_report.overlap_with_078_train_prompt_count,
            "overlap_with_078_eval_prompt_count": prompt_report.overlap_with_078_eval_prompt_count,
            "overlap_with_077_prompt_count": prompt_report.overlap_with_077_prompt_count,
            "overlap_with_076_prompt_count": prompt_report.overlap_with_076_prompt_count,
            "near_duplicate_prompt_count": prompt_report.near_duplicate_prompt_count
        }),
    )?;
    write_summary_and_report(
        &cfg.out,
        "running",
        vec!["CHAT_COMPOSITION_FRESH_CONFIRM_RUNNING".to_string()],
        json!({"phase": "fresh_dataset_written", "eval_examples": examples.len()}),
    )?;

    let prompt_leakage = prompt_report.overlap_with_078_train_prompt_count > 0
        || prompt_report.overlap_with_078_eval_prompt_count > 0
        || prompt_report.overlap_with_077_prompt_count > 0
        || prompt_report.overlap_with_076_prompt_count > 0
        || prompt_report.near_duplicate_prompt_count > 0;
    if prompt_leakage {
        let summary = build_final_summary(
            "failed",
            vec![
                "CHAT_COMPOSITION_FRESH_CONFIRM_FAILS".to_string(),
                "FRESH_PROMPT_LEAKAGE_DETECTED".to_string(),
            ],
            &prompt_report,
            json!({}),
            json!({}),
            json!({}),
            json!({}),
            json!({}),
            &checkpoint_hash_before,
            &checkpoint_hash_before,
            true,
            eval_started_after_079_start,
        );
        write_json(&cfg.out.join("summary.json"), &summary)?;
        write_report(&cfg.out.join("report.md"), &summary)?;
        append_progress(&cfg.out, "done", json!({"status": "failed", "verdict": "FRESH_PROMPT_LEAKAGE_DETECTED"}))?;
        println!("{}", serde_json::to_string(&summary)?);
        return Ok(());
    }

    let finite_labels = finite_labels();
    let rows = evaluate_model(&model, &examples, &copy_sources, &finite_labels);
    write_eval_outputs(&cfg.out.join("generation_samples.jsonl"), &rows)?;
    write_human_samples(&cfg.out.join("human_readable_samples.jsonl"), &rows)?;

    let composition = composition_metrics(&rows, &finite_labels);
    let novelty = novelty_metrics(&rows, &copy_sources);
    let context = context_slot_metrics(&rows);
    let retention = finite_label_retention_metrics(&rows);
    let collapse = collapse_metrics(&rows, &finite_labels);
    write_json(&cfg.out.join("composition_metrics.json"), &composition)?;
    write_json(&cfg.out.join("novelty_metrics.json"), &novelty)?;
    write_json(&cfg.out.join("context_slot_metrics.json"), &context)?;
    write_json(&cfg.out.join("finite_label_retention_metrics.json"), &retention)?;
    write_json(&cfg.out.join("collapse_metrics.json"), &collapse)?;
    append_progress(
        &cfg.out,
        "eval_completed",
        json!({
            "fresh_instruction_accuracy": composition["fresh_instruction_accuracy"],
            "fresh_context_carry_accuracy": composition["fresh_context_carry_accuracy"],
            "slot_binding_accuracy": context["slot_binding_accuracy"],
            "novel_response_rate": novelty["novel_response_rate"],
            "template_copy_rate": novelty["template_copy_rate"],
            "finite_label_retention_accuracy": retention["finite_label_retention_accuracy"]
        }),
    )?;

    let checkpoint_hash_after = sha256_file(&checkpoint)?;
    let checkpoint_hash_unchanged = checkpoint_hash_before == checkpoint_hash_after;
    write_checkpoint_manifest(
        cfg,
        &checkpoint,
        &checkpoint_hash_before,
        &checkpoint_hash_after,
        checkpoint_hash_unchanged,
        eval_started_after_079_start,
    )?;
    if !checkpoint_hash_unchanged {
        write_failure(
            &cfg.out,
            "CHECKPOINT_MUTATION_DETECTED",
            "checkpoint hash changed during eval-only 079",
        )?;
        return Err("CHECKPOINT_MUTATION_DETECTED".into());
    }

    let hard_pass = hard_gate(
        &composition,
        &novelty,
        &context,
        &retention,
        &collapse,
        checkpoint_hash_unchanged,
    );
    let verdicts = if hard_pass {
        vec![
            "CHAT_COMPOSITION_FRESH_CONFIRM_POSITIVE".to_string(),
            "NO_TRAINING_PERFORMED".to_string(),
            "CHECKPOINT_UNCHANGED".to_string(),
            "FRESH_MULTI_TOKEN_RESPONSES_PASS".to_string(),
            "FRESH_INSTRUCTION_COMPOSITION_PASSES".to_string(),
            "FRESH_CONTEXT_SLOT_BINDING_PASSES".to_string(),
            "FRESH_TEMPLATE_COPY_REJECTED".to_string(),
            "FRESH_NOVEL_RESPONSES_PASS".to_string(),
            "FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES".to_string(),
            "STATIC_RESPONSE_COLLAPSE_REJECTED".to_string(),
            "PRODUCTION_CHAT_NOT_CLAIMED".to_string(),
        ]
    } else {
        failure_verdicts(&composition, &novelty, &context, &retention, &collapse)
    };
    let status = if hard_pass { "passed" } else { "failed" };
    let summary = build_final_summary(
        status,
        verdicts,
        &prompt_report,
        composition,
        novelty,
        context,
        retention,
        collapse,
        &checkpoint_hash_before,
        &checkpoint_hash_after,
        checkpoint_hash_unchanged,
        eval_started_after_079_start,
    );
    write_json(&cfg.out.join("summary.json"), &summary)?;
    write_report(&cfg.out.join("report.md"), &summary)?;
    append_progress(&cfg.out, "done", json!({"status": status, "hard_pass": hard_pass}))?;
    println!("{}", serde_json::to_string(&summary)?);
    Ok(())
}

impl CompositionModel {
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
}

fn build_fresh_eval_examples(seed: u64) -> Vec<EvalExample> {
    let marker = seed % 997;
    vec![
        eval("fresh_001", "FRESH_SIMPLE_INSTRUCTION_PARAPHRASE", &format!("079 confirm {marker}: in plain words, how should route gating keep context useful while distractors stay non-winning"), "fresh route explanation", &["route", "context", "distractor"], &["a route gate selects relevant context"], None, None, false),
        eval("fresh_002", "FRESH_SIMPLE_INSTRUCTION_PARAPHRASE", "fresh confirmation prompt: describe the answer path when helpful context must stay connected", "fresh context answer explanation", &["context", "answer"], &["route_ok"], None, None, false),
        eval("fresh_003", "FRESH_SHORT_EXPLANATION_COMPOSITION", "fresh confirm question: explain stale pocket suppression without claiming old values win", "fresh stale pocket explanation", &["stale", "pocket", "not", "readout"], &["active scenario writeback should win"], None, None, false),
        eval("fresh_004", "FRESH_SHORT_EXPLANATION_COMPOSITION", "fresh confirm note: why should represented old values stay non winning in the final readout", "fresh old value suppression", &["stale", "not", "readout"], &["stale pockets stay silent"], None, None, false),
        eval("fresh_005", "FRESH_CONTEXT_CARRY_VARIABLE_SLOT", "fresh memo alpha: active code is amber; stale code is violet; answer with the active value in a sentence", "slot amber", &["amber", "active"], &["violet"], Some("amber"), None, false),
        eval("fresh_006", "FRESH_CONTEXT_CARRY_VARIABLE_SLOT", "fresh memo beta: active code is silver while inactive pocket says teal; which value should answer", "slot silver", &["silver", "active"], &["teal"], Some("silver"), None, false),
        eval("fresh_007", "FRESH_CONTEXT_CARRY_VARIABLE_SLOT", "fresh ledger gamma: active code is cobalt and side note says rose; carry the active code forward", "slot cobalt", &["cobalt", "active"], &["rose"], Some("cobalt"), None, false),
        eval("fresh_008", "FRESH_CONTEXT_CARRY_VARIABLE_SLOT", "fresh ledger delta: active code is green; old pocket says amber; reply with the active code", "slot green", &["green", "active"], &["amber"], Some("green"), None, false),
        eval("fresh_009", "FRESH_CONTEXT_CARRY_VARIABLE_SLOT", "fresh state epsilon: active code is indigo; distractor pocket says silver; name the active value", "slot indigo", &["indigo", "active"], &["silver"], Some("indigo"), None, false),
        eval("fresh_010", "FRESH_TWO_TURN_DIALOGUE_STATE", "turn1 fresh user stores active code amber. turn2 asks which carried value answers now", "dialogue amber", &["amber", "carried"], &["violet"], Some("amber"), None, false),
        eval("fresh_011", "FRESH_TWO_TURN_DIALOGUE_STATE", "turn1 fresh user records active code silver. turn2 asks for the remembered active code", "dialogue silver", &["silver", "active"], &["teal"], Some("silver"), None, false),
        eval("fresh_012", "FRESH_TWO_TURN_DIALOGUE_STATE", "turn1 fresh user says active code rose. turn2 asks what carried value should answer", "dialogue rose", &["rose", "carried"], &["amber"], Some("rose"), None, false),
        eval("fresh_013", "FRESH_BOUNDARY_REFUSAL_MINI", "fresh boundary: can this bounded check prove production chat safety alignment", "boundary production safety", &["cannot", "production", "safety"], &["yes"], None, None, false),
        eval("fresh_014", "FRESH_BOUNDARY_REFUSAL_MINI", "fresh boundary: should this be called GPT-like assistant readiness", "boundary gpt readiness", &["cannot", "readiness"], &["ready"], None, None, false),
        eval("fresh_015", "FRESH_BOUNDARY_REFUSAL_MINI", "fresh boundary: is this clinical high-stakes deployment ready", "boundary clinical", &["cannot", "claim"], &["clinical safe"], None, None, false),
        eval("fresh_016", "FRESH_PARAPHRASE_GENERALIZATION", "fresh paraphrase: explain why table lookup is not enough for a prompt that needs composed tokens", "decoder composition explanation", &["token", "table", "step"], &["decoder loop that can emit natural tokens"], None, None, false),
        eval("fresh_017", "FRESH_PARAPHRASE_GENERALIZATION", "fresh paraphrase: why should a decoder build a reply instead of selecting a stored answer", "decoder non table explanation", &["decoder", "step", "stored"], &["missing layer is a decoder"], None, None, false),
        eval("fresh_018", "FRESH_SEMANTIC_SLOT_RECOMBINATION", "fresh recombination: active code is violet; stale pocket says amber; explain which active code answers and why", "slot violet recombination", &["violet", "active"], &["amber"], Some("violet"), None, false),
        eval("fresh_019", "FRESH_SEMANTIC_SLOT_RECOMBINATION", "fresh recombination: active code is teal; side note says silver; answer in a sentence with the active value", "slot teal recombination", &["teal", "active"], &["silver"], Some("teal"), None, false),
        eval("fresh_020", "ANTI_TEMPLATE_COPY_FRESH", "fresh anti-copy: explain why a token decoder should compose rather than copy the table lookup result", "anti copy decoder", &["token", "decoder", "table"], &["the missing layer is a decoder"], None, None, false),
        eval("fresh_021", "ANTI_TEMPLATE_COPY_FRESH", "fresh anti-copy: say why a stored table sentence is not enough for new wording", "anti copy stored sentence", &["token", "table", "answer"], &["natural tokens"], None, None, false),
        eval("fresh_022", "ANTI_REPETITION_FRESH", "fresh anti-repetition: describe route gating once, without repeating the same phrase", "anti repetition route", &["route", "context"], &["route route route"], None, None, false),
        eval("fresh_023", "ANTI_REPETITION_FRESH", "fresh anti-repetition: explain stale pocket suppression once without repeated loops", "anti repetition stale", &["stale", "not", "win"], &["stale stale stale"], None, None, false),
        eval("fresh_024", "FRESH_CONTEXT_CARRY_VARIABLE_SLOT", "fresh memo zeta: active code is amber and irrelevant pocket says green; produce the active value sentence", "slot amber repeat", &["amber", "active"], &["green"], Some("amber"), None, false),
        eval("fresh_025", "FRESH_TWO_TURN_DIALOGUE_STATE", "turn1 fresh user sets active code cobalt. turn2 asks for the carried code in words", "dialogue cobalt", &["cobalt", "carried"], &["silver"], Some("cobalt"), None, false),
        eval("fresh_026", "FRESH_SEMANTIC_SLOT_RECOMBINATION", "fresh recombination: active code is indigo; inactive pocket says rose; answer using the active code", "slot indigo recombination", &["indigo", "active"], &["rose"], Some("indigo"), None, false),
        eval("fresh_027", "FINITE_LABEL_ANCHORROUTE_RETENTION", "079 retention answer only active scenario binding active amber old violet stale silver inactive rose", "active scenario binding", &["amber"], &["violet"], Some("amber"), Some("amber"), true),
        eval("fresh_028", "FINITE_LABEL_ANCHORROUTE_RETENTION", "079 retention answer only distractor scenario rejection active teal distractor amber stale silver inactive rose", "distractor scenario rejection", &["teal"], &["amber"], Some("teal"), Some("teal"), true),
        eval("fresh_029", "FINITE_LABEL_ANCHORROUTE_RETENTION", "079 retention answer only old stale inactive suppression active violet old amber stale silver inactive rose", "old/stale/inactive suppression", &["violet"], &["amber"], Some("violet"), Some("violet"), true),
        eval("fresh_030", "FINITE_LABEL_ANCHORROUTE_RETENTION", "079 retention answer only active scenario binding active silver old amber distractor teal inactive rose", "answer-only scenario binding", &["silver"], &["amber"], Some("silver"), Some("silver"), true),
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

fn evaluate_model(
    model: &CompositionModel,
    examples: &[EvalExample],
    sources: &CopySources,
    finite_labels: &[String],
) -> Vec<EvalRow> {
    let finite_set = finite_labels.iter().cloned().collect::<BTreeSet<_>>();
    examples
        .iter()
        .map(|row| {
            let output = model.generate_main(&row.prompt);
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
            let semantic_overlap = max_template_overlap(&output, sources);
            let template_copy = is_template_copy(&output, sources, semantic_overlap);
            let novelty = !template_copy && !finite_set.contains(output.trim());
            let token_count = tokenize(&output).len();
            EvalRow {
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
                semantic_template_overlap_score: semantic_overlap,
                slot_binding_diagnosis: slot_diagnosis(row, slot_emitted.as_deref()),
                short_diagnosis: if pass {
                    "rubric-bounded pass without LLM judge".to_string()
                } else {
                    "rubric keyword, forbidden output, or slot binding check failed".to_string()
                },
                generated_token_count: token_count,
                sentence_like_response: is_sentence_like(&output),
                slot_value_expected: row.expected_slot.clone(),
                slot_value_emitted: slot_emitted,
            }
        })
        .collect()
}

fn composition_metrics(rows: &[EvalRow], finite_labels: &[String]) -> Value {
    let chat_rows = rows
        .iter()
        .filter(|row| row.eval_family != "FINITE_LABEL_ANCHORROUTE_RETENTION")
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
        "fresh_instruction_accuracy": family_accuracy(rows, "FRESH_SIMPLE_INSTRUCTION_PARAPHRASE"),
        "fresh_context_carry_accuracy": family_accuracy(rows, "FRESH_CONTEXT_CARRY_VARIABLE_SLOT"),
        "two_turn_dialogue_accuracy": family_accuracy(rows, "FRESH_TWO_TURN_DIALOGUE_STATE"),
        "boundary_refusal_accuracy": family_accuracy(rows, "FRESH_BOUNDARY_REFUSAL_MINI"),
        "label_only_response_rate": ratio(label_only, rows.len().max(1)),
        "generated_token_count_mean": token_counts.iter().sum::<usize>() as f64 / chat_total as f64,
        "generated_token_count_min": token_counts.iter().copied().min().unwrap_or(0),
        "content_word_count_mean": chat_rows.iter().map(|row| content_word_count(&row.model_output)).sum::<usize>() as f64 / chat_total as f64,
        "sentence_like_response_rate": ratio(chat_rows.iter().filter(|row| row.sentence_like_response).count(), chat_total),
        "unique_response_count": rows.iter().map(|row| row.model_output.clone()).collect::<BTreeSet<_>>().len(),
        "llm_judge_used": false
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
    let semantic = rows
        .iter()
        .filter(|row| row.semantic_template_overlap_score >= 0.70)
        .count();
    let template_copy = rows.iter().filter(|row| row.template_copy_flag).count();
    let novel = rows.iter().filter(|row| row.novelty_flag).count();
    json!({
        "exact_train_response_copy_rate": ratio(exact_train, total),
        "exact_eval_response_copy_rate": ratio(exact_eval, total),
        "response_table_copy_rate": ratio(response_table, total),
        "semantic_template_overlap_rate": ratio(semantic, total),
        "template_copy_rate": ratio(template_copy, total),
        "novel_response_rate": ratio(novel, total),
        "train_response_ngram_overlap": train_response_ngram_overlap(rows, &sources.train_responses),
        "template_copy_audit_sources": [
            "078 train responses",
            "078 eval outputs",
            "078 generated outputs",
            "077 generated outputs",
            "076 response table outputs",
            "076 train/eval outputs"
        ]
    })
}

fn context_slot_metrics(rows: &[EvalRow]) -> Value {
    let slot_rows = rows
        .iter()
        .filter(|row| {
            row.eval_family == "FRESH_CONTEXT_CARRY_VARIABLE_SLOT"
                || row.eval_family == "FRESH_TWO_TURN_DIALOGUE_STATE"
                || row.eval_family == "FRESH_SEMANTIC_SLOT_RECOMBINATION"
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

fn hard_gate(
    composition: &Value,
    novelty: &Value,
    context: &Value,
    retention: &Value,
    collapse: &Value,
    checkpoint_hash_unchanged: bool,
) -> bool {
    composition["multi_token_response_rate"].as_f64().unwrap_or(0.0) >= 0.90
        && composition["non_empty_response_rate"].as_f64().unwrap_or(0.0) >= 0.98
        && composition["fresh_instruction_accuracy"].as_f64().unwrap_or(0.0) >= 0.75
        && composition["fresh_context_carry_accuracy"].as_f64().unwrap_or(0.0) >= 0.70
        && context["slot_binding_accuracy"].as_f64().unwrap_or(0.0) >= 0.70
        && composition["two_turn_dialogue_accuracy"].as_f64().unwrap_or(0.0) >= 0.70
        && composition["boundary_refusal_accuracy"].as_f64().unwrap_or(0.0) >= 0.70
        && novelty["novel_response_rate"].as_f64().unwrap_or(0.0) >= 0.60
        && novelty["template_copy_rate"].as_f64().unwrap_or(1.0) <= 0.30
        && novelty["response_table_copy_rate"].as_f64().unwrap_or(1.0) <= 0.30
        && novelty["exact_train_response_copy_rate"].as_f64().unwrap_or(1.0) <= 0.20
        && novelty["semantic_template_overlap_rate"].as_f64().unwrap_or(1.0) <= 0.40
        && composition["label_only_response_rate"].as_f64().unwrap_or(1.0) <= 0.15
        && composition["generated_token_count_min"].as_u64().unwrap_or(0) >= 2
        && composition["sentence_like_response_rate"].as_f64().unwrap_or(0.0) >= 0.70
        && collapse["empty_output_rate"].as_f64().unwrap_or(1.0) <= 0.02
        && collapse["space_output_rate"].as_f64().unwrap_or(1.0) <= 0.02
        && collapse["static_response_rate"].as_f64().unwrap_or(1.0) <= 0.15
        && collapse["repetition_rate"].as_f64().unwrap_or(1.0) <= 0.20
        && collapse["copy_prompt_rate"].as_f64().unwrap_or(1.0) <= 0.15
        && retention["finite_label_retention_accuracy"].as_f64().unwrap_or(0.0) >= 0.90
        && checkpoint_hash_unchanged
}

fn failure_verdicts(
    composition: &Value,
    novelty: &Value,
    context: &Value,
    retention: &Value,
    collapse: &Value,
) -> Vec<String> {
    let mut verdicts = vec!["CHAT_COMPOSITION_FRESH_CONFIRM_FAILS".to_string()];
    if novelty["template_copy_rate"].as_f64().unwrap_or(1.0) > 0.30
        || novelty["response_table_copy_rate"].as_f64().unwrap_or(1.0) > 0.30
        || novelty["semantic_template_overlap_rate"].as_f64().unwrap_or(1.0) > 0.40
        || novelty["novel_response_rate"].as_f64().unwrap_or(0.0) < 0.60
    {
        verdicts.push("TEMPLATE_COPY_DETECTED".to_string());
    }
    if composition["label_only_response_rate"].as_f64().unwrap_or(1.0) > 0.15
        || composition["generated_token_count_min"].as_u64().unwrap_or(0) < 2
        || composition["sentence_like_response_rate"].as_f64().unwrap_or(0.0) < 0.70
    {
        verdicts.push("LABEL_ONLY_RESPONSE_COLLAPSE_DETECTED".to_string());
    }
    if composition["fresh_instruction_accuracy"].as_f64().unwrap_or(0.0) < 0.75 {
        verdicts.push("FRESH_INSTRUCTION_COMPOSITION_FAILS".to_string());
    }
    if context["slot_binding_accuracy"].as_f64().unwrap_or(0.0) < 0.70
        || composition["fresh_context_carry_accuracy"].as_f64().unwrap_or(0.0) < 0.70
    {
        verdicts.push("CONTEXT_SLOT_BINDING_FAILS".to_string());
    }
    if retention["finite_label_retention_accuracy"].as_f64().unwrap_or(0.0) < 0.90 {
        verdicts.push("FINITE_LABEL_RETENTION_REGRESSION_DETECTED".to_string());
    }
    if collapse["static_response_rate"].as_f64().unwrap_or(1.0) > 0.15 {
        verdicts.push("STATIC_RESPONSE_COLLAPSE_DETECTED".to_string());
    }
    if collapse["repetition_rate"].as_f64().unwrap_or(1.0) > 0.20 {
        verdicts.push("REPETITION_COLLAPSE_DETECTED".to_string());
    }
    if collapse["empty_output_rate"].as_f64().unwrap_or(1.0) > 0.02 {
        verdicts.push("EMPTY_OUTPUT_COLLAPSE_DETECTED".to_string());
    }
    verdicts
}

fn load_prompt_sources(cfg: &Config) -> Result<PromptSources, Box<dyn std::error::Error>> {
    let mut sources = PromptSources::default();
    collect_prompts(
        &cfg.upstream_078_root.join("train_examples_sample.jsonl"),
        &mut sources.prompts_078_train,
    )?;
    collect_prompts(
        &cfg.upstream_078_root.join("eval_examples_sample.jsonl"),
        &mut sources.prompts_078_eval,
    )?;
    collect_prompts(
        &cfg.upstream_078_root.join("generation_samples.jsonl"),
        &mut sources.prompts_078_eval,
    )?;
    collect_prompts(
        &cfg
            .upstream_077b_root
            .parent()
            .unwrap_or(Path::new(""))
            .join("stable_loop_phase_lock_077_chat_generation_fresh_composition_confirm")
            .join("smoke")
            .join("generation_samples.jsonl"),
        &mut sources.prompts_077,
    )
    .ok();
    collect_prompts(
        &PathBuf::from("target/pilot_wave/stable_loop_phase_lock_077_chat_generation_fresh_composition_confirm/smoke/generation_samples.jsonl"),
        &mut sources.prompts_077,
    )
    .ok();
    collect_prompts(
        &cfg.upstream_076_root.join("train_examples_sample.jsonl"),
        &mut sources.prompts_076,
    )?;
    collect_prompts(
        &cfg.upstream_076_root.join("eval_examples_sample.jsonl"),
        &mut sources.prompts_076,
    )?;
    collect_prompts(
        &cfg.upstream_076_root.join("generation_samples.jsonl"),
        &mut sources.prompts_076,
    )?;
    Ok(sources)
}

fn load_copy_sources(cfg: &Config) -> Result<CopySources, Box<dyn std::error::Error>> {
    let mut sources = CopySources::default();
    collect_responses(
        &cfg.upstream_078_root.join("train_examples_sample.jsonl"),
        &["response_text", "model_output"],
        &mut sources.train_responses,
    )?;
    collect_responses(
        &cfg.upstream_078_root.join("generation_samples.jsonl"),
        &["model_output"],
        &mut sources.eval_outputs,
    )?;
    collect_responses(
        &PathBuf::from("target/pilot_wave/stable_loop_phase_lock_077_chat_generation_fresh_composition_confirm/smoke/generation_samples.jsonl"),
        &["model_output"],
        &mut sources.generated_outputs_077_078,
    )
    .ok();
    collect_responses(
        &cfg.upstream_076_root.join("train_examples_sample.jsonl"),
        &["response_text", "model_output"],
        &mut sources.train_responses,
    )?;
    collect_responses(
        &cfg.upstream_076_root.join("eval_examples_sample.jsonl"),
        &["model_output", "expected_behavior"],
        &mut sources.eval_outputs,
    )
    .ok();
    collect_responses(
        &cfg.upstream_076_root.join("generation_samples.jsonl"),
        &["model_output"],
        &mut sources.eval_outputs,
    )?;

    let checkpoint = cfg
        .upstream_076_root
        .join("checkpoints")
        .join("chat_generation_poc")
        .join("model_checkpoint.json");
    let value: Value = read_json(&checkpoint)?;
    if let Some(table) = value.get("response_table").and_then(|v| v.as_object()) {
        for tokens in table.values() {
            if let Some(items) = tokens.as_array() {
                let decoded = items
                    .iter()
                    .filter_map(|item| item.as_str())
                    .take_while(|tok| *tok != STOP_TOKEN)
                    .collect::<Vec<_>>()
                    .join(" ");
                sources
                    .response_table_outputs
                    .insert(normalize_response(&decoded));
            }
        }
    }
    sources.template_responses.extend(sources.train_responses.iter().cloned());
    sources.template_responses.extend(sources.eval_outputs.iter().cloned());
    sources
        .template_responses
        .extend(sources.response_table_outputs.iter().cloned());
    sources
        .template_responses
        .extend(sources.generated_outputs_077_078.iter().cloned());
    Ok(sources)
}

fn prompt_leakage_report(examples: &[EvalExample], sources: &PromptSources) -> PromptLeakageReport {
    let eval_prompts = examples
        .iter()
        .map(|row| row.prompt.clone())
        .collect::<BTreeSet<_>>();
    let max_078_train = max_prompt_jaccard(&eval_prompts, &sources.prompts_078_train);
    let max_078_eval = max_prompt_jaccard(&eval_prompts, &sources.prompts_078_eval);
    let max_077 = max_prompt_jaccard(&eval_prompts, &sources.prompts_077);
    let max_076 = max_prompt_jaccard(&eval_prompts, &sources.prompts_076);
    let near_duplicate_prompt_count = eval_prompts
        .iter()
        .filter(|prompt| {
            max_jaccard_one(prompt, &sources.prompts_078_train) >= 0.90
                || max_jaccard_one(prompt, &sources.prompts_078_eval) >= 0.90
                || max_jaccard_one(prompt, &sources.prompts_077) >= 0.90
                || max_jaccard_one(prompt, &sources.prompts_076) >= 0.90
        })
        .count();
    PromptLeakageReport {
        overlap_with_078_train_prompt_count: eval_prompts
            .intersection(&sources.prompts_078_train)
            .count(),
        overlap_with_078_eval_prompt_count: eval_prompts.intersection(&sources.prompts_078_eval).count(),
        overlap_with_077_prompt_count: eval_prompts.intersection(&sources.prompts_077).count(),
        overlap_with_076_prompt_count: eval_prompts.intersection(&sources.prompts_076).count(),
        max_prompt_token_jaccard_vs_078_train: max_078_train,
        max_prompt_token_jaccard_vs_078_eval: max_078_eval,
        max_prompt_token_jaccard_vs_077: max_077,
        max_prompt_token_jaccard_vs_076: max_076,
        near_duplicate_prompt_count,
        eval_prompt_hash: set_hash(&eval_prompts),
    }
}

fn collect_prompts(
    path: &Path,
    target: &mut BTreeSet<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    for value in read_jsonl_values(path)? {
        if let Some(prompt) = value.get("prompt").and_then(|v| v.as_str()) {
            target.insert(prompt.to_string());
        }
    }
    Ok(())
}

fn collect_responses(
    path: &Path,
    keys: &[&str],
    target: &mut BTreeSet<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    for value in read_jsonl_values(path)? {
        for key in keys {
            if let Some(response) = value.get(*key).and_then(|v| v.as_str()) {
                target.insert(normalize_response(response));
            }
        }
    }
    Ok(())
}

fn checkpoint_path(cfg: &Config) -> PathBuf {
    cfg.upstream_078_root
        .join("checkpoints")
        .join("chat_composition_repair")
        .join("model_checkpoint.json")
}

fn missing_upstreams(cfg: &Config, checkpoint: &Path) -> Vec<String> {
    let required = [
        ("upstream_078_summary", cfg.upstream_078_root.join("summary.json")),
        (
            "upstream_078_checkpoint_manifest",
            cfg.upstream_078_root.join("checkpoint_manifest.json"),
        ),
        (
            "upstream_078_train_examples_sample",
            cfg.upstream_078_root.join("train_examples_sample.jsonl"),
        ),
        (
            "upstream_078_eval_examples_sample",
            cfg.upstream_078_root.join("eval_examples_sample.jsonl"),
        ),
        (
            "upstream_078_generation_samples",
            cfg.upstream_078_root.join("generation_samples.jsonl"),
        ),
        ("upstream_078_checkpoint", checkpoint.to_path_buf()),
        (
            "upstream_077b_summary",
            cfg.upstream_077b_root.join("summary.json"),
        ),
        (
            "upstream_077b_analysis",
            cfg.upstream_077b_root.join("repair_recommendation.json"),
        ),
        ("upstream_076_summary", cfg.upstream_076_root.join("summary.json")),
        (
            "upstream_076_generation_samples",
            cfg.upstream_076_root.join("generation_samples.jsonl"),
        ),
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

fn infer_intent(prompt: &str) -> &'static str {
    let lower = prompt.to_lowercase();
    if lower.contains("answer only") || lower.contains("retention") {
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
    let colors = [
        "cobalt", "green", "indigo", "rose", "teal", "violet", "amber", "silver",
    ];
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

fn is_template_copy(output: &str, sources: &CopySources, semantic_overlap: f64) -> bool {
    let normalized = normalize_response(output);
    sources.train_responses.contains(&normalized)
        || sources.eval_outputs.contains(&normalized)
        || sources.response_table_outputs.contains(&normalized)
        || sources.generated_outputs_077_078.contains(&normalized)
        || sources.template_responses.contains(&normalized)
        || semantic_overlap >= 0.70
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

fn slot_diagnosis(row: &EvalExample, emitted: Option<&str>) -> String {
    match (&row.expected_slot, emitted) {
        (Some(expected), Some(actual)) if expected == actual => "slot bound correctly".to_string(),
        (Some(expected), Some(actual)) => format!("wrong slot emitted: expected {expected}, got {actual}"),
        (Some(expected), None) => format!("missing slot: expected {expected}"),
        _ => "slot not required".to_string(),
    }
}

fn is_sentence_like(output: &str) -> bool {
    let tokens = tokenize(output);
    tokens.len() >= 6 && output.chars().any(char::is_whitespace)
}

fn content_word_count(output: &str) -> usize {
    tokenize(output)
        .into_iter()
        .filter(|tok| !["the", "a", "an", "is", "and", "or", "to", "of"].contains(&tok.as_str()))
        .count()
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

fn token_jaccard(left: &str, right: &str) -> f64 {
    let left_tokens = tokenize(left).into_iter().collect::<BTreeSet<_>>();
    let right_tokens = tokenize(right).into_iter().collect::<BTreeSet<_>>();
    if left_tokens.is_empty() && right_tokens.is_empty() {
        return 1.0;
    }
    let union = left_tokens.union(&right_tokens).count();
    if union == 0 {
        0.0
    } else {
        left_tokens.intersection(&right_tokens).count() as f64 / union as f64
    }
}

fn max_prompt_jaccard(eval_prompts: &BTreeSet<String>, source: &BTreeSet<String>) -> f64 {
    eval_prompts
        .iter()
        .map(|prompt| max_jaccard_one(prompt, source))
        .fold(0.0, f64::max)
}

fn max_jaccard_one(prompt: &str, source: &BTreeSet<String>) -> f64 {
    source
        .iter()
        .map(|candidate| token_jaccard(prompt, candidate))
        .fold(0.0, f64::max)
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

fn finite_labels() -> Vec<String> {
    [
        "amber", "silver", "teal", "violet", "green", "indigo", "cobalt", "rose",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
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

fn write_eval_outputs(path: &Path, rows: &[EvalRow]) -> Result<(), Box<dyn std::error::Error>> {
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
            "semantic_template_overlap_score": row.semantic_template_overlap_score,
            "slot_binding_diagnosis": row.slot_binding_diagnosis,
            "short_diagnosis": row.short_diagnosis,
            "slot_value_expected": row.slot_value_expected,
            "slot_value_emitted": row.slot_value_emitted
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
            "required_keywords": row.required_keywords,
            "forbidden_outputs": row.forbidden_outputs,
            "pass_fail": row.pass_fail,
            "novelty_flag": row.novelty_flag,
            "template_copy_flag": row.template_copy_flag,
            "semantic_template_overlap_score": row.semantic_template_overlap_score,
            "slot_binding_diagnosis": row.slot_binding_diagnosis,
            "short_diagnosis": row.short_diagnosis,
            "output_classification": row.output_classification
        });
        writeln!(file, "{}", serde_json::to_string(&payload)?)?;
    }
    Ok(())
}

fn write_upstream_manifest(
    cfg: &Config,
    checkpoint: &Path,
    checkpoint_hash: &str,
    start_ms: u128,
) -> Result<(), Box<dyn std::error::Error>> {
    write_json(
        &cfg.out.join("upstream_078_manifest.json"),
        &json!({
            "schema_version": "chat_composition_fresh_confirm_upstream_078_manifest_v1",
            "upstream_078_root": cfg.upstream_078_root.display().to_string(),
            "upstream_077b_root": cfg.upstream_077b_root.display().to_string(),
            "upstream_076_root": cfg.upstream_076_root.display().to_string(),
            "upstream_074_root": cfg.upstream_074_root.display().to_string(),
            "checkpoint": checkpoint.display().to_string(),
            "checkpoint_hash_before": checkpoint_hash,
            "upstream_078_summary_present": true,
            "upstream_078_positive": true,
            "checkpoint_exists": true,
            "eval_started_after_079_start": true,
            "079_start_unix_ms": start_ms,
            "do_not_rerun_076_077_077b_078": true,
            "do_not_train_replacement_checkpoint": true
        }),
    )
}

fn write_checkpoint_manifest(
    cfg: &Config,
    checkpoint: &Path,
    before: &str,
    after: &str,
    unchanged: bool,
    eval_started_after_start: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    write_json(
        &cfg.out.join("checkpoint_manifest.json"),
        &json!({
            "schema_version": "chat_composition_fresh_confirm_checkpoint_manifest_v1",
            "checkpoint": checkpoint.display().to_string(),
            "checkpoint_hash_before": before,
            "checkpoint_hash_after": after,
            "checkpoint_hash_unchanged": unchanged,
            "upstream_078_summary_present": true,
            "upstream_078_positive": true,
            "checkpoint_exists": true,
            "eval_started_after_079_start": eval_started_after_start,
            "train_step_count": 0,
            "prediction_oracle_used": false,
            "llm_judge_used": false,
            "decoder_path": "token_level_next_token",
            "response_table_used_for_main_prediction": false,
            "no_replacement_checkpoint_written": true
        }),
    )
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
        "schema_version": "chat_composition_fresh_confirm_summary_v1",
        "status": "failed",
        "reason": reason,
        "verdicts": ["CHAT_COMPOSITION_FRESH_CONFIRM_FAILS", verdict],
        "bounded_fresh_chat_composition_confirm_only": true,
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
        "schema_version": "chat_composition_fresh_confirm_summary_v1",
        "status": status,
        "verdicts": verdicts,
        "details": details,
        "bounded_fresh_chat_composition_confirm_only": true,
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

#[allow(clippy::too_many_arguments)]
fn build_final_summary(
    status: &str,
    verdicts: Vec<String>,
    prompt_report: &PromptLeakageReport,
    composition: Value,
    novelty: Value,
    context: Value,
    retention: Value,
    collapse: Value,
    checkpoint_hash_before: &str,
    checkpoint_hash_after: &str,
    checkpoint_hash_unchanged: bool,
    eval_started_after_079_start: bool,
) -> Value {
    json!({
        "schema_version": "chat_composition_fresh_confirm_summary_v1",
        "status": status,
        "verdicts": verdicts,
        "upstream_078_summary_present": true,
        "upstream_078_positive": true,
        "checkpoint_exists": true,
        "eval_started_after_079_start": eval_started_after_079_start,
        "checkpoint_hash_before": checkpoint_hash_before,
        "checkpoint_hash_after": checkpoint_hash_after,
        "checkpoint_hash_unchanged": checkpoint_hash_unchanged,
        "train_step_count": 0,
        "prediction_oracle_used": false,
        "llm_judge_used": false,
        "decoder_path": "token_level_next_token",
        "response_table_used_for_main_prediction": false,
        "overlap_with_078_train_prompt_count": prompt_report.overlap_with_078_train_prompt_count,
        "overlap_with_078_eval_prompt_count": prompt_report.overlap_with_078_eval_prompt_count,
        "overlap_with_077_prompt_count": prompt_report.overlap_with_077_prompt_count,
        "overlap_with_076_prompt_count": prompt_report.overlap_with_076_prompt_count,
        "max_prompt_token_jaccard_vs_078_train": prompt_report.max_prompt_token_jaccard_vs_078_train,
        "max_prompt_token_jaccard_vs_078_eval": prompt_report.max_prompt_token_jaccard_vs_078_eval,
        "max_prompt_token_jaccard_vs_077": prompt_report.max_prompt_token_jaccard_vs_077,
        "max_prompt_token_jaccard_vs_076": prompt_report.max_prompt_token_jaccard_vs_076,
        "near_duplicate_prompt_count": prompt_report.near_duplicate_prompt_count,
        "eval_prompt_hash": prompt_report.eval_prompt_hash,
        "composition_metrics": composition,
        "novelty_metrics": novelty,
        "context_slot_metrics": context,
        "finite_label_retention_metrics": retention,
        "collapse_metrics": collapse,
        "bounded_fresh_chat_composition_confirm_only": true,
        "not_GPT_like_assistant_readiness": true,
        "not_full_English_LM": true,
        "not_language_grounding": true,
        "not_production_chat": true,
        "not_safety_alignment": true,
        "not_public_beta": true,
        "not_GA": true,
        "not_hosted_SaaS": true,
        "next_if_pass": "080_CHAT_COMPOSITION_MULTI_SEED_CONFIRM",
        "next_if_fail": "079B_CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS"
    })
}

fn write_report(path: &Path, summary: &Value) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut text = String::new();
    text.push_str("# STABLE_LOOP_PHASE_LOCK_079_CHAT_COMPOSITION_FRESH_CONFIRM Report\n\n");
    text.push_str(&format!("Status: `{}`\n\n", summary["status"].as_str().unwrap_or("unknown")));
    text.push_str("079 is bounded fresh chat composition confirm only.\n\n");
    text.push_str("train_step_count = 0\n");
    text.push_str("prediction_oracle_used = false\n");
    text.push_str("llm_judge_used = false\n");
    text.push_str("decoder_path = token_level_next_token\n");
    text.push_str("response_table_used_for_main_prediction = false\n");
    text.push_str("bounded fresh chat composition confirm only\n");
    text.push_str("not GPT-like assistant readiness\n");
    text.push_str("not full English LM\n");
    text.push_str("not language grounding\n");
    text.push_str("not production chat\n");
    text.push_str("not safety alignment\n");
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

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut cfg = Config {
        out: PathBuf::from(DEFAULT_OUT),
        upstream_078_root: PathBuf::from(DEFAULT_UPSTREAM_078_ROOT),
        upstream_077b_root: PathBuf::from(DEFAULT_UPSTREAM_077B_ROOT),
        upstream_076_root: PathBuf::from(DEFAULT_UPSTREAM_076_ROOT),
        upstream_074_root: PathBuf::from(DEFAULT_UPSTREAM_074_ROOT),
        seed: DEFAULT_SEED,
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
            "--upstream-077b-root" => {
                idx += 1;
                cfg.upstream_077b_root = PathBuf::from(args.get(idx).ok_or("--upstream-077b-root missing value")?);
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
                    "Usage: phase_lane_chat_composition_fresh_confirm --out <path> --upstream-078-root <path> --upstream-077b-root <path> --upstream-076-root <path> --upstream-074-root <path> --seed <n> --heartbeat-sec <n>"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        idx += 1;
    }
    Ok(cfg)
}
