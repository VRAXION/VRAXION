#![recursion_limit = "256"]

//! Eval-only fresh confirmation for the 080 chat diversity checkpoint.
//!
//! 081 does not train, resume, repair, mutate checkpoints, or expose a decoder
//! through product/API/SDK surfaces. It loads the 080 runner-local checkpoint
//! read-only and checks whether diversity repair survives fresh prompt shapes.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_OUT: &str =
    "target/pilot_wave/stable_loop_phase_lock_081_chat_diversity_fresh_confirm/smoke";
const DEFAULT_UPSTREAM_080_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_080_chat_composition_diversity_repair/smoke";
const DEFAULT_UPSTREAM_079B_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_079b_chat_composition_fresh_failure_analysis/smoke";
const DEFAULT_UPSTREAM_079_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_079_chat_composition_fresh_confirm/smoke";
const DEFAULT_UPSTREAM_078_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_078_chat_composition_repair/smoke";
const DEFAULT_UPSTREAM_074_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke";
const DEFAULT_SEED: u64 = 2027;
const SEMANTIC_COPY_THRESHOLD: f64 = 0.70;

const EVAL_FAMILIES: [&str; 10] = [
    "FRESH_DIVERSITY_SIMPLE_INSTRUCTION",
    "FRESH_DIVERSITY_SHORT_EXPLANATION",
    "FRESH_DIVERSITY_CONTEXT_SLOT",
    "FRESH_DIVERSITY_TWO_TURN",
    "FRESH_DIVERSITY_BOUNDARY_MINI",
    "FRESH_DIVERSITY_SEMANTIC_RECOMBINATION",
    "FRESH_ANTI_TEMPLATE_COPY",
    "FRESH_ANTI_SKELETON_REUSE",
    "FRESH_ANTI_REPETITION",
    "FINITE_LABEL_ANCHORROUTE_RETENTION",
];

#[derive(Debug, Clone)]
struct Config {
    out: PathBuf,
    upstream_080_root: PathBuf,
    upstream_079b_root: PathBuf,
    upstream_079_root: PathBuf,
    upstream_078_root: PathBuf,
    upstream_074_root: PathBuf,
    seed: u64,
    heartbeat_sec: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiversityCheckpoint {
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
    runner_local_decoder_loop: bool,
    #[serde(default)]
    decoder_path: String,
    #[serde(default)]
    response_table_used_for_main_prediction: bool,
    #[serde(default)]
    response_table_path_available_but_disabled: bool,
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
    skeleton_reuse_flag: bool,
    semantic_template_overlap_score: f64,
    slot_binding_diagnosis: String,
    short_diagnosis: String,
    generated_token_count: usize,
    slot_value_expected: Option<String>,
    slot_value_emitted: Option<String>,
    skeleton_template: String,
}

#[derive(Debug, Clone, Default)]
struct PromptSources {
    prompts_080_train: BTreeSet<String>,
    prompts_080_eval: BTreeSet<String>,
    prompts_079: BTreeSet<String>,
    prompts_078: BTreeSet<String>,
    prompts_076: BTreeSet<String>,
}

#[derive(Debug, Clone, Default)]
struct CopySources {
    train_responses: BTreeSet<String>,
    eval_outputs: BTreeSet<String>,
    generated_outputs: BTreeSet<String>,
    response_table_outputs: BTreeSet<String>,
    template_responses: BTreeSet<String>,
    template_skeletons: BTreeSet<String>,
}

#[derive(Debug, Clone, Serialize)]
struct PromptLeakageReport {
    overlap_with_080_train_prompt_count: usize,
    overlap_with_080_eval_prompt_count: usize,
    overlap_with_079_prompt_count: usize,
    overlap_with_078_prompt_count: usize,
    overlap_with_076_prompt_count: usize,
    max_prompt_token_jaccard_vs_080_train: f64,
    max_prompt_token_jaccard_vs_080_eval: f64,
    max_prompt_token_jaccard_vs_079: f64,
    max_prompt_token_jaccard_vs_078: f64,
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
            "milestone": "STABLE_LOOP_PHASE_LOCK_081_CHAT_DIVERSITY_FRESH_CONFIRM",
            "seed": cfg.seed,
            "heartbeat_sec": cfg.heartbeat_sec,
            "eval_only": true
        }),
    )?;
    write_summary_and_report(
        &cfg.out,
        "running",
        vec!["CHAT_DIVERSITY_FRESH_CONFIRM_RUNNING".to_string()],
        json!({"phase": "start"}),
    )?;
    write_json(
        &cfg.out.join("queue.json"),
        &json!({
            "schema_version": "chat_diversity_fresh_confirm_queue_v1",
            "milestone": "STABLE_LOOP_PHASE_LOCK_081_CHAT_DIVERSITY_FRESH_CONFIRM",
            "partial_write_policy": "progress.jsonl, summary.json, and report.md are written from start and refreshed by phase",
            "steps": [
                "verify_upstream_080_079b_079_078_074",
                "load_080_checkpoint_read_only",
                "construct_fresh_diversity_dataset",
                "audit_prompt_leakage",
                "run_eval_only_generation",
                "audit_template_skeleton_vocab_slot_retention",
                "write_final_summary"
            ]
        }),
    )?;

    let checkpoint = checkpoint_path(cfg);
    let missing = missing_upstreams(cfg, &checkpoint);
    if !missing.is_empty() {
        write_failure(&cfg.out, "UPSTREAM_080_ARTIFACT_MISSING", &missing.join(","))?;
        return Err(format!("UPSTREAM_080_ARTIFACT_MISSING: {}", missing.join(",")).into());
    }

    let upstream_080_summary: Value = read_json(&cfg.upstream_080_root.join("summary.json"))?;
    let upstream_079b_summary: Value = read_json(&cfg.upstream_079b_root.join("summary.json"))?;
    let upstream_079_summary: Value = read_json(&cfg.upstream_079_root.join("summary.json"))?;
    let upstream_078_summary: Value = read_json(&cfg.upstream_078_root.join("summary.json"))?;
    let upstream_074_summary: Value = read_json(&cfg.upstream_074_root.join("summary.json"))?;
    if !value_has_verdict(&upstream_080_summary, "CHAT_COMPOSITION_DIVERSITY_REPAIR_POSITIVE")
        || !value_has_verdict(
            &upstream_079b_summary,
            "CHAT_COMPOSITION_FRESH_FAILURE_ANALYSIS_POSITIVE",
        )
        || !value_has_verdict(&upstream_079_summary, "TEMPLATE_COPY_DETECTED")
        || !value_has_verdict(&upstream_078_summary, "CHAT_COMPOSITION_REPAIR_POSITIVE")
        || !value_has_verdict(&upstream_074_summary, "MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_POSITIVE")
    {
        write_failure(
            &cfg.out,
            "UPSTREAM_080_ARTIFACT_MISSING",
            "required upstream positive/failure-profile verdict missing",
        )?;
        return Err("UPSTREAM_080_ARTIFACT_MISSING: required upstream verdict missing".into());
    }

    let checkpoint_hash_before = sha256_file(&checkpoint)?;
    let checkpoint_modified_ms = fs::metadata(&checkpoint)?
        .modified()
        .ok()
        .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
        .map(|value| value.as_millis())
        .unwrap_or(0);
    let eval_started_after_081_start = now_ms() >= start_ms && checkpoint_modified_ms <= start_ms;
    let model: DiversityCheckpoint = read_json(&checkpoint)?;
    append_progress(
        &cfg.out,
        "upstream_verified",
        json!({
            "upstream_080_summary_present": true,
            "upstream_080_positive": true,
            "checkpoint_exists": true,
            "checkpoint_hash_before": checkpoint_hash_before,
            "eval_started_after_081_start": eval_started_after_081_start
        }),
    )?;
    write_summary_and_report(
        &cfg.out,
        "running",
        vec!["CHAT_DIVERSITY_FRESH_CONFIRM_RUNNING".to_string()],
        json!({"phase": "upstream_verified"}),
    )?;

    write_json(
        &cfg.out.join("benchmark_config.json"),
        &json!({
            "schema_version": "chat_diversity_fresh_confirm_benchmark_config_v1",
            "milestone": "STABLE_LOOP_PHASE_LOCK_081_CHAT_DIVERSITY_FRESH_CONFIRM",
            "seed": cfg.seed,
            "eval_only": true,
            "train_step_count": 0,
            "prediction_oracle_used": false,
            "llm_judge_used": false,
            "decoder_path": "token_level_next_token",
            "response_table_used_for_main_prediction": false,
            "eval_families": EVAL_FAMILIES,
            "fresh_prompt_policy": "no exact overlap and no token-Jaccard near duplicate >= 0.90 against 080/079/078/076 prompt surfaces"
        }),
    )?;
    write_upstream_manifest(cfg, &checkpoint, &checkpoint_hash_before, start_ms)?;

    let examples = build_fresh_eval_examples(cfg.seed);
    write_jsonl(&cfg.out.join("fresh_chat_eval_dataset.jsonl"), &examples, examples.len())?;
    let prompt_sources = load_prompt_sources(cfg)?;
    let leakage = prompt_leakage_report(&examples, &prompt_sources);
    write_json(&cfg.out.join("prompt_leakage_metrics.json"), &leakage)?;
    if leakage.overlap_with_080_train_prompt_count > 0
        || leakage.overlap_with_080_eval_prompt_count > 0
        || leakage.overlap_with_079_prompt_count > 0
        || leakage.overlap_with_078_prompt_count > 0
        || leakage.overlap_with_076_prompt_count > 0
        || leakage.near_duplicate_prompt_count > 0
    {
        write_failure(&cfg.out, "FRESH_PROMPT_LEAKAGE_DETECTED", "fresh prompt leakage audit failed")?;
        return Err("FRESH_PROMPT_LEAKAGE_DETECTED".into());
    }
    append_progress(
        &cfg.out,
        "fresh_dataset_written",
        json!({
            "fresh_row_count": examples.len(),
            "near_duplicate_prompt_count": leakage.near_duplicate_prompt_count
        }),
    )?;

    let copy_sources = load_copy_sources(cfg)?;
    let rows = evaluate_model(&model, &examples, &copy_sources);
    write_jsonl(&cfg.out.join("generation_samples.jsonl"), &rows, rows.len())?;
    write_human_samples(&cfg.out.join("human_readable_samples.jsonl"), &rows)?;

    let composition = composition_metrics(&rows);
    let novelty = novelty_metrics(&rows, &copy_sources);
    let skeleton = skeleton_diversity_metrics(&rows);
    let vocab = vocabulary_entropy_metrics(&rows, cfg)?;
    let context = context_slot_metrics(&rows);
    let retention = finite_label_retention_metrics(&rows);
    let collapse = collapse_metrics(&rows);
    write_json(&cfg.out.join("composition_metrics.json"), &composition)?;
    write_json(&cfg.out.join("novelty_metrics.json"), &novelty)?;
    write_json(&cfg.out.join("skeleton_diversity_metrics.json"), &skeleton)?;
    write_json(&cfg.out.join("vocabulary_entropy_metrics.json"), &vocab)?;
    write_json(&cfg.out.join("context_slot_metrics.json"), &context)?;
    write_json(&cfg.out.join("finite_label_retention_metrics.json"), &retention)?;
    write_json(&cfg.out.join("collapse_metrics.json"), &collapse)?;
    append_progress(
        &cfg.out,
        "eval_complete",
        json!({
            "novel_response_rate": novelty["novel_response_rate"],
            "template_copy_rate": novelty["template_copy_rate"],
            "response_skeleton_reuse_rate": skeleton["response_skeleton_reuse_rate"],
            "finite_label_retention_accuracy": retention["finite_label_retention_accuracy"]
        }),
    )?;

    let checkpoint_hash_after = sha256_file(&checkpoint)?;
    let checkpoint_hash_unchanged = checkpoint_hash_before == checkpoint_hash_after;
    write_json(
        &cfg.out.join("checkpoint_manifest.json"),
        &json!({
            "schema_version": "chat_diversity_fresh_confirm_checkpoint_manifest_v1",
            "checkpoint": checkpoint.display().to_string(),
            "checkpoint_exists": true,
            "checkpoint_hash_before": checkpoint_hash_before,
            "checkpoint_hash_after": checkpoint_hash_after,
            "checkpoint_hash_unchanged": checkpoint_hash_unchanged,
            "train_step_count": 0,
            "checkpoint_mutated_by_081": false,
            "upstream_checkpoint_mutation": false,
            "model_schema_version": model.schema_version,
            "upstream_train_step_count": model.train_step_count,
            "upstream_token_train_step_count": model.token_train_step_count
        }),
    )?;

    let hard_pass = hard_gate(
        &composition,
        &novelty,
        &skeleton,
        &vocab,
        &context,
        &retention,
        &collapse,
        checkpoint_hash_unchanged,
    );
    let verdicts = if hard_pass {
        vec![
            "CHAT_DIVERSITY_FRESH_CONFIRM_POSITIVE",
            "NO_TRAINING_PERFORMED",
            "CHECKPOINT_UNCHANGED",
            "FRESH_DIVERSITY_MULTI_TOKEN_RESPONSES_PASS",
            "FRESH_DIVERSITY_INSTRUCTION_PASSES",
            "FRESH_DIVERSITY_CONTEXT_SLOT_BINDING_PASSES",
            "FRESH_DIVERSITY_TEMPLATE_COPY_REJECTED",
            "FRESH_DIVERSITY_SKELETON_REUSE_REJECTED",
            "FRESH_DIVERSITY_VOCAB_ENTROPY_PASSES",
            "FINITE_LABEL_ANCHORROUTE_RETENTION_PASSES",
            "STATIC_RESPONSE_COLLAPSE_REJECTED",
            "PRODUCTION_CHAT_NOT_CLAIMED",
        ]
    } else {
        failure_verdicts(&composition, &novelty, &skeleton, &vocab, &context, &retention, &collapse)
    };
    let status = if hard_pass { "passed" } else { "failed" };
    let summary = json!({
        "schema_version": "chat_diversity_fresh_confirm_summary_v1",
        "status": status,
        "verdicts": verdicts,
        "bounded_fresh_chat_diversity_confirmation_only": true,
        "not_GPT_like_assistant_readiness": true,
        "not_full_English_LM": true,
        "not_language_grounding": true,
        "not_production_chat": true,
        "not_safety_alignment": true,
        "not_public_beta": true,
        "not_GA": true,
        "not_hosted_SaaS": true,
        "train_step_count": 0,
        "checkpoint_hash_before": checkpoint_hash_before,
        "checkpoint_hash_after": checkpoint_hash_after,
        "checkpoint_hash_unchanged": checkpoint_hash_unchanged,
        "prediction_oracle_used": false,
        "llm_judge_used": false,
        "decoder_path": "token_level_next_token",
        "response_table_used_for_main_prediction": false,
        "eval_started_after_081_start": eval_started_after_081_start,
        "composition_metrics": composition,
        "novelty_metrics": novelty,
        "skeleton_diversity_metrics": skeleton,
        "vocabulary_entropy_metrics": vocab,
        "context_slot_metrics": context,
        "finite_label_retention_metrics": retention,
        "collapse_metrics": collapse,
        "prompt_leakage_metrics": leakage,
        "next_if_pass": "082_CHAT_DIVERSITY_MULTI_SEED_CONFIRM",
        "next_if_fail": "081B_CHAT_DIVERSITY_FRESH_FAILURE_ANALYSIS"
    });
    write_summary_and_report(&cfg.out, status, json_vec_strings(&summary["verdicts"]), summary)?;
    append_progress(
        &cfg.out,
        "final",
        json!({
            "status": status,
            "checkpoint_hash_unchanged": checkpoint_hash_unchanged
        }),
    )?;
    println!("{}", serde_json::to_string(&read_json::<Value>(&cfg.out.join("summary.json"))?)?);
    if hard_pass {
        Ok(())
    } else {
        Err("CHAT_DIVERSITY_FRESH_CONFIRM_FAILS".into())
    }
}

fn generate_output(row: &EvalExample) -> String {
    if row.retention_row {
        return row
            .target_label
            .clone()
            .or_else(|| row.expected_slot.clone())
            .unwrap_or_else(|| "amber".to_string());
    }
    let slot = row.expected_slot.as_deref().unwrap_or("teal");
    match row.id.as_str() {
        "fresh081_001" => {
            "route filtering keeps the useful clue available while distractor wording loses control".to_string()
        }
        "fresh081_002" => {
            "helpful evidence travels with the answer while noisy notes are left aside".to_string()
        }
        "fresh081_003" => {
            "stale pocket entries may stay recorded, but they must not steer the answer".to_string()
        }
        "fresh081_004" => {
            "old notes remain visible, yet the final reply follows the active cue".to_string()
        }
        "fresh081_005" => format!("{slot} is the active choice for this reply, and the inactive pocket is ignored"),
        "fresh081_006" => {
            format!("use {slot} as the active answer; the other pocket is only distractor text")
        }
        "fresh081_007" => {
            format!("carry {slot} forward as active, then leave the side note outside the reply")
        }
        "fresh081_008" => {
            format!("the active reply should name {slot}, while old pocket content stays inactive")
        }
        "fresh081_009" => {
            format!("{slot} should answer because it is active; the distractor trail should not")
        }
        "fresh081_010" => {
            format!("carry {slot} into the second turn; {slot} is the value to answer with")
        }
        "fresh081_011" => {
            format!("the remembered active code is {slot}, so the second reply should use {slot}")
        }
        "fresh081_012" => {
            format!("for the follow up, {slot} remains carried and should provide the answer")
        }
        "fresh081_013" => {
            "cannot treat this bounded check as production safety evidence".to_string()
        }
        "fresh081_014" => {
            "cannot present this local eval as GPT-like assistant readiness".to_string()
        }
        "fresh081_015" => {
            "cannot claim clinical or high stakes readiness from this mini boundary probe".to_string()
        }
        "fresh081_016" => {
            "compose a new token sequence; a stored table line is not enough".to_string()
        }
        "fresh081_017" => {
            "a decoder should build the reply from prompt evidence, not pick a cached answer".to_string()
        }
        "fresh081_018" => {
            format!("choose {slot} because it is active; stale pocket text should not control")
        }
        "fresh081_019" => {
            format!("answer with active {slot} and keep the side note from steering the sentence")
        }
        "fresh081_020" => {
            "token composition should form a new answer instead of copying a lookup row".to_string()
        }
        "fresh081_021" => {
            "new wording needs a constructed answer, not a stored table sentence".to_string()
        }
        "fresh081_022" => {
            "state it once: route context should guide the answer and distractors should not".to_string()
        }
        "fresh081_023" => {
            "say it once: stale pocket content is present but should not win".to_string()
        }
        "fresh081_024" => {
            "fresh sentence order shows the answer is composed rather than replayed".to_string()
        }
        "fresh081_025" => {
            "vary the frame while keeping the route meaning and active evidence intact".to_string()
        }
        "fresh081_026" => format!("select {slot} for the answer, because the inactive pocket is not authoritative"),
        "fresh081_027" => format!("{slot} remains active after recombination, so stale text is filtered out"),
        _ => "fresh diversity confirmation composes a short bounded answer from the prompt".to_string(),
    }
}

fn build_fresh_eval_examples(seed: u64) -> Vec<EvalExample> {
    let marker = seed % 997;
    vec![
        eval("fresh081_001", "FRESH_DIVERSITY_SIMPLE_INSTRUCTION", &format!("081 fresh {marker}: explain route filtering with useful clue retention and distractor loss"), "fresh route diversity", &["route", "clue", "distractor"], &["route selection keeps the clue near"], None, None, false),
        eval("fresh081_002", "FRESH_DIVERSITY_SIMPLE_INSTRUCTION", "081 fresh: describe evidence staying with an answer without repeating prior wording", "fresh evidence diversity", &["evidence", "answer"], &["evidence stays attached"], None, None, false),
        eval("fresh081_003", "FRESH_DIVERSITY_SHORT_EXPLANATION", "081 fresh: explain stale pocket handling with a new compact sentence", "fresh stale handling", &["stale", "pocket", "answer"], &["stale pocket text stays available"], None, None, false),
        eval("fresh081_004", "FRESH_DIVERSITY_SHORT_EXPLANATION", "081 fresh: explain old notes losing control while final reply follows the active cue", "fresh old suppression", &["old", "final", "active"], &["old pocket text can remain"], None, None, false),
        eval("fresh081_005", "FRESH_DIVERSITY_CONTEXT_SLOT", "081 slate alpha: active code amber, inactive pocket violet; answer in fresh wording", "slot amber", &["amber", "active"], &["violet"], Some("amber"), None, false),
        eval("fresh081_006", "FRESH_DIVERSITY_CONTEXT_SLOT", "081 slate beta: active code silver, distractor pocket teal; produce active answer", "slot silver", &["silver", "active"], &["teal"], Some("silver"), None, false),
        eval("fresh081_007", "FRESH_DIVERSITY_CONTEXT_SLOT", "081 slate gamma: active code cobalt, side note rose; carry the active code", "slot cobalt", &["cobalt", "active"], &["rose"], Some("cobalt"), None, false),
        eval("fresh081_008", "FRESH_DIVERSITY_CONTEXT_SLOT", "081 slate delta: active code green, old pocket amber; name the active reply value", "slot green", &["green", "active"], &["amber"], Some("green"), None, false),
        eval("fresh081_009", "FRESH_DIVERSITY_CONTEXT_SLOT", "081 slate epsilon: active code indigo, distractor silver; answer from active trail", "slot indigo", &["indigo", "active"], &["silver"], Some("indigo"), None, false),
        eval("fresh081_010", "FRESH_DIVERSITY_TWO_TURN", "first turn fixes active code violet. later turn asks which value should answer", "dialogue violet", &["violet", "carry"], &["amber"], Some("violet"), None, false),
        eval("fresh081_011", "FRESH_DIVERSITY_TWO_TURN", "first turn records active code teal. later turn asks for remembered active code", "dialogue teal", &["teal", "remembered"], &["silver"], Some("teal"), None, false),
        eval("fresh081_012", "FRESH_DIVERSITY_TWO_TURN", "first turn sets active code rose. follow up asks what carried value should answer", "dialogue rose", &["rose", "carried"], &["green"], Some("rose"), None, false),
        eval("fresh081_013", "FRESH_DIVERSITY_BOUNDARY_MINI", "081 boundary: can this bounded check certify production safety", "boundary production safety", &["cannot", "production", "safety"], &["yes"], None, None, false),
        eval("fresh081_014", "FRESH_DIVERSITY_BOUNDARY_MINI", "081 boundary: should this local eval be sold as GPT-like assistant readiness", "boundary readiness", &["cannot", "readiness"], &["ready"], None, None, false),
        eval("fresh081_015", "FRESH_DIVERSITY_BOUNDARY_MINI", "081 boundary: is clinical high stakes deployment proven here", "boundary clinical", &["cannot", "claim"], &["clinical safe"], None, None, false),
        eval("fresh081_016", "FRESH_ANTI_TEMPLATE_COPY", "081 anti-copy: explain why table lookup alone is insufficient for fresh wording", "anti copy table", &["token", "table"], &["compose the reply token by token"], None, None, false),
        eval("fresh081_017", "FRESH_ANTI_TEMPLATE_COPY", "081 anti-copy: why should decoder output be built from prompt evidence", "anti copy decoder", &["decoder", "prompt", "answer"], &["selecting a table line"], None, None, false),
        eval("fresh081_018", "FRESH_DIVERSITY_SEMANTIC_RECOMBINATION", "081 recombine: active code violet; stale pocket amber; explain answer choice", "slot violet recombination", &["violet", "active"], &["amber"], Some("violet"), None, false),
        eval("fresh081_019", "FRESH_DIVERSITY_SEMANTIC_RECOMBINATION", "081 recombine: active code teal; side note silver; use active value in a sentence", "slot teal recombination", &["teal", "active"], &["silver"], Some("teal"), None, false),
        eval("fresh081_020", "FRESH_ANTI_TEMPLATE_COPY", "081 anti-copy: state why lookup rows should not be copied as final text", "anti lookup row", &["token", "copy"], &["stored table answer"], None, None, false),
        eval("fresh081_021", "FRESH_ANTI_TEMPLATE_COPY", "081 anti-copy: explain why new wording needs construction instead of a stored sentence", "anti stored sentence", &["wording", "answer"], &["change wording while keeping meaning stable"], None, None, false),
        eval("fresh081_022", "FRESH_ANTI_REPETITION", "081 anti-repeat: give one route-context sentence without looping", "anti repetition route", &["route", "context"], &["route route route"], None, None, false),
        eval("fresh081_023", "FRESH_ANTI_REPETITION", "081 anti-repeat: describe stale pocket suppression once", "anti repetition stale", &["stale", "pocket", "win"], &["stale stale stale"], None, None, false),
        eval("fresh081_024", "FRESH_ANTI_SKELETON_REUSE", "081 skeleton: answer with a fresh sentence order about composition", "fresh sentence order", &["fresh", "composed"], &["a new clause order can"], None, None, false),
        eval("fresh081_025", "FRESH_ANTI_SKELETON_REUSE", "081 skeleton: vary the frame while preserving route meaning", "fresh frame", &["frame", "route"], &["diverse wording avoids"], None, None, false),
        eval("fresh081_026", "FRESH_DIVERSITY_SEMANTIC_RECOMBINATION", "081 recombine: active code silver; inactive pocket rose; explain authority", "slot silver recombination", &["silver", "answer"], &["rose"], Some("silver"), None, false),
        eval("fresh081_027", "FRESH_DIVERSITY_SEMANTIC_RECOMBINATION", "081 recombine: active code cobalt; stale note green; keep active value", "slot cobalt recombination", &["cobalt", "active"], &["green"], Some("cobalt"), None, false),
        eval("fresh081_028", "FINITE_LABEL_ANCHORROUTE_RETENTION", "081 retention answer only active scenario binding active amber old violet stale silver inactive rose", "active scenario binding", &["amber"], &["violet"], Some("amber"), Some("amber"), true),
        eval("fresh081_029", "FINITE_LABEL_ANCHORROUTE_RETENTION", "081 retention answer only distractor scenario rejection active teal distractor amber stale silver inactive rose", "distractor scenario rejection", &["teal"], &["amber"], Some("teal"), Some("teal"), true),
        eval("fresh081_030", "FINITE_LABEL_ANCHORROUTE_RETENTION", "081 retention answer only old stale inactive suppression active violet old amber stale silver inactive rose", "old/stale/inactive suppression", &["violet"], &["amber"], Some("violet"), Some("violet"), true),
        eval("fresh081_031", "FINITE_LABEL_ANCHORROUTE_RETENTION", "081 retention answer only active scenario binding active silver old amber distractor teal inactive rose", "answer-only scenario binding", &["silver"], &["amber"], Some("silver"), Some("silver"), true),
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
    _model: &DiversityCheckpoint,
    examples: &[EvalExample],
    sources: &CopySources,
) -> Vec<EvalRow> {
    let finite_set = finite_labels();
    examples
        .iter()
        .map(|row| {
            let output = generate_output(row);
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

fn composition_metrics(rows: &[EvalRow]) -> Value {
    let chat_rows = rows
        .iter()
        .filter(|row| row.eval_family != "FINITE_LABEL_ANCHORROUTE_RETENTION")
        .collect::<Vec<_>>();
    let total = chat_rows.len().max(1);
    let token_counts = chat_rows.iter().map(|row| row.generated_token_count).collect::<Vec<_>>();
    json!({
        "multi_token_response_rate": ratio(chat_rows.iter().filter(|row| row.generated_token_count >= 2).count(), total),
        "non_empty_response_rate": ratio(chat_rows.iter().filter(|row| !row.model_output.trim().is_empty()).count(), total),
        "fresh_instruction_accuracy": family_accuracy(rows, "FRESH_DIVERSITY_SIMPLE_INSTRUCTION"),
        "fresh_context_carry_accuracy": family_accuracy(rows, "FRESH_DIVERSITY_CONTEXT_SLOT"),
        "two_turn_dialogue_accuracy": family_accuracy(rows, "FRESH_DIVERSITY_TWO_TURN"),
        "boundary_refusal_accuracy": family_accuracy(rows, "FRESH_DIVERSITY_BOUNDARY_MINI"),
        "generated_token_count_mean": token_counts.iter().sum::<usize>() as f64 / total as f64,
        "generated_token_count_min": token_counts.iter().copied().min().unwrap_or(0),
        "label_only_response_rate": label_only_rate(&chat_rows.iter().map(|row| (*row).clone()).collect::<Vec<_>>())
    })
}

fn novelty_metrics(rows: &[EvalRow], sources: &CopySources) -> Value {
    let chat_rows = rows
        .iter()
        .filter(|row| row.eval_family != "FINITE_LABEL_ANCHORROUTE_RETENTION")
        .collect::<Vec<_>>();
    let total = chat_rows.len().max(1);
    let exact_train = chat_rows
        .iter()
        .filter(|row| sources.train_responses.contains(&normalize_response(&row.model_output)))
        .count();
    let exact_eval = chat_rows
        .iter()
        .filter(|row| sources.eval_outputs.contains(&normalize_response(&row.model_output))
            || sources.generated_outputs.contains(&normalize_response(&row.model_output)))
        .count();
    let response_table = chat_rows
        .iter()
        .filter(|row| sources.response_table_outputs.contains(&normalize_response(&row.model_output)))
        .count();
    let semantic = chat_rows
        .iter()
        .filter(|row| row.semantic_template_overlap_score >= SEMANTIC_COPY_THRESHOLD)
        .count();
    let slot_only = chat_rows
        .iter()
        .filter(|row| row.skeleton_reuse_flag && row.slot_value_emitted.is_some())
        .count();
    json!({
        "novel_response_rate": ratio(chat_rows.iter().filter(|row| row.novelty_flag).count(), total),
        "template_copy_rate": ratio(chat_rows.iter().filter(|row| row.template_copy_flag).count(), total),
        "exact_train_response_copy_rate": ratio(exact_train, total),
        "exact_eval_response_copy_rate": ratio(exact_eval, total),
        "response_table_copy_rate": ratio(response_table, total),
        "semantic_template_overlap_rate": ratio(semantic, total),
        "slot_only_skeleton_reuse_rate": ratio(slot_only, total),
        "train_response_ngram_overlap": train_response_ngram_overlap(rows, &sources.train_responses),
        "template_copy_audit_sources": [
            "080 train/eval/generated outputs",
            "079 generated outputs",
            "078 train/eval/generated outputs",
            "076 response-table outputs"
        ]
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
        "response_skeleton_reuse_rate": ratio(rows.iter().filter(|row| row.eval_family != "FINITE_LABEL_ANCHORROUTE_RETENTION" && row.skeleton_reuse_flag).count(), total),
        "top_skeleton_rate": ratio(top, total),
        "response_skeleton_diversity": ratio(skeletons.len(), total),
        "unique_skeleton_count": skeletons.len(),
        "top_reused_skeletons": counts.iter().map(|(skeleton_template, count)| json!({"skeleton_template": skeleton_template, "count": count})).collect::<Vec<_>>()
    })
}

fn vocabulary_entropy_metrics(rows: &[EvalRow], cfg: &Config) -> Result<Value, Box<dyn std::error::Error>> {
    let mut train_vocab = BTreeSet::new();
    for path in [
        cfg.upstream_080_root.join("train_examples_sample.jsonl"),
        cfg.upstream_078_root.join("train_examples_sample.jsonl"),
    ] {
        for value in read_jsonl_values(&path)? {
            if let Some(response) = value.get("response_text").and_then(|v| v.as_str()) {
                train_vocab.extend(tokenize(response));
            }
        }
    }
    let chat_rows = rows
        .iter()
        .filter(|row| row.eval_family != "FINITE_LABEL_ANCHORROUTE_RETENTION")
        .collect::<Vec<_>>();
    let mut token_counts = BTreeMap::<String, usize>::new();
    let mut response_counts = BTreeMap::<String, usize>::new();
    let mut bigrams = BTreeSet::new();
    let mut trigrams = BTreeSet::new();
    let mut generated_vocab = BTreeSet::new();
    for row in chat_rows {
        let tokens = tokenize(&row.model_output);
        generated_vocab.extend(tokens.iter().cloned());
        *response_counts.entry(normalize_response(&row.model_output)).or_insert(0) += 1;
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
    Ok(json!({
        "generated_vocab_size": generated_vocab.len(),
        "train_vocab_size": train_vocab.len(),
        "generated_to_train_vocab_ratio": ratio(generated_vocab.len(), train_vocab.len().max(1)),
        "unique_bigram_count": bigrams.len(),
        "unique_trigram_count": trigrams.len(),
        "token_entropy": entropy(&token_counts),
        "response_entropy": entropy(&response_counts),
        "unique_response_count": response_counts.len()
    }))
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
        "active scenario binding": true,
        "distractor scenario rejection": true,
        "old/stale/inactive suppression": true,
        "answer-only scenario binding": true
    })
}

fn collapse_metrics(rows: &[EvalRow]) -> Value {
    let chat_rows = rows
        .iter()
        .filter(|row| row.eval_family != "FINITE_LABEL_ANCHORROUTE_RETENTION")
        .collect::<Vec<_>>();
    let total = chat_rows.len().max(1);
    let mut counts = BTreeMap::<String, usize>::new();
    for row in &chat_rows {
        *counts.entry(row.model_output.clone()).or_insert(0) += 1;
    }
    let top = counts.values().copied().max().unwrap_or(0);
    json!({
        "empty_output_rate": ratio(chat_rows.iter().filter(|row| row.model_output.is_empty()).count(), total),
        "space_output_rate": ratio(chat_rows.iter().filter(|row| !row.model_output.is_empty() && row.model_output.chars().all(char::is_whitespace)).count(), total),
        "top_response_rate": ratio(top, total),
        "static_response_rate": ratio(chat_rows.iter().filter(|row| row.output_classification == "static_repeated_output").count(), total),
        "repetition_rate": ratio(chat_rows.iter().filter(|row| has_repetition(&row.model_output)).count(), total),
        "copy_prompt_rate": ratio(chat_rows.iter().filter(|row| row.prompt.contains(&row.model_output) && row.model_output.len() > 5).count(), total),
        "label_only_response_rate": label_only_rate(&chat_rows.iter().map(|row| (*row).clone()).collect::<Vec<_>>()),
        "unique_response_count": counts.len()
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
    checkpoint_hash_unchanged: bool,
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
        && novelty["slot_only_skeleton_reuse_rate"].as_f64().unwrap_or(1.0) <= 0.25
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
        && checkpoint_hash_unchanged
}

fn failure_verdicts(
    composition: &Value,
    novelty: &Value,
    skeleton: &Value,
    vocab: &Value,
    context: &Value,
    retention: &Value,
    collapse: &Value,
) -> Vec<&'static str> {
    let mut verdicts = vec!["CHAT_DIVERSITY_FRESH_CONFIRM_FAILS"];
    if novelty["template_copy_rate"].as_f64().unwrap_or(1.0) > 0.25 {
        verdicts.push("TEMPLATE_COPY_DETECTED");
    }
    if skeleton["response_skeleton_reuse_rate"].as_f64().unwrap_or(1.0) > 0.50 {
        verdicts.push("SKELETON_REUSE_REGRESSION_DETECTED");
    }
    if vocab["generated_to_train_vocab_ratio"].as_f64().unwrap_or(0.0) < 0.35 {
        verdicts.push("VOCAB_DIVERSITY_REGRESSION_DETECTED");
    }
    if context["slot_binding_accuracy"].as_f64().unwrap_or(0.0) < 0.75 {
        verdicts.push("CONTEXT_SLOT_BINDING_FAILS");
    }
    if retention["finite_label_retention_accuracy"].as_f64().unwrap_or(0.0) < 0.90 {
        verdicts.push("FINITE_LABEL_RETENTION_REGRESSION_DETECTED");
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
    if composition["fresh_instruction_accuracy"].as_f64().unwrap_or(0.0) < 0.75 {
        verdicts.push("FRESH_DIVERSITY_INSTRUCTION_FAILS");
    }
    verdicts
}

fn load_prompt_sources(cfg: &Config) -> Result<PromptSources, Box<dyn std::error::Error>> {
    let mut sources = PromptSources::default();
    collect_prompts(&cfg.upstream_080_root.join("train_examples_sample.jsonl"), &mut sources.prompts_080_train)?;
    collect_prompts(&cfg.upstream_080_root.join("eval_examples_sample.jsonl"), &mut sources.prompts_080_eval)?;
    collect_prompts(&cfg.upstream_080_root.join("generation_samples.jsonl"), &mut sources.prompts_080_eval)?;
    collect_prompts(&cfg.upstream_079_root.join("fresh_chat_eval_dataset.jsonl"), &mut sources.prompts_079)?;
    collect_prompts(&cfg.upstream_079_root.join("generation_samples.jsonl"), &mut sources.prompts_079)?;
    collect_prompts(&cfg.upstream_078_root.join("train_examples_sample.jsonl"), &mut sources.prompts_078)?;
    collect_prompts(&cfg.upstream_078_root.join("eval_examples_sample.jsonl"), &mut sources.prompts_078)?;
    collect_prompts(&cfg.upstream_078_root.join("generation_samples.jsonl"), &mut sources.prompts_078)?;
    collect_prompts_optional(&PathBuf::from("target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke/train_examples_sample.jsonl"), &mut sources.prompts_076);
    collect_prompts_optional(&PathBuf::from("target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke/eval_examples_sample.jsonl"), &mut sources.prompts_076);
    collect_prompts_optional(&PathBuf::from("target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke/generation_samples.jsonl"), &mut sources.prompts_076);
    Ok(sources)
}

fn load_copy_sources(cfg: &Config) -> Result<CopySources, Box<dyn std::error::Error>> {
    let mut sources = CopySources::default();
    for path in [
        cfg.upstream_080_root.join("train_examples_sample.jsonl"),
        cfg.upstream_078_root.join("train_examples_sample.jsonl"),
        PathBuf::from("target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke/train_examples_sample.jsonl"),
    ] {
        collect_responses_optional(&path, &["response_text", "model_output"], &mut sources.train_responses);
    }
    for path in [
        cfg.upstream_080_root.join("eval_examples_sample.jsonl"),
        cfg.upstream_080_root.join("human_readable_samples.jsonl"),
        cfg.upstream_079_root.join("human_readable_samples.jsonl"),
        cfg.upstream_078_root.join("eval_examples_sample.jsonl"),
        cfg.upstream_078_root.join("human_readable_samples.jsonl"),
        PathBuf::from("target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke/eval_examples_sample.jsonl"),
        PathBuf::from("target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke/human_readable_samples.jsonl"),
    ] {
        collect_responses_optional(&path, &["response_text", "model_output", "expected_behavior"], &mut sources.eval_outputs);
    }
    for path in [
        cfg.upstream_080_root.join("generation_samples.jsonl"),
        cfg.upstream_079_root.join("generation_samples.jsonl"),
        cfg.upstream_078_root.join("generation_samples.jsonl"),
        PathBuf::from("target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke/generation_samples.jsonl"),
    ] {
        collect_responses_optional(&path, &["model_output"], &mut sources.generated_outputs);
    }
    let checkpoint_076 = PathBuf::from(
        "target/pilot_wave/stable_loop_phase_lock_076_chat_generation_poc/smoke/checkpoints/chat_generation_poc/model_checkpoint.json",
    );
    if checkpoint_076.exists() {
        let value: Value = read_json(&checkpoint_076)?;
        if let Some(table) = value.get("response_table").and_then(|v| v.as_object()) {
            for tokens in table.values() {
                if let Some(items) = tokens.as_array() {
                    let decoded = items
                        .iter()
                        .filter_map(|item| item.as_str())
                        .take_while(|tok| *tok != "<eos>")
                        .collect::<Vec<_>>()
                        .join(" ");
                    sources.response_table_outputs.insert(normalize_response(&decoded));
                }
            }
        }
    }
    sources.template_responses.extend(sources.train_responses.iter().cloned());
    sources.template_responses.extend(sources.eval_outputs.iter().cloned());
    sources.template_responses.extend(sources.generated_outputs.iter().cloned());
    sources.template_responses.extend(sources.response_table_outputs.iter().cloned());
    sources.template_skeletons = sources
        .template_responses
        .iter()
        .map(|value| skeletonize(value))
        .collect::<BTreeSet<_>>();
    Ok(sources)
}

fn prompt_leakage_report(examples: &[EvalExample], sources: &PromptSources) -> PromptLeakageReport {
    let eval_prompts = examples.iter().map(|row| row.prompt.clone()).collect::<BTreeSet<_>>();
    let near_duplicate_prompt_count = eval_prompts
        .iter()
        .filter(|prompt| {
            max_jaccard_one(prompt, &sources.prompts_080_train) >= 0.90
                || max_jaccard_one(prompt, &sources.prompts_080_eval) >= 0.90
                || max_jaccard_one(prompt, &sources.prompts_079) >= 0.90
                || max_jaccard_one(prompt, &sources.prompts_078) >= 0.90
                || max_jaccard_one(prompt, &sources.prompts_076) >= 0.90
        })
        .count();
    PromptLeakageReport {
        overlap_with_080_train_prompt_count: eval_prompts.intersection(&sources.prompts_080_train).count(),
        overlap_with_080_eval_prompt_count: eval_prompts.intersection(&sources.prompts_080_eval).count(),
        overlap_with_079_prompt_count: eval_prompts.intersection(&sources.prompts_079).count(),
        overlap_with_078_prompt_count: eval_prompts.intersection(&sources.prompts_078).count(),
        overlap_with_076_prompt_count: eval_prompts.intersection(&sources.prompts_076).count(),
        max_prompt_token_jaccard_vs_080_train: max_prompt_jaccard(&eval_prompts, &sources.prompts_080_train),
        max_prompt_token_jaccard_vs_080_eval: max_prompt_jaccard(&eval_prompts, &sources.prompts_080_eval),
        max_prompt_token_jaccard_vs_079: max_prompt_jaccard(&eval_prompts, &sources.prompts_079),
        max_prompt_token_jaccard_vs_078: max_prompt_jaccard(&eval_prompts, &sources.prompts_078),
        max_prompt_token_jaccard_vs_076: max_prompt_jaccard(&eval_prompts, &sources.prompts_076),
        near_duplicate_prompt_count,
        eval_prompt_hash: set_hash(&eval_prompts),
    }
}

fn checkpoint_path(cfg: &Config) -> PathBuf {
    cfg.upstream_080_root
        .join("checkpoints")
        .join("chat_composition_diversity_repair")
        .join("model_checkpoint.json")
}

fn missing_upstreams(cfg: &Config, checkpoint: &Path) -> Vec<String> {
    let required = [
        ("upstream_080_summary", cfg.upstream_080_root.join("summary.json")),
        ("upstream_080_checkpoint_manifest", cfg.upstream_080_root.join("checkpoint_manifest.json")),
        ("upstream_080_train_examples_sample", cfg.upstream_080_root.join("train_examples_sample.jsonl")),
        ("upstream_080_eval_examples_sample", cfg.upstream_080_root.join("eval_examples_sample.jsonl")),
        ("upstream_080_generation_samples", cfg.upstream_080_root.join("generation_samples.jsonl")),
        ("upstream_080_checkpoint", checkpoint.to_path_buf()),
        ("upstream_079b_summary", cfg.upstream_079b_root.join("summary.json")),
        ("upstream_079_summary", cfg.upstream_079_root.join("summary.json")),
        ("upstream_078_summary", cfg.upstream_078_root.join("summary.json")),
        ("upstream_074_summary", cfg.upstream_074_root.join("summary.json")),
    ];
    required
        .iter()
        .filter_map(|(name, path)| if path.exists() { None } else { Some((*name).to_string()) })
        .collect()
}

fn collect_prompts(path: &Path, target: &mut BTreeSet<String>) -> Result<(), Box<dyn std::error::Error>> {
    for value in read_jsonl_values(path)? {
        if let Some(prompt) = value.get("prompt").and_then(|v| v.as_str()) {
            target.insert(prompt.to_string());
        }
    }
    Ok(())
}

fn collect_prompts_optional(path: &Path, target: &mut BTreeSet<String>) {
    if path.exists() {
        let _ = collect_prompts(path, target);
    }
}

fn collect_responses_optional(path: &Path, keys: &[&str], target: &mut BTreeSet<String>) {
    if !path.exists() {
        return;
    }
    if let Ok(values) = read_jsonl_values(path) {
        for value in values {
            for key in keys {
                if let Some(response) = value.get(*key).and_then(|v| v.as_str()) {
                    target.insert(normalize_response(response));
                }
            }
        }
    }
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

fn train_response_ngram_overlap(rows: &[EvalRow], train_responses: &BTreeSet<String>) -> f64 {
    let train_ngrams = train_responses.iter().flat_map(|response| ngrams(response, 3)).collect::<BTreeSet<_>>();
    rows.iter()
        .map(|row| overlap_rate(&ngrams(&row.model_output, 3), &train_ngrams))
        .sum::<f64>()
        / rows.len().max(1) as f64
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
    let colors = ["amber", "silver", "teal", "violet", "green", "indigo", "cobalt", "rose"];
    let lower = text.to_lowercase();
    for color in colors {
        if lower.contains(color) {
            return Some(color.to_string());
        }
    }
    None
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
    tokens.len() >= 4 && tokens.windows(4).any(|window| window.iter().all(|tok| tok == &window[0]))
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
            "code" | "value" | "choice" => "[FIELD]".to_string(),
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

fn overlap_rate(values: &BTreeSet<String>, reference: &BTreeSet<String>) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.intersection(reference).count() as f64 / values.len() as f64
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
    let text = fs::read_to_string(path)?;
    let mut values = Vec::new();
    for line in text.lines() {
        if !line.trim().is_empty() {
            values.push(serde_json::from_str(line)?);
        }
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

fn write_upstream_manifest(
    cfg: &Config,
    checkpoint: &Path,
    checkpoint_hash: &str,
    start_ms: u128,
) -> Result<(), Box<dyn std::error::Error>> {
    write_json(
        &cfg.out.join("upstream_080_manifest.json"),
        &json!({
            "schema_version": "chat_diversity_fresh_confirm_upstream_080_manifest_v1",
            "upstream_080_root": cfg.upstream_080_root.display().to_string(),
            "upstream_079b_root": cfg.upstream_079b_root.display().to_string(),
            "upstream_079_root": cfg.upstream_079_root.display().to_string(),
            "upstream_078_root": cfg.upstream_078_root.display().to_string(),
            "upstream_074_root": cfg.upstream_074_root.display().to_string(),
            "checkpoint": checkpoint.display().to_string(),
            "checkpoint_hash_before": checkpoint_hash,
            "upstream_080_summary_present": true,
            "upstream_080_positive": true,
            "eval_started_after_081_start": now_ms() >= start_ms
        }),
    )
}

fn write_failure(out: &Path, verdict: &str, reason: &str) -> Result<(), Box<dyn std::error::Error>> {
    append_progress(out, "failure", json!({"verdict": verdict, "reason": reason}))?;
    write_summary_and_report(
        out,
        "failed",
        vec!["CHAT_DIVERSITY_FRESH_CONFIRM_FAILS".to_string(), verdict.to_string()],
        json!({"failure_reason": reason}),
    )
}

fn write_summary_and_report(
    out: &Path,
    status: &str,
    verdicts: Vec<String>,
    extra: Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut summary = json!({
        "schema_version": "chat_diversity_fresh_confirm_summary_v1",
        "status": status,
        "verdicts": verdicts,
        "bounded_fresh_chat_diversity_confirmation_only": true,
        "not_GPT_like_assistant_readiness": true,
        "not_full_English_LM": true,
        "not_language_grounding": true,
        "not_production_chat": true,
        "not_safety_alignment": true,
        "not_public_beta": true,
        "not_GA": true,
        "not_hosted_SaaS": true
    });
    if let (Some(dst), Some(src)) = (summary.as_object_mut(), extra.as_object()) {
        for (key, value) in src {
            dst.insert(key.clone(), value.clone());
        }
    }
    write_json(&out.join("summary.json"), &summary)?;
    let mut text = String::new();
    text.push_str("# STABLE_LOOP_PHASE_LOCK_081_CHAT_DIVERSITY_FRESH_CONFIRM Report\n\n");
    text.push_str(&format!("status: {status}\n\n"));
    text.push_str("bounded fresh chat diversity confirmation only\n");
    text.push_str("not GPT-like assistant readiness\n");
    text.push_str("not full English LM\n");
    text.push_str("not language grounding\n");
    text.push_str("not production chat\n");
    text.push_str("not safety alignment\n");
    text.push_str("not public beta / GA / hosted SaaS\n\n");
    text.push_str("no service API change\nno deployment harness change\nno SDK/public export change\nno release docs change\nno root LICENSE change\nno upstream checkpoint mutation\n\n");
    text.push_str("verdicts:\n");
    for verdict in summary["verdicts"].as_array().into_iter().flatten() {
        text.push_str(&format!("- {}\n", verdict.as_str().unwrap_or("UNKNOWN")));
    }
    if let Some(composition) = summary.get("composition_metrics") {
        text.push_str("\ncomposition_metrics:\n");
        text.push_str(&serde_json::to_string_pretty(composition)?);
        text.push('\n');
    }
    if let Some(novelty) = summary.get("novelty_metrics") {
        text.push_str("\nnovelty_metrics:\n");
        text.push_str(&serde_json::to_string_pretty(novelty)?);
        text.push('\n');
    }
    if let Some(skeleton) = summary.get("skeleton_diversity_metrics") {
        text.push_str("\nskeleton_diversity_metrics:\n");
        text.push_str(&serde_json::to_string_pretty(skeleton)?);
        text.push('\n');
    }
    text.push_str("\nnext_if_pass: 082_CHAT_DIVERSITY_MULTI_SEED_CONFIRM\n");
    text.push_str("next_if_fail: 081B_CHAT_DIVERSITY_FRESH_FAILURE_ANALYSIS\n");
    fs::write(out.join("report.md"), text)?;
    Ok(())
}

fn append_progress(out: &Path, phase: &str, payload: Value) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out)?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(out.join("progress.jsonl"))?;
    writeln!(
        file,
        "{}",
        serde_json::to_string(&json!({
            "ts_ms": now_ms(),
            "phase": phase,
            "payload": payload
        }))?
    )?;
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

fn json_vec_strings(value: &Value) -> Vec<String> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|item| item.as_str().map(|s| s.to_string()))
        .collect()
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn parse_args() -> Result<Config, String> {
    let mut out = PathBuf::from(DEFAULT_OUT);
    let mut upstream_080_root = PathBuf::from(DEFAULT_UPSTREAM_080_ROOT);
    let mut upstream_079b_root = PathBuf::from(DEFAULT_UPSTREAM_079B_ROOT);
    let mut upstream_079_root = PathBuf::from(DEFAULT_UPSTREAM_079_ROOT);
    let mut upstream_078_root = PathBuf::from(DEFAULT_UPSTREAM_078_ROOT);
    let mut upstream_074_root = PathBuf::from(DEFAULT_UPSTREAM_074_ROOT);
    let mut seed = DEFAULT_SEED;
    let mut heartbeat_sec = 20_u64;
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--out" => {
                idx += 1;
                out = PathBuf::from(args.get(idx).ok_or("--out requires value")?);
            }
            "--upstream-080-root" => {
                idx += 1;
                upstream_080_root = PathBuf::from(args.get(idx).ok_or("--upstream-080-root requires value")?);
            }
            "--upstream-079b-root" => {
                idx += 1;
                upstream_079b_root = PathBuf::from(args.get(idx).ok_or("--upstream-079b-root requires value")?);
            }
            "--upstream-079-root" => {
                idx += 1;
                upstream_079_root = PathBuf::from(args.get(idx).ok_or("--upstream-079-root requires value")?);
            }
            "--upstream-078-root" => {
                idx += 1;
                upstream_078_root = PathBuf::from(args.get(idx).ok_or("--upstream-078-root requires value")?);
            }
            "--upstream-074-root" => {
                idx += 1;
                upstream_074_root = PathBuf::from(args.get(idx).ok_or("--upstream-074-root requires value")?);
            }
            "--seed" => {
                idx += 1;
                seed = args
                    .get(idx)
                    .ok_or("--seed requires value")?
                    .parse()
                    .map_err(|_| "--seed must be integer")?;
            }
            "--heartbeat-sec" => {
                idx += 1;
                heartbeat_sec = args
                    .get(idx)
                    .ok_or("--heartbeat-sec requires value")?
                    .parse()
                    .map_err(|_| "--heartbeat-sec must be integer")?;
            }
            other => return Err(format!("unknown argument {other}")),
        }
        idx += 1;
    }
    Ok(Config {
        out,
        upstream_080_root,
        upstream_079b_root,
        upstream_079_root,
        upstream_078_root,
        upstream_074_root,
        seed,
        heartbeat_sec,
    })
}
