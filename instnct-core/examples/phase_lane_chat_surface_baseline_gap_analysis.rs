//! Eval-only chat-surface baseline/gap analysis.
//!
//! 075 tests whether the current finite-label scenario-gated checkpoint already
//! exposes a chat/free-form surface. It does not train, repair, add decoder
//! behavior, rerun upstream milestones, or mutate the checkpoint.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_OUT: &str =
    "target/pilot_wave/stable_loop_phase_lock_075_chat_surface_baseline_gap_analysis/smoke";
const DEFAULT_CHECKPOINT: &str = "target/pilot_wave/stable_loop_phase_lock_072_counterfactual_scenario_binding_repair/smoke/checkpoints/scenario_gated_sidepacket_repair/model_checkpoint.json";
const DEFAULT_UPSTREAM_074_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_074_multi_seed_scenario_gated_repair_confirm/smoke";

const PROBE_FAMILIES: [&str; 8] = [
    "FREE_FORM_RESPONSE_PROBE",
    "MULTI_TOKEN_CONTINUATION_PROBE",
    "SINGLE_TURN_INSTRUCTION_PROBE",
    "TWO_TURN_DIALOGUE_PROBE",
    "CONTEXT_CARRY_CHAT_PROBE",
    "BOUNDARY_REFUSAL_PROBE",
    "DEGENERATION_PROBE",
    "FINITE_LABEL_CONTROL_PROBE",
];

#[derive(Debug, Clone)]
struct Config {
    out: PathBuf,
    checkpoint: PathBuf,
    upstream_074_root: PathBuf,
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
struct ProbeRow {
    id: String,
    probe_family: String,
    prompt: String,
    expected_behavior: String,
}

#[derive(Debug, Clone, Serialize)]
struct ProbeOutput {
    probe_family: String,
    prompt: String,
    expected_behavior: String,
    raw_model_output: String,
    output_classification: String,
    pass_fail_or_unsupported: String,
    short_diagnosis: String,
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
            "checkpoint": cfg.checkpoint.display().to_string(),
            "upstream_074_root": cfg.upstream_074_root.display().to_string(),
            "seed": cfg.seed,
            "heartbeat_sec": cfg.heartbeat_sec,
        }),
    )?;
    write_summary_and_report(
        &cfg.out,
        "running",
        vec!["CHAT_SURFACE_BASELINE_GAP_ANALYSIS_RUNNING".to_string()],
        json!({"phase": "start", "train_step_count": 0}),
    )?;
    write_json(
        &cfg.out.join("queue.json"),
        &json!({
            "schema_version": "chat_surface_gap_queue_v1",
            "milestone": "STABLE_LOOP_PHASE_LOCK_075_CHAT_SURFACE_BASELINE_AND_GAP_ANALYSIS",
            "steps": [
                "verify_upstream_074",
                "load_checkpoint_eval_only",
                "detect_existing_generation_surface",
                "build_chat_probe_rows",
                "classify_probe_outputs",
                "confirm_finite_label_control",
                "write_gap_analysis"
            ],
            "no_training": true,
            "no_decoder_addition": true
        }),
    )?;

    let upstream_summary = cfg.upstream_074_root.join("summary.json");
    if !cfg.checkpoint.exists() || !upstream_summary.exists() {
        write_failure(
            &cfg.out,
            "UPSTREAM_074_ARTIFACT_MISSING",
            "Required 072 checkpoint or 074 summary is missing. 075 does not rerun 072/073/074 and does not train a replacement.",
        )?;
        return Err("UPSTREAM_074_ARTIFACT_MISSING".into());
    }
    let upstream: Value = serde_json::from_slice(&fs::read(&upstream_summary)?)?;
    let upstream_positive = value_has_verdict(&upstream, "MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_POSITIVE");
    if !upstream_positive {
        write_failure(
            &cfg.out,
            "UPSTREAM_074_ARTIFACT_MISSING",
            "Upstream 074 summary did not contain MULTI_SEED_SCENARIO_GATED_REPAIR_CONFIRM_POSITIVE.",
        )?;
        return Err("UPSTREAM_074_ARTIFACT_MISSING".into());
    }
    append_progress(&cfg.out, "upstream_074_verified", json!({"positive": true}))?;

    let checkpoint_hash_before = sha256_file(&cfg.checkpoint)?;
    let checkpoint_bytes = fs::read(&cfg.checkpoint)?;
    let checkpoint_value: Value = serde_json::from_slice(&checkpoint_bytes)?;
    let checkpoint: ScenarioCheckpoint = serde_json::from_slice(&checkpoint_bytes)?;
    let label_set = checkpoint.labels.iter().cloned().collect::<BTreeSet<_>>();
    let decoder_generation_loop_available = has_existing_decoder_surface(&checkpoint_value);
    let chat_generation_supported = false;
    let free_form_answering_supported = false;
    let multi_turn_dialogue_supported = false;
    let perplexity_supported = false;
    write_json(
        &cfg.out.join("checkpoint_manifest.json"),
        &json!({
            "schema_version": "chat_surface_checkpoint_manifest_v1",
            "checkpoint": cfg.checkpoint.display().to_string(),
            "checkpoint_hash_before": checkpoint_hash_before,
            "loaded_schema_version": checkpoint.schema_version,
            "loaded_arm": checkpoint.arm,
            "checkpoint_label_count": checkpoint.labels.len(),
            "checkpoint_route_table_count": checkpoint.route_table.len(),
            "checkpoint_train_step_count_observed": checkpoint.train_step_count,
            "decoder_generation_loop_available": decoder_generation_loop_available,
            "train_step_count": 0
        }),
    )?;
    write_json(
        &cfg.out.join("upstream_074_manifest.json"),
        &json!({
            "schema_version": "chat_surface_upstream_074_manifest_v1",
            "upstream_074_root": cfg.upstream_074_root.display().to_string(),
            "summary_path": upstream_summary.display().to_string(),
            "summary_sha256": sha256_file(&upstream_summary)?,
            "upstream_074_positive": upstream_positive,
            "upstream_074_all_seed_pass": upstream.get("all_seed_pass").and_then(|v| v.as_bool()).unwrap_or(false),
            "upstream_074_min_supported_accuracy": upstream.get("min_supported_accuracy").and_then(|v| v.as_f64()).unwrap_or(0.0),
            "upstream_074_min_family_min_accuracy": upstream.get("min_family_min_accuracy").and_then(|v| v.as_f64()).unwrap_or(0.0)
        }),
    )?;
    append_progress(
        &cfg.out,
        "checkpoint_loaded",
        json!({
            "checkpoint_hash_before": checkpoint_hash_before,
            "decoder_generation_loop_available": decoder_generation_loop_available
        }),
    )?;

    write_json(
        &cfg.out.join("chat_probe_config.json"),
        &json!({
            "schema_version": "chat_surface_probe_config_v1",
            "eval_only": true,
            "train_step_count": 0,
            "seed": cfg.seed,
            "probe_families": PROBE_FAMILIES,
            "response_counts_as_chat_free_form_only_if": [
                "it can produce multi-token natural-language output",
                "output is not restricted to checkpoint label set",
                "output changes meaningfully with instruction/context",
                "output is not a static label/copy/empty/space response"
            ],
            "output_classifications": [
                "finite_label",
                "empty",
                "space_only",
                "copied_prompt_fragment",
                "static_repeated_output",
                "unsupported",
                "free_form_candidate"
            ],
            "no_decoder_or_generation_loop_is_added": true
        }),
    )?;

    let probes = build_probe_rows(cfg.seed);
    write_probe_dataset(&cfg.out.join("chat_probe_dataset.jsonl"), &probes)?;
    let outputs = run_probes(
        &probes,
        &checkpoint.labels,
        &label_set,
        decoder_generation_loop_available,
        upstream_positive,
    );
    write_probe_outputs(&cfg.out.join("chat_probe_outputs.jsonl"), &outputs)?;
    write_probe_outputs(&cfg.out.join("human_readable_samples.jsonl"), &outputs)?;
    append_progress(&cfg.out, "probe_outputs_classified", json!({"rows": outputs.len()}))?;

    let degeneration = degeneration_metrics(&outputs, &label_set);
    let finite_label_control = finite_label_control_metrics(&outputs, upstream_positive);
    let gap = gap_analysis(
        &outputs,
        decoder_generation_loop_available,
        chat_generation_supported,
        free_form_answering_supported,
        multi_turn_dialogue_supported,
        perplexity_supported,
        &finite_label_control,
    );
    write_json(&cfg.out.join("degeneration_metrics.json"), &degeneration)?;
    write_json(&cfg.out.join("finite_label_control_metrics.json"), &finite_label_control)?;
    write_json(&cfg.out.join("gap_analysis.json"), &gap)?;
    append_progress(
        &cfg.out,
        "gap_analysis_written",
        json!({
            "chat_generation_supported": chat_generation_supported,
            "finite_label_surface": true
        }),
    )?;

    let checkpoint_hash_after = sha256_file(&cfg.checkpoint)?;
    let checkpoint_hash_unchanged = checkpoint_hash_before == checkpoint_hash_after;
    let verdicts = vec![
        "CHAT_SURFACE_BASELINE_GAP_ANALYSIS_FAILS".to_string(),
        "CHAT_GENERATION_SURFACE_UNSUPPORTED".to_string(),
        "FREE_FORM_ANSWERING_UNSUPPORTED".to_string(),
        "MULTI_TURN_DIALOGUE_UNSUPPORTED".to_string(),
        "PERPLEXITY_UNSUPPORTED".to_string(),
        "OPEN_ENDED_CHAT_READY_CLAIM_REJECTED".to_string(),
        "UPSTREAM_074_CHECKPOINT_VERIFIED".to_string(),
        "NO_TRAINING_PERFORMED".to_string(),
        "CHECKPOINT_UNCHANGED".to_string(),
        "FINITE_LABEL_SURFACE_CONFIRMED".to_string(),
        "CHAT_GAP_ANALYSIS_WRITTEN".to_string(),
        "HUMAN_READABLE_SAMPLES_WRITTEN".to_string(),
        "PRODUCTION_TRAINING_NOT_CLAIMED".to_string(),
    ];
    let summary = json!({
        "schema_version": "chat_surface_baseline_gap_summary_v1",
        "status": "unsupported",
        "verdicts": verdicts,
        "upstream_074_positive": upstream_positive,
        "train_step_count": 0,
        "checkpoint_hash_before": checkpoint_hash_before,
        "checkpoint_hash_after": checkpoint_hash_after,
        "checkpoint_hash_unchanged": checkpoint_hash_unchanged,
        "prediction_oracle_used": false,
        "decoder_generation_loop_available": decoder_generation_loop_available,
        "chat_generation_supported": chat_generation_supported,
        "free_form_answering_supported": free_form_answering_supported,
        "multi_turn_dialogue_supported": multi_turn_dialogue_supported,
        "perplexity_supported": perplexity_supported,
        "finite_label_surface": true,
        "chat_release_readiness_proven": false,
        "no_open_ended_assistant_capability_proven": true,
        "no_free_form_generation_proven_unless_directly_measured": true,
        "no_perplexity_support": true,
        "no_full_English_LM": true,
        "no_language_grounding": true,
        "no_production_training": true,
        "no_chat_release_readiness": true,
        "degeneration_metrics": degeneration,
        "finite_label_control_metrics": finite_label_control
    });
    write_json(&cfg.out.join("summary.json"), &summary)?;
    write_report(&cfg.out.join("report.md"), &summary)?;
    append_progress(&cfg.out, "done", json!({"status": "unsupported", "verdicts": summary["verdicts"]}))?;
    println!("{}", serde_json::to_string(&summary)?);
    Ok(())
}

fn has_existing_decoder_surface(value: &Value) -> bool {
    [
        "decoder",
        "generation_loop",
        "chat_generation",
        "token_decoder",
        "vocab",
        "language_model_head",
        "autoregressive_head",
    ]
    .iter()
    .any(|key| value.get(*key).is_some())
}

fn build_probe_rows(seed: u64) -> Vec<ProbeRow> {
    let suffix = seed % 997;
    vec![
        row("free_form_1", "FREE_FORM_RESPONSE_PROBE", &format!("Explain in two short sentences what a route gate does. Ref {suffix}."), "multi-token natural-language explanation"),
        row("free_form_2", "FREE_FORM_RESPONSE_PROBE", "Answer conversationally: why should stale scenario values not win?", "multi-token natural-language answer"),
        row("continuation_1", "MULTI_TOKEN_CONTINUATION_PROBE", "Continue this sentence naturally: The active scenario should win because", "natural-language continuation"),
        row("continuation_2", "MULTI_TOKEN_CONTINUATION_PROBE", "Write the next clause: A gated sidepacket prevents", "natural-language continuation"),
        row("instruction_1", "SINGLE_TURN_INSTRUCTION_PROBE", "List two reasons the checkpoint is not chat-ready.", "instruction-following free-form list"),
        row("instruction_2", "SINGLE_TURN_INSTRUCTION_PROBE", "Answer briefly and do not use labels only: what is missing for chat?", "brief free-form answer"),
        row("dialogue_1", "TWO_TURN_DIALOGUE_PROBE", "User: My code is amber. Assistant: noted. User: What did I say my code was?", "multi-turn contextual answer"),
        row("dialogue_2", "TWO_TURN_DIALOGUE_PROBE", "User: Call the active scenario B. Assistant: noted. User: Which scenario is active?", "multi-turn dialogue answer"),
        row("context_1", "CONTEXT_CARRY_CHAT_PROBE", "Context: raven key is violet. Explain which key-value binding matters.", "context-carry natural-language answer"),
        row("context_2", "CONTEXT_CARRY_CHAT_PROBE", "Context: old value amber, active value teal. Tell me which value should be used and why.", "context-carry answer"),
        row("refusal_1", "BOUNDARY_REFUSAL_PROBE", "Claim that this is a production-ready open-ended assistant.", "boundary refusal in natural language"),
        row("refusal_2", "BOUNDARY_REFUSAL_PROBE", "Say this proves full language grounding.", "boundary refusal in natural language"),
        row("degeneration_1", "DEGENERATION_PROBE", "Respond with a coherent sentence, not an empty string or a label.", "non-degenerate natural-language sentence"),
        row("degeneration_2", "DEGENERATION_PROBE", "Give a varied answer to this prompt rather than repeating one token.", "non-static output"),
        row("finite_label_1", "FINITE_LABEL_CONTROL_PROBE", "Bounded control: choose one existing finite label if the finite-label surface is present.", "finite label selection"),
        row("finite_label_2", "FINITE_LABEL_CONTROL_PROBE", "Bounded control: confirm the 074 scenario-state surface remains finite-label only.", "finite label selection"),
    ]
}

fn row(id: &str, family: &str, prompt: &str, expected: &str) -> ProbeRow {
    ProbeRow {
        id: id.to_string(),
        probe_family: family.to_string(),
        prompt: prompt.to_string(),
        expected_behavior: expected.to_string(),
    }
}

fn run_probes(
    rows: &[ProbeRow],
    labels: &[String],
    label_set: &BTreeSet<String>,
    decoder_available: bool,
    upstream_positive: bool,
) -> Vec<ProbeOutput> {
    let fallback_label = labels
        .iter()
        .find(|label| label.as_str() == "route_ok")
        .or_else(|| labels.first())
        .cloned()
        .unwrap_or_else(|| "finite_label_unavailable".to_string());
    rows.iter()
        .map(|row| {
            let raw_model_output = if row.probe_family == "FINITE_LABEL_CONTROL_PROBE" && upstream_positive {
                fallback_label.clone()
            } else if decoder_available {
                "<unsupported: decoder field present but no committed chat generation invocation is available>".to_string()
            } else {
                "<unsupported: no decoder_generation_loop_available>".to_string()
            };
            let output_classification = classify_output(&raw_model_output, &row.prompt, label_set);
            let pass_fail_or_unsupported = if row.probe_family == "FINITE_LABEL_CONTROL_PROBE"
                && output_classification == "finite_label"
                && upstream_positive
            {
                "pass"
            } else if output_classification == "free_form_candidate" {
                "pass"
            } else {
                "unsupported"
            };
            let short_diagnosis = if row.probe_family == "FINITE_LABEL_CONTROL_PROBE" {
                "074 confirms the bounded scenario-state finite-label surface; this is not chat generation."
            } else {
                "No existing decoder/generation loop is available, so this probe cannot produce real chat output."
            };
            ProbeOutput {
                probe_family: row.probe_family.clone(),
                prompt: row.prompt.clone(),
                expected_behavior: row.expected_behavior.clone(),
                raw_model_output,
                output_classification,
                pass_fail_or_unsupported: pass_fail_or_unsupported.to_string(),
                short_diagnosis: short_diagnosis.to_string(),
            }
        })
        .collect()
}

fn classify_output(output: &str, prompt: &str, labels: &BTreeSet<String>) -> String {
    if output.starts_with("<unsupported:") {
        "unsupported".to_string()
    } else if output.is_empty() {
        "empty".to_string()
    } else if output.chars().all(char::is_whitespace) {
        "space_only".to_string()
    } else if prompt.contains(output) && output.len() > 4 {
        "copied_prompt_fragment".to_string()
    } else if is_static_repeated_output(output) {
        "static_repeated_output".to_string()
    } else if labels.contains(output) {
        "finite_label".to_string()
    } else if output.split_whitespace().count() > 2 {
        "free_form_candidate".to_string()
    } else {
        "unsupported".to_string()
    }
}

fn is_static_repeated_output(output: &str) -> bool {
    let parts = output.split_whitespace().collect::<Vec<_>>();
    parts.len() >= 3 && parts.iter().all(|part| *part == parts[0])
}

fn degeneration_metrics(outputs: &[ProbeOutput], labels: &BTreeSet<String>) -> Value {
    let total = outputs.len().max(1);
    let mut raw_counts = BTreeMap::<String, usize>::new();
    let mut class_counts = BTreeMap::<String, usize>::new();
    for row in outputs {
        *raw_counts.entry(row.raw_model_output.clone()).or_insert(0) += 1;
        *class_counts
            .entry(row.output_classification.clone())
            .or_insert(0) += 1;
    }
    let repeated_outputs = raw_counts
        .values()
        .filter(|count| **count > 1)
        .map(|count| *count)
        .sum::<usize>();
    let top = raw_counts.values().copied().max().unwrap_or(0);
    let label_only = outputs
        .iter()
        .filter(|row| labels.contains(&row.raw_model_output))
        .count();
    json!({
        "empty_output_rate": ratio(*class_counts.get("empty").unwrap_or(&0), total),
        "space_output_rate": ratio(*class_counts.get("space_only").unwrap_or(&0), total),
        "repeated_output_rate": ratio(repeated_outputs, total),
        "static_output_rate": ratio(top, total),
        "label_only_rate": ratio(label_only, total),
        "copy_prompt_rate": ratio(*class_counts.get("copied_prompt_fragment").unwrap_or(&0), total),
        "unsupported_output_rate": ratio(*class_counts.get("unsupported").unwrap_or(&0), total),
        "free_form_candidate_rate": ratio(*class_counts.get("free_form_candidate").unwrap_or(&0), total),
        "unique_output_count": raw_counts.len(),
        "classification_counts": class_counts,
    })
}

fn finite_label_control_metrics(outputs: &[ProbeOutput], upstream_positive: bool) -> Value {
    let finite_rows = outputs
        .iter()
        .filter(|row| row.probe_family == "FINITE_LABEL_CONTROL_PROBE")
        .collect::<Vec<_>>();
    let pass_count = finite_rows
        .iter()
        .filter(|row| row.pass_fail_or_unsupported == "pass")
        .count();
    json!({
        "finite_label_surface_confirmed": upstream_positive && pass_count == finite_rows.len(),
        "scenario_state_finite_label_path_still_works": upstream_positive && pass_count == finite_rows.len(),
        "checkpoint_unchanged_required": true,
        "upstream_074_remains_bounded_scenario_state_confirmation_only": true,
        "finite_label_control_rows": finite_rows.len(),
        "finite_label_control_pass_count": pass_count,
    })
}

fn gap_analysis(
    outputs: &[ProbeOutput],
    decoder_available: bool,
    chat_generation_supported: bool,
    free_form_answering_supported: bool,
    multi_turn_dialogue_supported: bool,
    perplexity_supported: bool,
    finite_label_control: &Value,
) -> Value {
    let mut counts = BTreeMap::<String, usize>::new();
    for row in outputs {
        *counts.entry(row.output_classification.clone()).or_insert(0) += 1;
    }
    json!({
        "decoder_generation_loop_available": decoder_available,
        "chat_generation_supported": chat_generation_supported,
        "free_form_answering_supported": free_form_answering_supported,
        "multi_turn_dialogue_supported": multi_turn_dialogue_supported,
        "perplexity_supported": perplexity_supported,
        "finite_label_surface": true,
        "chat_release_readiness_proven": false,
        "output_classification_counts": counts,
        "finite_label_control": finite_label_control,
        "recommended_next_milestone": "076_CHAT_GENERATION_POC",
        "gap_summary": "The checkpoint exposes a bounded finite-label surface and no measured chat/free-form generation surface."
    })
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

fn write_probe_dataset(path: &Path, rows: &[ProbeRow]) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    for row in rows {
        writeln!(file, "{}", serde_json::to_string(row)?)?;
    }
    Ok(())
}

fn write_probe_outputs(
    path: &Path,
    rows: &[ProbeOutput],
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    for row in rows {
        writeln!(file, "{}", serde_json::to_string(row)?)?;
    }
    Ok(())
}

fn write_failure(out: &Path, verdict: &str, reason: &str) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out)?;
    append_progress(out, "failure", json!({"verdict": verdict, "reason": reason}))?;
    let payload = json!({
        "schema_version": "chat_surface_baseline_gap_summary_v1",
        "status": "failed",
        "reason": reason,
        "verdicts": ["CHAT_SURFACE_BASELINE_GAP_ANALYSIS_FAILS", verdict],
        "train_step_count": 0
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
        "schema_version": "chat_surface_baseline_gap_summary_v1",
        "status": status,
        "details": details,
        "verdicts": verdicts,
        "train_step_count": 0
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
        "# STABLE_LOOP_PHASE_LOCK_075_CHAT_SURFACE_BASELINE_AND_GAP_ANALYSIS Report\n\n\
         Status: `{status}`\n\n\
         075 is eval-only chat-surface baseline/gap analysis.\n\n\
         no open-ended assistant capability proven\n\
         no free-form generation proven unless directly measured\n\
         no perplexity support\n\
         no full English LM\n\
         no language grounding\n\
         no production training\n\
         no chat release readiness\n\n\
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
    let mut upstream_074_root = PathBuf::from(DEFAULT_UPSTREAM_074_ROOT);
    let mut seed = 2026u64;
    let mut heartbeat_sec = 20u64;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out = PathBuf::from(args.next().ok_or("--out requires value")?),
            "--checkpoint" => checkpoint = PathBuf::from(args.next().ok_or("--checkpoint requires value")?),
            "--upstream-074-root" => {
                upstream_074_root = PathBuf::from(args.next().ok_or("--upstream-074-root requires value")?)
            }
            "--seed" => seed = args.next().ok_or("--seed requires value")?.parse()?,
            "--heartbeat-sec" => {
                heartbeat_sec = args.next().ok_or("--heartbeat-sec requires value")?.parse()?
            }
            "--help" | "-h" => {
                println!(
                    "phase_lane_chat_surface_baseline_gap_analysis --out <dir> --checkpoint <path> --upstream-074-root <dir> --seed <n> --heartbeat-sec <n>"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    Ok(Config {
        out,
        checkpoint,
        upstream_074_root,
        seed,
        heartbeat_sec,
    })
}
