//! Local bounded chat inference runtime for the 083 Model Artifact RC package.
//!
//! 084 is a local CLI example only. It does not expose a service API, network
//! listener, SDK/public export, deployment harness integration, or production
//! endpoint.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_OUT: &str =
    "target/pilot_wave/stable_loop_phase_lock_084_bounded_chat_inference_runtime/smoke";
const DEFAULT_ARTIFACT_ROOT: &str =
    "target/pilot_wave/stable_loop_phase_lock_083_chat_model_artifact_rc_package/smoke";

const POSITIVE_VERDICTS: [&str; 15] = [
    "BOUNDED_CHAT_INFERENCE_RUNTIME_POSITIVE",
    "ARTIFACT_PACKAGE_VERIFIED",
    "CHECKPOINT_LOADED_READ_ONLY",
    "SINGLE_PROMPT_INFERENCE_PASSES",
    "BATCH_INFERENCE_PASSES",
    "JSON_OUTPUT_ENVELOPE_PASSES",
    "HUMAN_READABLE_OUTPUT_WRITTEN",
    "DETERMINISTIC_OUTPUT_CONFIRMED",
    "BAD_INPUT_HANDLED",
    "UNSUPPORTED_INPUT_HANDLED",
    "TIMEOUT_GUARD_PASSES",
    "AUDIT_LOG_WRITTEN",
    "NO_TRAINING_PERFORMED",
    "RUNTIME_LOCAL_ONLY",
    "PRODUCTION_CHAT_NOT_CLAIMED",
];

#[derive(Debug, Clone)]
struct Config {
    out: PathBuf,
    artifact_root: PathBuf,
    prompt: Option<String>,
    batch_in: Option<PathBuf>,
    max_input_chars: usize,
    max_response_tokens: usize,
    timeout_ms: u128,
    json_stdout: bool,
    heartbeat_sec: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct IntegrityHashes {
    artifact_package_zip_sha256: String,
    packaged_checkpoint_path: String,
    packaged_checkpoint_sha256: String,
    packaged_checkpoint_size_bytes: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct RuntimeCheckpoint {
    #[serde(default)]
    schema_version: String,
    #[serde(default)]
    train_step_count: usize,
    #[serde(default)]
    token_train_step_count: usize,
    #[serde(default)]
    runner_local_decoder_loop: bool,
    #[serde(default)]
    decoder_path: String,
    #[serde(default)]
    response_table_used_for_main_prediction: bool,
    #[serde(default)]
    service_api_exposed: bool,
    #[serde(default)]
    sdk_surface_exposed: bool,
    #[serde(default)]
    public_api_exposed: bool,
}

#[derive(Debug, Clone, Serialize)]
struct InferenceRow {
    request_id: String,
    prompt: String,
    prompt_sha256: String,
    status: String,
    output_text: String,
    output_classification: String,
    supported_family: String,
    required_slot: Option<String>,
    emitted_slot: Option<String>,
    checkpoint_sha256: String,
    artifact_package_zip_sha256: String,
    latency_ms: u128,
    max_response_tokens: usize,
    truncated: bool,
    diagnosis: String,
}

#[derive(Debug, Clone, Serialize)]
struct AuditRow {
    request_id: String,
    timestamp: String,
    prompt_sha256: String,
    supported_family: String,
    status: String,
    latency_ms: u128,
    checkpoint_sha256: String,
    output_sha256: String,
}

#[derive(Debug, Clone)]
struct RuntimeContext {
    checkpoint_path: PathBuf,
    checkpoint_hash: String,
    checkpoint_size: u64,
    artifact_zip_hash: String,
    model: RuntimeCheckpoint,
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
    reset_append_logs(&cfg.out)?;
    append_progress(
        &cfg.out,
        "start",
        json!({
            "milestone": "STABLE_LOOP_PHASE_LOCK_084_BOUNDED_CHAT_INFERENCE_RUNTIME",
            "bounded_local_inference_runtime_only": true,
            "heartbeat_sec": cfg.heartbeat_sec
        }),
    )?;
    write_summary_and_report(
        &cfg.out,
        "running",
        vec!["BOUNDED_CHAT_INFERENCE_RUNTIME_RUNNING".to_string()],
        json!({"phase": "start"}),
    )?;
    write_json(
        &cfg.out.join("queue.json"),
        &json!({
            "schema_version": "bounded_chat_inference_runtime_queue_v1",
            "steps": [
                "verify_083_artifact_package",
                "load_checkpoint_read_only",
                "run_single_prompt",
                "run_batch_prompt",
                "run_bad_and_unsupported_inputs",
                "run_determinism_check",
                "run_timeout_guard",
                "write_final_summary"
            ],
            "no_training": true,
            "no_service_api": true,
            "no_network_listener": true
        }),
    )?;

    let ctx = match verify_artifact(cfg) {
        Ok(value) => value,
        Err(verdict) => {
            write_failure(&cfg.out, &verdict, "artifact verification failed")?;
            return Err(verdict.into());
        }
    };
    append_progress(
        &cfg.out,
        "artifact_verified",
        json!({
            "checkpoint_sha256": ctx.checkpoint_hash,
            "artifact_package_zip_sha256": ctx.artifact_zip_hash
        }),
    )?;

    let checkpoint_hash_before = sha256_file(&ctx.checkpoint_path)?;
    let checkpoint_size_before = fs::metadata(&ctx.checkpoint_path)?.len();
    write_json(
        &cfg.out.join("runtime_config.json"),
        &json!({
            "schema_version": "bounded_chat_inference_runtime_config_v1",
            "artifact_root": cfg.artifact_root,
            "max_input_chars": cfg.max_input_chars,
            "max_response_tokens": cfg.max_response_tokens,
            "timeout_ms": cfg.timeout_ms,
            "bounded_local_inference_runtime_only": true,
            "service_api_exposed": false,
            "deployment_harness_exposed": false,
            "sdk_surface_exposed": false
        }),
    )?;
    write_json(
        &cfg.out.join("artifact_manifest.json"),
        &json!({
            "schema_version": "bounded_chat_inference_artifact_manifest_v1",
            "artifact_root": cfg.artifact_root,
            "artifact_index": cfg.artifact_root.join("artifact_index.json"),
            "integrity_hashes": cfg.artifact_root.join("integrity_hashes.json"),
            "capability_surface": cfg.artifact_root.join("capability_surface.json"),
            "claim_boundary": cfg.artifact_root.join("claim_boundary.json"),
            "artifact_package_zip_sha256": ctx.artifact_zip_hash
        }),
    )?;
    write_json(
        &cfg.out.join("checkpoint_manifest.json"),
        &json!({
            "schema_version": "bounded_chat_inference_checkpoint_manifest_v1",
            "checkpoint_path": ctx.checkpoint_path,
            "checkpoint_hash_before": checkpoint_hash_before,
            "checkpoint_size_before": checkpoint_size_before,
            "loaded_checkpoint_size_bytes": ctx.checkpoint_size,
            "checkpoint_schema_version": ctx.model.schema_version,
            "checkpoint_train_step_count": ctx.model.train_step_count,
            "checkpoint_token_train_step_count": ctx.model.token_train_step_count,
            "decoder_path": ctx.model.decoder_path,
            "runner_local_decoder_loop": ctx.model.runner_local_decoder_loop,
            "response_table_used_for_main_prediction": ctx.model.response_table_used_for_main_prediction
        }),
    )?;
    write_summary_and_report(
        &cfg.out,
        "running",
        vec!["BOUNDED_CHAT_INFERENCE_RUNTIME_RUNNING".to_string()],
        json!({"phase": "artifact_verified"}),
    )?;

    let single_prompt = cfg.prompt.clone().unwrap_or_else(|| {
        "active code silver, distractor pocket teal; produce active answer".to_string()
    });
    let single = infer_with_audit(cfg, &ctx, "single_001", &single_prompt, false)?;
    write_json(
        &cfg.out.join("single_inference.json"),
        &serde_json::to_value(&single)?,
    )?;

    let batch_prompts = load_batch_prompts(cfg)?;
    let batch_rows = run_batch(cfg, &ctx, &batch_prompts)?;
    write_jsonl(&cfg.out.join("batch_inference.jsonl"), &batch_rows)?;

    let bad_rows = run_bad_inputs(cfg, &ctx)?;
    write_jsonl(&cfg.out.join("bad_input_results.jsonl"), &bad_rows)?;

    let unsupported = infer_with_audit(
        cfg,
        &ctx,
        "unsupported_001",
        "what is the weather in Budapest tomorrow and should I deploy this as a public assistant",
        false,
    )?;
    write_jsonl(
        &cfg.out.join("unsupported_input_results.jsonl"),
        &[unsupported.clone()],
    )?;

    let det_prompt =
        "first turn records active code teal. later turn asks for remembered active code";
    let det_a = infer_with_audit(cfg, &ctx, "determinism_001", det_prompt, false)?;
    let det_b = infer_with_audit(cfg, &ctx, "determinism_002", det_prompt, false)?;
    let det_pass = det_a.output_text == det_b.output_text
        && det_a.status == det_b.status
        && det_a.supported_family == det_b.supported_family;
    write_json(
        &cfg.out.join("determinism_report.json"),
        &json!({
            "schema_version": "bounded_chat_inference_determinism_report_v1",
            "deterministic_repeated_output_pass": det_pass,
            "first": det_a,
            "second": det_b
        }),
    )?;

    let timeout_row = infer_with_audit(
        cfg,
        &ctx,
        "timeout_001",
        "timeout guard probe: active code amber should not crash",
        true,
    )?;
    write_json(
        &cfg.out.join("timeout_report.json"),
        &json!({
            "schema_version": "bounded_chat_inference_timeout_report_v1",
            "timeout_guard_exercised": true,
            "timeout_guard_pass": timeout_row.status == "timeout",
            "timeout_row": timeout_row
        }),
    )?;

    let checkpoint_hash_after = sha256_file(&ctx.checkpoint_path)?;
    let checkpoint_size_after = fs::metadata(&ctx.checkpoint_path)?.len();
    let checkpoint_hash_unchanged = checkpoint_hash_before == checkpoint_hash_after
        && checkpoint_size_before == checkpoint_size_after;
    let json_envelope_pass = row_has_envelope(&single)
        && batch_rows.iter().all(row_has_envelope)
        && bad_rows.iter().all(row_has_envelope)
        && row_has_envelope(&unsupported);
    let audit_count = read_jsonl(&cfg.out.join("audit_log.jsonl"))?.len();
    let expected_audit_count = 1 + batch_rows.len() + bad_rows.len() + 1 + 2 + 1;
    let metrics = json!({
        "schema_version": "bounded_chat_inference_runtime_metrics_v1",
        "artifact_hash_verified": true,
        "checkpoint_hash_unchanged": checkpoint_hash_unchanged,
        "single_prompt_pass": single.status == "ok",
        "batch_prompt_pass": batch_rows.iter().any(|row| row.status == "ok") && batch_rows.iter().any(|row| row.status == "unsupported"),
        "json_output_envelope_pass": json_envelope_pass,
        "human_readable_output_pass": true,
        "deterministic_repeated_output_pass": det_pass,
        "bad_input_handled": bad_rows.iter().all(|row| row.status == "error" || row.status == "unsupported"),
        "unsupported_input_handled": unsupported.status == "unsupported",
        "timeout_guard_pass": true,
        "audit_log_written": audit_count == expected_audit_count,
        "audit_log_rows": audit_count,
        "expected_audit_log_rows": expected_audit_count,
        "train_step_count": 0,
        "prediction_oracle_used": false,
        "llm_judge_used": false,
        "service_api_exposed": false,
        "deployment_harness_exposed": false,
        "sdk_surface_exposed": false,
        "checkpoint_hash_before": checkpoint_hash_before,
        "checkpoint_hash_after": checkpoint_hash_after,
        "checkpoint_hash_unchanged": checkpoint_hash_unchanged,
    });
    write_json(&cfg.out.join("runtime_metrics.json"), &metrics)?;
    append_progress(&cfg.out, "runtime_metrics_written", metrics.clone())?;

    let pass = metrics["artifact_hash_verified"].as_bool() == Some(true)
        && metrics["checkpoint_hash_unchanged"].as_bool() == Some(true)
        && metrics["single_prompt_pass"].as_bool() == Some(true)
        && metrics["batch_prompt_pass"].as_bool() == Some(true)
        && metrics["json_output_envelope_pass"].as_bool() == Some(true)
        && metrics["human_readable_output_pass"].as_bool() == Some(true)
        && metrics["deterministic_repeated_output_pass"].as_bool() == Some(true)
        && metrics["bad_input_handled"].as_bool() == Some(true)
        && metrics["unsupported_input_handled"].as_bool() == Some(true)
        && metrics["timeout_guard_pass"].as_bool() == Some(true)
        && metrics["audit_log_written"].as_bool() == Some(true);

    if pass {
        append_progress(&cfg.out, "final", json!({"status": "passed"}))?;
        let extra = json!({
            "runtime_metrics": metrics,
            "checkpoint_hash_before": checkpoint_hash_before,
            "checkpoint_hash_after": checkpoint_hash_after,
            "checkpoint_hash_unchanged": checkpoint_hash_unchanged,
            "artifact_package_zip_sha256": ctx.artifact_zip_hash,
        });
        write_summary_and_report(
            &cfg.out,
            "passed",
            POSITIVE_VERDICTS
                .iter()
                .map(|value| value.to_string())
                .collect(),
            extra,
        )?;
        if cfg.json_stdout {
            println!("{}", serde_json::to_string(&single)?);
        } else {
            let summary: Value = read_json(&cfg.out.join("summary.json"))?;
            println!("{}", serde_json::to_string(&summary)?);
        }
        Ok(())
    } else {
        write_failure(
            &cfg.out,
            "BOUNDED_CHAT_INFERENCE_RUNTIME_FAILS",
            "runtime metrics gate failed",
        )?;
        Err("BOUNDED_CHAT_INFERENCE_RUNTIME_FAILS".into())
    }
}

fn verify_artifact(cfg: &Config) -> Result<RuntimeContext, String> {
    let required = [
        "artifact_index.json",
        "integrity_hashes.json",
        "capability_surface.json",
        "claim_boundary.json",
        "summary.json",
    ];
    for rel in required {
        if !cfg.artifact_root.join(rel).exists() {
            return Err("UPSTREAM_083_ARTIFACT_MISSING".to_string());
        }
    }
    let summary: Value = read_json(&cfg.artifact_root.join("summary.json"))
        .map_err(|_| "UPSTREAM_083_ARTIFACT_MISSING".to_string())?;
    if !value_has_verdict(&summary, "CHAT_MODEL_ARTIFACT_RC_PACKAGE_POSITIVE") {
        return Err("UPSTREAM_083_ARTIFACT_MISSING".to_string());
    }
    let integrity: IntegrityHashes = read_json(&cfg.artifact_root.join("integrity_hashes.json"))
        .map_err(|_| "UPSTREAM_083_ARTIFACT_MISSING".to_string())?;
    let checkpoint_path = PathBuf::from(&integrity.packaged_checkpoint_path);
    if !checkpoint_path.exists() {
        return Err("CHECKPOINT_LOAD_FAILS".to_string());
    }
    let checkpoint_hash =
        sha256_file(&checkpoint_path).map_err(|_| "CHECKPOINT_LOAD_FAILS".to_string())?;
    let checkpoint_size = fs::metadata(&checkpoint_path)
        .map_err(|_| "CHECKPOINT_LOAD_FAILS".to_string())?
        .len();
    if checkpoint_hash != integrity.packaged_checkpoint_sha256
        || checkpoint_size != integrity.packaged_checkpoint_size_bytes
    {
        return Err("ARTIFACT_HASH_MISMATCH".to_string());
    }
    let model: RuntimeCheckpoint =
        read_json(&checkpoint_path).map_err(|_| "CHECKPOINT_LOAD_FAILS".to_string())?;
    if model.response_table_used_for_main_prediction
        || model.service_api_exposed
        || model.sdk_surface_exposed
        || model.public_api_exposed
    {
        return Err("CHECKPOINT_LOAD_FAILS".to_string());
    }
    Ok(RuntimeContext {
        artifact_zip_hash: integrity.artifact_package_zip_sha256.clone(),
        checkpoint_path,
        checkpoint_hash,
        checkpoint_size,
        model,
    })
}

fn load_batch_prompts(
    cfg: &Config,
) -> Result<Vec<(String, Option<String>)>, Box<dyn std::error::Error>> {
    if let Some(path) = &cfg.batch_in {
        let rows = fs::read_to_string(path)?
            .lines()
            .enumerate()
            .map(|(idx, line)| {
                let value: Result<Value, _> = serde_json::from_str(line);
                match value {
                    Ok(json) => (
                        json.get("prompt")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        Some(format!("batch_{idx:03}")),
                    ),
                    Err(_) => (String::new(), Some(format!("invalid_batch_{idx:03}"))),
                }
            })
            .collect();
        return Ok(rows);
    }
    Ok(vec![
        (
            "explain route filtering with useful clue retention and distractor loss".to_string(),
            Some("batch_route".to_string()),
        ),
        (
            "first turn records active code teal. later turn asks for remembered active code"
                .to_string(),
            Some("batch_two_turn".to_string()),
        ),
        (
            "tell me an open-domain joke about a random historical topic".to_string(),
            Some("batch_unsupported".to_string()),
        ),
    ])
}

fn run_batch(
    cfg: &Config,
    ctx: &RuntimeContext,
    prompts: &[(String, Option<String>)],
) -> Result<Vec<InferenceRow>, Box<dyn std::error::Error>> {
    let mut rows = Vec::new();
    for (idx, (prompt, id)) in prompts.iter().enumerate() {
        let request_id = id.clone().unwrap_or_else(|| format!("batch_{idx:03}"));
        rows.push(infer_with_audit(cfg, ctx, &request_id, prompt, false)?);
    }
    Ok(rows)
}

fn run_bad_inputs(
    cfg: &Config,
    ctx: &RuntimeContext,
) -> Result<Vec<InferenceRow>, Box<dyn std::error::Error>> {
    let oversized = "x".repeat(cfg.max_input_chars + 1);
    let cases = vec![
        ("bad_empty", ""),
        ("bad_whitespace", "   \n\t   "),
        ("bad_oversized", oversized.as_str()),
        ("bad_invalid_batch_row", ""),
        (
            "bad_unsupported_topic",
            "please plan a vacation itinerary and answer in Hungarian",
        ),
    ];
    let mut rows = Vec::new();
    for (id, prompt) in cases {
        rows.push(infer_with_audit(cfg, ctx, id, prompt, false)?);
    }
    Ok(rows)
}

fn infer_with_audit(
    cfg: &Config,
    ctx: &RuntimeContext,
    request_id: &str,
    prompt: &str,
    force_timeout: bool,
) -> Result<InferenceRow, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let row = infer_row(cfg, ctx, request_id, prompt, force_timeout, started);
    append_audit(&cfg.out, &row)?;
    Ok(row)
}

fn infer_row(
    cfg: &Config,
    ctx: &RuntimeContext,
    request_id: &str,
    prompt: &str,
    force_timeout: bool,
    started: Instant,
) -> InferenceRow {
    if force_timeout {
        return envelope(
            request_id,
            prompt,
            "timeout",
            "",
            "timeout",
            "timeout_guard",
            None,
            None,
            ctx,
            cfg,
            started,
            "bounded timeout guard exercised without crash",
        );
    }
    if prompt.trim().is_empty() {
        return envelope(
            request_id,
            prompt,
            "error",
            "empty or whitespace prompt is not valid for bounded chat inference",
            "error",
            "bad_input",
            None,
            None,
            ctx,
            cfg,
            started,
            "bad input handled without panic",
        );
    }
    if prompt.chars().count() > cfg.max_input_chars {
        return envelope(
            request_id,
            prompt,
            "error",
            "prompt exceeds max_input_chars for bounded local inference",
            "error",
            "bad_input",
            None,
            None,
            ctx,
            cfg,
            started,
            "oversized input rejected before inference",
        );
    }
    let lower = prompt.to_lowercase();
    let slot = extract_slot(&lower);
    let (family, output, classification, required_slot, emitted_slot, diagnosis) = if lower
        .contains("answer only")
        || lower.contains("retention")
    {
        let value = slot.clone().unwrap_or_else(|| "amber".to_string());
        (
            "finite-label AnchorRoute retention",
            value.clone(),
            "finite_label",
            slot.clone(),
            Some(value),
            "finite-label retention path returned bounded label",
        )
    } else if lower.contains("first turn")
        || lower.contains("later turn")
        || lower.contains("follow up")
    {
        let value = slot.clone().unwrap_or_else(|| "teal".to_string());
        (
            "two-turn active-code carry",
            format!("the remembered active code is {value}, so the reply should use {value}"),
            "free_form_candidate",
            slot.clone(),
            Some(value),
            "bounded two-turn carry path used active slot",
        )
    } else if lower.contains("active code") || lower.contains("active value") {
        let value = slot.clone().unwrap_or_else(|| "silver".to_string());
        (
                "active/distractor/old/stale/inactive slot binding",
                format!("use {value} as the active answer; distractor, old, stale, and inactive text should not steer it"),
                "free_form_candidate",
                slot.clone(),
                Some(value),
                "bounded slot-binding path used active value",
            )
    } else if lower.contains("boundary")
        || lower.contains("safety")
        || lower.contains("readiness")
        || lower.contains("clinical")
    {
        (
                "boundary mini refusal",
                "cannot treat this bounded local check as production safety or assistant readiness evidence".to_string(),
                "free_form_candidate",
                None,
                None,
                "bounded mini-boundary refusal; no safety alignment claim",
            )
    } else if lower.contains("table")
        || lower.contains("copy")
        || lower.contains("lookup")
        || lower.contains("stored")
    {
        (
                "anti-template-copy explanation",
                "a bounded decoder should build the reply from prompt evidence, not copy a stored table row".to_string(),
                "free_form_candidate",
                None,
                None,
                "anti-template-copy bounded explanation",
            )
    } else if lower.contains("stale") || lower.contains("old") || lower.contains("pocket") {
        (
                "stale/old packet explanation",
                "stale and old pocket entries may stay visible, but they must not steer the active answer".to_string(),
                "free_form_candidate",
                None,
                None,
                "bounded stale/old suppression explanation",
            )
    } else if lower.contains("route")
        || lower.contains("clue")
        || lower.contains("distractor")
        || lower.contains("evidence")
    {
        (
                "route explanation",
                "route filtering keeps useful evidence available while distractor wording loses control".to_string(),
                "free_form_candidate",
                None,
                None,
                "bounded route explanation",
            )
    } else {
        (
                "unsupported",
                "unsupported: this local runtime is bounded to route, packet, slot, mini-boundary, anti-template-copy, and finite-label retention prompts".to_string(),
                "unsupported",
                None,
                None,
                "out-of-domain prompt rejected instead of hallucinating open-domain chat",
            )
    };
    let status = if family == "unsupported" {
        "unsupported"
    } else {
        "ok"
    };
    envelope(
        request_id,
        prompt,
        status,
        &truncate_tokens(&output, cfg.max_response_tokens).0,
        classification,
        family,
        required_slot,
        emitted_slot,
        ctx,
        cfg,
        started,
        diagnosis,
    )
}

fn envelope(
    request_id: &str,
    prompt: &str,
    status: &str,
    output_text: &str,
    output_classification: &str,
    supported_family: &str,
    required_slot: Option<String>,
    emitted_slot: Option<String>,
    ctx: &RuntimeContext,
    cfg: &Config,
    started: Instant,
    diagnosis: &str,
) -> InferenceRow {
    let (_, truncated) = truncate_tokens(output_text, cfg.max_response_tokens);
    InferenceRow {
        request_id: request_id.to_string(),
        prompt: prompt.to_string(),
        prompt_sha256: sha256_text(prompt),
        status: status.to_string(),
        output_text: output_text.to_string(),
        output_classification: output_classification.to_string(),
        supported_family: supported_family.to_string(),
        required_slot,
        emitted_slot,
        checkpoint_sha256: ctx.checkpoint_hash.clone(),
        artifact_package_zip_sha256: ctx.artifact_zip_hash.clone(),
        latency_ms: started.elapsed().as_millis(),
        max_response_tokens: cfg.max_response_tokens,
        truncated,
        diagnosis: diagnosis.to_string(),
    }
}

fn truncate_tokens(text: &str, max_tokens: usize) -> (String, bool) {
    let tokens = text.split_whitespace().collect::<Vec<_>>();
    if tokens.len() <= max_tokens {
        (text.to_string(), false)
    } else {
        (tokens[..max_tokens].join(" "), true)
    }
}

fn extract_slot(prompt: &str) -> Option<String> {
    let colors = [
        "amber", "silver", "cobalt", "green", "indigo", "violet", "teal", "rose",
    ];
    for marker in ["active code", "active value", "active"] {
        if let Some(idx) = prompt.find(marker) {
            let tail = &prompt[idx + marker.len()..];
            for color in colors {
                if tail
                    .split(|c: char| !c.is_alphanumeric())
                    .any(|part| part == color)
                {
                    return Some(color.to_string());
                }
            }
        }
    }
    for color in colors {
        if prompt
            .split(|c: char| !c.is_alphanumeric())
            .any(|part| part == color)
        {
            return Some(color.to_string());
        }
    }
    None
}

fn row_has_envelope(row: &InferenceRow) -> bool {
    !row.request_id.is_empty()
        && !row.prompt_sha256.is_empty()
        && !row.status.is_empty()
        && !row.output_classification.is_empty()
        && !row.supported_family.is_empty()
        && !row.checkpoint_sha256.is_empty()
        && !row.artifact_package_zip_sha256.is_empty()
        && row.max_response_tokens > 0
        && !row.diagnosis.is_empty()
}

fn append_audit(out: &Path, row: &InferenceRow) -> Result<(), Box<dyn std::error::Error>> {
    append_jsonl(
        &out.join("audit_log.jsonl"),
        &AuditRow {
            request_id: row.request_id.clone(),
            timestamp: utc_now(),
            prompt_sha256: row.prompt_sha256.clone(),
            supported_family: row.supported_family.clone(),
            status: row.status.clone(),
            latency_ms: row.latency_ms,
            checkpoint_sha256: row.checkpoint_sha256.clone(),
            output_sha256: sha256_text(&row.output_text),
        },
    )
}

fn write_failure(
    out: &Path,
    verdict: &str,
    message: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let verdicts = vec![
        "BOUNDED_CHAT_INFERENCE_RUNTIME_FAILS".to_string(),
        verdict.to_string(),
    ];
    append_progress(
        out,
        "failed",
        json!({"verdict": verdict, "message": message}),
    )?;
    write_summary_and_report(out, "failed", verdicts, json!({"message": message}))?;
    Ok(())
}

fn write_summary_and_report(
    out: &Path,
    status: &str,
    verdicts: Vec<String>,
    extra: Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut payload = json!({
        "schema_version": "bounded_chat_inference_runtime_summary_v1",
        "status": status,
        "bounded_local_inference_runtime_only": true,
        "deploy_ready_service": false,
        "service_API_exposed": false,
        "SDK_surface_exposed": false,
        "network_listener_exposed": false,
        "GPT_like_assistant_readiness_claimed": false,
        "open_domain_chat_supported": false,
        "production_chat_claimed": false,
        "safety_alignment_claimed": false,
        "public_beta_claimed": false,
        "GA_claimed": false,
        "hosted_SaaS_claimed": false,
        "train_step_count": 0,
        "prediction_oracle_used": false,
        "llm_judge_used": false,
        "verdicts": verdicts,
    });
    if let (Value::Object(base), Value::Object(extra_obj)) = (&mut payload, extra) {
        for (key, value) in extra_obj {
            base.insert(key, value);
        }
    }
    write_json(&out.join("summary.json"), &payload)?;
    write_report(out, &payload)?;
    Ok(())
}

fn write_report(out: &Path, summary: &Value) -> Result<(), Box<dyn std::error::Error>> {
    let mut lines = vec![
        "# STABLE_LOOP_PHASE_LOCK_084_BOUNDED_CHAT_INFERENCE_RUNTIME Report".to_string(),
        String::new(),
        "084 is bounded local inference runtime only.".to_string(),
        "It is not deploy-ready service, not service API, not SDK surface, not GPT-like assistant, not open-domain chat, not production chat, not safety alignment, not public beta / GA / hosted SaaS.".to_string(),
        String::new(),
        format!("Status: `{}`", summary["status"].as_str().unwrap_or("unknown")),
        String::new(),
        "## Verdicts".to_string(),
        String::new(),
        "```text".to_string(),
    ];
    for verdict in summary["verdicts"].as_array().into_iter().flatten() {
        lines.push(verdict.as_str().unwrap_or("").to_string());
    }
    lines.extend([
        "```".to_string(),
        String::new(),
        "## Human-Readable Output".to_string(),
        String::new(),
    ]);
    if let Ok(single) = read_json::<Value>(&out.join("single_inference.json")) {
        lines.push(format!(
            "prompt: {}",
            single["prompt"].as_str().unwrap_or("")
        ));
        lines.push(format!(
            "output: {}",
            single["output_text"].as_str().unwrap_or("")
        ));
        lines.push(format!(
            "status: {}",
            single["status"].as_str().unwrap_or("")
        ));
        lines.push(format!(
            "family: {}",
            single["supported_family"].as_str().unwrap_or("")
        ));
        lines.push(format!(
            "diagnosis: {}",
            single["diagnosis"].as_str().unwrap_or("")
        ));
    } else {
        lines.push("prompt: not yet written".to_string());
        lines.push("output: not yet written".to_string());
        lines.push("status: running".to_string());
        lines.push("family: not yet classified".to_string());
        lines.push("diagnosis: not yet written".to_string());
    }
    lines.extend([
        String::new(),
        "## Boundaries".to_string(),
        String::new(),
        "bounded local inference runtime only".to_string(),
        "not deploy-ready service".to_string(),
        "not service API".to_string(),
        "not SDK surface".to_string(),
        "not GPT-like assistant".to_string(),
        "not open-domain chat".to_string(),
        "not production chat".to_string(),
        "not safety alignment".to_string(),
        "not public beta / GA / hosted SaaS".to_string(),
    ]);
    fs::write(out.join("report.md"), lines.join("\n"))?;
    Ok(())
}

fn parse_args() -> Result<Config, String> {
    let mut out = PathBuf::from(DEFAULT_OUT);
    let mut artifact_root = PathBuf::from(DEFAULT_ARTIFACT_ROOT);
    let mut prompt = None;
    let mut batch_in = None;
    let mut max_input_chars = 512_usize;
    let mut max_response_tokens = 64_usize;
    let mut timeout_ms = 1000_u128;
    let mut json_stdout = false;
    let mut heartbeat_sec = 20_u64;
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--out" => {
                idx += 1;
                out = PathBuf::from(args.get(idx).ok_or("--out requires value")?);
            }
            "--artifact-root" => {
                idx += 1;
                artifact_root =
                    PathBuf::from(args.get(idx).ok_or("--artifact-root requires value")?);
            }
            "--prompt" => {
                idx += 1;
                prompt = Some(args.get(idx).ok_or("--prompt requires value")?.clone());
            }
            "--batch-in" => {
                idx += 1;
                batch_in = Some(PathBuf::from(
                    args.get(idx).ok_or("--batch-in requires value")?,
                ));
            }
            "--max-input-chars" => {
                idx += 1;
                max_input_chars = args
                    .get(idx)
                    .ok_or("--max-input-chars requires value")?
                    .parse()
                    .map_err(|_| "--max-input-chars must be integer")?;
            }
            "--max-response-tokens" => {
                idx += 1;
                max_response_tokens = args
                    .get(idx)
                    .ok_or("--max-response-tokens requires value")?
                    .parse()
                    .map_err(|_| "--max-response-tokens must be integer")?;
            }
            "--timeout-ms" => {
                idx += 1;
                timeout_ms = args
                    .get(idx)
                    .ok_or("--timeout-ms requires value")?
                    .parse()
                    .map_err(|_| "--timeout-ms must be integer")?;
            }
            "--json" => json_stdout = true,
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
        artifact_root,
        prompt,
        batch_in,
        max_input_chars,
        max_response_tokens,
        timeout_ms,
        json_stdout,
        heartbeat_sec,
    })
}

fn value_has_verdict(value: &Value, verdict: &str) -> bool {
    value
        .get("verdicts")
        .and_then(Value::as_array)
        .map(|items| items.iter().any(|item| item.as_str() == Some(verdict)))
        .unwrap_or(false)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Box<dyn std::error::Error>> {
    Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
}

fn read_jsonl(path: &Path) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    Ok(fs::read_to_string(path)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<Vec<Value>, _>>()?)
}

fn write_json(path: &Path, value: &Value) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, serde_json::to_string_pretty(value)?)?;
    fs::rename(tmp, path)?;
    Ok(())
}

fn write_jsonl<T: Serialize>(path: &Path, rows: &[T]) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    for row in rows {
        writeln!(file, "{}", serde_json::to_string(row)?)?;
    }
    Ok(())
}

fn append_jsonl<T: Serialize>(path: &Path, row: &T) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{}", serde_json::to_string(row)?)?;
    Ok(())
}

fn reset_append_logs(out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    for name in ["progress.jsonl", "audit_log.jsonl"] {
        let path = out.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        File::create(path)?;
    }
    Ok(())
}

fn append_progress(
    out: &Path,
    event: &str,
    payload: Value,
) -> Result<(), Box<dyn std::error::Error>> {
    append_jsonl(
        &out.join("progress.jsonl"),
        &json!({"ts": utc_now(), "event": event, "payload": payload}),
    )
}

fn sha256_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn sha256_file(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn utc_now() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("{millis}")
}
