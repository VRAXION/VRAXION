//! Smoke runner for the 057 INSTNCT SDK release candidate.

use instnct_core::sdk_candidate::{
    evaluate_candidate, export_visual_candidate, infer_candidate, load_checkpoint_candidate,
    progress::append_progress, rollback_candidate, save_checkpoint_candidate, train_candidate,
    ClaimBoundary, DataRef, EvalSuiteRef, InputBatch, IntendedUse, LoadCheckpointRequest, RunRef,
    SaveCheckpointRequest, SdkCallContext, TrainCandidateConfig, VisualExportConfig,
    SDK_CANDIDATE_SCHEMA_VERSION,
};
use serde::Serialize;
use serde_json::json;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out = parse_out()?;
    fs::create_dir_all(&out)?;
    let progress_path = out.join("progress.jsonl");
    let context = SdkCallContext::research(Some(progress_path.clone()));

    write_json(
        &out.join("queue.json"),
        &json!({
            "schema_version": SDK_CANDIDATE_SCHEMA_VERSION,
            "phase_lock": "STABLE_LOOP_PHASE_LOCK_057_INSTNCT_SDK_RELEASE_CANDIDATE",
            "operations": [
                "train",
                "save",
                "load",
                "infer",
                "evaluate",
                "export_visual",
                "rollback",
                "invalid_schema_rejection",
                "regulated_use_rejection"
            ]
        }),
    )?;

    let train = train_candidate(
        TrainCandidateConfig {
            context: context.clone(),
            out_dir: out.clone(),
            seed: 2026,
        },
        DataRef {
            id: "sdk_candidate_smoke_data".to_string(),
            description: "bounded deterministic smoke data".to_string(),
        },
    );
    let checkpoint = train.value.clone().ok_or("train failed")?;

    let save = save_checkpoint_candidate(
        &checkpoint,
        SaveCheckpointRequest {
            context: context.clone(),
            destination: out.join("checkpoints").join("sdk_candidate_saved.ckpt"),
        },
    );
    let checkpoint_hash = save.value.clone().ok_or("save failed")?;

    let load = load_checkpoint_candidate(
        LoadCheckpointRequest {
            context: context.clone(),
            source: checkpoint_hash.path.clone(),
        },
        &checkpoint_hash.sha256,
    );
    let loaded_checkpoint = load.value.clone().ok_or("load failed")?;

    let infer = infer_candidate(
        &loaded_checkpoint,
        InputBatch {
            context: context.clone(),
            inputs: vec![
                "route: source -> relay -> target".to_string(),
                "memory: key=lamp query=color".to_string(),
                "non_route: parity 8".to_string(),
            ],
        },
    );
    let inference = infer.value.clone().ok_or("infer failed")?;

    let evaluate = evaluate_candidate(
        &loaded_checkpoint,
        EvalSuiteRef {
            context: context.clone(),
            suite_id: "sdk_candidate_eval_smoke".to_string(),
        },
    );
    let eval_report = evaluate.value.clone().ok_or("evaluate failed")?;

    let visual = export_visual_candidate(
        RunRef {
            context: context.clone(),
            run_id: "sdk_candidate_smoke_run".to_string(),
            out_dir: out.clone(),
        },
        VisualExportConfig {
            context: context.clone(),
            out_dir: out.join("visual_export"),
        },
    );
    let visual_ref = visual.value.clone().ok_or("visual export failed")?;

    let rollback = rollback_candidate(
        RunRef {
            context: context.clone(),
            run_id: "sdk_candidate_smoke_run".to_string(),
            out_dir: out.clone(),
        },
        &loaded_checkpoint,
    );
    let rollback_report = rollback.value.clone().ok_or("rollback failed")?;

    let mut invalid_schema_context = context.clone();
    invalid_schema_context.schema_version = "wrong_schema".to_string();
    append_progress(
        &progress_path,
        "invalid_schema_rejection",
        "start",
        "invalid schema rejection start",
    )?;
    let invalid_schema = infer_candidate(
        &loaded_checkpoint,
        InputBatch {
            context: invalid_schema_context,
            inputs: vec!["route invalid schema".to_string()],
        },
    );
    append_progress(
        &progress_path,
        "invalid_schema_rejection",
        "completed",
        "invalid schema rejection completed",
    )?;

    let mut regulated_context = context.clone();
    regulated_context.intended_use = IntendedUse::Clinical;
    append_progress(
        &progress_path,
        "regulated_use_rejection",
        "start",
        "regulated use rejection start",
    )?;
    let regulated = train_candidate(
        TrainCandidateConfig {
            context: regulated_context,
            out_dir: out.join("regulated_blocked"),
            seed: 1,
        },
        DataRef {
            id: "regulated".to_string(),
            description: "clinical request must be rejected".to_string(),
        },
    );
    append_progress(
        &progress_path,
        "regulated_use_rejection",
        "completed",
        "regulated use rejection completed",
    )?;

    write_json(
        &out.join("sdk_manifest.json"),
        &json!({
            "schema_version": SDK_CANDIDATE_SCHEMA_VERSION,
            "production_default_training_enabled": false,
            "public_beta_promoted": false,
            "production_api_ready": false,
            "claim_boundary": ClaimBoundary::default(),
        }),
    )?;
    write_json(
        &out.join("api_surface_snapshot.json"),
        &json!({
            "schema_version": SDK_CANDIDATE_SCHEMA_VERSION,
            "calls": [
                "train_candidate",
                "infer_candidate",
                "evaluate_candidate",
                "save_checkpoint_candidate",
                "load_checkpoint_candidate",
                "rollback_candidate",
                "export_visual_candidate"
            ]
        }),
    )?;
    write_json(
        &out.join("error_envelope_examples.json"),
        &json!({
            "invalid_schema": invalid_schema.error,
            "regulated_use": regulated.error,
        }),
    )?;
    write_json(
        &out.join("checkpoint_metrics.json"),
        &json!({
            "checkpoint_id": loaded_checkpoint.checkpoint_id,
            "checkpoint_hash": loaded_checkpoint.sha256,
            "checkpoint_save_load_pass": load.ok,
            "checkpoint_hash_algorithm": "SHA-256"
        }),
    )?;
    write_jsonl(&out.join("inference_samples.jsonl"), &inference.samples)?;
    write_json(&out.join("eval_report.json"), &eval_report)?;
    write_json(
        &out.join("visual_export_manifest.json"),
        &json!({
            "schema_version": visual_ref.schema_version,
            "out_dir": visual_ref.out_dir,
            "manifest_path": visual_ref.manifest_path
        }),
    )?;
    fs::write(
        out.join("claim_boundary.md"),
        ClaimBoundary::default().statement,
    )?;

    let positive = train.ok
        && save.ok
        && load.ok
        && infer.ok
        && evaluate.ok
        && visual.ok
        && rollback.ok
        && !invalid_schema.ok
        && !regulated.ok
        && invalid_schema
            .error
            .as_ref()
            .is_some_and(|error| error.code == "UNKNOWN_SCHEMA_VERSION")
        && regulated
            .error
            .as_ref()
            .is_some_and(|error| error.code == "POLICY_GUARD_REJECTED");

    write_json(
        &out.join("summary.json"),
        &json!({
            "schema_version": SDK_CANDIDATE_SCHEMA_VERSION,
            "sdk_release_candidate_gate_pass": positive,
            "production_default_training_enabled": false,
            "public_beta_promoted": false,
            "production_api_ready": false,
            "verdicts": if positive {
                vec![
                    "SDK_RELEASE_CANDIDATE_POSITIVE",
                    "SDK_API_SURFACE_DEFINED",
                    "SDK_ERROR_ENVELOPE_POSITIVE",
                    "SDK_PROGRESS_EVENTS_POSITIVE",
                    "SDK_CHECKPOINT_SAVE_LOAD_POSITIVE",
                    "SDK_INFERENCE_SMOKE_POSITIVE",
                    "SDK_EVALUATION_SMOKE_POSITIVE",
                    "SDK_VISUAL_EXPORT_SMOKE_POSITIVE",
                    "POLICY_GUARD_REJECTS_REGULATED_USE",
                    "PRODUCTION_READY_NOT_CLAIMED"
                ]
            } else {
                vec!["SDK_RELEASE_CANDIDATE_FAILS"]
            },
            "rollback_success": rollback_report.rollback_success,
        }),
    )?;
    fs::write(
        out.join("report.md"),
        format!(
            "# STABLE_LOOP_PHASE_LOCK_057_INSTNCT_SDK_RELEASE_CANDIDATE Report\n\nStatus: {}.\n\nThis is SDK release-candidate engineering only. This is not production API readiness. This is not a new model/training result. This is not public beta.\n\nCheckpoint hash algorithm: SHA-256.\n\nVisual export schema: visual_snapshot_v1.\n\n",
            if positive { "positive" } else { "failed" }
        ),
    )?;

    if !positive {
        return Err("SDK candidate smoke failed".into());
    }
    println!("instnct_sdk_candidate_smoke_out={}", out.display());
    Ok(())
}

fn parse_out() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut out = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out = args.next().map(PathBuf::from),
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    Ok(out.unwrap_or_else(|| {
        PathBuf::from(
            "target/pilot_wave/stable_loop_phase_lock_057_instnct_sdk_release_candidate/smoke",
        )
    }))
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn write_jsonl<T: Serialize>(path: &Path, rows: &[T]) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    for row in rows {
        serde_json::to_writer(&mut file, row)?;
        file.write_all(b"\n")?;
    }
    Ok(())
}
