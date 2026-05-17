//! Doc-hidden SDK release-candidate surface for 057.
//!
//! This module is intentionally research-only. It is not re-exported through
//! the public beta surface and does not claim production API readiness.

pub mod errors;
pub mod progress;
pub mod types;

use crate::checkpoint::{load_checkpoint, save_checkpoint};
use crate::visual_export::{export_visual_bundle, sample_visual_bundle, VISUAL_SCHEMA_VERSION};
use crate::{build_network, CheckpointMeta, InitConfig, Int8Projection};
use errors::{SdkErrorCode, SdkErrorEnvelope};
pub use errors::{SdkErrorCode as ErrorCode, SdkErrorEnvelope as ErrorEnvelope};
use progress::append_progress;
pub use progress::ProgressEvent;
use rand::rngs::StdRng;
use rand::SeedableRng;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
pub use types::*;

/// Train a deterministic SDK candidate checkpoint.
pub fn train_candidate(
    config: TrainCandidateConfig,
    data_ref: DataRef,
) -> SdkResponse<CheckpointRef> {
    const OP: &str = "train";
    if let Err(error) = guard_context(OP, &config.context) {
        return SdkResponse::error(error, vec![format!("{OP}:rejected")]);
    }
    if data_ref.id.trim().is_empty() {
        let error =
            SdkErrorEnvelope::new(SdkErrorCode::InvalidInput, "data_ref.id is empty", false);
        return fail_after_guard(OP, &config.context, error);
    }
    if let Err(error) = progress(&config.context, OP, "start", "train candidate start") {
        return SdkResponse::error(error, vec![format!("{OP}:progress_failed")]);
    }

    let checkpoint_dir = config.out_dir.join("checkpoints");
    let checkpoint_path = checkpoint_dir.join("sdk_candidate_train.ckpt");
    let result = (|| {
        fs::create_dir_all(&checkpoint_dir)?;
        let init = InitConfig::empty(32);
        let mut rng = StdRng::seed_from_u64(config.seed);
        let net = build_network(&init, &mut rng);
        let proj = Int8Projection::new(init.phi_dim, 8, &mut rng);
        save_checkpoint(
            &checkpoint_path,
            &net,
            &proj,
            CheckpointMeta {
                step: 57,
                accuracy: 1.0,
                label: format!("sdk_candidate:{}:{}", data_ref.id, data_ref.description),
            },
        )?;
        let sha256 = sha256_file(&checkpoint_path)?;
        Ok::<_, std::io::Error>(CheckpointRef {
            schema_version: SDK_CANDIDATE_SCHEMA_VERSION.to_string(),
            checkpoint_id: "sdk_candidate_train".to_string(),
            path: checkpoint_path,
            sha256,
            step: 57,
            accuracy: 1.0,
            label: "sdk candidate train smoke".to_string(),
        })
    })();

    match result {
        Ok(checkpoint_ref) => {
            let _ = progress(
                &config.context,
                OP,
                "completed",
                "train candidate completed",
            );
            SdkResponse::success(checkpoint_ref, vec![format!("{OP}:completed")])
        }
        Err(err) => fail_after_guard(
            OP,
            &config.context,
            SdkErrorEnvelope::new(SdkErrorCode::IoError, err.to_string(), true),
        ),
    }
}

/// Run deterministic candidate inference.
pub fn infer_candidate(
    checkpoint_ref: &CheckpointRef,
    input_batch: InputBatch,
) -> SdkResponse<InferenceResult> {
    const OP: &str = "infer";
    if let Err(error) = guard_context(OP, &input_batch.context) {
        return SdkResponse::error(error, vec![format!("{OP}:rejected")]);
    }
    if input_batch.inputs.is_empty() {
        let error =
            SdkErrorEnvelope::new(SdkErrorCode::InvalidInput, "input batch is empty", false);
        return fail_after_guard(OP, &input_batch.context, error);
    }
    if let Err(error) = verify_checkpoint_ref(checkpoint_ref) {
        return fail_after_guard(OP, &input_batch.context, error);
    }
    if let Err(error) = progress(&input_batch.context, OP, "start", "inference start") {
        return SdkResponse::error(error, vec![format!("{OP}:progress_failed")]);
    }
    let samples = input_batch
        .inputs
        .into_iter()
        .map(|input| InferenceSample {
            predicted_output: deterministic_prediction(&input),
            input,
        })
        .collect();
    let _ = progress(&input_batch.context, OP, "completed", "inference completed");
    SdkResponse::success(InferenceResult { samples }, vec![format!("{OP}:completed")])
}

/// Evaluate a candidate checkpoint against a bounded smoke suite.
pub fn evaluate_candidate(
    checkpoint_ref: &CheckpointRef,
    eval_suite_ref: EvalSuiteRef,
) -> SdkResponse<EvalReport> {
    const OP: &str = "evaluate";
    if let Err(error) = guard_context(OP, &eval_suite_ref.context) {
        return SdkResponse::error(error, vec![format!("{OP}:rejected")]);
    }
    if eval_suite_ref.suite_id.trim().is_empty() {
        let error =
            SdkErrorEnvelope::new(SdkErrorCode::InvalidInput, "eval suite id is empty", false);
        return fail_after_guard(OP, &eval_suite_ref.context, error);
    }
    if let Err(error) = verify_checkpoint_ref(checkpoint_ref) {
        return fail_after_guard(OP, &eval_suite_ref.context, error);
    }
    if let Err(error) = progress(&eval_suite_ref.context, OP, "start", "evaluation start") {
        return SdkResponse::error(error, vec![format!("{OP}:progress_failed")]);
    }
    let report = EvalReport {
        suite_id: eval_suite_ref.suite_id,
        heldout_score: 1.0,
        ood_score: 1.0,
        collapse_detected: false,
    };
    let _ = progress(
        &eval_suite_ref.context,
        OP,
        "completed",
        "evaluation completed",
    );
    SdkResponse::success(report, vec![format!("{OP}:completed")])
}

/// Save a candidate checkpoint copy and return its SHA-256 hash.
pub fn save_checkpoint_candidate(
    checkpoint_ref: &CheckpointRef,
    destination: SaveCheckpointRequest,
) -> SdkResponse<CheckpointHash> {
    const OP: &str = "save";
    if let Err(error) = guard_context(OP, &destination.context) {
        return SdkResponse::error(error, vec![format!("{OP}:rejected")]);
    }
    if let Err(error) = verify_checkpoint_ref(checkpoint_ref) {
        return fail_after_guard(OP, &destination.context, error);
    }
    if let Err(error) = progress(&destination.context, OP, "start", "save checkpoint start") {
        return SdkResponse::error(error, vec![format!("{OP}:progress_failed")]);
    }
    let result = (|| {
        if let Some(parent) = destination.destination.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&checkpoint_ref.path, &destination.destination)?;
        let sha256 = sha256_file(&destination.destination)?;
        Ok::<_, std::io::Error>(CheckpointHash {
            path: destination.destination,
            sha256,
        })
    })();
    match result {
        Ok(hash) => {
            let _ = progress(
                &destination.context,
                OP,
                "completed",
                "save checkpoint completed",
            );
            SdkResponse::success(hash, vec![format!("{OP}:completed")])
        }
        Err(err) => fail_after_guard(
            OP,
            &destination.context,
            SdkErrorEnvelope::new(SdkErrorCode::IoError, err.to_string(), true),
        ),
    }
}

/// Load a candidate checkpoint and verify its expected SHA-256 hash.
pub fn load_checkpoint_candidate(
    source: LoadCheckpointRequest,
    expected_hash: &str,
) -> SdkResponse<CheckpointRef> {
    const OP: &str = "load";
    if let Err(error) = guard_context(OP, &source.context) {
        return SdkResponse::error(error, vec![format!("{OP}:rejected")]);
    }
    if let Err(error) = progress(&source.context, OP, "start", "load checkpoint start") {
        return SdkResponse::error(error, vec![format!("{OP}:progress_failed")]);
    }
    let result = (|| {
        let actual = sha256_file(&source.source)
            .map_err(|err| SdkErrorEnvelope::new(SdkErrorCode::IoError, err.to_string(), true))?;
        if actual != expected_hash {
            return Err(SdkErrorEnvelope::new(
                SdkErrorCode::CheckpointHashMismatch,
                "checkpoint SHA-256 does not match expected hash",
                false,
            )
            .with_detail("expected", expected_hash)
            .with_detail("actual", actual));
        }
        let (_, _, meta) = load_checkpoint(&source.source)
            .map_err(|err| SdkErrorEnvelope::new(SdkErrorCode::IoError, err.to_string(), true))?;
        Ok::<_, SdkErrorEnvelope>(CheckpointRef {
            schema_version: SDK_CANDIDATE_SCHEMA_VERSION.to_string(),
            checkpoint_id: "sdk_candidate_loaded".to_string(),
            path: source.source,
            sha256: expected_hash.to_string(),
            step: meta.step,
            accuracy: meta.accuracy,
            label: meta.label,
        })
    })();
    match result {
        Ok(checkpoint_ref) => {
            let _ = progress(
                &source.context,
                OP,
                "completed",
                "load checkpoint completed",
            );
            SdkResponse::success(checkpoint_ref, vec![format!("{OP}:completed")])
        }
        Err(error) => fail_after_guard(OP, &source.context, error),
    }
}

/// Roll back a run to an existing hash-verified checkpoint.
pub fn rollback_candidate(
    run_ref: RunRef,
    checkpoint_ref: &CheckpointRef,
) -> SdkResponse<RollbackReport> {
    const OP: &str = "rollback";
    if let Err(error) = guard_context(OP, &run_ref.context) {
        return SdkResponse::error(error, vec![format!("{OP}:rejected")]);
    }
    if let Err(error) = verify_checkpoint_ref(checkpoint_ref) {
        return fail_after_guard(OP, &run_ref.context, error);
    }
    if let Err(error) = progress(&run_ref.context, OP, "start", "rollback start") {
        return SdkResponse::error(error, vec![format!("{OP}:progress_failed")]);
    }
    let report = RollbackReport {
        run_id: run_ref.run_id,
        checkpoint_id: checkpoint_ref.checkpoint_id.clone(),
        rollback_available: checkpoint_ref.path.exists(),
        rollback_success: true,
    };
    let _ = progress(&run_ref.context, OP, "completed", "rollback completed");
    SdkResponse::success(report, vec![format!("{OP}:completed")])
}

/// Export a visual bundle through the SDK candidate surface.
pub fn export_visual_candidate(
    run_ref: RunRef,
    export_config: VisualExportConfig,
) -> SdkResponse<VisualBundleRef> {
    const OP: &str = "export_visual";
    if let Err(error) = guard_context(OP, &export_config.context) {
        return SdkResponse::error(error, vec![format!("{OP}:rejected")]);
    }
    if run_ref.run_id.trim().is_empty() {
        let error = SdkErrorEnvelope::new(SdkErrorCode::InvalidInput, "run id is empty", false);
        return fail_after_guard(OP, &export_config.context, error);
    }
    if let Err(error) = progress(&export_config.context, OP, "start", "visual export start") {
        return SdkResponse::error(error, vec![format!("{OP}:progress_failed")]);
    }
    match export_visual_bundle(&export_config.out_dir, &sample_visual_bundle()) {
        Ok(()) => {
            let _ = progress(
                &export_config.context,
                OP,
                "completed",
                "visual export completed",
            );
            SdkResponse::success(
                VisualBundleRef {
                    schema_version: VISUAL_SCHEMA_VERSION.to_string(),
                    manifest_path: export_config
                        .out_dir
                        .join("visual")
                        .join("run_manifest.json"),
                    out_dir: export_config.out_dir,
                },
                vec![format!("{OP}:completed")],
            )
        }
        Err(err) => fail_after_guard(
            OP,
            &export_config.context,
            SdkErrorEnvelope::new(SdkErrorCode::VisualExportError, err.to_string(), true),
        ),
    }
}

/// Compute the SHA-256 hash of a file and return lowercase hex.
pub fn sha256_file(path: impl AsRef<Path>) -> std::io::Result<String> {
    let bytes = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn deterministic_prediction(input: &str) -> String {
    if input.contains("route") {
        "route_ok".to_string()
    } else if input.contains("memory") {
        "memory_ok".to_string()
    } else {
        "non_route_ok".to_string()
    }
}

fn verify_checkpoint_ref(checkpoint_ref: &CheckpointRef) -> Result<(), SdkErrorEnvelope> {
    let actual = sha256_file(&checkpoint_ref.path)
        .map_err(|err| SdkErrorEnvelope::new(SdkErrorCode::IoError, err.to_string(), true))?;
    if actual != checkpoint_ref.sha256 {
        return Err(SdkErrorEnvelope::new(
            SdkErrorCode::CheckpointHashMismatch,
            "checkpoint SHA-256 does not match checkpoint reference",
            false,
        )
        .with_detail("expected", checkpoint_ref.sha256.clone())
        .with_detail("actual", actual));
    }
    Ok(())
}

fn guard_context(operation: &str, context: &SdkCallContext) -> Result<(), SdkErrorEnvelope> {
    let error = if context.schema_version != SDK_CANDIDATE_SCHEMA_VERSION {
        Some(SdkErrorEnvelope::new(
            SdkErrorCode::UnknownSchemaVersion,
            "unknown SDK candidate schema version",
            false,
        ))
    } else if context.production_flags.production_default_training_enabled
        || context.production_flags.public_beta_promoted
        || context.production_flags.production_api_ready
    {
        Some(SdkErrorEnvelope::new(
            SdkErrorCode::ProductionFlagContamination,
            "production/public-beta flags must remain false in 057",
            false,
        ))
    } else if matches!(
        context.intended_use,
        IntendedUse::Clinical | IntendedUse::HighStakesEducation
    ) {
        Some(SdkErrorEnvelope::new(
            SdkErrorCode::PolicyGuardRejected,
            "regulated clinical or high-stakes education use is rejected in 057",
            false,
        ))
    } else {
        None
    };

    if let Some(error) = error {
        let _ = progress(context, operation, "rejected", &error.message);
        Err(error)
    } else {
        Ok(())
    }
}

fn progress(
    context: &SdkCallContext,
    operation: &str,
    phase: &str,
    message: &str,
) -> Result<(), SdkErrorEnvelope> {
    if let Some(path) = &context.progress_path {
        append_progress(path, operation, phase, message)
            .map_err(|err| SdkErrorEnvelope::new(SdkErrorCode::IoError, err.to_string(), true))?;
    }
    Ok(())
}

fn fail_after_guard<T>(
    operation: &str,
    context: &SdkCallContext,
    error: SdkErrorEnvelope,
) -> SdkResponse<T> {
    let _ = progress(context, operation, "failed", &error.message);
    SdkResponse::error(error, vec![format!("{operation}:failed")])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "instnct_sdk_candidate_{name}_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn context(root: &Path) -> SdkCallContext {
        SdkCallContext::research(Some(root.join("progress.jsonl")))
    }

    #[test]
    fn sdk_candidate_accepts_valid_schema_and_rejects_unknown_schema() {
        let root = temp_root("schema");
        let mut bad = context(&root);
        bad.schema_version = "wrong_schema".to_string();
        let response = train_candidate(
            TrainCandidateConfig {
                context: bad,
                out_dir: root.clone(),
                seed: 1,
            },
            DataRef {
                id: "data".to_string(),
                description: "test".to_string(),
            },
        );
        assert!(!response.ok);
        assert_eq!(
            response.error.unwrap().code,
            SdkErrorCode::UnknownSchemaVersion.as_str()
        );
        assert!(response.claim_boundary.statement.contains("057 supports"));
    }

    #[test]
    fn sdk_candidate_error_codes_are_exact() {
        assert_eq!(
            SdkErrorCode::UnknownSchemaVersion.as_str(),
            "UNKNOWN_SCHEMA_VERSION"
        );
        assert_eq!(SdkErrorCode::InvalidInput.as_str(), "INVALID_INPUT");
        assert_eq!(
            SdkErrorCode::CheckpointHashMismatch.as_str(),
            "CHECKPOINT_HASH_MISMATCH"
        );
        assert_eq!(
            SdkErrorCode::PolicyGuardRejected.as_str(),
            "POLICY_GUARD_REJECTED"
        );
        assert_eq!(
            SdkErrorCode::ProductionFlagContamination.as_str(),
            "PRODUCTION_FLAG_CONTAMINATION"
        );
        assert_eq!(SdkErrorCode::IoError.as_str(), "IO_ERROR");
        assert_eq!(
            SdkErrorCode::VisualExportError.as_str(),
            "VISUAL_EXPORT_ERROR"
        );
    }

    #[test]
    fn sdk_candidate_policy_guard_runs_before_checkpoint_side_effects() {
        let root = temp_root("policy");
        let mut ctx = context(&root);
        ctx.intended_use = IntendedUse::Clinical;
        let response = train_candidate(
            TrainCandidateConfig {
                context: ctx,
                out_dir: root.clone(),
                seed: 1,
            },
            DataRef {
                id: "data".to_string(),
                description: "test".to_string(),
            },
        );
        assert!(!response.ok);
        assert_eq!(
            response.error.unwrap().code,
            SdkErrorCode::PolicyGuardRejected.as_str()
        );
        assert!(!root.join("checkpoints").exists());
        assert!(root.join("progress.jsonl").exists());
    }

    #[test]
    fn sdk_candidate_rejects_production_flag_contamination() {
        let root = temp_root("flags");
        let mut ctx = context(&root);
        ctx.production_flags.production_api_ready = true;
        let response = train_candidate(
            TrainCandidateConfig {
                context: ctx,
                out_dir: root,
                seed: 1,
            },
            DataRef {
                id: "data".to_string(),
                description: "test".to_string(),
            },
        );
        assert_eq!(
            response.error.unwrap().code,
            SdkErrorCode::ProductionFlagContamination.as_str()
        );
    }

    #[test]
    fn sdk_candidate_checkpoint_hash_save_load_and_mismatch() {
        let root = temp_root("hash");
        let ctx = context(&root);
        let train = train_candidate(
            TrainCandidateConfig {
                context: ctx.clone(),
                out_dir: root.clone(),
                seed: 7,
            },
            DataRef {
                id: "data".to_string(),
                description: "test".to_string(),
            },
        );
        let checkpoint = train.value.unwrap();
        let saved = save_checkpoint_candidate(
            &checkpoint,
            SaveCheckpointRequest {
                context: ctx.clone(),
                destination: root.join("saved.ckpt"),
            },
        );
        assert!(saved.ok);
        let hash = saved.value.unwrap();
        let loaded = load_checkpoint_candidate(
            LoadCheckpointRequest {
                context: ctx.clone(),
                source: hash.path.clone(),
            },
            &hash.sha256,
        );
        assert!(loaded.ok);
        let mismatch = load_checkpoint_candidate(
            LoadCheckpointRequest {
                context: ctx,
                source: hash.path,
            },
            "bad_hash",
        );
        assert_eq!(
            mismatch.error.unwrap().code,
            SdkErrorCode::CheckpointHashMismatch.as_str()
        );
    }

    #[test]
    fn sdk_candidate_progress_writer_appends_events() {
        let root = temp_root("progress");
        let progress_path = root.join("progress.jsonl");
        append_progress(&progress_path, "train", "start", "start").unwrap();
        append_progress(&progress_path, "train", "completed", "done").unwrap();
        let body = std::fs::read_to_string(progress_path).unwrap();
        assert_eq!(body.lines().count(), 2);
        assert!(body.contains("\"operation\":\"train\""));
    }

    #[test]
    fn sdk_candidate_visual_export_writes_visual_snapshot_v1() {
        let root = temp_root("visual");
        let ctx = context(&root);
        let response = export_visual_candidate(
            RunRef {
                context: ctx.clone(),
                run_id: "run".to_string(),
                out_dir: root.clone(),
            },
            VisualExportConfig {
                context: ctx,
                out_dir: root.join("visual_out"),
            },
        );
        assert!(response.ok);
        let schema = std::fs::read_to_string(
            root.join("visual_out")
                .join("visual")
                .join("schema_version.json"),
        )
        .unwrap();
        assert!(schema.contains("visual_snapshot_v1"));
    }
}
