//! Public candidate types for the doc-hidden 057 SDK module.

use crate::sdk_candidate::errors::SdkErrorEnvelope;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// SDK candidate schema version.
pub const SDK_CANDIDATE_SCHEMA_VERSION: &str = "instnct_sdk_candidate_v1";

/// Claim boundary emitted by every SDK response.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimBoundary {
    /// Boundary statement.
    pub statement: String,
    /// Production API readiness claim.
    pub production_api_ready: bool,
    /// Public beta promotion claim.
    pub public_beta_promoted: bool,
    /// Clinical readiness claim.
    pub clinical_ready: bool,
    /// High-stakes education readiness claim.
    pub high_stakes_education_ready: bool,
    /// Full VRAXION claim.
    pub full_vraxion_claimed: bool,
    /// Language grounding claim.
    pub language_grounding_claimed: bool,
    /// Consciousness claim.
    pub consciousness_claimed: bool,
    /// Biological/FlyWire equivalence claim.
    pub biological_flywire_equivalence_claimed: bool,
    /// Physical quantum behavior claim.
    pub physical_quantum_behavior_claimed: bool,
}

impl Default for ClaimBoundary {
    fn default() -> Self {
        Self {
            statement: "057 supports SDK release-candidate engineering only; it is not production API readiness, public beta, clinical use, high-stakes education use, full VRAXION, language grounding, consciousness, biological/FlyWire equivalence, or physical quantum behavior.".to_string(),
            production_api_ready: false,
            public_beta_promoted: false,
            clinical_ready: false,
            high_stakes_education_ready: false,
            full_vraxion_claimed: false,
            language_grounding_claimed: false,
            consciousness_claimed: false,
            biological_flywire_equivalence_claimed: false,
            physical_quantum_behavior_claimed: false,
        }
    }
}

/// Structured SDK response envelope.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SdkResponse<T> {
    /// SDK candidate schema version.
    pub schema_version: String,
    /// Whether the call succeeded.
    pub ok: bool,
    /// Success value.
    pub value: Option<T>,
    /// Error envelope.
    pub error: Option<SdkErrorEnvelope>,
    /// Audit events emitted by the operation.
    pub audit_events: Vec<String>,
    /// Claim boundary attached to every response.
    pub claim_boundary: ClaimBoundary,
}

impl<T> SdkResponse<T> {
    /// Create a success response.
    pub fn success(value: T, audit_events: Vec<String>) -> Self {
        Self {
            schema_version: SDK_CANDIDATE_SCHEMA_VERSION.to_string(),
            ok: true,
            value: Some(value),
            error: None,
            audit_events,
            claim_boundary: ClaimBoundary::default(),
        }
    }

    /// Create an error response.
    pub fn error(error: SdkErrorEnvelope, audit_events: Vec<String>) -> Self {
        Self {
            schema_version: SDK_CANDIDATE_SCHEMA_VERSION.to_string(),
            ok: false,
            value: None,
            error: Some(error),
            audit_events,
            claim_boundary: ClaimBoundary::default(),
        }
    }
}

/// Production flags that must remain false in 057.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProductionFlags {
    /// Whether production default training is enabled.
    pub production_default_training_enabled: bool,
    /// Whether public beta has been promoted.
    pub public_beta_promoted: bool,
    /// Whether production API readiness is claimed.
    pub production_api_ready: bool,
}

/// Intended-use category for policy guard checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntendedUse {
    /// Bounded internal research or reproduction.
    Research,
    /// Private technical evaluation.
    InternalEvaluation,
    /// Clinical use that must be rejected in 057.
    Clinical,
    /// High-stakes education use that must be rejected in 057.
    HighStakesEducation,
}

/// Shared call context.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdkCallContext {
    /// Request schema version.
    pub schema_version: String,
    /// Intended use for policy guard.
    pub intended_use: IntendedUse,
    /// Production/public-beta flags.
    pub production_flags: ProductionFlags,
    /// Optional append-only progress file.
    pub progress_path: Option<PathBuf>,
}

impl SdkCallContext {
    /// Create a research-safe context.
    pub fn research(progress_path: Option<PathBuf>) -> Self {
        Self {
            schema_version: SDK_CANDIDATE_SCHEMA_VERSION.to_string(),
            intended_use: IntendedUse::Research,
            production_flags: ProductionFlags::default(),
            progress_path,
        }
    }
}

/// Reference to input data for candidate training.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataRef {
    /// Data reference identifier.
    pub id: String,
    /// Human-readable data description.
    pub description: String,
}

/// Candidate training config.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainCandidateConfig {
    /// Shared call context.
    pub context: SdkCallContext,
    /// Output directory.
    pub out_dir: PathBuf,
    /// Deterministic seed.
    pub seed: u64,
}

/// Reference to a candidate checkpoint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CheckpointRef {
    /// SDK candidate schema version.
    pub schema_version: String,
    /// Stable checkpoint id.
    pub checkpoint_id: String,
    /// Checkpoint file path.
    pub path: PathBuf,
    /// SHA-256 hash of checkpoint bytes.
    pub sha256: String,
    /// Stored step.
    pub step: usize,
    /// Stored accuracy.
    pub accuracy: f64,
    /// Stored label.
    pub label: String,
}

/// SHA-256 hash result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointHash {
    /// Checkpoint file path.
    pub path: PathBuf,
    /// SHA-256 hash of checkpoint bytes.
    pub sha256: String,
}

/// Save checkpoint request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SaveCheckpointRequest {
    /// Shared call context.
    pub context: SdkCallContext,
    /// Destination checkpoint path.
    pub destination: PathBuf,
}

/// Load checkpoint request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoadCheckpointRequest {
    /// Shared call context.
    pub context: SdkCallContext,
    /// Source checkpoint path.
    pub source: PathBuf,
}

/// Batch of inference inputs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputBatch {
    /// Shared call context.
    pub context: SdkCallContext,
    /// Input strings.
    pub inputs: Vec<String>,
}

/// One deterministic inference sample.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceSample {
    /// Input string.
    pub input: String,
    /// Predicted output.
    pub predicted_output: String,
}

/// Inference result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceResult {
    /// Inference samples.
    pub samples: Vec<InferenceSample>,
}

/// Evaluation suite reference.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalSuiteRef {
    /// Shared call context.
    pub context: SdkCallContext,
    /// Evaluation suite id.
    pub suite_id: String,
}

/// Evaluation report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    /// Evaluation suite id.
    pub suite_id: String,
    /// Heldout score.
    pub heldout_score: f64,
    /// OOD score.
    pub ood_score: f64,
    /// Collapse flag.
    pub collapse_detected: bool,
}

/// Run reference.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunRef {
    /// Shared call context.
    pub context: SdkCallContext,
    /// Run identifier.
    pub run_id: String,
    /// Run output directory.
    pub out_dir: PathBuf,
}

/// Rollback report.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RollbackReport {
    /// Run identifier.
    pub run_id: String,
    /// Checkpoint id selected for rollback.
    pub checkpoint_id: String,
    /// Whether rollback is available.
    pub rollback_available: bool,
    /// Whether rollback hash verification succeeded.
    pub rollback_success: bool,
}

/// Visual export config.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisualExportConfig {
    /// Shared call context.
    pub context: SdkCallContext,
    /// Visual output directory.
    pub out_dir: PathBuf,
}

/// Visual bundle reference.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisualBundleRef {
    /// Visual schema version.
    pub schema_version: String,
    /// Visual output directory.
    pub out_dir: PathBuf,
    /// Manifest path.
    pub manifest_path: PathBuf,
}
