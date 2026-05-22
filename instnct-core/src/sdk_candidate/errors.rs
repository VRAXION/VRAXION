//! Structured error envelopes for the 057 SDK candidate.

use crate::sdk_candidate::types::{ClaimBoundary, SDK_CANDIDATE_SCHEMA_VERSION};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Exact error codes emitted by the SDK candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SdkErrorCode {
    /// Request schema did not match the SDK candidate schema.
    UnknownSchemaVersion,
    /// Request shape or payload was invalid.
    InvalidInput,
    /// Checkpoint hash did not match the expected SHA-256.
    CheckpointHashMismatch,
    /// Intended use was rejected by the 056 policy boundary.
    PolicyGuardRejected,
    /// Production/public beta flags were set when they must remain false.
    ProductionFlagContamination,
    /// Filesystem or checkpoint IO failed.
    IoError,
    /// Visual export failed.
    VisualExportError,
}

impl SdkErrorCode {
    /// Return the exact wire code for this error.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UnknownSchemaVersion => "UNKNOWN_SCHEMA_VERSION",
            Self::InvalidInput => "INVALID_INPUT",
            Self::CheckpointHashMismatch => "CHECKPOINT_HASH_MISMATCH",
            Self::PolicyGuardRejected => "POLICY_GUARD_REJECTED",
            Self::ProductionFlagContamination => "PRODUCTION_FLAG_CONTAMINATION",
            Self::IoError => "IO_ERROR",
            Self::VisualExportError => "VISUAL_EXPORT_ERROR",
        }
    }
}

/// Structured SDK candidate error envelope.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SdkErrorEnvelope {
    /// SDK candidate schema version.
    pub schema_version: String,
    /// Exact machine-readable error code.
    pub code: String,
    /// Human-readable error message.
    pub message: String,
    /// Whether the caller can retry without changing request semantics.
    pub retryable: bool,
    /// Optional structured details.
    pub details: BTreeMap<String, String>,
    /// Claim boundary attached to every error.
    pub claim_boundary: ClaimBoundary,
}

impl SdkErrorEnvelope {
    /// Create a new error envelope.
    pub fn new(code: SdkErrorCode, message: impl Into<String>, retryable: bool) -> Self {
        Self {
            schema_version: SDK_CANDIDATE_SCHEMA_VERSION.to_string(),
            code: code.as_str().to_string(),
            message: message.into(),
            retryable,
            details: BTreeMap::new(),
            claim_boundary: ClaimBoundary::default(),
        }
    }

    /// Attach one detail key/value pair.
    pub fn with_detail(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.details.insert(key.into(), value.into());
        self
    }
}
