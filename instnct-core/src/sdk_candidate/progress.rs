//! Append-only progress writer for the 057 SDK candidate.

use crate::sdk_candidate::types::SDK_CANDIDATE_SCHEMA_VERSION;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// One append-only SDK progress event.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProgressEvent {
    /// SDK candidate schema version.
    pub schema_version: String,
    /// Operation name.
    pub operation: String,
    /// Operation phase.
    pub phase: String,
    /// Event message.
    pub message: String,
    /// Wall-clock event timestamp in milliseconds since Unix epoch.
    pub timestamp_ms: u128,
}

impl ProgressEvent {
    /// Create a progress event with current wall-clock timestamp.
    pub fn new(
        operation: impl Into<String>,
        phase: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis())
            .unwrap_or(0);
        Self {
            schema_version: SDK_CANDIDATE_SCHEMA_VERSION.to_string(),
            operation: operation.into(),
            phase: phase.into(),
            message: message.into(),
            timestamp_ms,
        }
    }
}

/// Append one event to a JSONL progress file.
pub fn append_progress(
    path: impl AsRef<Path>,
    operation: &str,
    phase: &str,
    message: &str,
) -> io::Result<ProgressEvent> {
    let event = ProgressEvent::new(operation, phase, message);
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, &event).map_err(io::Error::other)?;
    file.write_all(b"\n")?;
    Ok(event)
}
