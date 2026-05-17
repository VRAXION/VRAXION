//! Visual bundle writer.

use crate::visual_export::schema::{GraphSnapshot, TickSnapshot, VisualBundle};
use serde::Serialize;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;

/// Error returned by the visual exporter.
#[derive(Debug)]
pub enum VisualExportError {
    /// Filesystem error.
    Io(std::io::Error),
    /// Serialization error.
    Serde(serde_json::Error),
}

impl fmt::Display for VisualExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "visual export IO error: {err}"),
            Self::Serde(err) => write!(f, "visual export serialization error: {err}"),
        }
    }
}

impl std::error::Error for VisualExportError {}

impl From<std::io::Error> for VisualExportError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for VisualExportError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serde(value)
    }
}

/// Export a complete visual bundle under `out/visual`.
pub fn export_visual_bundle(out: &Path, bundle: &VisualBundle) -> Result<(), VisualExportError> {
    let visual = out.join("visual");
    fs::create_dir_all(visual.join("graph"))?;
    fs::create_dir_all(visual.join("ticks"))?;

    write_json(&visual.join("schema_version.json"), &bundle.schema)?;
    write_json(&visual.join("run_manifest.json"), &bundle.manifest)?;
    write_jsonl(
        &visual.join("checkpoint_index.jsonl"),
        &bundle.checkpoint_index,
    )?;
    write_jsonl(&visual.join("metrics.jsonl"), &bundle.metrics)?;
    write_jsonl(&visual.join("mutation_events.jsonl"), &bundle.events)?;
    write_jsonl(&visual.join("route_traces.jsonl"), &bundle.route_traces)?;
    write_jsonl(
        &visual.join("pocket_summaries.jsonl"),
        &bundle.pocket_summaries,
    )?;

    for graph in &bundle.graphs {
        write_json(&visual.join(graph_path(graph)), graph)?;
    }
    for tick in &bundle.ticks {
        write_json(&visual.join(tick_path(tick)), tick)?;
    }
    Ok(())
}

/// Write a pretty JSON file.
pub fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), VisualExportError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    Ok(())
}

/// Write a JSONL file.
pub fn write_jsonl<T: Serialize>(path: &Path, rows: &[T]) -> Result<(), VisualExportError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)?;
    for row in rows {
        serde_json::to_writer(&mut file, row)?;
        file.write_all(b"\n")?;
    }
    Ok(())
}

fn graph_path(graph: &GraphSnapshot) -> String {
    format!("graph/checkpoint_{:03}.json", graph.checkpoint)
}

fn tick_path(tick: &TickSnapshot) -> String {
    format!(
        "ticks/checkpoint_{:03}_tick_{:03}.json",
        tick.checkpoint,
        tick.tick.unwrap_or(0)
    )
}
