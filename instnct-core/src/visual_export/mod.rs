//! Research-only visual export helpers.
//!
//! This module is intentionally doc-hidden from the public beta surface. It
//! normalizes run snapshots into a renderer-agnostic visual schema for tools
//! and reviewer labs.

pub mod exporter;
pub mod ingest;
pub mod larger_playback;
pub mod schema;

pub use exporter::{export_visual_bundle, write_json, write_jsonl, VisualExportError};
pub use ingest::bundle_from_049_adversarial_run;
pub use larger_playback::larger_playback_visual_bundle;
pub use schema::{
    sample_visual_bundle, CheckpointIndexRow, EdgeRole, EventKind, GraphEdge, GraphNode,
    GraphSnapshot, MetricRow, MutationEvent, NodeRole, PocketSummary, RouteTrace, RunManifest,
    SchemaVersion, TickSnapshot, VisualBundle, VisualMetadata, VISUAL_SCHEMA_VERSION,
};
