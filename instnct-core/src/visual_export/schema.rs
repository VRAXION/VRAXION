//! Versioned visual snapshot schema.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

/// Current visual snapshot schema identifier.
pub const VISUAL_SCHEMA_VERSION: &str = "visual_snapshot_v1";

/// Free-form metadata map used for optional, renderer-agnostic extensions.
pub type VisualMetadata = BTreeMap<String, Value>;

/// Visual role for graph nodes.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeRole {
    /// Main highway backbone node.
    Highway,
    /// Side-pocket node.
    Pocket,
    /// Route source node.
    Source,
    /// Route target node.
    Target,
    /// Relay node.
    Relay,
    /// Candidate node.
    Candidate,
}

/// Visual role for graph edges.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeRole {
    /// Highway backbone edge.
    Highway,
    /// In-pocket edge.
    Pocket,
    /// Pocket-to-highway bridge.
    Bridge,
    /// Candidate mutable edge.
    Candidate,
    /// Pruned edge.
    Pruned,
}

/// Event kind for replay and diff timelines.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    /// A candidate edge or node was introduced.
    Mutation,
    /// An edge or candidate was pruned.
    Prune,
    /// A missing successor was repaired.
    Repair,
    /// A route edge was crystallized.
    Crystallize,
}

/// Schema version file.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SchemaVersion {
    /// Schema identifier.
    pub schema_version: String,
    /// Viewer-supported schema identifiers.
    pub viewer_supported_versions: Vec<String>,
}

/// Run-level manifest.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RunManifest {
    /// Schema identifier.
    pub schema_version: String,
    /// Stable run identifier.
    pub run_id: String,
    /// Human-readable run label.
    pub label: String,
    /// Checkpoints available in this bundle.
    pub checkpoints: Vec<u32>,
    /// Whether tick snapshots are available.
    pub has_ticks: bool,
    /// Boundary text for this visual package.
    pub claim_boundary: String,
}

/// Checkpoint index row written as JSONL.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CheckpointIndexRow {
    /// Schema identifier.
    pub schema_version: String,
    /// Stable run identifier.
    pub run_id: String,
    /// Checkpoint number.
    pub checkpoint: u32,
    /// Graph snapshot path relative to the visual directory.
    pub graph_path: String,
    /// Tick snapshot paths relative to the visual directory.
    pub tick_paths: Vec<String>,
    /// Short checkpoint summary.
    pub summary: String,
}

/// Node in a graph snapshot.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GraphNode {
    /// Stable node id.
    pub id: String,
    /// Display label.
    pub label: String,
    /// Semantic node role.
    pub role: NodeRole,
    /// Optional pocket id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pocket_id: Option<String>,
    /// Layout x coordinate.
    pub x: f64,
    /// Layout y coordinate.
    pub y: f64,
    /// Activity score.
    pub activity: f64,
    /// Optional selected phase.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_phase: Option<u8>,
    /// Optional route order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route_order: Option<u32>,
    /// Whether this node is active.
    pub is_active: bool,
    /// Whether this node is pruned.
    pub is_pruned: bool,
    /// Optional metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: VisualMetadata,
}

/// Edge in a graph snapshot.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GraphEdge {
    /// Stable edge id.
    pub id: String,
    /// Source node id.
    pub source: String,
    /// Target node id.
    pub target: String,
    /// Semantic edge role.
    pub role: EdgeRole,
    /// Edge weight.
    pub weight: f64,
    /// Whether the edge is directed.
    pub directed: bool,
    /// Active flow value.
    pub active_flow: f64,
    /// Whether the edge is retained.
    pub is_retained: bool,
    /// Whether the edge is pruned.
    pub is_pruned: bool,
    /// Optional metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: VisualMetadata,
}

/// Pocket cluster summary.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PocketSummary {
    /// Stable pocket id.
    pub id: String,
    /// Pocket kind.
    pub kind: String,
    /// Node ids in the pocket.
    pub node_ids: Vec<String>,
    /// Bridge nodes connecting the pocket to a route or highway.
    pub bridge_nodes: Vec<String>,
    /// Mutation count.
    pub mutation_count: u32,
    /// Prune ratio.
    pub prune_ratio: f64,
}

/// Route trace for highlighting successor flow.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RouteTrace {
    /// Stable route id.
    pub id: String,
    /// Source node id.
    pub source: String,
    /// Target node id.
    pub target: String,
    /// Ordered route node ids.
    pub node_order: Vec<String>,
    /// Ordered route edge ids.
    pub edge_order: Vec<String>,
    /// Route status.
    pub status: String,
}

/// Timeline event.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MutationEvent {
    /// Stable event id.
    pub id: String,
    /// Schema identifier.
    pub schema_version: String,
    /// Stable run identifier.
    pub run_id: String,
    /// Checkpoint number.
    pub checkpoint: u32,
    /// Optional tick number.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tick: Option<u32>,
    /// Event kind.
    pub kind: EventKind,
    /// Affected node ids.
    pub node_ids: Vec<String>,
    /// Affected edge ids.
    pub edge_ids: Vec<String>,
    /// Event label.
    pub label: String,
}

/// Metrics row written as JSONL.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MetricRow {
    /// Schema identifier.
    pub schema_version: String,
    /// Stable run identifier.
    pub run_id: String,
    /// Checkpoint number.
    pub checkpoint: u32,
    /// Heldout score.
    pub heldout_score: f64,
    /// OOD score.
    pub ood_score: f64,
    /// Route-order accuracy.
    pub route_order_accuracy: f64,
    /// Missing successor count.
    pub missing_successor_count: u32,
    /// Output entropy.
    pub output_entropy: f64,
    /// Collapse flag.
    pub collapse_detected: bool,
}

/// Full graph snapshot.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GraphSnapshot {
    /// Schema identifier.
    pub schema_version: String,
    /// Stable run identifier.
    pub run_id: String,
    /// Checkpoint number.
    pub checkpoint: u32,
    /// Optional tick number.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tick: Option<u32>,
    /// Nodes.
    pub nodes: Vec<GraphNode>,
    /// Edges.
    pub edges: Vec<GraphEdge>,
    /// Pocket summaries.
    pub pockets: Vec<PocketSummary>,
    /// Route traces.
    pub routes: Vec<RouteTrace>,
    /// Optional metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: VisualMetadata,
}

/// Tick snapshot uses the same graph snapshot shape.
pub type TickSnapshot = GraphSnapshot;

/// Complete visual bundle ready to export.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VisualBundle {
    /// Schema version record.
    pub schema: SchemaVersion,
    /// Run manifest.
    pub manifest: RunManifest,
    /// Checkpoint index rows.
    pub checkpoint_index: Vec<CheckpointIndexRow>,
    /// Metrics rows.
    pub metrics: Vec<MetricRow>,
    /// Mutation/prune/repair events.
    pub events: Vec<MutationEvent>,
    /// Route traces.
    pub route_traces: Vec<RouteTrace>,
    /// Pocket summaries.
    pub pocket_summaries: Vec<PocketSummary>,
    /// Graph snapshots.
    pub graphs: Vec<GraphSnapshot>,
    /// Tick snapshots.
    pub ticks: Vec<TickSnapshot>,
}

/// Build the deterministic 052 smoke visual bundle.
pub fn sample_visual_bundle() -> VisualBundle {
    let run_id = "stable_loop_phase_lock_052_visual_sample".to_string();
    let schema = SchemaVersion {
        schema_version: VISUAL_SCHEMA_VERSION.to_string(),
        viewer_supported_versions: vec![VISUAL_SCHEMA_VERSION.to_string()],
    };
    let manifest = RunManifest {
        schema_version: VISUAL_SCHEMA_VERSION.to_string(),
        run_id: run_id.clone(),
        label: "052 smoke minimal visual sample".to_string(),
        checkpoints: vec![0, 10],
        has_ticks: true,
        claim_boundary: "visual infrastructure only; no model capability claim".to_string(),
    };
    let pockets = vec![
        PocketSummary {
            id: "p_left".to_string(),
            kind: "side_pocket".to_string(),
            node_ids: vec!["n_l1".to_string(), "n_l2".to_string()],
            bridge_nodes: vec!["n_h1".to_string()],
            mutation_count: 2,
            prune_ratio: 0.5,
        },
        PocketSummary {
            id: "p_right".to_string(),
            kind: "side_pocket".to_string(),
            node_ids: vec!["n_r1".to_string(), "n_r2".to_string()],
            bridge_nodes: vec!["n_h3".to_string()],
            mutation_count: 3,
            prune_ratio: 0.33,
        },
    ];
    let route = RouteTrace {
        id: "r_main".to_string(),
        source: "n_src".to_string(),
        target: "n_tgt".to_string(),
        node_order: vec![
            "n_src".to_string(),
            "n_h1".to_string(),
            "n_h2".to_string(),
            "n_h3".to_string(),
            "n_tgt".to_string(),
        ],
        edge_order: vec![
            "e_src_h1".to_string(),
            "e_h1_h2".to_string(),
            "e_h2_h3".to_string(),
            "e_h3_tgt".to_string(),
        ],
        status: "passing".to_string(),
    };
    let nodes = sample_nodes();
    let graph_000 = GraphSnapshot {
        schema_version: VISUAL_SCHEMA_VERSION.to_string(),
        run_id: run_id.clone(),
        checkpoint: 0,
        tick: None,
        nodes: nodes.clone(),
        edges: sample_edges(false),
        pockets: pockets.clone(),
        routes: vec![route.clone()],
        metadata: BTreeMap::from([("phase".to_string(), json!("pre_prune"))]),
    };
    let graph_010 = GraphSnapshot {
        schema_version: VISUAL_SCHEMA_VERSION.to_string(),
        run_id: run_id.clone(),
        checkpoint: 10,
        tick: None,
        nodes,
        edges: sample_edges(true),
        pockets: pockets.clone(),
        routes: vec![route.clone()],
        metadata: BTreeMap::from([("phase".to_string(), json!("post_prune"))]),
    };
    let tick = GraphSnapshot {
        schema_version: VISUAL_SCHEMA_VERSION.to_string(),
        run_id: run_id.clone(),
        checkpoint: 10,
        tick: Some(0),
        nodes: graph_010.nodes.clone(),
        edges: graph_010.edges.clone(),
        pockets: pockets.clone(),
        routes: vec![route.clone()],
        metadata: BTreeMap::from([("active_tick".to_string(), json!(0))]),
    };
    VisualBundle {
        schema,
        manifest,
        checkpoint_index: vec![
            CheckpointIndexRow {
                schema_version: VISUAL_SCHEMA_VERSION.to_string(),
                run_id: run_id.clone(),
                checkpoint: 0,
                graph_path: "graph/checkpoint_000.json".to_string(),
                tick_paths: vec![],
                summary: "pre-prune candidate graph".to_string(),
            },
            CheckpointIndexRow {
                schema_version: VISUAL_SCHEMA_VERSION.to_string(),
                run_id: run_id.clone(),
                checkpoint: 10,
                graph_path: "graph/checkpoint_010.json".to_string(),
                tick_paths: vec!["ticks/checkpoint_010_tick_000.json".to_string()],
                summary: "post-prune retained route".to_string(),
            },
        ],
        metrics: vec![
            MetricRow {
                schema_version: VISUAL_SCHEMA_VERSION.to_string(),
                run_id: run_id.clone(),
                checkpoint: 0,
                heldout_score: 0.62,
                ood_score: 0.58,
                route_order_accuracy: 0.71,
                missing_successor_count: 2,
                output_entropy: 3.1,
                collapse_detected: false,
            },
            MetricRow {
                schema_version: VISUAL_SCHEMA_VERSION.to_string(),
                run_id: run_id.clone(),
                checkpoint: 10,
                heldout_score: 1.0,
                ood_score: 1.0,
                route_order_accuracy: 1.0,
                missing_successor_count: 0,
                output_entropy: 4.2,
                collapse_detected: false,
            },
        ],
        events: vec![
            MutationEvent {
                id: "ev_mut_001".to_string(),
                schema_version: VISUAL_SCHEMA_VERSION.to_string(),
                run_id: run_id.clone(),
                checkpoint: 0,
                tick: None,
                kind: EventKind::Mutation,
                node_ids: vec!["n_l2".to_string()],
                edge_ids: vec!["e_l1_l2_candidate".to_string()],
                label: "candidate pocket edge added".to_string(),
            },
            MutationEvent {
                id: "ev_prune_001".to_string(),
                schema_version: VISUAL_SCHEMA_VERSION.to_string(),
                run_id,
                checkpoint: 10,
                tick: None,
                kind: EventKind::Prune,
                node_ids: vec!["n_r2".to_string()],
                edge_ids: vec!["e_r1_r2_pruned".to_string()],
                label: "non-route pocket branch pruned".to_string(),
            },
        ],
        route_traces: vec![route],
        pocket_summaries: pockets,
        graphs: vec![graph_000, graph_010],
        ticks: vec![tick],
    }
}

fn sample_nodes() -> Vec<GraphNode> {
    vec![
        node(
            "n_src",
            "Source",
            NodeRole::Source,
            None,
            -3.0,
            0.0,
            1.0,
            Some(0),
            Some(0),
            true,
        ),
        node(
            "n_h1",
            "H1",
            NodeRole::Highway,
            None,
            -1.8,
            0.0,
            0.9,
            Some(1),
            Some(1),
            true,
        ),
        node(
            "n_h2",
            "H2",
            NodeRole::Highway,
            None,
            -0.6,
            0.0,
            0.88,
            Some(2),
            Some(2),
            true,
        ),
        node(
            "n_h3",
            "H3",
            NodeRole::Highway,
            None,
            0.6,
            0.0,
            0.82,
            Some(3),
            Some(3),
            true,
        ),
        node(
            "n_tgt",
            "Target",
            NodeRole::Target,
            None,
            1.8,
            0.0,
            1.0,
            Some(0),
            Some(4),
            true,
        ),
        node(
            "n_l1",
            "L1",
            NodeRole::Pocket,
            Some("p_left"),
            -1.8,
            -1.0,
            0.35,
            None,
            None,
            false,
        ),
        node(
            "n_l2",
            "L2",
            NodeRole::Candidate,
            Some("p_left"),
            -0.9,
            -1.5,
            0.25,
            None,
            None,
            false,
        ),
        node(
            "n_r1",
            "R1",
            NodeRole::Pocket,
            Some("p_right"),
            0.6,
            1.0,
            0.4,
            None,
            None,
            false,
        ),
        node(
            "n_r2",
            "R2",
            NodeRole::Pocket,
            Some("p_right"),
            1.4,
            1.5,
            0.18,
            None,
            None,
            false,
        ),
    ]
}

#[allow(clippy::too_many_arguments)]
fn node(
    id: &str,
    label: &str,
    role: NodeRole,
    pocket_id: Option<&str>,
    x: f64,
    y: f64,
    activity: f64,
    selected_phase: Option<u8>,
    route_order: Option<u32>,
    is_active: bool,
) -> GraphNode {
    GraphNode {
        id: id.to_string(),
        label: label.to_string(),
        role,
        pocket_id: pocket_id.map(str::to_string),
        x,
        y,
        activity,
        selected_phase,
        route_order,
        is_active,
        is_pruned: false,
        metadata: BTreeMap::new(),
    }
}

fn sample_edges(post_prune: bool) -> Vec<GraphEdge> {
    let mut edges = vec![
        edge(
            "e_src_h1",
            "n_src",
            "n_h1",
            EdgeRole::Highway,
            1.0,
            0.9,
            true,
            false,
        ),
        edge(
            "e_h1_h2",
            "n_h1",
            "n_h2",
            EdgeRole::Highway,
            1.0,
            0.88,
            true,
            false,
        ),
        edge(
            "e_h2_h3",
            "n_h2",
            "n_h3",
            EdgeRole::Highway,
            1.0,
            0.86,
            true,
            false,
        ),
        edge(
            "e_h3_tgt",
            "n_h3",
            "n_tgt",
            EdgeRole::Highway,
            1.0,
            0.82,
            true,
            false,
        ),
        edge(
            "e_h1_l1",
            "n_h1",
            "n_l1",
            EdgeRole::Bridge,
            0.4,
            0.1,
            true,
            false,
        ),
        edge(
            "e_h3_r1",
            "n_h3",
            "n_r1",
            EdgeRole::Bridge,
            0.4,
            0.1,
            true,
            false,
        ),
        edge(
            "e_l1_l2_candidate",
            "n_l1",
            "n_l2",
            EdgeRole::Candidate,
            0.3,
            0.0,
            !post_prune,
            false,
        ),
    ];
    if post_prune {
        edges.push(edge(
            "e_h2_l2_added",
            "n_h2",
            "n_l2",
            EdgeRole::Candidate,
            0.5,
            0.2,
            true,
            false,
        ));
        edges.push(edge(
            "e_r1_r2_pruned",
            "n_r1",
            "n_r2",
            EdgeRole::Pruned,
            0.1,
            0.0,
            false,
            true,
        ));
    }
    edges
}

fn edge(
    id: &str,
    source: &str,
    target: &str,
    role: EdgeRole,
    weight: f64,
    active_flow: f64,
    is_retained: bool,
    is_pruned: bool,
) -> GraphEdge {
    GraphEdge {
        id: id.to_string(),
        source: source.to_string(),
        target: target.to_string(),
        role,
        weight,
        directed: true,
        active_flow,
        is_retained,
        is_pruned,
        metadata: BTreeMap::new(),
    }
}
