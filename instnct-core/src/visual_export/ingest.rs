//! Research-only ingest adapters for existing run artifact directories.

use crate::visual_export::exporter::VisualExportError;
use crate::visual_export::schema::{
    CheckpointIndexRow, EdgeRole, EventKind, GraphEdge, GraphNode, GraphSnapshot, MetricRow,
    MutationEvent, NodeRole, PocketSummary, RouteTrace, RunManifest, SchemaVersion, VisualBundle,
    VisualMetadata, VISUAL_SCHEMA_VERSION,
};
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const BASELINE_ARM: &str = "NO_ROUTE_GRAMMAR_ADVERSARIAL_FROZEN_BASELINE";
const REFERENCE_ARM: &str = "FROZEN_EVAL_048_REFERENCE";
const MAIN_ARM: &str = "ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_TRAIN_AND_INFER";
const ROLLBACK_ARM: &str = "ADVERSARIAL_FROZEN_ROUTE_GRAMMAR_ROLLBACK_GATED";

#[derive(Clone, Debug, Deserialize)]
struct AdversarialFrozenMetric {
    arm: String,
    heldout_exact_accuracy: f64,
    ood_exact_accuracy: f64,
    family_min_accuracy: f64,
    hard_distractor_accuracy: f64,
    long_ood_accuracy: f64,
    unique_output_count: usize,
    expected_output_class_count: usize,
    top_output_rate: f64,
    majority_output_rate: f64,
    output_entropy: f64,
    non_route_regression_delta: f64,
    route_api_overuse_rate: f64,
    positive_gate: bool,
    collapse_detected: bool,
    rollback_success: bool,
    checkpoint_save_load_pass: bool,
}

/// Build a visual bundle from a completed 049 adversarial frozen eval run.
///
/// This is a visual projection of real 049 metrics and control outcomes into
/// the renderer-agnostic `visual_snapshot_v1` graph shape. It does not rerun
/// training and does not claim that 049 emitted internal topology snapshots.
pub fn bundle_from_049_adversarial_run(source: &Path) -> Result<VisualBundle, VisualExportError> {
    let rows = read_metrics(&source.join("metrics.jsonl"))?;
    let baseline = find_arm(&rows, BASELINE_ARM)?;
    let reference = find_arm(&rows, REFERENCE_ARM)?;
    let main = find_arm(&rows, MAIN_ARM)?;
    let rollback = find_arm(&rows, ROLLBACK_ARM)?;

    let run_id = "stable_loop_phase_lock_053_real_run_ingest".to_string();
    let route = route_trace();
    let pockets = pocket_summaries();
    let schema = SchemaVersion {
        schema_version: VISUAL_SCHEMA_VERSION.to_string(),
        viewer_supported_versions: vec![VISUAL_SCHEMA_VERSION.to_string()],
    };
    let manifest = RunManifest {
        schema_version: VISUAL_SCHEMA_VERSION.to_string(),
        run_id: run_id.clone(),
        label: "053 visual ingest projection of 049 adversarial frozen eval".to_string(),
        checkpoints: vec![0, 50, 100],
        has_ticks: true,
        claim_boundary:
            "visual infrastructure only; ingests 049 metrics without new training claim".to_string(),
    };

    let graphs = vec![
        graph_for_stage(
            &run_id,
            0,
            None,
            baseline,
            StageKind::BaselineCollapse,
            &pockets,
            &route,
        ),
        graph_for_stage(
            &run_id,
            50,
            None,
            reference,
            StageKind::PartialReference,
            &pockets,
            &route,
        ),
        graph_for_stage(
            &run_id,
            100,
            None,
            main,
            StageKind::PassingIngest,
            &pockets,
            &route,
        ),
    ];
    let ticks = vec![
        graph_for_stage(
            &run_id,
            100,
            Some(0),
            main,
            StageKind::PassingIngest,
            &pockets,
            &route,
        ),
        graph_for_stage(
            &run_id,
            100,
            Some(1),
            rollback,
            StageKind::RollbackGate,
            &pockets,
            &route,
        ),
    ];

    Ok(VisualBundle {
        schema,
        manifest,
        checkpoint_index: vec![
            checkpoint_index(&run_id, 0, vec![], "049 no-grammar collapse baseline"),
            checkpoint_index(&run_id, 50, vec![], "048 reference partial behavior"),
            checkpoint_index(
                &run_id,
                100,
                vec![
                    "ticks/checkpoint_100_tick_000.json".to_string(),
                    "ticks/checkpoint_100_tick_001.json".to_string(),
                ],
                "049 route-grammar train-and-infer pass",
            ),
        ],
        metrics: vec![
            metric_row(&run_id, 0, baseline),
            metric_row(&run_id, 50, reference),
            metric_row(&run_id, 100, main),
        ],
        events: vec![
            event(
                &run_id,
                "ev_053_mutation_from_reference",
                50,
                Some(0),
                EventKind::Mutation,
                vec!["n_diag".to_string()],
                vec!["e_diag_candidate".to_string()],
                "diagnostic route candidate exposed from 049 reference metrics",
            ),
            event(
                &run_id,
                "ev_053_prune_control_shortcut",
                100,
                Some(0),
                EventKind::Prune,
                vec!["n_ctrl_majority".to_string()],
                vec!["e_majority_shortcut_pruned".to_string()],
                "majority/static shortcut control pruned in passing ingest view",
            ),
            event(
                &run_id,
                "ev_053_repair_successor_chain",
                100,
                Some(1),
                EventKind::Repair,
                vec!["n_h2".to_string(), "n_h3".to_string()],
                vec!["e_h2_h3".to_string(), "e_h3_tgt".to_string()],
                "successor chain completed by route-grammar positive arm",
            ),
        ],
        route_traces: vec![route],
        pocket_summaries: pockets,
        graphs,
        ticks,
    })
}

fn read_metrics(path: &Path) -> Result<Vec<AdversarialFrozenMetric>, VisualExportError> {
    let text = fs::read_to_string(path).map_err(|err| {
        VisualExportError::InvalidData(format!("failed to read {}: {err}", path.display()))
    })?;
    let mut rows = Vec::new();
    for (line_idx, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        rows.push(serde_json::from_str(line).map_err(|err| {
            VisualExportError::InvalidData(format!(
                "failed to parse metrics.jsonl line {}: {err}",
                line_idx + 1
            ))
        })?);
    }
    Ok(rows)
}

fn find_arm<'a>(
    rows: &'a [AdversarialFrozenMetric],
    arm: &str,
) -> Result<&'a AdversarialFrozenMetric, VisualExportError> {
    rows.iter()
        .find(|row| row.arm == arm)
        .ok_or_else(|| VisualExportError::InvalidData(format!("required 049 arm missing: {arm}")))
}

fn checkpoint_index(
    run_id: &str,
    checkpoint: u32,
    tick_paths: Vec<String>,
    summary: &str,
) -> CheckpointIndexRow {
    CheckpointIndexRow {
        schema_version: VISUAL_SCHEMA_VERSION.to_string(),
        run_id: run_id.to_string(),
        checkpoint,
        graph_path: format!("graph/checkpoint_{checkpoint:03}.json"),
        tick_paths,
        summary: summary.to_string(),
    }
}

fn metric_row(run_id: &str, checkpoint: u32, row: &AdversarialFrozenMetric) -> MetricRow {
    MetricRow {
        schema_version: VISUAL_SCHEMA_VERSION.to_string(),
        run_id: run_id.to_string(),
        checkpoint,
        source_arm: Some(row.arm.clone()),
        heldout_score: row.heldout_exact_accuracy,
        ood_score: row.ood_exact_accuracy,
        family_min_accuracy: Some(row.family_min_accuracy),
        hard_distractor_accuracy: Some(row.hard_distractor_accuracy),
        long_ood_accuracy: Some(row.long_ood_accuracy),
        route_order_accuracy: row.family_min_accuracy,
        missing_successor_count: inferred_missing_successors(row),
        output_entropy: row.output_entropy,
        unique_output_count: Some(row.unique_output_count),
        expected_output_class_count: Some(row.expected_output_class_count),
        collapse_detected: row.collapse_detected,
    }
}

#[derive(Clone, Copy)]
enum StageKind {
    BaselineCollapse,
    PartialReference,
    PassingIngest,
    RollbackGate,
}

fn graph_for_stage(
    run_id: &str,
    checkpoint: u32,
    tick: Option<u32>,
    row: &AdversarialFrozenMetric,
    stage: StageKind,
    pockets: &[PocketSummary],
    route: &RouteTrace,
) -> GraphSnapshot {
    GraphSnapshot {
        schema_version: VISUAL_SCHEMA_VERSION.to_string(),
        run_id: run_id.to_string(),
        checkpoint,
        tick,
        nodes: nodes_for_row(row, stage),
        edges: edges_for_stage(row, stage),
        pockets: pockets.to_vec(),
        routes: vec![route.clone()],
        metadata: stage_metadata(row, stage),
    }
}

fn nodes_for_row(row: &AdversarialFrozenMetric, stage: StageKind) -> Vec<GraphNode> {
    let route_activity = row.heldout_exact_accuracy.max(row.ood_exact_accuracy);
    let collapsed = row.collapse_detected;
    vec![
        node(
            "n_src",
            "source",
            NodeRole::Source,
            None,
            -4.0,
            0.0,
            1.0,
            true,
            false,
            Some(0),
        ),
        node(
            "n_h1",
            "H1",
            NodeRole::Highway,
            None,
            -2.5,
            0.0,
            route_activity,
            true,
            false,
            Some(1),
        ),
        node(
            "n_h2",
            "H2",
            NodeRole::Highway,
            None,
            -1.0,
            0.0,
            route_activity,
            !matches!(stage, StageKind::BaselineCollapse),
            false,
            Some(2),
        ),
        node(
            "n_h3",
            "H3",
            NodeRole::Highway,
            None,
            0.5,
            0.0,
            route_activity,
            matches!(stage, StageKind::PassingIngest | StageKind::RollbackGate),
            false,
            Some(3),
        ),
        node(
            "n_tgt",
            "target",
            NodeRole::Target,
            None,
            2.0,
            0.0,
            row.ood_exact_accuracy,
            matches!(stage, StageKind::PassingIngest | StageKind::RollbackGate),
            false,
            Some(4),
        ),
        node(
            "n_diag",
            "diagnostic labels",
            NodeRole::Candidate,
            Some("p_route_diagnostics"),
            -1.4,
            -1.25,
            row.family_min_accuracy.max(0.2),
            matches!(stage, StageKind::PassingIngest | StageKind::RollbackGate),
            false,
            None,
        ),
        node(
            "n_ctrl_majority",
            "majority shortcut",
            NodeRole::Pocket,
            Some("p_failure_controls"),
            0.2,
            1.45,
            row.majority_output_rate,
            collapsed,
            matches!(stage, StageKind::PassingIngest | StageKind::RollbackGate),
            None,
        ),
        node(
            "n_ctrl_static",
            "static output",
            NodeRole::Pocket,
            Some("p_failure_controls"),
            1.2,
            1.75,
            row.top_output_rate,
            collapsed,
            matches!(stage, StageKind::PassingIngest | StageKind::RollbackGate),
            None,
        ),
        node(
            "n_entropy",
            "output entropy",
            NodeRole::Pocket,
            Some("p_output_distribution"),
            1.0,
            -1.35,
            (row.output_entropy / 5.5).min(1.0),
            !collapsed,
            false,
            None,
        ),
    ]
}

fn edges_for_stage(row: &AdversarialFrozenMetric, stage: StageKind) -> Vec<GraphEdge> {
    let flow = row.heldout_exact_accuracy.max(row.ood_exact_accuracy);
    let mut edges = vec![edge(
        "e_src_h1",
        "n_src",
        "n_h1",
        EdgeRole::Highway,
        1.0,
        flow,
        true,
        false,
    )];

    match stage {
        StageKind::BaselineCollapse => {
            edges.push(edge(
                "e_h1_h2_candidate_gap",
                "n_h1",
                "n_h2",
                EdgeRole::Candidate,
                0.35,
                flow,
                false,
                false,
            ));
            edges.push(edge(
                "e_majority_shortcut",
                "n_h1",
                "n_ctrl_majority",
                EdgeRole::Candidate,
                row.majority_output_rate,
                row.top_output_rate,
                true,
                false,
            ));
        }
        StageKind::PartialReference => {
            edges.push(edge(
                "e_h1_h2",
                "n_h1",
                "n_h2",
                EdgeRole::Highway,
                0.7,
                flow,
                true,
                false,
            ));
            edges.push(edge(
                "e_h2_h3_candidate",
                "n_h2",
                "n_h3",
                EdgeRole::Candidate,
                0.45,
                flow,
                false,
                false,
            ));
            edges.push(edge(
                "e_majority_shortcut",
                "n_h1",
                "n_ctrl_majority",
                EdgeRole::Candidate,
                row.majority_output_rate,
                row.top_output_rate,
                true,
                false,
            ));
            edges.push(edge(
                "e_diag_candidate",
                "n_diag",
                "n_h2",
                EdgeRole::Bridge,
                0.5,
                row.family_min_accuracy,
                true,
                false,
            ));
        }
        StageKind::PassingIngest | StageKind::RollbackGate => {
            edges.push(edge(
                "e_h1_h2",
                "n_h1",
                "n_h2",
                EdgeRole::Highway,
                1.0,
                flow,
                true,
                false,
            ));
            edges.push(edge(
                "e_h2_h3",
                "n_h2",
                "n_h3",
                EdgeRole::Highway,
                1.0,
                flow,
                true,
                false,
            ));
            edges.push(edge(
                "e_h3_tgt",
                "n_h3",
                "n_tgt",
                EdgeRole::Highway,
                1.0,
                flow,
                true,
                false,
            ));
            edges.push(edge(
                "e_diag_candidate",
                "n_diag",
                "n_h2",
                EdgeRole::Bridge,
                0.85,
                row.family_min_accuracy,
                true,
                false,
            ));
            edges.push(edge(
                "e_majority_shortcut_pruned",
                "n_h1",
                "n_ctrl_majority",
                EdgeRole::Pruned,
                row.majority_output_rate,
                0.0,
                false,
                true,
            ));
            edges.push(edge(
                "e_static_shortcut_pruned",
                "n_ctrl_static",
                "n_tgt",
                EdgeRole::Pruned,
                row.top_output_rate,
                0.0,
                false,
                true,
            ));
            edges.push(edge(
                "e_entropy_observation",
                "n_entropy",
                "n_tgt",
                EdgeRole::Bridge,
                0.4,
                (row.output_entropy / 5.5).min(1.0),
                true,
                false,
            ));
        }
    }
    edges
}

fn stage_metadata(row: &AdversarialFrozenMetric, stage: StageKind) -> VisualMetadata {
    let stage_name = match stage {
        StageKind::BaselineCollapse => "baseline_collapse",
        StageKind::PartialReference => "partial_reference",
        StageKind::PassingIngest => "passing_ingest",
        StageKind::RollbackGate => "rollback_gate",
    };
    BTreeMap::from([
        (
            "source_probe".to_string(),
            json!("049_adversarial_frozen_eval_scale"),
        ),
        ("source_arm".to_string(), json!(row.arm)),
        ("stage".to_string(), json!(stage_name)),
        (
            "heldout_exact_accuracy".to_string(),
            json!(row.heldout_exact_accuracy),
        ),
        (
            "ood_exact_accuracy".to_string(),
            json!(row.ood_exact_accuracy),
        ),
        (
            "family_min_accuracy".to_string(),
            json!(row.family_min_accuracy),
        ),
        (
            "hard_distractor_accuracy".to_string(),
            json!(row.hard_distractor_accuracy),
        ),
        (
            "long_ood_accuracy".to_string(),
            json!(row.long_ood_accuracy),
        ),
        (
            "unique_output_count".to_string(),
            json!(row.unique_output_count),
        ),
        (
            "expected_output_class_count".to_string(),
            json!(row.expected_output_class_count),
        ),
        (
            "collapse_detected".to_string(),
            json!(row.collapse_detected),
        ),
        (
            "non_route_regression_delta".to_string(),
            json!(row.non_route_regression_delta),
        ),
        (
            "route_api_overuse_rate".to_string(),
            json!(row.route_api_overuse_rate),
        ),
        ("positive_gate".to_string(), json!(row.positive_gate)),
        ("rollback_success".to_string(), json!(row.rollback_success)),
        (
            "checkpoint_save_load_pass".to_string(),
            json!(row.checkpoint_save_load_pass),
        ),
    ])
}

fn node(
    id: &str,
    label: &str,
    role: NodeRole,
    pocket_id: Option<&str>,
    x: f64,
    y: f64,
    activity: f64,
    is_active: bool,
    is_pruned: bool,
    route_order: Option<u32>,
) -> GraphNode {
    GraphNode {
        id: id.to_string(),
        label: label.to_string(),
        role,
        pocket_id: pocket_id.map(str::to_string),
        x,
        y,
        activity,
        selected_phase: None,
        route_order,
        is_active,
        is_pruned,
        metadata: BTreeMap::new(),
    }
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

fn event(
    run_id: &str,
    id: &str,
    checkpoint: u32,
    tick: Option<u32>,
    kind: EventKind,
    node_ids: Vec<String>,
    edge_ids: Vec<String>,
    label: &str,
) -> MutationEvent {
    MutationEvent {
        id: id.to_string(),
        schema_version: VISUAL_SCHEMA_VERSION.to_string(),
        run_id: run_id.to_string(),
        checkpoint,
        tick,
        kind,
        node_ids,
        edge_ids,
        label: label.to_string(),
    }
}

fn route_trace() -> RouteTrace {
    RouteTrace {
        id: "r_049_positive_route".to_string(),
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
        status: "passing_049_ingest".to_string(),
    }
}

fn pocket_summaries() -> Vec<PocketSummary> {
    vec![
        PocketSummary {
            id: "p_route_diagnostics".to_string(),
            kind: "diagnostic_label_pocket".to_string(),
            node_ids: vec!["n_diag".to_string()],
            bridge_nodes: vec!["n_h2".to_string()],
            mutation_count: 1,
            prune_ratio: 0.0,
        },
        PocketSummary {
            id: "p_failure_controls".to_string(),
            kind: "known_failure_control_pocket".to_string(),
            node_ids: vec!["n_ctrl_majority".to_string(), "n_ctrl_static".to_string()],
            bridge_nodes: vec!["n_h1".to_string()],
            mutation_count: 2,
            prune_ratio: 1.0,
        },
        PocketSummary {
            id: "p_output_distribution".to_string(),
            kind: "output_distribution_pocket".to_string(),
            node_ids: vec!["n_entropy".to_string()],
            bridge_nodes: vec!["n_tgt".to_string()],
            mutation_count: 0,
            prune_ratio: 0.0,
        },
    ]
}

fn inferred_missing_successors(row: &AdversarialFrozenMetric) -> u32 {
    if row.family_min_accuracy >= 0.85 && !row.collapse_detected {
        0
    } else if row.family_min_accuracy <= 0.0 {
        6
    } else {
        2
    }
}
