//! Deterministic larger playback visual bundle for 054.

use crate::visual_export::schema::{
    CheckpointIndexRow, EdgeRole, EventKind, GraphEdge, GraphNode, GraphSnapshot, MetricRow,
    MutationEvent, NodeRole, PocketSummary, RouteTrace, RunManifest, SchemaVersion, TickSnapshot,
    VisualBundle, VisualMetadata, VISUAL_SCHEMA_VERSION,
};
use serde_json::json;
use std::collections::BTreeMap;

const RUN_ID: &str = "stable_loop_phase_lock_054_larger_playback_smoke";
const CHECKPOINTS: [u32; 12] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110];
const HIGHWAY_COUNT: usize = 48;
const POCKET_COUNT: usize = 10;
const POCKET_NODES: usize = 8;

/// Build a deterministic larger-but-bounded visual playback bundle.
pub fn larger_playback_visual_bundle() -> VisualBundle {
    let schema = SchemaVersion {
        schema_version: VISUAL_SCHEMA_VERSION.to_string(),
        viewer_supported_versions: vec![VISUAL_SCHEMA_VERSION.to_string()],
    };
    let manifest = RunManifest {
        schema_version: VISUAL_SCHEMA_VERSION.to_string(),
        run_id: RUN_ID.to_string(),
        label: "054 larger deterministic playback smoke".to_string(),
        checkpoints: CHECKPOINTS.to_vec(),
        has_ticks: true,
        claim_boundary:
            "visual playback infrastructure only; deterministic graph projection, no training claim"
                .to_string(),
    };
    let pockets = pocket_summaries();
    let route = route_trace();
    let mut graphs = Vec::new();
    let mut ticks = Vec::new();
    let mut checkpoint_index = Vec::new();
    let mut metrics = Vec::new();
    let mut events = Vec::new();

    for (idx, checkpoint) in CHECKPOINTS.iter().copied().enumerate() {
        graphs.push(graph_snapshot(checkpoint, None, idx, &pockets, &route));
        let tick_paths = if matches!(checkpoint, 30 | 70 | 110) {
            ticks.push(tick_snapshot(checkpoint, 0, idx, &pockets, &route));
            ticks.push(tick_snapshot(checkpoint, 1, idx, &pockets, &route));
            vec![
                format!("ticks/checkpoint_{checkpoint:03}_tick_000.json"),
                format!("ticks/checkpoint_{checkpoint:03}_tick_001.json"),
            ]
        } else {
            vec![]
        };
        checkpoint_index.push(CheckpointIndexRow {
            schema_version: VISUAL_SCHEMA_VERSION.to_string(),
            run_id: RUN_ID.to_string(),
            checkpoint,
            graph_path: format!("graph/checkpoint_{checkpoint:03}.json"),
            tick_paths,
            summary: format!("larger playback checkpoint {checkpoint:03}"),
        });
        metrics.push(metric_row(checkpoint, idx));
        events.extend(events_for_checkpoint(checkpoint, idx));
    }

    VisualBundle {
        schema,
        manifest,
        checkpoint_index,
        metrics,
        events,
        route_traces: vec![route],
        pocket_summaries: pockets,
        graphs,
        ticks,
    }
}

fn graph_snapshot(
    checkpoint: u32,
    tick: Option<u32>,
    index: usize,
    pockets: &[PocketSummary],
    route: &RouteTrace,
) -> GraphSnapshot {
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "source_probe".to_string(),
        json!("054_larger_playback_smoke"),
    );
    metadata.insert("checkpoint_index".to_string(), json!(index));
    metadata.insert(
        "graph_node_count".to_string(),
        json!(HIGHWAY_COUNT + 2 + POCKET_COUNT * POCKET_NODES),
    );
    metadata.insert("graph_edge_count".to_string(), json!(edge_count_for(index)));
    metadata.insert(
        "event_count".to_string(),
        json!(events_for_checkpoint(checkpoint, index).len()),
    );
    if let Some(tick) = tick {
        metadata.insert("tick_phase".to_string(), json!(tick));
    }
    GraphSnapshot {
        schema_version: VISUAL_SCHEMA_VERSION.to_string(),
        run_id: RUN_ID.to_string(),
        checkpoint,
        tick,
        nodes: nodes_for(index, tick),
        edges: edges_for(index, tick),
        pockets: pockets.to_vec(),
        routes: vec![route.clone()],
        metadata,
    }
}

fn tick_snapshot(
    checkpoint: u32,
    tick: u32,
    index: usize,
    pockets: &[PocketSummary],
    route: &RouteTrace,
) -> TickSnapshot {
    graph_snapshot(
        checkpoint,
        Some(tick),
        index + tick as usize,
        pockets,
        route,
    )
}

fn nodes_for(index: usize, tick: Option<u32>) -> Vec<GraphNode> {
    let mut nodes = Vec::with_capacity(HIGHWAY_COUNT + 2 + POCKET_COUNT * POCKET_NODES);
    nodes.push(node(
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
    ));
    for i in 0..HIGHWAY_COUNT {
        let x = -3.6 + i as f64 * 0.16;
        let y = ((i as f64) * 0.33).sin() * 0.18;
        let order = Some(i as u32 + 1);
        let activity = if i <= index * 4 + 3 {
            0.78 + (i % 5) as f64 * 0.035
        } else {
            0.28
        };
        let is_active = i <= index * 4 + 3 || tick == Some((i % 2) as u32);
        nodes.push(node(
            &format!("n_h{i:02}"),
            &format!("H{i:02}"),
            NodeRole::Highway,
            None,
            x,
            y,
            activity.min(1.0),
            is_active,
            false,
            order,
        ));
    }
    nodes.push(node(
        "n_tgt",
        "target",
        NodeRole::Target,
        None,
        4.2,
        0.0,
        1.0,
        index >= 9,
        false,
        Some(HIGHWAY_COUNT as u32 + 1),
    ));

    for pocket in 0..POCKET_COUNT {
        let anchor = 3 + pocket * 4;
        let side = if pocket % 2 == 0 { -1.0 } else { 1.0 };
        for local in 0..POCKET_NODES {
            let id = format!("n_p{pocket:02}_{local:02}");
            let role = if local % 3 == 0 {
                NodeRole::Candidate
            } else {
                NodeRole::Pocket
            };
            let x = -3.6 + anchor as f64 * 0.16 + (local % 3) as f64 * 0.18;
            let y = side * (0.75 + (local / 3) as f64 * 0.34);
            let active_wave = index + tick.unwrap_or(0) as usize;
            let is_active = (pocket + local + active_wave) % 7 == 0;
            let is_pruned = index > 4 && (pocket + local) % 5 == 0;
            nodes.push(node(
                &id,
                &format!("P{pocket:02}.{local:02}"),
                role,
                Some(&format!("p_{pocket:02}")),
                x,
                y,
                if is_pruned {
                    0.08
                } else {
                    0.18 + (local as f64 * 0.045)
                },
                is_active,
                is_pruned,
                None,
            ));
        }
    }
    nodes
}

fn edges_for(index: usize, tick: Option<u32>) -> Vec<GraphEdge> {
    let mut edges = Vec::with_capacity(edge_count_for(index));
    edges.push(edge(
        "e_src_h00",
        "n_src",
        "n_h00",
        EdgeRole::Highway,
        1.0,
        1.0,
        true,
        false,
    ));
    for i in 0..(HIGHWAY_COUNT - 1) {
        let retained = i <= index * 4 + 3;
        edges.push(edge(
            &format!("e_h{i:02}_h{:02}", i + 1),
            &format!("n_h{i:02}"),
            &format!("n_h{:02}", i + 1),
            EdgeRole::Highway,
            0.85 + ((i + index) % 6) as f64 * 0.025,
            if retained { 0.92 } else { 0.28 },
            true,
            false,
        ));
    }
    edges.push(edge(
        "e_h47_tgt",
        "n_h47",
        "n_tgt",
        EdgeRole::Highway,
        1.0,
        if index >= 9 { 1.0 } else { 0.35 },
        index >= 9,
        false,
    ));

    for pocket in 0..POCKET_COUNT {
        let anchor = 3 + pocket * 4;
        edges.push(edge(
            &format!("e_bridge_h{anchor:02}_p{pocket:02}"),
            &format!("n_h{anchor:02}"),
            &format!("n_p{pocket:02}_00"),
            EdgeRole::Bridge,
            0.45,
            0.25 + (index as f64 * 0.04).min(0.55),
            true,
            false,
        ));
        for local in 0..(POCKET_NODES - 1) {
            let should_prune = index > 3 && (pocket + local + index) % 4 == 0;
            edges.push(edge(
                &format!("e_p{pocket:02}_{local:02}_{:02}", local + 1),
                &format!("n_p{pocket:02}_{local:02}"),
                &format!("n_p{pocket:02}_{:02}", local + 1),
                if should_prune {
                    EdgeRole::Pruned
                } else if local % 2 == 0 {
                    EdgeRole::Candidate
                } else {
                    EdgeRole::Pocket
                },
                0.25 + local as f64 * 0.03,
                if should_prune {
                    0.0
                } else {
                    0.18 + tick.unwrap_or(0) as f64 * 0.2
                },
                !should_prune,
                should_prune,
            ));
        }
        for local in 0..(POCKET_NODES - 2) {
            let should_prune = index > 6 && (pocket + local + index) % 5 == 0;
            edges.push(edge(
                &format!("e_p{pocket:02}_skip_{local:02}_{:02}", local + 2),
                &format!("n_p{pocket:02}_{local:02}"),
                &format!("n_p{pocket:02}_{:02}", local + 2),
                if should_prune {
                    EdgeRole::Pruned
                } else {
                    EdgeRole::Candidate
                },
                0.2 + local as f64 * 0.025,
                if should_prune {
                    0.0
                } else {
                    0.1 + tick.unwrap_or(0) as f64 * 0.16
                },
                !should_prune,
                should_prune,
            ));
        }
        if index >= 5 {
            edges.push(edge(
                &format!("e_repair_p{pocket:02}_h{:02}", anchor + 1),
                &format!("n_p{pocket:02}_03"),
                &format!("n_h{:02}", anchor + 1),
                EdgeRole::Bridge,
                0.55,
                0.52,
                true,
                false,
            ));
        }
        if index >= 8 {
            edges.push(edge(
                &format!("e_pruned_shortcut_p{pocket:02}"),
                &format!("n_p{pocket:02}_07"),
                "n_tgt",
                EdgeRole::Pruned,
                0.18,
                0.0,
                false,
                true,
            ));
        }
    }
    edges
}

fn edge_count_for(index: usize) -> usize {
    1 + (HIGHWAY_COUNT - 1)
        + 1
        + POCKET_COUNT
        + POCKET_COUNT * (POCKET_NODES - 1)
        + POCKET_COUNT * (POCKET_NODES - 2)
        + if index >= 5 { POCKET_COUNT } else { 0 }
        + if index >= 8 { POCKET_COUNT } else { 0 }
}

fn pocket_summaries() -> Vec<PocketSummary> {
    (0..POCKET_COUNT)
        .map(|pocket| PocketSummary {
            id: format!("p_{pocket:02}"),
            kind: if pocket % 2 == 0 {
                "route_side_pocket"
            } else {
                "control_side_pocket"
            }
            .to_string(),
            node_ids: (0..POCKET_NODES)
                .map(|local| format!("n_p{pocket:02}_{local:02}"))
                .collect(),
            bridge_nodes: vec![format!("n_h{:02}", 3 + pocket * 4)],
            mutation_count: (pocket as u32 + 2) * 3,
            prune_ratio: 0.18 + pocket as f64 * 0.035,
        })
        .collect()
}

fn route_trace() -> RouteTrace {
    let mut node_order = vec!["n_src".to_string()];
    node_order.extend((0..HIGHWAY_COUNT).map(|idx| format!("n_h{idx:02}")));
    node_order.push("n_tgt".to_string());
    let mut edge_order = vec!["e_src_h00".to_string()];
    edge_order.extend((0..(HIGHWAY_COUNT - 1)).map(|idx| format!("e_h{idx:02}_h{:02}", idx + 1)));
    edge_order.push("e_h47_tgt".to_string());
    RouteTrace {
        id: "r_054_larger_route".to_string(),
        source: "n_src".to_string(),
        target: "n_tgt".to_string(),
        node_order,
        edge_order,
        status: "deterministic_playback".to_string(),
    }
}

fn metric_row(checkpoint: u32, index: usize) -> MetricRow {
    let progress = index as f64 / (CHECKPOINTS.len() - 1) as f64;
    MetricRow {
        schema_version: VISUAL_SCHEMA_VERSION.to_string(),
        run_id: RUN_ID.to_string(),
        checkpoint,
        source_arm: Some("LARGER_PLAYBACK_DETERMINISTIC_VISUAL".to_string()),
        heldout_score: (0.45 + progress * 0.5).min(0.98),
        ood_score: (0.38 + progress * 0.54).min(0.96),
        family_min_accuracy: Some((progress * 0.94).min(0.94)),
        hard_distractor_accuracy: Some((0.2 + progress * 0.72).min(0.92)),
        long_ood_accuracy: Some((0.25 + progress * 0.68).min(0.93)),
        route_order_accuracy: (0.35 + progress * 0.62).min(0.97),
        missing_successor_count: (12usize.saturating_sub(index + 1)) as u32,
        output_entropy: 2.2 + progress * 3.0,
        unique_output_count: Some(12 + index * 5),
        expected_output_class_count: Some(75),
        collapse_detected: false,
    }
}

fn events_for_checkpoint(checkpoint: u32, index: usize) -> Vec<MutationEvent> {
    let mut events = Vec::new();
    if index % 2 == 0 {
        events.push(event(
            &format!("ev_054_mut_{checkpoint:03}"),
            checkpoint,
            Some(0),
            EventKind::Mutation,
            vec![format!("n_p{:02}_02", index % POCKET_COUNT)],
            vec![format!("e_p{:02}_02_03", index % POCKET_COUNT)],
            "candidate pocket edge introduced",
        ));
    }
    if index >= 3 {
        events.push(event(
            &format!("ev_054_prune_{checkpoint:03}"),
            checkpoint,
            Some(0),
            EventKind::Prune,
            vec![format!("n_p{:02}_05", (index + 2) % POCKET_COUNT)],
            vec![format!(
                "e_pruned_shortcut_p{:02}",
                (index + 2) % POCKET_COUNT
            )],
            "shortcut edge pruned",
        ));
    }
    if index >= 5 {
        events.push(event(
            &format!("ev_054_repair_{checkpoint:03}"),
            checkpoint,
            Some(1),
            EventKind::Repair,
            vec![format!("n_h{:02}", index * 3 % HIGHWAY_COUNT)],
            vec![format!(
                "e_h{:02}_h{:02}",
                index * 3 % (HIGHWAY_COUNT - 1),
                index * 3 % (HIGHWAY_COUNT - 1) + 1
            )],
            "successor continuity repaired",
        ));
    }
    if index >= 8 {
        events.push(event(
            &format!("ev_054_crystallize_{checkpoint:03}"),
            checkpoint,
            Some(1),
            EventKind::Crystallize,
            vec![format!("n_h{:02}", index * 4 % HIGHWAY_COUNT)],
            vec![format!(
                "e_h{:02}_h{:02}",
                index * 4 % (HIGHWAY_COUNT - 1),
                index * 4 % (HIGHWAY_COUNT - 1) + 1
            )],
            "route edge crystallized",
        ));
    }
    events
}

fn event(
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
        run_id: RUN_ID.to_string(),
        checkpoint,
        tick,
        kind,
        node_ids,
        edge_ids,
        label: label.to_string(),
    }
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
        metadata: VisualMetadata::new(),
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
        metadata: VisualMetadata::new(),
    }
}
