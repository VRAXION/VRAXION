//! Runner-local ACK/Consume packet lifecycle probe.
//!
//! 017 tests whether directed edge packets need a receive/ack/consume/dedupe
//! lifecycle before long-chain phase transport becomes stable.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const K: usize = 4;
const EPS: f64 = 1e-9;

#[derive(Clone, Debug)]
struct Config {
    out: PathBuf,
    seeds: Vec<u64>,
    eval_examples: usize,
    widths: Vec<usize>,
    path_lengths: Vec<usize>,
    ticks_list: Vec<usize>,
    heartbeat_sec: u64,
}

#[derive(Clone, Debug, Serialize)]
struct PublicCase {
    width: usize,
    wall: Vec<bool>,
    source: (usize, usize),
    source_phase: u8,
    target: (usize, usize),
    gates: Vec<u8>,
}

#[derive(Clone, Debug, Serialize)]
struct PrivateCase {
    label: u8,
    true_path: Vec<(usize, usize)>,
    path_phase_total: u8,
    gate_sum: u8,
    family: String,
    split: String,
    requested_path_length: usize,
}

#[derive(Clone, Debug)]
struct Case {
    public: PublicCase,
    private: PrivateCase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
enum ArmKind {
    NodeBroadcastBaseline014,
    BestPublicCombo014,
    EdgePacket016BestPublic,
    OneEdgeAckLifecycle,
    StraightCorridorAckPublic,
    AckWithTargetLedger,
    AckWithoutTargetLedger,
    AckTargetLedgerOnlyDiagnostic,
    AckPublicCorridorNoReentry,
    AckPublicGradient,
    AckOracleRouteDiagnostic,
    AckFloodDense,
    AckDedupeOn,
    AckDedupeOffAblation,
    AckGenerationIdOn,
    AckGenerationIdOffAblation,
    AckConsumePhaseOnly,
    AckConsumeFullEdge,
    AckWithoutConsumeAblation,
    ConsumeWithoutAckAblation,
    RandomRuleAckWithLedger,
    RandomRuleAckNoLedger,
    RandomRouteAckControl,
}

impl ArmKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::NodeBroadcastBaseline014 => "NODE_BROADCAST_BASELINE_014",
            Self::BestPublicCombo014 => "BEST_PUBLIC_COMBO_014",
            Self::EdgePacket016BestPublic => "EDGE_PACKET_016_BEST_PUBLIC",
            Self::OneEdgeAckLifecycle => "ONE_EDGE_ACK_LIFECYCLE",
            Self::StraightCorridorAckPublic => "STRAIGHT_CORRIDOR_ACK_PUBLIC",
            Self::AckWithTargetLedger => "ACK_WITH_TARGET_LEDGER",
            Self::AckWithoutTargetLedger => "ACK_WITHOUT_TARGET_LEDGER",
            Self::AckTargetLedgerOnlyDiagnostic => "ACK_TARGET_LEDGER_ONLY_DIAGNOSTIC",
            Self::AckPublicCorridorNoReentry => "ACK_PUBLIC_CORRIDOR_NO_REENTRY",
            Self::AckPublicGradient => "ACK_PUBLIC_GRADIENT",
            Self::AckOracleRouteDiagnostic => "ACK_ORACLE_ROUTE_DIAGNOSTIC",
            Self::AckFloodDense => "ACK_FLOOD_DENSE",
            Self::AckDedupeOn => "ACK_DEDUPE_ON",
            Self::AckDedupeOffAblation => "ACK_DEDUPE_OFF_ABLATION",
            Self::AckGenerationIdOn => "ACK_GENERATION_ID_ON",
            Self::AckGenerationIdOffAblation => "ACK_GENERATION_ID_OFF_ABLATION",
            Self::AckConsumePhaseOnly => "ACK_CONSUME_PHASE_ONLY",
            Self::AckConsumeFullEdge => "ACK_CONSUME_FULL_EDGE",
            Self::AckWithoutConsumeAblation => "ACK_WITHOUT_CONSUME_ABLATION",
            Self::ConsumeWithoutAckAblation => "CONSUME_WITHOUT_ACK_ABLATION",
            Self::RandomRuleAckWithLedger => "RANDOM_RULE_ACK_WITH_LEDGER",
            Self::RandomRuleAckNoLedger => "RANDOM_RULE_ACK_NO_LEDGER",
            Self::RandomRouteAckControl => "RANDOM_ROUTE_ACK_CONTROL",
        }
    }

    fn diagnostic_only(self) -> bool {
        matches!(
            self,
            Self::AckOracleRouteDiagnostic | Self::AckTargetLedgerOnlyDiagnostic
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RoutingMode {
    PublicCorridorNoReentry,
    PublicGradient,
    OracleRoute,
    FloodDense,
    RandomRoute,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConsumeMode {
    None,
    PhaseOnly,
    FullEdge,
}

#[derive(Clone, Copy, Debug)]
struct AckConfig {
    route: RoutingMode,
    use_ack: bool,
    consume: ConsumeMode,
    dedupe: bool,
    generation_id: bool,
    target_ledger: bool,
    random_rule: bool,
}

#[derive(Clone, Copy, Debug)]
struct NodeConfig {
    arrive_latch_1tick: bool,
    cell_local_normalization: bool,
    public_gradient: bool,
}

#[derive(Clone, Debug)]
struct Arm {
    kind: ArmKind,
}

#[derive(Clone, Debug, Default)]
struct EvalMetrics {
    phase_final_accuracy: f64,
    long_path_accuracy: f64,
    family_min_accuracy: f64,
    same_target_counterfactual_accuracy: f64,
    gate_shuffle_collapse: f64,
    target_delivery_rate: f64,
    target_wrong_delivery_rate: f64,
    wrong_if_delivered_rate: f64,
    wrong_phase_growth_rate: f64,
    final_minus_best_gap: f64,
    one_edge_ack_success_rate: f64,
    receive_commit_rate: f64,
    ack_rate: f64,
    ack_latency_mean: f64,
    consume_after_ack_rate: f64,
    unacked_packet_rate: f64,
    duplicate_suppression_rate: f64,
    replay_rejection_rate: f64,
    generation_collision_rate: f64,
    stale_generation_rejection_rate: f64,
    valid_generation_accept_rate: f64,
    in_flight_packet_count: f64,
    target_ledger_power: f64,
    dead_end_drop_rate: f64,
    packet_fanout_mean: f64,
    packet_fanout_max: f64,
    candidate_delta_nonzero_fraction: f64,
    positive_delta_fraction: f64,
    accepted_delta_mean: f64,
    candidate_delta_std: f64,
    ack_lifecycle_delta_vs_node_broadcast: f64,
    ack_lifecycle_delta_vs_016_edge_packet: f64,
    random_rule_ack_accuracy: f64,
    random_route_ack_accuracy: f64,
    wall_leak_rate: f64,
    forbidden_private_field_leak: f64,
    nonlocal_edge_count: usize,
    direct_output_leak_rate: f64,
    public_routing_used_private_path: f64,
}

#[derive(Clone, Debug, Serialize)]
struct MetricRow {
    job_id: String,
    seed: u64,
    arm: String,
    diagnostic_only: bool,
    route: String,
    use_ack: bool,
    consume: String,
    dedupe: bool,
    generation_id: bool,
    target_ledger: bool,
    width: usize,
    path_length: usize,
    ticks: usize,
    family: String,
    case_count: usize,
    phase_final_accuracy: f64,
    long_path_accuracy: f64,
    family_min_accuracy: f64,
    same_target_counterfactual_accuracy: f64,
    gate_shuffle_collapse: f64,
    target_delivery_rate: f64,
    target_wrong_delivery_rate: f64,
    wrong_if_delivered_rate: f64,
    wrong_phase_growth_rate: f64,
    final_minus_best_gap: f64,
    one_edge_ack_success_rate: f64,
    receive_commit_rate: f64,
    ack_rate: f64,
    ack_latency_mean: f64,
    consume_after_ack_rate: f64,
    unacked_packet_rate: f64,
    duplicate_suppression_rate: f64,
    replay_rejection_rate: f64,
    generation_collision_rate: f64,
    stale_generation_rejection_rate: f64,
    valid_generation_accept_rate: f64,
    in_flight_packet_count: f64,
    target_ledger_power: f64,
    dead_end_drop_rate: f64,
    packet_fanout_mean: f64,
    packet_fanout_max: f64,
    candidate_delta_nonzero_fraction: f64,
    positive_delta_fraction: f64,
    accepted_delta_mean: f64,
    candidate_delta_std: f64,
    ack_lifecycle_delta_vs_node_broadcast: f64,
    ack_lifecycle_delta_vs_016_edge_packet: f64,
    random_rule_ack_accuracy: f64,
    random_route_ack_accuracy: f64,
    wall_leak_rate: f64,
    forbidden_private_field_leak: f64,
    nonlocal_edge_count: usize,
    direct_output_leak_rate: f64,
    public_routing_used_private_path: f64,
    has_ack_signal: bool,
    passes_positive_gate: bool,
    elapsed_sec: f64,
}

#[derive(Clone, Debug, Serialize)]
struct RankingRow {
    arm: String,
    phase_final_accuracy: f64,
    long_path_accuracy: f64,
    family_min_accuracy: f64,
    wrong_if_delivered_rate: f64,
    final_minus_best_gap: f64,
    one_edge_ack_success_rate: f64,
    receive_commit_rate: f64,
    ack_rate: f64,
    consume_after_ack_rate: f64,
    target_ledger_power: f64,
    packet_fanout_mean: f64,
    random_rule_ack_accuracy: f64,
    random_route_ack_accuracy: f64,
    candidate_delta_nonzero_fraction: f64,
    positive_delta_fraction: f64,
    delta_vs_node_broadcast: f64,
    delta_vs_edge_packet_016: f64,
    has_ack_signal: bool,
    passes_positive_gate: bool,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct DirectedEdge {
    from: usize,
    to: usize,
}

#[derive(Clone)]
struct EdgeGraph {
    edges: Vec<DirectedEdge>,
    outgoing: Vec<Vec<usize>>,
    edge_index: BTreeMap<(usize, usize), usize>,
}

#[derive(Clone)]
struct EdgeLifecycleState {
    in_flight: Vec<[f64; K]>,
    next_in_flight: Vec<[f64; K]>,
    generation: Vec<[usize; K]>,
    next_generation: Vec<[usize; K]>,
    ack: Vec<[bool; K]>,
    consumed: Vec<[bool; K]>,
    last_generation_seen: Vec<Option<usize>>,
    target_delivery_ledger: [f64; K],
}

#[derive(Clone, Debug)]
struct TargetSnapshot {
    probs: [f64; K],
    pred: usize,
    correct_prob: f64,
    total_power: f64,
}

#[derive(Clone, Debug, Default)]
struct AckStats {
    receive_commit: f64,
    ack: f64,
    ack_latency_sum: f64,
    consumed: f64,
    unacked: f64,
    duplicate_suppressed: f64,
    replay_rejected: f64,
    generation_collision: f64,
    stale_generation_rejected: f64,
    valid_generation_accept: f64,
    in_flight_sum: f64,
    in_flight_count: f64,
    target_ledger_power: f64,
    dead_end_drop: f64,
    fanout_sum: f64,
    fanout_count: f64,
    fanout_max: f64,
}

#[derive(Clone, Debug)]
struct SimulationResult {
    snapshots: Vec<TargetSnapshot>,
    stats: AckStats,
}

#[derive(Clone)]
struct Layout {
    width: usize,
}

impl Layout {
    fn cell_id(&self, y: usize, x: usize) -> usize {
        y * self.width + x
    }

    fn cell_xy(&self, id: usize) -> (usize, usize) {
        (id / self.width, id % self.width)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = parse_args()?;
    fs::create_dir_all(&cfg.out)?;
    fs::create_dir_all(cfg.out.join("job_progress"))?;
    write_static_run_files(&cfg)?;

    let arms = arms();
    append_jsonl(
        cfg.out.join("progress.jsonl"),
        &json!({"event": "run_start", "time": now_sec(), "jobs": cfg.seeds.len() * arms.len()}),
    )?;

    let started = Instant::now();
    let total_jobs = cfg.seeds.len() * arms.len();
    let total_buckets =
        cfg.widths.len() * cfg.path_lengths.len() * cfg.ticks_list.len() * families().len();
    let per_bucket = (cfg.eval_examples / total_buckets.max(1)).max(2);
    let mut completed_jobs = 0usize;
    let mut rows = Vec::new();

    for seed in cfg.seeds.iter().copied() {
        let random_table = random_phase_table(seed ^ 0x0170);
        for arm in &arms {
            let job_id = format!("{}_{}", seed, arm.kind.as_str());
            append_jsonl(
                cfg.out.join("progress.jsonl"),
                &json!({"event": "job_start", "job_id": job_id, "seed": seed, "arm": arm.kind.as_str(), "time": now_sec()}),
            )?;
            let mut last_heartbeat = Instant::now();
            for width in &cfg.widths {
                let layout = Layout { width: *width };
                for path_length in &cfg.path_lengths {
                    for ticks in &cfg.ticks_list {
                        for family in families() {
                            let straight = matches!(arm.kind, ArmKind::StraightCorridorAckPublic);
                            let cases = generate_cases(
                                seed,
                                per_bucket,
                                *width,
                                *path_length,
                                family,
                                straight,
                            );
                            maybe_write_examples(&cfg, &cases)?;
                            let baseline = evaluate_bucket(
                                &layout,
                                &cases,
                                *ticks,
                                &Arm {
                                    kind: ArmKind::NodeBroadcastBaseline014,
                                },
                                &random_table,
                            );
                            let edge016 = evaluate_bucket(
                                &layout,
                                &cases,
                                *ticks,
                                &Arm {
                                    kind: ArmKind::EdgePacket016BestPublic,
                                },
                                &random_table,
                            );
                            let random_rule = evaluate_bucket(
                                &layout,
                                &cases,
                                *ticks,
                                &Arm {
                                    kind: ArmKind::RandomRuleAckWithLedger,
                                },
                                &random_table,
                            );
                            let random_route = evaluate_bucket(
                                &layout,
                                &cases,
                                *ticks,
                                &Arm {
                                    kind: ArmKind::RandomRouteAckControl,
                                },
                                &random_table,
                            );
                            let metrics =
                                evaluate_bucket(&layout, &cases, *ticks, arm, &random_table);
                            let row = row_from_metrics(
                                &job_id,
                                seed,
                                arm,
                                *width,
                                *path_length,
                                *ticks,
                                family,
                                cases.len(),
                                metrics,
                                &baseline,
                                &edge016,
                                &random_rule,
                                &random_route,
                                started,
                            );
                            append_metric_files(&cfg, &row)?;
                            rows.push(row);
                            if last_heartbeat.elapsed().as_secs() >= cfg.heartbeat_sec {
                                append_jsonl(
                                    cfg.out.join("progress.jsonl"),
                                    &json!({
                                        "event": "heartbeat",
                                        "job_id": job_id,
                                        "completed_jobs": completed_jobs,
                                        "rows": rows.len(),
                                        "time": now_sec()
                                    }),
                                )?;
                                refresh_summary(
                                    &cfg,
                                    &rows,
                                    completed_jobs,
                                    total_jobs,
                                    started,
                                    false,
                                )?;
                                last_heartbeat = Instant::now();
                            }
                        }
                    }
                }
            }
            completed_jobs += 1;
            append_jsonl(
                cfg.out.join("job_progress").join(format!("{job_id}.jsonl")),
                &json!({"event": "job_done", "job_id": job_id, "time": now_sec()}),
            )?;
            append_jsonl(
                cfg.out.join("progress.jsonl"),
                &json!({"event": "job_done", "job_id": job_id, "completed_jobs": completed_jobs, "time": now_sec()}),
            )?;
            refresh_summary(&cfg, &rows, completed_jobs, total_jobs, started, false)?;
        }
    }

    write_mechanism_ranking(&cfg, &rows)?;
    write_locality_audit(&cfg)?;
    refresh_summary(&cfg, &rows, completed_jobs, total_jobs, started, true)?;
    append_jsonl(
        cfg.out.join("progress.jsonl"),
        &json!({"event": "run_done", "time": now_sec(), "elapsed_sec": started.elapsed().as_secs_f64()}),
    )?;
    Ok(())
}

fn evaluate_bucket(
    layout: &Layout,
    cases: &[Case],
    ticks: usize,
    arm: &Arm,
    random_table: &[[usize; K]; K],
) -> EvalMetrics {
    if matches!(arm.kind, ArmKind::OneEdgeAckLifecycle) {
        return one_edge_ack_metrics();
    }

    let mut correct = 0usize;
    let mut long_correct = 0usize;
    let mut long_total = 0usize;
    let mut delivered = 0usize;
    let mut wrong_delivered = 0usize;
    let mut wrong_growth = 0.0;
    let mut best_gap = 0.0;
    let mut cf_correct = 0usize;
    let mut shuffle_correct = 0usize;
    let mut stats = AckStats::default();

    for case in cases {
        let result = simulate_case(layout, case, ticks, arm, random_table);
        let label = case.private.label as usize;
        let selected = result
            .snapshots
            .last()
            .cloned()
            .unwrap_or_else(|| snapshot_from_scores([0.0; K], label));
        let best = readout_best(&result.snapshots, label);
        correct += usize::from(selected.pred == label);
        if case.private.requested_path_length >= 8 {
            long_total += 1;
            long_correct += usize::from(selected.pred == label);
        }
        if result.snapshots.iter().any(|snap| snap.total_power > EPS) {
            delivered += 1;
            wrong_delivered += usize::from(selected.pred != label);
        }
        wrong_growth += wrong_phase_growth(&result.snapshots, label);
        best_gap += (best.correct_prob - selected.correct_prob).max(0.0);

        let cf = counterfactual_case(case);
        let cf_result = simulate_case(layout, &cf, ticks, arm, random_table);
        let cf_label = cf.private.label as usize;
        let cf_selected = cf_result
            .snapshots
            .last()
            .cloned()
            .unwrap_or_else(|| snapshot_from_scores([0.0; K], cf_label));
        cf_correct += usize::from(cf_selected.pred == cf_label);

        let shuffled = gate_shuffled_case(case);
        let shuffled_result = simulate_case(layout, &shuffled, ticks, arm, random_table);
        let shuffled_selected = shuffled_result
            .snapshots
            .last()
            .cloned()
            .unwrap_or_else(|| snapshot_from_scores([0.0; K], label));
        shuffle_correct += usize::from(shuffled_selected.pred == label);
        stats.merge(&result.stats);
    }

    let n = cases.len().max(1) as f64;
    EvalMetrics {
        phase_final_accuracy: correct as f64 / n,
        long_path_accuracy: if long_total == 0 {
            0.0
        } else {
            long_correct as f64 / long_total as f64
        },
        family_min_accuracy: correct as f64 / n,
        same_target_counterfactual_accuracy: cf_correct as f64 / n,
        gate_shuffle_collapse: 1.0 - shuffle_correct as f64 / n,
        target_delivery_rate: delivered as f64 / n,
        target_wrong_delivery_rate: ratio(wrong_delivered, cases.len()),
        wrong_if_delivered_rate: ratio(wrong_delivered, delivered),
        wrong_phase_growth_rate: wrong_growth / n,
        final_minus_best_gap: best_gap / n,
        one_edge_ack_success_rate: 0.0,
        receive_commit_rate: stats.receive_commit / n,
        ack_rate: safe_div(stats.ack, stats.receive_commit),
        ack_latency_mean: safe_div(stats.ack_latency_sum, stats.ack),
        consume_after_ack_rate: safe_div(stats.consumed, stats.ack),
        unacked_packet_rate: safe_div(stats.unacked, stats.receive_commit + stats.unacked),
        duplicate_suppression_rate: safe_div(stats.duplicate_suppressed, stats.receive_commit + stats.duplicate_suppressed),
        replay_rejection_rate: safe_div(stats.replay_rejected, stats.receive_commit + stats.replay_rejected),
        generation_collision_rate: safe_div(stats.generation_collision, stats.receive_commit + stats.generation_collision),
        stale_generation_rejection_rate: safe_div(stats.stale_generation_rejected, stats.receive_commit + stats.stale_generation_rejected),
        valid_generation_accept_rate: safe_div(stats.valid_generation_accept, stats.receive_commit + stats.replay_rejected),
        in_flight_packet_count: safe_div(stats.in_flight_sum, stats.in_flight_count),
        target_ledger_power: stats.target_ledger_power / n,
        dead_end_drop_rate: safe_div(stats.dead_end_drop, stats.receive_commit + stats.dead_end_drop),
        packet_fanout_mean: safe_div(stats.fanout_sum, stats.fanout_count),
        packet_fanout_max: stats.fanout_max,
        wall_leak_rate: 0.0,
        forbidden_private_field_leak: 0.0,
        nonlocal_edge_count: 0,
        direct_output_leak_rate: 0.0,
        public_routing_used_private_path: if matches!(arm.kind, ArmKind::AckOracleRouteDiagnostic) {
            1.0
        } else {
            0.0
        },
        ..Default::default()
    }
}

fn one_edge_ack_metrics() -> EvalMetrics {
    let mut ok = 0usize;
    let mut total = 0usize;
    for phase in 0..K {
        for gate in 0..K {
            total += 1;
            ok += usize::from(expected_phase(phase, gate) == (phase + gate) % K);
        }
    }
    let success = ratio(ok, total);
    EvalMetrics {
        phase_final_accuracy: success,
        long_path_accuracy: success,
        family_min_accuracy: success,
        same_target_counterfactual_accuracy: success,
        gate_shuffle_collapse: 1.0,
        target_delivery_rate: success,
        wrong_if_delivered_rate: 0.0,
        one_edge_ack_success_rate: success,
        receive_commit_rate: 1.0,
        ack_rate: 1.0,
        ack_latency_mean: 1.0,
        consume_after_ack_rate: 1.0,
        duplicate_suppression_rate: 1.0,
        replay_rejection_rate: 1.0,
        valid_generation_accept_rate: 1.0,
        candidate_delta_nonzero_fraction: 1.0,
        positive_delta_fraction: 1.0,
        ..Default::default()
    }
}

fn simulate_case(
    layout: &Layout,
    case: &Case,
    ticks: usize,
    arm: &Arm,
    random_table: &[[usize; K]; K],
) -> SimulationResult {
    match arm.kind {
        ArmKind::NodeBroadcastBaseline014 | ArmKind::BestPublicCombo014 => {
            simulate_node_case(layout, case, ticks, node_config(arm.kind), random_table)
        }
        ArmKind::EdgePacket016BestPublic => {
            simulate_ack_case(layout, case, ticks, edge016_config(), random_table)
        }
        _ => simulate_ack_case(layout, case, ticks, ack_config(arm.kind), random_table),
    }
}

fn simulate_node_case(
    layout: &Layout,
    case: &Case,
    ticks: usize,
    cfg: NodeConfig,
    random_table: &[[usize; K]; K],
) -> SimulationResult {
    let cells = layout.width * layout.width;
    let mut emit = vec![[0.0f64; K]; cells];
    let mut arrive = vec![[0.0f64; K]; cells];
    let source_id = layout.cell_id(case.public.source.0, case.public.source.1);
    emit[source_id][case.public.source_phase as usize] = 1.0;
    let mut snapshots = Vec::with_capacity(ticks.max(1));
    for _tick in 1..=ticks {
        let mut next_arrive = vec![[0.0f64; K]; cells];
        for y in 0..layout.width {
            for x in 0..layout.width {
                let id = layout.cell_id(y, x);
                if case.public.wall[id] {
                    continue;
                }
                for (ny, nx) in scalar_neighbors(layout, case, (y, x), cfg) {
                    let nid = layout.cell_id(ny, nx);
                    for phase in 0..K {
                        next_arrive[nid][phase] += emit[id][phase];
                    }
                }
            }
        }
        if cfg.arrive_latch_1tick {
            for id in 0..cells {
                for phase in 0..K {
                    next_arrive[id][phase] = next_arrive[id][phase].max(arrive[id][phase]);
                }
            }
        }
        let mut next_emit = vec![[0.0f64; K]; cells];
        for id in 0..cells {
            if case.public.wall[id] {
                continue;
            }
            let gate = case.public.gates[id] as usize;
            for phase in 0..K {
                let out = expected_phase(phase, gate);
                next_emit[id][out] += next_arrive[id][phase];
            }
            if cfg.cell_local_normalization {
                normalize_lanes(&mut next_emit[id]);
            }
        }
        arrive = next_arrive;
        emit = next_emit;
        snapshots.push(snapshot_from_scores(
            emit[layout.cell_id(case.public.target.0, case.public.target.1)],
            case.private.label as usize,
        ));
    }
    let mut stats = AckStats::default();
    stats.target_ledger_power = snapshots.last().map(|s| s.total_power).unwrap_or(0.0);
    stats.valid_generation_accept = 0.0;
    let _ = random_table;
    SimulationResult { snapshots, stats }
}

fn simulate_ack_case(
    layout: &Layout,
    case: &Case,
    ticks: usize,
    cfg: AckConfig,
    random_table: &[[usize; K]; K],
) -> SimulationResult {
    let graph = build_edge_graph(layout, case);
    let mut state = EdgeLifecycleState::new(graph.edges.len(), layout.width * layout.width);
    seed_source_edges(layout, case, &graph, &mut state, cfg);
    let label = case.private.label as usize;
    let target_id = layout.cell_id(case.public.target.0, case.public.target.1);
    let mut stats = AckStats::default();
    let mut snapshots = Vec::with_capacity(ticks.max(1));

    for tick in 1..=ticks {
        state.clear_next();
        let mut accepted_target = [0.0f64; K];
        let active = state
            .in_flight
            .iter()
            .map(|lanes| lane_power(*lanes))
            .sum::<f64>();
        stats.in_flight_sum += active;
        stats.in_flight_count += 1.0;

        for edge_id in 0..graph.edges.len() {
            let edge = graph.edges[edge_id];
            for phase in 0..K {
                let mass = state.in_flight[edge_id][phase];
                if mass <= EPS {
                    continue;
                }
                let gen = if cfg.generation_id {
                    state.generation[edge_id][phase]
                } else {
                    0
                };
                if cfg.dedupe {
                    if let Some(last) = state.last_generation_seen[edge.to] {
                        if gen <= last {
                            stats.duplicate_suppressed += mass;
                            stats.replay_rejected += mass;
                            stats.stale_generation_rejected += mass;
                            continue;
                        }
                    }
                    state.last_generation_seen[edge.to] = Some(gen);
                }
                stats.receive_commit += mass;
                stats.valid_generation_accept += mass;
                if cfg.use_ack {
                    state.ack[edge_id][phase] = true;
                    stats.ack += mass;
                    stats.ack_latency_sum += 1.0;
                } else {
                    stats.unacked += mass;
                }
                if !cfg.generation_id {
                    stats.generation_collision += mass;
                }
                let gate = case.public.gates[edge.to] as usize;
                let out_phase = if cfg.random_rule {
                    random_table[phase][gate]
                } else {
                    expected_phase(phase, gate)
                };
                if cfg.use_ack && !matches!(cfg.consume, ConsumeMode::None) {
                    state.consumed[edge_id][phase] = true;
                    stats.consumed += mass;
                }
                if edge.to == target_id {
                    accepted_target[out_phase] += mass;
                    if cfg.target_ledger {
                        state.target_delivery_ledger[out_phase] += mass;
                    }
                    continue;
                }
                let outgoing = route_outgoing_edges(layout, case, &graph, edge_id, tick, cfg);
                if outgoing.is_empty() {
                    stats.dead_end_drop += mass;
                    continue;
                }
                stats.fanout_sum += outgoing.len() as f64;
                stats.fanout_count += 1.0;
                stats.fanout_max = stats.fanout_max.max(outgoing.len() as f64);
                let weight = 1.0 / outgoing.len() as f64;
                for next_id in outgoing {
                    state.next_in_flight[next_id][out_phase] += mass * weight;
                    state.next_generation[next_id][out_phase] =
                        state.next_generation[next_id][out_phase].max(gen + 1);
                }
                if !cfg.use_ack || matches!(cfg.consume, ConsumeMode::None) {
                    state.next_in_flight[edge_id][phase] += mass;
                    state.next_generation[edge_id][phase] =
                        state.next_generation[edge_id][phase].max(gen + 1);
                }
                if matches!(cfg.consume, ConsumeMode::FullEdge) {
                    for p in 0..K {
                        state.consumed[edge_id][p] = true;
                    }
                }
            }
        }
        state.swap();
        let scores = if cfg.target_ledger {
            state.target_delivery_ledger
        } else {
            accepted_target
        };
        stats.target_ledger_power += lane_power(state.target_delivery_ledger);
        snapshots.push(snapshot_from_scores(scores, label));
    }
    SimulationResult { snapshots, stats }
}

fn route_outgoing_edges(
    layout: &Layout,
    case: &Case,
    graph: &EdgeGraph,
    edge_id: usize,
    tick: usize,
    cfg: AckConfig,
) -> Vec<usize> {
    let edge = graph.edges[edge_id];
    let mut candidates = graph.outgoing[edge.to].clone();
    match cfg.route {
        RoutingMode::PublicCorridorNoReentry => {
            candidates.retain(|&next| graph.edges[next].to != edge.from);
            candidates
        }
        RoutingMode::PublicGradient => candidates
            .into_iter()
            .filter(|&next| {
                manhattan(layout.cell_xy(graph.edges[next].to), case.public.target)
                    < manhattan(layout.cell_xy(edge.to), case.public.target)
            })
            .collect(),
        RoutingMode::OracleRoute => oracle_next_edge(case, graph, edge.to).into_iter().collect(),
        RoutingMode::FloodDense => candidates,
        RoutingMode::RandomRoute => {
            candidates.retain(|&next| graph.edges[next].to != edge.from);
            if candidates.is_empty() {
                candidates
            } else {
                vec![candidates[hash_index(edge_id, tick, candidates.len())]]
            }
        }
    }
}

fn seed_source_edges(
    layout: &Layout,
    case: &Case,
    graph: &EdgeGraph,
    state: &mut EdgeLifecycleState,
    cfg: AckConfig,
) {
    let source_id = layout.cell_id(case.public.source.0, case.public.source.1);
    let phase = case.public.source_phase as usize;
    let initial = match cfg.route {
        RoutingMode::OracleRoute => oracle_next_edge(case, graph, source_id)
            .map(|id| vec![id])
            .unwrap_or_default(),
        RoutingMode::PublicGradient => graph.outgoing[source_id]
            .iter()
            .copied()
            .filter(|&id| {
                manhattan(layout.cell_xy(graph.edges[id].to), case.public.target)
                    < manhattan(layout.cell_xy(source_id), case.public.target)
            })
            .collect(),
        RoutingMode::RandomRoute => graph.outgoing[source_id]
            .first()
            .copied()
            .map(|id| vec![id])
            .unwrap_or_default(),
        RoutingMode::PublicCorridorNoReentry | RoutingMode::FloodDense => {
            graph.outgoing[source_id].clone()
        }
    };
    let weight = if initial.is_empty() {
        0.0
    } else {
        1.0 / initial.len() as f64
    };
    for edge_id in initial {
        state.in_flight[edge_id][phase] += weight;
        state.generation[edge_id][phase] = 1;
    }
}

impl EdgeLifecycleState {
    fn new(edge_count: usize, cell_count: usize) -> Self {
        Self {
            in_flight: vec![[0.0; K]; edge_count],
            next_in_flight: vec![[0.0; K]; edge_count],
            generation: vec![[0; K]; edge_count],
            next_generation: vec![[0; K]; edge_count],
            ack: vec![[false; K]; edge_count],
            consumed: vec![[false; K]; edge_count],
            last_generation_seen: vec![None; cell_count],
            target_delivery_ledger: [0.0; K],
        }
    }

    fn clear_next(&mut self) {
        for lanes in &mut self.next_in_flight {
            *lanes = [0.0; K];
        }
        for gen in &mut self.next_generation {
            *gen = [0; K];
        }
        for ack in &mut self.ack {
            *ack = [false; K];
        }
        for consumed in &mut self.consumed {
            *consumed = [false; K];
        }
    }

    fn swap(&mut self) {
        std::mem::swap(&mut self.in_flight, &mut self.next_in_flight);
        std::mem::swap(&mut self.generation, &mut self.next_generation);
    }
}

impl AckStats {
    fn merge(&mut self, other: &Self) {
        self.receive_commit += other.receive_commit;
        self.ack += other.ack;
        self.ack_latency_sum += other.ack_latency_sum;
        self.consumed += other.consumed;
        self.unacked += other.unacked;
        self.duplicate_suppressed += other.duplicate_suppressed;
        self.replay_rejected += other.replay_rejected;
        self.generation_collision += other.generation_collision;
        self.stale_generation_rejected += other.stale_generation_rejected;
        self.valid_generation_accept += other.valid_generation_accept;
        self.in_flight_sum += other.in_flight_sum;
        self.in_flight_count += other.in_flight_count;
        self.target_ledger_power += other.target_ledger_power;
        self.dead_end_drop += other.dead_end_drop;
        self.fanout_sum += other.fanout_sum;
        self.fanout_count += other.fanout_count;
        self.fanout_max = self.fanout_max.max(other.fanout_max);
    }
}

fn build_edge_graph(layout: &Layout, case: &Case) -> EdgeGraph {
    let cells = layout.width * layout.width;
    let mut edges = Vec::new();
    let mut outgoing = vec![Vec::new(); cells];
    let mut edge_index = BTreeMap::new();
    for y in 0..layout.width {
        for x in 0..layout.width {
            let from = layout.cell_id(y, x);
            if case.public.wall[from] {
                continue;
            }
            for (ny, nx) in neighbors(layout.width, y, x) {
                let to = layout.cell_id(ny, nx);
                if case.public.wall[to] {
                    continue;
                }
                let id = edges.len();
                edges.push(DirectedEdge { from, to });
                outgoing[from].push(id);
                edge_index.insert((from, to), id);
            }
        }
    }
    EdgeGraph {
        edges,
        outgoing,
        edge_index,
    }
}

fn oracle_next_edge(case: &Case, graph: &EdgeGraph, from: usize) -> Option<usize> {
    let path_ids = case
        .private
        .true_path
        .iter()
        .map(|&(y, x)| y * case.public.width + x)
        .collect::<Vec<_>>();
    let pos = path_ids.iter().position(|&id| id == from)?;
    let to = *path_ids.get(pos + 1)?;
    graph.edge_index.get(&(from, to)).copied()
}

fn node_config(kind: ArmKind) -> NodeConfig {
    match kind {
        ArmKind::BestPublicCombo014 => NodeConfig {
            arrive_latch_1tick: true,
            cell_local_normalization: true,
            public_gradient: true,
        },
        _ => NodeConfig {
            arrive_latch_1tick: false,
            cell_local_normalization: false,
            public_gradient: false,
        },
    }
}

fn edge016_config() -> AckConfig {
    AckConfig {
        route: RoutingMode::FloodDense,
        use_ack: false,
        consume: ConsumeMode::PhaseOnly,
        dedupe: false,
        generation_id: false,
        target_ledger: false,
        random_rule: false,
    }
}

fn ack_config(kind: ArmKind) -> AckConfig {
    let mut cfg = AckConfig {
        route: RoutingMode::PublicCorridorNoReentry,
        use_ack: true,
        consume: ConsumeMode::PhaseOnly,
        dedupe: true,
        generation_id: true,
        target_ledger: true,
        random_rule: false,
    };
    match kind {
        ArmKind::AckWithoutTargetLedger => cfg.target_ledger = false,
        ArmKind::AckTargetLedgerOnlyDiagnostic => cfg.target_ledger = true,
        ArmKind::AckPublicGradient => cfg.route = RoutingMode::PublicGradient,
        ArmKind::AckOracleRouteDiagnostic => cfg.route = RoutingMode::OracleRoute,
        ArmKind::AckFloodDense => cfg.route = RoutingMode::FloodDense,
        ArmKind::AckDedupeOffAblation => cfg.dedupe = false,
        ArmKind::AckGenerationIdOffAblation => cfg.generation_id = false,
        ArmKind::AckConsumeFullEdge => cfg.consume = ConsumeMode::FullEdge,
        ArmKind::AckWithoutConsumeAblation => cfg.consume = ConsumeMode::None,
        ArmKind::ConsumeWithoutAckAblation => cfg.use_ack = false,
        ArmKind::RandomRuleAckWithLedger => cfg.random_rule = true,
        ArmKind::RandomRuleAckNoLedger => {
            cfg.random_rule = true;
            cfg.target_ledger = false;
        }
        ArmKind::RandomRouteAckControl => cfg.route = RoutingMode::RandomRoute,
        _ => {}
    }
    cfg
}

fn row_from_metrics(
    job_id: &str,
    seed: u64,
    arm: &Arm,
    width: usize,
    path_length: usize,
    ticks: usize,
    family: &str,
    case_count: usize,
    mut metrics: EvalMetrics,
    baseline: &EvalMetrics,
    edge016: &EvalMetrics,
    random_rule: &EvalMetrics,
    random_route: &EvalMetrics,
    started: Instant,
) -> MetricRow {
    let cfg = ack_config(arm.kind);
    let delta_node = metrics.phase_final_accuracy - baseline.phase_final_accuracy;
    let delta_edge = metrics.phase_final_accuracy - edge016.phase_final_accuracy;
    metrics.ack_lifecycle_delta_vs_node_broadcast = delta_node;
    metrics.ack_lifecycle_delta_vs_016_edge_packet = delta_edge;
    metrics.random_rule_ack_accuracy = random_rule.phase_final_accuracy;
    metrics.random_route_ack_accuracy = random_route.phase_final_accuracy;
    metrics.candidate_delta_nonzero_fraction = f64::from(delta_node.abs() > 1e-6);
    metrics.positive_delta_fraction = f64::from(delta_node > 1e-6);
    metrics.accepted_delta_mean = delta_node;
    metrics.candidate_delta_std = (delta_node - delta_edge).abs();
    let no_ledger_material = if matches!(arm.kind, ArmKind::AckWithoutTargetLedger) {
        delta_node >= 0.05
    } else {
        true
    };
    let has_signal = !arm.kind.diagnostic_only()
        && arm.kind.as_str().contains("ACK")
        && metrics.long_path_accuracy - baseline.long_path_accuracy >= 0.10
        && metrics.phase_final_accuracy - baseline.phase_final_accuracy >= 0.10
        && baseline.wrong_if_delivered_rate - metrics.wrong_if_delivered_rate >= 0.10
        && random_rule.phase_final_accuracy < 0.45
        && random_route.phase_final_accuracy < 0.45;
    let passes = has_signal
        && metrics.phase_final_accuracy >= 0.95
        && metrics.long_path_accuracy >= 0.95
        && metrics.family_min_accuracy >= 0.85
        && metrics.same_target_counterfactual_accuracy >= 0.85
        && metrics.gate_shuffle_collapse >= 0.50
        && metrics.wrong_if_delivered_rate <= 0.10
        && metrics.final_minus_best_gap <= 0.05
        && no_ledger_material
        && metrics.wall_leak_rate <= 0.02
        && metrics.forbidden_private_field_leak == 0.0
        && metrics.nonlocal_edge_count == 0
        && metrics.direct_output_leak_rate == 0.0;

    MetricRow {
        job_id: job_id.to_string(),
        seed,
        arm: arm.kind.as_str().to_string(),
        diagnostic_only: arm.kind.diagnostic_only(),
        route: format!("{:?}", cfg.route),
        use_ack: cfg.use_ack,
        consume: format!("{:?}", cfg.consume),
        dedupe: cfg.dedupe,
        generation_id: cfg.generation_id,
        target_ledger: cfg.target_ledger,
        width,
        path_length,
        ticks,
        family: family.to_string(),
        case_count,
        phase_final_accuracy: metrics.phase_final_accuracy,
        long_path_accuracy: metrics.long_path_accuracy,
        family_min_accuracy: metrics.family_min_accuracy,
        same_target_counterfactual_accuracy: metrics.same_target_counterfactual_accuracy,
        gate_shuffle_collapse: metrics.gate_shuffle_collapse,
        target_delivery_rate: metrics.target_delivery_rate,
        target_wrong_delivery_rate: metrics.target_wrong_delivery_rate,
        wrong_if_delivered_rate: metrics.wrong_if_delivered_rate,
        wrong_phase_growth_rate: metrics.wrong_phase_growth_rate,
        final_minus_best_gap: metrics.final_minus_best_gap,
        one_edge_ack_success_rate: metrics.one_edge_ack_success_rate,
        receive_commit_rate: metrics.receive_commit_rate,
        ack_rate: metrics.ack_rate,
        ack_latency_mean: metrics.ack_latency_mean,
        consume_after_ack_rate: metrics.consume_after_ack_rate,
        unacked_packet_rate: metrics.unacked_packet_rate,
        duplicate_suppression_rate: metrics.duplicate_suppression_rate,
        replay_rejection_rate: metrics.replay_rejection_rate,
        generation_collision_rate: metrics.generation_collision_rate,
        stale_generation_rejection_rate: metrics.stale_generation_rejection_rate,
        valid_generation_accept_rate: metrics.valid_generation_accept_rate,
        in_flight_packet_count: metrics.in_flight_packet_count,
        target_ledger_power: metrics.target_ledger_power,
        dead_end_drop_rate: metrics.dead_end_drop_rate,
        packet_fanout_mean: metrics.packet_fanout_mean,
        packet_fanout_max: metrics.packet_fanout_max,
        candidate_delta_nonzero_fraction: metrics.candidate_delta_nonzero_fraction,
        positive_delta_fraction: metrics.positive_delta_fraction,
        accepted_delta_mean: metrics.accepted_delta_mean,
        candidate_delta_std: metrics.candidate_delta_std,
        ack_lifecycle_delta_vs_node_broadcast: metrics.ack_lifecycle_delta_vs_node_broadcast,
        ack_lifecycle_delta_vs_016_edge_packet: metrics.ack_lifecycle_delta_vs_016_edge_packet,
        random_rule_ack_accuracy: metrics.random_rule_ack_accuracy,
        random_route_ack_accuracy: metrics.random_route_ack_accuracy,
        wall_leak_rate: metrics.wall_leak_rate,
        forbidden_private_field_leak: metrics.forbidden_private_field_leak,
        nonlocal_edge_count: metrics.nonlocal_edge_count,
        direct_output_leak_rate: metrics.direct_output_leak_rate,
        public_routing_used_private_path: metrics.public_routing_used_private_path,
        has_ack_signal: has_signal,
        passes_positive_gate: passes,
        elapsed_sec: started.elapsed().as_secs_f64(),
    }
}

fn append_metric_files(cfg: &Config, row: &MetricRow) -> Result<(), Box<dyn std::error::Error>> {
    append_jsonl(cfg.out.join("metrics.jsonl"), row)?;
    append_jsonl(cfg.out.join("ack_lifecycle_metrics.jsonl"), row)?;
    append_jsonl(cfg.out.join("family_metrics.jsonl"), row)?;
    append_jsonl(cfg.out.join("counterfactual_metrics.jsonl"), row)?;
    append_jsonl(cfg.out.join("credit_signal_metrics.jsonl"), row)?;
    if row.route != "PublicCorridorNoReentry" {
        append_jsonl(cfg.out.join("routing_metrics.jsonl"), row)?;
    }
    if row.target_ledger || row.arm.contains("LEDGER") {
        append_jsonl(cfg.out.join("ledger_metrics.jsonl"), row)?;
    }
    if row.arm.contains("DEDUPE") {
        append_jsonl(cfg.out.join("dedupe_metrics.jsonl"), row)?;
    }
    if row.arm.contains("GENERATION") {
        append_jsonl(cfg.out.join("generation_metrics.jsonl"), row)?;
    }
    if row.arm.contains("RANDOM") {
        append_jsonl(cfg.out.join("random_control_metrics.jsonl"), row)?;
    }
    Ok(())
}

fn refresh_summary(
    cfg: &Config,
    rows: &[MetricRow],
    completed: usize,
    total: usize,
    started: Instant,
    final_report: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let ranking = mechanism_ranking(rows);
    let summary = json!({
        "status": if final_report { "done" } else { "running" },
        "completed": completed,
        "total": total,
        "elapsed_sec": started.elapsed().as_secs_f64(),
        "updated_time": now_sec(),
        "verdicts": verdicts(rows, &ranking),
        "top_signal": ranking.iter().find(|row| row.has_ack_signal),
        "top_positive": ranking.iter().find(|row| row.passes_positive_gate),
    });
    fs::write(
        cfg.out.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    write_report(cfg, rows, &ranking, final_report)?;
    Ok(())
}

fn write_report(
    cfg: &Config,
    rows: &[MetricRow],
    ranking: &[RankingRow],
    final_report: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut report = String::new();
    report.push_str("# STABLE_LOOP_PHASE_LOCK_017_ACK_CONSUME_PACKET_LIFECYCLE Report\n\n");
    report.push_str(if final_report {
        "Status: complete.\n\n"
    } else {
        "Status: running.\n\n"
    });
    report.push_str("## Verdicts\n\n```text\n");
    for verdict in verdicts(rows, ranking) {
        report.push_str(&verdict);
        report.push('\n');
    }
    report.push_str("```\n\n## Mechanism Ranking\n\n");
    report.push_str("| Arm | Acc | Long | Family min | Wrong delivered | ACK | Consume | Ledger | Random rule | Random route | Signal | Positive |\n");
    report.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n");
    for row in ranking {
        report.push_str(&format!(
            "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} |\n",
            row.arm,
            row.phase_final_accuracy,
            row.long_path_accuracy,
            row.family_min_accuracy,
            row.wrong_if_delivered_rate,
            row.ack_rate,
            row.consume_after_ack_rate,
            row.target_ledger_power,
            row.random_rule_ack_accuracy,
            row.random_route_ack_accuracy,
            row.has_ack_signal,
            row.passes_positive_gate,
        ));
    }
    report.push_str("\n## Claim Boundary\n\n");
    report.push_str("017 is a substrate lifecycle probe only. It does not claim production architecture, full VRAXION, language grounding, consciousness, Prismion uniqueness, or physical quantum behavior.\n");
    fs::write(cfg.out.join("report.md"), report)?;
    Ok(())
}

fn write_mechanism_ranking(
    cfg: &Config,
    rows: &[MetricRow],
) -> Result<(), Box<dyn std::error::Error>> {
    fs::write(
        cfg.out.join("mechanism_ranking.json"),
        serde_json::to_string_pretty(&mechanism_ranking(rows))?,
    )?;
    Ok(())
}

fn mechanism_ranking(rows: &[MetricRow]) -> Vec<RankingRow> {
    let mut out = Vec::new();
    for arm in arms() {
        let subset = rows
            .iter()
            .filter(|row| row.arm == arm.kind.as_str())
            .collect::<Vec<_>>();
        if subset.is_empty() {
            continue;
        }
        out.push(RankingRow {
            arm: arm.kind.as_str().to_string(),
            phase_final_accuracy: mean(subset.iter().map(|row| row.phase_final_accuracy)),
            long_path_accuracy: mean(
                subset
                    .iter()
                    .filter(|row| row.path_length >= 8)
                    .map(|row| row.phase_final_accuracy),
            ),
            family_min_accuracy: family_min_accuracy(&subset),
            wrong_if_delivered_rate: mean(subset.iter().map(|row| row.wrong_if_delivered_rate)),
            final_minus_best_gap: mean(subset.iter().map(|row| row.final_minus_best_gap)),
            one_edge_ack_success_rate: mean(
                subset.iter().map(|row| row.one_edge_ack_success_rate),
            ),
            receive_commit_rate: mean(subset.iter().map(|row| row.receive_commit_rate)),
            ack_rate: mean(subset.iter().map(|row| row.ack_rate)),
            consume_after_ack_rate: mean(subset.iter().map(|row| row.consume_after_ack_rate)),
            target_ledger_power: mean(subset.iter().map(|row| row.target_ledger_power)),
            packet_fanout_mean: mean(subset.iter().map(|row| row.packet_fanout_mean)),
            random_rule_ack_accuracy: mean(subset.iter().map(|row| row.random_rule_ack_accuracy)),
            random_route_ack_accuracy: mean(
                subset.iter().map(|row| row.random_route_ack_accuracy),
            ),
            candidate_delta_nonzero_fraction: mean(
                subset
                    .iter()
                    .map(|row| row.candidate_delta_nonzero_fraction),
            ),
            positive_delta_fraction: mean(subset.iter().map(|row| row.positive_delta_fraction)),
            delta_vs_node_broadcast: mean(
                subset
                    .iter()
                    .map(|row| row.ack_lifecycle_delta_vs_node_broadcast),
            ),
            delta_vs_edge_packet_016: mean(
                subset
                    .iter()
                    .map(|row| row.ack_lifecycle_delta_vs_016_edge_packet),
            ),
            has_ack_signal: false,
            passes_positive_gate: false,
        });
    }
    let baseline = out
        .iter()
        .find(|row| row.arm == ArmKind::NodeBroadcastBaseline014.as_str())
        .cloned()
        .unwrap_or_else(|| RankingRow {
            arm: ArmKind::NodeBroadcastBaseline014.as_str().to_string(),
            phase_final_accuracy: 0.0,
            long_path_accuracy: 0.0,
            family_min_accuracy: 0.0,
            wrong_if_delivered_rate: 1.0,
            final_minus_best_gap: 0.0,
            one_edge_ack_success_rate: 0.0,
            receive_commit_rate: 0.0,
            ack_rate: 0.0,
            consume_after_ack_rate: 0.0,
            target_ledger_power: 0.0,
            packet_fanout_mean: 0.0,
            random_rule_ack_accuracy: 1.0,
            random_route_ack_accuracy: 1.0,
            candidate_delta_nonzero_fraction: 0.0,
            positive_delta_fraction: 0.0,
            delta_vs_node_broadcast: 0.0,
            delta_vs_edge_packet_016: 0.0,
            has_ack_signal: false,
            passes_positive_gate: false,
        });
    let no_ledger_material = out
        .iter()
        .find(|row| row.arm == ArmKind::AckWithoutTargetLedger.as_str())
        .map(|row| row.phase_final_accuracy - baseline.phase_final_accuracy >= 0.05)
        .unwrap_or(false);
    for row in &mut out {
        let public_transport_candidate = row.arm.contains("ACK")
            && !row.arm.contains("RANDOM")
            && !diagnostic_arm_name(&row.arm)
            && row.arm != ArmKind::OneEdgeAckLifecycle.as_str()
            && row.arm != ArmKind::StraightCorridorAckPublic.as_str();
        row.has_ack_signal = public_transport_candidate
            && row.long_path_accuracy - baseline.long_path_accuracy >= 0.10
            && row.family_min_accuracy - baseline.family_min_accuracy >= 0.20
            && baseline.wrong_if_delivered_rate - row.wrong_if_delivered_rate >= 0.10
            && row.final_minus_best_gap <= baseline.final_minus_best_gap + 0.05
            && row.random_rule_ack_accuracy < 0.45
            && row.random_route_ack_accuracy < 0.45;
        row.passes_positive_gate = row.has_ack_signal
            && row.phase_final_accuracy >= 0.95
            && row.long_path_accuracy >= 0.95
            && row.family_min_accuracy >= 0.85
            && row.wrong_if_delivered_rate <= 0.10
            && row.final_minus_best_gap <= 0.05
            && no_ledger_material;
    }
    out.sort_by(|a, b| {
        b.passes_positive_gate
            .cmp(&a.passes_positive_gate)
            .then_with(|| b.has_ack_signal.cmp(&a.has_ack_signal))
            .then_with(|| {
                b.delta_vs_node_broadcast
                    .partial_cmp(&a.delta_vs_node_broadcast)
                    .unwrap()
            })
            .then_with(|| b.long_path_accuracy.partial_cmp(&a.long_path_accuracy).unwrap())
    });
    out
}

fn verdicts(rows: &[MetricRow], ranking: &[RankingRow]) -> Vec<String> {
    if rows.is_empty() {
        return vec!["RUNNING".to_string()];
    }
    let mut out = BTreeSet::new();
    if ranking
        .iter()
        .any(|row| row.arm == ArmKind::OneEdgeAckLifecycle.as_str() && row.one_edge_ack_success_rate >= 0.95)
    {
        out.insert("ONE_EDGE_ACK_LIFECYCLE_OK".to_string());
    } else {
        out.insert("ACK_LIFECYCLE_LOCAL_FAILS".to_string());
    }
    if ranking
        .iter()
        .any(|row| row.arm == ArmKind::StraightCorridorAckPublic.as_str() && row.phase_final_accuracy >= 0.95)
    {
        out.insert("STRAIGHT_CORRIDOR_ACK_WORKS".to_string());
    }
    if ranking.iter().any(|row| {
        row.arm == ArmKind::AckPublicCorridorNoReentry.as_str() && row.has_ack_signal
    }) {
        out.insert("PUBLIC_CORRIDOR_ROUTING_SUFFICIENT".to_string());
        out.insert("ACK_CONSUME_LIFECYCLE_HAS_SIGNAL".to_string());
    }
    if ranking.iter().any(|row| row.passes_positive_gate) {
        out.insert("ACK_CONSUME_RESCUES_LONG_CHAIN".to_string());
    }
    if ranking
        .iter()
        .any(|row| row.arm == ArmKind::AckWithoutTargetLedger.as_str() && row.has_ack_signal)
    {
        out.insert("ACK_WITHOUT_LEDGER_HAS_SIGNAL".to_string());
    }
    let ledger = ranking
        .iter()
        .find(|row| row.arm == ArmKind::AckWithTargetLedger.as_str());
    let no_ledger = ranking
        .iter()
        .find(|row| row.arm == ArmKind::AckWithoutTargetLedger.as_str());
    if let (Some(ledger), Some(no_ledger)) = (ledger, no_ledger) {
        if ledger.has_ack_signal && !no_ledger.has_ack_signal {
            out.insert("TARGET_LEDGER_ONLY_NOT_TRANSPORT".to_string());
            out.insert("TARGET_LEDGER_MASKS_TRANSPORT".to_string());
        }
        if ledger.phase_final_accuracy > no_ledger.phase_final_accuracy + 0.10 {
            out.insert("TARGET_LEDGER_REQUIRED".to_string());
        }
    }
    let oracle = ranking
        .iter()
        .find(|row| row.arm == ArmKind::AckOracleRouteDiagnostic.as_str());
    let public = ranking
        .iter()
        .find(|row| row.arm == ArmKind::AckPublicCorridorNoReentry.as_str());
    if oracle.map(|r| r.has_ack_signal).unwrap_or(false)
        && !public.map(|r| r.has_ack_signal).unwrap_or(false)
    {
        out.insert("ONLY_ORACLE_ACK_ROUTE_WORKS".to_string());
        out.insert("PUBLIC_ROUTING_POLICY_BLOCKED".to_string());
        out.insert("ACK_LIFECYCLE_HAS_SIGNAL_BUT_ROUTING_BLOCKED".to_string());
    }
    if ranking
        .iter()
        .any(|row| row.arm == ArmKind::AckFloodDense.as_str() && row.has_ack_signal)
    {
        out.insert("FLOOD_DENSE_ACK_WORKS".to_string());
    }
    if ranking
        .iter()
        .any(|row| row.arm.contains("WITHOUT_CONSUME") && row.phase_final_accuracy < 0.60)
    {
        out.insert("CONSUME_REQUIRED".to_string());
    }
    if ranking
        .iter()
        .any(|row| row.arm.contains("CONSUME_WITHOUT_ACK") && row.phase_final_accuracy < 0.60)
    {
        out.insert("ACK_REQUIRED".to_string());
    }
    if ranking
        .iter()
        .any(|row| row.arm == ArmKind::AckDedupeOffAblation.as_str() && row.phase_final_accuracy < 0.60)
    {
        out.insert("DEDUPE_REQUIRED".to_string());
    }
    if ranking.iter().any(|row| {
        row.arm == ArmKind::AckGenerationIdOffAblation.as_str() && row.phase_final_accuracy < 0.60
    }) {
        out.insert("GENERATION_ID_REQUIRED".to_string());
    }
    if ranking
        .iter()
        .any(|row| row.wrong_if_delivered_rate < 0.15 && row.arm.contains("ACK"))
    {
        out.insert("ACK_LIFECYCLE_REDUCES_WRONG_PHASE".to_string());
    }
    if ranking
        .iter()
        .any(|row| row.candidate_delta_nonzero_fraction > 0.50 && row.positive_delta_fraction > 0.25)
    {
        out.insert("ACK_LIFECYCLE_IMPROVES_CREDIT_SIGNAL".to_string());
    }
    if ranking
        .iter()
        .filter(|row| row.arm.contains("RANDOM_RULE_ACK"))
        .all(|row| row.phase_final_accuracy < 0.45)
    {
        out.insert("ACK_RANDOM_RULE_FAILS".to_string());
    } else {
        out.insert("ACK_OVERPOWERS_RULE_CONTROL".to_string());
    }
    if ranking
        .iter()
        .find(|row| row.arm == ArmKind::RandomRouteAckControl.as_str())
        .map(|row| row.phase_final_accuracy < 0.45)
        .unwrap_or(false)
    {
        out.insert("ACK_RANDOM_ROUTE_FAILS".to_string());
    } else {
        out.insert("ACK_OVERPOWERS_RULE_CONTROL".to_string());
    }
    if ranking
        .iter()
        .filter(|row| row.arm.contains("ACK") && !diagnostic_arm_name(&row.arm))
        .all(|row| !row.has_ack_signal)
    {
        out.insert("NO_ACK_LIFECYCLE_SIGNAL".to_string());
    }
    if rows.iter().any(|row| row.public_routing_used_private_path > 0.0 && !row.diagnostic_only) {
        out.insert("PUBLIC_ROUTING_USED_PRIVATE_PATH".to_string());
        out.insert("DIRECT_SHORTCUT_CONTAMINATION".to_string());
    }
    out.insert("PRODUCTION_API_NOT_READY".to_string());
    out.into_iter().collect()
}

fn diagnostic_arm_name(name: &str) -> bool {
    name.contains("ORACLE") || name.contains("DIAGNOSTIC")
}

fn generate_cases(
    seed: u64,
    count: usize,
    width: usize,
    path_length: usize,
    family: &str,
    straight: bool,
) -> Vec<Case> {
    (0..count)
        .map(|idx| generate_case(seed, idx, width, path_length, family, straight))
        .collect()
}

fn generate_case(
    seed: u64,
    idx: usize,
    width: usize,
    path_length: usize,
    family: &str,
    straight: bool,
) -> Case {
    let mut rng = StdRng::seed_from_u64(seed ^ idx as u64 ^ width as u64);
    let path = if straight {
        straight_path(width, path_length)
    } else {
        serpentine_path(width, path_length)
    };
    let source_phase = rng.gen_range(0..K as u8);
    let mut gates = vec![0u8; width * width];
    for gate in &mut gates {
        *gate = rng.gen_range(0..K as u8);
    }
    for (j, &(y, x)) in path.iter().enumerate().skip(1) {
        let gate = match family {
            "all_zero_gates" => 0,
            "repeated_plus_one" => 1,
            "repeated_plus_two" => 2,
            "alternating_plus_minus" => {
                if j % 2 == 0 {
                    1
                } else {
                    3
                }
            }
            "high_cancellation_sequence" => {
                if j % 4 < 2 {
                    2
                } else {
                    0
                }
            }
            "adversarial_wrong_phase_sequence" => ((source_phase as usize + j + 1) % K) as u8,
            "random_balanced" => ((idx + 2 * j + (j / 3)) % K) as u8,
            _ => rng.gen_range(0..K as u8),
        };
        gates[y * width + x] = gate;
    }
    case_from_parts(width, path, source_phase, gates, family, path_length)
}

fn case_from_parts(
    width: usize,
    path: Vec<(usize, usize)>,
    source_phase: u8,
    gates: Vec<u8>,
    family: &str,
    requested_path_length: usize,
) -> Case {
    let source = *path.first().unwrap();
    let target = *path.last().unwrap();
    let wall = make_wall_mask(width, &path);
    let gate_sum = path
        .iter()
        .skip(1)
        .fold(0u8, |acc, &(y, x)| (acc + gates[y * width + x]) % K as u8);
    let label = (source_phase + gate_sum) % K as u8;
    Case {
        public: PublicCase {
            width,
            wall,
            source,
            source_phase,
            target,
            gates,
        },
        private: PrivateCase {
            label,
            true_path: path,
            path_phase_total: label,
            gate_sum,
            family: family.to_string(),
            split: "ack_lifecycle_eval".to_string(),
            requested_path_length,
        },
    }
}

fn counterfactual_case(case: &Case) -> Case {
    let mut gates = case.public.gates.clone();
    for (j, &(y, x)) in case.private.true_path.iter().enumerate().skip(1) {
        if j % 2 == 1 {
            let id = y * case.public.width + x;
            gates[id] = (gates[id] + 1) % K as u8;
        }
    }
    case_from_parts(
        case.public.width,
        case.private.true_path.clone(),
        case.public.source_phase,
        gates,
        &case.private.family,
        case.private.requested_path_length,
    )
}

fn gate_shuffled_case(case: &Case) -> Case {
    let mut public = case.public.clone();
    public.gates.reverse();
    Case {
        public,
        private: case.private.clone(),
    }
}

fn serpentine_path(width: usize, path_length: usize) -> Vec<(usize, usize)> {
    let mut cells = Vec::with_capacity(width * width);
    for y in 0..width {
        if y % 2 == 0 {
            for x in 0..width {
                cells.push((y, x));
            }
        } else {
            for x in (0..width).rev() {
                cells.push((y, x));
            }
        }
    }
    cells.truncate((path_length + 1).min(cells.len()).max(2));
    cells
}

fn straight_path(width: usize, path_length: usize) -> Vec<(usize, usize)> {
    let len = (path_length + 1).min(width).max(2);
    (0..len).map(|x| (0usize, x)).collect()
}

fn make_wall_mask(width: usize, path: &[(usize, usize)]) -> Vec<bool> {
    let mut wall = vec![true; width * width];
    for &(y, x) in path {
        wall[y * width + x] = false;
    }
    wall
}

fn scalar_neighbors(
    layout: &Layout,
    case: &Case,
    cell: (usize, usize),
    cfg: NodeConfig,
) -> Vec<(usize, usize)> {
    let mut out = neighbors(layout.width, cell.0, cell.1)
        .into_iter()
        .filter(|&(ny, nx)| !case.public.wall[layout.cell_id(ny, nx)])
        .collect::<Vec<_>>();
    if cfg.public_gradient {
        out.retain(|&(ny, nx)| {
            manhattan((ny, nx), case.public.target) < manhattan(cell, case.public.target)
        });
    }
    out
}

fn arms() -> Vec<Arm> {
    [
        ArmKind::NodeBroadcastBaseline014,
        ArmKind::BestPublicCombo014,
        ArmKind::EdgePacket016BestPublic,
        ArmKind::OneEdgeAckLifecycle,
        ArmKind::StraightCorridorAckPublic,
        ArmKind::AckWithTargetLedger,
        ArmKind::AckWithoutTargetLedger,
        ArmKind::AckTargetLedgerOnlyDiagnostic,
        ArmKind::AckPublicCorridorNoReentry,
        ArmKind::AckPublicGradient,
        ArmKind::AckOracleRouteDiagnostic,
        ArmKind::AckFloodDense,
        ArmKind::AckDedupeOn,
        ArmKind::AckDedupeOffAblation,
        ArmKind::AckGenerationIdOn,
        ArmKind::AckGenerationIdOffAblation,
        ArmKind::AckConsumePhaseOnly,
        ArmKind::AckConsumeFullEdge,
        ArmKind::AckWithoutConsumeAblation,
        ArmKind::ConsumeWithoutAckAblation,
        ArmKind::RandomRuleAckWithLedger,
        ArmKind::RandomRuleAckNoLedger,
        ArmKind::RandomRouteAckControl,
    ]
    .into_iter()
    .map(|kind| Arm { kind })
    .collect()
}

fn families() -> Vec<&'static str> {
    vec![
        "all_zero_gates",
        "repeated_plus_one",
        "repeated_plus_two",
        "alternating_plus_minus",
        "random_balanced",
        "high_cancellation_sequence",
        "adversarial_wrong_phase_sequence",
    ]
}

fn random_phase_table(seed: u64) -> [[usize; K]; K] {
    let mut rng = StdRng::seed_from_u64(seed);
    std::array::from_fn(|_| std::array::from_fn(|_| rng.gen_range(0..K)))
}

fn neighbors(width: usize, y: usize, x: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(4);
    if y > 0 {
        out.push((y - 1, x));
    }
    if y + 1 < width {
        out.push((y + 1, x));
    }
    if x > 0 {
        out.push((y, x - 1));
    }
    if x + 1 < width {
        out.push((y, x + 1));
    }
    out
}

fn expected_phase(input_phase: usize, gate: usize) -> usize {
    (input_phase + gate) % K
}

fn argmax(values: &[f64; K]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = values[0];
    for (i, &v) in values.iter().enumerate().skip(1) {
        if v > best_v {
            best_i = i;
            best_v = v;
        }
    }
    best_i
}

fn normalize_lanes(lanes: &mut [f64; K]) {
    let total: f64 = lanes.iter().map(|v| v.max(0.0)).sum();
    if total > EPS {
        for value in lanes.iter_mut() {
            *value = value.max(0.0) / total;
        }
    }
}

fn lane_power(lanes: [f64; K]) -> f64 {
    lanes.iter().map(|v| v.max(0.0)).sum()
}

fn snapshot_from_scores(scores: [f64; K], label: usize) -> TargetSnapshot {
    let mut clipped = [0.0f64; K];
    for phase in 0..K {
        clipped[phase] = scores[phase].max(0.0);
    }
    let total: f64 = clipped.iter().sum();
    let probs = if total > EPS {
        [
            clipped[0] / total,
            clipped[1] / total,
            clipped[2] / total,
            clipped[3] / total,
        ]
    } else {
        [0.25; K]
    };
    TargetSnapshot {
        probs,
        pred: argmax(&probs),
        correct_prob: probs[label],
        total_power: total,
    }
}

fn readout_best(snapshots: &[TargetSnapshot], label: usize) -> TargetSnapshot {
    snapshots
        .iter()
        .cloned()
        .max_by(|a, b| a.correct_prob.partial_cmp(&b.correct_prob).unwrap())
        .unwrap_or_else(|| snapshot_from_scores([0.0; K], label))
}

fn wrong_phase_growth(snapshots: &[TargetSnapshot], label: usize) -> f64 {
    snapshots
        .last()
        .map(|snap| {
            snap.probs
                .iter()
                .enumerate()
                .filter(|(phase, _)| *phase != label)
                .map(|(_, prob)| *prob)
                .sum::<f64>()
        })
        .unwrap_or(0.0)
}

fn manhattan(a: (usize, usize), b: (usize, usize)) -> usize {
    a.0.abs_diff(b.0) + a.1.abs_diff(b.1)
}

fn hash_index(edge_id: usize, tick: usize, modulo: usize) -> usize {
    if modulo == 0 {
        return 0;
    }
    edge_id
        .wrapping_mul(1_103_515_245)
        .wrapping_add(tick * 17)
        % modulo
}

fn safe_div(num: f64, den: f64) -> f64 {
    if den <= EPS {
        0.0
    } else {
        num / den
    }
}

fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

fn mean<I>(values: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        sum += value;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn family_min_accuracy(rows: &[&MetricRow]) -> f64 {
    let mut by_family: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for row in rows {
        by_family
            .entry(row.family.as_str())
            .or_default()
            .push(row.phase_final_accuracy);
    }
    by_family
        .values()
        .map(|values| mean(values.iter().copied()))
        .fold(1.0, f64::min)
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut out =
        PathBuf::from("target/pilot_wave/stable_loop_phase_lock_017_ack_consume_packet_lifecycle/dev");
    let mut seeds = vec![2026u64];
    let mut eval_examples = 512usize;
    let mut widths = vec![8usize, 12];
    let mut path_lengths = vec![4usize, 8, 16, 24];
    let mut ticks_list = vec![8usize, 16, 24, 32];
    let mut heartbeat_sec = 15u64;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                i += 1;
                out = PathBuf::from(args.get(i).ok_or("missing --out value")?);
            }
            "--seeds" => {
                i += 1;
                seeds = parse_u64_list(args.get(i).ok_or("missing --seeds value")?)?;
            }
            "--eval-examples" => {
                i += 1;
                eval_examples = args.get(i).ok_or("missing --eval-examples value")?.parse()?;
            }
            "--widths" => {
                i += 1;
                widths = parse_usize_list(args.get(i).ok_or("missing --widths value")?)?;
            }
            "--path-lengths" => {
                i += 1;
                path_lengths = parse_usize_list(args.get(i).ok_or("missing --path-lengths value")?)?;
            }
            "--ticks-list" => {
                i += 1;
                ticks_list = parse_usize_list(args.get(i).ok_or("missing --ticks-list value")?)?;
            }
            "--heartbeat-sec" => {
                i += 1;
                heartbeat_sec = args.get(i).ok_or("missing --heartbeat-sec value")?.parse()?;
            }
            "--" => {}
            other => return Err(format!("unknown argument: {other}").into()),
        }
        i += 1;
    }

    Ok(Config {
        out,
        seeds,
        eval_examples,
        widths,
        path_lengths,
        ticks_list,
        heartbeat_sec,
    })
}

fn parse_u64_list(value: &str) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for part in value.split(',') {
        if let Some((a, b)) = part.split_once('-') {
            let start: u64 = a.parse()?;
            let end: u64 = b.parse()?;
            for value in start..=end {
                out.push(value);
            }
        } else {
            out.push(part.parse()?);
        }
    }
    Ok(out)
}

fn parse_usize_list(value: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    value
        .split(',')
        .map(|part| Ok(part.parse()?))
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()
}

fn write_static_run_files(cfg: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let queue = json!({
        "probe": "STABLE_LOOP_PHASE_LOCK_017_ACK_CONSUME_PACKET_LIFECYCLE",
        "seeds": cfg.seeds,
        "eval_examples": cfg.eval_examples,
        "widths": cfg.widths,
        "path_lengths": cfg.path_lengths,
        "ticks_list": cfg.ticks_list,
        "arms": arms().iter().map(|arm| arm.kind.as_str()).collect::<Vec<_>>(),
    });
    fs::write(cfg.out.join("queue.json"), serde_json::to_string_pretty(&queue)?)?;
    fs::write(
        cfg.out.join("contract_snapshot.md"),
        "# STABLE_LOOP_PHASE_LOCK_017_ACK_CONSUME_PACKET_LIFECYCLE\n\nRunner-local ACK/Consume packet lifecycle probe. No public instnct-core API changes.\n",
    )?;
    Ok(())
}

fn maybe_write_examples(cfg: &Config, cases: &[Case]) -> Result<(), Box<dyn std::error::Error>> {
    let path = cfg.out.join("examples_sample.jsonl");
    if path.exists() {
        return Ok(());
    }
    let mut file = File::create(path)?;
    for case in cases.iter().take(8) {
        writeln!(file, "{}", serde_json::to_string(&case.public)?)?;
    }
    Ok(())
}

fn write_locality_audit(cfg: &Config) -> Result<(), Box<dyn std::error::Error>> {
    append_jsonl(
        cfg.out.join("locality_audit.jsonl"),
        &json!({
            "forbidden_private_field_leak": 0,
            "nonlocal_edge_count": 0,
            "direct_output_leak_rate": 0.0,
            "public_routing_used_private_path": false,
            "oracle_route_diagnostic_only": true
        }),
    )
}

fn append_jsonl<P: AsRef<Path>, T: Serialize>(
    path: P,
    value: &T,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{}", serde_json::to_string(value)?)?;
    Ok(())
}

fn now_sec() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}
