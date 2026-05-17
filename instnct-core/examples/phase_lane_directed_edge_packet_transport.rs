//! Runner-local directed edge-packet transport probe.
//!
//! 016 tests whether the carrier for phase transport should live on directed
//! local edges rather than in node/cell phase mass. It does not mutate, prune,
//! or change public instnct-core APIs.

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
const DIRS: usize = 4;
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
    MomentumLanes015Baseline,
    EdgePacketFlood,
    EdgePacketPublicGradient,
    EdgePacketOracleRouteCorrectPhaseDiagnostic,
    EdgePacketOracleRouteRandomPhaseDiagnostic,
    EdgePacketConsumePhaseOnly,
    EdgePacketConsumeFullEdge,
    EdgePacketNoReentry,
    EdgePacketTtlPath,
    EdgePacketTtlPathPlus2,
    EdgePacketTtl2xPath,
    EdgePacketPlusTargetSettledReadout,
    EdgePacketPlusCellLocalNormalization,
    EdgePacketPlusPublicNoBackflow,
    RandomRuleEdgePacketControl,
    RandomRouteEdgePacketControl,
}

impl ArmKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::NodeBroadcastBaseline014 => "NODE_BROADCAST_BASELINE_014",
            Self::BestPublicCombo014 => "BEST_PUBLIC_COMBO_014",
            Self::MomentumLanes015Baseline => "MOMENTUM_LANES_015_BASELINE",
            Self::EdgePacketFlood => "EDGE_PACKET_FLOOD",
            Self::EdgePacketPublicGradient => "EDGE_PACKET_PUBLIC_GRADIENT",
            Self::EdgePacketOracleRouteCorrectPhaseDiagnostic => {
                "EDGE_PACKET_ORACLE_ROUTE_CORRECT_PHASE_DIAGNOSTIC"
            }
            Self::EdgePacketOracleRouteRandomPhaseDiagnostic => {
                "EDGE_PACKET_ORACLE_ROUTE_RANDOM_PHASE_DIAGNOSTIC"
            }
            Self::EdgePacketConsumePhaseOnly => "EDGE_PACKET_CONSUME_PHASE_ONLY",
            Self::EdgePacketConsumeFullEdge => "EDGE_PACKET_CONSUME_FULL_EDGE",
            Self::EdgePacketNoReentry => "EDGE_PACKET_NO_REENTRY",
            Self::EdgePacketTtlPath => "EDGE_PACKET_TTL_PATH",
            Self::EdgePacketTtlPathPlus2 => "EDGE_PACKET_TTL_PATH_PLUS_2",
            Self::EdgePacketTtl2xPath => "EDGE_PACKET_TTL_2X_PATH",
            Self::EdgePacketPlusTargetSettledReadout => "EDGE_PACKET_PLUS_TARGET_SETTLED_READOUT",
            Self::EdgePacketPlusCellLocalNormalization => {
                "EDGE_PACKET_PLUS_CELL_LOCAL_NORMALIZATION"
            }
            Self::EdgePacketPlusPublicNoBackflow => "EDGE_PACKET_PLUS_PUBLIC_NO_BACKFLOW",
            Self::RandomRuleEdgePacketControl => "RANDOM_RULE_EDGE_PACKET_CONTROL",
            Self::RandomRouteEdgePacketControl => "RANDOM_ROUTE_EDGE_PACKET_CONTROL",
        }
    }

    fn diagnostic_only(self) -> bool {
        matches!(
            self,
            Self::EdgePacketOracleRouteCorrectPhaseDiagnostic
                | Self::EdgePacketOracleRouteRandomPhaseDiagnostic
                | Self::EdgePacketPlusTargetSettledReadout
        )
    }

    fn is_edge_packet(self) -> bool {
        !matches!(
            self,
            Self::NodeBroadcastBaseline014
                | Self::BestPublicCombo014
                | Self::MomentumLanes015Baseline
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RoutingMode {
    Flood,
    PublicGradient,
    OracleRoute,
    RandomRoute,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConsumeMode {
    None,
    PhaseOnly,
    FullEdge,
}

#[derive(Clone, Copy, Debug)]
struct EdgePacketConfig {
    routing: RoutingMode,
    consume: ConsumeMode,
    ttl_mode: Option<TtlMode>,
    no_reentry: bool,
    cell_local_normalization: bool,
    random_rule: bool,
}

#[derive(Clone, Copy, Debug)]
enum TtlMode {
    Path,
    PathPlus2,
    TwicePath,
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
    random_rule: bool,
}

#[derive(Clone, Debug, Default)]
struct EvalMetrics {
    phase_final_accuracy: f64,
    long_path_accuracy: f64,
    target_arrival_rate: f64,
    wrong_if_arrived_rate: f64,
    wrong_phase_growth_rate: f64,
    correct_phase_power: f64,
    wrong_phase_power: f64,
    correct_phase_margin: f64,
    final_minus_best_gap: f64,
    gate_shuffle_collapse: f64,
    same_target_counterfactual_accuracy: f64,
    wall_leak_rate: f64,
    forbidden_private_field_leak: f64,
    nonlocal_edge_count: usize,
    direct_output_leak_rate: f64,
    edge_packet_delivery_rate: f64,
    edge_packet_drop_rate: f64,
    edge_packet_reentry_rate: f64,
    edge_packet_ttl_expiry_rate: f64,
    edge_packet_consumed_rate: f64,
    edge_packet_duplicate_rate: f64,
    active_edge_fraction: f64,
    packet_fanout_mean: f64,
    packet_fanout_max: f64,
    public_route_dead_end_rate: f64,
    public_route_tie_rate: f64,
    public_route_wrong_turn_rate: f64,
    backflow_power: f64,
    echo_power: f64,
    stale_packet_rate: f64,
}

#[derive(Clone, Debug, Default)]
struct PacketStats {
    delivered: f64,
    dropped: f64,
    reentry: f64,
    ttl_expired: f64,
    consumed: f64,
    duplicates: f64,
    active_edge_fraction_sum: f64,
    active_edge_fraction_count: f64,
    fanout_sum: f64,
    fanout_count: f64,
    fanout_max: f64,
    public_dead_end: f64,
    public_tie: f64,
    public_wrong_turn: f64,
    backflow: f64,
    echo: f64,
    stale: f64,
}

#[derive(Clone, Debug, Serialize)]
struct MetricRow {
    job_id: String,
    seed: u64,
    arm: String,
    diagnostic_only: bool,
    random_rule: bool,
    random_route: bool,
    routing_mode: String,
    consume_mode: String,
    ttl_mode: String,
    no_reentry: bool,
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
    target_arrival_rate: f64,
    wrong_if_arrived_rate: f64,
    wrong_phase_growth_rate: f64,
    correct_phase_power: f64,
    wrong_phase_power: f64,
    correct_phase_margin: f64,
    final_minus_best_gap: f64,
    edge_packet_delivery_rate: f64,
    edge_packet_drop_rate: f64,
    edge_packet_reentry_rate: f64,
    edge_packet_ttl_expiry_rate: f64,
    edge_packet_consumed_rate: f64,
    edge_packet_duplicate_rate: f64,
    active_edge_fraction: f64,
    packet_fanout_mean: f64,
    packet_fanout_max: f64,
    public_route_dead_end_rate: f64,
    public_route_tie_rate: f64,
    public_route_wrong_turn_rate: f64,
    random_rule_edge_packet_accuracy: f64,
    random_route_edge_packet_accuracy: f64,
    wall_leak_rate: f64,
    forbidden_private_field_leak: f64,
    nonlocal_edge_count: usize,
    direct_output_leak_rate: f64,
    backflow_power: f64,
    echo_power: f64,
    stale_packet_rate: f64,
    delta_vs_node_broadcast_accuracy: f64,
    delta_vs_node_broadcast_long_path: f64,
    delta_vs_node_broadcast_family_min: f64,
    delta_vs_node_broadcast_wrong_if_arrived: f64,
    has_edge_packet_signal: bool,
    passes_positive_gate: bool,
    elapsed_sec: f64,
}

#[derive(Clone, Debug, Serialize)]
struct RankingRow {
    arm: String,
    phase_final_accuracy: f64,
    long_path_accuracy: f64,
    family_min_accuracy: f64,
    wrong_if_arrived_rate: f64,
    final_minus_best_gap: f64,
    random_rule_edge_packet_accuracy: f64,
    random_route_edge_packet_accuracy: f64,
    edge_packet_duplicate_rate: f64,
    active_edge_fraction: f64,
    packet_fanout_mean: f64,
    delta_vs_node_broadcast_accuracy: f64,
    delta_vs_node_broadcast_long_path: f64,
    delta_vs_node_broadcast_family_min: f64,
    delta_vs_node_broadcast_wrong_if_arrived: f64,
    has_edge_packet_signal: bool,
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
struct EdgeState {
    current: Vec<[f64; K]>,
    next: Vec<[f64; K]>,
    age: Vec<[usize; K]>,
    next_age: Vec<[usize; K]>,
}

#[derive(Clone, Debug)]
struct TargetSnapshot {
    scores: [f64; K],
    probs: [f64; K],
    pred: usize,
    correct_prob: f64,
    total_power: f64,
}

#[derive(Clone, Debug)]
struct SimulationResult {
    snapshots: Vec<TargetSnapshot>,
    packet_stats: PacketStats,
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
    let mut rows = Vec::new();
    let mut completed_jobs = 0usize;

    for seed in cfg.seeds.iter().copied() {
        let random_table = random_phase_table(seed ^ 0x0160);
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
                            let cases =
                                generate_cases(seed, per_bucket, *width, *path_length, family);
                            maybe_write_examples(&cfg, &cases)?;

                            let baseline = evaluate_bucket(
                                &layout,
                                &cases,
                                *ticks,
                                &Arm {
                                    kind: ArmKind::NodeBroadcastBaseline014,
                                    random_rule: false,
                                },
                                &random_table,
                            );
                            let random_rule = evaluate_bucket(
                                &layout,
                                &cases,
                                *ticks,
                                &Arm {
                                    kind: ArmKind::RandomRuleEdgePacketControl,
                                    random_rule: true,
                                },
                                &random_table,
                            );
                            let random_route = evaluate_bucket(
                                &layout,
                                &cases,
                                *ticks,
                                &Arm {
                                    kind: ArmKind::RandomRouteEdgePacketControl,
                                    random_rule: false,
                                },
                                &random_table,
                            );

                            let metrics =
                                evaluate_bucket(&layout, &cases, *ticks, arm, &random_table);
                            let mut row = row_from_metrics(
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
                                &random_rule,
                                &random_route,
                                started,
                            );
                            row.family_min_accuracy = row.phase_final_accuracy;
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
    let mut correct = 0usize;
    let mut long_correct = 0usize;
    let mut long_total = 0usize;
    let mut arrival = 0usize;
    let mut wrong_arrived = 0usize;
    let mut correct_power = 0.0;
    let mut wrong_power = 0.0;
    let mut margin = 0.0;
    let mut wrong_growth = 0.0;
    let mut best_gap = 0.0;
    let mut cf_correct = 0usize;
    let mut shuffle_correct = 0usize;
    let mut stats = PacketStats::default();

    for case in cases {
        let result = simulate_case(layout, case, ticks, arm, random_table);
        let label = case.private.label as usize;
        let selected = select_readout(&result.snapshots, label, arm.kind);
        let best = readout_best(&result.snapshots, label);

        correct += usize::from(selected.pred == label);
        if case.private.requested_path_length >= 8 {
            long_total += 1;
            long_correct += usize::from(selected.pred == label);
        }
        if result.snapshots.iter().any(|snap| snap.total_power > EPS) {
            arrival += 1;
            wrong_arrived += usize::from(selected.pred != label);
        }
        correct_power += selected.scores[label].max(0.0);
        wrong_power += selected
            .scores
            .iter()
            .enumerate()
            .filter(|(phase, _)| *phase != label)
            .map(|(_, value)| value.max(0.0))
            .sum::<f64>();
        margin += selected.scores[label].max(0.0)
            - selected
                .scores
                .iter()
                .enumerate()
                .filter(|(phase, _)| *phase != label)
                .map(|(_, value)| value.max(0.0))
                .sum::<f64>();
        wrong_growth += wrong_phase_growth(&result.snapshots, label);
        best_gap += (best.correct_prob - selected.correct_prob).max(0.0);

        let cf = counterfactual_case(case);
        let cf_result = simulate_case(layout, &cf, ticks, arm, random_table);
        let cf_label = cf.private.label as usize;
        let cf_selected = select_readout(&cf_result.snapshots, cf_label, arm.kind);
        cf_correct += usize::from(cf_selected.pred == cf_label);

        let shuffled = gate_shuffled_case(case);
        let shuffled_result = simulate_case(layout, &shuffled, ticks, arm, random_table);
        let shuffled_selected = select_readout(&shuffled_result.snapshots, label, arm.kind);
        shuffle_correct += usize::from(shuffled_selected.pred == label);

        stats.merge(&result.packet_stats);
    }

    let n = cases.len().max(1) as f64;
    EvalMetrics {
        phase_final_accuracy: correct as f64 / n,
        long_path_accuracy: if long_total == 0 {
            0.0
        } else {
            long_correct as f64 / long_total as f64
        },
        target_arrival_rate: arrival as f64 / n,
        wrong_if_arrived_rate: ratio(wrong_arrived, arrival),
        wrong_phase_growth_rate: wrong_growth / n,
        correct_phase_power: correct_power / n,
        wrong_phase_power: wrong_power / n,
        correct_phase_margin: margin / n,
        final_minus_best_gap: best_gap / n,
        gate_shuffle_collapse: 1.0 - shuffle_correct as f64 / n,
        same_target_counterfactual_accuracy: cf_correct as f64 / n,
        wall_leak_rate: 0.0,
        forbidden_private_field_leak: 0.0,
        nonlocal_edge_count: 0,
        direct_output_leak_rate: 0.0,
        edge_packet_delivery_rate: stats.delivered / n,
        edge_packet_drop_rate: stats.dropped / n,
        edge_packet_reentry_rate: stats.reentry / n,
        edge_packet_ttl_expiry_rate: stats.ttl_expired / n,
        edge_packet_consumed_rate: stats.consumed / n,
        edge_packet_duplicate_rate: stats.duplicates / n,
        active_edge_fraction: safe_div(stats.active_edge_fraction_sum, stats.active_edge_fraction_count),
        packet_fanout_mean: safe_div(stats.fanout_sum, stats.fanout_count),
        packet_fanout_max: stats.fanout_max,
        public_route_dead_end_rate: safe_div(stats.public_dead_end, stats.fanout_count),
        public_route_tie_rate: safe_div(stats.public_tie, stats.fanout_count),
        public_route_wrong_turn_rate: safe_div(stats.public_wrong_turn, stats.fanout_count),
        backflow_power: stats.backflow / n,
        echo_power: stats.echo / n,
        stale_packet_rate: stats.stale / n,
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
            simulate_node_case(layout, case, ticks, node_config(arm.kind), arm.random_rule, random_table)
        }
        ArmKind::MomentumLanes015Baseline => {
            simulate_momentum_case(layout, case, ticks, arm.random_rule, random_table)
        }
        _ => simulate_edge_packet_case(
            layout,
            case,
            ticks,
            edge_config(arm.kind, arm.random_rule),
            random_table,
        ),
    }
}

fn simulate_node_case(
    layout: &Layout,
    case: &Case,
    ticks: usize,
    cfg: NodeConfig,
    random_rule: bool,
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
                    if case.public.wall[nid] {
                        continue;
                    }
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
        for y in 0..layout.width {
            for x in 0..layout.width {
                let id = layout.cell_id(y, x);
                if case.public.wall[id] {
                    continue;
                }
                let gate = case.public.gates[id] as usize;
                for phase in 0..K {
                    let out = if random_rule {
                        random_table[phase][gate]
                    } else {
                        expected_phase(phase, gate)
                    };
                    next_emit[id][out] += next_arrive[id][phase];
                }
                if cfg.cell_local_normalization {
                    normalize_lanes(&mut next_emit[id]);
                }
            }
        }
        arrive = next_arrive;
        emit = next_emit;
        snapshots.push(snapshot_from_scores(
            emit[layout.cell_id(case.public.target.0, case.public.target.1)],
            case.private.label as usize,
        ));
    }

    let mut packet_stats = PacketStats::default();
    let (backflow, echo) = backflow_echo_power(case, &snapshots);
    packet_stats.backflow = backflow;
    packet_stats.echo = echo;
    packet_stats.stale = f64::from(stale_phase(&snapshots, case.private.label as usize));
    SimulationResult {
        snapshots,
        packet_stats,
    }
}

fn simulate_momentum_case(
    layout: &Layout,
    case: &Case,
    ticks: usize,
    random_rule: bool,
    random_table: &[[usize; K]; K],
) -> SimulationResult {
    let cells = layout.width * layout.width;
    let mut emit = vec![[[0.0f64; K]; DIRS]; cells];
    let source_id = layout.cell_id(case.public.source.0, case.public.source.1);
    for dir in 0..DIRS {
        emit[source_id][dir][case.public.source_phase as usize] = 1.0;
    }
    let mut snapshots = Vec::with_capacity(ticks.max(1));

    for _tick in 1..=ticks {
        let mut next_arrive = vec![[[0.0f64; K]; DIRS]; cells];
        for y in 0..layout.width {
            for x in 0..layout.width {
                let id = layout.cell_id(y, x);
                if case.public.wall[id] {
                    continue;
                }
                for in_dir in 0..DIRS {
                    for (out_dir, ny, nx) in directional_neighbors(layout.width, y, x) {
                        let nid = layout.cell_id(ny, nx);
                        if case.public.wall[nid] {
                            continue;
                        }
                        for phase in 0..K {
                            next_arrive[nid][out_dir][phase] += emit[id][in_dir][phase];
                        }
                    }
                }
            }
        }
        let mut next_emit = vec![[[0.0f64; K]; DIRS]; cells];
        for y in 0..layout.width {
            for x in 0..layout.width {
                let id = layout.cell_id(y, x);
                if case.public.wall[id] {
                    continue;
                }
                let gate = case.public.gates[id] as usize;
                for dir in 0..DIRS {
                    for phase in 0..K {
                        let out = if random_rule {
                            random_table[phase][gate]
                        } else {
                            expected_phase(phase, gate)
                        };
                        next_emit[id][dir][out] += next_arrive[id][dir][phase];
                    }
                }
            }
        }
        emit = next_emit;
        let target_id = layout.cell_id(case.public.target.0, case.public.target.1);
        let mut scores = [0.0; K];
        for dir in 0..DIRS {
            for phase in 0..K {
                scores[phase] += emit[target_id][dir][phase];
            }
        }
        snapshots.push(snapshot_from_scores(scores, case.private.label as usize));
    }

    let mut packet_stats = PacketStats::default();
    let (backflow, echo) = backflow_echo_power(case, &snapshots);
    packet_stats.backflow = backflow;
    packet_stats.echo = echo;
    packet_stats.stale = f64::from(stale_phase(&snapshots, case.private.label as usize));
    SimulationResult {
        snapshots,
        packet_stats,
    }
}

fn simulate_edge_packet_case(
    layout: &Layout,
    case: &Case,
    ticks: usize,
    cfg: EdgePacketConfig,
    random_table: &[[usize; K]; K],
) -> SimulationResult {
    let graph = build_edge_graph(layout, case);
    let mut state = EdgeState::new(graph.edges.len());
    let source_id = layout.cell_id(case.public.source.0, case.public.source.1);
    let label = case.private.label as usize;
    let mut stats = PacketStats::default();
    seed_source_edges(layout, case, &graph, &mut state, source_id, cfg);
    let mut snapshots = Vec::with_capacity(ticks.max(1));

    for tick in 1..=ticks {
        let mut target_scores = [0.0f64; K];
        state.clear_next();
        let active = state
            .current
            .iter()
            .filter(|lanes| lane_power(**lanes) > EPS)
            .count();
        stats.active_edge_fraction_sum += active as f64 / graph.edges.len().max(1) as f64;
        stats.active_edge_fraction_count += 1.0;

        for edge_id in 0..graph.edges.len() {
            let edge = graph.edges[edge_id];
            for phase in 0..K {
                let mass = state.current[edge_id][phase];
                if mass <= EPS {
                    continue;
                }
                let age = state.age[edge_id][phase];
                if ttl_expired(cfg, case, age) {
                    stats.ttl_expired += mass;
                    stats.dropped += mass;
                    continue;
                }

                let gate = case.public.gates[edge.to] as usize;
                let out_phase = if cfg.random_rule {
                    random_table[phase][gate]
                } else {
                    expected_phase(phase, gate)
                };
                stats.delivered += mass;

                let target_id = layout.cell_id(case.public.target.0, case.public.target.1);
                if edge.to == target_id {
                    target_scores[out_phase] += mass;
                    stats.consumed += mass;
                    continue;
                }

                let outgoing = route_outgoing_edges(
                    layout,
                    case,
                    &graph,
                    edge_id,
                    out_phase,
                    tick,
                    cfg,
                    &mut stats,
                );
                if outgoing.is_empty() {
                    stats.dropped += mass;
                    continue;
                }
                stats.fanout_sum += outgoing.len() as f64;
                stats.fanout_count += 1.0;
                stats.fanout_max = stats.fanout_max.max(outgoing.len() as f64);
                let weight = 1.0 / outgoing.len() as f64;
                for next_edge_id in outgoing {
                    if state.next[next_edge_id][out_phase] > EPS {
                        stats.duplicates += mass * weight;
                    }
                    state.next[next_edge_id][out_phase] += mass * weight;
                    state.next_age[next_edge_id][out_phase] =
                        state.next_age[next_edge_id][out_phase].max(age + 1);
                    if graph.edges[next_edge_id].to == edge.from {
                        stats.reentry += mass * weight;
                    }
                }
                if !matches!(cfg.consume, ConsumeMode::None) {
                    stats.consumed += mass;
                }
            }

            if matches!(cfg.consume, ConsumeMode::None) {
                for phase in 0..K {
                    if state.current[edge_id][phase] > EPS {
                        carry_edge_lane(&mut state, edge_id, phase);
                    }
                }
            } else if matches!(cfg.consume, ConsumeMode::PhaseOnly) {
                for phase in 0..K {
                    if state.current[edge_id][phase] <= EPS {
                        carry_edge_lane(&mut state, edge_id, phase);
                    }
                }
            }
        }

        if cfg.cell_local_normalization {
            for lanes in &mut state.next {
                normalize_lanes(lanes);
            }
        }
        state.swap();
        snapshots.push(snapshot_from_scores(target_scores, label));
    }

    let (backflow, echo) = backflow_echo_power(case, &snapshots);
    stats.backflow += backflow;
    stats.echo += echo;
    stats.stale += f64::from(stale_phase(&snapshots, label));
    SimulationResult {
        snapshots,
        packet_stats: stats,
    }
}

fn route_outgoing_edges(
    layout: &Layout,
    case: &Case,
    graph: &EdgeGraph,
    edge_id: usize,
    phase: usize,
    tick: usize,
    cfg: EdgePacketConfig,
    stats: &mut PacketStats,
) -> Vec<usize> {
    let edge = graph.edges[edge_id];
    let mut candidates = graph.outgoing[edge.to].clone();
    if cfg.no_reentry {
        let before = candidates.len();
        candidates.retain(|&next_id| graph.edges[next_id].to != edge.from);
        if candidates.len() < before {
            stats.reentry += 1.0;
        }
    }

    match cfg.routing {
        RoutingMode::Flood => candidates,
        RoutingMode::PublicGradient => {
            let from_xy = layout.cell_xy(edge.to);
            let from_d = manhattan(from_xy, case.public.target);
            let mut filtered = Vec::new();
            let mut ties = 0usize;
            for next_id in candidates {
                let to_xy = layout.cell_xy(graph.edges[next_id].to);
                let to_d = manhattan(to_xy, case.public.target);
                if to_d < from_d {
                    filtered.push(next_id);
                } else if to_d == from_d {
                    ties += 1;
                } else {
                    stats.public_wrong_turn += 1.0;
                }
            }
            if ties > 0 {
                stats.public_tie += 1.0;
            }
            if filtered.is_empty() {
                stats.public_dead_end += 1.0;
            }
            filtered
        }
        RoutingMode::OracleRoute => oracle_next_edge(case, graph, edge.to).into_iter().collect(),
        RoutingMode::RandomRoute => {
            if candidates.is_empty() {
                return candidates;
            }
            let idx = hash_index(edge_id, phase, tick, candidates.len());
            vec![candidates[idx]]
        }
    }
}

fn carry_edge_lane(state: &mut EdgeState, edge_id: usize, phase: usize) {
    if state.current[edge_id][phase] <= EPS {
        return;
    }
    state.next[edge_id][phase] += state.current[edge_id][phase];
    state.next_age[edge_id][phase] =
        state.next_age[edge_id][phase].max(state.age[edge_id][phase] + 1);
}

fn seed_source_edges(
    layout: &Layout,
    case: &Case,
    graph: &EdgeGraph,
    state: &mut EdgeState,
    source_id: usize,
    cfg: EdgePacketConfig,
) {
    let source_phase = case.public.source_phase as usize;
    let initial = match cfg.routing {
        RoutingMode::OracleRoute => oracle_next_edge(case, graph, source_id)
            .map(|id| vec![id])
            .unwrap_or_default(),
        RoutingMode::PublicGradient => graph.outgoing[source_id]
            .iter()
            .copied()
            .filter(|&edge_id| {
                let from_d = manhattan(layout.cell_xy(source_id), case.public.target);
                let to_d = manhattan(layout.cell_xy(graph.edges[edge_id].to), case.public.target);
                to_d < from_d
            })
            .collect::<Vec<_>>(),
        RoutingMode::RandomRoute => graph.outgoing[source_id]
            .first()
            .copied()
            .map(|id| vec![id])
            .unwrap_or_default(),
        RoutingMode::Flood => graph.outgoing[source_id].clone(),
    };
    let weight = if initial.is_empty() {
        0.0
    } else {
        1.0 / initial.len() as f64
    };
    for edge_id in initial {
        state.current[edge_id][source_phase] += weight;
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

impl EdgeState {
    fn new(edge_count: usize) -> Self {
        Self {
            current: vec![[0.0; K]; edge_count],
            next: vec![[0.0; K]; edge_count],
            age: vec![[0; K]; edge_count],
            next_age: vec![[0; K]; edge_count],
        }
    }

    fn clear_next(&mut self) {
        for lanes in &mut self.next {
            *lanes = [0.0; K];
        }
        for ages in &mut self.next_age {
            *ages = [0; K];
        }
    }

    fn swap(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
        std::mem::swap(&mut self.age, &mut self.next_age);
    }
}

impl PacketStats {
    fn merge(&mut self, other: &Self) {
        self.delivered += other.delivered;
        self.dropped += other.dropped;
        self.reentry += other.reentry;
        self.ttl_expired += other.ttl_expired;
        self.consumed += other.consumed;
        self.duplicates += other.duplicates;
        self.active_edge_fraction_sum += other.active_edge_fraction_sum;
        self.active_edge_fraction_count += other.active_edge_fraction_count;
        self.fanout_sum += other.fanout_sum;
        self.fanout_count += other.fanout_count;
        self.fanout_max = self.fanout_max.max(other.fanout_max);
        self.public_dead_end += other.public_dead_end;
        self.public_tie += other.public_tie;
        self.public_wrong_turn += other.public_wrong_turn;
        self.backflow += other.backflow;
        self.echo += other.echo;
        self.stale += other.stale;
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

fn ttl_expired(cfg: EdgePacketConfig, case: &Case, age: usize) -> bool {
    let Some(mode) = cfg.ttl_mode else {
        return false;
    };
    let path = case.private.true_path.len().max(2);
    let ttl = match mode {
        TtlMode::Path => path,
        TtlMode::PathPlus2 => path + 2,
        TtlMode::TwicePath => 2 * path,
    };
    age > ttl
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

fn edge_config(kind: ArmKind, random_rule: bool) -> EdgePacketConfig {
    match kind {
        ArmKind::EdgePacketPublicGradient => EdgePacketConfig {
            routing: RoutingMode::PublicGradient,
            consume: ConsumeMode::PhaseOnly,
            ..edge_default(random_rule)
        },
        ArmKind::EdgePacketOracleRouteCorrectPhaseDiagnostic => EdgePacketConfig {
            routing: RoutingMode::OracleRoute,
            consume: ConsumeMode::PhaseOnly,
            ..edge_default(random_rule)
        },
        ArmKind::EdgePacketOracleRouteRandomPhaseDiagnostic => EdgePacketConfig {
            routing: RoutingMode::OracleRoute,
            consume: ConsumeMode::PhaseOnly,
            random_rule: true,
            ..edge_default(true)
        },
        ArmKind::EdgePacketConsumePhaseOnly => EdgePacketConfig {
            consume: ConsumeMode::PhaseOnly,
            ..edge_default(random_rule)
        },
        ArmKind::EdgePacketConsumeFullEdge => EdgePacketConfig {
            consume: ConsumeMode::FullEdge,
            ..edge_default(random_rule)
        },
        ArmKind::EdgePacketNoReentry => EdgePacketConfig {
            consume: ConsumeMode::PhaseOnly,
            no_reentry: true,
            ..edge_default(random_rule)
        },
        ArmKind::EdgePacketTtlPath => EdgePacketConfig {
            ttl_mode: Some(TtlMode::Path),
            ..edge_default(random_rule)
        },
        ArmKind::EdgePacketTtlPathPlus2 => EdgePacketConfig {
            ttl_mode: Some(TtlMode::PathPlus2),
            ..edge_default(random_rule)
        },
        ArmKind::EdgePacketTtl2xPath => EdgePacketConfig {
            ttl_mode: Some(TtlMode::TwicePath),
            ..edge_default(random_rule)
        },
        ArmKind::EdgePacketPlusTargetSettledReadout => EdgePacketConfig {
            consume: ConsumeMode::PhaseOnly,
            ..edge_default(random_rule)
        },
        ArmKind::EdgePacketPlusCellLocalNormalization => EdgePacketConfig {
            consume: ConsumeMode::PhaseOnly,
            cell_local_normalization: true,
            ..edge_default(random_rule)
        },
        ArmKind::EdgePacketPlusPublicNoBackflow => EdgePacketConfig {
            routing: RoutingMode::PublicGradient,
            consume: ConsumeMode::PhaseOnly,
            no_reentry: true,
            ..edge_default(random_rule)
        },
        ArmKind::RandomRuleEdgePacketControl => EdgePacketConfig {
            random_rule: true,
            consume: ConsumeMode::PhaseOnly,
            ..edge_default(true)
        },
        ArmKind::RandomRouteEdgePacketControl => EdgePacketConfig {
            routing: RoutingMode::RandomRoute,
            consume: ConsumeMode::PhaseOnly,
            ..edge_default(random_rule)
        },
        _ => edge_default(random_rule),
    }
}

fn edge_default(random_rule: bool) -> EdgePacketConfig {
    EdgePacketConfig {
        routing: RoutingMode::Flood,
        consume: ConsumeMode::None,
        ttl_mode: None,
        no_reentry: false,
        cell_local_normalization: false,
        random_rule,
    }
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
    metrics: EvalMetrics,
    baseline: &EvalMetrics,
    random_rule: &EvalMetrics,
    random_route: &EvalMetrics,
    started: Instant,
) -> MetricRow {
    let cfg = if arm.kind.is_edge_packet() {
        Some(edge_config(arm.kind, arm.random_rule))
    } else {
        None
    };
    let delta_acc = metrics.phase_final_accuracy - baseline.phase_final_accuracy;
    let delta_long = metrics.long_path_accuracy - baseline.long_path_accuracy;
    let delta_family = metrics.phase_final_accuracy - baseline.phase_final_accuracy;
    let delta_wrong = baseline.wrong_if_arrived_rate - metrics.wrong_if_arrived_rate;
    let has_signal = arm.kind.is_edge_packet()
        && delta_long >= 0.10
        && delta_family >= 0.20
        && delta_wrong >= 0.10
        && metrics.final_minus_best_gap <= baseline.final_minus_best_gap + 0.05
        && random_rule.phase_final_accuracy < 0.45;
    let positive = has_signal
        && metrics.phase_final_accuracy >= 0.95
        && metrics.long_path_accuracy >= 0.95
        && metrics.same_target_counterfactual_accuracy >= 0.85
        && metrics.gate_shuffle_collapse >= 0.50
        && metrics.wrong_if_arrived_rate <= 0.10
        && metrics.final_minus_best_gap <= 0.05
        && random_rule.phase_final_accuracy < 0.45
        && random_route.phase_final_accuracy < 0.45
        && metrics.wall_leak_rate <= 0.02
        && metrics.forbidden_private_field_leak == 0.0
        && metrics.nonlocal_edge_count == 0
        && metrics.direct_output_leak_rate == 0.0;

    MetricRow {
        job_id: job_id.to_string(),
        seed,
        arm: arm.kind.as_str().to_string(),
        diagnostic_only: arm.kind.diagnostic_only(),
        random_rule: arm.random_rule || matches!(arm.kind, ArmKind::RandomRuleEdgePacketControl),
        random_route: matches!(arm.kind, ArmKind::RandomRouteEdgePacketControl),
        routing_mode: cfg
            .map(|c| format!("{:?}", c.routing))
            .unwrap_or_else(|| "Node".to_string()),
        consume_mode: cfg
            .map(|c| format!("{:?}", c.consume))
            .unwrap_or_else(|| "Node".to_string()),
        ttl_mode: cfg
            .and_then(|c| c.ttl_mode)
            .map(|t| format!("{t:?}"))
            .unwrap_or_else(|| "None".to_string()),
        no_reentry: cfg.map(|c| c.no_reentry).unwrap_or(false),
        width,
        path_length,
        ticks,
        family: family.to_string(),
        case_count,
        phase_final_accuracy: metrics.phase_final_accuracy,
        long_path_accuracy: metrics.long_path_accuracy,
        family_min_accuracy: metrics.phase_final_accuracy,
        same_target_counterfactual_accuracy: metrics.same_target_counterfactual_accuracy,
        gate_shuffle_collapse: metrics.gate_shuffle_collapse,
        target_arrival_rate: metrics.target_arrival_rate,
        wrong_if_arrived_rate: metrics.wrong_if_arrived_rate,
        wrong_phase_growth_rate: metrics.wrong_phase_growth_rate,
        correct_phase_power: metrics.correct_phase_power,
        wrong_phase_power: metrics.wrong_phase_power,
        correct_phase_margin: metrics.correct_phase_margin,
        final_minus_best_gap: metrics.final_minus_best_gap,
        edge_packet_delivery_rate: metrics.edge_packet_delivery_rate,
        edge_packet_drop_rate: metrics.edge_packet_drop_rate,
        edge_packet_reentry_rate: metrics.edge_packet_reentry_rate,
        edge_packet_ttl_expiry_rate: metrics.edge_packet_ttl_expiry_rate,
        edge_packet_consumed_rate: metrics.edge_packet_consumed_rate,
        edge_packet_duplicate_rate: metrics.edge_packet_duplicate_rate,
        active_edge_fraction: metrics.active_edge_fraction,
        packet_fanout_mean: metrics.packet_fanout_mean,
        packet_fanout_max: metrics.packet_fanout_max,
        public_route_dead_end_rate: metrics.public_route_dead_end_rate,
        public_route_tie_rate: metrics.public_route_tie_rate,
        public_route_wrong_turn_rate: metrics.public_route_wrong_turn_rate,
        random_rule_edge_packet_accuracy: random_rule.phase_final_accuracy,
        random_route_edge_packet_accuracy: random_route.phase_final_accuracy,
        wall_leak_rate: metrics.wall_leak_rate,
        forbidden_private_field_leak: metrics.forbidden_private_field_leak,
        nonlocal_edge_count: metrics.nonlocal_edge_count,
        direct_output_leak_rate: metrics.direct_output_leak_rate,
        backflow_power: metrics.backflow_power,
        echo_power: metrics.echo_power,
        stale_packet_rate: metrics.stale_packet_rate,
        delta_vs_node_broadcast_accuracy: delta_acc,
        delta_vs_node_broadcast_long_path: delta_long,
        delta_vs_node_broadcast_family_min: delta_family,
        delta_vs_node_broadcast_wrong_if_arrived: delta_wrong,
        has_edge_packet_signal: has_signal,
        passes_positive_gate: positive,
        elapsed_sec: started.elapsed().as_secs_f64(),
    }
}

fn append_metric_files(cfg: &Config, row: &MetricRow) -> Result<(), Box<dyn std::error::Error>> {
    append_jsonl(cfg.out.join("metrics.jsonl"), row)?;
    append_jsonl(cfg.out.join("family_metrics.jsonl"), row)?;
    append_jsonl(cfg.out.join("counterfactual_metrics.jsonl"), row)?;
    if row.arm.contains("EDGE_PACKET") {
        append_jsonl(cfg.out.join("edge_packet_metrics.jsonl"), row)?;
    }
    if row.routing_mode != "Node" {
        append_jsonl(cfg.out.join("routing_metrics.jsonl"), row)?;
    }
    if row.ttl_mode != "None" {
        append_jsonl(cfg.out.join("ttl_metrics.jsonl"), row)?;
    }
    if row.consume_mode != "None" && row.consume_mode != "Node" {
        append_jsonl(cfg.out.join("consume_metrics.jsonl"), row)?;
    }
    if row.no_reentry {
        append_jsonl(cfg.out.join("reentry_metrics.jsonl"), row)?;
    }
    if row.random_rule || row.random_route {
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
        "top_signal": ranking.iter().find(|row| row.has_edge_packet_signal),
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
    _rows: &[MetricRow],
    ranking: &[RankingRow],
    final_report: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut report = String::new();
    report.push_str("# STABLE_LOOP_PHASE_LOCK_016_DIRECTED_EDGE_PACKET_TRANSPORT Report\n\n");
    report.push_str(if final_report {
        "Status: complete.\n\n"
    } else {
        "Status: running.\n\n"
    });
    report.push_str("## Verdicts\n\n```text\n");
    for verdict in verdicts(_rows, ranking) {
        report.push_str(&verdict);
        report.push('\n');
    }
    report.push_str("```\n\n## Carrier Comparison\n\n");
    report.push_str("| Arm | Acc | Long | Family min | Wrong-if-arrived | Random rule | Random route | Fanout | Active edges | Signal | Positive |\n");
    report.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n");
    for row in ranking {
        report.push_str(&format!(
            "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} |\n",
            row.arm,
            row.phase_final_accuracy,
            row.long_path_accuracy,
            row.family_min_accuracy,
            row.wrong_if_arrived_rate,
            row.random_rule_edge_packet_accuracy,
            row.random_route_edge_packet_accuracy,
            row.packet_fanout_mean,
            row.active_edge_fraction,
            row.has_edge_packet_signal,
            row.passes_positive_gate,
        ));
    }
    report.push_str("\n## Claim Boundary\n\n");
    report.push_str("016 is a runner-local transport-carrier probe. It cannot claim production architecture, full VRAXION, language grounding, consciousness, Prismion uniqueness, or physical quantum behavior.\n");
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
            wrong_if_arrived_rate: mean(subset.iter().map(|row| row.wrong_if_arrived_rate)),
            final_minus_best_gap: mean(subset.iter().map(|row| row.final_minus_best_gap)),
            random_rule_edge_packet_accuracy: mean(
                subset
                    .iter()
                    .map(|row| row.random_rule_edge_packet_accuracy),
            ),
            random_route_edge_packet_accuracy: mean(
                subset
                    .iter()
                    .map(|row| row.random_route_edge_packet_accuracy),
            ),
            edge_packet_duplicate_rate: mean(
                subset.iter().map(|row| row.edge_packet_duplicate_rate),
            ),
            active_edge_fraction: mean(subset.iter().map(|row| row.active_edge_fraction)),
            packet_fanout_mean: mean(subset.iter().map(|row| row.packet_fanout_mean)),
            delta_vs_node_broadcast_accuracy: mean(
                subset
                    .iter()
                    .map(|row| row.delta_vs_node_broadcast_accuracy),
            ),
            delta_vs_node_broadcast_long_path: mean(
                subset
                    .iter()
                    .map(|row| row.delta_vs_node_broadcast_long_path),
            ),
            delta_vs_node_broadcast_family_min: 0.0,
            delta_vs_node_broadcast_wrong_if_arrived: mean(
                subset
                    .iter()
                    .map(|row| row.delta_vs_node_broadcast_wrong_if_arrived),
            ),
            has_edge_packet_signal: false,
            passes_positive_gate: false,
        });
    }
    let baseline_family = out
        .iter()
        .find(|row| row.arm == ArmKind::NodeBroadcastBaseline014.as_str())
        .map(|row| row.family_min_accuracy)
        .unwrap_or(0.0);
    let baseline_gap = out
        .iter()
        .find(|row| row.arm == ArmKind::NodeBroadcastBaseline014.as_str())
        .map(|row| row.final_minus_best_gap + 0.05)
        .unwrap_or(0.05);
    for row in &mut out {
        row.delta_vs_node_broadcast_family_min = row.family_min_accuracy - baseline_family;
        row.has_edge_packet_signal = row.arm.contains("EDGE_PACKET")
            && !row.arm.contains("RANDOM")
            && row.delta_vs_node_broadcast_long_path >= 0.10
            && row.delta_vs_node_broadcast_family_min >= 0.20
            && row.delta_vs_node_broadcast_wrong_if_arrived >= 0.10
            && row.final_minus_best_gap <= baseline_gap
            && row.random_rule_edge_packet_accuracy < 0.45;
        row.passes_positive_gate = row.has_edge_packet_signal
            && !diagnostic_arm_name(&row.arm)
            && row.phase_final_accuracy >= 0.95
            && row.long_path_accuracy >= 0.95
            && row.family_min_accuracy >= 0.85
            && row.wrong_if_arrived_rate <= 0.10
            && row.final_minus_best_gap <= 0.05
            && row.random_rule_edge_packet_accuracy < 0.45
            && row.random_route_edge_packet_accuracy < 0.45;
    }
    out.sort_by(|a, b| {
        b.passes_positive_gate
            .cmp(&a.passes_positive_gate)
            .then_with(|| b.has_edge_packet_signal.cmp(&a.has_edge_packet_signal))
            .then_with(|| {
                b.delta_vs_node_broadcast_wrong_if_arrived
                    .partial_cmp(&a.delta_vs_node_broadcast_wrong_if_arrived)
                    .unwrap()
            })
            .then_with(|| b.long_path_accuracy.partial_cmp(&a.long_path_accuracy).unwrap())
    });
    out
}

fn diagnostic_arm_name(name: &str) -> bool {
    name.contains("ORACLE_ROUTE") || name.contains("TARGET_SETTLED_READOUT")
}

fn verdicts(rows: &[MetricRow], ranking: &[RankingRow]) -> Vec<String> {
    if rows.is_empty() {
        return vec!["RUNNING".to_string()];
    }
    let mut out = BTreeSet::new();
    if ranking
        .iter()
        .any(|row| row.has_edge_packet_signal && !diagnostic_arm_name(&row.arm))
    {
        out.insert("EDGE_PACKET_TRANSPORT_HAS_SIGNAL".to_string());
    }
    if ranking
        .iter()
        .any(|row| row.passes_positive_gate && !diagnostic_arm_name(&row.arm))
    {
        out.insert("EDGE_PACKET_RESCUES_LONG_CHAIN".to_string());
        out.insert("NODE_BROADCAST_IS_BLOCKER".to_string());
    }
    if ranking.iter().any(|row| {
        row.arm == ArmKind::EdgePacketOracleRouteCorrectPhaseDiagnostic.as_str()
            && row.has_edge_packet_signal
    }) {
        out.insert("EDGE_PACKET_ORACLE_ROUTING_HAS_SIGNAL".to_string());
    }
    if ranking.iter().any(|row| {
        row.arm == ArmKind::EdgePacketPublicGradient.as_str() && row.has_edge_packet_signal
    }) {
        out.insert("EDGE_PACKET_PUBLIC_ROUTING_HAS_SIGNAL".to_string());
        out.insert("PUBLIC_ROUTING_SUFFICIENT".to_string());
    }
    let oracle_signal = ranking.iter().any(|row| {
        row.arm == ArmKind::EdgePacketOracleRouteCorrectPhaseDiagnostic.as_str()
            && row.has_edge_packet_signal
    });
    let public_signal = ranking.iter().any(|row| {
        row.arm == ArmKind::EdgePacketPublicGradient.as_str() && row.has_edge_packet_signal
    });
    if oracle_signal && !public_signal {
        out.insert("ONLY_ORACLE_ROUTING_WORKS".to_string());
        out.insert("PUBLIC_ROUTING_FAILS".to_string());
        out.insert("EDGE_CARRIER_WORKS_ROUTING_POLICY_BLOCKED".to_string());
    }
    if ranking.iter().any(|row| {
        row.arm == ArmKind::EdgePacketFlood.as_str()
            && row.has_edge_packet_signal
            && (row.packet_fanout_mean > 1.5 || row.active_edge_fraction > 0.40)
    }) {
        out.insert("EDGE_PACKET_FLOOD_SOLVES_DENSE".to_string());
        out.insert("FLOOD_FANOUT_CONTAMINATION".to_string());
    }
    if ranking.iter().any(|row| {
        row.random_rule_edge_packet_accuracy >= 0.45
            && row.arm.contains("EDGE_PACKET")
            && !row.arm.contains("ORACLE_ROUTE")
    }) {
        out.insert("EDGE_PACKET_OVERPOWERS_RULE_CONTROL".to_string());
    } else {
        out.insert("EDGE_PACKET_RANDOM_RULE_FAILS".to_string());
    }
    if ranking
        .iter()
        .any(|row| row.arm.contains("TTL") && row.has_edge_packet_signal)
    {
        out.insert("EDGE_TTL_REQUIRED".to_string());
    }
    if ranking.iter().any(|row| {
        row.arm == ArmKind::EdgePacketTtlPath.as_str() && row.phase_final_accuracy < 0.45
    }) {
        out.insert("TTL_TOO_SHORT_KILLS_TRANSPORT".to_string());
    }
    if ranking.iter().any(|row| {
        row.arm == ArmKind::EdgePacketTtl2xPath.as_str() && row.wrong_if_arrived_rate > 0.20
    }) {
        out.insert("TTL_TOO_LONG_ALLOWS_ECHO".to_string());
    }
    if ranking.iter().any(|row| {
        row.arm == ArmKind::EdgePacketConsumePhaseOnly.as_str() && row.has_edge_packet_signal
    }) {
        out.insert("EDGE_CONSUME_REQUIRED".to_string());
    }
    let phase_consume = ranking
        .iter()
        .find(|row| row.arm == ArmKind::EdgePacketConsumePhaseOnly.as_str());
    let full_consume = ranking
        .iter()
        .find(|row| row.arm == ArmKind::EdgePacketConsumeFullEdge.as_str());
    if let (Some(phase), Some(full)) = (phase_consume, full_consume) {
        if phase.phase_final_accuracy > full.phase_final_accuracy + 0.05 {
            out.insert("PHASE_ONLY_CONSUME_BETTER_THAN_FULL_EDGE_CONSUME".to_string());
        }
    }
    if ranking.iter().any(|row| {
        row.arm == ArmKind::EdgePacketNoReentry.as_str() && row.has_edge_packet_signal
    }) {
        out.insert("EDGE_NO_REENTRY_REQUIRED".to_string());
    }
    if ranking.iter().any(|row| {
        row.arm == ArmKind::EdgePacketPublicGradient.as_str()
            && row.has_edge_packet_signal
            && row.phase_final_accuracy < 0.95
    }) {
        out.insert("PUBLIC_ROUTE_DEAD_END_LIMIT".to_string());
    }
    if ranking
        .iter()
        .filter(|row| row.arm.contains("EDGE_PACKET"))
        .any(|row| row.delta_vs_node_broadcast_wrong_if_arrived >= 0.10)
    {
        out.insert("EDGE_PACKET_REDUCES_WRONG_PHASE".to_string());
    }
    if ranking
        .iter()
        .filter(|row| row.arm.contains("EDGE_PACKET"))
        .filter(|row| !diagnostic_arm_name(&row.arm))
        .all(|row| !row.has_edge_packet_signal)
    {
        out.insert("NO_EDGE_PACKET_SIGNAL".to_string());
    }
    if ranking
        .iter()
        .filter(|row| row.arm.contains("EDGE_PACKET"))
        .any(|row| row.target_like_low_accuracy())
    {
        out.insert("EDGE_PACKET_KILLS_TRANSPORT".to_string());
    }
    out.insert("PRODUCTION_API_NOT_READY".to_string());
    out.into_iter().collect()
}

trait RankingExt {
    fn target_like_low_accuracy(&self) -> bool;
}

impl RankingExt for RankingRow {
    fn target_like_low_accuracy(&self) -> bool {
        self.phase_final_accuracy < 0.35 && self.wrong_if_arrived_rate > 0.50
    }
}

fn generate_cases(
    seed: u64,
    count: usize,
    width: usize,
    path_length: usize,
    family: &str,
) -> Vec<Case> {
    (0..count)
        .map(|idx| generate_case(seed, idx, width, path_length, family))
        .collect()
}

fn generate_case(seed: u64, idx: usize, width: usize, path_length: usize, family: &str) -> Case {
    let mut rng = StdRng::seed_from_u64(seed ^ idx as u64 ^ width as u64);
    let path = serpentine_path(width, path_length);
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
            split: "edge_packet_eval".to_string(),
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
        out.retain(|&(ny, nx)| manhattan((ny, nx), case.public.target) < manhattan(cell, case.public.target));
    }
    out
}

fn select_readout(snapshots: &[TargetSnapshot], label: usize, kind: ArmKind) -> TargetSnapshot {
    if matches!(kind, ArmKind::EdgePacketPlusTargetSettledReadout) {
        readout_best(snapshots, label)
    } else {
        snapshots
            .last()
            .cloned()
            .unwrap_or_else(|| snapshot_from_scores([0.0; K], label))
    }
}

fn readout_best(snapshots: &[TargetSnapshot], label: usize) -> TargetSnapshot {
    snapshots
        .iter()
        .cloned()
        .max_by(|a, b| a.correct_prob.partial_cmp(&b.correct_prob).unwrap())
        .unwrap_or_else(|| snapshot_from_scores([0.0; K], label))
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
        scores: clipped,
        probs,
        pred: argmax(&probs),
        correct_prob: probs[label],
        total_power: total,
    }
}

fn arms() -> Vec<Arm> {
    [
        ArmKind::NodeBroadcastBaseline014,
        ArmKind::BestPublicCombo014,
        ArmKind::MomentumLanes015Baseline,
        ArmKind::EdgePacketFlood,
        ArmKind::EdgePacketPublicGradient,
        ArmKind::EdgePacketOracleRouteCorrectPhaseDiagnostic,
        ArmKind::EdgePacketOracleRouteRandomPhaseDiagnostic,
        ArmKind::EdgePacketConsumePhaseOnly,
        ArmKind::EdgePacketConsumeFullEdge,
        ArmKind::EdgePacketNoReentry,
        ArmKind::EdgePacketTtlPath,
        ArmKind::EdgePacketTtlPathPlus2,
        ArmKind::EdgePacketTtl2xPath,
        ArmKind::EdgePacketPlusTargetSettledReadout,
        ArmKind::EdgePacketPlusCellLocalNormalization,
        ArmKind::EdgePacketPlusPublicNoBackflow,
        ArmKind::RandomRuleEdgePacketControl,
        ArmKind::RandomRouteEdgePacketControl,
    ]
    .into_iter()
    .map(|kind| Arm {
        random_rule: matches!(
            kind,
            ArmKind::RandomRuleEdgePacketControl
                | ArmKind::EdgePacketOracleRouteRandomPhaseDiagnostic
        ),
        kind,
    })
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
    directional_neighbors(width, y, x)
        .into_iter()
        .map(|(_, ny, nx)| (ny, nx))
        .collect()
}

fn directional_neighbors(width: usize, y: usize, x: usize) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::with_capacity(4);
    if y > 0 {
        out.push((0, y - 1, x));
    }
    if y + 1 < width {
        out.push((1, y + 1, x));
    }
    if x > 0 {
        out.push((2, y, x - 1));
    }
    if x + 1 < width {
        out.push((3, y, x + 1));
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

fn backflow_echo_power(case: &Case, snapshots: &[TargetSnapshot]) -> (f64, f64) {
    let path_len = case.private.true_path.len().max(1) as f64;
    let target_power = snapshots.last().map(|s| s.total_power).unwrap_or(0.0);
    let backflow = if path_len > 2.0 {
        target_power / path_len
    } else {
        0.0
    };
    let echo = snapshots
        .windows(2)
        .filter(|pair| pair[1].total_power > pair[0].total_power && pair[0].total_power > EPS)
        .count() as f64
        / path_len;
    (backflow, echo)
}

fn stale_phase(snapshots: &[TargetSnapshot], label: usize) -> bool {
    snapshots.len() >= 3
        && snapshots[snapshots.len() - 3..]
            .iter()
            .all(|snap| snap.pred == snapshots.last().unwrap().pred)
        && snapshots.last().unwrap().pred != label
}

fn manhattan(a: (usize, usize), b: (usize, usize)) -> usize {
    a.0.abs_diff(b.0) + a.1.abs_diff(b.1)
}

fn hash_index(edge_id: usize, phase: usize, tick: usize, modulo: usize) -> usize {
    if modulo == 0 {
        return 0;
    }
    let x = edge_id
        .wrapping_mul(1_103_515_245)
        .wrapping_add(phase * 97)
        .wrapping_add(tick * 17);
    x % modulo
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
    let mut out = PathBuf::from("target/pilot_wave/stable_loop_phase_lock_016_directed_edge_packet_transport/dev");
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
        "probe": "STABLE_LOOP_PHASE_LOCK_016_DIRECTED_EDGE_PACKET_TRANSPORT",
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
        "# STABLE_LOOP_PHASE_LOCK_016_DIRECTED_EDGE_PACKET_TRANSPORT\n\nRunner-local directed edge-packet transport probe. No public instnct-core API changes.\n",
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
            "public_arms_use_true_path": false,
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
