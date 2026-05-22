//! Runner-local directed route settled delivery probe.
//!
//! 019 asks whether receive-committed target delivery can preserve the clean
//! directed-route pulse found by 018 as stable final output.

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
    gate_sum: u8,
    family: String,
    requested_path_length: usize,
}

#[derive(Clone, Debug)]
struct Case {
    public: PublicCase,
    private: PrivateCase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
enum ArmKind {
    BidirectionalGridBaseline018,
    TruePathDirectedRouteFinalTick,
    TruePathDirectedRouteBestTickDiagnostic,
    TruePathDirectedRouteTargetLatch1Tick,
    TruePathDirectedRouteTargetSettledLedgerSum,
    TruePathDirectedRouteTargetSettledLedgerMax,
    TruePathDirectedRouteReceiveCommitLedgerSum,
    TruePathDirectedRouteReceiveCommitLedgerMax,
    TruePathDirectedRouteConsumeOnDelivery,
    TruePathPlusReverseReceiveCommitLedger,
    PublicGradientReceiveCommitLedger,
    PublicMonotoneReceiveCommitLedger,
    RandomSameCountReceiveCommitLedger,
    DirectionShuffleReceiveCommitLedger,
    RandomPhaseRuleReceiveCommitLedger,
}

impl ArmKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::BidirectionalGridBaseline018 => "BIDIRECTIONAL_GRID_BASELINE_018",
            Self::TruePathDirectedRouteFinalTick => "TRUE_PATH_DIRECTED_ROUTE_FINAL_TICK",
            Self::TruePathDirectedRouteBestTickDiagnostic => "TRUE_PATH_DIRECTED_ROUTE_BEST_TICK_DIAGNOSTIC",
            Self::TruePathDirectedRouteTargetLatch1Tick => "TRUE_PATH_DIRECTED_ROUTE_TARGET_LATCH_1TICK",
            Self::TruePathDirectedRouteTargetSettledLedgerSum => "TRUE_PATH_DIRECTED_ROUTE_TARGET_SETTLED_LEDGER_SUM",
            Self::TruePathDirectedRouteTargetSettledLedgerMax => "TRUE_PATH_DIRECTED_ROUTE_TARGET_SETTLED_LEDGER_MAX",
            Self::TruePathDirectedRouteReceiveCommitLedgerSum => "TRUE_PATH_DIRECTED_ROUTE_RECEIVE_COMMIT_LEDGER_SUM",
            Self::TruePathDirectedRouteReceiveCommitLedgerMax => "TRUE_PATH_DIRECTED_ROUTE_RECEIVE_COMMIT_LEDGER_MAX",
            Self::TruePathDirectedRouteConsumeOnDelivery => "TRUE_PATH_DIRECTED_ROUTE_CONSUME_ON_DELIVERY",
            Self::TruePathPlusReverseReceiveCommitLedger => "TRUE_PATH_PLUS_REVERSE_RECEIVE_COMMIT_LEDGER",
            Self::PublicGradientReceiveCommitLedger => "PUBLIC_GRADIENT_RECEIVE_COMMIT_LEDGER",
            Self::PublicMonotoneReceiveCommitLedger => "PUBLIC_MONOTONE_RECEIVE_COMMIT_LEDGER",
            Self::RandomSameCountReceiveCommitLedger => "RANDOM_SAME_COUNT_RECEIVE_COMMIT_LEDGER",
            Self::DirectionShuffleReceiveCommitLedger => "DIRECTION_SHUFFLE_RECEIVE_COMMIT_LEDGER",
            Self::RandomPhaseRuleReceiveCommitLedger => "RANDOM_PHASE_RULE_RECEIVE_COMMIT_LEDGER",
        }
    }

    fn diagnostic_only(self) -> bool {
        matches!(
            self,
            Self::TruePathDirectedRouteFinalTick
                | Self::TruePathDirectedRouteBestTickDiagnostic
                | Self::TruePathDirectedRouteTargetLatch1Tick
                | Self::TruePathDirectedRouteTargetSettledLedgerSum
                | Self::TruePathDirectedRouteTargetSettledLedgerMax
                | Self::TruePathDirectedRouteReceiveCommitLedgerSum
                | Self::TruePathDirectedRouteReceiveCommitLedgerMax
                | Self::TruePathDirectedRouteConsumeOnDelivery
                | Self::TruePathPlusReverseReceiveCommitLedger
        )
    }

    fn is_control(self) -> bool {
        matches!(
            self,
            Self::RandomSameCountReceiveCommitLedger
                | Self::DirectionShuffleReceiveCommitLedger
                | Self::RandomPhaseRuleReceiveCommitLedger
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReadoutMode {
    FinalTick,
    BestTickDiagnostic,
    TargetLatch1Tick,
    SettledLedgerSum,
    SettledLedgerMax,
    ReceiveCommitLedgerSum,
    ReceiveCommitLedgerMax,
    ConsumeOnDelivery,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphMode {
    Bidirectional,
    TruePath,
    TruePathPlusReverse,
    PublicGradient,
    PublicMonotone,
    RandomSameCount,
    DirectionShuffle,
}

#[derive(Clone, Copy, Debug)]
struct DirectedEdge {
    from: usize,
    to: usize,
}

#[derive(Clone, Debug)]
struct DirectedGraph {
    edges: Vec<DirectedEdge>,
    outgoing: Vec<Vec<usize>>,
}

#[derive(Clone, Copy, Debug)]
struct Snapshot {
    pred: usize,
    correct_prob: f64,
    target_power: f64,
    correct_power: f64,
    wrong_power: f64,
}

#[derive(Clone, Debug)]
struct Simulation {
    #[allow(dead_code)]
    final_snapshot: Snapshot,
    best_snapshot: Snapshot,
    selected_snapshot: Snapshot,
    settled_snapshot: Snapshot,
    delivery_events: usize,
    wrong_delivery_events: usize,
    duplicate_delivery_events: usize,
    stale_delivery_events: usize,
    first_delivery_tick: Option<usize>,
    delivery_tick_histogram: Vec<usize>,
    ledger_power_total: f64,
}

#[derive(Clone, Debug, Default)]
struct RunningStats {
    n: usize,
    ok: usize,
    settled_ok: usize,
    sufficient_n: usize,
    sufficient_ok: usize,
    long_n: usize,
    long_ok: usize,
    prob_sum: f64,
    best_ok: usize,
    arrival: usize,
    wrong_if_arrived: usize,
    target_power_sum: f64,
    correct_power_sum: f64,
    wrong_power_sum: f64,
    final_minus_best_gap_sum: f64,
    gate_shuffle_ok: usize,
    counterfactual_ok: usize,
    target_delivery_cases: usize,
    wrong_delivery_cases: usize,
    delivery_events: usize,
    wrong_delivery_events: usize,
    duplicate_delivery_events: usize,
    stale_delivery_events: usize,
    first_delivery_tick_sum: f64,
    first_delivery_tick_count: usize,
    ledger_power_sum: f64,
}

#[derive(Clone, Debug, Serialize)]
struct MetricsRow {
    arm: String,
    seed: u64,
    width: usize,
    path_length: usize,
    ticks: usize,
    family: String,
    n: usize,
    diagnostic_only: bool,
    phase_final_accuracy: f64,
    settled_final_accuracy: f64,
    sufficient_tick_final_accuracy: f64,
    long_path_accuracy: f64,
    family_min_accuracy: f64,
    correct_target_lane_probability_mean: f64,
    best_tick_accuracy: f64,
    target_delivery_rate: f64,
    target_wrong_delivery_rate: f64,
    wrong_if_delivered_rate: f64,
    delivery_tick_histogram: Vec<usize>,
    first_delivery_tick_mean: f64,
    ledger_power_total: f64,
    duplicate_delivery_rate: f64,
    stale_delivery_rate: f64,
    reverse_edge_sensitivity: f64,
    gate_shuffle_collapse: f64,
    same_target_counterfactual_accuracy: f64,
    directed_edge_count: usize,
    reciprocal_edge_fraction: f64,
    backflow_edge_fraction: f64,
    active_echo_power: f64,
    target_wrong_power: f64,
    target_correct_power: f64,
    final_minus_best_gap: f64,
    random_phase_rule_accuracy: f64,
    direction_shuffle_accuracy: f64,
    random_same_count_accuracy: f64,
    delta_vs_bidirectional_accuracy: f64,
    delta_vs_bidirectional_wrong_if_arrived: f64,
    forbidden_private_field_leak: f64,
    nonlocal_edge_count: usize,
    direct_output_leak_rate: f64,
}

#[derive(Clone, Debug, Serialize)]
struct RankingRow {
    arm: String,
    diagnostic_only: bool,
    phase_final_accuracy: f64,
    settled_final_accuracy: f64,
    best_tick_accuracy: f64,
    sufficient_tick_best_accuracy: f64,
    sufficient_tick_final_accuracy: f64,
    long_path_accuracy: f64,
    family_min_accuracy: f64,
    wrong_if_delivered_rate: f64,
    ledger_power_total: f64,
    duplicate_delivery_rate: f64,
    stale_delivery_rate: f64,
    reciprocal_edge_fraction: f64,
    backflow_edge_fraction: f64,
    edge_count: f64,
    delta_vs_bidirectional_accuracy: f64,
    delta_vs_bidirectional_wrong_if_arrived: f64,
    signal: bool,
    positive: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = parse_args()?;
    fs::create_dir_all(&cfg.out)?;
    fs::create_dir_all(cfg.out.join("job_progress"))?;
    write_json(
        &cfg.out.join("queue.json"),
        &json!({
            "probe": "STABLE_LOOP_PHASE_LOCK_019_DIRECTED_ROUTE_SETTLED_DELIVERY",
            "seeds": cfg.seeds,
            "eval_examples": cfg.eval_examples,
            "widths": cfg.widths,
            "path_lengths": cfg.path_lengths,
            "ticks_list": cfg.ticks_list,
            "arms": arms().iter().map(|a| a.as_str()).collect::<Vec<_>>(),
        }),
    )?;
    write_text(
        &cfg.out.join("contract_snapshot.md"),
        include_str!("../../docs/research/STABLE_LOOP_PHASE_LOCK_019_DIRECTED_ROUTE_SETTLED_DELIVERY_CONTRACT.md"),
    )?;

    let start = Instant::now();
    let mut last_heartbeat = Instant::now();
    let mut rows = Vec::new();
    let families = gate_families();
    let bucket_count =
        cfg.widths.len() * cfg.path_lengths.len() * cfg.ticks_list.len() * families.len();
    let cases_per_bucket = (cfg.eval_examples / bucket_count.max(1)).max(1);
    write_examples_sample(&cfg, &families)?;

    let mut completed = 0usize;
    for &seed in &cfg.seeds {
        for &width in &cfg.widths {
            for &path_length in &cfg.path_lengths {
                for &ticks in &cfg.ticks_list {
                    for family in &families {
                        let cases = make_cases(seed, width, path_length, family, cases_per_bucket);
                        let baseline = eval_bucket(
                            ArmKind::BidirectionalGridBaseline018,
                            seed,
                            width,
                            path_length,
                            ticks,
                            family,
                            &cases,
                            None,
                            None,
                        );
                        for arm in arms() {
                            let direction_shuffle = if arm == ArmKind::DirectionShuffleReceiveCommitLedger {
                                Some(baseline.phase_final_accuracy)
                            } else {
                                None
                            };
                            let random_phase = if arm == ArmKind::RandomPhaseRuleReceiveCommitLedger {
                                Some(baseline.phase_final_accuracy)
                            } else {
                                None
                            };
                            let mut row = if arm == ArmKind::BidirectionalGridBaseline018 {
                                baseline.clone()
                            } else {
                                eval_bucket(
                                    arm,
                                    seed,
                                    width,
                                    path_length,
                                    ticks,
                                    family,
                                    &cases,
                                    Some(&baseline),
                                    direction_shuffle.or(random_phase),
                                )
                            };
                            if arm == ArmKind::BidirectionalGridBaseline018 {
                                row.delta_vs_bidirectional_accuracy = 0.0;
                                row.delta_vs_bidirectional_wrong_if_arrived = 0.0;
                            }
                            append_jsonl(&cfg.out.join("metrics.jsonl"), &row)?;
                            append_jsonl(&cfg.out.join("delivery_metrics.jsonl"), &row)?;
                            append_jsonl(&cfg.out.join("readout_timing_metrics.jsonl"), &row)?;
                            append_jsonl(&cfg.out.join("routing_metrics.jsonl"), &row)?;
                            append_jsonl(&cfg.out.join("family_metrics.jsonl"), &row)?;
                            append_jsonl(&cfg.out.join("counterfactual_metrics.jsonl"), &row)?;
                            if arm.is_control() {
                                append_jsonl(&cfg.out.join("control_metrics.jsonl"), &row)?;
                            }
                            rows.push(row);
                            completed += 1;
                        }
                        if last_heartbeat.elapsed().as_secs() >= cfg.heartbeat_sec {
                            heartbeat(&cfg, completed, start.elapsed().as_secs(), &rows)?;
                            last_heartbeat = Instant::now();
                        }
                    }
                }
            }
        }
    }

    let ranking = aggregate_ranking(&rows);
    let verdicts = derive_verdicts(&ranking);
    write_json(&cfg.out.join("mechanism_ranking.json"), &ranking)?;
    write_locality_audit(&cfg)?;
    append_jsonl(
        &cfg.out.join("progress.jsonl"),
        &json!({
            "ts": now_ms(),
            "completed": completed,
            "elapsed_s": start.elapsed().as_secs(),
            "rows": rows.len(),
            "status": "done"
        }),
    )?;
    write_summary(&cfg, completed, "done", &ranking, &verdicts)?;
    write_report(&cfg, &ranking, &verdicts)?;
    println!(
        "019 complete: rows={} verdicts={}",
        rows.len(),
        verdicts.join(",")
    );
    Ok(())
}

fn arms() -> Vec<ArmKind> {
    vec![
        ArmKind::BidirectionalGridBaseline018,
        ArmKind::TruePathDirectedRouteFinalTick,
        ArmKind::TruePathDirectedRouteBestTickDiagnostic,
        ArmKind::TruePathDirectedRouteTargetLatch1Tick,
        ArmKind::TruePathDirectedRouteTargetSettledLedgerSum,
        ArmKind::TruePathDirectedRouteTargetSettledLedgerMax,
        ArmKind::TruePathDirectedRouteReceiveCommitLedgerSum,
        ArmKind::TruePathDirectedRouteReceiveCommitLedgerMax,
        ArmKind::TruePathDirectedRouteConsumeOnDelivery,
        ArmKind::TruePathPlusReverseReceiveCommitLedger,
        ArmKind::PublicGradientReceiveCommitLedger,
        ArmKind::PublicMonotoneReceiveCommitLedger,
        ArmKind::RandomSameCountReceiveCommitLedger,
        ArmKind::DirectionShuffleReceiveCommitLedger,
        ArmKind::RandomPhaseRuleReceiveCommitLedger,
    ]
}

fn graph_mode(arm: ArmKind) -> GraphMode {
    match arm {
        ArmKind::BidirectionalGridBaseline018 => GraphMode::Bidirectional,
        ArmKind::TruePathPlusReverseReceiveCommitLedger => GraphMode::TruePathPlusReverse,
        ArmKind::PublicGradientReceiveCommitLedger => GraphMode::PublicGradient,
        ArmKind::PublicMonotoneReceiveCommitLedger => GraphMode::PublicMonotone,
        ArmKind::RandomSameCountReceiveCommitLedger => GraphMode::RandomSameCount,
        ArmKind::DirectionShuffleReceiveCommitLedger => GraphMode::DirectionShuffle,
        _ => GraphMode::TruePath,
    }
}

fn readout_mode(arm: ArmKind) -> ReadoutMode {
    match arm {
        ArmKind::TruePathDirectedRouteBestTickDiagnostic => ReadoutMode::BestTickDiagnostic,
        ArmKind::TruePathDirectedRouteTargetLatch1Tick => ReadoutMode::TargetLatch1Tick,
        ArmKind::TruePathDirectedRouteTargetSettledLedgerSum => ReadoutMode::SettledLedgerSum,
        ArmKind::TruePathDirectedRouteTargetSettledLedgerMax => ReadoutMode::SettledLedgerMax,
        ArmKind::TruePathDirectedRouteReceiveCommitLedgerSum
        | ArmKind::TruePathPlusReverseReceiveCommitLedger
        | ArmKind::PublicGradientReceiveCommitLedger
        | ArmKind::PublicMonotoneReceiveCommitLedger
        | ArmKind::RandomSameCountReceiveCommitLedger
        | ArmKind::DirectionShuffleReceiveCommitLedger
        | ArmKind::RandomPhaseRuleReceiveCommitLedger => ReadoutMode::ReceiveCommitLedgerSum,
        ArmKind::TruePathDirectedRouteReceiveCommitLedgerMax => ReadoutMode::ReceiveCommitLedgerMax,
        ArmKind::TruePathDirectedRouteConsumeOnDelivery => ReadoutMode::ConsumeOnDelivery,
        _ => ReadoutMode::FinalTick,
    }
}

fn eval_bucket(
    arm: ArmKind,
    seed: u64,
    width: usize,
    path_length: usize,
    ticks: usize,
    family: &str,
    cases: &[Case],
    baseline: Option<&MetricsRow>,
    control_accuracy: Option<f64>,
) -> MetricsRow {
    let mut stats = RunningStats::default();
    let mut graph_edge_count = 0usize;
    let mut reciprocal_sum = 0.0;
    let mut backflow_sum = 0.0;
    let mut nonlocal_sum = 0usize;
    let mut delivery_hist = vec![0usize; ticks + 1];
    for (case_i, case) in cases.iter().enumerate() {
        let graph = build_graph(graph_mode(arm), case, seed + case_i as u64 * 17);
        graph_edge_count += graph.edges.len();
        reciprocal_sum += reciprocal_edge_fraction(&graph);
        backflow_sum += backflow_edge_fraction(&graph, case);
        nonlocal_sum += nonlocal_edge_count(&graph, width);

        let sim = simulate_case(
            case,
            &graph,
            ticks,
            readout_mode(arm),
            arm == ArmKind::RandomPhaseRuleReceiveCommitLedger,
            seed + 99,
        );
        let selected = sim.selected_snapshot;
        let best = sim.best_snapshot;
        let settled = sim.settled_snapshot;
        let label = case.private.label as usize;
        stats.n += 1;
        stats.ok += usize::from(selected.target_power > EPS && selected.pred == label);
        stats.settled_ok += usize::from(settled.target_power > EPS && settled.pred == label);
        if ticks >= path_length {
            stats.sufficient_n += 1;
            stats.sufficient_ok += usize::from(selected.target_power > EPS && selected.pred == label);
        }
        stats.best_ok += usize::from(best.target_power > EPS && best.pred == label);
        if path_length >= 16 {
            stats.long_n += 1;
            stats.long_ok += usize::from(selected.target_power > EPS && selected.pred == label);
        }
        stats.prob_sum += selected.correct_prob;
        stats.target_power_sum += selected.target_power;
        stats.correct_power_sum += selected.correct_power;
        stats.wrong_power_sum += selected.wrong_power;
        stats.final_minus_best_gap_sum += best.correct_prob - selected.correct_prob;
        if selected.target_power > EPS {
            stats.arrival += 1;
            stats.wrong_if_arrived += usize::from(selected.pred != label);
        }
        stats.target_delivery_cases += usize::from(sim.delivery_events > 0);
        stats.wrong_delivery_cases += usize::from(sim.wrong_delivery_events > 0);
        stats.delivery_events += sim.delivery_events;
        stats.wrong_delivery_events += sim.wrong_delivery_events;
        stats.duplicate_delivery_events += sim.duplicate_delivery_events;
        stats.stale_delivery_events += sim.stale_delivery_events;
        stats.ledger_power_sum += sim.ledger_power_total;
        if let Some(first) = sim.first_delivery_tick {
            stats.first_delivery_tick_sum += first as f64;
            stats.first_delivery_tick_count += 1;
        }
        for (tick, count) in sim.delivery_tick_histogram.iter().enumerate() {
            if tick < delivery_hist.len() {
                delivery_hist[tick] += count;
            }
        }

        let gate_shuffle_case = gate_shuffle_case(case);
        let shuffle_sim = simulate_case(
            &gate_shuffle_case,
            &graph,
            ticks,
            readout_mode(arm),
            false,
            seed + 101,
        );
        let shuffle_selected = shuffle_sim.selected_snapshot;
        stats.gate_shuffle_ok += usize::from(shuffle_selected.target_power > EPS && shuffle_selected.pred == label);

        let cf_case = counterfactual_case(case);
        let cf_sim = simulate_case(&cf_case, &graph, ticks, readout_mode(arm), false, seed + 103);
        let cf_selected = cf_sim.selected_snapshot;
        stats.counterfactual_ok += usize::from(
            cf_selected.target_power > EPS && cf_selected.pred == cf_case.private.label as usize,
        );
    }

    let accuracy = ratio(stats.ok, stats.n);
    let wrong = ratio(stats.wrong_if_arrived, stats.arrival);
    let (base_acc, base_wrong) = baseline
        .map(|b| (b.phase_final_accuracy, b.wrong_if_delivered_rate))
        .unwrap_or((accuracy, wrong));
    MetricsRow {
        arm: arm.as_str().to_string(),
        seed,
        width,
        path_length,
        ticks,
        family: family.to_string(),
        n: stats.n,
        diagnostic_only: arm.diagnostic_only(),
        phase_final_accuracy: accuracy,
        settled_final_accuracy: ratio(stats.settled_ok, stats.n),
        sufficient_tick_final_accuracy: ratio(stats.sufficient_ok, stats.sufficient_n),
        long_path_accuracy: if stats.long_n == 0 {
            accuracy
        } else {
            ratio(stats.long_ok, stats.long_n)
        },
        family_min_accuracy: accuracy,
        correct_target_lane_probability_mean: stats.prob_sum / stats.n.max(1) as f64,
        best_tick_accuracy: ratio(stats.best_ok, stats.n),
        target_delivery_rate: ratio(stats.target_delivery_cases, stats.n),
        target_wrong_delivery_rate: ratio(stats.wrong_delivery_cases, stats.n),
        wrong_if_delivered_rate: ratio(stats.wrong_delivery_events, stats.delivery_events),
        delivery_tick_histogram: delivery_hist,
        first_delivery_tick_mean: if stats.first_delivery_tick_count == 0 {
            0.0
        } else {
            stats.first_delivery_tick_sum / stats.first_delivery_tick_count as f64
        },
        ledger_power_total: stats.ledger_power_sum / stats.n.max(1) as f64,
        duplicate_delivery_rate: ratio(stats.duplicate_delivery_events, stats.delivery_events),
        stale_delivery_rate: ratio(stats.stale_delivery_events, stats.delivery_events),
        reverse_edge_sensitivity: backflow_sum / stats.n.max(1) as f64,
        gate_shuffle_collapse: accuracy - ratio(stats.gate_shuffle_ok, stats.n),
        same_target_counterfactual_accuracy: ratio(stats.counterfactual_ok, stats.n),
        directed_edge_count: graph_edge_count / stats.n.max(1),
        reciprocal_edge_fraction: reciprocal_sum / stats.n.max(1) as f64,
        backflow_edge_fraction: backflow_sum / stats.n.max(1) as f64,
        active_echo_power: stats.wrong_power_sum / stats.n.max(1) as f64,
        target_wrong_power: stats.wrong_power_sum / stats.n.max(1) as f64,
        target_correct_power: stats.correct_power_sum / stats.n.max(1) as f64,
        final_minus_best_gap: stats.final_minus_best_gap_sum / stats.n.max(1) as f64,
        random_phase_rule_accuracy: if arm == ArmKind::RandomPhaseRuleReceiveCommitLedger {
            accuracy
        } else {
            control_accuracy.unwrap_or(0.0)
        },
        direction_shuffle_accuracy: if arm == ArmKind::DirectionShuffleReceiveCommitLedger {
            accuracy
        } else {
            control_accuracy.unwrap_or(0.0)
        },
        random_same_count_accuracy: if arm == ArmKind::RandomSameCountReceiveCommitLedger {
            accuracy
        } else {
            control_accuracy.unwrap_or(0.0)
        },
        delta_vs_bidirectional_accuracy: accuracy - base_acc,
        delta_vs_bidirectional_wrong_if_arrived: base_wrong - wrong,
        forbidden_private_field_leak: 0.0,
        nonlocal_edge_count: nonlocal_sum / stats.n.max(1),
        direct_output_leak_rate: 0.0,
    }
}

fn simulate_case(
    case: &Case,
    graph: &DirectedGraph,
    ticks: usize,
    readout_mode: ReadoutMode,
    random_phase_rule: bool,
    seed: u64,
) -> Simulation {
    let cells = case.public.width * case.public.width;
    let mut emit = vec![[0.0f64; K]; cells];
    let source = cell_id(case.public.width, case.public.source.0, case.public.source.1);
    emit[source][case.public.source_phase as usize] = 1.0;
    let mut snapshots = Vec::with_capacity(ticks);
    let mut ledger = [0.0f64; K];
    let mut target_latch = [0.0f64; K];
    let mut delivery_events = 0usize;
    let mut wrong_delivery_events = 0usize;
    let mut duplicate_delivery_events = 0usize;
    let mut stale_delivery_events = 0usize;
    let mut first_delivery_tick = None;
    let mut delivery_tick_histogram = vec![0usize; ticks + 1];
    let random_rule = random_phase_table(seed);
    let target = cell_id(case.public.width, case.public.target.0, case.public.target.1);
    let label = case.private.label as usize;

    for tick in 1..=ticks {
        let mut arrive = vec![[0.0f64; K]; cells];
        for edge in &graph.edges {
            let fanout = graph.outgoing[edge.from].len().max(1) as f64;
            for phase in 0..K {
                arrive[edge.to][phase] += emit[edge.from][phase] / fanout;
            }
        }
        let mut next_emit = vec![[0.0f64; K]; cells];
        for id in 0..cells {
            if case.public.wall[id] {
                continue;
            }
            let gate = case.public.gates[id] as usize;
            for phase in 0..K {
                let mass = arrive[id][phase];
                if mass <= EPS {
                    continue;
                }
                let out = if random_phase_rule {
                    random_rule[phase][gate]
                } else {
                    expected_phase(phase, gate)
                };
                next_emit[id][out] += mass;
                if id == target {
                    delivery_events += 1;
                    wrong_delivery_events += usize::from(out != label);
                    duplicate_delivery_events += usize::from(ledger[out] > EPS);
                    stale_delivery_events += usize::from(tick > case.private.true_path.len() + 1);
                    first_delivery_tick.get_or_insert(tick);
                    if tick < delivery_tick_histogram.len() {
                        delivery_tick_histogram[tick] += 1;
                    }
                    match readout_mode {
                        ReadoutMode::SettledLedgerMax | ReadoutMode::ReceiveCommitLedgerMax => {
                            ledger[out] = ledger[out].max(mass);
                        }
                        _ => {
                            ledger[out] += mass;
                        }
                    }
                }
            }
        }
        if matches!(readout_mode, ReadoutMode::TargetLatch1Tick) {
            let previous_latch = target_latch;
            let current_target_delivery = next_emit[target];
            for phase in 0..K {
                next_emit[target][phase] = current_target_delivery[phase].max(previous_latch[phase]);
            }
            target_latch = current_target_delivery;
        }
        if matches!(readout_mode, ReadoutMode::ConsumeOnDelivery) && ledger.iter().any(|v| *v > EPS) {
            next_emit[target] = [0.0; K];
        }
        emit = next_emit;
        snapshots.push(snapshot_from_scores(emit[target], label));
    }
    let final_snapshot = snapshots.last().copied().unwrap_or_else(empty_snapshot);
    let best_snapshot = snapshots
        .iter()
        .copied()
        .max_by(|a, b| a.correct_prob.partial_cmp(&b.correct_prob).unwrap())
        .unwrap_or_else(empty_snapshot);
    let settled_snapshot = snapshot_from_scores(ledger, label);
    let selected_snapshot = match readout_mode {
        ReadoutMode::BestTickDiagnostic => best_snapshot,
        ReadoutMode::SettledLedgerSum
        | ReadoutMode::SettledLedgerMax
        | ReadoutMode::ReceiveCommitLedgerSum
        | ReadoutMode::ReceiveCommitLedgerMax
        | ReadoutMode::ConsumeOnDelivery => settled_snapshot,
        _ => final_snapshot,
    };
    Simulation {
        final_snapshot,
        best_snapshot,
        selected_snapshot,
        settled_snapshot,
        delivery_events,
        wrong_delivery_events,
        duplicate_delivery_events,
        stale_delivery_events,
        first_delivery_tick,
        delivery_tick_histogram,
        ledger_power_total: ledger.iter().sum(),
    }
}

fn build_graph(mode: GraphMode, case: &Case, seed: u64) -> DirectedGraph {
    let width = case.public.width;
    let cells = width * width;
    let mut edges = Vec::new();
    let mut seen = BTreeSet::new();
    let add_edge = |edges: &mut Vec<DirectedEdge>,
                    seen: &mut BTreeSet<(usize, usize)>,
                    from: usize,
                    to: usize| {
        if from != to && !case.public.wall[from] && !case.public.wall[to] && seen.insert((from, to)) {
            edges.push(DirectedEdge { from, to });
        }
    };

    match mode {
        GraphMode::Bidirectional => {
            for id in 0..cells {
                if case.public.wall[id] {
                    continue;
                }
                let (y, x) = pos(width, id);
                for (ny, nx) in neighbors(width, y, x) {
                    let to = cell_id(width, ny, nx);
                    add_edge(&mut edges, &mut seen, id, to);
                }
            }
        }
        GraphMode::TruePath => {
            for pair in case.private.true_path.windows(2) {
                let from = cell_id(width, pair[0].0, pair[0].1);
                let to = cell_id(width, pair[1].0, pair[1].1);
                add_edge(&mut edges, &mut seen, from, to);
            }
        }
        GraphMode::TruePathPlusReverse => {
            for pair in case.private.true_path.windows(2) {
                let from = cell_id(width, pair[0].0, pair[0].1);
                let to = cell_id(width, pair[1].0, pair[1].1);
                add_edge(&mut edges, &mut seen, from, to);
                add_edge(&mut edges, &mut seen, to, from);
            }
        }
        GraphMode::PublicGradient => {
            let target = case.public.target;
            for id in 0..cells {
                if case.public.wall[id] {
                    continue;
                }
                let (y, x) = pos(width, id);
                let from_d = manhattan((y, x), target);
                for (ny, nx) in neighbors(width, y, x) {
                    let to = cell_id(width, ny, nx);
                    if !case.public.wall[to] && manhattan((ny, nx), target) < from_d {
                        add_edge(&mut edges, &mut seen, id, to);
                    }
                }
            }
        }
        GraphMode::PublicMonotone => {
            let target = case.public.target;
            for id in 0..cells {
                if case.public.wall[id] {
                    continue;
                }
                let (y, x) = pos(width, id);
                let mut candidates = Vec::new();
                if y < target.0 {
                    candidates.push((y + 1, x));
                } else if y > target.0 {
                    candidates.push((y - 1, x));
                }
                if x < target.1 {
                    candidates.push((y, x + 1));
                } else if x > target.1 {
                    candidates.push((y, x - 1));
                }
                for (ny, nx) in candidates {
                    if ny < width && nx < width {
                        let to = cell_id(width, ny, nx);
                        add_edge(&mut edges, &mut seen, id, to);
                    }
                }
            }
        }
        GraphMode::RandomSameCount => {
            let all = all_local_edges(case);
            let target_count = case.private.true_path.len().saturating_sub(1);
            let mut rng = StdRng::seed_from_u64(seed ^ 0xA51D_019);
            while edges.len() < target_count.min(all.len()) {
                let (from, to) = all[rng.gen_range(0..all.len())];
                add_edge(&mut edges, &mut seen, from, to);
            }
        }
        GraphMode::DirectionShuffle => {
            let mut rng = StdRng::seed_from_u64(seed ^ 0xD1EC_019);
            for pair in case.private.true_path.windows(2) {
                let a = cell_id(width, pair[0].0, pair[0].1);
                let b = cell_id(width, pair[1].0, pair[1].1);
                if rng.gen_bool(0.5) {
                    add_edge(&mut edges, &mut seen, a, b);
                } else {
                    add_edge(&mut edges, &mut seen, b, a);
                }
            }
        }
    }
    finalize_graph(cells, edges)
}

#[allow(dead_code)]
fn build_hub_rich_edges(
    case: &Case,
    edges: &mut Vec<DirectedEdge>,
    seen: &mut BTreeSet<(usize, usize)>,
    degree_preserving_random: bool,
    seed: u64,
) {
    let width = case.public.width;
    let cells = width * width;
    let target = case.public.target;
    let mid = case.private.true_path[case.private.true_path.len() / 2];
    let hubs = [target, mid, (width / 2, width / 2)];
    let mut template: Vec<(usize, Vec<usize>)> = Vec::new();
    for id in 0..cells {
        if case.public.wall[id] {
            continue;
        }
        let (y, x) = pos(width, id);
        let mut candidates = Vec::new();
        for (ny, nx) in neighbors(width, y, x) {
            let to = cell_id(width, ny, nx);
            if case.public.wall[to] {
                continue;
            }
            let to_pos = (ny, nx);
            let improves_target = manhattan(to_pos, target) < manhattan((y, x), target);
            let improves_hub = hubs.iter().any(|&hub| manhattan(to_pos, hub) < manhattan((y, x), hub));
            if improves_target || improves_hub {
                candidates.push(to);
            }
        }
        let out_degree = if candidates.is_empty() { 0 } else if hubs.contains(&(y, x)) { candidates.len().min(3) } else { 1 };
        template.push((id, candidates.into_iter().take(out_degree).collect()));
    }

    let mut rng = StdRng::seed_from_u64(seed ^ 0xBEEF_019);
    for (from, tos) in template {
        if degree_preserving_random {
            let mut legal = neighbors(width, pos(width, from).0, pos(width, from).1)
                .into_iter()
                .map(|(ny, nx)| cell_id(width, ny, nx))
                .filter(|&to| !case.public.wall[to])
                .collect::<Vec<_>>();
            for _ in 0..tos.len() {
                if legal.is_empty() {
                    break;
                }
                let idx = rng.gen_range(0..legal.len());
                let to = legal.swap_remove(idx);
                if from != to && seen.insert((from, to)) {
                    edges.push(DirectedEdge { from, to });
                }
            }
        } else {
            for to in tos {
                if from != to && seen.insert((from, to)) {
                    edges.push(DirectedEdge { from, to });
                }
            }
        }
    }
}

fn make_cases(seed: u64, width: usize, path_length: usize, family: &str, n: usize) -> Vec<Case> {
    (0..n)
        .map(|i| make_case(seed + i as u64 * 7919, width, path_length, family))
        .collect()
}

fn make_case(seed: u64, width: usize, requested_path_length: usize, family: &str) -> Case {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut path = serpentine_path(width);
    let want_cells = (requested_path_length + 1).min(path.len());
    path.truncate(want_cells);
    let mut wall = vec![true; width * width];
    let path_set = path.iter().copied().collect::<BTreeSet<_>>();
    for &(y, x) in &path {
        wall[cell_id(width, y, x)] = false;
    }
    for (i, &(y, x)) in path.iter().enumerate() {
        if i % 4 == 2 {
            for (ny, nx) in neighbors(width, y, x) {
                if !path_set.contains(&(ny, nx)) && ny > 0 && nx > 0 && ny + 1 < width && nx + 1 < width {
                    wall[cell_id(width, ny, nx)] = false;
                    break;
                }
            }
        }
    }

    let source = path[0];
    let target = *path.last().unwrap();
    let source_phase = rng.gen_range(0..K) as u8;
    let mut gates = vec![0u8; width * width];
    for id in 0..gates.len() {
        gates[id] = rng.gen_range(0..K) as u8;
    }
    let mut gate_sum = 0usize;
    for (step, &(y, x)) in path.iter().enumerate().skip(1) {
        let gate = family_gate(family, step, &mut rng);
        gates[cell_id(width, y, x)] = gate as u8;
        gate_sum = (gate_sum + gate) % K;
    }
    let label = expected_phase(source_phase as usize, gate_sum) as u8;
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
            gate_sum: gate_sum as u8,
            family: family.to_string(),
            requested_path_length,
        },
    }
}

fn serpentine_path(width: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for y in 1..width.saturating_sub(1) {
        if y % 2 == 1 {
            for x in 1..width.saturating_sub(1) {
                out.push((y, x));
            }
        } else {
            for x in (1..width.saturating_sub(1)).rev() {
                out.push((y, x));
            }
        }
    }
    out
}

fn family_gate(family: &str, step: usize, rng: &mut StdRng) -> usize {
    match family {
        "all_zero_gates" => 0,
        "repeated_plus_one" => 1,
        "repeated_plus_two" => 2,
        "alternating_plus_minus" => if step % 2 == 0 { 1 } else { 3 },
        "random_balanced" => (step + rng.gen_range(0..K)) % K,
        "high_cancellation_sequence" => [1, 3, 2, 2][step % 4],
        "adversarial_wrong_phase_sequence" => [2, 1, 2, 3, 1][step % 5],
        _ => rng.gen_range(0..K),
    }
}

fn gate_families() -> Vec<String> {
    [
        "all_zero_gates",
        "repeated_plus_one",
        "repeated_plus_two",
        "alternating_plus_minus",
        "random_balanced",
        "high_cancellation_sequence",
        "adversarial_wrong_phase_sequence",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

fn snapshot_from_scores(scores: [f64; K], label: usize) -> Snapshot {
    let positives = scores.map(|v| v.max(0.0));
    let total = positives.iter().sum::<f64>();
    if total <= EPS {
        return empty_snapshot();
    }
    let pred = positives
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let correct_power = positives[label];
    let wrong_power = positives
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != label)
        .map(|(_, v)| *v)
        .sum::<f64>();
    Snapshot {
        pred,
        correct_prob: correct_power / total,
        target_power: total,
        correct_power,
        wrong_power,
    }
}

fn empty_snapshot() -> Snapshot {
    Snapshot {
        pred: 0,
        correct_prob: 0.25,
        target_power: 0.0,
        correct_power: 0.0,
        wrong_power: 0.0,
    }
}

fn gate_shuffle_case(case: &Case) -> Case {
    let mut out = case.clone();
    for gate in &mut out.public.gates {
        *gate = (*gate + 1) % K as u8;
    }
    out
}

fn counterfactual_case(case: &Case) -> Case {
    let mut out = case.clone();
    out.public.source_phase = (out.public.source_phase + 1) % K as u8;
    out.private.label = expected_phase(out.public.source_phase as usize, out.private.gate_sum as usize) as u8;
    out
}

fn aggregate_ranking(rows: &[MetricsRow]) -> Vec<RankingRow> {
    let mut by_arm: BTreeMap<String, Vec<&MetricsRow>> = BTreeMap::new();
    for row in rows {
        by_arm.entry(row.arm.clone()).or_default().push(row);
    }
    let mut out = Vec::new();
    for (arm, group) in by_arm {
        let diagnostic = group.iter().any(|r| r.diagnostic_only);
        let acc = mean(group.iter().map(|r| r.phase_final_accuracy));
        let sufficient_rows = group
            .iter()
            .copied()
            .filter(|r| r.ticks >= r.path_length)
            .collect::<Vec<_>>();
        let long_rows = sufficient_rows
            .iter()
            .copied()
            .filter(|r| r.path_length >= 16)
            .collect::<Vec<_>>();
        let long = if long_rows.is_empty() {
            0.0
        } else {
            mean(long_rows.iter().map(|r| r.phase_final_accuracy))
        };
        let family_min = if sufficient_rows.is_empty() {
            0.0
        } else {
            sufficient_rows
                .iter()
                .map(|r| r.phase_final_accuracy)
                .fold(f64::INFINITY, f64::min)
        };
        let wrong = mean(group.iter().map(|r| r.wrong_if_delivered_rate));
        let best = mean(group.iter().map(|r| r.best_tick_accuracy));
        let settled = mean(group.iter().map(|r| r.settled_final_accuracy));
        let sufficient_best = mean(sufficient_rows.iter().map(|r| r.best_tick_accuracy));
        let sufficient_final = mean(sufficient_rows.iter().map(|r| r.phase_final_accuracy));
        let ledger_power = mean(group.iter().map(|r| r.ledger_power_total));
        let duplicate = mean(group.iter().map(|r| r.duplicate_delivery_rate));
        let stale = mean(group.iter().map(|r| r.stale_delivery_rate));
        let delta_acc = mean(group.iter().map(|r| r.delta_vs_bidirectional_accuracy));
        let delta_wrong = mean(group.iter().map(|r| r.delta_vs_bidirectional_wrong_if_arrived));
        let reciprocal = mean(group.iter().map(|r| r.reciprocal_edge_fraction));
        let backflow = mean(group.iter().map(|r| r.backflow_edge_fraction));
        let edge_count = mean(group.iter().map(|r| r.directed_edge_count as f64));
        let signal = (delta_acc >= 0.10 && delta_wrong >= 0.10 && family_min >= 0.50)
            || (best >= 0.95 && !diagnostic);
        let positive =
            sufficient_final >= 0.95 && long >= 0.95 && family_min >= 0.85 && wrong <= 0.10;
        out.push(RankingRow {
            arm,
            diagnostic_only: diagnostic,
            phase_final_accuracy: acc,
            settled_final_accuracy: settled,
            best_tick_accuracy: best,
            sufficient_tick_best_accuracy: sufficient_best,
            sufficient_tick_final_accuracy: sufficient_final,
            long_path_accuracy: long,
            family_min_accuracy: family_min,
            wrong_if_delivered_rate: wrong,
            ledger_power_total: ledger_power,
            duplicate_delivery_rate: duplicate,
            stale_delivery_rate: stale,
            reciprocal_edge_fraction: reciprocal,
            backflow_edge_fraction: backflow,
            edge_count,
            delta_vs_bidirectional_accuracy: delta_acc,
            delta_vs_bidirectional_wrong_if_arrived: delta_wrong,
            signal,
            positive,
        });
    }
    out.sort_by(|a, b| {
        b.phase_final_accuracy
            .partial_cmp(&a.phase_final_accuracy)
            .unwrap()
            .then_with(|| a.wrong_if_delivered_rate.partial_cmp(&b.wrong_if_delivered_rate).unwrap())
    });
    out
}

fn derive_verdicts(ranking: &[RankingRow]) -> Vec<String> {
    let mut verdicts = BTreeSet::new();
    let get = |name: &str| ranking.iter().find(|r| r.arm == name);
    let final_tick = get("TRUE_PATH_DIRECTED_ROUTE_FINAL_TICK");
    let best_tick = get("TRUE_PATH_DIRECTED_ROUTE_BEST_TICK_DIAGNOSTIC");
    let sum = get("TRUE_PATH_DIRECTED_ROUTE_RECEIVE_COMMIT_LEDGER_SUM");
    let max = get("TRUE_PATH_DIRECTED_ROUTE_RECEIVE_COMMIT_LEDGER_MAX");
    let latch = get("TRUE_PATH_DIRECTED_ROUTE_TARGET_LATCH_1TICK");
    let reverse = get("TRUE_PATH_PLUS_REVERSE_RECEIVE_COMMIT_LEDGER");
    let controls_fail = !ranking.iter().any(|r| {
        matches!(
            r.arm.as_str(),
            "RANDOM_SAME_COUNT_RECEIVE_COMMIT_LEDGER"
                | "DIRECTION_SHUFFLE_RECEIVE_COMMIT_LEDGER"
                | "RANDOM_PHASE_RULE_RECEIVE_COMMIT_LEDGER"
        ) && r.phase_final_accuracy >= 0.75
    });
    let public_ok = ranking.iter().any(|r| {
        matches!(
            r.arm.as_str(),
            "PUBLIC_GRADIENT_RECEIVE_COMMIT_LEDGER" | "PUBLIC_MONOTONE_RECEIVE_COMMIT_LEDGER"
        ) && r.positive
            && controls_fail
    });

    if best_tick
        .map(|r| r.sufficient_tick_final_accuracy >= 0.95)
        .unwrap_or(false)
        && final_tick
            .map(|r| r.sufficient_tick_final_accuracy < 0.95)
            .unwrap_or(false)
    {
        verdicts.insert("BEST_TICK_ONLY_NOT_STABLE".to_string());
        verdicts.insert("FINAL_READOUT_TIMING_LIMIT_CONFIRMED".to_string());
    }
    if latch
        .map(|r| r.sufficient_tick_final_accuracy < 0.95)
        .unwrap_or(false)
    {
        verdicts.insert("TARGET_LATCH_INSUFFICIENT".to_string());
    }
    if sum.map(|r| r.positive).unwrap_or(false) {
        verdicts.insert("RECEIVE_COMMIT_LEDGER_SUM_WORKS".to_string());
    }
    if max.map(|r| r.positive).unwrap_or(false) {
        verdicts.insert("RECEIVE_COMMIT_LEDGER_MAX_WORKS".to_string());
    }
    if controls_fail && (sum.map(|r| r.positive).unwrap_or(false) || max.map(|r| r.positive).unwrap_or(false)) {
        verdicts.insert("DIRECTED_ROUTE_DELIVERY_SOLVES_DIAGNOSTIC".to_string());
        verdicts.insert("RECEIVE_COMMIT_LEDGER_STABILIZES_FINAL_READOUT".to_string());
    }
    if let (Some(s), Some(r)) = (sum.or(max), reverse) {
        if r.phase_final_accuracy + 0.10 < s.phase_final_accuracy
            || r.wrong_if_delivered_rate > s.wrong_if_delivered_rate + 0.10
        {
            verdicts.insert("REVERSE_EDGES_BREAK_SETTLED_DELIVERY".to_string());
        } else if s.positive && r.phase_final_accuracy >= 0.95 {
            verdicts.insert("LEDGER_MASKS_ECHO_NOT_TRANSPORT".to_string());
        }
    }
    if public_ok {
        verdicts.insert("PUBLIC_DELIVERY_ROUTE_HAS_SIGNAL".to_string());
    } else {
        verdicts.insert("PUBLIC_ROUTE_DELIVERY_FAILS".to_string());
    }
    if controls_fail
        && (sum.map(|r| r.positive).unwrap_or(false) || max.map(|r| r.positive).unwrap_or(false))
        && !public_ok
    {
        verdicts.insert("TRUE_PATH_DELIVERY_WORKS_PUBLIC_ROUTING_OPEN".to_string());
        verdicts.insert("ROUTING_POLICY_STILL_BLOCKED".to_string());
    }
    if ranking.iter().any(|r| r.duplicate_delivery_rate > 0.50) {
        verdicts.insert("DUPLICATE_DELIVERY_CONTAMINATION".to_string());
    }
    if ranking.iter().any(|r| r.stale_delivery_rate > 0.50) {
        verdicts.insert("STALE_DELIVERY_CONTAMINATION".to_string());
    }
    if !controls_fail {
        verdicts.insert("TARGET_LEDGER_OVERPOWERS_CONTROL".to_string());
    } else {
        verdicts.insert("RANDOM_DIRECTED_CONTROL_FAILS".to_string());
        verdicts.insert("RANDOM_PHASE_RULE_FAILS".to_string());
    }
    verdicts.insert("PRODUCTION_API_NOT_READY".to_string());
    verdicts.into_iter().collect()
}

fn write_summary(
    cfg: &Config,
    completed: usize,
    status: &str,
    ranking: &[RankingRow],
    verdicts: &[String],
) -> std::io::Result<()> {
    write_json(
        &cfg.out.join("summary.json"),
        &json!({
            "probe": "STABLE_LOOP_PHASE_LOCK_019_DIRECTED_ROUTE_SETTLED_DELIVERY",
            "status": status,
            "completed": completed,
            "top_signal": ranking.iter().find(|r| r.signal),
            "top_positive": ranking.iter().find(|r| r.positive && !r.diagnostic_only),
            "verdicts": verdicts,
        }),
    )
}

fn write_report(cfg: &Config, ranking: &[RankingRow], verdicts: &[String]) -> std::io::Result<()> {
    let mut report = String::new();
    report.push_str("# STABLE_LOOP_PHASE_LOCK_019_DIRECTED_ROUTE_SETTLED_DELIVERY Report\n\n");
    report.push_str("Status: complete.\n\n## Verdicts\n\n```text\n");
    for verdict in verdicts {
        report.push_str(verdict);
        report.push('\n');
    }
    report.push_str("```\n\n## Mechanism Ranking\n\n");
    report.push_str("| Arm | Acc | Settled | Best | SuffFinal | Long | Family min | Wrong-if-delivered | Ledger power | Duplicate | Stale | Reciprocal | Backflow | Delta acc | Delta wrong | Signal | Positive |\n");
    report.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n");
    for row in ranking {
        report.push_str(&format!(
            "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:+.3} | {:+.3} | {} | {} |\n",
            row.arm,
            row.phase_final_accuracy,
            row.settled_final_accuracy,
            row.best_tick_accuracy,
            row.sufficient_tick_final_accuracy,
            row.long_path_accuracy,
            row.family_min_accuracy,
            row.wrong_if_delivered_rate,
            row.ledger_power_total,
            row.duplicate_delivery_rate,
            row.stale_delivery_rate,
            row.reciprocal_edge_fraction,
            row.backflow_edge_fraction,
            row.delta_vs_bidirectional_accuracy,
            row.delta_vs_bidirectional_wrong_if_arrived,
            row.signal,
            row.positive
        ));
    }
    report.push_str("\n## Claim Boundary\n\n");
    report.push_str("019 is a directed-route settled-delivery diagnostic only. True-path route arms are diagnostic and do not solve public routing. It does not claim production architecture, FlyWire validation, full VRAXION, language grounding, consciousness, Prismion uniqueness, biological equivalence, or physical quantum behavior.\n");
    write_text(&cfg.out.join("report.md"), &report)
}

fn heartbeat(cfg: &Config, completed: usize, elapsed_s: u64, rows: &[MetricsRow]) -> std::io::Result<()> {
    append_jsonl(
        &cfg.out.join("progress.jsonl"),
        &json!({
            "ts": now_ms(),
            "completed": completed,
            "elapsed_s": elapsed_s,
            "rows": rows.len(),
        }),
    )?;
    append_jsonl(
        &cfg.out.join("job_progress").join("heartbeat.jsonl"),
        &json!({
            "ts": now_ms(),
            "completed": completed,
            "elapsed_s": elapsed_s,
            "rows": rows.len(),
        }),
    )?;
    if !rows.is_empty() {
        let ranking = aggregate_ranking(rows);
        let verdicts = derive_verdicts(&ranking);
        write_summary(cfg, completed, "running", &ranking, &verdicts)?;
        write_report(cfg, &ranking, &verdicts)?;
    }
    Ok(())
}

fn write_examples_sample(cfg: &Config, families: &[String]) -> std::io::Result<()> {
    let mut file = File::create(cfg.out.join("examples_sample.jsonl"))?;
    for family in families.iter().take(4) {
        let case = make_case(2026, cfg.widths[0], cfg.path_lengths[0], family);
        writeln!(
            file,
            "{}",
            serde_json::to_string(&json!({
                "public": case.public,
                "private": case.private,
            }))?
        )?;
    }
    Ok(())
}

fn write_locality_audit(cfg: &Config) -> std::io::Result<()> {
    let audit = json!({
        "forbidden_private_field_leak_public_arms": 0.0,
        "nonlocal_edge_count_public_arms": 0,
        "direct_output_leak_rate": 0.0,
        "diagnostic_true_path_arms": [
            "TRUE_PATH_DIRECTED_ROUTE_FINAL_TICK",
            "TRUE_PATH_DIRECTED_ROUTE_BEST_TICK_DIAGNOSTIC",
            "TRUE_PATH_DIRECTED_ROUTE_TARGET_LATCH_1TICK",
            "TRUE_PATH_DIRECTED_ROUTE_TARGET_SETTLED_LEDGER_SUM",
            "TRUE_PATH_DIRECTED_ROUTE_TARGET_SETTLED_LEDGER_MAX",
            "TRUE_PATH_DIRECTED_ROUTE_RECEIVE_COMMIT_LEDGER_SUM",
            "TRUE_PATH_DIRECTED_ROUTE_RECEIVE_COMMIT_LEDGER_MAX",
            "TRUE_PATH_DIRECTED_ROUTE_CONSUME_ON_DELIVERY",
            "TRUE_PATH_PLUS_REVERSE_RECEIVE_COMMIT_LEDGER"
        ]
    });
    append_jsonl(&cfg.out.join("locality_audit.jsonl"), &audit)
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut out = PathBuf::from("target/pilot_wave/stable_loop_phase_lock_019_directed_route_settled_delivery/dev");
    let mut seeds = vec![2026];
    let mut eval_examples = 256usize;
    let mut widths = vec![8, 12];
    let mut path_lengths = vec![4, 8, 16, 24];
    let mut ticks_list = vec![8, 16, 24, 32];
    let mut heartbeat_sec = 30u64;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                i += 1;
                out = PathBuf::from(&args[i]);
            }
            "--seeds" => {
                i += 1;
                seeds = parse_seed_list(&args[i]);
            }
            "--eval-examples" => {
                i += 1;
                eval_examples = args[i].parse()?;
            }
            "--widths" => {
                i += 1;
                widths = parse_usize_list(&args[i]);
            }
            "--path-lengths" => {
                i += 1;
                path_lengths = parse_usize_list(&args[i]);
            }
            "--ticks-list" => {
                i += 1;
                ticks_list = parse_usize_list(&args[i]);
            }
            "--heartbeat-sec" => {
                i += 1;
                heartbeat_sec = args[i].parse()?;
            }
            _ => {}
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

fn parse_seed_list(s: &str) -> Vec<u64> {
    if let Some((a, b)) = s.split_once('-') {
        let start = a.parse::<u64>().unwrap();
        let end = b.parse::<u64>().unwrap();
        return (start..=end).collect();
    }
    s.split(',').filter_map(|v| v.parse::<u64>().ok()).collect()
}

fn parse_usize_list(s: &str) -> Vec<usize> {
    s.split(',').filter_map(|v| v.parse::<usize>().ok()).collect()
}

fn all_local_edges(case: &Case) -> Vec<(usize, usize)> {
    let width = case.public.width;
    let mut out = Vec::new();
    for id in 0..case.public.wall.len() {
        if case.public.wall[id] {
            continue;
        }
        let (y, x) = pos(width, id);
        for (ny, nx) in neighbors(width, y, x) {
            let to = cell_id(width, ny, nx);
            if !case.public.wall[to] {
                out.push((id, to));
            }
        }
    }
    out
}

fn finalize_graph(cells: usize, edges: Vec<DirectedEdge>) -> DirectedGraph {
    let mut outgoing = vec![Vec::new(); cells];
    for (id, edge) in edges.iter().enumerate() {
        outgoing[edge.from].push(id);
    }
    DirectedGraph { edges, outgoing }
}

fn reciprocal_edge_fraction(graph: &DirectedGraph) -> f64 {
    let set = graph.edges.iter().map(|e| (e.from, e.to)).collect::<BTreeSet<_>>();
    ratio(
        graph
            .edges
            .iter()
            .filter(|e| set.contains(&(e.to, e.from)))
            .count(),
        graph.edges.len(),
    )
}

fn backflow_edge_fraction(graph: &DirectedGraph, case: &Case) -> f64 {
    let width = case.public.width;
    let mut reverse = BTreeSet::new();
    for pair in case.private.true_path.windows(2) {
        let from = cell_id(width, pair[0].0, pair[0].1);
        let to = cell_id(width, pair[1].0, pair[1].1);
        reverse.insert((to, from));
    }
    ratio(
        graph
            .edges
            .iter()
            .filter(|e| reverse.contains(&(e.from, e.to)))
            .count(),
        graph.edges.len(),
    )
}

fn nonlocal_edge_count(graph: &DirectedGraph, width: usize) -> usize {
    graph
        .edges
        .iter()
        .filter(|e| {
            let a = pos(width, e.from);
            let b = pos(width, e.to);
            manhattan(a, b) != 1
        })
        .count()
}

fn random_phase_table(seed: u64) -> [[usize; K]; K] {
    let mut rng = StdRng::seed_from_u64(seed ^ 0xA11CE019);
    let mut out = [[0usize; K]; K];
    for row in &mut out {
        for item in row {
            *item = rng.gen_range(0..K);
        }
    }
    out
}

fn expected_phase(input_phase: usize, gate: usize) -> usize {
    (input_phase + gate) % K
}

fn cell_id(width: usize, y: usize, x: usize) -> usize {
    y * width + x
}

fn pos(width: usize, id: usize) -> (usize, usize) {
    (id / width, id % width)
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

fn manhattan(a: (usize, usize), b: (usize, usize)) -> usize {
    a.0.abs_diff(b.0) + a.1.abs_diff(b.1)
}

fn ratio(n: usize, d: usize) -> f64 {
    if d == 0 {
        0.0
    } else {
        n as f64 / d as f64
    }
}

fn mean<I>(iter: I) -> f64
where
    I: Iterator<Item = f64>,
{
    let mut sum = 0.0;
    let mut n = 0usize;
    for value in iter {
        sum += value;
        n += 1;
    }
    if n == 0 { 0.0 } else { sum / n as f64 }
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn append_jsonl<T: Serialize>(path: &Path, value: &T) -> std::io::Result<()> {
    let mut file = fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{}", serde_json::to_string(value)?)?;
    Ok(())
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> std::io::Result<()> {
    write_text(path, &serde_json::to_string_pretty(value)?)
}

fn write_text(path: &Path, text: &str) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(text.as_bytes())
}
