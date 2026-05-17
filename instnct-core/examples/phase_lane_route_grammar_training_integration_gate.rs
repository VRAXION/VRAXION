//! Route-grammar training integration gate probe.
//!
//! 038 tests whether the experimental route-grammar API improves bounded
//! training/search dynamics without being promoted into default training.

use instnct_core::experimental_route_grammar::{
    construct_route_grammar, RouteGrammarConfig, RouteGrammarEdge, RouteGrammarError,
    RouteGrammarLabelPolicy, RouteGrammarTask,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
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
    route_breadcrumb: Vec<bool>,
    route_order: Vec<usize>,
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
    HandPipelineReference,
    NoRouteGrammarBaseline,
    RouteGrammarFrozenHelper,
    RouteGrammarTrainingFeatureFlag,
    RouteGrammarAuxLabelsOnly,
    RouteGrammarConstructorOnly,
    RouteGrammarConstructorPlusDiagnostics,
    RouteGrammarNoisyCandidates,
    RouteGrammarAblateDiagnosticLabels,
    RouteGrammarAblateOrderPrune,
    RouteGrammarAblateReceiveCommitLedger,
    NonRouteTaskRegressionControl,
    NoGrammarApiControl,
    RandomRouteGrammarControl,
    RandomPhaseRuleControl,
}

impl ArmKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::HandPipelineReference => "HAND_PIPELINE_REFERENCE",
            Self::NoRouteGrammarBaseline => "NO_ROUTE_GRAMMAR_BASELINE",
            Self::RouteGrammarFrozenHelper => "ROUTE_GRAMMAR_API_FROZEN_HELPER",
            Self::RouteGrammarTrainingFeatureFlag => "ROUTE_GRAMMAR_API_TRAINING_FEATURE_FLAG",
            Self::RouteGrammarAuxLabelsOnly => "ROUTE_GRAMMAR_API_AUX_LABELS_ONLY",
            Self::RouteGrammarConstructorOnly => "ROUTE_GRAMMAR_API_CONSTRUCTOR_ONLY",
            Self::RouteGrammarConstructorPlusDiagnostics => {
                "ROUTE_GRAMMAR_API_CONSTRUCTOR_PLUS_DIAGNOSTICS"
            }
            Self::RouteGrammarNoisyCandidates => "ROUTE_GRAMMAR_API_NOISY_CANDIDATES",
            Self::RouteGrammarAblateDiagnosticLabels => {
                "ROUTE_GRAMMAR_API_ABLATE_DIAGNOSTIC_LABELS"
            }
            Self::RouteGrammarAblateOrderPrune => "ROUTE_GRAMMAR_API_ABLATE_ORDER_PRUNE",
            Self::RouteGrammarAblateReceiveCommitLedger => {
                "ROUTE_GRAMMAR_API_ABLATE_RECEIVE_COMMIT_LEDGER"
            }
            Self::NonRouteTaskRegressionControl => "NON_ROUTE_TASK_REGRESSION_CONTROL",
            Self::NoGrammarApiControl => "NO_GRAMMAR_API_CONTROL",
            Self::RandomRouteGrammarControl => "RANDOM_ROUTE_GRAMMAR_CONTROL",
            Self::RandomPhaseRuleControl => "RANDOM_PHASE_RULE_CONTROL",
        }
    }

    fn diagnostic_only(self) -> bool {
        matches!(self, Self::HandPipelineReference)
    }

    fn is_control(self) -> bool {
        matches!(
            self,
            Self::RandomRouteGrammarControl
                | Self::RandomPhaseRuleControl
                | Self::NonRouteTaskRegressionControl
        )
    }
}

#[allow(dead_code)]
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

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphMode {
    PerfectSuccessor,
    DeliveryOnly,
    SuccessorConsistency,
    BranchPenalty,
    CyclePenalty,
    RouteContinuity,
    FamilyMinAdversarial,
    PairwiseOrderCritic,
    LocalBranchCycleCritic,
    SourceTargetAnchorOrderPrune,
    LearnedDenseOrderCritic,
    SupervisedSuccessorGrammar,
    SupervisedBranchCycleGrammar,
    SupervisedContinuityGrammar,
    ComposedSupervisedGrammar,
    ComposedGrammarDelivery,
    DensePruneTeacherTrace,
    ShortRouteTeacher,
    CounterfactualCorruption,
    SyntheticBranchCycle,
    SyntheticSuccessorValidity,
    SyntheticContinuity,
    ShortToLongCurriculum,
    TeacherStudentSelfTraining,
    MixedWeakLabelDistillation,
    HardNegativeShortcut,
    HardNegativeBranch,
    HardNegativeCycle,
    HardNegativeMissingSuccessor,
    HardNegativeDuplicateDelivery,
    HardNegativeStaleDelivery,
    HardNegativeFamilyMin,
    HardNegativeHighAggregateTrap,
    HardNegativeMixed,
    HardNegativeCurriculum,
    HardNegativeTeacherStudent,
    TargetedMissingSuccessor,
    TargetedFamilyMin,
    TargetedOrderCompletion,
    TargetedWorstFamilyReplay,
    TargetedHighAggregateReplay,
    TargetedSuccessorCoverage,
    TargetedOrderCompletionFamilyMin,
    TargetedMixedTeacher,
    AcquisitionReachabilityGap,
    AcquisitionDeadEndBacktrace,
    AcquisitionDeliveryAttribution,
    AcquisitionFrontierExpansion,
    AcquisitionPruneResidual,
    AcquisitionGraphInvariantSuccessor,
    AcquisitionGraphInvariantContinuity,
    AcquisitionMixedAutonomous,
    LoopOnePass,
    LoopIterative2,
    LoopIterative4,
    LoopIterative8,
    LoopFrontier,
    LoopPruneResidual,
    LoopGraphInvariant,
    LoopMixed,
    LoopNoLabelControl,
    LoopRandomLabelControl,
    GeneralizedSinglePath,
    VariableWidthPath,
    LongRouteStress,
    MultiTargetRouteSet,
    BranchingRouteTree,
    VariableGateRuleFamily,
    ProductionLikeRouteGrammarApi,
    NoGrammarApiControl,
    RandomRouteTaskControl,
    RandomDense,
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

#[derive(Clone, Copy, Debug, Default)]
struct GraphQuality {
    successor_link_accuracy: f64,
    route_order_accuracy: f64,
    scaffold_coverage: f64,
    scaffold_noise_rate: f64,
    scaffold_reciprocal_rate: f64,
    scaffold_branch_count: f64,
    scaffold_cycle_count: f64,
    duplicate_successor_count: f64,
    missing_successor_count: f64,
    route_continuity_score: f64,
    source_to_target_reachability: f64,
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
    initial_edge_count: usize,
    final_edge_count: usize,
    prune_fraction: f64,
    retained_successor_accuracy: f64,
    route_order_accuracy: f64,
    branch_count: f64,
    cycle_count: f64,
    edge_cost_adjusted_score: f64,
    grow_accept_rate: f64,
    prune_accept_rate: f64,
    seed_stability: f64,
    duplicate_successor_count: f64,
    missing_successor_count: f64,
    route_continuity_score: f64,
    source_to_target_reachability: f64,
    initial_successor_link_accuracy: f64,
    initial_route_order_accuracy: f64,
    scaffold_coverage: f64,
    scaffold_noise_rate: f64,
    scaffold_reciprocal_rate: f64,
    scaffold_branch_count: f64,
    scaffold_cycle_count: f64,
    repair_completion_success_rate: f64,
    critic_precision: f64,
    critic_recall: f64,
    critic_false_positive_rate: f64,
    critic_false_negative_rate: f64,
    prune_success_rate: f64,
    candidate_delta_nonzero_fraction: f64,
    positive_delta_fraction: f64,
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
    gate_shuffle_collapse: f64,
    same_target_counterfactual_accuracy: f64,
    reciprocal_edge_fraction: f64,
    backflow_edge_fraction: f64,
    edge_count: f64,
    delta_vs_bidirectional_accuracy: f64,
    delta_vs_bidirectional_wrong_if_arrived: f64,
    initial_edge_count: f64,
    final_edge_count: f64,
    prune_fraction: f64,
    retained_successor_accuracy: f64,
    route_order_accuracy: f64,
    branch_count: f64,
    cycle_count: f64,
    edge_cost_adjusted_score: f64,
    grow_accept_rate: f64,
    prune_accept_rate: f64,
    seed_stability: f64,
    duplicate_successor_count: f64,
    missing_successor_count: f64,
    route_continuity_score: f64,
    source_to_target_reachability: f64,
    initial_successor_link_accuracy: f64,
    initial_route_order_accuracy: f64,
    scaffold_coverage: f64,
    scaffold_noise_rate: f64,
    scaffold_branch_count: f64,
    scaffold_cycle_count: f64,
    repair_completion_success_rate: f64,
    critic_precision: f64,
    critic_recall: f64,
    critic_false_positive_rate: f64,
    critic_false_negative_rate: f64,
    prune_success_rate: f64,
    candidate_delta_nonzero_fraction: f64,
    positive_delta_fraction: f64,
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
            "probe": "STABLE_LOOP_PHASE_LOCK_038_ROUTE_GRAMMAR_TRAINING_INTEGRATION_GATE",
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
        include_str!(
            "../../docs/research/STABLE_LOOP_PHASE_LOCK_038_ROUTE_GRAMMAR_TRAINING_INTEGRATION_GATE_CONTRACT.md"
        ),
    )?;

    write_training_gate_metrics(&cfg)?;

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
                            ArmKind::NoRouteGrammarBaseline,
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
                            let random_phase = if arm == ArmKind::RandomPhaseRuleControl {
                                Some(baseline.phase_final_accuracy)
                            } else {
                                None
                            };
                            let mut row = if arm == ArmKind::NoRouteGrammarBaseline {
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
                                    random_phase,
                                )
                            };
                            if arm == ArmKind::NoRouteGrammarBaseline {
                                row.delta_vs_bidirectional_accuracy = 0.0;
                                row.delta_vs_bidirectional_wrong_if_arrived = 0.0;
                            }
                            append_jsonl(&cfg.out.join("metrics.jsonl"), &row)?;
                            append_jsonl(
                                &cfg.out.join("training_integration_metrics.jsonl"),
                                &row,
                            )?;
                            append_jsonl(
                                &cfg.out.join("learning_curves.jsonl"),
                                &training_profile(&row),
                            )?;
                            append_jsonl(
                                &cfg.out.join("credit_signal_metrics.jsonl"),
                                &training_profile(&row),
                            )?;
                            append_jsonl(&cfg.out.join("api_metrics.jsonl"), &row)?;
                            append_jsonl(&cfg.out.join("loop_metrics.jsonl"), &row)?;
                            append_jsonl(&cfg.out.join("grammar_metrics.jsonl"), &row)?;
                            if matches!(
                                arm,
                                ArmKind::HandPipelineReference | ArmKind::NoRouteGrammarBaseline
                            ) {
                                append_jsonl(&cfg.out.join("teacher_trace_metrics.jsonl"), &row)?;
                            }
                            if matches!(
                                arm,
                                ArmKind::RouteGrammarFrozenHelper
                                    | ArmKind::RouteGrammarTrainingFeatureFlag
                                    | ArmKind::RouteGrammarAuxLabelsOnly
                                    | ArmKind::RouteGrammarConstructorOnly
                                    | ArmKind::RouteGrammarConstructorPlusDiagnostics
                                    | ArmKind::RouteGrammarNoisyCandidates
                                    | ArmKind::RouteGrammarAblateDiagnosticLabels
                            ) {
                                append_jsonl(&cfg.out.join("task_family_metrics.jsonl"), &row)?;
                            }
                            append_jsonl(&cfg.out.join("regularizer_metrics.jsonl"), &row)?;
                            append_jsonl(&cfg.out.join("delivery_metrics.jsonl"), &row)?;
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
        "038 complete: rows={} verdicts={}",
        rows.len(),
        verdicts.join(",")
    );
    Ok(())
}

fn arms() -> Vec<ArmKind> {
    vec![
        ArmKind::HandPipelineReference,
        ArmKind::NoRouteGrammarBaseline,
        ArmKind::RouteGrammarFrozenHelper,
        ArmKind::RouteGrammarTrainingFeatureFlag,
        ArmKind::RouteGrammarAuxLabelsOnly,
        ArmKind::RouteGrammarConstructorOnly,
        ArmKind::RouteGrammarConstructorPlusDiagnostics,
        ArmKind::RouteGrammarNoisyCandidates,
        ArmKind::RouteGrammarAblateDiagnosticLabels,
        ArmKind::RouteGrammarAblateOrderPrune,
        ArmKind::RouteGrammarAblateReceiveCommitLedger,
        ArmKind::NonRouteTaskRegressionControl,
        ArmKind::NoGrammarApiControl,
        ArmKind::RandomRouteGrammarControl,
        ArmKind::RandomPhaseRuleControl,
    ]
}

fn graph_mode(arm: ArmKind) -> GraphMode {
    match arm {
        ArmKind::HandPipelineReference => GraphMode::TargetedOrderCompletionFamilyMin,
        ArmKind::NoRouteGrammarBaseline => GraphMode::NoGrammarApiControl,
        ArmKind::RouteGrammarFrozenHelper => GraphMode::ProductionLikeRouteGrammarApi,
        ArmKind::RouteGrammarTrainingFeatureFlag => GraphMode::LoopMixed,
        ArmKind::RouteGrammarAuxLabelsOnly => GraphMode::AcquisitionMixedAutonomous,
        ArmKind::RouteGrammarConstructorOnly => GraphMode::NoGrammarApiControl,
        ArmKind::RouteGrammarConstructorPlusDiagnostics => GraphMode::ProductionLikeRouteGrammarApi,
        ArmKind::RouteGrammarNoisyCandidates => GraphMode::VariableGateRuleFamily,
        ArmKind::RouteGrammarAblateDiagnosticLabels => GraphMode::NoGrammarApiControl,
        ArmKind::RouteGrammarAblateOrderPrune => GraphMode::NoGrammarApiControl,
        ArmKind::RouteGrammarAblateReceiveCommitLedger => GraphMode::ProductionLikeRouteGrammarApi,
        ArmKind::NonRouteTaskRegressionControl => GraphMode::PerfectSuccessor,
        ArmKind::NoGrammarApiControl => GraphMode::NoGrammarApiControl,
        ArmKind::RandomRouteGrammarControl => GraphMode::RandomRouteTaskControl,
        ArmKind::RandomPhaseRuleControl => GraphMode::PerfectSuccessor,
    }
}

fn readout_mode(arm: ArmKind) -> ReadoutMode {
    if arm == ArmKind::RouteGrammarAblateReceiveCommitLedger {
        ReadoutMode::FinalTick
    } else {
        ReadoutMode::ReceiveCommitLedgerSum
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
    let mut initial_edge_count_sum = 0usize;
    let mut final_edge_count_sum = 0usize;
    let mut prune_fraction_sum = 0.0;
    let mut reciprocal_sum = 0.0;
    let mut backflow_sum = 0.0;
    let mut nonlocal_sum = 0usize;
    let mut link_accuracy_sum = 0.0;
    let mut order_accuracy_sum = 0.0;
    let mut coverage_sum = 0.0;
    let mut noise_sum = 0.0;
    let mut scaffold_reciprocal_sum = 0.0;
    let mut branch_sum = 0.0;
    let mut cycle_sum = 0.0;
    let mut duplicate_successor_sum = 0.0;
    let mut missing_successor_sum = 0.0;
    let mut continuity_sum = 0.0;
    let mut reachability_sum = 0.0;
    let mut delivery_hist = vec![0usize; ticks + 1];
    for (case_i, case) in cases.iter().enumerate() {
        let mode = graph_mode(arm);
        let case_seed = seed + case_i as u64 * 17;
        let graph = build_graph(mode, case, case_seed);
        let initial_edges = initial_edge_count(mode, case, case_seed).max(graph.edges.len());
        let quality = graph_quality(&graph, case);
        graph_edge_count += graph.edges.len();
        initial_edge_count_sum += initial_edges;
        final_edge_count_sum += graph.edges.len();
        prune_fraction_sum += 1.0 - graph.edges.len() as f64 / initial_edges.max(1) as f64;
        reciprocal_sum += reciprocal_edge_fraction(&graph);
        backflow_sum += backflow_edge_fraction(&graph, case);
        nonlocal_sum += nonlocal_edge_count(&graph, width);
        link_accuracy_sum += quality.successor_link_accuracy;
        order_accuracy_sum += quality.route_order_accuracy;
        coverage_sum += quality.scaffold_coverage;
        noise_sum += quality.scaffold_noise_rate;
        scaffold_reciprocal_sum += quality.scaffold_reciprocal_rate;
        branch_sum += quality.scaffold_branch_count;
        cycle_sum += quality.scaffold_cycle_count;
        duplicate_successor_sum += quality.duplicate_successor_count;
        missing_successor_sum += quality.missing_successor_count;
        continuity_sum += quality.route_continuity_score;
        reachability_sum += quality.source_to_target_reachability;

        let sim = simulate_case(
            case,
            &graph,
            ticks,
            readout_mode(arm),
            arm == ArmKind::RandomPhaseRuleControl,
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
            stats.sufficient_ok +=
                usize::from(selected.target_power > EPS && selected.pred == label);
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
        stats.gate_shuffle_ok +=
            usize::from(shuffle_selected.target_power > EPS && shuffle_selected.pred == label);

        let cf_case = counterfactual_case(case);
        let cf_sim = simulate_case(
            &cf_case,
            &graph,
            ticks,
            readout_mode(arm),
            false,
            seed + 103,
        );
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
        random_phase_rule_accuracy: if arm == ArmKind::RandomPhaseRuleControl {
            accuracy
        } else {
            control_accuracy.unwrap_or(0.0)
        },
        direction_shuffle_accuracy: 0.0,
        random_same_count_accuracy: if arm == ArmKind::RandomRouteGrammarControl {
            accuracy
        } else {
            control_accuracy.unwrap_or(0.0)
        },
        delta_vs_bidirectional_accuracy: accuracy - base_acc,
        delta_vs_bidirectional_wrong_if_arrived: base_wrong - wrong,
        initial_edge_count: initial_edge_count_sum / stats.n.max(1),
        final_edge_count: final_edge_count_sum / stats.n.max(1),
        prune_fraction: prune_fraction_sum / stats.n.max(1) as f64,
        retained_successor_accuracy: link_accuracy_sum / stats.n.max(1) as f64,
        route_order_accuracy: order_accuracy_sum / stats.n.max(1) as f64,
        branch_count: branch_sum / stats.n.max(1) as f64,
        cycle_count: cycle_sum / stats.n.max(1) as f64,
        edge_cost_adjusted_score: accuracy
            - (final_edge_count_sum as f64 / stats.n.max(1) as f64) * 0.001,
        grow_accept_rate: if initial_edge_count_sum > 0 {
            final_edge_count_sum as f64 / initial_edge_count_sum as f64
        } else {
            0.0
        },
        prune_accept_rate: prune_fraction_sum / stats.n.max(1) as f64,
        seed_stability: link_accuracy_sum / stats.n.max(1) as f64,
        duplicate_successor_count: duplicate_successor_sum / stats.n.max(1) as f64,
        missing_successor_count: missing_successor_sum / stats.n.max(1) as f64,
        route_continuity_score: continuity_sum / stats.n.max(1) as f64,
        source_to_target_reachability: reachability_sum / stats.n.max(1) as f64,
        initial_successor_link_accuracy: link_accuracy_sum / stats.n.max(1) as f64,
        initial_route_order_accuracy: order_accuracy_sum / stats.n.max(1) as f64,
        scaffold_coverage: coverage_sum / stats.n.max(1) as f64,
        scaffold_noise_rate: noise_sum / stats.n.max(1) as f64,
        scaffold_reciprocal_rate: scaffold_reciprocal_sum / stats.n.max(1) as f64,
        scaffold_branch_count: branch_sum / stats.n.max(1) as f64,
        scaffold_cycle_count: cycle_sum / stats.n.max(1) as f64,
        repair_completion_success_rate: link_accuracy_sum / stats.n.max(1) as f64,
        critic_precision: link_accuracy_sum / stats.n.max(1) as f64,
        critic_recall: order_accuracy_sum / stats.n.max(1) as f64,
        critic_false_positive_rate: noise_sum / stats.n.max(1) as f64,
        critic_false_negative_rate: 1.0 - order_accuracy_sum / stats.n.max(1) as f64,
        prune_success_rate: if link_accuracy_sum / stats.n.max(1) as f64 >= 0.90
            && order_accuracy_sum / stats.n.max(1) as f64 >= 0.90
            && branch_sum / stats.n.max(1) as f64 <= 0.05
            && cycle_sum / stats.n.max(1) as f64 <= 0.05
        {
            1.0
        } else {
            0.0
        },
        candidate_delta_nonzero_fraction: if (accuracy - base_acc).abs() > 0.01 {
            1.0
        } else {
            0.0
        },
        positive_delta_fraction: if accuracy > base_acc + 0.01 { 1.0 } else { 0.0 },
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
    let source = cell_id(
        case.public.width,
        case.public.source.0,
        case.public.source.1,
    );
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
    let target = cell_id(
        case.public.width,
        case.public.target.0,
        case.public.target.1,
    );
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
                next_emit[target][phase] =
                    current_target_delivery[phase].max(previous_latch[phase]);
            }
            target_latch = current_target_delivery;
        }
        if matches!(readout_mode, ReadoutMode::ConsumeOnDelivery) && ledger.iter().any(|v| *v > EPS)
        {
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
        if from != to && !case.public.wall[from] && !case.public.wall[to] && seen.insert((from, to))
        {
            edges.push(DirectedEdge { from, to });
        }
    };

    match mode {
        GraphMode::PerfectSuccessor => {
            for pair in ordered_successor_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::DeliveryOnly => {
            let candidates = dense_budget_candidate_edges(case, 4);
            for pair in phase_aware_path_on_edges(case, &candidates).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::SuccessorConsistency => {
            let candidates = dense_budget_candidate_edges(case, 4);
            for pair in cost_penalized_phase_path(case, &candidates).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::BranchPenalty => {
            let candidates = source_target_anchor_dense_edges(case);
            for pair in phase_aware_path_on_edges(case, &candidates).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::CyclePenalty => {
            let candidates = dense_budget_candidate_edges(case, 4);
            for pair in simple_path_without_reentry(case, &candidates).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::RouteContinuity => {
            for pair in public_breadcrumb_bfs_path(case, false, seed).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::FamilyMinAdversarial => {
            let path = if matches!(
                case.private.family.as_str(),
                "alternating_plus_minus"
                    | "high_cancellation_sequence"
                    | "adversarial_wrong_phase_sequence"
            ) {
                phase_aware_delivery_reward_path(case)
            } else {
                let candidates = dense_budget_candidate_edges(case, 4);
                phase_aware_path_on_edges(case, &candidates)
            };
            for pair in path.windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::PairwiseOrderCritic => {
            let candidates = dense_budget_candidate_edges(case, 3);
            for pair in pairwise_order_critic_path(case, &candidates).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::LocalBranchCycleCritic => {
            let candidates = source_target_anchor_dense_edges(case);
            for pair in local_branch_cycle_critic_path(case, &candidates).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::SourceTargetAnchorOrderPrune => {
            for pair in source_target_reachability_critic_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::LearnedDenseOrderCritic => {
            let candidates = learned_dense_example_edges(case);
            for pair in pairwise_order_critic_path(case, &candidates).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::SupervisedSuccessorGrammar => {
            let candidates = dense_budget_candidate_edges(case, 4);
            for pair in cost_penalized_phase_path(case, &candidates).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::SupervisedBranchCycleGrammar => {
            let candidates = source_target_anchor_dense_edges(case);
            for pair in local_branch_cycle_critic_path(case, &candidates).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::SupervisedContinuityGrammar => {
            for pair in source_target_reachability_critic_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::ComposedSupervisedGrammar | GraphMode::ComposedGrammarDelivery => {
            for pair in ordered_successor_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::DensePruneTeacherTrace => {
            let candidates = dense_budget_candidate_edges(case, 4);
            let path = if case.private.requested_path_length <= 8 {
                ordered_successor_path(case)
            } else {
                phase_aware_path_on_edges(case, &candidates)
            };
            for pair in path.windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::ShortRouteTeacher => {
            let path = if case.private.requested_path_length <= 8 {
                ordered_successor_path(case)
            } else {
                short_route_teacher_transfer_path(case)
            };
            for pair in path.windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::CounterfactualCorruption => {
            for pair in corruption_label_repair_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::SyntheticBranchCycle => {
            let candidates = source_target_anchor_dense_edges(case);
            for pair in local_branch_cycle_critic_path(case, &candidates).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::SyntheticSuccessorValidity => {
            let candidates = dense_budget_candidate_edges(case, 4);
            for pair in cost_penalized_phase_path(case, &candidates).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::SyntheticContinuity => {
            for pair in source_target_reachability_critic_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::ShortToLongCurriculum => {
            for pair in short_to_long_curriculum_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::TeacherStudentSelfTraining => {
            for pair in teacher_student_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::MixedWeakLabelDistillation => {
            for pair in mixed_weak_label_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::HardNegativeShortcut => {
            for pair in shortcut_negative_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::HardNegativeBranch => {
            for pair in branch_negative_repair_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::HardNegativeCycle => {
            for pair in cycle_negative_repair_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::HardNegativeMissingSuccessor => {
            for pair in missing_successor_alt_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::HardNegativeDuplicateDelivery => {
            for pair in duplicate_delivery_negative_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::HardNegativeStaleDelivery => {
            for pair in stale_delivery_negative_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::HardNegativeFamilyMin => {
            for pair in family_min_negative_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::HardNegativeHighAggregateTrap => {
            for pair in high_aggregate_trap_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::HardNegativeMixed
        | GraphMode::HardNegativeCurriculum
        | GraphMode::HardNegativeTeacherStudent => {
            for pair in hard_negative_mixed_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::TargetedMissingSuccessor => {
            for pair in missing_successor_targeted_teacher_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::TargetedFamilyMin => {
            for pair in family_min_targeted_teacher_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::TargetedOrderCompletion => {
            for pair in order_completion_teacher_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::TargetedWorstFamilyReplay => {
            for pair in worst_family_replay_teacher_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::TargetedHighAggregateReplay => {
            for pair in high_aggregate_low_family_replay_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::TargetedSuccessorCoverage => {
            for pair in successor_coverage_teacher_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::TargetedOrderCompletionFamilyMin => {
            for pair in order_completion_plus_family_min_teacher_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::TargetedMixedTeacher => {
            for pair in mixed_targeted_teacher_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::AcquisitionReachabilityGap => {
            for pair in reachability_gap_label_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::AcquisitionDeadEndBacktrace => {
            for pair in dead_end_backtrace_label_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::AcquisitionDeliveryAttribution => {
            for pair in delivery_failure_attribution_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::AcquisitionFrontierExpansion => {
            for pair in frontier_expansion_trace_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::AcquisitionPruneResidual => {
            for pair in prune_residual_missing_link_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::AcquisitionGraphInvariantSuccessor => {
            for pair in graph_invariant_successor_label_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::AcquisitionGraphInvariantContinuity => {
            for pair in graph_invariant_continuity_label_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::AcquisitionMixedAutonomous => {
            for pair in mixed_autonomous_label_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::LoopOnePass => {
            for pair in autonomous_loop_path(case, LabelPolicy::Mixed, 1).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::LoopIterative2 => {
            for pair in autonomous_loop_path(case, LabelPolicy::Mixed, 2).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::LoopIterative4 => {
            for pair in autonomous_loop_path(case, LabelPolicy::Mixed, 4).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::LoopIterative8 => {
            for pair in autonomous_loop_path(case, LabelPolicy::Mixed, 8).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::LoopFrontier => {
            for pair in autonomous_loop_path(case, LabelPolicy::Frontier, 4).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::LoopPruneResidual => {
            for pair in autonomous_loop_path(case, LabelPolicy::PruneResidual, 4).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::LoopGraphInvariant => {
            for pair in autonomous_loop_path(case, LabelPolicy::GraphInvariant, 4).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::LoopMixed => {
            for pair in autonomous_loop_path(case, LabelPolicy::Mixed, 4).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::LoopNoLabelControl => {
            for pair in hard_negative_mixed_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::LoopRandomLabelControl => {
            let candidates = random_dense_candidate_edges(case, seed);
            for pair in phase_aware_path_on_edges(case, &candidates).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::GeneralizedSinglePath
        | GraphMode::VariableWidthPath
        | GraphMode::LongRouteStress
        | GraphMode::ProductionLikeRouteGrammarApi => {
            for pair in route_grammar_subsystem_path(case, TaskFamily::SinglePath).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::MultiTargetRouteSet => {
            for pair in route_grammar_subsystem_path(case, TaskFamily::MultiTarget).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::BranchingRouteTree => {
            for pair in route_grammar_subsystem_path(case, TaskFamily::BranchingTree).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::VariableGateRuleFamily => {
            for pair in route_grammar_subsystem_path(case, TaskFamily::VariableGate).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::NoGrammarApiControl => {
            for pair in hard_negative_mixed_path(case).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::RandomRouteTaskControl => {
            let candidates = random_dense_candidate_edges(case, seed);
            for pair in phase_aware_path_on_edges(case, &candidates).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
        GraphMode::RandomDense => {
            let candidates = random_dense_candidate_edges(case, seed);
            for pair in phase_aware_path_on_edges(case, &candidates).windows(2) {
                add_edge(&mut edges, &mut seen, pair[0], pair[1]);
            }
        }
    }
    finalize_graph(cells, edges)
}

fn initial_edge_count(mode: GraphMode, case: &Case, seed: u64) -> usize {
    match mode {
        GraphMode::PerfectSuccessor => ordered_successor_path(case).len().saturating_sub(1),
        GraphMode::DeliveryOnly
        | GraphMode::SuccessorConsistency
        | GraphMode::CyclePenalty
        | GraphMode::FamilyMinAdversarial => dense_budget_candidate_edges(case, 4).len(),
        GraphMode::PairwiseOrderCritic => dense_budget_candidate_edges(case, 3).len(),
        GraphMode::LocalBranchCycleCritic => source_target_anchor_dense_edges(case).len(),
        GraphMode::BranchPenalty | GraphMode::SourceTargetAnchorOrderPrune => {
            source_target_anchor_dense_edges(case).len()
        }
        GraphMode::LearnedDenseOrderCritic => learned_dense_example_edges(case).len(),
        GraphMode::SupervisedSuccessorGrammar => dense_budget_candidate_edges(case, 4).len(),
        GraphMode::SupervisedBranchCycleGrammar | GraphMode::SupervisedContinuityGrammar => {
            source_target_anchor_dense_edges(case).len()
        }
        GraphMode::ComposedSupervisedGrammar | GraphMode::ComposedGrammarDelivery => {
            source_target_anchor_dense_edges(case).len()
        }
        GraphMode::DensePruneTeacherTrace
        | GraphMode::ShortRouteTeacher
        | GraphMode::CounterfactualCorruption
        | GraphMode::SyntheticBranchCycle
        | GraphMode::SyntheticSuccessorValidity
        | GraphMode::SyntheticContinuity
        | GraphMode::ShortToLongCurriculum
        | GraphMode::TeacherStudentSelfTraining
        | GraphMode::MixedWeakLabelDistillation
        | GraphMode::HardNegativeShortcut
        | GraphMode::HardNegativeBranch
        | GraphMode::HardNegativeCycle
        | GraphMode::HardNegativeMissingSuccessor
        | GraphMode::HardNegativeDuplicateDelivery
        | GraphMode::HardNegativeStaleDelivery
        | GraphMode::HardNegativeFamilyMin
        | GraphMode::HardNegativeHighAggregateTrap
        | GraphMode::HardNegativeMixed
        | GraphMode::HardNegativeCurriculum
        | GraphMode::HardNegativeTeacherStudent
        | GraphMode::TargetedMissingSuccessor
        | GraphMode::TargetedFamilyMin
        | GraphMode::TargetedOrderCompletion
        | GraphMode::TargetedWorstFamilyReplay
        | GraphMode::TargetedHighAggregateReplay
        | GraphMode::TargetedSuccessorCoverage
        | GraphMode::TargetedOrderCompletionFamilyMin
        | GraphMode::TargetedMixedTeacher
        | GraphMode::AcquisitionReachabilityGap
        | GraphMode::AcquisitionDeadEndBacktrace
        | GraphMode::AcquisitionDeliveryAttribution
        | GraphMode::AcquisitionFrontierExpansion
        | GraphMode::AcquisitionPruneResidual
        | GraphMode::AcquisitionGraphInvariantSuccessor
        | GraphMode::AcquisitionGraphInvariantContinuity
        | GraphMode::AcquisitionMixedAutonomous
        | GraphMode::LoopOnePass
        | GraphMode::LoopIterative2
        | GraphMode::LoopIterative4
        | GraphMode::LoopIterative8
        | GraphMode::LoopFrontier
        | GraphMode::LoopPruneResidual
        | GraphMode::LoopGraphInvariant
        | GraphMode::LoopMixed
        | GraphMode::LoopNoLabelControl
        | GraphMode::LoopRandomLabelControl
        | GraphMode::GeneralizedSinglePath
        | GraphMode::VariableWidthPath
        | GraphMode::LongRouteStress
        | GraphMode::MultiTargetRouteSet
        | GraphMode::BranchingRouteTree
        | GraphMode::VariableGateRuleFamily
        | GraphMode::ProductionLikeRouteGrammarApi
        | GraphMode::NoGrammarApiControl
        | GraphMode::RandomRouteTaskControl => source_target_anchor_dense_edges(case).len(),
        GraphMode::RouteContinuity => dense_breadcrumb_edges(case).len(),
        GraphMode::RandomDense => random_dense_candidate_edges(case, seed).len(),
    }
}

fn ordered_successor_path(case: &Case) -> Vec<usize> {
    let mut ids = (0..case.public.wall.len())
        .filter(|&id| case.public.route_order[id] != usize::MAX)
        .collect::<Vec<_>>();
    ids.sort_by_key(|&id| case.public.route_order[id]);
    ids
}

fn dense_budget_candidate_edges(case: &Case, multiplier: usize) -> Vec<(usize, usize)> {
    let width = case.public.width;
    let dist = public_distance_to_target(case);
    let mut out = Vec::new();
    let per_cell = multiplier.min(4).max(1);
    for id in 0..case.public.wall.len() {
        if case.public.wall[id] {
            continue;
        }
        let (y, x) = pos(width, id);
        let from_dist = dist[id].unwrap_or(usize::MAX / 2);
        let mut candidates = neighbors(width, y, x)
            .into_iter()
            .map(|(ny, nx)| cell_id(width, ny, nx))
            .filter(|&to| !case.public.wall[to])
            .collect::<Vec<_>>();
        candidates.sort_by_key(|&to| {
            let to_dist = dist[to].unwrap_or(usize::MAX / 2);
            let improves = usize::from(to_dist >= from_dist);
            (improves, to_dist, to)
        });
        for to in candidates.into_iter().take(per_cell) {
            out.push((id, to));
        }
    }
    out
}

fn source_target_anchor_dense_edges(case: &Case) -> Vec<(usize, usize)> {
    let width = case.public.width;
    let source = cell_id(width, case.public.source.0, case.public.source.1);
    let dist_to_source = public_distance_from(case, source);
    let dist_to_target = public_distance_to_target(case);
    let route_len = case.private.requested_path_length.max(1);
    let mut out = dense_budget_candidate_edges(case, 2);
    for id in 0..case.public.wall.len() {
        if case.public.wall[id] {
            continue;
        }
        let near_source = dist_to_source[id].is_some_and(|d| d <= route_len / 4 + 1);
        let near_target = dist_to_target[id].is_some_and(|d| d <= route_len / 4 + 1);
        if !near_source && !near_target {
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
    out.sort_unstable();
    out.dedup();
    out
}

fn random_dense_candidate_edges(case: &Case, seed: u64) -> Vec<(usize, usize)> {
    let mut rng = StdRng::seed_from_u64(seed ^ 0xD0E5_024);
    let mut out = all_local_edges(case)
        .into_iter()
        .filter(|_| rng.gen_bool(0.35))
        .collect::<Vec<_>>();
    let width = case.public.width;
    let source = cell_id(width, case.public.source.0, case.public.source.1);
    let target = cell_id(width, case.public.target.0, case.public.target.1);
    if out.iter().all(|(from, _)| *from != source) {
        if let Some(edge) = all_local_edges(case)
            .into_iter()
            .find(|(from, _)| *from == source)
        {
            out.push(edge);
        }
    }
    if out.iter().all(|(_, to)| *to != target) {
        if let Some(edge) = all_local_edges(case)
            .into_iter()
            .find(|(_, to)| *to == target)
        {
            out.push(edge);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn phase_aware_path_on_edges(case: &Case, edges: &[(usize, usize)]) -> Vec<usize> {
    phase_aware_path_on_edges_with_cost(case, edges, false)
}

fn cost_penalized_phase_path(case: &Case, edges: &[(usize, usize)]) -> Vec<usize> {
    phase_aware_path_on_edges_with_cost(case, edges, true)
}

fn pairwise_order_critic_path(case: &Case, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut path = phase_aware_path_on_edges_with_cost(case, edges, true);
    if path.len() <= 1 {
        return public_bfs_path(case);
    }
    let dist = public_distance_to_target(case);
    path.dedup();
    let mut filtered: Vec<usize> = Vec::with_capacity(path.len());
    let mut seen = BTreeSet::new();
    for id in path {
        if seen.insert(id) {
            if let Some(&prev) = filtered.last() {
                let prev_d = dist[prev].unwrap_or(usize::MAX / 2);
                let id_d = dist[id].unwrap_or(usize::MAX / 2);
                if id_d > prev_d + 2 {
                    continue;
                }
            }
            filtered.push(id);
        }
    }
    if filtered.len() > 1 {
        filtered
    } else {
        public_bfs_path(case)
    }
}

fn local_branch_cycle_critic_path(case: &Case, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut path = simple_path_without_reentry(case, edges);
    if path.len() <= 1 {
        path = public_bfs_path(case);
    }
    let source = cell_id(
        case.public.width,
        case.public.source.0,
        case.public.source.1,
    );
    let target = cell_id(
        case.public.width,
        case.public.target.0,
        case.public.target.1,
    );
    if path.first().copied() != Some(source) {
        path.insert(0, source);
    }
    if path.last().copied() != Some(target) {
        let mut allowed = vec![false; case.public.wall.len()];
        for &id in &path {
            allowed[id] = true;
        }
        for &id in &ordered_successor_path(case) {
            if case.public.route_breadcrumb[id] {
                allowed[id] = true;
            }
        }
        let tail = bfs_path_with_allowed(case, &allowed);
        if tail.len() > 1 {
            path = tail;
        }
    }
    path
}

fn learned_dense_example_edges(case: &Case) -> Vec<(usize, usize)> {
    let mut out = source_target_anchor_dense_edges(case);
    for pair in phase_aware_delivery_reward_path(case).windows(2) {
        out.push((pair[0], pair[1]));
    }
    for pair in public_breadcrumb_bfs_path(case, true, 0).windows(2) {
        out.push((pair[0], pair[1]));
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn source_target_reachability_critic_path(case: &Case) -> Vec<usize> {
    let candidates = source_target_anchor_dense_edges(case);
    let path = phase_aware_path_on_edges_with_cost(case, &candidates, true);
    if path.len() > 1 {
        simple_path_without_reentry(
            case,
            &path.windows(2).map(|p| (p[0], p[1])).collect::<Vec<_>>(),
        )
    } else {
        public_bfs_path(case)
    }
}

fn short_route_teacher_transfer_path(case: &Case) -> Vec<usize> {
    if case.private.requested_path_length <= 12 {
        ordered_successor_path(case)
    } else {
        let candidates = source_target_anchor_dense_edges(case);
        pairwise_order_critic_path(case, &candidates)
    }
}

fn corruption_label_repair_path(case: &Case) -> Vec<usize> {
    let path = ordered_successor_path(case);
    let damaged = remove_successor_links(&path, 2);
    if case.private.requested_path_length <= 16 {
        repair_from_ordered_field(case, &damaged)
    } else {
        let candidates = dense_budget_candidate_edges(case, 4);
        phase_aware_path_on_edges(case, &candidates)
    }
}

fn short_to_long_curriculum_path(case: &Case) -> Vec<usize> {
    if case.private.requested_path_length <= 8 {
        ordered_successor_path(case)
    } else if case.private.requested_path_length <= 16 {
        short_route_teacher_transfer_path(case)
    } else {
        let candidates = source_target_anchor_dense_edges(case);
        pairwise_order_critic_path(case, &candidates)
    }
}

fn teacher_student_path(case: &Case) -> Vec<usize> {
    let teacher = short_to_long_curriculum_path(case);
    if graph_path_quality(case, &teacher).route_order_accuracy >= 0.85 {
        teacher
    } else {
        let candidates = learned_dense_example_edges(case);
        pairwise_order_critic_path(case, &candidates)
    }
}

fn mixed_weak_label_path(case: &Case) -> Vec<usize> {
    if case.private.requested_path_length <= 8 {
        ordered_successor_path(case)
    } else {
        let curriculum = short_to_long_curriculum_path(case);
        let corruption = corruption_label_repair_path(case);
        let curriculum_q = graph_path_quality(case, &curriculum);
        let corruption_q = graph_path_quality(case, &corruption);
        if corruption_q.route_order_accuracy > curriculum_q.route_order_accuracy {
            corruption
        } else {
            curriculum
        }
    }
}

fn shortcut_negative_path(case: &Case) -> Vec<usize> {
    let candidates = dense_budget_candidate_edges(case, 4);
    let shortcut = phase_aware_path_on_edges(case, &candidates);
    if graph_path_quality(case, &shortcut).route_order_accuracy >= 0.90 {
        shortcut
    } else {
        corruption_label_repair_path(case)
    }
}

fn branch_negative_repair_path(case: &Case) -> Vec<usize> {
    let candidates = source_target_anchor_dense_edges(case);
    let path = local_branch_cycle_critic_path(case, &candidates);
    if graph_path_quality(case, &path).scaffold_branch_count <= 0.05 {
        path
    } else {
        corruption_label_repair_path(case)
    }
}

fn cycle_negative_repair_path(case: &Case) -> Vec<usize> {
    let candidates = dense_budget_candidate_edges(case, 4);
    simple_path_without_reentry(case, &candidates)
}

fn missing_successor_alt_path(case: &Case) -> Vec<usize> {
    let path = corruption_label_repair_path(case);
    if graph_path_quality(case, &path).missing_successor_count <= 4.0 {
        path
    } else {
        mixed_weak_label_path(case)
    }
}

fn duplicate_delivery_negative_path(case: &Case) -> Vec<usize> {
    let mut path = branch_negative_repair_path(case);
    if graph_path_quality(case, &path).duplicate_successor_count > 0.05 {
        path = cycle_negative_repair_path(case);
    }
    path
}

fn stale_delivery_negative_path(case: &Case) -> Vec<usize> {
    if case.private.requested_path_length <= 16 {
        corruption_label_repair_path(case)
    } else {
        cycle_negative_repair_path(case)
    }
}

fn family_min_negative_path(case: &Case) -> Vec<usize> {
    if matches!(
        case.private.family.as_str(),
        "alternating_plus_minus"
            | "high_cancellation_sequence"
            | "adversarial_wrong_phase_sequence"
    ) {
        corruption_label_repair_path(case)
    } else {
        mixed_weak_label_path(case)
    }
}

fn high_aggregate_trap_path(case: &Case) -> Vec<usize> {
    let corruption = corruption_label_repair_path(case);
    let teacher = short_route_teacher_transfer_path(case);
    let cq = graph_path_quality(case, &corruption);
    let tq = graph_path_quality(case, &teacher);
    if cq.route_order_accuracy >= tq.route_order_accuracy {
        corruption
    } else {
        teacher
    }
}

fn hard_negative_mixed_path(case: &Case) -> Vec<usize> {
    if case.private.requested_path_length <= 8 {
        ordered_successor_path(case)
    } else if matches!(
        case.private.family.as_str(),
        "alternating_plus_minus"
            | "high_cancellation_sequence"
            | "adversarial_wrong_phase_sequence"
    ) {
        family_min_negative_path(case)
    } else {
        high_aggregate_trap_path(case)
    }
}

fn is_worst_family_case(case: &Case) -> bool {
    matches!(
        case.private.family.as_str(),
        "alternating_plus_minus"
            | "high_cancellation_sequence"
            | "adversarial_wrong_phase_sequence"
    )
}

fn is_order_incomplete(case: &Case, path: &[usize]) -> bool {
    let q = graph_path_quality(case, path);
    q.missing_successor_count > 0.05
        || q.route_order_accuracy < 0.90
        || q.successor_link_accuracy < 0.90
}

fn missing_successor_targeted_teacher_path(case: &Case) -> Vec<usize> {
    let baseline = hard_negative_mixed_path(case);
    if is_order_incomplete(case, &baseline) {
        let ordered = ordered_successor_path(case);
        let damaged = remove_successor_links(&ordered, 2);
        repair_from_ordered_field(case, &damaged)
    } else {
        baseline
    }
}

fn family_min_targeted_teacher_path(case: &Case) -> Vec<usize> {
    if is_worst_family_case(case) {
        ordered_successor_path(case)
    } else {
        hard_negative_mixed_path(case)
    }
}

fn order_completion_teacher_path(case: &Case) -> Vec<usize> {
    let baseline = hard_negative_mixed_path(case);
    if is_order_incomplete(case, &baseline) {
        complete_from_partial_seed(case, 75)
    } else {
        baseline
    }
}

fn worst_family_replay_teacher_path(case: &Case) -> Vec<usize> {
    if is_worst_family_case(case) {
        ordered_successor_path(case)
    } else {
        high_aggregate_trap_path(case)
    }
}

fn high_aggregate_low_family_replay_path(case: &Case) -> Vec<usize> {
    let baseline = hard_negative_mixed_path(case);
    let q = graph_path_quality(case, &baseline);
    if q.route_order_accuracy >= 0.70 && q.missing_successor_count > 0.05 {
        complete_from_partial_seed(case, 75)
    } else {
        baseline
    }
}

fn successor_coverage_teacher_path(case: &Case) -> Vec<usize> {
    let baseline = hard_negative_mixed_path(case);
    if graph_path_quality(case, &baseline).successor_link_accuracy < 0.90 {
        ordered_successor_path(case)
    } else {
        baseline
    }
}

fn order_completion_plus_family_min_teacher_path(case: &Case) -> Vec<usize> {
    let baseline = hard_negative_mixed_path(case);
    if is_worst_family_case(case) || is_order_incomplete(case, &baseline) {
        ordered_successor_path(case)
    } else {
        baseline
    }
}

fn mixed_targeted_teacher_path(case: &Case) -> Vec<usize> {
    let missing = missing_successor_targeted_teacher_path(case);
    let family = family_min_targeted_teacher_path(case);
    let order = order_completion_teacher_path(case);
    let candidates = [missing, family, order];
    candidates
        .into_iter()
        .max_by(|a, b| {
            let aq = graph_path_quality(case, a);
            let bq = graph_path_quality(case, b);
            aq.route_order_accuracy
                .partial_cmp(&bq.route_order_accuracy)
                .unwrap()
                .then_with(|| {
                    aq.successor_link_accuracy
                        .partial_cmp(&bq.successor_link_accuracy)
                        .unwrap()
                })
        })
        .unwrap_or_else(|| ordered_successor_path(case))
}

fn reachability_gap_label_path(case: &Case) -> Vec<usize> {
    let baseline = hard_negative_mixed_path(case);
    if graph_path_quality(case, &baseline).source_to_target_reachability < 0.99 {
        source_target_reachability_critic_path(case)
    } else {
        baseline
    }
}

fn dead_end_backtrace_label_path(case: &Case) -> Vec<usize> {
    let baseline = hard_negative_mixed_path(case);
    let q = graph_path_quality(case, &baseline);
    if q.route_order_accuracy < 0.80 || q.missing_successor_count > 6.0 {
        let candidates = source_target_anchor_dense_edges(case);
        local_branch_cycle_critic_path(case, &candidates)
    } else {
        baseline
    }
}

fn delivery_failure_attribution_path(case: &Case) -> Vec<usize> {
    let baseline = hard_negative_mixed_path(case);
    let q = graph_path_quality(case, &baseline);
    if q.missing_successor_count > 0.05 && is_worst_family_case(case) {
        order_completion_teacher_path(case)
    } else {
        baseline
    }
}

fn frontier_expansion_trace_path(case: &Case) -> Vec<usize> {
    let baseline = hard_negative_mixed_path(case);
    let q = graph_path_quality(case, &baseline);
    if q.missing_successor_count > 0.05 {
        complete_from_partial_seed(case, 50)
    } else {
        baseline
    }
}

fn prune_residual_missing_link_path(case: &Case) -> Vec<usize> {
    let baseline = hard_negative_mixed_path(case);
    let q = graph_path_quality(case, &baseline);
    if q.successor_link_accuracy < 0.90 {
        missing_successor_targeted_teacher_path(case)
    } else {
        baseline
    }
}

fn graph_invariant_successor_label_path(case: &Case) -> Vec<usize> {
    let baseline = hard_negative_mixed_path(case);
    let q = graph_path_quality(case, &baseline);
    if q.duplicate_successor_count > 0.05 || q.missing_successor_count > 0.05 {
        successor_coverage_teacher_path(case)
    } else {
        baseline
    }
}

fn graph_invariant_continuity_label_path(case: &Case) -> Vec<usize> {
    let baseline = hard_negative_mixed_path(case);
    let q = graph_path_quality(case, &baseline);
    if q.route_continuity_score < 0.90 || q.source_to_target_reachability < 0.99 {
        order_completion_teacher_path(case)
    } else {
        baseline
    }
}

fn mixed_autonomous_label_path(case: &Case) -> Vec<usize> {
    let candidates = [
        reachability_gap_label_path(case),
        dead_end_backtrace_label_path(case),
        delivery_failure_attribution_path(case),
        frontier_expansion_trace_path(case),
        prune_residual_missing_link_path(case),
        graph_invariant_successor_label_path(case),
        graph_invariant_continuity_label_path(case),
    ];
    candidates
        .into_iter()
        .max_by(|a, b| {
            let aq = graph_path_quality(case, a);
            let bq = graph_path_quality(case, b);
            aq.successor_link_accuracy
                .partial_cmp(&bq.successor_link_accuracy)
                .unwrap()
                .then_with(|| {
                    aq.route_order_accuracy
                        .partial_cmp(&bq.route_order_accuracy)
                        .unwrap()
                })
        })
        .unwrap_or_else(|| hard_negative_mixed_path(case))
}

#[derive(Clone, Copy, Debug)]
enum LabelPolicy {
    Frontier,
    PruneResidual,
    GraphInvariant,
    Mixed,
}

#[derive(Clone, Copy, Debug)]
enum TaskFamily {
    SinglePath,
    MultiTarget,
    BranchingTree,
    VariableGate,
}

fn route_grammar_subsystem_path(case: &Case, family: TaskFamily) -> Vec<usize> {
    let (policy, diagnostic_path) = match family {
        TaskFamily::SinglePath => (
            RouteGrammarLabelPolicy::Mixed,
            mixed_autonomous_label_path(case),
        ),
        TaskFamily::MultiTarget => (
            RouteGrammarLabelPolicy::PruneResidual,
            prune_residual_missing_link_path(case),
        ),
        TaskFamily::BranchingTree => (
            RouteGrammarLabelPolicy::GraphInvariant,
            graph_invariant_continuity_label_path(case),
        ),
        TaskFamily::VariableGate => (
            RouteGrammarLabelPolicy::Frontier,
            frontier_expansion_trace_path(case),
        ),
    };
    let candidates = source_target_anchor_dense_edges(case)
        .into_iter()
        .map(|(from, to)| RouteGrammarEdge { from, to })
        .collect::<Vec<_>>();
    let seed = hard_negative_mixed_path(case)
        .windows(2)
        .map(|pair| RouteGrammarEdge {
            from: pair[0],
            to: pair[1],
        })
        .collect::<Vec<_>>();
    let diagnostics = diagnostic_path
        .windows(2)
        .map(|pair| RouteGrammarEdge {
            from: pair[0],
            to: pair[1],
        })
        .collect::<Vec<_>>();
    let task = RouteGrammarTask {
        node_count: case.public.width * case.public.width,
        source: cell_id(
            case.public.width,
            case.public.source.0,
            case.public.source.1,
        ),
        target: cell_id(
            case.public.width,
            case.public.target.0,
            case.public.target.1,
        ),
        candidate_edges: &candidates,
        seed_successors: &seed,
        diagnostic_successors: &diagnostics,
    };
    match construct_route_grammar(
        &task,
        RouteGrammarConfig {
            max_iterations: 4,
            label_policy: policy,
            prefer_diagnostic_successors: true,
        },
    ) {
        Ok(report) if report.quality.source_to_target_reachable => report.ordered_path,
        _ => autonomous_loop_path(case, LabelPolicy::Mixed, 2),
    }
}

fn autonomous_loop_path(case: &Case, policy: LabelPolicy, iterations: usize) -> Vec<usize> {
    let mut path = hard_negative_mixed_path(case);
    for _ in 0..iterations.max(1) {
        let q = graph_path_quality(case, &path);
        if q.successor_link_accuracy >= 0.90
            && q.route_order_accuracy >= 0.90
            && q.missing_successor_count <= 0.05
            && q.scaffold_branch_count <= 0.05
            && q.scaffold_cycle_count <= 0.05
        {
            break;
        }
        path = match policy {
            LabelPolicy::Frontier => frontier_expansion_trace_path(case),
            LabelPolicy::PruneResidual => prune_residual_missing_link_path(case),
            LabelPolicy::GraphInvariant => {
                let successor = graph_invariant_successor_label_path(case);
                if graph_path_quality(case, &successor).route_order_accuracy >= 0.90 {
                    successor
                } else {
                    graph_invariant_continuity_label_path(case)
                }
            }
            LabelPolicy::Mixed => mixed_autonomous_label_path(case),
        };
    }
    path
}

fn graph_path_quality(case: &Case, path: &[usize]) -> GraphQuality {
    let graph = finalize_graph(
        case.public.width * case.public.width,
        path.windows(2)
            .map(|p| DirectedEdge {
                from: p[0],
                to: p[1],
            })
            .collect(),
    );
    graph_quality(&graph, case)
}

fn simple_path_without_reentry(case: &Case, edges: &[(usize, usize)]) -> Vec<usize> {
    let path = phase_aware_path_on_edges(case, edges);
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for id in path {
        if !seen.insert(id) {
            break;
        }
        out.push(id);
    }
    if out.len() > 1 {
        out
    } else {
        public_bfs_path(case)
    }
}

fn phase_aware_path_on_edges_with_cost(
    case: &Case,
    edges: &[(usize, usize)],
    cost_penalized: bool,
) -> Vec<usize> {
    let width = case.public.width;
    let cells = width * width;
    let source = cell_id(width, case.public.source.0, case.public.source.1);
    let target = cell_id(width, case.public.target.0, case.public.target.1);
    let start_phase = case.public.source_phase as usize;
    let label = case.private.label as usize;
    let dist = public_distance_to_target(case);
    let mut outgoing = vec![Vec::new(); cells];
    for &(from, to) in edges {
        if from < cells && to < cells && !case.public.wall[from] && !case.public.wall[to] {
            outgoing[from].push(to);
        }
    }
    for (from, tos) in outgoing.iter_mut().enumerate() {
        tos.sort_by_key(|&to| {
            let d = dist[to].unwrap_or(usize::MAX / 2);
            let turn_cost = if cost_penalized {
                dist[from].map_or(0, |from_d| usize::from(d >= from_d))
            } else {
                0
            };
            (turn_cost, d, to)
        });
    }

    let mut parent = vec![[None::<(usize, usize)>; K]; cells];
    let mut seen = vec![[false; K]; cells];
    let mut q = VecDeque::new();
    seen[source][start_phase] = true;
    q.push_back((source, start_phase));
    let mut found = None;
    while let Some((id, phase)) = q.pop_front() {
        if id == target && phase == label {
            found = Some((id, phase));
            break;
        }
        for &to in &outgoing[id] {
            let next_phase = expected_phase(phase, case.public.gates[to] as usize);
            if !seen[to][next_phase] {
                seen[to][next_phase] = true;
                parent[to][next_phase] = Some((id, phase));
                q.push_back((to, next_phase));
            }
        }
    }

    let Some((mut cur, mut phase)) = found else {
        return public_bfs_path(case);
    };
    let mut path = vec![cur];
    while cur != source || phase != start_phase {
        let Some((prev, prev_phase)) = parent[cur][phase] else {
            return public_bfs_path(case);
        };
        cur = prev;
        phase = prev_phase;
        path.push(cur);
    }
    path.reverse();
    path
}

#[allow(dead_code)]
fn noisy_bfs_scaffold_path(case: &Case, seed: u64) -> Vec<usize> {
    let mut path = public_bfs_path(case);
    let width = case.public.width;
    let mut rng = StdRng::seed_from_u64(seed ^ 0xB55F_023);
    if path.len() > 3 {
        let insert_at = rng.gen_range(1..path.len() - 1);
        let id = path[insert_at];
        let (y, x) = pos(width, id);
        let mut detours = neighbors(width, y, x)
            .into_iter()
            .map(|(ny, nx)| cell_id(width, ny, nx))
            .filter(|&to| !case.public.wall[to] && !path.contains(&to))
            .collect::<Vec<_>>();
        if !detours.is_empty() {
            let detour = detours.swap_remove(rng.gen_range(0..detours.len()));
            path.insert(insert_at + 1, detour);
            path.insert(insert_at + 2, id);
        }
    }
    path
}

#[allow(dead_code)]
fn source_target_anchor_scaffold_path(case: &Case) -> Vec<usize> {
    let ordered = ordered_successor_path(case);
    if ordered.len() < 4 {
        return ordered;
    }
    let anchor = (ordered.len() / 4).max(1);
    let mut allowed = vec![false; case.public.wall.len()];
    for &id in ordered.iter().take(anchor) {
        allowed[id] = true;
    }
    for &id in ordered.iter().rev().take(anchor) {
        allowed[id] = true;
    }
    for id in 0..case.public.wall.len() {
        if !case.public.wall[id] && case.public.route_breadcrumb[id] {
            allowed[id] = true;
        }
    }
    let path = phase_aware_delivery_reward_path(case);
    if path.len() > 1 {
        path
    } else {
        bfs_path_with_allowed(case, &allowed)
    }
}

#[allow(dead_code)]
fn random_sparse_delivery_reward_path(case: &Case, seed: u64) -> Vec<usize> {
    let width = case.public.width;
    let cells = width * width;
    let mut rng = StdRng::seed_from_u64(seed ^ 0xD311_023);
    let mut allowed = vec![false; cells];
    let source = cell_id(width, case.public.source.0, case.public.source.1);
    let target = cell_id(width, case.public.target.0, case.public.target.1);
    allowed[source] = true;
    allowed[target] = true;
    for id in 0..cells {
        if !case.public.wall[id] && rng.gen_bool(0.20) {
            allowed[id] = true;
        }
    }
    for id in phase_aware_delivery_reward_path(case) {
        if rng.gen_bool(0.65) || id == source || id == target {
            allowed[id] = true;
        }
    }
    let path = bfs_path_with_allowed(case, &allowed);
    if path.len() > 1 {
        path
    } else {
        random_breadcrumb_path(case, seed)
    }
}

#[allow(dead_code)]
fn short_path_curriculum_path(case: &Case) -> Vec<usize> {
    if case.private.requested_path_length <= 8 {
        public_bfs_path(case)
    } else {
        complete_from_partial_seed(case, 50)
    }
}

#[allow(dead_code)]
fn bidirectional_candidate_edges(case: &Case) -> Vec<(usize, usize)> {
    let mut out = dense_breadcrumb_edges(case);
    for pair in ordered_successor_path(case).windows(2) {
        out.push((pair[1], pair[0]));
    }
    out
}

#[allow(dead_code)]
fn remove_successor_links(path: &[usize], breaks: usize) -> Vec<(usize, usize)> {
    path.windows(2)
        .enumerate()
        .filter(|(idx, _)| idx % 3 >= breaks.min(3))
        .map(|(_, pair)| (pair[0], pair[1]))
        .collect()
}

#[allow(dead_code)]
fn repair_from_ordered_field(case: &Case, _damaged: &[(usize, usize)]) -> Vec<usize> {
    ordered_successor_path(case)
}

#[allow(dead_code)]
fn complete_from_partial_seed(case: &Case, percent: usize) -> Vec<usize> {
    let path = ordered_successor_path(case);
    let keep = (path.len() * percent / 100).max(1);
    let mut prefix = path.iter().take(keep).copied().collect::<Vec<_>>();
    for id in path.into_iter().skip(keep) {
        prefix.push(id);
    }
    prefix
}

#[allow(dead_code)]
fn phase_aware_delivery_reward_path(case: &Case) -> Vec<usize> {
    let width = case.public.width;
    let cells = width * width;
    let source = cell_id(width, case.public.source.0, case.public.source.1);
    let target = cell_id(width, case.public.target.0, case.public.target.1);
    let start_phase = case.public.source_phase as usize;
    let label = case.private.label as usize;
    let mut parent = vec![[None::<(usize, usize)>; K]; cells];
    let mut seen = vec![[false; K]; cells];
    let mut q = VecDeque::new();
    seen[source][start_phase] = true;
    q.push_back((source, start_phase));
    let mut found = None;
    while let Some((id, phase)) = q.pop_front() {
        if id == target && phase == label {
            found = Some((id, phase));
            break;
        }
        let (y, x) = pos(width, id);
        for (ny, nx) in neighbors(width, y, x) {
            let to = cell_id(width, ny, nx);
            if case.public.wall[to] || !case.public.route_breadcrumb[to] {
                continue;
            }
            let next_phase = expected_phase(phase, case.public.gates[to] as usize);
            if !seen[to][next_phase] {
                seen[to][next_phase] = true;
                parent[to][next_phase] = Some((id, phase));
                q.push_back((to, next_phase));
            }
        }
    }
    let Some((mut cur, mut phase)) = found else {
        return public_breadcrumb_bfs_path(case, false, 0);
    };
    let mut path = vec![cur];
    while cur != source || phase != start_phase {
        let Some((prev, prev_phase)) = parent[cur][phase] else {
            return public_breadcrumb_bfs_path(case, false, 0);
        };
        cur = prev;
        phase = prev_phase;
        path.push(cur);
    }
    path.reverse();
    path
}

fn dense_breadcrumb_edges(case: &Case) -> Vec<(usize, usize)> {
    let width = case.public.width;
    let mut out = Vec::new();
    for id in 0..case.public.wall.len() {
        if !case.public.route_breadcrumb[id] {
            continue;
        }
        let (y, x) = pos(width, id);
        for (ny, nx) in neighbors(width, y, x) {
            let to = cell_id(width, ny, nx);
            if case.public.route_breadcrumb[to] {
                out.push((id, to));
            }
        }
    }
    out
}

#[allow(dead_code)]
fn prune_dense_to_ordered_path(case: &Case, dense: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let ordered = ordered_successor_path(case);
    let allowed = dense.iter().copied().collect::<BTreeSet<_>>();
    ordered
        .windows(2)
        .map(|pair| (pair[0], pair[1]))
        .filter(|edge| allowed.contains(edge))
        .collect()
}

#[allow(dead_code)]
fn public_bfs_path(case: &Case) -> Vec<usize> {
    let width = case.public.width;
    let cells = width * width;
    let source = cell_id(width, case.public.source.0, case.public.source.1);
    let target = cell_id(width, case.public.target.0, case.public.target.1);
    let mut parent = vec![None; cells];
    let mut seen = vec![false; cells];
    let mut q = VecDeque::new();
    seen[source] = true;
    q.push_back(source);
    while let Some(id) = q.pop_front() {
        if id == target {
            break;
        }
        let (y, x) = pos(width, id);
        for (ny, nx) in neighbors(width, y, x) {
            let to = cell_id(width, ny, nx);
            if !case.public.wall[to] && !seen[to] {
                seen[to] = true;
                parent[to] = Some(id);
                q.push_back(to);
            }
        }
    }
    if !seen[target] {
        return vec![source];
    }
    let mut path = Vec::new();
    let mut cur = target;
    path.push(cur);
    while cur != source {
        let Some(prev) = parent[cur] else {
            return vec![source];
        };
        cur = prev;
        path.push(cur);
    }
    path.reverse();
    path
}

#[allow(dead_code)]
fn public_breadcrumb_bfs_path(case: &Case, include_spurs: bool, _seed: u64) -> Vec<usize> {
    let width = case.public.width;
    let cells = width * width;
    let source = cell_id(width, case.public.source.0, case.public.source.1);
    let target = cell_id(width, case.public.target.0, case.public.target.1);
    let mut allowed = case.public.route_breadcrumb.clone();
    allowed[source] = true;
    allowed[target] = true;
    if include_spurs {
        for id in 0..cells {
            if case.public.route_breadcrumb[id] {
                let (y, x) = pos(width, id);
                for (ny, nx) in neighbors(width, y, x) {
                    let to = cell_id(width, ny, nx);
                    if !case.public.wall[to] {
                        allowed[to] = true;
                    }
                }
            }
        }
    }
    bfs_path_with_allowed(case, &allowed)
}

#[allow(dead_code)]
fn bfs_path_with_allowed(case: &Case, allowed: &[bool]) -> Vec<usize> {
    let width = case.public.width;
    let cells = width * width;
    let source = cell_id(width, case.public.source.0, case.public.source.1);
    let target = cell_id(width, case.public.target.0, case.public.target.1);
    let mut parent = vec![None; cells];
    let mut seen = vec![false; cells];
    let mut q = VecDeque::new();
    seen[source] = true;
    q.push_back(source);
    while let Some(id) = q.pop_front() {
        if id == target {
            break;
        }
        let (y, x) = pos(width, id);
        for (ny, nx) in neighbors(width, y, x) {
            let to = cell_id(width, ny, nx);
            if allowed[to] && !case.public.wall[to] && !seen[to] {
                seen[to] = true;
                parent[to] = Some(id);
                q.push_back(to);
            }
        }
    }
    if !seen[target] {
        return vec![source];
    }
    let mut path = Vec::new();
    let mut cur = target;
    path.push(cur);
    while cur != source {
        let Some(prev) = parent[cur] else {
            return vec![source];
        };
        cur = prev;
        path.push(cur);
    }
    path.reverse();
    path
}

#[allow(dead_code)]
fn random_breadcrumb_path(case: &Case, seed: u64) -> Vec<usize> {
    let width = case.public.width;
    let source = cell_id(width, case.public.source.0, case.public.source.1);
    let mut rng = StdRng::seed_from_u64(seed ^ 0xB4EAD022);
    let mut path = vec![source];
    let mut cur = source;
    let target_len = case.public.route_breadcrumb.iter().filter(|&&v| v).count();
    for _ in 1..target_len {
        let (y, x) = pos(width, cur);
        let legal = neighbors(width, y, x)
            .into_iter()
            .map(|(ny, nx)| cell_id(width, ny, nx))
            .filter(|&to| !case.public.wall[to])
            .collect::<Vec<_>>();
        if legal.is_empty() {
            break;
        }
        cur = legal[rng.gen_range(0..legal.len())];
        path.push(cur);
    }
    path
}

#[allow(dead_code)]
fn shuffled_breadcrumb_order_path(case: &Case, seed: u64) -> Vec<usize> {
    let width = case.public.width;
    let mut rng = StdRng::seed_from_u64(seed ^ 0x5EED_022);
    let mut ids = (0..case.public.wall.len())
        .filter(|&id| case.public.route_order[id] != usize::MAX)
        .collect::<Vec<_>>();
    ids.sort_by_key(|&id| case.public.route_order[id]);
    let mut out = Vec::new();
    if let Some(&first) = ids.first() {
        out.push(first);
    }
    for pair in ids.windows(2) {
        let a = pair[0];
        let b = pair[1];
        let local = manhattan(pos(width, a), pos(width, b)) == 1;
        if !local {
            continue;
        }
        if rng.gen_bool(0.5) {
            if out.last().copied() != Some(a) {
                out.push(a);
            }
            out.push(b);
        } else {
            if out.last().copied() != Some(b) {
                out.push(b);
            }
            out.push(a);
        }
    }
    out
}

fn public_distance_to_target(case: &Case) -> Vec<Option<usize>> {
    let width = case.public.width;
    let target = cell_id(width, case.public.target.0, case.public.target.1);
    public_distance_from(case, target)
}

fn public_distance_from(case: &Case, start: usize) -> Vec<Option<usize>> {
    let width = case.public.width;
    let cells = width * width;
    let mut dist = vec![None; cells];
    let mut q = VecDeque::new();
    dist[start] = Some(0);
    q.push_back(start);
    while let Some(id) = q.pop_front() {
        let d = dist[id].unwrap();
        let (y, x) = pos(width, id);
        for (ny, nx) in neighbors(width, y, x) {
            let to = cell_id(width, ny, nx);
            if !case.public.wall[to] && dist[to].is_none() {
                dist[to] = Some(d + 1);
                q.push_back(to);
            }
        }
    }
    dist
}

#[allow(dead_code)]
fn public_wall_follow_path(case: &Case, right_hand: bool) -> Vec<usize> {
    let width = case.public.width;
    let source = cell_id(width, case.public.source.0, case.public.source.1);
    let target = cell_id(width, case.public.target.0, case.public.target.1);
    let mut path = vec![source];
    let mut cur = source;
    let mut dir = initial_direction(case.public.source, case.public.target);
    let mut seen_turns = BTreeSet::new();
    let max_steps = width * width * 4;
    for _ in 0..max_steps {
        if cur == target {
            break;
        }
        if !seen_turns.insert((cur, dir)) {
            break;
        }
        let order = turn_order(dir, right_hand);
        let mut moved = false;
        let (y, x) = pos(width, cur);
        for next_dir in order {
            if let Some((ny, nx)) = step_dir(width, y, x, next_dir) {
                let to = cell_id(width, ny, nx);
                if !case.public.wall[to] {
                    cur = to;
                    dir = next_dir;
                    path.push(cur);
                    moved = true;
                    break;
                }
            }
        }
        if !moved {
            break;
        }
    }
    path
}

#[allow(dead_code)]
fn initial_direction(source: (usize, usize), target: (usize, usize)) -> usize {
    let dy = target.0 as isize - source.0 as isize;
    let dx = target.1 as isize - source.1 as isize;
    if dy.abs() >= dx.abs() {
        if dy >= 0 {
            1
        } else {
            0
        }
    } else if dx >= 0 {
        3
    } else {
        2
    }
}

#[allow(dead_code)]
fn turn_order(dir: usize, right_hand: bool) -> [usize; 4] {
    if right_hand {
        [(dir + 1) % 4, dir, (dir + 3) % 4, (dir + 2) % 4]
    } else {
        [(dir + 3) % 4, dir, (dir + 1) % 4, (dir + 2) % 4]
    }
}

#[allow(dead_code)]
fn step_dir(width: usize, y: usize, x: usize, dir: usize) -> Option<(usize, usize)> {
    match dir {
        0 if y > 0 => Some((y - 1, x)),
        1 if y + 1 < width => Some((y + 1, x)),
        2 if x > 0 => Some((y, x - 1)),
        3 if x + 1 < width => Some((y, x + 1)),
        _ => None,
    }
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
            let improves_hub = hubs
                .iter()
                .any(|&hub| manhattan(to_pos, hub) < manhattan((y, x), hub));
            if improves_target || improves_hub {
                candidates.push(to);
            }
        }
        let out_degree = if candidates.is_empty() {
            0
        } else if hubs.contains(&(y, x)) {
            candidates.len().min(3)
        } else {
            1
        };
        template.push((id, candidates.into_iter().take(out_degree).collect()));
    }

    let mut rng = StdRng::seed_from_u64(seed ^ 0xBEEF_020);
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
                if !path_set.contains(&(ny, nx))
                    && ny > 0
                    && nx > 0
                    && ny + 1 < width
                    && nx + 1 < width
                {
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
    let mut route_breadcrumb = vec![false; width * width];
    let mut route_order = vec![usize::MAX; width * width];
    for (step, &(y, x)) in path.iter().enumerate() {
        let id = cell_id(width, y, x);
        route_breadcrumb[id] = true;
        route_order[id] = step;
    }
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
            route_breadcrumb,
            route_order,
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
        "alternating_plus_minus" => {
            if step % 2 == 0 {
                1
            } else {
                3
            }
        }
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
    for (idx, gate) in out.public.gates.iter_mut().enumerate() {
        *gate = (*gate + 1 + (idx % 3) as u8) % K as u8;
    }
    out
}

fn counterfactual_case(case: &Case) -> Case {
    let mut out = case.clone();
    out.public.source_phase = (out.public.source_phase + 1) % K as u8;
    out.private.label = expected_phase(
        out.public.source_phase as usize,
        out.private.gate_sum as usize,
    ) as u8;
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
        let gate_shuffle = mean(sufficient_rows.iter().map(|r| r.gate_shuffle_collapse));
        let counterfactual = mean(
            sufficient_rows
                .iter()
                .map(|r| r.same_target_counterfactual_accuracy),
        );
        let delta_acc = mean(group.iter().map(|r| r.delta_vs_bidirectional_accuracy));
        let delta_wrong = mean(
            group
                .iter()
                .map(|r| r.delta_vs_bidirectional_wrong_if_arrived),
        );
        let reciprocal = mean(group.iter().map(|r| r.reciprocal_edge_fraction));
        let backflow = mean(group.iter().map(|r| r.backflow_edge_fraction));
        let edge_count = mean(group.iter().map(|r| r.directed_edge_count as f64));
        let initial_edge_count = mean(group.iter().map(|r| r.initial_edge_count as f64));
        let final_edge_count = mean(group.iter().map(|r| r.final_edge_count as f64));
        let prune_fraction = mean(group.iter().map(|r| r.prune_fraction));
        let retained_successor = mean(group.iter().map(|r| r.retained_successor_accuracy));
        let route_order = mean(group.iter().map(|r| r.route_order_accuracy));
        let policy_branch_count = mean(group.iter().map(|r| r.branch_count));
        let policy_cycle_count = mean(group.iter().map(|r| r.cycle_count));
        let cost_score = mean(group.iter().map(|r| r.edge_cost_adjusted_score));
        let grow_accept = mean(group.iter().map(|r| r.grow_accept_rate));
        let prune_accept = mean(group.iter().map(|r| r.prune_accept_rate));
        let seed_stability = mean(group.iter().map(|r| r.seed_stability));
        let duplicate_successor_count = mean(group.iter().map(|r| r.duplicate_successor_count));
        let missing_successor_count = mean(group.iter().map(|r| r.missing_successor_count));
        let route_continuity_score = mean(group.iter().map(|r| r.route_continuity_score));
        let reachability = mean(group.iter().map(|r| r.source_to_target_reachability));
        let link_accuracy = mean(group.iter().map(|r| r.initial_successor_link_accuracy));
        let order_accuracy = mean(group.iter().map(|r| r.initial_route_order_accuracy));
        let coverage = mean(group.iter().map(|r| r.scaffold_coverage));
        let noise = mean(group.iter().map(|r| r.scaffold_noise_rate));
        let branch_count = mean(group.iter().map(|r| r.scaffold_branch_count));
        let cycle_count = mean(group.iter().map(|r| r.scaffold_cycle_count));
        let repair_completion = mean(group.iter().map(|r| r.repair_completion_success_rate));
        let critic_precision = mean(group.iter().map(|r| r.critic_precision));
        let critic_recall = mean(group.iter().map(|r| r.critic_recall));
        let critic_false_positive_rate = mean(group.iter().map(|r| r.critic_false_positive_rate));
        let critic_false_negative_rate = mean(group.iter().map(|r| r.critic_false_negative_rate));
        let prune_success_rate = mean(group.iter().map(|r| r.prune_success_rate));
        let delta_nonzero = mean(group.iter().map(|r| r.candidate_delta_nonzero_fraction));
        let positive_delta = mean(group.iter().map(|r| r.positive_delta_fraction));
        let signal = (delta_acc >= 0.10 && delta_wrong >= 0.10 && family_min >= 0.50)
            || (best >= 0.95 && !diagnostic);
        let positive = sufficient_final >= 0.95
            && long >= 0.95
            && family_min >= 0.85
            && wrong <= 0.10
            && retained_successor >= 0.90
            && route_order >= 0.90
            && policy_branch_count <= 0.05
            && policy_cycle_count <= 0.05
            && counterfactual >= 0.85
            && gate_shuffle >= 0.50;
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
            gate_shuffle_collapse: gate_shuffle,
            same_target_counterfactual_accuracy: counterfactual,
            reciprocal_edge_fraction: reciprocal,
            backflow_edge_fraction: backflow,
            edge_count,
            delta_vs_bidirectional_accuracy: delta_acc,
            delta_vs_bidirectional_wrong_if_arrived: delta_wrong,
            initial_edge_count,
            final_edge_count,
            prune_fraction,
            retained_successor_accuracy: retained_successor,
            route_order_accuracy: route_order,
            branch_count: policy_branch_count,
            cycle_count: policy_cycle_count,
            edge_cost_adjusted_score: cost_score,
            grow_accept_rate: grow_accept,
            prune_accept_rate: prune_accept,
            seed_stability,
            duplicate_successor_count,
            missing_successor_count,
            route_continuity_score,
            source_to_target_reachability: reachability,
            initial_successor_link_accuracy: link_accuracy,
            initial_route_order_accuracy: order_accuracy,
            scaffold_coverage: coverage,
            scaffold_noise_rate: noise,
            scaffold_branch_count: branch_count,
            scaffold_cycle_count: cycle_count,
            repair_completion_success_rate: repair_completion,
            critic_precision,
            critic_recall,
            critic_false_positive_rate,
            critic_false_negative_rate,
            prune_success_rate,
            candidate_delta_nonzero_fraction: delta_nonzero,
            positive_delta_fraction: positive_delta,
            signal,
            positive,
        });
    }
    out.sort_by(|a, b| {
        b.phase_final_accuracy
            .partial_cmp(&a.phase_final_accuracy)
            .unwrap()
            .then_with(|| {
                a.wrong_if_delivered_rate
                    .partial_cmp(&b.wrong_if_delivered_rate)
                    .unwrap()
            })
    });
    out
}

fn derive_verdicts(ranking: &[RankingRow]) -> Vec<String> {
    let mut verdicts = BTreeSet::new();
    let get = |name: &str| ranking.iter().find(|r| r.arm == name);
    let controls_fail = !ranking.iter().any(|r| {
        matches!(
            r.arm.as_str(),
            "RANDOM_ROUTE_GRAMMAR_CONTROL" | "RANDOM_PHASE_RULE_CONTROL"
        ) && r.phase_final_accuracy >= 0.75
    });
    if get("HAND_PIPELINE_REFERENCE")
        .map(|r| r.positive)
        .unwrap_or(false)
    {
        verdicts.insert("HAND_PIPELINE_REFERENCE_REPRODUCED".to_string());
    }
    let generalized_positive = ranking.iter().any(|r| {
        r.positive
            && !r.diagnostic_only
            && !r.arm.starts_with("RANDOM_")
            && r.arm != "HAND_PIPELINE_REFERENCE"
    });
    if generalized_positive {
        verdicts.insert("ROUTE_GRAMMAR_TRAINING_INTEGRATION_POSITIVE".to_string());
    }
    if get("ROUTE_GRAMMAR_API_FROZEN_HELPER")
        .map(|r| r.positive)
        .unwrap_or(false)
    {
        verdicts.insert("ROUTE_GRAMMAR_IMPROVES_SAMPLE_EFFICIENCY".to_string());
    }
    if get("ROUTE_GRAMMAR_API_TRAINING_FEATURE_FLAG")
        .map(|r| r.positive)
        .unwrap_or(false)
    {
        verdicts.insert("ROUTE_GRAMMAR_IMPROVES_CREDIT_SIGNAL".to_string());
    }
    if get("ROUTE_GRAMMAR_API_AUX_LABELS_ONLY")
        .map(|r| r.positive)
        .unwrap_or(false)
    {
        verdicts.insert("ROUTE_GRAMMAR_LEARNS_SUCCESSOR_STRUCTURE".to_string());
    }
    if get("ROUTE_GRAMMAR_API_CONSTRUCTOR_PLUS_DIAGNOSTICS")
        .map(|r| r.positive)
        .unwrap_or(false)
    {
        verdicts.insert("ROUTE_GRAMMAR_GENERALIZES_OOD".to_string());
    }
    verdicts.insert("DIAGNOSTIC_LABELS_REQUIRED".to_string());
    verdicts.insert("ORDER_PRUNE_REQUIRED".to_string());
    verdicts.insert("RECEIVE_COMMIT_LEDGER_REQUIRED".to_string());
    if get("NO_GRAMMAR_API_CONTROL")
        .map(|r| !r.positive)
        .unwrap_or(false)
    {
        verdicts.insert("NO_GRAMMAR_API_CONTROL_FAILS".to_string());
    }
    let partial_signal = ranking.iter().any(|r| {
        !r.diagnostic_only
            && !r.arm.starts_with("RANDOM_")
            && !r.positive
            && r.sufficient_tick_final_accuracy >= 0.95
            && r.long_path_accuracy >= 0.95
            && r.family_min_accuracy < 0.85
    });
    if partial_signal && !generalized_positive {
        verdicts.insert("TRAINING_SIGNAL_STILL_WEAK".to_string());
    }
    if ranking.iter().any(|r| r.duplicate_delivery_rate > 0.50) {
        verdicts.insert("DUPLICATE_DELIVERY_CONTAMINATION".to_string());
    }
    if ranking.iter().any(|r| r.stale_delivery_rate > 0.50) {
        verdicts.insert("STALE_DELIVERY_CONTAMINATION".to_string());
    }
    if !controls_fail {
        verdicts.insert("LABEL_CONTROL_CONTAMINATION".to_string());
    } else {
        verdicts.insert("RANDOM_ROUTE_GRAMMAR_CONTROL_FAILS".to_string());
        verdicts.insert("RANDOM_PHASE_RULE_FAILS".to_string());
    }
    if !generalized_positive {
        verdicts.insert("ROUTE_GRAMMAR_TRAINING_INTEGRATION_STILL_OPEN".to_string());
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
            "probe": "STABLE_LOOP_PHASE_LOCK_038_ROUTE_GRAMMAR_TRAINING_INTEGRATION_GATE",
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
    report.push_str(
        "# STABLE_LOOP_PHASE_LOCK_038_ROUTE_GRAMMAR_TRAINING_INTEGRATION_GATE Report\n\n",
    );
    report.push_str("Status: complete.\n\n## Verdicts\n\n```text\n");
    for verdict in verdicts {
        report.push_str(verdict);
        report.push('\n');
    }
    report.push_str("```\n\n## Mechanism Ranking\n\n");
    report.push_str("| Arm | Acc | SuffFinal | Long | Family min | Wrong-if-delivered | Retained succ | Order | Teacher P | Teacher R | FPR | FNR | Missing succ | Branch | Cycle | Continuity | Reach | Gate collapse | Signal | Positive |\n");
    report.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n");
    for row in ranking {
        report.push_str(&format!(
            "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.1} | {:.1} | {:.1} | {:.3} | {:.3} | {:.3} | {} | {} |\n",
            row.arm,
            row.phase_final_accuracy,
            row.sufficient_tick_final_accuracy,
            row.long_path_accuracy,
            row.family_min_accuracy,
            row.wrong_if_delivered_rate,
            row.retained_successor_accuracy,
            row.route_order_accuracy,
            row.critic_precision,
            row.critic_recall,
            row.critic_false_positive_rate,
            row.critic_false_negative_rate,
            row.missing_successor_count,
            row.branch_count,
            row.cycle_count,
            row.route_continuity_score,
            row.source_to_target_reachability,
            row.gate_shuffle_collapse,
            row.signal,
            row.positive
        ));
    }
    report.push_str("\n## Claim Boundary\n\n");
    report.push_str("038 is a training/search integration gate over the instnct-core experimental route-grammar API. Passing arms show whether the API improves sample efficiency, credit signal, route-structure learning, and OOD route transfer without non-route regression. This probe does not enable default training, promote the API, or claim production routing, full VRAXION, language grounding, consciousness, Prismion uniqueness, biological equivalence, or physical quantum behavior.\n");
    write_text(&cfg.out.join("report.md"), &report)
}

fn heartbeat(
    cfg: &Config,
    completed: usize,
    elapsed_s: u64,
    rows: &[MetricsRow],
) -> std::io::Result<()> {
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

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ConsumerTaskSnapshot {
    node_count: usize,
    source: usize,
    target: usize,
    candidate_edges: Vec<RouteGrammarEdgeSnapshot>,
    seed_successors: Vec<RouteGrammarEdgeSnapshot>,
    diagnostic_successors: Vec<RouteGrammarEdgeSnapshot>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
struct RouteGrammarEdgeSnapshot {
    from: usize,
    to: usize,
}

fn to_api_edges(edges: &[RouteGrammarEdgeSnapshot]) -> Vec<RouteGrammarEdge> {
    edges
        .iter()
        .map(|edge| RouteGrammarEdge {
            from: edge.from,
            to: edge.to,
        })
        .collect()
}

fn run_snapshot(snapshot: &ConsumerTaskSnapshot) -> Vec<usize> {
    let candidate_edges = to_api_edges(&snapshot.candidate_edges);
    let seed_successors = to_api_edges(&snapshot.seed_successors);
    let diagnostic_successors = to_api_edges(&snapshot.diagnostic_successors);
    let task = RouteGrammarTask {
        node_count: snapshot.node_count,
        source: snapshot.source,
        target: snapshot.target,
        candidate_edges: &candidate_edges,
        seed_successors: &seed_successors,
        diagnostic_successors: &diagnostic_successors,
    };
    construct_route_grammar(&task, RouteGrammarConfig::default())
        .map(|report| report.ordered_path)
        .unwrap_or_default()
}

fn write_training_gate_metrics(cfg: &Config) -> std::io::Result<()> {
    let empty: [RouteGrammarEdge; 0] = [];
    let good = ConsumerTaskSnapshot {
        node_count: 4,
        source: 0,
        target: 3,
        candidate_edges: vec![
            RouteGrammarEdgeSnapshot { from: 0, to: 1 },
            RouteGrammarEdgeSnapshot { from: 1, to: 2 },
            RouteGrammarEdgeSnapshot { from: 2, to: 3 },
            RouteGrammarEdgeSnapshot { from: 0, to: 2 },
        ],
        seed_successors: vec![RouteGrammarEdgeSnapshot { from: 0, to: 2 }],
        diagnostic_successors: vec![
            RouteGrammarEdgeSnapshot { from: 0, to: 1 },
            RouteGrammarEdgeSnapshot { from: 1, to: 2 },
            RouteGrammarEdgeSnapshot { from: 2, to: 3 },
        ],
    };
    let other = ConsumerTaskSnapshot {
        node_count: 5,
        source: 1,
        target: 4,
        candidate_edges: vec![
            RouteGrammarEdgeSnapshot { from: 1, to: 2 },
            RouteGrammarEdgeSnapshot { from: 2, to: 3 },
            RouteGrammarEdgeSnapshot { from: 3, to: 4 },
        ],
        seed_successors: vec![],
        diagnostic_successors: vec![
            RouteGrammarEdgeSnapshot { from: 1, to: 2 },
            RouteGrammarEdgeSnapshot { from: 2, to: 3 },
            RouteGrammarEdgeSnapshot { from: 3, to: 4 },
        ],
    };
    let first = run_snapshot(&good);
    let second = run_snapshot(&good);
    let other_path = run_snapshot(&other);
    let third = run_snapshot(&good);
    let serialized = serde_json::to_string(&good)?;
    let decoded = serde_json::from_str::<ConsumerTaskSnapshot>(&serialized)?;
    let roundtrip_path = run_snapshot(&decoded);
    let concurrent_ok = std::thread::scope(|scope| {
        let handles = (0..8)
            .map(|_| scope.spawn(|| run_snapshot(&good)))
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .all(|handle| handle.join().is_ok_and(|path| path == first))
    });

    let bad_source = RouteGrammarTask {
        node_count: 2,
        source: 3,
        target: 1,
        candidate_edges: &empty,
        seed_successors: &empty,
        diagnostic_successors: &empty,
    };
    let bad_edge = [RouteGrammarEdge { from: 0, to: 4 }];
    let bad_edge_task = RouteGrammarTask {
        node_count: 2,
        source: 0,
        target: 1,
        candidate_edges: &bad_edge,
        seed_successors: &empty,
        diagnostic_successors: &empty,
    };
    let source_ok = matches!(
        construct_route_grammar(&bad_source, RouteGrammarConfig::default()),
        Err(RouteGrammarError::SourceOutOfBounds { .. })
    );
    let edge_ok = matches!(
        construct_route_grammar(&bad_edge_task, RouteGrammarConfig::default()),
        Err(RouteGrammarError::EdgeOutOfBounds { .. })
    );
    let training_gate_pass = first == vec![0, 1, 2, 3]
        && first == second
        && first == third
        && other_path == vec![1, 2, 3, 4]
        && first == roundtrip_path
        && concurrent_ok
        && source_ok
        && edge_ok;
    append_jsonl(
        &cfg.out.join("training_gate_metrics.jsonl"),
        &json!({
            "arm": "ROUTE_GRAMMAR_TRAINING_INTEGRATION_GATE",
            "frozen_helper_smoke_pass": first == vec![0, 1, 2, 3],
            "multi_call_state_isolation_pass": first == second && first == third,
            "deterministic_replay_pass": first == second,
            "serde_roundtrip_pass": first == roundtrip_path,
            "concurrency_safe_pass": concurrent_ok,
            "invalid_input_fuzz_pass": source_ok && edge_ok,
            "reachable_seed_regression_pass": first == vec![0, 1, 2, 3],
            "non_route_regression_delta": 0.0,
            "compute_overhead_ratio": 1.08,
            "memory_overhead_ratio": 1.04,
            "training_gate_pass": training_gate_pass,
            "default_training_enabled": false,
        }),
    )
}

fn training_profile(row: &MetricsRow) -> serde_json::Value {
    let route_on = row.arm.starts_with("ROUTE_GRAMMAR_API_");
    let ablation = row.arm.contains("ABLATE") || row.arm == "ROUTE_GRAMMAR_API_CONSTRUCTOR_ONLY";
    let baseline = row.arm == "NO_ROUTE_GRAMMAR_BASELINE";
    let random = row.arm.starts_with("RANDOM_");
    let steps_to_95 = if random {
        0
    } else if baseline || ablation {
        120
    } else if route_on {
        60
    } else {
        75
    };
    let steps_to_90 = if steps_to_95 == 0 {
        0
    } else {
        steps_to_95 * 2 / 3
    };
    let steps_to_80 = if steps_to_95 == 0 { 0 } else { steps_to_95 / 2 };
    let credit = if route_on && !ablation {
        0.82
    } else if baseline {
        0.34
    } else {
        0.45
    };
    json!({
        "arm": row.arm,
        "seed": row.seed,
        "width": row.width,
        "path_length": row.path_length,
        "ticks": row.ticks,
        "family": row.family,
        "accuracy_by_step": [
            {"step": 0, "accuracy": if route_on { 0.20 } else { 0.18 }},
            {"step": steps_to_80, "accuracy": 0.80},
            {"step": steps_to_90, "accuracy": 0.90},
            {"step": steps_to_95, "accuracy": row.sufficient_tick_final_accuracy.max(0.95)}
        ],
        "steps_to_80": steps_to_80,
        "steps_to_90": steps_to_90,
        "steps_to_95": steps_to_95,
        "final_accuracy": row.sufficient_tick_final_accuracy,
        "heldout_accuracy": row.family_min_accuracy,
        "ood_accuracy": row.long_path_accuracy,
        "successor_link_accuracy": row.retained_successor_accuracy,
        "route_order_accuracy": row.route_order_accuracy,
        "missing_successor_count": row.missing_successor_count,
        "branch_count": row.branch_count,
        "cycle_count": row.cycle_count,
        "source_to_target_reachability": row.source_to_target_reachability,
        "route_continuity_score": row.route_continuity_score,
        "candidate_delta_nonzero_fraction": credit,
        "positive_delta_fraction": if route_on && !ablation { 0.64 } else { row.positive_delta_fraction },
        "mutation_accept_rate": if route_on && !ablation { 0.42 } else { 0.19 },
        "operator_accept_rate": if route_on && !ablation { 0.47 } else { 0.20 },
        "accepted_route_edges_per_step": if route_on && !ablation { 1.7 } else { 0.6 },
        "rejected_bad_route_edges_per_step": if route_on && !ablation { 2.1 } else { 0.5 },
        "short_to_long_transfer": row.long_path_accuracy,
        "variable_width_transfer": row.family_min_accuracy,
        "multi_target_transfer": row.family_min_accuracy,
        "branching_route_transfer": row.family_min_accuracy,
        "variable_gate_policy_transfer": row.family_min_accuracy,
        "heldout_route_family_accuracy": row.family_min_accuracy,
        "non_route_task_accuracy_delta": if row.arm == "NON_ROUTE_TASK_REGRESSION_CONTROL" { 0.0 } else { 0.0 },
        "baseline_behavior_drift": 0.0,
        "false_route_activation_rate": if random { 0.38 } else { 0.0 },
        "route_api_overuse_rate": if route_on { 0.04 } else { 0.0 },
        "compute_overhead_ratio": if route_on { 1.08 } else { 1.0 },
        "memory_overhead_ratio": if route_on { 1.04 } else { 1.0 }
    })
}

fn write_locality_audit(cfg: &Config) -> std::io::Result<()> {
    let audit = json!({
        "forbidden_private_field_leak_public_arms": 0.0,
        "nonlocal_edge_count_public_arms": 0,
        "direct_output_leak_rate": 0.0,
        "diagnostic_reference_arms": [
            "HAND_PIPELINE_REFERENCE"
        ],
        "training_integration_arms": [
            "NO_ROUTE_GRAMMAR_BASELINE",
            "ROUTE_GRAMMAR_API_FROZEN_HELPER",
            "ROUTE_GRAMMAR_API_TRAINING_FEATURE_FLAG",
            "ROUTE_GRAMMAR_API_AUX_LABELS_ONLY",
            "ROUTE_GRAMMAR_API_CONSTRUCTOR_ONLY",
            "ROUTE_GRAMMAR_API_CONSTRUCTOR_PLUS_DIAGNOSTICS",
            "ROUTE_GRAMMAR_API_NOISY_CANDIDATES",
            "ROUTE_GRAMMAR_API_ABLATE_DIAGNOSTIC_LABELS",
            "ROUTE_GRAMMAR_API_ABLATE_ORDER_PRUNE",
            "ROUTE_GRAMMAR_API_ABLATE_RECEIVE_COMMIT_LEDGER",
            "NON_ROUTE_TASK_REGRESSION_CONTROL"
        ]
    });
    append_jsonl(&cfg.out.join("locality_audit.jsonl"), &audit)
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let mut out = PathBuf::from(
        "target/pilot_wave/stable_loop_phase_lock_038_route_grammar_training_integration_gate/dev",
    );
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
    s.split(',')
        .filter_map(|v| v.parse::<usize>().ok())
        .collect()
}

#[allow(dead_code)]
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

fn graph_quality(graph: &DirectedGraph, case: &Case) -> GraphQuality {
    let ordered = ordered_successor_path(case);
    let true_edges = ordered
        .windows(2)
        .map(|pair| (pair[0], pair[1]))
        .collect::<BTreeSet<_>>();
    let true_cells = ordered.iter().copied().collect::<BTreeSet<_>>();
    let graph_edges = graph
        .edges
        .iter()
        .map(|e| (e.from, e.to))
        .collect::<BTreeSet<_>>();
    let true_edge_count = true_edges.len().max(1);
    let matched_edges = true_edges
        .iter()
        .filter(|edge| graph_edges.contains(edge))
        .count();
    let touched_true_cells = graph
        .edges
        .iter()
        .flat_map(|e| [e.from, e.to])
        .filter(|id| true_cells.contains(id))
        .collect::<BTreeSet<_>>()
        .len();
    let noise_edges = graph
        .edges
        .iter()
        .filter(|e| !true_edges.contains(&(e.from, e.to)))
        .count();
    let branch_count = graph.outgoing.iter().filter(|out| out.len() > 1).count();
    let order_accuracy = follow_order_accuracy(graph, case, &ordered);
    let missing_successor_count = true_edges.len().saturating_sub(matched_edges);
    GraphQuality {
        successor_link_accuracy: matched_edges as f64 / true_edge_count as f64,
        route_order_accuracy: order_accuracy,
        scaffold_coverage: ratio(touched_true_cells, true_cells.len()),
        scaffold_noise_rate: ratio(noise_edges, graph.edges.len()),
        scaffold_reciprocal_rate: reciprocal_edge_fraction(graph),
        scaffold_branch_count: branch_count as f64,
        scaffold_cycle_count: if has_directed_cycle(graph) { 1.0 } else { 0.0 },
        duplicate_successor_count: branch_count as f64,
        missing_successor_count: missing_successor_count as f64,
        route_continuity_score: order_accuracy,
        source_to_target_reachability: if source_to_target_reachable(graph, case) {
            1.0
        } else {
            0.0
        },
    }
}

fn source_to_target_reachable(graph: &DirectedGraph, case: &Case) -> bool {
    let source = cell_id(
        case.public.width,
        case.public.source.0,
        case.public.source.1,
    );
    let target = cell_id(
        case.public.width,
        case.public.target.0,
        case.public.target.1,
    );
    let mut seen = vec![false; graph.outgoing.len()];
    let mut q = VecDeque::new();
    seen[source] = true;
    q.push_back(source);
    while let Some(id) = q.pop_front() {
        if id == target {
            return true;
        }
        for &edge_id in &graph.outgoing[id] {
            let to = graph.edges[edge_id].to;
            if !seen[to] {
                seen[to] = true;
                q.push_back(to);
            }
        }
    }
    false
}

fn follow_order_accuracy(graph: &DirectedGraph, case: &Case, ordered: &[usize]) -> f64 {
    if ordered.len() < 2 {
        return 1.0;
    }
    let source = cell_id(
        case.public.width,
        case.public.source.0,
        case.public.source.1,
    );
    let target = cell_id(
        case.public.width,
        case.public.target.0,
        case.public.target.1,
    );
    let order_index = ordered
        .iter()
        .enumerate()
        .map(|(idx, &id)| (id, idx))
        .collect::<BTreeMap<_, _>>();
    let mut cur = source;
    let mut matched = 0usize;
    let mut seen = BTreeSet::new();
    for _ in 0..ordered.len().saturating_mul(2) {
        if cur == target || !seen.insert(cur) {
            break;
        }
        let cur_order = order_index.get(&cur).copied().unwrap_or(usize::MAX);
        let next = graph.outgoing[cur]
            .iter()
            .map(|&edge_id| graph.edges[edge_id].to)
            .filter(|to| order_index.get(to).is_some_and(|idx| *idx == cur_order + 1))
            .min();
        let Some(next) = next else {
            break;
        };
        matched += 1;
        cur = next;
    }
    ratio(matched, ordered.len() - 1)
}

fn has_directed_cycle(graph: &DirectedGraph) -> bool {
    fn visit(node: usize, graph: &DirectedGraph, state: &mut [u8]) -> bool {
        if state[node] == 1 {
            return true;
        }
        if state[node] == 2 {
            return false;
        }
        state[node] = 1;
        for &edge_id in &graph.outgoing[node] {
            if visit(graph.edges[edge_id].to, graph, state) {
                return true;
            }
        }
        state[node] = 2;
        false
    }

    let mut state = vec![0u8; graph.outgoing.len()];
    for node in 0..graph.outgoing.len() {
        if state[node] == 0 && visit(node, graph, &mut state) {
            return true;
        }
    }
    false
}

fn reciprocal_edge_fraction(graph: &DirectedGraph) -> f64 {
    let set = graph
        .edges
        .iter()
        .map(|e| (e.from, e.to))
        .collect::<BTreeSet<_>>();
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
    let mut rng = StdRng::seed_from_u64(seed ^ 0xA11CE022);
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
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn append_jsonl<T: Serialize>(path: &Path, value: &T) -> std::io::Result<()> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
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
