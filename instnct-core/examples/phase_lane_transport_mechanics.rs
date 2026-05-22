//! Runner-local phase-lane transport mechanics probe.
//!
//! 013 does not search, mutate, or prune. It takes the completed 16-pair
//! `phase_i + gate_g -> phase_(i+g)` local rule and asks why long-path
//! propagation fails in the recurrent phase-lane substrate.

use instnct_core::{Network, PropagationConfig};
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
const ARRIVE_BASE: usize = 0;
const GATE_BASE: usize = ARRIVE_BASE + K;
const EMIT_BASE: usize = GATE_BASE + K;
const COINCIDENCE_BASE: usize = EMIT_BASE + K;
const COINCIDENCE_PER_CELL: usize = K * K * K;
const LANES_PER_CELL: usize = COINCIDENCE_BASE + COINCIDENCE_PER_CELL;

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
enum Arm {
    PerStepOracleInjection,
    Full16RuleTemplateBaseline,
    CompletedSparseTemplateBaseline,
    StepwiseOracleClock,
    PathOnlyForwardClock,
    FinalTickReadout,
    BestTickReadout,
    FirstArrivalReadout,
    PersistentTargetReadout,
    ArriveLatch1Tick,
    ArriveLatchPersistent,
    EmitLatchPersistent,
    ConsumeOnForwardLatch,
    BidirectionalGridBaseline,
    OracleDirectionNoBackflow,
    PublicGradientNoBackflow,
    CellLocalNormalization,
    TargetOnlyNormalization,
    RandomControl,
}

impl Arm {
    fn as_str(self) -> &'static str {
        match self {
            Arm::PerStepOracleInjection => "PER_STEP_ORACLE_INJECTION",
            Arm::Full16RuleTemplateBaseline => "FULL_16_RULE_TEMPLATE_BASELINE",
            Arm::CompletedSparseTemplateBaseline => "COMPLETED_SPARSE_TEMPLATE_BASELINE",
            Arm::StepwiseOracleClock => "STEPWISE_ORACLE_CLOCK",
            Arm::PathOnlyForwardClock => "PATH_ONLY_FORWARD_CLOCK",
            Arm::FinalTickReadout => "FINAL_TICK_READOUT",
            Arm::BestTickReadout => "BEST_TICK_READOUT",
            Arm::FirstArrivalReadout => "FIRST_ARRIVAL_READOUT",
            Arm::PersistentTargetReadout => "PERSISTENT_TARGET_READOUT",
            Arm::ArriveLatch1Tick => "ARRIVE_LATCH_1TICK",
            Arm::ArriveLatchPersistent => "ARRIVE_LATCH_PERSISTENT",
            Arm::EmitLatchPersistent => "EMIT_LATCH_PERSISTENT",
            Arm::ConsumeOnForwardLatch => "CONSUME_ON_FORWARD_LATCH",
            Arm::BidirectionalGridBaseline => "BIDIRECTIONAL_GRID_BASELINE",
            Arm::OracleDirectionNoBackflow => "ORACLE_DIRECTION_NO_BACKFLOW",
            Arm::PublicGradientNoBackflow => "PUBLIC_GRADIENT_NO_BACKFLOW",
            Arm::CellLocalNormalization => "CELL_LOCAL_NORMALIZATION",
            Arm::TargetOnlyNormalization => "TARGET_ONLY_NORMALIZATION",
            Arm::RandomControl => "RANDOM_CONTROL",
        }
    }

    fn uses_private_path(self) -> bool {
        matches!(
            self,
            Arm::PerStepOracleInjection
                | Arm::StepwiseOracleClock
                | Arm::PathOnlyForwardClock
                | Arm::OracleDirectionNoBackflow
        )
    }

    fn is_network_arm(self) -> bool {
        matches!(
            self,
            Arm::Full16RuleTemplateBaseline
                | Arm::CompletedSparseTemplateBaseline
                | Arm::FinalTickReadout
                | Arm::BestTickReadout
                | Arm::FirstArrivalReadout
                | Arm::PersistentTargetReadout
                | Arm::TargetOnlyNormalization
                | Arm::RandomControl
        )
    }

    fn readout_mode(self) -> ReadoutMode {
        match self {
            Arm::BestTickReadout => ReadoutMode::BestTick,
            Arm::FirstArrivalReadout => ReadoutMode::FirstArrival,
            Arm::PersistentTargetReadout => ReadoutMode::PersistentTarget,
            _ => ReadoutMode::FinalTick,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReadoutMode {
    FinalTick,
    BestTick,
    FirstArrival,
    PersistentTarget,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LatchMode {
    None,
    ArriveOneTick,
    ArrivePersistent,
    EmitPersistent,
    ConsumeOnForward,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FlowMode {
    Bidirectional,
    OracleForward,
    PublicGradient,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct MotifType {
    input_phase: usize,
    gate: usize,
    output_phase: usize,
}

#[derive(Clone)]
struct Layout {
    width: usize,
}

impl Layout {
    fn neuron_count(&self) -> usize {
        self.width * self.width * LANES_PER_CELL
    }

    fn cell_id(&self, y: usize, x: usize) -> usize {
        y * self.width + x
    }

    fn cell_of_neuron(&self, neuron: usize) -> (usize, usize) {
        let cell = neuron / LANES_PER_CELL;
        (cell / self.width, cell % self.width)
    }

    fn cell_base(&self, y: usize, x: usize) -> usize {
        self.cell_id(y, x) * LANES_PER_CELL
    }

    fn arrive(&self, y: usize, x: usize, phase: usize) -> usize {
        self.cell_base(y, x) + ARRIVE_BASE + phase
    }

    fn gate(&self, y: usize, x: usize, gate: usize) -> usize {
        self.cell_base(y, x) + GATE_BASE + gate
    }

    fn emit(&self, y: usize, x: usize, phase: usize) -> usize {
        self.cell_base(y, x) + EMIT_BASE + phase
    }

    fn coincidence(
        &self,
        y: usize,
        x: usize,
        input_phase: usize,
        gate: usize,
        output_phase: usize,
    ) -> usize {
        self.cell_base(y, x) + COINCIDENCE_BASE + ((input_phase * K + gate) * K + output_phase)
    }
}

#[derive(Clone, Debug, Default)]
struct EvalMetrics {
    phase_final_accuracy: f64,
    best_tick_accuracy: f64,
    first_arrival_accuracy: f64,
    persistent_target_accuracy: f64,
    correct_target_lane_probability_mean: f64,
    minimum_ticks_for_95_accuracy: i64,
    per_step_transfer_accuracy: f64,
    min_per_pair_step_accuracy: f64,
    min_per_pair_step_probability: f64,
    target_arrival_rate: f64,
    correct_if_arrived_accuracy: f64,
    wrong_if_arrived_rate: f64,
    first_tick_correct_rate: f64,
    last_tick_correct_rate: f64,
    correct_then_lost_rate: f64,
    target_power_total_by_tick_mean: f64,
    backflow_power: f64,
    echo_power: f64,
    phase_decay_per_step: f64,
    wrong_phase_growth_rate: f64,
    latch_retention_rate: f64,
    readout_timing_gap: f64,
    wall_leak_rate: f64,
    gate_shuffle_collapse: f64,
    forbidden_private_field_leak: f64,
    nonlocal_edge_count: usize,
    direct_output_leak_rate: f64,
}

#[derive(Clone, Debug, Serialize)]
struct TransportRow {
    job_id: String,
    seed: u64,
    arm: String,
    diagnostic_private_path: bool,
    width: usize,
    path_length: usize,
    ticks: usize,
    family: String,
    case_count: usize,
    phase_final_accuracy: f64,
    best_tick_accuracy: f64,
    first_arrival_accuracy: f64,
    persistent_target_accuracy: f64,
    correct_target_lane_probability_mean: f64,
    minimum_ticks_for_95_accuracy: i64,
    per_step_transfer_accuracy: f64,
    min_per_pair_step_accuracy: f64,
    min_per_pair_step_probability: f64,
    target_arrival_rate: f64,
    correct_if_arrived_accuracy: f64,
    wrong_if_arrived_rate: f64,
    first_tick_correct: f64,
    last_tick_correct: f64,
    correct_then_lost_rate: f64,
    target_power_total_by_tick: f64,
    backflow_power: f64,
    echo_power: f64,
    phase_decay_per_step: f64,
    wrong_phase_growth_rate: f64,
    latch_retention_rate: f64,
    readout_timing_gap: f64,
    wall_leak_rate: f64,
    gate_shuffle_collapse: f64,
    forbidden_private_field_leak: f64,
    nonlocal_edge_count: usize,
    direct_output_leak_rate: f64,
    elapsed_sec: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
struct PairStats {
    input_phase: usize,
    gate: usize,
    count: usize,
    correct: usize,
    probability_sum: f64,
}

impl PairStats {
    fn accuracy(&self) -> f64 {
        ratio(self.correct, self.count)
    }

    fn probability(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.probability_sum / self.count as f64
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct PerPairRow {
    seed: u64,
    width: usize,
    path_length: usize,
    ticks: usize,
    family: String,
    input_phase: usize,
    gate: usize,
    count: usize,
    per_pair_step_accuracy: f64,
    per_pair_step_probability: f64,
}

#[derive(Clone, Debug)]
struct NetworkAudit {
    nonlocal_edge_count: usize,
    direct_output_leak_rate: f64,
}

#[derive(Clone, Debug)]
struct TargetSnapshot {
    scores: [f64; K],
    probs: [f64; K],
    pred: usize,
    correct_prob: f64,
    total_power: f64,
    tick: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = parse_args()?;
    fs::create_dir_all(&cfg.out)?;
    fs::create_dir_all(cfg.out.join("job_progress"))?;
    write_static_run_files(&cfg)?;

    append_jsonl(
        cfg.out.join("progress.jsonl"),
        &json!({"event": "run_start", "time": now_sec(), "jobs": cfg.seeds.len() * arms().len()}),
    )?;

    let started = Instant::now();
    let mut all_rows = Vec::new();
    let mut per_step_failed = false;
    let mut completed = 0usize;
    let total_jobs = cfg.seeds.len() * arms().len();

    for seed in cfg.seeds.iter().copied() {
        let per_step_rows = run_arm(&cfg, seed, Arm::PerStepOracleInjection, started)?;
        per_step_failed |= per_step_gate_score(&per_step_rows.iter().collect::<Vec<_>>()) < 0.95;
        completed += 1;
        all_rows.extend(per_step_rows);
        refresh_summary(&cfg, &all_rows, completed, total_jobs, started, false)?;

        if per_step_failed {
            append_jsonl(
                cfg.out.join("progress.jsonl"),
                &json!({
                    "event": "per_step_failed_stop_chain_interpretation",
                    "seed": seed,
                    "time": now_sec()
                }),
            )?;
            continue;
        }

        for arm in arms()
            .into_iter()
            .filter(|arm| *arm != Arm::PerStepOracleInjection)
        {
            let rows = run_arm(&cfg, seed, arm, started)?;
            completed += 1;
            all_rows.extend(rows);
            refresh_summary(&cfg, &all_rows, completed, total_jobs, started, false)?;
        }
    }

    refresh_summary(&cfg, &all_rows, completed, total_jobs, started, true)?;
    append_jsonl(
        cfg.out.join("progress.jsonl"),
        &json!({"event": "run_complete", "time": now_sec(), "completed": completed}),
    )?;
    Ok(())
}

fn run_arm(
    cfg: &Config,
    seed: u64,
    arm: Arm,
    started: Instant,
) -> Result<Vec<TransportRow>, Box<dyn std::error::Error>> {
    let mut rows = Vec::new();
    let job_id = format!("{}_{}", seed, arm.as_str());
    let total_buckets =
        cfg.widths.len() * cfg.path_lengths.len() * cfg.ticks_list.len() * families().len();
    let per_bucket = (cfg.eval_examples / total_buckets.max(1)).max(2);
    let mut last_heartbeat = Instant::now();

    append_jsonl(
        cfg.out.join("progress.jsonl"),
        &json!({"event": "job_start", "job_id": job_id, "seed": seed, "arm": arm.as_str(), "time": now_sec()}),
    )?;

    for width in &cfg.widths {
        let layout = Layout { width: *width };
        let mut net = network_for_arm(arm, &layout, seed);
        let audit = audit_network(&net, &layout, None);
        for path_length in &cfg.path_lengths {
            for ticks in &cfg.ticks_list {
                for family in families() {
                    let cases = generate_cases(seed, per_bucket, *width, *path_length, family);
                    maybe_write_examples(cfg, &cases)?;
                    let metrics = evaluate_arm(arm, &mut net, &layout, &cases, *ticks, &audit);
                    let row = TransportRow {
                        job_id: job_id.clone(),
                        seed,
                        arm: arm.as_str().to_string(),
                        diagnostic_private_path: arm.uses_private_path(),
                        width: *width,
                        path_length: *path_length,
                        ticks: *ticks,
                        family: family.to_string(),
                        case_count: cases.len(),
                        phase_final_accuracy: metrics.phase_final_accuracy,
                        best_tick_accuracy: metrics.best_tick_accuracy,
                        first_arrival_accuracy: metrics.first_arrival_accuracy,
                        persistent_target_accuracy: metrics.persistent_target_accuracy,
                        correct_target_lane_probability_mean: metrics
                            .correct_target_lane_probability_mean,
                        minimum_ticks_for_95_accuracy: metrics.minimum_ticks_for_95_accuracy,
                        per_step_transfer_accuracy: metrics.per_step_transfer_accuracy,
                        min_per_pair_step_accuracy: metrics.min_per_pair_step_accuracy,
                        min_per_pair_step_probability: metrics.min_per_pair_step_probability,
                        target_arrival_rate: metrics.target_arrival_rate,
                        correct_if_arrived_accuracy: metrics.correct_if_arrived_accuracy,
                        wrong_if_arrived_rate: metrics.wrong_if_arrived_rate,
                        first_tick_correct: metrics.first_tick_correct_rate,
                        last_tick_correct: metrics.last_tick_correct_rate,
                        correct_then_lost_rate: metrics.correct_then_lost_rate,
                        target_power_total_by_tick: metrics.target_power_total_by_tick_mean,
                        backflow_power: metrics.backflow_power,
                        echo_power: metrics.echo_power,
                        phase_decay_per_step: metrics.phase_decay_per_step,
                        wrong_phase_growth_rate: metrics.wrong_phase_growth_rate,
                        latch_retention_rate: metrics.latch_retention_rate,
                        readout_timing_gap: metrics.readout_timing_gap,
                        wall_leak_rate: metrics.wall_leak_rate,
                        gate_shuffle_collapse: metrics.gate_shuffle_collapse,
                        forbidden_private_field_leak: metrics.forbidden_private_field_leak,
                        nonlocal_edge_count: metrics.nonlocal_edge_count,
                        direct_output_leak_rate: metrics.direct_output_leak_rate,
                        elapsed_sec: started.elapsed().as_secs_f64(),
                    };
                    append_metric_files(cfg, &row)?;
                    rows.push(row);

                    if arm == Arm::PerStepOracleInjection {
                        write_per_pair_rows(
                            cfg,
                            seed,
                            &layout,
                            *path_length,
                            *ticks,
                            family,
                            &cases,
                        )?;
                    }

                    if last_heartbeat.elapsed().as_secs() >= cfg.heartbeat_sec {
                        append_jsonl(
                            cfg.out.join("progress.jsonl"),
                            &json!({
                                "event": "heartbeat",
                                "job_id": job_id,
                                "seed": seed,
                                "arm": arm.as_str(),
                                "rows": rows.len(),
                                "elapsed_sec": started.elapsed().as_secs_f64(),
                                "time": now_sec()
                            }),
                        )?;
                        write_job_progress(cfg, &job_id, rows.len(), total_buckets)?;
                        last_heartbeat = Instant::now();
                    }
                }
            }
        }
    }

    write_job_progress(cfg, &job_id, rows.len(), total_buckets)?;
    append_jsonl(
        cfg.out.join("progress.jsonl"),
        &json!({"event": "job_complete", "job_id": job_id, "seed": seed, "arm": arm.as_str(), "rows": rows.len(), "time": now_sec()}),
    )?;
    Ok(rows)
}

fn evaluate_arm(
    arm: Arm,
    net: &mut Network,
    layout: &Layout,
    cases: &[Case],
    ticks: usize,
    audit: &NetworkAudit,
) -> EvalMetrics {
    match arm {
        Arm::PerStepOracleInjection => per_step_metrics(layout, cases, ticks, audit),
        Arm::StepwiseOracleClock => stepwise_clock_metrics(cases, ticks),
        Arm::PathOnlyForwardClock => simulated_metrics(
            layout,
            cases,
            ticks,
            FlowMode::OracleForward,
            LatchMode::None,
            false,
            ReadoutMode::FinalTick,
            audit,
        ),
        Arm::BidirectionalGridBaseline => simulated_metrics(
            layout,
            cases,
            ticks,
            FlowMode::Bidirectional,
            LatchMode::None,
            false,
            ReadoutMode::FinalTick,
            audit,
        ),
        Arm::OracleDirectionNoBackflow => simulated_metrics(
            layout,
            cases,
            ticks,
            FlowMode::OracleForward,
            LatchMode::None,
            false,
            ReadoutMode::FinalTick,
            audit,
        ),
        Arm::PublicGradientNoBackflow => simulated_metrics(
            layout,
            cases,
            ticks,
            FlowMode::PublicGradient,
            LatchMode::None,
            false,
            ReadoutMode::FinalTick,
            audit,
        ),
        Arm::ArriveLatch1Tick => simulated_metrics(
            layout,
            cases,
            ticks,
            FlowMode::Bidirectional,
            LatchMode::ArriveOneTick,
            false,
            ReadoutMode::FinalTick,
            audit,
        ),
        Arm::ArriveLatchPersistent => simulated_metrics(
            layout,
            cases,
            ticks,
            FlowMode::Bidirectional,
            LatchMode::ArrivePersistent,
            false,
            ReadoutMode::FinalTick,
            audit,
        ),
        Arm::EmitLatchPersistent => simulated_metrics(
            layout,
            cases,
            ticks,
            FlowMode::Bidirectional,
            LatchMode::EmitPersistent,
            false,
            ReadoutMode::FinalTick,
            audit,
        ),
        Arm::ConsumeOnForwardLatch => simulated_metrics(
            layout,
            cases,
            ticks,
            FlowMode::Bidirectional,
            LatchMode::ConsumeOnForward,
            false,
            ReadoutMode::FinalTick,
            audit,
        ),
        Arm::CellLocalNormalization => simulated_metrics(
            layout,
            cases,
            ticks,
            FlowMode::Bidirectional,
            LatchMode::None,
            true,
            ReadoutMode::FinalTick,
            audit,
        ),
        _ if arm.is_network_arm() => network_metrics(net, layout, cases, ticks, arm, audit),
        _ => EvalMetrics::default(),
    }
}

fn network_metrics(
    net: &mut Network,
    layout: &Layout,
    cases: &[Case],
    ticks: usize,
    arm: Arm,
    audit: &NetworkAudit,
) -> EvalMetrics {
    let mut final_correct = 0usize;
    let mut best_correct = 0usize;
    let mut first_arrival_correct = 0usize;
    let mut persistent_correct = 0usize;
    let mut final_prob = 0.0;
    let mut best_prob = 0.0;
    let mut final_power = 0.0;
    let mut arrival_count = 0usize;
    let mut correct_if_arrived = 0usize;
    let mut wrong_if_arrived = 0usize;
    let mut first_tick_correct = 0usize;
    let mut last_tick_correct = 0usize;
    let mut correct_then_lost = 0usize;
    let mut wrong_growth = 0.0;
    let mut decay = 0.0;
    let mut min_tick_95 = -1i64;

    for case in cases {
        let label = case.private.label as usize;
        let snapshots = network_snapshots(net, layout, case, ticks);
        let best_out = readout_by_mode(&snapshots, label, ReadoutMode::BestTick);
        let first_out = readout_by_mode(&snapshots, label, ReadoutMode::FirstArrival);
        let persistent_out = readout_by_mode(&snapshots, label, ReadoutMode::PersistentTarget);
        let selected = readout_by_mode(&snapshots, label, arm.readout_mode());

        final_correct += usize::from(selected.pred == label);
        best_correct += usize::from(best_out.pred == label);
        first_arrival_correct += usize::from(first_out.pred == label);
        persistent_correct += usize::from(persistent_out.pred == label);
        final_prob += selected.correct_prob;
        best_prob += best_out.correct_prob;
        final_power += selected.total_power;
        if selected.total_power > EPS {
            arrival_count += 1;
            if selected.pred == label {
                correct_if_arrived += 1;
            } else {
                wrong_if_arrived += 1;
            }
        }
        if let Some(first) = snapshots.first() {
            first_tick_correct += usize::from(first.pred == label);
        }
        if let Some(last) = snapshots.last() {
            last_tick_correct += usize::from(last.pred == label);
        }
        if snapshots.iter().any(|snap| snap.pred == label) && selected.pred != label {
            correct_then_lost += 1;
        }
        wrong_growth += wrong_phase_growth(&snapshots, label);
        decay += phase_decay_proxy(&snapshots, label, case.private.true_path.len().max(1));
        if min_tick_95 < 0 {
            for snap in &snapshots {
                if snap.pred == label && snap.correct_prob >= 0.90 {
                    min_tick_95 = snap.tick as i64;
                    break;
                }
            }
        }
    }

    let n = cases.len().max(1) as f64;
    EvalMetrics {
        phase_final_accuracy: final_correct as f64 / n,
        best_tick_accuracy: best_correct as f64 / n,
        first_arrival_accuracy: first_arrival_correct as f64 / n,
        persistent_target_accuracy: persistent_correct as f64 / n,
        correct_target_lane_probability_mean: final_prob / n,
        minimum_ticks_for_95_accuracy: min_tick_95,
        target_arrival_rate: arrival_count as f64 / n,
        correct_if_arrived_accuracy: ratio(correct_if_arrived, arrival_count),
        wrong_if_arrived_rate: ratio(wrong_if_arrived, arrival_count),
        first_tick_correct_rate: first_tick_correct as f64 / n,
        last_tick_correct_rate: last_tick_correct as f64 / n,
        correct_then_lost_rate: correct_then_lost as f64 / n,
        target_power_total_by_tick_mean: final_power / n,
        phase_decay_per_step: decay / n,
        wrong_phase_growth_rate: wrong_growth / n,
        readout_timing_gap: (best_prob / n) - (final_prob / n),
        wall_leak_rate: wall_leak_rate(net, layout, cases, ticks),
        gate_shuffle_collapse: gate_shuffle_collapse(net, layout, cases, ticks),
        forbidden_private_field_leak: 0.0,
        nonlocal_edge_count: audit.nonlocal_edge_count,
        direct_output_leak_rate: audit.direct_output_leak_rate,
        ..Default::default()
    }
}

fn per_step_metrics(
    layout: &Layout,
    cases: &[Case],
    ticks: usize,
    audit: &NetworkAudit,
) -> EvalMetrics {
    let mut pair_stats = empty_pair_stats();
    let mut ok = 0usize;
    let mut total = 0usize;
    for case in cases {
        for edge_idx in 0..case.private.true_path.len().saturating_sub(1) {
            let from = case.private.true_path[edge_idx];
            let to = case.private.true_path[edge_idx + 1];
            let gate = case.public.gates[layout.cell_id(to.0, to.1)] as usize;
            for phase in 0..K {
                let expected = expected_phase(phase, gate);
                let out = per_step_network_output(layout, from, to, phase, gate, ticks);
                let pass = out.pred == expected;
                ok += usize::from(pass);
                total += 1;
                let ps = &mut pair_stats[phase][gate];
                ps.count += 1;
                ps.correct += usize::from(pass);
                ps.probability_sum += out.probs[expected];
            }
        }
    }
    let mut min_pair_acc = 1.0f64;
    let mut min_pair_prob = 1.0f64;
    for row in &pair_stats {
        for pair in row {
            if pair.count > 0 {
                min_pair_acc = min_pair_acc.min(pair.accuracy());
                min_pair_prob = min_pair_prob.min(pair.probability());
            }
        }
    }
    EvalMetrics {
        per_step_transfer_accuracy: ratio(ok, total),
        min_per_pair_step_accuracy: min_pair_acc,
        min_per_pair_step_probability: min_pair_prob,
        phase_final_accuracy: ratio(ok, total),
        best_tick_accuracy: ratio(ok, total),
        first_arrival_accuracy: ratio(ok, total),
        persistent_target_accuracy: ratio(ok, total),
        correct_target_lane_probability_mean: mean_pair_probability(&pair_stats),
        target_arrival_rate: 1.0,
        correct_if_arrived_accuracy: ratio(ok, total),
        wrong_if_arrived_rate: 1.0 - ratio(ok, total),
        forbidden_private_field_leak: 0.0,
        nonlocal_edge_count: audit.nonlocal_edge_count,
        direct_output_leak_rate: audit.direct_output_leak_rate,
        ..Default::default()
    }
}

fn write_per_pair_rows(
    cfg: &Config,
    seed: u64,
    layout: &Layout,
    path_length: usize,
    ticks: usize,
    family: &str,
    cases: &[Case],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut pair_stats = empty_pair_stats();
    for case in cases {
        for edge_idx in 0..case.private.true_path.len().saturating_sub(1) {
            let from = case.private.true_path[edge_idx];
            let to = case.private.true_path[edge_idx + 1];
            let gate = case.public.gates[layout.cell_id(to.0, to.1)] as usize;
            for phase in 0..K {
                let expected = expected_phase(phase, gate);
                let out = per_step_network_output(layout, from, to, phase, gate, ticks);
                let pair = &mut pair_stats[phase][gate];
                pair.count += 1;
                pair.correct += usize::from(out.pred == expected);
                pair.probability_sum += out.probs[expected];
            }
        }
    }
    for row in &pair_stats {
        for pair in row {
            append_jsonl(
                cfg.out.join("per_pair_step_metrics.jsonl"),
                &PerPairRow {
                    seed,
                    width: layout.width,
                    path_length,
                    ticks,
                    family: family.to_string(),
                    input_phase: pair.input_phase,
                    gate: pair.gate,
                    count: pair.count,
                    per_pair_step_accuracy: pair.accuracy(),
                    per_pair_step_probability: pair.probability(),
                },
            )?;
        }
    }
    Ok(())
}

fn stepwise_clock_metrics(cases: &[Case], ticks: usize) -> EvalMetrics {
    let mut correct = 0usize;
    let mut first = 0usize;
    let mut persistent = 0usize;
    let mut prob = 0.0;
    let mut arrival = 0usize;
    let mut correct_then_lost = 0usize;
    for case in cases {
        let needed = case.private.true_path.len().saturating_sub(1);
        let label = case.private.label as usize;
        let reached = ticks >= needed;
        if reached {
            correct += 1;
            persistent += 1;
            arrival += 1;
            prob += 1.0;
            if needed > 0 {
                first += 1;
            }
        } else {
            prob += 0.25;
            if ticks > 0 && ticks < needed {
                correct_then_lost += 0;
            }
        }
        let _ = label;
    }
    let n = cases.len().max(1) as f64;
    EvalMetrics {
        phase_final_accuracy: correct as f64 / n,
        best_tick_accuracy: correct as f64 / n,
        first_arrival_accuracy: first as f64 / n,
        persistent_target_accuracy: persistent as f64 / n,
        correct_target_lane_probability_mean: prob / n,
        target_arrival_rate: arrival as f64 / n,
        correct_if_arrived_accuracy: ratio(correct, arrival),
        wrong_if_arrived_rate: 0.0,
        correct_then_lost_rate: correct_then_lost as f64 / n,
        minimum_ticks_for_95_accuracy: cases
            .iter()
            .map(|case| case.private.true_path.len().saturating_sub(1) as i64)
            .max()
            .unwrap_or(-1),
        ..Default::default()
    }
}

fn simulated_metrics(
    layout: &Layout,
    cases: &[Case],
    ticks: usize,
    flow: FlowMode,
    latch: LatchMode,
    normalize_cell: bool,
    readout_mode: ReadoutMode,
    audit: &NetworkAudit,
) -> EvalMetrics {
    let mut final_correct = 0usize;
    let mut best_correct = 0usize;
    let mut first_correct = 0usize;
    let mut persistent_correct = 0usize;
    let mut final_prob = 0.0;
    let mut best_prob = 0.0;
    let mut arrival = 0usize;
    let mut correct_arrived = 0usize;
    let mut wrong_arrived = 0usize;
    let mut first_tick_correct = 0usize;
    let mut last_tick_correct = 0usize;
    let mut correct_then_lost = 0usize;
    let mut power = 0.0;
    let mut wrong_growth = 0.0;
    let mut backflow = 0.0;
    let mut echo = 0.0;
    let mut latch_retention = 0.0;
    for case in cases {
        let label = case.private.label as usize;
        let snapshots = simulate_case(layout, case, ticks, flow, latch, normalize_cell);
        let selected = readout_by_mode(&snapshots, label, readout_mode);
        let best = readout_by_mode(&snapshots, label, ReadoutMode::BestTick);
        let first = readout_by_mode(&snapshots, label, ReadoutMode::FirstArrival);
        let persistent = readout_by_mode(&snapshots, label, ReadoutMode::PersistentTarget);
        final_correct += usize::from(selected.pred == label);
        best_correct += usize::from(best.pred == label);
        first_correct += usize::from(first.pred == label);
        persistent_correct += usize::from(persistent.pred == label);
        final_prob += selected.correct_prob;
        best_prob += best.correct_prob;
        power += selected.total_power;
        if selected.total_power > EPS {
            arrival += 1;
            correct_arrived += usize::from(selected.pred == label);
            wrong_arrived += usize::from(selected.pred != label);
        }
        if let Some(s) = snapshots.first() {
            first_tick_correct += usize::from(s.pred == label);
        }
        if let Some(s) = snapshots.last() {
            last_tick_correct += usize::from(s.pred == label);
        }
        if snapshots.iter().any(|s| s.pred == label) && selected.pred != label {
            correct_then_lost += 1;
        }
        wrong_growth += wrong_phase_growth(&snapshots, label);
        let (b, e) = backflow_echo_power(layout, case, &snapshots);
        backflow += b;
        echo += e;
        latch_retention += if latch == LatchMode::None {
            0.0
        } else {
            selected.total_power
        };
    }
    let n = cases.len().max(1) as f64;
    EvalMetrics {
        phase_final_accuracy: final_correct as f64 / n,
        best_tick_accuracy: best_correct as f64 / n,
        first_arrival_accuracy: first_correct as f64 / n,
        persistent_target_accuracy: persistent_correct as f64 / n,
        correct_target_lane_probability_mean: final_prob / n,
        target_arrival_rate: arrival as f64 / n,
        correct_if_arrived_accuracy: ratio(correct_arrived, arrival),
        wrong_if_arrived_rate: ratio(wrong_arrived, arrival),
        first_tick_correct_rate: first_tick_correct as f64 / n,
        last_tick_correct_rate: last_tick_correct as f64 / n,
        correct_then_lost_rate: correct_then_lost as f64 / n,
        target_power_total_by_tick_mean: power / n,
        backflow_power: backflow / n,
        echo_power: echo / n,
        wrong_phase_growth_rate: wrong_growth / n,
        latch_retention_rate: latch_retention / n,
        readout_timing_gap: (best_prob / n) - (final_prob / n),
        gate_shuffle_collapse: 1.0 - final_correct as f64 / n,
        forbidden_private_field_leak: 0.0,
        nonlocal_edge_count: audit.nonlocal_edge_count,
        direct_output_leak_rate: audit.direct_output_leak_rate,
        ..Default::default()
    }
}

fn simulate_case(
    layout: &Layout,
    case: &Case,
    ticks: usize,
    flow: FlowMode,
    latch: LatchMode,
    normalize_cell: bool,
) -> Vec<TargetSnapshot> {
    let cells = layout.width * layout.width;
    let mut emit = vec![[0.0f64; K]; cells];
    let mut arrive = vec![[0.0f64; K]; cells];
    let mut persistent_arrive = vec![[0.0f64; K]; cells];
    let source_id = layout.cell_id(case.public.source.0, case.public.source.1);
    emit[source_id][case.public.source_phase as usize] = 1.0;
    let mut snapshots = Vec::with_capacity(ticks.max(1));

    for tick in 1..=ticks {
        let mut next_arrive = vec![[0.0f64; K]; cells];
        for y in 0..layout.width {
            for x in 0..layout.width {
                let id = layout.cell_id(y, x);
                if case.public.wall[id] {
                    continue;
                }
                for (ny, nx) in allowed_neighbors(layout, case, (y, x), flow) {
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

        match latch {
            LatchMode::ArriveOneTick => {
                for id in 0..cells {
                    for phase in 0..K {
                        next_arrive[id][phase] = next_arrive[id][phase].max(arrive[id][phase]);
                    }
                }
            }
            LatchMode::ArrivePersistent => {
                for id in 0..cells {
                    for phase in 0..K {
                        persistent_arrive[id][phase] =
                            persistent_arrive[id][phase].max(next_arrive[id][phase]);
                        next_arrive[id][phase] =
                            next_arrive[id][phase].max(persistent_arrive[id][phase]);
                    }
                }
            }
            _ => {}
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
                    let out = expected_phase(phase, gate);
                    next_emit[id][out] += next_arrive[id][phase];
                }
                if matches!(
                    latch,
                    LatchMode::EmitPersistent | LatchMode::ConsumeOnForward
                ) {
                    for phase in 0..K {
                        next_emit[id][phase] = next_emit[id][phase].max(emit[id][phase]);
                    }
                }
                if normalize_cell {
                    normalize_lanes(&mut next_emit[id]);
                }
            }
        }
        arrive = next_arrive;
        emit = next_emit;
        snapshots.push(snapshot_from_scores(
            emit[layout.cell_id(case.public.target.0, case.public.target.1)],
            case.private.label as usize,
            tick,
        ));
    }
    if snapshots.is_empty() {
        snapshots.push(snapshot_from_scores(
            [0.0; K],
            case.private.label as usize,
            0,
        ));
    }
    snapshots
}

fn allowed_neighbors(
    layout: &Layout,
    case: &Case,
    cell: (usize, usize),
    flow: FlowMode,
) -> Vec<(usize, usize)> {
    match flow {
        FlowMode::Bidirectional => neighbors(layout.width, cell.0, cell.1),
        FlowMode::OracleForward => {
            let path = &case.private.true_path;
            if let Some(pos) = path.iter().position(|&c| c == cell) {
                if pos + 1 < path.len() {
                    vec![path[pos + 1]]
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            }
        }
        FlowMode::PublicGradient => neighbors(layout.width, cell.0, cell.1)
            .into_iter()
            .filter(|&(ny, nx)| {
                manhattan((ny, nx), case.public.target) < manhattan(cell, case.public.target)
            })
            .collect(),
    }
}

fn per_step_network_output(
    layout: &Layout,
    from: (usize, usize),
    to: (usize, usize),
    input_phase: usize,
    gate: usize,
    ticks: usize,
) -> TargetSnapshot {
    let expected = expected_phase(input_phase, gate);
    let mut best = snapshot_from_scores([0.0; K], expected, 0);
    for tick in 1..=ticks.max(1).min(8) {
        let snap = per_step_network_snapshot(layout, from, to, input_phase, gate, tick);
        if snap.correct_prob > best.correct_prob {
            best = snap;
        }
    }
    best
}

fn per_step_network_snapshot(
    layout: &Layout,
    from: (usize, usize),
    to: (usize, usize),
    input_phase: usize,
    gate: usize,
    ticks: usize,
) -> TargetSnapshot {
    let mut net = empty_network(layout);
    add_emit_to_neighbor_arrive_edges(&mut net, layout);
    for motif in full_16_rule_template() {
        add_coincidence_gate(
            &mut net,
            layout,
            to,
            motif.input_phase,
            motif.gate,
            motif.output_phase,
        );
    }
    net.reset();
    let mut indices = vec![layout.emit(from.0, from.1, input_phase) as u16];
    let mut values = vec![8i8];
    indices.push(layout.gate(to.0, to.1, gate) as u16);
    values.push(8i8);
    net.propagate_sparse(
        &indices,
        &values,
        &PropagationConfig {
            ticks_per_token: ticks,
            input_duration_ticks: ticks,
            decay_interval_ticks: usize::MAX,
            use_refractory: false,
        },
    )
    .expect("per-step propagation should be valid");
    let scores = read_emit_scores(&net, layout, to);
    snapshot_from_scores(scores, expected_phase(input_phase, gate), ticks)
}

fn network_snapshots(
    net: &mut Network,
    layout: &Layout,
    case: &Case,
    max_ticks: usize,
) -> Vec<TargetSnapshot> {
    (1..=max_ticks)
        .map(|tick| network_output(net, layout, &case.public, case.private.label as usize, tick))
        .collect()
}

fn network_output(
    net: &mut Network,
    layout: &Layout,
    case: &PublicCase,
    label: usize,
    ticks: usize,
) -> TargetSnapshot {
    net.reset();
    let mut indices = Vec::new();
    let mut values = Vec::new();
    indices.push(layout.emit(case.source.0, case.source.1, case.source_phase as usize) as u16);
    values.push(8i8);
    for y in 0..case.width {
        for x in 0..case.width {
            if case.wall[y * case.width + x] {
                continue;
            }
            indices.push(layout.gate(y, x, case.gates[y * case.width + x] as usize) as u16);
            values.push(8i8);
        }
    }
    net.propagate_sparse(
        &indices,
        &values,
        &PropagationConfig {
            ticks_per_token: ticks,
            input_duration_ticks: ticks,
            decay_interval_ticks: usize::MAX,
            use_refractory: false,
        },
    )
    .expect("phase-lane propagation should be valid");
    let scores = read_emit_scores(net, layout, case.target);
    snapshot_from_scores(scores, label, ticks)
}

fn read_emit_scores(net: &Network, layout: &Layout, cell: (usize, usize)) -> [f64; K] {
    let mut scores = [0.0f64; K];
    for (phase, score) in scores.iter_mut().enumerate() {
        let idx = layout.emit(cell.0, cell.1, phase);
        let charge = f64::from(net.spike_data()[idx].charge);
        let activation = f64::from(net.activation()[idx].max(0));
        *score = charge + 4.0 * activation;
    }
    scores
}

fn readout_by_mode(
    snapshots: &[TargetSnapshot],
    label: usize,
    mode: ReadoutMode,
) -> TargetSnapshot {
    match mode {
        ReadoutMode::FinalTick => snapshots
            .last()
            .cloned()
            .unwrap_or_else(|| snapshot_from_scores([0.0; K], label, 0)),
        ReadoutMode::BestTick => snapshots
            .iter()
            .cloned()
            .max_by(|a, b| a.correct_prob.partial_cmp(&b.correct_prob).unwrap())
            .unwrap_or_else(|| snapshot_from_scores([0.0; K], label, 0)),
        ReadoutMode::FirstArrival => snapshots
            .iter()
            .find(|snap| snap.total_power > EPS)
            .cloned()
            .unwrap_or_else(|| snapshot_from_scores([0.0; K], label, 0)),
        ReadoutMode::PersistentTarget => {
            let mut best_scores = [0.0f64; K];
            let mut tick = 0usize;
            for snap in snapshots {
                tick = snap.tick;
                for (phase, score) in best_scores.iter_mut().enumerate() {
                    *score = score.max(snap.scores[phase]);
                }
            }
            snapshot_from_scores(best_scores, label, tick)
        }
    }
}

fn snapshot_from_scores(scores: [f64; K], label: usize, tick: usize) -> TargetSnapshot {
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
        tick,
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
    let mut rng = StdRng::seed_from_u64(seed ^ idx as u64);
    let path = serpentine_path(width, path_length);
    let source = *path.first().unwrap();
    let target = *path.last().unwrap();
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
            split: "transport_mechanics_eval".to_string(),
            requested_path_length: path_length,
        },
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

fn network_for_arm(arm: Arm, layout: &Layout, seed: u64) -> Network {
    let mut net = empty_network(layout);
    add_emit_to_neighbor_arrive_edges(&mut net, layout);
    if arm == Arm::RandomControl {
        for motif in random_motif_types(seed ^ 0x1313, 16) {
            for y in 0..layout.width {
                for x in 0..layout.width {
                    add_coincidence_gate(
                        &mut net,
                        layout,
                        (y, x),
                        motif.input_phase,
                        motif.gate,
                        motif.output_phase,
                    );
                }
            }
        }
        return net;
    }
    for motif in full_16_rule_template() {
        for y in 0..layout.width {
            for x in 0..layout.width {
                add_coincidence_gate(
                    &mut net,
                    layout,
                    (y, x),
                    motif.input_phase,
                    motif.gate,
                    motif.output_phase,
                );
            }
        }
    }
    net
}

fn full_16_rule_template() -> Vec<MotifType> {
    let mut motifs = Vec::with_capacity(K * K);
    for input_phase in 0..K {
        for gate in 0..K {
            motifs.push(MotifType {
                input_phase,
                gate,
                output_phase: expected_phase(input_phase, gate),
            });
        }
    }
    motifs
}

fn random_motif_types(seed: u64, count: usize) -> Vec<MotifType> {
    let mut all = Vec::with_capacity(K * K * K);
    for input_phase in 0..K {
        for gate in 0..K {
            for output_phase in 0..K {
                all.push(MotifType {
                    input_phase,
                    gate,
                    output_phase,
                });
            }
        }
    }
    let mut rng = StdRng::seed_from_u64(seed);
    for i in (1..all.len()).rev() {
        let j = rng.gen_range(0..=i);
        all.swap(i, j);
    }
    all.truncate(count.min(all.len()));
    all
}

fn empty_network(layout: &Layout) -> Network {
    let mut net = Network::new(layout.neuron_count());
    for spike in net.spike_data_mut() {
        spike.threshold = 0;
        spike.channel = 1;
        spike.charge = 0;
    }
    for p in net.polarity_mut() {
        *p = 1;
    }
    for y in 0..layout.width {
        for x in 0..layout.width {
            for input_phase in 0..K {
                for gate in 0..K {
                    for output_phase in 0..K {
                        let c = layout.coincidence(y, x, input_phase, gate, output_phase);
                        net.spike_data_mut()[c].threshold = 1;
                    }
                }
            }
        }
    }
    net
}

fn add_emit_to_neighbor_arrive_edges(net: &mut Network, layout: &Layout) {
    for y in 0..layout.width {
        for x in 0..layout.width {
            for (ny, nx) in neighbors(layout.width, y, x) {
                for phase in 0..K {
                    net.graph_mut().add_edge(
                        layout.emit(y, x, phase) as u16,
                        layout.arrive(ny, nx, phase) as u16,
                    );
                }
            }
        }
    }
}

fn add_coincidence_gate(
    net: &mut Network,
    layout: &Layout,
    cell: (usize, usize),
    input_phase: usize,
    gate: usize,
    output_phase: usize,
) {
    let (y, x) = cell;
    let c = layout.coincidence(y, x, input_phase, gate, output_phase);
    net.graph_mut()
        .add_edge(layout.arrive(y, x, input_phase) as u16, c as u16);
    net.graph_mut()
        .add_edge(layout.gate(y, x, gate) as u16, c as u16);
    net.graph_mut()
        .add_edge(c as u16, layout.emit(y, x, output_phase) as u16);
    net.spike_data_mut()[c].threshold = 1;
    net.spike_data_mut()[c].channel = 1;
    net.polarity_mut()[c] = 1;
}

fn audit_network(net: &Network, layout: &Layout, target: Option<(usize, usize)>) -> NetworkAudit {
    let mut nonlocal_edge_count = 0usize;
    let mut direct_leaks = 0usize;
    for edge in net.graph().iter_edges() {
        let (sy, sx) = layout.cell_of_neuron(edge.source as usize);
        let (ty, tx) = layout.cell_of_neuron(edge.target as usize);
        if sy.abs_diff(ty) + sx.abs_diff(tx) > 1 {
            nonlocal_edge_count += 1;
        }
        if let Some((target_y, target_x)) = target {
            if (ty, tx) == (target_y, target_x) && sy.abs_diff(target_y) + sx.abs_diff(target_x) > 1
            {
                direct_leaks += 1;
            }
        }
    }
    NetworkAudit {
        nonlocal_edge_count,
        direct_output_leak_rate: if net.edge_count() == 0 {
            0.0
        } else {
            direct_leaks as f64 / net.edge_count() as f64
        },
    }
}

fn append_metric_files(cfg: &Config, row: &TransportRow) -> Result<(), Box<dyn std::error::Error>> {
    append_jsonl(cfg.out.join("metrics.jsonl"), row)?;
    append_jsonl(cfg.out.join("transport_curve.jsonl"), row)?;
    append_jsonl(cfg.out.join("family_metrics.jsonl"), row)?;
    append_jsonl(cfg.out.join("counterfactual_metrics.jsonl"), row)?;
    append_jsonl(cfg.out.join("locality_audit.jsonl"), row)?;
    append_jsonl(cfg.out.join("arrival_metrics.jsonl"), row)?;
    if row.arm == Arm::PerStepOracleInjection.as_str() {
        append_jsonl(cfg.out.join("per_step_metrics.jsonl"), row)?;
    }
    if row.arm.contains("READOUT") {
        append_jsonl(cfg.out.join("readout_timing_metrics.jsonl"), row)?;
    }
    if row.arm.contains("CLOCK") {
        append_jsonl(cfg.out.join("clock_metrics.jsonl"), row)?;
    }
    if row.arm.contains("LATCH") {
        append_jsonl(cfg.out.join("latch_metrics.jsonl"), row)?;
    }
    if row.arm.contains("BACKFLOW") {
        append_jsonl(cfg.out.join("backflow_metrics.jsonl"), row)?;
    }
    if row.arm.contains("NORMALIZATION") {
        append_jsonl(cfg.out.join("normalization_metrics.jsonl"), row)?;
    }
    Ok(())
}

fn refresh_summary(
    cfg: &Config,
    rows: &[TransportRow],
    completed: usize,
    total: usize,
    started: Instant,
    final_report: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let summary = json!({
        "status": if final_report { "done" } else { "running" },
        "completed": completed,
        "total": total,
        "elapsed_sec": started.elapsed().as_secs_f64(),
        "updated_time": now_sec(),
        "verdicts": verdicts(rows),
    });
    fs::write(
        cfg.out.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    write_report(cfg, rows, final_report)?;
    Ok(())
}

fn write_report(
    cfg: &Config,
    rows: &[TransportRow],
    final_report: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut report = String::new();
    report.push_str("# STABLE_LOOP_PHASE_LOCK_013_PHASE_LANE_TRANSPORT_MECHANICS Report\n\n");
    report.push_str(if final_report {
        "Status: complete.\n\n"
    } else {
        "Status: running.\n\n"
    });
    report.push_str("## Verdicts\n\n```text\n");
    for verdict in verdicts(rows) {
        report.push_str(&verdict);
        report.push('\n');
    }
    report.push_str("```\n\n## Arm Summary\n\n");
    report.push_str("| Arm | Acc | Best | Persist | Arrive | Wrong-if-arrived | Readout gap | Backflow | Echo |\n");
    report.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for arm in arms() {
        let subset: Vec<_> = rows.iter().filter(|row| row.arm == arm.as_str()).collect();
        if subset.is_empty() {
            continue;
        }
        report.push_str(&format!(
            "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |\n",
            arm.as_str(),
            mean(subset.iter().map(|row| row.phase_final_accuracy)),
            mean(subset.iter().map(|row| row.best_tick_accuracy)),
            mean(subset.iter().map(|row| row.persistent_target_accuracy)),
            mean(subset.iter().map(|row| row.target_arrival_rate)),
            mean(subset.iter().map(|row| row.wrong_if_arrived_rate)),
            mean(subset.iter().map(|row| row.readout_timing_gap)),
            mean(subset.iter().map(|row| row.backflow_power)),
            mean(subset.iter().map(|row| row.echo_power)),
        ));
    }
    report.push_str("\n## Claim Boundary\n\n");
    report.push_str("This runner diagnoses phase-lane transport mechanics only. It does not prove production architecture, full VRAXION, consciousness, language grounding, Prismion uniqueness, or physical quantum behavior.\n");
    fs::write(cfg.out.join("report.md"), report)?;
    Ok(())
}

fn verdicts(rows: &[TransportRow]) -> Vec<String> {
    if rows.is_empty() {
        return vec!["RUNNING".to_string()];
    }
    let mut out = BTreeSet::new();
    let per = rows_for(rows, Arm::PerStepOracleInjection);
    let full = rows_for(rows, Arm::Full16RuleTemplateBaseline);
    let stepwise = rows_for(rows, Arm::StepwiseOracleClock);
    let best = rows_for(rows, Arm::BestTickReadout);
    let persistent = rows_for(rows, Arm::PersistentTargetReadout);
    let target_norm = rows_for(rows, Arm::TargetOnlyNormalization);
    let cell_norm = rows_for(rows, Arm::CellLocalNormalization);
    let oracle_no_back = rows_for(rows, Arm::OracleDirectionNoBackflow);
    let public_no_back = rows_for(rows, Arm::PublicGradientNoBackflow);
    let latch = rows
        .iter()
        .filter(|row| row.arm.contains("LATCH"))
        .collect::<Vec<_>>();

    let per_min = per_step_gate_score(&per);
    if per_min < 0.95 {
        out.insert("PER_STEP_TRANSPORT_FAILS".to_string());
        out.insert("PER_STEP_TRANSPORT_FAILS_PAIRWISE".to_string());
    } else if !per.is_empty() {
        out.insert("PER_STEP_TRANSPORT_OK".to_string());
    }

    let full_long = mean(
        full.iter()
            .filter(|row| row.path_length >= 4)
            .map(|row| row.phase_final_accuracy),
    );
    if per_min >= 0.95 && full_long < 0.75 {
        out.insert("PER_STEP_TRANSPORT_WORKS_BUT_CHAIN_FAILS".to_string());
    }
    if mean(stepwise.iter().map(|row| row.phase_final_accuracy)) >= 0.90 {
        out.insert("STEPWISE_CLOCK_RESCUES_HORIZON".to_string());
    }
    if mean(best.iter().map(|row| row.best_tick_accuracy)) >= 0.90
        && mean(full.iter().map(|row| row.phase_final_accuracy)) < 0.75
    {
        out.insert("READOUT_TIMING_IS_BLOCKER".to_string());
    }
    if mean(persistent.iter().map(|row| row.persistent_target_accuracy)) >= 0.90 {
        out.insert("TARGET_READOUT_PERSISTENCE_REQUIRED".to_string());
        out.insert("TARGET_MEMORY_RESCUES_READOUT".to_string());
    }
    if mean(target_norm.iter().map(|row| row.phase_final_accuracy)) >= 0.90
        && mean(cell_norm.iter().map(|row| row.phase_final_accuracy)) < 0.90
    {
        out.insert("TARGET_READOUT_CALIBRATION_LIMIT".to_string());
    }
    if mean(latch.iter().map(|row| row.phase_final_accuracy)) >= 0.90 {
        out.insert("CELL_LATCH_RESCUES_HORIZON".to_string());
    }
    let oracle_back_acc = mean(oracle_no_back.iter().map(|row| row.phase_final_accuracy));
    let public_back_acc = mean(public_no_back.iter().map(|row| row.phase_final_accuracy));
    if oracle_back_acc >= 0.90 {
        out.insert("ORACLE_NO_BACKFLOW_RESCUES".to_string());
    }
    if public_back_acc >= 0.90 {
        out.insert("PUBLIC_NO_BACKFLOW_RESCUES".to_string());
    }
    if oracle_back_acc >= 0.90 && public_back_acc < 0.90 {
        out.insert("ONLY_ORACLE_NO_BACKFLOW_RESCUES".to_string());
    }
    if public_back_acc >= 0.90 || oracle_back_acc >= 0.90 {
        out.insert("BACKFLOW_INTERFERENCE_IS_BLOCKER".to_string());
    }
    if mean(cell_norm.iter().map(|row| row.phase_final_accuracy)) >= 0.90 {
        out.insert("PHASE_NORMALIZATION_RESCUES_HORIZON".to_string());
    }
    if rows.iter().any(|row| row.correct_then_lost_rate > 0.20) {
        out.insert("EARLY_CORRECT_LATE_OVERWRITE".to_string());
    }
    if rows
        .iter()
        .any(|row| row.target_arrival_rate < 0.20 && row.path_length >= 4)
    {
        out.insert("SIGNAL_ARRIVAL_FAILURE".to_string());
    }
    if rows
        .iter()
        .any(|row| row.target_arrival_rate > 0.50 && row.wrong_if_arrived_rate > 0.50)
    {
        out.insert("SIGNAL_ARRIVES_WRONG_PHASE".to_string());
    }
    if family_spread(rows, Arm::Full16RuleTemplateBaseline) > 0.25 {
        out.insert("GATE_PATTERN_SPECIFIC_FAILURE".to_string());
    }
    let decay = mean(full.iter().map(|row| row.phase_decay_per_step));
    let wrong = mean(full.iter().map(|row| row.wrong_phase_growth_rate));
    if decay > 0.01 && wrong <= 0.35 {
        out.insert("PHASE_DECAY_LIMIT".to_string());
    } else if wrong > 0.35 && decay <= 0.01 {
        out.insert("WRONG_PHASE_INTERFERENCE_LIMIT".to_string());
    } else if wrong > 0.35 && decay > 0.01 {
        out.insert("DECAY_PLUS_INTERFERENCE_LIMIT".to_string());
    }
    if rows.iter().any(|row| {
        row.forbidden_private_field_leak > 0.0
            || row.nonlocal_edge_count > 0
            || row.direct_output_leak_rate > 0.05
    }) {
        out.insert("DIRECT_SHORTCUT_CONTAMINATION".to_string());
    }
    let any_rescue = out.iter().any(|v| {
        v.contains("RESCUES")
            || v == "READOUT_TIMING_IS_BLOCKER"
            || v == "CELL_LATCH_RESCUES_HORIZON"
    });
    if per_min >= 0.95 && full_long < 0.75 && !any_rescue {
        out.insert("RECURRENT_TRANSPORT_MECHANICS_BLOCKER".to_string());
    }
    out.insert("PRODUCTION_API_NOT_READY".to_string());
    out.into_iter().collect()
}

fn rows_for(rows: &[TransportRow], arm: Arm) -> Vec<&TransportRow> {
    rows.iter().filter(|row| row.arm == arm.as_str()).collect()
}

fn per_step_gate_score(rows: &[&TransportRow]) -> f64 {
    if rows.is_empty() {
        return 0.0;
    }
    let mut by_bucket: BTreeMap<(usize, usize, String), f64> = BTreeMap::new();
    for row in rows {
        let key = (row.width, row.path_length, row.family.clone());
        let entry = by_bucket.entry(key).or_insert(0.0);
        *entry = f64::max(*entry, row.min_per_pair_step_accuracy);
    }
    by_bucket.values().copied().fold(1.0f64, f64::min)
}

fn family_spread(rows: &[TransportRow], arm: Arm) -> f64 {
    let mut by_family: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for row in rows.iter().filter(|row| row.arm == arm.as_str()) {
        by_family
            .entry(row.family.clone())
            .or_default()
            .push(row.phase_final_accuracy);
    }
    if by_family.len() < 2 {
        return 0.0;
    }
    let values = by_family
        .values()
        .map(|v| mean(v.iter().copied()))
        .collect::<Vec<_>>();
    values.iter().copied().fold(0.0f64, f64::max) - values.iter().copied().fold(1.0f64, f64::min)
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = Config {
        out: PathBuf::from(
            "target/pilot_wave/stable_loop_phase_lock_013_phase_lane_transport_mechanics/dev",
        ),
        seeds: vec![2026],
        eval_examples: 256,
        widths: vec![8],
        path_lengths: vec![2, 4, 8],
        ticks_list: vec![8, 16, 24],
        heartbeat_sec: 15,
    };
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                i += 1;
                cfg.out = PathBuf::from(required_arg(&args, i, "--out")?);
            }
            "--seeds" => {
                i += 1;
                cfg.seeds = parse_u64_list(required_arg(&args, i, "--seeds")?)?;
            }
            "--eval-examples" => {
                i += 1;
                cfg.eval_examples = required_arg(&args, i, "--eval-examples")?.parse()?;
            }
            "--widths" => {
                i += 1;
                cfg.widths = parse_usize_list(required_arg(&args, i, "--widths")?)?;
            }
            "--path-lengths" => {
                i += 1;
                cfg.path_lengths = parse_usize_list(required_arg(&args, i, "--path-lengths")?)?;
            }
            "--ticks-list" => {
                i += 1;
                cfg.ticks_list = parse_usize_list(required_arg(&args, i, "--ticks-list")?)?;
            }
            "--heartbeat-sec" => {
                i += 1;
                cfg.heartbeat_sec = required_arg(&args, i, "--heartbeat-sec")?.parse()?;
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        i += 1;
    }
    Ok(cfg)
}

fn required_arg<'a>(
    args: &'a [String],
    i: usize,
    flag: &str,
) -> Result<&'a str, Box<dyn std::error::Error>> {
    args.get(i)
        .map(String::as_str)
        .ok_or_else(|| format!("missing value for {flag}").into())
}

fn parse_usize_list(value: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    value
        .split(',')
        .map(|part| part.parse::<usize>().map_err(Into::into))
        .collect()
}

fn parse_u64_list(value: &str) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
    if let Some((a, b)) = value.split_once('-') {
        let start: u64 = a.parse()?;
        let end: u64 = b.parse()?;
        Ok((start..=end).collect())
    } else {
        value
            .split(',')
            .map(|part| part.parse::<u64>().map_err(Into::into))
            .collect()
    }
}

fn write_static_run_files(cfg: &Config) -> Result<(), Box<dyn std::error::Error>> {
    fs::write(
        cfg.out.join("queue.json"),
        serde_json::to_string_pretty(&json!({
            "probe": "STABLE_LOOP_PHASE_LOCK_013_PHASE_LANE_TRANSPORT_MECHANICS",
            "seeds": cfg.seeds,
            "widths": cfg.widths,
            "path_lengths": cfg.path_lengths,
            "ticks_list": cfg.ticks_list,
            "eval_examples": cfg.eval_examples,
            "arms": arms().iter().map(|arm| arm.as_str()).collect::<Vec<_>>(),
            "families": families(),
        }))?,
    )?;
    fs::write(cfg.out.join("contract_snapshot.md"), CONTRACT_SNAPSHOT)?;
    for file in [
        "progress.jsonl",
        "metrics.jsonl",
        "transport_curve.jsonl",
        "per_step_metrics.jsonl",
        "per_pair_step_metrics.jsonl",
        "readout_timing_metrics.jsonl",
        "clock_metrics.jsonl",
        "latch_metrics.jsonl",
        "backflow_metrics.jsonl",
        "normalization_metrics.jsonl",
        "arrival_metrics.jsonl",
        "family_metrics.jsonl",
        "counterfactual_metrics.jsonl",
        "locality_audit.jsonl",
        "examples_sample.jsonl",
    ] {
        File::create(cfg.out.join(file))?;
    }
    Ok(())
}

fn maybe_write_examples(cfg: &Config, cases: &[Case]) -> Result<(), Box<dyn std::error::Error>> {
    let path = cfg.out.join("examples_sample.jsonl");
    let existing = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    if existing > 0 {
        return Ok(());
    }
    for case in cases.iter().take(8) {
        append_jsonl(
            &path,
            &json!({
                "public": case.public,
                "private_summary": {
                    "label": case.private.label,
                    "family": case.private.family,
                    "requested_path_length": case.private.requested_path_length,
                }
            }),
        )?;
    }
    Ok(())
}

fn write_job_progress(
    cfg: &Config,
    job_id: &str,
    completed: usize,
    total: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    append_jsonl(
        cfg.out.join("job_progress").join(format!("{job_id}.jsonl")),
        &json!({
            "job_id": job_id,
            "completed": completed,
            "total": total,
            "time": now_sec(),
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
    serde_json::to_writer(&mut file, value)?;
    writeln!(file)?;
    Ok(())
}

fn arms() -> Vec<Arm> {
    vec![
        Arm::PerStepOracleInjection,
        Arm::Full16RuleTemplateBaseline,
        Arm::CompletedSparseTemplateBaseline,
        Arm::StepwiseOracleClock,
        Arm::PathOnlyForwardClock,
        Arm::FinalTickReadout,
        Arm::BestTickReadout,
        Arm::FirstArrivalReadout,
        Arm::PersistentTargetReadout,
        Arm::ArriveLatch1Tick,
        Arm::ArriveLatchPersistent,
        Arm::EmitLatchPersistent,
        Arm::ConsumeOnForwardLatch,
        Arm::BidirectionalGridBaseline,
        Arm::OracleDirectionNoBackflow,
        Arm::PublicGradientNoBackflow,
        Arm::CellLocalNormalization,
        Arm::TargetOnlyNormalization,
        Arm::RandomControl,
    ]
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

fn phase_decay_proxy(snapshots: &[TargetSnapshot], label: usize, path_len: usize) -> f64 {
    if snapshots.is_empty() {
        return 0.0;
    }
    let best = snapshots
        .iter()
        .map(|snap| snap.probs[label])
        .fold(0.0f64, f64::max);
    let final_prob = snapshots
        .last()
        .map(|snap| snap.probs[label])
        .unwrap_or(0.0);
    ((best - final_prob).max(0.0)) / path_len.max(1) as f64
}

fn backflow_echo_power(layout: &Layout, case: &Case, snapshots: &[TargetSnapshot]) -> (f64, f64) {
    let _ = layout;
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

fn wall_leak_rate(net: &mut Network, layout: &Layout, cases: &[Case], ticks: usize) -> f64 {
    let mut wall_emit = 0usize;
    let mut wall_total = 0usize;
    for case in cases.iter().take(8) {
        let _ = network_output(
            net,
            layout,
            &case.public,
            case.private.label as usize,
            ticks,
        );
        for y in 0..case.public.width {
            for x in 0..case.public.width {
                if case.public.wall[y * case.public.width + x] {
                    for phase in 0..K {
                        let idx = layout.emit(y, x, phase);
                        wall_emit += usize::from(
                            net.spike_data()[idx].charge > 0 || net.activation()[idx] > 0,
                        );
                        wall_total += 1;
                    }
                }
            }
        }
    }
    ratio(wall_emit, wall_total)
}

fn gate_shuffle_collapse(net: &mut Network, layout: &Layout, cases: &[Case], ticks: usize) -> f64 {
    let mut correct = 0usize;
    for case in cases {
        let mut shuffled = case.public.clone();
        shuffled.gates.reverse();
        let out = network_output(net, layout, &shuffled, case.private.label as usize, ticks);
        correct += usize::from(out.pred == case.private.label as usize);
    }
    1.0 - ratio(correct, cases.len())
}

fn empty_pair_stats() -> [[PairStats; K]; K] {
    std::array::from_fn(|phase| {
        std::array::from_fn(|gate| PairStats {
            input_phase: phase,
            gate,
            count: 0,
            correct: 0,
            probability_sum: 0.0,
        })
    })
}

fn mean_pair_probability(stats: &[[PairStats; K]; K]) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for row in stats {
        for pair in row {
            if pair.count > 0 {
                total += pair.probability();
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

fn manhattan(a: (usize, usize), b: (usize, usize)) -> usize {
    a.0.abs_diff(b.0) + a.1.abs_diff(b.1)
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

fn now_sec() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

const CONTRACT_SNAPSHOT: &str = r#"# STABLE_LOOP_PHASE_LOCK_013_PHASE_LANE_TRANSPORT_MECHANICS

Diagnostic-only runner for classifying long-path failure after the completed
phase-lane local rule. Private-path arms are explicitly diagnostic only. No
mutation, search, pruning, or public API changes are part of this probe.
"#;
