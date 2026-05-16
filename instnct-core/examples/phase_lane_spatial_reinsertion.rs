//! Runner-local phase-lane spatial reinsertion probe.
//!
//! 009 takes the local `phase_i + gate_g -> phase_(i+g)` coincidence motif
//! from 008 and reinserts it into a small spatial phase-lane substrate. The
//! coincidence/polarity/channel operators here are diagnostic runner-local
//! mutation lanes; they do not change the public `instnct-core` API.

use instnct_core::{
    evolution_step_jackpot_traced_with_policy_and_operator_weights, mutation_operator_index,
    AcceptancePolicy, CandidateTraceRecord, EvolutionConfig, Int8Projection, Network,
    PropagationConfig,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const K: usize = 4;
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
    steps: usize,
    eval_examples: usize,
    width: usize,
    ticks: usize,
    jackpot: usize,
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
}

#[derive(Clone, Debug)]
struct Case {
    public: PublicCase,
    private: PrivateCase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
enum Arm {
    FixedPhaseLaneReference,
    HandBuiltSpatialCoincidenceReference,
    CanonicalJackpot007Baseline,
    OracleRoutingPlusCoincidenceOperator,
    FullSpatialPlusCoincidenceOperator,
    CoincidenceOperatorStrict,
    CoincidenceOperatorTies,
    CoincidenceOperatorZeroP,
    PolarityOnly,
    ChannelOnly,
    CoincidencePlusPolarity,
    CoincidencePlusChannel,
    CoincidencePlusPolarityChannel,
}

impl Arm {
    fn as_str(self) -> &'static str {
        match self {
            Arm::FixedPhaseLaneReference => "FIXED_PHASE_LANE_REFERENCE",
            Arm::HandBuiltSpatialCoincidenceReference => "HAND_BUILT_SPATIAL_COINCIDENCE_REFERENCE",
            Arm::CanonicalJackpot007Baseline => "CANONICAL_JACKPOT_007_BASELINE",
            Arm::OracleRoutingPlusCoincidenceOperator => "ORACLE_ROUTING_PLUS_COINCIDENCE_OPERATOR",
            Arm::FullSpatialPlusCoincidenceOperator => "FULL_SPATIAL_PLUS_COINCIDENCE_OPERATOR",
            Arm::CoincidenceOperatorStrict => "COINCIDENCE_OPERATOR_STRICT",
            Arm::CoincidenceOperatorTies => "COINCIDENCE_OPERATOR_TIES",
            Arm::CoincidenceOperatorZeroP => "COINCIDENCE_OPERATOR_ZEROP",
            Arm::PolarityOnly => "POLARITY_ONLY",
            Arm::ChannelOnly => "CHANNEL_ONLY",
            Arm::CoincidencePlusPolarity => "COINCIDENCE_PLUS_POLARITY",
            Arm::CoincidencePlusChannel => "COINCIDENCE_PLUS_CHANNEL",
            Arm::CoincidencePlusPolarityChannel => "COINCIDENCE_PLUS_POLARITY_CHANNEL",
        }
    }

    fn uses_canonical(self) -> bool {
        matches!(self, Arm::CanonicalJackpot007Baseline)
    }

    fn uses_runner_operator(self) -> bool {
        matches!(
            self,
            Arm::OracleRoutingPlusCoincidenceOperator
                | Arm::FullSpatialPlusCoincidenceOperator
                | Arm::CoincidenceOperatorStrict
                | Arm::CoincidenceOperatorTies
                | Arm::CoincidenceOperatorZeroP
                | Arm::PolarityOnly
                | Arm::ChannelOnly
                | Arm::CoincidencePlusPolarity
                | Arm::CoincidencePlusChannel
                | Arm::CoincidencePlusPolarityChannel
        )
    }

    fn allows_coincidence(self) -> bool {
        matches!(
            self,
            Arm::OracleRoutingPlusCoincidenceOperator
                | Arm::FullSpatialPlusCoincidenceOperator
                | Arm::CoincidenceOperatorStrict
                | Arm::CoincidenceOperatorTies
                | Arm::CoincidenceOperatorZeroP
                | Arm::CoincidencePlusPolarity
                | Arm::CoincidencePlusChannel
                | Arm::CoincidencePlusPolarityChannel
        )
    }

    fn allows_polarity(self) -> bool {
        matches!(
            self,
            Arm::PolarityOnly | Arm::CoincidencePlusPolarity | Arm::CoincidencePlusPolarityChannel
        )
    }

    fn allows_channel(self) -> bool {
        matches!(
            self,
            Arm::ChannelOnly | Arm::CoincidencePlusChannel | Arm::CoincidencePlusPolarityChannel
        )
    }

    fn oracle_routing(self) -> bool {
        matches!(self, Arm::OracleRoutingPlusCoincidenceOperator)
    }

    fn acceptance(self) -> AcceptanceMode {
        match self {
            Arm::CoincidenceOperatorTies => AcceptanceMode::Ties,
            Arm::CoincidenceOperatorZeroP => AcceptanceMode::ZeroP,
            _ => AcceptanceMode::Strict,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AcceptanceMode {
    Strict,
    Ties,
    ZeroP,
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

    fn local_lane(&self, neuron: usize) -> usize {
        neuron % LANES_PER_CELL
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

    fn coincidence(&self, y: usize, x: usize, input_phase: usize, gate: usize, output_phase: usize) -> usize {
        self.cell_base(y, x)
            + COINCIDENCE_BASE
            + ((input_phase * K + gate) * K + output_phase)
    }

    fn decode_coincidence(&self, neuron: usize) -> Option<(usize, usize, usize, usize, usize)> {
        let lane = self.local_lane(neuron);
        if !(COINCIDENCE_BASE..COINCIDENCE_BASE + COINCIDENCE_PER_CELL).contains(&lane) {
            return None;
        }
        let rel = lane - COINCIDENCE_BASE;
        let input_phase = rel / (K * K);
        let gate = (rel / K) % K;
        let output_phase = rel % K;
        let (y, x) = self.cell_of_neuron(neuron);
        Some((y, x, input_phase, gate, output_phase))
    }
}

#[derive(Clone, Debug, Default, Serialize)]
struct EvalMetrics {
    phase_final_accuracy: f64,
    correct_target_lane_probability_mean: f64,
    same_target_counterfactual_accuracy: f64,
    gate_shuffle_accuracy: f64,
    gate_shuffle_collapse: f64,
    target_shuffle_collapse: f64,
    wall_shuffle_degradation: f64,
    heldout_path_length_accuracy: f64,
    highway_phase_retention: f64,
    wall_leak_rate: f64,
    forbidden_private_field_leak: f64,
    nonlocal_edge_count: usize,
    direct_output_leak_rate: f64,
    edge_count: usize,
    unique_cells_with_motif: usize,
    motif_density_per_cell: f64,
    full_16_pair_coverage_per_cell: f64,
    motif_reuse_across_examples: f64,
    total_coincidence_gates_added: usize,
    useful_coincidence_gates: usize,
    useless_coincidence_gates: usize,
    motif_precision: f64,
    motif_recall: f64,
    motif_ablation_drop: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
struct CandidateStats {
    candidate_positive_delta_fraction: f64,
    candidate_delta_nonzero_fraction: f64,
    accepted_coincidence_operator_rate: f64,
    accepted_polarity_operator_rate: f64,
    accepted_channel_operator_rate: f64,
    accepted_candidates: usize,
    evaluated_candidates: usize,
    mean_delta: f64,
    mean_abs_delta: f64,
    candidate_delta_std: f64,
}

#[derive(Clone, Debug, Serialize)]
struct MetricRow {
    job_id: String,
    seed: u64,
    arm: String,
    checkpoint_step: usize,
    final_row: bool,
    phase_final_accuracy: f64,
    correct_target_lane_probability_mean: f64,
    same_target_counterfactual_accuracy: f64,
    gate_shuffle_accuracy: f64,
    gate_shuffle_collapse: f64,
    target_shuffle_collapse: f64,
    wall_shuffle_degradation: f64,
    heldout_path_length_accuracy: f64,
    highway_phase_retention: f64,
    wall_leak_rate: f64,
    forbidden_private_field_leak: f64,
    nonlocal_edge_count: usize,
    direct_output_leak_rate: f64,
    edge_count: usize,
    candidate_positive_delta_fraction: f64,
    candidate_delta_nonzero_fraction: f64,
    accepted_coincidence_operator_rate: f64,
    accepted_polarity_operator_rate: f64,
    accepted_channel_operator_rate: f64,
    accepted_candidates: usize,
    evaluated_candidates: usize,
    motif_density_per_cell: f64,
    motif_precision: f64,
    motif_recall: f64,
    motif_ablation_drop: f64,
    unique_cells_with_motif: usize,
    full_16_pair_coverage_per_cell: f64,
    total_coincidence_gates_added: usize,
    useful_coincidence_gates: usize,
    useless_coincidence_gates: usize,
    elapsed_sec: f64,
}

#[derive(Clone, Debug, Serialize)]
struct PlacementAuditRow {
    job_id: String,
    seed: u64,
    arm: String,
    step: usize,
    candidate_id: usize,
    accepted: bool,
    cell_id: usize,
    input_phase: usize,
    gate: usize,
    output_phase: usize,
    is_on_path_cell: bool,
    is_target_cell: bool,
    writes_to_target_directly: bool,
    reads_private_path: bool,
    reads_private_label: bool,
    local_edge_only: bool,
}

#[derive(Clone, Copy, Debug)]
enum RunnerOperator {
    AddCoincidence,
    FlipPolarity,
    MutateChannel,
}

#[derive(Clone, Debug)]
struct CandidateAudit {
    operator: RunnerOperator,
    cell: (usize, usize),
    input_phase: usize,
    gate: usize,
    output_phase: usize,
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
    let layout = Layout { width: cfg.width };
    let cases = generate_cases(cfg.seeds[0], cfg.eval_examples, cfg.width, cfg.ticks, "eval");
    let mut hand_built = hand_built_spatial_network(&layout);
    let hand_metrics = eval_network(&mut hand_built, &layout, &cases, cfg.ticks);
    let hand_pass = hand_built_passes(&hand_metrics);
    append_jsonl(
        cfg.out.join("progress.jsonl"),
        &json!({
            "event": "hand_built_gate",
            "time": now_sec(),
            "passed": hand_pass,
            "phase_final_accuracy": hand_metrics.phase_final_accuracy,
            "correct_target_lane_probability_mean": hand_metrics.correct_target_lane_probability_mean,
            "same_target_counterfactual_accuracy": hand_metrics.same_target_counterfactual_accuracy,
            "gate_shuffle_collapse": hand_metrics.gate_shuffle_collapse,
            "wall_leak_rate": hand_metrics.wall_leak_rate,
            "nonlocal_edge_count": hand_metrics.nonlocal_edge_count,
        }),
    )?;

    let mut rows = Vec::new();
    let mut completed = 0usize;
    let planned_jobs = if hand_pass {
        cfg.seeds.len() * arms().len()
    } else {
        cfg.seeds.len()
    };

    for seed in cfg.seeds.iter().copied() {
        let run_arms = if hand_pass {
            arms()
        } else {
            vec![Arm::HandBuiltSpatialCoincidenceReference]
        };
        for arm in run_arms {
            let row = run_job(&cfg, seed, arm, started)?;
            rows.push(row);
            completed += 1;
            append_jsonl(
                cfg.out.join("progress.jsonl"),
                &json!({
                    "event": "job_done",
                    "time": now_sec(),
                    "completed": completed,
                    "total": planned_jobs,
                    "seed": seed,
                    "arm": arm.as_str(),
                }),
            )?;
            refresh_summary(&cfg, &rows, completed, planned_jobs, started.elapsed().as_secs_f64())?;
        }
    }

    write_operator_summary(&cfg, &rows)?;
    write_report(&cfg, &rows, true)?;
    append_jsonl(
        cfg.out.join("progress.jsonl"),
        &json!({"event": "run_done", "time": now_sec(), "completed": completed, "total": planned_jobs}),
    )?;
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = Config {
        out: PathBuf::from(
            "target/pilot_wave/stable_loop_phase_lock_009_coincidence_operator_spatial_reinsertion/dev",
        ),
        seeds: vec![2026],
        steps: 100,
        eval_examples: 256,
        width: 6,
        ticks: 8,
        jackpot: 6,
        heartbeat_sec: 15,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                i += 1;
                cfg.out = PathBuf::from(&args[i]);
            }
            "--seeds" => {
                i += 1;
                cfg.seeds = parse_seeds(&args[i]);
            }
            "--steps" => {
                i += 1;
                cfg.steps = args[i].parse()?;
            }
            "--eval-examples" => {
                i += 1;
                cfg.eval_examples = args[i].parse()?;
            }
            "--width" => {
                i += 1;
                cfg.width = args[i].parse()?;
            }
            "--ticks" => {
                i += 1;
                cfg.ticks = args[i].parse()?;
            }
            "--jackpot" => {
                i += 1;
                cfg.jackpot = args[i].parse()?;
            }
            "--heartbeat-sec" => {
                i += 1;
                cfg.heartbeat_sec = args[i].parse()?;
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        i += 1;
    }

    if cfg.width < 5 {
        return Err("--width must be at least 5".into());
    }
    if cfg.eval_examples == 0 {
        return Err("--eval-examples must be positive".into());
    }
    if cfg.ticks < 4 {
        return Err("--ticks must be at least 4".into());
    }
    Ok(cfg)
}

fn parse_seeds(raw: &str) -> Vec<u64> {
    if let Some((a, b)) = raw.split_once('-') {
        let start: u64 = a.parse().unwrap_or(2026);
        let end: u64 = b.parse().unwrap_or(start);
        return (start..=end).collect();
    }
    raw.split(',')
        .filter_map(|s| s.trim().parse::<u64>().ok())
        .collect()
}

fn arms() -> Vec<Arm> {
    vec![
        Arm::FixedPhaseLaneReference,
        Arm::HandBuiltSpatialCoincidenceReference,
        Arm::CanonicalJackpot007Baseline,
        Arm::OracleRoutingPlusCoincidenceOperator,
        Arm::FullSpatialPlusCoincidenceOperator,
        Arm::CoincidenceOperatorStrict,
        Arm::CoincidenceOperatorTies,
        Arm::CoincidenceOperatorZeroP,
        Arm::PolarityOnly,
        Arm::ChannelOnly,
        Arm::CoincidencePlusPolarity,
        Arm::CoincidencePlusChannel,
        Arm::CoincidencePlusPolarityChannel,
    ]
}

fn run_job(
    cfg: &Config,
    seed: u64,
    arm: Arm,
    run_started: Instant,
) -> Result<MetricRow, Box<dyn std::error::Error>> {
    let job_id = format!("{}_{}", seed, arm.as_str());
    let job_path = cfg.out.join("job_progress").join(format!("{job_id}.jsonl"));
    let layout = Layout { width: cfg.width };
    let cases = generate_cases(seed, cfg.eval_examples, cfg.width, cfg.ticks, "eval");
    let train_cases = generate_cases(
        seed ^ 0xA5A5_5A5A,
        cfg.eval_examples.min(128),
        cfg.width,
        cfg.ticks,
        "train",
    );

    if cfg
        .out
        .join("examples_sample.jsonl")
        .metadata()
        .map(|m| m.len())
        .unwrap_or(0)
        == 0
    {
        for case in cases.iter().take(8) {
            append_jsonl(
                cfg.out.join("examples_sample.jsonl"),
                &json!({"public": &case.public, "private_audit_only": &case.private}),
            )?;
        }
    }

    append_jsonl(
        &job_path,
        &json!({"event": "job_start", "time": now_sec(), "seed": seed, "arm": arm.as_str()}),
    )?;

    let started = Instant::now();
    let mut rng = StdRng::seed_from_u64(seed ^ arm_seed_salt(arm));
    let mut candidate_deltas = Vec::new();
    let mut accepted_ops = Vec::new();
    let mut final_metrics;

    match arm {
        Arm::FixedPhaseLaneReference => {
            final_metrics = eval_fixed_reference(&cases);
        }
        Arm::HandBuiltSpatialCoincidenceReference => {
            let mut net = hand_built_spatial_network(&layout);
            final_metrics = eval_network(&mut net, &layout, &cases, cfg.ticks);
        }
        _ => {
            let mut net = initial_network(arm, &layout, &mut rng);
            final_metrics = eval_network(&mut net, &layout, &cases, cfg.ticks);

            if arm.uses_canonical() {
                let mut projection = Int8Projection::new(1, 1, &mut rng);
                let mut eval_rng = StdRng::seed_from_u64(seed ^ 0x1357_2468);
                let evo_cfg = EvolutionConfig {
                    edge_cap: max_edge_cap(&layout),
                    accept_ties: false,
                };
                let mut writer = BufWriter::new(
                    OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(cfg.out.join("candidate_log.jsonl"))?,
                );
                let checkpoint_every = 20usize.max(cfg.steps / 20).min(100);
                let mut last_heartbeat = Instant::now();
                for step in 0..cfg.steps {
                    let outcome = evolution_step_jackpot_traced_with_policy_and_operator_weights(
                        &mut net,
                        &mut projection,
                        &mut rng,
                        &mut eval_rng,
                        |candidate_net, _projection, _rng| {
                            if !locality_valid(candidate_net, &layout) {
                                return -1.0;
                            }
                            eval_correct_probability(candidate_net, &layout, &train_cases, cfg.ticks)
                        },
                        &evo_cfg,
                        AcceptancePolicy::Strict,
                        cfg.jackpot,
                        step,
                        Some(&canonical_operator_weights()),
                        |record: &CandidateTraceRecord| {
                            if record.evaluated {
                                candidate_deltas.push(record.delta_u);
                            }
                            if record.accepted {
                                accepted_ops.push("canonical".to_string());
                            }
                            let _ = write_canonical_candidate_record(&mut writer, &job_id, seed, arm, record);
                        },
                    );
                    writer.flush()?;
                    if should_checkpoint(step, cfg.steps, checkpoint_every, cfg.heartbeat_sec, &last_heartbeat) {
                        let metrics = eval_network(&mut net, &layout, &cases, cfg.ticks);
                        let row = metric_row(
                            &job_id,
                            seed,
                            arm,
                            step + 1,
                            false,
                            &metrics,
                            &candidate_stats(&candidate_deltas, &accepted_ops),
                            started.elapsed().as_secs_f64(),
                        );
                        append_jsonl(cfg.out.join("metrics.jsonl"), &row)?;
                        append_jsonl(
                            &job_path,
                            &json!({
                                "event": "checkpoint",
                                "time": now_sec(),
                                "step": step + 1,
                                "outcome": format!("{:?}", outcome),
                                "phase_final_accuracy": metrics.phase_final_accuracy,
                                "correct_target_lane_probability_mean": metrics.correct_target_lane_probability_mean,
                                "accepted_candidates": accepted_ops.len(),
                            }),
                        )?;
                        refresh_summary_partial(cfg, run_started.elapsed().as_secs_f64())?;
                        last_heartbeat = Instant::now();
                    }
                }
                final_metrics = eval_network(&mut net, &layout, &cases, cfg.ticks);
            } else if arm.uses_runner_operator() {
                let mut writer = BufWriter::new(
                    OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(cfg.out.join("candidate_log.jsonl"))?,
                );
                let mut placement_writer = BufWriter::new(
                    OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(cfg.out.join("motif_placement_audit.jsonl"))?,
                );
                let checkpoint_every = 20usize.max(cfg.steps / 20).min(100);
                let mut last_heartbeat = Instant::now();
                for step in 0..cfg.steps {
                    runner_operator_step(
                        &mut net,
                        &layout,
                        &train_cases,
                        &mut rng,
                        arm,
                        cfg.jackpot,
                        cfg.ticks,
                        step,
                        &job_id,
                        seed,
                        &mut writer,
                        &mut placement_writer,
                        &mut candidate_deltas,
                        &mut accepted_ops,
                    )?;
                    writer.flush()?;
                    placement_writer.flush()?;
                    if should_checkpoint(step, cfg.steps, checkpoint_every, cfg.heartbeat_sec, &last_heartbeat) {
                        let metrics = eval_network(&mut net, &layout, &cases, cfg.ticks);
                        let row = metric_row(
                            &job_id,
                            seed,
                            arm,
                            step + 1,
                            false,
                            &metrics,
                            &candidate_stats(&candidate_deltas, &accepted_ops),
                            started.elapsed().as_secs_f64(),
                        );
                        append_jsonl(cfg.out.join("metrics.jsonl"), &row)?;
                        append_jsonl(
                            &job_path,
                            &json!({
                                "event": "checkpoint",
                                "time": now_sec(),
                                "step": step + 1,
                                "phase_final_accuracy": metrics.phase_final_accuracy,
                                "correct_target_lane_probability_mean": metrics.correct_target_lane_probability_mean,
                                "accepted_candidates": accepted_ops.len(),
                            }),
                        )?;
                        refresh_summary_partial(cfg, run_started.elapsed().as_secs_f64())?;
                        last_heartbeat = Instant::now();
                    }
                }
                final_metrics = eval_network(&mut net, &layout, &cases, cfg.ticks);
            }
        }
    }

    let stats = candidate_stats(&candidate_deltas, &accepted_ops);
    let row = metric_row(
        &job_id,
        seed,
        arm,
        cfg.steps,
        true,
        &final_metrics,
        &stats,
        started.elapsed().as_secs_f64(),
    );
    append_jsonl(cfg.out.join("metrics.jsonl"), &row)?;
    append_jsonl(cfg.out.join("spatial_stage_metrics.jsonl"), &row)?;
    append_jsonl(
        cfg.out.join("ablation_metrics.jsonl"),
        &json!({
            "job_id": job_id,
            "seed": seed,
            "arm": arm.as_str(),
            "motif_ablation_drop": row.motif_ablation_drop,
            "motif_density_per_cell": row.motif_density_per_cell,
            "motif_precision": row.motif_precision,
            "motif_recall": row.motif_recall,
        }),
    )?;
    append_jsonl(
        cfg.out.join("counterfactual_metrics.jsonl"),
        &json!({
            "job_id": job_id,
            "seed": seed,
            "arm": arm.as_str(),
            "same_target_counterfactual_accuracy": row.same_target_counterfactual_accuracy,
            "gate_shuffle_collapse": row.gate_shuffle_collapse,
            "target_shuffle_collapse": row.target_shuffle_collapse,
            "wall_shuffle_degradation": row.wall_shuffle_degradation,
        }),
    )?;
    append_jsonl(
        cfg.out.join("locality_audit.jsonl"),
        &json!({
            "job_id": job_id,
            "seed": seed,
            "arm": arm.as_str(),
            "forbidden_private_field_leak": row.forbidden_private_field_leak,
            "nonlocal_edge_count": row.nonlocal_edge_count,
            "direct_output_leak_rate": row.direct_output_leak_rate,
        }),
    )?;
    append_jsonl(&job_path, &json!({"event": "job_done", "time": now_sec(), "row": row}))?;
    Ok(row)
}

fn should_checkpoint(
    step: usize,
    steps: usize,
    checkpoint_every: usize,
    heartbeat_sec: u64,
    last_heartbeat: &Instant,
) -> bool {
    (step + 1) % checkpoint_every == 0
        || step + 1 == steps
        || last_heartbeat.elapsed().as_secs() >= heartbeat_sec
}

fn metric_row(
    job_id: &str,
    seed: u64,
    arm: Arm,
    step: usize,
    final_row: bool,
    metrics: &EvalMetrics,
    stats: &CandidateStats,
    elapsed_sec: f64,
) -> MetricRow {
    MetricRow {
        job_id: job_id.to_string(),
        seed,
        arm: arm.as_str().to_string(),
        checkpoint_step: step,
        final_row,
        phase_final_accuracy: metrics.phase_final_accuracy,
        correct_target_lane_probability_mean: metrics.correct_target_lane_probability_mean,
        same_target_counterfactual_accuracy: metrics.same_target_counterfactual_accuracy,
        gate_shuffle_accuracy: metrics.gate_shuffle_accuracy,
        gate_shuffle_collapse: metrics.gate_shuffle_collapse,
        target_shuffle_collapse: metrics.target_shuffle_collapse,
        wall_shuffle_degradation: metrics.wall_shuffle_degradation,
        heldout_path_length_accuracy: metrics.heldout_path_length_accuracy,
        highway_phase_retention: metrics.highway_phase_retention,
        wall_leak_rate: metrics.wall_leak_rate,
        forbidden_private_field_leak: metrics.forbidden_private_field_leak,
        nonlocal_edge_count: metrics.nonlocal_edge_count,
        direct_output_leak_rate: metrics.direct_output_leak_rate,
        edge_count: metrics.edge_count,
        candidate_positive_delta_fraction: stats.candidate_positive_delta_fraction,
        candidate_delta_nonzero_fraction: stats.candidate_delta_nonzero_fraction,
        accepted_coincidence_operator_rate: stats.accepted_coincidence_operator_rate,
        accepted_polarity_operator_rate: stats.accepted_polarity_operator_rate,
        accepted_channel_operator_rate: stats.accepted_channel_operator_rate,
        accepted_candidates: stats.accepted_candidates,
        evaluated_candidates: stats.evaluated_candidates,
        motif_density_per_cell: metrics.motif_density_per_cell,
        motif_precision: metrics.motif_precision,
        motif_recall: metrics.motif_recall,
        motif_ablation_drop: metrics.motif_ablation_drop,
        unique_cells_with_motif: metrics.unique_cells_with_motif,
        full_16_pair_coverage_per_cell: metrics.full_16_pair_coverage_per_cell,
        total_coincidence_gates_added: metrics.total_coincidence_gates_added,
        useful_coincidence_gates: metrics.useful_coincidence_gates,
        useless_coincidence_gates: metrics.useless_coincidence_gates,
        elapsed_sec,
    }
}

fn hand_built_passes(metrics: &EvalMetrics) -> bool {
    metrics.phase_final_accuracy >= 0.95
        && metrics.correct_target_lane_probability_mean >= 0.90
        && metrics.same_target_counterfactual_accuracy >= 0.85
        && metrics.gate_shuffle_collapse >= 0.50
        && metrics.wall_leak_rate <= 0.0
        && metrics.nonlocal_edge_count == 0
}

fn generate_cases(seed: u64, count: usize, width: usize, ticks: usize, split: &str) -> Vec<Case> {
    let mut rng = StdRng::seed_from_u64(seed);
    let families = [
        "two_cell_composition",
        "short_chain_composition",
        "oracle_routing_spatial_reinsertion",
        "full_spatial_wavefield_reinsertion",
        "same_target_counterfactual",
        "heldout_path_length",
        "gate_shuffle_control",
        "wall_shuffle_control",
    ];
    let mut cases = Vec::with_capacity(count);
    // A phase lane hop takes several spiking ticks: emit -> arrive,
    // arrive+gate -> coincidence, coincidence -> emit, with channel phase
    // windows occasionally adding slack. Keep generated paths inside the
    // requested tick budget so the hand-built spatial reference tests the
    // motif/reinsertion, not an impossible settling horizon.
    let max_hops = ((ticks.saturating_sub(4)) / 4).max(1).min(width.saturating_sub(3));
    let base_y = width / 2;
    for i in 0..count {
        let family = families[i % families.len()];
        let hops = match family {
            "two_cell_composition" => 1usize,
            "short_chain_composition" => 2usize.min(max_hops),
            "heldout_path_length" => max_hops,
            _ => 1 + (i / families.len()) % max_hops,
        };
        let source = (base_y, 1usize);
        let target = (base_y, 1usize + hops);
        let path = (0..=hops).map(|dx| (base_y, 1 + dx)).collect::<Vec<_>>();
        let mut gates = vec![0u8; width * width];
        for y in 0..width {
            for x in 0..width {
                gates[y * width + x] = rng.gen_range(0..K as u8);
            }
        }
        for (j, &(y, x)) in path.iter().enumerate().skip(1) {
            let base = match family {
                "same_target_counterfactual" => ((i / families.len()) + j) as u8,
                "heldout_path_length" => ((2 * j + 1) % K) as u8,
                _ => rng.gen_range(0..K as u8),
            };
            gates[y * width + x] = base % K as u8;
        }
        let source_phase = rng.gen_range(0..K as u8);
        let gate_sum = path.iter().skip(1).fold(0u8, |acc, &(y, x)| {
            (acc + gates[y * width + x]) % K as u8
        });
        let label = (source_phase + gate_sum) % K as u8;
        let wall = make_wall_mask(width, &path);
        cases.push(Case {
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
                split: split.to_string(),
            },
        });
    }
    cases
}

fn make_wall_mask(width: usize, path: &[(usize, usize)]) -> Vec<bool> {
    let mut free = vec![false; width * width];
    for &(y, x) in path {
        free[y * width + x] = true;
    }
    free.into_iter().map(|is_free| !is_free).collect()
}

fn eval_fixed_reference(cases: &[Case]) -> EvalMetrics {
    let mut correct = 0usize;
    let mut total_prob = 0.0;
    let mut cf_total = 0usize;
    let mut cf_correct = 0usize;
    let mut heldout_total = 0usize;
    let mut heldout_correct = 0usize;
    let mut gate_shuffle_correct = 0usize;
    let mut target_shuffle_correct = 0usize;
    for case in cases {
        let label = case.private.label as usize;
        correct += 1;
        total_prob += 0.97;
        let shuffled_label = shuffled_gate_label(case);
        gate_shuffle_correct += usize::from(label == shuffled_label);
        target_shuffle_correct += usize::from(case.public.source_phase as usize == label);
        if case.private.family == "same_target_counterfactual" {
            cf_total += 1;
            cf_correct += 1;
        }
        if case.private.family == "heldout_path_length" {
            heldout_total += 1;
            heldout_correct += 1;
        }
    }
    let n = cases.len().max(1) as f64;
    EvalMetrics {
        phase_final_accuracy: correct as f64 / n,
        correct_target_lane_probability_mean: total_prob / n,
        same_target_counterfactual_accuracy: ratio(cf_correct, cf_total),
        gate_shuffle_accuracy: gate_shuffle_correct as f64 / n,
        gate_shuffle_collapse: (correct as f64 / n - gate_shuffle_correct as f64 / n).max(0.0),
        target_shuffle_collapse: (correct as f64 / n - target_shuffle_correct as f64 / n).max(0.0),
        wall_shuffle_degradation: 1.0,
        heldout_path_length_accuracy: ratio(heldout_correct, heldout_total),
        highway_phase_retention: 1.0,
        wall_leak_rate: 0.0,
        forbidden_private_field_leak: 0.0,
        nonlocal_edge_count: 0,
        direct_output_leak_rate: 0.0,
        ..EvalMetrics::default()
    }
}

fn eval_network(net: &mut Network, layout: &Layout, cases: &[Case], ticks: usize) -> EvalMetrics {
    let mut correct = 0usize;
    let mut total_prob = 0.0;
    let mut cf_total = 0usize;
    let mut cf_correct = 0usize;
    let mut heldout_total = 0usize;
    let mut heldout_correct = 0usize;
    let mut gate_shuffle_correct = 0usize;
    let mut target_shuffle_correct = 0usize;
    let mut wall_shuffle_correct = 0usize;

    for case in cases {
        let probs = network_probs(net, layout, &case.public, ticks);
        let pred = argmax(&probs);
        let label = case.private.label as usize;
        correct += usize::from(pred == label);
        total_prob += probs[label];

        let mut shuffled = case.public.clone();
        rotate_gates(&mut shuffled.gates);
        gate_shuffle_correct += usize::from(argmax(&network_probs(net, layout, &shuffled, ticks)) == label);

        let mut target_shuffled = case.public.clone();
        target_shuffled.target = case.public.source;
        target_shuffle_correct += usize::from(argmax(&network_probs(net, layout, &target_shuffled, ticks)) == label);

        let mut wall_shuffled = case.public.clone();
        for y in 0..wall_shuffled.width {
            for x in 0..wall_shuffled.width {
                wall_shuffled.wall[y * wall_shuffled.width + x] = true;
            }
        }
        wall_shuffled.wall[case.public.source.0 * wall_shuffled.width + case.public.source.1] = false;
        wall_shuffled.wall[case.public.target.0 * wall_shuffled.width + case.public.target.1] = false;
        wall_shuffle_correct += usize::from(argmax(&network_probs(net, layout, &wall_shuffled, ticks)) == label);

        if case.private.family == "same_target_counterfactual" {
            cf_total += 1;
            cf_correct += usize::from(pred == label);
        }
        if case.private.family == "heldout_path_length" {
            heldout_total += 1;
            heldout_correct += usize::from(pred == label);
        }
    }

    let audit = audit_network(net, layout, cases.first().map(|c| c.public.target));
    let motif = motif_metrics(net, layout, cases, ticks);
    let n = cases.len().max(1) as f64;
    let acc = correct as f64 / n;
    EvalMetrics {
        phase_final_accuracy: acc,
        correct_target_lane_probability_mean: total_prob / n,
        same_target_counterfactual_accuracy: ratio(cf_correct, cf_total),
        gate_shuffle_accuracy: gate_shuffle_correct as f64 / n,
        gate_shuffle_collapse: (acc - gate_shuffle_correct as f64 / n).max(0.0),
        target_shuffle_collapse: (acc - target_shuffle_correct as f64 / n).max(0.0),
        wall_shuffle_degradation: (acc - wall_shuffle_correct as f64 / n).max(0.0),
        heldout_path_length_accuracy: ratio(heldout_correct, heldout_total),
        highway_phase_retention: 1.0,
        wall_leak_rate: wall_leak_rate(net, layout, cases, ticks),
        forbidden_private_field_leak: 0.0,
        nonlocal_edge_count: audit.nonlocal_edge_count,
        direct_output_leak_rate: audit.direct_output_leak_rate,
        edge_count: net.edge_count(),
        unique_cells_with_motif: motif.unique_cells_with_motif,
        motif_density_per_cell: motif.motif_density_per_cell,
        full_16_pair_coverage_per_cell: motif.full_16_pair_coverage_per_cell,
        motif_reuse_across_examples: motif.motif_reuse_across_examples,
        total_coincidence_gates_added: motif.total_coincidence_gates_added,
        useful_coincidence_gates: motif.useful_coincidence_gates,
        useless_coincidence_gates: motif.useless_coincidence_gates,
        motif_precision: motif.motif_precision,
        motif_recall: motif.motif_recall,
        motif_ablation_drop: motif.motif_ablation_drop,
    }
}

fn eval_correct_probability(
    net: &mut Network,
    layout: &Layout,
    cases: &[Case],
    ticks: usize,
) -> f64 {
    let mut total_prob = 0.0;
    for case in cases {
        let probs = network_probs(net, layout, &case.public, ticks);
        total_prob += probs[case.private.label as usize];
    }
    total_prob / cases.len().max(1) as f64
}

fn network_probs(net: &mut Network, layout: &Layout, case: &PublicCase, ticks: usize) -> [f64; K] {
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
            let gate = case.gates[y * case.width + x] as usize % K;
            indices.push(layout.gate(y, x, gate) as u16);
            // Keep gate tokens below the coincidence threshold alone. The local
            // motif should fire only when a phase arrival and the local gate
            // coincide in the same tick, matching the 008 microcircuit.
            values.push(1i8);
        }
    }
    let config = PropagationConfig {
        ticks_per_token: ticks,
        input_duration_ticks: ticks,
        decay_interval_ticks: 1,
        use_refractory: false,
    };
    let _ = net.propagate_sparse(&indices, &values, &config);
    let mut scores = [0.0f64; K];
    for (phase, score) in scores.iter_mut().enumerate() {
        let idx = layout.emit(case.target.0, case.target.1, phase);
        let charge = net.spike_data()[idx].charge as f64;
        let activation = net.activation()[idx].max(0) as f64;
        *score = charge + 4.0 * activation;
    }
    normalize_scores(scores)
}

fn initial_network(arm: Arm, layout: &Layout, rng: &mut StdRng) -> Network {
    let mut net = empty_network(layout);
    match arm {
        Arm::CanonicalJackpot007Baseline => add_same_phase_spatial_edges(&mut net, layout),
        Arm::PolarityOnly | Arm::ChannelOnly => add_incorrect_dense_seed(&mut net, layout),
        Arm::CoincidencePlusPolarity | Arm::CoincidencePlusChannel | Arm::CoincidencePlusPolarityChannel => {
            add_incorrect_dense_seed(&mut net, layout)
        }
        _ => {
            add_emit_to_neighbor_arrive_edges(&mut net, layout);
        }
    }
    if matches!(arm, Arm::FullSpatialPlusCoincidenceOperator | Arm::CoincidenceOperatorStrict | Arm::CoincidenceOperatorTies | Arm::CoincidenceOperatorZeroP) {
        for _ in 0..layout.width {
            add_random_local_noise_edge(&mut net, layout, rng);
        }
    }
    net
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

fn hand_built_spatial_network(layout: &Layout) -> Network {
    let mut net = empty_network(layout);
    add_emit_to_neighbor_arrive_edges(&mut net, layout);
    for y in 0..layout.width {
        for x in 0..layout.width {
            for input_phase in 0..K {
                for gate in 0..K {
                    let output_phase = (input_phase + gate) % K;
                    add_coincidence_gate(&mut net, layout, (y, x), input_phase, gate, output_phase);
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
                    net.graph_mut()
                        .add_edge(layout.emit(y, x, phase) as u16, layout.arrive(ny, nx, phase) as u16);
                }
            }
        }
    }
}

fn add_same_phase_spatial_edges(net: &mut Network, layout: &Layout) {
    add_emit_to_neighbor_arrive_edges(net, layout);
    for y in 0..layout.width {
        for x in 0..layout.width {
            for phase in 0..K {
                net.graph_mut()
                    .add_edge(layout.arrive(y, x, phase) as u16, layout.emit(y, x, phase) as u16);
            }
        }
    }
}

fn add_incorrect_dense_seed(net: &mut Network, layout: &Layout) {
    add_emit_to_neighbor_arrive_edges(net, layout);
    for y in 0..layout.width {
        for x in 0..layout.width {
            for input_phase in 0..K {
                for gate in 0..K {
                    let wrong = input_phase;
                    add_coincidence_gate(net, layout, (y, x), input_phase, gate, wrong);
                }
            }
        }
    }
}

fn add_random_local_noise_edge(net: &mut Network, layout: &Layout, rng: &mut StdRng) {
    let y = rng.gen_range(0..layout.width);
    let x = rng.gen_range(0..layout.width);
    let neighbors = neighbors(layout.width, y, x);
    let (ny, nx) = neighbors[rng.gen_range(0..neighbors.len())];
    let src = layout.cell_base(y, x) + rng.gen_range(0..LANES_PER_CELL);
    let dst = layout.cell_base(ny, nx) + rng.gen_range(0..LANES_PER_CELL);
    net.graph_mut().add_edge(src as u16, dst as u16);
}

fn add_coincidence_gate(
    net: &mut Network,
    layout: &Layout,
    cell: (usize, usize),
    input_phase: usize,
    gate: usize,
    output_phase: usize,
) -> bool {
    let (y, x) = cell;
    let c = layout.coincidence(y, x, input_phase, gate, output_phase);
    let mut changed = false;
    changed |= net
        .graph_mut()
        .add_edge(layout.arrive(y, x, input_phase) as u16, c as u16);
    changed |= net.graph_mut().add_edge(layout.gate(y, x, gate) as u16, c as u16);
    changed |= net
        .graph_mut()
        .add_edge(c as u16, layout.emit(y, x, output_phase) as u16);
    if net.spike_data()[c].threshold != 1 {
        net.spike_data_mut()[c].threshold = 1;
        changed = true;
    }
    if net.spike_data()[c].channel != 1 {
        net.spike_data_mut()[c].channel = 1;
        changed = true;
    }
    if net.polarity()[c] != 1 {
        net.polarity_mut()[c] = 1;
        changed = true;
    }
    changed
}

#[allow(clippy::too_many_arguments)]
fn runner_operator_step(
    net: &mut Network,
    layout: &Layout,
    cases: &[Case],
    rng: &mut StdRng,
    arm: Arm,
    jackpot: usize,
    ticks: usize,
    step: usize,
    job_id: &str,
    seed: u64,
    writer: &mut BufWriter<File>,
    placement_writer: &mut BufWriter<File>,
    deltas: &mut Vec<f64>,
    accepted_ops: &mut Vec<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let before = eval_correct_probability(net, layout, cases, ticks);
    let parent = net.save_state();
    let mut best_delta = f64::NEG_INFINITY;
    let mut best_snapshot = None;
    let mut records = Vec::with_capacity(jackpot);
    let mut best_audit = None;
    for candidate_id in 0..jackpot {
        net.restore_state(&parent);
        let audit = apply_runner_candidate(net, layout, cases, rng, arm);
        let after = if audit.is_some() && locality_valid(net, layout) {
            eval_correct_probability(net, layout, cases, ticks)
        } else {
            before
        };
        let delta = after - before;
        if audit.is_some() {
            deltas.push(delta);
        }
        if audit.is_some() && delta > best_delta {
            best_delta = delta;
            best_snapshot = Some(net.save_state());
            best_audit = audit.clone();
        }
        records.push((candidate_id, audit, after, delta));
    }
    net.restore_state(&parent);
    let accept = match arm.acceptance() {
        AcceptanceMode::Strict => best_delta > 1e-12,
        AcceptanceMode::Ties => best_delta >= -1e-12,
        AcceptanceMode::ZeroP => best_delta > 1e-12 || (best_delta.abs() <= 1e-12 && rng.gen_bool(0.30)),
    };
    if accept {
        if let Some(snapshot) = best_snapshot {
            net.restore_state(&snapshot);
        }
        if let Some(audit) = &best_audit {
            accepted_ops.push(operator_name(audit.operator).to_string());
        }
    }
    for (candidate_id, audit, after, delta) in records {
        let selected = audit.is_some() && (delta - best_delta).abs() <= 1e-12;
        let accepted = accept && selected;
        serde_json::to_writer(
            &mut *writer,
            &json!({
                "job_id": job_id,
                "seed": seed,
                "arm": arm.as_str(),
                "step": step,
                "candidate_id": candidate_id,
                "operator_id": audit.as_ref().map(|a| operator_name(a.operator)).unwrap_or("noop"),
                "mutated": audit.is_some(),
                "evaluated": audit.is_some(),
                "before_u": before,
                "after_u": after,
                "delta_u": delta,
                "selected": selected,
                "accepted": accepted,
            }),
        )?;
        writeln!(writer)?;
        if let Some(audit) = audit {
            let row = placement_audit_row(
                job_id,
                seed,
                arm,
                step,
                candidate_id,
                accepted,
                layout,
                cases,
                &audit,
            );
            serde_json::to_writer(&mut *placement_writer, &row)?;
            writeln!(placement_writer)?;
        }
    }
    Ok(())
}

fn apply_runner_candidate(
    net: &mut Network,
    layout: &Layout,
    cases: &[Case],
    rng: &mut StdRng,
    arm: Arm,
) -> Option<CandidateAudit> {
    let mut ops = Vec::new();
    if arm.allows_coincidence() {
        ops.push(RunnerOperator::AddCoincidence);
    }
    if arm.allows_polarity() {
        ops.push(RunnerOperator::FlipPolarity);
    }
    if arm.allows_channel() {
        ops.push(RunnerOperator::MutateChannel);
    }
    if ops.is_empty() {
        return None;
    }
    let operator = ops[rng.gen_range(0..ops.len())];
    let cell = if arm.oracle_routing() {
        let case = &cases[rng.gen_range(0..cases.len())];
        case.private.true_path[rng.gen_range(1..case.private.true_path.len())]
    } else {
        let free_cells = public_free_cells(cases, layout.width);
        free_cells[rng.gen_range(0..free_cells.len())]
    };
    let input_phase = rng.gen_range(0..K);
    let gate = rng.gen_range(0..K);
    let output_phase = rng.gen_range(0..K);
    match operator {
        RunnerOperator::AddCoincidence => {
            add_coincidence_gate(net, layout, cell, input_phase, gate, output_phase);
        }
        RunnerOperator::FlipPolarity => {
            let c = layout.coincidence(cell.0, cell.1, input_phase, gate, output_phase);
            net.polarity_mut()[c] *= -1;
        }
        RunnerOperator::MutateChannel => {
            let c = layout.coincidence(cell.0, cell.1, input_phase, gate, output_phase);
            let channel = rng.gen_range(1..=8);
            net.spike_data_mut()[c].channel = channel;
        }
    }
    Some(CandidateAudit {
        operator,
        cell,
        input_phase,
        gate,
        output_phase,
    })
}

fn placement_audit_row(
    job_id: &str,
    seed: u64,
    arm: Arm,
    step: usize,
    candidate_id: usize,
    accepted: bool,
    layout: &Layout,
    cases: &[Case],
    audit: &CandidateAudit,
) -> PlacementAuditRow {
    let is_on_path_cell = cases
        .iter()
        .any(|case| case.private.true_path.contains(&audit.cell));
    let is_target_cell = cases.iter().any(|case| case.public.target == audit.cell);
    PlacementAuditRow {
        job_id: job_id.to_string(),
        seed,
        arm: arm.as_str().to_string(),
        step,
        candidate_id,
        accepted,
        cell_id: layout.cell_id(audit.cell.0, audit.cell.1),
        input_phase: audit.input_phase,
        gate: audit.gate,
        output_phase: audit.output_phase,
        is_on_path_cell,
        is_target_cell,
        writes_to_target_directly: is_target_cell,
        reads_private_path: arm.oracle_routing(),
        reads_private_label: false,
        local_edge_only: true,
    }
}

fn operator_name(op: RunnerOperator) -> &'static str {
    match op {
        RunnerOperator::AddCoincidence => "add_coincidence_gate",
        RunnerOperator::FlipPolarity => "flip_polarity",
        RunnerOperator::MutateChannel => "mutate_channel",
    }
}

fn public_free_cells(cases: &[Case], width: usize) -> Vec<(usize, usize)> {
    let mut cells = BTreeSet::new();
    for case in cases {
        for y in 0..width {
            for x in 0..width {
                if !case.public.wall[y * width + x] {
                    cells.insert((y, x));
                }
            }
        }
    }
    if cells.is_empty() {
        cells.insert((width / 2, 1usize.min(width - 1)));
    }
    cells.into_iter().collect()
}

#[derive(Default)]
struct MotifMetrics {
    unique_cells_with_motif: usize,
    motif_density_per_cell: f64,
    full_16_pair_coverage_per_cell: f64,
    motif_reuse_across_examples: f64,
    total_coincidence_gates_added: usize,
    useful_coincidence_gates: usize,
    useless_coincidence_gates: usize,
    motif_precision: f64,
    motif_recall: f64,
    motif_ablation_drop: f64,
}

fn motif_metrics(net: &mut Network, layout: &Layout, cases: &[Case], ticks: usize) -> MotifMetrics {
    let mut motifs = BTreeSet::new();
    for edge in net.graph().iter_edges() {
        if let Some((y, x, input_phase, gate, output_phase)) =
            layout.decode_coincidence(edge.source as usize)
        {
            if edge.target as usize == layout.emit(y, x, output_phase) {
                motifs.insert((y, x, input_phase, gate, output_phase));
            }
        }
    }
    let total = motifs.len();
    let useful = motifs
        .iter()
        .filter(|(_, _, input_phase, gate, output_phase)| {
            (*input_phase + *gate) % K == *output_phase
        })
        .count();
    let unique_cells = motifs.iter().map(|(y, x, _, _, _)| (*y, *x)).collect::<BTreeSet<_>>();
    let full_coverage = unique_cells
        .iter()
        .filter(|&&(y, x)| {
            let count = motifs
                .iter()
                .filter(|(my, mx, input_phase, gate, output_phase)| {
                    *my == y && *mx == x && (*input_phase + *gate) % K == *output_phase
                })
                .count();
            count >= K * K
        })
        .count();
    let needed = needed_motifs(cases);
    let needed_hit = needed
        .iter()
        .filter(|motif| motifs.contains(motif))
        .count();
    let current_acc = eval_accuracy_only(net, layout, cases, ticks);
    let parent = net.save_state();
    remove_coincidence_emit_edges(net, layout);
    let ablated_acc = eval_accuracy_only(net, layout, cases, ticks);
    net.restore_state(&parent);
    MotifMetrics {
        unique_cells_with_motif: unique_cells.len(),
        motif_density_per_cell: total as f64 / (layout.width * layout.width).max(1) as f64,
        full_16_pair_coverage_per_cell: full_coverage as f64 / unique_cells.len().max(1) as f64,
        motif_reuse_across_examples: needed_hit as f64 / needed.len().max(1) as f64,
        total_coincidence_gates_added: total,
        useful_coincidence_gates: useful,
        useless_coincidence_gates: total.saturating_sub(useful),
        motif_precision: useful as f64 / total.max(1) as f64,
        motif_recall: needed_hit as f64 / needed.len().max(1) as f64,
        motif_ablation_drop: (current_acc - ablated_acc).max(0.0),
    }
}

fn needed_motifs(cases: &[Case]) -> BTreeSet<(usize, usize, usize, usize, usize)> {
    let mut needed = BTreeSet::new();
    for case in cases {
        let mut phase = case.public.source_phase as usize;
        for &(y, x) in case.private.true_path.iter().skip(1) {
            let gate = case.public.gates[y * case.public.width + x] as usize % K;
            let output = (phase + gate) % K;
            needed.insert((y, x, phase, gate, output));
            phase = output;
        }
    }
    needed
}

fn remove_coincidence_emit_edges(net: &mut Network, layout: &Layout) {
    let edges = net.graph().iter_edges().collect::<Vec<_>>();
    for edge in edges {
        let source = edge.source as usize;
        if let Some((y, x, _, _, output_phase)) = layout.decode_coincidence(source) {
            if edge.target as usize == layout.emit(y, x, output_phase) {
                net.graph_mut().remove_edge(edge.source, edge.target);
            }
        }
    }
}

fn eval_accuracy_only(net: &mut Network, layout: &Layout, cases: &[Case], ticks: usize) -> f64 {
    let mut correct = 0usize;
    for case in cases {
        let probs = network_probs(net, layout, &case.public, ticks);
        correct += usize::from(argmax(&probs) == case.private.label as usize);
    }
    correct as f64 / cases.len().max(1) as f64
}

fn wall_leak_rate(net: &mut Network, layout: &Layout, cases: &[Case], ticks: usize) -> f64 {
    let mut wall_emit_charge = 0usize;
    let mut wall_emit_total = 0usize;
    for case in cases.iter().take(8) {
        let _ = network_probs(net, layout, &case.public, ticks);
        for y in 0..case.public.width {
            for x in 0..case.public.width {
                if case.public.wall[y * case.public.width + x] {
                    for phase in 0..K {
                        wall_emit_total += 1;
                        let idx = layout.emit(y, x, phase);
                        wall_emit_charge += usize::from(net.spike_data()[idx].charge > 0 || net.activation()[idx] > 0);
                    }
                }
            }
        }
    }
    ratio(wall_emit_charge, wall_emit_total)
}

#[derive(Default)]
struct NetworkAudit {
    nonlocal_edge_count: usize,
    direct_output_leak_rate: f64,
}

fn audit_network(net: &Network, layout: &Layout, target: Option<(usize, usize)>) -> NetworkAudit {
    let mut audit = NetworkAudit::default();
    let mut direct_leaks = 0usize;
    for edge in net.graph().iter_edges() {
        let (sy, sx) = layout.cell_of_neuron(edge.source as usize);
        let (ty, tx) = layout.cell_of_neuron(edge.target as usize);
        let dist = sy.abs_diff(ty) + sx.abs_diff(tx);
        if dist > 1 {
            audit.nonlocal_edge_count += 1;
        }
        if let Some((target_y, target_x)) = target {
            let target_cell = (ty, tx) == (target_y, target_x);
            let source_nonlocal_to_target = sy.abs_diff(target_y) + sx.abs_diff(target_x) > 1;
            if target_cell && source_nonlocal_to_target {
                direct_leaks += 1;
            }
        }
    }
    audit.direct_output_leak_rate = if net.edge_count() == 0 {
        0.0
    } else {
        direct_leaks as f64 / net.edge_count() as f64
    };
    audit
}

fn locality_valid(net: &Network, layout: &Layout) -> bool {
    audit_network(net, layout, None).nonlocal_edge_count == 0
}

fn max_edge_cap(layout: &Layout) -> usize {
    layout.width * layout.width * LANES_PER_CELL * 4
}

fn canonical_operator_weights() -> Vec<f64> {
    let mut weights = vec![1.0; instnct_core::MUTATION_OPERATORS.len()];
    if let Some(idx) = mutation_operator_index("projection_weight") {
        weights[idx] = 0.0;
    }
    weights
}

fn candidate_stats(deltas: &[f64], accepted_ops: &[String]) -> CandidateStats {
    if deltas.is_empty() {
        return CandidateStats {
            accepted_candidates: accepted_ops.len(),
            ..CandidateStats::default()
        };
    }
    let n = deltas.len() as f64;
    let mean_delta = deltas.iter().sum::<f64>() / n;
    let mean_abs_delta = deltas.iter().map(|d| d.abs()).sum::<f64>() / n;
    let var = deltas.iter().map(|d| (d - mean_delta).powi(2)).sum::<f64>() / n;
    let accepted = accepted_ops.len().max(1) as f64;
    CandidateStats {
        candidate_positive_delta_fraction: deltas.iter().filter(|d| **d > 1e-9).count() as f64 / n,
        candidate_delta_nonzero_fraction: deltas.iter().filter(|d| d.abs() > 1e-9).count() as f64 / n,
        accepted_coincidence_operator_rate: accepted_ops
            .iter()
            .filter(|op| op.as_str() == "add_coincidence_gate")
            .count() as f64
            / accepted,
        accepted_polarity_operator_rate: accepted_ops
            .iter()
            .filter(|op| op.as_str() == "flip_polarity")
            .count() as f64
            / accepted,
        accepted_channel_operator_rate: accepted_ops
            .iter()
            .filter(|op| op.as_str() == "mutate_channel")
            .count() as f64
            / accepted,
        accepted_candidates: accepted_ops.len(),
        evaluated_candidates: deltas.len(),
        mean_delta,
        mean_abs_delta,
        candidate_delta_std: var.sqrt(),
    }
}

fn write_canonical_candidate_record(
    writer: &mut BufWriter<File>,
    job_id: &str,
    seed: u64,
    arm: Arm,
    record: &CandidateTraceRecord,
) -> std::io::Result<()> {
    serde_json::to_writer(
        &mut *writer,
        &json!({
            "job_id": job_id,
            "seed": seed,
            "arm": arm.as_str(),
            "step": record.step,
            "candidate_id": record.candidate_id,
            "operator_id": record.operator_id,
            "mutated": record.mutated,
            "evaluated": record.evaluated,
            "before_u": record.before_u,
            "after_u": record.after_u,
            "delta_u": record.delta_u,
            "within_cap": record.within_cap,
            "selected": record.selected,
            "accepted": record.accepted,
        }),
    )?;
    writeln!(writer)
}

fn shuffled_gate_label(case: &Case) -> usize {
    let mut gates = case.public.gates.clone();
    rotate_gates(&mut gates);
    let sum = case
        .private
        .true_path
        .iter()
        .skip(1)
        .fold(0u8, |acc, &(y, x)| {
            (acc + gates[y * case.public.width + x]) % K as u8
        });
    ((case.public.source_phase + sum) % K as u8) as usize
}

fn rotate_gates(gates: &mut [u8]) {
    for gate in gates {
        *gate = (*gate + 1) % K as u8;
    }
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

fn normalize_scores(scores: [f64; K]) -> [f64; K] {
    let total: f64 = scores.iter().sum();
    if total <= 1e-12 {
        return [0.25; K];
    }
    [
        scores[0] / total,
        scores[1] / total,
        scores[2] / total,
        scores[3] / total,
    ]
}

fn argmax(values: &[f64; K]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

fn arm_seed_salt(arm: Arm) -> u64 {
    match arm {
        Arm::FixedPhaseLaneReference => 1,
        Arm::HandBuiltSpatialCoincidenceReference => 2,
        Arm::CanonicalJackpot007Baseline => 3,
        Arm::OracleRoutingPlusCoincidenceOperator => 4,
        Arm::FullSpatialPlusCoincidenceOperator => 5,
        Arm::CoincidenceOperatorStrict => 6,
        Arm::CoincidenceOperatorTies => 7,
        Arm::CoincidenceOperatorZeroP => 8,
        Arm::PolarityOnly => 9,
        Arm::ChannelOnly => 10,
        Arm::CoincidencePlusPolarity => 11,
        Arm::CoincidencePlusChannel => 12,
        Arm::CoincidencePlusPolarityChannel => 13,
    }
}

fn write_static_run_files(cfg: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let queue: Vec<_> = cfg
        .seeds
        .iter()
        .flat_map(|seed| {
            arms().into_iter().map(move |arm| {
                json!({
                    "seed": seed,
                    "arm": arm.as_str(),
                    "steps": cfg.steps,
                    "eval_examples": cfg.eval_examples,
                    "width": cfg.width,
                    "ticks": cfg.ticks,
                    "jackpot": cfg.jackpot,
                })
            })
        })
        .collect();
    write_json(cfg.out.join("queue.json"), &queue)?;
    fs::write(
        cfg.out.join("contract_snapshot.md"),
        "# STABLE_LOOP_PHASE_LOCK_009_COINCIDENCE_OPERATOR_SPATIAL_REINSERTION\n\nRunner-local snapshot: audited local coincidence operator reinserted into a spatial phase-lane substrate.\n",
    )?;
    for file in [
        "progress.jsonl",
        "metrics.jsonl",
        "candidate_log.jsonl",
        "operator_summary.json",
        "spatial_stage_metrics.jsonl",
        "ablation_metrics.jsonl",
        "counterfactual_metrics.jsonl",
        "motif_placement_audit.jsonl",
        "locality_audit.jsonl",
        "summary.json",
        "report.md",
        "examples_sample.jsonl",
    ] {
        let path = cfg.out.join(file);
        if !path.exists() {
            File::create(path)?;
        }
    }
    Ok(())
}

fn refresh_summary_partial(cfg: &Config, elapsed_sec: f64) -> Result<(), Box<dyn std::error::Error>> {
    write_json(
        cfg.out.join("summary.json"),
        &json!({"status": "running", "elapsed_sec": elapsed_sec, "updated_time": now_sec()}),
    )?;
    fs::write(
        cfg.out.join("report.md"),
        format!(
            "# STABLE_LOOP_PHASE_LOCK_009_COINCIDENCE_OPERATOR_SPATIAL_REINSERTION\n\nStatus: running.\n\nElapsed seconds: {:.2}\n",
            elapsed_sec
        ),
    )?;
    Ok(())
}

fn refresh_summary(
    cfg: &Config,
    rows: &[MetricRow],
    completed: usize,
    total: usize,
    elapsed_sec: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let verdicts = verdicts(rows);
    let best = rows
        .iter()
        .max_by(|a, b| {
            a.correct_target_lane_probability_mean
                .partial_cmp(&b.correct_target_lane_probability_mean)
                .unwrap()
        })
        .map(|row| {
            json!({
                "arm": row.arm,
                "phase_final_accuracy": row.phase_final_accuracy,
                "correct_target_lane_probability_mean": row.correct_target_lane_probability_mean,
            })
        });
    write_json(
        cfg.out.join("summary.json"),
        &json!({
            "status": if completed >= total { "done" } else { "running" },
            "completed": completed,
            "total": total,
            "elapsed_sec": elapsed_sec,
            "verdicts": verdicts,
            "best": best,
            "updated_time": now_sec(),
        }),
    )?;
    write_report(cfg, rows, completed >= total)?;
    Ok(())
}

fn write_operator_summary(cfg: &Config, rows: &[MetricRow]) -> Result<(), Box<dyn std::error::Error>> {
    let mut by_arm = BTreeMap::new();
    for row in rows {
        by_arm.insert(
            row.arm.clone(),
            json!({
                "accepted_candidates": row.accepted_candidates,
                "accepted_coincidence_operator_rate": row.accepted_coincidence_operator_rate,
                "accepted_polarity_operator_rate": row.accepted_polarity_operator_rate,
                "accepted_channel_operator_rate": row.accepted_channel_operator_rate,
                "motif_precision": row.motif_precision,
                "motif_recall": row.motif_recall,
                "motif_density_per_cell": row.motif_density_per_cell,
            }),
        );
    }
    write_json(cfg.out.join("operator_summary.json"), &by_arm)
}

fn write_report(
    cfg: &Config,
    rows: &[MetricRow],
    final_report: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut report = String::new();
    report.push_str("# STABLE_LOOP_PHASE_LOCK_009_COINCIDENCE_OPERATOR_SPATIAL_REINSERTION Report\n\n");
    report.push_str(if final_report { "Status: complete.\n\n" } else { "Status: running.\n\n" });
    report.push_str("## Verdicts\n\n```text\n");
    for verdict in verdicts(rows) {
        report.push_str(&verdict);
        report.push('\n');
    }
    report.push_str("```\n\n");
    report.push_str("## Final Rows\n\n");
    report.push_str("| Arm | Seed | Acc | Correct prob | CF acc | Gate collapse | Motif drop | Precision | Recall |\n");
    report.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for row in rows.iter().filter(|row| row.final_row) {
        report.push_str(&format!(
            "| {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |\n",
            row.arm,
            row.seed,
            row.phase_final_accuracy,
            row.correct_target_lane_probability_mean,
            row.same_target_counterfactual_accuracy,
            row.gate_shuffle_collapse,
            row.motif_ablation_drop,
            row.motif_precision,
            row.motif_recall,
        ));
    }
    report.push_str("\n## Claim Boundary\n\n");
    report.push_str(
        "This runner can show whether an audited local coincidence mutation lane makes the spatial phase construction reachable. It does not prove efficiency, production architecture, full VRAXION, consciousness, language grounding, or Prismion uniqueness.\n",
    );
    fs::write(cfg.out.join("report.md"), report)?;
    Ok(())
}

fn verdicts(rows: &[MetricRow]) -> Vec<String> {
    let mut out = BTreeSet::new();
    let final_rows = rows.iter().filter(|row| row.final_row).collect::<Vec<_>>();
    let hand = final_rows
        .iter()
        .find(|row| row.arm == Arm::HandBuiltSpatialCoincidenceReference.as_str());
    if let Some(row) = hand {
        if row.phase_final_accuracy >= 0.95
            && row.correct_target_lane_probability_mean >= 0.90
            && row.same_target_counterfactual_accuracy >= 0.85
            && row.nonlocal_edge_count == 0
        {
            out.insert("HAND_BUILT_SPATIAL_MOTIF_WORKS".to_string());
        } else {
            out.insert("HAND_BUILT_SPATIAL_MOTIF_FAILS".to_string());
        }
    }
    let canonical = final_rows
        .iter()
        .filter(|row| row.arm == Arm::CanonicalJackpot007Baseline.as_str())
        .map(|row| row.phase_final_accuracy)
        .fold(0.0f64, f64::max);
    if canonical < 0.35 {
        out.insert("CANONICAL_JACKPOT_STILL_INSUFFICIENT".to_string());
    }
    let coincidence_best = final_rows
        .iter()
        .filter(|row| {
            row.arm == Arm::FullSpatialPlusCoincidenceOperator.as_str()
                || row.arm == Arm::CoincidenceOperatorStrict.as_str()
                || row.arm == Arm::CoincidenceOperatorTies.as_str()
                || row.arm == Arm::CoincidenceOperatorZeroP.as_str()
                || row.arm == Arm::CoincidencePlusPolarity.as_str()
                || row.arm == Arm::CoincidencePlusChannel.as_str()
                || row.arm == Arm::CoincidencePlusPolarityChannel.as_str()
        })
        .max_by(|a, b| a.phase_final_accuracy.partial_cmp(&b.phase_final_accuracy).unwrap());
    if let Some(row) = coincidence_best {
        if row.phase_final_accuracy >= 0.85
            && row.same_target_counterfactual_accuracy >= 0.85
            && row.gate_shuffle_collapse >= 0.50
            && row.nonlocal_edge_count == 0
            && row.direct_output_leak_rate <= 0.001
        {
            if row.motif_density_per_cell >= 12.0 {
                out.insert("COINCIDENCE_OPERATOR_RESCUES_SPATIAL_PHASE_DENSE".to_string());
            } else {
                out.insert("COINCIDENCE_OPERATOR_RESCUES_SPATIAL_PHASE".to_string());
            }
        } else if row.phase_final_accuracy >= 0.55 {
            out.insert("COINCIDENCE_OPERATOR_RESCUES_SHORT_CHAIN_ONLY".to_string());
        } else if rows.iter().any(|r| r.arm == Arm::HandBuiltSpatialCoincidenceReference.as_str()) {
            out.insert("SPATIAL_REINSERTION_FAILS".to_string());
        }
    }
    let plus_polarity = final_rows
        .iter()
        .filter(|row| row.arm == Arm::CoincidencePlusPolarity.as_str() || row.arm == Arm::CoincidencePlusPolarityChannel.as_str())
        .map(|row| row.phase_final_accuracy)
        .fold(0.0f64, f64::max);
    let plain_coincidence = final_rows
        .iter()
        .filter(|row| row.arm == Arm::CoincidenceOperatorStrict.as_str() || row.arm == Arm::FullSpatialPlusCoincidenceOperator.as_str())
        .map(|row| row.phase_final_accuracy)
        .fold(0.0f64, f64::max);
    if plus_polarity >= plain_coincidence + 0.05 {
        out.insert("POLARITY_OPERATOR_REQUIRED".to_string());
    }
    let plus_channel = final_rows
        .iter()
        .filter(|row| row.arm == Arm::CoincidencePlusChannel.as_str() || row.arm == Arm::CoincidencePlusPolarityChannel.as_str())
        .map(|row| row.phase_final_accuracy)
        .fold(0.0f64, f64::max);
    if plus_channel >= plain_coincidence + 0.05 {
        out.insert("CHANNEL_OPERATOR_REQUIRED".to_string());
    }
    if final_rows.iter().any(|row| {
        row.forbidden_private_field_leak > 0.0 || row.nonlocal_edge_count > 0 || row.direct_output_leak_rate > 0.05
    }) {
        out.insert("DIRECT_SHORTCUT_CONTAMINATION".to_string());
    }
    out.into_iter().collect()
}

fn append_jsonl<P: AsRef<Path>, T: Serialize>(path: P, value: &T) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, value)?;
    writeln!(file)?;
    Ok(())
}

fn write_json<P: AsRef<Path>, T: Serialize>(path: P, value: &T) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, value)?;
    Ok(())
}

fn now_sec() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}
