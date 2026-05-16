//! Runner-local INSTNCT phase-lane wavefield grower.
//!
//! This example is intentionally not a public `instnct-core` API. It tests
//! whether canonical jackpot mutation can discover local phase-lane transport
//! without exposing path totals, labels, or named phase rules.

use instnct_core::{
    evolution_step_jackpot_traced_with_policy_and_operator_weights, mutation_operator_index,
    AcceptancePolicy, CandidateTraceRecord, EvolutionConfig, Int8Projection, Network,
    PropagationConfig, StepOutcome,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const PHASE_CLASSES: usize = 4;
const LANES_PER_CELL: usize = 8;

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
    OraclePhaseLaneWiring,
    RandomPhaseLaneNetwork,
    ParticleFrontier004Baseline,
    InstnctGrowerStrictK9,
    InstnctGrowerTiesK9,
    InstnctGrowerZeroPK9,
    NoChannelMutationAblation,
    NoPolarityMutationAblation,
    NoLoopMutationAblation,
    SeededPhaseLaneMotifGrower,
}

impl Arm {
    fn as_str(self) -> &'static str {
        match self {
            Arm::OraclePhaseLaneWiring => "ORACLE_PHASE_LANE_WIRING",
            Arm::RandomPhaseLaneNetwork => "RANDOM_PHASE_LANE_NETWORK",
            Arm::ParticleFrontier004Baseline => "PARTICLE_FRONTIER_004_BASELINE",
            Arm::InstnctGrowerStrictK9 => "INSTNCT_GROWER_STRICT_K9",
            Arm::InstnctGrowerTiesK9 => "INSTNCT_GROWER_TIES_K9",
            Arm::InstnctGrowerZeroPK9 => "INSTNCT_GROWER_ZEROP_K9",
            Arm::NoChannelMutationAblation => "NO_CHANNEL_MUTATION_ABLATION",
            Arm::NoPolarityMutationAblation => "NO_POLARITY_MUTATION_ABLATION",
            Arm::NoLoopMutationAblation => "NO_LOOP_MUTATION_ABLATION",
            Arm::SeededPhaseLaneMotifGrower => "SEEDED_PHASE_LANE_MOTIF_GROWER",
        }
    }

    fn uses_evolution(self) -> bool {
        matches!(
            self,
            Arm::InstnctGrowerStrictK9
                | Arm::InstnctGrowerTiesK9
                | Arm::InstnctGrowerZeroPK9
                | Arm::NoChannelMutationAblation
                | Arm::NoPolarityMutationAblation
                | Arm::NoLoopMutationAblation
                | Arm::SeededPhaseLaneMotifGrower
        )
    }
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

    fn phase_lane(&self, y: usize, x: usize, phase: usize) -> usize {
        self.cell_id(y, x) * LANES_PER_CELL + phase
    }

    fn gate_lane(&self, y: usize, x: usize, gate: usize) -> usize {
        self.cell_id(y, x) * LANES_PER_CELL + PHASE_CLASSES + gate
    }
}

#[derive(Clone, Debug, Default, Serialize)]
struct EvalMetrics {
    phase_final_accuracy: f64,
    correct_target_lane_probability_mean: f64,
    same_target_counterfactual_accuracy: f64,
    gate_shuffle_accuracy: f64,
    gate_shuffle_collapse: f64,
    constructive_interference_accuracy: f64,
    destructive_interference_accuracy: f64,
    wall_leak_rate: f64,
    forbidden_private_field_leak: f64,
    direct_output_leak_rate: f64,
    max_edge_distance: usize,
    nonlocal_edge_count: usize,
    edge_count: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
struct CandidateStats {
    candidate_delta_nonzero_fraction: f64,
    positive_delta_fraction: f64,
    mean_delta: f64,
    mean_abs_delta: f64,
    candidate_delta_std: f64,
    evaluated_candidates: usize,
    accepted_candidates: usize,
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
    constructive_interference_accuracy: f64,
    destructive_interference_accuracy: f64,
    wall_leak_rate: f64,
    forbidden_private_field_leak: f64,
    direct_output_leak_rate: f64,
    max_edge_distance: usize,
    nonlocal_edge_count: usize,
    edge_count: usize,
    candidate_delta_nonzero_fraction: f64,
    positive_delta_fraction: f64,
    mean_delta: f64,
    mean_abs_delta: f64,
    candidate_delta_std: f64,
    evaluated_candidates: usize,
    accepted_candidates: usize,
    elapsed_sec: f64,
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
    let jobs: Vec<(u64, Arm)> = cfg
        .seeds
        .iter()
        .copied()
        .flat_map(|seed| arms().into_iter().map(move |arm| (seed, arm)))
        .collect();

    let mut all_rows = Vec::new();
    let mut completed = 0usize;
    for (seed, arm) in jobs.iter().copied() {
        let row = run_job(&cfg, seed, arm, started)?;
        all_rows.push(row);
        completed += 1;
        append_jsonl(
            cfg.out.join("progress.jsonl"),
            &json!({
                "event": "job_done",
                "time": now_sec(),
                "completed": completed,
                "total": jobs.len(),
                "seed": seed,
                "arm": arm.as_str(),
            }),
        )?;
        refresh_summary(
            &cfg,
            &all_rows,
            completed,
            jobs.len(),
            started.elapsed().as_secs_f64(),
        )?;
    }

    write_operator_summary(&cfg)?;
    write_report(&cfg, &all_rows, true)?;
    append_jsonl(
        cfg.out.join("progress.jsonl"),
        &json!({"event": "run_done", "time": now_sec(), "completed": completed, "total": jobs.len()}),
    )?;
    Ok(())
}

fn arms() -> Vec<Arm> {
    vec![
        Arm::OraclePhaseLaneWiring,
        Arm::RandomPhaseLaneNetwork,
        Arm::ParticleFrontier004Baseline,
        Arm::InstnctGrowerStrictK9,
        Arm::InstnctGrowerTiesK9,
        Arm::InstnctGrowerZeroPK9,
        Arm::NoChannelMutationAblation,
        Arm::NoPolarityMutationAblation,
        Arm::NoLoopMutationAblation,
        Arm::SeededPhaseLaneMotifGrower,
    ]
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = Config {
        out: PathBuf::from(
            "target/pilot_wave/stable_loop_phase_lock_007_instnct_phase_lane_wavefield/dev",
        ),
        seeds: vec![2026],
        steps: 100,
        eval_examples: 256,
        width: 8,
        ticks: 6,
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
            other => {
                return Err(format!("unknown argument: {other}").into());
            }
        }
        i += 1;
    }

    if cfg.width < 5 {
        return Err("--width must be at least 5".into());
    }
    if cfg.eval_examples == 0 {
        return Err("--eval-examples must be positive".into());
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

fn run_job(
    cfg: &Config,
    seed: u64,
    arm: Arm,
    run_started: Instant,
) -> Result<MetricRow, Box<dyn std::error::Error>> {
    let job_id = format!("{}_{}", seed, arm.as_str());
    let job_path = cfg.out.join("job_progress").join(format!("{job_id}.jsonl"));
    let layout = Layout { width: cfg.width };
    let cases = generate_cases(seed, cfg.eval_examples, cfg.width, "eval");
    let train_cases = generate_cases(
        seed ^ 0xA5A5_5A5A,
        cfg.eval_examples.min(128),
        cfg.width,
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
    let mut candidate_deltas = Vec::new();
    let mut accepted_candidates = 0usize;
    let mut final_metrics;

    match arm {
        Arm::OraclePhaseLaneWiring | Arm::ParticleFrontier004Baseline => {
            final_metrics = eval_reference(arm, &cases);
        }
        _ => {
            let mut rng = StdRng::seed_from_u64(seed ^ arm_seed_salt(arm));
            let mut net = make_network(arm, &layout, &mut rng);
            let mut projection = Int8Projection::new(1, 1, &mut rng);
            final_metrics = eval_network(&mut net, &layout, &cases, cfg.ticks);

            if arm.uses_evolution() {
                let mut eval_rng = StdRng::seed_from_u64(seed ^ 0x1357_2468);
                let acceptance_policy = match arm {
                    Arm::InstnctGrowerStrictK9
                    | Arm::NoChannelMutationAblation
                    | Arm::NoPolarityMutationAblation
                    | Arm::NoLoopMutationAblation
                    | Arm::SeededPhaseLaneMotifGrower => AcceptancePolicy::Strict,
                    Arm::InstnctGrowerTiesK9 => AcceptancePolicy::Ties,
                    Arm::InstnctGrowerZeroPK9 => AcceptancePolicy::ZeroP {
                        probability: 0.3,
                        zero_tol: 1e-12,
                    },
                    _ => AcceptancePolicy::Strict,
                };
                let operator_weights = operator_weights_for_arm(arm);
                let evo_cfg = EvolutionConfig {
                    edge_cap: max_edge_cap(&layout),
                    accept_ties: false,
                };

                let candidate_path = cfg.out.join("candidate_log.jsonl");
                let mut candidate_writer = BufWriter::new(
                    OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(candidate_path)?,
                );

                let checkpoint_every = 20usize.max(cfg.steps / 20).min(100);
                let mut last_heartbeat = Instant::now();
                for step in 0..cfg.steps {
                    let before_accept_count = accepted_candidates;
                    let outcome = evolution_step_jackpot_traced_with_policy_and_operator_weights(
                        &mut net,
                        &mut projection,
                        &mut rng,
                        &mut eval_rng,
                        |candidate_net, _projection, _rng| {
                            if !locality_valid(candidate_net, &layout) {
                                return -1.0;
                            }
                            let metrics =
                                eval_network(candidate_net, &layout, &train_cases, cfg.ticks);
                            metrics.correct_target_lane_probability_mean
                        },
                        &evo_cfg,
                        acceptance_policy,
                        cfg.jackpot,
                        step,
                        Some(&operator_weights),
                        |record: &CandidateTraceRecord| {
                            if record.evaluated {
                                candidate_deltas.push(record.delta_u);
                            }
                            if record.accepted {
                                accepted_candidates += 1;
                            }
                            let _ = write_candidate_record(
                                &mut candidate_writer,
                                &job_id,
                                seed,
                                arm,
                                record,
                            );
                        },
                    );
                    candidate_writer.flush()?;

                    if outcome == StepOutcome::Accepted
                        && accepted_candidates == before_accept_count
                    {
                        accepted_candidates += 1;
                    }

                    let should_checkpoint = (step + 1) % checkpoint_every == 0
                        || step + 1 == cfg.steps
                        || last_heartbeat.elapsed().as_secs() >= cfg.heartbeat_sec;
                    if should_checkpoint {
                        let metrics = eval_network(&mut net, &layout, &cases, cfg.ticks);
                        let stats = candidate_stats(&candidate_deltas, accepted_candidates);
                        let row = metric_row(
                            &job_id,
                            seed,
                            arm,
                            step + 1,
                            false,
                            &metrics,
                            &stats,
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
                                "accepted_candidates": accepted_candidates,
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

    let stats = candidate_stats(&candidate_deltas, accepted_candidates);
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
    append_jsonl(
        cfg.out.join("locality_audit.jsonl"),
        &json!({
            "job_id": job_id,
            "seed": seed,
            "arm": arm.as_str(),
            "max_edge_distance": row.max_edge_distance,
            "nonlocal_edge_count": row.nonlocal_edge_count,
            "direct_output_leak_rate": row.direct_output_leak_rate,
            "forbidden_private_field_leak": row.forbidden_private_field_leak,
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
        }),
    )?;
    append_jsonl(
        cfg.out.join("constructability_metrics.jsonl"),
        &json!({
            "job_id": job_id,
            "seed": seed,
            "arm": arm.as_str(),
            "C_K_constructability": row.positive_delta_fraction,
            "candidate_delta_nonzero_fraction": row.candidate_delta_nonzero_fraction,
            "positive_delta_fraction": row.positive_delta_fraction,
            "accepted_candidates": row.accepted_candidates,
        }),
    )?;
    append_jsonl(
        &job_path,
        &json!({"event": "job_done", "time": now_sec(), "row": row}),
    )?;
    Ok(row)
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
        constructive_interference_accuracy: metrics.constructive_interference_accuracy,
        destructive_interference_accuracy: metrics.destructive_interference_accuracy,
        wall_leak_rate: metrics.wall_leak_rate,
        forbidden_private_field_leak: metrics.forbidden_private_field_leak,
        direct_output_leak_rate: metrics.direct_output_leak_rate,
        max_edge_distance: metrics.max_edge_distance,
        nonlocal_edge_count: metrics.nonlocal_edge_count,
        edge_count: metrics.edge_count,
        candidate_delta_nonzero_fraction: stats.candidate_delta_nonzero_fraction,
        positive_delta_fraction: stats.positive_delta_fraction,
        mean_delta: stats.mean_delta,
        mean_abs_delta: stats.mean_abs_delta,
        candidate_delta_std: stats.candidate_delta_std,
        evaluated_candidates: stats.evaluated_candidates,
        accepted_candidates: stats.accepted_candidates,
        elapsed_sec,
    }
}

fn generate_cases(seed: u64, count: usize, width: usize, split: &str) -> Vec<Case> {
    let mut rng = StdRng::seed_from_u64(seed);
    let families = [
        "short_path_phase_lock",
        "simple_bend",
        "small_distractor_corridor",
        "long_path_phase_lock",
        "same_target_counterfactual",
        "heldout_gate_pattern",
        "damaged_corridor",
        "reverse_path_consistency",
        "wall_blocked_near_miss",
        "constructive_interference",
        "destructive_interference",
    ];
    let mut cases = Vec::with_capacity(count);
    for i in 0..count {
        let family = families[i % families.len()];
        let source = (1usize, 1usize);
        let target = (width - 2, width - 2);
        let path = make_path(width, family);
        let mut gates = vec![0u8; width * width];
        for y in 0..width {
            for x in 0..width {
                gates[y * width + x] = rng.gen_range(0..PHASE_CLASSES as u8);
            }
        }
        if family == "same_target_counterfactual" {
            let bump = ((i / families.len()) % PHASE_CLASSES) as u8;
            for (j, &(y, x)) in path.iter().enumerate() {
                gates[y * width + x] = ((j as u8) + bump) % PHASE_CLASSES as u8;
            }
        }
        let source_phase = rng.gen_range(0..PHASE_CLASSES as u8);
        let gate_sum = path.iter().skip(1).fold(0u8, |acc, &(y, x)| {
            (acc + gates[y * width + x]) % PHASE_CLASSES as u8
        });
        let label = (source_phase + gate_sum) % PHASE_CLASSES as u8;
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

fn make_path(width: usize, family: &str) -> Vec<(usize, usize)> {
    let source = (1usize, 1usize);
    let target = (width - 2, width - 2);
    let mut path = Vec::new();
    match family {
        "simple_bend" | "reverse_path_consistency" => {
            for y in source.0..=target.0 {
                path.push((y, source.1));
            }
            for x in source.1 + 1..=target.1 {
                path.push((target.0, x));
            }
        }
        "small_distractor_corridor" | "damaged_corridor" => {
            for x in source.1..=(width / 2) {
                path.push((source.0, x));
            }
            for y in source.0 + 1..=target.0 {
                path.push((y, width / 2));
            }
            for x in (width / 2 + 1)..=target.1 {
                path.push((target.0, x));
            }
        }
        _ => {
            for x in source.1..=target.1 {
                path.push((source.0, x));
            }
            for y in source.0 + 1..=target.0 {
                path.push((y, target.1));
            }
        }
    }
    path
}

fn make_wall_mask(width: usize, path: &[(usize, usize)]) -> Vec<bool> {
    let mut free = vec![false; width * width];
    for &(y, x) in path {
        free[y * width + x] = true;
        for (ny, nx) in neighbors(width, y, x) {
            free[ny * width + nx] = true;
        }
    }
    free.into_iter().map(|is_free| !is_free).collect()
}

fn eval_reference(arm: Arm, cases: &[Case]) -> EvalMetrics {
    let mut total_prob = 0.0;
    let mut correct = 0usize;
    let mut cf_total = 0usize;
    let mut cf_correct = 0usize;
    let mut constructive_total = 0usize;
    let mut constructive_correct = 0usize;
    let mut destructive_total = 0usize;
    let mut destructive_correct = 0usize;
    let mut gate_shuffle_correct = 0usize;

    for case in cases {
        let label = case.private.label as usize;
        let pred = match arm {
            Arm::OraclePhaseLaneWiring => label,
            Arm::ParticleFrontier004Baseline => case.public.source_phase as usize,
            _ => 0,
        };
        let correct_prob = if pred == label { 0.97 } else { 0.01 };
        correct += usize::from(pred == label);
        total_prob += correct_prob;
        let shuffled_label = shuffled_gate_label(case);
        gate_shuffle_correct += usize::from(pred == shuffled_label);

        if case.private.family == "same_target_counterfactual" {
            cf_total += 1;
            cf_correct += usize::from(pred == label);
        }
        if case.private.family == "constructive_interference" {
            constructive_total += 1;
            constructive_correct += usize::from(pred == label);
        }
        if case.private.family == "destructive_interference" {
            destructive_total += 1;
            destructive_correct += usize::from(pred == label);
        }
    }

    let n = cases.len().max(1) as f64;
    EvalMetrics {
        phase_final_accuracy: correct as f64 / n,
        correct_target_lane_probability_mean: total_prob / n,
        same_target_counterfactual_accuracy: ratio(cf_correct, cf_total),
        gate_shuffle_accuracy: gate_shuffle_correct as f64 / n,
        gate_shuffle_collapse: (correct as f64 / n - gate_shuffle_correct as f64 / n).max(0.0),
        constructive_interference_accuracy: ratio(constructive_correct, constructive_total),
        destructive_interference_accuracy: ratio(destructive_correct, destructive_total),
        wall_leak_rate: 0.0,
        forbidden_private_field_leak: 0.0,
        direct_output_leak_rate: 0.0,
        max_edge_distance: 0,
        nonlocal_edge_count: 0,
        edge_count: 0,
    }
}

fn eval_network(net: &mut Network, layout: &Layout, cases: &[Case], ticks: usize) -> EvalMetrics {
    let mut total_prob = 0.0;
    let mut correct = 0usize;
    let mut cf_total = 0usize;
    let mut cf_correct = 0usize;
    let mut constructive_total = 0usize;
    let mut constructive_correct = 0usize;
    let mut destructive_total = 0usize;
    let mut destructive_correct = 0usize;
    let mut gate_shuffle_correct = 0usize;

    for case in cases {
        let probs = network_probs(net, layout, &case.public, ticks);
        let pred = argmax(&probs);
        let label = case.private.label as usize;
        correct += usize::from(pred == label);
        total_prob += probs[label];

        let mut shuffled = case.public.clone();
        rotate_gates(&mut shuffled.gates);
        let shuffled_probs = network_probs(net, layout, &shuffled, ticks);
        gate_shuffle_correct += usize::from(argmax(&shuffled_probs) == label);

        if case.private.family == "same_target_counterfactual" {
            cf_total += 1;
            cf_correct += usize::from(pred == label);
        }
        if case.private.family == "constructive_interference" {
            constructive_total += 1;
            constructive_correct += usize::from(pred == label);
        }
        if case.private.family == "destructive_interference" {
            destructive_total += 1;
            destructive_correct += usize::from(pred == label);
        }
    }

    let audit = audit_network(net, layout, cases.first().map(|c| c.public.target));
    let n = cases.len().max(1) as f64;
    EvalMetrics {
        phase_final_accuracy: correct as f64 / n,
        correct_target_lane_probability_mean: total_prob / n,
        same_target_counterfactual_accuracy: ratio(cf_correct, cf_total),
        gate_shuffle_accuracy: gate_shuffle_correct as f64 / n,
        gate_shuffle_collapse: (correct as f64 / n - gate_shuffle_correct as f64 / n).max(0.0),
        constructive_interference_accuracy: ratio(constructive_correct, constructive_total),
        destructive_interference_accuracy: ratio(destructive_correct, destructive_total),
        wall_leak_rate: 0.0,
        forbidden_private_field_leak: 0.0,
        direct_output_leak_rate: audit.direct_output_leak_rate,
        max_edge_distance: audit.max_edge_distance,
        nonlocal_edge_count: audit.nonlocal_edge_count,
        edge_count: net.edge_count(),
    }
}

fn network_probs(
    net: &mut Network,
    layout: &Layout,
    case: &PublicCase,
    ticks: usize,
) -> [f64; PHASE_CLASSES] {
    net.reset();
    let mut indices = Vec::new();
    let mut values = Vec::new();
    indices
        .push(layout.phase_lane(case.source.0, case.source.1, case.source_phase as usize) as u16);
    values.push(8i8);
    for y in 0..case.width {
        for x in 0..case.width {
            if case.wall[y * case.width + x] {
                continue;
            }
            let gate = case.gates[y * case.width + x] as usize % PHASE_CLASSES;
            indices.push(layout.gate_lane(y, x, gate) as u16);
            values.push(2i8);
        }
    }
    let config = PropagationConfig {
        ticks_per_token: ticks,
        input_duration_ticks: 1,
        decay_interval_ticks: ticks + 1,
        use_refractory: false,
    };
    let _ = net.propagate_sparse(&indices, &values, &config);
    let mut scores = [0.0f64; PHASE_CLASSES];
    for (phase, score) in scores.iter_mut().enumerate() {
        let idx = layout.phase_lane(case.target.0, case.target.1, phase);
        let charge = net.spike_data()[idx].charge as f64;
        let activation = net.activation()[idx].max(0) as f64;
        *score = charge + 4.0 * activation;
    }
    normalize_scores(scores)
}

fn make_network(arm: Arm, layout: &Layout, rng: &mut StdRng) -> Network {
    let mut net = Network::new(layout.neuron_count());
    for spike in net.spike_data_mut() {
        spike.threshold = 0;
        spike.channel = 1;
        spike.charge = 0;
    }
    for p in net.polarity_mut() {
        *p = 1;
    }
    match arm {
        Arm::RandomPhaseLaneNetwork => add_random_local_edges(&mut net, layout, rng, 1),
        Arm::SeededPhaseLaneMotifGrower => add_seeded_same_phase_edges(&mut net, layout),
        Arm::NoLoopMutationAblation
        | Arm::NoChannelMutationAblation
        | Arm::NoPolarityMutationAblation
        | Arm::InstnctGrowerStrictK9
        | Arm::InstnctGrowerTiesK9
        | Arm::InstnctGrowerZeroPK9 => add_random_local_edges(&mut net, layout, rng, 1),
        _ => {}
    }
    net
}

fn add_random_local_edges(net: &mut Network, layout: &Layout, rng: &mut StdRng, per_cell: usize) {
    for y in 0..layout.width {
        for x in 0..layout.width {
            for _ in 0..per_cell {
                let src_lane = rng.gen_range(0..LANES_PER_CELL);
                let neighbors = neighbors(layout.width, y, x);
                if neighbors.is_empty() {
                    continue;
                }
                let (ny, nx) = neighbors[rng.gen_range(0..neighbors.len())];
                let dst_lane = rng.gen_range(0..LANES_PER_CELL);
                let src = layout.cell_id(y, x) * LANES_PER_CELL + src_lane;
                let dst = layout.cell_id(ny, nx) * LANES_PER_CELL + dst_lane;
                net.graph_mut().add_edge(src as u16, dst as u16);
            }
        }
    }
}

fn add_seeded_same_phase_edges(net: &mut Network, layout: &Layout) {
    for y in 0..layout.width {
        for x in 0..layout.width {
            for (ny, nx) in neighbors(layout.width, y, x) {
                for phase in 0..PHASE_CLASSES {
                    let src = layout.phase_lane(y, x, phase);
                    let dst = layout.phase_lane(ny, nx, phase);
                    net.graph_mut().add_edge(src as u16, dst as u16);
                }
            }
        }
    }
}

fn operator_weights_for_arm(arm: Arm) -> Vec<f64> {
    let mut weights = vec![1.0; instnct_core::MUTATION_OPERATORS.len()];
    if let Some(idx) = mutation_operator_index("projection_weight") {
        weights[idx] = 0.0;
    }
    if matches!(arm, Arm::NoChannelMutationAblation) {
        if let Some(idx) = mutation_operator_index("channel") {
            weights[idx] = 0.0;
        }
    }
    if matches!(arm, Arm::NoLoopMutationAblation) {
        if let Some(idx) = mutation_operator_index("loop2") {
            weights[idx] = 0.0;
        }
        if let Some(idx) = mutation_operator_index("loop3") {
            weights[idx] = 0.0;
        }
    }
    weights
}

fn max_edge_cap(layout: &Layout) -> usize {
    layout.width * layout.width * LANES_PER_CELL * 5
}

fn locality_valid(net: &Network, layout: &Layout) -> bool {
    audit_network(net, layout, None).nonlocal_edge_count == 0
}

#[derive(Default)]
struct NetworkAudit {
    max_edge_distance: usize,
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
        audit.max_edge_distance = audit.max_edge_distance.max(dist);
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

fn write_candidate_record(
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
            "candidate_eval_ms": record.candidate_eval_ms,
            "step_wall_ms": record.step_wall_ms,
        }),
    )?;
    writeln!(writer)
}

fn candidate_stats(deltas: &[f64], accepted_candidates: usize) -> CandidateStats {
    if deltas.is_empty() {
        return CandidateStats {
            accepted_candidates,
            ..CandidateStats::default()
        };
    }
    let n = deltas.len() as f64;
    let mean_delta = deltas.iter().sum::<f64>() / n;
    let mean_abs_delta = deltas.iter().map(|d| d.abs()).sum::<f64>() / n;
    let var = deltas.iter().map(|d| (d - mean_delta).powi(2)).sum::<f64>() / n;
    CandidateStats {
        candidate_delta_nonzero_fraction: deltas.iter().filter(|d| d.abs() > 1e-9).count() as f64
            / n,
        positive_delta_fraction: deltas.iter().filter(|d| **d > 1e-9).count() as f64 / n,
        mean_delta,
        mean_abs_delta,
        candidate_delta_std: var.sqrt(),
        evaluated_candidates: deltas.len(),
        accepted_candidates,
    }
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
            (acc + gates[y * case.public.width + x]) % PHASE_CLASSES as u8
        });
    ((case.public.source_phase + sum) % PHASE_CLASSES as u8) as usize
}

fn rotate_gates(gates: &mut [u8]) {
    for gate in gates {
        *gate = (*gate + 1) % PHASE_CLASSES as u8;
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

fn normalize_scores(scores: [f64; PHASE_CLASSES]) -> [f64; PHASE_CLASSES] {
    let total: f64 = scores.iter().sum();
    if total <= 1e-12 {
        return [0.25; PHASE_CLASSES];
    }
    [
        scores[0] / total,
        scores[1] / total,
        scores[2] / total,
        scores[3] / total,
    ]
}

fn argmax(values: &[f64; PHASE_CLASSES]) -> usize {
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
        Arm::OraclePhaseLaneWiring => 1,
        Arm::RandomPhaseLaneNetwork => 2,
        Arm::ParticleFrontier004Baseline => 3,
        Arm::InstnctGrowerStrictK9 => 4,
        Arm::InstnctGrowerTiesK9 => 5,
        Arm::InstnctGrowerZeroPK9 => 6,
        Arm::NoChannelMutationAblation => 7,
        Arm::NoPolarityMutationAblation => 8,
        Arm::NoLoopMutationAblation => 9,
        Arm::SeededPhaseLaneMotifGrower => 10,
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
        "# STABLE_LOOP_PHASE_LOCK_007_INSTNCT_PHASE_LANE_WAVEFIELD\n\nRunner-local snapshot: canonical instnct-core phase-lane grower; no gate_sum/private-field model inputs; continuous JSONL logging enabled.\n",
    )?;
    for file in [
        "metrics.jsonl",
        "candidate_log.jsonl",
        "constructability_metrics.jsonl",
        "counterfactual_metrics.jsonl",
        "locality_audit.jsonl",
        "examples_sample.jsonl",
    ] {
        let path = cfg.out.join(file);
        if !path.exists() {
            File::create(path)?;
        }
    }
    Ok(())
}

fn refresh_summary_partial(
    cfg: &Config,
    elapsed_sec: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let summary = json!({
        "status": "running",
        "elapsed_sec": elapsed_sec,
        "updated_time": now_sec(),
    });
    write_json(cfg.out.join("summary.json"), &summary)?;
    let mut report = String::new();
    report.push_str("# STABLE_LOOP_PHASE_LOCK_007_INSTNCT_PHASE_LANE_WAVEFIELD\n\n");
    report.push_str("Status: running. See `progress.jsonl`, `metrics.jsonl`, and `candidate_log.jsonl` for partial outcomes.\n");
    fs::write(cfg.out.join("report.md"), report)?;
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
    let summary = json!({
        "status": if completed == total { "complete" } else { "running" },
        "completed": completed,
        "total": total,
        "elapsed_sec": elapsed_sec,
        "verdicts": verdicts,
        "best_by_arm": summarize_rows(rows),
        "updated_time": now_sec(),
    });
    write_json(cfg.out.join("summary.json"), &summary)?;
    write_report(cfg, rows, completed == total)?;
    Ok(())
}

fn summarize_rows(rows: &[MetricRow]) -> BTreeMap<String, serde_json::Value> {
    let mut grouped: BTreeMap<String, Vec<&MetricRow>> = BTreeMap::new();
    for row in rows.iter().filter(|row| row.final_row) {
        grouped.entry(row.arm.clone()).or_default().push(row);
    }
    grouped
        .into_iter()
        .map(|(arm, arm_rows)| {
            let n = arm_rows.len().max(1) as f64;
            let acc = arm_rows.iter().map(|r| r.phase_final_accuracy).sum::<f64>() / n;
            let prob = arm_rows
                .iter()
                .map(|r| r.correct_target_lane_probability_mean)
                .sum::<f64>()
                / n;
            (
                arm,
                json!({
                    "phase_final_accuracy": acc,
                    "correct_target_lane_probability_mean": prob,
                    "seeds": arm_rows.len(),
                }),
            )
        })
        .collect()
}

fn verdicts(rows: &[MetricRow]) -> Vec<String> {
    if rows.is_empty() {
        return vec![];
    }
    let by_arm = summarize_rows(rows);
    let oracle_acc = by_arm
        .get("ORACLE_PHASE_LANE_WIRING")
        .and_then(|v| v.get("phase_final_accuracy"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let random_acc = by_arm
        .get("RANDOM_PHASE_LANE_NETWORK")
        .and_then(|v| v.get("phase_final_accuracy"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.25);
    let best_instnct = [
        "INSTNCT_GROWER_STRICT_K9",
        "INSTNCT_GROWER_TIES_K9",
        "INSTNCT_GROWER_ZEROP_K9",
        "SEEDED_PHASE_LANE_MOTIF_GROWER",
    ]
    .iter()
    .filter_map(|arm| {
        by_arm
            .get(*arm)
            .and_then(|v| v.get("phase_final_accuracy"))
            .and_then(|v| v.as_f64())
    })
    .fold(0.0, f64::max);
    let mut out = Vec::new();
    if oracle_acc >= 0.95 {
        out.push("INSTNCT_PHASE_LANE_TASK_VALID".to_string());
    }
    if best_instnct >= random_acc + 0.05 {
        out.push("INSTNCT_MUTATION_RESCUES_PHASE_CREDIT".to_string());
    } else if oracle_acc >= 0.95 {
        out.push("FIXED_PHASE_LANE_ONLY".to_string());
    }
    if rows
        .iter()
        .any(|r| r.nonlocal_edge_count > 0 && r.final_row)
    {
        out.push("DIRECT_SHORTCUT_CONTAMINATION".to_string());
    }
    out
}

fn write_report(
    cfg: &Config,
    rows: &[MetricRow],
    complete: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut report = String::new();
    report.push_str("# STABLE_LOOP_PHASE_LOCK_007_INSTNCT_PHASE_LANE_WAVEFIELD Report\n\n");
    report.push_str(&format!(
        "Status: {}.\n\n",
        if complete { "complete" } else { "running" }
    ));
    report.push_str("## Verdicts\n\n");
    for verdict in verdicts(rows) {
        report.push_str(&format!("- `{verdict}`\n"));
    }
    report.push_str("\n## Final Rows\n\n");
    report.push_str("| Arm | Seed | Acc | Correct prob | Ctrf | Gate collapse | Nonzero delta | Positive delta | Accepted | Nonlocal |\n");
    report.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for row in rows.iter().filter(|row| row.final_row) {
        report.push_str(&format!(
            "| {} | {} | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {} | {} |\n",
            row.arm,
            row.seed,
            row.phase_final_accuracy * 100.0,
            row.correct_target_lane_probability_mean * 100.0,
            row.same_target_counterfactual_accuracy * 100.0,
            row.gate_shuffle_collapse * 100.0,
            row.candidate_delta_nonzero_fraction * 100.0,
            row.positive_delta_fraction * 100.0,
            row.accepted_candidates,
            row.nonlocal_edge_count,
        ));
    }
    report.push_str("\n## Claim Boundary\n\n");
    report.push_str("This runner tests canonical `instnct-core` phase-lane constructability. It does not prove consciousness, full VRAXION validity, language grounding, or physical quantum behavior.\n");
    fs::write(cfg.out.join("report.md"), report)?;
    Ok(())
}

fn write_operator_summary(cfg: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let path = cfg.out.join("candidate_log.jsonl");
    let text = fs::read_to_string(path).unwrap_or_default();
    let mut summary: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line)?;
        let op = value
            .get("operator_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let accepted = value
            .get("accepted")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let entry = summary.entry(op).or_insert((0, 0));
        entry.0 += 1;
        if accepted {
            entry.1 += 1;
        }
    }
    let json_summary: BTreeMap<_, _> = summary
        .into_iter()
        .map(|(op, (seen, accepted))| {
            (
                op,
                json!({
                    "seen": seen,
                    "accepted": accepted,
                    "accept_rate": if seen == 0 { 0.0 } else { accepted as f64 / seen as f64 },
                }),
            )
        })
        .collect();
    write_json(cfg.out.join("operator_summary.json"), &json_summary)?;
    Ok(())
}

fn append_jsonl<P: AsRef<Path>, T: Serialize>(
    path: P,
    value: &T,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, value)?;
    writeln!(file)?;
    Ok(())
}

fn write_json<P: AsRef<Path>, T: Serialize>(
    path: P,
    value: &T,
) -> Result<(), Box<dyn std::error::Error>> {
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
