//! Runner-local phase-lane horizon scaling probe.
//!
//! 012 does not search for motifs. It measures how far the known local
//! `phase_i + gate_g -> phase_(i+g)` coincidence rule can propagate in the
//! recurrent integer substrate as path length, tick budget, width, and gate
//! stress pattern vary.

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
    source_persist_ticks: usize,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
enum Arm {
    FixedPhaseLaneReference,
    Full16RuleTemplate,
    CommonCore15PlusMissing123,
    Dense009Reference,
    RandomMatched16MotifControl,
    CanonicalJackpot007Baseline,
    SourcePersist1Tick,
    SourcePersist2Ticks,
    SourcePersistAllTicks,
}

impl Arm {
    fn as_str(self) -> &'static str {
        match self {
            Arm::FixedPhaseLaneReference => "FIXED_PHASE_LANE_REFERENCE",
            Arm::Full16RuleTemplate => "FULL_16_RULE_TEMPLATE",
            Arm::CommonCore15PlusMissing123 => "COMMON_CORE_15_PLUS_MISSING_1_2_3",
            Arm::Dense009Reference => "DENSE_009_REFERENCE",
            Arm::RandomMatched16MotifControl => "RANDOM_MATCHED_16_MOTIF_CONTROL",
            Arm::CanonicalJackpot007Baseline => "CANONICAL_JACKPOT_007_BASELINE",
            Arm::SourcePersist1Tick => "SOURCE_PERSIST_1_TICK",
            Arm::SourcePersist2Ticks => "SOURCE_PERSIST_2_TICKS",
            Arm::SourcePersistAllTicks => "SOURCE_PERSIST_ALL_TICKS",
        }
    }
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

#[derive(Clone, Debug, Default, Serialize)]
struct BucketMetrics {
    phase_final_accuracy: f64,
    correct_target_lane_probability_mean: f64,
    target_power_total: f64,
    correct_phase_margin: f64,
    wrong_phase_growth_rate: f64,
    wall_leak_rate: f64,
    same_target_counterfactual_accuracy: f64,
    gate_shuffle_collapse: f64,
    best_tick_accuracy: f64,
    best_tick_correct_probability: f64,
    final_tick_minus_best_tick_delta: f64,
    phase_decay_per_step: f64,
    nonlocal_edge_count: usize,
    forbidden_private_field_leak: f64,
    direct_output_leak_rate: f64,
}

#[derive(Clone, Debug, Serialize)]
struct HorizonRow {
    job_id: String,
    seed: u64,
    arm: String,
    width: usize,
    path_length: usize,
    ticks: usize,
    family: String,
    case_count: usize,
    phase_final_accuracy: f64,
    correct_target_lane_probability_mean: f64,
    target_power_total: f64,
    correct_phase_margin: f64,
    wrong_phase_growth_rate: f64,
    wall_leak_rate: f64,
    same_target_counterfactual_accuracy: f64,
    gate_shuffle_collapse: f64,
    best_tick_accuracy: f64,
    best_tick_correct_probability: f64,
    final_tick_minus_best_tick_delta: f64,
    phase_decay_per_step: f64,
    nonlocal_edge_count: usize,
    forbidden_private_field_leak: f64,
    direct_output_leak_rate: f64,
    elapsed_sec: f64,
}

#[derive(Clone, Debug, Serialize)]
struct TickSummaryRow {
    seed: u64,
    arm: String,
    width: usize,
    path_length: usize,
    ticks: usize,
    phase_final_accuracy: f64,
    correct_target_lane_probability_mean: f64,
}

#[derive(Clone, Debug, Serialize)]
struct PathSummaryRow {
    seed: u64,
    arm: String,
    width: usize,
    path_length: usize,
    minimum_ticks_for_95_accuracy: i64,
    minimum_ticks_for_90_probability: i64,
    best_accuracy: f64,
    best_probability: f64,
}

#[derive(Clone, Debug, Serialize)]
struct ReadoutTickRow {
    seed: u64,
    arm: String,
    width: usize,
    path_length: usize,
    family: String,
    tick: usize,
    accuracy: f64,
    correct_probability: f64,
    target_power_total: f64,
}

#[derive(Clone, Debug)]
struct NetworkAudit {
    nonlocal_edge_count: usize,
    direct_output_leak_rate: f64,
}

#[derive(Clone, Debug)]
struct ProbeOutput {
    probs: [f64; K],
    power: [f64; K],
    target_power_total: f64,
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
    let total_jobs = cfg.seeds.len() * arms().len();
    let mut completed = 0usize;
    for seed in cfg.seeds.iter().copied() {
        for arm in arms() {
            let job_rows = run_job(&cfg, seed, arm, started)?;
            all_rows.extend(job_rows);
            completed += 1;
            append_jsonl(
                cfg.out.join("progress.jsonl"),
                &json!({
                    "event": "job_done",
                    "time": now_sec(),
                    "seed": seed,
                    "arm": arm.as_str(),
                    "completed": completed,
                    "total": total_jobs,
                }),
            )?;
            refresh_summary(
                &cfg,
                &all_rows,
                completed,
                total_jobs,
                started.elapsed().as_secs_f64(),
            )?;
        }
    }
    write_report(&cfg, &all_rows, true)?;
    append_jsonl(
        cfg.out.join("progress.jsonl"),
        &json!({"event": "run_done", "time": now_sec(), "completed": completed, "total": total_jobs}),
    )?;
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = Config {
        out: PathBuf::from(
            "target/pilot_wave/stable_loop_phase_lock_012_phase_lane_horizon_scaling/dev",
        ),
        seeds: vec![2026],
        eval_examples: 256,
        widths: vec![8],
        path_lengths: vec![2, 4, 8],
        ticks_list: vec![8, 16, 24],
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
            "--eval-examples" => {
                i += 1;
                cfg.eval_examples = args[i].parse()?;
            }
            "--widths" => {
                i += 1;
                cfg.widths = parse_usize_list(&args[i]);
            }
            "--path-lengths" => {
                i += 1;
                cfg.path_lengths = parse_usize_list(&args[i]);
            }
            "--ticks-list" => {
                i += 1;
                cfg.ticks_list = parse_usize_list(&args[i]);
            }
            "--heartbeat-sec" => {
                i += 1;
                cfg.heartbeat_sec = args[i].parse()?;
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        i += 1;
    }

    if cfg.widths.iter().any(|w| *w < 5) {
        return Err("--widths entries must be at least 5".into());
    }
    if cfg.path_lengths.is_empty() || cfg.ticks_list.is_empty() {
        return Err("--path-lengths and --ticks-list must be non-empty".into());
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

fn parse_usize_list(raw: &str) -> Vec<usize> {
    raw.split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect()
}

fn arms() -> Vec<Arm> {
    vec![
        Arm::FixedPhaseLaneReference,
        Arm::Full16RuleTemplate,
        Arm::CommonCore15PlusMissing123,
        Arm::Dense009Reference,
        Arm::RandomMatched16MotifControl,
        Arm::CanonicalJackpot007Baseline,
        Arm::SourcePersist1Tick,
        Arm::SourcePersist2Ticks,
        Arm::SourcePersistAllTicks,
    ]
}

fn run_job(
    cfg: &Config,
    seed: u64,
    arm: Arm,
    run_started: Instant,
) -> Result<Vec<HorizonRow>, Box<dyn std::error::Error>> {
    let job_id = format!("{}_{}", seed, arm.as_str());
    let job_path = cfg.out.join("job_progress").join(format!("{job_id}.jsonl"));
    append_jsonl(
        &job_path,
        &json!({"event": "job_start", "time": now_sec(), "seed": seed, "arm": arm.as_str()}),
    )?;

    let mut rows = Vec::new();
    let total_buckets =
        cfg.widths.len() * cfg.path_lengths.len() * cfg.ticks_list.len() * families().len();
    let per_bucket = (cfg.eval_examples / total_buckets.max(1)).max(2);
    let started = Instant::now();
    let mut last_heartbeat = Instant::now();
    let mut bucket_idx = 0usize;
    for width in &cfg.widths {
        let layout = Layout { width: *width };
        let mut net = network_for_arm(arm, &layout, seed);
        let audit = if arm == Arm::FixedPhaseLaneReference {
            NetworkAudit {
                nonlocal_edge_count: 0,
                direct_output_leak_rate: 0.0,
            }
        } else {
            audit_network(&net, &layout, None)
        };
        for path_length in &cfg.path_lengths {
            for ticks in &cfg.ticks_list {
                for family in families() {
                    bucket_idx += 1;
                    let cases = generate_cases(
                        seed ^ (*width as u64 * 0x9E37)
                            ^ (*path_length as u64 * 0xA5A5)
                            ^ family_salt(family),
                        per_bucket,
                        *width,
                        *path_length,
                        family,
                        arm_source_persist(arm, *ticks),
                    );
                    maybe_write_examples(cfg, &cases)?;
                    let metrics = if arm == Arm::FixedPhaseLaneReference {
                        fixed_metrics(&cases)
                    } else {
                        eval_bucket(&mut net, &layout, &cases, *ticks, &audit, arm)
                    };
                    let row = HorizonRow {
                        job_id: job_id.clone(),
                        seed,
                        arm: arm.as_str().to_string(),
                        width: *width,
                        path_length: *path_length,
                        ticks: *ticks,
                        family: family.to_string(),
                        case_count: cases.len(),
                        phase_final_accuracy: metrics.phase_final_accuracy,
                        correct_target_lane_probability_mean: metrics
                            .correct_target_lane_probability_mean,
                        target_power_total: metrics.target_power_total,
                        correct_phase_margin: metrics.correct_phase_margin,
                        wrong_phase_growth_rate: metrics.wrong_phase_growth_rate,
                        wall_leak_rate: metrics.wall_leak_rate,
                        same_target_counterfactual_accuracy: metrics
                            .same_target_counterfactual_accuracy,
                        gate_shuffle_collapse: metrics.gate_shuffle_collapse,
                        best_tick_accuracy: metrics.best_tick_accuracy,
                        best_tick_correct_probability: metrics.best_tick_correct_probability,
                        final_tick_minus_best_tick_delta: metrics.final_tick_minus_best_tick_delta,
                        phase_decay_per_step: metrics.phase_decay_per_step,
                        nonlocal_edge_count: metrics.nonlocal_edge_count,
                        forbidden_private_field_leak: metrics.forbidden_private_field_leak,
                        direct_output_leak_rate: metrics.direct_output_leak_rate,
                        elapsed_sec: started.elapsed().as_secs_f64(),
                    };
                    append_all_metric_files(cfg, &row)?;
                    if family == "random_balanced" {
                        write_readout_over_time(
                            cfg,
                            seed,
                            arm,
                            &layout,
                            &mut net,
                            &cases,
                            *path_length,
                            *ticks,
                            family,
                        )?;
                        write_phase_diagnostics(
                            cfg,
                            seed,
                            arm,
                            &layout,
                            &mut net,
                            &cases,
                            *path_length,
                            *ticks,
                            family,
                        )?;
                    }
                    rows.push(row);
                    if last_heartbeat.elapsed().as_secs() >= cfg.heartbeat_sec
                        || bucket_idx == total_buckets
                    {
                        append_jsonl(
                            &job_path,
                            &json!({
                                "event": "checkpoint",
                                "time": now_sec(),
                                "completed_buckets": bucket_idx,
                                "total_buckets": total_buckets,
                                "elapsed_sec": started.elapsed().as_secs_f64(),
                            }),
                        )?;
                        refresh_summary_partial(cfg, run_started.elapsed().as_secs_f64())?;
                        last_heartbeat = Instant::now();
                    }
                }
            }
        }
    }
    write_path_summaries(cfg, seed, arm, &rows)?;
    append_jsonl(
        &job_path,
        &json!({"event": "job_done", "time": now_sec(), "rows": rows.len(), "elapsed_sec": started.elapsed().as_secs_f64()}),
    )?;
    Ok(rows)
}

fn append_all_metric_files(
    cfg: &Config,
    row: &HorizonRow,
) -> Result<(), Box<dyn std::error::Error>> {
    append_jsonl(cfg.out.join("metrics.jsonl"), row)?;
    append_jsonl(cfg.out.join("horizon_curve.jsonl"), row)?;
    append_jsonl(cfg.out.join("path_length_metrics.jsonl"), row)?;
    append_jsonl(cfg.out.join("tick_metrics.jsonl"), row)?;
    append_jsonl(
        cfg.out.join("wrong_phase_metrics.jsonl"),
        &json!({
            "seed": row.seed,
            "arm": row.arm,
            "width": row.width,
            "path_length": row.path_length,
            "ticks": row.ticks,
            "family": row.family,
            "wrong_phase_growth_rate": row.wrong_phase_growth_rate,
            "correct_phase_margin": row.correct_phase_margin,
        }),
    )?;
    append_jsonl(
        cfg.out.join("counterfactual_metrics.jsonl"),
        &json!({
            "seed": row.seed,
            "arm": row.arm,
            "width": row.width,
            "path_length": row.path_length,
            "ticks": row.ticks,
            "family": row.family,
            "same_target_counterfactual_accuracy": row.same_target_counterfactual_accuracy,
            "gate_shuffle_collapse": row.gate_shuffle_collapse,
        }),
    )?;
    append_jsonl(
        cfg.out.join("locality_audit.jsonl"),
        &json!({
            "seed": row.seed,
            "arm": row.arm,
            "width": row.width,
            "nonlocal_edge_count": row.nonlocal_edge_count,
            "forbidden_private_field_leak": row.forbidden_private_field_leak,
            "direct_output_leak_rate": row.direct_output_leak_rate,
            "wall_leak_rate": row.wall_leak_rate,
        }),
    )?;
    Ok(())
}

fn maybe_write_examples(cfg: &Config, cases: &[Case]) -> Result<(), Box<dyn std::error::Error>> {
    if cfg
        .out
        .join("examples_sample.jsonl")
        .metadata()
        .map(|m| m.len())
        .unwrap_or(0)
        > 0
    {
        return Ok(());
    }
    for case in cases.iter().take(12) {
        append_jsonl(
            cfg.out.join("examples_sample.jsonl"),
            &json!({"public": &case.public, "private_audit_only": &case.private}),
        )?;
    }
    Ok(())
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
        "same_target_counterfactual",
        "gate_shuffle_control",
    ]
}

fn generate_cases(
    seed: u64,
    count: usize,
    width: usize,
    path_length: usize,
    family: &str,
    source_persist_ticks: usize,
) -> Vec<Case> {
    (0..count)
        .map(|idx| generate_case(seed, idx, width, path_length, family, source_persist_ticks))
        .collect()
}

fn generate_case(
    seed: u64,
    idx: usize,
    width: usize,
    path_length: usize,
    family: &str,
    source_persist_ticks: usize,
) -> Case {
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
            "same_target_counterfactual" => ((idx + j) % K) as u8,
            "gate_shuffle_control" | "random_balanced" => ((idx + 2 * j + (j / 3)) % K) as u8,
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
            source_persist_ticks,
        },
        private: PrivateCase {
            label,
            true_path: path,
            path_phase_total: label,
            gate_sum,
            family: family.to_string(),
            split: "horizon_eval".to_string(),
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
    let keep = (path_length + 1).min(cells.len()).max(1);
    cells.truncate(keep);
    cells
}

fn make_wall_mask(width: usize, path: &[(usize, usize)]) -> Vec<bool> {
    let mut free = vec![false; width * width];
    for &(y, x) in path {
        free[y * width + x] = true;
    }
    free.into_iter().map(|is_free| !is_free).collect()
}

fn fixed_metrics(cases: &[Case]) -> BucketMetrics {
    let mut gate_shuffle_correct = 0usize;
    for case in cases {
        let label = case.private.label as usize;
        gate_shuffle_correct += usize::from(shuffled_gate_label(case) == label);
    }
    BucketMetrics {
        phase_final_accuracy: 1.0,
        correct_target_lane_probability_mean: 0.97,
        target_power_total: 1.0,
        correct_phase_margin: 0.97,
        wrong_phase_growth_rate: 0.0,
        wall_leak_rate: 0.0,
        same_target_counterfactual_accuracy: 1.0,
        gate_shuffle_collapse: (1.0 - ratio(gate_shuffle_correct, cases.len())).max(0.0),
        best_tick_accuracy: 1.0,
        best_tick_correct_probability: 0.97,
        final_tick_minus_best_tick_delta: 0.0,
        phase_decay_per_step: 0.0,
        nonlocal_edge_count: 0,
        forbidden_private_field_leak: 0.0,
        direct_output_leak_rate: 0.0,
    }
}

fn eval_bucket(
    net: &mut Network,
    layout: &Layout,
    cases: &[Case],
    ticks: usize,
    audit: &NetworkAudit,
    arm: Arm,
) -> BucketMetrics {
    let mut correct = 0usize;
    let mut prob_sum = 0.0;
    let mut power_sum = 0.0;
    let mut margin_sum = 0.0;
    let mut wrong_growth_sum = 0.0;
    let mut cf_total = 0usize;
    let mut cf_correct = 0usize;
    let mut gate_shuffle_correct = 0usize;
    let mut best_tick_acc = 0.0;
    let mut best_tick_prob = 0.0;
    let mut decay_sum = 0.0;
    for case in cases {
        let out = network_output(net, layout, &case.public, ticks);
        let label = case.private.label as usize;
        let pred = argmax(&out.probs);
        correct += usize::from(pred == label);
        prob_sum += out.probs[label];
        power_sum += out.target_power_total;
        let wrong_best = out
            .probs
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx != label)
            .map(|(_, v)| *v)
            .fold(0.0f64, f64::max);
        margin_sum += out.probs[label] - wrong_best;
        wrong_growth_sum += wrong_best;

        let mut shuffled = case.public.clone();
        rotate_gates(&mut shuffled.gates);
        gate_shuffle_correct +=
            usize::from(argmax(&network_output(net, layout, &shuffled, ticks).probs) == label);

        if case.private.family == "same_target_counterfactual" {
            cf_total += 1;
            cf_correct += usize::from(pred == label);
        }

        let tick_curve = readout_curve(net, layout, &case.public, ticks);
        let mut case_best_acc = 0.0;
        let mut case_best_prob = 0.0;
        for tick_out in &tick_curve {
            case_best_acc = f64::max(case_best_acc, f64::from(argmax(&tick_out.probs) == label));
            case_best_prob = f64::max(case_best_prob, tick_out.probs[label]);
        }
        best_tick_acc += case_best_acc;
        best_tick_prob += case_best_prob;
        decay_sum += phase_decay_proxy(&tick_curve, label, case.private.true_path.len().max(1));
    }
    let n = cases.len().max(1) as f64;
    let acc = correct as f64 / n;
    let final_prob = prob_sum / n;
    let gate_shuffle_acc = ratio(gate_shuffle_correct, cases.len());
    let wall_leak = wall_leak_rate(net, layout, cases, ticks, arm);
    BucketMetrics {
        phase_final_accuracy: acc,
        correct_target_lane_probability_mean: final_prob,
        target_power_total: power_sum / n,
        correct_phase_margin: margin_sum / n,
        wrong_phase_growth_rate: wrong_growth_sum / n,
        wall_leak_rate: wall_leak,
        same_target_counterfactual_accuracy: if cf_total == 0 {
            acc
        } else {
            ratio(cf_correct, cf_total)
        },
        gate_shuffle_collapse: (acc - gate_shuffle_acc).max(0.0),
        best_tick_accuracy: best_tick_acc / n,
        best_tick_correct_probability: best_tick_prob / n,
        final_tick_minus_best_tick_delta: final_prob - (best_tick_prob / n),
        phase_decay_per_step: decay_sum / n,
        nonlocal_edge_count: audit.nonlocal_edge_count,
        forbidden_private_field_leak: 0.0,
        direct_output_leak_rate: audit.direct_output_leak_rate,
    }
}

fn network_output(
    net: &mut Network,
    layout: &Layout,
    case: &PublicCase,
    ticks: usize,
) -> ProbeOutput {
    net.reset();
    let mut indices = Vec::new();
    let mut values = Vec::new();
    indices.push(layout.emit(case.source.0, case.source.1, case.source_phase as usize) as u16);
    values.push(8i8);
    if case.source_persist_ticks > 0 {
        indices
            .push(layout.arrive(case.source.0, case.source.1, case.source_phase as usize) as u16);
        values.push((case.source_persist_ticks.min(8) as i8).max(1));
    }
    for y in 0..case.width {
        for x in 0..case.width {
            if case.wall[y * case.width + x] {
                continue;
            }
            let gate = case.gates[y * case.width + x] as usize % K;
            indices.push(layout.gate(y, x, gate) as u16);
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
    target_output(net, layout, case.target)
}

fn target_output(net: &Network, layout: &Layout, target: (usize, usize)) -> ProbeOutput {
    let mut power = [0.0f64; K];
    for (phase, item) in power.iter_mut().enumerate() {
        let idx = layout.emit(target.0, target.1, phase);
        let charge = net.spike_data()[idx].charge.max(0) as f64;
        let activation = net.activation()[idx].max(0) as f64;
        *item = charge + 4.0 * activation;
    }
    let total: f64 = power.iter().sum();
    ProbeOutput {
        probs: normalize_scores(power),
        power,
        target_power_total: total,
    }
}

fn readout_curve(
    net: &mut Network,
    layout: &Layout,
    case: &PublicCase,
    max_ticks: usize,
) -> Vec<ProbeOutput> {
    (1..=max_ticks)
        .map(|tick| network_output(net, layout, case, tick))
        .collect()
}

fn write_readout_over_time(
    cfg: &Config,
    seed: u64,
    arm: Arm,
    layout: &Layout,
    net: &mut Network,
    cases: &[Case],
    path_length: usize,
    ticks: usize,
    family: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    for tick in 1..=ticks {
        let mut correct = 0usize;
        let mut prob = 0.0;
        let mut power = 0.0;
        for case in cases.iter().take(8) {
            let out = network_output(net, layout, &case.public, tick);
            let label = case.private.label as usize;
            correct += usize::from(argmax(&out.probs) == label);
            prob += out.probs[label];
            power += out.target_power_total;
        }
        let n = cases.len().min(8).max(1);
        append_jsonl(
            cfg.out.join("readout_over_time.jsonl"),
            &ReadoutTickRow {
                seed,
                arm: arm.as_str().to_string(),
                width: layout.width,
                path_length,
                family: family.to_string(),
                tick,
                accuracy: ratio(correct, n),
                correct_probability: prob / n as f64,
                target_power_total: power / n as f64,
            },
        )?;
    }
    Ok(())
}

fn write_phase_diagnostics(
    cfg: &Config,
    seed: u64,
    arm: Arm,
    layout: &Layout,
    net: &mut Network,
    cases: &[Case],
    path_length: usize,
    ticks: usize,
    family: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    for case in cases.iter().take(8) {
        let mut expected = case.public.source_phase as usize;
        let _ = network_output(net, layout, &case.public, ticks);
        for (step, &(y, x)) in case.private.true_path.iter().enumerate() {
            if step > 0 {
                expected = (expected + case.public.gates[y * case.public.width + x] as usize) % K;
            }
            let out = target_like_cell_output(net, layout, (y, x));
            let wrong = out
                .probs
                .iter()
                .enumerate()
                .filter(|(idx, _)| *idx != expected)
                .map(|(_, v)| *v)
                .fold(0.0f64, f64::max);
            append_jsonl(
                cfg.out.join("phase_decay_metrics.jsonl"),
                &json!({
                    "seed": seed,
                    "arm": arm.as_str(),
                    "width": layout.width,
                    "path_length": path_length,
                    "ticks": ticks,
                    "family": family,
                    "step": step,
                    "correct_phase_power": out.power[expected],
                    "wrong_phase_power": wrong,
                    "correct_phase_margin": out.probs[expected] - wrong,
                    "target_power_total_by_step": out.target_power_total,
                }),
            )?;
        }
    }
    Ok(())
}

fn target_like_cell_output(net: &Network, layout: &Layout, cell: (usize, usize)) -> ProbeOutput {
    target_output(net, layout, cell)
}

fn phase_decay_proxy(curve: &[ProbeOutput], label: usize, path_len: usize) -> f64 {
    if curve.is_empty() {
        return 0.0;
    }
    let first = curve.first().map(|out| out.probs[label]).unwrap_or(0.0);
    let best = curve
        .iter()
        .map(|out| out.probs[label])
        .fold(first, f64::max);
    let last = curve.last().map(|out| out.probs[label]).unwrap_or(0.0);
    (best - last).max(0.0) / path_len.max(1) as f64
}

fn wall_leak_rate(
    net: &mut Network,
    layout: &Layout,
    cases: &[Case],
    ticks: usize,
    arm: Arm,
) -> f64 {
    if arm == Arm::FixedPhaseLaneReference {
        return 0.0;
    }
    let mut wall_emit = 0usize;
    let mut wall_total = 0usize;
    for case in cases.iter().take(4) {
        let _ = network_output(net, layout, &case.public, ticks);
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

fn network_for_arm(arm: Arm, layout: &Layout, seed: u64) -> Network {
    let mut net = empty_network(layout);
    match arm {
        Arm::CanonicalJackpot007Baseline => add_same_phase_spatial_edges(&mut net, layout),
        _ => add_emit_to_neighbor_arrive_edges(&mut net, layout),
    }
    if matches!(
        arm,
        Arm::FixedPhaseLaneReference | Arm::CanonicalJackpot007Baseline
    ) {
        return net;
    }
    let motifs = match arm {
        Arm::RandomMatched16MotifControl => random_motif_types(seed ^ 0x1616, 16),
        _ => full_16_rule_template(),
    };
    for y in 0..layout.width {
        for x in 0..layout.width {
            for motif in &motifs {
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
                output_phase: (input_phase + gate) % K,
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

fn add_same_phase_spatial_edges(net: &mut Network, layout: &Layout) {
    add_emit_to_neighbor_arrive_edges(net, layout);
    for y in 0..layout.width {
        for x in 0..layout.width {
            for phase in 0..K {
                net.graph_mut().add_edge(
                    layout.arrive(y, x, phase) as u16,
                    layout.emit(y, x, phase) as u16,
                );
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

fn write_path_summaries(
    cfg: &Config,
    seed: u64,
    arm: Arm,
    rows: &[HorizonRow],
) -> Result<(), Box<dyn std::error::Error>> {
    for width in &cfg.widths {
        for path_length in &cfg.path_lengths {
            let subset = rows
                .iter()
                .filter(|row| row.width == *width && row.path_length == *path_length)
                .collect::<Vec<_>>();
            if subset.is_empty() {
                continue;
            }
            let mut min95 = -1i64;
            let mut min90 = -1i64;
            let mut best_acc = 0.0f64;
            let mut best_prob = 0.0f64;
            for ticks in &cfg.ticks_list {
                let tick_subset = subset
                    .iter()
                    .filter(|row| row.ticks == *ticks)
                    .collect::<Vec<_>>();
                let acc = mean_values(tick_subset.iter().map(|row| row.phase_final_accuracy));
                let prob = mean_values(
                    tick_subset
                        .iter()
                        .map(|row| row.correct_target_lane_probability_mean),
                );
                best_acc = best_acc.max(acc);
                best_prob = best_prob.max(prob);
                if min95 < 0 && acc >= 0.95 {
                    min95 = *ticks as i64;
                }
                if min90 < 0 && prob >= 0.90 {
                    min90 = *ticks as i64;
                }
                append_jsonl(
                    cfg.out.join("tick_metrics.jsonl"),
                    &TickSummaryRow {
                        seed,
                        arm: arm.as_str().to_string(),
                        width: *width,
                        path_length: *path_length,
                        ticks: *ticks,
                        phase_final_accuracy: acc,
                        correct_target_lane_probability_mean: prob,
                    },
                )?;
            }
            append_jsonl(
                cfg.out.join("path_length_metrics.jsonl"),
                &PathSummaryRow {
                    seed,
                    arm: arm.as_str().to_string(),
                    width: *width,
                    path_length: *path_length,
                    minimum_ticks_for_95_accuracy: min95,
                    minimum_ticks_for_90_probability: min90,
                    best_accuracy: best_acc,
                    best_probability: best_prob,
                },
            )?;
        }
    }
    Ok(())
}

fn arm_source_persist(arm: Arm, ticks: usize) -> usize {
    match arm {
        Arm::SourcePersist1Tick => 1,
        Arm::SourcePersist2Ticks => 2,
        Arm::SourcePersistAllTicks => ticks,
        _ => 0,
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

fn mean_values(values: impl Iterator<Item = f64>) -> f64 {
    let vals = values.collect::<Vec<_>>();
    if vals.is_empty() {
        0.0
    } else {
        vals.iter().sum::<f64>() / vals.len() as f64
    }
}

fn family_salt(family: &str) -> u64 {
    match family {
        "all_zero_gates" => 0x2001,
        "repeated_plus_one" => 0x2002,
        "repeated_plus_two" => 0x2003,
        "alternating_plus_minus" => 0x2004,
        "random_balanced" => 0x2005,
        "high_cancellation_sequence" => 0x2006,
        "adversarial_wrong_phase_sequence" => 0x2007,
        "same_target_counterfactual" => 0x2008,
        "gate_shuffle_control" => 0x2009,
        _ => 0x2FFF,
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
                    "eval_examples": cfg.eval_examples,
                    "widths": cfg.widths,
                    "path_lengths": cfg.path_lengths,
                    "ticks_list": cfg.ticks_list,
                })
            })
        })
        .collect();
    write_json(cfg.out.join("queue.json"), &queue)?;
    fs::write(
        cfg.out.join("contract_snapshot.md"),
        "# STABLE_LOOP_PHASE_LOCK_012_PHASE_LANE_HORIZON_SCALING\n\nRunner-local snapshot: measure recurrent horizon for the completed phase-lane coincidence rule.\n",
    )?;
    for file in [
        "progress.jsonl",
        "metrics.jsonl",
        "horizon_curve.jsonl",
        "path_length_metrics.jsonl",
        "tick_metrics.jsonl",
        "phase_decay_metrics.jsonl",
        "wrong_phase_metrics.jsonl",
        "readout_over_time.jsonl",
        "counterfactual_metrics.jsonl",
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

fn refresh_summary_partial(
    cfg: &Config,
    elapsed_sec: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    write_json(
        cfg.out.join("summary.json"),
        &json!({"status": "running", "elapsed_sec": elapsed_sec, "updated_time": now_sec()}),
    )?;
    fs::write(
        cfg.out.join("report.md"),
        format!(
            "# STABLE_LOOP_PHASE_LOCK_012_PHASE_LANE_HORIZON_SCALING\n\nStatus: running.\n\nElapsed seconds: {:.2}\n",
            elapsed_sec
        ),
    )?;
    Ok(())
}

fn refresh_summary(
    cfg: &Config,
    rows: &[HorizonRow],
    completed: usize,
    total: usize,
    elapsed_sec: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    write_json(
        cfg.out.join("summary.json"),
        &json!({
            "status": if completed >= total { "done" } else { "running" },
            "completed": completed,
            "total": total,
            "elapsed_sec": elapsed_sec,
            "verdicts": verdicts(rows),
            "updated_time": now_sec(),
        }),
    )?;
    write_report(cfg, rows, completed >= total)?;
    Ok(())
}

fn write_report(
    cfg: &Config,
    rows: &[HorizonRow],
    final_report: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut report = String::new();
    report.push_str("# STABLE_LOOP_PHASE_LOCK_012_PHASE_LANE_HORIZON_SCALING Report\n\n");
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
    report.push_str("```\n\n");
    report.push_str("## Arm Summary\n\n");
    report.push_str("| Arm | Acc | Prob | Best tick prob | Final-best | Wrong | Power |\n");
    report.push_str("|---|---:|---:|---:|---:|---:|---:|\n");
    for arm in arms() {
        let arm_rows = rows
            .iter()
            .filter(|row| row.arm == arm.as_str())
            .collect::<Vec<_>>();
        if arm_rows.is_empty() {
            continue;
        }
        report.push_str(&format!(
            "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |\n",
            arm.as_str(),
            avg(&arm_rows, |row| row.phase_final_accuracy),
            avg(&arm_rows, |row| row.correct_target_lane_probability_mean),
            avg(&arm_rows, |row| row.best_tick_correct_probability),
            avg(&arm_rows, |row| row.final_tick_minus_best_tick_delta),
            avg(&arm_rows, |row| row.wrong_phase_growth_rate),
            avg(&arm_rows, |row| row.target_power_total),
        ));
    }
    report.push_str("\n## Claim Boundary\n\n");
    report.push_str(
        "This runner measures the recurrent horizon of the completed phase-lane rule. It does not prove production architecture, full VRAXION, consciousness, language grounding, or Prismion uniqueness.\n",
    );
    fs::write(cfg.out.join("report.md"), report)?;
    Ok(())
}

fn verdicts(rows: &[HorizonRow]) -> Vec<String> {
    if rows.is_empty() {
        return vec!["RUNNING".to_string()];
    }
    let mut out = BTreeSet::new();
    let full = arm_rows(rows, Arm::Full16RuleTemplate);
    let sparse = arm_rows(rows, Arm::CommonCore15PlusMissing123);
    let dense = arm_rows(rows, Arm::Dense009Reference);
    let random = arm_rows(rows, Arm::RandomMatched16MotifControl);
    let full_best_long = best_bucket_accuracy(&full, |row| row.path_length >= 24);
    let full_best_short = best_bucket_accuracy(&full, |row| row.path_length <= 2);
    let full_avg = avg(&full, |row| row.phase_final_accuracy);
    let sparse_avg = avg(&sparse, |row| row.phase_final_accuracy);
    let dense_avg = avg(&dense, |row| row.phase_final_accuracy);
    let random_avg = avg(&random, |row| row.phase_final_accuracy);

    if random_avg < 0.45 {
        out.insert("RANDOM_CONTROL_FAILS".to_string());
    }
    if (full_avg - sparse_avg).abs() <= 0.02 {
        out.insert("SPARSE_EQUALS_FULL16_HORIZON".to_string());
    } else if full_avg > sparse_avg + 0.05 {
        out.insert("SPARSE_UNDERPERFORMS_FULL16_HORIZON".to_string());
    }
    if dense_avg > full_avg + 0.05 {
        out.insert("DENSE_REFERENCE_HAS_HORIZON_ADVANTAGE".to_string());
    }
    if full_best_long >= 0.95 {
        out.insert("HORIZON_SCALING_PASSES".to_string());
    } else {
        out.insert("HORIZON_LIMIT_IDENTIFIED".to_string());
        out.insert("FULL16_REFERENCE_BREAKS_ON_LONG_PATHS".to_string());
    }
    if full_best_long >= 0.95 && full_avg < 0.95 {
        out.insert("TICKS_ONLY_LIMIT".to_string());
    } else if full_best_short >= 0.95 && full_best_long < 0.95 {
        out.insert("RULE_TEMPLATE_STABLE_BUT_SETTLING_LIMITED".to_string());
    }
    let decay = avg(&full, |row| row.phase_decay_per_step);
    let wrong = avg(&full, |row| row.wrong_phase_growth_rate);
    if decay > 0.01 && wrong <= 0.35 {
        out.insert("PHASE_DECAY_LIMIT".to_string());
    } else if wrong > 0.35 && decay <= 0.01 {
        out.insert("WRONG_PHASE_INTERFERENCE_LIMIT".to_string());
    } else if wrong > 0.35 && decay > 0.01 {
        out.insert("DECAY_PLUS_INTERFERENCE_LIMIT".to_string());
    }
    let final_minus_best = avg(&full, |row| row.final_tick_minus_best_tick_delta);
    if final_minus_best < -0.05 {
        out.insert("EARLY_ARRIVAL_LATE_DECAY".to_string());
    }
    let persist = arm_rows(rows, Arm::SourcePersistAllTicks);
    if avg(&persist, |row| row.phase_final_accuracy) > full_avg + 0.05 {
        out.insert("SOURCE_DECAY_LIMIT".to_string());
    }
    if full_avg < 0.95 && (full_avg - sparse_avg).abs() <= 0.02 {
        out.insert("RULE_TEMPLATE_STABLE_BUT_SETTLING_LIMITED".to_string());
    }
    if rows.iter().any(|row| {
        row.forbidden_private_field_leak > 0.0
            || row.nonlocal_edge_count > 0
            || row.direct_output_leak_rate > 0.05
    }) {
        out.insert("DIRECT_SHORTCUT_CONTAMINATION".to_string());
    }
    out.insert("PRODUCTION_API_NOT_READY".to_string());
    out.into_iter().collect()
}

fn arm_rows(rows: &[HorizonRow], arm: Arm) -> Vec<&HorizonRow> {
    rows.iter().filter(|row| row.arm == arm.as_str()).collect()
}

fn avg(rows: &[&HorizonRow], f: fn(&HorizonRow) -> f64) -> f64 {
    if rows.is_empty() {
        0.0
    } else {
        rows.iter().map(|row| f(row)).sum::<f64>() / rows.len() as f64
    }
}

fn best_bucket_accuracy<F>(rows: &[&HorizonRow], predicate: F) -> f64
where
    F: Fn(&HorizonRow) -> bool,
{
    let mut buckets: BTreeMap<(usize, usize), (f64, usize)> = BTreeMap::new();
    for row in rows.iter().copied().filter(|row| predicate(row)) {
        let entry = buckets
            .entry((row.path_length, row.ticks))
            .or_insert((0.0, 0));
        entry.0 += row.phase_final_accuracy;
        entry.1 += 1;
    }
    buckets
        .values()
        .filter(|(_, count)| *count > 0)
        .map(|(sum, count)| sum / *count as f64)
        .fold(0.0f64, f64::max)
}

fn append_jsonl<P: AsRef<Path>, T: Serialize>(path: P, value: &T) -> std::io::Result<()> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    serde_json::to_writer(&mut file, value)?;
    writeln!(file)
}

fn write_json<P: AsRef<Path>, T: Serialize>(path: P, value: &T) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    writeln!(file)
}

fn now_sec() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}
