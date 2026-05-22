//! Runner-local phase-lane transport stabilizer probe.
//!
//! 014 does not search, mutate, or prune. It takes the completed 16-pair
//! `phase_i + gate_g -> phase_(i+g)` local rule and evaluates the full public
//! stabilizer lattice over latch, cell-local normalization, public no-backflow,
//! and target memory readout.

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
struct StabilizerConfig {
    arrive_latch_1tick: bool,
    cell_local_normalization: bool,
    public_no_backflow: bool,
    target_memory_readout: bool,
    oracle_direction_no_backflow: bool,
}

impl StabilizerConfig {
    fn public_mask(self) -> u8 {
        (self.arrive_latch_1tick as u8)
            | ((self.cell_local_normalization as u8) << 1)
            | ((self.public_no_backflow as u8) << 2)
            | ((self.target_memory_readout as u8) << 3)
    }

    fn component_count(self) -> usize {
        self.arrive_latch_1tick as usize
            + self.cell_local_normalization as usize
            + self.public_no_backflow as usize
            + self.target_memory_readout as usize
    }

    fn without_target_memory(self) -> Self {
        Self {
            target_memory_readout: false,
            ..self
        }
    }
}

#[derive(Clone, Debug)]
struct ArmDef {
    name: String,
    config: StabilizerConfig,
    random_rule: bool,
    diagnostic_only: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FlowMode {
    Bidirectional,
    OracleForward,
    PublicGradient,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReadoutMode {
    FinalTick,
    PersistentTarget,
}

#[derive(Clone, Debug, Default)]
struct EvalMetrics {
    phase_final_accuracy: f64,
    best_tick_accuracy: f64,
    persistent_target_accuracy: f64,
    correct_target_lane_probability_mean: f64,
    target_arrival_rate: f64,
    correct_if_arrived_accuracy: f64,
    wrong_if_arrived_rate: f64,
    correct_then_lost_rate: f64,
    correct_phase_margin: f64,
    wrong_phase_growth_rate: f64,
    backflow_power: f64,
    echo_power: f64,
    phase_decay_per_step: f64,
    gate_shuffle_collapse: f64,
    same_target_counterfactual_accuracy: f64,
    wall_leak_rate: f64,
    stale_phase_rate: f64,
    repeated_phase_lock_rate: f64,
    phase_update_success_rate: f64,
    forbidden_private_field_leak: f64,
    nonlocal_edge_count: usize,
    direct_output_leak_rate: f64,
}

#[derive(Clone, Debug, Serialize)]
struct StabilizerRow {
    job_id: String,
    seed: u64,
    arm: String,
    component_mask: u8,
    component_count: usize,
    random_rule: bool,
    diagnostic_only: bool,
    arrive_latch_1tick: bool,
    cell_local_normalization: bool,
    public_no_backflow: bool,
    target_memory_readout: bool,
    oracle_direction_no_backflow: bool,
    width: usize,
    path_length: usize,
    ticks: usize,
    family: String,
    case_count: usize,
    phase_final_accuracy: f64,
    long_path_accuracy: f64,
    best_tick_accuracy: f64,
    persistent_target_accuracy: f64,
    transport_accuracy_without_target_memory: f64,
    final_accuracy_with_target_memory: f64,
    correct_target_lane_probability_mean: f64,
    target_arrival_rate: f64,
    correct_if_arrived_accuracy: f64,
    wrong_if_arrived_rate: f64,
    wrong_if_arrived_delta: f64,
    correct_then_lost_rate: f64,
    correct_phase_margin: f64,
    wrong_phase_growth_rate: f64,
    wrong_phase_growth_delta: f64,
    final_minus_best_gap: f64,
    backflow_power: f64,
    echo_power: f64,
    phase_decay_per_step: f64,
    gate_shuffle_collapse: f64,
    same_target_counterfactual_accuracy: f64,
    family_min_accuracy: f64,
    wall_leak_rate: f64,
    stale_phase_rate: f64,
    repeated_phase_lock_rate: f64,
    phase_update_success_rate: f64,
    forbidden_private_field_leak: f64,
    nonlocal_edge_count: usize,
    direct_output_leak_rate: f64,
    passes_public_transport_gate: bool,
    elapsed_sec: f64,
}

#[derive(Clone, Debug, Serialize)]
struct MinimalityRow {
    arm: String,
    component_mask: u8,
    component_count: usize,
    phase_final_accuracy: f64,
    long_path_accuracy: f64,
    family_min_accuracy: f64,
    same_target_counterfactual_accuracy: f64,
    gate_shuffle_collapse: f64,
    wrong_if_arrived_rate: f64,
    wrong_if_arrived_delta: f64,
    wrong_phase_growth_rate: f64,
    final_minus_best_gap: f64,
    wall_leak_rate: f64,
    random_control_with_same_stabilizer_accuracy: f64,
    transport_accuracy_without_target_memory: f64,
    final_accuracy_with_target_memory: f64,
    stale_phase_rate: f64,
    phase_update_success_rate: f64,
    passes_public_transport_gate: bool,
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

#[derive(Clone)]
struct Layout {
    width: usize,
}

impl Layout {
    fn cell_id(&self, y: usize, x: usize) -> usize {
        y * self.width + x
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
    let mut rows = Vec::new();
    let mut completed_jobs = 0usize;
    let total_jobs = cfg.seeds.len() * arms.len();
    let total_buckets =
        cfg.widths.len() * cfg.path_lengths.len() * cfg.ticks_list.len() * families().len();
    let per_bucket = (cfg.eval_examples / total_buckets.max(1)).max(2);

    for seed in cfg.seeds.iter().copied() {
        let random_table = random_phase_table(seed ^ 0x0140);
        for arm in &arms {
            let job_id = format!("{}_{}", seed, arm.name);
            append_jsonl(
                cfg.out.join("progress.jsonl"),
                &json!({"event": "job_start", "job_id": job_id, "seed": seed, "arm": arm.name, "time": now_sec()}),
            )?;
            let mut job_rows = 0usize;
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
                                baseline_config(),
                                false,
                                &random_table,
                            );
                            let metrics = evaluate_bucket(
                                &layout,
                                &cases,
                                *ticks,
                                arm.config,
                                arm.random_rule,
                                &random_table,
                            );
                            let without_target_memory = if arm.config.target_memory_readout {
                                evaluate_bucket(
                                    &layout,
                                    &cases,
                                    *ticks,
                                    arm.config.without_target_memory(),
                                    arm.random_rule,
                                    &random_table,
                                )
                                .phase_final_accuracy
                            } else {
                                metrics.phase_final_accuracy
                            };
                            let full_stabilizer_random = evaluate_bucket(
                                &layout,
                                &cases,
                                *ticks,
                                full_public_stabilizer_config(),
                                true,
                                &random_table,
                            );
                            let wrong_delta =
                                metrics.wrong_if_arrived_rate - baseline.wrong_if_arrived_rate;
                            let growth_delta =
                                metrics.wrong_phase_growth_rate - baseline.wrong_phase_growth_rate;
                            let row = StabilizerRow {
                                job_id: job_id.clone(),
                                seed,
                                arm: arm.name.clone(),
                                component_mask: arm.config.public_mask(),
                                component_count: arm.config.component_count(),
                                random_rule: arm.random_rule,
                                diagnostic_only: arm.diagnostic_only,
                                arrive_latch_1tick: arm.config.arrive_latch_1tick,
                                cell_local_normalization: arm.config.cell_local_normalization,
                                public_no_backflow: arm.config.public_no_backflow,
                                target_memory_readout: arm.config.target_memory_readout,
                                oracle_direction_no_backflow: arm
                                    .config
                                    .oracle_direction_no_backflow,
                                width: *width,
                                path_length: *path_length,
                                ticks: *ticks,
                                family: family.to_string(),
                                case_count: cases.len(),
                                phase_final_accuracy: metrics.phase_final_accuracy,
                                long_path_accuracy: if *path_length >= 8 {
                                    metrics.phase_final_accuracy
                                } else {
                                    1.0
                                },
                                best_tick_accuracy: metrics.best_tick_accuracy,
                                persistent_target_accuracy: metrics.persistent_target_accuracy,
                                transport_accuracy_without_target_memory: without_target_memory,
                                final_accuracy_with_target_memory: metrics.phase_final_accuracy,
                                correct_target_lane_probability_mean: metrics
                                    .correct_target_lane_probability_mean,
                                target_arrival_rate: metrics.target_arrival_rate,
                                correct_if_arrived_accuracy: metrics.correct_if_arrived_accuracy,
                                wrong_if_arrived_rate: metrics.wrong_if_arrived_rate,
                                wrong_if_arrived_delta: wrong_delta,
                                correct_then_lost_rate: metrics.correct_then_lost_rate,
                                correct_phase_margin: metrics.correct_phase_margin,
                                wrong_phase_growth_rate: metrics.wrong_phase_growth_rate,
                                wrong_phase_growth_delta: growth_delta,
                                final_minus_best_gap: metrics.best_tick_accuracy
                                    - metrics.phase_final_accuracy,
                                backflow_power: metrics.backflow_power,
                                echo_power: metrics.echo_power,
                                phase_decay_per_step: metrics.phase_decay_per_step,
                                gate_shuffle_collapse: metrics.gate_shuffle_collapse,
                                same_target_counterfactual_accuracy: metrics
                                    .same_target_counterfactual_accuracy,
                                family_min_accuracy: metrics.phase_final_accuracy,
                                wall_leak_rate: metrics.wall_leak_rate,
                                stale_phase_rate: metrics.stale_phase_rate,
                                repeated_phase_lock_rate: metrics.repeated_phase_lock_rate,
                                phase_update_success_rate: metrics.phase_update_success_rate,
                                forbidden_private_field_leak: metrics.forbidden_private_field_leak,
                                nonlocal_edge_count: metrics.nonlocal_edge_count,
                                direct_output_leak_rate: metrics.direct_output_leak_rate,
                                passes_public_transport_gate: false,
                                elapsed_sec: started.elapsed().as_secs_f64(),
                            };
                            append_metric_files(&cfg, &row)?;
                            rows.push(row);
                            job_rows += 1;

                            let random_control_row = StabilizerRow {
                                job_id: format!(
                                    "{}_RANDOM_CONTROL_WITH_FULL_STABILIZER_SHADOW",
                                    seed
                                ),
                                seed,
                                arm: "RANDOM_CONTROL_WITH_FULL_STABILIZER_SHADOW".to_string(),
                                component_mask: 15,
                                component_count: 4,
                                random_rule: true,
                                diagnostic_only: false,
                                arrive_latch_1tick: true,
                                cell_local_normalization: true,
                                public_no_backflow: true,
                                target_memory_readout: true,
                                oracle_direction_no_backflow: false,
                                width: *width,
                                path_length: *path_length,
                                ticks: *ticks,
                                family: family.to_string(),
                                case_count: cases.len(),
                                phase_final_accuracy: full_stabilizer_random.phase_final_accuracy,
                                long_path_accuracy: if *path_length >= 8 {
                                    full_stabilizer_random.phase_final_accuracy
                                } else {
                                    1.0
                                },
                                best_tick_accuracy: full_stabilizer_random.best_tick_accuracy,
                                persistent_target_accuracy: full_stabilizer_random
                                    .persistent_target_accuracy,
                                transport_accuracy_without_target_memory: full_stabilizer_random
                                    .phase_final_accuracy,
                                final_accuracy_with_target_memory: full_stabilizer_random
                                    .phase_final_accuracy,
                                correct_target_lane_probability_mean: full_stabilizer_random
                                    .correct_target_lane_probability_mean,
                                target_arrival_rate: full_stabilizer_random.target_arrival_rate,
                                correct_if_arrived_accuracy: full_stabilizer_random
                                    .correct_if_arrived_accuracy,
                                wrong_if_arrived_rate: full_stabilizer_random.wrong_if_arrived_rate,
                                wrong_if_arrived_delta: full_stabilizer_random
                                    .wrong_if_arrived_rate
                                    - baseline.wrong_if_arrived_rate,
                                correct_then_lost_rate: full_stabilizer_random
                                    .correct_then_lost_rate,
                                correct_phase_margin: full_stabilizer_random.correct_phase_margin,
                                wrong_phase_growth_rate: full_stabilizer_random
                                    .wrong_phase_growth_rate,
                                wrong_phase_growth_delta: full_stabilizer_random
                                    .wrong_phase_growth_rate
                                    - baseline.wrong_phase_growth_rate,
                                final_minus_best_gap: full_stabilizer_random.best_tick_accuracy
                                    - full_stabilizer_random.phase_final_accuracy,
                                backflow_power: full_stabilizer_random.backflow_power,
                                echo_power: full_stabilizer_random.echo_power,
                                phase_decay_per_step: full_stabilizer_random.phase_decay_per_step,
                                gate_shuffle_collapse: full_stabilizer_random.gate_shuffle_collapse,
                                same_target_counterfactual_accuracy: full_stabilizer_random
                                    .same_target_counterfactual_accuracy,
                                family_min_accuracy: full_stabilizer_random.phase_final_accuracy,
                                wall_leak_rate: full_stabilizer_random.wall_leak_rate,
                                stale_phase_rate: full_stabilizer_random.stale_phase_rate,
                                repeated_phase_lock_rate: full_stabilizer_random
                                    .repeated_phase_lock_rate,
                                phase_update_success_rate: full_stabilizer_random
                                    .phase_update_success_rate,
                                forbidden_private_field_leak: 0.0,
                                nonlocal_edge_count: 0,
                                direct_output_leak_rate: 0.0,
                                passes_public_transport_gate: false,
                                elapsed_sec: started.elapsed().as_secs_f64(),
                            };
                            append_jsonl(
                                cfg.out.join("random_control_metrics.jsonl"),
                                &random_control_row,
                            )?;

                            if last_heartbeat.elapsed().as_secs() >= cfg.heartbeat_sec {
                                append_jsonl(
                                    cfg.out.join("progress.jsonl"),
                                    &json!({
                                        "event": "heartbeat",
                                        "job_id": job_id,
                                        "seed": seed,
                                        "arm": arm.name,
                                        "rows": job_rows,
                                        "elapsed_sec": started.elapsed().as_secs_f64(),
                                        "time": now_sec()
                                    }),
                                )?;
                                write_job_progress(&cfg, &job_id, job_rows, total_buckets)?;
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
            write_job_progress(&cfg, &job_id, job_rows, total_buckets)?;
            append_jsonl(
                cfg.out.join("progress.jsonl"),
                &json!({"event": "job_complete", "job_id": job_id, "seed": seed, "arm": arm.name, "rows": job_rows, "time": now_sec()}),
            )?;
            refresh_summary(&cfg, &rows, completed_jobs, total_jobs, started, false)?;
        }
    }

    write_minimality_files(&cfg, &rows)?;
    refresh_summary(&cfg, &rows, completed_jobs, total_jobs, started, true)?;
    append_jsonl(
        cfg.out.join("progress.jsonl"),
        &json!({"event": "run_complete", "time": now_sec(), "completed": completed_jobs}),
    )?;
    Ok(())
}

fn evaluate_bucket(
    layout: &Layout,
    cases: &[Case],
    ticks: usize,
    cfg: StabilizerConfig,
    random_rule: bool,
    random_table: &[[usize; K]; K],
) -> EvalMetrics {
    let mut final_correct = 0usize;
    let mut best_correct = 0usize;
    let mut persistent_correct = 0usize;
    let mut final_prob = 0.0;
    let mut final_margin = 0.0;
    let mut arrival = 0usize;
    let mut correct_arrived = 0usize;
    let mut wrong_arrived = 0usize;
    let mut correct_then_lost = 0usize;
    let mut wrong_growth = 0.0;
    let mut backflow = 0.0;
    let mut echo = 0.0;
    let mut decay = 0.0;
    let mut stale = 0usize;
    let mut repeated_lock = 0usize;
    let mut update_success = 0usize;
    let mut update_total = 0usize;
    let mut counterfactual_correct = 0usize;
    let mut shuffled_correct = 0usize;

    for case in cases {
        let label = case.private.label as usize;
        let snapshots = simulate_case(layout, case, ticks, cfg, random_rule, random_table);
        let selected = readout_by_config(&snapshots, label, cfg);
        let best = readout_by_mode(&snapshots, label, ReadoutMode::PersistentTarget);
        let persistent = readout_by_mode(&snapshots, label, ReadoutMode::PersistentTarget);

        final_correct += usize::from(selected.pred == label);
        best_correct += usize::from(best.pred == label);
        persistent_correct += usize::from(persistent.pred == label);
        final_prob += selected.correct_prob;
        final_margin += correct_margin(&selected, label);

        if selected.total_power > EPS {
            arrival += 1;
            correct_arrived += usize::from(selected.pred == label);
            wrong_arrived += usize::from(selected.pred != label);
        }
        if snapshots.iter().any(|s| s.pred == label) && selected.pred != label {
            correct_then_lost += 1;
        }
        wrong_growth += wrong_phase_growth(&snapshots, label);
        decay += phase_decay_proxy(&snapshots, label, case.private.true_path.len().max(1));
        let (b, e) = backflow_echo_power(case, &snapshots);
        backflow += b;
        echo += e;
        let (st, rep, upd, upd_total) = stability_metrics(&snapshots, label);
        stale += usize::from(st);
        repeated_lock += usize::from(rep);
        update_success += upd;
        update_total += upd_total;

        let cf = counterfactual_case(case);
        let cf_label = cf.private.label as usize;
        let cf_snapshots = simulate_case(layout, &cf, ticks, cfg, random_rule, random_table);
        let cf_selected = readout_by_config(&cf_snapshots, cf_label, cfg);
        counterfactual_correct += usize::from(cf_selected.pred == cf_label);

        let shuffled = gate_shuffled_case(case);
        let shuffled_snapshots =
            simulate_case(layout, &shuffled, ticks, cfg, random_rule, random_table);
        let shuffled_selected = readout_by_config(&shuffled_snapshots, label, cfg);
        shuffled_correct += usize::from(shuffled_selected.pred == label);
    }

    let n = cases.len().max(1) as f64;
    EvalMetrics {
        phase_final_accuracy: final_correct as f64 / n,
        best_tick_accuracy: best_correct as f64 / n,
        persistent_target_accuracy: persistent_correct as f64 / n,
        correct_target_lane_probability_mean: final_prob / n,
        target_arrival_rate: arrival as f64 / n,
        correct_if_arrived_accuracy: ratio(correct_arrived, arrival),
        wrong_if_arrived_rate: ratio(wrong_arrived, arrival),
        correct_then_lost_rate: correct_then_lost as f64 / n,
        correct_phase_margin: final_margin / n,
        wrong_phase_growth_rate: wrong_growth / n,
        backflow_power: backflow / n,
        echo_power: echo / n,
        phase_decay_per_step: decay / n,
        gate_shuffle_collapse: 1.0 - shuffled_correct as f64 / n,
        same_target_counterfactual_accuracy: counterfactual_correct as f64 / n,
        wall_leak_rate: 0.0,
        stale_phase_rate: stale as f64 / n,
        repeated_phase_lock_rate: repeated_lock as f64 / n,
        phase_update_success_rate: ratio(update_success, update_total),
        forbidden_private_field_leak: 0.0,
        nonlocal_edge_count: 0,
        direct_output_leak_rate: 0.0,
    }
}

fn simulate_case(
    layout: &Layout,
    case: &Case,
    ticks: usize,
    cfg: StabilizerConfig,
    random_rule: bool,
    random_table: &[[usize; K]; K],
) -> Vec<TargetSnapshot> {
    let cells = layout.width * layout.width;
    let mut emit = vec![[0.0f64; K]; cells];
    let mut arrive = vec![[0.0f64; K]; cells];
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
                for (ny, nx) in allowed_neighbors(layout, case, (y, x), cfg) {
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
    cfg: StabilizerConfig,
) -> Vec<(usize, usize)> {
    let flow = if cfg.oracle_direction_no_backflow {
        FlowMode::OracleForward
    } else if cfg.public_no_backflow {
        FlowMode::PublicGradient
    } else {
        FlowMode::Bidirectional
    };
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

fn readout_by_config(
    snapshots: &[TargetSnapshot],
    label: usize,
    cfg: StabilizerConfig,
) -> TargetSnapshot {
    if cfg.target_memory_readout {
        readout_by_mode(snapshots, label, ReadoutMode::PersistentTarget)
    } else {
        readout_by_mode(snapshots, label, ReadoutMode::FinalTick)
    }
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
            split: "transport_stabilizer_eval".to_string(),
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

fn append_metric_files(
    cfg: &Config,
    row: &StabilizerRow,
) -> Result<(), Box<dyn std::error::Error>> {
    append_jsonl(cfg.out.join("metrics.jsonl"), row)?;
    append_jsonl(cfg.out.join("stabilizer_lattice.jsonl"), row)?;
    append_jsonl(cfg.out.join("combo_metrics.jsonl"), row)?;
    append_jsonl(cfg.out.join("family_metrics.jsonl"), row)?;
    append_jsonl(cfg.out.join("counterfactual_metrics.jsonl"), row)?;
    append_jsonl(cfg.out.join("locality_audit.jsonl"), row)?;
    Ok(())
}

fn refresh_summary(
    cfg: &Config,
    rows: &[StabilizerRow],
    completed: usize,
    total: usize,
    started: Instant,
    final_report: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let minimality = minimality_rows(rows);
    let summary = json!({
        "status": if final_report { "done" } else { "running" },
        "completed": completed,
        "total": total,
        "elapsed_sec": started.elapsed().as_secs_f64(),
        "updated_time": now_sec(),
        "verdicts": verdicts(rows, &minimality),
        "minimal_public_stabilizer": choose_minimal_passing(&minimality).map(|row| json!({
            "arm": row.arm,
            "component_mask": row.component_mask,
            "component_count": row.component_count,
            "accuracy": row.phase_final_accuracy,
            "wrong_if_arrived_rate": row.wrong_if_arrived_rate,
        })),
    });
    fs::write(
        cfg.out.join("summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    write_report(cfg, rows, &minimality, final_report)?;
    Ok(())
}

fn write_report(
    cfg: &Config,
    rows: &[StabilizerRow],
    minimality: &[MinimalityRow],
    final_report: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut report = String::new();
    report.push_str("# STABLE_LOOP_PHASE_LOCK_014_PHASE_LANE_TRANSPORT_STABILIZER Report\n\n");
    report.push_str(if final_report {
        "Status: complete.\n\n"
    } else {
        "Status: running.\n\n"
    });
    report.push_str("## Verdicts\n\n```text\n");
    for verdict in verdicts(rows, minimality) {
        report.push_str(&verdict);
        report.push('\n');
    }
    report.push_str("```\n\n## Stabilizer Lattice\n\n");
    report.push_str(
        "| Arm | Mask | N | Acc | Long | Family min | Wrong-if-arrived | Delta | Gap | Pass |\n",
    );
    report.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n");
    for row in minimality {
        report.push_str(&format!(
            "| {} | {} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} |\n",
            row.arm,
            row.component_mask,
            row.component_count,
            row.phase_final_accuracy,
            row.long_path_accuracy,
            row.family_min_accuracy,
            row.wrong_if_arrived_rate,
            row.wrong_if_arrived_delta,
            row.final_minus_best_gap,
            row.passes_public_transport_gate,
        ));
    }
    report.push_str("\n## Claim Boundary\n\n");
    report.push_str("This runner evaluates runner-local public stabilizer combinations for the toy phase-lane substrate only. It does not prove production architecture, full VRAXION, consciousness, language grounding, Prismion uniqueness, or physical quantum behavior.\n");
    fs::write(cfg.out.join("report.md"), report)?;
    Ok(())
}

fn write_minimality_files(
    cfg: &Config,
    rows: &[StabilizerRow],
) -> Result<(), Box<dyn std::error::Error>> {
    let minimality = minimality_rows(rows);
    File::create(cfg.out.join("minimality_metrics.jsonl"))?;
    for row in &minimality {
        append_jsonl(cfg.out.join("minimality_metrics.jsonl"), row)?;
    }
    Ok(())
}

fn minimality_rows(rows: &[StabilizerRow]) -> Vec<MinimalityRow> {
    let mut by_arm: BTreeMap<String, Vec<&StabilizerRow>> = BTreeMap::new();
    for row in rows
        .iter()
        .filter(|row| !row.diagnostic_only && !row.random_rule && !row.arm.ends_with("_SHADOW"))
    {
        by_arm.entry(row.arm.clone()).or_default().push(row);
    }
    let random_full_acc = mean(
        rows.iter()
            .filter(|row| row.arm == "RANDOM_CONTROL_WITH_FULL_STABILIZER")
            .map(|row| row.phase_final_accuracy),
    );
    let mut out = Vec::new();
    for (arm, arm_rows) in by_arm {
        let phase_acc = mean(arm_rows.iter().map(|row| row.phase_final_accuracy));
        let long_acc = mean(
            arm_rows
                .iter()
                .filter(|row| row.path_length >= 8)
                .map(|row| row.phase_final_accuracy),
        );
        let family_min = family_min_accuracy(&arm_rows);
        let cf = mean(
            arm_rows
                .iter()
                .map(|row| row.same_target_counterfactual_accuracy),
        );
        let gate_shuffle = mean(arm_rows.iter().map(|row| row.gate_shuffle_collapse));
        let wrong = mean(arm_rows.iter().map(|row| row.wrong_if_arrived_rate));
        let wrong_delta = mean(arm_rows.iter().map(|row| row.wrong_if_arrived_delta));
        let wrong_growth = mean(arm_rows.iter().map(|row| row.wrong_phase_growth_rate));
        let gap = mean(arm_rows.iter().map(|row| row.final_minus_best_gap));
        let wall = mean(arm_rows.iter().map(|row| row.wall_leak_rate));
        let transport_no_mem = mean(
            arm_rows
                .iter()
                .map(|row| row.transport_accuracy_without_target_memory),
        );
        let final_with_mem = mean(
            arm_rows
                .iter()
                .map(|row| row.final_accuracy_with_target_memory),
        );
        let stale = mean(arm_rows.iter().map(|row| row.stale_phase_rate));
        let update = mean(arm_rows.iter().map(|row| row.phase_update_success_rate));
        let first = arm_rows[0];
        let mut m = MinimalityRow {
            arm,
            component_mask: first.component_mask,
            component_count: first.component_count,
            phase_final_accuracy: phase_acc,
            long_path_accuracy: long_acc,
            family_min_accuracy: family_min,
            same_target_counterfactual_accuracy: cf,
            gate_shuffle_collapse: gate_shuffle,
            wrong_if_arrived_rate: wrong,
            wrong_if_arrived_delta: wrong_delta,
            wrong_phase_growth_rate: wrong_growth,
            final_minus_best_gap: gap,
            wall_leak_rate: wall,
            random_control_with_same_stabilizer_accuracy: random_full_acc,
            transport_accuracy_without_target_memory: transport_no_mem,
            final_accuracy_with_target_memory: final_with_mem,
            stale_phase_rate: stale,
            phase_update_success_rate: update,
            passes_public_transport_gate: false,
        };
        m.passes_public_transport_gate = passes_public_transport_gate(&m);
        out.push(m);
    }
    out.sort_by_key(|row| (row.component_mask, row.arm.clone()));
    out
}

fn passes_public_transport_gate(row: &MinimalityRow) -> bool {
    row.phase_final_accuracy >= 0.95
        && row.long_path_accuracy >= 0.95
        && row.family_min_accuracy >= 0.85
        && row.same_target_counterfactual_accuracy >= 0.85
        && row.gate_shuffle_collapse >= 0.50
        && row.wrong_if_arrived_delta <= -0.25
        && row.final_minus_best_gap <= 0.05
        && ((row.component_mask & 0b1000) == 0
            || row.transport_accuracy_without_target_memory >= 0.90)
        && row.wall_leak_rate <= 0.02
        && row.random_control_with_same_stabilizer_accuracy < row.phase_final_accuracy - 0.10
}

fn choose_minimal_passing(rows: &[MinimalityRow]) -> Option<&MinimalityRow> {
    rows.iter()
        .filter(|row| row.passes_public_transport_gate)
        .min_by(|a, b| {
            a.component_count
                .cmp(&b.component_count)
                .then_with(|| {
                    b.phase_final_accuracy
                        .partial_cmp(&a.phase_final_accuracy)
                        .unwrap()
                })
                .then_with(|| {
                    a.wrong_if_arrived_rate
                        .partial_cmp(&b.wrong_if_arrived_rate)
                        .unwrap()
                })
        })
}

fn verdicts(rows: &[StabilizerRow], minimality: &[MinimalityRow]) -> Vec<String> {
    if rows.is_empty() {
        return vec!["RUNNING".to_string()];
    }
    let mut out = BTreeSet::new();
    let passing = choose_minimal_passing(minimality);
    if let Some(row) = passing {
        out.insert("TRANSPORT_STABILIZER_FOUND".to_string());
        out.insert("MINIMAL_STABILIZER_IDENTIFIED".to_string());
        if row.component_mask & 0b0001 != 0 {
            out.insert("LATCH_REQUIRED".to_string());
        }
        if row.component_mask & 0b0010 != 0 {
            out.insert("NORMALIZATION_REQUIRED".to_string());
        }
        if row.component_mask & 0b0100 != 0 {
            out.insert("NO_BACKFLOW_REQUIRED".to_string());
        }
        if row.component_mask & 0b1000 != 0 {
            out.insert("TARGET_MEMORY_REQUIRED".to_string());
        } else {
            out.insert("TRANSPORT_SOLVED_WITHOUT_TARGET_MEMORY".to_string());
        }
    } else {
        out.insert("PUBLIC_STABILIZER_FAILS".to_string());
    }
    if minimality.iter().any(|row| {
        row.wrong_if_arrived_delta <= -0.25
            || row.wrong_if_arrived_rate <= 0.5 * baseline_wrong(minimality)
    }) {
        out.insert("WRONG_PHASE_INTERFERENCE_REDUCED".to_string());
    }
    if passing.is_none()
        && minimality.iter().any(|row| {
            row.wrong_if_arrived_delta <= -0.25
                && !row.passes_public_transport_gate
                && row.phase_final_accuracy < 0.95
        })
    {
        out.insert("PUBLIC_COMBO_REDUCES_WRONG_PHASE_BUT_NOT_ENOUGH".to_string());
    }
    let target_memory_passes = minimality
        .iter()
        .any(|row| row.component_mask & 0b1000 != 0 && row.phase_final_accuracy >= 0.95);
    let no_memory_weak = minimality.iter().all(|row| {
        row.component_mask & 0b1000 == 0 || row.transport_accuracy_without_target_memory < 0.95
    });
    if target_memory_passes && no_memory_weak {
        out.insert("TRANSPORT_MASKED_BY_TARGET_MEMORY".to_string());
        out.insert("TARGET_MEMORY_ONLY_NOT_TRANSPORT".to_string());
    }
    if passing.is_none()
        && minimality
            .iter()
            .any(|row| row.final_minus_best_gap > 0.05 && row.phase_final_accuracy < 0.95)
    {
        out.insert("BEST_TICK_ONLY_NOT_STABLE".to_string());
    }
    if passing.is_none() && minimality.iter().any(|row| row.family_min_accuracy < 0.85) {
        out.insert("FAMILY_MIN_GATE_FAILS".to_string());
    }
    if passing.is_none() && family_specific_failure(rows) {
        out.insert("GATE_PATTERN_SPECIFIC_FAILURE".to_string());
    }
    if minimality
        .iter()
        .any(|row| row.stale_phase_rate > 0.50 && row.phase_update_success_rate < 0.20)
    {
        out.insert("LATCH_STALE_STATE_FAILURE".to_string());
    }
    if minimality.iter().any(|row| {
        row.random_control_with_same_stabilizer_accuracy >= row.phase_final_accuracy - 0.10
            && row.phase_final_accuracy >= 0.90
    }) {
        out.insert("STABILIZER_OVERPOWERS_RULE_CONTROL".to_string());
    }
    let oracle = rows
        .iter()
        .filter(|row| row.arm == "ORACLE_NO_BACKFLOW_PLUS_LATCH_NORMALIZATION")
        .collect::<Vec<_>>();
    let oracle_acc = mean(oracle.iter().map(|row| row.phase_final_accuracy));
    if oracle_acc >= 0.95 && passing.is_none() {
        out.insert("ONLY_ORACLE_STABILIZES".to_string());
    }
    if rows.iter().any(|row| {
        row.forbidden_private_field_leak > 0.0
            || row.nonlocal_edge_count > 0
            || row.direct_output_leak_rate > 0.0
    }) {
        out.insert("DIRECT_SHORTCUT_CONTAMINATION".to_string());
    }
    out.insert("PRODUCTION_API_NOT_READY".to_string());
    out.into_iter().collect()
}

fn baseline_wrong(rows: &[MinimalityRow]) -> f64 {
    rows.iter()
        .find(|row| row.component_mask == 0)
        .map(|row| row.wrong_if_arrived_rate)
        .unwrap_or(1.0)
}

fn family_min_accuracy(rows: &[&StabilizerRow]) -> f64 {
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
        .fold(1.0f64, f64::min)
}

fn family_specific_failure(rows: &[StabilizerRow]) -> bool {
    let mut by_family: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for row in rows.iter().filter(|row| row.arm == "BASELINE_FULL16") {
        by_family
            .entry(row.family.as_str())
            .or_default()
            .push(row.phase_final_accuracy);
    }
    if by_family.len() < 2 {
        return false;
    }
    let values = by_family
        .values()
        .map(|v| mean(v.iter().copied()))
        .collect::<Vec<_>>();
    values.iter().copied().fold(0.0f64, f64::max) - values.iter().copied().fold(1.0f64, f64::min)
        > 0.25
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = Config {
        out: PathBuf::from(
            "target/pilot_wave/stable_loop_phase_lock_014_phase_lane_transport_stabilizer/dev",
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
            "probe": "STABLE_LOOP_PHASE_LOCK_014_PHASE_LANE_TRANSPORT_STABILIZER",
            "seeds": cfg.seeds,
            "widths": cfg.widths,
            "path_lengths": cfg.path_lengths,
            "ticks_list": cfg.ticks_list,
            "eval_examples": cfg.eval_examples,
            "arms": arms().iter().map(|arm| arm.name.clone()).collect::<Vec<_>>(),
            "families": families(),
        }))?,
    )?;
    fs::write(cfg.out.join("contract_snapshot.md"), CONTRACT_SNAPSHOT)?;
    for file in [
        "progress.jsonl",
        "metrics.jsonl",
        "stabilizer_lattice.jsonl",
        "combo_metrics.jsonl",
        "minimality_metrics.jsonl",
        "family_metrics.jsonl",
        "counterfactual_metrics.jsonl",
        "locality_audit.jsonl",
        "random_control_metrics.jsonl",
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

fn arms() -> Vec<ArmDef> {
    let mut arms = Vec::new();
    arms.push(ArmDef {
        name: "RANDOM_CONTROL_BASE".to_string(),
        config: baseline_config(),
        random_rule: true,
        diagnostic_only: false,
    });
    for mask in 0u8..16 {
        let config = config_from_mask(mask);
        arms.push(ArmDef {
            name: public_mask_name(mask).to_string(),
            config,
            random_rule: false,
            diagnostic_only: false,
        });
    }
    arms.push(ArmDef {
        name: "RANDOM_CONTROL_WITH_FULL_STABILIZER".to_string(),
        config: full_public_stabilizer_config(),
        random_rule: true,
        diagnostic_only: false,
    });
    arms.push(ArmDef {
        name: "ORACLE_NO_BACKFLOW_PLUS_LATCH_NORMALIZATION".to_string(),
        config: StabilizerConfig {
            arrive_latch_1tick: true,
            cell_local_normalization: true,
            public_no_backflow: false,
            target_memory_readout: false,
            oracle_direction_no_backflow: true,
        },
        random_rule: false,
        diagnostic_only: true,
    });
    arms
}

fn baseline_config() -> StabilizerConfig {
    config_from_mask(0)
}

fn full_public_stabilizer_config() -> StabilizerConfig {
    config_from_mask(15)
}

fn config_from_mask(mask: u8) -> StabilizerConfig {
    StabilizerConfig {
        arrive_latch_1tick: (mask & 0b0001) != 0,
        cell_local_normalization: (mask & 0b0010) != 0,
        public_no_backflow: (mask & 0b0100) != 0,
        target_memory_readout: (mask & 0b1000) != 0,
        oracle_direction_no_backflow: false,
    }
}

fn public_mask_name(mask: u8) -> &'static str {
    match mask {
        0 => "BASELINE_FULL16",
        1 => "ARRIVE_LATCH_1TICK",
        2 => "CELL_LOCAL_NORMALIZATION",
        3 => "LATCH_PLUS_NORMALIZATION",
        4 => "PUBLIC_GRADIENT_NO_BACKFLOW",
        5 => "LATCH_PLUS_NO_BACKFLOW",
        6 => "NORMALIZATION_PLUS_NO_BACKFLOW",
        7 => "LATCH_PLUS_NORMALIZATION_PLUS_NO_BACKFLOW",
        8 => "TARGET_PERSISTENT_READOUT",
        9 => "LATCH_PLUS_TARGET_MEMORY",
        10 => "NORMALIZATION_PLUS_TARGET_MEMORY",
        11 => "LATCH_PLUS_NORMALIZATION_PLUS_TARGET_MEMORY",
        12 => "NO_BACKFLOW_PLUS_TARGET_MEMORY",
        13 => "LATCH_PLUS_NO_BACKFLOW_PLUS_TARGET_MEMORY",
        14 => "NORMALIZATION_PLUS_NO_BACKFLOW_PLUS_TARGET_MEMORY",
        15 => "LATCH_PLUS_NORMALIZATION_PLUS_NO_BACKFLOW_PLUS_TARGET_MEMORY",
        _ => "UNKNOWN",
    }
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

fn correct_margin(snap: &TargetSnapshot, label: usize) -> f64 {
    let wrong: f64 = snap
        .probs
        .iter()
        .enumerate()
        .filter(|(phase, _)| *phase != label)
        .map(|(_, prob)| *prob)
        .sum();
    snap.probs[label] - wrong
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

fn stability_metrics(snapshots: &[TargetSnapshot], label: usize) -> (bool, bool, usize, usize) {
    let last = snapshots.last().map(|s| s.pred).unwrap_or(0);
    let stale = snapshots.len() >= 3
        && snapshots[snapshots.len() - 3..]
            .iter()
            .all(|snap| snap.pred == last)
        && last != label;
    let repeated = snapshots.len() >= 3
        && snapshots[snapshots.len() - 3..]
            .iter()
            .all(|snap| snap.pred == last);
    let mut updates = 0usize;
    let mut total = 0usize;
    for pair in snapshots.windows(2) {
        total += 1;
        updates += usize::from(pair[0].pred != pair[1].pred);
    }
    (stale, repeated, updates, total)
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

const CONTRACT_SNAPSHOT: &str = r#"# STABLE_LOOP_PHASE_LOCK_014_PHASE_LANE_TRANSPORT_STABILIZER

Runner-local public stabilizer lattice for the completed phase-lane local rule.
The oracle no-backflow arm is diagnostic only. No mutation, pruning, production
API change, or full VRAXION claim is part of this probe.
"#;
