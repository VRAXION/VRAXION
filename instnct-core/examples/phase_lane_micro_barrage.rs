//! Runner-local phase-lane micro-barrage.
//!
//! 015 is a mechanism selector, not a transport-solved claim. It reuses the
//! 014 corridor/task family shape and compares small field-transport mechanics
//! against BASELINE_FULL16_014 and BEST_PUBLIC_COMBO_014.

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
enum MicroMechanic {
    BaselineFull16,
    BestPublicCombo014,
    SignedPhaseCancellation,
    DualLayerEbField,
    MomentumLanes,
    EmitOnceConsume,
    RefractoryCell,
    NoReentryMomentum,
    PhaseCompetitionPerCell,
    ArrivalWindowReadoutDiagnostic,
    MomentumPlusConsume,
    SignedPlusCellNormalization,
    DualLayerPlusDamping,
}

impl MicroMechanic {
    fn as_str(self) -> &'static str {
        match self {
            Self::BaselineFull16 => "BASELINE_FULL16_014",
            Self::BestPublicCombo014 => "BEST_PUBLIC_COMBO_014",
            Self::SignedPhaseCancellation => "SIGNED_PHASE_CANCELLATION",
            Self::DualLayerEbField => "DUAL_LAYER_EB_FIELD",
            Self::MomentumLanes => "MOMENTUM_LANES",
            Self::EmitOnceConsume => "EMIT_ONCE_CONSUME",
            Self::RefractoryCell => "REFRACTORY_CELL",
            Self::NoReentryMomentum => "NO_REENTRY_MOMENTUM",
            Self::PhaseCompetitionPerCell => "PHASE_COMPETITION_PER_CELL",
            Self::ArrivalWindowReadoutDiagnostic => "ARRIVAL_WINDOW_READOUT_DIAGNOSTIC",
            Self::MomentumPlusConsume => "MOMENTUM_PLUS_CONSUME",
            Self::SignedPlusCellNormalization => "SIGNED_PLUS_CELL_NORMALIZATION",
            Self::DualLayerPlusDamping => "DUAL_LAYER_PLUS_DAMPING",
        }
    }

    fn diagnostic_only(self) -> bool {
        self == Self::ArrivalWindowReadoutDiagnostic
    }
}

#[derive(Clone, Copy, Debug)]
struct MechanicConfig {
    arrive_latch_1tick: bool,
    cell_local_normalization: bool,
    public_gradient: bool,
    signed_cancellation: bool,
    dual_layer: bool,
    dual_damping: bool,
    momentum_lanes: bool,
    emit_once_consume: bool,
    refractory_ticks: usize,
    no_reentry: bool,
    phase_competition: bool,
}

#[derive(Clone, Debug)]
struct Arm {
    name: String,
    mechanic: MicroMechanic,
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
    net_power: f64,
    cancelled_correct_rate: f64,
    cancelled_wrong_rate: f64,
    final_minus_best_gap: f64,
    gate_shuffle_collapse: f64,
    same_target_counterfactual_accuracy: f64,
    echo_power: f64,
    backflow_power: f64,
    stale_phase_rate: f64,
    reentry_count: f64,
}

#[derive(Clone, Debug, Serialize)]
struct MechanismRow {
    job_id: String,
    seed: u64,
    arm: String,
    mechanic: String,
    random_rule: bool,
    diagnostic_only: bool,
    width: usize,
    path_length: usize,
    ticks: usize,
    family: String,
    case_count: usize,
    phase_final_accuracy: f64,
    long_path_accuracy: f64,
    family_min_accuracy: f64,
    target_arrival_rate: f64,
    wrong_if_arrived_rate: f64,
    wrong_phase_growth_rate: f64,
    correct_phase_power: f64,
    wrong_phase_power: f64,
    net_power: f64,
    cancelled_correct_rate: f64,
    cancelled_wrong_rate: f64,
    final_minus_best_gap: f64,
    gate_shuffle_collapse: f64,
    same_target_counterfactual_accuracy: f64,
    echo_power: f64,
    backflow_power: f64,
    stale_phase_rate: f64,
    reentry_count: f64,
    random_control_accuracy: f64,
    delta_vs_baseline_accuracy: f64,
    delta_vs_baseline_wrong_if_arrived: f64,
    has_micro_signal: bool,
    elapsed_sec: f64,
}

#[derive(Clone, Debug, Serialize)]
struct RankingRow {
    mechanic: String,
    phase_final_accuracy: f64,
    long_path_accuracy: f64,
    family_min_accuracy: f64,
    wrong_if_arrived_rate: f64,
    wrong_phase_growth_rate: f64,
    final_minus_best_gap: f64,
    random_control_accuracy: f64,
    delta_vs_baseline_accuracy: f64,
    delta_vs_baseline_wrong_if_arrived: f64,
    has_micro_signal: bool,
}

#[derive(Clone, Debug)]
struct TargetSnapshot {
    probs: [f64; K],
    pred: usize,
    correct_prob: f64,
    total_power: f64,
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
    let total_jobs = cfg.seeds.len() * arms.len();
    let mut completed_jobs = 0usize;
    let total_buckets =
        cfg.widths.len() * cfg.path_lengths.len() * cfg.ticks_list.len() * families().len();
    let per_bucket = (cfg.eval_examples / total_buckets.max(1)).max(2);

    for seed in cfg.seeds.iter().copied() {
        let random_table = random_phase_table(seed ^ 0x0150);
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
                                MicroMechanic::BaselineFull16,
                                false,
                                &random_table,
                            );
                            let random_control = evaluate_bucket(
                                &layout,
                                &cases,
                                *ticks,
                                arm.mechanic,
                                true,
                                &random_table,
                            );
                            let metrics = evaluate_bucket(
                                &layout,
                                &cases,
                                *ticks,
                                arm.mechanic,
                                arm.random_rule,
                                &random_table,
                            );
                            let delta_acc =
                                metrics.phase_final_accuracy - baseline.phase_final_accuracy;
                            let delta_wrong =
                                baseline.wrong_if_arrived_rate - metrics.wrong_if_arrived_rate;
                            let row = MechanismRow {
                                job_id: job_id.clone(),
                                seed,
                                arm: arm.name.clone(),
                                mechanic: arm.mechanic.as_str().to_string(),
                                random_rule: arm.random_rule,
                                diagnostic_only: arm.mechanic.diagnostic_only(),
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
                                family_min_accuracy: metrics.phase_final_accuracy,
                                target_arrival_rate: metrics.target_arrival_rate,
                                wrong_if_arrived_rate: metrics.wrong_if_arrived_rate,
                                wrong_phase_growth_rate: metrics.wrong_phase_growth_rate,
                                correct_phase_power: metrics.correct_phase_power,
                                wrong_phase_power: metrics.wrong_phase_power,
                                net_power: metrics.net_power,
                                cancelled_correct_rate: metrics.cancelled_correct_rate,
                                cancelled_wrong_rate: metrics.cancelled_wrong_rate,
                                final_minus_best_gap: metrics.final_minus_best_gap,
                                gate_shuffle_collapse: metrics.gate_shuffle_collapse,
                                same_target_counterfactual_accuracy: metrics
                                    .same_target_counterfactual_accuracy,
                                echo_power: metrics.echo_power,
                                backflow_power: metrics.backflow_power,
                                stale_phase_rate: metrics.stale_phase_rate,
                                reentry_count: metrics.reentry_count,
                                random_control_accuracy: random_control.phase_final_accuracy,
                                delta_vs_baseline_accuracy: delta_acc,
                                delta_vs_baseline_wrong_if_arrived: delta_wrong,
                                has_micro_signal: bucket_has_micro_signal(
                                    &metrics,
                                    &baseline,
                                    &random_control,
                                ),
                                elapsed_sec: started.elapsed().as_secs_f64(),
                            };
                            append_metric_files(&cfg, &row)?;
                            rows.push(row);
                            job_rows += 1;
                            if last_heartbeat.elapsed().as_secs() >= cfg.heartbeat_sec {
                                append_jsonl(
                                    cfg.out.join("progress.jsonl"),
                                    &json!({
                                        "event": "heartbeat",
                                        "job_id": job_id,
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

    write_mechanism_ranking(&cfg, &rows)?;
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
    mechanic: MicroMechanic,
    random_rule: bool,
    random_table: &[[usize; K]; K],
) -> EvalMetrics {
    let mut correct = 0usize;
    let mut best_correct = 0usize;
    let mut arrival = 0usize;
    let mut wrong_arrived = 0usize;
    let mut correct_power = 0.0;
    let mut wrong_power = 0.0;
    let mut net_power = 0.0;
    let mut cancelled_correct = 0usize;
    let mut cancelled_wrong = 0usize;
    let mut wrong_growth = 0.0;
    let mut echo = 0.0;
    let mut backflow = 0.0;
    let mut stale = 0usize;
    let mut reentry = 0.0;
    let mut cf_correct = 0usize;
    let mut shuffled_correct = 0usize;

    for case in cases {
        let label = case.private.label as usize;
        let snapshots = simulate_case(layout, case, ticks, mechanic, random_rule, random_table);
        let selected = if mechanic == MicroMechanic::ArrivalWindowReadoutDiagnostic {
            readout_best(&snapshots, label)
        } else {
            snapshots
                .last()
                .cloned()
                .unwrap_or_else(|| snapshot_from_scores([0.0; K], label, 0))
        };
        let best = readout_best(&snapshots, label);
        correct += usize::from(selected.pred == label);
        best_correct += usize::from(best.pred == label);
        let wrong: f64 = selected
            .probs
            .iter()
            .enumerate()
            .filter(|(phase, _)| *phase != label)
            .map(|(_, prob)| *prob)
            .sum();
        correct_power += selected.probs[label] * selected.total_power;
        wrong_power += wrong * selected.total_power;
        net_power += selected.total_power;
        if selected.total_power > EPS {
            arrival += 1;
            wrong_arrived += usize::from(selected.pred != label);
        }
        if best.probs[label] > selected.probs[label] + 0.05 {
            cancelled_correct += 1;
        }
        if wrong < 0.10 {
            cancelled_wrong += 1;
        }
        wrong_growth += wrong_phase_growth(&snapshots, label);
        let (b, e) = backflow_echo_power(case, &snapshots);
        backflow += b;
        echo += e;
        stale += usize::from(stale_phase(&snapshots, label));
        reentry += reentry_proxy(case, &snapshots);

        let cf = counterfactual_case(case);
        let cf_snaps = simulate_case(layout, &cf, ticks, mechanic, random_rule, random_table);
        let cf_out = cf_snaps
            .last()
            .cloned()
            .unwrap_or_else(|| snapshot_from_scores([0.0; K], cf.private.label as usize, 0));
        cf_correct += usize::from(cf_out.pred == cf.private.label as usize);

        let shuffled = gate_shuffled_case(case);
        let shuf_snaps = simulate_case(
            layout,
            &shuffled,
            ticks,
            mechanic,
            random_rule,
            random_table,
        );
        let shuf_out = shuf_snaps
            .last()
            .cloned()
            .unwrap_or_else(|| snapshot_from_scores([0.0; K], label, 0));
        shuffled_correct += usize::from(shuf_out.pred == label);
    }
    let n = cases.len().max(1) as f64;
    EvalMetrics {
        phase_final_accuracy: correct as f64 / n,
        long_path_accuracy: correct as f64 / n,
        target_arrival_rate: arrival as f64 / n,
        wrong_if_arrived_rate: ratio(wrong_arrived, arrival),
        wrong_phase_growth_rate: wrong_growth / n,
        correct_phase_power: correct_power / n,
        wrong_phase_power: wrong_power / n,
        net_power: net_power / n,
        cancelled_correct_rate: cancelled_correct as f64 / n,
        cancelled_wrong_rate: cancelled_wrong as f64 / n,
        final_minus_best_gap: (best_correct as f64 / n) - (correct as f64 / n),
        gate_shuffle_collapse: 1.0 - shuffled_correct as f64 / n,
        same_target_counterfactual_accuracy: cf_correct as f64 / n,
        echo_power: echo / n,
        backflow_power: backflow / n,
        stale_phase_rate: stale as f64 / n,
        reentry_count: reentry / n,
    }
}

fn simulate_case(
    layout: &Layout,
    case: &Case,
    ticks: usize,
    mechanic: MicroMechanic,
    random_rule: bool,
    random_table: &[[usize; K]; K],
) -> Vec<TargetSnapshot> {
    let cfg = mechanic_config(mechanic);
    if cfg.momentum_lanes {
        simulate_momentum_case(layout, case, ticks, cfg, random_rule, random_table)
    } else {
        simulate_scalar_case(layout, case, ticks, cfg, random_rule, random_table)
    }
}

fn simulate_scalar_case(
    layout: &Layout,
    case: &Case,
    ticks: usize,
    cfg: MechanicConfig,
    random_rule: bool,
    random_table: &[[usize; K]; K],
) -> Vec<TargetSnapshot> {
    let cells = layout.width * layout.width;
    let mut emit = vec![[0.0f64; K]; cells];
    let mut arrive = vec![[0.0f64; K]; cells];
    let mut companion = vec![[0.0f64; K]; cells];
    let mut consumed = vec![false; cells];
    let mut cooldown = vec![0usize; cells];
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
                if consumed[id] || cooldown[id] > 0 {
                    cooldown[id] = cooldown[id].saturating_sub(1);
                    continue;
                }
                let gate = case.public.gates[id] as usize;
                let mut incoming = next_arrive[id];
                if cfg.dual_layer {
                    for phase in 0..K {
                        incoming[phase] += 0.50 * companion[id][phase];
                    }
                }
                for (phase, value) in incoming.iter().copied().enumerate() {
                    let out = if random_rule {
                        random_table[phase][gate]
                    } else {
                        expected_phase(phase, gate)
                    };
                    next_emit[id][out] += value;
                }
                if cfg.signed_cancellation {
                    cancel_opposites(&mut next_emit[id]);
                }
                if cfg.phase_competition {
                    winner_take_all(&mut next_emit[id]);
                }
                if cfg.cell_local_normalization {
                    normalize_lanes(&mut next_emit[id]);
                }
                if cfg.dual_damping {
                    scale_lanes(&mut next_emit[id], 0.82);
                }
                if lane_power(next_emit[id]) > EPS {
                    if cfg.emit_once_consume {
                        consumed[id] = true;
                    }
                    if cfg.refractory_ticks > 0 {
                        cooldown[id] = cfg.refractory_ticks;
                    }
                }
            }
        }
        if cfg.dual_layer {
            for id in 0..cells {
                for phase in 0..K {
                    companion[id][phase] = 0.50 * companion[id][phase] + 0.50 * emit[id][phase];
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
    snapshots
}

fn simulate_momentum_case(
    layout: &Layout,
    case: &Case,
    ticks: usize,
    cfg: MechanicConfig,
    random_rule: bool,
    random_table: &[[usize; K]; K],
) -> Vec<TargetSnapshot> {
    let cells = layout.width * layout.width;
    let mut emit = vec![[[0.0f64; K]; DIRS]; cells];
    let mut consumed = vec![false; cells];
    let mut cooldown = vec![0usize; cells];
    let source_id = layout.cell_id(case.public.source.0, case.public.source.1);
    for dir in 0..DIRS {
        emit[source_id][dir][case.public.source_phase as usize] = 1.0;
    }
    let mut snapshots = Vec::with_capacity(ticks.max(1));

    for tick in 1..=ticks {
        let mut next_arrive = vec![[[0.0f64; K]; DIRS]; cells];
        for y in 0..layout.width {
            for x in 0..layout.width {
                let id = layout.cell_id(y, x);
                if case.public.wall[id] {
                    continue;
                }
                for in_dir in 0..DIRS {
                    for (out_dir, ny, nx) in directional_neighbors(layout.width, y, x) {
                        if cfg.no_reentry && out_dir == reverse_dir(in_dir) {
                            continue;
                        }
                        if cfg.public_gradient
                            && manhattan((ny, nx), case.public.target)
                                >= manhattan((y, x), case.public.target)
                        {
                            continue;
                        }
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
                if consumed[id] || cooldown[id] > 0 {
                    cooldown[id] = cooldown[id].saturating_sub(1);
                    continue;
                }
                let gate = case.public.gates[id] as usize;
                for dir in 0..DIRS {
                    for phase in 0..K {
                        let out_phase = if random_rule {
                            random_table[phase][gate]
                        } else {
                            expected_phase(phase, gate)
                        };
                        next_emit[id][dir][out_phase] += next_arrive[id][dir][phase];
                    }
                    if cfg.phase_competition {
                        winner_take_all(&mut next_emit[id][dir]);
                    }
                    if cfg.cell_local_normalization {
                        normalize_lanes(&mut next_emit[id][dir]);
                    }
                }
                if momentum_power(next_emit[id]) > EPS {
                    if cfg.emit_once_consume {
                        consumed[id] = true;
                    }
                    if cfg.refractory_ticks > 0 {
                        cooldown[id] = cfg.refractory_ticks;
                    }
                }
            }
        }
        emit = next_emit;
        let target_id = layout.cell_id(case.public.target.0, case.public.target.1);
        let mut scores = [0.0f64; K];
        for dir in 0..DIRS {
            for (phase, score) in scores.iter_mut().enumerate() {
                *score += emit[target_id][dir][phase];
            }
        }
        snapshots.push(snapshot_from_scores(
            scores,
            case.private.label as usize,
            tick,
        ));
    }
    snapshots
}

fn scalar_neighbors(
    layout: &Layout,
    case: &Case,
    cell: (usize, usize),
    cfg: MechanicConfig,
) -> Vec<(usize, usize)> {
    neighbors(layout.width, cell.0, cell.1)
        .into_iter()
        .filter(|&(ny, nx)| {
            !cfg.public_gradient
                || manhattan((ny, nx), case.public.target) < manhattan(cell, case.public.target)
        })
        .collect()
}

fn snapshot_from_scores(scores: [f64; K], label: usize, _tick: usize) -> TargetSnapshot {
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
        .unwrap_or_else(|| snapshot_from_scores([0.0; K], label, 0))
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
            split: "micro_barrage_eval".to_string(),
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

fn append_metric_files(cfg: &Config, row: &MechanismRow) -> Result<(), Box<dyn std::error::Error>> {
    append_jsonl(cfg.out.join("metrics.jsonl"), row)?;
    append_jsonl(cfg.out.join("mechanism_metrics.jsonl"), row)?;
    append_jsonl(cfg.out.join("family_metrics.jsonl"), row)?;
    append_jsonl(cfg.out.join("counterfactual_metrics.jsonl"), row)?;
    if row.random_rule {
        append_jsonl(cfg.out.join("random_control_metrics.jsonl"), row)?;
    }
    Ok(())
}

fn refresh_summary(
    cfg: &Config,
    rows: &[MechanismRow],
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
        "top_signal": ranking.iter().find(|row| row.has_micro_signal),
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
    rows: &[MechanismRow],
    ranking: &[RankingRow],
    final_report: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut report = String::new();
    report.push_str("# STABLE_LOOP_PHASE_LOCK_015_MICRO_BARRAGE Report\n\n");
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
    report.push_str("| Mechanism | Acc | Long | Family min | Wrong-if-arrived | Wrong drop | Random | Gap | Signal |\n");
    report.push_str("|---|---:|---:|---:|---:|---:|---:|---:|---|\n");
    for row in ranking {
        report.push_str(&format!(
            "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {} |\n",
            row.mechanic,
            row.phase_final_accuracy,
            row.long_path_accuracy,
            row.family_min_accuracy,
            row.wrong_if_arrived_rate,
            row.delta_vs_baseline_wrong_if_arrived,
            row.random_control_accuracy,
            row.final_minus_best_gap,
            row.has_micro_signal,
        ));
    }
    report.push_str("\n## Claim Boundary\n\n");
    report.push_str("This runner is a mechanism selector only. It does not claim stable transport solved, production architecture, full VRAXION, consciousness, language grounding, Prismion uniqueness, or physical quantum behavior.\n");
    fs::write(cfg.out.join("report.md"), report)?;
    Ok(())
}

fn write_mechanism_ranking(
    cfg: &Config,
    rows: &[MechanismRow],
) -> Result<(), Box<dyn std::error::Error>> {
    let ranking = mechanism_ranking(rows);
    fs::write(
        cfg.out.join("mechanism_ranking.json"),
        serde_json::to_string_pretty(&ranking)?,
    )?;
    Ok(())
}

fn mechanism_ranking(rows: &[MechanismRow]) -> Vec<RankingRow> {
    let baseline_rows = rows
        .iter()
        .filter(|row| row.mechanic == MicroMechanic::BaselineFull16.as_str() && !row.random_rule)
        .collect::<Vec<_>>();
    let baseline = aggregate_for_mechanic(MicroMechanic::BaselineFull16, &baseline_rows, 0.0);
    let mut out = Vec::new();
    for mechanic in mechanics() {
        let subset = rows
            .iter()
            .filter(|row| row.mechanic == mechanic.as_str() && !row.random_rule)
            .collect::<Vec<_>>();
        if subset.is_empty() {
            continue;
        }
        let random_acc = mean(
            rows.iter()
                .filter(|row| row.mechanic == mechanic.as_str() && row.random_rule)
                .map(|row| row.phase_final_accuracy),
        );
        let mut row = aggregate_for_mechanic(mechanic, &subset, random_acc);
        row.delta_vs_baseline_accuracy = row.phase_final_accuracy - baseline.phase_final_accuracy;
        row.delta_vs_baseline_wrong_if_arrived =
            baseline.wrong_if_arrived_rate - row.wrong_if_arrived_rate;
        row.has_micro_signal = has_micro_signal(&row, &baseline);
        out.push(row);
    }
    out.sort_by(|a, b| {
        b.has_micro_signal
            .cmp(&a.has_micro_signal)
            .then_with(|| {
                b.delta_vs_baseline_wrong_if_arrived
                    .partial_cmp(&a.delta_vs_baseline_wrong_if_arrived)
                    .unwrap()
            })
            .then_with(|| {
                b.long_path_accuracy
                    .partial_cmp(&a.long_path_accuracy)
                    .unwrap()
            })
    });
    out
}

fn aggregate_for_mechanic(
    mechanic: MicroMechanic,
    rows: &[&MechanismRow],
    random_acc: f64,
) -> RankingRow {
    RankingRow {
        mechanic: mechanic.as_str().to_string(),
        phase_final_accuracy: mean(rows.iter().map(|row| row.phase_final_accuracy)),
        long_path_accuracy: mean(
            rows.iter()
                .filter(|row| row.path_length >= 8)
                .map(|row| row.phase_final_accuracy),
        ),
        family_min_accuracy: family_min_accuracy(rows),
        wrong_if_arrived_rate: mean(rows.iter().map(|row| row.wrong_if_arrived_rate)),
        wrong_phase_growth_rate: mean(rows.iter().map(|row| row.wrong_phase_growth_rate)),
        final_minus_best_gap: mean(rows.iter().map(|row| row.final_minus_best_gap)),
        random_control_accuracy: random_acc,
        delta_vs_baseline_accuracy: 0.0,
        delta_vs_baseline_wrong_if_arrived: 0.0,
        has_micro_signal: false,
    }
}

fn has_micro_signal(row: &RankingRow, baseline: &RankingRow) -> bool {
    row.long_path_accuracy - baseline.long_path_accuracy >= 0.10
        && row.family_min_accuracy - baseline.family_min_accuracy >= 0.20
        && baseline.wrong_if_arrived_rate - row.wrong_if_arrived_rate >= 0.10
        && row.final_minus_best_gap <= baseline.final_minus_best_gap + 0.05
        && row.random_control_accuracy < 0.45
}

fn bucket_has_micro_signal(
    metrics: &EvalMetrics,
    baseline: &EvalMetrics,
    random: &EvalMetrics,
) -> bool {
    metrics.long_path_accuracy - baseline.long_path_accuracy >= 0.10
        && metrics.phase_final_accuracy - baseline.phase_final_accuracy >= 0.10
        && baseline.wrong_if_arrived_rate - metrics.wrong_if_arrived_rate >= 0.10
        && metrics.final_minus_best_gap <= baseline.final_minus_best_gap + 0.05
        && random.phase_final_accuracy < 0.45
}

fn verdicts(rows: &[MechanismRow], ranking: &[RankingRow]) -> Vec<String> {
    if rows.is_empty() {
        return vec!["RUNNING".to_string()];
    }
    let mut out = BTreeSet::new();
    for row in ranking.iter().filter(|row| row.has_micro_signal) {
        match row.mechanic.as_str() {
            "SIGNED_PHASE_CANCELLATION" | "SIGNED_PLUS_CELL_NORMALIZATION" => {
                out.insert("SIGNED_CANCELLATION_HAS_SIGNAL".to_string());
            }
            "DUAL_LAYER_EB_FIELD" | "DUAL_LAYER_PLUS_DAMPING" => {
                out.insert("DUAL_LAYER_HAS_SIGNAL".to_string());
            }
            "MOMENTUM_LANES" | "MOMENTUM_PLUS_CONSUME" => {
                out.insert("MOMENTUM_HAS_SIGNAL".to_string());
            }
            "PHASE_COMPETITION_PER_CELL" => {
                out.insert("PHASE_COMPETITION_HAS_SIGNAL".to_string());
            }
            _ => {}
        }
    }
    if ranking
        .iter()
        .any(|row| row.mechanic.contains("SIGNED") && row.family_min_accuracy < 0.35)
    {
        out.insert("SIGNED_CANCELLATION_KILLS_CORRECT_SIGNAL".to_string());
    }
    if ranking
        .iter()
        .filter(|row| row.mechanic.contains("DUAL_LAYER"))
        .all(|row| !row.has_micro_signal)
    {
        out.insert("DUAL_LAYER_NO_SIGNAL".to_string());
    }
    let baseline_gap = ranking
        .iter()
        .find(|row| row.mechanic == MicroMechanic::BaselineFull16.as_str())
        .map(|row| row.final_minus_best_gap)
        .unwrap_or(0.0);
    if ranking
        .iter()
        .any(|row| row.mechanic.contains("CONSUME") && row.final_minus_best_gap <= baseline_gap)
    {
        out.insert("CONSUME_REDUCES_ECHO".to_string());
    }
    if ranking
        .iter()
        .any(|row| row.mechanic == "REFRACTORY_CELL" && row.family_min_accuracy > 0.40)
    {
        out.insert("REFRACTORY_REDUCES_STALE_PHASE".to_string());
    }
    if ranking
        .iter()
        .any(|row| row.mechanic == "NO_REENTRY_MOMENTUM" && row.has_micro_signal)
    {
        out.insert("NO_REENTRY_REDUCES_BACKFLOW".to_string());
    }
    if ranking
        .iter()
        .any(|row| row.mechanic == "ARRIVAL_WINDOW_READOUT_DIAGNOSTIC")
    {
        out.insert("READOUT_WINDOW_ONLY_NOT_TRANSPORT".to_string());
    }
    if ranking
        .iter()
        .any(|row| row.random_control_accuracy >= 0.65 && row.phase_final_accuracy >= 0.80)
    {
        out.insert("MECHANISM_OVERPOWERS_RULE_CONTROL".to_string());
    }
    if ranking.iter().all(|row| !row.has_micro_signal) {
        out.insert("NO_MICRO_MECHANIC_RESCUES".to_string());
    }
    out.insert("PRODUCTION_API_NOT_READY".to_string());
    out.into_iter().collect()
}

fn family_min_accuracy(rows: &[&MechanismRow]) -> f64 {
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

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = Config {
        out: PathBuf::from("target/pilot_wave/stable_loop_phase_lock_015_micro_barrage/dev"),
        seeds: vec![2026],
        eval_examples: 256,
        widths: vec![8, 12],
        path_lengths: vec![4, 8, 16, 24],
        ticks_list: vec![8, 16, 24, 32],
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
            "probe": "STABLE_LOOP_PHASE_LOCK_015_MICRO_BARRAGE",
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
        "mechanism_metrics.jsonl",
        "random_control_metrics.jsonl",
        "family_metrics.jsonl",
        "counterfactual_metrics.jsonl",
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
    let mut out = Vec::new();
    for mechanic in mechanics() {
        out.push(Arm {
            name: mechanic.as_str().to_string(),
            mechanic,
            random_rule: false,
        });
        out.push(Arm {
            name: format!("RANDOM_WITH_{}", mechanic.as_str()),
            mechanic,
            random_rule: true,
        });
    }
    out.push(Arm {
        name: "RANDOM_CONTROL_BASE".to_string(),
        mechanic: MicroMechanic::BaselineFull16,
        random_rule: true,
    });
    out
}

fn mechanics() -> Vec<MicroMechanic> {
    vec![
        MicroMechanic::BaselineFull16,
        MicroMechanic::BestPublicCombo014,
        MicroMechanic::SignedPhaseCancellation,
        MicroMechanic::DualLayerEbField,
        MicroMechanic::MomentumLanes,
        MicroMechanic::EmitOnceConsume,
        MicroMechanic::RefractoryCell,
        MicroMechanic::NoReentryMomentum,
        MicroMechanic::PhaseCompetitionPerCell,
        MicroMechanic::ArrivalWindowReadoutDiagnostic,
        MicroMechanic::MomentumPlusConsume,
        MicroMechanic::SignedPlusCellNormalization,
        MicroMechanic::DualLayerPlusDamping,
    ]
}

fn mechanic_config(mechanic: MicroMechanic) -> MechanicConfig {
    match mechanic {
        MicroMechanic::BaselineFull16 => MechanicConfig::default(),
        MicroMechanic::BestPublicCombo014 => MechanicConfig {
            arrive_latch_1tick: true,
            cell_local_normalization: true,
            public_gradient: true,
            ..Default::default()
        },
        MicroMechanic::SignedPhaseCancellation => MechanicConfig {
            signed_cancellation: true,
            ..Default::default()
        },
        MicroMechanic::DualLayerEbField => MechanicConfig {
            dual_layer: true,
            ..Default::default()
        },
        MicroMechanic::MomentumLanes => MechanicConfig {
            momentum_lanes: true,
            ..Default::default()
        },
        MicroMechanic::EmitOnceConsume => MechanicConfig {
            emit_once_consume: true,
            ..Default::default()
        },
        MicroMechanic::RefractoryCell => MechanicConfig {
            refractory_ticks: 2,
            ..Default::default()
        },
        MicroMechanic::NoReentryMomentum => MechanicConfig {
            momentum_lanes: true,
            no_reentry: true,
            ..Default::default()
        },
        MicroMechanic::PhaseCompetitionPerCell => MechanicConfig {
            phase_competition: true,
            ..Default::default()
        },
        MicroMechanic::ArrivalWindowReadoutDiagnostic => MechanicConfig {
            ..Default::default()
        },
        MicroMechanic::MomentumPlusConsume => MechanicConfig {
            momentum_lanes: true,
            no_reentry: true,
            emit_once_consume: true,
            ..Default::default()
        },
        MicroMechanic::SignedPlusCellNormalization => MechanicConfig {
            signed_cancellation: true,
            cell_local_normalization: true,
            ..Default::default()
        },
        MicroMechanic::DualLayerPlusDamping => MechanicConfig {
            dual_layer: true,
            dual_damping: true,
            ..Default::default()
        },
    }
}

impl Default for MechanicConfig {
    fn default() -> Self {
        Self {
            arrive_latch_1tick: false,
            cell_local_normalization: false,
            public_gradient: false,
            signed_cancellation: false,
            dual_layer: false,
            dual_damping: false,
            momentum_lanes: false,
            emit_once_consume: false,
            refractory_ticks: 0,
            no_reentry: false,
            phase_competition: false,
        }
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

fn reverse_dir(dir: usize) -> usize {
    match dir {
        0 => 1,
        1 => 0,
        2 => 3,
        _ => 2,
    }
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

fn scale_lanes(lanes: &mut [f64; K], factor: f64) {
    for value in lanes {
        *value *= factor;
    }
}

fn cancel_opposites(lanes: &mut [f64; K]) {
    for (a, b) in [(0usize, 2usize), (1usize, 3usize)] {
        let av = lanes[a].max(0.0);
        let bv = lanes[b].max(0.0);
        if av >= bv {
            lanes[a] = av - bv;
            lanes[b] = 0.0;
        } else {
            lanes[b] = bv - av;
            lanes[a] = 0.0;
        }
    }
}

fn winner_take_all(lanes: &mut [f64; K]) {
    let winner = argmax(&[
        lanes[0].max(0.0),
        lanes[1].max(0.0),
        lanes[2].max(0.0),
        lanes[3].max(0.0),
    ]);
    for phase in 0..K {
        if phase != winner {
            lanes[phase] = 0.0;
        }
    }
}

fn lane_power(lanes: [f64; K]) -> f64 {
    lanes.iter().map(|v| v.max(0.0)).sum()
}

fn momentum_power(lanes: [[f64; K]; DIRS]) -> f64 {
    lanes
        .iter()
        .flat_map(|row| row.iter())
        .map(|v| v.max(0.0))
        .sum()
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

fn reentry_proxy(case: &Case, snapshots: &[TargetSnapshot]) -> f64 {
    let path_len = case.private.true_path.len().max(1) as f64;
    snapshots
        .windows(2)
        .filter(|pair| pair[1].total_power > pair[0].total_power)
        .count() as f64
        / path_len
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

const CONTRACT_SNAPSHOT: &str = r#"# STABLE_LOOP_PHASE_LOCK_015_MICRO_BARRAGE

Mechanism-selection micro-barrage for phase-lane long-chain transport. The
probe compares runner-local field/transport mechanics and only reports which
principle has signal. It does not claim stable transport solved or change public
instnct-core APIs.
"#;
