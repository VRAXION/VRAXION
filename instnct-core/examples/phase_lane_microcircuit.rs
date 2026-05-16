//! Runner-local phase-lane microcircuit probe.
//!
//! This example isolates the local construct that 007 identified as missing:
//! phase_i + gate_g -> phase_(i+g). It uses `instnct-core::Network` and the
//! canonical traced jackpot mutator for repair/growth stages. A runner-local
//! explicit coincidence operator is available only as a diagnostic.

use instnct_core::{
    evolution_step_jackpot_traced_with_policy_and_operator_weights, mutation_operator_index,
    AcceptancePolicy, CandidateTraceRecord, EvolutionConfig, Int8Projection, Network,
    PropagationConfig,
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

const K: usize = 4;
const PHASE_BASE: usize = 0;
const GATE_BASE: usize = PHASE_BASE + K;
const COINCIDENCE_BASE: usize = GATE_BASE + K;
const OUTPUT_BASE: usize = COINCIDENCE_BASE + K * K;
const NEURON_COUNT: usize = OUTPUT_BASE + K;

#[derive(Clone, Debug)]
struct Config {
    out: PathBuf,
    seeds: Vec<u64>,
    steps: usize,
    jackpot: usize,
    heartbeat_sec: u64,
    include_explicit_coincidence: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
enum DamageKind {
    Edge,
    Threshold,
    Channel,
    Polarity,
}

impl DamageKind {
    fn as_str(self) -> &'static str {
        match self {
            DamageKind::Edge => "edge_removed",
            DamageKind::Threshold => "threshold_shifted",
            DamageKind::Channel => "channel_shifted",
            DamageKind::Polarity => "polarity_flipped",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
enum Stage {
    HandBuilt,
    DamageRepair { kind: DamageKind, level: usize },
    PartialSeed { present_pairs: usize },
    RandomGrowth,
    ExplicitCoincidencePartial { present_pairs: usize },
    ExplicitCoincidenceRandom,
}

impl Stage {
    fn name(&self) -> String {
        match self {
            Stage::HandBuilt => "HAND_BUILT_PHASE_LANE_MOTIF".to_string(),
            Stage::DamageRepair { kind, level } => {
                format!("DAMAGE_REPAIR_MOTIF_{}_{}", kind.as_str(), level)
            }
            Stage::PartialSeed { present_pairs } => {
                format!("PARTIAL_SEED_COMPLETION_{}_PAIRS", present_pairs)
            }
            Stage::RandomGrowth => "RANDOM_GROWTH_BASELINE".to_string(),
            Stage::ExplicitCoincidencePartial { present_pairs } => {
                format!(
                    "EXPLICIT_COINCIDENCE_OPERATOR_PARTIAL_{}_PAIRS",
                    present_pairs
                )
            }
            Stage::ExplicitCoincidenceRandom => "EXPLICIT_COINCIDENCE_OPERATOR_RANDOM".to_string(),
        }
    }

    fn uses_canonical_evolution(&self) -> bool {
        matches!(
            self,
            Stage::DamageRepair { .. } | Stage::PartialSeed { .. } | Stage::RandomGrowth
        )
    }

    fn uses_explicit_operator(&self) -> bool {
        matches!(
            self,
            Stage::ExplicitCoincidencePartial { .. } | Stage::ExplicitCoincidenceRandom
        )
    }
}

#[derive(Clone, Debug, Serialize)]
struct PublicPair {
    phase_lane: u8,
    gate_lane: u8,
}

#[derive(Clone, Debug, Serialize)]
struct PrivatePair {
    label: u8,
}

#[derive(Clone, Debug)]
struct Case {
    public: PublicPair,
    private: PrivatePair,
}

#[derive(Clone, Debug, Default, Serialize)]
struct EvalMetrics {
    all_16_phase_gate_pairs_accuracy: f64,
    single_step_phase_rotation_accuracy: f64,
    correct_phase_probability_mean: f64,
    edge_count: usize,
    solved: bool,
    forbidden_private_field_leak: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
struct CandidateStats {
    positive_candidate_delta_fraction: f64,
    nonzero_candidate_delta_fraction: f64,
    mean_delta: f64,
    mean_abs_delta: f64,
    candidate_delta_std: f64,
    evaluated_candidates: usize,
    accepted_candidates: usize,
    accepted_operator_rate: f64,
}

#[derive(Clone, Debug, Serialize)]
struct MetricRow {
    job_id: String,
    seed: u64,
    stage: String,
    checkpoint_step: usize,
    final_row: bool,
    all_16_phase_gate_pairs_accuracy: f64,
    single_step_phase_rotation_accuracy: f64,
    correct_phase_probability_mean: f64,
    motif_repair_success_rate: f64,
    partial_seed_completion_rate: f64,
    random_growth_success_rate: f64,
    positive_candidate_delta_fraction: f64,
    nonzero_candidate_delta_fraction: f64,
    accepted_operator_rate: f64,
    accepted_candidates: usize,
    evaluated_candidates: usize,
    edge_count: usize,
    threshold_sensitivity: f64,
    channel_sensitivity: f64,
    polarity_sensitivity: f64,
    damage_level: usize,
    forbidden_private_field_leak: f64,
    elapsed_sec: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = parse_args()?;
    fs::create_dir_all(&cfg.out)?;
    fs::create_dir_all(cfg.out.join("job_progress"))?;
    write_static_run_files(&cfg)?;

    let stages = stages(&cfg);
    let total = cfg.seeds.len() * stages.len();
    append_jsonl(
        cfg.out.join("progress.jsonl"),
        &json!({"event": "run_start", "time": now_sec(), "jobs": total}),
    )?;

    let started = Instant::now();
    let mut rows = Vec::new();
    let mut completed = 0usize;
    for seed in cfg.seeds.iter().copied() {
        for stage in stages.iter().cloned() {
            let row = run_job(&cfg, seed, stage.clone(), started)?;
            rows.push(row);
            completed += 1;
            append_jsonl(
                cfg.out.join("progress.jsonl"),
                &json!({
                    "event": "job_done",
                    "time": now_sec(),
                    "completed": completed,
                    "total": total,
                    "seed": seed,
                    "stage": stage.name(),
                }),
            )?;
            refresh_summary(
                &cfg,
                &rows,
                completed,
                total,
                started.elapsed().as_secs_f64(),
            )?;
        }
    }

    write_operator_summary(&cfg)?;
    write_report(&cfg, &rows, true)?;
    append_jsonl(
        cfg.out.join("progress.jsonl"),
        &json!({"event": "run_done", "time": now_sec(), "completed": completed, "total": total}),
    )?;
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = Config {
        out: PathBuf::from(
            "target/pilot_wave/stable_loop_phase_lock_008_phase_lane_microcircuit/dev",
        ),
        seeds: vec![2026],
        steps: 100,
        jackpot: 6,
        heartbeat_sec: 15,
        include_explicit_coincidence: false,
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
            "--jackpot" => {
                i += 1;
                cfg.jackpot = args[i].parse()?;
            }
            "--heartbeat-sec" => {
                i += 1;
                cfg.heartbeat_sec = args[i].parse()?;
            }
            "--include-explicit-coincidence" => {
                cfg.include_explicit_coincidence = true;
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        i += 1;
    }

    if cfg.seeds.is_empty() {
        return Err("--seeds produced no seeds".into());
    }
    Ok(cfg)
}

fn parse_seeds(raw: &str) -> Vec<u64> {
    if let Some((a, b)) = raw.split_once('-') {
        let start = a.parse::<u64>().unwrap_or(2026);
        let end = b.parse::<u64>().unwrap_or(start);
        return (start..=end).collect();
    }
    raw.split(',')
        .filter_map(|s| s.trim().parse::<u64>().ok())
        .collect()
}

fn stages(cfg: &Config) -> Vec<Stage> {
    let mut out = vec![Stage::HandBuilt];
    for kind in [
        DamageKind::Edge,
        DamageKind::Threshold,
        DamageKind::Channel,
        DamageKind::Polarity,
    ] {
        for level in 1..=3 {
            out.push(Stage::DamageRepair { kind, level });
        }
    }
    for present_pairs in [4usize, 8, 12] {
        out.push(Stage::PartialSeed { present_pairs });
    }
    out.push(Stage::RandomGrowth);
    if cfg.include_explicit_coincidence {
        out.push(Stage::ExplicitCoincidencePartial { present_pairs: 4 });
        out.push(Stage::ExplicitCoincidenceRandom);
    }
    out
}

fn run_job(
    cfg: &Config,
    seed: u64,
    stage: Stage,
    run_started: Instant,
) -> Result<MetricRow, Box<dyn std::error::Error>> {
    let job_id = format!("{}_{}", seed, stage.name());
    let job_path = cfg.out.join("job_progress").join(format!("{job_id}.jsonl"));
    append_jsonl(
        &job_path,
        &json!({"event": "job_start", "time": now_sec(), "seed": seed, "stage": stage.name()}),
    )?;
    let started = Instant::now();
    let cases = all_cases();
    let mut rng = StdRng::seed_from_u64(seed ^ stage_seed_salt(&stage));
    let mut net = initial_network(&stage, &mut rng);
    let mut projection = Int8Projection::new(1, 1, &mut rng);
    let mut candidate_deltas = Vec::new();
    let mut accepted_candidates = 0usize;

    let mut final_metrics = eval_network(&mut net, &cases);
    if stage.uses_canonical_evolution() {
        let mut eval_rng = StdRng::seed_from_u64(seed ^ 0xCAFE_BABE);
        let evo_cfg = EvolutionConfig {
            edge_cap: 96,
            accept_ties: false,
        };
        let operator_weights = canonical_operator_weights();
        let checkpoint_every = 20usize.max(cfg.steps / 20).min(100);
        let mut last_heartbeat = Instant::now();
        let mut writer = BufWriter::new(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(cfg.out.join("candidate_log.jsonl"))?,
        );
        for step in 0..cfg.steps {
            evolution_step_jackpot_traced_with_policy_and_operator_weights(
                &mut net,
                &mut projection,
                &mut rng,
                &mut eval_rng,
                |candidate_net, _projection, _rng| {
                    eval_network(candidate_net, &cases).correct_phase_probability_mean
                },
                &evo_cfg,
                AcceptancePolicy::Strict,
                cfg.jackpot,
                step,
                Some(&operator_weights),
                |record| {
                    if record.evaluated {
                        candidate_deltas.push(record.delta_u);
                    }
                    if record.accepted {
                        accepted_candidates += 1;
                    }
                    let _ = write_candidate_record(&mut writer, &job_id, seed, &stage, record);
                },
            );
            writer.flush()?;
            if (step + 1) % checkpoint_every == 0
                || step + 1 == cfg.steps
                || last_heartbeat.elapsed().as_secs() >= cfg.heartbeat_sec
            {
                let metrics = eval_network(&mut net, &cases);
                let stats = candidate_stats(&candidate_deltas, accepted_candidates);
                let row = metric_row(
                    &job_id,
                    seed,
                    &stage,
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
                        "accuracy": metrics.all_16_phase_gate_pairs_accuracy,
                        "probability": metrics.correct_phase_probability_mean,
                        "accepted_candidates": accepted_candidates,
                    }),
                )?;
                refresh_summary_partial(cfg, run_started.elapsed().as_secs_f64())?;
                last_heartbeat = Instant::now();
            }
        }
        final_metrics = eval_network(&mut net, &cases);
    } else if stage.uses_explicit_operator() {
        let checkpoint_every = 20usize.max(cfg.steps / 20).min(100);
        let mut last_heartbeat = Instant::now();
        let mut writer = BufWriter::new(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(cfg.out.join("candidate_log.jsonl"))?,
        );
        for step in 0..cfg.steps {
            explicit_coincidence_step(
                &mut net,
                &cases,
                &mut rng,
                cfg.jackpot,
                step,
                &job_id,
                seed,
                &stage,
                &mut writer,
                &mut candidate_deltas,
                &mut accepted_candidates,
            )?;
            writer.flush()?;
            if (step + 1) % checkpoint_every == 0
                || step + 1 == cfg.steps
                || last_heartbeat.elapsed().as_secs() >= cfg.heartbeat_sec
            {
                let metrics = eval_network(&mut net, &cases);
                let stats = candidate_stats(&candidate_deltas, accepted_candidates);
                let row = metric_row(
                    &job_id,
                    seed,
                    &stage,
                    step + 1,
                    false,
                    &metrics,
                    &stats,
                    started.elapsed().as_secs_f64(),
                );
                append_jsonl(cfg.out.join("metrics.jsonl"), &row)?;
                refresh_summary_partial(cfg, run_started.elapsed().as_secs_f64())?;
                last_heartbeat = Instant::now();
            }
        }
        final_metrics = eval_network(&mut net, &cases);
    }

    let stats = candidate_stats(&candidate_deltas, accepted_candidates);
    let row = metric_row(
        &job_id,
        seed,
        &stage,
        cfg.steps,
        true,
        &final_metrics,
        &stats,
        started.elapsed().as_secs_f64(),
    );
    append_jsonl(cfg.out.join("metrics.jsonl"), &row)?;
    append_jsonl(cfg.out.join("motif_sensitivity.jsonl"), &row)?;
    if matches!(stage, Stage::DamageRepair { .. }) {
        append_jsonl(cfg.out.join("damage_level_success_curve.jsonl"), &row)?;
    }
    append_jsonl(
        &job_path,
        &json!({"event": "job_done", "time": now_sec(), "row": row}),
    )?;
    Ok(row)
}

fn all_cases() -> Vec<Case> {
    let mut out = Vec::with_capacity(16);
    for phase in 0..K {
        for gate in 0..K {
            out.push(Case {
                public: PublicPair {
                    phase_lane: phase as u8,
                    gate_lane: gate as u8,
                },
                private: PrivatePair {
                    label: ((phase + gate) % K) as u8,
                },
            });
        }
    }
    out
}

fn initial_network(stage: &Stage, rng: &mut StdRng) -> Network {
    match stage {
        Stage::HandBuilt => hand_built_motif(),
        Stage::DamageRepair { kind, level } => {
            let mut net = hand_built_motif();
            damage_motif(&mut net, *kind, *level);
            net
        }
        Stage::PartialSeed { present_pairs } => partial_seed_motif(*present_pairs),
        Stage::RandomGrowth => random_local_motif(rng),
        Stage::ExplicitCoincidencePartial { present_pairs } => partial_seed_motif(*present_pairs),
        Stage::ExplicitCoincidenceRandom => random_local_motif(rng),
    }
}

fn hand_built_motif() -> Network {
    let mut net = empty_motif();
    for phase in 0..K {
        for gate in 0..K {
            add_correct_pair(&mut net, phase, gate);
        }
    }
    net
}

fn empty_motif() -> Network {
    let mut net = Network::new(NEURON_COUNT);
    for spike in net.spike_data_mut() {
        spike.charge = 0;
        spike.threshold = 0;
        spike.channel = 1;
    }
    for phase in 0..K {
        for gate in 0..K {
            net.spike_data_mut()[coincidence(phase, gate)].threshold = 1;
        }
    }
    for p in net.polarity_mut() {
        *p = 1;
    }
    net
}

fn partial_seed_motif(present_pairs: usize) -> Network {
    let mut net = empty_motif();
    for pair in 0..present_pairs.min(K * K) {
        let phase = pair / K;
        let gate = pair % K;
        add_correct_pair(&mut net, phase, gate);
    }
    net
}

fn random_local_motif(rng: &mut StdRng) -> Network {
    let mut net = empty_motif();
    for _ in 0..16 {
        let source = rng.gen_range(0..NEURON_COUNT) as u16;
        let target = rng.gen_range(0..NEURON_COUNT) as u16;
        net.graph_mut().add_edge(source, target);
    }
    net
}

fn add_correct_pair(net: &mut Network, phase: usize, gate: usize) {
    let output = (phase + gate) % K;
    let c = coincidence(phase, gate);
    net.graph_mut()
        .add_edge(phase_input(phase) as u16, c as u16);
    net.graph_mut().add_edge(gate_input(gate) as u16, c as u16);
    net.graph_mut()
        .add_edge(c as u16, output_lane(output) as u16);
    net.spike_data_mut()[c].threshold = 1;
    net.spike_data_mut()[c].channel = 1;
    net.polarity_mut()[c] = 1;
}

fn damage_motif(net: &mut Network, kind: DamageKind, level: usize) {
    for pair in 0..level.min(K * K) {
        let phase = pair / K;
        let gate = pair % K;
        let c = coincidence(phase, gate);
        match kind {
            DamageKind::Edge => {
                net.graph_mut()
                    .remove_edge(phase_input(phase) as u16, c as u16);
            }
            DamageKind::Threshold => {
                net.spike_data_mut()[c].threshold = 2;
            }
            DamageKind::Channel => {
                net.spike_data_mut()[c].channel = 5;
            }
            DamageKind::Polarity => {
                net.polarity_mut()[c] = -1;
            }
        }
    }
}

fn eval_network(net: &mut Network, cases: &[Case]) -> EvalMetrics {
    let mut correct = 0usize;
    let mut total_prob = 0.0;
    for case in cases {
        let probs = predict_probs(net, &case.public);
        let pred = argmax(&probs);
        let label = case.private.label as usize;
        correct += usize::from(pred == label);
        total_prob += probs[label];
    }
    let acc = correct as f64 / cases.len().max(1) as f64;
    EvalMetrics {
        all_16_phase_gate_pairs_accuracy: acc,
        single_step_phase_rotation_accuracy: acc,
        correct_phase_probability_mean: total_prob / cases.len().max(1) as f64,
        edge_count: net.edge_count(),
        solved: acc >= 1.0,
        forbidden_private_field_leak: 0.0,
    }
}

fn predict_probs(net: &mut Network, pair: &PublicPair) -> [f64; K] {
    net.reset();
    let indices = [
        phase_input(pair.phase_lane as usize) as u16,
        gate_input(pair.gate_lane as usize) as u16,
    ];
    let values = [1i8, 1i8];
    let config = PropagationConfig {
        ticks_per_token: 2,
        input_duration_ticks: 1,
        decay_interval_ticks: 16,
        use_refractory: false,
    };
    let _ = net.propagate_sparse(&indices, &values, &config);
    let mut scores = [0.0f64; K];
    for (phase, score) in scores.iter_mut().enumerate() {
        let idx = output_lane(phase);
        *score = net.spike_data()[idx].charge as f64 + 4.0 * net.activation()[idx].max(0) as f64;
    }
    normalize_scores(scores)
}

fn explicit_coincidence_step(
    net: &mut Network,
    cases: &[Case],
    rng: &mut StdRng,
    jackpot: usize,
    step: usize,
    job_id: &str,
    seed: u64,
    stage: &Stage,
    writer: &mut BufWriter<File>,
    deltas: &mut Vec<f64>,
    accepted_candidates: &mut usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let before = eval_network(net, cases).correct_phase_probability_mean;
    let parent = net.save_state();
    let mut best_delta = f64::NEG_INFINITY;
    let mut best_snapshot = None;
    let mut records = Vec::with_capacity(jackpot);
    for candidate_id in 0..jackpot {
        net.restore_state(&parent);
        let mutated = add_random_coincidence_gate(net, rng);
        let after = if mutated {
            eval_network(net, cases).correct_phase_probability_mean
        } else {
            before
        };
        let delta = after - before;
        if mutated {
            deltas.push(delta);
        }
        if mutated && delta > best_delta {
            best_delta = delta;
            best_snapshot = Some(net.save_state());
        }
        records.push((candidate_id, mutated, after, delta));
    }
    net.restore_state(&parent);
    let accepted = best_delta > 0.0;
    if accepted {
        if let Some(snapshot) = best_snapshot {
            net.restore_state(&snapshot);
            *accepted_candidates += 1;
        }
    }
    for (candidate_id, mutated, after, delta) in records {
        serde_json::to_writer(
            &mut *writer,
            &json!({
                "job_id": job_id,
                "seed": seed,
                "stage": stage.name(),
                "step": step,
                "candidate_id": candidate_id,
                "operator_id": "add_coincidence_gate",
                "mutated": mutated,
                "evaluated": mutated,
                "before_u": before,
                "after_u": after,
                "delta_u": delta,
                "selected": mutated && (delta - best_delta).abs() <= 1e-12,
                "accepted": accepted && mutated && (delta - best_delta).abs() <= 1e-12,
            }),
        )?;
        writeln!(writer)?;
    }
    Ok(())
}

fn add_random_coincidence_gate(net: &mut Network, rng: &mut StdRng) -> bool {
    let phase = rng.gen_range(0..K);
    let gate = rng.gen_range(0..K);
    let output = rng.gen_range(0..K);
    let c = coincidence(phase, gate);
    let mut mutated = false;
    mutated |= net
        .graph_mut()
        .add_edge(phase_input(phase) as u16, c as u16);
    mutated |= net.graph_mut().add_edge(gate_input(gate) as u16, c as u16);
    mutated |= net
        .graph_mut()
        .add_edge(c as u16, output_lane(output) as u16);
    if net.spike_data()[c].threshold != 1 {
        net.spike_data_mut()[c].threshold = 1;
        mutated = true;
    }
    if net.spike_data()[c].channel != 1 {
        net.spike_data_mut()[c].channel = 1;
        mutated = true;
    }
    if net.polarity()[c] != 1 {
        net.polarity_mut()[c] = 1;
        mutated = true;
    }
    mutated
}

fn phase_input(phase: usize) -> usize {
    PHASE_BASE + phase
}

fn gate_input(gate: usize) -> usize {
    GATE_BASE + gate
}

fn coincidence(phase: usize, gate: usize) -> usize {
    COINCIDENCE_BASE + phase * K + gate
}

fn output_lane(phase: usize) -> usize {
    OUTPUT_BASE + phase
}

fn canonical_operator_weights() -> Vec<f64> {
    let mut weights = vec![1.0; instnct_core::MUTATION_OPERATORS.len()];
    if let Some(idx) = mutation_operator_index("projection_weight") {
        weights[idx] = 0.0;
    }
    weights
}

fn metric_row(
    job_id: &str,
    seed: u64,
    stage: &Stage,
    step: usize,
    final_row: bool,
    metrics: &EvalMetrics,
    stats: &CandidateStats,
    elapsed_sec: f64,
) -> MetricRow {
    let is_repair = matches!(stage, Stage::DamageRepair { .. });
    let is_partial = matches!(
        stage,
        Stage::PartialSeed { .. } | Stage::ExplicitCoincidencePartial { .. }
    );
    let is_random = matches!(
        stage,
        Stage::RandomGrowth | Stage::ExplicitCoincidenceRandom
    );
    let damage_level = match stage {
        Stage::DamageRepair { level, .. } => *level,
        _ => 0,
    };
    let threshold_sensitivity = if matches!(
        stage,
        Stage::DamageRepair {
            kind: DamageKind::Threshold,
            ..
        }
    ) {
        1.0 - metrics.all_16_phase_gate_pairs_accuracy
    } else {
        0.0
    };
    let channel_sensitivity = if matches!(
        stage,
        Stage::DamageRepair {
            kind: DamageKind::Channel,
            ..
        }
    ) {
        1.0 - metrics.all_16_phase_gate_pairs_accuracy
    } else {
        0.0
    };
    let polarity_sensitivity = if matches!(
        stage,
        Stage::DamageRepair {
            kind: DamageKind::Polarity,
            ..
        }
    ) {
        1.0 - metrics.all_16_phase_gate_pairs_accuracy
    } else {
        0.0
    };
    MetricRow {
        job_id: job_id.to_string(),
        seed,
        stage: stage.name(),
        checkpoint_step: step,
        final_row,
        all_16_phase_gate_pairs_accuracy: metrics.all_16_phase_gate_pairs_accuracy,
        single_step_phase_rotation_accuracy: metrics.single_step_phase_rotation_accuracy,
        correct_phase_probability_mean: metrics.correct_phase_probability_mean,
        motif_repair_success_rate: if is_repair && metrics.solved {
            1.0
        } else {
            0.0
        },
        partial_seed_completion_rate: if is_partial && metrics.solved {
            1.0
        } else {
            0.0
        },
        random_growth_success_rate: if is_random && metrics.solved {
            1.0
        } else {
            0.0
        },
        positive_candidate_delta_fraction: stats.positive_candidate_delta_fraction,
        nonzero_candidate_delta_fraction: stats.nonzero_candidate_delta_fraction,
        accepted_operator_rate: stats.accepted_operator_rate,
        accepted_candidates: stats.accepted_candidates,
        evaluated_candidates: stats.evaluated_candidates,
        edge_count: metrics.edge_count,
        threshold_sensitivity,
        channel_sensitivity,
        polarity_sensitivity,
        damage_level,
        forbidden_private_field_leak: metrics.forbidden_private_field_leak,
        elapsed_sec,
    }
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
        positive_candidate_delta_fraction: deltas.iter().filter(|d| **d > 1e-9).count() as f64 / n,
        nonzero_candidate_delta_fraction: deltas.iter().filter(|d| d.abs() > 1e-9).count() as f64
            / n,
        mean_delta,
        mean_abs_delta,
        candidate_delta_std: var.sqrt(),
        evaluated_candidates: deltas.len(),
        accepted_candidates,
        accepted_operator_rate: accepted_candidates as f64 / n,
    }
}

fn write_candidate_record(
    writer: &mut BufWriter<File>,
    job_id: &str,
    seed: u64,
    stage: &Stage,
    record: &CandidateTraceRecord,
) -> std::io::Result<()> {
    serde_json::to_writer(
        &mut *writer,
        &json!({
            "job_id": job_id,
            "seed": seed,
            "stage": stage.name(),
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

fn argmax(values: &[f64; K]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn normalize_scores(scores: [f64; K]) -> [f64; K] {
    let total = scores.iter().sum::<f64>();
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

fn stage_seed_salt(stage: &Stage) -> u64 {
    let mut h = 0x9E37_79B9_7F4A_7C15u64;
    for b in stage.name().as_bytes() {
        h ^= *b as u64;
        h = h.rotate_left(7).wrapping_mul(0xA24B_AED4_963E_E407);
    }
    h
}

fn write_static_run_files(cfg: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let queue: Vec<_> = cfg
        .seeds
        .iter()
        .flat_map(|seed| {
            stages(cfg).into_iter().map(move |stage| {
                json!({
                    "seed": seed,
                    "stage": stage.name(),
                    "steps": cfg.steps,
                    "jackpot": cfg.jackpot,
                })
            })
        })
        .collect();
    write_json(cfg.out.join("queue.json"), &queue)?;
    fs::write(
        cfg.out.join("contract_snapshot.md"),
        "# STABLE_LOOP_PHASE_LOCK_008_PHASE_LANE_MICROCIRCUIT\n\nRunner-local snapshot: single-cell phase_i + gate_g -> phase_(i+g) microcircuit probe.\n",
    )?;
    for file in [
        "progress.jsonl",
        "metrics.jsonl",
        "candidate_log.jsonl",
        "motif_sensitivity.jsonl",
        "damage_level_success_curve.jsonl",
        "examples_sample.jsonl",
    ] {
        let path = cfg.out.join(file);
        if !path.exists() {
            File::create(path)?;
        }
    }
    for case in all_cases() {
        append_jsonl(
            cfg.out.join("examples_sample.jsonl"),
            &json!({"public": case.public, "private_label_only": case.private}),
        )?;
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
        "# STABLE_LOOP_PHASE_LOCK_008_PHASE_LANE_MICROCIRCUIT\n\nStatus: running. See JSONL files for partial outcomes.\n",
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
    write_json(
        cfg.out.join("summary.json"),
        &json!({
            "status": if completed == total { "complete" } else { "running" },
            "completed": completed,
            "total": total,
            "elapsed_sec": elapsed_sec,
            "verdicts": verdicts(rows),
            "stage_summary": summarize_rows(rows),
            "updated_time": now_sec(),
        }),
    )?;
    write_report(cfg, rows, completed == total)?;
    Ok(())
}

fn summarize_rows(rows: &[MetricRow]) -> BTreeMap<String, serde_json::Value> {
    let mut grouped: BTreeMap<String, Vec<&MetricRow>> = BTreeMap::new();
    for row in rows.iter().filter(|r| r.final_row) {
        grouped.entry(row.stage.clone()).or_default().push(row);
    }
    grouped
        .into_iter()
        .map(|(stage, stage_rows)| {
            let n = stage_rows.len().max(1) as f64;
            let acc = stage_rows
                .iter()
                .map(|r| r.all_16_phase_gate_pairs_accuracy)
                .sum::<f64>()
                / n;
            let repair = stage_rows
                .iter()
                .map(|r| r.motif_repair_success_rate)
                .sum::<f64>()
                / n;
            (
                stage,
                json!({
                    "all_16_phase_gate_pairs_accuracy": acc,
                    "motif_repair_success_rate": repair,
                    "seeds": stage_rows.len(),
                }),
            )
        })
        .collect()
}

fn verdicts(rows: &[MetricRow]) -> Vec<String> {
    let finals: Vec<&MetricRow> = rows.iter().filter(|r| r.final_row).collect();
    let mut out = Vec::new();
    if finals.iter().any(|r| {
        r.stage == "HAND_BUILT_PHASE_LANE_MOTIF" && r.all_16_phase_gate_pairs_accuracy >= 1.0
    }) {
        out.push("PHASE_LANE_MOTIF_REPRESENTABLE".to_string());
    } else if finals
        .iter()
        .any(|r| r.stage == "HAND_BUILT_PHASE_LANE_MOTIF")
    {
        out.push("REPRESENTATION_INSUFFICIENT".to_string());
    }
    let repair_rows: Vec<_> = finals
        .iter()
        .filter(|r| r.stage.starts_with("DAMAGE_REPAIR_MOTIF"))
        .collect();
    if !repair_rows.is_empty() {
        let repair_rate = repair_rows
            .iter()
            .map(|r| r.motif_repair_success_rate)
            .sum::<f64>()
            / repair_rows.len() as f64;
        if repair_rate > 0.5 {
            out.push("MOTIF_REPAIRABLE_BY_CANONICAL_MUTATION".to_string());
        } else {
            out.push("MOTIF_NOT_REPAIRABLE_BY_CANONICAL_MUTATION".to_string());
        }
    }
    let partial_rows: Vec<_> = finals
        .iter()
        .filter(|r| r.stage.starts_with("PARTIAL_SEED_COMPLETION"))
        .collect();
    if !partial_rows.is_empty()
        && partial_rows
            .iter()
            .any(|r| r.partial_seed_completion_rate >= 1.0)
    {
        out.push("PARTIAL_SEED_REQUIRED".to_string());
    }
    let random_rows: Vec<_> = finals
        .iter()
        .filter(|r| r.stage == "RANDOM_GROWTH_BASELINE")
        .collect();
    if !random_rows.is_empty() {
        if random_rows
            .iter()
            .any(|r| r.random_growth_success_rate >= 1.0)
        {
            out.push("RANDOM_GROWTH_SUCCEEDS".to_string());
        } else {
            out.push("MOTIF_NOT_GROWABLE_FROM_RANDOM".to_string());
        }
    }
    if finals.iter().any(|r| {
        r.stage.starts_with("EXPLICIT_COINCIDENCE_OPERATOR")
            && r.all_16_phase_gate_pairs_accuracy >= 1.0
    }) {
        out.push("EXPLICIT_COINCIDENCE_OPERATOR_REQUIRED".to_string());
    }
    let polarity_rows: Vec<_> = finals
        .iter()
        .filter(|r| r.stage.starts_with("DAMAGE_REPAIR_MOTIF_polarity_flipped"))
        .collect();
    if !polarity_rows.is_empty()
        && polarity_rows
            .iter()
            .all(|r| r.motif_repair_success_rate <= 0.0)
    {
        out.push("POLARITY_MUTATION_REQUIRED".to_string());
    }
    out
}

fn write_report(
    cfg: &Config,
    rows: &[MetricRow],
    complete: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut report = String::new();
    report.push_str("# STABLE_LOOP_PHASE_LOCK_008_PHASE_LANE_MICROCIRCUIT Report\n\n");
    report.push_str(&format!(
        "Status: {}.\n\n",
        if complete { "complete" } else { "running" }
    ));
    report.push_str("## Verdicts\n\n");
    for verdict in verdicts(rows) {
        report.push_str(&format!("- `{verdict}`\n"));
    }
    report.push_str("\n## Final Rows\n\n");
    report.push_str("| Stage | Seed | Acc | Prob | Positive delta | Accepted | Edges |\n");
    report.push_str("|---|---:|---:|---:|---:|---:|---:|\n");
    for row in rows.iter().filter(|r| r.final_row) {
        report.push_str(&format!(
            "| {} | {} | {:.1}% | {:.1}% | {:.1}% | {} | {} |\n",
            row.stage,
            row.seed,
            row.all_16_phase_gate_pairs_accuracy * 100.0,
            row.correct_phase_probability_mean * 100.0,
            row.positive_candidate_delta_fraction * 100.0,
            row.accepted_candidates,
            row.edge_count,
        ));
    }
    report.push_str("\n## Claim Boundary\n\n");
    report.push_str("This probe tests one local `phase_i + gate_g -> phase_(i+g)` motif in `instnct-core`. It does not prove full spatial phase-lock, full VRAXION, consciousness, language grounding, or Prismion uniqueness.\n");
    fs::write(cfg.out.join("report.md"), report)?;
    Ok(())
}

fn write_operator_summary(cfg: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let text = fs::read_to_string(cfg.out.join("candidate_log.jsonl")).unwrap_or_default();
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
    let out: BTreeMap<_, _> = summary
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
    write_json(cfg.out.join("operator_summary.json"), &out)?;
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
