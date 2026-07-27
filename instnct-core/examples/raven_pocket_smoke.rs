//! Small Raven-style pocket-routing smoke for the official VRAXION engine.
//!
//! This is intentionally simple and artifact-heavy:
//! - the model sees a 3x3 symbol board with one missing cell,
//! - it sees nine shuffled pockets, each holding one symbol,
//! - it must point at the pocket that holds the missing symbol.
//!
//! The run writes progress and raw row predictions continuously. It is a smoke
//! probe for the real graph/evolution/crystallize stack, not a new model.

use instnct_core::{
    build_network, evolution_step_jackpot, evolution_step_jackpot_traced, save_checkpoint,
    CandidateTraceRecord, CheckpointMeta, InitConfig, Int8Projection, MutationUndo, Network,
    MUTATION_OPERATORS,
};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use serde_json::json;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const SYMBOLS: [char; 9] = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'];
const FAMILY_COUNT: usize = 4;
const POCKET_CLASSES: usize = 9;

const GRID_BASE: usize = 0; // 9 cells * 9 symbols = 81
const MISSING_BASE: usize = 81; // 9 cells = 9
const POCKET_BASE: usize = 90; // 9 pockets * 9 symbols = 81
const FAMILY_BASE: usize = 171; // 4 families
const TARGET_BASE: usize = FAMILY_BASE + FAMILY_COUNT; // optional target hint, 9 symbols
const INPUT_FEATURES_USED: usize = TARGET_BASE + 9;
const INPUT_STRENGTH: i32 = 7;

#[derive(Clone, Debug)]
struct Config {
    out: PathBuf,
    seed: u64,
    train_rows: usize,
    test_rows: usize,
    steps: usize,
    eval_every: usize,
    candidates: usize,
    crystallize_samples: usize,
    heartbeat_sec: u64,
    h: usize,
    task_mode: String,
    compass_candidates: usize,
    heatmap_candidates: usize,
    heatmap_clone_steps: usize,
    mask_rounds: usize,
    mask_candidates: usize,
    mask_fraction: f64,
    ga_population: usize,
    ga_generations: usize,
    ga_elite_fraction: f64,
    ga_mutation_steps: usize,
    ga_validation_rows: usize,
    ga_selection_mode: String,
    ga_batch_rows: usize,
    ga_validation_batch_rows: usize,
    genome_population: usize,
    genome_generations: usize,
    genome_len: usize,
    genome_edges_per_neuron: usize,
    genome_mutation_bytes: usize,
    genome_batch_rows: usize,
    genome_mode: String,
    genome_random_fraction: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            out: PathBuf::from("target/codex_smoke/raven_pocket_smoke"),
            seed: 7001,
            train_rows: 240,
            test_rows: 120,
            steps: 60,
            eval_every: 10,
            candidates: 9,
            crystallize_samples: 20,
            heartbeat_sec: 20,
            h: 256,
            task_mode: "full".to_string(),
            compass_candidates: 0,
            heatmap_candidates: 0,
            heatmap_clone_steps: 1,
            mask_rounds: 0,
            mask_candidates: 0,
            mask_fraction: 0.10,
            ga_population: 0,
            ga_generations: 0,
            ga_elite_fraction: 0.20,
            ga_mutation_steps: 8,
            ga_validation_rows: 0,
            ga_selection_mode: "train_only".to_string(),
            ga_batch_rows: 0,
            ga_validation_batch_rows: 0,
            genome_population: 0,
            genome_generations: 0,
            genome_len: 64,
            genome_edges_per_neuron: 12,
            genome_mutation_bytes: 2,
            genome_batch_rows: 0,
            genome_mode: "blind".to_string(),
            genome_random_fraction: 0.25,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct Sample {
    row_id: String,
    split: String,
    family: String,
    family_id: usize,
    grid: Vec<Option<usize>>,
    pockets: Vec<usize>,
    expected_symbol: usize,
    expected_pocket: usize,
    target_hint: Option<usize>,
    prompt_text: String,
}

#[derive(Clone, Debug, Serialize)]
struct EvalSummary {
    split: String,
    rows: usize,
    accuracy: f64,
    avg_margin: f64,
    family_accuracy: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, Serialize)]
struct RowPrediction {
    row_id: String,
    split: String,
    family: String,
    prompt_text: String,
    expected_symbol: String,
    expected_pocket: String,
    selected_pocket: String,
    selected_symbol_in_pocket: String,
    correct: bool,
    margin: f64,
    scores: Vec<i32>,
}

#[derive(Clone, Debug, Serialize)]
struct CompassCandidate {
    candidate_id: usize,
    operator_id: String,
    mutated: bool,
    evaluated: bool,
    before_u: f64,
    after_u: f64,
    delta_u: f64,
    within_cap: bool,
    selected: bool,
    accepted: bool,
    candidate_eval_ms: f64,
    step_wall_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
struct HeatmapCandidate {
    candidate_id: usize,
    operator_id: String,
    mutated: bool,
    before_u: f64,
    after_u: f64,
    delta_u: f64,
    bucket: String,
    action_summary: String,
    details: serde_json::Value,
}

#[derive(Clone, Debug, Serialize)]
struct MaskCandidate {
    round: usize,
    candidate_id: usize,
    source: u16,
    target: u16,
    train_delta_u: f64,
    selected_for_mask: bool,
    detail: serde_json::Value,
}

#[derive(Clone)]
struct GaIndividual {
    id: usize,
    parent_id: Option<usize>,
    net: Network,
    proj: Int8Projection,
}

struct GaScoredIndividual {
    individual: GaIndividual,
    train_fitness: f64,
    validation_fitness: f64,
    validation_accuracy: f64,
    guard_passed: bool,
}

#[derive(Clone, Debug, Serialize)]
struct GaGenerationMetric {
    generation: usize,
    best_id: usize,
    best_parent_id: Option<usize>,
    best_train_fitness: f64,
    best_train_accuracy: f64,
    best_validation_fitness: f64,
    best_validation_accuracy: f64,
    best_test_accuracy: f64,
    best_train_margin: f64,
    best_validation_margin: f64,
    best_test_margin: f64,
    mean_train_fitness: f64,
    mean_validation_fitness: f64,
    validation_guard_floor: f64,
    validation_guard_pass_count: usize,
    validation_guard_fallback_fill_count: usize,
    train_pool_rows: usize,
    train_batch_rows: usize,
    validation_batch_rows: usize,
    train_batch_start: usize,
    validation_batch_start: usize,
    elite_count: usize,
    population_size: usize,
    best_edge_count: usize,
}

#[derive(Clone)]
struct GenomeIndividual {
    id: usize,
    parent_id: Option<usize>,
    genome: Vec<u8>,
}

struct MetaDnsProposer {
    byte_scores: Vec<[f64; 256]>,
    byte_counts: Vec<[u32; 256]>,
}

impl MetaDnsProposer {
    fn new(genome_len: usize) -> Self {
        Self {
            byte_scores: vec![[0.0_f64; 256]; genome_len],
            byte_counts: vec![[0_u32; 256]; genome_len],
        }
    }

    fn observe_scored(&mut self, scored: &[(f64, GenomeIndividual)], window: usize) {
        if scored.is_empty() {
            return;
        }
        let window = window.max(1).min(scored.len());
        let best = scored.first().map(|(fitness, _)| *fitness).unwrap_or(0.0);
        let worst = scored.last().map(|(fitness, _)| *fitness).unwrap_or(best);
        let denom = (best - worst).abs().max(1.0e-9);
        let mean = scored.iter().map(|(fitness, _)| *fitness).sum::<f64>() / scored.len() as f64;

        for (rank, (fitness, individual)) in scored.iter().enumerate() {
            let signed = if rank < window {
                ((*fitness - mean) / denom).max(0.0)
            } else if rank >= scored.len().saturating_sub(window) {
                -((mean - *fitness) / denom).max(0.0)
            } else {
                continue;
            };
            if signed == 0.0 {
                continue;
            }
            for (pos, byte) in individual.genome.iter().enumerate() {
                if pos >= self.byte_scores.len() {
                    break;
                }
                let idx = *byte as usize;
                self.byte_scores[pos][idx] += signed;
                self.byte_counts[pos][idx] = self.byte_counts[pos][idx].saturating_add(1);
            }
        }
    }

    fn propose_child(&self, parent: &[u8], mutation_bytes: usize, rng: &mut StdRng) -> Vec<u8> {
        let mut child = parent.to_vec();
        for (pos, byte) in child.iter_mut().enumerate() {
            if rng.gen_bool(0.35) {
                if let Some(best_byte) = self.best_byte_for_position(pos) {
                    *byte = best_byte;
                }
            }
        }
        mutate_genome(&mut child, mutation_bytes.max(1), rng);
        child
    }

    fn best_byte_for_position(&self, pos: usize) -> Option<u8> {
        let scores = self.byte_scores.get(pos)?;
        let counts = self.byte_counts.get(pos)?;
        let mut best_idx = None;
        let mut best_score = f64::NEG_INFINITY;
        for idx in 0..256 {
            if counts[idx] == 0 {
                continue;
            }
            let score = scores[idx];
            if score > best_score {
                best_score = score;
                best_idx = Some(idx as u8);
            }
        }
        best_idx
    }

    fn stats(&self) -> serde_json::Value {
        let mut observed_cells = 0usize;
        let mut positive_cells = 0usize;
        let mut negative_cells = 0usize;
        let mut positions_with_signal = 0usize;
        let mut best_score_sum = 0.0;
        for pos in 0..self.byte_scores.len() {
            let mut saw_signal = false;
            let mut best_score = f64::NEG_INFINITY;
            for idx in 0..256 {
                if self.byte_counts[pos][idx] == 0 {
                    continue;
                }
                observed_cells += 1;
                let score = self.byte_scores[pos][idx];
                if score > 0.0 {
                    positive_cells += 1;
                    saw_signal = true;
                } else if score < 0.0 {
                    negative_cells += 1;
                }
                if score > best_score {
                    best_score = score;
                }
            }
            if saw_signal {
                positions_with_signal += 1;
                best_score_sum += best_score;
            }
        }
        json!({
            "observed_byte_cells": observed_cells,
            "positive_byte_cells": positive_cells,
            "negative_byte_cells": negative_cells,
            "positions_with_positive_signal": positions_with_signal,
            "mean_best_positive_position_score": if positions_with_signal == 0 {
                0.0
            } else {
                best_score_sum / positions_with_signal as f64
            },
            "meaning": "The meta-DNS proposer learns which byte values appeared in better or worse genome codes; it is not end-to-end differentiable VRAXION."
        })
    }
}

fn main() -> io::Result<()> {
    let cfg = parse_args()?;
    fs::create_dir_all(&cfg.out)?;
    let progress_path = cfg.out.join("progress.jsonl");
    let started = Instant::now();

    write_json(
        &cfg.out.join("queue.json"),
        &json!({
            "status": "running",
            "task": "official_vraxion_raven_pocket_smoke",
            "seed": cfg.seed,
            "train_rows": cfg.train_rows,
            "test_rows": cfg.test_rows,
            "steps": cfg.steps,
            "eval_every": cfg.eval_every,
            "candidates": cfg.candidates,
            "crystallize_samples": cfg.crystallize_samples,
            "heartbeat_sec": cfg.heartbeat_sec,
            "h": cfg.h,
            "task_mode": cfg.task_mode,
            "compass_candidates": cfg.compass_candidates,
            "heatmap_candidates": cfg.heatmap_candidates,
            "heatmap_clone_steps": cfg.heatmap_clone_steps,
            "mask_rounds": cfg.mask_rounds,
            "mask_candidates": cfg.mask_candidates,
            "mask_fraction": cfg.mask_fraction,
            "ga_population": cfg.ga_population,
            "ga_generations": cfg.ga_generations,
            "ga_elite_fraction": cfg.ga_elite_fraction,
            "ga_mutation_steps": cfg.ga_mutation_steps,
            "ga_validation_rows": cfg.ga_validation_rows,
            "ga_selection_mode": cfg.ga_selection_mode,
            "ga_batch_rows": cfg.ga_batch_rows,
            "ga_validation_batch_rows": cfg.ga_validation_batch_rows,
            "genome_population": cfg.genome_population,
            "genome_generations": cfg.genome_generations,
            "genome_len": cfg.genome_len,
            "genome_edges_per_neuron": cfg.genome_edges_per_neuron,
            "genome_mutation_bytes": cfg.genome_mutation_bytes,
            "genome_batch_rows": cfg.genome_batch_rows,
            "genome_mode": cfg.genome_mode,
            "genome_random_fraction": cfg.genome_random_fraction,
            "input_features_used": INPUT_FEATURES_USED,
            "random_baseline": 1.0 / POCKET_CLASSES as f64,
            "no_black_box": true
        }),
    )?;
    append_progress(
        &progress_path,
        "queue_written",
        started.elapsed().as_secs_f64(),
        json!({}),
    )?;

    let mut data_rng = StdRng::seed_from_u64(cfg.seed ^ 0xDADA_1001);
    let train = make_samples("train", cfg.train_rows, &mut data_rng, &cfg.task_mode);
    let test = make_samples("test", cfg.test_rows, &mut data_rng, &cfg.task_mode);
    write_jsonl(&cfg.out.join("curriculum_train.jsonl"), &train)?;
    write_jsonl(&cfg.out.join("curriculum_test.jsonl"), &test)?;
    append_progress(
        &progress_path,
        "curriculum_written",
        started.elapsed().as_secs_f64(),
        json!({"train_rows": train.len(), "test_rows": test.len()}),
    )?;

    let mut init = InitConfig::phi(cfg.h);
    init.accept_ties = false;
    let mut init_rng = StdRng::seed_from_u64(cfg.seed ^ 0x1234_5678);
    let mut net = build_network(&init, &mut init_rng);
    let mut proj = Int8Projection::new(init.phi_dim, POCKET_CLASSES, &mut init_rng);
    let evo_cfg = init.evolution_config();
    let mut mut_rng = StdRng::seed_from_u64(cfg.seed ^ 0xBEEF_0001);
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed ^ 0xBEEF_0002);

    let before_train = eval_dataset("train", &mut net, &proj, &train, &init, None)?;
    let before_test = eval_dataset("test", &mut net, &proj, &test, &init, None)?;
    append_progress(
        &progress_path,
        "initial_eval",
        started.elapsed().as_secs_f64(),
        json!({
            "train_accuracy": before_train.accuracy,
            "test_accuracy": before_test.accuracy,
            "edge_count": net.edge_count()
        }),
    )?;

    if cfg.genome_population > 0 || cfg.genome_generations > 0 {
        run_hash_genome_experiment(
            &cfg,
            &progress_path,
            started,
            &train,
            &test,
            &init,
            &mut init_rng,
            &mut mut_rng,
            &before_train,
            &before_test,
        )?;
        return Ok(());
    }

    if cfg.ga_population > 0 || cfg.ga_generations > 0 {
        run_persistent_ga(
            &cfg,
            &progress_path,
            started,
            &train,
            &test,
            &init,
            &mut init_rng,
            &mut mut_rng,
            &before_train,
            &before_test,
        )?;
        return Ok(());
    }

    if cfg.mask_rounds > 0 {
        run_mask_probe(
            &cfg,
            &progress_path,
            started,
            &train,
            &test,
            &init,
            &mut net,
            &proj,
            &mut mut_rng,
            &before_train,
            &before_test,
        )?;
        return Ok(());
    }

    if cfg.heatmap_candidates > 0 {
        run_heatmap(
            &cfg,
            &progress_path,
            started,
            &train,
            &test,
            &init,
            &mut net,
            &mut proj,
            &mut mut_rng,
            &before_train,
            &before_test,
        )?;
        return Ok(());
    }

    if cfg.compass_candidates > 0 {
        run_compass(
            &cfg,
            &progress_path,
            started,
            &train,
            &test,
            &init,
            &evo_cfg,
            &mut net,
            &mut proj,
            &mut mut_rng,
            &mut eval_rng,
            &before_train,
            &before_test,
        )?;
        return Ok(());
    }

    let mut metrics_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(cfg.out.join("training_metrics.jsonl"))?;
    let mut last_heartbeat = Instant::now();

    for step in 1..=cfg.steps {
        let outcome = evolution_step_jackpot(
            &mut net,
            &mut proj,
            &mut mut_rng,
            &mut eval_rng,
            |candidate_net, candidate_proj, _rng| {
                smooth_fitness(candidate_net, candidate_proj, &train, &init)
            },
            &evo_cfg,
            cfg.candidates,
        );

        if step % cfg.eval_every == 0
            || step == cfg.steps
            || last_heartbeat.elapsed().as_secs() >= cfg.heartbeat_sec
        {
            let train_eval = eval_dataset("train", &mut net, &proj, &train, &init, None)?;
            let test_eval = eval_dataset("test", &mut net, &proj, &test, &init, None)?;
            let metric = json!({
                "step": step,
                "elapsed_sec": started.elapsed().as_secs_f64(),
                "outcome": format!("{outcome:?}"),
                "train_accuracy": train_eval.accuracy,
                "test_accuracy": test_eval.accuracy,
                "train_avg_margin": train_eval.avg_margin,
                "test_avg_margin": test_eval.avg_margin,
                "edge_count": net.edge_count()
            });
            writeln!(metrics_file, "{}", serde_json::to_string(&metric)?)?;
            metrics_file.flush()?;
            append_progress(
                &progress_path,
                "training_eval",
                started.elapsed().as_secs_f64(),
                metric,
            )?;
            save_checkpoint(
                cfg.out.join("checkpoint_latest.ckpt"),
                &net,
                &proj,
                CheckpointMeta {
                    step,
                    accuracy: test_eval.accuracy,
                    label: format!("raven_pocket_smoke seed={} step={step}", cfg.seed),
                },
            )?;
            last_heartbeat = Instant::now();
        }
    }

    let crystallize_before = net.edge_count();
    let pruned = if cfg.crystallize_samples > 0 {
        append_progress(
            &progress_path,
            "crystallize_start",
            started.elapsed().as_secs_f64(),
            json!({"samples": cfg.crystallize_samples, "edges_before": crystallize_before}),
        )?;
        crystallize(
            &mut net,
            &proj,
            &train,
            &init,
            cfg.crystallize_samples,
            &mut mut_rng,
        )
    } else {
        0
    };
    append_progress(
        &progress_path,
        "crystallize_done",
        started.elapsed().as_secs_f64(),
        json!({"pruned": pruned, "edges_before": crystallize_before, "edges_after": net.edge_count()}),
    )?;

    let row_path = cfg.out.join("row_level_predictions.jsonl");
    let final_train = eval_dataset("train", &mut net, &proj, &train, &init, Some(&row_path))?;
    let final_test = eval_dataset("test", &mut net, &proj, &test, &init, Some(&row_path))?;
    let accepted = count_outcomes(&cfg.out.join("training_metrics.jsonl"), "Accepted")?;
    let rejected = count_outcomes(&cfg.out.join("training_metrics.jsonl"), "Rejected")?;
    let skipped = count_outcomes(&cfg.out.join("training_metrics.jsonl"), "Skipped")?;

    save_checkpoint(
        cfg.out.join("checkpoint_final.ckpt"),
        &net,
        &proj,
        CheckpointMeta {
            step: cfg.steps,
            accuracy: final_test.accuracy,
            label: format!("raven_pocket_smoke seed={} final", cfg.seed),
        },
    )?;

    let summary = json!({
        "status": "complete",
        "task": "official_vraxion_raven_pocket_smoke",
        "seed": cfg.seed,
        "task_mode": cfg.task_mode,
        "h": cfg.h,
        "phi_dim": init.phi_dim,
        "input_zone": {"start": 0, "end": init.input_end()},
        "output_zone": {"start": init.output_start(), "end": init.neuron_count},
        "overlap_zone": {"start": init.output_start(), "end": init.input_end()},
        "model": {
            "engine": "official_instnct_core_network",
            "projection": "Int8Projection",
            "mutation_schedule": "canonical weighted evolution operators",
            "crystallize_used": cfg.crystallize_samples > 0,
            "crystallize_pruned_edges": pruned
        },
        "train": final_train,
        "test": final_test,
        "random_baseline_accuracy": 1.0 / POCKET_CLASSES as f64,
        "accepted_steps_seen_in_metrics": accepted,
        "rejected_steps_seen_in_metrics": rejected,
        "skipped_steps_seen_in_metrics": skipped,
        "edge_count_final": net.edge_count(),
        "elapsed_sec": started.elapsed().as_secs_f64(),
        "raw_predictions_file": "row_level_predictions.jsonl"
    });
    write_json(&cfg.out.join("summary.json"), &summary)?;
    write_report(&cfg.out.join("report.md"), &summary)?;
    append_progress(
        &progress_path,
        "complete",
        started.elapsed().as_secs_f64(),
        summary,
    )?;
    Ok(())
}

fn parse_args() -> io::Result<Config> {
    let mut cfg = Config::default();
    let args: Vec<String> = env::args().collect();
    let mut i = 1usize;
    while i < args.len() {
        let key = &args[i];
        let value = if i + 1 < args.len() {
            Some(args[i + 1].clone())
        } else {
            None
        };
        match key.as_str() {
            "--out" => {
                cfg.out = PathBuf::from(value.ok_or_else(|| arg_err("--out needs a value"))?);
                i += 2;
            }
            "--seed" => {
                cfg.seed = parse_value("--seed", value)?;
                i += 2;
            }
            "--train-rows" => {
                cfg.train_rows = parse_value("--train-rows", value)?;
                i += 2;
            }
            "--test-rows" => {
                cfg.test_rows = parse_value("--test-rows", value)?;
                i += 2;
            }
            "--steps" => {
                cfg.steps = parse_value("--steps", value)?;
                i += 2;
            }
            "--eval-every" => {
                cfg.eval_every = parse_value("--eval-every", value)?;
                i += 2;
            }
            "--candidates" => {
                cfg.candidates = parse_value("--candidates", value)?;
                i += 2;
            }
            "--crystallize-samples" => {
                cfg.crystallize_samples = parse_value("--crystallize-samples", value)?;
                i += 2;
            }
            "--heartbeat-sec" => {
                cfg.heartbeat_sec = parse_value("--heartbeat-sec", value)?;
                i += 2;
            }
            "--h" => {
                cfg.h = parse_value("--h", value)?;
                i += 2;
            }
            "--task-mode" => {
                cfg.task_mode = value.ok_or_else(|| arg_err("--task-mode needs a value"))?;
                i += 2;
            }
            "--compass-candidates" => {
                cfg.compass_candidates = parse_value("--compass-candidates", value)?;
                i += 2;
            }
            "--heatmap-candidates" => {
                cfg.heatmap_candidates = parse_value("--heatmap-candidates", value)?;
                i += 2;
            }
            "--heatmap-clone-steps" => {
                cfg.heatmap_clone_steps = parse_value("--heatmap-clone-steps", value)?;
                i += 2;
            }
            "--mask-rounds" => {
                cfg.mask_rounds = parse_value("--mask-rounds", value)?;
                i += 2;
            }
            "--mask-candidates" => {
                cfg.mask_candidates = parse_value("--mask-candidates", value)?;
                i += 2;
            }
            "--mask-fraction" => {
                cfg.mask_fraction = parse_value("--mask-fraction", value)?;
                i += 2;
            }
            "--ga-population" => {
                cfg.ga_population = parse_value("--ga-population", value)?;
                i += 2;
            }
            "--ga-generations" => {
                cfg.ga_generations = parse_value("--ga-generations", value)?;
                i += 2;
            }
            "--ga-elite-fraction" => {
                cfg.ga_elite_fraction = parse_value("--ga-elite-fraction", value)?;
                i += 2;
            }
            "--ga-mutation-steps" => {
                cfg.ga_mutation_steps = parse_value("--ga-mutation-steps", value)?;
                i += 2;
            }
            "--ga-validation-rows" => {
                cfg.ga_validation_rows = parse_value("--ga-validation-rows", value)?;
                i += 2;
            }
            "--ga-selection-mode" => {
                cfg.ga_selection_mode =
                    value.ok_or_else(|| arg_err("--ga-selection-mode needs a value"))?;
                i += 2;
            }
            "--ga-batch-rows" => {
                cfg.ga_batch_rows = parse_value("--ga-batch-rows", value)?;
                i += 2;
            }
            "--ga-validation-batch-rows" => {
                cfg.ga_validation_batch_rows = parse_value("--ga-validation-batch-rows", value)?;
                i += 2;
            }
            "--genome-population" => {
                cfg.genome_population = parse_value("--genome-population", value)?;
                i += 2;
            }
            "--genome-generations" => {
                cfg.genome_generations = parse_value("--genome-generations", value)?;
                i += 2;
            }
            "--genome-len" => {
                cfg.genome_len = parse_value("--genome-len", value)?;
                i += 2;
            }
            "--genome-edges-per-neuron" => {
                cfg.genome_edges_per_neuron = parse_value("--genome-edges-per-neuron", value)?;
                i += 2;
            }
            "--genome-mutation-bytes" => {
                cfg.genome_mutation_bytes = parse_value("--genome-mutation-bytes", value)?;
                i += 2;
            }
            "--genome-batch-rows" => {
                cfg.genome_batch_rows = parse_value("--genome-batch-rows", value)?;
                i += 2;
            }
            "--genome-mode" => {
                cfg.genome_mode = value.ok_or_else(|| arg_err("--genome-mode needs a value"))?;
                i += 2;
            }
            "--genome-random-fraction" => {
                cfg.genome_random_fraction = parse_value("--genome-random-fraction", value)?;
                i += 2;
            }
            _ => return Err(arg_err(&format!("unknown argument: {key}"))),
        }
    }
    match cfg.task_mode.as_str() {
        "full" | "pocket_lookup" | "pocket_only_lookup" | "pocket_id_hint"
        | "pocket_id_only" | "pocket_match_hint" | "symbol_match_only" | "full_match_hint"
        | "pocket_id_grid_noise" | "pattern_fixed_pocket" => {}
        _ => return Err(arg_err(
                "--task-mode must be full, pocket_lookup, pocket_only_lookup, pocket_id_hint, pocket_id_only, pocket_match_hint, symbol_match_only, full_match_hint, pocket_id_grid_noise, or pattern_fixed_pocket",
        )),
    }
    if cfg.eval_every == 0 {
        cfg.eval_every = 1;
    }
    if cfg.candidates == 0 {
        cfg.candidates = 1;
    }
    if cfg.mask_fraction <= 0.0 {
        cfg.mask_fraction = 0.10;
    }
    if cfg.mask_fraction > 1.0 {
        cfg.mask_fraction = 1.0;
    }
    if cfg.ga_generations > 0 && cfg.ga_population == 0 {
        cfg.ga_population = 128;
    }
    if cfg.ga_population > 0 && cfg.ga_generations == 0 {
        cfg.ga_generations = 10;
    }
    if cfg.ga_elite_fraction <= 0.0 {
        cfg.ga_elite_fraction = 0.20;
    }
    if cfg.ga_elite_fraction > 1.0 {
        cfg.ga_elite_fraction = 1.0;
    }
    if cfg.ga_mutation_steps == 0 {
        cfg.ga_mutation_steps = 8;
    }
    if cfg.ga_batch_rows > cfg.train_rows {
        cfg.ga_batch_rows = cfg.train_rows;
    }
    if cfg.ga_validation_batch_rows > cfg.train_rows {
        cfg.ga_validation_batch_rows = cfg.train_rows;
    }
    match cfg.ga_selection_mode.as_str() {
        "train_only" | "validation_guard" => {}
        _ => {
            return Err(arg_err(
                "--ga-selection-mode must be train_only or validation_guard",
            ))
        }
    }
    if cfg.genome_generations > 0 && cfg.genome_population == 0 {
        cfg.genome_population = 128;
    }
    if cfg.genome_population > 0 && cfg.genome_generations == 0 {
        cfg.genome_generations = 10;
    }
    if cfg.genome_len == 0 {
        cfg.genome_len = 64;
    }
    if cfg.genome_edges_per_neuron == 0 {
        cfg.genome_edges_per_neuron = 12;
    }
    if cfg.genome_batch_rows > cfg.train_rows {
        cfg.genome_batch_rows = cfg.train_rows;
    }
    match cfg.genome_mode.as_str() {
        "blind" | "meta" | "string_rule" | "u64_barcode" | "u64_slot_barcode"
        | "u64_gate_sampled" | "rule_dna_gate" | "rule_dna_gate_mid"
        | "rule_dna_gate_strict" => {}
        _ => {
            return Err(arg_err(
                "--genome-mode must be blind, meta, string_rule, u64_barcode, u64_slot_barcode, u64_gate_sampled, rule_dna_gate, rule_dna_gate_mid, or rule_dna_gate_strict",
            ))
        }
    }
    if cfg.genome_random_fraction < 0.0 {
        cfg.genome_random_fraction = 0.0;
    }
    if cfg.genome_random_fraction > 1.0 {
        cfg.genome_random_fraction = 1.0;
    }
    Ok(cfg)
}

#[allow(clippy::too_many_arguments)]
fn run_hash_genome_experiment(
    cfg: &Config,
    progress_path: &Path,
    started: Instant,
    train: &[Sample],
    test: &[Sample],
    init: &InitConfig,
    init_rng: &mut StdRng,
    mut_rng: &mut StdRng,
    before_train: &EvalSummary,
    before_test: &EvalSummary,
) -> io::Result<()> {
    let population_size = cfg.genome_population.max(2);
    let elite_count = ((population_size as f64) * cfg.ga_elite_fraction)
        .ceil()
        .max(1.0) as usize;
    let elite_count = elite_count.min(population_size);
    let validation_rows = cfg.ga_validation_rows.min(train.len().saturating_sub(1));
    let (fitness_train, validation) = if validation_rows > 0 {
        train.split_at(train.len() - validation_rows)
    } else {
        (train, &train[0..0])
    };
    let batch_rows = if cfg.genome_batch_rows > 0 {
        cfg.genome_batch_rows.min(fitness_train.len())
    } else {
        fitness_train.len()
    };
    append_progress(
        progress_path,
        "hash_genome_start",
        started.elapsed().as_secs_f64(),
        json!({
            "population": population_size,
            "generations": cfg.genome_generations,
            "genome_len": cfg.genome_len,
            "edges_per_neuron": cfg.genome_edges_per_neuron,
            "mutation_bytes": cfg.genome_mutation_bytes,
            "batch_rows": batch_rows,
            "fitness_train_rows": fitness_train.len(),
            "validation_rows": validation.len(),
            "genome_mode": cfg.genome_mode,
            "genome_random_fraction": cfg.genome_random_fraction,
            "meaning": "Genome bytes build the network; selection is over genome codes, not direct edge mutations. In meta mode, a learned byte recommender proposes some children from prior scored genomes while random exploration remains active."
        }),
    )?;

    let mut next_id = 0usize;
    let mut population = Vec::with_capacity(population_size);
    for _ in 0..population_size {
        let mut genome = vec![0u8; cfg.genome_len];
        for byte in &mut genome {
            *byte = init_rng.gen_range(0..=255);
        }
        population.push(GenomeIndividual {
            id: next_id,
            parent_id: None,
            genome,
        });
        next_id += 1;
    }

    let mut generation_metrics = Vec::new();
    let mut meta_proposer = MetaDnsProposer::new(cfg.genome_len);
    let meta_enabled = cfg.genome_mode == "meta";
    let mut validation_lock_accuracy = f64::NEG_INFINITY;
    let mut validation_lock_individual: Option<GenomeIndividual> = None;
    let mut validation_lock_metric: Option<serde_json::Value> = None;
    for generation in 0..cfg.genome_generations {
        let batch_start = rotating_batch_start(fitness_train.len(), batch_rows, generation, 271);
        let train_batch = &fitness_train[batch_start..batch_start + batch_rows];
        let mut scored = Vec::with_capacity(population.len());
        let mut fitness_sum = 0.0;
        for individual in population {
            let mut net = network_from_genome_for_mode(
                &individual.genome,
                init,
                cfg.genome_edges_per_neuron,
                &cfg.genome_mode,
            );
            let proj = projection_from_genome(&individual.genome, init);
            let fitness = smooth_fitness(&mut net, &proj, train_batch, init);
            fitness_sum += fitness;
            scored.push((fitness, individual));
        }
        scored.sort_by(|a, b| cmp_f64(b.0, a.0));
        if meta_enabled {
            meta_proposer.observe_scored(&scored, elite_count.saturating_mul(2));
        }

        let best_fitness = scored[0].0;
        let best_individual = scored[0].1.clone();
        let mut best_net = network_from_genome_for_mode(
            &best_individual.genome,
            init,
            cfg.genome_edges_per_neuron,
            &cfg.genome_mode,
        );
        let best_proj = projection_from_genome(&best_individual.genome, init);
        let best_train = eval_dataset(
            "train_batch",
            &mut best_net,
            &best_proj,
            train_batch,
            init,
            None,
        )?;
        let best_validation = if validation.is_empty() {
            None
        } else {
            Some(eval_dataset(
                "validation",
                &mut best_net,
                &best_proj,
                validation,
                init,
                None,
            )?)
        };
        let best_test = eval_dataset("test", &mut best_net, &best_proj, test, init, None)?;
        let best_validation_accuracy = best_validation
            .as_ref()
            .map(|summary| summary.accuracy)
            .unwrap_or(0.0);
        if let Some(validation_summary) = &best_validation {
            if validation_summary.accuracy > validation_lock_accuracy {
                validation_lock_accuracy = validation_summary.accuracy;
                validation_lock_individual = Some(best_individual.clone());
                validation_lock_metric = Some(json!({
                    "generation": generation,
                    "best_id": best_individual.id,
                    "best_parent_id": best_individual.parent_id,
                    "validation_accuracy": validation_summary.accuracy,
                    "validation_avg_margin": validation_summary.avg_margin,
                    "train_batch_accuracy": best_train.accuracy,
                    "test_accuracy_diagnostic_only": best_test.accuracy
                }));
            }
        }
        let metric = json!({
            "generation": generation,
            "best_id": best_individual.id,
            "best_parent_id": best_individual.parent_id,
            "best_fitness": best_fitness,
            "best_train_accuracy": best_train.accuracy,
            "best_validation_accuracy": best_validation_accuracy,
            "best_test_accuracy": best_test.accuracy,
            "best_train_margin": best_train.avg_margin,
            "best_validation_margin": best_validation
                .as_ref()
                .map(|summary| summary.avg_margin)
                .unwrap_or(0.0),
            "best_test_margin": best_test.avg_margin,
            "mean_fitness": fitness_sum / scored.len().max(1) as f64,
            "batch_start": batch_start,
            "batch_rows": train_batch.len(),
            "edge_count": best_net.edge_count(),
            "elite_count": elite_count,
            "population_size": scored.len(),
            "genome_mode": cfg.genome_mode,
            "genome_random_fraction": cfg.genome_random_fraction
        });
        append_jsonl_value(
            &cfg.out.join("hash_genome_generation_metrics.jsonl"),
            &metric,
        )?;
        if meta_enabled {
            let meta_metric = json!({
                "generation": generation,
                "best_fitness": best_fitness,
                "best_test_accuracy": best_test.accuracy,
                "random_child_fraction": cfg.genome_random_fraction,
                "meta_child_fraction": 1.0 - cfg.genome_random_fraction,
                "meta_model": meta_proposer.stats()
            });
            append_jsonl_value(
                &cfg.out.join("meta_dns_generation_metrics.jsonl"),
                &meta_metric,
            )?;
        }
        append_progress(
            progress_path,
            "hash_genome_generation_complete",
            started.elapsed().as_secs_f64(),
            metric.clone(),
        )?;
        generation_metrics.push(metric);

        let elites: Vec<GenomeIndividual> = scored
            .iter()
            .take(elite_count)
            .map(|(_, individual)| individual.clone())
            .collect();
        let mut new_population = elites.clone();
        while new_population.len() < population_size {
            let parent = elites[new_population.len() % elites.len()].clone();
            let mut child = parent.clone();
            child.id = next_id;
            child.parent_id = Some(parent.id);
            next_id += 1;
            if meta_enabled && !mut_rng.gen_bool(cfg.genome_random_fraction) {
                child.genome = meta_proposer.propose_child(
                    &parent.genome,
                    cfg.genome_mutation_bytes.max(1),
                    mut_rng,
                );
            } else {
                mutate_genome(&mut child.genome, cfg.genome_mutation_bytes.max(1), mut_rng);
            }
            new_population.push(child);
        }
        population = new_population;
    }

    let mut final_scored = Vec::with_capacity(population.len());
    for individual in population {
        let mut net = network_from_genome_for_mode(
            &individual.genome,
            init,
            cfg.genome_edges_per_neuron,
            &cfg.genome_mode,
        );
        let proj = projection_from_genome(&individual.genome, init);
        let fitness = smooth_fitness(&mut net, &proj, fitness_train, init);
        final_scored.push((fitness, individual));
    }
    final_scored.sort_by(|a, b| cmp_f64(b.0, a.0));
    let best_fitness = final_scored[0].0;
    let best_individual = final_scored[0].1.clone();
    let mut best_net = network_from_genome_for_mode(
        &best_individual.genome,
        init,
        cfg.genome_edges_per_neuron,
        &cfg.genome_mode,
    );
    let best_proj = projection_from_genome(&best_individual.genome, init);
    let final_train = eval_dataset("train", &mut best_net, &best_proj, train, init, None)?;
    let final_test = eval_dataset(
        "test",
        &mut best_net,
        &best_proj,
        test,
        init,
        Some(&cfg.out.join("hash_genome_best_row_level_predictions.jsonl")),
    )?;
    let validation_locked_best = if let Some(individual) = validation_lock_individual.clone() {
        let mut locked_net = network_from_genome_for_mode(
            &individual.genome,
            init,
            cfg.genome_edges_per_neuron,
            &cfg.genome_mode,
        );
        let locked_proj = projection_from_genome(&individual.genome, init);
        let locked_train = eval_dataset("train", &mut locked_net, &locked_proj, train, init, None)?;
        let locked_validation = if validation.is_empty() {
            None
        } else {
            Some(eval_dataset(
                "validation",
                &mut locked_net,
                &locked_proj,
                validation,
                init,
                None,
            )?)
        };
        let locked_test = eval_dataset(
            "test",
            &mut locked_net,
            &locked_proj,
            test,
            init,
            Some(
                &cfg.out
                    .join("hash_genome_validation_locked_row_level_predictions.jsonl"),
            ),
        )?;
        json!({
            "enabled": true,
            "id": individual.id,
            "parent_id": individual.parent_id,
            "lock_metric": validation_lock_metric,
            "train_accuracy": locked_train.accuracy,
            "validation_accuracy": locked_validation
                .as_ref()
                .map(|summary| summary.accuracy)
                .unwrap_or(0.0),
            "test_accuracy": locked_test.accuracy,
            "train_avg_margin": locked_train.avg_margin,
            "validation_avg_margin": locked_validation
                .as_ref()
                .map(|summary| summary.avg_margin)
                .unwrap_or(0.0),
            "test_avg_margin": locked_test.avg_margin,
            "edge_count": locked_net.edge_count()
        })
    } else {
        json!({"enabled": false})
    };
    let best_generation = generation_metrics
        .iter()
        .max_by(|a, b| {
            cmp_f64(
                a["best_test_accuracy"].as_f64().unwrap_or(0.0),
                b["best_test_accuracy"].as_f64().unwrap_or(0.0),
            )
        })
        .cloned();
    let summary = json!({
        "status": "complete",
        "task": "hash_genome_network_builder_smoke",
        "seed": cfg.seed,
        "population": population_size,
        "generations": cfg.genome_generations,
        "genome_len": cfg.genome_len,
        "edges_per_neuron": cfg.genome_edges_per_neuron,
        "mutation_bytes": cfg.genome_mutation_bytes,
        "batch_rows": batch_rows,
        "fitness_train_rows": fitness_train.len(),
        "validation_rows": validation.len(),
        "genome_mode": cfg.genome_mode,
        "genome_random_fraction": cfg.genome_random_fraction,
        "meta_dns_proposer": if meta_enabled {
            meta_proposer.stats()
        } else {
            json!({"enabled": false})
        },
        "string_rule_dns": if cfg.genome_mode == "string_rule" {
            string_rule_dns_report(&best_individual.genome, cfg.genome_edges_per_neuron)
        } else if cfg.genome_mode == "u64_barcode"
            || cfg.genome_mode == "u64_slot_barcode"
            || cfg.genome_mode == "u64_gate_sampled"
        {
            u64_barcode_dns_report(
                &best_individual.genome,
                cfg.genome_edges_per_neuron,
                &cfg.genome_mode,
            )
        } else if cfg.genome_mode == "rule_dna_gate"
            || cfg.genome_mode == "rule_dna_gate_mid"
            || cfg.genome_mode == "rule_dna_gate_strict"
        {
            rule_dna_gate_report(
                &best_individual.genome,
                cfg.genome_edges_per_neuron,
                if cfg.genome_mode == "rule_dna_gate_mid" {
                    1
                } else if cfg.genome_mode == "rule_dna_gate_strict" {
                    2
                } else {
                    0
                },
            )
        } else {
            json!({"enabled": false})
        },
        "before_single_baseline": {
            "train_accuracy": before_train.accuracy,
            "test_accuracy": before_test.accuracy,
            "train_avg_margin": before_train.avg_margin,
            "test_avg_margin": before_test.avg_margin
        },
        "final_best": {
            "id": best_individual.id,
            "parent_id": best_individual.parent_id,
            "fitness": best_fitness,
            "train_accuracy": final_train.accuracy,
            "test_accuracy": final_test.accuracy,
            "train_avg_margin": final_train.avg_margin,
            "test_avg_margin": final_test.avg_margin,
            "edge_count": best_net.edge_count()
        },
        "validation_locked_best": validation_locked_best,
        "best_generation_by_test_accuracy": best_generation,
        "test_accuracy_delta_from_single_baseline": final_test.accuracy - before_test.accuracy,
        "interpretation_limit": if meta_enabled {
            "This tests a learned DNS-byte proposer that guides genome search from prior scored DNS codes. It is not end-to-end differentiable VRAXION."
        } else {
            "This tests a deterministic hash-genome builder, not a learned differentiable meta-network."
        }
    });
    write_json(&cfg.out.join("hash_genome_summary.json"), &summary)?;
    append_progress(
        progress_path,
        "hash_genome_complete",
        started.elapsed().as_secs_f64(),
        summary,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_persistent_ga(
    cfg: &Config,
    progress_path: &Path,
    started: Instant,
    train: &[Sample],
    test: &[Sample],
    init: &InitConfig,
    init_rng: &mut StdRng,
    mut_rng: &mut StdRng,
    before_train: &EvalSummary,
    before_test: &EvalSummary,
) -> io::Result<()> {
    let population_size = cfg.ga_population.max(2);
    let elite_count = ((population_size as f64) * cfg.ga_elite_fraction)
        .ceil()
        .max(1.0) as usize;
    let elite_count = elite_count.min(population_size);
    let validation_rows = if cfg.ga_validation_rows > 0 && train.len() > 1 {
        cfg.ga_validation_rows.min(train.len() - 1)
    } else {
        0
    };
    let train_fit_rows = train.len().saturating_sub(validation_rows).max(1);
    let train_fit = &train[..train_fit_rows];
    let validation = if validation_rows > 0 {
        &train[train_fit_rows..]
    } else {
        train_fit
    };
    let validation_guard_enabled =
        cfg.ga_selection_mode == "validation_guard" && validation_rows > 0;
    let dynamic_train_batch_rows = if cfg.ga_batch_rows > 0 {
        cfg.ga_batch_rows.min(train_fit.len())
    } else {
        train_fit.len()
    };
    let dynamic_validation_batch_rows = if cfg.ga_validation_batch_rows > 0 {
        cfg.ga_validation_batch_rows.min(validation.len())
    } else {
        validation.len()
    };
    append_progress(
        progress_path,
        "ga_start",
        started.elapsed().as_secs_f64(),
        json!({
            "population": population_size,
            "generations": cfg.ga_generations,
            "elite_count": elite_count,
            "mutation_steps": cfg.ga_mutation_steps,
            "selection_mode": cfg.ga_selection_mode,
            "train_fit_rows": train_fit.len(),
            "validation_rows": validation.len(),
            "dynamic_train_batch_rows": dynamic_train_batch_rows,
            "dynamic_validation_batch_rows": dynamic_validation_batch_rows,
            "validation_guard_enabled": validation_guard_enabled,
            "meaning": "Persistent independent population; elites survive, children mutate for fixed scout lifetime."
        }),
    )?;

    let mut next_id = 0usize;
    let mut population = Vec::with_capacity(population_size);
    for _ in 0..population_size {
        let net = build_network(init, init_rng);
        let proj = Int8Projection::new(init.phi_dim, POCKET_CLASSES, init_rng);
        population.push(GaIndividual {
            id: next_id,
            parent_id: None,
            net,
            proj,
        });
        next_id += 1;
    }

    let mut generation_metrics = Vec::new();
    let mut validation_guard_floor = 0.0f64;
    for generation in 0..cfg.ga_generations {
        let train_start =
            rotating_batch_start(train_fit.len(), dynamic_train_batch_rows, generation, 0);
        let validation_start = rotating_batch_start(
            validation.len(),
            dynamic_validation_batch_rows,
            generation,
            137,
        );
        let train_batch = &train_fit[train_start..train_start + dynamic_train_batch_rows];
        let validation_batch =
            &validation[validation_start..validation_start + dynamic_validation_batch_rows];
        let mut scored: Vec<GaScoredIndividual> = Vec::with_capacity(population.len());
        let mut train_fitness_sum = 0.0;
        let mut validation_fitness_sum = 0.0;
        for mut individual in population {
            let train_fitness =
                smooth_fitness(&mut individual.net, &individual.proj, train_batch, init);
            let validation_fitness = smooth_fitness(
                &mut individual.net,
                &individual.proj,
                validation_batch,
                init,
            );
            let validation_eval = eval_dataset(
                "validation",
                &mut individual.net,
                &individual.proj,
                validation_batch,
                init,
                None,
            )?;
            let guard_passed = !validation_guard_enabled
                || validation_eval.accuracy + 1.0e-12 >= validation_guard_floor;
            train_fitness_sum += train_fitness;
            validation_fitness_sum += validation_fitness;
            scored.push(GaScoredIndividual {
                individual,
                train_fitness,
                validation_fitness,
                validation_accuracy: validation_eval.accuracy,
                guard_passed,
            });
        }
        scored.sort_by(|a, b| cmp_f64(b.train_fitness, a.train_fitness));
        let population_count = scored.len();
        let guard_pass_count = scored.iter().filter(|row| row.guard_passed).count();

        let mut elite_indices: Vec<usize> = Vec::with_capacity(elite_count);
        if validation_guard_enabled {
            for (idx, row) in scored.iter().enumerate() {
                if row.guard_passed && elite_indices.len() < elite_count {
                    elite_indices.push(idx);
                }
            }
            if elite_indices.len() < elite_count {
                let mut fallback_indices: Vec<usize> = (0..scored.len()).collect();
                fallback_indices.sort_by(|a, b| {
                    cmp_f64(
                        scored[*b].validation_accuracy,
                        scored[*a].validation_accuracy,
                    )
                    .then_with(|| cmp_f64(scored[*b].train_fitness, scored[*a].train_fitness))
                });
                for idx in fallback_indices {
                    if elite_indices.len() >= elite_count {
                        break;
                    }
                    if !elite_indices.contains(&idx) {
                        elite_indices.push(idx);
                    }
                }
            }
        } else {
            elite_indices.extend(0..elite_count.min(scored.len()));
        }
        let fallback_fill_count = elite_indices
            .iter()
            .filter(|idx| !scored[**idx].guard_passed)
            .count();

        let best_idx = elite_indices.first().copied().unwrap_or(0);
        let mut best_for_eval = scored[best_idx].individual.clone();
        let best_train_fitness = scored[best_idx].train_fitness;
        let best_validation_fitness = scored[best_idx].validation_fitness;
        let best_validation_accuracy = scored[best_idx].validation_accuracy;
        let best_train = eval_dataset(
            "train_batch",
            &mut best_for_eval.net,
            &best_for_eval.proj,
            train_batch,
            init,
            None,
        )?;
        let best_validation = eval_dataset(
            "validation_batch",
            &mut best_for_eval.net,
            &best_for_eval.proj,
            validation_batch,
            init,
            None,
        )?;
        let best_test = eval_dataset(
            "test",
            &mut best_for_eval.net,
            &best_for_eval.proj,
            test,
            init,
            None,
        )?;
        let best_edge_count = best_for_eval.net.edge_count();
        let metric = GaGenerationMetric {
            generation,
            best_id: best_for_eval.id,
            best_parent_id: best_for_eval.parent_id,
            best_train_fitness,
            best_train_accuracy: best_train.accuracy,
            best_validation_fitness,
            best_validation_accuracy,
            best_test_accuracy: best_test.accuracy,
            best_train_margin: best_train.avg_margin,
            best_validation_margin: best_validation.avg_margin,
            best_test_margin: best_test.avg_margin,
            mean_train_fitness: train_fitness_sum / population_count.max(1) as f64,
            mean_validation_fitness: validation_fitness_sum / population_count.max(1) as f64,
            validation_guard_floor,
            validation_guard_pass_count: guard_pass_count,
            validation_guard_fallback_fill_count: fallback_fill_count,
            train_pool_rows: train_fit.len(),
            train_batch_rows: train_batch.len(),
            validation_batch_rows: validation_batch.len(),
            train_batch_start: train_start,
            validation_batch_start: validation_start,
            elite_count,
            population_size: population_count,
            best_edge_count,
        };
        append_jsonl_value(&cfg.out.join("ga_generation_metrics.jsonl"), &metric)?;
        append_progress(
            progress_path,
            "ga_generation_complete",
            started.elapsed().as_secs_f64(),
            json!(metric),
        )?;
        generation_metrics.push(metric);
        if validation_guard_enabled {
            validation_guard_floor = validation_guard_floor.max(best_validation.accuracy);
        }

        let elites: Vec<GaIndividual> = elite_indices
            .iter()
            .map(|idx| scored[*idx].individual.clone())
            .collect();
        let mut new_population = elites.clone();
        while new_population.len() < population_size {
            let parent = elites[new_population.len() % elites.len()].clone();
            let mut child = parent.clone();
            child.id = next_id;
            child.parent_id = Some(parent.id);
            next_id += 1;
            for _ in 0..cfg.ga_mutation_steps {
                let operator_index = sample_baseline_operator_for_example(mut_rng);
                let _ = apply_detailed_mutation(
                    operator_index,
                    &mut child.net,
                    &mut child.proj,
                    mut_rng,
                    init,
                );
            }
            new_population.push(child);
        }
        population = new_population;
    }

    let mut final_scored: Vec<GaScoredIndividual> = Vec::with_capacity(population.len());
    for mut individual in population {
        let train_fitness = smooth_fitness(&mut individual.net, &individual.proj, train_fit, init);
        let validation_fitness =
            smooth_fitness(&mut individual.net, &individual.proj, validation, init);
        let validation_eval = eval_dataset(
            "validation_full",
            &mut individual.net,
            &individual.proj,
            validation,
            init,
            None,
        )?;
        final_scored.push(GaScoredIndividual {
            individual,
            train_fitness,
            validation_fitness,
            validation_accuracy: validation_eval.accuracy,
            guard_passed: true,
        });
    }
    if validation_guard_enabled {
        final_scored.sort_by(|a, b| {
            cmp_f64(b.validation_accuracy, a.validation_accuracy)
                .then_with(|| cmp_f64(b.train_fitness, a.train_fitness))
        });
    } else {
        final_scored.sort_by(|a, b| cmp_f64(b.train_fitness, a.train_fitness));
    }
    let final_train_fitness = final_scored[0].train_fitness;
    let final_validation_fitness = final_scored[0].validation_fitness;
    let final_validation_accuracy = final_scored[0].validation_accuracy;
    let best = &mut final_scored[0].individual;
    let final_train = eval_dataset(
        "train_fit",
        &mut best.net,
        &best.proj,
        train_fit,
        init,
        None,
    )?;
    let final_validation = eval_dataset(
        "validation",
        &mut best.net,
        &best.proj,
        validation,
        init,
        None,
    )?;
    let final_test = eval_dataset(
        "test",
        &mut best.net,
        &best.proj,
        test,
        init,
        Some(&cfg.out.join("ga_best_row_level_predictions.jsonl")),
    )?;

    let best_generation = generation_metrics
        .iter()
        .max_by(|a, b| cmp_f64(a.best_test_accuracy, b.best_test_accuracy))
        .cloned();
    let summary = json!({
        "status": "complete",
        "task": "persistent_population_ga_smoke",
        "seed": cfg.seed,
        "task_mode": cfg.task_mode,
        "population": population_size,
        "generations": cfg.ga_generations,
        "elite_fraction": cfg.ga_elite_fraction,
        "elite_count": elite_count,
        "mutation_steps_per_child": cfg.ga_mutation_steps,
        "selection_mode": cfg.ga_selection_mode,
        "validation_guard_enabled": validation_guard_enabled,
        "train_fit_rows": train_fit.len(),
        "validation_rows": validation.len(),
        "dynamic_train_batch_rows": dynamic_train_batch_rows,
        "dynamic_validation_batch_rows": dynamic_validation_batch_rows,
        "before_single_baseline": {
            "train_accuracy": before_train.accuracy,
            "test_accuracy": before_test.accuracy,
            "train_avg_margin": before_train.avg_margin,
            "test_avg_margin": before_test.avg_margin
        },
        "final_best": {
            "id": best.id,
            "parent_id": best.parent_id,
            "train_fitness": final_train_fitness,
            "validation_fitness": final_validation_fitness,
            "train_accuracy": final_train.accuracy,
            "validation_accuracy": final_validation_accuracy,
            "test_accuracy": final_test.accuracy,
            "train_avg_margin": final_train.avg_margin,
            "validation_avg_margin": final_validation.avg_margin,
            "test_avg_margin": final_test.avg_margin,
            "edge_count": best.net.edge_count()
        },
        "best_generation_by_test_accuracy": best_generation,
        "test_accuracy_delta_from_single_baseline": final_test.accuracy - before_test.accuracy,
        "generation_metrics_file": "ga_generation_metrics.jsonl",
        "row_predictions_file": "ga_best_row_level_predictions.jsonl",
        "no_core_modified": true,
        "interpretation_limit": "This is a persistent-population routing smoke, not a broad capability claim."
    });
    write_json(&cfg.out.join("ga_summary.json"), &summary)?;
    write_ga_report(&cfg.out.join("report.md"), &summary)?;
    append_progress(
        progress_path,
        "ga_complete",
        started.elapsed().as_secs_f64(),
        summary,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_mask_probe(
    cfg: &Config,
    progress_path: &Path,
    started: Instant,
    train: &[Sample],
    test: &[Sample],
    init: &InitConfig,
    net: &mut Network,
    proj: &Int8Projection,
    rng: &mut StdRng,
    before_train: &EvalSummary,
    before_test: &EvalSummary,
) -> io::Result<()> {
    append_progress(
        progress_path,
        "mask_probe_start",
        started.elapsed().as_secs_f64(),
        json!({
            "rounds": cfg.mask_rounds,
            "candidates_per_round": cfg.mask_candidates,
            "mask_fraction": cfg.mask_fraction,
            "meaning": "Temporarily remove roads whose removal improves train smooth fitness, then retest."
        }),
    )?;

    let mut all_candidates: Vec<MaskCandidate> = Vec::new();
    let mut round_reports = Vec::new();
    let mut permanently_masked: Vec<serde_json::Value> = Vec::new();

    for round in 0..cfg.mask_rounds {
        let round_before_train = eval_dataset("train", net, proj, train, init, None)?;
        let round_before_test = eval_dataset("test", net, proj, test, init, None)?;
        let baseline_u = smooth_fitness(net, proj, train, init);
        append_progress(
            progress_path,
            "mask_round_start",
            started.elapsed().as_secs_f64(),
            json!({
                "round": round,
                "edge_count": net.edge_count(),
                "train_accuracy": round_before_train.accuracy,
                "test_accuracy": round_before_test.accuracy,
                "train_u": baseline_u
            }),
        )?;

        let mut round_candidates: Vec<MaskCandidate> = Vec::new();
        let attempts = cfg.mask_candidates.max(1).min(net.edge_count().max(1) * 2);
        for candidate_id in 0..attempts {
            if net.edge_count() == 0 {
                break;
            }
            let edge_idx = rng.gen_range(0..net.edge_count());
            let edge = net.graph().iter_edges().nth(edge_idx);
            if let Some(edge) = edge {
                let source = edge.source;
                let target = edge.target;
                if !net.graph_mut().remove_edge(source, target) {
                    continue;
                }
                let after_u = smooth_fitness(net, proj, train, init);
                if !net.graph_mut().add_edge(source, target) {
                    return Err(arg_err("mask probe could not restore sampled edge"));
                }
                round_candidates.push(MaskCandidate {
                    round,
                    candidate_id,
                    source,
                    target,
                    train_delta_u: after_u - baseline_u,
                    selected_for_mask: false,
                    detail: edge_detail_json("mask_test_remove_edge", source, target, init),
                });
            }

            if candidate_id % 128 == 0 && candidate_id > 0 {
                let mut partial = all_candidates.clone();
                partial.extend(round_candidates.clone());
                write_jsonl(&cfg.out.join("mask_candidates.partial.jsonl"), &partial)?;
                append_progress(
                    progress_path,
                    "mask_round_partial",
                    started.elapsed().as_secs_f64(),
                    json!({"round": round, "candidate_id": candidate_id, "records": partial.len()}),
                )?;
            }
        }

        round_candidates.sort_by(|a, b| cmp_f64(b.train_delta_u, a.train_delta_u));
        let target_masks = ((round_candidates.len() as f64) * cfg.mask_fraction).ceil() as usize;
        let mut masked_this_round = Vec::new();
        let mut seen_edges: BTreeMap<String, bool> = BTreeMap::new();
        for candidate in round_candidates.iter_mut() {
            if masked_this_round.len() >= target_masks.max(1) {
                break;
            }
            if candidate.train_delta_u <= 0.0 {
                break;
            }
            let key = format!("{}->{}", candidate.source, candidate.target);
            if seen_edges.contains_key(&key) {
                continue;
            }
            if net
                .graph_mut()
                .remove_edge(candidate.source, candidate.target)
            {
                candidate.selected_for_mask = true;
                seen_edges.insert(key, true);
                let detail = json!({
                    "round": round,
                    "source": candidate.source,
                    "target": candidate.target,
                    "train_delta_u": candidate.train_delta_u,
                    "detail": candidate.detail
                });
                masked_this_round.push(detail.clone());
                permanently_masked.push(detail);
            }
        }

        let round_after_train = eval_dataset("train", net, proj, train, init, None)?;
        let round_after_test = eval_dataset("test", net, proj, test, init, None)?;
        let report = json!({
            "round": round,
            "baseline_train_u": baseline_u,
            "candidates_tested": round_candidates.len(),
            "positive_removal_candidates": round_candidates.iter().filter(|c| c.train_delta_u > 0.0).count(),
            "masked_edge_count": masked_this_round.len(),
            "masked_edges": masked_this_round,
            "before": {
                "train_accuracy": round_before_train.accuracy,
                "test_accuracy": round_before_test.accuracy,
                "train_avg_margin": round_before_train.avg_margin,
                "test_avg_margin": round_before_test.avg_margin
            },
            "after": {
                "train_accuracy": round_after_train.accuracy,
                "test_accuracy": round_after_test.accuracy,
                "train_avg_margin": round_after_train.avg_margin,
                "test_avg_margin": round_after_test.avg_margin
            }
        });
        round_reports.push(report.clone());
        all_candidates.extend(round_candidates);
        write_jsonl(
            &cfg.out.join("mask_candidates.partial.jsonl"),
            &all_candidates,
        )?;
        write_json(
            &cfg.out.join("mask_round_report.partial.json"),
            &json!(round_reports),
        )?;
        append_progress(
            progress_path,
            "mask_round_complete",
            started.elapsed().as_secs_f64(),
            report,
        )?;
    }

    let after_train = eval_dataset("train", net, proj, train, init, None)?;
    let after_test = eval_dataset(
        "test",
        net,
        proj,
        test,
        init,
        Some(&cfg.out.join("row_level_predictions_masked.jsonl")),
    )?;
    write_jsonl(&cfg.out.join("mask_candidates.jsonl"), &all_candidates)?;

    let report = json!({
        "status": "complete",
        "task": "iterative_bad_road_mask_probe",
        "meaning": "This tests whether repeatedly removing roads whose temporary removal improves train smooth fitness leaves a cleaner network.",
        "seed": cfg.seed,
        "task_mode": cfg.task_mode,
        "rounds": cfg.mask_rounds,
        "mask_candidates_per_round": cfg.mask_candidates,
        "mask_fraction": cfg.mask_fraction,
        "before": {
            "train_accuracy": before_train.accuracy,
            "test_accuracy": before_test.accuracy,
            "train_avg_margin": before_train.avg_margin,
            "test_avg_margin": before_test.avg_margin
        },
        "after": {
            "train_accuracy": after_train.accuracy,
            "test_accuracy": after_test.accuracy,
            "train_avg_margin": after_train.avg_margin,
            "test_avg_margin": after_test.avg_margin
        },
        "delta": {
            "train_accuracy": after_train.accuracy - before_train.accuracy,
            "test_accuracy": after_test.accuracy - before_test.accuracy,
            "train_avg_margin": after_train.avg_margin - before_train.avg_margin,
            "test_avg_margin": after_test.avg_margin - before_test.avg_margin
        },
        "total_masked_edges": permanently_masked.len(),
        "masked_edges": permanently_masked,
        "round_reports": round_reports,
        "artifacts": {
            "raw_candidates": "mask_candidates.jsonl",
            "partial_candidates": "mask_candidates.partial.jsonl",
            "row_level_predictions_after_mask": "row_level_predictions_masked.jsonl",
            "progress": "progress.jsonl"
        }
    });
    write_json(&cfg.out.join("iterative_mask_report.json"), &report)?;
    write_mask_report(&cfg.out.join("report.md"), &report)?;
    append_progress(
        progress_path,
        "mask_probe_complete",
        started.elapsed().as_secs_f64(),
        report,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_heatmap(
    cfg: &Config,
    progress_path: &Path,
    started: Instant,
    train: &[Sample],
    test: &[Sample],
    init: &InitConfig,
    net: &mut Network,
    proj: &mut Int8Projection,
    mut_rng: &mut StdRng,
    before_train: &EvalSummary,
    before_test: &EvalSummary,
) -> io::Result<()> {
    append_progress(
        progress_path,
        "heatmap_start",
        started.elapsed().as_secs_f64(),
        json!({"candidates": cfg.heatmap_candidates}),
    )?;
    let before_u = smooth_fitness(net, proj, train, init);
    let parent_snapshot = net.save_state();
    let parent_projection = proj.clone();
    let mut records = Vec::with_capacity(cfg.heatmap_candidates);
    let clone_steps = cfg.heatmap_clone_steps.max(1);

    for candidate_id in 0..cfg.heatmap_candidates {
        net.restore_state(&parent_snapshot);
        *proj = parent_projection.clone();

        let mut steps = Vec::new();
        let mut mutated_count = 0usize;
        let mut first_operator = "none".to_string();
        let mut first_action = "noop".to_string();
        let mut first_details = json!({"kind": "noop"});
        for step_idx in 0..clone_steps {
            let operator_index = sample_baseline_operator_for_example(mut_rng);
            let operator_id = MUTATION_OPERATORS[operator_index].id.to_string();
            let (mutated, details, action_summary) =
                apply_detailed_mutation(operator_index, net, proj, mut_rng, init);
            if step_idx == 0 {
                first_operator = operator_id.clone();
                first_action = action_summary.clone();
                first_details = details.clone();
            }
            if mutated {
                mutated_count += 1;
            }
            steps.push(json!({
                "step": step_idx,
                "operator_id": operator_id,
                "mutated": mutated,
                "action_summary": action_summary,
                "details": details
            }));
        }
        let after_u = if mutated_count > 0 {
            smooth_fitness(net, proj, train, init)
        } else {
            before_u
        };
        let delta_u = after_u - before_u;
        let bucket = if delta_u > 0.0 {
            "winner"
        } else if delta_u < 0.0 {
            "loser"
        } else {
            "flat"
        };
        records.push(HeatmapCandidate {
            candidate_id,
            operator_id: if clone_steps == 1 {
                first_operator
            } else {
                "clone_walk".to_string()
            },
            mutated: mutated_count > 0,
            before_u,
            after_u,
            delta_u,
            bucket: bucket.to_string(),
            action_summary: if clone_steps == 1 {
                first_action
            } else {
                format!("clone_walk:{mutated_count}_mutations")
            },
            details: if clone_steps == 1 {
                first_details
            } else {
                json!({
                    "kind": "clone_walk",
                    "clone_steps": clone_steps,
                    "kept_mutations": "all",
                    "mutated_count": mutated_count,
                    "steps": steps
                })
            },
        });

        if candidate_id % 128 == 0 && candidate_id > 0 {
            write_jsonl(&cfg.out.join("heatmap_candidates.partial.jsonl"), &records)?;
            append_progress(
                progress_path,
                "heatmap_partial",
                started.elapsed().as_secs_f64(),
                json!({"candidate_id": candidate_id, "records": records.len()}),
            )?;
        }
    }
    net.restore_state(&parent_snapshot);
    *proj = parent_projection;

    write_jsonl(&cfg.out.join("heatmap_candidates.jsonl"), &records)?;
    append_progress(
        progress_path,
        "heatmap_candidates_written",
        started.elapsed().as_secs_f64(),
        json!({"candidate_records": records.len()}),
    )?;

    let report = build_heatmap_report(
        &records,
        before_train,
        before_test,
        init,
        clone_steps,
        train.len(),
        test.len(),
    );
    write_json(&cfg.out.join("mutation_heatmap_report.json"), &report)?;
    write_heatmap_report(&cfg.out.join("report.md"), &report)?;
    append_progress(
        progress_path,
        "heatmap_complete",
        started.elapsed().as_secs_f64(),
        report,
    )?;
    Ok(())
}

fn sample_baseline_operator_for_example(rng: &mut impl Rng) -> usize {
    let total: u32 = MUTATION_OPERATORS.iter().map(|op| op.baseline_weight).sum();
    let roll = rng.gen_range(0..total);
    let mut upper = 0u32;
    for (idx, spec) in MUTATION_OPERATORS.iter().enumerate() {
        upper += spec.baseline_weight;
        if roll < upper {
            return idx;
        }
    }
    MUTATION_OPERATORS.len() - 1
}

fn apply_detailed_mutation(
    operator_index: usize,
    net: &mut Network,
    proj: &mut Int8Projection,
    rng: &mut StdRng,
    init: &InitConfig,
) -> (bool, serde_json::Value, String) {
    let operator_id = MUTATION_OPERATORS[operator_index].id;
    match operator_id {
        "add_edge" => {
            mutation_detail_from_undo(operator_id, net.mutate_add_edge_undo(rng), net, init)
        }
        "remove_edge" => {
            mutation_detail_from_undo(operator_id, net.mutate_remove_edge_undo(rng), net, init)
        }
        "rewire" => mutation_detail_from_undo(operator_id, net.mutate_rewire_undo(rng), net, init),
        "reverse" => {
            mutation_detail_from_undo(operator_id, net.mutate_reverse_undo(rng), net, init)
        }
        "mirror" => mutation_detail_from_undo(operator_id, net.mutate_mirror_undo(rng), net, init),
        "enhance" => {
            mutation_detail_from_undo(operator_id, net.mutate_enhance_undo(rng), net, init)
        }
        "theta" => mutation_detail_from_undo(operator_id, net.mutate_theta_undo(rng), net, init),
        "channel" => {
            mutation_detail_from_undo(operator_id, net.mutate_channel_undo(rng), net, init)
        }
        "loop2" => {
            mutation_detail_from_undo(operator_id, net.mutate_add_loop_undo(rng, 2), net, init)
        }
        "loop3" => {
            mutation_detail_from_undo(operator_id, net.mutate_add_loop_undo(rng, 3), net, init)
        }
        "projection_weight" => {
            let _backup = proj.mutate_one(rng);
            (
                true,
                json!({
                    "kind": "projection_weight",
                    "detail_available": false,
                    "detail_limit": "WeightBackup fields are private on the public API; this candidate can be scored but not heatmapped to a pocket column without a core trace extension."
                }),
                "projection_weight_unknown".to_string(),
            )
        }
        other => (
            false,
            json!({"kind": "unknown", "operator_id": other}),
            "unknown".to_string(),
        ),
    }
}

fn mutation_detail_from_undo(
    operator_id: &str,
    result: (bool, MutationUndo),
    net: &Network,
    init: &InitConfig,
) -> (bool, serde_json::Value, String) {
    let (mutated, undo) = result;
    if !mutated {
        return (
            false,
            json!({"kind": "noop", "operator_id": operator_id}),
            "noop".to_string(),
        );
    }
    match undo {
        MutationUndo::AddedEdge { source, target } => {
            let detail = edge_detail_json("add_edge", source, target, init);
            let action = format!(
                "add:{}->{}",
                zone(source as usize, init),
                zone(target as usize, init)
            );
            (true, detail, action)
        }
        MutationUndo::RemovedEdge { source, target } => {
            let detail = edge_detail_json("remove_edge", source, target, init);
            let action = format!(
                "remove:{}->{}",
                zone(source as usize, init),
                zone(target as usize, init)
            );
            (true, detail, action)
        }
        MutationUndo::ReversedEdge {
            new_source,
            new_target,
        } => {
            let old_source = new_target;
            let old_target = new_source;
            let detail = json!({
                "kind": "reverse_edge",
                "old": edge_detail_json("old_edge", old_source, old_target, init),
                "new": edge_detail_json("new_edge", new_source, new_target, init)
            });
            let action = format!(
                "reverse:{}->{}=>{}->{}",
                zone(old_source as usize, init),
                zone(old_target as usize, init),
                zone(new_source as usize, init),
                zone(new_target as usize, init)
            );
            (true, detail, action)
        }
        MutationUndo::Rewired {
            old_source,
            old_target,
            new_source,
            new_target,
        } => {
            let detail = json!({
                "kind": "rewire",
                "old": edge_detail_json("old_edge", old_source, old_target, init),
                "new": edge_detail_json("new_edge", new_source, new_target, init)
            });
            let action = format!(
                "rewire:{}->{}=>{}->{}",
                zone(old_source as usize, init),
                zone(old_target as usize, init),
                zone(new_source as usize, init),
                zone(new_target as usize, init)
            );
            (true, detail, action)
        }
        MutationUndo::AddedLoop { edges } => {
            let edge_json: Vec<_> = edges
                .iter()
                .map(|(source, target)| edge_detail_json("loop_edge", *source, *target, init))
                .collect();
            let action = format!("loop_edges:{}", edges.len());
            (
                true,
                json!({"kind": "added_loop", "edges": edge_json}),
                action,
            )
        }
        MutationUndo::Theta { index, old_value } => {
            let new_value = net.threshold_at(index);
            let detail = json!({
                "kind": "theta",
                "neuron": neuron_detail_json(index, init),
                "old_threshold": old_value,
                "new_threshold": new_value,
                "delta": new_value as i16 - old_value as i16
            });
            let action = format!(
                "theta:{}:{}",
                zone(index, init),
                signed_delta(new_value as i16 - old_value as i16)
            );
            (true, detail, action)
        }
        MutationUndo::Channel { index, old_value } => {
            let new_value = net.channel_at(index);
            let detail = json!({
                "kind": "channel",
                "neuron": neuron_detail_json(index, init),
                "old_channel": old_value,
                "new_channel": new_value,
                "delta": new_value as i16 - old_value as i16
            });
            let action = format!(
                "channel:{}:{}",
                zone(index, init),
                signed_delta(new_value as i16 - old_value as i16)
            );
            (true, detail, action)
        }
        MutationUndo::Polarity { index } => {
            let detail = json!({"kind": "polarity", "neuron": neuron_detail_json(index, init)});
            let action = format!("polarity:{}", zone(index, init));
            (true, detail, action)
        }
        MutationUndo::EdgeWeight {
            edge_index,
            old_weight,
        } => {
            let detail =
                json!({"kind": "edge_weight", "edge_index": edge_index, "old_weight": old_weight});
            (true, detail, "edge_weight_unknown".to_string())
        }
        MutationUndo::Noop => (
            false,
            json!({"kind": "noop", "operator_id": operator_id}),
            "noop".to_string(),
        ),
    }
}

fn edge_detail_json(kind: &str, source: u16, target: u16, init: &InitConfig) -> serde_json::Value {
    json!({
        "kind": kind,
        "source": neuron_detail_json(source as usize, init),
        "target": neuron_detail_json(target as usize, init),
        "exact_road": format!("{}->{}", source, target),
        "band_road": format!("{}->{}", road_band(source as usize), road_band(target as usize)),
        "transition": format!("{}->{}", zone(source as usize, init), zone(target as usize, init)),
        "target_pocket_band": output_pocket_band(target as usize, init),
        "source_pocket_band": output_pocket_band(source as usize, init)
    })
}

fn neuron_detail_json(index: usize, init: &InitConfig) -> serde_json::Value {
    json!({
        "index": index,
        "zone": zone(index, init),
        "road_band": road_band(index),
        "pocket_band": output_pocket_band(index, init)
    })
}

fn road_band(index: usize) -> String {
    let start = (index / 16) * 16;
    let end = start + 15;
    format!("N{start:03}_{end:03}")
}

fn zone(index: usize, init: &InitConfig) -> &'static str {
    if index < init.output_start() {
        "input"
    } else if index < init.input_end() {
        "overlap"
    } else {
        "output"
    }
}

fn output_pocket_band(index: usize, init: &InitConfig) -> Option<usize> {
    if index < init.output_start() || index >= init.neuron_count {
        return None;
    }
    let width = init.neuron_count - init.output_start();
    let rel = index - init.output_start();
    Some((rel * POCKET_CLASSES / width).min(POCKET_CLASSES - 1))
}

fn signed_delta(delta: i16) -> &'static str {
    if delta > 0 {
        "up"
    } else if delta < 0 {
        "down"
    } else {
        "same"
    }
}

fn rotating_batch_start(
    pool_len: usize,
    batch_len: usize,
    generation: usize,
    salt: usize,
) -> usize {
    if pool_len <= batch_len || batch_len == 0 {
        return 0;
    }
    let span = pool_len - batch_len + 1;
    (generation
        .wrapping_mul(batch_len.saturating_mul(3).max(1))
        .wrapping_add(salt))
        % span
}

fn genome_hash(genome: &[u8], salt: u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ salt;
    for byte in genome {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
        hash ^= hash.rotate_left(23);
    }
    hash
}

fn network_from_genome_for_mode(
    genome: &[u8],
    init: &InitConfig,
    edges_per_neuron: usize,
    mode: &str,
) -> Network {
    match mode {
        "string_rule" => network_from_string_rule_genome(genome, init, edges_per_neuron),
        "u64_barcode" => network_from_u64_barcode_genome(genome, init, edges_per_neuron),
        "u64_slot_barcode" => network_from_u64_slot_barcode_genome(genome, init, edges_per_neuron),
        "u64_gate_sampled" => network_from_u64_gate_sampled_genome(genome, init, edges_per_neuron),
        "rule_dna_gate" => network_from_rule_dna_gate_genome(genome, init, edges_per_neuron, 0),
        "rule_dna_gate_mid" => network_from_rule_dna_gate_genome(genome, init, edges_per_neuron, 1),
        "rule_dna_gate_strict" => {
            network_from_rule_dna_gate_genome(genome, init, edges_per_neuron, 2)
        }
        _ => network_from_genome(genome, init, edges_per_neuron),
    }
}

fn network_from_genome(genome: &[u8], init: &InitConfig, edges_per_neuron: usize) -> Network {
    let mut net = Network::new(init.neuron_count);
    for neuron in 0..init.neuron_count {
        let h = genome_hash(genome, neuron as u64);
        net.set_threshold(neuron, (h % 8) as u8);
        net.set_channel(neuron, ((h / 8) % 8 + 1) as u8);
        if h % 17 == 0 {
            net.polarity_mut()[neuron] = -1;
        }
        for slot in 0..edges_per_neuron {
            let eh = genome_hash(
                genome,
                (neuron as u64).wrapping_mul(131).wrapping_add(slot as u64),
            );
            let mut target = (eh as usize % init.neuron_count) as u16;
            if target as usize == neuron {
                target = ((target as usize + 1) % init.neuron_count) as u16;
            }
            let _ = net.graph_mut().add_edge(neuron as u16, target);
        }
    }
    net
}

fn network_from_string_rule_genome(
    genome: &[u8],
    init: &InitConfig,
    edges_per_neuron: usize,
) -> Network {
    let mut net = Network::new(init.neuron_count);
    let code_len = 8usize;
    for neuron in 0..init.neuron_count {
        let h = genome_hash(genome, 0xC0DE_0000 ^ neuron as u64);
        net.set_threshold(neuron, (h % 8) as u8);
        net.set_channel(neuron, ((h / 8) % 8 + 1) as u8);
        if ((h >> 9) & 7) == 0 {
            net.polarity_mut()[neuron] = -1;
        }
    }

    for source in 0..init.neuron_count {
        let mut scored_targets = Vec::with_capacity(init.neuron_count.saturating_sub(1));
        let source_code = neuron_string_code(genome, source, code_len);
        for target in 0..init.neuron_count {
            if source == target {
                continue;
            }
            let target_code = neuron_string_code(genome, target, code_len);
            let score =
                string_rule_connection_score(genome, &source_code, &target_code, source, target);
            scored_targets.push((score, target));
        }
        scored_targets.sort_by(|a, b| cmp_f64(b.0, a.0));
        for (_, target) in scored_targets.into_iter().take(edges_per_neuron.max(1)) {
            let _ = net.graph_mut().add_edge(source as u16, target as u16);
        }
    }
    net
}

fn neuron_string_code(genome: &[u8], neuron: usize, code_len: usize) -> Vec<u8> {
    let mut code = Vec::with_capacity(code_len);
    for pos in 0..code_len {
        let salt = 0x51A7_0000_u64
            .wrapping_add((neuron as u64).wrapping_mul(257))
            .wrapping_add((pos as u64).wrapping_mul(17));
        code.push((genome_hash(genome, salt) % 16) as u8);
    }
    code
}

fn string_rule_connection_score(
    genome: &[u8],
    source_code: &[u8],
    target_code: &[u8],
    source: usize,
    target: usize,
) -> f64 {
    let code_len = source_code.len().min(target_code.len()).max(1);
    let mut score = 0.0;
    let rule_count = 12usize;
    for rule in 0..rule_count {
        let raw = genome_hash(genome, 0xB17D_0000_u64.wrapping_add(rule as u64));
        let src_pos = (raw as usize) % code_len;
        let dst_pos = ((raw >> 4) as usize) % code_len;
        let op = ((raw >> 8) & 3) as u8;
        let weight = (((raw >> 10) & 15) as f64 - 7.5) / 3.0;
        let a = source_code[src_pos];
        let b = target_code[dst_pos];
        let hit = match op {
            0 => a == b,
            1 => (a ^ b) <= 3,
            2 => ((a as i16 - b as i16).abs() as u8) <= 2,
            _ => ((a.wrapping_add(b)) & 3) == ((raw >> 14) as u8 & 3),
        };
        if hit {
            score += weight;
        } else {
            score -= weight.abs() * 0.15;
        }
    }
    let long_jump = ((genome_hash(genome, 0xA11E_0000_u64 ^ source as u64) >> 3) & 15) as f64;
    let index_gap = source.abs_diff(target) as f64 / 256.0;
    score += long_jump * index_gap * 0.08;
    score
}

fn string_rule_dns_report(genome: &[u8], edges_per_neuron: usize) -> serde_json::Value {
    let code_len = 8usize;
    let rules: Vec<serde_json::Value> = (0..12usize)
        .map(|rule| {
            let raw = genome_hash(genome, 0xB17D_0000_u64.wrapping_add(rule as u64));
            json!({
                "rule": rule,
                "source_code_position": (raw as usize) % code_len,
                "target_code_position": ((raw >> 4) as usize) % code_len,
                "operation": match ((raw >> 8) & 3) as u8 {
                    0 => "same_symbol",
                    1 => "xor_close",
                    2 => "numeric_close",
                    _ => "sum_bucket",
                },
                "weight": (((raw >> 10) & 15) as f64 - 7.5) / 3.0
            })
        })
        .collect();
    json!({
        "enabled": true,
        "neuron_code_length_symbols": code_len,
        "alphabet_size": 16,
        "edges_per_neuron": edges_per_neuron,
        "rules": rules,
        "meaning": "Each neuron gets a deterministic 8-symbol code from the DNS. The rule table scores source-code plus target-code and the top scored targets become roads."
    })
}

fn network_from_u64_barcode_genome(
    genome: &[u8],
    init: &InitConfig,
    edges_per_neuron: usize,
) -> Network {
    let mut net = Network::new(init.neuron_count);
    let mut codes = Vec::with_capacity(init.neuron_count);
    for neuron in 0..init.neuron_count {
        let code = u64_neuron_barcode(genome, neuron);
        codes.push(code);
        net.set_threshold(neuron, (code & 7) as u8);
        net.set_channel(neuron, (((code >> 3) & 7) + 1) as u8);
        if ((code >> 6) & 7) == 0 {
            net.polarity_mut()[neuron] = -1;
        }
    }

    for source in 0..init.neuron_count {
        let source_code = codes[source];
        let mut top_targets: Vec<(f64, usize)> = Vec::with_capacity(edges_per_neuron.max(1));
        for (target, target_code) in codes.iter().enumerate() {
            if source == target {
                continue;
            }
            let score =
                u64_barcode_connection_score(genome, source_code, *target_code, source, target);
            insert_top_target(&mut top_targets, edges_per_neuron.max(1), score, target);
        }
        for (_, target) in top_targets {
            let _ = net.graph_mut().add_edge(source as u16, target as u16);
        }
    }
    net
}

fn insert_top_target(top_targets: &mut Vec<(f64, usize)>, limit: usize, score: f64, target: usize) {
    if limit == 0 {
        return;
    }
    if top_targets.len() < limit {
        top_targets.push((score, target));
        return;
    }
    let mut worst_idx = 0usize;
    let mut worst_score = top_targets[0].0;
    for (idx, (candidate_score, _)) in top_targets.iter().enumerate().skip(1) {
        if *candidate_score < worst_score {
            worst_score = *candidate_score;
            worst_idx = idx;
        }
    }
    if score > worst_score {
        top_targets[worst_idx] = (score, target);
    }
}

fn u64_neuron_barcode(genome: &[u8], neuron: usize) -> u64 {
    genome_hash(genome, 0x64BA_C0DE_0000_0000_u64 ^ neuron as u64)
}

fn u64_digit(code: u64, pos: usize) -> u8 {
    ((code >> ((pos & 15) * 4)) & 0xF) as u8
}

fn u64_barcode_connection_score(
    genome: &[u8],
    source_code: u64,
    target_code: u64,
    source: usize,
    target: usize,
) -> f64 {
    let mut score = 0.0;
    for rule in 0..12usize {
        let raw = genome_hash(genome, 0x64B1_7D00_u64.wrapping_add(rule as u64));
        let src_pos = raw as usize & 15;
        let dst_pos = (raw >> 4) as usize & 15;
        let op = ((raw >> 8) & 3) as u8;
        let weight = (((raw >> 10) & 15) as f64 - 7.5) / 3.0;
        let a = u64_digit(source_code, src_pos);
        let b = u64_digit(target_code, dst_pos);
        let hit = match op {
            0 => a == b,
            1 => (a ^ b) <= 3,
            2 => ((a as i16 - b as i16).abs() as u8) <= 2,
            _ => ((a.wrapping_add(b)) & 3) == ((raw >> 14) as u8 & 3),
        };
        if hit {
            score += weight;
        } else {
            score -= weight.abs() * 0.15;
        }
    }
    let jump_digit = u64_digit(source_code, 15) as f64;
    let index_gap = source.abs_diff(target) as f64 / 256.0;
    score += jump_digit * index_gap * 0.08;
    score
}

fn network_from_u64_slot_barcode_genome(
    genome: &[u8],
    init: &InitConfig,
    edges_per_neuron: usize,
) -> Network {
    let mut net = Network::new(init.neuron_count);
    let mut codes = Vec::with_capacity(init.neuron_count);
    for neuron in 0..init.neuron_count {
        let code = u64_neuron_barcode(genome, neuron);
        codes.push(code);
        net.set_threshold(neuron, (code & 7) as u8);
        net.set_channel(neuron, (((code >> 3) & 7) + 1) as u8);
        if ((code >> 6) & 7) == 0 {
            net.polarity_mut()[neuron] = -1;
        }
    }

    for source in 0..init.neuron_count {
        let source_code = codes[source];
        for slot in 0..edges_per_neuron.max(1) {
            let mut best_target = source;
            let mut best_score = f64::NEG_INFINITY;
            for candidate_idx in 0..4usize {
                let target = u64_slot_candidate_target(
                    genome,
                    source_code,
                    source,
                    slot,
                    candidate_idx,
                    init.neuron_count,
                    init.phi_dim,
                );
                if target == source {
                    continue;
                }
                let score = u64_barcode_connection_score(
                    genome,
                    source_code,
                    codes[target],
                    source,
                    target,
                );
                if score > best_score {
                    best_score = score;
                    best_target = target;
                }
            }
            if best_target != source {
                let _ = net.graph_mut().add_edge(source as u16, best_target as u16);
            }
        }
    }
    net
}

fn u64_slot_candidate_target(
    genome: &[u8],
    source_code: u64,
    source: usize,
    slot: usize,
    candidate_idx: usize,
    neuron_count: usize,
    phi_dim: usize,
) -> usize {
    if neuron_count <= 1 {
        return source;
    }
    let raw = genome_hash(
        genome,
        source_code
            .wrapping_add((slot as u64).wrapping_mul(0x9E37))
            .wrapping_add((candidate_idx as u64).wrapping_mul(0x85EB)),
    );
    let mode = (u64_digit(source_code, slot) ^ ((raw >> 11) as u8)) & 7;
    match mode {
        0 | 1 => {
            let radius = 1 + ((raw >> 16) as usize % 24);
            if raw & 1 == 0 {
                (source + radius) % neuron_count
            } else {
                (source + neuron_count - (radius % neuron_count)) % neuron_count
            }
        }
        2 | 3 => {
            let band = 16usize;
            let band_count = (neuron_count / band).max(1);
            let target_band = ((raw >> 20) as usize % band_count) * band;
            (target_band + ((raw >> 28) as usize % band)).min(neuron_count - 1)
        }
        4 | 5 => {
            let output_start = phi_dim.min(neuron_count.saturating_sub(1));
            let width = neuron_count.saturating_sub(output_start).max(1);
            output_start + ((raw >> 24) as usize % width)
        }
        _ => (raw as usize) % neuron_count,
    }
}

fn network_from_u64_gate_sampled_genome(
    genome: &[u8],
    init: &InitConfig,
    edges_per_neuron: usize,
) -> Network {
    let mut net = Network::new(init.neuron_count);
    let mut codes = Vec::with_capacity(init.neuron_count);
    for neuron in 0..init.neuron_count {
        let code = u64_neuron_barcode(genome, neuron);
        codes.push(code);
        net.set_threshold(neuron, (code & 7) as u8);
        net.set_channel(neuron, (((code >> 3) & 7) + 1) as u8);
        if ((code >> 6) & 7) == 0 {
            net.polarity_mut()[neuron] = -1;
        }
    }

    for source in 0..init.neuron_count {
        let source_code = codes[source];
        let threshold_raw = genome_hash(genome, 0x647A_7E00_u64 ^ source_code);
        let threshold = (((threshold_raw >> 12) & 15) as f64 - 7.5) / 4.0 - 0.75;
        let mut accepted = 0usize;
        for attempt in 0..96usize {
            if accepted >= edges_per_neuron.max(1) {
                break;
            }
            let target = u64_sampled_gate_target(
                genome,
                source_code,
                source,
                attempt,
                init.neuron_count,
                init.phi_dim,
            );
            if target == source {
                continue;
            }
            let score =
                u64_barcode_connection_score(genome, source_code, codes[target], source, target);
            if score >= threshold {
                let before = net.edge_count();
                let _ = net.graph_mut().add_edge(source as u16, target as u16);
                if net.edge_count() > before {
                    accepted += 1;
                }
            }
        }
        let minimum_edges = edges_per_neuron.max(1).min(6);
        for slot in accepted..minimum_edges {
            let target = u64_sampled_gate_target(
                genome,
                source_code,
                source,
                64 + slot,
                init.neuron_count,
                init.phi_dim,
            );
            if target != source {
                let _ = net.graph_mut().add_edge(source as u16, target as u16);
            }
        }
    }
    net
}

fn u64_sampled_gate_target(
    genome: &[u8],
    source_code: u64,
    source: usize,
    attempt: usize,
    neuron_count: usize,
    phi_dim: usize,
) -> usize {
    if neuron_count <= 1 {
        return source;
    }
    let raw = genome_hash(
        genome,
        source_code
            .wrapping_mul(0xD6E8_FD93)
            .wrapping_add((attempt as u64).wrapping_mul(0x9E37_79B1)),
    );
    let mode = ((raw >> 8) & 7) as u8;
    match mode {
        0 | 1 | 2 => (raw as usize) % neuron_count,
        3 | 4 => {
            let radius = 1 + ((raw >> 16) as usize % 32);
            if raw & 1 == 0 {
                (source + radius) % neuron_count
            } else {
                (source + neuron_count - (radius % neuron_count)) % neuron_count
            }
        }
        5 => {
            let band = 16usize;
            let band_count = (neuron_count / band).max(1);
            let target_band = ((raw >> 20) as usize % band_count) * band;
            (target_band + ((raw >> 28) as usize % band)).min(neuron_count - 1)
        }
        _ => {
            let output_start = phi_dim.min(neuron_count.saturating_sub(1));
            let width = neuron_count.saturating_sub(output_start).max(1);
            output_start + ((raw >> 24) as usize % width)
        }
    }
}

fn network_from_rule_dna_gate_genome(
    genome: &[u8],
    init: &InitConfig,
    edges_per_neuron: usize,
    threshold_mode: u8,
) -> Network {
    let mut net = Network::new(init.neuron_count);
    let mut codes = Vec::with_capacity(init.neuron_count);
    for neuron in 0..init.neuron_count {
        let code = u64_neuron_barcode(genome, neuron);
        codes.push(code);
        net.set_threshold(neuron, (code & 7) as u8);
        net.set_channel(neuron, (((code >> 3) & 7) + 1) as u8);
        if ((code >> 6) & 7) == 0 {
            net.polarity_mut()[neuron] = -1;
        }
    }

    let road_budget = edges_per_neuron.max(1);
    for source in 0..init.neuron_count {
        let source_code = codes[source];
        let threshold = rule_dna_gate_threshold(genome, source_code, source, threshold_mode);
        let order_seed = genome_hash(genome, 0xA7E5_0000_u64 ^ source_code);
        let step = ((order_seed as usize) | 1).max(1);
        let mut accepted = 0usize;
        for offset in 1..=init.neuron_count {
            if accepted >= road_budget {
                break;
            }
            let target = (source + offset.wrapping_mul(step)) % init.neuron_count;
            if target == source {
                continue;
            }
            let score =
                rule_dna_gate_connection_score(genome, source_code, codes[target], source, target);
            if score >= threshold {
                let before = net.edge_count();
                let _ = net.graph_mut().add_edge(source as u16, target as u16);
                if net.edge_count() > before {
                    accepted += 1;
                }
            }
        }
    }
    net
}

fn rule_dna_gate_threshold(
    genome: &[u8],
    source_code: u64,
    source: usize,
    threshold_mode: u8,
) -> i32 {
    let raw = genome_hash(
        genome,
        0xA7E5_7E00_u64 ^ source_code ^ ((source as u64).wrapping_mul(0x9E37_79B1)),
    );
    match threshold_mode {
        0 => ((raw >> 12) as i32 & 7) - 2,
        1 => ((raw >> 12) as i32 & 15) - 2,
        _ => (raw >> 12) as i32 & 15,
    }
}

fn rule_dna_gate_connection_score(
    genome: &[u8],
    source_code: u64,
    target_code: u64,
    source: usize,
    target: usize,
) -> i32 {
    let mut score = 0i32;
    for rule in 0..16usize {
        let raw = genome_hash(genome, 0xA7E5_B17D_u64.wrapping_add(rule as u64));
        let src_pos = raw as usize & 15;
        let dst_pos = (raw >> 4) as usize & 15;
        let op = ((raw >> 8) & 7) as u8;
        let weight = ((raw >> 11) & 7) as i32 + 1;
        let sign = if ((raw >> 14) & 1) == 0 { 1 } else { -1 };
        let limit = ((raw >> 15) & 3) as u8 + 1;
        let a = u64_digit(source_code, src_pos);
        let b = u64_digit(target_code, dst_pos);
        let hit = match op {
            0 => a == b,
            1 => (a ^ b) <= limit,
            2 => ((a as i16 - b as i16).abs() as u8) <= limit,
            3 => ((a.wrapping_add(b)) & 7) == ((raw >> 18) as u8 & 7),
            4 => (a & 1) == (b & 1),
            5 => ((a.wrapping_mul(3).wrapping_add(b)) & 15) == ((raw >> 21) as u8 & 15),
            6 => ((source as u8).wrapping_add(a) & 7) == ((target as u8).wrapping_add(b) & 7),
            _ => ((a ^ (b.rotate_left(1))) & 7) == ((raw >> 25) as u8 & 7),
        };
        if hit {
            score += sign * weight;
        }
    }
    score
}

fn rule_dna_gate_report(
    genome: &[u8],
    edges_per_neuron: usize,
    threshold_mode: u8,
) -> serde_json::Value {
    let rules: Vec<serde_json::Value> = (0..16usize)
        .map(|rule| {
            let raw = genome_hash(genome, 0xA7E5_B17D_u64.wrapping_add(rule as u64));
            json!({
                "rule": rule,
                "source_digit_position": raw as usize & 15,
                "target_digit_position": (raw >> 4) as usize & 15,
                "operation": match ((raw >> 8) & 7) as u8 {
                    0 => "same_digit",
                    1 => "xor_close",
                    2 => "numeric_close",
                    3 => "sum_bucket",
                    4 => "same_even_odd",
                    5 => "mixed_digit_bucket",
                    6 => "source_target_bucket",
                    _ => "rotated_xor_bucket",
                },
                "weight": ((raw >> 11) & 7) as i32 + 1,
                "direction": if ((raw >> 14) & 1) == 0 { "add_when_hit" } else { "subtract_when_hit" },
                "limit": ((raw >> 15) & 3) + 1
            })
        })
        .collect();

    let mut sample_pairs = Vec::new();
    for source in 0..3usize {
        let source_code = u64_neuron_barcode(genome, source);
        let threshold = rule_dna_gate_threshold(genome, source_code, source, threshold_mode);
        for target in 0..4usize {
            if source == target {
                continue;
            }
            let target_code = u64_neuron_barcode(genome, target);
            let score =
                rule_dna_gate_connection_score(genome, source_code, target_code, source, target);
            sample_pairs.push(json!({
                "source_neuron": source,
                "target_neuron": target,
                "source_barcode_hex": format!("{source_code:016X}"),
                "target_barcode_hex": format!("{target_code:016X}"),
                "score": score,
                "threshold": threshold,
                "connection": score >= threshold
            }));
        }
    }

    json!({
        "enabled": true,
        "mode": match threshold_mode {
            0 => "rule_dna_gate",
            1 => "rule_dna_gate_mid",
            _ => "rule_dna_gate_strict",
        },
        "barcode_type": "u64",
        "digits_per_neuron": 16,
        "digit_values": "0..15",
        "rule_count": 16,
        "max_roads_per_neuron": edges_per_neuron.max(1),
        "connection_rule": match threshold_mode {
            0 => "For each A+B neuron pair, the DNS rule table gives a score. Loose threshold uses -2..5 per source. If score >= threshold, the road opens. The cap only prevents a fully dense road storm.",
            1 => "For each A+B neuron pair, the DNS rule table gives a score. Mid threshold uses -2..13 per source. If score >= threshold, the road opens. The cap only prevents a fully dense road storm.",
            _ => "For each A+B neuron pair, the DNS rule table gives a score. Strict threshold uses 0..15 per source. If score >= threshold, the road opens. The cap only prevents a fully dense road storm.",
        },
        "rules": rules,
        "sample_pair_decisions": sample_pairs,
        "meaning": "This is the simple gate form: barcode A plus barcode B plus DNS rules decides connect or no-connect."
    })
}

fn u64_barcode_dns_report(
    genome: &[u8],
    edges_per_neuron: usize,
    genome_mode: &str,
) -> serde_json::Value {
    let rules: Vec<serde_json::Value> = (0..12usize)
        .map(|rule| {
            let raw = genome_hash(genome, 0x64B1_7D00_u64.wrapping_add(rule as u64));
            json!({
                "rule": rule,
                "source_digit_position": raw as usize & 15,
                "target_digit_position": (raw >> 4) as usize & 15,
                "operation": match ((raw >> 8) & 3) as u8 {
                    0 => "same_digit",
                    1 => "xor_close",
                    2 => "numeric_close",
                    _ => "sum_bucket",
                },
                "weight": (((raw >> 10) & 15) as f64 - 7.5) / 3.0
            })
        })
        .collect();
    json!({
        "enabled": true,
        "barcode_type": "u64",
        "genome_mode": genome_mode,
        "digits_per_neuron": 16,
        "digit_values": "0..15",
        "bytes_per_neuron_code": 8,
        "edges_per_neuron": edges_per_neuron,
        "target_selection": if genome_mode == "u64_slot_barcode" {
            "slot mode: each source-slot generates four candidate targets from the source barcode and DNS, then keeps the best local score. No all-pairs scan."
        } else if genome_mode == "u64_gate_sampled" {
            "sampled gate mode: the DNS proposes up to 64 candidate targets per source; rule(A,B) opens or closes the connection gate until max edges are reached. No all-pairs scan."
        } else {
            "pairwise mode: each source scores every possible target and keeps the top targets."
        },
        "rules": rules,
        "meaning": "Each neuron gets one u64 barcode. The 16 four-bit digits guide road growth."
    })
}

fn projection_from_genome(genome: &[u8], init: &InitConfig) -> Int8Projection {
    let seed = genome_hash(genome, 0x9E37_79B9_7F4A_7C15);
    let mut rng = StdRng::seed_from_u64(seed);
    Int8Projection::new(init.phi_dim, POCKET_CLASSES, &mut rng)
}

fn mutate_genome(genome: &mut [u8], mutation_bytes: usize, rng: &mut StdRng) {
    if genome.is_empty() {
        return;
    }
    for _ in 0..mutation_bytes {
        let idx = rng.gen_range(0..genome.len());
        let delta = rng.gen_range(1..=255u8);
        genome[idx] = genome[idx].wrapping_add(delta);
    }
}

#[allow(clippy::too_many_arguments)]
fn run_compass(
    cfg: &Config,
    progress_path: &Path,
    started: Instant,
    train: &[Sample],
    test: &[Sample],
    init: &InitConfig,
    evo_cfg: &instnct_core::EvolutionConfig,
    net: &mut Network,
    proj: &mut Int8Projection,
    mut_rng: &mut StdRng,
    eval_rng: &mut StdRng,
    before_train: &EvalSummary,
    before_test: &EvalSummary,
) -> io::Result<()> {
    append_progress(
        progress_path,
        "compass_start",
        started.elapsed().as_secs_f64(),
        json!({"candidates": cfg.compass_candidates}),
    )?;

    let mut records: Vec<CompassCandidate> = Vec::with_capacity(cfg.compass_candidates);
    let outcome = evolution_step_jackpot_traced(
        net,
        proj,
        mut_rng,
        eval_rng,
        |candidate_net, candidate_proj, _rng| {
            smooth_fitness(candidate_net, candidate_proj, train, init)
        },
        evo_cfg,
        cfg.compass_candidates,
        1,
        |record: &CandidateTraceRecord| {
            records.push(CompassCandidate {
                candidate_id: record.candidate_id,
                operator_id: record.operator_id.to_string(),
                mutated: record.mutated,
                evaluated: record.evaluated,
                before_u: record.before_u,
                after_u: record.after_u,
                delta_u: record.delta_u,
                within_cap: record.within_cap,
                selected: record.selected,
                accepted: record.accepted,
                candidate_eval_ms: record.candidate_eval_ms,
                step_wall_ms: record.step_wall_ms,
            });
        },
    );

    write_jsonl(&cfg.out.join("compass_candidates.jsonl"), &records)?;
    append_progress(
        progress_path,
        "compass_candidates_written",
        started.elapsed().as_secs_f64(),
        json!({"candidate_records": records.len(), "outcome": format!("{outcome:?}")}),
    )?;

    let after_train = eval_dataset("train", net, proj, train, init, None)?;
    let after_test = eval_dataset("test", net, proj, test, init, None)?;
    let row_path = cfg.out.join("row_level_predictions.jsonl");
    let _ = eval_dataset("test", net, proj, test, init, Some(&row_path))?;

    let mut by_operator: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut grouped: BTreeMap<String, Vec<&CompassCandidate>> = BTreeMap::new();
    for record in &records {
        grouped
            .entry(record.operator_id.clone())
            .or_default()
            .push(record);
    }
    for (op, rows) in grouped {
        let evaluated: Vec<&CompassCandidate> =
            rows.iter().copied().filter(|r| r.evaluated).collect();
        let positive = evaluated.iter().filter(|r| r.delta_u > 0.0).count();
        let avg_delta = if evaluated.is_empty() {
            0.0
        } else {
            evaluated.iter().map(|r| r.delta_u).sum::<f64>() / evaluated.len() as f64
        };
        let best_delta = evaluated
            .iter()
            .map(|r| r.delta_u)
            .fold(f64::NEG_INFINITY, f64::max);
        by_operator.insert(
            op,
            json!({
                "attempts": rows.len(),
                "evaluated": evaluated.len(),
                "positive_delta_count": positive,
                "positive_delta_rate": if evaluated.is_empty() { 0.0 } else { positive as f64 / evaluated.len() as f64 },
                "avg_delta": avg_delta,
                "best_delta": if best_delta.is_finite() { best_delta } else { 0.0 },
                "selected_count": rows.iter().filter(|r| r.selected).count(),
                "accepted_count": rows.iter().filter(|r| r.accepted).count()
            }),
        );
    }

    let mut sorted = records.clone();
    sorted.sort_by(|a, b| cmp_f64(b.delta_u, a.delta_u));
    let top_n = (sorted.len() / 10).max(1);
    let top_rows = &sorted[..top_n.min(sorted.len())];
    let bottom_rows = &sorted[sorted.len().saturating_sub(top_n)..];
    let mut top_operator_counts: BTreeMap<String, usize> = BTreeMap::new();
    for row in top_rows {
        *top_operator_counts
            .entry(row.operator_id.clone())
            .or_insert(0) += 1;
    }
    let mut bottom_operator_counts: BTreeMap<String, usize> = BTreeMap::new();
    for row in bottom_rows {
        *bottom_operator_counts
            .entry(row.operator_id.clone())
            .or_insert(0) += 1;
    }
    let mut positive_operator_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut negative_operator_counts: BTreeMap<String, usize> = BTreeMap::new();
    for row in records.iter().filter(|r| r.evaluated) {
        if row.delta_u > 0.0 {
            *positive_operator_counts
                .entry(row.operator_id.clone())
                .or_insert(0) += 1;
        } else if row.delta_u < 0.0 {
            *negative_operator_counts
                .entry(row.operator_id.clone())
                .or_insert(0) += 1;
        }
    }
    let common_denominator = build_common_denominator_report(
        &records,
        &positive_operator_counts,
        &negative_operator_counts,
        &top_operator_counts,
        &bottom_operator_counts,
    );
    write_json(
        &cfg.out.join("common_denominator_report.json"),
        &common_denominator,
    )?;
    let positive_delta_count = records
        .iter()
        .filter(|r| r.evaluated && r.delta_u > 0.0)
        .count();
    let best = sorted.first();
    let accepted = records.iter().find(|r| r.accepted);
    let holdout_survives = after_test.accuracy >= before_test.accuracy;

    let summary = json!({
        "status": "complete",
        "task": "official_vraxion_raven_mutation_compass_smoke",
        "seed": cfg.seed,
        "task_mode": cfg.task_mode,
        "compass_candidates": cfg.compass_candidates,
        "outcome": format!("{outcome:?}"),
        "before": {
            "train_accuracy": before_train.accuracy,
            "test_accuracy": before_test.accuracy,
            "train_avg_margin": before_train.avg_margin,
            "test_avg_margin": before_test.avg_margin
        },
        "after_selected_candidate": {
            "train_accuracy": after_train.accuracy,
            "test_accuracy": after_test.accuracy,
            "train_avg_margin": after_train.avg_margin,
            "test_avg_margin": after_test.avg_margin,
            "holdout_survives": holdout_survives
        },
        "direction_signal": {
            "positive_delta_count": positive_delta_count,
            "positive_delta_rate": if records.is_empty() { 0.0 } else { positive_delta_count as f64 / records.len() as f64 },
            "best_delta": best.map(|r| r.delta_u).unwrap_or(0.0),
            "best_operator": best.map(|r| r.operator_id.clone()).unwrap_or_else(|| "none".to_string()),
            "accepted_delta": accepted.map(|r| r.delta_u).unwrap_or(0.0),
            "accepted_operator": accepted.map(|r| r.operator_id.clone()).unwrap_or_else(|| "none".to_string()),
            "top_10_percent_operator_counts": top_operator_counts,
            "bottom_10_percent_operator_counts": bottom_operator_counts,
            "positive_operator_counts": positive_operator_counts,
            "negative_operator_counts": negative_operator_counts,
            "has_train_direction": positive_delta_count > 0,
            "has_holdout_direction": holdout_survives && outcome == instnct_core::StepOutcome::Accepted
        },
        "common_denominator_report": common_denominator,
        "by_operator": by_operator,
        "artifacts": {
            "candidate_records": "compass_candidates.jsonl",
            "common_denominator_report": "common_denominator_report.json",
            "raw_test_predictions_after_selected_candidate": "row_level_predictions.jsonl",
            "progress": "progress.jsonl"
        },
        "elapsed_sec": started.elapsed().as_secs_f64()
    });
    write_json(&cfg.out.join("compass_summary.json"), &summary)?;
    write_compass_report(&cfg.out.join("report.md"), &summary)?;
    append_progress(
        progress_path,
        "compass_complete",
        started.elapsed().as_secs_f64(),
        summary,
    )?;
    Ok(())
}

fn build_common_denominator_report(
    records: &[CompassCandidate],
    positive_operator_counts: &BTreeMap<String, usize>,
    negative_operator_counts: &BTreeMap<String, usize>,
    top_operator_counts: &BTreeMap<String, usize>,
    bottom_operator_counts: &BTreeMap<String, usize>,
) -> serde_json::Value {
    let evaluated: Vec<&CompassCandidate> = records.iter().filter(|r| r.evaluated).collect();
    let evaluated_n = evaluated.len().max(1);
    let positive_n = positive_operator_counts.values().sum::<usize>().max(1);
    let negative_n = negative_operator_counts.values().sum::<usize>().max(1);
    let top_n = top_operator_counts.values().sum::<usize>().max(1);
    let bottom_n = bottom_operator_counts.values().sum::<usize>().max(1);

    let mut all_operator_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut delta_sums: BTreeMap<String, f64> = BTreeMap::new();
    for row in evaluated {
        *all_operator_counts
            .entry(row.operator_id.clone())
            .or_insert(0) += 1;
        *delta_sums.entry(row.operator_id.clone()).or_insert(0.0) += row.delta_u;
    }

    let mut operator_rows = BTreeMap::new();
    for (op, all_count) in &all_operator_counts {
        let positive_count = positive_operator_counts.get(op).copied().unwrap_or(0);
        let negative_count = negative_operator_counts.get(op).copied().unwrap_or(0);
        let top_count = top_operator_counts.get(op).copied().unwrap_or(0);
        let bottom_count = bottom_operator_counts.get(op).copied().unwrap_or(0);
        let all_share = *all_count as f64 / evaluated_n as f64;
        let positive_share = positive_count as f64 / positive_n as f64;
        let negative_share = negative_count as f64 / negative_n as f64;
        let top_share = top_count as f64 / top_n as f64;
        let bottom_share = bottom_count as f64 / bottom_n as f64;
        operator_rows.insert(
            op.clone(),
            json!({
                "all_count": all_count,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "top_10_percent_count": top_count,
                "bottom_10_percent_count": bottom_count,
                "all_share": all_share,
                "positive_share": positive_share,
                "negative_share": negative_share,
                "top_10_percent_share": top_share,
                "bottom_10_percent_share": bottom_share,
                "positive_enrichment_over_all": positive_share - all_share,
                "top_enrichment_over_all": top_share - all_share,
                "top_minus_bottom_share": top_share - bottom_share,
                "avg_delta": delta_sums.get(op).copied().unwrap_or(0.0) / *all_count as f64
            }),
        );
    }

    let dominant_top_operator = operator_rows
        .iter()
        .max_by(|(_, a), (_, b)| {
            cmp_f64(
                a["top_enrichment_over_all"].as_f64().unwrap_or(0.0),
                b["top_enrichment_over_all"].as_f64().unwrap_or(0.0),
            )
        })
        .map(|(op, _)| op.clone())
        .unwrap_or_else(|| "none".to_string());
    let dominant_positive_operator = operator_rows
        .iter()
        .max_by(|(_, a), (_, b)| {
            cmp_f64(
                a["positive_enrichment_over_all"].as_f64().unwrap_or(0.0),
                b["positive_enrichment_over_all"].as_f64().unwrap_or(0.0),
            )
        })
        .map(|(op, _)| op.clone())
        .unwrap_or_else(|| "none".to_string());

    json!({
        "scope": "operator_level_common_denominator",
        "edge_level_commonality_available": false,
        "edge_level_commonality_requires_core_trace_extension": true,
        "interpretation_limit": "This report can say which mutation families are shared by winners. It cannot yet say which exact wires or neuron parameters form the shared internal bridge.",
        "dominant_top_enriched_operator": dominant_top_operator,
        "dominant_positive_enriched_operator": dominant_positive_operator,
        "operator_commonality": operator_rows
    })
}

fn build_heatmap_report(
    records: &[HeatmapCandidate],
    before_train: &EvalSummary,
    before_test: &EvalSummary,
    init: &InitConfig,
    clone_steps: usize,
    train_rows: usize,
    test_rows: usize,
) -> serde_json::Value {
    let winners: Vec<&HeatmapCandidate> = records.iter().filter(|r| r.delta_u > 0.0).collect();
    let losers: Vec<&HeatmapCandidate> = records.iter().filter(|r| r.delta_u < 0.0).collect();
    let flat: Vec<&HeatmapCandidate> = records.iter().filter(|r| r.delta_u == 0.0).collect();
    let winner_features = feature_counts(&winners);
    let loser_features = feature_counts(&losers);
    let flat_features = feature_counts(&flat);
    let contrast = contrast_features(
        &winner_features,
        winners.len(),
        &loser_features,
        losers.len(),
    );
    let top_modifier = top_contrast_rows(&contrast, 16);
    let intent = infer_intent(&top_modifier);

    json!({
        "status": "complete",
        "task": "mutation_heatmap_common_network_direction_probe",
        "input_zone": {"start": 0, "end": init.input_end()},
        "output_zone": {"start": init.output_start(), "end": init.neuron_count},
        "overlap_zone": {"start": init.output_start(), "end": init.input_end()},
        "pocket_band_note": "Output zone is split into 9 approximate pocket bands for heatmap orientation; this is not a hard architectural pocket boundary.",
        "before": {
            "train_accuracy": before_train.accuracy,
            "test_accuracy": before_test.accuracy,
            "train_avg_margin": before_train.avg_margin,
            "test_avg_margin": before_test.avg_margin
        },
        "candidate_count": records.len(),
        "clone_steps": clone_steps,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "rough_accuracy_ci_95": {
            "train_random_baseline_half_width": binomial_ci_half_width(1.0 / POCKET_CLASSES as f64, train_rows),
            "test_random_baseline_half_width": binomial_ci_half_width(1.0 / POCKET_CLASSES as f64, test_rows),
            "note": "This is a rough binomial half-width for exact accuracy only. The mutation score uses smooth margin fitness, so use this only as a sanity check for whether row counts are tiny."
        },
        "winner_count": winners.len(),
        "loser_count": losers.len(),
        "flat_count": flat.len(),
        "winner_rate": if records.is_empty() { 0.0 } else { winners.len() as f64 / records.len() as f64 },
        "mutation_detail_scope": {
            "edge_level_details_available": true,
            "theta_channel_details_available": true,
            "projection_weight_details_available": false,
            "core_modified": false
        },
        "winner_feature_counts": winner_features,
        "loser_feature_counts": loser_features,
        "flat_feature_counts": flat_features,
        "winner_minus_loser_contrast": contrast,
        "synthesized_modifier": {
            "meaning": "This is the shared direction suggested by many winning mutations, not a literal single mutation to apply.",
            "top_features": top_modifier,
            "intent_hypothesis": intent
        },
        "artifacts": {
            "raw_candidates": "heatmap_candidates.jsonl",
            "partial_candidates": "heatmap_candidates.partial.jsonl",
            "progress": "progress.jsonl"
        }
    })
}

fn feature_counts(rows: &[&HeatmapCandidate]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for row in rows {
        add_count(&mut counts, format!("operator:{}", row.operator_id));
        add_count(&mut counts, format!("action:{}", row.action_summary));
        collect_detail_features(&row.details, &mut counts);
    }
    counts
}

fn collect_detail_features(detail: &serde_json::Value, counts: &mut BTreeMap<String, usize>) {
    let kind = detail["kind"].as_str().unwrap_or("unknown");
    add_count(counts, format!("kind:{kind}"));
    match kind {
        "clone_walk" => {
            if let Some(steps) = detail["steps"].as_array() {
                for step in steps {
                    if step["mutated"].as_bool().unwrap_or(false) {
                        if let Some(operator) = step["operator_id"].as_str() {
                            add_count(counts, format!("clone_step_operator:{operator}"));
                        }
                        collect_detail_features(&step["details"], counts);
                    }
                }
            }
        }
        "add_edge" | "remove_edge" | "old_edge" | "new_edge" | "loop_edge" => {
            if let Some(exact) = detail["exact_road"].as_str() {
                add_count(counts, format!("exact_road:{kind}:{exact}"));
            }
            if let Some(band) = detail["band_road"].as_str() {
                add_count(counts, format!("band_road:{kind}:{band}"));
                add_count(counts, format!("band_road_any:{band}"));
            }
            if let Some(transition) = detail["transition"].as_str() {
                add_count(counts, format!("transition:{kind}:{transition}"));
                add_count(counts, format!("transition_any:{transition}"));
            }
            if let Some(band) = detail["target_pocket_band"].as_u64() {
                add_count(counts, format!("target_pocket_band:P{}", band + 1));
            }
            if let Some(band) = detail["source_pocket_band"].as_u64() {
                add_count(counts, format!("source_pocket_band:P{}", band + 1));
            }
        }
        "reverse_edge" | "rewire" => {
            if let Some(exact) = detail["old"]["exact_road"].as_str() {
                add_count(counts, format!("{kind}:old_exact:{exact}"));
            }
            if let Some(exact) = detail["new"]["exact_road"].as_str() {
                add_count(counts, format!("{kind}:new_exact:{exact}"));
            }
            if let Some(band) = detail["old"]["band_road"].as_str() {
                add_count(counts, format!("{kind}:old_band:{band}"));
            }
            if let Some(band) = detail["new"]["band_road"].as_str() {
                add_count(counts, format!("{kind}:new_band:{band}"));
                add_count(counts, format!("band_road_any:{band}"));
            }
            if let Some(old_transition) = detail["old"]["transition"].as_str() {
                add_count(counts, format!("{kind}:old:{old_transition}"));
            }
            if let Some(new_transition) = detail["new"]["transition"].as_str() {
                add_count(counts, format!("{kind}:new:{new_transition}"));
                add_count(counts, format!("transition_any:{new_transition}"));
            }
            if let Some(band) = detail["new"]["target_pocket_band"].as_u64() {
                add_count(
                    counts,
                    format!("{kind}:new_target_pocket_band:P{}", band + 1),
                );
                add_count(counts, format!("target_pocket_band:P{}", band + 1));
            }
        }
        "added_loop" => {
            if let Some(edges) = detail["edges"].as_array() {
                for edge in edges {
                    collect_detail_features(edge, counts);
                }
            }
        }
        "theta" => {
            let zone_name = detail["neuron"]["zone"].as_str().unwrap_or("unknown");
            let delta = detail["delta"].as_i64().unwrap_or(0);
            let dir = if delta > 0 {
                "up"
            } else if delta < 0 {
                "down"
            } else {
                "same"
            };
            add_count(counts, format!("theta:{zone_name}:{dir}"));
            if let Some(band) = detail["neuron"]["pocket_band"].as_u64() {
                add_count(counts, format!("theta_pocket_band:P{}:{dir}", band + 1));
            }
        }
        "channel" => {
            let zone_name = detail["neuron"]["zone"].as_str().unwrap_or("unknown");
            let delta = detail["delta"].as_i64().unwrap_or(0);
            let dir = if delta > 0 {
                "up"
            } else if delta < 0 {
                "down"
            } else {
                "same"
            };
            add_count(counts, format!("channel:{zone_name}:{dir}"));
            if let Some(band) = detail["neuron"]["pocket_band"].as_u64() {
                add_count(counts, format!("channel_pocket_band:P{}:{dir}", band + 1));
            }
        }
        _ => {}
    }
}

fn binomial_ci_half_width(p: f64, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    1.96 * ((p * (1.0 - p)) / n as f64).sqrt()
}

fn add_count(counts: &mut BTreeMap<String, usize>, key: String) {
    *counts.entry(key).or_insert(0) += 1;
}

fn contrast_features(
    winners: &BTreeMap<String, usize>,
    winner_n: usize,
    losers: &BTreeMap<String, usize>,
    loser_n: usize,
) -> serde_json::Value {
    let mut keys: Vec<String> = winners.keys().chain(losers.keys()).cloned().collect();
    keys.sort();
    keys.dedup();
    let mut rows = Vec::new();
    let wn = winner_n.max(1) as f64;
    let ln = loser_n.max(1) as f64;
    for key in keys {
        let wc = winners.get(&key).copied().unwrap_or(0);
        let lc = losers.get(&key).copied().unwrap_or(0);
        let winner_share = wc as f64 / wn;
        let loser_share = lc as f64 / ln;
        rows.push(json!({
            "feature": key,
            "winner_count": wc,
            "loser_count": lc,
            "winner_share": winner_share,
            "loser_share": loser_share,
            "winner_minus_loser_share": winner_share - loser_share
        }));
    }
    rows.sort_by(|a, b| {
        cmp_f64(
            b["winner_minus_loser_share"].as_f64().unwrap_or(0.0),
            a["winner_minus_loser_share"].as_f64().unwrap_or(0.0),
        )
    });
    serde_json::Value::Array(rows)
}

fn top_contrast_rows(contrast: &serde_json::Value, n: usize) -> Vec<serde_json::Value> {
    contrast
        .as_array()
        .map(|rows| rows.iter().take(n).cloned().collect())
        .unwrap_or_default()
}

fn infer_intent(top_rows: &[serde_json::Value]) -> String {
    let features: Vec<String> = top_rows
        .iter()
        .filter_map(|row| row["feature"].as_str().map(|s| s.to_string()))
        .collect();
    let joined = features.join(" | ");
    if joined.contains("reverse_edge:new:overlap->output")
        || joined.contains("transition_any:overlap->output")
    {
        "Winning mutations look like they are trying to route overlap/hub activity toward the output pocket side.".to_string()
    } else if joined.contains("reverse") {
        "Winning mutations are dominated by direction changes, suggesting existing roads may be pointed the wrong way.".to_string()
    } else if joined.contains("add_edge") {
        "Winning mutations look like they are adding missing roads rather than mainly tuning thresholds.".to_string()
    } else if joined.contains("remove_edge") {
        "Winning mutations look like pruning/noise removal is helping more than adding roads."
            .to_string()
    } else if joined.contains("theta") {
        "Winning mutations look like threshold gating changes are more important than new roads."
            .to_string()
    } else {
        "No single clear intent; the winner set is still mixed.".to_string()
    }
}

fn write_heatmap_report(path: &Path, report: &serde_json::Value) -> io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "# VRAXION Mutation Heatmap Probe")?;
    writeln!(file)?;
    writeln!(file, "This report asks what kind of network the successful mutations are collectively trying to build.")?;
    writeln!(file)?;
    writeln!(
        file,
        "- candidate_count = {}",
        report["candidate_count"].as_u64().unwrap_or(0)
    )?;
    writeln!(
        file,
        "- winner_count = {}",
        report["winner_count"].as_u64().unwrap_or(0)
    )?;
    writeln!(
        file,
        "- loser_count = {}",
        report["loser_count"].as_u64().unwrap_or(0)
    )?;
    writeln!(
        file,
        "- winner_rate = {:.4}",
        report["winner_rate"].as_f64().unwrap_or(0.0)
    )?;
    writeln!(
        file,
        "- intent_hypothesis = {}",
        report["synthesized_modifier"]["intent_hypothesis"]
            .as_str()
            .unwrap_or("unknown")
    )?;
    writeln!(file)?;
    writeln!(file, "This is a shared-direction heatmap, not a literal final network and not a broad capability claim.")?;
    Ok(())
}

fn write_mask_report(path: &Path, report: &serde_json::Value) -> io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "# VRAXION Iterative Bad-Road Mask Probe")?;
    writeln!(file)?;
    writeln!(file, "This report tests whether removing roads that temporarily hurt smooth train fitness leaves a cleaner network.")?;
    writeln!(file)?;
    writeln!(
        file,
        "- rounds = {}",
        report["rounds"].as_u64().unwrap_or(0)
    )?;
    writeln!(
        file,
        "- mask_candidates_per_round = {}",
        report["mask_candidates_per_round"].as_u64().unwrap_or(0)
    )?;
    writeln!(
        file,
        "- total_masked_edges = {}",
        report["total_masked_edges"].as_u64().unwrap_or(0)
    )?;
    writeln!(
        file,
        "- train_accuracy_delta = {:.4}",
        report["delta"]["train_accuracy"].as_f64().unwrap_or(0.0)
    )?;
    writeln!(
        file,
        "- test_accuracy_delta = {:.4}",
        report["delta"]["test_accuracy"].as_f64().unwrap_or(0.0)
    )?;
    writeln!(
        file,
        "- test_margin_delta = {:.4}",
        report["delta"]["test_avg_margin"].as_f64().unwrap_or(0.0)
    )?;
    writeln!(file)?;
    writeln!(
        file,
        "This is a pruning/masking probe, not a final locked network proof."
    )?;
    Ok(())
}

fn write_ga_report(path: &Path, summary: &serde_json::Value) -> io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "# VRAXION Persistent Population GA Smoke")?;
    writeln!(file)?;
    writeln!(file, "This report tests persistent independent individuals instead of one parent plus temporary scouts.")?;
    writeln!(file)?;
    writeln!(
        file,
        "- population = {}",
        summary["population"].as_u64().unwrap_or(0)
    )?;
    writeln!(
        file,
        "- generations = {}",
        summary["generations"].as_u64().unwrap_or(0)
    )?;
    writeln!(
        file,
        "- elite_count = {}",
        summary["elite_count"].as_u64().unwrap_or(0)
    )?;
    writeln!(
        file,
        "- mutation_steps_per_child = {}",
        summary["mutation_steps_per_child"].as_u64().unwrap_or(0)
    )?;
    writeln!(
        file,
        "- final_best_test_accuracy = {:.4}",
        summary["final_best"]["test_accuracy"]
            .as_f64()
            .unwrap_or(0.0)
    )?;
    writeln!(
        file,
        "- test_accuracy_delta_from_single_baseline = {:.4}",
        summary["test_accuracy_delta_from_single_baseline"]
            .as_f64()
            .unwrap_or(0.0)
    )?;
    writeln!(file)?;
    writeln!(
        file,
        "This is a GA smoke for pocket routing, not a broad capability claim."
    )?;
    Ok(())
}

fn parse_value<T: std::str::FromStr>(name: &str, value: Option<String>) -> io::Result<T>
where
    T::Err: std::fmt::Display,
{
    value
        .ok_or_else(|| arg_err(&format!("{name} needs a value")))?
        .parse::<T>()
        .map_err(|e| arg_err(&format!("{name} parse error: {e}")))
}

fn arg_err(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.to_string())
}

fn make_samples(split: &str, rows: usize, rng: &mut StdRng, task_mode: &str) -> Vec<Sample> {
    let mut samples = Vec::with_capacity(rows);
    for idx in 0..rows {
        if task_mode == "symbol_match_only" {
            let wanted_symbol = rng.gen_range(0..9);
            let is_match = idx % 2 == 0;
            let candidate_symbol = if is_match {
                wanted_symbol
            } else {
                let shift = rng.gen_range(1..9);
                (wanted_symbol + shift) % 9
            };
            let mut pockets: Vec<usize> = (0..9).collect();
            pockets[0] = candidate_symbol;
            let expected_pocket = if is_match { 0 } else { 1 };
            let grid = Vec::new();
            let target_hint = Some(wanted_symbol);
            let prompt_text = prompt_text(&grid, &pockets, target_hint, task_mode);
            samples.push(Sample {
                row_id: format!("{split}_{idx:05}"),
                split: split.to_string(),
                family: "SYMBOL_MATCH_ONLY".to_string(),
                family_id: 0,
                grid,
                pockets,
                expected_symbol: wanted_symbol,
                expected_pocket,
                target_hint,
                prompt_text,
            });
            continue;
        }

        if task_mode == "pocket_only_lookup"
            || task_mode == "pocket_id_hint"
            || task_mode == "pocket_id_only"
            || task_mode == "pocket_match_hint"
        {
            let expected_symbol = rng.gen_range(0..9);
            let mut pockets: Vec<usize> = (0..9).collect();
            pockets.shuffle(rng);
            let expected_pocket = if task_mode == "pocket_id_hint" || task_mode == "pocket_id_only"
            {
                rng.gen_range(0..9)
            } else {
                pockets
                    .iter()
                    .position(|&sym| sym == expected_symbol)
                    .expect("all symbols exist in pockets")
            };
            let expected_symbol = pockets[expected_pocket];
            let grid = Vec::new();
            let target_hint = if task_mode == "pocket_id_hint" || task_mode == "pocket_id_only" {
                Some(expected_pocket)
            } else {
                Some(expected_symbol)
            };
            let prompt_text = prompt_text(&grid, &pockets, target_hint, task_mode);
            samples.push(Sample {
                row_id: format!("{split}_{idx:05}"),
                split: split.to_string(),
                family: if task_mode == "pocket_id_only" {
                    "POCKET_ID_ONLY".to_string()
                } else if task_mode == "pocket_id_hint" {
                    "POCKET_ID_HINT".to_string()
                } else if task_mode == "pocket_match_hint" {
                    "POCKET_MATCH_HINT".to_string()
                } else {
                    "POCKET_ONLY_LOOKUP".to_string()
                },
                family_id: 0,
                grid,
                pockets,
                expected_symbol,
                expected_pocket,
                target_hint,
                prompt_text,
            });
            continue;
        }

        let family_id = idx % FAMILY_COUNT;
        let (family, mut grid) = make_grid(family_id, rng);
        let missing = rng.gen_range(0..9);
        let expected_symbol = grid[missing].expect("generated grid has all symbols before masking");
        grid[missing] = None;

        let mut pockets: Vec<usize> = (0..9).collect();
        if task_mode != "pattern_fixed_pocket" {
            pockets.shuffle(rng);
        }
        let expected_pocket = pockets
            .iter()
            .position(|&sym| sym == expected_symbol)
            .expect("all symbols exist in pockets");
        let target_hint = if task_mode == "pocket_lookup" {
            Some(expected_symbol)
        } else if task_mode == "full_match_hint" || task_mode == "pocket_id_grid_noise" {
            Some(expected_pocket)
        } else {
            None
        };
        let prompt_text = prompt_text(&grid, &pockets, target_hint, task_mode);

        samples.push(Sample {
            row_id: format!("{split}_{idx:05}"),
            split: split.to_string(),
            family: if task_mode == "full_match_hint" {
                "FULL_MATCH_HINT".to_string()
            } else if task_mode == "pocket_id_grid_noise" {
                "POCKET_ID_GRID_NOISE".to_string()
            } else {
                family.to_string()
            },
            family_id,
            grid,
            pockets,
            expected_symbol,
            expected_pocket,
            target_hint,
            prompt_text,
        });
    }
    samples
}

fn make_grid(family_id: usize, rng: &mut StdRng) -> (&'static str, Vec<Option<usize>>) {
    let mut symbols: Vec<usize> = (0..9).collect();
    symbols.shuffle(rng);
    let a = symbols[0];
    let b = symbols[1];
    let c = symbols[2];
    let d = symbols[3];
    let e = symbols[4];
    let f = symbols[5];
    let g = symbols[6];
    let h = symbols[7];
    let i = symbols[8];

    match family_id {
        0 => (
            "ROW_ROTATE_ABC",
            vec![
                Some(a),
                Some(b),
                Some(c),
                Some(b),
                Some(c),
                Some(a),
                Some(c),
                Some(a),
                Some(b),
            ],
        ),
        1 => (
            "COLUMN_ROTATE_ABC",
            vec![
                Some(a),
                Some(b),
                Some(c),
                Some(c),
                Some(a),
                Some(b),
                Some(b),
                Some(c),
                Some(a),
            ],
        ),
        2 => (
            "ROW_MIRROR_XYX",
            vec![
                Some(a),
                Some(d),
                Some(a),
                Some(b),
                Some(e),
                Some(b),
                Some(c),
                Some(f),
                Some(c),
            ],
        ),
        _ => (
            "ROW_PAIR_STEP",
            vec![
                Some(a),
                Some(a),
                Some(g),
                Some(b),
                Some(b),
                Some(h),
                Some(c),
                Some(c),
                Some(i),
            ],
        ),
    }
}

fn prompt_text(
    grid: &[Option<usize>],
    pockets: &[usize],
    target_hint: Option<usize>,
    task_mode: &str,
) -> String {
    let mut s = String::new();
    if task_mode != "pocket_only_lookup"
        && task_mode != "pocket_id_hint"
        && task_mode != "pocket_id_only"
        && task_mode != "pocket_match_hint"
        && task_mode != "symbol_match_only"
    {
        s.push_str("GRID:\n");
        for row in 0..3 {
            for col in 0..3 {
                if col > 0 {
                    s.push(' ');
                }
                let cell = row * 3 + col;
                match grid[cell] {
                    Some(sym) => s.push(SYMBOLS[sym]),
                    None => s.push('?'),
                }
            }
            s.push('\n');
        }
    }
    if let Some(sym) = target_hint {
        if task_mode == "full_match_hint" {
            s.push_str(&format!("MATCH_HINT=P{}\n", sym + 1));
        } else if task_mode == "pocket_id_grid_noise" {
            s.push_str(&format!("WANTED_POCKET=P{}\n", sym + 1));
        } else if task_mode == "symbol_match_only" {
            s.push_str(&format!("WANTED={}\n", SYMBOLS[sym]));
            s.push_str(&format!("CANDIDATE={}\n", SYMBOLS[pockets[0]]));
            s.push_str("YES=P1 NO=P2\n");
        } else if task_mode == "pocket_id_hint" || task_mode == "pocket_id_only" {
            s.push_str(&format!("WANTED_POCKET=P{}\n", sym + 1));
        } else {
            s.push_str(&format!("WANTED={}\n", SYMBOLS[sym]));
        }
    }
    if task_mode != "pocket_id_only" && task_mode != "symbol_match_only" {
        s.push_str("POCKETS:\n");
        for (idx, &sym) in pockets.iter().enumerate() {
            if idx > 0 {
                s.push(' ');
            }
            s.push_str(&format!("P{}={}", idx + 1, SYMBOLS[sym]));
        }
        s.push('\n');
    }
    s.push_str("OUTPUT:\n");
    s
}

fn encode(sample: &Sample, h: usize) -> Vec<i32> {
    let mut input = vec![0i32; h];
    if !sample.grid.is_empty() {
        for (cell, symbol) in sample.grid.iter().enumerate() {
            let grid_strength = if sample.family == "POCKET_ID_GRID_NOISE" {
                1
            } else {
                INPUT_STRENGTH
            };
            match symbol {
                Some(sym) => {
                    set_feature_strength(&mut input, GRID_BASE + cell * 9 + sym, grid_strength)
                }
                None => set_feature_strength(&mut input, MISSING_BASE + cell, grid_strength),
            }
        }
    }
    if sample.family != "POCKET_ID_ONLY"
        && sample.family != "POCKET_MATCH_HINT"
        && sample.family != "SYMBOL_MATCH_ONLY"
        && sample.family != "FULL_MATCH_HINT"
        && sample.family != "POCKET_ID_GRID_NOISE"
    {
        for (pocket_idx, &symbol) in sample.pockets.iter().enumerate() {
            set_feature(&mut input, POCKET_BASE + pocket_idx * 9 + symbol);
        }
    }
    if sample.family == "FULL_MATCH_HINT" {
        set_feature_strength(
            &mut input,
            POCKET_BASE + sample.expected_pocket,
            INPUT_STRENGTH * 16,
        );
    }
    if sample.family == "SYMBOL_MATCH_ONLY" {
        if let Some(wanted) = sample.target_hint {
            set_feature_strength(&mut input, TARGET_BASE + wanted, INPUT_STRENGTH * 4);
        }
        set_feature_strength(
            &mut input,
            POCKET_BASE + sample.pockets[0],
            INPUT_STRENGTH * 4,
        );
    }
    if sample.family == "POCKET_MATCH_HINT" {
        set_feature_strength(
            &mut input,
            POCKET_BASE + sample.expected_pocket,
            INPUT_STRENGTH * 4,
        );
    }
    if sample.family != "POCKET_ONLY_LOOKUP"
        && sample.family != "POCKET_ID_HINT"
        && sample.family != "POCKET_MATCH_HINT"
        && sample.family != "POCKET_ID_ONLY"
        && sample.family != "SYMBOL_MATCH_ONLY"
        && sample.family != "FULL_MATCH_HINT"
        && sample.family != "POCKET_ID_GRID_NOISE"
    {
        set_feature(&mut input, FAMILY_BASE + sample.family_id);
    }
    if let Some(sym) = sample.target_hint {
        if sample.family == "POCKET_ID_GRID_NOISE" {
            set_feature_strength(&mut input, TARGET_BASE + sym, INPUT_STRENGTH * 16);
        } else if sample.family == "FULL_MATCH_HINT" {
            // Already encoded above as the precomputed matching pocket lamp.
        } else if sample.family == "SYMBOL_MATCH_ONLY" {
            // Already encoded above as the wanted side of the equality probe.
        } else if sample.family == "POCKET_MATCH_HINT" {
            // This probe gives the model only the precomputed matching pocket lamp.
        } else if sample.family == "POCKET_ID_HINT" || sample.family == "POCKET_ONLY_LOOKUP" {
            set_feature_strength(&mut input, TARGET_BASE + sym, INPUT_STRENGTH * 4);
        } else {
            set_feature(&mut input, TARGET_BASE + sym);
        }
    }
    input
}

fn set_feature(input: &mut [i32], idx: usize) {
    set_feature_strength(input, idx, INPUT_STRENGTH);
}

fn set_feature_strength(input: &mut [i32], idx: usize, strength: i32) {
    if idx < input.len() {
        input[idx] = strength;
    }
}

fn smooth_fitness(
    net: &mut Network,
    proj: &Int8Projection,
    data: &[Sample],
    init: &InitConfig,
) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    for sample in data {
        let (_pred, scores, _charges) = predict(net, proj, sample, init);
        let correct = scores[sample.expected_pocket] as f64;
        let best_wrong = scores
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx != sample.expected_pocket)
            .map(|(_, score)| *score as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let margin = (correct - best_wrong) / 512.0;
        total += 1.0 / (1.0 + (-margin).exp());
    }
    total / data.len() as f64
}

fn predict(
    net: &mut Network,
    proj: &Int8Projection,
    sample: &Sample,
    init: &InitConfig,
) -> (usize, Vec<i32>, Vec<u8>) {
    net.reset();
    let input = encode(sample, init.neuron_count);
    net.propagate(&input, &init.propagation)
        .expect("propagation must succeed for generated fixed-size input");
    let charges = net.charge_vec(init.output_start()..init.neuron_count);
    let scores = proj.raw_scores(&charges);
    let pred = scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    (pred, scores, charges)
}

fn eval_dataset(
    split: &str,
    net: &mut Network,
    proj: &Int8Projection,
    data: &[Sample],
    init: &InitConfig,
    row_output_path: Option<&Path>,
) -> io::Result<EvalSummary> {
    let mut correct = 0usize;
    let mut margin_sum = 0.0;
    let mut family_total: BTreeMap<String, usize> = BTreeMap::new();
    let mut family_correct: BTreeMap<String, usize> = BTreeMap::new();
    let mut row_file = if let Some(path) = row_output_path {
        Some(OpenOptions::new().create(true).append(true).open(path)?)
    } else {
        None
    };

    for sample in data {
        let (pred, scores, _charges) = predict(net, proj, sample, init);
        let row_correct = pred == sample.expected_pocket;
        if row_correct {
            correct += 1;
            *family_correct.entry(sample.family.clone()).or_insert(0) += 1;
        }
        *family_total.entry(sample.family.clone()).or_insert(0) += 1;
        let correct_score = scores[sample.expected_pocket] as f64;
        let best_wrong = scores
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx != sample.expected_pocket)
            .map(|(_, score)| *score as f64)
            .fold(f64::NEG_INFINITY, f64::max);
        let margin = correct_score - best_wrong;
        margin_sum += margin;

        if let Some(file) = row_file.as_mut() {
            let row = RowPrediction {
                row_id: sample.row_id.clone(),
                split: split.to_string(),
                family: sample.family.clone(),
                prompt_text: sample.prompt_text.clone(),
                expected_symbol: SYMBOLS[sample.expected_symbol].to_string(),
                expected_pocket: format!("P{}", sample.expected_pocket + 1),
                selected_pocket: format!("P{}", pred + 1),
                selected_symbol_in_pocket: SYMBOLS[sample.pockets[pred]].to_string(),
                correct: row_correct,
                margin,
                scores,
            };
            writeln!(file, "{}", serde_json::to_string(&row)?)?;
        }
    }
    if let Some(file) = row_file.as_mut() {
        file.flush()?;
    }

    let mut family_accuracy = BTreeMap::new();
    for (family, total) in family_total {
        let c = family_correct.get(&family).copied().unwrap_or(0);
        family_accuracy.insert(family, c as f64 / total as f64);
    }

    Ok(EvalSummary {
        split: split.to_string(),
        rows: data.len(),
        accuracy: if data.is_empty() {
            0.0
        } else {
            correct as f64 / data.len() as f64
        },
        avg_margin: if data.is_empty() {
            0.0
        } else {
            margin_sum / data.len() as f64
        },
        family_accuracy,
    })
}

fn crystallize(
    net: &mut Network,
    proj: &Int8Projection,
    train: &[Sample],
    init: &InitConfig,
    sample_count: usize,
    rng: &mut StdRng,
) -> usize {
    let baseline = smooth_fitness(net, proj, train, init);
    let mut pruned = 0usize;
    let tests = sample_count.min(net.edge_count());
    for _ in 0..tests {
        if net.edge_count() < 50 {
            break;
        }
        let edge_idx = rng.gen_range(0..net.edge_count());
        let edge = net.graph().iter_edges().nth(edge_idx);
        if let Some(edge) = edge {
            let source = edge.source;
            let target = edge.target;
            net.graph_mut().remove_edge(source, target);
            let new_fitness = smooth_fitness(net, proj, train, init);
            if new_fitness >= baseline - 0.0001 {
                pruned += 1;
            } else {
                net.graph_mut().add_edge(source, target);
            }
        }
    }
    pruned
}

fn write_json(path: &Path, value: &serde_json::Value) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    let file = File::create(&tmp)?;
    serde_json::to_writer_pretty(file, value)?;
    fs::rename(tmp, path)?;
    Ok(())
}

fn write_jsonl<T: Serialize>(path: &Path, rows: &[T]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create(path)?;
    for row in rows {
        writeln!(file, "{}", serde_json::to_string(row)?)?;
    }
    file.flush()?;
    Ok(())
}

fn append_jsonl_value<T: Serialize>(path: &Path, row: &T) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{}", serde_json::to_string(row)?)?;
    file.flush()?;
    Ok(())
}

fn append_progress(
    path: &Path,
    event: &str,
    elapsed_sec: f64,
    data: serde_json::Value,
) -> io::Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(
        file,
        "{}",
        serde_json::to_string(&json!({
            "time_unix_ms": unix_ms(),
            "elapsed_sec": elapsed_sec,
            "event": event,
            "data": data
        }))?
    )?;
    file.flush()?;
    Ok(())
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn count_outcomes(path: &Path, needle: &str) -> io::Result<usize> {
    let text = fs::read_to_string(path)?;
    Ok(text.lines().filter(|line| line.contains(needle)).count())
}

fn write_report(path: &Path, summary: &serde_json::Value) -> io::Result<()> {
    let mut file = File::create(path)?;
    let train_acc = summary["train"]["accuracy"].as_f64().unwrap_or(0.0);
    let test_acc = summary["test"]["accuracy"].as_f64().unwrap_or(0.0);
    let baseline = summary["random_baseline_accuracy"].as_f64().unwrap_or(0.0);
    writeln!(file, "# VRAXION Raven Pocket Smoke")?;
    writeln!(file)?;
    writeln!(file, "This smoke uses the official VRAXION Rust network, Int8Projection, canonical evolution operators, and optional crystallize pruning.")?;
    writeln!(file)?;
    writeln!(file, "Task: infer the missing grid symbol, then select the shuffled pocket that contains that symbol.")?;
    writeln!(file)?;
    writeln!(file, "- train_accuracy = {:.4}", train_acc)?;
    writeln!(file, "- test_accuracy = {:.4}", test_acc)?;
    writeln!(file, "- random_baseline_accuracy = {:.4}", baseline)?;
    writeln!(file, "- raw_predictions_file = row_level_predictions.jsonl")?;
    writeln!(file)?;
    writeln!(
        file,
        "This is not natural-language reasoning and not a broad assistant capability test."
    )?;
    Ok(())
}

fn write_compass_report(path: &Path, summary: &serde_json::Value) -> io::Result<()> {
    let mut file = File::create(path)?;
    let before_train = summary["before"]["train_accuracy"].as_f64().unwrap_or(0.0);
    let before_test = summary["before"]["test_accuracy"].as_f64().unwrap_or(0.0);
    let after_train = summary["after_selected_candidate"]["train_accuracy"]
        .as_f64()
        .unwrap_or(0.0);
    let after_test = summary["after_selected_candidate"]["test_accuracy"]
        .as_f64()
        .unwrap_or(0.0);
    let positive_rate = summary["direction_signal"]["positive_delta_rate"]
        .as_f64()
        .unwrap_or(0.0);
    let best_delta = summary["direction_signal"]["best_delta"]
        .as_f64()
        .unwrap_or(0.0);
    writeln!(file, "# VRAXION Mutation Compass Smoke")?;
    writeln!(file)?;
    writeln!(file, "This smoke asks whether a cloud of mutations around one parent shows a usable direction signal.")?;
    writeln!(file)?;
    writeln!(file, "- before_train_accuracy = {:.4}", before_train)?;
    writeln!(file, "- after_train_accuracy = {:.4}", after_train)?;
    writeln!(file, "- before_test_accuracy = {:.4}", before_test)?;
    writeln!(file, "- after_test_accuracy = {:.4}", after_test)?;
    writeln!(file, "- positive_delta_rate = {:.4}", positive_rate)?;
    writeln!(file, "- best_delta = {:.6}", best_delta)?;
    writeln!(file, "- candidate_records = compass_candidates.jsonl")?;
    writeln!(file)?;
    writeln!(
        file,
        "This is an orientation probe, not a capability claim."
    )?;
    Ok(())
}

#[allow(dead_code)]
fn cmp_f64(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}
