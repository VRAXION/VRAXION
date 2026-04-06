//! Canonical public beta runner for Rust language evolution.
//!
//! Default story:
//! - H=256 phi-overlap geometry
//! - chain-50 init
//! - SDR 20% input
//! - learnable int8 projection
//! - density-capped paired eval
//! - **smooth cosine-bigram fitness** (proven +2.6pp peak vs stepwise argmax)
//! - **1+9 jackpot** (9 candidate mutations per step, best wins — Python parity at 24.6%)
//!
//! Run:
//! `cargo run --release --example evolve_language -- <corpus-path>`

use instnct_core::{
    build_network, evolution_step_jackpot, InitConfig, Int8Projection, Network, PropagationConfig,
    SdrTable, StepOutcome,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::Serialize;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const CHARS: usize = 27; // a-z (0..25) + space (26)
const SDR_ACTIVE_PCT: usize = 20;
const DEFAULT_CORPUS_PATH: &str =
    "S:/AI/work/VRAXION_DEV/instnct/data/traindat/fineweb_edu.traindat";
const DEFAULT_STEPS: usize = 30_000;
const DEFAULT_SEED_COUNT: usize = 6;
const DEFAULT_SEED_BASE: u64 = 42;
const DEFAULT_FULL_LEN: usize = 2_000;
const SEED_STRIDE: u64 = 1_000;
const PROGRESS_INTERVAL: usize = 5_000;

// ---------------------------------------------------------------------------
// Bigram table + smooth fitness helpers
// ---------------------------------------------------------------------------

/// 27×27 bigram probability table: bigram\[i\]\[j\] = P(next=j | current=i).
type BigramTable = Vec<[f64; CHARS]>;

fn build_bigram_table(corpus: &[u8]) -> BigramTable {
    let mut counts = vec![[0u64; CHARS]; CHARS];
    for pair in corpus.windows(2) {
        counts[pair[0] as usize][pair[1] as usize] += 1;
    }
    let mut bigram = vec![[0.0f64; CHARS]; CHARS];
    for (i, row) in counts.iter().enumerate() {
        let total: u64 = row.iter().sum();
        if total > 0 {
            for (j, &c) in row.iter().enumerate() {
                bigram[i][j] = c as f64 / total as f64;
            }
        }
    }
    bigram
}

fn softmax_27(scores: &[i32]) -> [f64; CHARS] {
    let max = scores.iter().copied().max().unwrap_or(0) as f64;
    let mut out = [0.0f64; CHARS];
    let mut sum = 0.0f64;
    for (i, &s) in scores.iter().enumerate() {
        let e = ((s as f64) - max).exp();
        out[i] = e;
        sum += e;
    }
    if sum < 1e-30 {
        out.fill(1.0 / CHARS as f64);
    } else {
        for v in out.iter_mut() {
            *v /= sum;
        }
    }
    out
}

fn cosine_27(a: &[f64; CHARS], b: &[f64; CHARS]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..CHARS {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    dot / denom
}

#[derive(Clone, Debug)]
struct RunnerConfig {
    corpus_path: String,
    steps: usize,
    seed_count: usize,
    seed_base: u64,
    full_len: usize,
    report_dir: Option<PathBuf>,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            corpus_path: DEFAULT_CORPUS_PATH.to_string(),
            steps: DEFAULT_STEPS,
            seed_count: DEFAULT_SEED_COUNT,
            seed_base: DEFAULT_SEED_BASE,
            full_len: DEFAULT_FULL_LEN,
            report_dir: None,
        }
    }
}

#[derive(Clone, Serialize)]
struct EvolutionResult {
    seed: u64,
    final_accuracy: f64,
    peak_accuracy: f64,
    edge_count: usize,
    accept_rate: f64,
}

#[derive(Serialize)]
struct Baselines {
    random_accuracy: f64,
    frequency_accuracy: f64,
    frequency_char: char,
    bigram_accuracy: f64,
}

#[derive(Serialize)]
struct SummaryMetrics {
    mean_accuracy: f64,
    min_accuracy: f64,
    max_accuracy: f64,
    spread_pp: f64,
    best_peak_accuracy: f64,
    mean_accept_rate: f64,
}

#[derive(Serialize)]
struct EnvReport {
    package_name: &'static str,
    package_version: &'static str,
    os: &'static str,
    arch: &'static str,
    current_dir: String,
    current_exe: String,
    corpus_path: String,
    steps: usize,
    seed_count: usize,
    seed_base: u64,
    seed_stride: u64,
    full_len: usize,
    neuron_count: usize,
    phi_dim: usize,
    chain_count: usize,
    input_end: usize,
    output_start: usize,
    edge_cap: usize,
    ticks_per_token: usize,
    input_duration_ticks: usize,
    decay_interval_ticks: usize,
    use_refractory: bool,
}

#[derive(Serialize)]
struct MetricsReport {
    baselines: Baselines,
    summary: SummaryMetrics,
    seeds: Vec<EvolutionResult>,
    runtime_seconds: f64,
}

fn print_usage() {
    println!(
        "Usage: cargo run --release --example evolve_language -- [corpus-path] [--steps N] [--seed-count N] [--seed-base N] [--full-len N] [--report-dir DIR]"
    );
}

fn parse_usize_flag(flag: &str, value: Option<String>) -> usize {
    value
        .unwrap_or_else(|| panic!("missing value for {flag}"))
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("invalid usize for {flag}"))
}

fn parse_u64_flag(flag: &str, value: Option<String>) -> u64 {
    value
        .unwrap_or_else(|| panic!("missing value for {flag}"))
        .parse::<u64>()
        .unwrap_or_else(|_| panic!("invalid u64 for {flag}"))
}

fn parse_cli() -> RunnerConfig {
    let mut cfg = RunnerConfig::default();
    let mut corpus_path: Option<String> = None;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            "--steps" => cfg.steps = parse_usize_flag("--steps", args.next()),
            "--seed-count" => cfg.seed_count = parse_usize_flag("--seed-count", args.next()),
            "--seed-base" => cfg.seed_base = parse_u64_flag("--seed-base", args.next()),
            "--full-len" => cfg.full_len = parse_usize_flag("--full-len", args.next()),
            "--report-dir" => {
                let value = args
                    .next()
                    .unwrap_or_else(|| panic!("missing value for --report-dir"));
                cfg.report_dir = Some(PathBuf::from(value));
            }
            _ if arg.starts_with("--") => panic!("unknown flag: {arg}"),
            _ => {
                assert!(
                    corpus_path.is_none(),
                    "unexpected extra positional argument: {arg}"
                );
                corpus_path = Some(arg);
            }
        }
    }

    if let Some(path) = corpus_path {
        cfg.corpus_path = path;
    }
    assert!(cfg.steps > 0, "--steps must be > 0");
    assert!(cfg.seed_count > 0, "--seed-count must be > 0");
    assert!(cfg.full_len > 0, "--full-len must be > 0");
    cfg
}

fn load_corpus(path: &str) -> Vec<u8> {
    let raw = fs::read(path).expect("cannot read corpus file");
    raw.iter()
        .filter_map(|&b| {
            if b.is_ascii_lowercase() {
                Some(b - b'a')
            } else if b.is_ascii_uppercase() {
                Some(b.to_ascii_lowercase() - b'a')
            } else if b == b' ' || b == b'\n' || b == b'\t' {
                Some(26)
            } else {
                None
            }
        })
        .collect()
}

fn sample_eval_offset(corpus_len: usize, len: usize, rng: &mut StdRng) -> Option<usize> {
    if corpus_len <= len {
        return None;
    }
    let max_offset = corpus_len - len - 1;
    Some(rng.gen_range(0..=max_offset))
}

fn bar(val: f64, max_val: f64, width: usize) -> String {
    let filled = ((val / max_val) * width as f64).round().min(width as f64) as usize;
    format!("{}{}", "#".repeat(filled), ".".repeat(width - filled))
}

fn char_label(c: u8) -> char {
    if c < 26 {
        (b'a' + c) as char
    } else {
        '_'
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_accuracy(
    net: &mut Network,
    projection: &Int8Projection,
    corpus: &[u8],
    len: usize,
    rng: &mut StdRng,
    sdr: &SdrTable,
    config: &PropagationConfig,
    output_start: usize,
    neuron_count: usize,
) -> f64 {
    let Some(off) = sample_eval_offset(corpus.len(), len, rng) else {
        return 0.0;
    };

    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), config).unwrap();
        if projection.predict(&net.charge()[output_start..neuron_count]) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

/// Smooth fitness: mean cosine similarity between predicted distribution
/// and the true bigram distribution P(next | current).
///
/// Unlike binary argmax accuracy, this gives continuous feedback —
/// a mutation that shifts the output distribution TOWARD the correct
/// answer is rewarded even if argmax doesn't flip yet.
/// Proven +2.6pp peak over stepwise (21.7% vs 19.1%, A/B test 2026-04-06).
#[allow(clippy::too_many_arguments)]
fn eval_smooth(
    net: &mut Network,
    projection: &Int8Projection,
    corpus: &[u8],
    len: usize,
    rng: &mut StdRng,
    sdr: &SdrTable,
    config: &PropagationConfig,
    output_start: usize,
    neuron_count: usize,
    bigram: &BigramTable,
) -> f64 {
    let Some(off) = sample_eval_offset(corpus.len(), len, rng) else {
        return 0.0;
    };

    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut total_cos = 0.0f64;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), config).unwrap();
        let scores = projection.raw_scores(&net.charge()[output_start..neuron_count]);
        let probs = softmax_27(&scores);
        let target = &bigram[seg[i] as usize];
        total_cos += cosine_27(&probs, target);
    }
    total_cos / len as f64
}

fn run_evolution(
    cfg: &RunnerConfig,
    seed: u64,
    corpus: &[u8],
    init: &InitConfig,
    bigram: &BigramTable,
) -> EvolutionResult {
    let output_start = init.output_start();
    let neuron_count = init.neuron_count;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = build_network(init, &mut rng);
    let mut projection = Int8Projection::new(
        init.phi_dim,
        CHARS,
        &mut StdRng::seed_from_u64(seed + 200),
    );
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let sdr = SdrTable::new(
        CHARS,
        neuron_count,
        init.input_end(),
        SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(seed + 100),
    )
    .unwrap();

    let mut accepted = 0u32;
    let mut rejected = 0u32;
    let evo_config = init.evolution_config();
    let mut peak_accuracy = 0.0f64;

    for step in 0..cfg.steps {
        // Smooth cosine-bigram fitness + 1+9 jackpot drives evolution.
        // Argmax accuracy is measured separately for reporting only.
        let outcome = evolution_step_jackpot(
            &mut net,
            &mut projection,
            &mut rng,
            &mut eval_rng,
            |net, proj, eval_rng| {
                eval_smooth(
                    net,
                    proj,
                    corpus,
                    100,
                    eval_rng,
                    &sdr,
                    &init.propagation,
                    output_start,
                    neuron_count,
                    bigram,
                )
            },
            &evo_config,
            9, // 9 candidates per step (Python parity)
        );
        match outcome {
            StepOutcome::Accepted => accepted += 1,
            StepOutcome::Rejected => rejected += 1,
            StepOutcome::Skipped => {}
        }

        if (step + 1) % PROGRESS_INTERVAL == 0 {
            // Report true argmax accuracy (the metric we care about)
            let full = eval_accuracy(
                &mut net,
                &projection,
                corpus,
                cfg.full_len,
                &mut eval_rng,
                &sdr,
                &init.propagation,
                output_start,
                neuron_count,
            );
            peak_accuracy = peak_accuracy.max(full);
            let tot = accepted + rejected;
            let rate = if tot > 0 {
                accepted as f64 / tot as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "  [seed={seed}] step {:>5}: |{}| {:.1}%  accept={:.0}%  edges={}",
                step + 1,
                bar(full, 0.30, 30),
                full * 100.0,
                rate,
                net.edge_count()
            );
        }
    }

    let final_acc = eval_accuracy(
        &mut net,
        &projection,
        corpus,
        cfg.full_len,
        &mut eval_rng,
        &sdr,
        &init.propagation,
        output_start,
        neuron_count,
    );
    peak_accuracy = peak_accuracy.max(final_acc);
    let rate = accepted as f64 / (accepted + rejected).max(1) as f64 * 100.0;
    println!(
        "  [seed={seed}] FINAL: {:.1}%  peak={:.1}%  edges={}  accept={:.0}%",
        final_acc * 100.0,
        peak_accuracy * 100.0,
        net.edge_count(),
        rate
    );
    EvolutionResult {
        seed,
        final_accuracy: final_acc,
        peak_accuracy,
        edge_count: net.edge_count(),
        accept_rate: rate,
    }
}

fn compute_baselines(corpus: &[u8]) -> Baselines {
    let mut freq = [0u64; CHARS];
    for &c in corpus {
        freq[c as usize] += 1;
    }
    let most_common = freq
        .iter()
        .enumerate()
        .max_by_key(|&(_, &count)| count)
        .map(|(i, _)| i)
        .unwrap_or(0);
    let freq_base = freq[most_common] as f64 / corpus.len() as f64;

    let mut bigram = vec![0u64; CHARS * CHARS];
    for bw in corpus.windows(2) {
        bigram[bw[0] as usize * CHARS + bw[1] as usize] += 1;
    }
    let mut bigram_ok = 0u64;
    for bw in corpus.windows(2) {
        let best = (0..CHARS)
            .max_by_key(|&n| bigram[bw[0] as usize * CHARS + n])
            .unwrap_or(0);
        if best == bw[1] as usize {
            bigram_ok += 1;
        }
    }

    Baselines {
        random_accuracy: 1.0 / CHARS as f64,
        frequency_accuracy: freq_base,
        frequency_char: char_label(most_common as u8),
        bigram_accuracy: bigram_ok as f64 / (corpus.len() - 1) as f64,
    }
}

fn compute_summary(results: &[EvolutionResult]) -> SummaryMetrics {
    let mean_accuracy = results.iter().map(|r| r.final_accuracy).sum::<f64>() / results.len() as f64;
    let min_accuracy = results
        .iter()
        .map(|r| r.final_accuracy)
        .fold(f64::INFINITY, f64::min);
    let max_accuracy = results
        .iter()
        .map(|r| r.final_accuracy)
        .fold(f64::NEG_INFINITY, f64::max);
    let best_peak_accuracy = results
        .iter()
        .map(|r| r.peak_accuracy)
        .fold(f64::NEG_INFINITY, f64::max);
    let mean_accept_rate =
        results.iter().map(|r| r.accept_rate).sum::<f64>() / results.len() as f64;

    SummaryMetrics {
        mean_accuracy,
        min_accuracy,
        max_accuracy,
        spread_pp: (max_accuracy - min_accuracy) * 100.0,
        best_peak_accuracy,
        mean_accept_rate,
    }
}

fn write_report_bundle(
    report_dir: &Path,
    cfg: &RunnerConfig,
    init: &InitConfig,
    baselines: &Baselines,
    summary: &SummaryMetrics,
    results: &[EvolutionResult],
    runtime_seconds: f64,
) {
    fs::create_dir_all(report_dir).expect("failed to create report directory");

    let argv = env::args().collect::<Vec<_>>().join("\n");
    let run_cmd = format!(
        "argv\n{argv}\n\ncwd\n{}\n",
        env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .display()
    );
    fs::write(report_dir.join("run_cmd.txt"), run_cmd).expect("failed to write run_cmd.txt");

    let env_report = EnvReport {
        package_name: env!("CARGO_PKG_NAME"),
        package_version: env!("CARGO_PKG_VERSION"),
        os: env::consts::OS,
        arch: env::consts::ARCH,
        current_dir: env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .display()
            .to_string(),
        current_exe: env::current_exe()
            .unwrap_or_else(|_| PathBuf::from("unknown"))
            .display()
            .to_string(),
        corpus_path: cfg.corpus_path.clone(),
        steps: cfg.steps,
        seed_count: cfg.seed_count,
        seed_base: cfg.seed_base,
        seed_stride: SEED_STRIDE,
        full_len: cfg.full_len,
        neuron_count: init.neuron_count,
        phi_dim: init.phi_dim,
        chain_count: init.chain_count,
        input_end: init.input_end(),
        output_start: init.output_start(),
        edge_cap: init.edge_cap(),
        ticks_per_token: init.propagation.ticks_per_token,
        input_duration_ticks: init.propagation.input_duration_ticks,
        decay_interval_ticks: init.propagation.decay_interval_ticks,
        use_refractory: init.propagation.use_refractory,
    };
    let env_json =
        serde_json::to_string_pretty(&env_report).expect("failed to serialize env report");
    fs::write(report_dir.join("env.json"), env_json).expect("failed to write env.json");

    let metrics = MetricsReport {
        baselines: Baselines {
            random_accuracy: baselines.random_accuracy,
            frequency_accuracy: baselines.frequency_accuracy,
            frequency_char: baselines.frequency_char,
            bigram_accuracy: baselines.bigram_accuracy,
        },
        summary: SummaryMetrics {
            mean_accuracy: summary.mean_accuracy,
            min_accuracy: summary.min_accuracy,
            max_accuracy: summary.max_accuracy,
            spread_pp: summary.spread_pp,
            best_peak_accuracy: summary.best_peak_accuracy,
            mean_accept_rate: summary.mean_accept_rate,
        },
        seeds: results.to_vec(),
        runtime_seconds,
    };
    let metrics_json =
        serde_json::to_string_pretty(&metrics).expect("failed to serialize metrics report");
    fs::write(report_dir.join("metrics.json"), metrics_json).expect("failed to write metrics.json");

    let summary_md = format!(
        "# `evolve_language` summary\n\n\
Status: completed\n\n\
## Configuration\n\n\
- Corpus: `{}`\n\
- Steps per seed: `{}`\n\
- Seed count: `{}`\n\
- Seed base: `{}`\n\
- Full eval length: `{}`\n\
- H: `{}`\n\
- Phi dim: `{}`\n\
- Chain count: `{}`\n\
- Edge cap: `{}`\n\n\
## Baselines\n\n\
- Random: `{:.1}%`\n\
- Frequency (`{}`): `{:.1}%`\n\
- Bigram: `{:.1}%`\n\n\
## Summary\n\n\
- Mean final accuracy: `{:.1}%`\n\
- Final range: `{:.1}% - {:.1}%`\n\
- Best observed peak: `{:.1}%`\n\
- Mean accept rate: `{:.1}%`\n\
- Runtime: `{:.1}s`\n\n\
## Notes\n\n\
- This runner is the canonical Rust public beta path.\n\
- Successful completion and report emission count as the beta pass condition.\n\
- Current results should be read as reproducibility and implementation evidence, not as a promoted breakthrough.\n",
        cfg.corpus_path,
        cfg.steps,
        cfg.seed_count,
        cfg.seed_base,
        cfg.full_len,
        init.neuron_count,
        init.phi_dim,
        init.chain_count,
        init.edge_cap(),
        baselines.random_accuracy * 100.0,
        baselines.frequency_char,
        baselines.frequency_accuracy * 100.0,
        baselines.bigram_accuracy * 100.0,
        summary.mean_accuracy * 100.0,
        summary.min_accuracy * 100.0,
        summary.max_accuracy * 100.0,
        summary.best_peak_accuracy * 100.0,
        summary.mean_accept_rate,
        runtime_seconds
    );
    fs::write(report_dir.join("summary.md"), summary_md).expect("failed to write summary.md");
}

fn main() {
    let cfg = parse_cli();
    let init = InitConfig::phi(256);
    let seeds: Vec<u64> = (0..cfg.seed_count)
        .map(|i| cfg.seed_base + i as u64 * SEED_STRIDE)
        .collect();

    println!("Loading corpus...");
    let corpus = load_corpus(&cfg.corpus_path);
    println!("  {} chars", corpus.len());

    let bigram = build_bigram_table(&corpus);

    let baselines = compute_baselines(&corpus);
    println!(
        "  Random: {:.1}%  Freq('{}'): {:.1}%  Bigram: {:.1}%",
        baselines.random_accuracy * 100.0,
        baselines.frequency_char,
        baselines.frequency_accuracy * 100.0,
        baselines.bigram_accuracy * 100.0
    );

    println!(
        "\n=== Canonical beta run: smooth fitness + 1+9 jackpot, chain-{} init ===",
        init.chain_count
    );
    println!(
        "H={}, {} steps, {} seeds, edge_cap={}\n",
        init.neuron_count,
        cfg.steps,
        seeds.len(),
        init.edge_cap()
    );

    let started = Instant::now();
    let results: Vec<EvolutionResult> = seeds
        .par_iter()
        .map(|&seed| run_evolution(&cfg, seed, &corpus, &init, &bigram))
        .collect();
    let runtime_seconds = started.elapsed().as_secs_f64();
    let summary = compute_summary(&results);

    println!("\n=== SUMMARY ===");
    println!("  Random:    {:.1}%", baselines.random_accuracy * 100.0);
    println!("  Frequency: {:.1}%", baselines.frequency_accuracy * 100.0);
    println!("  Bigram:    {:.1}%", baselines.bigram_accuracy * 100.0);
    for result in &results {
        println!(
            "  seed={:<6} final={:.1}%  peak={:.1}%  edges={}  accept={:.0}%",
            result.seed,
            result.final_accuracy * 100.0,
            result.peak_accuracy * 100.0,
            result.edge_count,
            result.accept_rate
        );
    }
    println!("  Mean:      {:.1}%", summary.mean_accuracy * 100.0);
    println!(
        "  Range:     {:.1}% - {:.1}%  (spread={:.1}pp)",
        summary.min_accuracy * 100.0,
        summary.max_accuracy * 100.0,
        summary.spread_pp
    );
    println!("  Best peak: {:.1}%", summary.best_peak_accuracy * 100.0);
    println!("  Runtime:   {:.1}s", runtime_seconds);

    if let Some(report_dir) = cfg.report_dir.as_deref() {
        write_report_bundle(
            report_dir,
            &cfg,
            &init,
            &baselines,
            &summary,
            &results,
            runtime_seconds,
        );
        println!("  Report:    {}", report_dir.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_eval_offset_boundary() {
        let mut rng = StdRng::seed_from_u64(1);
        let off = sample_eval_offset(102, 100, &mut rng).unwrap();
        assert!(off <= 1);
    }

    #[test]
    fn parse_cli_defaults() {
        let cfg = RunnerConfig::default();
        assert_eq!(cfg.steps, DEFAULT_STEPS);
        assert_eq!(cfg.seed_count, DEFAULT_SEED_COUNT);
        assert_eq!(cfg.seed_base, DEFAULT_SEED_BASE);
        assert_eq!(cfg.full_len, DEFAULT_FULL_LEN);
        assert!(cfg.report_dir.is_none());
    }

    #[test]
    fn paired_eval_noop_is_exactly_stable() {
        let init = InitConfig::phi(256);
        let corpus: Vec<u8> = (0..256).map(|i| (i % CHARS) as u8).collect();
        let sdr = SdrTable::new(
            CHARS,
            init.neuron_count,
            init.input_end(),
            SDR_ACTIVE_PCT,
            &mut StdRng::seed_from_u64(123),
        )
        .unwrap();
        let mut net = build_network(&init, &mut StdRng::seed_from_u64(321));
        let projection = Int8Projection::new(
            init.phi_dim,
            CHARS,
            &mut StdRng::seed_from_u64(654),
        );
        let mut eval_rng = StdRng::seed_from_u64(999);
        let eval_rng_snapshot = eval_rng.clone();

        let before = eval_accuracy(
            &mut net,
            &projection,
            &corpus,
            64,
            &mut eval_rng,
            &sdr,
            &init.propagation,
            init.output_start(),
            init.neuron_count,
        );
        eval_rng = eval_rng_snapshot;
        let after = eval_accuracy(
            &mut net,
            &projection,
            &corpus,
            64,
            &mut eval_rng,
            &sdr,
            &init.propagation,
            init.output_start(),
            init.neuron_count,
        );

        assert_eq!(before, after);
    }
}
