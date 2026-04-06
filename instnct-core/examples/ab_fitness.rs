//! A/B fitness function test: Stepwise (argmax accuracy) vs Smooth (cosine-bigram).
//!
//! Tests whether a smooth fitness landscape (cosine similarity to bigram distribution)
//! produces better evolution outcomes than the discrete binary accuracy metric.
//!
//! Both variants use IDENTICAL: init, mutation schedule, W evolution, eval segments.
//! ONLY the fitness function differs.
//!
//! Key hypothesis: the Python 24.4% result used cosine-bigram fitness, while Rust's
//! ~17-18% ceiling uses argmax accuracy. The smooth landscape gives evolution
//! continuous feedback vs the discrete step function.
//!
//! Run: cargo run --example ab_fitness --release -- <corpus-path>

use instnct_core::{
    build_network, evolution_step, EvolutionConfig, InitConfig, Int8Projection, Network,
    SdrTable, StepOutcome,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::fs;
use std::time::Instant;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const STEPS: usize = 30_000;

fn load_corpus(path: &str) -> Vec<u8> {
    let raw = fs::read(path).expect("cannot read corpus");
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

/// Build 27×27 bigram probability table from corpus.
/// bigram[i][j] = P(next=j | current=i)
fn build_bigram(corpus: &[u8]) -> Vec<[f64; CHARS]> {
    let mut counts = vec![[0u64; CHARS]; CHARS];
    for pair in corpus.windows(2) {
        counts[pair[0] as usize][pair[1] as usize] += 1;
    }
    let mut bigram = vec![[0.0f64; CHARS]; CHARS];
    for i in 0..CHARS {
        let total: u64 = counts[i].iter().sum();
        if total > 0 {
            for j in 0..CHARS {
                bigram[i][j] = counts[i][j] as f64 / total as f64;
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
        let uniform = 1.0 / CHARS as f64;
        out.fill(uniform);
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

/// Stepwise fitness: binary argmax accuracy over `len` characters.
#[allow(clippy::too_many_arguments)]
fn eval_stepwise(
    net: &mut Network,
    proj: &Int8Projection,
    corpus: &[u8],
    len: usize,
    rng: &mut StdRng,
    sdr: &SdrTable,
    init: &InitConfig,
) -> f64 {
    if corpus.len() <= len {
        return 0.0;
    }
    let off = rng.gen_range(0..=corpus.len() - len - 1);
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), &init.propagation)
            .unwrap();
        if proj.predict(&net.charge()[init.output_start()..init.neuron_count])
            == seg[i + 1] as usize
        {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

/// Smooth fitness: mean cosine similarity to bigram distribution.
#[allow(clippy::too_many_arguments)]
fn eval_smooth(
    net: &mut Network,
    proj: &Int8Projection,
    corpus: &[u8],
    len: usize,
    rng: &mut StdRng,
    sdr: &SdrTable,
    init: &InitConfig,
    bigram: &[[f64; CHARS]],
) -> f64 {
    if corpus.len() <= len {
        return 0.0;
    }
    let off = rng.gen_range(0..=corpus.len() - len - 1);
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut total_cos = 0.0f64;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), &init.propagation)
            .unwrap();
        let scores = proj.raw_scores(&net.charge()[init.output_start()..init.neuron_count]);
        let probs = softmax_27(&scores);
        let target = &bigram[seg[i] as usize];
        total_cos += cosine_27(&probs, target);
    }
    total_cos / len as f64
}

#[derive(Clone, Copy)]
enum Variant {
    Stepwise, // binary argmax accuracy
    Smooth,   // cosine to bigram distribution
}

impl Variant {
    fn name(&self) -> &str {
        match self {
            Self::Stepwise => "stepwise",
            Self::Smooth => "smooth",
        }
    }
}

struct Config {
    variant: Variant,
    seed: u64,
}

#[allow(dead_code)]
struct RunResult {
    variant_name: String,
    seed: u64,
    final_acc: f64,
    peak_acc: f64,
    final_edges: usize,
    accept_rate: f64,
    final_cosine: f64,
}

fn run_one(cfg: &Config, corpus: &[u8], bigram: &[[f64; CHARS]]) -> RunResult {
    let init = InitConfig::phi(256);
    let evo = EvolutionConfig {
        edge_cap: init.edge_cap(),
        accept_ties: false,
    };

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut net = build_network(&init, &mut rng);
    let mut proj = Int8Projection::new(
        init.phi_dim,
        CHARS,
        &mut StdRng::seed_from_u64(cfg.seed + 200),
    );
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed + 1000);
    let sdr = SdrTable::new(
        CHARS,
        init.neuron_count,
        init.input_end(),
        SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(cfg.seed + 100),
    )
    .unwrap();

    let mut peak_acc = 0.0f64;
    let mut accepted = 0u32;
    let mut total = 0u32;

    for step in 0..STEPS {
        let outcome = match cfg.variant {
            Variant::Stepwise => evolution_step(
                &mut net,
                &mut proj,
                &mut rng,
                &mut eval_rng,
                |n, p, e| eval_stepwise(n, p, corpus, 100, e, &sdr, &init),
                &evo,
            ),
            Variant::Smooth => evolution_step(
                &mut net,
                &mut proj,
                &mut rng,
                &mut eval_rng,
                |n, p, e| eval_smooth(n, p, corpus, 100, e, &sdr, &init, bigram),
                &evo,
            ),
        };
        match outcome {
            StepOutcome::Accepted => {
                accepted += 1;
                total += 1;
            }
            StepOutcome::Rejected => {
                total += 1;
            }
            StepOutcome::Skipped => {}
        }

        if (step + 1) % 10_000 == 0 {
            // Always measure ARGMAX ACCURACY for fair comparison (both variants)
            let mut cr = StdRng::seed_from_u64(cfg.seed + 6000 + step as u64);
            let acc = eval_stepwise(&mut net, &proj, corpus, 2000, &mut cr, &sdr, &init);
            if acc > peak_acc {
                peak_acc = acc;
            }
            let rate = if total > 0 {
                accepted as f64 / total as f64 * 100.0
            } else {
                0.0
            };

            // Also measure smooth fitness for both
            let mut sr = StdRng::seed_from_u64(cfg.seed + 7000 + step as u64);
            let smooth = eval_smooth(&mut net, &proj, corpus, 2000, &mut sr, &sdr, &init, bigram);

            println!(
                "  {} seed={} step {:>5}: acc={:.1}% cos={:.4} edges={} accept={:.1}%",
                cfg.variant.name(),
                cfg.seed,
                step + 1,
                acc * 100.0,
                smooth,
                net.edge_count(),
                rate
            );
        }
    }

    // Final eval: both metrics on 5000 chars
    let mut fr = StdRng::seed_from_u64(cfg.seed + 9999);
    let final_acc = eval_stepwise(&mut net, &proj, corpus, 5000, &mut fr, &sdr, &init);
    if final_acc > peak_acc {
        peak_acc = final_acc;
    }

    let mut sr = StdRng::seed_from_u64(cfg.seed + 9998);
    let final_cosine = eval_smooth(&mut net, &proj, corpus, 5000, &mut sr, &sdr, &init, bigram);

    let rate = if total > 0 {
        accepted as f64 / total as f64 * 100.0
    } else {
        0.0
    };
    println!(
        "  {} seed={} FINAL: acc={:.1}% cos={:.4} peak={:.1}% edges={} accept={:.1}%",
        cfg.variant.name(),
        cfg.seed,
        final_acc * 100.0,
        final_cosine,
        peak_acc * 100.0,
        net.edge_count(),
        rate
    );

    RunResult {
        variant_name: cfg.variant.name().to_string(),
        seed: cfg.seed,
        final_acc,
        peak_acc,
        final_edges: net.edge_count(),
        accept_rate: rate,
        final_cosine,
    }
}

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "S:/AI/work/VRAXION_DEV/instnct/data/traindat/fineweb_edu.traindat".to_string()
    });

    println!("=== A/B Fitness Function Test ===");
    println!("  A: Stepwise (binary argmax accuracy)");
    println!("  B: Smooth (cosine similarity to bigram distribution)");
    println!("  Everything else IDENTICAL: init, mutations, W evolution, seeds\n");

    let corpus = load_corpus(&corpus_path);
    println!("  {} chars", corpus.len());

    let bigram = build_bigram(&corpus);
    // Print top-3 bigrams for sanity check
    let mut top: Vec<(usize, usize, f64)> = Vec::new();
    for (i, row) in bigram.iter().enumerate() {
        for (j, &p) in row.iter().enumerate() {
            if p > 0.05 {
                top.push((i, j, p));
            }
        }
    }
    top.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    print!("  Bigram top-5: ");
    for (i, j, p) in top.iter().take(5) {
        let ic = if *i == 26 { '_' } else { (b'a' + *i as u8) as char };
        let jc = if *j == 26 { '_' } else { (b'a' + *j as u8) as char };
        print!("{}{}={:.1}% ", ic, jc, p * 100.0);
    }
    println!("\n");

    let seeds = [42u64, 123, 7, 1042, 555, 8042];

    let mut configs: Vec<Config> = Vec::new();
    for &v in &[Variant::Stepwise, Variant::Smooth] {
        for &s in &seeds {
            configs.push(Config { variant: v, seed: s });
        }
    }

    println!(
        "  {} configs: 2 variants × {} seeds\n",
        configs.len(),
        seeds.len()
    );

    let start = Instant::now();

    let results: Vec<RunResult> = configs
        .par_iter()
        .map(|cfg| run_one(cfg, &corpus, &bigram))
        .collect();

    let elapsed = start.elapsed().as_secs_f64();

    println!("\n=== SUMMARY ===\n");
    println!(
        "{:<12} {:>7} {:>7} {:>7} {:>8} {:>7} {:>7}",
        "Variant", "Mean%", "Best%", "Peak%", "Cosine", "Edges", "Accept"
    );
    println!("{}", "-".repeat(65));

    for v in &[Variant::Stepwise, Variant::Smooth] {
        let g: Vec<_> = results
            .iter()
            .filter(|r| r.variant_name == v.name())
            .collect();
        let n = g.len() as f64;
        let mean = g.iter().map(|r| r.final_acc).sum::<f64>() / n;
        let best = g.iter().map(|r| r.final_acc).fold(0.0f64, f64::max);
        let peak = g.iter().map(|r| r.peak_acc).fold(0.0f64, f64::max);
        let cosine_mean = g.iter().map(|r| r.final_cosine).sum::<f64>() / n;
        let edges = g.iter().map(|r| r.final_edges).sum::<usize>() / g.len();
        let accept = g.iter().map(|r| r.accept_rate).sum::<f64>() / n;
        println!(
            "{:<12} {:>6.1}% {:>6.1}% {:>6.1}% {:>8.4} {:>7} {:>6.1}%",
            v.name(),
            mean * 100.0,
            best * 100.0,
            peak * 100.0,
            cosine_mean,
            edges,
            accept
        );
    }

    println!("\nPer-seed:");
    println!(
        "{:<12} {:>6} {:>7} {:>7} {:>8} {:>7} {:>7}",
        "Variant", "Seed", "Acc%", "Peak%", "Cosine", "Edges", "Accept"
    );
    println!("{}", "-".repeat(62));
    for r in &results {
        println!(
            "{:<12} {:>6} {:>6.1}% {:>6.1}% {:>8.4} {:>7} {:>6.1}%",
            r.variant_name,
            r.seed,
            r.final_acc * 100.0,
            r.peak_acc * 100.0,
            r.final_cosine,
            r.final_edges,
            r.accept_rate
        );
    }

    println!("\n  Total time: {:.0}s", elapsed);
}
