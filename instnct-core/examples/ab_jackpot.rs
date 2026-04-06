//! A/B test: 1+1 ES vs 1+9 Jackpot (multi-candidate per step).
//!
//! Jackpot: each evolution step generates N candidate mutations from
//! the SAME parent state, evaluates each, accepts the BEST if it improves.
//! This is the Python "multi-worker" pattern (9 workers in parallel).
//!
//! Both use smooth cosine-bigram fitness.
//!
//! Run: cargo run --example ab_jackpot --release -- <corpus-path>

use instnct_core::{build_network, InitConfig, Int8Projection, Network, SdrTable};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::fs;
use std::time::Instant;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const STEPS: usize = 30_000;

type BigramTable = Vec<[f64; CHARS]>;

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
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

#[allow(clippy::too_many_arguments)]
fn eval_smooth(
    net: &mut Network, proj: &Int8Projection, corpus: &[u8], len: usize,
    rng: &mut StdRng, sdr: &SdrTable, init: &InitConfig, bigram: &BigramTable,
) -> f64 {
    if corpus.len() <= len { return 0.0; }
    let off = rng.gen_range(0..=corpus.len() - len - 1);
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut total_cos = 0.0f64;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), &init.propagation).unwrap();
        let scores = proj.raw_scores(&net.charge()[init.output_start()..init.neuron_count]);
        let probs = softmax_27(&scores);
        total_cos += cosine_27(&probs, &bigram[seg[i] as usize]);
    }
    total_cos / len as f64
}

#[allow(clippy::too_many_arguments)]
fn eval_accuracy(
    net: &mut Network, proj: &Int8Projection, corpus: &[u8], len: usize,
    rng: &mut StdRng, sdr: &SdrTable, init: &InitConfig,
) -> f64 {
    if corpus.len() <= len { return 0.0; }
    let off = rng.gen_range(0..=corpus.len() - len - 1);
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), &init.propagation).unwrap();
        if proj.predict(&net.charge()[init.output_start()..init.neuron_count]) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

/// Apply one random mutation from the full schedule (topology 90% + W 10%).
fn apply_mutation(net: &mut Network, proj: &mut Int8Projection, rng: &mut impl Rng) -> bool {
    let roll = rng.gen_range(0..100u32);
    match roll {
        0..25 => net.mutate_add_edge(rng),
        25..40 => net.mutate_remove_edge(rng),
        40..50 => net.mutate_rewire(rng),
        50..65 => net.mutate_reverse(rng),
        65..72 => net.mutate_mirror(rng),
        72..80 => net.mutate_enhance(rng),
        80..85 => net.mutate_theta(rng),
        85..90 => net.mutate_channel(rng),
        _ => { let _ = proj.mutate_one(rng); true }
    }
}

#[derive(Clone, Copy)]
enum Variant {
    Es1Plus1,     // standard 1+1 ES
    Jackpot9,     // 9 candidates per step, best wins
}

impl Variant {
    fn name(&self) -> &str {
        match self {
            Self::Es1Plus1 => "1+1_ES",
            Self::Jackpot9 => "1+9_jackpot",
        }
    }

    fn n_candidates(&self) -> usize {
        match self {
            Self::Es1Plus1 => 1,
            Self::Jackpot9 => 9,
        }
    }
}

struct Config { variant: Variant, seed: u64 }

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

fn run_one(cfg: &Config, corpus: &[u8], bigram: &BigramTable) -> RunResult {
    let init = InitConfig::phi(256);
    let edge_cap = init.edge_cap();
    let n_cand = cfg.variant.n_candidates();

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut net = build_network(&init, &mut rng);
    let mut proj = Int8Projection::new(init.phi_dim, CHARS,
        &mut StdRng::seed_from_u64(cfg.seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed + 1000);
    let sdr = SdrTable::new(CHARS, init.neuron_count, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(cfg.seed + 100)).unwrap();

    let mut peak_acc = 0.0f64;
    let mut accepted = 0u32;
    let mut total = 0u32;

    for step in 0..STEPS {
        // Paired eval: get baseline score
        let snap = eval_rng.clone();
        let before = eval_smooth(&mut net, &proj, corpus, 100, &mut eval_rng, &sdr, &init, bigram);
        eval_rng = snap;

        // Save parent state
        let parent_net = net.save_state();
        let parent_proj = proj.clone();
        let edges_before = net.edge_count();

        // Try N candidates, keep track of the best
        let mut best_delta = f64::NEG_INFINITY;
        let mut best_net_state = None;
        let mut best_proj = None;
        let mut any_mutated = false;

        for c in 0..n_cand {
            // Reset to parent state for each candidate
            net.restore_state(&parent_net);
            proj = parent_proj.clone();

            // Each candidate uses a different RNG stream
            let mut cand_rng = StdRng::seed_from_u64(
                cfg.seed.wrapping_add(300).wrapping_add(step as u64 * 100 + c as u64)
            );

            let mutated = apply_mutation(&mut net, &mut proj, &mut cand_rng);
            if !mutated {
                continue;
            }
            any_mutated = true;

            // Evaluate candidate (same eval segment as baseline via cloned rng)
            let eval_snap = eval_rng.clone();
            let after = eval_smooth(&mut net, &proj, corpus, 100, &mut eval_rng, &sdr, &init, bigram);
            eval_rng = eval_snap;

            // Edge cap check
            let edge_grew = net.edge_count() > edges_before;
            let within_cap = !edge_grew || net.edge_count() <= edge_cap;

            let delta = after - before;
            if delta > best_delta && within_cap {
                best_delta = delta;
                best_net_state = Some(net.save_state());
                best_proj = Some(proj.clone());
            }
        }

        // Advance eval_rng once for this step
        let _ = eval_smooth(&mut net, &proj, corpus, 100, &mut eval_rng, &sdr, &init, bigram);

        if !any_mutated {
            net.restore_state(&parent_net);
            proj = parent_proj;
            continue;
        }

        total += 1;

        // Accept best candidate if it improved
        if best_delta > 0.0 {
            if let (Some(net_s), Some(proj_s)) = (best_net_state, best_proj) {
                net.restore_state(&net_s);
                proj = proj_s;
                accepted += 1;
            }
        } else {
            // Reject all — restore parent
            net.restore_state(&parent_net);
            proj = parent_proj;
        }

        if (step + 1) % 10_000 == 0 {
            let mut cr = StdRng::seed_from_u64(cfg.seed + 6000 + step as u64);
            let acc = eval_accuracy(&mut net, &proj, corpus, 2000, &mut cr, &sdr, &init);
            if acc > peak_acc { peak_acc = acc; }
            let rate = if total > 0 { accepted as f64 / total as f64 * 100.0 } else { 0.0 };
            let mut sr = StdRng::seed_from_u64(cfg.seed + 7000 + step as u64);
            let cos = eval_smooth(&mut net, &proj, corpus, 2000, &mut sr, &sdr, &init, bigram);
            println!("  {} seed={} step {:>5}: acc={:.1}% cos={:.4} edges={} accept={:.1}%",
                cfg.variant.name(), cfg.seed, step + 1, acc * 100.0, cos,
                net.edge_count(), rate);
        }
    }

    let mut fr = StdRng::seed_from_u64(cfg.seed + 9999);
    let final_acc = eval_accuracy(&mut net, &proj, corpus, 5000, &mut fr, &sdr, &init);
    if final_acc > peak_acc { peak_acc = final_acc; }
    let mut sr = StdRng::seed_from_u64(cfg.seed + 9998);
    let final_cosine = eval_smooth(&mut net, &proj, corpus, 5000, &mut sr, &sdr, &init, bigram);
    let rate = if total > 0 { accepted as f64 / total as f64 * 100.0 } else { 0.0 };

    println!("  {} seed={} FINAL: acc={:.1}% cos={:.4} peak={:.1}% edges={} accept={:.1}%",
        cfg.variant.name(), cfg.seed, final_acc * 100.0, final_cosine,
        peak_acc * 100.0, net.edge_count(), rate);

    RunResult {
        variant_name: cfg.variant.name().to_string(),
        seed: cfg.seed, final_acc, peak_acc,
        final_edges: net.edge_count(), accept_rate: rate, final_cosine,
    }
}

fn main() {
    // Limit CPU usage: 4 threads on a 12-core = ~33%
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build_global()
        .ok();

    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "S:/AI/work/VRAXION_DEV/instnct/data/traindat/fineweb_edu.traindat".to_string()
    });

    println!("=== A/B Jackpot (Multi-Candidate) Test ===");
    println!("  A: 1+1 ES (1 candidate per step)");
    println!("  B: 1+9 Jackpot (9 candidates per step, best wins)");
    println!("  Both use smooth cosine-bigram fitness");
    println!("  Rayon threads: 4 (CPU-friendly)\n");

    let corpus = load_corpus(&corpus_path);
    println!("  {} chars", corpus.len());
    let bigram = build_bigram_table(&corpus);
    println!();

    let seeds = [42u64, 123, 7, 1042, 555, 8042];

    let mut configs: Vec<Config> = Vec::new();
    for &v in &[Variant::Es1Plus1, Variant::Jackpot9] {
        for &s in &seeds {
            configs.push(Config { variant: v, seed: s });
        }
    }
    println!("  {} configs: 2 variants x {} seeds", configs.len(), seeds.len());
    println!("  Note: 1+9 takes ~9x longer per step (9 evals per step)\n");

    let start = Instant::now();
    let results: Vec<RunResult> = configs.par_iter()
        .map(|cfg| run_one(cfg, &corpus, &bigram))
        .collect();
    let elapsed = start.elapsed().as_secs_f64();

    println!("\n=== SUMMARY ===\n");
    println!("{:<14} {:>7} {:>7} {:>7} {:>8} {:>7} {:>7}",
        "Variant", "Mean%", "Best%", "Peak%", "Cosine", "Edges", "Accept");
    println!("{}", "-".repeat(67));

    for v in &[Variant::Es1Plus1, Variant::Jackpot9] {
        let g: Vec<_> = results.iter().filter(|r| r.variant_name == v.name()).collect();
        let n = g.len() as f64;
        let mean = g.iter().map(|r| r.final_acc).sum::<f64>() / n;
        let best = g.iter().map(|r| r.final_acc).fold(0.0f64, f64::max);
        let peak = g.iter().map(|r| r.peak_acc).fold(0.0f64, f64::max);
        let cosine_mean = g.iter().map(|r| r.final_cosine).sum::<f64>() / n;
        let edges = g.iter().map(|r| r.final_edges).sum::<usize>() / g.len();
        let accept = g.iter().map(|r| r.accept_rate).sum::<f64>() / n;
        println!("{:<14} {:>6.1}% {:>6.1}% {:>6.1}% {:>8.4} {:>7} {:>6.1}%",
            v.name(), mean * 100.0, best * 100.0, peak * 100.0,
            cosine_mean, edges, accept);
    }

    println!("\nPer-seed:");
    println!("{:<14} {:>6} {:>7} {:>7} {:>8} {:>7} {:>7}",
        "Variant", "Seed", "Acc%", "Peak%", "Cosine", "Edges", "Accept");
    println!("{}", "-".repeat(64));
    for r in &results {
        println!("{:<14} {:>6} {:>6.1}% {:>6.1}% {:>8.4} {:>7} {:>6.1}%",
            r.variant_name, r.seed, r.final_acc * 100.0, r.peak_acc * 100.0,
            r.final_cosine, r.final_edges, r.accept_rate);
    }

    println!("\n  Total time: {:.0}s", elapsed);
}
