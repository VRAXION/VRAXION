//! A/B test: Fixed mutation schedule vs Adaptive operator selection.
//!
//! Adaptive: each operator type has a rolling accept rate (window=200).
//! Operators with higher accept rate get selected more often (softmax).
//! If topology stalls → W gets more budget. If W stalls → topology does.
//!
//! Both use smooth cosine-bigram fitness.
//!
//! Run: cargo run --example ab_adaptive_ops --release -- <corpus-path>

use instnct_core::{load_corpus, build_network, InitConfig, Int8Projection, Network, SdrTable};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const STEPS: usize = 30_000;
const WINDOW: usize = 200;

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

// ---- Operator definitions ----

const OP_COUNT: usize = 9;
const OP_NAMES: [&str; OP_COUNT] = [
    "add_edge", "remove", "rewire", "reverse",
    "mirror", "enhance", "theta", "channel", "W_proj",
];

fn apply_op(op: usize, net: &mut Network, proj: &mut Int8Projection, rng: &mut impl Rng) -> bool {
    match op {
        0 => net.mutate_add_edge(rng),
        1 => net.mutate_remove_edge(rng),
        2 => net.mutate_rewire(rng),
        3 => net.mutate_reverse(rng),
        4 => net.mutate_mirror(rng),
        5 => net.mutate_enhance(rng),
        6 => net.mutate_theta(rng),
        7 => net.mutate_channel(rng),
        8 => { let _ = proj.mutate_one(rng); true }
        _ => unreachable!(),
    }
}

// Fixed schedule weights (matches library evolution_step)
const FIXED_WEIGHTS: [f64; OP_COUNT] = [
    25.0, 15.0, 10.0, 15.0, 7.0, 8.0, 5.0, 5.0, 10.0,
];

// ---- Adaptive tracker ----

struct OpTracker {
    // Circular buffer per op: 1 = accepted, 0 = rejected
    history: Vec<Vec<u8>>,
    head: Vec<usize>,
    count: Vec<usize>,
    accepts: Vec<usize>,
}

impl OpTracker {
    fn new() -> Self {
        Self {
            history: (0..OP_COUNT).map(|_| vec![0u8; WINDOW]).collect(),
            head: vec![0; OP_COUNT],
            count: vec![0; OP_COUNT],
            accepts: vec![0; OP_COUNT],
        }
    }

    fn record(&mut self, op: usize, accepted: bool) {
        let h = self.head[op];
        // Remove oldest entry if buffer is full
        if self.count[op] >= WINDOW {
            if self.history[op][h] == 1 {
                self.accepts[op] -= 1;
            }
        } else {
            self.count[op] += 1;
        }
        // Write new entry
        self.history[op][h] = if accepted { 1 } else { 0 };
        if accepted {
            self.accepts[op] += 1;
        }
        self.head[op] = (h + 1) % WINDOW;
    }

    fn accept_rate(&self, op: usize) -> f64 {
        if self.count[op] == 0 { return 0.0; }
        self.accepts[op] as f64 / self.count[op] as f64
    }

    /// Pick operator weighted by accept rate + floor.
    /// Floor ensures even cold operators get tried occasionally.
    fn pick_adaptive(&self, rng: &mut impl Rng) -> usize {
        let floor = 0.01; // 1% minimum per operator
        let mut weights = [0.0f64; OP_COUNT];
        for (i, w) in weights.iter_mut().enumerate() {
            *w = self.accept_rate(i) + floor;
        }
        let total: f64 = weights.iter().sum();
        let mut r = rng.gen::<f64>() * total;
        for (i, &w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return i;
            }
        }
        OP_COUNT - 1
    }

    fn pick_fixed(rng: &mut impl Rng) -> usize {
        let total: f64 = FIXED_WEIGHTS.iter().sum();
        let mut r = rng.gen::<f64>() * total;
        for (i, &w) in FIXED_WEIGHTS.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return i;
            }
        }
        OP_COUNT - 1
    }

    fn summary(&self) -> String {
        let mut parts = Vec::new();
        for (i, &name) in OP_NAMES.iter().enumerate() {
            let rate = self.accept_rate(i);
            if self.count[i] > 0 {
                parts.push(format!("{}={:.0}%", name, rate * 100.0));
            }
        }
        parts.join(" ")
    }
}

// ---- Variants ----

#[derive(Clone, Copy)]
enum Variant {
    FixedSchedule,
    Adaptive,
}

impl Variant {
    fn name(&self) -> &str {
        match self {
            Self::FixedSchedule => "fixed_sched",
            Self::Adaptive => "adaptive",
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
    op_summary: String,
}

fn run_one(cfg: &Config, corpus: &[u8], bigram: &BigramTable) -> RunResult {
    let init = InitConfig::phi(256);
    let edge_cap = init.edge_cap();

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut net = build_network(&init, &mut rng);
    let mut proj = Int8Projection::new(init.phi_dim, CHARS,
        &mut StdRng::seed_from_u64(cfg.seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed + 1000);
    let sdr = SdrTable::new(CHARS, init.neuron_count, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(cfg.seed + 100)).unwrap();

    let mut tracker = OpTracker::new();
    let mut peak_acc = 0.0f64;
    let mut accepted = 0u32;
    let mut total = 0u32;

    for step in 0..STEPS {
        // Paired eval
        let snap = eval_rng.clone();
        let before = eval_smooth(&mut net, &proj, corpus, 100, &mut eval_rng, &sdr, &init, bigram);
        eval_rng = snap;

        // Snapshot
        let net_state = net.save_state();
        let proj_backup = proj.clone();
        let edges_before = net.edge_count();

        // Pick operator
        let op = match cfg.variant {
            Variant::FixedSchedule => OpTracker::pick_fixed(&mut rng),
            Variant::Adaptive => tracker.pick_adaptive(&mut rng),
        };

        let mutated = apply_op(op, &mut net, &mut proj, &mut rng);

        if !mutated {
            let _ = eval_smooth(&mut net, &proj, corpus, 100, &mut eval_rng, &sdr, &init, bigram);
            tracker.record(op, false);
            continue;
        }

        total += 1;
        let after = eval_smooth(&mut net, &proj, corpus, 100, &mut eval_rng, &sdr, &init, bigram);

        let edge_grew = net.edge_count() > edges_before;
        let within_cap = !edge_grew || net.edge_count() <= edge_cap;

        if after > before && within_cap {
            accepted += 1;
            tracker.record(op, true);
        } else {
            net.restore_state(&net_state);
            proj = proj_backup;
            tracker.record(op, false);
        }

        if (step + 1) % 10_000 == 0 {
            let mut cr = StdRng::seed_from_u64(cfg.seed + 6000 + step as u64);
            let acc = eval_accuracy(&mut net, &proj, corpus, 2000, &mut cr, &sdr, &init);
            if acc > peak_acc { peak_acc = acc; }
            let rate = if total > 0 { accepted as f64 / total as f64 * 100.0 } else { 0.0 };
            println!("  {} seed={} step {:>5}: acc={:.1}% edges={} accept={:.1}% | {}",
                cfg.variant.name(), cfg.seed, step + 1, acc * 100.0,
                net.edge_count(), rate, tracker.summary());
        }
    }

    let mut fr = StdRng::seed_from_u64(cfg.seed + 9999);
    let final_acc = eval_accuracy(&mut net, &proj, corpus, 5000, &mut fr, &sdr, &init);
    if final_acc > peak_acc { peak_acc = final_acc; }
    let mut sr = StdRng::seed_from_u64(cfg.seed + 9998);
    let final_cosine = eval_smooth(&mut net, &proj, corpus, 5000, &mut sr, &sdr, &init, bigram);
    let rate = if total > 0 { accepted as f64 / total as f64 * 100.0 } else { 0.0 };
    let op_summary = tracker.summary();

    println!("  {} seed={} FINAL: acc={:.1}% cos={:.4} peak={:.1}% edges={} accept={:.1}%",
        cfg.variant.name(), cfg.seed, final_acc * 100.0, final_cosine,
        peak_acc * 100.0, net.edge_count(), rate);

    RunResult {
        variant_name: cfg.variant.name().to_string(),
        seed: cfg.seed, final_acc, peak_acc,
        final_edges: net.edge_count(), accept_rate: rate,
        final_cosine, op_summary,
    }
}

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "S:/AI/work/VRAXION_DEV/instnct/data/traindat/fineweb_edu.traindat".to_string()
    });

    println!("=== A/B Adaptive Operator Selection ===");
    println!("  A: Fixed schedule (25% add, 15% remove, 10% rewire, etc.)");
    println!("  B: Adaptive (rolling accept rate picks operators, floor=1%)");
    println!("  Both use smooth cosine-bigram fitness\n");

    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars", corpus.len());
    let bigram = build_bigram_table(&corpus);
    println!();

    let seeds = [42u64, 123, 7, 1042, 555, 8042];

    let mut configs: Vec<Config> = Vec::new();
    for &v in &[Variant::FixedSchedule, Variant::Adaptive] {
        for &s in &seeds {
            configs.push(Config { variant: v, seed: s });
        }
    }
    println!("  {} configs: 2 variants x {} seeds\n", configs.len(), seeds.len());

    let start = Instant::now();
    let results: Vec<RunResult> = configs.par_iter()
        .map(|cfg| run_one(cfg, &corpus, &bigram))
        .collect();
    let elapsed = start.elapsed().as_secs_f64();

    println!("\n=== SUMMARY ===\n");
    println!("{:<14} {:>7} {:>7} {:>7} {:>8} {:>7} {:>7}",
        "Variant", "Mean%", "Best%", "Peak%", "Cosine", "Edges", "Accept");
    println!("{}", "-".repeat(67));

    for v in &[Variant::FixedSchedule, Variant::Adaptive] {
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

    // Print adaptive op distribution for each adaptive seed
    println!("\nAdaptive op rates (final):");
    for r in &results {
        if r.variant_name == "adaptive" {
            println!("  seed={}: {}", r.seed, r.op_summary);
        }
    }

    println!("\n  Total time: {:.0}s", elapsed);
}
