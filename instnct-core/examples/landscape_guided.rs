//! Landscape-Guided Neuron Build
//! 1. Float backprop × many seeds → map loss landscape
//! 2. Cluster float solutions → find "hot zones" in weight space
//! 3. Guided ternary search near hot zones (not blind 3^N)
//! 4. Freeze best → add next neuron → repeat
//!
//! Run: cargo run --example landscape_guided --release

use std::time::Instant;

// ── PRNG ──
struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 { self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.s }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn range(&mut self, lo: f32, hi: f32) -> f32 { lo + self.f32() * (hi - lo) }
    fn shuffle<T>(&mut self, v: &mut [T]) { for i in (1..v.len()).rev() { let j = self.next() as usize % (i+1); v.swap(i,j); } }
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

// ── Abstract reasoning corpus ──
// Each task: binary input → binary output, with a "reasoning rule"

struct Corpus {
    name: &'static str,
    desc: &'static str,
    n_in: usize,
    data: Vec<(Vec<f32>, f32)>,  // (input_bits, target)
}

fn corpus_bigger_half() -> Corpus {
    // "Are more than half the bits set?" — linear threshold
    let mut data = Vec::new();
    for v in 0..64u8 {
        let bits: Vec<f32> = (0..6).map(|i| ((v >> i) & 1) as f32).collect();
        let sum: u8 = (0..6).map(|i| (v >> i) & 1).sum();
        data.push((bits, if sum > 3 { 1.0 } else { 0.0 }));
    }
    Corpus { name: "MAJORITY6", desc: "More than half of 6 bits set?", n_in: 6, data }
}

fn corpus_compare() -> Corpus {
    // "Is left 3-bit number > right 3-bit number?"
    let mut data = Vec::new();
    for a in 0..8u8 { for b in 0..8u8 {
        let mut bits: Vec<f32> = (0..3).map(|i| ((a >> i) & 1) as f32).collect();
        bits.extend((0..3).map(|i| ((b >> i) & 1) as f32));
        data.push((bits, if a > b { 1.0 } else { 0.0 }));
    }}
    Corpus { name: "COMPARE3", desc: "Is left 3-bit > right 3-bit?", n_in: 6, data }
}

fn corpus_xor() -> Corpus {
    let mut data = Vec::new();
    for v in 0..4u8 {
        let bits: Vec<f32> = (0..2).map(|i| ((v >> i) & 1) as f32).collect();
        data.push((bits, ((v & 1) ^ ((v >> 1) & 1)) as f32));
    }
    Corpus { name: "XOR", desc: "a XOR b (not linearly separable)", n_in: 2, data }
}

fn corpus_parity4() -> Corpus {
    let mut data = Vec::new();
    for v in 0..16u8 {
        let bits: Vec<f32> = (0..4).map(|i| ((v >> i) & 1) as f32).collect();
        let sum: u8 = (0..4).map(|i| (v >> i) & 1).sum();
        data.push((bits, (sum % 2) as f32));
    }
    Corpus { name: "PARITY4", desc: "4-bit parity (hard for 1 neuron)", n_in: 4, data }
}

fn corpus_pattern110() -> Corpus {
    let mut data = Vec::new();
    for v in 0..64u8 {
        let bits: Vec<f32> = (0..6).map(|i| ((v >> i) & 1) as f32).collect();
        let mut has = false;
        for i in 0..4 { if bits[i]>0.5 && bits[i+1]>0.5 && bits[i+2]<0.5 { has = true; }}
        data.push((bits, if has { 1.0 } else { 0.0 }));
    }
    Corpus { name: "HAS_110", desc: "Contains subsequence 110?", n_in: 6, data }
}

// ── Float backprop for 1 neuron (sigmoid) ──

#[derive(Clone)]
struct FloatSolution {
    weights: Vec<f32>,
    bias: f32,
    accuracy: f32,
    loss: f32,
}

fn backprop_one_neuron(corpus: &Corpus, lr: f32, epochs: usize, seed: u64) -> FloatSolution {
    let mut rng = Rng::new(seed);
    let n = corpus.n_in;
    let mut w: Vec<f32> = (0..n).map(|_| rng.range(-2.0, 2.0)).collect();
    let mut b = rng.range(-1.0, 1.0);

    for _ in 0..epochs {
        for (x, y) in &corpus.data {
            let z: f32 = b + w.iter().zip(x).map(|(wi,xi)| wi*xi).sum::<f32>();
            let a = sigmoid(z);
            let err = a - y;
            let sd = a * (1.0 - a);
            let g = err * sd;
            for i in 0..n { w[i] -= lr * g * x[i]; }
            b -= lr * g;
        }
    }

    // Eval
    let mut correct = 0;
    let mut loss = 0.0f32;
    for (x, y) in &corpus.data {
        let z: f32 = b + w.iter().zip(x).map(|(wi,xi)| wi*xi).sum::<f32>();
        let a = sigmoid(z);
        let pred = if a > 0.5 { 1.0 } else { 0.0 };
        if (pred - y).abs() < 0.01 { correct += 1; }
        let eps = 1e-7;
        loss -= y * (a + eps).ln() + (1.0 - y) * (1.0 - a + eps).ln();
    }

    FloatSolution {
        weights: w, bias: b,
        accuracy: correct as f32 / corpus.data.len() as f32 * 100.0,
        loss: loss / corpus.data.len() as f32,
    }
}

// ── Loss landscape analysis ──

struct LandscapeReport {
    n_seeds: usize,
    n_perfect: usize,
    // Per-weight statistics across perfect (or best) solutions
    weight_means: Vec<f32>,
    weight_stds: Vec<f32>,
    weight_signs: Vec<[usize; 3]>,  // [negative, zero-ish, positive] count per weight
    bias_mean: f32,
    // Suggested ternary zones
    suggested_ternary: Vec<Vec<i8>>,  // top-K ternary combos to try
}

fn analyze_landscape(corpus: &Corpus, n_seeds: usize) -> (Vec<FloatSolution>, LandscapeReport) {
    let n = corpus.n_in;
    let mut solutions = Vec::new();

    // Run many seeds
    for seed in 0..n_seeds as u64 {
        let sol = backprop_one_neuron(corpus, 1.0, 3000, seed * 137 + 42);
        solutions.push(sol);
    }

    // Filter to best solutions (>= best_acc - 5%)
    solutions.sort_by(|a, b| b.accuracy.partial_cmp(&a.accuracy).unwrap());
    let best_acc = solutions[0].accuracy;
    let good: Vec<&FloatSolution> = solutions.iter().filter(|s| s.accuracy >= best_acc - 5.0).collect();
    let n_perfect = solutions.iter().filter(|s| s.accuracy >= 100.0).count();

    // Weight statistics across good solutions
    let mut means = vec![0.0f32; n];
    let mut bias_mean = 0.0f32;
    for sol in &good {
        for i in 0..n { means[i] += sol.weights[i]; }
        bias_mean += sol.bias;
    }
    let ng = good.len() as f32;
    for m in means.iter_mut() { *m /= ng; }
    bias_mean /= ng;

    let mut stds = vec![0.0f32; n];
    for sol in &good {
        for i in 0..n { stds[i] += (sol.weights[i] - means[i]).powi(2); }
    }
    for s in stds.iter_mut() { *s = (*s / ng).sqrt(); }

    // Sign distribution
    let mut signs = vec![[0usize; 3]; n];
    for sol in &good {
        for i in 0..n {
            if sol.weights[i] < -0.3 { signs[i][0] += 1; }
            else if sol.weights[i] > 0.3 { signs[i][2] += 1; }
            else { signs[i][1] += 1; }
        }
    }

    // Generate suggested ternary combos from the landscape
    // For each weight: if >80% agree on sign → lock it; otherwise try both
    let mut suggested = Vec::new();
    let threshold = (good.len() as f32 * 0.6) as usize;

    // Determine locked vs free positions
    let mut locked: Vec<Option<i8>> = vec![None; n];
    let mut free_positions = Vec::new();
    for i in 0..n {
        if signs[i][2] >= threshold { locked[i] = Some(1); }
        else if signs[i][0] >= threshold { locked[i] = Some(-1); }
        else if signs[i][1] >= threshold { locked[i] = Some(0); }
        else { free_positions.push(i); }
    }

    // Generate combos: locked positions fixed, free positions try all 3
    let n_free = free_positions.len();
    let n_combos = 3u32.pow(n_free as u32);

    // Also try 3 bias values
    for bias_try in [-1i8, 0, 1] {
        for combo in 0..n_combos {
            let mut w = vec![0i8; n];
            for i in 0..n {
                w[i] = locked[i].unwrap_or(0);
            }
            let mut r = combo;
            for &fp in &free_positions {
                w[fp] = (r % 3) as i8 - 1;
                r /= 3;
            }
            suggested.push(w);
        }
        let _ = bias_try; // used in the ternary eval phase
    }

    // Deduplicate
    suggested.sort();
    suggested.dedup();

    (solutions, LandscapeReport {
        n_seeds, n_perfect,
        weight_means: means, weight_stds: stds, weight_signs: signs,
        bias_mean,
        suggested_ternary: suggested,
    })
}

// ── Guided ternary search ──

fn guided_ternary_search(corpus: &Corpus, suggested: &[Vec<i8>]) -> (Vec<i8>, i8, i32, f32) {
    let n = corpus.n_in;
    let n_pat = corpus.data.len();

    let mut best_w = vec![0i8; n];
    let mut best_b: i8 = 0;
    let mut best_t: i32 = 0;
    let mut best_score = 0usize;

    for w_try in suggested {
        for b_try in [-1i8, 0, 1] {
            // Compute dots
            let dots: Vec<i32> = corpus.data.iter().map(|(x, _)| {
                let mut d = b_try as i32;
                for (wi, xi) in w_try.iter().zip(x) { d += (*wi as i32) * (*xi as i32); }
                d
            }).collect();

            let min_d = dots.iter().copied().min().unwrap_or(0);
            let max_d = dots.iter().copied().max().unwrap_or(0);

            for thresh in (min_d - 1)..=(max_d + 1) {
                let score = dots.iter().zip(&corpus.data).filter(|(&d, (_, y))| {
                    let pred = if d >= thresh { 1.0f32 } else { 0.0 };
                    (pred - y).abs() < 0.01
                }).count();
                if score > best_score {
                    best_score = score;
                    best_w = w_try.clone(); best_b = b_try; best_t = thresh;
                    if score == n_pat { return (best_w, best_b, best_t, 100.0); }
                }
            }
        }
    }

    (best_w, best_b, best_t, best_score as f32 / n_pat as f32 * 100.0)
}

// ── Blind ternary exhaustive (for comparison) ──

fn blind_ternary_exhaustive(corpus: &Corpus) -> (Vec<i8>, i8, i32, f32, u64) {
    let n = corpus.n_in;
    let nw = n + 1;
    let total = 3u64.pow(nw as u32);
    let n_pat = corpus.data.len();
    let mut best_w = vec![0i8; n];
    let mut best_b: i8 = 0;
    let mut best_t: i32 = 0;
    let mut best_score = 0usize;
    let mut combos_tried = 0u64;

    for combo in 0..total {
        let mut w = vec![0i8; n];
        let mut r = combo;
        for wi in w.iter_mut() { *wi = (r % 3) as i8 - 1; r /= 3; }
        let b = (r % 3) as i8 - 1;

        let dots: Vec<i32> = corpus.data.iter().map(|(x, _)| {
            let mut d = b as i32;
            for (wi, xi) in w.iter().zip(x) { d += (*wi as i32) * (*xi as i32); }
            d
        }).collect();
        let min_d = dots.iter().copied().min().unwrap_or(0);
        let max_d = dots.iter().copied().max().unwrap_or(0);

        for thresh in (min_d - 1)..=(max_d + 1) {
            combos_tried += 1;
            let score = dots.iter().zip(&corpus.data).filter(|(&d, (_, y))| {
                let pred = if d >= thresh { 1.0f32 } else { 0.0 };
                (pred - y).abs() < 0.01
            }).count();
            if score > best_score {
                best_score = score;
                best_w = w.clone(); best_b = b; best_t = thresh;
                if score == n_pat { return (best_w, best_b, best_t, 100.0, combos_tried); }
            }
        }
    }
    (best_w, best_b, best_t, best_score as f32 / n_pat as f32 * 100.0, combos_tried)
}

// ── Visualization helpers ──

fn sign_bar(counts: &[usize; 3], total: usize) -> String {
    let w = 20;
    let neg = counts[0] * w / total.max(1);
    let zero = counts[1] * w / total.max(1);
    let pos = counts[2] * w / total.max(1);
    let rest = w - neg - zero - pos;
    format!("[{}{}{}{}]",
        "-".repeat(neg), ".".repeat(zero), "+".repeat(pos), " ".repeat(rest))
}

fn weight_str(w: &[i8]) -> String {
    w.iter().map(|&v| match v { 1 => "+", -1 => "-", _ => "0" }).collect::<Vec<_>>().join("")
}

// ── Main pipeline ──

fn run_pipeline(corpus: &Corpus) {
    println!("\n  ══ {} ══", corpus.name);
    println!("  {}", corpus.desc);
    println!("  {} inputs, {} patterns\n", corpus.n_in, corpus.data.len());

    // Phase 1: Float landscape
    let n_seeds = 200;
    println!("  Phase 1: Float backprop x {} seeds...", n_seeds);
    let t0 = Instant::now();
    let (solutions, report) = analyze_landscape(corpus, n_seeds);
    let phase1_time = t0.elapsed();
    println!("    Best accuracy: {:.1}%  ({} perfect solutions)", solutions[0].accuracy, report.n_perfect);
    println!("    Time: {:.0}ms\n", phase1_time.as_millis());

    // Phase 2: Landscape visualization
    println!("  Phase 2: Loss landscape");
    println!("    Weight sign distribution (across top solutions):");
    println!("    {:>4}  {:>6} {:>6} {:>6}  {:22}  mean±std", "w_i", "neg", "zero", "pos", "distribution");
    for i in 0..corpus.n_in {
        let s = &report.weight_signs[i];
        let total = s[0] + s[1] + s[2];
        let bar = sign_bar(s, total);
        let locked = if s[2] as f32 / total as f32 > 0.6 { " → LOCK +" }
            else if s[0] as f32 / total as f32 > 0.6 { " → LOCK -" }
            else if s[1] as f32 / total as f32 > 0.6 { " → LOCK 0" }
            else { " → FREE" };
        println!("    w{:>2}: {:>5} {:>5} {:>5}  {}  {:+.2}±{:.2}{}",
            i, s[0], s[1], s[2], bar,
            report.weight_means[i], report.weight_stds[i], locked);
    }
    println!("    bias mean: {:+.2}", report.bias_mean);

    let n_free: usize = report.weight_signs.iter().filter(|s| {
        let total = (s[0] + s[1] + s[2]) as f32;
        s[0] as f32 / total < 0.6 && s[1] as f32 / total < 0.6 && s[2] as f32 / total < 0.6
    }).count();
    let n_locked = corpus.n_in - n_free;
    println!("\n    Locked: {}, Free: {} → search space: {} (vs {} blind)",
        n_locked, n_free,
        report.suggested_ternary.len() * 3, // ×3 for bias
        3u64.pow((corpus.n_in + 1) as u32));

    // Phase 3: Guided ternary search
    println!("\n  Phase 3: Guided ternary search ({} combos)...", report.suggested_ternary.len() * 3);
    let t1 = Instant::now();
    let (gw, gb, gt, gacc) = guided_ternary_search(corpus, &report.suggested_ternary);
    let guided_time = t1.elapsed();
    println!("    Result: [{}] b={:+} t={}  acc={:.1}%  ({:.0}ms)",
        weight_str(&gw), gb, gt, gacc, guided_time.as_millis());

    // Phase 4: Blind exhaustive (for comparison)
    println!("\n  Phase 4: Blind exhaustive (comparison)...");
    let t2 = Instant::now();
    let (bw, bb, bt, bacc, combos) = blind_ternary_exhaustive(corpus);
    let blind_time = t2.elapsed();
    println!("    Result: [{}] b={:+} t={}  acc={:.1}%  ({:.0}ms, {} combos)",
        weight_str(&bw), bb, bt, bacc, blind_time.as_millis(), combos);

    // Summary
    println!("\n  ── Summary ──");
    println!("    Float best:     {:.1}%", solutions[0].accuracy);
    println!("    Guided ternary: {:.1}%  (search: {:.0}ms)", gacc, guided_time.as_millis());
    println!("    Blind ternary:  {:.1}%  (search: {:.0}ms)", bacc, blind_time.as_millis());
    let speedup = blind_time.as_micros() as f64 / guided_time.as_micros().max(1) as f64;
    println!("    Speedup:        {:.1}x", speedup);
    if gacc >= bacc {
        println!("    Guided found SAME or BETTER result!");
    } else {
        println!("    ⚠ Blind found better ({:.1}% vs {:.1}%) — landscape missed a zone", bacc, gacc);
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Landscape-Guided Neuron Build                             ║");
    println!("║  Float landscape → guided ternary search                   ║");
    println!("║  200 seeds × backprop → sign analysis → targeted search    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    let corpora = vec![
        corpus_bigger_half(),
        corpus_compare(),
        corpus_xor(),
        corpus_parity4(),
        corpus_pattern110(),
    ];

    for corpus in &corpora {
        run_pipeline(corpus);
    }

    println!("\n══════════════════════════════════════════════════════════════");
    println!("  Pipeline: float backprop (map landscape) → lock obvious signs");
    println!("  → search only free positions → verify exhaustively");
    println!("  Next: freeze best neuron, add 2nd neuron, repeat");
    println!("══════════════════════════════════════════════════════════════");
}
