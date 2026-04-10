//! Sandwich-per-tick vs Alternating-layer tick comparison
//!
//! Mode A (alternating): tick 1 = sparse fires, tick 2 = dense fires, ...
//! Mode B (sandwich):    tick 1 = sparse→dense (full pass), tick 2 = sparse→dense again, ...
//!
//! Same weights, same neuron count, same total computation — which converges faster?
//!
//! Run: cargo run --example sandwich_vs_alternating --release

use rayon::prelude::*;
use rand::rngs::StdRng;
use rand::{SeedableRng, seq::SliceRandom};
use std::io::Write;
use std::time::Instant;

const DIGITS: usize = 5;
const INPUT_DIM: usize = 8;
const RHO: f32 = 4.0;

fn c19(x: f32, c: f32) -> f32 {
    let c = c.max(0.01);
    let l = 6.0 * c;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let scaled = x / c;
    let n = scaled.floor();
    let t = scaled - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sgn * h + RHO * h * h)
}

fn thermo_2(a: usize, b: usize) -> [f32; 8] {
    let mut v = [0.0f32; 8];
    for i in 0..a.min(4) { v[i] = 1.0; }
    for i in 0..b.min(4) { v[4 + i] = 1.0; }
    v
}

// ============================================================
// Shared worker types
// ============================================================

#[derive(Clone)]
struct SWorker {
    weights: Vec<i8>,
    c_val: f32,
    reads_b: Vec<usize>,
}

#[derive(Clone)]
struct DWorker {
    weights: Vec<i8>,
    c_val: f32,
    reads_a: Vec<usize>,
}

// ============================================================
// Mode A: Alternating (sparse and dense fire on separate ticks)
// ============================================================

#[derive(Clone)]
struct AlternatingNet {
    ticks: usize,
    a: Vec<SWorker>,
    b: Vec<DWorker>,
}

impl AlternatingNet {
    fn forward(&self, ia: usize, ib: usize) -> f32 {
        let input = thermo_2(ia, ib);
        let na = self.a.len();
        let nb = self.b.len();
        if na == 0 && nb == 0 { return 0.0; }
        let mut act_a = vec![0.0f32; na];
        let mut act_b = vec![0.0f32; nb];

        for t in 0..self.ticks {
            if t % 2 == 0 {
                // Sparse fires
                for (i, w) in self.a.iter().enumerate() {
                    let bias_idx = INPUT_DIM + w.reads_b.len();
                    let mut s = w.weights[bias_idx] as f32;
                    for j in 0..INPUT_DIM { s += input[j] * w.weights[j] as f32; }
                    for (k, &di) in w.reads_b.iter().enumerate() {
                        if di < nb { s += act_b[di] * w.weights[INPUT_DIM + k] as f32; }
                    }
                    act_a[i] = c19(s, w.c_val);
                }
            } else {
                // Dense fires
                let old_b = act_b.clone();
                for (i, w) in self.b.iter().enumerate() {
                    let ka = w.reads_a.len();
                    let bias_idx = ka + i;
                    let mut s = w.weights[bias_idx] as f32;
                    for (k, &si) in w.reads_a.iter().enumerate() {
                        if si < na { s += act_a[si] * w.weights[k] as f32; }
                    }
                    for j in 0..i { s += old_b[j] * w.weights[ka + j] as f32; }
                    act_b[i] = c19(s, w.c_val);
                }
            }
        }
        act_a.iter().sum::<f32>() + act_b.iter().sum::<f32>()
    }
}

// ============================================================
// Mode B: Sandwich (sparse→dense = one full tick)
// ============================================================

#[derive(Clone)]
struct SandwichNet {
    ticks: usize,
    a: Vec<SWorker>,
    b: Vec<DWorker>,
}

impl SandwichNet {
    fn forward(&self, ia: usize, ib: usize) -> f32 {
        let input = thermo_2(ia, ib);
        let na = self.a.len();
        let nb = self.b.len();
        if na == 0 && nb == 0 { return 0.0; }
        let mut act_a = vec![0.0f32; na];
        let mut act_b = vec![0.0f32; nb];

        for _t in 0..self.ticks {
            // Phase A: Sparse fires (reads input + dense feedback)
            for (i, w) in self.a.iter().enumerate() {
                let bias_idx = INPUT_DIM + w.reads_b.len();
                let mut s = w.weights[bias_idx] as f32;
                for j in 0..INPUT_DIM { s += input[j] * w.weights[j] as f32; }
                for (k, &di) in w.reads_b.iter().enumerate() {
                    if di < nb { s += act_b[di] * w.weights[INPUT_DIM + k] as f32; }
                }
                act_a[i] = c19(s, w.c_val);
            }

            // Phase B: Dense fires (reads fresh sparse output + dense neighbors)
            let old_b = act_b.clone();
            for (i, w) in self.b.iter().enumerate() {
                let ka = w.reads_a.len();
                let bias_idx = ka + i;
                let mut s = w.weights[bias_idx] as f32;
                for (k, &si) in w.reads_a.iter().enumerate() {
                    if si < na { s += act_a[si] * w.weights[k] as f32; }
                }
                for j in 0..i { s += old_b[j] * w.weights[ka + j] as f32; }
                act_b[i] = c19(s, w.c_val);
            }
        }
        act_a.iter().sum::<f32>() + act_b.iter().sum::<f32>()
    }
}

// ============================================================
// Mode C: Linear sandwich (sparse = LINEAR edge remap, no C19!)
// ============================================================

#[derive(Clone)]
struct LinearSandwichNet {
    ticks: usize,
    a: Vec<SWorker>,  // sparse layer — weights used but NO activation function
    b: Vec<DWorker>,  // dense layer — C19 as usual
}

impl LinearSandwichNet {
    fn forward(&self, ia: usize, ib: usize) -> f32 {
        let input = thermo_2(ia, ib);
        let na = self.a.len();
        let nb = self.b.len();
        if na == 0 && nb == 0 { return 0.0; }
        let mut act_a = vec![0.0f32; na];
        let mut act_b = vec![0.0f32; nb];

        for _t in 0..self.ticks {
            // Phase A: Sparse = LINEAR (no c19, just weighted sum)
            for (i, w) in self.a.iter().enumerate() {
                let bias_idx = INPUT_DIM + w.reads_b.len();
                let mut s = w.weights[bias_idx] as f32;
                for j in 0..INPUT_DIM { s += input[j] * w.weights[j] as f32; }
                for (k, &di) in w.reads_b.iter().enumerate() {
                    if di < nb { s += act_b[di] * w.weights[INPUT_DIM + k] as f32; }
                }
                act_a[i] = s;  // NO C19! Pure linear remapping
            }

            // Phase B: Dense fires WITH C19 (reads linear sparse output + neighbors)
            let old_b = act_b.clone();
            for (i, w) in self.b.iter().enumerate() {
                let ka = w.reads_a.len();
                let bias_idx = ka + i;
                let mut s = w.weights[bias_idx] as f32;
                for (k, &si) in w.reads_a.iter().enumerate() {
                    if si < na { s += act_a[si] * w.weights[k] as f32; }
                }
                for j in 0..i { s += old_b[j] * w.weights[ka + j] as f32; }
                act_b[i] = c19(s, w.c_val);
            }
        }
        // Output from dense only (sparse is just linear routing)
        act_b.iter().sum::<f32>()
    }
}

impl NetEval for LinearSandwichNet { fn fwd(&self, a: usize, b: usize) -> f32 { self.forward(a, b) } }

// ============================================================
// Generic eval + exhaustive search
// ============================================================

trait NetEval {
    fn accuracy(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut c = 0;
        for a in 0..DIGITS { for b in 0..DIGITS {
            if (self.fwd(a, b).round() as i32) == (op(a, b) as i32) { c += 1; }
        }}
        c as f64 / 25.0
    }
    fn mse(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut l = 0.0f64;
        for a in 0..DIGITS { for b in 0..DIGITS {
            let d = self.fwd(a, b) as f64 - op(a, b) as f64;
            l += d * d;
        }}
        l / 25.0
    }
    fn fwd(&self, a: usize, b: usize) -> f32;
}

impl NetEval for AlternatingNet { fn fwd(&self, a: usize, b: usize) -> f32 { self.forward(a, b) } }
impl NetEval for SandwichNet { fn fwd(&self, a: usize, b: usize) -> f32 { self.forward(a, b) } }

/// Exhaustive search for a sparse worker
fn search_sparse(
    na: usize, nb_available: usize, k: usize,
    c_step: f32, c_max: f32, rng: &mut StdRng,
    eval_fn: &(dyn Fn(&SWorker) -> (f64, f64) + Sync),
) -> SWorker {
    let reads_b: Vec<usize> = if k >= nb_available || nb_available == 0 {
        (0..nb_available).collect()
    } else {
        let mut idx: Vec<usize> = (0..nb_available).collect();
        idx.shuffle(rng);
        idx[..k].to_vec()
    };
    let kr = reads_b.len();
    let np = INPUT_DIM + kr + 1;
    let total: u32 = 1 << np;
    let c_steps: Vec<f32> = {
        let mut v = Vec::new(); let mut c = c_step;
        while c <= c_max { v.push(c); c += c_step; } v
    };
    let rb = reads_b.clone();

    let results: Vec<(f64, f64, f32, Vec<i8>)> = c_steps.par_iter().map(|&c_val| {
        let mut best_acc = 0.0f64; let mut best_mse = f64::MAX;
        let mut best_w: Vec<i8> = vec![-1; np];
        for config in 0..total {
            let w: Vec<i8> = (0..np).map(|bit|
                if (config >> bit) & 1 == 1 { 1i8 } else { -1i8 }).collect();
            let worker = SWorker { weights: w.clone(), c_val, reads_b: rb.clone() };
            let (acc, mse) = eval_fn(&worker);
            if acc > best_acc || (acc == best_acc && mse < best_mse) {
                best_acc = acc; best_mse = mse; best_w = w;
            }
        }
        (best_acc, best_mse, c_val, best_w)
    }).collect();

    let mut best = &results[0];
    for r in &results { if r.0 > best.0 || (r.0 == best.0 && r.1 < best.1) { best = r; } }
    SWorker { weights: best.3.clone(), c_val: best.2, reads_b }
}

/// Exhaustive search for a dense worker
fn search_dense(
    na_available: usize, nb_existing: usize, k: usize,
    c_step: f32, c_max: f32, rng: &mut StdRng,
    eval_fn: &(dyn Fn(&DWorker) -> (f64, f64) + Sync),
) -> DWorker {
    let reads_a: Vec<usize> = if k >= na_available || na_available == 0 {
        (0..na_available).collect()
    } else {
        let mut idx: Vec<usize> = (0..na_available).collect();
        idx.shuffle(rng);
        idx[..k].to_vec()
    };
    let ka = reads_a.len();
    let np = ka + nb_existing + 1;
    let total: u32 = 1 << np;
    let c_steps: Vec<f32> = {
        let mut v = Vec::new(); let mut c = c_step;
        while c <= c_max { v.push(c); c += c_step; } v
    };
    let ra = reads_a.clone();

    let results: Vec<(f64, f64, f32, Vec<i8>)> = c_steps.par_iter().map(|&c_val| {
        let mut best_acc = 0.0f64; let mut best_mse = f64::MAX;
        let mut best_w: Vec<i8> = vec![-1; np];
        for config in 0..total {
            let w: Vec<i8> = (0..np).map(|bit|
                if (config >> bit) & 1 == 1 { 1i8 } else { -1i8 }).collect();
            let worker = DWorker { weights: w.clone(), c_val, reads_a: ra.clone() };
            let (acc, mse) = eval_fn(&worker);
            if acc > best_acc || (acc == best_acc && mse < best_mse) {
                best_acc = acc; best_mse = mse; best_w = w;
            }
        }
        (best_acc, best_mse, c_val, best_w)
    }).collect();

    let mut best = &results[0];
    for r in &results { if r.0 > best.0 || (r.0 == best.0 && r.1 < best.1) { best = r; } }
    DWorker { weights: best.3.clone(), c_val: best.2, reads_a }
}

// ============================================================
// Build functions
// ============================================================

fn build_alternating(
    n_a: usize, n_b: usize, k: usize, ticks: usize,
    op: fn(usize, usize) -> usize, c_step: f32, c_max: f32, seed: u64,
) -> AlternatingNet {
    let mut net = AlternatingNet { ticks, a: Vec::new(), b: Vec::new() };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut added_a = 0usize;
    let mut added_b = 0usize;

    for _ in 0..(n_a + n_b) {
        let do_a = if added_a >= n_a { false }
                   else if added_b >= n_b { true }
                   else { added_a <= added_b };
        if do_a {
            let base = net.clone();
            let worker = search_sparse(added_a, added_b, k, c_step, c_max, &mut rng,
                &|w| {
                    let mut t = base.clone();
                    t.a.push(w.clone());
                    (t.accuracy(op), t.mse(op))
                });
            net.a.push(worker);
            added_a += 1;
        } else {
            let base = net.clone();
            let nb_existing = added_b;
            let worker = search_dense(added_a, nb_existing, k, c_step, c_max, &mut rng,
                &|w| {
                    let mut t = base.clone();
                    t.b.push(w.clone());
                    (t.accuracy(op), t.mse(op))
                });
            net.b.push(worker);
            added_b += 1;
        }
    }
    net
}

fn build_sandwich(
    n_a: usize, n_b: usize, k: usize, ticks: usize,
    op: fn(usize, usize) -> usize, c_step: f32, c_max: f32, seed: u64,
) -> SandwichNet {
    let mut net = SandwichNet { ticks, a: Vec::new(), b: Vec::new() };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut added_a = 0usize;
    let mut added_b = 0usize;

    for _ in 0..(n_a + n_b) {
        let do_a = if added_a >= n_a { false }
                   else if added_b >= n_b { true }
                   else { added_a <= added_b };
        if do_a {
            let base = net.clone();
            let worker = search_sparse(added_a, added_b, k, c_step, c_max, &mut rng,
                &|w| {
                    let mut t = base.clone();
                    t.a.push(w.clone());
                    (t.accuracy(op), t.mse(op))
                });
            net.a.push(worker);
            added_a += 1;
        } else {
            let base = net.clone();
            let nb_existing = added_b;
            let worker = search_dense(added_a, nb_existing, k, c_step, c_max, &mut rng,
                &|w| {
                    let mut t = base.clone();
                    t.b.push(w.clone());
                    (t.accuracy(op), t.mse(op))
                });
            net.b.push(worker);
            added_b += 1;
        }
    }
    net
}

fn build_linear_sandwich(
    n_a: usize, n_b: usize, k: usize, ticks: usize,
    op: fn(usize, usize) -> usize, c_step: f32, c_max: f32, seed: u64,
) -> LinearSandwichNet {
    let mut net = LinearSandwichNet { ticks, a: Vec::new(), b: Vec::new() };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut added_a = 0usize;
    let mut added_b = 0usize;

    for _ in 0..(n_a + n_b) {
        let do_a = if added_a >= n_a { false }
                   else if added_b >= n_b { true }
                   else { added_a <= added_b };
        if do_a {
            let base = net.clone();
            // For linear sparse: still search weights+C, but C is ignored in forward
            let worker = search_sparse(added_a, added_b, k, c_step, c_max, &mut rng,
                &|w| {
                    let mut t = base.clone();
                    t.a.push(w.clone());
                    (t.accuracy(op), t.mse(op))
                });
            net.a.push(worker);
            added_a += 1;
        } else {
            let base = net.clone();
            let nb_existing = added_b;
            let worker = search_dense(added_a, nb_existing, k, c_step, c_max, &mut rng,
                &|w| {
                    let mut t = base.clone();
                    t.b.push(w.clone());
                    (t.accuracy(op), t.mse(op))
                });
            net.b.push(worker);
            added_b += 1;
        }
    }
    net
}

fn op_add(a: usize, b: usize) -> usize { a + b }
fn op_mul(a: usize, b: usize) -> usize { a * b }
fn op_max(a: usize, b: usize) -> usize { a.max(b) }
fn op_min(a: usize, b: usize) -> usize { a.min(b) }
fn op_sub_abs(a: usize, b: usize) -> usize { if a > b { a - b } else { b - a } }

// ============================================================
// Main
// ============================================================

fn main() {
    println!("=== SANDWICH vs ALTERNATING tick comparison ===\n");
    println!("Alternating: tick 1=sparse, tick 2=dense, tick 3=sparse, ...");
    println!("Sandwich:    tick 1=sparse->dense, tick 2=sparse->dense, ...\n");
    println!("Same weights, same neurons. Sandwich gets 2x info flow per tick.\n");

    let c_step = 0.1;
    let c_max = 5.0;

    let ops: Vec<(&str, fn(usize, usize) -> usize)> = vec![
        ("ADD",   op_add),
        ("MAX",   op_max),
        ("MIN",   op_min),
        ("|a-b|", op_sub_abs),
        ("MUL",   op_mul),
    ];

    // =========================================================
    // TEST 1: Fair comparison — same total computation
    // Alternating 4 ticks = 2 sparse + 2 dense fires
    // Sandwich 2 ticks = 2 sparse + 2 dense fires (same compute!)
    // =========================================================
    println!("--- TEST 1: Fair comparison (4S+4D, K=all) ---");
    println!("  Alternating 4 ticks = Sandwich 2 ticks (same total compute)\n");

    println!("  {:>6} {:>10} {:>10} {:>10} {:>10}",
        "task", "alt t=4", "sand t=2", "alt t=8", "sand t=4");
    println!("  {}", "=".repeat(48));

    for &(name, op) in &ops {
        print!("  {:>6}", name);
        std::io::stdout().flush().unwrap();

        let alt4 = build_alternating(4, 4, 99, 4, op, c_step, c_max, 42);
        print!(" {:>9.0}%", alt4.accuracy(op) * 100.0);
        std::io::stdout().flush().unwrap();

        let sand2 = build_sandwich(4, 4, 99, 2, op, c_step, c_max, 42);
        print!(" {:>9.0}%", sand2.accuracy(op) * 100.0);
        std::io::stdout().flush().unwrap();

        let alt8 = build_alternating(4, 4, 99, 8, op, c_step, c_max, 42);
        print!(" {:>9.0}%", alt8.accuracy(op) * 100.0);
        std::io::stdout().flush().unwrap();

        let sand4 = build_sandwich(4, 4, 99, 4, op, c_step, c_max, 42);
        print!(" {:>9.0}%", sand4.accuracy(op) * 100.0);
        std::io::stdout().flush().unwrap();

        println!();
    }

    // =========================================================
    // TEST 2: Same tick count (sandwich gets more compute)
    // =========================================================
    println!("\n--- TEST 2: Same tick count (4S+4D, K=all) ---");
    println!("  Sandwich fires BOTH layers per tick = 2x compute per tick\n");

    println!("  {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "task", "mode", "t=2", "t=4", "t=6", "t=8", "t=12");
    println!("  {}", "=".repeat(64));

    for &(name, op) in &[("ADD", op_add as fn(usize,usize)->usize),
                          ("|a-b|", op_sub_abs), ("MUL", op_mul)] {
        // Alternating
        print!("  {:>6} {:>10}", name, "alt");
        std::io::stdout().flush().unwrap();
        for &t in &[2, 4, 6, 8, 12] {
            let net = build_alternating(4, 4, 99, t, op, c_step, c_max, 42);
            print!(" {:>9.0}%", net.accuracy(op) * 100.0);
            std::io::stdout().flush().unwrap();
        }
        println!();

        // Sandwich
        print!("  {:>6} {:>10}", "", "sandwich");
        std::io::stdout().flush().unwrap();
        for &t in &[2, 4, 6, 8, 12] {
            let net = build_sandwich(4, 4, 99, t, op, c_step, c_max, 42);
            print!(" {:>9.0}%", net.accuracy(op) * 100.0);
            std::io::stdout().flush().unwrap();
        }
        println!();
        println!();
    }

    // =========================================================
    // TEST 3: Robustness — 5 seeds, 4+4, K=4
    // =========================================================
    println!("--- TEST 3: Robustness (4S+4D, K=4, 5 seeds) ---\n");
    println!("  {:>6} {:>12} {:>12} {:>12} {:>12}",
        "task", "alt t=4", "sand t=2", "alt t=8", "sand t=4");
    println!("  {}", "=".repeat(54));

    for &(name, op) in &[("ADD", op_add as fn(usize,usize)->usize),
                          ("|a-b|", op_sub_abs), ("MUL", op_mul)] {
        print!("  {:>6}", name);
        std::io::stdout().flush().unwrap();

        for &(mode, ticks) in &[("alt", 4usize), ("sand", 2), ("alt", 8), ("sand", 4)] {
            let mut solved = 0;
            for seed in 0..5u64 {
                let acc = if mode == "alt" {
                    build_alternating(4, 4, 4, ticks, op, c_step, c_max, seed).accuracy(op)
                } else {
                    build_sandwich(4, 4, 4, ticks, op, c_step, c_max, seed).accuracy(op)
                };
                if acc >= 1.0 { solved += 1; }
            }
            print!(" {:>11}/5", solved);
            std::io::stdout().flush().unwrap();
        }
        println!();
    }

    // =========================================================
    // TEST 4: Scale up 8+8 (K=4, fair comparison)
    // =========================================================
    println!("\n--- TEST 4: Scale 8+8 (K=4, 3 seeds, fair compute) ---\n");
    println!("  {:>6} {:>12} {:>12}",
        "task", "alt t=4", "sand t=2");
    println!("  {}", "=".repeat(32));

    for &(name, op) in &[("ADD", op_add as fn(usize,usize)->usize),
                          ("|a-b|", op_sub_abs), ("MUL", op_mul)] {
        print!("  {:>6}", name);
        std::io::stdout().flush().unwrap();

        for &(mode, ticks) in &[("alt", 4usize), ("sand", 2)] {
            let mut solved = 0;
            for seed in 0..3u64 {
                let acc = if mode == "alt" {
                    build_alternating(8, 8, 4, ticks, op, c_step, c_max, seed).accuracy(op)
                } else {
                    build_sandwich(8, 8, 4, ticks, op, c_step, c_max, seed).accuracy(op)
                };
                if acc >= 1.0 { solved += 1; }
            }
            print!(" {:>11}/3", solved);
            std::io::stdout().flush().unwrap();
        }
        println!();
    }

    // =========================================================
    // TEST 5: 3-way comparison — C19 sandwich vs LINEAR sandwich vs alternating
    // =========================================================
    println!("\n--- TEST 5: C19 vs LINEAR sparse layer (sandwich t=2, 4S+4D, K=all) ---");
    println!("  c19_sand:  sparse=C19 neuron, dense=C19 neuron");
    println!("  lin_sand:  sparse=LINEAR (no activation!), dense=C19");
    println!("  alt:       both=C19, alternating ticks\n");

    println!("  {:>6} {:>10} {:>10} {:>10}",
        "task", "alt t=4", "c19 t=2", "linear t=2");
    println!("  {}", "=".repeat(38));

    for &(name, op) in &ops {
        print!("  {:>6}", name);
        std::io::stdout().flush().unwrap();

        let alt = build_alternating(4, 4, 99, 4, op, c_step, c_max, 42);
        print!(" {:>9.0}%", alt.accuracy(op) * 100.0);
        std::io::stdout().flush().unwrap();

        let c19s = build_sandwich(4, 4, 99, 2, op, c_step, c_max, 42);
        print!(" {:>9.0}%", c19s.accuracy(op) * 100.0);
        std::io::stdout().flush().unwrap();

        let lins = build_linear_sandwich(4, 4, 99, 2, op, c_step, c_max, 42);
        print!(" {:>9.0}%", lins.accuracy(op) * 100.0);
        std::io::stdout().flush().unwrap();

        println!();
    }

    // =========================================================
    // TEST 6b: Linear sandwich robustness (5 seeds, K=4)
    // =========================================================
    println!("\n--- TEST 6: Linear vs C19 sparse, 5 seeds (4+4, K=4, t=2) ---\n");
    println!("  {:>6} {:>12} {:>12}",
        "task", "c19_sand", "lin_sand");
    println!("  {}", "=".repeat(30));

    for &(name, op) in &[("ADD", op_add as fn(usize,usize)->usize),
                          ("|a-b|", op_sub_abs), ("MUL", op_mul)] {
        print!("  {:>6}", name);
        std::io::stdout().flush().unwrap();

        for mode in &["c19", "linear"] {
            let mut solved = 0;
            for seed in 0..5u64 {
                let acc = if *mode == "c19" {
                    build_sandwich(4, 4, 4, 2, op, c_step, c_max, seed).accuracy(op)
                } else {
                    build_linear_sandwich(4, 4, 4, 2, op, c_step, c_max, seed).accuracy(op)
                };
                if acc >= 1.0 { solved += 1; }
            }
            print!(" {:>11}/5", solved);
            std::io::stdout().flush().unwrap();
        }
        println!();
    }

    // =========================================================
    // TEST 7b: Linear sandwich tick sweep
    // =========================================================
    println!("\n--- TEST 7: Linear sandwich tick sweep (4+4, K=all) ---\n");
    println!("  {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "task", "mode", "t=1", "t=2", "t=4", "t=8");
    println!("  {}", "=".repeat(54));

    for &(name, op) in &[("ADD", op_add as fn(usize,usize)->usize),
                          ("|a-b|", op_sub_abs), ("MUL", op_mul)] {
        // C19 sandwich
        print!("  {:>6} {:>10}", name, "c19");
        std::io::stdout().flush().unwrap();
        for &t in &[1, 2, 4, 8] {
            let net = build_sandwich(4, 4, 99, t, op, c_step, c_max, 42);
            print!(" {:>9.0}%", net.accuracy(op) * 100.0);
            std::io::stdout().flush().unwrap();
        }
        println!();

        // Linear sandwich
        print!("  {:>6} {:>10}", "", "linear");
        std::io::stdout().flush().unwrap();
        for &t in &[1, 2, 4, 8] {
            let net = build_linear_sandwich(4, 4, 99, t, op, c_step, c_max, 42);
            print!(" {:>9.0}%", net.accuracy(op) * 100.0);
            std::io::stdout().flush().unwrap();
        }
        println!();
        println!();
    }

    println!("\n=== DONE ===");
}
