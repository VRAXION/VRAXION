//! Sparse-Dense Sandwich: recurrent tick-based architecture
//!
//! Layer A (Sparse): reads input + K dense neurons (feedback)
//! Layer B (Dense): reads K sparse neurons + all-to-all within dense layer
//! Alternating ticks: A fires, B fires, A fires, B fires...
//! Same weights every cycle — depth scales with tick count!
//!
//! Run: cargo run --example sparse_dense_tick --release

use rayon::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, seq::SliceRandom};
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
// Sandwich Network
// ============================================================

#[derive(Clone)]
struct SWorker {
    weights: Vec<i8>,   // [input:8] [dense_reads:K] [bias:1]
    c_val: f32,
    reads_b: Vec<usize>, // indices of dense neurons this sparse reads
}

#[derive(Clone)]
struct DWorker {
    weights: Vec<i8>,   // [sparse_reads:K] [dense_neighbors:i] [bias:1]
    c_val: f32,
    reads_a: Vec<usize>, // indices of sparse neurons this dense reads
}

#[derive(Clone)]
struct SandwichNet {
    ticks: usize,
    a: Vec<SWorker>,  // sparse layer
    b: Vec<DWorker>,  // dense layer
}

impl SandwichNet {
    fn new(ticks: usize) -> Self {
        SandwichNet { ticks, a: Vec::new(), b: Vec::new() }
    }

    fn forward(&self, ia: usize, ib: usize) -> f32 {
        let input = thermo_2(ia, ib);
        let na = self.a.len();
        let nb = self.b.len();
        if na == 0 && nb == 0 { return 0.0; }

        let mut act_a = vec![0.0f32; na];
        let mut act_b = vec![0.0f32; nb];

        for t in 0..self.ticks {
            if t % 2 == 0 {
                // Sparse layer fires
                for (i, w) in self.a.iter().enumerate() {
                    let bias_idx = INPUT_DIM + w.reads_b.len();
                    let mut s = w.weights[bias_idx] as f32;
                    for j in 0..INPUT_DIM {
                        s += input[j] * w.weights[j] as f32;
                    }
                    for (k, &di) in w.reads_b.iter().enumerate() {
                        if di < nb { s += act_b[di] * w.weights[INPUT_DIM + k] as f32; }
                    }
                    act_a[i] = c19(s, w.c_val);
                }
            } else {
                // Dense layer fires
                let old_b = act_b.clone();
                for (i, w) in self.b.iter().enumerate() {
                    let ka = w.reads_a.len();
                    let bias_idx = ka + i;  // [sparse:ka] [dense_neighbors:i] [bias:1]
                    let mut s = w.weights[bias_idx] as f32;
                    // Sparse reads
                    for (k, &si) in w.reads_a.iter().enumerate() {
                        if si < na { s += act_a[si] * w.weights[k] as f32; }
                    }
                    // Dense all-to-all (all previous dense neurons)
                    for j in 0..i {
                        s += old_b[j] * w.weights[ka + j] as f32;
                    }
                    act_b[i] = c19(s, w.c_val);
                }
            }
        }

        act_a.iter().sum::<f32>() + act_b.iter().sum::<f32>()
    }

    fn accuracy(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut c = 0;
        for a in 0..DIGITS { for b in 0..DIGITS {
            if (self.forward(a, b).round() as i32) == (op(a, b) as i32) { c += 1; }
        }}
        c as f64 / 25.0
    }

    fn mse(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut l = 0.0f64;
        for a in 0..DIGITS { for b in 0..DIGITS {
            let d = self.forward(a, b) as f64 - op(a, b) as f64;
            l += d * d;
        }}
        l / 25.0
    }

    /// Add best sparse neuron via exhaustive search
    fn add_sparse(
        &mut self, k: usize,
        op: fn(usize, usize) -> usize, c_step: f32, c_max: f32,
        rng: &mut StdRng,
    ) -> (f64, f32) {
        let nb = self.b.len();
        // Which dense neurons to read: all if k >= nb, else random K
        let reads_b: Vec<usize> = if k >= nb || nb == 0 {
            (0..nb).collect()
        } else {
            let mut indices: Vec<usize> = (0..nb).collect();
            indices.shuffle(rng);
            indices[..k].to_vec()
        };

        let kr = reads_b.len();
        let np = INPUT_DIM + kr + 1;  // input + dense reads + bias
        let total_binary: u32 = 1 << np;
        let c_steps: Vec<f32> = {
            let mut v = Vec::new();
            let mut c = c_step;
            while c <= c_max { v.push(c); c += c_step; }
            v
        };

        let base = self.clone();
        let rb = reads_b.clone();

        let results: Vec<(f64, f64, f32, Vec<i8>)> = c_steps.par_iter().map(|&c_val| {
            let mut best_acc = 0.0f64;
            let mut best_mse = f64::MAX;
            let mut best_w: Vec<i8> = vec![-1; np];
            for config in 0..total_binary {
                let w: Vec<i8> = (0..np).map(|bit|
                    if (config >> bit) & 1 == 1 { 1i8 } else { -1i8 }
                ).collect();
                let mut net = base.clone();
                net.a.push(SWorker { weights: w.clone(), c_val, reads_b: rb.clone() });
                let acc = net.accuracy(op);
                let mse = net.mse(op);
                if acc > best_acc || (acc == best_acc && mse < best_mse) {
                    best_acc = acc; best_mse = mse; best_w = w;
                }
            }
            (best_acc, best_mse, c_val, best_w)
        }).collect();

        let mut best = &results[0];
        for r in &results { if r.0 > best.0 || (r.0 == best.0 && r.1 < best.1) { best = r; } }

        let bc = best.2;
        let bw = best.3.clone();
        let ba = best.0;
        self.a.push(SWorker { weights: bw, c_val: bc, reads_b });
        (ba, bc)
    }

    /// Add best dense neuron via exhaustive search
    fn add_dense(
        &mut self, k: usize,
        op: fn(usize, usize) -> usize, c_step: f32, c_max: f32,
        rng: &mut StdRng,
    ) -> (f64, f32) {
        let na = self.a.len();
        let nb = self.b.len(); // current dense count = index of new neuron = # neighbors
        // Which sparse neurons to read
        let reads_a: Vec<usize> = if k >= na || na == 0 {
            (0..na).collect()
        } else {
            let mut indices: Vec<usize> = (0..na).collect();
            indices.shuffle(rng);
            indices[..k].to_vec()
        };

        let ka = reads_a.len();
        let np = ka + nb + 1;  // sparse reads + dense neighbors + bias
        let total_binary: u32 = 1 << np;
        let c_steps: Vec<f32> = {
            let mut v = Vec::new();
            let mut c = c_step;
            while c <= c_max { v.push(c); c += c_step; }
            v
        };

        let base = self.clone();
        let ra = reads_a.clone();

        let results: Vec<(f64, f64, f32, Vec<i8>)> = c_steps.par_iter().map(|&c_val| {
            let mut best_acc = 0.0f64;
            let mut best_mse = f64::MAX;
            let mut best_w: Vec<i8> = vec![-1; np];
            for config in 0..total_binary {
                let w: Vec<i8> = (0..np).map(|bit|
                    if (config >> bit) & 1 == 1 { 1i8 } else { -1i8 }
                ).collect();
                let mut net = base.clone();
                net.b.push(DWorker { weights: w.clone(), c_val, reads_a: ra.clone() });
                let acc = net.accuracy(op);
                let mse = net.mse(op);
                if acc > best_acc || (acc == best_acc && mse < best_mse) {
                    best_acc = acc; best_mse = mse; best_w = w;
                }
            }
            (best_acc, best_mse, c_val, best_w)
        }).collect();

        let mut best = &results[0];
        for r in &results { if r.0 > best.0 || (r.0 == best.0 && r.1 < best.1) { best = r; } }

        let bc = best.2;
        let bw = best.3.clone();
        let ba = best.0;
        self.b.push(DWorker { weights: bw, c_val: bc, reads_a });
        (ba, bc)
    }
}

// ============================================================
// Build helper
// ============================================================

fn build_sandwich(
    n_a: usize, n_b: usize, k: usize, ticks: usize,
    op: fn(usize, usize) -> usize, c_step: f32, c_max: f32,
    seed: u64, verbose: bool,
) -> SandwichNet {
    let mut net = SandwichNet::new(ticks);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut added_a = 0usize;
    let mut added_b = 0usize;

    for step in 0..(n_a + n_b) {
        let do_a = if added_a >= n_a { false }
                   else if added_b >= n_b { true }
                   else { added_a <= added_b };

        if do_a {
            let np = INPUT_DIM + added_b.min(k) + 1;
            if verbose {
                print!("    S[{}]: 2^{} ... ", added_a, np);
                std::io::stdout().flush().unwrap();
            }
            let t0 = Instant::now();
            let (acc, cv) = net.add_sparse(k, op, c_step, c_max, &mut rng);
            if verbose {
                println!("C={:.2} acc={:>5.1}% ({:.2}s)", cv, acc*100.0, t0.elapsed().as_secs_f64());
            }
            added_a += 1;
        } else {
            let ka = added_a.min(k);
            let np = ka + added_b + 1;
            if verbose {
                print!("    D[{}]: 2^{} ... ", added_b, np);
                std::io::stdout().flush().unwrap();
            }
            let t0 = Instant::now();
            let (acc, cv) = net.add_dense(k, op, c_step, c_max, &mut rng);
            if verbose {
                println!("C={:.2} acc={:>5.1}% ({:.2}s)", cv, acc*100.0, t0.elapsed().as_secs_f64());
            }
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
    println!("=== SPARSE-DENSE SANDWICH: recurrent tick architecture ===\n");
    println!("Sparse (A): reads input + K dense neurons (feedback)");
    println!("Dense  (B): reads K sparse neurons + all-to-all within B");
    println!("Alternating ticks, same weights reused every cycle\n");

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
    // TEST 1: Basic viability (4+4, K=all, ticks=4)
    // =========================================================
    println!("--- TEST 1: Does it work? (4S + 4D, K=all, ticks=4) ---\n");
    println!("  ADD build:");
    let demo = build_sandwich(4, 4, 99, 4, op_add, c_step, c_max, 42, true);
    println!("  Final: acc={:.1}% mse={:.4}\n", demo.accuracy(op_add)*100.0, demo.mse(op_add));

    println!("  All tasks:");
    println!("  {:>6} {:>6} {:>6}", "task", "acc%", "mse");
    println!("  {}", "=".repeat(22));
    for &(name, op) in &ops {
        let t0 = Instant::now();
        let net = build_sandwich(4, 4, 99, 4, op, c_step, c_max, 42, false);
        println!("  {:>6} {:>5.1}% {:>6.4}  ({:.1}s)",
            name, net.accuracy(op)*100.0, net.mse(op), t0.elapsed().as_secs_f64());
        std::io::stdout().flush().unwrap();
    }

    // =========================================================
    // TEST 2: Tick sweep — how many ticks needed?
    // =========================================================
    println!("\n--- TEST 2: Tick sweep (4S + 4D, K=all) ---\n");
    println!("  {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "task", "t=2", "t=4", "t=6", "t=8", "t=12", "t=16");
    println!("  {}", "=".repeat(46));

    for &(name, op) in &[("ADD", op_add as fn(usize,usize)->usize),
                          ("MIN", op_min), ("|a-b|", op_sub_abs)] {
        print!("  {:>6}", name);
        std::io::stdout().flush().unwrap();
        for &ticks in &[2, 4, 6, 8, 12, 16] {
            let net = build_sandwich(4, 4, 99, ticks, op, c_step, c_max, 42, false);
            print!(" {:>5.0}%", net.accuracy(op)*100.0);
            std::io::stdout().flush().unwrap();
        }
        println!();
    }

    // =========================================================
    // TEST 3: K sweep — sparse connections between layers
    // =========================================================
    println!("\n--- TEST 3: K sweep (4S + 4D, ticks=4) ---\n");
    println!("  {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "task", "K=1", "K=2", "K=3", "K=all", "mse_best");
    println!("  {}", "=".repeat(42));

    for &(name, op) in &[("ADD", op_add as fn(usize,usize)->usize),
                          ("MIN", op_min), ("|a-b|", op_sub_abs), ("MUL", op_mul)] {
        print!("  {:>6}", name);
        std::io::stdout().flush().unwrap();
        let mut best_mse = f64::MAX;
        for &k in &[1, 2, 3, 99] {
            let net = build_sandwich(4, 4, k, 4, op, c_step, c_max, 42, false);
            let acc = net.accuracy(op);
            let mse = net.mse(op);
            print!(" {:>5.0}%", acc*100.0);
            if mse < best_mse { best_mse = mse; }
            std::io::stdout().flush().unwrap();
        }
        println!(" {:>8.4}", best_mse);
    }

    // =========================================================
    // TEST 4: Layer ratio (total=8 neurons, ticks=4, K=all)
    // =========================================================
    println!("\n--- TEST 4: Layer ratio (8 neurons total, ticks=4, K=all) ---\n");
    println!("  {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "task", "2S+6D", "3S+5D", "4S+4D", "5S+3D", "6S+2D");
    println!("  {}", "=".repeat(48));

    for &(name, op) in &[("ADD", op_add as fn(usize,usize)->usize),
                          ("MIN", op_min), ("|a-b|", op_sub_abs)] {
        print!("  {:>6}", name);
        std::io::stdout().flush().unwrap();
        for &(na, nb) in &[(2,6), (3,5), (4,4), (5,3), (6,2)] {
            let net = build_sandwich(na, nb, 99, 4, op, c_step, c_max, 42, false);
            print!(" {:>7.0}%", net.accuracy(op)*100.0);
            std::io::stdout().flush().unwrap();
        }
        println!();
    }

    // =========================================================
    // TEST 5: Random seeds — robustness (4+4, K=3, ticks=4)
    // =========================================================
    println!("\n--- TEST 5: Random seeds (4S+4D, K=3, ticks=4, 10 seeds) ---\n");
    println!("  {:>6} {:>8} {:>8} {:>8} {:>8}",
        "task", "solved", "avg_acc", "best_mse", "worst_mse");
    println!("  {}", "=".repeat(42));

    for &(name, op) in &ops {
        let mut solved = 0;
        let mut total_acc = 0.0f64;
        let mut best_mse = f64::MAX;
        let mut worst_mse = 0.0f64;

        for seed in 0..10u64 {
            let net = build_sandwich(4, 4, 3, 4, op, c_step, c_max, seed, false);
            let acc = net.accuracy(op);
            let mse = net.mse(op);
            if acc >= 1.0 { solved += 1; }
            total_acc += acc;
            if mse < best_mse { best_mse = mse; }
            if mse > worst_mse { worst_mse = mse; }
        }
        println!("  {:>6} {:>7}/10 {:>7.1}% {:>8.4} {:>10.4}",
            name, solved, total_acc / 10.0 * 100.0, best_mse, worst_mse);
        std::io::stdout().flush().unwrap();
    }

    // =========================================================
    // TEST 6: Scale up — more neurons (K=all, ticks=4)
    // =========================================================
    println!("\n--- TEST 6: Scale up (K=all, ticks=4) ---\n");
    println!("  {:>6} {:>8} {:>8} {:>8} {:>8}",
        "task", "2+2", "4+4", "6+6", "8+8");
    println!("  {}", "=".repeat(38));

    for &(name, op) in &[("ADD", op_add as fn(usize,usize)->usize),
                          ("|a-b|", op_sub_abs), ("MUL", op_mul)] {
        print!("  {:>6}", name);
        std::io::stdout().flush().unwrap();
        for &n in &[2, 4, 6, 8] {
            let net = build_sandwich(n, n, 99, 4, op, c_step, c_max, 42, false);
            print!(" {:>7.0}%", net.accuracy(op)*100.0);
            std::io::stdout().flush().unwrap();
        }
        println!();
    }

    // =========================================================
    // TEST 7: K% sweep at 8+8 (3 tasks × 4 K × 5 seeds)
    // =========================================================
    println!("\n--- TEST 7: K% sweep (8S+8D, ticks=4, 5 seeds) ---\n");
    println!("  {:>6} {:>10} {:>10} {:>10} {:>10}",
        "task", "K=1(12%)", "K=2(25%)", "K=4(50%)", "K=8(all)");
    println!("  {}", "=".repeat(48));

    for &(name, op) in &[("ADD", op_add as fn(usize,usize)->usize),
                          ("|a-b|", op_sub_abs), ("MUL", op_mul)] {
        print!("  {:>6}", name);
        std::io::stdout().flush().unwrap();
        for &k in &[1usize, 2, 4, 99] {
            let mut solved = 0;
            let mut total_acc = 0.0f64;
            for seed in 0..5u64 {
                let net = build_sandwich(8, 8, k, 4, op, c_step, c_max, seed, false);
                let acc = net.accuracy(op);
                if acc >= 1.0 { solved += 1; }
                total_acc += acc;
            }
            print!(" {:>4}/5 {:>4.0}%", solved, total_acc / 5.0 * 100.0);
            std::io::stdout().flush().unwrap();
        }
        println!();
    }

    // =========================================================
    // TEST 8: Scale check 12+12 (ADD+MUL, K=1,2,4, 3 seeds)
    // =========================================================
    println!("\n--- TEST 8: Scale 12+12 (ticks=4, 3 seeds) ---\n");
    println!("  {:>6} {:>10} {:>10} {:>10}",
        "task", "K=1(8%)", "K=2(17%)", "K=4(33%)");
    println!("  {}", "=".repeat(38));

    for &(name, op) in &[("ADD", op_add as fn(usize,usize)->usize), ("MUL", op_mul)] {
        print!("  {:>6}", name);
        std::io::stdout().flush().unwrap();
        for &k in &[1usize, 2, 4] {
            let mut solved = 0;
            let mut total_acc = 0.0f64;
            for seed in 0..3u64 {
                let net = build_sandwich(12, 12, k, 4, op, c_step, c_max, seed, false);
                let acc = net.accuracy(op);
                if acc >= 1.0 { solved += 1; }
                total_acc += acc;
            }
            print!(" {:>3}/3 {:>4.0}%", solved, total_acc / 3.0 * 100.0);
            std::io::stdout().flush().unwrap();
        }
        println!();
    }

    println!("\n=== DONE ===");
}
