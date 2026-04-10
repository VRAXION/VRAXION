//! AB vs BA order + ratio sweep
//!
//! AB: sparse fires first → dense reads sparse output
//! BA: dense fires first → sparse reads dense output
//! + ratio sweep: how many sparse vs dense neurons?
//!
//! Run: cargo run --example ab_vs_ba_ratio --release

use rayon::prelude::*;
use rand::rngs::StdRng;
use rand::{SeedableRng, seq::SliceRandom};
use std::io::Write;

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

#[derive(Clone)]
struct SWorker { weights: Vec<i8>, c_val: f32, reads_b: Vec<usize> }
#[derive(Clone)]
struct DWorker { weights: Vec<i8>, c_val: f32, reads_a: Vec<usize> }

/// Configurable order sandwich
#[derive(Clone)]
struct OrderNet {
    ticks: usize,
    sparse_first: bool,  // true=AB (sparse→dense), false=BA (dense→sparse)
    a: Vec<SWorker>,      // sparse: reads input + dense feedback
    b: Vec<DWorker>,      // dense: reads sparse + all-to-all within
}

impl OrderNet {
    fn forward(&self, ia: usize, ib: usize) -> f32 {
        let input = thermo_2(ia, ib);
        let na = self.a.len();
        let nb = self.b.len();
        if na == 0 && nb == 0 { return 0.0; }
        let mut act_a = vec![0.0f32; na];
        let mut act_b = vec![0.0f32; nb];

        for _t in 0..self.ticks {
            if self.sparse_first {
                // AB: sparse first, then dense
                self.fire_sparse(&input, &mut act_a, &act_b, na, nb);
                self.fire_dense(&mut act_b, &act_a, na, nb);
            } else {
                // BA: dense first, then sparse
                self.fire_dense(&mut act_b, &act_a, na, nb);
                self.fire_sparse(&input, &mut act_a, &act_b, na, nb);
            }
        }
        act_a.iter().sum::<f32>() + act_b.iter().sum::<f32>()
    }

    fn fire_sparse(&self, input: &[f32; 8], act_a: &mut Vec<f32>, act_b: &Vec<f32>, _na: usize, nb: usize) {
        for (i, w) in self.a.iter().enumerate() {
            let bias_idx = INPUT_DIM + w.reads_b.len();
            let mut s = w.weights[bias_idx] as f32;
            for j in 0..INPUT_DIM { s += input[j] * w.weights[j] as f32; }
            for (k, &di) in w.reads_b.iter().enumerate() {
                if di < nb { s += act_b[di] * w.weights[INPUT_DIM + k] as f32; }
            }
            act_a[i] = c19(s, w.c_val);
        }
    }

    fn fire_dense(&self, act_b: &mut Vec<f32>, act_a: &Vec<f32>, na: usize, _nb: usize) {
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
}

fn build_order_net(
    n_a: usize, n_b: usize, k: usize, ticks: usize, sparse_first: bool,
    op: fn(usize, usize) -> usize, c_step: f32, c_max: f32, seed: u64,
) -> OrderNet {
    let mut net = OrderNet { ticks, sparse_first, a: Vec::new(), b: Vec::new() };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut added_a = 0usize;
    let mut added_b = 0usize;

    let c_steps: Vec<f32> = {
        let mut v = Vec::new(); let mut c = c_step;
        while c <= c_max { v.push(c); c += c_step; } v
    };

    for _ in 0..(n_a + n_b) {
        let do_a = if added_a >= n_a { false }
                   else if added_b >= n_b { true }
                   else { added_a <= added_b };
        if do_a {
            // Add sparse worker
            let reads_b: Vec<usize> = if k >= added_b || added_b == 0 {
                (0..added_b).collect()
            } else {
                let mut idx: Vec<usize> = (0..added_b).collect();
                idx.shuffle(&mut rng); idx[..k].to_vec()
            };
            let kr = reads_b.len();
            let np = INPUT_DIM + kr + 1;
            let total: u32 = 1 << np;
            let base = net.clone();
            let rb = reads_b.clone();

            let results: Vec<(f64, f64, f32, Vec<i8>)> = c_steps.par_iter().map(|&c_val| {
                let mut best_acc = 0.0f64; let mut best_mse = f64::MAX;
                let mut best_w = vec![-1i8; np];
                for config in 0..total {
                    let w: Vec<i8> = (0..np).map(|bit|
                        if (config >> bit) & 1 == 1 { 1i8 } else { -1i8 }).collect();
                    let mut t = base.clone();
                    t.a.push(SWorker { weights: w.clone(), c_val, reads_b: rb.clone() });
                    let acc = t.accuracy(op); let mse = t.mse(op);
                    if acc > best_acc || (acc == best_acc && mse < best_mse) {
                        best_acc = acc; best_mse = mse; best_w = w;
                    }
                }
                (best_acc, best_mse, c_val, best_w)
            }).collect();
            let mut best = &results[0];
            for r in &results { if r.0 > best.0 || (r.0 == best.0 && r.1 < best.1) { best = r; } }
            net.a.push(SWorker { weights: best.3.clone(), c_val: best.2, reads_b });
            added_a += 1;
        } else {
            // Add dense worker
            let reads_a: Vec<usize> = if k >= added_a || added_a == 0 {
                (0..added_a).collect()
            } else {
                let mut idx: Vec<usize> = (0..added_a).collect();
                idx.shuffle(&mut rng); idx[..k].to_vec()
            };
            let ka = reads_a.len();
            let np = ka + added_b + 1;
            let total: u32 = 1 << np;
            let base = net.clone();
            let ra = reads_a.clone();

            let results: Vec<(f64, f64, f32, Vec<i8>)> = c_steps.par_iter().map(|&c_val| {
                let mut best_acc = 0.0f64; let mut best_mse = f64::MAX;
                let mut best_w = vec![-1i8; np];
                for config in 0..total {
                    let w: Vec<i8> = (0..np).map(|bit|
                        if (config >> bit) & 1 == 1 { 1i8 } else { -1i8 }).collect();
                    let mut t = base.clone();
                    t.b.push(DWorker { weights: w.clone(), c_val, reads_a: ra.clone() });
                    let acc = t.accuracy(op); let mse = t.mse(op);
                    if acc > best_acc || (acc == best_acc && mse < best_mse) {
                        best_acc = acc; best_mse = mse; best_w = w;
                    }
                }
                (best_acc, best_mse, c_val, best_w)
            }).collect();
            let mut best = &results[0];
            for r in &results { if r.0 > best.0 || (r.0 == best.0 && r.1 < best.1) { best = r; } }
            net.b.push(DWorker { weights: best.3.clone(), c_val: best.2, reads_a });
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

fn main() {
    println!("=== AB vs BA order + ratio sweep ===\n");
    println!("AB = sparse fires first (input→sparse→dense)");
    println!("BA = dense fires first  (dense→sparse←input)\n");

    let c_step = 0.1;
    let c_max = 5.0;

    let ops: Vec<(&str, fn(usize, usize) -> usize)> = vec![
        ("ADD", op_add), ("MAX", op_max), ("MIN", op_min),
        ("|a-b|", op_sub_abs), ("MUL", op_mul),
    ];

    // =========================================================
    // TEST 1: AB vs BA (4+4, K=all, t=2, all tasks)
    // =========================================================
    println!("--- TEST 1: AB vs BA (4S+4D, K=all, t=2) ---\n");
    println!("  {:>6} {:>8} {:>8} {:>8} {:>8}",
        "task", "AB acc", "AB mse", "BA acc", "BA mse");
    println!("  {}", "=".repeat(40));

    for &(name, op) in &ops {
        let ab = build_order_net(4, 4, 99, 2, true, op, c_step, c_max, 42);
        let ba = build_order_net(4, 4, 99, 2, false, op, c_step, c_max, 42);
        println!("  {:>6} {:>7.0}% {:>8.4} {:>7.0}% {:>8.4}",
            name, ab.accuracy(op)*100.0, ab.mse(op),
            ba.accuracy(op)*100.0, ba.mse(op));
        std::io::stdout().flush().unwrap();
    }

    // =========================================================
    // TEST 2: AB vs BA robustness (5 seeds, K=4, t=2)
    // =========================================================
    println!("\n--- TEST 2: AB vs BA robustness (4+4, K=4, t=2, 5 seeds) ---\n");
    println!("  {:>6} {:>10} {:>10}",
        "task", "AB solved", "BA solved");
    println!("  {}", "=".repeat(28));

    for &(name, op) in &ops {
        print!("  {:>6}", name);
        std::io::stdout().flush().unwrap();
        for &sf in &[true, false] {
            let mut solved = 0;
            for seed in 0..5u64 {
                if build_order_net(4, 4, 4, 2, sf, op, c_step, c_max, seed).accuracy(op) >= 1.0 {
                    solved += 1;
                }
            }
            print!(" {:>9}/5", solved);
            std::io::stdout().flush().unwrap();
        }
        println!();
    }

    // =========================================================
    // TEST 3: Ratio sweep (8 neurons total, K=all, t=2, AB)
    // =========================================================
    println!("\n--- TEST 3: Ratio sweep (8 neurons, K=all, t=2, AB order) ---\n");
    println!("  {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "task", "1S+7D", "2S+6D", "3S+5D", "4S+4D", "5S+3D", "6S+2D", "7S+1D");
    println!("  {}", "=".repeat(62));

    for &(name, op) in &ops {
        print!("  {:>6}", name);
        std::io::stdout().flush().unwrap();
        for &(na, nb) in &[(1,7), (2,6), (3,5), (4,4), (5,3), (6,2), (7,1)] {
            let net = build_order_net(na, nb, 99, 2, true, op, c_step, c_max, 42);
            print!(" {:>7.0}%", net.accuracy(op)*100.0);
            std::io::stdout().flush().unwrap();
        }
        println!();
    }

    // =========================================================
    // TEST 4: Ratio sweep with BA order (same)
    // =========================================================
    println!("\n--- TEST 4: Ratio sweep (8 neurons, K=all, t=2, BA order) ---\n");
    println!("  {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "task", "1S+7D", "2S+6D", "3S+5D", "4S+4D", "5S+3D", "6S+2D", "7S+1D");
    println!("  {}", "=".repeat(62));

    for &(name, op) in &[("ADD", op_add as fn(usize,usize)->usize),
                          ("|a-b|", op_sub_abs), ("MUL", op_mul)] {
        print!("  {:>6}", name);
        std::io::stdout().flush().unwrap();
        for &(na, nb) in &[(1,7), (2,6), (3,5), (4,4), (5,3), (6,2), (7,1)] {
            let net = build_order_net(na, nb, 99, 2, false, op, c_step, c_max, 42);
            print!(" {:>7.0}%", net.accuracy(op)*100.0);
            std::io::stdout().flush().unwrap();
        }
        println!();
    }

    // =========================================================
    // TEST 5: Best ratio at 8+8 scale (K=4, 3 seeds)
    // =========================================================
    println!("\n--- TEST 5: Ratio at scale (12 neurons, K=4, t=2, 3 seeds) ---\n");
    println!("  {:>6} {:>6} {:>10} {:>10}",
        "task", "ratio", "AB solved", "BA solved");
    println!("  {}", "=".repeat(34));

    for &(name, op) in &[("ADD", op_add as fn(usize,usize)->usize),
                          ("|a-b|", op_sub_abs), ("MUL", op_mul)] {
        for &(na, nb) in &[(3,9), (4,8), (6,6), (8,4), (9,3)] {
            print!("  {:>6} {:>2}S+{:<2}D", name, na, nb);
            std::io::stdout().flush().unwrap();
            for &sf in &[true, false] {
                let mut solved = 0;
                for seed in 0..3u64 {
                    if build_order_net(na, nb, 4, 2, sf, op, c_step, c_max, seed).accuracy(op) >= 1.0 {
                        solved += 1;
                    }
                }
                print!(" {:>9}/3", solved);
                std::io::stdout().flush().unwrap();
            }
            println!();
        }
        println!();
    }

    println!("=== DONE ===");
}
