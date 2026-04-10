//! GCD extraction: find common denominator of converged float weights
//!
//! Float train → find base unit (GCD) → divide → integers → test
//! No search, no sweep — just division.
//!
//! Run: cargo run --example c19_gcd_extract --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const DIGITS: usize = 5;
const INPUT_DIM: usize = 8;
const LOCAL_CAP: usize = 3;
const TICKS: usize = 2;

fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.01);
    let rho = rho.max(0.0);
    let l = 6.0 * c;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let scaled = x / c;
    let n = scaled.floor();
    let t = scaled - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sgn * h + rho * h * h)
}

fn thermo_2(a: usize, b: usize) -> [f32; 8] {
    let mut v = [0.0f32; 8];
    for i in 0..a.min(4) { v[i] = 1.0; }
    for i in 0..b.min(4) { v[4 + i] = 1.0; }
    v
}

#[derive(Clone)]
struct Net {
    nc: usize, nw: usize,
    params: Vec<f32>, c_params: Vec<f32>, rho_params: Vec<f32>,
    offsets: Vec<usize>, local_counts: Vec<usize>,
}

impl Net {
    fn new(nc: usize, nw: usize, rng: &mut StdRng, scale: f32) -> Self {
        let mut net = Net {
            nc, nw: 0, params: Vec::new(), c_params: Vec::new(), rho_params: Vec::new(),
            offsets: Vec::new(), local_counts: Vec::new(),
        };
        for i in 0..nw {
            let nl = LOCAL_CAP.min(i);
            let np = INPUT_DIM + nl + nc + 1 + 1;
            net.offsets.push(net.params.len());
            net.local_counts.push(nl);
            net.params.extend((0..np).map(|_| rng.gen_range(-scale..scale)));
            net.c_params.push(1.0);
            net.rho_params.push(4.0);
            net.nw += 1;
        }
        net
    }

    fn forward(&self, a: usize, b: usize) -> f32 {
        let input = thermo_2(a, b);
        let nc = self.nc;
        let nw = self.nw;
        let mut act = vec![0.0f32; nc + nw];
        for _t in 0..TICKS {
            let mut cc = vec![0.0f32; nc];
            for i in 0..nw {
                let o = self.offsets[i];
                let nl = self.local_counts[i];
                let ww = self.params[o + INPUT_DIM + nl + nc];
                let slot = i % nc.max(1);
                if slot < nc { cc[slot] += act[nc + i] * ww; }
            }
            for i in 0..nc { act[i] = cc[i]; }
            let old = act.clone();
            for i in 0..nw {
                let o = self.offsets[i];
                let nl = self.local_counts[i];
                let mut s = self.params[o + INPUT_DIM + nl + nc + 1];
                for j in 0..INPUT_DIM { s += input[j] * self.params[o + j]; }
                let ls = i.saturating_sub(nl);
                for (k, wi) in (ls..i).enumerate() {
                    s += old[nc + wi] * self.params[o + INPUT_DIM + k];
                }
                for k in 0..nc { s += old[k] * self.params[o + INPUT_DIM + nl + k]; }
                act[nc + i] = c19(s, self.c_params[i], self.rho_params[i]);
            }
        }
        act[nc..].iter().sum()
    }

    fn mse(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut l = 0.0f64;
        for a in 0..DIGITS { for b in 0..DIGITS {
            let d = self.forward(a, b) as f64 - op(a, b) as f64;
            l += d * d;
        }}
        l / 25.0
    }

    fn accuracy(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut c = 0;
        for a in 0..DIGITS { for b in 0..DIGITS {
            if (self.forward(a, b).round() as i32) == (op(a, b) as i32) { c += 1; }
        }}
        c as f64 / 25.0
    }

    fn gradient(&mut self, op: fn(usize, usize) -> usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let eps = 1e-3f32;
        let n = self.params.len();
        let mut g = vec![0.0f32; n];
        for i in 0..n {
            let orig = self.params[i];
            self.params[i] = orig + eps; let lp = self.mse(op);
            self.params[i] = orig - eps; let lm = self.mse(op);
            self.params[i] = orig;
            g[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
        }
        let nw = self.c_params.len();
        let mut gc = vec![0.0f32; nw];
        let mut gr = vec![0.0f32; nw];
        for i in 0..nw {
            let orig = self.c_params[i];
            self.c_params[i] = orig + eps; let lp = self.mse(op);
            self.c_params[i] = orig - eps; let lm = self.mse(op);
            self.c_params[i] = orig;
            gc[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
            let orig = self.rho_params[i];
            self.rho_params[i] = orig + eps; let lp = self.mse(op);
            self.rho_params[i] = orig - eps; let lm = self.mse(op);
            self.rho_params[i] = orig;
            gr[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
        }
        (g, gc, gr)
    }
}

fn optimize(net: &mut Net, op: fn(usize, usize) -> usize) -> (f64, usize) {
    let mut lr = 0.01f32;
    let patience = 100;
    let mut stale = 0;
    let mut best_loss = net.mse(op);
    let mut step = 0;
    loop {
        let acc = net.accuracy(op);
        if acc >= 1.0 { return (acc, step); }
        if stale >= patience { return (acc, step); }
        let (g, gc, gr) = net.gradient(op);
        let gn: f32 = g.iter().chain(gc.iter()).chain(gr.iter()).map(|x| x*x).sum::<f32>().sqrt();
        if gn < 1e-8 { return (acc, step); }
        let old_p = net.params.clone();
        let old_c = net.c_params.clone();
        let old_r = net.rho_params.clone();
        let ol = net.mse(op);
        let mut improved = false;
        for att in 0..5 {
            for i in 0..net.params.len() { net.params[i] = old_p[i] - lr * g[i] / gn; }
            for i in 0..net.c_params.len() { net.c_params[i] = (old_c[i] - lr * gc[i] / gn).max(0.01); }
            for i in 0..net.rho_params.len() { net.rho_params[i] = (old_r[i] - lr * gr[i] / gn).max(0.0); }
            let nl = net.mse(op);
            if nl < ol { lr *= 1.1; if nl < best_loss - 1e-8 { best_loss = nl; stale = 0; improved = true; } break; }
            else { lr *= 0.5; if att == 4 { net.params = old_p.clone(); net.c_params = old_c.clone(); net.rho_params = old_r.clone(); } }
        }
        if !improved { stale += 1; }
        step += 1;
    }
}

/// Find the GCD-like base unit of a set of floats
/// Try candidate base units and find one that makes all values near-integer
fn find_base_unit(weights: &[f32]) -> (f32, Vec<i32>, f64) {
    let abs_vals: Vec<f32> = weights.iter().map(|w| w.abs()).filter(|&w| w > 0.01).collect();
    if abs_vals.is_empty() { return (1.0, weights.iter().map(|_| 0).collect(), 0.0); }

    let mut best_unit = 1.0f32;
    let mut best_ints: Vec<i32> = Vec::new();
    let mut best_err = f64::MAX;

    // Method 1: try each abs weight as potential base unit
    for &candidate in &abs_vals {
        let ints: Vec<i32> = weights.iter().map(|&w| (w / candidate).round() as i32).collect();
        let err: f64 = weights.iter().zip(ints.iter())
            .map(|(&w, &i)| (w - i as f32 * candidate) as f64)
            .map(|e| e * e).sum::<f64>() / weights.len() as f64;
        if err < best_err {
            best_err = err;
            best_unit = candidate;
            best_ints = ints;
        }
    }

    // Method 2: try fractions of the min value
    let min_abs = abs_vals.iter().fold(f32::MAX, |a, &b| a.min(b));
    for div in 1..=5 {
        let candidate = min_abs / div as f32;
        if candidate < 0.01 { continue; }
        let ints: Vec<i32> = weights.iter().map(|&w| (w / candidate).round() as i32).collect();
        let err: f64 = weights.iter().zip(ints.iter())
            .map(|(&w, &i)| (w - i as f32 * candidate) as f64)
            .map(|e| e * e).sum::<f64>() / weights.len() as f64;
        if err < best_err {
            best_err = err;
            best_unit = candidate;
            best_ints = ints;
        }
    }

    // Method 3: try fine grid
    for step in 1..=200 {
        let candidate = step as f32 * 0.01;
        let ints: Vec<i32> = weights.iter().map(|&w| (w / candidate).round() as i32).collect();
        let err: f64 = weights.iter().zip(ints.iter())
            .map(|(&w, &i)| (w - i as f32 * candidate) as f64)
            .map(|e| e * e).sum::<f64>() / weights.len() as f64;
        if err < best_err {
            best_err = err;
            best_unit = candidate;
            best_ints = ints;
        }
    }

    (best_unit, best_ints, best_err)
}

fn op_add(a: usize, b: usize) -> usize { a + b }
fn op_mul(a: usize, b: usize) -> usize { a * b }
fn op_max(a: usize, b: usize) -> usize { a.max(b) }
fn op_min(a: usize, b: usize) -> usize { a.min(b) }
fn op_sub_abs(a: usize, b: usize) -> usize { if a > b { a - b } else { b - a } }

fn main() {
    println!("=== GCD EXTRACTION: float → common denominator → integer ===\n");

    let nc = 3;
    let n_seeds = 20;
    let seeds: Vec<u64> = (1..=n_seeds as u64).collect();

    let tasks: Vec<(&str, fn(usize, usize) -> usize, usize)> = vec![
        ("ADD",   op_add,     3),
        ("MAX",   op_max,     3),
        ("MIN",   op_min,     3),
        ("|a-b|", op_sub_abs, 6),
        ("MUL",   op_mul,     6),
    ];

    println!("{:>8} {:>6} {:>10} {:>10} {:>10} {:>10} {:>8}",
        "task", "seed", "float_acc", "int_acc", "base_unit", "max_int", "int_range");
    println!("{}", "=".repeat(75));

    for &(name, op, nw) in &tasks {
        let mut n_float_solved = 0;
        let mut n_int_solved = 0;
        let mut all_max_ints: Vec<i32> = Vec::new();

        for &seed in &seeds {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = Net::new(nc, nw, &mut rng, 0.5);
            let (float_acc, _) = optimize(&mut net, op);

            if float_acc < 1.0 { continue; } // skip unsolved
            n_float_solved += 1;

            // Find GCD base unit for ALL params at once
            let (base_unit, int_weights, _err) = find_base_unit(&net.params);

            // Apply integer weights: replace params with int * base_unit
            let mut int_net = net.clone();
            for i in 0..int_net.params.len() {
                int_net.params[i] = int_weights[i] as f32 * base_unit;
            }
            // C and rho stay float (they go into LUT anyway)
            let int_acc = int_net.accuracy(op);

            let max_int = int_weights.iter().map(|w| w.abs()).max().unwrap_or(0);
            all_max_ints.push(max_int);

            if int_acc >= 1.0 { n_int_solved += 1; }

            let bits_needed = (max_int as f64 * 2.0 + 1.0).log2().ceil() as u32;

            println!("{:>8} {:>6} {:>9.0}% {:>9.0}% {:>10.4} {:>10} {:>5} bit",
                name, seed, float_acc * 100.0, int_acc * 100.0,
                base_unit, max_int, bits_needed);
        }

        let avg_max = if all_max_ints.is_empty() { 0.0 }
            else { all_max_ints.iter().sum::<i32>() as f64 / all_max_ints.len() as f64 };

        println!("{:>8} TOTAL: {}/{} float → {}/{} integer survived  avg_max_int={:.1}\n",
            name, n_float_solved, n_seeds, n_int_solved, n_float_solved, avg_max);
    }

    println!("=== DONE ===");
}
