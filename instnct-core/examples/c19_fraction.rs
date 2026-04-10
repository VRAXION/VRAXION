//! Fraction extraction: float weights → numerator/denominator
//!
//! Try denominators 1-30, find which makes all weights nearest to integers.
//! Deploy: integer numerators + scaled C in LUT. Pure integer chip.
//!
//! Run: cargo run --example c19_fraction --release

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

/// Try denominators 1..max_d, find which one makes all weights closest to integers
fn find_best_denominator(weights: &[f32], max_d: u32) -> Vec<(u32, Vec<i32>, f64, i32)> {
    let mut results = Vec::new();

    for d in 1..=max_d {
        let numerators: Vec<i32> = weights.iter().map(|&w| (w * d as f32).round() as i32).collect();
        let err: f64 = weights.iter().zip(numerators.iter())
            .map(|(&w, &n)| {
                let reconstructed = n as f32 / d as f32;
                (w - reconstructed) as f64
            })
            .map(|e| e * e).sum::<f64>() / weights.len() as f64;
        let max_num = numerators.iter().map(|n| n.abs()).max().unwrap_or(0);
        results.push((d, numerators, err, max_num));
    }

    results
}

fn op_add(a: usize, b: usize) -> usize { a + b }
fn op_mul(a: usize, b: usize) -> usize { a * b }
fn op_max(a: usize, b: usize) -> usize { a.max(b) }
fn op_min(a: usize, b: usize) -> usize { a.min(b) }
fn op_sub_abs(a: usize, b: usize) -> usize { if a > b { a - b } else { b - a } }

fn main() {
    println!("=== FRACTION EXTRACTION: weight = numerator / denominator ===\n");

    let nc = 3;

    let tasks: Vec<(&str, fn(usize, usize) -> usize, usize)> = vec![
        ("ADD",   op_add,     3),
        ("MAX",   op_max,     3),
        ("MIN",   op_min,     3),
        ("|a-b|", op_sub_abs, 6),
        ("MUL",   op_mul,     6),
    ];

    for &(name, op, nw) in &tasks {
        println!("========== {} ({} workers) ==========\n", name, nw);

        for seed in 1..=3u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = Net::new(nc, nw, &mut rng, 0.5);
            let (float_acc, steps) = optimize(&mut net, op);
            if float_acc < 1.0 { continue; }

            println!("  Seed {} (float={:.0}%, {} steps)", seed, float_acc * 100.0, steps);

            // Find best denominator for all params
            let denoms = find_best_denominator(&net.params, 30);

            // Show denominator sweep
            println!("    {:>5} {:>8} {:>10} {:>8}", "denom", "max_num", "MSE_err", "int_acc");
            println!("    {}", "-".repeat(35));

            for &(d, ref nums, err, max_num) in &denoms {
                // Test: replace params with numerator/denominator
                let mut test_net = net.clone();
                for i in 0..test_net.params.len() {
                    test_net.params[i] = nums[i] as f32 / d as f32;
                }
                let int_acc = test_net.accuracy(op);

                let bits = if max_num == 0 { 1 } else { (max_num as f64 * 2.0 + 1.0).log2().ceil() as u32 };

                // Only show if interesting
                if int_acc >= 0.80 || d <= 5 || d % 5 == 0 {
                    let marker = if int_acc >= 1.0 { " <-- 100%!" } else { "" };
                    println!("    {:>5} {:>5} ({} bit) {:>10.6} {:>7.0}%{}",
                        d, max_num, bits, err, int_acc * 100.0, marker);
                }
            }

            // Find smallest denominator that gives 100%
            let best_100 = denoms.iter().find(|&&(d, ref nums, _, _)| {
                let mut test = net.clone();
                for i in 0..test.params.len() {
                    test.params[i] = nums[i] as f32 / d as f32;
                }
                test.accuracy(op) >= 1.0
            });

            if let Some(&(d, ref nums, _, max_num)) = best_100 {
                let bits = if max_num == 0 { 1 } else { (max_num as f64 * 2.0 + 1.0).log2().ceil() as u32 };
                println!("\n    BEST: denominator={}, max_numerator=±{}, {} bit per weight", d, max_num, bits);

                // Show per-worker breakdown
                for w in 0..nw {
                    let o = net.offsets[w];
                    let nl = net.local_counts[w];
                    let np = INPUT_DIM + nl + nc + 1 + 1;
                    let float_w = &net.params[o..o+np];
                    let int_w = &nums[o..o+np];
                    println!("    Worker {}: float=[{}]",
                        w, float_w.iter().map(|v| format!("{:+.3}", v)).collect::<Vec<_>>().join(", "));
                    println!("             int/{}=[{}]",
                        d, int_w.iter().map(|v| format!("{:+}", v)).collect::<Vec<_>>().join(", "));
                }
            } else {
                println!("\n    No denominator 1-30 gives 100%");
            }
            println!();
        }
    }

    // =========================================================
    // Summary: all seeds, all tasks
    // =========================================================
    println!("========== SUMMARY (20 seeds) ==========\n");
    println!("{:>8} {:>10} {:>10} {:>10} {:>10}",
        "task", "float_100%", "best_d", "int_100%", "bits");
    println!("{}", "=".repeat(55));

    let seeds20: Vec<u64> = (1..=20).collect();

    for &(name, op, nw) in &tasks {
        let results: Vec<(f64, Option<(u32, i32)>)> = seeds20.iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = Net::new(nc, nw, &mut rng, 0.5);
            let (float_acc, _) = optimize(&mut net, op);
            if float_acc < 1.0 { return (float_acc, None); }

            // Find smallest working denominator
            for d in 1..=30u32 {
                let nums: Vec<i32> = net.params.iter().map(|&w| (w * d as f32).round() as i32).collect();
                let mut test = net.clone();
                for i in 0..test.params.len() { test.params[i] = nums[i] as f32 / d as f32; }
                if test.accuracy(op) >= 1.0 {
                    let max_num = nums.iter().map(|n| n.abs()).max().unwrap_or(0);
                    return (float_acc, Some((d, max_num)));
                }
            }
            (float_acc, None)
        }).collect();

        let float_solved = results.iter().filter(|r| r.0 >= 1.0).count();
        let int_solved = results.iter().filter(|r| r.1.is_some()).count();
        let best_denoms: Vec<u32> = results.iter().filter_map(|r| r.1.map(|x| x.0)).collect();
        let max_nums: Vec<i32> = results.iter().filter_map(|r| r.1.map(|x| x.1)).collect();

        let avg_d = if best_denoms.is_empty() { 0.0 } else { best_denoms.iter().sum::<u32>() as f64 / best_denoms.len() as f64 };
        let avg_bits = if max_nums.is_empty() { 0.0 } else {
            max_nums.iter().map(|&m| if m == 0 { 1.0 } else { (m as f64 * 2.0 + 1.0).log2().ceil() }).sum::<f64>() / max_nums.len() as f64
        };

        println!("{:>8} {:>7}/{:<2} {:>10.1} {:>7}/{:<2} {:>8.1}",
            name, float_solved, 20, avg_d, int_solved, float_solved, avg_bits);
    }

    println!("\n=== DONE ===");
}
