//! Greedy freeze-per-layer: how many neurons per freeze cycle?
//!
//! Sweep N=1,2,3,4 neurons per group:
//!   - Train N neurons (float gradient)
//!   - Fraction extract → i8 weights + LUT
//!   - Freeze as integer
//!   - Add next group, repeat
//!
//! Run: cargo run --example greedy_freeze --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

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

/// A neuron that's been frozen as integer weights + LUT
#[derive(Clone)]
struct FrozenNeuron {
    int_weights: Vec<i32>,  // integer numerators
    denom: u32,
    lut: Vec<i32>,          // lut[sum - min_sum] = output (in denom scale)
    min_sum: i32,
    n_local: usize,
}

/// Network with frozen + active neurons
#[derive(Clone)]
struct GreedyNet {
    nc: usize,
    frozen: Vec<FrozenNeuron>,
    // Active (trainable) neurons
    active_params: Vec<f32>,
    active_c: Vec<f32>,
    active_rho: Vec<f32>,
    active_offsets: Vec<usize>,
    active_local_counts: Vec<usize>,
    n_active: usize,
}

impl GreedyNet {
    fn new(nc: usize) -> Self {
        GreedyNet {
            nc, frozen: Vec::new(),
            active_params: Vec::new(), active_c: Vec::new(), active_rho: Vec::new(),
            active_offsets: Vec::new(), active_local_counts: Vec::new(), n_active: 0,
        }
    }

    fn total_workers(&self) -> usize { self.frozen.len() + self.n_active }

    fn add_active_neurons(&mut self, count: usize, rng: &mut StdRng, scale: f32) {
        for _ in 0..count {
            let i = self.total_workers();
            let nl = LOCAL_CAP.min(i);
            let np = INPUT_DIM + nl + self.nc + 1 + 1;
            self.active_offsets.push(self.active_params.len());
            self.active_local_counts.push(nl);
            self.active_params.extend((0..np).map(|_| rng.gen_range(-scale..scale)));
            self.active_c.push(1.0);
            self.active_rho.push(4.0);
            self.n_active += 1;
        }
    }

    fn forward(&self, a: usize, b: usize) -> f32 {
        let input = thermo_2(a, b);
        let nc = self.nc;
        let n_frozen = self.frozen.len();
        let n_active = self.n_active;
        let total = nc + n_frozen + n_active;
        let mut act = vec![0.0f32; total];

        for _t in 0..TICKS {
            // Phase 1: all workers write to connectome
            let mut cc = vec![0.0f32; nc];

            // Frozen workers write (integer)
            for i in 0..n_frozen {
                let f = &self.frozen[i];
                let nl = f.n_local;
                let ww = f.int_weights[INPUT_DIM + nl + nc] as f32 / f.denom as f32;
                let slot = i % nc.max(1);
                if slot < nc { cc[slot] += act[nc + i] * ww; }
            }
            // Active workers write (float)
            for i in 0..n_active {
                let gi = n_frozen + i; // global index
                let o = self.active_offsets[i];
                let nl = self.active_local_counts[i];
                let ww = self.active_params[o + INPUT_DIM + nl + nc];
                let slot = gi % nc.max(1);
                if slot < nc { cc[slot] += act[nc + gi] * ww; }
            }

            for k in 0..nc { act[k] = cc[k]; }
            let old = act.clone();

            // Phase 2: frozen workers compute (LUT)
            for i in 0..n_frozen {
                let f = &self.frozen[i];
                let nl = f.n_local;
                // Integer weighted sum
                let mut s: i32 = f.int_weights[INPUT_DIM + nl + nc + 1];
                for j in 0..INPUT_DIM {
                    if input[j] > 0.5 { s += f.int_weights[j]; }
                }
                let ls = i.saturating_sub(nl);
                for (k, wi) in (ls..i).enumerate() {
                    // Convert neighbor activation to int scale
                    let neighbor_int = (old[nc + wi] * f.denom as f32).round() as i32;
                    s += neighbor_int * f.int_weights[INPUT_DIM + k] / f.denom as i32;
                }
                for k in 0..nc {
                    let conn_int = (old[k] * f.denom as f32).round() as i32;
                    s += conn_int * f.int_weights[INPUT_DIM + nl + k] / f.denom as i32;
                }
                // LUT lookup
                let idx = (s - f.min_sum).max(0).min(f.lut.len() as i32 - 1) as usize;
                act[nc + i] = f.lut[idx] as f32 / f.denom as f32;
            }

            // Phase 3: active workers compute (float C19)
            for i in 0..n_active {
                let gi = n_frozen + i;
                let o = self.active_offsets[i];
                let nl = self.active_local_counts[i];
                let mut s = self.active_params[o + INPUT_DIM + nl + nc + 1];
                for j in 0..INPUT_DIM { s += input[j] * self.active_params[o + j]; }
                let ls = gi.saturating_sub(nl);
                for (k, wi) in (ls..gi).enumerate() {
                    s += old[nc + wi] * self.active_params[o + INPUT_DIM + k];
                }
                for k in 0..nc {
                    s += old[k] * self.active_params[o + INPUT_DIM + nl + k];
                }
                act[nc + gi] = c19(s, self.active_c[i], self.active_rho[i]);
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

    fn gradient_active(&mut self, op: fn(usize, usize) -> usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let eps = 1e-3f32;
        let n = self.active_params.len();
        let mut g = vec![0.0f32; n];
        for i in 0..n {
            let orig = self.active_params[i];
            self.active_params[i] = orig + eps; let lp = self.mse(op);
            self.active_params[i] = orig - eps; let lm = self.mse(op);
            self.active_params[i] = orig;
            g[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
        }
        let nw = self.active_c.len();
        let mut gc = vec![0.0f32; nw];
        let mut gr = vec![0.0f32; nw];
        for i in 0..nw {
            let orig = self.active_c[i];
            self.active_c[i] = orig + eps; let lp = self.mse(op);
            self.active_c[i] = orig - eps; let lm = self.mse(op);
            self.active_c[i] = orig;
            gc[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
            let orig = self.active_rho[i];
            self.active_rho[i] = orig + eps; let lp = self.mse(op);
            self.active_rho[i] = orig - eps; let lm = self.mse(op);
            self.active_rho[i] = orig;
            gr[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
        }
        (g, gc, gr)
    }

    /// Freeze all active neurons: extract fraction → LUT → frozen
    fn freeze_active(&mut self) {
        // Find best denominator for active params
        let mut best_denom = 1u32;
        let mut best_err = f64::MAX;
        for d in 1..=30u32 {
            let err: f64 = self.active_params.iter()
                .map(|&w| {
                    let q = (w * d as f32).round() as f32 / d as f32;
                    (w - q) as f64
                })
                .map(|e| e * e).sum::<f64>() / self.active_params.len() as f64;
            if err < best_err { best_err = err; best_denom = d; }
        }

        for i in 0..self.n_active {
            let o = self.active_offsets[i];
            let nl = self.active_local_counts[i];
            let np = INPUT_DIM + nl + self.nc + 1 + 1;

            let int_weights: Vec<i32> = self.active_params[o..o+np].iter()
                .map(|&w| (w * best_denom as f32).round() as i32).collect();

            // Build LUT
            let max_abs_sum: i32 = int_weights.iter().map(|w| w.abs()).sum();
            let min_sum = -max_abs_sum;
            let c = self.active_c[i];
            let rho = self.active_rho[i];
            let lut: Vec<i32> = (min_sum..=max_abs_sum).map(|s| {
                let float_in = s as f32 / best_denom as f32;
                let float_out = c19(float_in, c, rho);
                (float_out * best_denom as f32).round() as i32
            }).collect();

            self.frozen.push(FrozenNeuron {
                int_weights, denom: best_denom, lut, min_sum, n_local: nl,
            });
        }

        self.active_params.clear();
        self.active_c.clear();
        self.active_rho.clear();
        self.active_offsets.clear();
        self.active_local_counts.clear();
        self.n_active = 0;
    }
}

fn optimize_active(net: &mut GreedyNet, op: fn(usize, usize) -> usize) -> (f64, usize) {
    let mut lr = 0.01f32;
    let patience = 100;
    let mut stale = 0;
    let mut best_loss = net.mse(op);
    let mut step = 0;
    loop {
        let acc = net.accuracy(op);
        if acc >= 1.0 { return (acc, step); }
        if stale >= patience { return (acc, step); }
        let (g, gc, gr) = net.gradient_active(op);
        let gn: f32 = g.iter().chain(gc.iter()).chain(gr.iter()).map(|x| x*x).sum::<f32>().sqrt();
        if gn < 1e-8 { return (acc, step); }
        let old_p = net.active_params.clone();
        let old_c = net.active_c.clone();
        let old_r = net.active_rho.clone();
        let ol = net.mse(op);
        let mut improved = false;
        for att in 0..5 {
            for i in 0..net.active_params.len() { net.active_params[i] = old_p[i] - lr * g[i] / gn; }
            for i in 0..net.active_c.len() { net.active_c[i] = (old_c[i] - lr * gc[i] / gn).max(0.01); }
            for i in 0..net.active_rho.len() { net.active_rho[i] = (old_r[i] - lr * gr[i] / gn).max(0.0); }
            let nl = net.mse(op);
            if nl < ol { lr *= 1.1; if nl < best_loss - 1e-8 { best_loss = nl; stale = 0; improved = true; } break; }
            else { lr *= 0.5; if att == 4 { net.active_params = old_p.clone(); net.active_c = old_c.clone(); net.active_rho = old_r.clone(); } }
        }
        if !improved { stale += 1; }
        step += 1;
    }
}

fn op_add(a: usize, b: usize) -> usize { a + b }
fn op_mul(a: usize, b: usize) -> usize { a * b }
fn op_max(a: usize, b: usize) -> usize { a.max(b) }
fn op_min(a: usize, b: usize) -> usize { a.min(b) }
fn op_sub_abs(a: usize, b: usize) -> usize { if a > b { a - b } else { b - a } }

fn main() {
    println!("=== GREEDY FREEZE-PER-LAYER: optimal N per cycle ===\n");

    let nc = 3;
    let max_total = 12;
    let n_seeds = 10;
    let seeds: Vec<u64> = (1..=n_seeds as u64).collect();

    let tasks: Vec<(&str, fn(usize, usize) -> usize)> = vec![
        ("ADD",   op_add),
        ("MAX",   op_max),
        ("MIN",   op_min),
        ("|a-b|", op_sub_abs),
        ("MUL",   op_mul),
    ];

    let group_sizes = [1, 2, 3, 4];

    println!("{:>8} {:>4}  {:>10} {:>10} {:>10} {:>8}",
        "task", "N", "solved", "avg_total", "avg_steps", "avg_ms");
    println!("{}", "=".repeat(60));

    for &(name, op) in &tasks {
        // Also test all-at-once baseline
        for &gs in &group_sizes {
            let results: Vec<(bool, usize, usize, f64)> = seeds.par_iter().map(|&seed| {
                let t0 = Instant::now();
                let mut rng = StdRng::seed_from_u64(seed);
                let mut net = GreedyNet::new(nc);
                let mut total_steps = 0;

                let mut round = 0;
                loop {
                    if net.total_workers() >= max_total { break; }

                    // Add N neurons
                    net.add_active_neurons(gs, &mut rng, 0.5);

                    // Train active neurons
                    let (acc, steps) = optimize_active(&mut net, op);
                    total_steps += steps;

                    round += 1;

                    if acc >= 1.0 {
                        // Freeze and done
                        net.freeze_active();
                        // Verify frozen accuracy
                        let frozen_acc = net.accuracy(op);
                        if frozen_acc >= 1.0 {
                            return (true, net.total_workers(), total_steps, t0.elapsed().as_secs_f64() * 1000.0);
                        }
                        // If freeze broke it, continue adding
                    } else {
                        // Freeze anyway and try adding more
                        net.freeze_active();
                    }

                    // Check if we're stuck
                    if round > 2 && net.accuracy(op) < 0.2 { break; }
                }
                (false, net.total_workers(), total_steps, t0.elapsed().as_secs_f64() * 1000.0)
            }).collect();

            let solved = results.iter().filter(|r| r.0).count();
            let avg_neurons: f64 = results.iter().filter(|r| r.0).map(|r| r.1 as f64).sum::<f64>()
                / solved.max(1) as f64;
            let avg_steps: f64 = results.iter().map(|r| r.2 as f64).sum::<f64>() / n_seeds as f64;
            let avg_ms: f64 = results.iter().map(|r| r.3).sum::<f64>() / n_seeds as f64;

            println!("{:>8} {:>4}  {:>7}/{:<2} {:>10.1} {:>10.0} {:>7.0}ms",
                name, gs, solved, n_seeds, avg_neurons, avg_steps, avg_ms);
        }

        // All-at-once baseline (no freeze, train all at once)
        for &total_nw in &[3, 6] {
            let results: Vec<(bool, usize, f64)> = seeds.par_iter().map(|&seed| {
                let t0 = Instant::now();
                let mut rng = StdRng::seed_from_u64(seed);
                let mut net = GreedyNet::new(nc);
                net.add_active_neurons(total_nw, &mut rng, 0.5);
                let (acc, steps) = optimize_active(&mut net, op);
                (acc >= 1.0, steps, t0.elapsed().as_secs_f64() * 1000.0)
            }).collect();

            let solved = results.iter().filter(|r| r.0).count();
            let avg_steps: f64 = results.iter().map(|r| r.1 as f64).sum::<f64>() / n_seeds as f64;
            let avg_ms: f64 = results.iter().map(|r| r.2).sum::<f64>() / n_seeds as f64;

            println!("{:>8} all{} {:>7}/{:<2} {:>10} {:>10.0} {:>7.0}ms",
                name, total_nw, solved, n_seeds, total_nw, avg_steps, avg_ms);
        }

        println!();
    }

    println!("=== DONE ===");
}
