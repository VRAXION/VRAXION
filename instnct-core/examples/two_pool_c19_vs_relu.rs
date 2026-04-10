//! Two-Neighborhood C19 vs ReLU — Connectome Communication Test
//!
//! Two isolated neighborhoods (A sees input_a, B sees input_b).
//! Cross-cluster communication ONLY through passive connectome relay.
//! Does C19's periodic wave help more than ReLU in this bottleneck scenario?
//!
//! Run: cargo run --example two_pool_c19_vs_relu --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const DIGITS: usize = 5;
const INPUT_HALF: usize = 4; // thermo bits per digit
const INPUT_FULL: usize = 8; // both digits (single-pool baseline)
const LOCAL_CAP: usize = 3;
const RHO_INIT: f32 = 4.0;

fn relu(x: f32) -> f32 { x.max(0.0) }

fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.1);
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

fn thermo_a(a: usize) -> [f32; 4] {
    let mut v = [0.0f32; 4];
    for i in 0..a.min(4) { v[i] = 1.0; }
    v
}

fn thermo_b(b: usize) -> [f32; 4] {
    let mut v = [0.0f32; 4];
    for i in 0..b.min(4) { v[i] = 1.0; }
    v
}

fn thermo_full(a: usize, b: usize) -> [f32; 8] {
    let mut v = [0.0f32; 8];
    for i in 0..a.min(4) { v[i] = 1.0; }
    for i in 0..b.min(4) { v[4 + i] = 1.0; }
    v
}

// ============================================================
// Two-Pool Network
// ============================================================
#[derive(Clone)]
struct TwoPoolNet {
    nc: usize,
    nw_a: usize,
    nw_b: usize,
    ticks: usize,
    params: Vec<f32>,
    c_params: Vec<f32>,
    rho_params: Vec<f32>,
    offsets_a: Vec<usize>,
    local_counts_a: Vec<usize>,
    offsets_b: Vec<usize>,
    local_counts_b: Vec<usize>,
    use_c19: bool,
}

impl TwoPoolNet {
    fn new(nc: usize, nw_a: usize, nw_b: usize, ticks: usize,
           use_c19: bool, rng: &mut StdRng, scale: f32) -> Self {
        let mut net = TwoPoolNet {
            nc, nw_a, nw_b, ticks,
            params: Vec::new(), c_params: Vec::new(), rho_params: Vec::new(),
            offsets_a: Vec::new(), local_counts_a: Vec::new(),
            offsets_b: Vec::new(), local_counts_b: Vec::new(),
            use_c19,
        };
        // A workers
        for i in 0..nw_a {
            let nl = LOCAL_CAP.min(i);
            let np = INPUT_HALF + nl + nc + 1 + 1;
            net.offsets_a.push(net.params.len());
            net.local_counts_a.push(nl);
            net.params.extend((0..np).map(|_| rng.gen_range(-scale..scale)));
            net.c_params.push(1.0);
            net.rho_params.push(RHO_INIT);
        }
        // B workers
        for j in 0..nw_b {
            let nl = LOCAL_CAP.min(j);
            let np = INPUT_HALF + nl + nc + 1 + 1;
            net.offsets_b.push(net.params.len());
            net.local_counts_b.push(nl);
            net.params.extend((0..np).map(|_| rng.gen_range(-scale..scale)));
            net.c_params.push(1.0);
            net.rho_params.push(RHO_INIT);
        }
        net
    }

    fn forward(&self, a: usize, b: usize) -> f32 {
        let inp_a = thermo_a(a);
        let inp_b = thermo_b(b);
        let nc = self.nc;
        let nw_a = self.nw_a;
        let nw_b = self.nw_b;
        let total = nc + nw_a + nw_b;
        let mut act = vec![0.0f32; total];

        for _t in 0..self.ticks {
            // Phase 1: workers write to connectome
            let mut cc = vec![0.0f32; nc];
            for i in 0..nw_a {
                let o = self.offsets_a[i];
                let nl = self.local_counts_a[i];
                let ww = self.params[o + INPUT_HALF + nl + nc]; // write weight
                let slot = i % nc.max(1);
                if slot < nc { cc[slot] += act[nc + i] * ww; }
            }
            for j in 0..nw_b {
                let o = self.offsets_b[j];
                let nl = self.local_counts_b[j];
                let ww = self.params[o + INPUT_HALF + nl + nc];
                let slot = j % nc.max(1);
                if slot < nc { cc[slot] += act[nc + nw_a + j] * ww; }
            }
            // Passive relay
            for k in 0..nc { act[k] = cc[k]; }

            let old = act.clone();

            // Phase 2: A workers compute
            for i in 0..nw_a {
                let o = self.offsets_a[i];
                let nl = self.local_counts_a[i];
                let mut s = self.params[o + INPUT_HALF + nl + nc + 1]; // bias
                // Input a only
                for j in 0..INPUT_HALF { s += inp_a[j] * self.params[o + j]; }
                // Local within A
                let ls = i.saturating_sub(nl);
                for (k, wi) in (ls..i).enumerate() {
                    s += old[nc + wi] * self.params[o + INPUT_HALF + k];
                }
                // Read connectome
                for k in 0..nc {
                    s += old[k] * self.params[o + INPUT_HALF + nl + k];
                }
                let ci = i; // index into c_params
                act[nc + i] = if self.use_c19 {
                    c19(s, self.c_params[ci], self.rho_params[ci])
                } else {
                    relu(s)
                };
            }

            // Phase 3: B workers compute
            for j in 0..nw_b {
                let o = self.offsets_b[j];
                let nl = self.local_counts_b[j];
                let mut s = self.params[o + INPUT_HALF + nl + nc + 1]; // bias
                // Input b only
                for jj in 0..INPUT_HALF { s += inp_b[jj] * self.params[o + jj]; }
                // Local within B
                let ls = j.saturating_sub(nl);
                for (k, wj) in (ls..j).enumerate() {
                    s += old[nc + nw_a + wj] * self.params[o + INPUT_HALF + k];
                }
                // Read connectome
                for k in 0..nc {
                    s += old[k] * self.params[o + INPUT_HALF + nl + k];
                }
                let ci = nw_a + j; // index into c_params
                act[nc + nw_a + j] = if self.use_c19 {
                    c19(s, self.c_params[ci], self.rho_params[ci])
                } else {
                    relu(s)
                };
            }
        }
        act[nc..].iter().sum()
    }

    fn mse_loss(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut l = 0.0f64;
        for a in 0..DIGITS { for b in 0..DIGITS {
            let d = self.forward(a, b) as f64 - op(a, b) as f64;
            l += d * d;
        }}
        l / 25.0
    }

    fn native_accuracy(&self, op: fn(usize, usize) -> usize) -> f64 {
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
            self.params[i] = orig + eps; let lp = self.mse_loss(op);
            self.params[i] = orig - eps; let lm = self.mse_loss(op);
            self.params[i] = orig;
            g[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
        }
        let nw = self.c_params.len();
        let mut gc = vec![0.0f32; nw];
        let mut gr = vec![0.0f32; nw];
        if self.use_c19 {
            for i in 0..nw {
                let orig = self.c_params[i];
                self.c_params[i] = orig + eps; let lp = self.mse_loss(op);
                self.c_params[i] = orig - eps; let lm = self.mse_loss(op);
                self.c_params[i] = orig;
                gc[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
            }
            for i in 0..nw {
                let orig = self.rho_params[i];
                self.rho_params[i] = orig + eps; let lp = self.mse_loss(op);
                self.rho_params[i] = orig - eps; let lm = self.mse_loss(op);
                self.rho_params[i] = orig;
                gr[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
            }
        }
        (g, gc, gr)
    }
}

// ============================================================
// Single-Pool baseline (same as c19_rho_learnable)
// ============================================================
#[derive(Clone)]
struct SinglePoolNet {
    nc: usize,
    nw: usize,
    ticks: usize,
    params: Vec<f32>,
    c_params: Vec<f32>,
    rho_params: Vec<f32>,
    offsets: Vec<usize>,
    local_counts: Vec<usize>,
    use_c19: bool,
}

impl SinglePoolNet {
    fn new(nc: usize, nw: usize, ticks: usize,
           use_c19: bool, rng: &mut StdRng, scale: f32) -> Self {
        let mut net = SinglePoolNet {
            nc, nw: 0, ticks,
            params: Vec::new(), c_params: Vec::new(), rho_params: Vec::new(),
            offsets: Vec::new(), local_counts: Vec::new(), use_c19,
        };
        for i in 0..nw {
            let nl = LOCAL_CAP.min(i);
            let np = INPUT_FULL + nl + nc + 1 + 1;
            net.offsets.push(net.params.len());
            net.local_counts.push(nl);
            net.params.extend((0..np).map(|_| rng.gen_range(-scale..scale)));
            net.c_params.push(1.0);
            net.rho_params.push(RHO_INIT);
            net.nw += 1;
        }
        net
    }

    fn forward(&self, a: usize, b: usize) -> f32 {
        let input = thermo_full(a, b);
        let nc = self.nc;
        let nw = self.nw;
        let mut act = vec![0.0f32; nc + nw];
        for _t in 0..self.ticks {
            let mut cc = vec![0.0f32; nc];
            for i in 0..nw {
                let o = self.offsets[i];
                let nl = self.local_counts[i];
                let ww = self.params[o + INPUT_FULL + nl + nc];
                let wi = i % nc.max(1);
                if wi < nc { cc[wi] += act[nc + i] * ww; }
            }
            for i in 0..nc { act[i] = cc[i]; }
            let old = act.clone();
            for i in 0..nw {
                let o = self.offsets[i];
                let nl = self.local_counts[i];
                let mut s = self.params[o + INPUT_FULL + nl + nc + 1];
                for j in 0..INPUT_FULL { s += input[j] * self.params[o + j]; }
                let ls = i.saturating_sub(nl);
                for (k, wi) in (ls..i).enumerate() {
                    s += old[nc + wi] * self.params[o + INPUT_FULL + k];
                }
                for k in 0..nc { s += old[k] * self.params[o + INPUT_FULL + nl + k]; }
                act[nc + i] = if self.use_c19 {
                    c19(s, self.c_params[i], self.rho_params[i])
                } else {
                    relu(s)
                };
            }
        }
        act[nc..].iter().sum()
    }

    fn mse_loss(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut l = 0.0f64;
        for a in 0..DIGITS { for b in 0..DIGITS {
            let d = self.forward(a, b) as f64 - op(a, b) as f64;
            l += d * d;
        }}
        l / 25.0
    }

    fn native_accuracy(&self, op: fn(usize, usize) -> usize) -> f64 {
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
            self.params[i] = orig + eps; let lp = self.mse_loss(op);
            self.params[i] = orig - eps; let lm = self.mse_loss(op);
            self.params[i] = orig;
            g[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
        }
        let nw = self.c_params.len();
        let mut gc = vec![0.0f32; nw];
        let mut gr = vec![0.0f32; nw];
        if self.use_c19 {
            for i in 0..nw {
                let orig = self.c_params[i];
                self.c_params[i] = orig + eps; let lp = self.mse_loss(op);
                self.c_params[i] = orig - eps; let lm = self.mse_loss(op);
                self.c_params[i] = orig;
                gc[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
            }
            for i in 0..nw {
                let orig = self.rho_params[i];
                self.rho_params[i] = orig + eps; let lp = self.mse_loss(op);
                self.rho_params[i] = orig - eps; let lm = self.mse_loss(op);
                self.rho_params[i] = orig;
                gr[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
            }
        }
        (g, gc, gr)
    }
}

// ============================================================
// Generic optimizer (works with any type that has the right methods)
// ============================================================

trait Trainable: Clone {
    fn mse_loss(&self, op: fn(usize, usize) -> usize) -> f64;
    fn native_accuracy(&self, op: fn(usize, usize) -> usize) -> f64;
    fn gradient(&mut self, op: fn(usize, usize) -> usize) -> (Vec<f32>, Vec<f32>, Vec<f32>);
    fn params_mut(&mut self) -> &mut Vec<f32>;
    fn c_params_mut(&mut self) -> &mut Vec<f32>;
    fn rho_params_mut(&mut self) -> &mut Vec<f32>;
    fn params(&self) -> &Vec<f32>;
    fn c_params(&self) -> &Vec<f32>;
    fn rho_params(&self) -> &Vec<f32>;
}

impl Trainable for TwoPoolNet {
    fn mse_loss(&self, op: fn(usize, usize) -> usize) -> f64 { self.mse_loss(op) }
    fn native_accuracy(&self, op: fn(usize, usize) -> usize) -> f64 { self.native_accuracy(op) }
    fn gradient(&mut self, op: fn(usize, usize) -> usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) { self.gradient(op) }
    fn params_mut(&mut self) -> &mut Vec<f32> { &mut self.params }
    fn c_params_mut(&mut self) -> &mut Vec<f32> { &mut self.c_params }
    fn rho_params_mut(&mut self) -> &mut Vec<f32> { &mut self.rho_params }
    fn params(&self) -> &Vec<f32> { &self.params }
    fn c_params(&self) -> &Vec<f32> { &self.c_params }
    fn rho_params(&self) -> &Vec<f32> { &self.rho_params }
}

impl Trainable for SinglePoolNet {
    fn mse_loss(&self, op: fn(usize, usize) -> usize) -> f64 { self.mse_loss(op) }
    fn native_accuracy(&self, op: fn(usize, usize) -> usize) -> f64 { self.native_accuracy(op) }
    fn gradient(&mut self, op: fn(usize, usize) -> usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) { self.gradient(op) }
    fn params_mut(&mut self) -> &mut Vec<f32> { &mut self.params }
    fn c_params_mut(&mut self) -> &mut Vec<f32> { &mut self.c_params }
    fn rho_params_mut(&mut self) -> &mut Vec<f32> { &mut self.rho_params }
    fn params(&self) -> &Vec<f32> { &self.params }
    fn c_params(&self) -> &Vec<f32> { &self.c_params }
    fn rho_params(&self) -> &Vec<f32> { &self.rho_params }
}

fn optimize<T: Trainable>(net: &mut T, op: fn(usize, usize) -> usize) -> (f64, usize) {
    let mut lr = 0.01f32;
    let patience = 100;
    let mut stale = 0;
    let mut best_loss = net.mse_loss(op);
    let mut step = 0;
    loop {
        let acc = net.native_accuracy(op);
        if acc >= 1.0 { return (acc, step); }
        if stale >= patience { return (acc, step); }

        let (g, gc, gr) = net.gradient(op);
        let gn: f32 = g.iter().chain(gc.iter()).chain(gr.iter())
            .map(|x| x * x).sum::<f32>().sqrt();
        if gn < 1e-8 { return (acc, step); }

        let old_p = net.params().clone();
        let old_c = net.c_params().clone();
        let old_r = net.rho_params().clone();
        let ol = net.mse_loss(op);

        let mut improved = false;
        for att in 0..5 {
            for i in 0..old_p.len() { net.params_mut()[i] = old_p[i] - lr * g[i] / gn; }
            for i in 0..old_c.len() {
                net.c_params_mut()[i] = (old_c[i] - lr * gc[i] / gn).max(0.1);
            }
            for i in 0..old_r.len() {
                net.rho_params_mut()[i] = (old_r[i] - lr * gr[i] / gn).max(0.0);
            }
            let nl = net.mse_loss(op);
            if nl < ol {
                lr *= 1.1;
                if nl < best_loss - 1e-8 { best_loss = nl; stale = 0; improved = true; }
                break;
            } else {
                lr *= 0.5;
                if att == 4 {
                    net.params_mut().copy_from_slice(&old_p);
                    net.c_params_mut().copy_from_slice(&old_c);
                    net.rho_params_mut().copy_from_slice(&old_r);
                }
            }
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
    println!("=== TWO-NEIGHBORHOOD C19 vs ReLU — CONNECTOME COMMUNICATION ===\n");

    let nc = 3;
    let n_seeds = 50;
    let seeds: Vec<u64> = (1..=n_seeds as u64).collect();

    let tasks: Vec<(&str, fn(usize, usize) -> usize, usize)> = vec![
        ("ADD",   op_add,     3),
        ("MAX",   op_max,     3),
        ("MIN",   op_min,     3),
        ("|a-b|", op_sub_abs, 6),
        ("MUL",   op_mul,     6),
    ];

    // =========================================================
    // PART 1: Tick sweep — how many ticks for cross-cluster communication?
    // =========================================================
    println!("--- PART 1: Tick sweep (nw=3/side, nc=3, 50 seeds) ---\n");
    println!("{:>8} {:>5}  {:>10} {:>8}  {:>10} {:>8}",
        "task", "ticks", "ReLU_solv", "steps", "C19_solv", "steps");
    println!("{}", "=".repeat(60));

    for &(name, op, nw) in &tasks {
        for &ticks in &[2, 3, 4] {
            let relu_res: Vec<(f64, usize)> = seeds.par_iter().map(|&seed| {
                let mut rng = StdRng::seed_from_u64(seed);
                let mut net = TwoPoolNet::new(nc, nw, nw, ticks, false, &mut rng, 0.5);
                optimize(&mut net, op)
            }).collect();

            let c19_res: Vec<(f64, usize)> = seeds.par_iter().map(|&seed| {
                let mut rng = StdRng::seed_from_u64(seed);
                let mut net = TwoPoolNet::new(nc, nw, nw, ticks, true, &mut rng, 0.5);
                optimize(&mut net, op)
            }).collect();

            let relu_s = relu_res.iter().filter(|r| r.0 >= 1.0).count();
            let relu_st: f64 = relu_res.iter().map(|r| r.1 as f64).sum::<f64>() / n_seeds as f64;
            let c19_s = c19_res.iter().filter(|r| r.0 >= 1.0).count();
            let c19_st: f64 = c19_res.iter().map(|r| r.1 as f64).sum::<f64>() / n_seeds as f64;

            println!("{:>8} {:>5}  {:>7}/{:<2} {:>7.0}  {:>7}/{:<2} {:>7.0}",
                name, ticks, relu_s, n_seeds, relu_st, c19_s, n_seeds, c19_st);
        }
        println!();
    }

    // =========================================================
    // PART 2: Main comparison at ticks=3, sweep workers
    // =========================================================
    let best_ticks = 3; // likely the sweet spot
    println!("--- PART 2: Two-pool vs Single-pool (ticks={}, nc=3, 50 seeds) ---\n", best_ticks);
    println!("{:>8} {:>4}  {:>10} {:>10}  {:>10} {:>10}  {:>6}",
        "task", "nw", "1p_ReLU", "1p_C19", "2p_ReLU", "2p_C19", "C19gap");
    println!("{}", "=".repeat(72));

    for &(name, op, base_nw) in &tasks {
        for &nw_side in &[2, 3, 4, 5] {
            let total_nw = nw_side * 2;

            // Single-pool baseline (same total workers)
            let sp_relu: Vec<(f64, usize)> = seeds.par_iter().map(|&seed| {
                let mut rng = StdRng::seed_from_u64(seed);
                let mut net = SinglePoolNet::new(nc, total_nw, best_ticks, false, &mut rng, 0.5);
                optimize(&mut net, op)
            }).collect();
            let sp_c19: Vec<(f64, usize)> = seeds.par_iter().map(|&seed| {
                let mut rng = StdRng::seed_from_u64(seed);
                let mut net = SinglePoolNet::new(nc, total_nw, best_ticks, true, &mut rng, 0.5);
                optimize(&mut net, op)
            }).collect();

            // Two-pool
            let tp_relu: Vec<(f64, usize)> = seeds.par_iter().map(|&seed| {
                let mut rng = StdRng::seed_from_u64(seed);
                let mut net = TwoPoolNet::new(nc, nw_side, nw_side, best_ticks, false, &mut rng, 0.5);
                optimize(&mut net, op)
            }).collect();
            let tp_c19: Vec<(f64, usize)> = seeds.par_iter().map(|&seed| {
                let mut rng = StdRng::seed_from_u64(seed);
                let mut net = TwoPoolNet::new(nc, nw_side, nw_side, best_ticks, true, &mut rng, 0.5);
                optimize(&mut net, op)
            }).collect();

            let sp_r = sp_relu.iter().filter(|r| r.0 >= 1.0).count();
            let sp_c = sp_c19.iter().filter(|r| r.0 >= 1.0).count();
            let tp_r = tp_relu.iter().filter(|r| r.0 >= 1.0).count();
            let tp_c = tp_c19.iter().filter(|r| r.0 >= 1.0).count();

            // C19 advantage gap: how much MORE does C19 help in 2-pool vs 1-pool?
            let gap_1p = sp_c as i32 - sp_r as i32;
            let gap_2p = tp_c as i32 - tp_r as i32;
            let c19_gap = gap_2p - gap_1p;

            println!("{:>8} {:>4}  {:>7}/{:<2} {:>7}/{:<2}  {:>7}/{:<2} {:>7}/{:<2}  {:>+5}",
                name, nw_side,
                sp_r, n_seeds, sp_c, n_seeds,
                tp_r, n_seeds, tp_c, n_seeds,
                c19_gap);
        }
        println!();
    }

    // =========================================================
    // PART 3: Learned rho+C analysis (C19, 2-pool, 20 seeds)
    // =========================================================
    println!("--- PART 3: Learned rho+C per neighborhood (C19, ticks={}, 20 seeds) ---\n", best_ticks);

    let detail_seeds: Vec<u64> = (1..=20u64).collect();

    for &(name, op, nw) in &tasks {
        println!("  === {} ({}w/side, ticks={}) ===", name, nw, best_ticks);

        let results: Vec<(f64, usize, Vec<f32>, Vec<f32>)> = detail_seeds.iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = TwoPoolNet::new(nc, nw, nw, best_ticks, true, &mut rng, 0.5);
            let (acc, steps) = optimize(&mut net, op);
            (acc, steps, net.c_params.clone(), net.rho_params.clone())
        }).collect();

        // Per-worker stats, split by neighborhood
        println!("    Neighborhood A:");
        for w in 0..nw {
            let c_vals: Vec<f32> = results.iter().map(|r| r.2[w]).collect();
            let r_vals: Vec<f32> = results.iter().map(|r| r.3[w]).collect();
            let c_mean: f32 = c_vals.iter().sum::<f32>() / c_vals.len() as f32;
            let r_mean: f32 = r_vals.iter().sum::<f32>() / r_vals.len() as f32;
            let c_std: f32 = (c_vals.iter().map(|v| (v - c_mean).powi(2)).sum::<f32>() / c_vals.len() as f32).sqrt();
            let r_std: f32 = (r_vals.iter().map(|v| (v - r_mean).powi(2)).sum::<f32>() / r_vals.len() as f32).sqrt();
            println!("      Worker {}: C={:.2}+/-{:.2}  rho={:.2}+/-{:.2}", w, c_mean, c_std, r_mean, r_std);
        }
        println!("    Neighborhood B:");
        for w in 0..nw {
            let ci = nw + w;
            let c_vals: Vec<f32> = results.iter().map(|r| r.2[ci]).collect();
            let r_vals: Vec<f32> = results.iter().map(|r| r.3[ci]).collect();
            let c_mean: f32 = c_vals.iter().sum::<f32>() / c_vals.len() as f32;
            let r_mean: f32 = r_vals.iter().sum::<f32>() / r_vals.len() as f32;
            let c_std: f32 = (c_vals.iter().map(|v| (v - c_mean).powi(2)).sum::<f32>() / c_vals.len() as f32).sqrt();
            let r_std: f32 = (r_vals.iter().map(|v| (v - r_mean).powi(2)).sum::<f32>() / r_vals.len() as f32).sqrt();
            println!("      Worker {}: C={:.2}+/-{:.2}  rho={:.2}+/-{:.2}", w, c_mean, c_std, r_mean, r_std);
        }

        let solved = results.iter().filter(|r| r.0 >= 1.0).count();
        println!("    {}/{} solved\n", solved, detail_seeds.len());
    }

    println!("=== DONE ===");
}
