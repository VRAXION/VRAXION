//! C19 with BOTH rho AND C learnable per-neuron
//!
//! Question: does learnable rho improve over fixed rho=4.0?
//! Where does rho converge? Is it still ~4.0?
//!
//! Run: cargo run --example c19_rho_learnable --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const DIGITS: usize = 5;
const INPUT_DIM: usize = 8;
const LOCAL_CAP: usize = 3;
const TICKS: usize = 2;

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

fn thermo_2(a: usize, b: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; INPUT_DIM];
    for i in 0..a.min(4) { v[i] = 1.0; }
    for i in 0..b.min(4) { v[4 + i] = 1.0; }
    v
}

#[derive(Clone)]
struct FlatNet {
    n_connectome: usize,
    n_workers: usize,
    params: Vec<f32>,
    c_params: Vec<f32>,
    rho_params: Vec<f32>,
    worker_param_offsets: Vec<usize>,
    worker_local_counts: Vec<usize>,
    use_c19: bool,
    learn_rho: bool,
}

impl FlatNet {
    fn new_random(nc: usize, nw: usize, use_c19: bool, learn_rho: bool, rng: &mut StdRng, scale: f32) -> Self {
        let mut net = FlatNet {
            n_connectome: nc, n_workers: 0,
            params: Vec::new(), c_params: Vec::new(), rho_params: Vec::new(),
            worker_param_offsets: Vec::new(),
            worker_local_counts: Vec::new(),
            use_c19, learn_rho,
        };
        for i in 0..nw {
            let nl = LOCAL_CAP.min(i);
            let np = INPUT_DIM + nl + nc + 1 + 1;
            let init: Vec<f32> = (0..np).map(|_| rng.gen_range(-scale..scale)).collect();
            net.worker_param_offsets.push(net.params.len());
            net.worker_local_counts.push(nl);
            net.params.extend_from_slice(&init);
            net.c_params.push(1.0);
            net.rho_params.push(4.0); // init at known-good value
            net.n_workers += 1;
        }
        net
    }

    fn forward(&self, a: usize, b: usize) -> f32 {
        let input = thermo_2(a, b);
        let nc = self.n_connectome;
        let nw = self.n_workers;
        let mut act = vec![0.0f32; nc + nw];
        for _t in 0..TICKS {
            let mut cc = vec![0.0f32; nc];
            for i in 0..nw {
                let o = self.worker_param_offsets[i];
                let nl = self.worker_local_counts[i];
                let ww = self.params[o + INPUT_DIM + nl + nc];
                let wi = i % nc.max(1);
                if wi < nc { cc[wi] += act[nc + i] * ww; }
            }
            for i in 0..nc { act[i] = cc[i]; }
            let old = act.clone();
            for i in 0..nw {
                let o = self.worker_param_offsets[i];
                let nl = self.worker_local_counts[i];
                let mut s = self.params[o + INPUT_DIM + nl + nc + 1];
                for j in 0..INPUT_DIM { s += input[j] * self.params[o + j]; }
                let ls = i.saturating_sub(nl);
                for (k, wi) in (ls..i).enumerate() {
                    s += old[nc + wi] * self.params[o + INPUT_DIM + k];
                }
                for k in 0..nc { s += old[k] * self.params[o + INPUT_DIM + nl + k]; }
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
            if self.learn_rho {
                for i in 0..nw {
                    let orig = self.rho_params[i];
                    self.rho_params[i] = orig + eps; let lp = self.mse_loss(op);
                    self.rho_params[i] = orig - eps; let lm = self.mse_loss(op);
                    self.rho_params[i] = orig;
                    gr[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
                }
            }
        }
        (g, gc, gr)
    }
}

fn optimize(net: &mut FlatNet, op: fn(usize, usize) -> usize) -> (f64, usize) {
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

        let old_p = net.params.clone();
        let old_c = net.c_params.clone();
        let old_r = net.rho_params.clone();
        let ol = net.mse_loss(op);

        let mut improved = false;
        for att in 0..5 {
            for i in 0..net.params.len() { net.params[i] = old_p[i] - lr * g[i] / gn; }
            for i in 0..net.c_params.len() {
                net.c_params[i] = (old_c[i] - lr * gc[i] / gn).max(0.1);
            }
            if net.learn_rho {
                for i in 0..net.rho_params.len() {
                    net.rho_params[i] = (old_r[i] - lr * gr[i] / gn).max(0.0);
                }
            }
            let nl = net.mse_loss(op);
            if nl < ol {
                lr *= 1.1;
                if nl < best_loss - 1e-8 { best_loss = nl; stale = 0; improved = true; }
                break;
            } else {
                lr *= 0.5;
                if att == 4 {
                    net.params = old_p.clone();
                    net.c_params = old_c.clone();
                    net.rho_params = old_r.clone();
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
    println!("=== C19: LEARNABLE RHO + C vs FIXED RHO ===\n");

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
    // PART 1: Head-to-head — ReLU vs C19(rho=4 fix) vs C19(rho learnable)
    // =========================================================
    println!("--- PART 1: ReLU vs C19(rho=4) vs C19(rho+C learnable) — 50 seeds ---\n");
    println!("{:>8} {:>4}  {:>10} {:>8}  {:>10} {:>8}  {:>10} {:>8}",
        "task", "nw",
        "ReLU_solv", "steps",
        "C19fix_s", "steps",
        "C19learn", "steps");
    println!("{}", "=".repeat(80));

    for &(name, op, nw) in &tasks {
        // ReLU
        let relu_res: Vec<(f64, usize)> = seeds.par_iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = FlatNet::new_random(nc, nw, false, false, &mut rng, 0.5);
            optimize(&mut net, op)
        }).collect();

        // C19 rho=4 fixed, C learnable
        let c19fix_res: Vec<(f64, usize)> = seeds.par_iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = FlatNet::new_random(nc, nw, true, false, &mut rng, 0.5);
            optimize(&mut net, op)
        }).collect();

        // C19 rho+C both learnable
        let c19learn_res: Vec<(f64, usize)> = seeds.par_iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = FlatNet::new_random(nc, nw, true, true, &mut rng, 0.5);
            optimize(&mut net, op)
        }).collect();

        let relu_s = relu_res.iter().filter(|r| r.0 >= 1.0).count();
        let relu_st: f64 = relu_res.iter().map(|r| r.1 as f64).sum::<f64>() / n_seeds as f64;
        let fix_s = c19fix_res.iter().filter(|r| r.0 >= 1.0).count();
        let fix_st: f64 = c19fix_res.iter().map(|r| r.1 as f64).sum::<f64>() / n_seeds as f64;
        let learn_s = c19learn_res.iter().filter(|r| r.0 >= 1.0).count();
        let learn_st: f64 = c19learn_res.iter().map(|r| r.1 as f64).sum::<f64>() / n_seeds as f64;

        println!("{:>8} {:>4}  {:>7}/{:<2} {:>7.0}  {:>7}/{:<2} {:>7.0}  {:>7}/{:<2} {:>7.0}",
            name, nw,
            relu_s, n_seeds, relu_st,
            fix_s, n_seeds, fix_st,
            learn_s, n_seeds, learn_st);
    }

    // =========================================================
    // PART 2: Where does rho converge? (per task, 20 seeds)
    // =========================================================
    println!("\n--- PART 2: Learned rho + C values (20 seeds, converged) ---\n");

    let detail_seeds: Vec<u64> = (1..=20u64).collect();

    for &(name, op, nw) in &tasks {
        println!("  === {} ({} workers) ===", name, nw);

        let results: Vec<(f64, usize, Vec<f32>, Vec<f32>)> = detail_seeds.iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = FlatNet::new_random(nc, nw, true, true, &mut rng, 0.5);
            let (acc, steps) = optimize(&mut net, op);
            (acc, steps, net.c_params.clone(), net.rho_params.clone())
        }).collect();

        for (i, (acc, steps, c_vals, rho_vals)) in results.iter().enumerate() {
            let c_str: Vec<String> = c_vals.iter().map(|v| format!("{:.2}", v)).collect();
            let r_str: Vec<String> = rho_vals.iter().map(|v| format!("{:.2}", v)).collect();
            println!("    seed {:>2}: acc={:>3.0}% steps={:>4}  C=[{}]  rho=[{}]",
                i + 1, acc * 100.0, steps, c_str.join(", "), r_str.join(", "));
        }

        for w in 0..nw {
            let c_vals: Vec<f32> = results.iter().map(|r| r.2[w]).collect();
            let r_vals: Vec<f32> = results.iter().map(|r| r.3[w]).collect();
            let c_mean: f32 = c_vals.iter().sum::<f32>() / c_vals.len() as f32;
            let r_mean: f32 = r_vals.iter().sum::<f32>() / r_vals.len() as f32;
            let c_std: f32 = (c_vals.iter().map(|v| (v - c_mean).powi(2)).sum::<f32>() / c_vals.len() as f32).sqrt();
            let r_std: f32 = (r_vals.iter().map(|v| (v - r_mean).powi(2)).sum::<f32>() / r_vals.len() as f32).sqrt();
            println!("    Worker {}: C={:.2}±{:.2}  rho={:.2}±{:.2}", w, c_mean, c_std, r_mean, r_std);
        }

        let solved = results.iter().filter(|r| r.0 >= 1.0).count();
        println!("    {}/{} solved\n", solved, detail_seeds.len());
    }

    println!("=== DONE ===");
}
