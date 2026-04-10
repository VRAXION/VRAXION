//! Overnight Scaling + Generalization Test
//!
//! Deep questions to answer:
//!   1. Does the gradient pipeline work beyond 5 digits? (10, 20, 50 digits)
//!   2. Does it generalize? (train on 80% pairs, test on 20% held-out)
//!   3. How many workers needed as digit range grows?
//!   4. Does circuit reuse scale? (frozen ADD(a,b) helps with ADD(a,b,c) at 10+ digits?)
//!   5. Multi-task: can one network learn ADD and MAX simultaneously?
//!
//! Run: cargo run --example overnight_scaling --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rayon::prelude::*;
use std::time::Instant;

const TICKS: usize = 2;

fn relu(x: f32) -> f32 { x.max(0.0) }

fn thermo_2(a: usize, b: usize, max_d: usize) -> Vec<f32> {
    let bits = max_d - 1; // e.g. 5 digits → 4 thermo bits
    let mut v = vec![0.0f32; bits * 2];
    for i in 0..a.min(bits) { v[i] = 1.0; }
    for i in 0..b.min(bits) { v[bits + i] = 1.0; }
    v
}

// ============================================================
// Flat network (variable input dim)
// ============================================================
#[derive(Clone)]
struct Net {
    nc: usize,
    nw: usize,
    params: Vec<f32>,
    offsets: Vec<usize>,
    local_counts: Vec<usize>,
    input_dim: usize,
}

impl Net {
    fn new_random(nc: usize, nw: usize, input_dim: usize, rng: &mut StdRng, scale: f32) -> Self {
        let mut net = Net {
            nc, nw: 0, params: Vec::new(), offsets: Vec::new(),
            local_counts: Vec::new(), input_dim,
        };
        for i in 0..nw {
            let nl = 3usize.min(i);
            let np = input_dim + nl + nc + 1 + 1;
            let init: Vec<f32> = (0..np).map(|_| rng.gen_range(-scale..scale)).collect();
            net.offsets.push(net.params.len());
            net.local_counts.push(nl);
            net.params.extend_from_slice(&init);
            net.nw += 1;
        }
        net
    }

    fn forward(&self, input: &[f32]) -> f32 {
        let nc = self.nc;
        let nw = self.nw;
        let mut act = vec![0.0f32; nc + nw];
        for _t in 0..TICKS {
            let mut cc = vec![0.0f32; nc];
            for i in 0..nw {
                let o = self.offsets[i];
                let nl = self.local_counts[i];
                let ww = self.params[o + self.input_dim + nl + nc];
                let wi = i % nc.max(1);
                if wi < nc { cc[wi] += act[nc + i] * ww; }
            }
            for i in 0..nc { act[i] = cc[i]; }
            let old = act.clone();
            for i in 0..nw {
                let o = self.offsets[i];
                let nl = self.local_counts[i];
                let mut s = self.params[o + self.input_dim + nl + nc + 1];
                for j in 0..self.input_dim { if j < input.len() { s += input[j] * self.params[o + j]; } }
                let ls = i.saturating_sub(nl);
                for (k, wi) in (ls..i).enumerate() { s += old[nc + wi] * self.params[o + self.input_dim + k]; }
                for k in 0..nc { s += old[k] * self.params[o + self.input_dim + nl + k]; }
                act[nc + i] = relu(s);
            }
        }
        act[nc..].iter().sum()
    }

    fn mse_on_pairs(&self, pairs: &[(usize, usize, usize)], max_d: usize) -> f64 {
        let mut loss = 0.0f64;
        for &(a, b, target) in pairs {
            let input = thermo_2(a, b, max_d);
            let charge = self.forward(&input) as f64;
            loss += (charge - target as f64).powi(2);
        }
        loss / pairs.len() as f64
    }

    fn accuracy_on_pairs(&self, pairs: &[(usize, usize, usize)], max_d: usize) -> f64 {
        let mut correct = 0;
        for &(a, b, target) in pairs {
            let input = thermo_2(a, b, max_d);
            if (self.forward(&input).round() as i32) == (target as i32) { correct += 1; }
        }
        correct as f64 / pairs.len() as f64
    }

    fn gradient_on_pairs(&mut self, pairs: &[(usize, usize, usize)], max_d: usize) -> Vec<f32> {
        let eps = 1e-3f32;
        let n = self.params.len();
        let mut grad = vec![0.0f32; n];
        for i in 0..n {
            let orig = self.params[i];
            self.params[i] = orig + eps;
            let lp = self.mse_on_pairs(pairs, max_d);
            self.params[i] = orig - eps;
            let lm = self.mse_on_pairs(pairs, max_d);
            self.params[i] = orig;
            grad[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
        }
        grad
    }
}

fn optimize(net: &mut Net, pairs: &[(usize, usize, usize)], max_d: usize, steps: usize) -> (f64, usize) {
    let mut lr = 0.01f32;
    for step in 0..steps {
        let acc = net.accuracy_on_pairs(pairs, max_d);
        if acc >= 1.0 { return (acc, step); }
        let grad = net.gradient_on_pairs(pairs, max_d);
        let gn: f32 = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
        if gn < 1e-8 { return (acc, step); }
        let old = net.params.clone();
        let ol = net.mse_on_pairs(pairs, max_d);
        for att in 0..5 {
            for i in 0..net.params.len() { net.params[i] = old[i] - lr * grad[i] / gn; }
            if net.mse_on_pairs(pairs, max_d) < ol { lr *= 1.1; break; }
            else { lr *= 0.5; if att == 4 { net.params = old.clone(); } }
        }
    }
    (net.accuracy_on_pairs(pairs, max_d), steps)
}

fn make_all_pairs(max_d: usize, op: fn(usize, usize) -> usize) -> Vec<(usize, usize, usize)> {
    let mut pairs = Vec::new();
    for a in 0..max_d { for b in 0..max_d { pairs.push((a, b, op(a, b))); } }
    pairs
}

fn train_test_split(pairs: &[(usize, usize, usize)], test_frac: f64, seed: u64) -> (Vec<(usize, usize, usize)>, Vec<(usize, usize, usize)>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut shuffled = pairs.to_vec();
    shuffled.shuffle(&mut rng);
    let test_n = (pairs.len() as f64 * test_frac).ceil() as usize;
    let test = shuffled[..test_n].to_vec();
    let train = shuffled[test_n..].to_vec();
    (train, test)
}

fn op_add(a: usize, b: usize) -> usize { a + b }
fn op_max(a: usize, b: usize) -> usize { a.max(b) }
fn op_min(a: usize, b: usize) -> usize { a.min(b) }
fn op_sub_abs(a: usize, b: usize) -> usize { if a > b { a - b } else { b - a } }

fn main() {
    let t0 = Instant::now();
    println!("=== OVERNIGHT SCALING + GENERALIZATION TEST ===\n");

    let nc = 3;
    let n_seeds = 20;
    let seeds: Vec<u64> = (1..=n_seeds as u64).collect();

    // =========================================================
    // EXP 1: Scaling — how far does gradient pipeline reach?
    // =========================================================
    println!("--- EXP 1: Scaling — digit range × worker count (ADD) ---\n");
    println!("{:>6} {:>5} {:>6} {:>8} {:>8} {:>8} {:>8}",
        "digits", "pairs", "workers", "best", "mean", "solved", "time");
    println!("{}", "=".repeat(60));

    let tasks_exp1: Vec<(usize, usize)> = vec![
        (5, 3), (5, 6),
        (8, 3), (8, 6), (8, 10),
        (10, 6), (10, 10), (10, 15),
        (15, 10), (15, 15), (15, 20),
        (20, 15), (20, 20),
    ];

    for &(max_d, nw) in &tasks_exp1 {
        let input_dim = (max_d - 1) * 2;
        let all_pairs = make_all_pairs(max_d, op_add);
        let t1 = Instant::now();

        let results: Vec<(f64, usize)> = seeds.par_iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = Net::new_random(nc, nw, input_dim, &mut rng, 0.5);
            optimize(&mut net, &all_pairs, max_d, 3000)
        }).collect();

        let best = results.iter().map(|r| r.0).fold(0.0f64, f64::max);
        let mean: f64 = results.iter().map(|r| r.0).sum::<f64>() / results.len() as f64;
        let solved = results.iter().filter(|r| r.0 >= 1.0).count();

        println!("{:>6} {:>5} {:>7} {:>7.0}% {:>7.0}% {:>5}/{:>2} {:>7.1}s",
            max_d, all_pairs.len(), nw, best * 100.0, mean * 100.0,
            solved, n_seeds, t1.elapsed().as_secs_f64());
    }

    // =========================================================
    // EXP 2: Generalization — train/test split
    // =========================================================
    println!("\n--- EXP 2: Generalization (80% train, 20% test, ADD) ---\n");
    println!("{:>6} {:>5} {:>8} {:>8} {:>8} {:>8}",
        "digits", "workers", "train", "test", "gap", "time");
    println!("{}", "=".repeat(55));

    let gen_configs: Vec<(usize, usize)> = vec![
        (5, 4), (5, 6),
        (8, 6), (8, 10),
        (10, 10), (10, 15),
        (15, 15),
        (20, 20),
    ];

    for &(max_d, nw) in &gen_configs {
        let input_dim = (max_d - 1) * 2;
        let all_pairs = make_all_pairs(max_d, op_add);
        let t1 = Instant::now();

        let results: Vec<(f64, f64)> = seeds.par_iter().map(|&seed| {
            let (train, test) = train_test_split(&all_pairs, 0.2, seed);
            let mut rng = StdRng::seed_from_u64(seed + 1000);
            let mut net = Net::new_random(nc, nw, input_dim, &mut rng, 0.5);
            optimize(&mut net, &train, max_d, 3000);
            let train_acc = net.accuracy_on_pairs(&train, max_d);
            let test_acc = net.accuracy_on_pairs(&test, max_d);
            (train_acc, test_acc)
        }).collect();

        let mean_train: f64 = results.iter().map(|r| r.0).sum::<f64>() / results.len() as f64;
        let mean_test: f64 = results.iter().map(|r| r.1).sum::<f64>() / results.len() as f64;
        let gap = mean_train - mean_test;

        println!("{:>6} {:>7} {:>7.0}% {:>7.0}% {:>+7.1}pp {:>7.1}s",
            max_d, nw, mean_train * 100.0, mean_test * 100.0, gap * 100.0,
            t1.elapsed().as_secs_f64());
    }

    // =========================================================
    // EXP 3: Multi-task — ADD and MAX simultaneously
    // =========================================================
    println!("\n--- EXP 3: Multi-task (ADD + MAX on same network, 5 digits) ---\n");

    let max_d = 5;
    let input_dim = (max_d - 1) * 2;

    // Create pairs with both tasks encoded
    // We use extra input bits to signal which task: [thermo_a, thermo_b, task_bit]
    let input_dim_mt = input_dim + 1; // +1 for task selector

    println!("{:>8} {:>8} {:>8} {:>8} {:>8}",
        "workers", "ADD_acc", "MAX_acc", "combined", "time");
    println!("{}", "=".repeat(50));

    for &nw in &[4, 6, 8, 10] {
        let t1 = Instant::now();

        let results: Vec<(f64, f64)> = seeds.par_iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = Net::new_random(nc, nw, input_dim_mt, &mut rng, 0.5);

            // Build mixed pairs: task=0 → ADD, task=1 → MAX
            let mut pairs = Vec::new();
            for a in 0..max_d {
                for b in 0..max_d {
                    pairs.push((a, b, a + b, 0)); // ADD
                    pairs.push((a, b, a.max(b), 1)); // MAX
                }
            }

            // Custom training with task bit
            let mut lr = 0.01f32;
            let mt_loss = |net: &Net, pairs: &[(usize, usize, usize, usize)]| -> f64 {
                let mut loss = 0.0f64;
                for &(a, b, target, task) in pairs {
                    let mut input = thermo_2(a, b, max_d);
                    input.push(task as f32);
                    let charge = net.forward(&input) as f64;
                    loss += (charge - target as f64).powi(2);
                }
                loss / pairs.len() as f64
            };

            for _ in 0..3000 {
                let eps = 1e-3f32;
                let n = net.params.len();
                let mut grad = vec![0.0f32; n];
                let base_loss = mt_loss(&net, &pairs);

                for i in 0..n {
                    let orig = net.params[i];
                    net.params[i] = orig + eps;
                    let lp = mt_loss(&net, &pairs);
                    net.params[i] = orig - eps;
                    let lm = mt_loss(&net, &pairs);
                    net.params[i] = orig;
                    grad[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
                }

                let gn: f32 = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
                if gn < 1e-8 { break; }

                let old = net.params.clone();
                let ol = base_loss;
                for att in 0..5 {
                    for i in 0..n { net.params[i] = old[i] - lr * grad[i] / gn; }
                    if mt_loss(&net, &pairs) < ol { lr *= 1.1; break; }
                    else { lr *= 0.5; if att == 4 { net.params = old.clone(); } }
                }
            }

            // Evaluate per task
            let mut add_correct = 0;
            let mut max_correct = 0;
            for a in 0..max_d {
                for b in 0..max_d {
                    let mut input_add = thermo_2(a, b, max_d);
                    input_add.push(0.0);
                    if (net.forward(&input_add).round() as i32) == ((a + b) as i32) { add_correct += 1; }

                    let mut input_max = thermo_2(a, b, max_d);
                    input_max.push(1.0);
                    if (net.forward(&input_max).round() as i32) == (a.max(b) as i32) { max_correct += 1; }
                }
            }
            (add_correct as f64 / 25.0, max_correct as f64 / 25.0)
        }).collect();

        let add_mean: f64 = results.iter().map(|r| r.0).sum::<f64>() / results.len() as f64;
        let max_mean: f64 = results.iter().map(|r| r.1).sum::<f64>() / results.len() as f64;
        let combined = (add_mean + max_mean) / 2.0;

        println!("{:>8} {:>7.0}% {:>7.0}% {:>7.0}% {:>7.1}s",
            nw, add_mean * 100.0, max_mean * 100.0, combined * 100.0,
            t1.elapsed().as_secs_f64());
    }

    // =========================================================
    // EXP 4: All ops scaling comparison at 10 digits
    // =========================================================
    println!("\n--- EXP 4: All operations at 10 digits (10 workers) ---\n");

    let max_d = 10;
    let input_dim_10 = (max_d - 1) * 2;

    let ops: Vec<(&str, fn(usize, usize) -> usize)> = vec![
        ("ADD", op_add),
        ("MAX", op_max),
        ("MIN", op_min),
        ("|a-b|", op_sub_abs),
    ];

    println!("{:>8} {:>8} {:>8} {:>8} {:>8}",
        "task", "best", "mean", "solved", "time");
    println!("{}", "=".repeat(45));

    for &(name, op) in &ops {
        let all_pairs = make_all_pairs(max_d, op);
        let t1 = Instant::now();

        let results: Vec<f64> = seeds.par_iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = Net::new_random(nc, 10, input_dim_10, &mut rng, 0.5);
            let (acc, _) = optimize(&mut net, &all_pairs, max_d, 3000);
            acc
        }).collect();

        let best = results.iter().fold(0.0f64, |a, &b| a.max(b));
        let mean: f64 = results.iter().sum::<f64>() / results.len() as f64;
        let solved = results.iter().filter(|&&r| r >= 1.0).count();

        println!("{:>8} {:>7.0}% {:>7.0}% {:>5}/{:>2} {:>7.1}s",
            name, best * 100.0, mean * 100.0, solved, n_seeds, t1.elapsed().as_secs_f64());
    }

    // =========================================================
    // EXP 5: Composition depth — A→B→C chain
    // =========================================================
    println!("\n--- EXP 5: Composition depth (chain length, ADD, 5 digits) ---\n");
    println!("  Testing: can network learn a+b+c+d by chaining ADD circuits?\n");

    let max_d_chain = 5;
    let input_dim_chain_base = (max_d_chain - 1) * 2;

    for &n_inputs in &[2, 3, 4, 5] {
        let input_dim = (max_d_chain - 1) * n_inputs;
        let t1 = Instant::now();

        // Generate all combos
        let mut all_combos: Vec<(Vec<usize>, usize)> = Vec::new();
        let mut combo = vec![0usize; n_inputs];

        loop {
            let target: usize = combo.iter().sum();
            all_combos.push((combo.clone(), target));

            // Increment
            let mut carry = true;
            for i in (0..n_inputs).rev() {
                if carry {
                    combo[i] += 1;
                    if combo[i] >= max_d_chain { combo[i] = 0; } else { carry = false; }
                }
            }
            if carry { break; }
        }

        // Convert to pairs format (flatten input)
        let pairs: Vec<(Vec<f32>, usize)> = all_combos.iter().map(|(combo, target)| {
            let mut input = vec![0.0f32; input_dim];
            for (digit_idx, &val) in combo.iter().enumerate() {
                for i in 0..val.min(max_d_chain - 1) {
                    input[digit_idx * (max_d_chain - 1) + i] = 1.0;
                }
            }
            (input, *target)
        }).collect();

        let nw = n_inputs * 3; // scale workers with input count

        let results: Vec<f64> = seeds.par_iter().map(|&seed| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = Net::new_random(nc, nw, input_dim, &mut rng, 0.5);

            // Custom training on n-input combos
            let chain_loss = |net: &Net, pairs: &[(Vec<f32>, usize)]| -> f64 {
                let mut loss = 0.0f64;
                for (input, target) in pairs {
                    let charge = net.forward(input) as f64;
                    loss += (charge - *target as f64).powi(2);
                }
                loss / pairs.len() as f64
            };

            let mut lr = 0.01f32;
            for _ in 0..5000 {
                let eps = 1e-3f32;
                let n = net.params.len();
                let base_loss = chain_loss(&net, &pairs);
                let mut grad = vec![0.0f32; n];
                for i in 0..n {
                    let orig = net.params[i];
                    net.params[i] = orig + eps;
                    let lp = chain_loss(&net, &pairs);
                    net.params[i] = orig - eps;
                    let lm = chain_loss(&net, &pairs);
                    net.params[i] = orig;
                    grad[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
                }

                let gn: f32 = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
                if gn < 1e-8 { break; }
                let old = net.params.clone();
                let ol = base_loss;
                for att in 0..5 {
                    for i in 0..n { net.params[i] = old[i] - lr * grad[i] / gn; }
                    if chain_loss(&net, &pairs) < ol { lr *= 1.1; break; }
                    else { lr *= 0.5; if att == 4 { net.params = old.clone(); } }
                }
            }

            // Accuracy
            let mut correct = 0;
            for (input, target) in &pairs {
                if (net.forward(input).round() as i32) == (*target as i32) { correct += 1; }
            }
            correct as f64 / pairs.len() as f64
        }).collect();

        let best = results.iter().fold(0.0f64, |a, &b| a.max(b));
        let mean: f64 = results.iter().sum::<f64>() / results.len() as f64;
        let solved = results.iter().filter(|&&r| r >= 1.0).count();

        println!("  {}-input ADD ({}w, {} combos): best={:.0}% mean={:.0}% solved={}/{} ({:.1}s)",
            n_inputs, nw, pairs.len(), best * 100.0, mean * 100.0,
            solved, n_seeds, t1.elapsed().as_secs_f64());
    }

    println!("\n=== DONE ({:.1}s) ===", t0.elapsed().as_secs_f64());
}
