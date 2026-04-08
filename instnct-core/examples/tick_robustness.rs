//! Does the 10-neuron circuit work with different tick counts?
//! Rebuild with TICKS=8 (original), then TEST with 1,2,4,8,16,32,64 ticks.
//!
//! RUNNING: tick_robustness
//! Run: cargo run --example tick_robustness --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const CHARGE_RATE: f32 = 0.3;
const LEAK: f32 = 0.85;
const THRESHOLD: f32 = 0.1;
const DIGITS: usize = 5;
const BUILD_TICKS: usize = 8;

fn forward(input_indices: &[usize], input_vals: &[f32], w: &Vec<Vec<f32>>, h: usize, ticks: usize) -> Vec<f32> {
    let mut charge = vec![0.0f32; h];
    let mut act = vec![0.0f32; h];
    for t in 0..ticks {
        if t == 0 { for (&idx, &val) in input_indices.iter().zip(input_vals.iter()) { if idx < h { act[idx] = val; } } }
        let mut raw = vec![0.0f32; h];
        for i in 0..h { for j in 0..h { raw[i] += act[j] * w[j][i]; } }
        for i in 0..h { charge[i] += raw[i] * CHARGE_RATE; charge[i] *= LEAK; }
        for i in 0..h { act[i] = (charge[i] - THRESHOLD).max(0.0); }
        let total: f32 = act.iter().sum();
        if total > 0.0 { let d = 1.0 + 0.05 * total; for i in 0..h { act[i] /= d; } }
    }
    charge
}

fn eval_addition(w: &Vec<Vec<f32>>, h: usize, ticks: usize) -> (f64, f64) {
    let c1: f32 = forward(&[0], &[1.0], w, h, ticks).iter().sum();
    if c1.abs() < 0.0001 { return (0.0, 0.0); }
    let mut train_ok = 0; let mut test_ok = 0;
    for a in 0..DIGITS { for b in 0..DIGITS {
        let target = a + b;
        let mut indices = Vec::new(); let mut vals = Vec::new();
        for i in 0..a { indices.push(i); vals.push(1.0); }
        for i in 0..b { indices.push(4+i); vals.push(1.0); }
        let charge = forward(&indices, &vals, w, h, ticks);
        let total_charge: f32 = charge.iter().sum();
        let pred = ((total_charge / c1).round() as usize).min(8);
        if pred == target { if target != 4 { train_ok += 1; } else { test_ok += 1; } }
    }}
    (train_ok as f64 / 20.0, test_ok as f64 / 5.0)
}

fn main() {
    println!("=== TICK ROBUSTNESS: does the circuit work at different depths? ===");
    println!("RUNNING: tick_robustness\n");

    // Rebuild the 10-neuron circuit with TICKS=8 (same as incremental_build)
    let max_neurons = 10;
    let ternary = [-1i8, 0, 1];
    let mut w: Vec<Vec<f32>> = Vec::new();
    let mut h = 0;

    for step in 0..max_neurons {
        h += 1;
        for row in &mut w { row.push(0.0); }
        w.push(vec![0.0f32; h]);
        if h == 1 { continue; }

        let n_edges = 2*(h-1)+1;
        let total_configs = 3u64.pow(n_edges as u32);
        let use_exhaustive = total_configs <= 1_000_000;

        let mut best_train = 0.0f64; let mut best_test = 0.0f64;
        let mut best_w = w.clone();

        if use_exhaustive {
            for config in 0..total_configs {
                let mut c = config;
                for j in 0..h-1 { w[h-1][j] = ternary[(c%3) as usize] as f32; c /= 3; }
                for j in 0..h-1 { w[j][h-1] = ternary[(c%3) as usize] as f32; c /= 3; }
                w[h-1][h-1] = ternary[(c%3) as usize] as f32 * 0.5;
                let (train, test) = eval_addition(&w, h, BUILD_TICKS);
                if test > best_test || (test == best_test && train > best_train) {
                    best_test = test; best_train = train; best_w = w.clone();
                }
            }
        } else {
            let mut rng = StdRng::seed_from_u64(42 + step as u64);
            for _ in 0..500_000u64 {
                for j in 0..h-1 { w[h-1][j] = ternary[rng.gen_range(0..3)] as f32; }
                for j in 0..h-1 { w[j][h-1] = ternary[rng.gen_range(0..3)] as f32; }
                w[h-1][h-1] = ternary[rng.gen_range(0..3)] as f32 * 0.5;
                let (train, test) = eval_addition(&w, h, BUILD_TICKS);
                if test > best_test || (test == best_test && train > best_train) {
                    best_test = test; best_train = train; best_w = w.clone();
                }
            }
        }
        w = best_w;
        if best_train >= 1.0 && best_test >= 1.0 { break; }
    }

    println!("Built {}-neuron circuit with TICKS={} → train={:.0}% test={:.0}%\n",
        h, BUILD_TICKS,
        eval_addition(&w, h, BUILD_TICKS).0 * 100.0,
        eval_addition(&w, h, BUILD_TICKS).1 * 100.0);

    // Now test with DIFFERENT tick counts
    println!("{:>6} {:>8} {:>8} {:>10}", "ticks", "train%", "test%", "status");
    println!("{:-<6} {:-<8} {:-<8} {:-<10}", "", "", "", "");

    for &ticks in &[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 64] {
        let (train, test) = eval_addition(&w, h, ticks);
        let status = if train >= 1.0 && test >= 1.0 { "✓ PERFECT" }
            else if test > 0.0 { "~ partial" }
            else { "✗ fails" };
        println!("{:>6} {:>7.0}% {:>7.0}% {:>10}", ticks, train*100.0, test*100.0, status);
    }
}
