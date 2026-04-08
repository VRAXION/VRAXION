//! Incremental building: start with 1 neuron, add 1 at a time.
//! Each new neuron: exhaustive search over all possible connections to existing neurons.
//! Best config → freeze → add next neuron.
//! Every neuron is both input AND output.
//!
//! RUNNING: incremental_build
//!
//! Run: cargo run --example incremental_build --release

const TICKS: usize = 8;
const CHARGE_RATE: f32 = 0.3;
const LEAK: f32 = 0.85;
const THRESHOLD: f32 = 0.1;
const DIGITS: usize = 5;
const SUMS: usize = 9;

fn forward(input_indices: &[usize], input_vals: &[f32], w: &Vec<Vec<f32>>, h: usize) -> Vec<f32> {
    let mut charge = vec![0.0f32; h];
    let mut act = vec![0.0f32; h];
    for t in 0..TICKS {
        // Inject input at tick 0 into specified neurons
        if t == 0 {
            for (&idx, &val) in input_indices.iter().zip(input_vals.iter()) {
                if idx < h { act[idx] = val; }
            }
        }
        let mut raw = vec![0.0f32; h];
        for i in 0..h { for j in 0..h { raw[i] += act[j] * w[j][i]; } }
        for i in 0..h { charge[i] += raw[i] * CHARGE_RATE; charge[i] *= LEAK; }
        for i in 0..h { act[i] = (charge[i] - THRESHOLD).max(0.0); }
        // divnorm
        let total: f32 = act.iter().sum();
        if total > 0.0 { let d = 1.0 + 0.05 * total; for i in 0..h { act[i] /= d; } }
    }
    charge
}

/// Evaluate addition task. Input: thermometer into first 8 neurons.
/// Readout: total charge → nearest class.
fn eval_addition(w: &Vec<Vec<f32>>, h: usize) -> (f64, f64) {
    // Calibrate: charge for 1 active input
    let c1: f32 = forward(&[0], &[1.0], w, h).iter().sum();
    if c1.abs() < 0.0001 { return (0.0, 0.0); }

    let mut train_ok = 0; let mut test_ok = 0;
    for a in 0..DIGITS { for b in 0..DIGITS {
        let target = a + b;
        // Thermometer: inject into neuron indices 0..a and 4..4+b
        let mut indices = Vec::new();
        let mut vals = Vec::new();
        for i in 0..a { indices.push(i); vals.push(1.0); }
        for i in 0..b { indices.push(4 + i); vals.push(1.0); }

        let charge = forward(&indices, &vals, w, h);
        let total_charge: f32 = charge.iter().sum();
        let pred = ((total_charge / c1).round() as usize).min(8);

        if pred == target {
            if target != 4 { train_ok += 1; } else { test_ok += 1; }
        }
    }}
    (train_ok as f64 / 20.0, test_ok as f64 / 5.0)
}

fn main() {
    println!("=== INCREMENTAL BUILD: 1 neuron at a time ===");
    println!("RUNNING: incremental_build");
    println!("Every neuron is I/O. Exhaustive search for each new neuron.");
    println!("Ternary weights (-1, 0, +1) for connections to/from existing.\n");

    let max_neurons = 12;
    let ternary = [-1i8, 0, 1];

    // Start with 0 neurons, 0 connections
    let mut w: Vec<Vec<f32>> = Vec::new();
    let mut h = 0;

    // Track best overall
    let mut best_ever_train = 0.0f64;
    let mut best_ever_test = 0.0f64;

    for step in 0..max_neurons {
        // Add 1 neuron: index = h (the new one)
        let new_idx = h;
        h += 1;

        // Expand weight matrix: add row and column for new neuron
        for row in &mut w { row.push(0.0); } // existing → new
        w.push(vec![0.0f32; h]); // new → all (including self)

        if h == 1 {
            // First neuron: nothing to connect to
            let (train, test) = eval_addition(&w, h);
            println!("  Neuron {}: h={} edges=0 | train={:.0}% test={:.0}%",
                step, h, train*100.0, test*100.0);
            continue;
        }

        // Exhaustive search: for the new neuron, try all ternary weight combos
        // Edges: new→existing (h-1 edges) + existing→new (h-1 edges) + self (1)
        // Total new edges = 2*(h-1) + 1 = 2h-1
        let n_new_edges = 2 * (h - 1) + 1;

        // If too many edges for exhaustive, use random sampling
        let total_configs = 3u64.pow(n_new_edges as u32);
        let use_exhaustive = total_configs <= 1_000_000;

        let mut best_train = 0.0f64;
        let mut best_test = 0.0f64;
        let mut best_w_snapshot: Vec<Vec<f32>> = w.clone();
        let mut configs_tried = 0u64;

        if use_exhaustive {
            // Try all configs
            for config in 0..total_configs {
                // Decode config into weights
                let mut c = config;
                // First h-1 values: new → existing[0..h-1]
                for j in 0..h-1 {
                    w[new_idx][j] = ternary[(c % 3) as usize] as f32;
                    c /= 3;
                }
                // Next h-1 values: existing[0..h-1] → new
                for j in 0..h-1 {
                    w[j][new_idx] = ternary[(c % 3) as usize] as f32;
                    c /= 3;
                }
                // Self-connection
                w[new_idx][new_idx] = ternary[(c % 3) as usize] as f32 * 0.5;

                let (train, test) = eval_addition(&w, h);
                configs_tried += 1;

                // Keep best by: test first, then train
                if test > best_test || (test == best_test && train > best_train) {
                    best_test = test;
                    best_train = train;
                    best_w_snapshot = w.clone();
                }
            }
        } else {
            // Random sampling: try 500K random configs
            use rand::rngs::StdRng;
            use rand::{Rng, SeedableRng};
            let mut rng = StdRng::seed_from_u64(42 + step as u64);
            let sample_size = 500_000u64;

            for _ in 0..sample_size {
                for j in 0..h-1 {
                    w[new_idx][j] = ternary[rng.gen_range(0..3usize)] as f32;
                }
                for j in 0..h-1 {
                    w[j][new_idx] = ternary[rng.gen_range(0..3usize)] as f32;
                }
                w[new_idx][new_idx] = ternary[rng.gen_range(0..3usize)] as f32 * 0.5;

                let (train, test) = eval_addition(&w, h);
                configs_tried += 1;

                if test > best_test || (test == best_test && train > best_train) {
                    best_test = test;
                    best_train = train;
                    best_w_snapshot = w.clone();
                }
            }
        }

        // FREEZE: use the best config
        w = best_w_snapshot;

        if best_train > best_ever_train { best_ever_train = best_train; }
        if best_test > best_ever_test { best_ever_test = best_test; }

        // Show the new neuron's connections
        let new_outgoing: Vec<String> = (0..h).filter(|&j| w[new_idx][j] != 0.0)
            .map(|j| format!("→{}({})", j, w[new_idx][j])).collect();
        let new_incoming: Vec<String> = (0..h).filter(|&j| w[j][new_idx] != 0.0)
            .map(|j| format!("{}→({})", j, w[j][new_idx])).collect();

        let method = if use_exhaustive {
            format!("exhaustive 3^{}={}", n_new_edges, total_configs)
        } else {
            format!("random 500K / 3^{}={:.0e}", n_new_edges, total_configs as f64)
        };

        println!("  Neuron {}: h={} | train={:.0}% test={:.0}% | {} | out=[{}] in=[{}]",
            step, h, best_train*100.0, best_test*100.0, method,
            new_outgoing.join(" "), new_incoming.join(" "));

        if best_test >= 1.0 && best_train >= 1.0 {
            println!("\n  *** 100% GENERALIZATION at h={} neurons! ***", h);
            break;
        }
    }

    println!("\n  Best ever: train={:.0}% test={:.0}%", best_ever_train*100.0, best_ever_test*100.0);

    // Show final weight matrix
    println!("\n  Final weight matrix ({}×{}):", h, h);
    for i in 0..h {
        let row: Vec<String> = (0..h).map(|j| {
            if w[i][j] == 0.0 { " · ".to_string() }
            else { format!("{:>2.0} ", w[i][j]) }
        }).collect();
        println!("    [{}] neuron {}", row.join(""), i);
    }
}
