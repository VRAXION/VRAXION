//! Hardware benchmark: Ryzen 9 3900X + RTX 4070 Ti estimates
//!
//! Measure actual throughput on this machine for:
//! 1. Chip evaluation speed (evals/sec)
//! 2. Recurrent tick speed
//! 3. Max practical network size
//!
//! Run: cargo run --example hw_bench --release

use std::time::Instant;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn relu(x: f32) -> f32 { x.max(0.0) }

fn thermo(val: usize, bits: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; bits];
    for i in 0..val.min(bits) { v[i] = 1.0; }
    v
}

fn recurrent_tick(act: &mut [f32], input_thermo: &[f32], w: &[Vec<f32>], bias: &[f32]) {
    let n = w.len();
    let thermo_bits = input_thermo.len();
    for i in 0..n {
        let mut sum = bias[i];
        // Recurrent part
        for j in 0..n {
            sum += act[j] * w[i][j];
        }
        // Input part
        for j in 0..thermo_bits {
            sum += input_thermo[j] * w[i][n + j];
        }
        act[i] = relu(sum);
    }
}

fn main() {
    println!("=== HARDWARE BENCHMARK: Ryzen 9 3900X estimates ===\n");

    let mut rng = StdRng::seed_from_u64(42);

    // =========================================
    // BENCH 1: Chip eval throughput
    // =========================================
    println!("--- BENCH 1: Chip evaluation speed (single core) ---\n");

    for &(n, thermo_bits, label) in &[
        (3, 4, "3n×7 (ADD chip, 0..4)"),
        (3, 9, "3n×12 (ADD chip, 0..9)"),
        (3, 15, "3n×18 (ADD chip, 0..15)"),
        (5, 4, "5n×9"),
        (10, 4, "10n×14"),
        (20, 8, "20n×28"),
        (50, 8, "50n×58"),
        (100, 8, "100n×108"),
    ] {
        let input_dim = n + thermo_bits;
        let w: Vec<Vec<f32>> = (0..n).map(|_| (0..input_dim).map(|_| rng.gen_range(-2.0..2.0f32)).collect()).collect();
        let bias: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0..2.0f32)).collect();

        // Measure: how many recurrent ticks per second?
        let t_input = thermo(2, thermo_bits);
        let mut act = vec![0.0f32; n];
        let iters = 1_000_000u64;

        let start = Instant::now();
        for _ in 0..iters {
            recurrent_tick(&mut act, &t_input, &w, &bias);
        }
        let elapsed = start.elapsed().as_secs_f64();
        let ticks_per_sec = iters as f64 / elapsed;

        println!("  {:<22} {:>10.1}M ticks/sec  ({:.1} µs/tick)",
            label, ticks_per_sec / 1e6, elapsed / iters as f64 * 1e6);
    }

    // =========================================
    // BENCH 2: Random search throughput
    // =========================================
    println!("\n--- BENCH 2: Random search throughput (single core, 3-input ADD) ---\n");

    for &(n, thermo_bits, label) in &[
        (3, 4, "3n (0..4)"),
        (3, 9, "3n (0..9)"),
        (5, 4, "5n (0..4)"),
        (10, 4, "10n (0..4)"),
    ] {
        let input_dim = n + thermo_bits;
        let max_digit = if thermo_bits <= 4 { 5 } else { 10 };

        // Generate all 3-input combos
        let mut combos = Vec::new();
        for a in 0..max_digit { for b in 0..max_digit { for c in 0..max_digit {
            combos.push((a, b, c, a+b+c));
        }}}

        let n_evals = 100_000u64;
        let start = Instant::now();

        for _ in 0..n_evals {
            let w: Vec<Vec<f32>> = (0..n).map(|_| (0..input_dim).map(|_| rng.gen_range(-2.0..2.0f32)).collect()).collect();
            let bias: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0..2.0f32)).collect();

            let mut _score = 0usize;
            for &(a, b, c, _target) in &combos {
                let mut act = vec![0.0f32; n];
                recurrent_tick(&mut act, &thermo(a, thermo_bits), &w, &bias);
                recurrent_tick(&mut act, &thermo(b, thermo_bits), &w, &bias);
                recurrent_tick(&mut act, &thermo(c, thermo_bits), &w, &bias);
                let _sum: f32 = act.iter().sum();
                _score += 1; // dummy to prevent optimization
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        let evals_per_sec = n_evals as f64 / elapsed;

        let total_params = n * input_dim + n;
        let search_space = 5.0f64.powi(total_params as i32);

        println!("  {:<12} {:>8.0}K evals/sec  params={:<3} space=5^{}={:.1e}  full_sweep={:.1}h",
            label, evals_per_sec / 1e3, total_params, total_params, search_space,
            search_space / evals_per_sec / 3600.0);
    }

    // =========================================
    // BENCH 3: Modular network throughput
    // =========================================
    println!("\n--- BENCH 3: Modular network inference (N chips × ticks) ---\n");

    let n = 3;
    let thermo_bits = 4;
    let input_dim = n + thermo_bits;

    let w: Vec<Vec<f32>> = (0..n).map(|_| (0..input_dim).map(|_| rng.gen_range(-2.0..2.0f32)).collect()).collect();
    let bias: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0..2.0f32)).collect();

    for &n_chips in &[100, 1000, 10_000, 33_000, 100_000, 333_000, 1_000_000] {
        let t_input = thermo(2, thermo_bits);

        let start = Instant::now();
        let ticks = 100;
        for _ in 0..ticks {
            for _ in 0..n_chips {
                let mut act = vec![0.0f32; n];
                recurrent_tick(&mut act, &t_input, &w, &bias);
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        let total_neurons = n_chips * n;
        let ticks_per_sec = ticks as f64 / elapsed;

        println!("  {:>7} chips ({:>7} neurons): {:>8.1} ticks/sec  ({:.1} ms/tick)",
            n_chips, total_neurons, ticks_per_sec, elapsed / ticks as f64 * 1e3);
    }

    // =========================================
    // BENCH 4: 12-core estimate
    // =========================================
    println!("\n--- BENCH 4: Estimated 12-core (Ryzen 9 3900X) throughput ---\n");

    // Measure single-core baseline
    let n_chips_test = 33_000;
    let t_input = thermo(2, thermo_bits);
    let start = Instant::now();
    for _ in 0..10 {
        for _ in 0..n_chips_test {
            let mut act = vec![0.0f32; n];
            recurrent_tick(&mut act, &t_input, &w, &bias);
        }
    }
    let single_core = start.elapsed().as_secs_f64() / 10.0;

    println!("  33K chips (100K neurons), single core: {:.1} ms/tick", single_core * 1e3);
    println!("  33K chips (100K neurons), 12 cores:    ~{:.1} ms/tick", single_core * 1e3 / 12.0);
    println!("  33K chips (100K neurons), 24 threads:  ~{:.1} ms/tick", single_core * 1e3 / 20.0);
    println!();
    println!("  333K chips (1M neurons), 12 cores:    ~{:.0} ms/tick", single_core * 10.0 * 1e3 / 12.0);
    println!("  3.3M chips (10M neurons), 12 cores:   ~{:.0} ms/tick", single_core * 100.0 * 1e3 / 12.0);
    println!();

    let tick_12core = single_core / 12.0;
    println!("  At 100K neurons, depth-10 inference:  ~{:.0} ms", tick_12core * 10.0 * 1e3);
    println!("  At 100K neurons, depth-100 inference: ~{:.0} ms", tick_12core * 100.0 * 1e3);
    println!("  At 1M neurons, depth-10 inference:    ~{:.0} ms", tick_12core * 10.0 * 10.0 * 1e3);
    println!("  At 1M neurons, depth-100 inference:   ~{:.0} ms", tick_12core * 100.0 * 10.0 * 1e3);

    // =========================================
    // BENCH 5: Training estimates
    // =========================================
    println!("\n--- BENCH 5: Training time estimates (12-core) ---\n");

    // From bench 2 extrapolation
    let eval_rate_3n = 50_000.0; // approx evals/sec/core for 3n chip
    let eval_rate_12c = eval_rate_3n * 12.0;

    println!("  3-neuron chip (24 params, 5^24 = 6e16 space):");
    println!("    3M random + 500K perturb: ~{:.0} sec", 3_500_000.0 / eval_rate_12c);
    println!();
    println!("  Wiring search (24 float params):");
    println!("    5M random + 1M perturb: ~{:.0} sec", 6_000_000.0 / eval_rate_12c);
    println!();
    println!("  100K neuron network (33K chips):");
    println!("    Each chip: ~{:.0} sec to train", 3_500_000.0 / eval_rate_12c);
    println!("    Sequential: 33K × {:.0}s = {:.0} hours", 3_500_000.0 / eval_rate_12c,
        33_000.0 * 3_500_000.0 / eval_rate_12c / 3600.0);
    println!("    But chips are INDEPENDENT → parallel on 12 cores");
    println!("    Parallel: ~{:.0} hours", 33_000.0 * 3_500_000.0 / eval_rate_12c / 3600.0 / 12.0);

    println!("\n=== DONE ===");
}
