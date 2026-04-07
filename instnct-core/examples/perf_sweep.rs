//! Sweep: tokens/sec across network sizes and sparsity levels.
//!
//! Produces a table of measured throughput for H × density combinations,
//! using the library's Network::propagate (CSR skip-inactive path).

use instnct_core::{Network, PropagationConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

const WARMUP_TOKENS: usize = 50;
const MEASURE_TOKENS: usize = 200;
const RUNS: usize = 5;

fn make_network(h: usize, density_pct_x10: usize, seed: u64) -> Network {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = Network::new(h);

    // Add edges to target density (density_pct_x10 = density% × 10, e.g. 5 = 0.5%)
    let target_edges = (h as u64 * h as u64 * density_pct_x10 as u64 / 1000) as usize;
    let mut added = 0;
    let mut attempts = 0;
    while added < target_edges && attempts < target_edges * 20 {
        let src = (rng.gen::<u64>() % h as u64) as u16;
        let tgt = (rng.gen::<u64>() % h as u64) as u16;
        if net.graph_mut().add_edge(src, tgt) {
            added += 1;
        }
        attempts += 1;
    }

    // Randomize parameters
    for i in 0..h {
        let s = net.spike_data_mut();
        s[i].threshold = ((rng.gen::<u64>() % 10) + 1) as u8;
        s[i].channel = (rng.gen::<u64>() % 8 + 1) as u8;
    }
    for i in 0..h {
        if rng.gen::<u64>() % 4 == 0 {
            net.polarity_mut()[i] = -1;
        }
    }

    net
}

fn make_input(h: usize, seed: u64) -> Vec<i32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut input = vec![0i32; h];
    // Sparse input: ~5% neurons activated
    for i in 0..h {
        if rng.gen::<u64>() % 20 == 0 {
            input[i] = 1;
        }
    }
    input
}

fn measure_tokens_per_sec(net: &mut Network, inputs: &[Vec<i32>], config: &PropagationConfig) -> (f64, f64) {
    let n_inputs = inputs.len();

    // Warmup
    for i in 0..WARMUP_TOKENS {
        net.propagate(&inputs[i % n_inputs], config).unwrap();
    }

    // Measure multiple runs
    let mut times_ns = Vec::with_capacity(RUNS);
    for run in 0..RUNS {
        net.reset();
        let start = Instant::now();
        for i in 0..MEASURE_TOKENS {
            net.propagate(&inputs[(run * MEASURE_TOKENS + i) % n_inputs], config).unwrap();
        }
        let elapsed = start.elapsed().as_nanos() as f64;
        times_ns.push(elapsed / MEASURE_TOKENS as f64);
    }

    times_ns.sort_by(f64::total_cmp);
    let median_ns = times_ns[RUNS / 2];
    let tok_per_sec = 1_000_000_000.0 / median_ns;

    // Measure fire rate
    net.reset();
    let mut total_fires = 0u64;
    let mut total_neurons = 0u64;
    for i in 0..50 {
        net.propagate(&inputs[i % n_inputs], config).unwrap();
        let act = net.activation();
        total_fires += act.iter().filter(|&&a| a != 0).count() as u64;
        total_neurons += act.len() as u64;
    }
    let fire_rate = total_fires as f64 / total_neurons as f64;

    (tok_per_sec, fire_rate)
}

fn main() {
    let config = PropagationConfig::default();

    println!("╔══════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  INSTNCT PERFORMANCE SWEEP — Network::propagate (CSR skip-inactive)             ║");
    println!("║  Compact types (i8/u8/u16) + AoS SpikeData                                     ║");
    println!("║  {} warmup + {} measured tokens × {} runs, median                        ║", WARMUP_TOKENS, MEASURE_TOKENS, RUNS);
    println!("╚══════════════════════════════════════════════════════════════════════════════════╝");

    println!("\n{:>6} {:>8} {:>8} {:>10} {:>10} {:>12} {:>8}",
        "H", "density", "edges", "ns/token", "tok/sec", "edge_data", "fire%");
    println!("{}", "─".repeat(78));

    let sizes = [256, 512, 1024, 2048, 4096, 8192];
    let densities_x10 = [5, 10, 20, 30, 50]; // 0.5%, 1%, 2%, 3%, 5%

    for &h in &sizes {
        for &d in &densities_x10 {
            // Skip combos that would be too slow or too many edges
            let est_edges = h as u64 * h as u64 * d as u64 / 1000;
            if est_edges > 2_000_000 {
                continue; // too many edges, would take forever
            }

            let mut net = make_network(h, d, 42);
            let edge_count = net.edge_count();
            let edge_kb = edge_count * 4 / 1024; // 2×u16 per edge

            // Generate multiple input patterns
            let inputs: Vec<Vec<i32>> = (0..20).map(|s| make_input(h, 100 + s)).collect();

            let (tok_sec, fire_rate) = measure_tokens_per_sec(&mut net, &inputs, &config);
            let ns_per_tok = 1_000_000_000.0 / tok_sec;

            let density_str = format!("{:.1}%", d as f64 / 10.0);

            println!("{:>6} {:>8} {:>8} {:>10.0} {:>10.0} {:>10} KB {:>6.1}%",
                h, density_str, edge_count, ns_per_tok, tok_sec, edge_kb, fire_rate * 100.0);
        }
        println!("{}", "─".repeat(78));
    }

    // Cache boundary analysis
    println!("\n  CACHE BOUNDARIES:");
    println!("  L1 = 32-64 KB | L2 = 256KB-2MB | L3 = 4-96 MB");
    println!("  Edge data = edges × 4 bytes (2×u16 in CSR targets = edges × 2 bytes)");
    println!();

    // Header for extrapolated hardware
    println!("  EXTRAPOLATED HARDWARE (multiply tok/sec above):");
    println!("  ┌─────────────────────┬────────────┐");
    println!("  │ Hardware            │ Multiplier │");
    println!("  ├─────────────────────┼────────────┤");
    println!("  │ This VM (Xeon 2.1G) │    1.0x    │");
    println!("  │ Pi 5 (A76 2.4G)     │   ~0.8x    │");
    println!("  │ AMD 7800X3D (4.5G)  │   ~3-5x    │");
    println!("  │ Apple M4 Pro (4.5G) │   ~3-4x    │");
    println!("  └─────────────────────┴────────────┘");
}
