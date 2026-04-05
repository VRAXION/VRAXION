//! Quick size sweep: how fast is propagate at different network sizes?
//!
//! Run: cargo run --example size_sweep --release

use instnct_core::{Network, PropagationConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn main() {
    let config = PropagationConfig::default();
    let sizes = [64, 128, 256, 512, 1024, 2048, 4096];
    let density_pct = 5u64;
    let eval_tokens = 26; // alphabet
    let iters = 100;

    println!(
        "{:>6} {:>8} {:>12} {:>12} {:>12} {:>10}",
        "H", "edges", "ns/token", "ns/eval(26)", "evals/sec", "tokens/sec"
    );
    println!("{:-<6} {:-<8} {:-<12} {:-<12} {:-<12} {:-<10}", "", "", "", "", "", "");

    for &neuron_count in &sizes {
        let mut net = Network::new(neuron_count);

        // Build random graph at target density
        let mut rng = StdRng::seed_from_u64(42);
        let target_edges = (neuron_count as u64 * neuron_count as u64 * density_pct / 100) as usize;
        for _ in 0..target_edges * 3 {
            // overshoot attempts for self-loop/dup rejects
            net.mutate_add_edge(&mut rng);
            if net.edge_count() >= target_edges {
                break;
            }
        }

        // Randomize params
        for i in 0..neuron_count {
            net.threshold_mut()[i] = rng.gen_range(0..=15);
            net.channel_mut()[i] = rng.gen_range(1..=8);
            if rng.gen_ratio(1, 10) {
                net.polarity_mut()[i] = -1;
            }
        }

        // Benchmark: eval_tokens propagations = 1 evaluation
        let mut total_ns = 0u64;
        for _ in 0..iters {
            net.reset();
            let start = Instant::now();
            for token in 0..eval_tokens {
                let mut input = vec![0i32; neuron_count];
                input[token % neuron_count] = 1;
                net.propagate(&input, &config).unwrap();
            }
            total_ns += start.elapsed().as_nanos() as u64;
        }

        let ns_per_eval = total_ns / iters as u64;
        let ns_per_token = ns_per_eval / eval_tokens as u64;
        let evals_per_sec = 1_000_000_000u64 / ns_per_eval.max(1);
        let tokens_per_sec = 1_000_000_000u64 / ns_per_token.max(1);

        println!(
            "{:>6} {:>8} {:>10} ns {:>10} ns {:>10}/s {:>8}/s",
            neuron_count,
            net.edge_count(),
            ns_per_token,
            ns_per_eval,
            evals_per_sec,
            tokens_per_sec
        );
    }

    println!("\nContext: 1 eval = {eval_tokens} tokens (alphabet). Python H=256 ~ 400K ns/token.");
}
