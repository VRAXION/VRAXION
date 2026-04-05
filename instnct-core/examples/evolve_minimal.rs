//! Minimal evolution loop: add/remove edge + evaluate + accept/reject.
//!
//! Run: cargo run --example evolve_minimal
//! Release: cargo run --example evolve_minimal --release

use instnct_core::{Network, PropagationConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Simple evaluation: propagate N different inputs, count how many produce
/// distinct output patterns. More distinct outputs = better discrimination.
fn evaluate(net: &mut Network, config: &PropagationConfig, num_tokens: usize) -> u32 {
    let neuron_count = net.neuron_count();
    let mut score = 0u32;
    let mut prev_pattern: Vec<i32> = vec![];

    for token in 0..num_tokens {
        net.reset();
        let mut input = vec![0i32; neuron_count];
        input[token % neuron_count] = 1;
        net.propagate(&input, config).unwrap();

        let pattern = net.activation().to_vec();
        if pattern != prev_pattern {
            score += 1;
        }
        prev_pattern = pattern;
    }
    score
}

fn main() {
    let neuron_count = 64;
    let num_tokens = 16;
    let steps = 500;
    let seed = 42u64;

    let config = PropagationConfig::default();
    let mut net = Network::new(neuron_count);
    let mut rng = StdRng::seed_from_u64(seed);

    let mut best_score = evaluate(&mut net, &config, num_tokens);
    let mut accepted = 0u32;
    let mut rejected = 0u32;

    println!("Evolve: H={neuron_count}, {num_tokens} tokens, {steps} steps, seed={seed}");
    println!(
        "Initial: edges={}, score={best_score}/{num_tokens}\n",
        net.edge_count()
    );

    for step in 0..steps {
        let snapshot = net.save_state();

        // Mutation: 70% add, 30% remove (remove comes last in schedule logic)
        let mutated = if net.edge_count() == 0 || rng.gen_ratio(7, 10) {
            net.mutate_add_edge(&mut rng)
        } else {
            net.mutate_remove_edge(&mut rng)
        };
        if !mutated {
            continue;
        }

        let score = evaluate(&mut net, &config, num_tokens);

        if score >= best_score {
            best_score = score;
            accepted += 1;
        } else {
            net.restore_state(&snapshot);
            rejected += 1;
        }

        if (step + 1) % 100 == 0 {
            let total = accepted + rejected;
            let rate = if total > 0 {
                accepted as f64 / total as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "  step {:>3}: edges={:>3}  score={:>2}/{}  accepted={:<3} rejected={:<3} rate={:.0}%",
                step + 1, net.edge_count(), best_score, num_tokens, accepted, rejected, rate
            );
        }
    }

    println!(
        "\nFinal: edges={}  score={}/{}  accept_rate={:.0}%",
        net.edge_count(),
        best_score,
        num_tokens,
        accepted as f64 / (accepted + rejected).max(1) as f64 * 100.0
    );
}
