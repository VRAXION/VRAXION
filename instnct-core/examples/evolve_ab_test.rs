//! A/B test: compare mutation strategies on identical networks.
//!
//! Each strategy gets the same seed, same network size, same number of
//! evaluations. Only the mutation policy differs.
//!
//! Run: cargo run --example evolve_ab_test --release

use instnct_core::{Network, PropagationConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const NEURON_COUNT: usize = 64;
const NUM_TOKENS: usize = 16;
const EVALUATIONS: usize = 500;
const SEED: u64 = 42;

// ---- Evaluation ----

fn evaluate(net: &mut Network, config: &PropagationConfig) -> u32 {
    let neuron_count = net.neuron_count();
    let mut score = 0u32;
    let mut prev_pattern: Vec<i32> = vec![];

    for token in 0..NUM_TOKENS {
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

// ---- Strategies ----

struct RunResult {
    name: &'static str,
    scores: Vec<u32>,  // score at each eval step
    edges: Vec<usize>, // edge count at each eval step
    accepted: u32,
    rejected: u32,
}

/// Strategy A: 1 mutation per evaluation, fixed 70/30 add/remove ratio.
fn run_simple(add_pct: u32) -> RunResult {
    let config = PropagationConfig::default();
    let mut net = Network::new(NEURON_COUNT);
    let mut rng = StdRng::seed_from_u64(SEED);

    let mut best_score = evaluate(&mut net, &config);
    let mut accepted = 0u32;
    let mut rejected = 0u32;
    let mut scores = vec![best_score];
    let mut edges = vec![0usize];

    for _ in 0..EVALUATIONS {
        let snapshot = net.save_state();

        let mutated = if net.edge_count() == 0 || rng.gen_range(0..100) < add_pct {
            net.mutate_add_edge(&mut rng)
        } else {
            net.mutate_remove_edge(&mut rng)
        };
        if !mutated {
            scores.push(best_score);
            edges.push(net.edge_count());
            continue;
        }

        let score = evaluate(&mut net, &config);
        if score >= best_score {
            best_score = score;
            accepted += 1;
        } else {
            net.restore_state(&snapshot);
            rejected += 1;
        }
        scores.push(best_score);
        edges.push(net.edge_count());
    }

    RunResult {
        name: match add_pct {
            50 => "simple-50/50",
            70 => "simple-70/30",
            90 => "simple-90/10",
            _ => "simple-??/??",
        },
        scores,
        edges,
        accepted,
        rejected,
    }
}

/// Strategy B: add + remove + rewire (50/20/30 split).
fn run_with_rewire() -> RunResult {
    let config = PropagationConfig::default();
    let mut net = Network::new(NEURON_COUNT);
    let mut rng = StdRng::seed_from_u64(SEED);

    let mut best_score = evaluate(&mut net, &config);
    let mut accepted = 0u32;
    let mut rejected = 0u32;
    let mut scores = vec![best_score];
    let mut edges = vec![0usize];

    for _ in 0..EVALUATIONS {
        let snapshot = net.save_state();

        let roll = rng.gen_range(0..100u32);
        let mutated = if net.edge_count() == 0 || roll < 50 {
            net.mutate_add_edge(&mut rng)
        } else if roll < 70 {
            net.mutate_remove_edge(&mut rng)
        } else {
            net.mutate_rewire(&mut rng)
        };
        if !mutated {
            scores.push(best_score);
            edges.push(net.edge_count());
            continue;
        }

        let score = evaluate(&mut net, &config);
        if score >= best_score {
            best_score = score;
            accepted += 1;
        } else {
            net.restore_state(&snapshot);
            rejected += 1;
        }
        scores.push(best_score);
        edges.push(net.edge_count());
    }

    RunResult {
        name: "add+rem+rewire",
        scores,
        edges,
        accepted,
        rejected,
    }
}

/// Strategy C: adaptive burst — N mutations per evaluation, N adapts.
fn run_adaptive_burst() -> RunResult {
    let config = PropagationConfig::default();
    let mut net = Network::new(NEURON_COUNT);
    let mut rng = StdRng::seed_from_u64(SEED);

    let mut best_score = evaluate(&mut net, &config);
    let mut accepted = 0u32;
    let mut rejected = 0u32;
    let mut scores = vec![best_score];
    let mut edges = vec![0usize];

    // Burst controller state
    let mut burst: f64 = 1.0;
    let mut accept_ema: f64 = 0.5; // start neutral
    let mut rounds_since_improve: u32 = 0;

    for _ in 0..EVALUATIONS {
        let snapshot = net.save_state();

        // Apply burst mutations
        let burst_size = (burst as usize).max(1);
        let mut any_mutated = false;
        for _ in 0..burst_size {
            let mutated = if net.edge_count() == 0 || rng.gen_ratio(7, 10) {
                net.mutate_add_edge(&mut rng)
            } else {
                net.mutate_remove_edge(&mut rng)
            };
            if mutated {
                any_mutated = true;
            }
        }

        if !any_mutated {
            scores.push(best_score);
            edges.push(net.edge_count());
            continue;
        }

        // Evaluate
        let score = evaluate(&mut net, &config);
        let improved = score > best_score;

        if score >= best_score {
            best_score = score;
            accepted += 1;
            rounds_since_improve = 0;
        } else {
            net.restore_state(&snapshot);
            rejected += 1;
            rounds_since_improve += 1;
        }

        // Update burst controller (GPT 3-signal policy)
        accept_ema = 0.9 * accept_ema + 0.1 * if improved { 1.0 } else { 0.0 };
        let accuracy = best_score as f64 / NUM_TOKENS as f64;
        let accuracy_cap = 1.0 + (1.0 - accuracy) * 7.0; // max burst scales with remaining slack

        if accept_ema < 0.10 {
            burst = (burst / 2.0).max(1.0); // too aggressive, slow down
        } else if accept_ema > 0.30 {
            burst = (burst + 1.0).min(accuracy_cap); // too cautious, speed up
        }

        if rounds_since_improve > 50 {
            burst = (burst * 2.0).max(2.0).min(accuracy_cap); // stagnation escape
        }

        scores.push(best_score);
        edges.push(net.edge_count());
    }

    RunResult {
        name: "adaptive-burst",
        scores,
        edges,
        accepted,
        rejected,
    }
}

// ---- Reporting ----

fn print_result(result: &RunResult) {
    let total = result.accepted + result.rejected;
    let rate = if total > 0 {
        result.accepted as f64 / total as f64 * 100.0
    } else {
        0.0
    };
    let final_score = result.scores.last().copied().unwrap_or(0);
    let final_edges = result.edges.last().copied().unwrap_or(0);

    // Find first eval where max score reached
    let max_score = *result.scores.iter().max().unwrap_or(&0);
    let converged_at = result
        .scores
        .iter()
        .position(|&s| s == max_score)
        .unwrap_or(0);

    println!(
        "  {:<20} score={:>2}/{}  edges={:>3}  accept={:.0}%  converged_at=eval#{}",
        result.name, final_score, NUM_TOKENS, final_edges, rate, converged_at
    );
}

fn print_timeline(result: &RunResult) {
    print!("  {:<20} ", result.name);
    for (i, &score) in result.scores.iter().enumerate() {
        if i % 100 == 0 && i > 0 {
            print!(" {:>2}", score);
        }
    }
    println!(" -> {:>2}", result.scores.last().unwrap_or(&0));
}

fn main() {
    println!("A/B Test: H={NEURON_COUNT}, {NUM_TOKENS} tokens, {EVALUATIONS} evals, seed={SEED}\n");

    let results = [
        run_simple(50),
        run_simple(70),
        run_simple(90),
        run_with_rewire(),
        run_adaptive_burst(),
    ];

    println!("=== Results ===\n");
    for result in &results {
        print_result(result);
    }

    println!("\n=== Score timeline (every 100 evals) ===\n");
    for result in &results {
        print_timeline(result);
    }
}
