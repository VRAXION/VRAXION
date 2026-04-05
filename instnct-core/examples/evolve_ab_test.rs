//! A/B test: compare mutation strategies across multiple seeds.
//!
//! Each strategy gets the same seeds, same network size, same number of
//! evaluations. Only the mutation policy differs.
//!
//! Run: cargo run --example evolve_ab_test --release

use instnct_core::{Network, PropagationConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const NEURON_COUNT: usize = 64;
const NUM_TOKENS: usize = 16;
const EVALUATIONS: usize = 500;
const SEEDS: [u64; 5] = [42, 123, 7, 999, 2026];

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

struct RunResult {
    score: u32,
    edges: usize,
    accepted: u32,
    rejected: u32,
    converged_at: usize,
}

fn evolve(
    seed: u64,
    mut pick_mutation: impl FnMut(&mut Network, &mut StdRng) -> bool,
) -> RunResult {
    let config = PropagationConfig::default();
    let mut net = Network::new(NEURON_COUNT);
    let mut rng = StdRng::seed_from_u64(seed);

    let mut best_score = evaluate(&mut net, &config);
    let mut accepted = 0u32;
    let mut rejected = 0u32;
    let mut converged_at = 0usize;

    for step in 0..EVALUATIONS {
        let snapshot = net.save_state();

        if !pick_mutation(&mut net, &mut rng) {
            continue;
        }

        let score = evaluate(&mut net, &config);
        if score >= best_score {
            if score > best_score {
                converged_at = step;
            }
            best_score = score;
            accepted += 1;
        } else {
            net.restore_state(&snapshot);
            rejected += 1;
        }
    }

    RunResult {
        score: best_score,
        edges: net.edge_count(),
        accepted,
        rejected,
        converged_at,
    }
}

// ---- Strategies ----

fn strategy_simple(net: &mut Network, rng: &mut StdRng) -> bool {
    if net.edge_count() == 0 || rng.gen_ratio(7, 10) {
        net.mutate_add_edge(rng)
    } else {
        net.mutate_remove_edge(rng)
    }
}

fn strategy_with_rewire(net: &mut Network, rng: &mut StdRng) -> bool {
    let roll = rng.gen_range(0..100u32);
    if net.edge_count() == 0 || roll < 50 {
        net.mutate_add_edge(rng)
    } else if roll < 70 {
        net.mutate_remove_edge(rng)
    } else {
        net.mutate_rewire(rng)
    }
}

fn strategy_full_phased(net: &mut Network, rng: &mut StdRng) -> bool {
    let build_phase = net.edge_count() < NEURON_COUNT;
    let roll = rng.gen_range(0..100u32);
    if build_phase {
        match roll {
            0..80 => net.mutate_add_edge(rng),
            80..90 => net.mutate_rewire(rng),
            _ => net.mutate_theta(rng),
        }
    } else {
        match roll {
            0..30 => net.mutate_add_edge(rng),
            30..45 => net.mutate_remove_edge(rng),
            45..60 => net.mutate_rewire(rng),
            60..75 => net.mutate_theta(rng),
            75..90 => net.mutate_channel(rng),
            _ => net.mutate_polarity(rng),
        }
    }
}

// ---- Reporting ----

struct StrategyResult {
    name: &'static str,
    runs: Vec<RunResult>,
}

impl StrategyResult {
    fn avg_score(&self) -> f64 {
        self.runs.iter().map(|r| r.score as f64).sum::<f64>() / self.runs.len() as f64
    }
    fn avg_edges(&self) -> f64 {
        self.runs.iter().map(|r| r.edges as f64).sum::<f64>() / self.runs.len() as f64
    }
    fn avg_accept_rate(&self) -> f64 {
        let total_acc: u32 = self.runs.iter().map(|r| r.accepted).sum();
        let total_all: u32 = self.runs.iter().map(|r| r.accepted + r.rejected).sum();
        if total_all == 0 {
            0.0
        } else {
            total_acc as f64 / total_all as f64 * 100.0
        }
    }
    fn avg_convergence(&self) -> f64 {
        self.runs.iter().map(|r| r.converged_at as f64).sum::<f64>() / self.runs.len() as f64
    }
    fn min_score(&self) -> u32 {
        self.runs.iter().map(|r| r.score).min().unwrap_or(0)
    }
}

fn run_strategy(
    name: &'static str,
    strategy: fn(&mut Network, &mut StdRng) -> bool,
) -> StrategyResult {
    let runs: Vec<RunResult> = SEEDS.iter().map(|&seed| evolve(seed, strategy)).collect();
    StrategyResult { name, runs }
}

fn main() {
    println!(
        "A/B Test: H={NEURON_COUNT}, {NUM_TOKENS} tokens, {EVALUATIONS} evals, {} seeds\n",
        SEEDS.len()
    );

    let strategies: Vec<StrategyResult> = vec![
        run_strategy("simple-70/30", strategy_simple),
        run_strategy("add+rem+rewire", strategy_with_rewire),
        run_strategy("full-phased", strategy_full_phased),
    ];

    println!(
        "  {:<20} {:>9} {:>9} {:>7} {:>9} {:>9}",
        "strategy", "avg_score", "min_score", "edges", "accept%", "converge"
    );
    println!(
        "  {:-<20} {:-<9} {:-<9} {:-<7} {:-<9} {:-<9}",
        "", "", "", "", "", ""
    );
    for s in &strategies {
        println!(
            "  {:<20} {:>5.1}/{:<3} {:>5}/{:<3} {:>7.0} {:>8.0}% {:>9.0}",
            s.name,
            s.avg_score(),
            NUM_TOKENS,
            s.min_score(),
            NUM_TOKENS,
            s.avg_edges(),
            s.avg_accept_rate(),
            s.avg_convergence()
        );
    }

    println!("\n  Per-seed detail:");
    for s in &strategies {
        print!("  {:<20}", s.name);
        for run in &s.runs {
            print!("  {:>2}/{}", run.score, NUM_TOKENS);
        }
        println!();
    }
}
