//! A/B optimization benchmarks for the propagation hot path.
//!
//! Tests: A (edge sort by target), B (branchless spike via select_unpredictable)
//! Run: cargo bench --bench optimization_ab --features benchmarks

use instnct_core::__internal::propagate_token_unchecked;
use instnct_core::{
    ConnectionGraph, PropagationConfig, PropagationParameters, PropagationState,
    PropagationWorkspace,
};
use std::hint::black_box;
use std::time::Instant;

const WARMUP: usize = 100;
const PHASE_BASE: [u8; 8] = [7, 8, 10, 12, 13, 12, 10, 8]; // duplicated for inline bench

fn build_graph(neuron_count: usize, edge_prob_pct: u64) -> ConnectionGraph {
    let mut graph = ConnectionGraph::new(neuron_count);
    let mut rng: u64 = 42;
    for i in 0..neuron_count {
        for j in 0..neuron_count {
            if i == j { continue; }
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 100 < edge_prob_pct {
                graph.add_edge(i as u16, j as u16);
            }
        }
    }
    graph
}

fn build_params(neuron_count: usize) -> (Vec<u32>, Vec<u8>, Vec<i32>, Vec<i32>) {
    let threshold = vec![6u32; neuron_count];
    let mut channel = vec![1u8; neuron_count];
    let mut polarity = vec![1i32; neuron_count];
    for i in 0..neuron_count {
        channel[i] = (i % 8 + 1) as u8;
        if i % 10 == 0 { polarity[i] = -1; }
    }
    let mut input = vec![0i32; neuron_count];
    input[0] = 1;
    (threshold, channel, polarity, input)
}

fn timed_run(name: &str, iterations: usize, mut body: impl FnMut()) {
    for _ in 0..WARMUP { body(); } // warmup
    let mut times = Vec::new();
    for _ in 0..3 {
        let start = Instant::now();
        for _ in 0..iterations { body(); }
        times.push(start.elapsed().as_nanos() as f64 / iterations as f64);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("  {name:45} median={:>10.0} ns  min={:.0}  max={:.0}", times[1], times[0], times[2]);
}

// B: branchless spike using select_unpredictable (inlined propagation)
fn propagate_branchless(
    input: &[i32],
    graph: &ConnectionGraph,
    threshold: &[u32], channel: &[u8], polarity: &[i32],
    activation: &mut [i32], charge: &mut [u32],
    incoming: &mut [i32],
    config: &PropagationConfig,
) {
    let neuron_count = graph.neuron_count();
    let (edge_sources, edge_targets) = graph.edge_endpoints_pub();

    for tick in 0..config.ticks_per_token {
        if config.decay_interval_ticks > 0 && tick % config.decay_interval_ticks == 0 {
            for c in charge.iter_mut() { *c = c.saturating_sub(1); }
        }
        if tick < config.input_duration_ticks {
            for (a, &iv) in activation.iter_mut().zip(input.iter()) { *a += iv; }
        }
        let inc = &mut incoming[..neuron_count];
        inc.fill(0);
        for (sc, tc) in edge_sources.chunks_exact(4).zip(edge_targets.chunks_exact(4)) {
            inc[tc[0]] += activation[sc[0]];
            inc[tc[1]] += activation[sc[1]];
            inc[tc[2]] += activation[sc[2]];
            inc[tc[3]] += activation[sc[3]];
        }
        let rem = edge_sources.len() / 4 * 4;
        for i in rem..edge_sources.len() {
            inc[edge_targets[i]] += activation[edge_sources[i]];
        }
        for (c, &s) in charge.iter_mut().zip(inc.iter()) {
            *c = c.saturating_add_signed(s).min(15);
        }
        // BRANCHLESS spike using select_unpredictable
        let tip = tick % 8;
        for i in 0..neuron_count {
            let ch = channel[i] as usize;
            let pm: u16 = if (1..=8).contains(&ch) { PHASE_BASE[(tip + 9 - ch) & 7] as u16 } else { 10 };
            let fires = charge[i] as u16 * 10 >= (threshold[i] as u16 + 1) * pm;
            activation[i] = core::hint::select_unpredictable(fires, polarity[i], 0);
            charge[i] = core::hint::select_unpredictable(fires, 0, charge[i]);
        }
    }
}

fn main() {
    let config = PropagationConfig {
        ticks_per_token: 12,
        input_duration_ticks: 2,
        decay_interval_ticks: 6,
    };

    for &(h, prob, iters) in &[(256usize, 5u64, 2000usize), (1024, 2, 500), (2048, 1, 200), (4096, 1, 50)] {
        println!("\n=== H={h}, {prob}% density, {iters} iterations ===");

        let graph = build_graph(h, prob);
        let (threshold, channel, polarity, input) = build_params(h);
        println!("  edges: {}", graph.edge_count());

        // A: Edge sort
        let mut graph_sorted = graph.clone();
        graph_sorted.sort_edges_by_target();

        {
            let mut act = vec![0i32; h];
            let mut chg = vec![0u32; h];
            let mut ws = PropagationWorkspace::new(h);
            timed_run("A-baseline (unsorted)", iters, || {
                act.fill(0); chg.fill(0);
                propagate_token_unchecked(
                    black_box(&input), black_box(&graph),
                    black_box(&PropagationParameters { threshold: &threshold, channel: &channel, polarity: &polarity }),
                    black_box(&mut PropagationState { activation: &mut act, charge: &mut chg }),
                    black_box(&config), black_box(&mut ws),
                );
            });
        }
        {
            let mut act = vec![0i32; h];
            let mut chg = vec![0u32; h];
            let mut ws = PropagationWorkspace::new(h);
            timed_run("A-sorted (edges by target)", iters, || {
                act.fill(0); chg.fill(0);
                propagate_token_unchecked(
                    black_box(&input), black_box(&graph_sorted),
                    black_box(&PropagationParameters { threshold: &threshold, channel: &channel, polarity: &polarity }),
                    black_box(&mut PropagationState { activation: &mut act, charge: &mut chg }),
                    black_box(&config), black_box(&mut ws),
                );
            });
        }

        // B: Branchless spike (select_unpredictable)
        {
            let mut act = vec![0i32; h];
            let mut chg = vec![0u32; h];
            let mut inc = vec![0i32; h];
            timed_run("B-branching (current, if/else)", iters, || {
                act.fill(0); chg.fill(0);
                propagate_token_unchecked(
                    black_box(&input), black_box(&graph),
                    black_box(&PropagationParameters { threshold: &threshold, channel: &channel, polarity: &polarity }),
                    black_box(&mut PropagationState { activation: &mut act, charge: &mut chg }),
                    black_box(&config), black_box(&mut PropagationWorkspace::new(h)),
                );
            });
            timed_run("B-branchless (select_unpredictable)", iters, || {
                act.fill(0); chg.fill(0);
                propagate_branchless(
                    black_box(&input), black_box(&graph),
                    black_box(&threshold), black_box(&channel), black_box(&polarity),
                    black_box(&mut act), black_box(&mut chg), black_box(&mut inc),
                    black_box(&config),
                );
            });
        }

        println!();
    }
}
