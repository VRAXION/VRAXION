//! A/B/C optimization benchmarks for the propagation hot path.
//!
//! Tests: A (edge sort), B (branchless spike), C (AVX2 auto-vectorize)
//! Run: cargo bench --bench optimization_ab --features benchmarks

use instnct_core::__internal::propagate_token_unchecked;
use instnct_core::{
    ConnectionGraph, PropagationConfig, PropagationParameters, PropagationState,
    PropagationWorkspace,
};
use std::hint::black_box;
use std::time::Instant;

const WARMUP: usize = 100;
const PHASE_BASE: [u8; 8] = [7, 8, 10, 12, 13, 12, 10, 8];

fn build_graph(neuron_count: usize, edge_prob_pct: u64) -> ConnectionGraph {
    let mut graph = ConnectionGraph::new(neuron_count);
    let mut rng: u64 = 42;
    for i in 0..neuron_count {
        for j in 0..neuron_count {
            if i == j {
                continue;
            }
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            if (rng >> 32) % 100 < edge_prob_pct {
                graph.add_edge(i as u16, j as u16);
            }
        }
    }
    graph
}

fn build_params(n: usize) -> (Vec<u32>, Vec<u8>, Vec<i32>, Vec<i32>) {
    let threshold = vec![6u32; n];
    let mut channel = vec![1u8; n];
    let mut polarity = vec![1i32; n];
    for i in 0..n {
        channel[i] = (i % 8 + 1) as u8;
        if i % 10 == 0 {
            polarity[i] = -1;
        }
    }
    let mut input = vec![0i32; n];
    input[0] = 1;
    (threshold, channel, polarity, input)
}

fn timed_run(name: &str, iters: usize, mut body: impl FnMut()) {
    for _ in 0..WARMUP {
        body();
    }
    let mut times = Vec::new();
    for _ in 0..3 {
        let start = Instant::now();
        for _ in 0..iters {
            body();
        }
        times.push(start.elapsed().as_nanos() as f64 / iters as f64);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "  {name:45} median={:>10.0} ns  min={:.0}  max={:.0}",
        times[1], times[0], times[2]
    );
}

struct BenchInputs<'a> {
    input: &'a [i32],
    graph: &'a ConnectionGraph,
    threshold: &'a [u32],
    channel: &'a [u8],
    polarity: &'a [i32],
    config: &'a PropagationConfig,
}

struct BenchScratch {
    activation: Vec<i32>,
    charge: Vec<u32>,
    incoming: Vec<i32>,
}

impl BenchScratch {
    fn new(neuron_count: usize) -> Self {
        Self {
            activation: vec![0; neuron_count],
            charge: vec![0; neuron_count],
            incoming: vec![0; neuron_count],
        }
    }

    fn reset(&mut self) {
        self.activation.fill(0);
        self.charge.fill(0);
    }
}

// Shared inline propagation — exact same logic as mod.rs, used for B and C tests.
// The ONLY difference between scalar/branchless/avx2 variants is the spike decision.
fn propagate_inline(inputs: &BenchInputs<'_>, scratch: &mut BenchScratch, branchless: bool) {
    let n = inputs.graph.neuron_count();
    let (edge_src, edge_tgt) = inputs.graph.edge_endpoints_pub();
    for tick in 0..inputs.config.ticks_per_token {
        if inputs.config.decay_interval_ticks > 0 && tick % inputs.config.decay_interval_ticks == 0
        {
            for c in scratch.charge.iter_mut() {
                *c = c.saturating_sub(1);
            }
        }
        if tick < inputs.config.input_duration_ticks {
            for (a, &iv) in scratch.activation.iter_mut().zip(inputs.input.iter()) {
                *a += iv;
            }
        }
        let inc = &mut scratch.incoming[..n];
        inc.fill(0);
        for (sc, tc) in edge_src.chunks_exact(4).zip(edge_tgt.chunks_exact(4)) {
            inc[tc[0]] += scratch.activation[sc[0]];
            inc[tc[1]] += scratch.activation[sc[1]];
            inc[tc[2]] += scratch.activation[sc[2]];
            inc[tc[3]] += scratch.activation[sc[3]];
        }
        for i in (edge_src.len() / 4 * 4)..edge_src.len() {
            inc[edge_tgt[i]] += scratch.activation[edge_src[i]];
        }
        for (c, &s) in scratch.charge.iter_mut().zip(inc.iter()) {
            *c = c.saturating_add_signed(s).min(15);
        }
        let tip = tick % 8;
        if branchless {
            for i in 0..n {
                let ch = inputs.channel[i] as usize;
                let pm: u16 = if (1..=8).contains(&ch) {
                    PHASE_BASE[(tip + 9 - ch) & 7] as u16
                } else {
                    10
                };
                let fires = scratch.charge[i] as u16 * 10 >= (inputs.threshold[i] as u16 + 1) * pm;
                scratch.activation[i] =
                    core::hint::select_unpredictable(fires, inputs.polarity[i], 0);
                scratch.charge[i] = core::hint::select_unpredictable(fires, 0, scratch.charge[i]);
            }
        } else {
            for i in 0..n {
                let ch = inputs.channel[i] as usize;
                let pm: u16 = if (1..=8).contains(&ch) {
                    PHASE_BASE[(tip + 9 - ch) & 7] as u16
                } else {
                    10
                };
                if scratch.charge[i] as u16 * 10 >= (inputs.threshold[i] as u16 + 1) * pm {
                    scratch.activation[i] = inputs.polarity[i];
                    scratch.charge[i] = 0;
                } else {
                    scratch.activation[i] = 0;
                }
            }
        }
    }
}

// C: AVX2 variant — IDENTICAL logic to propagate_inline(branchless=false),
// only difference is #[target_feature] letting LLVM use 256-bit SIMD.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn propagate_avx2(inputs: &BenchInputs<'_>, scratch: &mut BenchScratch) {
    let n = inputs.graph.neuron_count();
    let (edge_src, edge_tgt) = inputs.graph.edge_endpoints_pub();
    for tick in 0..inputs.config.ticks_per_token {
        if inputs.config.decay_interval_ticks > 0 && tick % inputs.config.decay_interval_ticks == 0
        {
            for c in scratch.charge.iter_mut() {
                *c = c.saturating_sub(1);
            }
        }
        if tick < inputs.config.input_duration_ticks {
            for (a, &iv) in scratch.activation.iter_mut().zip(inputs.input.iter()) {
                *a += iv;
            }
        }
        let inc = &mut scratch.incoming[..n];
        inc.fill(0);
        for (sc, tc) in edge_src.chunks_exact(4).zip(edge_tgt.chunks_exact(4)) {
            inc[tc[0]] += scratch.activation[sc[0]];
            inc[tc[1]] += scratch.activation[sc[1]];
            inc[tc[2]] += scratch.activation[sc[2]];
            inc[tc[3]] += scratch.activation[sc[3]];
        }
        for i in (edge_src.len() / 4 * 4)..edge_src.len() {
            inc[edge_tgt[i]] += scratch.activation[edge_src[i]];
        }
        for (c, &s) in scratch.charge.iter_mut().zip(inc.iter()) {
            *c = c.saturating_add_signed(s).min(15);
        }
        // SAME if/else spike logic as propagate_inline(branchless=false)
        let tip = tick % 8;
        for i in 0..n {
            let ch = inputs.channel[i] as usize;
            let pm: u16 = if (1..=8).contains(&ch) {
                PHASE_BASE[(tip + 9 - ch) & 7] as u16
            } else {
                10
            };
            if scratch.charge[i] as u16 * 10 >= (inputs.threshold[i] as u16 + 1) * pm {
                scratch.activation[i] = inputs.polarity[i];
                scratch.charge[i] = 0;
            } else {
                scratch.activation[i] = 0;
            }
        }
    }
}

fn main() {
    let config = PropagationConfig {
        ticks_per_token: 12,
        input_duration_ticks: 2,
        decay_interval_ticks: 6,
    };

    for &(h, prob, iters) in &[
        (256usize, 5u64, 2000usize),
        (1024, 2, 2000),
        (2048, 1, 500),
        (4096, 1, 100),
    ] {
        println!("\n=== H={h}, {prob}% density, {iters} iterations ===");
        let graph = build_graph(h, prob);
        let (threshold, channel, polarity, input) = build_params(h);
        let inputs = BenchInputs {
            input: &input,
            graph: &graph,
            threshold: &threshold,
            channel: &channel,
            polarity: &polarity,
            config: &config,
        };
        println!("  edges: {}", graph.edge_count());

        // A: Edge sort by target
        let mut graph_sorted = graph.clone();
        graph_sorted.sort_edges_by_target();
        {
            let mut act = vec![0i32; h];
            let mut chg = vec![0u32; h];
            let mut ws = PropagationWorkspace::new(h);
            timed_run("A-baseline (unsorted)", iters, || {
                act.fill(0);
                chg.fill(0);
                propagate_token_unchecked(
                    black_box(&input),
                    black_box(&graph),
                    black_box(&PropagationParameters {
                        threshold: &threshold,
                        channel: &channel,
                        polarity: &polarity,
                    }),
                    black_box(&mut PropagationState {
                        activation: &mut act,
                        charge: &mut chg,
                    }),
                    black_box(&config),
                    black_box(&mut ws),
                );
            });
        }
        {
            let mut act = vec![0i32; h];
            let mut chg = vec![0u32; h];
            let mut ws = PropagationWorkspace::new(h);
            timed_run("A-sorted (edges by target)", iters, || {
                act.fill(0);
                chg.fill(0);
                propagate_token_unchecked(
                    black_box(&input),
                    black_box(&graph_sorted),
                    black_box(&PropagationParameters {
                        threshold: &threshold,
                        channel: &channel,
                        polarity: &polarity,
                    }),
                    black_box(&mut PropagationState {
                        activation: &mut act,
                        charge: &mut chg,
                    }),
                    black_box(&config),
                    black_box(&mut ws),
                );
            });
        }

        // B: Branchless spike (both use same pre-allocated buffers)
        {
            let mut scratch = BenchScratch::new(h);
            timed_run("B-branching (if/else spike)", iters, || {
                scratch.reset();
                propagate_inline(black_box(&inputs), black_box(&mut scratch), false);
            });
        }
        {
            let mut scratch = BenchScratch::new(h);
            timed_run("B-branchless (select_unpredictable)", iters, || {
                scratch.reset();
                propagate_inline(black_box(&inputs), black_box(&mut scratch), true);
            });
        }

        // C: AVX2 — SAME if/else logic, only target_feature differs
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            let mut scratch = BenchScratch::new(h);
            timed_run("C-scalar (if/else, no target_feature)", iters, || {
                scratch.reset();
                propagate_inline(black_box(&inputs), black_box(&mut scratch), false);
            });
            timed_run("C-avx2   (if/else, target_feature avx2)", iters, || {
                scratch.reset();
                unsafe {
                    propagate_avx2(black_box(&inputs), black_box(&mut scratch));
                }
            });
        }

        println!();
    }
}
