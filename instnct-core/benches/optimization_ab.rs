//! A/B/C optimization benchmarks for the propagation hot path.
//!
//! Tests: A (edge sort), B (branchless spike), C (AVX2 auto-vectorize)
//! Run: cargo bench --bench optimization_ab --features benchmarks

mod common;

use common::{build_graph, print_harness_header, timed_run};
use instnct_core::__internal::propagate_token_unchecked;
use instnct_core::{
    ConnectionGraph, PropagationConfig, PropagationParameters, PropagationState,
    PropagationWorkspace,
};
use std::hint::black_box;
const PHASE_BASE: [u8; 8] = [7, 8, 10, 12, 13, 12, 10, 8];

fn build_bench_fixture(neuron_count: usize) -> (Vec<u32>, Vec<u8>, Vec<i32>, Vec<i32>) {
    let threshold = vec![6u32; neuron_count];
    let mut channel = vec![1u8; neuron_count];
    let mut polarity = vec![1i32; neuron_count];
    for i in 0..neuron_count {
        channel[i] = (i % 8 + 1) as u8;
        if i % 10 == 0 {
            polarity[i] = -1;
        }
    }
    let mut input = vec![0i32; neuron_count];
    input[0] = 1;
    (threshold, channel, polarity, input)
}

fn compare(label: &str, baseline: f64, candidate: f64, noise_pct: f64) {
    let delta_pct = (candidate - baseline) / baseline * 100.0;
    let sig = if delta_pct.abs() > noise_pct * 3.0 {
        "SIGNIFICANT"
    } else if delta_pct.abs() > noise_pct {
        "borderline"
    } else {
        "WITHIN NOISE"
    };
    println!(
        "    {label:40} {:+.1}%  ({sig}, noise floor={:.1}%)",
        delta_pct, noise_pct
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
fn propagate_inline(
    inputs: &BenchInputs<'_>,
    scratch: &mut BenchScratch,
    use_branchless_spike: bool,
) {
    let neuron_count = inputs.graph.neuron_count();
    let (edge_src, edge_tgt) = inputs.graph.edge_endpoints_pub();
    for tick in 0..inputs.config.ticks_per_token {
        if inputs.config.decay_interval_ticks > 0 && tick % inputs.config.decay_interval_ticks == 0
        {
            for charge in scratch.charge.iter_mut() {
                *charge = charge.saturating_sub(1);
            }
        }
        if tick < inputs.config.input_duration_ticks {
            for (act, &input_val) in scratch.activation.iter_mut().zip(inputs.input.iter()) {
                *act += input_val;
            }
        }
        let incoming = &mut scratch.incoming[..neuron_count];
        incoming.fill(0);
        for (src_chunk, tgt_chunk) in edge_src.chunks_exact(4).zip(edge_tgt.chunks_exact(4)) {
            incoming[tgt_chunk[0]] += scratch.activation[src_chunk[0]];
            incoming[tgt_chunk[1]] += scratch.activation[src_chunk[1]];
            incoming[tgt_chunk[2]] += scratch.activation[src_chunk[2]];
            incoming[tgt_chunk[3]] += scratch.activation[src_chunk[3]];
        }
        for i in (edge_src.len() / 4 * 4)..edge_src.len() {
            incoming[edge_tgt[i]] += scratch.activation[edge_src[i]];
        }
        for (charge, &signal) in scratch.charge.iter_mut().zip(incoming.iter()) {
            *charge = charge.saturating_add_signed(signal).min(15);
        }
        let phase_tick = tick % 8;
        if use_branchless_spike {
            for i in 0..neuron_count {
                let channel_idx = inputs.channel[i] as usize;
                let phase_mult: u16 = if (1..=8).contains(&channel_idx) {
                    PHASE_BASE[(phase_tick + 9 - channel_idx) & 7] as u16
                } else {
                    10
                };
                let fires =
                    scratch.charge[i] as u16 * 10 >= (inputs.threshold[i] as u16 + 1) * phase_mult;
                scratch.activation[i] =
                    core::hint::select_unpredictable(fires, inputs.polarity[i], 0);
                scratch.charge[i] = core::hint::select_unpredictable(fires, 0, scratch.charge[i]);
            }
        } else {
            for i in 0..neuron_count {
                let channel_idx = inputs.channel[i] as usize;
                let phase_mult: u16 = if (1..=8).contains(&channel_idx) {
                    PHASE_BASE[(phase_tick + 9 - channel_idx) & 7] as u16
                } else {
                    10
                };
                if scratch.charge[i] as u16 * 10 >= (inputs.threshold[i] as u16 + 1) * phase_mult {
                    scratch.activation[i] = inputs.polarity[i];
                    scratch.charge[i] = 0;
                } else {
                    scratch.activation[i] = 0;
                }
            }
        }
    }
}

// C: AVX2 variant — IDENTICAL logic to propagate_inline(use_branchless_spike=false),
// only difference is #[target_feature] letting LLVM use 256-bit SIMD.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn propagate_avx2(inputs: &BenchInputs<'_>, scratch: &mut BenchScratch) {
    let neuron_count = inputs.graph.neuron_count();
    let (edge_src, edge_tgt) = inputs.graph.edge_endpoints_pub();
    for tick in 0..inputs.config.ticks_per_token {
        if inputs.config.decay_interval_ticks > 0 && tick % inputs.config.decay_interval_ticks == 0
        {
            for charge in scratch.charge.iter_mut() {
                *charge = charge.saturating_sub(1);
            }
        }
        if tick < inputs.config.input_duration_ticks {
            for (act, &input_val) in scratch.activation.iter_mut().zip(inputs.input.iter()) {
                *act += input_val;
            }
        }
        let incoming = &mut scratch.incoming[..neuron_count];
        incoming.fill(0);
        for (src_chunk, tgt_chunk) in edge_src.chunks_exact(4).zip(edge_tgt.chunks_exact(4)) {
            incoming[tgt_chunk[0]] += scratch.activation[src_chunk[0]];
            incoming[tgt_chunk[1]] += scratch.activation[src_chunk[1]];
            incoming[tgt_chunk[2]] += scratch.activation[src_chunk[2]];
            incoming[tgt_chunk[3]] += scratch.activation[src_chunk[3]];
        }
        for i in (edge_src.len() / 4 * 4)..edge_src.len() {
            incoming[edge_tgt[i]] += scratch.activation[edge_src[i]];
        }
        for (charge, &signal) in scratch.charge.iter_mut().zip(incoming.iter()) {
            *charge = charge.saturating_add_signed(signal).min(15);
        }
        // SAME if/else spike logic as propagate_inline(use_branchless_spike=false)
        let phase_tick = tick % 8;
        for i in 0..neuron_count {
            let channel_idx = inputs.channel[i] as usize;
            let phase_mult: u16 = if (1..=8).contains(&channel_idx) {
                PHASE_BASE[(phase_tick + 9 - channel_idx) & 7] as u16
            } else {
                10
            };
            if scratch.charge[i] as u16 * 10 >= (inputs.threshold[i] as u16 + 1) * phase_mult {
                scratch.activation[i] = inputs.polarity[i];
                scratch.charge[i] = 0;
            } else {
                scratch.activation[i] = 0;
            }
        }
    }
}

fn main() {
    print_harness_header();

    let config = PropagationConfig {
        ticks_per_token: 12,
        input_duration_ticks: 2,
        decay_interval_ticks: 6,
    };

    for &(neuron_count, edge_prob_pct, iterations) in &[
        (256usize, 5u64, 2000usize),
        (1024, 2, 2000),
        (2048, 1, 500),
        (4096, 1, 100),
    ] {
        println!("\n=== H={neuron_count}, {edge_prob_pct}% density, {iterations} iterations ===");
        let graph = build_graph(neuron_count, edge_prob_pct);
        let (threshold, channel, polarity, input) = build_bench_fixture(neuron_count);
        let inputs = BenchInputs {
            input: &input,
            graph: &graph,
            threshold: &threshold,
            channel: &channel,
            polarity: &polarity,
            config: &config,
        };
        println!("  edges: {}", graph.edge_count());

        // --- NOISE FLOOR: scalar vs scalar (identical code, separate runs) ---
        let ctrl_a;
        let ctrl_b;
        {
            let mut scratch = BenchScratch::new(neuron_count);
            ctrl_a = timed_run("CTRL-1 (scalar #1)", iterations, || {
                scratch.reset();
                propagate_inline(black_box(&inputs), black_box(&mut scratch), false);
            });
        }
        {
            let mut scratch = BenchScratch::new(neuron_count);
            ctrl_b = timed_run("CTRL-2 (scalar #2)", iterations, || {
                scratch.reset();
                propagate_inline(black_box(&inputs), black_box(&mut scratch), false);
            });
        }
        let noise_pct = ((ctrl_b.median_ns - ctrl_a.median_ns) / ctrl_a.median_ns * 100.0).abs();
        println!("    --> NOISE FLOOR: {noise_pct:.1}%\n");

        // A: Edge sort by target
        let mut graph_sorted = graph.clone();
        graph_sorted.sort_edges_by_target();
        let a_unsorted;
        {
            let mut activation = vec![0i32; neuron_count];
            let mut charge = vec![0u32; neuron_count];
            let mut workspace = PropagationWorkspace::new(neuron_count);
            a_unsorted = timed_run("A-baseline (unsorted)", iterations, || {
                activation.fill(0);
                charge.fill(0);
                propagate_token_unchecked(
                    black_box(&input),
                    black_box(&graph),
                    black_box(&PropagationParameters {
                        threshold: &threshold,
                        channel: &channel,
                        polarity: &polarity,
                    }),
                    black_box(&mut PropagationState {
                        activation: &mut activation,
                        charge: &mut charge,
                    }),
                    black_box(&config),
                    black_box(&mut workspace),
                );
            });
        }
        let a_sorted;
        {
            let mut activation = vec![0i32; neuron_count];
            let mut charge = vec![0u32; neuron_count];
            let mut workspace = PropagationWorkspace::new(neuron_count);
            a_sorted = timed_run("A-sorted (edges by target)", iterations, || {
                activation.fill(0);
                charge.fill(0);
                propagate_token_unchecked(
                    black_box(&input),
                    black_box(&graph_sorted),
                    black_box(&PropagationParameters {
                        threshold: &threshold,
                        channel: &channel,
                        polarity: &polarity,
                    }),
                    black_box(&mut PropagationState {
                        activation: &mut activation,
                        charge: &mut charge,
                    }),
                    black_box(&config),
                    black_box(&mut workspace),
                );
            });
        }
        compare(
            "sorted vs unsorted",
            a_unsorted.median_ns,
            a_sorted.median_ns,
            noise_pct,
        );
        println!();

        // B: Branchless spike
        let b_branching;
        {
            let mut scratch = BenchScratch::new(neuron_count);
            b_branching = timed_run("B-branching (if/else spike)", iterations, || {
                scratch.reset();
                propagate_inline(black_box(&inputs), black_box(&mut scratch), false);
            });
        }
        let b_branchless;
        {
            let mut scratch = BenchScratch::new(neuron_count);
            b_branchless = timed_run("B-branchless (select_unpredictable)", iterations, || {
                scratch.reset();
                propagate_inline(black_box(&inputs), black_box(&mut scratch), true);
            });
        }
        compare(
            "branchless vs branching",
            b_branching.median_ns,
            b_branchless.median_ns,
            noise_pct,
        );
        println!();

        // C: AVX2 — SAME if/else logic, only target_feature differs
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            let c_scalar;
            let c_avx2;
            {
                let mut scratch = BenchScratch::new(neuron_count);
                c_scalar = timed_run("C-scalar (if/else, no target_feature)", iterations, || {
                    scratch.reset();
                    propagate_inline(black_box(&inputs), black_box(&mut scratch), false);
                });
            }
            {
                let mut scratch = BenchScratch::new(neuron_count);
                c_avx2 = timed_run(
                    "C-avx2   (if/else, target_feature avx2)",
                    iterations,
                    || {
                        scratch.reset();
                        unsafe {
                            propagate_avx2(black_box(&inputs), black_box(&mut scratch));
                        }
                    },
                );
            }
            compare(
                "avx2 vs scalar",
                c_scalar.median_ns,
                c_avx2.median_ns,
                noise_pct,
            );
        }

        println!();
    }
}
