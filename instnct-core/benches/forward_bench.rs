mod common;

use common::{build_graph, print_harness_header, timed_run};
use instnct_core::__internal::propagate_token_unchecked;
use instnct_core::{
    ConnectionGraph, PropagationConfig, PropagationParameters, PropagationState,
    PropagationWorkspace,
};
use std::hint::black_box;

struct UniformFixture {
    threshold: Vec<u8>,
    channel: Vec<u8>,
    polarity: Vec<i8>,
    input: Vec<i32>,
}

struct ForwardCase {
    name: &'static str,
    neuron_count: usize,
    edge_probability_percent: u64,
    iterations: usize,
    config: PropagationConfig,
}

struct ForwardInputs<'a> {
    input: &'a [i32],
    graph: &'a ConnectionGraph,
    threshold: &'a [u8],
    channel: &'a [u8],
    polarity: &'a [i8],
    config: &'a PropagationConfig,
}

struct ForwardScratch {
    activation: Vec<i8>,
    charge: Vec<u8>,
    refractory: Vec<u8>,
    workspace: PropagationWorkspace,
}

impl ForwardScratch {
    fn new(neuron_count: usize) -> Self {
        Self {
            activation: vec![0; neuron_count],
            charge: vec![0; neuron_count],
            refractory: vec![0; neuron_count],
            workspace: PropagationWorkspace::new(neuron_count),
        }
    }

    fn reset(&mut self) {
        self.activation.fill(0);
        self.charge.fill(0);
        self.refractory.fill(0);
    }
}

fn build_uniform_fixture(neuron_count: usize) -> UniformFixture {
    let mut input = vec![0i32; neuron_count];
    if let Some(first) = input.first_mut() {
        *first = 1;
    }

    UniformFixture {
        threshold: vec![6u8; neuron_count],
        channel: vec![1u8; neuron_count],
        polarity: vec![1i8; neuron_count],
        input,
    }
}

fn describe_noise_floor(noise_pct: f64) -> &'static str {
    if noise_pct <= 5.0 {
        "stable"
    } else if noise_pct <= 10.0 {
        "borderline"
    } else {
        "noisy; not suitable for strong perf claims"
    }
}

fn run_forward(inputs: &ForwardInputs<'_>, scratch: &mut ForwardScratch) {
    scratch.reset();

    let params = PropagationParameters {
        threshold: inputs.threshold,
        channel: inputs.channel,
        polarity: inputs.polarity,
    };
    let mut state = PropagationState {
        activation: &mut scratch.activation,
        charge: &mut scratch.charge,
        refractory: &mut scratch.refractory,
    };

    propagate_token_unchecked(
        black_box(inputs.input),
        black_box(inputs.graph),
        black_box(&params),
        black_box(&mut state),
        black_box(inputs.config),
        black_box(&mut scratch.workspace),
    );
}

fn bench_case(case: &ForwardCase) {
    println!(
        "\n=== {} | H={}, {}% density, {} iterations ===",
        case.name, case.neuron_count, case.edge_probability_percent, case.iterations
    );

    let graph = build_graph(case.neuron_count, case.edge_probability_percent);
    let UniformFixture {
        threshold,
        channel,
        polarity,
        input,
    } = build_uniform_fixture(case.neuron_count);
    let inputs = ForwardInputs {
        input: &input,
        graph: &graph,
        threshold: &threshold,
        channel: &channel,
        polarity: &polarity,
        config: &case.config,
    };

    println!("  edges: {}", graph.edge_count());

    let ctrl_a = {
        let mut scratch = ForwardScratch::new(case.neuron_count);
        timed_run("CTRL-1 (scalar #1)", case.iterations, || {
            run_forward(black_box(&inputs), black_box(&mut scratch));
        })
    };
    let ctrl_b = {
        let mut scratch = ForwardScratch::new(case.neuron_count);
        timed_run("CTRL-2 (scalar #2)", case.iterations, || {
            run_forward(black_box(&inputs), black_box(&mut scratch));
        })
    };
    let noise_pct = ((ctrl_b.median_ns - ctrl_a.median_ns) / ctrl_a.median_ns * 100.0).abs();
    println!(
        "    --> NOISE FLOOR: {noise_pct:.1}% ({})",
        describe_noise_floor(noise_pct)
    );

    let throughput = {
        let mut scratch = ForwardScratch::new(case.neuron_count);
        timed_run(case.name, case.iterations, || {
            run_forward(black_box(&inputs), black_box(&mut scratch));
        })
    };

    if noise_pct > 10.0 {
        println!(
            "    --> {} measured at {:.0} ns/iter, but this run is too noisy for strong claims",
            case.name, throughput.median_ns
        );
    }
}

fn main() {
    print_harness_header();

    let cases = [
        ForwardCase {
            name: "propagate_h256_12ticks_i32",
            neuron_count: 256,
            edge_probability_percent: 5,
            iterations: 2_000,
            config: PropagationConfig {
                ticks_per_token: 12,
                input_duration_ticks: 2,
                decay_interval_ticks: 6,
                use_refractory: false,
            },
        },
        ForwardCase {
            name: "propagate_h1024_16ticks_i32",
            neuron_count: 1024,
            edge_probability_percent: 2,
            iterations: 2_000,
            config: PropagationConfig {
                ticks_per_token: 16,
                input_duration_ticks: 2,
                decay_interval_ticks: 6,
                use_refractory: false,
            },
        },
    ];

    for case in &cases {
        bench_case(case);
    }
}
