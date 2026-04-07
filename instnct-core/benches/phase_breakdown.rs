//! Phase breakdown: measure where time is spent in the propagation loop.
//!
//! Splits the forward pass into 4 phases and measures each independently
//! to identify the actual bottleneck before optimizing blindly.

mod common;

use common::{build_graph, print_harness_header, timed_run};
use instnct_core::ConnectionGraph;
use std::hint::black_box;

const PHASE_BASE: [u8; 8] = [7, 8, 10, 12, 13, 12, 10, 8];
const MAX_CHARGE: u32 = 15;
const TICKS: usize = 12;
const INPUT_DUR: usize = 2;
const DECAY_INT: usize = 6;

struct Fixture {
    graph: ConnectionGraph,
    threshold: Vec<u8>,
    channel: Vec<u8>,
    polarity: Vec<i8>,
    input: Vec<i32>,
    n: usize,
}

struct Scratch {
    activation: Vec<i8>,
    charge: Vec<u8>,
    incoming: Vec<i16>,
}

impl Scratch {
    fn new(n: usize) -> Self {
        Self {
            activation: vec![0; n],
            charge: vec![0; n],
            incoming: vec![0; n],
        }
    }
    fn reset(&mut self) {
        self.activation.fill(0);
        self.charge.fill(0);
    }
}

fn build_fixture(neuron_count: usize, edge_prob_pct: u64) -> Fixture {
    let graph = build_graph(neuron_count, edge_prob_pct);
    let mut input = vec![0i32; neuron_count];
    if let Some(first) = input.first_mut() {
        *first = 1;
    }
    Fixture {
        graph,
        threshold: vec![6u8; neuron_count],
        channel: vec![1u8; neuron_count],
        polarity: vec![1i8; neuron_count],
        input,
        n: neuron_count,
    }
}

// ---------------------------------------------------------------------------
// Phase-isolated benchmarks: run ALL ticks but only ONE phase per bench
// ---------------------------------------------------------------------------

/// Phase 1: charge decay only
fn phase_decay(f: &Fixture, s: &mut Scratch) {
    for tick in 0..TICKS {
        if DECAY_INT > 0 && tick % DECAY_INT == 0 {
            for ch in s.charge.iter_mut() {
                *ch = ch.saturating_sub(1);
            }
        }
    }
}

/// Phase 2: scatter-add only (the edge loop)
fn phase_scatter(f: &Fixture, s: &mut Scratch) {
    let (edge_src, edge_tgt) = f.graph.edge_endpoints_pub();
    let n = f.n;
    for _tick in 0..TICKS {
        let incoming = &mut s.incoming[..n];
        incoming.fill(0i16);
        for (sc, tc) in edge_src.chunks_exact(4).zip(edge_tgt.chunks_exact(4)) {
            incoming[tc[0] as usize] += s.activation[sc[0] as usize] as i16;
            incoming[tc[1] as usize] += s.activation[sc[1] as usize] as i16;
            incoming[tc[2] as usize] += s.activation[sc[2] as usize] as i16;
            incoming[tc[3] as usize] += s.activation[sc[3] as usize] as i16;
        }
        let rem = edge_src.len() / 4 * 4;
        for i in rem..edge_src.len() {
            incoming[edge_tgt[i] as usize] += s.activation[edge_src[i] as usize] as i16;
        }
    }
}

/// Phase 3: charge accumulation only
fn phase_charge_accum(f: &Fixture, s: &mut Scratch) {
    let n = f.n;
    for _tick in 0..TICKS {
        for (ch, &sig) in s.charge[..n].iter_mut().zip(s.incoming[..n].iter()) {
            *ch = { let val = (*ch as i16) + sig; val.clamp(0, MAX_CHARGE as i16) as u8 };
        }
    }
}

/// Phase 4: spike decision only
fn phase_spike(f: &Fixture, s: &mut Scratch) {
    let n = f.n;
    for tick in 0..TICKS {
        let phase_tick = tick % 8;
        for idx in 0..n {
            let ch_idx = f.channel[idx] as usize;
            let pm: u16 = if (1..=8).contains(&ch_idx) {
                PHASE_BASE[(phase_tick + 9 - ch_idx) & 7] as u16
            } else {
                10
            };
            let charge_x10 = s.charge[idx] as u16 * 10;
            let thresh_x10 = (f.threshold[idx] as u16 + 1) * pm;
            if charge_x10 >= thresh_x10 {
                s.activation[idx] = f.polarity[idx];
                s.charge[idx] = 0;
            } else {
                s.activation[idx] = 0;
            }
        }
    }
}

/// Full pass (all phases together) for reference
fn full_pass(f: &Fixture, s: &mut Scratch) {
    let (edge_src, edge_tgt) = f.graph.edge_endpoints_pub();
    let n = f.n;

    for tick in 0..TICKS {
        if DECAY_INT > 0 && tick % DECAY_INT == 0 {
            for ch in s.charge.iter_mut() {
                *ch = ch.saturating_sub(1);
            }
        }
        if tick < INPUT_DUR {
            for (act, &inp) in s.activation.iter_mut().zip(f.input.iter()) {
                *act = act.saturating_add(inp as i8);
            }
        }

        let incoming = &mut s.incoming[..n];
        incoming.fill(0i16);
        for (sc, tc) in edge_src.chunks_exact(4).zip(edge_tgt.chunks_exact(4)) {
            incoming[tc[0] as usize] += s.activation[sc[0] as usize] as i16;
            incoming[tc[1] as usize] += s.activation[sc[1] as usize] as i16;
            incoming[tc[2] as usize] += s.activation[sc[2] as usize] as i16;
            incoming[tc[3] as usize] += s.activation[sc[3] as usize] as i16;
        }
        let rem = edge_src.len() / 4 * 4;
        for i in rem..edge_src.len() {
            incoming[edge_tgt[i] as usize] += s.activation[edge_src[i] as usize] as i16;
        }

        for (ch, &sig) in s.charge[..n].iter_mut().zip(incoming[..n].iter()) {
            *ch = { let val = (*ch as i16) + sig; val.clamp(0, MAX_CHARGE as i16) as u8 };
        }

        let phase_tick = tick % 8;
        for idx in 0..n {
            let ch_idx = f.channel[idx] as usize;
            let pm: u16 = if (1..=8).contains(&ch_idx) {
                PHASE_BASE[(phase_tick + 9 - ch_idx) & 7] as u16
            } else {
                10
            };
            let charge_x10 = s.charge[idx] as u16 * 10;
            let thresh_x10 = (f.threshold[idx] as u16 + 1) * pm;
            if charge_x10 >= thresh_x10 {
                s.activation[idx] = f.polarity[idx];
                s.charge[idx] = 0;
            } else {
                s.activation[idx] = 0;
            }
        }
    }
}

struct SizeCase {
    name: &'static str,
    neuron_count: usize,
    edge_prob_pct: u64,
    iterations: usize,
}

fn run_breakdown(case: &SizeCase) {
    println!(
        "\n=== {} | H={}, {}% density ===",
        case.name, case.neuron_count, case.edge_prob_pct
    );

    let f = build_fixture(case.neuron_count, case.edge_prob_pct);
    println!("  edges: {}", f.graph.edge_count());

    let full = {
        let mut s = Scratch::new(case.neuron_count);
        timed_run("FULL (all phases)", case.iterations, || {
            s.reset();
            full_pass(black_box(&f), black_box(&mut s));
        })
    };

    let t_decay = {
        let mut s = Scratch::new(case.neuron_count);
        timed_run("phase 1: charge decay", case.iterations, || {
            s.reset();
            phase_decay(black_box(&f), black_box(&mut s));
        })
    };

    let t_scatter = {
        let mut s = Scratch::new(case.neuron_count);
        timed_run("phase 2: scatter-add (edge loop)", case.iterations, || {
            s.reset();
            phase_scatter(black_box(&f), black_box(&mut s));
        })
    };

    let t_accum = {
        let mut s = Scratch::new(case.neuron_count);
        timed_run("phase 3: charge accumulation", case.iterations, || {
            s.reset();
            phase_charge_accum(black_box(&f), black_box(&mut s));
        })
    };

    let t_spike = {
        let mut s = Scratch::new(case.neuron_count);
        timed_run("phase 4: spike decision", case.iterations, || {
            s.reset();
            phase_spike(black_box(&f), black_box(&mut s));
        })
    };

    let sum = t_decay.median_ns + t_scatter.median_ns + t_accum.median_ns + t_spike.median_ns;

    println!("\n  TIME BUDGET (% of full pass):");
    println!("    full pass:          {:>10.0} ns", full.median_ns);
    println!(
        "    charge decay:       {:>10.0} ns  ({:>5.1}%)",
        t_decay.median_ns,
        t_decay.median_ns / full.median_ns * 100.0
    );
    println!(
        "    scatter-add:        {:>10.0} ns  ({:>5.1}%)",
        t_scatter.median_ns,
        t_scatter.median_ns / full.median_ns * 100.0
    );
    println!(
        "    charge accumulate:  {:>10.0} ns  ({:>5.1}%)",
        t_accum.median_ns,
        t_accum.median_ns / full.median_ns * 100.0
    );
    println!(
        "    spike decision:     {:>10.0} ns  ({:>5.1}%)",
        t_spike.median_ns,
        t_spike.median_ns / full.median_ns * 100.0
    );
    println!(
        "    sum of phases:      {:>10.0} ns  (overhead: {:+.1}%)",
        sum,
        (sum - full.median_ns) / full.median_ns * 100.0
    );
}

fn main() {
    print_harness_header();

    let cases = [
        SizeCase {
            name: "small",
            neuron_count: 256,
            edge_prob_pct: 5,
            iterations: 5_000,
        },
        SizeCase {
            name: "medium",
            neuron_count: 1024,
            edge_prob_pct: 3,
            iterations: 2_000,
        },
        SizeCase {
            name: "large",
            neuron_count: 4096,
            edge_prob_pct: 1,
            iterations: 500,
        },
    ];

    for case in &cases {
        run_breakdown(case);
    }
}
