//! A/B benchmark: edge index compression for scatter-add.
//!
//! Baseline uses two Vec<usize> (16 bytes/edge on 64-bit).
//! Variants:
//!   - u16-split: two Vec<u16> (4 bytes/edge) — 4x bandwidth reduction
//!   - packed-u32: one Vec<u32> with (src<<16)|tgt (4 bytes/edge, single stream)
//!
//! The scatter-add at H=4096 is bandwidth-bound on sequential edge reads
//! (64.5% of time). Smaller indices = less data to stream through.

mod common;

use common::{build_graph, print_harness_header, timed_run};
use std::hint::black_box;

const TICKS: usize = 12;
const MAX_CHARGE: u32 = 15;
const PHASE_BASE: [u8; 8] = [7, 8, 10, 12, 13, 12, 10, 8];

// ---------------------------------------------------------------------------
// Fixture: all three representations built from the same graph
// ---------------------------------------------------------------------------

struct Fixture {
    // Baseline (current library format)
    sources_usize: Vec<usize>,
    targets_usize: Vec<usize>,
    // u16 split
    sources_u16: Vec<u16>,
    targets_u16: Vec<u16>,
    // packed u32: (src << 16) | tgt
    packed_u32: Vec<u32>,

    // Neuron data (shared)
    activation: Vec<i32>,
    charge: Vec<u32>,
    incoming: Vec<i32>,
    threshold: Vec<u32>,
    channel: Vec<u8>,
    polarity: Vec<i32>,
    input: Vec<i32>,
    n: usize,
}

fn build_fixture(neuron_count: usize, edge_prob_pct: u64) -> Fixture {
    let graph = common::build_graph(neuron_count, edge_prob_pct);
    let (src, tgt) = graph.edge_endpoints_pub();

    let sources_usize = src.to_vec();
    let targets_usize = tgt.to_vec();
    let sources_u16: Vec<u16> = src.iter().map(|&s| s as u16).collect();
    let targets_u16: Vec<u16> = tgt.iter().map(|&t| t as u16).collect();
    let packed_u32: Vec<u32> = src
        .iter()
        .zip(tgt.iter())
        .map(|(&s, &t)| ((s as u32) << 16) | (t as u32))
        .collect();

    let mut input = vec![0i32; neuron_count];
    if let Some(first) = input.first_mut() {
        *first = 1;
    }

    Fixture {
        sources_usize,
        targets_usize,
        sources_u16,
        targets_u16,
        packed_u32,
        activation: vec![0i32; neuron_count],
        charge: vec![0; neuron_count],
        incoming: vec![0; neuron_count],
        threshold: vec![6u32; neuron_count],
        channel: vec![1u8; neuron_count],
        polarity: vec![1i32; neuron_count],
        input,
        n: neuron_count,
    }
}

// ---------------------------------------------------------------------------
// Full propagation: baseline (Vec<usize>)
// ---------------------------------------------------------------------------

fn propagate_baseline(f: &mut Fixture) {
    let n = f.n;
    let edge_src = &f.sources_usize;
    let edge_tgt = &f.targets_usize;

    for tick in 0..TICKS {
        if tick % 6 == 0 {
            for ch in f.charge.iter_mut() {
                *ch = ch.saturating_sub(1);
            }
        }
        if tick < 2 {
            for (act, &inp) in f.activation.iter_mut().zip(f.input.iter()) {
                *act += inp;
            }
        }

        let incoming = &mut f.incoming[..n];
        incoming.fill(0);
        for (sc, tc) in edge_src.chunks_exact(4).zip(edge_tgt.chunks_exact(4)) {
            incoming[tc[0]] += f.activation[sc[0]];
            incoming[tc[1]] += f.activation[sc[1]];
            incoming[tc[2]] += f.activation[sc[2]];
            incoming[tc[3]] += f.activation[sc[3]];
        }
        let rem = edge_src.len() / 4 * 4;
        for i in rem..edge_src.len() {
            incoming[edge_tgt[i]] += f.activation[edge_src[i]];
        }

        for (ch, &sig) in f.charge[..n].iter_mut().zip(incoming.iter()) {
            *ch = ch.saturating_add_signed(sig).min(MAX_CHARGE);
        }

        let phase_tick = tick % 8;
        for idx in 0..n {
            let ch_idx = f.channel[idx] as usize;
            let pm: u16 = if (1..=8).contains(&ch_idx) {
                PHASE_BASE[(phase_tick + 9 - ch_idx) & 7] as u16
            } else {
                10
            };
            let charge_x10 = f.charge[idx] as u16 * 10;
            let thresh_x10 = (f.threshold[idx] as u16 + 1) * pm;
            if charge_x10 >= thresh_x10 {
                f.activation[idx] = f.polarity[idx];
                f.charge[idx] = 0;
            } else {
                f.activation[idx] = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// u16 split: two Vec<u16>
// ---------------------------------------------------------------------------

fn propagate_u16_split(f: &mut Fixture) {
    let n = f.n;
    let edge_src = &f.sources_u16;
    let edge_tgt = &f.targets_u16;

    for tick in 0..TICKS {
        if tick % 6 == 0 {
            for ch in f.charge.iter_mut() {
                *ch = ch.saturating_sub(1);
            }
        }
        if tick < 2 {
            for (act, &inp) in f.activation.iter_mut().zip(f.input.iter()) {
                *act += inp;
            }
        }

        let incoming = &mut f.incoming[..n];
        incoming.fill(0);
        for (sc, tc) in edge_src.chunks_exact(4).zip(edge_tgt.chunks_exact(4)) {
            incoming[sc[0] as usize]; // hint bounds — help LLVM
            incoming[tc[0] as usize] += f.activation[sc[0] as usize];
            incoming[tc[1] as usize] += f.activation[sc[1] as usize];
            incoming[tc[2] as usize] += f.activation[sc[2] as usize];
            incoming[tc[3] as usize] += f.activation[sc[3] as usize];
        }
        let rem = edge_src.len() / 4 * 4;
        for i in rem..edge_src.len() {
            incoming[edge_tgt[i] as usize] += f.activation[edge_src[i] as usize];
        }

        for (ch, &sig) in f.charge[..n].iter_mut().zip(incoming.iter()) {
            *ch = ch.saturating_add_signed(sig).min(MAX_CHARGE);
        }

        let phase_tick = tick % 8;
        for idx in 0..n {
            let ch_idx = f.channel[idx] as usize;
            let pm: u16 = if (1..=8).contains(&ch_idx) {
                PHASE_BASE[(phase_tick + 9 - ch_idx) & 7] as u16
            } else {
                10
            };
            let charge_x10 = f.charge[idx] as u16 * 10;
            let thresh_x10 = (f.threshold[idx] as u16 + 1) * pm;
            if charge_x10 >= thresh_x10 {
                f.activation[idx] = f.polarity[idx];
                f.charge[idx] = 0;
            } else {
                f.activation[idx] = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Packed u32: single Vec<u32>, (src << 16) | tgt
// ---------------------------------------------------------------------------

fn propagate_packed_u32(f: &mut Fixture) {
    let n = f.n;
    let packed = &f.packed_u32;

    for tick in 0..TICKS {
        if tick % 6 == 0 {
            for ch in f.charge.iter_mut() {
                *ch = ch.saturating_sub(1);
            }
        }
        if tick < 2 {
            for (act, &inp) in f.activation.iter_mut().zip(f.input.iter()) {
                *act += inp;
            }
        }

        let incoming = &mut f.incoming[..n];
        incoming.fill(0);
        for chunk in packed.chunks_exact(4) {
            let e0 = chunk[0];
            let e1 = chunk[1];
            let e2 = chunk[2];
            let e3 = chunk[3];
            incoming[(e0 & 0xFFFF) as usize] += f.activation[(e0 >> 16) as usize];
            incoming[(e1 & 0xFFFF) as usize] += f.activation[(e1 >> 16) as usize];
            incoming[(e2 & 0xFFFF) as usize] += f.activation[(e2 >> 16) as usize];
            incoming[(e3 & 0xFFFF) as usize] += f.activation[(e3 >> 16) as usize];
        }
        let rem = packed.len() / 4 * 4;
        for i in rem..packed.len() {
            let e = packed[i];
            incoming[(e & 0xFFFF) as usize] += f.activation[(e >> 16) as usize];
        }

        for (ch, &sig) in f.charge[..n].iter_mut().zip(incoming.iter()) {
            *ch = ch.saturating_add_signed(sig).min(MAX_CHARGE);
        }

        let phase_tick = tick % 8;
        for idx in 0..n {
            let ch_idx = f.channel[idx] as usize;
            let pm: u16 = if (1..=8).contains(&ch_idx) {
                PHASE_BASE[(phase_tick + 9 - ch_idx) & 7] as u16
            } else {
                10
            };
            let charge_x10 = f.charge[idx] as u16 * 10;
            let thresh_x10 = (f.threshold[idx] as u16 + 1) * pm;
            if charge_x10 >= thresh_x10 {
                f.activation[idx] = f.polarity[idx];
                f.charge[idx] = 0;
            } else {
                f.activation[idx] = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------

fn reset(f: &mut Fixture) {
    f.activation.fill(0);
    f.charge.fill(0);
}

struct ABCase {
    name: &'static str,
    neuron_count: usize,
    edge_prob_pct: u64,
    iterations: usize,
}

fn run_ab(case: &ABCase) {
    println!(
        "\n=== {} | H={}, {}% density ===",
        case.name, case.neuron_count, case.edge_prob_pct
    );

    let mut f = build_fixture(case.neuron_count, case.edge_prob_pct);
    let edge_count = f.sources_usize.len();
    let baseline_bytes = edge_count * 16; // 2 × Vec<usize>
    let compact_bytes = edge_count * 4; // 2 × Vec<u16> or 1 × Vec<u32>
    println!(
        "  edges: {} | baseline: {}KB | compact: {}KB (4x smaller)",
        edge_count,
        baseline_bytes / 1024,
        compact_bytes / 1024
    );

    // Control: two baseline runs for noise floor
    let ctrl_a = {
        timed_run("CTRL-A (baseline #1)", case.iterations, || {
            reset(&mut f);
            propagate_baseline(black_box(&mut f));
        })
    };
    let ctrl_b = {
        timed_run("CTRL-B (baseline #2)", case.iterations, || {
            reset(&mut f);
            propagate_baseline(black_box(&mut f));
        })
    };
    let noise_pct = ((ctrl_b.median_ns - ctrl_a.median_ns) / ctrl_a.median_ns * 100.0).abs();
    println!(
        "  NOISE: {noise_pct:.1}% ({})",
        if noise_pct <= 5.0 {
            "stable"
        } else if noise_pct <= 10.0 {
            "borderline"
        } else {
            "noisy"
        }
    );

    let t_u16 = {
        timed_run("u16-split (2×Vec<u16>)", case.iterations, || {
            reset(&mut f);
            propagate_u16_split(black_box(&mut f));
        })
    };

    let t_packed = {
        timed_run("packed-u32 (1×Vec<u32>)", case.iterations, || {
            reset(&mut f);
            propagate_packed_u32(black_box(&mut f));
        })
    };

    let base_ns = ctrl_a.median_ns;
    let u16_delta = (t_u16.median_ns - base_ns) / base_ns * 100.0;
    let packed_delta = (t_packed.median_ns - base_ns) / base_ns * 100.0;

    println!("\n  RESULTS:");
    println!("    baseline (usize):  {:>10.0} ns", base_ns);
    println!(
        "    u16-split:         {:>10.0} ns  ({:+.1}%)",
        t_u16.median_ns, u16_delta
    );
    println!(
        "    packed-u32:        {:>10.0} ns  ({:+.1}%)",
        t_packed.median_ns, packed_delta
    );

    let best_name = if u16_delta < packed_delta {
        "u16-split"
    } else {
        "packed-u32"
    };
    let best_delta = u16_delta.min(packed_delta);
    if best_delta < -noise_pct {
        println!(
            "    VERDICT: {best_name} wins by {:.1}%",
            best_delta.abs()
        );
    } else if best_delta.abs() < noise_pct {
        println!("    VERDICT: within noise — no clear winner");
    } else {
        println!("    VERDICT: compact formats slower (bounds check overhead?)");
    }
}

fn main() {
    print_harness_header();

    let cases = [
        ABCase {
            name: "small",
            neuron_count: 256,
            edge_prob_pct: 5,
            iterations: 5_000,
        },
        ABCase {
            name: "medium",
            neuron_count: 1024,
            edge_prob_pct: 3,
            iterations: 2_000,
        },
        ABCase {
            name: "large",
            neuron_count: 4096,
            edge_prob_pct: 1,
            iterations: 500,
        },
    ];

    for case in &cases {
        run_ab(case);
    }
}
