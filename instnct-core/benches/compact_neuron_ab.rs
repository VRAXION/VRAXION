//! A/B benchmark: compact neuron arrays (i8/u8) vs current (i32/u32).
//!
//! Tests full propagation with all neuron arrays shrunk to their minimum
//! viable type. The scatter-add reads activation[] which shrinks from
//! i32 (4B) to i8 (1B). The spike loop reads threshold/channel/polarity
//! and writes activation/charge — all shrink similarly.
//!
//! Variants:
//!   baseline:     i32 activation, u32 charge, u32 threshold, i32 polarity (current)
//!   compact:      i8 activation, u8 charge, u8 threshold, i8 polarity, i16 incoming
//!   compact+u16e: compact neurons + u16 edge indices (combined best)

mod common;

use common::{build_graph, print_harness_header, timed_run};
use std::hint::black_box;

const TICKS: usize = 12;
const MAX_CHARGE: u8 = 15;
const PHASE_BASE: [u8; 8] = [7, 8, 10, 12, 13, 12, 10, 8];

struct Fixture {
    // Edge data
    sources_usize: Vec<usize>,
    targets_usize: Vec<usize>,
    sources_u16: Vec<u16>,
    targets_u16: Vec<u16>,
    n: usize,

    // Baseline neuron data (wide)
    w_activation: Vec<i32>,
    w_charge: Vec<u32>,
    w_incoming: Vec<i32>,
    w_threshold: Vec<u32>,
    w_polarity: Vec<i32>,
    w_input: Vec<i32>,

    // Compact neuron data (narrow)
    c_activation: Vec<i8>,
    c_charge: Vec<u8>,
    c_incoming: Vec<i16>,
    c_threshold: Vec<u8>,
    c_polarity: Vec<i8>,
    c_input: Vec<i8>,

    // Shared (already u8)
    channel: Vec<u8>,
}

fn build_fixture(neuron_count: usize, edge_prob_pct: u64) -> Fixture {
    let graph = build_graph(neuron_count, edge_prob_pct);
    let (src, tgt) = graph.edge_endpoints_pub();

    let mut w_input = vec![0i32; neuron_count];
    let mut c_input = vec![0i8; neuron_count];
    if neuron_count > 0 {
        w_input[0] = 1;
        c_input[0] = 1;
    }

    Fixture {
        sources_usize: src.to_vec(),
        targets_usize: tgt.to_vec(),
        sources_u16: src.iter().map(|&s| s as u16).collect(),
        targets_u16: tgt.iter().map(|&t| t as u16).collect(),
        n: neuron_count,

        w_activation: vec![0; neuron_count],
        w_charge: vec![0; neuron_count],
        w_incoming: vec![0; neuron_count],
        w_threshold: vec![6; neuron_count],
        w_polarity: vec![1; neuron_count],
        w_input,

        c_activation: vec![0; neuron_count],
        c_charge: vec![0; neuron_count],
        c_incoming: vec![0; neuron_count],
        c_threshold: vec![6; neuron_count],
        c_polarity: vec![1; neuron_count],
        c_input,

        channel: vec![1u8; neuron_count],
    }
}

// ---------------------------------------------------------------------------
// Baseline: all i32/u32 + usize edges (current library)
// ---------------------------------------------------------------------------

fn propagate_baseline(f: &mut Fixture) {
    let n = f.n;
    let edge_src = &f.sources_usize;
    let edge_tgt = &f.targets_usize;

    for tick in 0..TICKS {
        if tick % 6 == 0 {
            for ch in f.w_charge.iter_mut() {
                *ch = ch.saturating_sub(1);
            }
        }
        if tick < 2 {
            for (act, &inp) in f.w_activation.iter_mut().zip(f.w_input.iter()) {
                *act += inp;
            }
        }

        let incoming = &mut f.w_incoming[..n];
        incoming.fill(0);
        for (sc, tc) in edge_src.chunks_exact(4).zip(edge_tgt.chunks_exact(4)) {
            incoming[tc[0]] += f.w_activation[sc[0]];
            incoming[tc[1]] += f.w_activation[sc[1]];
            incoming[tc[2]] += f.w_activation[sc[2]];
            incoming[tc[3]] += f.w_activation[sc[3]];
        }
        let rem = edge_src.len() / 4 * 4;
        for i in rem..edge_src.len() {
            incoming[edge_tgt[i]] += f.w_activation[edge_src[i]];
        }

        for (ch, &sig) in f.w_charge[..n].iter_mut().zip(incoming.iter()) {
            *ch = ch.saturating_add_signed(sig).min(15);
        }

        let phase_tick = tick % 8;
        for idx in 0..n {
            let ch_idx = f.channel[idx] as usize;
            let pm: u16 = if (1..=8).contains(&ch_idx) {
                PHASE_BASE[(phase_tick + 9 - ch_idx) & 7] as u16
            } else {
                10
            };
            let charge_x10 = f.w_charge[idx] as u16 * 10;
            let thresh_x10 = (f.w_threshold[idx] as u16 + 1) * pm;
            if charge_x10 >= thresh_x10 {
                f.w_activation[idx] = f.w_polarity[idx];
                f.w_charge[idx] = 0;
            } else {
                f.w_activation[idx] = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Compact: i8/u8/i16 neurons + usize edges
// ---------------------------------------------------------------------------

fn propagate_compact(f: &mut Fixture) {
    let n = f.n;
    let edge_src = &f.sources_usize;
    let edge_tgt = &f.targets_usize;

    for tick in 0..TICKS {
        if tick % 6 == 0 {
            for ch in f.c_charge.iter_mut() {
                *ch = ch.saturating_sub(1);
            }
        }
        if tick < 2 {
            for (act, &inp) in f.c_activation.iter_mut().zip(f.c_input.iter()) {
                *act = act.saturating_add(inp);
            }
        }

        let incoming = &mut f.c_incoming[..n];
        incoming.fill(0);
        for (sc, tc) in edge_src.chunks_exact(4).zip(edge_tgt.chunks_exact(4)) {
            incoming[tc[0]] += f.c_activation[sc[0]] as i16;
            incoming[tc[1]] += f.c_activation[sc[1]] as i16;
            incoming[tc[2]] += f.c_activation[sc[2]] as i16;
            incoming[tc[3]] += f.c_activation[sc[3]] as i16;
        }
        let rem = edge_src.len() / 4 * 4;
        for i in rem..edge_src.len() {
            incoming[edge_tgt[i]] += f.c_activation[edge_src[i]] as i16;
        }

        for (ch, &sig) in f.c_charge[..n].iter_mut().zip(incoming.iter()) {
            // saturating add: clamp to [0, 15]
            let val = (*ch as i16) + sig;
            *ch = val.clamp(0, MAX_CHARGE as i16) as u8;
        }

        let phase_tick = tick % 8;
        for idx in 0..n {
            let ch_idx = f.channel[idx] as usize;
            let pm: u16 = if (1..=8).contains(&ch_idx) {
                PHASE_BASE[(phase_tick + 9 - ch_idx) & 7] as u16
            } else {
                10
            };
            let charge_x10 = f.c_charge[idx] as u16 * 10;
            let thresh_x10 = (f.c_threshold[idx] as u16 + 1) * pm;
            if charge_x10 >= thresh_x10 {
                f.c_activation[idx] = f.c_polarity[idx];
                f.c_charge[idx] = 0;
            } else {
                f.c_activation[idx] = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Compact + u16 edges: the full combo
// ---------------------------------------------------------------------------

fn propagate_compact_u16e(f: &mut Fixture) {
    let n = f.n;
    let edge_src = &f.sources_u16;
    let edge_tgt = &f.targets_u16;

    for tick in 0..TICKS {
        if tick % 6 == 0 {
            for ch in f.c_charge.iter_mut() {
                *ch = ch.saturating_sub(1);
            }
        }
        if tick < 2 {
            for (act, &inp) in f.c_activation.iter_mut().zip(f.c_input.iter()) {
                *act = act.saturating_add(inp);
            }
        }

        let incoming = &mut f.c_incoming[..n];
        incoming.fill(0);
        for (sc, tc) in edge_src.chunks_exact(4).zip(edge_tgt.chunks_exact(4)) {
            incoming[tc[0] as usize] += f.c_activation[sc[0] as usize] as i16;
            incoming[tc[1] as usize] += f.c_activation[sc[1] as usize] as i16;
            incoming[tc[2] as usize] += f.c_activation[sc[2] as usize] as i16;
            incoming[tc[3] as usize] += f.c_activation[sc[3] as usize] as i16;
        }
        let rem = edge_src.len() / 4 * 4;
        for i in rem..edge_src.len() {
            incoming[edge_tgt[i] as usize] += f.c_activation[edge_src[i] as usize] as i16;
        }

        for (ch, &sig) in f.c_charge[..n].iter_mut().zip(incoming.iter()) {
            let val = (*ch as i16) + sig;
            *ch = val.clamp(0, MAX_CHARGE as i16) as u8;
        }

        let phase_tick = tick % 8;
        for idx in 0..n {
            let ch_idx = f.channel[idx] as usize;
            let pm: u16 = if (1..=8).contains(&ch_idx) {
                PHASE_BASE[(phase_tick + 9 - ch_idx) & 7] as u16
            } else {
                10
            };
            let charge_x10 = f.c_charge[idx] as u16 * 10;
            let thresh_x10 = (f.c_threshold[idx] as u16 + 1) * pm;
            if charge_x10 >= thresh_x10 {
                f.c_activation[idx] = f.c_polarity[idx];
                f.c_charge[idx] = 0;
            } else {
                f.c_activation[idx] = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------

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
    let edges = f.sources_usize.len();
    let base_bytes = edges * 16 + case.neuron_count * 24;
    let compact_bytes = edges * 4 + case.neuron_count * 6;
    println!(
        "  edges: {} | baseline hot data: {}KB | compact: {}KB ({:.1}x smaller)",
        edges,
        base_bytes / 1024,
        compact_bytes / 1024,
        base_bytes as f64 / compact_bytes as f64
    );

    let ctrl_a = timed_run("CTRL-A (baseline)", case.iterations, || {
        f.w_activation.fill(0);
        f.w_charge.fill(0);
        propagate_baseline(black_box(&mut f));
    });
    let ctrl_b = timed_run("CTRL-B (baseline)", case.iterations, || {
        f.w_activation.fill(0);
        f.w_charge.fill(0);
        propagate_baseline(black_box(&mut f));
    });
    let noise_pct = ((ctrl_b.median_ns - ctrl_a.median_ns) / ctrl_a.median_ns * 100.0).abs();
    println!(
        "  NOISE: {noise_pct:.1}% ({})",
        if noise_pct <= 5.0 { "stable" } else { "noisy" }
    );

    let t_compact = timed_run("compact neurons (i8/u8)", case.iterations, || {
        f.c_activation.fill(0);
        f.c_charge.fill(0);
        propagate_compact(black_box(&mut f));
    });

    let t_combo = timed_run("compact + u16 edges (full combo)", case.iterations, || {
        f.c_activation.fill(0);
        f.c_charge.fill(0);
        propagate_compact_u16e(black_box(&mut f));
    });

    let base_ns = ctrl_a.median_ns;
    let compact_delta = (t_compact.median_ns - base_ns) / base_ns * 100.0;
    let combo_delta = (t_combo.median_ns - base_ns) / base_ns * 100.0;

    println!("\n  RESULTS:");
    println!("    baseline:            {:>10.0} ns", base_ns);
    println!(
        "    compact neurons:     {:>10.0} ns  ({:+.1}%)",
        t_compact.median_ns, compact_delta
    );
    println!(
        "    compact + u16 edges: {:>10.0} ns  ({:+.1}%)",
        t_combo.median_ns, combo_delta
    );
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
