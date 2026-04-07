//! A/B benchmark: packed spike data (2 bytes/neuron) vs separate arrays (4 bytes/neuron).
//!
//! The spike decision loop reads charge, threshold, channel, polarity per neuron.
//! Currently these are 4 separate arrays (4 cache line streams).
//! Packed format: 2 bytes per neuron, 1 cache line stream:
//!   byte 0: charge (high nibble) | threshold (low nibble)
//!   byte 1: bit 7 = polarity (0=+1, 1=-1), bits 0-2 = channel (1-8 stored as 0-7)
//!
//! activation stays separate (i8 array) — it's the scatter-add hot path.

mod common;

use common::{build_graph, print_harness_header, timed_run};
use std::hint::black_box;

const TICKS: usize = 12;
const MAX_CHARGE: u8 = 15;
const PHASE_BASE: [u8; 8] = [7, 8, 10, 12, 13, 12, 10, 8];

// ---------------------------------------------------------------------------
// Baseline: separate arrays (current library layout)
// ---------------------------------------------------------------------------

struct BaselineFixture {
    // Edge data
    sources: Vec<u16>,
    targets: Vec<u16>,
    // Separate neuron arrays
    activation: Vec<i8>,
    charge: Vec<u8>,
    threshold: Vec<u8>,
    channel: Vec<u8>,
    polarity: Vec<i8>,
    incoming: Vec<i16>,
    input: Vec<i32>,
    n: usize,
}

fn build_baseline(neuron_count: usize, edge_prob_pct: u64) -> BaselineFixture {
    let graph = common::build_graph(neuron_count, edge_prob_pct);
    let (src, tgt) = graph.edge_endpoints_pub();
    let mut input = vec![0i32; neuron_count];
    if neuron_count > 0 {
        input[0] = 1;
    }
    BaselineFixture {
        sources: src.to_vec(),
        targets: tgt.to_vec(),
        activation: vec![0i8; neuron_count],
        charge: vec![0u8; neuron_count],
        threshold: vec![6u8; neuron_count],
        channel: vec![1u8; neuron_count],
        polarity: vec![1i8; neuron_count],
        incoming: vec![0i16; neuron_count],
        input,
        n: neuron_count,
    }
}

fn propagate_baseline(f: &mut BaselineFixture) {
    let n = f.n;
    let edge_src = &f.sources;
    let edge_tgt = &f.targets;

    for tick in 0..TICKS {
        if tick % 6 == 0 {
            for ch in f.charge.iter_mut() {
                *ch = ch.saturating_sub(1);
            }
        }
        if tick < 2 {
            for (act, &inp) in f.activation.iter_mut().zip(f.input.iter()) {
                *act = act.saturating_add(inp as i8);
            }
        }

        let incoming = &mut f.incoming[..n];
        incoming.fill(0);
        for (sc, tc) in edge_src.chunks_exact(4).zip(edge_tgt.chunks_exact(4)) {
            incoming[tc[0] as usize] += f.activation[sc[0] as usize] as i16;
            incoming[tc[1] as usize] += f.activation[sc[1] as usize] as i16;
            incoming[tc[2] as usize] += f.activation[sc[2] as usize] as i16;
            incoming[tc[3] as usize] += f.activation[sc[3] as usize] as i16;
        }
        let rem = edge_src.len() / 4 * 4;
        for i in rem..edge_src.len() {
            incoming[edge_tgt[i] as usize] += f.activation[edge_src[i] as usize] as i16;
        }

        for (ch, &sig) in f.charge[..n].iter_mut().zip(incoming.iter()) {
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
// Packed: 2 bytes per neuron for spike data
// ---------------------------------------------------------------------------
//
// Layout:
//   packed[idx*2]:     charge (high nibble 4 bits) | threshold (low nibble 4 bits)
//   packed[idx*2 + 1]: polarity (bit 7: 0=+1, 1=-1) | channel-1 (bits 0-2, 3 bits)
//
// Total: 2 bytes per neuron vs 4 bytes (charge + threshold + channel + polarity)
// activation and incoming stay as separate arrays (different access pattern).

struct PackedFixture {
    sources: Vec<u16>,
    targets: Vec<u16>,
    activation: Vec<i8>,
    packed: Vec<u8>,     // 2 bytes per neuron: [charge|threshold, polarity|channel]
    incoming: Vec<i16>,
    input: Vec<i32>,
    n: usize,
}

#[inline(always)]
fn pack_charge_threshold(charge: u8, threshold: u8) -> u8 {
    (charge << 4) | (threshold & 0x0F)
}

#[inline(always)]
fn unpack_charge(byte: u8) -> u8 {
    byte >> 4
}

#[inline(always)]
fn unpack_threshold(byte: u8) -> u8 {
    byte & 0x0F
}

#[inline(always)]
fn pack_polarity_channel(polarity: i8, channel: u8) -> u8 {
    let pol_bit = if polarity < 0 { 0x80u8 } else { 0u8 };
    pol_bit | ((channel - 1) & 0x07)
}

#[inline(always)]
fn unpack_polarity(byte: u8) -> i8 {
    if byte & 0x80 != 0 { -1 } else { 1 }
}

#[inline(always)]
fn unpack_channel(byte: u8) -> u8 {
    (byte & 0x07) + 1
}

fn build_packed(neuron_count: usize, edge_prob_pct: u64) -> PackedFixture {
    let graph = common::build_graph(neuron_count, edge_prob_pct);
    let (src, tgt) = graph.edge_endpoints_pub();
    let mut input = vec![0i32; neuron_count];
    if neuron_count > 0 {
        input[0] = 1;
    }

    // Pack: charge=0, threshold=6, polarity=+1, channel=1
    let mut packed = vec![0u8; neuron_count * 2];
    for idx in 0..neuron_count {
        packed[idx * 2] = pack_charge_threshold(0, 6);
        packed[idx * 2 + 1] = pack_polarity_channel(1, 1);
    }

    PackedFixture {
        sources: src.to_vec(),
        targets: tgt.to_vec(),
        activation: vec![0i8; neuron_count],
        packed,
        incoming: vec![0i16; neuron_count],
        input,
        n: neuron_count,
    }
}

fn propagate_packed(f: &mut PackedFixture) {
    let n = f.n;
    let edge_src = &f.sources;
    let edge_tgt = &f.targets;

    for tick in 0..TICKS {
        // Charge decay: unpack, decrement, repack
        if tick % 6 == 0 {
            for idx in 0..n {
                let b = &mut f.packed[idx * 2];
                let charge = unpack_charge(*b);
                let threshold = unpack_threshold(*b);
                *b = pack_charge_threshold(charge.saturating_sub(1), threshold);
            }
        }

        if tick < 2 {
            for (act, &inp) in f.activation.iter_mut().zip(f.input.iter()) {
                *act = act.saturating_add(inp as i8);
            }
        }

        // Scatter-add: same as baseline (activation is separate)
        let incoming = &mut f.incoming[..n];
        incoming.fill(0);
        for (sc, tc) in edge_src.chunks_exact(4).zip(edge_tgt.chunks_exact(4)) {
            incoming[tc[0] as usize] += f.activation[sc[0] as usize] as i16;
            incoming[tc[1] as usize] += f.activation[sc[1] as usize] as i16;
            incoming[tc[2] as usize] += f.activation[sc[2] as usize] as i16;
            incoming[tc[3] as usize] += f.activation[sc[3] as usize] as i16;
        }
        let rem = edge_src.len() / 4 * 4;
        for i in rem..edge_src.len() {
            incoming[edge_tgt[i] as usize] += f.activation[edge_src[i] as usize] as i16;
        }

        // Charge accumulation: unpack charge, add signal, repack
        for idx in 0..n {
            let b = &mut f.packed[idx * 2];
            let charge = unpack_charge(*b);
            let threshold = unpack_threshold(*b);
            let val = (charge as i16) + f.incoming[idx];
            let new_charge = val.clamp(0, MAX_CHARGE as i16) as u8;
            *b = pack_charge_threshold(new_charge, threshold);
        }

        // Spike decision: all data from packed[] (2 bytes per neuron)
        let phase_tick = tick % 8;
        for idx in 0..n {
            let ct = f.packed[idx * 2];
            let pc = f.packed[idx * 2 + 1];
            let charge = unpack_charge(ct);
            let threshold = unpack_threshold(ct);
            let channel = unpack_channel(pc);

            let ch_idx = channel as usize;
            let pm: u16 = if (1..=8).contains(&ch_idx) {
                PHASE_BASE[(phase_tick + 9 - ch_idx) & 7] as u16
            } else {
                10
            };
            let charge_x10 = charge as u16 * 10;
            let thresh_x10 = (threshold as u16 + 1) * pm;

            if charge_x10 >= thresh_x10 {
                f.activation[idx] = unpack_polarity(pc);
                f.packed[idx * 2] = pack_charge_threshold(0, threshold); // reset charge
            } else {
                f.activation[idx] = 0;
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

    let mut baseline = build_baseline(case.neuron_count, case.edge_prob_pct);
    let mut packed = build_packed(case.neuron_count, case.edge_prob_pct);

    let baseline_neuron_bytes = case.neuron_count * 4; // charge + threshold + channel + polarity
    let packed_neuron_bytes = case.neuron_count * 2;
    println!(
        "  edges: {} | spike data: baseline={}B packed={}B ({}x smaller)",
        baseline.sources.len(),
        baseline_neuron_bytes,
        packed_neuron_bytes,
        baseline_neuron_bytes / packed_neuron_bytes.max(1)
    );

    let ctrl_a = timed_run("CTRL-A (baseline #1)", case.iterations, || {
        baseline.activation.fill(0);
        baseline.charge.fill(0);
        propagate_baseline(black_box(&mut baseline));
    });
    let ctrl_b = timed_run("CTRL-B (baseline #2)", case.iterations, || {
        baseline.activation.fill(0);
        baseline.charge.fill(0);
        propagate_baseline(black_box(&mut baseline));
    });
    let noise_pct = ((ctrl_b.median_ns - ctrl_a.median_ns) / ctrl_a.median_ns * 100.0).abs();
    println!(
        "  NOISE: {noise_pct:.1}% ({})",
        if noise_pct <= 5.0 { "stable" } else { "noisy" }
    );

    let t_packed = timed_run("packed-2B (nibble+bit pack)", case.iterations, || {
        packed.activation.fill(0);
        for idx in 0..packed.n {
            let threshold = unpack_threshold(packed.packed[idx * 2]);
            packed.packed[idx * 2] = pack_charge_threshold(0, threshold);
        }
        propagate_packed(black_box(&mut packed));
    });

    let base_ns = ctrl_a.median_ns;
    let packed_delta = (t_packed.median_ns - base_ns) / base_ns * 100.0;

    println!("\n  RESULTS:");
    println!("    baseline (4 arrays): {:>10.0} ns", base_ns);
    println!(
        "    packed-2B:           {:>10.0} ns  ({:+.1}%)",
        t_packed.median_ns, packed_delta
    );

    if packed_delta < -noise_pct {
        println!("    VERDICT: packed wins by {:.1}%", packed_delta.abs());
    } else if packed_delta.abs() < noise_pct {
        println!("    VERDICT: within noise");
    } else {
        println!("    VERDICT: packed slower — unpack overhead > cache benefit");
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
        ABCase {
            name: "xlarge",
            neuron_count: 16384,
            edge_prob_pct: 0, // ~0.3% density from rng
            iterations: 100,
        },
    ];

    for case in &cases {
        run_ab(case);
    }
}
