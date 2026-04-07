//! A/B: Array-of-Structs spike data (1 stream) vs separate arrays (3 streams).
//!
//! Combine charge+threshold+channel into one contiguous struct array.
//! No packing/encoding — just different memory layout.
//! 3 separate streams → 1 stream, same total bytes.

mod common;

use common::{build_graph, print_harness_header, timed_run};
use std::hint::black_box;

const TICKS: usize = 12;
const MAX_CHARGE: u8 = 15;
const PHASE_BASE: [u8; 8] = [7, 8, 10, 12, 13, 12, 10, 8];

// ---------------------------------------------------------------------------
// AoS: charge + threshold + channel in one struct
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
#[repr(C)]
struct NeuronSpike {
    charge: u8,
    threshold: u8,
    channel: u8,
}

// 4-byte aligned version (pad to power of 2 for indexing)
#[derive(Clone, Copy)]
#[repr(C)]
struct NeuronSpike4 {
    charge: u8,
    threshold: u8,
    channel: u8,
    _pad: u8,
}

struct BaselineFixture {
    sources: Vec<u16>, targets: Vec<u16>,
    activation: Vec<i8>, charge: Vec<u8>, threshold: Vec<u8>,
    channel: Vec<u8>, polarity: Vec<i8>, incoming: Vec<i16>,
    input: Vec<i32>, n: usize,
}

struct AosFixture {
    sources: Vec<u16>, targets: Vec<u16>,
    activation: Vec<i8>, spike: Vec<NeuronSpike>,
    polarity: Vec<i8>, incoming: Vec<i16>,
    input: Vec<i32>, n: usize,
}

struct Aos4Fixture {
    sources: Vec<u16>, targets: Vec<u16>,
    activation: Vec<i8>, spike: Vec<NeuronSpike4>,
    polarity: Vec<i8>, incoming: Vec<i16>,
    input: Vec<i32>, n: usize,
}

fn build(nc: usize, ep: u64) -> (BaselineFixture, AosFixture, Aos4Fixture) {
    let graph = build_graph(nc, ep);
    let (src, tgt) = graph.edge_endpoints_pub();
    let mut input = vec![0i32; nc];
    if nc > 0 { input[0] = 1; }

    let b = BaselineFixture {
        sources: src.to_vec(), targets: tgt.to_vec(),
        activation: vec![0; nc], charge: vec![0; nc],
        threshold: vec![6; nc], channel: vec![1; nc],
        polarity: vec![1i8; nc], incoming: vec![0; nc],
        input: input.clone(), n: nc,
    };
    let a = AosFixture {
        sources: src.to_vec(), targets: tgt.to_vec(),
        activation: vec![0; nc],
        spike: vec![NeuronSpike { charge: 0, threshold: 6, channel: 1 }; nc],
        polarity: vec![1i8; nc], incoming: vec![0; nc],
        input: input.clone(), n: nc,
    };
    let a4 = Aos4Fixture {
        sources: src.to_vec(), targets: tgt.to_vec(),
        activation: vec![0; nc],
        spike: vec![NeuronSpike4 { charge: 0, threshold: 6, channel: 1, _pad: 0 }; nc],
        polarity: vec![1i8; nc], incoming: vec![0; nc],
        input: input.clone(), n: nc,
    };
    (b, a, a4)
}

// ---------------------------------------------------------------------------
// Baseline: 3 separate arrays
// ---------------------------------------------------------------------------

fn propagate_baseline(f: &mut BaselineFixture) {
    let n = f.n;
    for tick in 0..TICKS {
        if tick % 6 == 0 { for ch in f.charge.iter_mut() { *ch = ch.saturating_sub(1); } }
        if tick < 2 { for (a, &i) in f.activation.iter_mut().zip(f.input.iter()) { *a = a.saturating_add(i as i8); } }

        f.incoming[..n].fill(0);
        for (sc, tc) in f.sources.chunks_exact(4).zip(f.targets.chunks_exact(4)) {
            f.incoming[tc[0] as usize] += f.activation[sc[0] as usize] as i16;
            f.incoming[tc[1] as usize] += f.activation[sc[1] as usize] as i16;
            f.incoming[tc[2] as usize] += f.activation[sc[2] as usize] as i16;
            f.incoming[tc[3] as usize] += f.activation[sc[3] as usize] as i16;
        }
        let rem = f.sources.len() / 4 * 4;
        for i in rem..f.sources.len() { f.incoming[f.targets[i] as usize] += f.activation[f.sources[i] as usize] as i16; }

        for (ch, &sig) in f.charge[..n].iter_mut().zip(f.incoming.iter()) {
            let v = (*ch as i16) + sig; *ch = v.clamp(0, MAX_CHARGE as i16) as u8;
        }

        let pt = tick % 8;
        for idx in 0..n {
            let ci = f.channel[idx] as usize;
            let pm: u16 = if (1..=8).contains(&ci) { PHASE_BASE[(pt + 9 - ci) & 7] as u16 } else { 10 };
            let cx10 = f.charge[idx] as u16 * 10;
            let tx10 = (f.threshold[idx] as u16 + 1) * pm;
            if cx10 >= tx10 { f.activation[idx] = f.polarity[idx]; f.charge[idx] = 0; }
            else { f.activation[idx] = 0; }
        }
    }
}

// ---------------------------------------------------------------------------
// AoS 3-byte: single struct array
// ---------------------------------------------------------------------------

fn propagate_aos(f: &mut AosFixture) {
    let n = f.n;
    for tick in 0..TICKS {
        if tick % 6 == 0 { for s in f.spike.iter_mut() { s.charge = s.charge.saturating_sub(1); } }
        if tick < 2 { for (a, &i) in f.activation.iter_mut().zip(f.input.iter()) { *a = a.saturating_add(i as i8); } }

        f.incoming[..n].fill(0);
        for (sc, tc) in f.sources.chunks_exact(4).zip(f.targets.chunks_exact(4)) {
            f.incoming[tc[0] as usize] += f.activation[sc[0] as usize] as i16;
            f.incoming[tc[1] as usize] += f.activation[sc[1] as usize] as i16;
            f.incoming[tc[2] as usize] += f.activation[sc[2] as usize] as i16;
            f.incoming[tc[3] as usize] += f.activation[sc[3] as usize] as i16;
        }
        let rem = f.sources.len() / 4 * 4;
        for i in rem..f.sources.len() { f.incoming[f.targets[i] as usize] += f.activation[f.sources[i] as usize] as i16; }

        for (s, &sig) in f.spike[..n].iter_mut().zip(f.incoming.iter()) {
            let v = (s.charge as i16) + sig; s.charge = v.clamp(0, MAX_CHARGE as i16) as u8;
        }

        let pt = tick % 8;
        for idx in 0..n {
            let s = &f.spike[idx];
            let ci = s.channel as usize;
            let pm: u16 = if (1..=8).contains(&ci) { PHASE_BASE[(pt + 9 - ci) & 7] as u16 } else { 10 };
            let cx10 = s.charge as u16 * 10;
            let tx10 = (s.threshold as u16 + 1) * pm;
            if cx10 >= tx10 { f.activation[idx] = f.polarity[idx]; f.spike[idx].charge = 0; }
            else { f.activation[idx] = 0; }
        }
    }
}

// ---------------------------------------------------------------------------
// AoS 4-byte (padded): power-of-2 stride for better indexing
// ---------------------------------------------------------------------------

fn propagate_aos4(f: &mut Aos4Fixture) {
    let n = f.n;
    for tick in 0..TICKS {
        if tick % 6 == 0 { for s in f.spike.iter_mut() { s.charge = s.charge.saturating_sub(1); } }
        if tick < 2 { for (a, &i) in f.activation.iter_mut().zip(f.input.iter()) { *a = a.saturating_add(i as i8); } }

        f.incoming[..n].fill(0);
        for (sc, tc) in f.sources.chunks_exact(4).zip(f.targets.chunks_exact(4)) {
            f.incoming[tc[0] as usize] += f.activation[sc[0] as usize] as i16;
            f.incoming[tc[1] as usize] += f.activation[sc[1] as usize] as i16;
            f.incoming[tc[2] as usize] += f.activation[sc[2] as usize] as i16;
            f.incoming[tc[3] as usize] += f.activation[sc[3] as usize] as i16;
        }
        let rem = f.sources.len() / 4 * 4;
        for i in rem..f.sources.len() { f.incoming[f.targets[i] as usize] += f.activation[f.sources[i] as usize] as i16; }

        for (s, &sig) in f.spike[..n].iter_mut().zip(f.incoming.iter()) {
            let v = (s.charge as i16) + sig; s.charge = v.clamp(0, MAX_CHARGE as i16) as u8;
        }

        let pt = tick % 8;
        for idx in 0..n {
            let s = &f.spike[idx];
            let ci = s.channel as usize;
            let pm: u16 = if (1..=8).contains(&ci) { PHASE_BASE[(pt + 9 - ci) & 7] as u16 } else { 10 };
            let cx10 = s.charge as u16 * 10;
            let tx10 = (s.threshold as u16 + 1) * pm;
            if cx10 >= tx10 { f.activation[idx] = f.polarity[idx]; f.spike[idx].charge = 0; }
            else { f.activation[idx] = 0; }
        }
    }
}

// ---------------------------------------------------------------------------

struct Case { name: &'static str, n: usize, e: u64, iters: usize }

fn run(c: &Case) {
    println!("\n=== {} | H={}, {}% ===", c.name, c.n, c.e);
    let (mut b, mut a, mut a4) = build(c.n, c.e);
    println!("  edges: {} | baseline: {} B/neuron | aos: 3 B | aos4: 4 B",
        b.sources.len(), 3); // charge+threshold+channel = 3 bytes either way

    let ctrl = timed_run("baseline (3 arrays)", c.iters, || {
        b.activation.fill(0); b.charge.fill(0);
        propagate_baseline(black_box(&mut b));
    });
    let ctrl2 = timed_run("baseline #2", c.iters, || {
        b.activation.fill(0); b.charge.fill(0);
        propagate_baseline(black_box(&mut b));
    });
    let noise = ((ctrl2.median_ns - ctrl.median_ns) / ctrl.median_ns * 100.0).abs();
    println!("  NOISE: {noise:.1}%");

    let t_aos = timed_run("AoS 3-byte struct", c.iters, || {
        a.activation.fill(0); for s in a.spike.iter_mut() { s.charge = 0; }
        propagate_aos(black_box(&mut a));
    });
    let t_aos4 = timed_run("AoS 4-byte (padded)", c.iters, || {
        a4.activation.fill(0); for s in a4.spike.iter_mut() { s.charge = 0; }
        propagate_aos4(black_box(&mut a4));
    });

    let base = ctrl.median_ns;
    let d3 = (t_aos.median_ns - base) / base * 100.0;
    let d4 = (t_aos4.median_ns - base) / base * 100.0;
    println!("\n    baseline:   {:>10.0} ns", base);
    println!("    AoS 3-byte: {:>10.0} ns ({:+.1}%)", t_aos.median_ns, d3);
    println!("    AoS 4-byte: {:>10.0} ns ({:+.1}%)", t_aos4.median_ns, d4);
}

fn main() {
    print_harness_header();
    for c in &[
        Case { name: "small",  n: 256,  e: 5, iters: 5000 },
        Case { name: "medium", n: 1024, e: 3, iters: 2000 },
        Case { name: "large",  n: 4096, e: 1, iters: 500  },
    ] { run(c); }
}
