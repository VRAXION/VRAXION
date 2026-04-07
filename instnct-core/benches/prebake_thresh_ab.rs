//! A/B: pre-baked threshold (store 1-16) vs current (store 0-15, +1 at runtime).
//! Quick test to confirm arithmetic is free vs memory-bound spike loop.

mod common;

use common::{build_graph, print_harness_header, timed_run};
use std::hint::black_box;

const TICKS: usize = 12;
const MAX_CHARGE: u8 = 15;
const PHASE_BASE: [u8; 8] = [7, 8, 10, 12, 13, 12, 10, 8];

struct Fixture {
    sources: Vec<u16>,
    targets: Vec<u16>,
    activation: Vec<i8>,
    charge: Vec<u8>,
    threshold_raw: Vec<u8>,    // 0-15 (current: +1 at runtime)
    threshold_baked: Vec<u8>,  // 1-16 (pre-baked: no +1 needed)
    channel: Vec<u8>,
    polarity: Vec<i8>,
    incoming: Vec<i16>,
    input: Vec<i32>,
    n: usize,
}

fn build_fixture(neuron_count: usize, edge_prob_pct: u64) -> Fixture {
    let graph = build_graph(neuron_count, edge_prob_pct);
    let (src, tgt) = graph.edge_endpoints_pub();
    let mut input = vec![0i32; neuron_count];
    if neuron_count > 0 { input[0] = 1; }

    let threshold_raw = vec![6u8; neuron_count];
    let threshold_baked: Vec<u8> = threshold_raw.iter().map(|&t| t + 1).collect();

    Fixture {
        sources: src.to_vec(), targets: tgt.to_vec(),
        activation: vec![0; neuron_count], charge: vec![0; neuron_count],
        threshold_raw, threshold_baked,
        channel: vec![1u8; neuron_count], polarity: vec![1i8; neuron_count],
        incoming: vec![0; neuron_count], input, n: neuron_count,
    }
}

fn propagate_current(f: &mut Fixture) {
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
            let val = (*ch as i16) + sig; *ch = val.clamp(0, MAX_CHARGE as i16) as u8;
        }

        let pt = tick % 8;
        for idx in 0..n {
            let ci = f.channel[idx] as usize;
            let pm: u16 = if (1..=8).contains(&ci) { PHASE_BASE[(pt + 9 - ci) & 7] as u16 } else { 10 };
            let charge_x10 = f.charge[idx] as u16 * 10;
            let thresh_x10 = (f.threshold_raw[idx] as u16 + 1) * pm;  // ◄── +1 at runtime
            if charge_x10 >= thresh_x10 {
                f.activation[idx] = f.polarity[idx]; f.charge[idx] = 0;
            } else { f.activation[idx] = 0; }
        }
    }
}

fn propagate_prebaked(f: &mut Fixture) {
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
            let val = (*ch as i16) + sig; *ch = val.clamp(0, MAX_CHARGE as i16) as u8;
        }

        let pt = tick % 8;
        for idx in 0..n {
            let ci = f.channel[idx] as usize;
            let pm: u16 = if (1..=8).contains(&ci) { PHASE_BASE[(pt + 9 - ci) & 7] as u16 } else { 10 };
            let charge_x10 = f.charge[idx] as u16 * 10;
            let thresh_x10 = f.threshold_baked[idx] as u16 * pm;  // ◄── NO +1, pre-baked
            if charge_x10 >= thresh_x10 {
                f.activation[idx] = f.polarity[idx]; f.charge[idx] = 0;
            } else { f.activation[idx] = 0; }
        }
    }
}

struct Case { name: &'static str, n: usize, e: u64, iters: usize }

fn run(c: &Case) {
    println!("\n=== {} | H={}, {}% ===", c.name, c.n, c.e);
    let mut f = build_fixture(c.n, c.e);
    println!("  edges: {}", f.sources.len());

    let a = timed_run("current (+1 runtime)", c.iters, || {
        f.activation.fill(0); f.charge.fill(0);
        propagate_current(black_box(&mut f));
    });
    let b = timed_run("prebaked (no +1)", c.iters, || {
        f.activation.fill(0); f.charge.fill(0);
        propagate_prebaked(black_box(&mut f));
    });

    let d = (b.median_ns - a.median_ns) / a.median_ns * 100.0;
    println!("\n    current:  {:.0} ns", a.median_ns);
    println!("    prebaked: {:.0} ns ({:+.1}%)", b.median_ns, d);
}

fn main() {
    print_harness_header();
    for c in &[
        Case { name: "small",  n: 256,  e: 5, iters: 5000 },
        Case { name: "medium", n: 1024, e: 3, iters: 2000 },
        Case { name: "large",  n: 4096, e: 1, iters: 500  },
    ] { run(c); }
}
