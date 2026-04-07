//! A/B: dirty_member Vec<bool> vs bitset Vec<u64> for large H.
//!
//! The dirty_member bitmap is O(1) lookup either way, but:
//!   Vec<bool>: H bytes (100K = 100KB)
//!   Vec<u64>:  H/64 × 8 bytes (100K = 12.5KB, 8x smaller)
//!
//! Tests realistic sparse-tick propagation with large H.

mod common;

use common::{build_graph, print_harness_header, timed_run};
use instnct_core::ConnectionGraph;
use std::hint::black_box;

const TICKS: usize = 12;
const MAX_CHARGE: u8 = 15;
const PHASE_BASE: [u8; 8] = [7, 8, 10, 12, 13, 12, 10, 8];

// ---------------------------------------------------------------------------
// Bool bitmap (current implementation)
// ---------------------------------------------------------------------------

struct BoolFixture {
    sources: Vec<u16>,
    targets: Vec<u16>,
    activation: Vec<i8>,
    charge: Vec<u8>,
    threshold: Vec<u8>,
    channel: Vec<u8>,
    polarity: Vec<i8>,
    incoming: Vec<i16>,
    dirty_set: Vec<u16>,
    dirty_member: Vec<bool>,  // ← H bytes
    active: Vec<u16>,
    input: Vec<i32>,
    n: usize,
}

// ---------------------------------------------------------------------------
// Bitset bitmap (proposed)
// ---------------------------------------------------------------------------

struct BitsetFixture {
    sources: Vec<u16>,
    targets: Vec<u16>,
    activation: Vec<i8>,
    charge: Vec<u8>,
    threshold: Vec<u8>,
    channel: Vec<u8>,
    polarity: Vec<i8>,
    incoming: Vec<i16>,
    dirty_set: Vec<u16>,
    dirty_bits: Vec<u64>,  // ← H/64 × 8 bytes
    active: Vec<u16>,
    input: Vec<i32>,
    n: usize,
}

#[inline(always)]
fn bit_test(bits: &[u64], idx: usize) -> bool {
    (bits[idx >> 6] >> (idx & 63)) & 1 != 0
}

#[inline(always)]
fn bit_set(bits: &mut [u64], idx: usize) {
    bits[idx >> 6] |= 1u64 << (idx & 63);
}

#[inline(always)]
fn bit_clear(bits: &mut [u64], idx: usize) {
    bits[idx >> 6] &= !(1u64 << (idx & 63));
}

fn build(n: usize, n_edges: usize) -> (BoolFixture, BitsetFixture) {
    let mut graph = ConnectionGraph::new(n);
    let mut rng: u64 = 42;
    for _ in 0..n_edges {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let src = ((rng >> 32) % n as u64) as u16;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let tgt = ((rng >> 32) % n as u64) as u16;
        graph.add_edge(src, tgt);
    }
    let (src, tgt) = graph.edge_endpoints_pub();

    let mut input = vec![0i32; n];
    // Sparse input: 3 neurons
    if n > 200 { input[0] = 1; input[100] = 1; input[200] = 1; }
    else if n > 0 { input[0] = 1; }

    let bf = BoolFixture {
        sources: src.to_vec(), targets: tgt.to_vec(),
        activation: vec![0; n], charge: vec![0; n],
        threshold: vec![6; n], channel: vec![1; n],
        polarity: vec![1; n], incoming: vec![0; n],
        dirty_set: Vec::with_capacity(n),
        dirty_member: vec![false; n],
        active: Vec::with_capacity(n),
        input: input.clone(), n,
    };

    let bs = BitsetFixture {
        sources: src.to_vec(), targets: tgt.to_vec(),
        activation: vec![0; n], charge: vec![0; n],
        threshold: vec![6; n], channel: vec![1; n],
        polarity: vec![1; n], incoming: vec![0; n],
        dirty_set: Vec::with_capacity(n),
        dirty_bits: vec![0u64; (n + 63) / 64],
        active: Vec::with_capacity(n),
        input: input.clone(), n,
    };

    (bf, bs)
}

fn reset_bool(f: &mut BoolFixture) {
    f.activation.fill(0); f.charge.fill(0);
    for &idx in &f.dirty_set { f.dirty_member[idx as usize] = false; }
    f.dirty_set.clear();
}

fn reset_bitset(f: &mut BitsetFixture) {
    f.activation.fill(0); f.charge.fill(0);
    for &idx in &f.dirty_set { bit_clear(&mut f.dirty_bits, idx as usize); }
    f.dirty_set.clear();
}

// ---------------------------------------------------------------------------
// Sparse tick: Vec<bool> dirty_member
// ---------------------------------------------------------------------------

fn propagate_bool(f: &mut BoolFixture) {
    let n = f.n;
    for tick in 0..TICKS {
        // Decay
        if tick % 6 == 0 {
            for &idx in &f.dirty_set {
                f.charge[idx as usize] = f.charge[idx as usize].saturating_sub(1);
            }
        }
        // Input
        if tick < 2 {
            for i in 0..n {
                if f.input[i] != 0 {
                    f.activation[i] = f.activation[i].saturating_add(f.input[i] as i8);
                    if !f.dirty_member[i] {
                        f.dirty_member[i] = true;
                        f.dirty_set.push(i as u16);
                    }
                }
            }
        }
        // Scatter (edge list, skip inactive sources)
        let scatter_end = f.dirty_set.len();
        for di in 0..scatter_end {
            let idx = f.dirty_set[di] as usize;
            let act = f.activation[idx];
            if act == 0 { continue; }
            // Simple: iterate all edges, check source match (no CSR here)
            for ei in 0..f.sources.len() {
                if f.sources[ei] as usize == idx {
                    let tgt = f.targets[ei] as usize;
                    f.incoming[tgt] += act as i16;
                    if !f.dirty_member[tgt] {
                        f.dirty_member[tgt] = true;
                        f.dirty_set.push(tgt as u16);
                    }
                }
            }
        }
        // Charge accum
        for &idx in &f.dirty_set {
            let i = idx as usize;
            let val = (f.charge[i] as i16) + f.incoming[i];
            f.charge[i] = val.clamp(0, MAX_CHARGE as i16) as u8;
        }
        // Active set
        f.active.clear();
        for &idx in &f.dirty_set {
            if f.charge[idx as usize] > 0 { f.active.push(idx); }
        }
        // Clear activation
        for &idx in &f.dirty_set { f.activation[idx as usize] = 0; }
        // Spike
        let pt = tick % 8;
        for &idx16 in &f.active {
            let idx = idx16 as usize;
            let ci = f.channel[idx] as usize;
            let pm: u16 = if (1..=8).contains(&ci) { PHASE_BASE[(pt + 9 - ci) & 7] as u16 } else { 10 };
            if f.charge[idx] as u16 * 10 >= (f.threshold[idx] as u16 + 1) * pm {
                f.activation[idx] = f.polarity[idx];
                f.charge[idx] = 0;
            }
        }
        // Clear incoming
        for &idx in &f.dirty_set { f.incoming[idx as usize] = 0; }
        // Prune
        let charge = &f.charge;
        let activation = &f.activation;
        let member = &mut f.dirty_member;
        f.dirty_set.retain(|&idx| {
            let i = idx as usize;
            if charge[i] > 0 || activation[i] != 0 { true }
            else { member[i] = false; false }
        });
    }
}

// ---------------------------------------------------------------------------
// Sparse tick: Vec<u64> bitset dirty_member
// ---------------------------------------------------------------------------

fn propagate_bitset(f: &mut BitsetFixture) {
    let n = f.n;
    for tick in 0..TICKS {
        if tick % 6 == 0 {
            for &idx in &f.dirty_set {
                f.charge[idx as usize] = f.charge[idx as usize].saturating_sub(1);
            }
        }
        if tick < 2 {
            for i in 0..n {
                if f.input[i] != 0 {
                    f.activation[i] = f.activation[i].saturating_add(f.input[i] as i8);
                    if !bit_test(&f.dirty_bits, i) {
                        bit_set(&mut f.dirty_bits, i);
                        f.dirty_set.push(i as u16);
                    }
                }
            }
        }
        let scatter_end = f.dirty_set.len();
        for di in 0..scatter_end {
            let idx = f.dirty_set[di] as usize;
            let act = f.activation[idx];
            if act == 0 { continue; }
            for ei in 0..f.sources.len() {
                if f.sources[ei] as usize == idx {
                    let tgt = f.targets[ei] as usize;
                    f.incoming[tgt] += act as i16;
                    if !bit_test(&f.dirty_bits, tgt) {
                        bit_set(&mut f.dirty_bits, tgt);
                        f.dirty_set.push(tgt as u16);
                    }
                }
            }
        }
        for &idx in &f.dirty_set {
            let i = idx as usize;
            let val = (f.charge[i] as i16) + f.incoming[i];
            f.charge[i] = val.clamp(0, MAX_CHARGE as i16) as u8;
        }
        f.active.clear();
        for &idx in &f.dirty_set {
            if f.charge[idx as usize] > 0 { f.active.push(idx); }
        }
        for &idx in &f.dirty_set { f.activation[idx as usize] = 0; }
        let pt = tick % 8;
        for &idx16 in &f.active {
            let idx = idx16 as usize;
            let ci = f.channel[idx] as usize;
            let pm: u16 = if (1..=8).contains(&ci) { PHASE_BASE[(pt + 9 - ci) & 7] as u16 } else { 10 };
            if f.charge[idx] as u16 * 10 >= (f.threshold[idx] as u16 + 1) * pm {
                f.activation[idx] = f.polarity[idx];
                f.charge[idx] = 0;
            }
        }
        for &idx in &f.dirty_set { f.incoming[idx as usize] = 0; }
        let charge = &f.charge;
        let activation = &f.activation;
        let bits = &mut f.dirty_bits;
        f.dirty_set.retain(|&idx| {
            let i = idx as usize;
            if charge[i] > 0 || activation[i] != 0 { true }
            else { bit_clear(bits, i); false }
        });
    }
}

struct Case { name: &'static str, n: usize, edges: usize, iters: usize }

fn run(c: &Case) {
    println!("\n=== {} | H={}, {} edges ===", c.name, c.n, c.edges);
    let (mut bf, mut bs) = build(c.n, c.edges);

    let bool_bytes = c.n;
    let bits_bytes = (c.n + 63) / 64 * 8;
    println!("  dirty_member: Vec<bool>={}KB  Vec<u64>={}KB  ({:.0}x smaller)",
        bool_bytes / 1024, bits_bytes / 1024, bool_bytes as f64 / bits_bytes as f64);

    let t_bool = timed_run("Vec<bool> dirty_member", c.iters, || {
        reset_bool(&mut bf); propagate_bool(black_box(&mut bf));
    });
    let t_bits = timed_run("Vec<u64> bitset", c.iters, || {
        reset_bitset(&mut bs); propagate_bitset(black_box(&mut bs));
    });

    let d = (t_bits.median_ns - t_bool.median_ns) / t_bool.median_ns * 100.0;
    println!("\n    Vec<bool>: {:>10.0} ns", t_bool.median_ns);
    println!("    bitset:    {:>10.0} ns ({:+.1}%)", t_bits.median_ns, d);
}

fn main() {
    print_harness_header();
    for c in &[
        Case { name: "H=4096 sparse", n: 4096, edges: 30, iters: 5000 },
        Case { name: "H=16384 sparse", n: 16384, edges: 50, iters: 3000 },
        Case { name: "H=65536 sparse", n: 65536, edges: 100, iters: 1000 },
        Case { name: "H=100000 sparse", n: 100_000, edges: 100, iters: 500 },
    ] { run(c); }
}
