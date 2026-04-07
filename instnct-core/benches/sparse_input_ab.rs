//! A/B: sparse input (index list) vs dense input (Vec<i32>).
//!
//! The last O(H) in the sparse tick is input injection:
//!   for i in 0..H { if input[i] != 0 { ... } }
//!
//! With sparse input (list of active neuron indices), this becomes O(k)
//! where k = number of active input neurons (~20 for SDR 20%).
//!
//! Tests the full propagation pipeline with both input formats.

mod common;

use common::{print_harness_header, timed_run};
use instnct_core::ConnectionGraph;
use std::hint::black_box;

const TICKS: usize = 12;
const MAX_CHARGE: u8 = 15;
const PHASE_BASE: [u8; 8] = [7, 8, 10, 12, 13, 12, 10, 8];

struct Fixture {
    n: usize,
    // Edge data (CSR-like: source-indexed)
    csr_offsets: Vec<u32>,
    csr_targets: Vec<u16>,
    // Neuron data
    activation: Vec<i8>,
    charge: Vec<u8>,
    threshold: Vec<u8>,
    channel: Vec<u8>,
    polarity: Vec<i8>,
    incoming: Vec<i16>,
    // Dirty set
    dirty_set: Vec<u16>,
    dirty_member: Vec<bool>,
    active: Vec<u16>,
    // Dense input
    dense_input: Vec<i32>,
    // Sparse input (index list of active neurons + values)
    sparse_indices: Vec<u16>,
    sparse_values: Vec<i8>,
}

fn build_fixture(n: usize, n_edges: usize) -> Fixture {
    let mut graph = ConnectionGraph::new(n);
    let mut rng: u64 = 42;
    for _ in 0..n_edges {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let src = ((rng >> 32) % n as u64) as u16;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let tgt = ((rng >> 32) % n as u64) as u16;
        graph.add_edge(src, tgt);
    }

    // Build CSR
    let mut csr_offsets = vec![0u32; n + 1];
    let mut csr_targets = Vec::new();
    let mut out_count = vec![0u32; n];
    for edge in graph.iter_edges() {
        out_count[edge.source as usize] += 1;
    }
    let mut offset = 0u32;
    for i in 0..n {
        csr_offsets[i] = offset;
        offset += out_count[i];
    }
    csr_offsets[n] = offset;
    csr_targets.resize(graph.edge_count(), 0u16);
    let mut write_pos = csr_offsets[..n].to_vec();
    for edge in graph.iter_edges() {
        let pos = write_pos[edge.source as usize] as usize;
        csr_targets[pos] = edge.target;
        write_pos[edge.source as usize] += 1;
    }

    // Dense input: ~5% active (SDR-like)
    let mut dense_input = vec![0i32; n];
    let mut sparse_indices = Vec::new();
    let mut sparse_values = Vec::new();
    let mut rng2: u64 = 99;
    for i in 0..n {
        rng2 = rng2.wrapping_mul(6364136223846793005).wrapping_add(1);
        if (rng2 >> 32) % 20 == 0 {
            dense_input[i] = 1;
            sparse_indices.push(i as u16);
            sparse_values.push(1i8);
        }
    }

    Fixture {
        n,
        csr_offsets, csr_targets,
        activation: vec![0; n], charge: vec![0; n],
        threshold: vec![6; n], channel: vec![1; n],
        polarity: vec![1; n], incoming: vec![0; n],
        dirty_set: Vec::with_capacity(n),
        dirty_member: vec![false; n],
        active: Vec::with_capacity(n),
        dense_input, sparse_indices, sparse_values,
    }
}

fn reset(f: &mut Fixture) {
    f.activation.fill(0); f.charge.fill(0);
    for &idx in &f.dirty_set { f.dirty_member[idx as usize] = false; }
    f.dirty_set.clear();
}

// ---------------------------------------------------------------------------
// Dense input: for i in 0..H { if input[i] != 0 ... }  — O(H)
// ---------------------------------------------------------------------------

fn propagate_dense_input(f: &mut Fixture) {
    let n = f.n;
    let incoming = &mut f.incoming;

    for tick in 0..TICKS {
        if tick % 6 == 0 {
            for &idx in &f.dirty_set { f.charge[idx as usize] = f.charge[idx as usize].saturating_sub(1); }
        }

        // INPUT: O(H) — scan all neurons
        if tick < 2 {
            for i in 0..n {
                if f.dense_input[i] != 0 {
                    f.activation[i] = f.activation[i].saturating_add(f.dense_input[i] as i8);
                    if !f.dirty_member[i] { f.dirty_member[i] = true; f.dirty_set.push(i as u16); }
                }
            }
        }

        let scatter_end = f.dirty_set.len();
        for di in 0..scatter_end {
            let idx = f.dirty_set[di] as usize;
            let act = f.activation[idx];
            if act == 0 { continue; }
            let start = f.csr_offsets[idx] as usize;
            let end = f.csr_offsets[idx + 1] as usize;
            for &target in &f.csr_targets[start..end] {
                incoming[target as usize] += act as i16;
                if !f.dirty_member[target as usize] { f.dirty_member[target as usize] = true; f.dirty_set.push(target); }
            }
        }

        for &idx in &f.dirty_set {
            let i = idx as usize;
            let val = (f.charge[i] as i16) + incoming[i]; f.charge[i] = val.clamp(0, MAX_CHARGE as i16) as u8;
        }
        f.active.clear();
        for &idx in &f.dirty_set { if f.charge[idx as usize] > 0 { f.active.push(idx); } }
        for &idx in &f.dirty_set { f.activation[idx as usize] = 0; }

        let pt = tick % 8;
        for &idx16 in &f.active {
            let idx = idx16 as usize;
            let ci = f.channel[idx] as usize;
            let pm: u16 = if (1..=8).contains(&ci) { PHASE_BASE[(pt + 9 - ci) & 7] as u16 } else { 10 };
            if f.charge[idx] as u16 * 10 >= (f.threshold[idx] as u16 + 1) * pm {
                f.activation[idx] = f.polarity[idx]; f.charge[idx] = 0;
            }
        }

        for &idx in &f.dirty_set { incoming[idx as usize] = 0; }
        let charge = &f.charge; let activation = &f.activation; let member = &mut f.dirty_member;
        f.dirty_set.retain(|&idx| { let i = idx as usize; if charge[i] > 0 || activation[i] != 0 { true } else { member[i] = false; false } });
    }
}

// ---------------------------------------------------------------------------
// Sparse input: iterate index list — O(k) where k = active inputs
// ---------------------------------------------------------------------------

fn propagate_sparse_input(f: &mut Fixture) {
    let n = f.n;
    let incoming = &mut f.incoming;

    for tick in 0..TICKS {
        if tick % 6 == 0 {
            for &idx in &f.dirty_set { f.charge[idx as usize] = f.charge[idx as usize].saturating_sub(1); }
        }

        // INPUT: O(k) — only iterate sparse index list
        if tick < 2 {
            for (&idx16, &val) in f.sparse_indices.iter().zip(f.sparse_values.iter()) {
                let i = idx16 as usize;
                f.activation[i] = f.activation[i].saturating_add(val);
                if !f.dirty_member[i] { f.dirty_member[i] = true; f.dirty_set.push(idx16); }
            }
        }

        let scatter_end = f.dirty_set.len();
        for di in 0..scatter_end {
            let idx = f.dirty_set[di] as usize;
            let act = f.activation[idx];
            if act == 0 { continue; }
            let start = f.csr_offsets[idx] as usize;
            let end = f.csr_offsets[idx + 1] as usize;
            for &target in &f.csr_targets[start..end] {
                incoming[target as usize] += act as i16;
                if !f.dirty_member[target as usize] { f.dirty_member[target as usize] = true; f.dirty_set.push(target); }
            }
        }

        for &idx in &f.dirty_set {
            let i = idx as usize;
            let val = (f.charge[i] as i16) + incoming[i]; f.charge[i] = val.clamp(0, MAX_CHARGE as i16) as u8;
        }
        f.active.clear();
        for &idx in &f.dirty_set { if f.charge[idx as usize] > 0 { f.active.push(idx); } }
        for &idx in &f.dirty_set { f.activation[idx as usize] = 0; }

        let pt = tick % 8;
        for &idx16 in &f.active {
            let idx = idx16 as usize;
            let ci = f.channel[idx] as usize;
            let pm: u16 = if (1..=8).contains(&ci) { PHASE_BASE[(pt + 9 - ci) & 7] as u16 } else { 10 };
            if f.charge[idx] as u16 * 10 >= (f.threshold[idx] as u16 + 1) * pm {
                f.activation[idx] = f.polarity[idx]; f.charge[idx] = 0;
            }
        }

        for &idx in &f.dirty_set { incoming[idx as usize] = 0; }
        let charge = &f.charge; let activation = &f.activation; let member = &mut f.dirty_member;
        f.dirty_set.retain(|&idx| { let i = idx as usize; if charge[i] > 0 || activation[i] != 0 { true } else { member[i] = false; false } });
    }
}

struct Case { name: &'static str, n: usize, edges: usize, iters: usize }

fn run(c: &Case) {
    println!("\n=== {} | H={}, {} edges ===", c.name, c.n, c.edges);
    let mut f = build_fixture(c.n, c.edges);
    println!("  active inputs: {} / {} ({:.1}%)",
        f.sparse_indices.len(), c.n, f.sparse_indices.len() as f64 / c.n as f64 * 100.0);

    let t_dense = timed_run("dense input O(H)", c.iters, || {
        reset(&mut f); propagate_dense_input(black_box(&mut f));
    });
    let t_sparse = timed_run("sparse input O(k)", c.iters, || {
        reset(&mut f); propagate_sparse_input(black_box(&mut f));
    });

    let d = (t_sparse.median_ns - t_dense.median_ns) / t_dense.median_ns * 100.0;
    println!("\n    dense O(H):  {:>10.0} ns  ({} input scan per token)", t_dense.median_ns, c.n * 2);
    println!("    sparse O(k): {:>10.0} ns  ({} input indices)", t_sparse.median_ns, f.sparse_indices.len());
    println!("    delta: {:+.1}%", d);
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
