//! A/B: spike loop skip-inactive — only check neurons with charge > 0.
//!
//! For sparse networks (28 edges, 0% fire rate), the spike loop iterates
//! all 4096 neurons but only ~28 have any charge. Skip the rest.
//!
//! Approach: maintain an "active set" — neurons whose charge > 0 after
//! the charge accumulation phase. Only run spike decision on those.

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
    threshold: Vec<u8>,
    channel: Vec<u8>,
    polarity: Vec<i8>,
    incoming: Vec<i16>,
    active_neurons: Vec<u16>, // scratch: neurons with charge > 0
    input: Vec<i32>,
    n: usize,
}

fn build_fixture(n: usize, edge_prob_pct: u64) -> Fixture {
    let graph = build_graph(n, edge_prob_pct);
    let (src, tgt) = graph.edge_endpoints_pub();
    let mut input = vec![0i32; n];
    if n > 0 { input[0] = 1; }
    Fixture {
        sources: src.to_vec(), targets: tgt.to_vec(),
        activation: vec![0; n], charge: vec![0; n],
        threshold: vec![6; n], channel: vec![1; n],
        polarity: vec![1; n], incoming: vec![0; n],
        active_neurons: Vec::with_capacity(n),
        input, n,
    }
}

// Also test with very sparse edges (like the sweep showed)
fn build_sparse_fixture(n: usize, edge_count: usize) -> Fixture {
    let mut f = build_fixture(n, 0);
    // Manually add edges: chain from 0 → 1 → 2 → ... → edge_count
    let mut rng_state: u64 = 42;
    for _ in 0..edge_count {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let src = (rng_state >> 32) as u16 % n as u16;
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let tgt = (rng_state >> 32) as u16 % n as u16;
        if src != tgt {
            f.sources.push(src);
            f.targets.push(tgt);
        }
    }
    f
}

fn reset(f: &mut Fixture) {
    f.activation.fill(0);
    f.charge.fill(0);
}

// ---------------------------------------------------------------------------
// Baseline: check ALL neurons in spike loop
// ---------------------------------------------------------------------------

fn propagate_baseline(f: &mut Fixture) {
    let n = f.n;
    for tick in 0..TICKS {
        if tick % 6 == 0 {
            for ch in f.charge.iter_mut() { *ch = ch.saturating_sub(1); }
        }
        if tick < 2 {
            for (a, &i) in f.activation.iter_mut().zip(f.input.iter()) {
                *a = a.saturating_add(i as i8);
            }
        }

        // CSR-style skip-inactive scatter-add
        f.incoming[..n].fill(0);
        for i in 0..f.sources.len() {
            let src = f.sources[i] as usize;
            let act = f.activation[src];
            if act != 0 {
                f.incoming[f.targets[i] as usize] += act as i16;
            }
        }

        for idx in 0..n {
            let val = (f.charge[idx] as i16) + f.incoming[idx];
            f.charge[idx] = val.clamp(0, MAX_CHARGE as i16) as u8;
        }

        // SPIKE: check ALL neurons
        let pt = tick % 8;
        for idx in 0..n {
            let ci = f.channel[idx] as usize;
            let pm: u16 = if (1..=8).contains(&ci) {
                PHASE_BASE[(pt + 9 - ci) & 7] as u16
            } else { 10 };
            let cx10 = f.charge[idx] as u16 * 10;
            let tx10 = (f.threshold[idx] as u16 + 1) * pm;
            if cx10 >= tx10 {
                f.activation[idx] = f.polarity[idx];
                f.charge[idx] = 0;
            } else {
                f.activation[idx] = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Skip-inactive: only spike-check neurons with charge > 0
// ---------------------------------------------------------------------------

fn propagate_skip(f: &mut Fixture) {
    let n = f.n;
    for tick in 0..TICKS {
        if tick % 6 == 0 {
            for ch in f.charge.iter_mut() { *ch = ch.saturating_sub(1); }
        }
        if tick < 2 {
            for (a, &i) in f.activation.iter_mut().zip(f.input.iter()) {
                *a = a.saturating_add(i as i8);
            }
        }

        // CSR-style skip-inactive scatter-add (same as baseline)
        f.incoming[..n].fill(0);
        for i in 0..f.sources.len() {
            let src = f.sources[i] as usize;
            let act = f.activation[src];
            if act != 0 {
                f.incoming[f.targets[i] as usize] += act as i16;
            }
        }

        // Build active set: neurons that received any signal OR had prior charge
        f.active_neurons.clear();
        for idx in 0..n {
            if f.incoming[idx] != 0 || f.charge[idx] != 0 {
                f.active_neurons.push(idx as u16);
            }
        }

        // Charge accumulation: only active
        for &idx16 in &f.active_neurons {
            let idx = idx16 as usize;
            let val = (f.charge[idx] as i16) + f.incoming[idx];
            f.charge[idx] = val.clamp(0, MAX_CHARGE as i16) as u8;
        }

        // SPIKE: only active neurons (charge > 0 or just received signal)
        let pt = tick % 8;
        // First: clear ALL activation (silent by default)
        f.activation.fill(0);
        // Then: only check active neurons
        for &idx16 in &f.active_neurons {
            let idx = idx16 as usize;
            let ci = f.channel[idx] as usize;
            let pm: u16 = if (1..=8).contains(&ci) {
                PHASE_BASE[(pt + 9 - ci) & 7] as u16
            } else { 10 };
            let cx10 = f.charge[idx] as u16 * 10;
            let tx10 = (f.threshold[idx] as u16 + 1) * pm;
            if cx10 >= tx10 {
                f.activation[idx] = f.polarity[idx];
                f.charge[idx] = 0;
            }
            // else: activation already 0 from fill
        }
    }
}

// ---------------------------------------------------------------------------

struct Case {
    name: &'static str,
    n: usize,
    edges: usize,       // 0 = use edge_prob_pct
    edge_prob_pct: u64,
    iters: usize,
}

fn run(c: &Case) {
    println!("\n=== {} | H={}, {} edges ===", c.name, c.n,
        if c.edges > 0 { format!("{}", c.edges) } else { format!("~{}%", c.edge_prob_pct) });

    let mut f = if c.edges > 0 {
        build_sparse_fixture(c.n, c.edges)
    } else {
        build_fixture(c.n, c.edge_prob_pct)
    };
    println!("  actual edges: {}", f.sources.len());

    let ctrl = timed_run("baseline (all neurons)", c.iters, || {
        reset(&mut f); propagate_baseline(black_box(&mut f));
    });
    let ctrl2 = timed_run("baseline #2", c.iters, || {
        reset(&mut f); propagate_baseline(black_box(&mut f));
    });
    let noise = ((ctrl2.median_ns - ctrl.median_ns) / ctrl.median_ns * 100.0).abs();
    println!("  NOISE: {noise:.1}%");

    let t_skip = timed_run("skip-inactive spike", c.iters, || {
        reset(&mut f); propagate_skip(black_box(&mut f));
    });

    let base = ctrl.median_ns;
    let d = (t_skip.median_ns - base) / base * 100.0;
    println!("\n    baseline:       {:>10.0} ns", base);
    println!("    skip-inactive:  {:>10.0} ns ({:+.1}%)", t_skip.median_ns, d);
}

fn main() {
    print_harness_header();
    for c in &[
        // The sweep scenario: huge sparse network
        Case { name: "H=4096, 30 edges (sweep winner)", n: 4096, edges: 30, edge_prob_pct: 0, iters: 5000 },
        Case { name: "H=4096, 100 edges", n: 4096, edges: 100, edge_prob_pct: 0, iters: 5000 },
        Case { name: "H=4096, 1000 edges", n: 4096, edges: 1000, edge_prob_pct: 0, iters: 2000 },
        Case { name: "H=8192, 50 edges", n: 8192, edges: 50, edge_prob_pct: 0, iters: 3000 },
        // Dense for comparison
        Case { name: "H=4096, 1% dense", n: 4096, edges: 0, edge_prob_pct: 1, iters: 200 },
        Case { name: "H=256, 5% dense", n: 256, edges: 0, edge_prob_pct: 5, iters: 5000 },
    ] { run(c); }
}
