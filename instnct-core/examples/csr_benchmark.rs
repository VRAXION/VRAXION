//! Benchmark: current scatter-add vs CSR u16 vs skip-inactive vs both.
//!
//! Run: cargo run --example csr_benchmark --release

use instnct_core::Network;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

const TICKS: usize = 12;
const WARMUP: usize = 20;
const ITERS: usize = 100;

// ---- CSR builder ----

struct CsrGraph {
    offsets: Vec<u32>, // H+1 entries: neuron i's edges are targets[offsets[i]..offsets[i+1]]
    targets: Vec<u16>, // compact target indices
    neuron_count: usize,
}

impl CsrGraph {
    fn from_network(net: &Network) -> Self {
        let neuron_count = net.neuron_count();
        let mut adjacency: Vec<Vec<u16>> = vec![vec![]; neuron_count];
        for edge in net.graph().iter_edges() {
            adjacency[edge.source as usize].push(edge.target);
        }
        let mut offsets = Vec::with_capacity(neuron_count + 1);
        let mut targets = Vec::with_capacity(net.edge_count());
        let mut offset = 0u32;
        for adj in &adjacency {
            offsets.push(offset);
            for &t in adj {
                targets.push(t);
            }
            offset += adj.len() as u32;
        }
        offsets.push(offset);
        CsrGraph {
            offsets,
            targets,
            neuron_count,
        }
    }

    fn memory_bytes(&self) -> usize {
        self.offsets.len() * 4 + self.targets.len() * 2
    }
}

// ---- Propagation variants ----

/// A: Current approach — scan all edges, usize sources/targets
#[allow(clippy::too_many_arguments)]
fn propagate_current(
    activation: &mut [i32],
    charge: &mut [u32],
    input: &[i32],
    edge_src: &[usize],
    edge_tgt: &[usize],
    threshold: &[u32],
    channel: &[u8],
    polarity: &[i32],
    neuron_count: usize,
) {
    let phase_base: [u8; 8] = [7, 8, 10, 12, 13, 12, 10, 8];
    let mut incoming = vec![0i32; neuron_count];

    for tick in 0..TICKS {
        if tick % 6 == 0 {
            for c in charge.iter_mut() {
                *c = c.saturating_sub(1);
            }
        }
        if tick < 2 {
            for (a, &iv) in activation.iter_mut().zip(input.iter()) {
                *a += iv;
            }
        }
        incoming.fill(0);
        for i in 0..edge_src.len() {
            incoming[edge_tgt[i]] += activation[edge_src[i]];
        }
        for (c, &s) in charge.iter_mut().zip(incoming.iter()) {
            *c = c.saturating_add_signed(s).min(15);
        }
        let phase_tick = tick % 8;
        for i in 0..neuron_count {
            let ch = channel[i] as usize;
            let pm: u16 = if (1..=8).contains(&ch) {
                phase_base[(phase_tick + 9 - ch) & 7] as u16
            } else {
                10
            };
            if charge[i] as u16 * 10 >= (threshold[i] as u16 + 1) * pm {
                activation[i] = polarity[i];
                charge[i] = 0;
            } else {
                activation[i] = 0;
            }
        }
    }
}

/// B: Skip-inactive only (still usize edges, but skip zero activation sources)
fn propagate_skip_inactive(
    activation: &mut [i32],
    charge: &mut [u32],
    input: &[i32],
    csr: &CsrGraph,
    threshold: &[u32],
    channel: &[u8],
    polarity: &[i32],
) {
    let phase_base: [u8; 8] = [7, 8, 10, 12, 13, 12, 10, 8];
    let neuron_count = csr.neuron_count;
    let mut incoming = vec![0i32; neuron_count];

    for tick in 0..TICKS {
        if tick % 6 == 0 {
            for c in charge.iter_mut() {
                *c = c.saturating_sub(1);
            }
        }
        if tick < 2 {
            for (a, &iv) in activation.iter_mut().zip(input.iter()) {
                *a += iv;
            }
        }
        incoming.fill(0);
        // SKIP-INACTIVE: only process edges from neurons that fired
        #[allow(clippy::needless_range_loop)]
        for neuron in 0..neuron_count {
            let act = activation[neuron];
            if act == 0 {
                continue; // skip ~80% of neurons
            }
            let start = csr.offsets[neuron] as usize;
            let end = csr.offsets[neuron + 1] as usize;
            for &target in &csr.targets[start..end] {
                incoming[target as usize] += act;
            }
        }
        for (c, &s) in charge.iter_mut().zip(incoming.iter()) {
            *c = c.saturating_add_signed(s).min(15);
        }
        let phase_tick = tick % 8;
        for i in 0..neuron_count {
            let ch = channel[i] as usize;
            let pm: u16 = if (1..=8).contains(&ch) {
                phase_base[(phase_tick + 9 - ch) & 7] as u16
            } else {
                10
            };
            if charge[i] as u16 * 10 >= (threshold[i] as u16 + 1) * pm {
                activation[i] = polarity[i];
                charge[i] = 0;
            } else {
                activation[i] = 0;
            }
        }
    }
}

/// C: CSR u16 only (compact storage, but scan ALL edges including inactive)
fn propagate_csr_scan_all(
    activation: &mut [i32],
    charge: &mut [u32],
    input: &[i32],
    csr: &CsrGraph,
    threshold: &[u32],
    channel: &[u8],
    polarity: &[i32],
) {
    let phase_base: [u8; 8] = [7, 8, 10, 12, 13, 12, 10, 8];
    let neuron_count = csr.neuron_count;
    let mut incoming = vec![0i32; neuron_count];

    for tick in 0..TICKS {
        if tick % 6 == 0 {
            for c in charge.iter_mut() {
                *c = c.saturating_sub(1);
            }
        }
        if tick < 2 {
            for (a, &iv) in activation.iter_mut().zip(input.iter()) {
                *a += iv;
            }
        }
        incoming.fill(0);
        // CSR scan-all: same work as current, but u16 targets + CSR layout
        #[allow(clippy::needless_range_loop)]
        for neuron in 0..neuron_count {
            let act = activation[neuron];
            let start = csr.offsets[neuron] as usize;
            let end = csr.offsets[neuron + 1] as usize;
            for &target in &csr.targets[start..end] {
                incoming[target as usize] += act;
            }
        }
        for (c, &s) in charge.iter_mut().zip(incoming.iter()) {
            *c = c.saturating_add_signed(s).min(15);
        }
        let phase_tick = tick % 8;
        for i in 0..neuron_count {
            let ch = channel[i] as usize;
            let pm: u16 = if (1..=8).contains(&ch) {
                phase_base[(phase_tick + 9 - ch) & 7] as u16
            } else {
                10
            };
            if charge[i] as u16 * 10 >= (threshold[i] as u16 + 1) * pm {
                activation[i] = polarity[i];
                charge[i] = 0;
            } else {
                activation[i] = 0;
            }
        }
    }
}

// ---- Benchmark runner ----

fn bench(name: &str, iters: usize, mut body: impl FnMut()) -> f64 {
    for _ in 0..WARMUP {
        body();
    }
    let start = Instant::now();
    for _ in 0..iters {
        body();
    }
    let ns = start.elapsed().as_nanos() as f64 / iters as f64;
    println!("  {name:<35} {ns:>10.0} ns");
    ns
}

fn main() {
    let sizes: &[(usize, u64)] = &[(256, 5), (512, 5), (1024, 2), (2048, 1), (4096, 1)];

    for &(neuron_count, density_pct) in sizes {
        let mut net = Network::new(neuron_count);
        let mut rng = StdRng::seed_from_u64(42);
        let target_edges = (neuron_count as u64 * neuron_count as u64 * density_pct / 100) as usize;
        for _ in 0..target_edges * 3 {
            net.mutate_add_edge(&mut rng);
            if net.edge_count() >= target_edges {
                break;
            }
        }
        for i in 0..neuron_count {
            net.threshold_mut()[i] = rng.gen_range(0..=15);
            net.channel_mut()[i] = rng.gen_range(1..=8);
            if rng.gen_ratio(1, 10) {
                net.polarity_mut()[i] = -1;
            }
        }

        let csr = CsrGraph::from_network(&net);

        // Extract raw edge arrays for current variant
        let edge_src: Vec<usize> = net
            .graph()
            .iter_edges()
            .map(|e| e.source as usize)
            .collect();
        let edge_tgt: Vec<usize> = net
            .graph()
            .iter_edges()
            .map(|e| e.target as usize)
            .collect();

        let threshold = net.threshold().to_vec();
        let channel = net.channel().to_vec();
        let polarity = net.polarity().to_vec();
        let input = {
            let mut v = vec![0i32; neuron_count];
            v[0] = 1;
            v
        };

        let current_mem = edge_src.len() * 8 * 2;
        let csr_mem = csr.memory_bytes();

        println!(
            "\n=== H={neuron_count}, {density_pct}% density, {} edges ===",
            net.edge_count()
        );
        println!(
            "  Memory: current={} KB, CSR u16={} KB ({}x smaller)",
            current_mem / 1024,
            csr_mem / 1024,
            current_mem / csr_mem.max(1)
        );

        let baseline = bench("A: current (usize, scan all)", ITERS, || {
            let mut act = vec![0i32; neuron_count];
            let mut chg = vec![0u32; neuron_count];
            propagate_current(
                &mut act,
                &mut chg,
                &input,
                &edge_src,
                &edge_tgt,
                &threshold,
                &channel,
                &polarity,
                neuron_count,
            );
        });

        let skip = bench("B: skip-inactive (CSR, skip zero)", ITERS, || {
            let mut act = vec![0i32; neuron_count];
            let mut chg = vec![0u32; neuron_count];
            propagate_skip_inactive(
                &mut act, &mut chg, &input, &csr, &threshold, &channel, &polarity,
            );
        });

        let csr_all = bench("C: CSR u16 (scan all, compact mem)", ITERS, || {
            let mut act = vec![0i32; neuron_count];
            let mut chg = vec![0u32; neuron_count];
            propagate_csr_scan_all(
                &mut act, &mut chg, &input, &csr, &threshold, &channel, &polarity,
            );
        });

        // D is same as B (CSR + skip-inactive is the same implementation)
        println!("  ---");
        println!(
            "  B vs A: {:+.1}% (skip-inactive)",
            (skip - baseline) / baseline * 100.0
        );
        println!(
            "  C vs A: {:+.1}% (CSR compact only)",
            (csr_all - baseline) / baseline * 100.0
        );
    }
}
