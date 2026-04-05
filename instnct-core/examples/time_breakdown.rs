//! Time breakdown: measure each phase of the forward pass separately.
//!
//! Run: cargo run --example time_breakdown --release

use instnct_core::Network;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;
use std::time::Instant;

const TICKS: usize = 12;
const WARMUP: usize = 50;
const ITERS: usize = 500;
const PHASE_BASE: [u8; 8] = [7, 8, 10, 12, 13, 12, 10, 8];

fn main() {
    let sizes: &[(usize, u64)] = &[(256, 5), (512, 5), (1024, 2), (4096, 1)];

    for &(neuron_count, density_pct) in sizes {
        let mut net = Network::new(neuron_count);
        let mut rng = StdRng::seed_from_u64(42);
        let target_edges =
            (neuron_count as u64 * neuron_count as u64 * density_pct / 100) as usize;
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

        let edges = net.edge_count();
        let threshold = net.threshold().to_vec();
        let channel = net.channel().to_vec();
        let polarity = net.polarity().to_vec();
        let input = {
            let mut v = vec![0i32; neuron_count];
            v[0] = 1;
            v
        };

        // Build CSR once
        let mut csr_offsets = vec![0u32; neuron_count + 1];
        let mut csr_targets: Vec<u16> = Vec::with_capacity(edges);
        {
            let mut counts = vec![0u32; neuron_count];
            for edge in net.graph().iter_edges() {
                counts[edge.source as usize] += 1;
            }
            let mut offset = 0u32;
            for &count in &counts {
                csr_offsets.push(offset); // this is wrong, we already have zeros
            }
            // Redo properly
            csr_offsets.clear();
            let mut offset = 0u32;
            for &count in &counts {
                csr_offsets.push(offset);
                offset += count;
            }
            csr_offsets.push(offset);
            csr_targets.resize(edges, 0);
            let mut write_pos = csr_offsets.clone();
            for edge in net.graph().iter_edges() {
                let src = edge.source as usize;
                let pos = write_pos[src] as usize;
                csr_targets[pos] = edge.target;
                write_pos[src] += 1;
            }
        }

        // Allocate buffers
        let mut activation = vec![0i32; neuron_count];
        let mut charge = vec![0u32; neuron_count];
        let mut incoming = vec![0i32; neuron_count];

        // Helper: run full propagation to get realistic activation pattern
        let run_full = |act: &mut [i32], chg: &mut [u32], inc: &mut [i32]| {
            act.fill(0);
            chg.fill(0);
            for tick in 0..TICKS {
                if tick % 6 == 0 {
                    for c in chg.iter_mut() {
                        *c = c.saturating_sub(1);
                    }
                }
                if tick < 2 {
                    for (a, &iv) in act.iter_mut().zip(input.iter()) {
                        *a += iv;
                    }
                }
                inc.fill(0);
                for neuron in 0..neuron_count {
                    let a = act[neuron];
                    if a == 0 { continue; }
                    let start = csr_offsets[neuron] as usize;
                    let end = csr_offsets[neuron + 1] as usize;
                    for &t in &csr_targets[start..end] {
                        inc[t as usize] += a;
                    }
                }
                for (c, &s) in chg.iter_mut().zip(inc.iter()) {
                    *c = c.saturating_add_signed(s).min(15);
                }
                let pt = tick % 8;
                for i in 0..neuron_count {
                    let ch = channel[i] as usize;
                    let pm: u16 = if (1..=8).contains(&ch) {
                        PHASE_BASE[(pt + 9 - ch) & 7] as u16
                    } else { 10 };
                    if chg[i] as u16 * 10 >= (threshold[i] as u16 + 1) * pm {
                        act[i] = polarity[i];
                        chg[i] = 0;
                    } else {
                        act[i] = 0;
                    }
                }
            }
        };

        // Warmup
        for _ in 0..WARMUP {
            run_full(&mut activation, &mut charge, &mut incoming);
        }

        // Measure FULL
        let start = Instant::now();
        for _ in 0..ITERS {
            run_full(
                black_box(&mut activation),
                black_box(&mut charge),
                black_box(&mut incoming),
            );
        }
        let full_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;

        // Measure DECAY only
        let start = Instant::now();
        for _ in 0..ITERS {
            charge.fill(8);
            for _tick in 0..TICKS {
                for c in black_box(&mut charge).iter_mut() {
                    *c = c.saturating_sub(1);
                }
            }
        }
        let decay_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;

        // Measure SCATTER-ADD only (skip-inactive)
        run_full(&mut activation, &mut charge, &mut incoming); // get realistic activation
        let start = Instant::now();
        for _ in 0..ITERS {
            for _tick in 0..TICKS {
                black_box(&mut incoming).fill(0);
                for neuron in 0..neuron_count {
                    let a = black_box(&activation)[neuron];
                    if a == 0 { continue; }
                    let s = csr_offsets[neuron] as usize;
                    let e = csr_offsets[neuron + 1] as usize;
                    for &t in &csr_targets[s..e] {
                        incoming[t as usize] += a;
                    }
                }
            }
        }
        let scatter_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;

        // Measure SPIKE DECISION only
        let start = Instant::now();
        for _ in 0..ITERS {
            for tick in 0..TICKS {
                let pt = tick % 8;
                for i in 0..neuron_count {
                    let ch = black_box(&channel)[i] as usize;
                    let pm: u16 = if (1..=8).contains(&ch) {
                        PHASE_BASE[(pt + 9 - ch) & 7] as u16
                    } else { 10 };
                    if black_box(&charge)[i] as u16 * 10 >= (threshold[i] as u16 + 1) * pm {
                        black_box(&mut activation)[i] = polarity[i];
                        black_box(&mut charge)[i] = 0;
                    } else {
                        black_box(&mut activation)[i] = 0;
                    }
                }
            }
        }
        let spike_ns = start.elapsed().as_nanos() as f64 / ITERS as f64;

        // Count active neurons for context
        run_full(&mut activation, &mut charge, &mut incoming);
        let active = activation.iter().filter(|&&a| a != 0).count();

        println!("\n=== H={neuron_count}, {edges} edges, {density_pct}% density ===");
        println!("  Active neurons: {active}/{neuron_count} ({:.0}%)", active as f64 / neuron_count as f64 * 100.0);
        println!("  FULL propagate:   {:>10.0} ns", full_ns);
        println!("  --- per phase (12 ticks) ---");
        println!("  decay:            {:>10.0} ns  ({:>4.1}%)", decay_ns, decay_ns / full_ns * 100.0);
        println!("  scatter-add:      {:>10.0} ns  ({:>4.1}%)", scatter_ns, scatter_ns / full_ns * 100.0);
        println!("  spike decision:   {:>10.0} ns  ({:>4.1}%)", spike_ns, spike_ns / full_ns * 100.0);
        let other_ns = full_ns - decay_ns - scatter_ns - spike_ns;
        println!("  other (input+charge): {:>6.0} ns  ({:>4.1}%)", other_ns, other_ns / full_ns * 100.0);
    }
}
