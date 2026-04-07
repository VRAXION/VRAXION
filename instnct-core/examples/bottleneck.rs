//! Bottleneck analysis: which part of the forward pass is slow at large H?
//!
//! Run: cargo run --example bottleneck --release

use instnct_core::{Network, PropagationConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn main() {
    let sizes = [256, 512, 1024, 2048, 4096];
    let density_pct = 5u64;
    let iters = 50;

    println!(
        "{:>6} {:>8} {:>10} {:>10} {:>10}",
        "H", "edges", "ns/token", "edge_bytes", "fits_L2?"
    );
    println!("{:-<6} {:-<8} {:-<10} {:-<10} {:-<10}", "", "", "", "", "");

    for &h in &sizes {
        let mut net = Network::new(h);
        let mut rng = StdRng::seed_from_u64(42);
        let target = (h as u64 * h as u64 * density_pct / 100) as usize;
        for _ in 0..target * 3 {
            net.mutate_add_edge(&mut rng);
            if net.edge_count() >= target {
                break;
            }
        }
        for i in 0..h {
            net.spike_data_mut()[i].threshold = rng.gen_range(0..=15);
            net.spike_data_mut()[i].channel = rng.gen_range(1..=8);
        }

        let config = PropagationConfig::default();
        let input = vec![1i32; h];

        // Warmup
        for _ in 0..10 {
            net.reset();
            net.propagate(&input, &config).unwrap();
        }

        // Measure
        let start = Instant::now();
        for _ in 0..iters {
            net.reset();
            net.propagate(&input, &config).unwrap();
        }
        let ns_per_token = start.elapsed().as_nanos() as u64 / iters;

        let edges = net.edge_count();
        let edge_bytes = edges * 8 * 2; // sources + targets, usize each
        let neuron_bytes = h * (4 + 4 + 4 + 1 + 4); // activation + charge + threshold + channel + polarity
        let total_working_set = edge_bytes + neuron_bytes;
        let fits_l2 = if total_working_set < 512_000 {
            "yes"
        } else {
            "NO"
        };

        println!(
            "{:>6} {:>8} {:>8} ns {:>8} KB {:>10}",
            h,
            edges,
            ns_per_token,
            total_working_set / 1024,
            fits_l2
        );
    }

    println!("\nRyzen 3900X: L1=32KB, L2=512KB, L3=32MB/CCX");
    println!("The scatter-add loop streams edge arrays (sources+targets) every tick.");
    println!("When edge data > L2, every tick = L3 round-trip = massive slowdown.");
    println!("\nBreakdown per tick at H=4096:");

    // Detailed breakdown for H=4096
    let h = 4096;
    let edges = 839_000usize; // approximate
    println!(
        "  edge arrays:    {} KB  (sources + targets)",
        edges * 16 / 1024
    );
    println!(
        "  neuron arrays:  {} KB  (act + charge + threshold + channel + polarity)",
        h * 17 / 1024
    );
    println!("  incoming buf:   {} KB", h * 4 / 1024);
    println!("  total:          {} KB", (edges * 16 + h * 21) / 1024);
    println!(
        "  x12 ticks =     {} MB streamed per token",
        (edges * 16 * 12) / 1024 / 1024
    );
}
