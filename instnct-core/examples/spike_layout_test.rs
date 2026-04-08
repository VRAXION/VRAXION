//! Test: packed NeuronParams vs separate arrays for spike loop.
//!
//! Run: cargo run --example spike_layout_test --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

const TICKS: usize = 6;
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];
const ITERS: usize = 500;

// Packed: threshold + channel + polarity in one struct
#[derive(Clone, Copy)]
#[repr(C)]
struct NeuronParams {
    threshold: u8,
    channel: u8,
    polarity: i8,
    _pad: u8,
}

fn main() {
    let h_configs = [256, 512, 1024, 2048, 4096];

    println!("Spike Layout Test — packed vs SoA vs full-AoS");
    println!("{} iters × {} ticks", ITERS, TICKS);
    println!("{:>6} {:>12} {:>12} {:>12} {:>10} {:>10}",
        "H", "SoA_µs", "packed_µs", "diff%", "SoA_bytes", "pack_bytes");
    println!("{:-<6} {:-<12} {:-<12} {:-<12} {:-<10} {:-<10}", "", "", "", "", "", "");

    for &h in &h_configs {
        let mut rng = StdRng::seed_from_u64(42);

        // SoA layout
        let threshold_a: Vec<u8> = (0..h).map(|_| rng.gen_range(0..=7)).collect();
        let channel_a: Vec<u8> = (0..h).map(|_| rng.gen_range(1..=8)).collect();
        let polarity_a: Vec<i8> = (0..h).map(|_| if rng.gen_ratio(1, 10) { -1 } else { 1 }).collect();

        // Packed layout
        let params: Vec<NeuronParams> = (0..h).map(|i| NeuronParams {
            threshold: threshold_a[i],
            channel: channel_a[i],
            polarity: polarity_a[i],
            _pad: 0,
        }).collect();

        // Shared runtime state
        let incoming = vec![1i16; h];

        // --- Benchmark SoA ---
        let mut charge = vec![5i16; h];
        let mut activation = vec![0i8; h];
        let start = Instant::now();
        for _ in 0..ITERS {
            for tick in 0..TICKS {
                for i in 0..h {
                    charge[i] = charge[i].saturating_add(incoming[i]);
                    let pi = (tick as u8 + 9 - channel_a[i]) & 7;
                    let pm = PHASE_BASE[pi as usize];
                    if charge[i] * 10 >= (threshold_a[i] as i16 + 1) * pm {
                        activation[i] = polarity_a[i];
                        charge[i] = 0;
                    } else {
                        activation[i] = 0;
                    }
                }
            }
        }
        let soa_us = start.elapsed().as_micros();

        // --- Benchmark Packed ---
        let mut charge = vec![5i16; h];
        let mut activation = vec![0i8; h];
        let start = Instant::now();
        for _ in 0..ITERS {
            for tick in 0..TICKS {
                for i in 0..h {
                    charge[i] = charge[i].saturating_add(incoming[i]);
                    let p = &params[i];
                    let pi = (tick as u8 + 9 - p.channel) & 7;
                    let pm = PHASE_BASE[pi as usize];
                    if charge[i] * 10 >= (p.threshold as i16 + 1) * pm {
                        activation[i] = p.polarity;
                        charge[i] = 0;
                    } else {
                        activation[i] = 0;
                    }
                }
            }
        }
        let packed_us = start.elapsed().as_micros();

        let diff = (packed_us as f64 - soa_us as f64) / soa_us as f64 * 100.0;
        let soa_bytes = h * 3; // 3 separate arrays
        let pack_bytes = h * 4; // 4 bytes per packed struct

        println!("{:>6} {:>12} {:>12} {:>11.1}% {:>10} {:>10}",
            h, soa_us, packed_us, diff, soa_bytes, pack_bytes);
    }
}
