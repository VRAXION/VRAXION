//! Test: charge+activation packed vs separate
//!
//! Run: cargo run --example state_layout_test --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

const TICKS: usize = 6;
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];
const ITERS: usize = 500;

#[derive(Clone, Copy)]
#[repr(C)]
struct NeuronParams {
    threshold: u8,
    channel: u8,
    polarity: i8,
    _pad: u8,
}

// Packed state: charge + activation together
#[derive(Clone, Copy)]
#[repr(C)]
struct NeuronState {
    charge: i16,
    activation: i8,
    _pad: u8,
}

// Full packed: params + state all in one
#[derive(Clone, Copy)]
#[repr(C)]
struct NeuronFull {
    threshold: u8,
    channel: u8,
    polarity: i8,
    _pad: u8,
    charge: i16,
    activation: i8,
    _pad2: u8,
}

fn main() {
    let h_configs = [256, 512, 1024, 2048, 4096];

    println!("State Layout Test — separate vs packed-state vs full-packed");
    println!("{} iters × {} ticks", ITERS, TICKS);
    println!("{:>6} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "H", "separate", "pack_st", "full_pack", "st_diff%", "full_diff%");
    println!("{:-<6} {:-<10} {:-<10} {:-<10} {:-<10} {:-<10}", "", "", "", "", "", "");

    for &h in &h_configs {
        let mut rng = StdRng::seed_from_u64(42);

        let params: Vec<NeuronParams> = (0..h).map(|_| NeuronParams {
            threshold: rng.gen_range(0..=7), channel: rng.gen_range(1..=8),
            polarity: if rng.gen_ratio(1, 10) { -1 } else { 1 }, _pad: 0,
        }).collect();

        let incoming = vec![1i16; h];

        // --- A: params packed, charge+activation SEPARATE ---
        {
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
            let separate_us = start.elapsed().as_micros();

            // --- B: params packed, charge+activation PACKED ---
            let mut state: Vec<NeuronState> = (0..h).map(|_| NeuronState { charge: 5, activation: 0, _pad: 0 }).collect();
            let start = Instant::now();
            for _ in 0..ITERS {
                for tick in 0..TICKS {
                    for i in 0..h {
                        state[i].charge = state[i].charge.saturating_add(incoming[i]);
                        let p = &params[i];
                        let pi = (tick as u8 + 9 - p.channel) & 7;
                        let pm = PHASE_BASE[pi as usize];
                        if state[i].charge * 10 >= (p.threshold as i16 + 1) * pm {
                            state[i].activation = p.polarity;
                            state[i].charge = 0;
                        } else {
                            state[i].activation = 0;
                        }
                    }
                }
            }
            let packed_state_us = start.elapsed().as_micros();

            // --- C: EVERYTHING in one struct ---
            let mut full: Vec<NeuronFull> = (0..h).map(|i| NeuronFull {
                threshold: params[i].threshold, channel: params[i].channel,
                polarity: params[i].polarity, _pad: 0,
                charge: 5, activation: 0, _pad2: 0,
            }).collect();
            let start = Instant::now();
            for _ in 0..ITERS {
                for tick in 0..TICKS {
                    for i in 0..h {
                        full[i].charge = full[i].charge.saturating_add(incoming[i]);
                        let pi = (tick as u8 + 9 - full[i].channel) & 7;
                        let pm = PHASE_BASE[pi as usize];
                        if full[i].charge * 10 >= (full[i].threshold as i16 + 1) * pm {
                            full[i].activation = full[i].polarity;
                            full[i].charge = 0;
                        } else {
                            full[i].activation = 0;
                        }
                    }
                }
            }
            let full_us = start.elapsed().as_micros();

            let st_diff = (packed_state_us as f64 - separate_us as f64) / separate_us as f64 * 100.0;
            let full_diff = (full_us as f64 - separate_us as f64) / separate_us as f64 * 100.0;
            println!("{:>6} {:>10} {:>10} {:>10} {:>9.1}% {:>9.1}%",
                h, separate_us, packed_state_us, full_us, st_diff, full_diff);
        }
    }
}
