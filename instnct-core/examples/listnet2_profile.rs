//! Profile ListNet2 — measure each phase separately.
//!
//! Run: cargo run --example listnet2_profile --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

const VOCAB: usize = 27;
const TICKS: usize = 6;
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];
const P_SRC: usize = 0;
const P_THR: usize = 1;
const P_CHN: usize = 2;
const P_POL: usize = 3;
const P_TARGETS: usize = 4;

#[derive(Clone)]
struct ListNet2 {
    rows: Vec<Vec<u16>>,
    charge: Vec<i16>,
    activation: Vec<i8>,
    h: usize,
    input_dim: usize,
}

impl ListNet2 {
    fn new(h: usize, rng: &mut impl Rng) -> Self {
        let phi_dim = (h as f64 / 1.618).round() as usize;
        let mut rows = Vec::with_capacity(h);
        for i in 0..h {
            let thr = rng.gen_range(0..=7u16);
            let chn = rng.gen_range(1..=8u16);
            let pol = if rng.gen_ratio(1, 10) { 65535u16 } else { 1u16 };
            rows.push(vec![i as u16, thr, chn, pol]);
        }
        ListNet2 { rows, charge: vec![0; h], activation: vec![0; h], h, input_dim: phi_dim }
    }

    fn reset(&mut self) {
        self.charge.iter_mut().for_each(|c| *c = 0);
        self.activation.iter_mut().for_each(|a| *a = 0);
    }

    fn edge_count(&self) -> usize {
        self.rows.iter().map(|r| r.len().saturating_sub(P_TARGETS)).sum()
    }

    fn add_edge(&mut self, src: u16, tgt: u16) -> bool {
        if src == tgt || src as usize >= self.h || tgt as usize >= self.h { return false; }
        let ri = src as usize;
        match self.rows[ri][P_TARGETS..].binary_search(&tgt) {
            Ok(_) => false,
            Err(pos) => { self.rows[ri].insert(P_TARGETS + pos, tgt); true }
        }
    }
}

fn main() {
    let h_configs = [256, 512, 1024];
    let tokens = 500;
    let edge_cap = 300;

    println!("ListNet2 Phase Profiling — {} tokens × {} ticks", tokens, TICKS);
    println!("{:>6} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "H", "input_µs", "scatter_µs", "spike_µs", "save_µs", "total_µs");
    println!("{:-<6} {:-<10} {:-<10} {:-<10} {:-<10} {:-<10}", "", "", "", "", "", "");

    for &h in &h_configs {
        let mut rng = StdRng::seed_from_u64(42);
        let mut net = ListNet2::new(h, &mut rng);

        // Add edges to cap
        while net.edge_count() < edge_cap {
            let s = rng.gen_range(0..h) as u16;
            let t = rng.gen_range(0..h) as u16;
            net.add_edge(s, t);
        }

        // Build SDR
        let input_dim = (h as f64 / 1.618).round() as usize;
        let sdr: Vec<Vec<i8>> = (0..VOCAB).map(|sym| {
            let mut r = StdRng::seed_from_u64(sym as u64 + 9999);
            let mut p = vec![0i8; h]; let active = input_dim / 5; let mut placed = 0;
            while placed < active { let i = r.gen_range(0..input_dim); if p[i]==0 { p[i]=1; placed+=1; } } p
        }).collect();

        let corpus: Vec<u8> = b"the quick brown fox jumps over the lazy dog "
            .iter().map(|&b| if b >= b'a' && b <= b'z' { b - b'a' } else { 26 }).collect();

        // --- Profile: Input injection ---
        net.reset();
        let start = Instant::now();
        for t in 0..tokens {
            let sym = corpus[t % corpus.len()] as usize;
            for tick in 0..TICKS {
                if tick < 2 {
                    let input = &sdr[sym];
                    for i in 0..net.input_dim.min(input.len()) {
                        net.charge[i] = net.charge[i].saturating_add(input[i] as i16);
                    }
                }
            }
        }
        let input_us = start.elapsed().as_micros();

        // --- Profile: Scatter ---
        net.reset();
        // Prime some activations
        for i in 0..h.min(50) { net.activation[i] = 1; }
        let start = Instant::now();
        for _ in 0..tokens {
            for _ in 0..TICKS {
                let mut incoming = vec![0i16; h];
                for row in &net.rows {
                    let src = row[P_SRC] as usize;
                    if src >= h { continue; }
                    let act = net.activation[src];
                    if act != 0 && row.len() > P_TARGETS {
                        for &tgt in &row[P_TARGETS..] {
                            let t = tgt as usize;
                            if t < h { incoming[t] = incoming[t].saturating_add(act as i16); }
                        }
                    }
                }
                // consume incoming to prevent optimization
                std::hint::black_box(&incoming);
            }
        }
        let scatter_us = start.elapsed().as_micros();

        // --- Profile: Spike (read params from rows) ---
        net.reset();
        for i in 0..h { net.charge[i] = 5; } // prime charges
        let incoming_dummy = vec![1i16; h];
        let start = Instant::now();
        for _ in 0..tokens {
            for tick in 0..TICKS {
                for row in &net.rows {
                    let i = row[P_SRC] as usize;
                    if i >= h { continue; }
                    net.charge[i] = net.charge[i].saturating_add(incoming_dummy[i]);
                    let threshold = row[P_THR] as u8;
                    let channel = row[P_CHN] as u8;
                    let polarity = if row[P_POL] == 65535 { -1i8 } else { 1i8 };
                    let pi = (tick as u8 + 9 - channel) & 7;
                    let pm = PHASE_BASE[pi as usize];
                    if net.charge[i] * 10 >= (threshold as i16 + 1) * pm {
                        net.activation[i] = polarity;
                        net.charge[i] = 0;
                    } else {
                        net.activation[i] = 0;
                    }
                }
            }
        }
        let spike_us = start.elapsed().as_micros();

        // --- Profile: Save/restore ---
        let start = Instant::now();
        for _ in 0..1000 {
            let snap = net.rows.clone();
            std::hint::black_box(&snap);
        }
        let save_us = start.elapsed().as_micros() / 1000; // per-save

        let total = input_us + scatter_us + spike_us;
        println!("{:>6} {:>10} {:>10} {:>10} {:>10} {:>10}",
            h, input_us, scatter_us, spike_us, save_us, total);
    }

    println!("\nNow same but with SoA spike (separate threshold/channel/polarity arrays):");
    println!("{:>6} {:>10} {:>10}", "H", "spike_soa", "save_soa");
    println!("{:-<6} {:-<10} {:-<10}", "", "", "");

    for &h in &h_configs {
        let mut rng = StdRng::seed_from_u64(42);
        // SoA params
        let mut threshold = vec![0u8; h];
        let mut channel = vec![0u8; h];
        let mut polarity = vec![1i8; h];
        let mut charge = vec![5i16; h];
        let mut activation = vec![0i8; h];
        for i in 0..h {
            threshold[i] = rng.gen_range(0..=7);
            channel[i] = rng.gen_range(1..=8);
            if rng.gen_ratio(1, 10) { polarity[i] = -1; }
        }

        let incoming_dummy = vec![1i16; h];
        let start = Instant::now();
        for _ in 0..500 {
            for tick in 0..TICKS {
                for i in 0..h {
                    charge[i] = charge[i].saturating_add(incoming_dummy[i]);
                    let pi = (tick as u8 + 9 - channel[i]) & 7;
                    let pm = PHASE_BASE[pi as usize];
                    if charge[i] * 10 >= (threshold[i] as i16 + 1) * pm {
                        activation[i] = polarity[i];
                        charge[i] = 0;
                    } else {
                        activation[i] = 0;
                    }
                }
            }
        }
        let spike_soa = start.elapsed().as_micros();

        // SoA save: topology (Vec<Vec<u16>>) + 3 param arrays
        let mut rows: Vec<Vec<u16>> = Vec::with_capacity(h);
        for i in 0..h { rows.push(vec![i as u16]); }
        // Add edges
        while rows.iter().map(|r| r.len().saturating_sub(1)).sum::<usize>() < 300 {
            let s = rng.gen_range(0..h);
            let t = rng.gen_range(0..h) as u16;
            if s as u16 != t && !rows[s][1..].contains(&t) { rows[s].push(t); }
        }

        let start = Instant::now();
        for _ in 0..1000 {
            let s1 = rows.clone();
            let s2 = threshold.clone();
            let s3 = channel.clone();
            let s4 = polarity.clone();
            std::hint::black_box((&s1, &s2, &s3, &s4));
        }
        let save_soa = start.elapsed().as_micros() / 1000;

        println!("{:>6} {:>10} {:>10}", h, spike_soa, save_soa);
    }
}
