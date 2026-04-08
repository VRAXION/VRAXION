//! Thermometer encoding + weighted edges + high threshold + rate coding readout
//!
//! Setup:
//!   - Input: thermometer encoding (digit N = N active bits in a row)
//!   - Edges: int8 weight, mutatable
//!   - Threshold: starts high (4-7), mutatable
//!   - Readout: spike COUNT over 50 ticks (rate coding)
//!   - I/O: separated (no overlap)
//!
//! Run: cargo run --example thermo_addition --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

const DIGITS: usize = 5;  // 0-4
const SUMS: usize = 9;    // 0-8
const H: usize = 64;      // small network, easier to search
const TICKS: usize = 50;  // lots of ticks for rate coding
const INPUT_TICKS: usize = 20; // inject input for 20 ticks
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];
const EDGE_CAP: usize = 200;
const WALL_SECS: u64 = 120;

// Input layout: [0..8] = digit A thermometer, [8..16] = digit B thermometer
const A_START: usize = 0;
const A_END: usize = 8;
const B_START: usize = 8;
const B_END: usize = 16;
const INPUT_END: usize = 16;
const OUTPUT_START: usize = 32; // [32..64] = output zone

#[derive(Clone)]
struct Net {
    edges: Vec<(u16, u16, i8)>,  // (src, tgt, weight)
    threshold: Vec<u8>,
    channel: Vec<u8>,
    polarity: Vec<i8>,
    charge: Vec<i16>,
    activation: Vec<i8>,
    refractory: Vec<u8>,
    spike_count: Vec<u32>,
    h: usize,
}

impl Net {
    fn new(h: usize, rng: &mut impl Rng) -> Self {
        let mut threshold = vec![0u8; h]; let mut channel = vec![0u8; h]; let mut polarity = vec![1i8; h];
        for i in 0..h {
            threshold[i] = rng.gen_range(3..=7); // HIGH threshold for rate coding
            channel[i] = rng.gen_range(1..=8);
            if rng.gen_ratio(3, 10) { polarity[i] = -1; } // 30% inhibitory
        }
        Net { edges: Vec::new(), threshold, channel, polarity,
            charge: vec![0; h], activation: vec![0; h], refractory: vec![0; h], spike_count: vec![0; h], h }
    }

    fn reset(&mut self) {
        self.charge.iter_mut().for_each(|c| *c = 0);
        self.activation.iter_mut().for_each(|a| *a = 0);
        self.refractory.iter_mut().for_each(|r| *r = 0);
        self.spike_count.iter_mut().for_each(|s| *s = 0);
    }

    fn propagate(&mut self, input: &[i8], tick: usize) {
        let h = self.h;
        // Input injection (continuous for INPUT_TICKS)
        if tick < INPUT_TICKS {
            for i in 0..INPUT_END.min(input.len()) {
                if input[i] != 0 { self.charge[i] += input[i] as i16; }
            }
        }

        let mut incoming = vec![0i16; h];
        for &(src, tgt, weight) in &self.edges {
            let s = src as usize; let t = tgt as usize;
            if s >= h || t >= h { continue; }
            let act = self.activation[s];
            if act != 0 {
                incoming[t] = incoming[t].saturating_add(act as i16 * self.polarity[s] as i16 * weight as i16);
            }
        }

        for i in 0..h {
            if self.refractory[i] > 0 { self.refractory[i] -= 1; self.activation[i] = 0; continue; }
            self.charge[i] = self.charge[i].saturating_add(incoming[i]);
            if tick % 6 == 5 && self.charge[i] > 0 { self.charge[i] -= 1; }
            if self.charge[i] < 0 { self.charge[i] = 0; }
            let pi = (tick as u8 + 9 - self.channel[i]) & 7; let pm = PHASE_BASE[pi as usize];
            if self.charge[i] * 10 >= (self.threshold[i] as i16 + 1) * pm {
                self.activation[i] = 1; // polarity applied at edge output
                self.charge[i] = 0; self.refractory[i] = 1;
                self.spike_count[i] += 1;
            } else { self.activation[i] = 0; }
        }
    }

    fn add_edge(&mut self, s: u16, t: u16, w: i8) -> bool {
        if s == t || s as usize >= self.h || t as usize >= self.h { return false; }
        if self.edges.iter().any(|&(es,et,_)| es == s && et == t) { return false; }
        self.edges.push((s, t, w)); true
    }
    fn remove_edge(&mut self, rng: &mut impl Rng) -> bool {
        if self.edges.is_empty() { return false; } self.edges.swap_remove(rng.gen_range(0..self.edges.len())); true
    }
    fn rewire(&mut self, rng: &mut impl Rng) -> bool {
        if self.edges.is_empty() { return false; }
        let idx = rng.gen_range(0..self.edges.len()); let w = self.edges[idx].2;
        let s = rng.gen_range(0..self.h) as u16; let t = rng.gen_range(0..self.h) as u16;
        if s == t { return false; } if self.edges.iter().any(|&(es,et,_)| es == s && et == t) { return false; }
        self.edges[idx] = (s, t, w); true
    }
    fn mutate(&mut self, rng: &mut impl Rng) -> bool {
        match rng.gen_range(0..100u32) {
            0..15 => { if self.edges.len() >= EDGE_CAP { return false; } self.add_edge(rng.gen_range(0..self.h) as u16, rng.gen_range(0..self.h) as u16, rng.gen_range(1..=4)) }
            15..25 => self.remove_edge(rng), 25..45 => self.rewire(rng),
            45..65 => { // weight mutation
                if self.edges.is_empty() { return false; }
                let idx = rng.gen_range(0..self.edges.len());
                self.edges[idx].2 = rng.gen_range(-8..=8i8); true }
            65..80 => { let n = rng.gen_range(0..self.h); self.threshold[n] = rng.gen_range(0..=15); true }
            80..92 => { let n = rng.gen_range(0..self.h); self.channel[n] = rng.gen_range(1..=8); true }
            _ => { self.polarity[rng.gen_range(0..self.h)] *= -1; true }
        }
    }
    fn save(&self) -> (Vec<(u16,u16,i8)>, Vec<u8>, Vec<u8>, Vec<i8>) {
        (self.edges.clone(), self.threshold.clone(), self.channel.clone(), self.polarity.clone())
    }
    fn restore(&mut self, s: (Vec<(u16,u16,i8)>, Vec<u8>, Vec<u8>, Vec<i8>)) {
        self.edges = s.0; self.threshold = s.1; self.channel = s.2; self.polarity = s.3;
    }
}

/// Thermometer encoding: digit N → first N bits active
fn thermo_input(a: usize, b: usize) -> Vec<i8> {
    let mut input = vec![0i8; 64];
    for i in 0..a { input[A_START + i] = 1; } // digit A
    for i in 0..b { input[B_START + i] = 1; } // digit B
    input
}

/// Readout: spike count in output zone → argmax class
fn readout(net: &Net) -> usize {
    // Divide output zone into SUMS bins, sum spike counts per bin
    let zone_len = net.h - OUTPUT_START;
    let mut scores = vec![0u32; SUMS];
    for i in 0..zone_len {
        let class = i * SUMS / zone_len;
        scores[class] += net.spike_count[OUTPUT_START + i];
    }
    scores.iter().enumerate().max_by_key(|&(_, v)| *v).map(|(i, _)| i).unwrap_or(0)
}

fn eval(net: &mut Net, examples: &[(usize,usize,usize)]) -> f64 {
    let mut correct = 0;
    for &(a, b, target) in examples {
        net.reset();
        let input = thermo_input(a, b);
        for tick in 0..TICKS { net.propagate(&input, tick); }
        if readout(net) == target { correct += 1; }
    }
    correct as f64 / examples.len() as f64
}

fn main() {
    let all: Vec<_> = (0..DIGITS).flat_map(|a| (0..DIGITS).map(move |b| (a, b, a+b))).collect();
    let train: Vec<_> = all.iter().filter(|&&(_,_,s)| s != 4).cloned().collect();
    let test: Vec<_> = all.iter().filter(|&&(_,_,s)| s == 4).cloned().collect();

    println!("=== THERMOMETER + WEIGHTED + RATE CODING ===");
    println!("RUNNING: thermo_addition");
    println!("H={}, ticks={}, input_ticks={}, edge_cap={}, {}s/seed", H, TICKS, INPUT_TICKS, EDGE_CAP, WALL_SECS);
    println!("Encoding: thermometer (digit N = N active bits)");
    println!("Readout: spike count in output zone → argmax");
    println!("Train: sum≠4 (20), Test: sum=4 (5), Random: {:.0}%\n", 100.0/SUMS as f64);

    // Show encoding
    println!("Thermometer encoding:");
    for d in 0..DIGITS {
        let inp = thermo_input(d, 0);
        let bits_a: String = (A_START..A_END).map(|i| if inp[i] != 0 { '█' } else { '·' }).collect();
        println!("  digit {}: [{}]", d, bits_a);
    }
    println!("  2+3 input: A=[██······] B=[███·····] → 5 total active bits");
    println!("  1+4 input: A=[█·······] B=[████····] → 5 total active bits (same total!)\n");

    for &seed in &[42u64, 1042, 2042, 3042, 4042] {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut net = Net::new(H, &mut rng);

        // Init: bridges from input zone to output zone
        for _ in 0..30 {
            let s = rng.gen_range(0..INPUT_END) as u16;
            let t = rng.gen_range(OUTPUT_START..H) as u16;
            net.add_edge(s, t, rng.gen_range(1..=4));
        }
        // Also some hidden-to-hidden edges
        for _ in 0..20 {
            let s = rng.gen_range(INPUT_END..OUTPUT_START) as u16;
            let t = rng.gen_range(INPUT_END..H) as u16;
            net.add_edge(s, t, rng.gen_range(1..=3));
        }

        let deadline = Instant::now() + Duration::from_secs(WALL_SECS);
        let mut steps = 0;
        while Instant::now() < deadline {
            let before = eval(&mut net, &train);
            let genome = net.save();
            if net.mutate(&mut rng) {
                let after = eval(&mut net, &train);
                if after <= before { net.restore(genome); }
            }
            steps += 1;
        }

        let train_acc = eval(&mut net, &train);
        let test_acc = eval(&mut net, &test);
        let all_acc = eval(&mut net, &all);

        // Diagnose: show what each sum predicts
        let mut sum_detail = String::new();
        for target_sum in 0..SUMS {
            let examples_for_sum: Vec<_> = all.iter().filter(|&&(_,_,s)| s == target_sum).collect();
            if examples_for_sum.is_empty() { continue; }
            let mut correct = 0;
            for &&(a, b, target) in &examples_for_sum {
                net.reset();
                let input = thermo_input(a, b);
                for tick in 0..TICKS { net.propagate(&input, tick); }
                if readout(&net) == target { correct += 1; }
            }
            let acc = correct as f64 / examples_for_sum.len() as f64;
            if acc > 0.0 && acc < 1.0 { sum_detail.push_str(&format!(" s{}={}/{}", target_sum, correct, examples_for_sum.len())); }
        }

        println!("  seed {}: train={:.0}% test={:.0}% all={:.0}% | edges={} steps={} |{}",
            seed, train_acc*100.0, test_acc*100.0, all_acc*100.0, net.edges.len(), steps, sum_detail);
    }
}
