//! LIF Addition: fly-brain dual-variable model on addition task.
//! g (synaptic current) + v (membrane voltage), weighted edges, thermometer encoding.
//! Test: does the LIF model generalize where the single-charge model couldn't?
//!
//! RUNNING: lif_addition
//!
//! Run: cargo run --example lif_addition --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

const DIGITS: usize = 5;
const SUMS: usize = 9;
const H: usize = 64;
const TICKS: usize = 50;
const INPUT_TICKS: usize = 20;
const EDGE_CAP: usize = 200;
const WALL_SECS: u64 = 120;
const INPUT_END: usize = 16;     // [0..8] = digit A thermo, [8..16] = digit B thermo
const OUTPUT_START: usize = 32;  // [32..64] = output zone

#[derive(Clone)]
struct LIFNet {
    // Per-neuron dual variables
    g: Vec<i32>,           // synaptic current
    v: Vec<i32>,           // membrane voltage
    refractory: Vec<u8>,
    firing: Vec<bool>,
    spike_count: Vec<u32>,
    // Learnable params
    edges: Vec<(u16, u16, i16)>,  // (src, tgt, weight)
    threshold: Vec<u8>,           // per-neuron (0-15, used as-is, not +1)
    polarity: Vec<i8>,            // Dale's law: +1 excitatory, -1 inhibitory
    h: usize,
}

impl LIFNet {
    fn new(h: usize, rng: &mut impl Rng) -> Self {
        let mut threshold = vec![0u8; h]; let mut polarity = vec![1i8; h];
        for i in 0..h {
            threshold[i] = rng.gen_range(5..=10); // fly-like range
            if rng.gen_ratio(3, 10) { polarity[i] = -1; } // 30% inhibitory
        }
        LIFNet {
            g: vec![0; h], v: vec![0; h], refractory: vec![0; h],
            firing: vec![false; h], spike_count: vec![0; h],
            edges: Vec::new(), threshold, polarity, h,
        }
    }

    fn reset(&mut self) {
        self.g.iter_mut().for_each(|x| *x = 0);
        self.v.iter_mut().for_each(|x| *x = 0);
        self.refractory.iter_mut().for_each(|x| *x = 0);
        self.firing.iter_mut().for_each(|x| *x = false);
        self.spike_count.iter_mut().for_each(|x| *x = 0);
    }

    fn step(&mut self) {
        let h = self.h;

        // 1. Deliver spikes: firing src → g[tgt] += weight × polarity
        let mut g_in = vec![0i32; h];
        for &(src, tgt, weight) in &self.edges {
            let s = src as usize; let t = tgt as usize;
            if s >= h || t >= h { continue; }
            if self.firing[s] {
                g_in[t] += weight as i32 * self.polarity[s] as i32;
            }
        }

        // 2. Update each neuron (LIF dynamics)
        for i in 0..h {
            if self.refractory[i] > 0 {
                self.refractory[i] -= 1;
                self.firing[i] = false;
                continue;
            }

            // Add incoming spike current
            self.g[i] += g_in[i];

            // Synaptic decay: g -= g/5 (tau_syn ≈ 5 ticks)
            self.g[i] -= self.g[i] / 5;

            // Membrane integration: v += (-v + g) / 20 (tau_m ≈ 20 ticks)
            self.v[i] += (-self.v[i] + self.g[i]) / 20;

            // Clamp v to non-negative (optional, biological v can go negative)
            // Let it go negative for inhibition effect

            // Spike check
            if self.v[i] >= self.threshold[i] as i32 {
                self.firing[i] = true;
                self.v[i] = 0;
                self.g[i] = 0;
                self.refractory[i] = 2;
                self.spike_count[i] += 1;
            } else {
                self.firing[i] = false;
            }
        }
    }

    fn inject_thermo(&mut self, a: usize, b: usize) {
        // Thermometer: digit N = inject into first N neurons of each slot
        for i in 0..a.min(8) { self.g[i] += 5; }         // A slot [0..8]
        for i in 0..b.min(8) { self.g[8 + i] += 5; }     // B slot [8..16]
    }

    // Mutations
    fn add_edge(&mut self, s: u16, t: u16, w: i16) -> bool {
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
            0..15 => { if self.edges.len() >= EDGE_CAP { return false; }
                self.add_edge(rng.gen_range(0..self.h) as u16, rng.gen_range(0..self.h) as u16, rng.gen_range(1..=8)) }
            15..25 => self.remove_edge(rng),
            25..45 => self.rewire(rng),
            45..65 => { // weight mutation
                if self.edges.is_empty() { return false; }
                let idx = rng.gen_range(0..self.edges.len());
                self.edges[idx].2 = rng.gen_range(-16..=16i16); true }
            65..80 => { let n = rng.gen_range(0..self.h); self.threshold[n] = rng.gen_range(3..=15); true }
            _ => { self.polarity[rng.gen_range(0..self.h)] *= -1; true }
        }
    }
    fn save(&self) -> (Vec<(u16,u16,i16)>, Vec<u8>, Vec<i8>) {
        (self.edges.clone(), self.threshold.clone(), self.polarity.clone())
    }
    fn restore(&mut self, s: (Vec<(u16,u16,i16)>, Vec<u8>, Vec<i8>)) {
        self.edges = s.0; self.threshold = s.1; self.polarity = s.2;
    }
}

// Readout: spike count per output class
fn readout(net: &LIFNet) -> usize {
    let zone = OUTPUT_START..net.h;
    let zone_len = zone.len();
    let mut scores = vec![0u32; SUMS];
    for (i, idx) in zone.enumerate() {
        let class = i * SUMS / zone_len;
        scores[class] += net.spike_count[idx];
    }
    scores.iter().enumerate().max_by_key(|&(_, v)| *v).map(|(i, _)| i).unwrap_or(0)
}

fn eval(net: &mut LIFNet, examples: &[(usize,usize,usize)]) -> f64 {
    let mut correct = 0;
    for &(a, b, target) in examples {
        net.reset();
        for tick in 0..TICKS {
            if tick < INPUT_TICKS { net.inject_thermo(a, b); }
            net.step();
        }
        if readout(net) == target { correct += 1; }
    }
    correct as f64 / examples.len() as f64
}

fn main() {
    let all: Vec<_> = (0..DIGITS).flat_map(|a| (0..DIGITS).map(move |b| (a, b, a+b))).collect();
    let train: Vec<_> = all.iter().filter(|&&(_,_,s)| s != 4).cloned().collect();
    let test: Vec<_> = all.iter().filter(|&&(_,_,s)| s == 4).cloned().collect();

    println!("=== LIF ADDITION: Fly-brain dual-variable model ===");
    println!("RUNNING: lif_addition");
    println!("g (synaptic) + v (membrane), weighted edges, thermometer input");
    println!("H={}, ticks={}, edge_cap={}, {}s/seed", H, TICKS, EDGE_CAP, WALL_SECS);
    println!("Train: sum≠4 (20), Test: sum=4 (5), Random: {:.0}%\n", 100.0/SUMS as f64);

    for &seed in &[42u64, 1042, 2042, 3042, 4042] {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut net = LIFNet::new(H, &mut rng);

        // Init bridges from input → output
        for _ in 0..30 {
            let s = rng.gen_range(0..INPUT_END) as u16;
            let t = rng.gen_range(OUTPUT_START..H) as u16;
            net.add_edge(s, t, rng.gen_range(1..=4));
        }
        // Some hidden connections
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

        // Diagnose per-sum
        let mut detail = String::new();
        for s in 0..SUMS {
            let ex: Vec<_> = all.iter().filter(|&&(_,_,sum)| sum == s).cloned().collect();
            if ex.is_empty() { continue; }
            let mut c = 0;
            for &(a, b, target) in &ex { net.reset(); for tick in 0..TICKS { if tick < INPUT_TICKS { net.inject_thermo(a, b); } net.step(); } if readout(&net) == target { c += 1; } }
            if c > 0 && c < ex.len() { detail.push_str(&format!(" s{}={}/{}", s, c, ex.len())); }
        }

        println!("  seed {}: train={:.0}% test={:.0}% all={:.0}% | edges={} steps={} |{}",
            seed, train_acc*100.0, test_acc*100.0, all_acc*100.0, net.edges.len(), steps, detail);
    }
}
