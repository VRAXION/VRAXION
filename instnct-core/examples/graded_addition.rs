//! Graded potential addition: charge propagates through edges, not just spikes.
//! incoming[target] += activation[src] + charge[src] / LEAK_DIV
//!
//! Test: does graded propagation enable generalization?
//!
//! Run: cargo run --example graded_addition --release

use instnct_core::build_bigram_table;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

const DIGITS: usize = 5;
const SUMS: usize = 9;
const H: usize = 256;
const TICKS: usize = 6;
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];
const EDGE_CAP: usize = 300;
const WALL_SECS: u64 = 120;
const INPUT_END: usize = 128;
const OUTPUT_START: usize = 128;

#[derive(Clone)]
struct GradedNet {
    edges: Vec<(u16, u16)>,
    threshold: Vec<u8>,
    channel: Vec<u8>,
    polarity: Vec<i8>,
    charge: Vec<i16>,
    activation: Vec<i8>,
    leak_div: i16, // charge / leak_div = graded signal strength
    h: usize,
}

impl GradedNet {
    fn new(h: usize, leak_div: i16, rng: &mut impl Rng) -> Self {
        let mut threshold = vec![0u8; h]; let mut channel = vec![0u8; h]; let mut polarity = vec![1i8; h];
        for i in 0..h { threshold[i] = rng.gen_range(0..=7); channel[i] = rng.gen_range(1..=8); if rng.gen_ratio(1, 10) { polarity[i] = -1; } }
        GradedNet { edges: Vec::new(), threshold, channel, polarity, charge: vec![0; h], activation: vec![0; h], leak_div, h }
    }

    fn reset(&mut self) { self.charge.iter_mut().for_each(|c| *c = 0); self.activation.iter_mut().for_each(|a| *a = 0); }

    fn propagate(&mut self, input: &[i8], tick: usize) {
        let h = self.h;
        if tick < 2 { for i in 0..INPUT_END.min(input.len()) { self.charge[i] = self.charge[i].saturating_add(input[i] as i16); } }

        // GRADED propagation: spike + charge leak
        let mut incoming = vec![0i16; h];
        for &(src, tgt) in &self.edges {
            let s = src as usize; let t = tgt as usize;
            if s >= h || t >= h { continue; }
            let pol = self.polarity[s] as i16;
            let spike = self.activation[s] as i16 * pol;
            let graded = (self.charge[s] / self.leak_div) * pol;
            incoming[t] = incoming[t].saturating_add(spike + graded);
        }

        for i in 0..h {
            self.charge[i] = self.charge[i].saturating_add(incoming[i]);
            if tick % 6 == 5 && self.charge[i] > 0 { self.charge[i] -= 1; }
            let pi = (tick as u8 + 9 - self.channel[i]) & 7; let pm = PHASE_BASE[pi as usize];
            if self.charge[i] * 10 >= (self.threshold[i] as i16 + 1) * pm {
                self.activation[i] = self.polarity[i]; self.charge[i] = 0;
            } else { self.activation[i] = 0; }
        }
    }

    fn add_edge(&mut self, src: u16, tgt: u16) -> bool {
        if src == tgt || src as usize >= self.h || tgt as usize >= self.h { return false; }
        if self.edges.iter().any(|&(s,t)| s == src && t == tgt) { return false; }
        self.edges.push((src, tgt)); true
    }
    fn remove_edge(&mut self, rng: &mut impl Rng) -> bool {
        if self.edges.is_empty() { return false; } self.edges.swap_remove(rng.gen_range(0..self.edges.len())); true
    }
    fn rewire(&mut self, rng: &mut impl Rng) -> bool {
        if self.edges.is_empty() { return false; }
        let idx = rng.gen_range(0..self.edges.len());
        let s = rng.gen_range(0..self.h) as u16; let t = rng.gen_range(0..self.h) as u16;
        if s == t { return false; } if self.edges.iter().any(|&(es,et)| es == s && et == t) { return false; }
        self.edges[idx] = (s, t); true
    }
    fn mutate(&mut self, rng: &mut impl Rng) -> bool {
        match rng.gen_range(0..100u32) {
            0..25 => { if self.edges.len() >= EDGE_CAP { return false; } self.add_edge(rng.gen_range(0..self.h) as u16, rng.gen_range(0..self.h) as u16) }
            25..40 => self.remove_edge(rng), 40..65 => self.rewire(rng),
            65..80 => { let n = rng.gen_range(0..self.h); self.threshold[n] = rng.gen_range(0..=15); true }
            80..92 => { let n = rng.gen_range(0..self.h); self.channel[n] = rng.gen_range(1..=8); true }
            _ => { self.polarity[rng.gen_range(0..self.h)] *= -1; true }
        }
    }
    fn save(&self) -> (Vec<(u16,u16)>, Vec<u8>, Vec<u8>, Vec<i8>) {
        (self.edges.clone(), self.threshold.clone(), self.channel.clone(), self.polarity.clone())
    }
    fn restore(&mut self, s: (Vec<(u16,u16)>, Vec<u8>, Vec<u8>, Vec<i8>)) {
        self.edges = s.0; self.threshold = s.1; self.channel = s.2; self.polarity = s.3;
    }
}

#[derive(Clone)]
struct Proj { w: Vec<Vec<i8>>, h: usize, classes: usize }
impl Proj {
    fn new(h: usize, classes: usize, rng: &mut impl Rng) -> Self {
        Proj { w: (0..h).map(|_| (0..classes).map(|_| rng.gen_range(-2..=2i8)).collect()).collect(), h, classes }
    }
    fn predict(&self, charge: &[i16]) -> usize {
        let mut s = vec![0i64; self.classes];
        for i in OUTPUT_START..self.h { let ch = charge[i] as i64; if ch == 0 { continue; }
            for c in 0..self.classes { s[c] += ch * self.w[i][c] as i64; } }
        s.iter().enumerate().max_by_key(|&(_, v)| *v).map(|(i, _)| i).unwrap_or(0)
    }
    fn mutate(&mut self, rng: &mut impl Rng) -> (usize, usize, i8) {
        let i = rng.gen_range(OUTPUT_START..self.h); let c = rng.gen_range(0..self.classes);
        let old = self.w[i][c]; self.w[i][c] = rng.gen_range(-4..=4i8); (i, c, old)
    }
    fn undo(&mut self, i: usize, c: usize, old: i8) { self.w[i][c] = old; }
}

fn make_examples() -> Vec<(usize,usize,usize)> {
    (0..DIGITS).flat_map(|a| (0..DIGITS).map(move |b| (a, b, a+b))).collect()
}

fn make_sdr(h: usize) -> (Vec<Vec<i8>>, Vec<Vec<i8>>) {
    let half = INPUT_END / 2; let active = half / 5;
    let sdr_a: Vec<Vec<i8>> = (0..DIGITS).map(|d| { let mut rng = StdRng::seed_from_u64(d as u64 + 100);
        let mut p = vec![0i8; h]; let mut pl = 0; while pl < active { let i = rng.gen_range(0..half); if p[i]==0 { p[i]=1; pl+=1; } } p }).collect();
    let sdr_b: Vec<Vec<i8>> = (0..DIGITS).map(|d| { let mut rng = StdRng::seed_from_u64(d as u64 + 200);
        let mut p = vec![0i8; h]; let mut pl = 0; while pl < active { let i = rng.gen_range(half..INPUT_END); if p[i]==0 { p[i]=1; pl+=1; } } p }).collect();
    (sdr_a, sdr_b)
}

fn eval(net: &mut GradedNet, proj: &Proj, examples: &[(usize,usize,usize)], sdr_a: &[Vec<i8>], sdr_b: &[Vec<i8>]) -> f64 {
    let mut correct = 0;
    for &(a, b, target) in examples { net.reset();
        let mut input = vec![0i8; net.h]; for i in 0..net.h { input[i] = sdr_a[a][i].saturating_add(sdr_b[b][i]); }
        for tick in 0..TICKS { net.propagate(&input, tick); }
        if proj.predict(&net.charge) == target { correct += 1; } }
    correct as f64 / examples.len() as f64
}

fn run_test(label: &str, leak_div: i16, train: &[(usize,usize,usize)], test: &[(usize,usize,usize)], all: &[(usize,usize,usize)], sdr_a: &[Vec<i8>], sdr_b: &[Vec<i8>]) {
    println!("--- {} (leak_div={}) ---", label, leak_div);

    for &seed in &[42u64, 1042, 2042] {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut net = GradedNet::new(H, leak_div, &mut rng);
        let mut proj = Proj::new(H, SUMS, &mut rng);

        // Init bridges
        for _ in 0..50 { let s = rng.gen_range(0..INPUT_END) as u16; let t = rng.gen_range(OUTPUT_START..H) as u16; net.add_edge(s, t); }

        let deadline = Instant::now() + Duration::from_secs(WALL_SECS);
        while Instant::now() < deadline {
            let before = eval(&mut net, &proj, train, sdr_a, sdr_b);
            let roll = rng.gen_range(0..100u32);
            if roll < 25 { let (pi,pc,old) = proj.mutate(&mut rng); let after = eval(&mut net, &proj, train, sdr_a, sdr_b); if after <= before { proj.undo(pi,pc,old); } }
            else { let genome = net.save(); if net.mutate(&mut rng) { let after = eval(&mut net, &proj, train, sdr_a, sdr_b); if after <= before { net.restore(genome); } } }
        }

        let train_acc = eval(&mut net, &proj, train, sdr_a, sdr_b);
        let test_acc = eval(&mut net, &proj, test, sdr_a, sdr_b);
        let all_acc = eval(&mut net, &proj, all, sdr_a, sdr_b);

        // Ablation: remove all edges
        let edges_bak = net.edges.clone();
        net.edges.clear();
        let no_edge_acc = eval(&mut net, &proj, all, sdr_a, sdr_b);
        net.edges = edges_bak;

        println!("  seed {}: train={:.0}% test={:.0}% all={:.0}% noedge={:.0}% edges={}",
            seed, train_acc*100.0, test_acc*100.0, all_acc*100.0, no_edge_acc*100.0, net.edges.len());
    }
    println!();
}

fn main() {
    let all = make_examples();
    let train: Vec<_> = all.iter().filter(|&&(_,_,s)| s != 4).cloned().collect();
    let test: Vec<_> = all.iter().filter(|&&(_,_,s)| s == 4).cloned().collect();
    let (sdr_a, sdr_b) = make_sdr(H);

    println!("=== GRADED POTENTIAL ADDITION ===");
    println!("incoming = spike + charge/leak_div (graded signal)");
    println!("H={}, {}s/seed, separated I/O, train sum≠4, test sum=4\n", H, WALL_SECS);

    // Sweep leak_div: higher = weaker graded signal
    run_test("Spike only (baseline)", i16::MAX, &train, &test, &all, &sdr_a, &sdr_b);
    run_test("Graded strong", 1, &train, &test, &all, &sdr_a, &sdr_b);
    run_test("Graded medium", 2, &train, &test, &all, &sdr_a, &sdr_b);
    run_test("Graded weak", 4, &train, &test, &all, &sdr_a, &sdr_b);
}
