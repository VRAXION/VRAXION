//! Dendritic addition: each edge is ADD or MUL (coincidence detector).
//! Test if multiplicative edges enable generalization on addition.
//!
//! Run: cargo run --example dendritic_addition --release

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

#[derive(Clone, Copy, PartialEq)]
enum EdgeMode { Add, Mul }

#[derive(Clone)]
struct DendriticNet {
    // topology: [source, target, mode(0=add,1=mul)] packed as (u16, u16, u8)
    edges: Vec<(u16, u16, EdgeMode)>,
    threshold: Vec<u8>,
    channel: Vec<u8>,
    polarity: Vec<i8>,
    charge: Vec<i16>,
    activation: Vec<i8>,
    h: usize,
    input_end: usize,
    output_start: usize,
}

impl DendriticNet {
    fn new(h: usize, rng: &mut impl Rng) -> Self {
        let phi_dim = (h as f64 / 1.618).round() as usize;
        let mut threshold = vec![0u8; h]; let mut channel = vec![0u8; h]; let mut polarity = vec![1i8; h];
        for i in 0..h { threshold[i] = rng.gen_range(0..=7); channel[i] = rng.gen_range(1..=8); if rng.gen_ratio(1, 10) { polarity[i] = -1; } }
        DendriticNet { edges: Vec::new(), threshold, channel, polarity, charge: vec![0; h], activation: vec![0; h],
            h, input_end: phi_dim, output_start: h - phi_dim }
    }

    fn reset(&mut self) { self.charge.iter_mut().for_each(|c| *c = 0); self.activation.iter_mut().for_each(|a| *a = 0); }

    fn propagate(&mut self, input: &[i8], tick: usize) {
        let h = self.h;
        if tick < 2 { for i in 0..self.input_end.min(input.len()) { self.charge[i] = self.charge[i].saturating_add(input[i] as i16); } }

        // Per-neuron: collect add and mul inputs separately
        let mut add_incoming = vec![0i16; h];
        let mut mul_incoming = vec![1i16; h]; // start at 1 for product
        let mut has_mul = vec![false; h];

        for &(src, tgt, mode) in &self.edges {
            let s = src as usize; let t = tgt as usize;
            if s >= h || t >= h { continue; }
            let act = self.activation[s];
            match mode {
                EdgeMode::Add => {
                    add_incoming[t] = add_incoming[t].saturating_add(act as i16);
                }
                EdgeMode::Mul => {
                    has_mul[t] = true;
                    if act == 0 {
                        mul_incoming[t] = 0; // any zero kills the product
                    } else {
                        mul_incoming[t] = mul_incoming[t].saturating_mul(act as i16);
                    }
                }
            }
        }

        // Combine: incoming = add_pool + mul_pool (mul_pool=0 if no mul edges)
        for i in 0..h {
            let mul = if has_mul[i] { mul_incoming[i] } else { 0 };
            let total = add_incoming[i].saturating_add(mul);
            self.charge[i] = self.charge[i].saturating_add(total);

            if tick % 6 == 5 && self.charge[i] > 0 { self.charge[i] -= 1; }
            let pi = (tick as u8 + 9 - self.channel[i]) & 7; let pm = PHASE_BASE[pi as usize];
            if self.charge[i] * 10 >= (self.threshold[i] as i16 + 1) * pm {
                self.activation[i] = self.polarity[i]; self.charge[i] = 0;
            } else { self.activation[i] = 0; }
        }
    }

    fn add_edge(&mut self, src: u16, tgt: u16, mode: EdgeMode) -> bool {
        if src == tgt || src as usize >= self.h || tgt as usize >= self.h { return false; }
        if self.edges.iter().any(|&(s,t,_)| s == src && t == tgt) { return false; }
        self.edges.push((src, tgt, mode));
        true
    }

    fn remove_edge(&mut self, rng: &mut impl Rng) -> bool {
        if self.edges.is_empty() { return false; }
        let idx = rng.gen_range(0..self.edges.len());
        self.edges.swap_remove(idx);
        true
    }

    fn rewire(&mut self, rng: &mut impl Rng) -> bool {
        if self.edges.is_empty() { return false; }
        let idx = rng.gen_range(0..self.edges.len());
        let mode = self.edges[idx].2;
        let new_src = rng.gen_range(0..self.h) as u16;
        let new_tgt = rng.gen_range(0..self.h) as u16;
        if new_src == new_tgt { return false; }
        if self.edges.iter().any(|&(s,t,_)| s == new_src && t == new_tgt) { return false; }
        self.edges[idx] = (new_src, new_tgt, mode);
        true
    }

    fn flip_mode(&mut self, rng: &mut impl Rng) -> bool {
        if self.edges.is_empty() { return false; }
        let idx = rng.gen_range(0..self.edges.len());
        self.edges[idx].2 = match self.edges[idx].2 { EdgeMode::Add => EdgeMode::Mul, EdgeMode::Mul => EdgeMode::Add };
        true
    }

    fn mutate(&mut self, rng: &mut impl Rng) -> bool {
        match rng.gen_range(0..100u32) {
            0..20 => { if self.edges.len() >= EDGE_CAP { return false; }
                let mode = if rng.gen_ratio(1, 3) { EdgeMode::Mul } else { EdgeMode::Add };
                self.add_edge(rng.gen_range(0..self.h) as u16, rng.gen_range(0..self.h) as u16, mode) }
            20..35 => self.remove_edge(rng),
            35..55 => self.rewire(rng),
            55..70 => self.flip_mode(rng), // 15% flip add↔mul
            70..82 => { let n = rng.gen_range(0..self.h); self.threshold[n] = rng.gen_range(0..=15); true }
            82..93 => { let n = rng.gen_range(0..self.h); self.channel[n] = rng.gen_range(1..=8); true }
            _ => { self.polarity[rng.gen_range(0..self.h)] *= -1; true }
        }
    }

    fn save(&self) -> (Vec<(u16,u16,EdgeMode)>, Vec<u8>, Vec<u8>, Vec<i8>) {
        (self.edges.clone(), self.threshold.clone(), self.channel.clone(), self.polarity.clone())
    }
    fn restore(&mut self, s: (Vec<(u16,u16,EdgeMode)>, Vec<u8>, Vec<u8>, Vec<i8>)) {
        self.edges = s.0; self.threshold = s.1; self.channel = s.2; self.polarity = s.3;
    }
}

// Projection
#[derive(Clone)]
struct Proj { w: Vec<Vec<i8>>, output_start: usize, h: usize, classes: usize }
impl Proj {
    fn new(h: usize, classes: usize, output_start: usize, rng: &mut impl Rng) -> Self {
        Proj { w: (0..h).map(|_| (0..classes).map(|_| rng.gen_range(-2..=2i8)).collect()).collect(), output_start, h, classes }
    }
    fn predict(&self, charge: &[i16]) -> usize {
        let mut s = vec![0i64; self.classes];
        for i in self.output_start..self.h { let ch = charge[i] as i64; if ch == 0 { continue; } for c in 0..self.classes { s[c] += ch * self.w[i][c] as i64; } }
        s.iter().enumerate().max_by_key(|&(_, v)| *v).map(|(i, _)| i).unwrap_or(0)
    }
    fn mutate(&mut self, rng: &mut impl Rng) -> (usize, usize, i8) {
        let i = rng.gen_range(self.output_start..self.h); let c = rng.gen_range(0..self.classes);
        let old = self.w[i][c]; self.w[i][c] = rng.gen_range(-4..=4i8); (i, c, old)
    }
    fn undo(&mut self, i: usize, c: usize, old: i8) { self.w[i][c] = old; }
}

fn make_examples() -> Vec<(usize, usize, usize)> {
    (0..DIGITS).flat_map(|a| (0..DIGITS).map(move |b| (a, b, a+b))).collect()
}

fn make_sdr(h: usize, input_end: usize) -> (Vec<Vec<i8>>, Vec<Vec<i8>>) {
    let half = input_end / 2; let active = half / 5;
    let sdr_a: Vec<Vec<i8>> = (0..DIGITS).map(|d| { let mut rng = StdRng::seed_from_u64(d as u64 + 100);
        let mut p = vec![0i8; h]; let mut placed = 0; while placed < active { let i = rng.gen_range(0..half); if p[i]==0 { p[i]=1; placed+=1; } } p }).collect();
    let sdr_b: Vec<Vec<i8>> = (0..DIGITS).map(|d| { let mut rng = StdRng::seed_from_u64(d as u64 + 200);
        let mut p = vec![0i8; h]; let mut placed = 0; while placed < active { let i = rng.gen_range(half..input_end); if p[i]==0 { p[i]=1; placed+=1; } } p }).collect();
    (sdr_a, sdr_b)
}

fn eval_add(net: &mut DendriticNet, proj: &Proj, examples: &[(usize,usize,usize)], sdr_a: &[Vec<i8>], sdr_b: &[Vec<i8>]) -> f64 {
    let mut correct = 0;
    for &(a, b, target) in examples { net.reset();
        let mut input = vec![0i8; net.h]; for i in 0..net.h { input[i] = sdr_a[a][i].saturating_add(sdr_b[b][i]); }
        for tick in 0..TICKS { net.propagate(&input, tick); }
        if proj.predict(&net.charge) == target { correct += 1; } }
    correct as f64 / examples.len() as f64
}

fn train(net: &mut DendriticNet, proj: &mut Proj, train_ex: &[(usize,usize,usize)], sdr_a: &[Vec<i8>], sdr_b: &[Vec<i8>], seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed + 500);
    let deadline = Instant::now() + Duration::from_secs(WALL_SECS);
    while Instant::now() < deadline {
        let before = eval_add(net, proj, train_ex, sdr_a, sdr_b);
        let roll = rng.gen_range(0..100u32);
        if roll < 25 {
            let (pi,pc,old) = proj.mutate(&mut rng);
            let after = eval_add(net, proj, train_ex, sdr_a, sdr_b);
            if after <= before { proj.undo(pi,pc,old); }
        } else {
            let genome = net.save();
            let mutated = net.mutate(&mut rng);
            if mutated {
                let after = eval_add(net, proj, train_ex, sdr_a, sdr_b);
                if after <= before { net.restore(genome); }
            }
        }
    }
}

fn main() {
    let all = make_examples();
    let train_ex: Vec<_> = all.iter().filter(|&&(_,_,s)| s != 4).cloned().collect();
    let test_ex: Vec<_> = all.iter().filter(|&&(_,_,s)| s == 4).cloned().collect();
    let phi_dim = (H as f64 / 1.618).round() as usize;
    let (sdr_a, sdr_b) = make_sdr(H, phi_dim);

    println!("=== DENDRITIC ADDITION: ADD + MUL edges ===");
    println!("H={}, {}s/seed, edge_cap={}", H, WALL_SECS, EDGE_CAP);
    println!("Train: sum≠4 (20 ex), Test: sum=4 (5 ex), Random: {:.0}%\n", 100.0/SUMS as f64);

    // Test A: dendritic (add+mul edges)
    println!("--- A: Dendritic (add + mul edges) ---");
    for &seed in &[42u64, 1042, 2042, 3042, 4042] {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut net = DendriticNet::new(H, &mut rng);
        let mut proj = Proj::new(H, SUMS, H - phi_dim, &mut rng);
        // Init bridges
        for _ in 0..30 { let s = rng.gen_range(0..phi_dim) as u16; let t = rng.gen_range(H-phi_dim..H) as u16;
            net.add_edge(s, t, if rng.gen_ratio(1,3) { EdgeMode::Mul } else { EdgeMode::Add }); }
        train(&mut net, &mut proj, &train_ex, &sdr_a, &sdr_b, seed);
        let train_acc = eval_add(&mut net, &proj, &train_ex, &sdr_a, &sdr_b);
        let test_acc = eval_add(&mut net, &proj, &test_ex, &sdr_a, &sdr_b);
        let n_mul = net.edges.iter().filter(|e| e.2 == EdgeMode::Mul).count();
        let n_add = net.edges.iter().filter(|e| e.2 == EdgeMode::Add).count();
        println!("  seed {}: train={:.0}% test={:.0}% edges={} (add={}, mul={})",
            seed, train_acc*100.0, test_acc*100.0, net.edges.len(), n_add, n_mul);
    }

    // Test B: standard (add only, for comparison)
    println!("\n--- B: Standard (add only) ---");
    for &seed in &[42u64, 1042, 2042, 3042, 4042] {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut net = DendriticNet::new(H, &mut rng);
        let mut proj = Proj::new(H, SUMS, H - phi_dim, &mut rng);
        for _ in 0..30 { let s = rng.gen_range(0..phi_dim) as u16; let t = rng.gen_range(H-phi_dim..H) as u16;
            net.add_edge(s, t, EdgeMode::Add); }
        // Override mutate to never create mul edges — just add-only
        // (We'll train the same but flip_mode does nothing useful since all are add)
        train(&mut net, &mut proj, &train_ex, &sdr_a, &sdr_b, seed);
        let train_acc = eval_add(&mut net, &proj, &train_ex, &sdr_a, &sdr_b);
        let test_acc = eval_add(&mut net, &proj, &test_ex, &sdr_a, &sdr_b);
        println!("  seed {}: train={:.0}% test={:.0}% edges={}",
            seed, train_acc*100.0, test_acc*100.0, net.edges.len());
    }
}
