//! Bio combo: all biological features combined.
//! Add+Mul edges, Dale's Law (E/I), separated I/O, charge projection.
//!
//! Run: cargo run --example bio_combo --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

const DIGITS: usize = 5;
const SUMS: usize = 9;
const H: usize = 256;
const TICKS: usize = 12; // more ticks for deeper computation
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];
const EDGE_CAP: usize = 300;
const WALL_SECS: u64 = 180; // 3 min/seed for more search time
const INPUT_END: usize = 128;   // separated I/O
const OUTPUT_START: usize = 128;

#[derive(Clone, Copy, PartialEq)]
enum EdgeMode { Add, Mul }

#[derive(Clone)]
struct BioNet {
    edges: Vec<(u16, u16, EdgeMode)>,
    threshold: Vec<u8>,
    channel: Vec<u8>,
    is_excitatory: Vec<bool>,  // Dale's Law: true=E, false=I
    charge: Vec<i16>,
    activation: Vec<i8>,
    h: usize,
}

impl BioNet {
    fn new(h: usize, rng: &mut impl Rng) -> Self {
        let mut threshold = vec![0u8; h];
        let mut channel = vec![0u8; h];
        let mut is_excitatory = vec![true; h];
        for i in 0..h {
            threshold[i] = rng.gen_range(0..=7);
            channel[i] = rng.gen_range(1..=8);
            // Dale's Law: 70% excitatory, 30% inhibitory — FIXED at creation
            if rng.gen_ratio(3, 10) { is_excitatory[i] = false; }
        }
        BioNet { edges: Vec::new(), threshold, channel, is_excitatory,
            charge: vec![0; h], activation: vec![0; h], h }
    }

    fn reset(&mut self) {
        self.charge.iter_mut().for_each(|c| *c = 0);
        self.activation.iter_mut().for_each(|a| *a = 0);
    }

    fn polarity(&self, neuron: usize) -> i8 {
        if self.is_excitatory[neuron] { 1 } else { -1 }
    }

    fn propagate(&mut self, input: &[i8], tick: usize) {
        let h = self.h;
        if tick < 2 {
            for i in 0..INPUT_END.min(input.len()) {
                self.charge[i] = self.charge[i].saturating_add(input[i] as i16);
            }
        }

        let mut add_in = vec![0i16; h];
        let mut mul_in = vec![1i16; h];
        let mut has_mul = vec![false; h];

        for &(src, tgt, mode) in &self.edges {
            let s = src as usize; let t = tgt as usize;
            if s >= h || t >= h { continue; }
            // Dale's Law: output sign determined by source neuron type
            let act = self.activation[s] as i16 * self.polarity(s) as i16;
            match mode {
                EdgeMode::Add => { add_in[t] = add_in[t].saturating_add(act); }
                EdgeMode::Mul => {
                    has_mul[t] = true;
                    if act == 0 { mul_in[t] = 0; }
                    else { mul_in[t] = mul_in[t].saturating_mul(act); }
                }
            }
        }

        for i in 0..h {
            let mul = if has_mul[i] { mul_in[i] } else { 0 };
            let total = add_in[i].saturating_add(mul);
            self.charge[i] = self.charge[i].saturating_add(total);
            if tick % 6 == 5 && self.charge[i] > 0 { self.charge[i] -= 1; }
            let pi = (tick as u8 + 9 - self.channel[i]) & 7;
            let pm = PHASE_BASE[pi as usize];
            if self.charge[i] * 10 >= (self.threshold[i] as i16 + 1) * pm {
                // Activation is always +1 (polarity applied at output via Dale's Law)
                self.activation[i] = 1;
                self.charge[i] = 0;
            } else {
                self.activation[i] = 0;
            }
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
        self.edges.swap_remove(rng.gen_range(0..self.edges.len())); true
    }

    fn rewire(&mut self, rng: &mut impl Rng) -> bool {
        if self.edges.is_empty() { return false; }
        let idx = rng.gen_range(0..self.edges.len());
        let mode = self.edges[idx].2;
        let s = rng.gen_range(0..self.h) as u16; let t = rng.gen_range(0..self.h) as u16;
        if s == t { return false; }
        if self.edges.iter().any(|&(es,et,_)| es == s && et == t) { return false; }
        self.edges[idx] = (s, t, mode); true
    }

    fn flip_mode(&mut self, rng: &mut impl Rng) -> bool {
        if self.edges.is_empty() { return false; }
        let idx = rng.gen_range(0..self.edges.len());
        self.edges[idx].2 = match self.edges[idx].2 { EdgeMode::Add => EdgeMode::Mul, EdgeMode::Mul => EdgeMode::Add }; true
    }

    fn mutate(&mut self, rng: &mut impl Rng) -> bool {
        match rng.gen_range(0..100u32) {
            0..20 => { if self.edges.len() >= EDGE_CAP { return false; }
                self.add_edge(rng.gen_range(0..self.h) as u16, rng.gen_range(0..self.h) as u16,
                    if rng.gen_ratio(1,3) { EdgeMode::Mul } else { EdgeMode::Add }) }
            20..35 => self.remove_edge(rng),
            35..55 => self.rewire(rng),
            55..70 => self.flip_mode(rng),
            70..85 => { let n = rng.gen_range(0..self.h); self.threshold[n] = rng.gen_range(0..=15); true }
            85..95 => { let n = rng.gen_range(0..self.h); self.channel[n] = rng.gen_range(1..=8); true }
            _ => { /* Dale's Law: polarity is FIXED. Skip. */ false }
        }
    }

    fn save(&self) -> (Vec<(u16,u16,EdgeMode)>, Vec<u8>, Vec<u8>) {
        (self.edges.clone(), self.threshold.clone(), self.channel.clone())
    }
    fn restore(&mut self, s: (Vec<(u16,u16,EdgeMode)>, Vec<u8>, Vec<u8>)) {
        self.edges = s.0; self.threshold = s.1; self.channel = s.2;
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

fn eval(net: &mut BioNet, proj: &Proj, examples: &[(usize,usize,usize)], sdr_a: &[Vec<i8>], sdr_b: &[Vec<i8>]) -> f64 {
    let mut correct = 0;
    for &(a, b, target) in examples { net.reset();
        let mut input = vec![0i8; net.h]; for i in 0..net.h { input[i] = sdr_a[a][i].saturating_add(sdr_b[b][i]); }
        for tick in 0..TICKS { net.propagate(&input, tick); }
        if proj.predict(&net.charge) == target { correct += 1; } }
    correct as f64 / examples.len() as f64
}

fn main() {
    let all = make_examples();
    let train: Vec<_> = all.iter().filter(|&&(_,_,s)| s != 4).cloned().collect();
    let test: Vec<_> = all.iter().filter(|&&(_,_,s)| s == 4).cloned().collect();
    let (sdr_a, sdr_b) = make_sdr(H);

    println!("=== BIO COMBO: Add+Mul, Dale's Law, Separated I/O, 12 ticks ===");
    println!("H={}, {}s/seed, cap={}, ticks={}", H, WALL_SECS, EDGE_CAP, TICKS);
    println!("Train: sum≠4 (20), Test: sum=4 (5), Random: {:.0}%\n", 100.0/SUMS as f64);

    for &seed in &[42u64, 1042, 2042, 3042, 4042] {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut net = BioNet::new(H, &mut rng);
        let mut proj = Proj::new(H, SUMS, &mut rng);

        // Init: 50 bridges from input→output with mix of add/mul
        for _ in 0..50 {
            let s = rng.gen_range(0..INPUT_END) as u16;
            let t = rng.gen_range(OUTPUT_START..H) as u16;
            let mode = if rng.gen_ratio(1, 3) { EdgeMode::Mul } else { EdgeMode::Add };
            net.add_edge(s, t, mode);
        }

        let deadline = Instant::now() + Duration::from_secs(WALL_SECS);
        while Instant::now() < deadline {
            let before = eval(&mut net, &proj, &train, &sdr_a, &sdr_b);
            let roll = rng.gen_range(0..100u32);
            if roll < 25 {
                let (pi,pc,old) = proj.mutate(&mut rng);
                let after = eval(&mut net, &proj, &train, &sdr_a, &sdr_b);
                if after <= before { proj.undo(pi,pc,old); }
            } else {
                let genome = net.save();
                if net.mutate(&mut rng) {
                    let after = eval(&mut net, &proj, &train, &sdr_a, &sdr_b);
                    if after <= before { net.restore(genome); }
                }
            }
        }

        let train_acc = eval(&mut net, &proj, &train, &sdr_a, &sdr_b);
        let test_acc = eval(&mut net, &proj, &test, &sdr_a, &sdr_b);
        let all_acc = eval(&mut net, &proj, &all, &sdr_a, &sdr_b);
        let n_e = net.is_excitatory.iter().filter(|&&e| e).count();
        let n_mul = net.edges.iter().filter(|e| e.2 == EdgeMode::Mul).count();

        // Also test: remove all mul edges → does it drop?
        let save = net.edges.clone();
        net.edges.retain(|e| e.2 != EdgeMode::Mul);
        let no_mul_acc = eval(&mut net, &proj, &all, &sdr_a, &sdr_b);
        net.edges = save;

        println!("  seed {}: train={:.0}% test={:.0}% all={:.0}% | edges={} (mul={}) E/I={}/{} | no_mul={:.0}%",
            seed, train_acc*100.0, test_acc*100.0, all_acc*100.0,
            net.edges.len(), n_mul, n_e, H-n_e, no_mul_acc*100.0);
    }
}
