//! Sweep 5: Cache A/B/C/D with ListNet
//!
//! Run: cargo run --example overnight_cache --release

use instnct_core::build_bigram_table;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

const VOCAB: usize = 27;
const WALL_CLOCK_SECS: u64 = 60;
const SEEDS: [u64; 3] = [42, 1042, 2042];
const TICKS: usize = 6;
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];
const EDGE_CAP: usize = 300;

#[derive(Clone)]
struct ListNet {
    topology: Vec<Vec<u16>>, threshold: Vec<u8>, channel: Vec<u8>, polarity: Vec<i8>,
    charge: Vec<i16>, activation: Vec<i8>, h: usize, input_dim: usize, output_start: usize,
}
impl ListNet {
    fn new(h: usize, rng: &mut impl Rng) -> Self {
        let phi_dim = (h as f64 / 1.618).round() as usize;
        let mut threshold = vec![0u8; h]; let mut channel = vec![0u8; h]; let mut polarity = vec![1i8; h];
        for i in 0..h { threshold[i] = rng.gen_range(0..=7); channel[i] = rng.gen_range(1..=8); if rng.gen_ratio(1, 10) { polarity[i] = -1; } }
        ListNet { topology: Vec::new(), threshold, channel, polarity, charge: vec![0; h], activation: vec![0; h], h, input_dim: phi_dim, output_start: h - phi_dim }
    }
    fn reset(&mut self) { self.charge.iter_mut().for_each(|c| *c = 0); self.activation.iter_mut().for_each(|a| *a = 0); }
    fn edge_count(&self) -> usize { self.topology.iter().map(|r| r.len().saturating_sub(1)).sum() }
    fn propagate(&mut self, input: &[i8], tick: usize) {
        let h = self.h;
        if tick < 2 { for i in 0..self.input_dim.min(input.len()) { self.charge[i] = self.charge[i].saturating_add(input[i] as i16); } }
        let mut incoming = vec![0i16; h];
        for row in &self.topology { if row.len() < 2 { continue; } let src = row[0] as usize; if src >= h { continue; }
            let act = self.activation[src]; if act != 0 { for &tgt in &row[1..] { let t = tgt as usize; if t < h { incoming[t] = incoming[t].saturating_add(act as i16); } } } }
        for i in 0..h {
            self.charge[i] = self.charge[i].saturating_add(incoming[i]);
            if tick % 6 == 5 && self.charge[i] > 0 { self.charge[i] -= 1; }
            let pi = (tick as u8 + 9 - self.channel[i]) & 7; let pm = PHASE_BASE[pi as usize];
            if self.charge[i] * 10 >= (self.threshold[i] as i16 + 1) * pm { self.activation[i] = self.polarity[i]; self.charge[i] = 0; } else { self.activation[i] = 0; }
        }
    }
    fn readout(&self, nc: usize) -> Vec<f64> { let mut out = vec![0.0f64; nc]; let zl = self.h - self.output_start; if zl == 0 || nc == 0 { return out; } for i in 0..zl { out[i * nc / zl] += self.activation[self.output_start + i] as f64; } out }
    fn add_edge(&mut self, source: u16, target: u16) -> bool {
        if source == target || source as usize >= self.h || target as usize >= self.h { return false; }
        if let Some(ri) = self.topology.iter().position(|row| row.first() == Some(&source)) {
            match self.topology[ri][1..].binary_search(&target) { Ok(_) => false, Err(pos) => { self.topology[ri].insert(1 + pos, target); true } }
        } else { let nr = vec![source, target]; let pos = self.topology.partition_point(|row| row[0] < source); self.topology.insert(pos, nr); true }
    }
    fn remove_edge(&mut self, rng: &mut impl Rng) -> bool {
        let total = self.edge_count(); if total == 0 { return false; } let pick = rng.gen_range(0..total); let mut c = 0;
        for ri in 0..self.topology.len() { let e = self.topology[ri].len() - 1; if c + e > pick { self.topology[ri].remove(1 + (pick - c)); if self.topology[ri].len() <= 1 { self.topology.remove(ri); } return true; } c += e; } false
    }
    fn rewire(&mut self, rng: &mut impl Rng) -> bool {
        let total = self.edge_count(); if total == 0 { return false; } let pick = rng.gen_range(0..total); let mut c = 0;
        for ri in 0..self.topology.len() { let e = self.topology[ri].len() - 1; if c + e > pick { let source = self.topology[ri][0]; let nt = rng.gen_range(0..self.h) as u16; if nt == source { return false; } self.topology[ri].remove(1 + (pick - c)); if self.topology[ri].len() <= 1 { self.topology.remove(ri); } return self.add_edge(source, nt); } c += e; } false
    }
    fn mutate(&mut self, rng: &mut impl Rng, cap: usize) -> bool {
        match rng.gen_range(0..100u32) {
            0..30 => { if self.edge_count() >= cap { return false; } self.add_edge(rng.gen_range(0..self.h) as u16, rng.gen_range(0..self.h) as u16) }
            30..45 => self.remove_edge(rng), 45..70 => self.rewire(rng),
            70..85 => { let n = rng.gen_range(0..self.h); let o = self.threshold[n]; self.threshold[n] = rng.gen_range(0..=15); self.threshold[n] != o }
            85..95 => { let n = rng.gen_range(0..self.h); let o = self.channel[n]; self.channel[n] = rng.gen_range(1..=8); self.channel[n] != o }
            _ => { self.polarity[rng.gen_range(0..self.h)] *= -1; true }
        }
    }
    fn save(&self) -> (Vec<Vec<u16>>, Vec<u8>, Vec<u8>, Vec<i8>) { (self.topology.clone(), self.threshold.clone(), self.channel.clone(), self.polarity.clone()) }
    fn restore(&mut self, s: (Vec<Vec<u16>>, Vec<u8>, Vec<u8>, Vec<i8>)) { self.topology = s.0; self.threshold = s.1; self.channel = s.2; self.polarity = s.3; }
}

fn eval_cos(net: &mut ListNet, corpus: &[u8], tokens: usize, rng: &mut StdRng, sdr: &[Vec<i8>], bigram: &[Vec<f64>]) -> f64 {
    let start = rng.gen_range(0..corpus.len().saturating_sub(tokens + 1).max(1));
    let mut total = 0.0f64; let mut count = 0usize; net.reset();
    for t in 0..tokens { let pos = start + t; if pos + 1 >= corpus.len() { break; }
        for tick in 0..TICKS { net.propagate(&sdr[corpus[pos] as usize], tick); }
        let out = net.readout(VOCAB); let tgt = &bigram[corpus[pos + 1] as usize];
        let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
        for j in 0..VOCAB { dot += out[j]*tgt[j]; na += out[j]*out[j]; nb += tgt[j]*tgt[j]; }
        total += dot / (na.sqrt() * nb.sqrt()).max(1e-10); count += 1;
    }
    if count > 0 { total / count as f64 } else { 0.0 }
}

fn eval_acc(net: &mut ListNet, corpus: &[u8], len: usize, rng: &mut StdRng, sdr: &[Vec<i8>]) -> f64 {
    let start = rng.gen_range(0..corpus.len().saturating_sub(len + 1).max(1));
    let mut correct = 0usize; let mut count = 0usize; net.reset();
    for t in 0..len { let pos = start + t; if pos + 1 >= corpus.len() { break; }
        for tick in 0..TICKS { net.propagate(&sdr[corpus[pos] as usize], tick); }
        let out = net.readout(VOCAB);
        let pred = out.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0);
        if pred == corpus[pos + 1] as usize { correct += 1; } count += 1;
    }
    if count > 0 { correct as f64 / count as f64 } else { 0.0 }
}

struct CacheConfig { label: &'static str, cache: &'static str, h: usize }

fn main() {
    let raw = std::fs::read_to_string("instnct-core/tests/fixtures/alice_corpus.txt").expect("corpus not found");
    let corpus: Vec<u8> = raw.bytes().map(|b| match b { b'a'..=b'z' => b-b'a', _ => 26 }).collect();
    let bigram = build_bigram_table(&corpus, VOCAB);

    let h_max = 8192;
    let sdr: Vec<Vec<i8>> = (0..VOCAB).map(|sym| {
        let mut rng = StdRng::seed_from_u64(sym as u64 + 9999);
        let input_dim = (h_max as f64 / 1.618).round() as usize;
        let mut p = vec![0i8; h_max]; let active = input_dim / 5; let mut placed = 0;
        while placed < active { let i = rng.gen_range(0..input_dim); if p[i]==0 { p[i]=1; placed+=1; } } p
    }).collect();

    let configs = vec![
        CacheConfig { label: "A1", cache: "L1", h: 512 },
        CacheConfig { label: "A2", cache: "L1", h: 1024 },
        CacheConfig { label: "A3", cache: "L1 edge", h: 1400 },
        CacheConfig { label: "B1", cache: "L2", h: 2048 },
        CacheConfig { label: "B2", cache: "L2", h: 4096 },
        CacheConfig { label: "B3", cache: "L2", h: 8192 },
    ];

    println!("=== SWEEP 5: ListNet Cache A/B/C/D ===");
    println!("{}s/seed | {} seeds | edge_cap={} | WSS=22*H+2*E", WALL_CLOCK_SECS, SEEDS.len(), EDGE_CAP);
    println!("{:>3} {:>8} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8}", "", "cache", "H", "WSS", "steps", "step/s", "best%", "mean%");
    println!("{:-<3} {:-<8} {:-<7} {:-<8} {:-<8} {:-<8} {:-<8} {:-<8}", "", "", "", "", "", "", "", "");

    for cfg in &configs {
        let wss = 22 * cfg.h + 2 * EDGE_CAP;
        let wss_str = if wss < 1024 { format!("{}B", wss) } else if wss < 1048576 { format!("{:.1}KB", wss as f64 / 1024.0) } else { format!("{:.1}MB", wss as f64 / 1048576.0) };

        let mut all_acc = Vec::new(); let mut all_steps = Vec::new();
        for &seed in &SEEDS {
            let sdr_t: Vec<Vec<i8>> = sdr.iter().map(|p| p[..cfg.h].to_vec()).collect();
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = ListNet::new(cfg.h, &mut rng);
            let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
            let mut steps = 0usize;
            let deadline = Instant::now() + Duration::from_secs(WALL_CLOCK_SECS);
            while Instant::now() < deadline {
                let sr = eval_rng.clone(); let before = eval_cos(&mut net, &corpus, 20, &mut eval_rng, &sdr_t, &bigram); eval_rng = sr;
                let genome = net.save(); let mutated = net.mutate(&mut rng, EDGE_CAP);
                if !mutated { let _ = eval_cos(&mut net, &corpus, 20, &mut eval_rng, &sdr_t, &bigram); steps += 1; continue; }
                let after = eval_cos(&mut net, &corpus, 20, &mut eval_rng, &sdr_t, &bigram);
                if after < before { net.restore(genome); } steps += 1;
            }
            let acc = eval_acc(&mut net, &corpus, 500, &mut eval_rng, &sdr_t);
            all_acc.push(acc); all_steps.push(steps);
        }
        let best = all_acc.iter().cloned().fold(0.0f64, f64::max);
        let mean = all_acc.iter().sum::<f64>() / all_acc.len() as f64;
        let ms = all_steps.iter().sum::<usize>() / all_steps.len();
        println!("{:>3} {:>8} {:>7} {:>8} {:>8} {:>8.0} {:>7.1}% {:>7.1}%",
            cfg.label, cfg.cache, cfg.h, wss_str, ms, ms as f64 / WALL_CLOCK_SECS as f64, best * 100.0, mean * 100.0);
    }
}
