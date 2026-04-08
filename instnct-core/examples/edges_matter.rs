//! Do edges matter? Deterministic test.
//!
//! 1. Evolve a network with cheap eval (get ~300 edges, ~20% acc)
//! 2. Measure accuracy with ALL corpus (deterministic, no random start)
//! 3. Remove ALL edges → measure again
//! 4. Restore edges → measure again (sanity check)
//! 5. Shuffle edges randomly → measure again
//! 6. Also: train params-only (no edge mutations), compare
//!
//! Run: cargo run --example edges_matter --release

use instnct_core::build_bigram_table;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

const VOCAB: usize = 27;
const TICKS: usize = 6;
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];
const H: usize = 512;
const EDGE_CAP: usize = 300;
const TRAIN_STEPS: usize = 50_000;

#[derive(Clone)]
struct ListNet {
    topology: Vec<Vec<u16>>,
    threshold: Vec<u8>, channel: Vec<u8>, polarity: Vec<i8>,
    charge: Vec<i16>, activation: Vec<i8>,
    h: usize, input_dim: usize, output_start: usize,
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
    fn readout(&self, nc: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; nc]; let zl = self.h - self.output_start;
        if zl == 0 || nc == 0 { return out; }
        for i in 0..zl { out[i * nc / zl] += self.activation[self.output_start + i] as f64; } out
    }
    fn add_edge(&mut self, src: u16, tgt: u16) -> bool {
        if src == tgt || src as usize >= self.h || tgt as usize >= self.h { return false; }
        if let Some(ri) = self.topology.iter().position(|row| row.first() == Some(&src)) {
            match self.topology[ri][1..].binary_search(&tgt) { Ok(_) => false, Err(pos) => { self.topology[ri].insert(1 + pos, tgt); true } }
        } else { let pos = self.topology.partition_point(|row| row[0] < src); self.topology.insert(pos, vec![src, tgt]); true }
    }
    fn remove_edge(&mut self, rng: &mut impl Rng) -> bool {
        let total = self.edge_count(); if total == 0 { return false; } let pick = rng.gen_range(0..total); let mut c = 0;
        for ri in 0..self.topology.len() { let e = self.topology[ri].len() - 1; if c + e > pick { self.topology[ri].remove(1 + (pick - c)); if self.topology[ri].len() <= 1 { self.topology.remove(ri); } return true; } c += e; } false
    }
    fn rewire(&mut self, rng: &mut impl Rng) -> bool {
        let total = self.edge_count(); if total == 0 { return false; } let pick = rng.gen_range(0..total); let mut c = 0;
        for ri in 0..self.topology.len() { let e = self.topology[ri].len() - 1; if c + e > pick { let src = self.topology[ri][0]; let nt = rng.gen_range(0..self.h) as u16; if nt == src { return false; } self.topology[ri].remove(1 + (pick - c)); if self.topology[ri].len() <= 1 { self.topology.remove(ri); } return self.add_edge(src, nt); } c += e; } false
    }

    // Full mutation (edges + params)
    fn mutate_full(&mut self, rng: &mut impl Rng) -> bool {
        match rng.gen_range(0..100u32) {
            0..30 => { if self.edge_count() >= EDGE_CAP { return false; } self.add_edge(rng.gen_range(0..self.h) as u16, rng.gen_range(0..self.h) as u16) }
            30..45 => self.remove_edge(rng), 45..70 => self.rewire(rng),
            70..85 => { let n = rng.gen_range(0..self.h); self.threshold[n] = rng.gen_range(0..=15); true }
            85..95 => { let n = rng.gen_range(0..self.h); self.channel[n] = rng.gen_range(1..=8); true }
            _ => { self.polarity[rng.gen_range(0..self.h)] *= -1; true }
        }
    }

    // Params-only mutation (NO edge changes)
    fn mutate_params_only(&mut self, rng: &mut impl Rng) -> bool {
        match rng.gen_range(0..100u32) {
            0..50 => { let n = rng.gen_range(0..self.h); self.threshold[n] = rng.gen_range(0..=15); true }
            50..85 => { let n = rng.gen_range(0..self.h); self.channel[n] = rng.gen_range(1..=8); true }
            _ => { self.polarity[rng.gen_range(0..self.h)] *= -1; true }
        }
    }

    fn save(&self) -> (Vec<Vec<u16>>, Vec<u8>, Vec<u8>, Vec<i8>) { (self.topology.clone(), self.threshold.clone(), self.channel.clone(), self.polarity.clone()) }
    fn restore(&mut self, s: (Vec<Vec<u16>>, Vec<u8>, Vec<u8>, Vec<i8>)) { self.topology = s.0; self.threshold = s.1; self.channel = s.2; self.polarity = s.3; }
}

/// Deterministic eval over ENTIRE corpus (no random start)
fn eval_deterministic(net: &mut ListNet, corpus: &[u8], sdr: &[Vec<i8>]) -> (f64, usize, usize) {
    let mut correct = 0usize; let mut count = 0usize;
    net.reset();
    for pos in 0..corpus.len().saturating_sub(1) {
        for tick in 0..TICKS { net.propagate(&sdr[corpus[pos] as usize], tick); }
        let out = net.readout(VOCAB);
        let pred = out.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0);
        if pred == corpus[pos + 1] as usize { correct += 1; }
        count += 1;
    }
    (if count > 0 { correct as f64 / count as f64 } else { 0.0 }, correct, count)
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

fn train(net: &mut ListNet, corpus: &[u8], sdr: &[Vec<i8>], bigram: &[Vec<f64>], seed: u64, params_only: bool) -> usize {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let mut steps = 0;
    for _ in 0..TRAIN_STEPS {
        let sr = eval_rng.clone();
        let before = eval_cos(net, corpus, 20, &mut eval_rng, sdr, bigram);
        eval_rng = sr;
        let genome = net.save();
        let mutated = if params_only { net.mutate_params_only(&mut rng) } else { net.mutate_full(&mut rng) };
        if !mutated { let _ = eval_cos(net, corpus, 20, &mut eval_rng, sdr, bigram); steps += 1; continue; }
        let after = eval_cos(net, corpus, 20, &mut eval_rng, sdr, bigram);
        if after < before { net.restore(genome); }
        steps += 1;
    }
    steps
}

fn main() {
    let raw = std::fs::read_to_string("instnct-core/tests/fixtures/alice_corpus.txt").expect("corpus not found");
    let corpus: Vec<u8> = raw.bytes().map(|b| match b { b'a'..=b'z' => b-b'a', _ => 26 }).collect();
    let bigram = build_bigram_table(&corpus, VOCAB);

    let sdr: Vec<Vec<i8>> = (0..VOCAB).map(|sym| {
        let mut rng = StdRng::seed_from_u64(sym as u64 + 9999);
        let input_dim = (H as f64 / 1.618).round() as usize;
        let mut p = vec![0i8; H]; let active = input_dim / 5; let mut placed = 0;
        while placed < active { let i = rng.gen_range(0..input_dim); if p[i]==0 { p[i]=1; placed+=1; } } p
    }).collect();

    println!("=== DO EDGES MATTER? Deterministic test ===\n");

    // --- Test 1: Train with edges, then ablate ---
    println!("--- Test 1: Train full (edges+params), then ablate ---");
    let mut net = ListNet::new(H, &mut StdRng::seed_from_u64(42));
    train(&mut net, &corpus, &sdr, &bigram, 42, false);

    let (acc_full, cor_full, cnt) = eval_deterministic(&mut net, &corpus, &sdr);
    println!("  Trained (edges+params): {:.1}% ({}/{}) edges={}", acc_full * 100.0, cor_full, cnt, net.edge_count());

    let topo_backup = net.topology.clone();
    net.topology.clear();
    let (acc_no_edges, cor_no, _) = eval_deterministic(&mut net, &corpus, &sdr);
    println!("  Edges REMOVED:          {:.1}% ({}/{}) edges=0", acc_no_edges * 100.0, cor_no, cnt);

    net.topology = topo_backup;
    let (acc_restored, _, _) = eval_deterministic(&mut net, &corpus, &sdr);
    println!("  Edges RESTORED:         {:.1}% (sanity check)", acc_restored * 100.0);

    // Shuffle edges randomly (same count, random targets)
    let edge_count = net.edge_count();
    net.topology.clear();
    let mut rng = StdRng::seed_from_u64(9999);
    while net.edge_count() < edge_count {
        let s = rng.gen_range(0..H) as u16;
        let t = rng.gen_range(0..H) as u16;
        net.add_edge(s, t);
    }
    let (acc_random, _, _) = eval_deterministic(&mut net, &corpus, &sdr);
    println!("  RANDOM edges (same #):  {:.1}% edges={}", acc_random * 100.0, net.edge_count());

    let edge_diff = (acc_full - acc_no_edges) * 100.0;
    println!("\n  Edge contribution: {:.1}pp ({:.1}% - {:.1}%)", edge_diff, acc_full * 100.0, acc_no_edges * 100.0);
    if edge_diff.abs() < 1.0 { println!("  *** EDGES DON'T MATTER on this task ***"); }
    else { println!("  *** EDGES MATTER: {:.1}pp improvement ***", edge_diff); }

    // --- Test 2: Params-only training (no edges at all) ---
    println!("\n--- Test 2: Train PARAMS ONLY (zero edges ever) ---");
    let mut net2 = ListNet::new(H, &mut StdRng::seed_from_u64(42));
    train(&mut net2, &corpus, &sdr, &bigram, 42, true);
    let (acc_params, cor_p, _) = eval_deterministic(&mut net2, &corpus, &sdr);
    println!("  Params-only trained:    {:.1}% ({}/{}) edges=0", acc_params * 100.0, cor_p, cnt);

    println!("\n--- Summary ---");
    println!("  Full trained:     {:.1}%", acc_full * 100.0);
    println!("  Edges removed:    {:.1}%", acc_no_edges * 100.0);
    println!("  Random edges:     {:.1}%", acc_random * 100.0);
    println!("  Params-only:      {:.1}%", acc_params * 100.0);
}
