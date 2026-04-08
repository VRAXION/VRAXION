//! Freeze-Prune: evolve → prune (remove each, keep only if score drops) → freeze survivors → repeat
//!
//! Run: cargo run --example freeze_prune --release

use instnct_core::build_bigram_table;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

const VOCAB: usize = 27;
const TICKS: usize = 6;
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];
const MUTABLE_CAP: usize = 100;
const CYCLES: usize = 10;
const STEPS_PER_CYCLE: usize = 20_000;
const H: usize = 1024;
const PRUNE_EVAL_TOKENS: usize = 40; // more tokens for prune accuracy

#[derive(Clone)]
struct FreezeNet {
    frozen: Vec<Vec<u16>>,
    mutable: Vec<Vec<u16>>,
    threshold: Vec<u8>,
    channel: Vec<u8>,
    polarity: Vec<i8>,
    charge: Vec<i16>,
    activation: Vec<i8>,
    h: usize,
    input_dim: usize,
    output_start: usize,
}

impl FreezeNet {
    fn new(h: usize, rng: &mut impl Rng) -> Self {
        let phi_dim = (h as f64 / 1.618).round() as usize;
        let mut threshold = vec![0u8; h]; let mut channel = vec![0u8; h]; let mut polarity = vec![1i8; h];
        for i in 0..h { threshold[i] = rng.gen_range(0..=7); channel[i] = rng.gen_range(1..=8); if rng.gen_ratio(1, 10) { polarity[i] = -1; } }
        FreezeNet { frozen: Vec::new(), mutable: Vec::new(), threshold, channel, polarity,
            charge: vec![0; h], activation: vec![0; h], h, input_dim: phi_dim, output_start: h - phi_dim }
    }

    fn reset(&mut self) { self.charge.iter_mut().for_each(|c| *c = 0); self.activation.iter_mut().for_each(|a| *a = 0); }

    fn frozen_edges(&self) -> usize { self.frozen.iter().map(|r| r.len().saturating_sub(1)).sum() }
    fn mutable_edges(&self) -> usize { self.mutable.iter().map(|r| r.len().saturating_sub(1)).sum() }
    fn total_edges(&self) -> usize { self.frozen_edges() + self.mutable_edges() }

    fn propagate(&mut self, input: &[i8], tick: usize) {
        let h = self.h;
        if tick < 2 { for i in 0..self.input_dim.min(input.len()) { self.charge[i] = self.charge[i].saturating_add(input[i] as i16); } }
        let mut incoming = vec![0i16; h];
        for topo in [&self.frozen, &self.mutable] {
            for row in topo {
                if row.len() < 2 { continue; } let src = row[0] as usize; if src >= h { continue; }
                let act = self.activation[src]; if act != 0 {
                    for &tgt in &row[1..] { let t = tgt as usize; if t < h { incoming[t] = incoming[t].saturating_add(act as i16); } }
                }
            }
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

    fn readout(&self, nc: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; nc]; let zl = self.h - self.output_start;
        if zl == 0 || nc == 0 { return out; }
        for i in 0..zl { out[i * nc / zl] += self.activation[self.output_start + i] as f64; } out
    }

    fn add_mutable(&mut self, src: u16, tgt: u16) -> bool {
        if src == tgt || src as usize >= self.h || tgt as usize >= self.h { return false; }
        // Check frozen
        for row in &self.frozen { if row.first() == Some(&src) && row[1..].contains(&tgt) { return false; } }
        if let Some(ri) = self.mutable.iter().position(|row| row.first() == Some(&src)) {
            match self.mutable[ri][1..].binary_search(&tgt) { Ok(_) => false, Err(pos) => { self.mutable[ri].insert(1 + pos, tgt); true } }
        } else { let pos = self.mutable.partition_point(|row| row[0] < src); self.mutable.insert(pos, vec![src, tgt]); true }
    }

    fn remove_mutable(&mut self, rng: &mut impl Rng) -> bool {
        let total = self.mutable_edges(); if total == 0 { return false; }
        let pick = rng.gen_range(0..total); let mut c = 0;
        for ri in 0..self.mutable.len() { let e = self.mutable[ri].len() - 1; if c + e > pick {
            self.mutable[ri].remove(1 + (pick - c)); if self.mutable[ri].len() <= 1 { self.mutable.remove(ri); } return true; } c += e; } false
    }

    fn rewire_mutable(&mut self, rng: &mut impl Rng) -> bool {
        let total = self.mutable_edges(); if total == 0 { return false; }
        let pick = rng.gen_range(0..total); let mut c = 0;
        for ri in 0..self.mutable.len() { let e = self.mutable[ri].len() - 1; if c + e > pick {
            let src = self.mutable[ri][0]; let nt = rng.gen_range(0..self.h) as u16; if nt == src { return false; }
            self.mutable[ri].remove(1 + (pick - c)); if self.mutable[ri].len() <= 1 { self.mutable.remove(ri); }
            return self.add_mutable(src, nt); } c += e; } false
    }

    fn mutate(&mut self, rng: &mut impl Rng) -> bool {
        match rng.gen_range(0..100u32) {
            0..30 => { if self.mutable_edges() >= MUTABLE_CAP { return false; } self.add_mutable(rng.gen_range(0..self.h) as u16, rng.gen_range(0..self.h) as u16) }
            30..45 => self.remove_mutable(rng), 45..70 => self.rewire_mutable(rng),
            70..85 => { let n = rng.gen_range(0..self.h); self.threshold[n] = rng.gen_range(0..=15); true }
            85..95 => { let n = rng.gen_range(0..self.h); self.channel[n] = rng.gen_range(1..=8); true }
            _ => { self.polarity[rng.gen_range(0..self.h)] *= -1; true }
        }
    }

    /// Collect all mutable edges as flat list
    fn collect_mutable_edges(&self) -> Vec<(u16, u16)> {
        let mut edges = Vec::new();
        for row in &self.mutable {
            if row.len() < 2 { continue; }
            let src = row[0];
            for &tgt in &row[1..] { edges.push((src, tgt)); }
        }
        edges
    }

    /// Remove a specific edge from mutable
    fn remove_specific_mutable(&mut self, src: u16, tgt: u16) -> bool {
        if let Some(ri) = self.mutable.iter().position(|row| row.first() == Some(&src)) {
            if let Ok(pos) = self.mutable[ri][1..].binary_search(&tgt) {
                self.mutable[ri].remove(1 + pos);
                if self.mutable[ri].len() <= 1 { self.mutable.remove(ri); }
                return true;
            }
        }
        false
    }

    /// Add edge to frozen
    fn add_frozen(&mut self, src: u16, tgt: u16) {
        if let Some(fri) = self.frozen.iter().position(|r| r.first() == Some(&src)) {
            if self.frozen[fri][1..].binary_search(&tgt).is_err() {
                let pos = self.frozen[fri][1..].partition_point(|&x| x < tgt);
                self.frozen[fri].insert(1 + pos, tgt);
            }
        } else {
            let pos = self.frozen.partition_point(|r| r[0] < src);
            self.frozen.insert(pos, vec![src, tgt]);
        }
    }

    /// Prune-crystallize: test each mutable edge, freeze only those that matter
    fn prune_crystallize(&mut self, corpus: &[u8], eval_rng: &mut StdRng, sdr: &[Vec<i8>], bigram: &[Vec<f64>]) -> (usize, usize) {
        let edges = self.collect_mutable_edges();
        let total = edges.len();

        // Baseline score with all mutable edges
        let baseline = eval_cos_static(self, corpus, PRUNE_EVAL_TOKENS, eval_rng, sdr, bigram);

        let mut kept = 0usize;
        let mut pruned = 0usize;

        // Test each edge: remove → eval → if worse, restore and freeze
        for (src, tgt) in &edges {
            // Remove this edge temporarily
            self.remove_specific_mutable(*src, *tgt);

            // Eval without it
            let eval_snap = eval_rng.clone();
            let without = eval_cos_static(self, corpus, PRUNE_EVAL_TOKENS, eval_rng, sdr, bigram);
            *eval_rng = eval_snap;

            if without < baseline - 0.001 {
                // Score dropped → this edge matters → freeze it
                self.add_frozen(*src, *tgt);
                kept += 1;
            } else {
                // Score same or better → edge was noise → leave it removed
                pruned += 1;
            }
        }

        // Clear any remaining mutable (shouldn't be any, but safety)
        self.mutable.clear();

        (kept, pruned)
    }

    fn save_mutable(&self) -> (Vec<Vec<u16>>, Vec<u8>, Vec<u8>, Vec<i8>) {
        (self.mutable.clone(), self.threshold.clone(), self.channel.clone(), self.polarity.clone())
    }
    fn restore_mutable(&mut self, s: (Vec<Vec<u16>>, Vec<u8>, Vec<u8>, Vec<i8>)) {
        self.mutable = s.0; self.threshold = s.1; self.channel = s.2; self.polarity = s.3;
    }
}

fn eval_cos_static(net: &mut FreezeNet, corpus: &[u8], tokens: usize, rng: &mut StdRng, sdr: &[Vec<i8>], bigram: &[Vec<f64>]) -> f64 {
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

fn eval_acc(net: &mut FreezeNet, corpus: &[u8], len: usize, rng: &mut StdRng, sdr: &[Vec<i8>]) -> f64 {
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

    let seeds = [42u64, 1042, 2042];

    println!("Freeze-Prune Evolution — H={}, cap={}, {}steps/cycle, {} cycles",
        H, MUTABLE_CAP, STEPS_PER_CYCLE, CYCLES);
    println!("Crystallize = remove each edge, freeze only if score drops");
    println!("Corpus: {} bytes | Seeds: {:?}", corpus.len(), seeds);
    println!("================================================================\n");

    for &seed in &seeds {
        println!("--- seed {} ---", seed);
        println!("{:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
            "cycle", "frozen", "mut_pre", "kept", "pruned", "total", "acc%");
        println!("{:-<6} {:-<8} {:-<8} {:-<8} {:-<8} {:-<8} {:-<8}", "", "", "", "", "", "", "");

        let mut rng = StdRng::seed_from_u64(seed);
        let mut net = FreezeNet::new(H, &mut rng);
        let mut eval_rng = StdRng::seed_from_u64(seed + 1000);

        for cycle in 0..CYCLES {
            // Evolve phase
            for _ in 0..STEPS_PER_CYCLE {
                let sr = eval_rng.clone();
                let before = eval_cos_static(&mut net, &corpus, 20, &mut eval_rng, &sdr, &bigram);
                eval_rng = sr;
                let genome = net.save_mutable();
                let mutated = net.mutate(&mut rng);
                if !mutated { let _ = eval_cos_static(&mut net, &corpus, 20, &mut eval_rng, &sdr, &bigram); continue; }
                let after = eval_cos_static(&mut net, &corpus, 20, &mut eval_rng, &sdr, &bigram);
                if after < before { net.restore_mutable(genome); }
            }

            let mut_pre = net.mutable_edges();

            // Prune-crystallize: test each edge individually
            let (kept, pruned) = net.prune_crystallize(&corpus, &mut eval_rng, &sdr, &bigram);

            let acc = eval_acc(&mut net, &corpus, 500, &mut eval_rng, &sdr);

            println!("{:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>7.1}%",
                cycle, net.frozen_edges(), mut_pre, kept, pruned, net.total_edges(), acc * 100.0);
        }
        println!();
    }

    println!("Comparison: naive freeze-all gets ~20% oscillating, flat cap=300 gets ~20%");
}
