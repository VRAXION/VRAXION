//! Overnight ListNet sweep: 5 seeds, 120s/seed, H=256-4096
//!
//! Run: cargo run --example overnight_listnet_sweep --release

use instnct_core::build_bigram_table;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

const VOCAB: usize = 27;
const WALL_CLOCK_SECS: u64 = 120;
const SEEDS: [u64; 5] = [42, 1042, 2042, 3042, 4042];
const TICKS: usize = 6;
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];

#[derive(Clone)]
struct ListNet {
    topology: Vec<Vec<u16>>,
    threshold: Vec<u8>,
    channel: Vec<u8>,
    polarity: Vec<i8>,
    charge: Vec<i16>,
    activation: Vec<i8>,
    h: usize,
    input_dim: usize,
    output_start: usize,
}

impl ListNet {
    fn new(h: usize, rng: &mut impl Rng) -> Self {
        let phi_dim = (h as f64 / 1.618).round() as usize;
        let mut threshold = vec![0u8; h];
        let mut channel = vec![0u8; h];
        let mut polarity = vec![1i8; h];
        for i in 0..h {
            threshold[i] = rng.gen_range(0..=7);
            channel[i] = rng.gen_range(1..=8);
            if rng.gen_ratio(1, 10) { polarity[i] = -1; }
        }
        ListNet {
            topology: Vec::new(), threshold, channel, polarity,
            charge: vec![0; h], activation: vec![0; h],
            h, input_dim: phi_dim, output_start: h - phi_dim,
        }
    }

    fn reset(&mut self) {
        self.charge.iter_mut().for_each(|c| *c = 0);
        self.activation.iter_mut().for_each(|a| *a = 0);
    }

    fn edge_count(&self) -> usize {
        self.topology.iter().map(|r| r.len().saturating_sub(1)).sum()
    }

    fn propagate(&mut self, input: &[i8], tick: usize) {
        let h = self.h;
        if tick < 2 {
            for i in 0..self.input_dim.min(input.len()) {
                self.charge[i] = self.charge[i].saturating_add(input[i] as i16);
            }
        }
        let mut incoming = vec![0i16; h];
        for row in &self.topology {
            if row.len() < 2 { continue; }
            let src = row[0] as usize;
            if src >= h { continue; }
            let act = self.activation[src];
            if act != 0 {
                for &tgt in &row[1..] {
                    let t = tgt as usize;
                    if t < h { incoming[t] = incoming[t].saturating_add(act as i16); }
                }
            }
        }
        for i in 0..h {
            self.charge[i] = self.charge[i].saturating_add(incoming[i]);
            if tick % 6 == 5 && self.charge[i] > 0 { self.charge[i] -= 1; }
            let pi = (tick as u8 + 9 - self.channel[i]) & 7;
            let pm = PHASE_BASE[pi as usize];
            let te = (self.threshold[i] as i16 + 1) * pm;
            if self.charge[i] * 10 >= te {
                self.activation[i] = self.polarity[i];
                self.charge[i] = 0;
            } else {
                self.activation[i] = 0;
            }
        }
    }

    fn readout(&self, nc: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; nc];
        let zl = self.h - self.output_start;
        if zl == 0 || nc == 0 { return out; }
        for i in 0..zl {
            out[i * nc / zl] += self.activation[self.output_start + i] as f64;
        }
        out
    }

    fn find_row(&self, source: u16) -> Option<usize> {
        self.topology.iter().position(|row| row.first() == Some(&source))
    }

    fn add_edge(&mut self, source: u16, target: u16) -> bool {
        if source == target || source as usize >= self.h || target as usize >= self.h { return false; }
        if let Some(ri) = self.find_row(source) {
            match self.topology[ri][1..].binary_search(&target) {
                Ok(_) => false,
                Err(pos) => { self.topology[ri].insert(1 + pos, target); true }
            }
        } else {
            let new_row = vec![source, target];
            let pos = self.topology.partition_point(|row| row[0] < source);
            self.topology.insert(pos, new_row);
            true
        }
    }

    fn remove_edge(&mut self, rng: &mut impl Rng) -> bool {
        let total = self.edge_count();
        if total == 0 { return false; }
        let pick = rng.gen_range(0..total);
        let mut count = 0;
        for ri in 0..self.topology.len() {
            let edges = self.topology[ri].len() - 1;
            if count + edges > pick {
                let ti = 1 + (pick - count);
                self.topology[ri].remove(ti);
                if self.topology[ri].len() <= 1 { self.topology.remove(ri); }
                return true;
            }
            count += edges;
        }
        false
    }

    fn rewire(&mut self, rng: &mut impl Rng) -> bool {
        let total = self.edge_count();
        if total == 0 { return false; }
        let pick = rng.gen_range(0..total);
        let mut count = 0;
        for ri in 0..self.topology.len() {
            let edges = self.topology[ri].len() - 1;
            if count + edges > pick {
                let ti = 1 + (pick - count);
                let source = self.topology[ri][0];
                let new_tgt = rng.gen_range(0..self.h) as u16;
                if new_tgt == source { return false; }
                self.topology[ri].remove(ti);
                if self.topology[ri].len() <= 1 { self.topology.remove(ri); }
                return self.add_edge(source, new_tgt);
            }
            count += edges;
        }
        false
    }

    fn mutate(&mut self, rng: &mut impl Rng, edge_cap: usize) -> bool {
        let roll = rng.gen_range(0..100u32);
        match roll {
            0..30 => { if self.edge_count() >= edge_cap { return false; } let s = rng.gen_range(0..self.h) as u16; let t = rng.gen_range(0..self.h) as u16; self.add_edge(s, t) }
            30..45 => self.remove_edge(rng),
            45..70 => self.rewire(rng),
            70..85 => { let n = rng.gen_range(0..self.h); let o = self.threshold[n]; self.threshold[n] = rng.gen_range(0..=15); self.threshold[n] != o }
            85..95 => { let n = rng.gen_range(0..self.h); let o = self.channel[n]; self.channel[n] = rng.gen_range(1..=8); self.channel[n] != o }
            _ => { let n = rng.gen_range(0..self.h); self.polarity[n] *= -1; true }
        }
    }

    fn save(&self) -> (Vec<Vec<u16>>, Vec<u8>, Vec<u8>, Vec<i8>) {
        (self.topology.clone(), self.threshold.clone(), self.channel.clone(), self.polarity.clone())
    }
    fn restore(&mut self, s: (Vec<Vec<u16>>, Vec<u8>, Vec<u8>, Vec<i8>)) {
        self.topology = s.0; self.threshold = s.1; self.channel = s.2; self.polarity = s.3;
    }
}

fn eval_cos(net: &mut ListNet, corpus: &[u8], tokens: usize, rng: &mut StdRng, sdr: &[Vec<i8>], bigram: &[Vec<f64>]) -> f64 {
    let start = rng.gen_range(0..corpus.len().saturating_sub(tokens + 1).max(1));
    let mut total = 0.0f64; let mut count = 0usize;
    net.reset();
    for t in 0..tokens {
        let pos = start + t;
        if pos + 1 >= corpus.len() { break; }
        for tick in 0..TICKS { net.propagate(&sdr[corpus[pos] as usize], tick); }
        let out = net.readout(VOCAB);
        let tgt = &bigram[corpus[pos + 1] as usize];
        let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
        for j in 0..VOCAB { dot += out[j]*tgt[j]; na += out[j]*out[j]; nb += tgt[j]*tgt[j]; }
        total += dot / (na.sqrt() * nb.sqrt()).max(1e-10);
        count += 1;
    }
    if count > 0 { total / count as f64 } else { 0.0 }
}

fn eval_acc(net: &mut ListNet, corpus: &[u8], len: usize, rng: &mut StdRng, sdr: &[Vec<i8>]) -> f64 {
    let start = rng.gen_range(0..corpus.len().saturating_sub(len + 1).max(1));
    let mut correct = 0usize; let mut count = 0usize;
    net.reset();
    for t in 0..len {
        let pos = start + t;
        if pos + 1 >= corpus.len() { break; }
        for tick in 0..TICKS { net.propagate(&sdr[corpus[pos] as usize], tick); }
        let out = net.readout(VOCAB);
        let pred = out.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0);
        if pred == corpus[pos + 1] as usize { correct += 1; }
        count += 1;
    }
    if count > 0 { correct as f64 / count as f64 } else { 0.0 }
}

fn main() {
    let raw = std::fs::read_to_string("instnct-core/tests/fixtures/alice_corpus.txt").expect("corpus not found");
    let corpus: Vec<u8> = raw.bytes().map(|b| match b { b'a'..=b'z' => b-b'a', _ => 26 }).collect();
    let bigram = build_bigram_table(&corpus, VOCAB);

    let h_max = 4096;
    let sdr: Vec<Vec<i8>> = (0..VOCAB).map(|sym| {
        let mut rng = StdRng::seed_from_u64(sym as u64 + 9999);
        let input_dim = (h_max as f64 / 1.618).round() as usize;
        let mut p = vec![0i8; h_max];
        let active = input_dim / 5;
        let mut placed = 0;
        while placed < active { let i = rng.gen_range(0..input_dim); if p[i]==0 { p[i]=1; placed+=1; } }
        p
    }).collect();

    let configs: Vec<(usize, usize)> = vec![
        (256, 300), (512, 300), (1024, 300), (2048, 300), (4096, 300),
    ];

    println!("=== SWEEP 1: ListNet Full Sweep ===");
    println!("{}s/seed | {} seeds | Corpus: {} bytes", WALL_CLOCK_SECS, SEEDS.len(), corpus.len());
    println!("{:>6} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>40}",
        "H", "cap", "edges", "steps", "step/s", "best%", "mean%", "all_seeds");
    println!("{:-<6} {:-<6} {:-<8} {:-<8} {:-<8} {:-<8} {:-<8} {:-<40}",
        "", "", "", "", "", "", "", "");

    for &(h, cap) in &configs {
        let mut all_acc = Vec::new();
        let mut all_steps = Vec::new();
        let mut all_edges = Vec::new();

        for &seed in &SEEDS {
            let sdr_t: Vec<Vec<i8>> = sdr.iter().map(|p| p[..h].to_vec()).collect();
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = ListNet::new(h, &mut rng);
            let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
            let mut steps = 0usize;
            let deadline = Instant::now() + Duration::from_secs(WALL_CLOCK_SECS);

            while Instant::now() < deadline {
                let snap_rng = eval_rng.clone();
                let before = eval_cos(&mut net, &corpus, 20, &mut eval_rng, &sdr_t, &bigram);
                eval_rng = snap_rng;
                let genome = net.save();
                let mutated = net.mutate(&mut rng, cap);
                if !mutated {
                    let _ = eval_cos(&mut net, &corpus, 20, &mut eval_rng, &sdr_t, &bigram);
                    steps += 1; continue;
                }
                let after = eval_cos(&mut net, &corpus, 20, &mut eval_rng, &sdr_t, &bigram);
                if after < before { net.restore(genome); }
                steps += 1;
            }

            let acc = eval_acc(&mut net, &corpus, 500, &mut eval_rng, &sdr_t);
            all_acc.push(acc);
            all_steps.push(steps);
            all_edges.push(net.edge_count());
        }

        let best = all_acc.iter().cloned().fold(0.0f64, f64::max);
        let mean = all_acc.iter().sum::<f64>() / all_acc.len() as f64;
        let ms = all_steps.iter().sum::<usize>() / all_steps.len();
        let me = all_edges.iter().sum::<usize>() / all_edges.len();
        let seeds_str: Vec<String> = all_acc.iter().map(|a| format!("{:.1}%", a * 100.0)).collect();

        println!("{:>6} {:>6} {:>8} {:>8} {:>8.0} {:>7.1}% {:>7.1}% {:>40}",
            h, cap, me, ms, ms as f64 / WALL_CLOCK_SECS as f64,
            best * 100.0, mean * 100.0, seeds_str.join(", "));
    }

    // Sweep 2: Edge cap sweep at H=1024
    println!("\n=== SWEEP 2: Edge Cap Sweep (H=1024) ===");
    let edge_caps = [100, 200, 300, 500, 1000];
    let h = 1024;
    println!("{:>6} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "H", "cap", "edges", "steps", "step/s", "best%", "mean%");
    println!("{:-<6} {:-<6} {:-<8} {:-<8} {:-<8} {:-<8} {:-<8}",
        "", "", "", "", "", "", "");

    for &cap in &edge_caps {
        let mut all_acc = Vec::new();
        let mut all_steps = Vec::new();
        let mut all_edges = Vec::new();

        for &seed in &SEEDS {
            let sdr_t: Vec<Vec<i8>> = sdr.iter().map(|p| p[..h].to_vec()).collect();
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = ListNet::new(h, &mut rng);
            let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
            let mut steps = 0usize;
            let deadline = Instant::now() + Duration::from_secs(60);

            while Instant::now() < deadline {
                let snap_rng = eval_rng.clone();
                let before = eval_cos(&mut net, &corpus, 20, &mut eval_rng, &sdr_t, &bigram);
                eval_rng = snap_rng;
                let genome = net.save();
                let mutated = net.mutate(&mut rng, cap);
                if !mutated {
                    let _ = eval_cos(&mut net, &corpus, 20, &mut eval_rng, &sdr_t, &bigram);
                    steps += 1; continue;
                }
                let after = eval_cos(&mut net, &corpus, 20, &mut eval_rng, &sdr_t, &bigram);
                if after < before { net.restore(genome); }
                steps += 1;
            }

            let acc = eval_acc(&mut net, &corpus, 500, &mut eval_rng, &sdr_t);
            all_acc.push(acc);
            all_steps.push(steps);
            all_edges.push(net.edge_count());
        }

        let best = all_acc.iter().cloned().fold(0.0f64, f64::max);
        let mean = all_acc.iter().sum::<f64>() / all_acc.len() as f64;
        let ms = all_steps.iter().sum::<usize>() / all_steps.len();
        let me = all_edges.iter().sum::<usize>() / all_edges.len();

        println!("{:>6} {:>6} {:>8} {:>8} {:>8.0} {:>7.1}% {:>7.1}%",
            h, cap, me, ms, ms as f64 / 60.0, best * 100.0, mean * 100.0);
    }
}
