//! FlatNet — fixed-size array topology, zero heap allocation per neuron.
//!
//! connections[neuron] = [t0, t1, t2, ..., 0, 0, 0]  (max 8 targets)
//! counts[neuron] = how many are real
//!
//! Run: cargo run --example flatnet --release

use instnct_core::build_bigram_table;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

const VOCAB: usize = 27;
const WALL_CLOCK_SECS: u64 = 5;
const SEEDS: [u64; 3] = [42, 1042, 2042];
const TICKS: usize = 6;
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];
const MAX_TARGETS: usize = 16; // max fan-out per neuron

#[derive(Clone)]
struct FlatNet {
    /// Fixed-size target arrays: connections[neuron][0..counts[neuron]]
    connections: Vec<[u16; MAX_TARGETS]>,
    counts: Vec<u8>,
    /// Per-neuron params
    threshold: Vec<u8>,
    channel: Vec<u8>,
    polarity: Vec<i8>,
    /// Runtime
    charge: Vec<i16>,
    activation: Vec<i8>,
    /// Geometry
    h: usize,
    input_dim: usize,
    output_start: usize,
}

impl FlatNet {
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
        FlatNet {
            connections: vec![[0u16; MAX_TARGETS]; h],
            counts: vec![0u8; h],
            threshold, channel, polarity,
            charge: vec![0; h],
            activation: vec![0; h],
            h,
            input_dim: phi_dim,
            output_start: h - phi_dim,
        }
    }

    fn reset(&mut self) {
        self.charge.iter_mut().for_each(|c| *c = 0);
        self.activation.iter_mut().for_each(|a| *a = 0);
    }

    fn edge_count(&self) -> usize {
        self.counts.iter().map(|&c| c as usize).sum()
    }

    fn propagate(&mut self, input: &[i8], tick: usize) {
        let h = self.h;

        if tick < 2 {
            for i in 0..self.input_dim.min(input.len()) {
                self.charge[i] = self.charge[i].saturating_add(input[i] as i16);
            }
        }

        // Scatter from fixed-size arrays
        let mut incoming = vec![0i16; h];
        for src in 0..h {
            let act = self.activation[src];
            if act == 0 { continue; }
            let cnt = self.counts[src] as usize;
            for j in 0..cnt {
                let tgt = self.connections[src][j] as usize;
                incoming[tgt] = incoming[tgt].saturating_add(act as i16);
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
            let c = i * nc / zl;
            out[c] += self.activation[self.output_start + i] as f64;
        }
        out
    }

    fn add_edge(&mut self, src: u16, tgt: u16) -> bool {
        if src == tgt { return false; }
        let s = src as usize;
        if s >= self.h || tgt as usize >= self.h { return false; }
        let cnt = self.counts[s] as usize;
        if cnt >= MAX_TARGETS { return false; }
        // Duplicate check
        for j in 0..cnt {
            if self.connections[s][j] == tgt { return false; }
        }
        self.connections[s][cnt] = tgt;
        self.counts[s] += 1;
        true
    }

    fn remove_edge(&mut self, rng: &mut impl Rng) -> bool {
        let total = self.edge_count();
        if total == 0 { return false; }
        let pick = rng.gen_range(0..total);
        let mut acc = 0;
        for s in 0..self.h {
            let cnt = self.counts[s] as usize;
            if acc + cnt > pick {
                let idx = pick - acc;
                // Swap-remove
                self.counts[s] -= 1;
                let last = self.counts[s] as usize;
                self.connections[s][idx] = self.connections[s][last];
                return true;
            }
            acc += cnt;
        }
        false
    }

    fn rewire(&mut self, rng: &mut impl Rng) -> bool {
        let total = self.edge_count();
        if total == 0 { return false; }
        let pick = rng.gen_range(0..total);
        let mut acc = 0;
        for s in 0..self.h {
            let cnt = self.counts[s] as usize;
            if acc + cnt > pick {
                let idx = pick - acc;
                let new_tgt = rng.gen_range(0..self.h) as u16;
                if new_tgt == s as u16 { return false; }
                // Duplicate check
                for j in 0..cnt {
                    if self.connections[s][j] == new_tgt { return false; }
                }
                self.connections[s][idx] = new_tgt;
                return true;
            }
            acc += cnt;
        }
        false
    }

    fn mutate(&mut self, rng: &mut impl Rng, edge_cap: usize) -> bool {
        let roll = rng.gen_range(0..100u32);
        match roll {
            0..30 => {
                if self.edge_count() >= edge_cap { return false; }
                let s = rng.gen_range(0..self.h) as u16;
                let t = rng.gen_range(0..self.h) as u16;
                self.add_edge(s, t)
            }
            30..45 => self.remove_edge(rng),
            45..70 => self.rewire(rng),
            70..85 => {
                let n = rng.gen_range(0..self.h);
                let old = self.threshold[n];
                self.threshold[n] = rng.gen_range(0..=15);
                self.threshold[n] != old
            }
            85..95 => {
                let n = rng.gen_range(0..self.h);
                let old = self.channel[n];
                self.channel[n] = rng.gen_range(1..=8);
                self.channel[n] != old
            }
            _ => {
                let n = rng.gen_range(0..self.h);
                self.polarity[n] *= -1;
                true
            }
        }
    }

    fn save(&self) -> (Vec<[u16; MAX_TARGETS]>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<i8>) {
        (self.connections.clone(), self.counts.clone(), self.threshold.clone(), self.channel.clone(), self.polarity.clone())
    }

    fn restore(&mut self, s: (Vec<[u16; MAX_TARGETS]>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<i8>)) {
        self.connections = s.0; self.counts = s.1; self.threshold = s.2; self.channel = s.3; self.polarity = s.4;
    }
}

fn eval_cos(
    net: &mut FlatNet, corpus: &[u8], tokens: usize,
    rng: &mut StdRng, sdr: &[Vec<i8>], bigram: &[Vec<f64>],
) -> f64 {
    let start = rng.gen_range(0..corpus.len().saturating_sub(tokens + 1).max(1));
    let mut total = 0.0f64; let mut count = 0usize;
    net.reset();
    for t in 0..tokens {
        let pos = start + t;
        if pos + 1 >= corpus.len() { break; }
        let sym = corpus[pos] as usize;
        let tsym = corpus[pos + 1] as usize;
        for tick in 0..TICKS { net.propagate(&sdr[sym], tick); }
        let out = net.readout(VOCAB);
        let tgt = &bigram[tsym];
        let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
        for j in 0..VOCAB { dot += out[j]*tgt[j]; na += out[j]*out[j]; nb += tgt[j]*tgt[j]; }
        total += dot / (na.sqrt() * nb.sqrt()).max(1e-10);
        count += 1;
    }
    if count > 0 { total / count as f64 } else { 0.0 }
}

fn eval_acc(
    net: &mut FlatNet, corpus: &[u8], len: usize,
    rng: &mut StdRng, sdr: &[Vec<i8>],
) -> f64 {
    let start = rng.gen_range(0..corpus.len().saturating_sub(len + 1).max(1));
    let mut correct = 0usize; let mut count = 0usize;
    net.reset();
    for t in 0..len {
        let pos = start + t;
        if pos + 1 >= corpus.len() { break; }
        let sym = corpus[pos] as usize;
        let tsym = corpus[pos + 1] as usize;
        for tick in 0..TICKS { net.propagate(&sdr[sym], tick); }
        let out = net.readout(VOCAB);
        let pred = out.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i,_)| i).unwrap_or(0);
        if pred == tsym { correct += 1; }
        count += 1;
    }
    if count > 0 { correct as f64 / count as f64 } else { 0.0 }
}

fn main() {
    let raw = std::fs::read_to_string("instnct-core/tests/fixtures/alice_corpus.txt")
        .expect("corpus not found");
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
        (256, 300), (512, 300), (1024, 300), (2048, 300),
    ];

    println!("FlatNet — Fixed-Size Array Topology (max {} targets/neuron)", MAX_TARGETS);
    println!("Zero heap alloc per neuron. connections: [[u16; {}]; H]", MAX_TARGETS);
    println!("Wall clock: {}s/seed | Seeds: {:?} | Corpus: {} bytes",
        WALL_CLOCK_SECS, SEEDS, corpus.len());
    println!("================================================================\n");

    println!("{:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "H", "cap", "edges", "steps", "step/s", "best%", "mean%", "mem_KB");
    println!("{:-<6} {:-<8} {:-<8} {:-<8} {:-<8} {:-<8} {:-<8} {:-<8}",
        "", "", "", "", "", "", "", "");

    for &(h, cap) in &configs {
        let mut all_acc = Vec::new();
        let mut all_steps = Vec::new();
        let mut all_edges = Vec::new();

        for &seed in &SEEDS {
            let sdr_t: Vec<Vec<i8>> = sdr.iter().map(|p| p[..h].to_vec()).collect();
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = FlatNet::new(h, &mut rng);
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
        // Memory: connections = h * MAX_TARGETS * 2, counts = h, params = h * 3
        let mem_kb = (h * MAX_TARGETS * 2 + h * 4) / 1024;

        println!("{:>6} {:>8} {:>8} {:>8} {:>8.0} {:>7.1}% {:>7.1}% {:>8}",
            h, cap, me, ms, ms as f64 / WALL_CLOCK_SECS as f64,
            best * 100.0, mean * 100.0, mem_kb);
    }

    println!("\nComparison:");
    println!("  ListNet  H=256: 3917 step/s, 20.8% best");
    println!("  INSTNCT  H=256:  654 step/s, 20.4% best");
}
