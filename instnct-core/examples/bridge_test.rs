//! Bridge test: zero overlap + pre-built input→output bridges + charge projection
//!
//! Run: cargo run --example bridge_test --release

use instnct_core::build_bigram_table;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const VOCAB: usize = 27;
const TICKS: usize = 6;
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];
const H: usize = 512;
const EDGE_CAP: usize = 300;
const TRAIN_STEPS: usize = 50_000;
const INPUT_END: usize = 256;
const OUTPUT_START: usize = 256;

#[derive(Clone)]
struct Projection {
    w: Vec<Vec<i8>>, h: usize, vocab: usize, output_start: usize,
}
impl Projection {
    fn new(h: usize, vocab: usize, output_start: usize, rng: &mut impl Rng) -> Self {
        Projection { w: (0..h).map(|_| (0..vocab).map(|_| rng.gen_range(-2..=2i8)).collect()).collect(), h, vocab, output_start }
    }
    fn predict(&self, charge: &[i16]) -> Vec<f64> {
        let mut s = vec![0.0f64; self.vocab];
        for i in self.output_start..self.h { let ch = charge[i] as f64; if ch == 0.0 { continue; } for c in 0..self.vocab { s[c] += ch * self.w[i][c] as f64; } }
        s
    }
    fn mutate(&mut self, rng: &mut impl Rng) -> (usize, usize, i8) {
        let i = rng.gen_range(self.output_start..self.h); let c = rng.gen_range(0..self.vocab);
        let old = self.w[i][c]; self.w[i][c] = rng.gen_range(-4..=4i8); (i, c, old)
    }
    fn undo(&mut self, i: usize, c: usize, old: i8) { self.w[i][c] = old; }
}

#[derive(Clone)]
struct Net {
    topology: Vec<Vec<u16>>,
    threshold: Vec<u8>, channel: Vec<u8>, polarity: Vec<i8>,
    charge: Vec<i16>, activation: Vec<i8>,
    h: usize,
}
impl Net {
    fn new(h: usize, rng: &mut impl Rng) -> Self {
        let mut threshold = vec![0u8; h]; let mut channel = vec![0u8; h]; let mut polarity = vec![1i8; h];
        for i in 0..h { threshold[i] = rng.gen_range(0..=7); channel[i] = rng.gen_range(1..=8); if rng.gen_ratio(1, 10) { polarity[i] = -1; } }
        Net { topology: Vec::new(), threshold, channel, polarity, charge: vec![0; h], activation: vec![0; h], h }
    }
    fn reset(&mut self) { self.charge.iter_mut().for_each(|c| *c = 0); self.activation.iter_mut().for_each(|a| *a = 0); }
    fn edge_count(&self) -> usize { self.topology.iter().map(|r| r.len().saturating_sub(1)).sum() }
    fn propagate(&mut self, input: &[i8], tick: usize) {
        let h = self.h;
        if tick < 2 { for i in 0..INPUT_END.min(input.len()) { self.charge[i] = self.charge[i].saturating_add(input[i] as i16); } }
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
    fn save_topo(&self) -> Vec<Vec<u16>> { self.topology.clone() }
    fn restore_topo(&mut self, t: Vec<Vec<u16>>) { self.topology = t; }

    /// Pre-build direct bridges from input→output zone
    fn add_bridges(&mut self, count: usize, rng: &mut impl Rng) {
        let mut added = 0;
        while added < count {
            let src = rng.gen_range(0..INPUT_END) as u16;
            let tgt = rng.gen_range(OUTPUT_START..self.h) as u16;
            if self.add_edge(src, tgt) { added += 1; }
        }
    }

    /// Pre-build 2-hop chains: input→hidden→output (hidden = random neuron)
    fn add_chains(&mut self, count: usize, rng: &mut impl Rng) {
        for _ in 0..count {
            let src = rng.gen_range(0..INPUT_END) as u16;
            let mid = rng.gen_range(0..self.h) as u16;
            let tgt = rng.gen_range(OUTPUT_START..self.h) as u16;
            self.add_edge(src, mid);
            self.add_edge(mid, tgt);
        }
    }
}

fn eval_cos(net: &mut Net, proj: &Projection, corpus: &[u8], tokens: usize, rng: &mut StdRng, sdr: &[Vec<i8>], bigram: &[Vec<f64>]) -> f64 {
    let start = rng.gen_range(0..corpus.len().saturating_sub(tokens + 1).max(1));
    let mut total = 0.0f64; let mut count = 0usize; net.reset();
    for t in 0..tokens { let pos = start + t; if pos + 1 >= corpus.len() { break; }
        for tick in 0..TICKS { net.propagate(&sdr[corpus[pos] as usize], tick); }
        let out = proj.predict(&net.charge); let tgt = &bigram[corpus[pos + 1] as usize];
        let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
        for j in 0..VOCAB { dot += out[j]*tgt[j]; na += out[j]*out[j]; nb += tgt[j]*tgt[j]; }
        total += dot / (na.sqrt() * nb.sqrt()).max(1e-10); count += 1;
    }
    if count > 0 { total / count as f64 } else { 0.0 }
}

fn eval_acc(net: &mut Net, proj: &Projection, corpus: &[u8], sdr: &[Vec<i8>]) -> f64 {
    let mut correct = 0usize; let mut count = 0usize; net.reset();
    for pos in 0..corpus.len().saturating_sub(1).min(10000) {
        for tick in 0..TICKS { net.propagate(&sdr[corpus[pos] as usize], tick); }
        let out = proj.predict(&net.charge);
        let pred = out.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0);
        if pred == corpus[pos + 1] as usize { correct += 1; } count += 1;
    }
    if count > 0 { correct as f64 / count as f64 } else { 0.0 }
}

fn train(net: &mut Net, proj: &mut Projection, corpus: &[u8], sdr: &[Vec<i8>], bigram: &[Vec<f64>], seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed + 500);
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    for _ in 0..TRAIN_STEPS {
        let sr = eval_rng.clone();
        let before = eval_cos(net, proj, corpus, 20, &mut eval_rng, sdr, bigram);
        eval_rng = sr;
        let roll = rng.gen_range(0..100u32);
        if roll < 30 {
            let (pi, pc, old) = proj.mutate(&mut rng);
            let after = eval_cos(net, proj, corpus, 20, &mut eval_rng, sdr, bigram);
            if after <= before { proj.undo(pi, pc, old); }
        } else if roll < 50 {
            if net.edge_count() < EDGE_CAP { let topo = net.save_topo();
                let s = rng.gen_range(0..H) as u16; let t = rng.gen_range(0..H) as u16;
                if net.add_edge(s, t) { let after = eval_cos(net, proj, corpus, 20, &mut eval_rng, sdr, bigram); if after <= before { net.restore_topo(topo); } }
                else { let _ = eval_cos(net, proj, corpus, 20, &mut eval_rng, sdr, bigram); }
            } else { let _ = eval_cos(net, proj, corpus, 20, &mut eval_rng, sdr, bigram); }
        } else if roll < 60 {
            let topo = net.save_topo();
            if net.remove_edge(&mut rng) { let after = eval_cos(net, proj, corpus, 20, &mut eval_rng, sdr, bigram); if after <= before { net.restore_topo(topo); } }
            else { let _ = eval_cos(net, proj, corpus, 20, &mut eval_rng, sdr, bigram); }
        } else if roll < 70 {
            let topo = net.save_topo();
            if net.rewire(&mut rng) { let after = eval_cos(net, proj, corpus, 20, &mut eval_rng, sdr, bigram); if after <= before { net.restore_topo(topo); } }
            else { let _ = eval_cos(net, proj, corpus, 20, &mut eval_rng, sdr, bigram); }
        } else if roll < 85 {
            let n = rng.gen_range(0..H); let old = net.threshold[n]; net.threshold[n] = rng.gen_range(0..=15);
            let after = eval_cos(net, proj, corpus, 20, &mut eval_rng, sdr, bigram);
            if after <= before { net.threshold[n] = old; }
        } else if roll < 93 {
            let n = rng.gen_range(0..H); let old = net.channel[n]; net.channel[n] = rng.gen_range(1..=8);
            let after = eval_cos(net, proj, corpus, 20, &mut eval_rng, sdr, bigram);
            if after <= before { net.channel[n] = old; }
        } else {
            let n = rng.gen_range(0..H); net.polarity[n] *= -1;
            let after = eval_cos(net, proj, corpus, 20, &mut eval_rng, sdr, bigram);
            if after <= before { net.polarity[n] *= -1; }
        }
    }
}

fn main() {
    let raw = std::fs::read_to_string("instnct-core/tests/fixtures/alice_corpus.txt").expect("corpus not found");
    let corpus: Vec<u8> = raw.bytes().map(|b| match b { b'a'..=b'z' => b-b'a', _ => 26 }).collect();
    let bigram = build_bigram_table(&corpus, VOCAB);
    let sdr: Vec<Vec<i8>> = (0..VOCAB).map(|sym| {
        let mut rng = StdRng::seed_from_u64(sym as u64 + 9999);
        let mut p = vec![0i8; H]; let active = INPUT_END / 5; let mut placed = 0;
        while placed < active { let i = rng.gen_range(0..INPUT_END); if p[i]==0 { p[i]=1; placed+=1; } } p
    }).collect();

    println!("=== BRIDGE TEST: zero overlap + pre-built I/O paths ===");
    println!("H={}, input=[0..{}], output=[{}..{}], {}steps\n", H, INPUT_END, OUTPUT_START, H, TRAIN_STEPS);

    let configs: Vec<(&str, usize, bool)> = vec![
        ("A: no init, evolve edges",    0,   false),
        ("B: 50 direct bridges",        50,  true),
        ("C: 100 direct bridges",       100, true),
        ("D: 50 2-hop chains",          50,  false),
        ("E: proj-only (no edges)",     0,   false),
    ];

    for (label, bridge_count, direct) in &configs {
        print!("{:>30}:", label);

        let mut best_acc = 0.0f64;
        let mut results = Vec::new();

        for &seed in &[42u64, 1042] {
            let mut rng_init = StdRng::seed_from_u64(seed);
            let mut net = Net::new(H, &mut rng_init);
            let mut proj = Projection::new(H, VOCAB, OUTPUT_START, &mut rng_init);

            if *bridge_count > 0 {
                if *direct {
                    net.add_bridges(*bridge_count, &mut rng_init);
                } else {
                    net.add_chains(*bridge_count, &mut rng_init);
                }
            }

            let init_edges = net.edge_count();

            if label.contains("proj-only") {
                // No edge mutations — only proj + params
                let mut rng = StdRng::seed_from_u64(seed + 500);
                let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
                for _ in 0..TRAIN_STEPS {
                    let sr = eval_rng.clone();
                    let before = eval_cos(&mut net, &proj, &corpus, 20, &mut eval_rng, &sdr, &bigram);
                    eval_rng = sr;
                    let roll = rng.gen_range(0..100u32);
                    if roll < 40 {
                        let (pi, pc, old) = proj.mutate(&mut rng);
                        let after = eval_cos(&mut net, &proj, &corpus, 20, &mut eval_rng, &sdr, &bigram);
                        if after <= before { proj.undo(pi, pc, old); }
                    } else if roll < 70 {
                        let n = rng.gen_range(0..H); let old = net.threshold[n]; net.threshold[n] = rng.gen_range(0..=15);
                        let after = eval_cos(&mut net, &proj, &corpus, 20, &mut eval_rng, &sdr, &bigram);
                        if after <= before { net.threshold[n] = old; }
                    } else if roll < 85 {
                        let n = rng.gen_range(0..H); let old = net.channel[n]; net.channel[n] = rng.gen_range(1..=8);
                        let after = eval_cos(&mut net, &proj, &corpus, 20, &mut eval_rng, &sdr, &bigram);
                        if after <= before { net.channel[n] = old; }
                    } else {
                        let n = rng.gen_range(0..H); net.polarity[n] *= -1;
                        let after = eval_cos(&mut net, &proj, &corpus, 20, &mut eval_rng, &sdr, &bigram);
                        if after <= before { net.polarity[n] *= -1; }
                    }
                }
            } else {
                train(&mut net, &mut proj, &corpus, &sdr, &bigram, seed);
            }

            // Check output charge
            net.reset();
            for tick in 0..TICKS { net.propagate(&sdr[0], tick); }
            let out_charge: i16 = net.charge[OUTPUT_START..H].iter().map(|c| c.abs()).sum();

            let acc = eval_acc(&mut net, &proj, &corpus, &sdr);
            best_acc = best_acc.max(acc);
            results.push(format!("s{}={:.1}%({}e,ch={})", seed, acc*100.0, net.edge_count(), out_charge));
        }

        println!(" best={:.1}%  {}", best_acc*100.0, results.join(" | "));
    }
}
