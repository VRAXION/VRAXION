//! ListNet + Int8 Projection — fair test for whether edges matter
//!
//! Run: cargo run --example listnet_proj --release

use instnct_core::build_bigram_table;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const VOCAB: usize = 27;
const TICKS: usize = 6;
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];
const H: usize = 512;
const EDGE_CAP: usize = 300;
const TRAIN_STEPS: usize = 50_000;

// --- Int8 Projection ---
#[derive(Clone)]
struct Projection {
    w: Vec<Vec<i8>>,  // H × VOCAB
    h: usize,
    vocab: usize,
    output_start: usize,
}

impl Projection {
    fn new(h: usize, vocab: usize, output_start: usize, rng: &mut impl Rng) -> Self {
        let w = (0..h).map(|_| (0..vocab).map(|_| rng.gen_range(-2..=2i8)).collect()).collect();
        Projection { w, h, vocab, output_start }
    }

    fn predict(&self, charge: &[i16]) -> Vec<f64> {
        let mut scores = vec![0.0f64; self.vocab];
        // Read CHARGE (continuous) from output zone — not binary activation
        for i in self.output_start..self.h {
            let ch = charge[i] as f64;
            if ch == 0.0 { continue; }
            for c in 0..self.vocab {
                scores[c] += ch * self.w[i][c] as f64;
            }
        }
        scores
    }

    fn mutate(&mut self, rng: &mut impl Rng) -> (usize, usize, i8) {
        let i = rng.gen_range(self.output_start..self.h);
        let c = rng.gen_range(0..self.vocab);
        let old = self.w[i][c];
        self.w[i][c] = rng.gen_range(-4..=4i8);
        (i, c, old)
    }

    fn undo_mutate(&mut self, i: usize, c: usize, old: i8) {
        self.w[i][c] = old;
    }
}

// --- ListNet ---
#[derive(Clone)]
struct ListNet {
    topology: Vec<Vec<u16>>,
    threshold: Vec<u8>, channel: Vec<u8>, polarity: Vec<i8>,
    charge: Vec<i16>, activation: Vec<i8>,
    h: usize, input_end: usize, output_start: usize,
}

impl ListNet {
    fn new(h: usize, input_end: usize, output_start: usize, rng: &mut impl Rng) -> Self {
        let mut threshold = vec![0u8; h]; let mut channel = vec![0u8; h]; let mut polarity = vec![1i8; h];
        for i in 0..h { threshold[i] = rng.gen_range(0..=7); channel[i] = rng.gen_range(1..=8); if rng.gen_ratio(1, 10) { polarity[i] = -1; } }
        ListNet { topology: Vec::new(), threshold, channel, polarity, charge: vec![0; h], activation: vec![0; h], h, input_end, output_start }
    }
    fn reset(&mut self) { self.charge.iter_mut().for_each(|c| *c = 0); self.activation.iter_mut().for_each(|a| *a = 0); }
    fn edge_count(&self) -> usize { self.topology.iter().map(|r| r.len().saturating_sub(1)).sum() }
    fn propagate(&mut self, input: &[i8], tick: usize) {
        let h = self.h;
        if tick < 2 { for i in 0..self.input_end.min(input.len()) { self.charge[i] = self.charge[i].saturating_add(input[i] as i16); } }
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
}

// --- Eval ---
fn eval_cos(net: &mut ListNet, proj: &Projection, corpus: &[u8], tokens: usize, rng: &mut StdRng, sdr: &[Vec<i8>], bigram: &[Vec<f64>]) -> f64 {
    let start = rng.gen_range(0..corpus.len().saturating_sub(tokens + 1).max(1));
    let mut total = 0.0f64; let mut count = 0usize; net.reset();
    for t in 0..tokens { let pos = start + t; if pos + 1 >= corpus.len() { break; }
        for tick in 0..TICKS { net.propagate(&sdr[corpus[pos] as usize], tick); }
        let out = proj.predict(&net.charge);
        let tgt = &bigram[corpus[pos + 1] as usize];
        let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
        for j in 0..VOCAB { dot += out[j]*tgt[j]; na += out[j]*out[j]; nb += tgt[j]*tgt[j]; }
        total += dot / (na.sqrt() * nb.sqrt()).max(1e-10); count += 1;
    }
    if count > 0 { total / count as f64 } else { 0.0 }
}

fn eval_acc(net: &mut ListNet, proj: &Projection, corpus: &[u8], len: usize, sdr: &[Vec<i8>]) -> (f64, usize) {
    let mut correct = 0usize; let mut count = 0usize; net.reset();
    for pos in 0..corpus.len().saturating_sub(1).min(len) {
        for tick in 0..TICKS { net.propagate(&sdr[corpus[pos] as usize], tick); }
        let out = proj.predict(&net.charge);
        let pred = out.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0);
        if pred == corpus[pos + 1] as usize { correct += 1; } count += 1;
    }
    (if count > 0 { correct as f64 / count as f64 } else { 0.0 }, correct)
}

fn main() {
    let raw = std::fs::read_to_string("instnct-core/tests/fixtures/alice_corpus.txt").expect("corpus not found");
    let corpus: Vec<u8> = raw.bytes().map(|b| match b { b'a'..=b'z' => b-b'a', _ => 26 }).collect();
    let bigram = build_bigram_table(&corpus, VOCAB);

    let phi_dim = (H as f64 / 1.618).round() as usize;
    let input_end = phi_dim;
    let output_start = H - phi_dim;

    let sdr: Vec<Vec<i8>> = (0..VOCAB).map(|sym| {
        let mut rng = StdRng::seed_from_u64(sym as u64 + 9999);
        let mut p = vec![0i8; H]; let active = input_end / 5; let mut placed = 0;
        while placed < active { let i = rng.gen_range(0..input_end); if p[i]==0 { p[i]=1; placed+=1; } } p
    }).collect();

    println!("=== ListNet + Int8 Projection — Edge Ablation ===");
    println!("H={}, phi overlap, edge_cap={}, {} steps\n", H, EDGE_CAP, TRAIN_STEPS);

    let seeds = [42u64, 1042, 2042];

    for &seed in &seeds {
        println!("--- seed {} ---", seed);

        // --- A: Train full (edges + params + projection) ---
        let mut rng = StdRng::seed_from_u64(seed);
        let mut net = ListNet::new(H, input_end, output_start, &mut rng);
        let mut proj = Projection::new(H, VOCAB, output_start, &mut rng);
        let mut eval_rng = StdRng::seed_from_u64(seed + 1000);

        for _ in 0..TRAIN_STEPS {
            let sr = eval_rng.clone();
            let before = eval_cos(&mut net, &proj, &corpus, 20, &mut eval_rng, &sdr, &bigram);
            eval_rng = sr;

            let roll = rng.gen_range(0..100u32);
            if roll < 30 {
                // 30% projection mutation
                let (pi, pc, old) = proj.mutate(&mut rng);
                let after = eval_cos(&mut net, &proj, &corpus, 20, &mut eval_rng, &sdr, &bigram);
                if after <= before { proj.undo_mutate(pi, pc, old); }
            } else if roll < 45 {
                // 15% edge add
                if net.edge_count() < EDGE_CAP {
                    let topo = net.save_topo();
                    let s = rng.gen_range(0..H) as u16; let t = rng.gen_range(0..H) as u16;
                    if net.add_edge(s, t) {
                        let after = eval_cos(&mut net, &proj, &corpus, 20, &mut eval_rng, &sdr, &bigram);
                        if after <= before { net.restore_topo(topo); }
                    } else { let _ = eval_cos(&mut net, &proj, &corpus, 20, &mut eval_rng, &sdr, &bigram); }
                } else { let _ = eval_cos(&mut net, &proj, &corpus, 20, &mut eval_rng, &sdr, &bigram); }
            } else if roll < 55 {
                // 10% edge remove
                let topo = net.save_topo();
                if net.remove_edge(&mut rng) {
                    let after = eval_cos(&mut net, &proj, &corpus, 20, &mut eval_rng, &sdr, &bigram);
                    if after <= before { net.restore_topo(topo); }
                } else { let _ = eval_cos(&mut net, &proj, &corpus, 20, &mut eval_rng, &sdr, &bigram); }
            } else if roll < 70 {
                // 15% rewire
                let topo = net.save_topo();
                if net.rewire(&mut rng) {
                    let after = eval_cos(&mut net, &proj, &corpus, 20, &mut eval_rng, &sdr, &bigram);
                    if after <= before { net.restore_topo(topo); }
                } else { let _ = eval_cos(&mut net, &proj, &corpus, 20, &mut eval_rng, &sdr, &bigram); }
            } else if roll < 82 {
                // 12% threshold
                let n = rng.gen_range(0..H); let old = net.threshold[n];
                net.threshold[n] = rng.gen_range(0..=15);
                let after = eval_cos(&mut net, &proj, &corpus, 20, &mut eval_rng, &sdr, &bigram);
                if after <= before { net.threshold[n] = old; }
            } else if roll < 92 {
                // 10% channel
                let n = rng.gen_range(0..H); let old = net.channel[n];
                net.channel[n] = rng.gen_range(1..=8);
                let after = eval_cos(&mut net, &proj, &corpus, 20, &mut eval_rng, &sdr, &bigram);
                if after <= before { net.channel[n] = old; }
            } else {
                // 10% polarity
                let n = rng.gen_range(0..H); net.polarity[n] *= -1;
                let after = eval_cos(&mut net, &proj, &corpus, 20, &mut eval_rng, &sdr, &bigram);
                if after <= before { net.polarity[n] *= -1; }
            }
        }

        let (acc_full, _) = eval_acc(&mut net, &proj, &corpus, 10000, &sdr);
        let edges_full = net.edge_count();

        // --- Ablation: remove all edges ---
        let topo_backup = net.topology.clone();
        net.topology.clear();
        let (acc_no_edges, _) = eval_acc(&mut net, &proj, &corpus, 10000, &sdr);
        net.topology = topo_backup;

        // --- Random edges ---
        let edge_n = net.edge_count();
        net.topology.clear();
        let mut r2 = StdRng::seed_from_u64(9999);
        while net.edge_count() < edge_n { net.add_edge(r2.gen_range(0..H) as u16, r2.gen_range(0..H) as u16); }
        let (acc_random, _) = eval_acc(&mut net, &proj, &corpus, 10000, &sdr);

        // --- B: Projection-only (no edges, no edge mutations) ---
        let mut net_b = ListNet::new(H, input_end, output_start, &mut StdRng::seed_from_u64(seed));
        let mut proj_b = Projection::new(H, VOCAB, output_start, &mut StdRng::seed_from_u64(seed));
        let mut eval_rng_b = StdRng::seed_from_u64(seed + 1000);
        let mut rng_b = StdRng::seed_from_u64(seed + 500);

        for _ in 0..TRAIN_STEPS {
            let sr = eval_rng_b.clone();
            let before = eval_cos(&mut net_b, &proj_b, &corpus, 20, &mut eval_rng_b, &sdr, &bigram);
            eval_rng_b = sr;

            let roll = rng_b.gen_range(0..100u32);
            if roll < 30 {
                // projection mutation
                let (pi, pc, old) = proj_b.mutate(&mut rng_b);
                let after = eval_cos(&mut net_b, &proj_b, &corpus, 20, &mut eval_rng_b, &sdr, &bigram);
                if after <= before { proj_b.undo_mutate(pi, pc, old); }
            } else if roll < 60 {
                let n = rng_b.gen_range(0..H); let old = net_b.threshold[n];
                net_b.threshold[n] = rng_b.gen_range(0..=15);
                let after = eval_cos(&mut net_b, &proj_b, &corpus, 20, &mut eval_rng_b, &sdr, &bigram);
                if after <= before { net_b.threshold[n] = old; }
            } else if roll < 80 {
                let n = rng_b.gen_range(0..H); let old = net_b.channel[n];
                net_b.channel[n] = rng_b.gen_range(1..=8);
                let after = eval_cos(&mut net_b, &proj_b, &corpus, 20, &mut eval_rng_b, &sdr, &bigram);
                if after <= before { net_b.channel[n] = old; }
            } else {
                let n = rng_b.gen_range(0..H); net_b.polarity[n] *= -1;
                let after = eval_cos(&mut net_b, &proj_b, &corpus, 20, &mut eval_rng_b, &sdr, &bigram);
                if after <= before { net_b.polarity[n] *= -1; }
            }
        }
        let (acc_proj_only, _) = eval_acc(&mut net_b, &proj_b, &corpus, 10000, &sdr);

        let diff = (acc_full - acc_no_edges) * 100.0;
        println!("  Full (edge+param+proj): {:.1}% ({} edges)", acc_full * 100.0, edges_full);
        println!("  Edges REMOVED:          {:.1}%", acc_no_edges * 100.0);
        println!("  Random edges:           {:.1}%", acc_random * 100.0);
        println!("  Proj+params only:       {:.1}% (0 edges)", acc_proj_only * 100.0);
        println!("  Edge contribution:      {:+.1}pp\n", diff);
    }
}
