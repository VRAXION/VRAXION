//! Final layout test: ListNet + packed params vs original INSTNCT — full evolution step
//!
//! Run: cargo run --example final_layout_test --release

use instnct_core::{
    build_bigram_table, build_network, eval_smooth, evolution_step,
    EvolutionConfig, InitConfig, Int8Projection, SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

const VOCAB: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const SECS: u64 = 10;
const TICKS: usize = 6;
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];
const EDGE_CAP: usize = 300;

// --- ListNet + packed params ---

#[derive(Clone, Copy)]
#[repr(C)]
struct NeuronParams {
    threshold: u8,
    channel: u8,
    polarity: i8,
    _pad: u8,
}

#[derive(Clone)]
struct OptNet {
    topology: Vec<Vec<u16>>,
    params: Vec<NeuronParams>,
    charge: Vec<i16>,
    activation: Vec<i8>,
    h: usize,
    input_dim: usize,
    output_start: usize,
}

impl OptNet {
    fn new(h: usize, rng: &mut impl Rng) -> Self {
        let phi_dim = (h as f64 / 1.618).round() as usize;
        let params: Vec<NeuronParams> = (0..h).map(|_| NeuronParams {
            threshold: rng.gen_range(0..=7), channel: rng.gen_range(1..=8),
            polarity: if rng.gen_ratio(1, 10) { -1 } else { 1 }, _pad: 0,
        }).collect();
        OptNet { topology: Vec::new(), params, charge: vec![0; h], activation: vec![0; h],
            h, input_dim: phi_dim, output_start: h - phi_dim }
    }
    fn reset(&mut self) { self.charge.iter_mut().for_each(|c| *c = 0); self.activation.iter_mut().for_each(|a| *a = 0); }
    fn edge_count(&self) -> usize { self.topology.iter().map(|r| r.len().saturating_sub(1)).sum() }
    fn propagate(&mut self, input: &[i8], tick: usize) {
        let h = self.h;
        if tick < 2 { for i in 0..self.input_dim.min(input.len()) { self.charge[i] = self.charge[i].saturating_add(input[i] as i16); } }
        let mut incoming = vec![0i16; h];
        for row in &self.topology {
            if row.len() < 2 { continue; } let src = row[0] as usize; if src >= h { continue; }
            let act = self.activation[src]; if act != 0 { for &tgt in &row[1..] { let t = tgt as usize; if t < h { incoming[t] = incoming[t].saturating_add(act as i16); } } }
        }
        for i in 0..h {
            self.charge[i] = self.charge[i].saturating_add(incoming[i]);
            if tick % 6 == 5 && self.charge[i] > 0 { self.charge[i] -= 1; }
            let p = &self.params[i];
            let pi = (tick as u8 + 9 - p.channel) & 7;
            let pm = PHASE_BASE[pi as usize];
            if self.charge[i] * 10 >= (p.threshold as i16 + 1) * pm {
                self.activation[i] = p.polarity; self.charge[i] = 0;
            } else { self.activation[i] = 0; }
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
    fn mutate(&mut self, rng: &mut impl Rng) -> bool {
        match rng.gen_range(0..100u32) {
            0..30 => { if self.edge_count() >= EDGE_CAP { return false; } self.add_edge(rng.gen_range(0..self.h) as u16, rng.gen_range(0..self.h) as u16) }
            30..45 => self.remove_edge(rng), 45..70 => self.rewire(rng),
            70..85 => { let n = rng.gen_range(0..self.h); self.params[n].threshold = rng.gen_range(0..=15); true }
            85..95 => { let n = rng.gen_range(0..self.h); self.params[n].channel = rng.gen_range(1..=8); true }
            _ => { let n = rng.gen_range(0..self.h); self.params[n].polarity *= -1; true }
        }
    }
    fn save(&self) -> (Vec<Vec<u16>>, Vec<NeuronParams>) { (self.topology.clone(), self.params.clone()) }
    fn restore(&mut self, s: (Vec<Vec<u16>>, Vec<NeuronParams>)) { self.topology = s.0; self.params = s.1; }
}

fn opt_eval_cos(net: &mut OptNet, corpus: &[u8], tokens: usize, rng: &mut StdRng, sdr: &[Vec<i8>], bigram: &[Vec<f64>]) -> f64 {
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

fn main() {
    let raw = std::fs::read_to_string("instnct-core/tests/fixtures/alice_corpus.txt").expect("corpus not found");
    let corpus: Vec<u8> = raw.bytes().map(|b| match b { b'a'..=b'z' => b-b'a', _ => 26 }).collect();
    let bigram = build_bigram_table(&corpus, VOCAB);

    let h_configs = [256, 512, 1024, 2048];

    let h_max = 2048;
    let sdr_raw: Vec<Vec<i8>> = (0..VOCAB).map(|sym| {
        let mut rng = StdRng::seed_from_u64(sym as u64 + 9999);
        let input_dim = (h_max as f64 / 1.618).round() as usize;
        let mut p = vec![0i8; h_max]; let active = input_dim / 5; let mut placed = 0;
        while placed < active { let i = rng.gen_range(0..input_dim); if p[i]==0 { p[i]=1; placed+=1; } } p
    }).collect();

    println!("Final Layout: OptNet (ListNet + packed params) vs INSTNCT library");
    println!("{}s/seed, 1 seed, edge_cap={}", SECS, EDGE_CAP);
    println!("{:>6} {:>10} {:>10} {:>10} {:>10}", "H", "OptNet", "INSTNCT", "speedup", "µs/tok_opt");
    println!("{:-<6} {:-<10} {:-<10} {:-<10} {:-<10}", "", "", "", "", "");

    for &h in &h_configs {
        let sdr_t: Vec<Vec<i8>> = sdr_raw.iter().map(|p| p[..h].to_vec()).collect();
        let seed = 42u64;

        // --- OptNet ---
        let opt_steps = {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = OptNet::new(h, &mut rng);
            let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
            let mut steps = 0usize;
            let deadline = Instant::now() + Duration::from_secs(SECS);
            while Instant::now() < deadline {
                let sr = eval_rng.clone();
                let before = opt_eval_cos(&mut net, &corpus, 20, &mut eval_rng, &sdr_t, &bigram);
                eval_rng = sr;
                let genome = net.save();
                let mutated = net.mutate(&mut rng);
                if !mutated { let _ = opt_eval_cos(&mut net, &corpus, 20, &mut eval_rng, &sdr_t, &bigram); steps += 1; continue; }
                let after = opt_eval_cos(&mut net, &corpus, 20, &mut eval_rng, &sdr_t, &bigram);
                if after < before { net.restore(genome); }
                steps += 1;
            }
            steps
        };

        // --- INSTNCT library (1+1 ES, no jackpot, for fair comparison) ---
        let instnct_steps = {
            let mut rng = StdRng::seed_from_u64(seed);
            let init = InitConfig::empty(h);
            let mut net = build_network(&init, &mut rng);
            let sdr = SdrTable::new(VOCAB, h, init.input_end(), SDR_ACTIVE_PCT, &mut StdRng::seed_from_u64(seed + 100)).unwrap();
            let mut proj = Int8Projection::new(init.phi_dim, VOCAB, &mut rng);
            let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
            let evo_config = EvolutionConfig { edge_cap: EDGE_CAP, accept_ties: true };
            let prop_cfg = init.propagation.clone();
            let output_start = init.output_start();
            let neuron_count = init.neuron_count;
            let mut steps = 0usize;
            let deadline = Instant::now() + Duration::from_secs(SECS);
            while Instant::now() < deadline {
                evolution_step(&mut net, &mut proj, &mut rng, &mut eval_rng,
                    |net, proj, eval_rng| { eval_smooth(net, proj, &corpus, 20, eval_rng, &sdr, &prop_cfg, output_start, neuron_count, &bigram) },
                    &evo_config);
                steps += 1;
            }
            steps
        };

        let opt_sps = opt_steps as f64 / SECS as f64;
        let ins_sps = instnct_steps as f64 / SECS as f64;
        let speedup = opt_sps / ins_sps;
        let us_per_tok = 1_000_000.0 / (opt_sps * 2.0 * 20.0 * TICKS as f64);

        println!("{:>6} {:>10.0} {:>10.0} {:>9.1}x {:>10.1}",
            h, opt_sps, ins_sps, speedup, us_per_tok);
    }
}
