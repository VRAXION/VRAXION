//! ListNet2 — everything in one Vec<Vec<u16>>. Params + topology in each row.
//!
//! Row format: [source, threshold, channel, polarity, target1, target2, ...]
//! Sorted by source ascending, targets sorted ascending within row.
//!
//! Run: cargo run --example listnet2 --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

const VOCAB: usize = 27;
const TICKS: usize = 6;
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];

// Row layout constants
const P_SRC: usize = 0;
const P_THR: usize = 1;
const P_CHN: usize = 2;
const P_POL: usize = 3;
const P_TARGETS: usize = 4;

#[derive(Clone)]
struct ListNet2 {
    /// THE network: each row = [source, threshold, channel, polarity, targets...]
    rows: Vec<Vec<u16>>,
    /// Runtime state (not part of genome)
    charge: Vec<i16>,
    activation: Vec<i8>,
    h: usize,
    input_dim: usize,
    output_start: usize,
}

impl ListNet2 {
    fn new(h: usize, rng: &mut impl Rng) -> Self {
        // Create one row per neuron with random params, no targets yet
        let phi_dim = (h as f64 / 1.618).round() as usize;
        let mut rows = Vec::with_capacity(h);
        for i in 0..h {
            let thr = rng.gen_range(0..=7u16);
            let chn = rng.gen_range(1..=8u16);
            let pol = if rng.gen_ratio(1, 10) { 65535u16 } else { 1u16 }; // -1 as u16 = 65535
            rows.push(vec![i as u16, thr, chn, pol]);
        }
        ListNet2 {
            rows, charge: vec![0; h], activation: vec![0; h],
            h, input_dim: phi_dim, output_start: h - phi_dim,
        }
    }

    fn reset(&mut self) {
        self.charge.iter_mut().for_each(|c| *c = 0);
        self.activation.iter_mut().for_each(|a| *a = 0);
    }

    fn edge_count(&self) -> usize {
        self.rows.iter().map(|r| r.len().saturating_sub(P_TARGETS)).sum()
    }

    #[inline]
    fn polarity_decode(val: u16) -> i8 {
        if val == 65535 { -1 } else { 1 }
    }

    fn propagate(&mut self, input: &[i8], tick: usize) {
        let h = self.h;

        // Input injection
        if tick < 2 {
            for i in 0..self.input_dim.min(input.len()) {
                self.charge[i] = self.charge[i].saturating_add(input[i] as i16);
            }
        }

        // Scatter from rows
        let mut incoming = vec![0i16; h];
        for row in &self.rows {
            let src = row[P_SRC] as usize;
            if src >= h { continue; }
            let act = self.activation[src];
            if act != 0 && row.len() > P_TARGETS {
                for &tgt in &row[P_TARGETS..] {
                    let t = tgt as usize;
                    if t < h { incoming[t] = incoming[t].saturating_add(act as i16); }
                }
            }
        }

        // Charge + spike (read params from rows)
        for row in &self.rows {
            let i = row[P_SRC] as usize;
            if i >= h { continue; }

            self.charge[i] = self.charge[i].saturating_add(incoming[i]);

            if tick % 6 == 5 && self.charge[i] > 0 { self.charge[i] -= 1; }

            let threshold = row[P_THR] as u8;
            let channel = row[P_CHN] as u8;
            let polarity = Self::polarity_decode(row[P_POL]);

            let pi = (tick as u8 + 9 - channel) & 7;
            let pm = PHASE_BASE[pi as usize];
            if self.charge[i] * 10 >= (threshold as i16 + 1) * pm {
                self.activation[i] = polarity;
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
        for i in 0..zl { out[i * nc / zl] += self.activation[self.output_start + i] as f64; }
        out
    }

    // --- Mutations ---

    fn add_edge(&mut self, src: u16, tgt: u16) -> bool {
        if src == tgt || src as usize >= self.h || tgt as usize >= self.h { return false; }
        let ri = src as usize; // rows are 1:1 with neurons, sorted by index
        if ri >= self.rows.len() { return false; }
        match self.rows[ri][P_TARGETS..].binary_search(&tgt) {
            Ok(_) => false,
            Err(pos) => { self.rows[ri].insert(P_TARGETS + pos, tgt); true }
        }
    }

    fn remove_edge(&mut self, rng: &mut impl Rng) -> bool {
        let total = self.edge_count();
        if total == 0 { return false; }
        let pick = rng.gen_range(0..total);
        let mut c = 0;
        for ri in 0..self.rows.len() {
            let e = self.rows[ri].len().saturating_sub(P_TARGETS);
            if e > 0 && c + e > pick {
                self.rows[ri].remove(P_TARGETS + (pick - c));
                return true;
            }
            c += e;
        }
        false
    }

    fn rewire(&mut self, rng: &mut impl Rng) -> bool {
        let total = self.edge_count();
        if total == 0 { return false; }
        let pick = rng.gen_range(0..total);
        let mut c = 0;
        for ri in 0..self.rows.len() {
            let e = self.rows[ri].len().saturating_sub(P_TARGETS);
            if e > 0 && c + e > pick {
                let src = self.rows[ri][P_SRC];
                let nt = rng.gen_range(0..self.h) as u16;
                if nt == src { return false; }
                self.rows[ri].remove(P_TARGETS + (pick - c));
                return self.add_edge(src, nt);
            }
            c += e;
        }
        false
    }

    fn mutate(&mut self, rng: &mut impl Rng, cap: usize) -> bool {
        match rng.gen_range(0..100u32) {
            0..30 => { if self.edge_count() >= cap { return false; } self.add_edge(rng.gen_range(0..self.h) as u16, rng.gen_range(0..self.h) as u16) }
            30..45 => self.remove_edge(rng),
            45..70 => self.rewire(rng),
            70..85 => { let n = rng.gen_range(0..self.h); self.rows[n][P_THR] = rng.gen_range(0..=15); true }
            85..95 => { let n = rng.gen_range(0..self.h); self.rows[n][P_CHN] = rng.gen_range(1..=8); true }
            _ => { let n = rng.gen_range(0..self.h); self.rows[n][P_POL] = if self.rows[n][P_POL] == 1 { 65535 } else { 1 }; true }
        }
    }

    fn save(&self) -> Vec<Vec<u16>> { self.rows.clone() }
    fn restore(&mut self, s: Vec<Vec<u16>>) { self.rows = s; }
}

// --- Quick eval for speed test ---

fn eval_cos(net: &mut ListNet2, corpus: &[u8], tokens: usize, rng: &mut StdRng, sdr: &[Vec<i8>], bigram: &[Vec<f64>]) -> f64 {
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
    use instnct_core::build_bigram_table;
    use std::time::Duration;

    let raw = std::fs::read_to_string("instnct-core/tests/fixtures/alice_corpus.txt").expect("corpus not found");
    let corpus: Vec<u8> = raw.bytes().map(|b| match b { b'a'..=b'z' => b-b'a', _ => 26 }).collect();
    let bigram = build_bigram_table(&corpus, VOCAB);

    let h_configs = [256, 512, 1024, 2048, 4096];
    let seeds = [42u64, 1042, 2042];

    let h_max = 4096;
    let sdr: Vec<Vec<i8>> = (0..VOCAB).map(|sym| {
        let mut rng = StdRng::seed_from_u64(sym as u64 + 9999);
        let input_dim = (h_max as f64 / 1.618).round() as usize;
        let mut p = vec![0i8; h_max]; let active = input_dim / 5; let mut placed = 0;
        while placed < active { let i = rng.gen_range(0..input_dim); if p[i]==0 { p[i]=1; placed+=1; } } p
    }).collect();

    println!("ListNet2 — All-in-one rows: [src, thr, chn, pol, targets...]");
    println!("5s/seed speed test | 3 seeds");
    println!("{:>6} {:>8} {:>8} {:>8} {:>10}", "H", "edges", "steps", "step/s", "row_bytes");
    println!("{:-<6} {:-<8} {:-<8} {:-<8} {:-<10}", "", "", "", "", "");

    for &h in &h_configs {
        let mut all_steps = Vec::new();
        let mut all_edges = Vec::new();

        for &seed in &seeds {
            let sdr_t: Vec<Vec<i8>> = sdr.iter().map(|p| p[..h].to_vec()).collect();
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = ListNet2::new(h, &mut rng);
            let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
            let mut steps = 0usize;
            let deadline = Instant::now() + Duration::from_secs(5);

            while Instant::now() < deadline {
                let sr = eval_rng.clone();
                let before = eval_cos(&mut net, &corpus, 20, &mut eval_rng, &sdr_t, &bigram);
                eval_rng = sr;
                let genome = net.save();
                let mutated = net.mutate(&mut rng, 300);
                if !mutated { let _ = eval_cos(&mut net, &corpus, 20, &mut eval_rng, &sdr_t, &bigram); steps += 1; continue; }
                let after = eval_cos(&mut net, &corpus, 20, &mut eval_rng, &sdr_t, &bigram);
                if after < before { net.restore(genome); }
                steps += 1;
            }
            all_steps.push(steps);
            all_edges.push(net.edge_count());
        }

        let ms = all_steps.iter().sum::<usize>() / all_steps.len();
        let me = all_edges.iter().sum::<usize>() / all_edges.len();
        // Row memory: h rows × (4 header + avg targets) × 2 bytes + Vec overhead
        let avg_targets = me as f64 / h as f64;
        let row_bytes = h as f64 * (4.0 + avg_targets) * 2.0;

        println!("{:>6} {:>8} {:>8} {:>8.0} {:>9.1}K",
            h, me, ms, ms as f64 / 5.0, row_bytes / 1024.0);
    }

    println!("\nComparison (ListNet1 overnight):");
    println!("  H=256:  3813 step/s");
    println!("  H=512:  2085 step/s");
    println!("  H=1024: 1091 step/s");
    println!("  H=2048:  574 step/s");
    println!("  H=4096:  293 step/s");
}
