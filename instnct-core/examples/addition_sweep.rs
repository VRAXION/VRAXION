//! Addition sweep: H, edge_cap, steps, ListNet vs INSTNCT
//!
//! Run: cargo run --example addition_sweep --release

use instnct_core::{
    build_network, evolution_step, EvolutionConfig, InitConfig, Int8Projection,
    Network, SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

const DIGITS: usize = 5;
const SUMS: usize = 9;
const SDR_ACTIVE_PCT: usize = 20;
const TICKS: usize = 6;
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];
const WALL_SECS: u64 = 120;
const SEEDS: [u64; 3] = [42, 1042, 2042];

fn make_examples() -> Vec<(usize, usize, usize)> {
    let mut ex = Vec::new();
    for a in 0..DIGITS { for b in 0..DIGITS { ex.push((a, b, a + b)); } }
    ex
}

// ---- ListNet for addition ----

#[derive(Clone)]
struct ListNet {
    topology: Vec<Vec<u16>>, threshold: Vec<u8>, channel: Vec<u8>, polarity: Vec<i8>,
    charge: Vec<i16>, activation: Vec<i8>, h: usize, input_end: usize, output_start: usize,
}
impl ListNet {
    fn new(h: usize, rng: &mut impl Rng) -> Self {
        let input_end = h / 2; let output_start = h / 2;
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
    fn save(&self) -> (Vec<Vec<u16>>, Vec<u8>, Vec<u8>, Vec<i8>) { (self.topology.clone(), self.threshold.clone(), self.channel.clone(), self.polarity.clone()) }
    fn restore(&mut self, s: (Vec<Vec<u16>>, Vec<u8>, Vec<u8>, Vec<i8>)) { self.topology = s.0; self.threshold = s.1; self.channel = s.2; self.polarity = s.3; }
}

#[derive(Clone)]
struct Proj { w: Vec<Vec<i8>>, output_start: usize, h: usize, classes: usize }
impl Proj {
    fn new(h: usize, classes: usize, output_start: usize, rng: &mut impl Rng) -> Self {
        Proj { w: (0..h).map(|_| (0..classes).map(|_| rng.gen_range(-2..=2i8)).collect()).collect(), output_start, h, classes }
    }
    fn predict(&self, charge: &[i16]) -> usize {
        let mut s = vec![0i64; self.classes];
        for i in self.output_start..self.h { let ch = charge[i] as i64; if ch == 0 { continue; } for c in 0..self.classes { s[c] += ch * self.w[i][c] as i64; } }
        s.iter().enumerate().max_by_key(|&(_, v)| *v).map(|(i, _)| i).unwrap_or(0)
    }
    fn mutate(&mut self, rng: &mut impl Rng) -> (usize, usize, i8) {
        let i = rng.gen_range(self.output_start..self.h); let c = rng.gen_range(0..self.classes);
        let old = self.w[i][c]; self.w[i][c] = rng.gen_range(-4..=4i8); (i, c, old)
    }
    fn undo(&mut self, i: usize, c: usize, old: i8) { self.w[i][c] = old; }
}

fn make_sdr(h: usize, input_end: usize) -> (Vec<Vec<i8>>, Vec<Vec<i8>>) {
    let half = input_end / 2; let active = half / 5;
    let sdr_a: Vec<Vec<i8>> = (0..DIGITS).map(|d| { let mut rng = StdRng::seed_from_u64(d as u64 + 100);
        let mut p = vec![0i8; h]; let mut placed = 0; while placed < active { let i = rng.gen_range(0..half); if p[i]==0 { p[i]=1; placed+=1; } } p }).collect();
    let sdr_b: Vec<Vec<i8>> = (0..DIGITS).map(|d| { let mut rng = StdRng::seed_from_u64(d as u64 + 200);
        let mut p = vec![0i8; h]; let mut placed = 0; while placed < active { let i = rng.gen_range(half..input_end); if p[i]==0 { p[i]=1; placed+=1; } } p }).collect();
    (sdr_a, sdr_b)
}

fn eval_add_listnet(net: &mut ListNet, proj: &Proj, examples: &[(usize, usize, usize)], sdr_a: &[Vec<i8>], sdr_b: &[Vec<i8>]) -> f64 {
    let mut correct = 0;
    for &(a, b, target) in examples { net.reset();
        let mut input = vec![0i8; net.h]; for i in 0..net.h { input[i] = sdr_a[a][i].saturating_add(sdr_b[b][i]); }
        for tick in 0..TICKS { net.propagate(&input, tick); }
        if proj.predict(&net.charge) == target { correct += 1; } }
    correct as f64 / examples.len() as f64
}

fn main() {
    let examples = make_examples();
    println!("=== ADDITION SWEEP ===");
    println!("a+b, a,b in 0..5, 25 examples, random=11.1%\n");

    // Sweep 1: ListNet H sweep on addition
    println!("--- Sweep 1: ListNet H sweep ({}s/seed, {} seeds, edge_cap=300) ---", WALL_SECS, SEEDS.len());
    println!("{:>6} {:>8} {:>8} {:>8} {:>8} {:>30}", "H", "edges", "steps", "step/s", "best%", "all_seeds");
    println!("{:-<6} {:-<8} {:-<8} {:-<8} {:-<8} {:-<30}", "", "", "", "", "", "");

    for &h in &[128, 256, 512, 1024] {
        let (sdr_a, sdr_b) = make_sdr(h, h / 2);
        let mut all_acc = Vec::new(); let mut all_steps = Vec::new(); let mut all_edges = Vec::new();

        for &seed in &SEEDS {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = ListNet::new(h, &mut rng);
            let mut proj = Proj::new(h, SUMS, h / 2, &mut rng);
            // Init bridges
            for _ in 0..50 { let s = rng.gen_range(0..h/2) as u16; let t = rng.gen_range(h/2..h) as u16; net.add_edge(s, t); }

            let mut steps = 0usize;
            let deadline = Instant::now() + Duration::from_secs(WALL_SECS);
            while Instant::now() < deadline {
                let before = eval_add_listnet(&mut net, &proj, &examples, &sdr_a, &sdr_b);
                let roll = rng.gen_range(0..100u32);
                if roll < 25 { let (pi,pc,old) = proj.mutate(&mut rng); let after = eval_add_listnet(&mut net, &proj, &examples, &sdr_a, &sdr_b); if after <= before { proj.undo(pi,pc,old); } }
                else if roll < 45 { if net.edge_count() < 300 { let topo = net.save().0.clone(); let s = rng.gen_range(0..h) as u16; let t = rng.gen_range(0..h) as u16; net.add_edge(s, t); let after = eval_add_listnet(&mut net, &proj, &examples, &sdr_a, &sdr_b); if after <= before { net.topology = topo; } } }
                else if roll < 60 { let genome = net.save(); if net.rewire(&mut rng) { let after = eval_add_listnet(&mut net, &proj, &examples, &sdr_a, &sdr_b); if after <= before { net.restore(genome); } } }
                else if roll < 70 { let genome = net.save(); if net.remove_edge(&mut rng) { let after = eval_add_listnet(&mut net, &proj, &examples, &sdr_a, &sdr_b); if after <= before { net.restore(genome); } } }
                else if roll < 82 { let n = rng.gen_range(0..h); let old = net.threshold[n]; net.threshold[n] = rng.gen_range(0..=15); let after = eval_add_listnet(&mut net, &proj, &examples, &sdr_a, &sdr_b); if after <= before { net.threshold[n] = old; } }
                else if roll < 92 { let n = rng.gen_range(0..h); let old = net.channel[n]; net.channel[n] = rng.gen_range(1..=8); let after = eval_add_listnet(&mut net, &proj, &examples, &sdr_a, &sdr_b); if after <= before { net.channel[n] = old; } }
                else { let n = rng.gen_range(0..h); net.polarity[n] *= -1; let after = eval_add_listnet(&mut net, &proj, &examples, &sdr_a, &sdr_b); if after <= before { net.polarity[n] *= -1; } }
                steps += 1;
            }

            let acc = eval_add_listnet(&mut net, &proj, &examples, &sdr_a, &sdr_b);
            all_acc.push(acc); all_steps.push(steps); all_edges.push(net.edge_count());
        }
        let best = all_acc.iter().cloned().fold(0.0f64, f64::max);
        let ms = all_steps.iter().sum::<usize>() / all_steps.len();
        let me = all_edges.iter().sum::<usize>() / all_edges.len();
        let seeds_str: Vec<String> = all_acc.iter().map(|a| format!("{:.0}%", a*100.0)).collect();
        println!("{:>6} {:>8} {:>8} {:>8.0} {:>7.0}% {:>30}", h, me, ms, ms as f64 / WALL_SECS as f64, best*100.0, seeds_str.join(", "));
    }

    // Sweep 2: INSTNCT library on addition
    println!("\n--- Sweep 2: INSTNCT library ({}s/seed, {} seeds) ---", WALL_SECS, SEEDS.len());
    println!("{:>6} {:>8} {:>8} {:>8} {:>8}", "H", "edges", "steps", "step/s", "best%");
    println!("{:-<6} {:-<8} {:-<8} {:-<8} {:-<8}", "", "", "", "", "");

    for &h in &[128, 256, 512] {
        let mut all_acc = Vec::new(); let mut all_steps = Vec::new(); let mut all_edges = Vec::new();

        for &seed in &SEEDS {
            let mut rng = StdRng::seed_from_u64(seed);
            let init = InitConfig::empty(h);
            let mut net = build_network(&init, &mut rng);
            let sdr_a = SdrTable::new(DIGITS, h, init.input_end() / 2, SDR_ACTIVE_PCT, &mut StdRng::seed_from_u64(seed + 100)).unwrap();
            let sdr_b = SdrTable::new(DIGITS, h, init.input_end(), SDR_ACTIVE_PCT, &mut StdRng::seed_from_u64(seed + 200)).unwrap();
            let mut proj = Int8Projection::new(init.phi_dim, SUMS, &mut rng);
            let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
            let evo_config = EvolutionConfig { edge_cap: 300, accept_ties: false };
            let prop_cfg = init.propagation.clone();
            let output_start = init.output_start();
            let neuron_count = init.neuron_count;
            let examples_ref = &examples;

            let mut steps = 0usize;
            let deadline = Instant::now() + Duration::from_secs(WALL_SECS);
            while Instant::now() < deadline {
                evolution_step(
                    &mut net, &mut proj, &mut rng, &mut eval_rng,
                    |net, proj, _eval_rng| {
                        let mut correct = 0i32;
                        for &(a, b, target) in examples_ref { net.reset();
                            let pa = sdr_a.pattern(a); let pb = sdr_b.pattern(b);
                            let mut combined = vec![0i32; neuron_count]; for i in 0..neuron_count { combined[i] = pa[i] + pb[i]; }
                            for _ in 0..6 { let _ = net.propagate(&combined, &prop_cfg); }
                            if proj.predict(&net.charge_vec(output_start..neuron_count)) == target { correct += 1; } }
                        correct as f64 / examples_ref.len() as f64
                    },
                    &evo_config,
                );
                steps += 1;
            }

            let mut correct = 0;
            for &(a, b, target) in &examples { net.reset();
                let pa = sdr_a.pattern(a); let pb = sdr_b.pattern(b);
                let mut combined = vec![0i32; neuron_count]; for i in 0..neuron_count { combined[i] = pa[i] + pb[i]; }
                for _ in 0..6 { let _ = net.propagate(&combined, &prop_cfg); }
                if proj.predict(&net.charge_vec(output_start..neuron_count)) == target { correct += 1; } }
            let acc = correct as f64 / examples.len() as f64;

            all_acc.push(acc); all_steps.push(steps); all_edges.push(net.edge_count());
        }
        let best = all_acc.iter().cloned().fold(0.0f64, f64::max);
        let ms = all_steps.iter().sum::<usize>() / all_steps.len();
        let me = all_edges.iter().sum::<usize>() / all_edges.len();
        println!("{:>6} {:>8} {:>8} {:>8.0} {:>7.0}%", h, me, ms, ms as f64 / WALL_SECS as f64, best*100.0);
    }
}
