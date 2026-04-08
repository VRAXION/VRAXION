//! Addition task: do edges matter when COMPUTATION is needed?
//!
//! Task: input two digits (0-4), predict their sum (0-8).
//! This REQUIRES computation — can't be solved by direct input→output mapping.
//!
//! Test both INSTNCT library and ListNet, with edge ablation.
//!
//! Run: cargo run --example addition_edge_test --release

use instnct_core::{
    build_network, evolution_step, EvolutionConfig, InitConfig, Int8Projection,
    Network, SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIGITS: usize = 5;        // 0-4
const SUMS: usize = 9;          // 0-8
const H: usize = 256;
const SDR_ACTIVE_PCT: usize = 20;
const TRAIN_STEPS: usize = 50_000;
const TICKS: usize = 6;
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];

// --- ListNet version ---

#[derive(Clone)]
struct ListNet {
    topology: Vec<Vec<u16>>, threshold: Vec<u8>, channel: Vec<u8>, polarity: Vec<i8>,
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

#[derive(Clone)]
struct Proj {
    w: Vec<Vec<i8>>, output_start: usize, h: usize, classes: usize,
}
impl Proj {
    fn new(h: usize, classes: usize, output_start: usize, rng: &mut impl Rng) -> Self {
        Proj { w: (0..h).map(|_| (0..classes).map(|_| rng.gen_range(-2..=2i8)).collect()).collect(), output_start, h, classes }
    }
    fn predict(&self, charge: &[i16]) -> usize {
        let mut scores = vec![0i64; self.classes];
        for i in self.output_start..self.h { let ch = charge[i] as i64; if ch == 0 { continue; }
            for c in 0..self.classes { scores[c] += ch * self.w[i][c] as i64; } }
        scores.iter().enumerate().max_by_key(|&(_, s)| *s).map(|(i, _)| i).unwrap_or(0)
    }
    fn mutate(&mut self, rng: &mut impl Rng) -> (usize, usize, i8) {
        let i = rng.gen_range(self.output_start..self.h); let c = rng.gen_range(0..self.classes);
        let old = self.w[i][c]; self.w[i][c] = rng.gen_range(-4..=4i8); (i, c, old)
    }
    fn undo(&mut self, i: usize, c: usize, old: i8) { self.w[i][c] = old; }
}

/// Generate all 25 addition examples: (a, b) → a+b
fn make_examples() -> Vec<(usize, usize, usize)> {
    let mut ex = Vec::new();
    for a in 0..DIGITS { for b in 0..DIGITS { ex.push((a, b, a + b)); } }
    ex
}

/// Make SDR: 10 symbols (digit 0-4 in slot A, digit 0-4 in slot B)
/// Slot A uses neurons [0..input_end/2], slot B uses [input_end/2..input_end]
fn make_sdr(h: usize, input_end: usize) -> (Vec<Vec<i8>>, Vec<Vec<i8>>) {
    let half = input_end / 2;
    let active_per_digit = half / 5; // 20% active
    let sdr_a: Vec<Vec<i8>> = (0..DIGITS).map(|d| {
        let mut rng = StdRng::seed_from_u64(d as u64 + 100);
        let mut p = vec![0i8; h]; let mut placed = 0;
        while placed < active_per_digit { let i = rng.gen_range(0..half); if p[i]==0 { p[i]=1; placed+=1; } } p
    }).collect();
    let sdr_b: Vec<Vec<i8>> = (0..DIGITS).map(|d| {
        let mut rng = StdRng::seed_from_u64(d as u64 + 200);
        let mut p = vec![0i8; h]; let mut placed = 0;
        while placed < active_per_digit { let i = rng.gen_range(half..input_end); if p[i]==0 { p[i]=1; placed+=1; } } p
    }).collect();
    (sdr_a, sdr_b)
}

fn eval_addition_listnet(net: &mut ListNet, proj: &Proj, examples: &[(usize, usize, usize)], sdr_a: &[Vec<i8>], sdr_b: &[Vec<i8>]) -> f64 {
    let mut correct = 0;
    for &(a, b, target) in examples {
        net.reset();
        // Inject both digits simultaneously
        let mut input = vec![0i8; net.h];
        for i in 0..net.h { input[i] = sdr_a[a][i].saturating_add(sdr_b[b][i]); }
        for tick in 0..TICKS { net.propagate(&input, tick); }
        let pred = proj.predict(&net.charge);
        if pred == target { correct += 1; }
    }
    correct as f64 / examples.len() as f64
}

fn eval_addition_instnct(net: &mut Network, proj: &Int8Projection, examples: &[(usize, usize, usize)],
    sdr_a: &SdrTable, sdr_b: &SdrTable, prop_cfg: &instnct_core::PropagationConfig, output_start: usize, neuron_count: usize) -> f64 {
    let mut correct = 0;
    for &(a, b, target) in examples {
        net.reset();
        // Inject both patterns: combine SDR a + SDR b
        let pa = sdr_a.pattern(a);
        let pb = sdr_b.pattern(b);
        let mut combined = vec![0i32; neuron_count];
        for i in 0..neuron_count { combined[i] = pa[i] + pb[i]; }
        for _ in 0..6 {
            let _ = net.propagate(&combined, prop_cfg);
        }
        let charge = net.charge_vec(output_start..neuron_count);
        if proj.predict(&charge) == target { correct += 1; }
    }
    correct as f64 / examples.len() as f64
}

fn main() {
    let examples = make_examples();
    let random_baseline = 1.0 / SUMS as f64; // 11.1%

    println!("=== ADDITION TASK: DO EDGES MATTER? ===");
    println!("Task: a+b where a,b in 0..5, output in 0..9 (25 examples, random={:.1}%)\n", random_baseline * 100.0);

    // ===== LISTNET VERSION (separated I/O, no overlap) =====
    println!("--- ListNet (separated I/O, charge projection) ---");
    let input_end = H / 2;     // [0..128]
    let output_start = H / 2;  // [128..256]
    let (sdr_a, sdr_b) = make_sdr(H, input_end);

    for &seed in &[42u64, 1042, 2042] {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut net = ListNet::new(H, input_end, output_start, &mut rng);
        let mut proj = Proj::new(H, SUMS, output_start, &mut rng);

        // Add 50 bridge edges input→output
        for _ in 0..50 {
            let s = rng.gen_range(0..input_end) as u16;
            let t = rng.gen_range(output_start..H) as u16;
            net.add_edge(s, t);
        }

        // Train
        for _ in 0..TRAIN_STEPS {
            let before = eval_addition_listnet(&mut net, &proj, &examples, &sdr_a, &sdr_b);

            let roll = rng.gen_range(0..100u32);
            if roll < 30 {
                let (pi, pc, old) = proj.mutate(&mut rng);
                let after = eval_addition_listnet(&mut net, &proj, &examples, &sdr_a, &sdr_b);
                if after <= before { proj.undo(pi, pc, old); }
            } else if roll < 50 {
                let topo = net.save_topo();
                if net.edge_count() < 300 { net.add_edge(rng.gen_range(0..H) as u16, rng.gen_range(0..H) as u16); }
                let after = eval_addition_listnet(&mut net, &proj, &examples, &sdr_a, &sdr_b);
                if after <= before { net.restore_topo(topo); }
            } else if roll < 65 {
                let topo = net.save_topo();
                net.rewire(&mut rng);
                let after = eval_addition_listnet(&mut net, &proj, &examples, &sdr_a, &sdr_b);
                if after <= before { net.restore_topo(topo); }
            } else if roll < 75 {
                let topo = net.save_topo();
                net.remove_edge(&mut rng);
                let after = eval_addition_listnet(&mut net, &proj, &examples, &sdr_a, &sdr_b);
                if after <= before { net.restore_topo(topo); }
            } else if roll < 88 {
                let n = rng.gen_range(0..H); let old = net.threshold[n]; net.threshold[n] = rng.gen_range(0..=15);
                let after = eval_addition_listnet(&mut net, &proj, &examples, &sdr_a, &sdr_b);
                if after <= before { net.threshold[n] = old; }
            } else if roll < 95 {
                let n = rng.gen_range(0..H); let old = net.channel[n]; net.channel[n] = rng.gen_range(1..=8);
                let after = eval_addition_listnet(&mut net, &proj, &examples, &sdr_a, &sdr_b);
                if after <= before { net.channel[n] = old; }
            } else {
                let n = rng.gen_range(0..H); net.polarity[n] *= -1;
                let after = eval_addition_listnet(&mut net, &proj, &examples, &sdr_a, &sdr_b);
                if after <= before { net.polarity[n] *= -1; }
            }
        }

        let acc_full = eval_addition_listnet(&mut net, &proj, &examples, &sdr_a, &sdr_b);
        let edges = net.edge_count();

        // Ablation
        let topo = net.topology.clone();
        net.topology.clear();
        let acc_no_edges = eval_addition_listnet(&mut net, &proj, &examples, &sdr_a, &sdr_b);
        net.topology = topo;

        // Proj-only (train from scratch, no edges)
        let mut net2 = ListNet::new(H, input_end, output_start, &mut StdRng::seed_from_u64(seed));
        let mut proj2 = Proj::new(H, SUMS, output_start, &mut StdRng::seed_from_u64(seed));
        for _ in 0..TRAIN_STEPS {
            let before = eval_addition_listnet(&mut net2, &proj2, &examples, &sdr_a, &sdr_b);
            let roll = rng.gen_range(0..100u32);
            if roll < 40 { let (pi,pc,old) = proj2.mutate(&mut rng); let after = eval_addition_listnet(&mut net2, &proj2, &examples, &sdr_a, &sdr_b); if after <= before { proj2.undo(pi,pc,old); } }
            else if roll < 70 { let n = rng.gen_range(0..H); let old = net2.threshold[n]; net2.threshold[n] = rng.gen_range(0..=15); let after = eval_addition_listnet(&mut net2, &proj2, &examples, &sdr_a, &sdr_b); if after <= before { net2.threshold[n] = old; } }
            else if roll < 85 { let n = rng.gen_range(0..H); let old = net2.channel[n]; net2.channel[n] = rng.gen_range(1..=8); let after = eval_addition_listnet(&mut net2, &proj2, &examples, &sdr_a, &sdr_b); if after <= before { net2.channel[n] = old; } }
            else { let n = rng.gen_range(0..H); net2.polarity[n] *= -1; let after = eval_addition_listnet(&mut net2, &proj2, &examples, &sdr_a, &sdr_b); if after <= before { net2.polarity[n] *= -1; } }
        }
        let acc_proj_only = eval_addition_listnet(&mut net2, &proj2, &examples, &sdr_a, &sdr_b);

        let diff = (acc_full - acc_no_edges) * 100.0;
        println!("  seed {}: full={:.0}% ({} edges) | no-edges={:.0}% | proj-only={:.0}% | edge diff={:+.0}pp",
            seed, acc_full*100.0, edges, acc_no_edges*100.0, acc_proj_only*100.0, diff);
    }

    // ===== INSTNCT LIBRARY VERSION =====
    println!("\n--- INSTNCT library (phi overlap, empty init) ---");
    for &seed in &[42u64, 1042, 2042] {
        let mut rng = StdRng::seed_from_u64(seed);
        let init = InitConfig::empty(H);
        let mut net = build_network(&init, &mut rng);

        // SDR for digits: slot A in [0..input_end/2], slot B in [input_end/2..input_end]
        let sdr_a_lib = SdrTable::new(DIGITS, H, init.input_end() / 2, SDR_ACTIVE_PCT, &mut StdRng::seed_from_u64(seed + 100)).unwrap();
        let sdr_b_lib = SdrTable::new(DIGITS, H, init.input_end(), SDR_ACTIVE_PCT, &mut StdRng::seed_from_u64(seed + 200)).unwrap();
        // Offset sdr_b patterns to second half of input zone

        let mut proj = Int8Projection::new(init.phi_dim, SUMS, &mut rng);
        let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
        let evo_config = EvolutionConfig { edge_cap: 300, accept_ties: false };
        let prop_cfg = init.propagation.clone();
        let output_start = init.output_start();
        let neuron_count = init.neuron_count;

        // Train using evolution_step with addition fitness
        for _ in 0..TRAIN_STEPS {
            evolution_step(
                &mut net, &mut proj, &mut rng, &mut eval_rng,
                |net, proj, _eval_rng| {
                    let mut correct = 0i32;
                    for &(a, b, target) in &examples {
                        net.reset();
                        let pa = sdr_a_lib.pattern(a);
                        let pb = sdr_b_lib.pattern(b);
                        let mut combined = vec![0i32; neuron_count];
                        for i in 0..neuron_count { combined[i] = pa[i] + pb[i]; }
                        for _ in 0..6 { let _ = net.propagate(&combined, &prop_cfg); }
                        let charge = net.charge_vec(output_start..neuron_count);
                        if proj.predict(&charge) == target { correct += 1; }
                    }
                    correct as f64 / examples.len() as f64
                },
                &evo_config,
            );
        }

        let acc_full = eval_addition_instnct(&mut net, &proj, &examples, &sdr_a_lib, &sdr_b_lib, &prop_cfg, output_start, neuron_count);
        let edges = net.edge_count();

        // Ablation: empty network + same projection
        let mut net_empty = Network::new(H);
        let acc_empty = eval_addition_instnct(&mut net_empty, &proj, &examples, &sdr_a_lib, &sdr_b_lib, &prop_cfg, output_start, neuron_count);

        let diff = (acc_full - acc_empty) * 100.0;
        println!("  seed {}: full={:.0}% ({} edges) | empty+proj={:.0}% | edge diff={:+.0}pp",
            seed, acc_full*100.0, edges, acc_empty*100.0, diff);
    }
}
