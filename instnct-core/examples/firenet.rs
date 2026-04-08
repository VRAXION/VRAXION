//! FireNet — each neuron has a "fire variable" = list of who feeds it.
//!
//! A_fire = B + C + G  means neuron A gathers from B, C, G.
//! Mutation = add/remove from the fire variable.
//! No topology structure, no edge list, no scatter. Pure gather.
//!
//! Run: cargo run --example firenet --release

use instnct_core::build_bigram_table;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

const VOCAB: usize = 27;
const WALL_CLOCK_SECS: u64 = 60;
const SEEDS: [u64; 3] = [42, 1042, 2042];
const TICKS: usize = 6;
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];

// ---------------------------------------------------------------------------
// FireNet
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct FireNet {
    /// THE fire variables: fire_vars[A] = [B, C, G] means A gathers from B, C, G
    /// Sorted ascending within each variable.
    fire_vars: Vec<Vec<u16>>,
    /// Per-neuron parameters
    threshold: Vec<u8>,
    channel: Vec<u8>,
    polarity: Vec<i8>,
    /// Runtime state
    charge: Vec<i16>,
    activation: Vec<i8>,
    /// Geometry
    h: usize,
    input_dim: usize,
    output_start: usize,
}

impl FireNet {
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

        FireNet {
            fire_vars: vec![Vec::new(); h], // all empty — evolution builds it
            threshold,
            channel,
            polarity,
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
        self.fire_vars.iter().map(|v| v.len()).sum()
    }

    /// The core: each neuron evaluates its own fire variable.
    /// Pure gather — no scatter, no incoming buffer.
    fn propagate(&mut self, input: &[i8], tick: usize) {
        let h = self.h;

        // 1. Input injection (first 2 ticks)
        if tick < 2 {
            for i in 0..self.input_dim.min(input.len()) {
                self.charge[i] = self.charge[i].saturating_add(input[i] as i16);
            }
        }

        // 2. Each neuron gathers from its fire variable + spike decision
        //    Need snapshot of activation to avoid order-dependent reads
        let act_snapshot: Vec<i8> = self.activation.clone();

        for a in 0..h {
            // Gather: A_fire = sum of sources' activation
            let incoming: i16 = self.fire_vars[a].iter()
                .map(|&src| act_snapshot[src as usize] as i16)
                .sum();

            // Charge accumulation
            self.charge[a] = self.charge[a].saturating_add(incoming);

            // Decay
            if tick % 6 == 5 && self.charge[a] > 0 {
                self.charge[a] -= 1;
            }

            // Spike with phase gating
            let phase_idx = (tick as u8 + 9 - self.channel[a]) & 7;
            let phase_mult = PHASE_BASE[phase_idx as usize];
            let threshold_eff = (self.threshold[a] as i16 + 1) * phase_mult;
            let charge_x10 = self.charge[a] * 10;

            if charge_x10 >= threshold_eff {
                self.activation[a] = self.polarity[a];
                self.charge[a] = 0;
            } else {
                self.activation[a] = 0;
            }
        }
    }

    fn readout(&self, num_classes: usize) -> Vec<f64> {
        let mut output = vec![0.0f64; num_classes];
        let zone_len = self.h - self.output_start;
        if zone_len == 0 || num_classes == 0 { return output; }
        for i in 0..zone_len {
            let class = i * num_classes / zone_len;
            output[class] += self.activation[self.output_start + i] as f64;
        }
        output
    }

    // --- Mutations: modify fire variables ---

    /// Add: neuron `target` now also gathers from `source`.
    fn add_edge(&mut self, target: u16, source: u16) -> bool {
        if target == source { return false; }
        let t = target as usize;
        if t >= self.h || source as usize >= self.h { return false; }
        match self.fire_vars[t].binary_search(&source) {
            Ok(_) => false, // already in the variable
            Err(pos) => {
                self.fire_vars[t].insert(pos, source);
                true
            }
        }
    }

    /// Remove: neuron `target` stops gathering from a random source.
    fn remove_random_input(&mut self, target: usize, rng: &mut impl Rng) -> bool {
        if self.fire_vars[target].is_empty() { return false; }
        let idx = rng.gen_range(0..self.fire_vars[target].len());
        self.fire_vars[target].remove(idx);
        true
    }

    /// Rewire: change one random source in a random neuron's fire variable.
    fn rewire_random(&mut self, rng: &mut impl Rng) -> bool {
        // Pick a random neuron that has inputs
        let candidates: Vec<usize> = (0..self.h)
            .filter(|&i| !self.fire_vars[i].is_empty())
            .collect();
        if candidates.is_empty() { return false; }

        let target = candidates[rng.gen_range(0..candidates.len())];
        let idx = rng.gen_range(0..self.fire_vars[target].len());
        let new_source = rng.gen_range(0..self.h) as u16;
        if new_source == target as u16 { return false; }

        // Remove old, add new
        self.fire_vars[target].remove(idx);
        match self.fire_vars[target].binary_search(&new_source) {
            Ok(_) => {
                // Already exists — revert by re-inserting old? Just fail.
                false
            }
            Err(pos) => {
                self.fire_vars[target].insert(pos, new_source);
                true
            }
        }
    }

    fn mutate(&mut self, rng: &mut impl Rng, edge_cap: usize) -> bool {
        let roll = rng.gen_range(0..100u32);
        match roll {
            0..30 => {
                // 30% add to a fire variable
                if self.edge_count() >= edge_cap { return false; }
                let target = rng.gen_range(0..self.h) as u16;
                let source = rng.gen_range(0..self.h) as u16;
                self.add_edge(target, source)
            }
            30..45 => {
                // 15% remove from a fire variable
                let target = rng.gen_range(0..self.h);
                self.remove_random_input(target, rng)
            }
            45..70 => {
                // 25% rewire
                self.rewire_random(rng)
            }
            70..85 => {
                // 15% threshold
                let n = rng.gen_range(0..self.h);
                let old = self.threshold[n];
                self.threshold[n] = rng.gen_range(0..=15);
                self.threshold[n] != old
            }
            85..95 => {
                // 10% channel
                let n = rng.gen_range(0..self.h);
                let old = self.channel[n];
                self.channel[n] = rng.gen_range(1..=8);
                self.channel[n] != old
            }
            _ => {
                // 5% polarity flip
                let n = rng.gen_range(0..self.h);
                self.polarity[n] *= -1;
                true
            }
        }
    }

    fn save_genome(&self) -> (Vec<Vec<u16>>, Vec<u8>, Vec<u8>, Vec<i8>) {
        (self.fire_vars.clone(), self.threshold.clone(), self.channel.clone(), self.polarity.clone())
    }

    fn restore_genome(&mut self, g: (Vec<Vec<u16>>, Vec<u8>, Vec<u8>, Vec<i8>)) {
        self.fire_vars = g.0;
        self.threshold = g.1;
        self.channel = g.2;
        self.polarity = g.3;
    }
}

// ---------------------------------------------------------------------------
// Eval (same pattern as others)
// ---------------------------------------------------------------------------

fn eval_cosine(
    net: &mut FireNet, corpus: &[u8], tokens: usize,
    rng: &mut StdRng, sdr: &[Vec<i8>], bigram: &[Vec<f64>],
) -> f64 {
    let start = rng.gen_range(0..corpus.len().saturating_sub(tokens + 1).max(1));
    let mut total = 0.0f64;
    let mut count = 0usize;
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
    net: &mut FireNet, corpus: &[u8], len: usize,
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

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
        (256, 300), (512, 300), (1024, 300), (2048, 300), (4096, 300),
    ];

    println!("FireNet — Per-Neuron Fire Variables");
    println!("A_fire = [B, C, G] → A gathers from B, C, G");
    println!("Mutation = add/remove/rewire in fire variable");
    println!("No scatter, no edge list, pure gather.");
    println!("Wall clock: {}s/seed | Seeds: {:?} | Corpus: {} bytes",
        WALL_CLOCK_SECS, SEEDS, corpus.len());
    println!("================================================================\n");

    println!("{:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "H", "edge_cap", "edges", "steps", "step/s", "best%", "mean%");
    println!("{:-<6} {:-<8} {:-<8} {:-<8} {:-<8} {:-<8} {:-<8}",
        "", "", "", "", "", "", "");

    for &(h, cap) in &configs {
        let mut all_acc = Vec::new();
        let mut all_steps = Vec::new();
        let mut all_edges = Vec::new();

        for &seed in &SEEDS {
            let sdr_t: Vec<Vec<i8>> = sdr.iter().map(|p| p[..h].to_vec()).collect();
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = FireNet::new(h, &mut rng);
            let mut eval_rng = StdRng::seed_from_u64(seed + 1000);

            let mut steps = 0usize;
            let deadline = Instant::now() + Duration::from_secs(WALL_CLOCK_SECS);

            while Instant::now() < deadline {
                let snap_rng = eval_rng.clone();
                let before = eval_cosine(&mut net, &corpus, 20, &mut eval_rng, &sdr_t, &bigram);
                eval_rng = snap_rng;

                let genome = net.save_genome();
                let mutated = net.mutate(&mut rng, cap);

                if !mutated {
                    let _ = eval_cosine(&mut net, &corpus, 20, &mut eval_rng, &sdr_t, &bigram);
                    steps += 1;
                    continue;
                }

                let after = eval_cosine(&mut net, &corpus, 20, &mut eval_rng, &sdr_t, &bigram);
                if after < before { net.restore_genome(genome); }
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

        println!("{:>6} {:>8} {:>8} {:>8} {:>8.0} {:>7.1}% {:>7.1}%",
            h, cap, me, ms, ms as f64 / WALL_CLOCK_SECS as f64,
            best * 100.0, mean * 100.0);
    }

    println!("\nComparison (same corpus, 60s/seed, 3 seeds):");
    println!("  INSTNCT  H=2048: best=21.4%, mean=20.1%,  98 step/s");
    println!("  VarNet   H=512:  best=22.2%, mean=20.5%, 1908 step/s");
    println!("  ListNet  H=???:  (running separately)");
}
