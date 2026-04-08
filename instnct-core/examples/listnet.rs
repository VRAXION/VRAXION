//! ListNet — topology as sorted 2D list.
//!
//! The entire network topology is one sorted list of lists:
//!   [source, target1, target2, ...]
//! Rows sorted by source (ascending), targets sorted left-to-right (ascending).
//!
//! The propagation function just receives this list and executes it.
//!
//! Run: cargo run --example listnet --release

use instnct_core::build_bigram_table;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

const VOCAB: usize = 27;
const WALL_CLOCK_SECS: u64 = 60;
const SEEDS: [u64; 3] = [42, 1042, 2042];
const TICKS: usize = 6;

// ---------------------------------------------------------------------------
// Phase gating LUT (same as INSTNCT)
// ---------------------------------------------------------------------------
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];

// ---------------------------------------------------------------------------
// ListNet
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ListNet {
    /// The "program": sorted list of [source, target1, target2, ...]
    topology: Vec<Vec<u16>>,
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

impl ListNet {
    fn new(h: usize, edge_target: usize, rng: &mut impl Rng) -> Self {
        let phi_dim = (h as f64 / 1.618).round() as usize;
        let input_dim = phi_dim;
        let output_start = h - phi_dim;

        // Start empty — evolution builds the topology
        let topology = Vec::new();

        let mut threshold = vec![0u8; h];
        let mut channel = vec![0u8; h];
        let mut polarity = vec![1i8; h];
        for i in 0..h {
            threshold[i] = rng.gen_range(0..=7);
            channel[i] = rng.gen_range(1..=8);
            if rng.gen_ratio(1, 10) {
                polarity[i] = -1;
            }
        }

        ListNet {
            topology,
            threshold,
            channel,
            polarity,
            charge: vec![0i16; h],
            activation: vec![0i8; h],
            h,
            input_dim,
            output_start,
        }
    }

    fn reset(&mut self) {
        self.charge.iter_mut().for_each(|c| *c = 0);
        self.activation.iter_mut().for_each(|a| *a = 0);
    }

    fn edge_count(&self) -> usize {
        self.topology.iter().map(|row| row.len().saturating_sub(1)).sum()
    }

    /// The core: propagation from the sorted list.
    fn propagate(&mut self, input: &[i8], tick: usize) {
        let h = self.h;

        // 1. Input injection (first 2 ticks)
        if tick < 2 {
            for i in 0..self.input_dim.min(input.len()) {
                self.charge[i] = self.charge[i].saturating_add(input[i] as i16);
            }
        }

        // 2. Scatter from topology list — THE function
        let mut incoming = vec![0i16; h];
        for row in &self.topology {
            if row.len() < 2 { continue; }
            let source = row[0] as usize;
            if source >= h { continue; }
            let act = self.activation[source];
            if act != 0 {
                for &target in &row[1..] {
                    let t = target as usize;
                    if t < h {
                        incoming[t] = incoming[t].saturating_add(act as i16);
                    }
                }
            }
        }

        // 3. Charge accumulation + spike decision
        for i in 0..h {
            self.charge[i] = self.charge[i].saturating_add(incoming[i]);

            // Decay
            if tick % 6 == 5 && self.charge[i] > 0 {
                self.charge[i] -= 1;
            }

            // Spike with phase gating
            let phase_idx = (tick as u8 + 9 - self.channel[i]) & 7;
            let phase_mult = PHASE_BASE[phase_idx as usize];
            let threshold_eff = (self.threshold[i] as i16 + 1) * phase_mult;
            let charge_x10 = self.charge[i] * 10;

            if charge_x10 >= threshold_eff {
                self.activation[i] = self.polarity[i];
                self.charge[i] = 0;
            } else {
                self.activation[i] = 0;
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

    // --- Topology mutations ---

    /// Find the row index for a given source (binary search since sorted)
    fn find_row(&self, source: u16) -> Option<usize> {
        self.topology.iter().position(|row| row.first() == Some(&source))
    }

    /// Add edge source→target. Returns true if new.
    fn add_edge(&mut self, source: u16, target: u16) -> bool {
        if source == target { return false; }
        if source as usize >= self.h || target as usize >= self.h { return false; }

        if let Some(row_idx) = self.find_row(source) {
            let row = &mut self.topology[row_idx];
            // Check if target already exists (sorted, so binary search)
            match row[1..].binary_search(&target) {
                Ok(_) => false, // already exists
                Err(pos) => {
                    row.insert(1 + pos, target);
                    true
                }
            }
        } else {
            // New row for this source — insert sorted
            let new_row = vec![source, target];
            let insert_pos = self.topology.partition_point(|row| row[0] < source);
            self.topology.insert(insert_pos, new_row);
            true
        }
    }

    /// Remove a random edge. Returns true if removed.
    fn remove_edge(&mut self, rng: &mut impl Rng) -> bool {
        let total = self.edge_count();
        if total == 0 { return false; }

        let pick = rng.gen_range(0..total);
        let mut count = 0;
        for row_idx in 0..self.topology.len() {
            let edges_in_row = self.topology[row_idx].len() - 1;
            if count + edges_in_row > pick {
                let target_idx = 1 + (pick - count);
                self.topology[row_idx].remove(target_idx);
                // Remove row if only source remains
                if self.topology[row_idx].len() <= 1 {
                    self.topology.remove(row_idx);
                }
                return true;
            }
            count += edges_in_row;
        }
        false
    }

    /// Rewire: change one random target to a new random neuron.
    fn rewire(&mut self, rng: &mut impl Rng) -> bool {
        let total = self.edge_count();
        if total == 0 { return false; }

        let pick = rng.gen_range(0..total);
        let mut count = 0;
        for row_idx in 0..self.topology.len() {
            let edges_in_row = self.topology[row_idx].len() - 1;
            if count + edges_in_row > pick {
                let target_idx = 1 + (pick - count);
                let source = self.topology[row_idx][0];
                let new_target = rng.gen_range(0..self.h) as u16;
                if new_target == source { return false; }
                // Remove old, insert new in sorted position
                self.topology[row_idx].remove(target_idx);
                if self.topology[row_idx].len() <= 1 {
                    self.topology.remove(row_idx);
                }
                return self.add_edge(source, new_target);
            }
            count += edges_in_row;
        }
        false
    }

    /// Master mutation function.
    fn mutate(&mut self, rng: &mut impl Rng, edge_cap: usize) -> bool {
        let roll = rng.gen_range(0..100u32);
        match roll {
            0..30 => {
                // 30% add edge
                if self.edge_count() >= edge_cap { return false; }
                let s = rng.gen_range(0..self.h) as u16;
                let t = rng.gen_range(0..self.h) as u16;
                self.add_edge(s, t)
            }
            30..45 => {
                // 15% remove edge
                self.remove_edge(rng)
            }
            45..70 => {
                // 25% rewire
                self.rewire(rng)
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

    fn save_state(&self) -> (Vec<Vec<u16>>, Vec<u8>, Vec<u8>, Vec<i8>) {
        (self.topology.clone(), self.threshold.clone(), self.channel.clone(), self.polarity.clone())
    }

    fn restore_state(&mut self, snap: (Vec<Vec<u16>>, Vec<u8>, Vec<u8>, Vec<i8>)) {
        self.topology = snap.0;
        self.threshold = snap.1;
        self.channel = snap.2;
        self.polarity = snap.3;
    }
}

// ---------------------------------------------------------------------------
// Eval
// ---------------------------------------------------------------------------

fn eval_cosine_bigram(
    net: &mut ListNet,
    corpus: &[u8],
    eval_tokens: usize,
    eval_rng: &mut StdRng,
    sdr: &[Vec<i8>],
    bigram: &[Vec<f64>],
) -> f64 {
    let start = eval_rng.gen_range(0..corpus.len().saturating_sub(eval_tokens + 1).max(1));
    let mut total = 0.0f64;
    let mut count = 0usize;

    net.reset();
    for t in 0..eval_tokens {
        let pos = start + t;
        if pos + 1 >= corpus.len() { break; }
        let sym = corpus[pos] as usize;
        let target_sym = corpus[pos + 1] as usize;

        for tick in 0..TICKS {
            net.propagate(&sdr[sym], tick);
        }

        let output = net.readout(VOCAB);
        let target = &bigram[target_sym];

        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for j in 0..VOCAB {
            dot += output[j] * target[j];
            na += output[j] * output[j];
            nb += target[j] * target[j];
        }
        let denom = (na.sqrt() * nb.sqrt()).max(1e-10);
        total += dot / denom;
        count += 1;
    }
    if count > 0 { total / count as f64 } else { 0.0 }
}

fn eval_accuracy(
    net: &mut ListNet,
    corpus: &[u8],
    eval_len: usize,
    eval_rng: &mut StdRng,
    sdr: &[Vec<i8>],
) -> f64 {
    let start = eval_rng.gen_range(0..corpus.len().saturating_sub(eval_len + 1).max(1));
    let mut correct = 0usize;
    let mut count = 0usize;

    net.reset();
    for t in 0..eval_len {
        let pos = start + t;
        if pos + 1 >= corpus.len() { break; }
        let sym = corpus[pos] as usize;
        let target_sym = corpus[pos + 1] as usize;

        for tick in 0..TICKS {
            net.propagate(&sdr[sym], tick);
        }

        let output = net.readout(VOCAB);
        let predicted = output.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        if predicted == target_sym { correct += 1; }
        count += 1;
    }
    if count > 0 { correct as f64 / count as f64 } else { 0.0 }
}

// ---------------------------------------------------------------------------
// Evolution
// ---------------------------------------------------------------------------

fn evolve(
    h: usize,
    edge_cap: usize,
    corpus: &[u8],
    bigram: &[Vec<f64>],
    sdr: &[Vec<i8>],
    seed: u64,
    deadline: Instant,
) -> (f64, usize, usize) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = ListNet::new(h, edge_cap, &mut rng);
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let eval_tokens = 20;
    let mut steps = 0usize;

    while Instant::now() < deadline {
        let eval_rng_snap = eval_rng.clone();
        let before = eval_cosine_bigram(&mut net, corpus, eval_tokens, &mut eval_rng, sdr, bigram);
        eval_rng = eval_rng_snap;

        let snapshot = net.save_state();
        let mutated = net.mutate(&mut rng, edge_cap);

        if !mutated {
            let _ = eval_cosine_bigram(&mut net, corpus, eval_tokens, &mut eval_rng, sdr, bigram);
            steps += 1;
            continue;
        }

        let after = eval_cosine_bigram(&mut net, corpus, eval_tokens, &mut eval_rng, sdr, bigram);
        if after < before {
            net.restore_state(snapshot);
        }
        steps += 1;
    }

    let acc = eval_accuracy(&mut net, corpus, 500, &mut eval_rng, sdr);
    let edges = net.edge_count();
    (acc, steps, edges)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let raw = std::fs::read_to_string("instnct-core/tests/fixtures/alice_corpus.txt")
        .expect("corpus not found — run from repo root");
    let corpus: Vec<u8> = raw.bytes().map(|b| match b {
        b'a'..=b'z' => b - b'a',
        _ => 26,
    }).collect();
    let bigram = build_bigram_table(&corpus, VOCAB);

    // SDR patterns
    let h_max = 4096;
    let sdr: Vec<Vec<i8>> = (0..VOCAB).map(|sym| {
        let mut rng = StdRng::seed_from_u64(sym as u64 + 9999);
        let input_dim = (h_max as f64 / 1.618).round() as usize;
        let mut pattern = vec![0i8; h_max];
        let active = input_dim / 5;
        let mut placed = 0;
        while placed < active {
            let idx = rng.gen_range(0..input_dim);
            if pattern[idx] == 0 { pattern[idx] = 1; placed += 1; }
        }
        pattern
    }).collect();

    let configs: Vec<(usize, usize)> = vec![
        (256, 300),
        (512, 300),
        (1024, 300),
        (2048, 300),
        (4096, 300),
    ];

    println!("ListNet — Sorted 2D List Topology");
    println!("Format: [source, target1, target2, ...] sorted asc");
    println!("Mutations: add(30%), remove(15%), rewire(25%), theta(15%), channel(10%), polarity(5%)");
    println!("Wall clock: {} sec/seed | Seeds: {:?}", WALL_CLOCK_SECS, SEEDS);
    println!("Corpus: {} bytes", corpus.len());
    println!("================================================================\n");

    println!("{:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "H", "edge_cap", "edges", "steps", "step/s", "best%", "mean%");
    println!("{:-<6} {:-<8} {:-<8} {:-<8} {:-<8} {:-<8} {:-<8}",
        "", "", "", "", "", "", "");

    for &(h, edge_cap) in &configs {
        let mut all_acc = Vec::new();
        let mut all_steps = Vec::new();
        let mut all_edges = Vec::new();

        for &seed in &SEEDS {
            let sdr_trim: Vec<Vec<i8>> = sdr.iter().map(|p| p[..h].to_vec()).collect();
            let deadline = Instant::now() + Duration::from_secs(WALL_CLOCK_SECS);
            let (acc, steps, edges) = evolve(h, edge_cap, &corpus, &bigram, &sdr_trim, seed, deadline);
            all_acc.push(acc);
            all_steps.push(steps);
            all_edges.push(edges);
        }

        let best = all_acc.iter().cloned().fold(0.0f64, f64::max);
        let mean = all_acc.iter().sum::<f64>() / all_acc.len() as f64;
        let mean_steps = all_steps.iter().sum::<usize>() / all_steps.len();
        let mean_edges = all_edges.iter().sum::<usize>() / all_edges.len();
        let step_s = mean_steps as f64 / WALL_CLOCK_SECS as f64;

        println!("{:>6} {:>8} {:>8} {:>8} {:>8.0} {:>7.1}% {:>7.1}%",
            h, edge_cap, mean_edges, mean_steps, step_s,
            best * 100.0, mean * 100.0);
    }

    println!("\nComparison (same corpus, 60s/seed, 3 seeds):");
    println!("  INSTNCT  H=2048 edge_cap=300: best=21.4%, mean=20.1%, 6K steps");
    println!("  VarNet   H=512  fan_in=3:      best=22.2%, mean=20.5%, 115K steps");
}
