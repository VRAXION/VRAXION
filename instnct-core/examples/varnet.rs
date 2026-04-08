//! Variable-reference network: topology stored as per-node input references.
//!
//! Instead of a global edge list, each node has K input slots pointing to other nodes.
//! The "graph" IS the references. Mutation = change which node an input slot points to.
//!
//! Comparison with INSTNCT on English bigram task at fixed wall clock time.
//!
//! Run: cargo run --example varnet --release

use instnct_core::build_bigram_table;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

const VOCAB: usize = 27;
const WALL_CLOCK_SECS: u64 = 60;
const SEEDS: [u64; 3] = [42, 1042, 2042];
const TICKS: usize = 6;

// ---------------------------------------------------------------------------
// Variable-reference network
// ---------------------------------------------------------------------------

/// Each node has K input references (indices to other nodes) + parameters.
/// The topology is embedded in the input references — no separate edge list.
#[derive(Clone)]
struct VarNode {
    inputs: [u16; 3],   // 3 input slots — which nodes feed into me
    threshold: u8,       // 0-15, effective +1
    channel: u8,         // 1-8 phase gating
    polarity: i8,        // +1 or -1
    // --- runtime state (not part of genome) ---
    charge: i16,
    activation: i8,
}

#[derive(Clone)]
struct VarNet {
    nodes: Vec<VarNode>,
    input_dim: usize,     // nodes 0..input_dim receive input
    output_start: usize,  // nodes output_start..H are output
}

/// Phase gating LUT (same as INSTNCT)
const PHASE_BASE: [i16; 8] = [7, 8, 10, 12, 13, 12, 10, 8];

impl VarNet {
    fn new(h: usize, rng: &mut impl Rng) -> Self {
        // phi-overlap geometry
        let phi_dim = (h as f64 / 1.618).round() as usize;
        let input_dim = phi_dim;
        let output_start = h - phi_dim;

        let mut nodes = Vec::with_capacity(h);
        for i in 0..h {
            // Random input references — can point to any node (including self for recurrence)
            let inputs = [
                rng.gen_range(0..h) as u16,
                rng.gen_range(0..h) as u16,
                rng.gen_range(0..h) as u16,
            ];
            nodes.push(VarNode {
                inputs,
                threshold: rng.gen_range(0..=7),
                channel: rng.gen_range(1..=8),
                polarity: if rng.gen_ratio(1, 10) { -1 } else { 1 },
                charge: 0,
                activation: 0,
            });
        }
        VarNet { nodes, input_dim, output_start }
    }

    fn h(&self) -> usize {
        self.nodes.len()
    }

    fn reset(&mut self) {
        for node in &mut self.nodes {
            node.charge = 0;
            node.activation = 0;
        }
    }

    fn propagate(&mut self, input: &[i8], tick: usize) {
        let h = self.h();

        // 1. Input injection (first 2 ticks)
        if tick < 2 {
            for i in 0..self.input_dim.min(input.len()) {
                self.nodes[i].charge = self.nodes[i].charge.saturating_add(input[i] as i16);
            }
        }

        // 2. Gather incoming from input references
        //    Each node reads activation from its input slots
        //    (we need a scratch buffer to avoid read-write aliasing)
        let mut incoming = vec![0i16; h];
        for i in 0..h {
            let mut sum = 0i16;
            for &inp_idx in &self.nodes[i].inputs {
                let idx = inp_idx as usize;
                if idx < h {
                    sum = sum.saturating_add(self.nodes[idx].activation as i16);
                }
            }
            incoming[i] = sum;
        }

        // 3. Charge accumulation + spike decision
        for i in 0..h {
            let node = &mut self.nodes[i];
            node.charge = node.charge.saturating_add(incoming[i]);

            // Charge decay every 6 ticks
            if tick % 6 == 5 && node.charge > 0 {
                node.charge -= 1;
            }

            // Spike decision with phase gating
            let phase_idx = (tick as u8 + 9 - node.channel) & 7;
            let phase_mult = PHASE_BASE[phase_idx as usize];
            let threshold_eff = (node.threshold as i16 + 1) * phase_mult;
            let charge_x10 = node.charge * 10;

            if charge_x10 >= threshold_eff {
                node.activation = node.polarity;
                node.charge = 0;
            } else {
                node.activation = 0;
            }
        }
    }

    /// Readout: simple sum of activations in output zone per class
    fn readout(&self, num_classes: usize) -> Vec<f64> {
        let mut output = vec![0.0f64; num_classes];
        let output_zone = self.output_start..self.h();
        let zone_len = output_zone.len();
        if zone_len == 0 || num_classes == 0 {
            return output;
        }
        // Distribute output neurons across classes
        for (i, node_idx) in output_zone.enumerate() {
            let class = i * num_classes / zone_len;
            output[class] += self.nodes[node_idx].activation as f64;
        }
        output
    }

    // --- Mutations ---

    fn mutate(&mut self, rng: &mut impl Rng) -> bool {
        let h = self.h();
        let roll = rng.gen_range(0..100u32);
        match roll {
            0..60 => {
                // 60% — rewire: change one input reference
                let node = rng.gen_range(0..h);
                let slot = rng.gen_range(0..3usize);
                let new_target = rng.gen_range(0..h) as u16;
                if self.nodes[node].inputs[slot] == new_target {
                    return false;
                }
                self.nodes[node].inputs[slot] = new_target;
                true
            }
            60..80 => {
                // 20% — threshold
                let node = rng.gen_range(0..h);
                let old = self.nodes[node].threshold;
                self.nodes[node].threshold = rng.gen_range(0..=15);
                self.nodes[node].threshold != old
            }
            80..95 => {
                // 15% — channel
                let node = rng.gen_range(0..h);
                let old = self.nodes[node].channel;
                self.nodes[node].channel = rng.gen_range(1..=8);
                self.nodes[node].channel != old
            }
            _ => {
                // 5% — polarity flip
                let node = rng.gen_range(0..h);
                self.nodes[node].polarity *= -1;
                true
            }
        }
    }

    fn save_state(&self) -> Vec<VarNode> {
        self.nodes.clone()
    }

    fn restore_state(&mut self, snapshot: Vec<VarNode>) {
        self.nodes = snapshot;
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

fn eval_cosine_bigram(
    net: &mut VarNet,
    corpus: &[u8],
    eval_tokens: usize,
    eval_rng: &mut StdRng,
    sdr: &[Vec<i8>],
    bigram: &[Vec<f64>],
) -> f64 {
    let start = eval_rng.gen_range(0..corpus.len().saturating_sub(eval_tokens + 1).max(1));
    let mut total_score = 0.0f64;
    let mut count = 0usize;

    net.reset();
    for t in 0..eval_tokens {
        let pos = start + t;
        if pos + 1 >= corpus.len() { break; }

        let sym = corpus[pos] as usize;
        let target_sym = corpus[pos + 1] as usize;

        // Run TICKS per token
        for tick in 0..TICKS {
            net.propagate(&sdr[sym], tick);
        }

        // Readout and compare with bigram target
        let output = net.readout(VOCAB);
        let target = &bigram[target_sym];

        // Cosine similarity
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for j in 0..VOCAB {
            dot += output[j] * target[j];
            norm_a += output[j] * output[j];
            norm_b += target[j] * target[j];
        }
        let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-10);
        total_score += dot / denom;
        count += 1;
    }

    if count > 0 { total_score / count as f64 } else { 0.0 }
}

fn eval_accuracy(
    net: &mut VarNet,
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
            .map(|(i, _)| i)
            .unwrap_or(0);

        if predicted == target_sym {
            correct += 1;
        }
        count += 1;
    }

    if count > 0 { correct as f64 / count as f64 } else { 0.0 }
}

// ---------------------------------------------------------------------------
// Evolution loop
// ---------------------------------------------------------------------------

fn evolve(
    h: usize,
    corpus: &[u8],
    bigram: &[Vec<f64>],
    sdr: &[Vec<i8>],
    seed: u64,
    deadline: Instant,
) -> (f64, usize, usize) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = VarNet::new(h, &mut rng);
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);

    let eval_tokens = 20;
    let mut steps = 0usize;

    while Instant::now() < deadline {
        // Paired eval
        let eval_rng_snap = eval_rng.clone();
        let before = eval_cosine_bigram(&mut net, corpus, eval_tokens, &mut eval_rng, sdr, bigram);
        eval_rng = eval_rng_snap;

        // Save + mutate
        let snapshot = net.save_state();
        let mutated = net.mutate(&mut rng);

        if !mutated {
            // Advance eval_rng for parity
            let _ = eval_cosine_bigram(&mut net, corpus, eval_tokens, &mut eval_rng, sdr, bigram);
            steps += 1;
            continue;
        }

        // Eval after
        let after = eval_cosine_bigram(&mut net, corpus, eval_tokens, &mut eval_rng, sdr, bigram);

        // Accept/reject
        if after < before {
            net.restore_state(snapshot);
        }

        steps += 1;
    }

    // Final accuracy
    let acc = eval_accuracy(&mut net, corpus, 500, &mut eval_rng, sdr);
    let effective_edges = h * 3; // fixed fan-in of 3

    (acc, steps, effective_edges)
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

    // Build SDR patterns (simple: one-hot with some spread)
    let sdr: Vec<Vec<i8>> = (0..VOCAB).map(|sym| {
        let mut rng = StdRng::seed_from_u64(sym as u64 + 9999);
        let h_max = 4096; // max H we'll test
        let mut pattern = vec![0i8; h_max];
        let active_bits = h_max / 5; // 20%
        let mut placed = 0;
        while placed < active_bits.min(h_max / 5) {
            let idx = rng.gen_range(0..h_max * 618 / 1000); // within input_dim
            if pattern[idx] == 0 {
                pattern[idx] = 1;
                placed += 1;
            }
        }
        pattern
    }).collect();

    let h_configs = [256, 512, 1024, 2048, 4096];

    println!("VarNet — Variable-Reference Network vs INSTNCT");
    println!("Topology = per-node input refs, no edge list");
    println!("Fan-in = 3 fixed | Mutation = rewire ref (60%), theta (20%), channel (15%), polarity (5%)");
    println!("Wall clock: {} sec/seed | Seeds: {:?}", WALL_CLOCK_SECS, SEEDS);
    println!("Corpus: {} bytes", corpus.len());
    println!("================================================================\n");

    println!("{:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "H", "fan_in", "eff_edge", "steps", "step/s", "best%", "mean%");
    println!("{:-<6} {:-<8} {:-<8} {:-<8} {:-<8} {:-<8} {:-<8}",
        "", "", "", "", "", "", "");

    for &h in &h_configs {
        let mut all_acc = Vec::new();
        let mut all_steps = Vec::new();

        for &seed in &SEEDS {
            let sdr_trimmed: Vec<Vec<i8>> = sdr.iter()
                .map(|p| p[..h].to_vec())
                .collect();

            let deadline = Instant::now() + Duration::from_secs(WALL_CLOCK_SECS);
            let (acc, steps, _) = evolve(h, &corpus, &bigram, &sdr_trimmed, seed, deadline);

            all_acc.push(acc);
            all_steps.push(steps);
        }

        let best = all_acc.iter().cloned().fold(0.0f64, f64::max);
        let mean = all_acc.iter().sum::<f64>() / all_acc.len() as f64;
        let mean_steps = all_steps.iter().sum::<usize>() / all_steps.len();
        let step_s = mean_steps as f64 / WALL_CLOCK_SECS as f64;
        let eff_edges = h * 3;

        println!("{:>6} {:>8} {:>8} {:>8} {:>8.0} {:>7.1}% {:>7.1}%",
            h, 3, eff_edges, mean_steps, step_s,
            best * 100.0, mean * 100.0);
    }

    println!("\nFor comparison, INSTNCT results (same corpus, same time):");
    println!("  H=256:  best=20.4%, mean=19.2%, steps=39K");
    println!("  H=2048: best=21.4%, mean=20.1%, steps=6K");
}
