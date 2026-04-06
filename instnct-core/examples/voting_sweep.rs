//! Voting output sweep: dedicated output neuron groups vote for characters.
//!
//! No W matrix — output neurons directly vote by charge accumulation.
//! Input zone unchanged (0..158, SDR 20%). Output zone = dedicated voters.
//!
//! Sweep: output neurons per char × seeds × steps
//!
//! Run: cargo run --example voting_sweep --release -- <corpus-path>

use instnct_core::{load_corpus, Network, PropagationConfig, SdrTable};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const INPUT_DIM: usize = 158; // fixed, proven SDR input

// ---------------------------------------------------------------------------
// Voting readout: group output neurons, sum charge per group
// ---------------------------------------------------------------------------

fn predict_voting(charge: &[u32], output_start: usize, neurons_per_char: usize) -> usize {
    let mut best_char = 0;
    let mut best_vote: u32 = 0;
    for c in 0..CHARS {
        let group_start = output_start + c * neurons_per_char;
        let group_end = group_start + neurons_per_char;
        let vote: u32 = charge[group_start..group_end].iter().sum();
        if vote > best_vote {
            best_vote = vote;
            best_char = c;
        }
    }
    best_char
}

fn margin_fitness(charge: &[u32], output_start: usize, neurons_per_char: usize, correct: usize) -> f64 {
    let mut votes = Vec::with_capacity(CHARS);
    for c in 0..CHARS {
        let group_start = output_start + c * neurons_per_char;
        let group_end = group_start + neurons_per_char;
        let vote: u32 = charge[group_start..group_end].iter().sum();
        votes.push(vote);
    }
    let correct_vote = votes[correct] as f64;
    let best_wrong = votes.iter().enumerate()
        .filter(|&(i, _)| i != correct)
        .map(|(_, &v)| v as f64)
        .fold(0.0f64, f64::max);
    // Margin: how much the correct answer beats the best wrong answer
    // Positive = correct wins, negative = wrong answer wins
    correct_vote - best_wrong
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------


fn sample_eval_offset(corpus_len: usize, len: usize, rng: &mut StdRng) -> Option<usize> {
    if corpus_len <= len { return None; }
    Some(rng.gen_range(0..=corpus_len - len - 1))
}

fn bar(val: f64, max_val: f64, width: usize) -> String {
    let filled = ((val / max_val) * width as f64).round().min(width as f64) as usize;
    format!("{}{}", "#".repeat(filled), ".".repeat(width - filled))
}

// ---------------------------------------------------------------------------
// Eval with voting readout
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn eval_voting_accuracy(
    net: &mut Network,
    corpus: &[u8],
    len: usize,
    rng: &mut StdRng,
    sdr: &SdrTable,
    config: &PropagationConfig,
    output_start: usize,
    neurons_per_char: usize,
) -> f64 {
    let Some(off) = sample_eval_offset(corpus.len(), len, rng) else { return 0.0; };
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), config).unwrap();
        if predict_voting(net.charge(), output_start, neurons_per_char) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

// ---------------------------------------------------------------------------
// Evolution loop (no W matrix, topology + params only)
// ---------------------------------------------------------------------------

struct RunConfig {
    neurons_per_char: usize,
    seed: u64,
    steps: usize,
    use_margin: bool,
}

#[allow(dead_code)]
struct RunResult {
    neurons_per_char: usize,
    seed: u64,
    accuracy: f64,
    edge_count: usize,
    use_margin: bool,
}

fn run_one(cfg: &RunConfig, corpus: &[u8]) -> RunResult {
    let output_neurons = CHARS * cfg.neurons_per_char;
    let h = INPUT_DIM + output_neurons;
    let output_start = INPUT_DIM;
    let edge_cap = h * h * 7 / 100;

    // Build network: custom init (no phi-overlap, no chains for now)
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut net = Network::new(h);

    // Direct highways: input → output (like chain-50 but direct since no overlap)
    for _ in 0..50 {
        let src = rng.gen_range(0..INPUT_DIM) as u16;
        let tgt = rng.gen_range(output_start..h) as u16;
        net.graph_mut().add_edge(src, tgt);
    }

    // Fill to 5% density
    let target_edges = h * h * 5 / 100;
    for _ in 0..target_edges * 3 {
        net.mutate_add_edge(&mut rng);
        if net.edge_count() >= target_edges { break; }
    }

    // Random params
    for i in 0..h {
        net.threshold_mut()[i] = rng.gen_range(0..=7u32);
        net.channel_mut()[i] = rng.gen_range(1..=8u8);
        if rng.gen_ratio(1, 10) { net.polarity_mut()[i] = -1; }
    }

    let prop_config = PropagationConfig {
        ticks_per_token: 6,
        input_duration_ticks: 2,
        decay_interval_ticks: 6,
        use_refractory: false,
    };

    // SDR for input zone only (same as before)
    let sdr = SdrTable::new(CHARS, h, INPUT_DIM, SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(cfg.seed + 100)).unwrap();
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed + 1000);

    // Evolution loop: paired eval, density-capped acceptance, topology+params only
    for step in 0..cfg.steps {
        // Paired eval: before
        let eval_snapshot = eval_rng.clone();
        let before = if cfg.use_margin {
            eval_margin_fitness(&mut net, corpus, 100, &mut eval_rng, &sdr, &prop_config, output_start, cfg.neurons_per_char)
        } else {
            eval_voting_accuracy(&mut net, corpus, 100, &mut eval_rng, &sdr, &prop_config, output_start, cfg.neurons_per_char)
        };
        eval_rng = eval_snapshot;

        let snapshot = net.save_state();

        // Mutate: topology + params (no projection!)
        let roll = rng.gen_range(0..100u32);
        let mutated = match roll {
            0..28 => net.mutate_add_edge(&mut rng),
            28..44 => net.mutate_remove_edge(&mut rng),
            44..56 => net.mutate_rewire(&mut rng),
            56..72 => net.mutate_reverse(&mut rng),
            72..80 => net.mutate_mirror(&mut rng),
            80..86 => net.mutate_enhance(&mut rng),
            86..92 => net.mutate_theta(&mut rng),
            92..96 => net.mutate_channel(&mut rng),
            _ => net.mutate_polarity(&mut rng),
        };

        if !mutated {
            let _ = if cfg.use_margin {
                eval_margin_fitness(&mut net, corpus, 100, &mut eval_rng, &sdr, &prop_config, output_start, cfg.neurons_per_char)
            } else {
                eval_voting_accuracy(&mut net, corpus, 100, &mut eval_rng, &sdr, &prop_config, output_start, cfg.neurons_per_char)
            };
            continue;
        }

        let after = if cfg.use_margin {
            eval_margin_fitness(&mut net, corpus, 100, &mut eval_rng, &sdr, &prop_config, output_start, cfg.neurons_per_char)
        } else {
            eval_voting_accuracy(&mut net, corpus, 100, &mut eval_rng, &sdr, &prop_config, output_start, cfg.neurons_per_char)
        };

        let accepted = if net.edge_count() < edge_cap { after >= before } else { after > before };
        if !accepted {
            net.restore_state(&snapshot);
        }

        if (step + 1) % 5000 == 0 {
            let full = eval_voting_accuracy(&mut net, corpus, 2000, &mut eval_rng, &sdr, &prop_config, output_start, cfg.neurons_per_char);
            let label = if cfg.use_margin { "margin" } else { "top1" };
            println!("  npc={:<3} {:<6} seed={:<5} step {:>5}: |{}| {:.1}%  edges={}",
                cfg.neurons_per_char, label, cfg.seed, step + 1,
                bar(full, 0.30, 30), full * 100.0, net.edge_count());
        }
    }

    let final_acc = eval_voting_accuracy(&mut net, corpus, 5000, &mut eval_rng, &sdr, &prop_config, output_start, cfg.neurons_per_char);
    let label = if cfg.use_margin { "margin" } else { "top1" };
    println!("  npc={:<3} {:<6} seed={:<5} -> FINAL {:.1}%  edges={}",
        cfg.neurons_per_char, label, cfg.seed, final_acc * 100.0, net.edge_count());

    RunResult {
        neurons_per_char: cfg.neurons_per_char,
        seed: cfg.seed,
        accuracy: final_acc,
        edge_count: net.edge_count(),
        use_margin: cfg.use_margin,
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_margin_fitness(
    net: &mut Network,
    corpus: &[u8],
    len: usize,
    rng: &mut StdRng,
    sdr: &SdrTable,
    config: &PropagationConfig,
    output_start: usize,
    neurons_per_char: usize,
) -> f64 {
    let Some(off) = sample_eval_offset(corpus.len(), len, rng) else { return 0.0; };
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut total_margin = 0.0f64;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), config).unwrap();
        total_margin += margin_fitness(net.charge(), output_start, neurons_per_char, seg[i + 1] as usize);
    }
    total_margin / len as f64
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let steps = 15000;
    let seeds = [42u64, 123, 7];

    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars\n", corpus.len());

    // Sweep configs
    let mut configs = Vec::new();
    for &npc in &[5, 10, 20] {  // neurons per char
        for &seed in &seeds {
            // Top-1 accuracy fitness
            configs.push(RunConfig { neurons_per_char: npc, seed, steps, use_margin: false });
            // Margin fitness
            configs.push(RunConfig { neurons_per_char: npc, seed, steps, use_margin: true });
        }
    }

    let total_h: Vec<usize> = [5, 10, 20].iter().map(|&npc| INPUT_DIM + CHARS * npc).collect();
    println!("=== Voting Output Sweep ===");
    println!("  Input: {} neurons (SDR 20%, fixed)", INPUT_DIM);
    println!("  Output neurons/char: 5, 10, 20 → H={}, {}, {}", total_h[0], total_h[1], total_h[2]);
    println!("  Fitness: top1 accuracy vs margin");
    println!("  Seeds: {:?}, steps={}", seeds, steps);
    println!("  Total configs: {}\n", configs.len());

    let results: Vec<RunResult> = configs
        .par_iter()
        .map(|cfg| run_one(cfg, &corpus))
        .collect();

    // Summary
    println!("\n=== SUMMARY (mean across {} seeds) ===\n", seeds.len());
    println!("{:<5} {:<8} {:>8} {:>8}", "npc", "fitness", "mean%", "edges");
    println!("{}", "-".repeat(35));

    let mut seen = Vec::new();
    for r in &results {
        let key = (r.neurons_per_char, r.use_margin);
        if seen.contains(&key) { continue; }
        seen.push(key);
        let group: Vec<_> = results.iter()
            .filter(|x| x.neurons_per_char == r.neurons_per_char && x.use_margin == r.use_margin)
            .collect();
        let mean_acc = group.iter().map(|x| x.accuracy).sum::<f64>() / group.len() as f64;
        let mean_edges = group.iter().map(|x| x.edge_count).sum::<usize>() / group.len();
        let label = if r.use_margin { "margin" } else { "top1" };
        println!("{:<5} {:<8} {:>7.1}% {:>8}", r.neurons_per_char, label, mean_acc * 100.0, mean_edges);
    }

    println!("\nBaseline reference: W matrix readout at H=256 → ~17% mean");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predict_voting_basic() {
        // 27 chars × 10 npc = 270 output neurons, output_start=0 → need 270 charge
        let mut charge = vec![0u32; 270];
        // char 1 = neurons [10..20)
        charge[10] = 5;
        charge[15] = 3;
        assert_eq!(predict_voting(&charge, 0, 10), 1); // char 1 has vote=8
    }

    #[test]
    fn margin_positive_when_correct_wins() {
        // 27 chars × 5 npc = 135 output neurons
        let mut charge = vec![0u32; 135];
        charge[0] = 10; // char 0 vote = 10
        charge[5] = 3;  // char 1 vote = 3
        let m = margin_fitness(&charge, 0, 5, 0);
        assert!(m > 0.0, "margin should be positive when correct wins, got {m}");
    }

    #[test]
    fn margin_negative_when_correct_loses() {
        let mut charge = vec![0u32; 135];
        charge[0] = 3;  // char 0 vote = 3
        charge[5] = 10; // char 1 vote = 10
        let m = margin_fitness(&charge, 0, 5, 0);
        assert!(m < 0.0, "margin should be negative when correct loses, got {m}");
    }
}
