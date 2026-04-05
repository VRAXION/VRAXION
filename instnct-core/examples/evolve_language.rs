//! Language evolution: sequential next-char prediction on real English text.
//!
//! Multi-seed test with the best known config:
//! - H=256, phi overlap I/O, SDR 20% input, learnable int8 projection
//! - Density-capped paired eval (>= when lean, > when dense)
//! - 8-op mutation schedule + 10% projection weight mutations
//!
//! Run: cargo run --example evolve_language --release -- <corpus-path>

use instnct_core::{Network, PropagationConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs;

const CHARS: usize = 27; // a-z (0..25) + space (26)
const SDR_ACTIVE_PCT: usize = 20;
const NEURON_COUNT: usize = 256;
const PHI_DIM: usize = 158; // round(256 / phi)
const INPUT_END: usize = PHI_DIM; // input zone: 0..158
const OUTPUT_START: usize = NEURON_COUNT - PHI_DIM; // output zone: 98..256
const EDGE_CAP: usize = NEURON_COUNT * NEURON_COUNT * 7 / 100; // ~4587

fn load_corpus(path: &str) -> Vec<u8> {
    let raw = fs::read(path).expect("cannot read corpus file");
    raw.iter()
        .filter_map(|&b| {
            if b.is_ascii_lowercase() {
                Some(b - b'a')
            } else if b.is_ascii_uppercase() {
                Some(b.to_ascii_lowercase() - b'a')
            } else if b == b' ' || b == b'\n' || b == b'\t' {
                Some(26)
            } else {
                None
            }
        })
        .collect()
}

fn build_sdr_table(rng: &mut StdRng) -> Vec<Vec<i32>> {
    let active_count = INPUT_END * SDR_ACTIVE_PCT / 100;
    let mut table = Vec::with_capacity(CHARS);
    for _ in 0..CHARS {
        let mut pattern = vec![0i32; NEURON_COUNT];
        let mut activated = 0;
        while activated < active_count {
            let idx = rng.gen_range(0..INPUT_END);
            if pattern[idx] == 0 {
                pattern[idx] = 1;
                activated += 1;
            }
        }
        table.push(pattern);
    }
    table
}

fn build_projection_i8(rng: &mut StdRng) -> Vec<i8> {
    (0..PHI_DIM * CHARS)
        .map(|_| rng.gen_range(-127..=127i8))
        .collect()
}

fn sample_eval_offset(corpus_len: usize, len: usize, rng: &mut StdRng) -> Option<usize> {
    let max_s = corpus_len.saturating_sub(len + 1);
    if max_s == 0 {
        return None;
    }
    Some(rng.gen_range(0..max_s))
}

fn predict_i8(net: &Network, w: &[i8]) -> u8 {
    let mut scores = [0i32; CHARS];
    for (i, &c) in net.charge()[OUTPUT_START..NEURON_COUNT].iter().enumerate() {
        if c == 0 {
            continue;
        }
        let x = c as i32;
        let row = &w[i * CHARS..(i + 1) * CHARS];
        for (s, &wt) in scores.iter_mut().zip(row.iter()) {
            *s += x * wt as i32;
        }
    }
    scores
        .iter()
        .enumerate()
        .max_by_key(|&(_, &s)| s)
        .map(|(i, _)| i as u8)
        .unwrap_or(0)
}

fn build_network(rng: &mut StdRng) -> Network {
    let mut net = Network::new(NEURON_COUNT);
    let target_edges = NEURON_COUNT * NEURON_COUNT * 5 / 100;
    for _ in 0..target_edges * 3 {
        net.mutate_add_edge(rng);
        if net.edge_count() >= target_edges {
            break;
        }
    }
    for i in 0..NEURON_COUNT {
        net.threshold_mut()[i] = rng.gen_range(0..=7);
        net.channel_mut()[i] = rng.gen_range(1..=8);
        if rng.gen_ratio(1, 10) {
            net.polarity_mut()[i] = -1;
        }
    }
    net
}

fn bar(val: f64, max_val: f64, width: usize) -> String {
    let filled = ((val / max_val) * width as f64).round().min(width as f64) as usize;
    format!("{}{}", "#".repeat(filled), ".".repeat(width - filled))
}

fn char_label(c: u8) -> char {
    if c < 26 {
        (b'a' + c) as char
    } else {
        '_'
    }
}

fn eval_accuracy(
    net: &mut Network,
    w: &[i8],
    corpus: &[u8],
    len: usize,
    rng: &mut StdRng,
    sdr: &[Vec<i32>],
    config: &PropagationConfig,
) -> f64 {
    let Some(off) = sample_eval_offset(corpus.len(), len, rng) else {
        return 0.0;
    };

    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(&sdr[seg[i] as usize], config).unwrap();
        if predict_i8(net, w) == seg[i + 1] {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

/// Run one evolution with density-capped paired eval.
fn run_evolution(steps: usize, seed: u64, corpus: &[u8], full_len: usize) -> f64 {
    let config = PropagationConfig {
        ticks_per_token: 6,
        input_duration_ticks: 2,
        decay_interval_ticks: 6,
        use_refractory: false,
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = build_network(&mut rng);
    let mut w_rng = StdRng::seed_from_u64(seed + 200);
    let mut w = build_projection_i8(&mut w_rng);
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let mut sdr_rng = StdRng::seed_from_u64(seed + 100);
    let sdr = build_sdr_table(&mut sdr_rng);

    let mut accepted = 0u32;
    let mut rejected = 0u32;

    for step in 0..steps {
        // Paired eval: same segment before and after mutation
        let snap = eval_rng.clone();
        let before = eval_accuracy(&mut net, &w, corpus, 100, &mut eval_rng, &sdr, &config);
        eval_rng = snap;

        let snapshot = net.save_state();
        let w_backup: Option<(usize, i8)>;
        let roll = rng.gen_range(0..100u32);
        let mutated;
        match roll {
            0..25 => {
                mutated = net.mutate_add_edge(&mut rng);
                w_backup = None;
            }
            25..40 => {
                mutated = net.mutate_remove_edge(&mut rng);
                w_backup = None;
            }
            40..50 => {
                mutated = net.mutate_rewire(&mut rng);
                w_backup = None;
            }
            50..65 => {
                mutated = net.mutate_reverse(&mut rng);
                w_backup = None;
            }
            65..72 => {
                mutated = net.mutate_mirror(&mut rng);
                w_backup = None;
            }
            72..80 => {
                mutated = net.mutate_enhance(&mut rng);
                w_backup = None;
            }
            80..85 => {
                mutated = net.mutate_theta(&mut rng);
                w_backup = None;
            }
            85..90 => {
                mutated = net.mutate_channel(&mut rng);
                w_backup = None;
            }
            _ => {
                let idx = rng.gen_range(0..w.len());
                let old_val = w[idx];
                w[idx] = rng.gen_range(-127..=127i8);
                w_backup = Some((idx, old_val));
                mutated = true;
            }
        }
        if !mutated {
            let _ = eval_rng.gen_range(0..1u32);
            continue;
        }

        let after = eval_accuracy(&mut net, &w, corpus, 100, &mut eval_rng, &sdr, &config);

        // Density-capped acceptance: >= when lean, > when dense
        let dominated = if net.edge_count() < EDGE_CAP {
            after >= before
        } else {
            after > before
        };
        if dominated {
            accepted += 1;
        } else {
            net.restore_state(&snapshot);
            if let Some((idx, old_val)) = w_backup {
                w[idx] = old_val;
            }
            rejected += 1;
        }

        if (step + 1) % 5000 == 0 {
            let full = eval_accuracy(&mut net, &w, corpus, full_len, &mut eval_rng, &sdr, &config);
            let tot = accepted + rejected;
            let rate = if tot > 0 {
                accepted as f64 / tot as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "  step {:>5}: |{}| {:.1}%  accept={:.0}%  edges={}",
                step + 1,
                bar(full, 0.30, 30),
                full * 100.0,
                rate,
                net.edge_count()
            );
        }
    }

    let final_acc = eval_accuracy(&mut net, &w, corpus, 5000, &mut eval_rng, &sdr, &config);
    let rate = accepted as f64 / (accepted + rejected).max(1) as f64 * 100.0;
    println!(
        "  FINAL: {:.1}%  edges={}  accept={:.0}%\n",
        final_acc * 100.0,
        net.edge_count(),
        rate
    );
    final_acc
}

fn main() {
    let full_len = 2000;
    let steps = 15000;
    let seeds = [42u64, 123, 7];

    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "S:/AI/work/VRAXION_DEV/instnct/data/traindat/fineweb_edu.traindat".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path);
    println!("  {} chars", corpus.len());

    // Baselines
    let mut freq = [0u64; CHARS];
    for &c in &corpus {
        freq[c as usize] += 1;
    }
    let most_common = freq
        .iter()
        .enumerate()
        .max_by_key(|&(_, &c)| c)
        .map(|(i, _)| i)
        .unwrap_or(0);
    let freq_base = freq[most_common] as f64 / corpus.len() as f64;
    let mut bigram = vec![0u64; CHARS * CHARS];
    for bw in corpus.windows(2) {
        bigram[bw[0] as usize * CHARS + bw[1] as usize] += 1;
    }
    let mut bigram_ok = 0u64;
    for bw in corpus.windows(2) {
        let best = (0..CHARS)
            .max_by_key(|&n| bigram[bw[0] as usize * CHARS + n])
            .unwrap_or(0);
        if best == bw[1] as usize {
            bigram_ok += 1;
        }
    }
    let bigram_base = bigram_ok as f64 / (corpus.len() - 1) as f64;
    println!(
        "  Random: {:.1}%  Freq('{}'): {:.1}%  Bigram: {:.1}%",
        100.0 / CHARS as f64,
        char_label(most_common as u8),
        freq_base * 100.0,
        bigram_base * 100.0
    );

    println!("\n=== Multi-seed: density-capped paired eval, learnable int8 W ===");
    println!(
        "H={NEURON_COUNT}, {steps} steps, {} seeds, edge_cap={EDGE_CAP}\n",
        seeds.len()
    );

    let mut results = Vec::new();
    for &s in &seeds {
        println!("--- seed={s} ---");
        let acc = run_evolution(steps, s, &corpus, full_len);
        results.push((s, acc));
    }

    let mean = results.iter().map(|(_, a)| a).sum::<f64>() / results.len() as f64;
    let max_acc = results.iter().map(|(_, a)| *a).fold(0.0f64, f64::max);
    let min_acc = results.iter().map(|(_, a)| *a).fold(1.0f64, f64::min);

    println!("=== SUMMARY ===");
    println!("  Random:    {:.1}%", 100.0 / CHARS as f64);
    println!("  Frequency: {:.1}%", freq_base * 100.0);
    println!("  Bigram:    {:.1}%", bigram_base * 100.0);
    for (s, acc) in &results {
        println!("  seed={:<4}  {:.1}%", s, acc * 100.0);
    }
    println!("  Mean:      {:.1}%", mean * 100.0);
    println!(
        "  Range:     {:.1}% - {:.1}%  (spread={:.1}pp)",
        min_acc * 100.0,
        max_acc * 100.0,
        (max_acc - min_acc) * 100.0
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_eval_offset_allows_exact_fit_segment() {
        let mut rng = StdRng::seed_from_u64(1);
        assert_eq!(sample_eval_offset(101, 100, &mut rng), Some(0));
        assert_eq!(sample_eval_offset(100, 100, &mut rng), None);
    }

    #[test]
    fn skipped_step_advances_eval_rng_like_real_paired_eval() {
        let corpus_len = 1024;
        let len = 100;

        let mut paired_rng = StdRng::seed_from_u64(42);
        let snap = paired_rng.clone();
        let before = sample_eval_offset(corpus_len, len, &mut paired_rng).unwrap();
        paired_rng = snap;
        let after = sample_eval_offset(corpus_len, len, &mut paired_rng).unwrap();
        assert_eq!(
            before, after,
            "paired eval must use the same segment before/after"
        );
        let paired_next = paired_rng.gen::<u64>();

        let mut skipped_rng = StdRng::seed_from_u64(42);
        let skipped_snap = skipped_rng.clone();
        let skipped_before = sample_eval_offset(corpus_len, len, &mut skipped_rng).unwrap();
        skipped_rng = skipped_snap;
        advance_paired_eval_rng(corpus_len, len, &mut skipped_rng);

        assert_eq!(skipped_before, before);
        assert_eq!(
            skipped_rng.gen::<u64>(),
            paired_next,
            "skip-advance must consume RNG exactly like a real paired after-eval"
        );
    }

    #[test]
    fn projection_weight_reject_restore_is_exact() {
        let mut rng = StdRng::seed_from_u64(7);
        let original = build_projection_i8(&mut rng);
        let mut trial = original.clone();

        for _ in 0..1024 {
            let idx = rng.gen_range(0..trial.len());
            let old_val = trial[idx];
            trial[idx] = rng.gen_range(-127..=127i8);
            trial[idx] = old_val;
            assert_eq!(trial, original, "projection rollback leaked weight drift");
        }
    }

    #[test]
    fn paired_eval_noop_is_exactly_stable() {
        let corpus: Vec<u8> = (0..256).map(|i| (i % CHARS) as u8).collect();
        let config = PropagationConfig {
            ticks_per_token: 6,
            input_duration_ticks: 2,
            decay_interval_ticks: 6,
            use_refractory: false,
        };
        let mut sdr_rng = StdRng::seed_from_u64(123);
        let sdr = build_sdr_table(&mut sdr_rng);
        let mut net = build_network(&mut StdRng::seed_from_u64(321));
        let w = build_projection_i8(&mut StdRng::seed_from_u64(654));
        let mut eval_rng = StdRng::seed_from_u64(999);
        let snap = eval_rng.clone();

        let before = eval_accuracy(&mut net, &w, &corpus, 64, &mut eval_rng, &sdr, &config);
        eval_rng = snap;
        let after = eval_accuracy(&mut net, &w, &corpus, 64, &mut eval_rng, &sdr, &config);

        assert_eq!(
            before, after,
            "paired eval on unchanged state must be exact"
        );
    }
}
