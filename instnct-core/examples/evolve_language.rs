//! Evolution with trigram distribution loss on real English.
//!
//! - Builds trigram table from embedded English corpus
//! - Input: 2 context chars → activate input neurons
//! - Output: read 26 output neuron charges → distribution
//! - Loss: cross-entropy(target_distribution, predicted_distribution)
//!
//! Run: cargo run --example evolve_language --release

use instnct_core::{Network, PropagationConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const ALPHABET: usize = 26;
const SPACE: usize = 26; // treat space as 27th "letter" for context
const CHARS: usize = 27; // A-Z + space
const SMOOTH_ALPHA: f64 = 1.0; // Laplace smoothing for predictions
const SMOOTH_BETA: f64 = 0.5; // Laplace smoothing for targets

// I/O layout: SDR input over all neurons, last 26 = output readout
const SDR_ACTIVE_PCT: usize = 20; // 20% of neurons active per letter
const OUTPUT_START: usize = 0; // output = first 26 neurons (overlap with input is OK)
const MIN_NEURONS: usize = 64;

// Embedded English corpus (lowercase, letters + space only)
const CORPUS: &str = "the quick brown fox jumps over the lazy dog \
a the cat sat on the mat and looked at the birds outside the window \
she said that the weather was getting better and they should go for a walk \
it is important to understand that language follows patterns and these patterns \
can be learned by observing the frequency of letter combinations in text \
the most common words in english are the and to of a in that it is for \
when thinking about how letters combine we notice that certain pairs appear \
much more often than others for example th is extremely common while zx almost \
never occurs this statistical regularity is what makes language predictable \
the brain exploits these regularities to process language efficiently and \
a spiking network can learn to do the same by evolving its connection topology \
to match the statistical structure of the input signal that flows through it \
the interference patterns that emerge from this process are the basis of thought";

// ---- Trigram table ----

struct TrigramTable {
    counts: Vec<u32>, // [prev2][prev1][next] flattened, CHARS × CHARS × ALPHABET
}

impl TrigramTable {
    fn new() -> Self {
        Self {
            counts: vec![0u32; CHARS * CHARS * ALPHABET],
        }
    }

    fn index(prev2: usize, prev1: usize, next: usize) -> usize {
        prev2 * CHARS * ALPHABET + prev1 * ALPHABET + next
    }

    fn build_from_corpus(corpus: &str) -> Self {
        let mut table = Self::new();
        let chars: Vec<usize> = corpus
            .chars()
            .filter_map(|c| {
                if c.is_ascii_lowercase() {
                    Some((c as usize) - ('a' as usize))
                } else if c == ' ' {
                    Some(SPACE)
                } else {
                    None
                }
            })
            .collect();

        for window in chars.windows(3) {
            let prev2 = window[0];
            let prev1 = window[1];
            let next = window[2];
            if next < ALPHABET {
                // only predict letters, not space
                table.counts[Self::index(prev2, prev1, next)] += 1;
            }
        }
        table
    }

    /// Get smoothed target distribution for a context pair.
    fn target_distribution(&self, prev2: usize, prev1: usize) -> [f64; ALPHABET] {
        let mut dist = [0.0f64; ALPHABET];
        let mut sum = 0.0;
        for (next, slot) in dist.iter_mut().enumerate() {
            let count = self.counts[Self::index(prev2, prev1, next)] as f64;
            let smoothed = count + SMOOTH_BETA;
            *slot = smoothed;
            sum += smoothed;
        }
        for d in dist.iter_mut() {
            *d /= sum;
        }
        dist
    }

    /// Get all unique contexts that have at least `min_count` observations.
    fn contexts(&self, min_count: u32) -> Vec<(usize, usize)> {
        let mut result = vec![];
        for prev2 in 0..CHARS {
            for prev1 in 0..CHARS {
                let total: u32 = (0..ALPHABET)
                    .map(|next| self.counts[Self::index(prev2, prev1, next)])
                    .sum();
                if total >= min_count {
                    result.push((prev2, prev1));
                }
            }
        }
        result
    }
}

// ---- SDR encoding ----

/// Build SDR table: each of 27 chars gets a unique random sparse pattern.
fn build_sdr_table(neuron_count: usize, rng: &mut StdRng) -> Vec<Vec<i32>> {
    let active_count = neuron_count * SDR_ACTIVE_PCT / 100;
    let mut table = Vec::with_capacity(CHARS);
    for _ in 0..CHARS {
        let mut pattern = vec![0i32; neuron_count];
        let mut activated = 0;
        while activated < active_count {
            let idx = rng.gen_range(0..neuron_count);
            if pattern[idx] == 0 {
                pattern[idx] = 1;
                activated += 1;
            }
        }
        table.push(pattern);
    }
    table
}

// ---- Cross-entropy ----

fn cross_entropy(target: &[f64; ALPHABET], predicted: &[f64; ALPHABET]) -> f64 {
    let mut loss = 0.0;
    for (&t, &p) in target.iter().zip(predicted.iter()) {
        if t > 0.0 {
            loss -= t * p.ln();
        }
    }
    loss
}

/// Read output neuron activations and convert to smoothed distribution.
/// Uses absolute activation values: +1 and -1 both count as "active".
fn output_distribution(net: &Network) -> [f64; ALPHABET] {
    let mut dist = [0.0f64; ALPHABET];
    let mut sum = 0.0;
    for (i, slot) in dist.iter_mut().enumerate() {
        // Use absolute activation — both excitatory and inhibitory firing = signal
        let act = net.activation()[OUTPUT_START + i].unsigned_abs() as f64;
        let charge = net.charge()[OUTPUT_START + i] as f64;
        let signal = act + charge * 0.5; // activation + partial charge
        let smoothed = signal + SMOOTH_ALPHA;
        *slot = smoothed;
        sum += smoothed;
    }
    for d in dist.iter_mut() {
        *d /= sum;
    }
    dist
}

// ---- Evaluation ----

fn evaluate(
    net: &mut Network,
    config: &PropagationConfig,
    trigrams: &TrigramTable,
    contexts: &[(usize, usize)],
    sdr: &[Vec<i32>],
    max_contexts: usize,
    rng: &mut StdRng,
) -> f64 {
    let num_contexts = contexts.len().min(max_contexts);
    if num_contexts == 0 {
        return f64::MAX;
    }

    let mut total_loss = 0.0;

    for _ in 0..num_contexts {
        let &(prev2, prev1) = &contexts[rng.gen_range(0..contexts.len())];
        let target = trigrams.target_distribution(prev2, prev1);

        // Reset and inject SDR context
        net.reset();
        net.propagate(&sdr[prev2], config).unwrap(); // first context char
        net.propagate(&sdr[prev1], config).unwrap(); // second context char

        let predicted = output_distribution(net);
        total_loss += cross_entropy(&target, &predicted);
    }

    total_loss / num_contexts as f64
}

// ---- Main ----

fn main() {
    let neuron_count = 128; // small but enough for 27 input + 26 output + 75 hidden
    let steps = 2000;
    let seed = 42u64;
    let quick_contexts = 5; // quick eval: 5 random contexts
    let full_contexts = 50; // full eval: 50 random contexts

    assert!(
        neuron_count >= MIN_NEURONS,
        "need at least {MIN_NEURONS} neurons for I/O"
    );

    println!("Building trigram table from embedded corpus...");
    let trigrams = TrigramTable::build_from_corpus(CORPUS);
    let contexts = trigrams.contexts(2); // contexts with at least 2 observations
    println!(
        "  corpus: {} chars, {} active contexts\n",
        CORPUS.len(),
        contexts.len()
    );

    let config = PropagationConfig {
        ticks_per_token: 6,
        input_duration_ticks: 2,
        decay_interval_ticks: 6,
    };
    let mut net = Network::new(neuron_count);
    let mut rng = StdRng::seed_from_u64(seed);

    // Random init: 5% density so the network has signal flow from the start
    let target_edges = neuron_count * neuron_count * 5 / 100;
    for _ in 0..target_edges * 3 {
        net.mutate_add_edge(&mut rng);
        if net.edge_count() >= target_edges {
            break;
        }
    }
    // Randomize params
    for i in 0..neuron_count {
        net.threshold_mut()[i] = rng.gen_range(0..=3); // low threshold for relay-heavy init
        net.channel_mut()[i] = rng.gen_range(1..=8);
        if rng.gen_ratio(1, 10) {
            net.polarity_mut()[i] = -1;
        }
    }

    // Build SDR table
    let sdr = build_sdr_table(neuron_count, &mut rng);
    println!(
        "Init: {} edges, {} neurons, SDR {}% active ({} per char)",
        net.edge_count(),
        neuron_count,
        SDR_ACTIVE_PCT,
        neuron_count * SDR_ACTIVE_PCT / 100
    );

    // Initial eval
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let mut best_loss = evaluate(
        &mut net,
        &config,
        &trigrams,
        &contexts,
        &sdr,
        full_contexts,
        &mut eval_rng,
    );
    let mut accepted = 0u32;
    let mut rejected = 0u32;

    println!(
        "Evolve: H={neuron_count}, {steps} steps, seed={seed}, quick={quick_contexts}, full={full_contexts}"
    );
    println!("Initial: edges={}, loss={best_loss:.4}\n", net.edge_count());

    for step in 0..steps {
        let snapshot = net.save_state();

        // Mutate: add(50%) / remove(20%) / rewire(15%) / theta(10%) / channel(5%)
        let roll = rng.gen_range(0..100u32);
        let mutated = match roll {
            0..50 => net.mutate_add_edge(&mut rng),
            50..70 => net.mutate_remove_edge(&mut rng),
            70..85 => net.mutate_rewire(&mut rng),
            85..95 => net.mutate_theta(&mut rng),
            _ => net.mutate_channel(&mut rng),
        };
        if !mutated {
            continue;
        }

        // Quick eval (cheap, frequent)
        let loss = evaluate(
            &mut net,
            &config,
            &trigrams,
            &contexts,
            &sdr,
            quick_contexts,
            &mut eval_rng,
        );

        if loss <= best_loss {
            best_loss = loss;
            accepted += 1;
        } else {
            net.restore_state(&snapshot);
            rejected += 1;
        }

        if (step + 1) % 200 == 0 {
            // Full eval for reporting
            let full_loss = evaluate(
                &mut net,
                &config,
                &trigrams,
                &contexts,
                &sdr,
                full_contexts,
                &mut eval_rng,
            );
            let total = accepted + rejected;
            let rate = if total > 0 {
                accepted as f64 / total as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "  step {:>4}: edges={:>4}  quick_loss={:.4}  full_loss={:.4}  accept={:.0}%",
                step + 1,
                net.edge_count(),
                best_loss,
                full_loss,
                rate
            );
        }
    }

    // Final full eval
    let final_loss = evaluate(
        &mut net,
        &config,
        &trigrams,
        &contexts,
        &sdr,
        full_contexts,
        &mut eval_rng,
    );
    println!(
        "\nFinal: edges={}  loss={:.4}  accept_rate={:.0}%",
        net.edge_count(),
        final_loss,
        accepted as f64 / (accepted + rejected).max(1) as f64 * 100.0
    );

    // Baseline: random distribution loss (uniform prediction)
    let uniform = [1.0 / ALPHABET as f64; ALPHABET];
    let mut baseline_loss = 0.0;
    for &(prev2, prev1) in contexts.iter().take(50) {
        let target = trigrams.target_distribution(prev2, prev1);
        baseline_loss += cross_entropy(&target, &uniform);
    }
    baseline_loss /= 50.0f64.min(contexts.len() as f64);
    println!("Baseline (uniform): loss={baseline_loss:.4}");
    println!(
        "Improvement vs uniform: {:.1}%",
        (1.0 - final_loss / baseline_loss) * 100.0
    );

    // Debug: show what the network actually outputs for "th?" context
    println!("\n--- Debug: output for context 'th' ---");
    net.reset();
    net.propagate(&sdr[19], &config).unwrap(); // T
    net.propagate(&sdr[7], &config).unwrap(); // H

    print!("  Output charges: ");
    let mut nonzero = 0;
    for i in 0..ALPHABET {
        let c = net.charge()[OUTPUT_START + i];
        if c > 0 {
            print!("{}={} ", (b'A' + i as u8) as char, c);
            nonzero += 1;
        }
    }
    println!("\n  Non-zero outputs: {nonzero}/{ALPHABET}");

    print!("  Output activations: ");
    for i in 0..ALPHABET {
        let a = net.activation()[OUTPUT_START + i];
        if a != 0 {
            print!("{}={} ", (b'A' + i as u8) as char, a);
        }
    }
    println!();

    // Also check: are ANY neurons active after propagation?
    let total_active = net.activation().iter().filter(|&&a| a != 0).count();
    let total_charged = net.charge().iter().filter(|&&c| c > 0).count();
    println!("  Total active neurons: {total_active}/{neuron_count}");
    println!("  Total charged neurons: {total_charged}/{neuron_count}");
}
