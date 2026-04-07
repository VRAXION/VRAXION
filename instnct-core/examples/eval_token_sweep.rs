//! Sweep: eval token count vs accuracy vs step speed.
//!
//! The evolution bottleneck is now eval, not forward pass.
//! Fewer eval tokens = faster steps but noisier fitness.
//! Question: what's the sweet spot?

use instnct_core::{
    build_bigram_table, build_network, eval_accuracy, eval_smooth, evolution_step_jackpot,
    load_corpus, InitConfig, Int8Projection, SdrTable, StepOutcome,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const CORPUS_PATH: &str = "instnct-core/tests/fixtures/beta_smoke_corpus.txt";
const H: usize = 256;
const TOTAL_EVALS: usize = 60_000; // fixed total eval budget (tokens evaluated)

fn run_trial(eval_tokens: usize, jackpot: usize, corpus: &[u8], bigram: &[Vec<f64>], seed: u64) -> TrialResult {
    let init = InitConfig::phi(H);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = build_network(&init, &mut rng);
    let mut proj = Int8Projection::new(init.phi_dim, CHARS, &mut StdRng::seed_from_u64(seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let sdr = SdrTable::new(CHARS, H, init.input_end(), SDR_ACTIVE_PCT, &mut StdRng::seed_from_u64(seed + 100)).unwrap();
    let evo_config = init.evolution_config();
    let output_start = init.output_start();

    // Calculate steps from fixed eval budget
    // Each step does (1 + jackpot) evals × eval_tokens tokens
    let evals_per_step = (1 + jackpot) * eval_tokens;
    let steps = TOTAL_EVALS / evals_per_step;

    let mut accepted = 0u32;
    let mut peak_acc = 0.0f64;

    let t0 = Instant::now();

    for step in 0..steps {
        let outcome = evolution_step_jackpot(
            &mut net, &mut proj, &mut rng, &mut eval_rng,
            |net, proj, eval_rng| {
                eval_accuracy(net, proj, corpus, eval_tokens, eval_rng, &sdr, &init.propagation, output_start, H)
            },
            &evo_config, jackpot,
        );
        if let StepOutcome::Accepted = outcome { accepted += 1; }

        // Periodic check with full eval
        if steps > 0 && (step + 1) % (steps / 4).max(1) == 0 {
            let acc = eval_accuracy(&mut net, &proj, corpus, 500, &mut eval_rng, &sdr, &init.propagation, output_start, H);
            peak_acc = peak_acc.max(acc);
        }
    }

    let elapsed = t0.elapsed();
    let final_acc = eval_accuracy(&mut net, &proj, corpus, 500, &mut eval_rng, &sdr, &init.propagation, output_start, H);
    peak_acc = peak_acc.max(final_acc);

    TrialResult {
        eval_tokens, jackpot, steps, evals_per_step,
        peak_acc, final_acc, accepted,
        steps_per_sec: steps as f64 / elapsed.as_secs_f64(),
        elapsed_secs: elapsed.as_secs_f64(),
        edges: net.edge_count(),
    }
}

struct TrialResult {
    eval_tokens: usize,
    jackpot: usize,
    steps: usize,
    evals_per_step: usize,
    peak_acc: f64,
    final_acc: f64,
    accepted: u32,
    steps_per_sec: f64,
    elapsed_secs: f64,
    edges: usize,
}

fn main() {
    let corpus = load_corpus(CORPUS_PATH).expect("failed to load corpus");
    let bigram = build_bigram_table(&corpus, CHARS);

    println!("╔═══════════════════════════════════════════════════════════════════════╗");
    println!("║  EVAL TOKEN SWEEP — fixed {}K eval budget, variable token count  ║", TOTAL_EVALS / 1000);
    println!("║  H={}, fewer tokens = more steps but noisier fitness              ║", H);
    println!("╚═══════════════════════════════════════════════════════════════════════╝\n");

    println!("{:>6} {:>7} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>7}",
        "tok/ev", "jackpot", "steps", "step/s", "peak%", "final%", "accepts", "edges", "time");
    println!("{}", "─".repeat(85));

    let configs = [
        (10, 1),  (10, 3),  (10, 9),
        (20, 1),  (20, 3),  (20, 9),
        (50, 1),  (50, 3),  (50, 9),
        (100, 1), (100, 3), (100, 9),
    ];

    let mut results = Vec::new();

    for &(eval_tok, jackpot) in &configs {
        // Run 2 seeds, take better
        let r1 = run_trial(eval_tok, jackpot, &corpus, &bigram, 42);
        let r2 = run_trial(eval_tok, jackpot, &corpus, &bigram, 137);
        let r = if r1.peak_acc >= r2.peak_acc { r1 } else { r2 };

        println!("{:>6} {:>7} {:>7} {:>8.0} {:>7.1}% {:>7.1}% {:>8} {:>8} {:>6.1}s",
            r.eval_tokens, r.jackpot, r.steps, r.steps_per_sec,
            r.peak_acc * 100.0, r.final_acc * 100.0,
            r.accepted, r.edges, r.elapsed_secs);

        results.push(r);
    }

    println!("{}", "─".repeat(85));

    let best = results.iter().max_by(|a, b| a.peak_acc.partial_cmp(&b.peak_acc).unwrap()).unwrap();
    let fastest = results.iter().max_by(|a, b| a.steps_per_sec.partial_cmp(&b.steps_per_sec).unwrap()).unwrap();

    println!("\n  VERDICT:");
    println!("    Best accuracy:  {}tok × {}jk → {:.1}% peak ({} steps, {:.0} step/s)",
        best.eval_tokens, best.jackpot, best.peak_acc * 100.0, best.steps, best.steps_per_sec);
    println!("    Fastest steps:  {}tok × {}jk → {:.0} step/s ({:.1}% peak)",
        fastest.eval_tokens, fastest.jackpot, fastest.steps_per_sec, fastest.peak_acc * 100.0);
}
