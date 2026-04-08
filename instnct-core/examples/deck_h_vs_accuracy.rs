//! H vs accuracy at fixed wall-clock time.
//!
//! The question: is H=1024 (fast, L1) better than H=4096 (slow, L2)
//! when you give both the SAME amount of time?
//!
//! More neurons = richer topology search space, but fewer steps/sec.
//! Fewer neurons = more steps, but limited expressivity.
//!
//! Run: cargo run --example deck_h_vs_accuracy --release

use instnct_core::{
    build_bigram_table, build_network, eval_accuracy, eval_smooth, evolution_step_jackpot,
    EvolutionConfig, InitConfig, Int8Projection, SdrTable, StepOutcome,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::{Duration, Instant};

const VOCAB: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const WALL_CLOCK_SECS: u64 = 60; // 1 minute per config per seed
const SEEDS: [u64; 3] = [42, 1042, 2042];
const EVAL_TOKENS: usize = 20;
const FULL_EVAL_LEN: usize = 500;

fn main() {
    // Load real corpus
    let raw = std::fs::read_to_string("instnct-core/tests/fixtures/alice_corpus.txt")
        .expect("corpus file not found — run from repo root");
    let corpus: Vec<u8> = raw.bytes().map(|b| {
        match b {
            b'a'..=b'z' => b - b'a',
            _ => 26,
        }
    }).collect();
    let bigram = build_bigram_table(&corpus, VOCAB);

    let h_configs: Vec<usize> = vec![256, 512, 1024, 2048, 4096];

    println!("H vs Accuracy — Fixed Wall Clock ({} sec per seed)", WALL_CLOCK_SECS);
    println!("Corpus: {} bytes, {} vocab", corpus.len(), VOCAB);
    println!("Seeds: {:?}", SEEDS);
    println!("===========================================================\n");

    println!("{:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>6}",
        "H", "WSS", "steps", "step/s", "best%", "mean%", "edges", "accept");
    println!("{:-<6} {:-<8} {:-<8} {:-<8} {:-<8} {:-<8} {:-<8} {:-<6}",
        "", "", "", "", "", "", "", "");

    for &h in &h_configs {
        let mut all_acc = Vec::new();
        let mut all_steps = Vec::new();
        let mut all_edges = Vec::new();
        let mut all_accept = Vec::new();

        for &seed in &SEEDS {
            let mut rng = StdRng::seed_from_u64(seed);
            let init = InitConfig::empty(h);
            let mut net = build_network(&init, &mut rng);
            let sdr = SdrTable::new(
                VOCAB, h, init.input_end(), SDR_ACTIVE_PCT,
                &mut StdRng::seed_from_u64(seed + 100),
            ).unwrap();
            let mut proj = Int8Projection::new(init.phi_dim, VOCAB, &mut rng);
            let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
            let evo_config = EvolutionConfig {
                edge_cap: 300, // fixed across all H — prevents edge flooding
                accept_ties: true,
            };
            let prop_cfg = init.propagation.clone();
            let output_start = init.output_start();
            let neuron_count = init.neuron_count;

            let mut accepted = 0u32;
            let mut total = 0u32;
            let mut step_count = 0usize;

            let deadline = Instant::now() + Duration::from_secs(WALL_CLOCK_SECS);

            while Instant::now() < deadline {
                let outcome = evolution_step_jackpot(
                    &mut net, &mut proj, &mut rng, &mut eval_rng,
                    |net, proj, eval_rng| {
                        eval_smooth(net, proj, &corpus, EVAL_TOKENS, eval_rng,
                            &sdr, &prop_cfg, output_start, neuron_count, &bigram)
                    },
                    &evo_config,
                    9, // 1+9 jackpot, same as canonical runner
                );
                match outcome {
                    StepOutcome::Accepted => { accepted += 1; total += 1; },
                    StepOutcome::Rejected => { total += 1; },
                    StepOutcome::Skipped => {},
                }
                step_count += 1;
            }

            // Final eval
            let acc = eval_accuracy(
                &mut net, &proj, &corpus, FULL_EVAL_LEN, &mut eval_rng,
                &sdr, &prop_cfg, output_start, neuron_count,
            );

            all_acc.push(acc);
            all_steps.push(step_count);
            all_edges.push(net.edge_count());
            all_accept.push(if total > 0 { accepted as f64 / total as f64 } else { 0.0 });
        }

        let best_acc = all_acc.iter().cloned().fold(0.0f64, f64::max);
        let mean_acc = all_acc.iter().sum::<f64>() / all_acc.len() as f64;
        let mean_steps = all_steps.iter().sum::<usize>() / all_steps.len();
        let mean_edges = all_edges.iter().sum::<usize>() / all_edges.len();
        let mean_accept = all_accept.iter().sum::<f64>() / all_accept.len() as f64;
        let step_per_sec = mean_steps as f64 / WALL_CLOCK_SECS as f64;
        let wss = 22 * h;
        let wss_str = if wss < 1024 {
            format!("{}B", wss)
        } else if wss < 1048576 {
            format!("{:.0}KB", wss as f64 / 1024.0)
        } else {
            format!("{:.1}MB", wss as f64 / 1048576.0)
        };

        println!("{:>6} {:>8} {:>8} {:>8.0} {:>7.1}% {:>7.1}% {:>8} {:>5.1}%",
            h, wss_str, mean_steps, step_per_sec,
            best_acc * 100.0, mean_acc * 100.0,
            mean_edges, mean_accept * 100.0);
    }

    println!("\nNotes:");
    println!("  - Each seed runs for exactly {} seconds", WALL_CLOCK_SECS);
    println!("  - More H = fewer steps but richer search space");
    println!("  - Edge cap from InitConfig::phi() defaults");
    println!("  - The winner is whichever H achieves highest accuracy in fixed time");
}
