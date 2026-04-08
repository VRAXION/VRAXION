//! A/B/C/D cache boundary test — accuracy at fixed wall clock.
//!
//! WSS = 22*H + 2*E bytes | L1=32KB, L2=512KB, L3=4MB (Van Gogh)
//!
//! Run: cargo run --example deck_cache_abcd --release

use instnct_core::{
    build_bigram_table, build_network, eval_accuracy, eval_smooth, evolution_step_jackpot,
    EvolutionConfig, InitConfig, Int8Projection, SdrTable, StepOutcome,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::{Duration, Instant};

const VOCAB: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const WALL_CLOCK_SECS: u64 = 60;
const SEEDS: [u64; 3] = [42, 1042, 2042];
const EVAL_TOKENS: usize = 20;
const FULL_EVAL_LEN: usize = 500;
const EDGE_CAP: usize = 300;

struct TestConfig {
    label: &'static str,
    cache: &'static str,
    h: usize,
}

fn main() {
    let raw = std::fs::read_to_string("instnct-core/tests/fixtures/alice_corpus.txt")
        .expect("corpus not found — run from repo root");
    let corpus: Vec<u8> = raw.bytes().map(|b| match b {
        b'a'..=b'z' => b - b'a',
        _ => 26,
    }).collect();
    let bigram = build_bigram_table(&corpus, VOCAB);

    // Each config lands in a different cache level (WSS = 22*H + 2*300)
    let configs = vec![
        // L1: WSS ≤ 32KB
        TestConfig { label: "A1", cache: "L1", h: 512 },     // 11.9KB
        TestConfig { label: "A2", cache: "L1", h: 1024 },    // 23.0KB
        TestConfig { label: "A3", cache: "L1 edge", h: 1400 }, // 30.7KB — right at L1 boundary
        // L2: 32KB < WSS ≤ 512KB
        TestConfig { label: "B1", cache: "L2", h: 2048 },    // 44.6KB
        TestConfig { label: "B2", cache: "L2", h: 4096 },    // 88.6KB
        TestConfig { label: "B3", cache: "L2", h: 8192 },    // 176.6KB
        TestConfig { label: "B4", cache: "L2 edge", h: 23000 }, // 494.7KB — right at L2 boundary
        // L3: 512KB < WSS ≤ 4MB
        TestConfig { label: "C1", cache: "L3", h: 32768 },   // 703.6KB
        TestConfig { label: "C2", cache: "L3", h: 65536 },   // 1.4MB
    ];

    println!("Steam Deck Cache A/B/C/D — Fixed {} sec/seed, edge_cap={}", WALL_CLOCK_SECS, EDGE_CAP);
    println!("Corpus: {} bytes | Seeds: {:?}", corpus.len(), SEEDS);
    println!("WSS = 22*H + 2*E | L1=32KB, L2=512KB, L3=4MB");
    println!("================================================================\n");

    println!("{:>3} {:>8} {:>7} {:>8} {:>8} {:>8} {:>8} {:>6} {:>6}",
        "", "cache", "H", "WSS", "steps", "step/s", "best%", "mean%", "edges");
    println!("{:-<3} {:-<8} {:-<7} {:-<8} {:-<8} {:-<8} {:-<8} {:-<6} {:-<6}",
        "", "", "", "", "", "", "", "", "");

    for cfg in &configs {
        let wss = 22 * cfg.h + 2 * EDGE_CAP;
        let wss_str = if wss < 1024 {
            format!("{}B", wss)
        } else if wss < 1048576 {
            format!("{:.1}KB", wss as f64 / 1024.0)
        } else {
            format!("{:.2}MB", wss as f64 / 1048576.0)
        };

        let mut all_acc = Vec::new();
        let mut all_steps = Vec::new();
        let mut all_edges = Vec::new();

        for &seed in &SEEDS {
            let mut rng = StdRng::seed_from_u64(seed);
            let init = InitConfig::empty(cfg.h);
            let mut net = build_network(&init, &mut rng);
            let sdr = SdrTable::new(
                VOCAB, cfg.h, init.input_end(), SDR_ACTIVE_PCT,
                &mut StdRng::seed_from_u64(seed + 100),
            ).unwrap();
            let mut proj = Int8Projection::new(init.phi_dim, VOCAB, &mut rng);
            let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
            let evo_config = EvolutionConfig { edge_cap: EDGE_CAP, accept_ties: true };
            let prop_cfg = init.propagation.clone();
            let output_start = init.output_start();
            let neuron_count = init.neuron_count;

            let mut step_count = 0usize;
            let deadline = Instant::now() + Duration::from_secs(WALL_CLOCK_SECS);

            while Instant::now() < deadline {
                evolution_step_jackpot(
                    &mut net, &mut proj, &mut rng, &mut eval_rng,
                    |net, proj, eval_rng| {
                        eval_smooth(net, proj, &corpus, EVAL_TOKENS, eval_rng,
                            &sdr, &prop_cfg, output_start, neuron_count, &bigram)
                    },
                    &evo_config, 9,
                );
                step_count += 1;
            }

            let acc = eval_accuracy(
                &mut net, &proj, &corpus, FULL_EVAL_LEN, &mut eval_rng,
                &sdr, &prop_cfg, output_start, neuron_count,
            );

            all_acc.push(acc);
            all_steps.push(step_count);
            all_edges.push(net.edge_count());
        }

        let best = all_acc.iter().cloned().fold(0.0f64, f64::max);
        let mean = all_acc.iter().sum::<f64>() / all_acc.len() as f64;
        let mean_steps = all_steps.iter().sum::<usize>() / all_steps.len();
        let mean_edges = all_edges.iter().sum::<usize>() / all_edges.len();
        let step_s = mean_steps as f64 / WALL_CLOCK_SECS as f64;

        println!("{:>3} {:>8} {:>7} {:>8} {:>8} {:>8.0} {:>7.1}% {:>5.1}% {:>6}",
            cfg.label, cfg.cache, cfg.h, wss_str, mean_steps, step_s,
            best * 100.0, mean * 100.0, mean_edges);
    }
}
