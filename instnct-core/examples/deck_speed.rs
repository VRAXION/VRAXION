//! Steam Deck speed sweep: measure tok/sec and step/sec across H sizes.
//!
//! Covers L1 (32K), L2 (512K), L3 (4M) cache boundaries on Van Gogh APU.
//!
//! Run: cargo run --example deck_speed --release

use instnct_core::{
    build_bigram_table, build_network, eval_smooth, evolution_step,
    EvolutionConfig, InitConfig, Int8Projection, Network, PropagationConfig, SdrTable,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

const VOCAB: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;

fn main() {
    let sizes: Vec<usize> = vec![128, 256, 512, 1024, 2048, 4096, 8192];
    let edge_targets = [30, 100, 200];

    // Build a small corpus
    let text = b"the quick brown fox jumps over the lazy dog and the cat sat on the mat ";
    let corpus: Vec<u8> = text.iter().map(|&b| {
        match b {
            b'a'..=b'z' => b - b'a',
            _ => 26, // space
        }
    }).collect();
    let bigram = build_bigram_table(&corpus, VOCAB);

    println!("Steam Deck Speed Sweep (AMD Van Gogh, L1d=32K, L2=512K, L3=4M)");
    println!("================================================================\n");

    // Part 1: Raw propagation speed
    println!("=== RAW PROPAGATION (tok/sec) ===");
    println!("{:>6} {:>6} {:>12} {:>12} {:>12} {:>8}", "H", "edges", "30e", "100e", "200e", "bytes");
    println!("{:-<6} {:-<6} {:-<12} {:-<12} {:-<12} {:-<8}", "", "", "", "", "", "");

    for &h in &sizes {
        let neuron_bytes = h * 6;
        print!("{:>6}", h);

        for &target_edges in &edge_targets {
            let mut rng = StdRng::seed_from_u64(42);
            let cfg = InitConfig::phi(h);
            let mut net = build_network(&cfg, &mut rng);

            // Adjust edge count
            while net.edge_count() > target_edges {
                net.mutate_remove_edge(&mut rng);
            }
            while net.edge_count() < target_edges {
                net.mutate_add_edge(&mut rng);
            }

            let sdr = SdrTable::new(
                VOCAB, h, cfg.input_end(), SDR_ACTIVE_PCT,
                &mut StdRng::seed_from_u64(100),
            ).unwrap();
            let prop_config = PropagationConfig::default();

            // Warmup
            for i in 0..20 {
                let sym = corpus[i % corpus.len()] as usize;
                net.propagate(sdr.pattern(sym), &prop_config);
            }
            net.reset();

            // Measure
            let tokens = 500;
            let start = Instant::now();
            for i in 0..tokens {
                let sym = corpus[i % corpus.len()] as usize;
                net.propagate(sdr.pattern(sym), &prop_config);
            }
            let elapsed = start.elapsed();
            let tok_per_sec = tokens as f64 / elapsed.as_secs_f64();
            print!(" {:>12.0}", tok_per_sec);
        }
        println!(" {:>8}", h * 6);
    }

    println!();

    // Part 2: Evolution step speed (the actual training loop)
    println!("=== EVOLUTION STEP (step/sec, 100 edges) ===");
    println!("{:>6} {:>6} {:>10} {:>12}", "H", "edges", "step/s", "eff_tok/s");
    println!("{:-<6} {:-<6} {:-<10} {:-<12}", "", "", "", "");

    let evo_config = EvolutionConfig { edge_cap: 300, accept_ties: true };

    for &h in &sizes {
        if h > 8192 { continue; }

        let mut rng = StdRng::seed_from_u64(42);
        let cfg = InitConfig::phi(h);
        let mut net = build_network(&cfg, &mut rng);
        let sdr = SdrTable::new(
            VOCAB, h, cfg.input_end(), SDR_ACTIVE_PCT,
            &mut StdRng::seed_from_u64(100),
        ).unwrap();
        let mut proj = Int8Projection::new(cfg.phi_dim, VOCAB, &mut rng);
        let mut eval_rng = StdRng::seed_from_u64(123);
        let prop_cfg = cfg.propagation.clone();
        let output_start = cfg.output_start();
        let neuron_count = cfg.neuron_count;

        // Adjust to ~100 edges
        while net.edge_count() > 100 {
            net.mutate_remove_edge(&mut rng);
        }
        while net.edge_count() < 100 {
            net.mutate_add_edge(&mut rng);
        }
        let edges = net.edge_count();

        let eval_tokens = 20usize;

        // Warmup
        for _ in 0..10 {
            evolution_step(
                &mut net, &mut proj, &mut rng, &mut eval_rng,
                |net, proj, eval_rng| {
                    eval_smooth(net, proj, &corpus, eval_tokens, eval_rng,
                        &sdr, &prop_cfg, output_start, neuron_count, &bigram)
                },
                &evo_config,
            );
        }

        // Measure
        let steps = 200;
        let start = Instant::now();
        for _ in 0..steps {
            evolution_step(
                &mut net, &mut proj, &mut rng, &mut eval_rng,
                |net, proj, eval_rng| {
                    eval_smooth(net, proj, &corpus, eval_tokens, eval_rng,
                        &sdr, &prop_cfg, output_start, neuron_count, &bigram)
                },
                &evo_config,
            );
        }
        let elapsed = start.elapsed();
        let steps_per_sec = steps as f64 / elapsed.as_secs_f64();
        // 2 evals per step (before + after), each eval = eval_tokens * ticks
        let effective_tok = steps_per_sec * 2.0 * eval_tokens as f64;

        println!("{:>6} {:>6} {:>10.1} {:>12.0}", h, edges, steps_per_sec, effective_tok);
    }
}
