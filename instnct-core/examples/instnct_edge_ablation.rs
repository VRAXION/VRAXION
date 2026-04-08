//! INSTNCT library edge ablation: train with the real library, then test empty network.
//!
//! Run: cargo run --example instnct_edge_ablation --release

use instnct_core::{
    build_bigram_table, build_network, eval_accuracy, eval_smooth, evolution_step,
    EvolutionConfig, InitConfig, Int8Projection, Network, SdrTable,
};
use rand::rngs::StdRng;
use rand::SeedableRng;

const VOCAB: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const TRAIN_STEPS: usize = 30_000;
const EVAL_LEN: usize = 2000;

fn main() {
    let raw = std::fs::read_to_string("instnct-core/tests/fixtures/alice_corpus.txt")
        .expect("corpus not found");
    let corpus: Vec<u8> = raw.bytes().map(|b| match b { b'a'..=b'z' => b-b'a', _ => 26 }).collect();
    let bigram = build_bigram_table(&corpus, VOCAB);

    println!("=== INSTNCT LIBRARY EDGE ABLATION ===");
    println!("Train with real library, then compare trained vs empty network + same projection\n");

    for &h in &[256usize] {
        println!("--- H={}, phi init, {} steps ---", h, TRAIN_STEPS);

        for &seed in &[42u64, 1042, 2042] {
            let mut rng = StdRng::seed_from_u64(seed);
            let init = InitConfig::phi(h);
            let mut net = build_network(&init, &mut rng);
            let sdr = SdrTable::new(
                VOCAB, h, init.input_end(), SDR_ACTIVE_PCT,
                &mut StdRng::seed_from_u64(seed + 100),
            ).unwrap();
            let mut proj = Int8Projection::new(init.phi_dim, VOCAB, &mut rng);
            let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
            let evo_config = EvolutionConfig { edge_cap: init.edge_cap(), accept_ties: false };
            let prop_cfg = init.propagation.clone();
            let output_start = init.output_start();
            let neuron_count = init.neuron_count;

            let init_edges = net.edge_count();

            // Train
            for _ in 0..TRAIN_STEPS {
                evolution_step(
                    &mut net, &mut proj, &mut rng, &mut eval_rng,
                    |net, proj, eval_rng| {
                        eval_smooth(net, proj, &corpus, 20, eval_rng,
                            &sdr, &prop_cfg, output_start, neuron_count, &bigram)
                    },
                    &evo_config,
                );
            }

            // Eval trained network
            let acc_trained = eval_accuracy(
                &mut net, &proj, &corpus, EVAL_LEN, &mut eval_rng.clone(),
                &sdr, &prop_cfg, output_start, neuron_count,
            );
            let trained_edges = net.edge_count();

            // Eval EMPTY network (zero edges) with SAME trained projection
            let mut net_empty = Network::new(h);
            let acc_empty = eval_accuracy(
                &mut net_empty, &proj, &corpus, EVAL_LEN, &mut eval_rng.clone(),
                &sdr, &prop_cfg, output_start, neuron_count,
            );

            // Eval UNTRAINED network (fresh phi init) with SAME trained projection
            let mut net_fresh = build_network(&init, &mut StdRng::seed_from_u64(seed + 9999));
            let acc_fresh = eval_accuracy(
                &mut net_fresh, &proj, &corpus, EVAL_LEN, &mut eval_rng.clone(),
                &sdr, &prop_cfg, output_start, neuron_count,
            );
            let fresh_edges = net_fresh.edge_count();

            // Eval trained network with FRESH (untrained) projection
            let fresh_proj = Int8Projection::new(init.phi_dim, VOCAB, &mut StdRng::seed_from_u64(seed + 8888));
            let acc_fresh_proj = eval_accuracy(
                &mut net, &fresh_proj, &corpus, EVAL_LEN, &mut eval_rng.clone(),
                &sdr, &prop_cfg, output_start, neuron_count,
            );

            let diff = (acc_trained - acc_empty) * 100.0;
            println!("  seed {}:", seed);
            println!("    trained net + trained proj:  {:.1}% ({} edges, init={})", acc_trained * 100.0, trained_edges, init_edges);
            println!("    EMPTY net   + trained proj:  {:.1}% (0 edges)", acc_empty * 100.0);
            println!("    fresh net   + trained proj:  {:.1}% ({} edges)", acc_fresh * 100.0, fresh_edges);
            println!("    trained net + FRESH proj:    {:.1}%", acc_fresh_proj * 100.0);
            println!("    edge contribution: {:+.1}pp", diff);
        }
        println!();
    }
}
