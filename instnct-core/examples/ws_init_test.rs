//! Watts-Strogatz small-world initialization A/B test.
//!
//! Compares WS ring+rewire init against current chain-50 init.
//! Small-world graphs have high clustering + short path lengths.
//!
//! Run: cargo run --example ws_init_test --release -- <corpus-path>

use instnct_core::{load_corpus,
    build_network, eval_accuracy, evolution_step, InitConfig, Int8Projection, Network,
    SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const STEPS: usize = 30_000;


/// Build a Watts-Strogatz small-world network.
/// Start with a ring where each neuron connects to K nearest neighbors,
/// then rewire each edge with probability P.
fn build_ws_network(h: usize, k: usize, rewire_p: f64, rng: &mut StdRng) -> Network {
    let mut net = Network::new(h);

    // Phase 1: Ring lattice — each neuron connects to K/2 neighbors in each direction
    let half_k = k / 2;
    for i in 0..h {
        for offset in 1..=half_k {
            let j = (i + offset) % h;
            net.graph_mut().add_edge(i as u16, j as u16);
            net.graph_mut().add_edge(j as u16, i as u16); // bidirectional
        }
    }
    let ring_edges = net.edge_count();

    // Phase 2: Rewire each edge with probability P
    let edges: Vec<_> = net.graph().iter_edges().collect();
    let mut rewired = 0usize;
    for e in &edges {
        if rng.gen_bool(rewire_p) {
            let new_target = rng.gen_range(0..h) as u16;
            if new_target != e.source && new_target != e.target {
                net.graph_mut().remove_edge(e.source, e.target);
                if !net.graph_mut().add_edge(e.source, new_target) {
                    net.graph_mut().add_edge(e.source, e.target); // restore if dup
                } else {
                    rewired += 1;
                }
            }
        }
    }

    // Phase 3: Random params (same as build_network)
    for i in 0..h {
        net.threshold_mut()[i] = rng.gen_range(0..=7u32);
        net.channel_mut()[i] = rng.gen_range(1..=8u8);
        if rng.gen_ratio(1, 10) { net.polarity_mut()[i] = -1; }
    }

    println!("    WS(k={},p={:.2}): ring_edges={} rewired={} final_edges={}",
        k, rewire_p, ring_edges, rewired, net.edge_count());
    net
}

struct Config {
    label: String,
    init_type: InitType,
    seed: u64,
}

enum InitType {
    Chain50,
    WattsStrogatz { k: usize, rewire_p: f64 },
}

#[allow(dead_code)]
struct RunResult {
    label: String,
    seed: u64,
    final_acc: f64,
    peak_acc: f64,
    final_edges: usize,
}

fn run_one(cfg: &Config, corpus: &[u8]) -> RunResult {
    let init = InitConfig::phi(256);
    let evo = init.evolution_config();

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut net = match &cfg.init_type {
        InitType::Chain50 => build_network(&init, &mut rng),
        InitType::WattsStrogatz { k, rewire_p } => build_ws_network(256, *k, *rewire_p, &mut rng),
    };

    let mut proj = Int8Projection::new(init.phi_dim, CHARS,
        &mut StdRng::seed_from_u64(cfg.seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed + 1000);
    let sdr = SdrTable::new(CHARS, init.neuron_count, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(cfg.seed + 100)).unwrap();

    let mut peak = 0.0f64;

    for step in 0..STEPS {
        let _ = evolution_step(
            &mut net, &mut proj, &mut rng, &mut eval_rng,
            |n, p, e| eval_accuracy(n, p, corpus, 100, e, &sdr, &init.propagation, init.output_start(), init.neuron_count),
            &evo,
        );

        if (step + 1) % 10_000 == 0 {
            let mut cr = StdRng::seed_from_u64(cfg.seed + 6000 + step as u64);
            let acc = eval_accuracy(&mut net, &proj, corpus, 2000, &mut cr, &sdr, &init.propagation, init.output_start(), init.neuron_count);
            if acc > peak { peak = acc; }
            println!("  {} seed={:<4} step {:>5}: {:.1}%  edges={}",
                cfg.label, cfg.seed, step + 1, acc * 100.0, net.edge_count());
        }
    }

    let mut fr = StdRng::seed_from_u64(cfg.seed + 9999);
    let final_acc = eval_accuracy(&mut net, &proj, corpus, 5000, &mut fr, &sdr, &init.propagation, init.output_start(), init.neuron_count);
    if final_acc > peak { peak = final_acc; }
    println!("  {} seed={:<4} FINAL: {:.1}%  peak={:.1}%  edges={}",
        cfg.label, cfg.seed, final_acc * 100.0, peak * 100.0, net.edge_count());

    RunResult { label: cfg.label.clone(), seed: cfg.seed, final_acc, peak_acc: peak,
        final_edges: net.edge_count() }
}

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });
    println!("=== Watts-Strogatz Init A/B Test ===\n");

    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars\n", corpus.len());

    let seeds = [42u64, 123, 7];

    let mut configs: Vec<Config> = Vec::new();

    // Control: chain-50 (current best)
    for &s in &seeds {
        configs.push(Config { label: "chain-50".into(), init_type: InitType::Chain50, seed: s });
    }

    // WS variants: k=10,20,40 with p=0.1,0.3
    for &k in &[10usize, 20, 40] {
        for &p in &[0.1f64, 0.3] {
            for &s in &seeds {
                configs.push(Config {
                    label: format!("WS(k={},p={:.1})", k, p),
                    init_type: InitType::WattsStrogatz { k, rewire_p: p },
                    seed: s,
                });
            }
        }
    }

    println!("  {} configs: chain-50 × 3 seeds + WS variants × 3 seeds\n", configs.len());

    let results: Vec<RunResult> = configs.par_iter()
        .map(|cfg| run_one(cfg, &corpus))
        .collect();

    println!("\n=== SUMMARY ===\n");
    println!("{:<20} {:>8} {:>8} {:>8} {:>8}",
        "Init", "mean%", "best%", "peak%", "edges");
    println!("{}", "-".repeat(56));

    // Group by label
    let labels: Vec<String> = {
        let mut l: Vec<String> = results.iter().map(|r| r.label.clone()).collect();
        l.dedup();
        l.sort();
        l.dedup();
        l
    };

    for label in &labels {
        let g: Vec<_> = results.iter().filter(|r| r.label == *label).collect();
        let mean = g.iter().map(|r| r.final_acc).sum::<f64>() / g.len() as f64;
        let best = g.iter().map(|r| r.final_acc).fold(0.0f64, f64::max);
        let peak = g.iter().map(|r| r.peak_acc).fold(0.0f64, f64::max);
        let edges = g.iter().map(|r| r.final_edges).sum::<usize>() / g.len();
        println!("{:<20} {:>7.1}% {:>7.1}% {:>7.1}% {:>8}",
            label, mean * 100.0, best * 100.0, peak * 100.0, edges);
    }
}
