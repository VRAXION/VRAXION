//! Paired mutation test: every topology mutation is ALWAYS paired with a W mutation.
//!
//! Tests the hypothesis that co-evolving topology + W together (accepted/rejected
//! as a unit) breaks the deadlock where W blocks topology evolution.
//!
//! Variants:
//!   A: topology only (90%) + W only (10%) — current library schedule (control)
//!   B: topology + 1 W mutation always paired
//!   C: topology + 2 W mutations always paired
//!   D: topology + 5 W mutations always paired
//!   E: W only (projection mutations, no topology) — to see W ceiling alone
//!
//! Run: cargo run --example paired_mutation_test --release -- <corpus-path>

use instnct_core::{load_corpus,
    build_network, eval_accuracy, EvolutionConfig, evolution_step, InitConfig, Int8Projection,
    Network, SdrTable, StepOutcome,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const STEPS: usize = 30_000;


fn apply_topology_mutation(net: &mut Network, rng: &mut impl Rng) -> bool {
    let roll = rng.gen_range(0..100u32);
    match roll {
        0..28 => net.mutate_add_edge(rng),
        28..44 => net.mutate_remove_edge(rng),
        44..55 => net.mutate_rewire(rng),
        55..72 => net.mutate_reverse(rng),
        72..80 => net.mutate_mirror(rng),
        80..88 => net.mutate_enhance(rng),
        88..94 => net.mutate_theta(rng),
        _ => net.mutate_channel(rng),
    }
}

#[derive(Clone, Copy)]
enum Variant {
    LibraryControl,      // A: current library schedule (90% topo, 10% W)
    PairedW1,            // B: topology + 1 W per step
    PairedW2,            // C: topology + 2 W per step
    PairedW5,            // D: topology + 5 W per step
    WOnly,               // E: only W mutations, no topology
}

impl Variant {
    fn name(&self) -> &str {
        match self {
            Self::LibraryControl => "library_ctrl",
            Self::PairedW1 => "topo+1W",
            Self::PairedW2 => "topo+2W",
            Self::PairedW5 => "topo+5W",
            Self::WOnly => "W_only",
        }
    }
}

struct Config {
    variant: Variant,
    seed: u64,
}

#[allow(dead_code)]
struct RunResult {
    variant_name: String,
    seed: u64,
    final_acc: f64,
    peak_acc: f64,
    final_edges: usize,
    accept_rate: f64,
}

fn run_one(cfg: &Config, corpus: &[u8]) -> RunResult {
    let init = InitConfig::phi(256);
    let evo = EvolutionConfig { edge_cap: init.edge_cap(), accept_ties: false };

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut net = build_network(&init, &mut rng);
    let mut proj = Int8Projection::new(init.phi_dim, CHARS,
        &mut StdRng::seed_from_u64(cfg.seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed + 1000);
    let sdr = SdrTable::new(CHARS, init.neuron_count, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(cfg.seed + 100)).unwrap();

    let mut peak = 0.0f64;
    let mut accepted = 0u32;
    let mut total = 0u32;

    for step in 0..STEPS {
        match cfg.variant {
            Variant::LibraryControl => {
                // Use library evolution_step (current default)
                let outcome = evolution_step(
                    &mut net, &mut proj, &mut rng, &mut eval_rng,
                    |n, p, e| eval_accuracy(n, p, corpus, 100, e, &sdr, &init.propagation, init.output_start(), init.neuron_count),
                    &evo,
                );
                match outcome {
                    StepOutcome::Accepted => { accepted += 1; total += 1; }
                    StepOutcome::Rejected => { total += 1; }
                    StepOutcome::Skipped => {}
                }
            }
            Variant::PairedW1 | Variant::PairedW2 | Variant::PairedW5 => {
                let w_count = match cfg.variant {
                    Variant::PairedW1 => 1,
                    Variant::PairedW2 => 2,
                    Variant::PairedW5 => 5,
                    _ => unreachable!(),
                };

                // Paired eval
                let snap = eval_rng.clone();
                let before = eval_accuracy(&mut net, &proj, corpus, 100, &mut eval_rng, &sdr, &init.propagation, init.output_start(), init.neuron_count);
                eval_rng = snap;

                let net_state = net.save_state();
                let proj_backup = proj.clone();
                let edges_before = net.edge_count();

                // Topology mutation
                let topo_ok = apply_topology_mutation(&mut net, &mut rng);

                // ALWAYS also mutate W (paired!)
                for _ in 0..w_count {
                    proj.mutate_one(&mut rng);
                }

                if !topo_ok {
                    // Topology failed but W changed — still eval
                    // (W-only change is valid)
                }

                total += 1;

                // Edge cap
                let edge_grew = net.edge_count() > edges_before;
                let within_cap = !edge_grew || net.edge_count() <= evo.edge_cap;

                let after = eval_accuracy(&mut net, &proj, corpus, 100, &mut eval_rng, &sdr, &init.propagation, init.output_start(), init.neuron_count);

                if after > before && within_cap {
                    accepted += 1;
                } else {
                    net.restore_state(&net_state);
                    proj = proj_backup;
                }
            }
            Variant::WOnly => {
                // Paired eval
                let snap = eval_rng.clone();
                let before = eval_accuracy(&mut net, &proj, corpus, 100, &mut eval_rng, &sdr, &init.propagation, init.output_start(), init.neuron_count);
                eval_rng = snap;

                let proj_backup = proj.clone();
                proj.mutate_one(&mut rng);
                total += 1;

                let after = eval_accuracy(&mut net, &proj, corpus, 100, &mut eval_rng, &sdr, &init.propagation, init.output_start(), init.neuron_count);

                if after > before {
                    accepted += 1;
                } else {
                    proj = proj_backup;
                }
            }
        }

        if (step + 1) % 10_000 == 0 {
            let mut cr = StdRng::seed_from_u64(cfg.seed + 6000 + step as u64);
            let acc = eval_accuracy(&mut net, &proj, corpus, 2000, &mut cr, &sdr, &init.propagation, init.output_start(), init.neuron_count);
            if acc > peak { peak = acc; }
            let rate = if total > 0 { accepted as f64 / total as f64 * 100.0 } else { 0.0 };
            println!("  {} seed={} step {:>5}: {:.1}% edges={} accept={:.1}%",
                cfg.variant.name(), cfg.seed, step + 1, acc * 100.0, net.edge_count(), rate);
        }
    }

    let mut fr = StdRng::seed_from_u64(cfg.seed + 9999);
    let final_acc = eval_accuracy(&mut net, &proj, corpus, 5000, &mut fr, &sdr, &init.propagation, init.output_start(), init.neuron_count);
    if final_acc > peak { peak = final_acc; }

    let rate = if total > 0 { accepted as f64 / total as f64 * 100.0 } else { 0.0 };
    println!("  {} seed={} FINAL: {:.1}% peak={:.1}% edges={} accept={:.1}%",
        cfg.variant.name(), cfg.seed, final_acc * 100.0, peak * 100.0,
        net.edge_count(), rate);

    RunResult {
        variant_name: cfg.variant.name().to_string(),
        seed: cfg.seed, final_acc, peak_acc: peak,
        final_edges: net.edge_count(), accept_rate: rate,
    }
}

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });

    println!("=== Paired Mutation Test ===");
    println!("  Does always pairing topology + W mutations break the co-evolution deadlock?\n");

    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars\n", corpus.len());

    let seeds = [42u64, 123, 7, 1042, 555, 8042];
    let variants = [
        Variant::LibraryControl,
        Variant::PairedW1,
        Variant::PairedW2,
        Variant::PairedW5,
        Variant::WOnly,
    ];

    let mut configs: Vec<Config> = Vec::new();
    for &v in &variants {
        for &s in &seeds {
            configs.push(Config { variant: v, seed: s });
        }
    }

    println!("  {} configs: {} variants × {} seeds\n", configs.len(), variants.len(), seeds.len());

    let start = Instant::now();

    let results: Vec<RunResult> = configs.par_iter()
        .map(|cfg| run_one(cfg, &corpus))
        .collect();

    let elapsed = start.elapsed().as_secs_f64();

    println!("\n=== SUMMARY ===\n");
    println!("{:<15} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "Variant", "Mean%", "Best%", "Peak%", "Edges", "Accept");
    println!("{}", "-".repeat(57));

    for v in &variants {
        let g: Vec<_> = results.iter().filter(|r| r.variant_name == v.name()).collect();
        let n = g.len() as f64;
        let mean = g.iter().map(|r| r.final_acc).sum::<f64>() / n;
        let best = g.iter().map(|r| r.final_acc).fold(0.0f64, f64::max);
        let peak = g.iter().map(|r| r.peak_acc).fold(0.0f64, f64::max);
        let edges = g.iter().map(|r| r.final_edges).sum::<usize>() / g.len();
        let accept = g.iter().map(|r| r.accept_rate).sum::<f64>() / n;
        println!("{:<15} {:>6.1}% {:>6.1}% {:>6.1}% {:>7} {:>6.1}%",
            v.name(), mean * 100.0, best * 100.0, peak * 100.0, edges, accept);
    }

    println!("\nPer-seed:");
    println!("{:<15} {:>6} {:>7} {:>7} {:>7} {:>7}",
        "Variant", "Seed", "Final%", "Peak%", "Edges", "Accept");
    println!("{}", "-".repeat(57));
    for r in &results {
        println!("{:<15} {:>6} {:>6.1}% {:>6.1}% {:>7} {:>6.1}%",
            r.variant_name, r.seed, r.final_acc * 100.0, r.peak_acc * 100.0,
            r.final_edges, r.accept_rate);
    }

    println!("\n  Total time: {:.0}s", elapsed);
}
