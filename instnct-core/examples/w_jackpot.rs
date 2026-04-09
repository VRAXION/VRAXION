//! W Jackpot: evolve ONLY the W projection with jackpot-style selection.
//!
//! Instead of 1 W mutation per step (1+1 ES), try N W mutations,
//! evaluate each, keep the BEST. This is the "jackpot" mechanism.
//!
//! Also tests: W-only (no topology mutation) vs W+topology jackpot.
//! Also tests: different N values (1, 5, 12, 25, 50).
//!
//! Run: cargo run --example w_jackpot --release -- <corpus-path>

use instnct_core::{load_corpus,
    build_network, eval_accuracy, InitConfig, Int8Projection, Network, SdrTable,
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
enum Mode {
    WOnlyBurst(usize),       // only W mutations, pick best of N
    WJackpot(usize),         // N candidates: each = 1 topo + 1 W, pick best
    TopoThenWJackpot(usize), // 1 topo mutation, then N W candidates, pick best W
    LibraryControl,           // standard 1+1 ES (library schedule)
}

impl Mode {
    fn name(&self) -> String {
        match self {
            Self::WOnlyBurst(n) => format!("W_only_N{}", n),
            Self::WJackpot(n) => format!("W+topo_jack_N{}", n),
            Self::TopoThenWJackpot(n) => format!("topo1_Wjack_N{}", n),
            Self::LibraryControl => "library_1+1".into(),
        }
    }
}

struct Config {
    mode: Mode,
    seed: u64,
}

#[allow(dead_code)]
struct RunResult {
    mode_name: String,
    seed: u64,
    final_acc: f64,
    peak_acc: f64,
    final_edges: usize,
    accepted: u32,
}

fn run_one(cfg: &Config, corpus: &[u8]) -> RunResult {
    let init = InitConfig::phi(256);
    let edge_cap = init.edge_cap();

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut net = build_network(&init, &mut rng);
    let mut proj = Int8Projection::new(init.phi_dim, CHARS,
        &mut StdRng::seed_from_u64(cfg.seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed + 1000);
    let sdr = SdrTable::new(CHARS, init.neuron_count, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(cfg.seed + 100)).unwrap();

    let mut peak = 0.0f64;
    let mut accepted = 0u32;

    for step in 0..STEPS {
        // Paired eval — get baseline
        let snap = eval_rng.clone();
        let before = eval_accuracy(&mut net, &proj, corpus, 100, &mut eval_rng, &sdr, &init.propagation, init.output_start(), init.neuron_count);
        eval_rng = snap;

        match cfg.mode {
            Mode::WOnlyBurst(n) => {
                // Try N different W mutations, keep the best one
                let mut best_proj = proj.clone();
                let mut best_score = before;

                for _ in 0..n {
                    let mut candidate = proj.clone();
                    candidate.mutate_one(&mut rng);

                    let eval_snap = eval_rng.clone();
                    let score = eval_accuracy(&mut net, &candidate, corpus, 100,
                        &mut eval_rng, &sdr, &init.propagation, init.output_start(), init.neuron_count);
                    eval_rng = eval_snap;

                    if score > best_score {
                        best_score = score;
                        best_proj = candidate;
                    }
                }

                // Advance eval_rng once for the "real" step
                let _ = eval_accuracy(&mut net, &best_proj, corpus, 100,
                    &mut eval_rng, &sdr, &init.propagation, init.output_start(), init.neuron_count);

                if best_score > before {
                    proj = best_proj;
                    accepted += 1;
                }
            }

            Mode::WJackpot(n) => {
                // N candidates: each does 1 topo mutation + 1 W mutation
                // Pick the best overall
                let net_state = net.save_state();
                let mut best_proj = proj.clone();
                let mut best_net_state = net.save_state();
                let mut best_score = before;
                let mut best_edges = net.edge_count();

                for _ in 0..n {
                    // Reset to pre-mutation state
                    net.restore_state(&net_state);
                    let mut candidate_proj = proj.clone();

                    // Topo mutation
                    apply_topology_mutation(&mut net, &mut rng);
                    // W mutation
                    candidate_proj.mutate_one(&mut rng);

                    let eval_snap = eval_rng.clone();
                    let score = eval_accuracy(&mut net, &candidate_proj, corpus, 100,
                        &mut eval_rng, &sdr, &init.propagation, init.output_start(), init.neuron_count);
                    eval_rng = eval_snap;

                    // Edge cap check
                    let within_cap = net.edge_count() <= edge_cap;

                    if score > best_score && within_cap {
                        best_score = score;
                        best_proj = candidate_proj;
                        best_net_state = net.save_state();
                        best_edges = net.edge_count();
                    }
                }

                // Advance eval_rng
                net.restore_state(&net_state);
                let _ = eval_accuracy(&mut net, &proj, corpus, 100, &mut eval_rng, &sdr, &init.propagation, init.output_start(), init.neuron_count);

                if best_score > before {
                    net.restore_state(&best_net_state);
                    proj = best_proj;
                    accepted += 1;
                    let _ = best_edges;
                } else {
                    net.restore_state(&net_state);
                }
            }

            Mode::TopoThenWJackpot(n) => {
                // 1 topology mutation (accept if better), then N W candidates
                let net_state = net.save_state();
                let edges_before = net.edge_count();

                // Topo step
                let topo_ok = apply_topology_mutation(&mut net, &mut rng);
                let edge_grew = net.edge_count() > edges_before;
                let within_cap = !edge_grew || net.edge_count() <= edge_cap;

                if topo_ok && within_cap {
                    let topo_score_snap = eval_rng.clone();
                    let topo_score = eval_accuracy(&mut net, &proj, corpus, 100,
                        &mut eval_rng, &sdr, &init.propagation, init.output_start(), init.neuron_count);
                    eval_rng = topo_score_snap;

                    if topo_score <= before {
                        net.restore_state(&net_state);
                    }
                } else if topo_ok {
                    net.restore_state(&net_state);
                }

                // W jackpot: try N W mutations on current network state
                let mut best_proj = proj.clone();
                let mut best_score = before;

                for _ in 0..n {
                    let mut candidate = proj.clone();
                    candidate.mutate_one(&mut rng);

                    let eval_snap = eval_rng.clone();
                    let score = eval_accuracy(&mut net, &candidate, corpus, 100,
                        &mut eval_rng, &sdr, &init.propagation, init.output_start(), init.neuron_count);
                    eval_rng = eval_snap;

                    if score > best_score {
                        best_score = score;
                        best_proj = candidate;
                    }
                }

                let _ = eval_accuracy(&mut net, &best_proj, corpus, 100,
                    &mut eval_rng, &sdr, &init.propagation, init.output_start(), init.neuron_count);

                if best_score > before {
                    proj = best_proj;
                    accepted += 1;
                }
            }

            Mode::LibraryControl => {
                let net_state = net.save_state();
                let proj_backup = proj.clone();
                let edges_before = net.edge_count();

                // Library schedule mutation
                let roll = rng.gen_range(0..100u32);
                let mutated = match roll {
                    0..25 => net.mutate_add_edge(&mut rng),
                    25..40 => net.mutate_remove_edge(&mut rng),
                    40..50 => net.mutate_rewire(&mut rng),
                    50..65 => net.mutate_reverse(&mut rng),
                    65..72 => net.mutate_mirror(&mut rng),
                    72..80 => net.mutate_enhance(&mut rng),
                    80..85 => net.mutate_theta(&mut rng),
                    85..90 => net.mutate_channel(&mut rng),
                    _ => { proj.mutate_one(&mut rng); true }
                };

                if !mutated {
                    let _ = eval_accuracy(&mut net, &proj, corpus, 100,
                        &mut eval_rng, &sdr, &init.propagation, init.output_start(), init.neuron_count);
                    continue;
                }

                let edge_grew = net.edge_count() > edges_before;
                let within_cap = !edge_grew || net.edge_count() <= edge_cap;
                let after = eval_accuracy(&mut net, &proj, corpus, 100,
                    &mut eval_rng, &sdr, &init.propagation, init.output_start(), init.neuron_count);

                if after > before && within_cap {
                    accepted += 1;
                } else {
                    net.restore_state(&net_state);
                    proj = proj_backup;
                }
            }
        }

        if (step + 1) % 10_000 == 0 {
            let mut cr = StdRng::seed_from_u64(cfg.seed + 6000 + step as u64);
            let acc = eval_accuracy(&mut net, &proj, corpus, 2000, &mut cr, &sdr, &init.propagation, init.output_start(), init.neuron_count);
            if acc > peak { peak = acc; }
            println!("  {} seed={} step {:>5}: {:.1}% edges={} accepted={}",
                cfg.mode.name(), cfg.seed, step + 1, acc * 100.0,
                net.edge_count(), accepted);
        }
    }

    let mut fr = StdRng::seed_from_u64(cfg.seed + 9999);
    let final_acc = eval_accuracy(&mut net, &proj, corpus, 5000, &mut fr, &sdr, &init.propagation, init.output_start(), init.neuron_count);
    if final_acc > peak { peak = final_acc; }

    println!("  {} seed={} FINAL: {:.1}% peak={:.1}% edges={} accepted={}",
        cfg.mode.name(), cfg.seed, final_acc * 100.0, peak * 100.0,
        net.edge_count(), accepted);

    RunResult {
        mode_name: cfg.mode.name(), seed: cfg.seed, final_acc, peak_acc: peak,
        final_edges: net.edge_count(), accepted,
    }
}

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });

    println!("=== W Jackpot Experiment ===\n");

    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars\n", corpus.len());

    let seeds = [42u64, 123, 7, 1042, 555, 8042];

    let mut configs: Vec<Config> = Vec::new();

    // Control
    for &s in &seeds {
        configs.push(Config { mode: Mode::LibraryControl, seed: s });
    }

    // W-only jackpot: N = 1, 5, 12, 25, 50
    for &n in &[1usize, 5, 12, 25, 50] {
        for &s in &seeds {
            configs.push(Config { mode: Mode::WOnlyBurst(n), seed: s });
        }
    }

    // W+topo jackpot: N = 5, 12, 25
    for &n in &[5usize, 12, 25] {
        for &s in &seeds {
            configs.push(Config { mode: Mode::WJackpot(n), seed: s });
        }
    }

    // Topo-first then W jackpot: N = 5, 12, 25
    for &n in &[5usize, 12, 25] {
        for &s in &seeds {
            configs.push(Config { mode: Mode::TopoThenWJackpot(n), seed: s });
        }
    }

    println!("  {} configs total\n", configs.len());

    let start = Instant::now();

    let results: Vec<RunResult> = configs.par_iter()
        .map(|cfg| run_one(cfg, &corpus))
        .collect();

    let elapsed = start.elapsed().as_secs_f64();

    println!("\n=== SUMMARY ===\n");
    println!("{:<22} {:>7} {:>7} {:>7} {:>7} {:>8}",
        "Mode", "Mean%", "Best%", "Peak%", "Edges", "Accepted");
    println!("{}", "-".repeat(62));

    let mut labels: Vec<String> = results.iter().map(|r| r.mode_name.clone()).collect();
    labels.sort();
    labels.dedup();

    for label in &labels {
        let g: Vec<_> = results.iter().filter(|r| r.mode_name == *label).collect();
        let n = g.len() as f64;
        let mean = g.iter().map(|r| r.final_acc).sum::<f64>() / n;
        let best = g.iter().map(|r| r.final_acc).fold(0.0f64, f64::max);
        let peak = g.iter().map(|r| r.peak_acc).fold(0.0f64, f64::max);
        let edges = g.iter().map(|r| r.final_edges).sum::<usize>() / g.len();
        let acc_count = g.iter().map(|r| r.accepted as f64).sum::<f64>() / n;
        println!("{:<22} {:>6.1}% {:>6.1}% {:>6.1}% {:>7} {:>8.0}",
            label, mean * 100.0, best * 100.0, peak * 100.0, edges, acc_count);
    }

    println!("\n  Total time: {:.0}s", elapsed);
}
