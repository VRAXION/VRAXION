//! Highway init sweep: structured I/O paths vs random init at various network sizes.
//!
//! Tests the hypothesis that pre-built I→O highways make the search space
//! more navigable, especially at larger H where random init is sparser.
//!
//! Run: cargo run --example highway_sweep --release

use instnct_core::{
    evolution_step, EvolutionConfig, Int8Projection, Network, PropagationConfig, SdrTable,
    StepOutcome,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs;

const PHI: f64 = 1.618033988749895;
const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;

/// Derived dimensions for a given H.
struct Dims {
    neuron_count: usize,    // H
    phi_dim: usize,         // round(H / phi)
    input_end: usize,       // = phi_dim
    output_start: usize,    // = H - phi_dim
    edge_cap: usize,        // H² * 7%
}

impl Dims {
    fn new(h: usize) -> Self {
        let phi_dim = (h as f64 / PHI).round() as usize;
        Self {
            neuron_count: h,
            phi_dim,
            input_end: phi_dim,
            output_start: h - phi_dim,
            edge_cap: h * h * 7 / 100,
        }
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
enum HighwayStrategy {
    /// No highways — pure random init (baseline).
    Random,
    /// Direct edges: random pure_input → random pure_output.
    Direct { count: usize },
    /// 2-hop through overlap hub: input→overlap, overlap→output.
    OverlapHub { count: usize },
    /// Multi-hop chain: input → hub₁ → hub₂ → output.
    Chain { count: usize },
    /// Forest: T evenly-spaced trunks (3-hop I→overlap→overlap→O) + cross-connections.
    Forest { trunks: usize, cross_connections: usize },
    /// Forest + trunk edge protection (accepted mutations that remove trunk edges are rolled back).
    ForestProtected { trunks: usize, cross_connections: usize },
}

impl HighwayStrategy {
    fn label(&self) -> String {
        match self {
            Self::Random => "random".to_string(),
            Self::Direct { count } => format!("direct-{count}"),
            Self::OverlapHub { count } => format!("hub-{count}"),
            Self::Chain { count } => format!("chain-{count}"),
            Self::Forest { trunks, .. } => format!("forest-{trunks}"),
            Self::ForestProtected { trunks, .. } => format!("forestP-{trunks}"),
        }
    }
}

fn load_corpus(path: &str) -> Vec<u8> {
    let raw = fs::read(path).expect("cannot read corpus");
    raw.iter()
        .filter_map(|&b| {
            if b.is_ascii_lowercase() { Some(b - b'a') }
            else if b.is_ascii_uppercase() { Some(b.to_ascii_lowercase() - b'a') }
            else if b == b' ' || b == b'\n' || b == b'\t' { Some(26) }
            else { None }
        })
        .collect()
}

fn build_network_with_highways(
    dims: &Dims,
    strategy: HighwayStrategy,
    rng: &mut StdRng,
) -> (Network, HashSet<(u16, u16)>) {
    let mut net = Network::new(dims.neuron_count);
    let mut protected = HashSet::new();

    // Phase 1: add highways
    match strategy {
        HighwayStrategy::Random => {} // no highways
        HighwayStrategy::Direct { count } => {
            // Direct: pure_input(0..output_start) → pure_output(input_end..H)
            for _ in 0..count {
                let src = rng.gen_range(0..dims.output_start) as u16;
                let tgt = rng.gen_range(dims.input_end..dims.neuron_count) as u16;
                net.graph_mut().add_edge(src, tgt);
            }
        }
        HighwayStrategy::OverlapHub { count } => {
            // 2-hop: pure_input → overlap_hub, overlap_hub → pure_output
            let overlap_start = dims.output_start;
            let overlap_end = dims.input_end;
            if overlap_end > overlap_start {
                for _ in 0..count {
                    let hub = rng.gen_range(overlap_start..overlap_end) as u16;
                    let src = rng.gen_range(0..overlap_start) as u16;
                    let tgt = rng.gen_range(overlap_end..dims.neuron_count) as u16;
                    net.graph_mut().add_edge(src, hub);
                    net.graph_mut().add_edge(hub, tgt);
                }
            }
        }
        HighwayStrategy::Chain { count } => {
            // 3-hop: input → hub₁ (overlap low) → hub₂ (overlap high) → output
            let overlap_start = dims.output_start;
            let overlap_end = dims.input_end;
            let overlap_mid = (overlap_start + overlap_end) / 2;
            if overlap_end > overlap_start + 1 {
                for _ in 0..count {
                    let src = rng.gen_range(0..overlap_start) as u16;
                    let hub1 = rng.gen_range(overlap_start..overlap_mid) as u16;
                    let hub2 = rng.gen_range(overlap_mid..overlap_end) as u16;
                    let tgt = rng.gen_range(overlap_end..dims.neuron_count) as u16;
                    net.graph_mut().add_edge(src, hub1);
                    net.graph_mut().add_edge(hub1, hub2);
                    net.graph_mut().add_edge(hub2, tgt);
                }
            }
        }
        HighwayStrategy::Forest { trunks, cross_connections }
        | HighwayStrategy::ForestProtected { trunks, cross_connections } => {
            let track = matches!(strategy, HighwayStrategy::ForestProtected { .. });
            let overlap_start = dims.output_start;
            let overlap_end = dims.input_end;
            let overlap_mid = (overlap_start + overlap_end) / 2;
            let pure_out_len = dims.neuron_count - overlap_end;

            // Build trunks: evenly-spaced input → random overlap hubs → evenly-spaced output
            let mut trunk_nodes: Vec<[u16; 4]> = Vec::with_capacity(trunks);
            for t in 0..trunks {
                let src = (t * overlap_start / trunks) as u16;
                let hub1 = rng.gen_range(overlap_start..overlap_mid) as u16;
                let hub2 = rng.gen_range(overlap_mid..overlap_end) as u16;
                let tgt = (overlap_end + t * pure_out_len / trunks.max(1)) as u16;

                for &(s, d) in &[(src, hub1), (hub1, hub2), (hub2, tgt)] {
                    if net.graph_mut().add_edge(s, d) && track {
                        protected.insert((s, d));
                    }
                }
                trunk_nodes.push([src, hub1, hub2, tgt]);
            }

            // Cross-connections between hub nodes of different trunks
            for _ in 0..cross_connections {
                let t1 = rng.gen_range(0..trunks);
                let t2 = if trunks > 1 {
                    (t1 + rng.gen_range(1..trunks)) % trunks
                } else {
                    t1
                };
                let node_a = trunk_nodes[t1][rng.gen_range(1..3_usize)];
                let node_b = trunk_nodes[t2][rng.gen_range(1..3_usize)];
                if net.graph_mut().add_edge(node_a, node_b) && track {
                    protected.insert((node_a, node_b));
                }
            }
        }
    }

    // Phase 2: fill remaining to target density (5%)
    let target_edges = dims.neuron_count * dims.neuron_count * 5 / 100;
    for _ in 0..target_edges * 3 {
        net.mutate_add_edge(rng);
        if net.edge_count() >= target_edges { break; }
    }

    // Phase 3: random parameters
    for i in 0..dims.neuron_count {
        net.threshold_mut()[i] = rng.gen_range(0..=7);
        net.channel_mut()[i] = rng.gen_range(1..=8);
        if rng.gen_ratio(1, 10) { net.polarity_mut()[i] = -1; }
    }

    (net, protected)
}

fn sample_eval_offset(corpus_len: usize, len: usize, rng: &mut StdRng) -> Option<usize> {
    if corpus_len <= len { return None; }
    let max_offset = corpus_len - len - 1;
    Some(rng.gen_range(0..=max_offset))
}

#[allow(clippy::too_many_arguments)]
fn eval_accuracy(
    net: &mut Network,
    projection: &Int8Projection,
    corpus: &[u8],
    len: usize,
    rng: &mut StdRng,
    sdr: &SdrTable,
    config: &PropagationConfig,
    dims: &Dims,
) -> f64 {
    let Some(off) = sample_eval_offset(corpus.len(), len, rng) else { return 0.0; };
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), config).unwrap();
        if projection.predict(&net.charge()[dims.output_start..dims.neuron_count]) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

struct RunConfig {
    h: usize,
    strategy: HighwayStrategy,
    seed: u64,
    steps: usize,
}

struct RunResult {
    h: usize,
    strategy_label: String,
    final_accuracy: f64,
    edge_count: usize,
}

fn run_one(cfg: &RunConfig, corpus: &[u8]) -> RunResult {
    let dims = Dims::new(cfg.h);
    let prop_config = PropagationConfig {
        ticks_per_token: 6,
        input_duration_ticks: 2,
        decay_interval_ticks: 6,
        use_refractory: false,
    };
    let evo_config = EvolutionConfig { edge_cap: dims.edge_cap, accept_ties: true };

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let (mut net, protected) = build_network_with_highways(&dims, cfg.strategy, &mut rng);
    let mut projection = Int8Projection::new(dims.phi_dim, CHARS, &mut StdRng::seed_from_u64(cfg.seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed + 1000);
    let sdr = SdrTable::new(CHARS, dims.neuron_count, dims.input_end, SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(cfg.seed + 100)).unwrap();

    let has_protection = !protected.is_empty();

    for _ in 0..cfg.steps {
        let pre_snapshot = if has_protection { Some(net.save_state()) } else { None };
        let proj_backup = if has_protection { Some(projection.clone()) } else { None };

        let outcome = evolution_step(
            &mut net, &mut projection, &mut rng, &mut eval_rng,
            |net, proj, eval_rng| eval_accuracy(net, proj, corpus, 100, eval_rng, &sdr, &prop_config, &dims),
            &evo_config,
        );

        if has_protection && outcome == StepOutcome::Accepted {
            let violated = protected.iter().any(|&(s, t)| !net.graph().has_edge(s, t));
            if violated {
                net.restore_state(pre_snapshot.as_ref().unwrap());
                projection = proj_backup.unwrap();
            }
        }
    }

    let final_acc = eval_accuracy(&mut net, &projection, corpus, 5000, &mut eval_rng, &sdr, &prop_config, &dims);
    let label = cfg.strategy.label();
    println!("  H={:<5} {:<14} seed={:<5} -> {:.1}%  edges={}", cfg.h, label, cfg.seed, final_acc * 100.0, net.edge_count());

    RunResult {
        h: cfg.h,
        strategy_label: label,
        final_accuracy: final_acc,
        edge_count: net.edge_count(),
    }
}

fn main() {
    let steps = 15000;
    let seeds = [42u64, 123, 7];

    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "S:/AI/work/VRAXION_DEV/instnct/data/traindat/fineweb_edu.traindat".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path);
    println!("  {} chars\n", corpus.len());

    // Build configs: H=256 forest topology experiment
    let mut configs = Vec::new();
    let h = 256;
    for &seed in &seeds {
        // Controls
        configs.push(RunConfig { h, strategy: HighwayStrategy::Random, seed, steps });
        configs.push(RunConfig { h, strategy: HighwayStrategy::Chain { count: 50 }, seed, steps });
        // Forest experiments: T trunks + T*2 cross-connections, with/without protection
        for &trunks in &[10, 20, 40] {
            let cross = trunks * 2;
            configs.push(RunConfig {
                h, strategy: HighwayStrategy::Forest { trunks, cross_connections: cross }, seed, steps,
            });
            configs.push(RunConfig {
                h, strategy: HighwayStrategy::ForestProtected { trunks, cross_connections: cross }, seed, steps,
            });
        }
    }

    println!("=== Forest Topology Sweep: {} configs, {steps} steps each ===\n", configs.len());

    let results: Vec<RunResult> = configs
        .par_iter()
        .map(|cfg| run_one(cfg, &corpus))
        .collect();

    // Summary: group by (H, strategy), average across seeds
    println!("\n=== SUMMARY (mean across {} seeds) ===\n", seeds.len());
    println!("{:<6} {:<16} {:>8} {:>8}", "H", "strategy", "mean%", "edges");
    println!("{}", "-".repeat(44));

    let mut seen = Vec::new();
    for r in &results {
        let key = (r.h, r.strategy_label.clone());
        if seen.contains(&key) { continue; }
        seen.push(key.clone());
        let group: Vec<_> = results.iter()
            .filter(|x| x.h == r.h && x.strategy_label == r.strategy_label)
            .collect();
        let mean_acc = group.iter().map(|x| x.final_accuracy).sum::<f64>() / group.len() as f64;
        let mean_edges = group.iter().map(|x| x.edge_count).sum::<usize>() / group.len();
        println!("{:<6} {:<16} {:>7.1}% {:>8}", r.h, r.strategy_label, mean_acc * 100.0, mean_edges);
    }
}
