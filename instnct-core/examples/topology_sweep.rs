//! Spatial topology sweep: Grid vs Tree vs Ring vs Butterfly.
//!
//! Each topology constrains which edges are allowed. Evolution can only
//! add/rewire edges within the allowed set. Tests whether structured
//! topologies improve over random.
//!
//! Run: cargo run --example topology_sweep --release -- <corpus-path>

use instnct_core::{load_corpus, 
    InitConfig, Int8Projection, Network, PropagationConfig, SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;

// ---------------------------------------------------------------------------
// Topology constraints: which edges are allowed?
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Topology {
    Random,    // anything goes (baseline)
    Grid,      // 16×16 grid, connections at powers-of-2 distance
    Ring,      // circular, connections at ±1,±2,±4,...,±128
    Butterfly, // XOR connections: src→src⊕(2^k) for k=0..7, plus ≤2 bit diff
    Tree,      // layered: 8 layers of 32, connect to same/next layer
}

impl Topology {
    fn label(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Grid => "grid",
            Self::Ring => "ring",
            Self::Butterfly => "butterfly",
            Self::Tree => "tree",
        }
    }

    fn can_connect(self, src: usize, tgt: usize, h: usize) -> bool {
        if src == tgt { return false; }
        if src >= h || tgt >= h { return false; }
        match self {
            Self::Random => true,
            Self::Grid => grid_can_connect(src, tgt, h),
            Self::Ring => ring_can_connect(src, tgt, h),
            Self::Butterfly => butterfly_can_connect(src, tgt),
            Self::Tree => tree_can_connect(src, tgt, h),
        }
    }
}

/// Grid 16×16: allowed if Manhattan-component distances are powers of 2 or diagonals.
fn grid_can_connect(src: usize, tgt: usize, _h: usize) -> bool {
    let (sr, sc) = (src / 16, src % 16);
    let (tr, tc) = (tgt / 16, tgt % 16);
    let dr = (sr as isize - tr as isize).unsigned_abs();
    let dc = (sc as isize - tc as isize).unsigned_abs();

    // Same row, power-of-2 column distance
    if dr == 0 && dc > 0 && dc.is_power_of_two() && dc <= 8 { return true; }
    // Same column, power-of-2 row distance
    if dc == 0 && dr > 0 && dr.is_power_of_two() && dr <= 8 { return true; }
    // Diagonal ±1
    if dr <= 1 && dc <= 1 { return true; }
    false
}

/// Ring: circular distance ±1,±2,±4,±8,...,±128.
fn ring_can_connect(src: usize, tgt: usize, h: usize) -> bool {
    let dist = ((src as isize - tgt as isize).unsigned_abs()).min(
        ((src + h) as isize - tgt as isize).unsigned_abs().min(
            ((tgt + h) as isize - src as isize).unsigned_abs()
        )
    );
    dist > 0 && dist.is_power_of_two() && dist <= 128
}

/// Butterfly: src and tgt differ in at most 2 bits (covers 1-hop and 2-hop butterfly).
fn butterfly_can_connect(src: usize, tgt: usize) -> bool {
    let diff = src ^ tgt;
    let bits = diff.count_ones();
    (1..=2).contains(&bits)
}

/// Tree: 8 layers of 32 neurons. Connect within same layer or to adjacent layer.
fn tree_can_connect(src: usize, tgt: usize, h: usize) -> bool {
    let layer_size = h / 8;
    let src_layer = src / layer_size;
    let tgt_layer = tgt / layer_size;
    let layer_dist = (src_layer as isize - tgt_layer as isize).unsigned_abs();
    // Same layer or adjacent layer (±1), or skip-connect (±2)
    layer_dist <= 2
}

// ---------------------------------------------------------------------------
// Constrained mutation: add edge only if topology allows it
// ---------------------------------------------------------------------------

fn constrained_add_edge(net: &mut Network, topo: Topology, rng: &mut impl Rng) -> bool {
    let h = net.neuron_count();
    // Try up to 20 times to find an allowed edge
    for _ in 0..20 {
        let src = rng.gen_range(0..h);
        let tgt = rng.gen_range(0..h);
        if topo.can_connect(src, tgt, h) {
            return net.graph_mut().add_edge(src as u16, tgt as u16);
        }
    }
    false
}

fn constrained_rewire(net: &mut Network, topo: Topology, rng: &mut impl Rng) -> bool {
    let h = net.neuron_count();
    let edge_count = net.graph().edge_count();
    if edge_count == 0 || h < 2 { return false; }

    // Pick random existing edge
    let edges: Vec<_> = net.graph().iter_edges().collect();
    let edge = edges[rng.gen_range(0..edges.len())];

    // Try to find new allowed target
    for _ in 0..20 {
        let new_tgt = rng.gen_range(0..h);
        if topo.can_connect(edge.source as usize, new_tgt, h) {
            net.graph_mut().remove_edge(edge.source, edge.target);
            if net.graph_mut().add_edge(edge.source, new_tgt as u16) {
                return true;
            }
            net.graph_mut().add_edge(edge.source, edge.target);
            return false;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------


fn sample_eval_offset(corpus_len: usize, len: usize, rng: &mut StdRng) -> Option<usize> {
    if corpus_len <= len { return None; }
    Some(rng.gen_range(0..=corpus_len - len - 1))
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
    output_start: usize,
    neuron_count: usize,
) -> f64 {
    let Some(off) = sample_eval_offset(corpus.len(), len, rng) else { return 0.0; };
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), config).unwrap();
        if projection.predict(&net.charge()[output_start..neuron_count]) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

// ---------------------------------------------------------------------------
// Evolution with constrained topology
// ---------------------------------------------------------------------------

struct SweepConfig {
    topo: Topology,
    seed: u64,
    steps: usize,
}

#[allow(dead_code)]
struct SweepResult {
    label: &'static str,
    seed: u64,
    accuracy: f64,
    edge_count: usize,
}

fn run_one(cfg: &SweepConfig, corpus: &[u8]) -> SweepResult {
    let init = InitConfig::phi(256);
    let h = init.neuron_count;
    let output_start = init.output_start();
    let edge_cap = init.edge_cap();

    let mut rng = StdRng::seed_from_u64(cfg.seed);

    // Build network with constrained edges
    let mut net = Network::new(h);

    // Chain-50 highways (unconstrained, for fair I→O reachability)
    let overlap_start = init.output_start();
    let overlap_end = init.input_end();
    let overlap_mid = (overlap_start + overlap_end) / 2;
    for _ in 0..50 {
        let src = rng.gen_range(0..overlap_start) as u16;
        let hub1 = rng.gen_range(overlap_start..overlap_mid) as u16;
        let hub2 = rng.gen_range(overlap_mid..overlap_end) as u16;
        let tgt = rng.gen_range(overlap_end..h) as u16;
        net.graph_mut().add_edge(src, hub1);
        net.graph_mut().add_edge(hub1, hub2);
        net.graph_mut().add_edge(hub2, tgt);
    }

    // Fill to 5% with CONSTRAINED edges
    let target_edges = h * h * 5 / 100;
    for _ in 0..target_edges * 5 {
        constrained_add_edge(&mut net, cfg.topo, &mut rng);
        if net.edge_count() >= target_edges { break; }
    }

    // Random params
    for i in 0..h {
        net.threshold_mut()[i] = rng.gen_range(0..=7u32);
        net.channel_mut()[i] = rng.gen_range(1..=8u8);
        if rng.gen_ratio(1, 10) { net.polarity_mut()[i] = -1; }
    }

    let mut proj = Int8Projection::new(init.phi_dim, CHARS, &mut StdRng::seed_from_u64(cfg.seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed + 1000);
    let sdr = SdrTable::new(CHARS, h, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(cfg.seed + 100)).unwrap();

    // Evolution with constrained add/rewire
    for _ in 0..cfg.steps {
        let eval_snapshot = eval_rng.clone();
        let before = eval_accuracy(
            &mut net, &proj, corpus, 100, &mut eval_rng, &sdr,
            &init.propagation, output_start, h,
        );
        eval_rng = eval_snapshot;

        let snapshot = net.save_state();
        let mut weight_backup = None;

        let roll = rng.gen_range(0..100u32);
        let mutated = match roll {
            0..25 => constrained_add_edge(&mut net, cfg.topo, &mut rng),
            25..40 => net.mutate_remove_edge(&mut rng),
            40..50 => constrained_rewire(&mut net, cfg.topo, &mut rng),
            50..65 => net.mutate_reverse(&mut rng),
            65..72 => net.mutate_mirror(&mut rng),
            72..80 => net.mutate_enhance(&mut rng),
            80..85 => net.mutate_theta(&mut rng),
            85..90 => net.mutate_channel(&mut rng),
            _ => { weight_backup = Some(proj.mutate_one(&mut rng)); true }
        };

        if !mutated {
            let _ = eval_accuracy(
                &mut net, &proj, corpus, 100, &mut eval_rng, &sdr,
                &init.propagation, output_start, h,
            );
            continue;
        }

        let after = eval_accuracy(
            &mut net, &proj, corpus, 100, &mut eval_rng, &sdr,
            &init.propagation, output_start, h,
        );

        let accepted = if net.edge_count() < edge_cap { after >= before } else { after > before };
        if !accepted {
            net.restore_state(&snapshot);
            if let Some(backup) = weight_backup {
                proj.rollback(backup);
            }
        }
    }

    let final_acc = eval_accuracy(
        &mut net, &proj, corpus, 5000, &mut eval_rng, &sdr,
        &init.propagation, output_start, h,
    );

    println!("  {:<12} seed={:<5} -> {:.1}%  edges={}", cfg.topo.label(), cfg.seed, final_acc * 100.0, net.edge_count());

    SweepResult {
        label: cfg.topo.label(),
        seed: cfg.seed,
        accuracy: final_acc,
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
    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars\n", corpus.len());

    let topos = [Topology::Random, Topology::Grid, Topology::Ring, Topology::Butterfly, Topology::Tree];

    // Count allowed edges per topology
    let h = 256;
    for &topo in &topos {
        let mut allowed = 0u64;
        for src in 0..h {
            for tgt in 0..h {
                if topo.can_connect(src, tgt, h) { allowed += 1; }
            }
        }
        println!("  {:<12} allowed edges: {:>6} / {} ({:.1}%)",
            topo.label(), allowed, h * h, allowed as f64 / (h * h) as f64 * 100.0);
    }
    println!();

    let mut configs = Vec::new();
    for &topo in &topos {
        for &seed in &seeds {
            configs.push(SweepConfig { topo, seed, steps });
        }
    }

    println!("=== Topology Sweep: {} configs, {} steps ===\n", configs.len(), steps);

    let results: Vec<SweepResult> = configs
        .par_iter()
        .map(|cfg| run_one(cfg, &corpus))
        .collect();

    println!("\n=== SUMMARY (mean across {} seeds) ===\n", seeds.len());
    println!("{:<12} {:>8} {:>8}", "topology", "mean%", "edges");
    println!("{}", "-".repeat(32));

    let mut seen = Vec::new();
    for r in &results {
        if seen.contains(&r.label) { continue; }
        seen.push(r.label);
        let group: Vec<_> = results.iter().filter(|x| x.label == r.label).collect();
        let mean_acc = group.iter().map(|x| x.accuracy).sum::<f64>() / group.len() as f64;
        let mean_edges = group.iter().map(|x| x.edge_count).sum::<usize>() / group.len();
        println!("{:<12} {:>7.1}% {:>8}", r.label, mean_acc * 100.0, mean_edges);
    }
}
