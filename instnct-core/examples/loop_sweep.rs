//! Loop mutation sweep: test coordinated loop additions as mutation operators.
//!
//! Loop-N = add a circular path of N edges (A→B→C→...→A) as ONE mutation step.
//! This creates recurrent circuits that can sustain signal = memory for bigrams.
//!
//! Tests: loop-2 through loop-8, single add_edge control, mixed+loop variants.
//! Also tests: loop on empty network (0 init edges) vs standard init.
//!
//! Run: cargo run --example loop_sweep --release -- <corpus-path>

use instnct_core::{load_corpus, 
    build_network, InitConfig, Int8Projection, Network, PropagationConfig, SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const STEPS: usize = 30_000;
const EVAL_LEN_SHORT: usize = 100;
const EVAL_LEN_LONG: usize = 2000;
const LOG_INTERVAL: usize = 10_000;


// ---------------------------------------------------------------------------
// Loop mutations
// ---------------------------------------------------------------------------

/// Add a loop of size N: pick N random distinct neurons, add edges A→B→C→...→A.
/// Returns (success, edges_actually_added).
fn mutate_add_loop(net: &mut Network, loop_size: usize, rng: &mut impl Rng) -> (bool, usize) {
    let h = net.neuron_count();
    if h < loop_size { return (false, 0); }

    // Pick N distinct random neurons
    let mut neurons: Vec<u16> = Vec::with_capacity(loop_size);
    for _ in 0..loop_size * 10 { // retry limit
        let n = rng.gen_range(0..h) as u16;
        if !neurons.contains(&n) {
            neurons.push(n);
            if neurons.len() == loop_size { break; }
        }
    }
    if neurons.len() < loop_size { return (false, 0); }

    // Add edges forming the loop: 0→1→2→...→(N-1)→0
    let mut added = 0usize;
    for i in 0..loop_size {
        let src = neurons[i];
        let tgt = neurons[(i + 1) % loop_size];
        if net.graph_mut().add_edge(src, tgt) {
            added += 1;
        }
    }
    (added > 0, added)
}

/// Standard mixed mutation (library schedule) - for control/comparison.
fn apply_mixed_mutation(net: &mut Network, proj: &mut Int8Projection, rng: &mut impl Rng) -> bool {
    let roll = rng.gen_range(0..100u32);
    match roll {
        0..25 => net.mutate_add_edge(rng),
        25..40 => net.mutate_remove_edge(rng),
        40..50 => net.mutate_rewire(rng),
        50..65 => net.mutate_reverse(rng),
        65..72 => net.mutate_mirror(rng),
        72..80 => net.mutate_enhance(rng),
        80..85 => net.mutate_theta(rng),
        85..90 => net.mutate_channel(rng),
        _ => { proj.mutate_one(rng); true }
    }
}

/// Mixed mutation with loop additions replacing part of the add_edge budget.
/// 15% loop-N, 10% add_edge (was 25%), rest unchanged.
fn apply_mixed_with_loop(
    net: &mut Network, proj: &mut Int8Projection, rng: &mut impl Rng, loop_size: usize,
) -> bool {
    let roll = rng.gen_range(0..100u32);
    match roll {
        0..15 => { let (ok, _) = mutate_add_loop(net, loop_size, rng); ok }
        15..25 => net.mutate_add_edge(rng),
        25..40 => net.mutate_remove_edge(rng),
        40..50 => net.mutate_rewire(rng),
        50..65 => net.mutate_reverse(rng),
        65..72 => net.mutate_mirror(rng),
        72..80 => net.mutate_enhance(rng),
        80..85 => net.mutate_theta(rng),
        85..90 => net.mutate_channel(rng),
        _ => { proj.mutate_one(rng); true }
    }
}

// ---------------------------------------------------------------------------
// Eval
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn eval_accuracy(
    net: &mut Network, proj: &Int8Projection, corpus: &[u8], len: usize,
    rng: &mut StdRng, sdr: &SdrTable, prop: &PropagationConfig,
    output_start: usize, neuron_count: usize,
) -> f64 {
    if corpus.len() <= len { return 0.0; }
    let off = rng.gen_range(0..=corpus.len() - len - 1);
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), prop).unwrap();
        if proj.predict(&net.charge()[output_start..neuron_count]) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum MutationType {
    SingleAdd,             // control: single random edge
    LoopOnly(usize),       // only loop-N mutations
    MixedControl,          // standard library schedule
    MixedWithLoop(usize),  // library schedule but 15% replaced with loop-N
}

impl MutationType {
    fn name(&self) -> String {
        match self {
            Self::SingleAdd => "single_add".into(),
            Self::LoopOnly(n) => format!("loop_{}_only", n),
            Self::MixedControl => "mixed_ctrl".into(),
            Self::MixedWithLoop(n) => format!("mixed+loop{}", n),
        }
    }
}

#[derive(Clone)]
struct Config {
    mutation: MutationType,
    seed: u64,
    start_empty: bool, // if true, start from 0 edges (no init topology)
}

impl Config {
    fn label(&self) -> String {
        let empty_tag = if self.start_empty { "_empty" } else { "" };
        format!("{}{}", self.mutation.name(), empty_tag)
    }
}

#[allow(dead_code)]
struct RunResult {
    label: String,
    seed: u64,
    final_acc: f64,
    peak_acc: f64,
    final_edges: usize,
    loops_added: usize,
    accept_rate: f64,
}

fn run_one(cfg: &Config, corpus: &[u8]) -> RunResult {
    let init = InitConfig::phi(256);
    let edge_cap = init.edge_cap();
    let os = init.output_start();
    let nc = init.neuron_count;
    let prop = init.propagation;

    let mut rng = StdRng::seed_from_u64(cfg.seed);

    let mut net = if cfg.start_empty {
        // Empty network with random params only (no edges)
        let mut n = Network::new(nc);
        for i in 0..nc {
            n.threshold_mut()[i] = rng.gen_range(0..=7u8);
            n.channel_mut()[i] = rng.gen_range(1..=8u8);
            if rng.gen_ratio(1, 10) { n.polarity_mut()[i] = -1; }
        }
        n
    } else {
        build_network(&init, &mut rng)
    };

    let mut proj = Int8Projection::new(init.phi_dim, CHARS,
        &mut StdRng::seed_from_u64(cfg.seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed + 1000);
    let sdr = SdrTable::new(CHARS, nc, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(cfg.seed + 100)).unwrap();

    let mut peak_acc = 0.0f64;
    let mut accepted = 0u32;
    let mut total_tried = 0u32;
    let mut total_loops_added = 0usize;

    for step in 0..STEPS {
        // Paired eval
        let snap = eval_rng.clone();
        let before = eval_accuracy(&mut net, &proj, corpus, EVAL_LEN_SHORT,
            &mut eval_rng, &sdr, &prop, os, nc);
        eval_rng = snap;

        let net_state = net.save_state();
        let proj_backup = proj.clone();
        let edges_before = net.edge_count();

        // Apply mutation
        let mutated = match &cfg.mutation {
            MutationType::SingleAdd => net.mutate_add_edge(&mut rng),
            MutationType::LoopOnly(n) => {
                let (ok, added) = mutate_add_loop(&mut net, *n, &mut rng);
                if ok { total_loops_added += 1; }
                let _ = added;
                ok
            }
            MutationType::MixedControl => apply_mixed_mutation(&mut net, &mut proj, &mut rng),
            MutationType::MixedWithLoop(n) => {
                let is_loop_roll = rng.gen_range(0..100u32) < 15;
                if is_loop_roll {
                    // Undo the roll consumption for the mixed function
                    let (ok, _) = mutate_add_loop(&mut net, *n, &mut rng);
                    if ok { total_loops_added += 1; }
                    ok
                } else {
                    apply_mixed_with_loop(&mut net, &mut proj, &mut rng, *n)
                }
            }
        };

        if !mutated {
            let _ = eval_accuracy(&mut net, &proj, corpus, EVAL_LEN_SHORT,
                &mut eval_rng, &sdr, &prop, os, nc);
            continue;
        }

        total_tried += 1;

        // Edge cap check
        let edge_grew = net.edge_count() > edges_before;
        let within_cap = !edge_grew || net.edge_count() <= edge_cap;

        let after = eval_accuracy(&mut net, &proj, corpus, EVAL_LEN_SHORT,
            &mut eval_rng, &sdr, &prop, os, nc);

        if after > before && within_cap {
            accepted += 1;
            if after > peak_acc { peak_acc = after; }
        } else {
            net.restore_state(&net_state);
            proj = proj_backup;
        }

        // Log
        if (step + 1) % LOG_INTERVAL == 0 {
            let mut cr = StdRng::seed_from_u64(cfg.seed + 6000 + step as u64);
            let acc = eval_accuracy(&mut net, &proj, corpus, EVAL_LEN_LONG,
                &mut cr, &sdr, &prop, os, nc);
            let rate = if total_tried > 0 { accepted as f64 / total_tried as f64 * 100.0 } else { 0.0 };
            println!("  {} seed={} step {:>5}: {:.1}% edges={} accept={:.1}% loops={}",
                cfg.label(), cfg.seed, step + 1, acc * 100.0, net.edge_count(), rate, total_loops_added);
        }
    }

    // Final eval
    let mut fr = StdRng::seed_from_u64(cfg.seed + 9999);
    let final_acc = eval_accuracy(&mut net, &proj, corpus, EVAL_LEN_LONG * 2,
        &mut fr, &sdr, &prop, os, nc);
    if final_acc > peak_acc { peak_acc = final_acc; }

    let rate = if total_tried > 0 { accepted as f64 / total_tried as f64 * 100.0 } else { 0.0 };
    println!("  {} seed={} FINAL: {:.1}% peak={:.1}% edges={} accept={:.1}% loops={}",
        cfg.label(), cfg.seed, final_acc * 100.0, peak_acc * 100.0,
        net.edge_count(), rate, total_loops_added);

    RunResult {
        label: cfg.label(),
        seed: cfg.seed,
        final_acc,
        peak_acc,
        final_edges: net.edge_count(),
        loops_added: total_loops_added,
        accept_rate: rate,
    }
}

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });
    println!("=== Loop Mutation Sweep ===\n");

    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars\n", corpus.len());

    let seeds = [42u64, 123, 7, 1042, 555, 8042];

    let mut configs: Vec<Config> = Vec::new();

    // --- Standard init (chain-50 + 5% density) ---

    // Control: single add_edge only
    for &s in &seeds {
        configs.push(Config { mutation: MutationType::SingleAdd, seed: s, start_empty: false });
    }

    // Control: mixed schedule (library default)
    for &s in &seeds {
        configs.push(Config { mutation: MutationType::MixedControl, seed: s, start_empty: false });
    }

    // Loop-only: sizes 2-8
    for loop_size in 2..=8 {
        for &s in &seeds {
            configs.push(Config {
                mutation: MutationType::LoopOnly(loop_size), seed: s, start_empty: false,
            });
        }
    }

    // Mixed + loop: sizes 2-5 (15% loop, rest library schedule)
    for loop_size in 2..=5 {
        for &s in &seeds {
            configs.push(Config {
                mutation: MutationType::MixedWithLoop(loop_size), seed: s, start_empty: false,
            });
        }
    }

    // --- Empty init (0 edges) ---

    // Loop-only from empty: sizes 3-5
    for loop_size in 3..=5 {
        for &s in &seeds {
            configs.push(Config {
                mutation: MutationType::LoopOnly(loop_size), seed: s, start_empty: true,
            });
        }
    }

    // Mixed + loop from empty: sizes 3-5
    for loop_size in 3..=5 {
        for &s in &seeds {
            configs.push(Config {
                mutation: MutationType::MixedWithLoop(loop_size), seed: s, start_empty: true,
            });
        }
    }

    println!("  {} configs total ({} seeds each)\n", configs.len(), seeds.len());
    println!("  Standard init: single_add, mixed_ctrl, loop_2..8_only, mixed+loop2..5");
    println!("  Empty init:    loop_3..5_only_empty, mixed+loop3..5_empty\n");

    let results: Vec<RunResult> = configs.par_iter()
        .map(|cfg| run_one(cfg, &corpus))
        .collect();

    // --- Summary ---
    println!("\n=== SUMMARY ===\n");
    println!("{:<22} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "Config", "Mean%", "Best%", "Peak%", "Edges", "Accept", "Loops");
    println!("{}", "-".repeat(72));

    // Group by label
    let mut labels: Vec<String> = results.iter().map(|r| r.label.clone()).collect();
    labels.sort();
    labels.dedup();

    for label in &labels {
        let g: Vec<_> = results.iter().filter(|r| r.label == *label).collect();
        let n = g.len() as f64;
        let mean = g.iter().map(|r| r.final_acc).sum::<f64>() / n;
        let best = g.iter().map(|r| r.final_acc).fold(0.0f64, f64::max);
        let peak = g.iter().map(|r| r.peak_acc).fold(0.0f64, f64::max);
        let edges = g.iter().map(|r| r.final_edges).sum::<usize>() / g.len();
        let accept = g.iter().map(|r| r.accept_rate).sum::<f64>() / n;
        let loops = g.iter().map(|r| r.loops_added).sum::<usize>() / g.len();
        println!("{:<22} {:>6.1}% {:>6.1}% {:>6.1}% {:>7} {:>6.1}% {:>7}",
            label, mean * 100.0, best * 100.0, peak * 100.0, edges, accept, loops);
    }

    // Top 10 individual results
    let mut sorted: Vec<&RunResult> = results.iter().collect();
    sorted.sort_by(|a, b| b.final_acc.partial_cmp(&a.final_acc).unwrap());

    println!("\n=== TOP 15 INDIVIDUAL RESULTS ===\n");
    println!("{:<22} {:>6} {:>7} {:>7} {:>7} {:>7}",
        "Config", "Seed", "Final%", "Peak%", "Edges", "Loops");
    println!("{}", "-".repeat(62));
    for r in sorted.iter().take(15) {
        println!("{:<22} {:>6} {:>6.1}% {:>6.1}% {:>7} {:>7}",
            r.label, r.seed, r.final_acc * 100.0, r.peak_acc * 100.0,
            r.final_edges, r.loops_added);
    }
}
