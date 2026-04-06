//! Pocket evolve: 5 pairs x 2 individuals = 10 independent evolution runs.
//!
//! Pure evolution experiment — no breeding. Master seed derives all individual
//! seeds deterministically. Each individual evolves 30K steps with spatial
//! pocket-constrained mutation and strict acceptance.
//!
//! Run: cargo run --example pocket_evolve --release -- <corpus-path>

use instnct_core::{
    save_checkpoint, CheckpointMeta, Int8Projection, Network, PropagationConfig, SdrTable,
    WeightBackup,
};
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use rayon::prelude::*;
use std::fs;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MASTER_SEED: u64 = 1337;
const N_PAIRS: usize = 5;
const STEPS: usize = 30_000;
const EVAL_LEN_SHORT: usize = 100;
const EVAL_LEN_LONG: usize = 2000;
const LOG_INTERVAL: usize = 5000;
const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;

const POCKET_H: usize = 256;
const POCKET_OVERLAP: usize = 60;
const POCKET_STEP: usize = POCKET_H - POCKET_OVERLAP; // 196
const POCKET_PHI: usize = 158;
const N_POCKETS: usize = 2;

const PAIRWISE_EVAL_LEN: usize = 5000;

// Derived geometry for 2-pocket chain
const TOTAL_H: usize = POCKET_H + (N_POCKETS - 1) * POCKET_STEP; // 452
const SDR_INPUT_END: usize = POCKET_PHI; // [0..158)
const OUTPUT_START: usize = POCKET_STEP + (POCKET_H - POCKET_PHI); // 294
const OUT_DIM: usize = TOTAL_H - OUTPUT_START; // 158

// ---------------------------------------------------------------------------
// Pocket zone geometry (copied from pocket_chain.rs)
// ---------------------------------------------------------------------------

struct PocketZone {
    start: usize,
    end: usize,
}

impl PocketZone {
    fn contains(&self, neuron: usize) -> bool {
        neuron >= self.start && neuron < self.end
    }
}

fn pocket_zone(pocket_idx: usize) -> PocketZone {
    let start = pocket_idx * POCKET_STEP;
    PocketZone {
        start,
        end: start + POCKET_H,
    }
}

// ---------------------------------------------------------------------------
// Pocket mutation operators (copied from pocket_chain.rs)
// ---------------------------------------------------------------------------

fn pocket_add_edge(net: &mut Network, zone: &PocketZone, rng: &mut impl Rng) -> bool {
    let range = zone.end - zone.start;
    if range < 2 {
        return false;
    }
    for _ in 0..30 {
        let src = zone.start + rng.gen_range(0..range);
        let tgt = zone.start + rng.gen_range(0..range);
        if src != tgt && net.graph_mut().add_edge(src as u16, tgt as u16) {
            return true;
        }
    }
    false
}

fn pocket_remove_edge(net: &mut Network, zone: &PocketZone, rng: &mut impl Rng) -> bool {
    let edges: Vec<_> = net
        .graph()
        .iter_edges()
        .filter(|e| zone.contains(e.source as usize) && zone.contains(e.target as usize))
        .collect();
    if edges.is_empty() {
        return false;
    }
    let e = edges[rng.gen_range(0..edges.len())];
    net.graph_mut().remove_edge(e.source, e.target);
    true
}

fn pocket_rewire(net: &mut Network, zone: &PocketZone, rng: &mut impl Rng) -> bool {
    let edges: Vec<_> = net
        .graph()
        .iter_edges()
        .filter(|e| zone.contains(e.source as usize) && zone.contains(e.target as usize))
        .collect();
    if edges.is_empty() {
        return false;
    }
    let e = edges[rng.gen_range(0..edges.len())];
    let range = zone.end - zone.start;
    for _ in 0..30 {
        let new_tgt = zone.start + rng.gen_range(0..range);
        if new_tgt != e.source as usize {
            net.graph_mut().remove_edge(e.source, e.target);
            if net.graph_mut().add_edge(e.source, new_tgt as u16) {
                return true;
            }
            net.graph_mut().add_edge(e.source, e.target);
            return false;
        }
    }
    false
}

fn pocket_reverse(net: &mut Network, zone: &PocketZone, rng: &mut impl Rng) -> bool {
    let edges: Vec<_> = net
        .graph()
        .iter_edges()
        .filter(|e| zone.contains(e.source as usize) && zone.contains(e.target as usize))
        .collect();
    if edges.is_empty() {
        return false;
    }
    let e = edges[rng.gen_range(0..edges.len())];
    net.graph_mut().remove_edge(e.source, e.target);
    if net.graph_mut().add_edge(e.target, e.source) {
        return true;
    }
    net.graph_mut().add_edge(e.source, e.target);
    false
}

fn pocket_param(net: &mut Network, zone: &PocketZone, rng: &mut impl Rng) -> bool {
    let range = zone.end - zone.start;
    let idx = zone.start + rng.gen_range(0..range);
    let roll = rng.gen_range(0..3u32);
    match roll {
        0 => {
            net.threshold_mut()[idx] = rng.gen_range(0..=7);
            true
        }
        1 => {
            net.channel_mut()[idx] = rng.gen_range(1..=8);
            true
        }
        _ => {
            net.polarity_mut()[idx] *= -1;
            true
        }
    }
}

/// One mutation step within a pocket. Returns (mutated, Option<WeightBackup>).
fn pocket_mutate(
    net: &mut Network,
    proj: &mut Int8Projection,
    zone: &PocketZone,
    rng: &mut impl Rng,
    is_last: bool,
) -> (bool, Option<WeightBackup>) {
    let roll = rng.gen_range(0..100u32);
    match roll {
        0..25 => (pocket_add_edge(net, zone, rng), None),
        25..40 => (pocket_remove_edge(net, zone, rng), None),
        40..55 => (pocket_rewire(net, zone, rng), None),
        55..70 => (pocket_reverse(net, zone, rng), None),
        70..85 => (pocket_param(net, zone, rng), None),
        _ => {
            if is_last {
                let backup = proj.mutate_one(rng);
                (true, Some(backup))
            } else {
                (pocket_param(net, zone, rng), None)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Corpus loading
// ---------------------------------------------------------------------------

fn load_corpus(path: &str) -> Vec<u8> {
    let raw = fs::read(path).expect("cannot read corpus");
    raw.iter()
        .filter_map(|&b| {
            if b.is_ascii_lowercase() {
                Some(b - b'a')
            } else if b.is_ascii_uppercase() {
                Some(b.to_ascii_lowercase() - b'a')
            } else if b == b' ' || b == b'\n' || b == b'\t' {
                Some(26)
            } else {
                None
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn eval_accuracy(
    net: &mut Network,
    proj: &Int8Projection,
    corpus: &[u8],
    len: usize,
    rng: &mut StdRng,
    sdr: &SdrTable,
    config: &PropagationConfig,
) -> f64 {
    if corpus.len() <= len {
        return 0.0;
    }
    let off = rng.gen_range(0..=corpus.len() - len - 1);
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), config).unwrap();
        if proj.predict(&net.charge()[OUTPUT_START..TOTAL_H]) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

/// Eval that returns per-char predictions (for pairwise agreement).
fn eval_predictions(
    net: &mut Network,
    proj: &Int8Projection,
    corpus: &[u8],
    offset: usize,
    len: usize,
    sdr: &SdrTable,
    config: &PropagationConfig,
) -> Vec<usize> {
    net.reset();
    let seg = &corpus[offset..offset + len + 1];
    let mut preds = Vec::with_capacity(len);
    for &token in &seg[..len] {
        net.propagate(sdr.pattern(token as usize), config).unwrap();
        preds.push(proj.predict(&net.charge()[OUTPUT_START..TOTAL_H]));
    }
    preds
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bar(val: f64, max_val: f64, width: usize) -> String {
    let filled = ((val / max_val) * width as f64).round().min(width as f64) as usize;
    format!("{}{}", "#".repeat(filled), ".".repeat(width - filled))
}

/// Count edges within a specific pocket zone.
fn pocket_edge_count(net: &Network, zone: &PocketZone) -> usize {
    net.graph()
        .iter_edges()
        .filter(|e| zone.contains(e.source as usize) && zone.contains(e.target as usize))
        .count()
}

/// Count cross-pocket edges (not fully inside any single pocket).
fn cross_pocket_edge_count(net: &Network) -> usize {
    let total = net.edge_count();
    let mut in_pocket = 0;
    for p in 0..N_POCKETS {
        let zone = pocket_zone(p);
        in_pocket += pocket_edge_count(net, &zone);
    }
    total - in_pocket
}

/// Mean inhibitory neuron count per pocket.
fn pocket_inhibitory_counts(net: &Network) -> Vec<usize> {
    (0..N_POCKETS)
        .map(|p| {
            let zone = pocket_zone(p);
            net.polarity()[zone.start..zone.end]
                .iter()
                .filter(|&&v| v == -1)
                .count()
        })
        .collect()
}

/// Mean threshold per pocket.
fn pocket_mean_thresholds(net: &Network) -> Vec<f64> {
    (0..N_POCKETS)
        .map(|p| {
            let zone = pocket_zone(p);
            let sum: u32 = net.threshold()[zone.start..zone.end].iter().sum();
            sum as f64 / (zone.end - zone.start) as f64
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Individual
// ---------------------------------------------------------------------------

struct Individual {
    name: String,
    pair: char,
    #[allow(dead_code)]
    role: char,
    seed: u64,
    net: Network,
    proj: Int8Projection,
    sdr: SdrTable,
    mut_rng: StdRng,
    eval_rng: StdRng,
    // tracked stats
    accuracy: f64,
    peak_accuracy: f64,
    edges: usize,
    accepted: u32,
    total_tried: u32,
}

impl Individual {
    fn new(name: String, pair: char, role: char, ind_seed: u64, sdr_seed: u64) -> Self {
        let mut init_rng = StdRng::seed_from_u64(ind_seed);
        let mut proj_rng = StdRng::seed_from_u64(ind_seed.wrapping_add(200));
        let eval_rng = StdRng::seed_from_u64(ind_seed.wrapping_add(1000));
        let mut_rng = StdRng::seed_from_u64(ind_seed.wrapping_add(500));

        let mut net = Network::new(TOTAL_H);

        // Init each pocket with chain-50 + 5% density + random params
        for p in 0..N_POCKETS {
            let zone = pocket_zone(p);
            let zone_h = zone.end - zone.start;
            let phi = (zone_h as f64 / 1.618).round() as usize;
            let zone_os = zone.start + zone_h - phi;
            let zone_ie = zone.start + phi;
            let zone_om = (zone_os + zone_ie) / 2;

            // Chain-50 within pocket
            if zone_ie > zone_os + 1 {
                for _ in 0..50 {
                    let s = init_rng.gen_range(zone.start..zone_os) as u16;
                    let h1 = init_rng.gen_range(zone_os..zone_om) as u16;
                    let h2 = init_rng.gen_range(zone_om..zone_ie) as u16;
                    let t = init_rng.gen_range(zone_ie..zone.end) as u16;
                    net.graph_mut().add_edge(s, h1);
                    net.graph_mut().add_edge(h1, h2);
                    net.graph_mut().add_edge(h2, t);
                }
            }

            // 5% density fill within pocket
            let target = zone_h * zone_h * 5 / 100;
            for _ in 0..target * 3 {
                pocket_add_edge(&mut net, &zone, &mut init_rng);
                let pocket_edges = pocket_edge_count(&net, &zone);
                if pocket_edges >= target {
                    break;
                }
            }

            // Random params
            for i in zone.start..zone.end {
                net.threshold_mut()[i] = init_rng.gen_range(0..=7u32);
                net.channel_mut()[i] = init_rng.gen_range(1..=8u8);
                if init_rng.gen_ratio(1, 10) {
                    net.polarity_mut()[i] = -1;
                }
            }
        }

        // SDR (same seed for all individuals -> identical tables)
        let sdr = SdrTable::new(
            CHARS,
            TOTAL_H,
            SDR_INPUT_END,
            SDR_ACTIVE_PCT,
            &mut StdRng::seed_from_u64(sdr_seed),
        )
        .unwrap();

        // Projection from last pocket's output zone
        let proj = Int8Projection::new(OUT_DIM, CHARS, &mut proj_rng);

        let edges = net.edge_count();

        // Log per-pocket edge breakdown at init
        let pa = pocket_edge_count(&net, &pocket_zone(0));
        let pb = pocket_edge_count(&net, &pocket_zone(1));
        let x = cross_pocket_edge_count(&net);
        println!(
            "  {} seed={} H={} edges={} (pA={} pB={} x={})",
            name, ind_seed, TOTAL_H, edges, pa, pb, x
        );

        Individual {
            name,
            pair,
            role,
            seed: ind_seed,
            net,
            proj,
            sdr,
            mut_rng,
            eval_rng,
            accuracy: 0.0,
            peak_accuracy: 0.0,
            edges,
            accepted: 0,
            total_tried: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Evolution loop (per individual)
// ---------------------------------------------------------------------------

fn evolve(ind: &mut Individual, corpus: &[u8], config: &PropagationConfig) {
    for step in 0..STEPS {
        // Paired eval: clone eval_rng state before
        let snap = ind.eval_rng.clone();
        let before = eval_accuracy(
            &mut ind.net,
            &ind.proj,
            corpus,
            EVAL_LEN_SHORT,
            &mut ind.eval_rng,
            &ind.sdr,
            config,
        );
        ind.eval_rng = snap;

        let state = ind.net.save_state();

        // Pick random pocket, mutate within it
        let pocket_idx = ind.mut_rng.gen_range(0..N_POCKETS);
        let zone = pocket_zone(pocket_idx);
        let is_last = pocket_idx == N_POCKETS - 1;
        let (mutated, weight_backup) =
            pocket_mutate(&mut ind.net, &mut ind.proj, &zone, &mut ind.mut_rng, is_last);

        if !mutated {
            // Burn eval_rng to keep it in sync
            let _ = eval_accuracy(
                &mut ind.net,
                &ind.proj,
                corpus,
                EVAL_LEN_SHORT,
                &mut ind.eval_rng,
                &ind.sdr,
                config,
            );
            continue;
        }

        ind.total_tried += 1;

        let after = eval_accuracy(
            &mut ind.net,
            &ind.proj,
            corpus,
            EVAL_LEN_SHORT,
            &mut ind.eval_rng,
            &ind.sdr,
            config,
        );
        let accepted = after > before; // strict, no ties

        if accepted {
            ind.accepted += 1;
        } else {
            ind.net.restore_state(&state);
            if let Some(backup) = weight_backup {
                ind.proj.rollback(backup);
            }
        }

        // Periodic logging
        if (step + 1) % LOG_INTERVAL == 0 {
            let mut cr = StdRng::seed_from_u64(ind.seed + 6000 + step as u64);
            let acc = eval_accuracy(
                &mut ind.net,
                &ind.proj,
                corpus,
                EVAL_LEN_LONG,
                &mut cr,
                &ind.sdr,
                config,
            );
            if acc > ind.peak_accuracy {
                ind.peak_accuracy = acc;
            }
            ind.accuracy = acc;
            ind.edges = ind.net.edge_count();

            let pa = pocket_edge_count(&ind.net, &pocket_zone(0));
            let pb = pocket_edge_count(&ind.net, &pocket_zone(1));
            let x = cross_pocket_edge_count(&ind.net);
            let inh = pocket_inhibitory_counts(&ind.net);
            let theta = pocket_mean_thresholds(&ind.net);
            let accept_pct = if ind.total_tried > 0 {
                ind.accepted as f64 / ind.total_tried as f64 * 100.0
            } else {
                0.0
            };

            println!(
                "  [{:>5}] {} |{}| {:.1}%  edges={} (pA={} pB={} x={})  inh=[{},{}] \u{03b8}=[{:.1},{:.1}]  accept={:.1}%  peak={:.1}%",
                step + 1,
                ind.name,
                bar(acc, 0.30, 10),
                acc * 100.0,
                ind.edges,
                pa, pb, x,
                inh[0], inh[1],
                theta[0], theta[1],
                accept_pct,
                ind.peak_accuracy * 100.0,
            );
        }
    }

    // Final long eval
    let mut fr = StdRng::seed_from_u64(ind.seed + 9999);
    let final_acc = eval_accuracy(
        &mut ind.net,
        &ind.proj,
        corpus,
        EVAL_LEN_LONG,
        &mut fr,
        &ind.sdr,
        config,
    );
    if final_acc > ind.peak_accuracy {
        ind.peak_accuracy = final_acc;
    }
    ind.accuracy = final_acc;
    ind.edges = ind.net.edge_count();

    let accept_pct = if ind.total_tried > 0 {
        ind.accepted as f64 / ind.total_tried as f64 * 100.0
    } else {
        0.0
    };
    println!(
        "  {} FINAL: {:.2}%  peak={:.1}%  edges={}  accepted={}/{}({:.1}%)  seed={}",
        ind.name,
        final_acc * 100.0,
        ind.peak_accuracy * 100.0,
        ind.edges,
        ind.accepted,
        ind.total_tried,
        accept_pct,
        ind.seed,
    );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "S:/AI/work/VRAXION_DEV/instnct/data/traindat/fineweb_edu.traindat".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path);
    println!("  {} chars\n", corpus.len());

    let config = PropagationConfig {
        ticks_per_token: 6,
        input_duration_ticks: 2,
        decay_interval_ticks: 6,
        use_refractory: false,
    };

    // Derive all seeds from master
    let mut seed_gen = StdRng::seed_from_u64(MASTER_SEED);
    let sdr_seed = seed_gen.next_u64();

    let pair_letters = ['A', 'B', 'C', 'D', 'E'];
    let roles = ['F', 'M'];

    println!("=== POCKET EVOLVE: {} pairs x 2 = {} individuals, {} steps ===", N_PAIRS, N_PAIRS * 2, STEPS);
    println!("  H={} pockets={} overlap={} SDR=[0..{}) W=[{}..{})", TOTAL_H, N_POCKETS, POCKET_OVERLAP, SDR_INPUT_END, OUTPUT_START, TOTAL_H);
    println!("  master_seed={} sdr_seed={}\n", MASTER_SEED, sdr_seed);

    println!("Initializing individuals...");
    let mut individuals: Vec<Individual> = Vec::with_capacity(N_PAIRS * 2);
    for &pair in &pair_letters {
        for &role in &roles {
            let ind_seed = seed_gen.next_u64();
            let name = format!("{}{}", pair, role);
            individuals.push(Individual::new(name, pair, role, ind_seed, sdr_seed));
        }
    }
    println!();

    // Evolve all in parallel
    println!("Evolving {} individuals in parallel ({} steps each)...\n", individuals.len(), STEPS);
    individuals.par_iter_mut().for_each(|ind| {
        evolve(ind, &corpus, &config);
    });

    // Save checkpoints
    println!("\nSaving checkpoints...");
    for ind in &individuals {
        let path = format!("checkpoints/pocket_evolve/{}_final.ckpt", ind.name);
        save_checkpoint(
            &path,
            &ind.net,
            &ind.proj,
            CheckpointMeta {
                step: STEPS,
                accuracy: ind.accuracy,
                label: format!(
                    "pocket_evolve {} seed={} peak={:.1}%",
                    ind.name,
                    ind.seed,
                    ind.peak_accuracy * 100.0
                ),
            },
        )
        .unwrap();
        println!("  saved {}", path);
    }

    // -----------------------------------------------------------------------
    // Final summary — sorted by accuracy
    // -----------------------------------------------------------------------
    println!("\n=== RESULTS (sorted by accuracy) ===\n");

    let mut sorted: Vec<usize> = (0..individuals.len()).collect();
    sorted.sort_by(|&a, &b| {
        individuals[b]
            .accuracy
            .partial_cmp(&individuals[a].accuracy)
            .unwrap()
    });

    println!(
        "  {:<4}  {:<4}  {:<4}  {:>7}  {:>7}  {:>6}  {:>8}  Seed",
        "Rank", "Name", "Pair", "Final%", "Peak%", "Edges", "Accept%"
    );
    println!(
        "  {:<4}  {:<4}  {:<4}  {:>7}  {:>7}  {:>6}  {:>8}  ----",
        "----", "----", "----", "------", "-----", "-----", "-------"
    );
    for (rank, &idx) in sorted.iter().enumerate() {
        let ind = &individuals[idx];
        let accept_pct = if ind.total_tried > 0 {
            ind.accepted as f64 / ind.total_tried as f64 * 100.0
        } else {
            0.0
        };
        println!(
            "  {:<4}  {:<4}  {:<4}  {:>6.2}%  {:>6.2}%  {:>6}  {:>7.1}%  {}",
            rank + 1,
            ind.name,
            ind.pair,
            ind.accuracy * 100.0,
            ind.peak_accuracy * 100.0,
            ind.edges,
            accept_pct,
            ind.seed,
        );
    }

    // Statistics
    let accs: Vec<f64> = individuals.iter().map(|i| i.accuracy).collect();
    let n = accs.len() as f64;
    let mean = accs.iter().sum::<f64>() / n;

    let mut sorted_accs: Vec<f64> = accs.clone();
    sorted_accs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if sorted_accs.len().is_multiple_of(2) {
        (sorted_accs[sorted_accs.len() / 2 - 1] + sorted_accs[sorted_accs.len() / 2]) / 2.0
    } else {
        sorted_accs[sorted_accs.len() / 2]
    };
    let variance = accs.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / n;
    let stddev = variance.sqrt();

    // Top-5 / bottom-5 by sorted index
    let top5_accs: Vec<f64> = sorted.iter().take(5).map(|&i| individuals[i].accuracy).collect();
    let bot5_accs: Vec<f64> = sorted.iter().skip(5).map(|&i| individuals[i].accuracy).collect();
    let top5_mean = top5_accs.iter().sum::<f64>() / top5_accs.len() as f64;
    let bot5_mean = bot5_accs.iter().sum::<f64>() / bot5_accs.len() as f64;

    println!();
    println!(
        "Mean: {:.1}%   Median: {:.1}%   StdDev: {:.1}pp",
        mean * 100.0,
        median * 100.0,
        stddev * 100.0
    );
    println!(
        "Top-5 mean: {:.1}%   Bottom-5 mean: {:.1}%",
        top5_mean * 100.0,
        bot5_mean * 100.0
    );

    // Pair summary
    println!("\nPair summary:");
    for &pair in &pair_letters {
        let pair_inds: Vec<&Individual> = individuals.iter().filter(|i| i.pair == pair).collect();
        let f_acc = pair_inds
            .iter()
            .find(|i| i.name.ends_with('F'))
            .map(|i| i.accuracy)
            .unwrap_or(0.0);
        let m_acc = pair_inds
            .iter()
            .find(|i| i.name.ends_with('M'))
            .map(|i| i.accuracy)
            .unwrap_or(0.0);
        let best = f_acc.max(m_acc);
        let spread = (f_acc - m_acc).abs();
        println!(
            "  Pair {}: {}F={:.1}% {}M={:.1}%  best={:.1}%  spread={:.1}pp",
            pair,
            pair,
            f_acc * 100.0,
            pair,
            m_acc * 100.0,
            best * 100.0,
            spread * 100.0,
        );
    }

    // -----------------------------------------------------------------------
    // Pairwise prediction agreement
    // -----------------------------------------------------------------------
    println!("\n=== PAIRWISE AGREEMENT ({} chars) ===\n", PAIRWISE_EVAL_LEN);

    // Pick a fixed offset for pairwise eval
    let mut pw_rng = StdRng::seed_from_u64(MASTER_SEED.wrapping_add(7777));
    let pw_offset = pw_rng.gen_range(0..=corpus.len() - PAIRWISE_EVAL_LEN - 1);
    let pw_targets: Vec<u8> = corpus[pw_offset + 1..pw_offset + PAIRWISE_EVAL_LEN + 1].to_vec();

    // Collect predictions for all individuals (sequential — needs &mut net)
    let mut all_preds: Vec<Vec<usize>> = Vec::with_capacity(individuals.len());
    for ind in &mut individuals {
        let preds = eval_predictions(
            &mut ind.net,
            &ind.proj,
            &corpus,
            pw_offset,
            PAIRWISE_EVAL_LEN,
            &ind.sdr,
            &config,
        );
        all_preds.push(preds);
    }

    let names: Vec<String> = individuals.iter().map(|i| i.name.clone()).collect();
    let n_ind = individuals.len();

    // Header
    print!("       ");
    for name in &names {
        print!("{:>6}", name);
    }
    println!();

    // Matrix
    for i in 0..n_ind {
        print!("  {:>4}", names[i]);
        for j in 0..n_ind {
            if i == j {
                print!("    --");
            } else {
                let agree: usize = all_preds[i]
                    .iter()
                    .zip(all_preds[j].iter())
                    .filter(|(&a, &b)| a == b)
                    .count();
                let pct = agree as f64 / PAIRWISE_EVAL_LEN as f64 * 100.0;
                print!("{:>6.1}", pct);
            }
        }
        println!();
    }

    // Both-correct and oracle stats
    println!();
    print!("  Both-correct %:");
    println!();
    print!("       ");
    for name in &names {
        print!("{:>6}", name);
    }
    println!();
    for i in 0..n_ind {
        print!("  {:>4}", names[i]);
        for j in 0..n_ind {
            if i == j {
                // Individual accuracy on this segment
                let correct: usize = all_preds[i]
                    .iter()
                    .zip(pw_targets.iter())
                    .filter(|(&p, &t)| p == t as usize)
                    .count();
                let pct = correct as f64 / PAIRWISE_EVAL_LEN as f64 * 100.0;
                print!("{:>6.1}", pct);
            } else {
                let both: usize = (0..PAIRWISE_EVAL_LEN)
                    .filter(|&k| {
                        all_preds[i][k] == pw_targets[k] as usize
                            && all_preds[j][k] == pw_targets[k] as usize
                    })
                    .count();
                let pct = both as f64 / PAIRWISE_EVAL_LEN as f64 * 100.0;
                print!("{:>6.1}", pct);
            }
        }
        println!();
    }

    // Oracle stats
    let oracle_all: usize = (0..PAIRWISE_EVAL_LEN)
        .filter(|&k| {
            all_preds
                .iter()
                .any(|preds| preds[k] == pw_targets[k] as usize)
        })
        .count();
    let oracle_all_pct = oracle_all as f64 / PAIRWISE_EVAL_LEN as f64 * 100.0;

    // Top 5 oracle
    let top5_indices: Vec<usize> = sorted.iter().take(5).copied().collect();
    let oracle_top5: usize = (0..PAIRWISE_EVAL_LEN)
        .filter(|&k| {
            top5_indices
                .iter()
                .any(|&i| all_preds[i][k] == pw_targets[k] as usize)
        })
        .count();
    let oracle_top5_pct = oracle_top5 as f64 / PAIRWISE_EVAL_LEN as f64 * 100.0;

    // Pairwise oracle matrix
    println!();
    println!("  Oracle % (either correct):");
    print!("       ");
    for name in &names {
        print!("{:>6}", name);
    }
    println!();
    for i in 0..n_ind {
        print!("  {:>4}", names[i]);
        for j in 0..n_ind {
            if i == j {
                print!("    --");
            } else {
                let oracle: usize = (0..PAIRWISE_EVAL_LEN)
                    .filter(|&k| {
                        all_preds[i][k] == pw_targets[k] as usize
                            || all_preds[j][k] == pw_targets[k] as usize
                    })
                    .count();
                let pct = oracle as f64 / PAIRWISE_EVAL_LEN as f64 * 100.0;
                print!("{:>6.1}", pct);
            }
        }
        println!();
    }

    println!();
    println!("Oracle (any of {} correct): {:.1}%", n_ind, oracle_all_pct);
    println!("Oracle (top 5 only):        {:.1}%", oracle_top5_pct);
}
