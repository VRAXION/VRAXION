//! Pocket cross-breed: N-pair hierarchical breeding experiment.
//!
//! 2*N individuals in N pairs evolve in parallel (rayon par_iter_mut), then
//! within-pair breed every generation and cross-pair breed every
//! CROSS_BREED_INTERVAL generations.  The hybrid replaces the worst individual
//! across all pairs.
//!
//! Run: cargo run --example pocket_cross --release -- <corpus-path>

use instnct_core::{
    save_checkpoint, CheckpointMeta, Int8Projection, Network, PropagationConfig, SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const POCKET_H: usize = 256;
const POCKET_PHI: usize = 158;
const POCKET_OVERLAP: usize = 60;
const POCKET_STEP: usize = POCKET_H - POCKET_OVERLAP; // 196
const N_POCKETS: usize = 2;

const GENERATIONS: usize = 6;
const STEPS_PER_GEN: usize = 30_000;
const CROSS_BREED_INTERVAL: usize = 2; // cross-breed every 2 gens
const EVAL_LEN_SHORT: usize = 100;
const EVAL_LEN_LONG: usize = 2000;
const LOG_INTERVAL: usize = 5000;

const CRYSTALLIZE_MAX_ROUNDS: usize = 15;
const CRYSTALLIZE_START_PCT: f64 = 0.50;      // start aggressive
const CRYSTALLIZE_MIN_PCT: f64 = 0.02;        // stop when below this
const CRYSTALLIZE_DROP_THRESHOLD_PP: f64 = 0.02;

const N_PAIRS: usize = 10;
const PAIR_SEEDS: [(u64, u64); 10] = [
    (42, 123),
    (7, 1042),
    (256, 789),
    (1337, 2048),
    (555, 666),
    (100, 200),
    (300, 400),
    (500, 600),
    (700, 800),
    (900, 1000),
];

const SDR_SEED: u64 = 42 + 100; // same SDR for all individuals

const CHECKPOINT_DIR: &str = "checkpoints/pocket_cross";

// ---------------------------------------------------------------------------
// Pocket geometry
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct PocketZone {
    start: usize,
    end: usize,
}

impl PocketZone {
    fn contains(&self, neuron: usize) -> bool {
        neuron >= self.start && neuron < self.end
    }
}

fn total_neurons() -> usize {
    POCKET_H + (N_POCKETS - 1) * POCKET_STEP // 452
}

fn pocket_zone(pocket_idx: usize) -> PocketZone {
    let start = pocket_idx * POCKET_STEP;
    PocketZone {
        start,
        end: start + POCKET_H,
    }
}

fn sdr_input_end() -> usize {
    POCKET_PHI // 158
}

fn output_start() -> usize {
    let last = pocket_zone(N_POCKETS - 1);
    last.start + (POCKET_H - POCKET_PHI) // 294
}

fn out_dim() -> usize {
    total_neurons() - output_start() // 158
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
// Pocket mutations
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

fn pocket_mutate(
    net: &mut Network,
    proj: &mut Int8Projection,
    zone: &PocketZone,
    rng: &mut impl Rng,
    is_last: bool,
) -> bool {
    let roll = rng.gen_range(0..100u32);
    match roll {
        0..25 => pocket_add_edge(net, zone, rng),
        25..40 => pocket_remove_edge(net, zone, rng),
        40..55 => pocket_rewire(net, zone, rng),
        55..70 => pocket_reverse(net, zone, rng),
        70..85 => pocket_param(net, zone, rng),
        _ => {
            if is_last {
                let _ = proj.mutate_one(rng);
                true
            } else {
                pocket_param(net, zone, rng)
            }
        }
    }
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
    os: usize,
    h: usize,
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
        if proj.predict(&net.charge()[os..h]) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

#[allow(clippy::too_many_arguments)]
fn eval_accuracy_segment(
    net: &mut Network,
    proj: &Int8Projection,
    segment: &[u8],
    sdr: &SdrTable,
    config: &PropagationConfig,
    os: usize,
    h: usize,
) -> f64 {
    let len = segment.len() - 1;
    if len == 0 {
        return 0.0;
    }
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(sdr.pattern(segment[i] as usize), config)
            .unwrap();
        if proj.predict(&net.charge()[os..h]) == segment[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

// ---------------------------------------------------------------------------
// Edge statistics
// ---------------------------------------------------------------------------

struct EdgeStats {
    total: usize,
    pocket_a: usize,
    pocket_b: usize,
    cross: usize,
}

fn count_edges(net: &Network) -> EdgeStats {
    let za = pocket_zone(0);
    let zb = pocket_zone(1);
    let mut pa = 0usize;
    let mut pb = 0usize;
    let mut cross = 0usize;
    for e in net.graph().iter_edges() {
        let s = e.source as usize;
        let t = e.target as usize;
        let in_a = za.contains(s) && za.contains(t);
        let in_b = zb.contains(s) && zb.contains(t);
        if in_a {
            pa += 1;
        } else if in_b {
            pb += 1;
        } else {
            cross += 1;
        }
    }
    EdgeStats {
        total: net.edge_count(),
        pocket_a: pa,
        pocket_b: pb,
        cross,
    }
}

struct PocketParamStats {
    inhibitory_a: usize,
    inhibitory_b: usize,
    mean_threshold_a: f64,
    mean_threshold_b: f64,
}

fn pocket_param_stats(net: &Network) -> PocketParamStats {
    let za = pocket_zone(0);
    let zb = pocket_zone(1);
    let mut inh_a = 0usize;
    let mut inh_b = 0usize;
    let mut sum_th_a = 0u64;
    let mut sum_th_b = 0u64;
    for i in za.start..za.end {
        if net.polarity()[i] == -1 {
            inh_a += 1;
        }
        sum_th_a += net.threshold()[i] as u64;
    }
    for i in zb.start..zb.end {
        if net.polarity()[i] == -1 {
            inh_b += 1;
        }
        sum_th_b += net.threshold()[i] as u64;
    }
    PocketParamStats {
        inhibitory_a: inh_a,
        inhibitory_b: inh_b,
        mean_threshold_a: sum_th_a as f64 / POCKET_H as f64,
        mean_threshold_b: sum_th_b as f64 / POCKET_H as f64,
    }
}

fn edge_set(net: &Network) -> HashSet<(u16, u16)> {
    net.graph()
        .iter_edges()
        .map(|e| (e.source, e.target))
        .collect()
}

fn edge_set_in_zone(net: &Network, zone: &PocketZone) -> HashSet<(u16, u16)> {
    net.graph()
        .iter_edges()
        .filter(|e| zone.contains(e.source as usize) && zone.contains(e.target as usize))
        .map(|e| (e.source, e.target))
        .collect()
}

fn jaccard(a: &HashSet<(u16, u16)>, b: &HashSet<(u16, u16)>) -> f64 {
    let intersection = a.intersection(b).count();
    let union_count = a.union(b).count();
    if union_count == 0 {
        0.0
    } else {
        intersection as f64 / union_count as f64
    }
}

fn bar(val: f64, max_val: f64, width: usize) -> String {
    let filled = ((val / max_val) * width as f64).round().min(width as f64) as usize;
    format!("{}{}", "#".repeat(filled), ".".repeat(width - filled))
}

// ---------------------------------------------------------------------------
// Individual + Pair structs
// ---------------------------------------------------------------------------

struct Individual {
    name: String,
    net: Network,
    proj: Int8Projection,
    sdr: SdrTable,
    mut_rng: StdRng,
    eval_rng: StdRng,
    accuracy: f64,
    generation_born: usize,
    edges: usize,
}

struct Pair {
    first: Option<Individual>,  // "first" role
    second: Option<Individual>, // "second" role
    label: String,
    pair_idx: usize,
    first_is_older: bool,
}

// ---------------------------------------------------------------------------
// Network initialization (2-pocket, chain-50 + 5% density)
// ---------------------------------------------------------------------------

fn init_network(seed: u64, rng: &mut StdRng) -> Network {
    let h = total_neurons();
    let mut net = Network::new(h);

    for p in 0..N_POCKETS {
        let zone = pocket_zone(p);
        let zone_h = zone.end - zone.start;
        let phi = (zone_h as f64 / 1.618).round() as usize;
        let zone_os = zone.start + zone_h - phi;
        let zone_ie = zone.start + phi;
        let zone_om = (zone_os + zone_ie) / 2;

        // Chain-50
        if zone_ie > zone_os + 1 {
            for _ in 0..50 {
                let s = rng.gen_range(zone.start..zone_os) as u16;
                let h1 = rng.gen_range(zone_os..zone_om) as u16;
                let h2 = rng.gen_range(zone_om..zone_ie) as u16;
                let t = rng.gen_range(zone_ie..zone.end) as u16;
                net.graph_mut().add_edge(s, h1);
                net.graph_mut().add_edge(h1, h2);
                net.graph_mut().add_edge(h2, t);
            }
        }

        // 5% density fill
        let target = zone_h * zone_h * 5 / 100;
        for _ in 0..target * 3 {
            pocket_add_edge(&mut net, &zone, rng);
            let pocket_edges: usize = net
                .graph()
                .iter_edges()
                .filter(|e| zone.contains(e.source as usize) && zone.contains(e.target as usize))
                .count();
            if pocket_edges >= target {
                break;
            }
        }

        // Random params
        for i in zone.start..zone.end {
            net.threshold_mut()[i] = rng.gen_range(0..=7u32);
            net.channel_mut()[i] = rng.gen_range(1..=8u8);
            if rng.gen_ratio(1, 10) {
                net.polarity_mut()[i] = -1;
            }
        }
    }

    let es = count_edges(&net);
    println!(
        "    [init] seed={} H={} edges={} (pA={} pB={} cross={})",
        seed,
        h,
        es.total,
        es.pocket_a,
        es.pocket_b,
        es.cross
    );
    net
}

fn make_sdr() -> SdrTable {
    SdrTable::new(
        CHARS,
        total_neurons(),
        sdr_input_end(),
        SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(SDR_SEED),
    )
    .unwrap()
}

fn create_individual(name: &str, seed: u64) -> Individual {
    println!("  --- Initializing {} (seed {}) ---", name, seed);
    let mut init_rng = StdRng::seed_from_u64(seed);
    let net = init_network(seed, &mut init_rng);
    let proj = Int8Projection::new(out_dim(), CHARS, &mut StdRng::seed_from_u64(seed + 200));
    let edges = net.edge_count();
    Individual {
        name: name.to_string(),
        net,
        proj,
        sdr: make_sdr(),
        mut_rng: StdRng::seed_from_u64(seed),
        eval_rng: StdRng::seed_from_u64(seed + 1000),
        accuracy: 0.0,
        generation_born: 0,
        edges,
    }
}

fn create_pair(pair_idx: usize, first_seed: u64, second_seed: u64) -> Pair {
    let label = format!("Pair {}", pair_idx);
    let first_name = format!("P{}_First", pair_idx);
    let second_name = format!("P{}_Second", pair_idx);
    println!("\n  === Creating {} (seeds: {}, {}) ===", label, first_seed, second_seed);
    let first = create_individual(&first_name, first_seed);
    let second = create_individual(&second_name, second_seed);
    Pair {
        first: Some(first),
        second: Some(second),
        label,
        pair_idx,
        first_is_older: true,
    }
}

// ---------------------------------------------------------------------------
// Evolution: run STEPS_PER_GEN mutation steps on an individual
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn evolve_individual(
    indiv: &mut Individual,
    corpus: &[u8],
    prop: &PropagationConfig,
    pair_label: &str,
    best_ever: &mut f64,
) -> (f64, usize) {
    let h = total_neurons();
    let os = output_start();
    let mut accepted = 0u32;
    let mut total_tried = 0u32;
    let mut last_log_accepted = 0u32;
    let mut last_log_total = 0u32;

    println!(
        "    --- Evolving {} ({}) for {} steps ---",
        indiv.name, pair_label, STEPS_PER_GEN
    );

    for step in 0..STEPS_PER_GEN {
        // Paired eval
        let snap = indiv.eval_rng.clone();
        let before = eval_accuracy(
            &mut indiv.net,
            &indiv.proj,
            corpus,
            EVAL_LEN_SHORT,
            &mut indiv.eval_rng,
            &indiv.sdr,
            prop,
            os,
            h,
        );
        indiv.eval_rng = snap;

        let state = indiv.net.save_state();
        let pocket_idx = indiv.mut_rng.gen_range(0..N_POCKETS);
        let zone = pocket_zone(pocket_idx);
        let is_last = pocket_idx == N_POCKETS - 1;

        let proj_clone = indiv.proj.clone();
        let mutated = pocket_mutate(&mut indiv.net, &mut indiv.proj, &zone, &mut indiv.mut_rng, is_last);
        let w_backup_option = if mutated {
            Some(proj_clone)
        } else {
            None
        };

        if !mutated {
            let _ = eval_accuracy(
                &mut indiv.net,
                &indiv.proj,
                corpus,
                EVAL_LEN_SHORT,
                &mut indiv.eval_rng,
                &indiv.sdr,
                prop,
                os,
                h,
            );
            continue;
        }

        total_tried += 1;

        let after = eval_accuracy(
            &mut indiv.net,
            &indiv.proj,
            corpus,
            EVAL_LEN_SHORT,
            &mut indiv.eval_rng,
            &indiv.sdr,
            prop,
            os,
            h,
        );
        let accept = after > before;

        if accept {
            accepted += 1;
        } else {
            indiv.net.restore_state(&state);
            if let Some(backup) = w_backup_option {
                indiv.proj = backup;
            }
        }

        // Periodic logging
        if (step + 1) % LOG_INTERVAL == 0 {
            let mut cr = StdRng::seed_from_u64(SDR_SEED + 6000 + step as u64);
            let acc = eval_accuracy(
                &mut indiv.net,
                &indiv.proj,
                corpus,
                EVAL_LEN_LONG,
                &mut cr,
                &indiv.sdr,
                prop,
                os,
                h,
            );
            if acc > *best_ever {
                *best_ever = acc;
            }

            let es = count_edges(&indiv.net);
            let ps = pocket_param_stats(&indiv.net);
            let interval_accepted = accepted - last_log_accepted;
            let interval_total = total_tried - last_log_total;
            let accept_rate = if interval_total > 0 {
                interval_accepted as f64 / interval_total as f64
            } else {
                0.0
            };

            println!(
                "    [{:>6}] {} ({}) |{}| {:.1}%  edges={} (pA={} pB={} x={})  \
                 inh=[{},{}] theta=[{:.1},{:.1}]  accept={:.1}%  best_ever={:.1}%",
                step + 1,
                indiv.name,
                pair_label,
                bar(acc, 0.30, 20),
                acc * 100.0,
                es.total,
                es.pocket_a,
                es.pocket_b,
                es.cross,
                ps.inhibitory_a,
                ps.inhibitory_b,
                ps.mean_threshold_a,
                ps.mean_threshold_b,
                accept_rate * 100.0,
                *best_ever * 100.0,
            );

            last_log_accepted = accepted;
            last_log_total = total_tried;
        }
    }

    // Final accuracy
    let mut fr = StdRng::seed_from_u64(SDR_SEED + 9999);
    let final_acc = eval_accuracy(
        &mut indiv.net,
        &indiv.proj,
        corpus,
        EVAL_LEN_LONG,
        &mut fr,
        &indiv.sdr,
        prop,
        os,
        h,
    );
    if final_acc > *best_ever {
        *best_ever = final_acc;
    }
    let final_edges = indiv.net.edge_count();

    let overall_rate = if total_tried > 0 {
        accepted as f64 / total_tried as f64
    } else {
        0.0
    };
    println!(
        "    {} ({}) FINAL: {:.2}%  edges={}  accepted={}/{}({:.1}%)",
        indiv.name,
        pair_label,
        final_acc * 100.0,
        final_edges,
        accepted,
        total_tried,
        overall_rate * 100.0,
    );

    indiv.accuracy = final_acc;
    indiv.edges = final_edges;

    (final_acc, final_edges)
}

// ---------------------------------------------------------------------------
// Breeding: union edges + copy params + crystallize
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn breed(
    parent_a: &Network,
    parent_b: &Network,
    parent_a_proj: &Int8Projection,
    parent_a_name: &str,
    parent_b_name: &str,
    corpus: &[u8],
    sdr: &SdrTable,
    prop: &PropagationConfig,
    gen: usize,
    breed_rng: &mut StdRng,
    label: &str,
) -> (Network, Int8Projection) {
    let h = total_neurons();
    let os = output_start();

    println!("\n    === {} (gen {}) ===", label, gen);
    println!(
        "    Parents: {} ({} edges) x {} ({} edges)",
        parent_a_name,
        parent_a.edge_count(),
        parent_b_name,
        parent_b.edge_count(),
    );

    // --- Edge overlap analysis ---
    let a_edges = edge_set(parent_a);
    let b_edges = edge_set(parent_b);
    let total_jaccard = jaccard(&a_edges, &b_edges);
    let shared = a_edges.intersection(&b_edges).count();

    let za = pocket_zone(0);
    let zb = pocket_zone(1);
    let ae_0 = edge_set_in_zone(parent_a, &za);
    let be_0 = edge_set_in_zone(parent_b, &za);
    let ae_1 = edge_set_in_zone(parent_a, &zb);
    let be_1 = edge_set_in_zone(parent_b, &zb);
    let jaccard_0 = jaccard(&ae_0, &be_0);
    let jaccard_1 = jaccard(&ae_1, &be_1);

    println!(
        "    Edge overlap: shared={} Jaccard={:.3} (pocket0={:.3} pocket1={:.3})",
        shared, total_jaccard, jaccard_0, jaccard_1
    );
    println!(
        "    {} edges: {} (p0={} p1={})",
        parent_a_name,
        parent_a.edge_count(),
        ae_0.len(),
        ae_1.len(),
    );
    println!(
        "    {} edges: {} (p0={} p1={})",
        parent_b_name,
        parent_b.edge_count(),
        be_0.len(),
        be_1.len(),
    );

    // --- Step 1: Create child with union of all edges ---
    let mut child = Network::new(h);

    let mut added_from_a = 0usize;
    for e in parent_a.graph().iter_edges() {
        if child.graph_mut().add_edge(e.source, e.target) {
            added_from_a += 1;
        }
    }

    let mut added_from_b = 0usize;
    let mut skipped_dups = 0usize;
    for e in parent_b.graph().iter_edges() {
        if child.graph_mut().add_edge(e.source, e.target) {
            added_from_b += 1;
        } else {
            skipped_dups += 1;
        }
    }

    let union_edges = child.edge_count();
    println!(
        "    Union: {} edges (from_{}={} from_{}={} dups_skipped={})",
        union_edges, parent_a_name, added_from_a, parent_b_name, added_from_b, skipped_dups
    );

    // --- Step 2: Copy first parent's parameters ---
    for i in 0..h {
        child.threshold_mut()[i] = parent_a.threshold()[i];
        child.channel_mut()[i] = parent_a.channel()[i];
        child.polarity_mut()[i] = parent_a.polarity()[i];
    }
    println!(
        "    Copied {}'s neuron parameters (threshold/channel/polarity)",
        parent_a_name
    );

    // --- Step 3: Inherit W from first parent (not fresh!) ---
    let child_proj = parent_a_proj.clone();
    println!(
        "    Inherited W from {} ({}x{}, {} weights)",
        parent_a_name,
        out_dim(),
        CHARS,
        child_proj.weight_count()
    );

    // --- Step 4: Adaptive batch crystallize ---
    println!("    --- Adaptive Crystallize ---");

    let cryst_off = breed_rng.gen_range(0..=corpus.len() - EVAL_LEN_LONG - 1);
    let cryst_seg = &corpus[cryst_off..cryst_off + EVAL_LEN_LONG + 1];

    let baseline_acc =
        eval_accuracy_segment(&mut child, &child_proj, cryst_seg, sdr, prop, os, h);
    println!(
        "    Crystallize baseline: {:.2}% ({} edges)",
        baseline_acc * 100.0,
        child.edge_count()
    );

    let mut current_acc = baseline_acc;
    let mut removal_pct = CRYSTALLIZE_START_PCT;

    for round in 0..CRYSTALLIZE_MAX_ROUNDS {
        let edges_before = child.edge_count();
        if edges_before == 0 {
            break;
        }
        let batch_size = ((edges_before as f64 * removal_pct) as usize).max(1);

        // Collect and shuffle edges
        let all_edges: Vec<_> = child.graph().iter_edges().collect();
        let mut indices: Vec<usize> = (0..all_edges.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = breed_rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        // Batch remove
        let batch: Vec<(u16, u16)> = indices[..batch_size.min(indices.len())]
            .iter()
            .map(|&i| (all_edges[i].source, all_edges[i].target))
            .collect();
        for &(src, tgt) in &batch {
            child.graph_mut().remove_edge(src, tgt);
        }

        let after_acc =
            eval_accuracy_segment(&mut child, &child_proj, cryst_seg, sdr, prop, os, h);
        let edges_after = child.edge_count();

        let accepted = after_acc >= current_acc - CRYSTALLIZE_DROP_THRESHOLD_PP;

        if accepted {
            // Keep the removal, get greedier
            let old_pct = removal_pct;
            removal_pct = (removal_pct * 1.5).min(0.80);
            current_acc = after_acc;
            println!(
                "      Round {:>2}: ACCEPT  removed {} edges ({}->{})  acc {:.2}%  pct {:.0}%->>{:.0}%",
                round,
                batch.len(),
                edges_before,
                edges_after,
                after_acc * 100.0,
                old_pct * 100.0,
                removal_pct * 100.0,
            );
        } else {
            // Undo, back off
            for &(src, tgt) in &batch {
                child.graph_mut().add_edge(src, tgt);
            }
            let old_pct = removal_pct;
            removal_pct /= 2.0;
            println!(
                "      Round {:>2}: REJECT  tried {} edges, acc would be {:.2}% (need >={:.2}%)  pct {:.0}%->>{:.0}%",
                round,
                batch.len(),
                after_acc * 100.0,
                (current_acc - CRYSTALLIZE_DROP_THRESHOLD_PP) * 100.0,
                old_pct * 100.0,
                removal_pct * 100.0,
            );
            if removal_pct < CRYSTALLIZE_MIN_PCT {
                println!("      Removal rate below {:.0}%, stopping.", CRYSTALLIZE_MIN_PCT * 100.0);
                break;
            }
        }
    }

    let final_es = count_edges(&child);
    let child_acc = eval_accuracy_segment(&mut child, &child_proj, cryst_seg, sdr, prop, os, h);
    println!(
        "    Crystallize done: {:.2}% edges={} (p0={} p1={} x={}) (was {} before)",
        child_acc * 100.0,
        final_es.total,
        final_es.pocket_a,
        final_es.pocket_b,
        final_es.cross,
        union_edges,
    );

    // Standard eval
    let mut init_rng = StdRng::seed_from_u64(SDR_SEED + 5555 + gen as u64);
    let init_acc = eval_accuracy(
        &mut child,
        &child_proj,
        corpus,
        EVAL_LEN_LONG,
        &mut init_rng,
        sdr,
        prop,
        os,
        h,
    );
    println!(
        "    Child initial accuracy (2000 char, rng eval): {:.2}%\n",
        init_acc * 100.0
    );

    (child, child_proj)
}

// ---------------------------------------------------------------------------
// Checkpoint helper
// ---------------------------------------------------------------------------

fn do_save_checkpoint(
    label: &str,
    net: &Network,
    proj: &Int8Projection,
    step: usize,
    accuracy: f64,
) {
    let path = format!("{}/{}.ckpt", CHECKPOINT_DIR, label);
    let meta = CheckpointMeta {
        step,
        accuracy,
        label: label.to_string(),
    };
    match save_checkpoint(&path, net, proj, meta) {
        Ok(()) => println!("    [checkpoint] saved: {}", path),
        Err(e) => eprintln!("    [checkpoint] FAILED to save {}: {}", path, e),
    }
}

// ---------------------------------------------------------------------------
// Per-generation tracking
// ---------------------------------------------------------------------------

struct PairResult {
    pair_label: String,
    first_name: String,
    first_acc: f64,
    second_name: String,
    second_acc: f64,
    child_acc: f64,
    child_edges: usize,
}

struct GenRecord {
    gen: usize,
    pair_results: Vec<PairResult>,
    cross_breed_happened: bool,
    hybrid_acc: f64,
    hybrid_replaced: String,
    best_overall: f64,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "S:/AI/work/VRAXION_DEV/instnct/data/traindat/fineweb_edu.traindat".to_string()
    });

    println!("================================================================");
    println!("  POCKET CROSS-BREED EXPERIMENT (N={} pairs)", N_PAIRS);
    println!("================================================================");
    println!("  Generations:       {}", GENERATIONS);
    println!("  Steps/gen:         {}", STEPS_PER_GEN);
    println!("  Cross-breed every: {} gens", CROSS_BREED_INTERVAL);
    println!("  Pockets:           {} (H={})", N_POCKETS, total_neurons());
    println!("  SDR input:         [0..{})", sdr_input_end());
    println!("  W output:          [{}..{})", output_start(), total_neurons());
    println!(
        "  Out dim:           {} (W: {}x{})",
        out_dim(),
        out_dim(),
        CHARS
    );
    println!("  Pairs:             {} ({} individuals)", N_PAIRS, N_PAIRS * 2);
    for (i, (s1, s2)) in PAIR_SEEDS.iter().enumerate() {
        println!("    Pair {:>2}: seeds ({}, {})", i, s1, s2);
    }
    println!();

    println!("Loading corpus from: {}", corpus_path);
    let corpus = load_corpus(&corpus_path);
    println!("  {} chars loaded\n", corpus.len());

    // Ensure checkpoint directory exists
    fs::create_dir_all(CHECKPOINT_DIR).expect("cannot create checkpoint dir");

    let prop = PropagationConfig {
        ticks_per_token: 6,
        input_duration_ticks: 2,
        decay_interval_ticks: 6,
        use_refractory: false,
    };

    // --- Initialize all pairs ---
    let mut pairs: Vec<Pair> = Vec::with_capacity(N_PAIRS);
    for (i, &(s1, s2)) in PAIR_SEEDS.iter().enumerate() {
        pairs.push(create_pair(i, s1, s2));
    }
    println!();

    let mut best_ever: Vec<f64> = vec![0.0; N_PAIRS];

    let mut records: Vec<GenRecord> = Vec::new();

    // Track peak accuracy per pair and overall
    let mut peak_per_pair: Vec<f64> = vec![0.0; N_PAIRS];
    let mut peak_overall = 0.0f64;

    // Track cross-breed impact
    let mut cross_breed_count = 0usize;
    let mut cross_breed_improvements: Vec<f64> = Vec::new();

    let total_start = Instant::now();

    for gen in 0..GENERATIONS {
        let gen_start = Instant::now();
        println!(
            "\n================================================================"
        );
        println!(
            "  GENERATION {} / {}",
            gen,
            GENERATIONS - 1,
        );
        for pair in &pairs {
            let first = pair.first.as_ref().unwrap();
            let second = pair.second.as_ref().unwrap();
            println!(
                "  {} [idx={}]: {} (older: {}) + {}",
                pair.label,
                pair.pair_idx,
                first.name,
                if pair.first_is_older { "yes" } else { "no" },
                second.name,
            );
        }
        println!(
            "================================================================\n"
        );

        // =================================================================
        // Phase 1: PARALLEL evolution of ALL individuals (rayon par_iter_mut)
        // =================================================================
        println!("  >>> PHASE 1: Parallel evolution of all {} individuals <<<\n", N_PAIRS * 2);

        // Take all individuals out of pairs for parallel processing
        let mut work: Vec<(Individual, String, usize)> = Vec::with_capacity(N_PAIRS * 2);
        for pair in pairs.iter_mut() {
            let first = pair.first.take().unwrap();
            let second = pair.second.take().unwrap();
            work.push((first, pair.label.clone(), pair.pair_idx));
            work.push((second, pair.label.clone(), pair.pair_idx));
        }

        // best_ever clones for parallel -- each individual gets its own copy
        // (we merge maximums after)
        let corpus_ref = &corpus;
        let prop_ref = &prop;
        let best_ever_snapshot = best_ever.clone();

        // Evolve ALL in parallel
        let results: Vec<(Individual, f64)> = work
            .into_par_iter()
            .map(|(mut ind, pair_label, pair_idx)| {
                let mut local_best = best_ever_snapshot[pair_idx];
                evolve_individual(
                    &mut ind,
                    corpus_ref,
                    prop_ref,
                    &pair_label,
                    &mut local_best,
                );
                (ind, local_best)
            })
            .collect();

        // Put individuals back into pairs and merge best_ever
        // Move individuals back into pairs and merge best_ever
        let mut result_iter = results.into_iter();
        for (i, pair) in pairs.iter_mut().enumerate() {
            let (first_ind, first_best) = result_iter.next().unwrap();
            let (second_ind, second_best) = result_iter.next().unwrap();
            best_ever[i] = best_ever[i].max(first_best).max(second_best);
            pair.first = Some(first_ind);
            pair.second = Some(second_ind);
        }

        // =================================================================
        // Phase 2: Within-pair breed (sequential across pairs)
        // =================================================================
        println!("\n  >>> PHASE 2: Within-pair breeding ({} pairs) <<<\n", N_PAIRS);

        let step_num = (gen + 1) * STEPS_PER_GEN;
        let h = total_neurons();
        let os = output_start();

        // Collect child results for GenRecord
        let mut pair_results: Vec<PairResult> = Vec::with_capacity(N_PAIRS);

        for pair_idx in 0..N_PAIRS {
            let pair = &mut pairs[pair_idx];
            let first = pair.first.as_ref().unwrap();
            let second = pair.second.as_ref().unwrap();

            // Save pre-breed checkpoints
            do_save_checkpoint(
                &format!("pair{}_gen{}_{}_pre_breed", pair_idx, gen, first.name),
                &first.net,
                &first.proj,
                step_num,
                first.accuracy,
            );
            do_save_checkpoint(
                &format!("pair{}_gen{}_{}_pre_breed", pair_idx, gen, second.name),
                &second.net,
                &second.proj,
                step_num,
                second.accuracy,
            );

            // Breed
            let (s1, s2) = PAIR_SEEDS[pair_idx];
            let mut breed_rng = StdRng::seed_from_u64(
                s1 + s2 + gen as u64 * 7919,
            );
            let breed_sdr = make_sdr();
            let (child_net, child_proj) = breed(
                &first.net,
                &second.net,
                &first.proj,
                &first.name,
                &second.name,
                &corpus,
                &breed_sdr,
                &prop,
                gen,
                &mut breed_rng,
                &format!("WITHIN-PAIR BREED ({})", pair.label),
            );

            // Eval child
            let mut child_eval_rng = StdRng::seed_from_u64(SDR_SEED + 3333 + pair_idx as u64 * 1000 + gen as u64);
            let mut child_net_eval = child_net.clone();
            let child_sdr = make_sdr();
            let child_acc = eval_accuracy(
                &mut child_net_eval,
                &child_proj,
                &corpus,
                EVAL_LEN_LONG,
                &mut child_eval_rng,
                &child_sdr,
                &prop,
                os,
                h,
            );
            let child_edges = child_net.edge_count();
            if child_acc > best_ever[pair_idx] {
                best_ever[pair_idx] = child_acc;
            }

            do_save_checkpoint(
                &format!("pair{}_gen{}_child", pair_idx, gen),
                &child_net,
                &child_proj,
                step_num,
                child_acc,
            );

            let child_name = format!("Child_P{}_gen{}", pair_idx, gen);
            println!(
                "    Child ({}) accuracy: {:.2}%  edges={}",
                child_name,
                child_acc * 100.0,
                child_edges
            );

            // Record results before replacement
            let first_name = first.name.clone();
            let first_acc = first.accuracy;
            let second_name = second.name.clone();
            let second_acc = second.accuracy;

            pair_results.push(PairResult {
                pair_label: pair.label.clone(),
                first_name: first_name.clone(),
                first_acc,
                second_name: second_name.clone(),
                second_acc,
                child_acc,
                child_edges,
            });

            // Replace older parent
            let new_seed = s1 + s2 + (gen as u64 + 1) * 10000;
            let child_individual = Individual {
                name: child_name.clone(),
                net: child_net,
                proj: child_proj,
                sdr: make_sdr(),
                mut_rng: StdRng::seed_from_u64(new_seed),
                eval_rng: StdRng::seed_from_u64(new_seed + 1000),
                accuracy: child_acc,
                generation_born: gen,
                edges: child_edges,
            };

            if pair.first_is_older {
                let older_name = &pair.first.as_ref().unwrap().name;
                let older_acc = pair.first.as_ref().unwrap().accuracy;
                println!(
                    "    Replacing {} (older) with {} in {}",
                    older_name, child_name, pair.label
                );
                println!(
                    "      {} {:.2}% -> {} {:.2}% (delta={:+.2}pp)",
                    older_name,
                    older_acc * 100.0,
                    child_name,
                    child_acc * 100.0,
                    (child_acc - older_acc) * 100.0,
                );
                pair.first = Some(child_individual);
            } else {
                let older_name = &pair.second.as_ref().unwrap().name;
                let older_acc = pair.second.as_ref().unwrap().accuracy;
                println!(
                    "    Replacing {} (older) with {} in {}",
                    older_name, child_name, pair.label
                );
                println!(
                    "      {} {:.2}% -> {} {:.2}% (delta={:+.2}pp)",
                    older_name,
                    older_acc * 100.0,
                    child_name,
                    child_acc * 100.0,
                    (child_acc - older_acc) * 100.0,
                );
                pair.second = Some(child_individual);
            }
            pair.first_is_older = !pair.first_is_older;
        }

        // =================================================================
        // Phase 3: Cross-breed (every CROSS_BREED_INTERVAL generations)
        // =================================================================
        let mut cross_happened = false;
        let mut hybrid_acc_final = 0.0;
        let mut hybrid_replaced_name = String::new();

        if (gen + 1) % CROSS_BREED_INTERVAL == 0 {
            println!("\n  >>> PHASE 3: CROSS-PAIR BREED (gen {}) <<<\n", gen);
            cross_happened = true;
            cross_breed_count += 1;

            // Find best individual per pair
            let mut pair_bests: Vec<(usize, f64, String)> = pairs
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let f = p.first.as_ref().unwrap();
                    let s = p.second.as_ref().unwrap();
                    if f.accuracy >= s.accuracy {
                        (i, f.accuracy, f.name.clone())
                    } else {
                        (i, s.accuracy, s.name.clone())
                    }
                })
                .collect();
            pair_bests.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let (pair_idx_1, best_1_acc, best_1_name) = &pair_bests[0];
            let (pair_idx_2, best_2_acc, best_2_name) = &pair_bests[1];

            // Get references to the best individuals from top-2 pairs
            let p1 = &pairs[*pair_idx_1];
            let p1_first = p1.first.as_ref().unwrap();
            let p1_second = p1.second.as_ref().unwrap();
            let best_1_ind = if p1_first.accuracy >= p1_second.accuracy {
                p1_first
            } else {
                p1_second
            };

            let p2 = &pairs[*pair_idx_2];
            let p2_first = p2.first.as_ref().unwrap();
            let p2_second = p2.second.as_ref().unwrap();
            let best_2_ind = if p2_first.accuracy >= p2_second.accuracy {
                p2_first
            } else {
                p2_second
            };

            println!(
                "    Best from {} (pair {}): {} ({:.2}%, {} edges)",
                pairs[*pair_idx_1].label,
                pair_idx_1,
                best_1_name,
                best_1_acc * 100.0,
                best_1_ind.net.edge_count(),
            );
            println!(
                "    Best from {} (pair {}): {} ({:.2}%, {} edges)",
                pairs[*pair_idx_2].label,
                pair_idx_2,
                best_2_name,
                best_2_acc * 100.0,
                best_2_ind.net.edge_count(),
            );

            // Jaccard between cross-pair candidates
            let cross_a_edges = edge_set(&best_1_ind.net);
            let cross_b_edges = edge_set(&best_2_ind.net);
            let cross_jaccard = jaccard(&cross_a_edges, &cross_b_edges);
            let za = pocket_zone(0);
            let zb = pocket_zone(1);
            let ca_0 = edge_set_in_zone(&best_1_ind.net, &za);
            let cb_0 = edge_set_in_zone(&best_2_ind.net, &za);
            let ca_1 = edge_set_in_zone(&best_1_ind.net, &zb);
            let cb_1 = edge_set_in_zone(&best_2_ind.net, &zb);
            let cross_j0 = jaccard(&ca_0, &cb_0);
            let cross_j1 = jaccard(&ca_1, &cb_1);
            let cross_shared = cross_a_edges.intersection(&cross_b_edges).count();
            let cross_union_count = cross_a_edges.union(&cross_b_edges).count();

            println!(
                "    Cross-pair Jaccard: {:.3} (pocket0={:.3} pocket1={:.3}) shared={} union={}",
                cross_jaccard, cross_j0, cross_j1, cross_shared, cross_union_count,
            );

            // Save pre-cross-breed checkpoints
            do_save_checkpoint(
                &format!("gen{}_cross_best_pair{}_{}", gen, pair_idx_1, best_1_name),
                &best_1_ind.net,
                &best_1_ind.proj,
                step_num,
                *best_1_acc,
            );
            do_save_checkpoint(
                &format!("gen{}_cross_best_pair{}_{}", gen, pair_idx_2, best_2_name),
                &best_2_ind.net,
                &best_2_ind.proj,
                step_num,
                *best_2_acc,
            );

            // Use a shared SDR for the cross breed
            let cross_sdr = make_sdr();

            let (s1_1, s2_1) = PAIR_SEEDS[*pair_idx_1];
            let (s1_2, s2_2) = PAIR_SEEDS[*pair_idx_2];
            let mut cross_breed_rng = StdRng::seed_from_u64(
                s1_1 + s1_2 + s2_1 + s2_2 + gen as u64 * 13331,
            );

            let (hybrid_net, hybrid_proj) = breed(
                &best_1_ind.net,
                &best_2_ind.net,
                &best_1_ind.proj,
                best_1_name,
                best_2_name,
                &corpus,
                &cross_sdr,
                &prop,
                gen,
                &mut cross_breed_rng,
                "CROSS-PAIR BREED",
            );

            // Eval hybrid
            let mut hybrid_eval_rng = StdRng::seed_from_u64(SDR_SEED + 7777 + gen as u64);
            let mut hybrid_net_eval = hybrid_net.clone();
            let hybrid_sdr = make_sdr();
            let hybrid_acc = eval_accuracy(
                &mut hybrid_net_eval,
                &hybrid_proj,
                &corpus,
                EVAL_LEN_LONG,
                &mut hybrid_eval_rng,
                &hybrid_sdr,
                &prop,
                os,
                h,
            );
            hybrid_acc_final = hybrid_acc;
            let hybrid_edges = hybrid_net.edge_count();

            let hybrid_name = format!("Hybrid_gen{}", gen);

            println!(
                "    Hybrid ({}) accuracy: {:.2}%  edges={}",
                hybrid_name,
                hybrid_acc * 100.0,
                hybrid_edges,
            );

            do_save_checkpoint(
                &format!("gen{}_hybrid", gen),
                &hybrid_net,
                &hybrid_proj,
                step_num,
                hybrid_acc,
            );

            // Find the WORST individual across ALL pairs
            let mut candidates: Vec<(String, f64, usize, &str)> = Vec::new();
            for (pi, pair) in pairs.iter().enumerate() {
                let f = pair.first.as_ref().unwrap();
                let s = pair.second.as_ref().unwrap();
                candidates.push((f.name.clone(), f.accuracy, pi, "first"));
                candidates.push((s.name.clone(), s.accuracy, pi, "second"));
            }

            let worst_idx = candidates
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            let (ref worst_name, worst_acc, worst_pair_idx, worst_slot) = candidates[worst_idx];
            let improvement = hybrid_acc - worst_acc;
            cross_breed_improvements.push(improvement);

            println!(
                "\n    WORST individual: {} ({:.2}%) in Pair {} [{}]",
                worst_name,
                worst_acc * 100.0,
                worst_pair_idx,
                worst_slot,
            );
            println!(
                "    Replacing {} with {} ({:.2}% -> {:.2}%, delta={:+.2}pp)",
                worst_name,
                hybrid_name,
                worst_acc * 100.0,
                hybrid_acc * 100.0,
                improvement * 100.0,
            );

            hybrid_replaced_name = worst_name.clone();

            // Build the replacement Individual
            let hybrid_seed = SDR_SEED + gen as u64 * 20000 + 5000;
            let hybrid_individual = Individual {
                name: hybrid_name,
                net: hybrid_net,
                proj: hybrid_proj,
                sdr: make_sdr(),
                mut_rng: StdRng::seed_from_u64(hybrid_seed),
                eval_rng: StdRng::seed_from_u64(hybrid_seed + 1000),
                accuracy: hybrid_acc,
                generation_born: gen,
                edges: hybrid_edges,
            };

            let target_pair = &mut pairs[worst_pair_idx];
            match worst_slot {
                "first" => target_pair.first = Some(hybrid_individual),
                "second" => target_pair.second = Some(hybrid_individual),
                _ => unreachable!(),
            }

            // Update best_ever for all pairs with hybrid
            for be in best_ever.iter_mut() {
                if hybrid_acc > *be {
                    *be = hybrid_acc;
                }
            }
        }

        // =================================================================
        // Per-generation summary
        // =================================================================
        let gen_elapsed = gen_start.elapsed();

        // Update peaks
        let mut gen_best = 0.0f64;
        for (i, pair) in pairs.iter().enumerate() {
            let f = pair.first.as_ref().unwrap();
            let s = pair.second.as_ref().unwrap();
            let pair_best = f.accuracy.max(s.accuracy);
            if pair_best > peak_per_pair[i] {
                peak_per_pair[i] = pair_best;
            }
            if pair_best > gen_best {
                gen_best = pair_best;
            }
        }
        if gen_best > peak_overall {
            peak_overall = gen_best;
        }

        println!(
            "\n  +----------- Gen {} Summary -----------+",
            gen
        );
        for (i, pair) in pairs.iter().enumerate() {
            let f = pair.first.as_ref().unwrap();
            let s = pair.second.as_ref().unwrap();
            println!(
                "  | {} [{}]: {} {:.2}%  {} {:.2}%",
                pair.label,
                i,
                f.name,
                f.accuracy * 100.0,
                s.name,
                s.accuracy * 100.0,
            );
            println!(
                "  |   edges: {} / {}   born: gen {} / gen {}",
                f.edges,
                s.edges,
                f.generation_born,
                s.generation_born,
            );
        }
        println!("  | Within-pair children:");
        for pr in &pair_results {
            println!(
                "  |   {}: {:.2}% ({} edges)",
                pr.pair_label, pr.child_acc * 100.0, pr.child_edges
            );
        }
        if cross_happened {
            println!(
                "  | CROSS-BREED: hybrid={:.2}%  replaced={}",
                hybrid_acc_final * 100.0,
                hybrid_replaced_name,
            );
        }
        println!(
            "  | Best overall: {:.2}%",
            peak_overall * 100.0,
        );
        for (i, peak) in peak_per_pair.iter().enumerate() {
            println!(
                "  |   Pair {} peak: {:.2}%",
                i,
                peak * 100.0,
            );
        }
        println!(
            "  | Time: {:.1}s",
            gen_elapsed.as_secs_f64()
        );
        println!(
            "  +--------------------------------------+\n"
        );

        records.push(GenRecord {
            gen,
            pair_results,
            cross_breed_happened: cross_happened,
            hybrid_acc: hybrid_acc_final,
            hybrid_replaced: hybrid_replaced_name.clone(),
            best_overall: peak_overall,
        });
    }

    let total_elapsed = total_start.elapsed();

    // =====================================================================
    // Final summary
    // =====================================================================
    println!(
        "\n================================================================"
    );
    println!("  FINAL SUMMARY -- POCKET CROSS-BREED EXPERIMENT (N={})", N_PAIRS);
    println!(
        "================================================================\n"
    );

    // History table per pair
    for pair_idx in 0..N_PAIRS {
        println!("  --- Generation History: Pair {} ---\n", pair_idx);
        println!(
            "  {:<4} {:>18} {:>8} {:>18} {:>8} {:>9} {:>8}",
            "Gen", "First", "acc%", "Second", "acc%", "Child%", "Edges"
        );
        println!("  {}", "-".repeat(80));
        for r in &records {
            if pair_idx < r.pair_results.len() {
                let pr = &r.pair_results[pair_idx];
                println!(
                    "  {:<4} {:>18} {:>7.2}% {:>18} {:>7.2}% {:>8.2}% {:>8}",
                    r.gen,
                    pr.first_name,
                    pr.first_acc * 100.0,
                    pr.second_name,
                    pr.second_acc * 100.0,
                    pr.child_acc * 100.0,
                    pr.child_edges,
                );
            }
        }
        println!("  {}", "-".repeat(80));
        println!();
    }

    // Cross-breed and overall table
    println!("  --- Cross-Breed & Overall ---\n");
    println!(
        "  {:<4} {:>6} {:>9} {:>18} {:>10}",
        "Gen", "Cross?", "Hybrid%", "Replaced", "BestAll%"
    );
    println!("  {}", "-".repeat(55));
    for r in &records {
        println!(
            "  {:<4} {:>6} {:>9} {:>18} {:>9.2}%",
            r.gen,
            if r.cross_breed_happened { "YES" } else { "no" },
            if r.cross_breed_happened {
                format!("{:.2}%", r.hybrid_acc * 100.0)
            } else {
                "---".to_string()
            },
            if r.cross_breed_happened {
                r.hybrid_replaced.clone()
            } else {
                "---".to_string()
            },
            r.best_overall * 100.0,
        );
    }
    println!("  {}", "-".repeat(55));

    // Peak accuracy
    println!("\n  --- Peak Accuracy ---");
    for (i, peak) in peak_per_pair.iter().enumerate() {
        println!(
            "  Pair {} peak: {:.2}%",
            i,
            peak * 100.0
        );
    }
    println!(
        "  Overall peak: {:.2}%",
        peak_overall * 100.0
    );

    // Cross-breed impact
    println!("\n  --- Cross-Breed Impact ---");
    println!(
        "  Cross-breeds performed: {} / {} possible",
        cross_breed_count,
        GENERATIONS / CROSS_BREED_INTERVAL
    );
    if !cross_breed_improvements.is_empty() {
        let avg_improvement: f64 =
            cross_breed_improvements.iter().sum::<f64>() / cross_breed_improvements.len() as f64;
        let positive_count = cross_breed_improvements
            .iter()
            .filter(|&&x| x > 0.0)
            .count();
        println!(
            "  Improvements: {}/{} were positive",
            positive_count,
            cross_breed_improvements.len()
        );
        println!(
            "  Average improvement: {:+.2}pp",
            avg_improvement * 100.0
        );
        for (i, &imp) in cross_breed_improvements.iter().enumerate() {
            println!(
                "    Cross-breed {}: {:+.2}pp vs replaced individual",
                i + 1,
                imp * 100.0,
            );
        }
    } else {
        println!("  No cross-breeds were performed.");
    }

    // Lineage trace
    println!("\n  --- Final Lineage ---");
    let mut final_best_name = String::new();
    let mut final_best_acc = 0.0f64;
    for (i, pair) in pairs.iter().enumerate() {
        let f = pair.first.as_ref().unwrap();
        let s = pair.second.as_ref().unwrap();
        for (indiv, slot_name) in [(f, "slot 1"), (s, "slot 2")] {
            let is_hybrid = indiv.name.starts_with("Hybrid");
            let is_child = indiv.name.starts_with("Child");
            let origin = if is_hybrid {
                "cross-breed"
            } else if is_child {
                "within-pair breed"
            } else {
                "original"
            };
            println!(
                "  Pair {} {}: {} ({:.2}%, {} edges, born gen {}, origin: {})",
                i,
                slot_name,
                indiv.name,
                indiv.accuracy * 100.0,
                indiv.edges,
                indiv.generation_born,
                origin,
            );
            if indiv.accuracy > final_best_acc {
                final_best_acc = indiv.accuracy;
                final_best_name = indiv.name.clone();
            }
        }
    }
    println!(
        "\n  BEST FINAL INDIVIDUAL: {} ({:.2}%)",
        final_best_name,
        final_best_acc * 100.0
    );

    // Did cross-breeding help?
    let within_only_peak = records
        .iter()
        .flat_map(|r| r.pair_results.iter().map(|pr| pr.child_acc))
        .fold(0.0f64, f64::max);
    let hybrid_peak = records
        .iter()
        .filter(|r| r.cross_breed_happened)
        .map(|r| r.hybrid_acc)
        .fold(0.0f64, f64::max);
    println!(
        "\n  Within-pair-only peak child accuracy: {:.2}%",
        within_only_peak * 100.0
    );
    if hybrid_peak > 0.0 {
        println!(
            "  Cross-breed hybrid peak accuracy:     {:.2}%",
            hybrid_peak * 100.0
        );
        if hybrid_peak > within_only_peak {
            println!(
                "  --> Cross-breeding IMPROVED over within-pair by {:.2}pp",
                (hybrid_peak - within_only_peak) * 100.0
            );
        } else {
            println!(
                "  --> Cross-breeding did NOT exceed within-pair peak (delta={:+.2}pp)",
                (hybrid_peak - within_only_peak) * 100.0
            );
        }
    }

    println!(
        "\n  Total time: {:.1}s ({:.1}s per generation)\n",
        total_elapsed.as_secs_f64(),
        total_elapsed.as_secs_f64() / GENERATIONS as f64,
    );
}
