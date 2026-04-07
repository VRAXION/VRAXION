//! Pocket breed: continuous breeding cycle between two 2-pocket networks.
//!
//! Woman (stable, seed 42) and Male (reactive, seed 123) evolve independently
//! for STEPS_PER_GEN steps, then breed via edge union + crystallize pruning.
//! The child replaces the older parent each generation.
//!
//! Run: cargo run --example pocket_breed --release -- <corpus-path>

use instnct_core::{load_corpus, 
    save_checkpoint, CheckpointMeta, Int8Projection, Network, PropagationConfig, SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
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

const GENERATIONS: usize = 5;
const STEPS_PER_GEN: usize = 30_000;
const EVAL_LEN_SHORT: usize = 100;
const EVAL_LEN_LONG: usize = 2000;
const LOG_INTERVAL: usize = 5000;

const CRYSTALLIZE_MAX_ROUNDS: usize = 5;
const CRYSTALLIZE_MAX_REMOVAL_PCT: f64 = 0.30;
const CRYSTALLIZE_DROP_THRESHOLD_PP: f64 = 0.02;

const WOMAN_SEED: u64 = 42;
const MALE_SEED: u64 = 123;

const CHECKPOINT_DIR: &str = "checkpoints/pocket_breed";

// ---------------------------------------------------------------------------
// Pocket geometry (same as pocket_chain.rs)
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
    POCKET_PHI // 158 — first pocket's input zone
}

fn output_start() -> usize {
    let last = pocket_zone(N_POCKETS - 1);
    last.start + (POCKET_H - POCKET_PHI) // 196 + 98 = 294
}

fn out_dim() -> usize {
    total_neurons() - output_start() // 452 - 294 = 158
}

// ---------------------------------------------------------------------------
// Corpus loading
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Pocket mutations (from pocket_chain.rs)
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

/// Evaluate on a fixed corpus segment (for crystallize — deterministic).
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
// Edge statistics helpers
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
    let union = a.union(b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

fn bar(val: f64, max_val: f64, width: usize) -> String {
    let filled = ((val / max_val) * width as f64).round().min(width as f64) as usize;
    format!("{}{}", "#".repeat(filled), ".".repeat(width - filled))
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
            net.threshold_mut()[i] = rng.gen_range(0..=7u8);
            net.channel_mut()[i] = rng.gen_range(1..=8u8);
            if rng.gen_ratio(1, 10) {
                net.polarity_mut()[i] = -1;
            }
        }
    }

    let es = count_edges(&net);
    println!(
        "  [init] seed={} H={} edges={} (pA={} pB={} cross={})",
        seed,
        h,
        es.total,
        es.pocket_a,
        es.pocket_b,
        es.cross
    );
    net
}

// ---------------------------------------------------------------------------
// Evolution: run STEPS_PER_GEN mutation steps on a parent
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn evolve_parent(
    name: &str,
    net: &mut Network,
    proj: &mut Int8Projection,
    corpus: &[u8],
    sdr: &SdrTable,
    prop: &PropagationConfig,
    mut_rng: &mut StdRng,
    eval_rng: &mut StdRng,
    best_ever: &mut f64,
) -> (f64, usize) {
    let h = total_neurons();
    let os = output_start();
    let mut accepted = 0u32;
    let mut total_tried = 0u32;
    let mut last_log_accepted = 0u32;
    let mut last_log_total = 0u32;

    println!("  --- Evolving {} for {} steps ---", name, STEPS_PER_GEN);

    for step in 0..STEPS_PER_GEN {
        // Paired eval
        let snap = eval_rng.clone();
        let before = eval_accuracy(net, proj, corpus, EVAL_LEN_SHORT, eval_rng, sdr, prop, os, h);
        *eval_rng = snap;

        let state = net.save_state();
        // Pick random pocket, mutate within it
        let pocket_idx = mut_rng.gen_range(0..N_POCKETS);
        let zone = pocket_zone(pocket_idx);
        let is_last = pocket_idx == N_POCKETS - 1;

        // Save W backup before mutating (pocket_mutate may mutate W for last pocket)
        let proj_clone = proj.clone();
        let mutated = pocket_mutate(net, proj, &zone, mut_rng, is_last);
        let w_backup_option = if mutated {
            Some(proj_clone)
        } else {
            None
        };

        if !mutated {
            let _ = eval_accuracy(net, proj, corpus, EVAL_LEN_SHORT, eval_rng, sdr, prop, os, h);
            continue;
        }

        total_tried += 1;

        let after = eval_accuracy(net, proj, corpus, EVAL_LEN_SHORT, eval_rng, sdr, prop, os, h);
        let accept = after > before;

        if accept {
            accepted += 1;
        } else {
            net.restore_state(&state);
            if let Some(backup) = w_backup_option {
                *proj = backup;
            }
        }

        // Periodic logging
        if (step + 1) % LOG_INTERVAL == 0 {
            let mut cr = StdRng::seed_from_u64(WOMAN_SEED + 6000 + step as u64);
            let acc = eval_accuracy(net, proj, corpus, EVAL_LEN_LONG, &mut cr, sdr, prop, os, h);
            if acc > *best_ever {
                *best_ever = acc;
            }

            let es = count_edges(net);
            let ps = pocket_param_stats(net);
            let interval_accepted = accepted - last_log_accepted;
            let interval_total = total_tried - last_log_total;
            let accept_rate = if interval_total > 0 {
                interval_accepted as f64 / interval_total as f64
            } else {
                0.0
            };

            println!(
                "  [{:>6}] {} |{}| {:.1}%  edges={} (pA={} pB={} x={})  \
                 inh=[{},{}] theta=[{:.1},{:.1}]  accept={:.1}%  best_ever={:.1}%",
                step + 1,
                name,
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
    let mut fr = StdRng::seed_from_u64(WOMAN_SEED + 9999);
    let final_acc = eval_accuracy(net, proj, corpus, EVAL_LEN_LONG, &mut fr, sdr, prop, os, h);
    if final_acc > *best_ever {
        *best_ever = final_acc;
    }
    let final_edges = net.edge_count();

    let overall_rate = if total_tried > 0 {
        accepted as f64 / total_tried as f64
    } else {
        0.0
    };
    println!(
        "  {} FINAL: {:.2}%  edges={}  accepted={}/{}({:.1}%)",
        name,
        final_acc * 100.0,
        final_edges,
        accepted,
        total_tried,
        overall_rate * 100.0,
    );

    (final_acc, final_edges)
}

// ---------------------------------------------------------------------------
// Breeding: union edges + copy params + crystallize
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn breed(
    woman: &Network,
    male: &Network,
    _woman_proj: &Int8Projection,
    corpus: &[u8],
    sdr: &SdrTable,
    prop: &PropagationConfig,
    gen: usize,
    breed_rng: &mut StdRng,
) -> (Network, Int8Projection) {
    let h = total_neurons();
    let os = output_start();

    println!("\n  === BREEDING (gen {}) ===", gen);

    // --- Edge overlap analysis ---
    let woman_edges = edge_set(woman);
    let male_edges = edge_set(male);
    let total_jaccard = jaccard(&woman_edges, &male_edges);
    let shared = woman_edges.intersection(&male_edges).count();

    let za = pocket_zone(0);
    let zb = pocket_zone(1);
    let we_a = edge_set_in_zone(woman, &za);
    let me_a = edge_set_in_zone(male, &za);
    let we_b = edge_set_in_zone(woman, &zb);
    let me_b = edge_set_in_zone(male, &zb);
    let jaccard_a = jaccard(&we_a, &me_a);
    let jaccard_b = jaccard(&we_b, &me_b);

    println!(
        "  Edge overlap: shared={} Jaccard={:.3} (pA={:.3} pB={:.3})",
        shared, total_jaccard, jaccard_a, jaccard_b
    );
    println!(
        "  Woman edges: {} (pA={} pB={})",
        woman.edge_count(),
        we_a.len(),
        we_b.len()
    );
    println!(
        "  Male edges:  {} (pA={} pB={})",
        male.edge_count(),
        me_a.len(),
        me_b.len()
    );

    // --- Step 1: Create child with union of all edges ---
    let mut child = Network::new(h);

    // Add all Woman edges
    let mut added_from_woman = 0usize;
    for e in woman.graph().iter_edges() {
        if child.graph_mut().add_edge(e.source, e.target) {
            added_from_woman += 1;
        }
    }

    // Add all Male edges (skip duplicates)
    let mut added_from_male = 0usize;
    let mut skipped_dups = 0usize;
    for e in male.graph().iter_edges() {
        if child.graph_mut().add_edge(e.source, e.target) {
            added_from_male += 1;
        } else {
            skipped_dups += 1;
        }
    }

    let union_edges = child.edge_count();
    println!(
        "  Union: {} edges (from_woman={} from_male={} dups_skipped={})",
        union_edges, added_from_woman, added_from_male, skipped_dups
    );

    // --- Step 2: Copy Woman's parameters ---
    for i in 0..h {
        child.threshold_mut()[i] = woman.threshold()[i];
        child.channel_mut()[i] = woman.channel()[i];
        child.polarity_mut()[i] = woman.polarity()[i];
    }
    println!("  Copied Woman's neuron parameters (threshold/channel/polarity)");

    // --- Step 3: Fresh W projection ---
    let mut child_proj = Int8Projection::new(out_dim(), CHARS, breed_rng);
    println!(
        "  Fresh W projection: {}x{} ({} weights)",
        out_dim(),
        CHARS,
        child_proj.weight_count()
    );

    // --- Step 4: Crystallize (iterative edge pruning) ---
    println!("  --- Crystallize ---");

    // Pick a fixed corpus segment for crystallize eval
    let cryst_off = breed_rng.gen_range(0..=corpus.len() - EVAL_LEN_LONG - 1);
    let cryst_seg = &corpus[cryst_off..cryst_off + EVAL_LEN_LONG + 1];

    // Train the child's W a little before crystallize (quick projection tuning)
    // so that eval is meaningful
    {
        let mut tune_rng = StdRng::seed_from_u64(WOMAN_SEED + gen as u64 * 1000 + 777);
        let mut tune_eval_rng = StdRng::seed_from_u64(WOMAN_SEED + gen as u64 * 1000 + 888);
        println!("  Quick W tuning (1000 steps) before crystallize...");
        let mut tune_accepted = 0u32;
        for _ in 0..1000 {
            let snap = tune_eval_rng.clone();
            let before = eval_accuracy(
                &mut child,
                &child_proj,
                corpus,
                EVAL_LEN_SHORT,
                &mut tune_eval_rng,
                sdr,
                prop,
                os,
                h,
            );
            tune_eval_rng = snap;

            let backup = child_proj.mutate_one(&mut tune_rng);
            let after = eval_accuracy(
                &mut child,
                &child_proj,
                corpus,
                EVAL_LEN_SHORT,
                &mut tune_eval_rng,
                sdr,
                prop,
                os,
                h,
            );
            if after > before {
                tune_accepted += 1;
            } else {
                child_proj.rollback(backup);
            }
        }
        println!("  W tune done: accepted {}/1000", tune_accepted);
    }

    let baseline_acc =
        eval_accuracy_segment(&mut child, &child_proj, cryst_seg, sdr, prop, os, h);
    println!(
        "  Crystallize baseline: {:.2}% ({} edges)",
        baseline_acc * 100.0,
        child.edge_count()
    );

    let mut current_acc = baseline_acc;

    for round in 0..CRYSTALLIZE_MAX_ROUNDS {
        let edges_before = child.edge_count();
        let max_removable = (edges_before as f64 * CRYSTALLIZE_MAX_REMOVAL_PCT) as usize;

        // Collect all edges, try removing each
        let all_edges: Vec<_> = child.graph().iter_edges().collect();

        // Shuffle for fairness
        let mut edge_indices: Vec<usize> = (0..all_edges.len()).collect();
        for i in (1..edge_indices.len()).rev() {
            let j = breed_rng.gen_range(0..=i);
            edge_indices.swap(i, j);
        }

        let mut removable: Vec<(u16, u16)> = Vec::new();
        let mut tested = 0usize;

        for &idx in &edge_indices {
            if removable.len() >= max_removable {
                break;
            }
            let e = all_edges[idx];

            // Temporarily remove
            child.graph_mut().remove_edge(e.source, e.target);
            let trial_acc =
                eval_accuracy_segment(&mut child, &child_proj, cryst_seg, sdr, prop, os, h);

            if trial_acc >= current_acc {
                // Safe to remove — accuracy didn't drop
                removable.push((e.source, e.target));
            }
            // Restore for next test
            child.graph_mut().add_edge(e.source, e.target);
            tested += 1;
        }

        if removable.is_empty() {
            println!(
                "    Round {}: tested {} edges, none removable. Stopping.",
                round, tested
            );
            break;
        }

        // Batch remove all marked edges
        for &(src, tgt) in &removable {
            child.graph_mut().remove_edge(src, tgt);
        }

        let edges_after = child.edge_count();
        let after_acc =
            eval_accuracy_segment(&mut child, &child_proj, cryst_seg, sdr, prop, os, h);

        println!(
            "    Round {}: tested={} removable={} edges {}->{} acc {:.2}%->{:.2}%",
            round,
            tested,
            removable.len(),
            edges_before,
            edges_after,
            current_acc * 100.0,
            after_acc * 100.0,
        );

        // Check if accuracy dropped too much from baseline
        if after_acc < baseline_acc - CRYSTALLIZE_DROP_THRESHOLD_PP {
            // Undo this round — re-add removed edges
            println!(
                "    Accuracy dropped {:.2}pp below baseline ({:.2}%), undoing round.",
                (baseline_acc - after_acc) * 100.0,
                baseline_acc * 100.0,
            );
            for &(src, tgt) in &removable {
                child.graph_mut().add_edge(src, tgt);
            }
            break;
        }

        current_acc = after_acc;
    }

    let final_es = count_edges(&child);
    let child_acc = eval_accuracy_segment(&mut child, &child_proj, cryst_seg, sdr, prop, os, h);
    println!(
        "  Crystallize done: {:.2}% edges={} (pA={} pB={} x={}) (was {} before)",
        child_acc * 100.0,
        final_es.total,
        final_es.pocket_a,
        final_es.pocket_b,
        final_es.cross,
        union_edges,
    );

    // Initial eval on standard rng
    let mut init_rng = StdRng::seed_from_u64(WOMAN_SEED + 5555 + gen as u64);
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
        "  Child initial accuracy (2000 char, rng eval): {:.2}%\n",
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
        Ok(()) => println!("  [checkpoint] saved: {}", path),
        Err(e) => eprintln!("  [checkpoint] FAILED to save {}: {}", path, e),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });

    println!("=== Pocket Breed Experiment ===");
    println!("  Generations: {}", GENERATIONS);
    println!("  Steps/gen:   {}", STEPS_PER_GEN);
    println!("  Pockets:     {} (H={})", N_POCKETS, total_neurons());
    println!(
        "  SDR input:   [0..{})",
        sdr_input_end()
    );
    println!(
        "  W output:    [{}..{})",
        output_start(),
        total_neurons()
    );
    println!(
        "  Out dim:     {} (W: {}x{})",
        out_dim(),
        out_dim(),
        CHARS
    );
    println!("  Woman seed:  {}", WOMAN_SEED);
    println!("  Male seed:   {}", MALE_SEED);
    println!();

    println!("Loading corpus from: {}", corpus_path);
    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars loaded\n", corpus.len());

    let h = total_neurons();
    let os = output_start();

    // Shared SDR (same seed for both parents)
    let sdr = SdrTable::new(
        CHARS,
        h,
        sdr_input_end(),
        SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(WOMAN_SEED + 100),
    )
    .unwrap();

    let prop = PropagationConfig {
        ticks_per_token: 6,
        input_duration_ticks: 2,
        decay_interval_ticks: 6,
        use_refractory: false,
    };

    // --- Initialize parents ---
    println!("--- Initializing Woman (seed {}) ---", WOMAN_SEED);
    let mut woman_rng = StdRng::seed_from_u64(WOMAN_SEED);
    let mut woman_net = init_network(WOMAN_SEED, &mut woman_rng);
    let mut woman_proj = Int8Projection::new(
        out_dim(),
        CHARS,
        &mut StdRng::seed_from_u64(WOMAN_SEED + 200),
    );

    println!("--- Initializing Male (seed {}) ---", MALE_SEED);
    let mut male_rng = StdRng::seed_from_u64(MALE_SEED);
    let mut male_net = init_network(MALE_SEED, &mut male_rng);
    let mut male_proj = Int8Projection::new(
        out_dim(),
        CHARS,
        &mut StdRng::seed_from_u64(MALE_SEED + 200),
    );
    println!();

    // Eval rngs
    let mut woman_eval_rng = StdRng::seed_from_u64(WOMAN_SEED + 1000);
    let mut male_eval_rng = StdRng::seed_from_u64(MALE_SEED + 1000);

    let mut best_ever = 0.0f64;

    // Per-generation tracking for final summary
    struct GenRecord {
        gen: usize,
        woman_acc: f64,
        male_acc: f64,
        child_acc: f64,
        child_edges: usize,
        breed_improvement: f64,
    }
    let mut records: Vec<GenRecord> = Vec::new();

    let total_start = Instant::now();

    // Which parent is "older"? Woman starts older (gets replaced first).
    // Alternates each generation.
    let mut woman_is_older = true;

    for gen in 0..GENERATIONS {
        let gen_start = Instant::now();
        println!(
            "================================================================"
        );
        println!(
            "  GENERATION {} / {}   (older parent: {})",
            gen,
            GENERATIONS - 1,
            if woman_is_older { "Woman" } else { "Male" }
        );
        println!(
            "================================================================\n"
        );

        // --- Evolve Woman ---
        println!("  >> Evolving Woman <<");
        let (woman_acc, woman_edges) = evolve_parent(
            "Woman",
            &mut woman_net,
            &mut woman_proj,
            &corpus,
            &sdr,
            &prop,
            &mut woman_rng,
            &mut woman_eval_rng,
            &mut best_ever,
        );
        println!();

        // --- Evolve Male ---
        println!("  >> Evolving Male <<");
        let (male_acc, male_edges) = evolve_parent(
            "Male",
            &mut male_net,
            &mut male_proj,
            &corpus,
            &sdr,
            &prop,
            &mut male_rng,
            &mut male_eval_rng,
            &mut best_ever,
        );
        println!();

        // --- Save parent checkpoints before breed ---
        do_save_checkpoint(
            &format!("gen{}_woman_pre_breed", gen),
            &woman_net,
            &woman_proj,
            gen * STEPS_PER_GEN + STEPS_PER_GEN,
            woman_acc,
        );
        do_save_checkpoint(
            &format!("gen{}_male_pre_breed", gen),
            &male_net,
            &male_proj,
            gen * STEPS_PER_GEN + STEPS_PER_GEN,
            male_acc,
        );

        // --- Breed ---
        let mut breed_rng = StdRng::seed_from_u64(WOMAN_SEED + MALE_SEED + gen as u64 * 7919);
        let (child_net, child_proj) = breed(
            &woman_net,
            &male_net,
            &woman_proj,
            &corpus,
            &sdr,
            &prop,
            gen,
            &mut breed_rng,
        );

        // Eval child
        let mut child_eval_rng = StdRng::seed_from_u64(WOMAN_SEED + 3333 + gen as u64);
        let mut child_net_eval = child_net.clone();
        let child_acc = eval_accuracy(
            &mut child_net_eval,
            &child_proj,
            &corpus,
            EVAL_LEN_LONG,
            &mut child_eval_rng,
            &sdr,
            &prop,
            os,
            h,
        );
        let child_edges = child_net.edge_count();

        if child_acc > best_ever {
            best_ever = child_acc;
        }

        // Save child checkpoint
        do_save_checkpoint(
            &format!("gen{}_child", gen),
            &child_net,
            &child_proj,
            gen * STEPS_PER_GEN + STEPS_PER_GEN,
            child_acc,
        );

        // --- Replace older parent with child ---
        let better_parent_acc = woman_acc.max(male_acc);
        let breed_improvement = child_acc - better_parent_acc;

        if woman_is_older {
            println!("  Replacing Woman (older) with Child");
            println!(
                "    Woman {:.2}% -> Child {:.2}% (delta={:+.2}pp)",
                woman_acc * 100.0,
                child_acc * 100.0,
                (child_acc - woman_acc) * 100.0,
            );
            woman_net = child_net;
            woman_proj = child_proj;
            woman_rng = StdRng::seed_from_u64(WOMAN_SEED + (gen as u64 + 1) * 10000);
            woman_eval_rng = StdRng::seed_from_u64(WOMAN_SEED + (gen as u64 + 1) * 10000 + 1000);
        } else {
            println!("  Replacing Male (older) with Child");
            println!(
                "    Male {:.2}% -> Child {:.2}% (delta={:+.2}pp)",
                male_acc * 100.0,
                child_acc * 100.0,
                (child_acc - male_acc) * 100.0,
            );
            male_net = child_net;
            male_proj = child_proj;
            male_rng = StdRng::seed_from_u64(MALE_SEED + (gen as u64 + 1) * 10000);
            male_eval_rng = StdRng::seed_from_u64(MALE_SEED + (gen as u64 + 1) * 10000 + 1000);
        }

        // Alternate
        woman_is_older = !woman_is_older;

        let gen_elapsed = gen_start.elapsed();

        // --- Generation summary ---
        println!(
            "\n  +-- Gen {} Summary --+",
            gen
        );
        println!(
            "  | Woman:  {:.2}%  edges={}",
            woman_acc * 100.0,
            woman_edges
        );
        println!(
            "  | Male:   {:.2}%  edges={}",
            male_acc * 100.0,
            male_edges
        );
        println!(
            "  | Child:  {:.2}%  edges={}",
            child_acc * 100.0,
            child_edges
        );
        println!(
            "  | Breed improvement: {:+.2}pp vs best parent",
            breed_improvement * 100.0
        );
        println!(
            "  | Best ever: {:.2}%",
            best_ever * 100.0
        );
        println!(
            "  | Time: {:.1}s",
            gen_elapsed.as_secs_f64()
        );
        println!(
            "  +-------------------+\n"
        );

        records.push(GenRecord {
            gen,
            woman_acc,
            male_acc,
            child_acc,
            child_edges,
            breed_improvement,
        });
    }

    let total_elapsed = total_start.elapsed();

    // ---------------------------------------------------------------------------
    // Final summary
    // ---------------------------------------------------------------------------
    println!(
        "\n================================================================"
    );
    println!("  FINAL SUMMARY");
    println!(
        "================================================================\n"
    );

    println!(
        "  {:<5} {:>9} {:>9} {:>9} {:>10} {:>8}",
        "Gen", "Woman%", "Male%", "Child%", "Breed+pp", "Edges"
    );
    println!("  {}", "-".repeat(55));
    for r in &records {
        println!(
            "  {:<5} {:>8.2}% {:>8.2}% {:>8.2}% {:>+9.2}pp {:>8}",
            r.gen,
            r.woman_acc * 100.0,
            r.male_acc * 100.0,
            r.child_acc * 100.0,
            r.breed_improvement * 100.0,
            r.child_edges,
        );
    }
    println!("  {}", "-".repeat(55));
    println!(
        "  Peak accuracy across all generations: {:.2}%",
        best_ever * 100.0
    );

    // Edge efficiency
    if let Some(last) = records.last() {
        let eff = if last.child_edges > 0 {
            last.child_acc / last.child_edges as f64
        } else {
            0.0
        };
        println!(
            "  Final edge efficiency (acc/edges): {:.6}",
            eff
        );
    }

    println!(
        "  Total time: {:.1}s ({:.1}s per generation)\n",
        total_elapsed.as_secs_f64(),
        total_elapsed.as_secs_f64() / GENERATIONS as f64,
    );
}
