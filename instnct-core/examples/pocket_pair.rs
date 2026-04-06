//! Pocket pair: two separate H=256 pockets (Female + Male) chained end-to-end.
//!
//! Female processes SDR input, her output charge feeds into Male's input.
//! Male's output charge goes through W projection for prediction.
//! Trained END-TO-END: mutation in either pocket, eval on the full chain.
//! 5 units (A-E) × 10 cores, 30K steps, strict acceptance.
//!
//! Run: cargo run --example pocket_pair --release -- <corpus-path>

use instnct_core::{load_corpus, 
    save_checkpoint, CheckpointMeta, Int8Projection, Network, PropagationConfig, SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use rayon::prelude::*;
use std::fs;
use std::time::Instant;

const MASTER_SEED: u64 = 1337;
const N_UNITS: usize = 5;
const H: usize = 256;
const PHI_DIM: usize = 158;
const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const STEPS: usize = 30_000;
const EVAL_LEN_SHORT: usize = 100;
const EVAL_LEN_LONG: usize = 2000;
const LOG_INTERVAL: usize = 5000;
const CHECKPOINT_DIR: &str = "checkpoints/pocket_pair";

fn output_start() -> usize { H - PHI_DIM } // 98

/// Build one H=256 pocket with chain-50 + 5% density + random params.
fn build_pocket(rng: &mut StdRng) -> Network {
    let mut net = Network::new(H);
    let os = output_start();
    let ie = PHI_DIM;
    let mid = (os + ie) / 2;

    // Chain-50
    for _ in 0..50 {
        let s = rng.gen_range(0..os) as u16;
        let h1 = rng.gen_range(os..mid) as u16;
        let h2 = rng.gen_range(mid..ie) as u16;
        let t = rng.gen_range(ie..H) as u16;
        net.graph_mut().add_edge(s, h1);
        net.graph_mut().add_edge(h1, h2);
        net.graph_mut().add_edge(h2, t);
    }

    // 5% density
    let target = H * H * 5 / 100;
    for _ in 0..target * 3 {
        let s = rng.gen_range(0..H) as u16;
        let t = rng.gen_range(0..H) as u16;
        if s != t {
            net.graph_mut().add_edge(s, t);
        }
        if net.edge_count() >= target { break; }
    }

    // Random params
    for i in 0..H {
        net.threshold_mut()[i] = rng.gen_range(0..=7u32);
        net.channel_mut()[i] = rng.gen_range(1..=8u8);
        if rng.gen_ratio(1, 10) { net.polarity_mut()[i] = -1; }
    }
    net
}

/// Transfer Female output charge → Male input signal.
/// Direct charge mapping: output_zone[i] → input[i] as i32.
fn charge_transfer(female: &Network) -> Vec<i32> {
    let os = output_start();
    let mut input = vec![0i32; H];
    // Female output zone [98..256] → Male input zone [0..158]
    for (i, &c) in female.charge()[os..H].iter().enumerate() {
        if i < PHI_DIM {
            input[i] = c as i32;
        }
    }
    input
}

/// Eval the full chain: SDR → Female → charge transfer → Male → W → prediction.
#[allow(clippy::too_many_arguments)]
fn eval_chain(
    female: &mut Network, male: &mut Network, proj: &Int8Projection,
    corpus: &[u8], len: usize, rng: &mut StdRng,
    sdr: &SdrTable, config: &PropagationConfig,
) -> f64 {
    if corpus.len() <= len { return 0.0; }
    let off = rng.gen_range(0..=corpus.len() - len - 1);
    let seg = &corpus[off..off + len + 1];
    let os = output_start();

    female.reset();
    male.reset();
    let mut correct = 0u32;
    for i in 0..len {
        // Female processes SDR
        female.propagate(sdr.pattern(seg[i] as usize), config).unwrap();
        // Female charge → Male input
        let transfer = charge_transfer(female);
        male.propagate(&transfer, config).unwrap();
        // Predict from Male output zone
        if proj.predict(&male.charge()[os..H]) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

/// Mutate a random pocket (Female or Male). Returns which was mutated.
fn mutate_unit(
    female: &mut Network, male: &mut Network, proj: &mut Int8Projection,
    rng: &mut impl Rng,
) -> (bool, bool) { // (mutated, was_male)
    let target_male = rng.gen_bool(0.5);
    let net = if target_male { male } else { female };

    let roll = rng.gen_range(0..100u32);
    let mutated = match roll {
        0..25 => { // add edge
            for _ in 0..30 {
                let s = rng.gen_range(0..H) as u16;
                let t = rng.gen_range(0..H) as u16;
                if s != t && net.graph_mut().add_edge(s, t) { return (true, target_male); }
            }
            false
        }
        25..40 => { // remove edge
            let edges: Vec<_> = net.graph().iter_edges().collect();
            if edges.is_empty() { return (false, target_male); }
            let e = edges[rng.gen_range(0..edges.len())];
            net.graph_mut().remove_edge(e.source, e.target);
            true
        }
        40..55 => { // rewire
            let edges: Vec<_> = net.graph().iter_edges().collect();
            if edges.is_empty() { return (false, target_male); }
            let e = edges[rng.gen_range(0..edges.len())];
            for _ in 0..30 {
                let new_t = rng.gen_range(0..H) as u16;
                if new_t != e.source {
                    net.graph_mut().remove_edge(e.source, e.target);
                    if net.graph_mut().add_edge(e.source, new_t) { return (true, target_male); }
                    net.graph_mut().add_edge(e.source, e.target);
                    return (false, target_male);
                }
            }
            false
        }
        55..70 => { // reverse
            let edges: Vec<_> = net.graph().iter_edges().collect();
            if edges.is_empty() { return (false, target_male); }
            let e = edges[rng.gen_range(0..edges.len())];
            net.graph_mut().remove_edge(e.source, e.target);
            if net.graph_mut().add_edge(e.target, e.source) { return (true, target_male); }
            net.graph_mut().add_edge(e.source, e.target);
            false
        }
        70..85 => { // param mutation
            let idx = rng.gen_range(0..H);
            match rng.gen_range(0..3u32) {
                0 => { net.threshold_mut()[idx] = rng.gen_range(0..=7); true }
                1 => { net.channel_mut()[idx] = rng.gen_range(1..=8); true }
                _ => { net.polarity_mut()[idx] *= -1; true }
            }
        }
        _ => { // W mutation (only if targeting male)
            if target_male {
                let _ = proj.mutate_one(rng);
                true
            } else {
                // param mutation on female instead
                let idx = rng.gen_range(0..H);
                net.threshold_mut()[idx] = rng.gen_range(0..=7);
                true
            }
        }
    };
    (mutated, target_male)
}

struct Unit {
    name: String,
    seed: u64,
    female: Network,
    male: Network,
    proj: Int8Projection,
    sdr: SdrTable,
    mut_rng: StdRng,
    eval_rng: StdRng,
    accuracy: f64,
    peak: f64,
    accepted: u32,
    total_tried: u32,
}


fn evolve_unit(unit: &mut Unit, corpus: &[u8], prop: &PropagationConfig) {
    let mut last_log_accepted = 0u32;
    let mut last_log_total = 0u32;

    for step in 0..STEPS {
        // Paired eval
        let snap = unit.eval_rng.clone();
        let before = eval_chain(
            &mut unit.female, &mut unit.male, &unit.proj,
            corpus, EVAL_LEN_SHORT, &mut unit.eval_rng, &unit.sdr, prop,
        );
        unit.eval_rng = snap;

        // Save state for rollback
        let f_state = unit.female.save_state();
        let m_state = unit.male.save_state();
        let proj_backup = unit.proj.clone();

        let (mutated, _was_male) = mutate_unit(
            &mut unit.female, &mut unit.male, &mut unit.proj, &mut unit.mut_rng,
        );

        if !mutated {
            let _ = eval_chain(
                &mut unit.female, &mut unit.male, &unit.proj,
                corpus, EVAL_LEN_SHORT, &mut unit.eval_rng, &unit.sdr, prop,
            );
            continue;
        }

        unit.total_tried += 1;

        let after = eval_chain(
            &mut unit.female, &mut unit.male, &unit.proj,
            corpus, EVAL_LEN_SHORT, &mut unit.eval_rng, &unit.sdr, prop,
        );

        if after > before {
            unit.accepted += 1;
        } else {
            unit.female.restore_state(&f_state);
            unit.male.restore_state(&m_state);
            unit.proj = proj_backup;
        }

        if (step + 1) % LOG_INTERVAL == 0 {
            let mut cr = StdRng::seed_from_u64(unit.seed.wrapping_add(6000 + step as u64));
            let acc = eval_chain(
                &mut unit.female, &mut unit.male, &unit.proj,
                corpus, EVAL_LEN_LONG, &mut cr, &unit.sdr, prop,
            );
            if acc > unit.peak { unit.peak = acc; }

            let int_acc = unit.accepted - last_log_accepted;
            let int_tot = unit.total_tried - last_log_total;
            let rate = if int_tot > 0 { int_acc as f64 / int_tot as f64 } else { 0.0 };

            println!(
                "  [{:>5}] {} |{}|{:.1}% F_edges={} M_edges={} accept={:.1}% peak={:.1}%",
                step + 1, unit.name,
                bar(acc, 0.30, 15),
                acc * 100.0,
                unit.female.edge_count(),
                unit.male.edge_count(),
                rate * 100.0,
                unit.peak * 100.0,
            );
            last_log_accepted = unit.accepted;
            last_log_total = unit.total_tried;
        }
    }

    // Final eval
    let mut fr = StdRng::seed_from_u64(unit.seed.wrapping_add(9999));
    unit.accuracy = eval_chain(
        &mut unit.female, &mut unit.male, &unit.proj,
        corpus, EVAL_LEN_LONG, &mut fr, &unit.sdr, prop,
    );
    if unit.accuracy > unit.peak { unit.peak = unit.accuracy; }

    let rate = if unit.total_tried > 0 {
        unit.accepted as f64 / unit.total_tried as f64
    } else { 0.0 };
    println!(
        "  {} FINAL: {:.2}% peak={:.2}% F_edges={} M_edges={} accept={}/{} ({:.1}%)",
        unit.name, unit.accuracy * 100.0, unit.peak * 100.0,
        unit.female.edge_count(), unit.male.edge_count(),
        unit.accepted, unit.total_tried, rate * 100.0,
    );
}

fn bar(val: f64, max_val: f64, width: usize) -> String {
    let filled = ((val / max_val) * width as f64).round().min(width as f64) as usize;
    format!("{}{}", "#".repeat(filled), ".".repeat(width - filled))
}

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });

    let prop = PropagationConfig {
        ticks_per_token: 6,
        input_duration_ticks: 2,
        decay_interval_ticks: 6,
        use_refractory: false,
    };

    println!("=== POCKET PAIR: 5 units × (Female + Male) ===");
    println!("  Master seed: {}", MASTER_SEED);
    println!("  Each pocket: H={}, phi_dim={}", H, PHI_DIM);
    println!("  Chain: SDR → Female(256) → charge → Male(256) → W");
    println!("  Steps: {}, strict acceptance", STEPS);
    println!();

    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars\n", corpus.len());

    fs::create_dir_all(CHECKPOINT_DIR).ok();

    // Derive seeds from master
    let mut seed_gen = StdRng::seed_from_u64(MASTER_SEED);
    let sdr_seed = seed_gen.next_u64();
    let pair_names = ['A', 'B', 'C', 'D', 'E'];

    let mut units: Vec<Unit> = Vec::with_capacity(N_UNITS * 2);

    for (_i, &pname) in pair_names.iter().enumerate().take(N_UNITS) {
        for &role in &['F', 'M'] {
            let ind_seed = seed_gen.next_u64();
            let name = format!("{}{}", pname, role);

            let female = build_pocket(&mut StdRng::seed_from_u64(ind_seed));
            let male = build_pocket(&mut StdRng::seed_from_u64(ind_seed.wrapping_add(1)));
            let proj = Int8Projection::new(PHI_DIM, CHARS,
                &mut StdRng::seed_from_u64(ind_seed.wrapping_add(200)));
            let sdr = SdrTable::new(CHARS, H, PHI_DIM, SDR_ACTIVE_PCT,
                &mut StdRng::seed_from_u64(sdr_seed)).unwrap();
            let eval_rng = StdRng::seed_from_u64(ind_seed.wrapping_add(1000));
            let mut_rng = StdRng::seed_from_u64(ind_seed.wrapping_add(500));

            println!("  {} seed={} F_edges={} M_edges={}", name, ind_seed,
                female.edge_count(), male.edge_count());

            units.push(Unit {
                name, seed: ind_seed, female, male, proj, sdr,
                mut_rng, eval_rng,
                accuracy: 0.0, peak: 0.0, accepted: 0, total_tried: 0,
            });
        }
    }

    println!("\n=== EVOLVING {} units in parallel ===\n", units.len());
    let start = Instant::now();

    units.par_iter_mut().for_each(|unit| {
        evolve_unit(unit, &corpus, &prop);
    });

    let elapsed = start.elapsed().as_secs_f64();

    // Sort by accuracy
    units.sort_by(|a, b| b.accuracy.partial_cmp(&a.accuracy).unwrap());

    println!("\n=== RESULTS (sorted) ===\n");
    println!("  Rank  Name  Final%   Peak%  F_edges  M_edges  Accept%  Seed");
    println!("  ----  ----  ------   -----  -------  -------  -------  ----");
    for (i, u) in units.iter().enumerate() {
        let rate = if u.total_tried > 0 { u.accepted as f64 / u.total_tried as f64 * 100.0 } else { 0.0 };
        println!("  {:>4}  {:>4}  {:>5.1}%  {:>5.1}%  {:>7}  {:>7}  {:>6.1}%  {}",
            i + 1, u.name, u.accuracy * 100.0, u.peak * 100.0,
            u.female.edge_count(), u.male.edge_count(), rate, u.seed);
    }

    let mean = units.iter().map(|u| u.accuracy).sum::<f64>() / units.len() as f64;
    let top5_mean = units[..5].iter().map(|u| u.accuracy).sum::<f64>() / 5.0;
    println!("\n  Mean: {:.1}%  Top-5: {:.1}%  Time: {:.1}s", mean * 100.0, top5_mean * 100.0, elapsed);

    // Save checkpoints — BOTH pockets per unit
    for u in &units {
        let _ = save_checkpoint(
            format!("{}/{}_female.ckpt", CHECKPOINT_DIR, u.name),
            &u.female, &u.proj,
            CheckpointMeta { step: STEPS, accuracy: u.accuracy, label: format!("{}_female", u.name) },
        );
        let _ = save_checkpoint(
            format!("{}/{}_male.ckpt", CHECKPOINT_DIR, u.name),
            &u.male, &u.proj,
            CheckpointMeta { step: STEPS, accuracy: u.accuracy, label: format!("{}_male", u.name) },
        );
    }
    println!("\n  Checkpoints saved to {}/\n", CHECKPOINT_DIR);

    // =========================================================================
    // MERGE PHASE: top Female + overlay top Males → crystallize → evolve
    // =========================================================================

    // Find the best female-role unit and top male-role units
    // Role is encoded in name: last char is 'F' or 'M'
    let mut best_female: Option<usize> = None;
    let mut male_indices: Vec<usize> = Vec::new();

    // Top 5 are already sorted by accuracy (index 0 = best)
    for (i, u) in units.iter().enumerate() {
        if i >= 5 { break; } // only top 5
        let is_female = u.name.ends_with('F');
        if is_female && best_female.is_none() {
            best_female = Some(i);
        } else if !is_female {
            male_indices.push(i);
        }
    }

    if best_female.is_none() {
        println!("  No female in top 5, skipping merge.");
        return;
    }
    let fi = best_female.unwrap();

    println!("=== MERGE PHASE ===\n");
    println!("  A_Female = {} ({:.1}%, F_edges={}, M_edges={})",
        units[fi].name, units[fi].accuracy * 100.0,
        units[fi].female.edge_count(), units[fi].male.edge_count());
    println!("  Males to overlay:");
    for &mi in &male_indices {
        println!("    {} ({:.1}%, M_edges={})", units[mi].name,
            units[mi].accuracy * 100.0, units[mi].male.edge_count());
    }

    // A_Female pocket: take the best female's female pocket as-is
    let a_female = units[fi].female.clone();
    println!("\n  A_Female: {} edges (from {})", a_female.edge_count(), units[fi].name);

    // A_Male pocket: overlay all top males' male pockets
    let mut a_male = Network::new(H);

    // Copy best male's params as base
    let best_male_idx = male_indices[0];
    for i in 0..H {
        a_male.threshold_mut()[i] = units[best_male_idx].male.threshold()[i];
        a_male.channel_mut()[i] = units[best_male_idx].male.channel()[i];
        a_male.polarity_mut()[i] = units[best_male_idx].male.polarity()[i];
    }

    let mut total_added = 0usize;
    let mut total_dups = 0usize;
    for &mi in &male_indices {
        let mut added = 0usize;
        let mut dups = 0usize;
        for e in units[mi].male.graph().iter_edges() {
            if a_male.graph_mut().add_edge(e.source, e.target) {
                added += 1;
            } else {
                dups += 1;
            }
        }
        println!("    + {} male: added={} dups={} total={}",
            units[mi].name, added, dups, a_male.edge_count());
        total_added += added;
        total_dups += dups;
    }
    println!("  A_Male merged: {} edges (added={} dups={})\n",
        a_male.edge_count(), total_added, total_dups);

    // Use best male's W
    let a_proj = units[best_male_idx].proj.clone();
    println!("  W inherited from {} ({:.1}%)", units[best_male_idx].name,
        units[best_male_idx].accuracy * 100.0);

    // SDR for eval
    let merge_sdr = SdrTable::new(CHARS, H, PHI_DIM, SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(sdr_seed)).unwrap();

    // Eval pre-crystallize
    let mut pre_rng = StdRng::seed_from_u64(MASTER_SEED.wrapping_add(77777));
    let mut a_female_eval = a_female.clone();
    let pre_acc = eval_chain(&mut a_female_eval, &mut a_male, &a_proj,
        &corpus, EVAL_LEN_LONG, &mut pre_rng, &merge_sdr, &prop);
    println!("  Pre-crystallize accuracy: {:.2}% ({} male edges)\n",
        pre_acc * 100.0, a_male.edge_count());

    // --- Adaptive batch crystallize on A_Male ---
    // Eval = full chain (A_Female → A_Male → W)
    println!("  --- Adaptive Crystallize (A_Male) ---");

    let mut cryst_rng = StdRng::seed_from_u64(MASTER_SEED.wrapping_add(88888));
    let cryst_off = cryst_rng.gen_range(0..=corpus.len() - EVAL_LEN_LONG - 1);
    let cryst_seg = &corpus[cryst_off..cryst_off + EVAL_LEN_LONG + 1];

    // Crystallize eval uses full chain
    let cryst_eval = |female: &Network, male: &mut Network, proj: &Int8Projection, sdr: &SdrTable| -> f64 {
        let mut f_clone = female.clone();
        let os = output_start();
        let len = cryst_seg.len() - 1;
        f_clone.reset();
        male.reset();
        let mut correct = 0u32;
        for i in 0..len {
            f_clone.propagate(sdr.pattern(cryst_seg[i] as usize), &prop).unwrap();
            let transfer = charge_transfer(&f_clone);
            male.propagate(&transfer, &prop).unwrap();
            if proj.predict(&male.charge()[os..H]) == cryst_seg[i + 1] as usize {
                correct += 1;
            }
        }
        correct as f64 / len as f64
    };

    let baseline_acc = cryst_eval(&a_female, &mut a_male, &a_proj, &merge_sdr);
    println!("    Baseline: {:.2}% ({} edges)", baseline_acc * 100.0, a_male.edge_count());

    let mut current_acc = baseline_acc;
    let mut removal_pct = 0.50f64;
    let min_pct = 0.02f64;

    for round in 0..20 {
        let edges_before = a_male.edge_count();
        if edges_before == 0 { break; }
        let batch_size = ((edges_before as f64 * removal_pct) as usize).max(1);

        let all_edges: Vec<_> = a_male.graph().iter_edges().collect();
        let mut indices: Vec<usize> = (0..all_edges.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = cryst_rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        let batch: Vec<(u16, u16)> = indices[..batch_size.min(indices.len())]
            .iter()
            .map(|&i| (all_edges[i].source, all_edges[i].target))
            .collect();
        for &(s, t) in &batch { a_male.graph_mut().remove_edge(s, t); }

        let after_acc = cryst_eval(&a_female, &mut a_male, &a_proj, &merge_sdr);

        if after_acc >= current_acc - 0.02 {
            let old = removal_pct;
            removal_pct = (removal_pct * 1.5).min(0.80);
            current_acc = after_acc;
            println!("    Round {:>2}: ACCEPT  -{} edges ({}->{}) acc {:.2}% pct {:.0}%→{:.0}%",
                round, batch.len(), edges_before, a_male.edge_count(),
                after_acc * 100.0, old * 100.0, removal_pct * 100.0);
        } else {
            for &(s, t) in &batch { a_male.graph_mut().add_edge(s, t); }
            let old = removal_pct;
            removal_pct /= 2.0;
            println!("    Round {:>2}: REJECT  tried -{} edges, acc {:.2}% pct {:.0}%→{:.0}%",
                round, batch.len(), after_acc * 100.0, old * 100.0, removal_pct * 100.0);
            if removal_pct < min_pct {
                println!("    Converged (pct < {:.0}%).", min_pct * 100.0);
                break;
            }
        }
    }

    let post_acc = cryst_eval(&a_female, &mut a_male, &a_proj, &merge_sdr);
    println!("\n  Post-crystallize: {:.2}% ({} male edges)", post_acc * 100.0, a_male.edge_count());

    // --- Evolve the merged pair ---
    println!("\n  --- Evolving merged A pair ({} steps) ---\n", STEPS);

    let merge_eval_seed = MASTER_SEED.wrapping_add(99999);
    let merge_mut_seed = MASTER_SEED.wrapping_add(55555);
    let mut merged_unit = Unit {
        name: "A_merged".to_string(),
        seed: merge_eval_seed,
        female: a_female,
        male: a_male,
        proj: a_proj,
        sdr: merge_sdr,
        mut_rng: StdRng::seed_from_u64(merge_mut_seed),
        eval_rng: StdRng::seed_from_u64(merge_eval_seed),
        accuracy: 0.0,
        peak: post_acc,
        accepted: 0,
        total_tried: 0,
    };

    let merge_start = Instant::now();
    evolve_unit(&mut merged_unit, &corpus, &prop);
    let merge_time = merge_start.elapsed().as_secs_f64();

    println!("\n=== MERGE RESULT ===");
    println!("  A_merged FINAL: {:.2}%  peak={:.2}%  F_edges={}  M_edges={}  time={:.1}s",
        merged_unit.accuracy * 100.0, merged_unit.peak * 100.0,
        merged_unit.female.edge_count(), merged_unit.male.edge_count(), merge_time);
    println!("  vs best individual ({}): {:.2}%",
        units[0].name, units[0].accuracy * 100.0);
    let delta = merged_unit.accuracy - units[0].accuracy;
    println!("  Delta: {:+.2}pp {}", delta * 100.0,
        if delta > 0.0 { "IMPROVEMENT!" } else { "(no improvement)" });

    // Save merged checkpoints
    let _ = save_checkpoint(
        format!("{}/A_merged_female.ckpt", CHECKPOINT_DIR),
        &merged_unit.female, &merged_unit.proj,
        CheckpointMeta { step: STEPS * 2, accuracy: merged_unit.accuracy, label: "A_merged_female".into() },
    );
    let _ = save_checkpoint(
        format!("{}/A_merged_male.ckpt", CHECKPOINT_DIR),
        &merged_unit.male, &merged_unit.proj,
        CheckpointMeta { step: STEPS * 2, accuracy: merged_unit.accuracy, label: "A_merged_male".into() },
    );
    println!("  Checkpoints saved.");
}
