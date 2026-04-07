//! Shared Female: one frozen Female pocket, multiple Males trained against it.
//!
//! Tests whether Males trained on the same Female develop compatible "languages"
//! that can be meaningfully merged.
//!
//! Run: cargo run --example pocket_shared --release -- <corpus-path>

use instnct_core::{load_corpus, 
    load_checkpoint, save_checkpoint, CheckpointMeta, Int8Projection, Network,
    PropagationConfig, SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs;
use std::time::Instant;

const MASTER_SEED: u64 = 1337;
const H: usize = 256;
const PHI_DIM: usize = 158;
const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const N_MALES: usize = 10;
const STEPS: usize = 30_000;
const EVAL_LEN_SHORT: usize = 100;
const EVAL_LEN_LONG: usize = 2000;
const LOG_INTERVAL: usize = 5000;
const CHECKPOINT_DIR: &str = "checkpoints/pocket_shared";
const FEMALE_CKPT: &str = "checkpoints/pocket_pair/BF_female.ckpt";

fn output_start() -> usize { H - PHI_DIM }

fn charge_transfer(female: &Network) -> Vec<i32> {
    let os = output_start();
    let mut input = vec![0i32; H];
    for (i, &c) in female.charge()[os..H].iter().enumerate() {
        if i < PHI_DIM { input[i] = c as i32; }
    }
    input
}

fn build_male(rng: &mut StdRng) -> Network {
    let mut net = Network::new(H);
    let os = output_start();
    let ie = PHI_DIM;
    let mid = (os + ie) / 2;
    for _ in 0..50 {
        let s = rng.gen_range(0..os) as u16;
        let h1 = rng.gen_range(os..mid) as u16;
        let h2 = rng.gen_range(mid..ie) as u16;
        let t = rng.gen_range(ie..H) as u16;
        net.graph_mut().add_edge(s, h1);
        net.graph_mut().add_edge(h1, h2);
        net.graph_mut().add_edge(h2, t);
    }
    let target = H * H * 5 / 100;
    for _ in 0..target * 3 {
        let s = rng.gen_range(0..H) as u16;
        let t = rng.gen_range(0..H) as u16;
        if s != t { net.graph_mut().add_edge(s, t); }
        if net.edge_count() >= target { break; }
    }
    for i in 0..H {
        net.threshold_mut()[i] = rng.gen_range(0..=7u8);
        net.channel_mut()[i] = rng.gen_range(1..=8u8);
        if rng.gen_ratio(1, 10) { net.polarity_mut()[i] = -1; }
    }
    net
}

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
        female.propagate(sdr.pattern(seg[i] as usize), config).unwrap();
        let transfer = charge_transfer(female);
        male.propagate(&transfer, config).unwrap();
        if proj.predict(&male.charge()[os..H]) == seg[i + 1] as usize { correct += 1; }
    }
    correct as f64 / len as f64
}

/// Mutate ONLY the Male pocket + W (Female is frozen)
fn mutate_male(male: &mut Network, proj: &mut Int8Projection, rng: &mut impl Rng) -> bool {
    let roll = rng.gen_range(0..100u32);
    match roll {
        0..25 => {
            for _ in 0..30 {
                let s = rng.gen_range(0..H) as u16;
                let t = rng.gen_range(0..H) as u16;
                if s != t && male.graph_mut().add_edge(s, t) { return true; }
            }
            false
        }
        25..40 => {
            let edges: Vec<_> = male.graph().iter_edges().collect();
            if edges.is_empty() { return false; }
            let e = edges[rng.gen_range(0..edges.len())];
            male.graph_mut().remove_edge(e.source, e.target);
            true
        }
        40..55 => {
            let edges: Vec<_> = male.graph().iter_edges().collect();
            if edges.is_empty() { return false; }
            let e = edges[rng.gen_range(0..edges.len())];
            for _ in 0..30 {
                let new_t = rng.gen_range(0..H) as u16;
                if new_t != e.source {
                    male.graph_mut().remove_edge(e.source, e.target);
                    if male.graph_mut().add_edge(e.source, new_t) { return true; }
                    male.graph_mut().add_edge(e.source, e.target);
                    return false;
                }
            }
            false
        }
        55..70 => {
            let edges: Vec<_> = male.graph().iter_edges().collect();
            if edges.is_empty() { return false; }
            let e = edges[rng.gen_range(0..edges.len())];
            male.graph_mut().remove_edge(e.source, e.target);
            if male.graph_mut().add_edge(e.target, e.source) { return true; }
            male.graph_mut().add_edge(e.source, e.target);
            false
        }
        70..85 => {
            let idx = rng.gen_range(0..H);
            match rng.gen_range(0..3u32) {
                0 => { male.threshold_mut()[idx] = rng.gen_range(0..=7); true }
                1 => { male.channel_mut()[idx] = rng.gen_range(1..=8); true }
                _ => { male.polarity_mut()[idx] *= -1; true }
            }
        }
        _ => { let _ = proj.mutate_one(rng); true }
    }
}

struct MaleUnit {
    name: String,
    seed: u64,
    male: Network,
    proj: Int8Projection,
    sdr: SdrTable,
    female: Network, // clone of shared female (each thread needs its own mutable copy)
    mut_rng: StdRng,
    eval_rng: StdRng,
    accuracy: f64,
    peak: f64,
    accepted: u32,
    total_tried: u32,
}

fn bar(val: f64, max_val: f64, width: usize) -> String {
    let filled = ((val / max_val) * width as f64).round().min(width as f64) as usize;
    format!("{}{}", "#".repeat(filled), ".".repeat(width - filled))
}


fn edge_set(net: &Network) -> HashSet<(u16, u16)> {
    net.graph().iter_edges().map(|e| (e.source, e.target)).collect()
}

fn jaccard(a: &HashSet<(u16, u16)>, b: &HashSet<(u16, u16)>) -> f64 {
    let inter = a.intersection(b).count();
    let union_size = a.union(b).count();
    if union_size == 0 { 0.0 } else { inter as f64 / union_size as f64 }
}

fn evolve_male_unit(u: &mut MaleUnit, corpus: &[u8], prop: &PropagationConfig) {
    for step in 0..STEPS {
        let snap = u.eval_rng.clone();
        let before = eval_chain(&mut u.female, &mut u.male, &u.proj,
            corpus, EVAL_LEN_SHORT, &mut u.eval_rng, &u.sdr, prop);
        u.eval_rng = snap;

        let m_state = u.male.save_state();
        let proj_backup = u.proj.clone();
        let mutated = mutate_male(&mut u.male, &mut u.proj, &mut u.mut_rng);

        if !mutated {
            let _ = eval_chain(&mut u.female, &mut u.male, &u.proj,
                corpus, EVAL_LEN_SHORT, &mut u.eval_rng, &u.sdr, prop);
            continue;
        }
        u.total_tried += 1;

        let after = eval_chain(&mut u.female, &mut u.male, &u.proj,
            corpus, EVAL_LEN_SHORT, &mut u.eval_rng, &u.sdr, prop);

        if after > before {
            u.accepted += 1;
        } else {
            u.male.restore_state(&m_state);
            u.proj = proj_backup;
        }

        if (step + 1) % LOG_INTERVAL == 0 {
            let mut cr = StdRng::seed_from_u64(u.seed.wrapping_add(6000 + step as u64));
            let acc = eval_chain(&mut u.female, &mut u.male, &u.proj,
                corpus, EVAL_LEN_LONG, &mut cr, &u.sdr, prop);
            if acc > u.peak { u.peak = acc; }
            let rate = if u.total_tried > 0 { u.accepted as f64 / u.total_tried as f64 } else { 0.0 };
            println!("  [{:>5}] {} |{}|{:.1}% edges={} accept={:.1}% peak={:.1}%",
                step + 1, u.name, bar(acc, 0.30, 15), acc * 100.0,
                u.male.edge_count(), rate * 100.0, u.peak * 100.0);
        }
    }

    let mut fr = StdRng::seed_from_u64(u.seed.wrapping_add(9999));
    u.accuracy = eval_chain(&mut u.female, &mut u.male, &u.proj,
        corpus, EVAL_LEN_LONG, &mut fr, &u.sdr, prop);
    if u.accuracy > u.peak { u.peak = u.accuracy; }
    let rate = if u.total_tried > 0 { u.accepted as f64 / u.total_tried as f64 * 100.0 } else { 0.0 };
    println!("  {} FINAL: {:.2}% peak={:.2}% edges={} accept={}/{} ({:.1}%)",
        u.name, u.accuracy * 100.0, u.peak * 100.0,
        u.male.edge_count(), u.accepted, u.total_tried, rate);
}

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });
    let prop = PropagationConfig {
        ticks_per_token: 6, input_duration_ticks: 2,
        decay_interval_ticks: 6, use_refractory: false,
    };

    println!("=== SHARED FEMALE EXPERIMENT ===");
    println!("  1 frozen Female (BF), {} Males trained against it", N_MALES);
    println!("  Steps: {}, strict acceptance, Male+W only mutations\n", STEPS);

    // Load shared female
    println!("  Loading shared Female from: {}", FEMALE_CKPT);
    let (shared_female, _bf_proj, meta) = load_checkpoint(FEMALE_CKPT)
        .expect("cannot load BF female checkpoint");
    println!("  Female: {} edges, label: {}, acc: {:.2}%\n",
        shared_female.edge_count(), meta.label, meta.accuracy * 100.0);

    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  Corpus: {} chars\n", corpus.len());

    fs::create_dir_all(CHECKPOINT_DIR).ok();

    let mut seed_gen = StdRng::seed_from_u64(MASTER_SEED.wrapping_add(77777));
    let sdr_seed = {
        let mut sg = StdRng::seed_from_u64(MASTER_SEED);
        sg.next_u64()
    };

    let mut units: Vec<MaleUnit> = Vec::new();
    for i in 0..N_MALES {
        let male_seed = seed_gen.next_u64();
        let name = format!("M{}", i);
        let male = build_male(&mut StdRng::seed_from_u64(male_seed));
        let proj = Int8Projection::new(PHI_DIM, CHARS,
            &mut StdRng::seed_from_u64(male_seed.wrapping_add(200)));
        let sdr = SdrTable::new(CHARS, H, PHI_DIM, SDR_ACTIVE_PCT,
            &mut StdRng::seed_from_u64(sdr_seed)).unwrap();
        let female_clone = shared_female.clone();

        println!("  {} seed={} male_edges={}", name, male_seed, male.edge_count());

        units.push(MaleUnit {
            name, seed: male_seed, male, proj, sdr,
            female: female_clone,
            mut_rng: StdRng::seed_from_u64(male_seed.wrapping_add(500)),
            eval_rng: StdRng::seed_from_u64(male_seed.wrapping_add(1000)),
            accuracy: 0.0, peak: 0.0, accepted: 0, total_tried: 0,
        });
    }

    println!("\n=== EVOLVING {} Males (Female frozen) ===\n", N_MALES);
    let start = Instant::now();

    units.par_iter_mut().for_each(|u| {
        evolve_male_unit(u, &corpus, &prop);
    });

    let elapsed = start.elapsed().as_secs_f64();

    // Sort by accuracy
    units.sort_by(|a, b| b.accuracy.partial_cmp(&a.accuracy).unwrap());

    println!("\n=== RESULTS (sorted) ===\n");
    println!("  Rank  Name  Final%   Peak%  Edges  Accept%  Seed");
    println!("  ----  ----  ------   -----  -----  -------  ----");
    for (i, u) in units.iter().enumerate() {
        let rate = if u.total_tried > 0 { u.accepted as f64 / u.total_tried as f64 * 100.0 } else { 0.0 };
        println!("  {:>4}  {:>4}  {:>5.1}%  {:>5.1}%  {:>5}  {:>6.1}%  {}",
            i + 1, u.name, u.accuracy * 100.0, u.peak * 100.0,
            u.male.edge_count(), rate, u.seed);
    }

    let mean = units.iter().map(|u| u.accuracy).sum::<f64>() / units.len() as f64;
    let top5 = units[..5].iter().map(|u| u.accuracy).sum::<f64>() / 5.0;
    println!("\n  Mean: {:.1}%  Top-5: {:.1}%  Time: {:.1}s\n", mean * 100.0, top5 * 100.0, elapsed);

    // === MALE JACCARD MATRIX ===
    println!("=== MALE JACCARD (do shared-Female Males converge?) ===\n");
    let edge_sets: Vec<_> = units.iter().map(|u| edge_set(&u.male)).collect();
    print!("       ");
    for u in &units { print!(" {:>5}", u.name); }
    println!();
    for (i, ui) in units.iter().enumerate() {
        print!("  {:>4} ", ui.name);
        for (j, _) in units.iter().enumerate() {
            if i == j { print!(" {:>5}", "--"); continue; }
            print!(" {:>4.1}%", jaccard(&edge_sets[i], &edge_sets[j]) * 100.0);
        }
        println!();
    }

    // === PAIRWISE PREDICTION AGREEMENT ===
    println!("\n=== PREDICTION AGREEMENT (same 2000 chars) ===\n");
    let seg_off = 50000usize;
    let segment = &corpus[seg_off..seg_off + EVAL_LEN_LONG + 1];
    let os = output_start();

    let mut all_preds: Vec<Vec<usize>> = Vec::new();
    let sdr_eval = SdrTable::new(CHARS, H, PHI_DIM, SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(sdr_seed)).unwrap();

    for u in &mut units {
        u.female.reset();
        u.male.reset();
        let mut preds = Vec::with_capacity(EVAL_LEN_LONG);
        for &tok in segment[..EVAL_LEN_LONG].iter() {
            u.female.propagate(sdr_eval.pattern(tok as usize), &prop).unwrap();
            let transfer = charge_transfer(&u.female);
            u.male.propagate(&transfer, &prop).unwrap();
            preds.push(u.proj.predict(&u.male.charge()[os..H]));
        }
        all_preds.push(preds);
    }

    print!("       ");
    for u in &units { print!(" {:>5}", u.name); }
    println!();
    for (i, ui) in units.iter().enumerate() {
        print!("  {:>4} ", ui.name);
        for (j, _) in units.iter().enumerate() {
            if i == j { print!(" {:>5}", "--"); continue; }
            let agree = (0..EVAL_LEN_LONG)
                .filter(|&k| all_preds[i][k] == all_preds[j][k]).count();
            print!(" {:>4.1}%", agree as f64 / EVAL_LEN_LONG as f64 * 100.0);
        }
        println!();
    }

    // Oracle
    let oracle = (0..EVAL_LEN_LONG).filter(|&k| {
        all_preds.iter().any(|p| p[k] == segment[k + 1] as usize)
    }).count();
    let top5_oracle = (0..EVAL_LEN_LONG).filter(|&k| {
        all_preds[..5].iter().any(|p| p[k] == segment[k + 1] as usize)
    }).count();
    println!("\n  Oracle (all {}): {:.1}%", N_MALES, oracle as f64 / EVAL_LEN_LONG as f64 * 100.0);
    println!("  Oracle (top 5):  {:.1}%", top5_oracle as f64 / EVAL_LEN_LONG as f64 * 100.0);

    // === MERGE TOP MALES ===
    println!("\n=== MERGE TOP 5 MALES ===\n");
    let mut merged_male = Network::new(H);
    // Copy best male's params
    for i in 0..H {
        merged_male.threshold_mut()[i] = units[0].male.threshold()[i];
        merged_male.channel_mut()[i] = units[0].male.channel()[i];
        merged_male.polarity_mut()[i] = units[0].male.polarity()[i];
    }
    for u in units[..5].iter() {
        let mut added = 0usize;
        let mut dups = 0usize;
        for e in u.male.graph().iter_edges() {
            if merged_male.graph_mut().add_edge(e.source, e.target) { added += 1; } else { dups += 1; }
        }
        println!("  + {} ({:.1}%): added={} dups={} total={}",
            u.name, u.accuracy * 100.0, added, dups, merged_male.edge_count());
    }

    // Eval merged male with shared female + best W
    let merged_proj = units[0].proj.clone();
    let mut eval_female = shared_female.clone();
    let mut eval_rng = StdRng::seed_from_u64(MASTER_SEED.wrapping_add(33333));
    let merged_acc = eval_chain(&mut eval_female, &mut merged_male, &merged_proj,
        &corpus, EVAL_LEN_LONG, &mut eval_rng,
        &SdrTable::new(CHARS, H, PHI_DIM, SDR_ACTIVE_PCT,
            &mut StdRng::seed_from_u64(sdr_seed)).unwrap(),
        &prop);

    println!("\n  Merged male: {} edges, accuracy: {:.2}%", merged_male.edge_count(), merged_acc * 100.0);
    println!("  vs best single ({}): {:.2}%", units[0].name, units[0].accuracy * 100.0);
    println!("  Delta: {:+.2}pp", (merged_acc - units[0].accuracy) * 100.0);

    // Jaccard merged vs individuals
    let merged_edges = edge_set(&merged_male);
    println!("\n  Merged Jaccard vs individuals:");
    for u in &units[..5] {
        let ind_edges = edge_set(&u.male);
        let shared = ind_edges.intersection(&merged_edges).count();
        println!("    vs {} ({:.1}%): Jaccard={:.1}% shared={}",
            u.name, u.accuracy * 100.0, jaccard(&ind_edges, &merged_edges) * 100.0, shared);
    }

    // Save
    for u in &units {
        let _ = save_checkpoint(
            format!("{}/{}_male.ckpt", CHECKPOINT_DIR, u.name),
            &u.male, &u.proj,
            CheckpointMeta { step: STEPS, accuracy: u.accuracy, label: u.name.clone() },
        );
    }
    let _ = save_checkpoint(
        format!("{}/merged_top5_male.ckpt", CHECKPOINT_DIR),
        &merged_male, &merged_proj,
        CheckpointMeta { step: STEPS, accuracy: merged_acc, label: "merged_top5".into() },
    );

    println!("\n  Checkpoints saved to {}/", CHECKPOINT_DIR);
    println!("\n  Total time: {:.1}s", elapsed);
}
