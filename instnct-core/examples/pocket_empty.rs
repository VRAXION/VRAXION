//! Pocket pair from EMPTY start + smooth fitness + jackpot.
//!
//! Two H=256 pockets (Female + Male) chained via charge transfer.
//! Both start from 0 edges. Smooth cosine-bigram fitness. Jackpot=9.
//!
//! This combines every finding from today's session:
//! - Empty start: 80% addition with 83 edges (vs 64% with 3400)
//! - Smooth fitness: +2.6pp over stepwise
//! - Jackpot: +3.4pp over 1+1 ES
//!
//! Control: prefilled (chain-50 + 5%) with same smooth+jackpot.
//!
//! Run: cargo run --example pocket_empty --release -- <corpus-path>

use instnct_core::{build_bigram_table, cosine_similarity, load_corpus, softmax, Int8Projection, Network, SdrTable, PropagationConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const H: usize = 256;
const PHI_DIM: usize = 158;
const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const STEPS: usize = 30_000;
const JACKPOT: usize = 9;
const EDGE_CAP_PER_POCKET: usize = H * H * 7 / 100;

fn output_start() -> usize { H - PHI_DIM } // 98

/// Build prefilled pocket (chain-50 + 5% density + random params).
fn build_prefilled_pocket(rng: &mut StdRng) -> Network {
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
        net.threshold_mut()[i] = rng.gen_range(0..=7u32);
        net.channel_mut()[i] = rng.gen_range(1..=8u8);
        if rng.gen_ratio(1, 10) { net.polarity_mut()[i] = -1; }
    }
    net
}

/// Adaptive batch crystallize: remove edge batches, keep if accuracy holds.
#[allow(clippy::too_many_arguments)]
fn crystallize_pocket(
    female: &mut Network, male: &mut Network, proj: &Int8Projection,
    corpus: &[u8], sdr: &SdrTable, prop: &PropagationConfig,
    bigram: &[Vec<f64>], rng: &mut StdRng, target_pocket_is_male: bool,
) -> usize {
    let net = if target_pocket_is_male { &*male } else { &*female };
    let edges_before = net.edge_count();
    if edges_before < 20 { return 0; }

    let mut removal_pct = 0.30f64;
    let mut removed_total = 0usize;

    for _round in 0..12 {
        let target = if target_pocket_is_male { &*male } else { &*female };
        let edges_now = target.edge_count();
        if edges_now < 20 { break; }

        // Baseline
        let snap = rng.clone();
        let baseline = eval_smooth_chain(female, male, proj, corpus, 200, rng, sdr, prop, bigram);
        *rng = snap;

        // Pick random batch to remove
        let batch_size = ((edges_now as f64 * removal_pct) as usize).max(1);
        let target_mut = if target_pocket_is_male { &mut *male } else { &mut *female };
        let all_edges: Vec<_> = target_mut.graph().iter_edges().collect();
        let mut indices: Vec<usize> = (0..all_edges.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
        let batch: Vec<(u16, u16)> = indices[..batch_size.min(indices.len())]
            .iter().map(|&i| (all_edges[i].source, all_edges[i].target)).collect();

        for &(s, t) in &batch { target_mut.graph_mut().remove_edge(s, t); }

        let snap2 = rng.clone();
        let after = eval_smooth_chain(female, male, proj, corpus, 200, rng, sdr, prop, bigram);
        *rng = snap2;

        let target_check = if target_pocket_is_male { &mut *male } else { &mut *female };
        if after >= baseline - 0.01 {
            removal_pct = (removal_pct * 1.5).min(0.70);
            removed_total += batch.len();
        } else {
            for &(s, t) in &batch { target_check.graph_mut().add_edge(s, t); }
            removal_pct /= 2.0;
            if removal_pct < 0.03 { break; }
        }
    }
    removed_total
}

fn charge_transfer(female: &Network) -> Vec<i32> {
    let os = output_start();
    let mut input = vec![0i32; H];
    for (i, &c) in female.charge()[os..H].iter().enumerate() {
        if i < PHI_DIM { input[i] = c as i32; }
    }
    input
}

/// Smooth fitness: cosine to bigram over the full Female→Male chain.
#[allow(clippy::too_many_arguments)]
fn eval_smooth_chain(
    female: &mut Network, male: &mut Network, proj: &Int8Projection,
    corpus: &[u8], len: usize, rng: &mut StdRng,
    sdr: &SdrTable, prop: &PropagationConfig, bigram: &[Vec<f64>],
) -> f64 {
    if corpus.len() <= len { return 0.0; }
    let off = rng.gen_range(0..=corpus.len() - len - 1);
    let seg = &corpus[off..off + len + 1];
    let os = output_start();
    female.reset(); male.reset();
    let mut total_cos = 0.0f64;
    for i in 0..len {
        female.propagate(sdr.pattern(seg[i] as usize), prop).unwrap();
        let transfer = charge_transfer(female);
        male.propagate(&transfer, prop).unwrap();
        let scores = proj.raw_scores(&male.charge()[os..H]);
        let probs = softmax(&scores);
        total_cos += cosine_similarity(&probs, &bigram[seg[i] as usize]);
    }
    total_cos / len as f64
}

/// Argmax accuracy for reporting.
#[allow(clippy::too_many_arguments)]
fn eval_accuracy_chain(
    female: &mut Network, male: &mut Network, proj: &Int8Projection,
    corpus: &[u8], len: usize, rng: &mut StdRng,
    sdr: &SdrTable, prop: &PropagationConfig,
) -> f64 {
    if corpus.len() <= len { return 0.0; }
    let off = rng.gen_range(0..=corpus.len() - len - 1);
    let seg = &corpus[off..off + len + 1];
    let os = output_start();
    female.reset(); male.reset();
    let mut correct = 0u32;
    for i in 0..len {
        female.propagate(sdr.pattern(seg[i] as usize), prop).unwrap();
        let transfer = charge_transfer(female);
        male.propagate(&transfer, prop).unwrap();
        if proj.predict(&male.charge()[os..H]) == seg[i + 1] as usize { correct += 1; }
    }
    correct as f64 / len as f64
}

fn mutate_pocket(net: &mut Network, proj: &mut Int8Projection, is_male: bool, rng: &mut impl Rng) -> bool {
    let roll = rng.gen_range(0..100u32);
    match roll {
        0..22 => net.mutate_add_edge(rng),
        22..35 => net.mutate_remove_edge(rng),
        35..44 => net.mutate_rewire(rng),
        44..57 => net.mutate_reverse(rng),
        57..63 => net.mutate_mirror(rng),
        63..70 => net.mutate_enhance(rng),
        70..75 => net.mutate_theta(rng),
        75..85 => net.mutate_channel(rng),
        85..90 => net.mutate_add_loop(rng, 2),  // bidirectional pair
        90..95 => net.mutate_add_loop(rng, 3),  // triangle circuit
        _ => {
            if is_male { let _ = proj.mutate_one(rng); true }
            else { net.mutate_theta(rng) }
        }
    }
}

#[derive(Clone, Copy)]
enum Variant {
    CrystalFirst,  // prefill → crystallize → evolve (smart sparse start)
    Prefilled,     // prefill → evolve (control)
}
impl Variant {
    fn name(&self) -> &str { match self { Self::CrystalFirst => "crystal1st", Self::Prefilled => "prefill" } }
}

struct Config { variant: Variant, seed: u64 }

#[allow(dead_code)]
struct RunResult {
    variant_name: String, seed: u64,
    final_acc: f64, peak_acc: f64,
    f_edges: usize, m_edges: usize, accepted: u32,
}

fn run_one(cfg: &Config, corpus: &[u8], bigram: &[Vec<f64>]) -> RunResult {
    let prop = PropagationConfig {
        ticks_per_token: 6, input_duration_ticks: 2,
        decay_interval_ticks: 6, use_refractory: false,
    };

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    // Both variants start prefilled; CrystalFirst then prunes immediately.
    let mut female = build_prefilled_pocket(&mut rng);
    let mut male = build_prefilled_pocket(&mut rng);
    let mut proj = Int8Projection::new(PHI_DIM, CHARS, &mut StdRng::seed_from_u64(cfg.seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed + 1000);
    let sdr = SdrTable::new(CHARS, H, PHI_DIM, SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(cfg.seed + 100)).unwrap();

    // CrystalFirst: prefill both pockets, then crystallize to keep only useful edges.
    // This gives the network a smart sparse start — prefill provides paths,
    // crystallize removes what doesn't contribute.
    if matches!(cfg.variant, Variant::CrystalFirst) {
        let f_before = female.edge_count();
        let m_before = male.edge_count();

        let mut cryst_rng = StdRng::seed_from_u64(cfg.seed + 5000);
        let f_removed = crystallize_pocket(
            &mut female, &mut male, &proj, corpus, &sdr, &prop, bigram,
            &mut cryst_rng, false);
        let m_removed = crystallize_pocket(
            &mut female, &mut male, &proj, corpus, &sdr, &prop, bigram,
            &mut cryst_rng, true);

        println!("  {} seed={} crystallize: F={}→{} (-{}) M={}→{} (-{})",
            cfg.variant.name(), cfg.seed,
            f_before, female.edge_count(), f_removed,
            m_before, male.edge_count(), m_removed);
    }

    let init_f = female.edge_count();
    let init_m = male.edge_count();
    let mut peak_acc = 0.0f64;
    let mut accepted = 0u32;

    for step in 0..STEPS {
        // Paired eval
        let snap = eval_rng.clone();
        let before = eval_smooth_chain(&mut female, &mut male, &proj, corpus, 100,
            &mut eval_rng, &sdr, &prop, bigram);
        eval_rng = snap;

        let f_state = female.save_state();
        let m_state = male.save_state();
        let proj_backup = proj.clone();
        let f_edges_before = female.edge_count();
        let m_edges_before = male.edge_count();

        // Jackpot: 9 candidates
        let mut best_delta = f64::NEG_INFINITY;
        let mut best_f = None;
        let mut best_m = None;
        let mut best_p = None;
        let mut any = false;

        for c in 0..JACKPOT {
            female.restore_state(&f_state);
            male.restore_state(&m_state);
            proj = proj_backup.clone();

            let mut cr = StdRng::seed_from_u64(
                cfg.seed.wrapping_add(300).wrapping_add(step as u64 * 100 + c as u64));

            // Mutate one pocket (50/50 female/male)
            let target_male = cr.gen_bool(0.5);
            let mutated = if target_male {
                mutate_pocket(&mut male, &mut proj, true, &mut cr)
            } else {
                mutate_pocket(&mut female, &mut proj, false, &mut cr)
            };
            if !mutated { continue; }
            any = true;

            // Edge cap check per pocket
            let f_over = female.edge_count() > f_edges_before && female.edge_count() > EDGE_CAP_PER_POCKET;
            let m_over = male.edge_count() > m_edges_before && male.edge_count() > EDGE_CAP_PER_POCKET;
            if f_over || m_over { continue; }

            let cs = eval_rng.clone();
            let after = eval_smooth_chain(&mut female, &mut male, &proj, corpus, 100,
                &mut eval_rng, &sdr, &prop, bigram);
            eval_rng = cs;

            let delta = after - before;
            if delta > best_delta {
                best_delta = delta;
                best_f = Some(female.save_state());
                best_m = Some(male.save_state());
                best_p = Some(proj.clone());
            }
        }

        // Advance eval_rng
        female.restore_state(&f_state); male.restore_state(&m_state); proj = proj_backup.clone();
        let _ = eval_smooth_chain(&mut female, &mut male, &proj, corpus, 100,
            &mut eval_rng, &sdr, &prop, bigram);

        if !any { female.restore_state(&f_state); male.restore_state(&m_state); proj = proj_backup; continue; }

        if best_delta > 0.0 {
            if let (Some(fs), Some(ms), Some(ps)) = (best_f, best_m, best_p) {
                female.restore_state(&fs); male.restore_state(&ms); proj = ps;
                accepted += 1;
            }
        } else {
            female.restore_state(&f_state); male.restore_state(&m_state); proj = proj_backup;
        }

        if (step + 1) % 5_000 == 0 {
            let mut cr = StdRng::seed_from_u64(cfg.seed + 6000 + step as u64);
            let acc = eval_accuracy_chain(&mut female, &mut male, &proj, corpus, 2000,
                &mut cr, &sdr, &prop);
            if acc > peak_acc { peak_acc = acc; }
            println!("  {} seed={} step {:>5}: acc={:.1}% F={} M={} accepted={} (init F={} M={})",
                cfg.variant.name(), cfg.seed, step + 1, acc * 100.0,
                female.edge_count(), male.edge_count(), accepted, init_f, init_m);
        }
    }

    let mut fr = StdRng::seed_from_u64(cfg.seed + 9999);
    let final_acc = eval_accuracy_chain(&mut female, &mut male, &proj, corpus, 5000,
        &mut fr, &sdr, &prop);
    if final_acc > peak_acc { peak_acc = final_acc; }

    println!("  {} seed={} FINAL: acc={:.1}% peak={:.1}% F={} M={} accepted={}",
        cfg.variant.name(), cfg.seed, final_acc * 100.0, peak_acc * 100.0,
        female.edge_count(), male.edge_count(), accepted);

    RunResult {
        variant_name: cfg.variant.name().to_string(), seed: cfg.seed,
        final_acc, peak_acc,
        f_edges: female.edge_count(), m_edges: male.edge_count(), accepted,
    }
}

fn main() {
    rayon::ThreadPoolBuilder::new().num_threads(4).build_global().ok();

    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });

    println!("=== Pocket Pair: Crystal-First vs Prefilled ===");
    println!("  Both: smooth cosine-bigram fitness + jackpot=9");
    println!("  CrystalFirst: prefill → crystallize → evolve (smart sparse)");
    println!("  Prefilled: prefill → evolve (control)\n");

    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars", corpus.len());
    let bigram = build_bigram_table(&corpus, CHARS);
    println!();

    let seeds = [42u64, 123, 7, 1042, 555, 8042];

    let mut configs: Vec<Config> = Vec::new();
    for &v in &[Variant::CrystalFirst, Variant::Prefilled] {
        for &s in &seeds { configs.push(Config { variant: v, seed: s }); }
    }
    println!("  {} configs: 2 variants x {} seeds\n", configs.len(), seeds.len());

    let start = Instant::now();
    let results: Vec<RunResult> = configs.par_iter()
        .map(|cfg| run_one(cfg, &corpus, &bigram))
        .collect();
    let elapsed = start.elapsed().as_secs_f64();

    println!("\n=== SUMMARY ===\n");
    println!("{:<10} {:>7} {:>7} {:>7} {:>7} {:>7} {:>8}",
        "Variant", "Mean%", "Best%", "Peak%", "F_edg", "M_edg", "Accepted");
    println!("{}", "-".repeat(62));

    for v in &[Variant::CrystalFirst, Variant::Prefilled] {
        let g: Vec<_> = results.iter().filter(|r| r.variant_name == v.name()).collect();
        let n = g.len() as f64;
        let mean = g.iter().map(|r| r.final_acc).sum::<f64>() / n;
        let best = g.iter().map(|r| r.final_acc).fold(0.0f64, f64::max);
        let peak = g.iter().map(|r| r.peak_acc).fold(0.0f64, f64::max);
        let fe = g.iter().map(|r| r.f_edges).sum::<usize>() / g.len();
        let me = g.iter().map(|r| r.m_edges).sum::<usize>() / g.len();
        let ac = g.iter().map(|r| r.accepted as f64).sum::<f64>() / n;
        println!("{:<10} {:>6.1}% {:>6.1}% {:>6.1}% {:>7} {:>7} {:>8.0}",
            v.name(), mean * 100.0, best * 100.0, peak * 100.0, fe, me, ac);
    }

    println!("\nPer-seed:");
    println!("{:<10} {:>6} {:>7} {:>7} {:>7} {:>7} {:>8}",
        "Variant", "Seed", "Acc%", "Peak%", "F_edg", "M_edg", "Accepted");
    println!("{}", "-".repeat(58));
    for r in &results {
        println!("{:<10} {:>6} {:>6.1}% {:>6.1}% {:>7} {:>7} {:>8}",
            r.variant_name, r.seed, r.final_acc * 100.0, r.peak_acc * 100.0,
            r.f_edges, r.m_edges, r.accepted);
    }

    println!("\n  Total time: {:.0}s", elapsed);
}
