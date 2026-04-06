//! Continue evolving a saved pocket pair from checkpoints.
//!
//! Loads A_merged female+male pockets, evolves for STEPS more steps.
//! Run: cargo run --example pocket_continue --release -- <corpus-path>

use instnct_core::{
    load_checkpoint, save_checkpoint, CheckpointMeta, Int8Projection, Network,
    PropagationConfig, SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};
use std::fs;
use std::time::Instant;

const MASTER_SEED: u64 = 1337;
const H: usize = 256;
const PHI_DIM: usize = 158;
const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const STEPS: usize = 30_000;
const EVAL_LEN_SHORT: usize = 100;
const EVAL_LEN_LONG: usize = 2000;
const LOG_INTERVAL: usize = 5000;

const FEMALE_CKPT: &str = "checkpoints/pocket_pair/A_merged_female.ckpt";
const MALE_CKPT: &str = "checkpoints/pocket_pair/A_merged_male.ckpt";
const OUT_DIR: &str = "checkpoints/pocket_continue";

fn output_start() -> usize { H - PHI_DIM }

fn charge_transfer(female: &Network) -> Vec<i32> {
    let os = output_start();
    let mut input = vec![0i32; H];
    for (i, &c) in female.charge()[os..H].iter().enumerate() {
        if i < PHI_DIM { input[i] = c as i32; }
    }
    input
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

fn mutate_unit(
    female: &mut Network, male: &mut Network, proj: &mut Int8Projection,
    rng: &mut impl Rng,
) -> bool {
    let target_male = rng.gen_bool(0.5);
    let net = if target_male { male } else { female };
    let roll = rng.gen_range(0..100u32);
    match roll {
        0..25 => {
            for _ in 0..30 {
                let s = rng.gen_range(0..H) as u16;
                let t = rng.gen_range(0..H) as u16;
                if s != t && net.graph_mut().add_edge(s, t) { return true; }
            }
            false
        }
        25..40 => {
            let edges: Vec<_> = net.graph().iter_edges().collect();
            if edges.is_empty() { return false; }
            let e = edges[rng.gen_range(0..edges.len())];
            net.graph_mut().remove_edge(e.source, e.target);
            true
        }
        40..55 => {
            let edges: Vec<_> = net.graph().iter_edges().collect();
            if edges.is_empty() { return false; }
            let e = edges[rng.gen_range(0..edges.len())];
            for _ in 0..30 {
                let new_t = rng.gen_range(0..H) as u16;
                if new_t != e.source {
                    net.graph_mut().remove_edge(e.source, e.target);
                    if net.graph_mut().add_edge(e.source, new_t) { return true; }
                    net.graph_mut().add_edge(e.source, e.target);
                    return false;
                }
            }
            false
        }
        55..70 => {
            let edges: Vec<_> = net.graph().iter_edges().collect();
            if edges.is_empty() { return false; }
            let e = edges[rng.gen_range(0..edges.len())];
            net.graph_mut().remove_edge(e.source, e.target);
            if net.graph_mut().add_edge(e.target, e.source) { return true; }
            net.graph_mut().add_edge(e.source, e.target);
            false
        }
        70..85 => {
            let idx = rng.gen_range(0..H);
            match rng.gen_range(0..3u32) {
                0 => { net.threshold_mut()[idx] = rng.gen_range(0..=7); true }
                1 => { net.channel_mut()[idx] = rng.gen_range(1..=8); true }
                _ => { net.polarity_mut()[idx] *= -1; true }
            }
        }
        _ => {
            if target_male { let _ = proj.mutate_one(rng); true }
            else {
                let idx = rng.gen_range(0..H);
                net.threshold_mut()[idx] = rng.gen_range(0..=7);
                true
            }
        }
    }
}

fn bar(val: f64, max_val: f64, width: usize) -> String {
    let filled = ((val / max_val) * width as f64).round().min(width as f64) as usize;
    format!("{}{}", "#".repeat(filled), ".".repeat(width - filled))
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

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "S:/AI/work/VRAXION_DEV/instnct/data/traindat/fineweb_edu.traindat".to_string()
    });

    // Which generation is this?
    let gen: usize = std::env::args().nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    let prop = PropagationConfig {
        ticks_per_token: 6, input_duration_ticks: 2,
        decay_interval_ticks: 6, use_refractory: false,
    };

    println!("=== POCKET CONTINUE — Gen {} ===", gen);
    println!("  Loading female: {}", FEMALE_CKPT);
    println!("  Loading male:   {}", MALE_CKPT);

    let (female, proj, f_meta) = load_checkpoint(FEMALE_CKPT).expect("cannot load female");
    let (male, _, m_meta) = load_checkpoint(MALE_CKPT).expect("cannot load male");

    println!("  Female: {} edges, from: {}", female.edge_count(), f_meta.label);
    println!("  Male:   {} edges, from: {}", male.edge_count(), m_meta.label);
    println!("  Previous accuracy: {:.2}%", f_meta.accuracy * 100.0);

    let corpus = load_corpus(&corpus_path);
    println!("  Corpus: {} chars", corpus.len());

    let mut seed_gen = StdRng::seed_from_u64(MASTER_SEED);
    let sdr_seed = seed_gen.next_u64();
    let sdr = SdrTable::new(CHARS, H, PHI_DIM, SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(sdr_seed)).unwrap();

    // Different RNG seeds per generation so we don't repeat the same path
    let gen_offset = gen as u64 * 1_000_000;
    let mut mut_rng = StdRng::seed_from_u64(MASTER_SEED.wrapping_add(55555 + gen_offset));
    let mut eval_rng = StdRng::seed_from_u64(MASTER_SEED.wrapping_add(99999 + gen_offset));

    let mut female = female;
    let mut male = male;
    let mut proj = proj;
    let mut peak = f_meta.accuracy;
    let mut accepted = 0u32;
    let mut total_tried = 0u32;

    // Initial eval
    let mut init_rng = StdRng::seed_from_u64(MASTER_SEED.wrapping_add(77777 + gen_offset));
    let init_acc = eval_chain(&mut female, &mut male, &proj, &corpus, EVAL_LEN_LONG,
        &mut init_rng, &sdr, &prop);
    println!("\n  Starting accuracy: {:.2}%  F_edges={}  M_edges={}\n",
        init_acc * 100.0, female.edge_count(), male.edge_count());

    let start = Instant::now();

    for step in 0..STEPS {
        let snap = eval_rng.clone();
        let before = eval_chain(&mut female, &mut male, &proj, &corpus, EVAL_LEN_SHORT,
            &mut eval_rng, &sdr, &prop);
        eval_rng = snap;

        let f_state = female.save_state();
        let m_state = male.save_state();
        let proj_backup = proj.clone();

        let mutated = mutate_unit(&mut female, &mut male, &mut proj, &mut mut_rng);

        if !mutated {
            let _ = eval_chain(&mut female, &mut male, &proj, &corpus, EVAL_LEN_SHORT,
                &mut eval_rng, &sdr, &prop);
            continue;
        }

        total_tried += 1;

        let after = eval_chain(&mut female, &mut male, &proj, &corpus, EVAL_LEN_SHORT,
            &mut eval_rng, &sdr, &prop);

        if after > before {
            accepted += 1;
        } else {
            female.restore_state(&f_state);
            male.restore_state(&m_state);
            proj = proj_backup;
        }

        if (step + 1) % LOG_INTERVAL == 0 {
            let mut cr = StdRng::seed_from_u64(MASTER_SEED.wrapping_add(6000 + step as u64 + gen_offset));
            let acc = eval_chain(&mut female, &mut male, &proj, &corpus, EVAL_LEN_LONG,
                &mut cr, &sdr, &prop);
            if acc > peak { peak = acc; }

            let int_rate = if total_tried > 0 {
                accepted as f64 / total_tried as f64
            } else { 0.0 };

            println!("  [{:>5}] Gen{} |{}|{:.1}% F={} M={} accept={:.1}% peak={:.1}%",
                step + 1, gen, bar(acc, 0.30, 15), acc * 100.0,
                female.edge_count(), male.edge_count(),
                int_rate * 100.0, peak * 100.0);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();

    let mut fr = StdRng::seed_from_u64(MASTER_SEED.wrapping_add(9999 + gen_offset));
    let final_acc = eval_chain(&mut female, &mut male, &proj, &corpus, EVAL_LEN_LONG,
        &mut fr, &sdr, &prop);
    if final_acc > peak { peak = final_acc; }

    let rate = if total_tried > 0 { accepted as f64 / total_tried as f64 * 100.0 } else { 0.0 };
    println!("\n=== Gen {} RESULT ===", gen);
    println!("  Final: {:.2}%  Peak: {:.2}%  F_edges={}  M_edges={}",
        final_acc * 100.0, peak * 100.0, female.edge_count(), male.edge_count());
    println!("  Accepted: {}/{} ({:.1}%)  Time: {:.1}s",
        accepted, total_tried, rate, elapsed);
    println!("  vs start ({:.2}%): {:+.2}pp",
        init_acc * 100.0, (final_acc - init_acc) * 100.0);

    // Save — these become the input for next gen
    fs::create_dir_all(OUT_DIR).ok();
    let _ = save_checkpoint(
        format!("{}/gen{}_female.ckpt", OUT_DIR, gen), &female, &proj,
        CheckpointMeta { step: STEPS * gen, accuracy: final_acc, label: format!("gen{}_female", gen) },
    );
    let _ = save_checkpoint(
        format!("{}/gen{}_male.ckpt", OUT_DIR, gen), &male, &proj,
        CheckpointMeta { step: STEPS * gen, accuracy: final_acc, label: format!("gen{}_male", gen) },
    );
    println!("  Saved to {}/gen{}_*.ckpt", OUT_DIR, gen);
}
