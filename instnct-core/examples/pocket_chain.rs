//! Pocket chain v2: one big network, spatial mutation constraints.
//!
//! Single Network with H=844 (4 pockets × 256, overlapping by 60).
//! SDR input only to pocket A. W readout only from pocket D.
//! Each mutation step picks a random pocket and mutates within it.
//! Propagation runs the full network. Fitness = global accuracy.
//!
//! Run: cargo run --example pocket_chain --release -- <corpus-path>

use instnct_core::{load_corpus, Int8Projection, Network, SdrTable};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const POCKET_H: usize = 256;
const POCKET_PHI: usize = 158;
const POCKET_OVERLAP: usize = 60; // phi overlap between adjacent pockets
const POCKET_STEP: usize = POCKET_H - POCKET_OVERLAP; // 196 unique neurons per pocket

/// Pocket zone descriptor.
struct PocketZone {
    start: usize,
    end: usize,
}

impl PocketZone {
    fn contains(&self, neuron: usize) -> bool {
        neuron >= self.start && neuron < self.end
    }
}

fn total_neurons(n_pockets: usize) -> usize {
    POCKET_H + (n_pockets - 1) * POCKET_STEP
}

fn pocket_zone(pocket_idx: usize) -> PocketZone {
    let start = pocket_idx * POCKET_STEP;
    PocketZone { start, end: start + POCKET_H }
}

/// SDR input zone: first pocket's input range [0, phi_dim).
fn sdr_input_end(_n_pockets: usize) -> usize {
    POCKET_PHI
}

/// W output zone: last pocket's output range.
fn output_start(n_pockets: usize) -> usize {
    let last = pocket_zone(n_pockets - 1);
    last.start + (POCKET_H - POCKET_PHI) // output_start within last pocket
}


/// Add a random edge within a pocket zone.
fn pocket_add_edge(net: &mut Network, zone: &PocketZone, rng: &mut impl Rng) -> bool {
    let range = zone.end - zone.start;
    if range < 2 { return false; }
    for _ in 0..30 {
        let src = zone.start + rng.gen_range(0..range);
        let tgt = zone.start + rng.gen_range(0..range);
        if src != tgt && net.graph_mut().add_edge(src as u16, tgt as u16) {
            return true;
        }
    }
    false
}

/// Remove a random edge within a pocket zone.
fn pocket_remove_edge(net: &mut Network, zone: &PocketZone, rng: &mut impl Rng) -> bool {
    let edges: Vec<_> = net.graph().iter_edges()
        .filter(|e| zone.contains(e.source as usize) && zone.contains(e.target as usize))
        .collect();
    if edges.is_empty() { return false; }
    let e = edges[rng.gen_range(0..edges.len())];
    net.graph_mut().remove_edge(e.source, e.target);
    true
}

/// Rewire a random edge within a pocket zone.
fn pocket_rewire(net: &mut Network, zone: &PocketZone, rng: &mut impl Rng) -> bool {
    let edges: Vec<_> = net.graph().iter_edges()
        .filter(|e| zone.contains(e.source as usize) && zone.contains(e.target as usize))
        .collect();
    if edges.is_empty() { return false; }
    let e = edges[rng.gen_range(0..edges.len())];
    let range = zone.end - zone.start;
    for _ in 0..30 {
        let new_tgt = zone.start + rng.gen_range(0..range);
        if new_tgt != e.source as usize {
            net.graph_mut().remove_edge(e.source, e.target);
            if net.graph_mut().add_edge(e.source, new_tgt as u16) { return true; }
            net.graph_mut().add_edge(e.source, e.target);
            return false;
        }
    }
    false
}

/// Reverse a random edge within a pocket zone.
fn pocket_reverse(net: &mut Network, zone: &PocketZone, rng: &mut impl Rng) -> bool {
    let edges: Vec<_> = net.graph().iter_edges()
        .filter(|e| zone.contains(e.source as usize) && zone.contains(e.target as usize))
        .collect();
    if edges.is_empty() { return false; }
    let e = edges[rng.gen_range(0..edges.len())];
    net.graph_mut().remove_edge(e.source, e.target);
    if net.graph_mut().add_edge(e.target, e.source) { return true; }
    net.graph_mut().add_edge(e.source, e.target);
    false
}

/// Mutate a random neuron param within a pocket zone.
fn pocket_param(net: &mut Network, zone: &PocketZone, rng: &mut impl Rng) -> bool {
    let range = zone.end - zone.start;
    let idx = zone.start + rng.gen_range(0..range);
    let roll = rng.gen_range(0..3u32);
    match roll {
        0 => { net.threshold_mut()[idx] = rng.gen_range(0..=7); true }
        1 => { net.channel_mut()[idx] = rng.gen_range(1..=8); true }
        _ => { net.polarity_mut()[idx] *= -1; true }
    }
}

/// One mutation step within a pocket.
fn pocket_mutate(
    net: &mut Network, proj: &mut Int8Projection,
    zone: &PocketZone, rng: &mut impl Rng, is_last: bool,
) -> bool {
    let roll = rng.gen_range(0..100u32);
    match roll {
        0..25  => pocket_add_edge(net, zone, rng),
        25..40 => pocket_remove_edge(net, zone, rng),
        40..55 => pocket_rewire(net, zone, rng),
        55..70 => pocket_reverse(net, zone, rng),
        70..85 => pocket_param(net, zone, rng),
        _ => if is_last { let _ = proj.mutate_one(rng); true } else { pocket_param(net, zone, rng) },
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_accuracy(
    net: &mut Network, proj: &Int8Projection, corpus: &[u8], len: usize,
    rng: &mut StdRng, sdr: &SdrTable, config: &instnct_core::PropagationConfig,
    out_start: usize, total_h: usize,
) -> f64 {
    if corpus.len() <= len { return 0.0; }
    let off = rng.gen_range(0..=corpus.len() - len - 1);
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), config).unwrap();
        if proj.predict(&net.charge()[out_start..total_h]) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

fn bar(val: f64, max_val: f64, width: usize) -> String {
    let filled = ((val / max_val) * width as f64).round().min(width as f64) as usize;
    format!("{}{}", "#".repeat(filled), ".".repeat(width - filled))
}

#[allow(dead_code)]
struct RunResult {
    n_pockets: usize,
    seed: u64,
    final_acc: f64,
    peak_acc: f64,
    final_edges: usize,
}

fn run_one(n_pockets: usize, seed: u64, steps: usize, corpus: &[u8]) -> RunResult {
    let h = total_neurons(n_pockets);
    let out_start = output_start(n_pockets);
    let sdr_end = sdr_input_end(n_pockets);
    let out_dim = h - out_start; // W projection dimension

    // Propagation config (use standard)
    let prop = instnct_core::PropagationConfig {
        ticks_per_token: 6,
        input_duration_ticks: 2,
        decay_interval_ticks: 6,
        use_refractory: false,
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = Network::new(h);

    // Init each pocket with chain-50 + 5% density + random params
    for p in 0..n_pockets {
        let zone = pocket_zone(p);
        let zone_h = zone.end - zone.start;
        let phi = (zone_h as f64 / 1.618).round() as usize;
        let zone_os = zone.start + zone_h - phi; // output_start within zone
        let zone_ie = zone.start + phi;          // input_end within zone
        let zone_om = (zone_os + zone_ie) / 2;

        // Chain-50 within pocket
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

        // 5% density fill within pocket
        let target = zone_h * zone_h * 5 / 100;
        for _ in 0..target * 3 {
            pocket_add_edge(&mut net, &zone, &mut rng);
            let pocket_edges: usize = net.graph().iter_edges()
                .filter(|e| zone.contains(e.source as usize) && zone.contains(e.target as usize))
                .count();
            if pocket_edges >= target { break; }
        }

        // Random params
        for i in zone.start..zone.end {
            net.threshold_mut()[i] = rng.gen_range(0..=7u8);
            net.channel_mut()[i] = rng.gen_range(1..=8u8);
            if rng.gen_ratio(1, 10) { net.polarity_mut()[i] = -1; }
        }
    }

    // SDR for first pocket's input zone
    let sdr = SdrTable::new(CHARS, h, sdr_end, SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(seed + 100)).unwrap();

    // W projection from last pocket's output zone
    let mut proj = Int8Projection::new(out_dim, CHARS,
        &mut StdRng::seed_from_u64(seed + 200));

    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let mut peak_acc = 0.0f64;

    for step in 0..steps {
        // Paired eval
        let snap = eval_rng.clone();
        let before = eval_accuracy(&mut net, &proj, corpus, 100, &mut eval_rng, &sdr, &prop, out_start, h);
        eval_rng = snap;

        let state = net.save_state();

        // Pick random pocket, mutate within it
        let pocket_idx = rng.gen_range(0..n_pockets);
        let zone = pocket_zone(pocket_idx);
        let is_last = pocket_idx == n_pockets - 1;
        let mutated = pocket_mutate(&mut net, &mut proj, &zone, &mut rng, is_last);

        if !mutated {
            let _ = eval_accuracy(&mut net, &proj, corpus, 100, &mut eval_rng, &sdr, &prop, out_start, h);
            continue;
        }

        let after = eval_accuracy(&mut net, &proj, corpus, 100, &mut eval_rng, &sdr, &prop, out_start, h);
        let accepted = after > before; // strict, no ties

        if !accepted {
            net.restore_state(&state);
        }

        if (step + 1) % 5000 == 0 {
            let mut cr = StdRng::seed_from_u64(seed + 6000 + step as u64);
            let acc = eval_accuracy(&mut net, &proj, corpus, 2000, &mut cr, &sdr, &prop, out_start, h);
            if acc > peak_acc { peak_acc = acc; }
            println!("  {}pocket seed={:<4} step {:>5}: |{}| {:.1}%  edges={}",
                n_pockets, seed, step + 1, bar(acc, 0.30, 20), acc * 100.0, net.edge_count());
        }
    }

    let mut fr = StdRng::seed_from_u64(seed + 9999);
    let final_acc = eval_accuracy(&mut net, &proj, corpus, 5000, &mut fr, &sdr, &prop, out_start, h);
    println!("  {}pocket seed={:<4} FINAL: {:.1}%  peak={:.1}%  edges={}",
        n_pockets, seed, final_acc * 100.0, peak_acc * 100.0, net.edge_count());

    RunResult { n_pockets, seed, final_acc, peak_acc, final_edges: net.edge_count() }
}

fn main() {
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars\n", corpus.len());

    let pocket_counts = [1, 2, 4, 6];
    let seeds = [42u64, 123, 7];
    let steps = 30_000;

    println!("Pocket layout:");
    for &np in &pocket_counts {
        let h = total_neurons(np);
        let os = output_start(np);
        println!("  {} pockets: H={}, SDR input [0..{}), W output [{}..{}), {} overlap zones",
            np, h, sdr_input_end(np), os, h, np.saturating_sub(1));
    }
    println!();

    let mut configs: Vec<(usize, u64)> = Vec::new();
    for &np in &pocket_counts {
        for &s in &seeds {
            configs.push((np, s));
        }
    }

    println!("=== Pocket Chain: {} configs, {} steps ===\n", configs.len(), steps);

    let results: Vec<RunResult> = configs.par_iter()
        .map(|&(np, s)| run_one(np, s, steps, &corpus))
        .collect();

    println!("\n=== SUMMARY ===\n");
    println!("{:<10} {:>5} {:>8} {:>8} {:>8} {:>8}",
        "pockets", "H", "mean%", "best%", "peak%", "edges");
    println!("{}", "-".repeat(52));

    for &np in &pocket_counts {
        let g: Vec<_> = results.iter().filter(|r| r.n_pockets == np).collect();
        let mean = g.iter().map(|r| r.final_acc).sum::<f64>() / g.len() as f64;
        let best = g.iter().map(|r| r.final_acc).fold(0.0f64, f64::max);
        let peak = g.iter().map(|r| r.peak_acc).fold(0.0f64, f64::max);
        let edges = g.iter().map(|r| r.final_edges).sum::<usize>() / g.len();
        println!("{:<10} {:>5} {:>7.1}% {:>7.1}% {:>7.1}% {:>8}",
            format!("{} pocket", np), total_neurons(np), mean * 100.0, best * 100.0, peak * 100.0, edges);
    }
}
