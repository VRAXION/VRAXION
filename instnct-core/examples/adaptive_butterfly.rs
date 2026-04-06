//! Adaptive butterfly: bit-distance limit scales with log2(H).
//!
//! At H=256 (8bit), ≤2bit = 14.1% search space — the sweet spot.
//! At H=1024 (10bit), ≤2bit = only 5.4% — too restrictive!
//! Fix: use ≤3bit at H=1024 to match the ~14% proportion.
//!
//! Tests: {H=256, 512, 1024, 2048} × {adaptive, fixed-2bit, random} × 3 seeds
//! Measures accuracy AND propagation speed.
//!
//! Run: cargo run --example adaptive_butterfly --release -- <corpus-path>

use instnct_core::{load_corpus, InitConfig, Int8Projection, Network, SdrTable};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;

/// Adaptive bit limit: target ~14% of search space (matching H=256 ≤2bit).
fn adaptive_bits(h: usize) -> u32 {
    let addr_bits = (h as f64).log2().ceil() as u32;
    // Find max_bits where C(addr,1)+...+C(addr,max_bits) / (h-1) ≈ 14%
    let target_pct = 0.14;
    for bits in 1..=addr_bits {
        let mut allowed = 0u64;
        for d in 1..=bits {
            allowed += n_choose_k(addr_bits, d);
        }
        let pct = allowed as f64 / (h - 1) as f64;
        if pct >= target_pct { return bits; }
    }
    addr_bits
}

fn n_choose_k(n: u32, k: u32) -> u64 {
    if k > n { return 0; }
    let mut result = 1u64;
    for i in 0..k {
        result = result * (n - i) as u64 / (i + 1) as u64;
    }
    result
}

fn can_connect(src: usize, tgt: usize, max_bits: u32) -> bool {
    src != tgt && (src ^ tgt).count_ones() <= max_bits
}

fn constrained_add(net: &mut Network, max_bits: u32, rng: &mut impl Rng) -> bool {
    let h = net.neuron_count();
    for _ in 0..50 {
        let s = rng.gen_range(0..h);
        let t = rng.gen_range(0..h);
        if can_connect(s, t, max_bits) {
            return net.graph_mut().add_edge(s as u16, t as u16);
        }
    }
    false
}

fn constrained_rewire(net: &mut Network, max_bits: u32, rng: &mut impl Rng) -> bool {
    let h = net.neuron_count();
    let edges: Vec<_> = net.graph().iter_edges().collect();
    if edges.is_empty() { return false; }
    let e = edges[rng.gen_range(0..edges.len())];
    for _ in 0..50 {
        let t = rng.gen_range(0..h);
        if can_connect(e.source as usize, t, max_bits) {
            net.graph_mut().remove_edge(e.source, e.target);
            if net.graph_mut().add_edge(e.source, t as u16) { return true; }
            net.graph_mut().add_edge(e.source, e.target);
            return false;
        }
    }
    false
}


#[allow(clippy::too_many_arguments)]
fn eval_accuracy(
    net: &mut Network, proj: &Int8Projection, corpus: &[u8], len: usize,
    rng: &mut StdRng, sdr: &SdrTable, init: &InitConfig,
) -> f64 {
    if corpus.len() <= len { return 0.0; }
    let off = rng.gen_range(0..=corpus.len() - len - 1);
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), &init.propagation).unwrap();
        if proj.predict(&net.charge()[init.output_start()..init.neuron_count]) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

fn measure_prop_speed(net: &mut Network, sdr: &SdrTable, init: &InitConfig) -> u64 {
    let iters = 200u64;
    net.reset();
    for i in 0..10 { net.propagate(sdr.pattern(i % CHARS), &init.propagation).unwrap(); }
    net.reset();
    let t0 = Instant::now();
    for i in 0..iters as usize {
        net.propagate(sdr.pattern(i % CHARS), &init.propagation).unwrap();
    }
    t0.elapsed().as_nanos() as u64 / iters
}

#[derive(Clone, Copy)]
enum Topology { Adaptive, Fixed2bit, Random }

#[allow(dead_code)]
struct Res {
    h: usize, topo: &'static str, bits: u32, seed: u64,
    acc: f64, edges: usize, prop_ns: u64, search_pct: f64,
}

fn run_one(h: usize, topo: Topology, seed: u64, corpus: &[u8]) -> Res {
    let init = InitConfig::phi(h);
    let edge_cap = init.edge_cap();

    let (bits, topo_label) = match topo {
        Topology::Adaptive => (adaptive_bits(h), "adaptive"),
        Topology::Fixed2bit => (2, "fixed-2"),
        Topology::Random => (32, "random"), // 32 bits = no constraint
    };

    let is_random = bits >= 16;

    // Calculate search space
    let search_pct = if is_random {
        100.0
    } else {
        let mut allowed = 0u64;
        for s in 0..h { for t in 0..h { if can_connect(s, t, bits) { allowed += 1; } } }
        allowed as f64 / (h * h) as f64 * 100.0
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = Network::new(h);

    // Chain init for small H
    let os = init.output_start();
    let oe = init.input_end();
    if init.chain_count > 0 && oe > os + 1 {
        let om = (os + oe) / 2;
        for _ in 0..init.chain_count {
            let s = rng.gen_range(0..os) as u16;
            let h1 = rng.gen_range(os..om) as u16;
            let h2 = rng.gen_range(om..oe) as u16;
            let t = rng.gen_range(oe..h) as u16;
            net.graph_mut().add_edge(s, h1);
            net.graph_mut().add_edge(h1, h2);
            net.graph_mut().add_edge(h2, t);
        }
    }

    // Fill to 5%
    let target = h * h * 5 / 100;
    for _ in 0..target * 5 {
        if is_random { net.mutate_add_edge(&mut rng); }
        else { constrained_add(&mut net, bits, &mut rng); }
        if net.edge_count() >= target { break; }
    }

    for i in 0..h {
        net.threshold_mut()[i] = rng.gen_range(0..=7u32);
        net.channel_mut()[i] = rng.gen_range(1..=8u8);
        if rng.gen_ratio(1, 10) { net.polarity_mut()[i] = -1; }
    }

    let mut proj = Int8Projection::new(init.phi_dim, CHARS, &mut StdRng::seed_from_u64(seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let sdr = SdrTable::new(CHARS, h, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(seed + 100)).unwrap();

    let steps = 15000;
    for _ in 0..steps {
        let snap = eval_rng.clone();
        let before = eval_accuracy(&mut net, &proj, corpus, 100, &mut eval_rng, &sdr, &init);
        eval_rng = snap;

        let state = net.save_state();
        let mut wb = None;
        let roll = rng.gen_range(0..100u32);
        let ok = match roll {
            0..25 => if is_random { net.mutate_add_edge(&mut rng) }
                     else { constrained_add(&mut net, bits, &mut rng) },
            25..40 => net.mutate_remove_edge(&mut rng),
            40..50 => if is_random { net.mutate_rewire(&mut rng) }
                      else { constrained_rewire(&mut net, bits, &mut rng) },
            50..65 => net.mutate_reverse(&mut rng),
            65..72 => net.mutate_mirror(&mut rng),
            72..80 => net.mutate_enhance(&mut rng),
            80..85 => net.mutate_theta(&mut rng),
            85..90 => net.mutate_channel(&mut rng),
            _ => { wb = Some(proj.mutate_one(&mut rng)); true }
        };
        if !ok {
            let _ = eval_accuracy(&mut net, &proj, corpus, 100, &mut eval_rng, &sdr, &init);
            continue;
        }
        let after = eval_accuracy(&mut net, &proj, corpus, 100, &mut eval_rng, &sdr, &init);
        let accepted = if net.edge_count() < edge_cap { after >= before } else { after > before };
        if !accepted {
            net.restore_state(&state);
            if let Some(b) = wb { proj.rollback(b); }
        }
    }

    let prop_ns = measure_prop_speed(&mut net, &sdr, &init);
    let acc = eval_accuracy(&mut net, &proj, corpus, 5000, &mut eval_rng, &sdr, &init);

    println!("  H={:<5} {:<10} ≤{}bit  seed={:<5} -> {:.1}%  edges={}  prop={}ns  space={:.1}%",
        h, topo_label, bits, seed, acc * 100.0, net.edge_count(), prop_ns, search_pct);

    Res { h, topo: topo_label, bits, seed, acc, edges: net.edge_count(), prop_ns, search_pct }
}

fn main() {
    let seeds = [42u64, 123, 7];
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars\n", corpus.len());

    let sizes = [256, 512, 1024, 2048];

    // Show adaptive bit limits
    println!("Adaptive bit limits:");
    for &h in &sizes {
        let bits = adaptive_bits(h);
        let addr = (h as f64).log2().ceil() as u32;
        let mut allowed = 0u64;
        for d in 1..=bits { allowed += n_choose_k(addr, d); }
        let pct = allowed as f64 / (h - 1) as f64 * 100.0;
        println!("  H={:<5} addr={}bit  adaptive=≤{}bit  search={:.1}%", h, addr, bits, pct);
    }
    println!();

    let mut cfgs: Vec<(usize, Topology, u64)> = Vec::new();
    for &h in &sizes {
        for &seed in &seeds {
            cfgs.push((h, Topology::Random, seed));
            cfgs.push((h, Topology::Fixed2bit, seed));
            cfgs.push((h, Topology::Adaptive, seed));
        }
    }

    println!("=== Adaptive Butterfly: {} configs ===\n", cfgs.len());

    let results: Vec<Res> = cfgs.par_iter()
        .map(|&(h, topo, seed)| run_one(h, topo, seed, &corpus))
        .collect();

    println!("\n=== SUMMARY ===\n");
    println!("{:<6} {:<10} {:>5} {:>8} {:>7} {:>7} {:>10}",
        "H", "topology", "bits", "space%", "mean%", "edges", "prop_ns");
    println!("{}", "-".repeat(62));

    for &h in &sizes {
        for topo_label in ["random", "fixed-2", "adaptive"] {
            let g: Vec<_> = results.iter()
                .filter(|r| r.h == h && r.topo == topo_label)
                .collect();
            if g.is_empty() { continue; }
            let ma = g.iter().map(|r| r.acc).sum::<f64>() / g.len() as f64;
            let me = g.iter().map(|r| r.edges).sum::<usize>() / g.len();
            let mp = g.iter().map(|r| r.prop_ns).sum::<u64>() / g.len() as u64;
            let sp = g[0].search_pct;
            let bits = g[0].bits;
            println!("{:<6} {:<10} {:>4} {:>7.1}% {:>6.1}% {:>7} {:>9}ns",
                h, topo_label, bits, sp, ma * 100.0, me, mp);
        }
        println!();
    }
}
