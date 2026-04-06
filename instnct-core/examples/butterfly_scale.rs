//! Butterfly scaling test: does ≤2bit XOR constraint help at larger H?
//!
//! Tests H=256, 512, 1024 with butterfly ≤2bit vs random.
//! Measures both accuracy AND propagation speed.
//!
//! Run: cargo run --example butterfly_scale --release -- <corpus-path>

use instnct_core::{load_corpus, InitConfig, Int8Projection, Network, SdrTable};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;

fn butterfly_ok(src: usize, tgt: usize) -> bool {
    src != tgt && (1..=2).contains(&(src ^ tgt).count_ones())
}

fn bf_add(net: &mut Network, rng: &mut impl Rng) -> bool {
    let h = net.neuron_count();
    for _ in 0..30 {
        let s = rng.gen_range(0..h);
        let t = rng.gen_range(0..h);
        if butterfly_ok(s, t) { return net.graph_mut().add_edge(s as u16, t as u16); }
    }
    false
}

fn bf_rewire(net: &mut Network, rng: &mut impl Rng) -> bool {
    let h = net.neuron_count();
    let edges: Vec<_> = net.graph().iter_edges().collect();
    if edges.is_empty() { return false; }
    let e = edges[rng.gen_range(0..edges.len())];
    for _ in 0..30 {
        let t = rng.gen_range(0..h);
        if butterfly_ok(e.source as usize, t) {
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

struct Cfg { h: usize, butterfly: bool, seed: u64 }

#[allow(dead_code)]
struct Res { h: usize, butterfly: bool, seed: u64, acc: f64, edges: usize, prop_ns: u64 }

fn run_one(cfg: &Cfg, corpus: &[u8]) -> Res {
    let init = InitConfig::phi(cfg.h);
    let h = cfg.h;
    let edge_cap = init.edge_cap();

    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut net = Network::new(h);

    // Chain init (if H<512, chain_count=50; else 0 per InitConfig)
    let os = init.output_start();
    let oe = init.input_end();
    if oe > os + 1 {
        let om = (os + oe) / 2;
        let chains = if h < 512 { 50 } else { 0 };
        for _ in 0..chains {
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
        if cfg.butterfly { bf_add(&mut net, &mut rng); } else { net.mutate_add_edge(&mut rng); }
        if net.edge_count() >= target { break; }
    }

    for i in 0..h {
        net.threshold_mut()[i] = rng.gen_range(0..=7u32);
        net.channel_mut()[i] = rng.gen_range(1..=8u8);
        if rng.gen_ratio(1, 10) { net.polarity_mut()[i] = -1; }
    }

    let mut proj = Int8Projection::new(init.phi_dim, CHARS, &mut StdRng::seed_from_u64(cfg.seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(cfg.seed + 1000);
    let sdr = SdrTable::new(CHARS, h, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(cfg.seed + 100)).unwrap();

    let steps = 15000;
    for _ in 0..steps {
        let snap = eval_rng.clone();
        let before = eval_accuracy(&mut net, &proj, corpus, 100, &mut eval_rng, &sdr, &init);
        eval_rng = snap;

        let state = net.save_state();
        let mut wb = None;
        let roll = rng.gen_range(0..100u32);
        let ok = match roll {
            0..25 => if cfg.butterfly { bf_add(&mut net, &mut rng) } else { net.mutate_add_edge(&mut rng) },
            25..40 => net.mutate_remove_edge(&mut rng),
            40..50 => if cfg.butterfly { bf_rewire(&mut net, &mut rng) } else { net.mutate_rewire(&mut rng) },
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
    let label = if cfg.butterfly { "butterfly" } else { "random" };
    println!("  H={:<5} {:<10} seed={:<5} -> {:.1}%  edges={}  prop={}ns",
        h, label, cfg.seed, acc * 100.0, net.edge_count(), prop_ns);

    Res { h, butterfly: cfg.butterfly, seed: cfg.seed, acc, edges: net.edge_count(), prop_ns }
}

fn main() {
    let seeds = [42u64, 123, 7];
    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars\n", corpus.len());

    let mut cfgs = Vec::new();
    for &h in &[256, 512, 1024] {
        for &seed in &seeds {
            cfgs.push(Cfg { h, butterfly: false, seed });
            cfgs.push(Cfg { h, butterfly: true, seed });
        }
    }

    println!("=== Butterfly Scaling: {} configs ===\n", cfgs.len());

    let results: Vec<Res> = cfgs.par_iter().map(|c| run_one(c, &corpus)).collect();

    println!("\n=== SUMMARY ===\n");
    println!("{:<6} {:<10} {:>7} {:>7} {:>10}", "H", "topology", "mean%", "edges", "prop_ns");
    println!("{}", "-".repeat(44));

    for &h in &[256, 512, 1024] {
        for bf in [false, true] {
            let g: Vec<_> = results.iter().filter(|r| r.h == h && r.butterfly == bf).collect();
            if g.is_empty() { continue; }
            let ma = g.iter().map(|r| r.acc).sum::<f64>() / g.len() as f64;
            let me = g.iter().map(|r| r.edges).sum::<usize>() / g.len();
            let mp = g.iter().map(|r| r.prop_ns).sum::<u64>() / g.len() as u64;
            let label = if bf { "butterfly" } else { "random" };
            println!("{:<6} {:<10} {:>6.1}% {:>7} {:>9}ns", h, label, ma * 100.0, me, mp);
        }
    }
}
