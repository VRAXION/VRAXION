//! Butterfly bit-distance knee finder: at how many allowed bit differences
//! does accuracy start to drop vs unrestricted random?
//!
//! Run: cargo run --example butterfly_knee --release -- <corpus-path>

use instnct_core::{load_corpus, InitConfig, Int8Projection, Network, SdrTable};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;

fn can_connect(src: usize, tgt: usize, max_bits: u32) -> bool {
    if src == tgt { return false; }
    let diff = (src ^ tgt).count_ones();
    diff <= max_bits
}

fn constrained_add(net: &mut Network, max_bits: u32, rng: &mut impl Rng) -> bool {
    let h = net.neuron_count();
    for _ in 0..30 {
        let src = rng.gen_range(0..h);
        let tgt = rng.gen_range(0..h);
        if can_connect(src, tgt, max_bits) {
            return net.graph_mut().add_edge(src as u16, tgt as u16);
        }
    }
    false
}

fn constrained_rewire(net: &mut Network, max_bits: u32, rng: &mut impl Rng) -> bool {
    let h = net.neuron_count();
    let edges: Vec<_> = net.graph().iter_edges().collect();
    if edges.is_empty() || h < 2 { return false; }
    let edge = edges[rng.gen_range(0..edges.len())];
    for _ in 0..30 {
        let new_tgt = rng.gen_range(0..h);
        if can_connect(edge.source as usize, new_tgt, max_bits) {
            net.graph_mut().remove_edge(edge.source, edge.target);
            if net.graph_mut().add_edge(edge.source, new_tgt as u16) {
                return true;
            }
            net.graph_mut().add_edge(edge.source, edge.target);
            return false;
        }
    }
    false
}


fn sample_eval_offset(corpus_len: usize, len: usize, rng: &mut StdRng) -> Option<usize> {
    if corpus_len <= len { return None; }
    Some(rng.gen_range(0..=corpus_len - len - 1))
}

#[allow(clippy::too_many_arguments)]
fn eval_accuracy(
    net: &mut Network,
    projection: &Int8Projection,
    corpus: &[u8],
    len: usize,
    rng: &mut StdRng,
    sdr: &SdrTable,
    output_start: usize,
    neuron_count: usize,
    prop_config: &instnct_core::PropagationConfig,
) -> f64 {
    let Some(off) = sample_eval_offset(corpus.len(), len, rng) else { return 0.0; };
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), prop_config).unwrap();
        if projection.predict(&net.charge_vec(output_start..neuron_count)) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

fn run_one(max_bits: u32, seed: u64, corpus: &[u8]) -> (f64, usize) {
    let init = InitConfig::phi(256);
    let h = init.neuron_count;
    let output_start = init.output_start();
    let edge_cap = init.edge_cap();
    let is_random = max_bits >= 8;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = Network::new(h);

    // Chain-50 init (unconstrained for fair reachability)
    let os = init.output_start();
    let oe = init.input_end();
    let om = (os + oe) / 2;
    for _ in 0..50 {
        let src = rng.gen_range(0..os) as u16;
        let h1 = rng.gen_range(os..om) as u16;
        let h2 = rng.gen_range(om..oe) as u16;
        let tgt = rng.gen_range(oe..h) as u16;
        net.graph_mut().add_edge(src, h1);
        net.graph_mut().add_edge(h1, h2);
        net.graph_mut().add_edge(h2, tgt);
    }

    // Fill to 5% density
    let target_edges = h * h * 5 / 100;
    for _ in 0..target_edges * 5 {
        if is_random {
            net.mutate_add_edge(&mut rng);
        } else {
            constrained_add(&mut net, max_bits, &mut rng);
        }
        if net.edge_count() >= target_edges { break; }
    }

    // Random params
    for i in 0..h {
        net.spike_data_mut()[i].threshold = rng.gen_range(0..=7u8);
        net.spike_data_mut()[i].channel = rng.gen_range(1..=8u8);
        if rng.gen_ratio(1, 10) { net.polarity_mut()[i] = -1; }
    }

    let mut proj = Int8Projection::new(init.phi_dim, CHARS, &mut StdRng::seed_from_u64(seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let sdr = SdrTable::new(CHARS, h, init.input_end(), SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(seed + 100)).unwrap();

    let steps = 15000;
    for _ in 0..steps {
        let eval_snapshot = eval_rng.clone();
        let before = eval_accuracy(
            &mut net, &proj, corpus, 100, &mut eval_rng,
            &sdr, output_start, h, &init.propagation,
        );
        eval_rng = eval_snapshot;

        let snapshot = net.save_state();
        let mut wb = None;

        let roll = rng.gen_range(0..100u32);
        let mutated = match roll {
            0..25 => if is_random { net.mutate_add_edge(&mut rng) } else { constrained_add(&mut net, max_bits, &mut rng) },
            25..40 => net.mutate_remove_edge(&mut rng),
            40..50 => if is_random { net.mutate_rewire(&mut rng) } else { constrained_rewire(&mut net, max_bits, &mut rng) },
            50..65 => net.mutate_reverse(&mut rng),
            65..72 => net.mutate_mirror(&mut rng),
            72..80 => net.mutate_enhance(&mut rng),
            80..85 => net.mutate_theta(&mut rng),
            85..90 => net.mutate_channel(&mut rng),
            _ => { wb = Some(proj.mutate_one(&mut rng)); true }
        };

        if !mutated {
            let _ = eval_accuracy(&mut net, &proj, corpus, 100, &mut eval_rng, &sdr, output_start, h, &init.propagation);
            continue;
        }

        let after = eval_accuracy(&mut net, &proj, corpus, 100, &mut eval_rng, &sdr, output_start, h, &init.propagation);
        let accepted = if net.edge_count() < edge_cap { after >= before } else { after > before };
        if !accepted {
            net.restore_state(&snapshot);
            if let Some(backup) = wb { proj.rollback(backup); }
        }
    }

    let acc = eval_accuracy(&mut net, &proj, corpus, 5000, &mut eval_rng, &sdr, output_start, h, &init.propagation);
    (acc, net.edge_count())
}

fn main() {
    let seeds = [42u64, 123, 7];

    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars\n", corpus.len());

    let bit_limits: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 8]; // 8 = random

    // Show allowed edge counts
    let h = 256usize;
    for &bits in &bit_limits {
        let mut allowed = 0u64;
        for src in 0..h {
            for tgt in 0..h {
                if can_connect(src, tgt, bits) { allowed += 1; }
            }
        }
        let label = if bits >= 8 { "random".to_string() } else { format!("≤{}bit", bits) };
        println!("  {:<8} allowed: {:>6} ({:.1}%)", label, allowed, allowed as f64 / (h * h) as f64 * 100.0);
    }
    println!();

    // Build configs: (max_bits, seed)
    let configs: Vec<(u32, u64)> = bit_limits.iter()
        .flat_map(|&bits| seeds.iter().map(move |&seed| (bits, seed)))
        .collect();

    println!("=== Butterfly Knee: {} configs, 15K steps ===\n", configs.len());

    let results: Vec<(u32, u64, f64, usize)> = configs
        .par_iter()
        .map(|&(bits, seed)| {
            let (acc, edges) = run_one(bits, seed, &corpus);
            let label = if bits >= 8 { "random".to_string() } else { format!("≤{}bit", bits) };
            println!("  {:<8} seed={:<5} -> {:.1}%  edges={}", label, seed, acc * 100.0, edges);
            (bits, seed, acc, edges)
        })
        .collect();

    println!("\n=== SUMMARY ===\n");
    println!("{:<8} {:>8} {:>8} {:>8}", "limit", "allowed%", "mean%", "edges");
    println!("{}", "-".repeat(36));

    for &bits in &bit_limits {
        let group: Vec<_> = results.iter().filter(|r| r.0 == bits).collect();
        let mean_acc = group.iter().map(|r| r.2).sum::<f64>() / group.len() as f64;
        let mean_edges = group.iter().map(|r| r.3).sum::<usize>() / group.len();

        let mut allowed = 0u64;
        for src in 0..h { for tgt in 0..h { if can_connect(src, tgt, bits) { allowed += 1; } } }
        let pct = allowed as f64 / (h * h) as f64 * 100.0;

        let label = if bits >= 8 { "random".to_string() } else { format!("≤{}bit", bits) };
        println!("{:<8} {:>7.1}% {:>7.1}% {:>8}", label, pct, mean_acc * 100.0, mean_edges);
    }
}
