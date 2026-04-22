//! ABC input → Brain → char prediction (27 classes).
//!
//! Tests whether C embeddings as input help the Brain at the SAME task
//! it already solves (24.6% with SdrTable byte-level input).
//!
//! Pipeline:
//!   byte → C embed (via byte-pair with next byte) → quantize [0, max_charge]
//!   → inject into Brain input neurons 0..31
//!   → spiking propagation
//!   → Int8Projection(phi_dim, 27) → predicted char class
//!
//! This keeps the PROVEN output mechanism (Int8Projection + smooth cosine bigram)
//! and only changes the INPUT encoding. Fair A/B test:
//!   A) SdrTable byte input (existing 24.6% baseline)
//!   B) C embedding input (this experiment)

use instnct_core::{
    build_bigram_table, build_network, eval_accuracy, eval_smooth, evolution_step_jackpot,
    load_corpus, InitConfig, Int8Projection, Network, SdrTable, StepOutcome, VcbpTable,
};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::collections::VecDeque;
use std::env;
use std::path::Path;
use std::time::Instant;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const DEFAULT_STEPS: usize = 10_000;
const DEFAULT_SEEDS: usize = 3;
const DEFAULT_EVAL_LEN: usize = 100;
const DEFAULT_FULL_LEN: usize = 2_000;
const PROGRESS_INTERVAL: usize = 2_000;
const MAX_CHARGE: i32 = 7;

/// Map a corpus byte (0-26) to the byte-pair formed by that char's ASCII + space.
/// E.g., char 0 = 'a' → byte-pair ('a', ' ') = 0x6120, lookup C embedding.
/// This gives the Brain a richer input than one-hot SDR.
fn char_to_c_input(
    ch: u8,
    table: &VcbpTable,
    neuron_count: usize,
) -> Vec<i32> {
    let ascii = if ch < 26 { b'a' + ch } else { b' ' };
    // Form a byte-pair with space as context (most common bigram partner)
    let pair_id = VcbpTable::pair_id(ascii, b' ');
    let emb = table.embed_id(pair_id);
    let mut input = vec![0i32; neuron_count];
    let e = table.e.min(neuron_count);
    table.quantize_to_input(emb, &mut input[..e], MAX_CHARGE);
    input
}

/// Custom eval_accuracy that uses C embeddings instead of SdrTable.
fn eval_accuracy_c(
    net: &mut Network,
    proj: &Int8Projection,
    table: &VcbpTable,
    corpus: &[u8],
    len: usize,
    rng: &mut StdRng,
    propagation: &instnct_core::PropagationConfig,
    output_start: usize,
    neuron_count: usize,
) -> f64 {
    if corpus.len() <= len {
        return 0.0;
    }
    let off = rng.gen_range(0..=corpus.len() - len - 1);
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        let input = char_to_c_input(seg[i], table, neuron_count);
        net.propagate(&input, propagation).unwrap();
        if proj.predict(&net.charge_vec(output_start..neuron_count)) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

/// Custom eval_smooth that uses C embeddings instead of SdrTable.
fn eval_smooth_c(
    net: &mut Network,
    proj: &Int8Projection,
    table: &VcbpTable,
    corpus: &[u8],
    len: usize,
    rng: &mut StdRng,
    propagation: &instnct_core::PropagationConfig,
    output_start: usize,
    neuron_count: usize,
    bigram: &[Vec<f64>],
) -> f64 {
    if corpus.len() <= len {
        return 0.0;
    }
    let off = rng.gen_range(0..=corpus.len() - len - 1);
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut total_cos = 0.0f64;
    for i in 0..len {
        let input = char_to_c_input(seg[i], table, neuron_count);
        net.propagate(&input, propagation).unwrap();
        let scores = proj.raw_scores(&net.charge_vec(output_start..neuron_count));
        let probs = instnct_core::softmax(&scores);
        let target = &bigram[seg[i] as usize];
        total_cos += instnct_core::cosine_similarity(&probs, target);
    }
    total_cos / len as f64
}

// ── Rooted pathway (same as evolve_bytepair.rs) ──

fn bfs_forward(net: &Network, starts: &[usize], max_hops: usize) -> Vec<bool> {
    let h = net.neuron_count();
    let mut reached = vec![false; h];
    let mut queue = VecDeque::new();
    for &s in starts {
        reached[s] = true;
        queue.push_back((s, 0usize));
    }
    let mut adj: Vec<Vec<u16>> = vec![Vec::new(); h];
    for edge in net.graph().iter_edges() {
        adj[edge.source as usize].push(edge.target);
    }
    while let Some((node, depth)) = queue.pop_front() {
        if depth >= max_hops { continue; }
        for &tgt in &adj[node] {
            if !reached[tgt as usize] {
                reached[tgt as usize] = true;
                queue.push_back((tgt as usize, depth + 1));
            }
        }
    }
    reached
}

fn bfs_reverse(net: &Network, ends: &[usize], max_hops: usize) -> Vec<bool> {
    let h = net.neuron_count();
    let mut reached = vec![false; h];
    let mut queue = VecDeque::new();
    for &e in ends {
        reached[e] = true;
        queue.push_back((e, 0usize));
    }
    let mut rev_adj: Vec<Vec<u16>> = vec![Vec::new(); h];
    for edge in net.graph().iter_edges() {
        rev_adj[edge.target as usize].push(edge.source);
    }
    while let Some((node, depth)) = queue.pop_front() {
        if depth >= max_hops { continue; }
        for &src in &rev_adj[node] {
            if !reached[src as usize] {
                reached[src as usize] = true;
                queue.push_back((src as usize, depth + 1));
            }
        }
    }
    reached
}

fn seed_rooted_pathways(
    net: &mut Network,
    input_end: usize,
    output_start: usize,
    n_pathways: usize,
    rng: &mut impl Rng,
) -> usize {
    let h = net.neuron_count();
    let input_zone: Vec<usize> = (0..input_end).collect();
    let output_zone: Vec<usize> = (output_start..h).collect();
    let from_input = bfs_forward(net, &input_zone, 4);
    let to_output = bfs_reverse(net, &output_zone, 4);
    let input_anchors: Vec<usize> = (0..h).filter(|&n| from_input[n] && n < output_start).collect();
    let output_anchors: Vec<usize> = (0..h).filter(|&n| to_output[n] && n >= input_end).collect();
    if input_anchors.is_empty() || output_anchors.is_empty() {
        for _ in 0..n_pathways.min(5) {
            let src = rng.gen_range(0..input_end) as u16;
            let tgt = rng.gen_range(output_start..h) as u16;
            net.graph_mut().add_edge(src, tgt);
        }
        return 0;
    }
    let mut built = 0;
    for _ in 0..n_pathways {
        let anchor_in = input_anchors[rng.gen_range(0..input_anchors.len())];
        let anchor_out = output_anchors[rng.gen_range(0..output_anchors.len())];
        let available: Vec<usize> = (0..h).filter(|n| *n != anchor_in && *n != anchor_out).collect();
        if available.len() < 2 { continue; }
        let n_mid = rng.gen_range(2..=3.min(available.len()));
        let mut mids = Vec::new();
        let mut pool = available;
        for _ in 0..n_mid {
            let idx = rng.gen_range(0..pool.len());
            mids.push(pool.swap_remove(idx));
        }
        let mut chain = vec![anchor_in];
        chain.extend(&mids);
        chain.push(anchor_out);
        let mut added = false;
        for w in chain.windows(2) {
            if net.graph_mut().add_edge(w[0] as u16, w[1] as u16) { added = true; }
        }
        if net.graph_mut().add_edge(anchor_out as u16, anchor_in as u16) { added = true; }
        if added {
            for &n in &chain {
                let sd = &mut net.spike_data_mut()[n];
                if sd.threshold > 1 { sd.threshold -= 1; }
            }
            built += 1;
        }
    }
    built
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("Usage: evolve_abc_char <corpus.txt> <packed.bin> [--steps N] [--seeds N] [--mode sdr|abc]");
        std::process::exit(1);
    }

    let corpus_path = &args[0];
    let packed_path = &args[1];
    let mut steps = DEFAULT_STEPS;
    let mut seed_count = DEFAULT_SEEDS;
    let mut mode = "both".to_string(); // run both SdrTable and C-input

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => { i += 1; steps = args[i].parse().unwrap(); }
            "--seeds" => { i += 1; seed_count = args[i].parse().unwrap(); }
            "--mode" => { i += 1; mode = args[i].clone(); }
            other => panic!("unknown flag: {other}"),
        }
        i += 1;
    }

    // Load C embedding table
    println!("Loading C embedding table...");
    let table = VcbpTable::from_packed(Path::new(packed_path)).unwrap();
    println!("  {table}");

    // Load corpus (mapped to 0-26)
    println!("Loading corpus...");
    let corpus = load_corpus(corpus_path).expect("failed to load corpus");
    println!("  {} chars", corpus.len());

    // Bigram table for smooth fitness
    let bigram = build_bigram_table(&corpus, CHARS);

    // Baselines
    let random_baseline = 1.0 / CHARS as f64;
    let mut freq = vec![0u32; CHARS];
    for &c in &corpus { freq[c as usize] += 1; }
    let most = freq.iter().enumerate().max_by_key(|(_, &c)| c).unwrap();
    let freq_baseline = *most.1 as f64 / corpus.len() as f64;
    println!("  Random: {:.1}%  Freq: {:.1}%  ", random_baseline * 100.0, freq_baseline * 100.0);

    // Brain config: H=256 (same as proven char-level)
    let h = 256usize;
    let init = InitConfig::phi(h);
    let evo_config = init.evolution_config();

    println!("\n=== A/B Test: SdrTable vs C-embedding input, same Brain ===");
    println!("  H={h}, phi_dim={}, {steps} steps, {seed_count} seeds\n", init.phi_dim);

    let t_start = Instant::now();

    // ── Run A: Standard SdrTable (baseline) ──
    if mode == "sdr" || mode == "both" {
        println!("--- [A] SdrTable input (baseline) ---");
        for seed_idx in 0..seed_count {
            let seed = 42 + seed_idx as u64 * 1000;
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = build_network(&init, &mut rng);
            let mut proj = Int8Projection::new(init.phi_dim, CHARS, &mut StdRng::seed_from_u64(seed + 200));
            let sdr = SdrTable::new(CHARS, h, init.input_end(), SDR_ACTIVE_PCT, &mut StdRng::seed_from_u64(seed + 100)).unwrap();
            let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
            let mut accepted = 0u32;
            let mut rejected = 0u32;
            let mut peak = 0.0f64;

            for step in 0..steps {
                let outcome = evolution_step_jackpot(
                    &mut net, &mut proj, &mut rng, &mut eval_rng,
                    |n, p, er| eval_smooth(n, p, &corpus, DEFAULT_EVAL_LEN, er, &sdr, &init.propagation, init.output_start(), h, &bigram),
                    &evo_config, 9,
                );
                match outcome {
                    StepOutcome::Accepted => accepted += 1,
                    StepOutcome::Rejected => rejected += 1,
                    StepOutcome::Skipped => {}
                }
                if (step + 1) % PROGRESS_INTERVAL == 0 {
                    let acc = eval_accuracy(&mut net, &proj, &corpus, DEFAULT_FULL_LEN, &mut eval_rng, &sdr, &init.propagation, init.output_start(), h);
                    peak = peak.max(acc);
                    let rate = accepted as f64 / (accepted + rejected).max(1) as f64 * 100.0;
                    println!("  [A][seed={seed}] step {:>5}: acc={:.1}%  peak={:.1}%  accept={:.0}%  edges={}", step+1, acc*100.0, peak*100.0, rate, net.edge_count());
                }
            }
            let final_acc = eval_accuracy(&mut net, &proj, &corpus, DEFAULT_FULL_LEN, &mut eval_rng, &sdr, &init.propagation, init.output_start(), h);
            peak = peak.max(final_acc);
            println!("  [A][seed={seed}] FINAL: {:.1}%  peak={:.1}%  edges={}  accept={:.0}%",
                final_acc*100.0, peak*100.0, net.edge_count(),
                accepted as f64 / (accepted+rejected).max(1) as f64 * 100.0);
        }
    }

    // ── Run B: C-embedding input ──
    if mode == "abc" || mode == "both" {
        println!("\n--- [B] C-embedding input (ABC pipeline) ---");
        for seed_idx in 0..seed_count {
            let seed = 42 + seed_idx as u64 * 1000;
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = build_network(&init, &mut rng);

            // Add rooted pathways for better signal flow
            let n_paths = seed_rooted_pathways(&mut net, init.input_end(), init.output_start(), 30, &mut rng);
            println!("  [B][seed={seed}] Rooted pathways: {n_paths}, edges={}", net.edge_count());

            let mut proj = Int8Projection::new(init.phi_dim, CHARS, &mut StdRng::seed_from_u64(seed + 200));
            let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
            let mut accepted = 0u32;
            let mut rejected = 0u32;
            let mut peak = 0.0f64;

            for step in 0..steps {
                let outcome = evolution_step_jackpot(
                    &mut net, &mut proj, &mut rng, &mut eval_rng,
                    |n, p, er| eval_smooth_c(n, p, &table, &corpus, DEFAULT_EVAL_LEN, er, &init.propagation, init.output_start(), h, &bigram),
                    &evo_config, 9,
                );
                match outcome {
                    StepOutcome::Accepted => accepted += 1,
                    StepOutcome::Rejected => rejected += 1,
                    StepOutcome::Skipped => {}
                }
                if (step + 1) % PROGRESS_INTERVAL == 0 {
                    let acc = eval_accuracy_c(&mut net, &proj, &table, &corpus, DEFAULT_FULL_LEN, &mut eval_rng, &init.propagation, init.output_start(), h);
                    peak = peak.max(acc);
                    let rate = accepted as f64 / (accepted + rejected).max(1) as f64 * 100.0;
                    println!("  [B][seed={seed}] step {:>5}: acc={:.1}%  peak={:.1}%  accept={:.0}%  edges={}", step+1, acc*100.0, peak*100.0, rate, net.edge_count());
                }
            }
            let final_acc = eval_accuracy_c(&mut net, &proj, &table, &corpus, DEFAULT_FULL_LEN, &mut eval_rng, &init.propagation, init.output_start(), h);
            peak = peak.max(final_acc);
            println!("  [B][seed={seed}] FINAL: {:.1}%  peak={:.1}%  edges={}  accept={:.0}%",
                final_acc*100.0, peak*100.0, net.edge_count(),
                accepted as f64 / (accepted+rejected).max(1) as f64 * 100.0);
        }
    }

    println!("\nRuntime: {:.1}s", t_start.elapsed().as_secs_f64());
}
