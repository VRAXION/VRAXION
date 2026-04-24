//! Mutual Inhibition experiment: create TWO competing clusters in the output zone.
//!
//! The hypothesis: the Brain collapses to a single attractor because there's no
//! "hill" separating different stable states. Mutual inhibition creates this hill:
//!
//!   Cluster A ←(inhibit)→ Cluster B
//!
//!   Input activates A → A suppresses B → stable state: A alive, B dead
//!   Input activates B → B suppresses A → stable state: B alive, A dead
//!
//! This is winner-take-all dynamics — the "domb a völgyek közt" (hill between valleys).

use instnct_core::{
    build_network, cosine_similarity, evolution_step_jackpot, evolution_step_jackpot_traced,
    save_checkpoint, softmax, CandidateTraceRecord, CheckpointMeta, InitConfig, Int8Projection,
    Network, StepOutcome, VcbpTable,
};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use serde::Serialize;
use std::collections::VecDeque;
use std::env;
use std::fs::{create_dir_all, write, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const MAX_CHARGE: i32 = 7;
const DEFAULT_STEPS: usize = 20_000;
const DEFAULT_EVAL_LEN: usize = 100;
const DEFAULT_FULL_LEN: usize = 1_000;
const PROGRESS_INTERVAL: usize = 2_000;

// ── Reuse helpers from evolve_bytepair_proj ──

#[derive(Serialize)]
struct RunMeta {
    fixture: &'static str,
    arm: String,
    run_id: String,
    seed: u64,
    #[serde(rename = "H")]
    h: usize,
    steps: usize,
    jackpot: usize,
    ticks: usize,
    input_scatter: bool,
    corpus: String,
    packed: String,
    checkpoint: String,
    candidate_log: Option<String>,
}

struct CandidateLogWriter {
    writer: BufWriter<File>,
    run_id: String,
    arm: String,
    seed: u64,
    h: usize,
}

impl CandidateLogWriter {
    fn new(path: &Path, run_id: &str, arm: &str, seed: u64, h: usize) -> Self {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                create_dir_all(parent).expect("failed to create candidate log directory");
            }
        }
        let mut writer =
            BufWriter::new(File::create(path).expect("failed to create candidate log"));
        writeln!(
            writer,
            "run_id,arm,seed,H,step,candidate_id,operator_id,mutated,evaluated,before_U,after_U,delta_U,within_cap,selected,accepted,candidate_eval_ms,step_wall_ms"
        )
        .expect("failed to write candidate log header");
        Self {
            writer,
            run_id: run_id.to_string(),
            arm: arm.to_string(),
            seed,
            h,
        }
    }

    fn write_record(&mut self, record: &CandidateTraceRecord) {
        writeln!(
            self.writer,
            "{},{},{},{},{},{},{},{},{},{:.17},{:.17},{:.17},{},{},{},{:.6},{:.6}",
            self.run_id,
            self.arm,
            self.seed,
            self.h,
            record.step,
            record.candidate_id,
            record.operator_id,
            record.mutated,
            record.evaluated,
            record.before_u,
            record.after_u,
            record.delta_u,
            record.within_cap,
            record.selected,
            record.accepted,
            record.candidate_eval_ms,
            record.step_wall_ms
        )
        .expect("failed to write candidate log record");
    }

    fn flush(&mut self) {
        self.writer.flush().expect("failed to flush candidate log");
    }
}

fn quantize_embedding_to_input(
    table: &VcbpTable,
    embedding: &[f32],
    input: &mut [i32],
    input_end: usize,
    input_scatter: bool,
) {
    if !input_scatter {
        table.quantize_to_input(embedding, &mut input[..table.e], MAX_CHARGE);
        return;
    }

    let mut base = vec![0i32; table.e];
    table.quantize_to_input(embedding, &mut base, MAX_CHARGE);
    for dst in input.iter_mut().take(input_end) {
        *dst = 0;
    }
    for idx in 0..input_end.min(input.len()) {
        input[idx] = base[idx % table.e];
    }
}

fn build_corpus_pairs(
    corpus: &[u8],
    table: &VcbpTable,
    max_classes: usize,
) -> (Vec<u16>, Vec<usize>, Vec<u16>, usize) {
    let n_pairs = corpus.len() / 2;
    let mut pair_ids = Vec::with_capacity(n_pairs);
    let mut freq = vec![0u32; 65536];
    for i in 0..n_pairs {
        let pid = VcbpTable::pair_id(corpus[i * 2], corpus[i * 2 + 1]);
        pair_ids.push(pid);
        freq[pid as usize] += 1;
    }
    let mut hot_freq: Vec<(u16, u32)> = (0..65536u32)
        .filter(|&v| table.is_hot(v as u16) && freq[v as usize] > 0)
        .map(|v| (v as u16, freq[v as usize]))
        .collect();
    hot_freq.sort_by(|a, b| b.1.cmp(&a.1));
    let n_classes = hot_freq.len().min(max_classes);
    hot_freq.truncate(n_classes);
    let top_ids: Vec<u16> = hot_freq.iter().map(|&(id, _)| id).collect();
    let mut top_to_idx = vec![usize::MAX; 65536];
    for (i, &id) in top_ids.iter().enumerate() {
        top_to_idx[id as usize] = i;
    }
    (pair_ids, top_to_idx, top_ids, n_classes)
}

fn build_pair_bigram(pair_ids: &[u16], hot_to_idx: &[usize], n_hot: usize) -> Vec<Vec<f64>> {
    let mut counts = vec![vec![0u32; n_hot]; n_hot];
    for i in 0..pair_ids.len().saturating_sub(1) {
        let ci = hot_to_idx[pair_ids[i] as usize];
        let ni = hot_to_idx[pair_ids[i + 1] as usize];
        if ci != usize::MAX && ni != usize::MAX {
            counts[ci][ni] += 1;
        }
    }
    counts
        .iter()
        .map(|row| {
            let total: f64 = row.iter().map(|&c| c as f64).sum();
            if total < 1.0 {
                vec![1.0 / n_hot as f64; n_hot]
            } else {
                row.iter().map(|&c| c as f64 / total).collect()
            }
        })
        .collect()
}

fn eval_accuracy_proj(
    net: &mut Network,
    proj: &Int8Projection,
    table: &VcbpTable,
    pair_ids: &[u16],
    hot_to_idx: &[usize],
    len: usize,
    rng: &mut StdRng,
    propagation: &instnct_core::PropagationConfig,
    output_start: usize,
    neuron_count: usize,
    input_end: usize,
    input_scatter: bool,
) -> f64 {
    let n = pair_ids.len();
    if n <= len + 1 {
        return 0.0;
    }
    let off = rng.gen_range(1..=n - len - 1);
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        let cur_id = pair_ids[off + i];
        let _prev_id = if i > 0 { pair_ids[off + i - 1] } else { cur_id };
        let tgt_id = pair_ids[off + i + 1];
        let tgt_idx = hot_to_idx[tgt_id as usize];
        let emb = table.embed_id(cur_id);
        let mut input = vec![0i32; neuron_count];
        quantize_embedding_to_input(table, emb, &mut input, input_end, input_scatter);
        net.propagate(&input, propagation).unwrap();
        let predicted_idx = proj.predict(&net.charge_vec(output_start..neuron_count));
        if tgt_idx != usize::MAX && predicted_idx == tgt_idx {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

fn eval_smooth_proj(
    net: &mut Network,
    proj: &Int8Projection,
    table: &VcbpTable,
    pair_ids: &[u16],
    hot_to_idx: &[usize],
    bigram: &[Vec<f64>],
    len: usize,
    rng: &mut StdRng,
    propagation: &instnct_core::PropagationConfig,
    output_start: usize,
    neuron_count: usize,
    input_end: usize,
    input_scatter: bool,
) -> f64 {
    let n = pair_ids.len();
    if n <= len + 1 {
        return 0.0;
    }
    let off = rng.gen_range(1..=n - len - 1);
    net.reset();
    let mut total_cos = 0.0f64;
    let mut counted = 0usize;
    for i in 0..len {
        let cur_id = pair_ids[off + i];
        let cur_idx = hot_to_idx[cur_id as usize];
        let emb = table.embed_id(cur_id);
        let mut input = vec![0i32; neuron_count];
        quantize_embedding_to_input(table, emb, &mut input, input_end, input_scatter);
        net.propagate(&input, propagation).unwrap();
        if cur_idx == usize::MAX {
            continue;
        }
        let scores = proj.raw_scores(&net.charge_vec(output_start..neuron_count));
        let probs = softmax(&scores);
        let target = &bigram[cur_idx];
        total_cos += cosine_similarity(&probs, target);
        counted += 1;
    }
    if counted == 0 {
        0.0
    } else {
        total_cos / counted as f64
    }
}

/// Seed mutual inhibition: two clusters in the output zone that suppress each other.
fn seed_mutual_inhibition(net: &mut Network, output_start: usize, h: usize, rng: &mut impl Rng) {
    let mid = output_start + (h - output_start) / 2;

    // Cluster A: output_start..mid
    // Cluster B: mid..h
    // Cross-inhibition: random edges from A→B and B→A, with inhibitory polarity

    let n_cross = 20; // number of inhibitory cross-connections per direction

    for _ in 0..n_cross {
        // A → B (A inhibits B)
        let src = rng.gen_range(output_start..mid) as u16;
        let tgt = rng.gen_range(mid..h) as u16;
        net.graph_mut().add_edge(src, tgt);
        // Make source neuron inhibitory (polarity = -1)
        // But we can't set per-edge polarity, only per-neuron.
        // So we set the SOURCE neuron to inhibitory.
        // This means it inhibits ALL its targets, not just cross-cluster.
        // Compromise: pick dedicated "inhibitor" neurons at cluster boundary.
    }

    for _ in 0..n_cross {
        // B → A (B inhibits A)
        let src = rng.gen_range(mid..h) as u16;
        let tgt = rng.gen_range(output_start..mid) as u16;
        net.graph_mut().add_edge(src, tgt);
    }

    // Set boundary neurons as inhibitory (last 5 of each cluster)
    let a_boundary_start = mid - 5;
    let b_boundary_start = h - 5;
    for n in a_boundary_start..mid {
        if n < h {
            net.polarity_mut()[n] = -1;
            // Lower threshold so they fire easily and suppress the other cluster
            net.spike_data_mut()[n].threshold = 1;
        }
    }
    for n in b_boundary_start..h {
        net.polarity_mut()[n] = -1;
        net.spike_data_mut()[n].threshold = 1;
    }

    println!(
        "  Mutual inhibition: {} cross-edges per direction, boundary neurons inhibitory",
        n_cross
    );
    println!("  Cluster A: neurons {}..{}", output_start, mid);
    println!("  Cluster B: neurons {}..{}", mid, h);
}

// Rooted pathways (same as other examples)
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
        if depth >= max_hops {
            continue;
        }
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
        if depth >= max_hops {
            continue;
        }
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
    let from_input = bfs_forward(net, &(0..input_end).collect::<Vec<_>>(), 4);
    let to_output = bfs_reverse(net, &(output_start..h).collect::<Vec<_>>(), 4);
    let input_anchors: Vec<usize> = (0..h)
        .filter(|&n| from_input[n] && n < output_start)
        .collect();
    let output_anchors: Vec<usize> = (0..h).filter(|&n| to_output[n] && n >= input_end).collect();
    if input_anchors.is_empty() || output_anchors.is_empty() {
        for _ in 0..n_pathways.min(5) {
            net.graph_mut().add_edge(
                rng.gen_range(0..input_end) as u16,
                rng.gen_range(output_start..h) as u16,
            );
        }
        return 0;
    }
    let mut built = 0;
    for _ in 0..n_pathways {
        let ai = input_anchors[rng.gen_range(0..input_anchors.len())];
        let ao = output_anchors[rng.gen_range(0..output_anchors.len())];
        let avail: Vec<usize> = (0..h).filter(|n| *n != ai && *n != ao).collect();
        if avail.len() < 2 {
            continue;
        }
        let nm = rng.gen_range(2..=3.min(avail.len()));
        let mut mids = Vec::new();
        let mut pool = avail;
        for _ in 0..nm {
            let idx = rng.gen_range(0..pool.len());
            mids.push(pool.swap_remove(idx));
        }
        let mut chain = vec![ai];
        chain.extend(&mids);
        chain.push(ao);
        let mut added = false;
        for w in chain.windows(2) {
            if net.graph_mut().add_edge(w[0] as u16, w[1] as u16) {
                added = true;
            }
        }
        if net.graph_mut().add_edge(ao as u16, ai as u16) {
            added = true;
        }
        if added {
            for &n in &chain {
                let sd = &mut net.spike_data_mut()[n];
                if sd.threshold > 1 {
                    sd.threshold -= 1;
                }
            }
            built += 1;
        }
    }
    built
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("Usage: evolve_mutual_inhibition <corpus.txt> <packed.bin> [--steps N]");
        std::process::exit(1);
    }

    let corpus_path = &args[0];
    let packed_path = &args[1];
    let mut steps = DEFAULT_STEPS;
    let mut cli_seed: u64 = 42;
    let mut cli_h: usize = 256;
    let mut jackpot: usize = 9;
    let mut cli_ticks: Option<usize> = None;
    let mut input_scatter = false;
    let mut candidate_log_path: Option<PathBuf> = None;
    let mut checkpoint_at_end: Option<PathBuf> = None;
    let mut arm = String::from("default");
    let mut run_id = String::from("default");
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => {
                i += 1;
                steps = args[i].parse().unwrap();
            }
            "--seed" => {
                i += 1;
                cli_seed = args[i].parse().unwrap();
            }
            "--H" => {
                i += 1;
                cli_h = args[i].parse().unwrap();
            }
            "--jackpot" => {
                i += 1;
                jackpot = args[i].parse().unwrap();
            }
            "--ticks" => {
                i += 1;
                cli_ticks = Some(args[i].parse().unwrap());
            }
            "--input-scatter" => {
                input_scatter = true;
            }
            "--candidate-log" => {
                i += 1;
                candidate_log_path = Some(PathBuf::from(&args[i]));
            }
            "--checkpoint-at-end" => {
                i += 1;
                checkpoint_at_end = Some(PathBuf::from(&args[i]));
            }
            "--arm" => {
                i += 1;
                arm = args[i].clone();
            }
            "--run-id" => {
                i += 1;
                run_id = args[i].clone();
            }
            other => panic!("unknown flag: {other}"),
        }
        i += 1;
    }

    println!("Loading...");
    let table = VcbpTable::from_packed(Path::new(packed_path)).unwrap();
    let corpus = std::fs::read(corpus_path).unwrap();
    let max_classes = 397;
    let (pair_ids, hot_to_idx, _hot_ids, n_classes) =
        build_corpus_pairs(&corpus, &table, max_classes);
    let bigram = build_pair_bigram(&pair_ids, &hot_to_idx, n_classes);

    let h = cli_h;
    let mut init = InitConfig::phi(h);
    if let Some(ticks) = cli_ticks {
        init.propagation.ticks_per_token = ticks;
    }
    let evo_config = init.evolution_config();

    println!("\n=== MUTUAL INHIBITION EXPERIMENT ===");
    println!(
        "  H={}, {} steps, {} classes, seed={}, jackpot={}, ticks={}, input_scatter={}, arm={}, run_id={}\n",
        h,
        steps,
        n_classes,
        cli_seed,
        jackpot,
        init.propagation.ticks_per_token,
        input_scatter,
        arm,
        run_id
    );

    let t_start = Instant::now();
    let seed = cli_seed;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = build_network(&init, &mut rng);

    // Seed rooted pathways
    let n_paths = seed_rooted_pathways(
        &mut net,
        init.input_end(),
        init.output_start(),
        30,
        &mut rng,
    );
    println!("  Rooted pathways: {n_paths}, edges={}", net.edge_count());

    // *** THE KEY: seed mutual inhibition ***
    seed_mutual_inhibition(&mut net, init.output_start(), h, &mut rng);
    println!("  After mutual inhibition: edges={}", net.edge_count());

    let mut proj = Int8Projection::new(
        init.phi_dim,
        n_classes,
        &mut StdRng::seed_from_u64(seed + 200),
    );
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let mut accepted = 0u32;
    let mut rejected = 0u32;
    let mut peak = 0.0f64;
    let mut candidate_log = candidate_log_path
        .as_deref()
        .map(|path| CandidateLogWriter::new(path, &run_id, &arm, seed, h));

    // Evolve with smooth cosine (champion fitness)
    for step in 0..steps {
        let outcome = if let Some(log) = candidate_log.as_mut() {
            evolution_step_jackpot_traced(
                &mut net,
                &mut proj,
                &mut rng,
                &mut eval_rng,
                |n, p, er| {
                    let cos = eval_smooth_proj(
                        n,
                        p,
                        &table,
                        &pair_ids,
                        &hot_to_idx,
                        &bigram,
                        DEFAULT_EVAL_LEN,
                        er,
                        &init.propagation,
                        init.output_start(),
                        h,
                        init.input_end(),
                        input_scatter,
                    );
                    // Anti-monopoly: reward more alive output neurons
                    let charges = n.charge_vec(init.output_start()..h);
                    let alive = charges.iter().filter(|&&c| c > 0).count();
                    let alive_frac = alive as f64 / (h - init.output_start()) as f64;
                    cos * (1.0 + 0.1 * alive_frac) // very gentle diversity pressure (λ=0.1)
                },
                &evo_config,
                jackpot,
                step,
                |record| log.write_record(record),
            )
        } else {
            evolution_step_jackpot(
                &mut net,
                &mut proj,
                &mut rng,
                &mut eval_rng,
                |n, p, er| {
                    let cos = eval_smooth_proj(
                        n,
                        p,
                        &table,
                        &pair_ids,
                        &hot_to_idx,
                        &bigram,
                        DEFAULT_EVAL_LEN,
                        er,
                        &init.propagation,
                        init.output_start(),
                        h,
                        init.input_end(),
                        input_scatter,
                    );
                    // Anti-monopoly: reward more alive output neurons
                    let charges = n.charge_vec(init.output_start()..h);
                    let alive = charges.iter().filter(|&&c| c > 0).count();
                    let alive_frac = alive as f64 / (h - init.output_start()) as f64;
                    cos * (1.0 + 0.1 * alive_frac) // very gentle diversity pressure (λ=0.1)
                },
                &evo_config,
                jackpot,
            )
        };
        match outcome {
            StepOutcome::Accepted => accepted += 1,
            StepOutcome::Rejected => rejected += 1,
            StepOutcome::Skipped => {}
        }
        if (step + 1) % PROGRESS_INTERVAL == 0 {
            let acc = eval_accuracy_proj(
                &mut net,
                &proj,
                &table,
                &pair_ids,
                &hot_to_idx,
                DEFAULT_FULL_LEN,
                &mut eval_rng,
                &init.propagation,
                init.output_start(),
                h,
                init.input_end(),
                input_scatter,
            );
            peak = peak.max(acc);
            let rate = accepted as f64 / (accepted + rejected).max(1) as f64 * 100.0;
            println!(
                "  step {:>5}: acc={:.2}%  peak={:.2}%  accept={:.0}%  edges={}",
                step + 1,
                acc * 100.0,
                peak * 100.0,
                rate,
                net.edge_count()
            );
        }
    }

    let final_acc = eval_accuracy_proj(
        &mut net,
        &proj,
        &table,
        &pair_ids,
        &hot_to_idx,
        DEFAULT_FULL_LEN.min(pair_ids.len() / 2),
        &mut eval_rng,
        &init.propagation,
        init.output_start(),
        h,
        init.input_end(),
        input_scatter,
    );
    peak = peak.max(final_acc);
    let final_accept_rate_pct = accepted as f64 / (accepted + rejected).max(1) as f64 * 100.0;
    println!(
        "\n  FINAL: {:.2}%  peak={:.2}%  edges={}  accept={:.0}%",
        final_acc * 100.0,
        peak * 100.0,
        net.edge_count(),
        final_accept_rate_pct
    );

    if let Some(checkpoint_path) = checkpoint_at_end.as_deref() {
        save_checkpoint(
            checkpoint_path,
            &net,
            &proj,
            CheckpointMeta {
                step: steps,
                accuracy: final_acc,
                label: format!("phase_b fixture=mutual_inhibition arm={arm} run_id={run_id}"),
            },
        )
        .expect("failed to save final checkpoint");

        let meta = RunMeta {
            fixture: "mutual_inhibition",
            arm: arm.clone(),
            run_id: run_id.clone(),
            seed,
            h,
            steps,
            jackpot,
            ticks: init.propagation.ticks_per_token,
            input_scatter,
            corpus: corpus_path.to_string(),
            packed: packed_path.to_string(),
            checkpoint: checkpoint_path.display().to_string(),
            candidate_log: candidate_log_path.as_ref().map(|p| p.display().to_string()),
        };
        let meta_json = serde_json::to_string_pretty(&meta).expect("failed to serialize run meta");
        let meta_path = checkpoint_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("run_meta.json");
        write(&meta_path, meta_json).expect("failed to write run_meta.json");
        println!("  Checkpoint: {}", checkpoint_path.display());
        println!("  Run meta:   {}", meta_path.display());
    }

    // Alive-frac mean over 20 deterministically-spaced corpus pairs (for SUMMARY)
    let alive_samples = 20usize.min(pair_ids.len().max(1));
    let mut alive_frac_sum = 0.0f64;
    let out_zone = h - init.output_start();
    for k in 0..alive_samples {
        let pid = pair_ids[(k * pair_ids.len()) / alive_samples];
        net.reset();
        let emb = table.embed_id(pid);
        let mut input_buf = vec![0i32; h];
        quantize_embedding_to_input(&table, emb, &mut input_buf, init.input_end(), input_scatter);
        net.propagate(&input_buf, &init.propagation).unwrap();
        let charges = net.charge_vec(init.output_start()..h);
        let alive = charges.iter().filter(|&&c| c > 0).count();
        alive_frac_sum += alive as f64 / out_zone.max(1) as f64;
    }
    let alive_frac_mean = alive_frac_sum / alive_samples as f64;

    // Quick adversarial check: do different inputs produce different outputs?
    println!("\n  --- ADVERSARIAL DIVERSITY CHECK ---");
    let test_pairs = [
        VcbpTable::pair_id(b't', b'h'),
        VcbpTable::pair_id(b'e', b' '),
        VcbpTable::pair_id(b' ', b't'),
        VcbpTable::pair_id(b'a', b'l'),
    ];
    let mut charges_list: Vec<Vec<u8>> = Vec::new();
    let mut preds: Vec<usize> = Vec::new();
    for &pid in &test_pairs {
        net.reset();
        let emb = table.embed_id(pid);
        let mut input = vec![0i32; h];
        quantize_embedding_to_input(&table, emb, &mut input, init.input_end(), input_scatter);
        net.propagate(&input, &init.propagation).unwrap();
        let charges = net.charge_vec(init.output_start()..h);
        let alive = charges.iter().filter(|&&c| c > 0).count();
        let pred = proj.predict(&charges);
        preds.push(pred);
        charges_list.push(charges);
        let (hi, lo) = VcbpTable::pair_bytes(pid);
        println!(
            "    '{}{}' → pred={}, alive={}/158",
            hi as char, lo as char, pred, alive
        );
    }
    let unique: std::collections::HashSet<usize> = preds.iter().cloned().collect();
    let mut diffs = 0usize;
    let mut pairs = 0usize;
    for i in 0..charges_list.len() {
        for j in (i + 1)..charges_list.len() {
            diffs += charges_list[i]
                .iter()
                .zip(charges_list[j].iter())
                .filter(|(a, b)| a != b)
                .count();
            pairs += 1;
        }
    }
    println!(
        "    Unique predictions: {}/{}",
        unique.len(),
        test_pairs.len()
    );
    println!(
        "    Avg charge diff: {:.1}/158 ({:.1}%)",
        diffs as f64 / pairs as f64,
        diffs as f64 / pairs as f64 / 158.0 * 100.0
    );
    if unique.len() > 1 {
        println!("    ✓ DIVERSE OUTPUT — mutual inhibition creating multi-attractor!");
    } else {
        println!("    ⚠ Still constant — mutual inhibition not enough alone");
    }

    if let Some(log) = candidate_log.as_mut() {
        log.flush();
    }

    let wall_clock_s = t_start.elapsed().as_secs_f64();
    println!("\nRuntime: {:.1}s", wall_clock_s);

    // Machine-readable summary line for multi-seed drivers.
    println!(
        "SUMMARY {{\"fixture\":\"mutual_inhibition\",\"seed\":{},\"H\":{},\"phi_dim\":{},\"peak_acc\":{:.6},\"final_acc\":{:.6},\"accept_rate_pct\":{:.4},\"alive_frac_mean\":{:.6},\"edges\":{},\"unique_preds\":{},\"wall_clock_s\":{:.3}}}",
        seed, h, init.phi_dim, peak, final_acc, final_accept_rate_pct, alive_frac_mean,
        net.edge_count(), unique.len(), wall_clock_s
    );
}
