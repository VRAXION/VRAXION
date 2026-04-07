//! Exhaustive burst mutation sweep: test every mutation operator at multiple
//! burst sizes (1-8 mutations per eval step).
//!
//! Configs: 8 operators x 5 burst sizes x 3 seeds = 120 independent runs.
//! All configs run in parallel via rayon.
//!
//! Run: cargo run --example burst_sweep --release -- <corpus-path>

use instnct_core::{load_corpus, build_network, InitConfig, Int8Projection, Network, PropagationConfig, SdrTable};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const CHARS: usize = 27; // a-z (0..25) + space (26)
const SDR_ACTIVE_PCT: usize = 20;
const STEPS: usize = 30_000;
const EVAL_LEN: usize = 100;

const BURST_SIZES: &[usize] = &[1, 2, 3, 5, 8];
const SEEDS: &[u64] = &[42, 123, 7];

// ---- Operator enum ----

#[derive(Clone, Copy, Debug)]
enum Operator {
    AddEdge,
    RemoveEdge,
    Rewire,
    Reverse,
    Theta,
    Channel,
    Projection,
    Mixed,
}

impl Operator {
    fn name(&self) -> &str {
        match self {
            Operator::AddEdge => "add_edge",
            Operator::RemoveEdge => "remove_edge",
            Operator::Rewire => "rewire",
            Operator::Reverse => "reverse",
            Operator::Theta => "theta",
            Operator::Channel => "channel",
            Operator::Projection => "projection",
            Operator::Mixed => "mixed",
        }
    }

    fn all() -> &'static [Operator] {
        &[
            Operator::AddEdge,
            Operator::RemoveEdge,
            Operator::Rewire,
            Operator::Reverse,
            Operator::Theta,
            Operator::Channel,
            Operator::Projection,
            Operator::Mixed,
        ]
    }
}

// ---- Run result ----

#[allow(dead_code)]
struct RunResult {
    operator: Operator,
    burst: usize,
    seed: u64,
    final_accuracy: f64,
    peak_accuracy: f64,
    edge_count: usize,
    accept_rate: f64,
}

// ---- Corpus loader ----


// ---- Eval ----

fn sample_eval_offset(corpus_len: usize, len: usize, rng: &mut StdRng) -> Option<usize> {
    if corpus_len <= len {
        return None;
    }
    let max_offset = corpus_len - len - 1;
    Some(rng.gen_range(0..=max_offset))
}

#[allow(clippy::too_many_arguments)]
fn eval_accuracy(
    net: &mut Network,
    projection: &Int8Projection,
    corpus: &[u8],
    len: usize,
    rng: &mut StdRng,
    sdr: &SdrTable,
    config: &PropagationConfig,
    output_start: usize,
    neuron_count: usize,
) -> f64 {
    let Some(off) = sample_eval_offset(corpus.len(), len, rng) else {
        return 0.0;
    };

    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), config).unwrap();
        if projection.predict(&net.charge_vec(output_start..neuron_count)) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

// ---- Mixed mutation (library schedule replica) ----

fn apply_mixed_mutation(
    net: &mut Network,
    proj: &mut Int8Projection,
    rng: &mut impl Rng,
) -> bool {
    let roll = rng.gen_range(0..100u32);
    match roll {
        0..25 => net.mutate_add_edge(rng),
        25..40 => net.mutate_remove_edge(rng),
        40..50 => net.mutate_rewire(rng),
        50..65 => net.mutate_reverse(rng),
        65..72 => net.mutate_mirror(rng),
        72..80 => net.mutate_enhance(rng),
        80..85 => net.mutate_theta(rng),
        85..90 => net.mutate_channel(rng),
        _ => {
            let _ = proj.mutate_one(rng);
            true
        }
    }
}

// ---- Single config run ----

#[allow(clippy::too_many_arguments)]
fn run_config(
    operator: Operator,
    burst: usize,
    seed: u64,
    corpus: &[u8],
    init: &InitConfig,
) -> RunResult {
    let output_start = init.output_start();
    let neuron_count = init.neuron_count;
    let edge_cap = init.edge_cap();

    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = build_network(init, &mut rng);
    let mut proj = Int8Projection::new(
        init.phi_dim,
        CHARS,
        &mut StdRng::seed_from_u64(seed + 200),
    );
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let sdr = SdrTable::new(
        CHARS,
        neuron_count,
        init.input_end(),
        SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(seed + 100),
    )
    .unwrap();

    let mut accepted = 0u32;
    let mut total_tried = 0u32;
    let mut peak_accuracy = 0.0f64;

    for step in 0..STEPS {
        // Paired eval: snapshot eval_rng, compute before
        let snap = eval_rng.clone();
        let before = eval_accuracy(
            &mut net,
            &proj,
            corpus,
            EVAL_LEN,
            &mut eval_rng,
            &sdr,
            &init.propagation,
            output_start,
            neuron_count,
        );
        eval_rng = snap;

        // Snapshot state for rollback
        let net_state = net.save_state();
        let proj_backup = proj.clone();
        let edges_before = net.edge_count();

        // Apply BURST mutations
        let mut applied = 0u32;
        for _ in 0..burst {
            let success = match operator {
                Operator::AddEdge => net.mutate_add_edge(&mut rng),
                Operator::RemoveEdge => net.mutate_remove_edge(&mut rng),
                Operator::Rewire => net.mutate_rewire(&mut rng),
                Operator::Reverse => net.mutate_reverse(&mut rng),
                Operator::Theta => net.mutate_theta(&mut rng),
                Operator::Channel => net.mutate_channel(&mut rng),
                Operator::Projection => {
                    let _ = proj.mutate_one(&mut rng);
                    true
                }
                Operator::Mixed => apply_mixed_mutation(&mut net, &mut proj, &mut rng),
            };
            if success {
                applied += 1;
            }
        }

        if applied == 0 {
            // Advance eval_rng to stay in sync
            let _ = eval_accuracy(
                &mut net,
                &proj,
                corpus,
                EVAL_LEN,
                &mut eval_rng,
                &sdr,
                &init.propagation,
                output_start,
                neuron_count,
            );
            continue;
        }

        // Edge cap check
        let edge_grew = net.edge_count() > edges_before;
        let within_cap = !edge_grew || net.edge_count() <= edge_cap;

        let after = eval_accuracy(
            &mut net,
            &proj,
            corpus,
            EVAL_LEN,
            &mut eval_rng,
            &sdr,
            &init.propagation,
            output_start,
            neuron_count,
        );

        if after > before && within_cap {
            accepted += 1;
            if after > peak_accuracy {
                peak_accuracy = after;
            }
        } else {
            net.restore_state(&net_state);
            proj = proj_backup;
        }
        total_tried += 1;

        // Track peak from before scores too
        if before > peak_accuracy {
            peak_accuracy = before;
        }

        // Log every 10K
        if (step + 1) % 10_000 == 0 {
            let rate = if total_tried > 0 {
                accepted as f64 / total_tried as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "  {} burst={} seed={}  step {}: {:.1}%  edges={}  accept={:.1}%",
                operator.name(),
                burst,
                seed,
                step + 1,
                after * 100.0,
                net.edge_count(),
                rate,
            );
        }
    }

    let rate = if total_tried > 0 {
        accepted as f64 / total_tried as f64 * 100.0
    } else {
        0.0
    };

    // Final eval uses the last "after" accuracy from the loop
    // Compute a fresh final accuracy for reporting
    let final_accuracy = eval_accuracy(
        &mut net,
        &proj,
        corpus,
        EVAL_LEN,
        &mut eval_rng,
        &sdr,
        &init.propagation,
        output_start,
        neuron_count,
    );

    if final_accuracy > peak_accuracy {
        peak_accuracy = final_accuracy;
    }

    println!(
        "  {} burst={} seed={}  FINAL: {:.1}%  peak={:.1}%  edges={}",
        operator.name(),
        burst,
        seed,
        final_accuracy * 100.0,
        peak_accuracy * 100.0,
        net.edge_count(),
    );

    RunResult {
        operator,
        burst,
        seed,
        final_accuracy,
        peak_accuracy,
        edge_count: net.edge_count(),
        accept_rate: rate,
    }
}

// ---- Main ----

fn main() {
    let init = InitConfig::phi(256);

    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars", corpus.len());

    // Build config matrix
    let configs: Vec<(Operator, usize, u64)> = Operator::all()
        .iter()
        .flat_map(|&op| {
            BURST_SIZES
                .iter()
                .flat_map(move |&burst| SEEDS.iter().map(move |&seed| (op, burst, seed)))
        })
        .collect();

    println!(
        "\n=== BURST SWEEP: {} operators x {} bursts x {} seeds = {} configs ===",
        Operator::all().len(),
        BURST_SIZES.len(),
        SEEDS.len(),
        configs.len(),
    );
    println!(
        "H={}, {} steps, edge_cap={}\n",
        init.neuron_count,
        STEPS,
        init.edge_cap(),
    );

    // Run all configs in parallel
    let results: Vec<RunResult> = configs
        .par_iter()
        .map(|&(op, burst, seed)| run_config(op, burst, seed, &corpus, &init))
        .collect();

    // ---- Summary: group by (operator, burst) ----
    println!("\n=== SUMMARY ===\n");
    println!(
        "{:<14} {:>5}   {:>5}   {:>5}   {:>5}   {:>5}  {:>7}",
        "Operator", "Burst", "Mean%", "Best%", "Peak%", "Edges", "Accept%"
    );
    println!("{}", "-".repeat(62));

    struct GroupSummary {
        operator: Operator,
        burst: usize,
        mean_acc: f64,
        best_acc: f64,
        peak_acc: f64,
        mean_edges: usize,
        mean_accept: f64,
    }

    let mut summaries: Vec<GroupSummary> = Vec::new();

    for &op in Operator::all() {
        for &burst in BURST_SIZES {
            let group: Vec<&RunResult> = results
                .iter()
                .filter(|r| r.operator.name() == op.name() && r.burst == burst)
                .collect();

            if group.is_empty() {
                continue;
            }

            let n = group.len() as f64;
            let mean_acc = group.iter().map(|r| r.final_accuracy).sum::<f64>() / n;
            let best_acc = group
                .iter()
                .map(|r| r.final_accuracy)
                .fold(0.0f64, f64::max);
            let peak_acc = group
                .iter()
                .map(|r| r.peak_accuracy)
                .fold(0.0f64, f64::max);
            let mean_edges =
                (group.iter().map(|r| r.edge_count).sum::<usize>() as f64 / n) as usize;
            let mean_accept = group.iter().map(|r| r.accept_rate).sum::<f64>() / n;

            println!(
                "{:<14} {:>5}   {:>4.1}%   {:>4.1}%   {:>4.1}%   {:>5}  {:>6.1}%",
                op.name(),
                burst,
                mean_acc * 100.0,
                best_acc * 100.0,
                peak_acc * 100.0,
                mean_edges,
                mean_accept,
            );

            summaries.push(GroupSummary {
                operator: op,
                burst,
                mean_acc,
                best_acc,
                peak_acc,
                mean_edges,
                mean_accept,
            });
        }
    }

    // ---- Best configs (top 10 by mean) ----
    summaries.sort_by(|a, b| b.mean_acc.partial_cmp(&a.mean_acc).unwrap());

    println!("\n=== BEST CONFIGS (top 10 by mean) ===");
    for (i, s) in summaries.iter().take(10).enumerate() {
        println!(
            "  {:>2}. {} burst={}:  mean={:.1}%  best={:.1}%  peak={:.1}%  edges={}  accept={:.1}%",
            i + 1,
            s.operator.name(),
            s.burst,
            s.mean_acc * 100.0,
            s.best_acc * 100.0,
            s.peak_acc * 100.0,
            s.mean_edges,
            s.mean_accept,
        );
    }
}
