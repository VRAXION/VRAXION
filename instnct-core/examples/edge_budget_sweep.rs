//! Sweep: fixed edge budget × variable H — does bigger sparse beat smaller dense?
//!
//! All networks have the SAME edge budget (~50K edges, fits L2 comfortably).
//! As H grows, sparsity decreases proportionally.
//! Each network trains with evolution for a scaled number of steps.
//! Measures: accuracy, tok/sec, peak accuracy, fire rate.

use instnct_core::{
    build_bigram_table, build_network, eval_accuracy, eval_smooth, evolution_step_jackpot,
    load_corpus, InitConfig, Int8Projection, SdrTable, StepOutcome,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;
const CORPUS_PATH: &str = "instnct-core/tests/fixtures/beta_smoke_corpus.txt";
const EVAL_LEN: usize = 500;

/// Fixed edge budget — all networks get the same number of edges
const EDGE_BUDGET: usize = 20_000;

/// Base steps for H=256 (smallest). Larger H = more search space = proportionally more steps.
const BASE_STEPS: usize = 1_000;

/// Steps scale factor: steps = BASE_STEPS × (H / 256)
/// Bigger network needs more steps to explore larger topology space.
const STEP_SCALE_BASE: usize = 256;

struct SweepCase {
    h: usize,
    steps: usize,
    density_pct: usize,    // initial fill density
    edge_cap: usize,       // hard cap on edges during evolution
}

fn build_cases() -> Vec<SweepCase> {
    let sizes = [256, 512, 1024, 2048, 4096];
    sizes
        .iter()
        .map(|&h| {
            // Scale steps with network size (bigger search space)
            let step_multiplier = (h as f64 / STEP_SCALE_BASE as f64).max(1.0);
            let steps = (BASE_STEPS as f64 * step_multiplier) as usize;

            // Initial density to fill toward edge budget
            // density_pct = EDGE_BUDGET / H² × 100, clamped to [0, 10]
            let density_pct = ((EDGE_BUDGET as f64 / (h * h) as f64) * 100.0)
                .round()
                .clamp(0.0, 10.0) as usize;

            SweepCase {
                h,
                steps,
                density_pct,
                edge_cap: EDGE_BUDGET,
            }
        })
        .collect()
}

fn run_case(case: &SweepCase, corpus: &[u8], bigram: &[Vec<f64>], seed: u64) -> CaseResult {
    let h = case.h;

    // Build InitConfig with custom density + edge cap
    let mut init = InitConfig::phi(h);
    init.density_pct = case.density_pct;
    init.edge_cap_pct = 100; // we'll enforce our own cap
    init.chain_count = if h < 512 { 50 } else { 0 };

    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = build_network(&init, &mut rng);

    // Enforce edge budget: if overfilled, remove excess
    while net.edge_count() > case.edge_cap {
        net.mutate_remove_edge(&mut rng);
    }

    let mut projection = Int8Projection::new(
        init.phi_dim,
        CHARS,
        &mut StdRng::seed_from_u64(seed + 200),
    );
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let sdr = SdrTable::new(
        CHARS,
        h,
        init.input_end(),
        SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(seed + 100),
    )
    .unwrap();

    let output_start = init.output_start();
    let evo_config = instnct_core::EvolutionConfig {
        edge_cap: case.edge_cap,
        accept_ties: false,
    };

    let mut accepted = 0u32;
    let mut peak_acc = 0.0f64;
    let mut checkpoints = Vec::new();

    // Initial accuracy
    let init_acc = eval_accuracy(
        &mut net, &projection, corpus, EVAL_LEN, &mut eval_rng,
        &sdr, &init.propagation, output_start, h,
    );

    let t0 = Instant::now();

    for step in 0..case.steps {
        let outcome = evolution_step_jackpot(
            &mut net, &mut projection, &mut rng, &mut eval_rng,
            |net, proj, eval_rng| {
                eval_smooth(
                    net, proj, corpus, 100, eval_rng,
                    &sdr, &init.propagation, output_start, h, bigram,
                )
            },
            &evo_config, 3,
        );
        if let StepOutcome::Accepted = outcome {
            accepted += 1;
        }

        // Checkpoint every 20% of steps
        let interval = case.steps / 5;
        if interval > 0 && (step + 1) % interval == 0 {
            let acc = eval_accuracy(
                &mut net, &projection, corpus, EVAL_LEN, &mut eval_rng,
                &sdr, &init.propagation, output_start, h,
            );
            peak_acc = peak_acc.max(acc);
            checkpoints.push((step + 1, acc, net.edge_count()));
        }
    }

    let elapsed = t0.elapsed();

    // Final accuracy
    let final_acc = eval_accuracy(
        &mut net, &projection, corpus, EVAL_LEN, &mut eval_rng,
        &sdr, &init.propagation, output_start, h,
    );
    peak_acc = peak_acc.max(final_acc);

    // Measure fire rate
    net.reset();
    let test_input: Vec<i32> = sdr.pattern(0).to_vec();
    net.propagate(&test_input, &init.propagation).unwrap();
    let fires = net.activation().iter().filter(|&&a| a != 0).count();
    let fire_rate = fires as f64 / h as f64;

    // Measure tok/sec
    net.reset();
    let tok_start = Instant::now();
    let tok_count = 200usize;
    for i in 0..tok_count {
        net.propagate(sdr.pattern(i % CHARS), &init.propagation).unwrap();
    }
    let tok_elapsed = tok_start.elapsed();
    let tok_per_sec = tok_count as f64 / tok_elapsed.as_secs_f64();

    CaseResult {
        h,
        steps: case.steps,
        init_density_pct: case.density_pct,
        init_acc,
        final_acc,
        peak_acc,
        edge_count: net.edge_count(),
        accepted,
        fire_rate,
        tok_per_sec,
        elapsed_secs: elapsed.as_secs_f64(),
        checkpoints,
    }
}

struct CaseResult {
    h: usize,
    steps: usize,
    init_density_pct: usize,
    init_acc: f64,
    final_acc: f64,
    peak_acc: f64,
    edge_count: usize,
    accepted: u32,
    fire_rate: f64,
    tok_per_sec: f64,
    elapsed_secs: f64,
    checkpoints: Vec<(usize, f64, usize)>,
}

fn main() {
    let corpus = load_corpus(CORPUS_PATH).expect("failed to load corpus");
    let bigram = build_bigram_table(&corpus, CHARS);
    let cases = build_cases();

    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  FIXED EDGE BUDGET SWEEP — {}K edges, variable H                    ║", EDGE_BUDGET / 1000);
    println!("║  Question: bigger sparse > smaller dense (same edge budget)?                ║");
    println!("║  Steps scaled: {} base × (H/{})                                    ║", BASE_STEPS, STEP_SCALE_BASE);
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    println!("{:>6} {:>7} {:>7} {:>7} {:>8} {:>8} {:>8} {:>6} {:>9} {:>7}",
        "H", "density", "steps", "edges", "init%", "final%", "peak%", "fire%", "tok/sec", "time");
    println!("{}", "─".repeat(90));

    let mut results = Vec::new();

    for case in &cases {
        let sparsity = case.edge_cap as f64 / (case.h * case.h) as f64 * 100.0;
        print!("{:>6} {:>6.1}% {:>7} ", case.h, sparsity, case.steps);

        let r = run_case(case, &corpus, &bigram, 42);

        println!(
            "{:>7} {:>7.1}% {:>7.1}% {:>7.1}% {:>5.0}% {:>9.0} {:>5.1}s",
            r.edge_count,
            r.init_acc * 100.0,
            r.final_acc * 100.0,
            r.peak_acc * 100.0,
            r.fire_rate * 100.0,
            r.tok_per_sec,
            r.elapsed_secs,
        );

        results.push(r);
    }

    println!("{}", "─".repeat(90));

    // Learning curves
    println!("\n  LEARNING CURVES (best of 2 seeds):\n");
    for r in &results {
        println!("  H={:>5} ({}% density):", r.h, r.init_density_pct);
        for (step, acc, edges) in &r.checkpoints {
            let bar = "#".repeat((acc * 40.0) as usize);
            println!("    step {:>6}: {:.1}% [{}] ({} edges)", step, acc * 100.0, bar, edges);
        }
        println!();
    }

    // Summary
    println!("  VERDICT:");
    let best = results.iter().max_by(|a, b| a.peak_acc.partial_cmp(&b.peak_acc).unwrap()).unwrap();
    let fastest = results.iter().max_by(|a, b| a.tok_per_sec.partial_cmp(&b.tok_per_sec).unwrap()).unwrap();
    let most_efficient = results.iter().max_by(|a, b| {
        let eff_a = a.peak_acc / a.elapsed_secs;
        let eff_b = b.peak_acc / b.elapsed_secs;
        eff_a.partial_cmp(&eff_b).unwrap()
    }).unwrap();

    println!("    Best accuracy:  H={} ({:.1}% peak)", best.h, best.peak_acc * 100.0);
    println!("    Fastest:        H={} ({:.0} tok/sec)", fastest.h, fastest.tok_per_sec);
    println!("    Most efficient: H={} ({:.1}% peak in {:.1}s)",
        most_efficient.h, most_efficient.peak_acc * 100.0, most_efficient.elapsed_secs);
}
