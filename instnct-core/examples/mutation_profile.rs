//! Mutation profiler: tracks per-operator acceptance rates, score deltas,
//! and network health metrics across a long evolution run.
//!
//! Run: cargo run --example mutation_profile --release

use instnct_core::{load_corpus, Int8Projection, Network, PropagationConfig, SdrTable};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const CHARS: usize = 27;
const NEURON_COUNT: usize = 256;
const PHI_DIM: usize = 158;
const INPUT_END: usize = PHI_DIM;
const OUTPUT_START: usize = NEURON_COUNT - PHI_DIM;
const SDR_ACTIVE_PCT: usize = 20;
const EDGE_CAP: usize = NEURON_COUNT * NEURON_COUNT * 7 / 100;

const MUTATION_NAMES: [&str; 11] = [
    "add_edge", "remove_edge", "rewire", "reverse", "mirror",
    "enhance", "theta", "channel", "polarity", "projection", "TOTAL",
];

#[derive(Default, Clone, Copy)]
struct MutationStats {
    attempts: u32,
    mutated: u32,       // mutation actually changed something
    accepted: u32,
    rejected: u32,
    score_delta_sum: f64, // sum of (after - before) for accepted steps
}

#[derive(Default)]
struct NetworkHealth {
    silent_neurons: usize,      // charge == 0 after eval
    firing_neurons: usize,      // activation != 0 after eval
    edge_count: usize,
    mean_charge: f64,
}


fn sample_eval_offset(corpus_len: usize, len: usize, rng: &mut StdRng) -> Option<usize> {
    if corpus_len <= len { return None; }
    let max_offset = corpus_len - len - 1;
    Some(rng.gen_range(0..=max_offset))
}

fn eval_accuracy(
    net: &mut Network,
    projection: &Int8Projection,
    corpus: &[u8],
    len: usize,
    rng: &mut StdRng,
    sdr: &SdrTable,
    config: &PropagationConfig,
) -> f64 {
    let Some(off) = sample_eval_offset(corpus.len(), len, rng) else { return 0.0; };
    let seg = &corpus[off..off + len + 1];
    net.reset();
    let mut correct = 0u32;
    for i in 0..len {
        net.propagate(sdr.pattern(seg[i] as usize), config).unwrap();
        if projection.predict(&net.charge()[OUTPUT_START..NEURON_COUNT]) == seg[i + 1] as usize {
            correct += 1;
        }
    }
    correct as f64 / len as f64
}

fn measure_health(net: &Network) -> NetworkHealth {
    let charge = net.charge();
    let activation = net.activation();
    let silent = charge.iter().filter(|&&c| c == 0).count();
    let firing = activation.iter().filter(|&&a| a != 0).count();
    let mean_charge = if charge.is_empty() {
        0.0
    } else {
        charge.iter().map(|&c| c as f64).sum::<f64>() / charge.len() as f64
    };
    NetworkHealth {
        silent_neurons: silent,
        firing_neurons: firing,
        edge_count: net.edge_count(),
        mean_charge,
    }
}

fn classify_roll(roll: u32) -> usize {
    match roll {
        0..25 => 0,   // add_edge
        25..40 => 1,  // remove_edge
        40..50 => 2,  // rewire
        50..65 => 3,  // reverse
        65..72 => 4,  // mirror
        72..80 => 5,  // enhance
        80..85 => 6,  // theta
        85..90 => 7,  // channel
        90..95 => 8,  // polarity (taking from projection budget for more data)
        _ => 9,       // projection
    }
}

fn do_mutation(
    net: &mut Network,
    projection: &mut Int8Projection,
    rng: &mut impl Rng,
    mutation_type: usize,
) -> (bool, Option<instnct_core::WeightBackup>) {
    match mutation_type {
        0 => (net.mutate_add_edge(rng), None),
        1 => (net.mutate_remove_edge(rng), None),
        2 => (net.mutate_rewire(rng), None),
        3 => (net.mutate_reverse(rng), None),
        4 => (net.mutate_mirror(rng), None),
        5 => (net.mutate_enhance(rng), None),
        6 => (net.mutate_theta(rng), None),
        7 => (net.mutate_channel(rng), None),
        8 => (net.mutate_polarity(rng), None),
        _ => (true, Some(projection.mutate_one(rng))),
    }
}

fn build_network(rng: &mut StdRng) -> Network {
    let mut net = Network::new(NEURON_COUNT);
    let target_edges = NEURON_COUNT * NEURON_COUNT * 5 / 100;
    for _ in 0..target_edges * 3 {
        net.mutate_add_edge(rng);
        if net.edge_count() >= target_edges { break; }
    }
    for i in 0..NEURON_COUNT {
        net.threshold_mut()[i] = rng.gen_range(0..=7);
        net.channel_mut()[i] = rng.gen_range(1..=8);
        if rng.gen_ratio(1, 10) { net.polarity_mut()[i] = -1; }
    }
    net
}

fn print_header() {
    println!("{:<12} {:>6} {:>6} {:>6} {:>6} {:>7} {:>8}",
        "mutation", "tries", "mutd", "accpt", "rejct", "acc%", "avg_delta");
}

fn print_stats(stats: &[MutationStats; 11]) {
    for (i, s) in stats.iter().enumerate() {
        let total_decided = s.accepted + s.rejected;
        let acc_pct = if total_decided > 0 {
            s.accepted as f64 / total_decided as f64 * 100.0
        } else { 0.0 };
        let avg_delta = if s.accepted > 0 {
            s.score_delta_sum / s.accepted as f64 * 100.0
        } else { 0.0 };
        println!("{:<12} {:>6} {:>6} {:>6} {:>6} {:>6.1}% {:>+7.2}pp",
            MUTATION_NAMES[i], s.attempts, s.mutated, s.accepted, s.rejected, acc_pct, avg_delta);
    }
}

fn main() {
    let seed = 42u64;
    let steps = 100_000;
    let report_interval = 10_000;

    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });
    println!("Loading corpus...");
    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");
    println!("  {} chars\n", corpus.len());

    let config = PropagationConfig {
        ticks_per_token: 6,
        input_duration_ticks: 2,
        decay_interval_ticks: 6,
        use_refractory: false,
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = build_network(&mut rng);
    let mut projection = Int8Projection::new(PHI_DIM, CHARS, &mut StdRng::seed_from_u64(seed + 200));
    let mut eval_rng = StdRng::seed_from_u64(seed + 1000);
    let sdr = SdrTable::new(CHARS, NEURON_COUNT, INPUT_END, SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(seed + 100)).unwrap();

    let mut stats = [MutationStats::default(); 11];
    let mut window_stats = [MutationStats::default(); 11]; // per-window

    println!("=== Mutation Profile: seed={seed}, {steps} steps ===\n");

    for step in 0..steps {
        // Paired eval: before
        let eval_snapshot = eval_rng.clone();
        let before = eval_accuracy(&mut net, &projection, &corpus, 100, &mut eval_rng, &sdr, &config);
        eval_rng = eval_snapshot;

        // Select and apply mutation
        let snapshot = net.save_state();
        let roll = rng.gen_range(0..100u32);
        let mutation_type = classify_roll(roll);
        let (mutated, weight_backup) = do_mutation(&mut net, &mut projection, &mut rng, mutation_type);

        stats[mutation_type].attempts += 1;
        stats[10].attempts += 1;
        window_stats[mutation_type].attempts += 1;
        window_stats[10].attempts += 1;

        if !mutated {
            let _ = eval_accuracy(&mut net, &projection, &corpus, 100, &mut eval_rng, &sdr, &config);
            continue;
        }

        stats[mutation_type].mutated += 1;
        stats[10].mutated += 1;
        window_stats[mutation_type].mutated += 1;
        window_stats[10].mutated += 1;

        // Paired eval: after
        let after = eval_accuracy(&mut net, &projection, &corpus, 100, &mut eval_rng, &sdr, &config);

        let accepted = if net.edge_count() < EDGE_CAP {
            after >= before
        } else {
            after > before
        };

        if accepted {
            stats[mutation_type].accepted += 1;
            stats[mutation_type].score_delta_sum += after - before;
            stats[10].accepted += 1;
            stats[10].score_delta_sum += after - before;
            window_stats[mutation_type].accepted += 1;
            window_stats[mutation_type].score_delta_sum += after - before;
            window_stats[10].accepted += 1;
            window_stats[10].score_delta_sum += after - before;
        } else {
            net.restore_state(&snapshot);
            if let Some(backup) = weight_backup {
                projection.rollback(backup);
            }
            stats[mutation_type].rejected += 1;
            stats[10].rejected += 1;
            window_stats[mutation_type].rejected += 1;
            window_stats[10].rejected += 1;
        }

        if (step + 1) % report_interval == 0 {
            let full = eval_accuracy(&mut net, &projection, &corpus, 2000, &mut eval_rng, &sdr, &config);
            let health = measure_health(&net);
            println!("--- step {} | accuracy: {:.1}% | edges: {} | silent: {}/{} | firing: {} | mean_charge: {:.2} ---",
                step + 1, full * 100.0, health.edge_count,
                health.silent_neurons, NEURON_COUNT, health.firing_neurons, health.mean_charge);
            println!("Window (last {report_interval} steps):");
            print_header();
            print_stats(&window_stats);
            println!();
            window_stats = [MutationStats::default(); 11];
        }
    }

    println!("=== CUMULATIVE ({steps} steps) ===");
    print_header();
    print_stats(&stats);
}
