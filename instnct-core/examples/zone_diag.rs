//! Zone diagnostic: which neurons carry the signal?
//! Checks mean charge per zone after propagation on a trained network.
//!
//! Run: cargo run --example zone_diag --release -- <corpus-path>

use instnct_core::{load_corpus, 
    build_network, evolution_step, InitConfig, Int8Projection, SdrTable,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const CHARS: usize = 27;
const SDR_ACTIVE_PCT: usize = 20;


fn main() {
    let init = InitConfig::phi(256);
    let output_start = init.output_start(); // 98
    let input_end = init.input_end();       // 158
    let h = init.neuron_count;              // 256

    let corpus_path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/beta_smoke_corpus.txt".to_string()
    });
    let corpus = load_corpus(&corpus_path).expect("cannot read corpus");

    let mut rng = StdRng::seed_from_u64(42);
    let mut net = build_network(&init, &mut rng);
    let mut proj = Int8Projection::new(init.phi_dim, CHARS, &mut StdRng::seed_from_u64(242));
    let mut eval_rng = StdRng::seed_from_u64(1042);
    let sdr = SdrTable::new(CHARS, h, input_end, SDR_ACTIVE_PCT,
        &mut StdRng::seed_from_u64(142)).unwrap();
    let evo = init.evolution_config();

    // Train 15K steps
    println!("Training 15K steps...");
    for _ in 0..15000 {
        let _ = evolution_step(
            &mut net, &mut proj, &mut rng, &mut eval_rng,
            |net, proj, eval_rng| {
                let off = eval_rng.gen_range(0..corpus.len() - 101);
                let seg = &corpus[off..off + 101];
                net.reset();
                let mut correct = 0u32;
                for i in 0..100 {
                    net.propagate(sdr.pattern(seg[i] as usize), &init.propagation).unwrap();
                    if proj.predict(&net.charge_vec(output_start..h)) == seg[i + 1] as usize {
                        correct += 1;
                    }
                }
                correct as f64 / 100.0
            },
            &evo,
        );
    }

    // Diagnostic: run 1000 tokens, record charge per zone
    println!("\nDiagnostic on 1000 tokens:\n");

    let seg = &corpus[50000..51001];
    net.reset();

    let mut zone_charge_sum = [0u64; 4]; // input_only, overlap, output_only, total
    let mut zone_active_sum = [0u64; 4];
    let mut zone_sizes = [0usize; 3];
    zone_sizes[0] = output_start;               // input_only: 0..98
    zone_sizes[1] = input_end - output_start;    // overlap: 98..158
    zone_sizes[2] = h - input_end;               // output_only: 158..256

    let mut correct = 0u32;
    let mut correct_overlap_charge = 0u64;
    let mut wrong_overlap_charge = 0u64;
    let mut correct_count = 0u32;
    let mut wrong_count = 0u32;

    for i in 0..1000 {
        net.propagate(sdr.pattern(seg[i] as usize), &init.propagation).unwrap();

        let spike = net.spike_data();
        let activation = net.activation();

        // Zone charges
        let c_input: u64 = spike[..output_start].iter().map(|s| s.charge as u64).sum();
        let c_overlap: u64 = spike[output_start..input_end].iter().map(|s| s.charge as u64).sum();
        let c_output: u64 = spike[input_end..h].iter().map(|s| s.charge as u64).sum();
        zone_charge_sum[0] += c_input;
        zone_charge_sum[1] += c_overlap;
        zone_charge_sum[2] += c_output;
        zone_charge_sum[3] += c_input + c_overlap + c_output;

        // Zone active counts
        let a_input: u64 = activation[..output_start].iter().filter(|&&a| a != 0).count() as u64;
        let a_overlap: u64 = activation[output_start..input_end].iter().filter(|&&a| a != 0).count() as u64;
        let a_output: u64 = activation[input_end..h].iter().filter(|&&a| a != 0).count() as u64;
        zone_active_sum[0] += a_input;
        zone_active_sum[1] += a_overlap;
        zone_active_sum[2] += a_output;
        zone_active_sum[3] += a_input + a_overlap + a_output;

        // Correct vs wrong overlap charge
        let predicted = proj.predict(&net.charge_vec(output_start..h));
        if predicted == seg[i + 1] as usize {
            correct += 1;
            correct_overlap_charge += c_overlap;
            correct_count += 1;
        } else {
            wrong_overlap_charge += c_overlap;
            wrong_count += 1;
        }
    }

    let tokens = 1000.0;
    println!("Zone layout: input_only=[0..{output_start})  overlap=[{output_start}..{input_end})  output_only=[{input_end}..{h})");
    println!("Zone sizes:  {} / {} / {}\n", zone_sizes[0], zone_sizes[1], zone_sizes[2]);

    println!("Mean charge per neuron per token:");
    println!("  input_only  [{:>3} neurons]: {:.2}", zone_sizes[0],
        zone_charge_sum[0] as f64 / tokens / zone_sizes[0] as f64);
    println!("  overlap     [{:>3} neurons]: {:.2}", zone_sizes[1],
        zone_charge_sum[1] as f64 / tokens / zone_sizes[1] as f64);
    println!("  output_only [{:>3} neurons]: {:.2}", zone_sizes[2],
        zone_charge_sum[2] as f64 / tokens / zone_sizes[2] as f64);

    println!("\nMean active neurons per token:");
    println!("  input_only  [{:>3} neurons]: {:.1} ({:.0}%)", zone_sizes[0],
        zone_active_sum[0] as f64 / tokens, zone_active_sum[0] as f64 / tokens / zone_sizes[0] as f64 * 100.0);
    println!("  overlap     [{:>3} neurons]: {:.1} ({:.0}%)", zone_sizes[1],
        zone_active_sum[1] as f64 / tokens, zone_active_sum[1] as f64 / tokens / zone_sizes[1] as f64 * 100.0);
    println!("  output_only [{:>3} neurons]: {:.1} ({:.0}%)", zone_sizes[2],
        zone_active_sum[2] as f64 / tokens, zone_active_sum[2] as f64 / tokens / zone_sizes[2] as f64 * 100.0);

    println!("\nAccuracy: {:.1}% ({correct}/1000)", correct as f64 / 10.0);

    println!("\nOverlap charge when CORRECT vs WRONG:");
    let corr_mean = if correct_count > 0 { correct_overlap_charge as f64 / correct_count as f64 } else { 0.0 };
    let wrong_mean = if wrong_count > 0 { wrong_overlap_charge as f64 / wrong_count as f64 } else { 0.0 };
    println!("  correct predictions: mean overlap charge = {:.1}", corr_mean);
    println!("  wrong predictions:   mean overlap charge = {:.1}", wrong_mean);

    // Per-neuron charge heatmap (top 20)
    println!("\nTop 20 neurons by mean charge:");
    net.reset();
    let mut per_neuron_charge = vec![0u64; h];
    for &ch in seg.iter().take(1000) {
        net.propagate(sdr.pattern(ch as usize), &init.propagation).unwrap();
        for (n, s) in net.spike_data().iter().enumerate() {
            per_neuron_charge[n] += s.charge as u64;
        }
    }

    let mut neuron_charges: Vec<(usize, f64)> = (0..h).map(|n| (n, per_neuron_charge[n] as f64 / tokens)).collect();
    neuron_charges.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for &(n, mean_c) in neuron_charges.iter().take(20) {
        let zone = if n < output_start { "INPUT" }
            else if n < input_end { "OVERLAP" }
            else { "OUTPUT" };
        println!("  neuron {:>3} [{:<7}]: mean_charge={:.2}", n, zone, mean_c);
    }
}
