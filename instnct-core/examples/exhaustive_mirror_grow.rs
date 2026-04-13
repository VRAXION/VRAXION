//! Exhaustive Mirror Grower: add hidden neurons one-by-one,
//! exhaustive search all weights in intN, tied-weight decoder (Wᵀ).
//!
//! Architecture: 8 input → K hidden (C19) → 8 output (sigmoid, Wᵀ)
//! Each hidden neuron: dense (connects to all 8 inputs), int weights
//! Decoder: tied weights (Wᵀ) + separately optimized output biases
//! Search: exhaustive per neuron, frozen after accepted
//!
//! Run: cargo run --example exhaustive_mirror_grow --release

use std::time::Instant;

fn load_unique_bytes(path: &str) -> Vec<u8> {
    let text = std::fs::read(path).expect("failed to read corpus");
    let mut seen = [false; 256]; for &b in &text { seen[b as usize] = true; }
    (0..=255u8).filter(|&b| seen[b as usize]).collect()
}

fn byte_to_bits(b: u8) -> [f32; 8] {
    let mut bits = [0.0f32; 8]; for i in 0..8 { bits[i] = ((b >> i) & 1) as f32; } bits
}

fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.1); let rho = rho.max(0.0); let l = 6.0 * c;
    if x >= l { return x - l; } if x <= -l { return x + l; }
    let scaled = x / c; let n = scaled.floor(); let t = scaled - n;
    let h = t * (1.0 - t); let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sgn * h + rho * h * h)
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

// ══════════════════════════════════════════════════════
// FROZEN NEURON
// ══════════════════════════════════════════════════════
#[derive(Clone)]
struct FrozenNeuron {
    weights: Vec<i32>,  // 8 weights (one per input bit)
    bias: i32,
    c: f32,
    rho: f32,
}

impl FrozenNeuron {
    fn eval(&self, input: &[f32; 8]) -> f32 {
        let mut dot = self.bias as f32;
        for i in 0..8 { dot += self.weights[i] as f32 * input[i]; }
        c19(dot, self.c, self.rho)
    }
}

// ══════════════════════════════════════════════════════
// MIRROR EVAL — compute round-trip accuracy
// Given K hidden neurons, decoder uses tied weights (Wᵀ):
//   output_j = sigmoid(Σ_k w_kj * hidden_k + out_bias_j)
// where w_kj = neuron_k.weights[j] (tied!)
// Output biases are optimized per-bit by threshold sweep.
// ══════════════════════════════════════════════════════
fn eval_mirror_roundtrip(
    neurons: &[FrozenNeuron],
    inputs: &[[f32; 8]],
) -> (usize, [f32; 8]) {
    let n = inputs.len();
    let k = neurons.len();
    if k == 0 { return (0, [0.0; 8]); }

    // Compute hidden activations for all bytes
    let hidden: Vec<Vec<f32>> = inputs.iter().map(|inp| {
        neurons.iter().map(|neuron| neuron.eval(inp)).collect()
    }).collect();

    // For each output bit j, compute decoder pre-sigmoid:
    //   z_j = Σ_k w_kj * hidden_k + out_bias_j
    // Then find optimal out_bias_j via sweep
    let mut best_biases = [0.0f32; 8];
    let mut total_correct = 0usize;

    // First compute raw decoder outputs (without bias) for each byte × output bit
    let mut raw_z = vec![[0.0f32; 8]; n]; // [byte][output_bit]
    for bi in 0..n {
        for j in 0..8 {
            let mut z = 0.0f32;
            for ki in 0..k {
                // Tied weight: w_kj = neurons[ki].weights[j]
                z += neurons[ki].weights[j] as f32 * hidden[bi][ki];
            }
            raw_z[bi][j] = z;
        }
    }

    // For each output bit, find optimal bias
    for j in 0..8 {
        let mut best_acc_j = 0usize;
        let mut best_b = 0.0f32;

        // Collect all z values and their targets for this bit
        let mut z_targets: Vec<(f32, f32)> = (0..n).map(|bi| (raw_z[bi][j], inputs[bi][j])).collect();
        z_targets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Try bias values that place threshold between consecutive z values
        let mut candidates: Vec<f32> = vec![-100.0, 100.0];
        for i in 0..z_targets.len() {
            candidates.push(-z_targets[i].0); // bias = -z makes sigmoid(z+bias) = 0.5
            if i + 1 < z_targets.len() {
                candidates.push(-(z_targets[i].0 + z_targets[i+1].0) / 2.0);
            }
        }

        for &bias in &candidates {
            let acc_j = (0..n).filter(|&bi| {
                let out = sigmoid(raw_z[bi][j] + bias);
                (out - inputs[bi][j]).abs() < 0.4
            }).count();
            if acc_j > best_acc_j {
                best_acc_j = acc_j;
                best_b = bias;
            }
        }
        best_biases[j] = best_b;
    }

    // Count bytes where ALL 8 bits are correct
    let mut correct = 0;
    for bi in 0..n {
        let ok = (0..8).all(|j| {
            let out = sigmoid(raw_z[bi][j] + best_biases[j]);
            (out - inputs[bi][j]).abs() < 0.4
        });
        if ok { correct += 1; }
    }

    (correct, best_biases)
}

// ══════════════════════════════════════════════════════
// EXHAUSTIVE SEARCH for one new neuron
// ══════════════════════════════════════════════════════
fn search_neuron_exhaustive(
    existing: &[FrozenNeuron],
    inputs: &[[f32; 8]],
    max_int: i32,
    c_grid: &[f32],
    rho_grid: &[f32],
    current_acc: usize,
) -> Option<(FrozenNeuron, usize)> {
    let n_values = (2 * max_int + 1) as usize;
    let total_weight_combos = n_values.pow(9); // 8 weights + 1 bias
    let target = inputs.len();

    let mut best_neuron: Option<FrozenNeuron> = None;
    let mut best_acc = current_acc;
    let mut combos_tested = 0u64;
    let total = total_weight_combos as u64 * c_grid.len() as u64 * rho_grid.len() as u64;

    let t0 = Instant::now();

    for &c in c_grid {
        for &rho in rho_grid {
            for combo in 0..total_weight_combos {
                // Decode combo into 8 weights + 1 bias
                let mut r = combo;
                let mut weights = [0i32; 8];
                for w in &mut weights { *w = (r % n_values) as i32 - max_int; r /= n_values; }
                let bias = (r % n_values) as i32 - max_int;

                // Skip all-zero (useless)
                if weights.iter().all(|&w| w == 0) { continue; }

                combos_tested += 1;

                // Build candidate neuron
                let neuron = FrozenNeuron {
                    weights: weights.to_vec(), bias, c, rho,
                };

                // Eval with this neuron added
                let mut test_neurons = existing.to_vec();
                test_neurons.push(neuron.clone());

                let (acc, _) = eval_mirror_roundtrip(&test_neurons, inputs);

                if acc > best_acc {
                    best_acc = acc;
                    best_neuron = Some(neuron);

                    let elapsed = t0.elapsed().as_secs_f64();
                    let rate = combos_tested as f64 / elapsed;
                    println!("      [{:.0}%] NEW BEST: {}/{} w={:?} b={} c={:.1} rho={:.1} ({:.0} combo/s)",
                        combos_tested as f64 / total as f64 * 100.0,
                        acc, target, &weights, bias, c, rho, rate);

                    if acc == target { return Some((best_neuron.unwrap(), best_acc)); }
                }

                // Progress report
                if combos_tested % 500_000 == 0 {
                    let elapsed = t0.elapsed().as_secs_f64();
                    let rate = combos_tested as f64 / elapsed;
                    let eta = (total - combos_tested) as f64 / rate;
                    eprint!("\r      [{:.1}%] {}/{} combos, {:.0}/s, ETA {:.0}s, best={}/{}   ",
                        combos_tested as f64 / total as f64 * 100.0,
                        combos_tested, total, rate, eta, best_acc, target);
                }
            }
        }
    }
    eprintln!();

    if best_acc > current_acc { Some((best_neuron.unwrap(), best_acc)) } else { None }
}

// ══════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════
fn main() {
    let t0 = Instant::now();
    let unique = load_unique_bytes("instnct-core/tests/fixtures/alice_corpus.txt");
    let inputs: Vec<[f32; 8]> = unique.iter().map(|&b| byte_to_bits(b)).collect();

    println!("=== EXHAUSTIVE MIRROR GROWER ===");
    println!("{} unique bytes, dense (8 inputs per neuron), tied Wᵀ decoder", unique.len());
    println!("Add 1 neuron at a time, exhaustive intN search, freeze, repeat\n");

    let c_grid: Vec<f32> = vec![0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0];
    let rho_grid: Vec<f32> = vec![0.0, 0.5, 1.0, 2.0, 5.0, 10.0];

    for &(label, max_int) in &[("ternary", 1i32), ("int3", 3)] {
        let n_values = (2 * max_int + 1) as usize;
        let combos_per_neuron = n_values.pow(9) * c_grid.len() * rho_grid.len();

        println!("━━━ {} (±{}, {} values, {} combos/neuron) ━━━\n",
            label, max_int, n_values, combos_per_neuron);

        let mut neurons: Vec<FrozenNeuron> = Vec::new();
        let mut current_acc = 0usize;
        let max_neurons = 15;

        println!("  {:>4} {:>8} {:>8} {:>10} {:>20}",
            "step", "acc", "neurons", "params", "weights");
        println!("  {}", "─".repeat(60));

        for step in 0..max_neurons {
            let result = search_neuron_exhaustive(
                &neurons, &inputs, max_int, &c_grid, &rho_grid, current_acc,
            );

            if let Some((neuron, acc)) = result {
                current_acc = acc;
                let ws: Vec<String> = neuron.weights.iter().map(|w| format!("{:>2}", w)).collect();
                let total_params = (neurons.len() + 1) * 11; // 8w + bias + c + rho per neuron

                println!("  {:>4} {:>5}/{:<2} {:>8} {:>10} [{}] b={} c={:.1} ρ={:.1}",
                    step + 1, acc, unique.len(),
                    neurons.len() + 1, total_params,
                    ws.join(","), neuron.bias, neuron.c, neuron.rho);

                neurons.push(neuron);

                if current_acc == unique.len() {
                    println!("\n  ★★★ PERFECT: {} neurons, {} total params ★★★",
                        neurons.len(), neurons.len() * 11);

                    // Show all frozen neurons
                    println!("\n  Frozen network:");
                    for (i, n) in neurons.iter().enumerate() {
                        let ws: Vec<String> = n.weights.iter().map(|w| format!("{:>2}", w)).collect();
                        println!("    N{}: w=[{}] b={} c={:.1} ρ={:.1}",
                            i, ws.join(","), n.bias, n.c, n.rho);
                    }
                    break;
                }
            } else {
                println!("  {:>4} — no improvement found, stopping", step + 1);
                break;
            }
        }

        println!("\n  Final: {}/{}, {} neurons, {:.1}s\n",
            current_acc, unique.len(), neurons.len(), t0.elapsed().as_secs_f64());
    }

    println!("Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
