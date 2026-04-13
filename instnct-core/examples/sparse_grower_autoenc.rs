//! Sparse Grower Autoencoder: INSTNCT-native neuron-by-neuron growth
//!
//! Instead of dense MLP + backprop, grow neurons ONE AT A TIME:
//! - Each neuron connects to 2-4 selected parents (SPARSE)
//! - Int8 weights (not ternary) — found via mini-backprop per neuron
//! - C19 activation with learnable c, rho
//! - Freeze after each neuron
//! - Round-trip validation via linear decoder at each step
//!
//! Compare: dense MLP (H=4, 64 params, 100%) vs sparse grower (fewer edges?)
//!
//! Run: cargo run --example sparse_grower_autoenc --release

use std::time::Instant;

struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 { self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.s }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn normal(&mut self) -> f32 { let u1 = self.f32().max(1e-7); let u2 = self.f32(); (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() }
    fn pick(&mut self, n: usize) -> usize { self.next() as usize % n }
}

fn load_unique_bytes(path: &str) -> Vec<u8> {
    let text = std::fs::read(path).expect("failed to read corpus");
    let mut seen = [false; 256]; for &b in &text { seen[b as usize] = true; }
    (0..=255u8).filter(|&b| seen[b as usize]).collect()
}

fn byte_to_bits(b: u8) -> Vec<f32> {
    (0..8).map(|i| ((b >> i) & 1) as f32).collect()
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
// SPARSE NEURON — connects to selected parents only
// ══════════════════════════════════════════════════════
#[derive(Clone)]
struct SpNeuron {
    parents: Vec<usize>,      // indices into signal array
    weights: Vec<f32>,        // one per parent (float, will be quantized)
    bias: f32,
    c: f32,                   // C19 param
    rho: f32,                 // C19 param
}

impl SpNeuron {
    fn eval(&self, signals: &[f32]) -> f32 {
        let mut dot = self.bias;
        for (&p, &w) in self.parents.iter().zip(&self.weights) {
            dot += w * signals[p];
        }
        c19(dot, self.c, self.rho)
    }

    fn n_params(&self) -> usize {
        self.parents.len() + 1 + 2 // weights + bias + c + rho
    }

    /// Quantize weights + bias to int8, return (qweights, qbias, scale_w, scale_b)
    fn quantize_i8(&self) -> (Vec<i8>, i8, f32, f32) {
        let max_w = self.weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max).max(1e-7);
        let scale_w = max_w / 127.0;
        let qw: Vec<i8> = self.weights.iter().map(|&w| (w / scale_w).round().max(-127.0).min(127.0) as i8).collect();
        let scale_b = self.bias.abs().max(1e-7) / 127.0;
        let qb = (self.bias / scale_b).round().max(-127.0).min(127.0) as i8;
        (qw, qb, scale_w, scale_b)
    }
}

// ══════════════════════════════════════════════════════
// SPARSE NET — growing network
// ══════════════════════════════════════════════════════
#[derive(Clone)]
struct SparseNet {
    n_in: usize,
    neurons: Vec<SpNeuron>,
}

impl SparseNet {
    fn new(n_in: usize) -> Self { SparseNet { n_in, neurons: Vec::new() } }

    fn n_signals(&self) -> usize { self.n_in + self.neurons.len() }

    fn eval_all(&self, input: &[f32]) -> Vec<f32> {
        let mut signals: Vec<f32> = input.to_vec();
        for n in &self.neurons { signals.push(n.eval(&signals)); }
        signals
    }

    fn total_edges(&self) -> usize { self.neurons.iter().map(|n| n.parents.len()).sum() }
    fn total_params(&self) -> usize { self.neurons.iter().map(|n| n.n_params()).sum() }
}

// ══════════════════════════════════════════════════════
// LINEAR DECODER — trains on hidden activations to reconstruct input
// ══════════════════════════════════════════════════════
fn train_linear_decoder(
    all_signals: &[Vec<f32>],  // signals per byte
    targets: &[Vec<f32>],      // original bits per byte
    n_signals: usize,
) -> (Vec<Vec<f32>>, Vec<f32>) {
    // 8 output neurons, each linear: out_i = sigmoid(Σ w_ij * signal_j + b_i)
    let mut w = vec![vec![0.0f32; n_signals]; 8];
    let mut b = vec![0.0f32; 8];
    let lr = 0.1;

    for _ in 0..2000 {
        for (sigs, tgt) in all_signals.iter().zip(targets) {
            for i in 0..8 {
                let mut z = b[i];
                for j in 0..n_signals { z += w[i][j] * sigs[j]; }
                let a = sigmoid(z);
                let err = a - tgt[i];
                let g = err * a * (1.0 - a);
                for j in 0..n_signals { w[i][j] -= lr * g * sigs[j]; }
                b[i] -= lr * g;
            }
        }
    }
    (w, b)
}

fn eval_decoder(
    w: &[Vec<f32>], b: &[f32],
    all_signals: &[Vec<f32>],
    targets: &[Vec<f32>],
) -> usize {
    let mut correct = 0;
    for (sigs, tgt) in all_signals.iter().zip(targets) {
        let mut ok = true;
        for i in 0..8 {
            let mut z = b[i];
            for j in 0..sigs.len() { z += w[i][j] * sigs[j]; }
            let a = sigmoid(z);
            if (a - tgt[i]).abs() > 0.4 { ok = false; break; }
        }
        if ok { correct += 1; }
    }
    correct
}

// ══════════════════════════════════════════════════════
// NEURON SEARCH — find best neuron to add
// ══════════════════════════════════════════════════════
fn search_best_neuron(
    net: &SparseNet,
    all_current_signals: &[Vec<f32>],
    targets: &[Vec<f32>],
    unique_bytes: &[u8],
    current_acc: usize,
    rng: &mut Rng,
    max_fan: usize,
) -> Option<SpNeuron> {
    let n_sig = net.n_signals();
    let mut best_neuron: Option<SpNeuron> = None;
    let mut best_acc = current_acc;

    // Try many random parent sets + mini-backprop per candidate
    let n_candidates = 200;

    for _ in 0..n_candidates {
        // Random parent set (2-4 parents)
        let fan = 2 + rng.pick(max_fan - 1);
        let mut parents: Vec<usize> = Vec::new();
        for _ in 0..fan * 3 {
            if parents.len() >= fan { break; }
            let p = rng.pick(n_sig);
            if !parents.contains(&p) { parents.push(p); }
        }
        if parents.len() < 2 { continue; }

        // Initialize neuron with random weights
        let mut neuron = SpNeuron {
            parents: parents.clone(),
            weights: parents.iter().map(|_| rng.normal() * 0.5).collect(),
            bias: 0.0,
            c: 2.0 + rng.f32() * 6.0,
            rho: rng.f32() * 3.0,
        };

        // Mini-backprop: optimize this single neuron's weights to maximize
        // decoder accuracy when added to the network
        // We use a proxy: minimize residual error that current signals can't explain
        let mini_epochs = 300;
        let mini_lr = 0.05;

        for epoch in 0..mini_epochs {
            let lr = mini_lr * (1.0 - epoch as f32 / mini_epochs as f32 * 0.8);
            for (si, (sigs, tgt)) in all_current_signals.iter().zip(targets.iter()).enumerate() {
                // Forward through this neuron
                let mut dot = neuron.bias;
                for (&p, &w) in neuron.parents.iter().zip(&neuron.weights) {
                    dot += w * sigs[p];
                }
                let act = c19(dot, neuron.c, neuron.rho);

                // Simple proxy loss: push activations apart for different bytes
                // Use a contrastive signal: different targets should get different activations
                let target_hash = tgt.iter().enumerate().map(|(i, &v)| v * (i as f32 + 1.0)).sum::<f32>();
                let proxy_target = sigmoid(target_hash * 0.3);
                let err = act - proxy_target;

                // Gradient through C19 (numerical)
                let eps = 0.001;
                for k in 0..neuron.weights.len() {
                    let mut dot_p = neuron.bias;
                    for (&p, &w) in neuron.parents.iter().zip(&neuron.weights) { dot_p += w * sigs[p]; }
                    let orig_w = neuron.weights[k];
                    neuron.weights[k] = orig_w + eps;
                    let mut dot2 = neuron.bias;
                    for (&p, &w) in neuron.parents.iter().zip(&neuron.weights) { dot2 += w * sigs[p]; }
                    let act2 = c19(dot2, neuron.c, neuron.rho);
                    neuron.weights[k] = orig_w;
                    let grad = (act2 - act) / eps * err;
                    neuron.weights[k] -= lr * grad;
                }
                // Bias gradient
                let act_bp = c19(dot + eps, neuron.c, neuron.rho);
                let bias_grad = (act_bp - act) / eps * err;
                neuron.bias -= lr * bias_grad;

                // c, rho gradients
                let act_cp = c19(dot, neuron.c + eps, neuron.rho);
                neuron.c -= lr * 0.3 * (act_cp - act) / eps * err;
                let act_rp = c19(dot, neuron.c, neuron.rho + eps);
                neuron.rho -= lr * 0.3 * (act_rp - act) / eps * err;
                neuron.c = neuron.c.max(0.1).min(20.0);
                neuron.rho = neuron.rho.max(0.0).min(10.0);
            }
        }

        // Evaluate: add this neuron, retrain decoder on HIDDEN ONLY, check accuracy
        let n_in = 8;
        let mut hidden_signals: Vec<Vec<f32>> = Vec::new();
        for sigs in all_current_signals {
            let mut new_hidden: Vec<f32> = sigs[n_in..].to_vec(); // existing hidden only
            new_hidden.push(neuron.eval(&sigs));
            hidden_signals.push(new_hidden);
        }

        let n_hidden = hidden_signals[0].len();
        let (dec_w, dec_b) = train_linear_decoder(&hidden_signals, targets, n_hidden);
        let acc = eval_decoder(&dec_w, &dec_b, &hidden_signals, targets);

        if acc > best_acc {
            best_acc = acc;
            best_neuron = Some(neuron.clone());
        }
    }

    best_neuron
}

// ══════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════
fn main() {
    let t0 = Instant::now();
    let unique = load_unique_bytes("instnct-core/tests/fixtures/alice_corpus.txt");
    let targets: Vec<Vec<f32>> = unique.iter().map(|&b| byte_to_bits(b)).collect();

    println!("=== SPARSE GROWER AUTOENCODER ===");
    println!("{} unique bytes, C19 + int8, neuron-by-neuron growth", unique.len());
    println!("Each neuron: sparse (2-4 parents), mini-backprop per neuron, freeze");
    println!("Decoder: retrained linear decoder after each neuron addition\n");

    let max_neurons = 30;
    let max_fan = 5;

    for &seed in &[42u64, 123, 777] {
        println!("━━━ Seed {} ━━━\n", seed);
        let mut rng = Rng::new(seed);
        let mut net = SparseNet::new(8);

        // Initial signals = just the 8 input bits
        let mut all_signals: Vec<Vec<f32>> = unique.iter().map(|&b| {
            let inp = byte_to_bits(b);
            net.eval_all(&inp)
        }).collect();

        // Initial accuracy: decoder sees ONLY hidden neurons (NOT raw input)
        // With 0 hidden neurons, decoder has nothing → 0%
        let mut current_acc = 0usize;

        println!("  {:>4} {:>6} {:>8} {:>8} {:>10} {:>10} {:>20}",
            "step", "acc", "neurons", "edges", "params", "bytes(i8)", "parents");
        println!("  {}", "─".repeat(75));
        println!("  {:>4} {:>3}/{:<2} {:>8} {:>8} {:>10} {:>10} {:>20}",
            "init", current_acc, unique.len(), 0, 0, 0, 0, "(no hidden yet)");

        let mut stall = 0;
        for step in 0..max_neurons {
            let result = search_best_neuron(
                &net, &all_signals, &targets, &unique, current_acc, &mut rng, max_fan,
            );

            if let Some(neuron) = result {
                let parents_str: Vec<String> = neuron.parents.iter().map(|&p| {
                    if p < 8 { format!("x{}", p) } else { format!("N{}", p - 8) }
                }).collect();

                // Add neuron to net
                net.neurons.push(neuron.clone());

                // Recompute all signals
                all_signals = unique.iter().map(|&b| {
                    let inp = byte_to_bits(b);
                    net.eval_all(&inp)
                }).collect();

                // Retrain decoder on ONLY hidden neuron outputs (not raw input)
                let hidden_only: Vec<Vec<f32>> = all_signals.iter()
                    .map(|s| s[net.n_in..].to_vec())
                    .collect();
                let n_hidden = net.neurons.len();
                let (dec_w, dec_b) = train_linear_decoder(&hidden_only, &targets, n_hidden);
                let new_acc = eval_decoder(&dec_w, &dec_b, &hidden_only, &targets);

                let delta = new_acc as i32 - current_acc as i32;
                current_acc = new_acc;

                // Quantize check
                let (qw, qb, _, _) = neuron.quantize_i8();
                let qw_str: Vec<String> = qw.iter().map(|w| format!("{}", w)).collect();

                println!("  {:>4} {:>3}/{:<2} {:>8} {:>8} {:>10} {:>10} [{}] c={:.1} rho={:.1} Δ={:+}",
                    step + 1, new_acc, unique.len(),
                    net.neurons.len(), net.total_edges(), net.total_params(),
                    net.total_params(), // int8 bytes
                    parents_str.join(","), neuron.c, neuron.rho, delta);

                if current_acc == unique.len() {
                    println!("\n  ★★★ PERFECT at {} neurons, {} edges, {} params ★★★",
                        net.neurons.len(), net.total_edges(), net.total_params());
                    break;
                }

                if delta <= 0 { stall += 1; } else { stall = 0; }
                if stall > 10 { println!("  Stalled after {} steps", step + 1); break; }
            } else {
                stall += 1;
                if stall > 5 { println!("  No improvement found, stopping"); break; }
            }
        }

        // Final summary
        println!("\n  Final: {}/{} accuracy, {} neurons, {} edges, {} params ({} bytes int8)",
            current_acc, unique.len(), net.neurons.len(), net.total_edges(),
            net.total_params(), net.total_params());

        // Compare with dense MLP
        println!("\n  Comparison:");
        println!("    Dense MLP (C19, H=4, B=4):  64 params, 32 edges (dense), 100%");
        println!("    Dense MLP (C19, H=8, B=2):  106 params, 80 edges (dense), 100%");
        println!("    Sparse grower:               {} params, {} edges (sparse), {}%",
            net.total_params(), net.total_edges(),
            current_acc as f64 / unique.len() as f64 * 100.0);

        // Show sparsity
        if net.total_edges() > 0 {
            let max_possible = net.neurons.len() * net.n_signals(); // upper bound
            println!("    Sparsity: {}/{} possible edges used ({:.1}%)",
                net.total_edges(), max_possible,
                net.total_edges() as f64 / max_possible as f64 * 100.0);
        }

        println!("\n  Time: {:.1}s\n", t0.elapsed().as_secs_f64());

        if current_acc == unique.len() { break; }
    }

    println!("Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
