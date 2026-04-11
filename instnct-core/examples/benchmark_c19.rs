//! C19 vs ReLU vs GELU benchmark on synthetic multi-class classification
//!
//! Goal: determine if C19 rho=8's +2.2% advantage on byte-prediction
//! holds on a proper ML benchmark with train/test split, multiple seeds,
//! and multiple hidden layer sizes.
//!
//! Dataset: 10-class synthetic problem with 100 features.
//!   - 5 classes from concentric hypersphere shells (radius-based)
//!   - 5 classes from multi-modal Gaussian clusters in different subspaces
//!   - Features include nonlinear interactions and noise
//!   - 10,000 train / 2,500 test samples
//!
//! Architecture: MLP with 2 hidden layers, Adam optimizer, analytic backprop.
//! Hidden sizes: 64 and 256 (to test if advantage scales with width).
//! Seeds: 5 per configuration.
//!
//! Run: cargo run --example benchmark_c19 --release
//!
//! Output: instnct-core/benchmark_c19_log.txt

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use std::io::Write;
use std::time::Instant;

// ============================================================
// Activation functions with analytic derivatives
// ============================================================

fn c19_fwd(x: f32, rho: f32) -> f32 {
    let l = 6.0f32;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let n = x.floor();
    let t = x - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * h + rho * h * h
}

fn c19_deriv(x: f32, rho: f32) -> f32 {
    let l = 6.0f32;
    if x >= l || x <= -l { return 1.0; }
    let n = x.floor();
    let t = x - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    (sgn + 2.0 * rho * h) * (1.0 - 2.0 * t)
}

fn relu_fwd(x: f32) -> f32 { x.max(0.0) }
fn relu_deriv(x: f32) -> f32 { if x > 0.0 { 1.0 } else { 0.0 } }

fn gelu_fwd(x: f32) -> f32 {
    x * 0.5 * (1.0 + (0.7978846 * (x + 0.044715 * x * x * x)).tanh())
}
fn gelu_deriv(x: f32) -> f32 {
    let cdf = 0.5 * (1.0 + (0.7978846 * (x + 0.044715 * x * x * x)).tanh());
    let inner = 0.7978846 * (x + 0.044715 * x * x * x);
    let sech2 = 1.0 - inner.tanh().powi(2);
    let pdf = 0.7978846 * (1.0 + 3.0 * 0.044715 * x * x) * sech2;
    cdf + x * 0.5 * pdf
}

#[derive(Clone, Copy, PartialEq)]
enum Activation { C19, ReLU, GELU }

impl Activation {
    fn name(&self) -> &'static str {
        match self {
            Activation::C19 => "C19_rho8",
            Activation::ReLU => "ReLU",
            Activation::GELU => "GELU",
        }
    }
    fn fwd(&self, x: f32) -> f32 {
        match self {
            Activation::C19 => c19_fwd(x, 8.0),
            Activation::ReLU => relu_fwd(x),
            Activation::GELU => gelu_fwd(x),
        }
    }
    fn deriv(&self, x: f32) -> f32 {
        match self {
            Activation::C19 => c19_deriv(x, 8.0),
            Activation::ReLU => relu_deriv(x),
            Activation::GELU => gelu_deriv(x),
        }
    }
}

// ============================================================
// Softmax + cross-entropy
// ============================================================

fn softmax(logits: &[f32]) -> Vec<f32> {
    let mx = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&l| (l - mx).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / s).collect()
}

// ============================================================
// 2-hidden-layer MLP: IN -> H1 -> H2 -> N_CLASSES
// ============================================================

const N_CLASSES: usize = 10;
const N_FEATURES: usize = 100;

#[derive(Clone)]
struct MLP2 {
    act: Activation,
    // Layer 1: N_FEATURES -> h1
    h1: usize,
    w1: Vec<f32>, b1: Vec<f32>,
    // Layer 2: h1 -> h2
    h2: usize,
    w2: Vec<f32>, b2: Vec<f32>,
    // Output: h2 -> N_CLASSES
    w3: Vec<f32>, b3: Vec<f32>,
}

struct Cache {
    input: Vec<f32>,
    pre1: Vec<f32>, act1: Vec<f32>,
    pre2: Vec<f32>, act2: Vec<f32>,
    logits: Vec<f32>,
    probs: Vec<f32>,
}

impl MLP2 {
    fn new(h1: usize, h2: usize, act: Activation, rng: &mut StdRng) -> Self {
        let sc1 = (2.0 / N_FEATURES as f32).sqrt();
        let sc2 = (2.0 / h1 as f32).sqrt();
        let sc3 = (2.0 / h2 as f32).sqrt();
        MLP2 {
            act, h1, h2,
            w1: (0..h1 * N_FEATURES).map(|_| rng.gen_range(-sc1..sc1)).collect(),
            b1: vec![0.0; h1],
            w2: (0..h2 * h1).map(|_| rng.gen_range(-sc2..sc2)).collect(),
            b2: vec![0.0; h2],
            w3: (0..N_CLASSES * h2).map(|_| rng.gen_range(-sc3..sc3)).collect(),
            b3: vec![0.0; N_CLASSES],
        }
    }

    fn params(&self) -> usize {
        self.w1.len() + self.b1.len() +
        self.w2.len() + self.b2.len() +
        self.w3.len() + self.b3.len()
    }

    fn forward(&self, input: &[f32]) -> Cache {
        let (h1, h2) = (self.h1, self.h2);

        // Layer 1
        let mut pre1 = vec![0.0f32; h1];
        let mut act1 = vec![0.0f32; h1];
        for i in 0..h1 {
            let mut s = self.b1[i];
            for j in 0..N_FEATURES { s += input[j] * self.w1[i * N_FEATURES + j]; }
            pre1[i] = s;
            act1[i] = self.act.fwd(s);
        }

        // Layer 2
        let mut pre2 = vec![0.0f32; h2];
        let mut act2 = vec![0.0f32; h2];
        for i in 0..h2 {
            let mut s = self.b2[i];
            for j in 0..h1 { s += act1[j] * self.w2[i * h1 + j]; }
            pre2[i] = s;
            act2[i] = self.act.fwd(s);
        }

        // Output
        let mut logits = vec![0.0f32; N_CLASSES];
        for c in 0..N_CLASSES {
            let mut s = self.b3[c];
            for j in 0..h2 { s += act2[j] * self.w3[c * h2 + j]; }
            logits[c] = s;
        }
        let probs = softmax(&logits);

        Cache { input: input.to_vec(), pre1, act1, pre2, act2, logits, probs }
    }

    fn backward(&self, cache: &Cache, target: usize) -> Vec<f32> {
        let (h1, h2) = (self.h1, self.h2);
        let np = self.params();
        let mut grad = vec![0.0f32; np];

        // dL/d_logits = probs - one_hot(target)
        let mut dl = cache.probs.clone();
        dl[target] -= 1.0;

        // Offsets into flat grad vector
        let w1_off = 0;
        let b1_off = h1 * N_FEATURES;
        let w2_off = b1_off + h1;
        let b2_off = w2_off + h2 * h1;
        let w3_off = b2_off + h2;
        let b3_off = w3_off + N_CLASSES * h2;

        // Output layer grads
        let mut d_act2 = vec![0.0f32; h2];
        for c in 0..N_CLASSES {
            grad[b3_off + c] = dl[c];
            for j in 0..h2 {
                grad[w3_off + c * h2 + j] = dl[c] * cache.act2[j];
                d_act2[j] += dl[c] * self.w3[c * h2 + j];
            }
        }

        // Layer 2 backprop
        let mut d_act1 = vec![0.0f32; h1];
        for i in 0..h2 {
            let d_pre2 = d_act2[i] * self.act.deriv(cache.pre2[i]);
            grad[b2_off + i] = d_pre2;
            for j in 0..h1 {
                grad[w2_off + i * h1 + j] = d_pre2 * cache.act1[j];
                d_act1[j] += d_pre2 * self.w2[i * h1 + j];
            }
        }

        // Layer 1 backprop
        for i in 0..h1 {
            let d_pre1 = d_act1[i] * self.act.deriv(cache.pre1[i]);
            grad[b1_off + i] = d_pre1;
            for j in 0..N_FEATURES {
                grad[w1_off + i * N_FEATURES + j] = d_pre1 * cache.input[j];
            }
        }

        grad
    }

    fn param_vec(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.params());
        v.extend(&self.w1); v.extend(&self.b1);
        v.extend(&self.w2); v.extend(&self.b2);
        v.extend(&self.w3); v.extend(&self.b3);
        v
    }

    fn set_param_vec(&mut self, v: &[f32]) {
        let mut o = 0;
        let n = self.w1.len(); self.w1.copy_from_slice(&v[o..o+n]); o += n;
        let n = self.b1.len(); self.b1.copy_from_slice(&v[o..o+n]); o += n;
        let n = self.w2.len(); self.w2.copy_from_slice(&v[o..o+n]); o += n;
        let n = self.b2.len(); self.b2.copy_from_slice(&v[o..o+n]); o += n;
        let n = self.w3.len(); self.w3.copy_from_slice(&v[o..o+n]); o += n;
        let n = self.b3.len(); self.b3.copy_from_slice(&v[o..o+n]); o += n;
    }

    fn accuracy(&self, data: &[(Vec<f32>, usize)]) -> f64 {
        let correct = data.iter().filter(|(inp, t)| {
            let c = self.forward(inp);
            c.probs.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap().0 == *t
        }).count();
        correct as f64 / data.len() as f64
    }

    fn loss(&self, data: &[(Vec<f32>, usize)]) -> f64 {
        let mut l = 0.0f64;
        for (inp, t) in data {
            let c = self.forward(inp);
            l -= (c.probs[*t].max(1e-10) as f64).ln();
        }
        l / data.len() as f64
    }
}

// ============================================================
// Dataset generation: 10-class synthetic problem
// ============================================================

fn generate_dataset(n: usize, rng: &mut StdRng) -> Vec<(Vec<f32>, usize)> {
    let mut data = Vec::with_capacity(n);
    let samples_per_class = n / N_CLASSES;

    for class in 0..N_CLASSES {
        for _ in 0..samples_per_class {
            let mut features = vec![0.0f32; N_FEATURES];

            if class < 5 {
                // Classes 0-4: concentric hypersphere shells in first 20 dims
                // Radius bands: [0.5, 1.5), [1.5, 2.5), ..., [4.5, 5.5)
                let r_min = 0.5 + class as f32;
                let r_max = r_min + 1.0;
                // Sample a random direction in 20D, then scale to desired radius
                let mut raw = vec![0.0f32; 20];
                let mut norm = 0.0f32;
                for j in 0..20 {
                    raw[j] = randn(rng);
                    norm += raw[j] * raw[j];
                }
                norm = norm.sqrt().max(1e-6);
                let target_r = rng.gen_range(r_min..r_max);
                for j in 0..20 {
                    features[j] = raw[j] / norm * target_r;
                }
                // Add nonlinear interaction features in dims 20-39
                for j in 0..20 {
                    features[20 + j] = features[j] * features[(j + 1) % 20]; // pairwise products
                }
                // Noise in remaining dims
                for j in 40..N_FEATURES {
                    features[j] = randn(rng) * 0.3;
                }
            } else {
                // Classes 5-9: Gaussian clusters in different subspaces
                let subspace_start = (class - 5) * 15 + 10; // dims 10-84
                // Cluster center
                let center_val = 2.0 + (class - 5) as f32 * 0.8;
                let sign = if class % 2 == 0 { 1.0 } else { -1.0 };
                for j in 0..15 {
                    let dim = subspace_start + j;
                    if dim < N_FEATURES {
                        features[dim] = sign * center_val + randn(rng) * 0.8;
                    }
                }
                // Nonlinear features: sin/cos of subspace coords in dims 85-99
                for j in 0..15 {
                    let dim = 85 + j;
                    let src = subspace_start + j;
                    if dim < N_FEATURES && src < N_FEATURES {
                        features[dim] = (features[src] * 1.5).sin();
                    }
                }
                // Add noise to other dims
                for j in 0..N_FEATURES {
                    if j < subspace_start || j >= subspace_start + 15 {
                        if j < 85 {
                            features[j] += randn(rng) * 0.2;
                        }
                    }
                }
            }

            data.push((features, class));
        }
    }

    data.shuffle(rng);
    data
}

fn randn(rng: &mut StdRng) -> f32 {
    // Box-Muller transform
    let u1: f32 = rng.gen_range(1e-7..1.0);
    let u2: f32 = rng.gen_range(0.0..std::f32::consts::TAU);
    (-2.0 * u1.ln()).sqrt() * u2.cos()
}

// ============================================================
// Training with Adam
// ============================================================

fn train_and_evaluate(
    h1: usize, h2: usize, act: Activation, seed: u64,
    train: &[(Vec<f32>, usize)], test: &[(Vec<f32>, usize)],
    steps: usize, batch_size: usize, lr: f32,
    logf: &mut std::fs::File,
) -> (f64, f64, Vec<(usize, f64, f64)>) {
    // curve: (step, train_loss, test_acc)
    let mut rng = StdRng::seed_from_u64(seed);
    let mut net = MLP2::new(h1, h2, act, &mut rng);
    let np = net.params();

    let mut m = vec![0.0f32; np];
    let mut v = vec![0.0f32; np];
    let mut shuffle_rng = StdRng::seed_from_u64(seed + 1000);
    let mut indices: Vec<usize> = (0..train.len()).collect();

    let mut curve = Vec::new();

    let test_sub = if test.len() > 1000 { &test[..1000] } else { test };
    let train_sub = if train.len() > 1000 { &train[..1000] } else { train };

    for step in 1..=steps {
        indices.shuffle(&mut shuffle_rng);
        let batch = &indices[..batch_size.min(train.len())];
        let bs = batch.len() as f32;

        // Accumulate gradients
        let mut grad_sum = vec![0.0f32; np];
        for &idx in batch {
            let cache = net.forward(&train[idx].0);
            let g = net.backward(&cache, train[idx].1);
            for i in 0..np { grad_sum[i] += g[i]; }
        }
        for g in &mut grad_sum { *g /= bs; }

        // Adam update
        let mut pv = net.param_vec();
        let t = step as f32;
        let b1c = 1.0 - 0.9f32.powf(t);
        let b2c = 1.0 - 0.999f32.powf(t);
        for i in 0..np {
            m[i] = 0.9 * m[i] + 0.1 * grad_sum[i];
            v[i] = 0.999 * v[i] + 0.001 * grad_sum[i] * grad_sum[i];
            pv[i] -= lr * (m[i] / b1c) / ((v[i] / b2c).sqrt() + 1e-8);
        }
        net.set_param_vec(&pv);

        // Log every 500 steps
        if step % 500 == 0 || step == steps {
            let train_loss = net.loss(train_sub);
            let test_acc = net.accuracy(test_sub);
            curve.push((step, train_loss, test_acc));

            let msg = format!("    [{} h={},{} seed={}] step {:>5}: loss={:.4} test_acc={:.2}%",
                act.name(), h1, h2, seed, step, train_loss, test_acc * 100.0);
            log_msg(logf, &msg);
        }
    }

    let final_test_acc = net.accuracy(test);
    let final_loss = net.loss(test);
    (final_test_acc, final_loss, curve)
}

// ============================================================
// Logging
// ============================================================

fn log_msg(f: &mut std::fs::File, msg: &str) {
    let now = timestamp();
    let line = format!("[{}] {}\n", now, msg);
    print!("{}", line);
    f.write_all(line.as_bytes()).ok();
    f.flush().ok();
}

fn timestamp() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap();
    let secs = d.as_secs();
    let h = (secs / 3600) % 24;
    let m = (secs / 60) % 60;
    let s = secs % 60;
    format!("{:02}:{:02}:{:02}", h, m, s)
}

// ============================================================
// Checkpoint
// ============================================================

fn save_checkpoint(path: &str, results: &[(String, f64, f64)]) {
    let json = serde_json::to_string_pretty(
        &results.iter().map(|(name, acc, loss)| {
            serde_json::json!({"name": name, "acc": acc, "loss": loss})
        }).collect::<Vec<_>>()
    ).unwrap();
    std::fs::write(path, json).ok();
}

// ============================================================
// Main
// ============================================================

fn main() {
    let log_path = "instnct-core/benchmark_c19_log.txt";
    let ckpt_path = "instnct-core/benchmark_c19_ckpt.json";

    let mut logf = std::fs::OpenOptions::new()
        .create(true).append(true)
        .open(log_path).unwrap();

    log_msg(&mut logf, "================================================================");
    log_msg(&mut logf, "  C19 vs ReLU vs GELU BENCHMARK");
    log_msg(&mut logf, "  10-class synthetic classification (100 features)");
    log_msg(&mut logf, "  2-hidden-layer MLP, Adam optimizer, analytic backprop");
    log_msg(&mut logf, "================================================================");

    let t0 = Instant::now();

    // Generate dataset with fixed seed for reproducibility
    log_msg(&mut logf, "Generating dataset...");
    let mut data_rng = StdRng::seed_from_u64(12345);
    let all_data = generate_dataset(12500, &mut data_rng);
    let train_data = &all_data[..10000];
    let test_data = &all_data[10000..];
    log_msg(&mut logf, &format!("  Train: {} samples, Test: {} samples, Features: {}, Classes: {}",
        train_data.len(), test_data.len(), N_FEATURES, N_CLASSES));

    // Check class balance
    let mut class_counts = [0usize; N_CLASSES];
    for (_, c) in train_data { class_counts[*c] += 1; }
    log_msg(&mut logf, &format!("  Class distribution (train): {:?}", class_counts));

    // Compute baseline (random guess = 10%)
    log_msg(&mut logf, "  Random baseline: 10.0%");
    log_msg(&mut logf, "");

    // Configurations to test
    let activations = [Activation::C19, Activation::ReLU, Activation::GELU];
    let hidden_configs: Vec<(usize, usize)> = vec![(64, 32), (256, 128)];
    let seeds: Vec<u64> = vec![42, 137, 256, 777, 1024];
    let steps = 5000;
    let batch_size = 128;
    let lr = 0.001;

    let mut all_results: Vec<(String, f64, f64)> = Vec::new();

    // Store per-config results for summary
    struct ConfigResult {
        label: String,
        accs: Vec<f64>,
        losses: Vec<f64>,
    }
    let mut config_results: Vec<ConfigResult> = Vec::new();

    for &(h1, h2) in &hidden_configs {
        log_msg(&mut logf, &format!("========== Hidden: {} -> {} ==========", h1, h2));

        for &act in &activations {
            let label = format!("{} h={},{}", act.name(), h1, h2);
            log_msg(&mut logf, &format!("--- {} ---", label));

            let mut accs = Vec::new();
            let mut losses = Vec::new();

            for &seed in &seeds {
                let net_tmp = MLP2::new(h1, h2, act, &mut StdRng::seed_from_u64(seed));
                if seed == seeds[0] {
                    log_msg(&mut logf, &format!("  params={}", net_tmp.params()));
                }

                let (acc, loss, _curve) = train_and_evaluate(
                    h1, h2, act, seed,
                    train_data, test_data,
                    steps, batch_size, lr,
                    &mut logf,
                );
                accs.push(acc);
                losses.push(loss);

                let entry = (format!("{}_seed{}", label, seed), acc, loss);
                all_results.push(entry);
                save_checkpoint(ckpt_path, &all_results);
            }

            let mean_acc = accs.iter().sum::<f64>() / accs.len() as f64;
            let std_acc = (accs.iter().map(|a| (a - mean_acc).powi(2)).sum::<f64>() / accs.len() as f64).sqrt();
            let mean_loss = losses.iter().sum::<f64>() / losses.len() as f64;

            log_msg(&mut logf, &format!("  >>> {} : {:.2}% +/- {:.2}% (loss={:.4})",
                label, mean_acc * 100.0, std_acc * 100.0, mean_loss));
            log_msg(&mut logf, "");

            config_results.push(ConfigResult { label, accs, losses });
        }
    }

    // ============================================================
    // Final summary
    // ============================================================
    log_msg(&mut logf, "");
    log_msg(&mut logf, "================================================================");
    log_msg(&mut logf, "                    FINAL RESULTS");
    log_msg(&mut logf, "================================================================");
    log_msg(&mut logf, &format!("  {:>20} {:>12} {:>10} {:>10}",
        "Configuration", "Accuracy", "StdDev", "Loss"));
    log_msg(&mut logf, &format!("  {}", "=".repeat(55)));

    for cr in &config_results {
        let mean_acc = cr.accs.iter().sum::<f64>() / cr.accs.len() as f64;
        let std_acc = (cr.accs.iter().map(|a| (a - mean_acc).powi(2)).sum::<f64>() / cr.accs.len() as f64).sqrt();
        let mean_loss = cr.losses.iter().sum::<f64>() / cr.losses.len() as f64;
        log_msg(&mut logf, &format!("  {:>20} {:>10.2}% {:>8.2}% {:>10.4}",
            cr.label, mean_acc * 100.0, std_acc * 100.0, mean_loss));
    }

    // Head-to-head comparisons per hidden size
    log_msg(&mut logf, "");
    log_msg(&mut logf, "  HEAD-TO-HEAD COMPARISONS:");
    for &(h1, h2) in &hidden_configs {
        log_msg(&mut logf, &format!("  --- Hidden {},{} ---", h1, h2));
        let c19_cr = config_results.iter().find(|c| c.label.contains("C19") && c.label.contains(&format!("h={},{}", h1, h2)));
        let relu_cr = config_results.iter().find(|c| c.label.contains("ReLU") && c.label.contains(&format!("h={},{}", h1, h2)));
        let gelu_cr = config_results.iter().find(|c| c.label.contains("GELU") && c.label.contains(&format!("h={},{}", h1, h2)));

        if let (Some(c19), Some(relu), Some(gelu)) = (c19_cr, relu_cr, gelu_cr) {
            let c19_mean = c19.accs.iter().sum::<f64>() / c19.accs.len() as f64;
            let c19_std = (c19.accs.iter().map(|a| (a - c19_mean).powi(2)).sum::<f64>() / c19.accs.len() as f64).sqrt();
            let relu_mean = relu.accs.iter().sum::<f64>() / relu.accs.len() as f64;
            let relu_std = (relu.accs.iter().map(|a| (a - relu_mean).powi(2)).sum::<f64>() / relu.accs.len() as f64).sqrt();
            let gelu_mean = gelu.accs.iter().sum::<f64>() / gelu.accs.len() as f64;
            let gelu_std = (gelu.accs.iter().map(|a| (a - gelu_mean).powi(2)).sum::<f64>() / gelu.accs.len() as f64).sqrt();

            log_msg(&mut logf, &format!("    C19 rho=8: {:.2}% +/- {:.2}% | ReLU: {:.2}% +/- {:.2}% | GELU: {:.2}% +/- {:.2}%",
                c19_mean * 100.0, c19_std * 100.0,
                relu_mean * 100.0, relu_std * 100.0,
                gelu_mean * 100.0, gelu_std * 100.0));

            let delta_relu = (c19_mean - relu_mean) * 100.0;
            let delta_gelu = (c19_mean - gelu_mean) * 100.0;
            log_msg(&mut logf, &format!("    C19 vs ReLU: {:+.2}pp | C19 vs GELU: {:+.2}pp",
                delta_relu, delta_gelu));
        }
    }

    // Verdict
    log_msg(&mut logf, "");
    log_msg(&mut logf, "================================================================");
    log_msg(&mut logf, "  VERDICT");
    log_msg(&mut logf, "================================================================");

    // Compute overall C19 vs ReLU delta
    let c19_all: Vec<f64> = config_results.iter()
        .filter(|c| c.label.contains("C19"))
        .flat_map(|c| c.accs.clone())
        .collect();
    let relu_all: Vec<f64> = config_results.iter()
        .filter(|c| c.label.contains("ReLU"))
        .flat_map(|c| c.accs.clone())
        .collect();

    if !c19_all.is_empty() && !relu_all.is_empty() {
        let c19_avg = c19_all.iter().sum::<f64>() / c19_all.len() as f64;
        let relu_avg = relu_all.iter().sum::<f64>() / relu_all.len() as f64;
        let delta = (c19_avg - relu_avg) * 100.0;

        if delta > 1.0 {
            log_msg(&mut logf, &format!("  C19 advantage CONFIRMED: {:+.2}pp over ReLU", delta));
            log_msg(&mut logf, "  ==> C19 is a real improvement, not a toy-task artifact");
        } else if delta > 0.0 {
            log_msg(&mut logf, &format!("  C19 shows marginal advantage: {:+.2}pp over ReLU", delta));
            log_msg(&mut logf, "  ==> Effect is small; needs more investigation");
        } else {
            log_msg(&mut logf, &format!("  C19 shows NO advantage: {:+.2}pp vs ReLU", delta));
            log_msg(&mut logf, "  ==> The +2.2% was likely a toy-task artifact");
        }
    }

    log_msg(&mut logf, &format!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64()));
    log_msg(&mut logf, "================================================================");
    log_msg(&mut logf, "=== BENCHMARK COMPLETE ===");
}
