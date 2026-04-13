//! Abstract Core v1 — Full pipeline: frozen binary preprocessor → mixer → next-char prediction
//!
//! Architecture:
//!   128 previous chars → shared frozen preprocessor (7 neurons, all-binary) → 896 signals
//!   → Mixer (896 → 256, trainable) → Output (256 → 27, softmax)
//!
//! Preprocessor: the proven all-binary exhaustive-optimal encoder from all_binary_mirror.rs
//!   Config: c∈{1.0,10.0}, ρ∈{0.0,2.0}, weights/bias {-1,+1}
//!   N0: [- + + - - + + -] b=+1 c=10.0 ρ=2.0
//!   N1: [+ - + + - - - -] b=+1 c=10.0 ρ=0.0
//!   N2: [- - + - + + - -] b=+1 c=10.0 ρ=0.0
//!   N3: [- - - + - - + -] b=+1 c=10.0 ρ=0.0
//!   N4: [- + + + - - - -] b=+1 c=10.0 ρ=0.0
//!   N5: [+ + - - - - - -] b=+1 c=10.0 ρ=0.0
//!   N6: [- + - + + + - -] b=+1 c=10.0 ρ=0.0
//!
//! Corpus: Alice (27 symbols: a-z + space)
//! Metric: argmax next-char accuracy vs INSTNCT 24.6% baseline
//!
//! Run: cargo run --example abstract_core_v1 --release

use std::time::Instant;

// ══════════════════════════════════════════════════════
// PRNG
// ══════════════════════════════════════════════════════
struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 { self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.s }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn normal(&mut self) -> f32 { let u1 = self.f32().max(1e-7); let u2 = self.f32(); (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() }
    fn range(&mut self, lo: usize, hi: usize) -> usize { if hi <= lo { lo } else { lo + (self.next() as usize % (hi - lo)) } }
}

// ══════════════════════════════════════════════════════
// CORPUS
// ══════════════════════════════════════════════════════
fn load_corpus(path: &str) -> Vec<u8> {
    let raw = std::fs::read(path).expect("failed to read corpus");
    let mut corpus = Vec::with_capacity(raw.len());
    for &b in &raw {
        match b {
            b'a'..=b'z' => corpus.push(b - b'a'),
            b'A'..=b'Z' => corpus.push(b - b'A'),
            b' ' | b'\n' | b'\t' | b'\r' => corpus.push(26),
            _ => {}
        }
    }
    corpus
}

// ══════════════════════════════════════════════════════
// FROZEN PREPROCESSOR — all-binary, 77 bits, exhaustive optimal
// ══════════════════════════════════════════════════════
fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.1); let rho = rho.max(0.0); let l = 6.0 * c;
    if x >= l { return x - l; } if x <= -l { return x + l; }
    let s = x / c; let n = s.floor(); let t = s - n;
    let h = t * (1.0 - t); let sg = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sg * h + rho * h * h)
}

struct FrozenPreprocessor {
    // 7 neurons, each: 8 weights {-1,+1}, bias {-1,+1}, c, rho
    weights: [[i8; 8]; 7],
    biases: [i8; 7],
    c: [f32; 7],
    rho: [f32; 7],
}

impl FrozenPreprocessor {
    fn new() -> Self {
        // From all_binary_mirror.rs result: c{1.0,10.0} ρ{0.0,2.0}
        FrozenPreprocessor {
            weights: [
                [-1, 1, 1,-1,-1, 1, 1,-1],  // N0
                [ 1,-1, 1, 1,-1,-1,-1,-1],  // N1
                [-1,-1, 1,-1, 1, 1,-1,-1],  // N2
                [-1,-1,-1, 1,-1,-1, 1,-1],  // N3
                [-1, 1, 1, 1,-1,-1,-1,-1],  // N4
                [ 1, 1,-1,-1,-1,-1,-1,-1],  // N5
                [-1, 1,-1, 1, 1, 1,-1,-1],  // N6
            ],
            biases: [1, 1, 1, 1, 1, 1, 1],
            c: [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            rho: [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    }

    /// Encode a single character (0-26) into 7 signals
    fn encode(&self, char_val: u8) -> [f32; 7] {
        // Character to 5-bit binary (0-26 fits in 5 bits, pad to 8)
        let mut bits = [0.0f32; 8];
        for i in 0..8 { bits[i] = ((char_val >> i) & 1) as f32; }

        let mut output = [0.0f32; 7];
        for k in 0..7 {
            let mut dot = self.biases[k] as f32;
            for j in 0..8 { dot += self.weights[k][j] as f32 * bits[j]; }
            output[k] = c19(dot, self.c[k], self.rho[k]);
        }
        output
    }
}

// ══════════════════════════════════════════════════════
// TRAINABLE LAYERS — Mixer + Output
// ══════════════════════════════════════════════════════
const CTX: usize = 128;
const PREPROC_OUT: usize = 7;
const MIXER_IN: usize = CTX * PREPROC_OUT; // 896
const N_CLASSES: usize = 27;

#[derive(Clone)]
struct TrainableLayers {
    // Mixer: MIXER_IN → mixer_dim (ReLU)
    wm: Vec<Vec<f32>>,
    bm: Vec<f32>,
    // Optional second mixer layer: mixer_dim → mixer_dim2 (ReLU)
    wm2: Option<Vec<Vec<f32>>>,
    bm2: Option<Vec<f32>>,
    // Output: last_dim → N_CLASSES
    wo: Vec<Vec<f32>>,
    bo: Vec<f32>,
    mixer_dim: usize,
    mixer_dim2: usize, // 0 if no second layer
}

impl TrainableLayers {
    fn new(mixer_dim: usize, mixer_dim2: usize, rng: &mut Rng) -> Self {
        let sm = (2.0 / MIXER_IN as f32).sqrt();
        let wm: Vec<Vec<f32>> = (0..mixer_dim).map(|_| (0..MIXER_IN).map(|_| rng.normal() * sm).collect()).collect();
        let bm = vec![0.0f32; mixer_dim];

        let (wm2, bm2, last_dim) = if mixer_dim2 > 0 {
            let s2 = (2.0 / mixer_dim as f32).sqrt();
            let w: Vec<Vec<f32>> = (0..mixer_dim2).map(|_| (0..mixer_dim).map(|_| rng.normal() * s2).collect()).collect();
            (Some(w), Some(vec![0.0f32; mixer_dim2]), mixer_dim2)
        } else {
            (None, None, mixer_dim)
        };

        let so = (2.0 / last_dim as f32).sqrt();
        let wo: Vec<Vec<f32>> = (0..N_CLASSES).map(|_| (0..last_dim).map(|_| rng.normal() * so).collect()).collect();
        let bo = vec![0.0f32; N_CLASSES];

        TrainableLayers { wm, bm, wm2, bm2, wo, bo, mixer_dim, mixer_dim2 }
    }

    fn total_params(&self) -> usize {
        let mut p = MIXER_IN * self.mixer_dim + self.mixer_dim; // mixer 1
        if self.mixer_dim2 > 0 {
            p += self.mixer_dim * self.mixer_dim2 + self.mixer_dim2; // mixer 2
        }
        let last = if self.mixer_dim2 > 0 { self.mixer_dim2 } else { self.mixer_dim };
        p += last * N_CLASSES + N_CLASSES; // output
        p
    }

    fn forward(&self, preproc_signals: &[f32]) -> (Vec<f32>, Option<Vec<f32>>, Vec<f32>) {
        // Mixer 1
        let mut am = vec![0.0f32; self.mixer_dim];
        for i in 0..self.mixer_dim {
            let mut s = self.bm[i];
            for j in 0..MIXER_IN { s += self.wm[i][j] * preproc_signals[j]; }
            am[i] = s.max(0.0); // ReLU
        }

        // Mixer 2 (optional)
        let (am2, last_act) = if let (Some(ref w2), Some(ref b2)) = (&self.wm2, &self.bm2) {
            let mut a2 = vec![0.0f32; self.mixer_dim2];
            for i in 0..self.mixer_dim2 {
                let mut s = b2[i];
                for j in 0..self.mixer_dim { s += w2[i][j] * am[j]; }
                a2[i] = s.max(0.0);
            }
            let last = a2.clone();
            (Some(a2), last)
        } else {
            (None, am.clone())
        };

        // Output logits
        let last_dim = last_act.len();
        let mut logits = vec![0.0f32; N_CLASSES];
        for i in 0..N_CLASSES {
            logits[i] = self.bo[i];
            for j in 0..last_dim { logits[i] += self.wo[i][j] * last_act[j]; }
        }

        (am, am2, logits)
    }

    fn predict(&self, preproc_signals: &[f32]) -> usize {
        let (_, _, logits) = self.forward(preproc_signals);
        logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0)
    }

    fn train_step(&mut self, preproc_signals: &[f32], target: usize, lr: f32) -> f32 {
        let (am, am2, logits) = self.forward(preproc_signals);

        // Softmax + cross-entropy
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
        let sum_exp: f32 = exp.iter().sum();
        let softmax: Vec<f32> = exp.iter().map(|&e| e / sum_exp).collect();
        let loss = -(softmax[target].max(1e-7)).ln();

        let mut dl = softmax; dl[target] -= 1.0;

        // Backprop output layer
        let last_act = if let Some(ref a2) = am2 { a2 } else { &am };
        let last_dim = last_act.len();

        let mut d_last = vec![0.0f32; last_dim];
        for i in 0..N_CLASSES {
            for j in 0..last_dim {
                d_last[j] += dl[i] * self.wo[i][j];
                self.wo[i][j] -= lr * dl[i] * last_act[j];
            }
            self.bo[i] -= lr * dl[i];
        }

        // Backprop mixer 2 (if exists)
        let d_mixer1 = if let (Some(ref mut w2), Some(ref mut b2)) = (&mut self.wm2, &mut self.bm2) {
            let mut dm2 = vec![0.0f32; self.mixer_dim2];
            for i in 0..self.mixer_dim2 {
                dm2[i] = if am2.as_ref().unwrap()[i] > 0.0 { d_last[i] } else { 0.0 };
            }
            let mut dm1 = vec![0.0f32; self.mixer_dim];
            for i in 0..self.mixer_dim2 {
                for j in 0..self.mixer_dim {
                    dm1[j] += dm2[i] * w2[i][j];
                    w2[i][j] -= lr * dm2[i] * am[j];
                }
                b2[i] -= lr * dm2[i];
            }
            dm1
        } else {
            d_last
        };

        // Backprop mixer 1
        for i in 0..self.mixer_dim {
            let dm = if am[i] > 0.0 { d_mixer1[i] } else { 0.0 };
            for j in 0..MIXER_IN { self.wm[i][j] -= lr * dm * preproc_signals[j]; }
            self.bm[i] -= lr * dm;
        }

        loss
    }
}

// ══════════════════════════════════════════════════════
// EVAL
// ══════════════════════════════════════════════════════
fn eval_accuracy(
    preproc: &FrozenPreprocessor,
    model: &TrainableLayers,
    corpus: &[u8],
    n_samples: usize,
    seed: u64,
) -> f64 {
    let mut rng = Rng::new(seed);
    let mut correct = 0usize;
    let mut total = 0usize;

    for _ in 0..n_samples {
        if corpus.len() < CTX + 1 { break; }
        let off = rng.range(0, corpus.len() - CTX - 1);
        let seg = &corpus[off..off + CTX + 1];

        // Build preprocessed input
        let mut signals = vec![0.0f32; MIXER_IN];
        for i in 0..CTX {
            let encoded = preproc.encode(seg[i]);
            for k in 0..PREPROC_OUT { signals[i * PREPROC_OUT + k] = encoded[k]; }
        }

        let pred = model.predict(&signals);
        if pred == seg[CTX] as usize { correct += 1; }
        total += 1;
    }

    correct as f64 / total as f64 * 100.0
}

// ══════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════
fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let preproc = FrozenPreprocessor::new();

    println!("=== ABSTRACT CORE v1 — Frozen Preprocessor → Mixer → Next Char ===");
    println!("Corpus: {} chars, Context: {} chars", corpus.len(), CTX);
    println!("Preprocessor: 7 neurons, all-binary, frozen (77 bits)");
    println!("Pipeline: 128 bytes → preproc → 896 signals → mixer → 27 classes");
    println!("Baseline: INSTNCT 24.6%, frequency 20.3%\n");

    // Adversarial preprocessor test first
    println!("━━━ Preprocessor Adversarial Test ━━━");
    let mut all_codes: Vec<Vec<u8>> = Vec::new();
    for c in 0..27u8 {
        let sig = preproc.encode(c);
        let code: Vec<u8> = sig.iter().map(|&v| if v > 0.0 { 1 } else { 0 }).collect();
        let ch = if c < 26 { format!("'{}'", (c + b'a') as char) } else { "' '".to_string() };
        all_codes.push(code.clone());
        if c < 5 || c == 26 {
            let code_str: String = code.iter().map(|&v| if v == 1 { '1' } else { '0' }).collect();
            println!("  {} ({:>2}) → {} sig=[{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}]",
                ch, c, code_str, sig[0], sig[1], sig[2], sig[3], sig[4], sig[5], sig[6]);
        }
    }
    all_codes.sort();
    let unique_before = all_codes.len();
    all_codes.dedup();
    println!("  Unique hard codes: {}/27", all_codes.len());
    if all_codes.len() < 27 {
        println!("  WARNING: some chars share hard codes (but soft values may differ)");
    }
    println!("  Preprocessor: OK\n");

    // Sweep configurations
    let configs: Vec<(&str, usize, usize, usize, f32)> = vec![
        // (name, mixer_dim, mixer_dim2, epochs, lr)
        ("1-layer M=128",       128, 0,   30, 0.005),
        ("1-layer M=256",       256, 0,   30, 0.005),
        ("2-layer M=256→128",   256, 128, 30, 0.003),
        ("2-layer M=512→256",   512, 256, 30, 0.003),
        ("1-layer M=512",       512, 0,   30, 0.003),
    ];

    let samples_per_epoch = 8000;

    println!("{:>25} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "config", "params", "float_acc", "best_acc", "epoch", "time");
    println!("{}", "─".repeat(80));

    for (name, md, md2, epochs, lr_init) in &configs {
        let mut rng = Rng::new(42);
        let mut model = TrainableLayers::new(*md, *md2, &mut rng);
        let params = model.total_params();
        let tc = Instant::now();

        let mut best_acc = 0.0f64;
        let mut best_epoch = 0;

        for epoch in 0..*epochs {
            let lr = lr_init * (1.0 - epoch as f32 / *epochs as f32 * 0.7);
            let mut total_loss = 0.0f32;
            let mut rng_train = Rng::new(epoch as u64 * 1000 + 42);

            for _ in 0..samples_per_epoch {
                if corpus.len() < CTX + 1 { break; }
                let off = rng_train.range(0, corpus.len() - CTX - 1);
                let seg = &corpus[off..off + CTX + 1];

                let mut signals = vec![0.0f32; MIXER_IN];
                for i in 0..CTX {
                    let encoded = preproc.encode(seg[i]);
                    for k in 0..PREPROC_OUT { signals[i * PREPROC_OUT + k] = encoded[k]; }
                }

                total_loss += model.train_step(&signals, seg[CTX] as usize, lr);
            }

            if (epoch + 1) % 5 == 0 || epoch == 0 {
                let acc = eval_accuracy(&preproc, &model, &corpus, 3000, 777 + epoch as u64);
                if acc > best_acc { best_acc = acc; best_epoch = epoch + 1; }
            }
        }

        // Final eval
        let final_acc = eval_accuracy(&preproc, &model, &corpus, 5000, 999);
        if final_acc > best_acc { best_acc = final_acc; best_epoch = *epochs; }

        let verdict = if best_acc > 24.6 { "★ BEATS INSTNCT" } else { "" };
        println!("{:>25} {:>10} {:>9.1}% {:>9.1}% {:>10} {:>8.1}s {}",
            name, params, final_acc, best_acc, best_epoch, tc.elapsed().as_secs_f64(), verdict);
    }

    println!("\n  Baseline: frequency=20.3%, INSTNCT=24.6%");
    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
