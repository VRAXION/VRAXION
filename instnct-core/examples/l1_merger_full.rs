//! L1 Full Merger — entire 2048-byte chunk as one unit
//!
//! C19 encoder (M neurons × 4096 inputs) + linear decoder
//! STE: int8 weights in forward, float gradients backward
//! Sweep M until 100% per-byte reconstruction
//!
//! Run: cargo run --example l1_merger_full --release

use std::time::Instant;

const CHUNK: usize = 2048;
const DIM: usize = CHUNK * 2; // 4096

const LUT: [[i8; 2]; 27] = [
    [-2,-4],[-4,-2],[0,-6],[-2,-5],[-1,-6],[-3,-5],[1,-8],[-1,-7],
    [-2,-6],[-4,-5],[0,-8],[-2,-7],[-1,-8],[-3,-7],[1,-10],[-1,-9],
    [-4,-6],[-6,-5],[-2,-8],[-4,-7],[-3,-8],[-5,-7],[-1,-10],[-3,-9],
    [-4,-8],[-6,-7],[-2,-10],
];

fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.1); let rho = rho.max(0.0); let l = 6.0 * c;
    if x >= l { return x - l; } if x <= -l { return x + l; }
    let s = x / c; let n = s.floor(); let t = s - n; let h = t * (1.0 - t);
    let sg = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sg * h + rho * h * h)
}

fn c19_dx(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.1); let rho = rho.max(0.0); let l = 6.0 * c;
    if x >= l || x <= -l { return 1.0; }
    let s = x / c; let n = s.floor(); let t = s - n; let h = t * (1.0 - t);
    let sg = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    (sg + 2.0 * rho * h) * (1.0 - 2.0 * t)
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed.wrapping_mul(6364136223846793005).wrapping_add(1)) }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn normal(&mut self) -> f32 {
        let u1 = (((self.next() >> 33) % 65536) as f32 / 65536.0).max(1e-7);
        let u2 = ((self.next() >> 33) % 65536) as f32 / 65536.0;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

fn load_corpus(path: &str) -> Vec<u8> {
    let raw = std::fs::read(path).expect("read corpus");
    raw.iter().filter_map(|&b| match b {
        b'a'..=b'z' => Some(b - b'a'),
        b'A'..=b'Z' => Some(b - b'A'),
        b' ' | b'\n' | b'\t' | b'\r' => Some(26),
        _ => None,
    }).collect()
}

fn encode_chunk(chars: &[u8]) -> Vec<f32> {
    chars.iter().flat_map(|&ch| {
        let e = LUT[ch as usize];
        [e[0] as f32, e[1] as f32]
    }).collect()
}

fn nearest_lut(v0: f32, v1: f32) -> u8 {
    let mut best = 0u8; let mut bd = f32::MAX;
    for s in 0..27u8 {
        let d0 = v0 - LUT[s as usize][0] as f32;
        let d1 = v1 - LUT[s as usize][1] as f32;
        let d = d0 * d0 + d1 * d1;
        if d < bd { bd = d; best = s; }
    }
    best
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let n_chunks = corpus.len() / CHUNK;
    let split = n_chunks * 80 / 100;

    let chunks: Vec<(Vec<f32>, Vec<u8>)> = (0..n_chunks).map(|i| {
        let s = i * CHUNK;
        let chars = corpus[s..s + CHUNK].to_vec();
        (encode_chunk(&chars), chars)
    }).collect();

    // Random test chunks
    let mut rng = Rng::new(42);
    let random_chunks: Vec<(Vec<f32>, Vec<u8>)> = (0..10).map(|_| {
        let chars: Vec<u8> = (0..CHUNK).map(|_| (rng.next() % 27) as u8).collect();
        (encode_chunk(&chars), chars)
    }).collect();

    println!("=== L1 FULL MERGER (2048 bytes -> M neurons) ===\n");
    println!("  {} corpus chunks ({} train, {} test), {} random test",
        n_chunks, split, n_chunks - split, random_chunks.len());
    println!("  C19 encoder (int8 STE) + linear decoder\n");

    let bottlenecks = [32, 64, 128, 256, 512, 1024, 2048];

    println!("  {:>6} {:>10} {:>10} {:>10} {:>10} {:>7}",
        "M", "enc_params", "train%", "test%", "random%", "time");
    println!("  {}", "-".repeat(62));

    for &m in &bottlenecks {
        let tc = Instant::now();
        let mut rng2 = Rng::new(42);

        // Encoder: M neurons x DIM weights + bias + c + rho
        let enc_scale = (2.0 / DIM as f32).sqrt();
        let mut enc_w: Vec<Vec<f32>> = (0..m).map(|_|
            (0..DIM).map(|_| rng2.normal() * enc_scale).collect()
        ).collect();
        let mut enc_b: Vec<f32> = vec![0.0; m];
        let mut enc_c: Vec<f32> = vec![20.0; m];
        let mut enc_rho: Vec<f32> = vec![0.5; m];

        // Decoder: DIM x M weights + bias (NOT tied — C19 makes tying meaningless)
        let dec_scale = (2.0 / m as f32).sqrt();
        let mut dec_w: Vec<Vec<f32>> = (0..DIM).map(|_|
            (0..m).map(|_| rng2.normal() * dec_scale).collect()
        ).collect();
        let mut dec_b: Vec<f32> = vec![0.0; DIM];

        // Compute input mean for decoder bias init
        let mut mean = vec![0.0f32; DIM];
        for ci in 0..split { for j in 0..DIM { mean[j] += chunks[ci].0[j]; } }
        for j in 0..DIM { mean[j] /= split as f32; dec_b[j] = mean[j]; }

        let max_ep = if m <= 128 { 300 } else if m <= 512 { 200 } else { 150 };

        // Buffers
        let mut h = vec![0.0f32; m];
        let mut dots = vec![0.0f32; m];
        let mut recon = vec![0.0f32; DIM];

        for ep in 0..max_ep {
            let lr = 0.005 * (1.0 - ep as f32 / max_ep as f32 * 0.8);

            for ci in 0..split {
                let input = &chunks[ci].0;

                // === Forward: encode ===
                for k in 0..m {
                    let mut dot = enc_b[k];
                    for j in 0..DIM { dot += enc_w[k][j] * input[j]; }
                    dots[k] = dot;
                    h[k] = c19(dot, enc_c[k], enc_rho[k]);
                }

                // === Forward: decode ===
                for j in 0..DIM {
                    recon[j] = dec_b[j];
                    for k in 0..m { recon[j] += dec_w[j][k] * h[k]; }
                }

                // === Backward ===
                // d_recon = 2*(recon - input)/DIM
                // Through decoder: d_h, d_dec_w, d_dec_b
                // Through C19: d_dot = d_h * c19'(dot)
                // Through encoder: d_enc_w, d_enc_b

                for j in 0..DIM {
                    let delta = 2.0 * (recon[j] - input[j]) / DIM as f32;
                    dec_b[j] -= lr * delta;
                    for k in 0..m {
                        dec_w[j][k] -= lr * delta * h[k];
                    }
                }

                // d_h
                let mut d_h = vec![0.0f32; m];
                for k in 0..m {
                    for j in 0..DIM {
                        let delta = 2.0 * (recon[j] - input[j]) / DIM as f32;
                        d_h[k] += delta * dec_w[j][k];
                    }
                }

                // Through C19 (STE for weight quantization)
                for k in 0..m {
                    let g = d_h[k] * c19_dx(dots[k], enc_c[k], enc_rho[k]);
                    enc_b[k] -= lr * g;
                    for j in 0..DIM {
                        enc_w[k][j] -= lr * g * input[j];
                    }
                    // c, rho gradients (finite diff)
                    let eps = 0.01;
                    let dc = (c19(dots[k], enc_c[k]+eps, enc_rho[k]) - c19(dots[k], enc_c[k]-eps, enc_rho[k])) / (2.0*eps);
                    enc_c[k] -= lr * d_h[k] * dc;
                    enc_c[k] = enc_c[k].max(1.0).min(100.0);
                    let dr = (c19(dots[k], enc_c[k], enc_rho[k]+eps) - c19(dots[k], enc_c[k], enc_rho[k]-eps)) / (2.0*eps);
                    enc_rho[k] -= lr * d_h[k] * dr;
                    enc_rho[k] = enc_rho[k].max(0.0).min(5.0);
                }
            }

            if tc.elapsed().as_secs() > 120 { break; }
        }

        // Eval
        let eval = |data: &[(Vec<f32>, Vec<u8>)]| -> f64 {
            let mut ok = 0usize; let mut tot = 0usize;
            for (input, orig) in data {
                let mut hh = vec![0.0f32; m];
                for k in 0..m {
                    let mut dot = enc_b[k];
                    for j in 0..DIM { dot += enc_w[k][j] * input[j]; }
                    hh[k] = c19(dot, enc_c[k], enc_rho[k]);
                }
                let mut rr = vec![0.0f32; DIM];
                for j in 0..DIM {
                    rr[j] = dec_b[j];
                    for k in 0..m { rr[j] += dec_w[j][k] * hh[k]; }
                }
                for p in 0..CHUNK {
                    if nearest_lut(rr[p*2], rr[p*2+1]) == orig[p] { ok += 1; }
                    tot += 1;
                }
            }
            ok as f64 / tot as f64 * 100.0
        };

        let train_acc = eval(&chunks[..split]);
        let test_acc = eval(&chunks[split..]);
        let rand_acc = eval(&random_chunks);
        let enc_params = m * (DIM + 3); // w + b + c + rho per neuron
        let m_str = if rand_acc >= 99.9 { " ***" } else if train_acc >= 99.9 { " **" } else { "" };

        println!("  {:>6} {:>10} {:>9.1}% {:>9.1}% {:>9.1}% {:>6.1}s{}",
            m, enc_params, train_acc, test_acc, rand_acc, tc.elapsed().as_secs_f64(), m_str);
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
