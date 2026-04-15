//! L1 Byte Merger Sweep — minimum bottleneck for 100% round-trip
//!
//! Same methodology as L0 byte encoder: sweep output width from small,
//! find minimum N where 100% per-byte reconstruction is achieved.
//!
//! Input: 2048 bytes × LUT_2N → 4096 int8 values
//! Architecture: linear tied autoencoder (4096 → N → 4096)
//! Metric: per-byte accuracy (decode → nearest LUT entry → correct char?)
//! Method: SGD + momentum (full-batch), linear tied weights W/W^T
//!
//! Run: cargo run --example l1_merger_sweep --release

use std::time::Instant;

const CHUNK: usize = 2048;
const DIM: usize = CHUNK * 2; // 4096

const LUT: [[i8; 2]; 27] = [
    [  -2,  -4], [  -4,  -2], [   0,  -6], [  -2,  -5], [  -1,  -6],
    [  -3,  -5], [   1,  -8], [  -1,  -7], [  -2,  -6], [  -4,  -5],
    [   0,  -8], [  -2,  -7], [  -1,  -8], [  -3,  -7], [   1, -10],
    [  -1,  -9], [  -4,  -6], [  -6,  -5], [  -2,  -8], [  -4,  -7],
    [  -3,  -8], [  -5,  -7], [  -1, -10], [  -3,  -9], [  -4,  -8],
    [  -6,  -7], [  -2, -10],
];

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
    let mut best = 0u8;
    let mut bd = f32::MAX;
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

    // Compute minimum inter-LUT distance
    let mut min_d = f32::MAX;
    for i in 0..27 {
        for j in (i+1)..27 {
            let d0 = (LUT[i][0] - LUT[j][0]) as f32;
            let d1 = (LUT[i][1] - LUT[j][1]) as f32;
            let d = (d0 * d0 + d1 * d1).sqrt();
            if d < min_d { min_d = d; }
        }
    }

    // Chunk corpus
    let n_chunks = corpus.len() / CHUNK;
    let split = n_chunks * 80 / 100;
    let chunks: Vec<(Vec<f32>, Vec<u8>)> = (0..n_chunks)
        .map(|i| {
            let s = i * CHUNK;
            let chars = corpus[s..s + CHUNK].to_vec();
            let enc = encode_chunk(&chars);
            (enc, chars)
        }).collect();

    // Random synthetic test chunks
    let mut rng = Rng::new(42);
    let random_chunks: Vec<(Vec<f32>, Vec<u8>)> = (0..10)
        .map(|_| {
            let chars: Vec<u8> = (0..CHUNK).map(|_| (rng.next() % 27) as u8).collect();
            let enc = encode_chunk(&chars);
            (enc, chars)
        }).collect();

    println!("=== L1 BYTE MERGER SWEEP ===\n");
    println!("  Input: {} bytes x LUT_2N = {} int8 values", CHUNK, DIM);
    println!("  Corpus: {} chars, {} chunks ({} train, {} test)",
        corpus.len(), n_chunks, split, n_chunks - split);
    println!("  Min LUT distance: {:.2} (max recon error for 100%: {:.2})", min_d, min_d / 2.0);
    println!("  Info minimum: {} values (log2(27^{}) / 8)", (CHUNK as f64 * 4.75 / 8.0).ceil() as usize, CHUNK);
    println!();

    let bottlenecks = [64, 128, 256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3584];

    println!("  {:>6} {:>10} {:>10} {:>10} {:>10} {:>7} {:>7}",
        "N", "compress", "train%", "test%", "random%", "loss", "time");
    println!("  {}", "-".repeat(70));

    for &bn in &bottlenecks {
        let tc = Instant::now();
        let wlen = bn * DIM;

        // Init weights
        let mut rng2 = Rng::new(42);
        let scale = (2.0 / DIM as f32).sqrt();
        let mut w: Vec<f32> = (0..wlen).map(|_| rng2.normal() * scale).collect();
        let mut enc_b = vec![0.0f32; bn];
        let mut dec_b = vec![0.0f32; DIM];

        // Momentum
        let mut mw = vec![0.0f32; wlen];
        let mut me = vec![0.0f32; bn];
        let mut md = vec![0.0f32; DIM];

        // Pre-alloc work buffers
        let mut h = vec![0.0f32; bn];
        let mut o = vec![0.0f32; DIM];
        let mut delta = vec![0.0f32; DIM];
        let mut dh = vec![0.0f32; bn];
        let mut dw = vec![0.0f32; wlen];
        let mut de = vec![0.0f32; bn];
        let mut dd = vec![0.0f32; DIM];

        let max_ep = if bn <= 256 { 600 } else if bn <= 1024 { 400 } else { 250 };
        let mut last_loss = f32::MAX;

        for ep in 0..max_ep {
            let lr = 0.005 * (1.0 - ep as f32 / max_ep as f32 * 0.9);

            // Zero grads
            for v in dw.iter_mut() { *v = 0.0; }
            for v in de.iter_mut() { *v = 0.0; }
            for v in dd.iter_mut() { *v = 0.0; }
            let mut total_loss = 0.0f32;

            for ci in 0..split {
                let input = &chunks[ci].0;

                // Encode: h = W * input + enc_b
                for k in 0..bn {
                    h[k] = enc_b[k];
                    let row = k * DIM;
                    let mut s = 0.0f32;
                    for j in 0..DIM { s += w[row + j] * input[j]; }
                    h[k] += s;
                }

                // Decode: o = W^T * h + dec_b
                for j in 0..DIM {
                    o[j] = dec_b[j];
                    let mut s = 0.0f32;
                    for k in 0..bn { s += w[k * DIM + j] * h[k]; }
                    o[j] += s;
                }

                // Loss + delta
                for j in 0..DIM {
                    let diff = o[j] - input[j];
                    total_loss += diff * diff;
                    delta[j] = 2.0 * diff / DIM as f32;
                }

                // Backward: dh and dw (decoder path)
                for k in 0..bn { dh[k] = 0.0; }
                for j in 0..DIM {
                    let dj = delta[j];
                    for k in 0..bn {
                        dh[k] += dj * w[k * DIM + j];
                        dw[k * DIM + j] += dj * h[k];
                    }
                    dd[j] += dj;
                }

                // Backward: dw (encoder path)
                for k in 0..bn {
                    let dhk = dh[k];
                    let row = k * DIM;
                    for j in 0..DIM {
                        dw[row + j] += dhk * input[j];
                    }
                    de[k] += dhk;
                }
            }

            // Update with momentum
            let inv = 1.0 / split as f32;
            let mom = 0.9f32;
            for i in 0..wlen { dw[i] *= inv; mw[i] = mom * mw[i] + dw[i]; w[i] -= lr * mw[i]; }
            for k in 0..bn { de[k] *= inv; me[k] = mom * me[k] + de[k]; enc_b[k] -= lr * me[k]; }
            for j in 0..DIM { dd[j] *= inv; md[j] = mom * md[j] + dd[j]; dec_b[j] -= lr * md[j]; }

            last_loss = total_loss / (split * DIM) as f32;

            // Time limit
            if tc.elapsed().as_secs() > 180 { break; }
        }

        // Evaluate
        let eval = |data: &[(Vec<f32>, Vec<u8>)]| -> f64 {
            let mut ok = 0usize;
            let mut tot = 0usize;
            for (input, orig) in data {
                // Encode
                let mut hh = vec![0.0f32; bn];
                for k in 0..bn {
                    hh[k] = enc_b[k];
                    let row = k * DIM;
                    let mut s = 0.0f32;
                    for j in 0..DIM { s += w[row + j] * input[j]; }
                    hh[k] += s;
                }
                // Decode
                let mut oo = vec![0.0f32; DIM];
                for j in 0..DIM {
                    oo[j] = dec_b[j];
                    let mut s = 0.0f32;
                    for k in 0..bn { s += w[k * DIM + j] * hh[k]; }
                    oo[j] += s;
                }
                // Round-trip check
                for p in 0..CHUNK {
                    if nearest_lut(oo[p * 2], oo[p * 2 + 1]) == orig[p] { ok += 1; }
                    tot += 1;
                }
            }
            ok as f64 / tot as f64 * 100.0
        };

        let train_acc = eval(&chunks[..split]);
        let test_acc = eval(&chunks[split..]);
        let rand_acc = eval(&random_chunks);
        let compress = DIM as f64 / bn as f64;
        let params = wlen + bn + DIM;
        let m = if rand_acc >= 100.0 { " ***" } else if train_acc >= 100.0 && test_acc >= 100.0 { " **" } else { "" };

        println!("  {:>6} {:>9.1}x {:>9.1}% {:>9.1}% {:>9.1}% {:>7.4} {:>6.1}s{}",
            bn, compress, train_acc, test_acc, rand_acc, last_loss, tc.elapsed().as_secs_f64(), m);

        if rand_acc >= 100.0 && train_acc >= 100.0 && test_acc >= 100.0 {
            println!("\n  === MINIMUM BOTTLENECK: N={} ({:.1}x compression) ===", bn, compress);
            println!("  Output: {} chunks x {} = {} int8 values", CHUNK / CHUNK, bn, bn);
            println!("  Parameters: {} (W: {}x{} + biases)", params, bn, DIM);
            break;
        }
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
