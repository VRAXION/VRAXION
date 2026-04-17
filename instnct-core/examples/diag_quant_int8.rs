//! Quantization ladder test — int8 weights on small B0.
//!
//! Cél: kiderítsük hogy a Beukers gate kvantálható-e int8-ra, és ha igen,
//! QAT (quant-aware training) mennyivel jobb mint PTQ (post-training).
//!
//! 3 variáns, 1 seed, plateau-alapú early stopping:
//!   1. B0 float32    — baseline reference
//!   2. B0 PTQ int8   — edzz float32-ben, aztán EGYSZERRE kvantáld a súlyokat
//!   3. B0 QAT int8   — edzz VÉGIG fake-quant forward-dal (straight-through backward)
//!
//! Kisebb háló: nf=64 (gyors). Plateau: 30 epoch javulás nélkül → stop.
//!
//! Run:
//!   cargo run --release --example diag_quant_int8 -- <corpus_path>

use rayon::prelude::*;
use std::time::Instant;

const VOCAB: usize = 2000;
const DIM: usize = 16;
const NF: usize = 64;

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1)) }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn normal(&mut self) -> f32 {
        let u1 = (((self.next() >> 33) % 65536) as f32 / 65536.0).max(1e-7);
        let u2 = ((self.next() >> 33) % 65536) as f32 / 65536.0;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        if hi <= lo { lo } else { lo + (self.next() as usize % (hi - lo)) }
    }
}

fn load_corpus(p: &str) -> Vec<u8> {
    std::fs::read(p).expect("corpus read").iter().filter_map(|&b| match b {
        b'a'..=b'z' => Some(b - b'a'),
        b'A'..=b'Z' => Some(b - b'A'),
        b' ' | b'\n' | b'\t' | b'\r' => Some(26),
        _ => None,
    }).collect()
}

/// Per-tensor max-abs scale.
fn max_abs_1d(v: &[f32]) -> f32 {
    v.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-9)
}
fn max_abs_2d(v: &[Vec<f32>]) -> f32 {
    v.iter().flat_map(|r| r.iter()).map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-9)
}
fn max_abs_3d(v: &[Vec<Vec<f32>>]) -> f32 {
    v.iter().flat_map(|l| l.iter().flat_map(|r| r.iter())).map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-9)
}

/// Fake quantization to int8 grid, returning float.
/// scale = max|x|; output ∈ [-scale, scale] on 256 levels.
#[inline(always)]
fn fake_q_i8(x: f32, scale: f32) -> f32 {
    let q = (x / scale * 127.0).round().clamp(-127.0, 127.0);
    q * scale / 127.0
}

/// Quantize 3D weight tensor in-place-style (returns a new one).
fn quantize_ws(ws: &[Vec<Vec<f32>>]) -> Vec<Vec<Vec<f32>>> {
    let s = max_abs_3d(ws);
    ws.iter().map(|layer|
        layer.iter().map(|row|
            row.iter().map(|&x| fake_q_i8(x, s)).collect()
        ).collect()
    ).collect()
}
fn quantize_hw(hw: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let s = max_abs_2d(hw);
    hw.iter().map(|r|
        r.iter().map(|&x| fake_q_i8(x, s)).collect()
    ).collect()
}
fn quantize_bs(bs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let s = max_abs_2d(bs);
    bs.iter().map(|r|
        r.iter().map(|&x| fake_q_i8(x, s)).collect()
    ).collect()
}

/// Evaluate given already-initialized network weights.
#[allow(clippy::too_many_arguments)]
fn eval(
    corpus: &[u8], start: usize, end: usize,
    ws: &[Vec<Vec<f32>>], bs: &[Vec<f32>], hw: &[Vec<f32>], hb: &[f32],
    embed: &[[f32; DIM]], ctx: usize, mask_pos: usize, k: usize, hk: i32, n_proj: usize, nf: usize,
    samples: usize,
) -> f64 {
    let fan = k * DIM;
    let mut rng3 = Rng::new(999);
    let mut ok = 0usize; let mut tot = 0usize;
    for _ in 0..samples {
        if end < start + ctx + 1 { break; }
        let off = rng3.range(start, end.saturating_sub(ctx + 1));
        let chunk = &corpus[off..off + ctx];
        let emb: Vec<[f32; DIM]> = (0..ctx).map(|i|
            if i == mask_pos { [0.0; DIM] } else { embed[chunk[i] as usize] }
        ).collect();
        let mut co_e = vec![0.0f32; nf];
        for f in 0..nf {
            let mut pv = vec![0.0f32; n_proj];
            for p in 0..n_proj {
                pv[p] = bs[p][f];
                for ki in 0..k {
                    let pos = mask_pos as i32 + ki as i32 - hk;
                    if pos >= 0 && (pos as usize) < ctx {
                        for d in 0..DIM { pv[p] += ws[p][f][ki * DIM + d] * emb[pos as usize][d]; }
                    }
                }
            }
            let p = pv[0] * pv[1];
            co_e[f] = (p / (1.0 + p.abs())).max(-10.0).min(10.0);
        }
        let mut logits = hb.to_vec();
        for c in 0..27 { for f in 0..nf { logits[c] += hw[c][f] * co_e[f]; } }
        let pred = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|v| v.0).unwrap_or(0);
        if pred == chunk[mask_pos] as usize { ok += 1; }
        tot += 1;
    }
    if tot == 0 { 0.0 } else { ok as f64 / tot as f64 * 100.0 }
}

/// Run one variant with plateau-based early stopping.
/// mode: 0=float32, 1=QAT int8 (fake-quant during forward)
/// After training, mode 0/1 returns best_test directly.
/// For PTQ, call run with mode=0 to get float weights, then call ptq_eval() separately.
#[allow(clippy::too_many_arguments)]
fn run(
    tag: &str,
    corpus: &[u8],
    split: usize,
    mode: usize,
    seed: u64,
) -> (f64, usize, Vec<[f32; DIM]>, Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {
    let ctx = 32; let mask_pos = ctx / 2; let k = 7; let hk = 3i32; let fan = k * DIM;
    let n_proj = 2;
    let nf = NF;

    let mut rng = Rng::new(seed);
    let sc_e = (1.0 / DIM as f32).sqrt();
    let sc_c = (2.0 / fan as f32).sqrt();
    let sc_h = (2.0 / nf as f32).sqrt();

    let mut embed: Vec<[f32; DIM]> = (0..VOCAB).map(|_| {
        let mut v = [0.0; DIM]; for d in 0..DIM { v[d] = rng.normal() * sc_e; } v
    }).collect();
    let mut ws: Vec<Vec<Vec<f32>>> = (0..n_proj).map(|_|
        (0..nf).map(|_| (0..fan).map(|_| rng.normal() * sc_c).collect()).collect()
    ).collect();
    let mut bs: Vec<Vec<f32>> = (0..n_proj).map(|_| vec![0.0f32; nf]).collect();
    let mut hw: Vec<Vec<f32>> = (0..27).map(|_| (0..nf).map(|_| rng.normal() * sc_h).collect()).collect();
    let mut hb = vec![0.0f32; 27];

    let total_p = n_proj * (nf * fan + nf) + 27 * nf + 27;
    println!("[{}] start  mode={} nf={} params={}", tag, mode, nf, total_p);

    let samples = 2000usize;
    let max_ep = 300usize;
    let log_every = 10usize;
    let patience = 30usize; // if no improvement in 30 epochs, stop

    let mut best_test = 0.0f64;
    let mut best_ep = 0usize;
    let mut no_improve = 0usize;

    for ep in 0..max_ep {
        let lr = 0.01 * (1.0 - ep as f32 / max_ep as f32 * 0.8);
        let mut rt = Rng::new(seed.wrapping_mul(ep as u64 + 1).wrapping_add(42));

        // For QAT: quantize weights at start of epoch, use those in forward
        let ws_fwd: Vec<Vec<Vec<f32>>> = if mode == 1 { quantize_ws(&ws) } else { ws.clone() };
        let bs_fwd: Vec<Vec<f32>>     = if mode == 1 { quantize_bs(&bs) } else { bs.clone() };
        let hw_fwd: Vec<Vec<f32>>     = if mode == 1 { quantize_hw(&hw) } else { hw.clone() };

        for _ in 0..samples {
            let off = rt.range(0, split.saturating_sub(ctx + 1));
            let chunk = &corpus[off..off + ctx];
            let target = chunk[mask_pos] as usize;
            let emb: Vec<[f32; DIM]> = (0..ctx).map(|i|
                if i == mask_pos { [0.0; DIM] } else { embed[chunk[i] as usize] }
            ).collect();

            // Forward with (possibly quantized) weights
            let projs_by_f: Vec<Vec<f32>> = (0..nf).into_par_iter().map(|f| {
                let mut pv = vec![0.0f32; n_proj];
                for p in 0..n_proj {
                    pv[p] = bs_fwd[p][f];
                    for ki in 0..k {
                        let pos = mask_pos as i32 + ki as i32 - hk;
                        if pos >= 0 && (pos as usize) < ctx {
                            for d in 0..DIM {
                                pv[p] += ws_fwd[p][f][ki * DIM + d] * emb[pos as usize][d];
                            }
                        }
                    }
                }
                pv
            }).collect();

            let co: Vec<f32> = (0..nf).into_par_iter().map(|f| {
                let pv = &projs_by_f[f];
                let p = pv[0] * pv[1];
                (p / (1.0 + p.abs())).max(-10.0).min(10.0)
            }).collect();

            let mut logits = hb.clone();
            for c in 0..27 { for f in 0..nf { logits[c] += hw_fwd[c][f] * co[f]; } }
            let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut pr = vec![0.0f32; 27]; let mut s = 0.0f32;
            for c in 0..27 { pr[c] = (logits[c] - mx).exp(); s += pr[c]; }
            for c in 0..27 { pr[c] /= s; }
            pr[target] -= 1.0;

            let mut dc = vec![0.0f32; nf];
            for c in 0..27 {
                for f in 0..nf {
                    dc[f] += pr[c] * hw_fwd[c][f];
                    // Straight-through: update float hw directly
                    hw[c][f] -= lr * pr[c] * co[f];
                }
                hb[c] -= lr * pr[c];
            }

            // Numerical gradient for projections
            let eps_g = 0.01f32;
            let per_f: Vec<(Vec<Vec<f32>>, Vec<f32>, Vec<(usize, usize, f32)>)> =
                (0..nf).into_par_iter().map(|f| {
                    let mut w_grads: Vec<Vec<f32>> = vec![vec![0.0f32; fan]; n_proj];
                    let mut b_grads: Vec<f32> = vec![0.0f32; n_proj];
                    let mut embed_deltas: Vec<(usize, usize, f32)> = Vec::with_capacity(n_proj * k * DIM);
                    let mut pv = projs_by_f[f].clone();
                    for p in 0..n_proj {
                        let old = pv[p];
                        pv[p] = old + eps_g;
                        let pr_p = pv[0] * pv[1]; let co_plus = pr_p / (1.0 + pr_p.abs());
                        pv[p] = old - eps_g;
                        let pr_m = pv[0] * pv[1]; let co_minus = pr_m / (1.0 + pr_m.abs());
                        pv[p] = old;
                        let grad = dc[f] * (co_plus - co_minus) / (2.0 * eps_g);
                        b_grads[p] = grad;
                        for ki in 0..k {
                            let pos = mask_pos as i32 + ki as i32 - hk;
                            if pos >= 0 && (pos as usize) < ctx {
                                let pi = pos as usize;
                                for d in 0..DIM {
                                    w_grads[p][ki * DIM + d] = grad * emb[pi][d];
                                    if pi != mask_pos {
                                        embed_deltas.push((chunk[pi] as usize, d,
                                            grad * ws_fwd[p][f][ki * DIM + d] * 0.1 / n_proj as f32));
                                    }
                                }
                            }
                        }
                    }
                    (w_grads, b_grads, embed_deltas)
                }).collect();

            // Straight-through update: update float ws, bs
            for (f, (w_grads, b_grads, embed_deltas)) in per_f.iter().enumerate() {
                for p in 0..n_proj {
                    bs[p][f] -= lr * b_grads[p];
                    for w in 0..fan { ws[p][f][w] -= lr * w_grads[p][w]; }
                }
                for &(cidx, d, delta) in embed_deltas {
                    embed[cidx][d] -= lr * delta;
                }
            }
        }

        if (ep + 1) % log_every == 0 || ep + 1 == max_ep {
            let tr = eval(corpus, 0, split, &ws_fwd, &bs_fwd, &hw_fwd, &hb, &embed,
                          ctx, mask_pos, k, hk, n_proj, nf, 500);
            let te = eval(corpus, split, corpus.len(), &ws_fwd, &bs_fwd, &hw_fwd, &hb, &embed,
                          ctx, mask_pos, k, hk, n_proj, nf, 1000);
            if te > best_test + 0.01 {
                best_test = te;
                best_ep = ep + 1;
                no_improve = 0;
            } else {
                no_improve += log_every;
            }
            println!("[{}]  ep={:3} tr={:.1} te={:.1} best={:.1} (plateau:{})",
                     tag, ep + 1, tr, te, best_test, no_improve);

            if no_improve >= patience {
                println!("[{}] plateau @ ep={}, stop", tag, ep + 1);
                break;
            }
        }
    }

    println!("[{}] DONE best_test={:.2} @ ep={}", tag, best_test, best_ep);
    (best_test, best_ep, embed, ws, bs, hw, hb)
}

fn main() {
    let t0 = Instant::now();
    let corpus_path = std::env::args().nth(1)
        .unwrap_or_else(|| "instnct-core/tests/fixtures/alice_corpus.txt".to_string());
    println!("=== QUANTIZATION LADDER (int8) — corpus = {} ===", corpus_path);
    let corpus = load_corpus(&corpus_path);
    let split = corpus.len() * 80 / 100;
    println!("  corpus len = {}, split = {}/{}", corpus.len(), split, corpus.len() - split);
    println!("  B0 small (nf={}), plateau stop (30 ep), 1 seed", NF);
    println!();

    let seed = 42u64;

    // ---------- 1. Float32 baseline ----------
    println!("--- Running: B0 float32 ---");
    let (acc_f32, ep_f32, embed, ws, bs, hw, hb) =
        run("float32", &corpus, split, 0, seed);

    // ---------- 2. PTQ int8: quantize saved float weights, re-evaluate ----------
    println!();
    println!("--- Applying PTQ int8 to float32 model ---");
    let ws_q = quantize_ws(&ws);
    let bs_q = quantize_bs(&bs);
    let hw_q = quantize_hw(&hw);

    let ctx = 32; let mask_pos = ctx / 2; let k = 7; let hk = 3i32;
    let acc_ptq = eval(&corpus, split, corpus.len(), &ws_q, &bs_q, &hw_q, &hb, &embed,
                       ctx, mask_pos, k, hk, 2, NF, 1000);
    println!("[PTQ int8] test acc = {:.2}", acc_ptq);

    // ---------- 3. QAT int8: train from scratch with fake-quant forward ----------
    println!();
    println!("--- Running: B0 QAT int8 ---");
    let (acc_qat, ep_qat, _, _, _, _, _) =
        run("QAT-i8", &corpus, split, 1, seed);

    // ---------- Summary ----------
    println!();
    println!("=== SUMMARY ===");
    println!("  B0 float32   (bajnok ref)     best_test = {:.2}  @ ep {}", acc_f32, ep_f32);
    println!("  B0 PTQ int8  (post-hoc round) best_test = {:.2}  (loss: {:+.2}pp)",
             acc_ptq, acc_ptq - acc_f32);
    println!("  B0 QAT int8  (fake-quant fwd) best_test = {:.2}  @ ep {}  (loss: {:+.2}pp)",
             acc_qat, ep_qat, acc_qat - acc_f32);
    println!();
    println!("  Interpretation:");
    println!("    PTQ loss ~0 → weights already quantization-friendly (Beukers saturates naturally)");
    println!("    QAT loss ~0 → int8 viable; proceed to int4/ternary next");
    println!("    PTQ >> QAT loss → the network needed adaptation; QAT is the right approach");
    println!();
    println!("  Total wallclock: {:.1}s", t0.elapsed().as_secs_f64());
}
