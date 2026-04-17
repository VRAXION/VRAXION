//! Quantization ladder: float32 → int8 → int4 → ternary → binary
//!
//! Cél: meddig mehetünk le bit-szélességben mielőtt összeomlik a model.
//! Minden szintnél PTQ (post-hoc) és QAT (quant-aware training) is.
//!
//! 1 seed, plateau-alapú early stopping, kisebb háló (nf=64).
//!
//! 9 run total:
//!   1 × float32 train (baseline)
//!   4 × PTQ eval (ingyen, nem új training)
//!   4 × QAT train (int8, int4, ternary, binary)
//!
//! Run:
//!   cargo run --release --example diag_quant_ladder -- <corpus_path>

use rayon::prelude::*;
use std::time::Instant;

const VOCAB: usize = 2000;
const DIM: usize = 16;
const NF: usize = 64;

#[derive(Clone, Copy, PartialEq, Debug)]
enum QuantMode { None, Int8, Int4, Ternary, Binary }

impl QuantMode {
    fn tag(&self) -> &'static str {
        match self {
            QuantMode::None => "float32",
            QuantMode::Int8 => "int8",
            QuantMode::Int4 => "int4",
            QuantMode::Ternary => "ternary",
            QuantMode::Binary => "binary",
        }
    }
}

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

fn max_abs_2d(v: &[Vec<f32>]) -> f32 {
    v.iter().flat_map(|r| r.iter()).map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-9)
}
fn max_abs_3d(v: &[Vec<Vec<f32>>]) -> f32 {
    v.iter().flat_map(|l| l.iter().flat_map(|r| r.iter())).map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-9)
}

/// Symmetric int quantization: `bits` levels on each side of 0, total 2*levels+1.
#[inline(always)]
fn fake_q_int(x: f32, scale: f32, bits: u32) -> f32 {
    let levels = ((1u32 << (bits - 1)) - 1) as f32; // 127 for int8, 7 for int4
    let q = (x / scale * levels).round().clamp(-levels, levels);
    q * scale / levels
}

/// Ternary {-scale, 0, +scale} with 0.5·scale threshold.
#[inline(always)]
fn fake_q_ternary(x: f32, scale: f32) -> f32 {
    let thr = scale * 0.5;
    if x > thr { scale } else if x < -thr { -scale } else { 0.0 }
}

/// Binary {-scale, +scale} (sign only, no zero).
#[inline(always)]
fn fake_q_binary(x: f32, scale: f32) -> f32 {
    if x >= 0.0 { scale } else { -scale }
}

#[inline(always)]
fn apply_q(x: f32, scale: f32, mode: QuantMode) -> f32 {
    match mode {
        QuantMode::None => x,
        QuantMode::Int8 => fake_q_int(x, scale, 8),
        QuantMode::Int4 => fake_q_int(x, scale, 4),
        QuantMode::Ternary => fake_q_ternary(x, scale),
        QuantMode::Binary => fake_q_binary(x, scale),
    }
}

fn quantize_ws(ws: &[Vec<Vec<f32>>], mode: QuantMode) -> Vec<Vec<Vec<f32>>> {
    if mode == QuantMode::None { return ws.to_vec(); }
    let s = max_abs_3d(ws);
    ws.iter().map(|layer|
        layer.iter().map(|row|
            row.iter().map(|&x| apply_q(x, s, mode)).collect()
        ).collect()
    ).collect()
}
fn quantize_hw(hw: &[Vec<f32>], mode: QuantMode) -> Vec<Vec<f32>> {
    if mode == QuantMode::None { return hw.to_vec(); }
    let s = max_abs_2d(hw);
    hw.iter().map(|r|
        r.iter().map(|&x| apply_q(x, s, mode)).collect()
    ).collect()
}
fn quantize_bs(bs: &[Vec<f32>], mode: QuantMode) -> Vec<Vec<f32>> {
    if mode == QuantMode::None { return bs.to_vec(); }
    let s = max_abs_2d(bs);
    bs.iter().map(|r|
        r.iter().map(|&x| apply_q(x, s, mode)).collect()
    ).collect()
}

#[allow(clippy::too_many_arguments)]
fn eval(
    corpus: &[u8], start: usize, end: usize,
    ws: &[Vec<Vec<f32>>], bs: &[Vec<f32>], hw: &[Vec<f32>], hb: &[f32],
    embed: &[[f32; DIM]], ctx: usize, mask_pos: usize, k: usize, hk: i32, n_proj: usize, nf: usize,
    samples: usize,
) -> f64 {
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

#[allow(clippy::too_many_arguments)]
fn run(
    tag: &str,
    corpus: &[u8],
    split: usize,
    mode: QuantMode,
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
    println!("[{}] start mode={} nf={} params={}", tag, mode.tag(), nf, total_p);

    let samples = 2000usize;
    let max_ep = 300usize;
    let log_every = 10usize;
    let patience = 30usize;

    let mut best_test = 0.0f64;
    let mut best_ep = 0usize;
    let mut no_improve = 0usize;

    for ep in 0..max_ep {
        let lr = 0.01 * (1.0 - ep as f32 / max_ep as f32 * 0.8);
        let mut rt = Rng::new(seed.wrapping_mul(ep as u64 + 1).wrapping_add(42));

        // QAT: use quantized weights in forward (float weights still stored).
        let ws_fwd = quantize_ws(&ws, mode);
        let bs_fwd = quantize_bs(&bs, mode);
        let hw_fwd = quantize_hw(&hw, mode);

        for _ in 0..samples {
            let off = rt.range(0, split.saturating_sub(ctx + 1));
            let chunk = &corpus[off..off + ctx];
            let target = chunk[mask_pos] as usize;
            let emb: Vec<[f32; DIM]> = (0..ctx).map(|i|
                if i == mask_pos { [0.0; DIM] } else { embed[chunk[i] as usize] }
            ).collect();

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
                    hw[c][f] -= lr * pr[c] * co[f]; // straight-through: update float hw
                }
                hb[c] -= lr * pr[c];
            }

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
    println!("=== QUANTIZATION LADDER (int8 → int4 → ternary → binary) ===");
    println!("   corpus = {}", corpus_path);
    let corpus = load_corpus(&corpus_path);
    let split = corpus.len() * 80 / 100;
    println!("   corpus len = {}, split = {}/{}", corpus.len(), split, corpus.len() - split);
    println!("   B0 small (nf={}), plateau stop (30 ep), 1 seed", NF);
    println!();

    let seed = 42u64;
    let ctx = 32; let mask_pos = ctx / 2; let k = 7; let hk = 3i32;

    // ---- 1. Float32 baseline training ----
    println!("===== [1/5] TRAIN: float32 baseline =====");
    let (acc_f32, ep_f32, embed, ws, bs, hw, hb) =
        run("float32", &corpus, split, QuantMode::None, seed);

    // ---- 2. PTQ: apply each quantization to float weights, evaluate ----
    println!();
    println!("===== [2/5] PTQ: post-hoc rounding of float32 weights =====");
    let mut ptq_results: Vec<(QuantMode, f64)> = vec![];
    for &m in &[QuantMode::Int8, QuantMode::Int4, QuantMode::Ternary, QuantMode::Binary] {
        let ws_q = quantize_ws(&ws, m);
        let bs_q = quantize_bs(&bs, m);
        let hw_q = quantize_hw(&hw, m);
        let acc = eval(&corpus, split, corpus.len(), &ws_q, &bs_q, &hw_q, &hb, &embed,
                       ctx, mask_pos, k, hk, 2, NF, 1000);
        println!("  PTQ {:<8} → test acc = {:.2}  (loss: {:+.2}pp)", m.tag(), acc, acc - acc_f32);
        ptq_results.push((m, acc));
    }

    // ---- 3-5. QAT: train from scratch with fake-quant forward ----
    println!();
    let mut qat_results: Vec<(QuantMode, f64, usize)> = vec![];
    for (idx, &m) in [QuantMode::Int8, QuantMode::Int4, QuantMode::Ternary, QuantMode::Binary].iter().enumerate() {
        println!("===== [{}/5] TRAIN: QAT {} =====", idx + 2, m.tag());
        let (acc, ep, _, _, _, _, _) =
            run(&format!("QAT-{}", m.tag()), &corpus, split, m, seed);
        qat_results.push((m, acc, ep));
    }

    // ---- Summary ----
    println!();
    println!("════════════════════════════════════════════════════════════");
    println!("  FINAL SUMMARY — char-LM on FineWeb 30MB, nf={}", NF);
    println!("════════════════════════════════════════════════════════════");
    println!();
    println!("  {:<12}  {:<10}  {:<12}  {:<12}", "bit-width", "float32", "PTQ", "QAT");
    println!("  {:<12}  {:<10}  {:<12}  {:<12}", "----------", "-------", "---", "---");
    println!("  {:<12}  {:>10.2}  {:<12}  {:<12}", "float32", acc_f32, "—", "—");
    for (i, &m) in [QuantMode::Int8, QuantMode::Int4, QuantMode::Ternary, QuantMode::Binary].iter().enumerate() {
        let ptq = ptq_results[i].1;
        let qat = qat_results[i].1;
        let ptq_loss = ptq - acc_f32;
        let qat_loss = qat - acc_f32;
        println!("  {:<12}  {:>10}  {:>6.2}  ({:+.2})  {:>6.2}  ({:+.2})",
                 m.tag(), "", ptq, ptq_loss, qat, qat_loss);
    }
    println!();
    println!("  Total wallclock: {:.1}s", t0.elapsed().as_secs_f64());
}
