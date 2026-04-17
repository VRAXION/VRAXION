//! Staged / gradual int4 quantization test.
//!
//! A user ötlete: ne egyszerre kvantáljunk mindent, hanem
//! 1. Edzzünk float32-ben plateau-ig
//! 2. Keressük meg a "legközelebb int4 rácshoz" súlyokat
//! 3. Kvantáljuk őket, fagyasszuk meg (nem tanulnak tovább)
//! 4. Retrain a többire néhány epochig
//! 5. Ismételd, amíg minden súly kvantált
//!
//! Összehasonlítás az egy-lépéses int4 PTQ-val (64.30% az előző tesztben).
//!
//! Run:
//!   cargo run --release --example diag_quant_staged_int4 -- <corpus_path>

use rayon::prelude::*;
use std::time::Instant;

const VOCAB: usize = 2000;
const DIM: usize = 16;
const NF: usize = 64;
const INT4_LEVELS: f32 = 7.0; // 4 bit signed → levels {-7..+7}
const ROUNDS: usize = 10;      // 10 × 10% = 100%
const EPOCHS_PER_ROUND: usize = 20;

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

#[inline(always)]
fn q_int4(x: f32, scale: f32) -> f32 {
    let q = (x / scale * INT4_LEVELS).round().clamp(-INT4_LEVELS, INT4_LEVELS);
    q * scale / INT4_LEVELS
}

#[inline(always)]
fn q_error(x: f32, scale: f32) -> f32 {
    (x - q_int4(x, scale)).abs()
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

/// Train one epoch of B0 with optional freeze-mask support.
/// Frozen weights are held at their current (quantized) value in both forward and backward.
#[allow(clippy::too_many_arguments)]
fn train_one_epoch(
    corpus: &[u8], split: usize,
    embed: &mut Vec<[f32; DIM]>,
    ws: &mut Vec<Vec<Vec<f32>>>,
    bs: &mut Vec<Vec<f32>>,
    hw: &mut Vec<Vec<f32>>,
    hb: &mut Vec<f32>,
    ws_frozen: &[Vec<Vec<bool>>],
    hw_frozen: &[Vec<bool>],
    lr: f32, seed: u64, ep: u64,
    samples: usize,
    ctx: usize, mask_pos: usize, k: usize, hk: i32, n_proj: usize, nf: usize,
) {
    let fan = k * DIM;
    let mut rt = Rng::new(seed.wrapping_mul(ep + 1).wrapping_add(42));

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
                pv[p] = bs[p][f];
                for ki in 0..k {
                    let pos = mask_pos as i32 + ki as i32 - hk;
                    if pos >= 0 && (pos as usize) < ctx {
                        for d in 0..DIM {
                            pv[p] += ws[p][f][ki * DIM + d] * emb[pos as usize][d];
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
        for c in 0..27 { for f in 0..nf { logits[c] += hw[c][f] * co[f]; } }
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut pr = vec![0.0f32; 27]; let mut s = 0.0f32;
        for c in 0..27 { pr[c] = (logits[c] - mx).exp(); s += pr[c]; }
        for c in 0..27 { pr[c] /= s; }
        pr[target] -= 1.0;

        let mut dc = vec![0.0f32; nf];
        for c in 0..27 {
            for f in 0..nf {
                dc[f] += pr[c] * hw[c][f];
                if !hw_frozen[c][f] {
                    hw[c][f] -= lr * pr[c] * co[f];
                }
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
                                        grad * ws[p][f][ki * DIM + d] * 0.1 / n_proj as f32));
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
                for w in 0..fan {
                    if !ws_frozen[p][f][w] {
                        ws[p][f][w] -= lr * w_grads[p][w];
                    }
                }
            }
            for &(cidx, d, delta) in embed_deltas {
                embed[cidx][d] -= lr * delta;
            }
        }
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus_path = std::env::args().nth(1)
        .unwrap_or_else(|| "instnct-core/tests/fixtures/alice_corpus.txt".to_string());
    println!("=== STAGED int4 QUANTIZATION — corpus = {} ===", corpus_path);
    let corpus = load_corpus(&corpus_path);
    let split = corpus.len() * 80 / 100;
    println!("  corpus len = {}, split = {}/{}", corpus.len(), split, corpus.len() - split);
    println!("  B0 small (nf={}), {} rounds × {} epochs/round, quant batch = 10%",
             NF, ROUNDS, EPOCHS_PER_ROUND);
    println!();

    let seed = 42u64;
    let ctx = 32; let mask_pos = ctx / 2; let k = 7; let hk = 3i32; let fan = k * DIM;
    let n_proj = 2;
    let nf = NF;
    let samples = 2000;

    // ---- Init ----
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

    let ws_frozen_all_false: Vec<Vec<Vec<bool>>> =
        (0..n_proj).map(|_| (0..nf).map(|_| vec![false; fan]).collect()).collect();
    let hw_frozen_all_false: Vec<Vec<bool>> =
        (0..27).map(|_| vec![false; nf]).collect();

    // ---- Phase 1: float32 training ----
    println!("--- Phase 1: float32 training to plateau ---");
    let max_ep_init = 200;
    let patience = 30;
    let log_every = 10;
    let mut best_test = 0.0f64;
    let mut no_improve = 0usize;
    let mut final_ep = 0usize;

    for ep in 0..max_ep_init {
        let lr = 0.01 * (1.0 - ep as f32 / max_ep_init as f32 * 0.8);
        train_one_epoch(&corpus, split,
                        &mut embed, &mut ws, &mut bs, &mut hw, &mut hb,
                        &ws_frozen_all_false, &hw_frozen_all_false,
                        lr, seed, ep as u64, samples,
                        ctx, mask_pos, k, hk, n_proj, nf);
        final_ep = ep + 1;

        if (ep + 1) % log_every == 0 {
            let te = eval(&corpus, split, corpus.len(),
                         &ws, &bs, &hw, &hb, &embed,
                         ctx, mask_pos, k, hk, n_proj, nf, 1000);
            if te > best_test + 0.01 {
                best_test = te;
                no_improve = 0;
            } else {
                no_improve += log_every;
            }
            println!("[float32] ep={:3} te={:.1} best={:.1} (plateau:{})",
                     ep + 1, te, best_test, no_improve);
            if no_improve >= patience { break; }
        }
    }

    let acc_float = eval(&corpus, split, corpus.len(),
                         &ws, &bs, &hw, &hb, &embed,
                         ctx, mask_pos, k, hk, n_proj, nf, 1000);
    println!("[float32] DONE te={:.2} @ ep={}", acc_float, final_ep);

    // Compute global scale (frozen for the rest of the run)
    let max_ws = ws.iter().flat_map(|l| l.iter().flat_map(|r| r.iter()))
                     .map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-9);
    let max_hw = hw.iter().flat_map(|r| r.iter())
                     .map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-9);
    println!("  scales: ws={:.4}  hw={:.4}", max_ws, max_hw);

    // ---- Phase 2: staged quantization ----
    println!();
    println!("--- Phase 2: staged int4 quantization ({} rounds, 10% each) ---", ROUNDS);

    let mut ws_frozen: Vec<Vec<Vec<bool>>> =
        (0..n_proj).map(|_| (0..nf).map(|_| vec![false; fan]).collect()).collect();
    let mut hw_frozen: Vec<Vec<bool>> =
        (0..27).map(|_| vec![false; nf]).collect();

    let total_ws = n_proj * nf * fan;
    let total_hw = 27 * nf;
    let total_params = total_ws + total_hw;
    let per_round = total_params / ROUNDS;

    println!("  total quantizable params: {} (ws: {}, hw: {}), per round: {}",
             total_params, total_ws, total_hw, per_round);
    println!();

    let mut round_results: Vec<(usize, f64, usize)> = vec![];

    for round in 1..=ROUNDS {
        // Collect all (error, location) pairs for non-frozen weights
        let mut err_list: Vec<(f32, usize)> = Vec::with_capacity(total_params);
        // Encode location as index: 0..total_ws = ws, total_ws..total_ws+total_hw = hw
        for p in 0..n_proj {
            for f in 0..nf {
                for i in 0..fan {
                    if !ws_frozen[p][f][i] {
                        let idx = p * nf * fan + f * fan + i;
                        err_list.push((q_error(ws[p][f][i], max_ws), idx));
                    }
                }
            }
        }
        for c in 0..27 {
            for f in 0..nf {
                if !hw_frozen[c][f] {
                    let idx = total_ws + c * nf + f;
                    err_list.push((q_error(hw[c][f], max_hw), idx));
                }
            }
        }

        // Sort by error (ascending: smallest error = easiest to quantize)
        err_list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top per_round (easiest) and freeze them
        let to_freeze = per_round.min(err_list.len());
        let mut frozen_count_ws = 0;
        let mut frozen_count_hw = 0;
        for &(_, idx) in err_list.iter().take(to_freeze) {
            if idx < total_ws {
                let p = idx / (nf * fan);
                let f = (idx / fan) % nf;
                let i = idx % fan;
                ws[p][f][i] = q_int4(ws[p][f][i], max_ws);
                ws_frozen[p][f][i] = true;
                frozen_count_ws += 1;
            } else {
                let hi = idx - total_ws;
                let c = hi / nf;
                let f = hi % nf;
                hw[c][f] = q_int4(hw[c][f], max_hw);
                hw_frozen[c][f] = true;
                frozen_count_hw += 1;
            }
        }

        let total_frozen: usize = ws_frozen.iter().flat_map(|l| l.iter().flat_map(|r| r.iter())).filter(|&&b| b).count()
            + hw_frozen.iter().flat_map(|r| r.iter()).filter(|&&b| b).count();
        let pct = 100.0 * total_frozen as f64 / total_params as f64;

        println!("[round {}/{}]  freezing {} params (ws:{}, hw:{}), total frozen: {}/{} ({:.1}%)",
                 round, ROUNDS, to_freeze, frozen_count_ws, frozen_count_hw,
                 total_frozen, total_params, pct);

        // Retrain for EPOCHS_PER_ROUND epochs
        for ep in 0..EPOCHS_PER_ROUND {
            let lr = 0.005 * (1.0 - ep as f32 / EPOCHS_PER_ROUND as f32 * 0.5);
            let ep_total = (final_ep + (round - 1) * EPOCHS_PER_ROUND + ep) as u64;
            train_one_epoch(&corpus, split,
                            &mut embed, &mut ws, &mut bs, &mut hw, &mut hb,
                            &ws_frozen, &hw_frozen,
                            lr, seed, ep_total, samples,
                            ctx, mask_pos, k, hk, n_proj, nf);
        }

        let te = eval(&corpus, split, corpus.len(),
                     &ws, &bs, &hw, &hb, &embed,
                     ctx, mask_pos, k, hk, n_proj, nf, 1000);
        println!("[round {}/{}] frozen={:.1}% → te={:.2}", round, ROUNDS, pct, te);
        println!();
        round_results.push((round, te, total_frozen));
    }

    // ---- Final eval: all weights quantized ----
    let acc_final = eval(&corpus, split, corpus.len(),
                         &ws, &bs, &hw, &hb, &embed,
                         ctx, mask_pos, k, hk, n_proj, nf, 1000);

    println!("════════════════════════════════════════════════════════════");
    println!("  STAGED int4 SUMMARY");
    println!("════════════════════════════════════════════════════════════");
    println!();
    println!("  float32 baseline:         {:>6.2}", acc_float);
    println!();
    println!("  Round-by-round:");
    for &(r, acc, frozen) in &round_results {
        let pct = 100.0 * frozen as f64 / total_params as f64;
        let loss = acc - acc_float;
        println!("  Round {:>2} (frozen {:>5.1}%)  te={:>6.2}  loss={:+.2}pp", r, pct, acc, loss);
    }
    println!();
    println!("  final (100% int4):        {:>6.2}  loss={:+.2}pp", acc_final, acc_final - acc_float);
    println!();
    println!("  Compare to one-shot results (previous test):");
    println!("    one-shot int4 PTQ:  64.30  (loss −1.80pp)");
    println!("    one-shot int4 QAT:  62.80  (loss −3.30pp)");
    let staged_loss = acc_final - acc_float;
    println!("    staged int4:        {:>5.2}  (loss {:+.2}pp)", acc_final, staged_loss);
    println!();
    println!("  Total wallclock: {:.1}s", t0.elapsed().as_secs_f64());
}
