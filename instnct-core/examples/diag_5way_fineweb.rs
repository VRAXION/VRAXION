//! Integration smoke test for the parquet -> .txt -> load_corpus pipeline.
//!
//! Runs a tiny Beukers char-LM (gate_raw #0 only, nf=64, 30 epochs) on a
//! corpus file given on the command line. Proves the full extraction path
//! works end-to-end: the existing [`instnct_core::load_corpus`] opens the
//! `.txt` that `extract_fineweb_txt` produced and trains without fuss.
//!
//! This is **not** a replacement for `diag_5way_gauntlet.rs` — it is a 30-
//! second sanity run to confirm integration. Use the 5-way gauntlet for
//! actual research; it reads the same corpus file shape.
//!
//! # Usage
//!
//!   # against the bundled Alice fixture (100 KB, quick)
//!   cargo run --release --features parquet --example diag_5way_fineweb -- \
//!       instnct-core/tests/fixtures/alice_corpus.txt
//!
//!   # against the extracted FineWeb corpus (30 MB, ~5 min)
//!   cargo run --release --features parquet --example diag_5way_fineweb -- \
//!       "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt"

use instnct_core::load_corpus;
use rayon::prelude::*;
use std::time::Instant;

const VOCAB: usize = 27;
const DIM: usize = 16;

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

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "instnct-core/tests/fixtures/alice_corpus.txt".to_string()
    });
    let t_open = Instant::now();
    let corpus = match load_corpus(&path) {
        Ok(c) => c,
        Err(e) => { eprintln!("load_corpus({path:?}): {e}"); std::process::exit(1); }
    };
    println!("=== diag_5way_fineweb — integration smoke test ===");
    println!("  corpus path : {path}");
    println!("  corpus len  : {} classes ({:.2} MB raw indices)", corpus.len(), corpus.len() as f64 / 1e6);
    println!("  load took   : {:.2}s", t_open.elapsed().as_secs_f64());

    if corpus.len() < 1_000 {
        eprintln!("corpus too small (<1000 chars); refusing to train");
        std::process::exit(2);
    }

    // Character distribution sanity print.
    let mut hist = [0u64; VOCAB];
    for &c in &corpus { if (c as usize) < VOCAB { hist[c as usize] += 1; } }
    let total: u64 = hist.iter().sum();
    println!("  char distribution (top 5):");
    let mut pairs: Vec<(usize, u64)> = hist.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1));
    for (i, (c, n)) in pairs.iter().take(5).enumerate() {
        let ch = if *c == 26 { ' ' } else { (b'a' + *c as u8) as char };
        println!("    [{}] {:?} : {} ({:.2}%)", i, ch, n, 100.0 * *n as f64 / total as f64);
    }
    println!();

    let split = corpus.len() * 80 / 100;
    let nf = 64usize;
    let ctx = 32usize; let mask_pos = ctx / 2; let k = 7usize; let hk = 3i32;
    let fan = k * DIM;
    let n_proj = 2usize;

    let mut rng = Rng::new(42);
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
    let mut hw: Vec<Vec<f32>> = (0..VOCAB).map(|_| (0..nf).map(|_| rng.normal() * sc_h).collect()).collect();
    let mut hb = vec![0.0f32; VOCAB];

    let samples = 1000usize;
    let max_ep = 30usize;
    let log_every = 5usize;
    let mut best_test = 0.0f64;
    let t_train = Instant::now();

    for ep in 0..max_ep {
        let lr = 0.01 * (1.0 - ep as f32 / max_ep as f32 * 0.8);
        let mut rt = Rng::new(ep as u64 * 1000 + 42);
        for _ in 0..samples {
            if split <= ctx + 1 { break; }
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

            // Beukers baseline: ab / (1 + |ab|).
            let raw: Vec<f32> = (0..nf).map(|f| {
                let pp = projs_by_f[f][0] * projs_by_f[f][1];
                pp / (1.0 + pp.abs())
            }).collect();
            let co: Vec<f32> = raw.iter().map(|&r| r.max(-10.0).min(10.0)).collect();

            let mut logits = hb.clone();
            for c in 0..VOCAB {
                for f in 0..nf { logits[c] += hw[c][f] * co[f]; }
            }
            let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut pr = vec![0.0f32; VOCAB]; let mut s = 0.0f32;
            for c in 0..VOCAB { pr[c] = (logits[c] - mx).exp(); s += pr[c]; }
            for c in 0..VOCAB { pr[c] /= s; }
            pr[target] -= 1.0;

            let mut dc = vec![0.0f32; nf];
            for c in 0..VOCAB {
                for f in 0..nf {
                    dc[f] += pr[c] * hw[c][f];
                    hw[c][f] -= lr * pr[c] * co[f];
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
                        let pp_p = pv[0] * pv[1];
                        let co_plus = pp_p / (1.0 + pp_p.abs());
                        pv[p] = old - eps_g;
                        let pp_m = pv[0] * pv[1];
                        let co_minus = pp_m / (1.0 + pp_m.abs());
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
                                        embed_deltas.push((
                                            chunk[pi] as usize, d,
                                            grad * ws[p][f][ki * DIM + d] * 0.1 / n_proj as f32,
                                        ));
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
            let eval = |start: usize, end: usize| -> f64 {
                if end < start + ctx + 1 { return 0.0; }
                let mut rng3 = Rng::new(999);
                let mut ok = 0usize; let mut tot = 0usize;
                for _ in 0..500 {
                    let off = rng3.range(start, end.saturating_sub(ctx + 1));
                    let chunk = &corpus[off..off + ctx];
                    let emb: Vec<[f32; DIM]> = (0..ctx).map(|i|
                        if i == mask_pos { [0.0; DIM] } else { embed[chunk[i] as usize] }
                    ).collect();
                    let mut raw_e = vec![0.0f32; nf];
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
                        let pp = pv[0] * pv[1];
                        raw_e[f] = pp / (1.0 + pp.abs());
                    }
                    let co_e: Vec<f32> = raw_e.iter().map(|&r| r.max(-10.0).min(10.0)).collect();
                    let mut logits = hb.clone();
                    for c in 0..VOCAB { for f in 0..nf { logits[c] += hw[c][f] * co_e[f]; } }
                    let pred = logits.iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|v| v.0).unwrap_or(0);
                    if pred == chunk[mask_pos] as usize { ok += 1; }
                    tot += 1;
                }
                if tot == 0 { 0.0 } else { ok as f64 / tot as f64 * 100.0 }
            };
            let tr = eval(0, split);
            let te = eval(split, corpus.len());
            if te > best_test { best_test = te; }
            println!("  ep {:>3}  train {:>5.2}%  test {:>5.2}%", ep + 1, tr, te);
        }
    }

    println!();
    println!("--- smoke test done ---");
    println!("  best test  : {:.2}%", best_test);
    println!("  train time : {:.2}s", t_train.elapsed().as_secs_f64());
    println!("  exit OK — corpus loadable and trainable end-to-end");
}
