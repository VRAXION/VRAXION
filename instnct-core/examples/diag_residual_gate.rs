//! Residual Beukers sweep: raw adders vs bounded residuals.
//!
//! User request:
//!   1. First test Claude's simplest residual idea and a learnable-adder variant.
//!   2. Then test bounded residual variants.
//!
//! Variants:
//!   B0            = Beukers(ab)
//!   raw+0.10      = Beukers(ab) + 0.10 * ((a + b) / 2)
//!   raw+learnable = Beukers(ab) + λ_f * ((a + b) / 2), λ_f learned per feature
//!   bound+0.05    = Beukers(ab) + 0.05 * u/(1+|u|),  u=(a+b)/2
//!   bound+0.10    = Beukers(ab) + 0.10 * u/(1+|u|)
//!   bound+0.20    = Beukers(ab) + 0.20 * u/(1+|u|)
//!
//! Run from repo root:
//!   cargo run --release --example diag_residual_gate -- <corpus_path>

use rayon::prelude::*;
use std::cmp::Ordering;
use std::time::Instant;

const VOCAB: usize = 2000;
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

fn load_corpus(p: &str) -> Vec<u8> {
    std::fs::read(p).expect("corpus read").iter().filter_map(|&b| match b {
        b'a'..=b'z' => Some(b - b'a'),
        b'A'..=b'Z' => Some(b - b'A'),
        b' ' | b'\n' | b'\t' | b'\r' => Some(26),
        _ => None,
    }).collect()
}

#[derive(Clone, Copy, Default)]
struct GateInfo {
    y: f32,
    base_abs: f32,
    add_abs: f32,
}

#[inline(always)]
fn beukers(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

#[inline(always)]
fn gate(kind: usize, pv: &[f32], lambda: f32) -> GateInfo {
    let prod = pv[0] * pv[1];
    let base = beukers(prod);
    let u = 0.5 * (pv[0] + pv[1]);
    let add = match kind {
        0 => 0.0,
        1 => 0.10 * u,
        2 => lambda * u,
        3 => 0.05 * beukers(u),
        4 => 0.10 * beukers(u),
        5 => 0.20 * beukers(u),
        _ => 0.0,
    };
    GateInfo { y: base + add, base_abs: base.abs(), add_abs: add.abs() }
}

#[derive(Clone, Default)]
struct EvalStats {
    acc: f64,
    gate_signal_mean: f32,
    add_signal_mean: f32,
    logit_margin_mean: f32,
    entropy_mean: f32,
}

#[derive(Clone)]
struct RunResult {
    variant: String,
    seed: u64,
    params: usize,
    best_test: f64,
    final_train: EvalStats,
    final_test: EvalStats,
    final_comax: f32,
    final_lambda_mean: f32,
}

fn top1_top2_margin(logits: &[f32]) -> f32 {
    let mut best = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for &v in logits {
        if v > best {
            second = best;
            best = v;
        } else if v > second {
            second = v;
        }
    }
    best - second
}

fn entropy_from_probs(pr: &[f32]) -> f32 {
    let mut h = 0.0f32;
    for &p in pr {
        if p > 1e-8 { h -= p * p.ln(); }
    }
    h
}

#[allow(clippy::too_many_arguments)]
fn eval_split(
    corpus: &[u8],
    start: usize,
    end: usize,
    kind: usize,
    n_proj: usize,
    nf: usize,
    embed: &[[f32; DIM]],
    ws: &[Vec<Vec<f32>>],
    bs: &[Vec<f32>],
    hw: &[Vec<f32>],
    hb: &[f32],
    lambda_add: &[f32],
) -> EvalStats {
    let ctx = 32usize;
    let mask_pos = ctx / 2;
    let k = 7usize;
    let hk = 3i32;
    let mut rng = Rng::new(999);

    let mut ok = 0usize;
    let mut tot = 0usize;
    let mut gate_signal_sum = 0.0f32;
    let mut add_signal_sum = 0.0f32;
    let mut logit_margin_sum = 0.0f32;
    let mut entropy_sum = 0.0f32;

    for _ in 0..1000usize {
        if end < start + ctx + 1 { break; }
        let off = rng.range(start, end.saturating_sub(ctx + 1));
        let chunk = &corpus[off..off + ctx];

        let emb: Vec<[f32; DIM]> = (0..ctx).map(|i|
            if i == mask_pos { [0.0; DIM] } else { embed[chunk[i] as usize] }
        ).collect();

        let mut co = vec![0.0f32; nf];
        let mut sample_gate_sum = 0.0f32;
        let mut sample_add_sum = 0.0f32;

        for f in 0..nf {
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
            let gi = gate(kind, &pv, lambda_add[f]);
            co[f] = gi.y.max(-10.0).min(10.0);
            sample_gate_sum += gi.base_abs + gi.add_abs;
            sample_add_sum += gi.add_abs;
        }

        let mut logits = hb.to_vec();
        for c in 0..27 { for f in 0..nf { logits[c] += hw[c][f] * co[f]; } }
        let pred = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
            .map(|v| v.0).unwrap_or(0);

        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut pr = vec![0.0f32; 27];
        let mut s = 0.0f32;
        for c in 0..27 { pr[c] = (logits[c] - mx).exp(); s += pr[c]; }
        for c in 0..27 { pr[c] /= s; }

        if pred == chunk[mask_pos] as usize { ok += 1; }
        tot += 1;
        gate_signal_sum += sample_gate_sum / nf as f32;
        add_signal_sum += sample_add_sum / nf as f32;
        logit_margin_sum += top1_top2_margin(&logits);
        entropy_sum += entropy_from_probs(&pr);
    }

    if tot == 0 {
        EvalStats::default()
    } else {
        EvalStats {
            acc: ok as f64 / tot as f64 * 100.0,
            gate_signal_mean: gate_signal_sum / tot as f32,
            add_signal_mean: add_signal_sum / tot as f32,
            logit_margin_mean: logit_margin_sum / tot as f32,
            entropy_mean: entropy_sum / tot as f32,
        }
    }
}

fn run(
    tag: &str,
    corpus: &[u8],
    split: usize,
    variant: &str,
    kind: usize,
    n_proj: usize,
    nf: usize,
    seed: u64,
    learnable_lambda: bool,
) -> RunResult {
    let ctx = 32; let mask_pos = ctx / 2; let k = 7; let hk = 3i32; let fan = k * DIM;

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
    let mut lambda_add = vec![0.10f32; nf];

    let total_p = n_proj * (nf * fan + nf) + 27 * nf + 27 + if learnable_lambda { nf } else { 0 };
    println!("[{}-s{}] start  variant={} nf={} proj={} params={}", tag, seed, variant, nf, n_proj, total_p);

    let samples = 2000usize;
    let max_ep = 300usize;
    let log_every = 50usize;
    let mut best_test = 0.0f64;
    let mut final_comax = 0.0f32;
    let mut final_train = EvalStats::default();
    let mut final_test = EvalStats::default();

    for ep in 0..max_ep {
        let lr = 0.01 * (1.0 - ep as f32 / max_ep as f32 * 0.8);
        let mut rt = Rng::new(seed.wrapping_mul(ep as u64 + 1).wrapping_add(42));
        let mut co_max_abs = 0.0f32;

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
                gate(kind, &projs_by_f[f], lambda_add[f]).y.max(-10.0).min(10.0)
            }).collect();
            let local_co_max = co.par_iter().map(|x| x.abs()).reduce(|| 0.0f32, f32::max);
            if local_co_max > co_max_abs { co_max_abs = local_co_max; }

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
                    hw[c][f] -= lr * pr[c] * co[f];
                }
                hb[c] -= lr * pr[c];
            }

            if learnable_lambda {
                for f in 0..nf {
                    let u = 0.5 * (projs_by_f[f][0] + projs_by_f[f][1]);
                    lambda_add[f] -= lr * 0.10 * dc[f] * u;
                    lambda_add[f] = lambda_add[f].clamp(0.0, 0.50);
                }
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
                        let co_plus = gate(kind, &pv, lambda_add[f]).y;
                        pv[p] = old - eps_g;
                        let co_minus = gate(kind, &pv, lambda_add[f]).y;
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
                                            grad * ws[p][f][ki * DIM + d] * 0.1 / n_proj as f32
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
            let tr = eval_split(corpus, 0, split, kind, n_proj, nf, &embed, &ws, &bs, &hw, &hb, &lambda_add);
            let te = eval_split(corpus, split, corpus.len(), kind, n_proj, nf, &embed, &ws, &bs, &hw, &hb, &lambda_add);
            if te.acc > best_test { best_test = te.acc; }
            final_comax = co_max_abs;
            final_train = tr.clone();
            final_test = te.clone();
            let lambda_mean = lambda_add.iter().sum::<f32>() / nf as f32;
            println!(
                "[{}-s{}] ep={} tr={:.1} te={:.1} add={:.3} lm={:.3} ent={:.3} λ={:.3} co_max={:.3}",
                tag, seed, ep + 1, tr.acc, te.acc, te.add_signal_mean,
                te.logit_margin_mean, te.entropy_mean, lambda_mean, co_max_abs
            );
        }
    }

    let final_lambda_mean = lambda_add.iter().sum::<f32>() / nf as f32;
    println!("[{}-s{}] DONE best_test={:.2}", tag, seed, best_test);
    RunResult { variant: variant.to_string(), seed, params: total_p, best_test, final_train, final_test, final_comax, final_lambda_mean }
}

fn mean(xs: &[f64]) -> f64 { xs.iter().sum::<f64>() / xs.len() as f64 }
fn mean_f32(xs: &[f32]) -> f32 { xs.iter().sum::<f32>() / xs.len() as f32 }
fn std_f(xs: &[f64]) -> f64 {
    if xs.len() < 2 { return 0.0; }
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64;
    var.sqrt()
}

fn print_summary(all: &[RunResult], variant: &str) {
    let rs: Vec<&RunResult> = all.iter().filter(|r| r.variant == variant).collect();
    if rs.is_empty() { return; }
    let bests: Vec<f64> = rs.iter().map(|r| r.best_test).collect();
    let gate_sig: Vec<f32> = rs.iter().map(|r| r.final_test.gate_signal_mean).collect();
    let add_sig: Vec<f32> = rs.iter().map(|r| r.final_test.add_signal_mean).collect();
    let lm: Vec<f32> = rs.iter().map(|r| r.final_test.logit_margin_mean).collect();
    let ent: Vec<f32> = rs.iter().map(|r| r.final_test.entropy_mean).collect();
    let lambda: Vec<f32> = rs.iter().map(|r| r.final_lambda_mean).collect();
    let p = rs[0].params;
    let seeds_str: Vec<String> = rs.iter().map(|r| format!("{:.2}", r.best_test)).collect();
    println!("  {:<24} (params={:>6})  best_test = {:.2} ± {:.2}  seeds: {:?}",
        variant, p, mean(&bests), std_f(&bests), seeds_str);
    println!("    test: gate={:.3}  add={:.3}  logit_margin={:.3}  entropy={:.3}  λ_mean={:.3}",
        mean_f32(&gate_sig), mean_f32(&add_sig), mean_f32(&lm), mean_f32(&ent), mean_f32(&lambda));
}

fn main() {
    let t0 = Instant::now();
    let corpus_path = std::env::args().nth(1)
        .unwrap_or_else(|| "instnct-core/tests/fixtures/alice_corpus.txt".to_string());
    println!("=== RESIDUAL BEUKERS DIAG — corpus = {} ===", corpus_path);
    let corpus = load_corpus(&corpus_path);
    let split = corpus.len() * 80 / 100;
    println!("  corpus len = {}, split = {}/{}", corpus.len(), split, corpus.len() - split);
    println!("  300 epochs, 2000 samples/ep, 3 seeds/variant");
    println!();
    println!("  Variants:");
    println!("    B0           Beukers baseline                  nf=128  2-proj");
    println!("    raw+0.10     Beukers + 0.10 * ((a+b)/2)       nf=128  2-proj");
    println!("    raw+learn    Beukers + λ_f * ((a+b)/2)        nf=128  2-proj (+λ params)");
    println!("    bound+0.05   Beukers + 0.05 * B((a+b)/2)      nf=128  2-proj");
    println!("    bound+0.10   Beukers + 0.10 * B((a+b)/2)      nf=128  2-proj");
    println!("    bound+0.20   Beukers + 0.20 * B((a+b)/2)      nf=128  2-proj");
    println!("                 where B(u)=u/(1+|u|)");
    println!();

    let seeds: Vec<u64> = vec![42, 1337, 9999];
    let results = std::sync::Mutex::new(Vec::<RunResult>::new());
    let corpus_ref = &corpus;
    let results_ref = &results;

    for &(tag, variant, kind, learnable) in &[
        ("B0", "Beukers baseline", 0usize, false),
        ("r1", "raw residual 0.10", 1usize, false),
        ("rl", "raw residual learn", 2usize, true),
        ("b5", "bounded residual 0.05", 3usize, false),
        ("b1", "bounded residual 0.10", 4usize, false),
        ("b2", "bounded residual 0.20", 5usize, false),
    ] {
        rayon::scope(|s| {
            for &seed in &seeds {
                s.spawn(move |_| {
                    let r = run(tag, corpus_ref, split, variant, kind, 2, 128, seed, learnable);
                    results_ref.lock().unwrap().push(r);
                });
            }
        });
    }

    let all = results.into_inner().unwrap();

    println!();
    println!("=== RAW RESULTS ===");
    for r in &all {
        println!(
            "  {:<24} seed={:>4}  best={:.2}  final_te={:.2}  final_tr={:.2}  add={:.3}  lm={:.3}  ent={:.3}  λ={:.3}  co_max={:.3}",
            r.variant, r.seed, r.best_test, r.final_test.acc, r.final_train.acc,
            r.final_test.add_signal_mean, r.final_test.logit_margin_mean, r.final_test.entropy_mean,
            r.final_lambda_mean, r.final_comax
        );
    }

    println!();
    println!("=== SUMMARY (mean ± std, 3 seeds) ===");
    print_summary(&all, "Beukers baseline");
    print_summary(&all, "raw residual 0.10");
    print_summary(&all, "raw residual learn");
    print_summary(&all, "bounded residual 0.05");
    print_summary(&all, "bounded residual 0.10");
    print_summary(&all, "bounded residual 0.20");

    println!();
    println!("  Total wallclock: {:.1}s", t0.elapsed().as_secs_f64());
}
