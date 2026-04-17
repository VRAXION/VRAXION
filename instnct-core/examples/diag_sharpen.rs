//! Sharpened Beukers test — "excluded middle" intuition.
//!
//! Hypothesis: Beukers gate ab/(1+|ab|) is too "mushy in the middle" —
//! small signals linger as soft values rather than committing to 0. Adding
//! a soft decision gate σ(β·(|ab| - θ)) suppresses low-confidence signals
//! and lets strong signals pass through clean.
//!
//! Mechanism:
//!   m = ab / (1 + |ab|)             ← standard Beukers signal
//!   g = σ(β · (|ab| - θ))           ← decision gate on RAW signal |ab|
//!   co = m · g                      ← sharpened output
//!
//! NOTE: gate is on |ab|, not |m|, because |m| ∈ [0,1) makes θ ≥ 1 degenerate.
//! Using raw |ab| makes the ρ-threshold sweep meaningful.
//!
//! Threshold sweep (β = 8 fixed):
//!   θ = 1/ρ ≈ 0.755  (loose — lets most signals pass, only near-zero cut)
//!   θ = 1            (neutral — classical "unit threshold")
//!   θ = ρ   ≈ 1.325  (strict — Van der Laan perceptibility threshold)
//!
//! ρ = plastic number, root of x³ = x + 1, the 3D analog of golden ratio.
//!
//! All variants iso-param (2-proj, nf=128, ~32,411 params).
//! 4 variants × 3 seeds = 12 parallel runs on FineWeb 30MB.
//!
//! Run from repo root:
//!   cargo run --release --example diag_sharpen -- <corpus_path>

use rayon::prelude::*;
use std::time::Instant;

const VOCAB: usize = 2000;
const DIM: usize = 16;

const BETA: f32 = 8.0;
// Plastic number ρ: real root of x³ = x + 1
const RHO: f32 = 1.3247179572447460259;

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

/// Gate.
/// kind=0: B0 Beukers baseline
/// kind=1: sharpen θ = 1/ρ (loose)
/// kind=2: sharpen θ = 1   (neutral)
/// kind=3: sharpen θ = ρ   (strict)
#[inline(always)]
fn gate(kind: usize, pv: &[f32]) -> f32 {
    let p = pv[0] * pv[1];
    let m = p / (1.0 + p.abs());
    match kind {
        0 => m,
        1 => {
            let theta = 1.0 / RHO;
            let g = 1.0 / (1.0 + (-BETA * (p.abs() - theta)).exp());
            m * g
        }
        2 => {
            let theta = 1.0;
            let g = 1.0 / (1.0 + (-BETA * (p.abs() - theta)).exp());
            m * g
        }
        3 => {
            let theta = RHO;
            let g = 1.0 / (1.0 + (-BETA * (p.abs() - theta)).exp());
            m * g
        }
        _ => m,
    }
}

#[derive(Clone)]
struct RunResult {
    variant: String,
    seed: u64,
    params: usize,
    best_test: f64,
    final_test: f64,
    final_train: f64,
    final_comax: f32,
    // Telemetry: mean of g gate (measures how "open" the decision is on average)
    // If g_mean near 1: gate basically always open, no sharpening happens
    // If g_mean near 0: gate mostly closed, too aggressive suppression
    // Ideal: somewhere in between, varying with signal strength
    final_g_mean: f32,
}

fn run(
    tag: &str,
    corpus: &[u8],
    split: usize,
    variant: &str,
    kind: usize,
    seed: u64,
) -> RunResult {
    let ctx = 32; let mask_pos = ctx / 2; let k = 7; let hk = 3i32; let fan = k * DIM;
    let n_proj = 2usize;
    let nf = 128usize;

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
    let is_sharpen = kind != 0;
    let theta_val: f32 = match kind {
        1 => 1.0 / RHO,
        2 => 1.0,
        3 => RHO,
        _ => 0.0,
    };
    println!("[{}-s{}] start  variant={} nf={} proj={} params={} theta={:.4}",
             tag, seed, variant, nf, n_proj, total_p, theta_val);

    let samples = 2000usize;
    let max_ep = 300usize;
    let log_every = 50usize;
    let mut best_test = 0.0f64;
    let mut final_test = 0.0f64;
    let mut final_train = 0.0f64;
    let mut final_comax = 0.0f32;
    let mut final_g_mean = 0.0f32;

    for ep in 0..max_ep {
        let lr = 0.01 * (1.0 - ep as f32 / max_ep as f32 * 0.8);
        let mut rt = Rng::new(seed.wrapping_mul(ep as u64 + 1).wrapping_add(42));
        let mut co_max_abs = 0.0f32;
        let mut g_sum = 0.0f32;
        let mut g_count = 0usize;

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

            // Track g-gate opening for telemetry (sharpen variants only).
            if is_sharpen {
                for f in 0..nf {
                    let pv = &projs_by_f[f];
                    let ab_abs = (pv[0] * pv[1]).abs();
                    let g = 1.0 / (1.0 + (-BETA * (ab_abs - theta_val)).exp());
                    g_sum += g;
                    g_count += 1;
                }
            }

            let co: Vec<f32> = (0..nf).into_par_iter().map(|f| {
                gate(kind, &projs_by_f[f]).max(-10.0).min(10.0)
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

            // Numerical gradient — works through the sigmoid gate because
            // the whole gate function is pure on pv.
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
                        let co_plus = gate(kind, &pv);
                        pv[p] = old - eps_g;
                        let co_minus = gate(kind, &pv);
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
                    for w in 0..fan { ws[p][f][w] -= lr * w_grads[p][w]; }
                }
                for &(cidx, d, delta) in embed_deltas {
                    embed[cidx][d] -= lr * delta;
                }
            }
        }

        if (ep + 1) % log_every == 0 || ep + 1 == max_ep {
            let eval = |start: usize, end: usize| -> f64 {
                let mut rng3 = Rng::new(999);
                let mut ok = 0usize; let mut tot = 0usize;
                for _ in 0..1000 {
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
                        co_e[f] = gate(kind, &pv).max(-10.0).min(10.0);
                    }
                    let mut logits = hb.clone();
                    for c in 0..27 { for f in 0..nf { logits[c] += hw[c][f] * co_e[f]; } }
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
            final_test = te; final_train = tr; final_comax = co_max_abs;
            if g_count > 0 {
                final_g_mean = g_sum / g_count as f32;
            }
            if is_sharpen {
                println!("[{}-s{}]  ep={} tr={:.1} te={:.1} co_max={:.3}  g_mean={:.3}",
                    tag, seed, ep + 1, tr, te, co_max_abs, final_g_mean);
            } else {
                println!("[{}-s{}]  ep={} tr={:.1} te={:.1} co_max={:.3}",
                    tag, seed, ep + 1, tr, te, co_max_abs);
            }
        }
    }

    println!("[{}-s{}] DONE best_test={:.2}", tag, seed, best_test);
    RunResult {
        variant: variant.to_string(),
        seed, params: total_p,
        best_test, final_test, final_train, final_comax,
        final_g_mean,
    }
}

fn mean(xs: &[f64]) -> f64 { xs.iter().sum::<f64>() / xs.len() as f64 }
fn std_f(xs: &[f64]) -> f64 {
    if xs.len() < 2 { return 0.0; }
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / xs.len() as f64;
    var.sqrt()
}

fn main() {
    let t0 = Instant::now();
    let corpus_path = std::env::args().nth(1)
        .unwrap_or_else(|| "instnct-core/tests/fixtures/alice_corpus.txt".to_string());
    println!("=== SHARPENED BEUKERS GAUNTLET — corpus = {} ===", corpus_path);
    let corpus = load_corpus(&corpus_path);
    let split = corpus.len() * 80 / 100;
    println!("  corpus len = {}, split = {}/{}", corpus.len(), split, corpus.len() - split);
    println!("  300 epochs, 2000 samples/ep, 3 seeds per variant");
    println!();
    println!("  Variants (all iso-param: 2-proj, nf=128, ~32,411 params):");
    println!("    B0          Beukers baseline                              (no gate)");
    println!("    S-loose     sharpen θ = 1/ρ ≈ 0.755   (loose threshold)");
    println!("    S-neutral   sharpen θ = 1             (neutral threshold)");
    println!("    S-strict    sharpen θ = ρ   ≈ 1.325   (strict, Van der Laan threshold)");
    println!();
    println!("  Gate: co = (ab/(1+|ab|)) · σ(β·(|ab| - θ))   with β = 8");
    println!();

    let seeds: Vec<u64> = vec![42, 1337, 9999];
    let results = std::sync::Mutex::new(Vec::<RunResult>::new());
    let corpus_ref = &corpus;
    let results_ref = &results;

    rayon::scope(|s| {
        for &seed in &seeds {
            s.spawn(move |_| {
                let r = run("B0", corpus_ref, split, "Beukers baseline", 0, seed);
                results_ref.lock().unwrap().push(r);
            });
        }
        for &seed in &seeds {
            s.spawn(move |_| {
                let r = run("S-loose", corpus_ref, split, "sharpen θ=1/ρ", 1, seed);
                results_ref.lock().unwrap().push(r);
            });
        }
        for &seed in &seeds {
            s.spawn(move |_| {
                let r = run("S-neutr", corpus_ref, split, "sharpen θ=1", 2, seed);
                results_ref.lock().unwrap().push(r);
            });
        }
        for &seed in &seeds {
            s.spawn(move |_| {
                let r = run("S-strict", corpus_ref, split, "sharpen θ=ρ", 3, seed);
                results_ref.lock().unwrap().push(r);
            });
        }
    });

    let all = results.into_inner().unwrap();

    println!();
    println!("=== RAW RESULTS ===");
    for r in &all {
        if r.variant == "Beukers baseline" {
            println!("  {:<22} seed={:>4}  best={:.2}  final_te={:.2}  final_tr={:.2}  co_max={:.3}",
                r.variant, r.seed, r.best_test, r.final_test, r.final_train, r.final_comax);
        } else {
            println!("  {:<22} seed={:>4}  best={:.2}  final_te={:.2}  final_tr={:.2}  co_max={:.3}  g_mean={:.3}",
                r.variant, r.seed, r.best_test, r.final_test, r.final_train, r.final_comax, r.final_g_mean);
        }
    }

    println!();
    println!("=== SUMMARY (mean ± std, 3 seeds) ===");
    for var in &["Beukers baseline", "sharpen θ=1/ρ", "sharpen θ=1", "sharpen θ=ρ"] {
        let bests: Vec<f64> = all.iter().filter(|r| r.variant == *var).map(|r| r.best_test).collect();
        if !bests.is_empty() {
            let p = all.iter().find(|r| r.variant == *var).map(|r| r.params).unwrap_or(0);
            let seeds_str: Vec<String> = all.iter().filter(|r| r.variant == *var)
                .map(|r| format!("{:.2}", r.best_test)).collect();
            println!("  {:<22} (params={:>6})  best_test = {:.2} ± {:.2}  seeds: {:?}",
                var, p, mean(&bests), std_f(&bests), seeds_str);
        }
    }

    // Show g_mean per variant (how open the decision gate was on average)
    println!();
    println!("  g_mean telemetry (how often decision gate was open, mean across seeds):");
    for var in &["sharpen θ=1/ρ", "sharpen θ=1", "sharpen θ=ρ"] {
        let gs: Vec<f64> = all.iter().filter(|r| r.variant == *var)
            .map(|r| r.final_g_mean as f64).collect();
        if !gs.is_empty() {
            println!("    {:<18} g_mean = {:.3}  ({})", var, mean(&gs),
                if mean(&gs) > 0.9 { "gate almost always open — threshold too low" }
                else if mean(&gs) < 0.1 { "gate almost always closed — threshold too high" }
                else { "gate modulated by signal — threshold is meaningful" }
            );
        }
    }

    println!();
    println!("  Total wallclock: {:.1}s", t0.elapsed().as_secs_f64());
}
