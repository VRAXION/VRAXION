//! branch-budget test: enforce competition between the (ab) branch and the (c) branch.
//!
//! User's intuition: "N energy distributes among the three — if one uses it,
//! the others have less". Tests whether the c-projection EARNS its cost by
//! making it compete against the (a·b) multiplicative branch as a whole.
//!
//! Why 2-way not 3-way:
//!   `a` and `b` are NOT competitors — they are partners forming `a·b`. A
//!   softmax over (|a|, |b|, |c|) would pit them against each other and
//!   collapse the multiplicative signal. The conceptual unit of competition
//!   is (ab-branch) vs (c-branch), a 2-way budget.
//!
//! Mechanism:
//!   u_ab = (|a| + |b|) / 2
//!   [w_ab, w_c] = softmax([u_ab, |c|] / τ)   (weights sum to 1)
//!   a' = 2 · a · w_ab           (a and b share the SAME weight — preserves ab relationship)
//!   b' = 2 · b · w_ab
//!   c' = 2 · c · w_c
//!   co = Beukers(a' · b') / (1 + |c'|)
//!
//! If c is useless: network pushes w_c → 0; ab branch dominates; behavior
//!   degenerates toward B0 (plain Beukers on a,b).
//! If c is useful: network "pays" for c by giving it budget at the expense
//!   of the ab branch; the division term actually contributes.
//!
//! This directly answers: "does the c projection carry its weight?"
//!
//! A pure 3-way softmax(a,b,c) would be a secondary ablation — left for later.
//!
//! 2 variants × 3 seeds on FineWeb 30MB, ~8-10 min parallel.
//!
//! Run from repo root:
//!   cargo run --release --example diag_abc_budget -- <corpus_path>

use rayon::prelude::*;
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

/// Apply branch-budget softmax competition then α gate.
/// kind=0: B0 Beukers (2-proj)
/// kind=1: branch-budget α (3-proj with 2-way softmax competition between ab-branch and c-branch)
#[inline(always)]
fn gate(kind: usize, pv: &[f32]) -> f32 {
    match kind {
        0 => { let p = pv[0] * pv[1]; p / (1.0 + p.abs()) }
        1 => {
            // 2-way softmax: (ab-branch magnitude) vs (c-branch magnitude), τ=1.0
            let u_ab = (pv[0].abs() + pv[1].abs()) * 0.5;
            let u_c = pv[2].abs();
            let mx = u_ab.max(u_c);
            let e_ab = (u_ab - mx).exp();
            let e_c = (u_c - mx).exp();
            let s = e_ab + e_c;
            let w_ab = e_ab / s;
            let w_c = e_c / s;
            // Mean softmax weight over 2 branches = 0.5, so × 2 restores scale.
            // a and b share w_ab so their multiplicative relationship is preserved.
            let ap = 2.0 * pv[0] * w_ab;
            let bp = 2.0 * pv[1] * w_ab;
            let cp = 2.0 * pv[2] * w_c;
            // α formula: Beukers(a·b) / (1 + |c|)
            let prod = ap * bp;
            let beukers = prod / (1.0 + prod.abs());
            beukers / (1.0 + cp.abs())
        }
        _ => pv[0],
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
    // 2-way branch-budget weight telemetry: w_ab + w_c = 1.
    // If w_c is near 0: network pushed c out of budget → c unused.
    // If w_c is near 0.5: c is competitive with ab branch → c earned its cost.
    final_wab_mean: f32,
    final_wc_mean: f32,
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

    let total_p = n_proj * (nf * fan + nf) + 27 * nf + 27;
    let is_branch_budget = kind == 1;
    println!("[{}-s{}] start  variant={} nf={} proj={} params={}", tag, seed, variant, nf, n_proj, total_p);

    let samples = 2000usize;
    let max_ep = 300usize;
    let log_every = 50usize;
    let mut best_test = 0.0f64;
    let mut final_test = 0.0f64;
    let mut final_train = 0.0f64;
    let mut final_comax = 0.0f32;
    let mut final_wab = 0.0f32;
    let mut final_wc = 0.0f32;

    for ep in 0..max_ep {
        let lr = 0.01 * (1.0 - ep as f32 / max_ep as f32 * 0.8);
        let mut rt = Rng::new(seed.wrapping_mul(ep as u64 + 1).wrapping_add(42));
        let mut co_max_abs = 0.0f32;
        let mut wab_sum = 0.0f32;
        let mut wc_sum = 0.0f32;
        let mut w_count = 0usize;

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

            // Log 2-way branch-budget softmax weights for telemetry.
            if is_branch_budget {
                for f in 0..nf {
                    let pv = &projs_by_f[f];
                    let u_ab = (pv[0].abs() + pv[1].abs()) * 0.5;
                    let u_c = pv[2].abs();
                    let mx = u_ab.max(u_c);
                    let e_ab = (u_ab - mx).exp();
                    let e_c = (u_c - mx).exp();
                    let s = e_ab + e_c;
                    wab_sum += e_ab / s;
                    wc_sum += e_c / s;
                    w_count += 1;
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

            // Numerical gradient — works even through the softmax because the
            // whole gate function is just a pure function of pv.
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
            if w_count > 0 {
                final_wab = wab_sum / w_count as f32;
                final_wc = wc_sum / w_count as f32;
            }
            if is_branch_budget {
                println!("[{}-s{}]  ep={} tr={:.1} te={:.1} co_max={:.3}  w=(ab={:.3}, c={:.3})",
                    tag, seed, ep + 1, tr, te, co_max_abs, final_wab, final_wc);
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
        final_wab_mean: final_wab, final_wc_mean: final_wc,
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
    println!("=== branch-BUDGET GAUNTLET — corpus = {} ===", corpus_path);
    let corpus = load_corpus(&corpus_path);
    let split = corpus.len() * 80 / 100;
    println!("  corpus len = {}, split = {}/{}", corpus.len(), split, corpus.len() - split);
    println!("  300 epochs, 2000 samples/ep, 3 seeds per variant");
    println!();
    println!("  Variants (both iso-param):");
    println!("    B0              Beukers baseline                      nf=128  2-proj  ~32,411 params");
    println!("    branch-budget   α with softmax([u_ab, |c|]) · 2·w     nf=88   3-proj  ~32,235 params");
    println!("                    u_ab = (|a|+|b|)/2, a+b share w_ab");
    println!();

    let seeds: Vec<u64> = vec![42, 1337, 9999];
    let results = std::sync::Mutex::new(Vec::<RunResult>::new());
    let corpus_ref = &corpus;
    let results_ref = &results;

    rayon::scope(|s| {
        for &seed in &seeds {
            s.spawn(move |_| {
                let r = run("B0", corpus_ref, split, "Beukers baseline", 0, 2, 128, seed);
                results_ref.lock().unwrap().push(r);
            });
        }
        for &seed in &seeds {
            s.spawn(move |_| {
                let r = run("br-b", corpus_ref, split, "branch-budget softmax", 1, 3, 88, seed);
                results_ref.lock().unwrap().push(r);
            });
        }
    });

    let all = results.into_inner().unwrap();

    println!();
    println!("=== RAW RESULTS ===");
    for r in &all {
        if r.variant == "branch-budget softmax" {
            println!("  {:<24} seed={:>4}  best={:.2}  final_te={:.2}  final_tr={:.2}  co_max={:.3}  w=(ab={:.3}, c={:.3})",
                r.variant, r.seed, r.best_test, r.final_test, r.final_train, r.final_comax,
                r.final_wab_mean, r.final_wc_mean);
        } else {
            println!("  {:<24} seed={:>4}  best={:.2}  final_te={:.2}  final_tr={:.2}  co_max={:.3}",
                r.variant, r.seed, r.best_test, r.final_test, r.final_train, r.final_comax);
        }
    }

    println!();
    println!("=== SUMMARY (mean ± std, 3 seeds) ===");
    for var in &["Beukers baseline", "branch-budget softmax"] {
        let bests: Vec<f64> = all.iter().filter(|r| r.variant == *var).map(|r| r.best_test).collect();
        if !bests.is_empty() {
            let p = all.iter().find(|r| r.variant == *var).map(|r| r.params).unwrap_or(0);
            let seeds_str: Vec<String> = all.iter().filter(|r| r.variant == *var)
                .map(|r| format!("{:.2}", r.best_test)).collect();
            println!("  {:<26} (params={:>6})  best_test = {:.2} ± {:.2}  seeds: {:?}",
                var, p, mean(&bests), std_f(&bests), seeds_str);
        }
    }

    // Show branch-budget weight distribution — tells us if c earned its cost.
    let br_results: Vec<&RunResult> = all.iter().filter(|r| r.variant == "branch-budget softmax").collect();
    if !br_results.is_empty() {
        let wab: Vec<f64> = br_results.iter().map(|r| r.final_wab_mean as f64).collect();
        let wc: Vec<f64> = br_results.iter().map(|r| r.final_wc_mean as f64).collect();
        println!();
        println!("  branch-budget softmax weight distribution (final epoch, mean across seeds):");
        println!("    w_ab = {:.3}    (how much ab-branch is used)", mean(&wab));
        println!("    w_c  = {:.3}    (how much c-branch is used)", mean(&wc));
        println!("    (if w_c → 0: c was unused → gate degenerates toward pure Beukers)");
        println!("    (if w_c ≈ 0.5: c earned its budget → c is genuinely carrying its weight)");
    }

    println!();
    println!("  Total wallclock: {:.1}s", t0.elapsed().as_secs_f64());
}
