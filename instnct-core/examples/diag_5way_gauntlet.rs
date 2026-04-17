//! Diagnostic 5-way gauntlet — "közös nevező" intuition test.
//!
//! 5 variants side-by-side at nf=128 k=7, all rayon-parallel:
//!   B0 — Beukers baseline:    ab / (1 + |ab|)                                 [2-proj]
//!   NC — Original candidate:  ab / (|c| + ε)                                  [3-proj]
//!   α  — Modulated Beukers:   Beukers(ab) / (1 + |c|)                         [3-proj]
//!   γ  — Gated Beukers:       Beukers(ab) · sigmoid(c)                        [3-proj]
//!   δ  — Group-normed Beukers: Beukers(ab[f]) / (1 + mean_f(|Beukers(ab[f])|)) [2-proj]
//!
//! δ is the "user's intuition": divide each filter's output by the group's
//! shared activity level. No c projection — the group itself provides the
//! common reference.
//!
//! Backward for δ uses linearization: treat (1 + group_mean) as constant for
//! single-projection perturbation. This ignores the O(1/nf) feedback term
//! through the mean. Acceptable for a diagnostic at nf=128.
//!
//! Run from repo root: cargo run --example diag_5way_gauntlet --release

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

// Per-filter "raw" gate output. For δ this is just Beukers — the group-norm
// division happens after all filters are computed.
#[inline(always)]
fn gate_raw(gt: usize, pv: &[f32], eps_div: f32) -> f32 {
    match gt {
        0 => { let p = pv[0] * pv[1]; p / (1.0 + p.abs()) }                              // Beukers
        2 => { let p = pv[0] * pv[1]; p / (pv[2].abs() + eps_div) }                      // ab/(|c|+ε)
        3 => { let p = pv[0] * pv[1]; let b = p / (1.0 + p.abs()); b / (1.0 + pv[2].abs()) } // Modulated
        4 => {                                                                            // Gated (SwiGLU-like)
            let p = pv[0] * pv[1]; let b = p / (1.0 + p.abs());
            let s = 1.0 / (1.0 + (-pv[2]).exp());
            b * s
        }
        5 => { let p = pv[0] * pv[1]; p / (1.0 + p.abs()) }                              // Group-norm: raw = Beukers
        _ => pv[0],
    }
}

struct EpochStats {
    c_p1: f32, c_p50: f32, c_p99: f32, c_min: f32, c_max: f32,
    c_near_eps_frac: f32,
    co_max_abs: f32,
    gn_mean: f32,                 // group-mean for δ, 0 for others
    cgrad_norm_mean: f32,
    cgrad_norm_max: f32,
    cgrad_clipped_frac: f32,
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() { return 0.0; }
    let idx = ((sorted.len() - 1) as f32 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

struct RunResult {
    name: String,
    best_test: f64,
}

fn run(
    tag: &str,
    corpus: &[u8],
    split: usize,
    name: &str,
    gate_type: usize,
    n_proj: usize,
    nf: usize,
) -> RunResult {
    let ctx = 32; let mask_pos = ctx / 2; let k = 7; let hk = 3i32; let fan = k * DIM;
    let eps_div = 0.5f32;

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
    let mut hw: Vec<Vec<f32>> = (0..27).map(|_| (0..nf).map(|_| rng.normal() * sc_h).collect()).collect();
    let mut hb = vec![0.0f32; 27];

    let total_p = n_proj * (nf * fan + nf) + 27 * nf + 27;
    let cproj_idx = if gate_type == 2 || gate_type == 3 || gate_type == 4 { 2usize } else { usize::MAX };
    let is_groupnorm = gate_type == 5;

    println!("[{}] === {} === (gate={}, n_proj={}, nf={}, p={})",
        tag, name, gate_type, n_proj, nf, total_p);
    println!("[{}]   {:>4} {:>6} {:>6} | {:>8} {:>8} {:>8} {:>8} {:>8} | {:>6} {:>7} {:>7} | {:>7} {:>7} {:>5}",
        tag, "ep", "train", "test",
        "c_p1", "c_p50", "c_p99", "c_min", "c_max",
        "c<2ε%", "co_max", "gn_mean", "cg_mean", "cg_max", "clip%");

    let samples = 2000usize;
    let max_ep = 300usize;
    let log_every = 25usize;
    let mut best_test = 0.0f64;

    let mut c_values: Vec<f32> = Vec::with_capacity(samples * nf);
    let mut co_max_abs = 0.0f32;
    let mut gn_sum = 0.0f32;
    let mut gn_count = 0usize;
    let cgrad_norms: Vec<f32> = Vec::new();
    let cgrad_clipped = 0usize;
    let mut c_near_eps = 0usize;
    let mut c_total = 0usize;

    for ep in 0..max_ep {
        let lr = 0.01 * (1.0 - ep as f32 / max_ep as f32 * 0.8);
        let mut rt = Rng::new(ep as u64 * 1000 + 42);

        if ep % log_every == 0 {
            c_values.clear(); co_max_abs = 0.0; gn_sum = 0.0; gn_count = 0;
            c_near_eps = 0; c_total = 0;
        }

        for _ in 0..samples {
            let off = rt.range(0, split.saturating_sub(ctx + 1));
            let chunk = &corpus[off..off + ctx];
            let target = chunk[mask_pos] as usize;
            let emb: Vec<[f32; DIM]> = (0..ctx).map(|i|
                if i == mask_pos { [0.0; DIM] } else { embed[chunk[i] as usize] }
            ).collect();

            // Forward: per-filter projections in parallel.
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

            // Log |c| stats for 3-proj variants.
            if cproj_idx < n_proj {
                for f in 0..nf {
                    let ac = projs_by_f[f][cproj_idx].abs();
                    c_values.push(ac);
                    if ac < 2.0 * eps_div { c_near_eps += 1; }
                    c_total += 1;
                }
            }

            // Raw gate output per filter.
            let raw: Vec<f32> = (0..nf).into_par_iter().map(|f| {
                gate_raw(gate_type, &projs_by_f[f], eps_div)
            }).collect();

            // For δ: apply group normalization.
            let (co, group_denom) = if is_groupnorm {
                let mean = raw.iter().map(|x| x.abs()).sum::<f32>() / nf as f32;
                let denom = 1.0 + mean;
                let co: Vec<f32> = raw.iter().map(|&r| (r / denom).max(-10.0).min(10.0)).collect();
                (co, denom)
            } else {
                let co: Vec<f32> = raw.iter().map(|&r| r.max(-10.0).min(10.0)).collect();
                (co, 1.0f32)
            };
            if is_groupnorm {
                gn_sum += group_denom - 1.0; // log the mean part, not the +1
                gn_count += 1;
            }

            let local_co_max = co.par_iter().map(|x| x.abs()).reduce(|| 0.0f32, f32::max);
            if local_co_max > co_max_abs { co_max_abs = local_co_max; }

            // Softmax head.
            let mut logits = hb.clone();
            for c in 0..27 {
                for f in 0..nf { logits[c] += hw[c][f] * co[f]; }
            }
            let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut pr = vec![0.0f32; 27]; let mut s = 0.0f32;
            for c in 0..27 { pr[c] = (logits[c] - mx).exp(); s += pr[c]; }
            for c in 0..27 { pr[c] /= s; }
            pr[target] -= 1.0;

            // Head backward.
            let mut dc = vec![0.0f32; nf];
            for c in 0..27 {
                for f in 0..nf {
                    dc[f] += pr[c] * hw[c][f];
                    hw[c][f] -= lr * pr[c] * co[f];
                }
                hb[c] -= lr * pr[c];
            }

            // Per-filter numerical gradient, parallel over f.
            // For δ, we use linearization: treat group_denom as constant, divide gradient by it.
            // This ignores the O(1/nf) feedback through the group mean.
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
                        let co_plus = gate_raw(gate_type, &pv, eps_div) / group_denom;
                        pv[p] = old - eps_g;
                        let co_minus = gate_raw(gate_type, &pv, eps_div) / group_denom;
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

            // Apply (no clip in this gauntlet — we're measuring natural dynamics).
            for (f, (w_grads, b_grads, embed_deltas)) in per_f.iter().enumerate() {
                for p in 0..n_proj {
                    bs[p][f] -= lr * b_grads[p];
                    for w in 0..fan {
                        ws[p][f][w] -= lr * w_grads[p][w];
                    }
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
                        raw_e[f] = gate_raw(gate_type, &pv, eps_div);
                    }
                    let denom_e = if is_groupnorm {
                        1.0 + raw_e.iter().map(|x| x.abs()).sum::<f32>() / nf as f32
                    } else { 1.0 };
                    let co_e: Vec<f32> = raw_e.iter().map(|&r| (r / denom_e).max(-10.0).min(10.0)).collect();
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

            let stats = if cproj_idx < n_proj && !c_values.is_empty() {
                let mut sorted = c_values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let cgn_mean = if cgrad_norms.is_empty() { 0.0 } else {
                    cgrad_norms.iter().sum::<f32>() / cgrad_norms.len() as f32
                };
                let cgn_max = cgrad_norms.iter().cloned().fold(0.0f32, f32::max);
                let clip_frac = if cgrad_norms.is_empty() { 0.0 }
                    else { cgrad_clipped as f32 / cgrad_norms.len() as f32 };
                let near_frac = if c_total == 0 { 0.0 } else { c_near_eps as f32 / c_total as f32 };
                let gn_m = if gn_count == 0 { 0.0 } else { gn_sum / gn_count as f32 };
                EpochStats {
                    c_p1: percentile(&sorted, 0.01),
                    c_p50: percentile(&sorted, 0.50),
                    c_p99: percentile(&sorted, 0.99),
                    c_min: *sorted.first().unwrap_or(&0.0),
                    c_max: *sorted.last().unwrap_or(&0.0),
                    c_near_eps_frac: near_frac,
                    co_max_abs, gn_mean: gn_m,
                    cgrad_norm_mean: cgn_mean,
                    cgrad_norm_max: cgn_max,
                    cgrad_clipped_frac: clip_frac,
                }
            } else {
                let gn_m = if gn_count == 0 { 0.0 } else { gn_sum / gn_count as f32 };
                EpochStats { c_p1: 0.0, c_p50: 0.0, c_p99: 0.0, c_min: 0.0, c_max: 0.0,
                    c_near_eps_frac: 0.0, co_max_abs, gn_mean: gn_m,
                    cgrad_norm_mean: 0.0, cgrad_norm_max: 0.0, cgrad_clipped_frac: 0.0 }
            };

            println!("[{}]   {:>4} {:>6.1} {:>6.1} | {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} | {:>5.1}% {:>7.3} {:>7.3} | {:>7.3} {:>7.3} {:>4.1}%",
                tag, ep + 1, tr, te,
                stats.c_p1, stats.c_p50, stats.c_p99, stats.c_min, stats.c_max,
                stats.c_near_eps_frac * 100.0, stats.co_max_abs, stats.gn_mean,
                stats.cgrad_norm_mean, stats.cgrad_norm_max, stats.cgrad_clipped_frac * 100.0);
        }
    }

    println!("[{}]   best_test = {:.2}%", tag, best_test);
    RunResult { name: name.to_string(), best_test }
}

fn main() {
    let t0 = Instant::now();
    // Corpus path: optional CLI arg, defaults to the Alice fixture.
    let corpus_path = std::env::args().nth(1)
        .unwrap_or_else(|| "instnct-core/tests/fixtures/alice_corpus.txt".to_string());
    println!("=== DIAG 5-WAY — corpus = {} ===", corpus_path);
    let corpus = load_corpus(&corpus_path);
    let split = corpus.len() * 80 / 100;
    let nf = 128;

    println!("=== DIAG 5-WAY — közös-nevező gauntlet (rayon parallel) ===");
    println!("  threads available = {}", rayon::current_num_threads());
    println!("  corpus = {} chars, split = {}/{} (80/20)", corpus.len(), split, corpus.len() - split);
    println!("  config = nf={} k=7 ctx=32 samples/ep=2000 max_ep=300 eps_div=0.5", nf);
    println!("  tags:");
    println!("    B0 = Beukers baseline       ab/(1+|ab|)                            [2-proj, 32411 p]");
    println!("    NC = Original candidate     ab/(|c|+ε)                             [3-proj, 46875 p]");
    println!("    α  = Modulated Beukers      Beukers(ab)/(1+|c|)                    [3-proj, 46875 p]");
    println!("    γ  = Gated Beukers          Beukers(ab)·sigmoid(c)                 [3-proj, 46875 p]");
    println!("    δ  = Group-norm Beukers     Beukers(ab)/(1+mean_f(|Beukers(ab)|))  [2-proj, 32411 p]");
    println!();

    let results = std::sync::Mutex::new(Vec::<RunResult>::new());
    rayon::scope(|s| {
        s.spawn(|_| {
            let r = run("B0", &corpus, split, "Beukers baseline", 0, 2, nf);
            results.lock().unwrap().push(r);
        });
        s.spawn(|_| {
            let r = run("NC", &corpus, split, "Original ab/(|c|+ε)", 2, 3, nf);
            results.lock().unwrap().push(r);
        });
        s.spawn(|_| {
            let r = run("α", &corpus, split, "Modulated Beukers", 3, 3, nf);
            results.lock().unwrap().push(r);
        });
        s.spawn(|_| {
            let r = run("γ", &corpus, split, "Gated Beukers (SwiGLU-like)", 4, 3, nf);
            results.lock().unwrap().push(r);
        });
        s.spawn(|_| {
            let r = run("δ", &corpus, split, "Group-norm Beukers", 5, 2, nf);
            results.lock().unwrap().push(r);
        });
    });

    println!();
    println!("=== FINAL SUMMARY ===");
    let rs = results.into_inner().unwrap();
    // Sort by best_test descending.
    let mut rs_sorted = rs.clone();
    rs_sorted.sort_by(|a, b| b.best_test.partial_cmp(&a.best_test).unwrap_or(std::cmp::Ordering::Equal));
    for r in &rs_sorted {
        println!("  {:<40}  best_test = {:.2}%", r.name, r.best_test);
    }
    println!();
    println!("  Total wallclock: {:.1}s", t0.elapsed().as_secs_f64());
}

impl Clone for RunResult {
    fn clone(&self) -> Self {
        RunResult { name: self.name.clone(), best_test: self.best_test }
    }
}
