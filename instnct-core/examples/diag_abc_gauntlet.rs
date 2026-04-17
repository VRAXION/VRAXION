//! Diagnostic A: c-projection health logger + mandatory grad-norm clip for ab/(|c|+ε).
//!
//! Target config: nf=128 k=7 ctx=32 (the candidate's "promised" setup from SESSION_BRIEFING.md).
//! Runs Beukers (2-proj) + ab/(|c|+ε) (3-proj) SIDE-BY-SIDE with full instrumentation.
//!
//! Rayon parallelism:
//!   - 3 configs run concurrently via rayon::scope
//!   - Inside each config, per-filter forward projection and backward gradient
//!     computation parallelized via par_iter (work-stealing nested pool)
//!   - Embed gradients are collected thread-locally then merged sequentially
//!     per sample (no shared-state writes from parallel workers)
//!
//! Per 25-epoch checkpoint we log:
//!   - train/test masked-char accuracy
//!   - |c| percentile ladder (p1, p50, p99, min, max)  — denominator health
//!   - fraction of samples where |c| < 2*ε_div         — near-epsilon hit rate
//!   - max |co[f]| across the batch                    — gate-output saturation / outlier
//!   - c-projection grad norm before clip (mean/max)   — gradient explosion diagnosis
//!   - fraction of samples whose c-proj grad was clipped
//!
//! c-projection grad-norm clip ≤ 1.0 is applied PER SAMPLE, INDEPENDENT of any global clip.
//! This is the NRU/NMRU prescription (arXiv 2110.05177) — mandatory for division-style units.
//!
//! Run from repo root: cargo run --example diag_abc_gauntlet --release

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

#[inline(always)]
fn gate(gate_type: usize, pv: &[f32], eps_div: f32) -> f32 {
    match gate_type {
        0 => { let p = pv[0] * pv[1]; p / (1.0 + p.abs()) }                    // Beukers
        2 => { let p = pv[0] * pv[1]; p / (pv[2].abs() + eps_div) }            // ab/(|c|+ε)
        _ => pv[0],
    }
}

struct EpochStats {
    c_p1: f32, c_p50: f32, c_p99: f32, c_min: f32, c_max: f32,
    c_near_eps_frac: f32,
    co_max_abs: f32,
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
    lines: Vec<String>,
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
    enable_cproj_clip: bool,
) -> RunResult {
    let ctx = 32; let mask_pos = ctx / 2; let k = 7; let hk = 3i32; let fan = k * DIM;
    let eps_div = 0.5f32;
    let cproj_clip_norm = 1.0f32;

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
    let cproj_idx = if gate_type == 2 { 2usize } else { usize::MAX };

    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("[{}] === {} === (gate={}, n_proj={}, nf={}, p={}, clip={})",
        tag, name, gate_type, n_proj, nf, total_p, if enable_cproj_clip { "on" } else { "off" }));
    lines.push(format!("[{}]   {:>4} {:>6} {:>6} | {:>8} {:>8} {:>8} {:>8} {:>8} | {:>6} {:>8} | {:>8} {:>8} {:>6}",
        tag, "ep", "train", "test",
        "c_p1", "c_p50", "c_p99", "c_min", "c_max",
        "c<2ε%", "co_max", "cg_mean", "cg_max", "clip%"));

    // Stream the header rows immediately so the user sees the run has started.
    println!("{}", lines[0]);
    println!("{}", lines[1]);

    let samples = 2000usize;
    let max_ep = 300usize;
    let log_every = 25usize;
    let mut best_test = 0.0f64;

    let mut c_values: Vec<f32> = Vec::with_capacity(samples * nf);
    let mut co_max_abs = 0.0f32;
    let mut cgrad_norms: Vec<f32> = Vec::with_capacity(samples);
    let mut cgrad_clipped = 0usize;
    let mut c_near_eps = 0usize;
    let mut c_total = 0usize;

    for ep in 0..max_ep {
        let lr = 0.01 * (1.0 - ep as f32 / max_ep as f32 * 0.8);
        let mut rt = Rng::new(ep as u64 * 1000 + 42);

        if ep % log_every == 0 {
            c_values.clear(); co_max_abs = 0.0; cgrad_norms.clear();
            cgrad_clipped = 0; c_near_eps = 0; c_total = 0;
        }

        for _ in 0..samples {
            let off = rt.range(0, split.saturating_sub(ctx + 1));
            let chunk = &corpus[off..off + ctx];
            let target = chunk[mask_pos] as usize;
            let emb: Vec<[f32; DIM]> = (0..ctx).map(|i|
                if i == mask_pos { [0.0; DIM] } else { embed[chunk[i] as usize] }
            ).collect();

            // ---- Forward: per-filter projections, parallel over f ----
            // Result layout: projs_by_f[f][p]
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

            // Log |c| stats.
            if cproj_idx < n_proj {
                for f in 0..nf {
                    let ac = projs_by_f[f][cproj_idx].abs();
                    c_values.push(ac);
                    if ac < 2.0 * eps_div { c_near_eps += 1; }
                    c_total += 1;
                }
            }

            // Gate output co[] + track co_max_abs.
            let co: Vec<f32> = (0..nf).into_par_iter().map(|f| {
                gate(gate_type, &projs_by_f[f], eps_div).max(-10.0).min(10.0)
            }).collect();
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

            // Backprop through head: dc[f] + hw/hb updates.
            let mut dc = vec![0.0f32; nf];
            for c in 0..27 {
                for f in 0..nf {
                    dc[f] += pr[c] * hw[c][f];
                    hw[c][f] -= lr * pr[c] * co[f];
                }
                hb[c] -= lr * pr[c];
            }

            // ---- Backward: per-filter numerical gradient, parallel over f ----
            // Each task produces: (w_grads[p][fan], b_grads[p], embed_deltas: Vec<(char, d, delta)>)
            // w_grads and b_grads are applied sequentially after — unique per (p, f) so no race.
            // embed_deltas are merged sequentially per sample — only 27 possible chars so tiny.
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
                        let co_plus = gate(gate_type, &pv, eps_div);
                        pv[p] = old - eps_g;
                        let co_minus = gate(gate_type, &pv, eps_div);
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

            // ---- Apply: c-projection with norm clip, others unconditional ----
            let stage_c = cproj_idx < n_proj && enable_cproj_clip;
            let cproj_scale = if stage_c {
                let mut sq = 0.0f32;
                for (w_grads, b_grads, _) in &per_f {
                    for w in &w_grads[cproj_idx] { sq += w * w; }
                    sq += b_grads[cproj_idx] * b_grads[cproj_idx];
                }
                let norm = sq.sqrt();
                cgrad_norms.push(norm);
                if norm > cproj_clip_norm { cgrad_clipped += 1; cproj_clip_norm / norm } else { 1.0 }
            } else { 1.0 };

            for (f, (w_grads, b_grads, embed_deltas)) in per_f.iter().enumerate() {
                for p in 0..n_proj {
                    let scale = if p == cproj_idx && stage_c { cproj_scale } else { 1.0 };
                    bs[p][f] -= lr * scale * b_grads[p];
                    for w in 0..fan {
                        ws[p][f][w] -= lr * scale * w_grads[p][w];
                    }
                }
                for &(cidx, d, delta) in embed_deltas {
                    let p_of_delta = 0; // embed deltas are scaled once; scale is per-(p, f)
                    // NOTE: embed deltas were pre-scaled by /n_proj; the c-proj clip shouldn't apply
                    // to them because embed is shared across projections. Keep apply unscaled.
                    let _ = p_of_delta;
                    embed[cidx][d] -= lr * delta;
                }
            }
        }

        if (ep + 1) % log_every == 0 || ep + 1 == max_ep {
            // Evaluate.
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
                        co_e[f] = gate(gate_type, &pv, eps_div).max(-10.0).min(10.0);
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
                EpochStats {
                    c_p1: percentile(&sorted, 0.01),
                    c_p50: percentile(&sorted, 0.50),
                    c_p99: percentile(&sorted, 0.99),
                    c_min: *sorted.first().unwrap_or(&0.0),
                    c_max: *sorted.last().unwrap_or(&0.0),
                    c_near_eps_frac: near_frac,
                    co_max_abs,
                    cgrad_norm_mean: cgn_mean,
                    cgrad_norm_max: cgn_max,
                    cgrad_clipped_frac: clip_frac,
                }
            } else {
                EpochStats { c_p1: 0.0, c_p50: 0.0, c_p99: 0.0, c_min: 0.0, c_max: 0.0,
                    c_near_eps_frac: 0.0, co_max_abs, cgrad_norm_mean: 0.0, cgrad_norm_max: 0.0,
                    cgrad_clipped_frac: 0.0 }
            };

            let line = format!("[{}]   {:>4} {:>6.1} {:>6.1} | {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} | {:>5.1}% {:>8.3} | {:>8.3} {:>8.3} {:>5.1}%",
                tag, ep + 1, tr, te,
                stats.c_p1, stats.c_p50, stats.c_p99, stats.c_min, stats.c_max,
                stats.c_near_eps_frac * 100.0, stats.co_max_abs,
                stats.cgrad_norm_mean, stats.cgrad_norm_max, stats.cgrad_clipped_frac * 100.0);
            println!("{}", line);
            lines.push(line);
        }
    }

    let final_line = format!("[{}]   best_test = {:.2}%", tag, best_test);
    println!("{}", final_line);
    lines.push(final_line);

    RunResult { name: name.to_string(), lines, best_test }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let split = corpus.len() * 80 / 100;
    let nf = 128;

    println!("=== DIAG A — c-proj health + grad-norm clip (rayon parallel) ===");
    println!("  threads available = {}", rayon::current_num_threads());
    println!("  corpus = {} chars, split = {}/{} (80/20)", corpus.len(), split, corpus.len() - split);
    println!("  config = nf={} k=7 ctx=32 samples/ep=2000 max_ep=300 eps_div=0.5 clip_norm=1.0", nf);
    println!("  tags: B0=Beukers reference, NC=ab/(|c|+ε) NO clip, CL=ab/(|c|+ε) CLIP on");
    println!();

    // Run 3 configs in parallel threads. Each has fully independent state.
    // Output lines interleave but each line is self-tagged.
    let results = std::sync::Mutex::new(Vec::<RunResult>::new());
    rayon::scope(|s| {
        s.spawn(|_| {
            let r = run("B0", &corpus, split, "Beukers 2-proj (reference)", 0, 2, nf, false);
            results.lock().unwrap().push(r);
        });
        s.spawn(|_| {
            let r = run("NC", &corpus, split, "ab/(|c|+ε) 3-proj — NO clip", 2, 3, nf, false);
            results.lock().unwrap().push(r);
        });
        s.spawn(|_| {
            let r = run("CL", &corpus, split, "ab/(|c|+ε) 3-proj — CLIP on", 2, 3, nf, true);
            results.lock().unwrap().push(r);
        });
    });

    // Final summary.
    println!();
    println!("=== FINAL SUMMARY ===");
    let rs = results.into_inner().unwrap();
    for r in &rs {
        println!("  {:<40}  best_test = {:.2}%", r.name, r.best_test);
    }
    println!();
    println!("  Total wallclock: {:.1}s", t0.elapsed().as_secs_f64());
}
