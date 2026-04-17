//! Quantization sweep FILL: ternary mode + Code nf=64 dark spots.
//!
//! Adds:
//!   - Ternary {-1, 0, +1} staged INQ (BitNet b1.58 territory)
//!   - Code nf=64 cell (missing from original sweep)
//!
//! Target cell list (11 runs total):
//!   FineWeb: nf={32,64,96,128} × ternary         (4 runs)
//!   Code:    nf={32,96,128}    × ternary         (3 runs)
//!   Code:    nf=64             × {float,int4,binary,ternary}  (4 runs)
//!
//! Run: cargo run --release --example diag_quant_sweep_fill -- <fineweb> <code_corpus>

use rayon::prelude::*;
use std::time::Instant;

const VOCAB: usize = 2000;
const DIM: usize = 16;
const ROUNDS: usize = 10;
const EPOCHS_PER_ROUND: usize = 20;

#[derive(Clone, Copy, PartialEq, Debug)]
enum QuantMode { Float, Int4, Binary, Ternary }
impl QuantMode {
    fn tag(&self) -> &'static str {
        match self {
            QuantMode::Float => "float",
            QuantMode::Int4 => "int4",
            QuantMode::Binary => "binary",
            QuantMode::Ternary => "ternary",
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

const INT4_LEVELS: f32 = 7.0;
#[inline(always)]
fn q_int4(x: f32, scale: f32) -> f32 {
    let q = (x / scale * INT4_LEVELS).round().clamp(-INT4_LEVELS, INT4_LEVELS);
    q * scale / INT4_LEVELS
}
#[inline(always)]
fn q_binary(x: f32, scale: f32) -> f32 {
    if x >= 0.0 { scale } else { -scale }
}
#[inline(always)]
fn q_ternary(x: f32, scale: f32) -> f32 {
    // Standard TWN: threshold at scale/2 → {-scale, 0, +scale}
    let thr = scale * 0.5;
    if x > thr { scale } else if x < -thr { -scale } else { 0.0 }
}
#[inline(always)]
fn apply_q(x: f32, scale: f32, mode: QuantMode) -> f32 {
    match mode {
        QuantMode::Float => x,
        QuantMode::Int4 => q_int4(x, scale),
        QuantMode::Binary => q_binary(x, scale),
        QuantMode::Ternary => q_ternary(x, scale),
    }
}
#[inline(always)]
fn q_error(x: f32, scale: f32, mode: QuantMode) -> f32 {
    (x - apply_q(x, scale, mode)).abs()
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

#[derive(Clone, Debug)]
struct RunResult {
    task: String,
    nf: usize,
    mode: QuantMode,
    #[allow(dead_code)]
    float_acc: f64,
    final_acc: f64,
    #[allow(dead_code)]
    ep_plateau: usize,
}

#[allow(clippy::too_many_arguments)]
fn run_one(
    task: &str, corpus: &[u8], split: usize,
    nf: usize, mode: QuantMode, seed: u64,
) -> RunResult {
    let fan = 7 * DIM;
    let ctx = 32; let mask_pos = ctx / 2; let k = 7; let hk = 3i32;
    let n_proj = 2;
    let samples = 2000;

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

    let ws_none: Vec<Vec<Vec<bool>>> =
        (0..n_proj).map(|_| (0..nf).map(|_| vec![false; fan]).collect()).collect();
    let hw_none: Vec<Vec<bool>> = (0..27).map(|_| vec![false; nf]).collect();

    // Phase 1: float plateau
    let max_ep_init = 200;
    let patience = 30;
    let log_every = 10;
    let mut best_test = 0.0f64;
    let mut no_improve = 0usize;
    let mut final_ep = 0usize;

    println!("[{} nf={} {}] Phase 1: float plateau", task, nf, mode.tag());
    for ep in 0..max_ep_init {
        let lr = 0.01 * (1.0 - ep as f32 / max_ep_init as f32 * 0.8);
        train_one_epoch(corpus, split,
                        &mut embed, &mut ws, &mut bs, &mut hw, &mut hb,
                        &ws_none, &hw_none,
                        lr, seed, ep as u64, samples,
                        ctx, mask_pos, k, hk, n_proj, nf);
        final_ep = ep + 1;
        if (ep + 1) % log_every == 0 {
            let te = eval(corpus, split, corpus.len(), &ws, &bs, &hw, &hb, &embed,
                         ctx, mask_pos, k, hk, n_proj, nf, 1000);
            if te > best_test + 0.01 {
                best_test = te;
                no_improve = 0;
            } else {
                no_improve += log_every;
            }
            if no_improve >= patience { break; }
        }
    }
    let acc_float = eval(corpus, split, corpus.len(), &ws, &bs, &hw, &hb, &embed,
                         ctx, mask_pos, k, hk, n_proj, nf, 1000);
    println!("[{} nf={} {}] float done te={:.2} @ ep={}", task, nf, mode.tag(), acc_float, final_ep);

    if mode == QuantMode::Float {
        return RunResult {
            task: task.to_string(), nf, mode,
            float_acc: acc_float, final_acc: acc_float, ep_plateau: final_ep,
        };
    }

    // Phase 2: staged quantization
    let max_ws = ws.iter().flat_map(|l| l.iter().flat_map(|r| r.iter()))
                     .map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-9);
    let max_hw = hw.iter().flat_map(|r| r.iter())
                     .map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-9);

    let mut ws_frozen: Vec<Vec<Vec<bool>>> =
        (0..n_proj).map(|_| (0..nf).map(|_| vec![false; fan]).collect()).collect();
    let mut hw_frozen: Vec<Vec<bool>> =
        (0..27).map(|_| vec![false; nf]).collect();

    let total_ws = n_proj * nf * fan;
    let total_hw = 27 * nf;
    let total_params = total_ws + total_hw;
    let per_round = total_params / ROUNDS;

    println!("[{} nf={} {}] Phase 2: staged ({} rounds × {} ep)", task, nf, mode.tag(), ROUNDS, EPOCHS_PER_ROUND);
    for round in 1..=ROUNDS {
        let mut err_list: Vec<(f32, usize)> = Vec::with_capacity(total_params);
        for p in 0..n_proj {
            for f in 0..nf {
                for i in 0..fan {
                    if !ws_frozen[p][f][i] {
                        let idx = p * nf * fan + f * fan + i;
                        err_list.push((q_error(ws[p][f][i], max_ws, mode), idx));
                    }
                }
            }
        }
        for c in 0..27 {
            for f in 0..nf {
                if !hw_frozen[c][f] {
                    let idx = total_ws + c * nf + f;
                    err_list.push((q_error(hw[c][f], max_hw, mode), idx));
                }
            }
        }
        err_list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let to_freeze = per_round.min(err_list.len());
        for &(_, idx) in err_list.iter().take(to_freeze) {
            if idx < total_ws {
                let p = idx / (nf * fan);
                let f = (idx / fan) % nf;
                let i = idx % fan;
                ws[p][f][i] = apply_q(ws[p][f][i], max_ws, mode);
                ws_frozen[p][f][i] = true;
            } else {
                let hi = idx - total_ws;
                let c = hi / nf;
                let f = hi % nf;
                hw[c][f] = apply_q(hw[c][f], max_hw, mode);
                hw_frozen[c][f] = true;
            }
        }

        for ep in 0..EPOCHS_PER_ROUND {
            let lr = 0.005 * (1.0 - ep as f32 / EPOCHS_PER_ROUND as f32 * 0.5);
            let ep_total = (final_ep + (round - 1) * EPOCHS_PER_ROUND + ep) as u64;
            train_one_epoch(corpus, split,
                            &mut embed, &mut ws, &mut bs, &mut hw, &mut hb,
                            &ws_frozen, &hw_frozen,
                            lr, seed, ep_total, samples,
                            ctx, mask_pos, k, hk, n_proj, nf);
        }
    }
    let acc_final = eval(corpus, split, corpus.len(), &ws, &bs, &hw, &hb, &embed,
                         ctx, mask_pos, k, hk, n_proj, nf, 1000);
    println!("[{} nf={} {}] DONE final te={:.2} (vs float {:+.2}pp)",
             task, nf, mode.tag(), acc_final, acc_final - acc_float);

    RunResult {
        task: task.to_string(), nf, mode,
        float_acc: acc_float, final_acc: acc_final, ep_plateau: final_ep,
    }
}

fn main() {
    let t0 = Instant::now();
    let fineweb_path = std::env::args().nth(1)
        .unwrap_or_else(|| "S:/AI/MESSY TRAINING DATA - INPUT ONLY/Fineweb edu 10B/fineweb_edu_30m.txt".to_string());
    let code_path = std::env::args().nth(2)
        .unwrap_or_else(|| "instnct-core/tests/fixtures/code_corpus.txt".to_string());

    println!("=== QUANT SWEEP FILL: ternary + Code nf=64 dark spots ===");
    println!("   FineWeb: {}", fineweb_path);
    println!("   Code:    {}", code_path);
    println!();

    let fineweb_corpus = load_corpus(&fineweb_path);
    let fineweb_split = fineweb_corpus.len() * 80 / 100;
    let code_corpus = load_corpus(&code_path);
    let code_split = code_corpus.len() * 80 / 100;

    println!("   FineWeb: {} filtered bytes, split {}/{}",
             fineweb_corpus.len(), fineweb_split, fineweb_corpus.len() - fineweb_split);
    println!("   Code:    {} filtered bytes, split {}/{}",
             code_corpus.len(), code_split, code_corpus.len() - code_split);
    println!();

    let seed = 42u64;

    // Targeted cell list: 11 missing cells
    //   (task_tag, nf, mode)
    let cells: Vec<(&str, usize, QuantMode)> = vec![
        // FineWeb ternary across all 4 sizes
        ("FineWeb", 32,  QuantMode::Ternary),
        ("FineWeb", 64,  QuantMode::Ternary),
        ("FineWeb", 96,  QuantMode::Ternary),
        ("FineWeb", 128, QuantMode::Ternary),
        // Code ternary (3 existing sizes)
        ("Code",    32,  QuantMode::Ternary),
        ("Code",    96,  QuantMode::Ternary),
        ("Code",    128, QuantMode::Ternary),
        // Code nf=64 dark spot (all 4 modes)
        ("Code",    64,  QuantMode::Float),
        ("Code",    64,  QuantMode::Int4),
        ("Code",    64,  QuantMode::Binary),
        ("Code",    64,  QuantMode::Ternary),
    ];

    println!("  Total cells to fill: {}", cells.len());
    println!();

    let mut all_results: Vec<RunResult> = Vec::new();
    for (task_tag, nf, mode) in cells.iter() {
        let (corpus, split) = match *task_tag {
            "FineWeb" => (&fineweb_corpus[..], fineweb_split),
            "Code" => (&code_corpus[..], code_split),
            _ => panic!("Unknown task: {}", task_tag),
        };
        let r = run_one(task_tag, corpus, split, *nf, *mode, seed);
        all_results.push(r);
        println!();
    }

    // Summary
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  SUMMARY — fill sweep (ternary + Code nf=64)               ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    println!("  {:<8} {:>4} {:>10} {:>12}",
             "task", "nf", "mode", "final_acc");
    println!("  {:-<8} {:->4} {:->10} {:->12}", "", "", "", "");
    for r in &all_results {
        println!("  {:<8} {:>4} {:>10} {:>12.2}",
                 r.task, r.nf, r.mode.tag(), r.final_acc);
    }

    println!();
    println!("  Total wallclock: {:.1}s", t0.elapsed().as_secs_f64());
}
