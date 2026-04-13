// c19_hot_sweep — diagnostic exhaustive search in the hot backprop buckets
//
// For a fixed (parents, weights, threshold) setup, this tool:
//   1. Reproduces the c19_grower data generator byte-for-byte
//   2. Runs `finetune_seeds` parallel finite-diff finetunes on (c, rho)
//      and logs ALL seed endpoints (not just the best)
//   3. For each seed endpoint, runs a dense local grid search around it
//   4. Reports the global top-N dense result + compares to the c19_grower
//      sparse 10×6=60 quant grid
//
// This file is standalone — it does NOT modify c19_grower.rs. All code that
// overlaps is copied verbatim so numeric results are guaranteed identical.

use rayon::prelude::*;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

// ══════════════════════════════════════════════════════
// RNG — verbatim from c19_grower.rs
// ══════════════════════════════════════════════════════
struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 {
        self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.s
    }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn range(&mut self, lo: f32, hi: f32) -> f32 { lo + self.f32() * (hi - lo) }
    fn bool_p(&mut self, p: f32) -> bool { self.f32() < p }
}

// ══════════════════════════════════════════════════════
// C19 — verbatim from c19_grower.rs
// ══════════════════════════════════════════════════════
fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.1);
    let rho = rho.max(0.0);
    let l = 6.0 * c;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let scaled = x / c;
    let n = scaled.floor();
    let t = scaled - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sgn * h + rho * h * h)
}

// ══════════════════════════════════════════════════════
// FONT + DATA — verbatim from c19_grower.rs
// ══════════════════════════════════════════════════════
const FONT: [[u8; 9]; 10] = [
    [1,1,1, 1,0,1, 1,1,1], [0,1,0, 0,1,0, 0,1,0],
    [1,1,0, 0,1,0, 0,1,1], [1,1,0, 0,1,0, 1,1,0],
    [1,0,1, 1,1,1, 0,0,1], [0,1,1, 0,1,0, 1,1,0],
    [1,0,0, 1,1,0, 1,1,0], [1,1,1, 0,0,1, 0,0,1],
    [1,1,1, 1,1,1, 1,1,1], [1,1,1, 1,1,1, 0,1,1],
];

struct Data { train: Vec<(Vec<u8>, u8)>, val: Vec<(Vec<u8>, u8)>, test: Vec<(Vec<u8>, u8)> }

fn gen_data(
    label_fn: &dyn Fn(usize, &[u8]) -> Option<u8>,
    noise: f32,
    n_per: usize,
    seed: u64,
) -> Data {
    let mut rng = Rng::new(seed);
    let (mut tr, mut va, mut te) = (Vec::new(), Vec::new(), Vec::new());
    for d in 0..10 { for i in 0..n_per {
        let mut px = FONT[d].to_vec();
        for p in px.iter_mut() { if rng.bool_p(noise) { *p = 1 - *p; } }
        if let Some(label) = label_fn(d, &px) {
            match i % 5 { 0 => va.push((px, label)), 1 => te.push((px, label)), _ => tr.push((px, label)) }
        }
    }}
    Data { train: tr, val: va, test: te }
}

// ══════════════════════════════════════════════════════
// c19_weighted_mse — verbatim from c19_grower.rs
// ══════════════════════════════════════════════════════
fn c19_weighted_mse(dots: &[i32], targets: &[f32], sw: &[f32], c: f32, rho: f32) -> f32 {
    let mut l = 0.0f32;
    for (i, &d) in dots.iter().enumerate() {
        let out = c19(d as f32, c, rho);
        let e = out - targets[i];
        l += sw[i] * e * e;
    }
    l
}

// ══════════════════════════════════════════════════════
// finetune_c_rho — same step logic as c19_grower, but returns ALL seeds
// ══════════════════════════════════════════════════════
fn finetune_c_rho_log_all(
    dots: &[i32], targets: &[f32], sw: &[f32],
    n_seeds: usize, n_steps: usize, search_seed: u64,
) -> Vec<(f32, f32, f32)> {
    let dots_s: &[i32] = dots;
    let targets_s: &[f32] = targets;
    let sw_s: &[f32] = sw;

    (0..n_seeds).into_par_iter().map(|seed_i| {
        let mut rng = Rng::new(search_seed ^ ((seed_i as u64).wrapping_mul(2654435761).wrapping_add(1)));
        let mut c = rng.range(0.3, 3.0);
        let mut rho = rng.range(0.0, 8.0);

        let mut lr = 0.05f32;
        let mut best_c = c;
        let mut best_rho = rho;
        let mut best_loss = c19_weighted_mse(dots_s, targets_s, sw_s, c, rho);
        let mut stale = 0usize;
        let patience = 30usize;

        for _step in 0..n_steps {
            if stale >= patience { break; }
            let eps = 1e-3f32;
            let l_cp = c19_weighted_mse(dots_s, targets_s, sw_s, c + eps, rho);
            let l_cm = c19_weighted_mse(dots_s, targets_s, sw_s, (c - eps).max(0.1), rho);
            let l_rp = c19_weighted_mse(dots_s, targets_s, sw_s, c, rho + eps);
            let l_rm = c19_weighted_mse(dots_s, targets_s, sw_s, c, (rho - eps).max(0.0));
            let gc = (l_cp - l_cm) / (2.0 * eps);
            let gr = (l_rp - l_rm) / (2.0 * eps);
            let gn = (gc * gc + gr * gr).sqrt();
            if gn < 1e-8 { break; }

            let ol = c19_weighted_mse(dots_s, targets_s, sw_s, c, rho);
            let old_c = c;
            let old_rho = rho;
            let mut improved = false;
            for att in 0..5 {
                let nc = (old_c - lr * gc / gn).max(0.1);
                let nr = (old_rho - lr * gr / gn).max(0.0);
                let nl = c19_weighted_mse(dots_s, targets_s, sw_s, nc, nr);
                if nl < ol {
                    c = nc;
                    rho = nr;
                    lr *= 1.1;
                    if nl < best_loss - 1e-6 {
                        best_loss = nl;
                        best_c = nc;
                        best_rho = nr;
                        stale = 0;
                        improved = true;
                    }
                    break;
                } else {
                    lr *= 0.5;
                    if att == 4 { c = old_c; rho = old_rho; }
                }
            }
            if !improved { stale += 1; }
        }
        (best_c, best_rho, best_loss)
    }).collect()
}

// ══════════════════════════════════════════════════════
// Dense local grid — evaluate every point in ±radius box around center
// ══════════════════════════════════════════════════════
fn local_grid_eval(
    center_c: f32, center_rho: f32,
    c_radius: f32, rho_radius: f32,
    c_step: f32, rho_step: f32,
    dots: &[i32], targets: &[f32], sw: &[f32],
) -> Vec<(f32, f32, f32)> {
    let c_lo = (center_c - c_radius).max(0.1);
    let c_hi = center_c + c_radius;
    let rho_lo = (center_rho - rho_radius).max(0.0);
    let rho_hi = center_rho + rho_radius;

    let n_c = ((c_hi - c_lo) / c_step).floor() as usize + 1;
    let n_rho = ((rho_hi - rho_lo) / rho_step).floor() as usize + 1;

    let mut out = Vec::with_capacity(n_c * n_rho);
    for ic in 0..n_c {
        let cc = c_lo + (ic as f32) * c_step;
        for ir in 0..n_rho {
            let rr = rho_lo + (ir as f32) * rho_step;
            let l = c19_weighted_mse(dots, targets, sw, cc, rr);
            out.push((cc, rr, l));
        }
    }
    out
}

// ══════════════════════════════════════════════════════
// CLI
// ══════════════════════════════════════════════════════
struct Config {
    task: String,
    parents: Vec<usize>,
    weights: Vec<i8>,
    threshold: i32,
    data_seed: u64,
    n_per: usize,
    noise: f32,
    search_seed: u64,
    finetune_seeds: usize,
    finetune_steps: usize,
    c_radius: f32,
    rho_radius: f32,
    c_step: f32,
    rho_step: f32,
    out_file: String,
    top_n: usize,
}

fn parse_args() -> Config {
    let mut cfg = Config {
        task: "grid3_center".to_string(),
        parents: vec![1, 4],
        weights: vec![0, 1],
        threshold: 1,
        data_seed: 42,
        n_per: 200,
        noise: 0.1,
        search_seed: 0,
        finetune_seeds: 20,
        finetune_steps: 100,
        c_radius: 0.3,
        rho_radius: 1.0,
        c_step: 0.01,
        rho_step: 0.05,
        out_file: "target/c19_hot_sweep/report.json".to_string(),
        top_n: 20,
    };
    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        let next = || args.get(i + 1).expect("missing value after flag").clone();
        match a.as_str() {
            "--task" => { cfg.task = next(); i += 2; }
            "--parents" => {
                cfg.parents = next().split(',').map(|s| s.trim().parse().unwrap()).collect();
                i += 2;
            }
            "--weights" => {
                cfg.weights = next().split(',').map(|s| s.trim().parse().unwrap()).collect();
                i += 2;
            }
            "--threshold" => { cfg.threshold = next().parse().unwrap(); i += 2; }
            "--data-seed" => { cfg.data_seed = next().parse().unwrap(); i += 2; }
            "--n-per" => { cfg.n_per = next().parse().unwrap(); i += 2; }
            "--noise" => { cfg.noise = next().parse().unwrap(); i += 2; }
            "--search-seed" => { cfg.search_seed = next().parse().unwrap(); i += 2; }
            "--finetune-seeds" => { cfg.finetune_seeds = next().parse().unwrap(); i += 2; }
            "--finetune-steps" => { cfg.finetune_steps = next().parse().unwrap(); i += 2; }
            "--c-radius" => { cfg.c_radius = next().parse().unwrap(); i += 2; }
            "--rho-radius" => { cfg.rho_radius = next().parse().unwrap(); i += 2; }
            "--c-step" => { cfg.c_step = next().parse().unwrap(); i += 2; }
            "--rho-step" => { cfg.rho_step = next().parse().unwrap(); i += 2; }
            "--out-file" => { cfg.out_file = next(); i += 2; }
            "--top-n" => { cfg.top_n = next().parse().unwrap(); i += 2; }
            other => { eprintln!("unknown flag: {}", other); std::process::exit(2); }
        }
    }
    cfg
}

fn label_dispatch(task: &str) -> Box<dyn Fn(usize, &[u8]) -> Option<u8>> {
    match task {
        "grid3_horizontal_line" => Box::new(|_, px| {
            let r0 = px[0] == 1 && px[1] == 1 && px[2] == 1;
            let r1 = px[3] == 1 && px[4] == 1 && px[5] == 1;
            let r2 = px[6] == 1 && px[7] == 1 && px[8] == 1;
            Some(if r0 || r1 || r2 { 1 } else { 0 })
        }),
        "grid3_vertical_line" => Box::new(|_, px| {
            let c0 = px[0] == 1 && px[3] == 1 && px[6] == 1;
            let c1 = px[1] == 1 && px[4] == 1 && px[7] == 1;
            let c2 = px[2] == 1 && px[5] == 1 && px[8] == 1;
            Some(if c0 || c1 || c2 { 1 } else { 0 })
        }),
        "grid3_diagonal" => Box::new(|_, px| {
            Some(if px[0] == 1 && px[4] == 1 && px[8] == 1 { 1 } else { 0 })
        }),
        "grid3_center" => Box::new(|_, px| Some(px[4])),
        "grid3_corner" => Box::new(|_, px| {
            let c = px[0] == 1 || px[2] == 1 || px[6] == 1 || px[8] == 1;
            Some(if c { 1 } else { 0 })
        }),
        "grid3_diag_xor" => Box::new(|_, px| Some((px[0] ^ px[4] ^ px[8]) & 1)),
        "grid3_full_parity" => Box::new(|_, px| {
            Some((px[0]^px[1]^px[2]^px[3]^px[4]^px[5]^px[6]^px[7]^px[8]) & 1)
        }),
        "grid3_majority" => Box::new(|_, px| {
            let s: usize = px.iter().map(|&v| v as usize).sum();
            Some(if s >= 5 { 1 } else { 0 })
        }),
        "grid3_symmetry_h" => Box::new(|_, px| {
            let sym = px[0]==px[2] && px[3]==px[5] && px[6]==px[8];
            Some(if sym { 1 } else { 0 })
        }),
        "grid3_top_heavy" => Box::new(|_, px| {
            let top: usize = (px[0] as usize)+(px[1] as usize)+(px[2] as usize);
            let bot: usize = (px[6] as usize)+(px[7] as usize)+(px[8] as usize);
            Some(if top > bot { 1 } else { 0 })
        }),
        other => { eprintln!("unknown task: {}", other); std::process::exit(2); }
    }
}

// ══════════════════════════════════════════════════════
// main
// ══════════════════════════════════════════════════════
fn main() {
    let cfg = parse_args();
    let t0 = Instant::now();

    // Phase 0 — reproduce data + dots + targets + uniform sw
    let label_fn = label_dispatch(&cfg.task);
    let data = gen_data(label_fn.as_ref(), cfg.noise, cfg.n_per, cfg.data_seed);

    println!("===========================================================");
    println!("  c19_hot_sweep — {} (data_seed={} search_seed={})", cfg.task, cfg.data_seed, cfg.search_seed);
    println!("  N0 setup: parents={:?} weights={:?} threshold={}", cfg.parents, cfg.weights, cfg.threshold);
    println!("  Data: {} train / {} val / {} test", data.train.len(), data.val.len(), data.test.len());
    println!("  Finetune: {} seeds × {} steps (finite-diff gradient descent)", cfg.finetune_seeds, cfg.finetune_steps);
    println!("  Local grid: ±{:.2}c × ±{:.2}rho step {:.3}×{:.3}", cfg.c_radius, cfg.rho_radius, cfg.c_step, cfg.rho_step);
    println!("===========================================================");

    let np = data.train.len();
    let dots: Vec<i32> = (0..np).map(|pi| {
        let mut d = 0i32;
        for (&w, &p) in cfg.weights.iter().zip(&cfg.parents) {
            d += (w as i32) * (data.train[pi].0[p] as i32);
        }
        d
    }).collect();
    let targets: Vec<f32> = data.train.iter()
        .map(|(_, y)| if *y == 1 { 1.0f32 } else { -1.0 }).collect();
    let sw = vec![1.0f32 / np as f32; np];

    // Baseline loss at (c=1.0, rho=0.0) — reference point
    let baseline_loss = c19_weighted_mse(&dots, &targets, &sw, 1.0, 0.0);
    println!("\n[baseline] c=1.00 rho=0.00 loss={:.6}", baseline_loss);

    // Phase 1 — 20 seed finetune, log ALL endpoints
    println!("\n=== PHASE 1: finetune {} seeds ===", cfg.finetune_seeds);
    let seed_points = finetune_c_rho_log_all(
        &dots, &targets, &sw,
        cfg.finetune_seeds, cfg.finetune_steps, cfg.search_seed,
    );

    let mut sorted_seeds = seed_points.clone();
    sorted_seeds.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    println!("  seed endpoints (sorted by loss):");
    for (rank, (c, rho, l)) in sorted_seeds.iter().enumerate() {
        println!("    rank {:>2}: c={:>8.4}  rho={:>8.4}  loss={:.6}", rank, c, rho, l);
    }

    // Quick bucket analysis — how many unique regions did 20 seeds find?
    let mut buckets: Vec<((i32, i32), usize)> = Vec::new();
    for &(c, rho, _) in &sorted_seeds {
        let key = ((c * 2.0).round() as i32, (rho).round() as i32);
        if let Some(b) = buckets.iter_mut().find(|b| b.0 == key) {
            b.1 += 1;
        } else {
            buckets.push((key, 1));
        }
    }
    println!("  bucket count (round(c*2), round(rho)): {} distinct", buckets.len());
    for (key, count) in &buckets {
        println!("    bucket ({:.1}, {}): {} seed(s)", key.0 as f32 / 2.0, key.1, count);
    }

    // Phase 2 — dense local grid around each seed
    println!("\n=== PHASE 2: dense local grid around each seed ===");
    let sample_points = local_grid_eval(
        sorted_seeds[0].0, sorted_seeds[0].1,
        cfg.c_radius, cfg.rho_radius, cfg.c_step, cfg.rho_step,
        &dots, &targets, &sw,
    );
    println!("  grid size per seed: {} points", sample_points.len());

    // Dedupe via quantized (c, rho) key → retain only the lowest loss per bucket
    let mut dense_map: HashMap<(i32, i32), (f32, f32, f32)> = HashMap::new();
    for (seed_idx, &(sc, sr, _)) in sorted_seeds.iter().enumerate() {
        let points = local_grid_eval(
            sc, sr, cfg.c_radius, cfg.rho_radius, cfg.c_step, cfg.rho_step,
            &dots, &targets, &sw,
        );
        for (c, rho, l) in points {
            let key = ((c * 1e5).round() as i32, (rho * 1e5).round() as i32);
            let entry = dense_map.entry(key).or_insert((c, rho, f32::MAX));
            if l < entry.2 { *entry = (c, rho, l); }
        }
        let _ = seed_idx;
    }

    let mut dense_sorted: Vec<(f32, f32, f32)> = dense_map.values().cloned().collect();
    dense_sorted.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    println!("  total unique dense points evaluated: {}", dense_sorted.len());
    println!("\n  dense top-{}:", cfg.top_n);
    for (rank, (c, rho, l)) in dense_sorted.iter().take(cfg.top_n).enumerate() {
        println!("    rank {:>2}: c={:>8.4}  rho={:>8.4}  loss={:.6}", rank, c, rho, l);
    }

    // Phase 3 — sparse 10×6=60 grid comparison
    println!("\n=== PHASE 3: sparse grid baseline (c19_grower quant grid) ===");
    let c_sparse: &[f32] = &[0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 2.5, 3.0];
    let rho_sparse: &[f32] = &[0.0, 1.0, 2.0, 4.0, 6.0, 8.0];
    let mut sparse: Vec<(f32, f32, f32)> = Vec::new();
    for &c in c_sparse {
        for &r in rho_sparse {
            let l = c19_weighted_mse(&dots, &targets, &sw, c, r);
            sparse.push((c, r, l));
        }
    }
    sparse.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    println!("  sparse top-5:");
    for (rank, (c, rho, l)) in sparse.iter().take(5).enumerate() {
        println!("    rank {:>2}: c={:>8.4}  rho={:>8.4}  loss={:.6}", rank, c, rho, l);
    }

    // Deltas
    let sparse_best = sparse[0].2;
    let dense_best = dense_sorted[0].2;
    let finetune_best = sorted_seeds[0].2;
    println!("\n=== SUMMARY ===");
    println!("  baseline   (c=1, rho=0):     loss={:.6}", baseline_loss);
    println!("  sparse  10×6 = 60 points:    loss={:.6}  (best: c={:.4} rho={:.4})", sparse_best, sparse[0].0, sparse[0].1);
    println!("  finetune 20 seed best:       loss={:.6}  (best: c={:.4} rho={:.4})", finetune_best, sorted_seeds[0].0, sorted_seeds[0].1);
    println!("  dense hot-bucket best:       loss={:.6}  (best: c={:.4} rho={:.4})", dense_best, dense_sorted[0].0, dense_sorted[0].1);
    let rel_vs_sparse = if sparse_best > 0.0 { 100.0 * (sparse_best - dense_best) / sparse_best } else { 0.0 };
    let rel_vs_finetune = if finetune_best > 0.0 { 100.0 * (finetune_best - dense_best) / finetune_best } else { 0.0 };
    println!("  dense improvement vs sparse:   {:+.4} ({:+.2}%)", sparse_best - dense_best, rel_vs_sparse);
    println!("  dense improvement vs finetune: {:+.4} ({:+.2}%)", finetune_best - dense_best, rel_vs_finetune);

    // JSON dump
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!("  \"task\": \"{}\",\n", cfg.task));
    json.push_str(&format!("  \"parents\": {:?},\n", cfg.parents));
    json.push_str(&format!("  \"weights\": {:?},\n", cfg.weights));
    json.push_str(&format!("  \"threshold\": {},\n", cfg.threshold));
    json.push_str(&format!("  \"data_seed\": {},\n", cfg.data_seed));
    json.push_str(&format!("  \"n_per\": {},\n", cfg.n_per));
    json.push_str(&format!("  \"noise\": {:.4},\n", cfg.noise));
    json.push_str(&format!("  \"search_seed\": {},\n", cfg.search_seed));
    json.push_str(&format!("  \"finetune_seeds\": {},\n", cfg.finetune_seeds));
    json.push_str(&format!("  \"finetune_steps\": {},\n", cfg.finetune_steps));
    json.push_str(&format!("  \"c_radius\": {:.4},\n", cfg.c_radius));
    json.push_str(&format!("  \"rho_radius\": {:.4},\n", cfg.rho_radius));
    json.push_str(&format!("  \"c_step\": {:.4},\n", cfg.c_step));
    json.push_str(&format!("  \"rho_step\": {:.4},\n", cfg.rho_step));
    json.push_str(&format!("  \"n_train\": {},\n", np));
    json.push_str(&format!("  \"baseline_loss\": {:.6},\n", baseline_loss));
    json.push_str("  \"seed_endpoints\": [\n");
    for (rank, (c, r, l)) in sorted_seeds.iter().enumerate() {
        json.push_str(&format!("    {{\"rank\": {}, \"c\": {:.6}, \"rho\": {:.6}, \"loss\": {:.6}}}", rank, c, r, l));
        if rank + 1 < sorted_seeds.len() { json.push(','); }
        json.push('\n');
    }
    json.push_str("  ],\n");
    json.push_str("  \"buckets\": [\n");
    for (bi, (key, count)) in buckets.iter().enumerate() {
        json.push_str(&format!("    {{\"c_bucket\": {:.1}, \"rho_bucket\": {}, \"count\": {}}}",
            key.0 as f32 / 2.0, key.1, count));
        if bi + 1 < buckets.len() { json.push(','); }
        json.push('\n');
    }
    json.push_str("  ],\n");
    json.push_str("  \"dense_top\": [\n");
    let top_n = cfg.top_n.min(dense_sorted.len());
    for rank in 0..top_n {
        let (c, rho, l) = dense_sorted[rank];
        json.push_str(&format!("    {{\"rank\": {}, \"c\": {:.6}, \"rho\": {:.6}, \"loss\": {:.6}}}", rank, c, rho, l));
        if rank + 1 < top_n { json.push(','); }
        json.push('\n');
    }
    json.push_str("  ],\n");
    json.push_str(&format!("  \"sparse_best\": {{\"c\": {:.4}, \"rho\": {:.4}, \"loss\": {:.6}}},\n",
        sparse[0].0, sparse[0].1, sparse[0].2));
    json.push_str(&format!("  \"finetune_best\": {{\"c\": {:.4}, \"rho\": {:.4}, \"loss\": {:.6}}},\n",
        sorted_seeds[0].0, sorted_seeds[0].1, sorted_seeds[0].2));
    json.push_str(&format!("  \"dense_best\": {{\"c\": {:.4}, \"rho\": {:.4}, \"loss\": {:.6}}},\n",
        dense_sorted[0].0, dense_sorted[0].1, dense_sorted[0].2));
    json.push_str(&format!("  \"dense_points_evaluated\": {},\n", dense_sorted.len()));
    json.push_str(&format!("  \"elapsed_s\": {:.3}\n", t0.elapsed().as_secs_f32()));
    json.push_str("}\n");

    if let Some(parent) = Path::new(&cfg.out_file).parent() {
        fs::create_dir_all(parent).ok();
    }
    fs::write(&cfg.out_file, &json).expect("write report");
    println!("\nReport: {}", cfg.out_file);
    println!("Elapsed: {:.2}s", t0.elapsed().as_secs_f32());
}
