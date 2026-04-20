//! c19_grower — grid3 grower with C19 learnable activation + QAT → LUT bake
//!
//! Duplicates the `grid3_curriculum.rs` pipeline, then extends every accepted
//! neuron with a 4-phase C19 postprocess:
//!   Phase A — structure search (unchanged: scout + ternary + threshold)
//!   Phase B — finite-diff finetune of (c, rho) over weighted MSE
//!   Phase C — quantization grid search across 60 integer-friendly candidates
//!   Phase D — LUT bake: precomputed Vec<f32> of length (2*abs_sum + 1)
//!
//! State file schema is bumped to VER=2 (15 columns per N row); the loader
//! explicitly refuses VER=1 (old neuron_grower) files with a clear error.
//!
//! Runtime inference uses the real-valued LUT outputs for the ensemble sum,
//! while hidden parents still feed downstream dot products with thresholded
//! 0/1 values to preserve the AdaBoost accept-gate semantics.
//!
//! Run: cargo run --release --example c19_grower -p instnct-core -- \
//!        --task grid3_center --search-seed 1 --out-dir target/c19_grower/center_1
//!
//! This example does NOT modify `neuron_grower.rs` or `grid3_curriculum.rs`.

use std::collections::HashSet;
use std::io::Write as IoWrite;
use std::time::Instant;

use rayon::prelude::*;

// ══════════════════════════════════════════════════════
// CLI
// ══════════════════════════════════════════════════════
struct Config {
    task: String,
    search_seed: u64,
    out_dir: String,
    data_seed: u64,
    max_neurons: usize,
    max_fan: usize,
    n_proposals: usize,
    stall_limit: usize,
    noise: f32,
    n_per: usize,
    scout_top: usize,
    pair_top: usize,
    probe_epochs: usize,
    // C19 phase knobs
    finetune_seeds: usize,
    finetune_steps: usize,
    quant_tolerance: f32,
    skip_finetune: bool,
    skip_quant: bool,
    // Integer inference knobs (int8 LUT + int16 alpha)
    use_i8: bool,
    i8_assert_tol_pp: f32,
    // Forever-network knobs (opt-in; default = current behavior)
    task_list: Vec<String>,
    interactive: bool,
    exhaustive: bool,
    verbose_search: bool,
    allow_task_switch: bool,
    // Manual distribution-picker knobs:
    //   --force-pick N   → skip interactive, bake trained[N] instead of top
    //   --preview-only   → run STEP A-C, print the trained table, exit before
    //                      baking (useful for a two-step manual workflow:
    //                      preview to see the distribution, then re-run with
    //                      --force-pick to commit a specific candidate).
    //   --bake-best      → bake every trained candidate tentatively, measure
    //                      post-bake ensemble val, commit the candidate with
    //                      the highest score (ties broken by smaller ni).
    force_pick: Option<usize>,
    preview_only: bool,
    bake_best: bool,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut task: Option<String> = None;
    let mut search_seed: Option<u64> = None;
    let mut out_dir: Option<String> = None;
    let mut data_seed: u64 = 42;
    let mut max_neurons: usize = 32;
    let mut max_fan: usize = 6;
    let mut n_proposals: usize = 20;
    let mut stall_limit: usize = 16;
    let mut noise: f32 = 0.10;
    let mut n_per: usize = 200;
    let mut scout_top: usize = 12;
    let mut pair_top: usize = 8;
    let mut probe_epochs: usize = 400;
    let mut finetune_seeds: usize = 20;
    let mut finetune_steps: usize = 100;
    let mut quant_tolerance: f32 = 0.005;
    let mut skip_finetune: bool = false;
    let mut skip_quant: bool = false;
    // int8 LUT / int16 alpha quant defaults: on by default (fully integer path).
    let mut use_i8: bool = true;
    let mut i8_assert_tol_pp: f32 = 0.25;
    // Forever-network knobs (all opt-in; empty task_list + false flags =
    // bit-exact current behavior).
    let mut task_list: Vec<String> = Vec::new();
    let mut interactive: bool = false;
    let mut exhaustive: bool = false;
    let mut verbose_search: bool = false;
    let mut allow_task_switch: bool = false;
    let mut force_pick: Option<usize> = None;
    let mut preview_only: bool = false;
    let mut bake_best: bool = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--task" => { i += 1; task = Some(args[i].clone()); }
            "--search-seed" => {
                i += 1;
                search_seed = Some(args[i].parse().unwrap_or_else(|_| {
                    eprintln!("ERROR: --search-seed must be a non-negative integer");
                    std::process::exit(1);
                }));
            }
            "--out-dir" => { i += 1; out_dir = Some(args[i].clone()); }
            "--data-seed" => { i += 1; data_seed = args[i].parse().unwrap_or(42); }
            "--max-neurons" => { i += 1; max_neurons = args[i].parse().unwrap_or(32); }
            "--max-fan" => { i += 1; max_fan = args[i].parse().unwrap_or(6); }
            "--proposals" => { i += 1; n_proposals = args[i].parse().unwrap_or(20); }
            "--stall" => { i += 1; stall_limit = args[i].parse().unwrap_or(16); }
            "--noise" => { i += 1; noise = args[i].parse().unwrap_or(0.10); }
            "--n-per" => { i += 1; n_per = args[i].parse().unwrap_or(200); }
            "--scout-top" => { i += 1; scout_top = args[i].parse().unwrap_or(12); }
            "--pair-top" => { i += 1; pair_top = args[i].parse().unwrap_or(8); }
            "--probe-epochs" => { i += 1; probe_epochs = args[i].parse().unwrap_or(400); }
            "--finetune-seeds" => { i += 1; finetune_seeds = args[i].parse().unwrap_or(20); }
            "--finetune-steps" => { i += 1; finetune_steps = args[i].parse().unwrap_or(100); }
            "--quant-tolerance" => { i += 1; quant_tolerance = args[i].parse().unwrap_or(0.005); }
            "--skip-finetune" => { skip_finetune = true; }
            "--skip-quant" => { skip_quant = true; }
            // int8 LUT quant toggles
            "--i8" => { use_i8 = true; }
            "--no-i8" => { use_i8 = false; }
            "--i8-assert-tol" => {
                i += 1;
                i8_assert_tol_pp = args[i].parse().unwrap_or(0.25);
            }
            // Forever-network flags
            "--task-list" => {
                i += 1;
                task_list = args[i]
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "--interactive" => { interactive = true; }
            "--exhaustive" => { exhaustive = true; }
            "--verbose-search" => { verbose_search = true; }
            "--allow-task-switch" => { allow_task_switch = true; }
            "--force-pick" => {
                i += 1;
                force_pick = Some(args[i].parse().unwrap_or_else(|_| {
                    eprintln!("ERROR: --force-pick expects a non-negative integer (0-based trained candidate idx)");
                    std::process::exit(1);
                }));
            }
            "--preview-only" => { preview_only = true; }
            "--bake-best" => { bake_best = true; }
            _ => {}
        }
        i += 1;
    }

    let task = match task {
        Some(t) => t,
        None => {
            // In task-list mode, --task is optional — we'll use task_list[0] as
            // the initial task and bootstrap from there.
            if !task_list.is_empty() {
                task_list[0].clone()
            } else {
                eprintln!("ERROR: --task is required (or use --task-list t1,t2,...)");
                std::process::exit(2);
            }
        }
    };
    let search_seed = match search_seed {
        Some(s) => s,
        None => { eprintln!("ERROR: --search-seed is required"); std::process::exit(2); }
    };
    let out_dir = match out_dir {
        Some(o) => o,
        None => { eprintln!("ERROR: --out-dir is required"); std::process::exit(2); }
    };

    Config {
        task,
        search_seed,
        out_dir,
        data_seed,
        max_neurons,
        max_fan,
        n_proposals,
        stall_limit,
        noise,
        n_per,
        scout_top,
        pair_top,
        probe_epochs,
        finetune_seeds,
        finetune_steps,
        quant_tolerance,
        skip_finetune,
        skip_quant,
        use_i8,
        i8_assert_tol_pp,
        task_list,
        interactive,
        exhaustive,
        verbose_search,
        allow_task_switch,
        force_pick,
        preview_only,
        bake_best,
    }
}

// ══════════════════════════════════════════════════════
// PRNG + SIGMOID (duplicated verbatim from neuron_grower.rs)
// ══════════════════════════════════════════════════════
struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 { self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.s }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn range(&mut self, lo: f32, hi: f32) -> f32 { lo + self.f32() * (hi - lo) }
    fn bool_p(&mut self, p: f32) -> bool { self.f32() < p }
    fn pick(&mut self, n: usize) -> usize { self.next() as usize % n }
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

// ══════════════════════════════════════════════════════
// C19 ACTIVATION (verbatim from c19_rho_learnable.rs lines 19-31)
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
// FONT + DATA (Font9 for grid3_*)
// ══════════════════════════════════════════════════════
const FONT: [[u8; 9]; 10] = [
    [1,1,1, 1,0,1, 1,1,1], [0,1,0, 0,1,0, 0,1,0],
    [1,1,0, 0,1,0, 0,1,1], [1,1,0, 0,1,0, 1,1,0],
    [1,0,1, 1,1,1, 0,0,1], [0,1,1, 0,1,0, 1,1,0],
    [1,0,0, 1,1,0, 1,1,0], [1,1,1, 0,0,1, 0,0,1],
    [1,1,1, 1,1,1, 1,1,1], [1,1,1, 1,1,1, 0,1,1],
];

fn task_n_in(task: &str) -> usize {
    if task.starts_with("grid3_") {
        9
    } else {
        eprintln!("ERROR: unknown task '{}' (expected grid3_*)", task);
        std::process::exit(2);
    }
}

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
// NEURON + NET (extended with c19 fields + LUT)
// ══════════════════════════════════════════════════════
#[derive(Clone)]
struct Neuron {
    id: usize,
    parents: Vec<usize>,
    tick: u32,
    weights: Vec<i8>,          // ternary -1/0/+1
    threshold: i32,            // ternary threshold (dot >= threshold)
    alpha: f32,                // Freund-Schapire weight
    train_acc: f32,
    val_acc: f32,

    // C19 phase outputs
    c_float: f32,              // learned c pre-quant
    rho_float: f32,            // learned rho pre-quant
    c_quant: f32,              // quantized c used for LUT
    rho_quant: f32,            // quantized rho used for LUT
    lut_min_dot: i32,          // = -abs_sum
    lut: Vec<f32>,              // baked LUT, len = 2*abs_sum + 1

    // ── int8 LUT quantization (derived from `lut`, not persisted) ─────
    //
    // Per-neuron symmetric absmax quantization: `lut_scale = max(|lut|)/127`,
    // `lut_i8[i] = clip(round(lut[i]/lut_scale), -127, 127)`. Dequantized
    // lookup is `lut_i8[i] as f32 * lut_scale`. Lossless on all 48 neurons
    // tested in the quant analysis (see .claude/research/c19_i8_quant_analysis.md).
    // Degenerate all-zero LUT uses `lut_scale = 1.0, lut_i8 = [0; N]`.
    lut_i8: Vec<i8>,
    lut_scale: f32,
    // Per-network-shared int16 alpha quantization — see `quantize_alphas_i16`.
    // `alpha_scale` is the same value for every neuron in the same `Net`, stored
    // per-neuron for convenience so `eval_lut_i8` can dispatch without a Net ref.
    alpha_i16: i16,
    alpha_scale: f32,
}

impl Neuron {
    /// Integer dot product over ternary weights and {0,1} parent signals.
    fn dot(&self, sigs: &[u8]) -> i32 {
        let mut d = 0i32;
        for (&w, &p) in self.weights.iter().zip(&self.parents) {
            d += (w as i32) * (sigs[p] as i32);
        }
        d
    }

    /// Real-valued LUT eval — looks up the baked c19 output for the current dot.
    fn eval_lut(&self, sigs: &[u8]) -> f32 {
        let d = self.dot(sigs);
        let idx = d - self.lut_min_dot;
        if idx < 0 { return 0.0; }
        let u = idx as usize;
        if u >= self.lut.len() { return 0.0; }
        self.lut[u]
    }

    /// int8 LUT eval — identical shape to `eval_lut` but loads from the
    /// int8 table and multiplies by the per-neuron scale on the fly. This is
    /// the hot path when `--i8` is on.
    ///
    /// Semantics and out-of-range fallback must match `eval_lut` byte-for-byte
    /// at the sign-of-output level, because the hidden signal chain still uses
    /// `>= 0.0` on this return value (same as the float path).
    fn eval_lut_i8(&self, sigs: &[u8]) -> f32 {
        let d = self.dot(sigs);
        let idx = d - self.lut_min_dot;
        if idx < 0 { return 0.0; }
        let u = idx as usize;
        if u >= self.lut_i8.len() { return 0.0; }
        (self.lut_i8[u] as f32) * self.lut_scale
    }

    /// Dequantized alpha. Callers that want the pure float path should just
    /// read `self.alpha` directly.
    fn alpha_i16_f32(&self) -> f32 {
        (self.alpha_i16 as f32) * self.alpha_scale
    }

    /// Thresholded LUT prediction (0/1) — used by the AdaBoost accept gate
    /// and by the hidden parent signal chain (to keep dot products integer).
    ///
    /// Threshold semantics (`>= 0.0`) match `Net::predict`. This function is
    /// always driven by the float LUT, not the i8 LUT: the hidden signal
    /// chain decision only needs the sign, and the float LUT is guaranteed
    /// to have `lut[zero_idx] == 0.0` by `c19(0, c, rho) == 0`. The per-neuron
    /// absmax quant preserves this identity exactly, so the i8 and f32 paths
    /// agree on the sign bit at every idx (see quant analysis); picking f32
    /// here is a defense-in-depth choice for the accept gate.
    fn eval_lut_pred(&self, sigs: &[u8]) -> u8 {
        if self.eval_lut(sigs) >= 0.0 { 1 } else { 0 }
    }
}

struct Net {
    neurons: Vec<Neuron>,
    n_in: usize,
    sig_ticks: Vec<u32>,
    /// Runtime dispatch flag: when true, `predict` uses `eval_lut_i8` and
    /// `alpha_i16_f32()`; when false, the pre-patch pure-float path.
    use_i8: bool,
}
impl Net {
    fn new(n: usize) -> Self {
        Net { neurons: Vec::new(), n_in: n, sig_ticks: vec![0; n], use_i8: true }
    }
    fn n_sig(&self) -> usize { self.n_in + self.neurons.len() }

    /// Byte-only eval — hidden parents see thresholded 0/1 so downstream dot
    /// products stay integer. This is the contract used by the parent-pick
    /// phase and by the AdaBoost accept gate.
    fn eval_all(&self, inp: &[u8]) -> Vec<u8> {
        let mut s: Vec<u8> = inp.to_vec();
        for n in &self.neurons {
            s.push(n.eval_lut_pred(&s));
        }
        s
    }

    /// Ensemble predict using real-valued LUT outputs at the final sum.
    /// Intermediate hidden signals remain thresholded (consistent with
    /// `eval_all` so the per-neuron dot computation is still integer).
    ///
    /// Dispatch: when `self.use_i8`, loads the i8 LUT and dequantized alpha
    /// at each step; otherwise uses the original float path. The semantics
    /// (soft voting with `score += alpha * lut_out`, hidden-signal threshold
    /// at `>= 0.0`) are identical across both branches.
    fn predict(&self, inp: &[u8]) -> u8 {
        if self.neurons.is_empty() { return 0; }
        let mut sigs: Vec<u8> = inp.to_vec();
        let mut score = 0.0f32;
        if self.use_i8 {
            for n in &self.neurons {
                let lut_out = n.eval_lut_i8(&sigs);
                score += n.alpha_i16_f32() * lut_out;
                sigs.push(if lut_out >= 0.0 { 1 } else { 0 });
            }
        } else {
            for n in &self.neurons {
                let lut_out = n.eval_lut(&sigs);
                score += n.alpha * lut_out;
                sigs.push(if lut_out >= 0.0 { 1 } else { 0 });
            }
        }
        if score >= 0.0 { 1 } else { 0 }
    }

    /// Accuracy using the current `use_i8` setting.
    fn accuracy(&self, data: &[(Vec<u8>, u8)]) -> f32 {
        if self.neurons.is_empty() { return 50.0; }
        data.iter().filter(|(x, y)| self.predict(x) == *y).count() as f32 / data.len() as f32 * 100.0
    }

    /// Accuracy forced to the float path regardless of `self.use_i8`. Used by
    /// the post-bake QAT verification assertion to compare float vs i8.
    ///
    /// Inlines the float inference path directly instead of cloning `self`
    /// with `use_i8=false` — cloning a Net is expensive (it copies every LUT),
    /// and we want this check to be cheap so the post-bake assertion fires on
    /// every accepted neuron without slowing growth noticeably.
    fn accuracy_f32(&self, data: &[(Vec<u8>, u8)]) -> f32 {
        if self.neurons.is_empty() { return 50.0; }
        let mut correct = 0usize;
        for (inp, y) in data {
            let mut sigs: Vec<u8> = inp.clone();
            let mut score = 0.0f32;
            for n in &self.neurons {
                let lut_out = n.eval_lut(&sigs);
                score += n.alpha * lut_out;
                sigs.push(if lut_out >= 0.0 { 1 } else { 0 });
            }
            let pred: u8 = if score >= 0.0 { 1 } else { 0 };
            if pred == *y { correct += 1; }
        }
        correct as f32 / data.len() as f32 * 100.0
    }

    /// Accuracy forced to the i8 path regardless of `self.use_i8`. Mirror of
    /// `accuracy_f32` for the QAT verification.
    fn accuracy_i8(&self, data: &[(Vec<u8>, u8)]) -> f32 {
        if self.neurons.is_empty() { return 50.0; }
        let mut correct = 0usize;
        for (inp, y) in data {
            let mut sigs: Vec<u8> = inp.clone();
            let mut score = 0.0f32;
            for n in &self.neurons {
                let lut_out = n.eval_lut_i8(&sigs);
                score += n.alpha_i16_f32() * lut_out;
                sigs.push(if lut_out >= 0.0 { 1 } else { 0 });
            }
            let pred: u8 = if score >= 0.0 { 1 } else { 0 };
            if pred == *y { correct += 1; }
        }
        correct as f32 / data.len() as f32 * 100.0
    }

    fn add(&mut self, n: Neuron) {
        self.sig_ticks.push(n.tick);
        self.neurons.push(n);
    }
}

fn output_match_rate(a: &[u8], all_sigs: &[Vec<u8>], sig: usize) -> f32 {
    a.iter().enumerate().filter(|(i, v)| all_sigs[*i][sig] == **v).count() as f32 / a.len() as f32
}

#[derive(Clone)]
struct SignalScout {
    idx: usize,
    single_score: f32,
    single_sign: i8,
    probe_w: f32,
    rank_sum: usize,
}

#[derive(Clone)]
struct PairScout {
    a: usize,
    b: usize,
    score: f32,
    gain: f32,
}

fn sig_name(idx: usize, n_in: usize) -> String {
    if idx < n_in { format!("x{}", idx) } else { format!("N{}", idx - n_in) }
}

fn weighted_score(outputs: &[u8], labels: &[(Vec<u8>, u8)], sw: &[f32]) -> f32 {
    outputs.iter().zip(labels).zip(sw)
        .map(|((&pred, (_, y)), &wt)| if pred == *y { wt } else { 0.0 })
        .sum()
}

fn best_single_signal_scores(all_sigs: &[Vec<u8>], data: &Data, sw: &[f32], n_sig: usize) -> Vec<SignalScout> {
    let mut out = Vec::with_capacity(n_sig);
    for sig in 0..n_sig {
        let pos: f32 = data.train.iter().enumerate().zip(sw)
            .map(|((pi, (_, y)), &wt)| if all_sigs[pi][sig] == *y { wt } else { 0.0 })
            .sum();
        let neg: f32 = data.train.iter().enumerate().zip(sw)
            .map(|((pi, (_, y)), &wt)| if (1 - all_sigs[pi][sig]) == *y { wt } else { 0.0 })
            .sum();
        let (single_score, single_sign) = if pos >= neg { (pos, 1) } else { (neg, -1) };
        out.push(SignalScout { idx: sig, single_score, single_sign, probe_w: 0.0, rank_sum: 0 });
    }
    out
}

fn backprop_probe_all(all_sigs: &[Vec<u8>], data: &Data, sw: &[f32], n_sig: usize, epochs: usize) -> (Vec<f32>, f32) {
    let mut w = vec![0.0f32; n_sig];
    let mut b = 0.0f32;
    if n_sig == 0 { return (w, b); }
    for _ in 0..epochs {
        for (pi, (_, y)) in data.train.iter().enumerate() {
            let z = b + (0..n_sig).map(|i| w[i] * all_sigs[pi][i] as f32).sum::<f32>();
            let a = sigmoid(z);
            let g = (a - *y as f32) * sw[pi] * data.train.len() as f32;
            for i in 0..n_sig { w[i] -= 0.15 * g * all_sigs[pi][i] as f32; }
            b -= 0.15 * g;
        }
    }
    (w, b)
}

fn merge_signal_ranks(mut scouts: Vec<SignalScout>, probe_w: &[f32]) -> Vec<SignalScout> {
    let mut by_single: Vec<usize> = (0..scouts.len()).collect();
    by_single.sort_by(|&a, &b| scouts[b].single_score.partial_cmp(&scouts[a].single_score).unwrap());
    let mut single_rank = vec![0usize; scouts.len()];
    for (rank, idx) in by_single.iter().enumerate() { single_rank[*idx] = rank; }

    let mut by_probe: Vec<usize> = (0..probe_w.len()).collect();
    by_probe.sort_by(|&a, &b| probe_w[b].abs().partial_cmp(&probe_w[a].abs()).unwrap());
    let mut probe_rank = vec![0usize; probe_w.len()];
    for (rank, idx) in by_probe.iter().enumerate() { probe_rank[*idx] = rank; }

    for s in &mut scouts {
        s.probe_w = probe_w[s.idx];
        s.rank_sum = single_rank[s.idx] + probe_rank[s.idx];
    }
    scouts.sort_by(|a, b| {
        a.rank_sum.cmp(&b.rank_sum)
            .then_with(|| b.single_score.partial_cmp(&a.single_score).unwrap())
            .then_with(|| b.probe_w.abs().partial_cmp(&a.probe_w.abs()).unwrap())
    });
    scouts
}

fn best_small_ternary_score(parents: &[usize], all_sigs: &[Vec<u8>], data: &Data, sw: &[f32]) -> f32 {
    let ni = parents.len();
    if ni == 0 { return -1.0; }
    let np = data.train.len();
    let total = 3u64.pow(ni as u32);
    let mut best = -1.0f32;
    for combo in 0..total {
        let mut r = combo;
        let mut w = vec![0i8; ni];
        for wi in &mut w { *wi = (r % 3) as i8 - 1; r /= 3; }
        let dots: Vec<i32> = (0..np).map(|pi| {
            let mut d = -1i32;
            for (j, &pidx) in parents.iter().enumerate() { d += (w[j] as i32) * (all_sigs[pi][pidx] as i32); }
            d
        }).collect();
        let mn = dots.iter().copied().min().unwrap_or(0);
        let mx = dots.iter().copied().max().unwrap_or(0);
        for thresh in (mn - 1)..=(mx + 1) {
            let outs: Vec<u8> = dots.iter().map(|&d| if d >= thresh { 1 } else { 0 }).collect();
            let sc = weighted_score(&outs, &data.train, sw);
            if sc > best { best = sc; }
        }
    }
    best
}

fn pair_lifts_from_ranked(ranked: &[SignalScout], all_sigs: &[Vec<u8>], data: &Data, sw: &[f32], pair_top: usize) -> Vec<PairScout> {
    let top_n = ranked.len().min(pair_top.max(2));
    let mut out = Vec::new();
    for i in 0..top_n {
        for j in (i + 1)..top_n {
            let a = ranked[i].idx;
            let b = ranked[j].idx;
            let score = best_small_ternary_score(&[a, b], all_sigs, data, sw);
            let gain = score - ranked[i].single_score.max(ranked[j].single_score);
            out.push(PairScout { a, b, score, gain });
        }
    }
    out.sort_by(|a, b| b.gain.partial_cmp(&a.gain).unwrap().then_with(|| b.score.partial_cmp(&a.score).unwrap()));
    out
}

fn push_candidate(sets: &mut Vec<Vec<usize>>, seen: &mut HashSet<Vec<usize>>, parents: Vec<usize>) {
    if parents.len() < 2 { return; }
    let mut p = parents;
    p.sort_unstable();
    p.dedup();
    if p.len() < 2 { return; }
    if seen.insert(p.clone()) { sets.push(p); }
}

fn build_candidate_sets(
    ranked: &[SignalScout], pairs: &[PairScout], n_sig: usize, n_in: usize, cfg: &Config, step: usize,
) -> Vec<Vec<usize>> {
    let max_fan = cfg.max_fan.min(n_sig).max(2);
    let pool_n = ranked.len().min(cfg.scout_top.max(max_fan));
    let pool: Vec<usize> = ranked.iter().take(pool_n).map(|s| s.idx).collect();
    let hidden: Vec<usize> = ranked.iter().filter(|s| s.idx >= n_in).take(pool_n).map(|s| s.idx).collect();
    let raw: Vec<usize> = ranked.iter().filter(|s| s.idx < n_in).take(pool_n).map(|s| s.idx).collect();

    let mut sets = Vec::new();
    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    for &sz in &[2usize, 3, 4, 6, 8, 10, 12] {
        let sz = sz.min(max_fan).min(pool.len());
        if sz >= 2 { push_candidate(&mut sets, &mut seen, pool.iter().copied().take(sz).collect()); }
    }

    if !hidden.is_empty() {
        let mut mix = Vec::new();
        let h_take = hidden.len().min(max_fan / 2 + 1);
        mix.extend(hidden.iter().copied().take(h_take));
        for &p in &pool {
            if mix.len() >= max_fan { break; }
            if !mix.contains(&p) { mix.push(p); }
        }
        push_candidate(&mut sets, &mut seen, mix);
    }

    if !raw.is_empty() {
        push_candidate(&mut sets, &mut seen, raw.iter().copied().take(raw.len().min(max_fan)).collect());
    }

    for pair in pairs.iter().take(4) {
        let mut seeded = vec![pair.a, pair.b];
        for &p in &pool {
            if seeded.len() >= max_fan.min(6) { break; }
            if !seeded.contains(&p) { seeded.push(p); }
        }
        push_candidate(&mut sets, &mut seen, seeded);
    }

    let mut rng = Rng::new(cfg.search_seed ^ (step as u64 * 6361 + n_sig as u64 * 17 + 99));
    let mut stuck = 0;
    while sets.len() < cfg.n_proposals && !pool.is_empty() {
        let before = sets.len();
        let target = 2 + rng.pick(max_fan - 1);
        let mut cand = Vec::new();
        for _ in 0..pool.len() * 3 {
            if cand.len() >= target.min(pool.len()) { break; }
            let p = pool[rng.pick(pool.len())];
            if !cand.contains(&p) { cand.push(p); }
        }
        push_candidate(&mut sets, &mut seen, cand);
        if sets.len() == before { stuck += 1; if stuck > 32 { break; } } else { stuck = 0; }
    }

    let mut stuck = 0;
    while sets.len() < cfg.n_proposals && n_sig >= 2 {
        let before = sets.len();
        let target = 2 + rng.pick(max_fan - 1);
        let mut cand = Vec::new();
        for _ in 0..n_sig * 3 {
            if cand.len() >= target.min(n_sig) { break; }
            let p = rng.pick(n_sig);
            if !cand.contains(&p) { cand.push(p); }
        }
        push_candidate(&mut sets, &mut seen, cand);
        if sets.len() == before { stuck += 1; if stuck > 32 { break; } } else { stuck = 0; }
    }
    sets
}

fn print_scout(step: usize, ranked: &[SignalScout], pairs: &[PairScout], n_in: usize, proposal_sets: &[Vec<usize>]) {
    let top_sig: Vec<String> = ranked.iter().take(5).map(|s| {
        format!("{} s={:.3} sign={} |w|={:.2}", sig_name(s.idx, n_in), s.single_score, s.single_sign, s.probe_w.abs())
    }).collect();
    println!("    scout top: {}", top_sig.join(" | "));
    if !pairs.is_empty() {
        let top_pairs: Vec<String> = pairs.iter().take(3).map(|p| {
            format!("({},{}) +{:.3}", sig_name(p.a, n_in), sig_name(p.b, n_in), p.gain)
        }).collect();
        println!("    scout pairs: {}", top_pairs.join(" | "));
    }
    let cand_preview: Vec<String> = proposal_sets.iter().take(5).map(|set| {
        let names: Vec<String> = set.iter().map(|&p| sig_name(p, n_in)).collect();
        format!("[{}]", names.join(","))
    }).collect();
    println!("    scout cands: {}", cand_preview.join(" "));
    if step == usize::MAX { println!(); }
}

// ══════════════════════════════════════════════════════
// C19 PHASE B — finite-diff finetune of (c, rho)
// ══════════════════════════════════════════════════════
/// Weighted MSE loss over a precomputed dot vector and signed targets.
/// Kept as a free function (not a closure) so it can be called from inside
/// a rayon `par_iter` map without any closure-capture concerns.
fn c19_weighted_mse(dots: &[i32], targets: &[f32], sw: &[f32], c: f32, rho: f32) -> f32 {
    let mut l = 0.0f32;
    for (i, &d) in dots.iter().enumerate() {
        let out = c19(d as f32, c, rho);
        let e = out - targets[i];
        l += sw[i] * e * e;
    }
    l
}

/// Run `n_seeds` parallel finite-diff optimizations over weighted MSE loss.
/// Returns (c_best, rho_best, weighted_loss_best).
fn finetune_c_rho(
    parents: &[usize],
    weights: &[i8],
    all_sigs: &[Vec<u8>],
    data: &Data,
    sw: &[f32],
    n_seeds: usize,
    n_steps: usize,
    search_seed: u64,
) -> (f32, f32, f32) {
    let np = data.train.len();
    let dots: Vec<i32> = (0..np).map(|pi| {
        let mut d = 0i32;
        for (&w, &p) in weights.iter().zip(parents) {
            d += (w as i32) * (all_sigs[pi][p] as i32);
        }
        d
    }).collect();
    let targets: Vec<f32> = data.train.iter()
        .map(|(_, y)| if *y == 1 { 1.0f32 } else { -1.0 }).collect();

    // Borrow slices once so each rayon task captures &[T] (Sync) instead of
    // a closure that indirectly captures them.
    let dots_s: &[i32] = &dots;
    let targets_s: &[f32] = &targets;
    let sw_s: &[f32] = sw;

    // Run `n_seeds` parallel independent optimizations
    let seed_results: Vec<(f32, f32, f32)> = (0..n_seeds).into_par_iter().map(|seed_i| {
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
                    if att == 4 {
                        c = old_c;
                        rho = old_rho;
                    }
                }
            }
            if !improved { stale += 1; }
        }
        (best_c, best_rho, best_loss)
    }).collect();

    seed_results.into_iter()
        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
        .unwrap_or((1.0, 4.0, f32::MAX))
}

// ══════════════════════════════════════════════════════
// C19 PHASE C — quantization grid search
// ══════════════════════════════════════════════════════
const C_GRID: &[f32] = &[0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 2.5, 3.0];
const RHO_GRID: &[f32] = &[0.0, 1.0, 2.0, 4.0, 6.0, 8.0];

/// Evaluate a 10×6=60 grid of (c, rho) candidates and pick the best one that
/// stays within `tolerance` of the float-optimum loss. Returns the float values
/// unchanged if no grid candidate is within tolerance.
fn search_quant(
    parents: &[usize],
    weights: &[i8],
    c_float: f32,
    rho_float: f32,
    float_loss: f32,
    all_sigs: &[Vec<u8>],
    data: &Data,
    sw: &[f32],
    tolerance: f32,
) -> (f32, f32, f32) {
    let np = data.train.len();
    let dots: Vec<i32> = (0..np).map(|pi| {
        let mut d = 0i32;
        for (&w, &p) in weights.iter().zip(parents) {
            d += (w as i32) * (all_sigs[pi][p] as i32);
        }
        d
    }).collect();
    let targets: Vec<f32> = data.train.iter()
        .map(|(_, y)| if *y == 1 { 1.0f32 } else { -1.0 }).collect();

    let mut best_c = c_float;
    let mut best_rho = rho_float;
    let mut best_loss = float_loss;

    let mut candidates: Vec<(f32, f32, f32)> = Vec::with_capacity(C_GRID.len() * RHO_GRID.len());
    for &cg in C_GRID {
        for &rg in RHO_GRID {
            let l = c19_weighted_mse(&dots, &targets, sw, cg, rg);
            candidates.push((cg, rg, l));
        }
    }
    candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    if let Some(&(c, rho, l)) = candidates.first() {
        let max_allowed = (float_loss * (1.0 + tolerance)).max(float_loss + 1e-3);
        if l <= max_allowed {
            best_c = c;
            best_rho = rho;
            best_loss = l;
        }
    }

    (best_c, best_rho, best_loss)
}

// ══════════════════════════════════════════════════════
// C19 PHASE D — LUT bake
// ══════════════════════════════════════════════════════
/// Precompute a dense LUT over all integer dot values in [-abs_sum, +abs_sum].
fn bake_lut(weights: &[i8], c: f32, rho: f32) -> (Vec<f32>, i32) {
    let abs_sum: i32 = weights.iter().map(|w| w.abs() as i32).sum();
    let min_dot = -abs_sum;
    let size = (2 * abs_sum + 1) as usize;
    let mut lut = Vec::with_capacity(size);
    for d in min_dot..=abs_sum {
        lut.push(c19(d as f32, c, rho));
    }
    (lut, min_dot)
}

// ══════════════════════════════════════════════════════
// C19 PHASE D.5 — int8 LUT quantization (auto, per-neuron)
// ══════════════════════════════════════════════════════
//
// Per-neuron symmetric absmax int8 quant of a baked float LUT.
// Symmetric-absmax quantization recipe:
//   qmax = 127, qmin = -127 (NOT -128 — symmetric around zero)
//   scale = max(|lut|) / 127
//   lut_i8[i] = clip(round(lut[i] / scale), -127, 127)
//
// Degenerate case: if `max(|lut|) == 0`, returns `(vec![0; N], 1.0)` and the
// dequantized LUT is identically zero. This is the all-zero LUT that sometimes
// appears for neurons with `abs_sum == 0` (no active weights).
//
// Per the quant analysis, this is **lossless** on every c19 neuron tested
// (48/48 zero delta on train/val/test across 4 networks). It also preserves
// the `c19(0) == 0` identity bitwise, because `round(0.0 / scale) = 0` for
// any finite positive `scale`, so the zero-dot LUT slot is always exactly zero
// after dequantization.
fn bake_lut_i8(lut_f32: &[f32]) -> (Vec<i8>, f32) {
    if lut_f32.is_empty() {
        return (Vec::new(), 1.0);
    }
    let mut absmax = 0.0f32;
    for &v in lut_f32 {
        let a = v.abs();
        if a > absmax { absmax = a; }
    }
    if absmax == 0.0 {
        return (vec![0i8; lut_f32.len()], 1.0);
    }
    let scale = absmax / 127.0;
    let mut lut_i8 = Vec::with_capacity(lut_f32.len());
    for &v in lut_f32 {
        let qi = (v / scale).round() as i32;
        let clipped = qi.clamp(-127, 127) as i8;
        lut_i8.push(clipped);
    }
    (lut_i8, scale)
}

/// Per-network int16 alpha quantization (shared scale across all neurons).
///
/// Per-network symmetric int16 alpha quantization with shared scale (nbits=16):
/// `scale = max(|alpha|) / 32767`, with `alpha_i16[i] = clip(round(alpha[i]/scale), -32767, 32767)`.
///
/// The scale is **shared across the whole net**, so it must be recomputed any
/// time the set of neurons in the net changes (either after appending or
/// after a full reload from state.tsv). Returns the shared scale so the
/// caller can stamp it onto every neuron's `alpha_scale` field.
///
/// Degenerate case: all-zero alphas -> scale=1.0, alpha_i16=0.
///
/// Why i16 instead of i8: per the quant analysis, i16 alpha max_err is ~1e-5
/// (negligible) and gives ~256x more headroom than i8 alpha for networks with
/// a wider alpha dynamic range. i8 was also lossless on the 4 tested networks
/// but i16 is recommended as the production safety margin.
fn quantize_alphas_i16(net: &mut Net) -> f32 {
    let qmax: i32 = 32767;
    let qmin: i32 = -32767;
    let mut absmax = 0.0f32;
    for n in &net.neurons {
        let a = n.alpha.abs();
        if a > absmax { absmax = a; }
    }
    if absmax == 0.0 {
        for n in &mut net.neurons {
            n.alpha_i16 = 0;
            n.alpha_scale = 1.0;
        }
        return 1.0;
    }
    let scale = absmax / (qmax as f32);
    for n in &mut net.neurons {
        let qi = (n.alpha / scale).round() as i32;
        let clipped = qi.clamp(qmin, qmax) as i16;
        n.alpha_i16 = clipped;
        n.alpha_scale = scale;
    }
    scale
}

/// Re-derive (`lut_i8`, `lut_scale`) for every neuron in `net` from its
/// float LUT. Used by `load_state` to repopulate the transient fields after
/// a VER=2 state.tsv reload, and by the bake path as a convenience if the
/// caller wants to rebuild after mutating float LUTs wholesale.
fn rebake_all_i8(net: &mut Net) {
    for n in &mut net.neurons {
        let (lut_i8, scale) = bake_lut_i8(&n.lut);
        n.lut_i8 = lut_i8;
        n.lut_scale = scale;
    }
    let _ = quantize_alphas_i16(net);
}

// ══════════════════════════════════════════════════════
// PERSISTENT STATE (TSV) — schema VER=2, 15 columns per N row
// ══════════════════════════════════════════════════════
// HEAD\ttask\tdata_seed\tnoise\tn_per\tn_in\tVER=2
// N\tid\tparents\ttick\tweights\tthreshold\talpha\ttrain_acc\tval_acc\t
//   c_float\trho_float\tc_quant\trho_quant\tlut_min_dot\tlut(csv)
// Loader explicitly refuses files without VER=2 (old neuron_grower schema).

struct StateHead { task: String, data_seed: u64, noise: f32, n_per: usize, n_in: usize }

fn save_state(net: &Net, path: &str, head: &StateHead) -> std::io::Result<()> {
    if let Some(p) = std::path::Path::new(path).parent() { std::fs::create_dir_all(p)?; }
    // Write-to-tmp + atomic rename: POSIX rename() is a single inode swap,
    // so any failure / crash before the rename leaves the original state.tsv
    // untouched. After the rename, the new file is fully visible.
    let tmp = format!("{}.tmp", path);
    let mut f = std::fs::File::create(&tmp)?;
    writeln!(f, "HEAD\t{}\t{}\t{}\t{}\t{}\tVER=2",
        head.task, head.data_seed, head.noise, head.n_per, head.n_in)?;
    for n in &net.neurons {
        let ps: Vec<String> = n.parents.iter().map(|v| v.to_string()).collect();
        let ws: Vec<String> = n.weights.iter().map(|v| v.to_string()).collect();
        let lut: Vec<String> = n.lut.iter().map(|v| format!("{:.6}", v)).collect();
        writeln!(f,
            "N\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.2}\t{:.2}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}",
            n.id, ps.join(","), n.tick, ws.join(","), n.threshold,
            n.alpha, n.train_acc, n.val_acc,
            n.c_float, n.rho_float, n.c_quant, n.rho_quant,
            n.lut_min_dot, lut.join(",")
        )?;
    }
    f.sync_all()?; // fsync data + metadata before the rename
    drop(f);
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn load_state(path: &str) -> Result<Option<(StateHead, Net)>, String> {
    let s = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("failed to read state {}: {}", path, e)),
    };
    let mut lines = s.lines();
    let head_line = lines.next().ok_or_else(|| format!("empty state file: {}", path))?;
    let h: Vec<&str> = head_line.split('\t').collect();
    if h.is_empty() || h[0] != "HEAD" {
        return Err(format!("invalid state header in {} (not HEAD-prefixed)", path));
    }
    if h.len() < 7 || h[6] != "VER=2" {
        return Err(format!(
            "state file {} has no VER=2 — this is neuron_grower's old schema, use neuron_grower instead or delete state.tsv",
            path
        ));
    }
    let head = StateHead {
        task: h[1].to_string(),
        data_seed: h[2].parse().map_err(|_| format!("invalid data_seed in {}", path))?,
        noise: h[3].parse().map_err(|_| format!("invalid noise in {}", path))?,
        n_per: h[4].parse().map_err(|_| format!("invalid n_per in {}", path))?,
        n_in: h[5].parse().map_err(|_| format!("invalid n_in in {}", path))?,
    };
    let mut net = Net::new(head.n_in);
    for (ln, line) in lines.enumerate() {
        if line.is_empty() { continue; }
        let c: Vec<&str> = line.split('\t').collect();
        if c[0] != "N" { continue; }
        if c.len() != 15 {
            return Err(format!(
                "state schema mismatch at line {} in {} (expected 15 fields, got {})",
                ln + 2, path, c.len()
            ));
        }
        let parents: Vec<usize> = if c[2].is_empty() { Vec::new() }
            else { c[2].split(',').map(|v| v.parse()).collect::<Result<Vec<_>, _>>()
                .map_err(|_| format!("invalid parents at line {}", ln + 2))? };
        let weights: Vec<i8> = if c[4].is_empty() { Vec::new() }
            else { c[4].split(',').map(|v| v.parse()).collect::<Result<Vec<_>, _>>()
                .map_err(|_| format!("invalid weights at line {}", ln + 2))? };
        let lut: Vec<f32> = if c[14].is_empty() { Vec::new() }
            else { c[14].split(',').map(|v| v.parse()).collect::<Result<Vec<_>, _>>()
                .map_err(|_| format!("invalid lut at line {}", ln + 2))? };
        let n = Neuron {
            id: c[1].parse().map_err(|_| format!("invalid id at line {}", ln + 2))?,
            parents,
            tick: c[3].parse().map_err(|_| format!("invalid tick at line {}", ln + 2))?,
            weights,
            threshold: c[5].parse().map_err(|_| format!("invalid threshold at line {}", ln + 2))?,
            alpha: c[6].parse().map_err(|_| format!("invalid alpha at line {}", ln + 2))?,
            train_acc: c[7].parse().map_err(|_| format!("invalid train_acc at line {}", ln + 2))?,
            val_acc: c[8].parse().map_err(|_| format!("invalid val_acc at line {}", ln + 2))?,
            c_float: c[9].parse().map_err(|_| format!("invalid c_float at line {}", ln + 2))?,
            rho_float: c[10].parse().map_err(|_| format!("invalid rho_float at line {}", ln + 2))?,
            c_quant: c[11].parse().map_err(|_| format!("invalid c_quant at line {}", ln + 2))?,
            rho_quant: c[12].parse().map_err(|_| format!("invalid rho_quant at line {}", ln + 2))?,
            lut_min_dot: c[13].parse().map_err(|_| format!("invalid lut_min_dot at line {}", ln + 2))?,
            lut,
            // int8 fields are derived from the float LUT below — placeholder until
            // rebake_all_i8() rewrites them in a single post-load pass.
            lut_i8: Vec::new(),
            lut_scale: 1.0,
            alpha_i16: 0,
            alpha_scale: 1.0,
        };
        net.add(n);
    }
    // Repopulate transient int8 fields for every loaded neuron. Idempotent
    // and cheap (O(total_lut_entries)).
    rebake_all_i8(&mut net);
    Ok(Some((head, net)))
}

/// Replay AdaBoost sample weights on the loaded network. Uses the stored
/// `alpha` and the thresholded LUT output as the weak-learner prediction
/// (consistent with the accept gate used at growth time).
///
/// Superseded by `refit_alphas()` for forever-network mode — kept for
/// reference and possible single-task debug use.
#[allow(dead_code)]
fn replay_sw(net: &Net, data: &Data) -> Vec<f32> {
    let mut sw = vec![1.0 / data.train.len() as f32; data.train.len()];
    if net.neurons.is_empty() { return sw; }
    let all_sigs: Vec<Vec<u8>> = data.train.iter().map(|(x, _)| net.eval_all(x)).collect();
    for (ni, n) in net.neurons.iter().enumerate() {
        let sig_idx = net.n_in + ni;
        let alpha = n.alpha;
        let mut norm = 0.0f32;
        for (pi, ((_, y), wt)) in data.train.iter().zip(sw.iter_mut()).enumerate() {
            let pred = all_sigs[pi][sig_idx];
            let ys = if *y == 1 { 1.0 } else { -1.0 };
            let hs = if pred == 1 { 1.0 } else { -1.0 };
            *wt *= (-alpha * ys * hs).exp();
            norm += *wt;
        }
        if norm > 0.0 { for w in &mut sw { *w /= norm; } }
    }
    sw
}

/// Refit per-neuron AdaBoost alphas against the current task's labels.
///
/// This is the core "forever network" trick: the shared neuron body
/// (parents, weights, threshold, c/rho, LUT) is frozen — it computes
/// task-invariant features. Only the `alpha` coefficient — how much each
/// feature contributes to the ensemble sum — is task-specific. On a task
/// switch we recompute alphas from scratch so that the contribution of
/// each existing neuron matches its usefulness on the new task:
///
///   - Useful features (low weighted error) get high alpha.
///   - Useless or anti-correlated features (werr ≥ 0.499) get alpha = 0
///     and effectively vanish from the ensemble sum.
///   - Sample weights `sw` are overwritten in place via the same
///     sequential AdaBoost reweight that happens during growth, so the
///     result is identical to what `replay_sw` would give if the alphas
///     on disk had been tuned to the current task.
///
/// After this returns, callers should call `quantize_alphas_i16(net)` to
/// refresh the network-shared int16 alpha scale so the i8 runtime path
/// stays consistent.
fn refit_alphas(net: &mut Net, data: &Data, sw: &mut Vec<f32>) {
    // Start from uniform sample weights and sequentially refit.
    for w in sw.iter_mut() {
        *w = 1.0 / data.train.len() as f32;
    }
    if net.neurons.is_empty() {
        return;
    }

    // Precompute all per-sample signal outputs once. Safe because every
    // neuron's spike output is a pure function of its inputs and the
    // frozen topology — alpha only affects ensemble inference, not
    // per-neuron spike.
    let all_sigs: Vec<Vec<u8>> = data.train.iter().map(|(x, _)| net.eval_all(x)).collect();

    let n_neurons = net.neurons.len();
    let mut kept = 0usize;
    let mut zeroed = 0usize;
    for ni in 0..n_neurons {
        let sig_idx = net.n_in + ni;

        // Weighted error of this neuron on the current task.
        let mut werr = 0.0f32;
        for (pi, ((_, y), wt)) in data.train.iter().zip(sw.iter()).enumerate() {
            let pred = all_sigs[pi][sig_idx];
            if pred != *y {
                werr += *wt;
            }
        }

        // Alpha from the AdaBoost closed form, with the same ε clamp the
        // normal accept path uses. If werr ≥ 0.499 the neuron is worse
        // than random → alpha = 0 so it contributes nothing to the sum.
        let new_alpha = if werr >= 0.499 {
            zeroed += 1;
            0.0f32
        } else {
            kept += 1;
            0.5 * ((1.0 - werr).max(1e-6) / werr.max(1e-6)).ln()
        };

        net.neurons[ni].alpha = new_alpha;

        // AdaBoost reweight of `sw` for the NEXT neuron in the chain. A
        // zero-alpha neuron contributes no reweight (exp(0) = 1), which
        // is the correct "pass-through" semantics.
        if new_alpha != 0.0 {
            let mut norm = 0.0f32;
            for (pi, ((_, y), wt)) in data.train.iter().zip(sw.iter_mut()).enumerate() {
                let pred = all_sigs[pi][sig_idx];
                let ys = if *y == 1 { 1.0 } else { -1.0 };
                let hs = if pred == 1 { 1.0 } else { -1.0 };
                *wt *= (-new_alpha * ys * hs).exp();
                norm += *wt;
            }
            if norm > 0.0 {
                for w in sw.iter_mut() {
                    *w /= norm;
                }
            }
        }
    }

    println!(
        "  [refit_alphas] {} neurons: {} kept (task-useful), {} zeroed (werr>=0.499)",
        n_neurons, kept, zeroed
    );
}

// ══════════════════════════════════════════════════════
// CHECKPOINT (per-step JSON snapshot — human-readable, not used for load)
// ══════════════════════════════════════════════════════
fn save_checkpoint(net: &Net, path: &str, task: &str, step: usize, data: &Data) -> std::io::Result<()> {
    if let Some(p) = std::path::Path::new(path).parent() { std::fs::create_dir_all(p)?; }
    let et = net.accuracy(&data.train); let ev = net.accuracy(&data.val); let ete = net.accuracy(&data.test);
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "{{")?;
    writeln!(f, "\"task\":\"{}\",\"step\":{},\"n_inputs\":{},", task, step, net.n_in)?;
    writeln!(f, "\"ensemble_train\":{:.2},\"ensemble_val\":{:.2},\"ensemble_test\":{:.2},", et, ev, ete)?;
    writeln!(f, "\"neurons\":[")?;
    for (i, n) in net.neurons.iter().enumerate() {
        let wj: Vec<String> = n.weights.iter().map(|v| v.to_string()).collect();
        let pj: Vec<String> = n.parents.iter().map(|v| v.to_string()).collect();
        let lutj: Vec<String> = n.lut.iter().map(|v| format!("{:.6}", v)).collect();
        writeln!(f,
            "{{\"id\":{},\"parents\":[{}],\"tick\":{},\"weights\":[{}],\"threshold\":{},\"alpha\":{:.6},\"train_acc\":{:.2},\"val_acc\":{:.2},\"c_float\":{:.6},\"rho_float\":{:.6},\"c_quant\":{:.6},\"rho_quant\":{:.6},\"lut_min_dot\":{},\"lut\":[{}]}}{}",
            n.id, pj.join(","), n.tick, wj.join(","), n.threshold, n.alpha, n.train_acc, n.val_acc,
            n.c_float, n.rho_float, n.c_quant, n.rho_quant, n.lut_min_dot, lutj.join(","),
            if i < net.neurons.len()-1 { "," } else { "" })?;
    }
    writeln!(f, "]}}")?;
    f.sync_all()?;
    Ok(())
}

// ══════════════════════════════════════════════════════
// TRAINING TRACE RECORDER — c19_grower.v1 schema
// ══════════════════════════════════════════════════════
struct TraceEvent {
    event: &'static str,
    tick: u32,
    id: usize,
    parents: Vec<usize>,
    weights: Vec<i8>,
    threshold: i32,
    alpha: f32,
    train_acc: f32,
    val_acc: f32,
    c_float: f32,
    rho_float: f32,
    c_quant: f32,
    rho_quant: f32,
    lut_min_dot: i32,
    lut: Vec<f32>,
    finetune_loss: f32,
    quant_loss: f32,
    // int8 LUT + int16 alpha quant artifacts (derived, not persisted to state.tsv)
    lut_i8: Vec<i8>,
    lut_scale: f32,
    alpha_i16: i16,
    alpha_scale: f32,
}

#[allow(clippy::too_many_arguments)]
fn write_trace_json(
    out_dir: &str,
    cfg: &Config,
    n_in: usize,
    events: &[TraceEvent],
    final_val: f32,
    final_test: f32,
    final_neurons: usize,
    max_depth: u32,
    stall: usize,
) -> std::io::Result<()> {
    let path = format!("{}/trace.json", out_dir);
    if let Some(p) = std::path::Path::new(&path).parent() { std::fs::create_dir_all(p)?; }
    let mut f = std::fs::File::create(&path)?;
    writeln!(f, "{{")?;
    writeln!(f, "  \"schema\": \"c19_grower.v1\",")?;
    writeln!(f, "  \"c19_grower\": true,")?;
    writeln!(f, "  \"task\": \"{}\",", cfg.task)?;
    writeln!(f, "  \"data_seed\": {},", cfg.data_seed)?;
    writeln!(f, "  \"search_seed\": {},", cfg.search_seed)?;
    writeln!(f, "  \"n_in\": {},", n_in)?;
    writeln!(f, "  \"n_per\": {},", cfg.n_per)?;
    writeln!(f, "  \"noise\": {},", cfg.noise)?;
    writeln!(f, "  \"config\": {{")?;
    writeln!(f, "    \"finetune_seeds\": {},", cfg.finetune_seeds)?;
    writeln!(f, "    \"finetune_steps\": {},", cfg.finetune_steps)?;
    writeln!(f, "    \"quant_tolerance\": {:.6},", cfg.quant_tolerance)?;
    writeln!(f, "    \"skip_finetune\": {},", cfg.skip_finetune)?;
    writeln!(f, "    \"skip_quant\": {}", cfg.skip_quant)?;
    writeln!(f, "  }},")?;
    writeln!(f, "  \"events\": [")?;
    for (i, e) in events.iter().enumerate() {
        let ps: Vec<String> = e.parents.iter().map(|v| v.to_string()).collect();
        let ws: Vec<String> = e.weights.iter().map(|v| v.to_string()).collect();
        let lutj: Vec<String> = e.lut.iter().map(|v| format!("{:.6}", v)).collect();
        let lut_i8_j: Vec<String> = e.lut_i8.iter().map(|v| v.to_string()).collect();
        let comma = if i + 1 < events.len() { "," } else { "" };
        writeln!(f,
            "    {{\"event\":\"{}\",\"tick\":{},\"id\":{},\"parents\":[{}],\"weights\":[{}],\"threshold\":{},\"alpha\":{:.6},\"train_acc\":{:.1},\"val_acc\":{:.1},\"c_float\":{:.6},\"rho_float\":{:.6},\"c_quant\":{:.6},\"rho_quant\":{:.6},\"lut_min_dot\":{},\"lut_size\":{},\"lut\":[{}],\"finetune_loss\":{:.6},\"quant_loss\":{:.6},\"lut_i8\":[{}],\"lut_scale\":{:.9},\"alpha_i16\":{},\"alpha_scale\":{:.9}}}{}",
            e.event, e.tick, e.id, ps.join(","), ws.join(","),
            e.threshold, e.alpha, e.train_acc, e.val_acc,
            e.c_float, e.rho_float, e.c_quant, e.rho_quant, e.lut_min_dot,
            e.lut.len(), lutj.join(","), e.finetune_loss, e.quant_loss,
            lut_i8_j.join(","), e.lut_scale, e.alpha_i16, e.alpha_scale,
            comma
        )?;
    }
    writeln!(f, "  ],")?;
    writeln!(f, "  \"final\": {{")?;
    writeln!(f, "    \"best_val_acc\": {:.1},", final_val)?;
    writeln!(f, "    \"best_test_acc\": {:.1},", final_test)?;
    writeln!(f, "    \"total_neurons\": {},", final_neurons)?;
    writeln!(f, "    \"max_depth\": {},", max_depth)?;
    writeln!(f, "    \"stall_count\": {}", stall)?;
    writeln!(f, "  }}")?;
    writeln!(f, "}}")?;
    Ok(())
}

// ══════════════════════════════════════════════════════
// LABEL FUNCTIONS (one per task)
// ══════════════════════════════════════════════════════
//
// Returns `None` on unknown task — caller decides whether to exit or just
// skip (task-list mode prefers graceful skip with a warning over hard exit).
fn make_label_fn(task: &str) -> Option<Box<dyn Fn(usize, &[u8]) -> Option<u8>>> {
    Some(match task {
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
        "grid3_diag_xor" => Box::new(|_, px| {
            let parity = (px[0] ^ px[4] ^ px[8]) & 1;
            Some(parity)
        }),
        "grid3_full_parity" => Box::new(|_, px| {
            let parity =
                (px[0] ^ px[1] ^ px[2] ^
                 px[3] ^ px[4] ^ px[5] ^
                 px[6] ^ px[7] ^ px[8]) & 1;
            Some(parity)
        }),
        "grid3_majority" => Box::new(|_, px| {
            let s: usize = px.iter().map(|&v| v as usize).sum();
            Some(if s >= 5 { 1 } else { 0 })
        }),
        "grid3_symmetry_h" => Box::new(|_, px| {
            let sym = px[0] == px[2] && px[3] == px[5] && px[6] == px[8];
            Some(if sym { 1 } else { 0 })
        }),
        "grid3_top_heavy" => Box::new(|_, px| {
            let top: usize = (px[0] as usize) + (px[1] as usize) + (px[2] as usize);
            let bot: usize = (px[6] as usize) + (px[7] as usize) + (px[8] as usize);
            Some(if top > bot { 1 } else { 0 })
        }),
        "grid3_copy_bit_0" => Box::new(|_, px| Some(px[0])),
        "grid3_copy_bit_1" => Box::new(|_, px| Some(px[1])),
        "grid3_copy_bit_2" => Box::new(|_, px| Some(px[2])),
        "grid3_copy_bit_3" => Box::new(|_, px| Some(px[3])),
        "grid3_copy_bit_4" => Box::new(|_, px| Some(px[4])),
        "grid3_copy_bit_5" => Box::new(|_, px| Some(px[5])),
        "grid3_copy_bit_6" => Box::new(|_, px| Some(px[6])),
        "grid3_copy_bit_7" => Box::new(|_, px| Some(px[7])),
        "grid3_copy_bit_8" => Box::new(|_, px| Some(px[8])),
        _ => return None,
    })
}

// ══════════════════════════════════════════════════════
// INTERACTIVE PICK (stdin prompt in --interactive mode)
// ══════════════════════════════════════════════════════
enum Pick { Auto, Only(usize), Skip }

fn read_interactive_pick(n_candidates: usize) -> Pick {
    use std::io::{self, Write};
    if n_candidates == 0 { return Pick::Skip; }
    print!(
        "  pick [0-{}] / 'a' auto-top / 's' skip: ",
        n_candidates - 1
    );
    let _ = io::stdout().flush();
    let mut line = String::new();
    if io::stdin().read_line(&mut line).is_err() {
        return Pick::Auto;
    }
    let line = line.trim();
    match line {
        "" | "a" | "A" => Pick::Auto,
        "s" | "S" => Pick::Skip,
        _ => match line.parse::<usize>() {
            Ok(k) if k < n_candidates => Pick::Only(k),
            _ => {
                println!("  (invalid input '{}' — falling back to auto)", line);
                Pick::Auto
            }
        }
    }
}

// ══════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════
fn main() {
    let cfg = parse_args();
    let t0 = Instant::now();

    // Resolve the task schedule: either a single task (--task) or a list
    // (--task-list t1,t2,t3) for forever-network mode. The list mode grows
    // ONE network across all listed tasks; each task runs until 100% val
    // accuracy or stall/cap, then the next task starts with the same net.
    let tasks: Vec<String> = if cfg.task_list.is_empty() {
        vec![cfg.task.clone()]
    } else {
        cfg.task_list.clone()
    };

    // All grid3_* tasks share n_in=9. The task_n_in check below enforces that
    // every task in the list agrees — forever-network mode requires identical
    // input width, otherwise the frozen hidden-parent indices would shift.
    let n_in = task_n_in(&tasks[0]);
    for t in &tasks {
        let n_i = task_n_in(t);
        if n_i != n_in {
            eprintln!(
                "ERROR: task-list has mixed n_in — {} has {}, {} has {}",
                tasks[0], n_in, t, n_i
            );
            std::process::exit(2);
        }
        if make_label_fn(t).is_none() {
            eprintln!("ERROR: unknown task '{}' (expected one of the grid3_* tasks)", t);
            std::process::exit(2);
        }
    }

    let mut noise = cfg.noise;
    let mut n_per = cfg.n_per;
    let mut data_seed = cfg.data_seed;

    std::fs::create_dir_all(&cfg.out_dir).unwrap_or_else(|e| {
        eprintln!("ERROR: failed to create out-dir '{}': {}", cfg.out_dir, e);
        std::process::exit(1);
    });
    let state_path = format!("{}/state.tsv", cfg.out_dir);

    // In forever-network mode (--task-list or --allow-task-switch) we permit
    // loading a state whose task name differs from the CLI — the loaded
    // neurons become frozen hidden parents for the new task(s). The n_in
    // check is still strict because a different input width would shift all
    // hidden-parent indices.
    let task_switch_ok = !cfg.task_list.is_empty() || cfg.allow_task_switch;
    let (mut net, _loaded) = match load_state(&state_path) {
        Ok(Some((head, net))) => {
            if head.task != cfg.task && !task_switch_ok {
                eprintln!(
                    "ERROR: state task={} != cli task={} (use --allow-task-switch or --task-list to override)",
                    head.task, cfg.task
                );
                std::process::exit(2);
            }
            if head.n_in != n_in {
                eprintln!("ERROR: state n_in={} != task {} expected n_in={}", head.n_in, tasks[0], n_in);
                std::process::exit(2);
            }
            if data_seed != head.data_seed || (noise - head.noise).abs() > 1e-6 || n_per != head.n_per {
                println!("  [load] state data params override CLI: seed {}->{} noise {}->{} n_per {}->{}",
                    data_seed, head.data_seed, noise, head.noise, n_per, head.n_per);
            }
            data_seed = head.data_seed;
            noise = head.noise;
            n_per = head.n_per;
            println!("  [load] {} neurons from {} (state.task={})",
                net.neurons.len(), state_path, head.task);
            (net, true)
        }
        Ok(None) => {
            println!("  [load] no prior state at {} — starting fresh", state_path);
            (Net::new(n_in), false)
        }
        Err(e) => {
            eprintln!("ERROR: {}", e);
            std::process::exit(2);
        }
    };
    // Runtime dispatch toggle: picked up from CLI (default true). `load_state`
    // already called `rebake_all_i8` internally, but it constructs a fresh Net
    // with `use_i8: true`; we re-stamp here so the CLI override wins.
    net.use_i8 = cfg.use_i8;

    // Head + trace state live across tasks: they describe the CURRENT task
    // at save time, while the forever-network keeps growing underneath.
    let mut head = StateHead {
        task: tasks[0].clone(),
        data_seed,
        noise,
        n_per,
        n_in: net.n_in,
    };
    let mut trace_events: Vec<TraceEvent> = Vec::new();

    println!("===========================================================");
    if tasks.len() == 1 {
        println!("  c19_grower — {} (data_seed={} search_seed={})",
            tasks[0], cfg.data_seed, cfg.search_seed);
    } else {
        println!("  c19_grower FOREVER — {} tasks (data_seed={} search_seed={})",
            tasks.len(), cfg.data_seed, cfg.search_seed);
        for (i, t) in tasks.iter().enumerate() {
            println!("    task[{:2}] = {}", i, t);
        }
    }
    println!("  {} proposals/step, max_fan={}, stall={}, max_neurons={} (per task)",
        cfg.n_proposals, cfg.max_fan, cfg.stall_limit, cfg.max_neurons);
    println!("  scout_top={} pair_top={} probe_epochs={}",
        cfg.scout_top, cfg.pair_top, cfg.probe_epochs);
    println!("  c19: finetune_seeds={} finetune_steps={} quant_tol={:.4} skip_ft={} skip_q={}",
        cfg.finetune_seeds, cfg.finetune_steps, cfg.quant_tolerance, cfg.skip_finetune, cfg.skip_quant);
    println!("  i8 LUT: use_i8={} assert_tol={}pp (per-neuron absmax int8 + i16 alpha)",
        cfg.use_i8, cfg.i8_assert_tol_pp);
    println!("  flags: interactive={} exhaustive={} verbose_search={} allow_task_switch={} force_pick={:?} preview_only={} bake_best={}",
        cfg.interactive, cfg.exhaustive, cfg.verbose_search, cfg.allow_task_switch,
        cfg.force_pick, cfg.preview_only, cfg.bake_best);
    println!("===========================================================");

    // Carry the last-task data out of the task loop for the final report.
    let mut final_data_opt: Option<Data> = None;
    let mut final_stall: usize = 0;

    'task_loop: for (task_idx, task_name) in tasks.iter().enumerate() {
        // Rebuild label_fn + data for this task. Previous task's neurons
        // remain in `net` as frozen hidden parents — new neurons may pick them.
        let label_fn = make_label_fn(task_name).expect("task validated above");
        let data = gen_data(label_fn.as_ref(), noise, n_per, data_seed);

        // AdaBoost sample weights + alpha refit: the per-neuron alpha
        // coefficients are task-specific, so on every task entry we
        // refit them against the current task's labels. This also
        // rebuilds sw (the sequential sample weights). After refit we
        // requantize the network-shared i16 alpha scale so the i8
        // runtime path reflects the new alphas. For a fresh (unloaded)
        // net this is a no-op (no neurons to refit).
        //
        // Critical: without this step, old neurons' frozen alphas
        // dominate the ensemble sum and pollute the new task's
        // predictions. See the 2026-04-13 forever-network debug.
        let mut sw = vec![1.0 / data.train.len() as f32; data.train.len()];
        refit_alphas(&mut net, &data, &mut sw);
        quantize_alphas_i16(&mut net);

        let mut stall: usize = 0;
        let mut best_val = net.accuracy(&data.val);

        // Persist the CURRENT task name in the state file header so a
        // restart with --allow-task-switch can locate where it left off.
        head.task = task_name.clone();

        println!("\n===========================================================");
        println!("  TASK [{:2}/{:2}] {} | start: {} neurons, val={:.1}%",
            task_idx + 1, tasks.len(), task_name, net.neurons.len(), best_val);
        println!("  Data: {} train / {} val / {} test",
            data.train.len(), data.val.len(), data.test.len());
        println!("===========================================================");

        for step in 0..cfg.max_neurons {
        let t_step = Instant::now();
        let ens_val = net.accuracy(&data.val);
        let ens_test = net.accuracy(&data.test);
        println!("\n  Step {:3} | {} neurons | val={:.1}% test={:.1}% | {:.0}s elapsed",
            step, net.neurons.len(), ens_val, ens_test, t0.elapsed().as_secs_f64());

        if ens_val >= 99.0 { println!("  >> Target reached!"); break; }

        // Precompute signals (byte-level — hidden parents are thresholded)
        let all_sigs: Vec<Vec<u8>> = data.train.iter().map(|(x, _)| net.eval_all(x)).collect();
        let all_val_sigs: Vec<Vec<u8>> = data.val.iter().map(|(x, _)| net.eval_all(x)).collect();
        let n_sig = net.n_sig();

        // STEP A: Scout promising parents
        let single_scores = best_single_signal_scores(&all_sigs, &data, &sw, n_sig);
        let (probe_w, probe_b) = backprop_probe_all(&all_sigs, &data, &sw, n_sig, cfg.probe_epochs);
        let ranked = merge_signal_ranks(single_scores, &probe_w);
        let pairs = pair_lifts_from_ranked(&ranked, &all_sigs, &data, &sw, cfg.pair_top);
        let proposal_sets = build_candidate_sets(&ranked, &pairs, n_sig, net.n_in, &cfg, step);
        print_scout(step, &ranked, &pairs, net.n_in, &proposal_sets);
        if cfg.verbose_search {
            println!("    [verbose] scout: {} ranked signals, {} pair lifts, {} proposal sets",
                ranked.len(), pairs.len(), proposal_sets.len());
            for (i, ps) in proposal_sets.iter().enumerate().take(10) {
                let names: Vec<String> = ps.iter().map(|&p| sig_name(p, net.n_in)).collect();
                println!("      proposal[{:2}] parents=[{}]", i, names.join(","));
            }
            if proposal_sets.len() > 10 {
                println!("      ... ({} more)", proposal_sets.len() - 10);
            }
        }

        // STEP B: Rough proposal scoring (fractional weights, quick sigmoid eval)
        let mut proposals: Vec<(Vec<usize>, Vec<f32>, f32, f32)> = Vec::new();
        for (seed, parents) in proposal_sets.iter().enumerate() {
            let ni = parents.len();
            let mut rng = Rng::new(cfg.search_seed ^ (seed as u64 * 7919 + 31 + step as u64 * 104729));
            let w: Vec<f32> = parents.iter().map(|&p| {
                let base = probe_w[p];
                if base.abs() > 0.05 { base + rng.range(-0.15, 0.15) }
                else {
                    let s = ranked.iter().find(|r| r.idx == p).map(|r| r.single_sign as f32).unwrap_or(1.0);
                    s * rng.range(0.1, 0.8)
                }
            }).collect();
            let b = probe_b + rng.range(-0.2, 0.2);
            let score: f32 = data.train.iter().enumerate().map(|(pi, (_, y))| {
                let z: f32 = b + (0..ni).map(|i| w[i] * all_sigs[pi][parents[i]] as f32).sum::<f32>();
                if (if sigmoid(z) > 0.5 { 1u8 } else { 0 }) == *y { sw[pi] } else { 0.0 }
            }).sum();
            proposals.push((parents.clone(), w, b, score));
        }
        proposals.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

        if proposals.is_empty() { println!("    X No proposals"); break; }

        // STEP C: Backprop top-5 with 5 restarts each for consensus signs
        let top_k = proposals.len().min(5);
        struct Trained { parents: Vec<usize>, val_acc: f32, consensus: Vec<i8> }
        let mut trained: Vec<Trained> = Vec::new();

        for pi in 0..top_k {
            let (ref parents, ref init_w, init_b, _) = proposals[pi];
            let ni = parents.len();
            let mut all_converged: Vec<(Vec<f32>, f32)> = Vec::new();

            for restart in 0..5u64 {
                let mut rng = Rng::new(cfg.search_seed ^ (restart * 1000 + 77));
                let mut w: Vec<f32> = if restart == 0 { init_w.clone() }
                    else { init_w.iter().map(|&v| v + rng.range(-0.5, 0.5)).collect() };
                let mut b = if restart == 0 { init_b } else { init_b + rng.range(-0.3, 0.3) };

                for _ in 0..2000 {
                    for (pii, (_, y)) in data.train.iter().enumerate() {
                        let z: f32 = b + (0..ni).map(|i| w[i] * all_sigs[pii][parents[i]] as f32).sum::<f32>();
                        let a = sigmoid(z);
                        let g = (a - *y as f32) * a * (1.0 - a) * sw[pii] * data.train.len() as f32;
                        for i in 0..ni { w[i] -= 0.5 * g * all_sigs[pii][parents[i]] as f32; }
                        b -= 0.5 * g;
                    }
                }
                all_converged.push((w, b));
            }

            let consensus: Vec<i8> = (0..ni).map(|i| {
                let pos = all_converged.iter().filter(|(w, _)| w[i] > 0.3).count();
                let neg = all_converged.iter().filter(|(w, _)| w[i] < -0.3).count();
                if pos * 10 / 5 >= 7 { 1 } else if neg * 10 / 5 >= 7 { -1 } else { 2 }
            }).collect();

            let best_w = &all_converged[0].0; let best_b = all_converged[0].1;
            let val_acc = {
                let c = data.val.iter().enumerate().filter(|(vi, (_, y))| {
                    let z: f32 = best_b + (0..ni).map(|i| best_w[i] * all_val_sigs[*vi][parents[i]] as f32).sum::<f32>();
                    (if sigmoid(z) > 0.5 { 1u8 } else { 0 }) == *y
                }).count();
                c as f32 / data.val.len() as f32 * 100.0
            };

            trained.push(Trained { parents: parents.clone(), val_acc, consensus });
        }
        trained.sort_by(|a, b| b.val_acc.partial_cmp(&a.val_acc).unwrap());

        // Pretty-print the trained candidate table whenever any manual-pick
        // mode is on (verbose, interactive, preview-only, force-pick, or
        // bake-best), so the user always sees what they're choosing from.
        // In pure auto mode stay silent to match the old overnight log format.
        let show_table = cfg.verbose_search || cfg.interactive
            || cfg.preview_only || cfg.force_pick.is_some() || cfg.bake_best;
        if show_table {
            println!("  Trained candidates ({}):", trained.len());
            println!("    {:>3} | {:>6} | {:>2} | consensus | parents",
                "idx", "val%", "ni");
            for (i, tp) in trained.iter().enumerate() {
                let cs: String = tp.consensus.iter().map(|&s| match s {
                    1 => '+', -1 => '-', _ => '?',
                }).collect();
                let pnames: Vec<String> = tp.parents.iter()
                    .map(|&p| sig_name(p, net.n_in))
                    .collect();
                println!("    {:>3} | {:>5.1}% | {:>2} |   {:<7} | [{}]",
                    i, tp.val_acc, tp.parents.len(), cs, pnames.join(","));
            }
        }

        // --preview-only short-circuits BEFORE the bake: useful for a two-step
        // manual workflow. The first run prints the trained distribution and
        // exits; the user picks an idx and re-runs with --force-pick N.
        if cfg.preview_only {
            println!("    (--preview-only) distribution printed; exiting without baking.");
            let _ = std::io::stdout().flush();
            std::process::exit(0);
        }

        // Decide which trained candidate(s) to try baking:
        // 1. --bake-best → try ALL in tentative-bake mode, commit best after loop
        // 2. --interactive → stdin prompt (single pick)
        // 3. --force-pick N → only trained[N]
        // 4. otherwise → auto-top order (existing behavior, first-accept wins)
        let pick = if cfg.bake_best {
            // Iterate every candidate; the in-loop gate below will do
            // tentative add+measure+pop and track the best one externally.
            Pick::Auto
        } else if cfg.interactive {
            read_interactive_pick(trained.len())
        } else if let Some(k) = cfg.force_pick {
            if k < trained.len() {
                println!("    (--force-pick {}) committing trained[{}]", k, k);
                Pick::Only(k)
            } else {
                eprintln!(
                    "WARN: --force-pick {} out of range (trained has {} candidates), falling back to auto",
                    k, trained.len()
                );
                Pick::Auto
            }
        } else {
            Pick::Auto
        };
        let pick_order: Vec<usize> = match &pick {
            Pick::Auto => (0..trained.len()).collect(),
            Pick::Only(k) => vec![*k],
            Pick::Skip => {
                println!("    (skipped by user — no neuron added this step)");
                Vec::new()
            }
        };

        // STEP D+E+F: Ternary → c19 finetune → quant → LUT bake → accept gate
        let mut accepted = false;

        // --bake-best accumulator: saves the (new_val, pick_i, Neuron, ft_loss,
        // q_loss) of the best-baking candidate seen so far. Each successful
        // tentative bake is pop-ed after measurement so the next candidate
        // starts from the same step-start net state. After the loop, the
        // winner is re-added permanently through a mirror of the accept path.
        let mut best_baked: Option<(f32, usize, Neuron, f32, f32)> = None;

        for &pick_i in &pick_order {
            let tp = &trained[pick_i];
            if cfg.verbose_search {
                println!("    [bake] trying candidate idx={} val={:.1}%",
                    pick_i, tp.val_acc);
            }
            let ni = tp.parents.len();
            let np = data.train.len();

            // Ternary guided + blind search (bias-free threshold form)
            let locked: Vec<Option<i8>> = tp.consensus.iter().map(|&s| {
                if s == 1 { Some(1) } else if s == -1 { Some(-1) } else { None }
            }).collect();
            let free_pos: Vec<usize> = (0..ni).filter(|&i| locked[i].is_none()).collect();

            let mut bw = vec![0i8; ni]; let mut bt: i32 = 0;
            let mut bs = -1.0f32;

            let total_free = 3u64.pow(free_pos.len() as u32);
            for combo in 0..total_free {
                let mut w = vec![0i8; ni];
                for i in 0..ni { w[i] = locked[i].unwrap_or(0); }
                let mut r = combo;
                for &fp in &free_pos { w[fp] = (r % 3) as i8 - 1; r /= 3; }
                let _ = r;
                let dots: Vec<i32> = (0..np).map(|pi| {
                    let mut d = 0i32;
                    for (j, &pidx) in tp.parents.iter().enumerate() { d += (w[j] as i32) * (all_sigs[pi][pidx] as i32); }
                    d
                }).collect();
                let mn = dots.iter().copied().min().unwrap_or(0);
                let mx = dots.iter().copied().max().unwrap_or(0);
                for threshold in (mn-1)..=(mx+1) {
                    let outs: Vec<u8> = dots.iter().map(|&d| if d >= threshold { 1 } else { 0 }).collect();
                    let sc: f32 = outs.iter().zip(&data.train).zip(&sw)
                        .map(|((&pred, (_, y)), &wt)| if pred == *y { wt } else { 0.0 }).sum();
                    if sc > bs { bs=sc; bw=w.clone(); bt=threshold; }
                }
            }

            let total_blind = 3u64.pow(ni as u32);
            // --exhaustive lifts the 500K cap unconditionally; otherwise only
            // run the blind ternary sweep when the combo space is tractable.
            let run_blind = cfg.exhaustive || total_blind <= 500_000;
            if cfg.verbose_search {
                println!("    [ternary] ni={} total_blind=3^{}={} run_blind={} (cap=500K, exhaustive={})",
                    ni, ni, total_blind, run_blind, cfg.exhaustive);
            }
            if run_blind {
                for combo in 0..total_blind {
                    let mut w = vec![0i8; ni]; let mut r = combo;
                    for wi in w.iter_mut() { *wi = (r % 3) as i8 - 1; r /= 3; }
                    let _ = r;
                    let dots: Vec<i32> = (0..np).map(|pi| {
                        let mut d = 0i32;
                        for (j, &pidx) in tp.parents.iter().enumerate() { d += (w[j] as i32) * (all_sigs[pi][pidx] as i32); }
                        d
                    }).collect();
                    let mn = dots.iter().copied().min().unwrap_or(0);
                    let mx = dots.iter().copied().max().unwrap_or(0);
                    for threshold in (mn-1)..=(mx+1) {
                        let outs: Vec<u8> = dots.iter().map(|&d| if d >= threshold { 1 } else { 0 }).collect();
                        let sc: f32 = outs.iter().zip(&data.train).zip(&sw)
                            .map(|((&pred, (_, y)), &wt)| if pred == *y { wt } else { 0.0 }).sum();
                        if sc > bs { bs=sc; bw=w.clone(); bt=threshold; }
                    }
                }
            }

            // ── Phase B: finite-diff finetune of (c, rho) ────────────
            let (c_float, rho_float, ft_loss) = if cfg.skip_finetune {
                (1.0f32, 4.0f32, f32::NAN)
            } else {
                finetune_c_rho(
                    &tp.parents, &bw, &all_sigs, &data, &sw,
                    cfg.finetune_seeds, cfg.finetune_steps, cfg.search_seed,
                )
            };

            // ── Phase C: quantization grid search ────────────────────
            let (c_quant, rho_quant, q_loss) = if cfg.skip_quant {
                (c_float, rho_float, ft_loss)
            } else {
                let float_loss = if ft_loss.is_finite() { ft_loss } else {
                    // Compute loss for (c_float, rho_float) directly so tolerance check works
                    let dots_raw: Vec<i32> = (0..np).map(|pi| {
                        let mut d = 0i32;
                        for (&w, &p) in bw.iter().zip(&tp.parents) {
                            d += (w as i32) * (all_sigs[pi][p] as i32);
                        }
                        d
                    }).collect();
                    let mut l = 0.0f32;
                    for (i, &d) in dots_raw.iter().enumerate() {
                        let out = c19(d as f32, c_float, rho_float);
                        let ys = if data.train[i].1 == 1 { 1.0f32 } else { -1.0 };
                        let e = out - ys;
                        l += sw[i] * e * e;
                    }
                    l
                };
                search_quant(
                    &tp.parents, &bw, c_float, rho_float, float_loss,
                    &all_sigs, &data, &sw, cfg.quant_tolerance,
                )
            };

            // ── Phase D: LUT bake ────────────────────────────────────
            let (lut, lut_min_dot) = bake_lut(&bw, c_quant, rho_quant);

            // ── Phase D.5: int8 LUT auto-quant (per-neuron absmax) ────
            // See .claude/research/c19_i8_quant_analysis.md for proof of losslessness.
            // bake_lut_i8 handles the all-zero degenerate case internally.
            let (lut_i8, lut_scale) = bake_lut_i8(&lut);

            // ── LUT-thresholded predictions (replace `bo` for AdaBoost) ─
            let bo_lut: Vec<u8> = (0..np).map(|pi| {
                let mut d = 0i32;
                for (&w, &p) in bw.iter().zip(&tp.parents) {
                    d += (w as i32) * (all_sigs[pi][p] as i32);
                }
                let idx = d - lut_min_dot;
                let val = if idx < 0 { 0.0 }
                    else if (idx as usize) >= lut.len() { 0.0 }
                    else { lut[idx as usize] };
                if val >= 0.0 { 1u8 } else { 0 }
            }).collect();

            // Dedup check on LUT predictions
            let is_dup = (net.n_in..n_sig).any(|e| output_match_rate(&bo_lut, &all_sigs, e) >= 0.999);
            if is_dup {
                if cfg.verbose_search {
                    println!("      [reject] idx={} duplicate of existing neuron (LUT output match >= 99.9%)", pick_i);
                }
                continue;
            }

            let tick = tp.parents.iter().map(|&p| net.sig_ticks[p]).max().unwrap_or(0) + 1;
            let werr: f32 = bo_lut.iter().zip(&data.train).zip(&sw)
                .map(|((&pred, (_, y)), &w)| if pred == *y { 0.0 } else { w }).sum();
            if werr >= 0.499 {
                if cfg.verbose_search {
                    println!("      [reject] idx={} weighted error {:.4} >= 0.499 (AdaBoost fail)", pick_i, werr);
                }
                continue;
            }
            let alpha = 0.5 * ((1.0 - werr).max(1e-6) / werr.max(1e-6)).ln();

            // Train / val accuracy from the LUT-thresholded predictions
            let qr_train = {
                let c = bo_lut.iter().zip(&data.train).filter(|(&p, (_, y))| p == *y).count();
                c as f32 / data.train.len() as f32 * 100.0
            };
            let qr_val = {
                let c = data.val.iter().enumerate().filter(|(vi, (_, y))| {
                    let mut d = 0i32;
                    for (&w, &p) in bw.iter().zip(&tp.parents) {
                        d += (w as i32) * (all_val_sigs[*vi][p] as i32);
                    }
                    let idx = d - lut_min_dot;
                    let val = if idx < 0 { 0.0 }
                        else if (idx as usize) >= lut.len() { 0.0 }
                        else { lut[idx as usize] };
                    (if val >= 0.0 { 1u8 } else { 0 }) == *y
                }).count();
                c as f32 / data.val.len() as f32 * 100.0
            };

            let neuron = Neuron {
                id: net.neurons.len(),
                parents: tp.parents.clone(),
                tick,
                weights: bw.clone(),
                threshold: bt,
                alpha,
                train_acc: qr_train,
                val_acc: qr_val,
                c_float,
                rho_float,
                c_quant,
                rho_quant,
                lut_min_dot,
                lut: lut.clone(),
                // Transient int8 fields, stamped from bake_lut_i8 above.
                // alpha_i16 / alpha_scale are computed by quantize_alphas_i16
                // over the *whole* net after `net.add(neuron)` below, since the
                // alpha scale is network-shared.
                lut_i8: lut_i8.clone(),
                lut_scale,
                alpha_i16: 0,
                alpha_scale: 1.0,
            };
            net.add(neuron);
            // Re-derive the network-shared int16 alpha scale now that the new
            // neuron is in place. This writes alpha_i16 + alpha_scale on every
            // neuron (including the freshly added one).
            quantize_alphas_i16(&mut net);

            // Accuracy here uses the runtime dispatch (i8 path when cfg.use_i8).
            let new_val = net.accuracy(&data.val);

            if new_val < ens_val {
                if cfg.verbose_search {
                    println!("      [reject] idx={} ensemble val dropped {:.2}%→{:.2}% after adding neuron",
                        pick_i, ens_val, new_val);
                }
                net.sig_ticks.pop();
                net.neurons.pop();
                // The removal changes the absmax of alpha across the remaining
                // neurons, so re-quant alphas to keep them coherent. Otherwise a
                // later accept check would compare against a stale alpha_scale.
                quantize_alphas_i16(&mut net);
                continue;
            }

            // --bake-best: we've passed the val-drop gate with this candidate,
            // but instead of committing it we save it, pop the tentative add,
            // and continue so the next candidate gets a fair shot from the
            // same step-start net state. The winner (highest new_val, ties
            // broken by smaller ni) is re-added permanently after the loop.
            if cfg.bake_best {
                let saved = net.neurons.last().unwrap().clone();
                net.sig_ticks.pop();
                net.neurons.pop();
                quantize_alphas_i16(&mut net);
                let (replaces, note) = match &best_baked {
                    None => (true, "new best"),
                    Some((best_val, _, best_neuron, _, _)) => {
                        if new_val > *best_val {
                            (true, "new best (higher val)")
                        } else if (new_val - *best_val).abs() < 1e-6
                            && saved.parents.len() < best_neuron.parents.len()
                        {
                            (true, "new best (tie, smaller fan)")
                        } else {
                            (false, "not improving")
                        }
                    }
                };
                println!(
                    "      [bake-best] idx={} new_val={:.2}% ni={} — {}",
                    pick_i, new_val, saved.parents.len(), note
                );
                if replaces {
                    best_baked = Some((new_val, pick_i, saved, ft_loss, q_loss));
                }
                continue;
            }

            // ── Post-bake QAT verification ──────────────────────────
            // Compare i8 ensemble accuracy vs float ensemble accuracy on the
            // TRAIN set and assert the delta is within `cfg.i8_assert_tol_pp`.
            // If this triggers, the int8 quant path is broken for this neuron
            // (stronger than "noise" — we know from the analysis it should be
            // zero delta on every neuron seen so far). Fail fast so the user
            // is notified immediately and the run does not silently drift.
            //
            // The check happens here (after the accept gate) so that the exit
            // leaves the accepted float network in a debuggable state: the
            // state.tsv has NOT yet been persisted, so the user can re-run
            // with --no-i8 to isolate the issue.
            if cfg.use_i8 {
                let acc_f32 = net.accuracy_f32(&data.train);
                let acc_i8 = net.accuracy_i8(&data.train);
                let delta_pp = (acc_f32 - acc_i8).abs();
                if delta_pp > cfg.i8_assert_tol_pp {
                    eprintln!(
                        "\nERROR: int8 QAT assertion failed at neuron N{} — \
                         acc_f32={:.3}% acc_i8={:.3}% delta={:.3}pp > tol={:.3}pp",
                        net.neurons.len() - 1, acc_f32, acc_i8, delta_pp, cfg.i8_assert_tol_pp
                    );
                    eprintln!(
                        "       This should not happen: the per-neuron absmax int8 \
                         quant has been validated lossless on 48/48 c19 neurons.\n       \
                         Re-run with --no-i8 to compare against the float path, and \
                         inspect `lut_i8`, `lut_scale`, and the per-neuron LUT \
                         in the trace.json/final.json artifacts."
                    );
                    std::process::exit(3);
                }
            }

            // ACCEPTED — record trace event before persistence
            let accepted_neuron = net.neurons.last().unwrap();
            trace_events.push(TraceEvent {
                event: "neuron_added",
                tick: accepted_neuron.tick,
                id: accepted_neuron.id,
                parents: accepted_neuron.parents.clone(),
                weights: accepted_neuron.weights.clone(),
                threshold: accepted_neuron.threshold,
                alpha: accepted_neuron.alpha,
                train_acc: accepted_neuron.train_acc,
                val_acc: accepted_neuron.val_acc,
                c_float: accepted_neuron.c_float,
                rho_float: accepted_neuron.rho_float,
                c_quant: accepted_neuron.c_quant,
                rho_quant: accepted_neuron.rho_quant,
                lut_min_dot: accepted_neuron.lut_min_dot,
                lut: accepted_neuron.lut.clone(),
                finetune_loss: if ft_loss.is_finite() { ft_loss } else { 0.0 },
                quant_loss: if q_loss.is_finite() { q_loss } else { 0.0 },
                lut_i8: accepted_neuron.lut_i8.clone(),
                lut_scale: accepted_neuron.lut_scale,
                alpha_i16: accepted_neuron.alpha_i16,
                alpha_scale: accepted_neuron.alpha_scale,
            });

            let has_hidden = tp.parents.iter().any(|&p| p >= net.n_in);
            let pnames: Vec<String> = tp.parents.iter().map(|&p| sig_name(p, net.n_in)).collect();
            let wstr: String = bw.iter().map(|&v| match v { 1=>"+", -1=>"-", _=>"0" }).collect::<Vec<_>>().join("");
            // Read-back the stamped i8 fields from the accepted neuron for the log line.
            let i8_scale_disp = net.neurons.last().unwrap().lut_scale;
            let a16_disp = net.neurons.last().unwrap().alpha_i16;
            let a_scale_disp = net.neurons.last().unwrap().alpha_scale;

            println!("    >> N{}: [{}] thr={} tick={} parents=[{}] c={:.2} rho={:.1} lut_sz={} val={:.1}->{:.1}% hidden={} | i8_scale={:.4e} a_i16={} a_scale={:.4e} ({:.0}ms)",
                net.neurons.len()-1, wstr, bt, tick, pnames.join(","),
                c_quant, rho_quant, lut.len(),
                ens_val, new_val, has_hidden,
                i8_scale_disp, a16_disp, a_scale_disp,
                t_step.elapsed().as_millis());

            // AdaBoost reweight using bo_lut (not bo)
            let mut norm = 0.0f32;
            for ((pred, (_, y)), wt) in bo_lut.iter().zip(&data.train).zip(sw.iter_mut()) {
                let ys = if *y == 1 { 1.0 } else { -1.0 };
                let hs = if *pred == 1 { 1.0 } else { -1.0 };
                *wt *= (-alpha * ys * hs).exp();
                norm += *wt;
            }
            if norm > 0.0 { for w in &mut sw { *w /= norm; } }

            // Persist — order matters: state.tsv first (atomic rename, the
            // source of truth), then the per-neuron checkpoint JSON, then an
            // incremental trace.json flush so Brain Replay sees the full
            // history up to NOW even if the process is killed mid-run. A
            // trailing stdout flush makes sure the user actually sees the
            // "N{} added" log line on a piped stdout.
            //
            // All three IO calls are non-fatal: on failure we warn but keep
            // running. The previous atomic-renamed state.tsv is still the
            // source of truth, so "save failure on neuron N" means "neuron N
            // lives only in memory until the next successful save".
            if let Err(e) = save_state(&net, &state_path, &head) {
                eprintln!("WARN: save_state failed after N{}: {} (in-memory only)",
                    net.neurons.len() - 1, e);
            }
            let ckpt = format!("{}/checkpoints/n{:03}.json", cfg.out_dir, net.neurons.len());
            if let Err(e) = save_checkpoint(&net, &ckpt, task_name, step, &data) {
                eprintln!("WARN: save_checkpoint failed after N{}: {}",
                    net.neurons.len() - 1, e);
            }
            let cur_test = net.accuracy(&data.test);
            let cur_depth = net.neurons.iter().map(|n| n.tick).max().unwrap_or(0);
            if let Err(e) = write_trace_json(
                &cfg.out_dir, &cfg, net.n_in, &trace_events,
                new_val, cur_test, net.neurons.len(), cur_depth, stall,
            ) {
                eprintln!("WARN: trace.json flush failed after N{}: {}",
                    net.neurons.len() - 1, e);
            }
            let _ = std::io::stdout().flush();

            if new_val > best_val + 0.5 { best_val = new_val; stall = 0; }
            else { stall += 1; }

            accepted = true;
            break;
        }

        // --bake-best commit: after the tentative-bake loop above, the best
        // candidate (if any) is re-added permanently through a mirror of the
        // normal accept path (QAT + trace + log + AdaBoost reweight + persist).
        if cfg.bake_best && !accepted {
            if let Some((winner_val, winner_pick_i, winner_neuron, winner_ft_loss, winner_q_loss)) = best_baked {
                println!(
                    "    [bake-best] WINNER: idx={} val={:.2}% ({} parents)",
                    winner_pick_i, winner_val, winner_neuron.parents.len()
                );
                // Re-add the saved Neuron and re-derive the network-wide i16
                // alpha scale. The i8 fields on the saved Neuron will be
                // overwritten by quantize_alphas_i16 so they stay consistent.
                net.add(winner_neuron);
                quantize_alphas_i16(&mut net);
                let new_val = winner_val;

                // Post-bake QAT verification (same as the normal accept path).
                if cfg.use_i8 {
                    let acc_f32 = net.accuracy_f32(&data.train);
                    let acc_i8 = net.accuracy_i8(&data.train);
                    let delta_pp = (acc_f32 - acc_i8).abs();
                    if delta_pp > cfg.i8_assert_tol_pp {
                        eprintln!(
                            "\nERROR: int8 QAT assertion failed at bake-best N{} — \
                             acc_f32={:.3}% acc_i8={:.3}% delta={:.3}pp > tol={:.3}pp",
                            net.neurons.len() - 1, acc_f32, acc_i8, delta_pp, cfg.i8_assert_tol_pp
                        );
                        std::process::exit(3);
                    }
                }

                // Trace push, reading everything back from the re-added Neuron.
                let accepted_neuron = net.neurons.last().unwrap();
                trace_events.push(TraceEvent {
                    event: "neuron_added",
                    tick: accepted_neuron.tick,
                    id: accepted_neuron.id,
                    parents: accepted_neuron.parents.clone(),
                    weights: accepted_neuron.weights.clone(),
                    threshold: accepted_neuron.threshold,
                    alpha: accepted_neuron.alpha,
                    train_acc: accepted_neuron.train_acc,
                    val_acc: accepted_neuron.val_acc,
                    c_float: accepted_neuron.c_float,
                    rho_float: accepted_neuron.rho_float,
                    c_quant: accepted_neuron.c_quant,
                    rho_quant: accepted_neuron.rho_quant,
                    lut_min_dot: accepted_neuron.lut_min_dot,
                    lut: accepted_neuron.lut.clone(),
                    finetune_loss: if winner_ft_loss.is_finite() { winner_ft_loss } else { 0.0 },
                    quant_loss: if winner_q_loss.is_finite() { winner_q_loss } else { 0.0 },
                    lut_i8: accepted_neuron.lut_i8.clone(),
                    lut_scale: accepted_neuron.lut_scale,
                    alpha_i16: accepted_neuron.alpha_i16,
                    alpha_scale: accepted_neuron.alpha_scale,
                });

                let has_hidden = accepted_neuron.parents.iter().any(|&p| p >= net.n_in);
                let pnames: Vec<String> = accepted_neuron.parents.iter()
                    .map(|&p| sig_name(p, net.n_in))
                    .collect();
                let wstr: String = accepted_neuron.weights.iter()
                    .map(|&v| match v { 1 => "+", -1 => "-", _ => "0" })
                    .collect::<Vec<_>>()
                    .join("");
                let i8_scale_disp = accepted_neuron.lut_scale;
                let a16_disp = accepted_neuron.alpha_i16;
                let a_scale_disp = accepted_neuron.alpha_scale;
                let neuron_threshold = accepted_neuron.threshold;
                let neuron_tick = accepted_neuron.tick;
                let neuron_c_quant = accepted_neuron.c_quant;
                let neuron_rho_quant = accepted_neuron.rho_quant;
                let neuron_lut_len = accepted_neuron.lut.len();
                let winner_alpha = accepted_neuron.alpha;

                println!("    >> N{}: [{}] thr={} tick={} parents=[{}] c={:.2} rho={:.1} lut_sz={} val={:.1}->{:.1}% hidden={} | i8_scale={:.4e} a_i16={} a_scale={:.4e} ({:.0}ms) [bake-best idx={}]",
                    net.neurons.len() - 1, wstr, neuron_threshold, neuron_tick, pnames.join(","),
                    neuron_c_quant, neuron_rho_quant, neuron_lut_len,
                    ens_val, new_val, has_hidden,
                    i8_scale_disp, a16_disp, a_scale_disp,
                    t_step.elapsed().as_millis(), winner_pick_i);

                // AdaBoost sample-weight reweight: recompute bo_lut from the
                // re-added Neuron's weights + parents + LUT, then apply the
                // same exponential reweight as the normal accept path.
                let winner_parents = accepted_neuron.parents.clone();
                let winner_weights = accepted_neuron.weights.clone();
                let winner_lut = accepted_neuron.lut.clone();
                let winner_lut_min_dot = accepted_neuron.lut_min_dot;
                let bo_lut: Vec<u8> = (0..data.train.len()).map(|pi| {
                    let mut d = 0i32;
                    for (&w, &p) in winner_weights.iter().zip(&winner_parents) {
                        d += (w as i32) * (all_sigs[pi][p] as i32);
                    }
                    let idx = d - winner_lut_min_dot;
                    let val = if idx < 0 { 0.0 }
                        else if (idx as usize) >= winner_lut.len() { 0.0 }
                        else { winner_lut[idx as usize] };
                    if val >= 0.0 { 1u8 } else { 0 }
                }).collect();

                let mut norm = 0.0f32;
                for ((pred, (_, y)), wt) in bo_lut.iter().zip(&data.train).zip(sw.iter_mut()) {
                    let ys = if *y == 1 { 1.0 } else { -1.0 };
                    let hs = if *pred == 1 { 1.0 } else { -1.0 };
                    *wt *= (-winner_alpha * ys * hs).exp();
                    norm += *wt;
                }
                if norm > 0.0 { for w in sw.iter_mut() { *w /= norm; } }

                // Persist with the same crash-safe ordering as the normal path.
                if let Err(e) = save_state(&net, &state_path, &head) {
                    eprintln!("WARN: save_state failed after bake-best N{}: {} (in-memory only)",
                        net.neurons.len() - 1, e);
                }
                let ckpt = format!("{}/checkpoints/n{:03}.json", cfg.out_dir, net.neurons.len());
                if let Err(e) = save_checkpoint(&net, &ckpt, task_name, step, &data) {
                    eprintln!("WARN: save_checkpoint failed after bake-best N{}: {}",
                        net.neurons.len() - 1, e);
                }
                let cur_test = net.accuracy(&data.test);
                let cur_depth = net.neurons.iter().map(|n| n.tick).max().unwrap_or(0);
                if let Err(e) = write_trace_json(
                    &cfg.out_dir, &cfg, net.n_in, &trace_events,
                    new_val, cur_test, net.neurons.len(), cur_depth, stall,
                ) {
                    eprintln!("WARN: trace.json flush failed after bake-best N{}: {}",
                        net.neurons.len() - 1, e);
                }
                let _ = std::io::stdout().flush();

                if new_val > best_val + 0.5 { best_val = new_val; stall = 0; }
                else { stall += 1; }

                accepted = true;
            } else {
                println!("    [bake-best] no candidate passed the dedup+werr+val-drop gates");
            }
        }

        if !accepted {
            println!("    X No improvement ({:.0}ms)", t_step.elapsed().as_millis());
            stall += 1;
        }

        if stall >= cfg.stall_limit {
            println!("\n  Stalled {} steps on task {}.", cfg.stall_limit, task_name);
            break;
        }
    }

        // End of this task's step loop — capture state for either the next
        // task or the final report, and persist one more time to flush the
        // final state.tsv / trace for this task before moving on. This
        // matters most when a task exits via stall (mid-session) — we want
        // the last trace.json on disk to reflect that task's full run.
        let task_done_val = net.accuracy(&data.val);
        let task_done_test = net.accuracy(&data.test);
        let task_done_depth = net.neurons.iter().map(|n| n.tick).max().unwrap_or(0);
        println!(
            "\n  TASK END [{}]: {} neurons total, val={:.1}%{}",
            task_name,
            net.neurons.len(),
            task_done_val,
            if task_done_val >= 99.0 { " — 100% hit!" } else { " — moving on" },
        );
        if let Err(e) = save_state(&net, &state_path, &head) {
            eprintln!("WARN: save_state failed at task end [{}]: {}", task_name, e);
        }
        if let Err(e) = write_trace_json(
            &cfg.out_dir, &cfg, net.n_in, &trace_events,
            task_done_val, task_done_test, net.neurons.len(), task_done_depth, stall,
        ) {
            eprintln!("WARN: trace.json flush failed at task end [{}]: {}", task_name, e);
        }
        let _ = std::io::stdout().flush();
        final_stall = stall;
        final_data_opt = Some(data);
        // Fall through to next task in 'task_loop (implicit continue).
        let _ = task_idx; // silence unused if task-list has a single task
        if false { break 'task_loop; }
    }

    // Final report uses the LAST task's data view so the numbers match the
    // last banner the user saw. If no tasks ran (empty list), the sanity
    // check in main() at entry already exited.
    let data = final_data_opt.expect("task loop must run at least once");
    let stall = final_stall;

    let ft = net.accuracy(&data.train);
    let fv = net.accuracy(&data.val);
    let fte = net.accuracy(&data.test);
    let mt = net.neurons.iter().map(|n| n.tick).max().unwrap_or(0);
    let hid = net.neurons.iter().any(|n| n.parents.iter().any(|&p| p >= net.n_in));

    println!("\n  FINAL: {} neurons, depth={}, hidden={}", net.neurons.len(), mt, hid);
    println!("  train={:.1}% val={:.1}% test={:.1}% (last task: {})", ft, fv, fte, head.task);
    println!("  Time: {:.1}s", t0.elapsed().as_secs_f64());

    let ckpt = format!("{}/final.json", cfg.out_dir);
    if let Err(e) = save_checkpoint(&net, &ckpt, &head.task, net.neurons.len(), &data) {
        eprintln!("WARN: final save_checkpoint failed: {}", e);
    }
    if let Err(e) = save_state(&net, &state_path, &head) {
        eprintln!("WARN: final save_state failed: {}", e);
    }

    if let Err(e) = write_trace_json(
        &cfg.out_dir,
        &cfg,
        net.n_in,
        &trace_events,
        fv,
        fte,
        net.neurons.len(),
        mt,
        stall,
    ) {
        eprintln!("ERROR: failed to write trace.json: {}", e);
        std::process::exit(1);
    }
}
