//! Neuron Canonical Builder — single growing network, proper pipeline
//!
//! Per-neuron (serial multi-proposal search, NOT parallel):
//!   1. 20 random (parent_set + float init) proposals, scored on residual objective
//!   2. Top-5 → backprop restarts → sign consensus (local basin map)
//!   3. Guided ternary + blind ternary fallback on same parent set
//!   4. Accept gate: only if ensemble val improves, not duplicate, quantized holds
//!   5. Freeze + checkpoint with ensemble metrics
//!
//! Output: instnct-core/results/neuron_canonical/<seed>/
//!
//! Run: cargo run --example neuron_canonical --release
//! Run with seed: cargo run --example neuron_canonical --release -- --seed 123
//! Quick: cargo run --example neuron_canonical --release -- --seed 42 --max-neurons 5 --tasks digit_parity

use std::io::Write as IoWrite;
use std::sync::Arc;
use std::time::Instant;

// ══════════════════════════════════════════════════════
// CLI CONFIG
// ══════════════════════════════════════════════════════
struct Config {
    seed: u64,
    max_neurons: usize,
    max_fan: usize,
    n_proposals: usize,
    stall_limit: usize,
    wall_clock_min: u64,
    tasks: Vec<String>,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = Config {
        seed: 42,
        max_neurons: 20,
        max_fan: 10,
        n_proposals: 20,
        stall_limit: 5,
        wall_clock_min: 240,
        tasks: vec![
            "digit_parity".into(), "is_symmetric".into(), "digit_2_vs_3".into(),
        ],
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => { i += 1; cfg.seed = args[i].parse().unwrap_or(42); }
            "--max-neurons" => { i += 1; cfg.max_neurons = args[i].parse().unwrap_or(20); }
            "--max-fan" => { i += 1; cfg.max_fan = args[i].parse().unwrap_or(10); }
            "--proposals" => { i += 1; cfg.n_proposals = args[i].parse().unwrap_or(20); }
            "--stall" => { i += 1; cfg.stall_limit = args[i].parse().unwrap_or(5); }
            "--wall-clock-min" => { i += 1; cfg.wall_clock_min = args[i].parse().unwrap_or(240); }
            "--tasks" => { i += 1; cfg.tasks = args[i].split(',').map(|s| s.trim().to_string()).collect(); }
            _ => { if let Ok(s) = args[i].parse::<u64>() { cfg.seed = s; } }
        }
        i += 1;
    }
    cfg
}

// ══════════════════════════════════════════════════════
// REJECT COUNTERS
// ══════════════════════════════════════════════════════
#[derive(Default)]
struct RejectCounters {
    accept_count: usize,
    reject_duplicate: usize,
    reject_weighted_error: usize,
    reject_no_val_gain: usize,
}

struct TaskResult {
    name: String,
    n_neurons: usize,
    max_tick: u32,
    hidden_parent_used: bool,
    best_train: f32,
    best_val: f32,
    best_test: f32,
    wall_s: f64,
    counters: RejectCounters,
}

// ══════════════════════════════════════════════════════
// PRNG
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
// FONT + DATA
// ══════════════════════════════════════════════════════
const FONT: [[u8; 9]; 10] = [
    [1,1,1, 1,0,1, 1,1,1],
    [0,1,0, 0,1,0, 0,1,0],
    [1,1,0, 0,1,0, 0,1,1],
    [1,1,0, 0,1,0, 1,1,0],
    [1,0,1, 1,1,1, 0,0,1],
    [0,1,1, 0,1,0, 1,1,0],
    [1,0,0, 1,1,0, 1,1,0],
    [1,1,1, 0,0,1, 0,0,1],
    [1,1,1, 1,1,1, 1,1,1],
    [1,1,1, 1,1,1, 0,1,1],
];

struct Split { train: Vec<(Vec<u8>, u8)>, val: Vec<(Vec<u8>, u8)>, test: Vec<(Vec<u8>, u8)> }

fn gen_data(label_fn: &dyn Fn(usize, &[u8]) -> Option<u8>, noise: f32, n_per: usize, seed: u64) -> Split {
    let mut rng = Rng::new(seed);
    let (mut tr, mut va, mut te) = (Vec::new(), Vec::new(), Vec::new());
    for d in 0..10 {
        for i in 0..n_per {
            let mut px = FONT[d].to_vec();
            for p in px.iter_mut() { if rng.bool_p(noise) { *p = 1 - *p; } }
            if let Some(label) = label_fn(d, &px) {
                match i % 5 { 0 => va.push((px, label)), 1 => te.push((px, label)), _ => tr.push((px, label)) }
            }
        }
    }
    Split { train: tr, val: va, test: te }
}

// ══════════════════════════════════════════════════════
// FROZEN NEURON + NETWORK
// ══════════════════════════════════════════════════════
#[derive(Clone)]
struct Neuron {
    id: usize,
    parents: Vec<usize>,
    tick: u32,
    weights: Vec<i8>,
    bias: i8,
    threshold: i32,
    alpha: f32,
    train_acc: f32,
    val_acc: f32,
}

impl Neuron {
    fn eval(&self, sigs: &[u8]) -> u8 {
        let mut d = self.bias as i32;
        for (&w, &p) in self.weights.iter().zip(&self.parents) { d += (w as i32) * (sigs[p] as i32); }
        if d >= self.threshold { 1 } else { 0 }
    }
}

#[derive(Clone)]
struct Net {
    neurons: Vec<Neuron>,
    n_in: usize,
    sig_ticks: Vec<u32>,
}

impl Net {
    fn new(n: usize) -> Self { Net { neurons: Vec::new(), n_in: n, sig_ticks: vec![0; n] } }
    fn n_sig(&self) -> usize { self.n_in + self.neurons.len() }

    fn eval_all(&self, inp: &[u8]) -> Vec<u8> {
        let mut s: Vec<u8> = inp.to_vec();
        for n in &self.neurons { s.push(n.eval(&s)); }
        s
    }

    fn predict(&self, inp: &[u8]) -> u8 {
        if self.neurons.is_empty() { return 0; }
        let s = self.eval_all(inp);
        let score: f32 = self.neurons.iter().enumerate()
            .map(|(i, n)| n.alpha * if s[self.n_in + i] == 1 { 1.0 } else { -1.0 }).sum();
        if score >= 0.0 { 1 } else { 0 }
    }

    fn accuracy(&self, data: &[(Vec<u8>, u8)]) -> f32 {
        if self.neurons.is_empty() { return 50.0; }
        let c = data.iter().filter(|(x, y)| self.predict(x) == *y).count();
        c as f32 / data.len() as f32 * 100.0
    }

    fn add(&mut self, n: Neuron) { self.sig_ticks.push(n.tick); self.neurons.push(n); }
}

// ══════════════════════════════════════════════════════
// STEP A: PARALLEL PROPOSALS — (parent_set + float init)
// ══════════════════════════════════════════════════════

#[allow(dead_code)]
struct Proposal {
    parents: Vec<usize>,
    float_w: Vec<f32>,
    float_b: f32,
    _score: f32,
}

#[allow(dead_code)]
fn generate_proposals(
    data: &[(Vec<u8>, u8)], net: &Net, sw: &[f32],
    n_proposals: usize, max_fan: usize,
) -> Vec<Proposal> {
    let all_sigs: Vec<Vec<u8>> = data.iter().map(|(x, _)| net.eval_all(x)).collect();
    let n_sig = net.n_sig();
    let mut proposals = Vec::new();

    for seed in 0..n_proposals {
        let mut rng = Rng::new(seed as u64 * 7919 + 31);

        // Random parent set: pick 2..max_fan parents from available signals
        let n_parents = 2 + rng.pick(max_fan.min(n_sig) - 1);
        let n_parents = n_parents.min(n_sig);
        let mut parents: Vec<usize> = Vec::new();
        for _ in 0..n_parents * 3 {
            if parents.len() >= n_parents { break; }
            let p = rng.pick(n_sig);
            if !parents.contains(&p) { parents.push(p); }
        }
        if parents.is_empty() { continue; }

        // Random float init
        let ni = parents.len();
        let w: Vec<f32> = (0..ni).map(|_| rng.range(-2.0, 2.0)).collect();
        let b = rng.range(-1.0, 1.0);

        // Quick eval (weighted accuracy with float sigmoid)
        let score: f32 = data.iter().enumerate().map(|(pi, (_, y))| {
            let z: f32 = b + (0..ni).map(|i| w[i] * all_sigs[pi][parents[i]] as f32).sum::<f32>();
            let pred = if sigmoid(z) > 0.5 { 1u8 } else { 0 };
            if pred == *y { sw[pi] } else { 0.0 }
        }).sum();

        proposals.push(Proposal { parents, float_w: w, float_b: b, _score: score });
    }

    proposals.sort_by(|a, b| b._score.partial_cmp(&a._score).unwrap());
    proposals
}

// ══════════════════════════════════════════════════════
// STEP B: BACKPROP RESTARTS on top-K proposals
// ══════════════════════════════════════════════════════

#[allow(dead_code)]
struct TrainedProposal {
    parents: Vec<usize>,
    val_acc: f32,
    sign_consensus: Vec<i8>,
}

#[allow(dead_code)]
fn backprop_proposal(
    data: &[(Vec<u8>, u8)], val: &[(Vec<u8>, u8)],
    net: &Net, parents: &[usize], init_w: &[f32], init_b: f32,
    sw: &[f32], epochs: usize,
) -> TrainedProposal {
    let all_sigs: Vec<Vec<u8>> = data.iter().map(|(x, _)| net.eval_all(x)).collect();
    let all_val_sigs: Vec<Vec<u8>> = val.iter().map(|(x, _)| net.eval_all(x)).collect();
    let ni = parents.len();

    // Multi-restart: original + 4 perturbed
    let mut best_w = init_w.to_vec();
    let mut best_b = init_b;
    let mut best_score = 0.0f32;
    let mut all_converged: Vec<(Vec<f32>, f32)> = Vec::new();

    for restart in 0..5 {
        let mut rng = Rng::new(restart * 1000 + 77);
        let mut w: Vec<f32> = if restart == 0 {
            init_w.to_vec()
        } else {
            init_w.iter().map(|&v| v + rng.range(-0.5, 0.5)).collect()
        };
        let mut b = if restart == 0 { init_b } else { init_b + rng.range(-0.3, 0.3) };

        for _ in 0..epochs {
            for (pi, (_, y)) in data.iter().enumerate() {
                let z: f32 = b + (0..ni).map(|i| w[i] * all_sigs[pi][parents[i]] as f32).sum::<f32>();
                let a = sigmoid(z);
                let err = a - *y as f32;
                let g = err * a * (1.0 - a) * sw[pi] * data.len() as f32;
                for i in 0..ni { w[i] -= 0.5 * g * all_sigs[pi][parents[i]] as f32; }
                b -= 0.5 * g;
            }
        }

        let score: f32 = data.iter().enumerate().map(|(pi, (_, y))| {
            let z: f32 = b + (0..ni).map(|i| w[i] * all_sigs[pi][parents[i]] as f32).sum::<f32>();
            if (if sigmoid(z) > 0.5 { 1u8 } else { 0 }) == *y { sw[pi] } else { 0.0 }
        }).sum();

        all_converged.push((w.clone(), b));
        if score > best_score { best_score = score; best_w = w; best_b = b; }
    }

    // Sign consensus across restarts
    let sign_consensus: Vec<i8> = (0..ni).map(|i| {
        let pos = all_converged.iter().filter(|(w, _)| w[i] > 0.3).count();
        let neg = all_converged.iter().filter(|(w, _)| w[i] < -0.3).count();
        let total = all_converged.len();
        if pos * 10 / total >= 7 { 1 }
        else if neg * 10 / total >= 7 { -1 }
        else { 2 } // no consensus
    }).collect();

    // Val accuracy
    let val_acc = {
        let c = val.iter().enumerate().filter(|(pi, (_, y))| {
            let z: f32 = best_b + (0..ni).map(|i| best_w[i] * all_val_sigs[*pi][parents[i]] as f32).sum::<f32>();
            (if sigmoid(z) > 0.5 { 1u8 } else { 0 }) == *y
        }).count();
        c as f32 / val.len() as f32 * 100.0
    };

    TrainedProposal {
        parents: parents.to_vec(), val_acc, sign_consensus,
    }
}

// ══════════════════════════════════════════════════════
// STEP C: TERNARY QUANTIZATION (guided + blind)
// ══════════════════════════════════════════════════════

struct QuantResult {
    weights: Vec<i8>, bias: i8, threshold: i32,
    val_acc: f32, outputs: Vec<u8>,
}

struct CandidateResult {
    parents: Vec<usize>,
    weights: Vec<i8>,
    bias: i8,
    threshold: i32,
    val_acc: f32,
    outputs: Vec<u8>,
    fan_in: usize,
    sparsity: usize,
    dot_min: i32,
    dot_max: i32,
    distinct_bins: usize,
    max_match: f32,
}

#[allow(dead_code)]
fn ternary_on_parents(
    data: &[(Vec<u8>, u8)], val: &[(Vec<u8>, u8)],
    net: &Net, parents: &[usize], sw: &[f32],
    consensus: Option<&[i8]>,
) -> QuantResult {
    let all_sigs: Vec<Vec<u8>> = data.iter().map(|(x, _)| net.eval_all(x)).collect();
    let all_val: Vec<Vec<u8>> = val.iter().map(|(x, _)| net.eval_all(x)).collect();
    let ni = parents.len();
    let np = data.len();

    // Determine search space
    let (_combos, locked, free_pos) = if let Some(cons) = consensus {
        let mut locked: Vec<Option<i8>> = vec![None; ni];
        let mut free = Vec::new();
        for i in 0..ni {
            if cons[i] == 1 { locked[i] = Some(1); }
            else if cons[i] == -1 { locked[i] = Some(-1); }
            else { free.push(i); }
        }
        let n_free = free.len() + 1; // +1 for bias
        (3u64.pow(n_free as u32), locked, free)
    } else {
        let free: Vec<usize> = (0..ni).collect();
        (3u64.pow((ni + 1) as u32), vec![None; ni], free)
    };

    let _method = if consensus.is_some() { "guided" } else { "blind" };

    let mut bw = vec![0i8; ni]; let mut bb: i8 = 0; let mut bt: i32 = 0;
    let mut bs = -1.0f32; let mut bo = vec![0u8; np];

    if consensus.is_some() {
        // Guided: iterate free positions + bias
        let nf = free_pos.len();
        let total_free = 3u64.pow((nf + 1) as u32);
        for combo in 0..total_free {
            let mut w = vec![0i8; ni];
            for i in 0..ni { w[i] = locked[i].unwrap_or(0); }
            let mut r = combo;
            for &fp in &free_pos { w[fp] = (r % 3) as i8 - 1; r /= 3; }
            let b = (r % 3) as i8 - 1;
            eval_ternary(&all_sigs, parents, &w, b, data, sw, np, &mut bw, &mut bb, &mut bt, &mut bs, &mut bo);
        }
    }

    // Always also do blind (or if guided-only)
    let total_blind = 3u64.pow((ni + 1) as u32);
    if total_blind <= 500_000 { // safety: don't run blind if too large
        for combo in 0..total_blind {
            let mut w = vec![0i8; ni]; let mut r = combo;
            for wi in w.iter_mut() { *wi = (r % 3) as i8 - 1; r /= 3; }
            let b = (r % 3) as i8 - 1;
            eval_ternary(&all_sigs, parents, &w, b, data, sw, np, &mut bw, &mut bb, &mut bt, &mut bs, &mut bo);
        }
    }

    // Val accuracy of best
    let val_acc = {
        let c = val.iter().enumerate().filter(|(pi, (_, y))| {
            let mut d = bb as i32;
            for (&w, &p) in bw.iter().zip(parents) { d += (w as i32) * (all_val[*pi][p] as i32); }
            (if d >= bt { 1u8 } else { 0 }) == *y
        }).count();
        c as f32 / val.len() as f32 * 100.0
    };

    QuantResult { weights: bw, bias: bb, threshold: bt, val_acc, outputs: bo }
}

#[allow(dead_code)]
fn eval_ternary(
    all_sigs: &[Vec<u8>], parents: &[usize], w: &[i8], b: i8,
    data: &[(Vec<u8>, u8)], sw: &[f32], np: usize,
    bw: &mut Vec<i8>, bb: &mut i8, bt: &mut i32, bs: &mut f32, bo: &mut Vec<u8>,
) {
    let dots: Vec<i32> = (0..np).map(|pi| {
        let mut d = b as i32;
        for (wi, &pidx) in w.iter().zip(parents) { d += (*wi as i32) * (all_sigs[pi][pidx] as i32); }
        d
    }).collect();
    let min_d = dots.iter().copied().min().unwrap_or(0);
    let max_d = dots.iter().copied().max().unwrap_or(0);
    for thresh in (min_d - 1)..=(max_d + 1) {
        let outs: Vec<u8> = dots.iter().map(|&d| if d >= thresh { 1 } else { 0 }).collect();
        let score: f32 = outs.iter().zip(data).zip(sw)
            .map(|((&pred, (_, y)), &wt)| if pred == *y { wt } else { 0.0 }).sum();
        if score > *bs {
            *bs = score; *bw = w.to_vec(); *bb = b; *bt = thresh; *bo = outs;
        }
    }
}

// ══════════════════════════════════════════════════════
// STEP D: ACCEPT/REJECT GATE
// ══════════════════════════════════════════════════════

fn output_match_rate(a: &[u8], all_sigs: &[Vec<u8>], sig: usize) -> f32 {
    let same = a.iter().enumerate().filter(|(i, v)| all_sigs[*i][sig] == **v).count();
    same as f32 / a.len() as f32
}

// ══════════════════════════════════════════════════════
// CHECKPOINT
// ══════════════════════════════════════════════════════
fn save_checkpoint(net: &Net, path: &str, task: &str, step: usize, data: &Split) {
    if let Some(p) = std::path::Path::new(path).parent() { std::fs::create_dir_all(p).ok(); }
    let ens_train = net.accuracy(&data.train);
    let ens_val = net.accuracy(&data.val);
    let ens_test = net.accuracy(&data.test);
    let mut f = std::fs::File::create(path).unwrap();
    writeln!(f, "{{").unwrap();
    writeln!(f, "\"task\":\"{}\",\"step\":{},\"n_inputs\":{},", task, step, net.n_in).unwrap();
    writeln!(f, "\"ensemble_train\":{:.2},\"ensemble_val\":{:.2},\"ensemble_test\":{:.2},", ens_train, ens_val, ens_test).unwrap();
    writeln!(f, "\"neurons\":[").unwrap();
    for (i, n) in net.neurons.iter().enumerate() {
        let wj: Vec<String> = n.weights.iter().map(|v| v.to_string()).collect();
        let pj: Vec<String> = n.parents.iter().map(|v| v.to_string()).collect();
        writeln!(f, "{{\"id\":{},\"parents\":[{}],\"tick\":{},\"weights\":[{}],\"bias\":{},\"threshold\":{},\"alpha\":{:.6},\"train_acc\":{:.2},\"val_acc\":{:.2}}}{}",
            n.id, pj.join(","), n.tick, wj.join(","), n.bias, n.threshold, n.alpha, n.train_acc, n.val_acc,
            if i < net.neurons.len()-1 { "," } else { "" }).unwrap();
    }
    writeln!(f, "]}}").unwrap();
}

// ══════════════════════════════════════════════════════
// MAIN BUILD LOOP
// ══════════════════════════════════════════════════════
fn run_task(
    task_name: &str,
    label_fn: &dyn Fn(usize, &[u8]) -> Option<u8>,
    noise: f32, n_per: usize,
    cfg: &Config,
    out_dir: &str,
) -> TaskResult {
    let data = gen_data(label_fn, noise, n_per, cfg.seed);
    let mut net = Net::new(9);
    let mut sw = vec![1.0 / data.train.len() as f32; data.train.len()];
    let mut stall = 0;
    let mut best_val = 50.0f32;
    let t_task = Instant::now();
    let mut counters = RejectCounters::default();

    println!("\n== {} ==", task_name);
    println!("  Data: {} train / {} val / {} test", data.train.len(), data.val.len(), data.test.len());
    println!("  Config: max_neurons={} max_fan={} proposals={} stall={}",
        cfg.max_neurons, cfg.max_fan, cfg.n_proposals, cfg.stall_limit);

    for step in 0..cfg.max_neurons {
        let t0 = Instant::now();
        let ensemble_val = net.accuracy(&data.val);
        let ensemble_test = net.accuracy(&data.test);
        println!("  Step {:2} | {} neurons | val={:.1}% test={:.1}%",
            step, net.neurons.len(), ensemble_val, ensemble_test);

        if ensemble_val >= 99.0 { println!("  >> Target reached!"); break; }

        // Fan-in from config; lexicographic ranking prefers smaller fan-in on ties
        let step_max_fan = cfg.max_fan;

        // ALL 20 PROPOSALS IN PARALLEL: proposal + backprop + ternary per thread
        let t_parallel = Instant::now();
        let train_arc: Arc<Vec<(Vec<u8>, u8)>> = Arc::new(data.train.clone());
        let val_arc: Arc<Vec<(Vec<u8>, u8)>> = Arc::new(data.val.clone());
        let sw_arc: Arc<Vec<f32>> = Arc::new(sw.clone());
        let net_arc: Arc<Net> = Arc::new(net.clone());
        let n_proposals = cfg.n_proposals;

        let current_step = step;
        let handles: Vec<_> = (0..n_proposals).map(|seed| {
            let train = Arc::clone(&train_arc);
            let val = Arc::clone(&val_arc);
            let sw = Arc::clone(&sw_arc);
            let net = Arc::clone(&net_arc);
            let max_fan = step_max_fan;
            let step = current_step;

            std::thread::spawn(move || -> Option<CandidateResult> {
                let n_sig = net.n_sig();
                let all_sigs: Vec<Vec<u8>> = train.iter().map(|(x, _)| net.eval_all(x)).collect();

                // Generate proposal — step mixed in so proposals vary between steps
                let mut rng = Rng::new(seed as u64 * 7919 + 31 + step as u64 * 104729);
                let n_parents = 2 + rng.pick(max_fan.min(n_sig) - 1);
                let n_parents = n_parents.min(n_sig);
                let mut parents: Vec<usize> = Vec::new();
                for _ in 0..n_parents * 3 {
                    if parents.len() >= n_parents { break; }
                    let p = rng.pick(n_sig);
                    if !parents.contains(&p) { parents.push(p); }
                }
                if parents.is_empty() { return None; }

                let ni = parents.len();
                let init_w: Vec<f32> = (0..ni).map(|_| rng.range(-2.0, 2.0)).collect();
                let init_b = rng.range(-1.0, 1.0);

                // Backprop (5 restarts, 2000 epochs)
                let all_val_sigs: Vec<Vec<u8>> = val.iter().map(|(x, _)| net.eval_all(x)).collect();
                let mut best_w = init_w.clone();
                let mut best_b = init_b;
                let mut best_score = 0.0f32;
                let mut all_converged: Vec<(Vec<f32>, f32)> = Vec::new();

                for restart in 0..5u64 {
                    let mut rng2 = Rng::new(restart * 1000 + 77);
                    let mut w: Vec<f32> = if restart == 0 { init_w.clone() }
                        else { init_w.iter().map(|&v| v + rng2.range(-0.5, 0.5)).collect() };
                    let mut b = if restart == 0 { init_b } else { init_b + rng2.range(-0.3, 0.3) };

                    for _ in 0..2000 {
                        for (pi, (_, y)) in train.iter().enumerate() {
                            let z: f32 = b + (0..ni).map(|i| w[i] * all_sigs[pi][parents[i]] as f32).sum::<f32>();
                            let a = sigmoid(z);
                            let g = (a - *y as f32) * a * (1.0 - a) * sw[pi] * train.len() as f32;
                            for i in 0..ni { w[i] -= 0.5 * g * all_sigs[pi][parents[i]] as f32; }
                            b -= 0.5 * g;
                        }
                    }

                    let score: f32 = train.iter().enumerate().map(|(pi, (_, y))| {
                        let z: f32 = b + (0..ni).map(|i| w[i] * all_sigs[pi][parents[i]] as f32).sum::<f32>();
                        if (if sigmoid(z) > 0.5 { 1u8 } else { 0 }) == *y { sw[pi] } else { 0.0 }
                    }).sum();

                    all_converged.push((w.clone(), b));
                    if score > best_score { best_score = score; best_w = w; best_b = b; }
                }

                // Sign consensus
                let sign_consensus: Vec<i8> = (0..ni).map(|i| {
                    let pos = all_converged.iter().filter(|(w, _)| w[i] > 0.3).count();
                    let neg = all_converged.iter().filter(|(w, _)| w[i] < -0.3).count();
                    let total = all_converged.len();
                    if pos * 10 / total >= 7 { 1 }
                    else if neg * 10 / total >= 7 { -1 }
                    else { 2 }
                }).collect();

                // Ternary search (guided + blind)
                let np = train.len();
                let (locked, free_pos) = {
                    let mut locked: Vec<Option<i8>> = vec![None; ni];
                    let mut free = Vec::new();
                    for i in 0..ni {
                        if sign_consensus[i] == 1 { locked[i] = Some(1); }
                        else if sign_consensus[i] == -1 { locked[i] = Some(-1); }
                        else { free.push(i); }
                    }
                    (locked, free)
                };

                let mut bw = vec![0i8; ni]; let mut bb: i8 = 0; let mut bt: i32 = 0;
                let mut bs = -1.0f32; let mut bo = vec![0u8; np];

                // Guided ternary
                let nf = free_pos.len();
                let total_free = 3u64.pow((nf + 1) as u32);
                for combo in 0..total_free {
                    let mut w = vec![0i8; ni];
                    for i in 0..ni { w[i] = locked[i].unwrap_or(0); }
                    let mut r = combo;
                    for &fp in &free_pos { w[fp] = (r % 3) as i8 - 1; r /= 3; }
                    let b = (r % 3) as i8 - 1;
                    let dots: Vec<i32> = (0..np).map(|pi| {
                        let mut d = b as i32;
                        for (j, &pidx) in parents.iter().enumerate() { d += (w[j] as i32) * (all_sigs[pi][pidx] as i32); }
                        d
                    }).collect();
                    let mn = dots.iter().copied().min().unwrap_or(0);
                    let mx = dots.iter().copied().max().unwrap_or(0);
                    for thresh in (mn-1)..=(mx+1) {
                        let outs: Vec<u8> = dots.iter().map(|&d| if d >= thresh { 1 } else { 0 }).collect();
                        let sc: f32 = outs.iter().zip(train.iter()).zip(sw.iter())
                            .map(|((&pred, (_, y)), &wt)| if pred == *y { wt } else { 0.0 }).sum();
                        if sc > bs { bs=sc; bw=w.clone(); bb=b; bt=thresh; bo=outs; }
                    }
                }

                // Blind ternary
                let total_blind = 3u64.pow((ni + 1) as u32);
                if total_blind <= 500_000 {
                    for combo in 0..total_blind {
                        let mut w = vec![0i8; ni]; let mut r = combo;
                        for wi in w.iter_mut() { *wi = (r % 3) as i8 - 1; r /= 3; }
                        let b = (r % 3) as i8 - 1;
                        let dots: Vec<i32> = (0..np).map(|pi| {
                            let mut d = b as i32;
                            for (j, &pidx) in parents.iter().enumerate() { d += (w[j] as i32) * (all_sigs[pi][pidx] as i32); }
                            d
                        }).collect();
                        let mn = dots.iter().copied().min().unwrap_or(0);
                        let mx = dots.iter().copied().max().unwrap_or(0);
                        for thresh in (mn-1)..=(mx+1) {
                            let outs: Vec<u8> = dots.iter().map(|&d| if d >= thresh { 1 } else { 0 }).collect();
                            let sc: f32 = outs.iter().zip(train.iter()).zip(sw.iter())
                                .map(|((&pred, (_, y)), &wt)| if pred == *y { wt } else { 0.0 }).sum();
                            if sc > bs { bs=sc; bw=w.clone(); bb=b; bt=thresh; bo=outs; }
                        }
                    }
                }

                // Quantized val accuracy
                let qr_val_acc = {
                    let c = val.iter().enumerate().filter(|(pi, (_, y))| {
                        let mut d = bb as i32;
                        for (&w, &pidx) in bw.iter().zip(&parents) { d += (w as i32) * (all_val_sigs[*pi][pidx] as i32); }
                        (if d >= bt { 1u8 } else { 0 }) == *y
                    }).count();
                    c as f32 / val.len() as f32 * 100.0
                };

                // Compute diagnostics for winner selection
                let sparsity = bw.iter().filter(|&&w| w == 0).count();
                let mut dots_diag: Vec<i32> = (0..np).map(|pi| {
                    let mut d = bb as i32;
                    for (j, &pidx) in parents.iter().enumerate() { d += (bw[j] as i32) * (all_sigs[pi][pidx] as i32); }
                    d
                }).collect();
                let dot_min = dots_diag.iter().copied().min().unwrap_or(0);
                let dot_max = dots_diag.iter().copied().max().unwrap_or(0);
                dots_diag.sort(); dots_diag.dedup();
                let distinct_bins = dots_diag.len();

                // Max output match rate with existing neurons
                let max_match = (net.n_in..n_sig).map(|e| {
                    bo.iter().enumerate().filter(|(i, v)| all_sigs[*i][e] == **v).count() as f32 / bo.len() as f32
                }).fold(0.0f32, f32::max);

                Some(CandidateResult {
                    parents, weights: bw, bias: bb, threshold: bt,
                    val_acc: qr_val_acc, outputs: bo,
                    fan_in: ni, sparsity, dot_min, dot_max, distinct_bins, max_match,
                })
            })
        }).collect();

        // Collect results
        let mut candidates: Vec<CandidateResult> = handles.into_iter()
            .filter_map(|h| h.join().ok().flatten())
            .collect();

        // Compute delta_ensemble_val for each candidate (sequential, needs net)
        let mut scored: Vec<(usize, f32)> = Vec::new(); // (index, delta_val)
        for (ci, cand) in candidates.iter().enumerate() {
            if cand.max_match >= 0.999 { scored.push((ci, -999.0)); continue; }
            let werr: f32 = cand.outputs.iter().zip(&data.train).zip(&sw)
                .map(|((&pred, (_, y)), &w)| if pred == *y { 0.0 } else { w }).sum();
            if werr >= 0.499 { scored.push((ci, -999.0)); continue; }
            let alpha = 0.5 * ((1.0 - werr).max(1e-6) / werr.max(1e-6)).ln();
            let tick = cand.parents.iter().map(|&p| net.sig_ticks[p]).max().unwrap_or(0) + 1;
            let neuron = Neuron {
                id: net.neurons.len(), parents: cand.parents.clone(), tick,
                weights: cand.weights.clone(), bias: cand.bias, threshold: cand.threshold,
                alpha, train_acc: cand.val_acc, val_acc: cand.val_acc,
            };
            net.add(neuron);
            let delta = net.accuracy(&data.val) - ensemble_val;
            net.sig_ticks.pop();
            net.neurons.pop();
            scored.push((ci, delta));
        }

        // Lexicographic sort by: delta_ensemble_val desc → fan_in asc → max_match asc → bins asc
        scored.sort_by(|a, b| {
            let da = a.1; let db = b.1;
            let d_cmp = db.partial_cmp(&da).unwrap();
            if d_cmp != std::cmp::Ordering::Equal { return d_cmp; }
            let ca = &candidates[a.0]; let cb = &candidates[b.0];
            let fan_cmp = ca.fan_in.cmp(&cb.fan_in);
            if fan_cmp != std::cmp::Ordering::Equal { return fan_cmp; }
            let match_cmp = ca.max_match.partial_cmp(&cb.max_match).unwrap();
            if match_cmp != std::cmp::Ordering::Equal { return match_cmp; }
            ca.distinct_bins.cmp(&cb.distinct_bins)
        });

        // Reorder candidates by score
        let ranked_indices: Vec<usize> = scored.iter().map(|&(ci, _)| ci).collect();

        let t_parallel_ms = t_parallel.elapsed().as_millis();
        println!("    {} proposals parallel in {}ms (max_fan={})",
            candidates.len(), t_parallel_ms, step_max_fan);

        // Accept gate: try candidates in delta_ensemble_val order
        let mut accepted = false;

        for &ci in &ranked_indices {
            let cand = &candidates[ci];
            let delta = scored.iter().find(|&&(i, _)| i == ci).map(|&(_, d)| d).unwrap_or(-999.0);
            if delta <= 0.0 { counters.reject_no_val_gain += 1; continue; }

            if cand.max_match >= 0.999 { counters.reject_duplicate += 1; continue; }

            let werr: f32 = cand.outputs.iter().zip(&data.train).zip(&sw)
                .map(|((&pred, (_, y)), &w)| if pred == *y { 0.0 } else { w }).sum();
            if werr >= 0.499 { counters.reject_weighted_error += 1; continue; }
            let alpha = 0.5 * ((1.0 - werr).max(1e-6) / werr.max(1e-6)).ln();
            let tick = cand.parents.iter().map(|&p| net.sig_ticks[p]).max().unwrap_or(0) + 1;

            let neuron = Neuron {
                id: net.neurons.len(), parents: cand.parents.clone(), tick,
                weights: cand.weights.clone(), bias: cand.bias, threshold: cand.threshold,
                alpha, train_acc: cand.val_acc, val_acc: cand.val_acc,
            };
            net.add(neuron);
            let new_val = net.accuracy(&data.val);

            // ACCEPTED (delta > 0 already checked above)
            counters.accept_count += 1;
            let has_hidden = cand.parents.iter().any(|&p| p >= net.n_in);
            let pnames: Vec<String> = cand.parents.iter().map(|&p| {
                if p < 9 { format!("x{}", p) } else { format!("N{}", p - 9) }
            }).collect();
            let wstr: String = cand.weights.iter().map(|&v| match v { 1=>"+", -1=>"-", _=>"0" }).collect::<Vec<_>>().join("");

            println!("    >> N{}: [{}] b={:+} t={} tick={} parents=[{}] val={:.1}->{:.1}% hidden={} fan={}",
                net.neurons.len()-1, wstr, cand.bias, cand.threshold, tick,
                pnames.join(","), ensemble_val, new_val, has_hidden, cand.fan_in);
            println!("       {}ms parallel, bins={} sparse={} match={:.3}",
                t_parallel_ms, cand.distinct_bins, cand.sparsity, cand.max_match);

            // Reweight (AdaBoost)
            let mut norm = 0.0f32;
            for ((pred, (_, y)), wt) in cand.outputs.iter().zip(&data.train).zip(sw.iter_mut()) {
                let ys = if *y == 1 { 1.0 } else { -1.0 };
                let hs = if *pred == 1 { 1.0 } else { -1.0 };
                *wt *= (-alpha * ys * hs).exp();
                norm += *wt;
            }
            if norm > 0.0 { for w in &mut sw { *w /= norm; } }

            // Checkpoint
            let ckpt = format!("{}/{}_{:03}.json", out_dir, task_name, net.neurons.len());
            save_checkpoint(&net, &ckpt, task_name, step, &data);

            if new_val > best_val + 0.5 { best_val = new_val; stall = 0; }
            else { stall += 1; }

            accepted = true;
            break;
        }

        if !accepted {
            println!("    X All {} proposals rejected ({:.0}ms)", candidates.len(), t0.elapsed().as_millis());
            stall += 1;
        }

        if stall >= cfg.stall_limit { println!("  X Stalled {} steps", cfg.stall_limit); break; }
    }

    let ft = net.accuracy(&data.train);
    let fv = net.accuracy(&data.val);
    let fte = net.accuracy(&data.test);
    let mt = net.neurons.iter().map(|n| n.tick).max().unwrap_or(0);
    let hid = net.neurons.iter().any(|n| n.parents.iter().any(|&p| p >= net.n_in));

    println!("\n  RESULT: {} neurons, depth={}, hidden={}, train={:.1}% val={:.1}% test={:.1}% ({:.1}s)",
        net.neurons.len(), mt, hid, ft, fv, fte, t_task.elapsed().as_secs_f64());
    println!("  Counters: accept={} dup={} werr={} no_gain={}",
        counters.accept_count, counters.reject_duplicate, counters.reject_weighted_error, counters.reject_no_val_gain);

    let ckpt = format!("{}/{}_final.json", out_dir, task_name);
    save_checkpoint(&net, &ckpt, task_name, net.neurons.len(), &data);

    TaskResult {
        name: task_name.to_string(),
        n_neurons: net.neurons.len(), max_tick: mt,
        hidden_parent_used: hid,
        best_train: ft, best_val: fv, best_test: fte,
        wall_s: t_task.elapsed().as_secs_f64(),
        counters,
    }
}

fn save_summary(results: &[TaskResult], seed: u64, total_s: f64, out_dir: &str) {
    let path = format!("{}/summary.json", out_dir);
    if let Some(p) = std::path::Path::new(&path).parent() { std::fs::create_dir_all(p).ok(); }
    let mut f = std::fs::File::create(&path).unwrap();
    writeln!(f, "{{").unwrap();
    writeln!(f, "\"seed\":{},\"wall_clock_s\":{:.1},", seed, total_s).unwrap();
    writeln!(f, "\"tasks\":[").unwrap();
    for (i, r) in results.iter().enumerate() {
        write!(f, "{{\"name\":\"{}\",\"n_neurons\":{},\"max_tick\":{},", r.name, r.n_neurons, r.max_tick).unwrap();
        write!(f, "\"hidden_parent_used\":{},", r.hidden_parent_used).unwrap();
        write!(f, "\"best_train\":{:.2},\"best_val\":{:.2},\"best_test\":{:.2},", r.best_train, r.best_val, r.best_test).unwrap();
        write!(f, "\"wall_s\":{:.1},", r.wall_s).unwrap();
        write!(f, "\"accept_count\":{},\"reject_duplicate\":{},\"reject_weighted_error\":{},\"reject_no_val_gain\":{}",
            r.counters.accept_count, r.counters.reject_duplicate, r.counters.reject_weighted_error, r.counters.reject_no_val_gain).unwrap();
        writeln!(f, "}}{}", if i < results.len() - 1 { "," } else { "" }).unwrap();
    }
    writeln!(f, "],").unwrap();
    let any_hidden = results.iter().any(|r| r.hidden_parent_used);
    let any_depth = results.iter().any(|r| r.max_tick > 1);
    let cont = any_hidden || any_depth;
    let reason = if cont { "depth or hidden parents found" } else { "no depth or hidden parents" };
    writeln!(f, "\"verdict\":{{\"any_hidden_parent\":{},\"any_depth_gt_1\":{},\"continue\":{},\"reason\":\"{}\"}}",
        any_hidden, any_depth, cont, reason).unwrap();
    writeln!(f, "}}").unwrap();
}

fn main() {
    let t0 = Instant::now();
    let cfg = parse_args();

    let out_dir = format!("results/neuron_canonical/{}", cfg.seed);
    std::fs::create_dir_all(&out_dir).unwrap();

    println!("===========================================================");
    println!("  Neuron Canonical Builder (seed={})                       ", cfg.seed);
    println!("  proposals -> backprop -> ternary -> accept gate          ");
    println!("===========================================================");
    println!("  max_neurons={} max_fan={} proposals={} stall={}",
        cfg.max_neurons, cfg.max_fan, cfg.n_proposals, cfg.stall_limit);
    println!("  Tasks: {:?}", cfg.tasks);

    let all_tasks: Vec<(&str, Box<dyn Fn(usize, &[u8]) -> Option<u8>>, f32, usize)> = vec![
        ("digit_parity",  Box::new(|_, px| { let p: usize = px.iter().map(|&v| v as usize).sum(); Some((p % 2) as u8) }), 0.10, 100),
        ("is_symmetric",  Box::new(|_, px| { let s = px[0]==px[2] && px[3]==px[5] && px[6]==px[8]; Some(if s {1} else {0}) }), 0.15, 100),
        ("digit_2_vs_3",  Box::new(|d, _| if d == 2 { Some(0) } else if d == 3 { Some(1) } else { None }), 0.15, 200),
        ("is_digit_0",    Box::new(|d, _| Some(if d == 0 { 1 } else { 0 })), 0.15, 100),
        ("is_digit_even", Box::new(|d, _| Some(if d % 2 == 0 { 1 } else { 0 })), 0.15, 100),
        ("is_digit_gt_4", Box::new(|d, _| Some(if d > 4 { 1 } else { 0 })), 0.15, 100),
    ];

    let tasks: Vec<_> = all_tasks.into_iter()
        .filter(|(name, _, _, _)| cfg.tasks.iter().any(|t| t == name))
        .collect();

    let mut results = Vec::new();
    for (name, label_fn, noise, n_per) in &tasks {
        if t0.elapsed().as_secs() > cfg.wall_clock_min * 60 {
            println!("\n  Wall-clock limit ({} min) reached.", cfg.wall_clock_min);
            break;
        }
        let r = run_task(name, label_fn.as_ref(), *noise, *n_per, &cfg, &out_dir);
        results.push(r);
    }

    save_summary(&results, cfg.seed, t0.elapsed().as_secs_f64(), &out_dir);

    println!("\n  Total: {:.1}s", t0.elapsed().as_secs_f64());
    println!("  Output: {}/summary.json", out_dir);

    let any_hidden = results.iter().any(|r| r.hidden_parent_used);
    let any_depth = results.iter().any(|r| r.max_tick > 1);
    if any_hidden || any_depth {
        println!("  Verdict: CONTINUE");
    } else {
        println!("  Verdict: STOP (no depth/hidden)");
    }
}
