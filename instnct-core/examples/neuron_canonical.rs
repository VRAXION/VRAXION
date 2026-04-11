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
//! Run with seed: cargo run --example neuron_canonical --release -- 123

use std::io::Write as IoWrite;
use std::time::Instant;

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

struct Proposal {
    parents: Vec<usize>,
    float_w: Vec<f32>,
    float_b: f32,
    _score: f32,
}

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

struct TrainedProposal {
    parents: Vec<usize>,
    val_acc: f32,
    sign_consensus: Vec<i8>,  // per weight: +1/-1/0 = consensus sign, 2 = no consensus
}

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
    max_neurons: usize, max_fan: usize,
    out_dir: &str,
    data_seed: u64,
) {
    let data = gen_data(label_fn, noise, n_per, data_seed);
    let mut net = Net::new(9);
    let mut sw = vec![1.0 / data.train.len() as f32; data.train.len()];
    let mut stall = 0;
    let mut best_val = 50.0f32;

    println!("\n══ {} ══", task_name);
    println!("  Data: {} train / {} val / {} test\n", data.train.len(), data.val.len(), data.test.len());

    for step in 0..max_neurons {
        let t0 = Instant::now();
        let ensemble_val = net.accuracy(&data.val);
        let ensemble_test = net.accuracy(&data.test);
        println!("  Step {:2} │ {} neurons │ ensemble val={:.1}% test={:.1}%",
            step, net.neurons.len(), ensemble_val, ensemble_test);

        if ensemble_val >= 99.0 { println!("  ✓ Target reached!"); break; }

        // STEP A: parallel proposals
        let proposals = generate_proposals(&data.train, &net, &sw, 20, max_fan);
        if proposals.is_empty() { println!("  ✗ No proposals"); break; }

        // STEP B: backprop top-5
        let top_k = proposals.len().min(5);
        let mut trained: Vec<TrainedProposal> = Vec::new();
        for p in proposals.iter().take(top_k) {
            let tp = backprop_proposal(
                &data.train, &data.val, &net,
                &p.parents, &p.float_w, p.float_b, &sw, 2000,
            );
            trained.push(tp);
        }
        trained.sort_by(|a, b| b.val_acc.partial_cmp(&a.val_acc).unwrap());

        // STEP C+D: try each trained proposal, quantize, accept/reject
        let all_train_sigs: Vec<Vec<u8>> = data.train.iter().map(|(x, _)| net.eval_all(x)).collect();
        let mut accepted = false;

        for tp in &trained {
            // Quantize (guided + blind)
            let qr = ternary_on_parents(
                &data.train, &data.val, &net, &tp.parents, &sw,
                Some(&tp.sign_consensus),
            );

            // Accept gate
            let is_dup = (net.n_in..net.n_sig()).any(|e| output_match_rate(&qr.outputs, &all_train_sigs, e) >= 0.999);
            if is_dup { continue; }

            // Would this improve ensemble val?
            let tick = tp.parents.iter().map(|&p| net.sig_ticks[p]).max().unwrap_or(0) + 1;
            let werr: f32 = qr.outputs.iter().zip(&data.train).zip(&sw)
                .map(|((&pred, (_, y)), &w)| if pred == *y { 0.0 } else { w }).sum();
            if werr >= 0.499 { continue; }
            let alpha = 0.5 * ((1.0 - werr).max(1e-6) / werr.max(1e-6)).ln();

            // Temporarily add and check val
            let neuron = Neuron {
                id: net.neurons.len(), parents: tp.parents.clone(), tick,
                weights: qr.weights.clone(), bias: qr.bias, threshold: qr.threshold,
                alpha, train_acc: qr.val_acc, val_acc: qr.val_acc,
            };
            net.add(neuron);
            let new_val = net.accuracy(&data.val);

            if new_val <= ensemble_val {
                // Reject: remove
                net.sig_ticks.pop();
                net.neurons.pop();
                continue;
            }

            // ACCEPTED
            let has_hidden = tp.parents.iter().any(|&p| p >= net.n_in);
            let pnames: Vec<String> = tp.parents.iter().map(|&p| {
                if p < 9 { format!("x{}", p) } else { format!("N{}", p - 9) }
            }).collect();
            let wstr: String = qr.weights.iter().map(|&v| match v { 1=>"+", -1=>"-", _=>"0" }).collect::<Vec<_>>().join("");

            println!("    ✓ N{}: [{}] b={:+} t={} tick={} parents=[{}] val={:.1}→{:.1}% hidden={}  ({:.0}ms)",
                net.neurons.len()-1, wstr, qr.bias, qr.threshold, tick,
                pnames.join(","), ensemble_val, new_val, has_hidden, t0.elapsed().as_millis());

            // Reweight (AdaBoost)
            let mut norm = 0.0f32;
            for ((pred, (_, y)), wt) in qr.outputs.iter().zip(&data.train).zip(sw.iter_mut()) {
                let ys = if *y == 1 { 1.0 } else { -1.0 };
                let hs = if *pred == 1 { 1.0 } else { -1.0 };
                *wt *= (-alpha * ys * hs).exp();
                norm += *wt;
            }
            if norm > 0.0 { for w in &mut sw { *w /= norm; } }

            // Checkpoint (with ensemble metrics)
            let ckpt = format!("{}/{}_{:03}.json", out_dir, task_name, net.neurons.len());
            save_checkpoint(&net, &ckpt, task_name, step, &data);

            if new_val > best_val + 0.5 { best_val = new_val; stall = 0; }
            else { stall += 1; }

            accepted = true;
            break;
        }

        if !accepted {
            println!("    ✗ All proposals rejected (no val improvement or duplicate)");
            stall += 1;
        }

        if stall >= 5 { println!("  ✗ Stalled 5 steps"); break; }
    }

    let ft = net.accuracy(&data.train);
    let fv = net.accuracy(&data.val);
    let fte = net.accuracy(&data.test);
    let mt = net.neurons.iter().map(|n| n.tick).max().unwrap_or(0);
    let hid = net.neurons.iter().any(|n| n.parents.iter().any(|&p| p >= net.n_in));

    println!("\n  RESULT: {} neurons, depth={}, hidden={}, train={:.1}% val={:.1}% test={:.1}%",
        net.neurons.len(), mt, hid, ft, fv, fte);

    // Save final
    let ckpt = format!("{}/{}_final.json", out_dir, task_name);
    save_checkpoint(&net, &ckpt, task_name, net.neurons.len(), &data);
}

fn main() {
    let t0 = Instant::now();

    // Seed from CLI arg, default 42
    let data_seed: u64 = std::env::args().nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    let out_dir = format!("results/neuron_canonical/{}", data_seed);
    std::fs::create_dir_all(&out_dir).unwrap();

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Neuron Canonical Builder (seed={:5})                           ║", data_seed);
    println!("║  Serial multi-proposal search (NOT parallel)                    ║");
    println!("║  Per-neuron: proposals → backprop → ternary → accept gate       ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    let tasks: Vec<(&str, Box<dyn Fn(usize, &[u8]) -> Option<u8>>, f32, usize)> = vec![
        ("is_digit_0",    Box::new(|d, _| Some(if d == 0 { 1 } else { 0 })), 0.15, 100),
        ("is_digit_even", Box::new(|d, _| Some(if d % 2 == 0 { 1 } else { 0 })), 0.15, 100),
        ("is_digit_gt_4", Box::new(|d, _| Some(if d > 4 { 1 } else { 0 })), 0.15, 100),
        ("digit_2_vs_3",  Box::new(|d, _| if d == 2 { Some(0) } else if d == 3 { Some(1) } else { None }), 0.15, 200),
        ("is_symmetric",  Box::new(|_, px| { let s = px[0]==px[2] && px[3]==px[5] && px[6]==px[8]; Some(if s {1} else {0}) }), 0.15, 100),
        ("digit_parity",  Box::new(|_, px| { let p: usize = px.iter().map(|&v| v as usize).sum(); Some((p % 2) as u8) }), 0.10, 100),
    ];

    for (name, label_fn, noise, n_per) in &tasks {
        run_task(name, label_fn.as_ref(), *noise, *n_per, 20, 10, &out_dir, data_seed);
    }

    println!("\n  Total: {:.1}s", t0.elapsed().as_secs_f64());
    println!("  Output: {}/", out_dir);
    println!("  Rerun: cargo run --example neuron_canonical --release -- {}", data_seed);
}
