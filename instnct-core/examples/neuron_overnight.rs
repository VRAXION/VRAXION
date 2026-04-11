//! Overnight DAG Grower — Multi-task neuron build with adversarial metrics
//!
//! DAG only (no recurrence). Tick = 1 + max(parent ticks).
//! Proves: can the builder produce depth > 1 and use hidden parents?
//!
//! Output: instnct-core/results/neuron_overnight/<seed>/
//!   summary.json, run.log, checkpoints/<task>_nNN.json
//!
//! Run: cargo run --example neuron_overnight --release

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
    fn bool_p(&mut self, p: f32) -> bool { self.f32() < p }
}

fn _sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

// ══════════════════════════════════════════════════════
// 3×3 FONT
// ══════════════════════════════════════════════════════
const FONT: [[u8; 9]; 10] = [
    [1,1,1, 1,0,1, 1,1,1],  // 0
    [0,1,0, 0,1,0, 0,1,0],  // 1
    [1,1,0, 0,1,0, 0,1,1],  // 2
    [1,1,0, 0,1,0, 1,1,0],  // 3
    [1,0,1, 1,1,1, 0,0,1],  // 4
    [0,1,1, 0,1,0, 1,1,0],  // 5
    [1,0,0, 1,1,0, 1,1,0],  // 6
    [1,1,1, 0,0,1, 0,0,1],  // 7
    [1,1,1, 1,1,1, 1,1,1],  // 8
    [1,1,1, 1,1,1, 0,1,1],  // 9
];

// ══════════════════════════════════════════════════════
// DATA
// ══════════════════════════════════════════════════════
struct Split {
    train: Vec<(Vec<u8>, u8)>,
    val:   Vec<(Vec<u8>, u8)>,
    test:  Vec<(Vec<u8>, u8)>,
}

fn gen_binary_task(
    label_fn: impl Fn(usize, &[u8]) -> Option<u8>,  // digit, pixels → Some(label) or None (skip)
    noise: f32, n_per_digit: usize, seed: u64,
) -> Split {
    let mut rng = Rng::new(seed);
    let mut train = Vec::new();
    let mut val = Vec::new();
    let mut test = Vec::new();
    for digit in 0..10 {
        let template = &FONT[digit];
        for i in 0..n_per_digit {
            let mut pixels = template.to_vec();
            for p in pixels.iter_mut() { if rng.bool_p(noise) { *p = 1 - *p; } }
            if let Some(label) = label_fn(digit, &pixels) {
                match i % 5 {
                    0 => val.push((pixels, label)),
                    1 => test.push((pixels, label)),
                    _ => train.push((pixels, label)),
                }
            }
        }
    }
    Split { train, val, test }
}

// ══════════════════════════════════════════════════════
// TASK DEFINITIONS
// ══════════════════════════════════════════════════════
struct TaskDef {
    name: &'static str,
    noise: f32,
    n_per_digit: usize,
}

fn task_suite() -> Vec<(TaskDef, Box<dyn Fn(usize, &[u8]) -> Option<u8>>)> {
    vec![
        (TaskDef { name: "is_digit_0", noise: 0.15, n_per_digit: 100 },
         Box::new(|d, _| Some(if d == 0 { 1 } else { 0 }))),

        (TaskDef { name: "is_digit_even", noise: 0.15, n_per_digit: 100 },
         Box::new(|d, _| Some(if d % 2 == 0 { 1 } else { 0 }))),

        (TaskDef { name: "is_digit_gt_4", noise: 0.15, n_per_digit: 100 },
         Box::new(|d, _| Some(if d > 4 { 1 } else { 0 }))),

        (TaskDef { name: "digit_2_vs_3", noise: 0.15, n_per_digit: 200 },
         Box::new(|d, _| {
             if d == 2 { Some(0) } else if d == 3 { Some(1) } else { None }
         })),

        (TaskDef { name: "is_symmetric", noise: 0.15, n_per_digit: 100 },
         Box::new(|_, px| {
             let sym = px[0] == px[2] && px[3] == px[5] && px[6] == px[8];
             Some(if sym { 1 } else { 0 })
         })),

        (TaskDef { name: "digit_parity", noise: 0.10, n_per_digit: 100 },
         Box::new(|_, px| {
             let pop: usize = px.iter().map(|&p| p as usize).sum();
             Some((pop % 2) as u8)
         })),
    ]
}

// ══════════════════════════════════════════════════════
// FROZEN NEURON + NETWORK (from v2, unchanged core)
// ══════════════════════════════════════════════════════
#[derive(Clone)]
struct FrozenNeuron {
    id: usize,
    parents: Vec<usize>,
    parent_ticks: Vec<u32>,
    tick: u32,
    weights: Vec<i8>,
    bias: i8,
    threshold: i32,
    alpha: f32,
    train_acc: f32,
    val_acc: f32,
}

impl FrozenNeuron {
    fn eval(&self, signals: &[u8]) -> u8 {
        let mut dot = self.bias as i32;
        for (&w, &p) in self.weights.iter().zip(&self.parents) {
            dot += (w as i32) * (signals[p] as i32);
        }
        if dot >= self.threshold { 1 } else { 0 }
    }
}

struct Network {
    neurons: Vec<FrozenNeuron>,
    n_inputs: usize,
    signal_ticks: Vec<u32>,
}

impl Network {
    fn new(n_inputs: usize) -> Self {
        Network { neurons: Vec::new(), n_inputs, signal_ticks: vec![0; n_inputs] }
    }
    fn n_signals(&self) -> usize { self.n_inputs + self.neurons.len() }

    fn eval_all(&self, input: &[u8]) -> Vec<u8> {
        let mut signals: Vec<u8> = input.to_vec();
        for neuron in &self.neurons { signals.push(neuron.eval(&signals)); }
        signals
    }

    fn predict(&self, input: &[u8]) -> u8 {
        if self.neurons.is_empty() { return 0; }
        let signals = self.eval_all(input);
        let score: f32 = self.neurons.iter().enumerate().map(|(i, n)| {
            let y = if signals[self.n_inputs + i] == 1 { 1.0 } else { -1.0 };
            n.alpha * y
        }).sum();
        if score >= 0.0 { 1 } else { 0 }
    }

    fn accuracy(&self, data: &[(Vec<u8>, u8)]) -> f32 {
        if self.neurons.is_empty() { return 50.0; }
        let correct = data.iter().filter(|(x, y)| self.predict(x) == *y).count();
        correct as f32 / data.len() as f32 * 100.0
    }

    fn add_neuron(&mut self, neuron: FrozenNeuron) {
        self.signal_ticks.push(neuron.tick);
        self.neurons.push(neuron);
    }
}

// ══════════════════════════════════════════════════════
// SIGNAL RANKING (weighted correlation)
// ══════════════════════════════════════════════════════
fn rank_signals(data: &[(Vec<u8>, u8)], net: &Network, weights: &[f32]) -> Vec<(usize, f32)> {
    let all_signals: Vec<Vec<u8>> = data.iter().map(|(x, _)| net.eval_all(x)).collect();
    let n_sig = net.n_signals();
    let mut ranked: Vec<(usize, f32)> = Vec::new();

    for sig_idx in 0..n_sig {
        let mut sw = 0.0f32; let mut ss = 0.0f32; let mut st = 0.0f32; let mut sst = 0.0f32;
        for (i, (_, y)) in data.iter().enumerate() {
            let w = weights[i];
            let s = all_signals[i][sig_idx] as f32;
            let t = *y as f32;
            sw += w; ss += w * s; st += w * t; sst += w * s * t;
        }
        let d = sw.max(1e-9);
        ranked.push((sig_idx, (sst / d - (ss / d) * (st / d)).abs()));
    }
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    ranked
}

// ══════════════════════════════════════════════════════
// TERNARY EXHAUSTIVE on fixed parents
// ══════════════════════════════════════════════════════
fn ternary_search(
    data: &[(Vec<u8>, u8)], all_sigs: &[Vec<u8>], parents: &[usize], sw: &[f32],
) -> (Vec<i8>, i8, i32, f32, Vec<u8>) {
    let ni = parents.len();
    let total = 3u64.pow((ni + 1) as u32);
    let np = data.len();
    let mut bw = vec![0i8; ni]; let mut bb: i8 = 0; let mut bt: i32 = 0;
    let mut bs = -1.0f32; let mut bo = vec![0u8; np];

    for combo in 0..total {
        let mut w = vec![0i8; ni]; let mut r = combo;
        for wi in w.iter_mut() { *wi = (r % 3) as i8 - 1; r /= 3; }
        let b = (r % 3) as i8 - 1;

        let dots: Vec<i32> = (0..np).map(|pi| {
            let mut d = b as i32;
            for (wi, &pidx) in w.iter().zip(parents) { d += (*wi as i32) * (all_sigs[pi][pidx] as i32); }
            d
        }).collect();

        let min_d = dots.iter().copied().min().unwrap_or(0);
        let max_d = dots.iter().copied().max().unwrap_or(0);

        for thresh in (min_d - 1)..=(max_d + 1) {
            let outputs: Vec<u8> = dots.iter().map(|&d| if d >= thresh { 1 } else { 0 }).collect();
            let score: f32 = outputs.iter().zip(data).zip(sw).map(|((&pred, (_, y)), &wt)| {
                if pred == *y { wt } else { 0.0 }
            }).sum();
            if score > bs {
                bs = score; bw = w.clone(); bb = b; bt = thresh; bo = outputs;
            }
        }
    }
    (bw, bb, bt, bs * 100.0, bo)
}

fn output_match_rate(candidate: &[u8], all_sigs: &[Vec<u8>], sig_idx: usize) -> f32 {
    let same = candidate.iter().enumerate().filter(|(i, v)| all_sigs[*i][sig_idx] == **v).count();
    same as f32 / candidate.len() as f32
}

// ══════════════════════════════════════════════════════
// GREEDY PARENT SELECTION
// ══════════════════════════════════════════════════════
fn greedy_parents(
    data: &[(Vec<u8>, u8)], net: &Network, ranked: &[(usize, f32)],
    sw: &[f32], top_k: usize, max_fan: usize, log: &mut Vec<String>,
) -> (Vec<usize>, Vec<i8>, i8, i32, f32, Vec<u8>, u64) {
    let all_sigs: Vec<Vec<u8>> = data.iter().map(|(x, _)| net.eval_all(x)).collect();
    let candidates: Vec<usize> = ranked.iter().take(top_k).map(|(i, _)| *i).collect();
    let mut parents: Vec<usize> = Vec::new();
    let mut ba = 0.0f32; let mut bw = Vec::new(); let mut bb: i8 = 0; let mut bt: i32 = 0;
    let mut bouts: Vec<u8> = Vec::new(); let mut total_combos: u64 = 0;

    for round in 0..max_fan {
        let mut rbest: Option<(usize, Vec<i8>, i8, i32, f32, Vec<u8>)> = None;
        for &sig in &candidates {
            if parents.contains(&sig) { continue; }
            let mut tp = parents.clone(); tp.push(sig);
            let (w, b, t, acc, outs) = ternary_search(data, &all_sigs, &tp, sw);
            total_combos += 3u64.pow((tp.len() + 1) as u32);

            // Duplicate filter
            let dup = (net.n_inputs..net.n_signals()).any(|e| output_match_rate(&outs, &all_sigs, e) >= 0.999);
            if dup { continue; }

            if rbest.is_none() || acc > rbest.as_ref().unwrap().4 {
                rbest = Some((sig, w, b, t, acc, outs));
            }
        }
        if let Some((sig, w, b, t, acc, outs)) = rbest {
            if acc <= ba && round > 0 { break; }
            parents.push(sig); ba = acc; bw = w; bb = b; bt = t; bouts = outs;
            let sig_name = if sig < net.n_inputs { format!("x{}", sig) } else { format!("N{}", sig - net.n_inputs) };
            let msg = format!("      round {}: +{} → {} parents, wacc={:.1}%", round, sig_name, parents.len(), ba);
            log.push(msg.clone()); println!("{}", msg);
            if ba >= 100.0 { break; }
        } else { break; }
    }
    (parents, bw, bb, bt, ba, bouts, total_combos)
}

// ══════════════════════════════════════════════════════
// BUILD ONE TASK
// ══════════════════════════════════════════════════════
#[allow(dead_code)]
struct TaskResult {
    name: String,
    n_neurons: usize,
    max_tick: u32,
    hidden_parent_used: bool,
    hidden_parent_neurons: Vec<usize>,
    duplicate_rejections: usize,
    best_train: f32,
    best_val: f32,
    best_test: f32,
    later_useful: bool,
    neurons: Vec<FrozenNeuron>,
}

fn build_task(
    task_name: &str, data: &Split, seed: u64, out_dir: &str, log: &mut Vec<String>,
) -> TaskResult {
    let mut net = Network::new(9);
    let max_neurons = 20;
    let max_fan = 10;
    let top_k = 16;
    let max_stall = 5;
    let mut sw = vec![1.0 / data.train.len() as f32; data.train.len()];
    let mut best_val = 50.0f32;
    let mut stall = 0;
    let dup_rejections = 0usize;
    let mut hidden_parent_neurons: Vec<usize> = Vec::new();

    let msg = format!("  ── {} (seed={}) ──", task_name, seed);
    log.push(msg.clone()); println!("{}", msg);
    let msg = format!("    Data: {} train / {} val / {} test", data.train.len(), data.val.len(), data.test.len());
    log.push(msg.clone()); println!("{}", msg);

    for step in 0..max_neurons {
        let train_acc = net.accuracy(&data.train);
        let val_acc = net.accuracy(&data.val);
        let test_acc = net.accuracy(&data.test);

        let msg = format!("    Step {} ({} neurons): train={:.1}% val={:.1}% test={:.1}%",
            step, net.neurons.len(), train_acc, val_acc, test_acc);
        log.push(msg.clone()); println!("{}", msg);

        if val_acc >= 99.0 { log.push("    ✓ Target reached".into()); println!("    ✓ Target reached"); break; }

        let ranked = rank_signals(&data.train, &net, &sw);
        let (parents, weights, bias, threshold, _wacc, train_outputs, _combos) =
            greedy_parents(&data.train, &net, &ranked, &sw, top_k, max_fan, log);

        if parents.is_empty() { log.push("    ✗ No parents found".into()); println!("    ✗ No parents found"); break; }

        let has_hidden = parents.iter().any(|&p| p >= net.n_inputs);
        let parent_ticks: Vec<u32> = parents.iter().map(|&p| net.signal_ticks[p]).collect();
        let tick = parent_ticks.iter().copied().max().unwrap_or(0) + 1;

        // Weighted error + alpha
        let werr: f32 = train_outputs.iter().zip(&data.train).zip(&sw)
            .map(|((&pred, (_, y)), &w)| if pred == *y { 0.0 } else { w }).sum();
        if werr >= 0.499 {
            log.push(format!("    ✗ Weak learner error={:.3}", werr));
            println!("    ✗ Weak learner error={:.3}", werr);
            break;
        }
        let alpha = 0.5 * ((1.0 - werr).max(1e-6) / werr.max(1e-6)).ln();

        // Val accuracy of this neuron alone
        let all_val_sigs: Vec<Vec<u8>> = data.val.iter().map(|(x, _)| net.eval_all(x)).collect();
        let val_bit: f32 = {
            let c = data.val.iter().enumerate().filter(|(i, (_, y))| {
                let mut d = bias as i32;
                for (&w, &p) in weights.iter().zip(&parents) { d += (w as i32) * (all_val_sigs[*i][p] as i32); }
                (if d >= threshold { 1u8 } else { 0 }) == *y
            }).count();
            c as f32 / data.val.len() as f32 * 100.0
        };

        let pnames: Vec<String> = parents.iter().map(|&p| {
            if p < 9 { format!("x{}", p) } else { format!("N{}", p - 9) }
        }).collect();
        let wstr: String = weights.iter().map(|&v| match v { 1=>"+", -1=>"-", _=>"0"}).collect::<Vec<_>>().join("");

        let msg = format!("    N{}: [{}] b={:+} t={} tick={} parents=[{}] train={:.1}% val={:.1}% alpha={:.3} hidden={}",
            net.neurons.len(), wstr, bias, threshold, tick, pnames.join(","), _wacc, val_bit, alpha, has_hidden);
        log.push(msg.clone()); println!("{}", msg);

        if has_hidden { hidden_parent_neurons.push(net.neurons.len()); }

        let neuron = FrozenNeuron {
            id: net.neurons.len(), parents: parents.clone(), parent_ticks, tick,
            weights, bias, threshold, alpha, train_acc: _wacc, val_acc: val_bit,
        };
        net.add_neuron(neuron);

        // Reweight (AdaBoost)
        let mut norm = 0.0f32;
        for ((pred, (_, y)), wt) in train_outputs.iter().zip(&data.train).zip(sw.iter_mut()) {
            let ys = if *y == 1 { 1.0 } else { -1.0 };
            let hs = if *pred == 1 { 1.0 } else { -1.0 };
            *wt *= (-alpha * ys * hs).exp();
            norm += *wt;
        }
        if norm > 0.0 { for w in &mut sw { *w /= norm; } }

        // Checkpoint
        let ckpt = format!("{}/checkpoints/{}_{:02}.json", out_dir, task_name, net.neurons.len());
        save_neuron_checkpoint(&net, &ckpt, task_name);

        let new_val = net.accuracy(&data.val);
        if new_val > best_val + 0.5 { best_val = new_val; stall = 0; }
        else { stall += 1; if stall >= max_stall { log.push(format!("    ✗ Stalled {} steps", max_stall)); println!("    ✗ Stalled {} steps", max_stall); break; } }
    }

    let final_train = net.accuracy(&data.train);
    let final_val = net.accuracy(&data.val);
    let final_test = net.accuracy(&data.test);
    let max_tick = net.neurons.iter().map(|n| n.tick).max().unwrap_or(0);
    let first_val = if net.neurons.len() > 0 {
        // Eval with only first neuron
        let mut net1 = Network::new(9);
        net1.add_neuron(net.neurons[0].clone());
        net1.accuracy(&data.val)
    } else { 50.0 };
    let later_useful = final_val > first_val + 1.0;

    let msg = format!("    FINAL: {} neurons, max_tick={}, train={:.1}% val={:.1}% test={:.1}% hidden_parents={} later_useful={}",
        net.neurons.len(), max_tick, final_train, final_val, final_test, !hidden_parent_neurons.is_empty(), later_useful);
    log.push(msg.clone()); println!("{}", msg);

    TaskResult {
        name: task_name.to_string(), n_neurons: net.neurons.len(), max_tick,
        hidden_parent_used: !hidden_parent_neurons.is_empty(),
        hidden_parent_neurons, duplicate_rejections: dup_rejections,
        best_train: final_train, best_val: final_val, best_test: final_test,
        later_useful, neurons: net.neurons,
    }
}

fn save_neuron_checkpoint(net: &Network, path: &str, task: &str) {
    if let Some(parent) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut f = std::fs::File::create(path).unwrap();
    writeln!(f, "{{\"task\":\"{}\",\"n_inputs\":{},\"n_neurons\":{},\"neurons\":[", task, net.n_inputs, net.neurons.len()).unwrap();
    for (i, n) in net.neurons.iter().enumerate() {
        let w: Vec<String> = n.weights.iter().map(|v| v.to_string()).collect();
        let p: Vec<String> = n.parents.iter().map(|v| v.to_string()).collect();
        let pt: Vec<String> = n.parent_ticks.iter().map(|v| v.to_string()).collect();
        writeln!(f, "{{\"id\":{},\"parents\":[{}],\"parent_ticks\":[{}],\"tick\":{},\"weights\":[{}],\"bias\":{},\"threshold\":{},\"alpha\":{:.6},\"train_acc\":{:.2},\"val_acc\":{:.2}}}{}",
            n.id, p.join(","), pt.join(","), n.tick, w.join(","), n.bias, n.threshold, n.alpha, n.train_acc, n.val_acc,
            if i < net.neurons.len()-1 { "," } else { "" }).unwrap();
    }
    writeln!(f, "]}}").unwrap();
}

// ══════════════════════════════════════════════════════
// SUMMARY JSON
// ══════════════════════════════════════════════════════
fn write_summary(results: &[TaskResult], seed: u64, wall_s: f64, path: &str) {
    let mut f = std::fs::File::create(path).unwrap();
    let any_hidden = results.iter().any(|r| r.hidden_parent_used);
    let any_depth = results.iter().any(|r| r.max_tick > 1);
    let should_continue = any_hidden || any_depth;
    let reason = if should_continue {
        if any_hidden && any_depth { "hidden parents AND depth > 1 observed" }
        else if any_hidden { "hidden parents observed (no depth > 1)" }
        else { "depth > 1 observed (no hidden parents)" }
    } else { "no task produced hidden parents or depth > 1" };

    writeln!(f, "{{").unwrap();
    writeln!(f, "  \"seed\": {},", seed).unwrap();
    writeln!(f, "  \"wall_clock_s\": {:.1},", wall_s).unwrap();
    writeln!(f, "  \"tasks\": [").unwrap();
    for (ti, r) in results.iter().enumerate() {
        writeln!(f, "    {{").unwrap();
        writeln!(f, "      \"name\": \"{}\",", r.name).unwrap();
        writeln!(f, "      \"n_neurons\": {},", r.n_neurons).unwrap();
        writeln!(f, "      \"max_tick\": {},", r.max_tick).unwrap();
        writeln!(f, "      \"hidden_parent_used\": {},", r.hidden_parent_used).unwrap();
        writeln!(f, "      \"duplicate_rejections\": {},", r.duplicate_rejections).unwrap();
        writeln!(f, "      \"best_train\": {:.1},", r.best_train).unwrap();
        writeln!(f, "      \"best_val\": {:.1},", r.best_val).unwrap();
        writeln!(f, "      \"best_test\": {:.1},", r.best_test).unwrap();
        writeln!(f, "      \"later_neurons_useful\": {},", r.later_useful).unwrap();
        write!(f, "      \"neurons\": [").unwrap();
        for (ni, n) in r.neurons.iter().enumerate() {
            let pj: Vec<String> = n.parents.iter().map(|v| v.to_string()).collect();
            write!(f, "{{\"id\":{},\"parents\":[{}],\"tick\":{},\"train_acc\":{:.1},\"val_acc\":{:.1}}}",
                n.id, pj.join(","), n.tick, n.train_acc, n.val_acc).unwrap();
            if ni < r.neurons.len() - 1 { write!(f, ",").unwrap(); }
        }
        writeln!(f, "]").unwrap();
        writeln!(f, "    }}{}", if ti < results.len() - 1 { "," } else { "" }).unwrap();
    }
    writeln!(f, "  ],").unwrap();
    writeln!(f, "  \"verdict\": {{").unwrap();
    writeln!(f, "    \"any_hidden_parent\": {},", any_hidden).unwrap();
    writeln!(f, "    \"any_depth_gt_1\": {},", any_depth).unwrap();
    writeln!(f, "    \"continue\": {},", should_continue).unwrap();
    writeln!(f, "    \"reason\": \"{}\"", reason).unwrap();
    writeln!(f, "  }}").unwrap();
    writeln!(f, "}}").unwrap();
}

// ══════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════
fn main() {
    let total_t0 = Instant::now();
    let seeds: Vec<u64> = vec![42, 123, 7, 999, 2024, 314, 55, 808];
    let max_wall_secs = 4.0 * 3600.0; // 4 hours

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Overnight DAG Grower — Multi-task, adversarial-tested          ║");
    println!("║  6 tasks × up to 8 seeds, 4h wall-clock budget                 ║");
    println!("║  DAG only, tick = 1 + max(parent ticks), no recurrence         ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    let base_dir = "results/neuron_overnight";

    for (pass_idx, &seed) in seeds.iter().enumerate() {
        if total_t0.elapsed().as_secs_f64() > max_wall_secs {
            println!("\n  ⏰ Wall-clock budget exceeded, stopping.");
            break;
        }

        let pass_t0 = Instant::now();
        let out_dir = format!("{}/{}", base_dir, seed);
        std::fs::create_dir_all(format!("{}/checkpoints", out_dir)).unwrap();

        println!("\n══════════════════════════════════════════════════════════════");
        println!("  PASS {} — seed={}", pass_idx + 1, seed);
        println!("══════════════════════════════════════════════════════════════\n");

        let mut log: Vec<String> = Vec::new();
        let mut results: Vec<TaskResult> = Vec::new();

        let suite = task_suite();
        for (task_def, label_fn) in &suite {
            let data = gen_binary_task(|d, px| label_fn(d, px), task_def.noise, task_def.n_per_digit, seed);
            let result = build_task(task_def.name, &data, seed, &out_dir, &mut log);
            results.push(result);
            println!();
        }

        // Write log
        let log_path = format!("{}/run.log", out_dir);
        let mut lf = std::fs::File::create(&log_path).unwrap();
        for line in &log { writeln!(lf, "{}", line).unwrap(); }

        // Write summary
        let wall_s = pass_t0.elapsed().as_secs_f64();
        let summary_path = format!("{}/summary.json", out_dir);
        write_summary(&results, seed, wall_s, &summary_path);

        // Print pass summary
        println!("  ── Pass {} Summary ──", pass_idx + 1);
        println!("  {:16} │ Neur │ Depth │ Hidden │ Train │  Val  │ Test  │ Later?",
            "Task");
        println!("  ─────────────────┼──────┼───────┼────────┼───────┼───────┼───────┼──────");
        for r in &results {
            println!("  {:16} │  {:3} │   {:3} │  {:5} │ {:5.1} │ {:5.1} │ {:5.1} │ {}",
                r.name, r.n_neurons, r.max_tick,
                if r.hidden_parent_used { "YES" } else { "no" },
                r.best_train, r.best_val, r.best_test,
                if r.later_useful { "YES" } else { "no" });
        }

        let any_hidden = results.iter().any(|r| r.hidden_parent_used);
        let any_depth = results.iter().any(|r| r.max_tick > 1);

        println!("\n  Verdict: hidden_parents={} depth>1={}", any_hidden, any_depth);
        println!("  Output: {}/summary.json", out_dir);
        println!("  Pass time: {:.1}s", wall_s);

        // Hard stop: first pass must prove value
        if pass_idx == 0 && !any_hidden && !any_depth {
            println!("\n  ✗ HARD STOP: First pass produced no hidden parents and no depth > 1.");
            println!("    The builder cannot produce depth on these tasks.");
            println!("    Stopping — not burning the night on baseline-only results.");
            break;
        }
    }

    println!("\n  Total wall-clock: {:.1}s", total_t0.elapsed().as_secs_f64());
}
