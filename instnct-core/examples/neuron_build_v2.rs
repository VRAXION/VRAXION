//! Neuron Build v2 — Real sparse DAG grower, adversarial-tested
//!
//! Pipeline per neuron:
//!   1. RANK all available signals by correlation with target
//!   2. TOP-K candidates (16-20)
//!   3. GREEDY parent selection: add signals one at a time, keep if improves
//!   4. EXHAUSTIVE ternary search on final parent set
//!   5. FREEZE + checkpoint with full metadata
//!
//! Task: binary one-vs-rest on noisy 3×3 digits ("is this a 0?")
//! DAG only — no recurrence. Tick = 1 + max(parent ticks).
//!
//! Run: cargo run --example neuron_build_v2 --release

use std::time::Instant;
use std::io::Write as IoWrite;

// ── PRNG ──
struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 { self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.s }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn bool_p(&mut self, p: f32) -> bool { self.f32() < p }
}

// ═══════════════════════════════════════════
// 3×3 FONT
// ═══════════════════════════════════════════
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

// ═══════════════════════════════════════════
// DATA — explicit train / val / test splits
// ═══════════════════════════════════════════
struct Split {
    train: Vec<(Vec<u8>, u8)>,   // (pixels, target 0/1)
    val:   Vec<(Vec<u8>, u8)>,
    test:  Vec<(Vec<u8>, u8)>,
}

fn generate_data(target_digit: usize, noise: f32, n_per_digit: usize, seed: u64) -> Split {
    let mut rng = Rng::new(seed);
    let mut train = Vec::new();
    let mut val = Vec::new();
    let mut test = Vec::new();

    for digit in 0..10 {
        let template = &FONT[digit];
        let label = if digit == target_digit { 1u8 } else { 0 };
        for i in 0..n_per_digit {
            let mut pixels = template.to_vec();
            for p in pixels.iter_mut() {
                if rng.bool_p(noise) { *p = 1 - *p; }
            }
            // 60/20/20 split
            match i % 5 {
                0 => val.push((pixels, label)),
                1 => test.push((pixels, label)),
                _ => train.push((pixels, label)),
            }
        }
    }
    Split { train, val, test }
}

// ═══════════════════════════════════════════
// DAG NEURON
// ═══════════════════════════════════════════
#[derive(Clone)]
struct FrozenNeuron {
    id: usize,
    parents: Vec<usize>,    // signal indices (0..8 = inputs, 9+ = neuron outputs)
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

// ═══════════════════════════════════════════
// NETWORK
// ═══════════════════════════════════════════
struct Network {
    neurons: Vec<FrozenNeuron>,
    n_inputs: usize,           // raw input count (9 for 3x3)
    signal_ticks: Vec<u32>,    // tick for each signal (inputs=0, neurons=assigned)
}

impl Network {
    fn new(n_inputs: usize) -> Self {
        Network {
            neurons: Vec::new(),
            n_inputs,
            signal_ticks: vec![0; n_inputs],  // all inputs at tick 0
        }
    }

    fn n_signals(&self) -> usize { self.n_inputs + self.neurons.len() }

    // Eval all signals for one input pattern
    fn eval_all(&self, input: &[u8]) -> Vec<u8> {
        let mut signals: Vec<u8> = input.to_vec();
        for neuron in &self.neurons {
            signals.push(neuron.eval(&signals));
        }
        signals
    }

    // Get output of the LAST neuron (current best detector)
    fn predict(&self, input: &[u8]) -> u8 {
        if self.neurons.is_empty() { return 0; }
        let signals = self.eval_all(input);
        let score: f32 = self.neurons.iter().enumerate().map(|(i, neuron)| {
            let y = if signals[self.n_inputs + i] == 1 { 1.0 } else { -1.0 };
            neuron.alpha * y
        }).sum();
        if score >= 0.0 { 1 } else { 0 }
    }

    fn accuracy(&self, data: &[(Vec<u8>, u8)]) -> f32 {
        if self.neurons.is_empty() { return 50.0; } // random baseline
        let correct = data.iter().filter(|(x, y)| self.predict(x) == *y).count();
        correct as f32 / data.len() as f32 * 100.0
    }

    fn add_neuron(&mut self, neuron: FrozenNeuron) {
        self.signal_ticks.push(neuron.tick);
        self.neurons.push(neuron);
    }
}

// ═══════════════════════════════════════════
// STEP 1: RANK signals by correlation
// ═══════════════════════════════════════════
fn rank_signals(
    data: &[(Vec<u8>, u8)],
    net: &Network,
    sample_weights: &[f32],
) -> Vec<(usize, f32)> {
    // Compute all signal values for all patterns
    let all_signals: Vec<Vec<u8>> = data.iter().map(|(x, _)| net.eval_all(x)).collect();
    let n_sig = net.n_signals();
    let n_pat = data.len();

    let mut ranked: Vec<(usize, f32)> = Vec::new();

    for sig_idx in 0..n_sig {
        // Weighted correlation proxy: |E_w[s*t] - E_w[s]E_w[t]|
        let mut sum_w = 0.0f32;
        let mut sum_s = 0.0f32;
        let mut sum_t = 0.0f32;
        let mut sum_st = 0.0f32;
        for (i, (_, y)) in data.iter().enumerate() {
            let wt = sample_weights[i];
            let s = all_signals[i][sig_idx] as f32;
            let t = *y as f32;
            sum_w += wt;
            sum_s += wt * s;
            sum_t += wt * t;
            sum_st += wt * s * t;
        }
        let denom = sum_w.max(1e-9);
        let corr = (sum_st / denom - (sum_s / denom) * (sum_t / denom)).abs();
        ranked.push((sig_idx, corr));
    }

    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    ranked
}

// ═══════════════════════════════════════════
// STEP 2-3: GREEDY parent selection
// ═══════════════════════════════════════════
fn ternary_search_fixed_parents(
    data: &[(Vec<u8>, u8)],
    all_signals: &[Vec<u8>],
    parents: &[usize],
    sample_weights: &[f32],
) -> (Vec<i8>, i8, i32, f32, Vec<u8>) {
    let n_in = parents.len();
    let total = 3u64.pow((n_in + 1) as u32);
    let n_pat = data.len();
    let mut best_w = vec![0i8; n_in];
    let mut best_b: i8 = 0;
    let mut best_t: i32 = 0;
    let mut best_score = -1.0f32;
    let mut best_outputs = vec![0u8; n_pat];

    for combo in 0..total {
        let mut w = vec![0i8; n_in];
        let mut r = combo;
        for wi in w.iter_mut() { *wi = (r % 3) as i8 - 1; r /= 3; }
        let b = (r % 3) as i8 - 1;

        let dots: Vec<i32> = (0..n_pat).map(|pi| {
            let mut d = b as i32;
            for (wi, &pidx) in w.iter().zip(parents) {
                d += (*wi as i32) * (all_signals[pi][pidx] as i32);
            }
            d
        }).collect();

        let min_d = dots.iter().copied().min().unwrap_or(0);
        let max_d = dots.iter().copied().max().unwrap_or(0);

        for thresh in (min_d - 1)..=(max_d + 1) {
            let outputs: Vec<u8> = dots.iter().map(|&d| if d >= thresh { 1u8 } else { 0 }).collect();
            let score: f32 = outputs.iter().zip(data).zip(sample_weights).map(|((&pred, (_, y)), &wt)| {
                if pred == *y { wt } else { 0.0 }
            }).sum();
            if score > best_score {
                best_score = score;
                best_w = w.clone(); best_b = b; best_t = thresh;
                best_outputs = outputs;
                if (1.0 - score).abs() < 1e-6 { return (best_w, best_b, best_t, 100.0, best_outputs); }
            }
        }
    }
    let weighted_pct = best_score * 100.0;
    (best_w, best_b, best_t, weighted_pct, best_outputs)
}

fn output_match_rate(candidate: &[u8], all_signals: &[Vec<u8>], sig_idx: usize) -> f32 {
    let same = candidate.iter().enumerate().filter(|(i, v)| all_signals[*i][sig_idx] == **v).count();
    same as f32 / candidate.len() as f32
}

fn greedy_parent_selection(
    data: &[(Vec<u8>, u8)],
    net: &Network,
    ranked: &[(usize, f32)],
    sample_weights: &[f32],
    top_k: usize,
    max_fan_in: usize,
) -> (Vec<usize>, Vec<i8>, i8, i32, f32, Vec<u8>, u64, u64) {
    let all_signals: Vec<Vec<u8>> = data.iter().map(|(x, _)| net.eval_all(x)).collect();
    let candidates: Vec<usize> = ranked.iter().take(top_k).map(|(idx, _)| *idx).collect();

    let mut parents: Vec<usize> = Vec::new();
    let mut best_acc = 0.0f32;
    let mut best_w = Vec::new();
    let mut best_b: i8 = 0;
    let mut best_t: i32 = 0;
    let mut best_outputs: Vec<u8> = Vec::new();
    let mut parent_search_combos: u64 = 0;
    let mut ternary_search_combos: u64 = 0;

    // Start with best single signal
    if candidates.is_empty() {
        return (parents, best_w, best_b, best_t, best_acc, best_outputs, 0, 0);
    }

    // Greedy: try adding each candidate, keep if it improves accuracy
    for round in 0..max_fan_in {
        let mut round_best_sig: Option<usize> = None;
        let mut round_best_acc = best_acc;
        let mut round_best_w = best_w.clone();
        let mut round_best_b = best_b;
        let mut round_best_t = best_t;
        let mut round_best_outputs = best_outputs.clone();

        for &sig_idx in &candidates {
            if parents.contains(&sig_idx) { continue; }

            let mut try_parents = parents.clone();
            try_parents.push(sig_idx);

            parent_search_combos += 1;
            let (w, b, t, acc, outputs) = ternary_search_fixed_parents(data, &all_signals, &try_parents, sample_weights);
            ternary_search_combos += 3u64.pow((try_parents.len() + 1) as u32);

            let duplicates_hidden = (net.n_inputs..net.n_signals()).any(|existing| {
                output_match_rate(&outputs, &all_signals, existing) >= 0.999
            });
            if duplicates_hidden {
                continue;
            }

            if acc > round_best_acc {
                round_best_acc = acc;
                round_best_sig = Some(sig_idx);
                round_best_w = w;
                round_best_b = b;
                round_best_t = t;
                round_best_outputs = outputs;
            }

            if acc >= 100.0 { break; }
        }

        if let Some(sig) = round_best_sig {
            parents.push(sig);
            best_acc = round_best_acc;
            best_w = round_best_w;
            best_b = round_best_b;
            best_t = round_best_t;
            best_outputs = round_best_outputs;

            if best_acc >= 100.0 { break; }
        } else {
            break; // no improvement possible
        }

        // Log progress
        println!("      round {}: +sig {} → {} parents, acc={:.1}%",
            round, parents.last().unwrap(), parents.len(), best_acc);
    }

    (parents, best_w, best_b, best_t, best_acc, best_outputs, parent_search_combos, ternary_search_combos)
}

// ═══════════════════════════════════════════
// CHECKPOINT
// ═══════════════════════════════════════════
fn save_checkpoint(net: &Network, path: &str, task_info: &str) {
    let mut f = std::fs::File::create(path).unwrap();
    writeln!(f, "{{").unwrap();
    writeln!(f, "  \"task\": \"{}\",", task_info).unwrap();
    writeln!(f, "  \"n_inputs\": {},", net.n_inputs).unwrap();
    writeln!(f, "  \"n_neurons\": {},", net.neurons.len()).unwrap();
    writeln!(f, "  \"neurons\": [").unwrap();
    for (i, n) in net.neurons.iter().enumerate() {
        let w: Vec<String> = n.weights.iter().map(|v| v.to_string()).collect();
        let p: Vec<String> = n.parents.iter().map(|v| v.to_string()).collect();
        let pt: Vec<String> = n.parent_ticks.iter().map(|v| v.to_string()).collect();
        writeln!(f, "    {{").unwrap();
        writeln!(f, "      \"id\": {},", n.id).unwrap();
        writeln!(f, "      \"parents\": [{}],", p.join(",")).unwrap();
        writeln!(f, "      \"parent_ticks\": [{}],", pt.join(",")).unwrap();
        writeln!(f, "      \"tick\": {},", n.tick).unwrap();
        writeln!(f, "      \"weights\": [{}],", w.join(",")).unwrap();
        writeln!(f, "      \"bias\": {},", n.bias).unwrap();
        writeln!(f, "      \"threshold\": {},", n.threshold).unwrap();
        writeln!(f, "      \"alpha\": {:.6},", n.alpha).unwrap();
        writeln!(f, "      \"train_acc\": {:.2},", n.train_acc).unwrap();
        writeln!(f, "      \"val_acc\": {:.2}", n.val_acc).unwrap();
        writeln!(f, "    }}{}", if i < net.neurons.len()-1 { "," } else { "" }).unwrap();
    }
    writeln!(f, "  ]").unwrap();
    writeln!(f, "}}").unwrap();
}

// ═══════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════
fn main() {
    let t_total = Instant::now();

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Neuron Build v2 — Real DAG Grower (adversarial-tested)         ║");
    println!("║  Task: binary one-vs-rest on noisy 3×3 digits                   ║");
    println!("║  Pipeline: rank signals → greedy parents → exhaustive ternary   ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    let target_digit = 0;
    let noise = 0.15;
    let n_per_digit = 100;
    let data = generate_data(target_digit, noise, n_per_digit, 42);

    println!("  Task: 'Is this digit {}?'", target_digit);
    println!("  Data: {} train / {} val / {} test  (noise={:.0}%)",
        data.train.len(), data.val.len(), data.test.len(), noise * 100.0);
    println!("  Class balance: {}/{} positive in train\n",
        data.train.iter().filter(|(_, y)| *y == 1).count(), data.train.len());

    let mut net = Network::new(9);
    let max_neurons = 20;
    let max_fan_in = 10;  // conservative for CPU speed
    let top_k = 16;
    let mut best_val_acc = 50.0f32;
    let mut stall_count = 0;
    let max_stall = 5;
    let mut sample_weights = vec![1.0 / data.train.len() as f32; data.train.len()];

    for step in 0..max_neurons {
        let step_t0 = Instant::now();

        let train_acc = net.accuracy(&data.train);
        let val_acc = net.accuracy(&data.val);
        let test_acc = net.accuracy(&data.test);

        println!("  ── Step {} ({} neurons) ──", step, net.neurons.len());
        println!("    Current: train={:.1}% val={:.1}% test={:.1}%", train_acc, val_acc, test_acc);

        if val_acc >= 99.0 {
            println!("    ✓ Target accuracy reached on val!\n");
            break;
        }

        // STEP 1: Rank signals
        let t_rank = Instant::now();
        let ranked = rank_signals(&data.train, &net, &sample_weights);
        let rank_ms = t_rank.elapsed().as_millis();

        println!("    Signal ranking ({} signals, {}ms):", net.n_signals(), rank_ms);
        for (i, (sig_idx, corr)) in ranked.iter().take(8).enumerate() {
            let sig_name = if *sig_idx < 9 {
                format!("x{}", sig_idx)
            } else {
                format!("N{} (tick={})", sig_idx - 9, net.signal_ticks[*sig_idx])
            };
            println!("      #{}: {} corr={:.4}", i, sig_name, corr);
        }

        // STEP 2-3: Greedy parent selection + ternary search
        println!("    Greedy parent selection (top-{}, max fan-in {}):", top_k, max_fan_in);
        let t_search = Instant::now();
        let (parents, weights, bias, threshold, bit_acc, train_outputs, p_combos, t_combos) =
            greedy_parent_selection(&data.train, &net, &ranked, &sample_weights, top_k, max_fan_in);
        let search_ms = t_search.elapsed().as_millis();

        if parents.is_empty() {
            println!("    ✗ No useful parents found, stopping.\n");
            break;
        }

        // ADVERSARIAL CHECK: verify parents include hidden neurons if they exist
        let has_hidden_parent = parents.iter().any(|&p| p >= net.n_inputs);
        if net.neurons.len() > 0 && !has_hidden_parent {
            println!("    ⚠ WARN: no hidden neuron selected as parent (all raw inputs)");
        }

        // Compute tick
        let parent_ticks: Vec<u32> = parents.iter().map(|&p| net.signal_ticks[p]).collect();
        let tick = parent_ticks.iter().copied().max().unwrap_or(0) + 1;

        // Parent names for logging
        let parent_names: Vec<String> = parents.iter().map(|&p| {
            if p < 9 { format!("x{}", p) } else { format!("N{}", p - 9) }
        }).collect();
        let w_str: String = weights.iter().map(|&v| match v { 1=>"+", -1=>"-", _=>"0" }).collect::<Vec<_>>().join("");

        println!("    Result: parents=[{}] tick={}",
            parent_names.join(","), tick);
        println!("    Weights: [{}] b={:+} t={}", w_str, bias, threshold);
        println!("    Train bit_acc: {:.1}%", bit_acc);
        println!("    Cost: {} parent-selection tries, {} ternary combos ({}ms)",
            p_combos, t_combos, search_ms);

        // Eval on val
        let all_val_signals: Vec<Vec<u8>> = data.val.iter().map(|(x, _)| net.eval_all(x)).collect();
        let val_bit_acc = {
            let correct = data.val.iter().enumerate().filter(|(i, (_, y))| {
                let mut dot = bias as i32;
                for (&w, &p) in weights.iter().zip(&parents) {
                    dot += (w as i32) * (all_val_signals[*i][p] as i32);
                }
                (if dot >= threshold { 1u8 } else { 0 }) == *y
            }).count();
            correct as f32 / data.val.len() as f32 * 100.0
        };
        println!("    Val bit_acc: {:.1}%", val_bit_acc);

        let weighted_error: f32 = train_outputs.iter().zip(&data.train).zip(&sample_weights).map(|((&pred, (_, y)), &wt)| {
            if pred == *y { 0.0 } else { wt }
        }).sum();
        if weighted_error >= 0.499 {
            println!("    ✗ Weak learner error {:.3} >= 0.499, stopping.\n", weighted_error);
            break;
        }
        let alpha = 0.5 * ((1.0 - weighted_error).max(1e-6) / weighted_error.max(1e-6)).ln();
        println!("    Weighted error: {:.4}, alpha={:.4}", weighted_error, alpha);

        // STEP 4: Freeze
        let neuron = FrozenNeuron {
            id: net.neurons.len(),
            parents: parents.clone(),
            parent_ticks: parent_ticks,
            tick,
            weights,
            bias,
            threshold,
            alpha,
            train_acc: bit_acc,
            val_acc: val_bit_acc,
        };
        net.add_neuron(neuron);

        // Boosting-style residual focus for the next round.
        let mut norm = 0.0f32;
        for ((pred, (_, y)), wt) in train_outputs.iter().zip(&data.train).zip(sample_weights.iter_mut()) {
            let y_sign = if *y == 1 { 1.0 } else { -1.0 };
            let h_sign = if *pred == 1 { 1.0 } else { -1.0 };
            *wt *= (-alpha * y_sign * h_sign).exp();
            norm += *wt;
        }
        if norm > 0.0 {
            for wt in &mut sample_weights { *wt /= norm; }
        }

        // STEP 5: Checkpoint
        let ckpt = format!("ckpt_v2_n{:02}.json", net.neurons.len());
        save_checkpoint(&net, &ckpt, &format!("is_digit_{}", target_digit));

        let new_val = net.accuracy(&data.val);
        let new_test = net.accuracy(&data.test);
        let step_ms = step_t0.elapsed().as_millis();

        println!("    Network val={:.1}% test={:.1}%  (step {}ms)", new_val, new_test, step_ms);
        println!("    Checkpoint → {}\n", ckpt);

        // Stall detection
        if new_val > best_val_acc + 0.5 {
            best_val_acc = new_val;
            stall_count = 0;
        } else {
            stall_count += 1;
            if stall_count >= max_stall {
                println!("  ✗ Stalled for {} neurons, stopping.\n", max_stall);
                break;
            }
        }
    }

    // ═══════════════════════════════════════════
    // FINAL REPORT
    // ═══════════════════════════════════════════
    let total_time = t_total.elapsed();

    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  FINAL REPORT                                                   ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  Neurons:  {:3}                                                  ║", net.neurons.len());
    println!("║  Max tick: {:3}                                                  ║",
        net.neurons.iter().map(|n| n.tick).max().unwrap_or(0));
    println!("║  Train:   {:5.1}%                                                ║", net.accuracy(&data.train));
    println!("║  Val:     {:5.1}%                                                ║", net.accuracy(&data.val));
    println!("║  Test:    {:5.1}%                                                ║", net.accuracy(&data.test));
    println!("║  Time:    {:.1}s                                                 ║", total_time.as_secs_f64());
    println!("╠═══════════════════════════════════════════════════════════════════╣");

    // Adversarial checks
    println!("║  ADVERSARIAL CHECKS:                                            ║");

    // 1. Parent selection real?
    let any_hidden = net.neurons.iter().any(|n| n.parents.iter().any(|&p| p >= 9));
    println!("║  Parent selection uses hidden neurons: {}                       ║",
        if any_hidden { "YES ✓" } else { "NO ✗ (only raw inputs)" });

    // 2. Tick diversity (layers formed?)
    let max_tick = net.neurons.iter().map(|n| n.tick).max().unwrap_or(0);
    let tick_counts: Vec<usize> = (1..=max_tick).map(|t| net.neurons.iter().filter(|n| n.tick == t).count()).collect();
    println!("║  Tick distribution: {:?}        ║", tick_counts);
    println!("║  Layers formed: {}                                              ║",
        if max_tick >= 2 { "YES ✓" } else { "NO (only depth 1)" });

    // 3. Val/test gap
    let val_final = net.accuracy(&data.val);
    let test_final = net.accuracy(&data.test);
    let gap = (val_final - test_final).abs();
    println!("║  Val-test gap: {:.1}% {}                                       ║",
        gap, if gap < 10.0 { "✓ (honest)" } else { "⚠ (possible leak)" });

    // 4. Checkpoint integrity
    let last_ckpt = format!("ckpt_v2_n{:02}.json", net.neurons.len());
    let ckpt_exists = std::path::Path::new(&last_ckpt).exists();
    println!("║  Checkpoint saved: {}                                           ║",
        if ckpt_exists { "YES ✓" } else { "NO ✗" });

    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║  Per-neuron detail:                                             ║");
    for n in &net.neurons {
        let pnames: Vec<String> = n.parents.iter().map(|&p| {
            if p < 9 { format!("x{}", p) } else { format!("N{}", p-9) }
        }).collect();
        println!("║  N{:02} tick={} parents=[{:20}] train={:.1}% val={:.1}%      ║",
            n.id, n.tick, pnames.join(","), n.train_acc, n.val_acc);
    }
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    println!("\n  Rerun: cargo run --example neuron_build_v2 --release");
}
