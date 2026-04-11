//! Neuron Build — Real task, checkpoint, incremental grow
//! First real test: 3x3 pixel digit recognition (0-9)
//! Saves checkpoint after every neuron.
//!
//! Run: cargo run --example neuron_build --release

use std::time::Instant;
use std::io::Write as IoWrite;

// ── PRNG ──
struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 { self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.s }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn range(&mut self, lo: f32, hi: f32) -> f32 { lo + self.f32() * (hi - lo) }
    fn bool_p(&mut self, p: f32) -> bool { self.f32() < p }
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

// ═══════════════════════════════════════════════════════
// 3×3 PIXEL FONT — digits 0-9
// Each digit is 9 bits (row-major, top-left first)
//
//  bit layout:
//   0 1 2
//   3 4 5
//   6 7 8
// ═══════════════════════════════════════════════════════

const FONT: [[u8; 9]; 10] = [
    // 0: ███    1: .█.    2: ██.    3: ██.    4: █.█
    //    █.█       .█.       .██       .██       ███
    //    ███       .█.       ██.       ██.       ..█
    [1,1,1, 1,0,1, 1,1,1],  // 0
    [0,1,0, 0,1,0, 0,1,0],  // 1
    [1,1,0, 0,1,1, 1,1,0],  // 2
    [1,1,0, 0,1,1, 1,1,0],  // 3 — same as 2 in 3x3! change it
    [1,0,1, 1,1,1, 0,0,1],  // 4
    // 5: .██    6: █..    7: ███    8: ███    9: ███
    //    ██.       ███       ..█       ███       ███
    //    ██.       ███       ..█       ███       .██
    [0,1,1, 1,1,0, 1,1,0],  // 5
    [1,0,0, 1,1,1, 1,1,1],  // 6
    [1,1,1, 0,0,1, 0,0,1],  // 7
    [1,1,1, 1,1,1, 1,1,1],  // 8
    [1,1,1, 1,1,1, 0,1,1],  // 9
];

// Fix: make digit 3 distinct from 2
// 3: ██.     2: ██.
//    .█.        .██
//    ██.        ██.
const FONT_FIXED: [[u8; 9]; 10] = [
    [1,1,1, 1,0,1, 1,1,1],  // 0: full border
    [0,1,0, 0,1,0, 0,1,0],  // 1: vertical line
    [1,1,0, 0,1,0, 0,1,1],  // 2: Z shape
    [1,1,0, 0,1,0, 1,1,0],  // 3: backward Z
    [1,0,1, 1,1,1, 0,0,1],  // 4: h shape
    [0,1,1, 0,1,0, 1,1,0],  // 5: S shape
    [1,0,0, 1,1,0, 1,1,0],  // 6: L shape
    [1,1,1, 0,0,1, 0,0,1],  // 7: corner
    [1,1,1, 1,1,1, 1,1,1],  // 8: full
    [1,1,1, 1,1,1, 0,1,1],  // 9: full minus corner
];

fn print_digit(pixels: &[u8]) {
    for row in 0..3 {
        for col in 0..3 {
            print!("{}", if pixels[row*3+col] > 0 { "██" } else { "░░" });
        }
        println!();
    }
}

// ═══════════════════════════════════════════════════════
// DATA GENERATION
// ═══════════════════════════════════════════════════════

struct Dataset {
    train: Vec<(Vec<u8>, usize)>,   // (9 pixels, digit 0-9)
    test: Vec<(Vec<u8>, usize)>,
}

fn generate_data(rng: &mut Rng, noise: f32, n_per_digit: usize) -> Dataset {
    let mut train = Vec::new();
    let mut test = Vec::new();

    for digit in 0..10 {
        let template = &FONT_FIXED[digit];
        for i in 0..n_per_digit {
            let mut pixels = template.to_vec();
            // Add noise: flip each pixel with probability `noise`
            for p in pixels.iter_mut() {
                if rng.bool_p(noise) {
                    *p = 1 - *p;
                }
            }
            // 80/20 train/test split
            if i % 5 == 0 {
                test.push((pixels, digit));
            } else {
                train.push((pixels, digit));
            }
        }
    }

    Dataset { train, test }
}

// ═══════════════════════════════════════════════════════
// FROZEN NEURON + NETWORK
// ═══════════════════════════════════════════════════════

#[derive(Clone)]
struct Neuron {
    weights: Vec<i8>,
    bias: i8,
    threshold: i32,
    input_map: Vec<usize>,
    target_class: usize,    // which digit this neuron is trying to detect
    target_bit: usize,      // which bit of the class encoding
}

impl Neuron {
    fn eval(&self, all_values: &[u8]) -> u8 {
        let mut dot = self.bias as i32;
        for (&w, &idx) in self.weights.iter().zip(&self.input_map) {
            dot += (w as i32) * (all_values[idx] as i32);
        }
        if dot >= self.threshold { 1 } else { 0 }
    }
}

struct Network {
    neurons: Vec<Neuron>,
    n_original_inputs: usize,
}

impl Network {
    fn new(n_inputs: usize) -> Self {
        Network { neurons: Vec::new(), n_original_inputs: n_inputs }
    }

    fn eval_all(&self, input: &[u8]) -> Vec<u8> {
        let mut vals: Vec<u8> = input.to_vec();
        for neuron in &self.neurons {
            let out = neuron.eval(&vals);
            vals.push(out);
        }
        vals
    }

    // Classify: use last N neurons as output (one-vs-rest for each digit)
    fn classify(&self, input: &[u8]) -> usize {
        let vals = self.eval_all(input);
        // Find the output neuron with highest confidence
        // Simple: each neuron votes for its target_class
        let mut votes = vec![0i32; 10];
        for (i, neuron) in self.neurons.iter().enumerate() {
            let out = vals[self.n_original_inputs + i];
            if out == 1 {
                votes[neuron.target_class] += 1;
            } else {
                votes[neuron.target_class] -= 1;
            }
        }
        votes.iter().enumerate().max_by_key(|(_,&v)| v).map(|(i,_)| i).unwrap_or(0)
    }

    fn accuracy(&self, data: &[(Vec<u8>, usize)]) -> f32 {
        if self.neurons.is_empty() { return 0.0; }
        let correct = data.iter().filter(|(x, y)| self.classify(x) == *y).count();
        correct as f32 / data.len() as f32 * 100.0
    }

    fn save_checkpoint(&self, path: &str) {
        let mut f = std::fs::File::create(path).unwrap();
        writeln!(f, "{{").unwrap();
        writeln!(f, "  \"n_inputs\": {},", self.n_original_inputs).unwrap();
        writeln!(f, "  \"n_neurons\": {},", self.neurons.len()).unwrap();
        writeln!(f, "  \"neurons\": [").unwrap();
        for (i, n) in self.neurons.iter().enumerate() {
            let w: Vec<String> = n.weights.iter().map(|v| v.to_string()).collect();
            let m: Vec<String> = n.input_map.iter().map(|v| v.to_string()).collect();
            writeln!(f, "    {{\"weights\":[{}],\"bias\":{},\"threshold\":{},\"input_map\":[{}],\"target_class\":{},\"target_bit\":{}}}{}",
                w.join(","), n.bias, n.threshold, m.join(","), n.target_class, n.target_bit,
                if i < self.neurons.len()-1 { "," } else { "" }).unwrap();
        }
        writeln!(f, "  ]").unwrap();
        writeln!(f, "}}").unwrap();
    }
}

// ═══════════════════════════════════════════════════════
// SEARCH: landscape-guided + exhaustive
// ═══════════════════════════════════════════════════════

fn float_landscape_binary(
    patterns: &[(Vec<u8>, u8)],  // (all_values, target_bit)
    input_indices: &[usize],
    n_seeds: usize,
) -> Vec<[usize; 3]> {
    let n_in = input_indices.len();
    let mut signs = vec![[0usize; 3]; n_in];

    for seed in 0..n_seeds {
        let mut rng = Rng::new(seed as u64 * 137 + 42);
        let mut w: Vec<f32> = (0..n_in).map(|_| rng.range(-2.0, 2.0)).collect();
        let mut b = rng.range(-1.0, 1.0);

        for _ in 0..2000 {
            for (vals, target) in patterns {
                let z: f32 = b + (0..n_in).map(|i| w[i] * vals[input_indices[i]] as f32).sum::<f32>();
                let a = sigmoid(z);
                let g = (a - *target as f32) * a * (1.0 - a);
                for i in 0..n_in { w[i] -= g * vals[input_indices[i]] as f32; }
                b -= g;
            }
        }

        let correct = patterns.iter().filter(|(vals, target)| {
            let z: f32 = b + (0..n_in).map(|i| w[i] * vals[input_indices[i]] as f32).sum::<f32>();
            (if sigmoid(z) > 0.5 { 1u8 } else { 0 }) == *target
        }).count();
        if (correct as f32 / patterns.len() as f32) < 0.55 { continue; }

        for i in 0..n_in {
            if w[i] < -0.3 { signs[i][0] += 1; }
            else if w[i] > 0.3 { signs[i][2] += 1; }
            else { signs[i][1] += 1; }
        }
    }
    signs
}

fn exhaustive_search(
    patterns: &[(Vec<u8>, u8)],
    input_indices: &[usize],
) -> (Vec<i8>, i8, i32, f32) {
    let n_in = input_indices.len();
    let total = 3u64.pow((n_in + 1) as u32);
    let n_pat = patterns.len();
    let mut best_w = vec![0i8; n_in];
    let mut best_b: i8 = 0;
    let mut best_t: i32 = 0;
    let mut best_score = 0usize;

    for combo in 0..total {
        let mut w = vec![0i8; n_in];
        let mut r = combo;
        for wi in w.iter_mut() { *wi = (r % 3) as i8 - 1; r /= 3; }
        let b = (r % 3) as i8 - 1;

        let dots: Vec<i32> = patterns.iter().map(|(vals, _)| {
            let mut d = b as i32;
            for (wi, &idx) in w.iter().zip(input_indices) { d += (*wi as i32) * (vals[idx] as i32); }
            d
        }).collect();

        let min_d = dots.iter().copied().min().unwrap_or(0);
        let max_d = dots.iter().copied().max().unwrap_or(0);

        for thresh in (min_d-1)..=(max_d+1) {
            let score = dots.iter().zip(patterns).filter(|(&d, (_, y))| {
                (if d >= thresh { 1u8 } else { 0 }) == *y
            }).count();
            if score > best_score {
                best_score = score; best_w = w.clone(); best_b = b; best_t = thresh;
                if score == n_pat { return (best_w, best_b, best_t, 100.0); }
            }
        }
    }
    (best_w, best_b, best_t, best_score as f32 / n_pat as f32 * 100.0)
}

// ═══════════════════════════════════════════════════════
// MAIN BUILD LOOP
// ═══════════════════════════════════════════════════════

fn main() {
    let t_total = Instant::now();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  VRAXION Neuron Build — 3×3 Digit Recognition                  ║");
    println!("║  Real task, incremental grow, checkpoint after each neuron      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Show the font
    println!("  Font (3×3 pixel, 10 digits):\n");
    for d in 0..10 {
        println!("  Digit {}:", d);
        for row in 0..3 {
            print!("    ");
            for col in 0..3 {
                print!("{}", if FONT_FIXED[d][row*3+col] > 0 { "██" } else { "░░" });
            }
            println!();
        }
    }

    // Generate data
    let mut rng = Rng::new(42);
    let data = generate_data(&mut rng, 0.15, 50);  // 15% noise, 50 samples per digit
    println!("\n  Data: {} train, {} test (15% pixel noise)\n", data.train.len(), data.test.len());

    // Build network
    let mut net = Network::new(9);
    let max_neurons = 30;
    let max_fan_in = 12;  // max inputs per neuron (exhaustive limit)
    let n_classes = 10;

    // Strategy: for each digit, build neurons that detect it (one-vs-rest)
    // Cycle through digits, adding neurons for the worst-performing class

    for step in 0..max_neurons {
        let train_acc = net.accuracy(&data.train);
        let test_acc = net.accuracy(&data.test);

        println!("  ── Step {} ({} neurons) ── train: {:.1}% test: {:.1}%",
            step, net.neurons.len(), train_acc, test_acc);

        if train_acc >= 99.0 && test_acc >= 90.0 {
            println!("  ✓ Sufficient accuracy reached!\n");
            break;
        }

        // Find worst class
        let mut class_acc = vec![0.0f32; n_classes];
        let mut class_count = vec![0usize; n_classes];
        for (x, y) in &data.train {
            class_count[*y] += 1;
            if net.neurons.is_empty() || net.classify(x) == *y {
                class_acc[*y] += 1.0;
            }
        }
        for c in 0..n_classes {
            if class_count[c] > 0 { class_acc[c] /= class_count[c] as f32; }
        }

        let target_class = class_acc.iter().enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i,_)| i).unwrap_or(0);

        println!("    Target: digit {} (class acc: {:.0}%)", target_class, class_acc[target_class] * 100.0);

        // Build binary target: is this digit `target_class`?
        let n_avail = net.n_original_inputs + net.neurons.len();
        let n_in = n_avail.min(max_fan_in);
        let input_indices: Vec<usize> = (0..n_in).collect();

        // Prepare patterns: eval network on all train data, get extended values
        let train_patterns: Vec<(Vec<u8>, u8)> = data.train.iter().map(|(x, y)| {
            let vals = net.eval_all(x);
            let target = if *y == target_class { 1u8 } else { 0 };
            (vals, target)
        }).collect();

        // Float landscape
        let t0 = Instant::now();
        let _signs = float_landscape_binary(&train_patterns, &input_indices, 50);
        let landscape_ms = t0.elapsed().as_millis();

        // Exhaustive search
        let t1 = Instant::now();
        let (w, b, thresh, bit_acc) = exhaustive_search(&train_patterns, &input_indices);
        let search_ms = t1.elapsed().as_millis();

        let w_str: String = w.iter().map(|&v| match v { 1=>"+", -1=>"-", _=>"0" }).collect::<Vec<_>>().join("");
        println!("    Neuron: [{}] b={:+} t={} bit_acc={:.1}% (landscape {}ms, search {}ms)",
            w_str, b, thresh, bit_acc, landscape_ms, search_ms);

        // Add neuron
        let neuron = Neuron {
            weights: w, bias: b, threshold: thresh,
            input_map: input_indices,
            target_class, target_bit: 0,
        };
        net.neurons.push(neuron);

        // Checkpoint
        let ckpt_path = format!("checkpoint_n{:02}.json", net.neurons.len());
        net.save_checkpoint(&ckpt_path);
        println!("    Checkpoint → {}", ckpt_path);

        let new_train = net.accuracy(&data.train);
        let new_test = net.accuracy(&data.test);
        println!("    Network: train {:.1}% test {:.1}% ({:+.1}%)\n",
            new_train, new_test, new_test - test_acc);
    }

    // Final evaluation
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  FINAL RESULTS                                                 ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Neurons: {:3}                                                  ║", net.neurons.len());
    println!("║  Train accuracy: {:5.1}%                                        ║", net.accuracy(&data.train));
    println!("║  Test accuracy:  {:5.1}%                                        ║", net.accuracy(&data.test));
    println!("║  Time: {:.1}s                                                   ║", t_total.elapsed().as_secs_f64());
    println!("╠══════════════════════════════════════════════════════════════════╣");

    // Per-digit accuracy
    println!("║  Per-digit test accuracy:                                       ║");
    for d in 0..10 {
        let total = data.test.iter().filter(|(_, y)| *y == d).count();
        let correct = data.test.iter().filter(|(x, y)| *y == d && net.classify(x) == d).count();
        let acc = if total > 0 { correct as f32 / total as f32 * 100.0 } else { 0.0 };
        let bar_len = (acc / 5.0) as usize;
        println!("║  {:>2}: {:5.1}% [{}{}]  ({}/{})                  ║",
            d, acc, "█".repeat(bar_len), "░".repeat(20-bar_len), correct, total);
    }
    println!("╚══════════════════════════════════════════════════════════════════╝");

    // Save final checkpoint
    net.save_checkpoint("checkpoint_final.json");
    println!("\n  Final checkpoint → checkpoint_final.json");

    // Show some test predictions
    println!("\n  Sample predictions (first 20 test):");
    for (i, (x, y)) in data.test.iter().take(20).enumerate() {
        let pred = net.classify(x);
        let mark = if pred == *y { "✓" } else { "✗" };
        print!("  {} {} pred={} ", mark, y, pred);
        if (i + 1) % 5 == 0 { println!(); }
    }
    println!();
}
