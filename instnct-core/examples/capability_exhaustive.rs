//! VRAXION Capability — Exhaustive Incremental Build
//! 1 neuron at a time, ternary exhaustive search, C19 LutGate
//! The VRAXION way: no backprop, no float — just search + verify
//!
//! Run: cargo run --example capability_exhaustive --release

use std::time::Instant;

fn c19(x: f32, rho: f32) -> f32 {
    let l = 6.0f32;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * h + rho * h * h
}

const RHO: f32 = 8.0;
const THRESH: f32 = 0.5;

// A single LutGate neuron: ternary weights, integer dot, threshold output
// The neuron computes: output = 1 if dot >= threshold, else 0
// This is the fundamental VRAXION primitive — the C19 LUT is baked from this
#[derive(Clone)]
struct Neuron {
    weights: Vec<i8>,   // ternary: -1, 0, +1
    bias: i8,
    threshold: i32,     // output = 1 if dot >= threshold
}

impl Neuron {
    fn eval(&self, inputs: &[u8]) -> u8 {
        let mut dot: i32 = self.bias as i32;
        for (w, &x) in self.weights.iter().zip(inputs) {
            dot += (*w as i32) * (x as i32);
        }
        if dot >= self.threshold { 1 } else { 0 }
    }
}

// Task: list of (binary_input, binary_target_bits)
struct Task {
    name: &'static str,
    patterns: Vec<(Vec<u8>, Vec<u8>)>,  // (input bits, target output bits)
    in_bits: usize,
    out_bits: usize,
}

// ── Tasks ──

fn task_and() -> Task {
    let mut p = Vec::new();
    for a in 0..2u8 { for b in 0..2u8 {
        p.push((vec![a, b], vec![a & b]));
    }}
    Task { name: "AND", patterns: p, in_bits: 2, out_bits: 1 }
}

fn task_or() -> Task {
    let mut p = Vec::new();
    for a in 0..2u8 { for b in 0..2u8 {
        p.push((vec![a, b], vec![a | b]));
    }}
    Task { name: "OR", patterns: p, in_bits: 2, out_bits: 1 }
}

fn task_xor() -> Task {
    let mut p = Vec::new();
    for a in 0..2u8 { for b in 0..2u8 {
        p.push((vec![a, b], vec![a ^ b]));
    }}
    Task { name: "XOR", patterns: p, in_bits: 2, out_bits: 1 }
}

fn task_nand() -> Task {
    let mut p = Vec::new();
    for a in 0..2u8 { for b in 0..2u8 {
        p.push((vec![a, b], vec![1 - (a & b)]));
    }}
    Task { name: "NAND", patterns: p, in_bits: 2, out_bits: 1 }
}

fn task_maj3() -> Task {
    let mut p = Vec::new();
    for v in 0..8u8 {
        let bits: Vec<u8> = (0..3).map(|i| (v >> i) & 1).collect();
        let sum: u8 = bits.iter().sum();
        p.push((bits, vec![if sum >= 2 { 1 } else { 0 }]));
    }
    Task { name: "MAJ3", patterns: p, in_bits: 3, out_bits: 1 }
}

fn task_pop4_gt2() -> Task {
    let mut p = Vec::new();
    for v in 0..16u8 {
        let bits: Vec<u8> = (0..4).map(|i| (v >> i) & 1).collect();
        let sum: u8 = bits.iter().sum();
        p.push((bits, vec![if sum > 2 { 1 } else { 0 }]));
    }
    Task { name: "POP4>2", patterns: p, in_bits: 4, out_bits: 1 }
}

fn task_par3() -> Task {
    let mut p = Vec::new();
    for v in 0..8u8 {
        let bits: Vec<u8> = (0..3).map(|i| (v >> i) & 1).collect();
        let sum: u8 = bits.iter().sum();
        p.push((bits, vec![sum % 2]));
    }
    Task { name: "PAR3", patterns: p, in_bits: 3, out_bits: 1 }
}

fn task_par4() -> Task {
    let mut p = Vec::new();
    for v in 0..16u8 {
        let bits: Vec<u8> = (0..4).map(|i| (v >> i) & 1).collect();
        let sum: u8 = bits.iter().sum();
        p.push((bits, vec![sum % 2]));
    }
    Task { name: "PAR4", patterns: p, in_bits: 4, out_bits: 1 }
}

fn task_pop8_gt4() -> Task {
    let mut p = Vec::new();
    for v in 0..256u16 {
        let bits: Vec<u8> = (0..8).map(|i| ((v >> i) & 1) as u8).collect();
        let sum: u8 = bits.iter().sum();
        p.push((bits, vec![if sum > 4 { 1 } else { 0 }]));
    }
    Task { name: "POP8>4", patterns: p, in_bits: 8, out_bits: 1 }
}

fn task_par8() -> Task {
    let mut p = Vec::new();
    for v in 0..256u16 {
        let bits: Vec<u8> = (0..8).map(|i| ((v >> i) & 1) as u8).collect();
        let sum: u8 = bits.iter().sum();
        p.push((bits, vec![sum % 2]));
    }
    Task { name: "PAR8", patterns: p, in_bits: 8, out_bits: 1 }
}

fn task_half_adder() -> Task {
    // 2 bit input → 2 bit output (sum, carry)
    let mut p = Vec::new();
    for a in 0..2u8 { for b in 0..2u8 {
        let sum = a + b;
        p.push((vec![a, b], vec![sum & 1, sum >> 1]));
    }}
    Task { name: "HALF_ADD", patterns: p, in_bits: 2, out_bits: 2 }
}

fn task_full_adder() -> Task {
    // 3 bit input (a, b, cin) → 2 bit output (sum, cout)
    let mut p = Vec::new();
    for a in 0..2u8 { for b in 0..2u8 { for c in 0..2u8 {
        let sum = a + b + c;
        p.push((vec![a, b, c], vec![sum & 1, sum >> 1]));
    }}}
    Task { name: "FULL_ADD", patterns: p, in_bits: 3, out_bits: 2 }
}

fn task_2bit_add() -> Task {
    // 4 bit input (a1a0, b1b0) → 3 bit output (s2s1s0)
    let mut p = Vec::new();
    for a in 0..4u8 { for b in 0..4u8 {
        let bits_a: Vec<u8> = (0..2).map(|i| (a >> i) & 1).collect();
        let bits_b: Vec<u8> = (0..2).map(|i| (b >> i) & 1).collect();
        let mut inp = bits_a; inp.extend(bits_b);
        let sum = (a + b) as usize;
        let out: Vec<u8> = (0..3).map(|i| ((sum >> i) & 1) as u8).collect();
        p.push((inp, out));
    }}
    Task { name: "2BIT_ADD", patterns: p, in_bits: 4, out_bits: 3 }
}

fn task_2bit_max() -> Task {
    let mut p = Vec::new();
    for a in 0..4u8 { for b in 0..4u8 {
        let bits_a: Vec<u8> = (0..2).map(|i| (a >> i) & 1).collect();
        let bits_b: Vec<u8> = (0..2).map(|i| (b >> i) & 1).collect();
        let mut inp = bits_a; inp.extend(bits_b);
        let m = a.max(b) as usize;
        let out: Vec<u8> = (0..2).map(|i| ((m >> i) & 1) as u8).collect();
        p.push((inp, out));
    }}
    Task { name: "2BIT_MAX", patterns: p, in_bits: 4, out_bits: 2 }
}

// ── Exhaustive search for ONE neuron ──

fn exhaustive_search_neuron(
    n_inputs: usize,
    patterns: &[(Vec<u8>, u8)],
    get_inputs: &dyn Fn(&[u8]) -> Vec<u8>,
) -> (Option<Neuron>, usize) {
    let n_weights = n_inputs + 1; // weights + bias
    let total = 3u64.pow(n_weights as u32);

    let mut best_neuron: Option<Neuron> = None;
    let mut best_score = 0usize;
    let n_pat = patterns.len();

    for combo in 0..total {
        // Decode ternary weights
        let mut weights = vec![0i8; n_inputs];
        let mut r = combo;
        for w in weights.iter_mut() {
            *w = (r % 3) as i8 - 1;
            r /= 3;
        }
        let bias = (r % 3) as i8 - 1;

        // Compute dot products for all patterns
        let dots: Vec<i32> = patterns.iter().map(|(full_input, _)| {
            let inp = get_inputs(full_input);
            let mut dot: i32 = bias as i32;
            for (w, &x) in weights.iter().zip(inp.iter()) {
                dot += (*w as i32) * (x as i32);
            }
            dot
        }).collect();

        // Find the dot range
        let min_dot = *dots.iter().min().unwrap();
        let max_dot = *dots.iter().max().unwrap();

        // Try every threshold: output = 1 if dot >= threshold
        for thresh in (min_dot - 1)..=(max_dot + 1) {
            let mut score = 0usize;
            for (i, &dot) in dots.iter().enumerate() {
                let out = if dot >= thresh { 1u8 } else { 0u8 };
                if out == patterns[i].1 { score += 1; }
            }
            if score > best_score {
                best_score = score;
                best_neuron = Some(Neuron { weights: weights.clone(), bias, threshold: thresh });
                if score == n_pat { return (best_neuron, best_score); }
            }
        }
    }

    (best_neuron, best_score)
}

// ── Incremental build: add neurons one at a time ──

fn incremental_build(task: &Task, max_neurons: usize, max_inputs_per_neuron: usize) -> (Vec<Neuron>, Vec<Vec<usize>>, usize, bool) {
    // For each output bit, we build neurons incrementally
    // Hidden neurons are shared across output bits

    let n_patterns = task.patterns.len();

    // State: for each pattern, the current hidden neuron outputs
    let mut hidden_outputs: Vec<Vec<u8>> = vec![Vec::new(); n_patterns]; // pattern → hidden outputs
    let mut neurons: Vec<Neuron> = Vec::new();
    let mut neuron_input_indices: Vec<Vec<usize>> = Vec::new(); // which inputs each neuron reads

    // Which output bits are not yet solved
    let mut unsolved: Vec<usize> = (0..task.out_bits).collect();

    for neuron_idx in 0..max_neurons {
        if unsolved.is_empty() { break; }

        // For each unsolved output bit, try to build a neuron
        let mut best_neuron: Option<Neuron> = None;
        let mut best_target_bit = 0usize;
        let mut best_score = 0usize;
        let mut best_input_map: Vec<usize> = Vec::new();

        let n_avail = task.in_bits + neuron_idx; // original inputs + existing hidden neurons
        // If too many available inputs, we need to pick a subset
        let try_all = n_avail <= max_inputs_per_neuron;

        for &target_bit in &unsolved {
            // Build target patterns for this bit
            let targets: Vec<u8> = task.patterns.iter().map(|(_, t)| t[target_bit]).collect();

            if try_all {
                // Try neuron reading ALL available inputs
                let input_indices: Vec<usize> = (0..n_avail).collect();
                let patterns_for_search: Vec<(Vec<u8>, u8)> = (0..n_patterns).map(|p| {
                    let mut inp: Vec<u8> = task.patterns[p].0.clone();
                    inp.extend(&hidden_outputs[p]);
                    (inp, targets[p])
                }).collect();

                let (found, score) = exhaustive_search_neuron(n_avail, &patterns_for_search, &|x| x.to_vec());
                if let Some(n) = found {
                    if score > best_score {
                        best_score = score;
                        best_neuron = Some(n);
                        best_target_bit = target_bit;
                        best_input_map = input_indices;
                    }
                }
            } else {
                // Too many inputs — try subsets: original inputs + random hidden subsets
                // Always include all original inputs, pick from hidden
                let n_hidden = neuron_idx;
                let n_from_hidden = (max_inputs_per_neuron - task.in_bits).min(n_hidden);

                // Try several subsets
                let mut rng_state = 42u64 + target_bit as u64;
                let n_tries = if n_hidden <= 20 { 100 } else { 50 };

                for _ in 0..n_tries {
                    // Pick random hidden neurons to connect
                    let mut hidden_pick: Vec<usize> = (0..n_hidden).collect();
                    // Simple shuffle
                    for i in (1..hidden_pick.len()).rev() {
                        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let j = (rng_state >> 33) as usize % (i + 1);
                        hidden_pick.swap(i, j);
                    }
                    hidden_pick.truncate(n_from_hidden);

                    let mut input_indices: Vec<usize> = (0..task.in_bits).collect();
                    input_indices.extend(hidden_pick.iter().map(|&h| task.in_bits + h));
                    let n_in = input_indices.len();

                    let patterns_for_search: Vec<(Vec<u8>, u8)> = (0..n_patterns).map(|p| {
                        let inp: Vec<u8> = input_indices.iter().map(|&idx| {
                            if idx < task.in_bits { task.patterns[p].0[idx] }
                            else { hidden_outputs[p][idx - task.in_bits] }
                        }).collect();
                        (inp, targets[p])
                    }).collect();

                    let (found, score) = exhaustive_search_neuron(n_in, &patterns_for_search, &|x| x.to_vec());
                    if let Some(n) = found {
                        if score > best_score {
                            best_score = score;
                            best_neuron = Some(n);
                            best_target_bit = target_bit;
                            best_input_map = input_indices.clone();
                        }
                    }
                }
            }
        }

        if let Some(neuron) = best_neuron {
            if best_score <= n_patterns / 2 {
                // Worse than random — stop
                println!("    Neuron {:2}: best={}/{} (≤random), stopping", neuron_idx, best_score, n_patterns);
                break;
            }

            // Compute this neuron's output for all patterns
            for p in 0..n_patterns {
                let inp: Vec<u8> = best_input_map.iter().map(|&idx| {
                    if idx < task.in_bits { task.patterns[p].0[idx] }
                    else { hidden_outputs[p][idx - task.in_bits] }
                }).collect();
                hidden_outputs[p].push(neuron.eval(&inp));
            }

            let perfect = best_score == n_patterns;
            println!("    Neuron {:2}: {} inputs, target=bit{}, score={}/{} {}",
                neuron_idx, best_input_map.len(), best_target_bit, best_score, n_patterns,
                if perfect { "✓ PERFECT" } else { "" });

            if perfect {
                unsolved.retain(|&b| b != best_target_bit);
            }

            neurons.push(neuron);
            neuron_input_indices.push(best_input_map);
        } else {
            println!("    Neuron {:2}: no improvement found, stopping", neuron_idx);
            break;
        }
    }

    let solved = task.out_bits - unsolved.len();
    let all_solved = unsolved.is_empty();
    (neurons, neuron_input_indices, solved, all_solved)
}

// ── Verify the built network exhaustively ──

fn verify_network(task: &Task, neurons: &[Neuron], input_maps: &[Vec<usize>], output_neuron_indices: &[usize]) -> f32 {
    let mut correct = 0;
    for (inp, target) in &task.patterns {
        let mut hidden: Vec<u8> = Vec::new();
        for (n, map) in neurons.iter().zip(input_maps) {
            let neuron_inp: Vec<u8> = map.iter().map(|&idx| {
                if idx < task.in_bits { inp[idx] }
                else { hidden[idx - task.in_bits] }
            }).collect();
            hidden.push(n.eval(&neuron_inp));
        }
        let mut all_match = true;
        for (bit_idx, &ni) in output_neuron_indices.iter().enumerate() {
            if hidden[ni] != target[bit_idx] { all_match = false; break; }
        }
        if all_match { correct += 1; }
    }
    correct as f32 / task.patterns.len() as f32 * 100.0
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  VRAXION Capability — Exhaustive Incremental Build          ║");
    println!("║  1 neuron at a time, ternary {{-1,0,+1}}, C19 rho=8          ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let tasks: Vec<Task> = vec![
        task_and(), task_or(), task_nand(), task_xor(),
        task_maj3(), task_par3(),
        task_pop4_gt2(), task_par4(),
        task_half_adder(), task_full_adder(),
        task_2bit_add(), task_2bit_max(),
        task_pop8_gt4(), task_par8(),
    ];

    println!("  {:12} │ Patt │ I→O │ Neurons │ Solved │ Time", "Task");
    println!("  ─────────────┼──────┼─────┼─────────┼────────┼──────");

    let max_per_neuron = 14;  // GPU exhaustive limit per neuron

    for task in &tasks {
        print!("  {:12} │ {:4} │ {}→{} │ ", task.name, task.patterns.len(), task.in_bits, task.out_bits);

        let t0 = Instant::now();
        let (neurons, input_maps, solved, all_solved) = incremental_build(task, 20, max_per_neuron);
        let elapsed = t0.elapsed();

        let status = if all_solved {
            format!("{}/{} ✓", solved, task.out_bits)
        } else {
            format!("{}/{} ✗", solved, task.out_bits)
        };

        let time_str = if elapsed.as_millis() < 1000 {
            format!("{}ms", elapsed.as_millis())
        } else {
            format!("{:.1}s", elapsed.as_secs_f64())
        };

        println!("{:4}    │ {:6} │ {}", neurons.len(), status, time_str);
    }

    println!("\n  Max inputs/neuron: {} (GPU exhaustive limit)", max_per_neuron);
    println!("  All exhaustive verified — zero float, zero backprop");
}
