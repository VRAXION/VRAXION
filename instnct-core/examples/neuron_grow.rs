//! Neuron Grow — incremental build with landscape-guided search
//! Add 1 neuron at a time, max it out, freeze, add next.
//! Outputs JSON for the live HTML playground.
//!
//! Run: cargo run --example neuron_grow --release

use std::time::Instant;
use std::io::Write as IoWrite;

// ── PRNG ──
struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 { self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.s }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn range(&mut self, lo: f32, hi: f32) -> f32 { lo + self.f32() * (hi - lo) }
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

// ── Tasks ──

struct Task {
    name: &'static str,
    n_in: usize,
    data: Vec<(Vec<f32>, u8)>,
}

fn bits(v: usize, n: usize) -> Vec<f32> {
    (0..n).map(|i| if v & (1 << i) != 0 { 1.0 } else { 0.0 }).collect()
}

fn task_xor() -> Task {
    let data = (0..4).map(|v| {
        (bits(v, 2), (((v & 1) ^ ((v >> 1) & 1)) as u8))
    }).collect();
    Task { name: "XOR", n_in: 2, data }
}

fn task_maj6() -> Task {
    let data = (0..64).map(|v: usize| {
        let b = bits(v, 6);
        let s: f32 = b.iter().sum();
        (b, if s > 3.0 { 1u8 } else { 0 })
    }).collect();
    Task { name: "MAJORITY6", n_in: 6, data }
}

fn task_parity4() -> Task {
    let data = (0..16).map(|v: usize| {
        let b = bits(v, 4);
        let s: u8 = (0..4).map(|i| ((v >> i) & 1) as u8).sum();
        (b, s % 2)
    }).collect();
    Task { name: "PARITY4", n_in: 4, data }
}

fn task_parity6() -> Task {
    let data = (0..64).map(|v: usize| {
        let b = bits(v, 6);
        let s: u8 = (0..6).map(|i| ((v >> i) & 1) as u8).sum();
        (b, s % 2)
    }).collect();
    Task { name: "PARITY6", n_in: 6, data }
}

fn task_compare3() -> Task {
    let mut data = Vec::new();
    for a in 0..8u8 { for b in 0..8u8 {
        let mut inp = bits(a as usize, 3);
        inp.extend(bits(b as usize, 3));
        data.push((inp, if a > b { 1u8 } else { 0 }));
    }}
    Task { name: "COMPARE3", n_in: 6, data }
}

fn task_has110() -> Task {
    let data = (0..64).map(|v: usize| {
        let b = bits(v, 6);
        let mut has = false;
        for i in 0..4 { if b[i] > 0.5 && b[i+1] > 0.5 && b[i+2] < 0.5 { has = true; }}
        (b, if has { 1u8 } else { 0 })
    }).collect();
    Task { name: "HAS_110", n_in: 6, data }
}

fn task_symmetric4() -> Task {
    let data = (0..16).map(|v: usize| {
        let b = bits(v, 4);
        let sym = (b[0] == b[3]) && (b[1] == b[2]);
        (b, if sym { 1u8 } else { 0 })
    }).collect();
    Task { name: "SYMM4", n_in: 4, data }
}

// ── Frozen neuron ──

#[derive(Clone)]
struct FrozenNeuron {
    weights: Vec<i8>,   // over original + hidden inputs
    bias: i8,
    threshold: i32,
    input_map: Vec<usize>,  // which inputs this neuron reads
}

impl FrozenNeuron {
    fn eval(&self, all_values: &[u8]) -> u8 {
        let mut dot = self.bias as i32;
        for (&wi, &idx) in self.weights.iter().zip(&self.input_map) {
            dot += (wi as i32) * (all_values[idx] as i32);
        }
        if dot >= self.threshold { 1 } else { 0 }
    }
}

// ── Pipeline state ──

struct GrowState {
    neurons: Vec<FrozenNeuron>,
    // For each pattern: [original_inputs..., hidden_neuron_outputs...]
    pattern_values: Vec<Vec<u8>>,
    n_original: usize,
}

impl GrowState {
    fn new(task: &Task) -> Self {
        let pattern_values = task.data.iter().map(|(x, _)| {
            x.iter().map(|&v| if v > 0.5 { 1u8 } else { 0 }).collect()
        }).collect();
        GrowState { neurons: Vec::new(), pattern_values, n_original: task.n_in }
    }

    fn n_available(&self) -> usize {
        self.n_original + self.neurons.len()
    }

    fn eval_accuracy(&self, task: &Task, output_neuron: usize) -> f32 {
        let correct = task.data.iter().enumerate().filter(|(i, (_, y))| {
            self.pattern_values[*i][self.n_original + output_neuron] == *y
        }).count();
        correct as f32 / task.data.len() as f32 * 100.0
    }

    fn current_best_accuracy(&self, task: &Task) -> f32 {
        if self.neurons.is_empty() { return 0.0; }
        // Try each neuron as output — pick best
        let mut best = 0.0f32;
        for ni in 0..self.neurons.len() {
            let acc = self.eval_accuracy(task, ni);
            if acc > best { best = acc; }
        }
        best
    }

    fn best_output_neuron(&self, task: &Task) -> usize {
        let mut best_acc = 0.0f32;
        let mut best_ni = 0;
        for ni in 0..self.neurons.len() {
            let acc = self.eval_accuracy(task, ni);
            if acc > best_acc { best_acc = acc; best_ni = ni; }
        }
        best_ni
    }
}

// ── Float landscape for 1 neuron given current state ──

fn float_landscape(
    task: &Task,
    state: &GrowState,
    input_indices: &[usize],
    n_seeds: usize,
) -> Vec<[usize; 3]> {
    // Returns sign distribution per weight position
    let n_in = input_indices.len();
    let n_pat = task.data.len();
    let mut sign_counts = vec![[0usize; 3]; n_in]; // [neg, zero, pos]

    for seed in 0..n_seeds {
        let mut rng = Rng::new(seed as u64 * 137 + 42);
        let mut w: Vec<f32> = (0..n_in).map(|_| rng.range(-2.0, 2.0)).collect();
        let mut b = rng.range(-1.0, 1.0);

        // Train sigmoid neuron
        for _ in 0..3000 {
            for (pi, (_, y)) in task.data.iter().enumerate() {
                let vals = &state.pattern_values[pi];
                let z: f32 = b + (0..n_in).map(|i| w[i] * vals[input_indices[i]] as f32).sum::<f32>();
                let a = sigmoid(z);
                let err = a - *y as f32;
                let sd = a * (1.0 - a);
                let g = err * sd;
                for i in 0..n_in { w[i] -= 1.0 * g * vals[input_indices[i]] as f32; }
                b -= 1.0 * g;
            }
        }

        // Check accuracy
        let correct = task.data.iter().enumerate().filter(|(pi, (_, y))| {
            let vals = &state.pattern_values[*pi];
            let z: f32 = b + (0..n_in).map(|i| w[i] * vals[input_indices[i]] as f32).sum::<f32>();
            (if sigmoid(z) > 0.5 { 1u8 } else { 0 }) == *y
        }).count();

        let acc = correct as f32 / n_pat as f32;
        if acc < 0.55 { continue; } // skip bad solutions

        for i in 0..n_in {
            if w[i] < -0.3 { sign_counts[i][0] += 1; }
            else if w[i] > 0.3 { sign_counts[i][2] += 1; }
            else { sign_counts[i][1] += 1; }
        }
    }

    sign_counts
}

// ── Exhaustive search for 1 new neuron ──

fn search_neuron(
    task: &Task,
    state: &GrowState,
    input_indices: &[usize],
    sign_hints: &[[usize; 3]],
) -> (Vec<i8>, i8, i32, f32) {
    let n_in = input_indices.len();
    let n_pat = task.data.len();

    // Determine locked vs free
    let mut locked: Vec<Option<i8>> = vec![None; n_in];
    let mut free_pos = Vec::new();

    for i in 0..n_in {
        let total = (sign_hints[i][0] + sign_hints[i][1] + sign_hints[i][2]).max(1);
        let thresh = (total as f32 * 0.7) as usize;
        if sign_hints[i][2] >= thresh { locked[i] = Some(1); }
        else if sign_hints[i][0] >= thresh { locked[i] = Some(-1); }
        else if sign_hints[i][1] >= thresh { locked[i] = Some(0); }
        else { free_pos.push(i); }
    }

    // If too many free, fall back to full exhaustive
    let use_guided = free_pos.len() <= 10 && locked.iter().any(|l| l.is_some());

    let mut best_w = vec![0i8; n_in];
    let mut best_b: i8 = 0;
    let mut best_t: i32 = 0;
    let mut best_score = 0usize;

    if use_guided {
        // Guided: iterate free positions only
        let n_free = free_pos.len();
        let total_free = 3u32.pow(n_free as u32);

        for b_try in [-1i8, 0, 1] {
            for combo in 0..total_free {
                let mut w = vec![0i8; n_in];
                for i in 0..n_in { w[i] = locked[i].unwrap_or(0); }
                let mut r = combo;
                for &fp in &free_pos { w[fp] = (r % 3) as i8 - 1; r /= 3; }

                let dots: Vec<i32> = (0..n_pat).map(|pi| {
                    let vals = &state.pattern_values[pi];
                    let mut d = b_try as i32;
                    for (wi, &idx) in w.iter().zip(input_indices) { d += (*wi as i32) * (vals[idx] as i32); }
                    d
                }).collect();

                let min_d = dots.iter().copied().min().unwrap_or(0);
                let max_d = dots.iter().copied().max().unwrap_or(0);
                for thresh in (min_d-1)..=(max_d+1) {
                    let score = dots.iter().zip(&task.data).filter(|(&d, (_, y))| {
                        (if d >= thresh { 1u8 } else { 0 }) == *y
                    }).count();
                    if score > best_score {
                        best_score = score; best_w = w.clone(); best_b = b_try; best_t = thresh;
                        if score == n_pat { return (best_w, best_b, best_t, 100.0); }
                    }
                }
            }
        }
    }

    // Also try blind exhaustive (always, to not miss anything)
    let total = 3u64.pow((n_in + 1) as u32);
    for combo in 0..total {
        let mut w = vec![0i8; n_in];
        let mut r = combo;
        for wi in w.iter_mut() { *wi = (r % 3) as i8 - 1; r /= 3; }
        let b = (r % 3) as i8 - 1;

        let dots: Vec<i32> = (0..n_pat).map(|pi| {
            let vals = &state.pattern_values[pi];
            let mut d = b as i32;
            for (wi, &idx) in w.iter().zip(input_indices) { d += (*wi as i32) * (vals[idx] as i32); }
            d
        }).collect();
        let min_d = dots.iter().copied().min().unwrap_or(0);
        let max_d = dots.iter().copied().max().unwrap_or(0);
        for thresh in (min_d-1)..=(max_d+1) {
            let score = dots.iter().zip(&task.data).filter(|(&d, (_, y))| {
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

// ── Grow: add neurons one by one ──

fn grow(task: &Task, max_neurons: usize, max_inputs_per_neuron: usize) -> Vec<(FrozenNeuron, f32, Vec<[usize;3]>)> {
    let mut state = GrowState::new(task);
    let mut history: Vec<(FrozenNeuron, f32, Vec<[usize;3]>)> = Vec::new();

    println!("  Starting grow: {} inputs, {} patterns, max {} neurons\n",
        task.n_in, task.data.len(), max_neurons);

    for ni in 0..max_neurons {
        let n_avail = state.n_available();
        let n_in = n_avail.min(max_inputs_per_neuron);
        let input_indices: Vec<usize> = (0..n_in).collect();

        // Phase 1: Float landscape
        let t0 = Instant::now();
        let signs = float_landscape(task, &state, &input_indices, 100);
        let landscape_ms = t0.elapsed().as_millis();

        // Show landscape
        let locked_count = signs.iter().filter(|s| {
            let tot = (s[0]+s[1]+s[2]).max(1);
            s[0]*10/tot >= 7 || s[1]*10/tot >= 7 || s[2]*10/tot >= 7
        }).count();
        let free_count = n_in - locked_count;

        print!("  Neuron {:2}: landscape({}ms) ", ni, landscape_ms);
        for (_i, s) in signs.iter().enumerate() {
            let tot = (s[0]+s[1]+s[2]).max(1);
            let ch = if s[2]*10/tot >= 7 { '+' } else if s[0]*10/tot >= 7 { '-' } else if s[1]*10/tot >= 7 { '.' } else { '?' };
            print!("{}", ch);
        }
        print!(" (L:{} F:{}) ", locked_count, free_count);

        // Phase 2: Search
        let t1 = Instant::now();
        let (w, b, thresh, acc) = search_neuron(task, &state, &input_indices, &signs);
        let search_ms = t1.elapsed().as_millis();

        let w_str: String = w.iter().map(|&v| match v { 1=>"+", -1=>"-", _=>"0"}).collect::<Vec<_>>().join("");
        println!("→ [{}] b={:+} t={} acc={:.1}% ({}ms)",
            w_str, b, thresh, acc, search_ms);

        // Freeze this neuron
        let neuron = FrozenNeuron {
            weights: w.clone(), bias: b, threshold: thresh,
            input_map: input_indices.clone(),
        };

        // Compute outputs for all patterns
        for pi in 0..task.data.len() {
            let out = neuron.eval(&state.pattern_values[pi]);
            state.pattern_values[pi].push(out);
        }
        state.neurons.push(neuron.clone());

        let current_acc = state.current_best_accuracy(task);
        history.push((neuron, current_acc, signs));

        println!("           best output neuron: N{} → {:.1}%\n",
            state.best_output_neuron(task), current_acc);

        if current_acc >= 100.0 {
            println!("  ✓ SOLVED with {} neurons!\n", ni + 1);
            break;
        }

        if ni > 0 && acc <= 50.0 {
            println!("  ✗ No useful neuron found, stopping.\n");
            break;
        }
    }

    history
}

// ── Write JSON for playground ──

fn write_json(task: &Task, history: &[(FrozenNeuron, f32, Vec<[usize;3]>)], path: &str) {
    let mut f = std::fs::File::create(path).unwrap();
    write!(f, "{{\n").unwrap();
    write!(f, "  \"task\": \"{}\",\n", task.name).unwrap();
    write!(f, "  \"n_inputs\": {},\n", task.n_in).unwrap();
    write!(f, "  \"n_patterns\": {},\n", task.data.len()).unwrap();
    write!(f, "  \"neurons\": [\n").unwrap();

    for (i, (neuron, acc, signs)) in history.iter().enumerate() {
        let w_json: Vec<String> = neuron.weights.iter().map(|w| w.to_string()).collect();
        let map_json: Vec<String> = neuron.input_map.iter().map(|m| m.to_string()).collect();
        let signs_json: Vec<String> = signs.iter().map(|s| format!("[{},{},{}]", s[0], s[1], s[2])).collect();

        write!(f, "    {{\n").unwrap();
        write!(f, "      \"id\": {},\n", i).unwrap();
        write!(f, "      \"weights\": [{}],\n", w_json.join(",")).unwrap();
        write!(f, "      \"bias\": {},\n", neuron.bias).unwrap();
        write!(f, "      \"threshold\": {},\n", neuron.threshold).unwrap();
        write!(f, "      \"input_map\": [{}],\n", map_json.join(",")).unwrap();
        write!(f, "      \"accuracy\": {:.1},\n", acc).unwrap();
        write!(f, "      \"landscape\": [{}]\n", signs_json.join(",")).unwrap();
        write!(f, "    }}{}\n", if i < history.len()-1 { "," } else { "" }).unwrap();
    }

    // Also write patterns
    write!(f, "  ],\n  \"patterns\": [\n").unwrap();
    for (i, (inp, target)) in task.data.iter().enumerate() {
        let inp_json: Vec<String> = inp.iter().map(|v| if *v > 0.5 { "1".into() } else { "0".into() }).collect();
        write!(f, "    {{\"in\": [{}], \"target\": {}}}{}\n",
            inp_json.join(","), target,
            if i < task.data.len()-1 { "," } else { "" }).unwrap();
    }
    write!(f, "  ]\n}}\n").unwrap();
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Neuron Grow — Incremental Build + Landscape Guide         ║");
    println!("║  Float landscape → ternary search → freeze → add next     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let tasks: Vec<Task> = vec![
        task_xor(),
        task_parity4(),
        task_symmetric4(),
        task_maj6(),
        task_compare3(),
        task_has110(),
        task_parity6(),
    ];

    let max_neurons = 12;
    let max_fan_in = 12;

    let mut all_results: Vec<(&str, usize, f32)> = Vec::new();

    for task in &tasks {
        println!("══ {} ══", task.name);
        let history = grow(task, max_neurons, max_fan_in);

        let final_acc = history.last().map(|(_, a, _)| *a).unwrap_or(0.0);
        let n_neurons = history.len();
        all_results.push((task.name, n_neurons, final_acc));

        // Write JSON
        let json_path = format!("grow_{}.json", task.name.to_lowercase());
        write_json(task, &history, &json_path);
        println!("  JSON → {}\n", json_path);
    }

    // Summary table
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  SUMMARY                                                   ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Task          │ Neurons │ Accuracy │ Status               ║");
    println!("╠════════════════╪═════════╪══════════╪══════════════════════╣");
    for (name, n, acc) in &all_results {
        let status = if *acc >= 100.0 { "SOLVED ✓" } else if *acc >= 90.0 { "CLOSE ~" } else { "PARTIAL" };
        println!("║  {:13} │   {:3}   │  {:5.1}%  │ {:20} ║", name, n, acc, status);
    }
    println!("╚══════════════════════════════════════════════════════════════╝");
}
