//! Single Neuron Atlas — every method, every task, full picture
//! 1 neuron with N inputs: what can it learn?
//!
//! Methods: analytic (if possible), backprop, hessian-aware, random search,
//!          ternary exhaustive, ternary+threshold exhaustive
//!
//! Run: cargo run --example single_neuron_atlas --release

use std::time::Instant;

// ── Activations ──

fn c19(x: f32, rho: f32) -> f32 {
    let l = 6.0f32;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * h + rho * h * h
}

fn c19d(x: f32, rho: f32) -> f32 {
    let l = 6.0f32;
    if x >= l || x <= -l { return 1.0; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    (1.0 - 2.0 * t) * (sgn + 2.0 * rho * h)
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
fn sigmoid_d(x: f32) -> f32 { let s = sigmoid(x); s * (1.0 - s) }

const RHO: f32 = 8.0;

// ── PRNG ──

struct Rng { state: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { state: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 { self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.state }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn range_f32(&mut self, lo: f32, hi: f32) -> f32 { lo + self.f32() * (hi - lo) }
}

// ── Single neuron: w·x + bias → activation → threshold ──

struct FloatNeuron {
    weights: Vec<f32>,
    bias: f32,
}

impl FloatNeuron {
    fn dot(&self, x: &[f32]) -> f32 {
        self.bias + self.weights.iter().zip(x).map(|(w, xi)| w * xi).sum::<f32>()
    }

    fn eval_sigmoid(&self, x: &[f32]) -> f32 { sigmoid(self.dot(x)) }
    fn eval_c19(&self, x: &[f32]) -> f32 { c19(self.dot(x), RHO) }
    fn eval_relu(&self, x: &[f32]) -> f32 { self.dot(x).max(0.0) }
    fn eval_linear(&self, x: &[f32]) -> f32 { self.dot(x) }

    fn classify_sigmoid(&self, x: &[f32]) -> u8 { if self.eval_sigmoid(x) > 0.5 { 1 } else { 0 } }
    fn classify_threshold(&self, x: &[f32], t: f32) -> u8 { if self.dot(x) >= t { 1 } else { 0 } }
}

// ── Tasks ──

struct Task {
    name: &'static str,
    patterns: Vec<(Vec<f32>, f32)>,  // (input, target 0.0 or 1.0)
    n_in: usize,
    linearly_separable: bool,
}

fn bits_f32(val: usize, n: usize) -> Vec<f32> {
    (0..n).map(|i| if val & (1 << i) != 0 { 1.0 } else { 0.0 }).collect()
}

fn make_task(name: &'static str, n_in: usize, f: impl Fn(usize) -> bool, lin_sep: bool) -> Task {
    let n = 1 << n_in;
    let patterns: Vec<(Vec<f32>, f32)> = (0..n).map(|v| {
        (bits_f32(v, n_in), if f(v) { 1.0 } else { 0.0 })
    }).collect();
    Task { name, patterns, n_in, linearly_separable: lin_sep }
}

fn all_tasks() -> Vec<Task> {
    vec![
        make_task("AND",     2, |v| v == 3, true),
        make_task("OR",      2, |v| v >= 1, true),
        make_task("NAND",    2, |v| v != 3, true),
        make_task("XOR",     2, |v| v == 1 || v == 2, false),
        make_task("XNOR",    2, |v| v == 0 || v == 3, false),
        make_task("MAJ3",    3, |v| (v as u8).count_ones() >= 2, true),
        make_task("PAR3",    3, |v| (v as u8).count_ones() % 2 == 1, false),
        make_task("POP4>2",  4, |v| (v as u8).count_ones() > 2, true),
        make_task("PAR4",    4, |v| (v as u8).count_ones() % 2 == 1, false),
        make_task("POP5>2",  5, |v| (v as u8).count_ones() > 2, true),
        make_task("PAR5",    5, |v| (v as u8).count_ones() % 2 == 1, false),
        make_task("POP8>4",  8, |v| (v as u16).count_ones() > 4, true),
        make_task("PAR8",    8, |v| (v as u16).count_ones() % 2 == 1, false),
        // Non-symmetric tasks
        make_task("BIT0",    4, |v| v & 1 == 1, true),         // identity of bit 0
        make_task("IMPLY",   2, |v| !(v == 2), true),          // a→b = ¬a ∨ b
        make_task("MUX",     3, |v| { let (a,b,s) = (v&1, (v>>1)&1, (v>>2)&1); if s==0 {a==1} else {b==1} }, false),
    ]
}

// ── Method 1: Backprop (sigmoid) ──

fn backprop_sigmoid(task: &Task, lr: f32, epochs: usize, seed: u64) -> (FloatNeuron, f32) {
    let mut rng = Rng::new(seed);
    let mut n = FloatNeuron {
        weights: (0..task.n_in).map(|_| rng.range_f32(-1.0, 1.0)).collect(),
        bias: rng.range_f32(-0.5, 0.5),
    };

    for _ in 0..epochs {
        for (x, y) in &task.patterns {
            let z = n.dot(x);
            let a = sigmoid(z);
            let err = a - *y;      // d_loss/d_a for BCE-ish
            let da = sigmoid_d(z); // d_a/d_z
            let grad = err * da;
            for (w, xi) in n.weights.iter_mut().zip(x) {
                *w -= lr * grad * xi;
            }
            n.bias -= lr * grad;
        }
    }

    let acc = eval_acc_sigmoid(&n, &task.patterns);
    (n, acc)
}

fn eval_acc_sigmoid(n: &FloatNeuron, patterns: &[(Vec<f32>, f32)]) -> f32 {
    let correct = patterns.iter().filter(|(x, y)| {
        n.classify_sigmoid(x) == (*y as u8)
    }).count();
    correct as f32 / patterns.len() as f32 * 100.0
}

// ── Method 2: Backprop (C19) ──

fn backprop_c19(task: &Task, lr: f32, epochs: usize, seed: u64) -> (FloatNeuron, f32) {
    let mut rng = Rng::new(seed);
    let mut n = FloatNeuron {
        weights: (0..task.n_in).map(|_| rng.range_f32(-0.5, 0.5)).collect(),
        bias: rng.range_f32(-0.3, 0.3),
    };

    for _ in 0..epochs {
        for (x, y) in &task.patterns {
            let z = n.dot(x);
            let a = c19(z, RHO);
            let err = a - *y;
            let da = c19d(z, RHO);
            let grad: f32 = err * da;
            if grad.is_nan() || grad.abs() > 10.0 { continue; }
            for (w, xi) in n.weights.iter_mut().zip(x) {
                *w -= (lr * grad * xi).clamp(-0.1, 0.1);
            }
            n.bias -= (lr * grad).clamp(-0.1, 0.1);
        }
    }

    let acc = eval_acc_c19(&n, &task.patterns);
    (n, acc)
}

fn eval_acc_c19(n: &FloatNeuron, patterns: &[(Vec<f32>, f32)]) -> f32 {
    let correct = patterns.iter().filter(|(x, y)| {
        let out = c19(n.dot(x), RHO);
        let pred = if out > 0.5 { 1 } else { 0 };
        pred == (*y as u8)
    }).count();
    correct as f32 / patterns.len() as f32 * 100.0
}

// ── Method 3: Hessian-aware (Newton-style, sigmoid) ──

fn hessian_sigmoid(task: &Task, steps: usize, seed: u64) -> (FloatNeuron, f32) {
    let mut rng = Rng::new(seed);
    let dim = task.n_in + 1; // weights + bias
    let mut params: Vec<f32> = (0..dim).map(|_| rng.range_f32(-1.0, 1.0)).collect();

    for _ in 0..steps {
        let mut grad = vec![0.0f32; dim];
        let mut hess_diag = vec![0.0f32; dim]; // diagonal Hessian approximation

        for (x, y) in &task.patterns {
            let mut z = params[dim - 1]; // bias
            for i in 0..task.n_in { z += params[i] * x[i]; }
            let a = sigmoid(z);
            let err = a - *y;
            let sd = sigmoid_d(z);

            // Gradient
            for i in 0..task.n_in {
                grad[i] += err * sd * x[i];
                hess_diag[i] += sd * sd * x[i] * x[i]; // Gauss-Newton approx
            }
            grad[dim - 1] += err * sd;
            hess_diag[dim - 1] += sd * sd;
        }

        // Newton step: params -= grad / (hess + damping)
        let damping = 0.1;
        for i in 0..dim {
            let h = hess_diag[i] + damping;
            if h > 1e-8 {
                params[i] -= (grad[i] / h).clamp(-2.0, 2.0);
            }
        }
    }

    let n = FloatNeuron {
        weights: params[..task.n_in].to_vec(),
        bias: params[task.n_in],
    };
    let acc = eval_acc_sigmoid(&n, &task.patterns);
    (n, acc)
}

// ── Method 4: Random search (float) ──

fn random_search(task: &Task, n_trials: usize, seed: u64) -> (FloatNeuron, f32) {
    let mut rng = Rng::new(seed);
    let mut best_n = FloatNeuron { weights: vec![0.0; task.n_in], bias: 0.0 };
    let mut best_acc = 0.0f32;

    for _ in 0..n_trials {
        let scale = rng.range_f32(0.5, 5.0);
        let n = FloatNeuron {
            weights: (0..task.n_in).map(|_| rng.range_f32(-scale, scale)).collect(),
            bias: rng.range_f32(-scale, scale),
        };
        let acc = eval_acc_sigmoid(&n, &task.patterns);
        if acc > best_acc {
            best_acc = acc;
            best_n = n;
            if acc >= 100.0 { break; }
        }
    }
    (best_n, best_acc)
}

// ── Method 5: Ternary exhaustive (dot >= threshold) ──

fn ternary_exhaustive(task: &Task) -> (Option<Vec<i8>>, i8, i32, f32) {
    let n = task.n_in;
    let n_weights = n + 1;
    let total = 3u64.pow(n_weights as u32);
    let n_pat = task.patterns.len();

    let mut best_w: Option<Vec<i8>> = None;
    let mut best_bias: i8 = 0;
    let mut best_thresh: i32 = 0;
    let mut best_score = 0usize;

    for combo in 0..total {
        let mut weights = vec![0i8; n];
        let mut r = combo;
        for w in weights.iter_mut() { *w = (r % 3) as i8 - 1; r /= 3; }
        let bias = (r % 3) as i8 - 1;

        // Compute dots
        let dots: Vec<i32> = task.patterns.iter().map(|(x, _)| {
            let mut d = bias as i32;
            for (w, xi) in weights.iter().zip(x) { d += (*w as i32) * (*xi as i32); }
            d
        }).collect();

        let min_d = *dots.iter().min().unwrap();
        let max_d = *dots.iter().max().unwrap();

        for thresh in (min_d - 1)..=(max_d + 1) {
            let score = dots.iter().zip(&task.patterns).filter(|(&d, (_, y))| {
                let pred = if d >= thresh { 1.0 } else { 0.0 };
                (pred - *y).abs() < 0.01
            }).count();

            if score > best_score {
                best_score = score;
                best_w = Some(weights.clone());
                best_bias = bias;
                best_thresh = thresh;
                if score == n_pat { return (best_w, best_bias, best_thresh, 100.0); }
            }
        }
    }

    let acc = best_score as f32 / n_pat as f32 * 100.0;
    (best_w, best_bias, best_thresh, acc)
}

// ── Method 6: Float→Ternary rounding ──

fn float_to_ternary_round(n: &FloatNeuron, task: &Task) -> f32 {
    // Round float weights to nearest ternary, find best threshold
    let tw: Vec<i8> = n.weights.iter().map(|&w| {
        if w > 0.33 { 1 } else if w < -0.33 { -1 } else { 0 }
    }).collect();
    let tb = if n.bias > 0.33 { 1i8 } else if n.bias < -0.33 { -1 } else { 0 };

    let dots: Vec<i32> = task.patterns.iter().map(|(x, _)| {
        let mut d = tb as i32;
        for (w, xi) in tw.iter().zip(x) { d += (*w as i32) * (*xi as i32); }
        d
    }).collect();

    let min_d = *dots.iter().min().unwrap();
    let max_d = *dots.iter().max().unwrap();
    let n_pat = task.patterns.len();

    let mut best_score = 0usize;
    for thresh in (min_d - 1)..=(max_d + 1) {
        let score = dots.iter().zip(&task.patterns).filter(|(&d, (_, y))| {
            let pred = if d >= thresh { 1.0 } else { 0.0 };
            (pred - *y).abs() < 0.01
        }).count();
        if score > best_score { best_score = score; }
    }
    best_score as f32 / n_pat as f32 * 100.0
}

// ── Run all methods on one task ──

struct TaskResult {
    backprop_sigmoid: f32,
    backprop_c19: f32,
    hessian: f32,
    random: f32,
    ternary_exhaustive: f32,
    float_to_ternary: f32,
    best_float: f32,
}

fn run_all_methods(task: &Task) -> TaskResult {
    // Multi-seed for stochastic methods, take best
    let seeds = [42, 123, 7, 999, 2024, 314, 55, 808];

    let mut best_bp_sig = 0.0f32;
    let mut best_bp_c19 = 0.0f32;
    let mut best_hess = 0.0f32;
    let mut best_rand = 0.0f32;
    let mut best_float_neuron: Option<FloatNeuron> = None;
    let mut best_float_score = 0.0f32;

    for &seed in &seeds {
        let (n1, a1) = backprop_sigmoid(task, 0.5, 5000, seed);
        if a1 > best_bp_sig { best_bp_sig = a1; }
        if a1 > best_float_score { best_float_score = a1; best_float_neuron = Some(n1); }

        let (n2, a2) = backprop_c19(task, 0.01, 5000, seed);
        if a2 > best_bp_c19 { best_bp_c19 = a2; }

        let (n3, a3) = hessian_sigmoid(task, 500, seed);
        if a3 > best_hess { best_hess = a3; }
        if a3 > best_float_score { best_float_score = a3; best_float_neuron = Some(n3); }

        let (n4, a4) = random_search(task, 50000, seed);
        if a4 > best_rand { best_rand = a4; }
        if a4 > best_float_score { best_float_score = a4; best_float_neuron = Some(n4); }
    }

    // Ternary exhaustive (deterministic)
    let (_, _, _, tern_acc) = ternary_exhaustive(task);

    // Float→ternary rounding
    let f2t = if let Some(ref n) = best_float_neuron {
        float_to_ternary_round(n, task)
    } else { 0.0 };

    TaskResult {
        backprop_sigmoid: best_bp_sig,
        backprop_c19: best_bp_c19,
        hessian: best_hess,
        random: best_rand,
        ternary_exhaustive: tern_acc,
        float_to_ternary: f2t,
        best_float: best_float_score,
    }
}

fn main() {
    let t0 = Instant::now();

    println!("╔════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  SINGLE NEURON ATLAS — 1 neuron, every method, every task                                   ║");
    println!("║  8 seeds × 4 float methods + ternary exhaustive + float→ternary round                      ║");
    println!("╠════════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Task     │ LinSep │ BP_sig │ BP_c19 │ Hessian│ Random │ Best_F │ Tern_Ex│ F→T    │ Verdict  ║");
    println!("╠══════════╪════════╪════════╪════════╪════════╪════════╪════════╪════════╪════════╪══════════╣");

    let tasks = all_tasks();
    let mut n_perfect_float = 0;
    let mut n_perfect_tern = 0;

    for task in &tasks {
        let r = run_all_methods(task);

        let verdict = if r.ternary_exhaustive >= 100.0 { "TERN ✓" }
            else if r.best_float >= 100.0 { "FLOAT only" }
            else { "IMPOSSIBLE" };

        if r.best_float >= 100.0 { n_perfect_float += 1; }
        if r.ternary_exhaustive >= 100.0 { n_perfect_tern += 1; }

        println!("║ {:8} │   {}    │ {:5.1}% │ {:5.1}% │ {:5.1}% │ {:5.1}% │ {:5.1}% │ {:5.1}% │ {:5.1}% │ {:8} ║",
            task.name,
            if task.linearly_separable { "Y" } else { "N" },
            r.backprop_sigmoid, r.backprop_c19, r.hessian, r.random,
            r.best_float, r.ternary_exhaustive, r.float_to_ternary,
            verdict,
        );
    }

    println!("╠══════════╧════════╧════════╧════════╧════════╧════════╧════════╧════════╧════════╧══════════╣");
    println!("║  Float perfect: {}/{}  │  Ternary perfect: {}/{}  │  Time: {:.1}s                              ║",
        n_perfect_float, tasks.len(), n_perfect_tern, tasks.len(), t0.elapsed().as_secs_f64());
    println!("╠════════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  BP_sig  = backprop sigmoid (lr=0.5, 5000ep)     Hessian = Newton-step diagonal (500 steps) ║");
    println!("║  BP_c19  = backprop C19 rho=8 (lr=0.01, 5000ep)  Random  = 50K random float weights        ║");
    println!("║  Best_F  = best float across all methods          Tern_Ex = ternary exhaustive + threshold  ║");
    println!("║  F→T     = best float rounded to ternary          LinSep  = linearly separable (theory)     ║");
    println!("╚════════════════════════════════════════════════════════════════════════════════════════════════╝");
}
