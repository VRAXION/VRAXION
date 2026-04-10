//! Connectome Architecture Sweep: what's the best inter-cluster wiring?
//!
//! Tests 4 architectures for connecting local neuron clusters:
//!   A) Passive connectome — relay neurons that just sum (no activation)
//!   B) Active connectome — relay neurons WITH activation (ReLU)
//!   C) Connectome-only — all long-range info goes through connectome
//!   D) Connectome + sparse random — connectome + a few direct long-range edges
//!
//! Each architecture: incremental neuron-by-neuron build, exhaustive search
//! per neuron (ternary weights), freeze after solving.
//!
//! Tasks tested: ADD (linear), MUL (bilinear), MAX (threshold)
//!
//! Run: cargo run --example connectome_sweep --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const DIGITS: usize = 5;  // 0..4
const INPUT_DIM: usize = 8; // thermo_a(4) + thermo_b(4)
const LOCAL_CAP: usize = 3; // max local neighbors per neuron
const TICKS: usize = 2;     // recurrent ticks
const SEEDS: &[u64] = &[42, 123, 777, 314, 999, 1337, 2024, 55];  // 8 seeds

fn relu(x: f32) -> f32 { x.max(0.0) }

fn thermo_2(a: usize, b: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; INPUT_DIM];
    for i in 0..a.min(4) { v[i] = 1.0; }
    for i in 0..b.min(4) { v[4 + i] = 1.0; }
    v
}

// ============================================================
// Nearest-mean readout
// ============================================================
struct NearestMean { centroids: Vec<f32> }
impl NearestMean {
    fn fit(examples: &[(f32, usize)], n_classes: usize) -> Self {
        let mut sums = vec![0.0f32; n_classes];
        let mut counts = vec![0usize; n_classes];
        for &(s, c) in examples { sums[c] += s; counts[c] += 1; }
        NearestMean {
            centroids: (0..n_classes).map(|c| {
                if counts[c] > 0 { sums[c] / counts[c] as f32 } else { f32::NAN }
            }).collect()
        }
    }
    fn predict(&self, s: f32) -> usize {
        self.centroids.iter().enumerate()
            .filter(|(_, c)| !c.is_nan())
            .min_by(|a, b| (a.1 - s).abs().partial_cmp(&(b.1 - s).abs()).unwrap())
            .map(|(i, _)| i).unwrap_or(0)
    }
}

// ============================================================
// Connectome Network
// ============================================================

#[derive(Clone, Debug)]
struct Neuron {
    w_input: Vec<f32>,         // INPUT_DIM weights from external input
    w_local: Vec<f32>,         // weights from local neighbors
    local_indices: Vec<usize>, // which neurons are local neighbors
    w_conn_read: Vec<f32>,     // weights for reading from connectome neurons
    w_conn_write: f32,         // how much this neuron writes to its assigned connectome neuron
    conn_write_idx: usize,     // which connectome neuron it writes to
    w_sparse: Vec<f32>,        // weights for sparse random long-range connections
    sparse_indices: Vec<usize>,// which neurons are sparse targets
    bias: f32,
    frozen: bool,
    is_connectome: bool,       // is this a connectome relay neuron?
}

#[derive(Clone)]
struct ConnectomeNet {
    neurons: Vec<Neuron>,
    activations: Vec<f32>,
    n_connectome: usize,       // first N neurons are connectome
    use_connectome_activation: bool, // passive vs active
    use_sparse: bool,          // whether sparse random connections exist
}

impl ConnectomeNet {
    fn new(n_connectome: usize, active: bool, sparse: bool) -> Self {
        let mut net = ConnectomeNet {
            neurons: Vec::new(),
            activations: Vec::new(),
            n_connectome,
            use_connectome_activation: active,
            use_sparse: sparse,
        };
        // Create connectome neurons (no input weights, no local, just relay)
        for i in 0..n_connectome {
            net.neurons.push(Neuron {
                w_input: vec![0.0; INPUT_DIM],
                w_local: vec![],
                local_indices: vec![],
                w_conn_read: vec![],        // connectome neurons don't read other connectome
                w_conn_write: 0.0,
                conn_write_idx: 0,
                w_sparse: vec![],
                sparse_indices: vec![],
                bias: 0.0,
                frozen: false,
                is_connectome: true,
            });
            net.activations.push(0.0);
            let _ = i;
        }
        net
    }

    fn n_workers(&self) -> usize {
        self.neurons.len() - self.n_connectome
    }

    /// Add a worker neuron with given params.
    /// local_indices: indices of local neighbor neurons (workers only)
    /// sparse_indices: indices of random long-range neurons (workers only)
    fn add_worker(
        &mut self,
        w_input: Vec<f32>,
        w_local: Vec<f32>,
        local_indices: Vec<usize>,
        w_conn_read: Vec<f32>,
        w_conn_write: f32,
        conn_write_idx: usize,
        w_sparse: Vec<f32>,
        sparse_indices: Vec<usize>,
        bias: f32,
    ) {
        self.neurons.push(Neuron {
            w_input, w_local, local_indices,
            w_conn_read, w_conn_write,
            conn_write_idx: conn_write_idx % self.n_connectome.max(1),
            w_sparse, sparse_indices,
            bias,
            frozen: false,
            is_connectome: false,
        });
        self.activations.push(0.0);
    }

    fn freeze_last(&mut self) {
        if let Some(n) = self.neurons.last_mut() { n.frozen = true; }
    }

    fn reset(&mut self) {
        for a in &mut self.activations { *a = 0.0; }
    }

    /// Run ticks. Each tick:
    ///  1. Workers write to their connectome neuron (accumulate)
    ///  2. Connectome neurons optionally apply activation
    ///  3. All neurons compute new activations
    fn tick(&mut self, input: &[f32]) {
        let n = self.neurons.len();
        let nc = self.n_connectome;

        // Phase 1: Accumulate connectome charges from workers
        let mut conn_charges = vec![0.0f32; nc];
        for i in nc..n {
            let neuron = &self.neurons[i];
            let idx = neuron.conn_write_idx;
            if idx < nc {
                conn_charges[idx] += self.activations[i] * neuron.w_conn_write;
            }
        }

        // Phase 2: Set connectome activations
        for i in 0..nc {
            if self.use_connectome_activation {
                // Active: apply ReLU to accumulated charge + own input weights
                let mut sum = self.neurons[i].bias + conn_charges[i];
                for (j, &w) in self.neurons[i].w_input.iter().enumerate() {
                    if j < input.len() { sum += input[j] * w; }
                }
                self.activations[i] = relu(sum);
            } else {
                // Passive: just relay the sum, no activation
                self.activations[i] = conn_charges[i];
            }
        }

        // Phase 3: Compute worker activations
        let old_act = self.activations.clone();
        for i in nc..n {
            let neuron = &self.neurons[i];
            let mut sum = neuron.bias;

            // Input
            for (j, &w) in neuron.w_input.iter().enumerate() {
                if j < input.len() { sum += input[j] * w; }
            }

            // Local neighbors
            for (k, &idx) in neuron.local_indices.iter().enumerate() {
                if idx < n && k < neuron.w_local.len() {
                    sum += old_act[idx] * neuron.w_local[k];
                }
            }

            // Read from connectome
            for (k, &w) in neuron.w_conn_read.iter().enumerate() {
                if k < nc { sum += old_act[k] * w; }
            }

            // Sparse long-range
            for (k, &idx) in neuron.sparse_indices.iter().enumerate() {
                if idx < n && k < neuron.w_sparse.len() {
                    sum += old_act[idx] * neuron.w_sparse[k];
                }
            }

            self.activations[i] = relu(sum);
        }
    }

    /// Run on 2-digit input, return sum of worker activations
    fn eval_pair(&mut self, a: usize, b: usize) -> f32 {
        self.reset();
        let input = thermo_2(a, b);
        for _ in 0..TICKS {
            self.tick(&input);
        }
        // Sum of worker activations only
        self.activations[self.n_connectome..].iter().sum()
    }

    /// Count params for next worker neuron
    fn next_worker_params(&self) -> usize {
        let n_local = LOCAL_CAP.min(self.n_workers());
        let n_sparse = if self.use_sparse { 2 } else { 0 }; // 2 random long-range
        INPUT_DIM + n_local + self.n_connectome + 1 + n_sparse + 1
        // w_input + w_local + w_conn_read + w_conn_write + w_sparse + bias
    }
}

// ============================================================
// Task definitions
// ============================================================
struct Task {
    name: &'static str,
    op: fn(usize, usize) -> usize,
    n_classes: usize,
}

fn op_add(a: usize, b: usize) -> usize { a + b }
fn op_mul(a: usize, b: usize) -> usize { a * b }
fn op_max(a: usize, b: usize) -> usize { a.max(b) }
fn op_min(a: usize, b: usize) -> usize { a.min(b) }
fn op_sub_abs(a: usize, b: usize) -> usize { if a > b { a - b } else { b - a } }

fn make_tasks() -> Vec<Task> {
    vec![
        Task { name: "ADD", op: op_add, n_classes: 9 },
        Task { name: "MUL", op: op_mul, n_classes: 17 },
        Task { name: "MAX", op: op_max, n_classes: 5 },
        Task { name: "MIN", op: op_min, n_classes: 5 },
        Task { name: "SUB_ABS", op: op_sub_abs, n_classes: 5 },
    ]
}

// ============================================================
// Evaluate accuracy on all 25 pairs
// ============================================================
fn eval_accuracy(net: &mut ConnectomeNet, op: &fn(usize, usize) -> usize, n_classes: usize) -> f64 {
    let mut examples = Vec::with_capacity(25);
    for a in 0..DIGITS {
        for b in 0..DIGITS {
            let target = op(a, b);
            let charge = net.eval_pair(a, b);
            if charge.is_nan() || charge.is_infinite() { return 0.0; }
            examples.push((charge, target));
        }
    }
    let readout = NearestMean::fit(&examples, n_classes);
    examples.iter().filter(|&&(s, t)| readout.predict(s) == t).count() as f64 / 25.0
}

// ============================================================
// Incremental build: add neurons one by one, exhaustive/random search
// ============================================================
fn incremental_build(
    n_connectome: usize,
    active_connectome: bool,
    use_sparse: bool,
    max_workers: usize,
    op: &fn(usize, usize) -> usize,
    n_classes: usize,
    seed: u64,
) -> (f64, usize) {
    let ternary: Vec<f32> = vec![-1.0, 0.0, 1.0];
    let mut rng = StdRng::seed_from_u64(seed);

    let mut best_net = ConnectomeNet::new(n_connectome, active_connectome, use_sparse);
    let mut best_acc = eval_accuracy(&mut best_net, op, n_classes);

    for worker_i in 0..max_workers {
        let n_local = LOCAL_CAP.min(best_net.n_workers());
        let n_sparse = if use_sparse { 2.min(best_net.n_workers()) } else { 0 };
        let n_params = INPUT_DIM + n_local + n_connectome + 1 + n_sparse + 1;
        let total_configs = 3u64.saturating_pow(n_params as u32);

        // Pick local indices (last LOCAL_CAP workers)
        let worker_start = n_connectome;
        let total_workers = best_net.n_workers();
        let local_indices: Vec<usize> = if total_workers == 0 { vec![] }
            else {
                let start = worker_start + total_workers.saturating_sub(n_local);
                (start..worker_start + total_workers).collect()
            };

        // Pick sparse indices (random workers, not self, not local)
        let sparse_indices: Vec<usize> = if n_sparse > 0 && total_workers > n_local {
            let available: Vec<usize> = (worker_start..worker_start + total_workers)
                .filter(|i| !local_indices.contains(i))
                .collect();
            if available.is_empty() { vec![] }
            else {
                (0..n_sparse).map(|_| available[rng.gen_range(0..available.len())]).collect()
            }
        } else { vec![] };

        let actual_sparse = sparse_indices.len();

        // Search: exhaustive if small enough, random otherwise
        let use_exhaustive = total_configs <= 2_000_000;
        let sample_count = if use_exhaustive { total_configs } else { 2_000_000 };

        let mut worker_best_acc = best_acc;
        let mut worker_best_params: Option<(Vec<f32>, Vec<f32>, Vec<f32>, f32, usize, Vec<f32>, f32)> = None;

        for sample in 0..sample_count {
            let mut c = if use_exhaustive { sample } else { rng.gen_range(0..total_configs) };

            // Decode params from config index
            let w_input: Vec<f32> = (0..INPUT_DIM).map(|_| { let v = ternary[(c % 3) as usize]; c /= 3; v }).collect();
            let w_local: Vec<f32> = (0..n_local).map(|_| { let v = ternary[(c % 3) as usize]; c /= 3; v }).collect();
            let w_conn_read: Vec<f32> = (0..n_connectome).map(|_| { let v = ternary[(c % 3) as usize]; c /= 3; v }).collect();
            let w_conn_write = ternary[(c % 3) as usize]; c /= 3;
            let w_sparse: Vec<f32> = (0..actual_sparse).map(|_| { let v = ternary[(c % 3) as usize]; c /= 3; v }).collect();
            let bias = ternary[(c % 3) as usize];

            let conn_write_idx = if n_connectome > 0 { worker_i % n_connectome } else { 0 };

            // Build test net
            let mut test_net = best_net.clone();
            test_net.add_worker(
                w_input.clone(), w_local.clone(), local_indices.clone(),
                w_conn_read.clone(), w_conn_write, conn_write_idx,
                w_sparse.clone(), sparse_indices.clone(), bias,
            );

            let acc = eval_accuracy(&mut test_net, op, n_classes);
            if acc > worker_best_acc {
                worker_best_acc = acc;
                worker_best_params = Some((w_input, w_local, w_conn_read, w_conn_write, conn_write_idx, w_sparse, bias));
            }
            if worker_best_acc >= 1.0 { break; }
        }

        // Apply best worker
        if let Some((w_input, w_local, w_conn_read, w_conn_write, conn_write_idx, w_sparse, bias)) = worker_best_params {
            best_net.add_worker(
                w_input, w_local, local_indices,
                w_conn_read, w_conn_write, conn_write_idx,
                w_sparse, sparse_indices, bias,
            );
            best_net.freeze_last();
            best_acc = worker_best_acc;
        } else {
            // No improvement, add a zero worker to keep going
            best_net.add_worker(
                vec![0.0; INPUT_DIM], vec![0.0; n_local], local_indices,
                vec![0.0; n_connectome], 0.0, 0,
                vec![0.0; actual_sparse], sparse_indices, 0.0,
            );
            best_net.freeze_last();
        }

        let method = if use_exhaustive { "exh" } else { "rnd" };
        print!("  w{}: {:.0}% [{}:3^{}={}]",
            worker_i, best_acc * 100.0, method, n_params, total_configs);

        if best_acc >= 1.0 {
            println!(" SOLVED!");
            return (best_acc, worker_i + 1);
        }
    }
    println!();

    (best_acc, max_workers)
}

// ============================================================
// Also test: connectome neurons WITH input weights (they see input too)
// ============================================================
fn incremental_build_input_connectome(
    n_connectome: usize,
    max_workers: usize,
    op: &fn(usize, usize) -> usize,
    n_classes: usize,
    seed: u64,
) -> (f64, usize) {
    // First, search for good connectome neuron weights
    let ternary: Vec<f32> = vec![-1.0, 0.0, 1.0];
    let conn_params = INPUT_DIM + 1; // w_input + bias per connectome neuron
    let conn_configs = 3u64.pow(conn_params as u32); // 3^9 = 19683 per neuron

    // Search each connectome neuron independently
    let mut best_net = ConnectomeNet::new(0, true, false); // start empty, we'll add manually
    best_net.n_connectome = n_connectome;

    // Reset and build connectome neurons with input weights
    let mut net = ConnectomeNet {
        neurons: Vec::new(),
        activations: Vec::new(),
        n_connectome,
        use_connectome_activation: true,
        use_sparse: false,
    };

    // Search all connectome neurons jointly (if small enough)
    let total_conn_params = n_connectome * conn_params;
    let total_conn_configs = 3u64.saturating_pow(total_conn_params as u32);

    let search_configs = total_conn_configs.min(5_000_000);
    let use_exhaustive = total_conn_configs <= 5_000_000;
    let mut rng = StdRng::seed_from_u64(seed);

    let mut best_conn_acc = 0.0f64;
    let mut best_conn_neurons: Vec<(Vec<f32>, f32)> = vec![(vec![0.0; INPUT_DIM], 0.0); n_connectome];

    println!("  Searching connectome neurons ({}×{} params, 3^{}={})...",
        n_connectome, conn_params, total_conn_params, total_conn_configs);

    for sample in 0..search_configs {
        let mut c = if use_exhaustive { sample } else { rng.gen_range(0..total_conn_configs) };

        let mut conn_neurons = Vec::new();
        for _ in 0..n_connectome {
            let w_input: Vec<f32> = (0..INPUT_DIM).map(|_| { let v = ternary[(c % 3) as usize]; c /= 3; v }).collect();
            let bias = ternary[(c % 3) as usize]; c /= 3;
            conn_neurons.push((w_input, bias));
        }

        // Build net with these connectome neurons
        let mut test_net = ConnectomeNet {
            neurons: Vec::new(),
            activations: Vec::new(),
            n_connectome,
            use_connectome_activation: true,
            use_sparse: false,
        };
        for (w_input, bias) in &conn_neurons {
            test_net.neurons.push(Neuron {
                w_input: w_input.clone(),
                w_local: vec![], local_indices: vec![],
                w_conn_read: vec![], w_conn_write: 0.0, conn_write_idx: 0,
                w_sparse: vec![], sparse_indices: vec![],
                bias: *bias, frozen: false, is_connectome: true,
            });
            test_net.activations.push(0.0);
        }

        let acc = eval_accuracy(&mut test_net, op, n_classes);
        if acc > best_conn_acc {
            best_conn_acc = acc;
            best_conn_neurons = conn_neurons;
        }
        if best_conn_acc >= 1.0 { break; }
    }

    println!("  Connectome alone: {:.0}%", best_conn_acc * 100.0);

    // Build network with best connectome
    net.neurons.clear();
    net.activations.clear();
    for (w_input, bias) in &best_conn_neurons {
        net.neurons.push(Neuron {
            w_input: w_input.clone(),
            w_local: vec![], local_indices: vec![],
            w_conn_read: vec![], w_conn_write: 0.0, conn_write_idx: 0,
            w_sparse: vec![], sparse_indices: vec![],
            bias: *bias, frozen: true, is_connectome: true,
        });
        net.activations.push(0.0);
    }

    // Now incrementally add workers
    let mut best_acc = best_conn_acc;
    let ternary_vals = vec![-1.0f32, 0.0, 1.0];

    for worker_i in 0..max_workers {
        if best_acc >= 1.0 { return (best_acc, worker_i); }

        let n_local = LOCAL_CAP.min(net.n_workers());
        let n_params = INPUT_DIM + n_local + n_connectome + 1 + 1; // input + local + conn_read + conn_write + bias
        let total_configs = 3u64.saturating_pow(n_params as u32);
        let use_exh = total_configs <= 2_000_000;
        let sample_count = if use_exh { total_configs } else { 2_000_000 };

        let worker_start = n_connectome;
        let total_workers = net.n_workers();
        let local_indices: Vec<usize> = if total_workers == 0 { vec![] }
            else {
                let start = worker_start + total_workers.saturating_sub(n_local);
                (start..worker_start + total_workers).collect()
            };

        let mut worker_best_acc = best_acc;
        let mut worker_best: Option<(Vec<f32>, Vec<f32>, Vec<f32>, f32, f32)> = None;

        for sample in 0..sample_count {
            let mut c = if use_exh { sample } else { rng.gen_range(0..total_configs) };

            let w_input: Vec<f32> = (0..INPUT_DIM).map(|_| { let v = ternary_vals[(c % 3) as usize]; c /= 3; v }).collect();
            let w_local: Vec<f32> = (0..n_local).map(|_| { let v = ternary_vals[(c % 3) as usize]; c /= 3; v }).collect();
            let w_conn_read: Vec<f32> = (0..n_connectome).map(|_| { let v = ternary_vals[(c % 3) as usize]; c /= 3; v }).collect();
            let w_conn_write = ternary_vals[(c % 3) as usize]; c /= 3;
            let bias = ternary_vals[(c % 3) as usize];
            let conn_write_idx = worker_i % n_connectome.max(1);

            let mut test_net = net.clone();
            test_net.add_worker(
                w_input.clone(), w_local.clone(), local_indices.clone(),
                w_conn_read.clone(), w_conn_write, conn_write_idx,
                vec![], vec![], bias,
            );

            let acc = eval_accuracy(&mut test_net, op, n_classes);
            if acc > worker_best_acc {
                worker_best_acc = acc;
                worker_best = Some((w_input, w_local, w_conn_read, w_conn_write, bias));
            }
            if worker_best_acc >= 1.0 { break; }
        }

        if let Some((w_input, w_local, w_conn_read, w_conn_write, bias)) = worker_best {
            let conn_write_idx = worker_i % n_connectome.max(1);
            net.add_worker(w_input, w_local, local_indices, w_conn_read, w_conn_write, conn_write_idx, vec![], vec![], bias);
        } else {
            net.add_worker(
                vec![0.0; INPUT_DIM], vec![0.0; n_local], local_indices,
                vec![0.0; n_connectome], 0.0, 0, vec![], vec![], 0.0,
            );
        }
        net.freeze_last();
        best_acc = worker_best_acc;

        let method = if use_exh { "exh" } else { "rnd" };
        print!("  w{}: {:.0}% [{}:3^{}]", worker_i, best_acc * 100.0, method, n_params);
        if best_acc >= 1.0 {
            println!(" SOLVED!");
            return (best_acc, worker_i + 1);
        }
    }
    println!();

    (best_acc, max_workers)
}

// ============================================================
// Main: sweep all architectures × tasks
// ============================================================
fn main() {
    println!("=== CONNECTOME ARCHITECTURE SWEEP ===");
    println!("Digits: 0..{}, Input: thermo({}), Ticks: {}, Local cap: {}",
        DIGITS, INPUT_DIM, TICKS, LOCAL_CAP);
    println!();

    let tasks = make_tasks();
    let max_workers = 10;
    let n_connectome = 3;

    // Architecture configs: (name, n_connectome, active, sparse)
    let architectures: Vec<(&str, usize, bool, bool)> = vec![
        ("A: No connectome (baseline)",     0, false, false),
        ("B: Passive connectome (relay)",    n_connectome, false, false),
        ("C: Active connectome (ReLU)",      n_connectome, true,  false),
        ("D: Active conn + sparse",          n_connectome, true,  true),
    ];

    println!("{:<32} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Architecture", "ADD", "MUL", "MAX", "MIN", "|a-b|");
    println!("{}", "=".repeat(80));

    for (arch_name, nc, active, sparse) in &architectures {
        println!("\n--- {} ---", arch_name);

        // Run all tasks × seeds in parallel using rayon
        let combos: Vec<(usize, u64)> = (0..tasks.len())
            .flat_map(|ti| SEEDS.iter().map(move |&s| (ti, s)))
            .collect();

        let results: Vec<(usize, f64, usize, u64)> = combos.par_iter()
            .map(|&(task_idx, seed)| {
                let task = &tasks[task_idx];
                let (acc, neurons) = incremental_build(
                    *nc, *active, *sparse, max_workers,
                    &task.op, task.n_classes, seed,
                );
                (task_idx, acc, neurons, seed)
            })
            .collect();

        // Find best seed per task
        print!("{:<32}", arch_name);
        for task_idx in 0..tasks.len() {
            let task_results: Vec<&(usize, f64, usize, u64)> = results.iter()
                .filter(|(ti, _, _, _)| *ti == task_idx)
                .collect();
            let best = task_results.iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            let mean_acc: f64 = task_results.iter().map(|(_, a, _, _)| a).sum::<f64>() / task_results.len() as f64;
            print!(" {:>3.0}%/{:.0}%", best.1 * 100.0, mean_acc * 100.0);
        }
        println!();
        println!("{:<32} (best/mean over {} seeds)", "", SEEDS.len());
    }

    // Bonus: Active connectome WITH input weights (connectome neurons see input directly)
    println!("\n--- E: Input connectome (sees input) ---");
    println!("{}", "-".repeat(80));
    for task in &tasks {
        println!("\n  Task: {}", task.name);
        // Multi-seed parallel
        let results: Vec<(f64, usize)> = SEEDS.par_iter()
            .map(|&seed| {
                incremental_build_input_connectome(
                    n_connectome, max_workers,
                    &task.op, task.n_classes, seed,
                )
            })
            .collect();
        let best = results.iter().max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap();
        let mean: f64 = results.iter().map(|r| r.0).sum::<f64>() / results.len() as f64;
        println!("  => {}: best={:.0}% mean={:.0}% ({} workers, {} seeds)",
            task.name, best.0 * 100.0, mean * 100.0, best.1, SEEDS.len());
    }

    println!("\n=== DONE ===");
}
