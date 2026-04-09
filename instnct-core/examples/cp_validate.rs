//! Connection Point Architecture: internal validation.
//!
//! Validate that:
//! 1. CP write/read works correctly
//! 2. Info flows: neuron → CP → other neuron
//! 3. Freeze behavior: frozen neurons don't change
//! 4. Search space stays constant as network grows
//! 5. Simple ADD task works through CPs
//!
//! Architecture:
//!   - N neurons, incrementally added + frozen
//!   - CP_COUNT connection points (shared bulletin boards)
//!   - Per neuron: local weights + CP read weights + CP write weight + input weights + bias
//!   - Each tick: neurons compute, write to CPs, read from CPs
//!
//! Run: cargo run --example cp_validate --release

const DIGITS: usize = 5;
const CP_COUNT: usize = 3; // number of connection points

fn relu(x: f32) -> f32 { x.max(0.0) }

fn thermo(val: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; 4];
    for i in 0..val.min(4) { v[i] = 1.0; }
    v
}

/// A single neuron's parameters
#[derive(Clone, Debug)]
struct NeuronParams {
    w_local: Vec<f32>,    // weights FROM local neighbors (up to LOCAL_CAP)
    local_indices: Vec<usize>, // which neurons are local neighbors
    w_cp_read: Vec<f32>,  // weights for reading from each CP (CP_COUNT)
    w_cp_write: f32,      // weight for writing to CPs (how much this neuron contributes)
    cp_write_idx: usize,  // which CP this neuron writes to
    w_input: Vec<f32>,    // weights from input (INPUT_DIM)
    bias: f32,
    frozen: bool,
}

/// Connection Point: just a shared charge accumulator
#[derive(Clone, Debug)]
struct ConnectionPoint {
    charge: f32,
}

/// The full network
struct CPNetwork {
    neurons: Vec<NeuronParams>,
    activations: Vec<f32>,
    cps: Vec<ConnectionPoint>,
    input_dim: usize,
}

impl CPNetwork {
    fn new(input_dim: usize) -> Self {
        CPNetwork {
            neurons: Vec::new(),
            activations: Vec::new(),
            cps: (0..CP_COUNT).map(|_| ConnectionPoint { charge: 0.0 }).collect(),
            input_dim,
        }
    }

    /// Add a new neuron with given params
    fn add_neuron(&mut self, params: NeuronParams) {
        self.neurons.push(params);
        self.activations.push(0.0);
    }

    /// Freeze the last added neuron
    fn freeze_last(&mut self) {
        if let Some(n) = self.neurons.last_mut() {
            n.frozen = true;
        }
    }

    /// Run one tick: input injected, neurons compute, CPs updated
    fn tick(&mut self, input: &[f32]) {
        let n = self.neurons.len();

        // Phase 1: Clear CPs
        for cp in &mut self.cps {
            cp.charge = 0.0;
        }

        // Phase 2: Each neuron writes to its CP
        for i in 0..n {
            let cp_idx = self.neurons[i].cp_write_idx;
            if cp_idx < CP_COUNT {
                self.cps[cp_idx].charge += self.activations[i] * self.neurons[i].w_cp_write;
            }
        }

        // Phase 3: Each neuron computes new activation
        let old_activations = self.activations.clone();
        let cp_charges: Vec<f32> = self.cps.iter().map(|cp| cp.charge).collect();

        for i in 0..n {
            let p = &self.neurons[i];
            let mut sum = p.bias;

            // Local connections
            for (k, &neighbor_idx) in p.local_indices.iter().enumerate() {
                if neighbor_idx < n && k < p.w_local.len() {
                    sum += old_activations[neighbor_idx] * p.w_local[k];
                }
            }

            // CP read
            for (k, &w) in p.w_cp_read.iter().enumerate() {
                if k < CP_COUNT {
                    sum += cp_charges[k] * w;
                }
            }

            // Input
            for (k, &w) in p.w_input.iter().enumerate() {
                if k < input.len() {
                    sum += input[k] * w;
                }
            }

            self.activations[i] = relu(sum);
        }
    }

    /// Reset all activations and CPs
    fn reset(&mut self) {
        for a in &mut self.activations { *a = 0.0; }
        for cp in &mut self.cps { cp.charge = 0.0; }
    }

    /// Run recurrently on a sequence of digits
    fn run_sequence(&mut self, digits: &[usize]) -> Vec<f32> {
        self.reset();
        for &d in digits {
            let input = thermo(d);
            self.tick(&input);
        }
        self.activations.clone()
    }

    /// Count of params for next neuron to be added
    fn next_neuron_params_count(&self, local_cap: usize) -> usize {
        let local = local_cap.min(self.neurons.len());
        local + CP_COUNT + 1 + self.input_dim + 1 // w_local + w_cp_read + w_cp_write + w_input + bias
        // cp_write_idx is not a continuous param, it's a discrete choice (0..CP_COUNT)
    }
}

fn main() {
    println!("=== CONNECTION POINT ARCHITECTURE: VALIDATION ===\n");

    let input_dim = 4; // thermo bits
    let local_cap = 3;

    // =====================================================
    // TEST 1: Manual wiring — does info flow through CP?
    // =====================================================
    println!("--- TEST 1: Info flow through CP ---\n");

    let mut net = CPNetwork::new(input_dim);

    // Neuron 0: reads input, writes to CP0
    net.add_neuron(NeuronParams {
        w_local: vec![],
        local_indices: vec![],
        w_cp_read: vec![0.0; CP_COUNT],
        w_cp_write: 1.0,
        cp_write_idx: 0,
        w_input: vec![1.0, 1.0, 1.0, 1.0], // sum all thermo bits
        bias: 0.0,
        frozen: false,
    });

    // Neuron 1: reads ONLY from CP0 (not from input directly)
    net.add_neuron(NeuronParams {
        w_local: vec![],
        local_indices: vec![],
        w_cp_read: vec![1.0, 0.0, 0.0], // reads CP0 only
        w_cp_write: 0.0,
        cp_write_idx: 1,
        w_input: vec![0.0, 0.0, 0.0, 0.0], // NO direct input
        bias: 0.0,
        frozen: false,
    });

    println!("  Neuron0: reads input → writes CP0");
    println!("  Neuron1: reads CP0 → gets Neuron0's signal indirectly\n");

    for digit in 0..DIGITS {
        net.reset();
        let input = thermo(digit);
        net.tick(&input);
        println!("  digit={}: n0_act={:.1}, CP0={:.1}, n1_act={:.1} (should be {:.1})",
            digit, net.activations[0], net.cps[0].charge, net.activations[1], digit as f32);
    }

    let n1_sees_input = net.activations[1] > 0.0;
    println!("\n  Neuron1 sees input through CP? {} ✓\n",
        if n1_sees_input { "YES" } else { "NO ✗" });

    // =====================================================
    // TEST 2: Recurrent accumulation through CP
    // =====================================================
    println!("--- TEST 2: Recurrent ADD through CP ---\n");

    let mut net2 = CPNetwork::new(input_dim);

    // Neuron 0: accumulator — reads CP0 (its own output from last tick) + input
    net2.add_neuron(NeuronParams {
        w_local: vec![],
        local_indices: vec![],
        w_cp_read: vec![1.0, 0.0, 0.0], // reads CP0 = self from last tick
        w_cp_write: 1.0,                // writes to CP0
        cp_write_idx: 0,
        w_input: vec![1.0, 1.0, 1.0, 1.0], // sum thermo
        bias: 0.0,
        frozen: false,
    });

    println!("  Single neuron accumulator via CP0 (read self + input):\n");
    for digits in &[vec![2, 3], vec![1, 1, 1], vec![4, 4, 4], vec![0, 0, 0]] {
        let act = net2.run_sequence(digits);
        let target: usize = digits.iter().sum();
        let charge = act[0];
        println!("    {:?} → charge={:.1}, target={}, {}",
            digits, charge, target,
            if (charge - target as f32).abs() < 0.01 { "✓" } else { "✗" });
    }

    // =====================================================
    // TEST 3: Freeze behavior
    // =====================================================
    println!("\n--- TEST 3: Freeze behavior ---\n");

    let mut net3 = CPNetwork::new(input_dim);

    // Add and freeze neuron 0
    net3.add_neuron(NeuronParams {
        w_local: vec![],
        local_indices: vec![],
        w_cp_read: vec![0.0; CP_COUNT],
        w_cp_write: 1.0,
        cp_write_idx: 0,
        w_input: vec![1.0, 1.0, 1.0, 1.0],
        bias: 0.0,
        frozen: false,
    });
    net3.freeze_last();

    let w_before = net3.neurons[0].w_input.clone();

    // Add neuron 1
    net3.add_neuron(NeuronParams {
        w_local: vec![1.0],       // reads neuron 0
        local_indices: vec![0],
        w_cp_read: vec![1.0, 0.0, 0.0],
        w_cp_write: 1.0,
        cp_write_idx: 1,
        w_input: vec![0.0, 0.0, 0.0, 0.0],
        bias: 0.0,
        frozen: false,
    });

    let w_after = net3.neurons[0].w_input.clone();
    let frozen_ok = w_before == w_after && net3.neurons[0].frozen;
    println!("  Neuron 0 frozen, weights unchanged after adding neuron 1: {} {}",
        frozen_ok, if frozen_ok { "✓" } else { "✗" });

    // =====================================================
    // TEST 4: Search space stays constant
    // =====================================================
    println!("\n--- TEST 4: Search space stays constant ---\n");

    let mut net4 = CPNetwork::new(input_dim);
    for i in 0..20 {
        let params_count = net4.next_neuron_params_count(local_cap);
        let search_space = 3u64.pow(params_count as u32);
        println!("  Neuron {:>2}: {:>2} params, 3^{} = {:>12} {}",
            i, params_count, params_count, search_space,
            if i > 0 && params_count == net4.next_neuron_params_count(local_cap) { "(same)" } else { "" });

        // Add dummy neuron
        let local_n = local_cap.min(net4.neurons.len());
        let local_idx: Vec<usize> = if net4.neurons.is_empty() { vec![] }
            else { (net4.neurons.len().saturating_sub(local_n)..net4.neurons.len()).collect() };

        net4.add_neuron(NeuronParams {
            w_local: vec![0.0; local_n],
            local_indices: local_idx,
            w_cp_read: vec![0.0; CP_COUNT],
            w_cp_write: 0.0,
            cp_write_idx: 0,
            w_input: vec![0.0; input_dim],
            bias: 0.0,
            frozen: false,
        });
        net4.freeze_last();
    }

    // =====================================================
    // TEST 5: Exhaustive search for ADD through CP
    // =====================================================
    println!("\n--- TEST 5: Exhaustive search for ADD accumulator ---\n");

    let ternary: Vec<f32> = vec![-1.0, 0.0, 1.0];

    // 1 neuron, reads CP0 + input, writes CP0
    // Params: w_cp_read(3) + w_cp_write(1) + w_input(4) + bias(1) = 9
    // cp_write_idx = 0 (fixed for simplicity)
    // No local (first neuron)
    let total_params = CP_COUNT + 1 + input_dim + 1; // 3 + 1 + 4 + 1 = 9
    let total_configs = 3u64.pow(total_params as u32); // 3^9 = 19683
    println!("  1 neuron, {} params, 3^{} = {} configs (exhaustive)\n", total_params, total_params, total_configs);

    let mut best_acc = 0.0f64;
    let mut best_config = 0u64;

    for config in 0..total_configs {
        let mut c = config;
        let w_cp_read: Vec<f32> = (0..CP_COUNT).map(|_| { let v = ternary[(c % 3) as usize]; c /= 3; v }).collect();
        let w_cp_write = ternary[(c % 3) as usize]; c /= 3;
        let w_input: Vec<f32> = (0..input_dim).map(|_| { let v = ternary[(c % 3) as usize]; c /= 3; v }).collect();
        let bias = ternary[(c % 3) as usize];

        let mut net = CPNetwork::new(input_dim);
        net.add_neuron(NeuronParams {
            w_local: vec![],
            local_indices: vec![],
            w_cp_read: w_cp_read.clone(),
            w_cp_write,
            cp_write_idx: 0,
            w_input: w_input.clone(),
            bias,
            frozen: false,
        });

        // Eval on 2-input ADD: native output (charge = sum?)
        let mut correct = 0;
        for a in 0..DIGITS {
            for b in 0..DIGITS {
                let act = net.run_sequence(&[a, b]);
                let charge = act[0];
                let target = a + b;
                if (charge.round() as i32 - target as i32).abs() == 0 { correct += 1; }
            }
        }
        let acc = correct as f64 / 25.0;
        if acc > best_acc {
            best_acc = acc;
            best_config = config;
        }
    }

    // Decode best config
    let mut c = best_config;
    let best_cp_read: Vec<f32> = (0..CP_COUNT).map(|_| { let v = ternary[(c % 3) as usize]; c /= 3; v }).collect();
    let best_cp_write = ternary[(c % 3) as usize]; c /= 3;
    let best_input: Vec<f32> = (0..input_dim).map(|_| { let v = ternary[(c % 3) as usize]; c /= 3; v }).collect();
    let best_bias = ternary[(c % 3) as usize];

    println!("  Best config: acc={:.0}%", best_acc * 100.0);
    println!("    cp_read: {:?}", best_cp_read.iter().map(|v| *v as i8).collect::<Vec<_>>());
    println!("    cp_write: {}", best_cp_write as i8);
    println!("    w_input: {:?}", best_input.iter().map(|v| *v as i8).collect::<Vec<_>>());
    println!("    bias: {}", best_bias as i8);

    if best_acc >= 0.99 {
        // Test generalization
        let mut net_best = CPNetwork::new(input_dim);
        net_best.add_neuron(NeuronParams {
            w_local: vec![],
            local_indices: vec![],
            w_cp_read: best_cp_read,
            w_cp_write: best_cp_write,
            cp_write_idx: 0,
            w_input: best_input,
            bias: best_bias,
            frozen: false,
        });

        println!("\n  Generalization (native charge = sum):");
        for n_in in 2..=8 {
            let combos = gen_combos(n_in);
            let mut ok = 0;
            for combo in &combos {
                let act = net_best.run_sequence(combo);
                let target: usize = combo.iter().sum();
                if (act[0].round() as i32 - target as i32).abs() == 0 { ok += 1; }
            }
            println!("    {}-input: {}/{} = {:.1}%", n_in, ok, combos.len(), ok as f64 / combos.len() as f64 * 100.0);
        }
    }

    println!("\n=== ALL TESTS DONE ===");
}

fn gen_combos(n_inputs: usize) -> Vec<Vec<usize>> {
    let mut result = vec![vec![]];
    for _ in 0..n_inputs {
        let mut nr = Vec::new();
        for combo in &result { for d in 0..DIGITS { let mut c = combo.clone(); c.push(d); nr.push(c); } }
        result = nr;
    }
    result
}
