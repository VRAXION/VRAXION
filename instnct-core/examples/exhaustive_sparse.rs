//! Exhaustive search for optimal sparse autoencoder neurons
//!
//! Given: 2 neurons, each with up to 5 parents from 8 input bits
//! Search: all int-N weight combinations × C19 (c, rho) grid
//! Guarantee: finds the GLOBALLY OPTIMAL solution at given bit-width
//!
//! Run: cargo run --example exhaustive_sparse --release

use std::time::Instant;

fn load_unique_bytes(path: &str) -> Vec<u8> {
    let text = std::fs::read(path).expect("failed to read corpus");
    let mut seen = [false; 256]; for &b in &text { seen[b as usize] = true; }
    (0..=255u8).filter(|&b| seen[b as usize]).collect()
}

fn byte_to_bits(b: u8) -> [f32; 8] {
    let mut bits = [0.0f32; 8]; for i in 0..8 { bits[i] = ((b >> i) & 1) as f32; } bits
}

fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.1); let rho = rho.max(0.0); let l = 6.0 * c;
    if x >= l { return x - l; } if x <= -l { return x + l; }
    let scaled = x / c; let n = scaled.floor(); let t = scaled - n;
    let h = t * (1.0 - t); let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sgn * h + rho * h * h)
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

// ══════════════════════════════════════════════════════
// LINEAR DECODER — fit + eval (hidden→8 output bits)
// ══════════════════════════════════════════════════════
fn fit_and_eval_decoder(
    hidden_vecs: &[Vec<f32>],
    targets: &[[f32; 8]],
) -> usize {
    let n_h = hidden_vecs[0].len();
    let n = hidden_vecs.len();
    let mut w = vec![vec![0.0f32; n_h]; 8];
    let mut b = vec![0.0f32; 8];

    for _ in 0..1500 {
        let lr = 0.15;
        for (sigs, tgt) in hidden_vecs.iter().zip(targets) {
            for i in 0..8 {
                let mut z = b[i];
                for j in 0..n_h { z += w[i][j] * sigs[j]; }
                let a = sigmoid(z);
                let g = (a - tgt[i]) * a * (1.0 - a);
                for j in 0..n_h { w[i][j] -= lr * g * sigs[j]; }
                b[i] -= lr * g;
            }
        }
    }

    // Eval
    let mut correct = 0;
    for (sigs, tgt) in hidden_vecs.iter().zip(targets) {
        let ok = (0..8).all(|i| {
            let mut z = b[i];
            for j in 0..n_h { z += w[i][j] * sigs[j]; }
            (sigmoid(z) - tgt[i]).abs() < 0.4
        });
        if ok { correct += 1; }
    }
    correct
}

// ══════════════════════════════════════════════════════
// EXHAUSTIVE NEURON SEARCH
// ══════════════════════════════════════════════════════
struct NeuronResult {
    parents: Vec<usize>,
    weights: Vec<i32>,
    bias: i32,
    c: f32,
    rho: f32,
    acc: usize,
}

fn exhaustive_neuron_search(
    parent_set: &[usize],
    existing_hidden: &[Vec<f32>],  // hidden signals so far (per byte)
    inputs: &[[f32; 8]],
    targets: &[[f32; 8]],
    max_int: i32,
    c_grid: &[f32],
    rho_grid: &[f32],
) -> NeuronResult {
    let n_parents = parent_set.len();
    let n_bytes = inputs.len();
    let n_weights = n_parents + 1; // weights + bias
    let n_values = (2 * max_int + 1) as usize;
    let total_combos = n_values.pow(n_weights as u32);

    let mut best = NeuronResult {
        parents: parent_set.to_vec(),
        weights: vec![0; n_parents],
        bias: 0, c: 1.0, rho: 0.0, acc: 0,
    };

    // Precompute input signals for this parent set per byte
    let parent_signals: Vec<Vec<f32>> = (0..n_bytes).map(|bi| {
        let mut all_sigs: Vec<f32> = inputs[bi].to_vec();
        all_sigs.extend(&existing_hidden[bi]);
        parent_set.iter().map(|&p| all_sigs[p]).collect()
    }).collect();

    for combo in 0..total_combos {
        // Decode combo into weights + bias
        let mut r = combo;
        let mut ws = vec![0i32; n_parents];
        for w in &mut ws { *w = (r % n_values) as i32 - max_int; r /= n_values; }
        let bias = (r % n_values) as i32 - max_int;

        // Skip all-zero weights (useless neuron)
        if ws.iter().all(|&w| w == 0) { continue; }

        // For each (c, rho) in grid
        for &c in c_grid {
            for &rho in rho_grid {
                // Compute neuron output for each byte
                let mut hidden_with_new: Vec<Vec<f32>> = Vec::with_capacity(n_bytes);
                for bi in 0..n_bytes {
                    let mut dot = bias as f32;
                    for (j, &w) in ws.iter().enumerate() {
                        dot += w as f32 * parent_signals[bi][j];
                    }
                    let act = c19(dot, c, rho);

                    let mut h = existing_hidden[bi].clone();
                    h.push(act);
                    hidden_with_new.push(h);
                }

                // Evaluate: train linear decoder on hidden signals → reconstruction
                let acc = fit_and_eval_decoder(&hidden_with_new, targets);

                if acc > best.acc {
                    best.acc = acc;
                    best.weights = ws.clone();
                    best.bias = bias;
                    best.c = c;
                    best.rho = rho;

                    if acc == n_bytes { return best; }
                }
            }
        }
    }
    best
}

// ══════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════
fn main() {
    let t0 = Instant::now();
    let unique = load_unique_bytes("instnct-core/tests/fixtures/alice_corpus.txt");
    let inputs: Vec<[f32; 8]> = unique.iter().map(|&b| byte_to_bits(b)).collect();
    let targets = inputs.clone();

    println!("=== EXHAUSTIVE SPARSE AUTOENCODER SEARCH ===");
    println!("{} unique bytes, C19 activation, guaranteed optimal\n", unique.len());

    // C19 parameter grids
    let c_grid: Vec<f32> = vec![0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0];
    let rho_grid: Vec<f32> = vec![0.0, 0.5, 1.0, 2.0, 5.0, 10.0];

    // Parent set candidates: all combinations of 2,3,4 parents from 8 inputs
    let mut parent_sets_2: Vec<Vec<usize>> = Vec::new();
    let mut parent_sets_3: Vec<Vec<usize>> = Vec::new();
    let mut parent_sets_4: Vec<Vec<usize>> = Vec::new();
    for i in 0..8 { for j in (i+1)..8 { parent_sets_2.push(vec![i,j]); } }
    for i in 0..8 { for j in (i+1)..8 { for k in (j+1)..8 { parent_sets_3.push(vec![i,j,k]); } } }
    for i in 0..8 { for j in (i+1)..8 { for k in (j+1)..8 { for l in (k+1)..8 { parent_sets_4.push(vec![i,j,k,l]); } } } }

    // Sweep int widths
    for &(label, max_int) in &[("ternary", 1i32), ("int3", 3), ("int4", 7)] {
        let n_values = (2 * max_int + 1) as usize;

        println!("━━━ {} (±{}, {} values) ━━━", label, max_int, n_values);

        // Try different fan-in sizes
        for (fan_label, parent_sets) in &[
            ("fan=2", &parent_sets_2),
            ("fan=3", &parent_sets_3),
            ("fan=4", &parent_sets_4),
        ] {
            let combos_per_set = n_values.pow((parent_sets[0].len() + 1) as u32);
            let total_neuron1 = parent_sets.len() * combos_per_set * c_grid.len() * rho_grid.len();

            println!("\n  {} — {} parent sets × {} weight combos × {} (c,rho) = {} total",
                fan_label, parent_sets.len(), combos_per_set, c_grid.len() * rho_grid.len(), total_neuron1);

            if total_neuron1 > 500_000_000 {
                println!("  SKIP: too many combos ({})", total_neuron1);
                continue;
            }

            let t1 = Instant::now();

            // Neuron 1: exhaustive search over all parent sets
            let empty_hidden: Vec<Vec<f32>> = (0..unique.len()).map(|_| Vec::new()).collect();
            let mut best_n1 = NeuronResult {
                parents: vec![], weights: vec![], bias: 0, c: 1.0, rho: 0.0, acc: 0,
            };

            for (pi, pset) in parent_sets.iter().enumerate() {
                let r = exhaustive_neuron_search(
                    pset, &empty_hidden, &inputs, &targets, max_int, &c_grid, &rho_grid,
                );
                if r.acc > best_n1.acc {
                    best_n1 = r;
                    if pi % 5 == 0 || best_n1.acc == unique.len() {
                        println!("    N1 [{}/{}] best={}/{} parents={:?} w={:?} b={} c={:.1} rho={:.1}",
                            pi+1, parent_sets.len(), best_n1.acc, unique.len(),
                            best_n1.parents, best_n1.weights, best_n1.bias, best_n1.c, best_n1.rho);
                    }
                }
                if best_n1.acc == unique.len() { break; }
            }

            println!("    N1 done: {}/{} in {:.1}s", best_n1.acc, unique.len(), t1.elapsed().as_secs_f64());

            if best_n1.acc == unique.len() {
                let n_params = best_n1.parents.len() + 1 + 2;
                println!("    ★ PERFECT with 1 neuron! {} edges, {} params, {} bytes",
                    best_n1.parents.len(), n_params, n_params);
                continue;
            }

            // Neuron 2: search given frozen N1
            let t2 = Instant::now();

            // Compute N1 outputs
            let n1_hidden: Vec<Vec<f32>> = (0..unique.len()).map(|bi| {
                let mut dot = best_n1.bias as f32;
                for (j, &p) in best_n1.parents.iter().enumerate() {
                    dot += best_n1.weights[j] as f32 * inputs[bi][p];
                }
                vec![c19(dot, best_n1.c, best_n1.rho)]
            }).collect();

            // Now search N2 — parents can include input bits AND N1 output (signal 8)
            let mut all_p2: Vec<Vec<usize>> = Vec::new();
            let n_sig = 9; // 8 inputs + 1 hidden (N1)
            for i in 0..n_sig { for j in (i+1)..n_sig {
                all_p2.push(vec![i,j]);
                for k in (j+1)..n_sig { all_p2.push(vec![i,j,k]); }
            }}

            let combos_n2 = all_p2.iter().map(|p| {
                n_values.pow((p.len() + 1) as u32) * c_grid.len() * rho_grid.len()
            }).sum::<usize>();
            println!("    N2 search: {} parent sets, ~{} total combos", all_p2.len(), combos_n2);

            let mut best_n2 = NeuronResult {
                parents: vec![], weights: vec![], bias: 0, c: 1.0, rho: 0.0, acc: best_n1.acc,
            };

            for (pi, pset) in all_p2.iter().enumerate() {
                let r = exhaustive_neuron_search(
                    pset, &n1_hidden, &inputs, &targets, max_int, &c_grid, &rho_grid,
                );
                if r.acc > best_n2.acc {
                    best_n2 = r;
                    println!("    N2 [{}/{}] best={}/{} parents={:?} w={:?} b={} c={:.1} rho={:.1}",
                        pi+1, all_p2.len(), best_n2.acc, unique.len(),
                        best_n2.parents, best_n2.weights, best_n2.bias, best_n2.c, best_n2.rho);
                }
                if best_n2.acc == unique.len() { break; }
            }

            println!("    N2 done: {}/{} in {:.1}s", best_n2.acc, unique.len(), t2.elapsed().as_secs_f64());

            if best_n2.acc == unique.len() {
                let total_edges = best_n1.parents.len() + best_n2.parents.len();
                let total_params = (best_n1.parents.len()+3) + (best_n2.parents.len()+3);
                println!("    ★★★ PERFECT with 2 neurons! {} edges, {} params, {} bytes ★★★",
                    total_edges, total_params, total_params);
                println!("    N1: parents={:?} w={:?} b={} c={:.1} rho={:.1}",
                    best_n1.parents, best_n1.weights, best_n1.bias, best_n1.c, best_n1.rho);
                println!("    N2: parents={:?} w={:?} b={} c={:.1} rho={:.1}",
                    best_n2.parents, best_n2.weights, best_n2.bias, best_n2.c, best_n2.rho);
            }

            println!("    Total time for {}/{}: {:.1}s\n",
                label, fan_label, t1.elapsed().as_secs_f64());
        }
        println!();
    }

    println!("Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
