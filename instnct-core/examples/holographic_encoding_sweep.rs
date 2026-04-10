//! Holographic encoding sweep: which encoding makes one-shot learning work?
//!
//! The problem: thermometer vectors are NOT orthogonal → cross-talk kills recall.
//! Solution: find the encoding that maximizes orthogonality for holographic storage.
//!
//! Run: cargo run --example holographic_encoding_sweep --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIGITS: usize = 5;
const N_PAIRS: usize = DIGITS * DIGITS; // 25

/// Holographic memory: store with outer product, retrieve with matrix-vector multiply
fn holo_test(
    dim: usize,
    // For each (a,b) pair: a dim-dimensional encoding vector
    encodings: &Vec<Vec<f32>>,
    op: fn(usize, usize) -> usize,
    max_output: usize,
) -> f64 {
    let out_dim = max_output + 1;
    let mut matrix = vec![0.0f32; out_dim * dim];

    // Store: M += encoding(a,b) ⊗ one_hot(target)
    for a in 0..DIGITS { for b in 0..DIGITS {
        let target = op(a, b);
        if target >= out_dim { continue; }
        let enc = &encodings[a * DIGITS + b];
        for j in 0..dim {
            matrix[target * dim + j] += enc[j];
        }
    }}

    // Retrieve: output = M × encoding(a,b), argmax
    let mut correct = 0;
    for a in 0..DIGITS { for b in 0..DIGITS {
        let target = op(a, b);
        if target >= out_dim { continue; }
        let enc = &encodings[a * DIGITS + b];
        let mut output = vec![0.0f32; out_dim];
        for i in 0..out_dim {
            for j in 0..dim {
                output[i] += matrix[i * dim + j] * enc[j];
            }
        }
        let pred = output.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        if pred == target { correct += 1; }
    }}
    correct as f64 / N_PAIRS as f64
}

/// Measure orthogonality: average absolute inner product between different pairs
fn orthogonality_score(encodings: &Vec<Vec<f32>>) -> f64 {
    let n = encodings.len();
    let mut total = 0.0f64;
    let mut count = 0;
    for i in 0..n { for j in i+1..n {
        let dot: f32 = encodings[i].iter().zip(encodings[j].iter())
            .map(|(a, b)| a * b).sum();
        total += dot.abs() as f64;
        count += 1;
    }}
    if count > 0 { total / count as f64 } else { 0.0 }
}

/// Quantized holographic test
fn holo_test_quantized(
    dim: usize, encodings: &Vec<Vec<f32>>,
    op: fn(usize, usize) -> usize, max_output: usize, levels: usize,
) -> f64 {
    let out_dim = max_output + 1;
    let mut matrix = vec![0.0f32; out_dim * dim];
    for a in 0..DIGITS { for b in 0..DIGITS {
        let target = op(a, b);
        if target >= out_dim { continue; }
        let enc = &encodings[a * DIGITS + b];
        for j in 0..dim { matrix[target * dim + j] += enc[j]; }
    }}

    // Quantize
    let min_v = matrix.iter().fold(f32::MAX, |a, &b| a.min(b));
    let max_v = matrix.iter().fold(f32::MIN, |a, &b| a.max(b));
    let step = (max_v - min_v) / (levels - 1) as f32;
    if step < 1e-10 { return 0.0; }
    let qmatrix: Vec<f32> = matrix.iter().map(|&v| {
        let idx = ((v - min_v) / step).round() as usize;
        min_v + idx.min(levels - 1) as f32 * step
    }).collect();

    let mut correct = 0;
    for a in 0..DIGITS { for b in 0..DIGITS {
        let target = op(a, b);
        if target >= out_dim { continue; }
        let enc = &encodings[a * DIGITS + b];
        let mut output = vec![0.0f32; out_dim];
        for i in 0..out_dim { for j in 0..dim {
            output[i] += qmatrix[i * dim + j] * enc[j];
        }}
        let pred = output.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        if pred == target { correct += 1; }
    }}
    correct as f64 / N_PAIRS as f64
}

fn op_add(a: usize, b: usize) -> usize { a + b }
fn op_mul(a: usize, b: usize) -> usize { a * b }
fn op_max(a: usize, b: usize) -> usize { a.max(b) }
fn op_min(a: usize, b: usize) -> usize { a.min(b) }
fn op_sub_abs(a: usize, b: usize) -> usize { if a > b { a - b } else { b - a } }

fn main() {
    println!("=== HOLOGRAPHIC ENCODING SWEEP ===\n");
    println!("25 patterns, sweep encodings, find what makes holographic memory work\n");

    let ops: Vec<(&str, fn(usize, usize) -> usize, usize)> = vec![
        ("ADD",   op_add,     8),
        ("MAX",   op_max,     4),
        ("MIN",   op_min,     4),
        ("|a-b|", op_sub_abs, 4),
        ("MUL",   op_mul,     16),
    ];

    let n_seeds = 20;

    // =========================================================
    // Encoding strategies
    // =========================================================

    println!("--- Encoding strategies (dim=32, 20 seeds) ---\n");
    println!("{:>20} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "encoding", "ortho", "ADD", "MAX", "MIN", "|a-b|", "MUL");
    println!("{}", "=".repeat(80));

    let strategies: Vec<(&str, Box<dyn Fn(&mut StdRng, usize) -> Vec<Vec<f32>>>)> = vec![
        // 1. Thermometer (current, bad)
        ("thermo", Box::new(|_rng: &mut StdRng, dim: usize| {
            let mut encs = Vec::new();
            for a in 0..DIGITS { for b in 0..DIGITS {
                let mut v = vec![0.0f32; dim];
                for i in 0..a.min(4) { v[i] = 1.0; }
                for i in 0..b.min(4) { v[4 + i] = 1.0; }
                let norm: f32 = v.iter().map(|x| x*x).sum::<f32>().sqrt();
                if norm > 1e-8 { for x in &mut v { *x /= norm; } }
                encs.push(v);
            }}
            encs
        })),

        // 2. One-hot per pair (perfect ortho, dim=25)
        ("one-hot-pair", Box::new(|_rng: &mut StdRng, dim: usize| {
            let mut encs = Vec::new();
            for idx in 0..N_PAIRS {
                let mut v = vec![0.0f32; dim];
                if idx < dim { v[idx] = 1.0; }
                encs.push(v);
            }
            encs
        })),

        // 3. One-hot per value (a one-hot + b one-hot)
        ("one-hot-val", Box::new(|_rng: &mut StdRng, dim: usize| {
            let mut encs = Vec::new();
            for a in 0..DIGITS { for b in 0..DIGITS {
                let mut v = vec![0.0f32; dim];
                if a < dim { v[a] = 1.0; }
                if DIGITS + b < dim { v[DIGITS + b] = 1.0; }
                let norm: f32 = v.iter().map(|x| x*x).sum::<f32>().sqrt();
                if norm > 1e-8 { for x in &mut v { *x /= norm; } }
                encs.push(v);
            }}
            encs
        })),

        // 4. Random ±1 per pair (dim-bit random binary)
        ("rand-binary", Box::new(|rng: &mut StdRng, dim: usize| {
            let mut encs = Vec::new();
            for _ in 0..N_PAIRS {
                let v: Vec<f32> = (0..dim).map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 }).collect();
                let norm: f32 = v.iter().map(|x| x*x).sum::<f32>().sqrt();
                encs.push(v.iter().map(|x| x / norm).collect());
            }
            encs
        })),

        // 5. Random gaussian per pair
        ("rand-gauss", Box::new(|rng: &mut StdRng, dim: usize| {
            let mut encs = Vec::new();
            for _ in 0..N_PAIRS {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0f32)).collect();
                let norm: f32 = v.iter().map(|x| x*x).sum::<f32>().sqrt();
                if norm > 1e-8 { encs.push(v.iter().map(|x| x / norm).collect()); }
                else { encs.push(vec![0.0; dim]); }
            }
            encs
        })),

        // 6. Random per value (a gets rand vec + b gets rand vec, sum)
        ("rand-val-sum", Box::new(|rng: &mut StdRng, dim: usize| {
            let a_vecs: Vec<Vec<f32>> = (0..DIGITS).map(|_|
                (0..dim).map(|_| rng.gen_range(-1.0..1.0f32)).collect()
            ).collect();
            let b_vecs: Vec<Vec<f32>> = (0..DIGITS).map(|_|
                (0..dim).map(|_| rng.gen_range(-1.0..1.0f32)).collect()
            ).collect();
            let mut encs = Vec::new();
            for a in 0..DIGITS { for b in 0..DIGITS {
                let mut v: Vec<f32> = (0..dim).map(|i| a_vecs[a][i] + b_vecs[b][i]).collect();
                let norm: f32 = v.iter().map(|x| x*x).sum::<f32>().sqrt();
                if norm > 1e-8 { for x in &mut v { *x /= norm; } }
                encs.push(v);
            }}
            encs
        })),

        // 7. Random per value (multiply element-wise = binding)
        ("rand-val-mul", Box::new(|rng: &mut StdRng, dim: usize| {
            let a_vecs: Vec<Vec<f32>> = (0..DIGITS).map(|_|
                (0..dim).map(|_| if rng.gen_bool(0.5) { 1.0f32 } else { -1.0 }).collect()
            ).collect();
            let b_vecs: Vec<Vec<f32>> = (0..DIGITS).map(|_|
                (0..dim).map(|_| if rng.gen_bool(0.5) { 1.0f32 } else { -1.0 }).collect()
            ).collect();
            let mut encs = Vec::new();
            for a in 0..DIGITS { for b in 0..DIGITS {
                let v: Vec<f32> = (0..dim).map(|i| a_vecs[a][i] * b_vecs[b][i]).collect();
                let norm: f32 = v.iter().map(|x| x*x).sum::<f32>().sqrt();
                encs.push(v.iter().map(|x| x / norm).collect());
            }}
            encs
        })),

        // 8. Hadamard-like: orthogonal basis
        ("hadamard", Box::new(|_rng: &mut StdRng, dim: usize| {
            // Simple: use rows of a pseudo-Hadamard matrix
            let mut encs = Vec::new();
            for idx in 0..N_PAIRS {
                let mut v = vec![0.0f32; dim];
                // Generate pseudo-random but deterministic orthogonal-ish vector
                for i in 0..dim {
                    // Use bit patterns of idx and i
                    let bits = (idx * 7 + i * 13 + idx * i * 3) % 4;
                    v[i] = match bits { 0 => 1.0, 1 => -1.0, 2 => 0.5, _ => -0.5 };
                }
                let norm: f32 = v.iter().map(|x| x*x).sum::<f32>().sqrt();
                if norm > 1e-8 { for x in &mut v { *x /= norm; } }
                encs.push(v);
            }
            encs
        })),
    ];

    for (name, gen_fn) in &strategies {
        let mut results: Vec<f64> = vec![0.0; ops.len()];
        let mut ortho_sum = 0.0f64;

        for seed in 0..n_seeds {
            let mut rng = StdRng::seed_from_u64(seed as u64);
            let encodings = gen_fn(&mut rng, 32);
            ortho_sum += orthogonality_score(&encodings);

            for (oi, &(_, op, max_out)) in ops.iter().enumerate() {
                results[oi] += holo_test(32, &encodings, op, max_out);
            }
        }

        let ortho = ortho_sum / n_seeds as f64;
        print!("{:>20} {:>7.3}", name, ortho);
        for r in &results {
            print!(" {:>7.0}%", r / n_seeds as f64 * 100.0);
        }
        println!();
    }

    // =========================================================
    // Best encoding: dimension sweep
    // =========================================================
    println!("\n--- Best encoding (rand-binary) dimension sweep ---\n");
    print!("{:>8}", "task");
    let dims = [8, 16, 25, 32, 48, 64, 128, 256, 512, 1024];
    for &d in &dims { print!(" {:>5}", format!("d={}", d)); }
    println!("  min_100%");
    println!("{}", "=".repeat(8 + dims.len() * 6 + 10));

    for &(name, op, max_out) in &ops {
        print!("{:>8}", name);
        let mut min_dim = 0usize;
        for &dim in &dims {
            let mut solved = 0;
            for seed in 0..n_seeds {
                let mut rng = StdRng::seed_from_u64(seed as u64);
                let encodings: Vec<Vec<f32>> = (0..N_PAIRS).map(|_| {
                    let v: Vec<f32> = (0..dim).map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 }).collect();
                    let n: f32 = v.iter().map(|x| x*x).sum::<f32>().sqrt();
                    v.iter().map(|x| x / n).collect()
                }).collect();
                if holo_test(dim, &encodings, op, max_out) >= 1.0 { solved += 1; }
            }
            let rate = solved as f64 / n_seeds as f64;
            print!(" {:>4.0}%", rate * 100.0);
            if rate >= 0.9 && min_dim == 0 { min_dim = dim; }
        }
        if min_dim > 0 { println!("  d≈{}", min_dim); } else { println!("  >1024"); }
    }

    // =========================================================
    // Best encoding: quantization test
    // =========================================================
    println!("\n--- Quantization: rand-binary, dim=64 ---\n");
    println!("{:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "task", "float", "8bit", "4bit", "3bit", "2bit", "1bit");
    println!("{}", "=".repeat(55));

    for &(name, op, max_out) in &ops {
        let mut accs = vec![0.0f64; 6];
        for seed in 0..n_seeds {
            let mut rng = StdRng::seed_from_u64(seed as u64);
            let encodings: Vec<Vec<f32>> = (0..N_PAIRS).map(|_| {
                let v: Vec<f32> = (0..64).map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 }).collect();
                let n: f32 = v.iter().map(|x| x*x).sum::<f32>().sqrt();
                v.iter().map(|x| x / n).collect()
            }).collect();
            accs[0] += holo_test(64, &encodings, op, max_out);
            for (qi, &levels) in [256, 16, 8, 4, 2].iter().enumerate() {
                accs[qi+1] += holo_test_quantized(64, &encodings, op, max_out, levels);
            }
        }
        println!("{:>8} {:>7.0}% {:>7.0}% {:>7.0}% {:>7.0}% {:>7.0}% {:>7.0}%",
            name,
            accs[0]/n_seeds as f64*100.0, accs[1]/n_seeds as f64*100.0,
            accs[2]/n_seeds as f64*100.0, accs[3]/n_seeds as f64*100.0,
            accs[4]/n_seeds as f64*100.0, accs[5]/n_seeds as f64*100.0);
    }

    println!("\n=== DONE ===");
}
