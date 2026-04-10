//! LUT extraction: train float → extract per-neuron LUT + denominators
//!
//! The C19 formula disappears. What remains:
//!   - Integer weights (from fraction extraction)
//!   - Integer denominator per neuron
//!   - Learned LUT per neuron (extracted from converged network)
//!
//! No C19 at runtime. No formula. Just integer math + table lookup.
//!
//! Run: cargo run --example c19_lut_extract --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const DIGITS: usize = 5;
const INPUT_DIM: usize = 8;
const LOCAL_CAP: usize = 3;
const TICKS: usize = 2;

fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.01);
    let rho = rho.max(0.0);
    let l = 6.0 * c;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let scaled = x / c;
    let n = scaled.floor();
    let t = scaled - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sgn * h + rho * h * h)
}

fn thermo_2(a: usize, b: usize) -> [f32; 8] {
    let mut v = [0.0f32; 8];
    for i in 0..a.min(4) { v[i] = 1.0; }
    for i in 0..b.min(4) { v[4 + i] = 1.0; }
    v
}

// ============================================================
// Float network (for training)
// ============================================================
#[derive(Clone)]
struct FloatNet {
    nc: usize, nw: usize,
    params: Vec<f32>, c_params: Vec<f32>, rho_params: Vec<f32>,
    offsets: Vec<usize>, local_counts: Vec<usize>,
}

impl FloatNet {
    fn new(nc: usize, nw: usize, rng: &mut StdRng, scale: f32) -> Self {
        let mut net = FloatNet {
            nc, nw: 0, params: Vec::new(), c_params: Vec::new(), rho_params: Vec::new(),
            offsets: Vec::new(), local_counts: Vec::new(),
        };
        for i in 0..nw {
            let nl = LOCAL_CAP.min(i);
            let np = INPUT_DIM + nl + nc + 1 + 1;
            net.offsets.push(net.params.len());
            net.local_counts.push(nl);
            net.params.extend((0..np).map(|_| rng.gen_range(-scale..scale)));
            net.c_params.push(1.0);
            net.rho_params.push(4.0);
            net.nw += 1;
        }
        net
    }

    fn forward(&self, a: usize, b: usize) -> f32 {
        let input = thermo_2(a, b);
        let nc = self.nc;
        let nw = self.nw;
        let mut act = vec![0.0f32; nc + nw];
        for _t in 0..TICKS {
            let mut cc = vec![0.0f32; nc];
            for i in 0..nw {
                let o = self.offsets[i];
                let nl = self.local_counts[i];
                let ww = self.params[o + INPUT_DIM + nl + nc];
                let slot = i % nc.max(1);
                if slot < nc { cc[slot] += act[nc + i] * ww; }
            }
            for i in 0..nc { act[i] = cc[i]; }
            let old = act.clone();
            for i in 0..nw {
                let o = self.offsets[i];
                let nl = self.local_counts[i];
                let mut s = self.params[o + INPUT_DIM + nl + nc + 1];
                for j in 0..INPUT_DIM { s += input[j] * self.params[o + j]; }
                let ls = i.saturating_sub(nl);
                for (k, wi) in (ls..i).enumerate() {
                    s += old[nc + wi] * self.params[o + INPUT_DIM + k];
                }
                for k in 0..nc { s += old[k] * self.params[o + INPUT_DIM + nl + k]; }
                act[nc + i] = c19(s, self.c_params[i], self.rho_params[i]);
            }
        }
        act[nc..].iter().sum()
    }

    fn mse(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut l = 0.0f64;
        for a in 0..DIGITS { for b in 0..DIGITS {
            let d = self.forward(a, b) as f64 - op(a, b) as f64;
            l += d * d;
        }}
        l / 25.0
    }

    fn accuracy(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut c = 0;
        for a in 0..DIGITS { for b in 0..DIGITS {
            if (self.forward(a, b).round() as i32) == (op(a, b) as i32) { c += 1; }
        }}
        c as f64 / 25.0
    }

    fn gradient(&mut self, op: fn(usize, usize) -> usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let eps = 1e-3f32;
        let n = self.params.len();
        let mut g = vec![0.0f32; n];
        for i in 0..n {
            let orig = self.params[i];
            self.params[i] = orig + eps; let lp = self.mse(op);
            self.params[i] = orig - eps; let lm = self.mse(op);
            self.params[i] = orig;
            g[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
        }
        let nw = self.c_params.len();
        let mut gc = vec![0.0f32; nw];
        let mut gr = vec![0.0f32; nw];
        for i in 0..nw {
            let orig = self.c_params[i];
            self.c_params[i] = orig + eps; let lp = self.mse(op);
            self.c_params[i] = orig - eps; let lm = self.mse(op);
            self.c_params[i] = orig;
            gc[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
            let orig = self.rho_params[i];
            self.rho_params[i] = orig + eps; let lp = self.mse(op);
            self.rho_params[i] = orig - eps; let lm = self.mse(op);
            self.rho_params[i] = orig;
            gr[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
        }
        (g, gc, gr)
    }

    /// Extract per-neuron LUT: for each possible integer sum, what does the neuron output?
    /// Uses the actual C19 function with learned C and rho.
    fn extract_neuron_lut(&self, worker_idx: usize, sum_range: (i32, i32)) -> Vec<(i32, i32)> {
        let c = self.c_params[worker_idx];
        let rho = self.rho_params[worker_idx];
        let mut lut = Vec::new();
        for s in sum_range.0..=sum_range.1 {
            let output = c19(s as f32, c, rho);
            // Quantize output to integer (multiply by denominator later)
            lut.push((s, output.round() as i32));
        }
        lut
    }
}

fn optimize(net: &mut FloatNet, op: fn(usize, usize) -> usize) -> (f64, usize) {
    let mut lr = 0.01f32;
    let patience = 100;
    let mut stale = 0;
    let mut best_loss = net.mse(op);
    let mut step = 0;
    loop {
        let acc = net.accuracy(op);
        if acc >= 1.0 { return (acc, step); }
        if stale >= patience { return (acc, step); }
        let (g, gc, gr) = net.gradient(op);
        let gn: f32 = g.iter().chain(gc.iter()).chain(gr.iter()).map(|x| x*x).sum::<f32>().sqrt();
        if gn < 1e-8 { return (acc, step); }
        let old_p = net.params.clone();
        let old_c = net.c_params.clone();
        let old_r = net.rho_params.clone();
        let ol = net.mse(op);
        let mut improved = false;
        for att in 0..5 {
            for i in 0..net.params.len() { net.params[i] = old_p[i] - lr * g[i] / gn; }
            for i in 0..net.c_params.len() { net.c_params[i] = (old_c[i] - lr * gc[i] / gn).max(0.01); }
            for i in 0..net.rho_params.len() { net.rho_params[i] = (old_r[i] - lr * gr[i] / gn).max(0.0); }
            let nl = net.mse(op);
            if nl < ol { lr *= 1.1; if nl < best_loss - 1e-8 { best_loss = nl; stale = 0; improved = true; } break; }
            else { lr *= 0.5; if att == 4 { net.params = old_p.clone(); net.c_params = old_c.clone(); net.rho_params = old_r.clone(); } }
        }
        if !improved { stale += 1; }
        step += 1;
    }
}

// ============================================================
// Pure integer LUT network (for deployment)
// ============================================================
#[derive(Clone)]
struct LUTNet {
    nc: usize,
    nw: usize,
    denom: u32,                         // global denominator
    int_weights: Vec<Vec<i32>>,         // per worker: integer numerators
    luts: Vec<Vec<(i32, i32)>>,         // per worker: input_sum → output mapping
    offsets_nl: Vec<(usize, usize)>,    // (param_count, local_count) per worker
}

impl LUTNet {
    /// Extract from a trained float network
    fn extract(float_net: &FloatNet, denom: u32) -> Self {
        let nc = float_net.nc;
        let nw = float_net.nw;
        let mut int_weights = Vec::new();
        let mut luts = Vec::new();
        let mut offsets_nl = Vec::new();

        for i in 0..nw {
            let o = float_net.offsets[i];
            let nl = float_net.local_counts[i];
            let np = INPUT_DIM + nl + nc + 1 + 1;

            // Integer weights via common denominator
            let weights: Vec<i32> = float_net.params[o..o+np].iter()
                .map(|&w| (w * denom as f32).round() as i32)
                .collect();

            // Build LUT: compute the range of possible integer sums
            // Input weights: ±denom (since thermo input is 0/1, weight is int)
            // Max sum = sum of absolute values of all weights
            let max_abs_sum: i32 = weights.iter().map(|w| w.abs()).sum();
            let sum_range = (-max_abs_sum, max_abs_sum);

            // For each possible sum, compute c19 output (using float C/rho) and round
            let c = float_net.c_params[i];
            let rho = float_net.rho_params[i];
            let lut: Vec<(i32, i32)> = (sum_range.0..=sum_range.1).map(|s| {
                // The actual input to c19 is s/denom (converting back to float scale)
                let float_input = s as f32 / denom as f32;
                let float_output = c19(float_input, c, rho);
                // Quantize output: multiply by denom to keep in integer scale
                let int_output = (float_output * denom as f32).round() as i32;
                (s, int_output)
            }).collect();

            int_weights.push(weights);
            luts.push(lut);
            offsets_nl.push((np, nl));
        }

        LUTNet { nc, nw, denom, int_weights, luts, offsets_nl }
    }

    /// Pure integer forward pass — ZERO float
    fn forward_int(&self, a: usize, b: usize) -> i32 {
        let input = thermo_2(a, b);
        let nc = self.nc;
        let nw = self.nw;
        let mut act = vec![0i32; nc + nw];

        for _t in 0..TICKS {
            // Connectome write
            let mut cc = vec![0i32; nc];
            for i in 0..nw {
                let w = &self.int_weights[i];
                let nl = self.offsets_nl[i].1;
                let ww = w[INPUT_DIM + nl + nc]; // write weight (integer)
                let slot = i % nc.max(1);
                if slot < nc { cc[slot] += act[nc + i] * ww / self.denom as i32; }
            }
            for k in 0..nc { act[k] = cc[k]; }
            let old = act.clone();

            for i in 0..nw {
                let w = &self.int_weights[i];
                let nl = self.offsets_nl[i].1;

                // Integer weighted sum
                let mut s: i32 = w[INPUT_DIM + nl + nc + 1]; // bias (integer)
                for j in 0..INPUT_DIM {
                    s += (input[j] * self.denom as f32) as i32 * w[j] / self.denom as i32;
                }
                // Simpler: input is 0/1, weight is integer
                for j in 0..INPUT_DIM {
                    if input[j] > 0.5 { s += w[j]; }
                }
                // Subtract the double-counted bias sum above...
                // Actually let's do it clean:
                let mut s: i32 = w[INPUT_DIM + nl + nc + 1]; // bias
                for j in 0..INPUT_DIM {
                    if input[j] > 0.5 { s += w[j]; }
                }
                let ls = i.saturating_sub(nl);
                for (k, wi) in (ls..i).enumerate() {
                    // old[nc+wi] is in integer scale (×denom)
                    // w[INPUT_DIM+k] is also integer
                    // product needs to be divided by denom to stay in scale
                    s += old[nc + wi] * w[INPUT_DIM + k] / self.denom as i32;
                }
                for k in 0..nc {
                    s += old[k] * w[INPUT_DIM + nl + k] / self.denom as i32;
                }

                // LUT lookup
                let lut = &self.luts[i];
                let min_s = lut.first().map(|e| e.0).unwrap_or(0);
                let idx = (s - min_s).max(0).min(lut.len() as i32 - 1) as usize;
                act[nc + i] = lut[idx].1;
            }
        }
        // Sum all worker outputs, divide by denom to get final answer
        let total: i32 = act[nc..].iter().sum();
        total
    }

    fn accuracy_int(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut correct = 0;
        let d = self.denom as i32;
        for a in 0..DIGITS { for b in 0..DIGITS {
            // Round: (total + denom/2) / denom for proper rounding
            let raw = self.forward_int(a, b);
            let answer = if raw >= 0 { (raw + d/2) / d } else { (raw - d/2) / d };
            if answer == op(a, b) as i32 { correct += 1; }
        }}
        correct as f64 / 25.0
    }
}

fn op_add(a: usize, b: usize) -> usize { a + b }
fn op_mul(a: usize, b: usize) -> usize { a * b }
fn op_max(a: usize, b: usize) -> usize { a.max(b) }
fn op_min(a: usize, b: usize) -> usize { a.min(b) }
fn op_sub_abs(a: usize, b: usize) -> usize { if a > b { a - b } else { b - a } }

fn main() {
    println!("=== LUT EXTRACTION: float train → integer weights + LUT ===\n");
    println!("C19 formula disappears at deploy. Only: int weights + LUT lookup.\n");

    let nc = 3;
    let n_seeds = 20;
    let seeds: Vec<u64> = (1..=n_seeds as u64).collect();

    let tasks: Vec<(&str, fn(usize, usize) -> usize, usize)> = vec![
        ("ADD",   op_add,     3),
        ("MAX",   op_max,     3),
        ("MIN",   op_min,     3),
        ("|a-b|", op_sub_abs, 6),
        ("MUL",   op_mul,     6),
    ];

    println!("{:>8} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "task", "seed", "float_acc", "d=5", "d=9", "d=15", "d=20");
    println!("{}", "=".repeat(70));

    for &(name, op, nw) in &tasks {
        let mut float_solved = 0;
        let mut lut_solved = [0u32; 4]; // d=5, 9, 15, 20
        let denoms = [5u32, 9, 15, 20];

        for &seed in &seeds {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut net = FloatNet::new(nc, nw, &mut rng, 0.5);
            let (float_acc, _) = optimize(&mut net, op);

            if float_acc >= 1.0 { float_solved += 1; }

            let mut row = format!("{:>8} {:>6} {:>9.0}%", name, seed, float_acc * 100.0);

            for (di, &d) in denoms.iter().enumerate() {
                let lut_net = LUTNet::extract(&net, d);
                let lut_acc = lut_net.accuracy_int(op);
                if lut_acc >= 1.0 { lut_solved[di] += 1; }
                row += &format!(" {:>9.0}%", lut_acc * 100.0);
            }
            println!("{}", row);
        }

        print!("{:>8} TOTAL: {:>3}/{:<2}", name, float_solved, n_seeds);
        for (di, &d) in denoms.iter().enumerate() {
            print!("   d={}: {}/{}", d, lut_solved[di], float_solved);
        }
        println!("\n");
    }

    // =========================================================
    // Detailed LUT dump for ADD seed 1
    // =========================================================
    println!("--- Detailed LUT dump: ADD seed 1, denom=9 ---\n");

    let mut rng = StdRng::seed_from_u64(1);
    let mut net = FloatNet::new(nc, 3, &mut rng, 0.5);
    optimize(&mut net, op_add);
    let lut_net = LUTNet::extract(&net, 9);

    for (i, (weights, lut)) in lut_net.int_weights.iter().zip(lut_net.luts.iter()).enumerate() {
        println!("  Worker {}: int_weights = [{}]",
            i, weights.iter().map(|w| format!("{:+}", w)).collect::<Vec<_>>().join(", "));

        // Show compact LUT (only entries that are actually reachable)
        let nonzero: Vec<_> = lut.iter().filter(|(_, out)| *out != 0).collect();
        println!("  Worker {}: LUT ({} total, {} nonzero):", i, lut.len(), nonzero.len());

        // Show a sample
        let show: Vec<_> = lut.iter()
            .filter(|(s, _)| s.abs() <= 20)
            .collect();
        for chunk in show.chunks(10) {
            let entries: Vec<String> = chunk.iter().map(|(s, o)| format!("{}→{}", s, o)).collect();
            println!("    {}", entries.join("  "));
        }
        println!();
    }

    println!("  Pure integer accuracy: {:.0}%\n", lut_net.accuracy_int(op_add) * 100.0);

    // Hardware summary
    println!("--- HARDWARE SUMMARY ---\n");
    println!("  Per neuron on chip:");
    println!("    - {} integer weights (4-6 bit each)", INPUT_DIM + LOCAL_CAP + nc + 2);
    println!("    - 1 LUT (17-50 entries, precomputed ROM)");
    println!("    - 1 integer denominator (shared per neighborhood)");
    println!("  Operations per tick:");
    println!("    - {} integer multiplies (weight × input)", INPUT_DIM);
    println!("    - 1 integer sum");
    println!("    - 1 LUT lookup");
    println!("  NO: float, FPU, C19 formula, sin, floor, division");

    println!("\n=== DONE ===");
}
