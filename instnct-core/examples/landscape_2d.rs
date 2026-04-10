//! 2D loss landscape visualization
//!
//! Train a network → pick 2 random directions → sweep a grid → ASCII heatmap
//!
//! Run: cargo run --example landscape_2d --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIGITS: usize = 5;
const INPUT_DIM: usize = 8;
const LOCAL_CAP: usize = 3;
const TICKS: usize = 2;

fn relu(x: f32) -> f32 { x.max(0.0) }

fn thermo_2(a: usize, b: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; INPUT_DIM];
    for i in 0..a.min(4) { v[i] = 1.0; }
    for i in 0..b.min(4) { v[4 + i] = 1.0; }
    v
}

#[derive(Clone)]
struct FlatNet {
    n_connectome: usize,
    n_workers: usize,
    params: Vec<f32>,
    worker_param_offsets: Vec<usize>,
    worker_local_counts: Vec<usize>,
}

impl FlatNet {
    fn new_random(nc: usize, nw: usize, rng: &mut StdRng, scale: f32) -> Self {
        let mut net = FlatNet {
            n_connectome: nc, n_workers: 0, params: Vec::new(),
            worker_param_offsets: Vec::new(), worker_local_counts: Vec::new(),
        };
        for i in 0..nw {
            let nl = LOCAL_CAP.min(i);
            let np = INPUT_DIM + nl + nc + 1 + 1;
            let init: Vec<f32> = (0..np).map(|_| rng.gen_range(-scale..scale)).collect();
            net.worker_param_offsets.push(net.params.len());
            net.worker_local_counts.push(nl);
            net.params.extend_from_slice(&init);
            net.n_workers += 1;
        }
        net
    }

    fn forward(&self, a: usize, b: usize) -> f32 {
        let input = thermo_2(a, b);
        let nc = self.n_connectome;
        let nw = self.n_workers;
        let mut act = vec![0.0f32; nc + nw];
        for _t in 0..TICKS {
            let mut cc = vec![0.0f32; nc];
            for i in 0..nw {
                let o = self.worker_param_offsets[i];
                let nl = self.worker_local_counts[i];
                let ww = self.params[o + INPUT_DIM + nl + nc];
                let wi = i % nc.max(1);
                if wi < nc { cc[wi] += act[nc + i] * ww; }
            }
            for i in 0..nc { act[i] = cc[i]; }
            let old = act.clone();
            for i in 0..nw {
                let o = self.worker_param_offsets[i];
                let nl = self.worker_local_counts[i];
                let mut s = self.params[o + INPUT_DIM + nl + nc + 1];
                for j in 0..INPUT_DIM { s += input[j] * self.params[o + j]; }
                let ls = i.saturating_sub(nl);
                for (k, wi) in (ls..i).enumerate() { s += old[nc + wi] * self.params[o + INPUT_DIM + k]; }
                for k in 0..nc { s += old[k] * self.params[o + INPUT_DIM + nl + k]; }
                act[nc + i] = relu(s);
            }
        }
        act[nc..].iter().sum()
    }

    fn mse_loss(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut l = 0.0f64;
        for a in 0..DIGITS { for b in 0..DIGITS {
            let d = self.forward(a, b) as f64 - op(a, b) as f64;
            l += d * d;
        }}
        l / 25.0
    }

    fn native_accuracy(&self, op: fn(usize, usize) -> usize) -> f64 {
        let mut c = 0;
        for a in 0..DIGITS { for b in 0..DIGITS {
            if (self.forward(a, b).round() as i32) == (op(a, b) as i32) { c += 1; }
        }}
        c as f64 / 25.0
    }

    fn gradient(&mut self, op: fn(usize, usize) -> usize) -> Vec<f32> {
        let eps = 1e-3f32;
        let n = self.params.len();
        let mut g = vec![0.0f32; n];
        for i in 0..n {
            let orig = self.params[i];
            self.params[i] = orig + eps; let lp = self.mse_loss(op);
            self.params[i] = orig - eps; let lm = self.mse_loss(op);
            self.params[i] = orig;
            g[i] = ((lp - lm) / (2.0 * eps as f64)) as f32;
        }
        g
    }
}

fn optimize(net: &mut FlatNet, op: fn(usize, usize) -> usize, steps: usize) {
    let mut lr = 0.01f32;
    for _ in 0..steps {
        if net.native_accuracy(op) >= 1.0 { break; }
        let g = net.gradient(op);
        let gn: f32 = g.iter().map(|x| x * x).sum::<f32>().sqrt();
        if gn < 1e-8 { break; }
        let old = net.params.clone();
        let ol = net.mse_loss(op);
        for att in 0..5 {
            for i in 0..net.params.len() { net.params[i] = old[i] - lr * g[i] / gn; }
            if net.mse_loss(op) < ol { lr *= 1.1; break; }
            else { lr *= 0.5; if att == 4 { net.params = old.clone(); } }
        }
    }
}

fn op_add(a: usize, b: usize) -> usize { a + b }
fn op_sub_abs(a: usize, b: usize) -> usize { if a > b { a - b } else { b - a } }

fn print_landscape(
    name: &str,
    net: &FlatNet,
    dir1: &[f32], dir2: &[f32],
    op: fn(usize, usize) -> usize,
    range: f32, grid: usize,
) {
    println!("--- {} landscape (range=±{:.1}) ---", name, range);
    println!("    ★ = trained solution (center)");
    println!("    Accuracy: ░=0-24% ▒=25-49% ▓=50-79% █=80-95% ●=96-99% ★=100%\n");

    let step = 2.0 * range / grid as f32;

    // Y axis label
    print!("  {:>6} ", format!("+{:.1}", range));
    for _ in 0..grid { print!(" "); }
    println!();

    for iy in 0..grid {
        let y = range - iy as f32 * step;

        if iy == grid / 2 {
            print!("  dir2 → ");
        } else if iy == 0 || iy == grid - 1 {
            print!("  {:>6} ", format!("{:.1}", y));
        } else {
            print!("         ");
        }

        for ix in 0..grid {
            let x = -range + ix as f32 * step;

            let mut test = net.clone();
            for i in 0..test.params.len() {
                test.params[i] += x * dir1[i] + y * dir2[i];
            }
            let acc = test.native_accuracy(op);

            let ch = if (ix == grid / 2) && (iy == grid / 2) {
                '★'
            } else if acc >= 1.0 {
                '●'
            } else if acc >= 0.80 {
                '█'
            } else if acc >= 0.50 {
                '▓'
            } else if acc >= 0.25 {
                '▒'
            } else {
                '░'
            };
            print!("{}", ch);
        }

        if iy == grid / 2 {
            println!("  ← dir2");
        } else {
            println!();
        }
    }

    print!("  {:>6} ", format!("-{:.1}", range));
    for _ in 0..grid { print!(" "); }
    println!();
    print!("         ");
    print!("{:.1}", -range);
    for _ in 0..grid - 8 { print!(" "); }
    println!("+{:.1}", range);
    println!("                    ↑ dir1 ↑\n");
}

fn main() {
    println!("=== 2D LOSS LANDSCAPE ===\n");

    let nc = 3;

    // =========================================================
    // ADD — 3 workers (should be one big valley)
    // =========================================================
    let mut rng = StdRng::seed_from_u64(42);
    let mut net_add = FlatNet::new_random(nc, 3, &mut rng, 0.5);
    let start_acc = net_add.native_accuracy(op_add);
    optimize(&mut net_add, op_add, 2000);
    let final_acc = net_add.native_accuracy(op_add);
    println!("ADD (3 workers): {:.0}% → {:.0}%", start_acc * 100.0, final_acc * 100.0);

    // Random directions (normalized)
    let n = net_add.params.len();
    let mut dir1: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0f32)).collect();
    let mut dir2: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.0..1.0f32)).collect();
    let norm1: f32 = dir1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = dir2.iter().map(|x| x * x).sum::<f32>().sqrt();
    for d in &mut dir1 { *d /= norm1; }
    for d in &mut dir2 { *d /= norm2; }

    // Close-up view
    print_landscape("ADD close-up", &net_add, &dir1, &dir2, op_add, 0.3, 41);

    // Wide view
    print_landscape("ADD wide", &net_add, &dir1, &dir2, op_add, 2.0, 51);

    // Very wide
    print_landscape("ADD very wide", &net_add, &dir1, &dir2, op_add, 5.0, 51);

    // =========================================================
    // |a-b| — 4 workers
    // =========================================================
    let mut rng2 = StdRng::seed_from_u64(123);
    let mut net_abs = FlatNet::new_random(nc, 4, &mut rng2, 0.5);
    let start_acc2 = net_abs.native_accuracy(op_sub_abs);
    optimize(&mut net_abs, op_sub_abs, 2000);
    let final_acc2 = net_abs.native_accuracy(op_sub_abs);
    println!("|a-b| (4 workers): {:.0}% → {:.0}%", start_acc2 * 100.0, final_acc2 * 100.0);

    let n2 = net_abs.params.len();
    let mut d1: Vec<f32> = (0..n2).map(|_| rng2.gen_range(-1.0..1.0f32)).collect();
    let mut d2: Vec<f32> = (0..n2).map(|_| rng2.gen_range(-1.0..1.0f32)).collect();
    let nn1: f32 = d1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nn2: f32 = d2.iter().map(|x| x * x).sum::<f32>().sqrt();
    for d in &mut d1 { *d /= nn1; }
    for d in &mut d2 { *d /= nn2; }

    print_landscape("|a-b| close-up", &net_abs, &d1, &d2, op_sub_abs, 0.3, 41);
    print_landscape("|a-b| wide", &net_abs, &d1, &d2, op_sub_abs, 2.0, 51);

    // =========================================================
    // Show gradient path on ADD
    // =========================================================
    println!("--- ADD gradient path (from random start) ---\n");
    let mut rng3 = StdRng::seed_from_u64(77);
    let mut net_path = FlatNet::new_random(nc, 3, &mut rng3, 0.5);

    // Use same directions as ADD landscape
    let center = net_add.params.clone();

    // Project start position onto 2D
    let dx: f32 = net_path.params.iter().zip(center.iter()).zip(dir1.iter())
        .map(|((p, c), d)| (p - c) * d).sum();
    let dy: f32 = net_path.params.iter().zip(center.iter()).zip(dir2.iter())
        .map(|((p, c), d)| (p - c) * d).sum();

    println!("  Start: ({:.2}, {:.2}) acc={:.0}%", dx, dy, net_path.native_accuracy(op_add) * 100.0);

    // Run gradient and track path
    let mut path_points: Vec<(f32, f32, f64)> = vec![(dx, dy, net_path.native_accuracy(op_add))];
    let mut lr = 0.01f32;

    for step in 0..500 {
        if net_path.native_accuracy(op_add) >= 1.0 {
            let px: f32 = net_path.params.iter().zip(center.iter()).zip(dir1.iter())
                .map(|((p, c), d)| (p - c) * d).sum();
            let py: f32 = net_path.params.iter().zip(center.iter()).zip(dir2.iter())
                .map(|((p, c), d)| (p - c) * d).sum();
            path_points.push((px, py, 1.0));
            println!("  Step {}: ({:.2}, {:.2}) acc=100% SOLVED!", step, px, py);
            break;
        }

        let g = net_path.gradient(op_add);
        let gn: f32 = g.iter().map(|x| x * x).sum::<f32>().sqrt();
        if gn < 1e-8 { break; }

        let old = net_path.params.clone();
        let ol = net_path.mse_loss(op_add);
        for att in 0..5 {
            for i in 0..net_path.params.len() { net_path.params[i] = old[i] - lr * g[i] / gn; }
            if net_path.mse_loss(op_add) < ol { lr *= 1.1; break; }
            else { lr *= 0.5; if att == 4 { net_path.params = old.clone(); } }
        }

        if step % 20 == 0 {
            let px: f32 = net_path.params.iter().zip(center.iter()).zip(dir1.iter())
                .map(|((p, c), d)| (p - c) * d).sum();
            let py: f32 = net_path.params.iter().zip(center.iter()).zip(dir2.iter())
                .map(|((p, c), d)| (p - c) * d).sum();
            let acc = net_path.native_accuracy(op_add);
            path_points.push((px, py, acc));
            println!("  Step {:>3}: ({:>6.2}, {:>6.2}) acc={:.0}%", step, px, py, acc * 100.0);
        }
    }

    println!("\n=== DONE ===");
}
