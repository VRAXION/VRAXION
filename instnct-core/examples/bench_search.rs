//! Quick benchmark: how fast is exhaustive search per neuron?
use std::time::Instant;

const DIGITS: usize = 5;
const INPUT_DIM: usize = 8;

fn relu(x: f32) -> f32 { x.max(0.0) }
fn thermo_2(a: usize, b: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; INPUT_DIM];
    for i in 0..a.min(4) { v[i] = 1.0; }
    for i in 0..b.min(4) { v[4 + i] = 1.0; }
    v
}

fn bench_exhaustive(n_params: usize, binary: bool) -> (f64, u64) {
    let vals: Vec<f32> = if binary { vec![-1.0, 1.0] } else { vec![-1.0, 0.0, 1.0] };
    let base = vals.len() as u64;
    let total = base.pow(n_params as u32);

    let t0 = Instant::now();
    let mut best_score = 0.0f64;

    for config in 0..total {
        let mut c = config;
        let weights: Vec<f32> = (0..n_params).map(|_| { let v = vals[(c % base) as usize]; c /= base; v }).collect();

        // Simulate: eval on 25 pairs with 2 ticks
        let mut score = 0u32;
        for a in 0..DIGITS {
            for b in 0..DIGITS {
                let input = thermo_2(a, b);
                let mut act = 0.0f32;
                for _tick in 0..2 {
                    let mut sum = weights[n_params - 1]; // bias
                    for (j, &w) in weights.iter().take(INPUT_DIM.min(n_params - 1)).enumerate() {
                        if j < input.len() { sum += input[j] * w; }
                    }
                    sum += act * weights.get(INPUT_DIM).copied().unwrap_or(0.0); // recurrent
                    act = relu(sum);
                }
                if act > 0.0 { score += 1; }
            }
        }
        let acc = score as f64 / 25.0;
        if acc > best_score { best_score = acc; }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    (elapsed, total)
}

fn main() {
    println!("{:>6} {:>8} {:>12} {:>10} {:>12}", "params", "type", "configs", "time", "configs/sec");
    println!("{}", "-".repeat(55));

    for &binary in &[true, false] {
        let label = if binary { "binary" } else { "ternary" };
        for &n in &[10, 13, 16, 18, 20, 22, 24] {
            let base: u64 = if binary { 2 } else { 3 };
            let total = base.saturating_pow(n as u32);
            if total > 50_000_000 { 
                println!("{:>6} {:>8} {:>12} {:>10} {:>12}", n, label, format!("{:.1e}", total as f64), "skip", "-");
                continue; 
            }
            let (time, configs) = bench_exhaustive(n, binary);
            println!("{:>6} {:>8} {:>12} {:>9.2}s {:>10.0}/s",
                n, label, configs, time, configs as f64 / time);
        }
    }
}
