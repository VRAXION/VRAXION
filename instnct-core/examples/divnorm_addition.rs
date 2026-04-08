//! Divnorm addition: continuous activation (ReLU) + divisive normalization + matrix multiply.
//! The v4.2 "working kernel" recipe applied to addition with generalization test.
//!
//! RUNNING: divnorm_addition
//!
//! Three comparisons:
//!   A: v4.2 recipe (ReLU + divnorm + matrix multiply + leak)
//!   B: Same but WITHOUT divnorm
//!   C: Same but WITH spike (threshold → binary) instead of ReLU
//!
//! Run: cargo run --example divnorm_addition --release

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::{Duration, Instant};

const DIGITS: usize = 5;
const SUMS: usize = 9;
const H: usize = 64;
const V: usize = 16;  // input/output vocab (thermometer bits)
const TICKS: usize = 8;
const EDGE_CAP: usize = 200;
const WALL_SECS: u64 = 120;
const CHARGE_RATE: f32 = 0.3;
const LEAK: f32 = 0.85;
const THRESHOLD: f32 = 0.5;
const DIVNORM_ALPHA: f32 = 0.1;

#[derive(Clone)]
struct DivnormNet {
    mask: Vec<Vec<i8>>,    // ternary: -1, 0, +1 (H×H)
    w: Vec<Vec<f32>>,      // weight: 0.5 or 1.5 (H×H)
    charge: Vec<f32>,
    act: Vec<f32>,
    h: usize,
    use_divnorm: bool,
    use_spike: bool,       // if true: binary spike instead of ReLU
}

impl DivnormNet {
    fn new(h: usize, density: f32, use_divnorm: bool, use_spike: bool, rng: &mut impl Rng) -> Self {
        let mut mask = vec![vec![0i8; h]; h];
        let mut w = vec![vec![0.0f32; h]; h];
        for i in 0..h {
            for j in 0..h {
                if i == j { continue; }
                let r: f32 = rng.gen();
                if r < density / 2.0 { mask[i][j] = -1; }
                else if r > 1.0 - density / 2.0 { mask[i][j] = 1; }
                w[i][j] = if rng.gen::<f32>() > 0.5 { 0.5 } else { 1.5 };
            }
        }
        DivnormNet { mask, w, charge: vec![0.0; h], act: vec![0.0; h], h, use_divnorm, use_spike }
    }

    fn reset(&mut self) {
        self.charge.iter_mut().for_each(|c| *c = 0.0);
        self.act.iter_mut().for_each(|a| *a = 0.0);
    }

    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        let h = self.h;
        for t in 0..TICKS {
            // Input injection at tick 0
            if t == 0 {
                for i in 0..V.min(input.len()) {
                    self.act[i] = input[i];
                }
            }

            // Matrix multiply: raw = act @ (W * mask)
            let mut raw = vec![0.0f32; h];
            for i in 0..h {
                let mut sum = 0.0f32;
                for j in 0..h {
                    if self.mask[j][i] != 0 {
                        sum += self.act[j] * self.w[j][i] * self.mask[j][i] as f32;
                    }
                }
                raw[i] = sum + self.act[i] * 0.1; // self-connection
            }

            // Charge dynamics
            for i in 0..h {
                self.charge[i] += raw[i] * CHARGE_RATE;
                self.charge[i] *= LEAK;
            }

            // Activation
            if self.use_spike {
                // Binary spike: threshold → 0 or 1, reset charge
                for i in 0..h {
                    if self.charge[i] >= THRESHOLD {
                        self.act[i] = 1.0;
                        self.charge[i] = 0.0;
                    } else {
                        self.act[i] = 0.0;
                    }
                }
            } else {
                // ReLU: continuous activation proportional to charge
                for i in 0..h {
                    self.act[i] = (self.charge[i] - THRESHOLD).max(0.0);
                }
            }

            // Divnorm
            if self.use_divnorm {
                let total: f32 = self.act.iter().sum();
                if total > 0.0 {
                    let denom = 1.0 + DIVNORM_ALPHA * total;
                    for i in 0..h {
                        self.act[i] /= denom;
                    }
                }
            }

            // Clip
            for i in 0..h {
                self.charge[i] = self.charge[i].clamp(-THRESHOLD * 2.0, THRESHOLD * 2.0);
            }
        }

        // Output: last V neurons' charge
        let out_start = h - V;
        self.charge[out_start..h].to_vec()
    }

    // --- Mutations (same as v4.2: flip mask, toggle weight) ---
    fn mutate(&mut self, rng: &mut impl Rng) -> bool {
        match rng.gen_range(0..100u32) {
            0..40 => { // flip mask entry
                let i = rng.gen_range(0..self.h); let j = rng.gen_range(0..self.h);
                if i == j { return false; }
                self.mask[i][j] = match rng.gen_range(0..3u32) { 0 => -1, 1 => 0, _ => 1 };
                true
            }
            40..70 => { // toggle weight
                let i = rng.gen_range(0..self.h); let j = rng.gen_range(0..self.h);
                self.w[i][j] = if self.w[i][j] < 1.0 { 1.5 } else { 0.5 };
                true
            }
            _ => false // no param mutations (threshold/leak are fixed)
        }
    }

    fn save(&self) -> (Vec<Vec<i8>>, Vec<Vec<f32>>) {
        (self.mask.clone(), self.w.clone())
    }
    fn restore(&mut self, s: (Vec<Vec<i8>>, Vec<Vec<f32>>)) {
        self.mask = s.0; self.w = s.1;
    }
}

fn thermo_input(a: usize, b: usize) -> Vec<f32> {
    let mut input = vec![0.0f32; V];
    for i in 0..a.min(8) { input[i] = 1.0; }
    for i in 0..b.min(8) { input[8 + i] = 1.0; }
    input
}

fn eval(net: &mut DivnormNet, examples: &[(usize,usize,usize)]) -> f64 {
    let mut correct = 0;
    let out_start = net.h - V;
    for &(a, b, target) in examples {
        net.reset();
        let input = thermo_input(a, b);
        let output = net.forward(&input);
        // Argmax over output classes (bin output into SUMS classes)
        let mut scores = vec![0.0f32; SUMS];
        for (i, &val) in output.iter().enumerate() {
            let class = i * SUMS / V;
            scores[class] += val;
        }
        let pred = scores.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0);
        if pred == target { correct += 1; }
    }
    correct as f64 / examples.len() as f64
}

fn run_experiment(label: &str, use_divnorm: bool, use_spike: bool,
    train: &[(usize,usize,usize)], test: &[(usize,usize,usize)], all: &[(usize,usize,usize)]) {
    println!("--- {} ---", label);

    for &seed in &[42u64, 1042, 2042, 3042, 4042] {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut net = DivnormNet::new(H, 0.06, use_divnorm, use_spike, &mut rng);

        let deadline = Instant::now() + Duration::from_secs(WALL_SECS);
        let mut steps = 0;
        while Instant::now() < deadline {
            let before = eval(&mut net, train);
            let genome = net.save();
            if net.mutate(&mut rng) {
                let after = eval(&mut net, train);
                if after <= before { net.restore(genome); }
            }
            steps += 1;
        }

        let train_acc = eval(&mut net, train);
        let test_acc = eval(&mut net, test);
        let all_acc = eval(&mut net, all);
        let edges = net.mask.iter().flatten().filter(|&&m| m != 0).count();

        println!("  seed {}: train={:.0}% test={:.0}% all={:.0}% | edges={} steps={}",
            seed, train_acc*100.0, test_acc*100.0, all_acc*100.0, edges, steps);
    }
    println!();
}

fn main() {
    let all: Vec<_> = (0..DIGITS).flat_map(|a| (0..DIGITS).map(move |b| (a, b, a+b))).collect();
    let train: Vec<_> = all.iter().filter(|&&(_,_,s)| s != 4).cloned().collect();
    let test: Vec<_> = all.iter().filter(|&&(_,_,s)| s == 4).cloned().collect();

    println!("=== DIVNORM ADDITION: The v4.2 recipe on addition ===");
    println!("RUNNING: divnorm_addition");
    println!("H={}, ticks={}, density=6%, charge_rate={}, leak={}, threshold={}",
        H, TICKS, CHARGE_RATE, LEAK, THRESHOLD);
    println!("Train: sum≠4 (20), Test: sum=4 (5), Random: {:.0}%\n", 100.0/SUMS as f64);

    // A: Full v4.2 recipe (ReLU + divnorm)
    run_experiment("A: ReLU + divnorm (v4.2 recipe)", true, false, &train, &test, &all);

    // B: ReLU without divnorm
    run_experiment("B: ReLU only (no divnorm)", false, false, &train, &test, &all);

    // C: Spike + divnorm (INSTNCT-like but with divnorm)
    run_experiment("C: Spike + divnorm", true, true, &train, &test, &all);

    // D: Spike only (current INSTNCT equivalent)
    run_experiment("D: Spike only (INSTNCT-like)", false, true, &train, &test, &all);
}
