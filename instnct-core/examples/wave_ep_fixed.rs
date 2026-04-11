//! Fixed EP Implementation — Equilibrium Propagation (Scellier & Bengio 2017)
//!
//! Bug fixed from wave_ep_kill1.rs:
//!   The weight update had the WRONG SIGN. Original code did:
//!     W -= lr * (1/beta) * (nudge - free)
//!   Correct EP update is:
//!     W += lr * (1/beta) * (nudge - free)
//!
//!   The contrastive Hebbian update should make the free phase behave MORE like
//!   the nudged phase (which is pushed toward the target). Subtracting instead
//!   of adding causes anti-learning (weights move away from the solution).
//!
//! Run: cargo run --example wave_ep_fixed --release

// ============================================================
// Activation functions
// ============================================================

fn c19(x: f32, rho: f32) -> f32 {
    let l = 6.0f32;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * h + rho * h * h
}

fn tanh_act(x: f32) -> f32 { x.tanh() }

// ============================================================
// RNG (same as original)
// ============================================================

struct Rng { state: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { state: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn range_f32(&mut self, lo: f32, hi: f32) -> f32 { lo + self.f32() * (hi - lo) }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next() as usize) % (i + 1);
            v.swap(i, j);
        }
    }
}

// ============================================================
// Activation enum
// ============================================================

#[derive(Clone, Copy)]
enum Act { C19(f32), Tanh }

impl Act {
    fn apply(&self, x: f32) -> f32 {
        match self { Act::C19(rho) => c19(x, *rho), Act::Tanh => tanh_act(x) }
    }
    fn name(&self) -> &'static str {
        match self { Act::C19(_) => "c19", Act::Tanh => "tanh" }
    }
}

// ============================================================
// EP Network (1 hidden layer)
// ============================================================

struct EpNet {
    w1: Vec<f32>,  // h_dim x in_dim
    w2: Vec<f32>,  // out_dim x h_dim
    b1: Vec<f32>,  // h_dim
    b2: Vec<f32>,  // out_dim
    in_dim: usize,
    h_dim: usize,
    out_dim: usize,
}

impl EpNet {
    fn new(in_dim: usize, h_dim: usize, out_dim: usize, init_scale: f32, rng: &mut Rng) -> Self {
        let s1 = init_scale * (2.0 / in_dim as f32).sqrt();
        let s2 = init_scale * (2.0 / h_dim as f32).sqrt();
        EpNet {
            w1: (0..h_dim * in_dim).map(|_| rng.range_f32(-s1, s1)).collect(),
            w2: (0..out_dim * h_dim).map(|_| rng.range_f32(-s2, s2)).collect(),
            b1: vec![0.0; h_dim],
            b2: vec![0.0; out_dim],
            in_dim, h_dim, out_dim,
        }
    }
}

// ============================================================
// Settle step (1 hidden layer)
//
// Correct EP dynamics (Scellier & Bengio 2017):
//   ds_h/dt   = -s_h   + W1 * x        + W2^T * act(s_out) + b1
//   ds_out/dt = -s_out  + W2 * act(s_h) + b2  + beta*(y - act(s_out))
//
// NO sigma' multiplier. The self-decay -s and the drive are additive.
// ============================================================

fn settle_step(
    s_h: &[f32], s_out: &[f32],
    x: &[f32], net: &EpNet, dt: f32, act: Act, beta: f32, y: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let (in_d, h, out_d) = (net.in_dim, net.h_dim, net.out_dim);

    // Hidden layer: forward (W1*x) + backward (W2^T * act(s_out)) + bias
    let mut new_h = vec![0.0f32; h];
    for j in 0..h {
        let mut drive = net.b1[j];
        for i in 0..in_d { drive += net.w1[j * in_d + i] * x[i]; }
        for k in 0..out_d { drive += net.w2[k * h + j] * act.apply(s_out[k]); }
        new_h[j] = s_h[j] + dt * (-s_h[j] + drive);
    }

    // Output layer: forward (W2 * act(s_h)) + bias + nudge
    let mut new_out = vec![0.0f32; out_d];
    for k in 0..out_d {
        let mut drive = net.b2[k];
        for j in 0..h { drive += net.w2[k * h + j] * act.apply(s_h[j]); }
        // Nudge: beta * (y - act(s_out)), pushes output toward target
        let nudge = beta * (y[k] - act.apply(s_out[k]));
        new_out[k] = s_out[k] + dt * (-s_out[k] + drive + nudge);
    }

    (new_h, new_out)
}

// ============================================================
// XOR data
// ============================================================

fn xor_data() -> Vec<(Vec<f32>, Vec<f32>)> {
    vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ]
}

// ============================================================
// Train and evaluate EP on XOR
// ============================================================

fn train_xor(act: Act, h_dim: usize, beta: f32, t_max: usize, dt: f32,
             lr: f32, n_epochs: usize, seed: u64) -> (f32, f32, Vec<(f32, f32)>) {
    let data = xor_data();
    let in_dim = 2;
    let out_dim = 1;
    let mut rng = Rng::new(seed);
    let init_scale = 1.0;
    let mut net = EpNet::new(in_dim, h_dim, out_dim, init_scale, &mut rng);

    let mut indices: Vec<usize> = (0..data.len()).collect();
    let mut epoch_log: Vec<(f32, f32)> = Vec::new(); // (accuracy, mse) per epoch

    for epoch in 0..n_epochs {
        let lr_eff = if epoch < 10 { lr * (epoch as f32 + 1.0) / 10.0 } else { lr };
        rng.shuffle(&mut indices);

        for &idx in &indices {
            let (x, y) = &data[idx];

            // === FREE PHASE: settle from zero ===
            let mut s_h = vec![0.0f32; h_dim];
            let mut s_out = vec![0.0f32; out_dim];
            for _ in 0..t_max {
                let (new_h, new_out) = settle_step(&s_h, &s_out, x, &net, dt, act, 0.0, y);
                s_h = new_h;
                s_out = new_out;
            }
            let s_free_h = s_h;
            let s_free_out = s_out;

            // === NUDGED PHASE: settle from free equilibrium ===
            let mut s_h = s_free_h.clone();
            let mut s_out = s_free_out.clone();
            for _ in 0..t_max {
                let (new_h, new_out) = settle_step(&s_h, &s_out, x, &net, dt, act, beta, y);
                s_h = new_h;
                s_out = new_out;
            }
            let s_nudge_h = s_h;
            let s_nudge_out = s_out;

            // === WEIGHT UPDATE ===
            // FIX: The sign must be += (not -=)
            // Delta_W = (1/beta) * (nudge_correlation - free_correlation)
            // W += lr * Delta_W
            let inv_beta = 1.0 / beta;

            for j in 0..h_dim {
                let a_n = act.apply(s_nudge_h[j]);
                let a_f = act.apply(s_free_h[j]);
                for i in 0..in_dim {
                    // FIX: += not -=
                    net.w1[j * in_dim + i] += lr_eff * inv_beta * (a_n * x[i] - a_f * x[i]);
                }
                // FIX: += not -=
                net.b1[j] += lr_eff * inv_beta * (a_n - a_f);
            }

            for k in 0..out_dim {
                let ao_n = act.apply(s_nudge_out[k]);
                let ao_f = act.apply(s_free_out[k]);
                for j in 0..h_dim {
                    let ah_n = act.apply(s_nudge_h[j]);
                    let ah_f = act.apply(s_free_h[j]);
                    // FIX: += not -=
                    net.w2[k * h_dim + j] += lr_eff * inv_beta * (ao_n * ah_n - ao_f * ah_f);
                }
                // FIX: += not -=
                net.b2[k] += lr_eff * inv_beta * (ao_n - ao_f);
            }
        }

        // Log accuracy every 50 epochs
        if epoch % 50 == 0 || epoch == n_epochs - 1 {
            let mut correct = 0;
            let mut total_mse = 0.0f32;
            for (x, y) in &data {
                let mut s_h = vec![0.0f32; h_dim];
                let mut s_out = vec![0.0f32; out_dim];
                for _ in 0..t_max {
                    let (nh, no) = settle_step(&s_h, &s_out, x, &net, dt, act, 0.0, y);
                    s_h = nh; s_out = no;
                }
                let pred = act.apply(s_out[0]);
                let target = y[0];
                total_mse += (pred - target) * (pred - target);
                let pred_bin = if pred > 0.5 { 1.0 } else { 0.0 };
                if (pred_bin - target).abs() < 0.1 { correct += 1; }
            }
            let acc = correct as f32 / data.len() as f32;
            let mse = total_mse / data.len() as f32;
            epoch_log.push((acc, mse));
        }
    }

    // Final evaluation
    let mut correct = 0;
    let mut total_mse = 0.0f32;
    println!("  Predictions:");
    for (x, y) in &data {
        let mut s_h = vec![0.0f32; h_dim];
        let mut s_out = vec![0.0f32; out_dim];
        for _ in 0..t_max {
            let (nh, no) = settle_step(&s_h, &s_out, x, &net, dt, act, 0.0, y);
            s_h = nh; s_out = no;
        }
        let pred = act.apply(s_out[0]);
        let target = y[0];
        total_mse += (pred - target) * (pred - target);
        let pred_bin = if pred > 0.5 { 1.0 } else { 0.0 };
        let ok = (pred_bin - target).abs() < 0.1;
        if ok { correct += 1; }
        println!("    ({:.0},{:.0}) -> pred={:.4} (bin={:.0}) target={:.0} {}",
            x[0], x[1], pred, pred_bin, target, if ok { "OK" } else { "WRONG" });
    }

    let final_acc = correct as f32 / data.len() as f32;
    let final_mse = total_mse / data.len() as f32;
    (final_acc, final_mse, epoch_log)
}

// ============================================================
// MAIN
// ============================================================

fn main() {
    println!("================================================================");
    println!("  EP Fixed: Equilibrium Propagation XOR Test");
    println!("  Bug fix: weight update sign (was -=, now +=)");
    println!("================================================================");
    println!();

    // ---- Test 1: tanh on XOR ----
    println!("--- TEST 1: tanh + XOR ---");
    println!("  H=10, beta=0.5, T=50, dt=0.5, lr=0.01, epochs=500");
    let (acc, mse, log) = train_xor(Act::Tanh, 10, 0.5, 50, 0.5, 0.01, 500, 42);
    println!("  Training curve:");
    for (i, (a, m)) in log.iter().enumerate() {
        println!("    epoch {:>4}: acc={:.3} mse={:.4}", i * 50, a, m);
    }
    println!("  FINAL: accuracy={:.3} mse={:.4}", acc, mse);
    let tanh_pass = acc >= 0.99;
    println!("  VERDICT: {}", if tanh_pass { "PASS (tanh learns XOR)" } else { "FAIL" });
    println!();

    // ---- Test 2: tanh with different seeds (more epochs + wider hidden) ----
    println!("--- TEST 2: tanh + XOR (3 seeds, H=20, 1000 epochs) ---");
    let mut tanh_accs = Vec::new();
    for seed in [42u64, 123, 7] {
        let (acc, mse, _) = train_xor(Act::Tanh, 20, 0.5, 50, 0.5, 0.01, 1000, seed);
        println!("  seed={}: acc={:.3} mse={:.4}", seed, acc, mse);
        tanh_accs.push(acc);
    }
    let tanh_mean = tanh_accs.iter().sum::<f32>() / tanh_accs.len() as f32;
    println!("  Mean tanh accuracy: {:.3}", tanh_mean);
    println!();

    // ---- Test 3: C19 rho=8 on XOR ----
    println!("--- TEST 3: C19 rho=8 + XOR ---");
    println!("  H=10, beta=0.5, T=50, dt=0.5, lr=0.01, epochs=500");
    let (acc, mse, log) = train_xor(Act::C19(8.0), 10, 0.5, 50, 0.5, 0.01, 500, 42);
    println!("  Training curve:");
    for (i, (a, m)) in log.iter().enumerate() {
        println!("    epoch {:>4}: acc={:.3} mse={:.4}", i * 50, a, m);
    }
    println!("  FINAL: accuracy={:.3} mse={:.4}", acc, mse);
    println!();

    // ---- Test 4: C19 rho=8 with different seeds ----
    println!("--- TEST 4: C19 rho=8 + XOR (3 seeds) ---");
    let mut c19_accs = Vec::new();
    for seed in [42u64, 123, 7] {
        let (acc, mse, _) = train_xor(Act::C19(8.0), 10, 0.5, 50, 0.5, 0.01, 500, seed);
        println!("  seed={}: acc={:.3} mse={:.4}", seed, acc, mse);
        c19_accs.push(acc);
    }
    let c19_mean = c19_accs.iter().sum::<f32>() / c19_accs.len() as f32;
    println!("  Mean C19 accuracy: {:.3}", c19_mean);
    println!();

    // ---- Test 5: C19 with smaller dt (stability test) ----
    println!("--- TEST 5: C19 rho=8 + XOR (dt=0.1, more stable) ---");
    let (acc5, mse5, _) = train_xor(Act::C19(8.0), 10, 0.5, 50, 0.1, 0.01, 500, 42);
    println!("  acc={:.3} mse={:.4}", acc5, mse5);
    println!();

    // ---- Summary ----
    println!("================================================================");
    println!("  SUMMARY");
    println!("================================================================");
    println!("  tanh  mean accuracy: {:.3} (should be ~1.000)", tanh_mean);
    println!("  C19   mean accuracy: {:.3}", c19_mean);
    println!("  C19   dt=0.1 acc:    {:.3}", acc5);
    println!();
    println!("  Bug was: weight update used -= instead of +=");
    println!("  This caused anti-learning (weights moved AWAY from solution)");
    println!("================================================================");
}
