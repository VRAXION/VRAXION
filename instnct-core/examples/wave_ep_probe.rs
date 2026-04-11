//! Wave EP Probe — microscope view of bidirectional settling
//!
//! Tiny network (2-3 neurons per layer), XOR task
//! Prints EVERYTHING: every neuron state at every tick, both waves,
//! how they interact, where they cancel, where residual remains.
//!
//! Run: cargo run --example wave_ep_probe --release

use std::io::Write;

// ============================================================
// Activations
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
// RNG
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
}

// ============================================================
// Tiny EP Network with full probe output
// ============================================================

fn main() {
    let mut out = std::fs::File::create("instnct-core/wave_ep_probe_log.txt").unwrap();

    macro_rules! p {
        ($($arg:tt)*) => {{
            let s = format!($($arg)*);
            print!("{}\n", s);
            writeln!(out, "{}", s).ok();
        }};
    }

    p!("╔══════════════════════════════════════════════════════════╗");
    p!("║  WAVE EP PROBE — Microscope View of Bidirectional Flow  ║");
    p!("╚══════════════════════════════════════════════════════════╝");

    // XOR data
    let data: Vec<([f32; 2], f32)> = vec![
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];

    // Tiny network: 2 input → 3 hidden → 1 output
    let in_d = 2;
    let h_d = 3;
    let out_d = 1;
    let mut rng = Rng::new(42);
    let init_scale = 0.5;

    let s1 = init_scale * (2.0 / in_d as f32).sqrt();
    let s2 = init_scale * (2.0 / h_d as f32).sqrt();
    let mut w1: Vec<f32> = (0..h_d * in_d).map(|_| rng.range_f32(-s1, s1)).collect(); // 3×2
    let mut w2: Vec<f32> = (0..out_d * h_d).map(|_| rng.range_f32(-s2, s2)).collect(); // 1×3
    let mut b1 = vec![0.0f32; h_d];
    let mut b2 = vec![0.0f32; out_d];

    p!("");
    p!("=== INITIAL WEIGHTS ===");
    p!("  W1 (hidden←input, {}×{}):", h_d, in_d);
    for j in 0..h_d {
        p!("    h{}: [{:+.4}, {:+.4}]", j, w1[j*in_d], w1[j*in_d+1]);
    }
    p!("  W2 (output←hidden, {}×{}):", out_d, h_d);
    p!("    o0: [{:+.4}, {:+.4}, {:+.4}]", w2[0], w2[1], w2[2]);
    p!("  b1: [{:+.4}, {:+.4}, {:+.4}]", b1[0], b1[1], b1[2]);
    p!("  b2: [{:+.4}]", b2[0]);

    let acts: Vec<(&str, fn(f32) -> f32, f32)> = vec![
        ("C19_r8", |x| c19(x, 8.0), 8.0),
        ("C19_r16", |x| c19(x, 16.0), 16.0),
        ("tanh", |x| tanh_act(x), 0.0),
    ];

    let beta = 0.5f32;
    let dt = 0.5f32;
    let t_max = 30;

    for (act_name, act_fn, _rho) in &acts {
        p!("");
        p!("################################################################");
        p!("###  ACTIVATION: {}  ###", act_name);
        p!("################################################################");

        // Reset weights for fair comparison
        let mut rng2 = Rng::new(42);
        w1 = (0..h_d * in_d).map(|_| rng2.range_f32(-s1, s1)).collect();
        w2 = (0..out_d * h_d).map(|_| rng2.range_f32(-s2, s2)).collect();
        b1 = vec![0.0f32; h_d];
        b2 = vec![0.0f32; out_d];

        // ============================================
        // PROBE 1: Single sample, show ALL states per tick
        // ============================================
        p!("");
        p!("=== PROBE 1: Free Phase settling — input=(1,0), target=1 ===");
        p!("  Watching: how blue wave (input) and red wave (backward from output) interact");
        p!("");
        p!("  {:>4} | {:>10} {:>10} {:>10} | {:>10} | {:>10} {:>10} {:>10} | {:>10} | {:>8}",
            "tick", "s_h[0]", "s_h[1]", "s_h[2]", "s_out[0]",
            "act(h0)", "act(h1)", "act(h2)", "act(out)", "delta");
        p!("  {}+{}+{}+{}+{}",
            "-".repeat(4), "-".repeat(32), "-".repeat(11), "-".repeat(32), "-".repeat(11), );

        let x = [1.0f32, 0.0];
        let y = [1.0f32];
        let mut s_h = vec![0.0f32; h_d];
        let mut s_out = vec![0.0f32; out_d];

        for t in 0..t_max {
            let old_h = s_h.clone();
            let old_out = s_out.clone();

            // Compute drives BEFORE update (for display)
            let mut fwd_drive = vec![0.0f32; h_d]; // blue: W1*x
            let mut bwd_drive = vec![0.0f32; h_d]; // red: W2^T * act(s_out)
            for j in 0..h_d {
                fwd_drive[j] = b1[j];
                for i in 0..in_d { fwd_drive[j] += w1[j * in_d + i] * x[i]; }
                for k in 0..out_d { bwd_drive[j] += w2[k * h_d + j] * act_fn(s_out[k]); }
            }

            // Hidden layer update
            for j in 0..h_d {
                let drive = fwd_drive[j] + bwd_drive[j];
                s_h[j] = s_h[j] + dt * (-s_h[j] + drive);
            }

            // Output layer update
            for k in 0..out_d {
                let mut drive = b2[k];
                for j in 0..h_d { drive += w2[k * h_d + j] * act_fn(s_h[j]); }
                // Free phase: beta=0, no nudge
                s_out[k] = s_out[k] + dt * (-s_out[k] + drive);
            }

            let delta = s_h.iter().zip(&old_h).map(|(a,b)| (a-b).abs())
                .chain(s_out.iter().zip(&old_out).map(|(a,b)| (a-b).abs()))
                .fold(0.0f32, f32::max);

            p!("  {:>4} | {:>+10.4} {:>+10.4} {:>+10.4} | {:>+10.4} | {:>+10.4} {:>+10.4} {:>+10.4} | {:>+10.4} | {:>8.5}",
                t, s_h[0], s_h[1], s_h[2], s_out[0],
                act_fn(s_h[0]), act_fn(s_h[1]), act_fn(s_h[2]), act_fn(s_out[0]),
                delta);

            // Extra: show blue vs red drive at tick 0,1,5,10,20
            if t == 0 || t == 1 || t == 5 || t == 10 || t == 20 {
                p!("         🔵 blue(fwd): [{:+.4}, {:+.4}, {:+.4}]  🔴 red(bwd): [{:+.4}, {:+.4}, {:+.4}]",
                    fwd_drive[0], fwd_drive[1], fwd_drive[2],
                    bwd_drive[0], bwd_drive[1], bwd_drive[2]);
            }
        }

        let eq_out = act_fn(s_out[0]);
        p!("  → Equilibrium output: act(s_out) = {:.4} (target = 1.0)", eq_out);

        // ============================================
        // PROBE 2: Nudged phase (same sample)
        // ============================================
        p!("");
        p!("=== PROBE 2: Nudged Phase — same sample, beta={} ===", beta);
        p!("  Starting from free-phase equilibrium, now 🔴 nudges output toward target");
        p!("");

        let s_free_h = s_h.clone();
        let s_free_out = s_out.clone();
        let mut s_h = s_free_h.clone();
        let mut s_out = s_free_out.clone();

        p!("  {:>4} | {:>10} {:>10} {:>10} | {:>10} | {:>10} | {:>10}",
            "tick", "s_h[0]", "s_h[1]", "s_h[2]", "s_out[0]", "act(out)", "nudge_force");

        for t in 0..t_max {
            // Hidden
            for j in 0..h_d {
                let mut drive = b1[j];
                for i in 0..in_d { drive += w1[j * in_d + i] * x[i]; }
                for k in 0..out_d { drive += w2[k * h_d + j] * act_fn(s_out[k]); }
                s_h[j] = s_h[j] + dt * (-s_h[j] + drive);
            }
            // Output with nudge
            let nudge_force = -beta * (act_fn(s_out[0]) - y[0]);
            for k in 0..out_d {
                let mut drive = b2[k];
                for j in 0..h_d { drive += w2[k * h_d + j] * act_fn(s_h[j]); }
                let nudge = -beta * (act_fn(s_out[k]) - y[k]);
                s_out[k] = s_out[k] + dt * (-s_out[k] + drive + nudge);
            }

            if t < 10 || t % 5 == 0 {
                p!("  {:>4} | {:>+10.4} {:>+10.4} {:>+10.4} | {:>+10.4} | {:>+10.4} | {:>+10.4}",
                    t, s_h[0], s_h[1], s_h[2], s_out[0], act_fn(s_out[0]), nudge_force);
            }
        }

        p!("  → Nudged equilibrium: act(s_out) = {:.4}", act_fn(s_out[0]));
        p!("  → Free eq was:        act(s_out) = {:.4}", act_fn(s_free_out[0]));
        p!("  → Difference (learning signal): {:.6}", act_fn(s_out[0]) - act_fn(s_free_out[0]));

        // ============================================
        // PROBE 3: All 4 XOR samples — equilibrium states
        // ============================================
        p!("");
        p!("=== PROBE 3: All 4 XOR samples — free phase equilibrium ===");
        p!("  {:>8} | {:>10} {:>10} {:>10} | {:>10} | {:>10} | {:>6}",
            "input", "s_h[0]", "s_h[1]", "s_h[2]", "s_out[0]", "act(out)", "target");

        for &(ref x, target) in &data {
            let mut s_h = vec![0.0f32; h_d];
            let mut s_out = vec![0.0f32; out_d];
            let y = [target];
            for _ in 0..t_max {
                for j in 0..h_d {
                    let mut drive = b1[j];
                    for i in 0..in_d { drive += w1[j * in_d + i] * x[i]; }
                    for k in 0..out_d { drive += w2[k * h_d + j] * act_fn(s_out[k]); }
                    s_h[j] = s_h[j] + dt * (-s_h[j] + drive);
                }
                for k in 0..out_d {
                    let mut drive = b2[k];
                    for j in 0..h_d { drive += w2[k * h_d + j] * act_fn(s_h[j]); }
                    s_out[k] = s_out[k] + dt * (-s_out[k] + drive);
                }
            }
            let pred = act_fn(s_out[0]);
            let ok = if ((pred > 0.5) as u8 as f32 - target).abs() < 0.5 { "OK" } else { "MISS" };
            p!("  ({:.0},{:.0})    | {:>+10.4} {:>+10.4} {:>+10.4} | {:>+10.4} | {:>+10.4} | {:.0}  {}",
                x[0], x[1], s_h[0], s_h[1], s_h[2], s_out[0], pred, target, ok);
        }

        // ============================================
        // PROBE 4: Training — watch weights change
        // ============================================
        p!("");
        p!("=== PROBE 4: Training — watch learning happen ===");
        p!("  Training for 100 epochs, showing weights every 10 epochs");
        p!("  {:>5} | {:>26} | {:>14} | {:>6} {:>6} {:>6} {:>6}",
            "epoch", "W1[h0]", "W2[o0]", "(0,0)", "(0,1)", "(1,0)", "(1,1)");

        let lr = 0.05f32;

        for epoch in 0..100 {
            let lr_eff = if epoch < 10 { lr * (epoch as f32 + 1.0) / 10.0 } else { lr };

            for &(ref x_arr, target) in &data {
                let x = x_arr.as_slice();
                let y = [target];

                // Free phase
                let mut sh_f = vec![0.0f32; h_d];
                let mut so_f = vec![0.0f32; out_d];
                for _ in 0..t_max {
                    for j in 0..h_d {
                        let mut d = b1[j];
                        for i in 0..in_d { d += w1[j*in_d+i] * x[i]; }
                        for k in 0..out_d { d += w2[k*h_d+j] * act_fn(so_f[k]); }
                        sh_f[j] += dt * (-sh_f[j] + d);
                    }
                    for k in 0..out_d {
                        let mut d = b2[k];
                        for j in 0..h_d { d += w2[k*h_d+j] * act_fn(sh_f[j]); }
                        so_f[k] += dt * (-so_f[k] + d);
                    }
                }

                // Nudged phase
                let mut sh_n = sh_f.clone();
                let mut so_n = so_f.clone();
                for _ in 0..t_max {
                    for j in 0..h_d {
                        let mut d = b1[j];
                        for i in 0..in_d { d += w1[j*in_d+i] * x[i]; }
                        for k in 0..out_d { d += w2[k*h_d+j] * act_fn(so_n[k]); }
                        sh_n[j] += dt * (-sh_n[j] + d);
                    }
                    for k in 0..out_d {
                        let mut d = b2[k];
                        for j in 0..h_d { d += w2[k*h_d+j] * act_fn(sh_n[j]); }
                        let nudge = -beta * (act_fn(so_n[k]) - y[k]);
                        so_n[k] += dt * (-so_n[k] + d + nudge);
                    }
                }

                // Weight update
                let inv_b = 1.0 / beta;
                for j in 0..h_d {
                    let an = act_fn(sh_n[j]); let af = act_fn(sh_f[j]);
                    for i in 0..in_d {
                        w1[j*in_d+i] -= lr_eff * inv_b * (an * x[i] - af * x[i]);
                    }
                    b1[j] -= lr_eff * inv_b * (an - af);
                }
                for k in 0..out_d {
                    let aon = act_fn(so_n[k]); let aof = act_fn(so_f[k]);
                    for j in 0..h_d {
                        let ahn = act_fn(sh_n[j]); let ahf = act_fn(sh_f[j]);
                        w2[k*h_d+j] -= lr_eff * inv_b * (aon * ahn - aof * ahf);
                    }
                    b2[k] -= lr_eff * inv_b * (aon - aof);
                }
            }

            if epoch % 10 == 0 || epoch == 99 {
                // Eval all 4
                let mut preds = [0.0f32; 4];
                for (si, &(ref x_arr, _target)) in data.iter().enumerate() {
                    let x = x_arr.as_slice();
                    let mut sh = vec![0.0f32; h_d];
                    let mut so = vec![0.0f32; out_d];
                    for _ in 0..t_max {
                        for j in 0..h_d {
                            let mut d = b1[j];
                            for i in 0..in_d { d += w1[j*in_d+i] * x[i]; }
                            for k in 0..out_d { d += w2[k*h_d+j] * act_fn(so[k]); }
                            sh[j] += dt * (-sh[j] + d);
                        }
                        for k in 0..out_d {
                            let mut d = b2[k];
                            for j in 0..h_d { d += w2[k*h_d+j] * act_fn(sh[j]); }
                            so[k] += dt * (-so[k] + d);
                        }
                    }
                    preds[si] = act_fn(so[0]);
                }

                p!("  {:>5} | W1h0=[{:+.3},{:+.3}] b={:+.3} | W2=[{:+.3},{:+.3},{:+.3}] | {:.3} {:.3} {:.3} {:.3}",
                    epoch,
                    w1[0], w1[1], b1[0],
                    w2[0], w2[1], w2[2],
                    preds[0], preds[1], preds[2], preds[3]);
            }
        }

        // Final eval
        p!("");
        p!("  --- Final predictions (target: 0, 1, 1, 0) ---");
        for &(ref x_arr, target) in &data {
            let x = x_arr.as_slice();
            let mut sh = vec![0.0f32; h_d];
            let mut so = vec![0.0f32; out_d];
            for _ in 0..t_max {
                for j in 0..h_d {
                    let mut d = b1[j];
                    for i in 0..in_d { d += w1[j*in_d+i] * x[i]; }
                    for k in 0..out_d { d += w2[k*h_d+j] * act_fn(so[k]); }
                    sh[j] += dt * (-sh[j] + d);
                }
                for k in 0..out_d {
                    let mut d = b2[k];
                    for j in 0..h_d { d += w2[k*h_d+j] * act_fn(sh[j]); }
                    so[k] += dt * (-so[k] + d);
                }
            }
            let pred = act_fn(so[0]);
            let binary = if pred > 0.5 { 1 } else { 0 };
            let ok = if (binary as f32 - target).abs() < 0.5 { "OK" } else { "MISS" };
            p!("    ({:.0},{:.0}) → {:.4} (bin={}) target={:.0} {}",
                x_arr[0], x_arr[1], pred, binary, target, ok);
        }
    }

    p!("");
    p!("=== DONE ===");
}
