//! L1 Byte Merger — backprop STE with full int8 range
//!
//! Train float, round to int8 for forward (STE backward).
//! All params learnable: weights[-50..+50], bias, c, rho per neuron.
//! Sweep M=1,2,3 neurons for 100% on all 729 byte pairs.
//!
//! Run: cargo run --example l1_merger_backprop --release

use std::time::Instant;

const LUT: [[i8; 2]; 27] = [
    [-2,-4],[-4,-2],[0,-6],[-2,-5],[-1,-6],[-3,-5],[1,-8],[-1,-7],
    [-2,-6],[-4,-5],[0,-8],[-2,-7],[-1,-8],[-3,-7],[1,-10],[-1,-9],
    [-4,-6],[-6,-5],[-2,-8],[-4,-7],[-3,-8],[-5,-7],[-1,-10],[-3,-9],
    [-4,-8],[-6,-7],[-2,-10],
];

fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.1); let rho = rho.max(0.0); let l = 6.0 * c;
    if x >= l { return x - l; } if x <= -l { return x + l; }
    let s = x / c; let n = s.floor(); let t = s - n; let h = t * (1.0 - t);
    let sg = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sg * h + rho * h * h)
}

fn c19_dx(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.1); let rho = rho.max(0.0); let l = 6.0 * c;
    if x >= l || x <= -l { return 1.0; }
    let s = x / c; let n = s.floor(); let t = s - n; let h = t * (1.0 - t);
    let sg = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    (sg + 2.0 * rho * h) * (1.0 - 2.0 * t)
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed.wrapping_mul(6364136223846793005).wrapping_add(1)) }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + ((self.next() >> 33) % 65536) as f32 / 65536.0 * (hi - lo)
    }
}

fn round_clamp(x: f32, lo: f32, hi: f32) -> i8 {
    x.round().max(lo).min(hi) as i8
}

fn eval_codes(codes: &[[f32; 8]], m: usize) -> usize {
    let mut ok = 0;
    for i in 0..729 {
        let mut best = 0; let mut bd = f32::MAX;
        for j in 0..729 {
            let d: f32 = (0..m).map(|k| (codes[i][k] - codes[j][k]).powi(2)).sum();
            if d < bd { bd = d; best = j; }
        }
        if best == i { ok += 1; }
    }
    ok
}

fn main() {
    let t0 = Instant::now();

    // All 729 byte-pair inputs
    let inputs: Vec<[f32; 4]> = (0..27u8).flat_map(|a| {
        (0..27u8).map(move |b| [
            LUT[a as usize][0] as f32, LUT[a as usize][1] as f32,
            LUT[b as usize][0] as f32, LUT[b as usize][1] as f32,
        ])
    }).collect();

    println!("=== L1 MERGER — BACKPROP STE (full int8) ===\n");
    println!("  729 byte-pairs, C19 activation, all params learnable");
    println!("  weights int8 [-50,+50], c [0.5,100], rho [0,5]\n");

    let wclamp = 50.0f32;

    // M=1: max 249/729 (34%) — 1 neuron not enough
    for m in 2..=4usize {
        let tc = Instant::now();
        let mut best_score = 0u32;
        let mut best_cfg = String::new();

        let n_seeds = if m == 1 { 500 } else if m == 2 { 200 } else { 100 };
        let n_epochs = 3000;

        for seed in 0..n_seeds {
            let mut rng = Rng::new(seed * 7919 + 42);

            // Init
            let mut w = [[0.0f32; 4]; 8];
            let mut b = [0.0f32; 8];
            let mut cv = [0.0f32; 8];
            let mut rv = [0.0f32; 8];
            for k in 0..m {
                for j in 0..4 { w[k][j] = rng.uniform(-15.0, 15.0); }
                b[k] = rng.uniform(-15.0, 15.0);
                cv[k] = rng.uniform(5.0, 60.0);
                rv[k] = rng.uniform(0.0, 2.0);
            }

            // Adam state
            let mut mw = [[0.0f32; 4]; 8]; let mut vw = [[0.0f32; 4]; 8];
            let mut mb = [0.0f32; 8]; let mut vb = [0.0f32; 8];
            let mut mc = [0.0f32; 8]; let mut vc = [0.0f32; 8];
            let mut mr = [0.0f32; 8]; let mut vr = [0.0f32; 8];
            let mut adam_t = 0.0f32;

            let mut codes = [[0.0f32; 8]; 729];

            for ep in 0..n_epochs {
                let lr = 0.01 * (1.0 - ep as f32 / n_epochs as f32 * 0.8);

                // Quantize
                let mut qw = [[0i8; 4]; 8];
                let mut qb = [0i8; 8];
                for k in 0..m {
                    for j in 0..4 { qw[k][j] = round_clamp(w[k][j], -wclamp, wclamp); }
                    qb[k] = round_clamp(b[k], -wclamp, wclamp);
                }

                // Forward
                for (idx, inp) in inputs.iter().enumerate() {
                    for k in 0..m {
                        let dot = qb[k] as f32
                            + qw[k][0] as f32 * inp[0] + qw[k][1] as f32 * inp[1]
                            + qw[k][2] as f32 * inp[2] + qw[k][3] as f32 * inp[3];
                        codes[idx][k] = c19(dot, cv[k], rv[k]);
                    }
                }

                // Gradient: push apart nearest wrong neighbor
                let mut gw = [[0.0f32; 4]; 8];
                let mut gb = [0.0f32; 8];
                let mut gc = [0.0f32; 8];
                let mut gr = [0.0f32; 8];
                let mut n_active = 0u32;

                for i in 0..729usize {
                    let mut nj = 0usize; let mut nd = f32::MAX;
                    for j in 0..729 {
                        if j == i { continue; }
                        let d: f32 = (0..m).map(|k| (codes[i][k] - codes[j][k]).powi(2)).sum();
                        if d < nd { nd = d; nj = j; }
                    }

                    if nd < 2.0 {
                        n_active += 1;
                        let inp_i = &inputs[i];
                        let inp_j = &inputs[nj];

                        for k in 0..m {
                            let diff = codes[i][k] - codes[nj][k];
                            let sign = if diff > 0.0 { 1.0 } else if diff < 0.0 { -1.0 } else { 0.0 };

                            // Gradient through C19 for input i (push away)
                            let dot_i = qb[k] as f32
                                + qw[k][0] as f32 * inp_i[0] + qw[k][1] as f32 * inp_i[1]
                                + qw[k][2] as f32 * inp_i[2] + qw[k][3] as f32 * inp_i[3];
                            let g_i = c19_dx(dot_i, cv[k], rv[k]);

                            for j in 0..4 { gw[k][j] += sign * g_i * inp_i[j]; }
                            gb[k] += sign * g_i;

                            // c gradient (finite diff)
                            let eps = 0.01;
                            gc[k] += sign * (c19(dot_i, cv[k]+eps, rv[k]) - c19(dot_i, cv[k]-eps, rv[k])) / (2.0*eps);
                            gr[k] += sign * (c19(dot_i, cv[k], rv[k]+eps) - c19(dot_i, cv[k], rv[k]-eps)) / (2.0*eps);

                            // Also push j away
                            let dot_j = qb[k] as f32
                                + qw[k][0] as f32 * inp_j[0] + qw[k][1] as f32 * inp_j[1]
                                + qw[k][2] as f32 * inp_j[2] + qw[k][3] as f32 * inp_j[3];
                            let g_j = c19_dx(dot_j, cv[k], rv[k]);

                            for j in 0..4 { gw[k][j] -= sign * g_j * inp_j[j]; }
                            gb[k] -= sign * g_j;
                            gc[k] -= sign * (c19(dot_j, cv[k]+eps, rv[k]) - c19(dot_j, cv[k]-eps, rv[k])) / (2.0*eps);
                            gr[k] -= sign * (c19(dot_j, cv[k], rv[k]+eps) - c19(dot_j, cv[k], rv[k]-eps)) / (2.0*eps);
                        }
                    }
                }

                if n_active == 0 { break; } // all separated

                // Adam update
                adam_t += 1.0;
                let b1: f32 = 0.9; let b2: f32 = 0.999; let eps: f32 = 1e-8;
                let bc1 = 1.0 - b1.powf(adam_t);
                let bc2 = 1.0 - b2.powf(adam_t);
                let inv = 1.0 / n_active as f32;

                for k in 0..m {
                    for j in 0..4 {
                        let g = gw[k][j] * inv;
                        mw[k][j] = b1*mw[k][j] + (1.0-b1)*g;
                        vw[k][j] = b2*vw[k][j] + (1.0-b2)*g*g;
                        w[k][j] += lr * (mw[k][j]/bc1) / ((vw[k][j]/bc2).sqrt()+eps);
                        w[k][j] = w[k][j].max(-wclamp-0.5).min(wclamp+0.5);
                    }
                    let g = gb[k] * inv;
                    mb[k] = b1*mb[k]+(1.0-b1)*g; vb[k] = b2*vb[k]+(1.0-b2)*g*g;
                    b[k] += lr * (mb[k]/bc1)/((vb[k]/bc2).sqrt()+eps);
                    b[k] = b[k].max(-wclamp-0.5).min(wclamp+0.5);

                    let g = gc[k] * inv;
                    mc[k] = b1*mc[k]+(1.0-b1)*g; vc[k] = b2*vc[k]+(1.0-b2)*g*g;
                    cv[k] += lr * (mc[k]/bc1)/((vc[k]/bc2).sqrt()+eps);
                    cv[k] = cv[k].max(0.5).min(100.0);

                    let g = gr[k] * inv;
                    mr[k] = b1*mr[k]+(1.0-b1)*g; vr[k] = b2*vr[k]+(1.0-b2)*g*g;
                    rv[k] += lr * (mr[k]/bc1)/((vr[k]/bc2).sqrt()+eps);
                    rv[k] = rv[k].max(0.0).min(5.0);
                }
            }

            // Final eval quantized
            let mut qw_f = [[0i8; 4]; 8];
            let mut qb_f = [0i8; 8];
            for k in 0..m {
                for j in 0..4 { qw_f[k][j] = round_clamp(w[k][j], -wclamp, wclamp); }
                qb_f[k] = round_clamp(b[k], -wclamp, wclamp);
            }
            for (idx, inp) in inputs.iter().enumerate() {
                for k in 0..m {
                    let dot = qb_f[k] as f32
                        + qw_f[k][0] as f32*inp[0] + qw_f[k][1] as f32*inp[1]
                        + qw_f[k][2] as f32*inp[2] + qw_f[k][3] as f32*inp[3];
                    codes[idx][k] = c19(dot, cv[k], rv[k]);
                }
            }
            let score = eval_codes(&codes, m) as u32;

            if score > best_score {
                best_score = score;
                best_cfg = format!("seed={}", seed);
                for k in 0..m {
                    best_cfg += &format!("\n    N{}: w={:?} b={} c={:.1} rho={:.1}",
                        k, qw_f[k], qb_f[k], cv[k], rv[k]);
                }
                if score == 729 {
                    println!("  M={}: 729/729 at seed={} ({:.1}s)", m, seed, tc.elapsed().as_secs_f64());
                    for k in 0..m {
                        println!("    N{}: w={:?} b={} c={:.1} rho={:.1}",
                            k, qw_f[k], qb_f[k], cv[k], rv[k]);
                    }
                    break;
                }
            }

            // Progress
            if seed % 50 == 49 {
                println!("  M={}: seed {}/{} best={}/729 ({:.1}s)",
                    m, seed+1, n_seeds, best_score, tc.elapsed().as_secs_f64());
            }
        }

        if best_score < 729 {
            println!("  M={}: BEST {}/729 after {} seeds ({:.1}s)",
                m, best_score, n_seeds, tc.elapsed().as_secs_f64());
            println!("  {}", best_cfg);
        }

        println!();

        if best_score == 729 && m >= 2 { break; } // found minimum
    }

    println!("Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
