//! L1 2-Byte Merger — train, quantize, deploy LUT
//!
//! Train float (backprop STE) → quantize C19 output to int8 → verify 100%
//! Deploy: 729-entry LUT (27×27 byte pairs → 2 int8 output)
//! Replaces LUT_2N: merger_lut[char_a * 27 + char_b] → [int8, int8]
//!
//! Run: cargo run --example l1_merger_deploy --release

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

fn main() {
    let t0 = Instant::now();
    let m = 2usize;

    // All 729 byte-pair inputs
    let inputs: Vec<[f32; 4]> = (0..27u8).flat_map(|a| {
        (0..27u8).map(move |b| [
            LUT[a as usize][0] as f32, LUT[a as usize][1] as f32,
            LUT[b as usize][0] as f32, LUT[b as usize][1] as f32,
        ])
    }).collect();

    println!("=== L1 2-BYTE MERGER — TRAIN + QUANTIZE + DEPLOY ===\n");

    // ── Step 1: Backprop STE training ──
    println!("Step 1: Backprop STE training (M=2 neurons, full int8 weights)\n");

    let mut best_score = 0u32;
    let mut best_w = [[0i8; 4]; 2];
    let mut best_b = [0i8; 2];
    let mut best_c = [20.0f32; 2];
    let mut best_rho = [0.0f32; 2];
    let mut best_codes = vec![[0.0f32; 2]; 729];

    for seed in 0..100u64 {
        let mut rng = Rng::new(seed * 7919 + 42);
        let mut w: [[f32; 4]; 2] = [[0.0; 4]; 2];
        let mut b = [0.0f32; 2];
        let mut cv = [0.0f32; 2];
        let mut rv = [0.0f32; 2];

        for k in 0..2 {
            for j in 0..4 { w[k][j] = rng.uniform(-15.0, 15.0); }
            b[k] = rng.uniform(-15.0, 15.0);
            cv[k] = rng.uniform(5.0, 60.0);
            rv[k] = rng.uniform(0.0, 2.0);
        }

        // Adam
        let mut mw = [[0.0f32; 4]; 2]; let mut vw = [[0.0f32; 4]; 2];
        let mut mb = [0.0f32; 2]; let mut vb = [0.0f32; 2];
        let mut adam_t = 0.0f32;

        for ep in 0..3000 {
            let lr = 0.01 * (1.0 - ep as f32 / 3000.0 * 0.8);
            let wclamp = 50.0f32;

            let qw: [[i8; 4]; 2] = [
                [w[0][0].round().max(-wclamp).min(wclamp) as i8, w[0][1].round().max(-wclamp).min(wclamp) as i8,
                 w[0][2].round().max(-wclamp).min(wclamp) as i8, w[0][3].round().max(-wclamp).min(wclamp) as i8],
                [w[1][0].round().max(-wclamp).min(wclamp) as i8, w[1][1].round().max(-wclamp).min(wclamp) as i8,
                 w[1][2].round().max(-wclamp).min(wclamp) as i8, w[1][3].round().max(-wclamp).min(wclamp) as i8],
            ];
            let qb = [b[0].round().max(-wclamp).min(wclamp) as i8, b[1].round().max(-wclamp).min(wclamp) as i8];

            let codes: Vec<[f32; 2]> = inputs.iter().map(|inp| {
                let mut o = [0.0f32; 2];
                for k in 0..2 {
                    let dot = qb[k] as f32 + qw[k][0] as f32*inp[0] + qw[k][1] as f32*inp[1]
                        + qw[k][2] as f32*inp[2] + qw[k][3] as f32*inp[3];
                    o[k] = c19(dot, cv[k], rv[k]);
                }
                o
            }).collect();

            let mut gw = [[0.0f32; 4]; 2];
            let mut gb = [0.0f32; 2];
            let mut n_active = 0u32;

            for i in 0..729 {
                let mut nj = 0; let mut nd = f32::MAX;
                for j in 0..729 {
                    if j == i { continue; }
                    let d = (codes[i][0]-codes[j][0]).powi(2) + (codes[i][1]-codes[j][1]).powi(2);
                    if d < nd { nd = d; nj = j; }
                }
                if nd < 2.0 {
                    n_active += 1;
                    for k in 0..2 {
                        let sign = if codes[i][k] > codes[nj][k] { 1.0 } else if codes[i][k] < codes[nj][k] { -1.0 } else { 0.0 };
                        let dot_i = qb[k] as f32 + qw[k][0] as f32*inputs[i][0] + qw[k][1] as f32*inputs[i][1]
                            + qw[k][2] as f32*inputs[i][2] + qw[k][3] as f32*inputs[i][3];
                        let gi = c19_dx(dot_i, cv[k], rv[k]);
                        for j in 0..4 { gw[k][j] += sign * gi * inputs[i][j]; }
                        gb[k] += sign * gi;

                        let dot_j = qb[k] as f32 + qw[k][0] as f32*inputs[nj][0] + qw[k][1] as f32*inputs[nj][1]
                            + qw[k][2] as f32*inputs[nj][2] + qw[k][3] as f32*inputs[nj][3];
                        let gj = c19_dx(dot_j, cv[k], rv[k]);
                        for j in 0..4 { gw[k][j] -= sign * gj * inputs[nj][j]; }
                        gb[k] -= sign * gj;

                        // c/rho gradient (finite diff)
                        let eps = 0.01;
                        let dc = sign * (c19(dot_i, cv[k]+eps, rv[k]) - c19(dot_i, cv[k]-eps, rv[k])) / (2.0*eps);
                        cv[k] += lr * dc * 0.01;
                        cv[k] = cv[k].max(0.5).min(100.0);
                        let dr = sign * (c19(dot_i, cv[k], rv[k]+eps) - c19(dot_i, cv[k], rv[k]-eps)) / (2.0*eps);
                        rv[k] += lr * dr * 0.01;
                        rv[k] = rv[k].max(0.0).min(5.0);
                    }
                }
            }

            if n_active == 0 { break; }

            adam_t += 1.0;
            let b1: f32 = 0.9; let b2: f32 = 0.999; let eps: f32 = 1e-8;
            let bc1 = 1.0 - b1.powf(adam_t); let bc2 = 1.0 - b2.powf(adam_t);
            let inv = 1.0 / n_active as f32;
            for k in 0..2 {
                for j in 0..4 {
                    let g = gw[k][j] * inv;
                    mw[k][j] = b1*mw[k][j] + (1.0-b1)*g;
                    vw[k][j] = b2*vw[k][j] + (1.0-b2)*g*g;
                    w[k][j] += lr * (mw[k][j]/bc1) / ((vw[k][j]/bc2).sqrt()+eps);
                    w[k][j] = w[k][j].max(-50.5).min(50.5);
                }
                let g = gb[k] * inv;
                mb[k] = b1*mb[k]+(1.0-b1)*g; vb[k] = b2*vb[k]+(1.0-b2)*g*g;
                b[k] += lr * (mb[k]/bc1)/((vb[k]/bc2).sqrt()+eps);
                b[k] = b[k].max(-50.5).min(50.5);
            }
        }

        // Final eval
        let qw: [[i8; 4]; 2] = [
            [w[0][0].round().max(-50.0).min(50.0) as i8, w[0][1].round().max(-50.0).min(50.0) as i8,
             w[0][2].round().max(-50.0).min(50.0) as i8, w[0][3].round().max(-50.0).min(50.0) as i8],
            [w[1][0].round().max(-50.0).min(50.0) as i8, w[1][1].round().max(-50.0).min(50.0) as i8,
             w[1][2].round().max(-50.0).min(50.0) as i8, w[1][3].round().max(-50.0).min(50.0) as i8],
        ];
        let qb = [b[0].round().max(-50.0).min(50.0) as i8, b[1].round().max(-50.0).min(50.0) as i8];

        let codes: Vec<[f32; 2]> = inputs.iter().map(|inp| {
            let mut o = [0.0f32; 2];
            for k in 0..2 {
                let dot = qb[k] as f32 + qw[k][0] as f32*inp[0] + qw[k][1] as f32*inp[1]
                    + qw[k][2] as f32*inp[2] + qw[k][3] as f32*inp[3];
                o[k] = c19(dot, cv[k], rv[k]);
            }
            o
        }).collect();

        let mut ok = 0u32;
        for i in 0..729 {
            let mut best = 0; let mut bd = f32::MAX;
            for j in 0..729 {
                let d = (codes[i][0]-codes[j][0]).powi(2) + (codes[i][1]-codes[j][1]).powi(2);
                if d < bd { bd = d; best = j; }
            }
            if best == i { ok += 1; }
        }

        if ok > best_score {
            best_score = ok;
            best_w = qw; best_b = qb; best_c = cv; best_rho = rv;
            for i in 0..729 { best_codes[i] = codes[i]; }
            if ok == 729 {
                println!("  Float 729/729 at seed={}", seed);
                println!("    N0: w={:?} b={} c={:.1} rho={:.1}", qw[0], qb[0], cv[0], rv[0]);
                println!("    N1: w={:?} b={} c={:.1} rho={:.1}", qw[1], qb[1], cv[1], rv[1]);
                break;
            }
        }
    }

    // ── Step 2: Quantize C19 outputs to int8 ──
    println!("\nStep 2: Quantize float codes → int8\n");

    // Find range per dimension
    let mut min0 = f32::MAX; let mut max0 = f32::MIN;
    let mut min1 = f32::MAX; let mut max1 = f32::MIN;
    for c in &best_codes {
        if c[0] < min0 { min0 = c[0]; } if c[0] > max0 { max0 = c[0]; }
        if c[1] < min1 { min1 = c[1]; } if c[1] > max1 { max1 = c[1]; }
    }
    println!("  Float range: dim0=[{:.2}, {:.2}] dim1=[{:.2}, {:.2}]", min0, max0, min1, max1);

    // Scale to [-127, 127]
    let scale0 = if max0 - min0 > 0.0 { 254.0 / (max0 - min0) } else { 1.0 };
    let scale1 = if max1 - min1 > 0.0 { 254.0 / (max1 - min1) } else { 1.0 };

    let quantized: Vec<[i8; 2]> = best_codes.iter().map(|c| {
        [((c[0] - min0) * scale0 - 127.0).round().max(-128.0).min(127.0) as i8,
         ((c[1] - min1) * scale1 - 127.0).round().max(-128.0).min(127.0) as i8]
    }).collect();

    // Check uniqueness
    let mut unique = std::collections::HashSet::new();
    for q in &quantized { unique.insert((q[0], q[1])); }
    let n_unique = unique.len();
    println!("  Quantized: {}/729 unique int8 pairs", n_unique);

    // Verify round-trip
    let mut ok_quant = 0;
    for i in 0..729 {
        let mut best = 0; let mut bd = i32::MAX;
        for j in 0..729 {
            let d = (quantized[i][0] as i32 - quantized[j][0] as i32).pow(2)
                  + (quantized[i][1] as i32 - quantized[j][1] as i32).pow(2);
            if d < bd { bd = d; best = j; }
        }
        if best == i { ok_quant += 1; }
    }
    println!("  Round-trip after int8 quantization: {}/729", ok_quant);

    if ok_quant == 729 {
        println!("  *** 100% PRESERVED ***");
    } else {
        println!("  Collisions! Trying direct code assignment...\n");
        // Fallback: assign unique codes directly
        // Grid: 27×27, space codes evenly
        for i in 0..729 {
            let a = i / 27;
            let b = i % 27;
            // quantized[i] = [(a * 9) as i8 - 121, (b * 9) as i8 - 121]; // can't mutate, use new vec
        }
    }

    // ── Step 3: Build deploy LUT ──
    println!("\nStep 3: Deploy LUT (729 entries x 2 int8)\n");

    let deploy_lut: Vec<[i8; 2]> = if ok_quant == 729 {
        quantized
    } else {
        // Direct assignment: grid spacing
        (0..729).map(|i| {
            let a = (i / 27) as i8;
            let b = (i % 27) as i8;
            [(a * 9 - 117) as i8, (b * 9 - 117) as i8]
        }).collect()
    };

    // Print first few entries
    println!("  MERGER_LUT[char_a * 27 + char_b] -> [int8, int8]");
    println!("  Size: 729 x 2 = 1458 bytes\n");
    let chars = "abcdefghijklmnopqrstuvwxyz ";
    for a in 0..3u8 {
        for b in 0..5u8 {
            let idx = a as usize * 27 + b as usize;
            let ca = chars.as_bytes()[a as usize] as char;
            let cb = chars.as_bytes()[b as usize] as char;
            println!("    '{}{}' (idx={:>3}): [{:>4}, {:>4}]", ca, cb, idx, deploy_lut[idx][0], deploy_lut[idx][1]);
        }
        println!();
    }

    // ── Step 4: Verify deploy LUT 100% ──
    println!("Step 4: Verify deploy LUT round-trip\n");
    let mut ok_deploy = 0;
    for i in 0..729 {
        let mut best = 0; let mut bd = i32::MAX;
        for j in 0..729 {
            let d = (deploy_lut[i][0] as i32 - deploy_lut[j][0] as i32).pow(2)
                  + (deploy_lut[i][1] as i32 - deploy_lut[j][1] as i32).pow(2);
            if d < bd { bd = d; best = j; }
        }
        if best == i { ok_deploy += 1; }
    }
    println!("  Deploy LUT round-trip: {}/729", ok_deploy);
    if ok_deploy == 729 { println!("  *** DEPLOY READY ***"); }

    // Summary
    println!("\n=== PIPELINE SUMMARY ===\n");
    println!("  L0 (byte encoder):  raw byte -> char (0-26)    [zero compute]");
    println!("  L1 (2-byte merger): (char_a, char_b) -> merger_lut[a*27+b] -> [int8, int8]");
    println!("                      2048 bytes -> 1024 pairs -> 1024 x 2 = 2048 int8");
    println!("  Deploy: 1458 byte LUT, zero compute");
    println!("  Compression: 4096 -> 2048 int8 (2x)");
    println!("  Conv receptive field: k=3 covers 6 original bytes (was 3)\n");
    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
