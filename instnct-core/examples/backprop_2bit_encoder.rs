//! Backprop 2-bit encoder — train float, round to {-2,-1,0,+1,+2}
//!
//! Instead of exhaustive search (194s, 2M combos), can backprop find
//! a 2-neuron encoder that rounds to 2-bit and still gets 100%?
//!
//! Method: STE (Straight-Through Estimator)
//!   Forward: round weights to nearest int in {-2..+2}
//!   Backward: gradient flows through as if no rounding
//!
//! Run: cargo run --example backprop_2bit_encoder --release

use std::time::Instant;

fn c19(x: f32, c: f32) -> f32 {
    let c = c.max(0.1); let l = 6.0*c;
    if x >= l { return x-l; } if x <= -l { return x+l; }
    let s = x/c; let n = s.floor(); let t = s-n; let h = t*(1.0-t);
    let sg = if (n as i32)%2==0 { 1.0 } else { -1.0 }; c*sg*h
}

fn round_2bit(x: f32) -> f32 {
    x.round().max(-2.0).min(2.0)
}

fn eval_roundtrip(w: &[[f32;8];2], b: &[f32;2], cs: &[f32;2], quantize: bool) -> usize {
    let codes: Vec<[f32;2]> = (0..27u8).map(|ch| {
        let mut bits = [0.0f32;8]; for i in 0..8 { bits[i] = ((ch>>i)&1) as f32; }
        let mut o = [0.0f32;2];
        for k in 0..2 {
            let bk = if quantize { round_2bit(b[k]) } else { b[k] };
            let mut d = bk;
            for j in 0..8 {
                let wkj = if quantize { round_2bit(w[k][j]) } else { w[k][j] };
                d += wkj * bits[j];
            }
            o[k] = c19(d, cs[k]);
        }
        o
    }).collect();
    let mut ok = 0;
    for i in 0..27 {
        let mut best = 0; let mut bd = f32::MAX;
        for j in 0..27 {
            let d = (codes[i][0]-codes[j][0]).powi(2) + (codes[i][1]-codes[j][1]).powi(2);
            if d < bd { bd = d; best = j; }
        }
        if best == i { ok += 1; }
    }
    ok
}

fn main() {
    let t0 = Instant::now();

    println!("=== BACKPROP 2-BIT ENCODER (STE) ===\n");

    // Try multiple seeds and c values
    let mut best_overall = 0;
    let mut best_config = String::new();

    for &c_val in &[1.0, 2.0, 5.0, 10.0, 20.0] {
        for seed in 0..20u64 {
            // Init random float weights in [-2.5, 2.5]
            let mut w = [[0.0f32;8];2];
            let mut b = [0.0f32;2];
            let cs = [c_val, c_val];

            // Simple LCG for init
            let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            for k in 0..2 {
                for j in 0..8 {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    w[k][j] = ((s >> 33) % 500) as f32 / 100.0 - 2.5;
                }
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                b[k] = ((s >> 33) % 500) as f32 / 100.0 - 2.5;
            }

            // Train: make codes maximally separated
            // Loss: for each byte pair (i,j), push codes apart if i≠j, pull together if i==j
            for epoch in 0..2000 {
                let lr = 0.1 * (1.0 - epoch as f32 / 2000.0 * 0.9);

                // Compute all codes with STE (quantized forward)
                let codes: Vec<[f32;2]> = (0..27u8).map(|ch| {
                    let mut bits = [0.0f32;8]; for i in 0..8 { bits[i] = ((ch>>i)&1) as f32; }
                    let mut o = [0.0f32;2];
                    for k in 0..2 {
                        let mut d = round_2bit(b[k]);
                        for j in 0..8 { d += round_2bit(w[k][j]) * bits[j]; }
                        o[k] = c19(d, cs[k]);
                    }
                    o
                }).collect();

                // Gradient: push apart nearest wrong neighbor
                for i in 0..27u8 {
                    let mut bits_i = [0.0f32;8]; for bi in 0..8 { bits_i[bi] = ((i>>bi)&1) as f32; }

                    // Find nearest WRONG neighbor
                    let mut nearest_j = 0usize; let mut nearest_d = f32::MAX;
                    for j in 0..27 {
                        if j == i as usize { continue; }
                        let d = (codes[i as usize][0]-codes[j][0]).powi(2) + (codes[i as usize][1]-codes[j][1]).powi(2);
                        if d < nearest_d { nearest_d = d; nearest_j = j; }
                    }

                    if nearest_d < 0.5 {
                        // Push apart
                        let mut bits_j = [0.0f32;8]; for bi in 0..8 { bits_j[bi] = ((nearest_j as u8>>bi)&1) as f32; }
                        let diff_bits: Vec<f32> = (0..8).map(|bi| bits_i[bi] - bits_j[bi]).collect();

                        for k in 0..2 {
                            let sign = if codes[i as usize][k] > codes[nearest_j][k] { 1.0 } else { -1.0 };
                            for j in 0..8 {
                                // STE: gradient through round
                                w[k][j] += lr * sign * diff_bits[j] * 0.1;
                            }
                            b[k] += lr * sign * 0.01;
                        }
                    }
                }

                // Clamp to valid range
                for k in 0..2 { for j in 0..8 { w[k][j] = w[k][j].max(-2.5).min(2.5); } b[k] = b[k].max(-2.5).min(2.5); }
            }

            // Eval both float and quantized
            let float_score = eval_roundtrip(&w, &b, &cs, false);
            let quant_score = eval_roundtrip(&w, &b, &cs, true);

            if quant_score > best_overall {
                best_overall = quant_score;
                let qw: Vec<Vec<i8>> = w.iter().map(|row| row.iter().map(|&x| round_2bit(x) as i8).collect()).collect();
                let qb: Vec<i8> = b.iter().map(|&x| round_2bit(x) as i8).collect();
                best_config = format!("seed={} c={} float={}/27 quant={}/27\n  N0: {:?} b={}\n  N1: {:?} b={}",
                    seed, c_val, float_score, quant_score,
                    &qw[0], qb[0], &qw[1], qb[1]);

                if quant_score == 27 {
                    println!("  ★★★ FOUND 100% at seed={} c={} ({:.1}s)", seed, c_val, t0.elapsed().as_secs_f64());
                    println!("  {}", best_config);
                }
            }
        }
    }

    println!("\n  Best result: {}", best_config);

    // Compare with exhaustive search
    println!("\n━━━ COMPARISON ━━━");
    println!("  Exhaustive search: 194.5s → guaranteed 100%");
    println!("  Backprop STE:      {:.1}s → {}/27", t0.elapsed().as_secs_f64(), best_overall);
    if best_overall == 27 {
        println!("  ★★★ Backprop matches exhaustive!");
    } else {
        println!("  Backprop did NOT find 100% — exhaustive still needed");
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
