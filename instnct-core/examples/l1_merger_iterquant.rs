//! L1 Merger — iterative quantization
//!
//! 1. Train float until 729/729
//! 2. Find param closest to integer → round & freeze
//! 3. Re-train remaining float params
//! 4. Repeat until all params frozen
//! 5. Scale C19 outputs to int8, verify 100%
//!
//! Run: cargo run --example l1_merger_iterquant --release

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

// 10 params: w[0][0..4], b[0], w[1][0..4], b[1]
fn get_param(w: &[[f32;4];2], b: &[f32;2], idx: usize) -> f32 {
    if idx < 4 { w[0][idx] }
    else if idx == 4 { b[0] }
    else if idx < 9 { w[1][idx - 5] }
    else { b[1] }
}

fn set_param(w: &mut [[f32;4];2], b: &mut [f32;2], idx: usize, val: f32) {
    if idx < 4 { w[0][idx] = val; }
    else if idx == 4 { b[0] = val; }
    else if idx < 9 { w[1][idx - 5] = val; }
    else { b[1] = val; }
}

fn param_name(idx: usize) -> String {
    if idx < 4 { format!("w0[{}]", idx) }
    else if idx == 4 { "b0".into() }
    else if idx < 9 { format!("w1[{}]", idx - 5) }
    else { "b1".into() }
}

fn eval_score(codes: &[[f32;2]; 729]) -> usize {
    let mut ok = 0;
    for i in 0..729 {
        let mut best = 0; let mut bd = f32::MAX;
        for j in 0..729 {
            let d = (codes[i][0]-codes[j][0]).powi(2) + (codes[i][1]-codes[j][1]).powi(2);
            if d < bd { bd = d; best = j; }
        }
        if best == i { ok += 1; }
    }
    ok
}

fn compute_codes(w: &[[f32;4];2], b: &[f32;2], cv: &[f32;2], rv: &[f32;2],
                 inputs: &[[f32;4]; 729]) -> [[ f32;2]; 729] {
    let mut codes = [[0.0f32; 2]; 729];
    for (idx, inp) in inputs.iter().enumerate() {
        for k in 0..2 {
            let dot = b[k] + w[k][0]*inp[0] + w[k][1]*inp[1] + w[k][2]*inp[2] + w[k][3]*inp[3];
            codes[idx][k] = c19(dot, cv[k], rv[k]);
        }
    }
    codes
}

fn main() {
    let t0 = Instant::now();

    let mut inputs = [[0.0f32; 4]; 729];
    for a in 0..27u8 {
        for b in 0..27u8 {
            let idx = a as usize * 27 + b as usize;
            inputs[idx] = [LUT[a as usize][0] as f32, LUT[a as usize][1] as f32,
                           LUT[b as usize][0] as f32, LUT[b as usize][1] as f32];
        }
    }

    println!("=== L1 MERGER — ITERATIVE QUANTIZATION ===\n");
    println!("  Train float → freeze params one by one → re-train → int8 deploy\n");

    let mut best_overall_quant = 0;

    for seed in 0..50u64 {
        let mut rng = Rng::new(seed * 7919 + 42);
        let mut w = [[0.0f32; 4]; 2];
        let mut b = [0.0f32; 2];
        let mut cv = [0.0f32; 2];
        let mut rv = [0.0f32; 2];

        for k in 0..2 {
            for j in 0..4 { w[k][j] = rng.uniform(-15.0, 15.0); }
            b[k] = rng.uniform(-15.0, 15.0);
            cv[k] = rng.uniform(10.0, 60.0);
            rv[k] = rng.uniform(0.0, 2.0);
        }

        let mut frozen = [false; 10];
        let n_params = 10;

        // ── Phase 1: initial float training ──
        for ep in 0..3000 {
            let lr = 0.02 * (1.0 - ep as f32 / 3000.0 * 0.8);
            let codes = compute_codes(&w, &b, &cv, &rv, &inputs);

            let mut any_active = false;
            for i in 0..729 {
                let mut nj = 0; let mut nd = f32::MAX;
                for j in 0..729 { if j==i{continue;}
                    let d=(codes[i][0]-codes[j][0]).powi(2)+(codes[i][1]-codes[j][1]).powi(2);
                    if d<nd{nd=d;nj=j;}}

                if nd < 2.0 {
                    any_active = true;
                    for k in 0..2 {
                        let sign = if codes[i][k]>codes[nj][k]{1.0}else if codes[i][k]<codes[nj][k]{-1.0}else{0.0};
                        let dot_i = b[k]+w[k][0]*inputs[i][0]+w[k][1]*inputs[i][1]+w[k][2]*inputs[i][2]+w[k][3]*inputs[i][3];
                        let gi = c19_dx(dot_i, cv[k], rv[k]);

                        if !frozen[k*5+4] { b[k] += lr * sign * gi * 0.01; }
                        for j in 0..4 {
                            if !frozen[k*5+j] { w[k][j] += lr * sign * gi * inputs[i][j] * 0.01; }
                        }
                        // c/rho always trainable
                        let eps = 0.01;
                        let dc = (c19(dot_i,cv[k]+eps,rv[k])-c19(dot_i,cv[k]-eps,rv[k]))/(2.0*eps);
                        cv[k] += lr * sign * dc * 0.001;
                        cv[k] = cv[k].max(1.0).min(100.0);
                        let dr = (c19(dot_i,cv[k],rv[k]+eps)-c19(dot_i,cv[k],rv[k]-eps))/(2.0*eps);
                        rv[k] += lr * sign * dr * 0.001;
                        rv[k] = rv[k].max(0.0).min(5.0);
                    }
                }
            }
            if !any_active { break; }
        }

        let codes = compute_codes(&w, &b, &cv, &rv, &inputs);
        let float_score = eval_score(&codes);
        if float_score < 729 { continue; } // skip seeds that don't converge

        // ── Phase 2: iterative quantization ──
        let mut n_frozen = 0;
        while n_frozen < n_params {
            // Find unfrozen param closest to integer
            let mut best_idx = 0;
            let mut best_dist = f32::MAX;
            for p in 0..n_params {
                if frozen[p] { continue; }
                let v = get_param(&w, &b, p);
                let dist = (v - v.round()).abs();
                if dist < best_dist { best_dist = dist; best_idx = p; }
            }

            // Freeze it
            let old_val = get_param(&w, &b, best_idx);
            let new_val = old_val.round().max(-50.0).min(50.0);
            set_param(&mut w, &mut b, best_idx, new_val);
            frozen[best_idx] = true;
            n_frozen += 1;

            // Re-train remaining params
            for ep in 0..1000 {
                let lr = 0.01 * (1.0 - ep as f32 / 1000.0 * 0.8);
                let codes = compute_codes(&w, &b, &cv, &rv, &inputs);
                let mut any_active = false;

                for i in 0..729 {
                    let mut nj = 0; let mut nd = f32::MAX;
                    for j in 0..729 { if j==i{continue;}
                        let d=(codes[i][0]-codes[j][0]).powi(2)+(codes[i][1]-codes[j][1]).powi(2);
                        if d<nd{nd=d;nj=j;}}

                    if nd < 2.0 {
                        any_active = true;
                        for k in 0..2 {
                            let sign = if codes[i][k]>codes[nj][k]{1.0}else if codes[i][k]<codes[nj][k]{-1.0}else{0.0};
                            let dot_i = b[k]+w[k][0]*inputs[i][0]+w[k][1]*inputs[i][1]+w[k][2]*inputs[i][2]+w[k][3]*inputs[i][3];
                            let gi = c19_dx(dot_i, cv[k], rv[k]);

                            if !frozen[k*5+4] { b[k] += lr * sign * gi * 0.01; }
                            for j in 0..4 {
                                if !frozen[k*5+j] { w[k][j] += lr * sign * gi * inputs[i][j] * 0.01; }
                            }
                            let eps = 0.01;
                            let dc = (c19(dot_i,cv[k]+eps,rv[k])-c19(dot_i,cv[k]-eps,rv[k]))/(2.0*eps);
                            cv[k] += lr * sign * dc * 0.001;
                            cv[k] = cv[k].max(1.0).min(100.0);
                            let dr = (c19(dot_i,cv[k],rv[k]+eps)-c19(dot_i,cv[k],rv[k]-eps))/(2.0*eps);
                            rv[k] += lr * sign * dr * 0.001;
                            rv[k] = rv[k].max(0.0).min(5.0);
                        }
                    }
                }
                if !any_active { break; }
            }

            let codes = compute_codes(&w, &b, &cv, &rv, &inputs);
            let score = eval_score(&codes);
            if score < 729 {
                // This freeze broke it — but we can't unfreeze, so note it
                print!("!");
            }
        }

        // ── Phase 3: all weights/biases frozen as int, check codes ──
        let codes = compute_codes(&w, &b, &cv, &rv, &inputs);
        let final_float = eval_score(&codes);

        // Quantize C19 outputs to int8
        let mut min0=f32::MAX;let mut max0=f32::MIN;let mut min1=f32::MAX;let mut max1=f32::MIN;
        for c in &codes {
            if c[0]<min0{min0=c[0];} if c[0]>max0{max0=c[0];}
            if c[1]<min1{min1=c[1];} if c[1]>max1{max1=c[1];}
        }
        let s0 = if max0-min0>0.0 { 254.0/(max0-min0) } else { 1.0 };
        let s1 = if max1-min1>0.0 { 254.0/(max1-min1) } else { 1.0 };

        let qlut: Vec<[i8;2]> = codes.iter().map(|c| {
            [((c[0]-min0)*s0-127.0).round().max(-128.0).min(127.0) as i8,
             ((c[1]-min1)*s1-127.0).round().max(-128.0).min(127.0) as i8]
        }).collect();

        let mut unique = std::collections::HashSet::new();
        for q in &qlut { unique.insert((q[0],q[1])); }

        let mut ok_q = 0;
        for i in 0..729 {
            let mut best=0;let mut bd=i32::MAX;
            for j in 0..729 {
                let d=(qlut[i][0] as i32-qlut[j][0] as i32).pow(2)+(qlut[i][1] as i32-qlut[j][1] as i32).pow(2);
                if d<bd{bd=d;best=j;}}
            if best==i{ok_q+=1;}
        }

        if ok_q > best_overall_quant { best_overall_quant = ok_q; }

        let iw0: Vec<i8> = w[0].iter().map(|&v| v as i8).collect();
        let iw1: Vec<i8> = w[1].iter().map(|&v| v as i8).collect();

        println!("\n  seed={}: float={}/729, int8_out={}/729 ({} unique pairs)",
            seed, final_float, ok_q, unique.len());
        println!("    N0: w={:?} b={} c={:.1} rho={:.1}", iw0, b[0] as i8, cv[0], rv[0]);
        println!("    N1: w={:?} b={} c={:.1} rho={:.1}", iw1, b[1] as i8, cv[1], rv[1]);

        if ok_q == 729 {
            println!("\n  *** 100% INT8 OUTPUT — FULLY QUANTIZED ***");
            println!("  Deploy: 729 x 2 = 1458 byte LUT, zero compute");
            break;
        }
    }

    if best_overall_quant < 729 {
        println!("\n  Best int8 output: {}/729", best_overall_quant);
        println!("  Fallback: grid assignment (trivially 729/729)");
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
