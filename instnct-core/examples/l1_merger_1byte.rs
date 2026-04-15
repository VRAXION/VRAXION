//! L1 Merger — 1 byte unit: can we compress 2 int8 → 1 value?
//!
//! Input: 1 byte → 2 int8 (from LUT_2N)
//! Output: M neurons — sweep until 100% on all 27 chars
//! If M=1 works: 2048 bytes × 1 = 2048 values (50% compression!)
//! If M=2 needed: no gain (same as LUT output)
//!
//! Run: cargo run --example l1_merger_1byte --release

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

fn main() {
    let t0 = Instant::now();

    // All 27 single-byte inputs (2 int8 each from LUT)
    let inputs: Vec<[f32; 2]> = (0..27u8).map(|ch| {
        [LUT[ch as usize][0] as f32, LUT[ch as usize][1] as f32]
    }).collect();

    println!("=== L1 MERGER — 1 BYTE UNIT ===\n");
    println!("  27 chars, 2 int8 input (from LUT), exhaustive search\n");

    // Weight types
    struct WCfg { name: &'static str, range: Vec<i8> }
    let wcfgs = vec![
        WCfg { name: "binary", range: vec![-1, 1] },
        WCfg { name: "ternary", range: vec![-1, 0, 1] },
        WCfg { name: "2-bit", range: vec![-2, -1, 0, 1, 2] },
        WCfg { name: "3-bit", range: vec![-4,-3,-2,-1,0,1,2,3,4] },
        WCfg { name: "4-bit", range: (-8..=7).collect() },
        WCfg { name: "int8-small", range: (-20..=20).collect() },
        WCfg { name: "int8-med", range: (-50..=50).collect() },
    ];

    let c_vals: Vec<f32> = vec![1.0, 2.0, 5.0, 10.0, 20.0, 50.0];

    fn eval(codes: &Vec<Vec<f32>>, n: usize) -> usize {
        let mut ok = 0;
        for i in 0..27 {
            let mut best = 0; let mut bd = f32::MAX;
            for j in 0..27 {
                let d: f32 = (0..n).map(|k| (codes[i][k] - codes[j][k]).powi(2)).sum();
                if d < bd { bd = d; best = j; }
            }
            if best == i { ok += 1; }
        }
        ok
    }

    for wcfg in &wcfgs {
        let nv = wcfg.range.len();
        let combos = nv.pow(3); // 2 weights + 1 bias

        // C19
        for &use_c19 in &[true, false] {
            let act = if use_c19 { "C19" } else { "linear" };
            let c_list = if use_c19 { &c_vals[..] } else { &[1.0f32][..] };
            let tc = Instant::now();

            let mut codes: Vec<Vec<f32>> = (0..27).map(|_| Vec::new()).collect();
            let mut found = false;

            for ni in 0..4usize {
                let mut top_score = 0;
                let mut top_w = [0i8; 2];
                let mut top_b = 0i8;
                let mut top_c = 1.0f32;

                'search: for &cv in c_list {
                    for combo in 0..combos {
                        let mut rem = combo;
                        let w0 = wcfg.range[rem % nv]; rem /= nv;
                        let w1 = wcfg.range[rem % nv]; rem /= nv;
                        let b = wcfg.range[rem % nv];

                        let mut test = codes.clone();
                        for (ch, inp) in inputs.iter().enumerate() {
                            let dot = b as f32 + w0 as f32 * inp[0] + w1 as f32 * inp[1];
                            let out = if use_c19 { c19(dot, cv, 0.0) } else { dot };
                            test[ch].push(out);
                        }

                        let score = eval(&test, ni + 1);
                        if score > top_score {
                            top_score = score; top_w = [w0, w1]; top_b = b; top_c = cv;
                            if score == 27 { break 'search; }
                        }
                    }
                }

                // Commit
                for (ch, inp) in inputs.iter().enumerate() {
                    let dot = top_b as f32 + top_w[0] as f32 * inp[0] + top_w[1] as f32 * inp[1];
                    let out = if use_c19 { c19(dot, top_c, 0.0) } else { dot };
                    codes[ch].push(out);
                }

                let m = if top_score == 27 { " ***" } else { "" };
                println!("  {:>12}+{:<6} N={}: {}/27  w={:?} b={:+} c={}{}", wcfg.name, act, ni+1, top_score, top_w, top_b, top_c, m);

                if top_score == 27 {
                    found = true;
                    println!("    -> {} + {}: min {} neurons ({:.2}s)", wcfg.name, act, ni+1, tc.elapsed().as_secs_f64());
                    break;
                }
            }
            if !found {
                println!("    -> {} + {}: no 100% up to 4 neurons ({:.2}s)", wcfg.name, act, tc.elapsed().as_secs_f64());
            }
            println!();
        }
    }

    // Summary table
    println!("=== COMPRESSION SUMMARY ===\n");
    println!("  If M=1 works: 2048 bytes x 1 = 2048 values (50% compression)");
    println!("  If M=2 needed: 2048 bytes x 2 = 4096 values (0% - same as LUT)");
    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
