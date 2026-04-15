//! L1 Byte Merger — minimum neurons for 2-byte merge (729 combos)
//!
//! Same method as byte encoder: greedy neuron-by-neuron exhaustive search.
//! Input: 2 consecutive bytes → 4 int8 (from LUT_2N)
//! Output: M neurons, sweep until 100% round-trip on all 729 pairs.
//!
//! Run: cargo run --example l1_merger_min_neurons --release

use std::time::Instant;

const LUT: [[i8; 2]; 27] = [
    [-2,-4],[-4,-2],[0,-6],[-2,-5],[-1,-6],[-3,-5],[1,-8],[-1,-7],
    [-2,-6],[-4,-5],[0,-8],[-2,-7],[-1,-8],[-3,-7],[1,-10],[-1,-9],
    [-4,-6],[-6,-5],[-2,-8],[-4,-7],[-3,-8],[-5,-7],[-1,-10],[-3,-9],
    [-4,-8],[-6,-7],[-2,-10],
];

fn c19(x: f32, c: f32) -> f32 {
    let c = c.max(0.1); let l = 6.0 * c;
    if x >= l { return x - l; } if x <= -l { return x + l; }
    let s = x / c; let n = s.floor(); let t = s - n; let h = t * (1.0 - t);
    let sg = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * sg * h
}

fn eval(codes: &Vec<Vec<f32>>, n: usize) -> usize {
    let mut ok = 0;
    for i in 0..729 {
        let mut best = 0; let mut bd = f32::MAX;
        for j in 0..729 {
            let d: f32 = (0..n).map(|k| (codes[i][k] - codes[j][k]).powi(2)).sum();
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

    println!("=== L1 BYTE MERGER — MINIMUM NEURONS ===\n");
    println!("  2 bytes in (4 int8 from LUT) -> M neurons out");
    println!("  729 possible byte-pairs, exhaustive greedy search\n");

    struct WCfg { name: &'static str, range: Vec<i8> }
    let wcfgs = vec![
        WCfg { name: "binary {-1,+1}", range: vec![-1, 1] },
        WCfg { name: "ternary {-1,0,+1}", range: vec![-1, 0, 1] },
        WCfg { name: "2-bit {-2..+2}", range: vec![-2, -1, 0, 1, 2] },
    ];

    let c_vals: Vec<f32> = vec![1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
    let max_neurons = 6;

    for wcfg in &wcfgs {
        let nv = wcfg.range.len();
        let combos = nv.pow(5); // 4 weights + 1 bias

        // C19 activation
        {
            let tc = Instant::now();
            println!("--- {} + C19 ({} combos/neuron x {} c_vals) ---", wcfg.name, combos, c_vals.len());

            let mut codes: Vec<Vec<f32>> = (0..729).map(|_| Vec::new()).collect();

            for ni in 0..max_neurons {
                let mut top_score = 0;
                let mut top_w = [0i8; 4];
                let mut top_b = 0i8;
                let mut top_c = 1.0f32;

                'search: for &cv in &c_vals {
                    for combo in 0..combos {
                        let mut w = [0i8; 4];
                        let mut rem = combo;
                        for j in 0..4 { w[j] = wcfg.range[rem % nv]; rem /= nv; }
                        let b = wcfg.range[rem % nv];

                        let mut test = codes.clone();
                        for (idx, inp) in inputs.iter().enumerate() {
                            let dot = b as f32 + w[0] as f32 * inp[0] + w[1] as f32 * inp[1]
                                + w[2] as f32 * inp[2] + w[3] as f32 * inp[3];
                            test[idx].push(c19(dot, cv));
                        }

                        let score = eval(&test, ni + 1);
                        if score > top_score {
                            top_score = score; top_w = w; top_b = b; top_c = cv;
                            if score == 729 { break 'search; }
                        }
                    }
                }

                // Commit best neuron
                for (idx, inp) in inputs.iter().enumerate() {
                    let dot = top_b as f32 + top_w[0] as f32 * inp[0] + top_w[1] as f32 * inp[1]
                        + top_w[2] as f32 * inp[2] + top_w[3] as f32 * inp[3];
                    codes[idx].push(c19(dot, top_c));
                }

                let m = if top_score == 729 { " ★★★" } else { "" };
                println!("  N={}: {}/729  w={:?} b={:+} c={}{}", ni + 1, top_score, top_w, top_b, top_c, m);

                if top_score == 729 {
                    println!("  -> Minimum: {} neurons ({:.1}s)\n", ni + 1, tc.elapsed().as_secs_f64());
                    break;
                }
            }
            if codes[0].len() == max_neurons {
                println!("  -> No 100% found up to {} neurons ({:.1}s)\n", max_neurons, tc.elapsed().as_secs_f64());
            }
        }

        // Linear (no activation)
        {
            let tc = Instant::now();
            println!("--- {} + linear ({} combos/neuron) ---", wcfg.name, combos);

            let mut codes: Vec<Vec<f32>> = (0..729).map(|_| Vec::new()).collect();

            for ni in 0..max_neurons {
                let mut top_score = 0;
                let mut top_w = [0i8; 4];
                let mut top_b = 0i8;

                for combo in 0..combos {
                    let mut w = [0i8; 4];
                    let mut rem = combo;
                    for j in 0..4 { w[j] = wcfg.range[rem % nv]; rem /= nv; }
                    let b = wcfg.range[rem % nv];

                    let mut test = codes.clone();
                    for (idx, inp) in inputs.iter().enumerate() {
                        let dot = b as f32 + w[0] as f32 * inp[0] + w[1] as f32 * inp[1]
                            + w[2] as f32 * inp[2] + w[3] as f32 * inp[3];
                        test[idx].push(dot);
                    }

                    let score = eval(&test, ni + 1);
                    if score > top_score {
                        top_score = score; top_w = w; top_b = b;
                        if score == 729 { break; }
                    }
                }

                for (idx, inp) in inputs.iter().enumerate() {
                    let dot = top_b as f32 + top_w[0] as f32 * inp[0] + top_w[1] as f32 * inp[1]
                        + top_w[2] as f32 * inp[2] + top_w[3] as f32 * inp[3];
                    codes[idx].push(dot);
                }

                let m = if top_score == 729 { " ★★★" } else { "" };
                println!("  N={}: {}/729  w={:?} b={:+}{}", ni + 1, top_score, top_w, top_b, m);

                if top_score == 729 {
                    println!("  -> Minimum: {} neurons ({:.1}s)\n", ni + 1, tc.elapsed().as_secs_f64());
                    break;
                }
            }
            if codes[0].len() == max_neurons {
                println!("  -> No 100% found up to {} neurons ({:.1}s)\n", max_neurons, tc.elapsed().as_secs_f64());
            }
        }
    }

    // Summary
    println!("=== SUMMARY ===\n");
    println!("  2-byte merger: 4 int8 in -> M values out, 729 combos");
    println!("  Original: 2048 bytes x 2 = 4096 int8");
    println!("  Merged:   1024 units x M = 1024*M int8\n");
    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
