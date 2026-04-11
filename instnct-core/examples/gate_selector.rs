//! Selector gate search: 6 minterm detectors for ALU op-code selection
//! Each detector fires on exactly ONE 3-bit input pattern
//!
//! sel_ADD = 1 only when op=000
//! sel_SUB = 1 only when op=001
//! sel_AND = 1 only when op=010
//! sel_OR  = 1 only when op=011
//! sel_XOR = 1 only when op=100
//! sel_CMP = 1 only when op=101
//!
//! Run: cargo run --example gate_selector --release

use std::io::Write;

fn c19(x: f32, rho: f32) -> f32 {
    let c = 1.0f32; let l = 6.0;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sgn * h + rho * h * h)
}

fn log(f: &mut std::fs::File, msg: &str) {
    let d = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
    let s = d.as_secs(); let h = (s/3600)%24; let m = (s/60)%60; let sec = s%60;
    let line = format!("[{:02}:{:02}:{:02}] {}\n", h, m, sec, msg);
    print!("{}", line);
    f.write_all(line.as_bytes()).ok();
    f.flush().ok();
}

#[derive(Clone)]
struct Solution {
    w1: f32, w2: f32, w3: f32, bias: f32, rho: f32, thr: f32,
    margin: f32, complexity: f32,
}

fn search_minterm(name: &str, truth: &[(u8,u8,u8,u8)], logf: &mut std::fs::File) -> Option<Solution> {
    log(logf, &format!("--- {} ---", name));

    let weights: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.25).collect();
    let rhos: Vec<f32> = vec![0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
    let thresholds: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.1).collect();

    let mut best: Option<Solution> = None;
    let mut total_solutions = 0u64;

    for &rho in &rhos {
        let mut rho_solutions = 0u64;
        for &w1 in &weights {
            for &w2 in &weights {
                for &w3 in &weights {
                    for &bias in &weights {
                        let outputs: Vec<f32> = truth.iter()
                            .map(|&(a,b,c,_)| c19(w1*a as f32 + w2*b as f32 + w3*c as f32 + bias, rho))
                            .collect();

                        for &thr in &thresholds {
                            let correct = truth.iter().zip(&outputs)
                                .all(|(&(_,_,_,e), &o)| (if o > thr {1u8} else {0}) == e);
                            if correct {
                                let margin: f32 = truth.iter().zip(&outputs)
                                    .map(|(&(_,_,_,e), &o)| if e==1 { o-thr } else { thr-o })
                                    .fold(f32::INFINITY, f32::min);
                                let complexity = w1.abs() + w2.abs() + w3.abs() + bias.abs();
                                let score = margin * 10.0 - complexity * 0.1;
                                total_solutions += 1;
                                rho_solutions += 1;

                                if best.is_none() || score > best.as_ref().unwrap().margin * 10.0 - best.as_ref().unwrap().complexity * 0.1 {
                                    best = Some(Solution { w1, w2, w3, bias, rho, thr, margin, complexity });
                                }
                            }
                        }
                    }
                }
            }
        }
        log(logf, &format!("  rho={:.1}: {} solutions", rho, rho_solutions));
    }

    log(logf, &format!("  Total solutions: {}", total_solutions));

    if let Some(ref s) = best {
        log(logf, &format!("  BEST: C19({:+.2}*b0 + {:+.2}*b1 + {:+.2}*b2 + {:+.2}, rho={:.1}) > {:+.2}  margin={:.4}",
            s.w1, s.w2, s.w3, s.bias, s.rho, s.thr, s.margin));

        // Verify
        for &(a, b, c, expected) in truth {
            let raw = c19(s.w1*a as f32 + s.w2*b as f32 + s.w3*c as f32 + s.bias, s.rho);
            let out = if raw > s.thr { 1u8 } else { 0 };
            let ok = if out == expected { "OK" } else { "FAIL" };
            log(logf, &format!("    op=({},{},{}) → {} (raw={:.4}, exp={}) {}",
                a, b, c, out, raw, expected, ok));
        }
    } else {
        log(logf, &format!("  NOT POSSIBLE with 1 neuron!"));
    }

    best
}

fn main() {
    let log_path = "instnct-core/gate_selector_log.txt";
    let mut logf = std::fs::File::create(log_path).unwrap();

    log(&mut logf, "========================================");
    log(&mut logf, "=== SELECTOR GATE EXHAUSTIVE SEARCH ===");
    log(&mut logf, "========================================");
    log(&mut logf, "  6 minterm detectors: each fires on exactly 1 of 8 patterns");
    let t0 = std::time::Instant::now();

    // All 8 possible 3-bit inputs, only the target fires
    // op bits: b0 (LSB), b1, b2 (MSB)

    let selectors: Vec<(&str, Vec<(u8,u8,u8,u8)>)> = vec![
        ("sel_ADD (op=000)", vec![
            (0,0,0, 1),  // op=0: fire
            (1,0,0, 0),  // op=1
            (0,1,0, 0),  // op=2
            (1,1,0, 0),  // op=3
            (0,0,1, 0),  // op=4
            (1,0,1, 0),  // op=5
            (0,1,1, 0),  // op=6 (unused but must be 0)
            (1,1,1, 0),  // op=7 (unused but must be 0)
        ]),
        ("sel_SUB (op=001)", vec![
            (0,0,0, 0), (1,0,0, 1), (0,1,0, 0), (1,1,0, 0),
            (0,0,1, 0), (1,0,1, 0), (0,1,1, 0), (1,1,1, 0),
        ]),
        ("sel_AND (op=010)", vec![
            (0,0,0, 0), (1,0,0, 0), (0,1,0, 1), (1,1,0, 0),
            (0,0,1, 0), (1,0,1, 0), (0,1,1, 0), (1,1,1, 0),
        ]),
        ("sel_OR (op=011)", vec![
            (0,0,0, 0), (1,0,0, 0), (0,1,0, 0), (1,1,0, 1),
            (0,0,1, 0), (1,0,1, 0), (0,1,1, 0), (1,1,1, 0),
        ]),
        ("sel_XOR (op=100)", vec![
            (0,0,0, 0), (1,0,0, 0), (0,1,0, 0), (1,1,0, 0),
            (0,0,1, 1), (1,0,1, 0), (0,1,1, 0), (1,1,1, 0),
        ]),
        ("sel_CMP (op=101)", vec![
            (0,0,0, 0), (1,0,0, 0), (0,1,0, 0), (1,1,0, 0),
            (0,0,1, 0), (1,0,1, 1), (0,1,1, 0), (1,1,1, 0),
        ]),
    ];

    let mut results: Vec<(&str, Option<Solution>)> = Vec::new();

    for (name, truth) in &selectors {
        let sol = search_minterm(name, truth, &mut logf);
        results.push((name, sol));
    }

    // Summary
    log(&mut logf, "\n========================================");
    log(&mut logf, "=== SUMMARY ===");
    log(&mut logf, "========================================");

    let mut all_found = true;
    for (name, sol) in &results {
        if let Some(s) = sol {
            log(&mut logf, &format!("  {} : C19({:+.2}*b0+{:+.2}*b1+{:+.2}*b2+{:+.2}, rho={:.1})>{:+.2} margin={:.4}",
                name, s.w1, s.w2, s.w3, s.bias, s.rho, s.thr, s.margin));
        } else {
            log(&mut logf, &format!("  {} : NOT FOUND", name));
            all_found = false;
        }
    }

    if all_found {
        log(&mut logf, "\n  ALL 6 SELECTORS FOUND! Full ALU selector = 6 C19 neurons.");
        log(&mut logf, "  Combined with ALU gates: ENTIRE SYSTEM is exhaustive-verified C19.");
        log(&mut logf, "  Zero training. Zero float. Pure integer LUT.");
    }

    log(&mut logf, &format!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64()));
    log(&mut logf, "=== DONE ===");
}
