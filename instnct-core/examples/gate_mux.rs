//! 3-input gate exhaustive search with file logging + checkpoints
//! Gates: MUX, AND3, OR3, NOR3, XOR3, MAJ
//!
//! Run: cargo run --example gate_mux --release
//!
//! Output:  instnct-core/gate_mux_log.txt   (live progress)
//!          instnct-core/gate_mux_ckpt.bin  (checkpoint per gate)

use std::io::Write;

fn c19(x: f32, rho: f32) -> f32 {
    let c = 1.0f32; let l = 6.0;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sgn * h + rho * h * h)
}

// ============================================================

#[derive(Clone)]
struct Solution3 {
    w1: f32, w2: f32, w3: f32, bias: f32, rho: f32, thr: f32,
    margin: f32, complexity: f32,
}

struct GateResult {
    name: String,
    total_solutions: usize,
    best_margin: Option<Solution3>,
    best_simple: Option<Solution3>,
    best_overall: Option<Solution3>,
}

fn log(f: &mut std::fs::File, msg: &str) {
    let now = chrono_now();
    let line = format!("[{}] {}\n", now, msg);
    print!("{}", line);
    f.write_all(line.as_bytes()).ok();
    f.flush().ok();
}

fn chrono_now() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap();
    let secs = d.as_secs();
    let h = (secs / 3600) % 24;
    let m = (secs / 60) % 60;
    let s = secs % 60;
    format!("{:02}:{:02}:{:02}", h, m, s)
}

// Keep only top-K solutions by score to avoid OOM
const TOP_K: usize = 100;

fn insert_top(top: &mut Vec<Solution3>, s: Solution3, score_fn: impl Fn(&Solution3) -> f32) {
    let sc = score_fn(&s);
    if top.len() < TOP_K {
        top.push(s);
        top.sort_by(|a, b| score_fn(b).partial_cmp(&score_fn(a)).unwrap());
    } else if sc > score_fn(top.last().unwrap()) {
        *top.last_mut().unwrap() = s;
        top.sort_by(|a, b| score_fn(b).partial_cmp(&score_fn(a)).unwrap());
    }
}

fn search_3input(name: &str, truth: &[(u8,u8,u8,u8)], logf: &mut std::fs::File) -> GateResult {
    log(logf, &format!("=== {} START ===", name));
    log(logf, &format!("  Truth: {:?}", truth));

    let weights: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.25).collect();
    let rhos: Vec<f32> = vec![0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
    let thresholds: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.1).collect();

    // Only keep top-K by combined score (margin*10 - complexity*0.1)
    let mut top_solutions: Vec<Solution3> = Vec::with_capacity(TOP_K + 1);
    let mut total_solutions = 0u64;
    let mut checked = 0u64;
    let total_weight_combos = (weights.len() as u64).pow(4);
    let total_per_rho = total_weight_combos;
    let gate_start = std::time::Instant::now();

    for (ri, &rho) in rhos.iter().enumerate() {
        let rho_start = std::time::Instant::now();
        let mut rho_solutions = 0u64;

        for (wi, &w1) in weights.iter().enumerate() {
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
                                let s = Solution3 { w1, w2, w3, bias, rho, thr, margin, complexity };
                                insert_top(&mut top_solutions, s, |s| s.margin * 10.0 - s.complexity * 0.1);
                                total_solutions += 1;
                                rho_solutions += 1;
                            }
                        }
                        checked += 1;
                    }
                }
            }
            // Progress every 10 w1 steps
            if wi % 10 == 0 {
                let pct = (wi as f64 / weights.len() as f64) * 100.0;
                let elapsed = rho_start.elapsed().as_secs_f64();
                let eta = if pct > 0.0 { elapsed * (100.0 - pct) / pct } else { 0.0 };
                log(logf, &format!("  {} rho={:.1} [{}/{}] {:.0}% | {} sol | ETA {:.0}s",
                    name, rho, ri+1, rhos.len(), pct, rho_solutions, eta));
            }
        }

        let rho_time = rho_start.elapsed().as_secs_f64();
        log(logf, &format!("  {} rho={:.1} DONE in {:.1}s | {} solutions this rho | {} total",
            name, rho, rho_time, rho_solutions, total_solutions));
    }

    let gate_time = gate_start.elapsed().as_secs_f64();
    log(logf, &format!("  {} COMPLETE: {} combos checked, {} solutions found, {:.1}s",
        name, checked, total_solutions, gate_time));

    if total_solutions == 0 {
        log(logf, &format!("  {} => NOT POSSIBLE with 1 neuron!", name));
        return GateResult {
            name: name.to_string(),
            total_solutions: 0,
            best_margin: None,
            best_simple: None,
            best_overall: None,
        };
    }

    // top_solutions is sorted by combined score (best first)
    let best_overall = top_solutions[0].clone();

    // Find best by margin alone
    let mut by_margin = top_solutions.clone();
    by_margin.sort_by(|a,b| b.margin.partial_cmp(&a.margin).unwrap());
    let best_margin = by_margin[0].clone();

    // Find best by simplicity alone
    let mut by_simple = top_solutions.clone();
    by_simple.sort_by(|a,b| a.complexity.partial_cmp(&b.complexity).unwrap());
    let best_simple = by_simple[0].clone();

    // Log top 3 each
    log(logf, &format!("  {} TOP 3 by MARGIN:", name));
    for (i, s) in by_margin.iter().take(3).enumerate() {
        log(logf, &format!("    #{}: C19({:+.2}*x1+{:+.2}*x2+{:+.2}*x3+{:+.2}, rho={:.1})>{:+.2} margin={:.4}",
            i+1, s.w1, s.w2, s.w3, s.bias, s.rho, s.thr, s.margin));
    }
    log(logf, &format!("  {} TOP 3 by SIMPLICITY:", name));
    for (i, s) in by_simple.iter().take(3).enumerate() {
        log(logf, &format!("    #{}: C19({:+.2}*x1+{:+.2}*x2+{:+.2}*x3+{:+.2}, rho={:.1})>{:+.2} margin={:.4} cmplx={:.2}",
            i+1, s.w1, s.w2, s.w3, s.bias, s.rho, s.thr, s.margin, s.complexity));
    }
    log(logf, &format!("  {} BEST OVERALL: C19({:+.2}*x1+{:+.2}*x2+{:+.2}*x3+{:+.2}, rho={:.1})>{:+.2} margin={:.4}",
        name, best_overall.w1, best_overall.w2, best_overall.w3, best_overall.bias,
        best_overall.rho, best_overall.thr, best_overall.margin));

    // Verify best
    log(logf, &format!("  {} VERIFY:", name));
    for &(a, b, c, expected) in truth {
        let raw = c19(best_overall.w1*a as f32 + best_overall.w2*b as f32
                      + best_overall.w3*c as f32 + best_overall.bias, best_overall.rho);
        let out = if raw > best_overall.thr { 1u8 } else { 0 };
        let ok = if out == expected { "OK" } else { "FAIL" };
        log(logf, &format!("    {}({},{},{}) = {} (raw={:.4}, exp={}) {}",
            name, a, b, c, out, raw, expected, ok));
    }

    log(logf, &format!("=== {} END ===\n", name));

    GateResult {
        name: name.to_string(),
        total_solutions: total_solutions as usize,
        best_margin: Some(best_margin),
        best_simple: Some(best_simple),
        best_overall: Some(best_overall),
    }
}

// Checkpoint: save completed gate results
fn save_checkpoint(path: &str, results: &[GateResult]) {
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(b"GMX1").unwrap(); // magic
    f.write_all(&(results.len() as u32).to_le_bytes()).unwrap();
    for r in results {
        let name_bytes = r.name.as_bytes();
        f.write_all(&(name_bytes.len() as u16).to_le_bytes()).unwrap();
        f.write_all(name_bytes).unwrap();
        f.write_all(&(r.total_solutions as u64).to_le_bytes()).unwrap();
        let has_best = r.best_overall.is_some() as u8;
        f.write_all(&[has_best]).unwrap();
        if let Some(ref s) = r.best_overall {
            for v in &[s.w1, s.w2, s.w3, s.bias, s.rho, s.thr, s.margin, s.complexity] {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
    }
}

fn load_checkpoint(path: &str) -> Vec<(String, u64, Option<[f32; 8]>)> {
    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(_) => return vec![],
    };
    if data.len() < 8 || &data[0..4] != b"GMX1" { return vec![]; }
    let count = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    let mut off = 8;
    let mut out = Vec::new();
    for _ in 0..count {
        let nlen = u16::from_le_bytes(data[off..off+2].try_into().unwrap()) as usize;
        off += 2;
        let name = String::from_utf8_lossy(&data[off..off+nlen]).to_string();
        off += nlen;
        let total = u64::from_le_bytes(data[off..off+8].try_into().unwrap());
        off += 8;
        let has = data[off]; off += 1;
        let best = if has == 1 {
            let mut vals = [0f32; 8];
            for v in &mut vals {
                *v = f32::from_le_bytes(data[off..off+4].try_into().unwrap());
                off += 4;
            }
            Some(vals)
        } else { None };
        out.push((name, total, best));
    }
    out
}

fn main() {
    let log_path = "instnct-core/gate_mux_log.txt";
    let ckpt_path = "instnct-core/gate_mux_ckpt.bin";

    let mut logf = std::fs::OpenOptions::new()
        .create(true).append(true)
        .open(log_path).unwrap();

    log(&mut logf, "========================================");
    log(&mut logf, "=== 3-INPUT GATE EXHAUSTIVE SEARCH ===");
    log(&mut logf, "========================================");
    let t0 = std::time::Instant::now();

    // Check checkpoint for already completed gates
    let completed = load_checkpoint(ckpt_path);
    let completed_names: Vec<String> = completed.iter().map(|(n,_,_)| n.clone()).collect();
    log(&mut logf, &format!("Checkpoint: {} gates already done: {:?}", completed.len(), completed_names));

    // Gate definitions — order: easy first, critical last
    let gates: Vec<(&str, Vec<(u8,u8,u8,u8)>)> = vec![
        ("AND3", vec![
            (0,0,0, 0), (0,0,1, 0), (0,1,0, 0), (0,1,1, 0),
            (1,0,0, 0), (1,0,1, 0), (1,1,0, 0), (1,1,1, 1),
        ]),
        ("OR3", vec![
            (0,0,0, 0), (0,0,1, 1), (0,1,0, 1), (0,1,1, 1),
            (1,0,0, 1), (1,0,1, 1), (1,1,0, 1), (1,1,1, 1),
        ]),
        ("NOR3", vec![
            (0,0,0, 1), (0,0,1, 0), (0,1,0, 0), (0,1,1, 0),
            (1,0,0, 0), (1,0,1, 0), (1,1,0, 0), (1,1,1, 0),
        ]),
        ("MUX", vec![
            (0, 0, 0, 0), (0, 0, 1, 1), (0, 1, 0, 0), (0, 1, 1, 1),
            (1, 0, 0, 0), (1, 0, 1, 0), (1, 1, 0, 1), (1, 1, 1, 1),
        ]),
        ("XOR3", vec![  // CRITICAL: full adder SUM
            (0,0,0, 0), (0,0,1, 1), (0,1,0, 1), (0,1,1, 0),
            (1,0,0, 1), (1,0,1, 0), (1,1,0, 0), (1,1,1, 1),
        ]),
        ("MAJ", vec![   // CRITICAL: full adder CARRY
            (0,0,0, 0), (0,0,1, 0), (0,1,0, 0), (0,1,1, 1),
            (1,0,0, 0), (1,0,1, 1), (1,1,0, 1), (1,1,1, 1),
        ]),
    ];

    let mut results: Vec<GateResult> = Vec::new();

    for (name, truth) in &gates {
        if completed_names.contains(&name.to_string()) {
            log(&mut logf, &format!("SKIP {} (already in checkpoint)", name));
            // Reconstruct GateResult from checkpoint
            let (_, total, best) = completed.iter().find(|(n,_,_)| n == name).unwrap();
            results.push(GateResult {
                name: name.to_string(),
                total_solutions: *total as usize,
                best_margin: None,
                best_simple: None,
                best_overall: best.map(|v| Solution3 {
                    w1: v[0], w2: v[1], w3: v[2], bias: v[3],
                    rho: v[4], thr: v[5], margin: v[6], complexity: v[7],
                }),
            });
            continue;
        }

        let r = search_3input(name, truth, &mut logf);
        results.push(r);

        // Save checkpoint after each gate
        save_checkpoint(ckpt_path, &results);
        log(&mut logf, &format!("Checkpoint saved ({} gates done)", results.len()));
    }

    // Final summary
    log(&mut logf, "");
    log(&mut logf, "===========================================");
    log(&mut logf, "           FINAL SUMMARY");
    log(&mut logf, "===========================================");
    for r in &results {
        let status = if r.total_solutions > 0 {
            format!("{} solutions", r.total_solutions)
        } else {
            "NOT POSSIBLE".to_string()
        };
        let best_str = if let Some(ref s) = r.best_overall {
            format!("C19({:+.2}*x1+{:+.2}*x2+{:+.2}*x3+{:+.2}, rho={:.1})>{:+.2} margin={:.4}",
                s.w1, s.w2, s.w3, s.bias, s.rho, s.thr, s.margin)
        } else {
            "N/A".to_string()
        };
        log(&mut logf, &format!("  {:6} | {:>12} | {}", r.name, status, best_str));
    }

    // The big question
    let xor3_ok = results.iter().find(|r| r.name == "XOR3").map(|r| r.total_solutions > 0).unwrap_or(false);
    let maj_ok = results.iter().find(|r| r.name == "MAJ").map(|r| r.total_solutions > 0).unwrap_or(false);

    log(&mut logf, "");
    if xor3_ok && maj_ok {
        log(&mut logf, ">>> XOR3 + MAJ BOTH WORK! Full adder = 2 neurons! <<<");
        log(&mut logf, ">>> 4-bit adder = 8 neurons (was 20) <<<");
        log(&mut logf, ">>> 8-bit adder = 16 neurons (was 40) <<<");
    } else {
        log(&mut logf, &format!(">>> XOR3: {}  MAJ: {} <<<",
            if xor3_ok { "YES" } else { "NO" },
            if maj_ok { "YES" } else { "NO" }));
        if !xor3_ok || !maj_ok {
            log(&mut logf, ">>> Full adder stays at 5 neurons <<<");
        }
    }

    log(&mut logf, &format!("\nTotal time: {:.1}s", t0.elapsed().as_secs_f64()));
    log(&mut logf, "=== ALL DONE ===");
}
