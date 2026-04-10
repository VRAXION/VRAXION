//! Gate Exhaustive: FULL ranked search for every logic gate
//! Shows: total solutions, best margin, simplest, top 5 for each gate
//!
//! Run: cargo run --example gate_exhaustive --release

fn c19(x: f32, rho: f32) -> f32 {
    let c = 1.0f32; let l = 6.0;
    if x >= l { return x - l; } if x <= -l { return x + l; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sgn * h + rho * h * h)
}

#[derive(Clone)]
struct Solution {
    w1: f32, w2: f32, bias: f32, rho: f32, thr: f32,
    margin: f32,     // minimum distance from threshold (robustness)
    complexity: f32, // |w1|+|w2|+|bias| (simplicity score)
}

fn full_search(name: &str, truth: &[(u8,u8,u8)]) -> Vec<Solution> {
    let weights: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.25).collect(); // 81 values
    let rhos: Vec<f32> = vec![0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
    let thresholds: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.1).collect(); // 81 values

    let mut solutions = Vec::new();
    let mut checked = 0u64;

    for &rho in &rhos {
        for &w1 in &weights {
            for &w2 in &weights {
                for &bias in &weights {
                    let outputs: Vec<f32> = truth.iter()
                        .map(|&(a,b,_)| c19(w1*a as f32 + w2*b as f32 + bias, rho))
                        .collect();

                    for &thr in &thresholds {
                        let correct = truth.iter().zip(&outputs)
                            .all(|(&(_,_,e), &o)| (if o > thr {1u8} else {0}) == e);
                        if correct {
                            let margin: f32 = truth.iter().zip(&outputs)
                                .map(|(&(_,_,e), &o)| if e==1 { o-thr } else { thr-o })
                                .fold(f32::INFINITY, f32::min);
                            let complexity = w1.abs() + w2.abs() + bias.abs();
                            solutions.push(Solution { w1, w2, bias, rho, thr, margin, complexity });
                        }
                    }
                    checked += 1;
                }
            }
        }
    }
    println!("  Searched {} weight combos × {} thresholds = {} total evaluations",
             checked, thresholds.len(), checked * thresholds.len() as u64);
    solutions
}

fn print_solution(s: &Solution, label: &str) {
    println!("    {}: C19({:+.2}*a + {:+.2}*b + {:+.2}, rho={:.1}) > {:+.2}  margin={:.4} complexity={:.2}",
             label, s.w1, s.w2, s.bias, s.rho, s.thr, s.margin, s.complexity);
}

fn verify(s: &Solution, truth: &[(u8,u8,u8)], name: &str) {
    for &(a, b, expected) in truth {
        let raw = c19(s.w1*a as f32 + s.w2*b as f32 + s.bias, s.rho);
        let out = if raw > s.thr { 1u8 } else { 0 };
        print!("  {}({},{})={}", name, a, b, out);
        if out != expected { print!("✗"); }
    }
    println!();
}

fn main() {
    println!("=== GATE EXHAUSTIVE: full ranked search ===\n");

    let gates: Vec<(&str, Vec<(u8,u8,u8)>)> = vec![
        ("NOT",  vec![(0,0,1),(1,0,0)]),
        ("AND",  vec![(0,0,0),(0,1,0),(1,0,0),(1,1,1)]),
        ("OR",   vec![(0,0,0),(0,1,1),(1,0,1),(1,1,1)]),
        ("NAND", vec![(0,0,1),(0,1,1),(1,0,1),(1,1,0)]),
        ("NOR",  vec![(0,0,1),(0,1,0),(1,0,0),(1,1,0)]),
        ("XOR",  vec![(0,0,0),(0,1,1),(1,0,1),(1,1,0)]),
        ("XNOR", vec![(0,0,1),(0,1,0),(1,0,0),(1,1,1)]),
    ];

    let mut best_gates: Vec<(&str, Solution)> = Vec::new();

    for (name, truth) in &gates {
        println!("========== {} ==========", name);
        println!("  Truth: {:?}", truth);
        let mut sols = full_search(name, truth);
        println!("  Total solutions: {}\n", sols.len());

        if sols.is_empty() {
            println!("  NO SOLUTION FOUND!\n");
            continue;
        }

        // Best by margin (most robust)
        sols.sort_by(|a,b| b.margin.partial_cmp(&a.margin).unwrap());
        println!("  TOP 5 by MARGIN (robustness):");
        for (i, s) in sols.iter().take(5).enumerate() {
            print_solution(s, &format!("#{}", i+1));
        }
        let best_margin = sols[0].clone();

        // Best by simplicity (smallest weights)
        sols.sort_by(|a,b| a.complexity.partial_cmp(&b.complexity).unwrap());
        println!("\n  TOP 5 by SIMPLICITY (smallest weights):");
        for (i, s) in sols.iter().take(5).enumerate() {
            print_solution(s, &format!("#{}", i+1));
        }
        let simplest = sols[0].clone();

        // Best combined score: margin * 10 - complexity
        sols.sort_by(|a,b| {
            let sa = a.margin * 10.0 - a.complexity * 0.1;
            let sb = b.margin * 10.0 - b.complexity * 0.1;
            sb.partial_cmp(&sa).unwrap()
        });
        println!("\n  BEST OVERALL (margin×10 - complexity×0.1):");
        print_solution(&sols[0], "PICK");
        verify(&sols[0], truth, name);

        // Rho distribution
        let mut rho_counts = std::collections::HashMap::new();
        for s in &sols { *rho_counts.entry(format!("{:.1}", s.rho)).or_insert(0u32) += 1; }
        let mut rho_sorted: Vec<_> = rho_counts.iter().collect();
        rho_sorted.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
        print!("  Rho distribution: ");
        for (r, c) in rho_sorted.iter().take(5) { print!("rho={}:{} ", r, c); }
        println!("\n");

        best_gates.push((name, sols[0].clone()));
    }

    // =========================================================
    // FINAL: Export the BEST gate set
    // =========================================================
    println!("========================================");
    println!("  VRAXION C19 GATE LIBRARY (best overall)\n");
    for (name, s) in &best_gates {
        println!("  {:>5}: C19({:+6.2}*a + {:+6.2}*b + {:+6.2}, rho={:.1}) > {:+5.2}   margin={:.4}",
                 name, s.w1, s.w2, s.bias, s.rho, s.thr, s.margin);
    }

    // Save as JSON
    let mut json = String::from("{\n  \"vraxion_c19_gate_library_v2\": {\n");
    for (i, (name, s)) in best_gates.iter().enumerate() {
        json += &format!("    \"{}\": {{\"w1\":{:.2},\"w2\":{:.2},\"bias\":{:.2},\"rho\":{:.1},\"thr\":{:.2},\"margin\":{:.4}}}",
                         name, s.w1, s.w2, s.bias, s.rho, s.thr, s.margin);
        if i < best_gates.len()-1 { json += ","; }
        json += "\n";
    }
    json += "  }\n}\n";
    std::fs::write("gate_library_v2.json", &json).ok();
    println!("\n  Saved: gate_library_v2.json");

    println!("\n=== DONE ===");
}
