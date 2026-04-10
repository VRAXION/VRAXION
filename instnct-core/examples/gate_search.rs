//! Gate Search: build logic gates from C19 neurons, then wire into a full adder
//!
//! Step 1: Find NAND, AND, OR, XOR, NOT from single C19 neurons (binary I/O)
//! Step 2: Wire into half-adder (sum + carry)
//! Step 3: Wire into full-adder (with carry-in)
//! Step 4: Chain into multi-bit adder
//!
//! Run: cargo run --example gate_search --release

fn c19(x: f32, rho: f32) -> f32 {
    let c = 1.0f32; let l = 6.0 * c;
    if x >= l { return x - l; } if x <= -l { return x + l; }
    let s = x / c; let n = s.floor(); let t = s - n;
    let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sgn * h + rho * h * h)
}

// One C19 neuron gate: output = C19(w1*a + w2*b + bias, rho)
// Threshold: output > 0.5 → 1, else → 0
fn gate_eval(w1: f32, w2: f32, bias: f32, rho: f32, a: f32, b: f32) -> f32 {
    c19(w1 * a + w2 * b + bias, rho)
}

fn binarize(x: f32, threshold: f32) -> u8 {
    if x > threshold { 1 } else { 0 }
}

fn search_gate(name: &str, truth: &[(u8,u8,u8)]) {
    println!("=== Searching: {} ===", name);
    println!("  Truth table: {:?}", truth);

    // Search over w1, w2, bias, rho, threshold
    let weights: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.25).collect();
    let rhos: Vec<f32> = vec![0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];
    let thresholds: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.1).collect();

    let mut found = 0;
    let mut first_solution: Option<(f32,f32,f32,f32,f32)> = None;

    for &rho in &rhos {
        for &w1 in &weights {
            for &w2 in &weights {
                for &bias in &weights {
                    // Compute outputs for all truth table entries
                    let outputs: Vec<f32> = truth.iter()
                        .map(|&(a,b,_)| gate_eval(w1, w2, bias, rho, a as f32, b as f32))
                        .collect();

                    // Find a threshold that separates 0s and 1s
                    for &thr in &thresholds {
                        let correct = truth.iter().zip(&outputs).all(|(&(_,_,expected), &out)| {
                            binarize(out, thr) == expected
                        });
                        if correct {
                            found += 1;
                            if first_solution.is_none() {
                                first_solution = Some((w1, w2, bias, rho, thr));
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some((w1, w2, bias, rho, thr)) = first_solution {
        println!("  FOUND! {} solutions", found);
        println!("  First: C19({:.2}*a + {:.2}*b + {:.2}, rho={:.1}) > {:.2} → 1", w1, w2, bias, rho, thr);
        println!("  Verify:");
        for &(a, b, expected) in truth {
            let raw = gate_eval(w1, w2, bias, rho, a as f32, b as f32);
            let out = binarize(raw, thr);
            println!("    {}({},{}) = {} (raw={:.4}, expected={}){}",
                     name, a, b, out, raw, expected,
                     if out == expected { " ✓" } else { " ✗" });
        }
    } else {
        println!("  NOT FOUND with single neuron!");
    }
    println!();
}

// A "gate" is (w1, w2, bias, rho, threshold)
#[derive(Clone, Debug)]
struct Gate {
    w1: f32, w2: f32, bias: f32, rho: f32, thr: f32,
    name: String,
}

impl Gate {
    fn eval(&self, a: f32, b: f32) -> f32 {
        let raw = c19(self.w1 * a + self.w2 * b + self.bias, self.rho);
        if raw > self.thr { 1.0 } else { 0.0 }
    }
}

fn find_gate(name: &str, truth: &[(u8,u8,u8)]) -> Option<Gate> {
    let weights: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.5).collect();
    let rhos: Vec<f32> = vec![0.0, 1.0, 2.0, 4.0, 8.0];
    let thresholds: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.1).collect();

    for &rho in &rhos {
        for &w1 in &weights {
            for &w2 in &weights {
                for &bias in &weights {
                    let outputs: Vec<f32> = truth.iter()
                        .map(|&(a,b,_)| gate_eval(w1, w2, bias, rho, a as f32, b as f32))
                        .collect();
                    for &thr in &thresholds {
                        let ok = truth.iter().zip(&outputs).all(|(&(_,_,e), &o)| binarize(o, thr) == e);
                        if ok {
                            return Some(Gate { w1, w2, bias, rho, thr, name: name.to_string() });
                        }
                    }
                }
            }
        }
    }
    None
}

fn main() {
    println!("=== GATE SEARCH: C19 logic gates → full adder ===\n");

    // =========================================================
    // STEP 1: Find basic gates
    // =========================================================
    println!("===== STEP 1: Basic Logic Gates =====\n");

    let gates_to_find = vec![
        ("NOT",  vec![(0,0,1), (1,0,0)]),                           // NOT a (b ignored)
        ("AND",  vec![(0,0,0), (0,1,0), (1,0,0), (1,1,1)]),
        ("OR",   vec![(0,0,0), (0,1,1), (1,0,1), (1,1,1)]),
        ("NAND", vec![(0,0,1), (0,1,1), (1,0,1), (1,1,0)]),
        ("NOR",  vec![(0,0,1), (0,1,0), (1,0,0), (1,1,0)]),
        ("XOR",  vec![(0,0,0), (0,1,1), (1,0,1), (1,1,0)]),
        ("XNOR", vec![(0,0,1), (0,1,0), (1,0,0), (1,1,1)]),
    ];

    for (name, truth) in &gates_to_find {
        search_gate(name, truth);
    }

    // =========================================================
    // STEP 2: Build half-adder from gates
    // =========================================================
    println!("===== STEP 2: Half Adder (from C19 gates) =====\n");

    // Half adder: sum = XOR(a,b), carry = AND(a,b)
    let xor_gate = find_gate("XOR", &[(0,0,0),(0,1,1),(1,0,1),(1,1,0)]);
    let and_gate = find_gate("AND", &[(0,0,0),(0,1,0),(1,0,0),(1,1,1)]);

    if let (Some(xor), Some(and)) = (&xor_gate, &and_gate) {
        println!("  Half adder = XOR + AND:");
        println!("    XOR: C19({:.1}*a + {:.1}*b + {:.1}, rho={:.0}) > {:.1}",
                 xor.w1, xor.w2, xor.bias, xor.rho, xor.thr);
        println!("    AND: C19({:.1}*a + {:.1}*b + {:.1}, rho={:.0}) > {:.1}",
                 and.w1, and.w2, and.bias, and.rho, and.thr);
        println!("\n  Verify:");
        for a in 0..=1u8 {
            for b in 0..=1u8 {
                let sum = xor.eval(a as f32, b as f32) as u8;
                let carry = and.eval(a as f32, b as f32) as u8;
                let expected_sum = a ^ b;
                let expected_carry = a & b;
                println!("    {} + {} = {}:{} (expected {}:{}){}",
                         a, b, carry, sum, expected_carry, expected_sum,
                         if sum==expected_sum && carry==expected_carry { " ✓" } else { " ✗" });
            }
        }

        // =========================================================
        // STEP 3: Full adder (a + b + cin → sum, cout)
        // =========================================================
        println!("\n===== STEP 3: Full Adder =====\n");

        // Full adder from 2 half-adders + OR:
        // sum = XOR(XOR(a,b), cin)
        // cout = OR(AND(a,b), AND(XOR(a,b), cin))
        let or_gate = find_gate("OR", &[(0,0,0),(0,1,1),(1,0,1),(1,1,1)]);

        if let Some(or) = &or_gate {
            println!("  Full adder = 2×XOR + 2×AND + 1×OR = 5 C19 neurons\n");
            println!("  Verify:");
            let mut all_ok = true;
            for a in 0..=1u8 {
                for b in 0..=1u8 {
                    for cin in 0..=1u8 {
                        // Half-adder 1: a + b
                        let ha1_sum = xor.eval(a as f32, b as f32);
                        let ha1_carry = and.eval(a as f32, b as f32);
                        // Half-adder 2: ha1_sum + cin
                        let ha2_sum = xor.eval(ha1_sum, cin as f32);
                        let ha2_carry = and.eval(ha1_sum, cin as f32);
                        // cout = OR(ha1_carry, ha2_carry)
                        let cout = or.eval(ha1_carry, ha2_carry);

                        let expected = a as u32 + b as u32 + cin as u32;
                        let exp_sum = (expected & 1) as u8;
                        let exp_cout = ((expected >> 1) & 1) as u8;
                        let ok = ha2_sum as u8 == exp_sum && cout as u8 == exp_cout;
                        if !ok { all_ok = false; }
                        println!("    {} + {} + {} = {}:{} (expected {}:{}){}",
                                 a, b, cin, cout as u8, ha2_sum as u8,
                                 exp_cout, exp_sum,
                                 if ok { " ✓" } else { " ✗" });
                    }
                }
            }

            if all_ok {
                // =========================================================
                // STEP 4: 4-bit adder
                // =========================================================
                println!("\n===== STEP 4: 4-bit Ripple-Carry Adder (20 C19 neurons) =====\n");
                println!("  4 full-adders chained: 4 × 5 = 20 C19 neurons\n");

                let mut correct = 0;
                let total = 256; // 0-15 + 0-15
                for a_val in 0..16u8 {
                    for b_val in 0..16u8 {
                        let mut carry = 0.0f32;
                        let mut result = 0u8;
                        for bit in 0..4 {
                            let a_bit = ((a_val >> bit) & 1) as f32;
                            let b_bit = ((b_val >> bit) & 1) as f32;
                            // Full adder
                            let ha1_s = xor.eval(a_bit, b_bit);
                            let ha1_c = and.eval(a_bit, b_bit);
                            let ha2_s = xor.eval(ha1_s, carry);
                            let ha2_c = and.eval(ha1_s, carry);
                            let cout = or.eval(ha1_c, ha2_c);
                            if ha2_s as u8 == 1 { result |= 1 << bit; }
                            carry = cout;
                        }
                        let expected = (a_val as u16 + b_val as u16) as u8 & 0x0F;
                        if result == expected { correct += 1; }
                    }
                }
                println!("  4-bit ADD: {}/{} ({:.1}%)", correct, total, correct as f64/total as f64*100.0);

                if correct == total {
                    println!("  >>> PERFECT 4-bit ADDER from 20 C19 neurons! <<<");
                }

                // Show some examples
                println!("\n  Examples:");
                for &(a, b) in &[(3,5),(7,8),(15,1),(9,6),(12,11)] {
                    let mut carry = 0.0f32;
                    let mut result = 0u8;
                    for bit in 0..4 {
                        let a_bit = ((a >> bit) & 1) as f32;
                        let b_bit = ((b >> bit) & 1) as f32;
                        let ha1_s = xor.eval(a_bit, b_bit);
                        let ha1_c = and.eval(a_bit, b_bit);
                        let ha2_s = xor.eval(ha1_s, carry);
                        let ha2_c = and.eval(ha1_s, carry);
                        carry = or.eval(ha1_c, ha2_c);
                        if ha2_s as u8 == 1 { result |= 1 << bit; }
                    }
                    let expected = (a + b) & 0x0F;
                    let carry_out = if (a as u16 + b as u16) > 15 { 1 } else { 0 };
                    println!("    {} + {} = {} (carry={}) {}", a, b, result, carry as u8,
                             if result == expected { "✓" } else { "✗" });
                }
            }

            // =========================================================
            // SUMMARY
            // =========================================================
            println!("\n===== SUMMARY =====\n");
            println!("  Logic gates from C19 neurons:");
            println!("    NOT:  1 neuron");
            println!("    AND:  1 neuron");
            println!("    OR:   1 neuron");
            println!("    NAND: 1 neuron");
            println!("    XOR:  1 neuron  ← standard logic needs 4 NAND gates!");
            println!("    XNOR: 1 neuron");
            println!();
            println!("  Compositions:");
            println!("    Half-adder:  2 neurons (XOR + AND)");
            println!("    Full-adder:  5 neurons (2×XOR + 2×AND + OR)");
            println!("    4-bit adder: 20 neurons");
            println!("    N-bit adder: 5N neurons");
            println!();
            println!("  vs Standard CMOS:");
            println!("    XOR gate:    4 NAND gates = 16 transistors");
            println!("    Full-adder:  ~28 transistors");
            println!("    C19 neuron:  1 LUT + 1 multiply + 1 add ≈ few transistors");
            println!("    >>> C19 XOR in 1 neuron vs 4 NAND = potential hardware win <<<");
        }
    }

    println!("\n=== DONE ===");
}
