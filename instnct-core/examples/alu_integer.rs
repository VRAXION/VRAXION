//! ALU Integer Pipeline — gate-by-gate fraction quantization + LUT baking
//!
//! Process per gate:
//!   1. Start from exhaustive-searched float weights
//!   2. Find common denominator → integer weights
//!   3. Compute integer input range
//!   4. Bake C19 as LUT (integer input → binary output)
//!   5. Verify LUT matches float gate 100%
//!   6. Lock, move to next gate
//!
//! Then compose: full adder → 4-bit adder → subtractor → ALU
//!
//! Run: cargo run --example alu_integer --release

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

// ============================================================
// STEP 1: Float gate definition (from exhaustive search)
// ============================================================

#[derive(Clone, Debug)]
struct FloatGate {
    name: &'static str,
    w: Vec<f32>,   // weights (2 or 3 inputs)
    bias: f32,
    rho: f32,
    thr: f32,
    margin: f32,
    truth: Vec<(Vec<u8>, u8)>,  // (inputs, expected_output)
}

// ============================================================
// STEP 2: Fraction extraction → integer weights
// ============================================================

#[derive(Clone, Debug)]
struct IntGate {
    name: String,
    w_int: Vec<i32>,       // integer weights
    bias_int: i32,         // integer bias
    denom: i32,            // common denominator (multiply all floats by this)
    rho: f32,              // rho stays float (baked into LUT)
    thr: f32,              // threshold stays float (baked into LUT)
    thr_int: i32,          // threshold as integer? or we compare LUT output
    input_range: (i32, i32), // min/max possible integer sum
    lut: Vec<u8>,          // precomputed: lut[sum - min] = 0 or 1
    verified: bool,
}

fn find_common_denom(values: &[f32], max_denom: i32) -> Option<(i32, Vec<i32>)> {
    // Try denominators from 1 to max_denom, find one where all values become clean integers
    for d in 1..=max_denom {
        let ints: Vec<i32> = values.iter().map(|&v| (v * d as f32).round() as i32).collect();
        let ok = values.iter().zip(&ints).all(|(&v, &i)| {
            (v - i as f32 / d as f32).abs() < 1e-6
        });
        if ok {
            return Some((d, ints));
        }
    }
    None
}

fn quantize_gate(fg: &FloatGate, logf: &mut std::fs::File) -> Option<IntGate> {
    log(logf, &format!("--- Quantizing: {} ---", fg.name));
    log(logf, &format!("  Float weights: {:?}, bias={}, rho={}, thr={}", fg.w, fg.bias, fg.rho, fg.thr));

    // Collect all values that need integer conversion
    let mut values: Vec<f32> = fg.w.clone();
    values.push(fg.bias);

    let (denom, ints) = match find_common_denom(&values, 100) {
        Some(r) => r,
        None => {
            log(logf, &format!("  FAIL: no common denominator found up to 100"));
            return None;
        }
    };

    let n_inputs = fg.w.len();
    let w_int: Vec<i32> = ints[..n_inputs].to_vec();
    let bias_int = ints[n_inputs];

    log(logf, &format!("  Denominator: {}", denom));
    log(logf, &format!("  Integer weights: {:?}, bias_int={}", w_int, bias_int));

    // Compute integer input range
    // Inputs are 0 or 1, so the integer sum range is:
    let mut min_sum = bias_int;
    let mut max_sum = bias_int;
    for &w in &w_int {
        if w > 0 { max_sum += w; } else { min_sum += w; }
    }
    log(logf, &format!("  Integer sum range: [{}, {}]", min_sum, max_sum));

    // Bake LUT: for each possible integer sum, compute C19 and compare with threshold
    // The actual float input to C19 is: sum_int / denom
    let lut_size = (max_sum - min_sum + 1) as usize;
    let mut lut = vec![0u8; lut_size];

    log(logf, &format!("  LUT size: {} entries", lut_size));

    for sum_int in min_sum..=max_sum {
        let x = sum_int as f32 / denom as f32;
        let y = c19(x, fg.rho);
        let out = if y > fg.thr { 1u8 } else { 0 };
        lut[(sum_int - min_sum) as usize] = out;
    }

    // Print LUT
    let lut_str: String = lut.iter().map(|&b| if b == 1 { '1' } else { '0' }).collect();
    log(logf, &format!("  LUT: [{}]", lut_str));

    // Verify against truth table
    let mut all_correct = true;
    for (inputs, expected) in &fg.truth {
        // Compute integer sum
        let sum_int: i32 = inputs.iter().zip(&w_int)
            .map(|(&inp, &w)| inp as i32 * w)
            .sum::<i32>() + bias_int;

        let lut_idx = (sum_int - min_sum) as usize;
        let got = lut[lut_idx];

        if got != *expected {
            log(logf, &format!("  VERIFY FAIL: inputs={:?} sum_int={} lut[{}]={} expected={}",
                inputs, sum_int, lut_idx, got, expected));
            all_correct = false;
        }
    }

    if all_correct {
        log(logf, &format!("  VERIFY: 100% correct!"));
    } else {
        log(logf, &format!("  VERIFY: FAILED"));
    }

    // Bits needed
    let bits_per_weight = w_int.iter().map(|&w| {
        if w == 0 { 1 } else { (w.abs() as f32).log2().ceil() as u32 + 1 } // +1 for sign
    }).max().unwrap_or(1);
    let lut_bits = lut_size;
    log(logf, &format!("  Bits per weight: {}, LUT bits: {}", bits_per_weight, lut_bits));
    log(logf, &format!("  {} LOCKED as integer gate", fg.name));

    Some(IntGate {
        name: fg.name.to_string(),
        w_int,
        bias_int,
        denom,
        rho: fg.rho,
        thr: fg.thr,
        thr_int: 0, // not used, threshold is baked into LUT
        input_range: (min_sum, max_sum),
        lut,
        verified: all_correct,
    })
}

// ============================================================
// STEP 3: Compose circuits from integer gates
// ============================================================

fn int_gate_eval(g: &IntGate, inputs: &[u8]) -> u8 {
    let sum_int: i32 = inputs.iter().zip(&g.w_int)
        .map(|(&inp, &w)| inp as i32 * w)
        .sum::<i32>() + g.bias_int;
    let idx = (sum_int - g.input_range.0) as usize;
    g.lut[idx]
}

fn int_full_adder(xor3: &IntGate, maj: &IntGate, a: u8, b: u8, cin: u8) -> (u8, u8) {
    let sum = int_gate_eval(xor3, &[a, b, cin]);
    let cout = int_gate_eval(maj, &[a, b, cin]);
    (sum, cout)
}

fn int_add_nbits(xor3: &IntGate, maj: &IntGate, a: u32, b: u32, bits: u32) -> (u32, u32) {
    let mut carry = 0u8;
    let mut result = 0u32;
    for bit in 0..bits {
        let ab = ((a >> bit) & 1) as u8;
        let bb = ((b >> bit) & 1) as u8;
        let (s, c) = int_full_adder(xor3, maj, ab, bb, carry);
        if s == 1 { result |= 1 << bit; }
        carry = c;
    }
    (result, carry as u32)
}

// Subtractor: a - b = a + NOT(b) + 1
fn int_sub_nbits(xor3: &IntGate, maj: &IntGate, not_g: &IntGate, a: u32, b: u32, bits: u32) -> (u32, u32) {
    // NOT each bit of b
    let mut b_not = 0u32;
    for bit in 0..bits {
        let bb = ((b >> bit) & 1) as u8;
        let nb = int_gate_eval(not_g, &[bb, 0]); // NOT gate has w2=0
        if nb == 1 { b_not |= 1 << bit; }
    }
    // Add a + NOT(b) with carry_in = 1
    let mut carry = 1u8; // initial carry = 1 for two's complement
    let mut result = 0u32;
    for bit in 0..bits {
        let ab = ((a >> bit) & 1) as u8;
        let bb = ((b_not >> bit) & 1) as u8;
        let (s, c) = int_full_adder(xor3, maj, ab, bb, carry);
        if s == 1 { result |= 1 << bit; }
        carry = c;
    }
    (result, carry as u32)
}

// Comparator: uses subtractor carry output
// a + NOT(b) + 1: carry_out=1 means a>=b, carry_out=0 means a<b
fn int_compare_nbits(xor3: &IntGate, maj: &IntGate, not_g: &IntGate, _nor_g: &IntGate,
                     a: u32, b: u32, bits: u32) -> (bool, bool, bool) {
    let (diff, carry) = int_sub_nbits(xor3, maj, not_g, a, b, bits);
    let result_masked = diff & ((1 << bits) - 1);

    // carry=1 → a >= b (no borrow), carry=0 → a < b (borrow)
    let a_lt_b = carry == 0;
    let a_eq_b = carry == 1 && result_masked == 0;
    let a_gt_b = carry == 1 && result_masked != 0;
    (a_lt_b, a_eq_b, a_gt_b)
}

// ============================================================
// MAIN
// ============================================================

fn main() {
    let log_path = "instnct-core/alu_integer_log.txt";
    let mut logf = std::fs::OpenOptions::new()
        .create(true).append(true)
        .open(log_path).unwrap();

    log(&mut logf, "========================================");
    log(&mut logf, "=== ALU INTEGER PIPELINE ===");
    log(&mut logf, "========================================");
    let t0 = std::time::Instant::now();

    // ============================================================
    // PHASE 1: Define all float gates from exhaustive search results
    // ============================================================
    log(&mut logf, "\n=== PHASE 1: Float gate definitions ===");

    let gates_2input: Vec<FloatGate> = vec![
        FloatGate {
            name: "NOT", w: vec![-9.75], bias: -5.50, rho: 16.0, thr: -4.0, margin: 5.25,
            truth: vec![(vec![0], 1), (vec![1], 0)],
        },
        FloatGate {
            name: "AND", w: vec![10.0, 10.0], bias: -4.50, rho: 0.0, thr: 4.0, margin: 4.25,
            truth: vec![(vec![0,0], 0), (vec![0,1], 0), (vec![1,0], 0), (vec![1,1], 1)],
        },
        FloatGate {
            name: "OR", w: vec![8.75, 8.75], bias: 5.50, rho: 0.0, thr: 4.0, margin: 4.25,
            truth: vec![(vec![0,0], 0), (vec![0,1], 1), (vec![1,0], 1), (vec![1,1], 1)],
        },
        FloatGate {
            name: "NAND", w: vec![-10.0, -10.0], bias: 4.50, rho: 16.0, thr: -4.0, margin: 5.25,
            truth: vec![(vec![0,0], 1), (vec![0,1], 1), (vec![1,0], 1), (vec![1,1], 0)],
        },
        FloatGate {
            name: "NOR", w: vec![-9.75, -9.75], bias: -5.50, rho: 16.0, thr: -4.0, margin: 5.25,
            truth: vec![(vec![0,0], 1), (vec![0,1], 0), (vec![1,0], 0), (vec![1,1], 0)],
        },
        FloatGate {
            name: "XOR", w: vec![0.50, 0.50], bias: 0.0, rho: 16.0, thr: 0.6, margin: 0.6,
            truth: vec![(vec![0,0], 0), (vec![0,1], 1), (vec![1,0], 1), (vec![1,1], 0)],
        },
        FloatGate {
            name: "XNOR", w: vec![-0.50, 0.50], bias: 0.50, rho: 16.0, thr: 0.6, margin: 0.6,
            truth: vec![(vec![0,0], 1), (vec![0,1], 0), (vec![1,0], 0), (vec![1,1], 1)],
        },
    ];

    // 3-input gates from gate_mux.rs results
    let gates_3input: Vec<FloatGate> = vec![
        FloatGate {
            name: "AND3", w: vec![8.0, 8.0, 8.0], bias: -10.0, rho: 0.0, thr: 4.0, margin: 4.0,
            truth: vec![
                (vec![0,0,0], 0), (vec![0,0,1], 0), (vec![0,1,0], 0), (vec![0,1,1], 0),
                (vec![1,0,0], 0), (vec![1,0,1], 0), (vec![1,1,0], 0), (vec![1,1,1], 1),
            ],
        },
        FloatGate {
            name: "OR3", w: vec![8.75, 8.75, 8.75], bias: 5.50, rho: 0.0, thr: 4.0, margin: 4.25,
            truth: vec![
                (vec![0,0,0], 0), (vec![0,0,1], 1), (vec![0,1,0], 1), (vec![0,1,1], 1),
                (vec![1,0,0], 1), (vec![1,0,1], 1), (vec![1,1,0], 1), (vec![1,1,1], 1),
            ],
        },
        FloatGate {
            name: "NOR3", w: vec![-9.75, -9.75, -9.75], bias: -5.50, rho: 16.0, thr: -4.0, margin: 5.25,
            truth: vec![
                (vec![0,0,0], 1), (vec![0,0,1], 0), (vec![0,1,0], 0), (vec![0,1,1], 0),
                (vec![1,0,0], 0), (vec![1,0,1], 0), (vec![1,1,0], 0), (vec![1,1,1], 0),
            ],
        },
        FloatGate {
            name: "MUX", w: vec![-0.50, 5.25, 8.75], bias: -6.25, rho: 16.0, thr: 0.6, margin: 0.6,
            truth: vec![
                (vec![0,0,0], 0), (vec![0,0,1], 1), (vec![0,1,0], 0), (vec![0,1,1], 1),
                (vec![1,0,0], 0), (vec![1,0,1], 0), (vec![1,1,0], 1), (vec![1,1,1], 1),
            ],
        },
        FloatGate {
            name: "XOR3", w: vec![1.50, 1.50, 1.50], bias: 3.0, rho: 16.0, thr: 0.6, margin: 0.6,
            truth: vec![
                (vec![0,0,0], 0), (vec![0,0,1], 1), (vec![0,1,0], 1), (vec![0,1,1], 0),
                (vec![1,0,0], 1), (vec![1,0,1], 0), (vec![1,1,0], 0), (vec![1,1,1], 1),
            ],
        },
        FloatGate {
            name: "MAJ", w: vec![8.50, 8.50, 8.50], bias: -2.75, rho: 0.0, thr: 4.0, margin: 4.1875,
            truth: vec![
                (vec![0,0,0], 0), (vec![0,0,1], 0), (vec![0,1,0], 0), (vec![0,1,1], 1),
                (vec![1,0,0], 0), (vec![1,0,1], 1), (vec![1,1,0], 1), (vec![1,1,1], 1),
            ],
        },
    ];

    // ============================================================
    // PHASE 2: Quantize each gate to integers + bake LUT
    // ============================================================
    log(&mut logf, "\n=== PHASE 2: Integer quantization + LUT baking ===");

    let mut int_gates: std::collections::HashMap<String, IntGate> = std::collections::HashMap::new();
    let mut all_ok = true;

    for fg in gates_2input.iter().chain(gates_3input.iter()) {
        match quantize_gate(fg, &mut logf) {
            Some(ig) => {
                if !ig.verified { all_ok = false; }
                int_gates.insert(ig.name.clone(), ig);
            }
            None => {
                all_ok = false;
                log(&mut logf, &format!("  {} QUANTIZATION FAILED", fg.name));
            }
        }
    }

    log(&mut logf, &format!("\n  All gates quantized: {} | All verified: {}",
        int_gates.len(), if all_ok { "YES" } else { "NO" }));

    if !all_ok {
        log(&mut logf, "ABORTING: not all gates verified");
        return;
    }

    // ============================================================
    // PHASE 3: Compose and verify circuits with INTEGER gates
    // ============================================================
    log(&mut logf, "\n=== PHASE 3: Circuit composition (pure integer) ===");

    let xor3 = &int_gates["XOR3"];
    let maj = &int_gates["MAJ"];
    let not_g = &int_gates["NOT"];

    // --- Full adder verification ---
    log(&mut logf, "\n--- Full Adder (XOR3 + MAJ, pure integer) ---");
    let mut fa_ok = 0;
    let fa_total = 8;
    for a in 0..=1u8 {
        for b in 0..=1u8 {
            for cin in 0..=1u8 {
                let (sum, cout) = int_full_adder(xor3, maj, a, b, cin);
                let expected_sum = (a + b + cin) & 1;
                let expected_cout = (a + b + cin) >> 1;
                let ok = sum == expected_sum && cout == expected_cout;
                if ok { fa_ok += 1; }
                log(&mut logf, &format!("  {}+{}+{} = sum={} cout={} (exp s={} c={}) {}",
                    a, b, cin, sum, cout, expected_sum, expected_cout,
                    if ok { "OK" } else { "FAIL" }));
            }
        }
    }
    log(&mut logf, &format!("  Full adder: {}/{} correct", fa_ok, fa_total));

    // --- 4-bit adder ---
    log(&mut logf, "\n--- 4-bit Adder (pure integer) ---");
    let mut add4_ok = 0;
    let add4_total = 256;
    for a in 0..16u32 {
        for b in 0..16u32 {
            let (result, carry) = int_add_nbits(xor3, maj, a, b, 4);
            let expected = a + b;
            let expected_result = expected & 0xF;
            let expected_carry = expected >> 4;
            if result == expected_result && carry == expected_carry { add4_ok += 1; }
        }
    }
    log(&mut logf, &format!("  4-bit adder: {}/{} correct", add4_ok, add4_total));

    // --- 8-bit adder ---
    log(&mut logf, "\n--- 8-bit Adder (pure integer, sampling) ---");
    let mut add8_ok = 0;
    let add8_total = 1000;
    let mut rng_state = 42u64;
    for _ in 0..add8_total {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let a = ((rng_state >> 16) & 0xFF) as u32;
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let b = ((rng_state >> 16) & 0xFF) as u32;
        let (result, carry) = int_add_nbits(xor3, maj, a, b, 8);
        let expected = a + b;
        if result == (expected & 0xFF) && carry == (expected >> 8) { add8_ok += 1; }
    }
    log(&mut logf, &format!("  8-bit adder: {}/{} correct (sampled)", add8_ok, add8_total));

    // --- 4-bit subtractor ---
    log(&mut logf, "\n--- 4-bit Subtractor (pure integer) ---");
    let mut sub4_ok = 0;
    let sub4_total = 256;
    for a in 0..16u32 {
        for b in 0..16u32 {
            let (result, _borrow) = int_sub_nbits(xor3, maj, not_g, a, b, 4);
            let expected = (a as i32 - b as i32).rem_euclid(16) as u32;
            if result == expected { sub4_ok += 1; }
        }
    }
    log(&mut logf, &format!("  4-bit subtractor: {}/{} correct", sub4_ok, sub4_total));

    // --- 4-bit comparator ---
    log(&mut logf, "\n--- 4-bit Comparator (pure integer) ---");
    let mut cmp4_ok = 0;
    let cmp4_total = 256;
    for a in 0..16u32 {
        for b in 0..16u32 {
            let (lt, eq, gt) = int_compare_nbits(xor3, maj, not_g,
                int_gates.get("NOR").unwrap_or(&int_gates["NOR3"]), a, b, 4);
            let exp_lt = a < b;
            let exp_eq = a == b;
            let exp_gt = a > b;
            if lt == exp_lt && eq == exp_eq && gt == exp_gt { cmp4_ok += 1; }
        }
    }
    log(&mut logf, &format!("  4-bit comparator: {}/{} correct", cmp4_ok, cmp4_total));

    // ============================================================
    // PHASE 4: Summary + export
    // ============================================================
    log(&mut logf, "\n=== PHASE 4: Summary ===");
    log(&mut logf, "");

    log(&mut logf, "GATE LIBRARY (integer quantized):");
    log(&mut logf, &format!("{:<8} {:>6} {:>12} {:>6} {:>8} {:>6}",
        "Gate", "Denom", "Weights", "Bias", "LUT_sz", "OK"));
    for name in &["NOT","AND","OR","NAND","NOR","XOR","XNOR","AND3","OR3","NOR3","MUX","XOR3","MAJ"] {
        if let Some(g) = int_gates.get(*name) {
            log(&mut logf, &format!("{:<8} {:>6} {:>12} {:>6} {:>8} {:>6}",
                g.name, g.denom, format!("{:?}", g.w_int), g.bias_int, g.lut.len(),
                if g.verified { "YES" } else { "NO" }));
        }
    }

    log(&mut logf, "");
    log(&mut logf, "CIRCUIT VERIFICATION (pure integer, zero float):");
    log(&mut logf, &format!("  Full adder:      {}/8", fa_ok));
    log(&mut logf, &format!("  4-bit adder:     {}/256", add4_ok));
    log(&mut logf, &format!("  8-bit adder:     {}/1000 (sampled)", add8_ok));
    log(&mut logf, &format!("  4-bit subtractor:{}/256", sub4_ok));
    log(&mut logf, &format!("  4-bit comparator:{}/256", cmp4_ok));

    let total_neurons_4bit_alu = 8 + 4 + 4 + 3; // adder(8) + sub NOT(4) + comparator extras
    log(&mut logf, &format!("\n  Total neurons for 4-bit ALU: ~{}", total_neurons_4bit_alu));
    log(&mut logf, &format!("  Total LUT entries across all gates: {}",
        int_gates.values().map(|g| g.lut.len()).sum::<usize>()));

    // Export integer gate library as JSON
    let json_path = "instnct-core/gate_library_int.json";
    let mut jf = std::fs::File::create(json_path).unwrap();
    write!(jf, "{{\n  \"vraxion_c19_integer_gate_library\": {{\n").ok();
    let names: Vec<&&str> = vec![&"NOT",&"AND",&"OR",&"NAND",&"NOR",&"XOR",&"XNOR",
                                  &"AND3",&"OR3",&"NOR3",&"MUX",&"XOR3",&"MAJ"];
    for (i, name) in names.iter().enumerate() {
        if let Some(g) = int_gates.get(**name) {
            let lut_str: String = g.lut.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(",");
            write!(jf, "    \"{}\": {{\"w_int\":{:?},\"bias_int\":{},\"denom\":{},\"rho\":{:.1},\"lut\":[{}],\"lut_offset\":{}}}",
                g.name, g.w_int, g.bias_int, g.denom, g.rho, lut_str, g.input_range.0).ok();
            if i < names.len() - 1 { write!(jf, ",").ok(); }
            write!(jf, "\n").ok();
        }
    }
    write!(jf, "  }}\n}}\n").ok();
    log(&mut logf, &format!("\n  Exported: {}", json_path));

    log(&mut logf, &format!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64()));
    log(&mut logf, "=== ALU INTEGER PIPELINE COMPLETE ===");
}
