//! Neuron IQ — Raven's-style cognitive curriculum for 1 neuron
//! Progressive difficulty: does the neuron GENERALIZE or just memorize?
//!
//! Each level:
//!   - Has a hidden RULE that generates input→output
//!   - Neuron sees SOME examples (train), must predict UNSEEN (test)
//!   - Score = test accuracy (generalization, not memorization)
//!
//! Run: cargo run --example neuron_iq --release

use std::time::Instant;

// ── PRNG ──
struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 { self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.s }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn range(&mut self, lo: f32, hi: f32) -> f32 { lo + self.f32() * (hi - lo) }
    fn shuffle<T>(&mut self, v: &mut [T]) { for i in (1..v.len()).rev() { let j = self.next() as usize % (i+1); v.swap(i,j); } }
}

// ── Single neuron: dot product + threshold ──
// Searches over ternary weights {-1,0,+1} + best threshold
// This IS the VRAXION LutGate primitive

fn exhaustive_1neuron(
    train: &[(Vec<f32>, u8)],
    n_in: usize,
) -> (Vec<i8>, i8, i32, f32) {
    let n_w = n_in + 1;
    let total = 3u64.pow(n_w as u32);
    let n_pat = train.len();

    let mut best_w = vec![0i8; n_in];
    let mut best_b: i8 = 0;
    let mut best_t: i32 = 0;
    let mut best_score = 0usize;

    for combo in 0..total {
        let mut w = vec![0i8; n_in];
        let mut r = combo;
        for wi in w.iter_mut() { *wi = (r % 3) as i8 - 1; r /= 3; }
        let b = (r % 3) as i8 - 1;

        let dots: Vec<i32> = train.iter().map(|(x, _)| {
            let mut d = b as i32;
            for (wi, xi) in w.iter().zip(x) { d += (*wi as i32) * (*xi as i32); }
            d
        }).collect();

        let min_d = dots.iter().copied().min().unwrap_or(0);
        let max_d = dots.iter().copied().max().unwrap_or(0);

        for thresh in (min_d - 1)..=(max_d + 1) {
            let score = dots.iter().zip(train).filter(|(&d, (_, y))| {
                (if d >= thresh { 1u8 } else { 0u8 }) == *y
            }).count();
            if score > best_score {
                best_score = score;
                best_w = w.clone(); best_b = b; best_t = thresh;
                if score == n_pat { return (best_w, best_b, best_t, 100.0); }
            }
        }
    }
    (best_w, best_b, best_t, best_score as f32 / n_pat as f32 * 100.0)
}

fn eval_neuron(w: &[i8], b: i8, thresh: i32, data: &[(Vec<f32>, u8)]) -> f32 {
    let correct = data.iter().filter(|(x, y)| {
        let mut d = b as i32;
        for (wi, xi) in w.iter().zip(x.iter()) { d += (*wi as i32) * (*xi as i32); }
        (if d >= thresh { 1u8 } else { 0u8 }) == *y
    }).count();
    correct as f32 / data.len() as f32 * 100.0
}

// ── Encoding helpers ──

fn bits(val: usize, n: usize) -> Vec<f32> {
    (0..n).map(|i| if val & (1 << i) != 0 { 1.0 } else { 0.0 }).collect()
}

// ── CURRICULUM ──
// Each level returns (name, description, n_inputs, train_data, test_data)

struct IQLevel {
    name: &'static str,
    desc: &'static str,
    n_in: usize,
    train: Vec<(Vec<f32>, u8)>,
    test: Vec<(Vec<f32>, u8)>,
    can_1neuron: bool,  // theoretically solvable by 1 linear threshold?
}

fn curriculum(rng: &mut Rng) -> Vec<IQLevel> {
    let mut levels = Vec::new();

    // ── L01: SAME/DIFFERENT ──
    // "Are these two things the same?"
    // Rule: output 1 if input[0] == input[1]
    // This is XNOR — NOT linear! 1 neuron should fail.
    {
        let all: Vec<(Vec<f32>, u8)> = (0..4).map(|v| {
            let b = bits(v, 2);
            let y = if b[0] == b[1] { 1 } else { 0 };
            (b, y)
        }).collect();
        levels.push(IQLevel {
            name: "L01 Same?", desc: "Are both inputs identical?",
            n_in: 2, train: all.clone(), test: all, can_1neuron: false,
        });
    }

    // ── L02: DETECT PRESENCE ──
    // "Is there at least one 1 in the input?"
    // Rule: OR — linear, 1 neuron should solve
    {
        let mut all: Vec<(Vec<f32>, u8)> = (0..16).map(|v| {
            let b = bits(v, 4);
            let y = if b.iter().any(|&x| x > 0.5) { 1 } else { 0 };
            (b, y)
        }).collect();
        rng.shuffle(&mut all);
        let test = all.split_off(12);
        levels.push(IQLevel {
            name: "L02 Any?", desc: "Is at least one bit set? (4 bit)",
            n_in: 4, train: all, test, can_1neuron: true,
        });
    }

    // ── L03: COUNTING ──
    // "Are more than half the bits set?"
    // Rule: majority — linear threshold, 1 neuron
    {
        let mut all: Vec<(Vec<f32>, u8)> = (0..64).map(|v| {
            let b = bits(v, 6);
            let sum: f32 = b.iter().sum();
            (b, if sum > 3.0 { 1 } else { 0 })
        }).collect();
        rng.shuffle(&mut all);
        let test = all.split_off(48);
        levels.push(IQLevel {
            name: "L03 Most?", desc: "Are majority of bits set? (6 bit)",
            n_in: 6, train: all, test, can_1neuron: true,
        });
    }

    // ── L04: COMPARISON ──
    // "Is the left number bigger than the right?"
    // 2×3 bit numbers: a > b?
    // Linear — 1 neuron (weighted sum of a - b > 0)
    {
        let mut all: Vec<(Vec<f32>, u8)> = Vec::new();
        for a in 0..8u8 { for b in 0..8u8 {
            let mut inp = bits(a as usize, 3);
            inp.extend(bits(b as usize, 3));
            all.push((inp, if a > b { 1 } else { 0 }));
        }}
        rng.shuffle(&mut all);
        let test = all.split_off(48);
        levels.push(IQLevel {
            name: "L04 Bigger?", desc: "Is left 3-bit number > right?",
            n_in: 6, train: all, test, can_1neuron: true,
        });
    }

    // ── L05: PATTERN MATCH ──
    // "Does the input contain the subsequence 110?"
    // Check bits [i, i+1, i+2] for any i
    // Not linear in general — needs to check multiple positions
    {
        let mut all: Vec<(Vec<f32>, u8)> = (0..64).map(|v| {
            let b = bits(v, 6);
            let mut has = false;
            for i in 0..4 {
                if b[i] > 0.5 && b[i+1] > 0.5 && b[i+2] < 0.5 { has = true; }
            }
            (b, if has { 1 } else { 0 })
        }).collect();
        rng.shuffle(&mut all);
        let test = all.split_off(48);
        levels.push(IQLevel {
            name: "L05 Pattern?", desc: "Contains subsequence 110? (6 bit)",
            n_in: 6, train: all, test, can_1neuron: false,
        });
    }

    // ── L06: SYMMETRY ──
    // "Is the input a palindrome?" (bit reversal = same)
    // 4 bits: bit[0]==bit[3] AND bit[1]==bit[2]
    // Not linear — needs AND of two comparisons
    {
        let all: Vec<(Vec<f32>, u8)> = (0..16).map(|v| {
            let b = bits(v, 4);
            let sym = (b[0] == b[3]) && (b[1] == b[2]);
            (b, if sym { 1 } else { 0 })
        }).collect();
        // All 16 as both train and test (exhaustive, tiny)
        levels.push(IQLevel {
            name: "L06 Symmetric?", desc: "Is 4-bit input a palindrome?",
            n_in: 4, train: all.clone(), test: all, can_1neuron: false,
        });
    }

    // ── L07: ANALOGY (simple) ──
    // A:B :: C:?
    // Rule: if A has property P, B=1. Does C have property P?
    // Encode: [A_bits, B, C_bits] → predict D
    // Property P = "more 1s than 0s" (majority)
    // This is basically: does C have majority? (ignore A, B — they're examples)
    // But the POINT is: can the neuron learn to ignore the example and just apply the rule?
    // 2-bit A, 1-bit B, 2-bit C → 1-bit D
    {
        let mut all: Vec<(Vec<f32>, u8)> = Vec::new();
        for a in 0..4u8 {
            let a_maj = if (a as u16).count_ones() > 1 { 1.0f32 } else { 0.0 };
            for c in 0..4u8 {
                let c_maj: u8 = if (c as u16).count_ones() > 1 { 1 } else { 0 };
                let mut inp = bits(a as usize, 2);
                inp.push(a_maj); // B = "answer for A"
                inp.extend(bits(c as usize, 2));
                all.push((inp, c_maj));
            }
        }
        rng.shuffle(&mut all);
        let test = all.split_off(12);
        levels.push(IQLevel {
            name: "L07 Analogy", desc: "A:B::C:? (majority rule, 2 bit)",
            n_in: 5, train: all, test, can_1neuron: true,
        });
    }

    // ── L08: SEQUENCE PREDICTION ──
    // Given 3 bits of a sequence, predict the 4th
    // Rule: each bit = XOR of previous two (Fibonacci-style mod 2)
    // input = [b0, b1, b2], target = b1 XOR b2
    {
        let mut all: Vec<(Vec<f32>, u8)> = Vec::new();
        for b0 in 0..2u8 { for b1 in 0..2u8 { for b2 in 0..2u8 {
            let b3 = b1 ^ b2;
            all.push((vec![b0 as f32, b1 as f32, b2 as f32], b3));
        }}}
        levels.push(IQLevel {
            name: "L08 Sequence", desc: "Predict next: b3 = b1 XOR b2",
            n_in: 3, train: all.clone(), test: all, can_1neuron: false,
        });
    }

    // ── L09: RULE EXTRACTION ──
    // Given labeled examples as input, classify a new point
    // 3 "training examples" (each 2 bits + 1 label) + 1 query (2 bits) → label
    // Rule is always "is bit0 set?"
    // Input: [ex1_b0, ex1_b1, ex1_label, ex2_b0, ex2_b1, ex2_label, q_b0, q_b1]
    // Target: q_b0 (the rule is "output = bit 0")
    {
        let mut all: Vec<(Vec<f32>, u8)> = Vec::new();
        for q in 0..4u8 {
            // Generate consistent examples
            for e1 in 0..4u8 { for e2 in 0..4u8 {
                let mut inp = Vec::new();
                // Example 1
                inp.extend(bits(e1 as usize, 2));
                inp.push(if e1 & 1 == 1 { 1.0 } else { 0.0 }); // label = bit0
                // Example 2
                inp.extend(bits(e2 as usize, 2));
                inp.push(if e2 & 1 == 1 { 1.0 } else { 0.0 });
                // Query
                inp.extend(bits(q as usize, 2));
                let target = (q & 1) as u8;
                all.push((inp, target));
            }}
        }
        rng.shuffle(&mut all);
        let test = all.split_off(48);
        levels.push(IQLevel {
            name: "L09 Rule?", desc: "Extract rule from examples, apply to query",
            n_in: 8, train: all, test, can_1neuron: true, // can just read q_b0
        });
    }

    // ── L10: ODDBALL ──
    // "Which group is different?" — 3 groups of 2 bits, one is different
    // Input: [g1_0, g1_1, g2_0, g2_1, g3_0, g3_1]
    // Output: 1 if group 3 is the odd one out (differs from g1 and g2 which match)
    {
        let mut all: Vec<(Vec<f32>, u8)> = Vec::new();
        for g1 in 0..4u8 { for g2 in 0..4u8 { for g3 in 0..4u8 {
            let mut inp = bits(g1 as usize, 2);
            inp.extend(bits(g2 as usize, 2));
            inp.extend(bits(g3 as usize, 2));
            // g3 is oddball if g1==g2 and g3!=g1
            let target = if g1 == g2 && g3 != g1 { 1u8 } else { 0u8 };
            all.push((inp, target));
        }}}
        rng.shuffle(&mut all);
        let test = all.split_off(48);
        levels.push(IQLevel {
            name: "L10 Oddball", desc: "Is group 3 the odd one out?",
            n_in: 6, train: all, test, can_1neuron: false,
        });
    }

    levels
}

fn main() {
    let t0 = Instant::now();
    let mut rng = Rng::new(42);
    let levels = curriculum(&mut rng);

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  NEURON IQ — Cognitive Curriculum (1 neuron, ternary, exhaustive)      ║");
    println!("║  Can 1 LutGate neuron pass each reasoning test?                       ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║ Level          │ In │ Train  │ Test   │ Linear?│ Verdict               ║");
    println!("╠════════════════╪════╪════════╪════════╪════════╪═══════════════════════╣");

    let mut pass = 0;
    let mut total = 0;

    for level in &levels {
        total += 1;
        let lt = Instant::now();

        let (w, b, thresh, train_acc) = exhaustive_1neuron(&level.train, level.n_in);
        let test_acc = eval_neuron(&w, b, thresh, &level.test);

        let verdict = if test_acc >= 100.0 {
            pass += 1;
            "PASS ✓ (100%)"
        } else if test_acc >= 90.0 {
            pass += 1;
            "PASS ~ (≥90%)"
        } else if !level.can_1neuron {
            "FAIL (expected)"
        } else {
            "FAIL ✗"
        };

        let _elapsed = lt.elapsed();

        println!("║ {:14} │ {:2} │ {:5.1}% │ {:5.1}% │   {}    │ {:21} ║",
            level.name, level.n_in,
            train_acc, test_acc,
            if level.can_1neuron { "Y" } else { "N" },
            verdict,
        );

        // Show weights for interesting cases
        if test_acc >= 90.0 || level.can_1neuron {
            let w_str: Vec<String> = w.iter().map(|&v| match v { 1 => "+".into(), -1 => "-".into(), _ => "0".into() }).collect();
            println!("║   weights: [{}] b={:+} t={}{}",
                w_str.join(""), b, thresh,
                " ".repeat(47 - w_str.len()*1 - 10) + "║");
        }
    }

    println!("╠════════════════╧════╧════════╧════════╧════════╧═══════════════════════╣");
    println!("║  IQ Score: {}/{} levels passed │ Time: {:.1}s                             ║",
        pass, total, t0.elapsed().as_secs_f64());
    println!("╠══════════════════════════════════════════════════════════════════════════╣");

    println!("║                                                                        ║");
    println!("║  Level descriptions:                                                   ║");
    for level in &levels {
        println!("║  {:14} — {:52} ║", level.name, level.desc);
    }
    println!("║                                                                        ║");
    println!("║  Linear? = theoretically solvable by 1 linear threshold neuron         ║");
    println!("║  Train = accuracy on seen examples, Test = accuracy on UNSEEN          ║");
    println!("║  Method: ternary exhaustive search (all 3^(N+1) weight combos)         ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
}
