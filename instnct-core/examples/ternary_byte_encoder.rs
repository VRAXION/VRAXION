//! Ternary byte encoder — {-1, 0, +1} weights, exhaustive search
//!
//! Goals:
//!   1. How sparse? How many weights become 0 (skip)?
//!   2. Minimum neurons for 100% round-trip with ternary?
//!   3. Compare: ternary vs binary-only — does 0 help?
//!   4. Deploy format: add_list + sub_list per neuron
//!
//! Run: cargo run --example ternary_byte_encoder --release

use std::time::Instant;

fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.1); let rho = rho.max(0.0); let l = 6.0*c;
    if x >= l { return x-l; } if x <= -l { return x+l; }
    let s = x/c; let n = s.floor(); let t = s-n; let h = t*(1.0-t);
    let sg = if (n as i32)%2==0 { 1.0 } else { -1.0 }; c*(sg*h+rho*h*h)
}

fn eval_neurons(weights: &[([i8;8], i8)], cs: &[f32], rhos: &[f32], n: usize) -> usize {
    let codes: Vec<Vec<f32>> = (0..27u8).map(|ch| {
        let mut bits = [0.0f32;8]; for i in 0..8 { bits[i] = ((ch>>i)&1) as f32; }
        (0..n).map(|k| {
            let mut d = weights[k].1 as f32;
            for j in 0..8 { d += weights[k].0[j] as f32 * bits[j]; }
            c19(d, cs[k], rhos[k])
        }).collect()
    }).collect();
    let mut ok = 0;
    for i in 0..27 {
        let mut best = 0; let mut bd = f32::MAX;
        for j in 0..27 {
            let d: f32 = codes[i].iter().zip(&codes[j]).map(|(a,b)| (a-b)*(a-b)).sum();
            if d < bd { bd = d; best = j; }
        }
        if best == i { ok += 1; }
    }
    ok
}

fn greedy_search_ternary(n_neurons: usize, c_vals: &[f32], rho_vals: &[f32])
    -> (Vec<([i8;8], i8)>, Vec<f32>, Vec<f32>, usize)
{
    let mut neurons: Vec<([i8;8], i8)> = Vec::new();
    let mut cs: Vec<f32> = Vec::new();
    let mut rhos: Vec<f32> = Vec::new();

    for neuron in 0..n_neurons {
        let mut top_score = 0;
        let mut top_w = [0i8;8];
        let mut top_b = 0i8;
        let mut top_c = 1.0f32;
        let mut top_rho = 0.0f32;

        for &cv in c_vals {
            for &rv in rho_vals {
                // 3^9 = 19683 combos: 8 weights + 1 bias, each {-1, 0, +1}
                for combo in 0..19683u32 {
                    let mut w = [0i8;8];
                    let mut rem = combo;
                    for j in 0..8 {
                        w[j] = (rem % 3) as i8 - 1; // maps 0,1,2 → -1,0,+1
                        rem /= 3;
                    }
                    let b = (rem % 3) as i8 - 1;

                    let mut test = neurons.clone();
                    test.push((w, b));
                    let mut tc = cs.clone(); tc.push(cv);
                    let mut tr = rhos.clone(); tr.push(rv);

                    let score = eval_neurons(&test, &tc, &tr, neuron + 1);
                    if score > top_score {
                        top_score = score; top_w = w; top_b = b; top_c = cv; top_rho = rv;
                        if top_score == 27 { break; }
                    }
                }
                if top_score == 27 { break; }
            }
            if top_score == 27 { break; }
        }

        neurons.push((top_w, top_b));
        cs.push(top_c);
        rhos.push(top_rho);
    }

    let final_score = eval_neurons(&neurons, &cs, &rhos, n_neurons);
    (neurons, cs, rhos, final_score)
}

fn greedy_search_binary(n_neurons: usize, c_vals: &[f32], rho_vals: &[f32])
    -> (Vec<([i8;8], i8)>, Vec<f32>, Vec<f32>, usize)
{
    let mut neurons: Vec<([i8;8], i8)> = Vec::new();
    let mut cs: Vec<f32> = Vec::new();
    let mut rhos: Vec<f32> = Vec::new();

    for neuron in 0..n_neurons {
        let mut top_score = 0;
        let mut top_w = [0i8;8];
        let mut top_b = 0i8;
        let mut top_c = 1.0f32;
        let mut top_rho = 0.0f32;

        for &cv in c_vals {
            for &rv in rho_vals {
                for combo in 0..512u32 {
                    let mut w = [0i8;8];
                    for j in 0..8 { w[j] = if (combo>>j)&1==1 { 1 } else { -1 }; }
                    let b = if (combo>>8)&1==1 { 1i8 } else { -1 };

                    let mut test = neurons.clone();
                    test.push((w, b));
                    let mut tc = cs.clone(); tc.push(cv);
                    let mut tr = rhos.clone(); tr.push(rv);

                    let score = eval_neurons(&test, &tc, &tr, neuron + 1);
                    if score > top_score {
                        top_score = score; top_w = w; top_b = b; top_c = cv; top_rho = rv;
                        if top_score == 27 { break; }
                    }
                }
                if top_score == 27 { break; }
            }
            if top_score == 27 { break; }
        }

        neurons.push((top_w, top_b));
        cs.push(top_c);
        rhos.push(top_rho);
    }

    let final_score = eval_neurons(&neurons, &cs, &rhos, n_neurons);
    (neurons, cs, rhos, final_score)
}

fn main() {
    let t0 = Instant::now();
    let c_vals = vec![1.0, 2.0, 5.0, 10.0, 20.0];
    let rho_vals = vec![0.0, 0.5, 1.0, 2.0];

    println!("=== TERNARY vs BINARY BYTE ENCODER ===\n");
    println!("  27 symbols, 8 input bits, greedy neuron-by-neuron exhaustive\n");

    // ── Sweep: neuron count 2-8, ternary vs binary ──
    println!("  {:>2} {:>10} {:>10} {:>8}",
        "N", "binary", "ternary", "time");
    println!("  {}", "─".repeat(35));

    for n in 2..=8 {
        let tc = Instant::now();
        let (_, _, _, b_score) = greedy_search_binary(n, &c_vals, &rho_vals);
        let (t_neurons, t_cs, t_rhos, t_score) = greedy_search_ternary(n, &c_vals, &rho_vals);
        let bm = if b_score==27{"★★★"}else{""};
        let tm = if t_score==27{"★★★"}else{""};

        println!("  {:>2} {:>5}/27 {:>3} {:>5}/27 {:>3} {:>7.1}s",
            n, b_score, bm, t_score, tm, tc.elapsed().as_secs_f64());

        // If ternary hit 100%, print details
        if t_score == 27 {
            println!("\n  ★★★ TERNARY {} NEURONS = 100% ★★★", n);
            let mut total_add = 0; let mut total_sub = 0; let mut total_skip = 0;
            for k in 0..n {
                let (w, b) = &t_neurons[k];
                let adds: Vec<usize> = (0..8).filter(|&j| w[j] == 1).collect();
                let subs: Vec<usize> = (0..8).filter(|&j| w[j] == -1).collect();
                let skips: Vec<usize> = (0..8).filter(|&j| w[j] == 0).collect();
                total_add += adds.len(); total_sub += subs.len(); total_skip += skips.len();

                let bias_str = if b == &1 { "+1" } else if b == &-1 { "-1" } else { " 0" };
                println!("    N{}: bias={} add=bit{:?} sub=bit{:?} skip=bit{:?} c={} rho={}",
                    k, bias_str, adds, subs, skips, t_cs[k], t_rhos[k]);
            }
            let total_conn = total_add + total_sub;
            let total_possible = n * 8;
            let sparsity = total_skip as f64 / total_possible as f64 * 100.0;
            println!("\n  Connections: {} active / {} possible ({:.0}% sparse)",
                total_conn, total_possible, sparsity);
            println!("  Ops per byte: {} add + {} sub = {} total (was {} with binary)",
                total_add, total_sub, total_conn, total_possible);
            println!("  Deploy: {} entries in add_list + sub_list", total_conn);

            // If found early, continue sweep for smaller N
            if n <= 4 { continue; }
            break;
        }
    }

    // ── Also: binary sweep for comparison ──
    println!("\n━━━ BINARY-ONLY (for comparison) ━━━\n");
    for n in 2..=7 {
        let (_, _, _, score) = greedy_search_binary(n, &c_vals, &rho_vals);
        let m = if score==27{"★★★"}else{""};
        println!("  N={}: {}/27 {}", n, score, m);
        if score == 27 { println!("  → Binary minimum: {} neurons", n); break; }
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
