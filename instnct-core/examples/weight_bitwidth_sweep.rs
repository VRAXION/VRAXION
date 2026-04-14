//! Weight bit-width sweep — minimum neurons at each precision
//!
//! Binary {-1,+1}: 4 neurons for 100%
//! Ternary {-1,0,+1}: 3 neurons for 100%
//! 2-bit {-2..+2}: ? neurons
//! 3-bit {-4..+4}: ? neurons
//! Can we get to 2 neurons with wider weights?
//!
//! All are "binary" storage: N-bit integer = N bits per weight
//!
//! Run: cargo run --example weight_bitwidth_sweep --release

use std::time::Instant;

fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c=c.max(0.1); let rho=rho.max(0.0); let l=6.0*c;
    if x>=l{return x-l;} if x<=-l{return x+l;}
    let s=x/c; let n=s.floor(); let t=s-n; let h=t*(1.0-t);
    let sg=if(n as i32)%2==0{1.0}else{-1.0}; c*(sg*h+rho*h*h)
}

fn eval_neurons(weights: &[(Vec<i8>, i8)], cs: &[f32], rhos: &[f32], n: usize) -> usize {
    let codes: Vec<Vec<f32>> = (0..27u8).map(|ch| {
        let mut bits=[0.0f32;8]; for i in 0..8{bits[i]=((ch>>i)&1) as f32;}
        (0..n).map(|k| {
            let mut d=weights[k].1 as f32;
            for j in 0..8{d+=weights[k].0[j] as f32*bits[j];}
            c19(d, cs[k], rhos[k])
        }).collect()
    }).collect();
    let mut ok=0;
    for i in 0..27{let mut best=0; let mut bd=f32::MAX;
        for j in 0..27{let d:f32=codes[i].iter().zip(&codes[j]).map(|(a,b)|(a-b)*(a-b)).sum(); if d<bd{bd=d;best=j;}}
        if best==i{ok+=1;}} ok
}

fn greedy_search(n_neurons: usize, weight_range: &[i8], c_vals: &[f32], rho_vals: &[f32])
    -> (Vec<(Vec<i8>, i8)>, Vec<f32>, Vec<f32>, usize)
{
    let n_vals = weight_range.len();
    let combos_per_neuron = n_vals.pow(9); // 8 weights + 1 bias

    let mut neurons: Vec<(Vec<i8>, i8)> = Vec::new();
    let mut cs: Vec<f32> = Vec::new();
    let mut rhos: Vec<f32> = Vec::new();

    for neuron in 0..n_neurons {
        let mut top_score = 0;
        let mut top_w = vec![0i8;8];
        let mut top_b = 0i8;
        let mut top_c = 1.0f32;
        let mut top_rho = 0.0f32;

        for &cv in c_vals {
            for &rv in rho_vals {
                for combo in 0..combos_per_neuron {
                    let mut w = vec![0i8;8];
                    let mut rem = combo;
                    for j in 0..8 {
                        w[j] = weight_range[rem % n_vals];
                        rem /= n_vals;
                    }
                    let b = weight_range[rem % n_vals];

                    let mut test = neurons.clone();
                    test.push((w.clone(), b));
                    let mut tc = cs.clone(); tc.push(cv);
                    let mut tr = rhos.clone(); tr.push(rv);

                    let score = eval_neurons(&test, &tc, &tr, neuron+1);
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

    println!("=== WEIGHT BIT-WIDTH SWEEP ===\n");
    println!("  27 symbols, 8 input bits, C19 activation");
    println!("  Question: minimum neurons at each weight precision?\n");

    let c_vals = vec![1.0, 2.0, 5.0, 10.0, 20.0];
    let rho_vals = vec![0.0, 0.5, 1.0, 2.0];

    struct Cfg {
        name: &'static str,
        range: Vec<i8>,
        bits_per_weight: f32,
        max_neurons: usize,
    }

    let configs = vec![
        Cfg { name: "1-bit {-1,+1}", range: vec![-1, 1], bits_per_weight: 1.0, max_neurons: 5 },
        Cfg { name: "ternary {-1,0,+1}", range: vec![-1, 0, 1], bits_per_weight: 1.58, max_neurons: 4 },
        Cfg { name: "2-bit {-2..+2}", range: vec![-2,-1,0,1,2], bits_per_weight: 2.32, max_neurons: 4 },
        Cfg { name: "2-bit {-3..+3}", range: vec![-3,-2,-1,0,1,2,3], bits_per_weight: 2.81, max_neurons: 3 },
        Cfg { name: "3-bit {-4..+4}", range: vec![-4,-3,-2,-1,0,1,2,3,4], bits_per_weight: 3.17, max_neurons: 3 },
    ];

    println!("  {:>22} {:>6} {:>12} {:>10} {:>14} {:>8}",
        "weight_type", "bits/w", "combos/n", "min_N", "total_storage", "time");
    println!("  {}", "─".repeat(78));

    for cfg in &configs {
        let tc = Instant::now();
        let combos = cfg.range.len().pow(9);

        if combos > 50_000_000 {
            println!("  {:>22} {:>5.1}b {:>12}    SKIP (too many combos)",
                cfg.name, cfg.bits_per_weight, combos);
            continue;
        }

        let mut min_n = 0;
        for n in 2..=cfg.max_neurons {
            let (neurons, cs, rhos, score) = greedy_search(n, &cfg.range, &c_vals, &rho_vals);
            if score == 27 {
                min_n = n;
                let total_bits = (n as f32 * 9.0 * cfg.bits_per_weight).ceil() as usize;
                let total_bytes = (total_bits + 7) / 8;
                println!("  {:>22} {:>5.1}b {:>12} {:>8} N {:>10} bits {:>7.1}s ★★★",
                    cfg.name, cfg.bits_per_weight, combos, n, total_bits, tc.elapsed().as_secs_f64());

                // Print winning config
                for k in 0..n {
                    let w_str: String = neurons[k].0.iter().map(|&v|
                        if v>=0{format!("+{}",v)}else{format!("{}",v)}
                    ).collect::<Vec<_>>().join(",");
                    let zeros = neurons[k].0.iter().filter(|&&v| v==0).count();
                    println!("    N{}: [{}] b={:+} c={} rho={} ({}% sparse)",
                        k, w_str, neurons[k].1, cs[k], rhos[k],
                        zeros*100/8);
                }
                break;
            } else {
                println!("  {:>22} {:>5.1}b {:>12} {:>5} N={} {}/27 {:.1}s",
                    cfg.name, cfg.bits_per_weight, combos, "", n, score, tc.elapsed().as_secs_f64());
            }
        }
        if min_n == 0 {
            println!("  {:>22} — no 100% found up to {} neurons", cfg.name, cfg.max_neurons);
        }
        println!();
    }

    // Summary
    println!("━━━ SUMMARY ━━━\n");
    println!("  bits/w × min_neurons × 9 params = total info");
    println!("  Kevesebb neuron VAGY kevesebb bit → kisebb modell\n");

    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
