//! Byte interpreter minimum neurons — how few signals for 100% round-trip?
//!
//! Current: 7 neurons (binary {-1,+1}, C19) = 100% on 27 symbols
//! Theory: log2(27) = 4.75 → minimum 5 neurons needed
//!
//! Test: exhaustive search at each neuron count (2-7) with C19 and binary weights.
//! Also test: without C19 (pure linear) and with different c/rho values.
//!
//! Then: measure downstream prediction impact of fewer byte interpreter signals.
//!
//! Run: cargo run --example byte_interp_min_neurons --release

use std::time::Instant;

struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 { self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.s }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn normal(&mut self) -> f32 { let u1 = self.f32().max(1e-7); let u2 = self.f32(); (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() }
    fn range(&mut self, lo: usize, hi: usize) -> usize { if hi <= lo { lo } else { lo + (self.next() as usize % (hi - lo)) } }
}

fn load_corpus(path: &str) -> Vec<u8> {
    let raw = std::fs::read(path).expect("read");
    let mut c = Vec::new();
    for &b in &raw { match b { b'a'..=b'z' => c.push(b-b'a'), b'A'..=b'Z' => c.push(b-b'A'), b' '|b'\n'|b'\t'|b'\r' => c.push(26), _ => {} } }
    c
}

fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.1); let rho = rho.max(0.0); let l = 6.0*c;
    if x >= l { return x-l; } if x <= -l { return x+l; }
    let s = x/c; let n = s.floor(); let t = s-n; let h = t*(1.0-t);
    let sg = if (n as i32)%2==0 { 1.0 } else { -1.0 }; c*(sg*h+rho*h*h)
}

fn encode_byte_flex(ch: u8, weights: &[[i8;8]], biases: &[i8], cs: &[f32], rhos: &[f32], n_neurons: usize) -> Vec<f32> {
    let mut bits = [0.0f32;8];
    for i in 0..8 { bits[i] = ((ch>>i)&1) as f32; }
    let mut o = vec![0.0f32; n_neurons];
    for k in 0..n_neurons {
        let mut d = biases[k] as f32;
        for j in 0..8 { d += weights[k][j] as f32 * bits[j]; }
        o[k] = c19(d, cs[k], rhos[k]);
    }
    o
}

fn roundtrip_accuracy(weights: &[[i8;8]], biases: &[i8], cs: &[f32], rhos: &[f32], n_neurons: usize) -> usize {
    let unique: Vec<u8> = (0..27).collect();
    let codes: Vec<Vec<f32>> = unique.iter().map(|&ch| encode_byte_flex(ch, weights, biases, cs, rhos, n_neurons)).collect();
    let mut ok = 0;
    for (i, code) in codes.iter().enumerate() {
        let mut best = 0; let mut bd = f32::MAX;
        for (j, other) in codes.iter().enumerate() {
            let d: f32 = code.iter().zip(other).map(|(a,b)| (a-b)*(a-b)).sum();
            if d < bd { bd = d; best = j; }
        }
        if best == i { ok += 1; }
    }
    ok
}

// Greedy neuron-by-neuron exhaustive search
fn greedy_search(n_neurons: usize, c_vals: &[f32], rho_vals: &[f32]) -> (Vec<[i8;8]>, Vec<i8>, Vec<f32>, Vec<f32>, usize) {
    let mut best_w: Vec<[i8;8]> = Vec::new();
    let mut best_b: Vec<i8> = Vec::new();
    let mut best_c: Vec<f32> = Vec::new();
    let mut best_rho: Vec<f32> = Vec::new();

    for neuron in 0..n_neurons {
        let mut top_score = 0usize;
        let mut top_nw = [0i8;8];
        let mut top_nb = 0i8;
        let mut top_c = 1.0f32;
        let mut top_rho = 0.0f32;

        for &cv in c_vals {
            for &rv in rho_vals {
                // 2^9 = 512 combos for weights + bias
                for combo in 0..512u32 {
                    let mut nw = [0i8;8];
                    for j in 0..8 { nw[j] = if (combo>>j)&1==1 { 1 } else { -1 }; }
                    let nb = if (combo>>8)&1==1 { 1i8 } else { -1 };

                    let mut w = best_w.clone(); w.push(nw);
                    let mut b = best_b.clone(); b.push(nb);
                    let mut c = best_c.clone(); c.push(cv);
                    let mut r = best_rho.clone(); r.push(rv);

                    let score = roundtrip_accuracy(&w, &b, &c, &r, neuron+1);
                    if score > top_score {
                        top_score = score; top_nw = nw; top_nb = nb; top_c = cv; top_rho = rv;
                        if top_score == 27 { break; }
                    }
                }
                if top_score == 27 { break; }
            }
            if top_score == 27 { break; }
        }

        best_w.push(top_nw);
        best_b.push(top_nb);
        best_c.push(top_c);
        best_rho.push(top_rho);
    }

    let final_score = roundtrip_accuracy(&best_w, &best_b, &best_c, &best_rho, n_neurons);
    (best_w, best_b, best_c, best_rho, final_score)
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");

    println!("=== BYTE INTERPRETER MINIMUM NEURONS ===\n");
    println!("  27 symbols (a-z + space), theoretical minimum: ceil(log2(27)) = 5 neurons\n");

    let c_vals = vec![1.0, 2.0, 5.0, 10.0, 20.0];
    let rho_vals = vec![0.0, 0.5, 1.0, 2.0];

    println!("━━━ Greedy exhaustive search (binary weights, C19 activation) ━━━\n");
    println!("  {:>3} {:>6} {:>10} {:>8}",
        "N", "rt_acc", "bits/model", "time");
    println!("  {}", "─".repeat(35));

    for n in 2..=7 {
        let tc = Instant::now();
        let (w, b, c, rho, score) = greedy_search(n, &c_vals, &rho_vals);
        let model_bits = n * (8 + 1); // 8 weights + 1 bias, each 1 bit (plus c/rho choice)

        let mark = if score == 27 { " ★★★ 100%!" } else if score >= 25 { " ★★" } else if score >= 20 { " ★" } else { "" };
        println!("  {:>3} {:>3}/{:>2} {:>6} bits {:>7.1}s{}",
            n, score, 27, model_bits, tc.elapsed().as_secs_f64(), mark);

        if score == 27 {
            println!("    → MINIMUM FOUND: {} neurons = 100% round-trip!", n);
            println!("    c={:?}, rho={:?}", &c[..n], &rho[..n]);

            // Also test downstream prediction with this encoder
            let enc = move |chars: &[u8]| -> Vec<f32> {
                chars.iter().flat_map(|&ch| encode_byte_flex(ch, &w, &b, &c, &rho, n)).collect()
            };

            // Quick prediction test: 1L brain, 100 epochs
            let ctx = 16;
            let idim = ctx * n;
            let hdim = 256;
            let mut rng = Rng::new(42);
            let s1=(2.0/idim as f32).sqrt(); let s2=(2.0/hdim as f32).sqrt();
            let mut w1: Vec<Vec<f32>>=(0..hdim).map(|_|(0..idim).map(|_|rng.normal()*s1).collect()).collect();
            let mut b1=vec![0.0f32;hdim];
            let mut w2: Vec<Vec<f32>>=(0..27).map(|_|(0..hdim).map(|_|rng.normal()*s2).collect()).collect();
            let mut b2=vec![0.0f32;27];

            let samples=10000.min(corpus.len()/(ctx+1));
            for ep in 0..100 {
                let lr=0.01*(1.0-ep as f32/100.0*0.7);
                let mut rt=Rng::new(ep as u64*1000+42);
                for _ in 0..samples {
                    let off=rt.range(0,corpus.len()-ctx-1);
                    let input=enc(&corpus[off..off+ctx]);
                    let target=corpus[off+ctx] as usize;
                    let mut h=vec![0.0f32;hdim];
                    for k in 0..hdim { h[k]=b1[k]; for j in 0..idim { h[k]+=w1[k][j]*input[j]; } h[k]=h[k].max(0.0); }
                    let mut logits=vec![0.0f32;27];
                    for c2 in 0..27 { logits[c2]=b2[c2]; for k in 0..hdim { logits[c2]+=w2[c2][k]*h[k]; } }
                    let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
                    let mut p=vec![0.0f32;27]; let mut s=0.0f32;
                    for c2 in 0..27 { p[c2]=(logits[c2]-mx).exp(); s+=p[c2]; } for c2 in 0..27 { p[c2]/=s; }
                    let mut dl=p; dl[target]-=1.0;
                    let mut dh=vec![0.0f32;hdim];
                    for c2 in 0..27 { for k in 0..hdim { dh[k]+=dl[c2]*w2[c2][k]; w2[c2][k]-=lr*dl[c2]*h[k]; } b2[c2]-=lr*dl[c2]; }
                    for k in 0..hdim { if h[k]<=0.0{continue;} for j in 0..idim { w1[k][j]-=lr*dh[k]*input[j]; } b1[k]-=lr*dh[k]; }
                }
            }
            let mut eval_rng=Rng::new(999); let mut ok=0; let mut tot=0;
            for _ in 0..5000 { if corpus.len()<ctx+1{break;} let off=eval_rng.range(0,corpus.len()-ctx-1);
                let input=enc(&corpus[off..off+ctx]);
                let mut h=vec![0.0f32;hdim];
                for k in 0..hdim { h[k]=b1[k]; for j in 0..idim { h[k]+=w1[k][j]*input[j]; } h[k]=h[k].max(0.0); }
                let mut logits=vec![0.0f32;27];
                for c2 in 0..27 { logits[c2]=b2[c2]; for k in 0..hdim { logits[c2]+=w2[c2][k]*h[k]; } }
                let pred=logits.iter().enumerate().max_by(|a,b2| a.1.partial_cmp(b2.1).unwrap()).unwrap().0;
                if pred==corpus[off+ctx] as usize {ok+=1;} tot+=1;
            }
            let pred_acc = ok as f64/tot as f64*100.0;
            println!("    Downstream prediction (1L h=256 100ep): {:.1}%", pred_acc);
            break; // found minimum, stop
        }
    }

    // Also test: no C19 (identity activation)
    println!("\n━━━ Without C19 (linear activation) ━━━\n");
    for n in 2..=7 {
        let tc = Instant::now();
        let c_id = vec![100.0]; // large c → c19 becomes ~identity
        let rho_id = vec![0.0];
        let (_, _, _, _, score) = greedy_search(n, &c_id, &rho_id);
        let mark = if score == 27 { " ★★★" } else { "" };
        println!("  N={}: {}/27{}", n, score, mark);
        if score == 27 { println!("    → Linear: {} neurons suffice!", n); break; }
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
