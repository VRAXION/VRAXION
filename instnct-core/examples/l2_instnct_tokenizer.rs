//! L2 INSTNCT Tokenizer — greedy growth, round-trip encode/decode
//!
//! Input: 64 bytes × 2ch = 128 values
//! Output: 96 values (~75% compression)
//! Task: encode → decode → 100% per-byte round-trip
//! Method: INSTNCT greedy neuron growth (encoder) + simple MLP decoder
//!
//! Run: cargo run --example l2_instnct_tokenizer --release

use std::time::Instant;

const LUT: [[i8; 2]; 27] = [
    [-2,-4],[-4,-2],[0,-6],[-2,-5],[-1,-6],[-3,-5],[1,-8],[-1,-7],
    [-2,-6],[-4,-5],[0,-8],[-2,-7],[-1,-8],[-3,-7],[1,-10],[-1,-9],
    [-4,-6],[-6,-5],[-2,-8],[-4,-7],[-3,-8],[-5,-7],[-1,-10],[-3,-9],
    [-4,-8],[-6,-7],[-2,-10],
];

fn c19a(x:f32,c:f32,r:f32)->f32{let c=c.max(0.1);let r=r.max(0.0);let l=6.0*c;
    if x>=l{return x-l;}if x<=-l{return x+l;}
    let s=x/c;let n=s.floor();let t=s-n;let h=t*(1.0-t);
    let sg=if(n as i32)%2==0{1.0}else{-1.0};c*(sg*h+r*h*h)}

fn swish(x:f32)->f32{x/(1.0+(-x).exp())}
fn swish_g(x:f32)->f32{let s=1.0/(1.0+(-x).exp());s+x*s*(1.0-s)}

struct Rng(u64);
impl Rng{
    fn new(s:u64)->Self{Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1))}
    fn next(&mut self)->u64{self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);self.0}
    fn normal(&mut self)->f32{let u1=(((self.next()>>33)%65536)as f32/65536.0).max(1e-7);let u2=((self.next()>>33)%65536)as f32/65536.0;(-2.0*u1.ln()).sqrt()*(2.0*std::f32::consts::PI*u2).cos()}
    fn range(&mut self,lo:usize,hi:usize)->usize{if hi<=lo{lo}else{lo+(self.next()as usize%(hi-lo))}}
    fn rangef(&mut self,lo:f32,hi:f32)->f32{lo+((self.next()>>33)%65536)as f32/65536.0*(hi-lo)}
    fn choose_n(&mut self,n:usize,max:usize)->Vec<usize>{
        let mut p=Vec::new();while p.len()<n{let v=self.range(0,max);if!p.contains(&v){p.push(v);}}p}
}

fn load_corpus(p:&str)->Vec<u8>{std::fs::read(p).expect("r").iter().filter_map(|&b|match b{
    b'a'..=b'z'=>Some(b-b'a'),b'A'..=b'Z'=>Some(b-b'A'),b' '|b'\n'|b'\t'|b'\r'=>Some(26),_=>None}).collect()}

fn nearest_lut(v0:f32,v1:f32)->u8{
    let mut b=0u8;let mut bd=f32::MAX;
    for s in 0..27u8{let d0=v0-LUT[s as usize][0]as f32/16.0;let d1=v1-LUT[s as usize][1]as f32/16.0;
        let d=d0*d0+d1*d1;if d<bd{bd=d;b=s;}}b}

const CTX: usize = 64;
const IN_DIM: usize = CTX * 2;     // 128
const OUT_DIM: usize = 96;          // ~75% of input

// INSTNCT sparse neuron
#[derive(Clone)]
struct Neuron {
    sources: Vec<usize>,
    weights: Vec<f32>,
    bias: f32, c: f32, rho: f32,
}
impl Neuron {
    fn eval(&self, nodes: &[f32]) -> f32 {
        let mut dot = self.bias;
        for (i, &src) in self.sources.iter().enumerate() {
            if src < nodes.len() { dot += self.weights[i] * nodes[src]; }
        }
        c19a(dot, self.c, self.rho)
    }
}

// Encoder: inputs → hidden → OUT_DIM outputs
struct Encoder {
    hidden: Vec<Neuron>,
    // Output: linear read from nodes → OUT_DIM values
    out_sources: Vec<Vec<usize>>,  // [out_dim][fan_in]
    out_weights: Vec<Vec<f32>>,
    out_bias: Vec<f32>,
}

impl Encoder {
    fn new(fan_in: usize, rng: &mut Rng) -> Self {
        let mut out_sources = Vec::new();
        let mut out_weights = Vec::new();
        let mut out_bias = Vec::new();
        for _ in 0..OUT_DIM {
            let src = rng.choose_n(fan_in.min(IN_DIM), IN_DIM);
            let w: Vec<f32> = src.iter().map(|_| rng.rangef(-1.0, 1.0)).collect();
            out_sources.push(src);
            out_weights.push(w);
            out_bias.push(0.0);
        }
        Encoder { hidden: Vec::new(), out_sources, out_weights, out_bias }
    }

    fn n_nodes(&self) -> usize { IN_DIM + self.hidden.len() }

    fn encode(&self, input: &[f32]) -> Vec<f32> {
        let mut nodes = input.to_vec();
        for h in &self.hidden { nodes.push(h.eval(&nodes)); }
        // Read output
        (0..OUT_DIM).map(|j| {
            let mut v = self.out_bias[j];
            for (i, &src) in self.out_sources[j].iter().enumerate() {
                if src < nodes.len() { v += self.out_weights[j][i] * nodes[src]; }
            }
            v
        }).collect()
    }
}

// Decoder: MLP OUT_DIM → IN_DIM (simple, trained with random search)
struct Decoder {
    w1: Vec<Vec<f32>>, b1: Vec<f32>,
    w2: Vec<Vec<f32>>, b2: Vec<f32>,
    hid: usize,
}
impl Decoder {
    fn new(hid: usize, rng: &mut Rng) -> Self {
        let s1=(2.0/OUT_DIM as f32).sqrt(); let s2=(2.0/hid as f32).sqrt();
        Decoder{
            w1:(0..hid).map(|_|(0..OUT_DIM).map(|_|rng.normal()*s1).collect()).collect(),
            b1:vec![0.0;hid],
            w2:(0..IN_DIM).map(|_|(0..hid).map(|_|rng.normal()*s2).collect()).collect(),
            b2:vec![0.0;IN_DIM],hid}
    }
    fn decode(&self, encoded: &[f32]) -> Vec<f32> {
        let mut h = self.b1.clone();
        for j in 0..self.hid { for k in 0..OUT_DIM { h[j] += self.w1[j][k]*encoded[k]; } h[j]=swish(h[j]); }
        let mut out = self.b2.clone();
        for j in 0..IN_DIM { for k in 0..self.hid { out[j] += self.w2[j][k]*h[k]; } }
        out
    }
    fn train_step(&mut self, encoded: &[f32], target: &[f32], lr: f32) {
        let mut h = self.b1.clone();
        let mut h_pre = self.b1.clone();
        for j in 0..self.hid { for k in 0..OUT_DIM { h_pre[j]+=self.w1[j][k]*encoded[k]; } h[j]=swish(h_pre[j]); }
        let mut out = self.b2.clone();
        for j in 0..IN_DIM { for k in 0..self.hid { out[j]+=self.w2[j][k]*h[k]; } }
        // MSE gradient
        let mut dout = vec![0.0f32;IN_DIM];
        for j in 0..IN_DIM { dout[j] = 2.0*(out[j]-target[j])/IN_DIM as f32; }
        let mut dh = vec![0.0f32;self.hid];
        for j in 0..IN_DIM { for k in 0..self.hid { dh[k]+=dout[j]*self.w2[j][k]; self.w2[j][k]-=lr*dout[j]*h[k]; } self.b2[j]-=lr*dout[j]; }
        for j in 0..self.hid { let g=dh[j]*swish_g(h_pre[j]);
            for k in 0..OUT_DIM { self.w1[j][k]-=lr*g*encoded[k]; } self.b1[j]-=lr*g; }
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let split = corpus.len()*80/100;
    let encoded: Vec<[f32;2]> = corpus.iter().map(|&ch|
        [LUT[ch as usize][0]as f32/16.0, LUT[ch as usize][1]as f32/16.0]).collect();

    println!("=== INSTNCT TOKENIZER (round-trip) ===\n");
    println!("  Input: {} bytes = {} values", CTX, IN_DIM);
    println!("  Output: {} values ({:.0}% of input)", OUT_DIM, OUT_DIM as f64/IN_DIM as f64*100.0);
    println!("  Task: encode → decode → 100% per-byte round-trip\n");

    let fan_in = 6;
    let mut rng = Rng::new(42);
    let mut enc = Encoder::new(fan_in, &mut rng);
    let mut dec = Decoder::new(128, &mut rng);

    // Build eval chunks
    let stride = 16;
    let n_chunks = (split - CTX) / stride;
    let n_test = (corpus.len() - split - CTX).max(1) / stride;

    let make_input = |off: usize| -> Vec<f32> {
        (0..CTX).flat_map(|i| {
            if off+i < encoded.len() { vec![encoded[off+i][0], encoded[off+i][1]] }
            else { vec![0.0, 0.0] }
        }).collect()
    };

    let eval_roundtrip = |enc: &Encoder, dec: &Decoder, start: usize, end: usize, n: usize| -> f64 {
        let mut rng3 = Rng::new(999);
        let mut ok = 0usize; let mut tot = 0usize;
        for _ in 0..n {
            if end < start + CTX { break; }
            let off = rng3.range(start, end.saturating_sub(CTX));
            let input = make_input(off);
            let tokens = enc.encode(&input);
            let recon = dec.decode(&tokens);
            for p in 0..CTX {
                let orig = corpus[off+p];
                let pred = nearest_lut(recon[p*2], recon[p*2+1]);
                if pred == orig { ok += 1; }
                tot += 1;
            }
        }
        if tot==0{0.0}else{ok as f64/tot as f64*100.0}
    };

    // Phase 1: train decoder on current (random) encoder
    println!("  Phase 1: Train decoder on random encoder...");
    for ep in 0..200 {
        let lr = 0.005*(1.0-ep as f32/200.0*0.8);
        let mut rt = Rng::new(ep as u64*1000+42);
        for _ in 0..500.min(n_chunks) {
            let ci = rt.range(0, n_chunks);
            let off = ci * stride;
            let input = make_input(off);
            let tokens = enc.encode(&input);
            dec.train_step(&tokens, &input, lr);
        }
    }
    let tr = eval_roundtrip(&enc, &dec, 0, split, 200);
    let te = eval_roundtrip(&enc, &dec, split, corpus.len(), 200);
    println!("  Baseline (random encoder): train={:.1}% test={:.1}%\n", tr, te);

    // Phase 2: INSTNCT greedy growth + retrain decoder after each neuron
    let max_neurons = 100;
    let mutations = 3000;

    println!("  {:>4} {:>8} {:>8} {:>8} {:>6}",
        "#N", "train%", "test%", "delta", "time");
    println!("  {}", "-".repeat(42));

    let mut best_train = tr;

    for ni in 0..max_neurons {
        let n_nodes = enc.n_nodes();

        // Try random neurons
        let mut best_neuron: Option<Neuron> = None;
        let mut best_out_mod: Option<(usize, usize, usize, f32)> = None; // (out_j, slot, new_src, new_w)
        let mut best_score = best_train;

        for _ in 0..mutations {
            let sources = rng.choose_n(fan_in.min(n_nodes), n_nodes);
            let weights: Vec<f32> = sources.iter().map(|_| rng.rangef(-3.0, 3.0)).collect();
            let neuron = Neuron { sources, weights, bias: rng.rangef(-2.0, 2.0),
                c: rng.rangef(1.0, 25.0), rho: rng.rangef(0.0, 3.0) };

            enc.hidden.push(neuron.clone());
            let new_idx = enc.n_nodes() - 1;

            // Also rewire one output to use new neuron
            let oj = rng.range(0, OUT_DIM);
            let slot = rng.range(0, enc.out_sources[oj].len());
            let old_src = enc.out_sources[oj][slot];
            let old_w = enc.out_weights[oj][slot];
            enc.out_sources[oj][slot] = new_idx;
            enc.out_weights[oj][slot] = rng.rangef(-2.0, 2.0);
            let new_w = enc.out_weights[oj][slot];

            // Quick eval (small sample)
            let score = eval_roundtrip(&enc, &dec, 0, split, 50);
            if score > best_score {
                best_score = score;
                best_neuron = Some(neuron);
                best_out_mod = Some((oj, slot, new_idx, new_w));
            }

            // Restore
            enc.out_sources[oj][slot] = old_src;
            enc.out_weights[oj][slot] = old_w;
            enc.hidden.pop();
        }

        if let Some(neuron) = best_neuron {
            enc.hidden.push(neuron);
            if let Some((oj, slot, new_src, new_w)) = best_out_mod {
                enc.out_sources[oj][slot] = new_src;
                enc.out_weights[oj][slot] = new_w;
            }

            // Retrain decoder on updated encoder
            for ep in 0..50 {
                let lr = 0.003*(1.0-ep as f32/50.0*0.5);
                let mut rt = Rng::new((ni*1000+ep) as u64);
                for _ in 0..300.min(n_chunks) {
                    let ci = rt.range(0, n_chunks);
                    let input = make_input(ci*stride);
                    let tokens = enc.encode(&input);
                    dec.train_step(&tokens, &input, lr);
                }
            }

            let tr = eval_roundtrip(&enc, &dec, 0, split, 300);
            let te = eval_roundtrip(&enc, &dec, split, corpus.len(), 300);
            let delta = tr - best_train;
            best_train = tr;

            println!("  {:>4} {:>7.1}% {:>7.1}% {:>+7.1}% {:>5.0}s",
                ni+1, tr, te, delta, t0.elapsed().as_secs_f64());

            if tr >= 99.5 && te >= 99.5 {
                println!("\n  *** 100% round-trip! ***");
                break;
            }
        } else {
            println!("  {:>4} — no improvement", ni+1);
            break;
        }

        if t0.elapsed().as_secs() > 600 { println!("  Time limit."); break; }
    }

    // Final eval
    let tr = eval_roundtrip(&enc, &dec, 0, split, 500);
    let te = eval_roundtrip(&enc, &dec, split, corpus.len(), 500);
    println!("\n  Final: {} hidden neurons, {} total params", enc.hidden.len(),
        enc.hidden.len()*fan_in + OUT_DIM*fan_in);
    println!("  Train: {:.1}%, Test: {:.1}%", tr, te);
    println!("  Compression: {} → {} values ({:.0}%)", IN_DIM, OUT_DIM, OUT_DIM as f64/IN_DIM as f64*100.0);
    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
