//! L2 Hierarchical MLP — tile small MLPs, then merge upward
//!
//! Level 1: 1024 bytes → 64 windows of 16 → shared MLP → 64 × K features
//! Level 2: merge pairs → 32 × K features
//! Test: masked char prediction (mask center, predict from hierarchy)
//!
//! Run: cargo run --example l2_hierarchical_mlp --release

use std::time::Instant;

const LUT: [[i8; 2]; 27] = [
    [-2,-4],[-4,-2],[0,-6],[-2,-5],[-1,-6],[-3,-5],[1,-8],[-1,-7],
    [-2,-6],[-4,-5],[0,-8],[-2,-7],[-1,-8],[-3,-7],[1,-10],[-1,-9],
    [-4,-6],[-6,-5],[-2,-8],[-4,-7],[-3,-8],[-5,-7],[-1,-10],[-3,-9],
    [-4,-8],[-6,-7],[-2,-10],
];

fn swish(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
fn swish_g(x: f32) -> f32 { let s=1.0/(1.0+(-x).exp()); s+x*s*(1.0-s) }

struct Rng(u64);
impl Rng {
    fn new(s:u64)->Self{Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1))}
    fn next(&mut self)->u64{self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);self.0}
    fn normal(&mut self)->f32{let u1=(((self.next()>>33)%65536)as f32/65536.0).max(1e-7);let u2=((self.next()>>33)%65536)as f32/65536.0;(-2.0*u1.ln()).sqrt()*(2.0*std::f32::consts::PI*u2).cos()}
    fn range(&mut self,lo:usize,hi:usize)->usize{if hi<=lo{lo}else{lo+(self.next()as usize%(hi-lo))}}
}

fn load_corpus(path:&str)->Vec<u8>{
    let raw=std::fs::read(path).expect("read");
    raw.iter().filter_map(|&b|match b{b'a'..=b'z'=>Some(b-b'a'),b'A'..=b'Z'=>Some(b-b'A'),b' '|b'\n'|b'\t'|b'\r'=>Some(26),_=>None}).collect()
}

// Simple 2-layer MLP block (shared weights, tiled across positions)
struct MLPBlock {
    w1: Vec<Vec<f32>>, b1: Vec<f32>,
    w2: Vec<Vec<f32>>, b2: Vec<f32>,
    in_dim: usize, hid: usize, out_dim: usize,
}

impl MLPBlock {
    fn new(in_dim: usize, hid: usize, out_dim: usize, rng: &mut Rng) -> Self {
        let s1=(2.0/in_dim as f32).sqrt(); let s2=(2.0/hid as f32).sqrt();
        MLPBlock {
            w1:(0..hid).map(|_|(0..in_dim).map(|_|rng.normal()*s1).collect()).collect(),
            b1:vec![0.0;hid],
            w2:(0..out_dim).map(|_|(0..hid).map(|_|rng.normal()*s2).collect()).collect(),
            b2:vec![0.0;out_dim],
            in_dim, hid, out_dim,
        }
    }

    fn params(&self)->usize { self.in_dim*self.hid+self.hid+self.hid*self.out_dim+self.out_dim }

    fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut z1 = self.b1.clone();
        for j in 0..self.hid { for k in 0..self.in_dim { z1[j]+=self.w1[j][k]*input[k]; } }
        let a1: Vec<f32> = z1.iter().map(|&v| swish(v)).collect();
        let mut out = self.b2.clone();
        for j in 0..self.out_dim { for k in 0..self.hid { out[j]+=self.w2[j][k]*a1[k]; } }
        (z1, a1, out)
    }

    fn backward(&mut self, input: &[f32], z1: &[f32], a1: &[f32], d_out: &[f32], lr: f32) -> Vec<f32> {
        // Backprop through w2
        let mut da1 = vec![0.0f32; self.hid];
        for j in 0..self.out_dim {
            for k in 0..self.hid {
                da1[k] += d_out[j]*self.w2[j][k];
                self.w2[j][k] -= lr*d_out[j]*a1[k];
            }
            self.b2[j] -= lr*d_out[j];
        }
        // Backprop through swish + w1
        let mut d_in = vec![0.0f32; self.in_dim];
        for j in 0..self.hid {
            let g = da1[j]*swish_g(z1[j]);
            for k in 0..self.in_dim {
                d_in[k] += g*self.w1[j][k];
                self.w1[j][k] -= lr*g*input[k];
            }
            self.b1[j] -= lr*g;
        }
        d_in
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let split = corpus.len()*80/100;

    let encoded: Vec<[f32;2]> = corpus.iter().map(|&ch|
        [LUT[ch as usize][0] as f32/16.0, LUT[ch as usize][1] as f32/16.0]
    ).collect();

    let ctx = 512usize; // total context in bytes
    let win = 16usize;  // window size for L2 tiling
    let n_win = ctx/win; // 32 windows
    let win_input = win*2; // 32 values per window
    let feat = 16usize; // features per window output
    let mask_val = [1.0f32, 1.0];

    println!("=== HIERARCHICAL MLP ===\n");
    println!("  ctx={} bytes, win={}, n_windows={}", ctx, win, n_win);
    println!("  Level 1: {} windows × {} in → {} feat (shared MLP)", n_win, win_input, feat);
    println!("  Level 2: merge pairs → {} × {} feat", n_win/2, feat);
    println!("  Head: predict masked char (27-class)\n");

    let mut rng = Rng::new(42);

    // Level 1: window encoder (shared, tiled)
    let mut l1 = MLPBlock::new(win_input, 64, feat, &mut rng);

    // Level 2: pair merger (shared, tiled)
    let l2_in = feat*2; // two adjacent L1 outputs
    let mut l2 = MLPBlock::new(l2_in, 32, feat, &mut rng);

    // Level 3: another merge
    let mut l3 = MLPBlock::new(l2_in, 32, feat, &mut rng);

    // Head: takes L3 features at mask position's window → 27
    let head_in = feat; // L3 feature at the relevant window
    let mut head = MLPBlock::new(head_in, 64, 27, &mut rng);

    let total_params = l1.params() + l2.params() + l3.params() + head.params();
    println!("  L1: {} params (shared across {} windows)", l1.params(), n_win);
    println!("  L2: {} params (shared across {} merges)", l2.params(), n_win/2);
    println!("  L3: {} params (shared across {} merges)", l3.params(), n_win/4);
    println!("  Head: {} params", head.params());
    println!("  Total: {} params\n", total_params);

    let mask_byte = ctx/2; // mask center byte
    let mask_win = mask_byte/win; // which window contains mask

    println!("  {:>5} {:>7} {:>8} {:>8} {:>6}",
        "epoch", "loss", "train%", "test%", "time");
    println!("  {}", "-".repeat(42));

    let samples = 2000.min(split.saturating_sub(ctx+1));

    for ep in 0..1000 {
        let lr = 0.005*(1.0-ep as f32/1000.0*0.8);
        let mut rt = Rng::new(ep as u64*1000+42);
        let mut tloss=0.0f32; let mut n=0u32;

        for _ in 0..samples {
            let off = rt.range(0, split.saturating_sub(ctx+1));

            // Build window inputs (mask one byte)
            let mut l1_outs: Vec<Vec<f32>> = Vec::new();
            let mut l1_states: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new(); // (input, z1, a1)
            let mut l1_inputs: Vec<Vec<f32>> = Vec::new();

            for w in 0..n_win {
                let mut input = Vec::with_capacity(win_input);
                for i in 0..win {
                    let byte_idx = off + w*win + i;
                    if w*win+i == mask_byte {
                        input.extend_from_slice(&mask_val);
                    } else if byte_idx < encoded.len() {
                        input.push(encoded[byte_idx][0]);
                        input.push(encoded[byte_idx][1]);
                    } else {
                        input.push(0.0); input.push(0.0);
                    }
                }
                let (z1, a1, out) = l1.forward(&input);
                l1_inputs.push(input);
                l1_states.push((z1, a1, out.clone()));
                l1_outs.push(out);
            }

            // Level 2: merge pairs
            let n2 = n_win/2;
            let mut l2_outs: Vec<Vec<f32>> = Vec::new();
            let mut l2_states: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();
            let mut l2_inputs: Vec<Vec<f32>> = Vec::new();

            for i in 0..n2 {
                let mut inp: Vec<f32> = l1_outs[i*2].clone();
                inp.extend_from_slice(&l1_outs[i*2+1]);
                let (z1, a1, out) = l2.forward(&inp);
                l2_inputs.push(inp);
                l2_states.push((z1, a1, out.clone()));
                l2_outs.push(out);
            }

            // Level 3: merge pairs again
            let n3 = n2/2;
            let mut l3_outs: Vec<Vec<f32>> = Vec::new();
            let mut l3_states: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();
            let mut l3_inputs: Vec<Vec<f32>> = Vec::new();

            for i in 0..n3 {
                let mut inp: Vec<f32> = l2_outs[i*2].clone();
                inp.extend_from_slice(&l2_outs[i*2+1]);
                let (z1, a1, out) = l3.forward(&inp);
                l3_inputs.push(inp);
                l3_states.push((z1, a1, out.clone()));
                l3_outs.push(out);
            }

            // Head: use L3 feature at mask's window position
            let mask_l3 = mask_win / 4; // which L3 unit covers the mask
            let head_input = &l3_outs[mask_l3.min(l3_outs.len()-1)];
            let (hz1, ha1, logits) = head.forward(head_input);

            // Softmax + CE
            let target = corpus[off + mask_byte];
            let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut p = vec![0.0f32;27]; let mut s=0.0f32;
            for c in 0..27{p[c]=(logits[c]-mx).exp();s+=p[c];}
            for c in 0..27{p[c]/=s;}
            let loss = -(p[target as usize].max(1e-10).ln());
            if loss.is_nan() { continue; }
            tloss += loss; n += 1;
            p[target as usize] -= 1.0;

            // Backprop head
            let d_head = head.backward(head_input, &hz1, &ha1, &p, lr);

            // Backprop L3 at mask position
            let mi3 = mask_l3.min(n3-1);
            let d_l3 = l3.backward(&l3_inputs[mi3], &l3_states[mi3].0, &l3_states[mi3].1, &d_head, lr);

            // Backprop L2 (the two that fed into this L3)
            let l2a = mi3*2; let l2b = l2a+1;
            if l2a < n2 {
                let d_l2a: Vec<f32> = d_l3[..feat].to_vec();
                let _ = l2.backward(&l2_inputs[l2a], &l2_states[l2a].0, &l2_states[l2a].1, &d_l2a, lr);
            }
            if l2b < n2 {
                let d_l2b: Vec<f32> = d_l3[feat..].to_vec();
                let _ = l2.backward(&l2_inputs[l2b], &l2_states[l2b].0, &l2_states[l2b].1, &d_l2b, lr);
            }
        }

        if ep % 25 == 0 {
            let eval = |start:usize,end:usize| -> f64 {
                let mut rng3=Rng::new(999);
                let mut ok=0usize;let mut tot=0usize;
                for _ in 0..500 {
                    if end<start+ctx+1{break;}
                    let off=rng3.range(start,end.saturating_sub(ctx+1));
                    // Forward only
                    let mut l1o:Vec<Vec<f32>>=Vec::new();
                    for w in 0..n_win {
                        let mut inp=Vec::with_capacity(win_input);
                        for i in 0..win {
                            let bi=off+w*win+i;
                            if w*win+i==mask_byte{inp.extend_from_slice(&mask_val);}
                            else if bi<encoded.len(){inp.push(encoded[bi][0]);inp.push(encoded[bi][1]);}
                            else{inp.push(0.0);inp.push(0.0);}
                        }
                        l1o.push(l1.forward(&inp).2);
                    }
                    let mut l2o:Vec<Vec<f32>>=Vec::new();
                    for i in 0..n_win/2{let mut inp=l1o[i*2].clone();inp.extend(&l1o[i*2+1]);l2o.push(l2.forward(&inp).2);}
                    let mut l3o:Vec<Vec<f32>>=Vec::new();
                    for i in 0..n_win/4{let mut inp=l2o[i*2].clone();inp.extend(&l2o[i*2+1]);l3o.push(l3.forward(&inp).2);}
                    let mi=mask_win/4;
                    let logits=head.forward(&l3o[mi.min(l3o.len()-1)]).2;
                    let pred=logits.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|v|v.0).unwrap_or(0);
                    if pred==corpus[off+mask_byte]as usize{ok+=1;}
                    tot+=1;
                }
                if tot==0{0.0}else{ok as f64/tot as f64*100.0}
            };
            let tr=eval(0,split);
            let te=eval(split,corpus.len());
            println!("  {:>5} {:>7.3} {:>7.1}% {:>7.1}% {:>5.0}s",
                ep,if n>0{tloss/n as f32}else{0.0},tr,te,t0.elapsed().as_secs_f64());
        }

        if t0.elapsed().as_secs() > 300 { break; }
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
