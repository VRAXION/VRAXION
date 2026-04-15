//! L2 Conv Autoencoder — compress 2048 int8 → N → 2048 int8
//!
//! Encoder: strided conv (progressive compression)
//! Decoder: dense (N → 2048)
//! Sweep: bottleneck N, activation functions
//! Telemetry: loss, accuracy, grad norms per epoch
//!
//! Run: cargo run --example l2_conv_autoenc --release

use std::time::Instant;

const LUT: [[i8; 2]; 27] = [
    [-2,-4],[-4,-2],[0,-6],[-2,-5],[-1,-6],[-3,-5],[1,-8],[-1,-7],
    [-2,-6],[-4,-5],[0,-8],[-2,-7],[-1,-8],[-3,-7],[1,-10],[-1,-9],
    [-4,-6],[-6,-5],[-2,-8],[-4,-7],[-3,-8],[-5,-7],[-1,-10],[-3,-9],
    [-4,-8],[-6,-7],[-2,-10],
];

fn swish(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
fn swish_g(x: f32) -> f32 { let s=1.0/(1.0+(-x).exp()); s+x*s*(1.0-s) }
fn c19a(x: f32, c: f32, rho: f32) -> f32 {
    let c=c.max(0.1);let rho=rho.max(0.0);let l=6.0*c;
    if x>=l{return x-l;}if x<=-l{return x+l;}
    let s=x/c;let n=s.floor();let t=s-n;let h=t*(1.0-t);
    let sg=if(n as i32)%2==0{1.0}else{-1.0};c*(sg*h+rho*h*h)
}
fn c19g(x: f32, c: f32, rho: f32) -> f32 {
    let c=c.max(0.1);let rho=rho.max(0.0);let l=6.0*c;
    if x>=l||x<=-l{return 1.0;}
    let s=x/c;let n=s.floor();let t=s-n;let h=t*(1.0-t);
    let sg=if(n as i32)%2==0{1.0}else{-1.0};(sg+2.0*rho*h)*(1.0-2.0*t)
}

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

fn nearest_lut(v0:f32,v1:f32)->u8{
    let mut b=0u8;let mut bd=f32::MAX;
    for s in 0..27u8{let d0=v0-LUT[s as usize][0]as f32/16.0;let d1=v1-LUT[s as usize][1]as f32/16.0;
        let d=d0*d0+d1*d1;if d<bd{bd=d;b=s;}}b
}

const CHUNK:usize = 512; // smaller chunk for more training data
const DIM:usize = CHUNK*2; // 1024

struct ConvAutoEnc {
    // Encoder: 3 strided conv layers
    enc_w: Vec<Vec<Vec<f32>>>, // [layer][filter][k*ch_in]
    enc_b: Vec<Vec<f32>>,
    enc_ch: Vec<usize>, // channels per layer
    // Flatten → bottleneck
    flat_w: Vec<Vec<f32>>, flat_b: Vec<f32>,
    // Decoder: bottleneck → full output
    dec_w: Vec<Vec<f32>>, dec_b: Vec<f32>,
    bn: usize, flat_dim: usize,
    use_c19: bool,
    c_params: Vec<Vec<f32>>, rho_params: Vec<Vec<f32>>,
}

impl ConvAutoEnc {
    fn new(bn: usize, use_c19: bool, rng: &mut Rng) -> Self {
        let k = 5usize;
        let stride = 2usize;
        let chs = vec![2, 16, 32, 64]; // input, l1, l2, l3
        let mut enc_w = Vec::new();
        let mut enc_b = Vec::new();
        let mut c_params = Vec::new();
        let mut rho_params = Vec::new();

        for i in 0..3 {
            let fan = k * chs[i];
            let sc = (2.0/fan as f32).sqrt();
            enc_w.push((0..chs[i+1]).map(|_|(0..fan).map(|_|rng.normal()*sc).collect()).collect());
            enc_b.push(vec![0.0;chs[i+1]]);
            c_params.push(vec![5.0f32; chs[i+1]]);
            rho_params.push(vec![0.5f32; chs[i+1]]);
        }

        // After 3 stride-2 convs: 512→254→125→61 positions (approx)
        // With k=5,s=2: out = (in - k)/s + 1
        let mut pos = CHUNK; // 512
        for _ in 0..3 { pos = (pos - k)/stride + 1; }
        let flat_dim = pos * chs[3]; // 61 * 64 = 3904

        let s1 = (2.0/flat_dim as f32).sqrt();
        let s2 = (2.0/bn as f32).sqrt();

        ConvAutoEnc {
            enc_w, enc_b, enc_ch: chs,
            flat_w: (0..bn).map(|_|(0..flat_dim).map(|_|rng.normal()*s1).collect()).collect(),
            flat_b: vec![0.0;bn],
            dec_w: (0..DIM).map(|_|(0..bn).map(|_|rng.normal()*s2).collect()).collect(),
            dec_b: vec![0.0;DIM],
            bn, flat_dim, use_c19, c_params, rho_params,
        }
    }

    fn params(&self) -> usize {
        let enc: usize = self.enc_w.iter().zip(&self.enc_b).map(|(w,b)|w.len()*w[0].len()+b.len()).sum();
        enc + self.bn*self.flat_dim + self.bn + DIM*self.bn + DIM
    }

    fn act(&self, x: f32, li: usize, ni: usize) -> f32 {
        if self.use_c19 { c19a(x, self.c_params[li][ni], self.rho_params[li][ni]) }
        else { swish(x) }
    }

    fn act_grad(&self, x: f32, li: usize, ni: usize) -> f32 {
        if self.use_c19 { c19g(x, self.c_params[li][ni], self.rho_params[li][ni]) }
        else { swish_g(x) }
    }

    fn forward(&self, input: &[f32]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
        let k=5; let stride=2;
        let mut conv_pres = Vec::new();
        let mut conv_posts = Vec::new();
        let mut x = input.to_vec();
        let mut cur_len = CHUNK;
        let mut cur_ch = self.enc_ch[0];

        for li in 0..3 {
            let nf = self.enc_w[li].len();
            let new_len = (cur_len - k)/stride + 1;
            let mut pre = vec![0.0f32; new_len*nf];
            for p in 0..new_len {
                let sp = p*stride;
                for f in 0..nf {
                    let mut v = self.enc_b[li][f];
                    for ki in 0..k { for d in 0..cur_ch {
                        if sp+ki < cur_len { v += self.enc_w[li][f][ki*cur_ch+d]*x[(sp+ki)*cur_ch+d]; }
                    }}
                    pre[p*nf+f] = v;
                }
            }
            let post: Vec<f32> = pre.iter().enumerate().map(|(i,&v)| self.act(v, li, i % nf)).collect();
            conv_pres.push(pre); conv_posts.push(post.clone());
            x = post; cur_len = new_len; cur_ch = nf;
        }

        // Flatten → bottleneck (swish)
        let mut bn_val = self.flat_b.clone();
        for j in 0..self.bn { for i in 0..x.len().min(self.flat_dim) { bn_val[j] += self.flat_w[j][i]*x[i]; }
            bn_val[j] = swish(bn_val[j]); }

        // Decode (linear)
        let mut out = self.dec_b.clone();
        for j in 0..DIM { for i in 0..self.bn { out[j] += self.dec_w[j][i]*bn_val[i]; } }

        (conv_pres, conv_posts, bn_val, out)
    }

    fn train_step(&mut self, input: &[f32], lr: f32) -> f32 {
        let (conv_pres, conv_posts, bn_val, out) = self.forward(input);

        // MSE loss + gradient
        let mut loss = 0.0f32;
        let mut d_out = vec![0.0f32; DIM];
        for j in 0..DIM {
            let diff = out[j] - input[j];
            loss += diff*diff;
            d_out[j] = 2.0*diff / DIM as f32;
        }
        loss /= DIM as f32;

        // Backprop decoder
        let mut d_bn = vec![0.0f32; self.bn];
        for j in 0..DIM {
            for i in 0..self.bn {
                d_bn[i] += d_out[j]*self.dec_w[j][i];
                self.dec_w[j][i] -= lr*d_out[j]*bn_val[i];
            }
            self.dec_b[j] -= lr*d_out[j];
        }

        // Backprop bottleneck (swish)
        let flat = &conv_posts.last().unwrap();
        let mut d_flat = vec![0.0f32; self.flat_dim.min(flat.len())];
        for j in 0..self.bn {
            let pre_j = self.flat_b[j] + (0..self.flat_dim.min(flat.len())).map(|i| self.flat_w[j][i]*flat[i]).sum::<f32>();
            let g = d_bn[j] * swish_g(pre_j);
            for i in 0..self.flat_dim.min(flat.len()) {
                d_flat[i] += g*self.flat_w[j][i];
                self.flat_w[j][i] -= lr*g*flat[i];
            }
            self.flat_b[j] -= lr*g;
        }

        // Backprop conv layers (simplified — update weights, skip full chain for speed)
        let k=5; let stride=2;
        let mut dx = d_flat;
        for li in (0..3).rev() {
            let nf = self.enc_w[li].len();
            let ch_in = self.enc_ch[li];
            let x_in = if li==0 { input } else { &conv_posts[li-1] };
            let in_len = if li==0 { CHUNK } else { conv_pres[li-1].len()/self.enc_ch[li] };
            let out_len = conv_pres[li].len()/nf;

            // Apply activation grad
            for i in 0..dx.len().min(conv_pres[li].len()) {
                dx[i] *= self.act_grad(conv_pres[li][i], li, i % nf);
            }

            let mut dx_prev = vec![0.0f32; in_len*ch_in];
            for p in 0..out_len {
                let sp = p*stride;
                for f in 0..nf {
                    let idx = p*nf+f;
                    if idx >= dx.len() { continue; }
                    let dc = dx[idx];
                    for ki in 0..k { for d in 0..ch_in {
                        let ii = (sp+ki)*ch_in+d;
                        if ii < x_in.len() && ii < dx_prev.len() {
                            self.enc_w[li][f][ki*ch_in+d] -= lr*dc*x_in[ii];
                            dx_prev[ii] += dc*self.enc_w[li][f][ki*ch_in+d];
                        }
                    }}
                    self.enc_b[li][f] -= lr*dc;
                }
            }
            dx = dx_prev;
        }

        loss
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");

    // Encode corpus
    let encoded: Vec<f32> = corpus.iter().flat_map(|&ch|
        [LUT[ch as usize][0] as f32/16.0, LUT[ch as usize][1] as f32/16.0]
    ).collect();

    // Make chunks with stride
    let stride = 64;
    let n_chunks = (corpus.len()-CHUNK)/stride;
    let split = n_chunks*80/100;

    println!("=== L2 CONV AUTOENCODER ===\n");
    println!("  chunk={} bytes, {} chunks ({} train), stride={}", CHUNK, n_chunks, split, stride);

    struct Cfg { bn: usize, use_c19: bool, name: &'static str }
    let configs = vec![
        Cfg { bn: 256, use_c19: false, name: "swish bn=256" },
        Cfg { bn: 128, use_c19: false, name: "swish bn=128" },
        Cfg { bn: 256, use_c19: true,  name: "c19   bn=256" },
        Cfg { bn: 128, use_c19: true,  name: "c19   bn=128" },
    ];

    for cfg in &configs {
        let tc = Instant::now();
        let mut rng = Rng::new(42);
        let mut model = ConvAutoEnc::new(cfg.bn, cfg.use_c19, &mut rng);

        println!("\n  --- {} ({} params, flat_dim={}) ---\n", cfg.name, model.params(), model.flat_dim);
        println!("  {:>5} {:>8} {:>8} {:>8} {:>6}", "epoch", "loss", "train%", "test%", "time");

        let spe = 500.min(split);

        for ep in 0..500 {
            let lr = 0.003 * (1.0 - ep as f32/500.0*0.7);
            let mut rt = Rng::new(ep as u64*1000+42);
            let mut tloss=0.0f32; let mut n=0u32;

            for _ in 0..spe {
                let ci = rt.range(0, split);
                let off = ci*stride;
                let chunk: Vec<f32> = (0..DIM).map(|i| {
                    if off*2+i < encoded.len() { encoded[off*2+i] } else { 0.0 }
                }).collect();
                let loss = model.train_step(&chunk, lr);
                if !loss.is_nan() { tloss+=loss; n+=1; }
            }

            if ep % 25 == 0 {
                let eval = |start:usize,end:usize| -> f64 {
                    let mut ok=0usize;let mut tot=0usize;
                    let mut rng3=Rng::new(999);
                    for _ in 0..100.min(end-start) {
                        let ci=rng3.range(start,end);
                        let off=ci*stride;
                        let chunk:Vec<f32>=(0..DIM).map(|i|{
                            if off*2+i<encoded.len(){encoded[off*2+i]}else{0.0}
                        }).collect();
                        let(_,_,_,out)=model.forward(&chunk);
                        for p in 0..CHUNK {
                            let orig=corpus[off+p];
                            let pred=nearest_lut(out[p*2],out[p*2+1]);
                            if pred==orig{ok+=1;}
                            tot+=1;
                        }
                    }
                    if tot==0{0.0}else{ok as f64/tot as f64*100.0}
                };
                let tr=eval(0,split);
                let te=eval(split,n_chunks);
                println!("  {:>5} {:>8.4} {:>7.1}% {:>7.1}% {:>5.0}s",
                    ep,if n>0{tloss/n as f32}else{0.0},tr,te,tc.elapsed().as_secs_f64());
            }

            if tc.elapsed().as_secs() > 150 { break; }
        }
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
