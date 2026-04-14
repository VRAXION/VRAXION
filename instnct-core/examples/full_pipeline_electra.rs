//! Full 4-layer pipeline: C19 encoder → compressor → ELECTRA interpreter → brain
//!
//! L1: Fixed binary C19 byte encoder (byte→7, POPCOUNT) — FROZEN
//! L2: Autoencoder compressor (ctx×7 → bottleneck, 100% rekon) — trained then FROZEN
//! L3: ELECTRA interpreter (compressed → features, "which byte is fake?") — pretrained then FROZEN
//! L4: Brain (features → 27 prediction) — trained last
//!
//! Run: cargo run --example full_pipeline_electra --release

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

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

// ═══ L1: Fixed binary C19 byte encoder ═══
fn encode_byte(ch: u8) -> [f32;7] {
    const W: [[i8;8];7] = [[-1,1,1,-1,-1,1,1,-1],[1,-1,1,1,-1,-1,-1,-1],[-1,-1,1,-1,1,1,-1,-1],
        [-1,-1,-1,1,-1,-1,1,-1],[-1,1,1,1,-1,-1,-1,-1],[1,1,-1,-1,-1,-1,-1,-1],[-1,1,-1,1,1,1,-1,-1]];
    const B: [i8;7] = [1,1,1,1,1,1,1];
    const C: [f32;7] = [10.0;7];
    const RHO: [f32;7] = [2.0,0.0,0.0,0.0,0.0,0.0,0.0];
    let mut bits=[0.0f32;8]; for i in 0..8 { bits[i]=((ch>>i)&1) as f32; }
    let mut o=[0.0f32;7];
    for k in 0..7 { let mut d=B[k] as f32; for j in 0..8 { d+=W[k][j] as f32*bits[j]; } o[k]=c19(d,C[k],RHO[k]); }
    o
}

fn encode_seq(chars: &[u8]) -> Vec<f32> {
    chars.iter().flat_map(|&ch| encode_byte(ch).to_vec()).collect()
}

// ═══ L2: Tied-weight autoencoder compressor ═══
struct Compressor {
    w: Vec<Vec<f32>>, enc_b: Vec<f32>, dec_b: Vec<f32>,
    idim: usize, bneck: usize,
}

impl Compressor {
    fn new(idim: usize, bneck: usize, rng: &mut Rng) -> Self {
        let s = (2.0/idim as f32).sqrt();
        Compressor {
            w: (0..bneck).map(|_| (0..idim).map(|_| rng.normal()*s).collect()).collect(),
            enc_b: vec![0.0;bneck], dec_b: vec![0.0;idim], idim, bneck,
        }
    }
    fn encode(&self, input: &[f32]) -> Vec<f32> {
        let mut h = vec![0.0f32;self.bneck];
        for k in 0..self.bneck { h[k]=self.enc_b[k]; for j in 0..self.idim { h[k]+=self.w[k][j]*input[j]; } h[k]=sigmoid(h[k]); }
        h
    }
    fn train_step(&mut self, input: &[f32], lr: f32) {
        let h = self.encode(input);
        let mut o = vec![0.0f32;self.idim];
        for j in 0..self.idim { o[j]=self.dec_b[j]; for k in 0..self.bneck { o[j]+=self.w[k][j]*h[k]; } }
        let mut d_o = vec![0.0f32;self.idim];
        for j in 0..self.idim { d_o[j]=2.0*(o[j]-input[j])/self.idim as f32; }
        let mut d_h = vec![0.0f32;self.bneck];
        for j in 0..self.idim { for k in 0..self.bneck { d_h[k]+=d_o[j]*self.w[k][j]; self.w[k][j]-=lr*d_o[j]*h[k]; } self.dec_b[j]-=lr*d_o[j]; }
        for k in 0..self.bneck { let dh=d_h[k]*h[k]*(1.0-h[k]); for j in 0..self.idim { self.w[k][j]-=lr*dh*input[j]; } self.enc_b[k]-=lr*dh; }
    }
}

// ═══ L3: Interpreter (processes compressed signal) ═══
struct Interpreter {
    w1: Vec<Vec<f32>>, b1: Vec<f32>,
    w2: Vec<Vec<f32>>, b2: Vec<f32>,
    idim: usize, hdim: usize, odim: usize,
}

impl Interpreter {
    fn new(idim: usize, hdim: usize, odim: usize, rng: &mut Rng) -> Self {
        let s1=(2.0/idim as f32).sqrt(); let s2=(2.0/hdim as f32).sqrt();
        Interpreter {
            w1: (0..hdim).map(|_| (0..idim).map(|_| rng.normal()*s1).collect()).collect(), b1: vec![0.0;hdim],
            w2: (0..odim).map(|_| (0..hdim).map(|_| rng.normal()*s2).collect()).collect(), b2: vec![0.0;odim],
            idim, hdim, odim,
        }
    }
    fn params(&self) -> usize { self.idim*self.hdim+self.hdim + self.hdim*self.odim+self.odim }
    fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut h = vec![0.0f32;self.hdim];
        for k in 0..self.hdim { h[k]=self.b1[k]; for j in 0..self.idim { h[k]+=self.w1[k][j]*input[j]; } h[k]=h[k].max(0.0); }
        let mut o = vec![0.0f32;self.odim];
        for f in 0..self.odim { o[f]=self.b2[f]; for k in 0..self.hdim { o[f]+=self.w2[f][k]*h[k]; } }
        (h, o)
    }
}

// ═══ L4: Brain ═══
struct Brain {
    w1: Vec<Vec<f32>>, b1: Vec<f32>,
    w2: Vec<Vec<f32>>, b2: Vec<f32>,
    idim: usize, hdim: usize,
}

impl Brain {
    fn new(idim: usize, hdim: usize, rng: &mut Rng) -> Self {
        let s1=(2.0/idim as f32).sqrt(); let s2=(2.0/hdim as f32).sqrt();
        Brain {
            w1: (0..hdim).map(|_| (0..idim).map(|_| rng.normal()*s1).collect()).collect(), b1: vec![0.0;hdim],
            w2: (0..27).map(|_| (0..hdim).map(|_| rng.normal()*s2).collect()).collect(), b2: vec![0.0;27],
            idim, hdim,
        }
    }
    fn params(&self) -> usize { self.idim*self.hdim+self.hdim + self.hdim*27+27 }
    fn train_step(&mut self, input: &[f32], target: u8, lr: f32) {
        let mut h=vec![0.0f32;self.hdim];
        for k in 0..self.hdim { h[k]=self.b1[k]; for j in 0..self.idim { h[k]+=self.w1[k][j]*input[j]; } h[k]=h[k].max(0.0); }
        let mut logits=vec![0.0f32;27];
        for c in 0..27 { logits[c]=self.b2[c]; for k in 0..self.hdim { logits[c]+=self.w2[c][k]*h[k]; } }
        let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
        let mut p=vec![0.0f32;27]; let mut s=0.0f32;
        for c in 0..27 { p[c]=(logits[c]-mx).exp(); s+=p[c]; } for c in 0..27 { p[c]/=s; }
        let mut dl=p; dl[target as usize]-=1.0;
        let mut dh=vec![0.0f32;self.hdim];
        for c in 0..27 { for k in 0..self.hdim { dh[k]+=dl[c]*self.w2[c][k]; self.w2[c][k]-=lr*dl[c]*h[k]; } self.b2[c]-=lr*dl[c]; }
        for k in 0..self.hdim { if h[k]<=0.0{continue;} for j in 0..self.idim { self.w1[k][j]-=lr*dh[k]*input[j]; } self.b1[k]-=lr*dh[k]; }
    }
    fn predict(&self, input: &[f32]) -> usize {
        let mut h=vec![0.0f32;self.hdim];
        for k in 0..self.hdim { h[k]=self.b1[k]; for j in 0..self.idim { h[k]+=self.w1[k][j]*input[j]; } h[k]=h[k].max(0.0); }
        let mut logits=vec![0.0f32;27];
        for c in 0..27 { logits[c]=self.b2[c]; for k in 0..self.hdim { logits[c]+=self.w2[c][k]*h[k]; } }
        logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let ctx = 16usize;
    let idim = ctx * 7; // 112
    let bneck = 96; // 86% ratio → proven 99.8% reconstruction
    let interp_hdim = 128;
    let interp_odim = 64; // interpreter output features
    let brain_hdim = 256;

    println!("=== FULL 4-LAYER PIPELINE WITH ELECTRA ===\n");
    println!("  L1: C19 byte encoder (7 signal/byte, binary, FROZEN)");
    println!("  L2: Compressor ({}→{}, autoencoder, FROZEN after training)", idim, bneck);
    println!("  L3: Interpreter ({}→{}, ELECTRA pretrained, FROZEN)", bneck, interp_odim);
    println!("  L4: Brain ({}→27, prediction)\n", interp_odim);

    let samples = 10000.min(corpus.len() / (ctx+1));
    let mut rng = Rng::new(42);

    // ═══════════════════════════════════════════
    // STEP 1: Train L2 compressor (autoencoder)
    // ═══════════════════════════════════════════
    println!("━━━ STEP 1: Train compressor (autoencoder, {}→{}) ━━━", idim, bneck);
    let mut comp = Compressor::new(idim, bneck, &mut rng);
    for ep in 0..100 {
        let lr = 0.01 * (1.0 - ep as f32 / 100.0 * 0.7);
        let mut rt = Rng::new(ep as u64 * 1000 + 42);
        for _ in 0..samples {
            let off = rt.range(0, corpus.len()-ctx);
            comp.train_step(&encode_seq(&corpus[off..off+ctx]), lr);
        }
    }
    // Eval compressor
    let mut rt = Rng::new(999); let mut c_ok=0; let mut c_tot=0;
    for _ in 0..2000 { let off=rt.range(0,corpus.len()-ctx);
        let sig=encode_seq(&corpus[off..off+ctx]); let h=comp.encode(&sig);
        let mut o=vec![0.0f32;idim]; for j in 0..idim { o[j]=comp.dec_b[j]; for k in 0..bneck { o[j]+=comp.w[k][j]*h[k]; } }
        let pp_ref = crate::encode_byte; // reuse
        for i in 0..ctx { let rs=&o[i*7..(i+1)*7];
            let mut best=0u8; let mut bd=f32::MAX;
            for ch in 0..27u8 { let code=encode_byte(ch); let d:f32=code.iter().zip(rs).map(|(a,b)|(a-b)*(a-b)).sum(); if d<bd{bd=d;best=ch;} }
            if best==corpus[off+i]{c_ok+=1;} c_tot+=1; }
    }
    println!("  Compressor reconstruction: {:.1}% ({}/{})\n",
        c_ok as f64/c_tot as f64*100.0, c_ok, c_tot);

    // ═══════════════════════════════════════════
    // STEP 2: ELECTRA pretrain L3 interpreter
    // ═══════════════════════════════════════════
    println!("━━━ STEP 2: ELECTRA pretrain interpreter (corrupt detection) ━━━");
    let mut interp = Interpreter::new(bneck, interp_hdim, interp_odim, &mut rng);

    // ELECTRA: detect which positions have been corrupted
    // We work on the COMPRESSED signal — corrupt at byte level, encode, compress
    let corrupt_rate = 0.30; // 30% corruption for stronger signal
    let electra_lr_head = 0.05; // higher LR for simple head

    // Per-position detection head on interpreter features
    // We split interp_odim into ctx chunks: each position gets interp_odim/ctx features
    // Actually simpler: interpreter processes full compressed vector, outputs ctx scores
    let mut det_w: Vec<Vec<f32>> = (0..ctx).map(|_| (0..interp_odim).map(|_| rng.normal()*0.1).collect()).collect();
    let mut det_b = vec![0.0f32; ctx];

    for ep in 0..200 {
        let lr = 0.005 * (1.0 - ep as f32 / 200.0 * 0.7);
        let mut rt = Rng::new(ep as u64 * 1000 + 42);
        let mut n_correct = 0usize; let mut n_total = 0usize;

        for _ in 0..samples {
            let off = rt.range(0, corpus.len()-ctx);
            let mut chars: Vec<u8> = corpus[off..off+ctx].to_vec();
            let mut labels = vec![0.0f32; ctx];

            // Corrupt
            for i in 0..ctx {
                if rt.f32() < corrupt_rate {
                    let orig = chars[i];
                    let new_ch = rt.range(0, 27) as u8;
                    if new_ch != orig { chars[i] = new_ch; labels[i] = 1.0; }
                }
            }

            // L1+L2: encode and compress
            let encoded = encode_seq(&chars);
            let compressed = comp.encode(&encoded);

            // L3: interpreter forward
            let (interp_h, interp_out) = interp.forward(&compressed);

            // Detection head: per-position sigmoid
            let mut preds = vec![0.0f32; ctx];
            for pos in 0..ctx {
                let mut z = det_b[pos];
                for f in 0..interp_odim { z += det_w[pos][f] * interp_out[f]; }
                preds[pos] = sigmoid(z);
            }

            // BCE loss + backprop
            let mut d_interp_out = vec![0.0f32; interp_odim];
            for pos in 0..ctx {
                let err = preds[pos] - labels[pos];

                // Update detection head
                for f in 0..interp_odim {
                    d_interp_out[f] += err * det_w[pos][f];
                    det_w[pos][f] -= electra_lr_head * err * interp_out[f];
                }
                det_b[pos] -= electra_lr_head * err;

                if (preds[pos] > 0.5) == (labels[pos] > 0.5) { n_correct += 1; }
                n_total += 1;
            }

            // Backprop through interpreter
            let mut d_h = vec![0.0f32; interp_hdim];
            for f in 0..interp_odim {
                for k in 0..interp_hdim { d_h[k] += d_interp_out[f] * interp.w2[f][k]; interp.w2[f][k] -= lr * d_interp_out[f] * interp_h[k]; }
                interp.b2[f] -= lr * d_interp_out[f];
            }
            for k in 0..interp_hdim {
                if interp_h[k] <= 0.0 { continue; }
                for j in 0..bneck { interp.w1[k][j] -= lr * d_h[k] * compressed[j]; }
                interp.b1[k] -= lr * d_h[k];
            }
        }

        if (ep+1) % 40 == 0 {
            println!("  ep{:>3}: detect_acc={:.1}%", ep+1, n_correct as f64/n_total as f64*100.0);
        }
    }
    println!("  Interpreter params: {}\n", interp.params());

    // ═══════════════════════════════════════════
    // STEP 3: Train L4 brain on frozen L1-L3
    // ═══════════════════════════════════════════
    println!("━━━ STEP 3: Train brain on frozen pipeline ━━━\n");

    let eval_fn = |brain: &Brain, interp: &Interpreter, comp: &Compressor| -> f64 {
        let mut eval_rng = Rng::new(999); let mut ok=0; let mut tot=0;
        for _ in 0..5000 { if corpus.len()<ctx+1{break;} let off=eval_rng.range(0,corpus.len()-ctx-1);
            let encoded=encode_seq(&corpus[off..off+ctx]);
            let compressed=comp.encode(&encoded);
            let (_,features)=interp.forward(&compressed);
            let pred=brain.predict(&features);
            if pred==corpus[off+ctx] as usize {ok+=1;} tot+=1;
        }
        ok as f64/tot as f64*100.0
    };

    // A) ELECTRA pretrained interpreter + brain
    {
        let tc = Instant::now();
        let mut brain = Brain::new(interp_odim, brain_hdim, &mut rng);
        for ep in 0..200 {
            let lr = 0.01 * (1.0 - ep as f32 / 200.0 * 0.8);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..samples {
                let off = rt.range(0, corpus.len()-ctx-1);
                let encoded=encode_seq(&corpus[off..off+ctx]);
                let compressed=comp.encode(&encoded);
                let (_,features)=interp.forward(&compressed);
                brain.train_step(&features, corpus[off+ctx], lr);
            }
        }
        let acc = eval_fn(&brain, &interp, &comp);
        let total_p = interp.params() + brain.params();
        let m = if acc>50.0{" ★★"} else if acc>30.0{" ★"} else {""};
        println!("  A) ELECTRA interp (frozen) → brain: {:.1}% ({} params) {:.1}s{}",
            acc, total_p, tc.elapsed().as_secs_f64(), m);
    }

    // B) Random interpreter (frozen) + brain (control)
    {
        let tc = Instant::now();
        let rand_interp = Interpreter::new(bneck, interp_hdim, interp_odim, &mut Rng::new(999));
        let mut brain = Brain::new(interp_odim, brain_hdim, &mut rng);
        for ep in 0..200 {
            let lr = 0.01 * (1.0 - ep as f32 / 200.0 * 0.8);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..samples {
                let off = rt.range(0, corpus.len()-ctx-1);
                let encoded=encode_seq(&corpus[off..off+ctx]);
                let compressed=comp.encode(&encoded);
                let (_,features)=rand_interp.forward(&compressed);
                brain.train_step(&features, corpus[off+ctx], lr);
            }
        }
        let acc = eval_fn(&brain, &rand_interp, &comp);
        println!("  B) Random interp (frozen) → brain:  {:.1}% (control) {:.1}s", acc, tc.elapsed().as_secs_f64());
    }

    // C) No interpreter, brain on compressed directly
    {
        let tc = Instant::now();
        let mut brain = Brain::new(bneck, brain_hdim, &mut rng);
        for ep in 0..200 {
            let lr = 0.01 * (1.0 - ep as f32 / 200.0 * 0.8);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..samples {
                let off = rt.range(0, corpus.len()-ctx-1);
                let encoded=encode_seq(&corpus[off..off+ctx]);
                let compressed=comp.encode(&encoded);
                brain.train_step(&compressed, corpus[off+ctx], lr);
            }
        }
        let mut eval_rng=Rng::new(999); let mut ok=0; let mut tot=0;
        for _ in 0..5000 { if corpus.len()<ctx+1{break;} let off=eval_rng.range(0,corpus.len()-ctx-1);
            let encoded=encode_seq(&corpus[off..off+ctx]);
            let compressed=comp.encode(&encoded);
            let pred=brain.predict(&compressed);
            if pred==corpus[off+ctx] as usize {ok+=1;} tot+=1;
        }
        let acc = ok as f64/tot as f64*100.0;
        println!("  C) No interp, brain on compressed:  {:.1}% ({} params) {:.1}s", acc, brain.params(), tc.elapsed().as_secs_f64());
    }

    // D) No compressor, no interp, brain on raw encoded
    {
        let tc = Instant::now();
        let mut brain = Brain::new(idim, brain_hdim, &mut rng);
        for ep in 0..200 {
            let lr = 0.01 * (1.0 - ep as f32 / 200.0 * 0.8);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..samples {
                let off = rt.range(0, corpus.len()-ctx-1);
                let encoded=encode_seq(&corpus[off..off+ctx]);
                brain.train_step(&encoded, corpus[off+ctx], lr);
            }
        }
        let mut eval_rng=Rng::new(999); let mut ok=0; let mut tot=0;
        for _ in 0..5000 { if corpus.len()<ctx+1{break;} let off=eval_rng.range(0,corpus.len()-ctx-1);
            let encoded=encode_seq(&corpus[off..off+ctx]);
            let pred=brain.predict(&encoded);
            if pred==corpus[off+ctx] as usize {ok+=1;} tot+=1;
        }
        let acc = ok as f64/tot as f64*100.0;
        println!("  D) No comp, no interp, raw→brain:   {:.1}% ({} params) {:.1}s", acc, brain.params(), tc.elapsed().as_secs_f64());
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
