//! ELECTRA-style interpreter pretraining for the VRAXION pipeline
//!
//! Architecture (4 layers):
//!   L1: Fixed binary byte encoder (C19, POPCOUNT) — FROZEN
//!   L2: Frozen compressor (autoencoder, 86% ratio) — trained then FROZEN
//!   L3: Interpreter — pretrained with "corrupt detection" — then FROZEN
//!   L4: Brain — trained with prediction loss on frozen L1-L3
//!
//! ELECTRA training for L3:
//!   - Take real text, randomly corrupt some positions
//!   - L3 must detect WHICH positions are corrupted (per-position binary)
//!   - This forces L3 to learn what valid English looks like
//!
//! Then compare:
//!   A) Frozen ELECTRA interpreter + brain (prediction)
//!   B) End-to-end interpreter + brain (no pretraining)
//!   C) No interpreter, just brain directly
//!
//! Run: cargo run --example electra_interpreter --release

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

// L1: Fixed byte encoder
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

// L3: Interpreter — processes full context, outputs features per position
// Architecture: ctx×7 input → hidden → ctx×feat_dim output
struct Interpreter {
    // Shared MLP applied to each position with local context window
    w1: Vec<Vec<f32>>, b1: Vec<f32>,  // (window×7) → hidden
    w2: Vec<Vec<f32>>, b2: Vec<f32>,  // hidden → feat_dim
    window: usize, hidden: usize, feat_dim: usize,
}

impl Interpreter {
    fn new(window: usize, hidden: usize, feat_dim: usize, rng: &mut Rng) -> Self {
        let in_dim = window * 7;
        let s1 = (2.0/in_dim as f32).sqrt();
        let s2 = (2.0/hidden as f32).sqrt();
        Interpreter {
            w1: (0..hidden).map(|_| (0..in_dim).map(|_| rng.normal()*s1).collect()).collect(),
            b1: vec![0.0;hidden],
            w2: (0..feat_dim).map(|_| (0..hidden).map(|_| rng.normal()*s2).collect()).collect(),
            b2: vec![0.0;feat_dim],
            window, hidden, feat_dim,
        }
    }

    fn params(&self) -> usize {
        self.window*7*self.hidden + self.hidden + self.hidden*self.feat_dim + self.feat_dim
    }

    // Process one position (with padding)
    fn forward_pos(&self, encoded: &[f32], pos: usize, ctx: usize) -> Vec<f32> {
        let half = self.window / 2;
        let in_dim = self.window * 7;
        let mut input = vec![0.0f32; in_dim];

        for wi in 0..self.window {
            let src_pos = pos as i32 + wi as i32 - half as i32;
            if src_pos >= 0 && (src_pos as usize) < ctx {
                let src = src_pos as usize;
                for d in 0..7 { input[wi*7+d] = encoded[src*7+d]; }
            }
        }

        // Hidden (ReLU)
        let mut h = vec![0.0f32;self.hidden];
        for k in 0..self.hidden { h[k]=self.b1[k]; for j in 0..in_dim { h[k]+=self.w1[k][j]*input[j]; } h[k]=h[k].max(0.0); }
        // Output
        let mut out = vec![0.0f32;self.feat_dim];
        for f in 0..self.feat_dim { out[f]=self.b2[f]; for k in 0..self.hidden { out[f]+=self.w2[f][k]*h[k]; } }
        out
    }

    // Forward all positions → ctx × feat_dim
    fn forward_all(&self, encoded: &[f32], ctx: usize) -> Vec<Vec<f32>> {
        (0..ctx).map(|pos| self.forward_pos(encoded, pos, ctx)).collect()
    }
}

// ELECTRA head: features → is_corrupt (per position)
struct ElectraHead {
    w: Vec<f32>, b: f32, feat_dim: usize,
}

impl ElectraHead {
    fn new(feat_dim: usize, rng: &mut Rng) -> Self {
        let s = (2.0/feat_dim as f32).sqrt();
        ElectraHead { w: (0..feat_dim).map(|_| rng.normal()*s).collect(), b: 0.0, feat_dim }
    }

    fn predict(&self, features: &[f32]) -> f32 {
        let mut z = self.b;
        for j in 0..self.feat_dim { z += self.w[j] * features[j]; }
        sigmoid(z)
    }
}

// Prediction brain: takes interpreter features, predicts next char
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

    fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut h=vec![0.0f32;self.hdim];
        for k in 0..self.hdim { h[k]=self.b1[k]; for j in 0..self.idim { h[k]+=self.w1[k][j]*input[j]; } h[k]=h[k].max(0.0); }
        let mut logits=vec![0.0f32;27];
        for c in 0..27 { logits[c]=self.b2[c]; for k in 0..self.hdim { logits[c]+=self.w2[c][k]*h[k]; } }
        (h, logits)
    }

    fn train_step(&mut self, input: &[f32], target: u8, lr: f32) {
        let (h, logits) = self.forward(input);
        let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
        let mut p=vec![0.0f32;27]; let mut s=0.0f32;
        for c in 0..27 { p[c]=(logits[c]-mx).exp(); s+=p[c]; } for c in 0..27 { p[c]/=s; }
        let mut dl=p; dl[target as usize]-=1.0;
        let mut dh=vec![0.0f32;self.hdim];
        for c in 0..27 { for k in 0..self.hdim { dh[k]+=dl[c]*self.w2[c][k]; self.w2[c][k]-=lr*dl[c]*h[k]; } self.b2[c]-=lr*dl[c]; }
        for k in 0..self.hdim { if h[k]<=0.0{continue;} for j in 0..self.idim { self.w1[k][j]-=lr*dh[k]*input[j]; } self.b1[k]-=lr*dh[k]; }
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let ctx = 16usize;
    let corrupt_rate = 0.15; // 15% of positions get corrupted (like BERT/ELECTRA)

    println!("=== ELECTRA-STYLE INTERPRETER PRETRAINING ===\n");
    println!("  L1: Fixed byte encoder (binary, C19)");
    println!("  L3: Interpreter pretrained with corrupt detection");
    println!("  L4: Brain trained with prediction loss\n");
    println!("  ctx={}, corrupt_rate={:.0}%, corpus={} chars\n", ctx, corrupt_rate*100.0, corpus.len());

    // ═══════════════════════════════════════════
    // PHASE 1: ELECTRA pretraining of interpreter
    // ═══════════════════════════════════════════
    println!("━━━ PHASE 1: ELECTRA pretraining (corrupt detection) ━━━\n");

    let interp_window = 5;
    let interp_hidden = 64;
    let feat_dim = 16;

    let mut rng = Rng::new(42);
    let mut interp = Interpreter::new(interp_window, interp_hidden, feat_dim, &mut rng);
    let mut electra = ElectraHead::new(feat_dim, &mut rng);

    let electra_epochs = 100;
    let samples = 10000.min(corpus.len() / (ctx+1));

    for ep in 0..electra_epochs {
        let lr = 0.01 * (1.0 - ep as f32 / electra_epochs as f32 * 0.7);
        let mut rt = Rng::new(ep as u64 * 1000 + 42);
        let mut total_loss = 0.0f32;
        let mut n_correct = 0usize;
        let mut n_total = 0usize;

        for _ in 0..samples {
            let off = rt.range(0, corpus.len()-ctx-1);
            let mut chars: Vec<u8> = corpus[off..off+ctx].to_vec();
            let mut labels = vec![0.0f32; ctx]; // 0=real, 1=corrupt

            // Corrupt random positions
            for i in 0..ctx {
                if rt.f32() < corrupt_rate {
                    let orig = chars[i];
                    chars[i] = rt.range(0, 27) as u8;
                    if chars[i] != orig { labels[i] = 1.0; }
                }
            }

            let encoded = encode_seq(&chars);
            let features = interp.forward_all(&encoded, ctx);

            // Train ELECTRA head + backprop through interpreter
            for pos in 0..ctx {
                let pred = electra.predict(&features[pos]);
                let target = labels[pos];
                let err = pred - target; // BCE gradient

                // Update electra head
                for j in 0..feat_dim { electra.w[j] -= lr * err * features[pos][j]; }
                electra.b -= lr * err;

                // Backprop through interpreter
                let half = interp_window / 2;
                let in_dim = interp_window * 7;

                // d_features
                let mut d_feat = vec![0.0f32; feat_dim];
                for f in 0..feat_dim { d_feat[f] = err * electra.w[f]; }

                // Backprop interpreter layer 2
                let mut d_h = vec![0.0f32; interp_hidden];
                // Recompute hidden for this position
                let mut input = vec![0.0f32; in_dim];
                for wi in 0..interp_window {
                    let sp = pos as i32 + wi as i32 - half as i32;
                    if sp >= 0 && (sp as usize) < ctx {
                        for d in 0..7 { input[wi*7+d] = encoded[(sp as usize)*7+d]; }
                    }
                }
                let mut h_vals = vec![0.0f32; interp_hidden];
                for k in 0..interp_hidden { h_vals[k]=interp.b1[k]; for j in 0..in_dim { h_vals[k]+=interp.w1[k][j]*input[j]; } h_vals[k]=h_vals[k].max(0.0); }

                for f in 0..feat_dim {
                    for k in 0..interp_hidden { d_h[k]+=d_feat[f]*interp.w2[f][k]; interp.w2[f][k]-=lr*d_feat[f]*h_vals[k]; }
                    interp.b2[f]-=lr*d_feat[f];
                }
                for k in 0..interp_hidden {
                    if h_vals[k]<=0.0 { continue; }
                    for j in 0..in_dim { interp.w1[k][j]-=lr*d_h[k]*input[j]; }
                    interp.b1[k]-=lr*d_h[k];
                }

                total_loss += -(target*(pred.max(1e-7)).ln() + (1.0-target)*((1.0-pred).max(1e-7)).ln());
                if (pred > 0.5) == (target > 0.5) { n_correct += 1; }
                n_total += 1;
            }
        }

        if (ep+1) % 20 == 0 {
            println!("  ep{:>3}: loss={:.3} detect_acc={:.1}%",
                ep+1, total_loss/n_total as f32, n_correct as f64/n_total as f64*100.0);
        }
    }

    println!("\n  Interpreter params: {}", interp.params());
    println!("  ELECTRA pretraining done.\n");

    // ═══════════════════════════════════════════
    // PHASE 2: Train brain on frozen interpreter
    // ═══════════════════════════════════════════
    println!("━━━ PHASE 2: Brain training (prediction) ━━━\n");

    let brain_idim = ctx * feat_dim; // interpreter features flattened
    let brain_hdim = 256;

    // A) ELECTRA-pretrained interpreter (frozen) + brain
    {
        let tc = Instant::now();
        let mut brain = Brain::new(brain_idim, brain_hdim, &mut rng);
        for ep in 0..200 {
            let lr = 0.01 * (1.0 - ep as f32 / 200.0 * 0.8);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..samples {
                let off = rt.range(0, corpus.len()-ctx-1);
                let encoded = encode_seq(&corpus[off..off+ctx]);
                let features = interp.forward_all(&encoded, ctx);
                let flat: Vec<f32> = features.into_iter().flatten().collect();
                brain.train_step(&flat, corpus[off+ctx], lr);
            }
        }
        // Eval
        let mut eval_rng=Rng::new(999); let mut ok=0; let mut tot=0;
        for _ in 0..5000 { if corpus.len()<ctx+1{break;} let off=eval_rng.range(0,corpus.len()-ctx-1);
            let encoded=encode_seq(&corpus[off..off+ctx]);
            let features=interp.forward_all(&encoded, ctx);
            let flat: Vec<f32> = features.into_iter().flatten().collect();
            let (_,logits)=brain.forward(&flat);
            let pred=logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred==corpus[off+ctx] as usize {ok+=1;} tot+=1;
        }
        let acc = ok as f64/tot as f64*100.0;
        let total_params = interp.params() + brain.params();
        let m = if acc>80.0{" ★★★"} else if acc>60.0{" ★★"} else if acc>40.0{" ★"} else {""};
        println!("  A) ELECTRA interpreter (frozen) + brain: {:.1}% ({} params, {:.1}s){}",
            acc, total_params, tc.elapsed().as_secs_f64(), m);
    }

    // B) Random interpreter (frozen) + brain (control)
    {
        let tc = Instant::now();
        let rand_interp = Interpreter::new(interp_window, interp_hidden, feat_dim, &mut Rng::new(999));
        let mut brain = Brain::new(brain_idim, brain_hdim, &mut rng);
        for ep in 0..200 {
            let lr = 0.01 * (1.0 - ep as f32 / 200.0 * 0.8);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..samples {
                let off = rt.range(0, corpus.len()-ctx-1);
                let encoded = encode_seq(&corpus[off..off+ctx]);
                let features = rand_interp.forward_all(&encoded, ctx);
                let flat: Vec<f32> = features.into_iter().flatten().collect();
                brain.train_step(&flat, corpus[off+ctx], lr);
            }
        }
        let mut eval_rng=Rng::new(999); let mut ok=0; let mut tot=0;
        for _ in 0..5000 { if corpus.len()<ctx+1{break;} let off=eval_rng.range(0,corpus.len()-ctx-1);
            let encoded=encode_seq(&corpus[off..off+ctx]);
            let features=rand_interp.forward_all(&encoded, ctx);
            let flat: Vec<f32> = features.into_iter().flatten().collect();
            let (_,logits)=brain.forward(&flat);
            let pred=logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred==corpus[off+ctx] as usize {ok+=1;} tot+=1;
        }
        let acc = ok as f64/tot as f64*100.0;
        println!("  B) Random interpreter (frozen) + brain:  {:.1}% (control, {:.1}s)",
            acc, tc.elapsed().as_secs_f64());
    }

    // C) No interpreter, brain directly on encoded bytes
    {
        let tc = Instant::now();
        let raw_idim = ctx * 7;
        let mut brain = Brain::new(raw_idim, brain_hdim, &mut rng);
        for ep in 0..200 {
            let lr = 0.01 * (1.0 - ep as f32 / 200.0 * 0.8);
            let mut rt = Rng::new(ep as u64 * 1000 + 42);
            for _ in 0..samples {
                let off = rt.range(0, corpus.len()-ctx-1);
                let encoded = encode_seq(&corpus[off..off+ctx]);
                brain.train_step(&encoded, corpus[off+ctx], lr);
            }
        }
        let mut eval_rng=Rng::new(999); let mut ok=0; let mut tot=0;
        for _ in 0..5000 { if corpus.len()<ctx+1{break;} let off=eval_rng.range(0,corpus.len()-ctx-1);
            let encoded=encode_seq(&corpus[off..off+ctx]);
            let (_,logits)=brain.forward(&encoded);
            let pred=logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            if pred==corpus[off+ctx] as usize {ok+=1;} tot+=1;
        }
        let acc = ok as f64/tot as f64*100.0;
        println!("  C) No interpreter, brain on raw encoded:  {:.1}% ({} params, {:.1}s)",
            acc, brain.params(), tc.elapsed().as_secs_f64());
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
