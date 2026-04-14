//! Full proven pipeline: L0 (byte interp) → L1 (merger, frozen 100%) → L2 (conv) → prediction
//!
//! Each layer proven before the next:
//!   L0: 7-neuron byte interpreter (binary C19) — 100% round-trip ✓
//!   L1: Linear tied merger (112→96, 86%) — 100% reconstruction ✓
//!   L2: Conv pattern finder (on merged 96-dim) — trained
//!   Prediction: simple brain on L2 features
//!
//! Compare: with vs without merger (does 100% merger help or hurt?)
//! Eval on held-out 20% test set.
//!
//! Run: cargo run --example full_proven_pipeline --release

use std::time::Instant;

struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 { self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.s }
    fn normal(&mut self) -> f32 { let u1=(((self.next()>>33)%65536) as f32/65536.0).max(1e-7); let u2=((self.next()>>33)%65536) as f32/65536.0; (-2.0*u1.ln()).sqrt()*(2.0*std::f32::consts::PI*u2).cos() }
    fn range(&mut self, lo: usize, hi: usize) -> usize { if hi<=lo{lo}else{lo+(self.next() as usize%(hi-lo))} }
}

fn load_corpus(path: &str) -> Vec<u8> {
    let raw = std::fs::read(path).expect("read");
    let mut c = Vec::new();
    for &b in &raw { match b { b'a'..=b'z' => c.push(b-b'a'), b'A'..=b'Z' => c.push(b-b'A'), b' '|b'\n'|b'\t'|b'\r' => c.push(26), _ => {} } }
    c
}

fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c=c.max(0.1); let rho=rho.max(0.0); let l=6.0*c;
    if x>=l{return x-l;} if x<=-l{return x+l;}
    let s=x/c; let n=s.floor(); let t=s-n; let h=t*(1.0-t);
    let sg=if(n as i32)%2==0{1.0}else{-1.0}; c*(sg*h+rho*h*h)
}

// L0: 7-neuron byte interpreter (frozen, binary)
fn encode7(ch: u8) -> [f32;7] {
    const W:[[i8;8];7]=[[-1,1,1,-1,-1,1,1,-1],[1,-1,1,1,-1,-1,-1,-1],[-1,-1,1,-1,1,1,-1,-1],[-1,-1,-1,1,-1,-1,1,-1],[-1,1,1,1,-1,-1,-1,-1],[1,1,-1,-1,-1,-1,-1,-1],[-1,1,-1,1,1,1,-1,-1]];
    const B:[i8;7]=[1,1,1,1,1,1,1]; const C:[f32;7]=[10.0;7]; const RHO:[f32;7]=[2.0,0.0,0.0,0.0,0.0,0.0,0.0];
    let mut bits=[0.0f32;8]; for i in 0..8{bits[i]=((ch>>i)&1) as f32;}
    let mut o=[0.0f32;7];
    for k in 0..7{let mut d=B[k] as f32; for j in 0..8{d+=W[k][j] as f32*bits[j];} o[k]=c19(d,C[k],RHO[k]);} o
}

// L1: Linear tied merger (train then freeze)
struct Merger {
    w: Vec<Vec<f32>>, enc_b: Vec<f32>, dec_b: Vec<f32>,
    idim: usize, bneck: usize,
}
impl Merger {
    fn new(idim: usize, bneck: usize, rng: &mut Rng) -> Self {
        let s=(2.0/idim as f32).sqrt();
        Merger{w:(0..bneck).map(|_|(0..idim).map(|_|rng.normal()*s).collect()).collect(),
            enc_b:vec![0.0;bneck],dec_b:vec![0.0;idim],idim,bneck}
    }
    fn encode(&self, input: &[f32]) -> Vec<f32> {
        let mut h=vec![0.0f32;self.bneck];
        for k in 0..self.bneck{h[k]=self.enc_b[k]; for j in 0..self.idim{h[k]+=self.w[k][j]*input[j];}} h
    }
    fn train_step(&mut self, input: &[f32], lr: f32) {
        let h=self.encode(input);
        let mut o=vec![0.0f32;self.idim];
        for j in 0..self.idim{o[j]=self.dec_b[j]; for k in 0..self.bneck{o[j]+=self.w[k][j]*h[k];}}
        let mut d_o=vec![0.0f32;self.idim];
        for j in 0..self.idim{d_o[j]=2.0*(o[j]-input[j])/self.idim as f32;}
        let mut d_h=vec![0.0f32;self.bneck];
        for j in 0..self.idim{for k in 0..self.bneck{d_h[k]+=d_o[j]*self.w[k][j]; self.w[k][j]-=lr*d_o[j]*h[k];} self.dec_b[j]-=lr*d_o[j];}
        for k in 0..self.bneck{for j in 0..self.idim{self.w[k][j]-=lr*d_h[k]*input[j];} self.enc_b[k]-=lr*d_h[k];}
    }
}

// L2+Pred: Conv pattern finder + brain
struct ConvBrain {
    conv_w: Vec<Vec<f32>>, conv_b: Vec<f32>,
    w1: Vec<Vec<f32>>, b1: Vec<f32>,
    w2: Vec<Vec<f32>>, b2: Vec<f32>,
    kernel_size: usize, n_filters: usize, sig_dim: usize,
    n_chars: usize, brain_idim: usize, hdim: usize,
}
impl ConvBrain {
    fn new(n_chars: usize, kernel_size: usize, n_filters: usize, sig_dim: usize, hdim: usize, rng: &mut Rng) -> Self {
        let fan_in=kernel_size*sig_dim; let n_pos=n_chars-kernel_size+1; let brain_idim=n_pos*n_filters;
        let sc=(2.0/fan_in as f32).sqrt(); let s1=(2.0/brain_idim as f32).sqrt(); let s2=(2.0/hdim as f32).sqrt();
        ConvBrain{
            conv_w:(0..n_filters).map(|_|(0..fan_in).map(|_|rng.normal()*sc).collect()).collect(),conv_b:vec![0.0;n_filters],
            w1:(0..hdim).map(|_|(0..brain_idim).map(|_|rng.normal()*s1).collect()).collect(),b1:vec![0.0;hdim],
            w2:(0..27).map(|_|(0..hdim).map(|_|rng.normal()*s2).collect()).collect(),b2:vec![0.0;27],
            kernel_size,n_filters,sig_dim,n_chars,brain_idim,hdim}
    }
    fn params(&self)->usize{self.n_filters*self.kernel_size*self.sig_dim+self.n_filters+self.brain_idim*self.hdim+self.hdim+self.hdim*27+27}
    fn forward(&self, signals: &[f32]) -> Vec<f32> {
        let n_pos=self.n_chars-self.kernel_size+1;
        let mut co=vec![0.0f32;n_pos*self.n_filters];
        for p in 0..n_pos{for f in 0..self.n_filters{
            let mut v=self.conv_b[f];
            for ki in 0..self.kernel_size{for d in 0..self.sig_dim{v+=self.conv_w[f][ki*self.sig_dim+d]*signals[(p+ki)*self.sig_dim+d];}}
            co[p*self.n_filters+f]=v.max(0.0);
        }}
        let mut h=vec![0.0f32;self.hdim];
        for k in 0..self.hdim{h[k]=self.b1[k]; for j in 0..self.brain_idim{h[k]+=self.w1[k][j]*co[j];} h[k]=h[k].max(0.0);}
        let mut logits=vec![0.0f32;27];
        for c in 0..27{logits[c]=self.b2[c]; for k in 0..self.hdim{logits[c]+=self.w2[c][k]*h[k];}}
        logits
    }
    fn train_step(&mut self, signals: &[f32], target: u8, lr: f32) {
        let n_pos=self.n_chars-self.kernel_size+1;
        let mut co=vec![0.0f32;n_pos*self.n_filters];
        for p in 0..n_pos{for f in 0..self.n_filters{
            let mut v=self.conv_b[f];
            for ki in 0..self.kernel_size{for d in 0..self.sig_dim{v+=self.conv_w[f][ki*self.sig_dim+d]*signals[(p+ki)*self.sig_dim+d];}}
            co[p*self.n_filters+f]=v.max(0.0);
        }}
        let mut h=vec![0.0f32;self.hdim];
        for k in 0..self.hdim{h[k]=self.b1[k]; for j in 0..self.brain_idim{h[k]+=self.w1[k][j]*co[j];} h[k]=h[k].max(0.0);}
        let mut logits=vec![0.0f32;27];
        for c in 0..27{logits[c]=self.b2[c]; for k in 0..self.hdim{logits[c]+=self.w2[c][k]*h[k];}}
        let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
        let mut p2=vec![0.0f32;27]; let mut s=0.0f32;
        for c in 0..27{p2[c]=(logits[c]-mx).exp(); s+=p2[c];} for c in 0..27{p2[c]/=s;}
        let mut dl=p2; dl[target as usize]-=1.0;
        let mut dh=vec![0.0f32;self.hdim];
        for c in 0..27{for k in 0..self.hdim{dh[k]+=dl[c]*self.w2[c][k]; self.w2[c][k]-=lr*dl[c]*h[k];} self.b2[c]-=lr*dl[c];}
        let mut dc=vec![0.0f32;self.brain_idim];
        for k in 0..self.hdim{if h[k]<=0.0{continue;} for j in 0..self.brain_idim{dc[j]+=dh[k]*self.w1[k][j]; self.w1[k][j]-=lr*dh[k]*co[j];} self.b1[k]-=lr*dh[k];}
        for p in 0..n_pos{for f in 0..self.n_filters{
            let idx=p*self.n_filters+f; if co[idx]<=0.0{continue;}
            for ki in 0..self.kernel_size{for d in 0..self.sig_dim{self.conv_w[f][ki*self.sig_dim+d]-=lr*dc[idx]*signals[(p+ki)*self.sig_dim+d];}}
            self.conv_b[f]-=lr*dc[idx];
        }}
    }
}

fn main() {
    let t0=Instant::now();
    let corpus=load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let ctx=16; let split=corpus.len()*80/100;

    println!("=== FULL PROVEN PIPELINE ===\n");
    println!("  Train: {} chars, Test: {} chars (held-out)\n", split, corpus.len()-split);

    // ── Step 1: Train & freeze L1 merger ──
    println!("━━━ Step 1: Train L1 merger (linear tied 112→96) ━━━");
    let mut rng=Rng::new(42);
    let mut merger=Merger::new(112, 96, &mut rng);
    for ep in 0..300{let lr=0.01*(1.0-ep as f32/300.0*0.8); let mut rt=Rng::new(ep as u64*1000+42);
        for _ in 0..15000.min(split/ctx){let off=rt.range(0,split-ctx);
            let sig:Vec<f32>=corpus[off..off+ctx].iter().flat_map(|&ch|encode7(ch).to_vec()).collect();
            merger.train_step(&sig,lr);}}
    println!("  Merger trained & frozen.\n");

    // ── Step 2: Compare pipelines ──
    println!("━━━ Step 2: Train conv+brain on frozen L0+L1 ━━━\n");

    struct Cfg{name:&'static str, use_merger:bool, sig_dim:usize, k:usize, f:usize, h:usize}
    let configs=vec![
        Cfg{name:"L0→Conv(k=3,f=64)→brain(h=512)",       use_merger:false, sig_dim:7,  k:3, f:64,  h:512},
        Cfg{name:"L0→L1→Conv(k=3,f=64)→brain(h=512)",    use_merger:true,  sig_dim:6,  k:3, f:64,  h:512},
        Cfg{name:"L0→Conv(k=3,f=64)→brain(h=256)",       use_merger:false, sig_dim:7,  k:3, f:64,  h:256},
        Cfg{name:"L0→L1→Conv(k=3,f=64)→brain(h=256)",    use_merger:true,  sig_dim:6,  k:3, f:64,  h:256},
    ];

    println!("  {:>42} {:>8} {:>10} {:>10} {:>7}",
        "pipeline","params","train%","test%","time");
    println!("  {}","─".repeat(80));

    for cfg in &configs {
        let tc=Instant::now();
        let mut rng2=Rng::new(42);
        let n_chars=ctx;
        let cb=&mut ConvBrain::new(n_chars, cfg.k, cfg.f, cfg.sig_dim, cfg.h, &mut rng2);

        let samples=15000.min(split/(ctx+1));
        for ep in 0..200{let lr=0.01*(1.0-ep as f32/200.0*0.8); let mut rt=Rng::new(ep as u64*1000+42);
            for _ in 0..samples{let off=rt.range(0,split-ctx-1);
                let raw:Vec<f32>=corpus[off..off+ctx].iter().flat_map(|&ch|encode7(ch).to_vec()).collect();
                let signals = if cfg.use_merger {
                    // Merge: 112→96, then reshape as 16×6
                    let merged=merger.encode(&raw);
                    merged
                } else { raw };
                cb.train_step(&signals, corpus[off+ctx], lr);
            }
        }

        // Eval
        let eval=|start:usize,end:usize|->f64{
            let mut rng3=Rng::new(999); let mut ok=0; let mut tot=0;
            for _ in 0..5000{if end<start+ctx+1{break;} let off=rng3.range(start,end-ctx-1);
                let raw:Vec<f32>=corpus[off..off+ctx].iter().flat_map(|&ch|encode7(ch).to_vec()).collect();
                let signals=if cfg.use_merger{merger.encode(&raw)}else{raw};
                let logits=cb.forward(&signals);
                let pred=logits.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap()).unwrap().0;
                if pred==corpus[off+ctx] as usize{ok+=1;} tot+=1;}
            ok as f64/tot as f64*100.0};
        let tr_acc=eval(0,split);
        let te_acc=eval(split,corpus.len());
        let m=if te_acc>90.0{" ★★★"}else if te_acc>80.0{" ★★"}else if te_acc>50.0{" ★"}else{""};
        println!("  {:>42} {:>8} {:>9.1}% {:>9.1}% {:>6.1}s{}",
            cfg.name, cb.params(), tr_acc, te_acc, tc.elapsed().as_secs_f64(), m);
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
