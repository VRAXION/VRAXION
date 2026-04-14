//! Merger architecture sweep — find which architecture hits EXACT 100%
//!
//! Problem: sigmoid tied-weight AE tops at 99.98%, never 100%.
//! Test alternatives:
//!   A) Linear AE (no activation, tied weights)
//!   B) Linear AE (no activation, non-tied weights)
//!   C) ReLU AE (non-tied weights)
//!   D) Sigmoid AE (non-tied weights) — more capacity than tied
//!   E) Two-layer encoder: 112→hidden→bottleneck→hidden→112
//!
//! Eval on held-out 20% corpus. Target: EXACT 100%.
//!
//! Run: cargo run --example merger_arch_sweep --release

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

fn sigmoid(x: f32) -> f32 { 1.0/(1.0+(-x).exp()) }

fn encode7(ch: u8) -> [f32;7] {
    const W:[[i8;8];7]=[[-1,1,1,-1,-1,1,1,-1],[1,-1,1,1,-1,-1,-1,-1],[-1,-1,1,-1,1,1,-1,-1],[-1,-1,-1,1,-1,-1,1,-1],[-1,1,1,1,-1,-1,-1,-1],[1,1,-1,-1,-1,-1,-1,-1],[-1,1,-1,1,1,1,-1,-1]];
    const B:[i8;7]=[1,1,1,1,1,1,1]; const C:[f32;7]=[10.0;7]; const RHO:[f32;7]=[2.0,0.0,0.0,0.0,0.0,0.0,0.0];
    let mut bits=[0.0f32;8]; for i in 0..8{bits[i]=((ch>>i)&1) as f32;}
    let mut o=[0.0f32;7];
    for k in 0..7{let mut d=B[k] as f32; for j in 0..8{d+=W[k][j] as f32*bits[j];} o[k]=c19(d,C[k],RHO[k]);} o
}

fn encode_seq7(chars: &[u8]) -> Vec<f32> { chars.iter().flat_map(|&ch| encode7(ch).to_vec()).collect() }

fn check_acc(corpus: &[u8], ctx: usize, start: usize, end: usize, enc_fn: &dyn Fn(&[f32])->Vec<f32>, dec_fn: &dyn Fn(&[f32])->Vec<f32>) -> (usize, usize) {
    let mut ok=0; let mut tot=0; let mut pos=start;
    while pos+ctx<=end {
        let sig=encode_seq7(&corpus[pos..pos+ctx]);
        let h=enc_fn(&sig); let out=dec_fn(&h);
        for i in 0..ctx {
            let rs=&out[i*7..(i+1)*7];
            let mut best=0u8; let mut bd=f32::MAX;
            for ch in 0..27u8{let code=encode7(ch); let d:f32=code.iter().zip(rs).map(|(a,b)|(a-b)*(a-b)).sum(); if d<bd{bd=d;best=ch;}}
            if best==corpus[pos+i]{ok+=1;} tot+=1;
        }
        pos+=1; if tot>300_000{break;}
    }
    (ok, tot)
}

fn main() {
    let t0=Instant::now();
    let corpus=load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let ctx=16; let idim=ctx*7; // 112
    let split=corpus.len()*80/100;

    println!("=== MERGER ARCHITECTURE SWEEP FOR 100% ===\n");
    println!("  Input: {} dim, Train: {} chars, Test: {} chars\n", idim, split, corpus.len()-split);
    println!("  {:>35} {:>6} {:>12} {:>12} {:>7}",
        "architecture", "bneck", "train", "test", "time");
    println!("  {}", "─".repeat(75));

    let bneck = 96; // 86% ratio — proven sweet spot
    let epochs = 300;
    let samples = 15000.min(split/ctx);

    // A) Linear tied weights
    {
        let tc=Instant::now();
        let mut rng=Rng::new(42);
        let s=(2.0/idim as f32).sqrt();
        let mut w:Vec<Vec<f32>>=(0..bneck).map(|_|(0..idim).map(|_|rng.normal()*s).collect()).collect();
        let mut enc_b=vec![0.0f32;bneck]; let mut dec_b=vec![0.0f32;idim];
        for ep in 0..epochs { let lr=0.01*(1.0-ep as f32/epochs as f32*0.8); let mut rt=Rng::new(ep as u64*1000+42);
            for _ in 0..samples { let off=rt.range(0,split-ctx); let input=encode_seq7(&corpus[off..off+ctx]);
                // Encode (linear)
                let mut h=vec![0.0f32;bneck];
                for k in 0..bneck{h[k]=enc_b[k]; for j in 0..idim{h[k]+=w[k][j]*input[j];}}
                // Decode (tied Wᵀ)
                let mut o=vec![0.0f32;idim];
                for j in 0..idim{o[j]=dec_b[j]; for k in 0..bneck{o[j]+=w[k][j]*h[k];}}
                // Backprop
                let mut d_o=vec![0.0f32;idim];
                for j in 0..idim{d_o[j]=2.0*(o[j]-input[j])/idim as f32;}
                let mut d_h=vec![0.0f32;bneck];
                for j in 0..idim{for k in 0..bneck{d_h[k]+=d_o[j]*w[k][j]; w[k][j]-=lr*d_o[j]*h[k];} dec_b[j]-=lr*d_o[j];}
                for k in 0..bneck{for j in 0..idim{w[k][j]-=lr*d_h[k]*input[j];} enc_b[k]-=lr*d_h[k];}
            }
        }
        let enc=|x:&[f32]|->Vec<f32>{let mut h=vec![0.0f32;bneck]; for k in 0..bneck{h[k]=enc_b[k]; for j in 0..idim{h[k]+=w[k][j]*x[j];}} h};
        let dec=|h:&[f32]|->Vec<f32>{let mut o=vec![0.0f32;idim]; for j in 0..idim{o[j]=dec_b[j]; for k in 0..bneck{o[j]+=w[k][j]*h[k];}} o};
        let(tr_ok,tr_tot)=check_acc(&corpus,ctx,0,split,&enc,&dec);
        let(te_ok,te_tot)=check_acc(&corpus,ctx,split,corpus.len(),&enc,&dec);
        let tr_m=if tr_ok==tr_tot{"★★★"}else{""}; let te_m=if te_ok==te_tot{"★★★"}else{""};
        println!("  {:>35} {:>6} {:>7}/{:>7}{:>4} {:>7}/{:>7}{:>4} {:>6.1}s",
            "Linear tied", bneck, tr_ok,tr_tot,tr_m, te_ok,te_tot,te_m, tc.elapsed().as_secs_f64());
    }

    // B) Linear non-tied weights
    {
        let tc=Instant::now();
        let mut rng=Rng::new(42);
        let s1=(2.0/idim as f32).sqrt(); let s2=(2.0/bneck as f32).sqrt();
        let mut we:Vec<Vec<f32>>=(0..bneck).map(|_|(0..idim).map(|_|rng.normal()*s1).collect()).collect();
        let mut wd:Vec<Vec<f32>>=(0..idim).map(|_|(0..bneck).map(|_|rng.normal()*s2).collect()).collect();
        let mut enc_b=vec![0.0f32;bneck]; let mut dec_b=vec![0.0f32;idim];
        for ep in 0..epochs { let lr=0.01*(1.0-ep as f32/epochs as f32*0.8); let mut rt=Rng::new(ep as u64*1000+42);
            for _ in 0..samples { let off=rt.range(0,split-ctx); let input=encode_seq7(&corpus[off..off+ctx]);
                let mut h=vec![0.0f32;bneck];
                for k in 0..bneck{h[k]=enc_b[k]; for j in 0..idim{h[k]+=we[k][j]*input[j];}}
                let mut o=vec![0.0f32;idim];
                for j in 0..idim{o[j]=dec_b[j]; for k in 0..bneck{o[j]+=wd[j][k]*h[k];}}
                let mut d_o=vec![0.0f32;idim];
                for j in 0..idim{d_o[j]=2.0*(o[j]-input[j])/idim as f32;}
                let mut d_h=vec![0.0f32;bneck];
                for j in 0..idim{for k in 0..bneck{d_h[k]+=d_o[j]*wd[j][k]; wd[j][k]-=lr*d_o[j]*h[k];} dec_b[j]-=lr*d_o[j];}
                for k in 0..bneck{for j in 0..idim{we[k][j]-=lr*d_h[k]*input[j];} enc_b[k]-=lr*d_h[k];}
            }
        }
        let enc=|x:&[f32]|->Vec<f32>{let mut h=vec![0.0f32;bneck]; for k in 0..bneck{h[k]=enc_b[k]; for j in 0..idim{h[k]+=we[k][j]*x[j];}} h};
        let dec=|h:&[f32]|->Vec<f32>{let mut o=vec![0.0f32;idim]; for j in 0..idim{o[j]=dec_b[j]; for k in 0..bneck{o[j]+=wd[j][k]*h[k];}} o};
        let(tr_ok,tr_tot)=check_acc(&corpus,ctx,0,split,&enc,&dec);
        let(te_ok,te_tot)=check_acc(&corpus,ctx,split,corpus.len(),&enc,&dec);
        let tr_m=if tr_ok==tr_tot{"★★★"}else{""}; let te_m=if te_ok==te_tot{"★★★"}else{""};
        println!("  {:>35} {:>6} {:>7}/{:>7}{:>4} {:>7}/{:>7}{:>4} {:>6.1}s",
            "Linear non-tied", bneck, tr_ok,tr_tot,tr_m, te_ok,te_tot,te_m, tc.elapsed().as_secs_f64());
    }

    // C) ReLU encoder, linear decoder, non-tied
    {
        let tc=Instant::now();
        let mut rng=Rng::new(42);
        let s1=(2.0/idim as f32).sqrt(); let s2=(2.0/bneck as f32).sqrt();
        let mut we:Vec<Vec<f32>>=(0..bneck).map(|_|(0..idim).map(|_|rng.normal()*s1).collect()).collect();
        let mut wd:Vec<Vec<f32>>=(0..idim).map(|_|(0..bneck).map(|_|rng.normal()*s2).collect()).collect();
        let mut enc_b=vec![0.0f32;bneck]; let mut dec_b=vec![0.0f32;idim];
        for ep in 0..epochs { let lr=0.01*(1.0-ep as f32/epochs as f32*0.8); let mut rt=Rng::new(ep as u64*1000+42);
            for _ in 0..samples { let off=rt.range(0,split-ctx); let input=encode_seq7(&corpus[off..off+ctx]);
                let mut h=vec![0.0f32;bneck];
                for k in 0..bneck{h[k]=enc_b[k]; for j in 0..idim{h[k]+=we[k][j]*input[j];} h[k]=h[k].max(0.0);}
                let mut o=vec![0.0f32;idim];
                for j in 0..idim{o[j]=dec_b[j]; for k in 0..bneck{o[j]+=wd[j][k]*h[k];}}
                let mut d_o=vec![0.0f32;idim];
                for j in 0..idim{d_o[j]=2.0*(o[j]-input[j])/idim as f32;}
                let mut d_h=vec![0.0f32;bneck];
                for j in 0..idim{for k in 0..bneck{d_h[k]+=d_o[j]*wd[j][k]; wd[j][k]-=lr*d_o[j]*h[k];} dec_b[j]-=lr*d_o[j];}
                for k in 0..bneck{if h[k]<=0.0{continue;} for j in 0..idim{we[k][j]-=lr*d_h[k]*input[j];} enc_b[k]-=lr*d_h[k];}
            }
        }
        let enc=|x:&[f32]|->Vec<f32>{let mut h=vec![0.0f32;bneck]; for k in 0..bneck{h[k]=enc_b[k]; for j in 0..idim{h[k]+=we[k][j]*x[j];} h[k]=h[k].max(0.0);} h};
        let dec=|h:&[f32]|->Vec<f32>{let mut o=vec![0.0f32;idim]; for j in 0..idim{o[j]=dec_b[j]; for k in 0..bneck{o[j]+=wd[j][k]*h[k];}} o};
        let(tr_ok,tr_tot)=check_acc(&corpus,ctx,0,split,&enc,&dec);
        let(te_ok,te_tot)=check_acc(&corpus,ctx,split,corpus.len(),&enc,&dec);
        let tr_m=if tr_ok==tr_tot{"★★★"}else{""}; let te_m=if te_ok==te_tot{"★★★"}else{""};
        println!("  {:>35} {:>6} {:>7}/{:>7}{:>4} {:>7}/{:>7}{:>4} {:>6.1}s",
            "ReLU enc + linear dec (non-tied)", bneck, tr_ok,tr_tot,tr_m, te_ok,te_tot,te_m, tc.elapsed().as_secs_f64());
    }

    // D) Sigmoid non-tied
    {
        let tc=Instant::now();
        let mut rng=Rng::new(42);
        let s1=(2.0/idim as f32).sqrt(); let s2=(2.0/bneck as f32).sqrt();
        let mut we:Vec<Vec<f32>>=(0..bneck).map(|_|(0..idim).map(|_|rng.normal()*s1).collect()).collect();
        let mut wd:Vec<Vec<f32>>=(0..idim).map(|_|(0..bneck).map(|_|rng.normal()*s2).collect()).collect();
        let mut enc_b=vec![0.0f32;bneck]; let mut dec_b=vec![0.0f32;idim];
        for ep in 0..epochs { let lr=0.01*(1.0-ep as f32/epochs as f32*0.8); let mut rt=Rng::new(ep as u64*1000+42);
            for _ in 0..samples { let off=rt.range(0,split-ctx); let input=encode_seq7(&corpus[off..off+ctx]);
                let mut h=vec![0.0f32;bneck];
                for k in 0..bneck{h[k]=enc_b[k]; for j in 0..idim{h[k]+=we[k][j]*input[j];} h[k]=sigmoid(h[k]);}
                let mut o=vec![0.0f32;idim];
                for j in 0..idim{o[j]=dec_b[j]; for k in 0..bneck{o[j]+=wd[j][k]*h[k];}}
                let mut d_o=vec![0.0f32;idim];
                for j in 0..idim{d_o[j]=2.0*(o[j]-input[j])/idim as f32;}
                let mut d_h=vec![0.0f32;bneck];
                for j in 0..idim{for k in 0..bneck{d_h[k]+=d_o[j]*wd[j][k]; wd[j][k]-=lr*d_o[j]*h[k];} dec_b[j]-=lr*d_o[j];}
                for k in 0..bneck{let dh=d_h[k]*h[k]*(1.0-h[k]); for j in 0..idim{we[k][j]-=lr*dh*input[j];} enc_b[k]-=lr*dh;}
            }
        }
        let enc=|x:&[f32]|->Vec<f32>{let mut h=vec![0.0f32;bneck]; for k in 0..bneck{h[k]=enc_b[k]; for j in 0..idim{h[k]+=we[k][j]*x[j];} h[k]=sigmoid(h[k]);} h};
        let dec=|h:&[f32]|->Vec<f32>{let mut o=vec![0.0f32;idim]; for j in 0..idim{o[j]=dec_b[j]; for k in 0..bneck{o[j]+=wd[j][k]*h[k];}} o};
        let(tr_ok,tr_tot)=check_acc(&corpus,ctx,0,split,&enc,&dec);
        let(te_ok,te_tot)=check_acc(&corpus,ctx,split,corpus.len(),&enc,&dec);
        let tr_m=if tr_ok==tr_tot{"★★★"}else{""}; let te_m=if te_ok==te_tot{"★★★"}else{""};
        println!("  {:>35} {:>6} {:>7}/{:>7}{:>4} {:>7}/{:>7}{:>4} {:>6.1}s",
            "Sigmoid non-tied", bneck, tr_ok,tr_tot,tr_m, te_ok,te_tot,te_m, tc.elapsed().as_secs_f64());
    }

    // Also try bneck=112 (no compression, just transform)
    println!();
    {
        let tc=Instant::now();
        let mut rng=Rng::new(42);
        let s=(2.0/idim as f32).sqrt();
        let mut w:Vec<Vec<f32>>=(0..idim).map(|_|(0..idim).map(|_|rng.normal()*s).collect()).collect();
        let mut b=vec![0.0f32;idim];
        // Identity init
        for i in 0..idim{w[i][i]+=1.0;}
        for ep in 0..100 { let lr=0.001*(1.0-ep as f32/100.0*0.5); let mut rt=Rng::new(ep as u64*1000+42);
            for _ in 0..samples { let off=rt.range(0,split-ctx); let input=encode_seq7(&corpus[off..off+ctx]);
                let mut o=vec![0.0f32;idim];
                for j in 0..idim{o[j]=b[j]; for k in 0..idim{o[j]+=w[j][k]*input[k];}}
                for j in 0..idim{let d=2.0*(o[j]-input[j])/idim as f32; for k in 0..idim{w[j][k]-=lr*d*input[k];} b[j]-=lr*d;}
            }
        }
        let enc=|x:&[f32]|->Vec<f32>{x.to_vec()};
        let dec=|h:&[f32]|->Vec<f32>{let mut o=vec![0.0f32;idim]; for j in 0..idim{o[j]=b[j]; for k in 0..idim{o[j]+=w[j][k]*h[k];}} o};
        let(tr_ok,tr_tot)=check_acc(&corpus,ctx,0,split,&enc,&dec);
        let(te_ok,te_tot)=check_acc(&corpus,ctx,split,corpus.len(),&enc,&dec);
        let tr_m=if tr_ok==tr_tot{"★★★"}else{""}; let te_m=if te_ok==te_tot{"★★★"}else{""};
        println!("  {:>35} {:>6} {:>7}/{:>7}{:>4} {:>7}/{:>7}{:>4} {:>6.1}s",
            "Identity (no compression)", idim, tr_ok,tr_tot,tr_m, te_ok,te_tot,te_m, tc.elapsed().as_secs_f64());
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
