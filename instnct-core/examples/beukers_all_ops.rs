//! ALL missing operations sweep — what else can a neuron learn?
//!
//! Standard neuron has: multiply (w×x), add (bias)
//! Beukers adds: cross-multiply (a×b)
//! What about: log, exp, square, sqrt, abs, power?
//!
//! Run: cargo run --example beukers_all_ops --release

use std::time::Instant;
const VOCAB:usize=2000;const DIM:usize=16;
struct Rng(u64);
impl Rng{fn new(s:u64)->Self{Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1))}
fn next(&mut self)->u64{self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);self.0}
fn normal(&mut self)->f32{let u1=(((self.next()>>33)%65536)as f32/65536.0).max(1e-7);let u2=((self.next()>>33)%65536)as f32/65536.0;(-2.0*u1.ln()).sqrt()*(2.0*std::f32::consts::PI*u2).cos()}
fn range(&mut self,lo:usize,hi:usize)->usize{if hi<=lo{lo}else{lo+(self.next()as usize%(hi-lo))}}}
fn load_corpus(p:&str)->Vec<u8>{std::fs::read(p).expect("r").iter().filter_map(|&b|match b{
    b'a'..=b'z'=>Some(b-b'a'),b'A'..=b'Z'=>Some(b-b'A'),b' '|b'\n'|b'\t'|b'\r'=>Some(26),_=>None}).collect()}

fn run(corpus:&[u8],split:usize,name:&str,
    gate: &dyn Fn(f32,f32,&[f32])->f32, // (proj1, proj2, learnable_params) -> output
    n_lp: usize, init_lp: &[f32],
) -> f64 {
    let ctx=32;let mask_pos=ctx/2;let k=7;let hk=3i32;let fan=k*DIM;let nf=32;
    let mut rng=Rng::new(42);
    let sc_e=(1.0/DIM as f32).sqrt();let sc_c=(2.0/fan as f32).sqrt();let sc_h=(2.0/nf as f32).sqrt();
    let mut embed:Vec<[f32;DIM]>=(0..VOCAB).map(|_|{let mut v=[0.0;DIM];for d in 0..DIM{v[d]=rng.normal()*sc_e;}v}).collect();
    let mut w1:Vec<Vec<f32>>=(0..nf).map(|_|(0..fan).map(|_|rng.normal()*sc_c).collect()).collect();
    let mut b1=vec![0.0f32;nf];
    let mut w2:Vec<Vec<f32>>=(0..nf).map(|_|(0..fan).map(|_|rng.normal()*sc_c).collect()).collect();
    let mut b2=vec![0.0f32;nf];
    let mut lp:Vec<Vec<f32>>=(0..nf).map(|_|init_lp.to_vec()).collect();
    let mut hw:Vec<Vec<f32>>=(0..27).map(|_|(0..nf).map(|_|rng.normal()*sc_h).collect()).collect();
    let mut hb=vec![0.0f32;27];
    let samples=2000;let max_ep=300;let mut best_test=0.0f64;
    print!("  {:>35}:",name);
    for ep in 0..max_ep{
        let lr=0.01*(1.0-ep as f32/max_ep as f32*0.8);
        let mut rt=Rng::new(ep as u64*1000+42);
        for _ in 0..samples{
            let off=rt.range(0,split.saturating_sub(ctx+1));
            let chunk=&corpus[off..off+ctx];let target=chunk[mask_pos]as usize;
            let emb:Vec<[f32;DIM]>=(0..ctx).map(|i|if i==mask_pos{[0.0;DIM]}else{embed[chunk[i]as usize]}).collect();
            let mut p1=vec![0.0f32;nf];let mut p2=vec![0.0f32;nf];
            for f in 0..nf{let mut v1=b1[f];let mut v2=b2[f];
                for ki in 0..k{let pos=mask_pos as i32+ki as i32-hk;
                    if pos>=0&&(pos as usize)<ctx{for d in 0..DIM{
                        v1+=w1[f][ki*DIM+d]*emb[pos as usize][d];
                        v2+=w2[f][ki*DIM+d]*emb[pos as usize][d];}}}
                p1[f]=v1;p2[f]=v2;}
            let mut co=vec![0.0f32;nf];
            for f in 0..nf{co[f]=gate(p1[f],p2[f],&lp[f]).max(-10.0).min(10.0);}
            let mut logits=hb.clone();
            for c in 0..27{for f in 0..nf{logits[c]+=hw[c][f]*co[f];}}
            let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
            let mut pr=vec![0.0f32;27];let mut s=0.0f32;
            for c in 0..27{pr[c]=(logits[c]-mx).exp();s+=pr[c];}for c in 0..27{pr[c]/=s;}
            pr[target]-=1.0;
            let mut dc=vec![0.0f32;nf];
            for c in 0..27{for f in 0..nf{dc[f]+=pr[c]*hw[c][f];hw[c][f]-=lr*pr[c]*co[f];}hb[c]-=lr*pr[c];}
            // Numerical gradient for gate
            for f in 0..nf{let eps_g=0.01;
                // d/d(p1)
                let cp=gate(p1[f]+eps_g,p2[f],&lp[f]).max(-10.0).min(10.0);
                let cm=gate(p1[f]-eps_g,p2[f],&lp[f]).max(-10.0).min(10.0);
                let g1=dc[f]*(cp-cm)/(2.0*eps_g);
                let cp=gate(p1[f],p2[f]+eps_g,&lp[f]).max(-10.0).min(10.0);
                let cm=gate(p1[f],p2[f]-eps_g,&lp[f]).max(-10.0).min(10.0);
                let g2=dc[f]*(cp-cm)/(2.0*eps_g);
                for ki in 0..k{let pos=mask_pos as i32+ki as i32-hk;
                    if pos>=0&&(pos as usize)<ctx{let pi=pos as usize;
                        for d in 0..DIM{if pi!=mask_pos{embed[chunk[pi]as usize][d]-=lr*(g1+g2)*w1[f][ki*DIM+d]*0.25;}
                            w1[f][ki*DIM+d]-=lr*g1*emb[pi][d];w2[f][ki*DIM+d]-=lr*g2*emb[pi][d];}}}
                b1[f]-=lr*g1;b2[f]-=lr*g2;
                // d/d(learnable params)
                for pi in 0..n_lp{let mut pp=lp[f].clone();pp[pi]+=0.01;let mut pm=lp[f].clone();pm[pi]-=0.01;
                    let pg=(gate(p1[f],p2[f],&pp)-gate(p1[f],p2[f],&pm))/0.02;
                    lp[f][pi]-=lr*dc[f]*pg*0.1;lp[f][pi]=lp[f][pi].max(-10.0).min(10.0);}
            }
        }
        if ep%50==0{
            let eval=|start:usize,end:usize|->f64{let mut rng3=Rng::new(999);let mut ok=0usize;let mut tot=0usize;
                for _ in 0..1000{if end<start+ctx+1{break;}let off=rng3.range(start,end.saturating_sub(ctx+1));
                    let chunk=&corpus[off..off+ctx];
                    let emb:Vec<[f32;DIM]>=(0..ctx).map(|i|if i==mask_pos{[0.0;DIM]}else{embed[chunk[i]as usize]}).collect();
                    let mut co=vec![0.0f32;nf];
                    for f in 0..nf{let mut v1=b1[f];let mut v2=b2[f];
                        for ki in 0..k{let pos=mask_pos as i32+ki as i32-hk;
                            if pos>=0&&(pos as usize)<ctx{for d in 0..DIM{
                                v1+=w1[f][ki*DIM+d]*emb[pos as usize][d];v2+=w2[f][ki*DIM+d]*emb[pos as usize][d];}}}
                        co[f]=gate(v1,v2,&lp[f]).max(-10.0).min(10.0);}
                    let mut logits=hb.clone();for c in 0..27{for f in 0..nf{logits[c]+=hw[c][f]*co[f];}}
                    let pred=logits.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|v|v.0).unwrap_or(0);
                    if pred==chunk[mask_pos]as usize{ok+=1;}tot+=1;}
                if tot==0{0.0}else{ok as f64/tot as f64*100.0}};
            let te=eval(split,corpus.len());if te>best_test{best_test=te;}
            print!(" {:.1}",te);
        }
    }
    println!("  | {:.1}%",best_test);best_test
}

fn main(){
    let t0=Instant::now();
    let corpus=load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let split=corpus.len()*80/100;

    println!("=== ALL OPERATIONS SWEEP ===\n");

    // Baselines
    run(&corpus,split,"swish(a) [1-input baseline]",
        &|a,_,_|a/(1.0+(-a).exp()), 0, &[]);

    run(&corpus,split,"Beukers a×b/(1+|ab|)",
        &|a,b,_|{let p=a*b;p/(1.0+p.abs())}, 0, &[]);

    // Pure operations (2-input)
    run(&corpus,split,"square: a²/(1+a²)",
        &|a,_,_|a*a/(1.0+a*a), 0, &[]);

    run(&corpus,split,"abs: |a|×sign(b)",
        &|a,b,_|a.abs()*b.signum(), 0, &[]);

    run(&corpus,split,"log: sign(a)×ln(|a|+1)",
        &|a,_,_|a.signum()*(a.abs()+1.0).ln(), 0, &[]);

    run(&corpus,split,"exp: a×exp(-b²)",
        &|a,b,_|a*(-b*b).exp(), 0, &[]);

    run(&corpus,split,"power: sign(ab)×|ab|^α",
        &|a,b,p|{let ab=a*b;ab.signum()*(ab.abs()+0.01).powf(p[0].max(0.1).min(3.0))/(1.0+(ab.abs()+0.01).powf(p[0]))},
        1, &[0.5]);

    // Combinations
    run(&corpus,split,"Beukers + log: ab/(1+|ab|)+ln",
        &|a,b,_|{let p=a*b;p/(1.0+p.abs())+a.signum()*(a.abs()+1.0).ln()*0.3}, 0, &[]);

    run(&corpus,split,"Beukers + square: ab/(1+|ab|)+a²",
        &|a,b,_|{let p=a*b;p/(1.0+p.abs())+a*a/(1.0+a*a)*0.3}, 0, &[]);

    run(&corpus,split,"Beukers × exp: ab×exp(-|ab|)",
        &|a,b,_|{let p=a*b;p*(-p.abs()*0.1).exp()}, 0, &[]);

    println!("\n  Total time: {:.1}s",t0.elapsed().as_secs_f64());
}
