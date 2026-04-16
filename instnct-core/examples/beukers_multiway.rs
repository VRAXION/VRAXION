//! Beukers multi-way gate sweep: 2-input vs 3-input vs 4-input
//!
//! Does higher-order multiplication capture more?
//! Also: different normalization strategies
//!
//! Run: cargo run --example beukers_multiway --release

use std::time::Instant;
const VOCAB:usize=2000;const DIM:usize=16;

struct Rng(u64);
impl Rng{fn new(s:u64)->Self{Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1))}
fn next(&mut self)->u64{self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);self.0}
fn normal(&mut self)->f32{let u1=(((self.next()>>33)%65536)as f32/65536.0).max(1e-7);let u2=((self.next()>>33)%65536)as f32/65536.0;(-2.0*u1.ln()).sqrt()*(2.0*std::f32::consts::PI*u2).cos()}
fn range(&mut self,lo:usize,hi:usize)->usize{if hi<=lo{lo}else{lo+(self.next()as usize%(hi-lo))}}}
fn load_corpus(p:&str)->Vec<u8>{std::fs::read(p).expect("r").iter().filter_map(|&b|match b{
    b'a'..=b'z'=>Some(b-b'a'),b'A'..=b'Z'=>Some(b-b'A'),b' '|b'\n'|b'\t'|b'\r'=>Some(26),_=>None}).collect()}

fn run(corpus:&[u8],split:usize,name:&str,n_proj:usize,norm_type:usize,nf:usize)->f64{
    let ctx=32;let mask_pos=ctx/2;let k=7;let hk=3i32;let fan=k*DIM;
    let mut rng=Rng::new(42);
    let sc_e=(1.0/DIM as f32).sqrt();let sc_c=(2.0/fan as f32).sqrt();let sc_h=(2.0/nf as f32).sqrt();
    let mut embed:Vec<[f32;DIM]>=(0..VOCAB).map(|_|{let mut v=[0.0;DIM];for d in 0..DIM{v[d]=rng.normal()*sc_e;}v}).collect();
    // N parallel projections
    let mut ws:Vec<Vec<Vec<f32>>>=(0..n_proj).map(|_|
        (0..nf).map(|_|(0..fan).map(|_|rng.normal()*sc_c).collect()).collect()).collect();
    let mut bs:Vec<Vec<f32>>=(0..n_proj).map(|_|vec![0.0f32;nf]).collect();
    let mut hw:Vec<Vec<f32>>=(0..27).map(|_|(0..nf).map(|_|rng.normal()*sc_h).collect()).collect();
    let mut hb=vec![0.0f32;27];

    let total_p=n_proj*(nf*fan+nf)+27*nf+27;
    print!("  {:>28} p={:>6}:",name,total_p);

    let samples=2000;let max_ep=300;let mut best_test=0.0f64;

    for ep in 0..max_ep{
        let lr=0.01*(1.0-ep as f32/max_ep as f32*0.8);
        let mut rt=Rng::new(ep as u64*1000+42);
        for _ in 0..samples{
            let off=rt.range(0,split.saturating_sub(ctx+1));
            let chunk=&corpus[off..off+ctx];let target=chunk[mask_pos]as usize;
            let emb:Vec<[f32;DIM]>=(0..ctx).map(|i|if i==mask_pos{[0.0;DIM]}else{embed[chunk[i]as usize]}).collect();

            // Compute N projections per filter
            let mut projs=vec![vec![0.0f32;nf];n_proj];
            for p in 0..n_proj{for f in 0..nf{projs[p][f]=bs[p][f];
                for ki in 0..k{let pos=mask_pos as i32+ki as i32-hk;
                    if pos>=0&&(pos as usize)<ctx{for d in 0..DIM{
                        projs[p][f]+=ws[p][f][ki*DIM+d]*emb[pos as usize][d];}}}}}

            // Multi-way gate
            let mut co=vec![0.0f32;nf];
            for f in 0..nf{
                let mut prod=1.0f32;
                for p in 0..n_proj{prod*=projs[p][f];}
                co[f]=match norm_type{
                    0=>prod/(1.0+prod.abs()),                              // standard
                    1=>{let s=prod.signum();s*prod.abs().powf(1.0/n_proj as f32)/(1.0+prod.abs().powf(1.0/n_proj as f32))}, // nth-root norm
                    2=>prod/(1.0+prod.abs().powf(1.0/n_proj as f32)),      // mixed
                    _=>prod/(1.0+prod.abs()),
                };
            }

            let mut logits=hb.clone();
            for c in 0..27{for f in 0..nf{logits[c]+=hw[c][f]*co[f];}}
            let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
            let mut pr=vec![0.0f32;27];let mut s=0.0f32;
            for c in 0..27{pr[c]=(logits[c]-mx).exp();s+=pr[c];}for c in 0..27{pr[c]/=s;}
            pr[target]-=1.0;

            let mut dc=vec![0.0f32;nf];
            for c in 0..27{for f in 0..nf{dc[f]+=pr[c]*hw[c][f];hw[c][f]-=lr*pr[c]*co[f];}hb[c]-=lr*pr[c];}

            // Backprop through gate (numerical for simplicity)
            for f in 0..nf{
                for p in 0..n_proj{
                    // Approximate gradient: d(gate)/d(proj_p)
                    let mut prod_others=1.0f32;
                    for q in 0..n_proj{if q!=p{prod_others*=projs[q][f];}}
                    let full_prod:f32=(0..n_proj).map(|q|projs[q][f]).product();
                    let den=(1.0+full_prod.abs()).powi(2);
                    let grad=dc[f]*prod_others/den;

                    for ki in 0..k{let pos=mask_pos as i32+ki as i32-hk;
                        if pos>=0&&(pos as usize)<ctx{let pi=pos as usize;
                            for d in 0..DIM{
                                if pi!=mask_pos{embed[chunk[pi]as usize][d]-=lr*grad*ws[p][f][ki*DIM+d]*0.1/n_proj as f32;}
                                ws[p][f][ki*DIM+d]-=lr*grad*emb[pi][d];}}}
                    bs[p][f]-=lr*grad;
                }
            }
        }
        if ep%50==0{
            let eval=|start:usize,end:usize|->f64{let mut rng3=Rng::new(999);let mut ok=0usize;let mut tot=0usize;
                for _ in 0..1000{if end<start+ctx+1{break;}let off=rng3.range(start,end.saturating_sub(ctx+1));
                    let chunk=&corpus[off..off+ctx];
                    let emb:Vec<[f32;DIM]>=(0..ctx).map(|i|if i==mask_pos{[0.0;DIM]}else{embed[chunk[i]as usize]}).collect();
                    let mut co=vec![0.0f32;nf];
                    for f in 0..nf{let mut prod=1.0f32;
                        for p in 0..n_proj{let mut v=bs[p][f];
                            for ki in 0..k{let pos=mask_pos as i32+ki as i32-hk;
                                if pos>=0&&(pos as usize)<ctx{for d in 0..DIM{v+=ws[p][f][ki*DIM+d]*emb[pos as usize][d];}}}
                            prod*=v;}
                        co[f]=match norm_type{0=>prod/(1.0+prod.abs()),
                            1=>{let s=prod.signum();s*prod.abs().powf(1.0/n_proj as f32)/(1.0+prod.abs().powf(1.0/n_proj as f32))},
                            2=>prod/(1.0+prod.abs().powf(1.0/n_proj as f32)),_=>prod/(1.0+prod.abs())};}
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
    let nf=32; // toy size for quick comparison

    println!("=== BEUKERS MULTI-WAY GATE SWEEP ===\n");
    println!("  k=7, nf={}, ctx=32, norm variants\n",nf);

    // 2-input (baseline)
    run(&corpus,split,"2-way xy/(1+|xy|)",2,0,nf);

    // 3-input variants
    run(&corpus,split,"3-way xyz/(1+|xyz|)",3,0,nf);
    run(&corpus,split,"3-way nth-root norm",3,1,nf);
    run(&corpus,split,"3-way mixed norm",3,2,nf);

    // 4-input
    run(&corpus,split,"4-way xyzw/(1+|xyzw|)",4,0,nf);
    run(&corpus,split,"4-way nth-root norm",4,1,nf);

    // skip param match for speed

    println!("\n  Total time: {:.1}s",t0.elapsed().as_secs_f64());
}
