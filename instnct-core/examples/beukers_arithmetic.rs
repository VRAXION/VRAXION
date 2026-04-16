//! Complete arithmetic neuron sweep
//!
//! What operations should a neuron have?
//! 1. Beukers (multiply):  (W₁·x) × (W₂·x) / (1+|prod|)
//! 2. Division gate:       (W₁·x) / (|W₂·x| + ε)
//! 3. Full arithmetic:     (W₁·x) × (W₂·x) / (|W₃·x| + ε)
//! 4. Ratio + product:     (W₁·x)² / (|W₁·x × W₂·x| + ε)  (self-ratio)
//! 5. Beukers baseline for comparison
//!
//! Run: cargo run --example beukers_arithmetic --release

use std::time::Instant;
const VOCAB:usize=2000;const DIM:usize=16;

struct Rng(u64);
impl Rng{fn new(s:u64)->Self{Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1))}
fn next(&mut self)->u64{self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);self.0}
fn normal(&mut self)->f32{let u1=(((self.next()>>33)%65536)as f32/65536.0).max(1e-7);let u2=((self.next()>>33)%65536)as f32/65536.0;(-2.0*u1.ln()).sqrt()*(2.0*std::f32::consts::PI*u2).cos()}
fn range(&mut self,lo:usize,hi:usize)->usize{if hi<=lo{lo}else{lo+(self.next()as usize%(hi-lo))}}}
fn load_corpus(p:&str)->Vec<u8>{std::fs::read(p).expect("r").iter().filter_map(|&b|match b{
    b'a'..=b'z'=>Some(b-b'a'),b'A'..=b'Z'=>Some(b-b'A'),b' '|b'\n'|b'\t'|b'\r'=>Some(26),_=>None}).collect()}

fn run(corpus:&[u8],split:usize,name:&str,gate_type:usize,n_proj:usize,nf:usize)->f64{
    let ctx=32;let mask_pos=ctx/2;let k=7;let hk=3i32;let fan=k*DIM;
    let mut rng=Rng::new(42);
    let sc_e=(1.0/DIM as f32).sqrt();let sc_c=(2.0/fan as f32).sqrt();let sc_h=(2.0/nf as f32).sqrt();
    let mut embed:Vec<[f32;DIM]>=(0..VOCAB).map(|_|{let mut v=[0.0;DIM];for d in 0..DIM{v[d]=rng.normal()*sc_e;}v}).collect();
    let mut ws:Vec<Vec<Vec<f32>>>=(0..n_proj).map(|_|(0..nf).map(|_|(0..fan).map(|_|rng.normal()*sc_c).collect()).collect()).collect();
    let mut bs:Vec<Vec<f32>>=(0..n_proj).map(|_|vec![0.0f32;nf]).collect();
    let mut hw:Vec<Vec<f32>>=(0..27).map(|_|(0..nf).map(|_|rng.normal()*sc_h).collect()).collect();
    let mut hb=vec![0.0f32;27];
    let eps_div=0.5f32;

    let total_p=n_proj*(nf*fan+nf)+27*nf+27;
    print!("  {:>30} {:>1}proj p={:>5}:",name,n_proj,total_p);

    let samples=2000;let max_ep=300;let mut best_test=0.0f64;

    for ep in 0..max_ep{
        let lr=0.01*(1.0-ep as f32/max_ep as f32*0.8);
        let mut rt=Rng::new(ep as u64*1000+42);
        for _ in 0..samples{
            let off=rt.range(0,split.saturating_sub(ctx+1));
            let chunk=&corpus[off..off+ctx];let target=chunk[mask_pos]as usize;
            let emb:Vec<[f32;DIM]>=(0..ctx).map(|i|if i==mask_pos{[0.0;DIM]}else{embed[chunk[i]as usize]}).collect();

            let mut projs=vec![vec![0.0f32;nf];n_proj];
            for p in 0..n_proj{for f in 0..nf{projs[p][f]=bs[p][f];
                for ki in 0..k{let pos=mask_pos as i32+ki as i32-hk;
                    if pos>=0&&(pos as usize)<ctx{for d in 0..DIM{
                        projs[p][f]+=ws[p][f][ki*DIM+d]*emb[pos as usize][d];}}}}}

            let mut co=vec![0.0f32;nf];
            for f in 0..nf{
                co[f]=match gate_type{
                    0=>{// Beukers: a×b/(1+|a×b|)
                        let prod=projs[0][f]*projs[1][f];prod/(1.0+prod.abs())},
                    1=>{// Division: a/(|b|+ε)
                        projs[0][f]/(projs[1][f].abs()+eps_div)},
                    2=>{// Full arithmetic: a×b/(|c|+ε)
                        let prod=projs[0][f]*projs[1][f];
                        prod/(projs[2][f].abs()+eps_div)},
                    3=>{// Product + soft division: a×b/(1+|a×b|) / (|c|+ε) — nah too complex
                        // Instead: (a×b + a/|b+ε|) / 2 — mix of multiply and divide
                        let mult=projs[0][f]*projs[1][f]/(1.0+projs[0][f].abs()*projs[1][f].abs());
                        let divs=projs[0][f]/(projs[1][f].abs()+eps_div);
                        0.5*mult+0.5*divs.max(-5.0).min(5.0)},
                    4=>{// Soft ratio: a/(|a|+|b|+ε) — "what fraction of total is a?"
                        projs[0][f]/(projs[0][f].abs()+projs[1][f].abs()+eps_div)},
                    _=>projs[0][f]
                };
                co[f]=co[f].max(-10.0).min(10.0); // safety clamp
            }

            let mut logits=hb.clone();
            for c in 0..27{for f in 0..nf{logits[c]+=hw[c][f]*co[f];}}
            let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
            let mut pr=vec![0.0f32;27];let mut s=0.0f32;
            for c in 0..27{pr[c]=(logits[c]-mx).exp();s+=pr[c];}for c in 0..27{pr[c]/=s;}
            pr[target]-=1.0;

            let mut dc=vec![0.0f32;nf];
            for c in 0..27{for f in 0..nf{dc[f]+=pr[c]*hw[c][f];hw[c][f]-=lr*pr[c]*co[f];}hb[c]-=lr*pr[c];}

            // Numerical gradient for all gate types (simple, works for all)
            for f in 0..nf{
                for p in 0..n_proj{
                    let old=projs[p][f];
                    let eps_g=0.01;
                    // +eps
                    projs[p][f]=old+eps_g;
                    let co_plus=match gate_type{
                        0=>{let prod=projs[0][f]*projs[1][f];prod/(1.0+prod.abs())},
                        1=>projs[0][f]/(projs[1][f].abs()+eps_div),
                        2=>{let prod=projs[0][f]*projs[1][f];prod/(projs[2][f].abs()+eps_div)},
                        3=>{let m=projs[0][f]*projs[1][f]/(1.0+(projs[0][f]*projs[1][f]).abs());
                            let d=projs[0][f]/(projs[1][f].abs()+eps_div);0.5*m+0.5*d.max(-5.0).min(5.0)},
                        4=>projs[0][f]/(projs[0][f].abs()+projs[1][f].abs()+eps_div),
                        _=>projs[0][f]};
                    // -eps
                    projs[p][f]=old-eps_g;
                    let co_minus=match gate_type{
                        0=>{let prod=projs[0][f]*projs[1][f];prod/(1.0+prod.abs())},
                        1=>projs[0][f]/(projs[1][f].abs()+eps_div),
                        2=>{let prod=projs[0][f]*projs[1][f];prod/(projs[2][f].abs()+eps_div)},
                        3=>{let m=projs[0][f]*projs[1][f]/(1.0+(projs[0][f]*projs[1][f]).abs());
                            let d=projs[0][f]/(projs[1][f].abs()+eps_div);0.5*m+0.5*d.max(-5.0).min(5.0)},
                        4=>projs[0][f]/(projs[0][f].abs()+projs[1][f].abs()+eps_div),
                        _=>projs[0][f]};
                    projs[p][f]=old;
                    let grad=dc[f]*(co_plus-co_minus)/(2.0*eps_g);

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
                    for f in 0..nf{let mut pv=vec![0.0f32;n_proj];
                        for p in 0..n_proj{pv[p]=bs[p][f];
                            for ki in 0..k{let pos=mask_pos as i32+ki as i32-hk;
                                if pos>=0&&(pos as usize)<ctx{for d in 0..DIM{pv[p]+=ws[p][f][ki*DIM+d]*emb[pos as usize][d];}}}}
                        co[f]=match gate_type{
                            0=>{let prod=pv[0]*pv[1];prod/(1.0+prod.abs())},
                            1=>pv[0]/(pv[1].abs()+eps_div),
                            2=>{let prod=pv[0]*pv[1];prod/(pv[2].abs()+eps_div)},
                            3=>{let m=pv[0]*pv[1]/(1.0+(pv[0]*pv[1]).abs());
                                let d=pv[0]/(pv[1].abs()+eps_div);0.5*m+0.5*d.max(-5.0).min(5.0)},
                            4=>pv[0]/(pv[0].abs()+pv[1].abs()+eps_div),_=>pv[0]};
                        co[f]=co[f].max(-10.0).min(10.0);}
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
    let nf=32;

    println!("=== ARITHMETIC NEURON SWEEP ===\n");
    println!("  What operations should a neuron have?\n");

    run(&corpus,split,"Beukers (a×b/norm)",0,2,nf);
    run(&corpus,split,"Division (a/|b|+ε)",1,2,nf);
    run(&corpus,split,"Full arith (a×b/|c|+ε)",2,3,nf);
    run(&corpus,split,"Mix (multiply+divide)/2",3,2,nf);
    run(&corpus,split,"Soft ratio a/(|a|+|b|+ε)",4,2,nf);

    println!("\n  Total time: {:.1}s",t0.elapsed().as_secs_f64());
}
