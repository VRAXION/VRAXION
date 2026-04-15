//! Character Embedding — novel activation sweep (zeta-research inspired)
//!
//! Candidates from number theory research:
//! 1. GCU: x*cos(x) — beats swish on classification (paper: arXiv 2108.12943)
//! 2. Log-prime: sin(x*ln(1+|x|))/(1+|x|)^α — novel, NOT in literature!
//! 3. Padé rational: (a0+a1*x+a2*x²)/(1+b1*x+b2*x²) — learnable rational
//! 4. Snake: x + sin²(ax)/a — periodic with learnable frequency
//! 5. Swish baseline for comparison
//!
//! Run: cargo run --example char_embed_novel_sweep --release

use std::time::Instant;

const VOCAB: usize = 2000;
const DIM: usize = 16;

fn num_grad(f: &dyn Fn(f32)->f32, x: f32) -> f32 {
    let eps=0.001; (f(x+eps)-f(x-eps))/(2.0*eps)
}

struct Rng(u64);
impl Rng{fn new(s:u64)->Self{Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1))}
fn next(&mut self)->u64{self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);self.0}
fn normal(&mut self)->f32{let u1=(((self.next()>>33)%65536)as f32/65536.0).max(1e-7);let u2=((self.next()>>33)%65536)as f32/65536.0;(-2.0*u1.ln()).sqrt()*(2.0*std::f32::consts::PI*u2).cos()}
fn range(&mut self,lo:usize,hi:usize)->usize{if hi<=lo{lo}else{lo+(self.next()as usize%(hi-lo))}}}

fn load_corpus(p:&str)->Vec<u8>{std::fs::read(p).expect("r").iter().filter_map(|&b|match b{
    b'a'..=b'z'=>Some(b-b'a'),b'A'..=b'Z'=>Some(b-b'A'),b' '|b'\n'|b'\t'|b'\r'=>Some(26),_=>None}).collect()}

fn run_test(corpus:&[u8], split:usize, name:&str,
    act_fn: &dyn Fn(f32,&[f32])->f32, n_params:usize, init:&[f32],
) -> (f64, f32, f32) {
    let ctx=32; let mask_pos=ctx/2; let nf=64;
    let mut rng=Rng::new(42);
    let sc_e=(1.0/DIM as f32).sqrt();let sc_c=(2.0/(5*DIM)as f32).sqrt();let sc_h=(2.0/nf as f32).sqrt();
    let mut embed:Vec<[f32;DIM]>=(0..VOCAB).map(|_|{let mut v=[0.0;DIM];for d in 0..DIM{v[d]=rng.normal()*sc_e;}v}).collect();
    let mut conv_w:Vec<Vec<f32>>=(0..nf).map(|_|(0..5*DIM).map(|_|rng.normal()*sc_c).collect()).collect();
    let mut conv_b=vec![0.0f32;nf];
    let mut act_p:Vec<Vec<f32>>=(0..nf).map(|_|init.to_vec()).collect();
    let mut head_w:Vec<Vec<f32>>=(0..27).map(|_|(0..nf).map(|_|rng.normal()*sc_h).collect()).collect();
    let mut head_b=vec![0.0f32;27];

    let samples=5000; let max_ep=1500; let mut best_test=0.0f64;
    print!("  {:>12}:", name);

    for ep in 0..max_ep {
        let lr=0.01*(1.0-ep as f32/max_ep as f32*0.8);
        let mut rt=Rng::new(ep as u64*1000+42);
        for _ in 0..samples {
            let off=rt.range(0,split.saturating_sub(ctx+1));
            let chunk=&corpus[off..off+ctx]; let target=chunk[mask_pos]as usize;
            let emb_out:Vec<[f32;DIM]>=(0..ctx).map(|i|if i==mask_pos{[0.0;DIM]}else{embed[chunk[i]as usize]}).collect();
            let mut conv_pre=vec![0.0f32;nf];let mut conv_out=vec![0.0f32;nf];
            for f in 0..nf{let mut v=conv_b[f];
                for ki in 0..5{let pos=mask_pos as i32+ki as i32-2;
                    if pos>=0&&(pos as usize)<ctx{for d in 0..DIM{v+=conv_w[f][ki*DIM+d]*emb_out[pos as usize][d];}}}
                conv_pre[f]=v;conv_out[f]=act_fn(v,&act_p[f]);}
            let mut logits=head_b.clone();
            for c in 0..27{for f in 0..nf{logits[c]+=head_w[c][f]*conv_out[f];}}
            let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
            let mut p=vec![0.0f32;27];let mut s=0.0f32;
            for c in 0..27{p[c]=(logits[c]-mx).exp();s+=p[c];}for c in 0..27{p[c]/=s;}
            p[target]-=1.0;
            let mut d_conv=vec![0.0f32;nf];
            for c in 0..27{for f in 0..nf{d_conv[f]+=p[c]*head_w[c][f];head_w[c][f]-=lr*p[c]*conv_out[f];}head_b[c]-=lr*p[c];}
            for f in 0..nf{
                let pre=conv_pre[f]; let pf=act_p[f].clone();
                let ag=num_grad(&|x|act_fn(x,&pf),pre);
                let g=d_conv[f]*ag;
                for ki in 0..5{let pos=mask_pos as i32+ki as i32-2;
                    if pos>=0&&(pos as usize)<ctx{let pi=pos as usize;
                        for d in 0..DIM{if pi!=mask_pos{embed[chunk[pi]as usize][d]-=lr*g*conv_w[f][ki*DIM+d]*0.5;}
                            conv_w[f][ki*DIM+d]-=lr*g*emb_out[pi][d];}}}
                conv_b[f]-=lr*g;
                for pi in 0..n_params{let mut pp=pf.clone();pp[pi]+=0.01;let mut pm=pf.clone();pm[pi]-=0.01;
                    let pg=(act_fn(pre,&pp)-act_fn(pre,&pm))/0.02;
                    act_p[f][pi]-=lr*d_conv[f]*pg*0.1;
                    act_p[f][pi]=act_p[f][pi].max(-50.0).min(50.0);}
            }
        }
        if ep%150==0{
            let eval=|start:usize,end:usize|->f64{let mut rng3=Rng::new(999);let mut ok=0usize;let mut tot=0usize;
                for _ in 0..1000{if end<start+ctx+1{break;}let off=rng3.range(start,end.saturating_sub(ctx+1));
                    let chunk=&corpus[off..off+ctx];
                    let emb:Vec<[f32;DIM]>=(0..ctx).map(|i|if i==mask_pos{[0.0;DIM]}else{embed[chunk[i]as usize]}).collect();
                    let mut conv=vec![0.0f32;nf];
                    for f in 0..nf{let mut v=conv_b[f];for ki in 0..5{let pos=mask_pos as i32+ki as i32-2;
                        if pos>=0&&(pos as usize)<ctx{for d in 0..DIM{v+=conv_w[f][ki*DIM+d]*emb[pos as usize][d];}}}
                        conv[f]=act_fn(v,&act_p[f]);}
                    let mut logits=head_b.clone();for c in 0..27{for f in 0..nf{logits[c]+=head_w[c][f]*conv[f];}}
                    let pred=logits.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|v|v.0).unwrap_or(0);
                    if pred==chunk[mask_pos]as usize{ok+=1;}tot+=1;}
                if tot==0{0.0}else{ok as f64/tot as f64*100.0}};
            let te=eval(split,corpus.len());if te>best_test{best_test=te;}
            print!(" {:.1}",te);
        }
    }
    let vowels=[0,4,8,14,20];
    let dist=|a:usize,b:usize|->f32{(0..DIM).map(|d|(embed[a][d]-embed[b][d]).powi(2)).sum::<f32>().sqrt()};
    let mut vv=0.0f32;let mut vn=0u32;let mut vc=0.0f32;let mut vcn=0u32;
    for &i in &vowels{for &j in &vowels{if i<j{vv+=dist(i,j);vn+=1;}}}
    for &i in &vowels{for j in 0..27{if!vowels.contains(&j)&&j!=26{vc+=dist(i,j);vcn+=1;}}}
    let vv_avg=if vn>0{vv/vn as f32}else{0.0};let vc_avg=if vcn>0{vc/vcn as f32}else{0.0};
    println!("  | {:.1}% VV/VC={:.2}",best_test,vv_avg/(vc_avg+0.001));
    (best_test,vv_avg,vc_avg)
}

fn main(){
    let t0=Instant::now();
    let corpus=load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let split=corpus.len()*80/100;

    println!("=== NOVEL ACTIVATION SWEEP (D={}) ===\n",DIM);
    println!("  Zeta-research inspired candidates vs swish baseline\n");

    // 1. Swish baseline
    run_test(&corpus,split,"swish",
        &|x,_|x/(1.0+(-x).exp()), 0, &[]);

    // 2. GCU: x*cos(x) — paper says beats swish on classification
    run_test(&corpus,split,"gcu",
        &|x,_|x*x.cos(), 0, &[]);

    // 3. Log-prime: sin(x*ln(1+|x|))/(1+|x|)^α — NOVEL, not in literature!
    run_test(&corpus,split,"log_prime",
        &|x,p|{let alpha=p[0].max(0.01);(x*(1.0+x.abs()).ln()).sin()/(1.0+x.abs()).powf(alpha)},
        1, &[0.3]);

    // 4. Padé rational: (a0+a1*x+a2*x²)/(1+b1*x+b2*x²)
    run_test(&corpus,split,"pade",
        &|x,p|{let num=p[0]+p[1]*x+p[2]*x*x;let den=1.0+p[3]*x.abs()+p[4]*x*x;num/(den.max(0.1))},
        5, &[0.0, 1.0, 0.1, 0.1, 0.01]);

    // 5. Snake: x + sin²(ax)/a
    run_test(&corpus,split,"snake",
        &|x,p|{let a=p[0].max(0.1);x+(a*x).sin().powi(2)/a},
        1, &[5.0]);

    // 6. GCU + learnable freq: x*cos(ax) — GCU with tunable frequency
    run_test(&corpus,split,"gcu_freq",
        &|x,p|{let a=p[0].max(0.1);x*(a*x).cos()},
        1, &[1.0]);

    // 7. Swish + periodic residual: swish(x) + c*sin(ax) — hybrid
    run_test(&corpus,split,"swish_sin",
        &|x,p|{let a=p[0].max(0.1);let c=p[1];x/(1.0+(-x).exp())+c*(a*x).sin()},
        2, &[3.0, 0.3]);

    println!("\n  Total time: {:.1}s",t0.elapsed().as_secs_f64());
}
