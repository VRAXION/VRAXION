//! Beukers gate — fair comparison with swish nf=64
//!
//! Beukers nf=64 (2×64 proj) vs swish nf=64 (1×64 proj)
//! Beukers has 2× params but captures multiplicative interaction
//! Question: does Beukers beat swish 77.4% with same OR more params?
//!
//! Run: cargo run --example char_embed_beukers_full --release

use std::time::Instant;

const VOCAB: usize = 2000;
const DIM: usize = 16;

struct Rng(u64);
impl Rng{fn new(s:u64)->Self{Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1))}
fn next(&mut self)->u64{self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);self.0}
fn normal(&mut self)->f32{let u1=(((self.next()>>33)%65536)as f32/65536.0).max(1e-7);let u2=((self.next()>>33)%65536)as f32/65536.0;(-2.0*u1.ln()).sqrt()*(2.0*std::f32::consts::PI*u2).cos()}
fn range(&mut self,lo:usize,hi:usize)->usize{if hi<=lo{lo}else{lo+(self.next()as usize%(hi-lo))}}}

fn load_corpus(p:&str)->Vec<u8>{std::fs::read(p).expect("r").iter().filter_map(|&b|match b{
    b'a'..=b'z'=>Some(b-b'a'),b'A'..=b'Z'=>Some(b-b'A'),b' '|b'\n'|b'\t'|b'\r'=>Some(26),_=>None}).collect()}

fn swish(x:f32)->f32{x/(1.0+(-x).exp())}
fn swish_g(x:f32)->f32{let s=1.0/(1.0+(-x).exp());s+x*s*(1.0-s)}

fn run(corpus:&[u8], split:usize, name:&str, nf:usize, gate_type:usize) -> f64 {
    let ctx=32;let mask_pos=ctx/2;let k=5;let fan=k*DIM;
    let mut rng=Rng::new(42);
    let sc_e=(1.0/DIM as f32).sqrt();let sc_c=(2.0/fan as f32).sqrt();let sc_h=(2.0/nf as f32).sqrt();
    let mut embed:Vec<[f32;DIM]>=(0..VOCAB).map(|_|{let mut v=[0.0;DIM];for d in 0..DIM{v[d]=rng.normal()*sc_e;}v}).collect();
    let mut w1:Vec<Vec<f32>>=(0..nf).map(|_|(0..fan).map(|_|rng.normal()*sc_c).collect()).collect();
    let mut b1=vec![0.0f32;nf];
    let mut w2:Vec<Vec<f32>>=(0..nf).map(|_|(0..fan).map(|_|rng.normal()*sc_c).collect()).collect();
    let mut b2=vec![0.0f32;nf];
    let mut hw:Vec<Vec<f32>>=(0..27).map(|_|(0..nf).map(|_|rng.normal()*sc_h).collect()).collect();
    let mut hb=vec![0.0f32;27];

    let samples=5000;let max_ep=1500;let mut best_test=0.0f64;
    let total_params = if gate_type==0 { nf*fan+nf+27*nf+27 } else { 2*(nf*fan+nf)+27*nf+27 };
    print!("  {:>16} nf={:>3} params={:>6}:", name, nf, total_params);

    for ep in 0..max_ep{
        let lr=0.01*(1.0-ep as f32/max_ep as f32*0.8);
        let mut rt=Rng::new(ep as u64*1000+42);
        for _ in 0..samples{
            let off=rt.range(0,split.saturating_sub(ctx+1));
            let chunk=&corpus[off..off+ctx];let target=chunk[mask_pos]as usize;
            let emb:Vec<[f32;DIM]>=(0..ctx).map(|i|if i==mask_pos{[0.0;DIM]}else{embed[chunk[i]as usize]}).collect();
            let mut p1=vec![0.0f32;nf];let mut p2=vec![0.0f32;nf];
            for f in 0..nf{let mut v1=b1[f];let mut v2=b2[f];
                for ki in 0..k{let pos=mask_pos as i32+ki as i32-2;
                    if pos>=0&&(pos as usize)<ctx{for d in 0..DIM{
                        v1+=w1[f][ki*DIM+d]*emb[pos as usize][d];
                        v2+=w2[f][ki*DIM+d]*emb[pos as usize][d];}}}
                p1[f]=v1;p2[f]=v2;}
            let mut co=vec![0.0f32;nf];
            for f in 0..nf{co[f]=match gate_type{
                0=>swish(p1[f]),
                1=>{let prod=p1[f]*p2[f];prod/(1.0+prod.abs())}, // Beukers
                _=>swish(p1[f])};}
            let mut logits=hb.clone();
            for c in 0..27{for f in 0..nf{logits[c]+=hw[c][f]*co[f];}}
            let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
            let mut pr=vec![0.0f32;27];let mut s=0.0f32;
            for c in 0..27{pr[c]=(logits[c]-mx).exp();s+=pr[c];}for c in 0..27{pr[c]/=s;}
            pr[target]-=1.0;
            let mut dc=vec![0.0f32;nf];
            for c in 0..27{for f in 0..nf{dc[f]+=pr[c]*hw[c][f];hw[c][f]-=lr*pr[c]*co[f];}hb[c]-=lr*pr[c];}
            for f in 0..nf{
                let(d1,d2)=match gate_type{
                    0=>(dc[f]*swish_g(p1[f]),0.0),
                    1=>{let prod=p1[f]*p2[f];let den=(1.0+prod.abs()).powi(2);
                        (dc[f]*p2[f]/den,dc[f]*p1[f]/den)},
                    _=>(dc[f]*swish_g(p1[f]),0.0)};
                for ki in 0..k{let pos=mask_pos as i32+ki as i32-2;
                    if pos>=0&&(pos as usize)<ctx{let pi=pos as usize;
                        for d in 0..DIM{
                            if pi!=mask_pos{embed[chunk[pi]as usize][d]-=lr*(d1+d2)*w1[f][ki*DIM+d]*0.25;}
                            w1[f][ki*DIM+d]-=lr*d1*emb[pi][d];
                            if gate_type>0{w2[f][ki*DIM+d]-=lr*d2*emb[pi][d];}}}}
                b1[f]-=lr*d1;if gate_type>0{b2[f]-=lr*d2;}
            }
        }
        if ep%150==0{
            let eval=|start:usize,end:usize|->f64{let mut rng3=Rng::new(999);let mut ok=0usize;let mut tot=0usize;
                for _ in 0..1000{if end<start+ctx+1{break;}let off=rng3.range(start,end.saturating_sub(ctx+1));
                    let chunk=&corpus[off..off+ctx];
                    let emb:Vec<[f32;DIM]>=(0..ctx).map(|i|if i==mask_pos{[0.0;DIM]}else{embed[chunk[i]as usize]}).collect();
                    let mut co=vec![0.0f32;nf];
                    for f in 0..nf{let mut v1=b1[f];let mut v2=b2[f];
                        for ki in 0..k{let pos=mask_pos as i32+ki as i32-2;
                            if pos>=0&&(pos as usize)<ctx{for d in 0..DIM{
                                v1+=w1[f][ki*DIM+d]*emb[pos as usize][d];
                                v2+=w2[f][ki*DIM+d]*emb[pos as usize][d];}}}
                        co[f]=match gate_type{0=>swish(v1),1=>{let p=v1*v2;p/(1.0+p.abs())},_=>swish(v1)};}
                    let mut logits=hb.clone();for c in 0..27{for f in 0..nf{logits[c]+=hw[c][f]*co[f];}}
                    let pred=logits.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|v|v.0).unwrap_or(0);
                    if pred==chunk[mask_pos]as usize{ok+=1;}tot+=1;}
                if tot==0{0.0}else{ok as f64/tot as f64*100.0}};
            let te=eval(split,corpus.len());if te>best_test{best_test=te;}
            print!(" {:.1}",te);
        }
    }
    println!("  | best={:.1}%",best_test);
    best_test
}

fn main(){
    let t0=Instant::now();
    let corpus=load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let split=corpus.len()*80/100;

    println!("=== BEUKERS vs SWISH — FAIR COMPARISON ===\n");

    // Same filter count (nf=64), Beukers has 2x conv params
    run(&corpus,split,"swish",64,0);
    run(&corpus,split,"Beukers",64,1);

    // Same total params: Beukers nf=48 ≈ swish nf=64 in conv params
    run(&corpus,split,"swish",48,0);
    run(&corpus,split,"Beukers",48,1);

    // Bigger: does more params help?
    run(&corpus,split,"swish",96,0);
    run(&corpus,split,"Beukers",96,1);

    println!("\n  Total time: {:.1}s",t0.elapsed().as_secs_f64());
}
