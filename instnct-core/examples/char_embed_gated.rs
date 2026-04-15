//! Character Embedding — SwiGLU / Beukers gated activation
//!
//! 2-input gated activations (like GPT-4/LLaMA use):
//! Standard: output = act(W·x)
//! SwiGLU:   output = (W₁·x) ⊙ swish(W₂·x)    ← TWO projections!
//! Beukers:  output = (W₁·x) ⊙ (W₂·x) / (1+|W₁·x ⊙ W₂·x|)
//!
//! Same char embedding training, but conv layer uses gated activation
//!
//! Run: cargo run --example char_embed_gated --release

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

fn run_gated(corpus:&[u8], split:usize, name:&str, gate_type:usize) -> f64 {
    let ctx=32; let mask_pos=ctx/2; let nf=32; // half filters since 2x projection
    let k=5; let fan=k*DIM;
    let mut rng=Rng::new(42);
    let sc_e=(1.0/DIM as f32).sqrt();let sc_c=(2.0/fan as f32).sqrt();let sc_h=(2.0/nf as f32).sqrt();

    let mut embed:Vec<[f32;DIM]>=(0..VOCAB).map(|_|{let mut v=[0.0;DIM];for d in 0..DIM{v[d]=rng.normal()*sc_e;}v}).collect();
    // TWO sets of conv weights (the key difference!)
    let mut conv_w1:Vec<Vec<f32>>=(0..nf).map(|_|(0..fan).map(|_|rng.normal()*sc_c).collect()).collect();
    let mut conv_b1=vec![0.0f32;nf];
    let mut conv_w2:Vec<Vec<f32>>=(0..nf).map(|_|(0..fan).map(|_|rng.normal()*sc_c).collect()).collect();
    let mut conv_b2=vec![0.0f32;nf];
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

            // Two parallel convolutions
            let mut pre1=vec![0.0f32;nf]; let mut pre2=vec![0.0f32;nf];
            for f in 0..nf{
                let mut v1=conv_b1[f]; let mut v2=conv_b2[f];
                for ki in 0..k{let pos=mask_pos as i32+ki as i32-2;
                    if pos>=0&&(pos as usize)<ctx{for d in 0..DIM{
                        v1+=conv_w1[f][ki*DIM+d]*emb_out[pos as usize][d];
                        v2+=conv_w2[f][ki*DIM+d]*emb_out[pos as usize][d];}}}
                pre1[f]=v1; pre2[f]=v2;
            }

            // Gated combination
            let mut conv_out=vec![0.0f32;nf];
            for f in 0..nf {
                conv_out[f] = match gate_type {
                    0 => swish(pre1[f]), // baseline: just swish on W1 (ignore W2)
                    1 => pre1[f] * swish(pre2[f]), // SwiGLU
                    2 => { // Beukers gate
                        let prod = pre1[f] * pre2[f];
                        prod / (1.0 + prod.abs())
                    },
                    3 => pre1[f] * (1.0/(1.0+(-pre2[f]).exp())), // GLU (sigmoid gate)
                    _ => swish(pre1[f]),
                };
            }

            // Head
            let mut logits=head_b.clone();
            for c in 0..27{for f in 0..nf{logits[c]+=head_w[c][f]*conv_out[f];}}
            let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
            let mut p=vec![0.0f32;27];let mut s=0.0f32;
            for c in 0..27{p[c]=(logits[c]-mx).exp();s+=p[c];}for c in 0..27{p[c]/=s;}
            p[target]-=1.0;

            // Backprop head
            let mut d_conv=vec![0.0f32;nf];
            for c in 0..27{for f in 0..nf{d_conv[f]+=p[c]*head_w[c][f];head_w[c][f]-=lr*p[c]*conv_out[f];}head_b[c]-=lr*p[c];}

            // Backprop through gate → two conv weight sets
            for f in 0..nf {
                let (d1, d2) = match gate_type {
                    0 => (d_conv[f]*swish_g(pre1[f]), 0.0), // baseline
                    1 => { // SwiGLU: d/d(pre1) = swish(pre2), d/d(pre2) = pre1 * swish'(pre2)
                        (d_conv[f]*swish(pre2[f]), d_conv[f]*pre1[f]*swish_g(pre2[f]))
                    },
                    2 => { // Beukers: d/d(pre1) and d/d(pre2)
                        let prod=pre1[f]*pre2[f]; let denom=(1.0+prod.abs()).powi(2);
                        let dp=1.0/denom; // simplified
                        (d_conv[f]*dp*pre2[f], d_conv[f]*dp*pre1[f])
                    },
                    3 => { // GLU
                        let sig=1.0/(1.0+(-pre2[f]).exp());
                        (d_conv[f]*sig, d_conv[f]*pre1[f]*sig*(1.0-sig))
                    },
                    _ => (d_conv[f]*swish_g(pre1[f]), 0.0),
                };

                // Update conv weights
                for ki in 0..k{let pos=mask_pos as i32+ki as i32-2;
                    if pos>=0&&(pos as usize)<ctx{let pi=pos as usize;
                        for d in 0..DIM{
                            if pi!=mask_pos{
                                embed[chunk[pi]as usize][d]-=lr*(d1+d2)*conv_w1[f][ki*DIM+d]*0.25;
                            }
                            conv_w1[f][ki*DIM+d]-=lr*d1*emb_out[pi][d];
                            conv_w2[f][ki*DIM+d]-=lr*d2*emb_out[pi][d];
                        }}}
                conv_b1[f]-=lr*d1;
                conv_b2[f]-=lr*d2;
            }
        }

        if ep%150==0{
            let eval=|start:usize,end:usize|->f64{let mut rng3=Rng::new(999);let mut ok=0usize;let mut tot=0usize;
                for _ in 0..1000{if end<start+ctx+1{break;}let off=rng3.range(start,end.saturating_sub(ctx+1));
                    let chunk=&corpus[off..off+ctx];
                    let emb:Vec<[f32;DIM]>=(0..ctx).map(|i|if i==mask_pos{[0.0;DIM]}else{embed[chunk[i]as usize]}).collect();
                    let mut conv=vec![0.0f32;nf];
                    for f in 0..nf{let mut v1=conv_b1[f];let mut v2=conv_b2[f];
                        for ki in 0..k{let pos=mask_pos as i32+ki as i32-2;
                            if pos>=0&&(pos as usize)<ctx{for d in 0..DIM{
                                v1+=conv_w1[f][ki*DIM+d]*emb[pos as usize][d];
                                v2+=conv_w2[f][ki*DIM+d]*emb[pos as usize][d];}}}
                        conv[f]=match gate_type{0=>swish(v1),1=>v1*swish(v2),
                            2=>{let p=v1*v2;p/(1.0+p.abs())},3=>v1*(1.0/(1.0+(-v2).exp())),_=>swish(v1)};}
                    let mut logits=head_b.clone();for c in 0..27{for f in 0..nf{logits[c]+=head_w[c][f]*conv[f];}}
                    let pred=logits.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|v|v.0).unwrap_or(0);
                    if pred==chunk[mask_pos]as usize{ok+=1;}tot+=1;}
                if tot==0{0.0}else{ok as f64/tot as f64*100.0}};
            let te=eval(split,corpus.len());if te>best_test{best_test=te;}
            print!(" {:.1}",te);
        }
    }
    // Structure
    let vowels=[0,4,8,14,20];
    let dist=|a:usize,b:usize|->f32{(0..DIM).map(|d|(embed[a][d]-embed[b][d]).powi(2)).sum::<f32>().sqrt()};
    let mut vv=0.0f32;let mut vn=0u32;let mut vc=0.0f32;let mut vcn=0u32;
    for &i in &vowels{for &j in &vowels{if i<j{vv+=dist(i,j);vn+=1;}}}
    for &i in &vowels{for j in 0..27{if!vowels.contains(&j)&&j!=26{vc+=dist(i,j);vcn+=1;}}}
    let r=if vn>0&&vcn>0{(vv/vn as f32)/((vc/vcn as f32)+0.001)}else{0.0};
    println!("  | {:.1}% VV/VC={:.2}", best_test, r);
    best_test
}

fn main(){
    let t0=Instant::now();
    let corpus=load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let split=corpus.len()*80/100;

    println!("=== GATED ACTIVATION SWEEP (2-input, SwiGLU-style) ===\n");
    println!("  D=16, ctx=32, nf=32 (2x proj = same param count as nf=64 single)\n");

    run_gated(&corpus,split,"swish(base)",0);   // baseline: just swish, ignore W2
    run_gated(&corpus,split,"SwiGLU",1);         // GPT-4/LLaMA style
    run_gated(&corpus,split,"Beukers",2);        // novel: xy/(1+|xy|)
    run_gated(&corpus,split,"GLU",3);            // classic sigmoid gate

    println!("\n  Compare: single-input swish nf=64 = 77.4%");
    println!("  Total time: {:.1}s",t0.elapsed().as_secs_f64());
}
