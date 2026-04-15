//! Character Embedding — activation function sweep
//!
//! D=16, ctx=32, masked char prediction
//! Sweep: swish, C19, tanh, lrelu
//! Convergence curve + structure analysis for each
//!
//! Run: cargo run --example char_embed_act_sweep --release

use std::time::Instant;

const VOCAB: usize = 2000;
const DIM: usize = 16;

fn swish(x:f32)->f32{x/(1.0+(-x).exp())}
fn swish_g(x:f32)->f32{let s=1.0/(1.0+(-x).exp());s+x*s*(1.0-s)}
fn c19(x:f32)->f32{let c=10.0;let l=60.0;if x>=l{return x-l;}if x<=-l{return x+l;}
    let s=x/c;let n=s.floor();let t=s-n;let h=t*(1.0-t);let sg=if(n as i32)%2==0{1.0}else{-1.0};c*sg*h}
fn c19_g(x:f32)->f32{let c=10.0;let l=60.0;if x>=l||x<=-l{return 1.0;}
    let s=x/c;let n=s.floor();let t=s-n;let sg=if(n as i32)%2==0{1.0}else{-1.0};sg*(1.0-2.0*t)}
fn c19_full(x:f32,c:f32,rho:f32)->f32{let c=c.max(0.1);let rho=rho.max(0.0);let l=6.0*c;
    if x>=l{return x-l;}if x<=-l{return x+l;}
    let s=x/c;let n=s.floor();let t=s-n;let h=t*(1.0-t);let sg=if(n as i32)%2==0{1.0}else{-1.0};c*(sg*h+rho*h*h)}
fn c19_g_full(x:f32,c:f32,rho:f32)->f32{let c=c.max(0.1);let rho=rho.max(0.0);let l=6.0*c;
    if x>=l||x<=-l{return 1.0;}
    let s=x/c;let n=s.floor();let t=s-n;let h=t*(1.0-t);let sg=if(n as i32)%2==0{1.0}else{-1.0};(sg+2.0*rho*h)*(1.0-2.0*t)}
fn tanh_g(y:f32)->f32{1.0-y*y}
fn lrelu(x:f32)->f32{if x>0.0{x}else{0.01*x}}
fn lrelu_g(x:f32)->f32{if x>0.0{1.0}else{0.01}}

fn act(name:&str,x:f32)->f32{match name{"swish"=>swish(x),"c19"=>c19(x),"tanh"=>x.tanh(),"lrelu"=>lrelu(x),_=>swish(x)}}
fn act_g(name:&str,x:f32,y:f32)->f32{match name{"swish"=>swish_g(x),"c19"=>c19_g(x),"tanh"=>tanh_g(y),"lrelu"=>lrelu_g(x),_=>swish_g(x)}}

struct Rng(u64);
impl Rng{fn new(s:u64)->Self{Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1))}
fn next(&mut self)->u64{self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);self.0}
fn normal(&mut self)->f32{let u1=(((self.next()>>33)%65536)as f32/65536.0).max(1e-7);let u2=((self.next()>>33)%65536)as f32/65536.0;(-2.0*u1.ln()).sqrt()*(2.0*std::f32::consts::PI*u2).cos()}
fn range(&mut self,lo:usize,hi:usize)->usize{if hi<=lo{lo}else{lo+(self.next()as usize%(hi-lo))}}}

fn load_corpus(p:&str)->Vec<u8>{std::fs::read(p).expect("r").iter().filter_map(|&b|match b{
    b'a'..=b'z'=>Some(b-b'a'),b'A'..=b'Z'=>Some(b-b'A'),b' '|b'\n'|b'\t'|b'\r'=>Some(26),_=>None}).collect()}

fn run_embedding(corpus:&[u8], split:usize, act_name:&str) -> (f64, f32, f32) {
    let ctx=32; let mask_pos=ctx/2; let nf=64;
    let mut rng=Rng::new(42);
    let sc_e=(1.0/DIM as f32).sqrt();let sc_c=(2.0/(5*DIM)as f32).sqrt();let sc_h=(2.0/nf as f32).sqrt();

    let mut embed:Vec<[f32;DIM]>=(0..VOCAB).map(|_|{let mut v=[0.0;DIM];for d in 0..DIM{v[d]=rng.normal()*sc_e;}v}).collect();
    let mut conv_w:Vec<Vec<f32>>=(0..nf).map(|_|(0..5*DIM).map(|_|rng.normal()*sc_c).collect()).collect();
    let mut conv_b=vec![0.0f32;nf];
    // Learnable C19 params per neuron
    let mut conv_c=vec![10.0f32;nf];
    let mut conv_rho=vec![0.5f32;nf];
    let mut head_w:Vec<Vec<f32>>=(0..27).map(|_|(0..nf).map(|_|rng.normal()*sc_h).collect()).collect();
    let mut head_b=vec![0.0f32;27];

    let samples=5000; let max_ep=1500;
    let mut best_test=0.0f64;

    print!("  {:>6}:", act_name);
    for ep in 0..max_ep {
        let lr=0.01*(1.0-ep as f32/max_ep as f32*0.8);
        let mut rt=Rng::new(ep as u64*1000+42);
        for _ in 0..samples {
            let off=rt.range(0,split.saturating_sub(ctx+1));
            let chunk=&corpus[off..off+ctx];
            let target=chunk[mask_pos]as usize;

            let emb_out:Vec<[f32;DIM]>=(0..ctx).map(|i|if i==mask_pos{[0.0;DIM]}else{embed[chunk[i]as usize]}).collect();

            let mut conv_pre=vec![0.0f32;nf];let mut conv_out=vec![0.0f32;nf];
            for f in 0..nf{let mut v=conv_b[f];
                for ki in 0..5{let pos=mask_pos as i32+ki as i32-2;
                    if pos>=0&&(pos as usize)<ctx{for d in 0..DIM{v+=conv_w[f][ki*DIM+d]*emb_out[pos as usize][d];}}}
                conv_pre[f]=v;
                conv_out[f]=if act_name=="c19"{c19_full(v,conv_c[f],conv_rho[f])}else{act(act_name,v)};}

            let mut logits=head_b.clone();
            for c in 0..27{for f in 0..nf{logits[c]+=head_w[c][f]*conv_out[f];}}
            let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
            let mut p=vec![0.0f32;27];let mut s=0.0f32;
            for c in 0..27{p[c]=(logits[c]-mx).exp();s+=p[c];}for c in 0..27{p[c]/=s;}
            p[target]-=1.0;

            let mut d_conv=vec![0.0f32;nf];
            for c in 0..27{for f in 0..nf{d_conv[f]+=p[c]*head_w[c][f];head_w[c][f]-=lr*p[c]*conv_out[f];}head_b[c]-=lr*p[c];}

            for f in 0..nf{
                let g = if act_name=="c19" {
                    let ag=c19_g_full(conv_pre[f],conv_c[f],conv_rho[f]);
                    // Update c and rho
                    let eps=0.01;
                    let dc=(c19_full(conv_pre[f],conv_c[f]+eps,conv_rho[f])-c19_full(conv_pre[f],conv_c[f]-eps,conv_rho[f]))/(2.0*eps);
                    conv_c[f]-=lr*d_conv[f]*dc*0.1;conv_c[f]=conv_c[f].max(0.5).min(50.0);
                    let dr=(c19_full(conv_pre[f],conv_c[f],conv_rho[f]+eps)-c19_full(conv_pre[f],conv_c[f],conv_rho[f]-eps))/(2.0*eps);
                    conv_rho[f]-=lr*d_conv[f]*dr*0.1;conv_rho[f]=conv_rho[f].max(0.0).min(5.0);
                    d_conv[f]*ag
                } else { d_conv[f]*act_g(act_name,conv_pre[f],conv_out[f]) };
                let g=g; // rebind
                for ki in 0..5{let pos=mask_pos as i32+ki as i32-2;
                    if pos>=0&&(pos as usize)<ctx{let pi=pos as usize;
                        for d in 0..DIM{
                            if pi!=mask_pos{embed[chunk[pi]as usize][d]-=lr*g*conv_w[f][ki*DIM+d]*0.5;}
                            conv_w[f][ki*DIM+d]-=lr*g*emb_out[pi][d];}}}
                conv_b[f]-=lr*g;}
        }

        if ep%150==0{
            let eval=|start:usize,end:usize|->f64{let mut rng3=Rng::new(999);let mut ok=0usize;let mut tot=0usize;
                for _ in 0..1000{if end<start+ctx+1{break;}let off=rng3.range(start,end.saturating_sub(ctx+1));
                    let chunk=&corpus[off..off+ctx];
                    let emb:Vec<[f32;DIM]>=(0..ctx).map(|i|if i==mask_pos{[0.0;DIM]}else{embed[chunk[i]as usize]}).collect();
                    let mut conv=vec![0.0f32;nf];
                    for f in 0..nf{let mut v=conv_b[f];for ki in 0..5{let pos=mask_pos as i32+ki as i32-2;
                        if pos>=0&&(pos as usize)<ctx{for d in 0..DIM{v+=conv_w[f][ki*DIM+d]*emb[pos as usize][d];}}}
                    conv[f]=if act_name=="c19"{c19_full(v,conv_c[f],conv_rho[f])}else{act(act_name,v)};}
                    let mut logits=head_b.clone();for c in 0..27{for f in 0..nf{logits[c]+=head_w[c][f]*conv[f];}}
                    let pred=logits.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|v|v.0).unwrap_or(0);
                    if pred==chunk[mask_pos]as usize{ok+=1;}tot+=1;}
                if tot==0{0.0}else{ok as f64/tot as f64*100.0}};
            let te=eval(split,corpus.len());
            if te>best_test{best_test=te;}
            print!(" {:.1}", te);
        }
    }

    // Structure: V↔V vs V↔C
    let vowels=[0,4,8,14,20];
    let dist=|a:usize,b:usize|->f32{(0..DIM).map(|d|(embed[a][d]-embed[b][d]).powi(2)).sum::<f32>().sqrt()};
    let mut vv=0.0f32;let mut vn=0u32;let mut vc=0.0f32;let mut vcn=0u32;
    for &i in &vowels{for &j in &vowels{if i<j{vv+=dist(i,j);vn+=1;}}}
    for &i in &vowels{for j in 0..27{if!vowels.contains(&j)&&j!=26{vc+=dist(i,j);vcn+=1;}}}
    let vv_avg=if vn>0{vv/vn as f32}else{0.0};
    let vc_avg=if vcn>0{vc/vcn as f32}else{0.0};

    println!("  | best={:.1}% V↔V={:.2} V↔C={:.2} ratio={:.2}", best_test, vv_avg, vc_avg, vv_avg/vc_avg);
    (best_test, vv_avg, vc_avg)
}

fn main(){
    let t0=Instant::now();
    let corpus=load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let split=corpus.len()*80/100;

    println!("=== CHAR EMBEDDING — ACTIVATION SWEEP (D={}) ===\n",DIM);
    println!("  ctx=32, mask=center, 5K samples/ep, 1500 epochs\n");
    println!("  {:>6}: test% curve (every 150 ep) | summary","act");
    println!("  {}","-".repeat(75));

    for &a in &["swish","c19","tanh","lrelu"]{
        run_embedding(&corpus,split,a);
    }

    println!("\n  Total time: {:.1}s",t0.elapsed().as_secs_f64());
}
