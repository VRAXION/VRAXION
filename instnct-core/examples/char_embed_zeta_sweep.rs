//! Character Embedding — zeta/bessel inspired activation sweep
//!
//! Candidate activations inspired by number theory + signal processing:
//! 1. swish (baseline, 77.4%)
//! 2. C19 learnable (baseline, 77.1%)
//! 3. Dirichlet eta — alternating zeta, multi-harmonic decay
//! 4. Damped C19 — C19 with gaussian decay (like zeta's amplitude decay)
//! 5. Sinc — sin(x)/x, classic damped oscillation
//! 6. Bessel-like — cos with sqrt decay (simplified J₀)
//! 7. Multi-scale C19 — two frequencies (like zeta's multi-scale structure)
//!
//! Run: cargo run --example char_embed_zeta_sweep --release

use std::time::Instant;

const VOCAB: usize = 2000;
const DIM: usize = 16;

// ═══ Activation Functions ═══

// 1. Swish (baseline)
fn swish(x:f32)->f32{x/(1.0+(-x).exp())}
fn swish_d(x:f32)->f32{let s=1.0/(1.0+(-x).exp());s+x*s*(1.0-s)}

// 2. C19 with learnable c, rho
fn c19(x:f32,c:f32,rho:f32)->f32{let c=c.max(0.1);let rho=rho.max(0.0);let l=6.0*c;
    if x>=l{return x-l;}if x<=-l{return x+l;}
    let s=x/c;let n=s.floor();let t=s-n;let h=t*(1.0-t);let sg=if(n as i32)%2==0{1.0}else{-1.0};c*(sg*h+rho*h*h)}

// 3. Dirichlet eta (alternating zeta) — η(|x|+1) * sign(x)
//    η(s) = 1 - 1/2^s + 1/3^s - 1/4^s + 1/5^s
fn eta(x:f32,c:f32)->f32{
    let s=(x.abs()/c.max(0.1))+1.0; // shift so s>1 always
    let mut sum=0.0f32;
    for n in 1..=8u32{
        let sign=if n%2==1{1.0}else{-1.0};
        sum+=sign/(n as f32).powf(s);
    }
    sum*c*x.signum()
}

// 4. Damped C19 — C19 × gaussian envelope (decaying oscillations like zeta)
fn damped_c19(x:f32,c:f32,rho:f32)->f32{
    let raw=c19(x,c,rho);
    let decay=(-x*x/(36.0*c*c).max(1.0)).exp();
    raw*decay
}

// 5. Sinc-like — sin(πx/c)/(πx/c+ε), damped oscillation
fn sinc(x:f32,c:f32)->f32{
    let c=c.max(0.1);
    let t=std::f32::consts::PI*x/c;
    if t.abs()<0.001{c}else{c*t.sin()/(t.abs()+0.1)}
}

// 6. Bessel-like — cos(x/c - π/4) / sqrt(|x/c|+1), simplified J₀ envelope
fn bessel(x:f32,c:f32)->f32{
    let c=c.max(0.1);
    let t=x/c;
    c*(t-std::f32::consts::FRAC_PI_4).cos()/(t.abs()+1.0).sqrt()
}

// 7. Multi-scale C19 — two frequencies, like zeta's multi-harmonic structure
fn multi_c19(x:f32,c1:f32,c2:f32,rho:f32)->f32{
    c19(x,c1,rho)+0.5*c19(x,c2,0.0)
}

// Numerical gradient (finite differences, works for any function)
fn num_grad(f: &dyn Fn(f32)->f32, x:f32) -> f32 {
    let eps=0.001;
    (f(x+eps)-f(x-eps))/(2.0*eps)
}

// ═══ Infrastructure ═══

struct Rng(u64);
impl Rng{fn new(s:u64)->Self{Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1))}
fn next(&mut self)->u64{self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);self.0}
fn normal(&mut self)->f32{let u1=(((self.next()>>33)%65536)as f32/65536.0).max(1e-7);let u2=((self.next()>>33)%65536)as f32/65536.0;(-2.0*u1.ln()).sqrt()*(2.0*std::f32::consts::PI*u2).cos()}
fn range(&mut self,lo:usize,hi:usize)->usize{if hi<=lo{lo}else{lo+(self.next()as usize%(hi-lo))}}}

fn load_corpus(p:&str)->Vec<u8>{std::fs::read(p).expect("r").iter().filter_map(|&b|match b{
    b'a'..=b'z'=>Some(b-b'a'),b'A'..=b'Z'=>Some(b-b'A'),b' '|b'\n'|b'\t'|b'\r'=>Some(26),_=>None}).collect()}

fn run_test(corpus:&[u8], split:usize, name:&str,
    act_fn: &dyn Fn(f32,&[f32])->f32,  // activation(pre_act, params) -> output
    n_params: usize,                      // learnable params per neuron
    init_params: &[f32],                  // initial param values
) -> (f64, f32, f32) {
    let ctx=32; let mask_pos=ctx/2; let nf=64;
    let mut rng=Rng::new(42);
    let sc_e=(1.0/DIM as f32).sqrt();let sc_c=(2.0/(5*DIM)as f32).sqrt();let sc_h=(2.0/nf as f32).sqrt();

    let mut embed:Vec<[f32;DIM]>=(0..VOCAB).map(|_|{let mut v=[0.0;DIM];for d in 0..DIM{v[d]=rng.normal()*sc_e;}v}).collect();
    let mut conv_w:Vec<Vec<f32>>=(0..nf).map(|_|(0..5*DIM).map(|_|rng.normal()*sc_c).collect()).collect();
    let mut conv_b=vec![0.0f32;nf];
    let mut act_params:Vec<Vec<f32>>=(0..nf).map(|_|init_params.to_vec()).collect();
    let mut head_w:Vec<Vec<f32>>=(0..27).map(|_|(0..nf).map(|_|rng.normal()*sc_h).collect()).collect();
    let mut head_b=vec![0.0f32;27];

    let samples=5000; let max_ep=1500;
    let mut best_test=0.0f64;

    print!("  {:>12}:", name);
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
                conv_pre[f]=v;conv_out[f]=act_fn(v,&act_params[f]);}

            let mut logits=head_b.clone();
            for c in 0..27{for f in 0..nf{logits[c]+=head_w[c][f]*conv_out[f];}}
            let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
            let mut p=vec![0.0f32;27];let mut s=0.0f32;
            for c in 0..27{p[c]=(logits[c]-mx).exp();s+=p[c];}for c in 0..27{p[c]/=s;}
            p[target]-=1.0;

            let mut d_conv=vec![0.0f32;nf];
            for c in 0..27{for f in 0..nf{d_conv[f]+=p[c]*head_w[c][f];head_w[c][f]-=lr*p[c]*conv_out[f];}head_b[c]-=lr*p[c];}

            for f in 0..nf{
                // Numerical gradient for activation
                let pre=conv_pre[f];
                let params_f=act_params[f].clone();
                let act_grad=num_grad(&|x|act_fn(x,&params_f), pre);
                let g=d_conv[f]*act_grad;

                // Update conv weights
                for ki in 0..5{let pos=mask_pos as i32+ki as i32-2;
                    if pos>=0&&(pos as usize)<ctx{let pi=pos as usize;
                        for d in 0..DIM{
                            if pi!=mask_pos{embed[chunk[pi]as usize][d]-=lr*g*conv_w[f][ki*DIM+d]*0.5;}
                            conv_w[f][ki*DIM+d]-=lr*g*emb_out[pi][d];}}}
                conv_b[f]-=lr*g;

                // Update activation params (numerical gradient per param)
                for pi in 0..n_params{
                    let mut p_plus=params_f.clone(); p_plus[pi]+=0.01;
                    let mut p_minus=params_f.clone(); p_minus[pi]-=0.01;
                    let pg=(act_fn(pre,&p_plus)-act_fn(pre,&p_minus))/0.02;
                    act_params[f][pi]-=lr*d_conv[f]*pg*0.1;
                    // Clamp
                    if pi==0{act_params[f][pi]=act_params[f][pi].max(0.5).min(50.0);}
                    if pi==1{act_params[f][pi]=act_params[f][pi].max(0.0).min(5.0);}
                    if pi==2{act_params[f][pi]=act_params[f][pi].max(0.5).min(50.0);}
                }
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
                        conv[f]=act_fn(v,&act_params[f]);}
                    let mut logits=head_b.clone();for c in 0..27{for f in 0..nf{logits[c]+=head_w[c][f]*conv[f];}}
                    let pred=logits.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|v|v.0).unwrap_or(0);
                    if pred==chunk[mask_pos]as usize{ok+=1;}tot+=1;}
                if tot==0{0.0}else{ok as f64/tot as f64*100.0}};
            let te=eval(split,corpus.len());
            if te>best_test{best_test=te;}
            print!(" {:.1}", te);
        }
    }

    // V↔V vs V↔C structure
    let vowels=[0,4,8,14,20];
    let dist=|a:usize,b:usize|->f32{(0..DIM).map(|d|(embed[a][d]-embed[b][d]).powi(2)).sum::<f32>().sqrt()};
    let mut vv=0.0f32;let mut vn=0u32;let mut vc=0.0f32;let mut vcn=0u32;
    for &i in &vowels{for &j in &vowels{if i<j{vv+=dist(i,j);vn+=1;}}}
    for &i in &vowels{for j in 0..27{if!vowels.contains(&j)&&j!=26{vc+=dist(i,j);vcn+=1;}}}
    let vv_avg=if vn>0{vv/vn as f32}else{0.0};
    let vc_avg=if vcn>0{vc/vcn as f32}else{0.0};

    println!("  | {:.1}% VV/VC={:.2}", best_test, vv_avg/(vc_avg+0.001));
    (best_test, vv_avg, vc_avg)
}

fn main(){
    let t0=Instant::now();
    let corpus=load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let split=corpus.len()*80/100;

    println!("=== ZETA-INSPIRED ACTIVATION SWEEP (D={}) ===\n", DIM);
    println!("  ctx=32, mask=center, 1500 epochs, curve every 150 ep\n");

    // 1. Swish (baseline, no learnable params)
    run_test(&corpus, split, "swish",
        &|x,_| swish(x), 0, &[]);

    // 2. C19 learnable (c, rho)
    run_test(&corpus, split, "c19",
        &|x,p| c19(x, p[0], p[1]), 2, &[10.0, 0.5]);

    // 3. Dirichlet eta (learnable c)
    run_test(&corpus, split, "eta",
        &|x,p| eta(x, p[0]), 1, &[5.0]);

    // 4. Damped C19 (c, rho — with gaussian decay)
    run_test(&corpus, split, "damped_c19",
        &|x,p| damped_c19(x, p[0], p[1]), 2, &[10.0, 0.5]);

    // 5. Sinc (learnable c)
    run_test(&corpus, split, "sinc",
        &|x,p| sinc(x, p[0]), 1, &[5.0]);

    // 6. Bessel-like (learnable c)
    run_test(&corpus, split, "bessel",
        &|x,p| bessel(x, p[0]), 1, &[5.0]);

    // 7. Multi-scale C19 (c1, rho, c2)
    run_test(&corpus, split, "multi_c19",
        &|x,p| multi_c19(x, p[0], p[2], p[1]), 3, &[10.0, 0.5, 3.0]);

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
