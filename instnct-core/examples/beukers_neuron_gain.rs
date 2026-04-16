//! Beukers with per-neuron gain (α) — the neuron's OWN opinion
//!
//! Standard: act(W·x + bias)
//! With gain: act(α × (W·x) + bias)  — α = neuron sensitivity
//! Beukers + gain: α₁(W₁·x+b₁) × α₂(W₂·x+b₂) / norm
//!
//! Quick toy test: does per-neuron gain help?
//!
//! Run: cargo run --example beukers_neuron_gain --release

use std::time::Instant;
const VOCAB:usize=2000;const DIM:usize=16;

struct Rng(u64);
impl Rng{fn new(s:u64)->Self{Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1))}
fn next(&mut self)->u64{self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);self.0}
fn normal(&mut self)->f32{let u1=(((self.next()>>33)%65536)as f32/65536.0).max(1e-7);let u2=((self.next()>>33)%65536)as f32/65536.0;(-2.0*u1.ln()).sqrt()*(2.0*std::f32::consts::PI*u2).cos()}
fn range(&mut self,lo:usize,hi:usize)->usize{if hi<=lo{lo}else{lo+(self.next()as usize%(hi-lo))}}}
fn load_corpus(p:&str)->Vec<u8>{std::fs::read(p).expect("r").iter().filter_map(|&b|match b{
    b'a'..=b'z'=>Some(b-b'a'),b'A'..=b'Z'=>Some(b-b'A'),b' '|b'\n'|b'\t'|b'\r'=>Some(26),_=>None}).collect()}

fn run(corpus:&[u8],split:usize,name:&str,use_gain:bool,nf:usize)->f64{
    let ctx=32;let mask_pos=ctx/2;let k=7;let hk=3i32;let fan=k*DIM;
    let mut rng=Rng::new(42);
    let sc_e=(1.0/DIM as f32).sqrt();let sc_c=(2.0/fan as f32).sqrt();let sc_h=(2.0/nf as f32).sqrt();
    let mut embed:Vec<[f32;DIM]>=(0..VOCAB).map(|_|{let mut v=[0.0;DIM];for d in 0..DIM{v[d]=rng.normal()*sc_e;}v}).collect();
    let mut w1:Vec<Vec<f32>>=(0..nf).map(|_|(0..fan).map(|_|rng.normal()*sc_c).collect()).collect();
    let mut b1=vec![0.0f32;nf];
    let mut w2:Vec<Vec<f32>>=(0..nf).map(|_|(0..fan).map(|_|rng.normal()*sc_c).collect()).collect();
    let mut b2=vec![0.0f32;nf];
    // Per-neuron gain (α₁, α₂)
    let mut gain1=vec![1.0f32;nf];
    let mut gain2=vec![1.0f32;nf];
    let mut hw:Vec<Vec<f32>>=(0..27).map(|_|(0..nf).map(|_|rng.normal()*sc_h).collect()).collect();
    let mut hb=vec![0.0f32;27];

    let samples=2000;let max_ep=300;let mut best_test=0.0f64;
    print!("  {:>30}:",name);

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

            // Apply neuron gain
            let mut a=vec![0.0f32;nf];let mut b=vec![0.0f32;nf];
            for f in 0..nf{
                a[f]=if use_gain{gain1[f]*p1[f]}else{p1[f]};
                b[f]=if use_gain{gain2[f]*p2[f]}else{p2[f]};
            }

            let mut co=vec![0.0f32;nf];
            for f in 0..nf{let prod=a[f]*b[f];co[f]=prod/(1.0+prod.abs());}

            let mut logits=hb.clone();
            for c in 0..27{for f in 0..nf{logits[c]+=hw[c][f]*co[f];}}
            let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
            let mut pr=vec![0.0f32;27];let mut s=0.0f32;
            for c in 0..27{pr[c]=(logits[c]-mx).exp();s+=pr[c];}for c in 0..27{pr[c]/=s;}
            pr[target]-=1.0;

            let mut dc=vec![0.0f32;nf];
            for c in 0..27{for f in 0..nf{dc[f]+=pr[c]*hw[c][f];hw[c][f]-=lr*pr[c]*co[f];}hb[c]-=lr*pr[c];}

            for f in 0..nf{
                let prod=a[f]*b[f];let den=(1.0+prod.abs()).powi(2);
                let da=dc[f]*b[f]/den;
                let db=dc[f]*a[f]/den;

                // Gradient for gain
                if use_gain{
                    // d(output)/d(gain1) = d(output)/d(a) * d(a)/d(gain1) = da * p1
                    gain1[f]-=lr*da*p1[f]*0.01;
                    gain1[f]=gain1[f].max(0.01).min(10.0);
                    gain2[f]-=lr*db*p2[f]*0.01;
                    gain2[f]=gain2[f].max(0.01).min(10.0);
                }

                let d1=if use_gain{da*gain1[f]}else{da};
                let d2=if use_gain{db*gain2[f]}else{db};

                for ki in 0..k{let pos=mask_pos as i32+ki as i32-hk;
                    if pos>=0&&(pos as usize)<ctx{let pi=pos as usize;
                        for d in 0..DIM{
                            if pi!=mask_pos{embed[chunk[pi]as usize][d]-=lr*(d1+d2)*w1[f][ki*DIM+d]*0.25;}
                            w1[f][ki*DIM+d]-=lr*d1*emb[pi][d];
                            w2[f][ki*DIM+d]-=lr*d2*emb[pi][d];}}}
                b1[f]-=lr*d1;b2[f]-=lr*d2;
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
                                v1+=w1[f][ki*DIM+d]*emb[pos as usize][d];
                                v2+=w2[f][ki*DIM+d]*emb[pos as usize][d];}}}
                        let aa=if use_gain{gain1[f]*v1}else{v1};
                        let bb=if use_gain{gain2[f]*v2}else{v2};
                        let prod=aa*bb;co[f]=prod/(1.0+prod.abs());}
                    let mut logits=hb.clone();for c in 0..27{for f in 0..nf{logits[c]+=hw[c][f]*co[f];}}
                    let pred=logits.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|v|v.0).unwrap_or(0);
                    if pred==chunk[mask_pos]as usize{ok+=1;}tot+=1;}
                if tot==0{0.0}else{ok as f64/tot as f64*100.0}};
            let te=eval(split,corpus.len());if te>best_test{best_test=te;}
            print!(" {:.1}",te);
        }
    }

    // Show learned gains
    if use_gain{
        let avg_g1:f32=gain1.iter().sum::<f32>()/nf as f32;
        let avg_g2:f32=gain2.iter().sum::<f32>()/nf as f32;
        let min_g1=gain1.iter().cloned().fold(f32::MAX,f32::min);
        let max_g1=gain1.iter().cloned().fold(f32::MIN,f32::max);
        println!("  | {:.1}% gain1=[{:.2}..{:.2}] avg={:.2} gain2_avg={:.2}",
            best_test, min_g1, max_g1, avg_g1, avg_g2);
    } else {
        println!("  | {:.1}%", best_test);
    }
    best_test
}

fn main(){
    let t0=Instant::now();
    let corpus=load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let split=corpus.len()*80/100;

    println!("=== BEUKERS + NEURON GAIN (α) ===\n");
    println!("  Quick toy test: does per-neuron gain help?\n");

    // nf=32 toy
    run(&corpus,split,"Beukers nf=32 (no gain)",false,32);
    run(&corpus,split,"Beukers nf=32 + gain α",true,32);

    // nf=64
    run(&corpus,split,"Beukers nf=64 (no gain)",false,64);
    run(&corpus,split,"Beukers nf=64 + gain α",true,64);

    // Also test: swish with gain for comparison
    println!();
    println!("  (swish has no 2-proj so gain = just scaling, less meaningful)");

    println!("\n  Total time: {:.1}s",t0.elapsed().as_secs_f64());
}
