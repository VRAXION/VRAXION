//! Test: does adding a brain layer on top of frozen conv improve accuracy?
//!
//! A) Baseline: embed + Beukers conv + linear head → ~83%
//! B) Extra: embed + Beukers conv (FROZEN) + brain MLP + head → ???
//!
//! If B > A → the brain adds value on top of conv features
//!
//! Run: cargo run --example verify_brain_on_top --release

use std::time::Instant;
const VOCAB:usize=2000;const DIM:usize=16;

struct Rng(u64);
impl Rng{fn new(s:u64)->Self{Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1))}
fn next(&mut self)->u64{self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);self.0}
fn normal(&mut self)->f32{let u1=(((self.next()>>33)%65536)as f32/65536.0).max(1e-7);let u2=((self.next()>>33)%65536)as f32/65536.0;(-2.0*u1.ln()).sqrt()*(2.0*std::f32::consts::PI*u2).cos()}
fn range(&mut self,lo:usize,hi:usize)->usize{if hi<=lo{lo}else{lo+(self.next()as usize%(hi-lo))}}}

fn load_corpus(p:&str)->Vec<u8>{std::fs::read(p).expect("r").iter().filter_map(|&b|match b{
    b'a'..=b'z'=>Some(b-b'a'),b'A'..=b'Z'=>Some(b-b'A'),b' '|b'\n'|b'\t'|b'\r'=>Some(26),_=>None}).collect()}

fn swish(x:f32)->f32{x/(1.0+(-x).exp())}
fn swish_g(x:f32)->f32{let s=1.0/(1.0+(-x).exp());s+x*s*(1.0-s)}

fn main(){
    let t0=Instant::now();
    let corpus=load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let split=corpus.len()*80/100;

    let ctx=32;let mask_pos=ctx/2;let nf=96;let k=7;let hk=3i32;
    let mut rng=Rng::new(42);
    let sc_e=(1.0/DIM as f32).sqrt();let fan=k*DIM;let sc_c=(2.0/fan as f32).sqrt();

    // ── Phase 1: Train embedding + conv (same as before) ──
    println!("=== BRAIN ON TOP TEST ===\n");
    println!("  Phase 1: Training embedding + Beukers conv...");

    let mut embed:Vec<[f32;DIM]>=(0..VOCAB).map(|_|{let mut v=[0.0;DIM];for d in 0..DIM{v[d]=rng.normal()*sc_e;}v}).collect();
    let mut w1:Vec<Vec<f32>>=(0..nf).map(|_|(0..fan).map(|_|rng.normal()*sc_c).collect()).collect();
    let mut b1=vec![0.0f32;nf];
    let mut w2:Vec<Vec<f32>>=(0..nf).map(|_|(0..fan).map(|_|rng.normal()*sc_c).collect()).collect();
    let mut b2=vec![0.0f32;nf];
    // Simple head for Phase 1 training
    let sc_h=(2.0/nf as f32).sqrt();
    let mut hw:Vec<Vec<f32>>=(0..27).map(|_|(0..nf).map(|_|rng.normal()*sc_h).collect()).collect();
    let mut hb=vec![0.0f32;27];

    for ep in 0..1000{
        let lr=0.01*(1.0-ep as f32/1000.0*0.8);
        let mut rt=Rng::new(ep as u64*1000+42);
        for _ in 0..5000{
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
            for f in 0..nf{let prod=p1[f]*p2[f];co[f]=prod/(1.0+prod.abs());}
            let mut logits=hb.clone();
            for c in 0..27{for f in 0..nf{logits[c]+=hw[c][f]*co[f];}}
            let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
            let mut pr=vec![0.0f32;27];let mut s=0.0f32;
            for c in 0..27{pr[c]=(logits[c]-mx).exp();s+=pr[c];}for c in 0..27{pr[c]/=s;}
            pr[target]-=1.0;
            let mut dc=vec![0.0f32;nf];
            for c in 0..27{for f in 0..nf{dc[f]+=pr[c]*hw[c][f];hw[c][f]-=lr*pr[c]*co[f];}hb[c]-=lr*pr[c];}
            for f in 0..nf{let prod=p1[f]*p2[f];let den=(1.0+prod.abs()).powi(2);
                let d1=dc[f]*p2[f]/den;let d2=dc[f]*p1[f]/den;
                for ki in 0..k{let pos=mask_pos as i32+ki as i32-hk;
                    if pos>=0&&(pos as usize)<ctx{let pi=pos as usize;
                        for d in 0..DIM{if pi!=mask_pos{embed[chunk[pi]as usize][d]-=lr*(d1+d2)*w1[f][ki*DIM+d]*0.25;}
                            w1[f][ki*DIM+d]-=lr*d1*emb[pi][d];w2[f][ki*DIM+d]-=lr*d2*emb[pi][d];}}}
                b1[f]-=lr*d1;b2[f]-=lr*d2;}
        }
    }

    // Eval Phase 1 baseline
    let compute_conv=|chunk:&[u8], pos:usize|->Vec<f32>{
        let emb:Vec<[f32;DIM]>=(0..ctx).map(|i|if i==pos{[0.0;DIM]}else{embed[chunk[i]as usize]}).collect();
        let mut co=vec![0.0f32;nf];
        for f in 0..nf{let mut v1=b1[f];let mut v2=b2[f];
            for ki in 0..k{let p=pos as i32+ki as i32-hk;
                if p>=0&&(p as usize)<ctx{for d in 0..DIM{
                    v1+=w1[f][ki*DIM+d]*emb[p as usize][d];
                    v2+=w2[f][ki*DIM+d]*emb[p as usize][d];}}}
            let prod=v1*v2;co[f]=prod/(1.0+prod.abs());}
        co};

    let eval_head=|start:usize,end:usize|->f64{
        let mut rng3=Rng::new(999);let mut ok=0usize;let mut tot=0usize;
        for _ in 0..2000{if end<start+ctx+1{break;}let off=rng3.range(start,end.saturating_sub(ctx+1));
            let chunk=&corpus[off..off+ctx];let co=compute_conv(chunk,mask_pos);
            let mut logits=hb.clone();for c in 0..27{for f in 0..nf{logits[c]+=hw[c][f]*co[f];}}
            let pred=logits.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|v|v.0).unwrap_or(0);
            if pred==corpus[off+mask_pos]as usize{ok+=1;}tot+=1;}
        if tot==0{0.0}else{ok as f64/tot as f64*100.0}};

    let base_train=eval_head(0,split);
    let base_test=eval_head(split,corpus.len());
    println!("  Phase 1 done ({:.1}s)", t0.elapsed().as_secs_f64());
    println!("  A) Conv + linear head: train={:.1}% test={:.1}%\n", base_train, base_test);

    // ── Phase 2: FREEZE conv, train brain MLP on top ──
    println!("  Phase 2: Freeze conv, train brain MLP on top...\n");

    // Brain: conv_features (nf) + raw_embedding (DIM) → hidden → 27
    // Brain sees BOTH conv features AND raw embedding
    let brain_in = nf + DIM; // 96 + 16 = 112
    let brain_hid = 128;
    let sb1=(2.0/brain_in as f32).sqrt();let sb2=(2.0/brain_hid as f32).sqrt();
    let mut bw1:Vec<Vec<f32>>=(0..brain_hid).map(|_|(0..brain_in).map(|_|rng.normal()*sb1).collect()).collect();
    let mut bb1=vec![0.0f32;brain_hid];
    let mut bw2:Vec<Vec<f32>>=(0..27).map(|_|(0..brain_hid).map(|_|rng.normal()*sb2).collect()).collect();
    let mut bb2=vec![0.0f32;27];

    let brain_params = brain_hid*brain_in + brain_hid + 27*brain_hid + 27;
    println!("  Brain: {}→{}→27 ({} params), activation: Beukers gate\n", brain_in, brain_hid, brain_params);

    // Brain uses Beukers gate internally too
    let mut bw1b:Vec<Vec<f32>>=(0..brain_hid).map(|_|(0..brain_in).map(|_|rng.normal()*sb1).collect()).collect();
    let mut bb1b=vec![0.0f32;brain_hid];

    println!("  {:>5} {:>8} {:>8}", "epoch", "train%", "test%");
    println!("  {}", "-".repeat(25));

    for ep in 0..1000{
        let lr=0.005*(1.0-ep as f32/1000.0*0.8);
        let mut rt=Rng::new(ep as u64*1000+42);
        for _ in 0..5000{
            let off=rt.range(0,split.saturating_sub(ctx+1));
            let chunk=&corpus[off..off+ctx];let target=chunk[mask_pos]as usize;

            // Frozen conv features
            let co=compute_conv(chunk,mask_pos);
            // Raw embedding at mask position (zero since masked)
            // Actually: concat raw embeddings of NEIGHBORS with conv features
            let raw_emb = embed[corpus[off+mask_pos-1]as usize]; // left neighbor embedding

            // Brain input: conv features + left neighbor raw embedding
            let mut brain_input = co.clone();
            for d in 0..DIM { brain_input.push(raw_emb[d]); }

            // Brain forward: Beukers gate hidden
            let mut h1a=vec![0.0f32;brain_hid];let mut h1b=vec![0.0f32;brain_hid];
            for j in 0..brain_hid{h1a[j]=bb1[j];h1b[j]=bb1b[j];
                for i in 0..brain_in{h1a[j]+=bw1[j][i]*brain_input[i];h1b[j]+=bw1b[j][i]*brain_input[i];}}
            let mut h=vec![0.0f32;brain_hid];
            for j in 0..brain_hid{let prod=h1a[j]*h1b[j];h[j]=prod/(1.0+prod.abs());}

            // Output
            let mut logits=bb2.clone();
            for c in 0..27{for j in 0..brain_hid{logits[c]+=bw2[c][j]*h[j];}}
            let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
            let mut pr=vec![0.0f32;27];let mut s=0.0f32;
            for c in 0..27{pr[c]=(logits[c]-mx).exp();s+=pr[c];}for c in 0..27{pr[c]/=s;}
            pr[target]-=1.0;

            // Backprop brain only (conv is frozen)
            let mut dh=vec![0.0f32;brain_hid];
            for c in 0..27{for j in 0..brain_hid{dh[j]+=pr[c]*bw2[c][j];bw2[c][j]-=lr*pr[c]*h[j];}bb2[c]-=lr*pr[c];}
            for j in 0..brain_hid{
                let prod=h1a[j]*h1b[j];let den=(1.0+prod.abs()).powi(2);
                let da=dh[j]*h1b[j]/den;let db=dh[j]*h1a[j]/den;
                for i in 0..brain_in{bw1[j][i]-=lr*da*brain_input[i];bw1b[j][i]-=lr*db*brain_input[i];}
                bb1[j]-=lr*da;bb1b[j]-=lr*db;
            }
        }

        if ep%100==0{
            let eval_brain=|start:usize,end:usize|->f64{
                let mut rng3=Rng::new(999);let mut ok=0usize;let mut tot=0usize;
                for _ in 0..2000{if end<start+ctx+1{break;}let off=rng3.range(start,end.saturating_sub(ctx+1));
                    let chunk=&corpus[off..off+ctx];
                    let co=compute_conv(chunk,mask_pos);
                    let raw_emb=embed[corpus[off+mask_pos-1]as usize];
                    let mut bi=co.clone();for d in 0..DIM{bi.push(raw_emb[d]);}
                    let mut ha=vec![0.0f32;brain_hid];let mut hbb=vec![0.0f32;brain_hid];
                    for j in 0..brain_hid{ha[j]=bb1[j];hbb[j]=bb1b[j];
                        for i in 0..brain_in{ha[j]+=bw1[j][i]*bi[i];hbb[j]+=bw1b[j][i]*bi[i];}}
                    let mut hh=vec![0.0f32;brain_hid];
                    for j in 0..brain_hid{let p=ha[j]*hbb[j];hh[j]=p/(1.0+p.abs());}
                    let mut logits=bb2.clone();
                    for c in 0..27{for j in 0..brain_hid{logits[c]+=bw2[c][j]*hh[j];}}
                    let pred=logits.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|v|v.0).unwrap_or(0);
                    if pred==corpus[off+mask_pos]as usize{ok+=1;}tot+=1;}
                if tot==0{0.0}else{ok as f64/tot as f64*100.0}};
            let tr=eval_brain(0,split);let te=eval_brain(split,corpus.len());
            println!("  {:>5} {:>7.1}% {:>7.1}%",ep,tr,te);
        }
    }

    println!("\n  ━━━ COMPARISON ━━━");
    println!("  A) Conv + linear head:          train={:.1}% test={:.1}%", base_train, base_test);
    println!("  B) Conv (frozen) + Beukers brain: see above");
    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
