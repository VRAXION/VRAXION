//! Deep Beukers FAST — reduced samples/epochs for overnight feasibility
//!
//! 2-layer Beukers conv, but with 2000 samples/ep and 500 epochs
//! Enough to see the trend without 10-hour runtime
//!
//! Run: cargo run --example char_embed_beukers_deep_fast --release

use std::time::Instant;
const VOCAB:usize=2000;const DIM:usize=16;
struct Rng(u64);
impl Rng{fn new(s:u64)->Self{Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1))}
fn next(&mut self)->u64{self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);self.0}
fn normal(&mut self)->f32{let u1=(((self.next()>>33)%65536)as f32/65536.0).max(1e-7);let u2=((self.next()>>33)%65536)as f32/65536.0;(-2.0*u1.ln()).sqrt()*(2.0*std::f32::consts::PI*u2).cos()}
fn range(&mut self,lo:usize,hi:usize)->usize{if hi<=lo{lo}else{lo+(self.next()as usize%(hi-lo))}}}
fn load_corpus(p:&str)->Vec<u8>{std::fs::read(p).expect("r").iter().filter_map(|&b|match b{b'a'..=b'z'=>Some(b-b'a'),b'A'..=b'Z'=>Some(b-b'A'),b' '|b'\n'|b'\t'|b'\r'=>Some(26),_=>None}).collect()}
fn beukers(x:f32,y:f32)->f32{let p=x*y;p/(1.0+p.abs())}

fn run(corpus:&[u8],split:usize,name:&str,k1:usize,nf1:usize,k2:usize,nf2:usize)->f64{
    let ctx=32;let mask_pos=ctx/2;let mut rng=Rng::new(42);
    let sc_e=(1.0/DIM as f32).sqrt();
    let mut embed:Vec<[f32;DIM]>=(0..VOCAB).map(|_|{let mut v=[0.0;DIM];for d in 0..DIM{v[d]=rng.normal()*sc_e;}v}).collect();
    let fan1=k1*DIM;let sc1=(2.0/fan1 as f32).sqrt();
    let mut w1a:Vec<Vec<f32>>=(0..nf1).map(|_|(0..fan1).map(|_|rng.normal()*sc1).collect()).collect();
    let mut b1a=vec![0.0f32;nf1];
    let mut w1b:Vec<Vec<f32>>=(0..nf1).map(|_|(0..fan1).map(|_|rng.normal()*sc1).collect()).collect();
    let mut b1b=vec![0.0f32;nf1];
    let fan2=k2*nf1;let sc2=(2.0/fan2 as f32).sqrt();
    let mut w2a:Vec<Vec<f32>>=(0..nf2).map(|_|(0..fan2).map(|_|rng.normal()*sc2).collect()).collect();
    let mut b2a=vec![0.0f32;nf2];
    let mut w2b:Vec<Vec<f32>>=(0..nf2).map(|_|(0..fan2).map(|_|rng.normal()*sc2).collect()).collect();
    let mut b2b=vec![0.0f32;nf2];
    let sch=(2.0/nf2 as f32).sqrt();
    let mut hw:Vec<Vec<f32>>=(0..27).map(|_|(0..nf2).map(|_|rng.normal()*sch).collect()).collect();
    let mut hb=vec![0.0f32;27];
    let hk1=(k1/2)as i32;let hk2=(k2/2)as i32;
    let rf=k1+k2-1;
    let total_p=2*(nf1*fan1+nf1)+2*(nf2*fan2+nf2)+27*nf2+27;
    print!("  {:>25} rf={:>2} p={:>5}:",name,rf,total_p);
    let samples=2000;let max_ep=500;let mut best_test=0.0f64;
    for ep in 0..max_ep{
        let lr=0.01*(1.0-ep as f32/max_ep as f32*0.8);
        let mut rt=Rng::new(ep as u64*1000+42);
        for _ in 0..samples{
            let off=rt.range(0,split.saturating_sub(ctx+1));
            let chunk=&corpus[off..off+ctx];let target=chunk[mask_pos]as usize;
            let emb:Vec<[f32;DIM]>=(0..ctx).map(|i|if i==mask_pos{[0.0;DIM]}else{embed[chunk[i]as usize]}).collect();
            // L1 only near mask (k2 range)
            let l1_range=((mask_pos as i32-hk2-hk1).max(0)as usize)..((mask_pos as i32+hk2+hk1+1).min(ctx as i32)as usize);
            let mut l1_out=vec![vec![0.0f32;nf1];ctx];
            for p in l1_range.clone(){let mut pa=vec![0.0f32;nf1];let mut pb=vec![0.0f32;nf1];
                for f in 0..nf1{pa[f]=b1a[f];pb[f]=b1b[f];
                    for ki in 0..k1{let pos=p as i32+ki as i32-hk1;
                        if pos>=0&&(pos as usize)<ctx{for d in 0..DIM{
                            pa[f]+=w1a[f][ki*DIM+d]*emb[pos as usize][d];
                            pb[f]+=w1b[f][ki*DIM+d]*emb[pos as usize][d];}}}}
                for f in 0..nf1{l1_out[p][f]=beukers(pa[f],pb[f]);}}
            // L2 at mask only
            let mut p2a=vec![0.0f32;nf2];let mut p2b=vec![0.0f32;nf2];
            for f in 0..nf2{p2a[f]=b2a[f];p2b[f]=b2b[f];
                for ki in 0..k2{let pos=mask_pos as i32+ki as i32-hk2;
                    if pos>=0&&(pos as usize)<ctx{for d in 0..nf1{
                        p2a[f]+=w2a[f][ki*nf1+d]*l1_out[pos as usize][d];
                        p2b[f]+=w2b[f][ki*nf1+d]*l1_out[pos as usize][d];}}}}
            let mut l2o=vec![0.0f32;nf2];
            for f in 0..nf2{l2o[f]=beukers(p2a[f],p2b[f]);}
            let mut logits=hb.clone();
            for c in 0..27{for f in 0..nf2{logits[c]+=hw[c][f]*l2o[f];}}
            let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
            let mut pr=vec![0.0f32;27];let mut s=0.0f32;
            for c in 0..27{pr[c]=(logits[c]-mx).exp();s+=pr[c];}for c in 0..27{pr[c]/=s;}
            pr[target]-=1.0;
            let mut dl2=vec![0.0f32;nf2];
            for c in 0..27{for f in 0..nf2{dl2[f]+=pr[c]*hw[c][f];hw[c][f]-=lr*pr[c]*l2o[f];}hb[c]-=lr*pr[c];}
            // Backprop L2
            for f in 0..nf2{let prod=p2a[f]*p2b[f];let den=(1.0+prod.abs()).powi(2);
                let da=dl2[f]*p2b[f]/den;let db=dl2[f]*p2a[f]/den;
                for ki in 0..k2{let pos=mask_pos as i32+ki as i32-hk2;
                    if pos>=0&&(pos as usize)<ctx{for d in 0..nf1{
                        w2a[f][ki*nf1+d]-=lr*da*l1_out[pos as usize][d];
                        w2b[f][ki*nf1+d]-=lr*db*l1_out[pos as usize][d];}}}
                b2a[f]-=lr*da;b2b[f]-=lr*db;}
            // Backprop L1 (near mask only)
            for p in l1_range{for f1 in 0..nf1{let mut dl1=0.0f32;
                for f2 in 0..nf2{let ki2_i=p as i32-mask_pos as i32+hk2;
                    if ki2_i>=0&&(ki2_i as usize)<k2{let ki2=ki2_i as usize;
                        let prod=p2a[f2]*p2b[f2];let den=(1.0+prod.abs()).powi(2);
                        dl1+=dl2[f2]*(p2b[f2]*w2a[f2][ki2*nf1+f1]+p2a[f2]*w2b[f2][ki2*nf1+f1])/den;}}
                let mut pa_f=b1a[f1];let mut pb_f=b1b[f1];
                for ki in 0..k1{let pos=p as i32+ki as i32-hk1;
                    if pos>=0&&(pos as usize)<ctx{for d in 0..DIM{
                        pa_f+=w1a[f1][ki*DIM+d]*emb[pos as usize][d];
                        pb_f+=w1b[f1][ki*DIM+d]*emb[pos as usize][d];}}}
                let prod1=pa_f*pb_f;let den1=(1.0+prod1.abs()).powi(2);
                let da1=dl1*pb_f/den1;let db1=dl1*pa_f/den1;
                for ki in 0..k1{let pos=p as i32+ki as i32-hk1;
                    if pos>=0&&(pos as usize)<ctx{let pi=pos as usize;
                        for d in 0..DIM{if pi!=mask_pos{embed[chunk[pi]as usize][d]-=lr*(da1+db1)*w1a[f1][ki*DIM+d]*0.1;}
                            w1a[f1][ki*DIM+d]-=lr*da1*emb[pi][d];w1b[f1][ki*DIM+d]-=lr*db1*emb[pi][d];}}}
                b1a[f1]-=lr*da1;b1b[f1]-=lr*db1;}}
        }
        if ep%50==0{
            let eval=|start:usize,end:usize|->f64{let mut rng3=Rng::new(999);let mut ok=0usize;let mut tot=0usize;
                for _ in 0..500{if end<start+ctx+1{break;}let off=rng3.range(start,end.saturating_sub(ctx+1));
                    let chunk=&corpus[off..off+ctx];
                    let emb:Vec<[f32;DIM]>=(0..ctx).map(|i|if i==mask_pos{[0.0;DIM]}else{embed[chunk[i]as usize]}).collect();
                    let mut l1=vec![vec![0.0f32;nf1];ctx];
                    for p in 0..ctx{let mut pa=vec![0.0f32;nf1];let mut pb=vec![0.0f32;nf1];
                        for f in 0..nf1{pa[f]=b1a[f];pb[f]=b1b[f];
                            for ki in 0..k1{let pos=p as i32+ki as i32-hk1;
                                if pos>=0&&(pos as usize)<ctx{for d in 0..DIM{
                                    pa[f]+=w1a[f][ki*DIM+d]*emb[pos as usize][d];
                                    pb[f]+=w1b[f][ki*DIM+d]*emb[pos as usize][d];}}}}
                        for f in 0..nf1{l1[p][f]=beukers(pa[f],pb[f]);}}
                    let mut co=vec![0.0f32;nf2];
                    for f in 0..nf2{let mut va=b2a[f];let mut vb=b2b[f];
                        for ki in 0..k2{let pos=mask_pos as i32+ki as i32-hk2;
                            if pos>=0&&(pos as usize)<ctx{for d in 0..nf1{
                                va+=w2a[f][ki*nf1+d]*l1[pos as usize][d];
                                vb+=w2b[f][ki*nf1+d]*l1[pos as usize][d];}}}
                        co[f]=beukers(va,vb);}
                    let mut logits=hb.clone();for c in 0..27{for f in 0..nf2{logits[c]+=hw[c][f]*co[f];}}
                    let pred=logits.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|v|v.0).unwrap_or(0);
                    if pred==chunk[mask_pos]as usize{ok+=1;}tot+=1;}
                if tot==0{0.0}else{ok as f64/tot as f64*100.0}};
            let te=eval(split,corpus.len());if te>best_test{best_test=te;}
            print!(" {:.1}",te);
        }
    }
    println!("  | best={:.1}%",best_test);best_test
}

fn main(){
    let t0=Instant::now();
    let corpus=load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let split=corpus.len()*80/100;
    println!("=== DEEP BEUKERS FAST (2 layers, 500ep, 2K samples) ===\n");

    run(&corpus,split,"L1:k7,nf32 L2:k3,nf32",7,32,3,32);
    run(&corpus,split,"L1:k5,nf32 L2:k3,nf32",5,32,3,32);
    run(&corpus,split,"L1:k7,nf48 L2:k3,nf48",7,48,3,48);
    run(&corpus,split,"L1:k5,nf48 L2:k5,nf48",5,48,5,48);

    println!("\n  Ref: 1-layer Beukers k=7 nf=128 = 83.6%");
    println!("  Total time: {:.1}s",t0.elapsed().as_secs_f64());
}
