//! Quick verification: is everything 100% round-trip?
//!
//! 1. Embedding: char → 16 int8 → nearest neighbor → same char? (should be 100%)
//! 2. Conv output: char → embed → Beukers conv → nearest neighbor → same char?
//! 3. Full pipeline: text → embed → conv → decode each position → original text?
//!
//! Run: cargo run --example verify_roundtrip --release

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

fn beukers(x:f32,y:f32)->f32{let p=x*y;p/(1.0+p.abs())}

fn main(){
    let t0=Instant::now();
    let corpus=load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let split=corpus.len()*80/100;
    let chars="abcdefghijklmnopqrstuvwxyz ";

    println!("=== ROUND-TRIP VERIFICATION ===\n");

    // ── Step 1: Train embedding + conv (same as before, fast) ──
    let ctx=32;let mask_pos=ctx/2;let nf=96;let k=7;let hk=3i32;
    let mut rng=Rng::new(42);
    let sc_e=(1.0/DIM as f32).sqrt();let fan=k*DIM;let sc_c=(2.0/fan as f32).sqrt();let sc_h=(2.0/nf as f32).sqrt();
    let mut embed:Vec<[f32;DIM]>=(0..VOCAB).map(|_|{let mut v=[0.0;DIM];for d in 0..DIM{v[d]=rng.normal()*sc_e;}v}).collect();
    let mut w1:Vec<Vec<f32>>=(0..nf).map(|_|(0..fan).map(|_|rng.normal()*sc_c).collect()).collect();
    let mut b1=vec![0.0f32;nf];
    let mut w2:Vec<Vec<f32>>=(0..nf).map(|_|(0..fan).map(|_|rng.normal()*sc_c).collect()).collect();
    let mut b2=vec![0.0f32;nf];
    let mut hw:Vec<Vec<f32>>=(0..27).map(|_|(0..nf).map(|_|rng.normal()*sc_h).collect()).collect();
    let mut hb=vec![0.0f32;27];

    println!("  Training embedding + Beukers conv (1000 epochs)...");
    for ep in 0..1000{
        let lr=0.01*(1.0-ep as f32/1000.0*0.8);
        let mut rt=Rng::new(ep as u64*1000+42);
        for _ in 0..3000{
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
    println!("  Training done ({:.1}s)\n", t0.elapsed().as_secs_f64());

    // ── Test 1: Embedding round-trip (char → 16 int8 → nearest → char) ──
    println!("--- TEST 1: Embedding round-trip (char → 16-dim → nearest → char) ---\n");
    let mut embed_ok = 0;
    for c in 0..27 {
        let vec_c = &embed[c];
        // Find nearest neighbor in embedding table
        let mut best = 0; let mut bd = f32::MAX;
        for j in 0..27 {
            let d: f32 = (0..DIM).map(|d| (vec_c[d] - embed[j][d]).powi(2)).sum();
            if d < bd { bd = d; best = j; }
        }
        let ok = best == c;
        if ok { embed_ok += 1; }
        if !ok {
            println!("  FAIL: '{}' → nearest = '{}' (dist={:.4})",
                chars.as_bytes()[c] as char, chars.as_bytes()[best] as char, bd);
        }
    }
    println!("  Embedding round-trip: {}/27 ({}%)\n", embed_ok, embed_ok*100/27);

    // ── Test 2: Conv feature → decode char (can brain recover char from conv output?) ──
    println!("--- TEST 2: Conv round-trip (text → embed → conv → decode → text) ---\n");
    println!("  For each position: conv output → linear probe → which char?\n");

    // Train a tiny linear probe: conv_output (nf dim) → 27 classes
    let mut probe_w: Vec<Vec<f32>> = (0..27).map(|_| (0..nf).map(|_| 0.0f32).collect()).collect();
    let mut probe_b = vec![0.0f32; 27];

    // Collect conv features for training the probe
    for ep in 0..200 {
        let lr = 0.01 * (1.0 - ep as f32 / 200.0 * 0.5);
        let mut rt = Rng::new(ep as u64 * 5555 + 42);
        for _ in 0..3000 {
            let off = rt.range(0, split.saturating_sub(ctx + 1));
            let chunk = &corpus[off..off+ctx];
            // Pick random NON-mask position
            let pos = rt.range(0, ctx);
            let target = chunk[pos] as usize;

            // Compute conv at this position
            let emb: Vec<[f32;DIM]> = (0..ctx).map(|i| embed[chunk[i] as usize]).collect();
            let mut v1 = vec![0.0f32; nf]; let mut v2 = vec![0.0f32; nf];
            for f in 0..nf {
                v1[f] = b1[f]; v2[f] = b2[f];
                for ki in 0..k {
                    let p = pos as i32 + ki as i32 - hk;
                    if p >= 0 && (p as usize) < ctx {
                        for d in 0..DIM {
                            v1[f] += w1[f][ki*DIM+d] * emb[p as usize][d];
                            v2[f] += w2[f][ki*DIM+d] * emb[p as usize][d];
                        }
                    }
                }
            }
            let mut co = vec![0.0f32; nf];
            for f in 0..nf { let prod = v1[f]*v2[f]; co[f] = prod/(1.0+prod.abs()); }

            // Linear probe forward + backprop
            let mut logits = probe_b.clone();
            for c in 0..27 { for f in 0..nf { logits[c] += probe_w[c][f] * co[f]; } }
            let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut pr = vec![0.0f32; 27]; let mut s = 0.0f32;
            for c in 0..27 { pr[c] = (logits[c]-mx).exp(); s += pr[c]; }
            for c in 0..27 { pr[c] /= s; }
            pr[target] -= 1.0;
            for c in 0..27 { for f in 0..nf { probe_w[c][f] -= lr * pr[c] * co[f]; } probe_b[c] -= lr * pr[c]; }
        }
    }

    // Eval probe on test set
    let mut probe_ok = 0; let mut probe_tot = 0;
    let mut rng3 = Rng::new(999);
    for _ in 0..5000 {
        if corpus.len() < split + ctx + 1 { break; }
        let off = rng3.range(split, corpus.len().saturating_sub(ctx+1));
        let chunk = &corpus[off..off+ctx];
        let pos = rng3.range(0, ctx);
        let target = chunk[pos] as usize;

        let emb: Vec<[f32;DIM]> = (0..ctx).map(|i| embed[chunk[i] as usize]).collect();
        let mut co = vec![0.0f32; nf];
        for f in 0..nf {
            let mut v1 = b1[f]; let mut v2 = b2[f];
            for ki in 0..k { let p = pos as i32 + ki as i32 - hk;
                if p >= 0 && (p as usize) < ctx { for d in 0..DIM {
                    v1 += w1[f][ki*DIM+d] * emb[p as usize][d];
                    v2 += w2[f][ki*DIM+d] * emb[p as usize][d]; }}}
            let prod = v1*v2; co[f] = prod/(1.0+prod.abs());
        }

        let mut logits = probe_b.clone();
        for c in 0..27 { for f in 0..nf { logits[c] += probe_w[c][f] * co[f]; } }
        let pred = logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|v|v.0).unwrap_or(0);
        if pred == target { probe_ok += 1; }
        probe_tot += 1;
    }
    println!("  Conv → linear probe → char: {}/{} ({:.1}%)\n", probe_ok, probe_tot,
        probe_ok as f64 / probe_tot as f64 * 100.0);

    // ── Test 3: Full text round-trip ──
    println!("--- TEST 3: Full text decode example ---\n");
    let test_text = "the alice was in the garden";
    let test_chars: Vec<u8> = test_text.bytes().map(|b| match b {
        b'a'..=b'z' => b - b'a', b' ' => 26, _ => 26 }).collect();

    print!("  Original: \"{}\"", test_text);
    print!("\n  Decoded:  \"");

    let emb: Vec<[f32;DIM]> = test_chars.iter().map(|&c| embed[c as usize]).collect();
    let tlen = test_chars.len();
    let mut decode_ok = 0;

    for pos in 0..tlen {
        let mut co = vec![0.0f32; nf];
        for f in 0..nf {
            let mut v1 = b1[f]; let mut v2 = b2[f];
            for ki in 0..k { let p = pos as i32 + ki as i32 - hk;
                if p >= 0 && (p as usize) < tlen { for d in 0..DIM {
                    v1 += w1[f][ki*DIM+d] * emb[p as usize][d];
                    v2 += w2[f][ki*DIM+d] * emb[p as usize][d]; }}}
            let prod = v1*v2; co[f] = prod/(1.0+prod.abs());
        }
        let mut logits = probe_b.clone();
        for c in 0..27 { for f in 0..nf { logits[c] += probe_w[c][f] * co[f]; } }
        let pred = logits.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|v|v.0).unwrap_or(0);
        print!("{}", chars.as_bytes()[pred] as char);
        if pred == test_chars[pos] as usize { decode_ok += 1; }
    }
    println!("\"");
    println!("  Match: {}/{} ({:.1}%)\n", decode_ok, tlen, decode_ok as f64/tlen as f64*100.0);

    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
