//! L2 Full Embedding — all mergers into 1 unit, contrastive loss
//!
//! 256 mergers (512 bytes) → M output neurons, linear, int8 STE
//! Contrastive: push apart codes for different chunks
//! Test: round-trip on corpus chunks (can we tell them apart?)
//!
//! Run: cargo run --example l2_embed_full_contrastive --release

use std::time::Instant;

const LUT: [[i8; 2]; 27] = [
    [-2,-4],[-4,-2],[0,-6],[-2,-5],[-1,-6],[-3,-5],[1,-8],[-1,-7],
    [-2,-6],[-4,-5],[0,-8],[-2,-7],[-1,-8],[-3,-7],[1,-10],[-1,-9],
    [-4,-6],[-6,-5],[-2,-8],[-4,-7],[-3,-8],[-5,-7],[-1,-10],[-3,-9],
    [-4,-8],[-6,-7],[-2,-10],
];

struct Rng(u64);
impl Rng{
    fn new(s:u64)->Self{Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1))}
    fn next(&mut self)->u64{self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);self.0}
    fn normal(&mut self)->f32{let u1=(((self.next()>>33)%65536)as f32/65536.0).max(1e-7);let u2=((self.next()>>33)%65536)as f32/65536.0;(-2.0*u1.ln()).sqrt()*(2.0*std::f32::consts::PI*u2).cos()}
    fn range(&mut self,lo:usize,hi:usize)->usize{if hi<=lo{lo}else{lo+(self.next()as usize%(hi-lo))}}
}

fn load_corpus(p:&str)->Vec<u8>{std::fs::read(p).expect("r").iter().filter_map(|&b|match b{
    b'a'..=b'z'=>Some(b-b'a'),b'A'..=b'Z'=>Some(b-b'A'),b' '|b'\n'|b'\t'|b'\r'=>Some(26),_=>None}).collect()}

fn main(){
    let t0=Instant::now();
    let corpus=load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");

    // Encode corpus to LUT values
    let encoded:Vec<f32>=corpus.iter().flat_map(|&ch|
        [LUT[ch as usize][0]as f32/16.0,LUT[ch as usize][1]as f32/16.0]).collect();

    let chunk=512; // bytes
    let in_dim=chunk*2; // 1024 float values
    let stride=32; // overlapping chunks

    // Build chunks
    let n_chunks=(corpus.len()-chunk)/stride;
    let chunks:Vec<Vec<f32>>=(0..n_chunks).map(|i|{
        let off=i*stride;
        encoded[off*2..(off+chunk)*2].to_vec()
    }).collect();
    let split=n_chunks*80/100;

    println!("=== FULL EMBED — CONTRASTIVE (512 bytes → M neurons) ===\n");
    println!("  {} chunks ({} train, {} test), in_dim={}", n_chunks, split, n_chunks-split, in_dim);

    // Sweep output dims
    for &out_dim in &[64, 128, 256, 512] {
        let tc=Instant::now();
        let compression = in_dim as f64 / out_dim as f64;

        // Linear encoder: in_dim → out_dim (no activation, just dot products)
        // Use random projection init (like Johnson-Lindenstrauss)
        let mut rng=Rng::new(42);
        let scale=(1.0/in_dim as f32).sqrt();
        let mut w:Vec<Vec<f32>>=(0..out_dim).map(|_|
            (0..in_dim).map(|_|rng.normal()*scale).collect()
        ).collect();
        let mut b:Vec<f32>=vec![0.0;out_dim];

        let encode=|w:&Vec<Vec<f32>>,b:&Vec<f32>,input:&[f32]|->Vec<f32>{
            (0..w.len()).map(|j|{let mut v=b[j];
                for k in 0..input.len().min(w[j].len()){v+=w[j][k]*input[k];}v}).collect()
        };

        // Eval: for each chunk, is its code unique? (nearest neighbor = self among train set)
        let eval=|w:&Vec<Vec<f32>>,b:&Vec<f32>,start:usize,end:usize|->f64{
            let codes:Vec<Vec<f32>>=(start..end).map(|i|encode(w,b,&chunks[i])).collect();
            let n=codes.len();
            let mut ok=0usize;
            for i in 0..n{let mut best=0;let mut bd=f32::MAX;
                for j in 0..n{if i==j{continue;}
                    let d:f32=codes[i].iter().zip(&codes[j]).map(|(a,b)|(a-b)*(a-b)).sum();
                    if d<bd{bd=d;best=j;}}
                if best==i{ok+=1;} // nearest neighbor is itself (trivially true)
            }
            // Actually: check that no two different chunks have same code
            let mut collisions=0;
            for i in 0..n{for j in (i+1)..n{
                let d:f32=codes[i].iter().zip(&codes[j]).map(|(a,b)|(a-b)*(a-b)).sum();
                if d<0.001{collisions+=1;}
            }}
            (n-collisions) as f64/n as f64*100.0
        };

        // Random projection baseline
        let base=eval(&w,&b,0,split.min(500));

        // Contrastive training: push apart nearest pairs
        println!("  out_dim={} ({:.1}x compression), {} params",
            out_dim, compression, out_dim*in_dim+out_dim);
        println!("  {:>5} {:>8} {:>8} {:>6}","epoch","train%","test%","time");

        let train_n = split.min(500); // use subset for speed
        let mut best_test = 0.0f64;
        let mut plateau = 0u32;

        for ep in 0..500 {
            let lr=0.001*(1.0-ep as f32/500.0*0.7);

            // Compute codes for training chunks
            let codes:Vec<Vec<f32>>=(0..train_n).map(|i|encode(&w,&b,&chunks[i])).collect();

            // For each chunk, find nearest neighbor and push apart
            let mut n_active=0u32;
            for i in 0..train_n{
                let mut nj=0;let mut nd=f32::MAX;
                for j in 0..train_n{if j==i{continue;}
                    let d:f32=codes[i].iter().zip(&codes[j]).map(|(a,b)|(a-b)*(a-b)).sum();
                    if d<nd{nd=d;nj=j;}}

                if nd < 1.0 {
                    n_active+=1;
                    // Push i away from nj
                    for k in 0..out_dim{
                        let diff=codes[i][k]-codes[nj][k];
                        let sign=if diff>0.0{1.0}else if diff<0.0{-1.0}else{0.0};
                        // Gradient for linear: d_code/d_w[k][j] = input[j]
                        for j in 0..in_dim{
                            w[k][j]+=lr*sign*(chunks[i][j]-chunks[nj][j])*0.001;
                        }
                        b[k]+=lr*sign*0.001;
                    }
                }
            }

            if ep%25==0{
                let tr=eval(&w,&b,0,train_n);
                let te=eval(&w,&b,split,n_chunks.min(split+300));
                println!("  {:>5} {:>7.1}% {:>7.1}% {:>5.0}s",ep,tr,te,tc.elapsed().as_secs_f64());

                if te>best_test+0.5{best_test=te;plateau=0;}else{plateau+=1;}
                if te>=99.9{println!("  → 100%!\n");break;}
                if plateau>=8{println!("  → Plateau at {:.1}%\n",best_test);break;}
            }

            if n_active==0{
                let tr=eval(&w,&b,0,train_n);
                let te=eval(&w,&b,split,n_chunks.min(split+300));
                println!("  {:>5} {:>7.1}% {:>7.1}% {:>5.0}s  ALL SEPARATED",ep,tr,te,tc.elapsed().as_secs_f64());
                break;
            }

            if tc.elapsed().as_secs()>120{println!("  → Time limit\n");break;}
        }
    }

    println!("  Total time: {:.1}s",t0.elapsed().as_secs_f64());
}
