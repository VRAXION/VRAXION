//! L2 Embedding — 2 merger pairs, backprop STE, full int8
//!
//! Same approach as L1 merger winner: linear + full int8 weights + backprop
//! 531,441 possible inputs (27^4), sweep M=2,3,4 neurons
//! Sampled eval + full verification on winner
//!
//! Run: cargo run --example l2_embed_backprop --release

use std::time::Instant;

const LUT: [[i8; 2]; 27] = [
    [-2,-4],[-4,-2],[0,-6],[-2,-5],[-1,-6],[-3,-5],[1,-8],[-1,-7],
    [-2,-6],[-4,-5],[0,-8],[-2,-7],[-1,-8],[-3,-7],[1,-10],[-1,-9],
    [-4,-6],[-6,-5],[-2,-8],[-4,-7],[-3,-8],[-5,-7],[-1,-10],[-3,-9],
    [-4,-8],[-6,-7],[-2,-10],
];
const MW: [[i8;4];2] = [[-8,12,-7,-12],[-8,-6,-1,2]];
const MB: [i8;2] = [-1, 10];

fn merger_val(a: u8, b: u8) -> [f32; 2] {
    let i = [LUT[a as usize][0] as f32, LUT[a as usize][1] as f32,
             LUT[b as usize][0] as f32, LUT[b as usize][1] as f32];
    let mut o = [0.0f32; 2];
    for k in 0..2 { o[k] = MB[k] as f32 + MW[k][0] as f32*i[0] + MW[k][1] as f32*i[1]
        + MW[k][2] as f32*i[2] + MW[k][3] as f32*i[3]; }
    [o[0]/16.0, o[1]/16.0]
}

struct Rng(u64);
impl Rng{
    fn new(s:u64)->Self{Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1))}
    fn next(&mut self)->u64{self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);self.0}
    fn uniform(&mut self,lo:f32,hi:f32)->f32{lo+((self.next()>>33)%65536)as f32/65536.0*(hi-lo)}
}

fn main() {
    let t0 = Instant::now();

    // Pre-compute all 729 merger pair values
    let pair_vals: Vec<[f32;2]> = (0..27u8).flat_map(|a|
        (0..27u8).map(move |b| merger_val(a, b))
    ).collect();

    // Sample 3000 from 531K for training/eval
    let sample_size = 729; // same as L1 merger — fast enough for nearest-neighbor
    let mut rng = Rng::new(42);
    let sampled: Vec<[f32;4]> = (0..sample_size).map(|_| {
        let p1 = (rng.next() as usize) % 729;
        let p2 = (rng.next() as usize) % 729;
        [pair_vals[p1][0], pair_vals[p1][1], pair_vals[p2][0], pair_vals[p2][1]]
    }).collect();

    println!("=== L2 EMBED — BACKPROP STE (full int8, linear) ===\n");
    println!("  4 input values (2 merger pairs), sweep M=2,3,4 neurons");
    println!("  531,441 possible inputs, {} sampled for eval\n", sample_size);

    let activations: &[(&str, bool)] = &[("linear", false), ("c19", true)];

    fn c19a(x:f32,c:f32,r:f32)->f32{let c=c.max(0.1);let r=r.max(0.0);let l=6.0*c;
        if x>=l{return x-l;}if x<=-l{return x+l;}
        let s=x/c;let n=s.floor();let t=s-n;let h=t*(1.0-t);
        let sg=if(n as i32)%2==0{1.0}else{-1.0};c*(sg*h+r*h*h)}
    fn c19g(x:f32,c:f32,r:f32)->f32{let c=c.max(0.1);let r=r.max(0.0);let l=6.0*c;
        if x>=l||x<=-l{return 1.0;}let s=x/c;let n=s.floor();let t=s-n;let h=t*(1.0-t);
        let sg=if(n as i32)%2==0{1.0}else{-1.0};(sg+2.0*r*h)*(1.0-2.0*t)}

    for &(act_name, use_c19) in activations {
        for m in 2..=4usize {
            let tc = Instant::now();
            let mut best_score = 0usize;
            let mut best_cfg = String::new();

            let n_seeds = if m <= 3 { 200 } else { 100 };

            for seed in 0..n_seeds {
                let mut rng2 = Rng::new(seed*7919+42);
                let mut w = [[0.0f32;4];4];
                let mut b = [0.0f32;4];
                let mut cv = [0.0f32;4];
                let mut rv = [0.0f32;4];
                for k in 0..m {
                    for j in 0..4{w[k][j]=rng2.uniform(-20.0,20.0);}
                    b[k]=rng2.uniform(-20.0,20.0);
                    cv[k]=rng2.uniform(5.0,60.0);
                    rv[k]=rng2.uniform(0.0,2.0);
                }

                let wc = 50.0f32;

                for ep in 0..2000 {
                    let lr=0.02*(1.0-ep as f32/2000.0*0.8);

                    let qw:Vec<[i8;4]>=(0..m).map(|k|[
                        w[k][0].round().max(-wc).min(wc)as i8,w[k][1].round().max(-wc).min(wc)as i8,
                        w[k][2].round().max(-wc).min(wc)as i8,w[k][3].round().max(-wc).min(wc)as i8]).collect();
                    let qb:Vec<i8>=(0..m).map(|k|b[k].round().max(-wc).min(wc)as i8).collect();

                    let codes:Vec<Vec<f32>>=sampled.iter().map(|inp|{
                        (0..m).map(|k|{
                            let dot=qb[k]as f32+qw[k][0]as f32*inp[0]+qw[k][1]as f32*inp[1]
                                +qw[k][2]as f32*inp[2]+qw[k][3]as f32*inp[3];
                            if use_c19{c19a(dot,cv[k],rv[k])}else{dot}
                        }).collect()
                    }).collect();

                    let mut n_active=0u32;
                    for i in 0..sample_size{
                        let mut nj=0;let mut nd=f32::MAX;
                        for j in 0..sample_size{if j==i{continue;}
                            let d:f32=(0..m).map(|k|(codes[i][k]-codes[j][k]).powi(2)).sum();
                            if d<nd{nd=d;nj=j;}}
                        if nd<2.0{n_active+=1;
                            for k in 0..m{
                                let sign=if codes[i][k]>codes[nj][k]{1.0}
                                    else if codes[i][k]<codes[nj][k]{-1.0}else{0.0};
                                let dot_i=qb[k]as f32+qw[k][0]as f32*sampled[i][0]+qw[k][1]as f32*sampled[i][1]
                                    +qw[k][2]as f32*sampled[i][2]+qw[k][3]as f32*sampled[i][3];
                                let gi = if use_c19{c19g(dot_i,cv[k],rv[k])}else{1.0};
                                for j in 0..4{w[k][j]+=lr*sign*gi*sampled[i][j]*0.001;}
                                b[k]+=lr*sign*gi*0.001;
                                if use_c19{
                                    let eps=0.01;
                                    cv[k]+=lr*sign*(c19a(dot_i,cv[k]+eps,rv[k])-c19a(dot_i,cv[k]-eps,rv[k]))/(2.0*eps)*0.0001;
                                    cv[k]=cv[k].max(0.5).min(100.0);
                                    rv[k]+=lr*sign*(c19a(dot_i,cv[k],rv[k]+eps)-c19a(dot_i,cv[k],rv[k]-eps))/(2.0*eps)*0.0001;
                                    rv[k]=rv[k].max(0.0).min(5.0);
                                }
                            }
                        }
                    }
                    if n_active==0{break;}
                }

                // Eval
                let qw:Vec<[i8;4]>=(0..m).map(|k|[
                    w[k][0].round().max(-wc).min(wc)as i8,w[k][1].round().max(-wc).min(wc)as i8,
                    w[k][2].round().max(-wc).min(wc)as i8,w[k][3].round().max(-wc).min(wc)as i8]).collect();
                let qb:Vec<i8>=(0..m).map(|k|b[k].round().max(-wc).min(wc)as i8).collect();

                let codes:Vec<Vec<f32>>=sampled.iter().map(|inp|{
                    (0..m).map(|k|{let dot=qb[k]as f32+qw[k][0]as f32*inp[0]+qw[k][1]as f32*inp[1]
                        +qw[k][2]as f32*inp[2]+qw[k][3]as f32*inp[3];
                        if use_c19{c19a(dot,cv[k],rv[k])}else{dot}}).collect()}).collect();

                let mut ok=0;
                for i in 0..sample_size{let mut best=0;let mut bd=f32::MAX;
                    for j in 0..sample_size{let d:f32=(0..m).map(|k|(codes[i][k]-codes[j][k]).powi(2)).sum();
                        if d<bd{bd=d;best=j;}}
                    if best==i{ok+=1;}}

                if ok>best_score{
                    best_score=ok;
                    best_cfg=format!("seed={}", seed);
                    for k in 0..m{best_cfg+=&format!("\n    N{}: w={:?} b={} c={:.1} rho={:.1}",
                        k,qw[k],qb[k],cv[k],rv[k]);}

                    if ok==sample_size{
                        println!("  {} M={}: {}/{} *** seed={} ({:.1}s)",
                            act_name,m,ok,sample_size,seed,tc.elapsed().as_secs_f64());
                        for k in 0..m{println!("    N{}: w={:?} b={} c={:.1} rho={:.1}",
                            k,qw[k],qb[k],cv[k],rv[k]);}
                        break;
                    }
                }

                if seed%50==49{
                    println!("  {} M={}: best {}/{} after {} seeds ({:.1}s)",
                        act_name,m,best_score,sample_size,seed+1,tc.elapsed().as_secs_f64());
                }
            }

            if best_score<sample_size{
                println!("  {} M={}: BEST {}/{} ({:.1}%) in {:.1}s",
                    act_name,m,best_score,sample_size,
                    best_score as f64/sample_size as f64*100.0,tc.elapsed().as_secs_f64());
                println!("  {}", best_cfg);
            }
            println!();

            if best_score==sample_size{break;} // found minimum M
            if tc.elapsed().as_secs()>180{println!("  (time limit)\n");break;}
        }
    }

    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
