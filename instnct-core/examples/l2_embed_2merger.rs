//! L2 Embedding — 2 merger pairs → N neurons (round-trip)
//!
//! Input: 2 consecutive merger outputs = 4 float values
//! Output: N neuron outputs
//! Test: all 531,441 possible inputs (27^4) reconstructable
//! Method: exhaustive greedy search + backprop STE comparison
//!
//! Run: cargo run --example l2_embed_2merger --release

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

fn c19a(x:f32,c:f32,r:f32)->f32{let c=c.max(0.1);let r=r.max(0.0);let l=6.0*c;
    if x>=l{return x-l;}if x<=-l{return x+l;}
    let s=x/c;let n=s.floor();let t=s-n;let h=t*(1.0-t);
    let sg=if(n as i32)%2==0{1.0}else{-1.0};c*(sg*h+r*h*h)}

fn eval_roundtrip(codes: &Vec<Vec<f32>>, n: usize) -> usize {
    let total = codes.len();
    let mut ok = 0;
    for i in 0..total {
        let mut best = 0; let mut bd = f32::MAX;
        for j in 0..total {
            let d: f32 = (0..n).map(|k| (codes[i][k]-codes[j][k]).powi(2)).sum();
            if d < bd { bd = d; best = j; }
        }
        if best == i { ok += 1; }
    }
    ok
}

struct Rng(u64);
impl Rng{
    fn new(s:u64)->Self{Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1))}
    fn next(&mut self)->u64{self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);self.0}
    fn uniform(&mut self,lo:f32,hi:f32)->f32{lo+((self.next()>>33)%65536)as f32/65536.0*(hi-lo)}
}

fn main() {
    let t0 = Instant::now();

    // Pre-compute all 531,441 possible inputs (27^4)
    // But 531K nearest-neighbor is O(531K²) = too slow!
    // Use SAMPLED eval instead: random 5000 from 531K
    println!("=== L2 EMBEDDING — 2 MERGER PAIRS ===\n");

    // All 729 merger pair outputs (for one pair)
    let pair_vals: Vec<[f32;2]> = (0..27u8).flat_map(|a|
        (0..27u8).map(move |b| merger_val(a, b))
    ).collect(); // 729 entries

    // For 2 consecutive pairs: input = [pair1[0], pair1[1], pair2[0], pair2[1]]
    // Total: 729 × 729 = 531,441 combinations
    // Too many for full nearest-neighbor, so we sample

    // But for GREEDY SEARCH we can use a smaller subset
    // Use all 729 single pairs first (like merger did), then verify on pairs-of-pairs

    println!("  Single pair: 729 combos (same as merger)");
    println!("  Double pair: 729² = 531,441 combos (sample for eval)\n");

    // ══ Method 1: Exhaustive greedy search (same pattern as merger) ══
    println!("=== METHOD 1: EXHAUSTIVE GREEDY ===\n");

    // Input: 4 values from 2 merger pairs
    let inputs_4d: Vec<[f32;4]> = {
        let mut v = Vec::with_capacity(729*729);
        for p1 in 0..729 { for p2 in 0..729 {
            v.push([pair_vals[p1][0],pair_vals[p1][1],pair_vals[p2][0],pair_vals[p2][1]]);
        }}
        v
    }; // 531K entries

    // Subsample for eval
    let mut rng = Rng::new(42);
    let sample_size = 2000;
    let sample_idx: Vec<usize> = (0..sample_size).map(|_|
        (rng.next() as usize) % inputs_4d.len()
    ).collect();
    let sampled: Vec<[f32;4]> = sample_idx.iter().map(|&i| inputs_4d[i]).collect();

    struct WCfg { name: &'static str, range: Vec<i8> }
    let wcfgs = vec![
        WCfg { name: "ternary", range: vec![-1,0,1] },
        WCfg { name: "2-bit", range: vec![-2,-1,0,1,2] },
        WCfg { name: "3-bit", range: vec![-4,-3,-2,-1,0,1,2,3,4] },
    ];

    let c_vals: Vec<f32> = vec![1.0, 5.0, 10.0, 20.0, 50.0];

    for wcfg in &wcfgs {
        let nv = wcfg.range.len();
        let combos = nv.pow(5); // 4 weights + 1 bias

        for &use_c19 in &[true, false] {
            let act = if use_c19 {"C19"} else {"linear"};
            let c_list = if use_c19 {&c_vals[..]} else {&[1.0f32][..]};
            let tc = Instant::now();

            println!("  {} + {} ({} combos/neuron)", wcfg.name, act, combos * c_list.len());

            let mut codes: Vec<Vec<f32>> = (0..sample_size).map(|_| Vec::new()).collect();

            for ni in 0..6 {
                let mut top_score = 0;
                let mut top_w = [0i8;4];
                let mut top_b = 0i8;
                let mut top_c = 1.0f32;

                'search: for &cv in c_list {
                    for combo in 0..combos {
                        let mut w = [0i8;4];
                        let mut rem = combo;
                        for j in 0..4 { w[j] = wcfg.range[rem%nv]; rem/=nv; }
                        let b = wcfg.range[rem%nv];

                        let mut test = codes.clone();
                        for (idx, inp) in sampled.iter().enumerate() {
                            let dot = b as f32 + w[0] as f32*inp[0]+w[1] as f32*inp[1]
                                +w[2] as f32*inp[2]+w[3] as f32*inp[3];
                            let out = if use_c19 { c19a(dot, cv, 0.0) } else { dot };
                            test[idx].push(out);
                        }

                        let score = eval_roundtrip(&test, ni+1);
                        if score > top_score {
                            top_score = score; top_w = w; top_b = b; top_c = cv;
                            if score == sample_size { break 'search; }
                        }
                    }
                }

                for (idx, inp) in sampled.iter().enumerate() {
                    let dot = top_b as f32+top_w[0] as f32*inp[0]+top_w[1] as f32*inp[1]
                        +top_w[2] as f32*inp[2]+top_w[3] as f32*inp[3];
                    let out = if use_c19 { c19a(dot, top_c, 0.0) } else { dot };
                    codes[idx].push(out);
                }

                let pct = top_score as f64 / sample_size as f64 * 100.0;
                let m = if top_score == sample_size { " ***" } else { "" };
                println!("    N={}: {}/{} ({:.1}%)  w={:?} b={:+} c={}{}",
                    ni+1, top_score, sample_size, pct, top_w, top_b, top_c, m);

                if top_score == sample_size {
                    println!("    → {} neurons, {:.1}s\n", ni+1, tc.elapsed().as_secs_f64());
                    break;
                }
            }
            if codes[0].len() == 6 { println!("    → no 100% up to 6 neurons ({:.1}s)\n", tc.elapsed().as_secs_f64()); }

            if tc.elapsed().as_secs() > 120 { println!("    (time limit)\n"); break; }
        }
    }

    // ══ Method 2: Backprop STE ══
    println!("=== METHOD 2: BACKPROP STE ===\n");

    fn c19_dx(x:f32,c:f32,r:f32)->f32{let c=c.max(0.1);let r=r.max(0.0);let l=6.0*c;
        if x>=l||x<=-l{return 1.0;}let s=x/c;let n=s.floor();let t=s-n;let h=t*(1.0-t);
        let sg=if(n as i32)%2==0{1.0}else{-1.0};(sg+2.0*r*h)*(1.0-2.0*t)}

    for m in 2..=4 {
        let tc = Instant::now();
        let mut best_score = 0;

        for seed in 0..200u64 {
            let mut rng2 = Rng::new(seed*7919+42);
            let mut w = [[0.0f32;4];6];
            let mut b = [0.0f32;6];
            let mut cv = [0.0f32;6];
            let mut rv = [0.0f32;6];
            for k in 0..m {
                for j in 0..4{w[k][j]=rng2.uniform(-15.0,15.0);}
                b[k]=rng2.uniform(-15.0,15.0);
                cv[k]=rng2.uniform(5.0,60.0);
                rv[k]=rng2.uniform(0.0,2.0);
            }

            for ep in 0..2000 {
                let lr=0.01*(1.0-ep as f32/2000.0*0.8);
                let wc=50.0f32;
                let qw:Vec<[i8;4]>=(0..m).map(|k|[
                    w[k][0].round().max(-wc).min(wc)as i8,w[k][1].round().max(-wc).min(wc)as i8,
                    w[k][2].round().max(-wc).min(wc)as i8,w[k][3].round().max(-wc).min(wc)as i8]).collect();
                let qb:Vec<i8>=(0..m).map(|k|b[k].round().max(-wc).min(wc)as i8).collect();

                let codes:Vec<Vec<f32>>=sampled.iter().map(|inp|{
                    (0..m).map(|k|{let dot=qb[k]as f32+qw[k][0]as f32*inp[0]+qw[k][1]as f32*inp[1]
                        +qw[k][2]as f32*inp[2]+qw[k][3]as f32*inp[3];
                        c19a(dot,cv[k],rv[k])}).collect()}).collect();

                let mut n_active=0u32;
                for i in 0..sample_size{let mut nj=0;let mut nd=f32::MAX;
                    for j in 0..sample_size{if j==i{continue;}
                        let d:f32=(0..m).map(|k|(codes[i][k]-codes[j][k]).powi(2)).sum();
                        if d<nd{nd=d;nj=j;}}
                    if nd<2.0{n_active+=1;
                        for k in 0..m{let sign=if codes[i][k]>codes[nj][k]{1.0}else if codes[i][k]<codes[nj][k]{-1.0}else{0.0};
                            let dot_i=qb[k]as f32+qw[k][0]as f32*sampled[i][0]+qw[k][1]as f32*sampled[i][1]
                                +qw[k][2]as f32*sampled[i][2]+qw[k][3]as f32*sampled[i][3];
                            let gi=c19_dx(dot_i,cv[k],rv[k]);
                            for j in 0..4{w[k][j]+=lr*sign*gi*sampled[i][j]*0.001;}
                            b[k]+=lr*sign*gi*0.001;
                            let eps=0.01;
                            cv[k]+=lr*sign*(c19a(dot_i,cv[k]+eps,rv[k])-c19a(dot_i,cv[k]-eps,rv[k]))/(2.0*eps)*0.0001;
                            cv[k]=cv[k].max(0.5).min(100.0);
                        }}}
                if n_active==0{break;}
            }

            // Eval
            let wc=50.0;
            let qw:Vec<[i8;4]>=(0..m).map(|k|[w[k][0].round().max(-wc).min(wc)as i8,
                w[k][1].round().max(-wc).min(wc)as i8,w[k][2].round().max(-wc).min(wc)as i8,
                w[k][3].round().max(-wc).min(wc)as i8]).collect();
            let qb:Vec<i8>=(0..m).map(|k|b[k].round().max(-wc).min(wc)as i8).collect();
            let codes:Vec<Vec<f32>>=sampled.iter().map(|inp|{
                (0..m).map(|k|{let dot=qb[k]as f32+qw[k][0]as f32*inp[0]+qw[k][1]as f32*inp[1]
                    +qw[k][2]as f32*inp[2]+qw[k][3]as f32*inp[3];
                    c19a(dot,cv[k],rv[k])}).collect()}).collect();
            let score=eval_roundtrip(&codes,m);
            if score>best_score{best_score=score;
                if score==sample_size{
                    println!("  M={}: {}/{} (100%) seed={} ({:.1}s)",m,score,sample_size,seed,tc.elapsed().as_secs_f64());
                    for k in 0..m{println!("    N{}: w={:?} b={} c={:.1} rho={:.1}",k,qw[k],qb[k],cv[k],rv[k]);}
                    break;}}

            if seed%50==49{println!("  M={}: best {}/{} after {} seeds ({:.1}s)",m,best_score,sample_size,seed+1,tc.elapsed().as_secs_f64());}
        }
        if best_score<sample_size{println!("  M={}: best {}/{} ({:.1}%)",m,best_score,sample_size,best_score as f64/sample_size as f64*100.0);}
        println!();
        if tc.elapsed().as_secs()>180{break;}
    }

    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
