//! L1 Merger — activation function sweep for int8 output
//!
//! M=2 neurons, int8 weights, various activations
//! Goal: find activation where all 729 byte-pairs get distinct int8 outputs
//!
//! Run: cargo run --example l1_merger_act_sweep --release

use std::time::Instant;

const LUT: [[i8; 2]; 27] = [
    [-2,-4],[-4,-2],[0,-6],[-2,-5],[-1,-6],[-3,-5],[1,-8],[-1,-7],
    [-2,-6],[-4,-5],[0,-8],[-2,-7],[-1,-8],[-3,-7],[1,-10],[-1,-9],
    [-4,-6],[-6,-5],[-2,-8],[-4,-7],[-3,-8],[-5,-7],[-1,-10],[-3,-9],
    [-4,-8],[-6,-7],[-2,-10],
];

fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.1); let rho = rho.max(0.0); let l = 6.0 * c;
    if x >= l { return x - l; } if x <= -l { return x + l; }
    let s = x / c; let n = s.floor(); let t = s - n; let h = t * (1.0 - t);
    let sg = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    c * (sg * h + rho * h * h)
}

fn main() {
    let t0 = Instant::now();

    let inputs: Vec<[f32; 4]> = (0..27u8).flat_map(|a| {
        (0..27u8).map(move |b| [
            LUT[a as usize][0] as f32, LUT[a as usize][1] as f32,
            LUT[b as usize][0] as f32, LUT[b as usize][1] as f32,
        ])
    }).collect();

    println!("=== L1 MERGER — ACTIVATION SWEEP (M=2, int8 output) ===\n");
    println!("  729 byte-pairs, int8 weights [-50,+50], 2 neurons\n");

    // Weight range for exhaustive
    let range: Vec<i8> = (-20..=20).collect();
    let nv = range.len(); // 41
    let combos = nv.pow(3); // per neuron: 2 weights + 1 bias (using LUT input, so 4 weights is too many for exhaustive)

    // For speed: use small weight range for exhaustive, greedy neuron-by-neuron
    let range_small: Vec<i8> = (-10..=10).collect();
    let nvs = range_small.len(); // 21
    let combos_s = nvs.pow(5); // 4 weights + 1 bias = 21^5 = 4M — too many!

    // Use backprop instead for all activations
    println!("  Using backprop (100 seeds x 2000 epochs per activation)\n");

    // Activation functions
    struct Act {
        name: &'static str,
        // Will be called as act(dot, param)
        // param is a learnable float
    }

    let act_names = [
        "C19",
        "linear",
        "ReLU",
        "sin",
        "tanh",
        "mod256",
        "triangle",
        "sawtooth",
    ];

    fn apply_act(name: &str, x: f32, c: f32, rho: f32) -> f32 {
        match name {
            "C19" => c19(x, c, rho),
            "linear" => x,
            "ReLU" => x.max(0.0),
            "sin" => (x / c.max(1.0)).sin() * 127.0,
            "tanh" => (x / c.max(1.0)).tanh() * 127.0,
            "mod256" => ((x as i32).rem_euclid(256)) as f32 - 128.0,
            "triangle" => {
                let p = c.max(1.0) * 2.0;
                let t = (x / p - (x / p).floor()) * 2.0 - 1.0;
                t * 127.0
            },
            "sawtooth" => {
                let p = c.max(1.0);
                ((x / p) - (x / p).floor() - 0.5) * 254.0
            },
            _ => x,
        }
    }

    fn apply_act_dx(name: &str, x: f32, c: f32, rho: f32) -> f32 {
        match name {
            "C19" => {
                let c = c.max(0.1); let l = 6.0*c;
                if x >= l || x <= -l { return 1.0; }
                let s = x/c; let n = s.floor(); let t = s-n; let h = t*(1.0-t);
                let sg = if (n as i32)%2==0{1.0}else{-1.0};
                (sg + 2.0*rho.max(0.0)*h) * (1.0 - 2.0*t)
            },
            "linear" => 1.0,
            "ReLU" => if x > 0.0 { 1.0 } else { 0.0 },
            "sin" => (x / c.max(1.0)).cos() * 127.0 / c.max(1.0),
            "tanh" => { let t = (x/c.max(1.0)).tanh(); (1.0-t*t)*127.0/c.max(1.0) },
            "mod256" => 1.0, // STE
            "triangle" => 127.0 * 2.0 / (c.max(1.0) * 2.0), // approximate
            "sawtooth" => 254.0 / c.max(1.0),
            _ => 1.0,
        }
    }

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self { Rng(seed.wrapping_mul(6364136223846793005).wrapping_add(1)) }
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0
        }
        fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
            lo + ((self.next() >> 33) % 65536) as f32 / 65536.0 * (hi - lo)
        }
    }

    println!("  {:>12} {:>12} {:>12} {:>8}", "activation", "float_best", "int8_best", "time");
    println!("  {}", "-".repeat(50));

    for &act_name in &act_names {
        let tc = Instant::now();
        let mut best_float = 0usize;
        let mut best_int8 = 0usize;

        for seed in 0..100u64 {
            let mut rng = Rng::new(seed * 7919 + 42);
            let mut w = [[0.0f32; 4]; 2];
            let mut b = [0.0f32; 2];
            let mut cv = [0.0f32; 2];
            let mut rv = [0.0f32; 2];

            for k in 0..2 {
                for j in 0..4 { w[k][j] = rng.uniform(-15.0, 15.0); }
                b[k] = rng.uniform(-15.0, 15.0);
                cv[k] = rng.uniform(5.0, 60.0);
                rv[k] = rng.uniform(0.0, 2.0);
            }

            for ep in 0..2000 {
                let lr = 0.02 * (1.0 - ep as f32 / 2000.0 * 0.8);
                let wc = 50.0f32;

                // Quantize weights
                let qw: [[i8;4];2] = [
                    [w[0][0].round().max(-wc).min(wc) as i8, w[0][1].round().max(-wc).min(wc) as i8,
                     w[0][2].round().max(-wc).min(wc) as i8, w[0][3].round().max(-wc).min(wc) as i8],
                    [w[1][0].round().max(-wc).min(wc) as i8, w[1][1].round().max(-wc).min(wc) as i8,
                     w[1][2].round().max(-wc).min(wc) as i8, w[1][3].round().max(-wc).min(wc) as i8],
                ];
                let qb = [b[0].round().max(-wc).min(wc) as i8, b[1].round().max(-wc).min(wc) as i8];

                let codes: Vec<[f32;2]> = inputs.iter().map(|inp| {
                    let mut o = [0.0f32;2];
                    for k in 0..2 {
                        let dot = qb[k] as f32 + qw[k][0] as f32*inp[0] + qw[k][1] as f32*inp[1]
                            + qw[k][2] as f32*inp[2] + qw[k][3] as f32*inp[3];
                        o[k] = apply_act(act_name, dot, cv[k], rv[k]);
                    }
                    o
                }).collect();

                let mut n_active = 0u32;
                for i in 0..729 {
                    let mut nj=0;let mut nd=f32::MAX;
                    for j in 0..729{if j==i{continue;}
                        let d=(codes[i][0]-codes[j][0]).powi(2)+(codes[i][1]-codes[j][1]).powi(2);
                        if d<nd{nd=d;nj=j;}}
                    if nd < 2.0 {
                        n_active += 1;
                        for k in 0..2 {
                            let sign = if codes[i][k]>codes[nj][k]{1.0}else if codes[i][k]<codes[nj][k]{-1.0}else{0.0};
                            let dot_i = qb[k] as f32 + qw[k][0] as f32*inputs[i][0]+qw[k][1] as f32*inputs[i][1]
                                +qw[k][2] as f32*inputs[i][2]+qw[k][3] as f32*inputs[i][3];
                            let gi = apply_act_dx(act_name, dot_i, cv[k], rv[k]);
                            for j in 0..4 { w[k][j] += lr*sign*gi*inputs[i][j]*0.001; }
                            b[k] += lr*sign*gi*0.001;
                            // c gradient
                            let eps=0.01;
                            let dc=(apply_act(act_name,dot_i,cv[k]+eps,rv[k])-apply_act(act_name,dot_i,cv[k]-eps,rv[k]))/(2.0*eps);
                            cv[k]+=lr*sign*dc*0.0001;cv[k]=cv[k].max(0.5).min(100.0);
                        }
                    }
                }
                if n_active == 0 { break; }
            }

            // Eval float
            let wc=50.0;
            let qw:[[i8;4];2]=[
                [w[0][0].round().max(-wc).min(wc) as i8,w[0][1].round().max(-wc).min(wc) as i8,
                 w[0][2].round().max(-wc).min(wc) as i8,w[0][3].round().max(-wc).min(wc) as i8],
                [w[1][0].round().max(-wc).min(wc) as i8,w[1][1].round().max(-wc).min(wc) as i8,
                 w[1][2].round().max(-wc).min(wc) as i8,w[1][3].round().max(-wc).min(wc) as i8]];
            let qb=[b[0].round().max(-wc).min(wc) as i8,b[1].round().max(-wc).min(wc) as i8];

            let codes: Vec<[f32;2]> = inputs.iter().map(|inp| {
                let mut o=[0.0f32;2];
                for k in 0..2 {
                    let dot=qb[k] as f32+qw[k][0] as f32*inp[0]+qw[k][1] as f32*inp[1]
                        +qw[k][2] as f32*inp[2]+qw[k][3] as f32*inp[3];
                    o[k]=apply_act(act_name,dot,cv[k],rv[k]);
                }o}).collect();

            let mut ok_f=0;
            for i in 0..729{let mut best=0;let mut bd=f32::MAX;
                for j in 0..729{let d=(codes[i][0]-codes[j][0]).powi(2)+(codes[i][1]-codes[j][1]).powi(2);
                    if d<bd{bd=d;best=j;}}
                if best==i{ok_f+=1;}}
            if ok_f > best_float { best_float = ok_f; }

            // Eval int8 output
            let mut mn0=f32::MAX;let mut mx0=f32::MIN;let mut mn1=f32::MAX;let mut mx1=f32::MIN;
            for c in &codes{if c[0]<mn0{mn0=c[0];}if c[0]>mx0{mx0=c[0];}if c[1]<mn1{mn1=c[1];}if c[1]>mx1{mx1=c[1];}}
            let s0=if mx0-mn0>0.0{254.0/(mx0-mn0)}else{1.0};
            let s1=if mx1-mn1>0.0{254.0/(mx1-mn1)}else{1.0};

            let qlut:Vec<[i8;2]>=codes.iter().map(|c|{
                [((c[0]-mn0)*s0-127.0).round().max(-128.0).min(127.0) as i8,
                 ((c[1]-mn1)*s1-127.0).round().max(-128.0).min(127.0) as i8]
            }).collect();

            let mut ok_q=0;
            for i in 0..729{let mut best=0;let mut bd=i32::MAX;
                for j in 0..729{let d=(qlut[i][0] as i32-qlut[j][0] as i32).pow(2)+(qlut[i][1] as i32-qlut[j][1] as i32).pow(2);
                    if d<bd{bd=d;best=j;}}
                if best==i{ok_q+=1;}}
            if ok_q > best_int8 { best_int8 = ok_q; }

            if ok_q == 729 {
                println!("  {:>12}     729/729      729/729 {:>7.1}s ***", act_name, tc.elapsed().as_secs_f64());
                println!("    N0: w={:?} b={} c={:.1}", qw[0], qb[0], cv[0]);
                println!("    N1: w={:?} b={} c={:.1}", qw[1], qb[1], cv[1]);
                break;
            }
        }

        if best_int8 < 729 {
            let m = if best_int8 > 720 { " !!" } else if best_int8 > 700 { " !" } else { "" };
            println!("  {:>12} {:>8}/729 {:>8}/729 {:>7.1}s{}",
                act_name, best_float, best_int8, tc.elapsed().as_secs_f64(), m);
        }
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
