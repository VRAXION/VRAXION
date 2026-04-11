//! VRAXION Video Upscaler — Proof of Concept
//! EP settle, 1 hidden layer: 32→64→16
//! Using PROVEN stable EP code from cortex experiments
//!
//! Run: cargo run --example video_upscale_poc --release

use std::time::Instant;

fn c19(x: f32, rho: f32) -> f32 {
    let l = 6.0f32;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * h + rho * h * h
}

struct Rng { state: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { state: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 { self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.state }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn range_f32(&mut self, lo: f32, hi: f32) -> f32 { lo + self.f32() * (hi - lo) }
    fn usize(&mut self, n: usize) -> usize { (self.next() as usize) % n }
    fn shuffle<T>(&mut self, v: &mut [T]) { for i in (1..v.len()).rev() { let j = self.usize(i + 1); v.swap(i, j); } }
}

const RHO: f32 = 8.0;
const IN_DIM: usize = 32;  // 16 LR + 16 prev HR
const H_DIM: usize = 64;
const OUT_DIM: usize = 16; // 4×4 HR

// 1-hidden-layer EP (same proven pattern as cortex tests)
struct EpNet {
    w1: Vec<f32>, w2: Vec<f32>, b1: Vec<f32>, b2: Vec<f32>,
}

impl EpNet {
    fn new(rng: &mut Rng) -> Self {
        let s1 = (2.0 / IN_DIM as f32).sqrt() * 0.5; // smaller init for stability
        let s2 = (2.0 / H_DIM as f32).sqrt() * 0.5;
        EpNet {
            w1: (0..H_DIM * IN_DIM).map(|_| rng.range_f32(-s1, s1)).collect(),
            w2: (0..OUT_DIM * H_DIM).map(|_| rng.range_f32(-s2, s2)).collect(),
            b1: vec![0.0; H_DIM], b2: vec![0.0; OUT_DIM],
        }
    }
    fn weight_count(&self) -> usize { self.w1.len() + self.w2.len() + self.b1.len() + self.b2.len() }
}

fn settle_step(sh: &[f32], so: &[f32], x: &[f32], net: &EpNet, dt: f32, beta: f32, y: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let act = |v: f32| c19(v, RHO);
    let mut nh = vec![0.0f32; H_DIM];
    for j in 0..H_DIM {
        let mut d = net.b1[j];
        for i in 0..IN_DIM { d += net.w1[j * IN_DIM + i] * x[i]; }
        for k in 0..OUT_DIM { d += net.w2[k * H_DIM + j] * act(so[k]); }
        nh[j] = sh[j] + dt * (-sh[j] + d);
    }
    let mut no = vec![0.0f32; OUT_DIM];
    for k in 0..OUT_DIM {
        let mut d = net.b2[k];
        for j in 0..H_DIM { d += net.w2[k * H_DIM + j] * act(sh[j]); }
        no[k] = so[k] + dt * (-so[k] + d + beta * (y[k] - act(so[k])));
    }
    (nh, no)
}

fn settle(x: &[f32], net: &EpNet, ticks: usize, dt: f32, beta: f32, y: &[f32], init_out: Option<&[f32]>) -> (Vec<f32>, Vec<f32>) {
    let mut sh = vec![0.0f32; H_DIM];
    let mut so = init_out.map(|v| v.to_vec()).unwrap_or_else(|| vec![0.0f32; OUT_DIM]);
    for _ in 0..ticks { let (h, o) = settle_step(&sh, &so, x, net, dt, beta, y); sh = h; so = o; }
    (sh, so)
}

fn predict(x: &[f32], net: &EpNet, ticks: usize, dt: f32, init_out: Option<&[f32]>) -> Vec<f32> {
    let dummy = vec![0.0f32; OUT_DIM];
    let (_, so) = settle(x, net, ticks, dt, 0.0, &dummy, init_out);
    so.iter().map(|&s| c19(s, RHO)).collect()
}

fn train(net: &mut EpNet, data: &[(Vec<f32>, Vec<f32>)], ticks: usize, dt: f32, beta: f32, lr: f32, epochs: usize, rng: &mut Rng) {
    let act = |v: f32| c19(v, RHO);
    let mut idx: Vec<usize> = (0..data.len()).collect();
    let clip = 0.05f32;

    for ep in 0..epochs {
        let lr_e = if ep < 20 { lr * (ep as f32 + 1.0) / 20.0 } else { lr };
        rng.shuffle(&mut idx);
        for &i in &idx {
            let (x, y) = &data[i];
            let (sfh, sfo) = settle(x, net, ticks, dt, 0.0, y, None);
            let mut snh = sfh.clone(); let mut sno = sfo.clone();
            for _ in 0..ticks { let (h, o) = settle_step(&snh, &sno, x, net, dt, beta, y); snh = h; sno = o; }

            let ib = 1.0 / beta;
            for j in 0..H_DIM {
                let an = act(snh[j]); let af = act(sfh[j]);
                if an.is_nan() || af.is_nan() { continue; }
                for ii in 0..IN_DIM {
                    net.w1[j * IN_DIM + ii] += (lr_e * ib * (an * x[ii] - af * x[ii])).clamp(-clip, clip);
                }
                net.b1[j] += (lr_e * ib * (an - af)).clamp(-clip, clip);
            }
            for k in 0..OUT_DIM {
                let aon = act(sno[k]); let aof = act(sfo[k]);
                if aon.is_nan() || aof.is_nan() { continue; }
                for j in 0..H_DIM {
                    let ahn = act(snh[j]); let ahf = act(sfh[j]);
                    if ahn.is_nan() || ahf.is_nan() { continue; }
                    net.w2[k * H_DIM + j] += (lr_e * ib * (aon * ahn - aof * ahf)).clamp(-clip, clip);
                }
                net.b2[k] += (lr_e * ib * (aon - aof)).clamp(-clip, clip);
            }
        }
        if ep % 50 == 0 || ep == epochs - 1 {
            let mut mse = 0.0f32;
            let mut n_valid = 0;
            for (x, y) in data {
                let pred = predict(x, net, ticks, dt, None);
                if pred.iter().any(|v| v.is_nan()) { continue; }
                mse += pred.iter().zip(y).map(|(p, t)| (p - t) * (p - t)).sum::<f32>() / OUT_DIM as f32;
                n_valid += 1;
            }
            if n_valid > 0 {
                mse /= n_valid as f32;
                let psnr = if mse > 1e-10 { -10.0 * mse.log10() } else { 99.0 };
                println!("    Epoch {:4}: MSE={:.6}, PSNR={:.1} dB ({}/{})", ep, mse, psnr, n_valid, data.len());
            } else {
                println!("    Epoch {:4}: ALL NaN", ep);
            }
        }
    }
}

// Image generation
fn gen_hr_image(rng: &mut Rng, pattern: usize) -> Vec<f32> {
    let mut img = vec![0.0f32; 64];
    match pattern % 6 {
        0 => { for y in 0..8 { for x in 0..8 { img[y*8+x] = x as f32 / 7.0; }}}
        1 => { for y in 0..8 { for x in 0..8 { img[y*8+x] = y as f32 / 7.0; }}}
        2 => { for y in 0..8 { for x in 0..8 { img[y*8+x] = if (x+y)%2==0 {0.8} else {0.2}; }}}
        3 => { for y in 0..8 { for x in 0..8 { img[y*8+x] = if x>y {0.9} else {0.1}; }}}
        4 => { for y in 0..8 { for x in 0..8 { let d=((x as f32-3.5).powi(2)+(y as f32-3.5).powi(2)).sqrt(); img[y*8+x]=if d<3.0{0.8}else{0.2}; }}}
        _ => { for i in 0..64 { img[i] = rng.f32() * 0.8 + 0.1; }}
    }
    for i in 0..64 { img[i] = (img[i] + rng.range_f32(-0.03, 0.03)).clamp(0.0, 1.0); }
    img
}

fn downsample(hr: &[f32]) -> Vec<f32> {
    let mut lr = vec![0.0f32; 16];
    for ty in 0..4 { for tx in 0..4 {
        let mut s = 0.0f32;
        for dy in 0..2 { for dx in 0..2 { s += hr[(ty*2+dy)*8+(tx*2+dx)]; }}
        lr[ty*4+tx] = s / 4.0;
    }}
    lr
}

fn extract_hr_tile(hr: &[f32]) -> Vec<f32> {
    let mut t = vec![0.0f32; 16];
    for y in 0..4 { for x in 0..4 { t[y*4+x] = hr[(y+2)*8+(x+2)]; }}
    t
}

fn build_input(lr: &[f32], prev: &[f32]) -> Vec<f32> {
    let mut v = Vec::with_capacity(32);
    // Scale to [-0.5, 0.5] for EP stability
    v.extend(lr.iter().map(|&x| x - 0.5));
    v.extend(prev.iter().map(|&x| x - 0.5));
    v
}

fn compute_mse(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x,y)|(x-y)*(x-y)).sum::<f32>() / a.len() as f32 }
fn psnr(mse: f32) -> f32 { if mse > 1e-10 { -10.0 * mse.log10() } else { 99.0 } }

fn bilinear(lr: &[f32]) -> Vec<f32> {
    let mut hr = vec![0.0f32; 16];
    for y in 0..4 { for x in 0..4 {
        let sx = (x as f32+0.5)/4.0*4.0-0.5; let sy = (y as f32+0.5)/4.0*4.0-0.5;
        let x0 = sx.floor().max(0.0) as usize; let y0 = sy.floor().max(0.0) as usize;
        let x1 = (x0+1).min(3); let y1 = (y0+1).min(3);
        let fx = sx - sx.floor(); let fy = sy - sy.floor();
        hr[y*4+x] = lr[y0*4+x0]*(1.0-fx)*(1.0-fy) + lr[y0*4+x1]*fx*(1.0-fy) + lr[y1*4+x0]*(1.0-fx)*fy + lr[y1*4+x1]*fx*fy;
    }}
    hr
}

fn main() {
    println!("================================================================");
    println!("  VRAXION Video Upscaler — POC (1 hidden layer, stable EP)");
    println!("  32→64→16, temporal feedback, C19 rho=8");
    println!("================================================================\n");

    let ticks = 20;
    let dt = 0.5;
    let beta = 0.5;
    let lr = 0.005;
    let epochs = 300;

    let mut rng = Rng::new(42);
    let mut net = EpNet::new(&mut rng);
    println!("  Network: {}→{}→{}, weights: {} ({:.1} KB)", IN_DIM, H_DIM, OUT_DIM, net.weight_count(), net.weight_count() as f32/1024.0);
    println!("  Settle: {} ticks, dt={}, beta={}, lr={}\n", ticks, dt, beta, lr);

    // Generate data
    let mut data: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    for i in 0..300 {
        let hr = gen_hr_image(&mut rng, i);
        let lr = downsample(&hr);
        let hr_tile = extract_hr_tile(&hr);
        // Target also shifted to [-0.5, 0.5]
        let target: Vec<f32> = hr_tile.iter().map(|&x| x - 0.5).collect();
        // Cold start
        data.push((build_input(&lr, &vec![0.5f32; 16]), target.clone()));
        // Warm start (prev = noisy version of target)
        let prev: Vec<f32> = hr_tile.iter().map(|&x| (x + rng.range_f32(-0.1, 0.1)).clamp(0.0, 1.0)).collect();
        data.push((build_input(&lr, &prev), target));
    }
    println!("  Training data: {} pairs\n", data.len());

    let t0 = Instant::now();
    train(&mut net, &data, ticks, dt, beta, lr, epochs, &mut rng);
    println!("  Train time: {:.1}s\n", t0.elapsed().as_secs_f64());

    // Test spatial
    println!("  ── Spatial Test (50 images) ──");
    let (mut v_mse, mut b_mse) = (0.0f32, 0.0f32);
    let n_test = 50;
    for i in 0..n_test {
        let hr = gen_hr_image(&mut rng, i + 5000);
        let lr = downsample(&hr); let hr_tile = extract_hr_tile(&hr);
        let pred = predict(&build_input(&lr, &vec![0.5f32;16]), &net, ticks, dt, None);
        let pred_shifted: Vec<f32> = pred.iter().map(|x| x + 0.5).collect();
        if pred_shifted.iter().any(|v| v.is_nan()) { continue; }
        v_mse += compute_mse(&pred_shifted, &hr_tile);
        b_mse += compute_mse(&bilinear(&lr), &hr_tile);
    }
    v_mse /= n_test as f32; b_mse /= n_test as f32;
    println!("    Bilinear: {:.1} dB", psnr(b_mse));
    println!("    VRAXION:  {:.1} dB ({:+.1} vs bilinear)\n", psnr(v_mse), psnr(v_mse) - psnr(b_mse));

    // Test temporal
    println!("  ── Temporal Test (20 sequences × 5 frames) ──");
    let mut fpsnr = vec![0.0f32; 5];
    let mut bl_avg = 0.0f32;
    for _ in 0..20 {
        let bp = rng.usize(100);
        let mut prev: Option<Vec<f32>> = None;
        for f in 0..5 {
            let hr = gen_hr_image(&mut rng, bp + f); let lr = downsample(&hr); let hr_tile = extract_hr_tile(&hr);
            let pv = prev.clone().unwrap_or(vec![0.5f32; 16]);
            let pred = predict(&build_input(&lr, &pv), &net, ticks, dt, prev.as_deref());
            let ps: Vec<f32> = pred.iter().map(|x| x+0.5).collect();
            if !ps.iter().any(|v| v.is_nan()) {
                fpsnr[f] += psnr(compute_mse(&ps, &hr_tile));
                prev = Some(pred);
            } else { prev = None; }
            bl_avg += psnr(compute_mse(&bilinear(&lr), &hr_tile));
        }
    }
    bl_avg /= 100.0;
    for f in 0..5 { fpsnr[f] /= 20.0; println!("    Frame {}: {:.1} dB ({:+.1} vs bilinear)", f+1, fpsnr[f], fpsnr[f]-bl_avg); }
    println!("    Temporal gain: {:+.1} dB\n", fpsnr[4] - fpsnr[0]);

    // Speed
    let dummy = build_input(&vec![0.5f32;16], &vec![0.5f32;16]);
    let t = Instant::now();
    for _ in 0..1000 { let _ = predict(&dummy, &net, ticks, dt, None); }
    let per_tile = t.elapsed().as_secs_f64() / 1000.0;
    let fps_1 = 1.0 / (per_tile * 32400.0);
    println!("  ── Speed ──");
    println!("    {:.0} µs/tile, {:.1} FPS (1 core), {:.1} FPS (4 core)\n", per_tile*1e6, fps_1, fps_1*4.0);

    println!("================================================================");
    if psnr(v_mse) > psnr(b_mse) + 0.5 { println!("  ✓ VRAXION beats bilinear by {:.1} dB", psnr(v_mse)-psnr(b_mse)); }
    else { println!("  ⚠ VRAXION marginal vs bilinear ({:+.1} dB)", psnr(v_mse)-psnr(b_mse)); }
    if fpsnr[4] > fpsnr[0] + 0.3 { println!("  ✓ Temporal feedback helps ({:+.1} dB gain)", fpsnr[4]-fpsnr[0]); }
    else { println!("  ⚠ Temporal feedback minimal ({:+.1} dB)", fpsnr[4]-fpsnr[0]); }
    println!("================================================================");
}
