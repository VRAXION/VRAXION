//! VRAXION Video Upscaler — Backprop vs EP comparison
//! Tests 1, 2, 3 hidden layers with standard backprop
//! Same data as video_upscale_poc.rs but MLP + backprop training
//!
//! Run: cargo run --example video_upscale_backprop --release

use std::time::Instant;

fn c19(x: f32, rho: f32) -> f32 {
    let l = 6.0f32;
    if x >= l { return x - l; }
    if x <= -l { return x + l; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    sgn * h + rho * h * h
}

fn c19_deriv(x: f32, rho: f32) -> f32 {
    let l = 6.0f32;
    if x >= l || x <= -l { return 1.0; }
    let n = x.floor(); let t = x - n; let h = t * (1.0 - t);
    let sgn = if (n as i32) % 2 == 0 { 1.0 } else { -1.0 };
    let dt = 1.0 - 2.0 * t;
    dt * (sgn + 2.0 * rho * h)
}

fn relu(x: f32) -> f32 { x.max(0.0) }
fn relu_deriv(x: f32) -> f32 { if x > 0.0 { 1.0 } else { 0.0 } }

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
const OUT_DIM: usize = 16; // 4×4 HR tile

// Generic MLP: variable number of hidden layers
struct Mlp {
    layers: Vec<usize>,       // [in, h1, h2, ..., out]
    weights: Vec<Vec<f32>>,   // per layer: rows × cols (output × input)
    biases: Vec<Vec<f32>>,    // per layer
    use_c19: bool,
}

impl Mlp {
    fn new(layers: &[usize], use_c19: bool, rng: &mut Rng) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        for l in 0..layers.len()-1 {
            let fan_in = layers[l];
            let fan_out = layers[l+1];
            let s = (2.0 / fan_in as f32).sqrt() * if use_c19 { 0.3 } else { 1.0 };
            weights.push((0..fan_out * fan_in).map(|_| rng.range_f32(-s, s)).collect());
            biases.push(vec![0.0; fan_out]);
        }
        Mlp { layers: layers.to_vec(), weights, biases, use_c19 }
    }

    fn weight_count(&self) -> usize {
        self.weights.iter().zip(&self.biases).map(|(w,b)| w.len() + b.len()).sum()
    }

    fn act(&self, x: f32) -> f32 { if self.use_c19 { c19(x, RHO) } else { relu(x) } }
    fn act_d(&self, x: f32) -> f32 { if self.use_c19 { c19_deriv(x, RHO) } else { relu_deriv(x) } }

    // Forward: returns (pre_activations, activations) per layer
    fn forward(&self, input: &[f32]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let n_layers = self.layers.len() - 1;
        let mut pres = Vec::with_capacity(n_layers);
        let mut acts = Vec::with_capacity(n_layers + 1);
        acts.push(input.to_vec());

        for l in 0..n_layers {
            let inp = &acts[l];
            let out_dim = self.layers[l+1];
            let in_dim = self.layers[l];
            let mut pre = vec![0.0f32; out_dim];
            for j in 0..out_dim {
                let mut s = self.biases[l][j];
                for i in 0..in_dim {
                    s += self.weights[l][j * in_dim + i] * inp[i];
                }
                pre[j] = s;
            }
            pres.push(pre.clone());

            // Last layer: linear output (no activation) for regression
            let act: Vec<f32> = if l == n_layers - 1 {
                pre
            } else {
                pre.iter().map(|&x| self.act(x)).collect()
            };
            acts.push(act);
        }
        (pres, acts)
    }

    fn predict(&self, input: &[f32]) -> Vec<f32> {
        let (_, acts) = self.forward(input);
        acts.last().unwrap().clone()
    }

    // Backprop: returns gradients for all weights and biases
    fn backward(&self, pres: &[Vec<f32>], acts: &[Vec<f32>], target: &[f32]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let n_layers = self.layers.len() - 1;
        let mut dw: Vec<Vec<f32>> = self.weights.iter().map(|w| vec![0.0; w.len()]).collect();
        let mut db: Vec<Vec<f32>> = self.biases.iter().map(|b| vec![0.0; b.len()]).collect();

        // Output error (MSE gradient): d_loss/d_output = 2*(output - target) / N
        let out = &acts[n_layers];
        let mut delta: Vec<f32> = out.iter().zip(target).map(|(&o, &t)| 2.0 * (o - t) / OUT_DIM as f32).collect();

        // Backprop through layers
        for l in (0..n_layers).rev() {
            let inp = &acts[l];
            let in_dim = self.layers[l];
            let out_dim = self.layers[l+1];

            // If not output layer, multiply delta by activation derivative
            if l < n_layers - 1 {
                for j in 0..out_dim {
                    delta[j] *= self.act_d(pres[l][j]);
                }
            }

            // Gradient for weights and biases
            for j in 0..out_dim {
                for i in 0..in_dim {
                    dw[l][j * in_dim + i] += delta[j] * inp[i];
                }
                db[l][j] += delta[j];
            }

            // Propagate delta to previous layer
            if l > 0 {
                let prev_dim = in_dim;
                let mut new_delta = vec![0.0f32; prev_dim];
                for i in 0..prev_dim {
                    for j in 0..out_dim {
                        new_delta[i] += self.weights[l][j * in_dim + i] * delta[j];
                    }
                }
                delta = new_delta;
            }
        }
        (dw, db)
    }

    fn train(&mut self, data: &[(Vec<f32>, Vec<f32>)], lr: f32, epochs: usize, rng: &mut Rng) {
        let mut idx: Vec<usize> = (0..data.len()).collect();
        let clip = 1.0f32;

        for ep in 0..epochs {
            let lr_e = if ep < 20 { lr * (ep as f32 + 1.0) / 20.0 } else { lr };
            rng.shuffle(&mut idx);

            for &i in &idx {
                let (x, y) = &data[i];
                let (pres, acts) = self.forward(x);
                let (dw, db) = self.backward(&pres, &acts, y);

                for l in 0..self.weights.len() {
                    for k in 0..self.weights[l].len() {
                        self.weights[l][k] -= (lr_e * dw[l][k]).clamp(-clip, clip);
                    }
                    for k in 0..self.biases[l].len() {
                        self.biases[l][k] -= (lr_e * db[l][k]).clamp(-clip, clip);
                    }
                }
            }

            if ep % 100 == 0 || ep == epochs - 1 {
                let mut mse = 0.0f32;
                for (x, y) in data {
                    let pred = self.predict(x);
                    if pred.iter().any(|v| v.is_nan()) { println!("    Epoch {:4}: NaN!", ep); return; }
                    mse += pred.iter().zip(y).map(|(p, t)| (p - t) * (p - t)).sum::<f32>() / OUT_DIM as f32;
                }
                mse /= data.len() as f32;
                let psnr = if mse > 1e-10 { -10.0 * mse.log10() } else { 99.0 };
                println!("    Epoch {:4}: MSE={:.6}, PSNR={:.1} dB", ep, mse, psnr);
            }
        }
    }
}

// ── Image gen (same as POC) ──

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
    v.extend(lr.iter().map(|&x| x - 0.5));
    v.extend(prev.iter().map(|&x| x - 0.5));
    v
}

fn compute_mse(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x,y)|(x-y)*(x-y)).sum::<f32>() / a.len() as f32 }
fn psnr_val(mse: f32) -> f32 { if mse > 1e-10 { -10.0 * mse.log10() } else { 99.0 } }

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

fn test_config(name: &str, layers: &[usize], use_c19: bool, data: &[(Vec<f32>, Vec<f32>)], test_rng_seed: u64) {
    let act_name = if use_c19 { "C19" } else { "ReLU" };
    let arch: Vec<String> = layers.iter().map(|d| d.to_string()).collect();
    println!("\n  ── {} ({}, {}) ──", name, arch.join("→"), act_name);

    let mut rng = Rng::new(42);
    let mut net = Mlp::new(layers, use_c19, &mut rng);
    println!("    Weights: {} ({:.1} KB)", net.weight_count(), net.weight_count() as f32 * 4.0 / 1024.0);

    let lr = if use_c19 { 0.001 } else { 0.005 };
    let epochs = 500;
    println!("    lr={}, epochs={}\n", lr, epochs);

    let t0 = Instant::now();
    net.train(data, lr, epochs, &mut rng);
    println!("    Train: {:.1}s", t0.elapsed().as_secs_f64());

    // Spatial test
    let mut trng = Rng::new(test_rng_seed);
    let (mut v_mse, mut b_mse) = (0.0f32, 0.0f32);
    let n_test = 50;
    let mut nan_count = 0;
    for i in 0..n_test {
        let hr = gen_hr_image(&mut trng, i + 5000);
        let lr = downsample(&hr); let hr_tile = extract_hr_tile(&hr);
        let pred = net.predict(&build_input(&lr, &vec![0.5f32;16]));
        let pred_shifted: Vec<f32> = pred.iter().map(|x| x + 0.5).collect();
        if pred_shifted.iter().any(|v| v.is_nan()) { nan_count += 1; continue; }
        v_mse += compute_mse(&pred_shifted, &hr_tile);
        b_mse += compute_mse(&bilinear(&lr), &hr_tile);
    }
    if nan_count > 0 { println!("    ⚠ {} NaN predictions", nan_count); }
    let valid = n_test - nan_count;
    if valid > 0 {
        v_mse /= valid as f32; b_mse /= valid as f32;
        println!("    Bilinear: {:.1} dB", psnr_val(b_mse));
        println!("    Backprop: {:.1} dB ({:+.1} vs bilinear)", psnr_val(v_mse), psnr_val(v_mse) - psnr_val(b_mse));
    }

    // Temporal test
    let mut trng2 = Rng::new(test_rng_seed + 1000);
    let mut fpsnr = vec![0.0f32; 5];
    let mut bl_avg = 0.0f32;
    for _ in 0..20 {
        let bp = trng2.usize(100);
        let mut prev_pred: Option<Vec<f32>> = None;
        for f in 0..5 {
            let hr = gen_hr_image(&mut trng2, bp + f);
            let lr = downsample(&hr); let hr_tile = extract_hr_tile(&hr);
            let pv = prev_pred.clone().unwrap_or(vec![0.5f32; 16]);
            let pred = net.predict(&build_input(&lr, &pv));
            let ps: Vec<f32> = pred.iter().map(|x| x+0.5).collect();
            if !ps.iter().any(|v| v.is_nan()) {
                fpsnr[f] += psnr_val(compute_mse(&ps, &hr_tile));
                prev_pred = Some(pred);
            } else { prev_pred = None; }
            bl_avg += psnr_val(compute_mse(&bilinear(&lr), &hr_tile));
        }
    }
    bl_avg /= 100.0;
    println!("    Temporal: F1={:.1}, F5={:.1} dB (gain: {:+.1})", fpsnr[0]/20.0, fpsnr[4]/20.0, fpsnr[4]/20.0 - fpsnr[0]/20.0);

    // Speed
    let dummy = build_input(&vec![0.5f32;16], &vec![0.5f32;16]);
    let t = Instant::now();
    for _ in 0..1000 { let _ = net.predict(&dummy); }
    let per_tile = t.elapsed().as_secs_f64() / 1000.0;
    println!("    Speed: {:.0} µs/tile", per_tile * 1e6);
}

fn main() {
    println!("================================================================");
    println!("  VRAXION Video Upscaler — Backprop Depth Sweep");
    println!("  Testing 1/2/3 hidden layers × C19/ReLU");
    println!("================================================================");

    let mut rng = Rng::new(42);

    // Generate training data (same as POC)
    let mut data: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    for i in 0..300 {
        let hr = gen_hr_image(&mut rng, i);
        let lr = downsample(&hr);
        let hr_tile = extract_hr_tile(&hr);
        let target: Vec<f32> = hr_tile.iter().map(|&x| x - 0.5).collect();
        data.push((build_input(&lr, &vec![0.5f32; 16]), target.clone()));
        let prev: Vec<f32> = hr_tile.iter().map(|&x| (x + rng.range_f32(-0.1, 0.1)).clamp(0.0, 1.0)).collect();
        data.push((build_input(&lr, &prev), target));
    }
    println!("\n  Training data: {} pairs", data.len());

    let seed = 99999;

    // 1 hidden layer
    test_config("1 hidden", &[IN_DIM, H_DIM, OUT_DIM], false, &data, seed);
    test_config("1 hidden", &[IN_DIM, H_DIM, OUT_DIM], true, &data, seed);

    // 2 hidden layers
    test_config("2 hidden", &[IN_DIM, H_DIM, H_DIM, OUT_DIM], false, &data, seed);
    test_config("2 hidden", &[IN_DIM, H_DIM, H_DIM, OUT_DIM], true, &data, seed);

    // 3 hidden layers
    test_config("3 hidden", &[IN_DIM, H_DIM, H_DIM, H_DIM, OUT_DIM], false, &data, seed);
    test_config("3 hidden", &[IN_DIM, H_DIM, H_DIM, H_DIM, OUT_DIM], true, &data, seed);

    // Wider: 128 hidden
    test_config("1h wide", &[IN_DIM, 128, OUT_DIM], false, &data, seed);
    test_config("2h wide", &[IN_DIM, 128, 128, OUT_DIM], false, &data, seed);

    println!("\n================================================================");
    println!("  COMPARISON SUMMARY");
    println!("  EP baseline (from POC): -0.8 dB vs bilinear");
    println!("  See above for backprop results per config");
    println!("================================================================");
}
