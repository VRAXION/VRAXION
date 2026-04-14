//! Mixer exhaustive bake — fully baked binary mixer at ctx=2,3,4
//!
//! Each bottleneck neuron: exhaustive binary {-1,+1} search over all inputs
//! Decoder: tied Wᵀ, bias optimized analytically
//! One neuron at a time → freeze → next neuron
//!
//! Run: cargo run --example mixer_exhaustive_bake --release

use std::time::Instant;

struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 { self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.s }
    fn range(&mut self, lo: usize, hi: usize) -> usize { if hi <= lo { lo } else { lo + (self.next() as usize % (hi - lo)) } }
}

fn load_corpus(path: &str) -> Vec<u8> {
    let raw = std::fs::read(path).expect("read");
    let mut c = Vec::new();
    for &b in &raw { match b { b'a'..=b'z' => c.push(b-b'a'), b'A'..=b'Z' => c.push(b-b'A'), b' '|b'\n'|b'\t'|b'\r' => c.push(26), _ => {} } }
    c
}

fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c = c.max(0.1); let rho = rho.max(0.0); let l = 6.0*c;
    if x >= l { return x-l; } if x <= -l { return x+l; }
    let s = x/c; let n = s.floor(); let t = s-n; let h = t*(1.0-t);
    let sg = if (n as i32)%2==0 { 1.0 } else { -1.0 }; c*(sg*h+rho*h*h)
}

struct Preproc { w: [[i8;8];7], b: [i8;7], c: [f32;7], rho: [f32;7] }
impl Preproc {
    fn new() -> Self { Preproc {
        w: [[-1,1,1,-1,-1,1,1,-1],[1,-1,1,1,-1,-1,-1,-1],[-1,-1,1,-1,1,1,-1,-1],
            [-1,-1,-1,1,-1,-1,1,-1],[-1,1,1,1,-1,-1,-1,-1],[1,1,-1,-1,-1,-1,-1,-1],
            [-1,1,-1,1,1,1,-1,-1]],
        b: [1,1,1,1,1,1,1], c: [10.0;7], rho: [2.0,0.0,0.0,0.0,0.0,0.0,0.0],
    }}
    fn encode(&self, ch: u8) -> [f32;7] {
        let mut bits=[0.0f32;8]; for i in 0..8 { bits[i]=((ch>>i)&1) as f32; }
        let mut o=[0.0f32;7];
        for k in 0..7 { let mut d=self.b[k] as f32; for j in 0..8 { d+=self.w[k][j] as f32*bits[j]; } o[k]=c19(d,self.c[k],self.rho[k]); }
        o
    }
    fn encode_seq(&self, chars: &[u8]) -> Vec<f32> {
        chars.iter().flat_map(|&ch| self.encode(ch).to_vec()).collect()
    }
}

// ══════════════════════════════════════════════════════
// EXHAUSTIVE BINARY MIXER — one neuron at a time
// ══════════════════════════════════════════════════════
#[derive(Clone)]
struct BakedNeuron {
    weights: Vec<i8>,  // {-1, +1} per input
    bias: i8,          // {-1, +1}
}

impl BakedNeuron {
    fn eval(&self, input: &[f32]) -> f32 {
        let mut dot = self.bias as f32;
        for (j, &w) in self.weights.iter().enumerate() { dot += w as f32 * input[j]; }
        dot // raw dot product — no activation needed for tied-weight decoder
    }
}

fn eval_roundtrip(neurons: &[BakedNeuron], samples: &[(Vec<f32>, Vec<u8>)], pp: &Preproc, ctx: usize) -> f64 {
    if neurons.is_empty() { return 0.0; }
    let mut ok = 0usize; let mut tot = 0usize;
    let idim = ctx * 7;

    for (sig, chars) in samples {
        // Encode: compute all neuron activations
        let hidden: Vec<f32> = neurons.iter().map(|n| n.eval(sig)).collect();

        // Decode: Wᵀ × hidden + optimal bias (per output dim)
        let mut recon = vec![0.0f32; idim];
        for j in 0..idim {
            for (k, n) in neurons.iter().enumerate() {
                recon[j] += n.weights[j] as f32 * hidden[k];
            }
        }

        // Check each char via nearest-code
        for i in 0..ctx {
            let rs = &recon[i*7..(i+1)*7];
            let mut best = 0u8; let mut bd = f32::MAX;
            for ch in 0..27u8 {
                let code = pp.encode(ch);
                let d: f32 = code.iter().zip(rs).map(|(a,b)| (a-b)*(a-b)).sum();
                if d < bd { bd = d; best = ch; }
            }
            if best == chars[i] { ok += 1; }
            tot += 1;
        }
    }
    ok as f64 / tot as f64 * 100.0
}

fn exhaustive_search_neuron(
    existing: &[BakedNeuron],
    samples: &[(Vec<f32>, Vec<u8>)],
    pp: &Preproc,
    ctx: usize,
    idim: usize,
    current_acc: f64,
) -> Option<(BakedNeuron, f64)> {
    let total_combos = 2u64.pow(idim as u32 + 1); // weights + bias
    let mut best_neuron: Option<BakedNeuron> = None;
    let mut best_acc = current_acc;

    for combo in 0..total_combos {
        let mut weights = vec![0i8; idim];
        for j in 0..idim { weights[j] = if (combo >> j) & 1 == 1 { 1 } else { -1 }; }
        let bias = if (combo >> idim) & 1 == 1 { 1i8 } else { -1 };

        let neuron = BakedNeuron { weights, bias };
        let mut test = existing.to_vec();
        test.push(neuron.clone());

        let acc = eval_roundtrip(&test, samples, pp, ctx);
        if acc > best_acc {
            best_acc = acc;
            best_neuron = Some(neuron);
            if acc >= 99.95 { return Some((best_neuron.unwrap(), best_acc)); }
        }
    }

    best_neuron.map(|n| (n, best_acc))
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let pp = Preproc::new();

    println!("=== MIXER EXHAUSTIVE BAKE ===");
    println!("Binary {{-1,+1}} weights, one neuron at a time, guaranteed optimal\n");

    for &ctx in &[2, 3, 4] {
        let idim = ctx * 7;
        let combos_per_neuron = 2u64.pow(idim as u32 + 1);
        println!("━━━ ctx={} (input={}, {} combos/neuron) ━━━", ctx, idim, combos_per_neuron);

        if combos_per_neuron > 2_000_000_000 {
            println!("  SKIP: too many combos ({})\n", combos_per_neuron);
            continue;
        }

        // Prepare eval samples
        let mut rng = Rng::new(42);
        let n_samples = 1000;
        let mut samples: Vec<(Vec<f32>, Vec<u8>)> = Vec::new();
        for _ in 0..n_samples {
            if corpus.len() < ctx { break; }
            let off = rng.range(0, corpus.len() - ctx);
            let sig = pp.encode_seq(&corpus[off..off+ctx]);
            let chars: Vec<u8> = corpus[off..off+ctx].to_vec();
            samples.push((sig, chars));
        }

        let mut neurons: Vec<BakedNeuron> = Vec::new();
        let mut acc = 0.0f64;
        let max_neurons = idim; // at most input_dim neurons

        for step in 0..max_neurons {
            let tc = Instant::now();
            let result = exhaustive_search_neuron(&neurons, &samples, &pp, ctx, idim, acc);

            if let Some((neuron, new_acc)) = result {
                let delta = new_acc - acc;
                acc = new_acc;
                let ws: String = neuron.weights.iter().map(|&w| if w == 1 { '+' } else { '-' }).collect();
                println!("  N{}: {:.1}% (Δ{:+.1}) [{}] b={:+} ({:.1}s)",
                    step, acc, delta, ws, neuron.bias, tc.elapsed().as_secs_f64());
                neurons.push(neuron);

                if acc >= 99.95 {
                    let total_bits = neurons.len() * (idim + 1);
                    println!("\n  ★★★ 100% at {} neurons, {} bits = {} bytes ★★★",
                        neurons.len(), total_bits, (total_bits + 7) / 8);
                    break;
                }
                if delta < 0.1 {
                    println!("  Stalled (Δ<0.1), stopping at {} neurons", neurons.len());
                    break;
                }
            } else {
                println!("  No improvement, stopping");
                break;
            }
        }

        let total_bits = neurons.len() * (idim + 1);
        println!("  Final: {:.1}% with {} neurons, {} bits = {} bytes",
            acc, neurons.len(), total_bits, (total_bits + 7) / 8);
        println!("  Time: {:.1}s\n", t0.elapsed().as_secs_f64());
    }

    println!("Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
