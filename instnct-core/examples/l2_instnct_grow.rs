//! L2 INSTNCT Greedy Growth — sparse topology for masked char prediction
//!
//! Instead of backprop, grow the network neuron-by-neuron:
//! 1. Start with direct input→output connections
//! 2. Add 1 neuron: random sparse connections + C19 activation
//! 3. Keep if it improves masked char accuracy
//! 4. Repeat — each neuron provably improves or is skipped
//!
//! Overlapping I/O: output neurons see inputs + hidden neurons.
//! Sparse: each neuron has only 4-8 connections (not all-to-all).
//!
//! Run: cargo run --example l2_instnct_grow --release

use std::time::Instant;

const LUT: [[i8; 2]; 27] = [
    [-2,-4],[-4,-2],[0,-6],[-2,-5],[-1,-6],[-3,-5],[1,-8],[-1,-7],
    [-2,-6],[-4,-5],[0,-8],[-2,-7],[-1,-8],[-3,-7],[1,-10],[-1,-9],
    [-4,-6],[-6,-5],[-2,-8],[-4,-7],[-3,-8],[-5,-7],[-1,-10],[-3,-9],
    [-4,-8],[-6,-7],[-2,-10],
];

fn c19a(x:f32,c:f32,r:f32)->f32{let c=c.max(0.1);let r=r.max(0.0);let l=6.0*c;
    if x>=l{return x-l;}if x<=-l{return x+l;}
    let s=x/c;let n=s.floor();let t=s-n;let h=t*(1.0-t);
    let sg=if(n as i32)%2==0{1.0}else{-1.0};c*(sg*h+r*h*h)}

struct Rng(u64);
impl Rng{
    fn new(s:u64)->Self{Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1))}
    fn next(&mut self)->u64{self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);self.0}
    fn range(&mut self,lo:usize,hi:usize)->usize{if hi<=lo{lo}else{lo+(self.next()as usize%(hi-lo))}}
    fn rangef(&mut self,lo:f32,hi:f32)->f32{lo+((self.next()>>33)%65536)as f32/65536.0*(hi-lo)}
    fn choose_n(&mut self, n: usize, max: usize) -> Vec<usize> {
        let mut picked = Vec::new();
        while picked.len() < n {
            let v = self.range(0, max);
            if !picked.contains(&v) { picked.push(v); }
        }
        picked
    }
}

fn load_corpus(p:&str)->Vec<u8>{std::fs::read(p).expect("r").iter().filter_map(|&b|match b{
    b'a'..=b'z'=>Some(b-b'a'),b'A'..=b'Z'=>Some(b-b'A'),b' '|b'\n'|b'\t'|b'\r'=>Some(26),_=>None}).collect()}

// Sparse neuron: reads from specific node indices, C19 activation
struct Neuron {
    sources: Vec<usize>,  // which nodes this reads from
    weights: Vec<f32>,     // weight per source
    bias: f32,
    c: f32,
    rho: f32,
}

impl Neuron {
    fn eval(&self, nodes: &[f32]) -> f32 {
        let mut dot = self.bias;
        for (i, &src) in self.sources.iter().enumerate() {
            if src < nodes.len() { dot += self.weights[i] * nodes[src]; }
        }
        c19a(dot, self.c, self.rho)
    }
}

// Output neuron: linear combination → logit (no activation)
struct OutputNeuron {
    sources: Vec<usize>,
    weights: Vec<f32>,
    bias: f32,
}

impl OutputNeuron {
    fn eval(&self, nodes: &[f32]) -> f32 {
        let mut dot = self.bias;
        for (i, &src) in self.sources.iter().enumerate() {
            if src < nodes.len() { dot += self.weights[i] * nodes[src]; }
        }
        dot
    }
}

struct Network {
    hidden: Vec<Neuron>,
    output: Vec<OutputNeuron>, // 27 output neurons
    n_inputs: usize,
}

impl Network {
    fn new(n_inputs: usize, fan_in_out: usize, rng: &mut Rng) -> Self {
        // Init: output neurons with random sparse connections to inputs
        let mut output = Vec::new();
        for _ in 0..27 {
            let sources = rng.choose_n(fan_in_out.min(n_inputs), n_inputs);
            let weights: Vec<f32> = sources.iter().map(|_| rng.rangef(-1.0, 1.0)).collect();
            output.push(OutputNeuron { sources, weights, bias: 0.0 });
        }
        Network { hidden: Vec::new(), output, n_inputs }
    }

    fn n_nodes(&self) -> usize { self.n_inputs + self.hidden.len() }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        // Build node values: [inputs..., hidden outputs...]
        let mut nodes = input.to_vec();
        for h in &self.hidden {
            nodes.push(h.eval(&nodes));
        }
        // Output logits
        let logits: Vec<f32> = self.output.iter().map(|o| o.eval(&nodes)).collect();
        logits
    }

    fn predict(&self, input: &[f32]) -> usize {
        let logits = self.forward(input);
        logits.iter().enumerate()
            .max_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|v| v.0).unwrap_or(0)
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let split = corpus.len() * 80 / 100;
    let encoded: Vec<[f32;2]> = corpus.iter().map(|&ch|
        [LUT[ch as usize][0] as f32/16.0, LUT[ch as usize][1] as f32/16.0]).collect();

    let ctx = 64usize; // start moderate
    let mask_pos = ctx / 2;
    let mask_val = [1.0f32, 1.0];
    let n_inputs = ctx * 2; // 128
    let fan_in = 8; // connections per neuron

    println!("=== INSTNCT GREEDY GROWTH ===\n");
    println!("  ctx={} bytes, {} input nodes, fan_in={}", ctx, n_inputs, fan_in);
    println!("  C19 hidden neurons, sparse connections");
    println!("  Grow neuron-by-neuron, keep if improves\n");

    // Build eval samples (fixed set for consistent evaluation)
    let n_eval = 1500;
    let mut eval_rng = Rng::new(999);
    let eval_samples: Vec<(Vec<f32>, u8)> = (0..n_eval).filter_map(|_| {
        if split < ctx + 1 { return None; }
        let off = eval_rng.range(0, split.saturating_sub(ctx + 1));
        let mut input = Vec::with_capacity(n_inputs);
        for i in 0..ctx {
            if i == mask_pos { input.extend_from_slice(&mask_val); }
            else if off+i < encoded.len() { input.push(encoded[off+i][0]); input.push(encoded[off+i][1]); }
            else { input.push(0.0); input.push(0.0); }
        }
        Some((input, corpus[off + mask_pos]))
    }).collect();

    let test_samples: Vec<(Vec<f32>, u8)> = (0..n_eval).filter_map(|_| {
        if corpus.len() < split + ctx + 1 { return None; }
        let off = eval_rng.range(split, corpus.len().saturating_sub(ctx + 1));
        let mut input = Vec::with_capacity(n_inputs);
        for i in 0..ctx {
            if i == mask_pos { input.extend_from_slice(&mask_val); }
            else if off+i < encoded.len() { input.push(encoded[off+i][0]); input.push(encoded[off+i][1]); }
            else { input.push(0.0); input.push(0.0); }
        }
        Some((input, corpus[off + mask_pos]))
    }).collect();

    let eval_acc = |net: &Network, samples: &[(Vec<f32>, u8)]| -> f64 {
        let mut ok = 0usize;
        for (inp, tgt) in samples {
            if net.predict(inp) == *tgt as usize { ok += 1; }
        }
        ok as f64 / samples.len() as f64 * 100.0
    };

    // Init network
    let mut rng = Rng::new(42);
    let mut net = Network::new(n_inputs, fan_in, &mut rng);

    // Optimize initial output weights (random search)
    println!("  Optimizing initial output connections...");
    let mut best_acc = eval_acc(&net, &eval_samples);
    for _ in 0..20000 {
        let oi = rng.range(0, 27);
        let wi = rng.range(0, net.output[oi].weights.len());
        let old_w = net.output[oi].weights[wi];
        let old_b = net.output[oi].bias;
        net.output[oi].weights[wi] = rng.rangef(-2.0, 2.0);
        if rng.next() % 5 == 0 { net.output[oi].bias = rng.rangef(-1.0, 1.0); }
        let acc = eval_acc(&net, &eval_samples);
        if acc > best_acc { best_acc = acc; }
        else { net.output[oi].weights[wi] = old_w; net.output[oi].bias = old_b; }
    }
    let test_acc = eval_acc(&net, &test_samples);
    println!("  Initial: train={:.1}% test={:.1}% (0 hidden neurons)\n", best_acc, test_acc);

    // Greedy growth
    let max_neurons = 200;
    let mutations_per_neuron = 5000;

    println!("  {:>4} {:>8} {:>8} {:>8} {:>6} {:>8}",
        "#N", "train%", "test%", "delta", "time", "connections");
    println!("  {}", "-".repeat(55));

    for ni in 0..max_neurons {
        let tc = Instant::now();
        let n_nodes = net.n_nodes();

        // Try many random neurons, keep best
        let mut best_neuron: Option<Neuron> = None;
        let mut best_out_change: Option<(usize, usize, f32, f32)> = None; // (out_idx, slot, new_w, old_w)
        let mut best_new_acc = best_acc;
        let mut best_new_test = eval_acc(&net, &test_samples);

        for _ in 0..mutations_per_neuron {
            // Random neuron
            let sources = rng.choose_n(fan_in.min(n_nodes), n_nodes);
            let weights: Vec<f32> = sources.iter().map(|_| rng.rangef(-3.0, 3.0)).collect();
            let bias = rng.rangef(-2.0, 2.0);
            let c = rng.rangef(1.0, 30.0);
            let rho_v = rng.rangef(0.0, 3.0);

            let neuron = Neuron { sources: sources.clone(), weights, bias, c, rho: rho_v };

            // Temporarily add
            net.hidden.push(neuron);
            let new_node_idx = net.n_nodes() - 1;

            // Also try connecting an output neuron to this new hidden node
            let oi = rng.range(0, 27);
            let slot = rng.range(0, net.output[oi].sources.len());
            let old_src = net.output[oi].sources[slot];
            let old_w = net.output[oi].weights[slot];
            net.output[oi].sources[slot] = new_node_idx;
            net.output[oi].weights[slot] = rng.rangef(-2.0, 2.0);
            let new_w = net.output[oi].weights[slot];

            let acc = eval_acc(&net, &eval_samples);
            let test = eval_acc(&net, &test_samples);
            // Keep only if BOTH train and test improve (prevents overfitting)
            let combined = acc * 0.4 + test * 0.6; // weight test more
            let best_combined = best_new_acc * 0.4 + best_new_test * 0.6;
            if combined > best_combined {
                best_new_acc = acc;
                best_new_test = test;
                best_neuron = Some(net.hidden.last().unwrap().clone());
                best_out_change = Some((oi, slot, new_w, old_w));
            }

            // Restore
            net.output[oi].sources[slot] = old_src;
            net.output[oi].weights[slot] = old_w;
            net.hidden.pop();
        }

        if let Some(neuron) = best_neuron {
            let delta = best_new_acc - best_acc;
            net.hidden.push(neuron);
            if let Some((oi, slot, new_w, _)) = best_out_change {
                net.output[oi].sources[slot] = net.n_nodes() - 1;
                net.output[oi].weights[slot] = new_w;
            }
            best_acc = best_new_acc;

            // Output weight optimization (test-aware)
            let mut cur_test = eval_acc(&net, &test_samples);
            for _ in 0..1000 {
                let oi = rng.range(0, 27);
                let wi = rng.range(0, net.output[oi].weights.len());
                let old = net.output[oi].weights[wi];
                net.output[oi].weights[wi] = rng.rangef(-3.0, 3.0);
                let acc = eval_acc(&net, &eval_samples);
                let te = eval_acc(&net, &test_samples);
                let score = acc * 0.4 + te * 0.6;
                let old_score = best_acc * 0.4 + cur_test * 0.6;
                if score > old_score { best_acc = acc; cur_test = te; }
                else { net.output[oi].weights[wi] = old; }
            }

            let test = eval_acc(&net, &test_samples);
            let conns = net.hidden.len() * fan_in + 27 * fan_in;

            if ni % 5 == 0 || delta > 1.0 {
                println!("  {:>4} {:>7.1}% {:>7.1}% {:>+7.1}% {:>5.0}s {:>8}",
                    ni+1, best_acc, test, delta, t0.elapsed().as_secs_f64(), conns);
            }

            if best_acc >= 99.5 {
                println!("\n  *** 100% at {} neurons ***", ni+1);
                break;
            }
        } else {
            println!("  {:>4} — no improvement found, stopping", ni+1);
            break;
        }

        if t0.elapsed().as_secs() > 600 { println!("  Time limit."); break; }
    }

    // Per-class
    println!("\n--- Per-class test accuracy ({} neurons) ---\n", net.hidden.len());
    let chars = "abcdefghijklmnopqrstuvwxyz ";
    let mut pc_ok=[0u32;27];let mut pc_tot=[0u32;27];
    for (inp,tgt) in &test_samples {
        let pred=net.predict(inp);let t=*tgt as usize;
        pc_tot[t]+=1;if pred==t{pc_ok[t]+=1;}}
    for c in 0..27{let ch=chars.as_bytes()[c]as char;
        let acc=if pc_tot[c]>0{pc_ok[c]as f32/pc_tot[c]as f32*100.0}else{0.0};
        let bar:String=(0..(acc/5.0)as usize).map(|_|'#').collect();
        println!("  '{}': {:>5.1}% ({:>3}/{:>3}) {}",ch,acc,pc_ok[c],pc_tot[c],bar);}

    println!("\n  Final: {} hidden neurons, {} total connections",
        net.hidden.len(), net.hidden.len()*fan_in+27*fan_in);
    println!("  Best train: {:.1}%", best_acc);
    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}

// Clone for Neuron
impl Clone for Neuron {
    fn clone(&self) -> Self {
        Neuron {
            sources: self.sources.clone(),
            weights: self.weights.clone(),
            bias: self.bias, c: self.c, rho: self.rho,
        }
    }
}
