//! Abstract Core v5 — 400 epoch training on the two 93%+ configs
//!
//! v4 showed no plateau at 200 epochs. Push to 400 to find the ceiling.
//! Also test: the compact 2L-512-256 (196K params) vs the deeper 3L (226K params).
//!
//! Run: cargo run --example abstract_core_v5_400ep --release

use std::time::Instant;

struct Rng { s: u64 }
impl Rng {
    fn new(seed: u64) -> Self { Rng { s: seed.wrapping_mul(6364136223846793005).wrapping_add(1) } }
    fn next(&mut self) -> u64 { self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.s }
    fn f32(&mut self) -> f32 { ((self.next() >> 33) % 65536) as f32 / 65536.0 }
    fn normal(&mut self) -> f32 { let u1 = self.f32().max(1e-7); let u2 = self.f32(); (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() }
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
}

const CTX: usize = 16;
const PO: usize = 7;
const MI: usize = CTX * PO;
const NC: usize = 27;

#[derive(Clone)]
struct Mlp { layers: Vec<(Vec<Vec<f32>>, Vec<f32>)>, dims: Vec<usize> }
impl Mlp {
    fn new(dims: &[usize], rng: &mut Rng) -> Self {
        let mut layers = Vec::new();
        for i in 0..dims.len()-1 {
            let s = (2.0/dims[i] as f32).sqrt();
            layers.push(((0..dims[i+1]).map(|_| (0..dims[i]).map(|_| rng.normal()*s).collect()).collect(),
                         vec![0.0f32; dims[i+1]]));
        }
        Mlp { layers, dims: dims.to_vec() }
    }
    fn total_params(&self) -> usize { self.layers.iter().map(|(w,b)| w.len()*w[0].len()+b.len()).sum() }
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut act = input.to_vec(); let nl = self.layers.len();
        for (li,(w,b)) in self.layers.iter().enumerate() {
            let mut next = vec![0.0f32;w.len()];
            for i in 0..w.len() { next[i]=b[i]; for j in 0..act.len() { next[i]+=w[i][j]*act[j]; } if li<nl-1 { next[i]=next[i].max(0.0); } }
            act = next;
        } act
    }
    fn predict(&self, input: &[f32]) -> usize {
        let lo = self.forward(input);
        lo.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0)
    }
    fn train_step(&mut self, input: &[f32], target: usize, lr: f32) -> f32 {
        let mut acts = vec![input.to_vec()]; let nl = self.layers.len();
        for (li,(w,b)) in self.layers.iter().enumerate() {
            let prev = acts.last().unwrap();
            let mut a = vec![0.0f32;w.len()];
            for i in 0..w.len() { a[i]=b[i]; for j in 0..prev.len() { a[i]+=w[i][j]*prev[j]; } if li<nl-1 { a[i]=a[i].max(0.0); } }
            acts.push(a);
        }
        let logits = acts.last().unwrap();
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|&l| (l-mx).exp()).collect();
        let se: f32 = exp.iter().sum(); let sm: Vec<f32> = exp.iter().map(|&e| e/se).collect();
        let loss = -(sm[target].max(1e-7)).ln();
        let mut delta: Vec<f32> = sm; delta[target] -= 1.0;
        for li in (0..nl).rev() {
            let prev_act = acts[li].clone();
            let (ref mut w, ref mut b) = self.layers[li];
            let mut d_prev = vec![0.0f32; prev_act.len()];
            for i in 0..w.len() {
                let d = if li<nl-1 { if acts[li+1][i]>0.0 { delta[i] } else { 0.0 } } else { delta[i] };
                for j in 0..prev_act.len() { d_prev[j]+=d*w[i][j]; w[i][j]-=lr*d*prev_act[j]; }
                b[i]-=lr*d;
            } delta = d_prev;
        } loss
    }
}

fn make_signals(pp: &Preproc, corpus: &[u8], off: usize) -> Vec<f32> {
    let mut s = vec![0.0f32;MI];
    for i in 0..CTX { let e=pp.encode(corpus[off+i]); for k in 0..PO { s[i*PO+k]=e[k]; } } s
}

fn eval_acc(pp: &Preproc, m: &Mlp, corpus: &[u8], n: usize, seed: u64) -> f64 {
    let mut rng=Rng::new(seed); let mut ok=0; let mut tot=0;
    for _ in 0..n { if corpus.len()<CTX+1{break;} let off=rng.range(0,corpus.len()-CTX-1);
        if m.predict(&make_signals(pp,corpus,off))==corpus[off+CTX] as usize{ok+=1;} tot+=1; }
    ok as f64/tot as f64*100.0
}

fn eval_int8(pp: &Preproc, m: &Mlp, corpus: &[u8], n: usize, seed: u64) -> f64 {
    let scales: Vec<(f32,f32)> = m.layers.iter().map(|(w,b)| {
        (w.iter().flat_map(|r|r.iter()).map(|x|x.abs()).fold(0.0f32,f32::max).max(1e-7)/127.0,
         b.iter().map(|x|x.abs()).fold(0.0f32,f32::max).max(1e-7)/127.0)
    }).collect();
    let mut rng=Rng::new(seed); let mut ok=0; let mut tot=0; let nl=m.layers.len();
    for _ in 0..n { if corpus.len()<CTX+1{break;} let off=rng.range(0,corpus.len()-CTX-1);
        let mut act=make_signals(pp,corpus,off);
        for (li,(w,b)) in m.layers.iter().enumerate() {
            let (sw,sb)=scales[li]; let mut next=vec![0.0f32;w.len()];
            for i in 0..w.len() { next[i]=(b[i]/sb).round().max(-127.0).min(127.0)*sb;
                for j in 0..act.len() { next[i]+=(w[i][j]/sw).round().max(-127.0).min(127.0)*sw*act[j]; }
                if li<nl-1 { next[i]=next[i].max(0.0); } } act=next; }
        if act.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)|i).unwrap_or(0)==corpus[off+CTX] as usize{ok+=1;} tot+=1; }
    ok as f64/tot as f64*100.0
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let pp = Preproc::new();

    println!("=== ABSTRACT CORE v5 — 400 Epoch Push ===\n");

    for &(name, ref dims) in &[
        ("2L-512-256 400ep", vec![MI,512,256,NC]),
        ("3L-512-256-128 400ep", vec![MI,512,256,128,NC]),
    ] {
        let mut rng = Rng::new(42);
        let mut m = Mlp::new(dims, &mut rng);
        println!("━━━ {} ({} params) ━━━", name, m.total_params());

        let epochs = 400;
        let samples = 15000;
        let mut best = 0.0f64;

        for epoch in 0..epochs {
            let lr = 0.003 * (1.0 - epoch as f32 / epochs as f32 * 0.9);
            let mut rng_t = Rng::new(epoch as u64 * 1000 + 42);
            for _ in 0..samples {
                if corpus.len() < CTX+1 { break; }
                let off = rng_t.range(0, corpus.len()-CTX-1);
                m.train_step(&make_signals(&pp, &corpus, off), corpus[off+CTX] as usize, lr);
            }
            if (epoch+1) % 50 == 0 {
                let fa = eval_acc(&pp, &m, &corpus, 5000, 999+epoch as u64);
                let ia = eval_int8(&pp, &m, &corpus, 5000, 999+epoch as u64);
                if fa > best { best = fa; }
                println!("  epoch {:>3}: float={:.1}% int8={:.1}% best={:.1}%", epoch+1, fa, ia, best);
            }
        }
        let fa = eval_acc(&pp, &m, &corpus, 10000, 777);
        let ia = eval_int8(&pp, &m, &corpus, 10000, 777);
        if fa > best { best = fa; }
        println!("  FINAL (10K): float={:.1}% int8={:.1}% best={:.1}%", fa, ia, best);
        println!("  Time: {:.1}s\n", t0.elapsed().as_secs_f64());
    }

    println!("Baselines: frequency=20.3%, INSTNCT=24.6%, v4 best=93.3%");
    println!("Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
