//! L2 Word Tokenizer + Embedding — C19 word encoder
//!
//! Step 1: Build vocabulary from Alice corpus
//! Step 2: C19 word encoder (chars → embedding)
//! Step 3: Word2vec-style training (predict context words)
//! Step 4: Evaluate embedding quality (nearest neighbors)
//!
//! Run: cargo run --example l2_word_tokenizer --release

use std::collections::HashMap;
use std::time::Instant;

const LUT: [[i8; 2]; 27] = [
    [-2,-4],[-4,-2],[0,-6],[-2,-5],[-1,-6],[-3,-5],[1,-8],[-1,-7],
    [-2,-6],[-4,-5],[0,-8],[-2,-7],[-1,-8],[-3,-7],[1,-10],[-1,-9],
    [-4,-6],[-6,-5],[-2,-8],[-4,-7],[-3,-8],[-5,-7],[-1,-10],[-3,-9],
    [-4,-8],[-6,-7],[-2,-10],
];

fn c19(x: f32, c: f32, rho: f32) -> f32 {
    let c=c.max(0.1); let rho=rho.max(0.0); let l=6.0*c;
    if x>=l{return x-l;} if x<=-l{return x+l;}
    let s=x/c; let n=s.floor(); let t=s-n; let h=t*(1.0-t);
    let sg=if(n as i32)%2==0{1.0}else{-1.0}; c*(sg*h+rho*h*h)
}

fn c19_grad(x: f32, c: f32, rho: f32) -> f32 {
    let c=c.max(0.1); let rho=rho.max(0.0); let l=6.0*c;
    if x>=l||x<=-l{return 1.0;}
    let s=x/c; let n=s.floor(); let t=s-n; let h=t*(1.0-t);
    let sg=if(n as i32)%2==0{1.0}else{-1.0};
    (sg+2.0*rho*h)*(1.0-2.0*t)
}

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1)) }
    fn next(&mut self) -> u64 { self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); self.0 }
    fn normal(&mut self) -> f32 {
        let u1=(((self.next()>>33)%65536) as f32/65536.0).max(1e-7);
        let u2=((self.next()>>33)%65536) as f32/65536.0;
        (-2.0*u1.ln()).sqrt()*(2.0*std::f32::consts::PI*u2).cos()
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        if hi<=lo{lo}else{lo+(self.next() as usize%(hi-lo))}
    }
}

fn load_corpus(path: &str) -> Vec<u8> {
    let raw = std::fs::read(path).expect("read");
    raw.iter().filter_map(|&b| match b {
        b'a'..=b'z' => Some(b-b'a'), b'A'..=b'Z' => Some(b-b'A'),
        b' '|b'\n'|b'\t'|b'\r' => Some(26), _ => None,
    }).collect()
}

// Word encoder: padded chars → C19 layers → embedding
const MAX_WORD_LEN: usize = 20;
const PAD_CH: usize = 2; // channels per char from LUT
const WORD_INPUT: usize = MAX_WORD_LEN * PAD_CH; // 40
const EMB_DIM: usize = 16;

struct WordEncoder {
    // Layer 1: WORD_INPUT → 64
    w1: Vec<Vec<f32>>, b1: Vec<f32>, c1: Vec<f32>, rho1: Vec<f32>,
    // Layer 2: 64 → 64
    w2: Vec<Vec<f32>>, b2: Vec<f32>, c2: Vec<f32>, rho2: Vec<f32>,
    // Layer 3: 64 → EMB_DIM (linear output)
    w3: Vec<Vec<f32>>, b3: Vec<f32>,
}

impl WordEncoder {
    fn new(rng: &mut Rng) -> Self {
        let h = 64;
        let s1 = (2.0/WORD_INPUT as f32).sqrt();
        let s2 = (2.0/h as f32).sqrt();
        let s3 = (2.0/h as f32).sqrt();
        WordEncoder {
            w1: (0..h).map(|_|(0..WORD_INPUT).map(|_|rng.normal()*s1).collect()).collect(),
            b1: vec![0.0;h], c1: vec![5.0;h], rho1: vec![0.5;h],
            w2: (0..h).map(|_|(0..h).map(|_|rng.normal()*s2).collect()).collect(),
            b2: vec![0.0;h], c2: vec![5.0;h], rho2: vec![0.5;h],
            w3: (0..EMB_DIM).map(|_|(0..h).map(|_|rng.normal()*s3).collect()).collect(),
            b3: vec![0.0;EMB_DIM],
        }
    }

    fn params(&self) -> usize {
        WORD_INPUT*64 + 64 + 64*2 + 64*64 + 64 + 64*2 + 64*EMB_DIM + EMB_DIM
    }

    fn encode_word_input(word: &[u8]) -> Vec<f32> {
        let mut v = vec![0.0f32; WORD_INPUT];
        for (i, &ch) in word.iter().enumerate() {
            if i >= MAX_WORD_LEN { break; }
            v[i*PAD_CH] = LUT[ch as usize][0] as f32 / 16.0;
            v[i*PAD_CH+1] = LUT[ch as usize][1] as f32 / 16.0;
        }
        v
    }

    fn forward(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        // L1: C19
        let mut z1 = self.b1.clone();
        for j in 0..64 { for k in 0..WORD_INPUT { z1[j] += self.w1[j][k]*input[k]; } }
        let a1: Vec<f32> = z1.iter().enumerate().map(|(j,&v)| c19(v, self.c1[j], self.rho1[j])).collect();
        // L2: C19
        let mut z2 = self.b2.clone();
        for j in 0..64 { for k in 0..64 { z2[j] += self.w2[j][k]*a1[k]; } }
        let a2: Vec<f32> = z2.iter().enumerate().map(|(j,&v)| c19(v, self.c2[j], self.rho2[j])).collect();
        // L3: linear
        let mut emb = self.b3.clone();
        for j in 0..EMB_DIM { for k in 0..64 { emb[j] += self.w3[j][k]*a2[k]; } }
        (z1, a1, z2, a2, emb)
    }

    fn embed(&self, word: &[u8]) -> Vec<f32> {
        let input = Self::encode_word_input(word);
        self.forward(&input).4
    }

    // Word2vec CBOW: given context word embeddings (averaged), predict center word
    // Actually simpler: skip-gram: given center, predict context
    // We'll use: center embedding should be CLOSE to context embeddings
    fn train_step(&mut self, center: &[u8], context: &[u8], negative: &[u8], lr: f32) {
        let c_input = Self::encode_word_input(center);
        let (z1, a1, z2, a2, c_emb) = self.forward(&c_input);
        let ctx_emb = self.embed(context);
        let neg_emb = self.embed(negative);

        // Loss: dot(center, context) should be high, dot(center, negative) should be low
        // Sigmoid loss: -log(σ(c·ctx)) - log(σ(-c·neg))
        let pos_dot: f32 = c_emb.iter().zip(&ctx_emb).map(|(a,b)| a*b).sum();
        let neg_dot: f32 = c_emb.iter().zip(&neg_emb).map(|(a,b)| a*b).sum();

        let pos_sig = 1.0 / (1.0 + (-pos_dot).exp());
        let neg_sig = 1.0 / (1.0 + (-neg_dot).exp());

        // Gradient on center embedding
        let mut d_emb = vec![0.0f32; EMB_DIM];
        for i in 0..EMB_DIM {
            d_emb[i] = (pos_sig - 1.0) * ctx_emb[i] + neg_sig * neg_emb[i];
        }

        // Backprop through L3 (linear)
        let mut da2 = vec![0.0f32; 64];
        for j in 0..EMB_DIM {
            for k in 0..64 {
                da2[k] += d_emb[j] * self.w3[j][k];
                self.w3[j][k] -= lr * d_emb[j] * a2[k];
            }
            self.b3[j] -= lr * d_emb[j];
        }

        // Backprop through L2 (C19)
        let mut da1 = vec![0.0f32; 64];
        for j in 0..64 {
            let g = c19_grad(z2[j], self.c2[j], self.rho2[j]);
            let dz = da2[j] * g;
            for k in 0..64 {
                da1[k] += dz * self.w2[j][k];
                self.w2[j][k] -= lr * dz * a1[k];
            }
            self.b2[j] -= lr * dz;
            // c2, rho2 gradients
            let eps = 0.01;
            let dc = (c19(z2[j],self.c2[j]+eps,self.rho2[j])-c19(z2[j],self.c2[j]-eps,self.rho2[j]))/(2.0*eps);
            self.c2[j] -= lr * da2[j] * dc * 0.1;
            self.c2[j] = self.c2[j].max(0.5).min(50.0);
            let dr = (c19(z2[j],self.c2[j],self.rho2[j]+eps)-c19(z2[j],self.c2[j],self.rho2[j]-eps))/(2.0*eps);
            self.rho2[j] -= lr * da2[j] * dr * 0.1;
            self.rho2[j] = self.rho2[j].max(0.0).min(5.0);
        }

        // Backprop through L1 (C19)
        for j in 0..64 {
            let g = c19_grad(z1[j], self.c1[j], self.rho1[j]);
            let dz = da1[j] * g;
            for k in 0..WORD_INPUT {
                self.w1[j][k] -= lr * dz * c_input[k];
            }
            self.b1[j] -= lr * dz;
            let eps = 0.01;
            let dc = (c19(z1[j],self.c1[j]+eps,self.rho1[j])-c19(z1[j],self.c1[j]-eps,self.rho1[j]))/(2.0*eps);
            self.c1[j] -= lr * da1[j] * dc * 0.1;
            self.c1[j] = self.c1[j].max(0.5).min(50.0);
            let dr = (c19(z1[j],self.c1[j],self.rho1[j]+eps)-c19(z1[j],self.c1[j],self.rho1[j]-eps))/(2.0*eps);
            self.rho1[j] -= lr * da1[j] * dr * 0.1;
            self.rho1[j] = self.rho1[j].max(0.0).min(5.0);
        }
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");

    // ── Step 1: Build vocabulary ──
    println!("=== L2 WORD TOKENIZER + C19 EMBEDDING ===\n");

    // Split into words at spaces
    let mut words_seq: Vec<Vec<u8>> = Vec::new();
    let mut current_word: Vec<u8> = Vec::new();
    for &ch in &corpus {
        if ch == 26 { // space
            if !current_word.is_empty() {
                words_seq.push(current_word.clone());
                current_word.clear();
            }
        } else {
            current_word.push(ch);
        }
    }
    if !current_word.is_empty() { words_seq.push(current_word); }

    // Build vocab
    let mut word_counts: HashMap<Vec<u8>, usize> = HashMap::new();
    for w in &words_seq { *word_counts.entry(w.clone()).or_insert(0) += 1; }

    let mut vocab: Vec<(Vec<u8>, usize)> = word_counts.into_iter().collect();
    vocab.sort_by(|a,b| b.1.cmp(&a.1));

    let chars = "abcdefghijklmnopqrstuvwxyz ";
    let word_to_str = |w: &[u8]| -> String {
        w.iter().map(|&c| chars.as_bytes()[c as usize] as char).collect()
    };

    println!("  Step 1: Vocabulary\n");
    println!("  Total words in corpus: {}", words_seq.len());
    println!("  Unique words: {}", vocab.len());
    println!("  Max word length: {}", vocab.iter().map(|(w,_)|w.len()).max().unwrap_or(0));
    println!("  Avg word length: {:.1}", vocab.iter().map(|(w,c)| w.len()*c).sum::<usize>() as f64 / words_seq.len() as f64);

    println!("\n  Top 20 words:");
    for (i, (w, c)) in vocab.iter().take(20).enumerate() {
        println!("    {:>3}. {:>12} × {}", i+1, word_to_str(w), c);
    }

    // Build word→id map
    let word2id: HashMap<Vec<u8>, usize> = vocab.iter().enumerate().map(|(i,(w,_))| (w.clone(), i)).collect();

    // ── Step 2: Train C19 word encoder ──
    println!("\n  Step 2: C19 Word Encoder Training\n");
    println!("  Architecture: {}→64(C19)→64(C19)→{} (linear)", WORD_INPUT, EMB_DIM);

    let mut rng = Rng::new(42);
    let mut encoder = WordEncoder::new(&mut rng);
    println!("  Params: {} (+{} c/rho)\n", encoder.params(), 64*2*2);

    let split_w = words_seq.len() * 80 / 100;
    let window = 3; // context window

    println!("  {:>5} {:>8} {:>8} {:>6}", "epoch", "loss", "sim_acc", "time");
    println!("  {}", "-".repeat(35));

    for ep in 0..500 {
        let lr = 0.005 * (1.0 - ep as f32 / 500.0 * 0.8);
        let mut rt = Rng::new(ep as u64 * 1000 + 42);
        let mut total_loss = 0.0f32;
        let mut n = 0u32;

        for _ in 0..3000.min(split_w) {
            let pos = rt.range(window, split_w.saturating_sub(window));
            let center = &words_seq[pos];
            // Random context word within window
            let ctx_off = rt.range(1, window+1);
            let ctx_pos = if rt.next() % 2 == 0 { pos + ctx_off } else { pos.saturating_sub(ctx_off) };
            let ctx_pos = ctx_pos.min(split_w - 1);
            let context = &words_seq[ctx_pos];
            // Random negative
            let neg_pos = rt.range(0, split_w);
            let negative = &words_seq[neg_pos];

            encoder.train_step(center, context, negative, lr);

            let c_emb = encoder.embed(center);
            let ctx_emb = encoder.embed(context);
            let neg_emb = encoder.embed(negative);
            let pos_dot: f32 = c_emb.iter().zip(&ctx_emb).map(|(a,b)|a*b).sum();
            let neg_dot: f32 = c_emb.iter().zip(&neg_emb).map(|(a,b)|a*b).sum();
            let pos_sig = 1.0/(1.0+(-pos_dot).exp());
            let neg_sig = 1.0/(1.0+(-neg_dot).exp());
            let loss = -(pos_sig.max(1e-10).ln()) - ((1.0-neg_sig).max(1e-10).ln());
            if !loss.is_nan() { total_loss += loss; n += 1; }
        }

        if ep % 50 == 0 {
            // Eval: for random words, is nearest neighbor a semantically similar word?
            let mut sim_ok = 0usize; let mut sim_tot = 0usize;
            let mut rng3 = Rng::new(999);
            for _ in 0..200 {
                let pos = rng3.range(window, split_w.saturating_sub(window));
                let emb = encoder.embed(&words_seq[pos]);
                // Check: is any context word among top-5 nearest in embedding space?
                let mut dists: Vec<(usize, f32)> = (0..split_w.min(2000)).map(|i| {
                    let e2 = encoder.embed(&words_seq[i]);
                    let d: f32 = emb.iter().zip(&e2).map(|(a,b)|(a-b)*(a-b)).sum();
                    (i, d)
                }).collect();
                dists.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                // Top 5 nearest (excluding self)
                let top5: Vec<usize> = dists.iter().filter(|(i,_)| *i != pos).take(5).map(|(i,_)|*i).collect();
                // Are any actual context words in top5?
                for &t in &top5 {
                    if t.abs_diff(pos) <= window { sim_ok += 1; break; }
                }
                sim_tot += 1;
            }

            let avg_loss = if n>0{total_loss/n as f32}else{0.0};
            let sim_acc = if sim_tot>0{sim_ok as f64/sim_tot as f64*100.0}else{0.0};
            println!("  {:>5} {:>8.3} {:>7.1}% {:>5.0}s", ep, avg_loss, sim_acc, t0.elapsed().as_secs_f64());
        }

        if t0.elapsed().as_secs() > 300 { break; }
    }

    // ── Step 3: Show nearest neighbors ──
    println!("\n  Step 3: Nearest neighbors in embedding space\n");

    let test_words: Vec<&str> = vec!["the", "alice", "was", "said", "cat", "queen", "little", "very", "not", "had"];
    for &tw in &test_words {
        let w: Vec<u8> = tw.bytes().map(|b| b - b'a').collect();
        if word2id.contains_key(&w) {
            let emb = encoder.embed(&w);
            let mut dists: Vec<(usize, f32)> = vocab.iter().enumerate().map(|(i,(vw,_))| {
                let e2 = encoder.embed(vw);
                let d: f32 = emb.iter().zip(&e2).map(|(a,b)|(a-b)*(a-b)).sum();
                (i, d)
            }).collect();
            dists.sort_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let nn: Vec<String> = dists.iter().filter(|(i,_)| word_to_str(&vocab[*i].0) != tw)
                .take(5).map(|(i,d)| format!("{}({:.1})", word_to_str(&vocab[*i].0), d)).collect();
            println!("  {:>10} → {}", tw, nn.join(", "));
        }
    }

    println!("\n  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
