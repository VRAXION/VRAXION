//! Character Embedding — 16-dim, backprop training, int8 quantize
//!
//! Architecture: 2000 slots × 16 dim (universal)
//! Training: masked char prediction on Alice (uses 27 of 2000 slots)
//! Method: backprop float → quantize int8 → verify structure
//! Deploy: 32 KB LUT, zero compute
//!
//! Run: cargo run --example char_embedding_train --release

use std::time::Instant;

const VOCAB: usize = 2000;
const DIM: usize = 16;

fn swish(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
fn swish_g(x: f32) -> f32 { let s=1.0/(1.0+(-x).exp()); s+x*s*(1.0-s) }

struct Rng(u64);
impl Rng {
    fn new(s:u64)->Self{Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1))}
    fn next(&mut self)->u64{self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);self.0}
    fn normal(&mut self)->f32{let u1=(((self.next()>>33)%65536)as f32/65536.0).max(1e-7);let u2=((self.next()>>33)%65536)as f32/65536.0;(-2.0*u1.ln()).sqrt()*(2.0*std::f32::consts::PI*u2).cos()}
    fn range(&mut self,lo:usize,hi:usize)->usize{if hi<=lo{lo}else{lo+(self.next()as usize%(hi-lo))}}
}

fn load_corpus(p:&str)->Vec<u8>{std::fs::read(p).expect("r").iter().filter_map(|&b|match b{
    b'a'..=b'z'=>Some(b-b'a'),b'A'..=b'Z'=>Some(b-b'A'),b' '|b'\n'|b'\t'|b'\r'=>Some(26),_=>None}).collect()}

// Model: embedding + conv context + prediction head
struct EmbedModel {
    // Embedding table (float, will quantize later)
    embed: Vec<[f32; DIM]>,  // VOCAB × DIM
    // Conv k=5: local context enrichment
    conv_w: Vec<Vec<f32>>,   // n_filters × (5 × DIM)
    conv_b: Vec<f32>,
    n_filters: usize,
    // Head: conv_output → 27 (predict masked char)
    head_w: Vec<Vec<f32>>,   // 27 × n_filters
    head_b: Vec<f32>,
}

impl EmbedModel {
    fn new(n_filters: usize, rng: &mut Rng) -> Self {
        let sc_e = (1.0/DIM as f32).sqrt();
        let sc_c = (2.0/(5*DIM) as f32).sqrt();
        let sc_h = (2.0/n_filters as f32).sqrt();
        EmbedModel {
            embed: (0..VOCAB).map(|_| {
                let mut v = [0.0f32; DIM];
                for d in 0..DIM { v[d] = rng.normal() * sc_e; }
                v
            }).collect(),
            conv_w: (0..n_filters).map(|_| (0..5*DIM).map(|_| rng.normal()*sc_c).collect()).collect(),
            conv_b: vec![0.0; n_filters],
            n_filters,
            head_w: (0..27).map(|_| (0..n_filters).map(|_| rng.normal()*sc_h).collect()).collect(),
            head_b: vec![0.0; 27],
        }
    }

    fn params(&self) -> usize {
        27 * DIM + // only 27 chars used in Alice
        self.n_filters * 5 * DIM + self.n_filters +
        27 * self.n_filters + 27
    }

    fn train_step(&mut self, chars: &[u8], mask_pos: usize, lr: f32) -> (f32, bool) {
        let ctx = chars.len();
        let target = chars[mask_pos] as usize;

        // Forward: embed all positions (mask gets zero)
        let mut emb_out: Vec<[f32; DIM]> = Vec::with_capacity(ctx);
        for i in 0..ctx {
            if i == mask_pos {
                emb_out.push([0.0; DIM]); // mask = zero embedding
            } else {
                emb_out.push(self.embed[chars[i] as usize]);
            }
        }

        // Conv k=5 at mask position: read positions [mask-2..mask+2]
        let mut conv_out = vec![0.0f32; self.n_filters];
        let mut conv_pre = vec![0.0f32; self.n_filters];
        for f in 0..self.n_filters {
            let mut v = self.conv_b[f];
            for ki in 0..5 {
                let pos = mask_pos as i32 + ki as i32 - 2;
                if pos >= 0 && (pos as usize) < ctx {
                    for d in 0..DIM {
                        v += self.conv_w[f][ki*DIM+d] * emb_out[pos as usize][d];
                    }
                }
            }
            conv_pre[f] = v;
            conv_out[f] = swish(v);
        }

        // Head: predict masked char
        let mut logits = self.head_b.clone();
        for c in 0..27 { for f in 0..self.n_filters { logits[c] += self.head_w[c][f] * conv_out[f]; } }

        // Softmax + CE
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut p = vec![0.0f32;27]; let mut s=0.0f32;
        for c in 0..27{p[c]=(logits[c]-mx).exp();s+=p[c];}
        for c in 0..27{p[c]/=s;}
        let loss = -(p[target].max(1e-10).ln());
        let pred = logits.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|v|v.0).unwrap_or(0);
        p[target] -= 1.0;

        // Backprop head
        let mut d_conv = vec![0.0f32; self.n_filters];
        for c in 0..27 {
            for f in 0..self.n_filters {
                d_conv[f] += p[c] * self.head_w[c][f];
                self.head_w[c][f] -= lr * p[c] * conv_out[f];
            }
            self.head_b[c] -= lr * p[c];
        }

        // Backprop conv (swish)
        let mut d_emb = vec![[0.0f32; DIM]; ctx];
        for f in 0..self.n_filters {
            let g = d_conv[f] * swish_g(conv_pre[f]);
            for ki in 0..5 {
                let pos = mask_pos as i32 + ki as i32 - 2;
                if pos >= 0 && (pos as usize) < ctx {
                    let pi = pos as usize;
                    for d in 0..DIM {
                        d_emb[pi][d] += g * self.conv_w[f][ki*DIM+d];
                        self.conv_w[f][ki*DIM+d] -= lr * g * emb_out[pi][d];
                    }
                }
            }
            self.conv_b[f] -= lr * g;
        }

        // Backprop embedding (skip mask position)
        for i in 0..ctx {
            if i == mask_pos { continue; }
            let ch = chars[i] as usize;
            for d in 0..DIM {
                self.embed[ch][d] -= lr * d_emb[i][d];
            }
        }

        (loss, pred == target)
    }
}

fn main() {
    let t0 = Instant::now();
    let corpus = load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let split = corpus.len() * 80 / 100;
    let ctx = 32usize;

    println!("=== CHARACTER EMBEDDING TRAINING ===\n");
    println!("  Vocab: {} slots, Dim: {}, Table: {} bytes", VOCAB, DIM, VOCAB*DIM);
    println!("  Used: 27 chars (Alice), ctx={}, mask=center", ctx);

    let n_filters = 64;
    let mut rng = Rng::new(42);
    let mut model = EmbedModel::new(n_filters, &mut rng);
    println!("  Conv: k=5, f={}, Head: 27", n_filters);
    println!("  Trainable params: ~{}\n", model.params());

    let samples = 5000;
    let max_ep = 2000;
    let mask_pos = ctx / 2;

    println!("  {:>5} {:>7} {:>8} {:>8} {:>6}",
        "epoch", "loss", "train%", "test%", "time");
    println!("  {}", "-".repeat(42));

    let mut best_test = 0.0f64;

    for ep in 0..max_ep {
        let lr = 0.01 * (1.0 - ep as f32 / max_ep as f32 * 0.8);
        let mut rt = Rng::new(ep as u64 * 1000 + 42);
        let mut tloss = 0.0f32; let mut n = 0u32; let mut tok = 0u32;

        for _ in 0..samples {
            let off = rt.range(0, split.saturating_sub(ctx + 1));
            let chunk = &corpus[off..off+ctx];
            let (loss, correct) = model.train_step(chunk, mask_pos, lr);
            if !loss.is_nan() { tloss += loss; n += 1; if correct { tok += 1; } }
        }

        if ep % 50 == 0 {
            let eval = |start:usize, end:usize| -> f64 {
                let mut rng3 = Rng::new(999);
                let mut ok=0usize; let mut tot=0usize;
                for _ in 0..1000 {
                    if end < start+ctx+1 { break; }
                    let off = rng3.range(start, end.saturating_sub(ctx+1));
                    let chunk = &corpus[off..off+ctx];
                    // Forward only
                    let mut emb: Vec<[f32;DIM]> = chunk.iter().enumerate().map(|(i,&ch)|
                        if i==mask_pos{[0.0;DIM]}else{model.embed[ch as usize]}
                    ).collect();
                    let mut conv = vec![0.0f32;model.n_filters];
                    for f in 0..model.n_filters{let mut v=model.conv_b[f];
                        for ki in 0..5{let pos=mask_pos as i32+ki as i32-2;
                            if pos>=0&&(pos as usize)<ctx{for d in 0..DIM{
                                v+=model.conv_w[f][ki*DIM+d]*emb[pos as usize][d];}}}
                        conv[f]=swish(v);}
                    let mut logits=model.head_b.clone();
                    for c in 0..27{for f in 0..model.n_filters{logits[c]+=model.head_w[c][f]*conv[f];}}
                    let pred=logits.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|v|v.0).unwrap_or(0);
                    if pred==chunk[mask_pos] as usize{ok+=1;} tot+=1;
                }
                if tot==0{0.0}else{ok as f64/tot as f64*100.0}
            };
            let tr = eval(0, split);
            let te = eval(split, corpus.len());
            if te > best_test { best_test = te; }
            println!("  {:>5} {:>7.3} {:>7.1}% {:>7.1}% {:>5.0}s",
                ep, if n>0{tloss/n as f32}else{0.0}, tr, te, t0.elapsed().as_secs_f64());
        }

        if t0.elapsed().as_secs() > 300 { break; }
    }

    // ── Embedding structure analysis ──
    println!("\n--- Learned embedding structure ---\n");
    let chars = "abcdefghijklmnopqrstuvwxyz ";
    let vowels = [0,4,8,14,20]; // a,e,i,o,u
    let consonants: Vec<usize> = (0..26).filter(|c| !vowels.contains(c)).collect();

    // Avg distance vowel-vowel vs vowel-consonant
    let dist = |a:usize,b:usize| -> f32 {
        (0..DIM).map(|d| (model.embed[a][d]-model.embed[b][d]).powi(2)).sum::<f32>().sqrt()
    };

    let mut vv_d=0.0f32; let mut vv_n=0u32;
    let mut vc_d=0.0f32; let mut vc_n=0u32;
    let mut cc_d=0.0f32; let mut cc_n=0u32;
    for &i in &vowels { for &j in &vowels { if i<j { vv_d+=dist(i,j); vv_n+=1; }}}
    for &i in &vowels { for &j in &consonants { vc_d+=dist(i,j); vc_n+=1; }}
    for i in 0..consonants.len() { for j in (i+1)..consonants.len() { cc_d+=dist(consonants[i],consonants[j]); cc_n+=1; }}

    println!("  Avg distance vowel↔vowel:     {:.3}", if vv_n>0{vv_d/vv_n as f32}else{0.0});
    println!("  Avg distance vowel↔consonant: {:.3}", if vc_n>0{vc_d/vc_n as f32}else{0.0});
    println!("  Avg distance cons↔consonant:  {:.3}", if cc_n>0{cc_d/cc_n as f32}else{0.0});
    println!("  (If vowels cluster: V↔V < V↔C)\n");

    // Nearest neighbors for each char
    println!("  Nearest neighbors in 16-dim embedding:");
    for c in 0..27 {
        let mut dists: Vec<(usize,f32)> = (0..27).filter(|&j|j!=c).map(|j|(j,dist(c,j))).collect();
        dists.sort_by(|a,b|a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let nn: Vec<String> = dists.iter().take(3).map(|(j,d)|
            format!("{}({:.2})", chars.as_bytes()[*j] as char, d)).collect();
        let label = if vowels.contains(&c){"V"}else if c==26{"_"}else{"C"};
        println!("    {} '{}': {}", label, chars.as_bytes()[c] as char, nn.join(", "));
    }

    // ── Quantize to int8 ──
    println!("\n--- Quantize to int8 ---\n");
    let mut min_v=f32::MAX; let mut max_v=f32::MIN;
    for c in 0..27 { for d in 0..DIM {
        if model.embed[c][d]<min_v{min_v=model.embed[c][d];}
        if model.embed[c][d]>max_v{max_v=model.embed[c][d];}
    }}
    let scale = if max_v-min_v>0.0{254.0/(max_v-min_v)}else{1.0};

    let quant: Vec<[i8;DIM]> = (0..27).map(|c|{
        let mut q=[0i8;DIM];
        for d in 0..DIM{q[d]=((model.embed[c][d]-min_v)*scale-127.0).round().max(-128.0).min(127.0)as i8;}
        q
    }).collect();

    // Check uniqueness
    let mut all_unique = true;
    for i in 0..27{for j in(i+1)..27{
        let d:i32=(0..DIM).map(|d|(quant[i][d]as i32-quant[j][d]as i32).pow(2)).sum();
        if d==0{println!("  COLLISION: {} and {}", chars.as_bytes()[i]as char, chars.as_bytes()[j]as char); all_unique=false;}
    }}
    if all_unique{println!("  All 27 chars UNIQUE in int8 ✓");}

    println!("\n  Best test: {:.1}%", best_test);
    println!("  Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
