//! L2 Hierarchical MLP — sweep block size × activation × hidden dim
//!
//! Tiled shared MLPs, hierarchical merge, masked char prediction
//! Sweep: mergers_per_block (2,4,8), hidden (32,64,128), act (swish,c19)
//!
//! Run: cargo run --example l2_hier_sweep --release

use std::time::Instant;

const LUT: [[i8; 2]; 27] = [
    [-2,-4],[-4,-2],[0,-6],[-2,-5],[-1,-6],[-3,-5],[1,-8],[-1,-7],
    [-2,-6],[-4,-5],[0,-8],[-2,-7],[-1,-8],[-3,-7],[1,-10],[-1,-9],
    [-4,-6],[-6,-5],[-2,-8],[-4,-7],[-3,-8],[-5,-7],[-1,-10],[-3,-9],
    [-4,-8],[-6,-7],[-2,-10],
];

fn swish(x:f32)->f32{x/(1.0+(-x).exp())}
fn swish_g(x:f32)->f32{let s=1.0/(1.0+(-x).exp());s+x*s*(1.0-s)}
fn c19a(x:f32,c:f32,r:f32)->f32{let c=c.max(0.1);let r=r.max(0.0);let l=6.0*c;if x>=l{return x-l;}if x<=-l{return x+l;}let s=x/c;let n=s.floor();let t=s-n;let h=t*(1.0-t);let sg=if(n as i32)%2==0{1.0}else{-1.0};c*(sg*h+r*h*h)}
fn c19g(x:f32,c:f32,r:f32)->f32{let c=c.max(0.1);let r=r.max(0.0);let l=6.0*c;if x>=l||x<=-l{return 1.0;}let s=x/c;let n=s.floor();let t=s-n;let h=t*(1.0-t);let sg=if(n as i32)%2==0{1.0}else{-1.0};(sg+2.0*r*h)*(1.0-2.0*t)}

struct Rng(u64);
impl Rng{
    fn new(s:u64)->Self{Rng(s.wrapping_mul(6364136223846793005).wrapping_add(1))}
    fn next(&mut self)->u64{self.0=self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);self.0}
    fn normal(&mut self)->f32{let u1=(((self.next()>>33)%65536)as f32/65536.0).max(1e-7);let u2=((self.next()>>33)%65536)as f32/65536.0;(-2.0*u1.ln()).sqrt()*(2.0*std::f32::consts::PI*u2).cos()}
    fn range(&mut self,lo:usize,hi:usize)->usize{if hi<=lo{lo}else{lo+(self.next()as usize%(hi-lo))}}
}

fn load_corpus(p:&str)->Vec<u8>{std::fs::read(p).expect("r").iter().filter_map(|&b|match b{b'a'..=b'z'=>Some(b-b'a'),b'A'..=b'Z'=>Some(b-b'A'),b' '|b'\n'|b'\t'|b'\r'=>Some(26),_=>None}).collect()}

struct Block {
    w1:Vec<Vec<f32>>,b1:Vec<f32>,
    w2:Vec<Vec<f32>>,b2:Vec<f32>,
    c1:Vec<f32>,rho1:Vec<f32>,
    ind:usize,hid:usize,outd:usize,
    use_c19:bool,
}

impl Block {
    fn new(ind:usize,hid:usize,outd:usize,use_c19:bool,rng:&mut Rng)->Self{
        let s1=(2.0/ind as f32).sqrt();let s2=(2.0/hid as f32).sqrt();
        Block{
            w1:(0..hid).map(|_|(0..ind).map(|_|rng.normal()*s1).collect()).collect(),
            b1:vec![0.0;hid],
            w2:(0..outd).map(|_|(0..hid).map(|_|rng.normal()*s2).collect()).collect(),
            b2:vec![0.0;outd],
            c1:vec![5.0;hid],rho1:vec![0.5;hid],
            ind,hid,outd,use_c19,
        }
    }
    fn params(&self)->usize{self.ind*self.hid+self.hid+self.hid*self.outd+self.outd}
    fn act(&self,x:f32,j:usize)->f32{if self.use_c19{c19a(x,self.c1[j],self.rho1[j])}else{swish(x)}}
    fn act_g(&self,x:f32,j:usize)->f32{if self.use_c19{c19g(x,self.c1[j],self.rho1[j])}else{swish_g(x)}}

    fn forward(&self,input:&[f32])->(Vec<f32>,Vec<f32>,Vec<f32>){
        let mut z1=self.b1.clone();
        for j in 0..self.hid{for k in 0..self.ind{z1[j]+=self.w1[j][k]*input[k];}}
        let a1:Vec<f32>=z1.iter().enumerate().map(|(j,&v)|self.act(v,j)).collect();
        let mut out=self.b2.clone();
        for j in 0..self.outd{for k in 0..self.hid{out[j]+=self.w2[j][k]*a1[k];}}
        (z1,a1,out)
    }

    fn backward(&mut self,input:&[f32],z1:&[f32],a1:&[f32],d_out:&[f32],lr:f32){
        let mut da1=vec![0.0f32;self.hid];
        for j in 0..self.outd{for k in 0..self.hid{
            da1[k]+=d_out[j]*self.w2[j][k];
            self.w2[j][k]-=lr*d_out[j]*a1[k];
        }self.b2[j]-=lr*d_out[j];}
        for j in 0..self.hid{
            let g=da1[j]*self.act_g(z1[j],j);
            for k in 0..self.ind{self.w1[j][k]-=lr*g*input[k];}
            self.b1[j]-=lr*g;
            if self.use_c19{
                let eps=0.01;
                let dc=(c19a(z1[j],self.c1[j]+eps,self.rho1[j])-c19a(z1[j],self.c1[j]-eps,self.rho1[j]))/(2.0*eps);
                self.c1[j]-=lr*da1[j]*dc*0.05;self.c1[j]=self.c1[j].max(0.5).min(50.0);
                let dr=(c19a(z1[j],self.c1[j],self.rho1[j]+eps)-c19a(z1[j],self.c1[j],self.rho1[j]-eps))/(2.0*eps);
                self.rho1[j]-=lr*da1[j]*dr*0.05;self.rho1[j]=self.rho1[j].max(0.0).min(5.0);
            }
        }
    }
}

fn run_config(corpus:&[u8], encoded:&[[f32;2]], split:usize,
    mpb:usize, hid:usize, use_c19:bool, feat:usize) -> (f64, f64, usize) {
    let ctx=512usize;
    let ch=2;
    let n_win=ctx/mpb; // windows at level 1
    let win_in=mpb*ch;
    let mask_byte=ctx/2;
    let mask_val=[1.0f32,1.0];

    let mut rng=Rng::new(42);
    let mut l1=Block::new(win_in,hid,feat,use_c19,&mut rng);
    let l2_in=feat*2;
    let mut l2=Block::new(l2_in,hid.min(64),feat,use_c19,&mut rng);
    let mut l3=Block::new(l2_in,hid.min(64),feat,use_c19,&mut rng);
    let mut head=Block::new(feat,hid,27,false,&mut rng); // head always swish
    let tp=l1.params()+l2.params()+l3.params()+head.params();

    let mask_win=mask_byte/mpb;
    let samples=1500.min(split.saturating_sub(ctx+1));
    let tc=Instant::now();

    for ep in 0..400 {
        let lr=0.005*(1.0-ep as f32/400.0*0.8);
        let mut rt=Rng::new(ep as u64*1000+42);

        for _ in 0..samples {
            let off=rt.range(0,split.saturating_sub(ctx+1));

            // L1 forward all windows
            let mut l1o:Vec<Vec<f32>>=Vec::new();
            let mut l1s:Vec<(Vec<f32>,Vec<f32>,Vec<f32>)>=Vec::new();
            let mut l1i:Vec<Vec<f32>>=Vec::new();
            for w in 0..n_win {
                let mut inp=Vec::with_capacity(win_in);
                for i in 0..mpb{let bi=off+w*mpb+i;
                    if w*mpb+i==mask_byte{inp.extend_from_slice(&mask_val);}
                    else if bi<encoded.len(){inp.push(encoded[bi][0]);inp.push(encoded[bi][1]);}
                    else{inp.push(0.0);inp.push(0.0);}}
                let(z,a,o)=l1.forward(&inp);l1i.push(inp);l1s.push((z,a,o.clone()));l1o.push(o);
            }

            // L2
            let n2=n_win/2;
            let mut l2o:Vec<Vec<f32>>=Vec::new();
            let mut l2s:Vec<(Vec<f32>,Vec<f32>,Vec<f32>)>=Vec::new();
            let mut l2i:Vec<Vec<f32>>=Vec::new();
            for i in 0..n2{let mut inp=l1o[i*2].clone();inp.extend(&l1o[i*2+1]);
                let(z,a,o)=l2.forward(&inp);l2i.push(inp);l2s.push((z,a,o.clone()));l2o.push(o);}

            // L3
            let n3=n2/2;
            let mut l3o:Vec<Vec<f32>>=Vec::new();
            let mut l3s:Vec<(Vec<f32>,Vec<f32>,Vec<f32>)>=Vec::new();
            let mut l3i:Vec<Vec<f32>>=Vec::new();
            for i in 0..n3.max(1){
                let mut inp=if i*2<l2o.len(){l2o[i*2].clone()}else{vec![0.0;feat]};
                if i*2+1<l2o.len(){inp.extend(&l2o[i*2+1]);}else{inp.extend(vec![0.0;feat]);}
                let(z,a,o)=l3.forward(&inp);l3i.push(inp);l3s.push((z,a,o.clone()));l3o.push(o);
            }

            // Head
            let mi3=(mask_win/4).min(l3o.len()-1);
            let(hz,ha,logits)=head.forward(&l3o[mi3]);

            let target=corpus[off+mask_byte];
            let mx=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
            let mut p=vec![0.0f32;27];let mut s=0.0f32;
            for c in 0..27{p[c]=(logits[c]-mx).exp();s+=p[c];}
            for c in 0..27{p[c]/=s;}
            if p[target as usize]<1e-10{continue;}
            p[target as usize]-=1.0;

            // Backprop head + L3
            head.backward(&l3o[mi3],&hz,&ha,&p,lr);
            if mi3<l3s.len(){
                l3.backward(&l3i[mi3],&l3s[mi3].0,&l3s[mi3].1,&p,lr);
            }
        }

        if tc.elapsed().as_secs()>60{break;}
    }

    // Eval
    let eval=|start:usize,end:usize|->f64{
        let mut rng3=Rng::new(999);let mut ok=0usize;let mut tot=0usize;
        for _ in 0..300{if end<start+ctx+1{break;}
            let off=rng3.range(start,end.saturating_sub(ctx+1));
            let mut l1o:Vec<Vec<f32>>=Vec::new();
            for w in 0..n_win{let mut inp=Vec::with_capacity(win_in);
                for i in 0..mpb{let bi=off+w*mpb+i;
                    if w*mpb+i==mask_byte{inp.extend_from_slice(&mask_val);}
                    else if bi<encoded.len(){inp.push(encoded[bi][0]);inp.push(encoded[bi][1]);}
                    else{inp.push(0.0);inp.push(0.0);}}
                l1o.push(l1.forward(&inp).2);}
            let n2=n_win/2;let mut l2o:Vec<Vec<f32>>=Vec::new();
            for i in 0..n2{let mut inp=l1o[i*2].clone();inp.extend(&l1o[i*2+1]);l2o.push(l2.forward(&inp).2);}
            let n3=n2/2;let mut l3o:Vec<Vec<f32>>=Vec::new();
            for i in 0..n3.max(1){let mut inp=if i*2<l2o.len(){l2o[i*2].clone()}else{vec![0.0;feat]};
                if i*2+1<l2o.len(){inp.extend(&l2o[i*2+1]);}else{inp.extend(vec![0.0;feat]);}
                l3o.push(l3.forward(&inp).2);}
            let mi=(mask_win/4).min(l3o.len()-1);
            let logits=head.forward(&l3o[mi]).2;
            let pred=logits.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|v|v.0).unwrap_or(0);
            if pred==corpus[off+mask_byte]as usize{ok+=1;}tot+=1;}
        if tot==0{0.0}else{ok as f64/tot as f64*100.0}
    };
    let tr=eval(0,split);let te=eval(split,corpus.len());
    (tr,te,tp)
}

fn main(){
    let t0=Instant::now();
    let corpus=load_corpus("instnct-core/tests/fixtures/alice_corpus.txt");
    let split=corpus.len()*80/100;
    let encoded:Vec<[f32;2]>=corpus.iter().map(|&ch|[LUT[ch as usize][0]as f32/16.0,LUT[ch as usize][1]as f32/16.0]).collect();

    println!("=== HIERARCHICAL MLP SWEEP ===\n");
    println!("  512 byte context, 3 levels + head, masked center char\n");

    let feat=16;
    println!("  {:>5} {:>5} {:>6} {:>7} {:>8} {:>8}",
        "mpb","hid","act","params","train%","test%");
    println!("  {}","-".repeat(48));

    for &mpb in &[2,4,8,16]{
        for &hid in &[32,64,128]{
            for &(act_name,use_c19) in &[("swish",false),("c19",true)]{
                let(tr,te,tp)=run_config(&corpus,&encoded,split,mpb,hid,use_c19,feat);
                let m=if te>30.0{" **"}else if te>25.0{" *"}else{""};
                println!("  {:>5} {:>5} {:>6} {:>7} {:>7.1}% {:>7.1}%{}",
                    mpb,hid,act_name,tp,tr,te,m);
            }
        }
    }
    println!("\n  Total time: {:.1}s",t0.elapsed().as_secs_f64());
}
