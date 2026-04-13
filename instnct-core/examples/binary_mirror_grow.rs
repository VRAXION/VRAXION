use std::time::Instant;
fn load_unique_bytes(path: &str) -> Vec<u8> {
    let text = std::fs::read(path).expect("read"); let mut seen = [false;256];
    for &b in &text { seen[b as usize] = true; }
    (0..=255u8).filter(|&b| seen[b as usize]).collect()
}
fn byte_to_bits(b: u8) -> [f32;8] { let mut bits=[0.0f32;8]; for i in 0..8 { bits[i]=((b>>i)&1) as f32; } bits }
fn c19(x:f32,c:f32,rho:f32)->f32{let c=c.max(0.1);let rho=rho.max(0.0);let l=6.0*c;if x>=l{return x-l;}if x<=-l{return x+l;}let s=x/c;let n=s.floor();let t=s-n;let h=t*(1.0-t);let sg=if(n as i32)%2==0{1.0}else{-1.0};c*(sg*h+rho*h*h)}
fn sigmoid(x:f32)->f32{1.0/(1.0+(-x).exp())}
#[derive(Clone)] struct N{w:Vec<i32>,b:i32,c:f32,rho:f32}
impl N{fn eval(&self,inp:&[f32;8])->f32{let mut d=self.b as f32;for i in 0..8{d+=self.w[i] as f32*inp[i];}c19(d,self.c,self.rho)}}
fn eval_rt(neurons:&[N],inputs:&[[f32;8]])->(usize,[f32;8]){
    let n=inputs.len();let k=neurons.len();if k==0{return(0,[0.0;8]);}
    let hid:Vec<Vec<f32>>=inputs.iter().map(|inp|neurons.iter().map(|n|n.eval(inp)).collect()).collect();
    let mut rz=vec![[0.0f32;8];n];
    for bi in 0..n{for j in 0..8{let mut z=0.0f32;for ki in 0..k{z+=neurons[ki].w[j] as f32*hid[bi][ki];}rz[bi][j]=z;}}
    let mut bb=[0.0f32;8];
    for j in 0..8{let mut best_a=0;let mut best_b=0.0f32;
        let mut zt:Vec<(f32,f32)>=(0..n).map(|bi|(rz[bi][j],inputs[bi][j])).collect();zt.sort_by(|a,b|a.0.partial_cmp(&b.0).unwrap());
        let mut cands:Vec<f32>=vec![-100.0,100.0];for i in 0..zt.len(){cands.push(-zt[i].0);if i+1<zt.len(){cands.push(-(zt[i].0+zt[i+1].0)/2.0);}}
        for &bias in &cands{let a=(0..n).filter(|&bi|(sigmoid(rz[bi][j]+bias)-inputs[bi][j]).abs()<0.4).count();if a>best_a{best_a=a;best_b=bias;}}bb[j]=best_b;}
    let mut c=0;for bi in 0..n{if(0..8).all(|j|(sigmoid(rz[bi][j]+bb[j])-inputs[bi][j]).abs()<0.4){c+=1;}}(c,bb)
}
fn main(){
    let t0=Instant::now();
    let unique=load_unique_bytes("instnct-core/tests/fixtures/alice_corpus.txt");
    let inputs:Vec<[f32;8]>=unique.iter().map(|&b|byte_to_bits(b)).collect();
    let cg:Vec<f32>=vec![0.1,0.5,1.0,2.0,3.0,5.0,8.0,12.0,20.0];
    let rg:Vec<f32>=vec![0.0,0.5,1.0,2.0,5.0,10.0];
    // Binary {0,1} and signed binary {-1,+1}
    for &(label,vals) in &[("binary{0,1}",&[0i32,1][..]),("signed{-1,+1}",&[-1i32,1][..]),("ternary{-1,0,1}",&[-1i32,0,1][..])]{
        let nv=vals.len();let combos=nv.pow(9)*cg.len()*rg.len();
        println!("━━━ {} — {} combos/neuron ━━━",label,combos);
        let mut neurons:Vec<N>=Vec::new();let mut acc=0;
        for step in 0..20{
            let mut best_n:Option<N>=None;let mut best_a=acc;let ts=Instant::now();
            for &c in &cg{for &rho in &rg{
                for combo in 0..nv.pow(9){let mut r=combo;let mut w=[0i32;8];
                    for wi in &mut w{*wi=vals[r%nv];r/=nv;}let b=vals[r%nv];
                    if w.iter().all(|&x|x==0){continue;}
                    let n=N{w:w.to_vec(),b,c,rho};let mut tn=neurons.clone();tn.push(n.clone());
                    let(a,_)=eval_rt(&tn,&inputs);
                    if a>best_a{best_a=a;best_n=Some(n);if a==unique.len(){break;}}
                }if best_a==unique.len(){break;}}if best_a==unique.len(){break;}}
            if let Some(n)=best_n{
                acc=best_a;let ws:Vec<String>=n.w.iter().map(|w|format!("{:>2}",w)).collect();
                println!("  N{}: {}/{} [{}] b={} c={:.1} ρ={:.1} ({:.1}s)",
                    step,acc,unique.len(),ws.join(","),n.b,n.c,n.rho,ts.elapsed().as_secs_f64());
                neurons.push(n);
                if acc==unique.len(){println!("  ★★★ PERFECT: {} neurons ★★★",neurons.len());break;}
            }else{println!("  N{}: no improvement, stop",step);break;}
        }
        println!("  Final: {}/{}, {} neurons, {:.1}s\n",acc,unique.len(),neurons.len(),t0.elapsed().as_secs_f64());
        if acc==unique.len(){break;} // skip remaining if solved
    }
    println!("Total: {:.1}s",t0.elapsed().as_secs_f64());
}
