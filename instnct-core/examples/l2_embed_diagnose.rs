//! Diagnose: which inputs collide in the 99.9% ternary N=3 encoder?
//!
//! Run: cargo run --example l2_embed_diagnose --release

const LUT: [[i8; 2]; 27] = [
    [-2,-4],[-4,-2],[0,-6],[-2,-5],[-1,-6],[-3,-5],[1,-8],[-1,-7],
    [-2,-6],[-4,-5],[0,-8],[-2,-7],[-1,-8],[-3,-7],[1,-10],[-1,-9],
    [-4,-6],[-6,-5],[-2,-8],[-4,-7],[-3,-8],[-5,-7],[-1,-10],[-3,-9],
    [-4,-8],[-6,-7],[-2,-10],
];
const MW: [[i8;4];2] = [[-8,12,-7,-12],[-8,-6,-1,2]];
const MB: [i8;2] = [-1, 10];

fn merger_val(a: u8, b: u8) -> [f32; 2] {
    let i = [LUT[a as usize][0] as f32, LUT[a as usize][1] as f32,
             LUT[b as usize][0] as f32, LUT[b as usize][1] as f32];
    let mut o = [0.0f32; 2];
    for k in 0..2 { o[k] = MB[k] as f32 + MW[k][0] as f32*i[0] + MW[k][1] as f32*i[1]
        + MW[k][2] as f32*i[2] + MW[k][3] as f32*i[3]; }
    [o[0]/16.0, o[1]/16.0]
}

fn c19a(x:f32,c:f32)->f32{let c=c.max(0.1);let l=6.0*c;
    if x>=l{return x-l;}if x<=-l{return x+l;}
    let s=x/c;let n=s.floor();let t=s-n;let h=t*(1.0-t);
    let sg=if(n as i32)%2==0{1.0}else{-1.0};c*sg*h}

fn main() {
    let chars = "abcdefghijklmnopqrstuvwxyz ";

    // Best ternary N=3 config from sweep:
    // N1: w=[-1, 0, 1, -1] b=-1 c=50
    // N2: w=[-1, -1, -1, 0] b=-1 c=50
    // N3: w=[-1, -1, -1, -1] b=-1 c=1
    let neurons: Vec<([i8;4], i8, f32)> = vec![
        ([-1, 0, 1, -1], -1, 50.0),
        ([-1, -1, -1, 0], -1, 50.0),
        ([-1, -1, -1, -1], -1, 1.0),
    ];

    // Compute codes for ALL 531,441 possible inputs
    println!("Computing all 531,441 codes...\n");

    let pair_vals: Vec<[f32;2]> = (0..27u8).flat_map(|a|
        (0..27u8).map(move |b| merger_val(a, b))
    ).collect();

    struct Entry { p1: usize, p2: usize, code: [f32; 3] }
    let mut entries: Vec<Entry> = Vec::new();

    for p1 in 0..729 {
        for p2 in 0..729 {
            let inp = [pair_vals[p1][0], pair_vals[p1][1], pair_vals[p2][0], pair_vals[p2][1]];
            let mut code = [0.0f32; 3];
            for (ni, &(ref w, b, c)) in neurons.iter().enumerate() {
                let dot = b as f32 + w[0] as f32*inp[0]+w[1] as f32*inp[1]
                    +w[2] as f32*inp[2]+w[3] as f32*inp[3];
                code[ni] = c19a(dot, c);
            }
            entries.push(Entry { p1, p2, code });
        }
    }

    // Find collisions: entries with identical (or very close) codes but different inputs
    println!("Finding collisions...\n");

    let mut collisions = 0;
    let mut shown = 0;
    for i in 0..entries.len() {
        // Find nearest neighbor
        let mut best_j = i;
        let mut best_d = f32::MAX;
        // Only check a sample (full 531K² is too slow)
        // Instead: hash codes and find duplicates
        break; // too slow for brute force
    }

    // Better: group by quantized code and find groups with >1 entry
    use std::collections::HashMap;
    let mut code_groups: HashMap<[i32;3], Vec<usize>> = HashMap::new();
    for (i, e) in entries.iter().enumerate() {
        // Quantize to detect near-collisions
        let key = [(e.code[0]*1000.0) as i32, (e.code[1]*1000.0) as i32, (e.code[2]*1000.0) as i32];
        code_groups.entry(key).or_insert_with(Vec::new).push(i);
    }

    let collision_groups: Vec<_> = code_groups.iter().filter(|(_, v)| v.len() > 1).collect();
    println!("  Total unique quantized codes: {}", code_groups.len());
    println!("  Collision groups (same code, different input): {}\n", collision_groups.len());

    let pair_to_str = |p: usize| -> String {
        let a = p / 27; let b = p % 27;
        format!("{}{}", chars.as_bytes()[a] as char, chars.as_bytes()[b] as char)
    };

    // Show first 20 collision groups
    let mut sorted_groups: Vec<_> = collision_groups.iter().collect();
    sorted_groups.sort_by(|a,b| b.1.len().cmp(&a.1.len()));

    for (gi, (key, members)) in sorted_groups.iter().enumerate().take(20) {
        println!("  Collision #{} (code≈[{:.1},{:.1},{:.1}], {} members):",
            gi+1, key[0] as f32/1000.0, key[1] as f32/1000.0, key[2] as f32/1000.0, members.len());
        for &mi in members.iter().take(5) {
            let e = &entries[mi];
            println!("    pair1='{}' pair2='{}' → code=[{:.4},{:.4},{:.4}]",
                pair_to_str(e.p1), pair_to_str(e.p2), e.code[0], e.code[1], e.code[2]);
        }
        if members.len() > 5 { println!("    ... and {} more", members.len()-5); }
        println!();
    }

    // Stats
    let total_colliding: usize = collision_groups.iter().map(|(_, v)| v.len()).sum();
    println!("  Summary:");
    println!("  Total inputs: 531,441");
    println!("  Unique codes (quantized): {}", code_groups.len());
    println!("  Collision groups: {}", collision_groups.len());
    println!("  Inputs in collisions: {}", total_colliding);
    println!("  Collision-free: {}", 531441 - total_colliding);
    println!("  Round-trip accuracy: {:.2}%", (531441 - total_colliding) as f64 / 531441.0 * 100.0);
}
