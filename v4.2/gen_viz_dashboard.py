"""Generate multi-panel network dashboard."""
import json, sys
import numpy as np
sys.path.insert(0, 's:/AI/work/VRAXION_DEV/v4.2/model')
from graph import SelfWiringGraph
import random

# Load network
IO = 256
random.seed(42); np.random.seed(42)
net = SelfWiringGraph(IO)
d = np.load('s:/AI/work/VRAXION_DEV/v4.2/checkpoints/english_768n_step3000.npz')
net.mask[:] = 0; net.mask[d['rows'], d['cols']] = d['vals']
net.resync_alive()

H = net.H
edges = list(net.alive)

# Compute metrics
out_deg = np.array([(net.mask[i, :] != 0).sum() for i in range(H)])
in_deg = np.array([(net.mask[:, j] != 0).sum() for j in range(H)])
total_deg = out_deg + in_deg

# Degree distribution histogram
deg_hist = np.bincount(total_deg, minlength=20)[:20].tolist()

# Pos/neg per neuron
pos_out = np.array([(net.mask[i, :] > 0).sum() for i in range(H)])
neg_out = np.array([(net.mask[i, :] < 0).sum() for i in range(H)])

# Adjacency matrix (downsampled for viz)
# Group neurons into 32 buckets, show inter-bucket connectivity
N_BUCKETS = 32
bucket_size = H // N_BUCKETS
adj_matrix = np.zeros((N_BUCKETS, N_BUCKETS), dtype=float)
for r, c in edges:
    br, bc = min(r // bucket_size, N_BUCKETS-1), min(c // bucket_size, N_BUCKETS-1)
    adj_matrix[br][bc] += 1

# Normalize
adj_max = adj_matrix.max() if adj_matrix.max() > 0 else 1
adj_norm = (adj_matrix / adj_max).tolist()

# Edge length distribution (how far apart are connected neurons by ID)
edge_distances = [abs(r - c) for r, c in edges]
dist_hist = np.bincount(np.clip(np.array(edge_distances) // (H // 20), 0, 19), minlength=20)[:20].tolist()

# Clustering: build adjacency dict for faster lookup
from collections import Counter
adj_dict = {}
for r, c in edges:
    adj_dict.setdefault(r, []).append(c)
    adj_dict.setdefault(c, []).append(r)

# Label propagation with randomized init (multiple seeds to break symmetry)
best_n_clusters = 0
best_labels = None
for seed_lp in range(10):
    rng_lp = random.Random(seed_lp * 1000 + 7)
    labels = [rng_lp.randint(0, 11) for _ in range(H)]  # random init into 12 groups
    for iteration in range(50):
        changed = 0
        order = list(range(H))
        rng_lp.shuffle(order)
        for node in order:
            if total_deg[node] == 0:
                continue
            neighbors = adj_dict.get(node, [])
            if not neighbors:
                continue
            neighbor_labels = Counter(labels[nb] for nb in neighbors)
            best_label = neighbor_labels.most_common(1)[0][0]
            if labels[node] != best_label:
                labels[node] = best_label
                changed += 1
        if changed == 0:
            break
    nc = len(set(labels[i] for i in range(H) if total_deg[i] > 0))
    if nc > best_n_clusters:
        best_n_clusters = nc
        best_labels = labels[:]

if best_labels is None or best_n_clusters <= 1:
    # Fallback: cluster by degree quartile + pos/neg ratio
    print("Label propagation found 1 cluster, using degree-based grouping")
    best_labels = []
    for i in range(H):
        if total_deg[i] == 0:
            best_labels.append(-1)
        else:
            # 4 groups by degree, 3 by pos/neg ratio = 12 clusters
            dq = min(3, int(total_deg[i] / (total_deg.max()/4 + 1)))
            pr = 0 if pos_out[i] > neg_out[i] else (1 if neg_out[i] > pos_out[i] else 2)
            best_labels.append(dq * 3 + pr)
    best_n_clusters = len(set(l for l in best_labels if l >= 0))

labels = best_labels
cluster_counts = Counter(labels[i] for i in range(H) if total_deg[i] > 0)
top_clusters = cluster_counts.most_common(12)
cluster_sizes = [c for _, c in top_clusters]
cluster_ids = {cid: idx for idx, (cid, _) in enumerate(top_clusters)}

neuron_cluster = []
for i in range(H):
    if total_deg[i] == 0:
        neuron_cluster.append(-1)
    elif labels[i] in cluster_ids:
        neuron_cluster.append(cluster_ids[labels[i]])
    else:
        neuron_cluster.append(len(top_clusters))

# Signal flow: simulate forward pass with a test byte
def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

bp = make_bp(IO)
rows_s = np.array([r for r, c in edges], dtype=np.intp)
cols_s = np.array([c for r, c in edges], dtype=np.intp)
vals_s = net.mask[rows_s, cols_s]

# Feed byte 't' (116) and record activation per tick
activations_per_tick = []
charge_sim = np.zeros(H, dtype=np.float32)
act = np.zeros(H, dtype=np.float32)
retain = float(net.retention)
for t in range(6):
    if t == 0:
        act = act + bp[116] @ net.W_in  # 't'
    raw = np.zeros(H, dtype=np.float32)
    if len(rows_s):
        np.add.at(raw, cols_s, act[rows_s] * vals_s)
    charge_sim += raw
    charge_sim *= retain
    act = np.maximum(charge_sim - net.THRESHOLD, 0.0)
    charge_sim = np.clip(charge_sim, -1.0, 1.0)
    # Record neurons by charge magnitude (before threshold)
    tick_data = [(int(i), round(float(abs(charge_sim[i])), 4)) for i in range(H) if abs(charge_sim[i]) > 0.01]
    activations_per_tick.append(tick_data)
    print(f"  Tick {t}: {len(tick_data)} active neurons, charge_max={abs(charge_sim).max():.4f}, act_max={act.max():.4f}")

# Build dashboard data
dashboard = {
    'neurons': int(H),
    'edges': len(edges),
    'pos_edges': int(sum(1 for r, c in edges if net.mask[r, c] > 0)),
    'neg_edges': int(sum(1 for r, c in edges if net.mask[r, c] < 0)),
    'max_deg': int(total_deg.max()),
    'avg_deg': round(float(total_deg[total_deg > 0].mean()), 1),
    'density': round(len(edges) / (H * (H - 1)) * 100, 3),
    'deg_hist': deg_hist,
    'dist_hist': dist_hist,
    'adj_matrix': adj_norm,
    'adj_buckets': N_BUCKETS,
    'cluster_sizes': cluster_sizes,
    'n_clusters': len(top_clusters),
    'ticks_data': activations_per_tick,
    'nodes': [{
        'id': int(i),
        'deg': int(total_deg[i]),
        'out': int(out_deg[i]),
        'inp': int(in_deg[i]),
        'pos': int(pos_out[i]),
        'neg': int(neg_out[i]),
        'cluster': int(neuron_cluster[i]),
    } for i in range(H) if total_deg[i] > 0],
    'links': [{
        's': int(r), 't': int(c),
        'v': float(net.mask[r, c]),
    } for r, c in edges],
}

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return super().default(obj)

compact = json.dumps(dashboard, separators=(',', ':'), cls=NpEncoder)
print(f"Dashboard data: {len(compact)} bytes")
print(f"Clusters: {len(top_clusters)} (top sizes: {cluster_sizes[:6]})")
print(f"Tick activations: {[len(t) for t in activations_per_tick]}")

# HTML
html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SWG Network Dashboard</title>
<style>
@import url("https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Outfit:wght@200;300;400;700&display=swap");
:root{--bg:#060810;--card:rgba(10,16,28,0.95);--border:rgba(0,255,136,0.12);--accent:#00ff88;--neg:#ff3355;--text:#b8c4d0;--dim:#3a4858;--blue:#4488ff;--orange:#ff8844;--purple:#aa66ff}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:"JetBrains Mono",monospace;overflow-y:auto;overflow-x:hidden}
.header{padding:24px 32px 12px;border-bottom:1px solid var(--border)}
.header h1{font-family:"Outfit",sans-serif;font-weight:200;font-size:28px;color:var(--accent);letter-spacing:6px;text-transform:uppercase;text-shadow:0 0 30px rgba(0,255,136,0.2)}
.header .sub{font-size:10px;color:var(--dim);letter-spacing:3px;margin-top:4px}
.stats-bar{display:flex;gap:24px;padding:12px 32px;border-bottom:1px solid var(--border);flex-wrap:wrap}
.stat{text-align:center}
.stat .num{font-family:"Outfit";font-weight:700;font-size:22px;color:var(--accent)}
.stat .num.neg-c{color:var(--neg)}
.stat .num.blue-c{color:var(--blue)}
.stat .lbl{font-size:9px;color:var(--dim);letter-spacing:1px;text-transform:uppercase;margin-top:2px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:16px 32px}
@media(max-width:1200px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--border);border-radius:3px;padding:16px;min-height:200px}
.card h2{font-family:"Outfit";font-weight:300;font-size:14px;color:var(--accent);letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;padding-bottom:6px;border-bottom:1px solid rgba(0,255,136,0.08)}
canvas{display:block;width:100%;border-radius:2px}
.full-width{grid-column:1/-1}
.tick-label{display:inline-block;padding:2px 8px;margin:2px;font-size:9px;border:1px solid var(--border);border-radius:2px;cursor:pointer}
.tick-label:hover,.tick-label.active{background:rgba(0,255,136,0.15);border-color:var(--accent);color:#fff}
.legend{display:flex;gap:16px;margin-top:8px;font-size:9px;flex-wrap:wrap}
.legend span{display:flex;align-items:center;gap:4px}
.legend .dot{width:8px;height:8px;border-radius:50%;display:inline-block}
</style>
</head>
<body>
<div class="header">
<h1>SWG Network Dashboard</h1>
<div class="sub">768 neurons · self-wiring graph · english next-byte · interactive analysis</div>
</div>
<div class="stats-bar">
<div class="stat"><div class="num" id="s-n">-</div><div class="lbl">neurons</div></div>
<div class="stat"><div class="num" id="s-e">-</div><div class="lbl">synapses</div></div>
<div class="stat"><div class="num" id="s-p">-</div><div class="lbl">excitatory</div></div>
<div class="stat"><div class="num neg-c" id="s-ng">-</div><div class="lbl">inhibitory</div></div>
<div class="stat"><div class="num blue-c" id="s-d">-</div><div class="lbl">avg degree</div></div>
<div class="stat"><div class="num" id="s-dn">-</div><div class="lbl">density</div></div>
<div class="stat"><div class="num blue-c" id="s-cl">-</div><div class="lbl">clusters</div></div>
</div>
<div class="grid">

<div class="card">
<h2>Degree Distribution</h2>
<canvas id="c-deg" height="180"></canvas>
</div>

<div class="card">
<h2>Edge Distance Distribution</h2>
<canvas id="c-dist" height="180"></canvas>
<div class="legend"><span>how far apart are connected neurons (by neuron ID)</span></div>
</div>

<div class="card">
<h2>Connectivity Matrix (32x32 buckets)</h2>
<canvas id="c-adj" height="300"></canvas>
<div class="legend"><span>brighter = more connections between neuron groups</span></div>
</div>

<div class="card">
<h2>Cluster Sizes (label propagation)</h2>
<canvas id="c-clust" height="300"></canvas>
<div class="legend"><span>neurons that talk to each other group into communities</span></div>
</div>

<div class="card full-width">
<h2>Signal Propagation — feed byte 't' through 6 ticks</h2>
<div style="margin-bottom:8px">
<span class="tick-label active" onclick="setTick(0)">tick 0</span>
<span class="tick-label" onclick="setTick(1)">tick 1</span>
<span class="tick-label" onclick="setTick(2)">tick 2</span>
<span class="tick-label" onclick="setTick(3)">tick 3</span>
<span class="tick-label" onclick="setTick(4)">tick 4</span>
<span class="tick-label" onclick="setTick(5)">tick 5</span>
<span class="tick-label" onclick="setTick(-1)">all ticks</span>
</div>
<canvas id="c-ticks" height="250"></canvas>
<div class="legend">
<span><span class="dot" style="background:#00ff88"></span> tick 0 (input injection)</span>
<span><span class="dot" style="background:#4488ff"></span> tick 1</span>
<span><span class="dot" style="background:#aa66ff"></span> tick 2</span>
<span><span class="dot" style="background:#ff8844"></span> tick 3</span>
<span><span class="dot" style="background:#ff3355"></span> tick 4</span>
<span><span class="dot" style="background:#ffcc00"></span> tick 5</span>
</div>
</div>

<div class="card full-width">
<h2>Network Graph — colored by cluster</h2>
<canvas id="c-graph" height="500"></canvas>
<div class="legend" id="graph-legend"></div>
</div>

</div>

<script>
const D=__DATA__;

// Stats
document.getElementById("s-n").textContent=D.neurons;
document.getElementById("s-e").textContent=D.edges;
document.getElementById("s-p").textContent=D.pos_edges;
document.getElementById("s-ng").textContent=D.neg_edges;
document.getElementById("s-d").textContent=D.avg_deg;
document.getElementById("s-dn").textContent=D.density+"%";
document.getElementById("s-cl").textContent=D.n_clusters;

const COLORS=["#00ff88","#4488ff","#ff3355","#aa66ff","#ff8844","#ffcc00","#00ccff","#ff66aa","#88ff44","#ff4400","#44ffcc","#8866ff"];
const TICK_COLORS=["#00ff88","#4488ff","#aa66ff","#ff8844","#ff3355","#ffcc00"];

// Helper: bar chart
function barChart(canvasId, data, color, labels){
  const cv=document.getElementById(canvasId);
  const cx=cv.getContext("2d");
  const W=cv.offsetWidth, H=cv.offsetHeight;
  cv.width=W*2;cv.height=H*2;cx.scale(2,2);
  const max=Math.max(...data,1);
  const pad={t:10,b:20,l:30,r:10};
  const bw=(W-pad.l-pad.r)/data.length;
  // Grid
  cx.strokeStyle="rgba(0,255,136,0.06)";cx.lineWidth=0.5;
  for(let i=0;i<5;i++){const y=pad.t+(H-pad.t-pad.b)*i/4;cx.beginPath();cx.moveTo(pad.l,y);cx.lineTo(W-pad.r,y);cx.stroke()}
  // Bars
  data.forEach((v,i)=>{
    const bh=(v/max)*(H-pad.t-pad.b);
    const x=pad.l+i*bw+1;
    const y=H-pad.b-bh;
    cx.fillStyle=typeof color==="string"?color:color[i%color.length];
    cx.globalAlpha=0.8;
    cx.fillRect(x,y,bw-2,bh);
    cx.globalAlpha=1;
  });
  // Labels
  cx.fillStyle="#3a4858";cx.font="8px JetBrains Mono";cx.textAlign="center";
  data.forEach((v,i)=>{
    if(labels&&labels[i]!=null&&i%Math.ceil(data.length/10)===0)
      cx.fillText(labels[i],pad.l+i*bw+bw/2,H-4);
  });
  // Y axis
  cx.textAlign="right";
  for(let i=0;i<5;i++){
    const val=Math.round(max*(4-i)/4);
    const y=pad.t+(H-pad.t-pad.b)*i/4+3;
    cx.fillText(val,pad.l-4,y);
  }
}

// Degree histogram
barChart("c-deg",D.deg_hist,"#00ff88",D.deg_hist.map((_,i)=>i));

// Distance histogram
barChart("c-dist",D.dist_hist,"#4488ff",D.dist_hist.map((_,i)=>Math.round(i*D.neurons/20)));

// Adjacency matrix heatmap
(function(){
  const cv=document.getElementById("c-adj");
  const cx=cv.getContext("2d");
  const W=cv.offsetWidth,H2=cv.offsetHeight;
  cv.width=W*2;cv.height=H2*2;cx.scale(2,2);
  const n=D.adj_buckets;
  const pad=30;
  const cellW=(W-pad*2)/n, cellH=(H2-pad*2)/n;
  // Labels
  cx.fillStyle="#3a4858";cx.font="7px JetBrains Mono";cx.textAlign="center";
  for(let i=0;i<n;i+=4){cx.fillText(i*Math.floor(D.neurons/n),pad+i*cellW+cellW/2,H2-pad+12)}
  cx.textAlign="right";
  for(let i=0;i<n;i+=4){cx.fillText(i*Math.floor(D.neurons/n),pad-4,pad+i*cellH+cellH/2+3)}
  // Cells
  D.adj_matrix.forEach((row,i)=>{
    row.forEach((v,j)=>{
      if(v<0.01)return;
      const g=Math.floor(v*255);
      cx.fillStyle=`rgb(0,${g},${Math.floor(g/2)})`;
      cx.fillRect(pad+j*cellW,pad+i*cellH,cellW-0.5,cellH-0.5);
    });
  });
  // Border
  cx.strokeStyle="rgba(0,255,136,0.1)";cx.strokeRect(pad,pad,(W-pad*2),(H2-pad*2));
})();

// Cluster sizes
barChart("c-clust",D.cluster_sizes,COLORS,D.cluster_sizes.map((_,i)=>"C"+i));

// Signal propagation
let activeTick=0;
function setTick(t){
  activeTick=t;
  document.querySelectorAll(".tick-label").forEach((el,i)=>{
    if(t===-1&&i===6)el.classList.add("active");
    else if(i===t)el.classList.add("active");
    else el.classList.remove("active");
  });
  drawTicks();
}
function drawTicks(){
  const cv=document.getElementById("c-ticks");
  const cx=cv.getContext("2d");
  const W=cv.offsetWidth,H2=cv.offsetHeight;
  cv.width=W*2;cv.height=H2*2;cx.scale(2,2);
  const pad={t:10,b:20,l:40,r:10};
  // Find max activation
  let maxAct=0;
  D.ticks_data.forEach(td=>td.forEach(([_,v])=>{if(v>maxAct)maxAct=v}));
  if(maxAct===0)maxAct=1;
  // Grid
  cx.strokeStyle="rgba(0,255,136,0.04)";cx.lineWidth=0.5;
  for(let i=0;i<5;i++){const y=pad.t+(H2-pad.t-pad.b)*i/4;cx.beginPath();cx.moveTo(pad.l,y);cx.lineTo(W-pad.r,y);cx.stroke()}
  // Y labels
  cx.fillStyle="#3a4858";cx.font="8px JetBrains Mono";cx.textAlign="right";
  for(let i=0;i<5;i++){cx.fillText((maxAct*(4-i)/4).toFixed(1),pad.l-4,pad.t+(H2-pad.t-pad.b)*i/4+3)}

  const ticks=activeTick===-1?[0,1,2,3,4,5]:[activeTick];
  ticks.forEach(t=>{
    const td=D.ticks_data[t];
    if(!td)return;
    cx.fillStyle=TICK_COLORS[t];
    cx.globalAlpha=activeTick===-1?0.5:0.8;
    td.forEach(([nid,val])=>{
      const x=pad.l+(nid/D.neurons)*(W-pad.l-pad.r);
      const h=(val/maxAct)*(H2-pad.t-pad.b);
      cx.fillRect(x-1,H2-pad.b-h,3,h);
    });
  });
  cx.globalAlpha=1;
  // X label
  cx.fillStyle="#3a4858";cx.font="8px JetBrains Mono";cx.textAlign="center";
  cx.fillText("neuron ID",W/2,H2-2);
}
drawTicks();

// Network graph (cluster colored, force-ish layout)
(function(){
  const cv=document.getElementById("c-graph");
  const cx=cv.getContext("2d");
  const W=cv.offsetWidth,H2=cv.offsetHeight;
  cv.width=W*2;cv.height=H2*2;cx.scale(2,2);

  const N=D.nodes,L=D.links;
  const im={};N.forEach((n,i)=>{im[n.id]=i});
  const md=Math.max(...N.map(n=>n.deg));

  // Circle layout grouped by cluster
  const clusterOrder=[...N].sort((a,b)=>a.cluster-b.cluster||b.deg-a.deg);
  const cx2=W/2,cy=H2/2,r=Math.min(W,H2)*0.42;
  clusterOrder.forEach((n,i)=>{
    const a=(i/N.length)*Math.PI*2-Math.PI/2;
    n.x=cx2+Math.cos(a)*r;
    n.y=cy+Math.sin(a)*r;
  });

  // Edges
  cx.lineWidth=0.3;cx.globalAlpha=0.04;
  L.forEach(l=>{
    const si=im[l.s],ti=im[l.t];
    if(si==null||ti==null)return;
    const a=N[si],b=N[ti];
    cx.strokeStyle=l.v>0?"#00ff88":"#ff3355";
    cx.beginPath();cx.moveTo(a.x,a.y);cx.lineTo(b.x,b.y);cx.stroke();
  });

  // Nodes
  cx.globalAlpha=1;
  N.forEach(n=>{
    const sz=1.5+(n.deg/md)*3.5;
    const c=n.cluster>=0&&n.cluster<COLORS.length?COLORS[n.cluster]:"#555";
    cx.fillStyle=c;
    cx.beginPath();cx.arc(n.x,n.y,sz,0,Math.PI*2);cx.fill();
  });

  // Legend
  const leg=document.getElementById("graph-legend");
  for(let i=0;i<Math.min(D.n_clusters,12);i++){
    const s=document.createElement("span");
    s.innerHTML=`<span class="dot" style="background:${COLORS[i]}"></span>cluster ${i} (${D.cluster_sizes[i]})`;
    leg.appendChild(s);
  }
})();
</script>
</body>
</html>"""

html = html.replace('__DATA__', compact)

with open('s:/AI/work/VRAXION_DEV/v4.2/network_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Saved network_dashboard.html ({len(html)} bytes)")
