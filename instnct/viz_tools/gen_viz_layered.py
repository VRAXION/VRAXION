"""Generate layered network visualization — input left, output right, signal flow visible."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph
import random

# Load network
IO = 256
random.seed(42); np.random.seed(42)
net = SelfWiringGraph(IO)
d = np.load(ROOT / 'checkpoints' / 'english_768n_step3000.npz')
net.mask[:] = 0; net.mask[d['rows'], d['cols']] = d['vals']
net.resync_alive()

H = net.H
edges = list(net.alive)
print(f"{H} neurons, {len(edges)} edges")

# Compute topological depth via BFS from "input-proximal" neurons
# Neurons that receive most energy from input_projection are "close to input"
input_projection_energy = np.abs(net.input_projection).sum(axis=0)  # (H,) how much input each hidden neuron gets
output_projection_energy = np.abs(net.output_projection).sum(axis=1)  # (H,) how much each neuron contributes to output

# BFS depth from top input neurons
depth = np.full(H, -1, dtype=int)
# Start from neurons with highest input_projection energy
input_proximal = np.argsort(input_projection_energy)[-100:]  # top 100
for n in input_proximal:
    depth[n] = 0

# Build adjacency
adj = [[] for _ in range(H)]
for r, c in edges:
    adj[r].append(c)

# BFS
queue = list(input_proximal)
while queue:
    node = queue.pop(0)
    for nb in adj[node]:
        if depth[nb] < 0:
            depth[nb] = depth[node] + 1
            queue.append(nb)

# Neurons not reached = isolated from input (shouldn't happen)
max_depth = max(d for d in depth if d >= 0) if any(d >= 0 for d in depth) else 0
for i in range(H):
    if depth[i] < 0:
        depth[i] = max_depth + 1

print(f"Max depth: {max_depth}")
print(f"Depth distribution: {np.bincount(depth[:max_depth+2])}")

# Assign layers: compress to ~8 columns
n_layers = min(8, max_depth + 1)
layer = np.clip(depth * n_layers // (max_depth + 1), 0, n_layers - 1)

# Count per layer
for l in range(n_layers):
    count = (layer == l).sum()
    print(f"  Layer {l}: {count} neurons")

# Build viz data
# Position: x = layer (left to right), y = within-layer index
nodes_viz = []
layer_counts = [0] * n_layers
layer_totals = [(layer == l).sum() for l in range(n_layers)]

out_deg = np.array([(net.mask[i, :] != 0).sum() for i in range(H)])
in_deg = np.array([(net.mask[:, j] != 0).sum() for j in range(H)])
total_deg = out_deg + in_deg

for i in range(H):
    if total_deg[i] == 0:
        continue
    l = int(layer[i])
    y_idx = layer_counts[l]
    layer_counts[l] += 1
    nodes_viz.append({
        'id': int(i),
        'layer': l,
        'y_idx': y_idx,
        'y_total': int(layer_totals[l]),
        'deg': int(total_deg[i]),
        'out': int(out_deg[i]),
        'inp': int(in_deg[i]),
        'input_projection': round(float(input_projection_energy[i]), 2),
        'output_projection': round(float(output_projection_energy[i]), 2),
        'depth': int(depth[i]),
    })

links_viz = []
for r, c in edges:
    if total_deg[r] == 0 or total_deg[c] == 0:
        continue
    links_viz.append({
        's': int(r),
        't': int(c),
        'v': float(net.mask[r, c]),
        'sl': int(layer[r]),
        'tl': int(layer[c]),
    })

# Classify links
forward = sum(1 for l in links_viz if l['tl'] > l['sl'])
backward = sum(1 for l in links_viz if l['tl'] < l['sl'])
lateral = sum(1 for l in links_viz if l['tl'] == l['sl'])
print(f"Links: {forward} forward, {backward} backward, {lateral} lateral")

data = {'nodes': nodes_viz, 'links': links_viz, 'n_layers': int(n_layers), 'max_depth': int(max_depth)}

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return super().default(obj)

compact = json.dumps(data, separators=(',', ':'), cls=NpEncoder)

# Build HTML
html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SWG Signal Flow</title>
<style>
@import url("https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Outfit:wght@200;400;700&display=swap");
:root{--bg:#04060a;--panel:rgba(8,14,22,0.92);--border:rgba(0,255,136,0.15);--accent:#00ff88;--neg:#ff3355;--text:#b8c4d0;--dim:#445566}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:"JetBrains Mono",monospace;overflow:hidden;cursor:crosshair}
canvas{display:block}
.panel{position:fixed;background:var(--panel);border:1px solid var(--border);backdrop-filter:blur(12px);padding:14px;z-index:10;border-radius:2px}
#header{top:12px;left:12px;min-width:250px}
#header h1{font-family:"Outfit",sans-serif;font-weight:200;font-size:20px;color:var(--accent);letter-spacing:3px;text-transform:uppercase;text-shadow:0 0 20px rgba(0,255,136,0.15)}
#header .sub{font-size:9px;color:var(--dim);letter-spacing:2px;margin:4px 0 10px}
.sr{display:flex;justify-content:space-between;padding:2px 0;font-size:10px;border-bottom:1px solid rgba(255,255,255,0.03)}
.sr .l{color:var(--dim)} .sr .v{font-weight:600} .sr .v.p{color:var(--accent)} .sr .v.n{color:var(--neg)} .sr .v.f{color:#4488ff} .sr .v.b{color:#ff8844}
#hover{bottom:12px;left:12px;min-width:220px;font-size:10px;transition:opacity 0.15s}
#hover.empty{opacity:0.3}
#hover .nid{font-family:"Outfit";font-size:16px;color:var(--accent);font-weight:700}
.bar{height:3px;border-radius:1px;margin:3px 0;transition:width 0.2s}
.bar.ob{background:var(--accent)} .bar.ib{background:var(--neg)}
#controls{top:12px;right:12px;display:flex;flex-direction:column;gap:5px}
.btn{background:transparent;color:var(--accent);border:1px solid var(--border);padding:5px 12px;font-family:"JetBrains Mono",monospace;font-size:9px;letter-spacing:1px;cursor:pointer;transition:all 0.15s;text-transform:uppercase}
.btn:hover{background:rgba(0,255,136,0.1);border-color:var(--accent)}
.btn.active{background:rgba(0,255,136,0.15);border-color:var(--accent);color:#fff}
select.btn{appearance:none;-webkit-appearance:none}
.scanline{position:fixed;top:0;left:0;right:0;height:100vh;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,255,136,0.006) 2px,rgba(0,255,136,0.006) 4px);pointer-events:none;z-index:100}
.labels{position:fixed;bottom:12px;right:12px;z-index:10;font-size:9px;color:var(--dim)}
</style>
</head>
<body>
<div class="scanline"></div>
<div class="panel" id="header">
<h1>SWG Signal Flow</h1>
<div class="sub">input → hidden layers → output · 768 neurons</div>
<div class="sr"><span class="l">neurons</span><span class="v" id="sn">-</span></div>
<div class="sr"><span class="l">synapses</span><span class="v" id="se">-</span></div>
<div class="sr"><span class="l">forward →</span><span class="v f" id="sf">-</span></div>
<div class="sr"><span class="l">backward ←</span><span class="v b" id="sb">-</span></div>
<div class="sr"><span class="l">lateral ↔</span><span class="v" id="sl2">-</span></div>
<div class="sr"><span class="l">layers</span><span class="v" id="sly">-</span></div>
</div>
<div class="panel empty" id="hover">
<div class="nid" id="hi">hover a neuron</div>
<div class="sr"><span class="l">layer</span><span class="v" id="hly">-</span></div>
<div class="sr"><span class="l">degree</span><span class="v" id="hd">-</span></div>
<div class="sr"><span class="l">outgoing</span><span class="v p" id="ho">-</span></div>
<div class="bar ob" id="hob" style="width:0%"></div>
<div class="sr"><span class="l">incoming</span><span class="v n" id="hin">-</span></div>
<div class="bar ib" id="hib" style="width:0%"></div>
<div class="sr"><span class="l">input_projection energy</span><span class="v" id="hwi">-</span></div>
<div class="sr"><span class="l">output_projection energy</span><span class="v" id="hwo">-</span></div>
</div>
<div class="panel" id="controls">
<button class="btn active" id="be" onclick="togE()">edges on</button>
<select class="btn" id="ef" onchange="nr=true">
<option value="all">all edges</option>
<option value="fwd">forward only</option>
<option value="back">backward only</option>
<option value="lat">lateral only</option>
<option value="pos">excitatory only</option>
<option value="neg">inhibitory only</option>
</select>
<button class="btn" onclick="rv()">reset view</button>
</div>
<div class="labels">
<span style="color:#4488ff">→ forward</span> &nbsp;
<span style="color:#ff8844">← backward</span> &nbsp;
<span style="color:#888">↔ lateral</span>
</div>
<canvas id="c"></canvas>
<script>
const D=__DATA__;
const cv=document.getElementById("c"),cx=cv.getContext("2d");
let W,HH;function rs(){W=cv.width=innerWidth;HH=cv.height=innerHeight}
rs();

const N=D.nodes,L=D.links,NL=D.n_layers;
const im={};N.forEach((n,i)=>{im[n.id]=i});
L.forEach(l=>{l.si=im[l.s];l.ti=im[l.t]});
const md=Math.max(...N.map(n=>n.deg));
const fwd=L.filter(l=>l.tl>l.sl).length;
const bck=L.filter(l=>l.tl<l.sl).length;
const lat=L.filter(l=>l.tl===l.sl).length;

document.getElementById("sn").textContent=N.length;
document.getElementById("se").textContent=L.length;
document.getElementById("sf").textContent=fwd;
document.getElementById("sb").textContent=bck;
document.getElementById("sl2").textContent=lat;
document.getElementById("sly").textContent=NL;

// Position nodes: x by layer, y by index within layer
function layout(){
  const padX=120,padY=30;
  const usableW=W-padX*2, usableH=HH-padY*2;
  const colW=usableW/(NL-1||1);
  N.forEach(n=>{
    n.x=padX+n.layer*colW;
    const gap=usableH/(n.y_total||1);
    n.y=padY+n.y_idx*gap+gap/2;
  });
}
layout();
window.onresize=()=>{rs();layout();nr=true};

let se=true,zm=1,px=0,py=0,dg=false,ddx,ddy,hv=-1,nr=true;
function togE(){se=!se;const b=document.getElementById("be");b.textContent=se?"edges on":"edges off";b.classList.toggle("active",se);nr=true}
function rv(){zm=1;px=0;py=0;nr=true}

function render(){
  cx.clearRect(0,0,W,HH);

  // Layer columns
  cx.save();
  cx.globalAlpha=0.04;
  const padX=120,colW=(W-padX*2)/(NL-1||1);
  for(let i=0;i<NL;i++){
    const x=padX+i*colW;
    cx.strokeStyle="#00ff88";
    cx.beginPath();cx.moveTo(x,0);cx.lineTo(x,HH);cx.stroke();
  }
  cx.restore();

  // Layer labels
  cx.save();cx.globalAlpha=0.15;cx.fillStyle="#00ff88";cx.font="9px JetBrains Mono";cx.textAlign="center";
  for(let i=0;i<NL;i++){
    const x=padX+i*colW;
    const label=i===0?"input":i===NL-1?"output":"L"+i;
    cx.fillText(label,x,15);
  }
  cx.restore();

  cx.save();cx.translate(px,py);cx.scale(zm,zm);
  const fi=document.getElementById("ef").value;

  // Edges
  if(se){
    cx.lineWidth=0.4;
    L.forEach(l=>{
      if(l.si==null||l.ti==null)return;
      const isFwd=l.tl>l.sl, isBck=l.tl<l.sl, isLat=l.tl===l.sl;
      if(fi==="fwd"&&!isFwd)return;
      if(fi==="back"&&!isBck)return;
      if(fi==="lat"&&!isLat)return;
      if(fi==="pos"&&l.v<0)return;
      if(fi==="neg"&&l.v>0)return;

      const a=N[l.si],b=N[l.ti];
      const isHovered=(hv>=0&&(N[l.si].id===hv||N[l.ti].id===hv));
      if(!isHovered){cx.globalAlpha=0.04}else{cx.globalAlpha=0.7;cx.lineWidth=1.5}

      if(isFwd) cx.strokeStyle=l.v>0?"#4488ff":"#6644cc";
      else if(isBck) cx.strokeStyle=l.v>0?"#ff8844":"#cc4466";
      else cx.strokeStyle=l.v>0?"#00ff88":"#ff3355";

      // Curved lines for non-adjacent layers
      const dx=b.x-a.x, dy=b.y-a.y;
      cx.beginPath();cx.moveTo(a.x,a.y);
      if(Math.abs(l.tl-l.sl)<=1){
        cx.lineTo(b.x,b.y);
      } else {
        const cpx=(a.x+b.x)/2, cpy=(a.y+b.y)/2+dy*0.3;
        cx.quadraticCurveTo(cpx,cpy,b.x,b.y);
      }
      cx.stroke();
      cx.lineWidth=0.4;
    });
  }

  // Nodes
  cx.globalAlpha=1;
  N.forEach(n=>{
    const r=1+(n.deg/md)*4;
    const br=0.2+0.8*(n.deg/md);
    if(n.id===hv){
      cx.shadowBlur=15;cx.shadowColor="#00ff88";
      cx.fillStyle="#fff";cx.beginPath();cx.arc(n.x,n.y,r+2,0,Math.PI*2);cx.fill();
      cx.shadowBlur=0;
    }
    // Color by layer
    const lf=n.layer/(NL-1);
    const rr=Math.floor(30+lf*40);
    const gg=Math.floor(br*200+lf*55);
    const bb=Math.floor(100+lf*155);
    cx.fillStyle="rgb("+rr+","+gg+","+bb+")";
    cx.beginPath();cx.arc(n.x,n.y,r,0,Math.PI*2);cx.fill();
  });
  cx.restore();
}

cv.addEventListener("wheel",e=>{e.preventDefault();zm*=e.deltaY>0?0.92:1.08;zm=Math.max(0.05,Math.min(15,zm));nr=true});
cv.addEventListener("mousedown",e=>{dg=true;ddx=e.clientX-px;ddy=e.clientY-py});
cv.addEventListener("mouseup",()=>dg=false);
cv.addEventListener("mousemove",e=>{
  if(dg){px=e.clientX-ddx;py=e.clientY-ddy;nr=true;return}
  const mx=(e.clientX-px)/zm,my=(e.clientY-py)/zm;
  let best=-1,bd=12/zm;
  N.forEach(n=>{const dd=Math.sqrt((n.x-mx)**2+(n.y-my)**2);if(dd<bd){bd=dd;best=n.id}});
  if(best!==hv){hv=best;uh();nr=true}
});
function uh(){
  const el=document.getElementById("hover");
  if(hv<0){el.classList.add("empty");document.getElementById("hi").textContent="hover a neuron";return}
  el.classList.remove("empty");
  const n=N.find(n=>n.id===hv);
  document.getElementById("hi").textContent="neuron "+n.id;
  document.getElementById("hly").textContent="L"+n.layer+" (depth "+n.depth+")";
  document.getElementById("hd").textContent=n.deg;
  document.getElementById("ho").textContent=n.out;
  document.getElementById("hin").textContent=n.inp;
  document.getElementById("hob").style.width=(n.out/md*100)+"%";
  document.getElementById("hib").style.width=(n.inp/md*100)+"%";
  document.getElementById("hwi").textContent=n.input_projection.toFixed(1);
  document.getElementById("hwo").textContent=n.output_projection.toFixed(1);
}
function loop(){if(nr){render();nr=false}requestAnimationFrame(loop)}
loop();
</script>
</body>
</html>"""

html = html.replace('__DATA__', compact)

with (ROOT / 'network_viz_v2.html').open('w', encoding='utf-8') as f:
    f.write(html)

print(f"Saved network_viz_v2.html ({len(html)} bytes)")
