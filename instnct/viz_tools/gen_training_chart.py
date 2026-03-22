"""Generate live training dashboard that auto-refreshes from log data."""
import json, re, glob, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "training_chart_data.json"
DEFAULT_OUTPUT = ROOT / "training_live.html"

# Parse latest output
def parse_output():
    # Find the latest task output
    pattern = "S:/tmp/claude/S--AI/*/tasks/bilmoay45.output"
    files = glob.glob(pattern)
    if not files:
        # Fallback: parse from training_chart_data.json
        with DEFAULT_DATA.open() as f:
            return json.load(f)

    data = []
    for f in files:
        for l in open(f):
            m = re.search(r'\[\s*(\d+)\] eval=(\d+\.\d+)% edges=(\d+) \[A=(\d+)\|T=(\d+)\|D=(\d+)\] theta=(\d+\.\d+)\+/-(\d+\.\d+) decay=(\d+\.\d+)\+/-(\d+\.\d+) (\d+)s', l)
            if m:
                data.append({
                    'step': int(m.group(1)), 'eval': float(m.group(2)),
                    'edges': int(m.group(3)), 'A': int(m.group(4)),
                    'T': int(m.group(5)), 'D': int(m.group(6)),
                    'theta_m': float(m.group(7)), 'theta_s': float(m.group(8)),
                    'decay_m': float(m.group(9)), 'decay_s': float(m.group(10)),
                    'time': int(m.group(11))
                })
    return data

data = parse_output()
compact = json.dumps(data, separators=(',',':'))

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SWG Training Dashboard</title>
<style>
@import url("https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Outfit:wght@200;400;700&display=swap");
:root{--bg:#060810;--card:rgba(10,16,28,0.95);--border:rgba(0,255,136,0.12);--accent:#00ff88;--neg:#ff3355;--text:#b8c4d0;--dim:#3a4858;--blue:#4488ff;--orange:#ff8844;--purple:#aa66ff;--yellow:#ffcc00}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:"JetBrains Mono",monospace;overflow-y:auto}
.header{padding:20px 28px 10px;border-bottom:1px solid var(--border)}
.header h1{font-family:"Outfit",sans-serif;font-weight:200;font-size:24px;color:var(--accent);letter-spacing:5px;text-transform:uppercase;text-shadow:0 0 20px rgba(0,255,136,0.2)}
.header .sub{font-size:10px;color:var(--dim);letter-spacing:2px;margin-top:2px}
.stats{display:flex;gap:20px;padding:12px 28px;border-bottom:1px solid var(--border);flex-wrap:wrap}
.stat .num{font-family:"Outfit";font-weight:700;font-size:20px;color:var(--accent)}
.stat .num.b{color:var(--blue)} .stat .num.o{color:var(--orange)} .stat .num.p{color:var(--purple)}
.stat .lbl{font-size:8px;color:var(--dim);letter-spacing:1px;text-transform:uppercase;margin-top:1px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:14px 28px}
@media(max-width:1100px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--border);border-radius:3px;padding:14px}
.card h2{font-family:"Outfit";font-weight:300;font-size:13px;color:var(--accent);letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;padding-bottom:4px;border-bottom:1px solid rgba(0,255,136,0.06)}
canvas{display:block;width:100%}
.full{grid-column:1/-1}
.legend{display:flex;gap:12px;margin-top:8px;font-size:9px;flex-wrap:wrap}
.legend span{display:flex;align-items:center;gap:4px}
.legend .dot{width:8px;height:8px;border-radius:50%;display:inline-block}
.desc{font-size:9px;color:var(--dim);margin-top:6px;line-height:1.5}
.save-btn{background:transparent;color:var(--accent);border:1px solid var(--border);padding:4px 10px;font-family:"JetBrains Mono",monospace;font-size:9px;cursor:pointer;margin-top:6px;letter-spacing:1px}
.save-btn:hover{background:rgba(0,255,136,0.1);border-color:var(--accent)}
#save-all{position:fixed;bottom:16px;right:16px;z-index:10;background:rgba(0,255,136,0.15);color:#fff;border:2px solid var(--accent);padding:10px 20px;font-family:"JetBrains Mono",monospace;font-size:11px;cursor:pointer;letter-spacing:2px;text-transform:uppercase}
#save-all:hover{background:rgba(0,255,136,0.3)}
#status{position:fixed;bottom:60px;right:16px;z-index:10;font-size:10px;color:var(--accent);display:none}
</style>
</head>
<body>
<div class="header">
<h1>SWG Training Live</h1>
<div class="sub">768 neurons · per-neuron theta + decay · round-robin [A|T|D] · english next-byte</div>
</div>
<div class="stats" id="stats"></div>
<div class="grid">
<div class="card">
  <h2>Eval Accuracy (%)</h2><canvas id="c1" height="200"></canvas>
  <div class="legend"><span><span class="dot" style="background:#00ff88"></span> eval accuracy</span></div>
  <div class="desc">Next-byte prediction on held-out English text. Random baseline = 0.39%. Higher = better language understanding.</div>
</div>
<div class="card">
  <h2>Edges (synapses)</h2><canvas id="c2" height="200"></canvas>
  <div class="legend"><span><span class="dot" style="background:#4488ff"></span> active edges</span></div>
  <div class="desc">Total connections in the network. Grows as add mutations are accepted. More edges = more capacity but slower forward pass.</div>
</div>
<div class="card">
  <h2>Accept Counts [A|T|D]</h2><canvas id="c3" height="200"></canvas>
  <div class="legend">
    <span><span class="dot" style="background:#4488ff"></span> Add (edges)</span>
    <span><span class="dot" style="background:#ff8844"></span> Theta (thresholds)</span>
    <span><span class="dot" style="background:#aa66ff"></span> Decay (retention)</span>
  </div>
  <div class="desc">Cumulative accepted mutations per type. Round-robin: every 3rd step = theta, every 6th = decay, rest = add.</div>
</div>
<div class="card">
  <h2>Accept Rate per Type (%)</h2><canvas id="c4" height="200"></canvas>
  <div class="legend">
    <span><span class="dot" style="background:#4488ff"></span> Add rate</span>
    <span><span class="dot" style="background:#ff8844"></span> Theta rate</span>
    <span><span class="dot" style="background:#aa66ff"></span> Decay rate</span>
  </div>
  <div class="desc">What % of attempts succeed per type. Higher = that mutation type is still finding improvements. When it drops = that parameter is saturating.</div>
</div>
<div class="card">
  <h2>Theta — Per-Neuron Firing Threshold</h2><canvas id="c5" height="200"></canvas>
  <div class="legend">
    <span><span class="dot" style="background:#ff8844"></span> mean</span>
    <span style="color:#555">band = mean ± std</span>
  </div>
  <div class="desc">Each neuron's firing threshold (0-1). Low theta = fires easily (sensitive). High theta = needs strong signal (selective). Wider band = neurons are differentiating.</div>
</div>
<div class="card">
  <h2>Decay — Per-Neuron Charge Retention</h2><canvas id="c6" height="200"></canvas>
  <div class="legend">
    <span><span class="dot" style="background:#aa66ff"></span> mean</span>
    <span style="color:#555">band = mean ± std</span>
  </div>
  <div class="desc">How fast each neuron forgets (0.01=remembers almost everything, 0.5=forgets half per tick). Low decay = long memory neuron. High decay = fast response neuron.</div>
</div>
</div>
<button id="save-all" onclick="saveAllCharts()">SAVE ALL CHARTS AS PNG</button>
<div id="status"></div>
<script>
const D=__DATA__;
const last=D[D.length-1];

// Stats bar
const st=document.getElementById("stats");
const stats=[
["eval",last.eval+"%",""],["edges",last.edges,""],
["add",last.A,"b"],["theta",last.T,"o"],["decay",last.D,"p"],
["theta μ",last.theta_m.toFixed(3),"o"],["decay μ",last.decay_m.toFixed(3),"p"],
["step",last.step,""],["time",Math.floor(last.time/60)+"m",""]
];
stats.forEach(([l,v,c])=>{
const d=document.createElement("div");d.className="stat";
d.innerHTML=`<div class="num ${c}">${v}</div><div class="lbl">${l}</div>`;
st.appendChild(d);
});

function chart(id,series,colors,fill){
const cv=document.getElementById(id);
const cx=cv.getContext("2d");
const W=cv.offsetWidth,H=cv.offsetHeight;
cv.width=W*2;cv.height=H*2;cx.scale(2,2);
const pad={t:10,b:20,l:40,r:10};
const pw=W-pad.l-pad.r,ph=H-pad.t-pad.b;
// Find ranges
let ymin=Infinity,ymax=-Infinity;
series.forEach(s=>s.forEach(v=>{if(v<ymin)ymin=v;if(v>ymax)ymax=v;}));
if(ymin===ymax){ymin-=1;ymax+=1;}
const margin=(ymax-ymin)*0.05;ymin-=margin;ymax+=margin;
// Grid
cx.strokeStyle="rgba(0,255,136,0.04)";cx.lineWidth=0.5;
for(let i=0;i<5;i++){const y=pad.t+ph*i/4;cx.beginPath();cx.moveTo(pad.l,y);cx.lineTo(W-pad.r,y);cx.stroke();}
// Y labels
cx.fillStyle="#3a4858";cx.font="8px JetBrains Mono";cx.textAlign="right";
for(let i=0;i<5;i++){
const val=ymax-(ymax-ymin)*i/4;
cx.fillText(val.toFixed(val>10?0:val>1?1:3),pad.l-4,pad.t+ph*i/4+3);
}
// X labels
cx.textAlign="center";
const steps=D.map(d=>d.step);
for(let i=0;i<steps.length;i+=Math.ceil(steps.length/6)){
const x=pad.l+pw*i/(steps.length-1||1);
cx.fillText(steps[i],x,H-4);
}
// Lines
series.forEach((s,si)=>{
cx.strokeStyle=colors[si];cx.lineWidth=1.5;cx.globalAlpha=0.9;
cx.beginPath();
s.forEach((v,i)=>{
const x=pad.l+pw*i/(s.length-1||1);
const y=pad.t+ph*(1-(v-ymin)/(ymax-ymin));
if(i===0)cx.moveTo(x,y);else cx.lineTo(x,y);
});
cx.stroke();
// Fill area
if(fill&&fill[si]){
cx.globalAlpha=0.08;cx.lineTo(pad.l+pw,pad.t+ph);cx.lineTo(pad.l,pad.t+ph);cx.closePath();cx.fillStyle=colors[si];cx.fill();
}
cx.globalAlpha=1;
// Dots on last point
const lx=pad.l+pw;const ly=pad.t+ph*(1-(s[s.length-1]-ymin)/(ymax-ymin));
cx.fillStyle=colors[si];cx.beginPath();cx.arc(lx,ly,3,0,Math.PI*2);cx.fill();
});
}

// 1. Eval
chart("c1",[D.map(d=>d.eval)],["#00ff88"],[true]);
// 2. Edges
chart("c2",[D.map(d=>d.edges)],["#4488ff"],[true]);
// 3. Accept counts
chart("c3",[D.map(d=>d.A),D.map(d=>d.T),D.map(d=>d.D)],["#4488ff","#ff8844","#aa66ff"]);
// 4. Accept rate
const aRate=D.map((d,i)=>{
const addSteps=Math.floor((d.step)*4/6); // 4/6 of steps are add
const tSteps=Math.floor((d.step)/6);
const dSteps=Math.floor((d.step)/6);
return [addSteps>0?d.A/addSteps*100:0, tSteps>0?d.T/tSteps*100:0, dSteps>0?d.D/dSteps*100:0];
});
chart("c4",[aRate.map(r=>r[0]),aRate.map(r=>r[1]),aRate.map(r=>r[2])],["#4488ff","#ff8844","#aa66ff"]);
// 5. Theta (mean ± std as band)
chart("c5",[D.map(d=>d.theta_m),D.map(d=>d.theta_m+d.theta_s),D.map(d=>Math.max(0,d.theta_m-d.theta_s))],["#ff8844","rgba(255,136,68,0.3)","rgba(255,136,68,0.3)"]);
// 6. Decay
chart("c6",[D.map(d=>d.decay_m),D.map(d=>d.decay_m+d.decay_s),D.map(d=>Math.max(0,d.decay_m-d.decay_s))],["#aa66ff","rgba(170,102,255,0.3)","rgba(170,102,255,0.3)"]);

// Save chart as PNG
function saveCanvas(canvasId, filename) {
  const cv = document.getElementById(canvasId);
  const link = document.createElement('a');
  link.download = filename;
  link.href = cv.toDataURL('image/png');
  link.click();
}

function saveAllCharts() {
  const status = document.getElementById('status');
  status.style.display = 'block';
  status.textContent = 'Saving...';
  const charts = [
    ['c1', 'swg_eval_accuracy.png'],
    ['c2', 'swg_edges.png'],
    ['c3', 'swg_accept_counts.png'],
    ['c4', 'swg_accept_rate.png'],
    ['c5', 'swg_theta.png'],
    ['c6', 'swg_decay.png'],
  ];
  let i = 0;
  function next() {
    if (i >= charts.length) {
      status.textContent = 'All 6 charts saved!';
      setTimeout(() => status.style.display = 'none', 3000);
      return;
    }
    const [id, name] = charts[i];
    status.textContent = 'Saving ' + name + '...';
    saveCanvas(id, name);
    i++;
    setTimeout(next, 500);  // small delay between downloads
  }
  next();
}
</script>
</body>
</html>"""

html = html.replace('__DATA__', compact)

with DEFAULT_OUTPUT.open('w', encoding='utf-8') as f:
    f.write(html)
print(f"Saved training_live.html ({len(data)} data points, last step={data[-1]['step']})")
