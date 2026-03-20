"""Generate self-contained network visualization HTML."""
import json

with open('s:/AI/work/VRAXION_DEV/v4.2/network_viz_data.json') as f:
    data = json.load(f)

compact = json.dumps(data, separators=(',',':'))

html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SWG Neural Topology</title>
<style>
@import url("https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Outfit:wght@200;400;700&display=swap");
:root{--bg:#04060a;--panel:rgba(8,14,22,0.92);--border:rgba(0,255,136,0.15);--accent:#00ff88;--neg:#ff3355;--text:#b8c4d0;--dim:#445566;}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:"JetBrains Mono",monospace;overflow:hidden;cursor:crosshair}
canvas{display:block}
.panel{position:fixed;background:var(--panel);border:1px solid var(--border);backdrop-filter:blur(12px);padding:16px;z-index:10;border-radius:2px}
#header{top:16px;left:16px;min-width:260px}
#header h1{font-family:"Outfit",sans-serif;font-weight:200;font-size:22px;color:var(--accent);letter-spacing:4px;text-transform:uppercase;margin-bottom:4px;text-shadow:0 0 20px rgba(0,255,136,0.15)}
#header .sub{font-size:10px;color:var(--dim);letter-spacing:2px;margin-bottom:12px}
.sr{display:flex;justify-content:space-between;padding:3px 0;font-size:11px;border-bottom:1px solid rgba(255,255,255,0.03)}
.sr .l{color:var(--dim)} .sr .v{font-weight:600} .sr .v.p{color:var(--accent)} .sr .v.n{color:var(--neg)}
#hover{bottom:16px;left:16px;min-width:220px;font-size:11px;transition:opacity 0.2s}
#hover.empty{opacity:0.3}
#hover .nid{font-family:"Outfit";font-size:18px;color:var(--accent);font-weight:700}
.bar{height:3px;border-radius:2px;margin:4px 0;transition:width 0.3s}
.bar.ob{background:var(--accent)} .bar.ib{background:var(--neg)}
#controls{top:16px;right:16px;display:flex;flex-direction:column;gap:6px}
.btn{background:transparent;color:var(--accent);border:1px solid var(--border);padding:6px 14px;font-family:"JetBrains Mono",monospace;font-size:10px;letter-spacing:1px;cursor:pointer;transition:all 0.2s;text-transform:uppercase}
.btn:hover{background:rgba(0,255,136,0.1);border-color:var(--accent)}
.btn.active{background:rgba(0,255,136,0.15);border-color:var(--accent);color:#fff}
select.btn{appearance:none;-webkit-appearance:none}
.scanline{position:fixed;top:0;left:0;right:0;height:100vh;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,255,136,0.008) 2px,rgba(0,255,136,0.008) 4px);pointer-events:none;z-index:100}
.vig{position:fixed;top:0;left:0;right:0;bottom:0;background:radial-gradient(ellipse at center,transparent 50%,rgba(0,0,0,0.6) 100%);pointer-events:none;z-index:99}
</style>
</head>
<body>
<div class="scanline"></div>
<div class="vig"></div>
<div class="panel" id="header">
<h1>SWG Neural Topology</h1>
<div class="sub">768 neurons · self-wiring graph · english next-byte</div>
<div class="sr"><span class="l">neurons</span><span class="v" id="sn">-</span></div>
<div class="sr"><span class="l">synapses</span><span class="v" id="se">-</span></div>
<div class="sr"><span class="l">excitatory (+)</span><span class="v p" id="sp">-</span></div>
<div class="sr"><span class="l">inhibitory (−)</span><span class="v n" id="sng">-</span></div>
<div class="sr"><span class="l">max degree</span><span class="v" id="sm">-</span></div>
<div class="sr"><span class="l">avg degree</span><span class="v" id="sa">-</span></div>
<div class="sr"><span class="l">density</span><span class="v" id="sd">-</span></div>
</div>
<div class="panel empty" id="hover">
<div class="nid" id="hi">hover a neuron</div>
<div class="sr"><span class="l">degree</span><span class="v" id="hd">-</span></div>
<div class="sr"><span class="l">outgoing</span><span class="v p" id="ho">-</span></div>
<div class="bar ob" id="hob" style="width:0%"></div>
<div class="sr"><span class="l">incoming</span><span class="v n" id="hin">-</span></div>
<div class="bar ib" id="hib" style="width:0%"></div>
</div>
<div class="panel" id="controls">
<button class="btn active" id="be" onclick="toggleEdges()">edges on</button>
<button class="btn" id="bl" onclick="toggleLayout()">force layout</button>
<select class="btn" id="ef" onchange="nr=true">
<option value="all">all edges</option>
<option value="pos">excitatory only</option>
<option value="neg">inhibitory only</option>
</select>
<button class="btn" onclick="resetView()">reset view</button>
</div>
<canvas id="c"></canvas>
<script>
const D=__DATA__;
const cv=document.getElementById("c"),cx=cv.getContext("2d");
let W,H;function rs(){W=cv.width=innerWidth;H=cv.height=innerHeight}
rs();window.onresize=()=>{rs();lc();nr=true};
const N=D.nodes,L=D.links,im={};
N.forEach((n,i)=>{im[n.id]=i;n.x=0;n.y=0});
L.forEach(l=>{l.si=im[l.s];l.ti=im[l.t]});
const md=Math.max(...N.map(n=>n.deg));
const pc=L.filter(l=>l.v>0).length,nc=L.length-pc;
const ad=(N.reduce((a,n)=>a+n.deg,0)/N.length).toFixed(1);
const dn=(L.length/(N.length*(N.length-1))*100).toFixed(2);
document.getElementById("sn").textContent=N.length;
document.getElementById("se").textContent=L.length;
document.getElementById("sp").textContent=pc;
document.getElementById("sng").textContent=nc;
document.getElementById("sm").textContent=md;
document.getElementById("sa").textContent=ad;
document.getElementById("sd").textContent=dn+"%";
let se=true,lm="circle",zm=1,px=0,py=0,dg=false,dx,dy,hv=-1,nr=true;
function lc(){
const cx2=W/2,cy=H/2,r=Math.min(W,H)*0.38;
const s=[...N].sort((a,b)=>b.deg-a.deg);
s.forEach((n,i)=>{const a=(i/N.length)*Math.PI*2-Math.PI/2;n.x=cx2+Math.cos(a)*r;n.y=cy+Math.sin(a)*r});
}
lc();
function toggleEdges(){se=!se;const b=document.getElementById("be");b.textContent=se?"edges on":"edges off";b.classList.toggle("active",se);nr=true}
function toggleLayout(){
const b=document.getElementById("bl");
if(lm==="circle"){lm="force";b.textContent="circle layout";b.classList.add("active");rf(80)}
else{lm="circle";b.textContent="force layout";b.classList.remove("active");lc()}
nr=true}
function rf(it){
for(let t=0;t<it;t++){
N.forEach(n=>{n.fx=0;n.fy=0});
for(let i=0;i<N.length;i++)for(let j=i+1;j<N.length;j++){
let dx2=N[j].x-N[i].x,dy2=N[j].y-N[i].y,d=Math.sqrt(dx2*dx2+dy2*dy2)+1,f=3000/(d*d);
N[i].fx-=f*dx2/d;N[i].fy-=f*dy2/d;N[j].fx+=f*dx2/d;N[j].fy+=f*dy2/d}
L.forEach(l=>{if(l.si==null||l.ti==null)return;let a=N[l.si],b=N[l.ti],dx2=b.x-a.x,dy2=b.y-a.y,d=Math.sqrt(dx2*dx2+dy2*dy2)+1,f=d*0.002;
a.fx+=f*dx2;a.fy+=f*dy2;b.fx-=f*dx2;b.fy-=f*dy2});
N.forEach(n=>{n.fx+=(W/2-n.x)*0.005;n.fy+=(H/2-n.y)*0.005;n.x+=n.fx*0.4;n.y+=n.fy*0.4})}}
function resetView(){zm=1;px=0;py=0;nr=true}
function render(){
cx.clearRect(0,0,W,H);
cx.save();cx.globalAlpha=0.03;cx.strokeStyle="#00ff88";
for(let x=0;x<W;x+=80){cx.beginPath();cx.moveTo(x,0);cx.lineTo(x,H);cx.stroke()}
for(let y=0;y<H;y+=80){cx.beginPath();cx.moveTo(0,y);cx.lineTo(W,y);cx.stroke()}
cx.restore();
cx.save();cx.translate(px,py);cx.scale(zm,zm);
const fi=document.getElementById("ef").value;
if(se){cx.lineWidth=0.5;cx.globalAlpha=0.06;
L.forEach(l=>{if(l.si==null||l.ti==null)return;if(fi==="pos"&&l.v<0)return;if(fi==="neg"&&l.v>0)return;
const a=N[l.si],b=N[l.ti];cx.strokeStyle=l.v>0?"#00ff88":"#ff3355";cx.beginPath();cx.moveTo(a.x,a.y);cx.lineTo(b.x,b.y);cx.stroke()});
if(hv>=0){cx.globalAlpha=0.6;cx.lineWidth=1.5;
L.forEach(l=>{if(l.si==null||l.ti==null)return;
if(N[l.si].id!==hv&&N[l.ti].id!==hv)return;
const a=N[l.si],b=N[l.ti];cx.strokeStyle=l.v>0?"#00ff88":"#ff3355";cx.beginPath();cx.moveTo(a.x,a.y);cx.lineTo(b.x,b.y);cx.stroke()});
cx.lineWidth=0.5}}
cx.globalAlpha=1;
N.forEach(n=>{const r=1.5+(n.deg/md)*4,br=0.2+0.8*(n.deg/md);
if(n.id===hv){cx.shadowBlur=20;cx.shadowColor="#00ff88";cx.fillStyle="#fff";cx.beginPath();cx.arc(n.x,n.y,r+2,0,Math.PI*2);cx.fill();cx.shadowBlur=0}
const g=Math.floor(br*255),b2=Math.floor(br*100);
cx.fillStyle="rgb("+Math.floor(b2/2)+","+g+","+b2+")";cx.beginPath();cx.arc(n.x,n.y,r,0,Math.PI*2);cx.fill()});
cx.restore()}
cv.addEventListener("wheel",e=>{e.preventDefault();zm*=e.deltaY>0?0.92:1.08;zm=Math.max(0.1,Math.min(12,zm));nr=true});
cv.addEventListener("mousedown",e=>{dg=true;dx=e.clientX-px;dy=e.clientY-py});
cv.addEventListener("mouseup",()=>dg=false);
cv.addEventListener("mousemove",e=>{
if(dg){px=e.clientX-dx;py=e.clientY-dy;nr=true;return}
const mx=(e.clientX-px)/zm,my=(e.clientY-py)/zm;
let best=-1,bd=15/zm;
N.forEach(n=>{const d=Math.sqrt((n.x-mx)**2+(n.y-my)**2);if(d<bd){bd=d;best=n.id}});
if(best!==hv){hv=best;uh();nr=true}});
function uh(){
const el=document.getElementById("hover");
if(hv<0){el.classList.add("empty");document.getElementById("hi").textContent="hover a neuron";
["hd","ho","hin"].forEach(i=>document.getElementById(i).textContent="-");
["hob","hib"].forEach(i=>document.getElementById(i).style.width="0%");return}
el.classList.remove("empty");
const n=N.find(n=>n.id===hv);
document.getElementById("hi").textContent="neuron "+n.id;
document.getElementById("hd").textContent=n.deg;
document.getElementById("ho").textContent=n.out;
document.getElementById("hin").textContent=n.inp;
document.getElementById("hob").style.width=(n.out/md*100)+"%";
document.getElementById("hib").style.width=(n.inp/md*100)+"%"}
function loop(){if(nr){render();nr=false}requestAnimationFrame(loop)}
loop();
</script>
</body>
</html>"""

html = html.replace('__DATA__', compact)

with open('s:/AI/work/VRAXION_DEV/v4.2/network_viz_v2.html', 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Saved network_viz_v2.html ({len(html)} bytes)")
