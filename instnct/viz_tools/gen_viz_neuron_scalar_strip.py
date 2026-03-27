"""Generate a minimal single-field neuron scalar strip visualization."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIELD = "theta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a minimal per-neuron scalar strip. One field per output. "
            "Useful for theta/decay now, and rho/freq/phase/polarity later."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to render. Defaults to the latest canonical English checkpoint.",
    )
    parser.add_argument(
        "--field",
        default=DEFAULT_FIELD,
        choices=["theta", "decay", "rho", "freq", "phase", "polarity"],
        help="Scalar field to visualize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to instnct/network_<field>_strip.html",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional HTML title/header. Defaults to '<FIELD> Neuron Strip'.",
    )
    return parser.parse_args()


def _step_key(path: Path) -> tuple[int, str]:
    match = re.search(r"_step(\d+)\.npz$", path.name)
    if match:
        return int(match.group(1)), path.name
    return -1, path.name


def pick_default_checkpoint() -> Path:
    candidates = sorted((ROOT / "checkpoints").glob("english_1024n_step*.npz"), key=_step_key)
    if candidates:
        return candidates[-1]
    fallback = sorted((ROOT / "checkpoints").glob("english_1024n*.npz"), key=lambda p: p.stat().st_mtime)
    if fallback:
        return fallback[-1]
    generic = sorted((ROOT / "checkpoints").glob("*.npz"), key=lambda p: p.stat().st_mtime)
    if generic:
        return generic[-1]
    raise FileNotFoundError("No checkpoint found under instnct/checkpoints")


def load_field(path: Path, field: str) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        if field not in data.files:
            raise ValueError(f"Field '{field}' missing from checkpoint {path.name}")
        arr = np.asarray(data[field])
    if arr.ndim != 1:
        raise ValueError(f"Field '{field}' must be 1D, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def build_payload(path: Path, field: str, title: str) -> dict[str, object]:
    values = load_field(path, field)
    lo = float(values.min()) if values.size else 0.0
    hi = float(values.max()) if values.size else 0.0
    mean = float(values.mean()) if values.size else 0.0
    std = float(values.std()) if values.size else 0.0
    nonzero = int((values != 0).sum())

    return {
        "title": title,
        "checkpoint_name": path.name,
        "checkpoint_path": str(path),
        "field": field,
        "count": int(values.shape[0]),
        "values": values.tolist(),
        "min": lo,
        "max": hi,
        "mean": mean,
        "std": std,
        "nonzero": nonzero,
        "argmin": int(values.argmin()) if values.size else -1,
        "argmax": int(values.argmax()) if values.size else -1,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
@import url("https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Outfit:wght@200;400;700&display=swap");
:root{--bg:#04060a;--panel:rgba(8,14,22,0.92);--border:rgba(0,255,136,0.15);--accent:#00ff88;--hot:#ff8844;--cold:#224466;--text:#b8c4d0;--dim:#445566}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:"JetBrains Mono",monospace;overflow:hidden;cursor:crosshair}
canvas{display:block}
.panel{position:fixed;background:var(--panel);border:1px solid var(--border);backdrop-filter:blur(12px);padding:14px;z-index:10;border-radius:2px}
#header{top:12px;left:12px;min-width:280px}
#header h1{font-family:"Outfit",sans-serif;font-weight:200;font-size:20px;color:var(--accent);letter-spacing:3px;text-transform:uppercase;text-shadow:0 0 20px rgba(0,255,136,0.15)}
#header .sub{font-size:9px;color:var(--dim);letter-spacing:2px;margin:4px 0 10px}
.sr{display:flex;justify-content:space-between;padding:2px 0;font-size:10px;border-bottom:1px solid rgba(255,255,255,0.03)}
.sr .l{color:var(--dim)} .sr .v{font-weight:600}
.sr .v.a{color:var(--accent)} .sr .v.h{color:var(--hot)}
#hover{bottom:12px;left:12px;min-width:250px;font-size:10px;transition:opacity 0.15s}
#hover.empty{opacity:0.35}
#hover .nid{font-family:"Outfit";font-size:16px;color:var(--accent);font-weight:700}
#controls{top:12px;right:12px;display:flex;flex-direction:column;gap:5px}
.btn{background:transparent;color:var(--accent);border:1px solid var(--border);padding:5px 12px;font-family:"JetBrains Mono",monospace;font-size:9px;letter-spacing:1px;cursor:pointer;transition:all 0.15s;text-transform:uppercase}
.btn:hover{background:rgba(0,255,136,0.1);border-color:var(--accent)}
.scanline{position:fixed;top:0;left:0;right:0;height:100vh;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,255,136,0.006) 2px,rgba(0,255,136,0.006) 4px);pointer-events:none;z-index:100}
.legend{position:fixed;bottom:12px;right:12px;z-index:10;font-size:9px;color:var(--dim);background:var(--panel);border:1px solid var(--border);padding:10px;border-radius:2px}
</style>
</head>
<body>
<div class="scanline"></div>
<div class="panel" id="header">
  <h1>__TITLE__</h1>
  <div class="sub">single scalar field per neuron, one info / layer</div>
  <div class="sr"><span class="l">checkpoint</span><span class="v" id="s-ckpt">-</span></div>
  <div class="sr"><span class="l">field</span><span class="v a" id="s-field">-</span></div>
  <div class="sr"><span class="l">neuron count</span><span class="v" id="s-count">-</span></div>
  <div class="sr"><span class="l">min</span><span class="v" id="s-min">-</span></div>
  <div class="sr"><span class="l">max</span><span class="v h" id="s-max">-</span></div>
  <div class="sr"><span class="l">mean ± std</span><span class="v" id="s-mean">-</span></div>
  <div class="sr"><span class="l">nonzero</span><span class="v" id="s-nz">-</span></div>
  <div class="sr"><span class="l">argmin / argmax</span><span class="v" id="s-ext">-</span></div>
</div>
<div class="panel empty" id="hover">
  <div class="nid" id="h-title">hover a neuron</div>
  <div class="sr"><span class="l">neuron</span><span class="v" id="h-neuron">-</span></div>
  <div class="sr"><span class="l">value</span><span class="v" id="h-value">-</span></div>
  <div class="sr"><span class="l">normalized</span><span class="v" id="h-norm">-</span></div>
</div>
<div class="panel" id="controls">
  <button class="btn" onclick="resetView()">reset view</button>
</div>
<div class="legend">
  cold = low value<br>
  hot = high value
</div>
<canvas id="c"></canvas>
<script>
const D = __DATA__;
const cv = document.getElementById("c");
const cx = cv.getContext("2d");
let W=0,H=0;
function resize(){ W=cv.width=innerWidth; H=cv.height=innerHeight; fitView(); dirty=true; }
window.addEventListener("resize", resize);

const vals = D.values;
const N = vals.length;
const lo = D.min;
const hi = D.max;
const span = Math.max(1e-9, hi - lo);

document.getElementById("s-ckpt").textContent = D.checkpoint_name;
document.getElementById("s-field").textContent = D.field;
document.getElementById("s-count").textContent = D.count;
document.getElementById("s-min").textContent = lo.toFixed(4);
document.getElementById("s-max").textContent = hi.toFixed(4);
document.getElementById("s-mean").textContent = D.mean.toFixed(4) + " ± " + D.std.toFixed(4);
document.getElementById("s-nz").textContent = D.nonzero;
document.getElementById("s-ext").textContent = D.argmin + " / " + D.argmax;

let scale = 1;
let panY = 0;
let dragging = false;
let dragY = 0;
let hover = -1;
let dirty = true;

function fitView(){
  const topPad = 24;
  const botPad = 24;
  const stripHeight = Math.max(260, H - topPad - botPad);
  scale = stripHeight / N;
  panY = topPad;
}
function resetView(){ fitView(); dirty = true; }

function colorForValue(v){
  const t = (v - lo) / span;
  const r = Math.round(34 + t * 221);
  const g = Math.round(68 + t * 136);
  const b = Math.round(102 - t * 68);
  return `rgb(${r},${g},${b})`;
}

function updateHover(idx){
  hover = idx;
  const panel = document.getElementById("hover");
  if (idx < 0 || idx >= N){
    panel.classList.add("empty");
    document.getElementById("h-title").textContent = "hover a neuron";
    ["h-neuron","h-value","h-norm"].forEach(id => document.getElementById(id).textContent = "-");
    dirty = true;
    return;
  }
  panel.classList.remove("empty");
  const v = vals[idx];
  const n = (v - lo) / span;
  document.getElementById("h-title").textContent = `neuron ${idx}`;
  document.getElementById("h-neuron").textContent = idx;
  document.getElementById("h-value").textContent = v.toFixed(6);
  document.getElementById("h-norm").textContent = n.toFixed(4);
  dirty = true;
}

cv.addEventListener("mousedown", (e) => { dragging = true; dragY = e.clientY - panY; });
window.addEventListener("mouseup", () => { dragging = false; });
window.addEventListener("mousemove", (e) => {
  if (dragging){
    panY = e.clientY - dragY;
    dirty = true;
    return;
  }
  const idx = Math.floor((e.clientY - panY) / scale);
  updateHover(idx);
});
cv.addEventListener("wheel", (e) => {
  e.preventDefault();
  const my = e.clientY;
  const before = (my - panY) / scale;
  const factor = e.deltaY > 0 ? 0.92 : 1.08;
  scale = Math.max(0.25, Math.min(24, scale * factor));
  panY = my - before * scale;
  dirty = true;
}, { passive: false });

function render(){
  cx.clearRect(0, 0, W, H);
  const stripX = Math.max(360, W * 0.40);
  const stripW = Math.min(220, Math.max(80, W * 0.12));

  cx.save();
  cx.fillStyle = "#05080e";
  cx.fillRect(stripX, panY, stripW, N * scale);

  for (let i = 0; i < N; i += 1){
    const y = panY + i * scale;
    if (y + scale < 0 || y > H) continue;
    cx.fillStyle = colorForValue(vals[i]);
    cx.fillRect(stripX, y, stripW, Math.max(1, scale));
  }

  if (hover >= 0 && hover < N){
    const y = panY + hover * scale;
    cx.fillStyle = "rgba(255,255,255,0.12)";
    cx.fillRect(stripX - 18, y, stripW + 36, Math.max(1, scale));
    cx.strokeStyle = "#ffffff";
    cx.lineWidth = 2;
    cx.strokeRect(stripX - 1, y - 1, stripW + 2, Math.max(1, scale) + 2);
    cx.fillStyle = "#ff8844";
    cx.font = "11px JetBrains Mono";
    cx.fillText(String(hover), stripX + stripW + 12, y + Math.max(11, scale * 0.8));
  }

  cx.strokeStyle = "rgba(0,255,136,0.30)";
  cx.lineWidth = 1;
  cx.strokeRect(stripX, panY, stripW, N * scale);
  cx.restore();
}

function loop(){
  if (dirty){
    render();
    dirty = false;
  }
  requestAnimationFrame(loop);
}

resize();
loop();
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else pick_default_checkpoint()
    title = args.title or f"{args.field.upper()} Neuron Strip"
    output = args.output.resolve() if args.output else (ROOT / f"network_{args.field}_strip.html")
    payload = build_payload(checkpoint, args.field, title)
    compact = json.dumps(payload, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", compact)
    output.write_text(html, encoding="utf-8")
    print(
        f"Saved {output} from {checkpoint.name} "
        f"(field={args.field}, N={payload['count']}, min={payload['min']:.4f}, max={payload['max']:.4f})"
    )


if __name__ == "__main__":
    main()
