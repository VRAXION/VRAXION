"""Generate a single-page workbench for simple layer visualizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gen_viz_connection_matrix import build_payload as build_connection_payload
from gen_viz_connection_matrix import pick_default_checkpoint
from gen_viz_neuron_scalar_strip import load_field


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "network_layer_workbench.html"
SCALAR_FIELDS = ("theta", "decay", "rho", "freq", "phase", "polarity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one single-page HTML workbench containing simple per-layer "
            "visualizations for the current checkpoint."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to render. Defaults to the latest canonical English checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output HTML path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--title",
        default="SWG Layer Workbench",
        help="HTML title/header label.",
    )
    return parser.parse_args()


def build_scalar_layer(path: Path, field: str) -> dict[str, object] | None:
    try:
        values = load_field(path, field)
    except ValueError:
        return None

    lo = float(values.min()) if values.size else 0.0
    hi = float(values.max()) if values.size else 0.0
    mean = float(values.mean()) if values.size else 0.0
    std = float(values.std()) if values.size else 0.0
    nonzero = int((values != 0).sum())
    return {
        "kind": "scalar",
        "id": field,
        "label": field.upper(),
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


def build_scalar_layer_from_values(field: str, label: str, values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float32)
    lo = float(values.min()) if values.size else 0.0
    hi = float(values.max()) if values.size else 0.0
    mean = float(values.mean()) if values.size else 0.0
    std = float(values.std()) if values.size else 0.0
    nonzero = int((values != 0).sum())
    return {
        "kind": "scalar",
        "id": field,
        "label": label,
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


def build_binary_strip_layer(field: str, label: str, flags: np.ndarray) -> dict[str, object]:
    flags = np.asarray(flags, dtype=np.bool_)
    values = flags.astype(np.uint8, copy=False)
    nonzero = int(values.sum())
    return {
        "kind": "binary_strip",
        "id": field,
        "label": label,
        "count": int(values.shape[0]),
        "values": values.tolist(),
        "min": int(values.min()) if values.size else 0,
        "max": int(values.max()) if values.size else 0,
        "mean": float(values.mean()) if values.size else 0.0,
        "std": float(values.std()) if values.size else 0.0,
        "nonzero": nonzero,
        "argmin": int(values.argmin()) if values.size else -1,
        "argmax": int(values.argmax()) if values.size else -1,
    }


def build_matrix_layer(layer_id: str, label: str, hidden_size: int, rows: np.ndarray, cols: np.ndarray, out_degree: np.ndarray, in_degree: np.ndarray, blocked_self_edges: int = 0) -> dict[str, object]:
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    edge_set = set(zip(rows.tolist(), cols.tolist()))
    reciprocal_pairs = sum(1 for r, c in edge_set if r < c and (c, r) in edge_set)
    edge_count = int(rows.size)
    sink = int(((out_degree == 0) & (in_degree > 0)).sum())
    source_only = int(((out_degree > 0) & (in_degree == 0)).sum())
    isolated = int(((out_degree == 0) & (in_degree == 0)).sum())
    return {
        "kind": "matrix",
        "id": layer_id,
        "label": label,
        "description": f"Binary matrix layer: {layer_id}",
        "hidden_size": int(hidden_size),
        "rows": rows.tolist(),
        "cols": cols.tolist(),
        "edge_count": edge_count,
        "density": (edge_count / (hidden_size * (hidden_size - 1))) if hidden_size > 1 else 0.0,
        "reciprocal_pairs": reciprocal_pairs,
        "sink_count": sink,
        "source_only_count": source_only,
        "isolated_count": isolated,
        "blocked_self_edges": int(blocked_self_edges),
        "out_degree": out_degree.astype(np.int32, copy=False).tolist(),
        "in_degree": in_degree.astype(np.int32, copy=False).tolist(),
    }


def build_payload(path: Path, title: str) -> dict[str, object]:
    connection = build_connection_payload(path, title)
    rows = np.asarray(connection["rows"], dtype=np.int32)
    cols = np.asarray(connection["cols"], dtype=np.int32)
    hidden_size = int(connection["hidden_size"])
    out_degree = np.asarray(connection["out_degree"], dtype=np.int32)
    in_degree = np.asarray(connection["in_degree"], dtype=np.int32)

    layers: list[dict[str, object]] = [build_matrix_layer(
        "connection_presence",
        "CONNECTION",
        hidden_size,
        rows,
        cols,
        out_degree,
        in_degree,
        blocked_self_edges=int(connection["blocked_self_edges"]),
    )]

    edge_set = set(zip(rows.tolist(), cols.tolist()))
    reciprocal_pairs = sorted((r, c) for r, c in edge_set if (c, r) in edge_set)
    reciprocal_rows = np.asarray([r for r, _ in reciprocal_pairs], dtype=np.int32)
    reciprocal_cols = np.asarray([c for _, c in reciprocal_pairs], dtype=np.int32)
    reciprocal_out = np.bincount(reciprocal_rows, minlength=hidden_size).astype(np.int32)
    reciprocal_in = np.bincount(reciprocal_cols, minlength=hidden_size).astype(np.int32)
    layers.append(build_matrix_layer(
        "reciprocal_edges",
        "RECIPROCAL",
        hidden_size,
        reciprocal_rows,
        reciprocal_cols,
        reciprocal_out,
        reciprocal_in,
    ))

    layers.append(build_binary_strip_layer(
        "active_neuron",
        "ACTIVE",
        (out_degree + in_degree) > 0,
    ))
    layers.append(build_binary_strip_layer(
        "sink_neuron",
        "SINK",
        (out_degree == 0) & (in_degree > 0),
    ))
    layers.append(build_binary_strip_layer(
        "source_only_neuron",
        "SOURCE_ONLY",
        (out_degree > 0) & (in_degree == 0),
    ))
    layers.append(build_binary_strip_layer(
        "isolated_neuron",
        "ISOLATED",
        (out_degree == 0) & (in_degree == 0),
    ))

    for field in SCALAR_FIELDS:
        layer = build_scalar_layer(path, field)
        if layer is not None:
            layers.append(layer)

    return {
        "title": title,
        "checkpoint_name": path.name,
        "checkpoint_path": str(path),
        "layers": layers,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
@import url("https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Outfit:wght@200;400;700&display=swap");
:root{--bg:#04060a;--panel:rgba(8,14,22,0.92);--border:rgba(0,255,136,0.15);--accent:#00ff88;--hot:#ff8844;--block:#ff3355;--text:#b8c4d0;--dim:#445566}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:"JetBrains Mono",monospace;overflow:hidden;cursor:crosshair}
canvas{display:block}
.panel{position:fixed;background:var(--panel);border:1px solid var(--border);backdrop-filter:blur(12px);padding:14px;z-index:10;border-radius:2px}
#header{top:12px;left:12px;min-width:300px}
#header h1{font-family:"Outfit",sans-serif;font-weight:200;font-size:20px;color:var(--accent);letter-spacing:3px;text-transform:uppercase;text-shadow:0 0 20px rgba(0,255,136,0.15)}
#header .sub{font-size:9px;color:var(--dim);letter-spacing:2px;margin:4px 0 10px}
.sr{display:flex;justify-content:space-between;padding:2px 0;font-size:10px;border-bottom:1px solid rgba(255,255,255,0.03)}
.sr .l{color:var(--dim)} .sr .v{font-weight:600}
.sr .v.a{color:var(--accent)} .sr .v.h{color:var(--hot)} .sr .v.b{color:var(--block)}
#hover{bottom:12px;left:12px;min-width:260px;font-size:10px;transition:opacity 0.15s}
#hover.empty{opacity:0.35}
#hover .nid{font-family:"Outfit";font-size:16px;color:var(--accent);font-weight:700}
#controls{top:12px;right:12px;display:flex;flex-direction:column;gap:6px;max-width:220px}
#layer-tabs{display:flex;flex-wrap:wrap;gap:5px}
.btn{background:transparent;color:var(--accent);border:1px solid var(--border);padding:5px 12px;font-family:"JetBrains Mono",monospace;font-size:9px;letter-spacing:1px;cursor:pointer;transition:all 0.15s;text-transform:uppercase}
.btn:hover{background:rgba(0,255,136,0.1);border-color:var(--accent)}
.btn.active{background:rgba(0,255,136,0.15);border-color:var(--accent);color:#fff}
.scanline{position:fixed;top:0;left:0;right:0;height:100vh;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,255,136,0.006) 2px,rgba(0,255,136,0.006) 4px);pointer-events:none;z-index:100}
.legend{position:fixed;bottom:12px;right:12px;z-index:10;font-size:9px;color:var(--dim);background:var(--panel);border:1px solid var(--border);padding:10px;border-radius:2px;max-width:230px}
</style>
</head>
<body>
<div class="scanline"></div>
<div class="panel" id="header">
  <h1>__TITLE__</h1>
  <div class="sub">single page · one info per layer · simple masks first</div>
  <div class="sr"><span class="l">checkpoint</span><span class="v" id="s-ckpt">-</span></div>
  <div class="sr"><span class="l">active layer</span><span class="v a" id="s-layer">-</span></div>
  <div class="sr"><span class="l">summary 1</span><span class="v" id="s1">-</span></div>
  <div class="sr"><span class="l">summary 2</span><span class="v" id="s2">-</span></div>
  <div class="sr"><span class="l">summary 3</span><span class="v" id="s3">-</span></div>
  <div class="sr"><span class="l">summary 4</span><span class="v" id="s4">-</span></div>
</div>
<div class="panel empty" id="hover">
  <div class="nid" id="h-title">hover</div>
  <div class="sr"><span class="l">field 1</span><span class="v" id="h1">-</span></div>
  <div class="sr"><span class="l">field 2</span><span class="v" id="h2">-</span></div>
  <div class="sr"><span class="l">field 3</span><span class="v" id="h3">-</span></div>
</div>
<div class="panel" id="controls">
  <div id="layer-tabs"></div>
  <button class="btn active" id="btn-guides" onclick="toggleGuides()">guides on</button>
  <button class="btn" onclick="resetView()">reset view</button>
</div>
<div class="legend" id="legend"></div>
<canvas id="c"></canvas>
<script>
const D = __DATA__;
const cv = document.getElementById("c");
const cx = cv.getContext("2d");
let W=0,H=0;
function resize(){ W=cv.width=innerWidth; H=cv.height=innerHeight; fitView(); dirty=true; }
window.addEventListener("resize", resize);

const layers = D.layers;
let activeLayer = layers[0];
let scale = 1;
let panX = 0;
let panY = 0;
let dragging = false;
let dragX = 0;
let dragY = 0;
let hover = null;
let showGuides = true;
let dirty = true;

function setSummary(lines){
  document.getElementById("s1").textContent = lines[0] || "-";
  document.getElementById("s2").textContent = lines[1] || "-";
  document.getElementById("s3").textContent = lines[2] || "-";
  document.getElementById("s4").textContent = lines[3] || "-";
}
function setHover(title, rows){
  const panel = document.getElementById("hover");
  if (!rows){
    panel.classList.add("empty");
    document.getElementById("h-title").textContent = "hover";
    document.getElementById("h1").textContent = "-";
    document.getElementById("h2").textContent = "-";
    document.getElementById("h3").textContent = "-";
    return;
  }
  panel.classList.remove("empty");
  document.getElementById("h-title").textContent = title;
  document.getElementById("h1").textContent = rows[0] || "-";
  document.getElementById("h2").textContent = rows[1] || "-";
  document.getElementById("h3").textContent = rows[2] || "-";
}
function setLegend(lines){
  document.getElementById("legend").innerHTML = lines.join("<br>");
}

function updateHeader(){
  document.getElementById("s-ckpt").textContent = D.checkpoint_name;
  document.getElementById("s-layer").textContent = activeLayer.label;
  if (activeLayer.kind === "matrix"){
    setSummary([
      `edges ${activeLayer.edge_count}`,
      `density ${(activeLayer.density * 100).toFixed(3)}%`,
      `reciprocal ${activeLayer.reciprocal_pairs}`,
      `sink ${activeLayer.sink_count} · iso ${activeLayer.isolated_count}`,
    ]);
    setLegend([
      "green = connection present",
      "red = blocked diagonal",
      "orange = hover row / column",
    ]);
  } else if (activeLayer.kind === "binary_strip") {
    setSummary([
      `count ${activeLayer.count}`,
      `active ${activeLayer.nonzero}`,
      `inactive ${activeLayer.count - activeLayer.nonzero}`,
      `mean ${(activeLayer.mean * 100).toFixed(2)}% on`,
    ]);
    setLegend([
      "green = active binary flag",
      "dim = inactive binary flag",
      "one binary strip per neuron",
    ]);
  } else {
    setSummary([
      `count ${activeLayer.count}`,
      `min ${activeLayer.min.toFixed(4)} · max ${activeLayer.max.toFixed(4)}`,
      `mean ${activeLayer.mean.toFixed(4)} ± ${activeLayer.std.toFixed(4)}`,
      `nonzero ${activeLayer.nonzero}`,
    ]);
    setLegend([
      "cold = low value",
      "hot = high value",
      "one scalar field per neuron",
    ]);
  }
}

function buildTabs(){
  const container = document.getElementById("layer-tabs");
  container.innerHTML = "";
  layers.forEach((layer) => {
    const btn = document.createElement("button");
    btn.className = "btn" + (layer.id === activeLayer.id ? " active" : "");
    btn.textContent = layer.label;
    btn.onclick = () => {
      activeLayer = layer;
      hover = null;
      fitView();
      updateHeader();
      setHover("hover", null);
      buildTabs();
      dirty = true;
    };
    container.appendChild(btn);
  });
}

function fitView(){
  if (activeLayer.kind === "matrix"){
    const sidePad = 350;
    const topPad = 24;
    const matrixSpaceW = Math.max(220, W - sidePad - 24);
    const matrixSpaceH = Math.max(220, H - topPad - 24);
    scale = Math.min(matrixSpaceW / activeLayer.hidden_size, matrixSpaceH / activeLayer.hidden_size);
    panX = sidePad + Math.max(0, (matrixSpaceW - activeLayer.hidden_size * scale) / 2);
    panY = topPad + Math.max(0, (matrixSpaceH - activeLayer.hidden_size * scale) / 2);
  } else {
    const topPad = 24;
    const stripHeight = Math.max(260, H - topPad - 24);
    scale = stripHeight / activeLayer.count;
    panX = 0;
    panY = topPad;
  }
}

function toggleGuides(){
  showGuides = !showGuides;
  const btn = document.getElementById("btn-guides");
  btn.textContent = showGuides ? "guides on" : "guides off";
  btn.classList.toggle("active", showGuides);
  dirty = true;
}
function resetView(){ fitView(); dirty = true; }

function matrixCellAt(x, y){
  const col = Math.floor((x - panX) / scale);
  const row = Math.floor((y - panY) / scale);
  const N = activeLayer.hidden_size;
  if (row < 0 || row >= N || col < 0 || col >= N) return null;
  return { row, col };
}
function scalarIndexAt(y){
  const idx = Math.floor((y - panY) / scale);
  if (idx < 0 || idx >= activeLayer.count) return null;
  return idx;
}

function colorForScalar(v, lo, hi){
  const span = Math.max(1e-9, hi - lo);
  const t = (v - lo) / span;
  const r = Math.round(34 + t * 221);
  const g = Math.round(68 + t * 136);
  const b = Math.round(102 - t * 68);
  return `rgb(${r},${g},${b})`;
}

function renderMatrix(){
  const N = activeLayer.hidden_size;
  const edgeSet = activeLayer._edgeSet || (activeLayer._edgeSet = new Set(activeLayer.rows.map((r, i) => `${r}:${activeLayer.cols[i]}`)));
  cx.fillStyle = "#05080e";
  cx.fillRect(panX, panY, N * scale, N * scale);

  cx.save();
  cx.translate(panX, panY);
  cx.scale(scale, scale);

  if (showGuides && hover){
    cx.fillStyle = "rgba(255,136,68,0.10)";
    cx.fillRect(0, hover.row, N, 1);
    cx.fillRect(hover.col, 0, 1, N);
  }

  if (scale >= 1.5){
    cx.fillStyle = "rgba(255,51,85,0.45)";
    for (let i = 0; i < N; i += 1) cx.fillRect(i, i, 1, 1);
  } else {
    cx.strokeStyle = "rgba(255,51,85,0.65)";
    cx.lineWidth = Math.max(1 / scale, 1.4 / scale);
    cx.beginPath();
    cx.moveTo(0, 0);
    cx.lineTo(N, N);
    cx.stroke();
  }

  cx.fillStyle = "#00ff88";
  for (let i = 0; i < activeLayer.rows.length; i += 1){
    cx.fillRect(activeLayer.cols[i], activeLayer.rows[i], 1, 1);
  }

  if (hover){
    cx.lineWidth = 2 / scale;
    cx.strokeStyle = "#ffffff";
    cx.strokeRect(hover.col, hover.row, 1, 1);
  }

  cx.strokeStyle = "rgba(0,255,136,0.30)";
  cx.lineWidth = Math.max(1 / scale, 1.2 / scale);
  cx.strokeRect(0, 0, N, N);
  cx.restore();

  if (hover){
    const state = hover.row === hover.col ? "blocked diagonal" : (edgeSet.has(`${hover.row}:${hover.col}`) ? "connection present" : "no connection");
    setHover(
      `cell ${hover.row}, ${hover.col}`,
      [
        `source ${hover.row} · target ${hover.col}`,
        state,
        `out ${activeLayer.out_degree[hover.row]} · in ${activeLayer.in_degree[hover.col]}`,
      ]
    );
  } else {
    setHover("hover a cell", null);
  }
}

function renderScalar(){
  const stripX = Math.max(370, W * 0.40);
  const stripW = Math.min(220, Math.max(80, W * 0.12));
  cx.fillStyle = "#05080e";
  cx.fillRect(stripX, panY, stripW, activeLayer.count * scale);

  for (let i = 0; i < activeLayer.count; i += 1){
    const y = panY + i * scale;
    if (y + scale < 0 || y > H) continue;
    cx.fillStyle = colorForScalar(activeLayer.values[i], activeLayer.min, activeLayer.max);
    cx.fillRect(stripX, y, stripW, Math.max(1, scale));
  }

  if (showGuides && hover !== null){
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
  cx.strokeRect(stripX, panY, stripW, activeLayer.count * scale);

  if (hover !== null){
    const v = activeLayer.values[hover];
    const n = (v - activeLayer.min) / Math.max(1e-9, activeLayer.max - activeLayer.min);
    setHover(
      `neuron ${hover}`,
      [
        `${activeLayer.id} = ${v.toFixed(6)}`,
        `normalized ${n.toFixed(4)}`,
        hover === activeLayer.argmin ? "global min" : (hover === activeLayer.argmax ? "global max" : "in range"),
      ]
    );
  } else {
    setHover("hover a neuron", null);
  }
}

function renderBinaryStrip(){
  const stripX = Math.max(370, W * 0.40);
  const stripW = Math.min(220, Math.max(80, W * 0.12));
  cx.fillStyle = "#05080e";
  cx.fillRect(stripX, panY, stripW, activeLayer.count * scale);

  for (let i = 0; i < activeLayer.count; i += 1){
    const y = panY + i * scale;
    if (y + scale < 0 || y > H) continue;
    cx.fillStyle = activeLayer.values[i] ? "#00ff88" : "rgba(68,85,102,0.55)";
    cx.fillRect(stripX, y, stripW, Math.max(1, scale));
  }

  if (showGuides && hover !== null){
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
  cx.strokeRect(stripX, panY, stripW, activeLayer.count * scale);

  if (hover !== null){
    const v = !!activeLayer.values[hover];
    setHover(
      `neuron ${hover}`,
      [
        `${activeLayer.id} = ${v ? "on" : "off"}`,
        v ? "binary flag active" : "binary flag inactive",
        v ? "member of current mask" : "outside current mask",
      ]
    );
  } else {
    setHover("hover a neuron", null);
  }
}

function render(){
  cx.clearRect(0, 0, W, H);
  if (activeLayer.kind === "matrix") renderMatrix();
  else if (activeLayer.kind === "binary_strip") renderBinaryStrip();
  else renderScalar();
}

cv.addEventListener("mousedown", (e) => {
  dragging = true;
  dragX = e.clientX - panX;
  dragY = e.clientY - panY;
});
window.addEventListener("mouseup", () => { dragging = false; });
window.addEventListener("mousemove", (e) => {
  if (dragging){
    if (activeLayer.kind === "matrix"){
      panX = e.clientX - dragX;
    }
    panY = e.clientY - dragY;
    dirty = true;
    return;
  }
  if (activeLayer.kind === "matrix"){
    hover = matrixCellAt(e.clientX, e.clientY);
  } else {
    hover = scalarIndexAt(e.clientY);
  }
  dirty = true;
});
cv.addEventListener("wheel", (e) => {
  e.preventDefault();
  const mx = e.clientX;
  const my = e.clientY;
  if (activeLayer.kind === "matrix"){
    const beforeX = (mx - panX) / scale;
    const beforeY = (my - panY) / scale;
    const factor = e.deltaY > 0 ? 0.92 : 1.08;
    scale = Math.max(0.05, Math.min(50, scale * factor));
    panX = mx - beforeX * scale;
    panY = my - beforeY * scale;
  } else {
    const before = (my - panY) / scale;
    const factor = e.deltaY > 0 ? 0.92 : 1.08;
    scale = Math.max(0.25, Math.min(24, scale * factor));
    panY = my - before * scale;
  }
  dirty = true;
}, { passive: false });

function loop(){
  if (dirty){
    render();
    dirty = false;
  }
  requestAnimationFrame(loop);
}

buildTabs();
updateHeader();
resize();
loop();
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else pick_default_checkpoint()
    output = args.output.resolve()
    payload = build_payload(checkpoint, args.title)
    output.write_text(
        HTML_TEMPLATE.replace("__TITLE__", args.title).replace("__DATA__", json.dumps(payload, separators=(",", ":"))),
        encoding="utf-8",
    )
    print(
        f"Saved {output} from {checkpoint.name} "
        f"with {len(payload['layers'])} layers: "
        + ", ".join(layer["id"] for layer in payload["layers"])
    )


if __name__ == "__main__":
    main()
