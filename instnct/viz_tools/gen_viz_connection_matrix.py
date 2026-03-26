"""Generate a minimal binary connection-matrix HTML visualization."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "network_connection_matrix.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a minimal HxH connection-presence matrix from a checkpoint. "
            "Rows are source neurons, columns are target neurons, and the "
            "diagonal is blocked."
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
        default="SWG Connection Matrix",
        help="HTML title/header label.",
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


def load_connection_presence(path: Path) -> tuple[np.ndarray, np.ndarray, int, dict[str, int]]:
    with np.load(path, allow_pickle=True) as data:
        files = set(data.files)
        if {"rows", "cols"}.issubset(files):
            rows = np.asarray(data["rows"], dtype=np.int32)
            cols = np.asarray(data["cols"], dtype=np.int32)
            if "vals" in files:
                vals = np.asarray(data["vals"])
                keep = vals != 0
                rows = rows[keep]
                cols = cols[keep]
            if "H" in files:
                hidden_size = int(data["H"])
            elif "theta" in files:
                hidden_size = int(np.asarray(data["theta"]).shape[0])
            elif rows.size:
                hidden_size = int(max(rows.max(), cols.max()) + 1)
            else:
                raise ValueError(f"Cannot infer hidden size from {path}")
        elif "mmag" in files:
            matrix = np.asarray(data["mmag"])
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"Expected square mmag matrix in {path}")
            hidden_size = int(matrix.shape[0])
            rows, cols = np.where(matrix != 0)
            rows = rows.astype(np.int32, copy=False)
            cols = cols.astype(np.int32, copy=False)
        else:
            raise ValueError(
                f"Unsupported checkpoint format in {path}. Need rows/cols[/vals] or mmag."
            )

    raw_edges = int(rows.size)
    if rows.shape != cols.shape:
        raise ValueError(f"Mismatched rows/cols arrays in {path}")

    non_diag = rows != cols
    blocked_self_edges = int((~non_diag).sum())
    rows = rows[non_diag]
    cols = cols[non_diag]

    if hidden_size <= 0:
        raise ValueError(f"Invalid hidden size {hidden_size} in {path}")
    if rows.size and (rows.max() >= hidden_size or cols.max() >= hidden_size):
        raise ValueError(f"Edge index exceeds hidden size in {path}")

    if rows.size:
        order = np.lexsort((cols, rows))
        rows = rows[order]
        cols = cols[order]

    meta = {
        "raw_edges": raw_edges,
        "blocked_self_edges": blocked_self_edges,
    }
    return rows, cols, hidden_size, meta


def compute_stats(rows: np.ndarray, cols: np.ndarray, hidden_size: int) -> dict[str, object]:
    edge_count = int(rows.size)
    out_deg = np.bincount(rows, minlength=hidden_size).astype(np.int32)
    in_deg = np.bincount(cols, minlength=hidden_size).astype(np.int32)
    isolated = int(((out_deg == 0) & (in_deg == 0)).sum())
    sink = int(((out_deg == 0) & (in_deg > 0)).sum())
    source_only = int(((out_deg > 0) & (in_deg == 0)).sum())

    edge_set = set(zip(rows.tolist(), cols.tolist()))
    reciprocal_pairs = sum(1 for r, c in edge_set if r < c and (c, r) in edge_set)

    assert int(out_deg.sum()) == edge_count
    assert int(in_deg.sum()) == edge_count

    return {
        "hidden_size": hidden_size,
        "edge_count": edge_count,
        "density": (edge_count / (hidden_size * (hidden_size - 1))) if hidden_size > 1 else 0.0,
        "reciprocal_pairs": reciprocal_pairs,
        "sink_count": sink,
        "source_only_count": source_only,
        "isolated_count": isolated,
        "out_degree": out_deg.tolist(),
        "in_degree": in_deg.tolist(),
        "max_out_degree": int(out_deg.max()) if hidden_size else 0,
        "max_in_degree": int(in_deg.max()) if hidden_size else 0,
    }


def build_payload(path: Path, title: str) -> dict[str, object]:
    rows, cols, hidden_size, meta = load_connection_presence(path)
    stats = compute_stats(rows, cols, hidden_size)
    return {
        "title": title,
        "checkpoint_name": path.name,
        "checkpoint_path": str(path),
        "layer_name": "connection_presence",
        "description": "Binary hidden-to-hidden connectivity. Row=source, column=target.",
        "rows": rows.tolist(),
        "cols": cols.tolist(),
        **stats,
        **meta,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
@import url("https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Outfit:wght@200;400;700&display=swap");
:root{--bg:#04060a;--panel:rgba(8,14,22,0.92);--border:rgba(0,255,136,0.15);--accent:#00ff88;--warn:#ff8844;--block:#ff3355;--text:#b8c4d0;--dim:#445566;--grid:rgba(255,255,255,0.05)}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:"JetBrains Mono",monospace;overflow:hidden;cursor:crosshair}
canvas{display:block}
.panel{position:fixed;background:var(--panel);border:1px solid var(--border);backdrop-filter:blur(12px);padding:14px;z-index:10;border-radius:2px}
#header{top:12px;left:12px;min-width:280px}
#header h1{font-family:"Outfit",sans-serif;font-weight:200;font-size:20px;color:var(--accent);letter-spacing:3px;text-transform:uppercase;text-shadow:0 0 20px rgba(0,255,136,0.15)}
#header .sub{font-size:9px;color:var(--dim);letter-spacing:2px;margin:4px 0 10px}
.sr{display:flex;justify-content:space-between;padding:2px 0;font-size:10px;border-bottom:1px solid rgba(255,255,255,0.03)}
.sr .l{color:var(--dim)} .sr .v{font-weight:600}
.sr .v.a{color:var(--accent)} .sr .v.w{color:var(--warn)} .sr .v.b{color:var(--block)}
#hover{bottom:12px;left:12px;min-width:250px;font-size:10px;transition:opacity 0.15s}
#hover.empty{opacity:0.35}
#hover .nid{font-family:"Outfit";font-size:16px;color:var(--accent);font-weight:700}
#controls{top:12px;right:12px;display:flex;flex-direction:column;gap:5px}
.btn{background:transparent;color:var(--accent);border:1px solid var(--border);padding:5px 12px;font-family:"JetBrains Mono",monospace;font-size:9px;letter-spacing:1px;cursor:pointer;transition:all 0.15s;text-transform:uppercase}
.btn:hover{background:rgba(0,255,136,0.1);border-color:var(--accent)}
.btn.active{background:rgba(0,255,136,0.15);border-color:var(--accent);color:#fff}
.scanline{position:fixed;top:0;left:0;right:0;height:100vh;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,255,136,0.006) 2px,rgba(0,255,136,0.006) 4px);pointer-events:none;z-index:100}
.legend{position:fixed;bottom:12px;right:12px;z-index:10;font-size:9px;color:var(--dim);background:var(--panel);border:1px solid var(--border);padding:10px;border-radius:2px}
.legend .dot{display:inline-block;width:8px;height:8px;margin-right:5px}
</style>
</head>
<body>
<div class="scanline"></div>
<div class="panel" id="header">
  <h1>__TITLE__</h1>
  <div class="sub">row = source neuron, column = target neuron, diagonal blocked</div>
  <div class="sr"><span class="l">checkpoint</span><span class="v" id="s-ckpt">-</span></div>
  <div class="sr"><span class="l">hidden size</span><span class="v" id="s-h">-</span></div>
  <div class="sr"><span class="l">visible edges</span><span class="v a" id="s-e">-</span></div>
  <div class="sr"><span class="l">density</span><span class="v" id="s-d">-</span></div>
  <div class="sr"><span class="l">reciprocal pairs</span><span class="v w" id="s-r">-</span></div>
  <div class="sr"><span class="l">sink neurons</span><span class="v" id="s-sink">-</span></div>
  <div class="sr"><span class="l">source-only neurons</span><span class="v" id="s-src">-</span></div>
  <div class="sr"><span class="l">isolated neurons</span><span class="v" id="s-iso">-</span></div>
  <div class="sr"><span class="l">blocked self-edges</span><span class="v b" id="s-self">-</span></div>
</div>
<div class="panel empty" id="hover">
  <div class="nid" id="h-title">hover a cell</div>
  <div class="sr"><span class="l">source row</span><span class="v" id="h-src">-</span></div>
  <div class="sr"><span class="l">target col</span><span class="v" id="h-tgt">-</span></div>
  <div class="sr"><span class="l">state</span><span class="v" id="h-state">-</span></div>
  <div class="sr"><span class="l">source outgoing</span><span class="v" id="h-out">-</span></div>
  <div class="sr"><span class="l">target incoming</span><span class="v" id="h-in">-</span></div>
</div>
<div class="panel" id="controls">
  <button class="btn active" id="btn-grid" onclick="toggleGrid()">grid on</button>
  <button class="btn active" id="btn-hl" onclick="toggleHoverGuides()">guides on</button>
  <button class="btn" onclick="resetView()">reset view</button>
</div>
<div class="legend">
  <div><span class="dot" style="background:#00ff88"></span>connection present</div>
  <div><span class="dot" style="background:#ff3355"></span>blocked diagonal</div>
  <div><span class="dot" style="background:#ff8844"></span>hover row / column</div>
</div>
<canvas id="c"></canvas>
<script>
const D = __DATA__;
const cv = document.getElementById("c");
const cx = cv.getContext("2d");
let W=0, H=0;
function resize(){ W = cv.width = innerWidth; H = cv.height = innerHeight; fitView(); dirty = true; }
window.addEventListener("resize", resize);

const N = D.hidden_size;
const rows = D.rows;
const cols = D.cols;
const outDeg = D.out_degree;
const inDeg = D.in_degree;
const edgeSet = new Set(rows.map((r, i) => `${r}:${cols[i]}`));

document.getElementById("s-ckpt").textContent = D.checkpoint_name;
document.getElementById("s-h").textContent = D.hidden_size;
document.getElementById("s-e").textContent = D.edge_count;
document.getElementById("s-d").textContent = (D.density * 100).toFixed(3) + "%";
document.getElementById("s-r").textContent = D.reciprocal_pairs;
document.getElementById("s-sink").textContent = D.sink_count;
document.getElementById("s-src").textContent = D.source_only_count;
document.getElementById("s-iso").textContent = D.isolated_count;
document.getElementById("s-self").textContent = D.blocked_self_edges;

let scale = 1;
let panX = 0;
let panY = 0;
let dragging = false;
let dragX = 0;
let dragY = 0;
let hover = null;
let showGrid = true;
let showGuides = true;
let dirty = true;

function fitView(){
  const sidePad = 340;
  const topPad = 24;
  const matrixSpaceW = Math.max(220, W - sidePad - 24);
  const matrixSpaceH = Math.max(220, H - topPad - 24);
  scale = Math.min(matrixSpaceW / N, matrixSpaceH / N);
  panX = sidePad + Math.max(0, (matrixSpaceW - N * scale) / 2);
  panY = topPad + Math.max(0, (matrixSpaceH - N * scale) / 2);
}

function resetView(){ fitView(); dirty = true; }
function toggleGrid(){
  showGrid = !showGrid;
  const btn = document.getElementById("btn-grid");
  btn.textContent = showGrid ? "grid on" : "grid off";
  btn.classList.toggle("active", showGrid);
  dirty = true;
}
function toggleHoverGuides(){
  showGuides = !showGuides;
  const btn = document.getElementById("btn-hl");
  btn.textContent = showGuides ? "guides on" : "guides off";
  btn.classList.toggle("active", showGuides);
  dirty = true;
}

function screenToCell(x, y){
  const col = Math.floor((x - panX) / scale);
  const row = Math.floor((y - panY) / scale);
  if (row < 0 || row >= N || col < 0 || col >= N) return null;
  return { row, col };
}

function updateHover(cell){
  hover = cell;
  const panel = document.getElementById("hover");
  if (!cell){
    panel.classList.add("empty");
    document.getElementById("h-title").textContent = "hover a cell";
    ["h-src","h-tgt","h-state","h-out","h-in"].forEach(id => document.getElementById(id).textContent = "-");
    dirty = true;
    return;
  }
  panel.classList.remove("empty");
  const state = cell.row === cell.col ? "blocked diagonal" : (edgeSet.has(`${cell.row}:${cell.col}`) ? "connection present" : "no connection");
  document.getElementById("h-title").textContent = `cell ${cell.row}, ${cell.col}`;
  document.getElementById("h-src").textContent = cell.row;
  document.getElementById("h-tgt").textContent = cell.col;
  document.getElementById("h-state").textContent = state;
  document.getElementById("h-out").textContent = outDeg[cell.row];
  document.getElementById("h-in").textContent = inDeg[cell.col];
  dirty = true;
}

cv.addEventListener("mousedown", (e) => {
  dragging = true;
  dragX = e.clientX - panX;
  dragY = e.clientY - panY;
});
window.addEventListener("mouseup", () => { dragging = false; });
window.addEventListener("mousemove", (e) => {
  if (dragging){
    panX = e.clientX - dragX;
    panY = e.clientY - dragY;
    dirty = true;
    return;
  }
  updateHover(screenToCell(e.clientX, e.clientY));
});

cv.addEventListener("wheel", (e) => {
  e.preventDefault();
  const mx = e.clientX;
  const my = e.clientY;
  const beforeX = (mx - panX) / scale;
  const beforeY = (my - panY) / scale;
  const factor = e.deltaY > 0 ? 0.92 : 1.08;
  scale = Math.max(0.05, Math.min(50, scale * factor));
  panX = mx - beforeX * scale;
  panY = my - beforeY * scale;
  dirty = true;
}, { passive: false });

function render(){
  cx.clearRect(0, 0, W, H);

  cx.save();
  cx.fillStyle = "#05080e";
  cx.fillRect(panX, panY, N * scale, N * scale);
  cx.restore();

  cx.save();
  cx.translate(panX, panY);
  cx.scale(scale, scale);

  if (showGuides && hover){
    cx.fillStyle = "rgba(255,136,68,0.10)";
    cx.fillRect(0, hover.row, N, 1);
    cx.fillRect(hover.col, 0, 1, N);
  }

  if (scale >= 6 && showGrid){
    cx.beginPath();
    cx.strokeStyle = "rgba(255,255,255,0.05)";
    cx.lineWidth = 1 / scale;
    for (let i = 0; i <= N; i += 1){
      cx.moveTo(i, 0); cx.lineTo(i, N);
      cx.moveTo(0, i); cx.lineTo(N, i);
    }
    cx.stroke();
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
  for (let i = 0; i < rows.length; i += 1){
    cx.fillRect(cols[i], rows[i], 1, 1);
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
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = build_payload(checkpoint, args.title)
    compact = json.dumps(payload, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__TITLE__", args.title).replace("__DATA__", compact)
    output.write_text(html, encoding="utf-8")

    print(
        f"Saved {output} from {checkpoint.name} "
        f"(H={payload['hidden_size']}, edges={payload['edge_count']}, "
        f"blocked_self_edges={payload['blocked_self_edges']})"
    )


if __name__ == "__main__":
    main()
