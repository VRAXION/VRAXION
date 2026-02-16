"""
Minimal live control panel for Diamond Code training.

Zero dependencies beyond Python stdlib. Reads/writes logs/swarm/controls.json
which the training loop polls every step.

Usage:
    python tools/control_panel.py                          # default port 7777
    python tools/control_panel.py --port 8888              # custom port
    python tools/control_panel.py --controls path/to.json  # custom controls file
"""

import io
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import argparse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs
from urllib.request import Request, urlopen
from urllib.error import URLError
from PIL import Image

# Bumped whenever routes/logic change so stale processes are detectable.
_PANEL_VERSION = "2026.02.14a"

DEFAULT_CONTROLS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "swarm", "controls.json"
)

LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "swarm", "current.log"
)

INFLUX_URL = "http://localhost:8086"
INFLUX_ORG = "vraxion"
INFLUX_BUCKET = "diamond"
_influx_token = None

# Viridis colormap LUT (256 entries: dark purple -> blue -> green -> yellow)
_VIRIDIS = [
    (68,1,84),(68,2,86),(69,4,87),(69,5,89),(70,7,90),(70,8,92),(70,10,93),(70,11,94),
    (71,13,96),(71,14,97),(71,16,99),(71,17,100),(71,19,101),(72,20,103),(72,22,104),
    (72,23,105),(72,24,106),(72,26,108),(72,27,109),(72,28,110),(72,29,111),(72,31,112),
    (72,32,113),(72,33,115),(72,35,116),(72,36,117),(72,37,118),(72,38,119),(72,40,120),
    (72,41,121),(71,42,122),(71,44,122),(71,45,123),(71,46,124),(71,47,125),(70,48,126),
    (70,50,126),(70,51,127),(69,52,128),(69,53,129),(69,55,129),(68,56,130),(68,57,131),
    (67,58,131),(67,60,132),(66,61,132),(66,62,133),(65,63,133),(65,64,134),(64,66,134),
    (64,67,135),(63,68,135),(63,69,136),(62,70,136),(62,72,136),(61,73,137),(61,74,137),
    (60,75,137),(60,76,138),(59,77,138),(59,78,138),(58,80,138),(58,81,138),(57,82,139),
    (57,83,139),(56,84,139),(56,85,139),(55,86,139),(55,87,140),(54,88,140),(54,89,140),
    (53,90,140),(53,91,140),(52,92,140),(52,93,140),(51,94,140),(51,95,140),(51,96,141),
    (50,97,141),(50,98,141),(49,99,141),(49,100,141),(49,101,141),(48,102,141),(48,103,141),
    (47,104,141),(47,105,141),(46,106,141),(46,107,141),(46,108,141),(45,109,140),(45,110,140),
    (45,111,140),(44,112,140),(44,113,140),(44,114,140),(43,115,140),(43,116,140),(42,117,140),
    (42,118,140),(42,119,139),(41,120,139),(41,121,139),(41,122,139),(40,123,139),(40,124,138),
    (40,125,138),(39,126,138),(39,127,138),(39,128,137),(38,129,137),(38,130,137),(38,131,136),
    (37,132,136),(37,133,136),(37,134,135),(36,135,135),(36,136,135),(36,137,134),(36,138,134),
    (35,139,133),(35,140,133),(35,141,133),(35,142,132),(35,143,132),(34,144,131),(34,145,131),
    (34,146,130),(34,147,130),(34,148,129),(34,149,129),(34,150,128),(33,151,128),(33,152,127),
    (33,153,127),(33,154,126),(33,155,126),(33,156,125),(33,157,124),(34,158,124),(34,159,123),
    (34,160,123),(34,161,122),(34,162,121),(35,163,121),(35,164,120),(35,165,119),(36,166,119),
    (36,167,118),(37,168,117),(37,169,117),(38,170,116),(38,171,115),(39,172,114),(40,173,114),
    (40,174,113),(41,175,112),(42,176,111),(43,177,111),(43,178,110),(44,179,109),(45,180,108),
    (46,181,107),(47,182,106),(48,183,106),(49,184,105),(50,185,104),(51,186,103),(52,187,102),
    (53,188,101),(54,189,100),(56,190,99),(57,191,98),(58,192,97),(59,193,96),(61,194,95),
    (62,195,94),(63,196,93),(65,196,92),(66,197,91),(68,198,90),(69,199,89),(71,200,87),
    (73,201,86),(74,202,85),(76,203,84),(78,203,83),(79,204,82),(81,205,80),(83,206,79),
    (85,206,78),(87,207,77),(89,208,75),(91,208,74),(93,209,73),(95,210,71),(97,210,70),
    (99,211,69),(101,212,67),(103,212,66),(105,213,64),(108,213,63),(110,214,62),(112,214,60),
    (114,215,59),(116,215,57),(119,216,56),(121,216,55),(123,217,53),(125,217,52),(128,218,50),
    (130,218,49),(132,219,47),(135,219,46),(137,219,44),(139,220,43),(142,220,41),(144,221,40),
    (146,221,38),(149,221,37),(151,222,35),(154,222,34),(156,222,32),(158,223,31),(161,223,29),
    (163,223,28),(166,224,26),(168,224,25),(171,224,23),(173,225,22),(175,225,21),(178,225,19),
    (180,226,18),(183,226,17),(185,226,16),(188,227,15),(190,227,14),(193,227,13),(195,227,13),
    (197,228,12),(200,228,12),(202,228,12),(205,228,12),(207,229,12),(209,229,13),(212,229,13),
    (214,229,14),(216,230,15),(218,230,16),(221,230,17),(223,230,18),(225,230,20),(227,231,21),
    (229,231,23),(231,231,25),(233,231,27),(235,232,29),(237,232,31),(238,232,33),(240,232,36),
    (242,232,38),(243,233,40),(245,233,43),(246,233,46),(248,234,48),(249,234,51),(250,235,54),
    (252,235,57),(253,236,60),(253,236,63),(254,237,66),(254,237,69),(254,238,73),(254,239,76),
    (254,239,79),(254,240,83),(254,241,86),(253,241,90),(253,242,93),(253,243,97),(253,243,100),
    (253,244,104),(252,245,107),(252,245,111),(252,246,115),(251,247,118),(251,247,122),
    (251,248,126),(250,249,129),(250,249,133),(250,250,137),(249,251,141),(249,251,145),
    (249,252,148),(248,252,152),(248,253,156),(247,253,160),(247,254,164),(247,254,168),
    (246,254,172),(246,255,176),(246,255,180),(245,255,184),(245,255,188),(245,255,192),
    (244,255,196),(244,255,200),(244,255,204),(244,255,208),(244,255,212),(243,255,216),
    (243,255,220),(243,255,224),(243,255,228),(243,255,232),(243,255,236),(243,255,240),
    (243,255,244),(243,255,248),(243,255,252),(243,255,255),(253,231,37),
]

# Diamond colormap LUT: dark -> purple -> magenta -> orange -> yellow -> white
# Matches Grafana threshold palette for visual consistency
def _build_diamond_lut():
    anchors = [
        (0.00, (10, 10, 26)),    # near-black
        (0.10, (26, 10, 58)),    # deep indigo
        (0.20, (45, 27, 105)),   # dark purple
        (0.30, (74, 20, 134)),   # purple
        (0.40, (106, 27, 154)),  # violet
        (0.50, (140, 41, 129)),  # magenta
        (0.60, (181, 54, 122)),  # hot pink
        (0.70, (217, 78, 106)),  # coral
        (0.80, (232, 118, 75)),  # orange
        (0.88, (240, 160, 48)),  # amber
        (0.94, (245, 197, 24)),  # yellow
        (0.97, (253, 231, 37)),  # bright yellow
        (1.00, (255, 255, 255)), # white (saturated)
    ]
    lut = []
    for i in range(256):
        t = i / 255.0
        # Find surrounding anchors
        for k in range(len(anchors) - 1):
            if anchors[k][0] <= t <= anchors[k + 1][0]:
                t0, c0 = anchors[k]
                t1, c1 = anchors[k + 1]
                f = (t - t0) / (t1 - t0) if t1 > t0 else 0
                r = int(c0[0] + f * (c1[0] - c0[0]))
                g = int(c0[1] + f * (c1[1] - c0[1]))
                b = int(c0[2] + f * (c1[2] - c0[2]))
                lut.append((r, g, b))
                break
        else:
            lut.append(anchors[-1][1])
    return lut

_DIAMOND = _build_diamond_lut()


def _load_influx_token():
    """Load InfluxDB token from .influx_token file. Returns None on failure."""
    token_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".influx_token"
    )
    try:
        with open(token_file) as f:
            return f.read().strip()
    except Exception:
        return None


def influx_log_control_change(key, old_value, new_value):
    """Write a control_change point to InfluxDB via line protocol HTTP API.
    Shows as annotation in Grafana. Silently fails if InfluxDB is down."""
    global _influx_token
    if _influx_token is None:
        _influx_token = _load_influx_token() or ""
    if not _influx_token:
        return
    # Escape tag values (spaces, commas, equals)
    key_esc = str(key).replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")
    # Line protocol: measurement,tags fields timestamp
    ts_ns = int(time.time() * 1e9)
    line = (
        f'control_change,key={key_esc} '
        f'old_value="{old_value}",new_value="{new_value}",description="{key}: {old_value} -> {new_value}" '
        f'{ts_ns}'
    )
    url = f"{INFLUX_URL}/api/v2/write?org={INFLUX_ORG}&bucket={INFLUX_BUCKET}&precision=ns"
    try:
        req = Request(url, data=line.encode('utf-8'), method='POST')
        req.add_header('Authorization', f'Token {_influx_token}')
        req.add_header('Content-Type', 'text/plain; charset=utf-8')
        urlopen(req, timeout=2)
    except Exception:
        pass  # Don't break the control panel if InfluxDB is down


def influx_log_control_value(key, value):
    """Write a control_value point for time-series tracking in Grafana."""
    global _influx_token
    if _influx_token is None:
        _influx_token = _load_influx_token() or ""
    if not _influx_token:
        return
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return  # Only log numeric values as time-series
    key_esc = str(key).replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")
    ts_ns = int(time.time() * 1e9)
    line = f'control_value,key={key_esc} value={numeric} {ts_ns}'
    url = f"{INFLUX_URL}/api/v2/write?org={INFLUX_ORG}&bucket={INFLUX_BUCKET}&precision=ns"
    try:
        req = Request(url, data=line.encode('utf-8'), method='POST')
        req.add_header('Authorization', f'Token {_influx_token}')
        req.add_header('Content-Type', 'text/plain; charset=utf-8')
        urlopen(req, timeout=2)
    except Exception:
        pass

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Diamond Code — Live Controls</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Consolas', 'Courier New', monospace;
    background: #0a0a0f;
    color: #c8c8d0;
    min-height: 100vh;
    padding: 20px;
  }
  h1 {
    color: #7b68ee;
    font-size: 18px;
    margin-bottom: 4px;
    letter-spacing: 2px;
  }
  .subtitle {
    color: #555;
    font-size: 12px;
    margin-bottom: 20px;
  }
  .status-bar {
    background: #12121a;
    border: 1px solid #222;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 16px;
    font-size: 13px;
  }
  .status-bar .live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #2ecc71;
    border-radius: 50%;
    margin-right: 8px;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }
  .status-bar .metric { color: #7b68ee; font-weight: bold; }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-bottom: 16px;
  }
  .card {
    background: #12121a;
    border: 1px solid #222;
    border-radius: 6px;
    padding: 14px 16px;
  }
  .card label {
    display: block;
    color: #888;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
  }
  .card .row {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  input[type="number"], input[type="text"] {
    background: #1a1a25;
    border: 1px solid #333;
    color: #e0e0e0;
    font-family: inherit;
    font-size: 16px;
    padding: 6px 10px;
    border-radius: 4px;
    width: 120px;
    text-align: right;
  }
  input:focus { outline: none; border-color: #7b68ee; }
  button {
    background: #7b68ee;
    color: #fff;
    border: none;
    padding: 7px 16px;
    border-radius: 4px;
    font-family: inherit;
    font-size: 13px;
    cursor: pointer;
    transition: background 0.15s;
  }
  button:hover { background: #6a5acd; }
  button:active { background: #5548b0; }
  button.danger { background: #c0392b; }
  button.danger:hover { background: #a93226; }
  button.secondary { background: #333; color: #aaa; }
  button.secondary:hover { background: #444; color: #ccc; }

  /* Dataset Mixer */
  .mix-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:10px; }
  .mix-header .total { font-size:14px; font-weight:bold; }
  .mix-header .total.warn { color:#f1c40f; }
  .mix-bar-wrap { height:16px; background:#1a1a25; border-radius:4px; overflow:hidden; display:flex; margin-bottom:14px; border:1px solid #333; }
  .mix-bar-seg { height:100%; transition:width 0.3s; position:relative; min-width:0; }
  .mix-bar-seg span { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); font-size:9px; font-weight:bold; color:#000; white-space:nowrap; overflow:hidden; }
  .dataset-row {
    background:#0d0d14; border-radius:6px; padding:10px 12px; margin-bottom:6px;
    border:1px solid #1a1a25; transition:border-color 0.15s;
  }
  .dataset-row:hover { border-color:#333; }
  .dataset-row.active { border-color:#2ecc7133; }
  .dataset-top { display:flex; align-items:center; gap:8px; margin-bottom:4px; }
  .tier-badge {
    font-size:10px; font-weight:bold; padding:2px 8px; border-radius:10px;
    color:#000; letter-spacing:0.5px; flex-shrink:0;
  }
  .dataset-name { font-size:14px; font-weight:bold; letter-spacing:0.5px; flex:1; }
  .dataset-name.on { color:#2ecc71; }
  .dataset-name.off { color:#555; }
  .weight-val { font-size:14px; font-weight:bold; color:#7b68ee; font-family:inherit; min-width:36px; text-align:right; }
  .dataset-notes { font-size:11px; color:#555; font-style:italic; margin-bottom:6px; line-height:1.3; }
  .weight-bar-track { height:4px; background:#1a1a25; border-radius:2px; margin-bottom:6px; overflow:hidden; }
  .weight-bar-fill { height:100%; border-radius:2px; transition:width 0.2s, background 0.2s; }
  .dataset-controls { display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
  .dataset-controls input[type="range"] {
    -webkit-appearance:none; appearance:none; flex:1; min-width:120px; height:6px;
    background:#1a1a25; border-radius:3px; outline:none; border:1px solid #333;
  }
  .dataset-controls input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance:none; appearance:none; width:16px; height:16px;
    background:#7b68ee; border-radius:50%; cursor:pointer; border:2px solid #5548b0;
  }
  .dataset-controls input[type="range"]::-moz-range-thumb {
    width:14px; height:14px; background:#7b68ee; border-radius:50%; cursor:pointer; border:2px solid #5548b0;
  }
  .weight-presets { display:flex; gap:3px; }
  .weight-presets button {
    font-size:11px; padding:3px 8px; background:#1a1a25; border:1px solid #333;
    color:#888; border-radius:3px; cursor:pointer; transition:all 0.1s;
  }
  .weight-presets button:hover { border-color:#7b68ee; color:#7b68ee; }
  .weight-presets button.active { background:#7b68ee; color:#fff; border-color:#7b68ee; }
  .btn-off {
    font-size:11px; padding:3px 10px; background:#1a1a25; border:1px solid #c0392b;
    color:#c0392b; border-radius:3px; cursor:pointer; font-weight:bold;
  }
  .btn-off:hover { background:#c0392b; color:#fff; }
  .btn-off.is-off { background:#c0392b; color:#fff; }
  .btn-norm {
    font-size:12px; padding:5px 14px; background:#1a1a25; border:1px solid #7b68ee;
    color:#7b68ee; border-radius:4px; cursor:pointer; font-weight:bold;
  }
  .btn-norm:hover { background:#7b68ee; color:#fff; }

  .preset-row {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }
  .preset-row button {
    font-size: 12px;
    padding: 5px 12px;
    background: #1a1a25;
    border: 1px solid #333;
    color: #aaa;
  }
  .preset-row button:hover { border-color: #7b68ee; color: #7b68ee; }
  .toast {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: #2ecc71;
    color: #000;
    padding: 10px 20px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: bold;
    opacity: 0;
    transition: opacity 0.3s;
    pointer-events: none;
  }
  .toast.show { opacity: 1; }
  .toast.error { background: #c0392b; color: #fff; }
  .log-box {
    background: #12121a;
    border: 1px solid #222;
    border-radius: 6px;
    padding: 12px 16px;
    margin-top: 12px;
    font-size: 12px;
    max-height: 200px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
    color: #888;
  }
  .full-width { grid-column: 1 / -1; }
</style>
</head>
<body>

<h1>DIAMOND CODE</h1>
<p class="subtitle">Live Training Controls — writes to controls.json every click</p>

<div class="status-bar" id="statusBar">
  <span class="live-dot"></span>
  Loading...
</div>

<div class="grid">

  <!-- Learning Rate -->
  <div class="card">
    <label>Learning Rate</label>
    <div class="row">
      <input type="number" id="lr" step="0.0001" min="0" value="0.001">
      <button onclick="setVal('lr', parseFloat(document.getElementById('lr').value))">Set</button>
    </div>
    <div class="preset-row" style="margin-top:8px">
      <button onclick="setVal('lr', 0.01)">0.01</button>
      <button onclick="setVal('lr', 0.001)">1e-3</button>
      <button onclick="setVal('lr', 0.0003)">3e-4</button>
      <button onclick="setVal('lr', 0.0001)">1e-4</button>
      <button onclick="setVal('lr', 0.00001)">1e-5</button>
    </div>
  </div>

  <!-- Think Ticks + Effort Mode -->
  <div class="card">
    <label>Think Ticks <span id="tt_mode_badge" style="font-size:11px;padding:2px 6px;border-radius:4px;margin-left:8px"></span></label>
    <div id="tt_manual_section">
      <div class="row">
        <input type="number" id="think_ticks" step="1" min="0" max="20" value="0">
        <button onclick="setVal('think_ticks', parseInt(document.getElementById('think_ticks').value))">Set</button>
      </div>
      <div class="preset-row" style="margin-top:8px">
        <button onclick="setVal('think_ticks', 0)">0</button>
        <button onclick="setVal('think_ticks', 1)">1</button>
        <button onclick="setVal('think_ticks', 2)">2</button>
        <button onclick="setVal('think_ticks', 3)">3</button>
        <button onclick="setVal('think_ticks', 4)">4</button>
        <button onclick="setVal('think_ticks', 5)">5</button>
        <button onclick="setVal('think_ticks', 8)">8</button>
        <button onclick="setVal('think_ticks', 12)">12</button>
      </div>
    </div>

    <div style="margin-top:12px;padding-top:10px;border-top:1px solid #333">
      <label style="font-size:13px">Effort Mode</label>
      <div id="effort_btns" style="display:flex;gap:6px;flex-wrap:wrap;margin-top:6px">
        <button onclick="setEffort('fast')" data-mode="fast"
          style="flex:1;padding:8px 4px;border-radius:6px;font-weight:bold;font-size:12px;
                 border:2px solid #f44;background:#1a1a25;color:#f66;cursor:pointer">
          FAST</button>
        <button onclick="setEffort('medium')" data-mode="medium"
          style="flex:1;padding:8px 4px;border-radius:6px;font-weight:bold;font-size:12px;
                 border:2px solid #4f4;background:#1a1a25;color:#6f6;cursor:pointer">
          MEDIUM</button>
        <button onclick="setEffort('slow')" data-mode="slow"
          style="flex:1;padding:8px 4px;border-radius:6px;font-weight:bold;font-size:12px;
                 border:2px solid #44f;background:#1a1a25;color:#66f;cursor:pointer">
          SLOW</button>
        <button onclick="setEffort('random')" data-mode="random"
          style="flex:1;padding:8px 4px;border-radius:6px;font-weight:bold;font-size:12px;
                 border:2px solid #888;background:#1a1a25;color:#ccc;cursor:pointer">
          RANDOM</button>
      </div>
      <div id="effort_status" style="margin-top:6px;font-size:12px;color:#888"></div>
      <div style="margin-top:6px;font-size:11px;color:#555">
        Weights: primary 50% | adjacent 30% | far 20% (read+write)
      </div>
    </div>
  </div>

  <!-- Checkpoint Every -->
  <div class="card">
    <label>Checkpoint Every N Steps</label>
    <div class="row">
      <input type="number" id="checkpoint_every" step="50" min="10" value="500">
      <button onclick="setVal('checkpoint_every', parseInt(document.getElementById('checkpoint_every').value))">Set</button>
    </div>
    <div class="preset-row" style="margin-top:8px">
      <button onclick="setVal('checkpoint_every', 50)">50</button>
      <button onclick="setVal('checkpoint_every', 100)">100</button>
      <button onclick="setVal('checkpoint_every', 500)">500</button>
      <button onclick="setVal('checkpoint_every', 1000)">1K</button>
    </div>
  </div>

  <!-- Eval Every -->
  <div class="card">
    <label>Eval Every N Steps</label>
    <div class="row">
      <input type="number" id="eval_every" step="1" min="1" value="10">
      <button onclick="setVal('eval_every', parseInt(document.getElementById('eval_every').value))">Set</button>
    </div>
    <div class="preset-row" style="margin-top:8px">
      <button onclick="setVal('eval_every', 1)">1</button>
      <button onclick="setVal('eval_every', 5)">5</button>
      <button onclick="setVal('eval_every', 10)">10</button>
      <button onclick="setVal('eval_every', 50)">50</button>
    </div>
  </div>

  <!-- Data Mix -->
  <div class="card full-width">
    <label>Data Mix</label>
    <div id="mixHeader"></div>
    <div id="mixBar"></div>
    <div id="dataWeights"></div>
  </div>

  <!-- Being States -->
  <div class="card full-width">
    <label>Being States</label>
    <div id="beingStates"></div>
  </div>

  <!-- LCX Scratchpad (Zoom Levels or legacy RGB) -->
  <div class="card full-width">
    <label>LCX Scratchpad <span id="lcx_step" style="color:#666;font-size:12px"></span></label>
    <div id="lcx_active_ch" style="margin:6px 0;font-size:13px;color:#555">Waiting for LCX data...</div>
    <div id="lcx_norms" style="margin:6px 0"></div>
    <div id="lcx_canvases" style="display:flex;gap:12px;justify-content:center;margin-top:8px;flex-wrap:wrap"></div>
  </div>

</div>

<div class="log-box" id="logTail">Waiting for log data...</div>

<div class="toast" id="toast"></div>

<script>
const POLL_MS = 500;
let currentControls = {};
let metaCache = {};

// Fetch metadata once on load
(async function loadMeta() {
  try {
    const r = await fetch('/api/meta');
    metaCache = await r.json();
  } catch(e) { console.warn('Meta fetch failed:', e); }
})();

function toast(msg, isError) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast show' + (isError ? ' error' : '');
  setTimeout(() => t.className = 'toast', 1500);
}

async function setVal(key, value) {
  // Warn if setting think_ticks while effort_lock is active
  const eLock = currentControls.effort_lock || (currentControls.effort_mode ? 'random' : null);
  if (key === 'think_ticks' && eLock) {
    if (!confirm('Effort Mode is ' + eLock.toUpperCase() + ' — think_ticks is auto-controlled.\\nSet manually anyway? (will be overridden next step)')) return;
  }
  try {
    const r = await fetch('/api/set', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({key, value})
    });
    const d = await r.json();
    if (d.ok) {
      toast(key + ' = ' + value);
      poll();
    } else {
      toast('Error: ' + d.error, true);
    }
  } catch(e) { toast('Connection error', true); }
}

function setEffort(mode) { setVal('effort_lock', mode); }

function updateEffortLockUI(lock) {
  const badge = document.getElementById('tt_mode_badge');
  const manual = document.getElementById('tt_manual_section');
  const colors = {fast:'#f44', medium:'#4f4', slow:'#44f', random:'#888'};
  const labels = {fast:'FAST (R)', medium:'MEDIUM (G)', slow:'SLOW (B)', random:'RANDOM'};
  // Badge
  badge.textContent = (labels[lock] || lock).toUpperCase();
  badge.style.background = colors[lock] || '#333';
  badge.style.color = '#fff';
  // Dim manual TT section when not in a fixed mode
  manual.style.opacity = lock === 'random' ? '0.4' : '0.7';
  // Highlight active button
  const glows = {fast:'#f44', medium:'#4f4', slow:'#44f', random:'#888'};
  document.querySelectorAll('#effort_btns button').forEach(btn => {
    const m = btn.getAttribute('data-mode');
    btn.style.background = (m === lock) ? '#2a2a3a' : '#1a1a25';
    btn.style.boxShadow = (m === lock) ? '0 0 8px ' + (glows[m] || '#888') : 'none';
  });
  document.getElementById('effort_status').textContent = 'Active: ' + (labels[lock] || lock);
}

async function setDataWeight(name, value) {
  try {
    const r = await fetch('/api/set_data_weight', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name, value})
    });
    const d = await r.json();
    if (d.ok) {
      toast(name + ' = ' + value);
      poll();
    } else {
      toast('Error: ' + d.error, true);
    }
  } catch(e) { toast('Connection error', true); }
}

async function setBeingState(idx, state) {
  try {
    const r = await fetch('/api/set_being_state', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({idx, state})
    });
    const d = await r.json();
    if (d.ok) {
      toast('Being ' + idx + ' = ' + state);
      poll();
    } else {
      toast('Error: ' + d.error, true);
    }
  } catch(e) { toast('Connection error', true); }
}

const TIER_COLORS = ['#2ecc71','#3498db','#f1c40f','#e67e22','#e74c3c','#9b59b6'];
const MIX_COLORS = ['#2ecc71','#3498db','#f1c40f','#e67e22','#e74c3c','#9b59b6','#1abc9c','#e84393','#00cec9','#fd79a8','#6c5ce7'];

function renderDataWeights(weights) {
  const el = document.getElementById('dataWeights');
  const hdr = document.getElementById('mixHeader');
  const bar = document.getElementById('mixBar');
  if (!weights || Object.keys(weights).length === 0) {
    el.innerHTML = '<span style="color:#555">No traindat files</span>';
    hdr.innerHTML = ''; bar.innerHTML = '';
    return;
  }

  // Sort by tier (from meta), then alphabetically
  const sorted = Object.entries(weights).sort((a, b) => {
    const ta = (metaCache[a[0]] || {}).tier ?? 99;
    const tb = (metaCache[b[0]] || {}).tier ?? 99;
    if (ta !== tb) return ta - tb;
    return a[0].localeCompare(b[0]);
  });

  // Mix header: total + normalize button
  const total = sorted.reduce((s, [, w]) => s + w, 0);
  const totalFixed = total.toFixed(2);
  const warnCls = (Math.abs(total - 1.0) > 0.01 && total > 0) ? ' warn' : '';
  hdr.innerHTML = '<div class="mix-header">'
    + '<span class="total' + warnCls + '">Total: ' + totalFixed + '</span>'
    + '<button class="btn-norm" onclick="normalizeWeights()">NORMALIZE</button>'
    + '</div>';

  // Stacked mix bar
  if (total > 0) {
    let barHtml = '<div class="mix-bar-wrap">';
    let ci = 0;
    for (const [name, w] of sorted) {
      if (w <= 0) continue;
      const pct = (w / total * 100);
      const short = name.replace('.traindat', '');
      const color = MIX_COLORS[ci % MIX_COLORS.length];
      barHtml += '<div class="mix-bar-seg" style="width:' + pct.toFixed(1) + '%;background:' + color + '">';
      if (pct > 8) barHtml += '<span>' + short + ' ' + Math.round(pct) + '%</span>';
      barHtml += '</div>';
      ci++;
    }
    barHtml += '</div>';
    bar.innerHTML = barHtml;
  } else {
    bar.innerHTML = '<div class="mix-bar-wrap" style="justify-content:center;color:#555;font-size:11px;line-height:16px">No active datasets</div>';
  }

  // Per-dataset rows
  let html = '';
  let ci = 0;
  for (const [name, w] of sorted) {
    const short = name.replace('.traindat', '');
    const on = w > 0;
    const meta = metaCache[name] || {};
    const tier = meta.tier ?? '?';
    const notes = meta.notes || '';
    const tierColor = TIER_COLORS[Math.min(tier, TIER_COLORS.length - 1)] || '#555';
    const pct = total > 0 ? (w / total * 100) : 0;

    html += '<div class="dataset-row' + (on ? ' active' : '') + '">';

    // Top line: tier badge, name, weight value, OFF button
    html += '<div class="dataset-top">';
    html += '<span class="tier-badge" style="background:' + tierColor + '">T' + tier + '</span>';
    html += '<span class="dataset-name ' + (on ? 'on' : 'off') + '">' + short + '</span>';
    html += '<span class="weight-val">' + w.toFixed(2) + '</span>';
    html += '<button class="btn-off' + (on ? '' : ' is-off') + '" onclick="setDataWeight(\'' + name + '\', 0)">OFF</button>';
    html += '</div>';

    // Notes
    if (notes) {
      html += '<div class="dataset-notes">' + notes + '</div>';
    }

    // Weight bar
    html += '<div class="weight-bar-track"><div class="weight-bar-fill" style="width:' + Math.min(100, w * 100).toFixed(0) + '%;background:' + tierColor + '"></div></div>';

    // Slider + presets
    html += '<div class="dataset-controls">';
    html += '<input type="range" min="0" max="1" step="0.01" value="' + w + '" '
      + 'onchange="setDataWeight(\'' + name + '\', parseFloat(this.value))" '
      + 'oninput="this.closest(\'.dataset-row\').querySelector(\'.weight-val\').textContent=parseFloat(this.value).toFixed(2)">';
    html += '<div class="weight-presets">';
    for (const p of [0, 0.1, 0.3, 0.5, 0.7, 1.0]) {
      const isActive = Math.abs(w - p) < 0.005;
      html += '<button class="' + (isActive ? 'active' : '') + '" onclick="setDataWeight(\'' + name + '\', ' + p + ')">' + p + '</button>';
    }
    html += '</div></div>';

    html += '</div>';
    ci++;
  }
  el.innerHTML = html;
}

async function normalizeWeights() {
  const weights = currentControls.data_weights || {};
  const total = Object.values(weights).reduce((s, w) => s + w, 0);
  if (total <= 0) { toast('Nothing to normalize', true); return; }
  for (const [name, w] of Object.entries(weights)) {
    if (w > 0) {
      await setDataWeight(name, parseFloat((w / total).toFixed(3)));
    }
  }
  toast('Weights normalized to 1.0');
}

function renderBeingStates(states) {
  const el = document.getElementById('beingStates');
  if (!states || Object.keys(states).length === 0) {
    el.innerHTML = '<span style="color:#555">No beings configured</span>';
    return;
  }
  let html = '<div style="display:flex;flex-wrap:wrap;gap:8px">';
  for (const [idx, state] of Object.entries(states).sort((a,b) => parseInt(a[0]) - parseInt(b[0]))) {
    const colors = {active: '#2ecc71', frozen: '#3498db', null: '#555'};
    const c = colors[state] || '#555';
    html += `<div style="background:#1a1a25;border:1px solid ${c};border-radius:4px;padding:6px 10px;text-align:center">`;
    html += `<div style="color:${c};font-weight:bold;font-size:14px">B${idx}</div>`;
    html += `<div style="font-size:10px;color:${c};margin:2px 0">${state}</div>`;
    html += `<div style="display:flex;gap:3px;margin-top:4px">`;
    html += `<button style="font-size:10px;padding:2px 6px;background:${state==='active'?'#2ecc71':'#1a1a25'};border:1px solid #2ecc71;color:${state==='active'?'#000':'#2ecc71'}" onclick="setBeingState('${idx}','active')">A</button>`;
    html += `<button style="font-size:10px;padding:2px 6px;background:${state==='frozen'?'#3498db':'#1a1a25'};border:1px solid #3498db;color:${state==='frozen'?'#000':'#3498db'}" onclick="setBeingState('${idx}','frozen')">F</button>`;
    html += `<button style="font-size:10px;padding:2px 6px;background:${state==='null'?'#555':'#1a1a25'};border:1px solid #555;color:${state==='null'?'#000':'#555'}" onclick="setBeingState('${idx}','null')">N</button>`;
    html += `</div></div>`;
  }
  html += '</div>';
  el.innerHTML = html;
}

async function poll() {
  try {
    const r = await fetch('/api/controls');
    const d = await r.json();
    currentControls = d;

    // Update input values (only if not focused)
    for (const key of ['lr', 'think_ticks', 'checkpoint_every', 'eval_every']) {
      const el = document.getElementById(key);
      if (el && document.activeElement !== el && d[key] != null) {
        el.value = d[key];
      }
    }

    renderDataWeights(d.data_weights || {});
    renderBeingStates(d.being_states || {});

    // Sync effort lock UI
    const lock = d.effort_lock || (d.effort_mode ? 'random' : 'fast');
    updateEffortLockUI(lock);
  } catch(e) {}

  // Poll log tail
  try {
    const r = await fetch('/api/log_tail');
    const d = await r.json();
    if (d.lines) {
      const box = document.getElementById('logTail');
      box.textContent = d.lines;
      // Parse latest metrics for status bar
      const last = d.lines.trim().split('\n').filter(l => l.includes('bit_acc=')).pop();
      if (last) {
        const step = (last.match(/step (\d+)/) || [])[1] || '?';
        const loss = (last.match(/loss ([\d.]+)/) || [])[1] || '?';
        const bitAcc = (last.match(/bit_acc=([\d.]+)/) || [])[1] || '?';
        const byteMat = (last.match(/byte_match=([\d.]+)/) || [])[1] || '?';
        document.getElementById('statusBar').innerHTML =
          `<span class="live-dot"></span>` +
          `Step <span class="metric">${step}</span> &nbsp;|&nbsp; ` +
          `Loss <span class="metric">${loss}</span> &nbsp;|&nbsp; ` +
          `Bit Acc <span class="metric">${(parseFloat(bitAcc)*100).toFixed(1)}%</span> &nbsp;|&nbsp; ` +
          `Byte Match <span class="metric">${(parseFloat(byteMat)*100).toFixed(1)}%</span>`;
      }
    }
  } catch(e) {}

  // Poll LCX sidecar
  try {
    const lr = await fetch('/api/lcx');
    const ld = await lr.json();
    if (!ld.error) renderLCX(ld);
  } catch(e) {}
}

function renderLCX(d) {
  const isHash = d.lcx_mode === 'hash';
  const isZoom = d.num_levels > 0;

  document.getElementById('lcx_step').textContent = 'step ' + d.step +
    (isZoom ? ' [ZOOM x' + d.num_levels + ']' : isHash ? ' [HASH]' : '');

  if (isZoom) {
    renderZoomLCX(d);
  } else if (isHash && d.R) {
    renderLegacyHashLCX(d);
  } else if (d.R) {
    renderLegacyDenseLCX(d);
  }
}

function renderZoomLCX(d) {
  const nLevels = d.num_levels;
  const levelColors = [[0,255,255], [0,200,0], [255,200,0], [255,128,0], [255,0,128]];
  const levelNames = ['cyan', 'green', 'yellow', 'orange', 'pink'];
  const zg = d.zoom_gate;

  // Badge
  let badge = '<span style="color:#0ff;font-weight:bold">ZOOM LCX</span>';
  if (zg != null) badge += ' &nbsp;|&nbsp; gate: <span style="color:#7b68ee">' + zg.toFixed(3) + '</span>';
  document.getElementById('lcx_active_ch').innerHTML = badge;

  // Per-level norms
  let normHtml = '';
  let maxNorm = 0.001;
  const lvlNorms = [];
  for (let lvl = 0; lvl < nLevels; lvl++) {
    const data = d['L' + lvl];
    if (!data) continue;
    const norm = data.reduce(function(s,v){return s+v}, 0);
    lvlNorms.push(norm);
    if (norm > maxNorm) maxNorm = norm;
  }
  for (let lvl = 0; lvl < nLevels; lvl++) {
    const used = d['L' + lvl + '_used'] || 0;
    const total = d['L' + lvl + '_total'] || 0;
    const cols = d['L' + lvl + '_cols'] || Math.ceil(Math.sqrt(total || 1));
    const rows = d['L' + lvl + '_rows'] || Math.ceil((total || 1) / cols);
    const c = levelColors[lvl % levelColors.length];
    const cStr = 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
    const pct = ((lvlNorms[lvl] || 0) / maxNorm * 100).toFixed(0);
    normHtml += '<div style="display:flex;align-items:center;gap:6px;margin:2px 0">';
    normHtml += '<span style="color:' + cStr + ';width:30px;font-size:12px;font-weight:bold">L' + lvl + '</span>';
    normHtml += '<div style="flex:1;height:8px;background:#1a1a25;border-radius:4px;overflow:hidden">';
    normHtml += '<div style="width:' + pct + '%;height:100%;background:' + cStr + ';border-radius:4px"></div>';
    normHtml += '</div>';
    normHtml += '<span style="color:#888;font-size:11px;width:80px;text-align:right">' + used + '/' + total + ' (' + cols + 'x' + rows + ')</span>';
    normHtml += '</div>';
  }
  document.getElementById('lcx_norms').innerHTML = normHtml;

  // Generate canvases dynamically
  const container = document.getElementById('lcx_canvases');
  container.innerHTML = '';
  for (let lvl = 0; lvl < nLevels; lvl++) {
    const data = d['L' + lvl];
    if (!data) continue;
    const cols2 = d['L' + lvl + '_cols'] || Math.ceil(Math.sqrt(total || 1));
    const rows2 = d['L' + lvl + '_rows'] || Math.ceil((total || 1) / cols2);
    const used = d['L' + lvl + '_used'] || 0;
    const total = d['L' + lvl + '_total'] || 0;
    const c = levelColors[lvl % levelColors.length];
    const cStr = 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
    const borderC = 'rgb(' + (c[0]>>1) + ',' + (c[1]>>1) + ',' + (c[2]>>1) + ')';

    const wrapper = document.createElement('div');
    wrapper.style.textAlign = 'center';
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = Math.round(256 * rows2 / cols2);
    canvas.style.border = '2px solid ' + borderC;
    canvas.style.borderRadius = '6px';
    canvas.style.imageRendering = 'pixelated';
    wrapper.appendChild(canvas);
    const label = document.createElement('div');
    label.style.cssText = 'font-size:12px;margin-top:4px;font-weight:bold;color:' + cStr;
    label.textContent = 'L' + lvl + ' ' + cols2 + 'x' + rows2 + ' (' + used + '/' + total + ')';
    wrapper.appendChild(label);
    container.appendChild(wrapper);

    renderHashChannel(canvas, data, cols2, rows2, c);
  }
}

function renderLegacyHashLCX(d) {
  const side = d.side || d.num_bits || 8;
  const effort = d.effort_level || 0;
  const chNames = ['RED (fast)', 'GREEN (medium)', 'BLUE (slow)'];
  const chColors = ['#f44', '#4f4', '#44f'];

  let badge = '';
  for (let i = 0; i < 3; i++) {
    const active = i === effort;
    badge += '<span style="color:' + chColors[i] + ';opacity:' + (active?'1':'0.3') + ';' + (active?'font-weight:bold;':'') + 'margin-right:12px">';
    badge += (active ? '>> ' : '') + chNames[i] + (active ? ' <<' : '') + '</span>';
  }
  document.getElementById('lcx_active_ch').innerHTML = badge;

  const channels = [d.R, d.G, d.B];
  const norms = channels.map(function(ch) { return ch.reduce(function(s,v){return s+v},0); });
  const maxNorm = Math.max(norms[0], norms[1], norms[2], 0.001);
  let normHtml = '';
  ['R','G','B'].forEach(function(name, i) {
    const pct = (norms[i] / maxNorm * 100).toFixed(0);
    normHtml += '<div style="display:flex;align-items:center;gap:6px;margin:2px 0">';
    normHtml += '<span style="color:' + chColors[i] + ';width:14px;font-size:12px">' + name + '</span>';
    normHtml += '<div style="flex:1;height:8px;background:#1a1a25;border-radius:4px;overflow:hidden">';
    normHtml += '<div style="width:' + pct + '%;height:100%;background:' + chColors[i] + ';border-radius:4px"></div></div>';
    normHtml += '<span style="color:#888;font-size:11px;width:40px;text-align:right">' + norms[i].toFixed(2) + '</span></div>';
  });
  document.getElementById('lcx_norms').innerHTML = normHtml;

  const container = document.getElementById('lcx_canvases');
  container.innerHTML = '';
  const chData = [{n:'R',c:[255,0,0],d:d.R},{n:'G',c:[0,200,0],d:d.G},{n:'B',c:[60,100,255],d:d.B}];
  chData.forEach(function(ch) {
    const wrapper = document.createElement('div');
    wrapper.style.textAlign = 'center';
    const canvas = document.createElement('canvas');
    canvas.width = 256; canvas.height = 256;
    canvas.style.cssText = 'border:2px solid #333;border-radius:6px;image-rendering:pixelated';
    wrapper.appendChild(canvas);
    const label = document.createElement('div');
    label.style.cssText = 'font-size:12px;margin-top:4px;font-weight:bold;color:rgb('+ch.c.join(',')+')';
    label.textContent = ch.n;
    wrapper.appendChild(label);
    container.appendChild(wrapper);
    renderHashChannel(canvas, ch.d, side, side, ch.c);
  });
}

function renderLegacyDenseLCX(d) {
  const side = d.side || d.num_bits || 8;
  const container = document.getElementById('lcx_canvases');
  container.innerHTML = '';
  document.getElementById('lcx_active_ch').innerHTML = '<span style="color:#888">Dense LCX</span>';
  document.getElementById('lcx_norms').innerHTML = '';
  const chData = [{n:'R',c:[255,0,0],d:d.R},{n:'G',c:[0,200,0],d:d.G},{n:'B',c:[60,100,255],d:d.B}];
  chData.forEach(function(ch) {
    const wrapper = document.createElement('div');
    wrapper.style.textAlign = 'center';
    const canvas = document.createElement('canvas');
    canvas.width = 256; canvas.height = 256;
    canvas.style.cssText = 'border:2px solid #333;border-radius:6px;image-rendering:pixelated';
    wrapper.appendChild(canvas);
    const label = document.createElement('div');
    label.style.cssText = 'font-size:12px;margin-top:4px;font-weight:bold;color:rgb('+ch.c.join(',')+')';
    label.textContent = ch.n;
    wrapper.appendChild(label);
    container.appendChild(wrapper);
    renderChannel(canvas, ch.d, side, ch.c);
  });
}

// Hash LCX: slot norms as heatmap (values >= 0, brighter = more active)
// Accepts canvas element or ID string. cols x rows rectangle.
function renderHashChannel(canvasOrId, norms, cols, rows, color) {
  const canvas = typeof canvasOrId === 'string' ? document.getElementById(canvasOrId) : canvasOrId;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const cellW = canvas.width / cols;
  const cellH = canvas.height / rows;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const maxVal = Math.max.apply(null, norms.concat([0.001]));
  for (let i = 0; i < norms.length; i++) {
    const row = Math.floor(i / cols);
    const col = i % cols;
    const t = Math.min(1, norms[i] / maxVal);
    const r = Math.round(color[0] * t);
    const g = Math.round(color[1] * t);
    const b = Math.round(color[2] * t);
    ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
    ctx.fillRect(col * cellW, row * cellH, cellW, cellH);
  }
}

function renderChannel(canvasOrId, values, nb, color) {
  const canvas = typeof canvasOrId === 'string' ? document.getElementById(canvasOrId) : canvasOrId;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const cellW = canvas.width / nb;
  const cellH = canvas.height / nb;
  for (let i = 0; i < values.length; i++) {
    const row = Math.floor(i / nb);
    const col = i % nb;
    const intensity = Math.min(255, Math.max(0, (values[i] + 1) * 127.5));
    const r = Math.round(color[0] * intensity / 255);
    const g = Math.round(color[1] * intensity / 255);
    const b = Math.round(color[2] * intensity / 255);
    ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
    ctx.fillRect(col * cellW, row * cellH, cellW, cellH);
  }
}

function renderRGB(canvasId, R, G, B, nb) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const cellW = canvas.width / nb;
  const cellH = canvas.height / nb;
  for (let i = 0; i < R.length; i++) {
    const row = Math.floor(i / nb);
    const col = i % nb;
    const r = Math.min(255, Math.max(0, Math.round((R[i] + 1) * 127.5)));
    const g = Math.min(255, Math.max(0, Math.round((G[i] + 1) * 127.5)));
    const b = Math.min(255, Math.max(0, Math.round((B[i] + 1) * 127.5)));
    ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
    ctx.fillRect(col * cellW, row * cellH, cellW, cellH);
  }
}

poll();
setInterval(poll, POLL_MS);
</script>
</body>
</html>
"""


class ControlHandler(BaseHTTPRequestHandler):
    controls_path = DEFAULT_CONTROLS_PATH
    log_path = LOG_PATH

    def log_message(self, format, *args):
        # Silence default access logging
        pass

    def _send_json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_controls(self):
        try:
            with open(self.controls_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_controls(self, data):
        Path(self.controls_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.controls_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _serve_data_mix_html(self):
        """Serve a self-contained data mix HTML page for Grafana iframe embedding."""
        html = r'''<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#181b1f;color:#ccc;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;font-size:13px;padding:8px 12px;overflow-x:hidden}
.row{display:flex;align-items:center;gap:10px;padding:6px 8px;border-bottom:1px solid #2a2d33;transition:opacity 0.2s}
.row.off{opacity:0.4}
.row:hover{background:#1e2228}
.tier{display:inline-block;width:24px;height:20px;border-radius:3px;text-align:center;line-height:20px;font-weight:700;font-size:11px;color:#000;flex-shrink:0}
.t0{background:#4a9eff}.t1{background:#2ecc71}.t2{background:#f1c40f}.t3{background:#e67e22}.t4{background:#e74c3c}.t5{background:#9b59b6}.tX{background:#555}
.name{width:140px;flex-shrink:0;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.notes{flex:0 0 220px;font-size:11px;color:#888;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.toggle{position:relative;width:44px;height:22px;flex-shrink:0;cursor:pointer}
.toggle input{display:none}
.toggle .track{position:absolute;top:0;left:0;right:0;bottom:0;background:#3a3d44;border-radius:11px;transition:background 0.2s}
.toggle input:checked+.track{background:#2ecc71}
.toggle .knob{position:absolute;top:2px;left:2px;width:18px;height:18px;background:#fff;border-radius:50%;transition:transform 0.2s;box-shadow:0 1px 3px rgba(0,0,0,0.3)}
.toggle input:checked~.knob{transform:translateX(22px)}
.slider-wrap{flex:1;display:flex;align-items:center;gap:8px;min-width:0}
.slider-wrap input[type=range]{flex:1;height:6px;-webkit-appearance:none;appearance:none;background:#3a3d44;border-radius:3px;outline:none;transition:opacity 0.2s}
.slider-wrap input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;background:#4a9eff;border-radius:50%;cursor:pointer}
.slider-wrap input[type=range]:disabled{opacity:0.25;cursor:default}
.slider-wrap input[type=range]:disabled::-webkit-slider-thumb{background:#555;cursor:default}
.val{width:36px;text-align:right;font-variant-numeric:tabular-nums;font-size:12px;color:#aaa}
.bar{margin-top:8px;display:flex;height:8px;border-radius:4px;overflow:hidden;background:#2a2d33}
.bar div{transition:width 0.3s}
.hdr{display:flex;align-items:center;justify-content:space-between;padding:4px 8px 8px;border-bottom:2px solid #333}
.hdr .total{font-size:12px;color:#888}.hdr .total.warn{color:#e74c3c}
.btn{padding:4px 12px;border:1px solid #4a9eff;background:transparent;color:#4a9eff;border-radius:4px;cursor:pointer;font-size:12px;transition:all 0.15s}
.btn:hover{background:#4a9eff;color:#000}
.btn.apply{border-color:#2ecc71;color:#2ecc71}.btn.apply:hover{background:#2ecc71;color:#000}
.toast{position:fixed;bottom:8px;right:8px;background:#2ecc71;color:#000;padding:6px 14px;border-radius:4px;font-size:12px;opacity:0;transition:opacity 0.3s;pointer-events:none}
.toast.err{background:#e74c3c;color:#fff}
.toast.show{opacity:1}
</style></head><body>
<div class="hdr">
  <span class="total" id="total">Total: 0.00</span>
  <div style="display:flex;gap:6px">
    <button class="btn" onclick="normalize()">Normalize</button>
    <button class="btn apply" onclick="applyAll()">Apply</button>
  </div>
</div>
<div class="bar" id="mixBar"></div>
<div id="rows"></div>
<div class="toast" id="toast"></div>
<script>
const API='http://'+location.hostname+':7777';
const TIERS=['t0','t1','t2','t3','t4','t5'];
const TIER_COLORS=['#4a9eff','#2ecc71','#f1c40f','#e67e22','#e74c3c','#9b59b6'];
let FILES={};

function toast(msg,err){const t=document.getElementById('toast');t.textContent=msg;t.className='toast'+(err?' err':'')+' show';setTimeout(()=>t.className='toast',1500)}

async function load(){
  try{
    const r=await fetch(API+'/api/data_files');
    const d=await r.json();
    FILES=d.files||{};
    render();
  }catch(e){
    document.getElementById('rows').innerHTML='<div style="padding:20px;color:#666;text-align:center">Control panel not running. Start training to activate.</div>';
  }
}

function render(){
  const sorted=Object.entries(FILES).sort((a,b)=>{
    const ta=a[1].tier??99,tb=b[1].tier??99;
    return ta-tb||a[0].localeCompare(b[0]);
  });
  let html='';
  sorted.forEach(([name,info])=>{
    const short=name.replace('.traindat','');
    const on=info.weight>0;
    const t=info.tier??99;
    const tc=t<6?TIERS[t]:'tX';
    const tl=t<6?'T'+t:'?';
    html+=`<div class="row ${on?'':'off'}" id="row_${short}">`;
    html+=`<span class="tier ${tc}">${tl}</span>`;
    html+=`<span class="name" title="${name}">${short}</span>`;
    html+=`<label class="toggle"><input type="checkbox" ${on?'checked':''} onchange="toggleFile('${name}',this.checked)"><span class="track"></span><span class="knob"></span></label>`;
    html+=`<div class="slider-wrap">`;
    html+=`<input type="range" min="0" max="1" step="0.01" value="${info.weight||0}" ${on?'':'disabled'} id="sl_${short}" oninput="slideFile('${name}',this.value)">`;
    html+=`</div>`;
    html+=`<span class="val" id="v_${short}">${(info.weight||0).toFixed(2)}</span>`;
    html+=`<span class="notes" title="${info.notes||''}">${info.notes||''}</span>`;
    html+=`</div>`;
  });
  document.getElementById('rows').innerHTML=html;
  updateBar();
}

function toggleFile(name,on){
  const short=name.replace('.traindat','');
  const row=document.getElementById('row_'+short);
  const sl=document.getElementById('sl_'+short);
  const vl=document.getElementById('v_'+short);
  if(on){
    FILES[name].weight=FILES[name]._prev||0.5;
    sl.disabled=false;sl.value=FILES[name].weight;
    vl.textContent=FILES[name].weight.toFixed(2);
    row.classList.remove('off');
  }else{
    FILES[name]._prev=FILES[name].weight||0.5;
    FILES[name].weight=0;
    sl.disabled=true;sl.value=0;
    vl.textContent='0.00';
    row.classList.add('off');
  }
  updateBar();
}

function slideFile(name,val){
  const v=parseFloat(val);
  FILES[name].weight=v;
  const short=name.replace('.traindat','');
  document.getElementById('v_'+short).textContent=v.toFixed(2);
  updateBar();
}

function updateBar(){
  const sorted=Object.entries(FILES).sort((a,b)=>(a[1].tier??99)-(b[1].tier??99)||a[0].localeCompare(b[0]));
  const total=sorted.reduce((s,[,i])=>s+(i.weight||0),0);
  const el=document.getElementById('total');
  el.textContent='Total: '+total.toFixed(2);
  el.className='total'+(Math.abs(total-1)>0.01&&total>0?' warn':'');
  let bar='';
  if(total>0){sorted.forEach(([name,info])=>{
    if(info.weight>0){
      const t=info.tier??99;const c=t<6?TIER_COLORS[t]:'#555';
      const pct=(info.weight/total*100).toFixed(1);
      bar+=`<div style="width:${pct}%;background:${c}" title="${name.replace('.traindat','')}: ${pct}%"></div>`;
    }
  })}
  document.getElementById('mixBar').innerHTML=bar;
}

async function applyAll(){
  const weights={};
  Object.entries(FILES).forEach(([n,i])=>{weights[n]=i.weight||0});
  try{
    const r=await fetch(API+'/api/set_data_weights',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({weights})});
    const d=await r.json();
    if(d.ok)toast('Weights applied ('+d.count+' files)');
    else toast(d.error||'Error',true);
  }catch(e){toast('Connection error',true)}
}

async function normalize(){
  try{
    const r=await fetch(API+'/api/normalize_weights',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    const d=await r.json();
    if(d.ok){toast('Normalized');load()}
    else toast(d.error||'Error',true);
  }catch(e){toast('Connection error',true)}
}

load();
setInterval(load,5000);
</script></body></html>'''
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def _lcx_to_png(self, channel):
        """Generate a PNG image from LCX sidecar data.
        channel: 'r'/'g'/'b'/'rgb' (legacy) or 'L0'/'L1'/'L2' (zoom levels)."""
        lcx_path = os.path.join(os.path.dirname(self.controls_path), 'lcx_latest.json')
        with open(lcx_path, 'r') as f:
            d = json.load(f)

        # Zoom LCX: per-level PNG with viridis colormap (shared buffer zoom)
        if d.get('num_levels') and channel.startswith('L'):
            import math
            # Parse channel: 'L0', 'L1', 'L0_keys', 'L1_keys', etc.
            _ch_rest = channel[1:]  # '0', '1', '0_keys', etc.
            lvl = int(_ch_rest.split('_')[0]) if '_' in _ch_rest else int(_ch_rest)
            # Level cap: only active levels have data. Inactive → dark placeholder.
            _max_active = d.get('max_active_level', d.get('num_levels', 0) - 1)
            if lvl > _max_active or channel not in d:
                img = Image.new('RGB', (512, 512), (30, 30, 30))
                # Draw "INACTIVE" text centered
                from PIL import ImageDraw
                draw = ImageDraw.Draw(img)
                draw.text((180, 245), "INACTIVE", fill=(100, 100, 100))
                draw.text((155, 270), f"L{lvl} unlocks at tt={lvl}", fill=(80, 80, 80))
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                return buf.getvalue()
            norms = d.get(channel, [])
            n = len(norms)
            if n == 0:
                img = Image.new('RGB', (1, 1), (68, 1, 84))
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                return buf.getvalue()
            # Grid size from sidecar or sqrt
            _grid_sides = d.get('grid_sides', [])
            if lvl < len(_grid_sides):
                side = _grid_sides[lvl]
            else:
                side = d.get(f'L{lvl}_side', math.ceil(n ** 0.5))
            # Write-heat weighting: dim unwritten slots so activity pops
            heat_raw = d.get(f'L{lvl}_heat_raw', [])
            has_heat = len(heat_raw) == n and max(heat_raw) > 0
            # Normalization: use contrast-stretch (p5/p95) for visible detail
            sorted_norms = sorted(norms)
            p5 = sorted_norms[max(0, int(n * 0.05))]
            p95 = sorted_norms[min(n - 1, int(n * 0.95))]
            norm_range = max(p95 - p5, 0.001)
            img = Image.new('RGB', (side, side), _VIRIDIS[0])
            pixels = img.load()
            for i in range(n):
                r, c = i // side, i % side
                if c < side and r < side:
                    # Percentile-stretch normalization
                    t = max(0.0, min(1.0, (norms[i] - p5) / norm_range))
                    if has_heat:
                        # Blend: written slots full brightness, unwritten dim to 15%
                        heat_alpha = 1.0 if heat_raw[i] > 0 else 0.15
                        t = t * heat_alpha
                    idx = min(255, int(t * 255))
                    pixels[c, r] = _VIRIDIS[idx]
            img = img.resize((512, 512), Image.NEAREST)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return buf.getvalue()

        # Legacy R/G/B
        nb = d.get('num_bits', 8)
        R, G, B = d.get('R', []), d.get('G', []), d.get('B', [])
        img = Image.new('RGB', (nb, nb))
        pixels = img.load()
        for i in range(min(len(R), nb * nb)):
            row, col = i // nb, i % nb
            rv = min(255, max(0, int((R[i] + 1) * 127.5)))
            gv = min(255, max(0, int((G[i] + 1) * 127.5)))
            bv = min(255, max(0, int((B[i] + 1) * 127.5)))
            if channel == 'r':
                pixels[col, row] = (rv, 0, 0)
            elif channel == 'g':
                pixels[col, row] = (0, gv, 0)
            elif channel == 'b':
                pixels[col, row] = (0, 0, bv)
            else:
                pixels[col, row] = (rv, gv, bv)
        img = img.resize((256, 256), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

    def _lcx_composite_png(self):
        """Generate a single PNG with all LCX levels side by side."""
        lcx_path = os.path.join(os.path.dirname(self.controls_path), 'lcx_latest.json')
        with open(lcx_path, 'r') as f:
            d = json.load(f)
        n_levels = d.get('num_levels', 0)
        if n_levels == 0:
            img = Image.new('RGB', (256, 64), (30, 30, 30))
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return buf.getvalue()
        level_colors = [(0, 255, 255), (0, 200, 0), (255, 200, 0), (255, 128, 0), (255, 0, 100)]
        cell_px = 200  # each level image is 200x200
        gap = 4
        total_w = n_levels * cell_px + (n_levels - 1) * gap
        composite = Image.new('RGB', (total_w, cell_px + 24), (20, 20, 20))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(composite)
        for lvl in range(n_levels):
            key = f'L{lvl}'
            norms = d.get(key, [])
            n = len(norms)
            cols = d.get(key + '_cols', math.ceil(n ** 0.5) if n > 0 else 1)
            rows = d.get(key + '_rows', math.ceil(n / cols) if cols > 0 else 1)
            used = d.get(key + '_used', 0)
            total = d.get(key + '_total', n)
            color = level_colors[lvl % len(level_colors)]
            max_val = max(max(norms) if norms else 0, 0.001)
            tile = Image.new('RGB', (cols, rows))
            px = tile.load()
            for i in range(n):
                r, c = i // cols, i % cols
                if c < cols and r < rows:
                    t = min(1.0, norms[i] / max_val)
                    px[c, r] = (int(color[0] * t), int(color[1] * t), int(color[2] * t))
            scale = max(1, cell_px // max(cols, rows))
            tile = tile.resize((cols * scale, rows * scale), Image.NEAREST)
            x_off = lvl * (cell_px + gap)
            composite.paste(tile, (x_off, 0))
            # Border
            draw.rectangle([x_off, 0, x_off + cols * scale - 1, rows * scale - 1],
                           outline=tuple(min(255, c + 60) for c in color), width=1)
            # Label
            label = f"L{lvl} {cols}x{rows} ({used}/{total})"
            draw.text((x_off + 4, cell_px + 4), label, fill=color)
        buf = io.BytesIO()
        composite.save(buf, format='PNG')
        return buf.getvalue()

    def _lcx_heat_strip_png(self):
        """Render composite heat strip: all levels stacked, 128 bins each.
        Top sub-row per level: viridis heatmap (log2-scaled write count).
        Bottom sub-row: green/dark valid coverage."""
        from PIL import ImageDraw
        lcx_path = os.path.join(os.path.dirname(self.controls_path), 'lcx_latest.json')
        with open(lcx_path, 'r') as f:
            d = json.load(f)

        n_levels = d.get('num_levels', 0)
        max_active = d.get('max_active_level', 0)
        level_slots = d.get('level_slots', [])

        if n_levels == 0:
            img = Image.new('RGB', (256, 64), (30, 30, 30))
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return buf.getvalue()

        BINS = 128
        CELL_W = 4
        HEAT_H = 24
        VALID_H = 8
        LEVEL_H = HEAT_H + VALID_H
        GAP = 6
        MARGIN_L = 80
        PAD = 8
        STRIP_W = BINS * CELL_W  # 512

        img_w = MARGIN_L + STRIP_W + 8
        img_h = PAD + n_levels * LEVEL_H + max(0, n_levels - 1) * GAP + PAD
        img = Image.new('RGB', (img_w, img_h), (30, 30, 30))
        draw = ImageDraw.Draw(img)
        pixels = img.load()

        for lvl in range(n_levels):
            y_base = PAD + lvl * (LEVEL_H + GAP)
            heat_bins = d.get(f'L{lvl}_heat_bins', [])
            valid_bins = d.get(f'L{lvl}_valid_bins', [])
            slots = level_slots[lvl] if lvl < len(level_slots) else 0

            if lvl > max_active or not heat_bins:
                draw.text((4, y_base + 8), f"L{lvl} INACTIVE", fill=(80, 80, 80))
                continue

            # Label
            if slots >= 1000:
                label = f"L{lvl} ({slots/1000:.1f}K)"
            else:
                label = f"L{lvl} ({slots})"
            draw.text((4, y_base + 6), label, fill=(200, 200, 200))

            # Heat row (viridis, log2 scaled)
            for bi in range(min(BINS, len(heat_bins))):
                h = heat_bins[bi]
                t = math.log2(h + 1) / 15.0 if h > 0 else 0.0
                t = min(1.0, t)
                cidx = min(255, int(t * 255))
                color = _VIRIDIS[cidx]
                x0 = MARGIN_L + bi * CELL_W
                for dy in range(HEAT_H):
                    for dx in range(CELL_W):
                        pixels[x0 + dx, y_base + dy] = color

            # Valid row (green/dark)
            for bi in range(min(BINS, len(valid_bins))):
                v = valid_bins[bi]
                color = (115, 191, 105) if v > 0 else (30, 30, 30)
                x0 = MARGIN_L + bi * CELL_W
                for dy in range(VALID_H):
                    for dx in range(CELL_W):
                        pixels[x0 + dx, y_base + HEAT_H + dy] = color

        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

    def _lcx_rank_curve_png(self, level=0):
        """Render smooth 1D heatmap of sorted slot heat (matplotlib-style)."""
        from PIL import ImageDraw
        lcx_path = os.path.join(os.path.dirname(self.controls_path), 'lcx_latest.json')
        with open(lcx_path, 'r') as f:
            d = json.load(f)

        heat_raw = d.get(f'L{level}_heat_raw', [])
        STRIP_W = 900
        if not heat_raw or max(heat_raw) == 0:
            img = Image.new('RGB', (STRIP_W, 17), (30, 30, 30))
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return buf.getvalue()

        # Unsorted: original slot order reveals spatial clustering patterns
        n_slots = len(heat_raw)
        max_h = max(heat_raw)
        log_max = math.log2(max_h + 1)

        # 1) Build a 1-pixel-tall strip: each pixel = one slot in original order
        strip = Image.new('RGB', (n_slots, 1))
        spx = strip.load()
        for i, h in enumerate(heat_raw):
            t = math.log2(h + 1) / log_max if h > 0 else 0.0
            t = min(1.0, t)
            cidx = min(255, int(t * 255))
            spx[i, 0] = _DIAMOND[cidx]

        # 2) Bicubic upscale → smooth interpolated gradient
        strip = strip.resize((STRIP_W, 14), Image.BICUBIC)

        # 3) Composite: 14px heat + 1px gap + 2px utilization bar
        n_active = sum(1 for h in heat_raw if h > 0)
        util_w = int(n_active / n_slots * STRIP_W)
        composite = Image.new('RGB', (STRIP_W, 17), (20, 20, 28))
        composite.paste(strip, (0, 0))
        cpx = composite.load()
        for x in range(util_w):
            cpx[x, 15] = (200, 200, 200)
            cpx[x, 16] = (200, 200, 200)

        buf = io.BytesIO()
        composite.save(buf, format='PNG')
        return buf.getvalue()

    def _stage_badge_png(self):
        """Render a stage badge from controls.json — no InfluxDB dependency."""
        from PIL import ImageDraw, ImageFont
        ctrl = self._read_controls()
        stage = ctrl.get('stage', '?')
        effort_name = ctrl.get('effort_name', '')
        tt = ctrl.get('think_ticks', 0)
        use_lcx = ctrl.get('use_lcx', False)
        batch = ctrl.get('batch_size', 0)

        # Color per stage
        STAGE_COLORS = {
            'ALPHA': (100, 200, 255),   # cyan
            'BETA':  (202, 100, 228),   # purple
            'GAMMA': (115, 191, 105),   # green
            'DELTA': (255, 152, 48),    # orange
            'EPSILON': (255, 85, 85),   # red
            'ZETA':  (255, 215, 0),     # gold
        }
        color = STAGE_COLORS.get(stage.upper(), (180, 180, 180))

        # Build badge
        img_w, img_h = 180, 72
        img = Image.new('RGB', (img_w, img_h), (30, 30, 30))
        draw = ImageDraw.Draw(img)

        # Stage name (large)
        label = stage.upper()
        draw.text((img_w // 2, 6), label, fill=color, anchor='mt')

        # Subtitle: effort name + key params
        lcx_str = 'LCX' if use_lcx else 'no LCX'
        sub = f"{effort_name}  tt={tt}  b={batch}  {lcx_str}"
        draw.text((img_w // 2, 42), sub, fill=(160, 160, 160), anchor='mt')

        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

    def _send_png(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'image/png')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        # Strip query string for route matching (Grafana appends ?t=<ts> for cache busting)
        clean_path = self.path.split('?')[0]
        if clean_path == '/' or clean_path == '/index.html':
            self._send_html(HTML_PAGE)
        elif clean_path == '/api/lcx_all.png':
            try:
                self._send_png(self._lcx_composite_png())
            except Exception as e:
                print(f"  [PANEL] Composite PNG error: {e}", file=sys.stderr)
                self.send_error(500, f'LCX composite PNG failed: {e}')
        elif clean_path == '/api/stage_badge.png':
            try:
                self._send_png(self._stage_badge_png())
            except Exception as e:
                print(f"  [PANEL] Stage badge error: {e}", file=sys.stderr)
                self.send_error(500, f'Stage badge failed: {e}')
        elif re.match(r'/api/lcx_L(\d+)_rank\.png', clean_path):
            try:
                _lvl = int(re.match(r'/api/lcx_L(\d+)_rank\.png', clean_path).group(1))
                self._send_png(self._lcx_rank_curve_png(level=_lvl))
            except Exception as e:
                print(f"  [PANEL] Rank curve error: {e}", file=sys.stderr)
                self.send_error(500, f'Rank curve failed: {e}')
        elif clean_path == '/api/lcx_heat_strip.png':
            try:
                self._send_png(self._lcx_heat_strip_png())
            except Exception as e:
                print(f"  [PANEL] Heat strip error: {e}", file=sys.stderr)
                self.send_error(500, f'Heat strip failed: {e}')
        elif clean_path.startswith('/api/lcx_') and clean_path.endswith('.png'):
            try:
                ch = clean_path.split('lcx_')[1].split('.')[0]  # r, g, b, rgb, L0, L1, L2
                self._send_png(self._lcx_to_png(ch))
            except Exception as e:
                print(f"  [PANEL] PNG error for {self.path}: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                self.send_error(500, f'LCX PNG generation failed: {e}')
        elif clean_path == '/api/controls':
            self._send_json(self._read_controls())
        elif clean_path == '/api/log_tail':
            lines = ''
            try:
                with open(self.log_path, 'rb') as f:
                    # Read last 4KB
                    f.seek(0, 2)
                    size = f.tell()
                    f.seek(max(0, size - 4096))
                    lines = f.read().decode('utf-8', errors='replace')
                    # Keep last 15 lines
                    lines = '\n'.join(lines.strip().split('\n')[-15:])
            except Exception:
                lines = '(no log file)'
            self._send_json({'lines': lines})
        elif clean_path == '/api/lcx':
            lcx_path = os.path.join(os.path.dirname(self.controls_path), 'lcx_latest.json')
            try:
                with open(lcx_path, 'r') as f:
                    self._send_json(json.load(f))
            except Exception:
                self._send_json({'error': 'no sidecar'}, 404)
        elif clean_path == '/api/health':
            self._send_json({'version': _PANEL_VERSION, 'pid': os.getpid()})
        elif clean_path == '/api/meta':
            meta = {}
            data_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / 'data' / 'traindat'
            for mf in data_dir.glob('*.meta.json'):
                name = mf.name.replace('.meta.json', '.traindat')
                try:
                    with open(mf) as f:
                        meta[name] = json.load(f)
                except Exception:
                    pass
            self._send_json(meta)
        elif clean_path == '/api/data_files':
            controls = self._read_controls()
            weights = controls.get('data_weights', {})
            data_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / 'data' / 'traindat'
            for f in data_dir.glob('*.traindat'):
                if f.name not in weights:
                    weights[f.name] = 0.0
            files = {}
            for name, w in sorted(weights.items()):
                meta = {}
                meta_path = data_dir / name.replace('.traindat', '.meta.json')
                if meta_path.exists():
                    try:
                        with open(meta_path) as mf:
                            meta = json.load(mf)
                    except Exception:
                        pass
                files[name] = {
                    'weight': w,
                    'enabled': w > 0,
                    'tier': meta.get('tier', 99),
                    'task': meta.get('task', ''),
                    'notes': meta.get('notes', ''),
                    'size_mb': round(meta.get('actual_bytes', 0) / (1024 * 1024), 1),
                }
            self._send_json({'files': files})
        elif clean_path == '/data_mix':
            self._serve_data_mix_html()
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == '/api/set':
            key = body.get('key')
            value = body.get('value')
            if key in ('lr', 'think_ticks', 'batch_size', 'use_lcx', 'num_bits',
                       'stage', 'effort', 'effort_name',
                       'checkpoint_every', 'eval_every', 'eval_samples',
                       'temporal_fibonacci', 'effort_mode', 'effort_lock',
                       'agc_enabled', 'agc_low', 'agc_high'):
                controls = self._read_controls()
                old_value = controls.get(key, 'none')
                controls[key] = value
                self._write_controls(controls)
                influx_log_control_change(key, old_value, value)
                influx_log_control_value(key, value)
                print(f"  [PANEL] {key} = {value}")
                self._send_json({'ok': True})
            else:
                self._send_json({'ok': False, 'error': f'Unknown key: {key}'}, 400)

        elif self.path == '/api/set_data_weight':
            name = body.get('name')
            value = body.get('value', 0)
            controls = self._read_controls()
            if 'data_weights' not in controls:
                controls['data_weights'] = {}
            old_value = controls['data_weights'].get(name, 0)
            controls['data_weights'][name] = float(value)
            self._write_controls(controls)
            influx_log_control_change(f"data:{name}", old_value, value)
            print(f"  [PANEL] data_weight {name} = {value}")
            self._send_json({'ok': True})

        elif self.path == '/api/set_data_weights':
            weights = body.get('weights', {})
            if not isinstance(weights, dict):
                self._send_json({'ok': False, 'error': 'weights must be dict'}, 400)
                return
            controls = self._read_controls()
            old_weights = controls.get('data_weights', {})
            controls['data_weights'] = {k: float(v) for k, v in weights.items()}
            self._write_controls(controls)
            for k, v in weights.items():
                old_v = old_weights.get(k, 0)
                if abs(float(v) - float(old_v)) > 0.001:
                    influx_log_control_change(f"data:{k}", old_v, v)
            print(f"  [PANEL] data_weights bulk update: {len(weights)} files")
            self._send_json({'ok': True, 'count': len(weights)})

        elif self.path == '/api/normalize_weights':
            controls = self._read_controls()
            weights = controls.get('data_weights', {})
            total = sum(w for w in weights.values() if w > 0)
            if total <= 0:
                self._send_json({'ok': False, 'error': 'No active weights'})
                return
            for k in weights:
                if weights[k] > 0:
                    weights[k] = round(weights[k] / total, 3)
            controls['data_weights'] = weights
            self._write_controls(controls)
            print(f"  [PANEL] weights normalized (was {total:.3f})")
            self._send_json({'ok': True, 'total_before': total})

        elif self.path == '/api/set_being_state':
            idx = str(body.get('idx'))
            state = body.get('state')
            if state not in ('active', 'frozen', 'null'):
                self._send_json({'ok': False, 'error': f'Invalid state: {state}'}, 400)
                return
            controls = self._read_controls()
            if 'being_states' not in controls:
                controls['being_states'] = {}
            old_state = controls['being_states'].get(idx, 'unknown')
            controls['being_states'][idx] = state
            self._write_controls(controls)
            influx_log_control_change(f"being_{idx}", old_state, state)
            print(f"  [PANEL] being {idx} = {state}")
            self._send_json({'ok': True})

        elif self.path == '/api/advance_stage':
            controls = self._read_controls()
            PHASES = [
                ('INFANT',  {'use_lcx': False, 'think_ticks': 0, 'batch_size': 10}),
                ('CHILD',   {'use_lcx': True,  'think_ticks': 0, 'batch_size': 10, '_resize_lcx': 2000}),
                ('TEEN',    {'use_lcx': True,  'think_ticks': 0, 'batch_size': 10, '_resize_lcx': 20000}),
                ('RECALL',  {'use_lcx': True,  'think_ticks': 1, 'batch_size': 10, '_resize_lcx': 100000}),
                ('DEPTH',   {'use_lcx': True,  'think_ticks': 2, 'batch_size': 5,  '_resize_lcx': 200000}),
                ('SAGE',    {'use_lcx': True,  'think_ticks': 4, 'batch_size': 2,  '_resize_lcx': 200000}),
            ]
            current = controls.get('stage', 'INFANT')
            idx = next((i for i, (name, _) in enumerate(PHASES) if name == current), -1)
            if idx < len(PHASES) - 1:
                next_name, next_cfg = PHASES[idx + 1]
                old_stage = current
                controls['stage'] = next_name
                for k, v in next_cfg.items():
                    controls[k] = v
                self._write_controls(controls)
                influx_log_control_change('stage', old_stage, next_name)
                print(f"  [PANEL] stage advance: {old_stage} -> {next_name}")
                self._send_json({'ok': True, 'stage': next_name})
            else:
                self._send_json({'ok': False, 'error': f'Already at max stage: {current}'})

        elif self.path == '/api/resize_lcx':
            slots = body.get('slots')
            if not isinstance(slots, int) or slots < 100:
                self._send_json({'ok': False, 'error': 'slots must be int >= 100'}, 400)
                return
            controls = self._read_controls()
            controls['_resize_lcx'] = slots
            self._write_controls(controls)
            influx_log_control_change('resize_lcx', 0, slots)
            print(f"  [PANEL] resize_lcx command queued: {slots:,} slots")
            self._send_json({'ok': True, 'resize_target': slots})

        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()


def _kill_stale_panel(port):
    """Kill any existing control_panel process on the given port before starting."""
    try:
        # Quick probe: is anyone listening?
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        result = s.connect_ex(('127.0.0.1', port))
        s.close()
        if result != 0:
            return  # port free, nothing to kill

        # Try the health endpoint to see if it's one of ours
        from urllib.request import urlopen as _ul
        try:
            resp = json.loads(_ul(f'http://localhost:{port}/api/health', timeout=2).read())
            old_pid = resp.get('pid')
            old_ver = resp.get('version', '?')
            print(f"  [PANEL] Killing stale panel (pid={old_pid}, version={old_ver}) on :{port}")
            if sys.platform == 'win32':
                subprocess.run(['taskkill', '/PID', str(old_pid), '/F'],
                               capture_output=True, timeout=5)
            else:
                os.kill(old_pid, signal.SIGTERM)
            time.sleep(0.5)
        except Exception:
            # Old process doesn't have /api/health -- even more reason to kill it.
            # Find PIDs listening on the port via netstat.
            print(f"  [PANEL] Stale process on :{port} has no health endpoint -- force-killing")
            try:
                out = subprocess.check_output(
                    f'netstat -ano | findstr ":{port}" | findstr LISTENING',
                    shell=True, text=True, timeout=5
                )
                pids = set()
                for line in out.strip().split('\n'):
                    parts = line.split()
                    if parts:
                        pids.add(parts[-1])
                my_pid = str(os.getpid())
                for pid in pids:
                    if pid != my_pid and pid != '0':
                        print(f"  [PANEL] Killing pid {pid}")
                        subprocess.run(['taskkill', '/PID', pid, '/F'],
                                       capture_output=True, timeout=5)
                time.sleep(0.5)
            except Exception:
                pass
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description='Diamond Code Live Control Panel')
    parser.add_argument('--port', type=int, default=7777)
    parser.add_argument('--controls', type=str, default=DEFAULT_CONTROLS_PATH)
    parser.add_argument('--log', type=str, default=LOG_PATH)
    args = parser.parse_args()

    _kill_stale_panel(args.port)

    ControlHandler.controls_path = args.controls
    ControlHandler.log_path = args.log

    HTTPServer.allow_reuse_address = True
    server = HTTPServer(('0.0.0.0', args.port), ControlHandler)
    print(f"Diamond Code Control Panel  (v{_PANEL_VERSION}, pid={os.getpid()})")
    print(f"  URL:      http://localhost:{args.port}")
    print(f"  Controls: {args.controls}")
    print(f"  Log:      {args.log}")
    print(f"  Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == '__main__':
    main()
