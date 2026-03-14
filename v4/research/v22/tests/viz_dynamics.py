"""
v22 Internal Dynamics Visualization
====================================
Side-by-side: Spatial vs Temporal vs Listen/Think
Heatmaps + charge traces + energy + animated GIF
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio
from v22_best_config import SelfWiringGraph

np.random.seed(42)
random.seed(42)

V = 8
N = 64
TICKS = 8
BUDGET = 2000
INPUT_CHAR = 3
LISTEN_TICKS = 4

# --- Train a network briefly ---
net = SelfWiringGraph(N, V)
targets = np.random.permutation(V)
codebook = np.array([[float((i >> bit) & 1) for bit in range(V)]
                      for i in range(V)], dtype=np.float32)

best_acc = 0.0
for att in range(BUDGET):
    sm = net.mask.copy()
    sw = net.W.copy()
    net.mutate_structure(0.07)
    Weff = net.W * net.mask
    ch = np.zeros((V, N), dtype=np.float32)
    ac = np.zeros((V, N), dtype=np.float32)
    for t in range(TICKS):
        if t == 0:
            ac[:, :V] = codebook
        raw = ac @ Weff + ac * 0.1
        np.nan_to_num(raw, copy=False)
        ch += raw * 0.3
        ch *= net.leak
        ac = np.maximum(ch - net.threshold, 0)
        ch = np.clip(ch, -net.threshold * 2, net.threshold * 2)
    out = ch[:, net.out_start:net.out_start + V]
    pred = np.argmax(out, axis=1)
    acc = (pred == targets).mean()
    if acc >= best_acc:
        best_acc = acc
    else:
        net.mask = sm
        net.W = sw

print(f"Trained: {net.count_connections()} conns, acc={best_acc*100:.0f}%", flush=True)
Weff = net.W * net.mask

# --- Spike train for temporal ---
spike_train = np.zeros((TICKS, V), dtype=np.float32)
for t in range(TICKS):
    bit_idx = t % V
    spike_train[t, bit_idx] = float((INPUT_CHAR >> bit_idx) & 1)

# --- Record dynamics for 3 modes ---
def run_mode(mode):
    ch = np.zeros(N, dtype=np.float32)
    ac = np.zeros(N, dtype=np.float32)
    charges = np.zeros((TICKS, N), dtype=np.float32)
    for t in range(TICKS):
        if mode == "spatial":
            if t == 0:
                ac[:V] = codebook[INPUT_CHAR]
        elif mode == "temporal":
            ac[:V] = spike_train[t]
        elif mode == "listen_think":
            if t < LISTEN_TICKS:
                ac[:V] = spike_train[t]
            # else: no injection
        raw = ac @ Weff + ac * 0.1
        np.nan_to_num(raw, copy=False)
        ch += raw * 0.3
        ch *= net.leak
        ac = np.maximum(ch - net.threshold, 0)
        ch = np.clip(ch, -net.threshold * 2, net.threshold * 2)
        charges[t] = ch.copy()
    return charges

spatial = run_mode("spatial")
temporal = run_mode("temporal")
listen = run_mode("listen_think")
print("Dynamics recorded.", flush=True)

# --- Plot ---
vmax = max(np.abs(spatial).max(), np.abs(temporal).max(), np.abs(listen).max())
if vmax < 0.001:
    vmax = 1.0
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("v22 Internal Dynamics: Spatial vs Temporal vs Listen/Think",
             fontsize=14, fontweight="bold")

# Row 1: heatmaps (spatial vs temporal)
for col, (data, title) in enumerate([
    (spatial, "SPATIAL (input tick 0 only)"),
    (temporal, "TEMPORAL (spikes every tick)"),
]):
    ax = axes[0, col]
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
    ax.set_xlabel("Neuron index")
    ax.set_ylabel("Tick")
    ax.set_title(title, fontweight="bold")
    ax.set_yticks(range(TICKS))
    ax.axvline(V - 0.5, color="green", linewidth=2, linestyle="--")
    ax.axvline(net.out_start - 0.5, color="orange", linewidth=2, linestyle="--")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Charge")

# Row 2: listen/think heatmap + energy
ax = axes[1, 0]
im = ax.imshow(listen, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
ax.set_xlabel("Neuron index")
ax.set_ylabel("Tick")
ax.set_title("LISTEN/THINK (spikes 0-3, silence 4-7)", fontweight="bold")
ax.set_yticks(range(TICKS))
ax.axvline(V - 0.5, color="green", linewidth=2, linestyle="--")
ax.axvline(net.out_start - 0.5, color="orange", linewidth=2, linestyle="--")
ax.axhline(LISTEN_TICKS - 0.5, color="red", linewidth=2, linestyle="-")
plt.colorbar(im, ax=ax, shrink=0.8, label="Charge")

ax = axes[1, 1]
ax.plot(range(TICKS), np.abs(spatial).sum(axis=1), "b-o", lw=2, label="Spatial")
ax.plot(range(TICKS), np.abs(temporal).sum(axis=1), "r-s", lw=2, label="Temporal")
ax.plot(range(TICKS), np.abs(listen).sum(axis=1), "g-^", lw=2, label="Listen/Think")
ax.axvline(LISTEN_TICKS - 0.5, color="red", lw=1, ls="--", alpha=0.5)
ax.set_xlabel("Tick")
ax.set_ylabel("Total |charge| energy")
ax.set_title("Energy per tick", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)

# Row 3: charge traces
for col, (data, title) in enumerate([
    (spatial, "Spatial - top 10 neuron traces"),
    (temporal, "Temporal - top 10 neuron traces"),
]):
    ax = axes[2, col]
    activity = np.abs(data).max(axis=0)
    top10 = np.argsort(activity)[-10:]
    for idx in top10:
        zone = "IN" if idx < V else ("OUT" if idx >= net.out_start else "INT")
        ax.plot(range(TICKS), data[:, idx], "-o", markersize=3,
                label=f"n{idx}({zone})", linewidth=1.5)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Charge")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_dir = os.path.dirname(__file__)
png_path = os.path.join(out_dir, "viz_spatial_vs_temporal.png")
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved PNG: {png_path}", flush=True)

# --- GIF ---
frames = []
colors = ["blue" if i < V else "orange" if i >= net.out_start else "gray"
          for i in range(N)]
for t in range(TICKS):
    fig_f, ax_f = plt.subplots(1, 3, figsize=(15, 3))
    for col, (data, label) in enumerate([
        (spatial, "Spatial"),
        (temporal, "Temporal"),
        (listen, "Listen/Think"),
    ]):
        ax_f[col].bar(range(N), data[t], color=colors, width=1.0)
        ax_f[col].set_ylim(-vmax * 1.1, vmax * 1.1)
        suffix = ""
        if col == 2:
            suffix = " (LISTENING)" if t < LISTEN_TICKS else " (THINKING)"
        ax_f[col].set_title(f"{label} - tick {t}{suffix}")
        ax_f[col].set_xlabel("Neuron")
        ax_f[col].set_ylabel("Charge")
    plt.tight_layout()
    tmp = os.path.join(out_dir, f"_tmp_frame{t}.png")
    fig_f.savefig(tmp, dpi=100, bbox_inches="tight")
    frames.append(imageio.imread(tmp))
    plt.close(fig_f)
    os.remove(tmp)

gif_path = os.path.join(out_dir, "viz_spatial_vs_temporal.gif")
imageio.mimsave(gif_path, frames, duration=0.8, loop=0)
print(f"Saved GIF: {gif_path}", flush=True)
print("Done!", flush=True)
