"""
Interference Pattern Analysis
==============================
Is the pattern informative? Measures pattern QUALITY, not total energy.

Metrics:
1. Final state cosine distance matrix (27x27 per mode)
2. Output margin per char
3. Pattern divergence over ticks ("t" vs "h")
4. Neuron-level contrast heatmap
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio.v2 as imageio
from v22_best_config import SelfWiringGraph
from collections import Counter
from scipy.spatial.distance import cosine as cosine_dist

OUT_DIR = os.path.join(os.path.dirname(__file__), 'viz_interference')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)
random.seed(42)

# --- Setup ---
V = 8
N = 64
TICKS = 8
BUDGET = 2000
LISTEN_TICKS = 4

# Bigram targets
TEXT = 'the quick brown fox jumps over the lazy dog and the cat sat on the mat'
chars = sorted(set(TEXT))
VOCAB = len(chars)
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for c, i in c2i.items()}
bigrams = {}
for i in range(len(TEXT) - 1):
    bigrams.setdefault(c2i[TEXT[i]], []).append(c2i[TEXT[i + 1]])
TARGETS = np.array([Counter(bigrams.get(i, [0])).most_common(1)[0][0]
                     for i in range(VOCAB)])

codebook = np.array([[float((i >> bit) & 1) for bit in range(V)]
                      for i in range(VOCAB)], dtype=np.float32)

# --- Train network ---
net = SelfWiringGraph(N, V)
targets_perm = np.random.permutation(V)
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
            ac[:, :V] = np.eye(V, dtype=np.float32)
        raw = ac @ Weff + ac * 0.1
        np.nan_to_num(raw, copy=False)
        ch += raw * 0.3
        ch *= net.leak
        ac = np.maximum(ch - net.threshold, 0)
        ch = np.clip(ch, -net.threshold * 2, net.threshold * 2)
    out = ch[:, net.out_start:net.out_start + V]
    pred = np.argmax(out, axis=1)
    acc = (pred == targets_perm).mean()
    if acc >= best_acc:
        best_acc = acc
    else:
        net.mask = sm
        net.W = sw

print(f"Trained: {net.count_connections()} conns, acc={best_acc*100:.0f}%", flush=True)
Weff = net.W * net.mask

# --- Spike trains for temporal ---
spike_trains = np.zeros((VOCAB, TICKS, V), dtype=np.float32)
for ci in range(VOCAB):
    for t in range(TICKS):
        bit_idx = t % V
        spike_trains[ci, t, bit_idx] = float((ci >> bit_idx) & 1)


# --- Run all chars through all modes ---
def run_all_chars(mode):
    """Returns (final_charges[VOCAB, N], all_ticks[VOCAB, TICKS, N])"""
    all_final = np.zeros((VOCAB, N), dtype=np.float32)
    all_ticks = np.zeros((VOCAB, TICKS, N), dtype=np.float32)
    for ci in range(VOCAB):
        ch = np.zeros(N, dtype=np.float32)
        ac = np.zeros(N, dtype=np.float32)
        for t in range(TICKS):
            if mode == "spatial":
                if t == 0:
                    ac[:V] = codebook[ci]
            elif mode == "temporal":
                ac[:V] = spike_trains[ci, t]
            elif mode == "listen_think":
                if t < LISTEN_TICKS:
                    ac[:V] = spike_trains[ci, t]
            raw = ac @ Weff + ac * 0.1
            np.nan_to_num(raw, copy=False)
            ch += raw * 0.3
            ch *= net.leak
            ac = np.maximum(ch - net.threshold, 0)
            ch = np.clip(ch, -net.threshold * 2, net.threshold * 2)
            all_ticks[ci, t] = ch.copy()
        all_final[ci] = ch.copy()
    return all_final, all_ticks


modes = ["spatial", "temporal", "listen_think"]
mode_data = {}
for mode in modes:
    final, ticks = run_all_chars(mode)
    mode_data[mode] = {"final": final, "ticks": ticks}
    print(f"  {mode}: final charge range [{final.min():.3f}, {final.max():.3f}]", flush=True)


# ============================================================
# 1. Distance matrices
# ============================================================
print("\n1. Distance matrices...", flush=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Cosine Distance Between Final States (all char pairs)", fontsize=14, fontweight="bold")

mean_dists = {}
for idx, mode in enumerate(modes):
    final = mode_data[mode]["final"]
    dist_mat = np.zeros((VOCAB, VOCAB), dtype=np.float32)
    for i in range(VOCAB):
        for j in range(VOCAB):
            if i == j:
                dist_mat[i, j] = 0
            else:
                ni = np.linalg.norm(final[i])
                nj = np.linalg.norm(final[j])
                if ni < 1e-8 or nj < 1e-8:
                    dist_mat[i, j] = 1.0
                else:
                    dist_mat[i, j] = cosine_dist(final[i], final[j])

    mean_d = dist_mat[np.triu_indices(VOCAB, k=1)].mean()
    mean_dists[mode] = mean_d

    ax = axes[idx]
    im = ax.imshow(dist_mat, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(f"{mode}\nmean dist={mean_d:.3f}", fontweight="bold")
    labels = [i2c.get(i, '?') for i in range(VOCAB)]
    ax.set_xticks(range(VOCAB))
    ax.set_xticklabels(labels, fontsize=5)
    ax.set_yticks(range(VOCAB))
    ax.set_yticklabels(labels, fontsize=5)
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
path = os.path.join(OUT_DIR, "distance_matrices.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}", flush=True)
for m, d in mean_dists.items():
    print(f"    {m}: mean cosine dist = {d:.4f}", flush=True)


# ============================================================
# 2. Output margins
# ============================================================
print("\n2. Output margins...", flush=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Output Margin per Character (correct - best wrong)", fontsize=14, fontweight="bold")

for idx, mode in enumerate(modes):
    final = mode_data[mode]["final"]
    out_zone = final[:, net.out_start:net.out_start + V]
    margins = []
    for ci in range(min(VOCAB, V)):  # only chars that map to output neurons
        target = TARGETS[ci] if TARGETS[ci] < V else 0
        correct_val = out_zone[ci, target]
        other_vals = np.concatenate([out_zone[ci, :target], out_zone[ci, target+1:]])
        if len(other_vals) > 0:
            margin = correct_val - other_vals.max()
        else:
            margin = correct_val
        margins.append(margin)

    ax = axes[idx]
    colors = ["green" if m > 0 else "red" for m in margins]
    ax.bar(range(len(margins)), margins, color=colors)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Input char index")
    ax.set_ylabel("Margin")
    pos = sum(1 for m in margins if m > 0)
    ax.set_title(f"{mode}\n{pos}/{len(margins)} positive margins", fontweight="bold")
    ax.set_xticks(range(len(margins)))
    ax.set_xticklabels([i2c.get(i, '?') for i in range(len(margins))], fontsize=6)

plt.tight_layout()
path = os.path.join(OUT_DIR, "output_margins.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}", flush=True)


# ============================================================
# 3. Pattern divergence over ticks ("t" vs "h")
# ============================================================
print("\n3. Pattern divergence...", flush=True)

char_a = c2i.get('t', 0)
char_b = c2i.get('h', 1)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.set_title(f"Pattern Divergence: '{i2c[char_a]}' vs '{i2c[char_b]}' over ticks",
             fontsize=14, fontweight="bold")

for mode in modes:
    ticks_data = mode_data[mode]["ticks"]
    distances = []
    for t in range(TICKS):
        va = ticks_data[char_a, t]
        vb = ticks_data[char_b, t]
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na < 1e-8 or nb < 1e-8:
            distances.append(1.0)
        else:
            distances.append(cosine_dist(va, vb))
    ax.plot(range(TICKS), distances, "-o", linewidth=2, markersize=6, label=mode)

ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(LISTEN_TICKS - 0.5, color="red", linewidth=1, linestyle="--", alpha=0.3,
           label="listen|think boundary")
ax.set_xlabel("Tick")
ax.set_ylabel("Cosine distance")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
path = os.path.join(OUT_DIR, "pattern_divergence.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}", flush=True)


# ============================================================
# 4. Neuron-level contrast heatmap ("t" vs "h")
# ============================================================
print("\n4. Contrast heatmaps...", flush=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
fig.suptitle(f"Neuron Contrast: charge('{i2c[char_a]}') - charge('{i2c[char_b]}') at final tick",
             fontsize=14, fontweight="bold")

for idx, mode in enumerate(modes):
    final = mode_data[mode]["final"]
    diff = final[char_a] - final[char_b]
    vmax_c = max(np.abs(diff).max(), 0.01)

    ax = axes[idx]
    colors_bar = []
    for i in range(N):
        if i < V:
            colors_bar.append("blue")
        elif i >= net.out_start:
            colors_bar.append("orange")
        else:
            colors_bar.append("gray")
    ax.bar(range(N), diff, color=colors_bar, width=1.0)
    ax.set_ylim(-vmax_c * 1.2, vmax_c * 1.2)
    ax.axhline(0, color="black", linewidth=0.5)
    nonzero = np.sum(np.abs(diff) > 0.01)
    ax.set_title(f"{mode}\n{nonzero}/{N} neurons differ", fontweight="bold")
    ax.set_xlabel("Neuron (blue=IN, gray=INT, orange=OUT)")
    ax.set_ylabel("Charge difference")

plt.tight_layout()
path = os.path.join(OUT_DIR, "contrast_heatmap.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}", flush=True)


# ============================================================
# 5. Summary figure
# ============================================================
print("\n5. Summary...", flush=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Interference Pattern Analysis - Summary", fontsize=16, fontweight="bold")

# Top-left: mean cosine distance
ax = axes[0, 0]
ax.bar(modes, [mean_dists[m] for m in modes],
       color=["steelblue", "indianred", "seagreen"])
ax.set_ylabel("Mean cosine distance")
ax.set_title("Pattern Separation (higher = better)")
ax.set_ylim(0, 1)
for i, m in enumerate(modes):
    ax.text(i, mean_dists[m] + 0.02, f"{mean_dists[m]:.3f}", ha="center", fontweight="bold")

# Top-right: positive margin count
ax = axes[0, 1]
pos_counts = []
for mode in modes:
    final = mode_data[mode]["final"]
    out_zone = final[:, net.out_start:net.out_start + V]
    pos = 0
    for ci in range(min(VOCAB, V)):
        target = TARGETS[ci] if TARGETS[ci] < V else 0
        correct = out_zone[ci, target]
        others = np.concatenate([out_zone[ci, :target], out_zone[ci, target+1:]])
        if len(others) > 0 and correct > others.max():
            pos += 1
    pos_counts.append(pos)
ax.bar(modes, pos_counts, color=["steelblue", "indianred", "seagreen"])
ax.set_ylabel("Chars with positive margin")
ax.set_title(f"Correct Output Dominance (out of {min(VOCAB, V)})")

# Bottom-left: divergence traces
ax = axes[1, 0]
for mode in modes:
    ticks_data = mode_data[mode]["ticks"]
    distances = []
    for t in range(TICKS):
        va = ticks_data[char_a, t]
        vb = ticks_data[char_b, t]
        na = np.linalg.norm(va)
        nb = np.linalg.norm(vb)
        if na < 1e-8 or nb < 1e-8:
            distances.append(1.0)
        else:
            distances.append(cosine_dist(va, vb))
    ax.plot(range(TICKS), distances, "-o", linewidth=2, label=mode)
ax.set_xlabel("Tick")
ax.set_ylabel("Cosine distance")
ax.set_title(f"Divergence: '{i2c[char_a]}' vs '{i2c[char_b]}'")
ax.legend()
ax.grid(True, alpha=0.3)

# Bottom-right: contrast histogram
ax = axes[1, 1]
for mode in modes:
    final = mode_data[mode]["final"]
    diff = np.abs(final[char_a] - final[char_b])
    ax.hist(diff, bins=20, alpha=0.5, label=mode)
ax.set_xlabel("|Charge difference|")
ax.set_ylabel("Neuron count")
ax.set_title(f"Contrast Distribution: '{i2c[char_a]}' vs '{i2c[char_b]}'")
ax.legend()

plt.tight_layout()
path = os.path.join(OUT_DIR, "summary.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}", flush=True)


# ============================================================
# 6. Animation GIF: "t" vs "h" overlaid, tick by tick
# ============================================================
print("\n6. Animation...", flush=True)

frames = []
for t in range(TICKS):
    fig_f, ax_f = plt.subplots(1, 3, figsize=(18, 4))
    fig_f.suptitle(f"Tick {t}: charge for '{i2c[char_a]}' (solid) vs '{i2c[char_b]}' (dashed)",
                   fontsize=12, fontweight="bold")
    for col, mode in enumerate(modes):
        ticks_data = mode_data[mode]["ticks"]
        ch_a = ticks_data[char_a, t]
        ch_b = ticks_data[char_b, t]
        vmax_f = max(np.abs(ch_a).max(), np.abs(ch_b).max(), 0.01) * 1.2

        ax_f[col].bar(np.arange(N) - 0.15, ch_a, width=0.3, alpha=0.7,
                       color="steelblue", label=f"'{i2c[char_a]}'")
        ax_f[col].bar(np.arange(N) + 0.15, ch_b, width=0.3, alpha=0.7,
                       color="indianred", label=f"'{i2c[char_b]}'")
        ax_f[col].set_ylim(-vmax_f, vmax_f)
        suffix = ""
        if mode == "listen_think":
            suffix = " (LISTEN)" if t < LISTEN_TICKS else " (THINK)"
        ax_f[col].set_title(f"{mode}{suffix}")
        ax_f[col].set_xlabel("Neuron")
        ax_f[col].set_ylabel("Charge")
        ax_f[col].legend(fontsize=8)
    plt.tight_layout()
    tmp = os.path.join(OUT_DIR, f"_tmp_{t}.png")
    fig_f.savefig(tmp, dpi=100, bbox_inches="tight")
    frames.append(imageio.imread(tmp))
    plt.close(fig_f)
    os.remove(tmp)

gif_path = os.path.join(OUT_DIR, "animation.gif")
imageio.mimsave(gif_path, frames, duration=1.0, loop=0)
print(f"  Saved: {gif_path}", flush=True)

# ============================================================
# VERDICT
# ============================================================
print(f"\n{'='*60}", flush=True)
print("  VERDICT", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Mean cosine distance (higher = better separation):", flush=True)
for m in modes:
    print(f"    {m:15s}: {mean_dists[m]:.4f}", flush=True)
best = max(modes, key=lambda m: mean_dists[m])
print(f"  Winner: {best}", flush=True)
print(f"\n  Positive margins:", flush=True)
for i, m in enumerate(modes):
    print(f"    {m:15s}: {pos_counts[i]}/{min(VOCAB, V)}", flush=True)
print(f"{'='*60}", flush=True)
print("Done!", flush=True)
