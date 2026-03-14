"""
Linear Probe Test — Is the Readout the Bottleneck?
====================================================
Fit a linear probe on frozen charge vectors from 3 encoding modes.
If temporal + probe ~ spatial + probe → readout is the bottleneck.

Metrics:
- Training accuracy (all 27 samples)
- Leave-one-out cross-validation accuracy
- Lambda sweep for regularization
- Sanity checks (random baseline, determinism, charge magnitudes)
- PCA visualization + confusion matrices
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v22_best_config import SelfWiringGraph
from collections import Counter

OUT_DIR = os.path.join(os.path.dirname(__file__), 'viz_interference', 'probe_results')
os.makedirs(OUT_DIR, exist_ok=True)

# --- Constants ---
V = 8
N = 64
TICKS = 8
LISTEN_TICKS = 4

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

MODES = ["spatial", "temporal", "listen_think"]
SEEDS = [42, 77, 123]
LAMBDAS = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]


def train_network(seed, budget=2000):
    """Train a network and return it frozen."""
    np.random.seed(seed)
    random.seed(seed)
    net = SelfWiringGraph(N, V)
    targets_perm = np.random.permutation(V)
    best_acc = 0.0
    for att in range(budget):
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
    return net, best_acc


def collect_charges(net, mode):
    """Run all VOCAB chars through the network in given mode.
    Returns (VOCAB, N) charge matrix."""
    Weff = net.W * net.mask
    spike_trains = np.zeros((VOCAB, TICKS, V), dtype=np.float32)
    for ci in range(VOCAB):
        for t in range(TICKS):
            bit_idx = t % V
            spike_trains[ci, t, bit_idx] = float((ci >> bit_idx) & 1)

    charges = np.zeros((VOCAB, N), dtype=np.float32)
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
        charges[ci] = ch.copy()
    return charges


def ridge_probe(X, Y_onehot, lam):
    """Fit ridge regression: W = (X^T X + lam*I)^{-1} X^T Y"""
    n_features = X.shape[1]
    XtX = X.T @ X + lam * np.eye(n_features, dtype=np.float32)
    XtY = X.T @ Y_onehot
    W = np.linalg.solve(XtX, XtY)
    return W


def train_accuracy(X, Y, W):
    """Training accuracy: argmax(X @ W) vs Y"""
    preds = np.argmax(X @ W, axis=1)
    return (preds == Y).mean()


def loo_accuracy(X, Y, Y_onehot, lam):
    """Leave-one-out cross-validation accuracy."""
    n = len(Y)
    correct = 0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train = X[mask]
        Y_train = Y_onehot[mask]
        X_test = X[i:i+1]
        W = ridge_probe(X_train, Y_train, lam)
        pred = np.argmax(X_test @ W, axis=1)[0]
        if pred == Y[i]:
            correct += 1
    return correct / n


# ============================================================
# MAIN
# ============================================================
print("=" * 70, flush=True)
print("  LINEAR PROBE TEST", flush=True)
print("=" * 70, flush=True)

# --- Sanity check 1: Determinism ---
print("\n[SANITY] Determinism check...", flush=True)
net_det, _ = train_network(42)
ch1 = collect_charges(net_det, "spatial")
ch2 = collect_charges(net_det, "spatial")
det_ok = np.array_equal(ch1, ch2)
print(f"  Same input -> same output: {det_ok}", flush=True)
if not det_ok:
    print("  FATAL: forward pass is NOT deterministic! Aborting.", flush=True)
    sys.exit(1)

# --- Sanity check 2: Charge magnitudes ---
print("\n[SANITY] Charge magnitudes...", flush=True)
for mode in MODES:
    ch = collect_charges(net_det, mode)
    print(f"  {mode:15s}: min={ch.min():.4f} max={ch.max():.4f} "
          f"mean_abs={np.abs(ch).mean():.4f} nonzero={np.sum(np.abs(ch)>0.001)}/{ch.size}",
          flush=True)

# --- Main experiment: 3 seeds x 3 modes ---
print(f"\n[PROBE] Running {len(SEEDS)} seeds x {len(MODES)} modes...", flush=True)

Y = TARGETS
Y_onehot = np.zeros((VOCAB, VOCAB), dtype=np.float32)
for i in range(VOCAB):
    Y_onehot[i, Y[i]] = 1.0

all_results = {}  # (mode, seed) -> {train_acc, loo_acc, best_lambda, ...}

for seed in SEEDS:
    print(f"\n  --- Seed {seed} ---", flush=True)
    net, net_acc = train_network(seed)
    print(f"  Network trained: acc={net_acc*100:.0f}%", flush=True)

    for mode in MODES:
        X = collect_charges(net, mode)

        # Lambda sweep
        best_loo = -1
        best_lam = 0
        sweep_results = []
        for lam in LAMBDAS:
            W = ridge_probe(X, Y_onehot, lam)
            t_acc = train_accuracy(X, Y, W)
            l_acc = loo_accuracy(X, Y, Y_onehot, lam)
            sweep_results.append((lam, t_acc, l_acc))
            if l_acc > best_loo:
                best_loo = l_acc
                best_lam = lam

        # Best result
        W_best = ridge_probe(X, Y_onehot, best_lam)
        t_acc = train_accuracy(X, Y, W_best)
        preds = np.argmax(X @ W_best, axis=1)

        key = (mode, seed)
        all_results[key] = {
            'train_acc': t_acc,
            'loo_acc': best_loo,
            'best_lambda': best_lam,
            'preds': preds,
            'sweep': sweep_results,
            'X': X,
        }
        print(f"    {mode:15s}: train={t_acc*100:5.1f}%  LOO={best_loo*100:5.1f}%  "
              f"lambda={best_lam}", flush=True)

# --- Sanity check 3: Random baseline ---
print(f"\n[SANITY] Random baseline (shuffled Y, seed=42 network)...", flush=True)
net_rand, _ = train_network(42)
for mode in MODES:
    X = collect_charges(net_rand, mode)
    Y_shuf = np.random.permutation(Y)
    Y_oh_shuf = np.zeros((VOCAB, VOCAB), dtype=np.float32)
    for i in range(VOCAB):
        Y_oh_shuf[i, Y_shuf[i]] = 1.0
    # Use best lambda from real experiment
    best_lam = all_results[(mode, 42)]['best_lambda']
    W_rand = ridge_probe(X, Y_oh_shuf, best_lam)
    rand_loo = loo_accuracy(X, Y_shuf, Y_oh_shuf, best_lam)
    print(f"    {mode:15s}: random LOO={rand_loo*100:5.1f}% (expected ~3.7%)", flush=True)

# ============================================================
# VISUALIZATION
# ============================================================
print("\n[VIZ] Generating plots...", flush=True)

# --- 1. Probe accuracy bar chart ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Linear Probe Accuracy", fontsize=14, fontweight="bold")

for ax_idx, metric in enumerate(['train_acc', 'loo_acc']):
    ax = axes[ax_idx]
    title = "Training Accuracy" if metric == 'train_acc' else "Leave-One-Out CV"
    means = []
    stds = []
    for mode in MODES:
        vals = [all_results[(mode, s)][metric] for s in SEEDS]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    bars = ax.bar(MODES, means, yerr=stds, capsize=5,
                  color=["steelblue", "indianred", "seagreen"])
    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(1/VOCAB, color="gray", linestyle="--", label=f"Random ({100/VOCAB:.1f}%)")
    ax.legend()
    for i, m in enumerate(means):
        ax.text(i, m + stds[i] + 0.02, f"{m*100:.1f}%", ha="center", fontweight="bold")

plt.tight_layout()
path = os.path.join(OUT_DIR, "probe_accuracy.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}", flush=True)

# --- 2. Lambda sweep ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Lambda Sweep (LOO accuracy vs regularization)", fontsize=14, fontweight="bold")
for idx, mode in enumerate(MODES):
    ax = axes[idx]
    for seed in SEEDS:
        sweep = all_results[(mode, seed)]['sweep']
        lams = [s[0] for s in sweep]
        loos = [s[2] for s in sweep]
        ax.plot(range(len(lams)), loos, '-o', markersize=4, label=f"seed={seed}")
    ax.set_xticks(range(len(LAMBDAS)))
    ax.set_xticklabels([str(l) for l in LAMBDAS], fontsize=7)
    ax.set_xlabel("Lambda")
    ax.set_ylabel("LOO accuracy")
    ax.set_title(mode, fontweight="bold")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
path = os.path.join(OUT_DIR, "probe_lambda_sweep.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}", flush=True)

# --- 3. PCA visualization ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("PCA of Charge Vectors (seed=42, color=target char)", fontsize=14, fontweight="bold")
for idx, mode in enumerate(MODES):
    X = all_results[(mode, 42)]['X']
    # Manual PCA (no sklearn)
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    pc = X_centered @ Vt[:2].T  # project onto first 2 PCs
    var_explained = S[:2] ** 2 / (S ** 2).sum()

    ax = axes[idx]
    scatter = ax.scatter(pc[:, 0], pc[:, 1], c=Y, cmap="tab20", s=80, edgecolors="black")
    for i in range(VOCAB):
        ax.annotate(i2c[i], (pc[i, 0], pc[i, 1]), fontsize=7, ha="center", va="bottom")
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)")
    ax.set_title(mode, fontweight="bold")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
path = os.path.join(OUT_DIR, "probe_charge_pca.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}", flush=True)

# --- 4. Confusion matrix (seed=42) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Confusion Matrix (seed=42, best lambda)", fontsize=14, fontweight="bold")
for idx, mode in enumerate(MODES):
    preds = all_results[(mode, 42)]['preds']
    conf = np.zeros((VOCAB, VOCAB), dtype=int)
    for i in range(VOCAB):
        conf[Y[i], preds[i]] += 1
    ax = axes[idx]
    ax.imshow(conf, cmap="Blues")
    acc = (preds == Y).mean()
    ax.set_title(f"{mode} (acc={acc*100:.1f}%)", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.tight_layout()
path = os.path.join(OUT_DIR, "probe_confusion.png")
plt.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {path}", flush=True)

# ============================================================
# VERDICT
# ============================================================
print(f"\n{'='*70}", flush=True)
print("  VERDICT", flush=True)
print(f"{'='*70}", flush=True)

for mode in MODES:
    train_vals = [all_results[(mode, s)]['train_acc'] for s in SEEDS]
    loo_vals = [all_results[(mode, s)]['loo_acc'] for s in SEEDS]
    print(f"  {mode:15s}: train={np.mean(train_vals)*100:5.1f}% +/- {np.std(train_vals)*100:.1f}%  "
          f"LOO={np.mean(loo_vals)*100:5.1f}% +/- {np.std(loo_vals)*100:.1f}%", flush=True)

# Compare spatial vs temporal
s_loo = np.mean([all_results[('spatial', s)]['loo_acc'] for s in SEEDS])
t_loo = np.mean([all_results[('temporal', s)]['loo_acc'] for s in SEEDS])
gap = abs(s_loo - t_loo) * 100

print(f"\n  Spatial LOO: {s_loo*100:.1f}%")
print(f"  Temporal LOO: {t_loo*100:.1f}%")
print(f"  Gap: {gap:.1f}%")

if gap < 10:
    print(f"\n  >>> READOUT BOTTLENECK CONFIRMED <<<")
    print(f"  >>> Temporal info is there, the network just can't decode it.")
elif s_loo > t_loo:
    print(f"\n  >>> Spatial genuinely better ({gap:.1f}% gap)")
    print(f"  >>> Temporal loses information, not just readout issue.")
else:
    print(f"\n  >>> Temporal WINS? ({gap:.1f}% gap)")
    print(f"  >>> Surprising — temporal representations more linearly decodable!")

print(f"\n{'='*70}", flush=True)
print("Done!", flush=True)
