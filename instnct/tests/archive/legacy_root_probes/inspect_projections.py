"""
Inspect actual projection matrices — what does the data look like?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

V = 27
H = 81

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=120)

# ══════════════════════════════════════════════════════
# 1. What is the INPUT to the system?
# ══════════════════════════════════════════════════════
print("=" * 80)
print("  MI A BEMENET?")
print("=" * 80)
print(f"\n  A hálózat V={V} 'szót' ismer.")
print(f"  Egy input = one-hot vektor, hossza V={V}.")
print(f"\n  Pl. input[3] (a 3-as szó):")
inp = np.zeros(V)
inp[3] = 1.0
print(f"  {inp}")
print(f"  ^ Pontosan 1 db egyes, a többi nulla. Méret: ({V},)")

# ══════════════════════════════════════════════════════
# 2. input_projection matrices for different projections
# ══════════════════════════════════════════════════════
print(f"\n\n{'=' * 80}")
print(f"  input_projection MÁTRIXOK ÖSSZEHASONLÍTÁSA")
print(f"  Méret: ({V}, {H}) — minden input szóhoz {H} hidden neuron értéke")
print(f"{'=' * 80}")

# --- Identity ---
W_id = np.zeros((V, H))
W_id[np.arange(V), np.arange(V)] = 1.0

# --- Hadamard (symmetric) ---
rng = np.random.RandomState(42)
A = rng.randn(H, V)
Q, _ = np.linalg.qr(A)
W_had = Q[:, :V].T  # V × H

# --- Random 1x ---
rng2 = np.random.RandomState(42)
W_rnd = rng2.randn(V, H)
W_rnd /= np.linalg.norm(W_rnd, axis=1, keepdims=True)

# --- Random 3x (THE WINNER) ---
W_3x = W_rnd * 3.0

projections = {
    'identity':    W_id,
    'hadamard':    W_had,
    'random-1x':   W_rnd,
    'random-3x':   W_3x,
}

for name, W in projections.items():
    print(f"\n  ── {name} ──")
    print(f"  Shape: {W.shape}")
    print(f"  Értéktartomány: [{W.min():.4f}, {W.max():.4f}]")
    print(f"  Átlag abs érték: {np.abs(W).mean():.4f}")
    print(f"  Nulla elemek: {(W == 0).sum()}/{W.size} ({(W == 0).mean()*100:.0f}%)")
    print(f"  Sor-normák (L2): min={np.linalg.norm(W, axis=1).min():.3f} "
          f"max={np.linalg.norm(W, axis=1).max():.3f} "
          f"mean={np.linalg.norm(W, axis=1).mean():.3f}")

    # Show first 3 rows (= first 3 input words), first 12 columns (= first 12 hidden neurons)
    print(f"\n  Első 3 input szó → első 12 hidden neuron:")
    print(f"  {'':>10s}", end="")
    for h in range(12):
        print(f"  h{h:<3d}", end="")
    print()
    for v in range(3):
        print(f"  input[{v}] ", end="")
        for h in range(12):
            val = W[v, h]
            if val == 0:
                print(f"  {'·':>4s}", end="")
            else:
                print(f"  {val:+.2f}", end="")
        print(f"  ...")

# ══════════════════════════════════════════════════════
# 3. What happens when input hits the hidden layer?
# ══════════════════════════════════════════════════════
print(f"\n\n{'=' * 80}")
print(f"  MI TÖRTÉNIK AMIKOR AZ INPUT BEÉRKEZIK? (tick 0)")
print(f"  input[3] @ input_projection = milyen activation a {H} hidden neuronban?")
print(f"{'=' * 80}")

THRESHOLD = 0.5
inp = np.zeros(V)
inp[3] = 1.0

for name, W in projections.items():
    activation = inp @ W  # shape: (H,)
    above_thresh = (np.abs(activation) > THRESHOLD).sum()
    above_01 = (np.abs(activation) > 0.1).sum()
    nonzero = (activation != 0).sum()

    print(f"\n  ── {name} ──")
    print(f"  Injection vektor (input[3] @ input_projection):")
    print(f"    Nonzero elemek: {nonzero}/{H}")
    print(f"    |érték| > 0.1:  {above_01}/{H}")
    print(f"    |érték| > 0.5 (THRESHOLD): {above_thresh}/{H}")
    print(f"    min={activation.min():.4f}  max={activation.max():.4f}  "
          f"mean_abs={np.abs(activation).mean():.4f}")

    # Show the actual values for first 20 hidden neurons
    print(f"    Első 20 hidden neuron értéke:")
    print(f"    ", end="")
    for h in range(20):
        val = activation[h]
        if val == 0:
            print(f" {'·':>5s}", end="")
        elif abs(val) > THRESHOLD:
            print(f" {val:+.2f}*", end="")  # * = above threshold
        else:
            print(f" {val:+.3f}", end="")
    print(f" ...")

# ══════════════════════════════════════════════════════
# 4. After 1 tick of propagation
# ══════════════════════════════════════════════════════
print(f"\n\n{'=' * 80}")
print(f"  1 TICK UTÁN: charge és firing állapot")
print(f"  (injection → matmul mask → charge → threshold → fire?)")
print(f"{'=' * 80}")

# Build a simple mask for demonstration
np.random.seed(42)
DRIVE = 0.6
d = 0.04
r = np.random.rand(H, H)
mask = np.zeros((H, H), dtype=np.float32)
mask[r < d/2] = -DRIVE
mask[r > 1 - d/2] = DRIVE
np.fill_diagonal(mask, 0)

for name, W in projections.items():
    inp = np.zeros(V, dtype=np.float32)
    inp[3] = 1.0

    # Tick 0
    act = (inp @ W).astype(np.float32)
    # Tick 1
    raw = act @ mask
    charge = raw * 0.99  # retain
    firing = np.maximum(charge - THRESHOLD, 0)
    charge = np.clip(charge, -1, 1)

    n_fire_t0 = (act > THRESHOLD).sum()
    n_fire_t1 = (firing > 0).sum()

    print(f"\n  ── {name} ──")
    print(f"    Tick 0: {n_fire_t0:2d}/{H} neuron tüzel (injection > threshold)")
    print(f"    Tick 1: {n_fire_t1:2d}/{H} neuron tüzel (propagáció után)")
    print(f"    Charge range: [{charge.min():.3f}, {charge.max():.3f}]")

# ══════════════════════════════════════════════════════
# 5. output_projection: how does output reading work?
# ══════════════════════════════════════════════════════
print(f"\n\n{'=' * 80}")
print(f"  output_projection: HOGYAN OLVASSUK KI AZ OUTPUTOT?")
print(f"  hidden charge (H={H},) @ output_projection ({H},{V}) = logits ({V},)")
print(f"{'=' * 80}")

# Simulate a fake final charge
fake_charge = np.random.randn(H).astype(np.float32) * 0.3

for name in ['random-1x', 'random-3x']:
    W = projections[name]
    rng_out = np.random.RandomState(42)
    output_projection = rng_out.randn(H, V).astype(np.float32)
    output_projection /= np.linalg.norm(output_projection, axis=0, keepdims=True)
    if '3x' in name:
        output_projection = output_projection * 3.0

    logits = fake_charge @ output_projection
    print(f"\n  ── {name} output_projection ──")
    print(f"    Logit range: [{logits.min():.3f}, {logits.max():.3f}]")
    print(f"    Logit spread (max-min): {logits.max()-logits.min():.3f}")
    print(f"    Softmax entropy after: ", end="")
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    entropy = -np.sum(p * np.log(p + 1e-10))
    max_entropy = np.log(V)
    print(f"{entropy:.2f} / {max_entropy:.2f} (max)")
    print(f"    → {'ÉLES döntés' if entropy < max_entropy * 0.7 else 'SZÉTKENŐDIK'}")

print(f"\n{'=' * 80}")
print("DONE")
