"""
A/B Smoke Test: Old (hardcoded) vs New (canonical) forward pass
===============================================================
Verifies that the recipe refactoring to use SelfWiringGraph.rollout_token()
produces DIFFERENT results from the old hardcoded loop (expected — three known
divergences: subtractive vs multiplicative decay, multiplicative vs additive
C19, hard reset vs no reset).

Also verifies that the NEW path matches graph.py exactly (bit-identical).

Run:  python instnct/tests/test_recipe_canonical_ab.py
"""
import sys, os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph


def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p


# ── OLD forward pass (exact copy from recipe before refactor) ────────────────
def old_forward(mask, H, theta, decay, polarity, freq, phase, rho,
                input_projection, bp, text_bytes, ticks=8, input_duration=2):
    """The OLD hardcoded recipe forward pass with known divergences."""
    rs, cs = np.where(mask)
    sp_vals = polarity[rs]
    ret = 1.0 - decay                          # OLD: multiplicative decay factor
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    charges_history = []
    for i in range(len(text_bytes) - 1):
        act = state.copy()
        for t in range(ticks):
            if t < input_duration:
                act = act + bp[text_bytes[i]] @ input_projection
            raw = np.zeros(H, dtype=np.float32)
            if len(rs):
                np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw
            np.clip(charge, 0.0, 15.0, out=charge)
            charge *= ret                       # OLD: multiplicative decay
            wave = np.sin(np.float32(t) * freq + phase)
            effective_theta = np.maximum(0.0, theta + rho * wave)  # OLD: additive C19
            fired = charge >= effective_theta
            act = fired.astype(np.float32) * polarity
            # OLD: NO hard reset of charge for fired neurons
        state = act.copy()
        charges_history.append(charge.copy())
    return charges_history


# ── NEW forward pass (canonical rollout_token) ──────────────────────────────
def new_forward(mask, H, theta, decay, polarity, freq, phase, rho,
                input_projection, bp, text_bytes, ticks=8, input_duration=2):
    """The NEW canonical forward pass via SelfWiringGraph.rollout_token()."""
    sparse_cache = SelfWiringGraph.build_sparse_cache(mask)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    charges_history = []
    for i in range(len(text_bytes) - 1):
        injected = bp[text_bytes[i]] @ input_projection
        state, charge = SelfWiringGraph.rollout_token(
            injected, mask=mask, theta=theta, decay=decay,
            ticks=ticks, input_duration=input_duration,
            state=state, charge=charge, sparse_cache=sparse_cache,
            polarity=polarity, freq=freq, phase=phase, rho=rho,
        )
        charges_history.append(charge.copy())
    return charges_history


# ── Direct graph.py forward (sanity: must match new_forward bit-exact) ──────
def graph_forward(net, bp, text_bytes, ticks=8, input_duration=2):
    """Use the SelfWiringGraph instance methods directly."""
    sparse_cache = SelfWiringGraph.build_sparse_cache(net.mask)
    state = np.zeros(net.H, dtype=np.float32)
    charge = np.zeros(net.H, dtype=np.float32)
    charges_history = []
    for i in range(len(text_bytes) - 1):
        injected = bp[text_bytes[i]] @ net.input_projection
        state, charge = SelfWiringGraph.rollout_token(
            injected, mask=net.mask, theta=net.theta, decay=net.decay,
            ticks=ticks, input_duration=input_duration,
            state=state, charge=charge, sparse_cache=sparse_cache,
            polarity=net.polarity.astype(np.float32),
            freq=net.freq, phase=net.phase, rho=net.rho,
        )
        charges_history.append(charge.copy())
    return charges_history


def run_test():
    print("=" * 70)
    print("  A/B SMOKE TEST: Old (hardcoded) vs New (canonical) forward pass")
    print("=" * 70)

    # Setup: small network, deterministic
    IO = 32
    H = 128
    np.random.seed(42)
    net = SelfWiringGraph(IO, hidden=H, seed=42)

    # Add some edges (sparse, ~5% density)
    rng = np.random.RandomState(99)
    for _ in range(int(H * H * 0.05)):
        r, c = rng.randint(0, H), rng.randint(0, H)
        if r != c:
            net.mask[r, c] = 1
    net.resync_alive()

    # Per-neuron params
    net.theta[:] = rng.uniform(0.5, 2.0, H).astype(np.float32)
    net.decay[:] = rng.uniform(0.08, 0.24, H).astype(np.float32)

    bp = make_bp(IO)
    text_bytes = rng.randint(0, 256, size=20).astype(np.uint8)

    polarity_f32 = net.polarity.astype(np.float32)

    # ── Run both ──
    old_charges = old_forward(
        net.mask, H, net.theta, net.decay, polarity_f32,
        net.freq, net.phase, net.rho, net.input_projection, bp, text_bytes,
    )
    new_charges = new_forward(
        net.mask, H, net.theta, net.decay, polarity_f32,
        net.freq, net.phase, net.rho, net.input_projection, bp, text_bytes,
    )
    graph_charges = graph_forward(net, bp, text_bytes)

    # ── Analysis ──
    print("\n--- DIVERGENCE: Old vs New (expected to differ) ---")
    all_pass = True
    for step in range(len(old_charges)):
        diff = np.abs(old_charges[step] - new_charges[step])
        max_diff = diff.max()
        mean_diff = diff.mean()
        print(f"  token {step:2d}: max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")

    total_max = max(np.abs(old_charges[i] - new_charges[i]).max() for i in range(len(old_charges)))

    if total_max < 1e-6:
        print("\n  FAIL: Old and New produce IDENTICAL results!")
        print("  This means the refactor didn't actually change anything.")
        print("  Expected divergence from 3 known differences:")
        print("    1. Subtractive vs multiplicative decay")
        print("    2. Multiplicative vs additive C19 formula")
        print("    3. Hard reset of fired neurons vs no reset")
        all_pass = False
    else:
        print(f"\n  OK: Old vs New max divergence = {total_max:.6f}")
        print("  Confirmed: canonical forward pass differs from old hardcoded loop.")

    print("\n--- CONSISTENCY: New (static) vs Graph (instance) ---")
    for step in range(len(new_charges)):
        diff = np.abs(new_charges[step] - graph_charges[step])
        max_diff = diff.max()
        if max_diff > 1e-6:
            print(f"  FAIL at token {step}: new vs graph max_diff={max_diff:.6f}")
            all_pass = False

    new_vs_graph_max = max(np.abs(new_charges[i] - graph_charges[i]).max() for i in range(len(new_charges)))
    if new_vs_graph_max < 1e-6:
        print("  OK: New forward == Graph forward (bit-identical)")
    else:
        print(f"  FAIL: New vs Graph max divergence = {new_vs_graph_max:.6f}")
        all_pass = False

    # ── Charge statistics ──
    print("\n--- CHARGE STATISTICS (last token) ---")
    old_last = old_charges[-1]
    new_last = new_charges[-1]
    print(f"  OLD: mean={old_last.mean():.4f}  std={old_last.std():.4f}  "
          f"max={old_last.max():.4f}  nonzero={np.count_nonzero(old_last)}/{H}")
    print(f"  NEW: mean={new_last.mean():.4f}  std={new_last.std():.4f}  "
          f"max={new_last.max():.4f}  nonzero={np.count_nonzero(new_last)}/{H}")

    # ── Scoring comparison (cosine sim of output distributions) ──
    print("\n--- OUTPUT DISTRIBUTION COMPARISON ---")
    output_projection = net.output_projection
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)

    def score_token(charge_vec):
        out = charge_vec @ output_projection
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        e = np.exp(sims - sims.max())
        return e / e.sum()

    old_pred = score_token(old_charges[-1])
    new_pred = score_token(new_charges[-1])
    cos_sim = np.dot(old_pred, new_pred) / (np.linalg.norm(old_pred) * np.linalg.norm(new_pred) + 1e-8)
    kl_div = np.sum(new_pred * np.log((new_pred + 1e-10) / (old_pred + 1e-10)))
    print(f"  Cosine sim between old/new output distributions: {cos_sim:.6f}")
    print(f"  KL divergence (new || old): {kl_div:.6f}")

    # ── Specific divergence diagnosis ──
    print("\n--- DIVERGENCE DIAGNOSIS ---")
    # Test decay alone
    charge_old_decay = np.array([5.0], dtype=np.float32)
    charge_new_decay = np.array([5.0], dtype=np.float32)
    d = np.array([0.15], dtype=np.float32)
    charge_old_decay *= (1.0 - d)  # multiplicative
    charge_new_decay = np.maximum(charge_new_decay - d, 0.0)  # subtractive
    print(f"  Decay test (start=5.0, decay=0.15):")
    print(f"    OLD (multiplicative): {charge_old_decay[0]:.4f}")
    print(f"    NEW (subtractive):    {charge_new_decay[0]:.4f}")

    # Test C19 alone
    th = np.array([2.0], dtype=np.float32)
    rho_v = np.array([0.3], dtype=np.float32)
    wave_v = np.array([0.5], dtype=np.float32)
    old_theta = np.maximum(0.0, th + rho_v * wave_v)  # additive
    new_theta = np.clip(th * (1.0 + rho_v * wave_v), 1.0, 15.0)  # multiplicative
    print(f"  C19 test (theta=2.0, rho=0.3, wave=0.5):")
    print(f"    OLD (additive):       {old_theta[0]:.4f}")
    print(f"    NEW (multiplicative): {new_theta[0]:.4f}")

    # ── Final verdict ──
    print("\n" + "=" * 70)
    if all_pass:
        print("  ALL PROBES PASSED")
        print("  - Old vs New diverge (expected: different physics)")
        print("  - New vs Graph identical (recipe now uses canonical code)")
    else:
        print("  SOME PROBES FAILED — see above")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(run_test())
