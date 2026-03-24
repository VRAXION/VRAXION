"""
Knob-Conditioned SelfWiringGraph Prototype
===========================================
Core idea: physical parameters are NOT hidden weights — they're explicit
input neurons ("knobs") that the network wires TO and learns the EFFECT of.

Traditional:  input → [black box where add_rate is somewhere in weights] → output
This:         input = [data, add_rate=10, coupling=0.3] → [network learns edges TO knobs] → output
                        ↑ explicit, changeable                    ↑ edges carry the meaning

The network doesn't learn WHAT the parameter is — it learns what to DO with it.
Change the knob → output changes instantly through learned edges.

Knob edges use int8 weights (-128..+127) with a scale factor to convert to
float for the mask. This matches the int8 weight scheme used in the main model.
"""

import numpy as np
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.graph import SelfWiringGraph


# ---------------------------------------------------------------------------
# KnobConditionedGraph: wraps SelfWiringGraph with explicit parameter inputs
# ---------------------------------------------------------------------------

class KnobConditionedGraph:
    """
    Extends SelfWiringGraph with dedicated 'knob' neurons that receive
    explicit parameter values. The knob neurons are part of the hidden
    layer but get clamped to knob values each step.

    Knob edges use uint8 magnitude (0..255) + binary sign flag.
    Scale factor converts uint8 → float when writing to the graph mask.
    Default init value: 1 (matching uint8 granularity).
    """

    UINT8_SCALE = 1.0 / 255.0  # uint8 → float: 1 → ~0.00392, 255 → 1.0

    def __init__(self, vocab, knob_names, *, hidden_ratio=3, seed=42, **kwargs):
        self.knob_names = list(knob_names)
        self.n_knobs = len(self.knob_names)
        self.knob_index = {name: i for i, name in enumerate(self.knob_names)}

        self.graph = SelfWiringGraph(
            vocab, hidden_ratio=hidden_ratio, seed=seed, **kwargs
        )

        # Reserve first N hidden neurons as knob neurons
        self.knob_neuron_ids = list(range(self.n_knobs))
        self.knob_values = np.zeros(self.n_knobs, dtype=np.float32)

        # Knob edge storage: magnitude (uint8 0..255) + sign (bool: False=pos, True=neg)
        H = self.graph.H
        self.knob_magnitudes = np.zeros((self.n_knobs, H), dtype=np.uint8)   # 0..255
        self.knob_signs = np.zeros((self.n_knobs, H), dtype=np.bool_)        # False=+, True=-

        # Initialize from existing mask (convert float → uint8 magnitude + binary sign)
        for k in range(self.n_knobs):
            row = self.graph.mask[k, :]
            self.knob_signs[k, :] = row < 0
            self.knob_magnitudes[k, :] = np.clip(
                np.round(np.abs(row) / self.UINT8_SCALE), 0, 255
            ).astype(np.uint8)

        # Write back to ensure mask matches uint8 representation exactly
        self._sync_knob_mask()

    def _sign_mul(self, sign):
        """Binary sign → float multiplier: False → +1.0, True → -1.0"""
        return -1.0 if sign else 1.0

    def _sync_knob_mask(self):
        """Write uint8 magnitude * binary sign back to the float32 graph mask."""
        for k in range(self.n_knobs):
            sign_mul = np.where(self.knob_signs[k, :], -1.0, 1.0).astype(np.float32)
            self.graph.mask[k, :] = (
                self.knob_magnitudes[k, :].astype(np.float32) * sign_mul * self.UINT8_SCALE
            )
            self.graph.mask[k, k] = 0.0  # no self-loops
        self.graph.resync_alive()

    def set_knob_edge(self, knob_id, target, magnitude, sign=None):
        """Set a single knob edge: magnitude (0..255) and optional sign (bool)."""
        self.knob_magnitudes[knob_id, target] = np.uint8(min(magnitude, 255))
        if sign is not None:
            self.knob_signs[knob_id, target] = bool(sign)
        val = (float(self.knob_magnitudes[knob_id, target])
               * self._sign_mul(self.knob_signs[knob_id, target])
               * self.UINT8_SCALE)
        self.graph.mask[knob_id, target] = val
        if target == knob_id:
            self.graph.mask[knob_id, target] = 0.0

    def mutate_knob_edge(self, knob_id, target, rng=None):
        """Mutate magnitude, returns (old_mag, old_sign) for undo.

        Full-range random: pick any value 0..255. The selection pressure
        (keep-if-better) finds the right magnitude. This gives uint8 the
        same bold exploration as ternary, plus 256x finer resolution.
        """
        _rng = rng if rng is not None else np.random
        old_mag = int(self.knob_magnitudes[knob_id, target])
        old_sign = bool(self.knob_signs[knob_id, target])
        new_mag = int(_rng.randint(0, 256))
        self.set_knob_edge(knob_id, target, new_mag)
        return old_mag, old_sign

    def flip_knob_edge(self, knob_id, target):
        """Flip the binary sign of a knob edge (like the main model's _flip)."""
        old_sign = bool(self.knob_signs[knob_id, target])
        self.knob_signs[knob_id, target] = not old_sign
        val = (float(self.knob_magnitudes[knob_id, target])
               * self._sign_mul(not old_sign)
               * self.UINT8_SCALE)
        self.graph.mask[knob_id, target] = val
        return old_sign

    def set_knobs(self, **kwargs):
        for name, val in kwargs.items():
            if name not in self.knob_index:
                raise ValueError(f"Unknown knob: {name}. Available: {self.knob_names}")
            self.knob_values[self.knob_index[name]] = np.float32(val)

    def get_knob(self, name):
        return float(self.knob_values[self.knob_index[name]])

    def _clamp_knobs(self):
        """Inject knob values into their dedicated hidden neurons."""
        for i, val in enumerate(self.knob_values):
            self.graph.state[i] = val
            self.graph.charge[i] = val

    def forward(self, world_vec, ticks=6):
        """Forward with knob clamping: clamp knobs, then normal SWG forward."""
        self._clamp_knobs()
        return self.graph.forward(world_vec, ticks=ticks)

    def forward_token(self, token_id, ticks=6):
        """Forward for a single token index."""
        world = np.zeros(self.graph.V, dtype=np.float32)
        world[token_id] = 1.0
        return self.forward(world, ticks=ticks)

    def reset(self):
        self.graph.reset()
        self._clamp_knobs()

    @property
    def knob_edge_count(self):
        count = 0
        for k in range(self.n_knobs):
            count += int(np.count_nonzero(self.knob_magnitudes[k, :]))
        return count

    @property
    def knob_edge_strength(self):
        """Total uint8 magnitude across all knob edges."""
        total = 0
        for k in range(self.n_knobs):
            total += int(np.sum(self.knob_magnitudes[k, :]))
        return total

    def knob_influence_map(self):
        influence = {}
        for name, idx in self.knob_index.items():
            mags = self.knob_magnitudes[idx, :]
            signs = self.knob_signs[idx, :]
            targets = np.where(mags > 0)[0]
            influence[name] = {
                "n_targets": len(targets),
                "mean_magnitude": float(np.mean(mags[targets])) if len(targets) > 0 else 0.0,
                "total_magnitude": int(np.sum(mags)),
                "pos_edges": int(np.sum((mags > 0) & ~signs)),   # sign=False → positive
                "neg_edges": int(np.sum((mags > 0) & signs)),    # sign=True → negative
            }
        return influence


# ---------------------------------------------------------------------------
# Test 1: Knob sensitivity — changing knob changes output
# ---------------------------------------------------------------------------

def test_knob_sensitivity():
    print("=" * 60)
    print("TEST 1: Knob Sensitivity")
    print("=" * 60)

    knobs = ["add_rate", "coupling", "mass", "temperature"]
    kg = KnobConditionedGraph(vocab=32, knob_names=knobs, seed=42)

    kg.set_knobs(add_rate=0, coupling=0, mass=0, temperature=0)
    kg.reset()
    base_out = kg.forward_token(5).copy()

    results = {}
    for knob in knobs:
        kwargs = {k: 0 for k in knobs}
        kwargs[knob] = 1.0
        kg.set_knobs(**kwargs)
        kg.reset()
        out = kg.forward_token(5)
        delta = np.max(np.abs(out - base_out))
        results[knob] = delta
        print(f"  {knob:15s} = 1.0  →  max_delta = {delta:.6f}")

    any_effect = any(d > 1e-6 for d in results.values())
    print(f"\n  Any knob has effect: {any_effect}")
    print(f"  Status: {'PASS' if any_effect else 'FAIL'}")
    return any_effect


# ---------------------------------------------------------------------------
# Test 2: Knob linearity
# ---------------------------------------------------------------------------

def test_knob_linearity():
    print("\n" + "=" * 60)
    print("TEST 2: Knob Linearity (does effect scale with value?)")
    print("=" * 60)

    kg = KnobConditionedGraph(vocab=32, knob_names=["signal"], seed=42)

    kg.set_knobs(signal=0)
    kg.reset()
    base = kg.forward_token(5).copy()

    deltas = []
    values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for v in values:
        kg.set_knobs(signal=v)
        kg.reset()
        out = kg.forward_token(5)
        delta = float(np.sum(np.abs(out - base)))
        deltas.append(delta)
        print(f"  signal = {v:5.1f}  →  total_delta = {delta:.4f}")

    monotonic = all(deltas[i] <= deltas[i + 1] + 1e-8 for i in range(len(deltas) - 1))
    print(f"\n  Monotonic scaling: {monotonic}")
    if deltas[2] > 1e-10:
        print(f"  Ratio 10x/1x: {deltas[-1] / deltas[2]:.2f}x")
    print(f"  Status: {'PASS' if monotonic else 'WARN - non-monotonic'}")
    return monotonic


# ---------------------------------------------------------------------------
# Test 3: Knob independence
# ---------------------------------------------------------------------------

def test_knob_independence():
    print("\n" + "=" * 60)
    print("TEST 3: Knob Independence (different knobs → different effects)")
    print("=" * 60)

    knobs = ["alpha", "beta", "gamma"]
    kg = KnobConditionedGraph(vocab=32, knob_names=knobs, seed=42)

    kg.set_knobs(alpha=0, beta=0, gamma=0)
    kg.reset()
    base = kg.forward_token(5).copy()

    effects = {}
    for knob in knobs:
        kwargs = {k: 0 for k in knobs}
        kwargs[knob] = 1.0
        kg.set_knobs(**kwargs)
        kg.reset()
        out = kg.forward_token(5)
        effects[knob] = out - base

    print("\n  Cosine similarity between knob effects:")
    pairs = [("alpha", "beta"), ("alpha", "gamma"), ("beta", "gamma")]
    cosines = []
    for a, b in pairs:
        ea, eb = effects[a], effects[b]
        na, nb = np.linalg.norm(ea), np.linalg.norm(eb)
        if na > 1e-10 and nb > 1e-10:
            cos = float(np.dot(ea, eb) / (na * nb))
        else:
            cos = 0.0
        cosines.append(abs(cos))
        print(f"    {a} vs {b}: {cos:.4f}")

    all_identical = all(c > 0.999 for c in cosines)
    print(f"\n  All effects identical: {all_identical}")
    print(f"  Status: {'FAIL - knobs not distinguishable' if all_identical else 'PASS'}")
    return not all_identical


# ---------------------------------------------------------------------------
# Test 4: Knob edge wiring map
# ---------------------------------------------------------------------------

def test_knob_wiring():
    print("\n" + "=" * 60)
    print("TEST 4: Knob Wiring Map")
    print("=" * 60)

    knobs = ["add_rate", "coupling", "mass"]
    kg = KnobConditionedGraph(vocab=32, knob_names=knobs, seed=42)

    print(f"  Hidden size: {kg.graph.H}")
    print(f"  Knob neurons: {kg.knob_neuron_ids}")
    print(f"  Total edges from knobs: {kg.knob_edge_count}")
    print(f"  Total edge strength (int8 sum): {kg.knob_edge_strength}")

    influence = kg.knob_influence_map()
    for name, info in influence.items():
        print(f"\n  {name}:")
        print(f"    targets: {info['n_targets']} neurons")
        print(f"    mean magnitude (int8): {info['mean_magnitude']:.1f}")
        print(f"    total magnitude (int8): {info['total_magnitude']}")
        print(f"    pos/neg edges: {info['pos_edges']}/{info['neg_edges']}")

    has_wiring = kg.knob_edge_count > 0
    print(f"\n  Status: {'PASS' if has_wiring else 'FAIL - no edges from knobs'}")
    return has_wiring


# ---------------------------------------------------------------------------
# Test 5: Instant feedback — mutate knob edges, see output shift
# ---------------------------------------------------------------------------

def test_instant_feedback():
    print("\n" + "=" * 60)
    print("TEST 5: Instant Feedback (mutate knob edges → output shifts)")
    print("=" * 60)

    kg = KnobConditionedGraph(vocab=32, knob_names=["control"], seed=42)
    knob_id = kg.knob_neuron_ids[0]
    H = kg.graph.H

    kg.set_knobs(control=1.0)
    kg.reset()
    base = kg.forward_token(5).copy()

    rng = np.random.RandomState(99)
    n_mutations = 10
    shifts = []

    for trial in range(n_mutations):
        target = rng.randint(kg.n_knobs, H)
        old_mag, old_sign = kg.mutate_knob_edge(knob_id, target, rng)
        new_mag = int(kg.knob_magnitudes[knob_id, target])
        kg.graph.resync_alive()

        kg.reset()
        out = kg.forward_token(5)
        shift = float(np.sum(np.abs(out - base)))
        shifts.append(shift)
        print(f"  Mutation {trial+1:2d}: edge [{knob_id}→{target:3d}] "
              f"mag {old_mag}→{new_mag}  output_shift={shift:.4f}")

        # Restore
        kg.set_knob_edge(knob_id, target, old_mag, old_sign)

    kg.graph.resync_alive()

    avg_shift = np.mean(shifts)
    max_shift = np.max(shifts)
    print(f"\n  Avg shift per mutation: {avg_shift:.4f}")
    print(f"  Max shift: {max_shift:.4f}")
    responsive = max_shift > 1e-4
    print(f"  Status: {'PASS' if responsive else 'FAIL - mutations have no effect'}")
    return responsive


# ---------------------------------------------------------------------------
# Test 6: 2D Knob Sweep
# ---------------------------------------------------------------------------

def test_2d_knob_sweep():
    print("\n" + "=" * 60)
    print("TEST 6: 2D Knob Sweep (add_rate × coupling)")
    print("=" * 60)

    kg = KnobConditionedGraph(
        vocab=32, knob_names=["add_rate", "coupling"], seed=42
    )

    values = [0.0, 0.25, 0.5, 0.75, 1.0]
    grid = np.zeros((len(values), len(values)), dtype=np.float32)

    for i, ar in enumerate(values):
        for j, cp in enumerate(values):
            kg.set_knobs(add_rate=ar, coupling=cp)
            kg.reset()
            out = kg.forward_token(5)
            grid[i, j] = float(np.argmax(out))

    print("  Argmax token for (add_rate × coupling):")
    print(f"  {'':>10s}", end="")
    for v in values:
        print(f"  cp={v:.2f}", end="")
    print()
    for i, ar in enumerate(values):
        print(f"  ar={ar:.2f}  ", end="")
        for j in range(len(values)):
            print(f"  {grid[i,j]:5.0f} ", end="")
        print()

    unique_vals = len(np.unique(grid))
    print(f"\n  Unique output tokens: {unique_vals}")
    print(f"  Status: {'PASS' if unique_vals > 1 else 'NEUTRAL - uniform output'}")
    return unique_vals > 1


# ---------------------------------------------------------------------------
# Test 7: Learning loop — evolve knob edges to hit a target
# ---------------------------------------------------------------------------

def test_knob_learning():
    """
    The main event: can the network LEARN to use a knob?
    control=HIGH → output favors token 0
    control=LOW  → output favors token 15
    Method: mutate edges FROM the knob neuron, keep improvements.
    """
    print("\n" + "=" * 60)
    print("TEST 7: Knob Learning (evolve edges to map knob→target)")
    print("=" * 60)

    kg = KnobConditionedGraph(vocab=32, knob_names=["control"], seed=42)
    knob_id = kg.knob_neuron_ids[0]
    H = kg.graph.H
    rng = np.random.RandomState(123)

    TARGET_HIGH = 0
    TARGET_LOW = 15

    def score():
        s = 0.0
        kg.set_knobs(control=1.0)
        kg.reset()
        out_h = kg.forward_token(5)
        s += out_h[TARGET_HIGH] - np.max(np.delete(out_h, TARGET_HIGH))

        kg.set_knobs(control=0.0)
        kg.reset()
        out_l = kg.forward_token(5)
        s += out_l[TARGET_LOW] - np.max(np.delete(out_l, TARGET_LOW))
        return float(s)

    best_score = score()
    print(f"  Initial score: {best_score:.4f}")

    generations = 200
    improvements = 0
    t0 = time.time()

    for gen in range(generations):
        target = rng.randint(kg.n_knobs, H)

        # Randomly choose: mutate magnitude or flip sign
        if rng.rand() < 0.7:
            # Mutate magnitude (increase by 1..5)
            old_mag, old_sign = kg.mutate_knob_edge(knob_id, target, rng)
        else:
            # Flip sign
            old_sign = kg.flip_knob_edge(knob_id, target)
            old_mag = int(kg.knob_magnitudes[knob_id, target])
        kg.graph.resync_alive()

        new_score = score()
        if new_score > best_score:
            best_score = new_score
            improvements += 1
            new_mag = int(kg.knob_magnitudes[knob_id, target])
            new_neg = bool(kg.knob_signs[knob_id, target])
            if improvements <= 10 or improvements % 20 == 0:
                print(f"  Gen {gen:4d}: score {best_score:+.4f} "
                      f"(edge [{knob_id}→{target}] mag={new_mag} {'neg' if new_neg else 'pos'})")
        else:
            kg.set_knob_edge(knob_id, target, old_mag, old_sign)
            kg.graph.resync_alive()

    elapsed = time.time() - t0

    kg.set_knobs(control=1.0)
    kg.reset()
    out_h = kg.forward_token(5)
    pred_high = int(np.argmax(out_h))

    kg.set_knobs(control=0.0)
    kg.reset()
    out_l = kg.forward_token(5)
    pred_low = int(np.argmax(out_l))

    print(f"\n  Final score: {best_score:+.4f}")
    print(f"  Improvements: {improvements}/{generations}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  control=1.0 → argmax={pred_high} (target={TARGET_HIGH})")
    print(f"  control=0.0 → argmax={pred_low} (target={TARGET_LOW})")

    different = pred_high != pred_low
    learned = (pred_high == TARGET_HIGH) or (pred_low == TARGET_LOW)
    print(f"  Outputs differ by knob: {different}")
    print(f"  At least one target hit: {learned}")
    print(f"  Status: {'PASS' if different else 'PARTIAL'}")
    return different


# ---------------------------------------------------------------------------
# Test 8: Multi-knob learning
# ---------------------------------------------------------------------------

def test_multi_knob_learning():
    """Two knobs, each independently steers output."""
    print("\n" + "=" * 60)
    print("TEST 8: Multi-Knob Learning (2 knobs, independent targets)")
    print("=" * 60)

    kg = KnobConditionedGraph(
        vocab=32, knob_names=["knob_a", "knob_b"], seed=42
    )
    H = kg.graph.H
    rng = np.random.RandomState(456)

    TARGET_A = 3
    TARGET_B = 20

    def score():
        s = 0.0
        kg.set_knobs(knob_a=1.0, knob_b=0.0)
        kg.reset()
        out = kg.forward_token(5)
        s += out[TARGET_A] - np.max(np.delete(out, TARGET_A))

        kg.set_knobs(knob_a=0.0, knob_b=1.0)
        kg.reset()
        out = kg.forward_token(5)
        s += out[TARGET_B] - np.max(np.delete(out, TARGET_B))
        return float(s)

    best_score = score()
    print(f"  Initial score: {best_score:.4f}")

    generations = 300
    improvements = 0

    for gen in range(generations):
        knob_id = rng.randint(0, 2)
        target = rng.randint(2, H)

        if rng.rand() < 0.7:
            old_mag, old_sign = kg.mutate_knob_edge(knob_id, target, rng)
        else:
            old_sign = kg.flip_knob_edge(knob_id, target)
            old_mag = int(kg.knob_magnitudes[knob_id, target])
        kg.graph.resync_alive()

        new_score = score()
        if new_score > best_score:
            best_score = new_score
            improvements += 1
        else:
            kg.set_knob_edge(knob_id, target, old_mag, old_sign)
            kg.graph.resync_alive()

    combos = [(0, 0), (1, 0), (0, 1), (1, 1)]
    print(f"\n  Final results after {improvements} improvements:")
    results = []
    for a, b in combos:
        kg.set_knobs(knob_a=float(a), knob_b=float(b))
        kg.reset()
        out = kg.forward_token(5)
        tok = int(np.argmax(out))
        results.append(tok)
        print(f"  knob_a={a}, knob_b={b} → argmax={tok}")

    unique = len(set(results))
    print(f"\n  Unique outputs: {unique}/4")
    print(f"  Status: {'PASS' if unique >= 2 else 'FAIL'}")
    return unique >= 2


# ---------------------------------------------------------------------------
# Test 9: A/B — ternary vs uint8 knob edges
# ---------------------------------------------------------------------------

def _run_ternary_learning(seed, generations=300):
    """Old-style ternary knob learning (baseline)."""
    kg = KnobConditionedGraph(vocab=32, knob_names=["control"], seed=seed)
    knob_id = 0
    H = kg.graph.H
    rng = np.random.RandomState(seed + 1000)
    edge_mag = kg.graph.edge_magnitude

    TARGET_HIGH, TARGET_LOW = 0, 15

    # Zero out knob edges to start fresh (fair comparison)
    for t in range(H):
        kg.knob_magnitudes[knob_id, t] = 0
    kg._sync_knob_mask()

    def score():
        s = 0.0
        kg.set_knobs(control=1.0); kg.reset()
        out = kg.forward_token(5)
        s += out[TARGET_HIGH] - np.max(np.delete(out, TARGET_HIGH))
        kg.set_knobs(control=0.0); kg.reset()
        out = kg.forward_token(5)
        s += out[TARGET_LOW] - np.max(np.delete(out, TARGET_LOW))
        return float(s)

    best = score()
    improvements = 0
    scores_over_time = [best]

    for gen in range(generations):
        target = rng.randint(1, H)
        old_val = float(kg.graph.mask[knob_id, target])

        # Ternary: pick from {-edge_mag, 0, +edge_mag}
        options = [-edge_mag, 0.0, edge_mag]
        new_val = float(rng.choice(options))
        if new_val == old_val:
            scores_over_time.append(best)
            continue

        kg.graph.mask[knob_id, target] = np.float32(new_val)
        kg.graph.resync_alive()

        s = score()
        if s > best:
            best = s
            improvements += 1
        else:
            kg.graph.mask[knob_id, target] = np.float32(old_val)
            kg.graph.resync_alive()
        scores_over_time.append(best)

    kg.set_knobs(control=1.0); kg.reset()
    pred_h = int(np.argmax(kg.forward_token(5)))
    kg.set_knobs(control=0.0); kg.reset()
    pred_l = int(np.argmax(kg.forward_token(5)))
    hit = int(pred_h == TARGET_HIGH) + int(pred_l == TARGET_LOW)

    return best, improvements, hit, pred_h, pred_l, scores_over_time


def _run_uint8_learning(seed, generations=300):
    """New uint8 magnitude + binary sign knob learning."""
    kg = KnobConditionedGraph(vocab=32, knob_names=["control"], seed=seed)
    knob_id = 0
    H = kg.graph.H
    rng = np.random.RandomState(seed + 1000)

    TARGET_HIGH, TARGET_LOW = 0, 15

    # Zero out knob edges to start fresh (fair comparison)
    for t in range(H):
        kg.knob_magnitudes[knob_id, t] = 0
        kg.knob_signs[knob_id, t] = False
    kg._sync_knob_mask()

    def score():
        s = 0.0
        kg.set_knobs(control=1.0); kg.reset()
        out = kg.forward_token(5)
        s += out[TARGET_HIGH] - np.max(np.delete(out, TARGET_HIGH))
        kg.set_knobs(control=0.0); kg.reset()
        out = kg.forward_token(5)
        s += out[TARGET_LOW] - np.max(np.delete(out, TARGET_LOW))
        return float(s)

    best = score()
    improvements = 0
    scores_over_time = [best]

    for gen in range(generations):
        target = rng.randint(1, H)

        if rng.rand() < 0.7:
            old_mag, old_sign = kg.mutate_knob_edge(knob_id, target, rng)
        else:
            old_sign = kg.flip_knob_edge(knob_id, target)
            old_mag = int(kg.knob_magnitudes[knob_id, target])
        kg.graph.resync_alive()

        s = score()
        if s > best:
            best = s
            improvements += 1
        else:
            kg.set_knob_edge(knob_id, target, old_mag, old_sign)
            kg.graph.resync_alive()
        scores_over_time.append(best)

    kg.set_knobs(control=1.0); kg.reset()
    pred_h = int(np.argmax(kg.forward_token(5)))
    kg.set_knobs(control=0.0); kg.reset()
    pred_l = int(np.argmax(kg.forward_token(5)))
    hit = int(pred_h == TARGET_HIGH) + int(pred_l == TARGET_LOW)

    return best, improvements, hit, pred_h, pred_l, scores_over_time


def _run_bitmask_ternary_learning(seed, generations=300):
    """Ternary via 2 boolean masks: exists (0/1) + sign (+1/-1).
    Effective values: {-scale, 0, +scale}
    Mutation: flip one bit at a time (exists or sign).
    """
    kg = KnobConditionedGraph(vocab=32, knob_names=["control"], seed=seed)
    knob_id = 0
    H = kg.graph.H
    rng = np.random.RandomState(seed + 1000)
    scale = kg.graph.edge_magnitude  # same scale as float ternary

    TARGET_HIGH, TARGET_LOW = 0, 15

    # 2 boolean masks for knob edges
    exists = np.zeros(H, dtype=np.bool_)   # is edge present?
    sign = np.zeros(H, dtype=np.bool_)     # False=+1, True=-1

    def apply_masks():
        """Write bitmask ternary to graph mask."""
        for t in range(H):
            if exists[t] and t != knob_id:
                kg.graph.mask[knob_id, t] = scale * (-1.0 if sign[t] else 1.0)
            else:
                kg.graph.mask[knob_id, t] = 0.0
        kg.graph.resync_alive()

    # Zero out knob edges to start fresh
    for t in range(H):
        kg.knob_magnitudes[knob_id, t] = 0
    kg._sync_knob_mask()
    apply_masks()

    def score():
        s = 0.0
        kg.set_knobs(control=1.0); kg.reset()
        out = kg.forward_token(5)
        s += out[TARGET_HIGH] - np.max(np.delete(out, TARGET_HIGH))
        kg.set_knobs(control=0.0); kg.reset()
        out = kg.forward_token(5)
        s += out[TARGET_LOW] - np.max(np.delete(out, TARGET_LOW))
        return float(s)

    best = score()
    improvements = 0
    scores_over_time = [best]

    for gen in range(generations):
        target = rng.randint(1, H)

        # Flip one bit: 50% exists, 50% sign
        if rng.rand() < 0.5:
            # Toggle exists
            old_exists = bool(exists[target])
            exists[target] = not old_exists
            apply_masks()

            s = score()
            if s > best:
                best = s
                improvements += 1
            else:
                exists[target] = old_exists
                apply_masks()
        else:
            # Flip sign
            old_sign = bool(sign[target])
            sign[target] = not old_sign
            apply_masks()

            s = score()
            if s > best:
                best = s
                improvements += 1
            else:
                sign[target] = old_sign
                apply_masks()

        scores_over_time.append(best)

    kg.set_knobs(control=1.0); kg.reset()
    pred_h = int(np.argmax(kg.forward_token(5)))
    kg.set_knobs(control=0.0); kg.reset()
    pred_l = int(np.argmax(kg.forward_token(5)))
    hit = int(pred_h == TARGET_HIGH) + int(pred_l == TARGET_LOW)

    return best, improvements, hit, pred_h, pred_l, scores_over_time


def _run_int_ternary_learning(seed, generations=300):
    """Ternary via int8 array: values in {-1, 0, +1}, pick randomly.
    Effective mask value: int_val * scale.
    """
    kg = KnobConditionedGraph(vocab=32, knob_names=["control"], seed=seed)
    knob_id = 0
    H = kg.graph.H
    rng = np.random.RandomState(seed + 1000)
    scale = kg.graph.edge_magnitude

    TARGET_HIGH, TARGET_LOW = 0, 15

    # int8 ternary storage
    edges = np.zeros(H, dtype=np.int8)  # {-1, 0, +1}

    def apply_edges():
        kg.graph.mask[knob_id, :] = edges.astype(np.float32) * scale
        kg.graph.mask[knob_id, knob_id] = 0.0
        kg.graph.resync_alive()

    # Zero out knob edges to start fresh
    for t in range(H):
        kg.knob_magnitudes[knob_id, t] = 0
    kg._sync_knob_mask()
    apply_edges()

    def score():
        s = 0.0
        kg.set_knobs(control=1.0); kg.reset()
        out = kg.forward_token(5)
        s += out[TARGET_HIGH] - np.max(np.delete(out, TARGET_HIGH))
        kg.set_knobs(control=0.0); kg.reset()
        out = kg.forward_token(5)
        s += out[TARGET_LOW] - np.max(np.delete(out, TARGET_LOW))
        return float(s)

    best = score()
    improvements = 0
    scores_over_time = [best]

    for gen in range(generations):
        target = rng.randint(1, H)
        old_val = int(edges[target])

        # Pick random from {-1, 0, +1}
        new_val = rng.choice([-1, 0, 1])
        if new_val == old_val:
            scores_over_time.append(best)
            continue

        edges[target] = np.int8(new_val)
        apply_edges()

        s = score()
        if s > best:
            best = s
            improvements += 1
        else:
            edges[target] = np.int8(old_val)
            apply_edges()
        scores_over_time.append(best)

    kg.set_knobs(control=1.0); kg.reset()
    pred_h = int(np.argmax(kg.forward_token(5)))
    kg.set_knobs(control=0.0); kg.reset()
    pred_l = int(np.argmax(kg.forward_token(5)))
    hit = int(pred_h == TARGET_HIGH) + int(pred_l == TARGET_LOW)

    return best, improvements, hit, pred_h, pred_l, scores_over_time


def _run_lut4_learning(seed, generations=300):
    """int4 with baked lookup table — 16 hand-picked values.
    Index 0..15 maps to a float via LUT. Values are log-spaced,
    dense at extremes (where evolution converges).

    LUT (16 entries, symmetric around 0):
      idx:  0     1     2     3     4     5     6     7
      val: -1000 -500  -250  -100  -50   -10   -1     0
      idx:  8     9    10    11    12    13    14    15
      val:  0    +1   +10   +50  +100  +250  +500  +1000
    """
    LUT = np.array([
        -1000, -500, -250, -100, -50, -10, -1, 0,
        0, 1, 10, 50, 100, 250, 500, 1000,
    ], dtype=np.float32)
    LUT_SCALE = 1.0 / 1000.0  # normalize so max = 1.0

    kg = KnobConditionedGraph(vocab=32, knob_names=["control"], seed=seed)
    knob_id = 0
    H = kg.graph.H
    rng = np.random.RandomState(seed + 1000)

    TARGET_HIGH, TARGET_LOW = 0, 15

    # int4 index storage (0..15)
    indices = np.full(H, 7, dtype=np.uint8)  # init to index 7 = zero

    def apply_edges():
        kg.graph.mask[knob_id, :] = LUT[indices] * LUT_SCALE
        kg.graph.mask[knob_id, knob_id] = 0.0
        kg.graph.resync_alive()

    for t in range(H):
        kg.knob_magnitudes[knob_id, t] = 0
    kg._sync_knob_mask()
    apply_edges()

    def score():
        s = 0.0
        kg.set_knobs(control=1.0); kg.reset()
        out = kg.forward_token(5)
        s += out[TARGET_HIGH] - np.max(np.delete(out, TARGET_HIGH))
        kg.set_knobs(control=0.0); kg.reset()
        out = kg.forward_token(5)
        s += out[TARGET_LOW] - np.max(np.delete(out, TARGET_LOW))
        return float(s)

    best = score()
    improvements = 0
    scores_over_time = [best]

    for gen in range(generations):
        target = rng.randint(1, H)
        old_idx = int(indices[target])

        new_idx = rng.randint(0, 16)
        if new_idx == old_idx:
            scores_over_time.append(best)
            continue

        indices[target] = np.uint8(new_idx)
        apply_edges()

        s = score()
        if s > best:
            best = s
            improvements += 1
        else:
            indices[target] = np.uint8(old_idx)
            apply_edges()
        scores_over_time.append(best)

    kg.set_knobs(control=1.0); kg.reset()
    pred_h = int(np.argmax(kg.forward_token(5)))
    kg.set_knobs(control=0.0); kg.reset()
    pred_l = int(np.argmax(kg.forward_token(5)))
    hit = int(pred_h == TARGET_HIGH) + int(pred_l == TARGET_LOW)

    return best, improvements, hit, pred_h, pred_l, scores_over_time


def _run_int4_learning(seed, generations=300):
    """int4 signed: values in {-8..+7}, 16 options, random pick.
    Effective mask value: int4_val * scale / 7  (so ±7 maps to ±scale).
    """
    kg = KnobConditionedGraph(vocab=32, knob_names=["control"], seed=seed)
    knob_id = 0
    H = kg.graph.H
    rng = np.random.RandomState(seed + 1000)
    scale = kg.graph.edge_magnitude

    TARGET_HIGH, TARGET_LOW = 0, 15

    edges = np.zeros(H, dtype=np.int8)  # only use -8..+7

    def apply_edges():
        kg.graph.mask[knob_id, :] = edges.astype(np.float32) * (scale / 7.0)
        kg.graph.mask[knob_id, knob_id] = 0.0
        kg.graph.resync_alive()

    for t in range(H):
        kg.knob_magnitudes[knob_id, t] = 0
    kg._sync_knob_mask()
    apply_edges()

    def score():
        s = 0.0
        kg.set_knobs(control=1.0); kg.reset()
        out = kg.forward_token(5)
        s += out[TARGET_HIGH] - np.max(np.delete(out, TARGET_HIGH))
        kg.set_knobs(control=0.0); kg.reset()
        out = kg.forward_token(5)
        s += out[TARGET_LOW] - np.max(np.delete(out, TARGET_LOW))
        return float(s)

    best = score()
    improvements = 0
    scores_over_time = [best]

    for gen in range(generations):
        target = rng.randint(1, H)
        old_val = int(edges[target])

        new_val = rng.randint(-8, 8)  # -8..+7
        if new_val == old_val:
            scores_over_time.append(best)
            continue

        edges[target] = np.int8(new_val)
        apply_edges()

        s = score()
        if s > best:
            best = s
            improvements += 1
        else:
            edges[target] = np.int8(old_val)
            apply_edges()
        scores_over_time.append(best)

    kg.set_knobs(control=1.0); kg.reset()
    pred_h = int(np.argmax(kg.forward_token(5)))
    kg.set_knobs(control=0.0); kg.reset()
    pred_l = int(np.argmax(kg.forward_token(5)))
    hit = int(pred_h == TARGET_HIGH) + int(pred_l == TARGET_LOW)

    return best, improvements, hit, pred_h, pred_l, scores_over_time


def test_ab_ternary_vs_uint8():
    """4-way A/B: float ternary vs int ternary vs bitmask ternary vs uint8."""
    print("\n" + "=" * 60)
    print("TEST 9: A/B — flt_tern vs int_tern vs bit_tern vs uint8")
    print("=" * 60)

    n_trials = 10
    generations = 2000

    methods = ["flt_tern", "int4", "lut4", "uint8"]
    runners = {
        "flt_tern": _run_ternary_learning,
        "int4":     _run_int4_learning,
        "lut4":     _run_lut4_learning,
        "uint8":    _run_uint8_learning,
    }
    results_by_method = {m: {"scores": [], "hits": [], "imps": []} for m in methods}

    header = f"  {'seed':>6s}"
    for m in methods:
        header += f"  {m:>10s}"
    header += "  winner"
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    for trial in range(n_trials):
        seed = 100 + trial * 7
        trial_scores = {}

        for m in methods:
            sc, imp, hit, _, _, _ = runners[m](seed, generations)
            results_by_method[m]["scores"].append(sc)
            results_by_method[m]["hits"].append(hit)
            results_by_method[m]["imps"].append(imp)
            trial_scores[m] = sc

        winner = max(trial_scores, key=trial_scores.get)
        line = f"  {seed:6d}"
        for m in methods:
            line += f"  {trial_scores[m]:+10.4f}"
        line += f"  {winner}"
        print(line)

    # Summary
    print(f"\n  SUMMARY ({n_trials} trials, {generations} generations each):")
    header2 = f"  {'':20s}"
    for m in methods:
        header2 += f"  {m:>10s}"
    print(header2)

    means = {m: np.mean(results_by_method[m]["scores"]) for m in methods}
    line_score = f"  {'Mean score':20s}"
    line_hits = f"  {'Mean hits':20s}"
    line_imps = f"  {'Mean improvements':20s}"
    for m in methods:
        line_score += f"  {means[m]:+10.4f}"
        line_hits += f"  {np.mean(results_by_method[m]['hits']):10.2f}"
        line_imps += f"  {np.mean(results_by_method[m]['imps']):10.1f}"
    print(line_score)
    print(line_hits)
    print(line_imps)

    # Win count
    wins = {m: 0 for m in methods}
    for i in range(n_trials):
        trial_sc = {m: results_by_method[m]["scores"][i] for m in methods}
        wins[max(trial_sc, key=trial_sc.get)] += 1
    line_wins = f"  {'Wins':20s}"
    for m in methods:
        line_wins += f"  {wins[m]:10d}"
    print(line_wins)

    overall = max(means, key=means.get)
    print(f"\n  Overall winner: {overall} (score={means[overall]:+.4f})")

    return overall


# ===========================================================================

if __name__ == "__main__":
    print("KNOB-CONDITIONED SELFWIRINGGRAPH PROTOTYPE")
    print("=" * 60)
    print("Concept: parameters as explicit input neurons, not hidden weights")
    print("The network learns WHAT TO DO with the parameter, not the parameter itself")
    print()

    results = {}
    t0 = time.time()

    results["sensitivity"] = test_knob_sensitivity()
    results["linearity"] = test_knob_linearity()
    results["independence"] = test_knob_independence()
    results["wiring"] = test_knob_wiring()
    results["feedback"] = test_instant_feedback()
    results["2d_sweep"] = test_2d_knob_sweep()
    results["learning"] = test_knob_learning()
    results["multi_knob"] = test_multi_knob_learning()
    results["ab_ternary_vs_uint8"] = test_ab_ternary_vs_uint8()

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        print(f"  {name:20s}: {'PASS' if ok else 'FAIL/WARN'}")
    print(f"\n  {passed}/{total} passed in {elapsed:.2f}s")
