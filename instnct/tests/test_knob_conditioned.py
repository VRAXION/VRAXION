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
    """

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
        for kid in self.knob_neuron_ids:
            count += int(np.count_nonzero(self.graph.mask[kid, :]))
        return count

    @property
    def knob_edge_strength(self):
        total = 0.0
        for kid in self.knob_neuron_ids:
            total += float(np.sum(np.abs(self.graph.mask[kid, :])))
        return total

    def knob_influence_map(self):
        influence = {}
        for name, idx in self.knob_index.items():
            kid = self.knob_neuron_ids[idx]
            row = self.graph.mask[kid, :]
            targets = np.where(row != 0)[0]
            weights = row[targets]
            influence[name] = {
                "n_targets": len(targets),
                "mean_weight": float(np.mean(weights)) if len(weights) > 0 else 0.0,
                "abs_strength": float(np.sum(np.abs(weights))),
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
    print(f"  Total edge strength: {kg.knob_edge_strength:.2f}")

    influence = kg.knob_influence_map()
    for name, info in influence.items():
        print(f"\n  {name}:")
        print(f"    targets: {info['n_targets']} neurons")
        print(f"    mean weight: {info['mean_weight']:.4f}")
        print(f"    abs strength: {info['abs_strength']:.2f}")

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
        old_val = kg.graph.mask[knob_id, target]

        new_val = kg.graph.edge_magnitude if old_val <= 0 else -kg.graph.edge_magnitude
        kg.graph.mask[knob_id, target] = new_val
        kg.graph._sync_sparse_idx()
        kg.graph.alive = list(zip(*np.where(kg.graph.mask != 0)))
        kg.graph.alive_set = set(kg.graph.alive)

        kg.reset()
        out = kg.forward_token(5)
        shift = float(np.sum(np.abs(out - base)))
        shifts.append(shift)
        print(f"  Mutation {trial+1:2d}: edge [{knob_id}→{target:3d}] "
              f"{old_val:+.1f}→{new_val:+.1f}  output_shift={shift:.4f}")

        # Restore
        kg.graph.mask[knob_id, target] = old_val

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
        old_val = float(kg.graph.mask[knob_id, target])

        options = [-kg.graph.edge_magnitude, 0.0, kg.graph.edge_magnitude]
        new_val = float(rng.choice(options))
        if new_val == old_val:
            continue

        kg.graph.mask[knob_id, target] = np.float32(new_val)
        kg.graph.resync_alive()

        new_score = score()
        if new_score > best_score:
            best_score = new_score
            improvements += 1
            if improvements <= 10 or improvements % 20 == 0:
                print(f"  Gen {gen:4d}: score {best_score:+.4f} "
                      f"(edge [{knob_id}→{target}] = {new_val:+.1f})")
        else:
            kg.graph.mask[knob_id, target] = np.float32(old_val)
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
        old_val = float(kg.graph.mask[knob_id, target])

        options = [-kg.graph.edge_magnitude, 0.0, kg.graph.edge_magnitude]
        new_val = float(rng.choice(options))
        if new_val == old_val:
            continue

        kg.graph.mask[knob_id, target] = np.float32(new_val)
        kg.graph.resync_alive()

        new_score = score()
        if new_score > best_score:
            best_score = new_score
            improvements += 1
        else:
            kg.graph.mask[knob_id, target] = np.float32(old_val)
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

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        print(f"  {name:20s}: {'PASS' if ok else 'FAIL/WARN'}")
    print(f"\n  {passed}/{total} passed in {elapsed:.2f}s")
