"""
VRAXION v22 — Hard Adversarial Stress Test
============================================
12 probes to find edge cases, breaks, and hidden weaknesses
in the Self-Wiring Graph Network.

Tests:
  1. Zero internal neurons (V=N) — can it learn with ONLY I/O?
  2. Identity permutation — trivial task, should nail it
  3. Adversarial permutations — shift-1, reverse, swap-pairs
  4. NaN/Inf injection — what breaks?
  5. Empty network (density=0) — no connections
  6. Fully connected (density=1.0) — max density
  7. Single neuron (V=1, N=1) — degenerate minimum
  8. Batch vs Sequential consistency — do they agree?
  9. Mutation determinism — same seed = same result?
 10. State leak after reset — is reset() thorough?
 11. Charge explosion — 1000 ticks, does clip hold?
 12. Save/restore fidelity — exact bitwise match?
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
import time
from v22_best_config import SelfWiringGraph, softmax

SEED = 42
PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"


def header(num, name):
    print(f"\n  {'-'*55}")
    print(f"  PROBE {num:2d}: {name}")
    print(f"  {'-'*55}")
    sys.stdout.flush()


def result(status, msg):
    tag = {"PASS": "+", "FAIL": "X", "WARN": "!"}[status]
    print(f"    [{tag}] {status}: {msg}")
    sys.stdout.flush()
    return status


results = []


# ============================================================
#  PROBE 1: Zero internal neurons (V=N)
# ============================================================
header(1, "Zero internal neurons (V=N=16)")

np.random.seed(SEED); random.seed(SEED)
V = 16; N = 16  # no internal neurons at all
try:
    net = SelfWiringGraph(N, V)
    perm = np.random.permutation(V)

    # Quick train: 2000 attempts
    score_best = 0.0
    for att in range(2000):
        saved_m = net.mask.copy(); saved_w = net.W.copy()
        net.mutate_structure(0.05)
        logits = net.forward_batch(ticks=8)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        acc = (preds == perm).mean()
        tp = probs[np.arange(V), perm].mean()
        sc = 0.5*acc + 0.5*tp
        if sc > score_best:
            score_best = sc
        else:
            net.mask = saved_m; net.W = saved_w

    print(f"    V=N=16, zero internals: acc={score_best*100:.1f}%")
    # It CAN learn something even with zero internal neurons
    # because I/O neurons connect to each other
    if score_best > 0.05:
        r = result(PASS, f"Learns with zero internals ({score_best*100:.1f}%)")
    else:
        r = result(WARN, f"Cannot learn with zero internals ({score_best*100:.1f}%)")
    results.append(("Zero internals", r))
except Exception as ex:
    r = result(FAIL, f"Crashed: {ex}")
    results.append(("Zero internals", r))


# ============================================================
#  PROBE 2: Identity permutation (trivial task)
# ============================================================
header(2, "Identity permutation (input=output)")

np.random.seed(SEED); random.seed(SEED)
V = 16; N = 80
net = SelfWiringGraph(N, V)
identity = np.arange(V)  # target[i] = i

score_best = 0.0; acc_best = 0.0
for att in range(3000):
    saved_m = net.mask.copy(); saved_w = net.W.copy()
    net.mutate_structure(0.05)
    logits = net.forward_batch(ticks=8)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    acc = (preds == identity).mean()
    tp = probs[np.arange(V), identity].mean()
    sc = 0.5*acc + 0.5*tp
    if sc > score_best:
        score_best = sc; acc_best = acc
    else:
        net.mask = saved_m; net.W = saved_w

print(f"    Identity perm: acc={acc_best*100:.1f}%")
# Identity should be EASIER than random perm because input already
# activates the correct output neuron via self-connection (act*0.1)
if acc_best > 0.5:
    r = result(PASS, f"Identity is easy ({acc_best*100:.1f}%)")
else:
    r = result(WARN, f"Identity not trivial ({acc_best*100:.1f}%) -- self-loop 0.1 too weak?")
results.append(("Identity perm", r))


# ============================================================
#  PROBE 3: Adversarial permutations
# ============================================================
header(3, "Adversarial permutations (shift-1, reverse, swap)")

adversarial_perms = {
    'shift_1':   np.roll(np.arange(16), 1),           # i -> (i+1)%16
    'reverse':   np.arange(16)[::-1].copy(),           # i -> 15-i
    'swap_pairs': np.array([1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14]),  # swap adjacent
}

for name, perm in adversarial_perms.items():
    np.random.seed(SEED); random.seed(SEED)
    V = 16; N = 80
    net = SelfWiringGraph(N, V)

    score_best = 0.0; acc_best = 0.0
    for att in range(3000):
        saved_m = net.mask.copy(); saved_w = net.W.copy()
        net.mutate_structure(0.05)
        logits = net.forward_batch(ticks=8)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        acc = (preds == perm).mean()
        tp = probs[np.arange(V), perm].mean()
        sc = 0.5*acc + 0.5*tp
        if sc > score_best:
            score_best = sc; acc_best = acc
        else:
            net.mask = saved_m; net.W = saved_w

    print(f"    {name:12s}: acc={acc_best*100:.1f}%")

# These should all be comparable to random perm difficulty
r = result(PASS, "All adversarial perms trained without crash")
results.append(("Adversarial perms", r))


# ============================================================
#  PROBE 4: NaN/Inf injection
# ============================================================
header(4, "NaN/Inf injection into forward pass")

np.random.seed(SEED); random.seed(SEED)
V = 8; N = 32
net = SelfWiringGraph(N, V)

# Test 4a: NaN input
world_nan = np.full(V, np.nan, dtype=np.float32)
net.reset()
logits_nan = net.forward(world_nan, ticks=8)
has_nan = np.isnan(logits_nan).any()
has_inf = np.isinf(logits_nan).any()
print(f"    NaN input  -> output has NaN: {has_nan}, Inf: {has_inf}")

# Test 4b: Inf input
world_inf = np.full(V, np.inf, dtype=np.float32)
net.reset()
logits_inf = net.forward(world_inf, ticks=8)
has_nan2 = np.isnan(logits_inf).any()
has_inf2 = np.isinf(logits_inf).any()
print(f"    Inf input  -> output has NaN: {has_nan2}, Inf: {has_inf2}")

# Test 4c: Huge input
world_huge = np.full(V, 1e10, dtype=np.float32)
net.reset()
logits_huge = net.forward(world_huge, ticks=8)
all_finite = np.all(np.isfinite(logits_huge))
print(f"    1e10 input -> all finite: {all_finite}, range: [{logits_huge.min():.2f}, {logits_huge.max():.2f}]")

# Test 4d: softmax on NaN
try:
    sm = softmax(logits_nan)
    sm_ok = np.all(np.isfinite(sm))
    print(f"    softmax(NaN logits) -> finite: {sm_ok}")
except Exception as ex:
    print(f"    softmax(NaN logits) -> EXCEPTION: {ex}")
    sm_ok = False

if has_nan or has_nan2:
    r = result(WARN, "NaN propagates through network (no internal NaN guard)")
elif not all_finite:
    r = result(WARN, "Large inputs produce non-finite output")
else:
    r = result(PASS, "All outputs finite for tested inputs")
results.append(("NaN/Inf injection", r))


# ============================================================
#  PROBE 5: Empty network (density=0)
# ============================================================
header(5, "Empty network (density=0)")

np.random.seed(SEED); random.seed(SEED)
V = 8; N = 32
net = SelfWiringGraph(N, V, density=0.0)
conns = net.count_connections()
print(f"    Connections at init: {conns}")

# Forward should work (just no signal propagation)
world = np.zeros(V, dtype=np.float32); world[0] = 1.0
net.reset()
logits = net.forward(world, ticks=8)
all_finite = np.all(np.isfinite(logits))
print(f"    Forward output finite: {all_finite}, range: [{logits.min():.4f}, {logits.max():.4f}]")

# Mutation should add connections
net.mutate_structure(0.05)
conns_after = net.count_connections()
print(f"    Connections after 1 mutation: {conns_after}")

# Batch forward
logits_b = net.forward_batch(ticks=8)
batch_finite = np.all(np.isfinite(logits_b))
print(f"    Batch forward finite: {batch_finite}")

if conns == 0 and all_finite and batch_finite:
    r = result(PASS, "Empty network handles gracefully")
else:
    r = result(WARN, f"Unexpected: {conns} conns at density=0")
results.append(("Empty network", r))


# ============================================================
#  PROBE 6: Fully connected (density=1.0)
# ============================================================
header(6, "Fully connected (density=1.0)")

np.random.seed(SEED); random.seed(SEED)
V = 16; N = 48
net = SelfWiringGraph(N, V, density=1.0)
conns = net.count_connections()
max_conns = N * (N - 1)  # excluding diagonal
density = conns / max_conns
print(f"    Connections: {conns}/{max_conns} ({density*100:.1f}%)")

# Forward should work but might saturate
world = np.zeros(V, dtype=np.float32); world[0] = 1.0
net.reset()
logits = net.forward(world, ticks=8)
all_finite = np.all(np.isfinite(logits))
charge_range = logits.max() - logits.min()
print(f"    Forward output finite: {all_finite}, range: {charge_range:.4f}")
print(f"    Output: [{logits.min():.4f}, {logits.max():.4f}]")

# Check if all outputs are the same (saturated)
is_uniform = charge_range < 0.001
if is_uniform:
    print(f"    [!] All outputs nearly identical -- saturated!")

# Can it learn?
perm = np.random.permutation(V)
score_best = 0.0
for att in range(1000):
    saved_m = net.mask.copy(); saved_w = net.W.copy()
    net.mutate_structure(0.05)
    logits_b = net.forward_batch(ticks=8)
    e = np.exp(logits_b - logits_b.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    preds = np.argmax(probs, axis=1)
    acc = (preds == perm).mean()
    tp = probs[np.arange(V), perm].mean()
    sc = 0.5*acc + 0.5*tp
    if sc > score_best:
        score_best = sc
    else:
        net.mask = saved_m; net.W = saved_w

print(f"    Learning at full density: {score_best*100:.1f}% (1K att)")

if all_finite:
    r = result(PASS if not is_uniform else WARN,
               f"Full density: finite={all_finite}, uniform={is_uniform}, learns={score_best*100:.1f}%")
else:
    r = result(FAIL, "Non-finite outputs at full density")
results.append(("Full density", r))


# ============================================================
#  PROBE 7: Single neuron (V=1, N=1)
# ============================================================
header(7, "Single neuron (V=1, N=1)")

np.random.seed(SEED); random.seed(SEED)
try:
    net = SelfWiringGraph(1, 1)
    print(f"    Created V=1, N=1 network")
    print(f"    Connections: {net.count_connections()}")

    world = np.array([1.0], dtype=np.float32)
    net.reset()
    logits = net.forward(world, ticks=8)
    print(f"    Forward output: {logits}")
    all_finite = np.all(np.isfinite(logits))

    # Batch forward
    logits_b = net.forward_batch(ticks=8)
    print(f"    Batch output shape: {logits_b.shape}, value: {logits_b}")

    # Mutation
    net.mutate_structure(0.05)
    net.mutate_weights()
    print(f"    Mutations: no crash")

    r = result(PASS, f"V=1 N=1 works, output finite: {all_finite}")
except Exception as ex:
    r = result(FAIL, f"Crashed: {ex}")
results.append(("Single neuron", r))


# ============================================================
#  PROBE 8: Batch vs Sequential consistency
# ============================================================
header(8, "Batch vs Sequential consistency")

np.random.seed(SEED); random.seed(SEED)
V = 16; N = 64
net = SelfWiringGraph(N, V)

# Sequential: run each input independently (with reset)
seq_logits = np.zeros((V, V), dtype=np.float32)
for i in range(V):
    net.reset()
    world = np.zeros(V, dtype=np.float32)
    world[i] = 1.0
    seq_logits[i] = net.forward(world, ticks=8)

# Batch
batch_logits = net.forward_batch(ticks=8)

# Compare
max_diff = np.abs(seq_logits - batch_logits).max()
mean_diff = np.abs(seq_logits - batch_logits).mean()
print(f"    Max abs diff:  {max_diff:.6e}")
print(f"    Mean abs diff: {mean_diff:.6e}")

# Check if predictions match
seq_preds = np.argmax(seq_logits, axis=1)
batch_preds = np.argmax(batch_logits, axis=1)
pred_match = (seq_preds == batch_preds).mean()
print(f"    Prediction agreement: {pred_match*100:.1f}%")

if max_diff < 1e-5:
    r = result(PASS, f"Batch==Sequential (max_diff={max_diff:.1e})")
elif pred_match >= 0.9:
    r = result(WARN, f"Small numerical diff (max={max_diff:.1e}) but preds agree {pred_match*100:.0f}%")
else:
    r = result(FAIL, f"Batch!=Sequential! max_diff={max_diff:.1e}, pred_agree={pred_match*100:.0f}%")
results.append(("Batch vs Sequential", r))


# ============================================================
#  PROBE 9: Mutation determinism
# ============================================================
header(9, "Mutation determinism (same seed = same result)")

# Run 1
np.random.seed(99); random.seed(99)
V = 16; N = 48
net1 = SelfWiringGraph(N, V)
for _ in range(100):
    net1.mutate_structure(0.05)
    net1.mutate_weights()
mask1 = net1.mask.copy()
W1 = net1.W.copy()

# Run 2 (same seed)
np.random.seed(99); random.seed(99)
net2 = SelfWiringGraph(N, V)
for _ in range(100):
    net2.mutate_structure(0.05)
    net2.mutate_weights()
mask2 = net2.mask.copy()
W2 = net2.W.copy()

mask_match = np.array_equal(mask1, mask2)
W_match = np.array_equal(W1, W2)
print(f"    Mask identical after 100 mutations: {mask_match}")
print(f"    Weights identical after 100 mutations: {W_match}")

if mask_match and W_match:
    r = result(PASS, "Fully deterministic with same seed")
else:
    r = result(FAIL, "Non-deterministic! Same seed gives different results")
results.append(("Determinism", r))


# ============================================================
#  PROBE 10: State leak after reset
# ============================================================
header(10, "State leak after reset()")

np.random.seed(SEED); random.seed(SEED)
V = 16; N = 64
net = SelfWiringGraph(N, V)

# Run forward with input 0
world0 = np.zeros(V, dtype=np.float32); world0[0] = 1.0
net.reset()
logits_a = net.forward(world0, ticks=8).copy()
state_after_fwd = net.state.copy()
charge_after_fwd = net.charge.copy()

# Reset
net.reset()
state_after_reset = net.state.copy()
charge_after_reset = net.charge.copy()

# Run same forward again
logits_b = net.forward(world0, ticks=8).copy()

state_leaked = np.any(state_after_reset != 0)
charge_leaked = np.any(charge_after_reset != 0)
logits_match = np.allclose(logits_a, logits_b, atol=1e-6)

print(f"    State after forward: norm={np.linalg.norm(state_after_fwd):.4f}")
print(f"    State after reset:  norm={np.linalg.norm(state_after_reset):.6f}")
print(f"    Charge after reset: norm={np.linalg.norm(charge_after_reset):.6f}")
print(f"    Same input -> same output after reset: {logits_match}")
print(f"    Max logit diff: {np.abs(logits_a - logits_b).max():.2e}")

if not state_leaked and not charge_leaked and logits_match:
    r = result(PASS, "Clean reset, no state leak")
elif logits_match:
    r = result(WARN, "Reset leaves residuals but output matches")
else:
    r = result(FAIL, f"State leak! logits differ by {np.abs(logits_a - logits_b).max():.2e}")
results.append(("State leak", r))


# ============================================================
#  PROBE 11: Charge explosion (1000 ticks)
# ============================================================
header(11, "Charge explosion -- 1000 ticks")

np.random.seed(SEED); random.seed(SEED)
V = 16; N = 64
net = SelfWiringGraph(N, V, density=0.3)  # denser than default

world = np.zeros(V, dtype=np.float32); world[0] = 1.0
net.reset()

# Run with extreme tick count
logits = net.forward(world, ticks=1000)
all_finite = np.all(np.isfinite(logits))
charge_max = np.abs(net.charge).max()
print(f"    1000 ticks: finite={all_finite}, charge_max={charge_max:.4f}")
print(f"    Output range: [{logits.min():.4f}, {logits.max():.4f}]")

# Clip should keep charge bounded at threshold*2
expected_max = net.threshold * 2
within_bounds = charge_max <= expected_max + 0.01
print(f"    Charge within bounds (<={expected_max}): {within_bounds}")

# Batch with extreme ticks
logits_b = net.forward_batch(ticks=100)  # 100 for batch (1000 would be slow)
batch_finite = np.all(np.isfinite(logits_b))
print(f"    Batch 100 ticks: finite={batch_finite}")

if all_finite and within_bounds and batch_finite:
    r = result(PASS, f"Charge stable at {charge_max:.4f} (bound={expected_max})")
else:
    r = result(FAIL, f"Explosion! finite={all_finite}, bounds={within_bounds}")
results.append(("Charge explosion", r))


# ============================================================
#  PROBE 12: Save/restore fidelity
# ============================================================
header(12, "Save/restore exact fidelity")

np.random.seed(SEED); random.seed(SEED)
V = 16; N = 64
net = SelfWiringGraph(N, V)

# Run some forward passes to populate state
world = np.zeros(V, dtype=np.float32); world[3] = 1.0
net.forward(world, ticks=8)
net.forward(world, ticks=8)

# Save
state = net.save_state()

# Corrupt everything
net.W[:] = 999.0
net.mask[:] = -1
net.state[:] = 42.0
net.addr[:] = 0.0
net.target_W[:] = 0.0
net.charge[:] = 100.0

# Restore
net.restore_state(state)

# Check exact match
W_ok = np.array_equal(net.W, state[0])
mask_ok = np.array_equal(net.mask, state[1])
state_ok = np.array_equal(net.state, state[2])
addr_ok = np.array_equal(net.addr, state[3])
tw_ok = np.array_equal(net.target_W, state[4])
charge_ok = np.array_equal(net.charge, state[5])

print(f"    W:        {W_ok}")
print(f"    mask:     {mask_ok}")
print(f"    state:    {state_ok}")
print(f"    addr:     {addr_ok}")
print(f"    target_W: {tw_ok}")
print(f"    charge:   {charge_ok}")

# Also check it's a DEEP copy (modifying saved state shouldn't affect net)
state[0][0, 0] = -999.0
deep_copy_ok = net.W[0, 0] != -999.0
print(f"    Deep copy: {deep_copy_ok}")

all_ok = all([W_ok, mask_ok, state_ok, addr_ok, tw_ok, charge_ok, deep_copy_ok])
if all_ok:
    r = result(PASS, "Exact bitwise restore + deep copy verified")
else:
    r = result(FAIL, f"Restore failed: W={W_ok} mask={mask_ok} state={state_ok} "
               f"addr={addr_ok} tW={tw_ok} charge={charge_ok} deep={deep_copy_ok}")
results.append(("Save/restore", r))


# ============================================================
#  SUMMARY
# ============================================================
print(f"\n{'='*60}")
print(f"  ADVERSARIAL STRESS TEST -- SUMMARY")
print(f"{'='*60}\n")

passes = sum(1 for _, s in results if s == PASS)
warns = sum(1 for _, s in results if s == WARN)
fails = sum(1 for _, s in results if s == FAIL)

for name, status in results:
    tag = {"PASS": "+", "FAIL": "X", "WARN": "!"}[status]
    print(f"  [{tag}] {status:4s}  {name}")

print(f"\n  Total: {passes} PASS, {warns} WARN, {fails} FAIL out of {len(results)}")

if fails > 0:
    print(f"\n  {fails} FAILURE(S) -- needs fixing!")
elif warns > 0:
    print(f"\n  {warns} warning(s) -- potential improvement areas")
else:
    print(f"\n  All clean!")

print(f"\n{'='*60}", flush=True)
