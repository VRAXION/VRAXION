"""
Adversarial Stress Test — 19 probes for SelfWiringGraph
=========================================================
All probes must stay green or warning-only for the model to be considered valid.
"""

import sys, os, subprocess, tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import random
from lib.data import (
    fineweb_candidate_paths,
    load_fineweb_bytes,
    resolve_fineweb_path,
)
from model.graph import SelfWiringGraph, softmax

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


def main():
    results = []

    # PROBE 1: Zero internal neurons (V=N)
    header(1, "Zero internal neurons (V=N=16)")
    np.random.seed(SEED); random.seed(SEED)
    try:
        net = SelfWiringGraph(16, 16)
        perm = np.random.permutation(16)
        score_best = 0.0
        for att in range(2000):
            snapshot = net.save_state()
            net.mutate()
            logits = net.forward_batch(ticks=8)
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = e / e.sum(axis=1, keepdims=True)
            preds = np.argmax(probs, axis=1)
            acc = (preds == perm).mean()
            tp = probs[np.arange(16), perm].mean()
            sc = 0.5*acc + 0.5*tp
            if sc > score_best: score_best = sc
            else: net.restore_state(snapshot)
        r = result(PASS if score_best > 0.05 else WARN,
                   f"V=N=16: {score_best*100:.1f}%")
    except Exception as ex:
        r = result(FAIL, f"Crashed: {ex}")
    results.append(("Zero internals", r))

    # PROBE 2: Identity permutation
    header(2, "Identity permutation")
    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(80, 16)
    identity = np.arange(16)
    acc_best = 0.0
    for att in range(3000):
        snapshot = net.save_state()
        net.mutate()
        logits = net.forward_batch(ticks=8)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        acc = (np.argmax(probs, axis=1) == identity).mean()
        tp = probs[np.arange(16), identity].mean()
        sc = 0.5*acc + 0.5*tp
        if sc > 0: acc_best = max(acc_best, acc)
        else: net.restore_state(snapshot)
    r = result(PASS if acc_best > 0.5 else WARN, f"Identity: {acc_best*100:.1f}%")
    results.append(("Identity perm", r))

    # PROBE 3: Adversarial permutations
    header(3, "Adversarial permutations")
    for name, perm in [
        ('shift_1', np.roll(np.arange(16), 1)),
        ('reverse', np.arange(16)[::-1].copy()),
        ('swap_pairs', np.array([1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14])),
    ]:
        np.random.seed(SEED); random.seed(SEED)
        net = SelfWiringGraph(80, 16)
        for att in range(2000):
            snapshot = net.save_state()
            net.mutate()
            logits = net.forward_batch(ticks=8)
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = e / e.sum(axis=1, keepdims=True)
            acc = (np.argmax(probs, axis=1) == perm).mean()
            if acc > 0: pass
            else: net.restore_state(snapshot)
        print(f"    {name}: OK")
    r = result(PASS, "All adversarial perms trained without crash")
    results.append(("Adversarial perms", r))

    # PROBE 4: NaN/Inf injection
    header(4, "NaN/Inf injection")
    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(32, 8)
    net.reset()
    logits_nan = net.forward(np.full(8, np.nan, dtype=np.float32), ticks=8)
    net.reset()
    logits_inf = net.forward(np.full(8, np.inf, dtype=np.float32), ticks=8)
    net.reset()
    logits_huge = net.forward(np.full(8, 1e10, dtype=np.float32), ticks=8)
    all_finite = (np.all(np.isfinite(logits_nan)) and
                  np.all(np.isfinite(logits_inf)) and
                  np.all(np.isfinite(logits_huge)))
    r = result(PASS if all_finite else WARN, f"All finite: {all_finite}")
    results.append(("NaN/Inf injection", r))

    # PROBE 5: Empty network
    header(5, "Empty network (density=0)")
    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(32, 8, density=0.0)
    net.reset()
    logits = net.forward(np.zeros(8, dtype=np.float32), ticks=8)
    logits_b = net.forward_batch(ticks=8)
    ok = (np.all(np.isfinite(logits)) and np.all(np.isfinite(logits_b)) and
          net.count_connections() == 0)
    r = result(PASS if ok else FAIL,
               f"Empty network finite={np.all(np.isfinite(logits)) and np.all(np.isfinite(logits_b))}, "
               f"edges={net.count_connections()}")
    results.append(("Empty network", r))

    # PROBE 6: Fully connected
    header(6, "Fully connected (density=1.0)")
    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(48, 16, density=1.0)
    net.reset()
    logits = net.forward(np.zeros(16, dtype=np.float32), ticks=8)
    expected_edges = net.H * (net.H - 1)
    ok = np.all(np.isfinite(logits)) and net.count_connections() == expected_edges
    r = result(PASS if ok else FAIL,
               f"Full density finite={np.all(np.isfinite(logits))}, edges={net.count_connections()}/{expected_edges}")
    results.append(("Full density", r))

    # PROBE 7: Single neuron
    header(7, "Single neuron (V=1, N=1)")
    try:
        net = SelfWiringGraph(1, 1)
        net.reset()
        logits = net.forward(np.array([1.0], dtype=np.float32), ticks=8)
        logits_b = net.forward_batch(ticks=8)
        net.mutate()
        net.mutate()
        r = result(PASS, "V=1 N=1 works")
    except Exception as ex:
        r = result(FAIL, f"Crashed: {ex}")
    results.append(("Single neuron", r))

    # PROBE 8: Batch vs Sequential consistency
    header(8, "Batch vs Sequential consistency")
    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(64, 16)
    seq_logits = np.zeros((16, 16), dtype=np.float32)
    for i in range(16):
        net.reset()
        world = np.zeros(16, dtype=np.float32); world[i] = 1.0
        seq_logits[i] = net.forward(world, ticks=8)
    batch_logits = net.forward_batch(ticks=8)
    max_diff = np.abs(seq_logits - batch_logits).max()
    pred_match = (np.argmax(seq_logits, axis=1) == np.argmax(batch_logits, axis=1)).mean()
    r = result(PASS if max_diff < 1e-5 else (WARN if pred_match >= 0.9 else FAIL),
               f"max_diff={max_diff:.1e}, pred_agree={pred_match*100:.0f}%")
    results.append(("Batch vs Sequential", r))

    # PROBE 9: Mutation determinism
    header(9, "Mutation determinism")
    np.random.seed(99); random.seed(99)
    net1 = SelfWiringGraph(48, 16)
    for _ in range(100): net1.mutate()
    np.random.seed(99); random.seed(99)
    net2 = SelfWiringGraph(48, 16)
    for _ in range(100): net2.mutate()
    ok = np.array_equal(net1.mask, net2.mask)
    r = result(PASS if ok else FAIL, f"Deterministic: {ok}")
    results.append(("Determinism", r))

    # PROBE 10: State leak after reset
    header(10, "State leak after reset()")
    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(64, 16)
    world = np.zeros(16, dtype=np.float32); world[0] = 1.0
    net.reset()
    logits_a = net.forward(world, ticks=8).copy()
    net.reset()
    logits_b = net.forward(world, ticks=8).copy()
    ok = np.allclose(logits_a, logits_b, atol=1e-6)
    r = result(PASS if ok else FAIL, f"Same output after reset: {ok}")
    results.append(("State leak", r))

    # PROBE 11: Charge explosion
    header(11, "Charge explosion -- 1000 ticks")
    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(64, 16, density=0.3)
    net.reset()
    logits = net.forward(np.zeros(16, dtype=np.float32), ticks=1000)
    ok = np.all(np.isfinite(logits)) and np.abs(net.charge).max() <= 1.01
    r = result(PASS if ok else FAIL, f"Charge bounded: {ok}")
    results.append(("Charge explosion", r))

    # PROBE 12: Save/restore fidelity
    header(12, "Save/restore exact fidelity")
    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(64, 16)
    net.forward(np.zeros(16, dtype=np.float32), ticks=8)
    state = net.save_state()
    net.mask[:] = -1
    net.state[:] = 42
    net.charge[:] = 100
    net.loss_pct = 50
    net.drive = -7
    net.theta[:] = 0.91
    net.decay[:] = 0.73
    net.restore_state(state)
    alive_set = set(net.alive)
    mask_set = set(zip(*np.where(net.mask != 0)))
    ok = (
        np.array_equal(net.mask, state['mask']) and
        np.array_equal(net.state, state['state']) and
        np.array_equal(net.charge, state['charge']) and
        net.loss_pct == state['loss_pct'] and
        net.drive == state['drive'] and
        np.array_equal(net.theta, state['theta']) and
        np.array_equal(net.decay, state['decay']) and
        alive_set == mask_set and
        len(net.alive) == int((net.mask != 0).sum())
    )
    state['mask'][0, 0] = 99
    state['theta'][0] = 99
    state['decay'][0] = 99
    deep_ok = (net.mask[0, 0] != 99 and net.theta[0] != 99 and net.decay[0] != 99)
    r = result(PASS if ok and deep_ok else FAIL,
               f"Bitwise restore: {ok}, deep copy: {deep_ok}")
    results.append(("Save/restore", r))

    # PROBE 13: Default mutation reject restores all learned params
    header(13, "Default mutation reject restores theta/drive/loss/cache")
    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(64, 16)
    net.drive = np.int8(-1)
    before = net.save_state()
    old_randint = random.randint
    old_random = random.random
    seq = iter([2, 20, 1, 3, 0])  # no loss drift, no drive drift, theta drift, remove edge 0
    random.randint = lambda a, b: next(seq)
    random.random = lambda: 0.73
    try:
        undo = net.mutate()
    finally:
        random.randint = old_randint
        random.random = old_random
    net.replay(undo)
    alive_set = set(net.alive)
    mask_set = set(zip(*np.where(net.mask != 0)))
    ok = (
        np.array_equal(net.mask, before['mask']) and
        np.array_equal(net.theta, before['theta']) and
        np.array_equal(net.decay, before['decay']) and
        net.loss_pct == before['loss_pct'] and
        net.drive == before['drive'] and
        alive_set == mask_set and
        len(net.alive) == int((net.mask != 0).sum())
    )
    r = result(PASS if ok else FAIL, f"Reject restore exact: {ok}")
    results.append(("Reject restore", r))

    # PROBE 14: Alive cache coherence after direct mask restore
    header(14, "Alive cache coherence after direct mask restore")
    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(64, 16)
    sm = net.mask.copy()
    for _ in range(32):
        net.mutate()
    net.mask = sm; net.resync_alive()
    alive_set = set(net.alive)
    mask_set = set(zip(*np.where(net.mask != 0)))
    count_ok = len(net.alive) == int((net.mask != 0).sum())
    cells_ok = alive_set == mask_set
    diag_ok = all(r != c for r, c in net.alive)
    r = result(PASS if count_ok and cells_ok and diag_ok else FAIL,
               f"count={count_ok}, cells={cells_ok}, diag={diag_ok}")
    results.append(("Alive cache coherence", r))

    # PROBE 15: Constructor kwarg semantics
    header(15, "Constructor kwarg semantics")
    np.random.seed(SEED); random.seed(SEED)
    net_a = SelfWiringGraph(8, threshold=0.33, leak=0.90)
    net_b = SelfWiringGraph(32, 8, density=0.0, threshold=0.44, leak=0.80)
    ok = (
        net_a.H == 8 * net_a.NV_RATIO and
        np.allclose(net_a.theta, 0.33) and
        np.allclose(net_a.decay, 0.10) and
        net_b.H == 32 and
        net_b.count_connections() == 0 and
        np.allclose(net_b.theta, 0.44) and
        np.allclose(net_b.decay, 0.20)
    )
    r = result(PASS if ok else FAIL,
               f"one_arg_H={net_a.H}, explicit_H={net_b.H}, empty_edges={net_b.count_connections()}")
    results.append(("Constructor kwargs", r))

    # PROBE 16: Disk roundtrip exact fidelity
    header(16, "Disk save/load exact fidelity")
    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(64, 16, density=0.15, threshold=0.41, leak=0.82)
    for _ in range(8):
        net.mutate()
    logits_before = net.forward_batch(ticks=8)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "graph_roundtrip.npz")
        net.save(path)
        loaded = SelfWiringGraph.load(path)
    logits_after = loaded.forward_batch(ticks=8)
    ok = (
        loaded.V == net.V and
        loaded.H == net.H and
        np.array_equal(loaded.mask, net.mask) and
        np.array_equal(loaded.input_projection, net.input_projection) and
        np.array_equal(loaded.output_projection, net.output_projection) and
        np.array_equal(loaded.theta, net.theta) and
        np.array_equal(loaded.decay, net.decay) and
        np.allclose(logits_before, logits_after, atol=1e-6)
    )
    r = result(PASS if ok else FAIL, f"Roundtrip exact: {ok}")
    results.append(("Disk roundtrip", r))

    # PROBE 17: Forced mutation API compatibility
    header(17, "Forced mutation API compatibility")
    np.random.seed(SEED); random.seed(SEED)
    try:
        net = SelfWiringGraph(32, 8)
        before = net.count_connections()
        undo = net.mutate(forced_op='add', n_changes=2, freeze_params=True)
        after_add = net.count_connections()
        net.replay(undo)
        after_replay = net.count_connections()
        ok = after_add >= before and after_replay == before
        r = result(PASS if ok else FAIL,
                   f"forced_add={after_add-before}, replay_ok={after_replay == before}")
    except Exception as ex:
        r = result(FAIL, f"Crashed: {ex}")
    results.append(("Forced mutate API", r))

    # PROBE 18: Fineweb resolver semantics
    header(18, "Fineweb resolver semantics")
    env_var = "VRAXION_TEST_FINEWEB_PATH"
    old_env = os.environ.pop(env_var, None)
    try:
        with tempfile.TemporaryDirectory() as td:
            repo_root = Path(td)
            canonical, legacy = fineweb_candidate_paths(repo_root)
            canonical_hint = str(canonical)

            canonical.parent.mkdir(parents=True, exist_ok=True)
            canonical.write_bytes(bytes([1, 2, 3, 4]))
            resolved_canonical = resolve_fineweb_path(repo_root=repo_root, env_var=env_var)
            sample = load_fineweb_bytes(max_bytes=2, repo_root=repo_root, env_var=env_var)

            canonical.unlink()
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.write_bytes(bytes([5, 6, 7]))
            resolved_legacy = resolve_fineweb_path(repo_root=repo_root, env_var=env_var)

            missing_ok = False
            legacy.unlink()
            try:
                resolve_fineweb_path(repo_root=repo_root, env_var=env_var)
            except FileNotFoundError as ex:
                missing_ok = canonical_hint in str(ex)

            ok = (
                resolved_canonical == canonical and
                resolved_legacy == legacy and
                sample.tolist() == [1, 2] and
                missing_ok
            )
            r = result(PASS if ok else FAIL,
                       f"canonical={resolved_canonical == canonical}, "
                       f"legacy={resolved_legacy == legacy}, "
                       f"sample={sample.tolist()}, missing_hint={missing_ok}")
    except Exception as ex:
        r = result(FAIL, f"Crashed: {ex}")
    finally:
        if old_env is not None:
            os.environ[env_var] = old_env
    results.append(("Fineweb resolver", r))

    # PROBE 19: Active v4.2 tree must not hardcode Diamond Code corpus paths
    header(19, "No hardcoded Diamond Code corpus paths in active v4.2")
    repo_root = Path(__file__).resolve().parents[2]
    tracked = subprocess.check_output(
        ["git", "-C", str(repo_root), "ls-files", "v4.2"],
        text=True,
    ).splitlines()
    forbidden_hits = []
    for rel in tracked:
        if not rel.endswith(".py"):
            continue
        if rel == "v4.2/lib/data.py":
            continue
        if rel == "v4.2/tests/test_model.py":
            continue
        if rel.startswith("v4.2/tests/archive/"):
            continue
        if rel.startswith("v4.2/tests/gpu_experimental/"):
            continue
        text = (repo_root / rel).read_text(encoding="utf-8")
        if "Diamond Code/data/traindat" in text or "../Diamond Code" in text:
            forbidden_hits.append(rel)
    ok = not forbidden_hits
    hit_text = ", ".join(forbidden_hits[:3]) if forbidden_hits else "none"
    r = result(PASS if ok else FAIL, f"hardcoded hits: {hit_text}")
    results.append(("No hardcoded Diamond path", r))

    # SUMMARY
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
        print(f"\n  {fails} FAILURE(S)!")
    else:
        print(f"\n  All clean!")
    print(f"\n{'='*60}", flush=True)
    return fails


if __name__ == '__main__':
    sys.exit(main())
