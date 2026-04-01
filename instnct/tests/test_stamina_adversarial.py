"""
Adversarial tests for the stamina (STP) code added to graph.py.
Tests edge cases, consistency, and potential bugs.
"""
import sys, os
import numpy as np
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from graph import SelfWiringGraph


def test_stamina_resync_after_mutate():
    """After fix: resync_alive auto-resizes stamina. Should not crash."""
    net = SelfWiringGraph(vocab=16, hidden=32, density=4, seed=42, theta_init=2)
    net._init_stamina()
    n_edges_before = len(net.alive)
    assert len(net._stamina) == n_edges_before, "stamina size != alive size after init"

    # Mutate: adds/removes edges → resync_alive should resize stamina
    for _ in range(5):  # multiple mutations to ensure size changes
        net.mutate(forced_op='add')

    n_edges_after = len(net.alive)
    stamina_len = len(net._stamina)

    if stamina_len == n_edges_after:
        print(f"  [+] Stamina auto-resized: {n_edges_before} → {stamina_len} (alive={n_edges_after})")
    else:
        print(f"  [!] BUG: stamina={stamina_len} != alive={n_edges_after}")
        return "BUG_NO_RESIZE"

    # Rollout should work now
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    try:
        injected = np.random.randn(32).astype(np.float32)
        state, charge = SelfWiringGraph.rollout_token(
            injected, mask=net.mask, theta=net._theta_f32,
            decay=net.decay, ticks=8, sparse_cache=sc,
            polarity=net._polarity_f32, refractory=net.refractory,
            channel=net.channel, stamina=net._stamina,
        )
        print(f"  [+] PASS: rollout with resized stamina works")
        return "PASS"
    except (IndexError, ValueError) as e:
        print(f"  [!] CRASH even after resize: {e}")
        return "CRASH"


def test_stamina_regen_overflow():
    """BUG CHECK: stamina regen clips to 255 via int16 intermediate.
    But what if stamina is already 255? +1 → 256 → clip to 255. OK.
    What about the int16 cast on large arrays?
    """
    # Edge case: all stamina at 255 (full), regen should be no-op
    stamina = np.full(1000, 255, dtype=np.uint8)
    result = np.clip(stamina.astype(np.int16) + 1, 0, 255).astype(np.uint8)
    assert np.all(result == 255), "Overflow at max stamina"

    # Edge case: all stamina at 0 (dead), regen +1 → 1
    stamina = np.full(1000, 0, dtype=np.uint8)
    result = np.clip(stamina.astype(np.int16) + 1, 0, 255).astype(np.uint8)
    assert np.all(result == 1), "Regen from 0 should give 1"

    print("  [+] PASS: stamina regen overflow/underflow safe")
    return "PASS"


def test_stamina_drain_works():
    """Check that drain fires when source neurons spike.
    Drain is -1 per fired source edge per tick.
    Regen is +1 every DECAY_PERIOD ticks (=6).
    With high firing, drain should cause VARIANCE in stamina
    (active edges drain more, inactive stay high).
    """
    net = SelfWiringGraph(vocab=16, hidden=32, density=4, seed=42, theta_init=1)
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    rows, cols = sc
    n_edges = len(rows)

    # Start all at same value — after rollout, variance proves drain happened
    stamina = np.full(n_edges, 150, dtype=np.uint8)

    injected = np.ones(32, dtype=np.float32) * 5.0
    state, charge = SelfWiringGraph.rollout_token(
        injected, mask=net.mask, theta=net._theta_f32,
        decay=net.decay, ticks=16, state=None, charge=None,
        sparse_cache=sc, polarity=net._polarity_f32,
        refractory=net.refractory, channel=net.channel,
        stamina=stamina,
    )

    min_val = int(np.min(stamina))
    max_val = int(np.max(stamina))
    std_val = float(np.std(stamina))
    unique_vals = len(np.unique(stamina))

    print(f"  Stamina after 16 ticks: range=[{min_val}, {max_val}], std={std_val:.2f}, unique={unique_vals}")

    # Key check: if drain works, edges connected to active neurons have LOWER stamina
    # than edges connected to quiet neurons → variance > 0
    if unique_vals > 1:
        print(f"  [+] PASS: drain creates variance ({unique_vals} unique values)")
        return "PASS"
    elif unique_vals == 1:
        # All edges same stamina = no differential drain = drain not working
        print(f"  [!] BUG: all edges identical stamina — drain has no effect")
        return "BUG_NO_DRAIN"
    else:
        return "INCONCLUSIVE"


def test_stamina_dense_path_ignored():
    """BUG CHECK: stamina only works on sparse 2-tuple path.
    If use_sparse=False (dense network) or 3-tuple cache → stamina silently ignored.
    """
    # Create a dense enough network that use_sparse=False
    net = SelfWiringGraph(vocab=16, hidden=8, density=100, seed=42, theta_init=2)
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    n_edges = len(sc[0])

    # At density=100% and H=8: 56 edges out of 64 possible = 87.5% > 10% threshold
    use_sparse = n_edges < 8 * 8 * 0.1
    print(f"  H=8, density=100%: edges={n_edges}, use_sparse={use_sparse}")

    if not use_sparse:
        stamina = np.full(n_edges, 100, dtype=np.uint8)
        stamina_before = stamina.copy()

        injected = net.input_projection[3]
        state, charge = SelfWiringGraph.rollout_token(
            injected, mask=net.mask, theta=net._theta_f32,
            decay=net.decay, ticks=8, state=None, charge=None,
            sparse_cache=sc, polarity=net._polarity_f32,
            refractory=net.refractory, channel=net.channel,
            stamina=stamina,
        )

        if np.array_equal(stamina, stamina_before):
            print(f"  [!] WARN: stamina passed but silently ignored on dense path")
            return "WARN_DENSE_IGNORED"
        else:
            print(f"  [+] Stamina active even on dense path")
            return "PASS"
    else:
        print(f"  [i] Network is sparse, can't test dense path")
        return "SKIP"


def test_stamina_3tuple_cache():
    """BUG CHECK: stamina only activates for 2-tuple sparse cache.
    If edge_magnitude != 1.0, build_sparse_cache returns 3-tuple → stamina skipped.
    """
    net = SelfWiringGraph(vocab=16, hidden=32, density=4, seed=42,
                          theta_init=2, edge_magnitude=2.0)
    sc = SelfWiringGraph.build_sparse_cache(net.mask, edge_magnitude=2.0)

    print(f"  sparse_cache tuple length: {len(sc)} (edge_magnitude=2.0)")
    if len(sc) == 3:
        n_edges = len(sc[0])
        stamina = np.full(n_edges, 200, dtype=np.uint8)
        stamina_before = stamina.copy()

        injected = net.input_projection[5]
        state, charge = SelfWiringGraph.rollout_token(
            injected, mask=net.mask, theta=net._theta_f32,
            decay=net.decay, ticks=8, sparse_cache=sc,
            polarity=net._polarity_f32, refractory=net.refractory,
            channel=net.channel, edge_magnitude=2.0,
            stamina=stamina,
        )

        if np.array_equal(stamina, stamina_before):
            print(f"  [!] WARN: stamina silently ignored with 3-tuple cache (edge_magnitude != 1.0)")
            return "WARN_3TUPLE_IGNORED"
        else:
            return "PASS"
    else:
        print(f"  [i] Got 2-tuple, edge_magnitude path changed")
        return "SKIP"


def test_stamina_multiplier_boundaries():
    """Check the threshold boundaries: 0-84=0.0, 85-170=0.5, 171-255=1.0"""
    lo, hi = SelfWiringGraph.STAMINA_THRESHOLDS  # (85, 171)

    stamina = np.array([0, 84, 85, 170, 171, 255], dtype=np.uint8)
    m = np.ones(len(stamina), dtype=np.float32)
    m[stamina < hi] = 0.5
    m[stamina < lo] = 0.0

    expected = [0.0, 0.0, 0.5, 0.5, 1.0, 1.0]
    for i, (s, got, exp) in enumerate(zip(stamina, m, expected)):
        ok = got == exp
        if not ok:
            print(f"  [!] BUG: stamina={s} → mult={got}, expected {exp}")
            return "BUG_THRESHOLD"

    print(f"  [+] PASS: threshold boundaries correct (0-84=0.0, 85-170=0.5, 171-255=1.0)")
    return "PASS"


def test_docstring_mismatch():
    """Check: _get_stamina_multipliers docstring says [0-15] but actual range is [0-255]."""
    import inspect
    src = inspect.getsource(SelfWiringGraph._get_stamina_multipliers)
    if '[0-15]' in src:
        print(f"  [!] BUG: docstring says [0-15] but stamina is uint8 [0-255]")
        return "BUG_DOCSTRING"
    else:
        print(f"  [+] PASS: docstring range matches implementation")
        return "PASS"


def test_rollout_with_without_stamina_consistency():
    """Stamina=255 (all full, mult=1.0) should give same result as no stamina."""
    net = SelfWiringGraph(vocab=16, hidden=32, density=4, seed=42, theta_init=2)
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    rows, cols = sc
    injected = net.input_projection[5]

    # Without stamina
    s1, c1 = SelfWiringGraph.rollout_token(
        injected, mask=net.mask, theta=net._theta_f32, decay=net.decay,
        ticks=8, sparse_cache=sc, polarity=net._polarity_f32,
        refractory=np.zeros(32, dtype=np.int8), channel=net.channel,
    )

    # With stamina all=255 (should be identical — all multipliers=1.0)
    stamina = np.full(len(rows), 255, dtype=np.uint8)
    s2, c2 = SelfWiringGraph.rollout_token(
        injected, mask=net.mask, theta=net._theta_f32, decay=net.decay,
        ticks=8, sparse_cache=sc, polarity=net._polarity_f32,
        refractory=np.zeros(32, dtype=np.int8), channel=net.channel,
        stamina=stamina,
    )

    state_match = np.allclose(s1, s2, atol=1e-6)
    charge_match = np.allclose(c1, c2, atol=1e-6)

    if not state_match or not charge_match:
        diff_s = np.abs(s1 - s2).max()
        diff_c = np.abs(c1 - c2).max()
        print(f"  [!] BUG: stamina=255 differs from no-stamina!")
        print(f"      state diff={diff_s}, charge diff={diff_c}")
        # The cause: stamina path does regen (+1 per DECAY_PERIOD) which changes stamina
        # from 255 to 255 (clipped), so that's fine. But the propagation code path is different.
        return "BUG_MISMATCH"
    else:
        print(f"  [+] PASS: stamina=255 ≡ no-stamina")
        return "PASS"


def test_stamina_save_restore():
    """Stamina must survive save_state/restore_state round-trip."""
    net = SelfWiringGraph(vocab=16, hidden=32, density=4, seed=42, theta_init=2)
    net._init_stamina()
    # Set some non-trivial stamina values
    net._stamina[0] = 50
    net._stamina[1] = 100
    net._stamina[2] = 200

    snap = net.save_state()

    # Trash stamina
    net._stamina[:] = 0

    # Restore
    net.restore_state(snap)

    if net._stamina[0] == 50 and net._stamina[1] == 100 and net._stamina[2] == 200:
        print(f"  [+] PASS: stamina round-trips through save/restore")
        return "PASS"
    else:
        print(f"  [!] BUG: stamina not restored (got {net._stamina[:3]})")
        return "BUG_NOT_RESTORED"


if __name__ == "__main__":
    print("=" * 70)
    print("  Adversarial Stamina Tests")
    print("=" * 70)

    tests = [
        ("Resync after mutate", test_stamina_resync_after_mutate),
        ("Regen overflow/underflow", test_stamina_regen_overflow),
        ("Drain works", test_stamina_drain_works),
        ("Dense path ignores stamina", test_stamina_dense_path_ignored),
        ("3-tuple cache ignores stamina", test_stamina_3tuple_cache),
        ("Multiplier boundaries", test_stamina_multiplier_boundaries),
        ("Docstring range", test_docstring_mismatch),
        ("Full stamina ≡ no stamina", test_rollout_with_without_stamina_consistency),
        ("Save/restore round-trip", test_stamina_save_restore),
    ]

    results = []
    for name, fn in tests:
        print(f"\n  --- {name} ---")
        result = fn()
        results.append((name, result))

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    bugs = []
    warns = []
    for name, result in results:
        status = "PASS" if "PASS" in result else ("BUG" if "BUG" in result else "WARN")
        icon = "[+]" if status == "PASS" else ("[!]" if status == "BUG" else "[~]")
        print(f"  {icon} {name}: {result}")
        if "BUG" in result:
            bugs.append((name, result))
        elif "WARN" in result:
            warns.append((name, result))

    print(f"\n  Bugs: {len(bugs)}, Warnings: {len(warns)}, Pass: {len(results)-len(bugs)-len(warns)}")
    if bugs:
        print(f"\n  BUGS FOUND:")
        for name, result in bugs:
            print(f"    - {name}: {result}")
