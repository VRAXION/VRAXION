"""Unit tests for QuaternaryMask."""

import sys, os
import numpy as np
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))
from quaternary_mask import QuaternaryMask


# ------------------------------------------------------------------
# 1. Round-trip: bool mask -> quaternary -> bool mask
# ------------------------------------------------------------------

def test_roundtrip_random():
    """from_bool_mask -> to_bool_mask should be identity."""
    rng = np.random.RandomState(42)
    for H in [4, 16, 64, 256]:
        mask = rng.random((H, H)) < 0.05
        np.fill_diagonal(mask, False)
        qm = QuaternaryMask.from_bool_mask(mask)
        recovered = qm.to_bool_mask()
        assert np.array_equal(mask, recovered), f"Round-trip failed for H={H}"
    print("PASS: roundtrip_random")


def test_roundtrip_empty():
    mask = np.zeros((8, 8), dtype=bool)
    qm = QuaternaryMask.from_bool_mask(mask)
    assert np.array_equal(qm.to_bool_mask(), mask)
    assert qm.count_edges() == 0
    print("PASS: roundtrip_empty")


def test_roundtrip_bidir():
    """Bidirectional pair should encode as 3."""
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 2] = True
    mask[2, 0] = True  # bidir pair (0,2)
    qm = QuaternaryMask.from_bool_mask(mask)
    assert qm.get_pair(0, 2) == 3
    assert qm.get_pair(2, 0) == 3  # symmetric access
    assert qm.count_bidir() == 1
    assert qm.count_edges() == 2
    recovered = qm.to_bool_mask()
    assert np.array_equal(mask, recovered)
    print("PASS: roundtrip_bidir")


# ------------------------------------------------------------------
# 2. Edge list equivalence
# ------------------------------------------------------------------

def test_edge_list_matches_bool():
    """to_directed_edges() must produce same edge set as np.where(bool_mask)."""
    rng = np.random.RandomState(123)
    for H in [4, 32, 128]:
        mask = rng.random((H, H)) < 0.08
        np.fill_diagonal(mask, False)
        qm = QuaternaryMask.from_bool_mask(mask)
        # Reference: np.where on bool mask
        ref_rows, ref_cols = np.where(mask)
        ref_set = set(zip(ref_rows.tolist(), ref_cols.tolist()))
        # QuaternaryMask edges
        q_rows, q_cols = qm.to_directed_edges()
        q_set = set(zip(q_rows.tolist(), q_cols.tolist()))
        assert ref_set == q_set, f"Edge set mismatch for H={H}: diff={ref_set ^ q_set}"
    print("PASS: edge_list_matches_bool")


# ------------------------------------------------------------------
# 3. Exhaustive 4-neuron test
# ------------------------------------------------------------------

def test_exhaustive_4_neurons():
    """Test all 4 states on a small graph."""
    H = 4
    qm = QuaternaryMask(H)
    # Pair (0,1): forward only
    qm.set_pair(0, 1, 1)
    assert qm.get_pair(0, 1) == 1
    assert qm.get_pair(1, 0) == 2  # reversed perspective
    # Pair (0,2): backward only
    qm.set_pair(0, 2, 2)
    assert qm.get_pair(0, 2) == 2
    assert qm.get_pair(2, 0) == 1  # reversed perspective
    # Pair (1,3): bidir
    qm.set_pair(1, 3, 3)
    assert qm.get_pair(1, 3) == 3
    assert qm.get_pair(3, 1) == 3  # bidir is symmetric
    # Pair (2,3): empty
    assert qm.get_pair(2, 3) == 0
    assert qm.get_pair(3, 2) == 0

    # Check bool mask
    mask = qm.to_bool_mask()
    assert mask[0, 1] == True   # 0->1
    assert mask[1, 0] == False  # no 1->0
    assert mask[0, 2] == False  # no 0->2
    assert mask[2, 0] == True   # 2->0
    assert mask[1, 3] == True   # bidir
    assert mask[3, 1] == True   # bidir
    assert mask[2, 3] == False
    assert mask[3, 2] == False

    # Check counts
    assert qm.count_edges() == 4  # 0->1, 2->0, 1->3, 3->1
    assert qm.count_bidir() == 1  # pair (1,3)
    hist = qm.count_by_state()
    assert hist[0] == 3  # (0,3), (1,2), (2,3) are empty
    assert hist[1] == 1  # (0,1)
    assert hist[2] == 1  # (0,2)
    assert hist[3] == 1  # (1,3)
    print("PASS: exhaustive_4_neurons")


# ------------------------------------------------------------------
# 4. No self-loops
# ------------------------------------------------------------------

def test_no_self_loops():
    """Diagonal never appears in edge output."""
    rng = np.random.RandomState(99)
    mask = rng.random((32, 32)) < 0.1
    np.fill_diagonal(mask, False)
    qm = QuaternaryMask.from_bool_mask(mask)
    rows, cols = qm.to_directed_edges()
    assert not np.any(rows == cols), "Self-loop detected!"
    print("PASS: no_self_loops")


# ------------------------------------------------------------------
# 5. Mutation correctness + undo
# ------------------------------------------------------------------

def test_mutate_add_undo():
    rng = random.Random(42)
    qm = QuaternaryMask(16)
    original = qm.data.copy()
    undo = []
    qm.mutate_add(rng, undo)
    assert len(undo) == 1
    assert undo[0][0] == 'QA'
    assert qm.count_edges() == 1
    qm.apply_undo(undo)
    assert np.array_equal(qm.data, original)
    print("PASS: mutate_add_undo")


def test_mutate_remove_undo():
    rng = random.Random(42)
    mask = np.zeros((8, 8), dtype=bool)
    mask[1, 5] = True
    mask[3, 7] = True
    qm = QuaternaryMask.from_bool_mask(mask)
    original = qm.data.copy()
    edges_before = qm.count_edges()
    undo = []
    qm.mutate_remove(rng, undo)
    assert qm.count_edges() == edges_before - 1
    qm.apply_undo(undo)
    assert np.array_equal(qm.data, original)
    print("PASS: mutate_remove_undo")


def test_mutate_flip_undo():
    rng = random.Random(42)
    qm = QuaternaryMask(8)
    qm.set_pair(2, 5, 1)  # forward: 2->5
    original = qm.data.copy()
    undo = []
    qm.mutate_flip(rng, undo)
    assert len(undo) == 1
    # Should have flipped to 2 (backward: 5->2)
    assert qm.data[undo[0][1]] in (1, 2)
    assert qm.data[undo[0][1]] != undo[0][2]  # changed
    qm.apply_undo(undo)
    assert np.array_equal(qm.data, original)
    print("PASS: mutate_flip_undo")


def test_mutate_upgrade_undo():
    rng = random.Random(42)
    qm = QuaternaryMask(8)
    qm.set_pair(1, 4, 1)
    original = qm.data.copy()
    undo = []
    qm.mutate_upgrade(rng, undo)
    assert qm.data[undo[0][1]] == 3  # upgraded to bidir
    assert qm.count_bidir() == 1
    qm.apply_undo(undo)
    assert np.array_equal(qm.data, original)
    print("PASS: mutate_upgrade_undo")


def test_mutate_downgrade_undo():
    rng = random.Random(42)
    qm = QuaternaryMask(8)
    qm.set_pair(1, 4, 3)  # bidir
    original = qm.data.copy()
    undo = []
    qm.mutate_downgrade(rng, undo)
    assert qm.data[undo[0][1]] in (1, 2)
    assert qm.count_bidir() == 0
    qm.apply_undo(undo)
    assert np.array_equal(qm.data, original)
    print("PASS: mutate_downgrade_undo")


def test_mutate_rewire_undo():
    rng = random.Random(42)
    qm = QuaternaryMask(16)
    qm.set_pair(3, 10, 1)
    original = qm.data.copy()
    undo = []
    qm.mutate_rewire(rng, undo)
    if undo:  # rewire might fail if no dead pair found
        assert qm.count_edges() == 1  # same edge count
        qm.apply_undo(undo)
        assert np.array_equal(qm.data, original)
    print("PASS: mutate_rewire_undo")


# ------------------------------------------------------------------
# 6. Memory comparison
# ------------------------------------------------------------------

def test_memory_savings():
    H = 256
    bool_bytes = H * H  # np.bool_ = 1 byte
    qm = QuaternaryMask(H)
    quat_bytes = qm.memory_bytes
    ratio = quat_bytes / bool_bytes
    assert ratio < 0.55, f"Expected ~50% savings, got {ratio:.1%}"
    print(f"PASS: memory_savings (bool={bool_bytes}, quat={quat_bytes}, ratio={ratio:.1%})")


# ------------------------------------------------------------------
# Run all
# ------------------------------------------------------------------

if __name__ == '__main__':
    test_roundtrip_random()
    test_roundtrip_empty()
    test_roundtrip_bidir()
    test_edge_list_matches_bool()
    test_exhaustive_4_neurons()
    test_no_self_loops()
    test_mutate_add_undo()
    test_mutate_remove_undo()
    test_mutate_flip_undo()
    test_mutate_upgrade_undo()
    test_mutate_downgrade_undo()
    test_mutate_rewire_undo()
    test_memory_savings()
    print("\n=== ALL TESTS PASSED ===")
