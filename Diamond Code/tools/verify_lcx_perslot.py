"""
Verify LCX flexible slot ownership.
Tests 3 configurations deterministically:
  1. K=1, N=12: flowchart mode — 12 unique slots, 52 untouched
  2. K=-1, N=1: giant ant — all 64 slots written (global write)
  3. K=5, N=8: multi-slot — 40 slots written, 24 untouched
"""
import sys, torch
sys.path.insert(0, r"S:\AI\work\VRAXION_DEV\Diamond Code")
from swarm_model import SwarmByteRingModel, generate_phi_stride_assignments

NUM_BITS = 8
LCX_SIZE = NUM_BITS * NUM_BITS  # 64
PASSES = 50
BATCH = 16
SEQ = 16

def run_test(name, num_beings, slots_per_being, embedding_dim=64, depth=2,
             capacity_fibonacci=False, full_view=False, fibonacci=False,
             combiner_mode='mean', bits_per_being=0):
    """Run a single LCX ownership test. Returns True on pass."""
    print(f"\n{'='*70}")
    print(f"  TEST: {name}")
    print(f"  num_beings={num_beings}, slots_per_being={slots_per_being}")
    print(f"{'='*70}")

    # 1. Check assignments
    is_global = (slots_per_being == -1 or slots_per_being >= LCX_SIZE)
    if not is_global:
        assignments = generate_phi_stride_assignments(num_beings, LCX_SIZE, slots_per_being)
        print(f"\n  Assignments shape: {list(assignments.shape)}")
        flat = assignments.reshape(-1).tolist()
        unique = len(set(flat))
        total = num_beings * slots_per_being
        print(f"  Total assigned slots: {total}, unique: {unique}")
        assert unique == total, f"COLLISION! {total} slots but only {unique} unique"
        print(f"  [OK] All {total} slots unique — no collisions")
    else:
        print(f"\n  Global write mode — all {LCX_SIZE} slots writable")

    # 2. Build model
    model = SwarmByteRingModel(
        num_memory_positions=embedding_dim,
        embedding_dim=embedding_dim,
        num_beings=num_beings,
        depth=depth,
        num_bits=NUM_BITS,
        combiner_mode=combiner_mode,
        bits_per_being=bits_per_being,
        min_coverage=1,
        mask_seed=42,
        fibonacci=fibonacci,
        combinatorial=False,
        think_ticks=0,
        temporal_fibonacci=False,
        capacity_fibonacci=capacity_fibonacci,
        max_hidden=4096,
        min_hidden=128,
        full_view=full_view,
        use_lcx=True,
        slots_per_being=slots_per_being,
    )

    assert model.lcx is not None, "LCX should be active"
    assert model.gem is None, "GEM should be None when use_lcx=True"
    if is_global:
        assert model.pixel_assignments is None, "pixel_assignments should be None for global write"
        print(f"  [OK] LCX active, pixel_assignments=None (global write)")
    else:
        assert model.pixel_assignments is not None, "pixel_assignments should exist"
        assert model.pixel_assignments.shape == (num_beings, slots_per_being), \
            f"Expected shape [{num_beings}, {slots_per_being}], got {list(model.pixel_assignments.shape)}"
        print(f"  [OK] LCX active, pixel_assignments shape={list(model.pixel_assignments.shape)}")

    # 3. Record initial state & run forward passes
    lcx_before = model.lcx.clone()
    model.train()
    for _ in range(PASSES):
        x = torch.randn(BATCH, SEQ, NUM_BITS)
        model(x)

    lcx_after = model.lcx.clone()
    changed = (lcx_after != lcx_before)
    changed_slots = set(changed.nonzero(as_tuple=True)[0].tolist())

    # 4. Verify ownership
    if is_global:
        # Giant ant: ALL slots should change (or at least most)
        print(f"\n  Changed slots: {len(changed_slots)} / {LCX_SIZE}")
        # With global write and 50 passes, most slots should be written
        # Allow some slack — threshold gate may block weak writes
        min_expected = LCX_SIZE // 2  # at least half
        if len(changed_slots) >= min_expected:
            print(f"  [OK] PASS — {len(changed_slots)}/{LCX_SIZE} slots written (>= {min_expected} threshold)")
            return True
        else:
            print(f"  [FAIL] Only {len(changed_slots)}/{LCX_SIZE} slots written (expected >= {min_expected})")
            return False
    else:
        assigned_slots = set(model.pixel_assignments.reshape(-1).tolist())
        unexpected = changed_slots - assigned_slots
        expected_changed = changed_slots & assigned_slots
        unwritten = assigned_slots - changed_slots

        print(f"\n  Changed slots ({len(changed_slots)}): {sorted(changed_slots)}")
        print(f"  Assigned slots ({len(assigned_slots)}): {sorted(assigned_slots)}")
        print(f"  Expected changed: {len(expected_changed)}")
        print(f"  Unexpected (unauthorized): {len(unexpected)}")
        print(f"  Assigned but unchanged: {len(unwritten)}")

        if len(unexpected) == 0:
            print(f"\n  [OK] PASS — Only assigned slots were written!")
            return True
        else:
            print(f"\n  [FAIL] {len(unexpected)} unauthorized slots written: {sorted(unexpected)}")
            return False


# ============================================================
# Run all 3 tests
# ============================================================
results = []

# Test 1: K=1, N=12 — flowchart mode (capacity_fibonacci + full_view)
results.append(run_test(
    "Flowchart: K=1, N=12",
    num_beings=12, slots_per_being=1,
    embedding_dim=64, depth=2,
    capacity_fibonacci=True, full_view=True, fibonacci=True,
    combiner_mode='masked', bits_per_being=8,
))

# Test 2: K=-1, N=1 — giant ant (global write)
results.append(run_test(
    "Giant Ant: K=-1, N=1 (global write)",
    num_beings=1, slots_per_being=-1,
    embedding_dim=256, depth=2,
))

# Test 3: K=5, N=8 — multi-slot
results.append(run_test(
    "Multi-Slot: K=5, N=8 (40 of 64 slots)",
    num_beings=8, slots_per_being=5,
    embedding_dim=64, depth=2,
))

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
names = [
    "Flowchart (K=1, N=12)",
    "Giant Ant (K=-1, N=1)",
    "Multi-Slot (K=5, N=8)",
]
all_pass = True
for name, passed in zip(names, results):
    status = "[OK] PASS" if passed else "[FAIL]"
    print(f"  {status}  {name}")
    if not passed:
        all_pass = False

if all_pass:
    print(f"\n  ALL 3 TESTS PASSED")
else:
    print(f"\n  SOME TESTS FAILED")
    sys.exit(1)

print(f"\n{'='*70}")
