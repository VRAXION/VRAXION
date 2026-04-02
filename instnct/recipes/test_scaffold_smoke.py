"""
INSTNCT — Scaffolded Network: smoke test + telemetry
=====================================================
Quick test: does the layered scaffold approach work at all?

Architecture:
  Layer 1 (mass):   H=20, standard self-wiring
  Layer 2 (frozen): H=9 (3 triangle loops), connections frozen
  Layer 3 (mass):   H=20, standard self-wiring
  IO edges between layers

Phases:
  1. Build layered structure
  2. Train with frozen scaffold
  3. Melt: unfreeze everything into flat H=49
  4. Continue training

Telemetry: accuracy curve, edge counts per layer, fire rates,
           where accepts come from (which layer improves)
"""
import sys, time, random
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

VOCAB = 256; TICKS = 8; INPUT_DURATION = 2
PRED_NEURONS = list(range(0, 10))


def make_cycle3(rng, n=30):
    vals = list(rng.choice(10, size=3, replace=False))
    return [vals[i % 3] for i in range(n + 1)]


def build_scaffolded(seed):
    """Build the 3-layer scaffolded network."""
    L1 = 20; L2 = 9; L3 = 20
    H = L1 + L2 + L3  # 49 total

    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=0,
                          theta_init=3, decay_init=0.15, seed=seed)
    net.mask[:] = False

    # Layer 1: random sparse (neurons 0-19)
    rng = np.random.RandomState(seed)
    for _ in range(int(L1 * L1 * 0.06)):  # ~6% density
        r, c = rng.randint(0, L1), rng.randint(0, L1)
        if r != c: net.mask[r, c] = True

    # Layer 2: 3 frozen triangle loops (neurons 20-28)
    base = L1
    for loop in range(3):
        a = base + loop * 3
        b = base + loop * 3 + 1
        c = base + loop * 3 + 2
        net.mask[a, b] = True; net.mask[b, c] = True; net.mask[c, a] = True

    # Layer 3: random sparse (neurons 29-48)
    base3 = L1 + L2
    for _ in range(int(L3 * L3 * 0.06)):
        r, c = rng.randint(base3, base3 + L3), rng.randint(base3, base3 + L3)
        if r != c and r < H and c < H: net.mask[r, c] = True

    # IO edges: Layer1 → Layer2 (4 edges)
    for _ in range(4):
        src = rng.randint(0, L1)
        dst = rng.randint(L1, L1 + L2)
        net.mask[src, dst] = True

    # IO edges: Layer2 → Layer3 (4 edges)
    for _ in range(4):
        src = rng.randint(L1, L1 + L2)
        dst = rng.randint(L1 + L2, H)
        net.mask[src, dst] = True

    # Some Layer3 → output zone feedback
    for _ in range(2):
        src = rng.randint(L1 + L2, H)
        dst = rng.randint(0, min(10, H))
        if src != dst: net.mask[src, dst] = True

    net.resync_alive()

    # Track which edges are "frozen" (layer 2 internal)
    frozen_edges = set()
    for loop in range(3):
        a = L1 + loop * 3
        b = L1 + loop * 3 + 1
        c = L1 + loop * 3 + 2
        frozen_edges.add((a, b)); frozen_edges.add((b, c)); frozen_edges.add((c, a))

    return net, frozen_edges, (L1, L2, L3)


def eval_acc(net, seqs):
    net.reset()
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    H = net.H
    st = np.zeros(H, dtype=np.float32); ch = np.zeros(H, dtype=np.float32)
    c = 0; t = 0
    for seq in seqs:
        for i in range(len(seq) - 1):
            st, ch = SelfWiringGraph.rollout_token(
                net.input_projection[int(seq[i])], mask=net.mask,
                theta=net._theta_f32, decay=net.decay, ticks=TICKS,
                input_duration=INPUT_DURATION, state=st, charge=ch,
                sparse_cache=sc, polarity=net._polarity_f32,
                refractory=net.refractory, channel=net.channel)
            if int(np.argmax(ch[PRED_NEURONS])) == int(seq[i+1]): c += 1
            t += 1
    return c / t if t else 0.0


def get_telemetry(net, seqs, layers):
    """Deep telemetry: per-layer fire rates, charge, edge utilization."""
    L1, L2, L3 = layers
    H = net.H; net.reset()
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    st = np.zeros(H, dtype=np.float32); ch = np.zeros(H, dtype=np.float32)
    ref = np.zeros(H, dtype=np.int8)
    _dp = max(1, int(round(1.0 / max(float(np.mean(net.decay)), 0.001))))

    fires = np.zeros(H); n_ticks = 0

    for seq in seqs[:1]:
        for idx in range(min(15, len(seq)-1)):
            st, ch = SelfWiringGraph.rollout_token(
                net.input_projection[int(seq[idx])], mask=net.mask,
                theta=net._theta_f32, decay=net.decay, ticks=TICKS,
                input_duration=INPUT_DURATION, state=st, charge=ch,
                sparse_cache=sc, polarity=net._polarity_f32,
                refractory=net.refractory, channel=net.channel)
            fires += (np.abs(st) > 0).astype(np.float32)
            n_ticks += 1

    fr = fires / max(n_ticks, 1)
    # Per-layer stats
    l1_fr = float(np.mean(fr[:L1]))
    l2_fr = float(np.mean(fr[L1:L1+L2]))
    l3_fr = float(np.mean(fr[L1+L2:]))

    # Edge counts per zone
    rows, cols = np.where(net.mask)
    l1_edges = sum(1 for r, c in zip(rows, cols) if r < L1 and c < L1)
    l2_edges = sum(1 for r, c in zip(rows, cols) if L1 <= r < L1+L2 and L1 <= c < L1+L2)
    l3_edges = sum(1 for r, c in zip(rows, cols) if r >= L1+L2 and c >= L1+L2)
    cross_edges = len(rows) - l1_edges - l2_edges - l3_edges

    return {
        'l1_fr': l1_fr, 'l2_fr': l2_fr, 'l3_fr': l3_fr,
        'l1_e': l1_edges, 'l2_e': l2_edges, 'l3_e': l3_edges,
        'cross_e': cross_edges, 'total_e': len(rows),
    }


def mutate_respecting_frozen(net, frozen_edges):
    """Mutate but protect frozen edges from removal/rewire."""
    snap_mask = net.mask.copy()
    undo = net.mutate()
    # Restore any frozen edge that was removed
    for r, c in frozen_edges:
        if not net.mask[r, c] and snap_mask[r, c]:
            net.mask[r, c] = True
    net.resync_alive()
    return undo


def melt_network(net, frozen_edges):
    """Remove the frozen constraint — all edges now free to mutate."""
    # The edges stay but are no longer protected
    return set()  # empty frozen set = nothing protected


def main():
    SEED = 42; STEPS_SCAFFOLD = 1500; STEPS_MELT = 1500

    master_rng = np.random.RandomState(77)
    eval_seqs = [make_cycle3(master_rng, 30) for _ in range(3)]

    print("=" * 90)
    print("  Scaffolded Network: smoke test + telemetry")
    print(f"  Architecture: Mass(20) → Frozen Loops(9) → Mass(20) = H=49")
    print(f"  Scaffold phase: {STEPS_SCAFFOLD} steps | Melt phase: {STEPS_MELT} steps")
    print("=" * 90)

    # Build scaffolded network
    net, frozen_edges, layers = build_scaffolded(SEED)
    random.seed(SEED); np.random.seed(SEED)

    t0_init = get_telemetry(net, eval_seqs, layers)
    init_acc = eval_acc(net, eval_seqs)

    print(f"\n  INIT STATE:")
    print(f"    acc={init_acc:.3f}")
    print(f"    L1: {t0_init['l1_e']} edges, FR={t0_init['l1_fr']:.3f}")
    print(f"    L2: {t0_init['l2_e']} edges (frozen loops), FR={t0_init['l2_fr']:.3f}")
    print(f"    L3: {t0_init['l3_e']} edges, FR={t0_init['l3_fr']:.3f}")
    print(f"    Cross: {t0_init['cross_e']} edges")
    print(f"    Total: {t0_init['total_e']} edges, frozen: {len(frozen_edges)}")

    # Phase 1: Train with scaffold
    print(f"\n  === PHASE 1: Scaffold training ({STEPS_SCAFFOLD} steps) ===")
    best = init_acc; accepts = 0
    acc_curve = [(0, init_acc, 'scaffold')]
    LOG = max(1, STEPS_SCAFFOLD // 10)

    for step in range(1, STEPS_SCAFFOLD + 1):
        snap = net.save_state()
        mutate_respecting_frozen(net, frozen_edges)
        new = eval_acc(net, eval_seqs)
        if new > best:
            best = new; accepts += 1
        else:
            net.restore_state(snap)

        if step % LOG == 0:
            t = get_telemetry(net, eval_seqs, layers)
            acc_curve.append((step, best, 'scaffold'))
            print(f"    step {step:5d} | acc={best:.3f} | "
                  f"L1:{t['l1_e']}e FR={t['l1_fr']:.2f} | "
                  f"L2:{t['l2_e']}e FR={t['l2_fr']:.2f} | "
                  f"L3:{t['l3_e']}e FR={t['l3_fr']:.2f} | "
                  f"cross:{t['cross_e']} | acc#={accepts}")

    pre_melt = best
    pre_melt_t = get_telemetry(net, eval_seqs, layers)

    # Phase 2: MELT — unfreeze everything
    print(f"\n  === MELT: unfreezing all edges ===")
    frozen_edges = melt_network(net, frozen_edges)
    post_melt_acc = eval_acc(net, eval_seqs)
    print(f"    Pre-melt acc:  {pre_melt:.3f}")
    print(f"    Post-melt acc: {post_melt_acc:.3f} (should be same — melt doesn't change edges)")

    # Phase 3: Continue training (melted — full self-wiring)
    print(f"\n  === PHASE 2: Melted training ({STEPS_MELT} steps) ===")
    best = post_melt_acc; melt_accepts = 0

    for step in range(1, STEPS_MELT + 1):
        snap = net.save_state()
        net.mutate()  # full mutation — no frozen protection
        new = eval_acc(net, eval_seqs)
        if new > best:
            best = new; melt_accepts += 1
        else:
            net.restore_state(snap)

        if step % LOG == 0:
            t = get_telemetry(net, eval_seqs, layers)
            acc_curve.append((STEPS_SCAFFOLD + step, best, 'melted'))
            print(f"    step {step:5d} | acc={best:.3f} | "
                  f"L1:{t['l1_e']}e FR={t['l1_fr']:.2f} | "
                  f"L2:{t['l2_e']}e FR={t['l2_fr']:.2f} | "
                  f"L3:{t['l3_e']}e FR={t['l3_fr']:.2f} | "
                  f"cross:{t['cross_e']} | acc#={melt_accepts}")

    final_t = get_telemetry(net, eval_seqs, layers)

    # Baseline: flat H=49 standard INSTNCT
    print(f"\n  === BASELINE: flat H=49 standard ({STEPS_SCAFFOLD + STEPS_MELT} steps) ===")
    random.seed(SEED); np.random.seed(SEED)
    flat = SelfWiringGraph(vocab=VOCAB, hidden=49, density=4,
                           theta_init=1, decay_init=0.10, seed=SEED)
    flat_best = eval_acc(flat, eval_seqs); flat_acc = 0
    for step in range(1, STEPS_SCAFFOLD + STEPS_MELT + 1):
        snap = flat.save_state(); flat.mutate()
        new = eval_acc(flat, eval_seqs)
        if new > flat_best: flat_best = new; flat_acc += 1
        else: flat.restore_state(snap)
        if step % (LOG * 2) == 0:
            print(f"    step {step:5d} | acc={flat_best:.3f} | edges={len(flat.alive)} | acc#={flat_acc}")

    # Summary
    print(f"\n{'='*90}")
    print(f"  RESULTS")
    print(f"{'='*90}")
    print(f"  Init:                {init_acc:.3f}")
    print(f"  After scaffold:      {pre_melt:.3f} ({accepts} accepts)")
    print(f"  After melt+train:    {best:.3f} ({melt_accepts} more accepts)")
    print(f"  Flat baseline:       {flat_best:.3f} ({flat_acc} accepts)")
    print(f"  Scaffold advantage:  {best - flat_best:+.3f}")

    print(f"\n  TELEMETRY EVOLUTION:")
    print(f"  {'Phase':>10} | {'L1 edges':>8} | {'L2 edges':>8} | {'L3 edges':>8} | {'Cross':>5} | {'L1 FR':>5} | {'L2 FR':>5} | {'L3 FR':>5}")
    print(f"  {'init':>10} | {t0_init['l1_e']:8d} | {t0_init['l2_e']:8d} | {t0_init['l3_e']:8d} | {t0_init['cross_e']:5d} | {t0_init['l1_fr']:5.3f} | {t0_init['l2_fr']:5.3f} | {t0_init['l3_fr']:5.3f}")
    print(f"  {'scaffold':>10} | {pre_melt_t['l1_e']:8d} | {pre_melt_t['l2_e']:8d} | {pre_melt_t['l3_e']:8d} | {pre_melt_t['cross_e']:5d} | {pre_melt_t['l1_fr']:5.3f} | {pre_melt_t['l2_fr']:5.3f} | {pre_melt_t['l3_fr']:5.3f}")
    print(f"  {'final':>10} | {final_t['l1_e']:8d} | {final_t['l2_e']:8d} | {final_t['l3_e']:8d} | {final_t['cross_e']:5d} | {final_t['l1_fr']:5.3f} | {final_t['l2_fr']:5.3f} | {final_t['l3_fr']:5.3f}")

    # Did the loops survive the melt?
    survived = 0; total_frozen = 9  # 3 loops × 3 edges
    for loop in range(3):
        a = 20 + loop * 3; b = a + 1; c = a + 2
        if net.mask[a, b] and net.mask[b, c] and net.mask[c, a]:
            survived += 1
    print(f"\n  Loop survival after melt: {survived}/3 intact")

    # Accuracy curve
    print(f"\n  ACCURACY CURVE:")
    for step, acc, phase in acc_curve:
        bar = "█" * int(acc * 40)
        print(f"    step {step:5d} [{phase:>8}] {acc:.3f} {bar}")


if __name__ == "__main__":
    main()
