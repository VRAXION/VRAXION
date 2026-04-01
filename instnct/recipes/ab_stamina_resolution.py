"""
INSTNCT — Stamina Resolution A/B: Stepped vs Continuous
========================================================
A) Current: 3-step {0.0, 0.5, 1.0} thresholds, drain=-1, regen=+1/6tick
B) Continuous: stamina/255 smooth multiplier, drain=-2, regen=+1/6tick

Test on L2 Alt pattern with loop injection at plateau.
"""
import sys, time, random
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

VOCAB = 256; TICKS = 8; INPUT_DURATION = 2
PLATEAU_WINDOW = 500
PRED_NEURONS = list(range(0, 10))


def make_alternating(rng, n=48):
    a, b = rng.randint(0, 10, size=2)
    while b == a:
        b = rng.randint(0, 10)
    return [a if i % 2 == 0 else b for i in range(n + 1)]


def rollout_with_stamina(injected, *, mask, theta, decay, ticks, input_duration,
                         state, charge, sparse_cache, polarity, refractory,
                         channel, stamina, mode='stepped'):
    """Custom rollout with configurable stamina mode."""
    mask = np.asarray(mask)
    theta = np.asarray(theta, dtype=np.float32)
    decay = np.asarray(decay, dtype=np.float32)
    H = mask.shape[0]
    injected = np.asarray(injected, dtype=np.float32)

    act = np.zeros(H, dtype=np.float32) if state is None else state.copy()
    cur_charge = np.zeros(H, dtype=np.float32) if charge is None else charge.copy()
    rows_sp, cols_sp = sparse_cache

    _dp = max(1, int(round(1.0 / max(float(np.mean(decay)), 0.001))))
    DRAIN_AMOUNT = 2 if mode == 'continuous' else 1

    for tick in range(int(ticks)):
        # 1. DECAY
        if _dp > 0 and tick % _dp == 0:
            cur_charge = np.maximum(cur_charge - 1.0, 0.0)

        # 2. INPUT
        if tick < int(input_duration):
            act = act + injected

        # 3. PROPAGATE with stamina
        if len(rows_sp):
            # Regen: +1 every 6 ticks
            if tick % 6 == 0:
                stamina[:] = np.clip(stamina.astype(np.int16) + 1, 0, 255).astype(np.uint8)

            # Multiplier
            if mode == 'continuous':
                s_mult = stamina.astype(np.float32) / 255.0
            else:
                lo, hi = 85, 171
                s_mult = np.ones(len(rows_sp), dtype=np.float32)
                s_mult[stamina < hi] = 0.5
                s_mult[stamina < lo] = 0.0

            raw = np.zeros(H, dtype=np.float32)
            np.add.at(raw, cols_sp, act[rows_sp] * s_mult)
        else:
            raw = np.zeros(H, dtype=np.float32)

        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        cur_charge += raw
        np.clip(cur_charge, 0.0, 15.0, out=cur_charge)

        # 5. SPIKE with wave gating
        theta_mult = SelfWiringGraph.WAVE_LUT[channel, tick % 8]
        eff_theta = np.clip(theta * theta_mult, 1.0, 15.0)
        can_fire = (refractory == 0)
        fired = (cur_charge >= eff_theta) & can_fire
        refractory[refractory > 0] -= 1
        refractory[fired] = 1
        act = fired.astype(np.float32)
        if polarity is not None:
            act = act * polarity
        cur_charge[fired] = 0.0

        # 7. DRAIN
        if len(rows_sp):
            fired_sources = fired[rows_sp]
            if np.any(fired_sources):
                stamina[fired_sources] = np.clip(
                    stamina[fired_sources].astype(np.int16) - DRAIN_AMOUNT,
                    0, 255).astype(np.uint8)

    return act, cur_charge


def eval_net(net, seqs, stamina, mode):
    net.reset()
    H = net.H
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0

    for seq in seqs:
        for i in range(len(seq) - 1):
            injected = net.input_projection[int(seq[i])]
            stam_copy = stamina.copy()  # don't drain eval stamina permanently
            state, charge = rollout_with_stamina(
                injected, mask=net.mask, theta=net._theta_f32,
                decay=net.decay, ticks=TICKS, input_duration=INPUT_DURATION,
                state=state, charge=charge, sparse_cache=sc,
                polarity=net._polarity_f32, refractory=net.refractory,
                channel=net.channel, stamina=stam_copy, mode=mode,
            )
            pred_charges = charge[PRED_NEURONS] if max(PRED_NEURONS) < H else np.zeros(10)
            if int(np.argmax(pred_charges)) == int(seq[i + 1]):
                correct += 1
            total += 1

    return correct / total if total else 0.0


def insert_loop(net):
    H = net.H
    length = random.randint(3, 5)
    nodes = random.sample(range(H), min(length, H))
    for i in range(len(nodes)):
        r, c = nodes[i], nodes[(i + 1) % len(nodes)]
        if r != c and not net.mask[r, c]:
            net.mask[r, c] = True
    net.resync_alive()


def run_arm(label, mode, seed, steps, eval_sets):
    random.seed(seed); np.random.seed(seed)
    net = SelfWiringGraph(vocab=VOCAB, hidden=64, density=4,
                          theta_init=1, decay_init=0.10, seed=seed)
    n_edges = len(net.alive)
    stamina = np.full(n_edges, 255, dtype=np.uint8)

    def avg_acc():
        return np.mean([eval_net(net, [s], stamina, mode) for s in eval_sets])

    best_acc = avg_acc()
    accepts = 0; stale = 0; injections = 0
    log_every = max(1, steps // 20)

    for step in range(1, steps + 1):
        if stale >= PLATEAU_WINDOW:
            insert_loop(net)
            # Resize stamina for new edges
            n_new = len(net.alive)
            if n_new > len(stamina):
                stamina = np.concatenate([stamina,
                    np.full(n_new - len(stamina), 255, dtype=np.uint8)])
            elif n_new < len(stamina):
                stamina = stamina[:n_new]
            injections += 1; stale = 0
            new_acc = avg_acc()
            if new_acc > best_acc:
                best_acc = new_acc; accepts += 1

        state_snap = net.save_state()
        old_stamina = stamina.copy()
        undo = net.mutate()
        # Resize stamina after mutate
        n_now = len(net.alive)
        if n_now > len(stamina):
            stamina = np.concatenate([stamina,
                np.full(n_now - len(stamina), 255, dtype=np.uint8)])
        elif n_now < len(stamina):
            stamina = stamina[:n_now]

        new_acc = avg_acc()
        if new_acc > best_acc:
            best_acc = new_acc; accepts += 1; stale = 0
        else:
            net.restore_state(state_snap)
            stamina = old_stamina
            stale += 1

        if step % log_every == 0:
            s_min, s_max, s_std = int(stamina.min()), int(stamina.max()), float(stamina.std())
            inj_str = f" inj={injections}" if injections > 0 else ""
            print(f"    {label:>12} step {step:5d} | acc={best_acc:.3f} | edges={len(net.alive):4d} "
                  f"| stam=[{s_min},{s_max}] std={s_std:.1f} | acc#={accepts}{inj_str}")

    return best_acc, accepts, injections, len(net.alive), stamina


def main():
    master_rng = np.random.RandomState(99)
    eval_sets = [make_alternating(master_rng) for _ in range(5)]
    STEPS = 5000; SEED = 42

    print("=" * 95)
    print("  Stamina Resolution A/B: Stepped vs Continuous")
    print(f"  Task: L2 Alt | H=64 | Steps: {STEPS}")
    print(f"  A) Stepped: {{0.0, 0.5, 1.0}}, drain=-1")
    print(f"  B) Continuous: stamina/255, drain=-2")
    print(f"  C) No stamina (baseline)")
    print("=" * 95)

    results = []

    # C) No stamina baseline (use standard rollout)
    print(f"\n>>> No stamina (baseline)")
    random.seed(SEED); np.random.seed(SEED)
    net_c = SelfWiringGraph(vocab=VOCAB, hidden=64, density=4, theta_init=1, decay_init=0.10, seed=SEED)
    def eval_c(net, seqs):
        net.reset()
        H = net.H; sc = SelfWiringGraph.build_sparse_cache(net.mask)
        state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
        correct = 0; total = 0
        for seq in seqs:
            for i in range(len(seq) - 1):
                state, charge = SelfWiringGraph.rollout_token(
                    net.input_projection[int(seq[i])], mask=net.mask, theta=net._theta_f32,
                    decay=net.decay, ticks=TICKS, input_duration=INPUT_DURATION,
                    state=state, charge=charge, sparse_cache=sc,
                    polarity=net._polarity_f32, refractory=net.refractory, channel=net.channel)
                if int(np.argmax(charge[PRED_NEURONS])) == int(seq[i+1]):
                    correct += 1
                total += 1
        return correct / total if total else 0.0

    best_c = np.mean([eval_c(net_c, [s]) for s in eval_sets])
    acc_c = 0; stale_c = 0; inj_c = 0
    t0 = time.time()
    for step in range(1, STEPS + 1):
        if stale_c >= PLATEAU_WINDOW:
            insert_loop(net_c); inj_c += 1; stale_c = 0
            new = np.mean([eval_c(net_c, [s]) for s in eval_sets])
            if new > best_c: best_c = new; acc_c += 1
        snap = net_c.save_state()
        net_c.mutate()
        new = np.mean([eval_c(net_c, [s]) for s in eval_sets])
        if new > best_c: best_c = new; acc_c += 1; stale_c = 0
        else: net_c.restore_state(snap); stale_c += 1
    tc = time.time() - t0
    print(f"    Done in {tc:.0f}s: acc={best_c:.3f}")
    results.append(("No stamina", best_c, acc_c, inj_c))

    # A) Stepped
    print(f"\n>>> Stepped (current)")
    t0 = time.time()
    acc_a, accepts_a, inj_a, edges_a, stam_a = run_arm("Stepped", 'stepped', SEED, STEPS, eval_sets)
    print(f"    Done in {time.time()-t0:.0f}s")
    results.append(("Stepped", acc_a, accepts_a, inj_a))

    # B) Continuous
    print(f"\n>>> Continuous (smooth)")
    t0 = time.time()
    acc_b, accepts_b, inj_b, edges_b, stam_b = run_arm("Continuous", 'continuous', SEED, STEPS, eval_sets)
    print(f"    Done in {time.time()-t0:.0f}s")
    results.append(("Continuous", acc_b, accepts_b, inj_b))

    # Summary
    print(f"\n{'='*95}")
    print(f"  RESULTS")
    print(f"{'='*95}")
    print(f"  {'':>14} | {'Acc':>5} | {'Acc#':>4} | {'Inj':>3}")
    for label, acc, accepts, inj in results:
        print(f"  {label:>14} | {acc:5.3f} | {accepts:4d} | {inj:3d}")

    best = max(results, key=lambda x: x[1])
    print(f"\n  Winner: {best[0]} ({best[1]:.3f})")

    # Stamina distribution analysis
    if stam_a is not None and stam_b is not None:
        print(f"\n  Stamina distributions after training:")
        print(f"    Stepped:    [{stam_a.min()}, {stam_a.max()}] std={stam_a.std():.1f}")
        print(f"    Continuous: [{stam_b.min()}, {stam_b.max()}] std={stam_b.std():.1f}")


if __name__ == "__main__":
    main()
