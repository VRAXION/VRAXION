"""
INSTNCT — Guided Mutation: stamina + charge tells you what to change
=====================================================================
A) Blind mutation (current: random edge, random op)
B) Guided mutation (stamina + charge based targeting)

Rules:
  - charge high + not firing → ADD edge to/from this neuron
  - stamina == max → REMOVE (never used)
  - stamina == 0 → REWIRE (burnt out)
  - always fires → theta UP
  - never fires → theta DOWN
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
H = 64; MAX_STAMINA = 15; REGEN_PERIOD = 2


def make_alternating(rng, n=40):
    a, b = rng.randint(0, 10, size=2)
    while b == a: b = rng.randint(0, 10)
    return [a if i % 2 == 0 else b for i in range(n + 1)]

def make_cycle3(rng, n=40):
    vals = list(rng.choice(10, size=3, replace=False))
    return [vals[i % 3] for i in range(n + 1)]


def probe_network(net, seqs):
    """Run sequences through the network, collect diagnostics.
    Returns: accuracy, per-neuron stats, per-edge stamina."""
    net.reset()
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    rows, cols = sc
    n_edges = len(rows)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    refractory = np.zeros(H, dtype=np.int8)
    stamina = np.full(n_edges, MAX_STAMINA, dtype=np.int16)
    _dp = max(1, int(round(1.0 / max(float(np.mean(net.decay)), 0.001))))

    fire_counts = np.zeros(H, dtype=np.int32)
    charge_accum = np.zeros(H, dtype=np.float32)
    n_ticks = 0
    correct = 0; total = 0

    for seq in seqs:
        for idx in range(len(seq) - 1):
            inj = net.input_projection[int(seq[idx])]
            act = state.copy(); cur = charge.copy()
            for tick in range(TICKS):
                if _dp > 0 and tick % _dp == 0:
                    cur = np.maximum(cur - 1.0, 0.0)
                if tick < INPUT_DURATION:
                    act = act + inj
                if len(rows):
                    if tick % REGEN_PERIOD == 0:
                        stamina[:] = np.clip(stamina + 1, 0, MAX_STAMINA)
                    s_mult = stamina.astype(np.float32) / float(MAX_STAMINA)
                    raw = np.zeros(H, dtype=np.float32)
                    np.add.at(raw, cols, act[rows] * s_mult)
                else:
                    raw = np.zeros(H, dtype=np.float32)
                cur += raw; np.clip(cur, 0.0, 15.0, out=cur)
                eff = np.clip(net._theta_f32 * SelfWiringGraph.WAVE_LUT[net.channel, tick % 8], 1.0, 15.0)
                can = (refractory == 0)
                fired = (cur >= eff) & can
                refractory[refractory > 0] -= 1; refractory[fired] = 1
                act = fired.astype(np.float32) * net._polarity_f32
                cur[fired] = 0.0
                if len(rows):
                    fs = fired[rows]
                    if np.any(fs):
                        stamina[fs] = np.clip(stamina[fs] - 1, 0, MAX_STAMINA)
                fire_counts += fired.astype(np.int32)
                charge_accum += cur
                n_ticks += 1
            state = act; charge = cur
            if int(np.argmax(charge[PRED_NEURONS])) == int(seq[idx + 1]):
                correct += 1
            total += 1

    acc = correct / total if total else 0.0
    avg_charge = charge_accum / max(n_ticks, 1)
    fire_rate = fire_counts.astype(np.float32) / max(n_ticks, 1)

    return acc, fire_rate, avg_charge, stamina, rows, cols


def guided_mutate(net, fire_rate, avg_charge, stamina, rows, cols):
    """Mutate based on network diagnostics. Returns undo info."""
    alive = net.alive
    n_edges = len(alive)

    # Decide WHAT to do based on diagnostics
    r = random.random()

    if r < 0.3 and n_edges > 0:
        # --- REMOVE: highest stamina edge (least used) ---
        if len(stamina) == n_edges:
            worst_idx = int(np.argmax(stamina))
            edge_r, edge_c = alive[worst_idx]
            if net.mask[edge_r, edge_c]:
                net.mask[edge_r, edge_c] = False
                net.resync_alive()
                return ('guided_remove', edge_r, edge_c)

    elif r < 0.6:
        # --- ADD: to neuron with highest charge but lowest fire rate ---
        # "pressure without release" → needs a new path
        pressure = avg_charge * (1.0 - fire_rate)
        pressure[pressure < 0] = 0
        if np.max(pressure) > 0:
            target = int(np.argmax(pressure))
            source = random.randint(0, H - 1)
            if source != target and not net.mask[source, target]:
                net.mask[source, target] = True
                net.resync_alive()
                return ('guided_add', source, target)

    elif r < 0.8 and n_edges > 0:
        # --- REWIRE: lowest stamina edge (burnt out) → move to pressure point ---
        if len(stamina) == n_edges:
            burnt_idx = int(np.argmin(stamina))
            old_r, old_c = alive[burnt_idx]
            # New target: high pressure neuron
            pressure = avg_charge * (1.0 - fire_rate)
            pressure[pressure < 0] = 0
            if np.max(pressure) > 0:
                new_target = int(np.argmax(pressure))
                if old_r != new_target and not net.mask[old_r, new_target]:
                    net.mask[old_r, old_c] = False
                    net.mask[old_r, new_target] = True
                    net.resync_alive()
                    return ('guided_rewire', old_r, old_c, new_target)

    else:
        # --- THETA: adjust based on fire rate ---
        always_fire = np.where(fire_rate > 0.8)[0]
        never_fire = np.where((fire_rate < 0.01) & (avg_charge > 0.1))[0]
        if len(always_fire) > 0:
            idx = int(random.choice(always_fire))
            old_theta = int(net.theta[idx])
            new_theta = min(15, old_theta + 1)
            net.theta[idx] = np.uint8(new_theta)
            net._theta_f32[idx] = float(new_theta)
            return ('guided_theta_up', idx, old_theta)
        elif len(never_fire) > 0:
            idx = int(random.choice(never_fire))
            old_theta = int(net.theta[idx])
            new_theta = max(1, old_theta - 1)
            net.theta[idx] = np.uint8(new_theta)
            net._theta_f32[idx] = float(new_theta)
            return ('guided_theta_down', idx, old_theta)

    # Fallback: standard random mutation
    undo = net.mutate()
    return ('fallback', undo)


def undo_guided(net, info):
    """Undo a guided mutation."""
    op = info[0]
    if op == 'guided_remove':
        _, r, c = info
        net.mask[r, c] = True
        net.resync_alive()
    elif op == 'guided_add':
        _, r, c = info
        net.mask[r, c] = False
        net.resync_alive()
    elif op == 'guided_rewire':
        _, old_r, old_c, new_target = info
        net.mask[old_r, new_target] = False
        net.mask[old_r, old_c] = True
        net.resync_alive()
    elif op == 'guided_theta_up':
        _, idx, old_val = info
        net.theta[idx] = np.uint8(old_val)
        net._theta_f32[idx] = float(old_val)
    elif op == 'guided_theta_down':
        _, idx, old_val = info
        net.theta[idx] = np.uint8(old_val)
        net._theta_f32[idx] = float(old_val)
    elif op == 'fallback':
        net.replay(info[1])


def eval_acc(net, seqs):
    """Quick accuracy eval (no stamina, standard path)."""
    net.reset()
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for seq in seqs:
        for i in range(len(seq) - 1):
            state, charge = SelfWiringGraph.rollout_token(
                net.input_projection[int(seq[i])], mask=net.mask,
                theta=net._theta_f32, decay=net.decay, ticks=TICKS,
                input_duration=INPUT_DURATION, state=state, charge=charge,
                sparse_cache=sc, polarity=net._polarity_f32,
                refractory=net.refractory, channel=net.channel)
            if int(np.argmax(charge[PRED_NEURONS])) == int(seq[i+1]):
                correct += 1
            total += 1
    return correct / total if total else 0.0


def run_arm(label, use_guided, task_fn, seed, steps, eval_seqs):
    random.seed(seed); np.random.seed(seed)
    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=4,
                          theta_init=1, decay_init=0.10, seed=seed)
    probe_seqs = [task_fn(np.random.RandomState(seed + 50 + i), 30) for i in range(2)]

    best_acc = eval_acc(net, eval_seqs)
    accepts = 0; stale = 0; guided_ops = {}
    log_every = max(1, steps // 15)

    for step in range(1, steps + 1):
        # Loop injection at plateau (both arms get this)
        if stale >= 400:
            nodes = random.sample(range(H), random.randint(3, 5))
            for i in range(len(nodes)):
                r, c = nodes[i], nodes[(i+1) % len(nodes)]
                if r != c and not net.mask[r, c]: net.mask[r, c] = True
            net.resync_alive(); stale = 0
            new = eval_acc(net, eval_seqs)
            if new > best_acc: best_acc = new; accepts += 1

        if use_guided:
            # Probe network state
            _, fr, ac, stam, rows, cols = probe_network(net, probe_seqs)
            # Save state
            snap = net.save_state()
            # Guided mutation
            info = guided_mutate(net, fr, ac, stam, rows, cols)
            op_type = info[0]
            guided_ops[op_type] = guided_ops.get(op_type, 0) + 1
            # Eval
            new = eval_acc(net, eval_seqs)
            if new > best_acc:
                best_acc = new; accepts += 1; stale = 0
            else:
                undo_guided(net, info)
                # Also restore full state to be safe
                net.restore_state(snap)
                stale += 1
        else:
            snap = net.save_state()
            net.mutate()
            new = eval_acc(net, eval_seqs)
            if new > best_acc:
                best_acc = new; accepts += 1; stale = 0
            else:
                net.restore_state(snap); stale += 1

        if step % log_every == 0:
            print(f"    {label:>10} step {step:5d} | acc={best_acc:.3f} "
                  f"| edges={len(net.alive)} | acc#={accepts}")

    return best_acc, accepts, guided_ops


def main():
    STEPS = 3000; SEED = 42

    tasks = [
        ("Alt", make_alternating),
        ("Cyc3", make_cycle3),
    ]

    print("=" * 80)
    print("  Guided vs Blind Mutation")
    print(f"  H={H} | Steps={STEPS}")
    print("=" * 80)

    results = []
    for task_name, task_fn in tasks:
        master_rng = np.random.RandomState(77)
        eval_seqs = [task_fn(master_rng, 40) for _ in range(3)]

        print(f"\n  === {task_name} ===")

        # A) Blind
        print(f"\n  >>> Blind")
        t0 = time.time()
        acc_a, accepts_a, _ = run_arm("Blind", False, task_fn, SEED, STEPS, eval_seqs)
        ta = time.time() - t0
        print(f"    Done in {ta:.0f}s")

        # B) Guided
        print(f"\n  >>> Guided")
        t0 = time.time()
        acc_b, accepts_b, ops_b = run_arm("Guided", True, task_fn, SEED, STEPS, eval_seqs)
        tb = time.time() - t0
        print(f"    Done in {tb:.0f}s")
        if ops_b:
            print(f"    Ops: {ops_b}")

        results.append((task_name, acc_a, accepts_a, acc_b, accepts_b, ops_b))

    print(f"\n{'='*80}")
    print(f"  RESULTS")
    print(f"{'='*80}")
    print(f"  {'Task':>6} | {'Blind':>6} | {'Guided':>7} | {'Delta':>6} | {'Winner':>7}")
    for name, acc_a, _, acc_b, _, _ in results:
        delta = acc_b - acc_a
        winner = "GUIDED" if delta > 0.005 else ("BLIND" if delta < -0.005 else "TIE")
        print(f"  {name:>6} | {acc_a:6.3f} | {acc_b:7.3f} | {delta:+6.3f} | {winner:>7}")

    # Guided ops breakdown
    for name, _, _, _, _, ops in results:
        if ops:
            total = sum(ops.values())
            print(f"\n  {name} guided ops ({total} total):")
            for op, count in sorted(ops.items(), key=lambda x: -x[1]):
                print(f"    {op}: {count} ({100*count/total:.0f}%)")


if __name__ == "__main__":
    main()
