"""
INSTNCT — Hebbian Self-Wiring A/B
===================================
A) Guided mutation (external: probe → decide → mutate)
B) Hebbian (internal: forward pass modifies topology)
C) Hebbian + guided (both)

Hebbian rules (during forward pass):
  source fired AND target fired → stamina += 2 (strengthen)
  source fired AND target NOT   → stamina -= 1 (weaken)
  source NOT fired              → stamina += 1 (rest)
  stamina == 0 → edge DIES (mask off)
  co-firing neurons without edge → edge BORN (if < cap)
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
H = 64; MAX_STAMINA = 15
EDGE_BIRTH_THRESHOLD = 3  # co-fire this many ticks → new edge born
MAX_BIRTHS_PER_TOKEN = 2  # cap new edges per token (prevent explosion)


def make_alternating(rng, n=40):
    a, b = rng.randint(0, 10, size=2)
    while b == a: b = rng.randint(0, 10)
    return [a if i % 2 == 0 else b for i in range(n + 1)]

def make_cycle3(rng, n=40):
    vals = list(rng.choice(10, size=3, replace=False))
    return [vals[i % 3] for i in range(n + 1)]


def eval_hebbian(net, seq):
    """Forward pass WITH Hebbian self-wiring.
    The topology changes DURING inference."""
    net.reset()
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    rows, cols = sc
    n_edges = len(rows)
    state = np.zeros(H, dtype=np.float32)
    charge = np.zeros(H, dtype=np.float32)
    refractory = np.zeros(H, dtype=np.int8)
    stamina = np.full(n_edges, MAX_STAMINA, dtype=np.int16)
    _dp = max(1, int(round(1.0 / max(float(np.mean(net.decay)), 0.001))))

    # Co-fire tracker for edge birth
    cofire_count = np.zeros((H, H), dtype=np.int8)
    edges_died = 0; edges_born = 0

    correct = 0; total = 0

    for idx in range(len(seq) - 1):
        inj = net.input_projection[int(seq[idx])]
        act = state.copy(); cur = charge.copy()

        for tick in range(TICKS):
            if _dp > 0 and tick % _dp == 0:
                cur = np.maximum(cur - 1.0, 0.0)
            if tick < INPUT_DURATION:
                act = act + inj

            # Propagate with stamina
            rows, cols = np.where(net.mask)  # refresh after topology changes
            n_edges = len(rows)
            if n_edges > len(stamina):
                stamina = np.concatenate([stamina,
                    np.full(n_edges - len(stamina), MAX_STAMINA, dtype=np.int16)])
            elif n_edges < len(stamina):
                stamina = stamina[:n_edges]

            if n_edges > 0:
                s_mult = stamina[:n_edges].astype(np.float32) / float(MAX_STAMINA)
                raw = np.zeros(H, dtype=np.float32)
                np.add.at(raw, cols, act[rows] * s_mult)
            else:
                raw = np.zeros(H, dtype=np.float32)

            cur += raw
            np.clip(cur, 0.0, 15.0, out=cur)
            eff = np.clip(net._theta_f32 * SelfWiringGraph.WAVE_LUT[net.channel, tick % 8], 1.0, 15.0)
            can = (refractory == 0)
            fired = (cur >= eff) & can
            refractory[refractory > 0] -= 1; refractory[fired] = 1
            act = fired.astype(np.float32) * net._polarity_f32
            cur[fired] = 0.0

            # === HEBBIAN RULES ===
            if n_edges > 0:
                src_fired = fired[rows]
                tgt_fired = fired[cols]

                # Rule 1: source AND target fired → strengthen (+2)
                both = src_fired & tgt_fired
                if np.any(both):
                    stamina[both] = np.clip(stamina[both] + 2, 0, MAX_STAMINA)

                # Rule 2: source fired, target NOT → weaken (-1)
                src_only = src_fired & ~tgt_fired
                if np.any(src_only):
                    stamina[src_only] = np.clip(stamina[src_only] - 1, 0, MAX_STAMINA)

                # Rule 3: source NOT fired → rest (+1)
                resting = ~src_fired
                if np.any(resting):
                    stamina[resting] = np.clip(stamina[resting] + 1, 0, MAX_STAMINA)

                # DEATH: stamina == 0 → edge dies
                dead = np.where(stamina[:n_edges] == 0)[0]
                if len(dead) > 0:
                    for d in dead:
                        if d < len(rows):
                            net.mask[rows[d], cols[d]] = False
                    edges_died += len(dead)
                    net.resync_alive()
                    rows, cols = np.where(net.mask)
                    n_edges = len(rows)
                    stamina = stamina[:n_edges] if n_edges <= len(stamina) else np.concatenate([
                        stamina, np.full(n_edges - len(stamina), MAX_STAMINA, dtype=np.int16)])

            # BIRTH: co-firing neurons without edge → new edge
            if np.sum(fired) >= 2:
                fired_idx = np.where(fired)[0]
                births = 0
                for i in range(min(len(fired_idx), 5)):
                    for j in range(i + 1, min(len(fired_idx), 5)):
                        a_n, b_n = int(fired_idx[i]), int(fired_idx[j])
                        cofire_count[a_n, b_n] += 1
                        if cofire_count[a_n, b_n] >= EDGE_BIRTH_THRESHOLD:
                            if not net.mask[a_n, b_n] and a_n != b_n and births < MAX_BIRTHS_PER_TOKEN:
                                net.mask[a_n, b_n] = True
                                births += 1; edges_born += 1
                                cofire_count[a_n, b_n] = 0
                if births > 0:
                    net.resync_alive()
                    rows, cols = np.where(net.mask)
                    n_edges = len(rows)
                    if n_edges > len(stamina):
                        stamina = np.concatenate([stamina,
                            np.full(n_edges - len(stamina), MAX_STAMINA, dtype=np.int16)])

        state = act; charge = cur
        if int(np.argmax(charge[PRED_NEURONS])) == int(seq[idx + 1]):
            correct += 1
        total += 1

    acc = correct / total if total else 0.0
    return acc, edges_born, edges_died


def eval_standard(net, seq):
    """Standard eval: no self-wiring, no stamina."""
    net.reset()
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    st = np.zeros(H, dtype=np.float32); ch = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(seq) - 1):
        st, ch = SelfWiringGraph.rollout_token(
            net.input_projection[int(seq[i])], mask=net.mask,
            theta=net._theta_f32, decay=net.decay, ticks=TICKS,
            input_duration=INPUT_DURATION, state=st, charge=ch,
            sparse_cache=sc, polarity=net._polarity_f32,
            refractory=net.refractory, channel=net.channel)
        if int(np.argmax(ch[PRED_NEURONS])) == int(seq[i+1]):
            correct += 1
        total += 1
    return correct / total if total else 0.0


def run_arm(label, mode, task_fn, seed, steps, eval_seqs):
    random.seed(seed); np.random.seed(seed)
    net = SelfWiringGraph(vocab=VOCAB, hidden=H, density=4,
                          theta_init=1, decay_init=0.10, seed=seed)

    best_acc = eval_standard(net, eval_seqs[0])
    accepts = 0; stale = 0
    log_every = max(1, steps // 12)
    total_born = 0; total_died = 0

    for step in range(1, steps + 1):
        if stale >= 400:
            nodes = random.sample(range(H), random.randint(3, 5))
            for i in range(len(nodes)):
                r, c = nodes[i], nodes[(i+1) % len(nodes)]
                if r != c and not net.mask[r, c]: net.mask[r, c] = True
            net.resync_alive(); stale = 0

        snap = net.save_state()
        edges_before = len(net.alive)

        if mode == 'hebbian' or mode == 'both':
            # Run Hebbian: the eval itself modifies topology
            accs = []; born_total = 0; died_total = 0
            for s in eval_seqs:
                a, born, died = eval_hebbian(net, s)
                accs.append(a); born_total += born; died_total += died
            new_acc = np.mean(accs)
            total_born += born_total; total_died += died_total
        else:
            new_acc = np.mean([eval_standard(net, s) for s in eval_seqs])

        if mode == 'both' or mode == 'guided':
            # Also do guided mutation on top
            net.mutate()
            new_acc2 = np.mean([eval_standard(net, s) for s in eval_seqs])
            new_acc = max(new_acc, new_acc2)

        if new_acc > best_acc:
            best_acc = new_acc; accepts += 1; stale = 0
        else:
            net.restore_state(snap); stale += 1

        if step % log_every == 0:
            print(f"    {label:>10} step {step:5d} | acc={best_acc:.3f} "
                  f"| edges={len(net.alive)} | born={total_born} died={total_died} | acc#={accepts}")

    return best_acc, accepts, total_born, total_died


def main():
    STEPS = 2000; SEED = 42

    tasks = [("Alt", make_alternating), ("Cyc3", make_cycle3)]

    print("=" * 90)
    print("  Hebbian Self-Wiring A/B")
    print(f"  H={H} | Steps={STEPS}")
    print(f"  Hebbian: co-fire→strengthen, miss→weaken, die at 0, born at {EDGE_BIRTH_THRESHOLD} co-fires")
    print("=" * 90)

    results = []
    for task_name, task_fn in tasks:
        master_rng = np.random.RandomState(77)
        eval_seqs = [task_fn(master_rng, 40) for _ in range(3)]

        print(f"\n  === {task_name} ===")

        arms = [
            ("Standard", 'standard'),
            ("Hebbian", 'hebbian'),
            ("Guided", 'guided'),
            ("Hebb+Guide", 'both'),
        ]

        task_results = []
        for label, mode in arms:
            print(f"\n  >>> {label}")
            t0 = time.time()
            acc, accepts, born, died = run_arm(label, mode, task_fn, SEED, STEPS, eval_seqs)
            print(f"    Done in {time.time()-t0:.0f}s: acc={acc:.3f} born={born} died={died}")
            task_results.append((label, acc, accepts, born, died))

        results.append((task_name, task_results))

    # Summary
    print(f"\n{'='*90}")
    print(f"  RESULTS")
    print(f"{'='*90}")
    for task_name, task_results in results:
        print(f"\n  {task_name}:")
        print(f"  {'':>12} | {'Acc':>5} | {'Acc#':>4} | {'Born':>5} | {'Died':>5}")
        best = max(r[1] for r in task_results)
        for label, acc, accepts, born, died in task_results:
            marker = " <<<" if acc == best else ""
            print(f"  {label:>12} | {acc:5.3f} | {accepts:4d} | {born:5d} | {died:5d}{marker}")


if __name__ == "__main__":
    main()
