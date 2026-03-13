"""
Temperature Dynamics Tracking — Hova konvergál?
================================================
A temperature-modulated mutation MŰKÖDIK. Most azt akarjuk tudni:
MI TÖRTÉNIK BELÜL? A számok mögött milyen dinamika van?

Tracking minden 200 attempt-nél:
  - temperature értéke
  - accept rate (utolsó 200-ból)
  - accuracy (jelenlegi)
  - connection count (NO CAP!)
  - aktív neuronok száma
  - mutáció típus eloszlás
  - zóna eloszlás

64-class, 192 neuron, 16K attempt, combined scoring, NO DENSITY CAP.
"""

import numpy as np
import random
import time
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from v22_best_config import SelfWiringGraph, softmax

# === Config ===
V = 64
N = V + 64 + V  # 192 neurons (shared I/O)
TICKS = 6
MAX_ATT = 16000
LOG_EVERY = 200

# Temperature setup (unleashed)
TEMP_INIT = 1.0
TEMP_COOL = 0.95
TEMP_HEAT = 1.3
TEMP_MIN = 0.1
TEMP_MAX = 10.0
HEAT_INTERVAL = 100  # minden 100 stale-nél melegít


def count_active_neurons(net, V, ticks=6):
    """Hány neuron aktív legalább 1 inputra? (10 random input sample)"""
    active = set()
    for i in range(min(V, 10)):
        net.reset()
        w = np.zeros(V, dtype=np.float32)
        w[i] = 1.0
        net.forward(w, ticks)
        for j in range(net.N):
            if abs(net.charge[j]) > 0.01:
                active.add(j)
    return len(active)


def get_zone(temperature):
    if temperature < 0.3:
        return 'ultra_focus'
    if temperature < 1.0:
        return 'focus'
    if temperature < 2.0:
        return 'normal'
    if temperature < 5.0:
        return 'wide'
    return 'earthquake'


def temp_mutate_unleashed(net, temperature):
    """
    Temperature-modulated mutation. NO DENSITY CAP.

    Temperature controls:
    - rate: how many connections to mutate (scales with temp)
    - type distribution: high temp → more structural, low temp → more flips

    Returns mutation type string.
    """
    # Scale mutation rate with temperature
    base_rate = 0.05
    rate = base_rate * max(0.2, min(temperature, 5.0))

    # Temperature controls mutation type distribution
    if temperature < 0.5:
        # Ultra-focused: mostly flips (fine-tuning)
        weights = {'flip': 0.60, 'add': 0.10, 'remove': 0.10, 'rewire': 0.10, 'block': 0.10}
    elif temperature < 1.5:
        # Normal: balanced
        weights = {'flip': 0.35, 'add': 0.15, 'remove': 0.15, 'rewire': 0.15, 'block': 0.20}
    elif temperature < 3.0:
        # Wide: more structural
        weights = {'flip': 0.20, 'add': 0.25, 'remove': 0.20, 'rewire': 0.20, 'block': 0.15}
    else:
        # Earthquake: heavy restructuring
        weights = {'flip': 0.10, 'add': 0.30, 'remove': 0.25, 'rewire': 0.25, 'block': 0.10}

    # Sample mutation type
    types = list(weights.keys())
    probs = [weights[t] for t in types]
    mut_type = np.random.choice(types, p=probs)

    alive = np.argwhere(net.mask != 0)
    dead = np.argwhere(net.mask == 0)
    dead = dead[dead[:, 0] != dead[:, 1]] if len(dead) > 0 else dead

    if mut_type == 'flip' and len(alive) > 0:
        n = max(1, int(len(alive) * rate * 0.5))
        idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
        for j in range(len(idx)):
            r, c = int(idx[j][0]), int(idx[j][1])
            net.mask[r, c] *= -1

    elif mut_type == 'add' and len(dead) > 0:
        n = max(1, int(len(dead) * rate * 0.3))
        idx = dead[np.random.choice(len(dead), min(n, len(dead)), replace=False)]
        for j in range(len(idx)):
            r, c = int(idx[j][0]), int(idx[j][1])
            net.mask[r, c] = random.choice([-1.0, 1.0])
            net.W[r, c] = random.choice([np.float32(0.5), np.float32(1.5)])

    elif mut_type == 'remove' and len(alive) > 3:
        n = max(1, int(len(alive) * rate * 0.3))
        idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
        for j in range(len(idx)):
            net.mask[int(idx[j][0]), int(idx[j][1])] = 0

    elif mut_type == 'rewire' and len(alive) > 0:
        n = max(1, int(len(alive) * rate * 0.2))
        idx = alive[np.random.choice(len(alive), min(n, len(alive)), replace=False)]
        for j in range(len(idx)):
            r, c = int(idx[j][0]), int(idx[j][1])
            old_sign = net.mask[r, c]
            old_w = net.W[r, c]
            net.mask[r, c] = 0
            nc = random.randint(0, net.N - 1)
            while nc == r:
                nc = random.randint(0, net.N - 1)
            net.mask[r, nc] = old_sign
            net.W[r, nc] = old_w

    elif mut_type == 'block':
        # Block mutation: mutate a contiguous block of connections
        if len(alive) > 0:
            # Pick a random neuron and mutate all its outgoing connections
            neuron = random.randint(0, net.N - 1)
            row = net.mask[neuron]
            conns = np.argwhere(row != 0).flatten()
            if len(conns) > 0:
                n = max(1, int(len(conns) * rate))
                chosen = np.random.choice(conns, min(n, len(conns)), replace=False)
                for c in chosen:
                    if random.random() < 0.5:
                        net.mask[neuron, c] *= -1  # flip sign
                    else:
                        net.W[neuron, c] = np.float32(1.5) if net.W[neuron, c] < 1.0 else np.float32(0.5)

    return mut_type


def score_combined(probs, target):
    """50% accuracy + 50% target_prob."""
    acc = 1.0 if np.argmax(probs) == target else 0.0
    return 0.5 * acc + 0.5 * float(probs[target])


def train_tracked(seed, max_att=MAX_ATT):
    np.random.seed(seed)
    random.seed(seed)

    # Setup: random permutation task
    perm = np.random.permutation(V)
    inputs = list(range(V))

    # Create network — NO DENSITY CAP
    net = SelfWiringGraph(N, V, density=0.06, flip_rate=0.30)

    def evaluate():
        net.reset()
        total_score = 0.0
        correct = 0
        for p in range(2):  # 2 passes: warmup + scoring
            for i in range(len(inputs)):
                world = np.zeros(V, dtype=np.float32)
                world[inputs[i]] = 1.0
                logits = net.forward(world, TICKS)
                probs = softmax(logits)
                if p == 1:
                    s = score_combined(probs, perm[i])
                    total_score += s
                    if np.argmax(probs) == perm[i]:
                        correct += 1
        avg_score = total_score / len(inputs)
        acc = correct / len(inputs)
        net.last_acc = acc
        return avg_score, acc

    temperature = TEMP_INIT
    score, best_acc = evaluate()
    current_acc = best_acc
    stale = 0

    # Tracking
    log = []
    recent_mutations = {'flip': 0, 'add': 0, 'remove': 0, 'rewire': 0, 'block': 0}
    recent_zones = {'ultra_focus': 0, 'focus': 0, 'normal': 0, 'wide': 0, 'earthquake': 0}
    recent_accepted = 0
    recent_total = 0

    t0 = time.time()

    for att in range(max_att):
        # Track zone
        zone = get_zone(temperature)
        recent_zones[zone] += 1
        recent_total += 1

        # Mutate (track type)
        state = net.save_state()
        mut_type = temp_mutate_unleashed(net, temperature)
        recent_mutations[mut_type] += 1

        # Eval
        new_score, new_acc = evaluate()
        current_acc = new_acc

        if new_score > score:
            score = new_score
            best_acc = max(best_acc, new_acc)
            temperature = max(TEMP_MIN, temperature * TEMP_COOL)
            stale = 0
            recent_accepted += 1
        else:
            net.restore_state(state)
            stale += 1
            if stale > 0 and stale % HEAT_INTERVAL == 0:
                temperature = min(TEMP_MAX, temperature * TEMP_HEAT)

        # LOG minden LOG_EVERY-nél
        if (att + 1) % LOG_EVERY == 0:
            conns = net.count_connections()
            active = count_active_neurons(net, V)
            accept_rate = recent_accepted / max(1, recent_total)
            elapsed = time.time() - t0

            entry = {
                'attempt': att + 1,
                'temperature': round(temperature, 4),
                'accuracy': round(best_acc, 4),
                'current_acc': round(current_acc, 4),
                'accept_rate': round(accept_rate, 4),
                'connections': conns,
                'active_neurons': active,
                'total_neurons': net.N,
                'stale': stale,
                'mutations': dict(recent_mutations),
                'zones': dict(recent_zones),
                'dominant_zone': max(recent_zones, key=recent_zones.get),
                'elapsed': round(elapsed, 1),
            }
            log.append(entry)

            # Print compact log
            print(f"  att={att+1:5d} | temp={temperature:6.3f} | acc={best_acc*100:5.1f}% "
                  f"| cur={current_acc*100:5.1f}% | accept={accept_rate*100:4.1f}% "
                  f"| conns={conns:6d} | active={active:3d}/{net.N} "
                  f"| stale={stale:4d} | zone={zone}")

            # Reset trackers
            recent_mutations = {'flip': 0, 'add': 0, 'remove': 0, 'rewire': 0, 'block': 0}
            recent_zones = {'ultra_focus': 0, 'focus': 0, 'normal': 0, 'wide': 0, 'earthquake': 0}
            recent_accepted = 0
            recent_total = 0

        if stale >= 8000:
            print(f"  STALE EXIT at attempt {att+1}")
            break

    elapsed = time.time() - t0
    return best_acc, log, elapsed


def print_summary(seed, acc, log, elapsed):
    """Print detailed summary for one seed."""
    print(f"\n  FINAL: {acc*100:.1f}% in {elapsed:.1f}s")

    # Temperature journey
    print(f"\n  TEMPERATURE JOURNEY:")
    for entry in log:
        t = entry['temperature']
        bar = '█' * min(50, int(t * 5))
        print(f"    att={entry['attempt']:5d} temp={t:6.3f} {bar}")

    # Connection count journey
    print(f"\n  CONNECTION COUNT JOURNEY:")
    for entry in log:
        c = entry['connections']
        bar = '█' * min(50, c // 500)
        print(f"    att={entry['attempt']:5d} conns={c:6d} {bar}")

    # Active neuron journey
    print(f"\n  ACTIVE NEURONS JOURNEY:")
    for entry in log:
        a = entry['active_neurons']
        n = entry['total_neurons']
        bar = '█' * (a * 40 // n)
        print(f"    att={entry['attempt']:5d} active={a:3d}/{n} ({a*100//n:2d}%) {bar}")

    # Accept rate journey
    print(f"\n  ACCEPT RATE JOURNEY:")
    for entry in log:
        ar = entry['accept_rate']
        bar = '█' * int(ar * 40)
        print(f"    att={entry['attempt']:5d} accept={ar*100:5.1f}% {bar}")

    # Zone distribution (full run)
    print(f"\n  ZONE DISTRIBUTION (teljes futás):")
    total_zones = {}
    for entry in log:
        for z, cnt in entry['zones'].items():
            total_zones[z] = total_zones.get(z, 0) + cnt
    total = sum(total_zones.values())
    for z in ['ultra_focus', 'focus', 'normal', 'wide', 'earthquake']:
        pct = total_zones.get(z, 0) / max(1, total) * 100
        bar = '█' * int(pct / 2)
        print(f"    {z:15s} {pct:5.1f}% {bar}")

    # Mutation type distribution (full run)
    print(f"\n  MUTATION TYPE DISTRIBUTION (teljes futás):")
    total_muts = {}
    for entry in log:
        for m, cnt in entry['mutations'].items():
            total_muts[m] = total_muts.get(m, 0) + cnt
    total_m = sum(total_muts.values())
    for m in ['flip', 'add', 'remove', 'rewire', 'block']:
        pct = total_muts.get(m, 0) / max(1, total_m) * 100
        bar = '█' * int(pct / 2)
        print(f"    {m:10s} {pct:5.1f}% {bar}")


# === MAIN ===
if __name__ == "__main__":
    all_results = {}

    print("=" * 74)
    print("  TEMPERATURE DYNAMICS TRACKING")
    print("  64-class | 192 neurons | 16K attempts | combined scoring | NO CAP")
    print("=" * 74)

    for seed in [42, 123, 777]:
        print(f"\n{'='*74}")
        print(f"  SEED {seed}")
        print(f"{'='*74}")

        acc, log, elapsed = train_tracked(seed)
        print_summary(seed, acc, log, elapsed)

        all_results[seed] = {
            'accuracy': acc,
            'elapsed': elapsed,
            'log': log,
        }

    # === CROSS-SEED ANALYSIS ===
    print(f"\n{'='*74}")
    print(f"  CROSS-SEED ANALYSIS")
    print(f"{'='*74}")

    # Final accuracies
    print(f"\n  FINAL ACCURACIES:")
    for seed, res in all_results.items():
        print(f"    seed={seed:4d}: {res['accuracy']*100:.1f}%")
    avg_acc = np.mean([r['accuracy'] for r in all_results.values()])
    print(f"    AVERAGE:  {avg_acc*100:.1f}%")

    # Temperature convergence comparison
    print(f"\n  TEMPERATURE AT KEY POINTS:")
    print(f"    {'':8s} {'att=2000':>10s} {'att=4000':>10s} {'att=8000':>10s} {'att=12000':>10s} {'att=16000':>10s}")
    for seed, res in all_results.items():
        temps = {}
        for entry in res['log']:
            temps[entry['attempt']] = entry['temperature']
        row = f"    seed={seed:4d}"
        for checkpoint in [2000, 4000, 8000, 12000, 16000]:
            t = temps.get(checkpoint, None)
            row += f" {t:10.3f}" if t is not None else f" {'N/A':>10s}"
        print(row)

    # Connection count comparison
    print(f"\n  CONNECTIONS AT KEY POINTS:")
    print(f"    {'':8s} {'att=2000':>10s} {'att=4000':>10s} {'att=8000':>10s} {'att=12000':>10s} {'att=16000':>10s}")
    for seed, res in all_results.items():
        conns = {}
        for entry in res['log']:
            conns[entry['attempt']] = entry['connections']
        row = f"    seed={seed:4d}"
        for checkpoint in [2000, 4000, 8000, 12000, 16000]:
            c = conns.get(checkpoint, None)
            row += f" {c:10d}" if c is not None else f" {'N/A':>10s}"
        print(row)

    # Save full results as JSON
    output_path = os.path.join(os.path.dirname(__file__),
                               f"temp_dynamics_results_{time.strftime('%Y%m%d_%H%M%S')}.json")

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\n  Results saved to: {output_path}")

    print(f"\n{'='*74}")
    print(f"  DONE")
    print(f"{'='*74}")
