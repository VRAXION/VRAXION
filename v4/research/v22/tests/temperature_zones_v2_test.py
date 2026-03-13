"""
Temperature Zone Fix — Minden zónában NŐHET a hálózat
=====================================================
Diagnózis: a régi temperature zónák flip-only módba zárták a hálózatot
alacsony density-nél (35% vs baseline 98.6%). Az új v2: structure mutáció
MINDIG elérhető, temperature csak az arányt szabályozza.

5 módok tesztelése:
  1. baseline         — fix mutáció, nincs temperature, NO CAP
  2. old_temp_gradual — régi zónák (ultra_focus = mostly flip)
  3. new_temp_v2      — javított: minden zónában nőhet, cool 0.95, heat 100×1.3
  4. new_temp_balanced — javított + cool 0.98, heat 20×1.1
  5. new_temp_adaptive — cool/heat RATE a density-ből és accept rate-ből

64-class, 192 neuron, 8K attempt, 5 seeds, NO DENSITY CAP.
Combined scoring (0.5×acc + 0.5×target_prob).
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
N = V + 64 + V  # 192 neurons
TICKS = 6
MAX_ATT = 8000
LOG_EVERY = 1000
SEEDS = [42, 123, 777, 314, 999]

MAX_POSSIBLE_CONNS = N * (N - 1)  # 192*191 = 36672


def score_combined(probs, target):
    """50% accuracy + 50% target_prob."""
    acc = 1.0 if np.argmax(probs) == target else 0.0
    return 0.5 * acc + 0.5 * float(probs[target])


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


def get_density(net):
    conns = int((net.mask != 0).sum())
    return conns / MAX_POSSIBLE_CONNS


# ==============================================================================
# MUTATION FUNCTIONS
# ==============================================================================

def mutate_baseline(net, **kwargs):
    """Baseline: structure + weight mutation, no temperature. Returns mut_type."""
    if random.random() < 0.3:
        net.mutate_structure(0.02)
        net.mutate_weights()
        return 'both'
    else:
        net.mutate_structure(0.05)
        return 'structure'


def mutate_old_temp(net, temperature=1.0, **kwargs):
    """Old temperature zones — ultra_focus = mostly flip, locks density."""
    base_rate = 0.05
    rate = base_rate * max(0.2, min(temperature, 5.0))

    if temperature < 0.5:
        weights = {'flip': 0.60, 'add': 0.10, 'remove': 0.10, 'rewire': 0.10, 'block': 0.10}
    elif temperature < 1.5:
        weights = {'flip': 0.35, 'add': 0.15, 'remove': 0.15, 'rewire': 0.15, 'block': 0.20}
    elif temperature < 3.0:
        weights = {'flip': 0.20, 'add': 0.25, 'remove': 0.20, 'rewire': 0.20, 'block': 0.15}
    else:
        weights = {'flip': 0.10, 'add': 0.30, 'remove': 0.25, 'rewire': 0.25, 'block': 0.10}

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

    elif mut_type == 'block' and len(alive) > 0:
        neuron = random.randint(0, net.N - 1)
        row = net.mask[neuron]
        conns = np.argwhere(row != 0).flatten()
        if len(conns) > 0:
            n = max(1, int(len(conns) * rate))
            chosen = np.random.choice(conns, min(n, len(conns)), replace=False)
            for c in chosen:
                if random.random() < 0.5:
                    net.mask[neuron, c] *= -1
                else:
                    net.W[neuron, c] = np.float32(1.5) if net.W[neuron, c] < 1.0 else np.float32(0.5)

    return mut_type


def mutate_new_v2(net, temperature=1.0, **kwargs):
    """NEW v2: minden zónában nőhet a hálózat! Csak az arányok változnak."""
    # Flip arány: hideg = több flip, meleg = kevesebb
    flip_prob = max(0.2, 0.6 - temperature * 0.1)

    # Blokk méret: hideg = 1, meleg = több
    block_size = max(1, int(temperature))

    for _ in range(block_size):
        if random.random() < flip_prob:
            # FLIP (fázis váltás) — single connection
            alive = np.argwhere(net.mask != 0)
            if len(alive) > 0:
                idx = alive[np.random.choice(len(alive))]
                net.mask[int(idx[0]), int(idx[1])] *= -1
        else:
            # STRUCTURE (add/remove/rewire) — MINDIG ELÉRHETŐ!
            net.mutate_structure(0.05)

    # Weight mutáció: mindig 30% eséllyel
    if random.random() < 0.3:
        net.mutate_weights()

    return 'v2_mixed'


def mutate_new_balanced(net, temperature=1.0, **kwargs):
    """NEW balanced: same as v2 but with gentler cool/heat params."""
    # Same mutation logic as v2
    flip_prob = max(0.2, 0.6 - temperature * 0.1)
    block_size = max(1, int(temperature))

    for _ in range(block_size):
        if random.random() < flip_prob:
            alive = np.argwhere(net.mask != 0)
            if len(alive) > 0:
                idx = alive[np.random.choice(len(alive))]
                net.mask[int(idx[0]), int(idx[1])] *= -1
        else:
            net.mutate_structure(0.05)

    if random.random() < 0.3:
        net.mutate_weights()

    return 'balanced_mixed'


def mutate_new_adaptive(net, temperature=1.0, **kwargs):
    """NEW adaptive: same v2 mutation, temperature dynamics from density+accept."""
    flip_prob = max(0.2, 0.6 - temperature * 0.1)
    block_size = max(1, int(temperature))

    for _ in range(block_size):
        if random.random() < flip_prob:
            alive = np.argwhere(net.mask != 0)
            if len(alive) > 0:
                idx = alive[np.random.choice(len(alive))]
                net.mask[int(idx[0]), int(idx[1])] *= -1
        else:
            net.mutate_structure(0.05)

    if random.random() < 0.3:
        net.mutate_weights()

    return 'adaptive_mixed'


# ==============================================================================
# MODE CONFIGURATIONS
# ==============================================================================

MODES = {
    'baseline': {
        'mutate_fn': mutate_baseline,
        'use_temp': False,
        'desc': 'No temperature, standard mutation',
    },
    'old_temp_gradual': {
        'mutate_fn': mutate_old_temp,
        'use_temp': True,
        'temp_init': 1.0,
        'temp_cool': 0.95,
        'temp_heat': 1.3,
        'temp_min': 0.1,
        'temp_max': 10.0,
        'heat_interval': 100,
        'desc': 'Old zones (ultra_focus = mostly flip)',
    },
    'new_temp_v2': {
        'mutate_fn': mutate_new_v2,
        'use_temp': True,
        'temp_init': 1.0,
        'temp_cool': 0.95,
        'temp_heat': 1.3,
        'temp_min': 0.1,
        'temp_max': 10.0,
        'heat_interval': 100,
        'desc': 'NEW v2: structure always available, cool 0.95, heat 100×1.3',
    },
    'new_temp_balanced': {
        'mutate_fn': mutate_new_balanced,
        'use_temp': True,
        'temp_init': 1.0,
        'temp_cool': 0.98,
        'temp_heat': 1.1,
        'temp_min': 0.1,
        'temp_max': 10.0,
        'heat_interval': 20,
        'desc': 'NEW balanced: structure always available, cool 0.98, heat 20×1.1',
    },
    'new_temp_adaptive': {
        'mutate_fn': mutate_new_adaptive,
        'use_temp': True,
        'temp_init': 1.0,
        'temp_cool': None,  # adaptive — set per-step
        'temp_heat': None,  # adaptive — set per-step
        'temp_min': 0.1,
        'temp_max': 10.0,
        'heat_interval': 50,
        'desc': 'NEW adaptive: cool/heat from density + accept rate',
    },
}


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train_mode(mode_name, seed, max_att=MAX_ATT):
    cfg = MODES[mode_name]
    np.random.seed(seed)
    random.seed(seed)

    perm = np.random.permutation(V)
    inputs = list(range(V))
    net = SelfWiringGraph(N, V, density=0.06, flip_rate=0.30)

    def evaluate():
        net.reset()
        total_score = 0.0
        correct = 0
        for p in range(2):
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

    # Init
    temperature = cfg.get('temp_init', 1.0) if cfg['use_temp'] else 1.0
    score, best_acc = evaluate()
    stale = 0
    log = []
    recent_accepted = 0
    recent_total = 0
    t0 = time.time()

    for att in range(max_att):
        recent_total += 1
        state = net.save_state()

        # Mutate
        cfg['mutate_fn'](net, temperature=temperature)

        # Eval
        new_score, new_acc = evaluate()

        if new_score > score:
            score = new_score
            best_acc = max(best_acc, new_acc)
            stale = 0
            recent_accepted += 1

            if cfg['use_temp']:
                # Cool
                if mode_name == 'new_temp_adaptive':
                    density = get_density(net)
                    if density < 0.5:
                        cool = 0.99
                    elif density < 0.8:
                        cool = 0.97
                    else:
                        cool = 0.95
                else:
                    cool = cfg['temp_cool']
                temperature = max(cfg['temp_min'], temperature * cool)
        else:
            net.restore_state(state)
            stale += 1

            if cfg['use_temp']:
                heat_interval = cfg['heat_interval']
                if stale > 0 and stale % heat_interval == 0:
                    if mode_name == 'new_temp_adaptive':
                        accept_rate = recent_accepted / max(1, recent_total)
                        if accept_rate < 0.01:
                            heat = 1.05
                        elif accept_rate < 0.03:
                            heat = 1.02
                        else:
                            heat = 1.01
                    else:
                        heat = cfg['temp_heat']
                    temperature = min(cfg['temp_max'], temperature * heat)

        # LOG
        if (att + 1) % LOG_EVERY == 0:
            conns = net.count_connections()
            density = conns / MAX_POSSIBLE_CONNS
            accept_rate = recent_accepted / max(1, recent_total)
            zone = get_zone(temperature) if cfg['use_temp'] else 'N/A'

            entry = {
                'attempt': att + 1,
                'temperature': round(temperature, 4),
                'accuracy': round(best_acc, 4),
                'current_acc': round(new_acc, 4),
                'accept_rate': round(accept_rate, 4),
                'connections': conns,
                'density_pct': round(density * 100, 1),
                'stale': stale,
                'zone': zone,
            }
            log.append(entry)

            print(f"    att={att+1:5d} | temp={temperature:5.2f} | acc={best_acc*100:5.1f}% "
                  f"| accept={accept_rate*100:4.1f}% | conns={conns:6d} ({density*100:4.1f}%) "
                  f"| zone={zone}")

            recent_accepted = 0
            recent_total = 0

        if stale >= 6000:
            print(f"    STALE EXIT at attempt {att+1}")
            break

    elapsed = time.time() - t0
    return best_acc, log, elapsed


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("=" * 78)
    print("  TEMPERATURE ZONE FIX v2 — Growth in ALL zones")
    print("  64-class | 192 neurons | 8K attempts | 5 seeds | NO CAP")
    print("=" * 78)

    all_results = {}

    for mode_name in MODES:
        cfg = MODES[mode_name]
        print(f"\n{'='*78}")
        print(f"  MODE: {mode_name}")
        print(f"  {cfg['desc']}")
        print(f"{'='*78}")

        mode_accs = []
        mode_conns = []
        mode_logs = []

        for seed in SEEDS:
            print(f"\n  --- seed={seed} ---")
            acc, log, elapsed = train_mode(mode_name, seed)
            mode_accs.append(acc)
            final_conns = log[-1]['connections'] if log else 0
            final_density = log[-1]['density_pct'] if log else 0
            mode_conns.append(final_conns)
            mode_logs.append({'seed': seed, 'acc': acc, 'elapsed': elapsed, 'log': log})
            print(f"    FINAL: {acc*100:.1f}% | {final_conns} conns ({final_density:.1f}%) | {elapsed:.1f}s")

        avg_acc = np.mean(mode_accs)
        avg_conns = np.mean(mode_conns)
        print(f"\n  MODE SUMMARY: {mode_name}")
        print(f"    avg_acc   = {avg_acc*100:.1f}%")
        print(f"    avg_conns = {avg_conns:.0f} ({avg_conns/MAX_POSSIBLE_CONNS*100:.1f}%)")
        print(f"    accs      = {[f'{a*100:.1f}%' for a in mode_accs]}")

        all_results[mode_name] = {
            'avg_acc': float(avg_acc),
            'avg_conns': float(avg_conns),
            'avg_density': float(avg_conns / MAX_POSSIBLE_CONNS * 100),
            'accs': [float(a) for a in mode_accs],
            'conns': [int(c) for c in mode_conns],
            'logs': mode_logs,
        }

    # ===========================================================================
    # COMPARISON TABLE
    # ===========================================================================
    print(f"\n\n{'='*78}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*78}")
    print(f"\n  {'Mode':<22s} {'Avg Acc':>8s} {'Avg Conns':>10s} {'Density':>8s} {'Best Seed':>10s}")
    print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*8} {'-'*10}")

    for mode_name, res in all_results.items():
        best_seed_acc = max(res['accs'])
        print(f"  {mode_name:<22s} {res['avg_acc']*100:7.1f}% {res['avg_conns']:10.0f} "
              f"{res['avg_density']:7.1f}% {best_seed_acc*100:9.1f}%")

    # Connection growth trajectories
    print(f"\n  CONNECTION GROWTH (at key checkpoints):")
    checkpoints = [1000, 2000, 4000, 6000, 8000]
    print(f"  {'Mode':<22s}", end="")
    for cp in checkpoints:
        print(f" {'att='+str(cp):>10s}", end="")
    print()
    print(f"  {'-'*22}", end="")
    for _ in checkpoints:
        print(f" {'-'*10}", end="")
    print()

    for mode_name, res in all_results.items():
        print(f"  {mode_name:<22s}", end="")
        # Average connections across seeds at each checkpoint
        for cp in checkpoints:
            cp_conns = []
            for seed_data in res['logs']:
                for entry in seed_data['log']:
                    if entry['attempt'] == cp:
                        cp_conns.append(entry['connections'])
            if cp_conns:
                avg_c = np.mean(cp_conns)
                print(f" {avg_c:10.0f}", end="")
            else:
                print(f" {'N/A':>10s}", end="")
        print()

    # Temperature trajectories (for temp modes only)
    print(f"\n  TEMPERATURE TRAJECTORIES (avg across seeds):")
    print(f"  {'Mode':<22s}", end="")
    for cp in checkpoints:
        print(f" {'att='+str(cp):>10s}", end="")
    print()
    print(f"  {'-'*22}", end="")
    for _ in checkpoints:
        print(f" {'-'*10}", end="")
    print()

    for mode_name, res in all_results.items():
        if not MODES[mode_name]['use_temp']:
            continue
        print(f"  {mode_name:<22s}", end="")
        for cp in checkpoints:
            cp_temps = []
            for seed_data in res['logs']:
                for entry in seed_data['log']:
                    if entry['attempt'] == cp:
                        cp_temps.append(entry['temperature'])
            if cp_temps:
                avg_t = np.mean(cp_temps)
                print(f" {avg_t:10.3f}", end="")
            else:
                print(f" {'N/A':>10s}", end="")
        print()

    # KEY QUESTIONS
    print(f"\n  KEY ANSWERS:")
    bl = all_results.get('baseline', {})
    v2 = all_results.get('new_temp_v2', {})
    bal = all_results.get('new_temp_balanced', {})
    adp = all_results.get('new_temp_adaptive', {})
    old = all_results.get('old_temp_gradual', {})

    bl_conns = bl.get('avg_conns', 0)
    v2_conns = v2.get('avg_conns', 0)
    print(f"  1. v2 reaches baseline conns? baseline={bl_conns:.0f} vs v2={v2_conns:.0f} "
          f"({'YES' if v2_conns > bl_conns * 0.8 else 'NO'})")

    bl_acc = bl.get('avg_acc', 0)
    v2_acc = v2.get('avg_acc', 0)
    print(f"  2. v2 better than baseline? baseline={bl_acc*100:.1f}% vs v2={v2_acc*100:.1f}% "
          f"({'YES' if v2_acc > bl_acc else 'NO'})")

    adp_conns = adp.get('avg_conns', 0)
    print(f"  3. Adaptive controls density? density={adp.get('avg_density', 0):.1f}%")

    best_mode = max(all_results.items(), key=lambda x: x[1]['avg_acc'])
    print(f"  4. Best mode: {best_mode[0]} ({best_mode[1]['avg_acc']*100:.1f}% avg)")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__),
                               f"temp_zones_v2_results_{time.strftime('%Y%m%d_%H%M%S')}.json")

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Save summary (without full logs to keep file small)
    summary = {}
    for mode_name, res in all_results.items():
        summary[mode_name] = {
            'avg_acc': res['avg_acc'],
            'avg_conns': res['avg_conns'],
            'avg_density': res['avg_density'],
            'accs': res['accs'],
            'conns': res['conns'],
        }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=convert)
    print(f"\n  Results saved to: {output_path}")

    print(f"\n{'='*78}")
    print(f"  DONE")
    print(f"{'='*78}")
