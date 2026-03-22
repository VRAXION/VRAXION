"""
Overnight Sweep — runs until manually killed (Ctrl+C)
======================================================
Continuously tests SWG growth variants, logs everything,
maintains a live leaderboard.

Variants swept:
  - V: 16, 32, 64, 128
  - policy: add_only, add+rewire(30%), add+rewire(50%)
  - crystal: none, end_only (crystal after growth exhausted)
  - Seeds: rotating

Live output:
  overnight_sweep_live.txt    — leaderboard (tail -f this)
  overnight_sweep_log.jsonl   — full results
"""

import sys, os, time, json, random, signal
import numpy as np
from collections import defaultdict

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

# ─── Globals ────────────────────────────────────────────

BASE = os.path.dirname(os.path.abspath(__file__))
LOG_JSONL = os.path.join(BASE, "overnight_sweep_log.jsonl")
LOG_LIVE = os.path.join(BASE, "overnight_sweep_live.txt")
RUNNING = True


def handle_sigint(sig, frame):
    global RUNNING
    print("\n  [SIGINT] Finishing current trial, then stopping...")
    RUNNING = False

signal.signal(signal.SIGINT, handle_sigint)


# ─── Core ───────────────────────────────────────────────

def evaluate(net, targets, ticks=6):
    logits = net.forward_batch(ticks)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    V = min(net.V, len(targets))
    acc = (np.argmax(probs, axis=1)[:V] == targets[:V]).mean()
    tp = probs[np.arange(V), targets[:V]].mean()
    return float(0.5 * acc + 0.5 * tp)


def make_empty_net(V, seed):
    random.seed(seed)
    np.random.seed(seed)
    net = SelfWiringGraph(V)
    net.mask[:] = 0.0
    net.alive = []
    net.alive_set = set()
    net.state *= 0
    net.charge *= 0
    net.loss_pct = np.int8(15)
    net.mutation_drive = np.int8(0)
    return net


def pick_op(policy):
    if policy == 'add_only':
        return 'add'
    elif policy == 'add_rw30':
        return 'add' if random.random() < 0.7 else 'rewire'
    elif policy == 'add_rw50':
        return 'add' if random.random() < 0.5 else 'rewire'
    return 'add'


def run_crystal_phase(net, targets, ticks, patience):
    """Remove-only until patience consecutive rejects. Returns edges removed."""
    removed = 0
    stale = 0
    score = evaluate(net, targets, ticks)
    attempts = 0
    while stale < patience and net.alive and attempts < patience * 3:
        undo = net.mutate(forced_op='remove')
        new_score = evaluate(net, targets, ticks)
        if new_score >= score:
            score = new_score
            removed += 1
            stale = 0
        else:
            net.replay(undo)
            stale += 1
        attempts += 1
    return removed, score


def run_trial(V, seed, budget, policy, crystal_mode, ticks=6):
    net = make_empty_net(V, seed)
    rng_t = np.random.RandomState(seed + 1000)
    targets = np.arange(V)
    rng_t.shuffle(targets)

    score = evaluate(net, targets, ticks)
    best = score
    stale = 0
    accepts = 0
    crystal_removed = 0
    t0 = time.time()

    # Growth phase
    for att in range(1, budget + 1):
        op = pick_op(policy)
        if not net.alive and op == 'rewire':
            op = 'add'

        undo = net.mutate(forced_op=op)
        new_score = evaluate(net, targets, ticks)

        if new_score > score:
            score = new_score
            best = max(best, score)
            stale = 0
            accepts += 1
        else:
            net.replay(undo)
            stale += 1

        if best >= 0.999:
            break

    edges_before_crystal = net.count_connections()

    # Crystal phase (end_only)
    if crystal_mode == 'end_only':
        cr, score = run_crystal_phase(net, targets, ticks,
                                       patience=min(500, V * 10))
        crystal_removed = cr
        best = max(best, score)

    elapsed = time.time() - t0
    return {
        "V": V, "seed": seed, "budget": budget,
        "policy": policy, "crystal": crystal_mode,
        "best": round(best, 5),
        "edges_pre_crystal": edges_before_crystal,
        "edges_final": net.count_connections(),
        "crystal_removed": crystal_removed,
        "accepts": accepts,
        "elapsed": round(elapsed, 1),
    }


# ─── Sweep configs ──────────────────────────────────────

def get_configs():
    """All variant combinations."""
    configs = []
    for V in [16, 32, 64, 128]:
        budget = {16: 8000, 32: 16000, 64: 32000, 128: 64000}[V]
        for policy in ['add_only', 'add_rw30', 'add_rw50']:
            for crystal in ['none', 'end_only']:
                configs.append({
                    'V': V, 'budget': budget,
                    'policy': policy, 'crystal': crystal,
                })
    return configs


def config_key(cfg):
    return f"V={cfg['V']:3d} {cfg['policy']:10s} crystal={cfg['crystal']:8s}"


# ─── Leaderboard ────────────────────────────────────────

def update_leaderboard(all_results):
    """Write live leaderboard sorted by avg score per config."""
    grouped = defaultdict(list)
    for r in all_results:
        k = config_key(r)
        grouped[k].append(r['best'])

    board = []
    for k, scores in grouped.items():
        avg = np.mean(scores)
        std = np.std(scores) if len(scores) > 1 else 0
        board.append((avg, std, len(scores), k, max(scores), min(scores)))

    board.sort(reverse=True)

    lines = []
    lines.append(f"Overnight Sweep Leaderboard | {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total trials: {len(all_results)}")
    lines.append(f"{'='*80}")
    lines.append(f"  {'Rank':4s} {'Avg':>7s} {'Std':>6s} {'N':>3s} {'Max':>7s} {'Min':>7s}  Config")
    lines.append(f"  {'-'*4} {'-'*7} {'-'*6} {'-'*3} {'-'*7} {'-'*7}  {'-'*35}")

    for i, (avg, std, n, k, mx, mn) in enumerate(board):
        lines.append(f"  {i+1:4d} {avg:7.4f} {std:6.4f} {n:3d} {mx:7.4f} {mn:7.4f}  {k}")

    lines.append(f"{'='*80}")

    # Per-V winners
    lines.append("")
    for V in [16, 32, 64, 128]:
        v_board = [(a, k) for a, s, n, k, mx, mn in board if f"V={V:3d}" in k]
        if v_board:
            best_avg, best_k = v_board[0]
            lines.append(f"  V={V:3d} BEST: {best_avg:.4f}  {best_k}")

    text = "\n".join(lines) + "\n"
    with open(LOG_LIVE, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ─── Main loop ──────────────────────────────────────────

def main():
    configs = get_configs()
    all_results = []
    seed_counter = 42
    trial_num = 0

    # Load existing results if resuming
    if os.path.exists(LOG_JSONL):
        with open(LOG_JSONL, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_results.append(json.loads(line))
                    except:
                        pass
        if all_results:
            seed_counter = max(r.get('seed', 42) for r in all_results) + 1
            trial_num = len(all_results)
            print(f"  Resumed: {len(all_results)} existing results, seed starts at {seed_counter}")

    print(f"  Overnight Sweep | {len(configs)} configs | Ctrl+C to stop")
    print(f"  Log: {LOG_JSONL}")
    print(f"  Live: {LOG_LIVE}")
    print()

    while RUNNING:
        # Rotate through configs, each with a fresh seed
        cfg = configs[trial_num % len(configs)]
        seed = seed_counter
        seed_counter += 1
        trial_num += 1

        label = config_key(cfg)
        print(f"  [{trial_num:4d}] {label} seed={seed}...", end=" ", flush=True)

        try:
            result = run_trial(
                V=cfg['V'], seed=seed, budget=cfg['budget'],
                policy=cfg['policy'], crystal_mode=cfg['crystal'],
            )
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        all_results.append(result)

        # Append to log
        with open(LOG_JSONL, "a") as f:
            f.write(json.dumps(result) + "\n")

        # Update leaderboard
        board_text = update_leaderboard(all_results)

        print(f"best={result['best']:.4f} edges={result['edges_final']} "
              f"crystal={result['crystal_removed']} {result['elapsed']:.0f}s")

        # Print leaderboard every 24 trials (one full round)
        if trial_num % len(configs) == 0:
            print(f"\n{board_text}")

        if not RUNNING:
            break

    print(f"\n  STOPPED after {trial_num} trials.")
    print(update_leaderboard(all_results))


if __name__ == "__main__":
    main()

