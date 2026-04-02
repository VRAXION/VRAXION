"""
Vote-based mutation: network votes for next mutation via charge[0:7]
====================================================================
After each eval forward pass, the charge on neurons 0-7 is already
computed (zero overhead). argmax = which mutation type to apply.

Arms:
  A) RANDOM: burst 7× random op (current winner)
  B) VOTE-LAST: last token's charge[0:7] argmax → 1 mutation per step
  C) VOTE-BURST: collect votes from every Nth token → top 7 votes = 7 mutations
  D) VOTE-ACCUM: accumulate charge[0:7] across all tokens → argmax of sum → 7× that op

Cycle task (H=64, V=10) × 3 seeds, 5000 steps.
"""
import sys, os, numpy as np, random, time
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model'))
from graph import SelfWiringGraph

H = 64; V = 10; TICKS = 8; INPUT_DUR = 2
STEPS = 5000; REPORT = 1000; THRESHOLD = 0.00001
N_SCHED = 8  # neurons 0-7 = mutation vote
CYCLE_MAP = {i: (i + 1) % V for i in range(V)}
OPS = ['add', 'remove', 'rewire', 'theta', 'decay', 'polarity', 'add_loop', 'channel']

def eval_with_votes(net, offset=0):
    """Eval + collect per-token votes from charge[0:8]."""
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    tf = net.theta.astype(np.float32)
    pl = np.where(net.polarity, 1.0, -1.0).astype(np.float32)
    ok = 0; tot = 0; net.reset()
    votes = []
    charge_accum = np.zeros(N_SCHED, dtype=np.float32)

    for i in range(50):
        inp = i % V; target = CYCLE_MAP[inp]
        oh = np.zeros(H, dtype=np.float32); oh[offset + inp] = 3.0
        st, ch = SelfWiringGraph.rollout_token(
            oh, mask=net.mask, theta=tf, decay=net.decay,
            ticks=TICKS, input_duration=INPUT_DUR,
            state=net.state, charge=net.charge,
            sparse_cache=sc, polarity=pl, refractory=net.refractory)
        net.state[:] = st; net.charge[:] = ch
        if int(np.argmax(ch[offset:offset + V])) == target: ok += 1
        tot += 1

        # Collect vote from this token
        votes.append(int(np.argmax(ch[:N_SCHED])))
        charge_accum += ch[:N_SCHED]

    score = ok / tot
    last_vote = votes[-1]
    return score, votes, last_vote, charge_accum

def run_arm(args):
    label, method, seed = args
    np.random.seed(seed); random.seed(seed)
    net = SelfWiringGraph(V, hidden=H, density=10, theta_init=2, seed=seed)
    log = []; accepts = 0; t0 = time.time()
    vote_history = []

    for step in range(1, STEPS + 1):
        saved = net.save_state()
        score, votes, last_vote, charge_accum = eval_with_votes(net, N_SCHED if method != 'random' else 0)
        old_score = score

        if method == 'random':
            # Control: burst 7× random
            for _ in range(7):
                op = random.choice(OPS)
                try: net.mutate(forced_op=op)
                except: pass

        elif method == 'vote_last':
            # Last token's charge votes → 1 mutation
            op = OPS[last_vote % len(OPS)]
            try: net.mutate(forced_op=op)
            except: pass

        elif method == 'vote_burst':
            # Collect 7 votes from evenly spaced tokens
            indices = np.linspace(0, len(votes) - 1, 7, dtype=int)
            for idx in indices:
                op = OPS[votes[idx] % len(OPS)]
                try: net.mutate(forced_op=op)
                except: pass

        elif method == 'vote_accum':
            # Accumulated charge across all tokens → top op, repeat 7×
            top_op = OPS[int(np.argmax(charge_accum)) % len(OPS)]
            for _ in range(7):
                try: net.mutate(forced_op=top_op)
                except: pass

        elif method == 'vote_diverse':
            # Top 7 unique ops by accumulated charge (ranked)
            ranked = np.argsort(charge_accum)[::-1]
            for r in ranked[:7]:
                op = OPS[r % len(OPS)]
                try: net.mutate(forced_op=op)
                except: pass

        new_score = eval_with_votes(net, N_SCHED if method != 'random' else 0)[0]
        if new_score - old_score > THRESHOLD:
            accepts += 1
        else:
            net.restore_state(saved)

        if step % REPORT == 0:
            score = eval_with_votes(net, N_SCHED if method != 'random' else 0)[0]
            edges = net.count_connections()
            tm = float(net.theta.mean())
            sps = step / (time.time() - t0)

            # Track what the network votes for
            _, v, lv, ca = eval_with_votes(net, N_SCHED if method != 'random' else 0)
            from collections import Counter
            vote_dist = Counter(v)
            top_votes = vote_dist.most_common(3)
            vote_str = ' '.join(f'{OPS[k%len(OPS)]}={c}' for k, c in top_votes)

            log.append({'step': step, 'score': round(score, 4), 'edges': edges,
                        'theta': round(tm, 2), 'accepts': accepts})
            print(f'  [{step:5d}] {label}: {score*100:.1f}% e={edges} θ={tm:.2f} '
                  f'acc={accepts} {sps:.0f}sps | votes: {vote_str}', flush=True)

    final = eval_with_votes(net, N_SCHED if method != 'random' else 0)[0]
    return {'label': label, 'method': method, 'seed': seed,
            'final': round(final, 4), 'edges': net.count_connections(),
            'accepts': accepts, 'log': log}

if __name__ == '__main__':
    print('VOTE-BASED MUTATION TEST', flush=True)
    print(f'H={H}, V={V}, neurons 0-7 vote, cycle task', flush=True)
    print(f'5 methods × 3 seeds = 15 arms', flush=True)
    print()

    METHODS = ['random', 'vote_last', 'vote_burst', 'vote_accum', 'vote_diverse']
    SEEDS = [42, 777, 123]

    arms = [(f'{m}_s{s}', m, s) for m in METHODS for s in SEEDS]
    all_results = []
    for batch_start in range(0, len(arms), 3):
        batch = arms[batch_start:batch_start + 3]
        print(f'--- Batch: {[a[0] for a in batch]} ---', flush=True)
        with Pool(3) as pool:
            results = pool.map(run_arm, batch)
        all_results.extend(results)

    sep = '=' * 60
    print(f'\n{sep}', flush=True)
    print('  VOTE MUTATION RESULTS', flush=True)
    print(sep, flush=True)
    for method in METHODS:
        scores = [r['final'] for r in all_results if r['method'] == method]
        accs = [r['accepts'] for r in all_results if r['method'] == method]
        individual = ' '.join(f'{s*100:.0f}' for s in scores)
        print(f'  {method:15s}: {np.mean(scores)*100:.1f}% ± {np.std(scores)*100:.1f}% '
              f'acc={np.mean(accs):.0f}  [{individual}]')

    # Best
    best = max(METHODS, key=lambda m: np.mean([r['final'] for r in all_results if r['method'] == m]))
    print(f'\n  Winner: {best}')
    print(sep, flush=True)
