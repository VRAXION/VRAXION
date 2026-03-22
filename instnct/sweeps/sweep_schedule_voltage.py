"""
Schedule Voltage — rate% + leak + resting potential
=====================================================
Each mutation type has 3 params:
  - prob (0-100): % chance of running this type each step
  - leak (0-10): how fast prob moves toward resting per step
  - resting (0-100): equilibrium value (like neuron resting potential)

Every step: prob += (resting - prob) * leak / 100
Mutation can boost prob by +20.

SANITY CHECK: only add% is learnable, rest fixed.
Then full learnable if sanity passes.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

_bp = None; _all_data = None; _seq_len = 200; _n_train = 2
_input_projection = None; _output_projection = None; _bigram = None

def init_w(b, d, sl, nt, wi, wo, bg):
    global _bp, _all_data, _seq_len, _n_train, _input_projection, _output_projection, _bigram
    _bp, _all_data, _seq_len, _n_train = b, d, sl, nt
    _input_projection, _output_projection, _bigram = wi, wo, bg

def make_bp(io_dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, io_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def _eval_bigram(mask, H, theta, decay, seqs):
    rs, cs = np.where(mask != 0)
    sp_vals = mask[rs, cs]
    pat_norm = _bp / (np.linalg.norm(_bp, axis=1, keepdims=True) + 1e-8)
    ret = 1.0 - decay
    total = 0.0
    for text_bytes in seqs:
        state = np.zeros(H, dtype=np.float32)
        charge = np.zeros(H, dtype=np.float32)
        seq_score = 0.0; n = 0
        for i in range(len(text_bytes)-1):
            act = state.copy()
            for t in range(8):
                if t == 0:
                    act = act + _bp[text_bytes[i]] @ _input_projection
                raw = np.zeros(H, dtype=np.float32)
                if len(rs):
                    np.add.at(raw, cs, act[rs] * sp_vals)
                charge += raw; charge *= ret
                act = np.maximum(charge - theta, 0.0)
                charge = np.maximum(charge, 0.0)
            state = act.copy()
            out = charge @ _output_projection
            out_n = out / (np.linalg.norm(out) + 1e-8)
            sims = out_n @ pat_norm.T
            e = np.exp(sims - sims.max())
            pred = e / e.sum()
            target_dist = _bigram[text_bytes[i]]
            cos = np.dot(pred, target_dist) / (np.linalg.norm(pred) * np.linalg.norm(target_dist) + 1e-8)
            seq_score += cos
            n += 1
        total += seq_score / n if n else 0
    return total / len(seqs)

def worker_eval(args):
    mask_flat, theta, decay, H, seed, proposal_type = args
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    new_mask = mask; new_theta = theta; new_decay = decay

    if proposal_type == 'add':
        r = rng.randint(0, H-1); c = rng.randint(0, H-1)
        if r == c or mask[r, c] != 0:
            return {'delta': -1e9, 'type': 'add'}
        val = 0.6 if rng.random() < 0.5 else -0.6
        new_mask = mask.copy(); new_mask[r, c] = val
    elif proposal_type == 'flip':
        alive = list(zip(*np.where(mask != 0)))
        if not alive:
            return {'delta': -1e9, 'type': 'flip'}
        r, c = alive[rng.randint(0, len(alive)-1)]
        new_mask = mask.copy(); new_mask[r, c] = -mask[r, c]
    elif proposal_type == 'theta':
        idx = rng.randint(0, H-1)
        new_theta = theta.copy()
        new_theta[idx] = max(0.0, min(1.0, theta[idx] + rng.uniform(-0.05, 0.05)))
    elif proposal_type == 'decay':
        idx = rng.randint(0, H-1)
        new_decay = decay.copy()
        new_decay[idx] = max(0.01, min(0.5, decay[idx] + rng.uniform(-0.03, 0.03)))

    seqs = []
    data_len = len(_all_data)
    for _ in range(_n_train):
        off = np_rng.randint(0, data_len - _seq_len)
        seqs.append(_all_data[off:off+_seq_len])

    old_score = _eval_bigram(mask, H, theta, decay, seqs)
    new_score = _eval_bigram(new_mask, H, new_theta, new_decay, seqs)

    return {'delta': new_score - old_score, 'type': proposal_type,
            'new_mask_flat': new_mask.flatten() if new_score > old_score else None,
            'new_theta': new_theta if proposal_type == 'theta' else None,
            'new_decay': new_decay if proposal_type == 'decay' else None}

def eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay, text_bytes, bp):
    pat_norm = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    rs, cs = np.where(mask != 0); sp_vals = mask[rs, cs]
    ret = 1.0 - decay
    state = np.zeros(H, dtype=np.float32); charge = np.zeros(H, dtype=np.float32)
    correct = 0; total = 0
    for i in range(len(text_bytes)-1):
        act = state.copy()
        for t in range(8):
            if t == 0: act = act + bp[text_bytes[i]] @ input_projection
            raw = np.zeros(H, dtype=np.float32)
            if len(rs): np.add.at(raw, cs, act[rs] * sp_vals)
            charge += raw; charge *= ret
            act = np.maximum(charge - theta, 0.0)
            charge = np.maximum(charge, 0.0)
        state = act.copy()
        out = charge @ output_projection
        out_n = out / (np.linalg.norm(out) + 1e-8)
        sims = out_n @ pat_norm.T
        if np.argmax(sims) == text_bytes[i+1]: correct += 1
        total += 1
    return correct/total if total else 0


class ScheduleVoltage:
    """Per-mutation-type voltage system: prob + leak + resting."""

    def __init__(self, init_probs, init_leaks, init_restings):
        # prob: current firing probability (0-100)
        # leak: how fast prob decays toward resting (0-100, higher = faster)
        # resting: equilibrium value (0-100)
        self.prob = dict(init_probs)
        self.leak = dict(init_leaks)
        self.resting = dict(init_restings)

    def step(self):
        """Apply leak: prob moves toward resting."""
        for k in self.prob:
            diff = self.resting[k] - self.prob[k]
            self.prob[k] += diff * self.leak[k] / 100.0
            self.prob[k] = max(0.0, min(100.0, self.prob[k]))

    def pick_type(self, rng):
        """Roll dice for each type, return first that fires. Fallback = add."""
        # Shuffle order each time for fairness
        ops = list(self.prob.keys())
        rng.shuffle(ops)
        for op in ops:
            if rng.random() * 100 < self.prob[op]:
                return op
        return 'add'  # fallback

    def boost(self, op, amount=20):
        """Boost one op's prob."""
        self.prob[op] = min(100.0, self.prob[op] + amount)

    def mutate_meta(self, rng):
        """Mutate one random meta-param (leak or resting) of one random op."""
        ops = list(self.prob.keys())
        op = ops[rng.randint(0, len(ops)-1)]
        if rng.random() < 0.5:
            # Mutate leak
            self.leak[op] = max(0, min(100, self.leak[op] + rng.randint(-5, 5)))
        else:
            # Mutate resting
            self.resting[op] = max(0, min(100, self.resting[op] + rng.randint(-10, 10)))

    def summary(self):
        parts = []
        for k in ['add', 'flip', 'theta', 'decay']:
            if k in self.prob:
                parts.append(f"{k[0]}={self.prob[k]:.0f}%(r={self.resting[k]:.0f},l={self.leak[k]})")
        return ' '.join(parts)


def run_config(name, init_probs, init_leaks, init_restings, learnable_meta,
               bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection,
               max_steps=1500, n_workers=18, threshold=0.00005):
    mask = np.zeros((H, H), dtype=np.float32)
    theta = np.full(H, 0.03, dtype=np.float32)
    decay_arr = np.full(H, 0.15, dtype=np.float32)

    volt = ScheduleVoltage(init_probs, init_leaks, init_restings)
    sched_rng = random.Random(42)

    print(f"\n{'='*70}")
    print(f"  {name} (learnable_meta={learnable_meta})")
    print(f"  {volt.summary()}")
    print(f"{'='*70}")
    sys.stdout.flush()

    accepts = {'add': 0, 'flip': 0, 'theta': 0, 'decay': 0}
    type_attempts = {'add': 0, 'flip': 0, 'theta': 0, 'decay': 0}
    acc_history = []
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, input_projection, output_projection, bigram))
    try:
        for step in range(1, max_steps+1):
            # Pick mutation type via voltage
            ptype = volt.pick_type(sched_rng)
            if ptype in ('flip', 'theta', 'decay') and np.count_nonzero(mask) == 0:
                ptype = 'add'
            type_attempts[ptype] = type_attempts.get(ptype, 0) + 1

            mask_flat = mask.flatten()
            args = [(mask_flat, theta.copy(), decay_arr.copy(), H,
                     20000+step*50+w, ptype) for w in range(n_workers)]
            results = pool.map(worker_eval, args)

            best_r = max(results, key=lambda x: x['delta'])
            if best_r['delta'] > threshold:
                if best_r['type'] in ('add', 'flip') and best_r['new_mask_flat'] is not None:
                    mask = best_r['new_mask_flat'].reshape(H, H)
                    accepts[best_r['type']] += 1
                elif best_r['type'] == 'theta' and best_r['new_theta'] is not None:
                    theta = best_r['new_theta']
                    accepts['theta'] += 1
                elif best_r['type'] == 'decay' and best_r['new_decay'] is not None:
                    decay_arr = best_r['new_decay']
                    accepts['decay'] += 1

            # Voltage leak every step
            volt.step()

            # Boost + meta mutation every 20 steps
            if step % 20 == 0:
                ops = list(volt.prob.keys())
                op = ops[sched_rng.randint(0, len(ops)-1)]
                volt.boost(op, 20)

                if learnable_meta:
                    volt.mutate_meta(sched_rng)

            if step % 100 == 0:
                elapsed = time.time() - t0
                edges = int(np.count_nonzero(mask))
                ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay_arr, s, bp)
                              for s in eval_seqs])
                acc_history.append((step, ea))
                quality = ea / max(edges, 1) * 100

                print(f"  [{step:4d}] acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e "
                      f"A={accepts['add']}|F={accepts['flip']}|T={accepts['theta']}|D={accepts['decay']} "
                      f"{volt.summary()} {elapsed:.0f}s")
                sys.stdout.flush()

                if len(acc_history) >= 4:
                    last4 = [a for _, a in acc_history[-4:]]
                    if max(last4) - min(last4) < 0.01:
                        print(f"  PLATEAU @ step {step}")
                        break

    finally:
        pool.terminate(); pool.join()

    edges = int(np.count_nonzero(mask))
    ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay_arr, s, bp)
                  for s in eval_seqs])
    elapsed = time.time() - t0
    quality = ea / max(edges, 1) * 100

    print(f"\n  FINAL: acc={ea*100:.2f}% edges={edges} quality={quality:.3f}%/edge")
    print(f"  Accepts: A={accepts['add']}|F={accepts['flip']}|T={accepts['theta']}|D={accepts['decay']}")
    print(f"  Attempts: {type_attempts}")
    print(f"  Voltage: {volt.summary()}")
    print(f"  Time: {elapsed:.0f}s")
    sys.stdout.flush()
    return {'name': name, 'acc': ea, 'edges': edges, 'quality': quality,
            'time': elapsed, 'accepts': dict(accepts)}


if __name__ == "__main__":
    IO = 256; NV = 4; H = IO * NV
    SelfWiringGraph.NV_RATIO = NV
    bp = make_bp(IO)

    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    DATA = resolve_fineweb_path()
    ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB text")

    bigram = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "data", "bigram_table.npy"))

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off+200] for off in
                 [eval_rng.randint(0, len(ALL_DATA)-200) for _ in range(10)]]

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(IO)
    input_projection = ref.input_projection / ref.INJ_SCALE * 1.0
    output_projection = ref.output_projection / ref.INJ_SCALE * 1.0

    results = []

    # A: Sanity — fixed probs, no leak (baseline comparison)
    results.append(run_config("FIXED (no voltage)",
                              {'add': 50, 'flip': 20, 'theta': 15, 'decay': 15},
                              {'add': 0, 'flip': 0, 'theta': 0, 'decay': 0},
                              {'add': 50, 'flip': 20, 'theta': 15, 'decay': 15},
                              False,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # B: Voltage with slow leak (leak=5), resting=20 for all
    results.append(run_config("VOLTAGE slow leak",
                              {'add': 60, 'flip': 30, 'theta': 20, 'decay': 20},
                              {'add': 5, 'flip': 5, 'theta': 5, 'decay': 5},
                              {'add': 20, 'flip': 10, 'theta': 5, 'decay': 5},
                              False,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # C: Voltage with medium leak (leak=10)
    results.append(run_config("VOLTAGE medium leak",
                              {'add': 60, 'flip': 30, 'theta': 20, 'decay': 20},
                              {'add': 10, 'flip': 10, 'theta': 10, 'decay': 10},
                              {'add': 20, 'flip': 10, 'theta': 5, 'decay': 5},
                              False,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # D: Full learnable — voltage + meta-params evolve
    results.append(run_config("VOLTAGE learnable",
                              {'add': 60, 'flip': 30, 'theta': 20, 'decay': 20},
                              {'add': 5, 'flip': 5, 'theta': 5, 'decay': 5},
                              {'add': 20, 'flip': 10, 'theta': 5, 'decay': 5},
                              True,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # E: Full learnable, equal start
    results.append(run_config("VOLTAGE learn equal",
                              {'add': 50, 'flip': 50, 'theta': 50, 'decay': 50},
                              {'add': 5, 'flip': 5, 'theta': 5, 'decay': 5},
                              {'add': 25, 'flip': 25, 'theta': 25, 'decay': 25},
                              True,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    print(f"\n{'='*80}")
    print(f"  SUMMARY -- SCHEDULE VOLTAGE (8t, bigram 2seq, ReLU)")
    print(f"{'='*80}")
    print(f"  {'Name':<25} {'Acc%':>6} {'Edges':>6} {'Q(%/e)':>8} {'Time':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
    for r in results:
        print(f"  {r['name']:<25} {r['acc']*100:6.2f} {r['edges']:6d} {r['quality']:8.3f} {r['time']:5.0f}s")
    sys.stdout.flush()
