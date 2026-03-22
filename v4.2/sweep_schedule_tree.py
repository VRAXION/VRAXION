"""
Decision Tree Schedule — hierarchical cos^2 splits
====================================================
GPT co-designed mutation policy. 3 angles -> 4 leaf ops.

Tree:
         [axis1: structure vs param]
         /                          \
    structure                     param
       |                            |
  [axis2: add vs flip]      [axis3: theta vs decay]
     /       \                 /         \
   add      flip            theta       decay

Each angle 0-180: left = cos^2(a/2), right = sin^2(a/2)
Leaf prob = product of path splits.
Global leak: angles decay toward 90 (equal) each step.
Mutation: random angle +-10.

Configs:
A: Fixed tree (angles from best known ratios)
B: Learnable (angles mutate, no leak)
C: Learnable + leak (angles decay toward 90)
D: Learnable + strong leak
"""
import sys, os, time, random, math
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
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


class DecisionTree:
    """3-angle decision tree -> 4 leaf mutation probabilities."""

    def __init__(self, angles, leak_rate=0.0):
        # angles: [structure_vs_param, add_vs_flip, theta_vs_decay]
        # each in degrees 0-180
        self.angles = list(angles)
        self.leak_rate = leak_rate  # decay toward 90 per step (0 = no leak)

    def split(self, angle_deg):
        """cos^2(a/2) for left, sin^2(a/2) for right."""
        rad = math.radians(angle_deg) / 2
        return math.cos(rad)**2, math.sin(rad)**2

    def get_probs(self):
        """Return {add: p, flip: p, theta: p, decay: p} summing to 1."""
        struct_l, struct_r = self.split(self.angles[0])  # structure vs param
        add_l, add_r = self.split(self.angles[1])        # add vs flip
        th_l, th_r = self.split(self.angles[2])           # theta vs decay

        p_add = struct_l * add_l
        p_flip = struct_l * add_r
        p_theta = struct_r * th_l
        p_decay = struct_r * th_r

        return {'add': p_add, 'flip': p_flip, 'theta': p_theta, 'decay': p_decay}

    def pick_type(self, rng):
        """Sample one mutation type from the tree."""
        probs = self.get_probs()
        r = rng.random()
        cumulative = 0.0
        for op in ['add', 'flip', 'theta', 'decay']:
            cumulative += probs[op]
            if r < cumulative:
                return op
        return 'add'  # fallback

    def step_leak(self):
        """Decay angles toward 90 (equal split)."""
        if self.leak_rate > 0:
            for i in range(len(self.angles)):
                diff = 90.0 - self.angles[i]
                self.angles[i] += diff * self.leak_rate
                self.angles[i] = max(0.0, min(180.0, self.angles[i]))

    def mutate(self, rng, amount=10.0):
        """Mutate one random angle by +-amount."""
        idx = rng.randint(0, len(self.angles)-1)
        delta = rng.uniform(-amount, amount)
        self.angles[idx] = max(0.0, min(180.0, self.angles[idx] + delta))

    def summary(self):
        probs = self.get_probs()
        return (f"angles=[{self.angles[0]:.0f},{self.angles[1]:.0f},{self.angles[2]:.0f}] "
                f"-> a={probs['add']*100:.0f}% f={probs['flip']*100:.0f}% "
                f"t={probs['theta']*100:.0f}% d={probs['decay']*100:.0f}%")


def run_config(name, init_angles, leak_rate, learnable,
               bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection,
               max_steps=1500, n_workers=18, threshold=0.00005):
    mask = np.zeros((H, H), dtype=np.float32)
    theta = np.full(H, 0.03, dtype=np.float32)
    decay_arr = np.full(H, 0.15, dtype=np.float32)
    tree = DecisionTree(list(init_angles), leak_rate)
    sched_rng = random.Random(42)

    print(f"\n{'='*70}")
    print(f"  {name} (leak={leak_rate}, learnable={learnable})")
    print(f"  {tree.summary()}")
    print(f"{'='*70}")
    sys.stdout.flush()

    accepts = {'add': 0, 'flip': 0, 'theta': 0, 'decay': 0}
    attempts = {'add': 0, 'flip': 0, 'theta': 0, 'decay': 0}
    acc_history = []
    t0 = time.time()

    pool = Pool(n_workers, initializer=init_w,
                initargs=(bp, ALL_DATA, 200, 2, input_projection, output_projection, bigram))
    try:
        for step in range(1, max_steps+1):
            ptype = tree.pick_type(sched_rng)
            if ptype in ('flip', 'theta', 'decay') and np.count_nonzero(mask) == 0:
                ptype = 'add'
            attempts[ptype] = attempts.get(ptype, 0) + 1

            mask_flat = mask.flatten()
            args = [(mask_flat, theta.copy(), decay_arr.copy(), H,
                     21000+step*50+w, ptype) for w in range(n_workers)]
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

            # Leak
            tree.step_leak()

            # Mutate tree every 20 steps
            if learnable and step % 20 == 0:
                tree.mutate(sched_rng)

            if step % 100 == 0:
                elapsed = time.time() - t0
                edges = int(np.count_nonzero(mask))
                ea = np.mean([eval_accuracy_classic(mask, H, input_projection, output_projection, theta, decay_arr, s, bp)
                              for s in eval_seqs])
                acc_history.append((step, ea))
                quality = ea / max(edges, 1) * 100

                print(f"  [{step:4d}] acc={ea*100:.2f}% edges={edges} q={quality:.3f}%/e "
                      f"A={accepts['add']}|F={accepts['flip']}|T={accepts['theta']}|D={accepts['decay']} "
                      f"{tree.summary()} {elapsed:.0f}s")
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
    print(f"  Attempts: {attempts}")
    print(f"  Tree: {tree.summary()}")
    print(f"  Time: {elapsed:.0f}s")
    sys.stdout.flush()
    return {'name': name, 'acc': ea, 'edges': edges, 'quality': quality, 'time': elapsed}


if __name__ == "__main__":
    IO = 256; NV = 4; H = IO * NV
    SelfWiringGraph.NV_RATIO = NV
    bp = make_bp(IO)

    DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "Diamond Code", "data", "traindat", "fineweb_edu.traindat")
    with open(DATA, 'rb') as f:
        ALL_DATA = np.frombuffer(f.read(), dtype=np.uint8)
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

    # Hand-tuned angles from best known ratios (3a/1f/1t/1d = 50/17/17/17)
    # structure_vs_param: 50% structure -> angle ~60 (cos^2(30) = 0.75... need to solve)
    # Actually: for 3a/1f = 75% add in structure -> angle ~60
    # For 1t/1d = 50/50 param -> angle = 90
    # For structure 83% vs param 17% -> angle ~48

    results = []

    # A: Fixed tree — hand-tuned from 3a/1f/1t/1d (structure-heavy)
    # struct=75% -> angle~60, add=75%->~60, theta=50%->90
    results.append(run_config("FIXED structure-heavy",
                              [60, 60, 90], 0.0, False,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # B: Fixed tree — equal (all 90 degrees)
    results.append(run_config("FIXED equal",
                              [90, 90, 90], 0.0, False,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # C: Learnable, no leak
    results.append(run_config("LEARN no leak",
                              [60, 60, 90], 0.0, True,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # D: Learnable + medium leak (0.02 per step toward 90)
    results.append(run_config("LEARN + medium leak",
                              [60, 60, 90], 0.02, True,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    # E: Learnable + strong leak (0.05 per step)
    results.append(run_config("LEARN + strong leak",
                              [60, 60, 90], 0.05, True,
                              bp, ALL_DATA, bigram, eval_seqs, H, input_projection, output_projection))

    print(f"\n{'='*75}")
    print(f"  SUMMARY -- DECISION TREE SCHEDULE (8t, bigram 2seq, ReLU)")
    print(f"{'='*75}")
    print(f"  {'Name':<25} {'Acc%':>6} {'Edges':>6} {'Q(%/e)':>8} {'Time':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
    for r in results:
        print(f"  {r['name']:<25} {r['acc']*100:6.2f} {r['edges']:6d} {r['quality']:8.3f} {r['time']:5.0f}s")
    sys.stdout.flush()
