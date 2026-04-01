"""
Overnight stamina v2 — FIXED stamina persistence
=================================================
Key fix: stamina is NOT reset between evals. It persists across the
training loop, fatiguing and recovering naturally as tokens flow through.

When topology changes (add/remove edge), new edges get full stamina (15)
and removed edges lose their stamina.

Only 2 arms to save time:
  1. BASELINE: no stamina, full mutate (reference)
  2. STAMINA: persistent per-edge stamina, full mutate

5000 steps each, report every 250.
"""
import sys, os, numpy as np, random, time, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model'))
from graph import SelfWiringGraph
from lib.data import load_fineweb_bytes

H = 1024; IO = 256; SEQ_LEN = 150; TICKS = 8; INPUT_DUR = 2
SEED = 42; STEPS = 5000; N_EVAL = 8; REPORT = 250; THRESHOLD = 0.00005

def make_bp(seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, IO).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def build_stamina_for_edges(alive_list, old_alive_set=None, old_stamina_map=None):
    """Build stamina array for current alive edges.
    New edges get stamina=15, existing edges keep their stamina."""
    n = len(alive_list)
    stamina = np.full(n, SelfWiringGraph.DEFAULT_STAMINA, dtype=np.uint8)
    if old_stamina_map is not None:
        for i, (r, c) in enumerate(alive_list):
            if (r, c) in old_stamina_map:
                stamina[i] = old_stamina_map[(r, c)]
    return stamina

def stamina_to_map(alive_list, stamina):
    """Convert alive list + stamina array to dict for persistence."""
    return {(r, c): int(stamina[i]) for i, (r, c) in enumerate(alive_list)}

def eval_cos(net, seqs, bp, bigram, stamina=None):
    pn = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    theta_f = net.theta.astype(np.float32)
    pol = np.where(net.polarity, 1.0, -1.0).astype(np.float32)
    tot = 0.0
    for seq in seqs:
        net.reset()
        ss = 0.0; n = 0
        # Use a COPY of stamina so eval doesn't permanently drain
        stam = stamina.copy() if stamina is not None else None
        for i in range(len(seq) - 1):
            inj = bp[seq[i]] @ net.input_projection
            st, ch = SelfWiringGraph.rollout_token(
                inj, mask=net.mask, theta=theta_f, decay=net.decay,
                ticks=TICKS, input_duration=INPUT_DUR,
                state=net.state, charge=net.charge,
                sparse_cache=sc, polarity=pol,
                refractory=net.refractory, stamina=stam)
            net.state[:] = st; net.charge[:] = ch
            out = ch @ net.output_projection
            on = out / (np.linalg.norm(out) + 1e-8)
            sims = on @ pn.T
            e = np.exp(sims - sims.max()); pred = e / e.sum()
            tgt = bigram[seq[i]]
            cos = np.dot(pred, tgt) / (np.linalg.norm(pred) * np.linalg.norm(tgt) + 1e-8)
            ss += cos; n += 1
        tot += ss / n if n else 0.0
        # After each sequence, update persistent stamina from the fatigued copy
        if stamina is not None and stam is not None:
            stamina[:] = stam
    return tot / len(seqs)

def eval_diversity(net, seqs, bp, stamina=None):
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    theta_f = net.theta.astype(np.float32)
    pol = np.where(net.polarity, 1.0, -1.0).astype(np.float32)
    pn = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    preds = set()
    net.reset()
    seq = seqs[0]
    stam = stamina.copy() if stamina is not None else None
    for i in range(min(50, len(seq) - 1)):
        inj = bp[seq[i]] @ net.input_projection
        st, ch = SelfWiringGraph.rollout_token(
            inj, mask=net.mask, theta=theta_f, decay=net.decay,
            ticks=TICKS, input_duration=INPUT_DUR,
            state=net.state, charge=net.charge,
            sparse_cache=sc, polarity=pol,
            refractory=net.refractory, stamina=stam)
        net.state[:] = st; net.charge[:] = ch
        out = ch @ net.output_projection
        on = out / (np.linalg.norm(out) + 1e-8)
        preds.add(int(np.argmax(on @ pn.T)))
    return len(preds)

def run_arm(label, use_stamina, all_data, bigram, bp, eval_seqs):
    print(f'\n{"="*60}')
    print(f'  {label}')
    print(f'{"="*60}')
    sys.stdout.flush()

    np.random.seed(SEED); random.seed(SEED)
    net = SelfWiringGraph(IO, hidden_ratio=4, projection_scale=1.0, seed=SEED,
                          theta_init=1, density=4)

    np_rng = np.random.RandomState(SEED)
    log = []
    t0 = time.time()

    # Persistent stamina map (survives topology changes)
    stam_map = None
    if use_stamina:
        stam_map = stamina_to_map(net.alive,
                                   np.full(len(net.alive), SelfWiringGraph.DEFAULT_STAMINA, dtype=np.uint8))

    for step in range(1, STEPS + 1):
        saved = net.save_state()
        old_alive = list(net.alive)

        # Build stamina array for current edges
        stam = None
        if use_stamina:
            stam = build_stamina_for_edges(net.alive, old_stamina_map=stam_map)

        tr = [all_data[o:o + SEQ_LEN]
              for o in [np_rng.randint(0, len(all_data) - SEQ_LEN) for _ in range(2)]]

        # Eval with PERSISTENT stamina (copy for old score so drain doesn't bias)
        stam_old_copy = stam.copy() if stam is not None else None
        old_score = eval_cos(net, tr, bp, bigram, stam_old_copy)

        # Mutate
        undo = net.mutate()

        # Build stamina for potentially changed edges
        stam_new = None
        if use_stamina:
            stam_new = build_stamina_for_edges(net.alive, old_stamina_map=stam_map)

        new_score = eval_cos(net, tr, bp, bigram, stam_new)

        if new_score - old_score > THRESHOLD:
            # Accept: update persistent stamina map from the fatigued new stamina
            if use_stamina and stam_new is not None:
                stam_map = stamina_to_map(net.alive, stam_new)
        else:
            # Reject: restore state, keep old stamina
            net.restore_state(saved)
            if use_stamina and stam_old_copy is not None:
                stam_map = stamina_to_map(net.alive, stam_old_copy)

        if step % REPORT == 0:
            stam_eval = build_stamina_for_edges(net.alive, old_stamina_map=stam_map) if use_stamina else None
            cos = eval_cos(net, eval_seqs, bp, bigram, stam_eval)
            div = eval_diversity(net, eval_seqs, bp, stam_eval)
            edges = net.count_connections()
            theta_m = float(net.theta.mean())
            drive = int(net.mutation_drive)
            elapsed = time.time() - t0
            sps = step / elapsed

            es = set(zip(*[x.tolist() for x in np.where(net.mask)]))
            n2 = sum(1 for r, c in es if (c, r) in es) // 2

            # Stamina stats
            if use_stamina and stam_eval is not None:
                s_mean = float(stam_eval.mean())
                s_min = int(stam_eval.min())
                s_tired = int(np.sum(stam_eval < 5))
                stam_info = f' s={s_mean:.1f}[{s_min}] tired={s_tired}'
            else:
                stam_info = ''

            entry = {'step': step, 'cos': round(cos, 6), 'edges': edges,
                     'theta': round(theta_m, 2), 'drive': drive, '2cyc': n2,
                     'diversity': div, 'sps': round(sps, 2)}
            log.append(entry)
            print(f'  [{step:5d}] cos={cos:.6f} e={edges} θ={theta_m:.2f} '
                  f'd={drive} 2c={n2} div={div} {sps:.1f}sps{stam_info}')
            sys.stdout.flush()

    ckpt = ROOT / 'recipes' / 'checkpoints' / f'overnight_v2_{label}.npz'
    net.save(str(ckpt))
    print(f'  Saved: {ckpt}')
    return {'label': label, 'log': log}


if __name__ == '__main__':
    print('Loading data...')
    all_data = load_fineweb_bytes()
    bigram_path = ROOT / 'recipes' / 'data' / 'bigram_table.npy'
    bigram = np.load(bigram_path)
    bp = make_bp()

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [all_data[o:o + SEQ_LEN]
                 for o in [eval_rng.randint(0, len(all_data) - SEQ_LEN) for _ in range(N_EVAL)]]

    results = []
    for label, use_stam in [('baseline', False), ('stamina_persistent', True)]:
        r = run_arm(label, use_stam, all_data, bigram, bp, eval_seqs)
        results.append(r)

    print(f'\n{"="*60}')
    print(f'  SUMMARY')
    print(f'{"="*60}')
    for r in results:
        if r['log']:
            f = r['log'][-1]
            print(f'  {r["label"]:25s}: cos={f["cos"]:.6f} e={f["edges"]} div={f["diversity"]}')

    out_path = ROOT / 'recipes' / 'overnight_stamina_v2_results.json'
    with open(out_path, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'\nResults: {out_path}')
