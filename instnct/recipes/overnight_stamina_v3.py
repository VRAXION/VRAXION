"""
Overnight stamina v3 — correct stamina model
=============================================
Key insight from v1/v2 failures:
  v1: stamina resets per eval → no learning signal (all evals identical)
  v2: stamina persists across evals → all edges drain to 0, network dies

Correct model: stamina RESETS to full at start of each sequence,
but FATIGUES within a sequence (across tokens). This means:
  - Within a 150-token sequence: loops fatigue, creating temporal signal
  - Between training steps: fresh start, mutations can be fairly compared
  - The TOPOLOGY determines which edges fatigue fastest (high-use edges)

This is biologically correct: STP operates on ~100ms-10s timescale,
which maps to "within a sequence" not "across training epochs."

2 arms: baseline vs stamina. 5000 steps each.
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

def eval_cos(net, seqs, bp, bigram, use_stamina=False):
    pn = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    theta_f = net.theta.astype(np.float32)
    pol = np.where(net.polarity, 1.0, -1.0).astype(np.float32)
    n_edges = len(sc[0])
    tot = 0.0
    for seq in seqs:
        net.reset()
        # FRESH stamina per sequence — this is the key fix
        stam = np.full(n_edges, SelfWiringGraph.DEFAULT_STAMINA, dtype=np.uint8) if use_stamina else None
        ss = 0.0; n = 0
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
    return tot / len(seqs)

def eval_diversity(net, seqs, bp, use_stamina=False):
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    theta_f = net.theta.astype(np.float32)
    pol = np.where(net.polarity, 1.0, -1.0).astype(np.float32)
    pn = bp / (np.linalg.norm(bp, axis=1, keepdims=True) + 1e-8)
    n_edges = len(sc[0])
    preds = set()
    net.reset()
    seq = seqs[0]
    stam = np.full(n_edges, SelfWiringGraph.DEFAULT_STAMINA, dtype=np.uint8) if use_stamina else None
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

def eval_stamina_stats(net, seqs, bp, use_stamina=False):
    """Run one sequence and return stamina stats at end."""
    if not use_stamina:
        return {}
    sc = SelfWiringGraph.build_sparse_cache(net.mask)
    theta_f = net.theta.astype(np.float32)
    pol = np.where(net.polarity, 1.0, -1.0).astype(np.float32)
    n_edges = len(sc[0])
    net.reset()
    stam = np.full(n_edges, SelfWiringGraph.DEFAULT_STAMINA, dtype=np.uint8)
    seq = seqs[0]
    for i in range(len(seq) - 1):
        inj = bp[seq[i]] @ net.input_projection
        st, ch = SelfWiringGraph.rollout_token(
            inj, mask=net.mask, theta=theta_f, decay=net.decay,
            ticks=TICKS, input_duration=INPUT_DUR,
            state=net.state, charge=net.charge,
            sparse_cache=sc, polarity=pol,
            refractory=net.refractory, stamina=stam)
        net.state[:] = st; net.charge[:] = ch
    return {
        'mean': round(float(stam.mean()), 1),
        'min': int(stam.min()),
        'tired': int(np.sum(stam < 5)),
        'fresh': int(np.sum(stam >= 11)),
    }

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

    for step in range(1, STEPS + 1):
        saved = net.save_state()

        tr = [all_data[o:o + SEQ_LEN]
              for o in [np_rng.randint(0, len(all_data) - SEQ_LEN) for _ in range(2)]]

        old_score = eval_cos(net, tr, bp, bigram, use_stamina)
        undo = net.mutate()
        new_score = eval_cos(net, tr, bp, bigram, use_stamina)

        if new_score - old_score > THRESHOLD:
            pass  # keep
        else:
            net.restore_state(saved)

        if step % REPORT == 0:
            cos = eval_cos(net, eval_seqs, bp, bigram, use_stamina)
            div = eval_diversity(net, eval_seqs, bp, use_stamina)
            ss = eval_stamina_stats(net, eval_seqs, bp, use_stamina)
            edges = net.count_connections()
            theta_m = float(net.theta.mean())
            drive = int(net.mutation_drive)
            elapsed = time.time() - t0
            sps = step / elapsed

            es = set(zip(*[x.tolist() for x in np.where(net.mask)]))
            n2 = sum(1 for r, c in es if (c, r) in es) // 2

            stam_info = f' s={ss["mean"]}[{ss["min"]}] t={ss["tired"]} f={ss["fresh"]}' if ss else ''

            entry = {'step': step, 'cos': round(cos, 6), 'edges': edges,
                     'theta': round(theta_m, 2), 'drive': drive, '2cyc': n2,
                     'diversity': div, 'sps': round(sps, 2)}
            if ss: entry['stamina'] = ss
            log.append(entry)
            print(f'  [{step:5d}] cos={cos:.6f} e={edges} θ={theta_m:.2f} '
                  f'd={drive} 2c={n2} div={div} {sps:.1f}sps{stam_info}')
            sys.stdout.flush()

    ckpt = ROOT / 'recipes' / 'checkpoints' / f'overnight_v3_{label}.npz'
    net.save(str(ckpt))
    print(f'  Saved: {ckpt}')
    return {'label': label, 'log': log}


if __name__ == '__main__':
    print('Loading data...')
    all_data = load_fineweb_bytes()
    bigram = np.load(ROOT / 'recipes' / 'data' / 'bigram_table.npy')
    bp = make_bp()

    eval_rng = np.random.RandomState(9999)
    eval_seqs = [all_data[o:o + SEQ_LEN]
                 for o in [eval_rng.randint(0, len(all_data) - SEQ_LEN) for _ in range(N_EVAL)]]

    # Skip baseline — we already have it from v2 (identical result, same seed)
    # Only run stamina with the correct per-sequence reset model
    results = []
    for label, use_stam in [('stamina_per_seq', True)]:
        r = run_arm(label, use_stam, all_data, bigram, bp, eval_seqs)
        results.append(r)

    print(f'\n{"="*60}')
    print(f'  SUMMARY (compare with baseline cos=0.1791)')
    print(f'{"="*60}')
    for r in results:
        if r['log']:
            f = r['log'][-1]
            print(f'  {r["label"]:25s}: cos={f["cos"]:.6f} e={f["edges"]} div={f["diversity"]}')
            delta = f['cos'] - 0.1791
            print(f'  vs baseline: {delta:+.6f} ({"BETTER" if delta > 0.001 else "WORSE" if delta < -0.001 else "SAME"})')

    out_path = ROOT / 'recipes' / 'overnight_stamina_v3_results.json'
    with open(out_path, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'\nResults: {out_path}')
