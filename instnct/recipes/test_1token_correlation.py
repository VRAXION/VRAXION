"""
Test: does 1-token eval correlate with 100-token eval?
======================================================
Load a trained checkpoint, apply N random mutations, measure delta with
both 1-token and 100-token eval. Compute Pearson/Spearman correlation.

If correlation is high (>0.7), we can use 1-token eval for fast screening
and only confirm with 100-token — giving ~16x training speedup.
"""
import sys, os, time, random
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph
from quaternary_mask import QuaternaryMask

H = 256; TICKS = 12; INPUT_DURATION = 2
PHI = (1 + 5**0.5) / 2
IN_DIM = int(round(H / PHI))
OUT_DIM = int(round(H / PHI))
SDR_K = int(round(IN_DIM * 0.20))
SEQ_LEN = 100
N_MUTATIONS = 200  # test 200 random mutations

def build_sdr(n, dim, k, seed):
    rng = np.random.RandomState(seed)
    t = np.zeros((n, dim), np.float32)
    for v in range(n): t[v, rng.choice(dim, size=k, replace=False)] = 1.0
    return t

def build_freq_order(dim, bigram, seed=12345):
    freq = bigram.sum(axis=0) + bigram.sum(axis=1)
    rank = np.argsort(freq)[::-1]
    rng = np.random.RandomState(seed)
    p = np.zeros((256, dim), np.float32)
    for i, byte_idx in enumerate(rank):
        t = i / 255.0
        for d in range(dim):
            p[byte_idx, d] = np.sin(2 * np.pi * t * (d+1) / dim * 3) + rng.randn() * 0.3
    p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-8
    return p.astype(np.float32)

def eval_bigram_ntokens(qdata, theta, channel, pol_f, text_bytes, n_tokens, bp_in, bp_out):
    """Eval bigram cosine on first n_tokens of text_bytes."""
    qm = QuaternaryMask(H, qdata)
    sc = qm.to_directed_edges()
    _dummy = np.zeros((H, H), dtype=bool)
    state = np.zeros(H, np.float32); charge = np.zeros(H, np.float32)
    total_cos = 0.0; n = 0
    limit = min(n_tokens, len(text_bytes) - 1)
    for i in range(limit):
        inj = np.zeros(H, np.float32); inj[0:IN_DIM] = bp_in[text_bytes[i]]
        state, charge = SelfWiringGraph.rollout_token(
            inj, mask=_dummy, theta=theta,
            decay=np.float32(0.16), ticks=TICKS, input_duration=INPUT_DURATION,
            state=state, charge=charge, sparse_cache=sc,
            polarity=pol_f, channel=channel)
        logits = np.dot(bp_out, charge[H - OUT_DIM:])
        e = np.exp(logits - logits.max()); pred = e / e.sum()
        tgt = bigram[text_bytes[i]]
        cos = np.dot(pred, tgt) / (np.linalg.norm(pred) * np.linalg.norm(tgt) + 1e-8)
        total_cos += cos; n += 1
    return total_cos / n if n else 0


if __name__ == "__main__":
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    resolve_fineweb_path()
    ALL_DATA = load_fineweb_bytes()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram = np.load(os.path.join(BASE_DIR, "data", "bigram_table.npy"))
    bp_in = build_sdr(256, IN_DIM, SDR_K, 42)
    bp_out = build_freq_order(OUT_DIM, bigram)

    # Load checkpoint
    ckpt_path = os.path.join(BASE_DIR, "data", "build_checkpoint.npz")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: no checkpoint at {ckpt_path}")
        sys.exit(1)
    ckpt = np.load(ckpt_path)
    qdata = ckpt['qdata']
    theta = ckpt['theta']
    channel = ckpt['channel']
    pol_f = ckpt['pol_f']
    print(f"Loaded checkpoint: {QuaternaryMask(H, qdata).count_edges()} edges")

    # Pick eval sequences
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[off:off + SEQ_LEN]
                 for off in [eval_rng.randint(0, len(ALL_DATA) - SEQ_LEN)
                             for _ in range(5)]]

    # For each mutation: measure delta with 1-token, 5-token, 10-token, 100-token
    TOKEN_COUNTS = [1, 3, 5, 10, 50, 99]
    OPS = ['add', 'remove', 'reverse', 'loop3', 'loop5', 'theta', 'channel', 'flip']

    results = {n: [] for n in TOKEN_COUNTS}
    print(f"Testing {N_MUTATIONS} mutations x {len(TOKEN_COUNTS)} token counts...")
    print(f"Token counts: {TOKEN_COUNTS}")
    print()

    t0 = time.time()
    for i in range(N_MUTATIONS):
        rng = random.Random(1000 + i)
        op = OPS[i % len(OPS)]

        # Apply mutation
        qm_new = QuaternaryMask(H, qdata.copy())
        nt = theta.copy(); nc = channel.copy(); npf = pol_f.copy()
        undo = []

        if op == 'add':
            qm_new.mutate_add(rng, undo)
        elif op == 'remove':
            qm_new.mutate_remove(rng, undo)
        elif op == 'reverse':
            qm_new.mutate_flip(rng, undo)
        elif op in ('loop3', 'loop5'):
            loop_len = int(op[4:])
            nodes = [rng.randint(0, H-1)]
            ok = True
            for _ in range(loop_len - 1):
                n = rng.randint(0, H-1)
                if n in nodes: ok = False; break
                nodes.append(n)
            if ok:
                for k in range(loop_len):
                    r, c = nodes[k], nodes[(k+1) % loop_len]
                    if qm_new.get_pair(r, c) != 0: ok = False; break
                if ok:
                    for k in range(loop_len):
                        r, c = nodes[k], nodes[(k+1) % loop_len]
                        qm_new.set_pair(r, c, 1)
        elif op == 'theta':
            idx = rng.randint(0, H-1); nt = theta.copy()
            nt[idx] = float(rng.randint(1, 15))
        elif op == 'channel':
            idx = rng.randint(0, H-1); nc = channel.copy()
            nc[idx] = np.uint8(rng.randint(1, 8))
        elif op == 'flip':
            idx = rng.randint(0, H-1)
            npf = pol_f.copy(); npf[idx] *= -1

        # Eval at each token count using same sequence
        seq = eval_seqs[i % len(eval_seqs)]
        for n_tok in TOKEN_COUNTS:
            old_score = eval_bigram_ntokens(qdata, theta, channel, pol_f, seq, n_tok, bp_in, bp_out)
            new_score = eval_bigram_ntokens(qm_new.data, nt, nc, npf, seq, n_tok, bp_in, bp_out)
            results[n_tok].append(new_score - old_score)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{N_MUTATIONS}] {elapsed:.0f}s")

    # Compute correlations
    from scipy import stats as sp_stats

    ref = np.array(results[99])  # 99-token as reference (closest to full 100)
    print(f"\n{'='*60}")
    print(f"  CORRELATION: N-token delta vs 99-token delta")
    print(f"  ({N_MUTATIONS} mutations)")
    print(f"{'='*60}")
    print(f"  {'Tokens':>7s}  {'Pearson r':>10s}  {'Spearman r':>10s}  {'p-value':>10s}")
    print(f"  {'-'*7:>7s}  {'-'*10:>10s}  {'-'*10:>10s}  {'-'*10:>10s}")

    corr_data = {}
    for n_tok in TOKEN_COUNTS:
        if n_tok == 99:
            continue
        x = np.array(results[n_tok])
        pearson_r, pearson_p = sp_stats.pearsonr(x, ref)
        spearman_r, spearman_p = sp_stats.spearmanr(x, ref)
        print(f"  {n_tok:7d}  {pearson_r:10.3f}  {spearman_r:10.3f}  {spearman_p:10.2e}")
        corr_data[n_tok] = {
            'pearson_r': float(pearson_r), 'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
        }

    # Same-sign agreement (would accept/reject decision match?)
    print(f"\n  SAME-SIGN AGREEMENT (both positive or both negative):")
    for n_tok in TOKEN_COUNTS:
        if n_tok == 99: continue
        x = np.array(results[n_tok])
        agree = np.sum(np.sign(x) == np.sign(ref))
        print(f"  {n_tok:3d} tokens: {agree}/{len(ref)} ({agree/len(ref)*100:.1f}%)")

    print(f"\n{'='*60}")
    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.0f}s")

    # Save
    import json
    out = {
        'config': {'H': H, 'ticks': TICKS, 'n_mutations': N_MUTATIONS,
                   'token_counts': TOKEN_COUNTS, 'ops': OPS},
        'correlations': corr_data,
        'same_sign': {n_tok: float(np.sum(np.sign(np.array(results[n_tok])) == np.sign(ref)) / len(ref))
                      for n_tok in TOKEN_COUNTS if n_tok != 99},
    }
    out_path = os.path.join(BASE_DIR, "data", "1token_correlation.json")
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved to {out_path}")
