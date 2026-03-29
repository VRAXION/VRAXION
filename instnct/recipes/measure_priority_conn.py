"""
Priority connection measurement: raw accept rates
===================================================
Load checkpoint, try 200 priority mutations each variant, measure accept/deny.

A: natural polarity (inherit source neuron's polarity)
B: try BOTH polarities, pick better delta (luxury, 2x eval cost)
C: reciprocal pair (A->B natural + B->A natural, both at once)
"""
import sys, os, time, random
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR)); sys.path.insert(0, str(ROOT_DIR / "model"))
from graph import SelfWiringGraph

H = 256; TICKS = 8; INPUT_DURATION = 2
PHI = (1 + 5**0.5) / 2
IN_DIM = int(round(H / PHI))
OUT_DIM = int(round(H / PHI))
SDR_K = int(round(IN_DIM * 0.20))
WAVE_LUT = SelfWiringGraph.WAVE_LUT
THRESHOLD = 0.00005

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

BP_IN = build_sdr(256, IN_DIM, SDR_K, 42)

def eval_bigram(mask, theta, channel, pol_f, bp_out, bigram, data, n_seqs=3, seq_len=50):
    sc = SelfWiringGraph.build_sparse_cache(mask)
    rng = np.random.RandomState(int(time.time()*1000) % 2**31)
    total = 0.0
    for _ in range(n_seqs):
        off = rng.randint(0, len(data)-seq_len)
        seq = data[off:off+seq_len]
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32);s=0.0;n=0
        for i in range(len(seq)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[seq[i]]
            state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                state=state,charge=charge,sparse_cache=sc,polarity=pol_f,
                channel=channel)
            logits=np.dot(bp_out,charge[H-OUT_DIM:])
            e=np.exp(logits-logits.max());pred=e/e.sum()
            tgt=bigram[seq[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            s+=cos;n+=1
        total+=s/n if n else 0
    return total/n_seqs

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT_DIR))
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    resolve_fineweb_path(); ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram = np.load(os.path.join(BASE_DIR, "data", "bigram_table.npy"))
    bp_out = build_freq_order(OUT_DIM, bigram)

    # Load checkpoint
    ckpt = np.load(os.path.join(BASE_DIR, "data", "measure_100_checkpoint.npz"))
    mask = ckpt['mask'].copy()
    theta = ckpt['theta'].copy()
    channel = ckpt['channel'].copy()
    pol_f32 = ckpt['pol_f32'].copy()
    pol_bool = (pol_f32 > 0)

    in_deg = mask.sum(axis=0)
    out_deg = mask.sum(axis=1)
    total_deg = in_deg + out_deg
    top_neurons = np.argsort(total_deg)[::-1][:H//4]

    print(f"\n{'='*60}")
    print(f"  PRIORITY CONNECTION MEASUREMENT")
    print(f"  checkpoint: theta={theta.mean():.2f} edges={mask.sum()}")
    print(f"  top 25% neurons (degree>{total_deg[top_neurons[-1]]}): {len(top_neurons)}")
    print(f"{'='*60}")

    N_TRIALS = 200
    results = {}

    for mode in ['A_natural', 'B_best_polarity', 'C_reciprocal']:
        print(f"\n>> {mode}: ", end="")
        sys.stdout.flush()

        accepted = 0
        rejected_no_slot = 0
        rejected_worse = 0
        total_pos_delta = 0.0
        total_neg_delta = 0.0

        # Work on copy
        m = mask.copy()
        t = theta.copy()
        p = pol_f32.copy()
        pb = pol_bool.copy()

        for trial in range(N_TRIALS):
            src = int(np.random.choice(top_neurons))
            tgt = int(np.random.choice(top_neurons))
            if src == tgt or m[src, tgt]:
                rejected_no_slot += 1
                continue

            # Baseline score
            old_score = eval_bigram(m, t, channel, p, bp_out, bigram, ALL_DATA)

            if mode == 'A_natural':
                # Add connection with source's natural polarity
                nm = m.copy(); nm[src, tgt] = True
                new_score = eval_bigram(nm, t, channel, p, bp_out, bigram, ALL_DATA)
                delta = new_score - old_score

            elif mode == 'B_best_polarity':
                # Try natural polarity
                nm = m.copy(); nm[src, tgt] = True
                score_nat = eval_bigram(nm, t, channel, p, bp_out, bigram, ALL_DATA)
                # Try flipped polarity (flip source)
                np2 = p.copy()
                np2[src] = -np2[src]
                score_flip = eval_bigram(nm, t, channel, np2, bp_out, bigram, ALL_DATA)
                # Pick better
                if score_flip > score_nat:
                    new_score = score_flip
                    p_use = np2
                else:
                    new_score = score_nat
                    p_use = p
                delta = new_score - old_score

            elif mode == 'C_reciprocal':
                # Add both directions
                if m[tgt, src]:
                    rejected_no_slot += 1
                    continue
                nm = m.copy(); nm[src, tgt] = True; nm[tgt, src] = True
                new_score = eval_bigram(nm, t, channel, p, bp_out, bigram, ALL_DATA)
                delta = new_score - old_score

            if delta > THRESHOLD:
                accepted += 1
                total_pos_delta += delta
                # Apply mutation
                if mode == 'A_natural':
                    m[src, tgt] = True
                elif mode == 'B_best_polarity':
                    m[src, tgt] = True
                    p[:] = p_use
                elif mode == 'C_reciprocal':
                    m[src, tgt] = True; m[tgt, src] = True
                print("+", end="")
            else:
                rejected_worse += 1
                total_neg_delta += abs(delta)
                print(".", end="")
            sys.stdout.flush()

        total_tried = N_TRIALS - rejected_no_slot
        rate = accepted / max(total_tried, 1) * 100
        avg_pos = total_pos_delta / max(accepted, 1)
        avg_neg = total_neg_delta / max(rejected_worse, 1)

        results[mode] = {
            'tried': total_tried, 'accepted': accepted, 'no_slot': rejected_no_slot,
            'rejected': rejected_worse, 'rate': rate,
            'avg_pos_delta': avg_pos, 'avg_neg_delta': avg_neg
        }
        print(f"\n  tried={total_tried} accepted={accepted} rate={rate:.1f}% "
              f"avg_pos={avg_pos:.6f} avg_neg={avg_neg:.6f} no_slot={rejected_no_slot}")

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  {'mode':>20s}  {'tried':>5s}  {'accept':>6s}  {'rate':>5s}  {'avg+delta':>10s}  {'avg-delta':>10s}")
    for mode, r in results.items():
        print(f"  {mode:>20s}  {r['tried']:5d}  {r['accepted']:6d}  {r['rate']:4.1f}%  {r['avg_pos_delta']:10.6f}  {r['avg_neg_delta']:10.6f}")
    print(f"{'='*60}")
