"""
Breed from existing checkpoints: no training, just merge + crystallize + eval
==============================================================================
Load 2 trained networks, breed with consensus, fast crystallize, eval.
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

def eval_network(mask, theta, channel, pol_f32, bp_out, bigram, data, eval_seqs):
    sc = SelfWiringGraph.build_sparse_cache(mask)
    ea_list = []
    for s in eval_seqs:
        st=np.zeros(H,np.float32);ch=np.zeros(H,np.float32);cor=0;tot=0
        for i in range(len(s)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
            st,ch=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                state=st,charge=ch,sparse_cache=sc,polarity=pol_f32,channel=channel)
            logits=np.dot(bp_out,ch[H-OUT_DIM:])
            if np.argmax(logits)==s[i+1]:cor+=1
            tot+=1
        ea_list.append(cor/tot if tot else 0)
    return np.mean(ea_list)

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT_DIR))
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    resolve_fineweb_path(); ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram = np.load(os.path.join(BASE_DIR, "data", "bigram_table.npy"))
    bp_out = build_freq_order(OUT_DIM, bigram)
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[o:o+100] for o in [eval_rng.randint(0,len(ALL_DATA)-100) for _ in range(5)]]

    # Load 2 checkpoints
    ckpt_a = np.load(os.path.join(BASE_DIR, "data", "big_relu_main_network.npz"))
    ckpt_b = np.load(os.path.join(BASE_DIR, "data", "mini_ctrl_main_network.npz"))

    mask_a=ckpt_a['mask'];theta_a=ckpt_a['theta'];channel_a=ckpt_a['channel'];pol_a=ckpt_a['pol_f32']
    mask_b=ckpt_b['mask'];theta_b=ckpt_b['theta'];channel_b=ckpt_b['channel'];pol_b=ckpt_b['pol_f32']

    print(f"\n{'='*60}")
    print(f"  BREED FROM CHECKPOINTS")
    print(f"{'='*60}")

    # Eval parents
    score_a = eval_network(mask_a, theta_a, channel_a, pol_a, bp_out, bigram, ALL_DATA, eval_seqs)
    score_b = eval_network(mask_b, theta_b, channel_b, pol_b, bp_out, bigram, ALL_DATA, eval_seqs)
    print(f"  Parent A (big_relu): {score_a*100:.1f}%  edges={mask_a.sum()}")
    print(f"  Parent B (mini_ctrl): {score_b*100:.1f}%  edges={mask_b.sum()}")

    # Consensus breed
    better = 'A' if score_a >= score_b else 'B'
    mask_better = mask_a if better=='A' else mask_b
    mask_worse = mask_b if better=='A' else mask_a
    theta_c = (theta_a if better=='A' else theta_b).copy()
    channel_c = (channel_a if better=='A' else channel_b).copy()
    pol_c = (pol_a if better=='A' else pol_b).copy()

    consensus = mask_a & mask_b
    only_better = mask_better & ~mask_worse
    mask_c = consensus | only_better
    np.fill_diagonal(mask_c, False)

    print(f"\n  Consensus edges: {consensus.sum()}")
    print(f"  Better({better}) only: {only_better.sum()}")
    print(f"  Child total: {mask_c.sum()}")

    score_c = eval_network(mask_c, theta_c, channel_c, pol_c, bp_out, bigram, ALL_DATA, eval_seqs)
    print(f"  Child BEFORE crystallize: {score_c*100:.1f}%")

    # Fast crystallize: sample 300
    print(f"\n  Fast crystallize (300 sample)...", end="")
    sys.stdout.flush()
    alive = list(zip(*np.where(mask_c)))
    random.shuffle(alive)
    sample = alive[:300]
    base = score_c; removed = 0
    for r, c_idx in sample:
        mask_c[r, c_idx] = False
        sc = eval_network(mask_c, theta_c, channel_c, pol_c, bp_out, bigram, ALL_DATA, eval_seqs)
        if sc >= base - 0.0001:
            removed += 1; base = sc
        else:
            mask_c[r, c_idx] = True
    print(f" removed {removed}/300")

    score_pruned = eval_network(mask_c, theta_c, channel_c, pol_c, bp_out, bigram, ALL_DATA, eval_seqs)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Parent A: {score_a*100:.1f}%  ({mask_a.sum()} edges)")
    print(f"  Parent B: {score_b*100:.1f}%  ({mask_b.sum()} edges)")
    print(f"  Child before prune: {score_c*100:.1f}%  ({consensus.sum()+only_better.sum()} edges)")
    print(f"  Child after prune:  {score_pruned*100:.1f}%  ({mask_c.sum()} edges)")

    # Save child
    np.savez_compressed(os.path.join(BASE_DIR, "data", "breed_child.npz"),
        mask=mask_c, theta=theta_c, channel=channel_c, pol_f32=pol_c)
    print(f"  Saved breed_child.npz")
