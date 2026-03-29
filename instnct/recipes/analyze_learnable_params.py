"""
Learnable parameter distribution analysis
==========================================
Analyze ALL learnable params from trained checkpoints:
  - theta (int4 [1-15]): which values cluster? R/C/G zones?
  - mask (bool): density, in/out degree distribution
  - polarity (bool): E/I ratio

Also runs a FRESH training (B mode = best config) and captures
convergence snapshots every 200 steps for temporal analysis.
"""
import sys, os, time, random, json
import numpy as np
from multiprocessing import Pool
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR)); sys.path.insert(0, str(ROOT_DIR / "model"))
from graph import SelfWiringGraph

H = 256; N_WORKERS = 18; TICKS = 8; INPUT_DURATION = 2
THRESHOLD = 0.00005; EVAL_EVERY = 20
INIT_DENSITY = 0.05
PHI = (1 + 5**0.5) / 2
IN_DIM = int(round(H / PHI))
OUT_DIM = int(round(H / PHI))
SDR_K = int(round(IN_DIM * 0.20))
SCHEDULE = ['add','flip','theta','theta','theta','flip','flip','remove']

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
_bp_out=None;_all_data=None;_bigram=None;_pol=None;_freq=None;_phase=None;_rho=None

def init_w(bpo,data,bg,pol,fr,ph,rh):
    global _bp_out,_all_data,_bigram,_pol,_freq,_phase,_rho
    _bp_out=bpo;_all_data=data;_bigram=bg;_pol=pol;_freq=fr;_phase=ph;_rho=rh

def _eval_bigram(mask, theta, seqs):
    sc = SelfWiringGraph.build_sparse_cache(mask); total = 0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32);s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                state=state,charge=charge,sparse_cache=sc,polarity=_pol,
                freq=_freq,phase=_phase,rho=_rho)
            logits=np.dot(_bp_out,charge[H-OUT_DIM:])
            e=np.exp(logits-logits.max());pred=e/e.sum()
            tgt=_bigram[tb[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            s+=cos;n+=1
        total+=s/n if n else 0
    return total/len(seqs)

def worker_eval(args):
    mf,theta,seed,pt=args
    rng=random.Random(seed);nrng=np.random.RandomState(seed)
    mask=mf.reshape(H,H);nm=mask;nt=theta
    if pt=='add':
        r=rng.randint(0,H-1);c=rng.randint(0,H-1)
        if r==c or mask[r,c]:return{'delta':-1e9,'type':'add'}
        nm=mask.copy();nm[r,c]=True
    elif pt=='remove':
        alive=list(zip(*np.where(mask)))
        if len(alive)<1:return{'delta':-1e9,'type':'remove'}
        r,c=alive[rng.randint(0,len(alive)-1)];nm=mask.copy();nm[r,c]=False
    elif pt=='flip':
        alive=list(zip(*np.where(mask)))
        if len(alive)<1:return{'delta':-1e9,'type':'flip'}
        r,c=alive[rng.randint(0,len(alive)-1)];nc=rng.randint(0,H-1)
        if nc==r or nc==c or mask[r,nc]:return{'delta':-1e9,'type':'flip'}
        nm=mask.copy();nm[r,c]=False;nm[r,nc]=True
    elif pt=='theta':
        idx=rng.randint(0,H-1);nt=theta.copy();nt[idx]=float(rng.randint(1,15))
    seqs=[]
    for _ in range(2):
        off=nrng.randint(0,len(_all_data)-100);seqs.append(_all_data[off:off+100])
    old=_eval_bigram(mask,theta,seqs);new=_eval_bigram(nm,nt,seqs)
    return{'delta':float(new-old),'type':pt,
           'new_mask_flat':nm.flatten() if new>old else None,
           'new_theta':nt if pt=='theta' else None}

def ascii_hist(values, bins=15, width=40, label=""):
    counts, edges = np.histogram(values, bins=bins)
    mx = max(counts) if max(counts) > 0 else 1
    print(f"\n  {label}  (N={len(values)}, mean={np.mean(values):.3f}, std={np.std(values):.3f})")
    for i in range(bins):
        bar = '#' * int(counts[i] / mx * width)
        print(f"    [{edges[i]:6.2f},{edges[i+1]:6.2f}) {counts[i]:4d} |{bar}")

def analyze_snapshot(mask, theta, polarity, step, best):
    """Full analysis of learnable params at a snapshot."""
    print(f"\n{'='*60}")
    print(f"  SNAPSHOT step={step} best={best*100:.1f}%")
    print(f"{'='*60}")

    # --- THETA ---
    th = theta.astype(int)
    counts = np.bincount(th, minlength=16)[1:16]  # values 1-15
    total = counts.sum()
    print(f"\n  THETA distribution (int4 [1-15]):")
    print(f"    mean={theta.mean():.2f} std={theta.std():.2f} median={np.median(theta):.0f}")
    zones = {'R(1-4)':counts[0:4].sum(), 'C(5-10)':counts[4:10].sum(), 'G(11-15)':counts[10:15].sum()}
    print(f"    Zones: {', '.join(f'{k}={v}({v/total*100:.0f}%)' for k,v in zones.items())}")
    for v in range(1, 16):
        bar = '#' * int(counts[v-1] / max(counts.max(),1) * 35)
        print(f"    theta={v:2d}: {counts[v-1]:3d} ({counts[v-1]/total*100:4.1f}%) |{bar}")

    # --- MASK ---
    edges = mask.sum()
    density = edges / (H*H)
    in_deg = mask.sum(axis=0)   # columns = incoming edges per neuron
    out_deg = mask.sum(axis=1)  # rows = outgoing edges per neuron

    print(f"\n  MASK topology:")
    print(f"    edges={edges} density={density:.4f} ({density*100:.2f}%)")
    print(f"    in-degree:  mean={in_deg.mean():.1f} std={in_deg.std():.1f} [{in_deg.min()},{in_deg.max()}]")
    print(f"    out-degree: mean={out_deg.mean():.1f} std={out_deg.std():.1f} [{out_deg.min()},{out_deg.max()}]")

    # Zone analysis: input neurons [0:IN_DIM], output neurons [H-OUT_DIM:H], overlap
    overlap_start = H - OUT_DIM
    overlap_end = IN_DIM
    in_only = list(range(overlap_end, IN_DIM)) if overlap_end < IN_DIM else []  # should be empty with phi overlap
    out_only = list(range(overlap_start, H))
    overlap_zone = list(range(overlap_start, min(overlap_end, H)))
    input_zone = list(range(0, IN_DIM))
    output_zone = list(range(H - OUT_DIM, H))

    if len(input_zone) > 0:
        in_in = in_deg[input_zone].mean()
        in_out = out_deg[input_zone].mean()
        print(f"    Input zone [0:{IN_DIM}]:  in_deg={in_in:.1f} out_deg={in_out:.1f}")
    if len(output_zone) > 0:
        out_in = in_deg[output_zone].mean()
        out_out = out_deg[output_zone].mean()
        print(f"    Output zone [{H-OUT_DIM}:{H}]: in_deg={out_in:.1f} out_deg={out_out:.1f}")

    # Degree distribution histogram
    ascii_hist(in_deg.astype(float), bins=15, label="In-degree distribution")
    ascii_hist(out_deg.astype(float), bins=15, label="Out-degree distribution")

    # --- POLARITY ---
    n_exc = polarity.sum()
    n_inh = H - n_exc
    print(f"\n  POLARITY:")
    print(f"    Excitatory: {n_exc} ({n_exc/H*100:.1f}%)")
    print(f"    Inhibitory: {n_inh} ({n_inh/H*100:.1f}%)")

    # Inhibitory hub analysis
    inh_idx = np.where(~polarity)[0]
    if len(inh_idx) > 0:
        inh_out = out_deg[inh_idx]
        exc_out = out_deg[np.where(polarity)[0]]
        print(f"    Inh out-degree: mean={inh_out.mean():.1f} vs Exc: mean={exc_out.mean():.1f}")
        ratio = inh_out.mean() / max(exc_out.mean(), 0.01)
        print(f"    Inh/Exc fan-out ratio: {ratio:.2f}x (FlyWire: ~2x)")

    # Theta by zone
    if len(input_zone) > 0 and len(output_zone) > 0:
        th_in = theta[input_zone].mean()
        th_out = theta[output_zone].mean()
        th_mid = theta[IN_DIM:H-OUT_DIM].mean() if IN_DIM < H-OUT_DIM else 0
        print(f"\n  THETA by zone:")
        print(f"    Input zone:  mean={th_in:.2f}")
        print(f"    Output zone: mean={th_out:.2f}")
        if th_mid > 0:
            print(f"    Hidden zone: mean={th_mid:.2f}")

    # Theta by polarity
    th_exc = theta[polarity].mean() if polarity.any() else 0
    th_inh = theta[~polarity].mean() if (~polarity).any() else 0
    print(f"    Excitatory theta: mean={th_exc:.2f}")
    print(f"    Inhibitory theta: mean={th_inh:.2f}")

    sys.stdout.flush()

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

    print(f"\n{'='*60}")
    print(f"  LEARNABLE PARAM ANALYSIS (best config: B mode)")
    print(f"  H={H} IN={IN_DIM} OUT={OUT_DIM} SDR_K={SDR_K}")
    print(f"{'='*60}")

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
    pol = ref.polarity.copy()  # bool
    pol_f32 = np.where(pol, 1.0, -1.0).astype(np.float32)
    rho = np.full(H, 0.3, np.float32)
    irng = np.random.RandomState(42)
    mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask, False)
    theta = np.full(H, 1.0, np.float32)

    # Frozen wave (B mode winner)
    frng = np.random.RandomState(42)
    freq_arr = (frng.rand(H).astype(np.float32) * 1.5 + 0.5)
    phase_arr = (frng.rand(H).astype(np.float32) * 2.0 * np.pi)

    # Initial snapshot
    analyze_snapshot(mask, theta, pol, 0, 0.0)

    pool = Pool(N_WORKERS, initializer=init_w,
        initargs=(bp_out, ALL_DATA, bigram, pol_f32, freq_arr, phase_arr, rho))

    best=0;acc=0;t0=time.time()
    for step in range(1, 2001):
        pt=SCHEDULE[(step-1)%len(SCHEDULE)]
        if pt in('flip','remove','theta') and mask.sum()==0:pt='add'
        args=[(mask.flatten(),theta.copy(),1000+step*50+w,pt) for w in range(N_WORKERS)]
        res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])
        if br['delta']>THRESHOLD:
            if br['type'] in('add','remove','flip') and br['new_mask_flat'] is not None:
                mask[:]=br['new_mask_flat'].reshape(H,H);acc+=1
            elif br['type']=='theta' and br['new_theta'] is not None:
                theta[:]=br['new_theta'];acc+=1
        if step%EVAL_EVERY==0:
            ea_list=[]
            for s in eval_seqs:
                sc=SelfWiringGraph.build_sparse_cache(mask)
                st=np.zeros(H,np.float32);ch=np.zeros(H,np.float32);cor=0;tot=0
                for i in range(len(s)-1):
                    inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
                    st,ch=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                        decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                        state=st,charge=ch,sparse_cache=sc,polarity=pol_f32,
                        freq=freq_arr,phase=phase_arr,rho=rho)
                    logits=np.dot(bp_out,ch[H-OUT_DIM:])
                    if np.argmax(logits)==s[i+1]:cor+=1
                    tot+=1
                ea_list.append(cor/tot if tot else 0)
            ea=np.mean(ea_list)
            if ea>best:best=ea
            if step%200==0:
                print(f"\n  [{step:4d}] eval={ea*100:.1f}% best={best*100:.1f}% th={theta.mean():.1f} e={mask.sum()} acc={acc} {time.time()-t0:.0f}s")
                sys.stdout.flush()
            # Snapshots at 500, 1000, 1500, 2000
            if step in (500, 1000, 1500, 2000):
                analyze_snapshot(mask, theta, pol, step, best)

    pool.terminate();pool.join()
    print(f"\n  TOTAL TIME: {time.time()-t0:.0f}s")
    print(f"  FINAL BEST: {best*100:.1f}%")

    # Save final state
    np.savez_compressed(os.path.join(BASE_DIR,"data/learnable_params_final.npz"),
        mask=mask, theta=theta, polarity=pol, freq=freq_arr, phase=phase_arr)
    print(f"  Saved learnable_params_final.npz")
    sys.stdout.flush()
