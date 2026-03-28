"""
Output Projection Sweep: Random vs Language-Aware
=================================================
H=256, phi overlap (in=158, out=158), learnable theta.

A: RANDOM      — current best, random unit vectors (baseline 20.8%)
B: FREQ_ORDER  — byte vectors ordered by English frequency (e,t,a,o,i... nearby = similar)
C: BIGRAM_SIM  — byte vectors from bigram similarity (bytes with similar next-char distributions get similar vectors)
D: RIVAL_PAIRS — pairwise: each dimension = one "rivalry" between two frequently competing bytes

All produce 256 x out_dim projection matrices, same readout: logit = bp_out @ charge
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

H = 256; N_WORKERS = 18; TICKS = 8; INPUT_DURATION = 2
THRESHOLD = 0.00005; EVAL_EVERY = 20; THETA_INIT = 1.0
SCHEDULE = ['add', 'remove', 'flip', 'theta', 'theta', 'theta', 'decay', 'decay']
INIT_DENSITY = 0.05
PHI = (1 + 5**0.5) / 2
IN_DIM = int(round(H / PHI))   # 158
OUT_DIM = int(round(H / PHI))  # 158
SDR_K = int(round(IN_DIM * 0.20))

def build_sdr(n, dim, k, seed):
    rng = np.random.RandomState(seed)
    t = np.zeros((n, dim), np.float32)
    for v in range(n):
        t[v, rng.choice(dim, size=k, replace=False)] = 1.0
    return t

BP_IN = build_sdr(256, IN_DIM, SDR_K, 42)

def build_random_proj(out_dim, seed=12345):
    """A: Pure random unit vectors."""
    rng = np.random.RandomState(seed)
    p = rng.randn(256, out_dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

def build_freq_order_proj(out_dim, bigram):
    """B: Bytes ordered by frequency get nearby vectors (smooth frequency manifold)."""
    freq = bigram.sum(axis=0)  # incoming freq per byte
    freq += bigram.sum(axis=1)  # total involvement
    rank = np.argsort(freq)[::-1]  # most frequent first

    rng = np.random.RandomState(42)
    # Base: random vectors, but sorted bytes get smoothed neighbors
    p = np.zeros((256, out_dim), np.float32)
    for i, byte_idx in enumerate(rank):
        # Position on a curve through the space
        t = i / 255.0
        base = np.zeros(out_dim, np.float32)
        for d in range(out_dim):
            freq_d = (d + 1) / out_dim
            base[d] = np.sin(2 * np.pi * t * freq_d * 3) + rng.randn() * 0.3
        p[byte_idx] = base
    p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-8
    return p.astype(np.float32)

def build_bigram_sim_proj(out_dim, bigram):
    """C: Bytes with similar bigram distributions get similar vectors (SVD of bigram)."""
    # SVD of bigram matrix → first out_dim components
    U, S, Vt = np.linalg.svd(bigram, full_matrices=False)
    # Use first out_dim columns of U (left singular vectors)
    p = U[:, :out_dim] * S[:out_dim]  # scale by singular values
    p = p.astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-8
    return p

def build_rival_proj(out_dim, bigram):
    """D: Each dimension = a rivalry between two competing bytes.
    Pick top-N byte pairs that most frequently compete (highest co-occurrence in bigram rows)."""
    # For each context, find the top-2 bytes — those are rivals
    rivals = []
    for ctx in range(256):
        dist = bigram[ctx]
        top2 = np.argsort(dist)[::-1][:2]
        if dist[top2[0]] > 0.01 and dist[top2[1]] > 0.01:
            rivals.append((top2[0], top2[1], dist[top2[0]], dist[top2[1]]))

    # Sort by combined probability (strongest rivalries first)
    rivals.sort(key=lambda x: x[2] + x[3], reverse=True)

    # Build projection: each dim is a rivalry axis
    p = np.zeros((256, out_dim), np.float32)
    rng = np.random.RandomState(42)
    for d in range(min(out_dim, len(rivals))):
        a, b, pa, pb = rivals[d % len(rivals)]
        # byte a → +1 on this dim, byte b → -1, others get small random
        p[a, d] = 1.0
        p[b, d] = -1.0
        # Similar bytes get partial signal
        for other in range(256):
            if other != a and other != b:
                sim_a = np.dot(bigram[other], bigram[a]) / (np.linalg.norm(bigram[other]) * np.linalg.norm(bigram[a]) + 1e-8)
                sim_b = np.dot(bigram[other], bigram[b]) / (np.linalg.norm(bigram[other]) * np.linalg.norm(bigram[b]) + 1e-8)
                p[other, d] = (sim_a - sim_b) * 0.3
    # Add small random noise and normalize
    p += rng.randn(256, out_dim).astype(np.float32) * 0.1
    p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-8
    return p.astype(np.float32)

# Worker globals
_bp_out=None;_all_data=None;_bigram=None;_pol=None;_freq=None;_phase=None;_rho=None

def init_w(bpo,data,bg,pol,fr,ph,rh):
    global _bp_out,_all_data,_bigram,_pol,_freq,_phase,_rho
    _bp_out=bpo;_all_data=data;_bigram=bg;_pol=pol;_freq=fr;_phase=ph;_rho=rh

def _eval_bigram(mask,theta,decay,seqs):
    sc=SelfWiringGraph.build_sparse_cache(mask);total=0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32);s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,decay=decay,
                ticks=TICKS,input_duration=INPUT_DURATION,state=state,charge=charge,
                sparse_cache=sc,polarity=_pol,freq=_freq,phase=_phase,rho=_rho)
            logits=np.dot(_bp_out,charge[H-OUT_DIM:])
            e=np.exp(logits-logits.max());pred=e/e.sum()
            tgt=_bigram[tb[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            s+=cos;n+=1
        total+=s/n if n else 0
    return total/len(seqs)

def worker_eval(args):
    mf,theta,decay,seed,pt=args
    rng=random.Random(seed);nrng=np.random.RandomState(seed)
    mask=mf.reshape(H,H);nm=mask;nt=theta;nd=decay
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
        idx=rng.randint(0,H-1);nt=theta.copy();nt[idx]=rng.uniform(0.0,16.0)
    elif pt=='decay':
        idx=rng.randint(0,H-1);nd=decay.copy()
        nd[idx]=max(0.01,min(0.5,decay[idx]+rng.uniform(-0.03,0.03)))
    seqs=[]
    for _ in range(2):
        off=nrng.randint(0,len(_all_data)-100);seqs.append(_all_data[off:off+100])
    old=_eval_bigram(mask,theta,decay,seqs)
    new=_eval_bigram(nm,nt,nd,seqs)
    return{'delta':float(new-old),'type':pt,
           'new_mask_flat':nm.flatten() if new>old else None,
           'new_theta':nt if pt=='theta' else None,
           'new_decay':nd if pt=='decay' else None}

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    resolve_fineweb_path(); ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram = np.load(os.path.join(BASE_DIR, "data", "bigram_table.npy"))
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[o:o+100] for o in [eval_rng.randint(0,len(ALL_DATA)-100) for _ in range(5)]]

    MODES = [
        ('A', 'RANDOM',       build_random_proj(OUT_DIM)),
        ('B', 'FREQ_ORDER',   build_freq_order_proj(OUT_DIM, bigram)),
        ('C', 'BIGRAM_SVD',   build_bigram_sim_proj(OUT_DIM, bigram)),
        ('D', 'RIVAL_PAIRS',  build_rival_proj(OUT_DIM, bigram)),
    ]

    print(f"\n{'='*60}")
    print(f"  OUTPUT PROJECTION SWEEP: Random vs Language-Aware")
    print(f"  H={H}, phi overlap, in={IN_DIM} out={OUT_DIM}")
    print(f"{'='*60}")

    results = []
    for mk, label, bp_out in MODES:
        print(f"\n  {mk}: {label}")
        random.seed(42); np.random.seed(42)
        ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
        pol = ref.polarity.astype(np.float32)
        irng = np.random.RandomState(42)
        mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask, False)
        theta = np.full(H, THETA_INIT, np.float32)
        decay = np.random.RandomState(99).uniform(0.08, 0.24, H).astype(np.float32)

        pool = Pool(N_WORKERS, initializer=init_w,
            initargs=(bp_out, ALL_DATA, bigram, pol, ref.freq, ref.phase, ref.rho))
        best=0;stall=0;acc=0;t0=time.time()
        try:
            for step in range(1, 2001):
                pt=SCHEDULE[(step-1)%len(SCHEDULE)]
                if pt in('flip','remove','decay','theta') and mask.sum()==0:pt='add'
                args=[(mask.flatten(),theta.copy(),decay.copy(),1000+step*50+w,pt) for w in range(N_WORKERS)]
                res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])
                if br['delta']>THRESHOLD:
                    if br['type'] in('add','remove','flip') and br['new_mask_flat'] is not None:
                        mask[:]=br['new_mask_flat'].reshape(H,H);acc+=1
                    elif br['type']=='theta' and br['new_theta'] is not None:
                        theta[:]=br['new_theta'];acc+=1
                    elif br['type']=='decay' and br['new_decay'] is not None:
                        decay[:]=br['new_decay'];acc+=1
                if step%EVAL_EVERY==0:
                    el=time.time()-t0;ea_list=[]
                    for s in eval_seqs:
                        sc=SelfWiringGraph.build_sparse_cache(mask)
                        st=np.zeros(H,np.float32);ch=np.zeros(H,np.float32);cor=0;tot=0
                        for i in range(len(s)-1):
                            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
                            st,ch=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,decay=decay,
                                ticks=TICKS,input_duration=INPUT_DURATION,state=st,charge=ch,
                                sparse_cache=sc,polarity=pol,freq=ref.freq,phase=ref.phase,rho=ref.rho)
                            logits=np.dot(bp_out,ch[H-OUT_DIM:])
                            if np.argmax(logits)==s[i+1]:cor+=1
                            tot+=1
                        ea_list.append(cor/tot if tot else 0)
                    ea=np.mean(ea_list)
                    if ea>best:best=ea;stall=0
                    else:stall+=EVAL_EVERY
                    if step%400==0:
                        print(f"    [{step:4d}] eval={ea*100:.1f}% best={best*100:.1f}% theta={theta.mean():.2f} acc={acc} {el:.0f}s")
                    sys.stdout.flush()
                    if stall>=800 and step>=600:print(f"    ** STALL");break
        finally:pool.terminate();pool.join()
        elapsed=time.time()-t0
        results.append({'mode':mk,'label':label,'best':best,'acc':acc,'time':elapsed})
        print(f"    DONE: {label} best={best*100:.1f}% acc={acc} {elapsed:.0f}s")

    print(f"\n{'='*60}")
    print(f"  OUTPUT PROJECTION RESULTS")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['mode']}: {r['label']:15s} best={r['best']*100:5.1f}% acc={r['acc']:5d} {r['time']:.0f}s")
    best_r = max(results, key=lambda x: x['best'])
    print(f"\n  WINNER: {best_r['mode']} ({best_r['label']}) -- {best_r['best']*100:.1f}%")
    print(f"{'='*60}")
