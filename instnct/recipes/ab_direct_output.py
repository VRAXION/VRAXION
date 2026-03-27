"""
A/B: Random projection output vs Direct 256 output neurons (1:1 byte mapping)
Both use SDR_64 input. H=512 to fit 256 output neurons.

A: PROJ64    — 64 output neurons, random 256×64 projection → logits
B: DIRECT256 — 256 output neurons, charge = logits directly (no projection)
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

H = 512; IN_DIM = 64; IN_K = 13
N_WORKERS = 18; TICKS = 8; INPUT_DURATION = 2
THRESHOLD = 0.00005; EVAL_EVERY = 20
SCHEDULE = ['add','remove','flip','flip','decay','decay','decay','decay']

def build_sdr(n, dim, k, seed):
    rng = np.random.RandomState(seed)
    t = np.zeros((n, dim), np.float32)
    for v in range(n):
        t[v, rng.choice(dim, size=k, replace=False)] = 1.0
    return t

BP_IN = build_sdr(256, IN_DIM, IN_K, 42)

_bp_out=None;_all_data=None;_bigram=None
_pol=None;_freq=None;_phase=None;_rho=None
_out_dim=None;_direct=None

def init_w(bpo,data,bg,pol,fr,ph,rh,outd,direct):
    global _bp_out,_all_data,_bigram,_pol,_freq,_phase,_rho,_out_dim,_direct
    _bp_out=bpo;_all_data=data;_bigram=bg;_pol=pol;_freq=fr;_phase=ph;_rho=rh;_out_dim=outd;_direct=direct

def _get_logits(charge):
    if _direct:
        return charge[H-256:]
    else:
        return np.dot(_bp_out, charge[H-_out_dim:])

def _eval_bigram(mask, theta, decay, seqs):
    sc = SelfWiringGraph.build_sparse_cache(mask)
    total = 0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32)
        s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,decay=decay,
                ticks=TICKS,input_duration=INPUT_DURATION,state=state,charge=charge,
                sparse_cache=sc,polarity=_pol,freq=_freq,phase=_phase,rho=_rho)
            logits=_get_logits(charge)
            e=np.exp(logits-logits.max());pred=e/e.sum()
            tgt=_bigram[tb[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            s+=cos;n+=1
        total+=s/n if n else 0
    return total/len(seqs)

def worker_eval(args):
    mf,theta,decay,seed,pt=args
    rng=random.Random(seed);nrng=np.random.RandomState(seed)
    mask=mf.reshape(H,H);nm=mask;nd=decay
    if pt=='add':
        r=rng.randint(0,H-1);c=rng.randint(0,H-1)
        if r==c or mask[r,c]:return{'delta':-1e9,'type':'add'}
        nm=mask.copy();nm[r,c]=True
    elif pt=='remove':
        alive=list(zip(*np.where(mask)))
        if not alive:return{'delta':-1e9,'type':'remove'}
        r,c=alive[rng.randint(0,len(alive)-1)];nm=mask.copy();nm[r,c]=False
    elif pt=='flip':
        alive=list(zip(*np.where(mask)))
        if not alive:return{'delta':-1e9,'type':'flip'}
        r,c=alive[rng.randint(0,len(alive)-1)];nc=rng.randint(0,H-1)
        if nc==r or nc==c or mask[r,nc]:return{'delta':-1e9,'type':'flip'}
        nm=mask.copy();nm[r,c]=False;nm[r,nc]=True
    elif pt=='decay':
        idx=rng.randint(0,H-1);nd=decay.copy()
        nd[idx]=max(0.01,min(0.5,decay[idx]+rng.uniform(-0.03,0.03)))
    seqs=[]
    for _ in range(2):
        off=nrng.randint(0,len(_all_data)-100)
        seqs.append(_all_data[off:off+100])
    old=_eval_bigram(mask,theta,decay,seqs)
    new=_eval_bigram(nm,theta,nd,seqs)
    return{'delta':float(new-old),'type':pt,
           'new_mask_flat':nm.flatten() if new>old else None,
           'new_decay':nd if pt=='decay' else None}

if __name__ == "__main__":
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    resolve_fineweb_path(); ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram = np.load(os.path.join(BASE_DIR, "data", "bigram_table.npy"))
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[o:o+100] for o in [eval_rng.randint(0, len(ALL_DATA)-100) for _ in range(5)]]

    CONFIGS = [
        ('A', 'PROJ64 (64out+random proj)', 64, False),
        ('B', 'DIRECT256 (256out, 1:1 byte)', 256, True),
    ]

    for mk, label, out_dim, direct in CONFIGS:
        hidden = H - IN_DIM - out_dim
        if direct:
            bp_out = None
        else:
            rng = np.random.RandomState(12345)
            bp_out = rng.randn(256, out_dim).astype(np.float32)
            bp_out /= np.linalg.norm(bp_out, axis=1, keepdims=True)

        print(f"\n{'='*60}")
        print(f"  {mk}: {label}")
        print(f"  H={H}, in={IN_DIM}, out={out_dim}, hidden={hidden}")
        print(f"{'='*60}")

        random.seed(42); np.random.seed(42)
        ref = SelfWiringGraph(max(IN_DIM, 16), hidden=H, projection_scale=1.0)
        pol = ref.polarity.astype(np.float32)

        irng = np.random.RandomState(42)
        mask = (irng.rand(H, H) < 0.05).astype(bool)
        np.fill_diagonal(mask, False)
        theta = np.full(H, 5.0, np.float32)
        decay = np.random.RandomState(99).uniform(0.08, 0.24, H).astype(np.float32)
        print(f"  edges={int(mask.sum())}")

        pool = Pool(N_WORKERS, initializer=init_w,
            initargs=(bp_out, ALL_DATA, bigram, pol, ref.freq, ref.phase, ref.rho, out_dim, direct))

        best_eval=0; stall=0; accepts=0; t0=time.time()
        try:
            for step in range(1, 3001):
                pt = SCHEDULE[(step-1) % len(SCHEDULE)]
                if pt in ('flip','remove','decay') and mask.sum()==0: pt='add'
                args = [(mask.flatten(), theta.copy(), decay.copy(), 1000+step*50+w, pt)
                        for w in range(N_WORKERS)]
                res = pool.map(worker_eval, args)
                br = max(res, key=lambda x: x['delta'])
                if br['delta'] > THRESHOLD:
                    if br['type'] in ('add','remove','flip') and br['new_mask_flat'] is not None:
                        mask[:] = br['new_mask_flat'].reshape(H, H); accepts += 1
                    elif br['type'] == 'decay' and br['new_decay'] is not None:
                        decay[:] = br['new_decay']; accepts += 1

                if step % EVAL_EVERY == 0:
                    el = time.time() - t0
                    ea_list = []
                    for s in eval_seqs:
                        sc = SelfWiringGraph.build_sparse_cache(mask)
                        st=np.zeros(H,np.float32); ch=np.zeros(H,np.float32)
                        cor=0; tot=0
                        for i in range(len(s)-1):
                            inj=np.zeros(H,np.float32); inj[0:IN_DIM]=BP_IN[s[i]]
                            st,ch=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,decay=decay,
                                ticks=TICKS,input_duration=INPUT_DURATION,state=st,charge=ch,
                                sparse_cache=sc,polarity=pol,freq=ref.freq,phase=ref.phase,rho=ref.rho)
                            if direct: logits=ch[H-256:]
                            else: logits=np.dot(bp_out, ch[H-out_dim:])
                            if np.argmax(logits)==s[i+1]: cor+=1
                            tot+=1
                        ea_list.append(cor/tot if tot else 0)
                    ea = np.mean(ea_list)
                    if ea > best_eval: best_eval=ea; stall=0
                    else: stall += EVAL_EVERY
                    print(f"  [{mk}][{step:4d}] eval={ea*100:.1f}% best={best_eval*100:.1f}% "
                          f"edges={int(mask.sum())} acc={accepts} {el:.0f}s ({step/el:.1f}sps)")
                    sys.stdout.flush()
                    if stall >= 400 and step >= 300:
                        print(f"  ** STALL at step {step}"); break
        finally:
            pool.terminate(); pool.join()

        print(f"  [{mk}] DONE: best={best_eval*100:.1f}% {time.time()-t0:.0f}s")

    print(f"\n{'='*60}")
    print(f"  DIRECT OUTPUT: 256 neurons 1:1 vs 64 neurons + projection")
    print(f"{'='*60}")
