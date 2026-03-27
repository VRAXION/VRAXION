"""
CRITICAL FIX: State (spike) readout vs Charge readout
=====================================================
The canonical graph.py forward() uses self.readout(self.state) — SPIKES.
All sweep recipes incorrectly used charge for readout.

A: CHARGE readout (what we've been doing — incorrect)
B: STATE readout  (canonical — correct, spike-based)

Both: SDR_64 input, random_64 output, learnable theta, H=256.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

H = 256; IN_DIM = 64; IN_K = 13; OUT_DIM = 64
N_WORKERS = 18; TICKS = 8; INPUT_DURATION = 2
THRESHOLD = 0.00005; EVAL_EVERY = 20
SCHEDULE = ['add', 'remove', 'flip', 'theta', 'theta', 'theta', 'decay', 'decay']
INIT_DENSITY = 0.05; THETA_INIT = 1.0

def build_sdr(n, dim, k, seed):
    rng = np.random.RandomState(seed)
    t = np.zeros((n, dim), np.float32)
    for v in range(n):
        t[v, rng.choice(dim, size=k, replace=False)] = 1.0
    return t

BP_IN = build_sdr(256, IN_DIM, IN_K, 42)
rng_out = np.random.RandomState(12345)
BP_OUT = rng_out.randn(256, OUT_DIM).astype(np.float32)
BP_OUT /= np.linalg.norm(BP_OUT, axis=1, keepdims=True)

_all_data=None;_bigram=None;_pol=None;_freq=None;_phase=None;_rho=None
_use_state=None  # True=state readout, False=charge readout

def init_w(data,bg,pol,fr,ph,rh,use_state):
    global _all_data,_bigram,_pol,_freq,_phase,_rho,_use_state
    _all_data=data;_bigram=bg;_pol=pol;_freq=fr;_phase=ph;_rho=rh;_use_state=use_state

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
            # THE KEY DIFFERENCE:
            readout_vec = state[H-OUT_DIM:] if _use_state else charge[H-OUT_DIM:]
            logits=np.dot(BP_OUT, readout_vec)
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
        if not alive:return{'delta':-1e9,'type':'remove'}
        r,c=alive[rng.randint(0,len(alive)-1)];nm=mask.copy();nm[r,c]=False
    elif pt=='flip':
        alive=list(zip(*np.where(mask)))
        if not alive:return{'delta':-1e9,'type':'flip'}
        r,c=alive[rng.randint(0,len(alive)-1)];nc=rng.randint(0,H-1)
        if nc==r or nc==c or mask[r,nc]:return{'delta':-1e9,'type':'flip'}
        nm=mask.copy();nm[r,c]=False;nm[r,nc]=True
    elif pt=='theta':
        idx=rng.randint(0,H-1);nt=theta.copy()
        nt[idx]=rng.uniform(0.0, 16.0)
    elif pt=='decay':
        idx=rng.randint(0,H-1);nd=decay.copy()
        nd[idx]=max(0.01,min(0.5,decay[idx]+rng.uniform(-0.03,0.03)))
    seqs=[]
    for _ in range(2):
        off=nrng.randint(0,len(_all_data)-100)
        seqs.append(_all_data[off:off+100])
    old=_eval_bigram(mask,theta,decay,seqs)
    new=_eval_bigram(nm,nt,nd,seqs)
    return{'delta':float(new-old),'type':pt,
           'new_mask_flat':nm.flatten() if new>old else None,
           'new_theta':nt if pt=='theta' else None,
           'new_decay':nd if pt=='decay' else None}

if __name__ == "__main__":
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    resolve_fineweb_path(); ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram = np.load(os.path.join(BASE_DIR, "data", "bigram_table.npy"))
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[o:o+100] for o in [eval_rng.randint(0,len(ALL_DATA)-100) for _ in range(5)]]

    MODES = [
        ('A', 'CHARGE readout (incorrect)', False),
        ('B', 'STATE readout (canonical)', True),
    ]

    print(f"\n{'='*70}")
    print(f"  STATE vs CHARGE READOUT — SDR input, learnable theta")
    print(f"{'='*70}")

    all_results = []
    for mk, label, use_state in MODES:
        print(f"\n{'='*60}")
        print(f"  {mk}: {label}")
        print(f"{'='*60}")

        random.seed(42); np.random.seed(42)
        ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
        pol = ref.polarity.astype(np.float32)

        irng = np.random.RandomState(42)
        mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool)
        np.fill_diagonal(mask, False)
        theta = np.full(H, THETA_INIT, np.float32)
        decay = np.random.RandomState(99).uniform(0.08, 0.24, H).astype(np.float32)

        pool = Pool(N_WORKERS, initializer=init_w,
            initargs=(ALL_DATA, bigram, pol, ref.freq, ref.phase, ref.rho, use_state))

        best_eval=0;stall=0;accepts=0;t_acc=0;t0=time.time()
        try:
            for step in range(1, 2001):
                pt = SCHEDULE[(step-1) % len(SCHEDULE)]
                if pt in ('flip','remove','decay','theta') and mask.sum()==0: pt='add'
                args = [(mask.flatten(),theta.copy(),decay.copy(),1000+step*50+w,pt) for w in range(N_WORKERS)]
                res = pool.map(worker_eval, args)
                br = max(res, key=lambda x: x['delta'])
                if br['delta'] > THRESHOLD:
                    if br['type'] in ('add','remove','flip') and br['new_mask_flat'] is not None:
                        mask[:]=br['new_mask_flat'].reshape(H,H); accepts+=1
                    elif br['type']=='theta' and br['new_theta'] is not None:
                        theta[:]=br['new_theta']; t_acc+=1; accepts+=1
                    elif br['type']=='decay' and br['new_decay'] is not None:
                        decay[:]=br['new_decay']; accepts+=1

                if step % EVAL_EVERY == 0:
                    el=time.time()-t0
                    ea_list=[]
                    for s in eval_seqs:
                        sc=SelfWiringGraph.build_sparse_cache(mask)
                        st=np.zeros(H,np.float32);ch=np.zeros(H,np.float32);cor=0;tot=0
                        for i in range(len(s)-1):
                            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
                            st,ch=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,decay=decay,
                                ticks=TICKS,input_duration=INPUT_DURATION,state=st,charge=ch,
                                sparse_cache=sc,polarity=pol,freq=ref.freq,phase=ref.phase,rho=ref.rho)
                            rv = st[H-OUT_DIM:] if use_state else ch[H-OUT_DIM:]
                            logits=np.dot(BP_OUT, rv)
                            if np.argmax(logits)==s[i+1]:cor+=1
                            tot+=1
                        ea_list.append(cor/tot if tot else 0)
                    ea=np.mean(ea_list)
                    if ea>best_eval: best_eval=ea; stall=0
                    else: stall+=EVAL_EVERY

                    if step % 200 == 0:
                        print(f"  [{mk}][{step:4d}] eval={ea*100:.1f}% best={best_eval*100:.1f}% "
                              f"theta={theta.mean():.2f}[{theta.min():.1f},{theta.max():.1f}] "
                              f"t_acc={t_acc} acc={accepts} {el:.0f}s")
                    sys.stdout.flush()
                    if stall >= 800 and step >= 600:
                        print(f"  ** STALL at step {step}"); break
        finally:
            pool.terminate(); pool.join()

        elapsed=time.time()-t0
        all_results.append({'mode':mk,'label':label,'best':best_eval,
            'theta_final':float(theta.mean()),'t_acc':t_acc,'acc':accepts,'time':elapsed})
        print(f"\n  [{mk}] DONE: best={best_eval*100:.1f}% theta={theta.mean():.2f} {elapsed:.0f}s")

    print(f"\n{'='*70}")
    print(f"  STATE vs CHARGE READOUT RESULTS")
    print(f"{'='*70}")
    for r in all_results:
        print(f"  {r['mode']}: {r['label']:35s} best={r['best']*100:5.1f}% theta={r['theta_final']:.2f} {r['time']:.0f}s")
    print(f"{'='*70}")
