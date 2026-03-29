"""
A/B: rho float32 vs int8 quantized
====================================
rho = C19 wave modulation depth.
Currently: all neurons rho=0.3 (constant, never mutated).

A: float32 rho, learnable [0, 1.0] (add rho to schedule)
B: int8 rho [0-15] mapped to [0, 1.0] (16 levels, 0.067 step)
C: fix rho=0.3 (current default, 0 bytes)

Fresh run with latest baked optimizations:
  phi overlap, SDR input, FREQ output, int4 theta, fix decay.
  Deep telemetry: rho distribution, accept rates, accuracy.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool
from pathlib import Path

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

def build_sdr(n, dim, k, seed):
    rng = np.random.RandomState(seed)
    t = np.zeros((n, dim), np.float32)
    for v in range(n):
        t[v, rng.choice(dim, size=k, replace=False)] = 1.0
    return t

def build_freq_order(dim, bigram, seed=12345):
    freq = bigram.sum(axis=0) + bigram.sum(axis=1)
    rank = np.argsort(freq)[::-1]
    rng = np.random.RandomState(seed)
    p = np.zeros((256, dim), np.float32)
    for i, byte_idx in enumerate(rank):
        t = i / 255.0
        for d in range(dim):
            freq_d = (d + 1) / dim
            p[byte_idx, d] = np.sin(2 * np.pi * t * freq_d * 3) + rng.randn() * 0.3
    p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-8
    return p.astype(np.float32)

BP_IN = build_sdr(256, IN_DIM, SDR_K, 42)

_bp_out=None;_all_data=None;_bigram=None;_pol=None;_freq=None;_phase=None;_rho=None
_rho_mode=None  # 'float', 'int8', 'fix'

def init_w(bpo,data,bg,pol,fr,ph,rh,rho_mode):
    global _bp_out,_all_data,_bigram,_pol,_freq,_phase,_rho,_rho_mode
    _bp_out=bpo;_all_data=data;_bigram=bg;_pol=pol;_freq=fr;_phase=ph;_rho=rh;_rho_mode=rho_mode

def _eval_bigram(mask, theta, decay_scalar, rho, seqs):
    sc = SelfWiringGraph.build_sparse_cache(mask); total = 0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32);s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                decay=np.float32(decay_scalar),
                ticks=TICKS,input_duration=INPUT_DURATION,state=state,charge=charge,
                sparse_cache=sc,polarity=_pol,freq=_freq,phase=_phase,rho=rho)
            logits=np.dot(_bp_out,charge[H-OUT_DIM:])
            e=np.exp(logits-logits.max());pred=e/e.sum()
            tgt=_bigram[tb[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            s+=cos;n+=1
        total+=s/n if n else 0
    return total/len(seqs)

def worker_eval(args):
    mf,theta,rho,seed,pt=args
    rng=random.Random(seed);nrng=np.random.RandomState(seed)
    mask=mf.reshape(H,H);nm=mask;nt=theta;nr=rho
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
    elif pt=='rho':
        idx=rng.randint(0,H-1);nr=rho.copy()
        if _rho_mode == 'float':
            nr[idx] = rng.uniform(0.0, 1.0)
        elif _rho_mode == 'int4':
            nr[idx] = float(rng.randint(1, 15))  # raw int [1-15], no normalization
    seqs=[]
    for _ in range(2):
        off=nrng.randint(0,len(_all_data)-100);seqs.append(_all_data[off:off+100])
    old=_eval_bigram(mask,theta,0.16,rho,seqs)
    new=_eval_bigram(nm,nt,0.16,nr,seqs)
    return{'delta':float(new-old),'type':pt,
           'new_mask_flat':nm.flatten() if new>old else None,
           'new_theta':nt if pt=='theta' else None,
           'new_rho':nr if pt=='rho' else None}

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT_DIR))
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    from lib.archive import save_experiment
    resolve_fineweb_path(); ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram = np.load(os.path.join(BASE_DIR, "data", "bigram_table.npy"))
    bp_out = build_freq_order(OUT_DIM, bigram)
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[o:o+100] for o in [eval_rng.randint(0,len(ALL_DATA)-100) for _ in range(5)]]

    MODES = [
        ('A', 'float32 rho [0,1]', 'float',
         ['add','flip','theta','theta','rho','rho','flip','remove']),
        ('B', 'int4 rho [1-15] RAW', 'int4',
         ['add','flip','theta','theta','rho','rho','flip','remove']),
        ('C', 'FIX rho=0.3', 'fix',
         ['add','flip','theta','theta','theta','flip','flip','remove']),
    ]

    print(f"\n{'='*60}")
    print(f"  RHO SWEEP: float32 vs int8 vs fix")
    print(f"{'='*60}")

    results = []
    for mk, label, rho_mode, schedule in MODES:
        print(f"\n  {mk}: {label}  schedule={[''.join(s[0] for s in schedule)]}")

        random.seed(42); np.random.seed(42)
        ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
        pol_f32 = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)
        irng = np.random.RandomState(42)
        mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask, False)
        theta = np.full(H, 1.0, np.float32)  # theta starts at 1, learnable int4
        rho = np.full(H, 0.3, np.float32)    # rho starts at 0.3

        pool = Pool(N_WORKERS, initializer=init_w,
            initargs=(bp_out, ALL_DATA, bigram, pol_f32, ref.freq, ref.phase, rho, rho_mode))

        best=0;stall=0;acc=0;rho_acc=0;t0=time.time()
        for step in range(1, 2001):
            pt = schedule[(step-1) % len(schedule)]
            if pt in ('flip','remove','theta','rho') and mask.sum()==0: pt='add'
            if pt == 'rho' and rho_mode == 'fix': pt = 'flip'  # fix: no rho mutation

            args=[(mask.flatten(),theta.copy(),rho.copy(),1000+step*50+w,pt) for w in range(N_WORKERS)]
            res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])
            if br['delta']>THRESHOLD:
                if br['type'] in('add','remove','flip') and br['new_mask_flat'] is not None:
                    mask[:]=br['new_mask_flat'].reshape(H,H);acc+=1
                elif br['type']=='theta' and br['new_theta'] is not None:
                    theta[:]=br['new_theta'];acc+=1
                elif br['type']=='rho' and br['new_rho'] is not None:
                    rho[:]=br['new_rho'];rho_acc+=1;acc+=1
            if step%EVAL_EVERY==0:
                el=time.time()-t0;ea_list=[]
                for s in eval_seqs:
                    sc=SelfWiringGraph.build_sparse_cache(mask)
                    st=np.zeros(H,np.float32);ch=np.zeros(H,np.float32);cor=0;tot=0
                    for i in range(len(s)-1):
                        inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
                        st,ch=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                            decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                            state=st,charge=ch,sparse_cache=sc,polarity=pol_f32,
                            freq=ref.freq,phase=ref.phase,rho=rho)
                        logits=np.dot(bp_out,ch[H-OUT_DIM:])
                        if np.argmax(logits)==s[i+1]:cor+=1
                        tot+=1
                    ea_list.append(cor/tot if tot else 0)
                ea=np.mean(ea_list)
                if ea>best:best=ea;stall=0
                else:stall+=EVAL_EVERY
                if step%200==0:
                    uniq = len(np.unique(np.round(rho, 2)))
                    print(f"    [{step:4d}] eval={ea*100:.1f}% best={best*100:.1f}% "
                          f"theta={theta.mean():.1f} rho={rho.mean():.3f}[{rho.min():.2f},{rho.max():.2f}] "
                          f"uniq={uniq} rho_acc={rho_acc} acc={acc} {el:.0f}s")
                sys.stdout.flush()
                if stall>=800 and step>=600:print(f"    ** STALL");break
        pool.terminate();pool.join()

        elapsed=time.time()-t0
        # Rho distribution
        if rho_mode != 'fix':
            hist, bins = np.histogram(rho, bins=16, range=(0, 1))
            rho_dist = {f'{bins[i]:.2f}-{bins[i+1]:.2f}': int(hist[i]) for i in range(16) if hist[i] > 0}
        else:
            rho_dist = {'0.30': 256}

        results.append({'mode':mk,'label':label,'best':best,'rho_mean':float(rho.mean()),
                        'rho_std':float(rho.std()),'rho_acc':rho_acc,'acc':acc,
                        'rho_dist':rho_dist,'time':elapsed})
        print(f"    DONE: {label} best={best*100:.1f}% rho={rho.mean():.3f}+-{rho.std():.3f} "
              f"rho_acc={rho_acc} {elapsed:.0f}s")
        if rho_mode != 'fix':
            print(f"    Rho distribution: {rho_dist}")

    print(f"\n{'='*60}")
    print(f"  RHO RESULTS")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['mode']}: {r['label']:25s} best={r['best']*100:5.1f}% "
              f"rho={r['rho_mean']:.3f}+-{r['rho_std']:.3f} rho_acc={r['rho_acc']}")
    print(f"{'='*60}")
