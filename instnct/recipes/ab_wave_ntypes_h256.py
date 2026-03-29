"""
Wave type count sweep: 4 vs 6 vs 8 types at H=256
===================================================
All use sin()-derived LUT, binary freq x binary phase combos.
4 types = 2 freq x 2 phase (current winner 21.4%)
6 types = 2 freq x 3 phase
8 types = 2 freq x 4 phase
"""
import sys, os, time, random
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
RHO = 0.3

def make_lut(freq_vals, phase_vals):
    table = []
    for f in freq_vals:
        for p in phase_vals:
            row = [1.0 + RHO * np.sin(t * f + p) for t in range(TICKS)]
            table.append(row)
    return np.array(table, dtype=np.float32)

# 4 types: 2 freq x 2 phase (current winner)
LUT_4 = make_lut([0.5, 2.0], [0, np.pi])
# 6 types: 2 freq x 3 phase (0, 2pi/3, 4pi/3)
LUT_6 = make_lut([0.5, 2.0], [0, 2*np.pi/3, 4*np.pi/3])
# 8 types: 2 freq x 4 phase (0, pi/2, pi, 3pi/2)
LUT_8 = make_lut([0.5, 2.0], [0, np.pi/2, np.pi, 3*np.pi/2])

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
_bp_out=None;_all_data=None;_bigram=None;_pol=None;_wave_table=None;_neuron_types=None

def init_w(bpo,data,bg,pol,wt,nt):
    global _bp_out,_all_data,_bigram,_pol,_wave_table,_neuron_types
    _bp_out=bpo;_all_data=data;_bigram=bg;_pol=pol;_wave_table=wt;_neuron_types=nt

def _rollout_lut(injected, mask, theta, ticks, input_duration, state, charge, sparse_cache, polarity):
    HS = mask.shape[0]
    act = state.copy() if state is not None else np.zeros(HS, np.float32)
    cur_charge = charge.copy() if charge is not None else np.zeros(HS, np.float32)
    use_sparse = len(sparse_cache[0]) < HS * HS * 0.1
    mask_f32 = mask.astype(np.float32)
    n_types = _wave_table.shape[0]
    types = _neuron_types % n_types
    for tick in range(int(ticks)):
        if tick % 6 == 0:
            cur_charge = np.maximum(cur_charge - 1.0, 0.0)
        if tick < int(input_duration):
            act = act + injected
        if use_sparse:
            raw = SelfWiringGraph._sparse_mul_1d_from_cache(HS, act, sparse_cache)
        else:
            raw = act @ mask_f32
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        cur_charge += raw
        np.clip(cur_charge, 0.0, 15.0, out=cur_charge)
        theta_mult = _wave_table[types, tick]
        effective_theta = np.clip(theta * theta_mult, 1.0, 15.0)
        fired = (cur_charge >= effective_theta)
        act = fired.astype(np.float32)
        if polarity is not None:
            act = act * polarity
        cur_charge[fired] = 0.0
    return act, cur_charge

def _eval_bigram(mask, theta, seqs):
    sc = SelfWiringGraph.build_sparse_cache(mask); total = 0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32);s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=_rollout_lut(inj,mask=mask,theta=theta,
                ticks=TICKS,input_duration=INPUT_DURATION,state=state,charge=charge,
                sparse_cache=sc,polarity=_pol)
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

    MODES = [
        ('A', '4 types (2f x 2p)', LUT_4, 2),
        ('B', '6 types (2f x 3p)', LUT_6, 3),
        ('C', '8 types (2f x 4p)', LUT_8, 3),
    ]

    print(f"\n{'='*60}")
    print(f"  WAVE TYPE COUNT: 4 vs 6 vs 8 (H={H})")
    print(f"  ref: 4 types sin binary = 21.4%")
    print(f"{'='*60}")

    for mk, label, lut, bits in MODES:
        n = lut.shape[0]
        print(f"\n  {mk}: {label} ({n} rows, {bits} bit/neuron, {H*bits//8}B)")
        for i in range(n):
            row = ' '.join(f"{'E' if lut[i,t]<0.85 else ('H' if lut[i,t]>1.15 else '~')}" for t in range(TICKS))
            easy = sum(1 for t in range(TICKS) if lut[i,t] < 0.85)
            print(f"    type{i}: {row}  ({easy} easy ticks)")
    sys.stdout.flush()

    results = []
    for mk, label, lut, bits in MODES:
        print(f"\n>> {mk}: {label}")
        sys.stdout.flush()

        random.seed(42); np.random.seed(42)
        ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
        pol_f32 = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)
        irng = np.random.RandomState(42)
        mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask, False)
        theta = np.full(H, 1.0, np.float32)

        nrng = np.random.RandomState(42)
        n_types = lut.shape[0]
        neuron_types = nrng.randint(0, n_types, size=H).astype(np.int32)

        init_w(bp_out, ALL_DATA, bigram, pol_f32, lut, neuron_types)
        pool = Pool(N_WORKERS, initializer=init_w,
            initargs=(bp_out, ALL_DATA, bigram, pol_f32, lut, neuron_types))

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
                        st,ch=_rollout_lut(inj,mask=mask,theta=theta,
                            ticks=TICKS,input_duration=INPUT_DURATION,state=st,charge=ch,
                            sparse_cache=sc,polarity=pol_f32)
                        logits=np.dot(bp_out,ch[H-OUT_DIM:])
                        if np.argmax(logits)==s[i+1]:cor+=1
                        tot+=1
                    ea_list.append(cor/tot if tot else 0)
                ea=np.mean(ea_list)
                if ea>best:best=ea
                if step%400==0:
                    print(f"  [{mk}:{step:4d}] eval={ea*100:.1f}% best={best*100:.1f}% th={theta.mean():.1f} {time.time()-t0:.0f}s")
                    sys.stdout.flush()
        pool.terminate();pool.join()
        elapsed=time.time()-t0
        results.append({'mode':mk,'label':label,'best':float(best),'time':elapsed,
                        'n_types':n_types,'bits':bits})
        print(f"  >> DONE {mk}: best={best*100:.1f}% {elapsed:.0f}s")
        sys.stdout.flush()

    print(f"\n{'='*60}")
    print(f"  TYPE COUNT RESULTS (H={H})")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['mode']}: {r['label']:30s} types={r['n_types']}  bits={r['bits']}  best={r['best']*100:5.1f}%")
    print(f"{'='*60}")
