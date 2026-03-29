"""
Learnable channel: each neuron evolves which tick it specializes in
===================================================================
channel = int [0-7], learnable via mutation (like theta).
Triangle wave shape centered on neuron's channel.

A: fixed random channel (bounce 8 tri baseline = 19.0%)
B: learnable channel (in schedule: channel mutation)
C: learnable channel + more budget (2 channel slots)
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

# Precompute: for each peak tick (0-7), the triangle multipliers
WAVE_LUT = np.zeros((8, 8), dtype=np.float32)
for peak in range(8):
    for t in range(8):
        dist = min(abs(t - peak), 8 - abs(t - peak))
        WAVE_LUT[peak, t] = 0.7 + dist * (0.6 / 4.0)

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
_bp_out=None;_all_data=None;_bigram=None;_pol=None;_channels=None;_lut=None

def init_w(bpo,data,bg,pol,ch,lut):
    global _bp_out,_all_data,_bigram,_pol,_channels,_lut
    _bp_out=bpo;_all_data=data;_bigram=bg;_pol=pol;_channels=ch;_lut=lut

def _rollout(injected, mask, theta, channels, ticks, input_duration, state, charge, sparse_cache, polarity):
    HS = mask.shape[0]
    act = state.copy() if state is not None else np.zeros(HS, np.float32)
    cur_charge = charge.copy() if charge is not None else np.zeros(HS, np.float32)
    use_sparse = len(sparse_cache[0]) < HS * HS * 0.1
    mask_f32 = mask.astype(np.float32)
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
        theta_mult = WAVE_LUT[channels, tick]
        effective_theta = np.clip(theta * theta_mult, 1.0, 15.0)
        fired = (cur_charge >= effective_theta)
        act = fired.astype(np.float32)
        if polarity is not None:
            act = act * polarity
        cur_charge[fired] = 0.0
    return act, cur_charge

def _eval_bigram(mask, theta, channels, seqs):
    sc = SelfWiringGraph.build_sparse_cache(mask); total = 0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32);s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=_rollout(inj,mask=mask,theta=theta,channels=channels,
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
    mf,theta,channels,seed,pt=args
    rng=random.Random(seed);nrng=np.random.RandomState(seed)
    mask=mf.reshape(H,H);nm=mask;nt=theta;nc=channels
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
        r,c=alive[rng.randint(0,len(alive)-1)];nc2=rng.randint(0,H-1)
        if nc2==r or nc2==c or mask[r,nc2]:return{'delta':-1e9,'type':'flip'}
        nm=mask.copy();nm[r,c]=False;nm[r,nc2]=True
    elif pt=='theta':
        idx=rng.randint(0,H-1);nt=theta.copy();nt[idx]=float(rng.randint(1,15))
    elif pt=='channel':
        idx=rng.randint(0,H-1);nc=channels.copy();nc[idx]=rng.randint(0,7)
    seqs=[]
    for _ in range(2):
        off=nrng.randint(0,len(_all_data)-100);seqs.append(_all_data[off:off+100])
    old=_eval_bigram(mask,theta,channels,seqs)
    new=_eval_bigram(nm,nt,nc,seqs)
    return{'delta':float(new-old),'type':pt,
           'new_mask_flat':nm.flatten() if new>old else None,
           'new_theta':nt if pt=='theta' else None,
           'new_channels':nc if pt=='channel' else None}

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
        ('A', 'fixed random channel',
         ['add','flip','theta','theta','theta','flip','flip','remove']),
        ('B', 'learnable channel (1 slot)',
         ['add','flip','theta','channel','theta','channel','flip','remove']),
        ('C', 'learnable channel (2 slots)',
         ['add','flip','channel','channel','theta','theta','flip','remove']),
    ]

    print(f"\n{'='*60}")
    print(f"  LEARNABLE CHANNEL at H={H}")
    print(f"  ref: fixed random=19.0%, sin binary=21.4%")
    print(f"{'='*60}")
    print(f"\n  Wave LUT (triangle, peak per channel):")
    for ch in range(8):
        tags = ['E' if WAVE_LUT[ch,t]<0.85 else ('H' if WAVE_LUT[ch,t]>1.15 else '~') for t in range(8)]
        print(f"    ch{ch}: [{' '.join(f'{WAVE_LUT[ch,t]:.2f}{tags[t]}' for t in range(8))}]")
    sys.stdout.flush()

    results = []
    for mk, label, schedule in MODES:
        print(f"\n>> {mk}: {label}")
        print(f"   schedule={schedule}")
        sys.stdout.flush()

        random.seed(42); np.random.seed(42)
        ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
        pol_f32 = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)
        irng = np.random.RandomState(42)
        mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask, False)
        theta = np.full(H, 1.0, np.float32)
        nrng = np.random.RandomState(42)
        channels = nrng.randint(0, 8, size=H).astype(np.int32)

        init_w(bp_out, ALL_DATA, bigram, pol_f32, channels, WAVE_LUT)
        pool = Pool(N_WORKERS, initializer=init_w,
            initargs=(bp_out, ALL_DATA, bigram, pol_f32, channels, WAVE_LUT))

        best=0;acc=0;ch_acc=0;t0=time.time()
        for step in range(1, 2001):
            pt=schedule[(step-1)%len(schedule)]
            if pt in('flip','remove','theta','channel') and mask.sum()==0:pt='add'
            if pt=='channel' and 'channel' not in schedule:pt='theta'
            args=[(mask.flatten(),theta.copy(),channels.copy(),1000+step*50+w,pt) for w in range(N_WORKERS)]
            res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])
            if br['delta']>THRESHOLD:
                if br['type'] in('add','remove','flip') and br['new_mask_flat'] is not None:
                    mask[:]=br['new_mask_flat'].reshape(H,H);acc+=1
                elif br['type']=='theta' and br['new_theta'] is not None:
                    theta[:]=br['new_theta'];acc+=1
                elif br['type']=='channel' and br['new_channels'] is not None:
                    channels[:]=br['new_channels'];ch_acc+=1;acc+=1
            if step%EVAL_EVERY==0:
                ea_list=[]
                for s in eval_seqs:
                    sc=SelfWiringGraph.build_sparse_cache(mask)
                    st=np.zeros(H,np.float32);ch_s=np.zeros(H,np.float32);cor=0;tot=0
                    for i in range(len(s)-1):
                        inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
                        st,ch_s=_rollout(inj,mask=mask,theta=theta,channels=channels,
                            ticks=TICKS,input_duration=INPUT_DURATION,state=st,charge=ch_s,
                            sparse_cache=sc,polarity=pol_f32)
                        logits=np.dot(bp_out,ch_s[H-OUT_DIM:])
                        if np.argmax(logits)==s[i+1]:cor+=1
                        tot+=1
                    ea_list.append(cor/tot if tot else 0)
                ea=np.mean(ea_list)
                if ea>best:best=ea
                if step%200==0:
                    ch_dist = np.bincount(channels, minlength=8)
                    print(f"  [{mk}:{step:4d}] eval={ea*100:.1f}% best={best*100:.1f}% th={theta.mean():.1f} "
                          f"ch_acc={ch_acc} dist={list(ch_dist)} {time.time()-t0:.0f}s")
                    sys.stdout.flush()
        pool.terminate();pool.join()
        elapsed=time.time()-t0

        ch_dist = np.bincount(channels, minlength=8)
        results.append({'mode':mk,'label':label,'best':float(best),'ch_acc':ch_acc,
                        'ch_dist':list(map(int,ch_dist)),'time':elapsed})
        print(f"  >> DONE {mk}: best={best*100:.1f}% ch_acc={ch_acc} dist={list(ch_dist)} {elapsed:.0f}s")
        sys.stdout.flush()

    print(f"\n{'='*60}")
    print(f"  LEARNABLE CHANNEL RESULTS (H={H})")
    print(f"  ref: fixed=19.0%, sin=21.4%")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['mode']}: {r['label']:35s} best={r['best']*100:5.1f}% ch_acc={r['ch_acc']} dist={r['ch_dist']}")
    print(f"{'='*60}")
