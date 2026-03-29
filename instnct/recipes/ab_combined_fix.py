"""
COMBINED: new schedule + fixed channel bug
===========================================
The channel was using stale global _channel in eval.
Fix: pass channel through to _eval_bigram directly.

Schedule: add, enhance, reverse, mirror, flip, theta, channel, channel, remove
(2x channel budget since it was broken before)
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
WAVE_LUT = SelfWiringGraph.WAVE_LUT

SCHEDULE = ['add','enhance','reverse','mirror','flip','theta','channel','channel','remove']

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
_bp_out=None;_all_data=None;_bigram=None;_pol=None;_pol_bool=None

def init_w(bpo,data,bg,pol,pb):
    global _bp_out,_all_data,_bigram,_pol,_pol_bool
    _bp_out=bpo;_all_data=data;_bigram=bg;_pol=pol;_pol_bool=pb

def _eval_bigram(mask, theta, polarity_f, channel, seqs):
    """FIX: channel passed as argument, not global."""
    sc = SelfWiringGraph.build_sparse_cache(mask); total = 0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32);s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                state=state,charge=charge,sparse_cache=sc,polarity=polarity_f,
                channel=channel)
            logits=np.dot(_bp_out,charge[H-OUT_DIM:])
            e=np.exp(logits-logits.max());pred=e/e.sum()
            tgt=_bigram[tb[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            s+=cos;n+=1
        total+=s/n if n else 0
    return total/len(seqs)

def worker_eval(args):
    mf,theta,pol_bool,pol_f,channel,seed,pt=args
    rng=random.Random(seed);nrng=np.random.RandomState(seed)
    mask=mf.reshape(H,H);nm=mask;nt=theta;nc=channel
    npb=pol_bool;npf=pol_f

    if pt=='add':
        r=rng.randint(0,H-1);c=rng.randint(0,H-1)
        if r==c or mask[r,c]:return{'delta':-1e9,'type':'add'}
        nm=mask.copy();nm[r,c]=True
    elif pt=='enhance':
        in_deg=mask.sum(axis=0).astype(np.float64)+1.0
        top=np.argsort(in_deg)[::-1][:H//4]
        c=int(nrng.choice(top));r=rng.randint(0,H-1)
        if r==c or mask[r,c]:return{'delta':-1e9,'type':'enhance'}
        nm=mask.copy();nm[r,c]=True
    elif pt=='reverse':
        alive=list(zip(*np.where(mask)))
        if len(alive)<1:return{'delta':-1e9,'type':'reverse'}
        r,c=alive[rng.randint(0,len(alive)-1)]
        if r==c or mask[c,r]:return{'delta':-1e9,'type':'reverse'}
        nm=mask.copy();nm[r,c]=False;nm[c,r]=True
    elif pt=='mirror':
        alive=list(zip(*np.where(mask)))
        if len(alive)<1:return{'delta':-1e9,'type':'mirror'}
        r,c=alive[rng.randint(0,len(alive)-1)]
        if mask[c,r]:return{'delta':-1e9,'type':'mirror'}
        nm=mask.copy();nm[c,r]=True
        if pol_bool[r]==pol_bool[c]:
            npb=pol_bool.copy();npb[c]=not npb[c]
            npf=np.where(npb,1.0,-1.0).astype(np.float32)
    elif pt=='flip':
        idx=rng.randint(0,H-1)
        npb=pol_bool.copy();npb[idx]=not npb[idx]
        npf=np.where(npb,1.0,-1.0).astype(np.float32)
    elif pt=='theta':
        idx=rng.randint(0,H-1);nt=theta.copy();nt[idx]=float(rng.randint(1,15))
    elif pt=='channel':
        idx=rng.randint(0,H-1);nc=channel.copy();nc[idx]=np.uint8(rng.randint(1,8))
    elif pt=='remove':
        alive=list(zip(*np.where(mask)))
        if len(alive)<1:return{'delta':-1e9,'type':'remove'}
        r,c=alive[rng.randint(0,len(alive)-1)];nm=mask.copy();nm[r,c]=False

    seqs=[]
    for _ in range(2):
        off=nrng.randint(0,len(_all_data)-100);seqs.append(_all_data[off:off+100])
    # FIX: pass channel to eval, not global
    old=_eval_bigram(mask,theta,pol_f,channel,seqs)
    new=_eval_bigram(nm,nt,npf,nc,seqs)
    return{'delta':float(new-old),'type':pt,
           'new_mask_flat':nm.flatten() if new>old else None,
           'new_theta':nt if pt=='theta' else None,
           'new_channel':nc if pt=='channel' else None,
           'new_pol_bool':npb if pt in ('flip','mirror') else None,
           'new_pol_f':npf if pt in ('flip','mirror') else None}

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
    print(f"  COMBINED: new schedule + FIXED channel")
    print(f"  baselines: old sched=13.5%, new sched(broken ch)=20.8%")
    print(f"  channel peak (old sched)=23.8%")
    print(f"  schedule: {SCHEDULE}")
    print(f"{'='*60}")
    sys.stdout.flush()

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
    pol_f32 = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)
    pol_bool = ref.polarity.copy()
    irng = np.random.RandomState(42)
    mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask, False)
    theta = np.full(H, 1.0, np.float32)
    nrng = np.random.RandomState(42)
    channel = nrng.randint(1, 9, size=H).astype(np.uint8)

    init_w(bp_out, ALL_DATA, bigram, pol_f32, pol_bool)
    pool = Pool(N_WORKERS, initializer=init_w,
        initargs=(bp_out, ALL_DATA, bigram, pol_f32, pol_bool))

    best=0;acc=0;t0=time.time()
    op_counts = {op: 0 for op in set(SCHEDULE)}
    for step in range(1, 2001):
        pt=SCHEDULE[(step-1)%len(SCHEDULE)]
        if pt in SCHEDULE and mask.sum()==0:pt='add'
        args=[(mask.flatten(),theta.copy(),pol_bool.copy(),pol_f32.copy(),
               channel.copy(),1000+step*50+w,pt) for w in range(N_WORKERS)]
        res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])
        if br['delta']>THRESHOLD:
            if br['new_mask_flat'] is not None:
                mask[:]=br['new_mask_flat'].reshape(H,H)
            if br['new_pol_bool'] is not None:
                pol_bool[:]=br['new_pol_bool'];pol_f32[:]=br['new_pol_f']
            if br['new_theta'] is not None:
                theta[:]=br['new_theta']
            if br['new_channel'] is not None:
                channel[:]=br['new_channel']
            acc+=1;op_counts[br['type']]+=1
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
                        channel=channel)
                    logits=np.dot(bp_out,ch[H-OUT_DIM:])
                    if np.argmax(logits)==s[i+1]:cor+=1
                    tot+=1
                ea_list.append(cor/tot if tot else 0)
            ea=np.mean(ea_list)
            if ea>best:best=ea
            if step%200==0:
                recip=(mask & mask.T).sum()//2
                inh_pct=(~pol_bool).sum()/H*100
                print(f"  [{step:4d}] eval={ea*100:.1f}% best={best*100:.1f}% th={theta.mean():.1f} "
                      f"e={mask.sum()} recip={recip} inh={inh_pct:.0f}% "
                      f"ops={dict(op_counts)} {time.time()-t0:.0f}s")
                sys.stdout.flush()
    pool.terminate();pool.join()
    elapsed=time.time()-t0
    recip=(mask & mask.T).sum()//2;inh_pct=(~pol_bool).sum()/H*100
    print(f"\n  >> DONE: best={best*100:.1f}% theta={theta.mean():.1f} {elapsed:.0f}s")
    print(f"  >> edges={mask.sum()} recip={recip} inh={inh_pct:.1f}%")
    print(f"  >> ops: {dict(op_counts)}")
    print(f"  >> baselines: old=13.5%, new(broken ch)=20.8%, channel only=23.8%")
