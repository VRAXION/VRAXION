"""
Quick A/B: 8 ticks vs 16 ticks
================================
Same setup, only ticks differ. Does deeper processing help?
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR)); sys.path.insert(0, str(ROOT_DIR / "model"))
from graph import SelfWiringGraph

H = 256; N_WORKERS = 18; INPUT_DURATION = 2
THRESHOLD = 0.00005; EVAL_EVERY = 20
INIT_DENSITY = 0.05
PHI = (1 + 5**0.5) / 2
IN_DIM = int(round(H / PHI))
OUT_DIM = int(round(H / PHI))
SDR_K = int(round(IN_DIM * 0.20))
SCHEDULE = ['add','flip','theta','channel','theta','channel','flip','remove']

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
_bp_out=None;_all_data=None;_bigram=None;_pol=None;_channel=None;_ticks=None

def init_w(bpo,data,bg,pol,ch,ticks):
    global _bp_out,_all_data,_bigram,_pol,_channel,_ticks
    _bp_out=bpo;_all_data=data;_bigram=bg;_pol=pol;_channel=ch;_ticks=ticks

def _eval_bigram(mask, theta, seqs):
    sc = SelfWiringGraph.build_sparse_cache(mask); total = 0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32);s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                decay=np.float32(0.16),ticks=_ticks,input_duration=INPUT_DURATION,
                state=state,charge=charge,sparse_cache=sc,polarity=_pol,channel=_channel)
            logits=np.dot(_bp_out,charge[H-OUT_DIM:])
            e=np.exp(logits-logits.max());pred=e/e.sum()
            tgt=_bigram[tb[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            s+=cos;n+=1
        total+=s/n if n else 0
    return total/len(seqs)

def worker_eval(args):
    mf,theta,channel,seed,pt=args
    rng=random.Random(seed);nrng=np.random.RandomState(seed)
    mask=mf.reshape(H,H);nm=mask;nt=theta;nc=channel
    if pt=='add':
        r=rng.randint(0,H-1);c=rng.randint(0,H-1)
        if r==c or mask[r,c]:return{'delta':-1e9,'type':'add'}
        nm=mask.copy();nm[r,c]=True
    elif pt=='flip':
        idx=rng.randint(0,H-1)
        pb=(_pol>0);pb[idx]=not pb[idx]
        return{'delta':-1e9,'type':'flip'}  # skip flip for speed
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
    old=_eval_bigram(mask,theta,seqs);new=_eval_bigram(nm,nt,seqs)
    return{'delta':float(new-old),'type':pt,
           'new_mask_flat':nm.flatten() if new>old else None,
           'new_theta':nt if pt=='theta' else None,
           'new_channel':nc if pt=='channel' else None}

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

    MODES = [('A', 8), ('B', 16)]

    print(f"\n{'='*60}")
    print(f"  TICKS A/B: 8 vs 16")
    print(f"{'='*60}")
    sys.stdout.flush()

    results = []
    for mk, ticks in MODES:
        print(f"\n>> {mk}: {ticks} ticks")
        sys.stdout.flush()

        random.seed(42); np.random.seed(42)
        ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
        pol_f32 = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)
        irng = np.random.RandomState(42)
        mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask, False)
        theta = np.full(H, 1.0, np.float32)
        nrng = np.random.RandomState(42)
        channel = nrng.randint(1, 9, size=H).astype(np.uint8)

        init_w(bp_out, ALL_DATA, bigram, pol_f32, channel, ticks)
        pool = Pool(N_WORKERS, initializer=init_w,
            initargs=(bp_out, ALL_DATA, bigram, pol_f32, channel, ticks))

        best=0;t0=time.time()
        for step in range(1, 2001):
            pt=SCHEDULE[(step-1)%len(SCHEDULE)]
            if mask.sum()==0:pt='add'
            args=[(mask.flatten(),theta.copy(),channel.copy(),1000+step*50+w,pt) for w in range(N_WORKERS)]
            res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])
            if br['delta']>THRESHOLD:
                if br['new_mask_flat'] is not None:mask[:]=br['new_mask_flat'].reshape(H,H)
                if br['new_theta'] is not None:theta[:]=br['new_theta']
                if br['new_channel'] is not None:channel[:]=br['new_channel']
            if step%EVAL_EVERY==0:
                ea_list=[]
                for s in eval_seqs:
                    sc=SelfWiringGraph.build_sparse_cache(mask)
                    st=np.zeros(H,np.float32);ch=np.zeros(H,np.float32);cor=0;tot=0
                    for i in range(len(s)-1):
                        inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
                        st,ch=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                            decay=np.float32(0.16),ticks=ticks,input_duration=INPUT_DURATION,
                            state=st,charge=ch,sparse_cache=sc,polarity=pol_f32,channel=channel)
                        logits=np.dot(bp_out,ch[H-OUT_DIM:])
                        if np.argmax(logits)==s[i+1]:cor+=1
                        tot+=1
                    ea_list.append(cor/tot if tot else 0)
                ea=np.mean(ea_list)
                if ea>best:best=ea
                if step%400==0:
                    sps=step/(time.time()-t0)
                    print(f"  [{mk}:{step:4d}] eval={ea*100:.1f}% best={best*100:.1f}% th={theta.mean():.1f} {sps:.1f} sps")
                    sys.stdout.flush()
        pool.terminate();pool.join()
        elapsed=time.time()-t0
        results.append({'ticks':ticks,'best':float(best),'time':elapsed})
        print(f"  >> DONE {mk}: ticks={ticks} best={best*100:.1f}% {elapsed:.0f}s")

    print(f"\n{'='*60}")
    print(f"  TICKS RESULTS")
    print(f"{'='*60}")
    for r in results:
        print(f"  ticks={r['ticks']:2d}: best={r['best']*100:5.1f}% ({r['time']:.0f}s)")
