"""
Open run: observe where neuron stability types naturally converge
================================================================
No rules, just measure. H=256 (full scale), 3000 steps.
Track +-, ++, --, 00 distribution over time.
Also track: min/max/mean in-degree per type.
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
SCHEDULE = ['add','flip','theta','channel','theta','channel','flip','remove']

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

def analyze_types(mask, polarity):
    """Full analysis of neuron stability types."""
    H = mask.shape[0]
    in_deg = mask.sum(axis=0)  # incoming connections per neuron
    out_deg = mask.sum(axis=1)  # outgoing connections per neuron

    types = {'+-': [], '++': [], '--': [], '00': []}
    for i in range(H):
        incoming = mask[:, i]
        if not incoming.any():
            types['00'].append(i)
            continue
        sources = np.where(incoming)[0]
        has_exc = polarity[sources].any()
        has_inh = (~polarity[sources]).any()
        if has_exc and has_inh:
            types['+-'].append(i)
        elif has_exc:
            types['++'].append(i)
        else:
            types['--'].append(i)

    counts = {k: len(v) for k, v in types.items()}

    # Degree stats per type
    stats = {}
    for typ, neurons in types.items():
        if neurons:
            degs = in_deg[neurons]
            stats[typ] = f"{len(neurons)}n in={degs.mean():.1f}[{degs.min()}-{degs.max()}]"
        else:
            stats[typ] = "0n"

    # Connection count categories
    low_conn = (in_deg < 2).sum()   # 0 or 1 connection
    med_conn = ((in_deg >= 2) & (in_deg <= 5)).sum()
    high_conn = (in_deg > 5).sum()

    return counts, stats, {'low(<2)': int(low_conn), 'med(2-5)': int(med_conn), 'high(>5)': int(high_conn)}

_bp_out=None;_all_data=None;_bigram=None;_pol=None;_channel=None

def init_w(bpo,data,bg,pol,ch):
    global _bp_out,_all_data,_bigram,_pol,_channel
    _bp_out=bpo;_all_data=data;_bigram=bg;_pol=pol;_channel=ch

def _eval_bigram(mask, theta, seqs):
    sc = SelfWiringGraph.build_sparse_cache(mask); total = 0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32);s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                state=state,charge=charge,sparse_cache=sc,polarity=_pol,
                channel=_channel)
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
        idx=rng.randint(0,H-1);nc=channel.copy();nc[idx]=np.uint8(rng.randint(1,8))
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

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
    pol_f32 = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)
    pol_bool = ref.polarity.copy()
    irng = np.random.RandomState(42)
    mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask, False)
    theta = np.full(H, 1.0, np.float32)
    nrng = np.random.RandomState(42)
    channel = nrng.randint(1, 9, size=H).astype(np.uint8)

    counts, stats, conn = analyze_types(mask, pol_bool)
    print(f"\n{'='*70}")
    print(f"  STABILITY CONVERGENCE OBSERVATION (H={H}, 3000 steps)")
    print(f"{'='*70}")
    print(f"  Init: {counts}")
    print(f"  Init stats: {stats}")
    print(f"  Init connections: {conn}")
    sys.stdout.flush()

    init_w(bp_out, ALL_DATA, bigram, pol_f32, channel)
    pool = Pool(N_WORKERS, initializer=init_w,
        initargs=(bp_out, ALL_DATA, bigram, pol_f32, channel))

    best=0;acc=0;t0=time.time()
    history = []
    for step in range(1, 3001):
        pt=SCHEDULE[(step-1)%len(SCHEDULE)]
        if pt in('flip','remove','theta','channel') and mask.sum()==0:pt='add'
        args=[(mask.flatten(),theta.copy(),channel.copy(),1000+step*50+w,pt) for w in range(N_WORKERS)]
        res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])
        if br['delta']>THRESHOLD:
            if br['type'] in('add','remove','flip') and br['new_mask_flat'] is not None:
                mask[:]=br['new_mask_flat'].reshape(H,H);acc+=1
            elif br['type']=='theta' and br['new_theta'] is not None:
                theta[:]=br['new_theta'];acc+=1
            elif br['type']=='channel' and br['new_channel'] is not None:
                channel[:]=br['new_channel'];acc+=1
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
            if step%300==0:
                counts, stats, conn = analyze_types(mask, pol_bool)
                elapsed = time.time()-t0
                entry = {'step':step,'eval':float(ea),'best':float(best),
                         'counts':counts,'conn':conn,'theta_mean':float(theta.mean())}
                history.append(entry)
                print(f"\n  [{step:4d}] eval={ea*100:.1f}% best={best*100:.1f}% th={theta.mean():.1f} edges={mask.sum()} {elapsed:.0f}s")
                print(f"    Types: {counts}")
                print(f"    Stats: {stats}")
                print(f"    Conns: {conn}")
                sys.stdout.flush()

    pool.terminate();pool.join()
    elapsed=time.time()-t0

    print(f"\n{'='*70}")
    print(f"  FINAL (step 3000, best={best*100:.1f}%, {elapsed:.0f}s)")
    print(f"{'='*70}")
    counts, stats, conn = analyze_types(mask, pol_bool)
    print(f"  Types: {counts}")
    print(f"  Stats: {stats}")
    print(f"  Conns: {conn}")

    # Convergence trend
    print(f"\n  CONVERGENCE TREND:")
    print(f"  {'step':>5s}  {'eval':>6s}  {'+-':>4s}  {'++':>4s}  {'--':>4s}  {'00':>4s}  {'low':>5s}  {'med':>5s}  {'high':>5s}")
    for h in history:
        c = h['counts']
        cn = h['conn']
        print(f"  {h['step']:5d}  {h['eval']*100:5.1f}%  {c['+-']:4d}  {c['++']:4d}  {c['--']:4d}  {c['00']:4d}  {cn['low(<2)']:5d}  {cn['med(2-5)']:5d}  {cn['high(>5)']:5d}")
    sys.stdout.flush()
