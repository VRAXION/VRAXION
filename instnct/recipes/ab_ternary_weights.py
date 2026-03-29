"""
Ternary weights: per-connection polarity (+1/-1/0)
===================================================
Replace mask(bool) + polarity(bool per neuron) with
ternary connection matrix: +1 (exc), -1 (inh), 0 (no connection).

Uses the new schedule with ternary-native mutations.
Baseline: old schedule 13.5%, new schedule 20.8%

A: new schedule with per-NEURON polarity (reproduce 20.8%)
B: new schedule with per-CONNECTION ternary weights
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

SCHEDULE = ['add','enhance','reverse','mirror','flip_sign','theta','channel','channel','remove']

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

# Ternary rollout: uses int8 weight matrix directly
def rollout_ternary(injected, weights, theta, channel, ticks, input_duration, state, charge):
    """Forward pass using ternary weight matrix (+1/-1/0)."""
    H = weights.shape[0]
    act = state.copy() if state is not None else np.zeros(H, np.float32)
    cur_charge = charge.copy() if charge is not None else np.zeros(H, np.float32)

    for tick in range(int(ticks)):
        if tick % 6 == 0:
            cur_charge = np.maximum(cur_charge - 1.0, 0.0)
        if tick < int(input_duration):
            act = act + injected
        # Propagate: act @ weights (ternary matmul = just adds/subs)
        raw = act @ weights.astype(np.float32)
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        cur_charge += raw
        np.clip(cur_charge, 0.0, 15.0, out=cur_charge)
        # Spike with channel LUT
        theta_mult = WAVE_LUT[channel, tick % 8]
        eff = np.clip(theta * theta_mult, 1.0, 15.0)
        fired = cur_charge >= eff
        act = fired.astype(np.float32)  # no polarity multiply — it's IN the weights
        cur_charge[fired] = 0.0
    return act, cur_charge

_bp_out=None;_all_data=None;_bigram=None;_channel=None;_mode=None

def init_w(bpo,data,bg,ch,mode):
    global _bp_out,_all_data,_bigram,_channel,_mode
    _bp_out=bpo;_all_data=data;_bigram=bg;_channel=ch;_mode=mode

def _eval_ternary(weights, theta, seqs):
    total = 0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32);s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=rollout_ternary(inj,weights,theta,_channel,
                TICKS,INPUT_DURATION,state,charge)
            logits=np.dot(_bp_out,charge[H-OUT_DIM:])
            e=np.exp(logits-logits.max());pred=e/e.sum()
            tgt=_bigram[tb[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            s+=cos;n+=1
        total+=s/n if n else 0
    return total/len(seqs)

def _eval_standard(mask, theta, pol_f, seqs):
    sc = SelfWiringGraph.build_sparse_cache(mask); total = 0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32);s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                state=state,charge=charge,sparse_cache=sc,polarity=pol_f,
                channel=_channel)
            logits=np.dot(_bp_out,charge[H-OUT_DIM:])
            e=np.exp(logits-logits.max());pred=e/e.sum()
            tgt=_bigram[tb[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            s+=cos;n+=1
        total+=s/n if n else 0
    return total/len(seqs)

def worker_eval(args):
    data_flat,theta,channel,seed,pt=args
    rng=random.Random(seed);nrng=np.random.RandomState(seed)

    if _mode == 'ternary':
        weights = data_flat.reshape(H,H)  # int8 ternary
        nw=weights;nt=theta;nc=channel

        if pt=='add':
            r=rng.randint(0,H-1);c=rng.randint(0,H-1)
            if r==c or weights[r,c]!=0:return{'delta':-1e9,'type':'add'}
            nw=weights.copy()
            nw[r,c]=np.int8(1 if rng.random()<0.9 else -1)  # 90% exc, 10% inh

        elif pt=='enhance':
            in_deg=(weights!=0).sum(axis=0).astype(np.float64)+1.0
            top=np.argsort(in_deg)[::-1][:H//4]
            c=int(nrng.choice(top));r=rng.randint(0,H-1)
            if r==c or weights[r,c]!=0:return{'delta':-1e9,'type':'enhance'}
            nw=weights.copy();nw[r,c]=np.int8(1 if rng.random()<0.9 else -1)

        elif pt=='reverse':
            alive=list(zip(*np.where(weights!=0)))
            if len(alive)<1:return{'delta':-1e9,'type':'reverse'}
            r,c=alive[rng.randint(0,len(alive)-1)]
            if r==c or weights[c,r]!=0:return{'delta':-1e9,'type':'reverse'}
            nw=weights.copy();val=nw[r,c];nw[r,c]=0;nw[c,r]=val

        elif pt=='mirror':
            alive=list(zip(*np.where(weights!=0)))
            if len(alive)<1:return{'delta':-1e9,'type':'mirror'}
            r,c=alive[rng.randint(0,len(alive)-1)]
            if weights[c,r]!=0:return{'delta':-1e9,'type':'mirror'}
            nw=weights.copy();nw[c,r]=np.int8(-weights[r,c])  # opposite sign!

        elif pt=='flip_sign':
            alive=list(zip(*np.where(weights!=0)))
            if len(alive)<1:return{'delta':-1e9,'type':'flip_sign'}
            r,c=alive[rng.randint(0,len(alive)-1)]
            nw=weights.copy();nw[r,c]=np.int8(-nw[r,c])  # +1 <-> -1

        elif pt=='theta':
            idx=rng.randint(0,H-1);nt=theta.copy();nt[idx]=float(rng.randint(1,15))

        elif pt=='channel':
            idx=rng.randint(0,H-1);nc=channel.copy();nc[idx]=np.uint8(rng.randint(1,8))

        elif pt=='remove':
            alive=list(zip(*np.where(weights!=0)))
            if len(alive)<1:return{'delta':-1e9,'type':'remove'}
            r,c=alive[rng.randint(0,len(alive)-1)]
            nw=weights.copy();nw[r,c]=0

        seqs=[]
        for _ in range(2):
            off=nrng.randint(0,len(_all_data)-100);seqs.append(_all_data[off:off+100])
        old=_eval_ternary(weights,theta,seqs)
        new=_eval_ternary(nw,nt,seqs)
        return{'delta':float(new-old),'type':pt,
               'new_data_flat':nw.flatten() if new>old else None,
               'new_theta':nt if pt=='theta' else None,
               'new_channel':nc if pt=='channel' else None}

    else:  # standard mode
        mask=data_flat.reshape(H,H).astype(bool)
        # Reconstruct polarity from separate storage
        # (passed via channel trick — not ideal but works for A/B)
        return{'delta':-1e9,'type':pt}  # placeholder

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

    # Only ternary mode (baseline from prior runs)
    print(f"\n{'='*60}")
    print(f"  TERNARY WEIGHTS at H={H}")
    print(f"  baseline: old schedule=13.5%, new schedule=20.8%")
    print(f"  schedule: {SCHEDULE}")
    print(f"{'='*60}")
    sys.stdout.flush()

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
    pol_bool = ref.polarity.copy()

    # Init ternary weights from mask + polarity
    irng = np.random.RandomState(42)
    mask_init = (irng.rand(H,H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask_init, False)
    weights = np.zeros((H,H), dtype=np.int8)
    rows,cols = np.where(mask_init)
    for r,c in zip(rows,cols):
        weights[r,c] = np.int8(1) if pol_bool[r] else np.int8(-1)

    theta = np.full(H, 1.0, np.float32)
    nrng = np.random.RandomState(42)
    channel = nrng.randint(1, 9, size=H).astype(np.uint8)

    edges = (weights!=0).sum()
    exc = (weights==1).sum()
    inh = (weights==-1).sum()
    print(f"  Init: edges={edges} exc={exc} inh={inh} ({inh/max(edges,1)*100:.1f}% inh)")

    init_w(bp_out, ALL_DATA, bigram, channel, 'ternary')
    pool = Pool(N_WORKERS, initializer=init_w,
        initargs=(bp_out, ALL_DATA, bigram, channel, 'ternary'))

    best=0;acc=0;t0=time.time()
    op_counts = {op: 0 for op in SCHEDULE}
    for step in range(1, 2001):
        pt=SCHEDULE[(step-1)%len(SCHEDULE)]
        if pt in SCHEDULE and (weights!=0).sum()==0:pt='add'
        args=[(weights.flatten(),theta.copy(),channel.copy(),
               1000+step*50+w,pt) for w in range(N_WORKERS)]
        res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])
        if br['delta']>THRESHOLD:
            if br['type'] in('add','enhance','reverse','mirror','flip_sign','remove') and br['new_data_flat'] is not None:
                weights[:]=br['new_data_flat'].reshape(H,H);acc+=1;op_counts[br['type']]+=1
            elif br['type']=='theta' and br['new_theta'] is not None:
                theta[:]=br['new_theta'];acc+=1;op_counts['theta']+=1
            elif br['type']=='channel' and br['new_channel'] is not None:
                channel[:]=br['new_channel'];acc+=1;op_counts['channel']+=1
        if step%EVAL_EVERY==0:
            ea_list=[]
            for s in eval_seqs:
                st=np.zeros(H,np.float32);ch=np.zeros(H,np.float32);cor=0;tot=0
                for i in range(len(s)-1):
                    inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
                    st,ch=rollout_ternary(inj,weights,theta,channel,
                        TICKS,INPUT_DURATION,st,ch)
                    logits=np.dot(bp_out,ch[H-OUT_DIM:])
                    if np.argmax(logits)==s[i+1]:cor+=1
                    tot+=1
                ea_list.append(cor/tot if tot else 0)
            ea=np.mean(ea_list)
            if ea>best:best=ea
            if step%200==0:
                edges=(weights!=0).sum();exc=(weights==1).sum();inh=(weights==-1).sum()
                recip=0
                for r,c in zip(*np.where(weights!=0)):
                    if weights[c,r]!=0:recip+=1
                recip//=2
                print(f"  [{step:4d}] eval={ea*100:.1f}% best={best*100:.1f}% th={theta.mean():.1f} "
                      f"e={edges}(+{exc}/-{inh}) recip={recip} ops={dict(op_counts)} {time.time()-t0:.0f}s")
                sys.stdout.flush()
    pool.terminate();pool.join()
    elapsed=time.time()-t0

    edges=(weights!=0).sum();exc=(weights==1).sum();inh=(weights==-1).sum()
    recip=0
    for r,c in zip(*np.where(weights!=0)):
        if weights[c,r]!=0:recip+=1
    recip//=2
    print(f"\n  >> DONE: best={best*100:.1f}% theta={theta.mean():.1f} {elapsed:.0f}s")
    print(f"  >> edges={edges} exc={exc} inh={inh} ({inh/max(edges,1)*100:.1f}% inh)")
    print(f"  >> recip={recip} ({recip*2/max(edges,1)*100:.1f}%)")
    print(f"  >> ops: {dict(op_counts)}")
    print(f"  >> baselines: old schedule=13.5%, new schedule (per-neuron pol)=20.8%")
    print(f"  >> diff vs new schedule: {(best-0.208)*100:+.1f}%")
