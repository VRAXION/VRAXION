"""
Hybrid polarity: per-neuron default + per-connection override
==============================================================
- polarity[H] = neuron default (exc/inh)
- override[H×H] = sparse bool (True = this connection's sign is flipped)
- flip: full neuron invert + clear overrides (clean slate)
- flip_sign: toggle ONE connection's override

Schedule: add, enhance, reverse, mirror, flip, flip_sign, theta, remove
No channel (removed — was always 0 accepts)

Baseline: new schedule per-neuron = 20.8%
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

SCHEDULE = ['add','enhance','reverse','mirror','flip','flip_sign','theta','remove']

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

def compute_effective_polarity(mask, polarity, override):
    """Compute actual per-connection sign: polarity XOR override."""
    # For each connection (r,c): sign = polarity[r] XOR override[r,c]
    # polarity True=exc(+1), False=inh(-1)
    # override True=flipped, False=default
    # effective: polarity XOR override → if same=exc(+1), if diff=inh(-1)
    pol_f = np.where(polarity, 1.0, -1.0).astype(np.float32)  # (H,)
    # Broadcast: pol_f[r] for each connection r→c
    pol_matrix = np.tile(pol_f.reshape(-1,1), (1,H))  # (H,H) where [r,:] = polarity[r]
    # Flip where override is True
    pol_matrix[override] *= -1.0
    return pol_matrix  # (H,H) effective sign per connection

def rollout_hybrid(injected, mask, eff_pol, theta, channel, ticks, input_duration, state, charge):
    """Forward pass: mask × effective_polarity gives signed weight matrix."""
    HS = mask.shape[0]
    act = state.copy() if state is not None else np.zeros(HS, np.float32)
    cur_charge = charge.copy() if charge is not None else np.zeros(HS, np.float32)
    # Precompute signed weight matrix: mask * eff_pol
    weights = mask.astype(np.float32) * eff_pol  # (H,H) with +1/-1/0
    for tick in range(int(ticks)):
        if tick % 6 == 0:
            cur_charge = np.maximum(cur_charge - 1.0, 0.0)
        if tick < int(input_duration):
            act = act + injected
        raw = act @ weights
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        cur_charge += raw
        np.clip(cur_charge, 0.0, 15.0, out=cur_charge)
        theta_mult = WAVE_LUT[channel, tick % 8]
        eff = np.clip(theta * theta_mult, 1.0, 15.0)
        fired = cur_charge >= eff
        act = fired.astype(np.float32)  # polarity already in weights
        cur_charge[fired] = 0.0
    return act, cur_charge

_bp_out=None;_all_data=None;_bigram=None;_channel=None

def init_w(bpo,data,bg,ch):
    global _bp_out,_all_data,_bigram,_channel
    _bp_out=bpo;_all_data=data;_bigram=bg;_channel=ch

def _eval_bigram(mask, polarity, override, theta, seqs):
    eff_pol = compute_effective_polarity(mask, polarity, override)
    total = 0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32);s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=rollout_hybrid(inj,mask,eff_pol,theta,_channel,
                TICKS,INPUT_DURATION,state,charge)
            logits=np.dot(_bp_out,charge[H-OUT_DIM:])
            e=np.exp(logits-logits.max());pred=e/e.sum()
            tgt=_bigram[tb[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            s+=cos;n+=1
        total+=s/n if n else 0
    return total/len(seqs)

def worker_eval(args):
    mf,pol_b,ovr_f,theta,channel,seed,pt=args
    rng=random.Random(seed);nrng=np.random.RandomState(seed)
    mask=mf.reshape(H,H);override=ovr_f.reshape(H,H)
    nm=mask;npb=pol_b;novr=override;nt=theta;nc=channel

    if pt=='add':
        r=rng.randint(0,H-1);c=rng.randint(0,H-1)
        if r==c or mask[r,c]:return{'delta':-1e9,'type':'add'}
        nm=mask.copy();nm[r,c]=True
        # New connection: no override (inherits neuron polarity)

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
        novr=override.copy();novr[r,c]=False;novr[c,r]=override[r,c]

    elif pt=='mirror':
        alive=list(zip(*np.where(mask)))
        if len(alive)<1:return{'delta':-1e9,'type':'mirror'}
        r,c=alive[rng.randint(0,len(alive)-1)]
        if mask[c,r]:return{'delta':-1e9,'type':'mirror'}
        nm=mask.copy();nm[c,r]=True
        novr=override.copy();novr[c,r]=True  # mirror = opposite sign

    elif pt=='flip':
        # Full neuron invert + clear overrides
        idx=rng.randint(0,H-1)
        npb=pol_b.copy();npb[idx]=not npb[idx]
        novr=override.copy();novr[idx,:]=False  # clear outgoing overrides

    elif pt=='flip_sign':
        # Toggle ONE connection's override
        alive=list(zip(*np.where(mask)))
        if len(alive)<1:return{'delta':-1e9,'type':'flip_sign'}
        r,c=alive[rng.randint(0,len(alive)-1)]
        novr=override.copy();novr[r,c]=not novr[r,c]

    elif pt=='theta':
        idx=rng.randint(0,H-1);nt=theta.copy();nt[idx]=float(rng.randint(1,15))

    elif pt=='remove':
        alive=list(zip(*np.where(mask)))
        if len(alive)<1:return{'delta':-1e9,'type':'remove'}
        r,c=alive[rng.randint(0,len(alive)-1)]
        nm=mask.copy();nm[r,c]=False
        novr=override.copy();novr[r,c]=False

    seqs=[]
    for _ in range(2):
        off=nrng.randint(0,len(_all_data)-100);seqs.append(_all_data[off:off+100])
    old=_eval_bigram(mask,pol_b,override,theta,seqs)
    new=_eval_bigram(nm,npb,novr,nt,seqs)
    return{'delta':float(new-old),'type':pt,
           'new_mask_flat':nm.flatten() if new>old else None,
           'new_pol':npb if pt=='flip' else None,
           'new_override_flat':novr.flatten() if pt in ('flip','flip_sign','mirror','reverse','remove') else None,
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

    print(f"\n{'='*60}")
    print(f"  HYBRID POLARITY at H={H}")
    print(f"  baseline: new schedule per-neuron=20.8%")
    print(f"  schedule: {SCHEDULE}")
    print(f"  NO channel (removed)")
    print(f"{'='*60}")
    sys.stdout.flush()

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
    pol_bool = ref.polarity.copy()
    irng = np.random.RandomState(42)
    mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask, False)
    override = np.zeros((H,H), dtype=bool)  # no overrides initially
    theta = np.full(H, 1.0, np.float32)
    nrng = np.random.RandomState(42)
    channel = nrng.randint(1, 9, size=H).astype(np.uint8)

    init_w(bp_out, ALL_DATA, bigram, channel)
    pool = Pool(N_WORKERS, initializer=init_w,
        initargs=(bp_out, ALL_DATA, bigram, channel))

    best=0;acc=0;t0=time.time()
    op_counts = {op: 0 for op in SCHEDULE}
    for step in range(1, 2001):
        pt=SCHEDULE[(step-1)%len(SCHEDULE)]
        if pt in SCHEDULE and mask.sum()==0:pt='add'
        args=[(mask.flatten(),pol_bool.copy(),override.flatten(),theta.copy(),
               channel.copy(),1000+step*50+w,pt) for w in range(N_WORKERS)]
        res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])
        if br['delta']>THRESHOLD:
            if br['new_mask_flat'] is not None:
                mask[:]=br['new_mask_flat'].reshape(H,H)
            if br['new_pol'] is not None:
                pol_bool[:]=br['new_pol']
            if br['new_override_flat'] is not None:
                override[:]=br['new_override_flat'].reshape(H,H)
            if br['new_theta'] is not None:
                theta[:]=br['new_theta']
            acc+=1;op_counts[br['type']]+=1
        if step%EVAL_EVERY==0:
            ea_list=[]
            eff_pol=compute_effective_polarity(mask, pol_bool, override)
            for s in eval_seqs:
                st=np.zeros(H,np.float32);ch=np.zeros(H,np.float32);cor=0;tot=0
                for i in range(len(s)-1):
                    inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
                    st,ch=rollout_hybrid(inj,mask,eff_pol,theta,channel,
                        TICKS,INPUT_DURATION,st,ch)
                    logits=np.dot(bp_out,ch[H-OUT_DIM:])
                    if np.argmax(logits)==s[i+1]:cor+=1
                    tot+=1
                ea_list.append(cor/tot if tot else 0)
            ea=np.mean(ea_list)
            if ea>best:best=ea
            if step%200==0:
                n_ovr=override[mask].sum()
                n_edges=mask.sum()
                inh_n=(~pol_bool).sum()
                recip=(mask & mask.T).sum()//2
                print(f"  [{step:4d}] eval={ea*100:.1f}% best={best*100:.1f}% th={theta.mean():.1f} "
                      f"e={n_edges} ovr={n_ovr} inh_n={inh_n} recip={recip} "
                      f"ops={dict(op_counts)} {time.time()-t0:.0f}s")
                sys.stdout.flush()
    pool.terminate();pool.join()
    elapsed=time.time()-t0

    n_ovr=override[mask].sum();n_edges=mask.sum()
    inh_n=(~pol_bool).sum();recip=(mask & mask.T).sum()//2
    print(f"\n  >> DONE: best={best*100:.1f}% theta={theta.mean():.1f} {elapsed:.0f}s")
    print(f"  >> edges={n_edges} overrides={n_ovr} ({n_ovr/max(n_edges,1)*100:.1f}% connections flipped)")
    print(f"  >> inh neurons={inh_n} ({inh_n/H*100:.1f}%)")
    print(f"  >> recip={recip} ({recip*2/max(n_edges,1)*100:.1f}%)")
    print(f"  >> ops: {dict(op_counts)}")
    print(f"  >> baseline: new schedule per-neuron=20.8%")
    print(f"  >> diff: {(best-0.208)*100:+.1f}%")
