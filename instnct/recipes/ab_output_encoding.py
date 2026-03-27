"""
Output Encoding Sweep: Random projection vs SDR readout
========================================================
Input: always SDR_64 (proven best, 7.3%)
Tentacle I/O: first N=input, last M=output

A: RANDOM_OUT  — output 64 neurons, readout via 256×64 random projection (baseline)
B: SDR_OUT     — output 64 neurons, readout via 256×64 SDR table (K=13, 20% sparse)
C: SDR_OUT_32  — output 32 neurons, SDR readout (K=7, ~20% sparse) — more hidden neurons

Adaptive plateau detection.
"""
import sys, os, time, random, json
from collections import deque
import numpy as np
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

H = 256
N_WORKERS = 18
SEQ_LEN = 100
THRESHOLD = 0.00005
TICKS = 8
INPUT_DURATION = 2
THETA_INIT = 5.0
DECAY_LO = 0.08; DECAY_HI = 0.24
INIT_DENSITY = 0.05
EVAL_EVERY = 20
SCHEDULE = ['add', 'remove', 'flip', 'flip', 'decay', 'decay', 'decay', 'decay']
MAX_STEPS = 5000
PLATEAU_WINDOW = 8; PLATEAU_THRESH = 0.5; PLATEAU_STRIKES = 3; PLATEAU_MIN = 300

# Input: always SDR_64
IN_DIM = 64; IN_K = 13

def build_sdr(n_codes, dim, k, seed):
    rng = np.random.RandomState(seed)
    t = np.zeros((n_codes, dim), dtype=np.float32)
    for v in range(n_codes):
        active = rng.choice(dim, size=k, replace=False)
        t[v, active] = 1.0
    return t

def build_random_out(dim, seed=12345):
    rng = np.random.RandomState(seed)
    p = rng.randn(256, dim).astype(np.float32)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p

MODES = {
    'A': {'label': 'RANDOM_OUT_64', 'out_dim': 64, 'out_type': 'random'},
    'B': {'label': 'SDR_OUT_64',    'out_dim': 64, 'out_type': 'sdr', 'out_k': 13},
    'C': {'label': 'SDR_OUT_32',    'out_dim': 32, 'out_type': 'sdr', 'out_k': 7},
}

# Worker globals
_bp_in = None; _bp_out = None; _all_data = None; _bigram = None
_pol = None; _freq = None; _phase = None; _rho = None
_in_dim = None; _out_dim = None

def init_w(bp_in, bp_out, data, bigram, pol, fr, ph, rh, in_dim, out_dim):
    global _bp_in, _bp_out, _all_data, _bigram, _pol, _freq, _phase, _rho, _in_dim, _out_dim
    _bp_in=bp_in; _bp_out=bp_out; _all_data=data; _bigram=bigram
    _pol=pol; _freq=fr; _phase=ph; _rho=rh; _in_dim=in_dim; _out_dim=out_dim

def _eval_bigram(mask, theta, decay, seqs):
    sparse_cache = SelfWiringGraph.build_sparse_cache(mask)
    total = 0.0
    for text_bytes in seqs:
        state = np.zeros(H, np.float32); charge = np.zeros(H, np.float32)
        score = 0.0; n = 0
        for i in range(len(text_bytes)-1):
            inj = np.zeros(H, np.float32)
            inj[0:_in_dim] = _bp_in[text_bytes[i]]
            state, charge = SelfWiringGraph.rollout_token(
                inj, mask=mask, theta=theta, decay=decay,
                ticks=TICKS, input_duration=INPUT_DURATION,
                state=state, charge=charge, sparse_cache=sparse_cache,
                polarity=_pol, freq=_freq, phase=_phase, rho=_rho)
            logits = np.dot(_bp_out, charge[H-_out_dim:])
            e = np.exp(logits - logits.max())
            pred = e / e.sum()
            target = _bigram[text_bytes[i]]
            cos = np.dot(pred, target) / (np.linalg.norm(pred) * np.linalg.norm(target) + 1e-8)
            score += cos; n += 1
        total += score / n if n else 0
    return total / len(seqs)

def worker_eval(args):
    mask_flat, theta, decay, seed, ptype = args
    rng = random.Random(seed); np_rng = np.random.RandomState(seed)
    mask = mask_flat.reshape(H, H)
    new_mask=mask; new_decay=decay
    if ptype == 'add':
        r=rng.randint(0,H-1); c=rng.randint(0,H-1)
        if r==c or mask[r,c]: return {'delta':-1e9,'type':'add'}
        new_mask=mask.copy(); new_mask[r,c]=True
    elif ptype == 'remove':
        alive=list(zip(*np.where(mask)))
        if not alive: return {'delta':-1e9,'type':'remove'}
        r,c=alive[rng.randint(0,len(alive)-1)]
        new_mask=mask.copy(); new_mask[r,c]=False
    elif ptype == 'flip':
        alive=list(zip(*np.where(mask)))
        if not alive: return {'delta':-1e9,'type':'flip'}
        r,c=alive[rng.randint(0,len(alive)-1)]
        nc=rng.randint(0,H-1)
        if nc==r or nc==c or mask[r,nc]: return {'delta':-1e9,'type':'flip'}
        new_mask=mask.copy(); new_mask[r,c]=False; new_mask[r,nc]=True
    elif ptype == 'decay':
        idx=rng.randint(0,H-1); new_decay=decay.copy()
        new_decay[idx]=max(0.01,min(0.5,decay[idx]+rng.uniform(-0.03,0.03)))
    seqs=[]
    for _ in range(2):
        off=np_rng.randint(0,len(_all_data)-100)
        seqs.append(_all_data[off:off+100])
    old=_eval_bigram(mask,theta,decay,seqs)
    new=_eval_bigram(new_mask,theta,new_decay,seqs)
    return {'delta':float(new-old),'type':ptype,
            'new_mask_flat':new_mask.flatten() if new>old else None,
            'new_decay':new_decay if ptype=='decay' else None}

def eval_accuracy(mask, theta, decay, pol, freq, phase, rho, text, bp_in, bp_out, in_dim, out_dim):
    sc = SelfWiringGraph.build_sparse_cache(mask)
    state=np.zeros(H,np.float32); charge=np.zeros(H,np.float32)
    correct=0; total=0
    for i in range(len(text)-1):
        inj=np.zeros(H,np.float32); inj[0:in_dim]=bp_in[text[i]]
        state,charge=SelfWiringGraph.rollout_token(
            inj,mask=mask,theta=theta,decay=decay,ticks=TICKS,input_duration=INPUT_DURATION,
            state=state,charge=charge,sparse_cache=sc,polarity=pol,freq=freq,phase=phase,rho=rho)
        logits=np.dot(bp_out,charge[H-out_dim:])
        if np.argmax(logits)==text[i+1]: correct+=1
        total+=1
    return correct/total if total else 0

def ensure_conn(mask, rng, io_in, io_out):
    from collections import deque
    for _ in range(500):
        vis=set(range(io_in)); q=deque(range(io_in)); tgt=set(range(H-io_out,H)); ok=False
        while q:
            nd=q.popleft()
            for n in np.where(mask[nd])[0]:
                if n in tgt: ok=True; break
                if n not in vis: vis.add(n); q.append(n)
            if ok: break
        if ok: return
        s=rng.randint(io_in,H-io_out); t=rng.randint(io_in,H-io_out)
        if s!=t: mask[s,t]=True

def run_mode(mode_key, ALL_DATA, bigram, eval_seqs):
    cfg = MODES[mode_key]
    out_dim = cfg['out_dim']
    label = cfg['label']
    hidden = H - IN_DIM - out_dim

    # Build input (always SDR)
    bp_in = build_sdr(256, IN_DIM, IN_K, seed=42)
    # Build output
    if cfg['out_type'] == 'sdr':
        bp_out = build_sdr(256, out_dim, cfg['out_k'], seed=99)
    else:
        bp_out = build_random_out(out_dim)

    print(f"\n{'='*60}")
    print(f"  {mode_key}: {label}")
    print(f"  in={IN_DIM}(SDR), out={out_dim}({cfg['out_type']}), hidden={hidden}")
    print(f"{'='*60}")

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(max(IN_DIM, 16), hidden=H, projection_scale=1.0)
    pol = ref.polarity.astype(np.float32)

    init_rng = np.random.RandomState(42)
    mask = (init_rng.rand(H,H) < INIT_DENSITY).astype(bool)
    np.fill_diagonal(mask, False)
    ensure_conn(mask, init_rng, IN_DIM, out_dim)

    theta = np.full(H, THETA_INIT, np.float32)
    decay = np.random.RandomState(99).uniform(DECAY_LO, DECAY_HI, H).astype(np.float32)
    print(f"  Init: {int(mask.sum())} edges")

    pool = Pool(N_WORKERS, initializer=init_w,
                initargs=(bp_in, bp_out, ALL_DATA, bigram, pol, ref.freq, ref.phase, ref.rho, IN_DIM, out_dim))

    add_a=0;rem_a=0;flip_a=0;dec_a=0;accepts=0
    eval_hist=[]; plat_strikes=0; best_eval=0; stall=0; t0=time.time()

    try:
        for step in range(1, MAX_STEPS+1):
            pt = SCHEDULE[(step-1)%len(SCHEDULE)]
            if pt in ('flip','remove','decay') and mask.sum()==0: pt='add'
            args=[(mask.flatten(),theta.copy(),decay.copy(),1000+step*50+w,pt) for w in range(N_WORKERS)]
            results=pool.map(worker_eval,args)
            br=max(results,key=lambda x:x['delta'])
            ok=False
            if br['delta']>THRESHOLD:
                if br['type'] in ('add','remove','flip') and br['new_mask_flat'] is not None:
                    mask[:]=br['new_mask_flat'].reshape(H,H)
                    if br['type']=='add':add_a+=1
                    elif br['type']=='remove':rem_a+=1
                    else:flip_a+=1
                    ok=True
                elif br['type']=='decay' and br['new_decay'] is not None:
                    decay[:]=br['new_decay']; dec_a+=1; ok=True
                if ok: accepts+=1

            if step%EVAL_EVERY==0:
                elapsed=time.time()-t0
                ea=np.mean([eval_accuracy(mask,theta,decay,pol,ref.freq,ref.phase,ref.rho,
                    s,bp_in,bp_out,IN_DIM,out_dim) for s in eval_seqs])
                if ea>best_eval: best_eval=ea; stall=0
                else: stall+=EVAL_EVERY
                eval_hist.append({'step':step,'eval':round(ea*100,2)})
                print(f"  [{mode_key}][{step:4d}] eval={ea*100:.1f}% best={best_eval*100:.1f}% "
                      f"edges={int(mask.sum())} [A={add_a} R={rem_a} F={flip_a} D={dec_a}] "
                      f"{elapsed:.0f}s ({step/elapsed:.1f}sps)")
                sys.stdout.flush()
                if len(eval_hist)>=PLATEAU_WINDOW and step>=PLATEAU_MIN:
                    w=[e['eval'] for e in eval_hist[-PLATEAU_WINDOW:]]
                    if max(w)-min(w)<PLATEAU_THRESH:
                        plat_strikes+=1
                        if plat_strikes>=PLATEAU_STRIKES:
                            print(f"  ** PLATEAU at step {step}"); break
                    else: plat_strikes=0
                if stall>=400 and step>=PLATEAU_MIN:
                    print(f"  ** STALL at step {step}"); break
    finally:
        pool.terminate(); pool.join()

    elapsed=time.time()-t0
    print(f"  [{mode_key}] DONE: best={best_eval*100:.1f}% hidden={hidden} {elapsed:.0f}s")
    return {'mode':mode_key,'label':label,'best':best_eval,'hidden':hidden,
            'steps':eval_hist[-1]['step'] if eval_hist else 0,'time':elapsed}


if __name__ == "__main__":
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    resolve_fineweb_path(); ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB text")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram_path = os.path.join(BASE_DIR, "data", "bigram_table.npy")
    if os.path.exists(bigram_path): bigram = np.load(bigram_path)
    else:
        os.makedirs(os.path.join(BASE_DIR,"data"),exist_ok=True)
        counts=np.zeros((256,256),np.float64)
        for i in range(len(ALL_DATA)-1): counts[ALL_DATA[i],ALL_DATA[i+1]]+=1
        rs=counts.sum(axis=1,keepdims=True); rs[rs==0]=1
        bigram=(counts/rs).astype(np.float32); np.save(bigram_path,bigram)

    eval_rng=np.random.RandomState(9999)
    eval_seqs=[ALL_DATA[off:off+SEQ_LEN] for off in [eval_rng.randint(0,len(ALL_DATA)-SEQ_LEN) for _ in range(5)]]

    results=[]
    for m in ['A','B','C']:
        r=run_mode(m,ALL_DATA,bigram,eval_seqs)
        results.append(r)

    print(f"\n{'='*60}")
    print(f"  OUTPUT ENCODING SWEEP")
    print(f"{'='*60}")
    print(f"  {'Mode':<4} {'Label':<16} {'Hidden':>6} {'Best':>6} {'Steps':>6} {'Time':>6}")
    for r in results:
        print(f"  {r['mode']:<4} {r['label']:<16} {r['hidden']:6d} {r['best']*100:5.1f}% {r['steps']:6d} {r['time']:5.0f}s")
    best=max(results,key=lambda x:x['best'])
    print(f"\n  WINNER: {best['mode']} ({best['label']}) -- {best['best']*100:.1f}%")
    print(f"{'='*60}")
