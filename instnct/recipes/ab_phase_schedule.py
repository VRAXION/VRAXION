"""
Phase-based adaptive schedule
===============================
100 steps equal (1:1:1 all ops), measure score per op.
Then pump best op to 80% for 200 steps until plateau.
Reset and repeat.

Ops: add, flip, theta, channel, reverse, remove
(6 ops, keep it simple, no enhance/mirror for now)
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

ALL_OPS = ['add','flip','theta','channel','reverse','remove']

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
_bp_out=None;_all_data=None;_bigram=None;_pol=None

def init_w(bpo,data,bg,pol):
    global _bp_out,_all_data,_bigram,_pol
    _bp_out=bpo;_all_data=data;_bigram=bg;_pol=pol

def _eval_bigram(mask, theta, channel, polarity_f, seqs):
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
    mf,theta,channel,pol_f,seed,pt=args
    rng=random.Random(seed);nrng=np.random.RandomState(seed)
    mask=mf.reshape(H,H);nm=mask;nt=theta;nc=channel;npf=pol_f

    if pt=='add':
        r=rng.randint(0,H-1);c=rng.randint(0,H-1)
        if r==c or mask[r,c]:return{'delta':-1e9,'type':'add'}
        nm=mask.copy();nm[r,c]=True
    elif pt=='flip':
        idx=rng.randint(0,H-1)
        pol_bool_local = (pol_f > 0)
        pol_bool_local[idx] = not pol_bool_local[idx]
        npf=np.where(pol_bool_local,1.0,-1.0).astype(np.float32)
    elif pt=='theta':
        idx=rng.randint(0,H-1);nt=theta.copy();nt[idx]=float(rng.randint(1,15))
    elif pt=='channel':
        idx=rng.randint(0,H-1);nc=channel.copy();nc[idx]=np.uint8(rng.randint(1,8))
    elif pt=='reverse':
        alive=list(zip(*np.where(mask)))
        if len(alive)<1:return{'delta':-1e9,'type':'reverse'}
        r,c=alive[rng.randint(0,len(alive)-1)]
        if r==c or mask[c,r]:return{'delta':-1e9,'type':'reverse'}
        nm=mask.copy();nm[r,c]=False;nm[c,r]=True
    elif pt=='remove':
        alive=list(zip(*np.where(mask)))
        if len(alive)<1:return{'delta':-1e9,'type':'remove'}
        r,c=alive[rng.randint(0,len(alive)-1)];nm=mask.copy();nm[r,c]=False

    seqs=[]
    for _ in range(2):
        off=nrng.randint(0,len(_all_data)-100);seqs.append(_all_data[off:off+100])
    old=_eval_bigram(mask,theta,channel,pol_f,seqs)
    new=_eval_bigram(nm,nt,nc,npf,seqs)
    delta=float(new-old)
    return{'delta':delta,'type':pt,
           'new_mask_flat':nm.flatten() if new>old else None,
           'new_theta':nt if pt=='theta' else None,
           'new_channel':nc if pt=='channel' else None,
           'new_pol_f':npf if pt=='flip' else None}

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
    print(f"  PHASE-BASED ADAPTIVE SCHEDULE at H={H}")
    print(f"  ops: {ALL_OPS}")
    print(f"  100 step equal > measure > pump best 80% > plateau > reset")
    print(f"{'='*60}")
    sys.stdout.flush()

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
    pol_f32 = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)
    irng = np.random.RandomState(42)
    mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask, False)
    theta = np.full(H, 1.0, np.float32)
    nrng = np.random.RandomState(42)
    channel = nrng.randint(1, 9, size=H).astype(np.uint8)

    init_w(bp_out, ALL_DATA, bigram, pol_f32)
    pool = Pool(N_WORKERS, initializer=init_w,
        initargs=(bp_out, ALL_DATA, bigram, pol_f32))

    best=0;acc=0;t0=time.time()

    # Phase tracking
    MEASURE_PERIOD = 100
    PUMP_PERIOD = 200
    phase = 'measure'  # 'measure' or 'pump'
    phase_step = 0
    phase_num = 0
    pumped_op = None
    best_at_phase_start = 0

    # Score tracking per op
    op_delta_sum = {op: 0.0 for op in ALL_OPS}
    op_count = {op: 0 for op in ALL_OPS}
    op_accepts = {op: 0 for op in ALL_OPS}
    total_op_accepts = {op: 0 for op in ALL_OPS}

    phase_log = []

    for step in range(1, 3001):
        phase_step += 1

        # Choose op based on phase
        if phase == 'measure':
            # Equal: round-robin
            pt = ALL_OPS[(step-1) % len(ALL_OPS)]
        else:
            # Pump: 80% best op, 20% random others
            if random.random() < 0.8:
                pt = pumped_op
            else:
                others = [o for o in ALL_OPS if o != pumped_op]
                pt = random.choice(others)

        if pt in ALL_OPS and mask.sum()==0: pt='add'

        args=[(mask.flatten(),theta.copy(),channel.copy(),pol_f32.copy(),
               1000+step*50+w,pt) for w in range(N_WORKERS)]
        res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])

        if br['delta']>THRESHOLD:
            if br['new_mask_flat'] is not None:
                mask[:]=br['new_mask_flat'].reshape(H,H)
            if br['new_theta'] is not None:
                theta[:]=br['new_theta']
            if br['new_channel'] is not None:
                channel[:]=br['new_channel']
            if br['new_pol_f'] is not None:
                pol_f32[:]=br['new_pol_f']
            acc+=1
            total_op_accepts[br['type']]+=1
            op_accepts[br['type']]+=1
            op_delta_sum[br['type']]+=br['delta']
        op_count[br['type']]+=1

        # Phase transitions
        if phase == 'measure' and phase_step >= MEASURE_PERIOD:
            # Pick best op by average delta
            scores = {}
            for op in ALL_OPS:
                if op_count[op] > 0:
                    scores[op] = op_delta_sum[op] / op_count[op]
                else:
                    scores[op] = 0
            pumped_op = max(scores, key=scores.get)
            best_at_phase_start = best
            phase = 'pump'
            phase_step = 0
            phase_num += 1
            print(f"  PHASE {phase_num}: measure done → PUMP '{pumped_op}' "
                  f"(scores: {', '.join(f'{o}={scores[o]:.6f}' for o in ALL_OPS)}) "
                  f"best={best*100:.1f}%")
            # Reset counters
            op_delta_sum = {op: 0.0 for op in ALL_OPS}
            op_count = {op: 0 for op in ALL_OPS}
            op_accepts = {op: 0 for op in ALL_OPS}
            sys.stdout.flush()

        elif phase == 'pump' and phase_step >= PUMP_PERIOD:
            # Check if we improved
            improved = best > best_at_phase_start
            phase_log.append({'phase': phase_num, 'op': pumped_op,
                              'improved': improved, 'best': float(best),
                              'accepts': dict(op_accepts)})
            print(f"  PHASE {phase_num}: pump '{pumped_op}' done → "
                  f"{'IMPROVED' if improved else 'PLATEAU'} "
                  f"best={best*100:.1f}% accepts={dict(op_accepts)}")
            # Reset to measure
            phase = 'measure'
            phase_step = 0
            op_delta_sum = {op: 0.0 for op in ALL_OPS}
            op_count = {op: 0 for op in ALL_OPS}
            op_accepts = {op: 0 for op in ALL_OPS}
            sys.stdout.flush()

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
                print(f"  [{step:4d}] eval={ea*100:.1f}% best={best*100:.1f}% th={theta.mean():.1f} "
                      f"phase={phase}({pumped_op if phase=='pump' else 'equal'}) "
                      f"total_ops={dict(total_op_accepts)} {time.time()-t0:.0f}s")
                sys.stdout.flush()

    pool.terminate();pool.join()
    elapsed=time.time()-t0

    print(f"\n{'='*60}")
    print(f"  PHASE SCHEDULE RESULTS")
    print(f"{'='*60}")
    print(f"  best={best*100:.1f}% theta={theta.mean():.1f} {elapsed:.0f}s")
    print(f"  total ops: {dict(total_op_accepts)}")
    print(f"  baselines: fixed schedule=23.8%, new ops=20.8%")
    print(f"\n  PHASE LOG:")
    for p in phase_log:
        print(f"    Phase {p['phase']}: pump '{p['op']}' → "
              f"{'OK' if p['improved'] else 'FLAT'} best={p['best']*100:.1f}%")
