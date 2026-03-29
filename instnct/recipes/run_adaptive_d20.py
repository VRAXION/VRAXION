"""
Adaptive D20: 10 measure + 90 pump, recalibrate every 100 steps
================================================================
Each cycle:
  Step 1-10: equal round-robin, count accepts per op
  Step 11-100: distribute 90 steps proportional to accept counts
  Repeat 20 cycles = 2000 steps total

Deep telemetry: per-cycle rates, schedule, accuracy, theta, edges.
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
        pb=(pol_f>0);pb[idx]=not pb[idx]
        npf=np.where(pb,1.0,-1.0).astype(np.float32)
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
    return{'delta':float(new-old),'type':pt,
           'new_mask_flat':nm.flatten() if new>old else None,
           'new_theta':nt if pt=='theta' else None,
           'new_channel':nc if pt=='channel' else None,
           'new_pol_f':npf if pt=='flip' else None}

def build_pump_schedule(accept_counts, total_pump_steps=90):
    """Build pump schedule from accept counts, min 1 per op, add first remove last."""
    tickets = {op: max(1, accept_counts.get(op, 0)) for op in ALL_OPS}
    total_tickets = sum(tickets.values())

    # Distribute 90 steps proportionally
    raw = {op: int(total_pump_steps * tickets[op] / total_tickets) for op in ALL_OPS}

    # Fix rounding: distribute remainder to highest ticket ops
    remainder = total_pump_steps - sum(raw.values())
    sorted_ops = sorted(ALL_OPS, key=lambda o: tickets[o], reverse=True)
    for i in range(abs(remainder)):
        if remainder > 0:
            raw[sorted_ops[i % len(sorted_ops)]] += 1
        else:
            raw[sorted_ops[-(i % len(sorted_ops))-1]] -= 1

    # Build schedule: add FIRST, remove LAST, rest shuffled in between
    middle = []
    for op in ALL_OPS:
        if op not in ('add', 'remove'):
            middle.extend([op] * raw[op])
    random.shuffle(middle)

    schedule = ['add'] * raw['add'] + middle + ['remove'] * raw['remove']
    return schedule, tickets

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

    print(f"\n{'='*70}")
    print(f"  ADAPTIVE D20: 10 measure + 90 pump x 20 cycles = 2000 steps")
    print(f"  ops: {ALL_OPS}")
    print(f"  baselines: fixed=23.8%, D20 fixed=10.9%")
    print(f"{'='*70}")
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

    best=0;global_step=0;t0=time.time()
    total_ops = {op: 0 for op in ALL_OPS}
    cycle_log = []
    measure_window = 10  # adaptive: grows if signal too weak
    consecutive_flat = 0

    while global_step < 2000:
        cycle = len(cycle_log) + 1
        cycle_start_best = best

        # Adaptive pump size: 9× the measure window
        pump_window = measure_window * 9

        # === MEASURE PHASE: measure_window steps, round-robin ===
        measure_accepts = {op: 0 for op in ALL_OPS}
        measure_tried = {op: 0 for op in ALL_OPS}
        measure_deltas = {op: 0.0 for op in ALL_OPS}

        for ms in range(measure_window):
            if global_step >= 2000: break
            global_step += 1
            pt = ALL_OPS[ms % len(ALL_OPS)]
            if mask.sum()==0: pt='add'
            measure_tried[pt] += 1

            args=[(mask.flatten(),theta.copy(),channel.copy(),pol_f32.copy(),
                   1000+global_step*50+w,pt) for w in range(N_WORKERS)]
            res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])

            if br['delta']>THRESHOLD:
                if br['new_mask_flat'] is not None: mask[:]=br['new_mask_flat'].reshape(H,H)
                if br['new_theta'] is not None: theta[:]=br['new_theta']
                if br['new_channel'] is not None: channel[:]=br['new_channel']
                if br['new_pol_f'] is not None: pol_f32[:]=br['new_pol_f']
                measure_accepts[br['type']] += 1
                measure_deltas[br['type']] += br['delta']
                total_ops[br['type']] += 1

        # Build pump schedule from measured accepts
        pump_schedule, tickets = build_pump_schedule(measure_accepts, 90)

        # Eval after measure
        ea_list=[]
        for s in eval_seqs:
            sc=SelfWiringGraph.build_sparse_cache(mask)
            st=np.zeros(H,np.float32);ch=np.zeros(H,np.float32);cor=0;tot=0
            for i in range(len(s)-1):
                inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
                st,ch=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                    decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                    state=st,charge=ch,sparse_cache=sc,polarity=pol_f32,channel=channel)
                logits=np.dot(bp_out,ch[H-OUT_DIM:])
                if np.argmax(logits)==s[i+1]:cor+=1
                tot+=1
            ea_list.append(cor/tot if tot else 0)
        ea=np.mean(ea_list)
        if ea>best:best=ea

        # Log measure phase
        rate_str = ' '.join(f"{op}={measure_accepts[op]}/{measure_tried[op]}" for op in ALL_OPS)
        ticket_str = ' '.join(f"{op}={tickets[op]}" for op in ALL_OPS)
        sched_counts = {op: pump_schedule.count(op) for op in ALL_OPS}
        sched_str = ' '.join(f"{op}={sched_counts.get(op,0)}" for op in ALL_OPS)

        print(f"\n  CYCLE {cycle} @ step {global_step} (measure={measure_window}, pump={pump_window})")
        print(f"    MEASURE: {rate_str}")
        print(f"    TICKETS: {ticket_str}")
        print(f"    PUMP SCHED ({pump_window} steps): {sched_str}")
        print(f"    eval={ea*100:.1f}% best={best*100:.1f}% th={theta.mean():.1f} e={mask.sum()}")
        sys.stdout.flush()

        # === PUMP PHASE: pump_window steps with calculated schedule ===
        pump_accepts = {op: 0 for op in ALL_OPS}
        pump_deltas = {op: 0.0 for op in ALL_OPS}

        for ps in range(pump_window):
            if global_step >= 2000: break
            global_step += 1
            pt = pump_schedule[ps % len(pump_schedule)]
            if mask.sum()==0: pt='add'

            args=[(mask.flatten(),theta.copy(),channel.copy(),pol_f32.copy(),
                   1000+global_step*50+w,pt) for w in range(N_WORKERS)]
            res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])

            if br['delta']>THRESHOLD:
                if br['new_mask_flat'] is not None: mask[:]=br['new_mask_flat'].reshape(H,H)
                if br['new_theta'] is not None: theta[:]=br['new_theta']
                if br['new_channel'] is not None: channel[:]=br['new_channel']
                if br['new_pol_f'] is not None: pol_f32[:]=br['new_pol_f']
                pump_accepts[br['type']] += 1
                pump_deltas[br['type']] += br['delta']
                total_ops[br['type']] += 1

        # Eval after pump
        ea_list=[]
        for s in eval_seqs:
            sc=SelfWiringGraph.build_sparse_cache(mask)
            st=np.zeros(H,np.float32);ch=np.zeros(H,np.float32);cor=0;tot=0
            for i in range(len(s)-1):
                inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
                st,ch=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                    decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                    state=st,charge=ch,sparse_cache=sc,polarity=pol_f32,channel=channel)
                logits=np.dot(bp_out,ch[H-OUT_DIM:])
                if np.argmax(logits)==s[i+1]:cor+=1
                tot+=1
            ea_list.append(cor/tot if tot else 0)
        ea=np.mean(ea_list)
        if ea>best:best=ea

        pump_str = ' '.join(f"{op}={pump_accepts[op]}" for op in ALL_OPS)
        improved = best > cycle_start_best

        print(f"    PUMP RESULT: {pump_str}")
        print(f"    eval={ea*100:.1f}% best={best*100:.1f}% {'IMPROVED' if improved else 'FLAT'}")
        print(f"    {time.time()-t0:.0f}s elapsed")
        sys.stdout.flush()

        cycle_log.append({
            'cycle': cycle, 'step': global_step,
            'measure_window': measure_window,
            'measure_accepts': dict(measure_accepts),
            'tickets': dict(tickets),
            'pump_accepts': dict(pump_accepts),
            'best': float(best), 'improved': improved,
            'theta': float(theta.mean()), 'edges': int(mask.sum())
        })

        # === ADAPTIVE WINDOW: scale up if signal too weak ===
        total_measure_accepts = sum(measure_accepts.values())
        if total_measure_accepts < 3:
            consecutive_flat += 1
        else:
            consecutive_flat = 0

        if consecutive_flat >= 2 and measure_window < 1000:
            old_w = measure_window
            measure_window = min(measure_window * 10, 1000)
            print(f"    ** ADAPTIVE: measure window {old_w} -> {measure_window} (weak signal)")
            consecutive_flat = 0
        elif total_measure_accepts >= 5 and measure_window > 10:
            old_w = measure_window
            measure_window = max(measure_window // 10, 10)
            print(f"    ** ADAPTIVE: measure window {old_w} -> {measure_window} (strong signal)")

    pool.terminate();pool.join()
    elapsed=time.time()-t0

    print(f"\n{'='*70}")
    print(f"  ADAPTIVE D20 RESULTS")
    print(f"{'='*70}")
    print(f"  best={best*100:.1f}% theta={theta.mean():.1f} edges={mask.sum()} {elapsed:.0f}s")
    print(f"  total ops: {dict(total_ops)}")
    print(f"  baselines: fixed=23.8%, D20 fixed=10.9%")

    print(f"\n  CYCLE EVOLUTION:")
    print(f"  {'cy':>3s} {'step':>5s} {'best':>5s} {'top_op':>10s} {'tickets':>30s} {'imp':>4s}")
    for c in cycle_log:
        top = max(c['tickets'], key=c['tickets'].get)
        t_str = '/'.join(f"{c['tickets'][op]}" for op in ALL_OPS)
        print(f"  {c['cycle']:3d} {c['step']:5d} {c['best']*100:4.1f}% {top:>10s} {t_str:>30s} {'Y' if c['improved'] else '-'}")
    print(f"{'='*70}")

    # Save checkpoint
    np.savez_compressed(os.path.join(BASE_DIR,"data/adaptive_d20_final.npz"),
        mask=mask,theta=theta,channel=channel,pol_f32=pol_f32)
    print(f"  Saved adaptive_d20_final.npz")
