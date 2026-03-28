"""
Control Neurons v2: SEPARATE from compute network
==================================================
H=256 compute neurons (unchanged) + 5 extra control neurons.
Control neurons are NOT in the mask — they read aggregate stats
from the compute network and output mutation budgets.

Control inputs (what they "see"):
  - mean charge of compute network
  - edge count / density
  - mean theta
  - mean decay
  - recent accept rate

Control outputs (what they decide):
  charge[0] → add budget
  charge[1] → remove budget
  charge[2] → flip budget
  charge[3] → theta budget
  charge[4] → decay budget

Simple linear control: charge = w . inputs (learnable weights).
Weights mutate alongside the main network.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR)); sys.path.insert(0, str(ROOT_DIR / "model"))
from graph import SelfWiringGraph

H = 256; N_WORKERS = 18; TICKS = 8; INPUT_DURATION = 2
THRESHOLD = 0.00005; EVAL_EVERY = 20; THETA_INIT = 1.0
INIT_DENSITY = 0.05
PHI = (1 + 5**0.5) / 2
IN_DIM = int(round(H / PHI))
OUT_DIM = int(round(H / PHI))
SDR_K = int(round(IN_DIM * 0.20))

N_CTRL = 5
N_CTRL_INPUTS = 5  # mean_charge, density, mean_theta, mean_decay, accept_rate
OP_TYPES = ['add', 'remove', 'flip', 'theta', 'decay']

def build_sdr(n, dim, k, seed):
    rng = np.random.RandomState(seed)
    t = np.zeros((n, dim), np.float32)
    for v in range(n):
        t[v, rng.choice(dim, size=k, replace=False)] = 1.0
    return t

def build_freq_order(dim, bigram, seed=12345):
    freq = bigram.sum(axis=0) + bigram.sum(axis=1)
    rank = np.argsort(freq)[::-1]
    rng = np.random.RandomState(seed)
    p = np.zeros((256, dim), np.float32)
    for i, byte_idx in enumerate(rank):
        t = i / 255.0
        for d in range(dim):
            freq_d = (d + 1) / dim
            p[byte_idx, d] = np.sin(2 * np.pi * t * freq_d * 3) + rng.randn() * 0.3
    p /= np.linalg.norm(p, axis=1, keepdims=True) + 1e-8
    return p.astype(np.float32)

BP_IN = build_sdr(256, IN_DIM, SDR_K, 42)

_bp_out=None;_all_data=None;_bigram=None;_pol=None;_freq=None;_phase=None;_rho=None

def init_w(bpo,data,bg,pol,fr,ph,rh):
    global _bp_out,_all_data,_bigram,_pol,_freq,_phase,_rho
    _bp_out=bpo;_all_data=data;_bigram=bg;_pol=pol;_freq=fr;_phase=ph;_rho=rh

def _eval_bigram(mask, theta, decay, seqs):
    sc = SelfWiringGraph.build_sparse_cache(mask); total = 0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32);s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,decay=decay,
                ticks=TICKS,input_duration=INPUT_DURATION,state=state,charge=charge,
                sparse_cache=sc,polarity=_pol,freq=_freq,phase=_phase,rho=_rho)
            logits=np.dot(_bp_out,charge[H-OUT_DIM:])
            e=np.exp(logits-logits.max());pred=e/e.sum()
            tgt=_bigram[tb[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            s+=cos;n+=1
        total+=s/n if n else 0
    return total/len(seqs)

def worker_eval(args):
    mf,theta,decay,seed,pt=args
    rng=random.Random(seed);nrng=np.random.RandomState(seed)
    mask=mf.reshape(H,H);nm=mask;nt=theta;nd=decay
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
        idx=rng.randint(0,H-1);nt=theta.copy();nt[idx]=rng.uniform(0.0,16.0)
    elif pt=='decay':
        idx=rng.randint(0,H-1);nd=decay.copy()
        nd[idx]=max(0.01,min(0.5,decay[idx]+rng.uniform(-0.03,0.03)))
    seqs=[]
    for _ in range(2):
        off=nrng.randint(0,len(_all_data)-100);seqs.append(_all_data[off:off+100])
    old=_eval_bigram(mask,theta,decay,seqs)
    new=_eval_bigram(nm,nt,nd,seqs)
    return{'delta':float(new-old),'type':pt,
           'new_mask_flat':nm.flatten() if new>old else None,
           'new_theta':nt if pt=='theta' else None,
           'new_decay':nd if pt=='decay' else None}

def compute_ctrl_inputs(mask, theta, decay, recent_accepts, recent_total):
    """Gather aggregate stats for control neurons."""
    edges = int(mask.sum())
    density = edges / (H * H)
    return np.array([
        density * 10,                                    # density (scaled up)
        float(theta.mean()) / 16.0,                     # normalized theta
        float(decay.mean()),                             # mean decay
        recent_accepts / max(recent_total, 1),           # accept rate
        float(np.std(theta)) / 8.0,                     # theta diversity
    ], dtype=np.float32)

def ctrl_decide(ctrl_weights, ctrl_inputs):
    """Control neurons: simple linear + softmax → schedule proportions."""
    # ctrl_weights: (5 ops, 5 inputs)
    raw = ctrl_weights @ ctrl_inputs  # (5,)
    # Softmax → proportions
    raw = raw - raw.max()
    e = np.exp(raw)
    probs = e / (e.sum() + 1e-8)
    # Convert to schedule: 8 slots total
    schedule = []
    for i, op in enumerate(OP_TYPES):
        count = max(0, int(round(probs[i] * 8)))
        schedule.extend([op] * count)
    return schedule if schedule else ['add', 'flip', 'theta']

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT_DIR))
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    from lib.archive import save_experiment
    resolve_fineweb_path(); ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram = np.load(os.path.join(BASE_DIR, "data", "bigram_table.npy"))
    bp_out = build_freq_order(OUT_DIM, bigram)
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[o:o+100] for o in [eval_rng.randint(0,len(ALL_DATA)-100) for _ in range(5)]]

    print(f"\n{'='*60}")
    print(f"  CONTROL NEURONS v2 — separate from compute")
    print(f"  5 control neurons, linear weights, learnable")
    print(f"{'='*60}")

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
    pol = ref.polarity.astype(np.float32)
    irng = np.random.RandomState(42)
    mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask, False)
    theta = np.full(H, THETA_INIT, np.float32)
    decay = np.random.RandomState(99).uniform(0.08, 0.24, H).astype(np.float32)

    # Control weights: (5 ops × 5 inputs) — init with good defaults
    # Rows: add, remove, flip, theta, decay
    # Start biased toward the proven schedule: theta-heavy, some add/flip
    # Pre-converged weights from v2 run (skip learning phase)
    ctrl_weights = np.array([
        [-0.3177, 0.9153, 0.0, -0.2, 0.1],       # add
        [-0.3, 0.2, 0.0, 0.3, -0.1],              # remove
        [0.2, 0.0, 0.1, 0.8469, 0.3],             # flip
        [0.3, 0.2149, 0.3, -0.3, 0.3269],         # theta
        [-0.8536, 0.3, 0.2, 0.0821, -0.1989],     # decay
    ], dtype=np.float32)

    print(f"  edges={int(mask.sum())}")
    print(f"  ctrl_weights shape={ctrl_weights.shape}")

    pool = Pool(N_WORKERS, initializer=init_w,
        initargs=(bp_out, ALL_DATA, bigram, pol, ref.freq, ref.phase, ref.rho))

    best=0;acc=0;t_acc=0;add_a=0;rem_a=0;flip_a=0;dec_a=0;t0=time.time()
    recent_acc_window = 0; recent_total_window = 0
    history = []
    MONITOR_WINDOW = 15
    mut_rng = np.random.RandomState(77)

    # Initial schedule from ctrl
    ctrl_inp = compute_ctrl_inputs(mask, theta, decay, 0, 1)
    schedule = ctrl_decide(ctrl_weights, ctrl_inp)

    try:
        step = 0
        while True:
            step += 1

            # Update schedule from control every 50 steps
            if step % 50 == 1:
                ctrl_inp = compute_ctrl_inputs(mask, theta, decay, recent_acc_window, recent_total_window)
                schedule = ctrl_decide(ctrl_weights, ctrl_inp)
                recent_acc_window = 0; recent_total_window = 0

            # Mutate ctrl_weights occasionally (every 200 steps)
            if step % 200 == 0:
                row = mut_rng.randint(0, N_CTRL)
                col = mut_rng.randint(0, N_CTRL_INPUTS)
                old_w = ctrl_weights[row, col]
                ctrl_weights[row, col] = mut_rng.uniform(-1.0, 1.0)
                # We don't eval the ctrl mutation — just accept (meta-exploration)

            pt = schedule[(step-1) % len(schedule)] if schedule else 'add'
            edges = int(mask.sum())
            if pt in ('flip','remove','decay','theta') and edges == 0:
                pt = 'add'

            args = [(mask.flatten(),theta.copy(),decay.copy(),1000+step*50+w,pt) for w in range(N_WORKERS)]
            res = pool.map(worker_eval, args)
            br = max(res, key=lambda x: x['delta'])
            accepted = False
            if br['delta'] > THRESHOLD:
                if br['type'] in ('add','remove','flip') and br['new_mask_flat'] is not None:
                    mask[:]=br['new_mask_flat'].reshape(H,H); acc+=1; accepted=True
                    if br['type']=='add': add_a+=1
                    elif br['type']=='remove': rem_a+=1
                    else: flip_a+=1
                elif br['type']=='theta' and br['new_theta'] is not None:
                    theta[:]=br['new_theta']; t_acc+=1; acc+=1; accepted=True
                elif br['type']=='decay' and br['new_decay'] is not None:
                    decay[:]=br['new_decay']; dec_a+=1; acc+=1; accepted=True

            recent_total_window += 1
            if accepted: recent_acc_window += 1

            if step % EVAL_EVERY == 0:
                el = time.time()-t0
                ea_list = []
                for s in eval_seqs:
                    sc=SelfWiringGraph.build_sparse_cache(mask)
                    st=np.zeros(H,np.float32);ch=np.zeros(H,np.float32);cor=0;tot=0
                    for i in range(len(s)-1):
                        inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
                        st,ch=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,decay=decay,
                            ticks=TICKS,input_duration=INPUT_DURATION,state=st,charge=ch,
                            sparse_cache=sc,polarity=pol,freq=ref.freq,phase=ref.phase,rho=ref.rho)
                        logits=np.dot(bp_out,ch[H-OUT_DIM:])
                        if np.argmax(logits)==s[i+1]:cor+=1
                        tot+=1
                    ea_list.append(cor/tot if tot else 0)
                ea = np.mean(ea_list)
                if ea > best: best = ea
                edges = int(mask.sum())

                entry = {'step':step,'eval':ea,'edges':edges,
                         'theta_mean':float(theta.mean()),'decay_mean':float(decay.mean())}
                history.append(entry)

                if step % 100 == 0:
                    sched_str = ''.join(s[0].upper() for s in schedule)
                    print(f"  [{step:5d}] eval={ea*100:.1f}% best={best*100:.1f}% "
                          f"edges={edges} theta={theta.mean():.2f} "
                          f"sched=[{sched_str}]({len(schedule)}) "
                          f"[A={add_a} R={rem_a} F={flip_a} T={t_acc} D={dec_a}] "
                          f"{el:.0f}s ({step/el:.1f}sps)")
                sys.stdout.flush()

                # Intelligent stop
                if len(history) >= MONITOR_WINDOW and step >= 400:
                    w = history[-MONITOR_WINDOW:]
                    e_sp = max(h['eval'] for h in w) - min(h['eval'] for h in w)
                    ed_sp = max(h['edges'] for h in w) - min(h['edges'] for h in w)
                    th_sp = max(h['theta_mean'] for h in w) - min(h['theta_mean'] for h in w)
                    dc_sp = max(h['decay_mean'] for h in w) - min(h['decay_mean'] for h in w)
                    all_flat = e_sp < 0.02 and ed_sp < 20 and th_sp < 0.5 and dc_sp < 0.01

                    if step % 200 == 0:
                        print(f"         monitor: eval={e_sp:.3f} edge={ed_sp} "
                              f"theta={th_sp:.3f} decay={dc_sp:.4f} "
                              f"{'-> FLAT' if all_flat else '-> active'}")

                    if all_flat:
                        print(f"  ** INTELLIGENT STOP at step {step}")
                        break

                if step >= 10000:
                    print(f"  ** SAFETY CAP"); break

    finally:
        pool.terminate(); pool.join()

    elapsed = time.time()-t0; edges = int(mask.sum())
    save_experiment(
        name="control_neurons_v2",
        mask=mask, theta=theta, decay=decay, polarity=pol.astype(np.int8),
        bp_out=bp_out,
        config={'H':H,'in_dim':IN_DIM,'out_dim':OUT_DIM,'ctrl_weights':ctrl_weights.tolist()},
        result={'best':round(best,4),'edges':edges,
                'add':add_a,'remove':rem_a,'flip':flip_a,'theta_acc':t_acc,'decay_acc':dec_a,
                'final_ctrl_weights':ctrl_weights.tolist(),
                'final_schedule':list(schedule)},
    )

    print(f"\n{'='*60}")
    print(f"  DONE: best={best*100:.1f}% edges={edges} steps={step}")
    print(f"  [A={add_a} R={rem_a} F={flip_a} T={t_acc} D={dec_a}]")
    print(f"  theta={theta.mean():.2f} decay={decay.mean():.4f}")
    print(f"  Final schedule: {''.join(s[0].upper() for s in schedule)}")
    print(f"  Final ctrl weights:\n{ctrl_weights}")
    print(f"  {elapsed:.0f}s")
    print(f"  Compare: fixed schedule = 22.4%")
    print(f"{'='*60}")
