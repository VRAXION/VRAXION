"""
Theta Sweep: Find the firing sweet spot
========================================
SDR_64 input, random_64 output, tentacle I/O, H=256.
Sweep theta_init to find where neurons actually fire.

Theta is the firing threshold — charge must exceed theta to spike.
Too high = dead network (0% firing). Too low = chaos (100% firing).
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

H = 256; IN_DIM = 64; IN_K = 13; OUT_DIM = 64
N_WORKERS = 18; TICKS = 8; INPUT_DURATION = 2
THRESHOLD = 0.00005; EVAL_EVERY = 20
SCHEDULE = ['add','remove','flip','flip','decay','decay','decay','decay']
INIT_DENSITY = 0.05

THETA_VALUES = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

def build_sdr(n, dim, k, seed):
    rng = np.random.RandomState(seed)
    t = np.zeros((n, dim), np.float32)
    for v in range(n):
        t[v, rng.choice(dim, size=k, replace=False)] = 1.0
    return t

BP_IN = build_sdr(256, IN_DIM, IN_K, 42)
rng_out = np.random.RandomState(12345)
BP_OUT = rng_out.randn(256, OUT_DIM).astype(np.float32)
BP_OUT /= np.linalg.norm(BP_OUT, axis=1, keepdims=True)

_all_data=None;_bigram=None;_pol=None;_freq=None;_phase=None;_rho=None

def init_w(data,bg,pol,fr,ph,rh):
    global _all_data,_bigram,_pol,_freq,_phase,_rho
    _all_data=data;_bigram=bg;_pol=pol;_freq=fr;_phase=ph;_rho=rh

def _eval_bigram(mask, theta, decay, seqs):
    sc = SelfWiringGraph.build_sparse_cache(mask)
    total = 0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32)
        s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,decay=decay,
                ticks=TICKS,input_duration=INPUT_DURATION,state=state,charge=charge,
                sparse_cache=sc,polarity=_pol,freq=_freq,phase=_phase,rho=_rho)
            logits=np.dot(BP_OUT,charge[H-OUT_DIM:])
            e=np.exp(logits-logits.max());pred=e/e.sum()
            tgt=_bigram[tb[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            s+=cos;n+=1
        total+=s/n if n else 0
    return total/len(seqs)

def worker_eval(args):
    mf,theta,decay,seed,pt=args
    rng=random.Random(seed);nrng=np.random.RandomState(seed)
    mask=mf.reshape(H,H);nm=mask;nd=decay
    if pt=='add':
        r=rng.randint(0,H-1);c=rng.randint(0,H-1)
        if r==c or mask[r,c]:return{'delta':-1e9,'type':'add'}
        nm=mask.copy();nm[r,c]=True
    elif pt=='remove':
        alive=list(zip(*np.where(mask)))
        if not alive:return{'delta':-1e9,'type':'remove'}
        r,c=alive[rng.randint(0,len(alive)-1)];nm=mask.copy();nm[r,c]=False
    elif pt=='flip':
        alive=list(zip(*np.where(mask)))
        if not alive:return{'delta':-1e9,'type':'flip'}
        r,c=alive[rng.randint(0,len(alive)-1)];nc=rng.randint(0,H-1)
        if nc==r or nc==c or mask[r,nc]:return{'delta':-1e9,'type':'flip'}
        nm=mask.copy();nm[r,c]=False;nm[r,nc]=True
    elif pt=='decay':
        idx=rng.randint(0,H-1);nd=decay.copy()
        nd[idx]=max(0.01,min(0.5,decay[idx]+rng.uniform(-0.03,0.03)))
    seqs=[]
    for _ in range(2):
        off=nrng.randint(0,len(_all_data)-100)
        seqs.append(_all_data[off:off+100])
    old=_eval_bigram(mask,theta,decay,seqs)
    new=_eval_bigram(nm,theta,nd,seqs)
    return{'delta':float(new-old),'type':pt,
           'new_mask_flat':nm.flatten() if new>old else None,
           'new_decay':nd if pt=='decay' else None}

def measure_firing(mask, theta, decay, pol, freq, phase, rho, text):
    """Run one sequence, measure actual firing stats."""
    sc = SelfWiringGraph.build_sparse_cache(mask)
    state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32)
    spike_counts=[]; charge_vals=[]
    for i in range(min(50, len(text)-1)):
        inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[text[i]]
        state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,decay=decay,
            ticks=TICKS,input_duration=INPUT_DURATION,state=state,charge=charge,
            sparse_cache=sc,polarity=pol,freq=freq,phase=phase,rho=rho)
        spike_counts.append(int(np.sum(state != 0)))
        charge_vals.append(charge.copy())
    charges = np.array(charge_vals)
    return {
        'spike_mean': float(np.mean(spike_counts)),
        'firing_rate': float(np.mean(spike_counts)) / H,
        'charge_mean': float(charges.mean()),
        'charge_max': float(charges.max()),
        'charge_nonzero': float((charges > 0.01).mean()),
    }

if __name__ == "__main__":
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    resolve_fineweb_path(); ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram = np.load(os.path.join(BASE_DIR, "data", "bigram_table.npy"))
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[o:o+100] for o in [eval_rng.randint(0, len(ALL_DATA)-100) for _ in range(5)]]

    print(f"\n{'='*70}")
    print(f"  THETA SWEEP — SDR_64 in, random_64 out, H={H}")
    print(f"  Values: {THETA_VALUES}")
    print(f"{'='*70}")

    results = []
    for theta_init in THETA_VALUES:
        print(f"\n--- theta={theta_init} ---")

        random.seed(42); np.random.seed(42)
        ref = SelfWiringGraph(max(IN_DIM, 16), hidden=H, projection_scale=1.0)
        pol = ref.polarity.astype(np.float32)

        irng = np.random.RandomState(42)
        mask = (irng.rand(H, H) < INIT_DENSITY).astype(bool)
        np.fill_diagonal(mask, False)
        theta = np.full(H, theta_init, np.float32)
        decay = np.random.RandomState(99).uniform(0.08, 0.24, H).astype(np.float32)

        # Measure initial firing before any training
        fire0 = measure_firing(mask, theta, decay, pol, ref.freq, ref.phase, ref.rho, eval_seqs[0])
        print(f"  BEFORE: firing={fire0['firing_rate']*100:.1f}% "
              f"charge_mean={fire0['charge_mean']:.3f} charge_max={fire0['charge_max']:.2f} "
              f"nonzero={fire0['charge_nonzero']*100:.1f}%")

        pool = Pool(N_WORKERS, initializer=init_w,
            initargs=(ALL_DATA, bigram, pol, ref.freq, ref.phase, ref.rho))

        best_eval=0; stall=0; accepts=0; t0=time.time()
        try:
            for step in range(1, 601):  # 600 steps max per theta
                pt = SCHEDULE[(step-1) % len(SCHEDULE)]
                if pt in ('flip','remove','decay') and mask.sum()==0: pt='add'
                args = [(mask.flatten(), theta.copy(), decay.copy(), 1000+step*50+w, pt)
                        for w in range(N_WORKERS)]
                res = pool.map(worker_eval, args)
                br = max(res, key=lambda x: x['delta'])
                if br['delta'] > THRESHOLD:
                    if br['type'] in ('add','remove','flip') and br['new_mask_flat'] is not None:
                        mask[:] = br['new_mask_flat'].reshape(H, H); accepts += 1
                    elif br['type'] == 'decay' and br['new_decay'] is not None:
                        decay[:] = br['new_decay']; accepts += 1

                if step % EVAL_EVERY == 0:
                    el = time.time() - t0
                    ea_list = []
                    for s in eval_seqs:
                        sc=SelfWiringGraph.build_sparse_cache(mask)
                        st=np.zeros(H,np.float32);ch=np.zeros(H,np.float32);cor=0;tot=0
                        for i in range(len(s)-1):
                            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
                            st,ch=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,decay=decay,
                                ticks=TICKS,input_duration=INPUT_DURATION,state=st,charge=ch,
                                sparse_cache=sc,polarity=pol,freq=ref.freq,phase=ref.phase,rho=ref.rho)
                            logits=np.dot(BP_OUT,ch[H-OUT_DIM:])
                            if np.argmax(logits)==s[i+1]:cor+=1
                            tot+=1
                        ea_list.append(cor/tot if tot else 0)
                    ea = np.mean(ea_list)
                    if ea > best_eval: best_eval=ea; stall=0
                    else: stall += EVAL_EVERY

                    if step % 100 == 0:
                        fire = measure_firing(mask, theta, decay, pol, ref.freq, ref.phase, ref.rho, eval_seqs[0])
                        print(f"  [{step:4d}] eval={ea*100:.1f}% best={best_eval*100:.1f}% "
                              f"acc={accepts} firing={fire['firing_rate']*100:.1f}% "
                              f"ch_mean={fire['charge_mean']:.3f} ch_max={fire['charge_max']:.2f} "
                              f"{el:.0f}s")
                    sys.stdout.flush()
                    if stall >= 300 and step >= 200:
                        break
        finally:
            pool.terminate(); pool.join()

        fire_final = measure_firing(mask, theta, decay, pol, ref.freq, ref.phase, ref.rho, eval_seqs[0])
        elapsed = time.time() - t0
        results.append({
            'theta': theta_init, 'best': best_eval, 'accepts': accepts,
            'firing': fire_final['firing_rate'],
            'charge_mean': fire_final['charge_mean'],
            'charge_max': fire_final['charge_max'],
            'time': elapsed,
        })
        print(f"  RESULT: theta={theta_init} best={best_eval*100:.1f}% "
              f"firing={fire_final['firing_rate']*100:.1f}% accepts={accepts} {elapsed:.0f}s")

    print(f"\n{'='*70}")
    print(f"  THETA SWEEP RESULTS")
    print(f"{'='*70}")
    print(f"  {'Theta':>6} {'Best':>6} {'Firing':>8} {'ChgMean':>8} {'ChgMax':>7} {'Accept':>7}")
    for r in results:
        print(f"  {r['theta']:6.1f} {r['best']*100:5.1f}% {r['firing']*100:7.1f}% "
              f"{r['charge_mean']:8.3f} {r['charge_max']:7.2f} {r['accepts']:7d}")
    best = max(results, key=lambda x: x['best'])
    print(f"\n  WINNER: theta={best['theta']} -- {best['best']*100:.1f}% "
          f"(firing={best['firing']*100:.1f}%)")
    print(f"{'='*70}")
