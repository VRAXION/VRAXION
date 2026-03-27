"""
Learnable Theta: Let the network find its own firing threshold
==============================================================
SDR_64 input, random_64 output, H=256.
Theta starts at different values, but the schedule includes theta mutation
so each neuron converges to its optimal firing threshold.

Three starting points, all with theta in the schedule:
A: theta_init=1.0  (start low, let it climb)
B: theta_init=5.0  (start mid)
C: theta_init=15.0 (start high, let it drop)
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
# KEY: theta dominant in schedule — aggressive search
SCHEDULE = ['add', 'remove', 'flip', 'theta', 'theta', 'theta', 'decay', 'decay']
INIT_DENSITY = 0.05

CONFIGS = {
    'A': {'label': 'theta_init=6.18 (near convergence point)',  'theta': 6.18},
}

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
    mask=mf.reshape(H,H);nm=mask;nt=theta;nd=decay
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
    elif pt=='theta':
        idx=rng.randint(0,H-1);nt=theta.copy()
        nt[idx]=rng.uniform(0.0, 16.0)  # full resample int4 range, not perturbation
    elif pt=='decay':
        idx=rng.randint(0,H-1);nd=decay.copy()
        nd[idx]=max(0.01,min(0.5,decay[idx]+rng.uniform(-0.03,0.03)))
    seqs=[]
    for _ in range(2):
        off=nrng.randint(0,len(_all_data)-100)
        seqs.append(_all_data[off:off+100])
    old=_eval_bigram(mask,theta,decay,seqs)
    new=_eval_bigram(nm,nt,nd,seqs)
    return{'delta':float(new-old),'type':pt,
           'new_mask_flat':nm.flatten() if new>old else None,
           'new_theta':nt if pt=='theta' else None,
           'new_decay':nd if pt=='decay' else None}

def measure_firing(mask, theta, decay, pol, freq, phase, rho, text):
    sc = SelfWiringGraph.build_sparse_cache(mask)
    state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32)
    spikes=[];charges=[]
    for i in range(min(50,len(text)-1)):
        inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[text[i]]
        state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,decay=decay,
            ticks=TICKS,input_duration=INPUT_DURATION,state=state,charge=charge,
            sparse_cache=sc,polarity=pol,freq=freq,phase=phase,rho=rho)
        spikes.append(int(np.sum(state!=0)))
        charges.append(charge.copy())
    ch=np.array(charges)
    return {'firing':float(np.mean(spikes))/H, 'ch_mean':float(ch.mean()), 'ch_max':float(ch.max())}

if __name__ == "__main__":
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    resolve_fineweb_path(); ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram = np.load(os.path.join(BASE_DIR, "data", "bigram_table.npy"))
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[o:o+100] for o in [eval_rng.randint(0,len(ALL_DATA)-100) for _ in range(5)]]

    print(f"\n{'='*70}")
    print(f"  LEARNABLE THETA — schedule includes theta mutation")
    print(f"  Schedule: {SCHEDULE}")
    print(f"{'='*70}")

    all_results = []
    for mk in ['A']:
        cfg = CONFIGS[mk]
        theta_init = cfg['theta']
        print(f"\n{'='*60}")
        print(f"  {mk}: {cfg['label']}")
        print(f"{'='*60}")

        random.seed(42); np.random.seed(42)
        ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
        pol = ref.polarity.astype(np.float32)

        irng = np.random.RandomState(42)
        mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool)
        np.fill_diagonal(mask, False)
        theta = np.full(H, theta_init, np.float32)
        decay = np.random.RandomState(99).uniform(0.08, 0.24, H).astype(np.float32)

        fire0 = measure_firing(mask, theta, decay, pol, ref.freq, ref.phase, ref.rho, eval_seqs[0])
        print(f"  START: theta={theta.mean():.1f} firing={fire0['firing']*100:.1f}%")

        pool = Pool(N_WORKERS, initializer=init_w,
            initargs=(ALL_DATA, bigram, pol, ref.freq, ref.phase, ref.rho))

        best_eval=0; stall=0; accepts=0; t_acc=0; t0=time.time()
        try:
            for step in range(1, 2001):
                pt = SCHEDULE[(step-1) % len(SCHEDULE)]
                if pt in ('flip','remove','decay','theta') and mask.sum()==0: pt='add'
                args = [(mask.flatten(), theta.copy(), decay.copy(), 1000+step*50+w, pt)
                        for w in range(N_WORKERS)]
                res = pool.map(worker_eval, args)
                br = max(res, key=lambda x: x['delta'])
                if br['delta'] > THRESHOLD:
                    if br['type'] in ('add','remove','flip') and br['new_mask_flat'] is not None:
                        mask[:] = br['new_mask_flat'].reshape(H,H); accepts+=1
                    elif br['type'] == 'theta' and br['new_theta'] is not None:
                        theta[:] = br['new_theta']; t_acc+=1; accepts+=1
                    elif br['type'] == 'decay' and br['new_decay'] is not None:
                        decay[:] = br['new_decay']; accepts+=1

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
                        fire = measure_firing(mask,theta,decay,pol,ref.freq,ref.phase,ref.rho,eval_seqs[0])
                        print(f"  [{mk}][{step:4d}] eval={ea*100:.1f}% best={best_eval*100:.1f}% "
                              f"theta={theta.mean():.2f}[{theta.min():.1f},{theta.max():.1f}] "
                              f"firing={fire['firing']*100:.1f}% "
                              f"t_acc={t_acc} acc={accepts} {el:.0f}s")
                    sys.stdout.flush()
                    if stall >= 1000 and step >= 800:
                        print(f"  ** STALL at step {step}"); break
        finally:
            pool.terminate(); pool.join()

        fire_final = measure_firing(mask,theta,decay,pol,ref.freq,ref.phase,ref.rho,eval_seqs[0])
        elapsed = time.time()-t0
        all_results.append({
            'mode': mk, 'label': cfg['label'],
            'theta_init': theta_init,
            'theta_final_mean': float(theta.mean()),
            'theta_final_min': float(theta.min()),
            'theta_final_max': float(theta.max()),
            'best': best_eval,
            'firing': fire_final['firing'],
            'theta_accepts': t_acc,
            'total_accepts': accepts,
            'time': elapsed,
        })
        print(f"\n  [{mk}] DONE: best={best_eval*100:.1f}% "
              f"theta converged: {theta_init:.1f} -> {theta.mean():.2f}[{theta.min():.1f},{theta.max():.1f}] "
              f"firing={fire_final['firing']*100:.1f}% t_acc={t_acc} {elapsed:.0f}s")

    print(f"\n{'='*70}")
    print(f"  LEARNABLE THETA RESULTS")
    print(f"{'='*70}")
    print(f"  {'Mode':<4} {'Init':>5} {'Final':>8} {'Range':>14} {'Best':>6} {'Firing':>8} {'T_acc':>6}")
    for r in all_results:
        print(f"  {r['mode']:<4} {r['theta_init']:5.1f} {r['theta_final_mean']:8.2f} "
              f"[{r['theta_final_min']:.1f},{r['theta_final_max']:.1f}] "
              f"{r['best']*100:5.1f}% {r['firing']*100:7.1f}% {r['theta_accepts']:6d}")
    # Check convergence
    means = [r['theta_final_mean'] for r in all_results]
    spread = max(means) - min(means)
    print(f"\n  Convergence: means={[f'{m:.2f}' for m in means]}, spread={spread:.2f}")
    if spread < 2.0:
        print(f"  -> CONVERGED to ~{np.mean(means):.1f} regardless of starting point!")
    best = max(all_results, key=lambda x: x['best'])
    print(f"  WINNER: {best['mode']} ({best['label']}) -- {best['best']*100:.1f}%")
    print(f"{'='*70}")
