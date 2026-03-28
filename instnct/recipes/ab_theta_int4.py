"""
A/B: float32 theta vs int4 theta (0-15 integer)
================================================
Same setup: phi overlap, SDR input, FREQ output, H=256.
Deep telemetry: theta distribution, effective_theta, firing rates, per-tick stats.

A: float32 theta [0, 16] continuous (current)
B: int4 theta [0, 15] integer (quantized)
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
SCHEDULE = ['add', 'remove', 'flip', 'theta', 'theta', 'theta', 'decay', 'decay']
INIT_DENSITY = 0.05
PHI = (1 + 5**0.5) / 2
IN_DIM = int(round(H / PHI))
OUT_DIM = int(round(H / PHI))
SDR_K = int(round(IN_DIM * 0.20))

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
_quantize=False

def init_w(bpo,data,bg,pol,fr,ph,rh,quantize):
    global _bp_out,_all_data,_bigram,_pol,_freq,_phase,_rho,_quantize
    _bp_out=bpo;_all_data=data;_bigram=bg;_pol=pol;_freq=fr;_phase=ph;_rho=rh;_quantize=quantize

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
        idx=rng.randint(0,H-1);nt=theta.copy()
        if _quantize:
            nt[idx]=float(rng.randint(0,15))  # int4: 0-15 integer
        else:
            nt[idx]=rng.uniform(0.0,16.0)     # float32: continuous
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

def deep_telemetry(mask, theta, decay, pol, freq, phase, rho, text_bytes):
    """Run 50 tokens, collect per-tick firing stats."""
    sc = SelfWiringGraph.build_sparse_cache(mask)
    state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32)
    all_firing=[]; all_charge=[]; all_eff_theta=[]
    for i in range(min(50,len(text_bytes)-1)):
        inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[text_bytes[i]]
        state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,decay=decay,
            ticks=TICKS,input_duration=INPUT_DURATION,state=state,charge=charge,
            sparse_cache=sc,polarity=pol,freq=freq,phase=phase,rho=rho)
        all_firing.append(float(np.sum(state!=0))/H)
        all_charge.append(float(charge.mean()))
        # Effective theta at last tick
        wave = np.sin(7 * freq + phase)  # tick=7 (last)
        eff = np.clip(theta * (1.0 + rho * wave), 1.0, 15.0)
        all_eff_theta.append(float(eff.mean()))
    return {
        'firing_mean': float(np.mean(all_firing)),
        'firing_std': float(np.std(all_firing)),
        'charge_mean': float(np.mean(all_charge)),
        'eff_theta_mean': float(np.mean(all_eff_theta)),
    }

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

    MODES = [
        ('A', 'FLOAT32 theta [0,16]', False),
        ('B', 'INT4 theta [0,15]', True),
    ]

    results = []
    for mk, label, quantize in MODES:
        print(f"\n{'='*60}")
        print(f"  {mk}: {label}")
        print(f"{'='*60}")

        random.seed(42); np.random.seed(42)
        ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
        pol = ref.polarity.astype(np.float32)
        irng = np.random.RandomState(42)
        mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask, False)
        if quantize:
            theta = np.ones(H, np.float32)  # int4 init: 1
        else:
            theta = np.full(H, THETA_INIT, np.float32)
        decay = np.random.RandomState(99).uniform(0.08, 0.24, H).astype(np.float32)

        pool = Pool(N_WORKERS, initializer=init_w,
            initargs=(bp_out, ALL_DATA, bigram, pol, ref.freq, ref.phase, ref.rho, quantize))

        best=0;stall=0;acc=0;t_acc=0;t0=time.time()
        try:
            for step in range(1, 2001):
                pt=SCHEDULE[(step-1)%len(SCHEDULE)]
                if pt in('flip','remove','decay','theta') and mask.sum()==0:pt='add'
                args=[(mask.flatten(),theta.copy(),decay.copy(),1000+step*50+w,pt) for w in range(N_WORKERS)]
                res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])
                if br['delta']>THRESHOLD:
                    if br['type'] in('add','remove','flip') and br['new_mask_flat'] is not None:
                        mask[:]=br['new_mask_flat'].reshape(H,H);acc+=1
                    elif br['type']=='theta' and br['new_theta'] is not None:
                        theta[:]=br['new_theta'];t_acc+=1;acc+=1
                    elif br['type']=='decay' and br['new_decay'] is not None:
                        decay[:]=br['new_decay'];acc+=1
                if step%EVAL_EVERY==0:
                    el=time.time()-t0;ea_list=[]
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
                    ea=np.mean(ea_list)
                    if ea>best:best=ea;stall=0
                    else:stall+=EVAL_EVERY

                    if step%200==0:
                        tel = deep_telemetry(mask, theta, decay, pol, ref.freq, ref.phase, ref.rho, eval_seqs[0])
                        # Theta distribution
                        unique_vals = len(np.unique(np.round(theta, 1)))
                        theta_hist = np.histogram(theta, bins=16, range=(0,16))[0]
                        peak_bin = np.argmax(theta_hist)

                        print(f"  [{mk}][{step:4d}] eval={ea*100:.1f}% best={best*100:.1f}%")
                        print(f"    theta: mean={theta.mean():.2f} std={theta.std():.2f} "
                              f"min={theta.min():.1f} max={theta.max():.1f} "
                              f"unique={unique_vals} peak_bin={peak_bin}({peak_bin}-{peak_bin+1})")
                        print(f"    firing={tel['firing_mean']*100:.1f}% charge={tel['charge_mean']:.3f} "
                              f"eff_theta={tel['eff_theta_mean']:.2f}")
                        print(f"    acc={acc} t_acc={t_acc} {el:.0f}s ({step/el:.1f}sps)")
                    sys.stdout.flush()
                    if stall>=800 and step>=600:print(f"  ** STALL");break
        finally:pool.terminate();pool.join()

        elapsed=time.time()-t0
        tel_final = deep_telemetry(mask, theta, decay, pol, ref.freq, ref.phase, ref.rho, eval_seqs[0])

        # Theta final distribution
        if quantize:
            vals, counts = np.unique(theta.astype(int), return_counts=True)
            theta_dist = {int(v): int(c) for v, c in zip(vals, counts)}
        else:
            theta_dist = {f'{b}-{b+1}': int(c) for b, c in
                         zip(range(16), np.histogram(theta, bins=16, range=(0,16))[0]) if c > 0}

        results.append({'mode':mk,'label':label,'best':best,'quantize':quantize,
                        'theta_mean':float(theta.mean()),'theta_std':float(theta.std()),
                        'theta_dist':theta_dist,'tel':tel_final,
                        'acc':acc,'t_acc':t_acc,'time':elapsed})

        print(f"\n  [{mk}] DONE: best={best*100:.1f}% theta={theta.mean():.2f}+-{theta.std():.2f}")
        print(f"    firing={tel_final['firing_mean']*100:.1f}% eff_theta={tel_final['eff_theta_mean']:.2f}")
        print(f"    theta distribution: {theta_dist}")

    print(f"\n{'='*60}")
    print(f"  FLOAT32 vs INT4 THETA")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['mode']}: {r['label']:25s} best={r['best']*100:5.1f}% "
              f"theta={r['theta_mean']:.2f}+-{r['theta_std']:.2f} "
              f"firing={r['tel']['firing_mean']*100:.1f}%")
    print(f"{'='*60}")
