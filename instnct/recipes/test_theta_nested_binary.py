"""
Nested Binary Theta: 2 mutation types (fine + big)
===================================================
Values: [1, 5, 10, 15]
theta_fine: step to neighbor (1<->5, 5<->10, 10<->15)
theta_big:  jump across (1<->10, 1<->15, 5<->15)

Schedule: [add, remove, flip, theta_fine, theta_big, decay, decay, decay]

H=256, phi overlap, SDR input, FREQ output.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR)); sys.path.insert(0, str(ROOT_DIR / "model"))
from graph import SelfWiringGraph

H = 256; N_WORKERS = 18; TICKS = 8; INPUT_DURATION = 2
THRESHOLD = 0.00005; EVAL_EVERY = 20
SCHEDULE = ['add', 'remove', 'flip', 'theta_fine', 'theta_big', 'decay', 'decay', 'decay']
INIT_DENSITY = 0.05
PHI = (1 + 5**0.5) / 2
IN_DIM = int(round(H / PHI))
OUT_DIM = int(round(H / PHI))
SDR_K = int(round(IN_DIM * 0.20))
THETA_VALS = [1.0, 5.0, 10.0, 15.0]
THETA_NEIGHBORS = {1.0: [5.0], 5.0: [1.0, 10.0], 10.0: [5.0, 15.0], 15.0: [10.0]}
THETA_JUMPS = {1.0: [10.0, 15.0], 5.0: [15.0], 10.0: [1.0], 15.0: [1.0, 5.0]}

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
    elif pt=='theta_fine':
        idx=rng.randint(0,H-1);nt=theta.copy()
        cur = float(nt[idx])
        # Find closest known value
        closest = min(THETA_VALS, key=lambda v: abs(v - cur))
        neighbors = THETA_NEIGHBORS.get(closest, THETA_VALS)
        nt[idx] = rng.choice(neighbors)
    elif pt=='theta_big':
        idx=rng.randint(0,H-1);nt=theta.copy()
        cur = float(nt[idx])
        closest = min(THETA_VALS, key=lambda v: abs(v - cur))
        jumps = THETA_JUMPS.get(closest, THETA_VALS)
        nt[idx] = rng.choice(jumps)
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
           'new_theta':nt if pt.startswith('theta') else None,
           'new_decay':nd if pt=='decay' else None}

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
    print(f"  NESTED BINARY THETA: fine + big mutations")
    print(f"  Values: {THETA_VALS}")
    print(f"  Fine: {THETA_NEIGHBORS}")
    print(f"  Big:  {THETA_JUMPS}")
    print(f"  Schedule: {SCHEDULE}")
    print(f"{'='*60}")

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
    pol = ref.polarity.astype(np.float32)
    irng = np.random.RandomState(42)
    mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask, False)
    theta = np.full(H, 1.0, np.float32)  # start all relay
    decay = np.random.RandomState(99).uniform(0.08, 0.24, H).astype(np.float32)

    pool = Pool(N_WORKERS, initializer=init_w,
        initargs=(bp_out, ALL_DATA, bigram, pol, ref.freq, ref.phase, ref.rho))

    best=0;stall=0;acc=0;fine_acc=0;big_acc=0;t0=time.time()
    history=[];MONITOR_WINDOW=15

    try:
        step=0
        while True:
            step+=1
            pt=SCHEDULE[(step-1)%len(SCHEDULE)]
            if pt in('flip','remove','decay','theta_fine','theta_big') and mask.sum()==0:pt='add'
            args=[(mask.flatten(),theta.copy(),decay.copy(),1000+step*50+w,pt) for w in range(N_WORKERS)]
            res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])
            if br['delta']>THRESHOLD:
                if br['type'] in('add','remove','flip') and br['new_mask_flat'] is not None:
                    mask[:]=br['new_mask_flat'].reshape(H,H);acc+=1
                elif br['type']=='theta_fine' and br['new_theta'] is not None:
                    theta[:]=br['new_theta'];fine_acc+=1;acc+=1
                elif br['type']=='theta_big' and br['new_theta'] is not None:
                    theta[:]=br['new_theta'];big_acc+=1;acc+=1
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

                # Count neurons per theta value
                n1=int((theta<3).sum());n5=int(((theta>=3)&(theta<7.5)).sum())
                n10=int(((theta>=7.5)&(theta<12.5)).sum());n15=int((theta>=12.5).sum())

                entry={'step':step,'eval':ea,'edges':int(mask.sum()),
                       'theta_mean':float(theta.mean()),'decay_mean':float(decay.mean())}
                history.append(entry)

                if step%100==0:
                    print(f"  [{step:5d}] eval={ea*100:.1f}% best={best*100:.1f}% "
                          f"edges={int(mask.sum())} theta={theta.mean():.1f} "
                          f"[R={n1} S={n5} C={n10} G={n15}] "
                          f"fine={fine_acc} big={big_acc} acc={acc} "
                          f"{el:.0f}s ({step/el:.1f}sps)")
                sys.stdout.flush()

                if len(history)>=MONITOR_WINDOW and step>=400:
                    w=history[-MONITOR_WINDOW:]
                    e_sp=max(h['eval'] for h in w)-min(h['eval'] for h in w)
                    ed_sp=max(h['edges'] for h in w)-min(h['edges'] for h in w)
                    th_sp=max(h['theta_mean'] for h in w)-min(h['theta_mean'] for h in w)
                    dc_sp=max(h['decay_mean'] for h in w)-min(h['decay_mean'] for h in w)
                    all_flat=e_sp<0.02 and ed_sp<20 and th_sp<0.5 and dc_sp<0.01
                    if step%200==0:
                        print(f"         monitor: eval={e_sp:.3f} edge={ed_sp} theta={th_sp:.3f} "
                              f"{'-> FLAT' if all_flat else '-> active'}")
                    if all_flat:print(f"  ** STOP at step {step}");break
                if step>=10000:print("  ** CAP");break
    finally:pool.terminate();pool.join()

    elapsed=time.time()-t0;edges=int(mask.sum())
    n1=int((theta<3).sum());n5=int(((theta>=3)&(theta<7.5)).sum())
    n10=int(((theta>=7.5)&(theta<12.5)).sum());n15=int((theta>=12.5).sum())

    save_experiment(name="theta_nested_binary",
        mask=mask,theta=theta,decay=decay,polarity=pol.astype(np.int8),bp_out=bp_out,
        config={'H':H,'values':THETA_VALS,'schedule':SCHEDULE},
        result={'best':round(best,4),'edges':edges,'fine_acc':fine_acc,'big_acc':big_acc,
                'n1':n1,'n5':n5,'n10':n10,'n15':n15})

    print(f"\n{'='*60}")
    print(f"  DONE: best={best*100:.1f}% edges={edges} steps={step}")
    print(f"  Neurons: R(1)={n1} S(5)={n5} C(10)={n10} G(15)={n15}")
    print(f"  Fine accepts: {fine_acc}, Big accepts: {big_acc}")
    print(f"  {elapsed:.0f}s")
    print(f"  Compare: quad random resample=12.7%, int4=15.6%")
    print(f"{'='*60}")
