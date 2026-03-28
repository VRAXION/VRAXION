"""
Scale sweep: Does the 0.625 output ratio hold at different H?
=============================================================
Golden ratio proportions at each scale:
  in_dim  = H * 0.25
  out_dim = H * 0.625
  hidden  = H * 0.125
  SDR K   = in_dim * 0.20

H=128, 192, 256, 384 — same proportions, learnable theta.
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

N_WORKERS = 18; TICKS = 8; INPUT_DURATION = 2
THRESHOLD = 0.00005; EVAL_EVERY = 20
SCHEDULE = ['add', 'remove', 'flip', 'theta', 'theta', 'theta', 'decay', 'decay']
INIT_DENSITY = 0.05; THETA_INIT = 1.0

SCALES = [
    {'H': 128, 'in': 32,  'out': 80,  'K': 7},
    {'H': 192, 'in': 48,  'out': 120, 'K': 10},
    {'H': 256, 'in': 64,  'out': 160, 'K': 13},
    {'H': 384, 'in': 96,  'out': 240, 'K': 20},
]

def build_sdr(n, dim, k, seed):
    rng = np.random.RandomState(seed)
    t = np.zeros((n, dim), np.float32)
    for v in range(n):
        t[v, rng.choice(dim, size=k, replace=False)] = 1.0
    return t

_bp_in=None;_bp_out=None;_all_data=None;_bigram=None
_pol=None;_freq=None;_phase=None;_rho=None
_H=None;_in_dim=None;_out_dim=None

def init_w(bpi,bpo,data,bg,pol,fr,ph,rh,h,ind,outd):
    global _bp_in,_bp_out,_all_data,_bigram,_pol,_freq,_phase,_rho,_H,_in_dim,_out_dim
    _bp_in=bpi;_bp_out=bpo;_all_data=data;_bigram=bg
    _pol=pol;_freq=fr;_phase=ph;_rho=rh;_H=h;_in_dim=ind;_out_dim=outd

def _eval_bigram(mask, theta, decay, seqs):
    sc = SelfWiringGraph.build_sparse_cache(mask)
    total = 0.0
    for tb in seqs:
        state=np.zeros(_H,np.float32);charge=np.zeros(_H,np.float32)
        s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(_H,np.float32);inj[0:_in_dim]=_bp_in[tb[i]]
            state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,decay=decay,
                ticks=TICKS,input_duration=INPUT_DURATION,state=state,charge=charge,
                sparse_cache=sc,polarity=_pol,freq=_freq,phase=_phase,rho=_rho)
            logits=np.dot(_bp_out,charge[_H-_out_dim:])
            e=np.exp(logits-logits.max());pred=e/e.sum()
            tgt=_bigram[tb[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            s+=cos;n+=1
        total+=s/n if n else 0
    return total/len(seqs)

def worker_eval(args):
    mf,theta,decay,seed,pt=args
    H=_H
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
        nt[idx]=rng.uniform(0.0,16.0)
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

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    resolve_fineweb_path(); ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram = np.load(os.path.join(BASE_DIR, "data", "bigram_table.npy"))
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[o:o+100] for o in [eval_rng.randint(0,len(ALL_DATA)-100) for _ in range(5)]]

    print(f"\n{'='*70}")
    print(f"  SCALE SWEEP — 0.625 output ratio at different H")
    print(f"{'='*70}")

    results = []
    for sc in SCALES:
        H=sc['H']; in_dim=sc['in']; out_dim=sc['out']; K=sc['K']
        hidden=H-in_dim-out_dim
        bp_in = build_sdr(256, in_dim, K, 42)
        rng_out = np.random.RandomState(12345)
        bp_out = rng_out.randn(256, out_dim).astype(np.float32)
        bp_out /= np.linalg.norm(bp_out, axis=1, keepdims=True)

        print(f"\n{'='*60}")
        print(f"  H={H}, in={in_dim}(K={K}), out={out_dim}, hidden={hidden}")
        print(f"{'='*60}")

        random.seed(42); np.random.seed(42)
        ref = SelfWiringGraph(max(in_dim,16), hidden=H, projection_scale=1.0)
        pol = ref.polarity.astype(np.float32)
        irng = np.random.RandomState(42)
        mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool)
        np.fill_diagonal(mask, False)
        theta = np.full(H, THETA_INIT, np.float32)
        decay = np.random.RandomState(99).uniform(0.08, 0.24, H).astype(np.float32)
        edges = int(mask.sum())
        print(f"  edges={edges}")

        pool = Pool(N_WORKERS, initializer=init_w,
            initargs=(bp_in, bp_out, ALL_DATA, bigram, pol, ref.freq, ref.phase, ref.rho, H, in_dim, out_dim))

        best_eval=0;stall=0;accepts=0;t_acc=0;t0=time.time()
        try:
            for step in range(1, 3001):
                pt = SCHEDULE[(step-1) % len(SCHEDULE)]
                if pt in ('flip','remove','decay','theta') and mask.sum()==0: pt='add'
                args = [(mask.flatten(),theta.copy(),decay.copy(),1000+step*50+w,pt) for w in range(N_WORKERS)]
                res = pool.map(worker_eval, args)
                br = max(res, key=lambda x: x['delta'])
                if br['delta'] > THRESHOLD:
                    if br['type'] in ('add','remove','flip') and br['new_mask_flat'] is not None:
                        mask[:]=br['new_mask_flat'].reshape(H,H); accepts+=1
                    elif br['type']=='theta' and br['new_theta'] is not None:
                        theta[:]=br['new_theta']; t_acc+=1; accepts+=1
                    elif br['type']=='decay' and br['new_decay'] is not None:
                        decay[:]=br['new_decay']; accepts+=1
                if step % EVAL_EVERY == 0:
                    el=time.time()-t0
                    ea_list=[]
                    for s in eval_seqs:
                        sc2=SelfWiringGraph.build_sparse_cache(mask)
                        st=np.zeros(H,np.float32);ch=np.zeros(H,np.float32);cor=0;tot=0
                        for i in range(len(s)-1):
                            inj=np.zeros(H,np.float32);inj[0:in_dim]=bp_in[s[i]]
                            st,ch=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,decay=decay,
                                ticks=TICKS,input_duration=INPUT_DURATION,state=st,charge=ch,
                                sparse_cache=sc2,polarity=pol,freq=ref.freq,phase=ref.phase,rho=ref.rho)
                            logits=np.dot(bp_out,ch[H-out_dim:])
                            if np.argmax(logits)==s[i+1]:cor+=1
                            tot+=1
                        ea_list.append(cor/tot if tot else 0)
                    ea=np.mean(ea_list)
                    if ea>best_eval: best_eval=ea; stall=0
                    else: stall+=EVAL_EVERY
                    if step % 200 == 0:
                        sps = step/el if el > 0 else 0
                        print(f"  [H={H}][{step:4d}] eval={ea*100:.1f}% best={best_eval*100:.1f}% "
                              f"theta={theta.mean():.2f} acc={accepts} {el:.0f}s ({sps:.1f}sps)")
                    sys.stdout.flush()
                    if stall >= 800 and step >= 600:
                        print(f"  ** STALL"); break
        finally:
            pool.terminate(); pool.join()

        elapsed=time.time()-t0
        final_edges = int(mask.sum())
        results.append({'H':H,'in':in_dim,'out':out_dim,'hidden':hidden,'K':K,
                        'best':best_eval,'edges':final_edges,'acc':accepts,
                        'theta':float(theta.mean()),'time':elapsed})
        print(f"  DONE: H={H} best={best_eval*100:.1f}% edges={final_edges} theta={theta.mean():.2f} {elapsed:.0f}s")

    print(f"\n{'='*70}")
    print(f"  SCALE SWEEP RESULTS (0.625 output ratio)")
    print(f"{'='*70}")
    print(f"  {'H':>4} {'In':>4} {'Out':>4} {'Hid':>4} {'K':>3} {'Best':>6} {'Edges':>6} {'Theta':>6} {'sps':>5}")
    for r in results:
        sps = 2000/r['time'] if r['time']>0 else 0
        print(f"  {r['H']:4d} {r['in']:4d} {r['out']:4d} {r['hidden']:4d} {r['K']:3d} "
              f"{r['best']*100:5.1f}% {r['edges']:6d} {r['theta']:6.2f} {sps:5.1f}")
    print(f"{'='*70}")
