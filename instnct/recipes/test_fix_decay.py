"""
Phi Overlap Test: Input and output zones overlap, 0 pure hidden
===============================================================
H=256, phi nested downshift:
  Output = last H/phi = last 158 neurons  [98...255]
  Input  = first H/phi = first 158 neurons [0...157]  (SDR, K=32, 20%)
  Overlap = neurons [98...157] = 60 neurons (both I and O)
  Pure hidden = 0

Compare against current best (no overlap):
  Input  = first 64   [0...63]
  Output = last 160   [96...255]
  Hidden = 32          [64...95]
"""
import sys, os, time, random
import numpy as np
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "model"))
from graph import SelfWiringGraph

H = 256; N_WORKERS = 18; TICKS = 8; INPUT_DURATION = 2
THRESHOLD = 0.00005; EVAL_EVERY = 20
SCHEDULE = ['add', 'remove', 'flip', 'theta', 'theta', 'theta', 'flip', 'flip']
INIT_DENSITY = 0.05; THETA_INIT = 1.0
PHI = (1 + 5**0.5) / 2

def build_sdr(n, dim, k, seed):
    rng = np.random.RandomState(seed)
    t = np.zeros((n, dim), np.float32)
    for v in range(n):
        t[v, rng.choice(dim, size=k, replace=False)] = 1.0
    return t

# Phi overlap config
PHI_IN = int(round(H / PHI))    # 158
PHI_OUT = int(round(H / PHI))   # 158
PHI_K = int(round(PHI_IN * 0.20))  # 32
PHI_OVERLAP = PHI_IN + PHI_OUT - H  # 60

_bp_in=None;_bp_out=None;_all_data=None;_bigram=None
_pol=None;_freq=None;_phase=None;_rho=None
_in_dim=None;_out_dim=None

def init_w(bpi,bpo,data,bg,pol,fr,ph,rh,ind,outd):
    global _bp_in,_bp_out,_all_data,_bigram,_pol,_freq,_phase,_rho,_in_dim,_out_dim
    _bp_in=bpi;_bp_out=bpo;_all_data=data;_bigram=bg
    _pol=pol;_freq=fr;_phase=ph;_rho=rh;_in_dim=ind;_out_dim=outd

def _eval_bigram(mask, theta, decay, seqs):
    sc = SelfWiringGraph.build_sparse_cache(mask)
    total = 0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32)
        s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:_in_dim]=_bp_in[tb[i]]
            state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,decay=decay,
                ticks=TICKS,input_duration=INPUT_DURATION,state=state,charge=charge,
                sparse_cache=sc,polarity=_pol,freq=_freq,phase=_phase,rho=_rho)
            logits=np.dot(_bp_out,charge[H-_out_dim:])
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

def run_config(label, in_dim, out_dim, sdr_k, ALL_DATA, bigram, eval_seqs):
    overlap = max(0, in_dim + out_dim - H)
    pure_hidden = H - in_dim - out_dim + overlap
    bp_in = build_sdr(256, in_dim, sdr_k, 42)
    rng_out = np.random.RandomState(12345)
    bp_out = rng_out.randn(256, out_dim).astype(np.float32)
    bp_out /= np.linalg.norm(bp_out, axis=1, keepdims=True)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  in={in_dim}(K={sdr_k}), out={out_dim}, overlap={overlap}, pure_hidden={pure_hidden}")
    print(f"{'='*60}")

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(max(in_dim,16), hidden=H, projection_scale=1.0)
    pol = ref.polarity.astype(np.float32)
    irng = np.random.RandomState(42)
    mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool)
    np.fill_diagonal(mask, False)
    theta = np.full(H, THETA_INIT, np.float32)
    decay = np.random.RandomState(99).uniform(0.08, 0.24, H).astype(np.float32)

    pool = Pool(N_WORKERS, initializer=init_w,
        initargs=(bp_in, bp_out, ALL_DATA, bigram, pol, ref.freq, ref.phase, ref.rho, in_dim, out_dim))

    best_eval=0;stall=0;accepts=0;t_acc=0;t0=time.time()
    try:
        for step in range(1, 2001):
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
                    sc=SelfWiringGraph.build_sparse_cache(mask)
                    st=np.zeros(H,np.float32);ch=np.zeros(H,np.float32);cor=0;tot=0
                    for i in range(len(s)-1):
                        inj=np.zeros(H,np.float32);inj[0:in_dim]=bp_in[s[i]]
                        st,ch=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,decay=decay,
                            ticks=TICKS,input_duration=INPUT_DURATION,state=st,charge=ch,
                            sparse_cache=sc,polarity=pol,freq=ref.freq,phase=ref.phase,rho=ref.rho)
                        logits=np.dot(bp_out,ch[H-out_dim:])
                        if np.argmax(logits)==s[i+1]:cor+=1
                        tot+=1
                    ea_list.append(cor/tot if tot else 0)
                ea=np.mean(ea_list)
                if ea>best_eval: best_eval=ea; stall=0
                else: stall+=EVAL_EVERY
                if step % 200 == 0:
                    print(f"  [{step:4d}] eval={ea*100:.1f}% best={best_eval*100:.1f}% "
                          f"theta={theta.mean():.2f} acc={accepts} {el:.0f}s")
                sys.stdout.flush()
                if stall >= 800 and step >= 600:
                    print(f"  ** STALL"); break
    finally:
        pool.terminate(); pool.join()

    elapsed=time.time()-t0
    print(f"\n  DONE: best={best_eval*100:.1f}% theta={theta.mean():.2f} acc={accepts} {elapsed:.0f}s")
    return {'label':label,'best':best_eval,'in':in_dim,'out':out_dim,
            'overlap':overlap,'hidden':pure_hidden,'theta':float(theta.mean()),
            'acc':accepts,'time':elapsed}

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    resolve_fineweb_path(); ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB")
    print(f"Phi = {PHI:.6f}")
    print(f"H/phi = {H/PHI:.1f} -> {PHI_IN}")
    print(f"Overlap = {PHI_OVERLAP}")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram = np.load(os.path.join(BASE_DIR, "data", "bigram_table.npy"))
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[o:o+100] for o in [eval_rng.randint(0,len(ALL_DATA)-100) for _ in range(5)]]

    r = run_config(
        f"FIX DECAY 0.16 (2 extra flips): in={PHI_IN} out={PHI_OUT} overlap={PHI_OVERLAP}",
        PHI_IN, PHI_OUT, PHI_K, ALL_DATA, bigram, eval_seqs)

    print(f"\n{'='*60}")
    print(f"  FIX DECAY 0.16 (2 extra flips) RESULT")
    print(f"  in={r['in']}, out={r['out']}, overlap={r['overlap']}, pure_hidden={r['hidden']}")
    print(f"  best={r['best']*100:.1f}%  theta={r['theta']:.2f}")
    print(f"  Compare: current best (in=64,out=160,hid=32) = 20.0%")
    print(f"{'='*60}")
