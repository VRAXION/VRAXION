"""
C19 Soft-Wave: ALL 3 params learnable as uint8 [0-255]
======================================================
freq, phase, rho — all int8, all mutate together.
Schedule: add, flip, theta, c19, c19, theta, flip, remove
c19 slot: 1/3 chance each for freq/phase/rho mutation.
Goal: see where each converges — what bins emerge.
"""
import sys, os, time, random, json
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
SCHEDULE = ['add','flip','theta','c19','c19','theta','flip','remove']

# --- INT8 to FLOAT mappings ---
def freq_i2f(v):  return 0.5 + (v / 255.0) * 1.5      # [0-255] -> [0.5, 2.0]
def phase_i2f(v): return (v / 255.0) * 2.0 * np.pi      # [0-255] -> [0, 2pi]
def rho_i2f(v):   return v / 255.0                       # [0-255] -> [0.0, 1.0]

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

def _eval_bigram(mask, theta, freq_f, phase_f, rho_f, seqs):
    sc = SelfWiringGraph.build_sparse_cache(mask); total = 0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32);s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                state=state,charge=charge,sparse_cache=sc,polarity=_pol,
                freq=freq_f,phase=phase_f,rho=rho_f)
            logits=np.dot(_bp_out,charge[H-OUT_DIM:])
            e=np.exp(logits-logits.max());pred=e/e.sum()
            tgt=_bigram[tb[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            s+=cos;n+=1
        total+=s/n if n else 0
    return total/len(seqs)

def worker_eval(args):
    mf,theta,freq_i,phase_i,rho_i,seed,pt=args
    rng=random.Random(seed);nrng=np.random.RandomState(seed)
    mask=mf.reshape(H,H)
    nm=mask;nt=theta;nfi=freq_i;npi=phase_i;nri=rho_i

    # Convert int8 -> float for eval
    freq_f = freq_i2f(freq_i.astype(np.float32))
    phase_f = phase_i2f(phase_i.astype(np.float32))
    rho_f = rho_i2f(rho_i.astype(np.float32))

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
        idx=rng.randint(0,H-1);nt=theta.copy();nt[idx]=float(rng.randint(1,15))
    elif pt=='c19':
        idx=rng.randint(0,H-1)
        choice=rng.randint(0,2)  # 0=freq, 1=phase, 2=rho
        if choice==0:
            nfi=freq_i.copy();nfi[idx]=np.uint8(rng.randint(0,255))
        elif choice==1:
            npi=phase_i.copy();npi[idx]=np.uint8(rng.randint(0,255))
        else:
            nri=rho_i.copy();nri[idx]=np.uint8(rng.randint(0,255))

    seqs=[]
    for _ in range(2):
        off=nrng.randint(0,len(_all_data)-100);seqs.append(_all_data[off:off+100])

    # old uses ORIGINAL int arrays, new uses MUTATED
    old=_eval_bigram(mask,theta,freq_f,phase_f,rho_f,seqs)
    new=_eval_bigram(nm,nt,
        freq_i2f(nfi.astype(np.float32)),
        phase_i2f(npi.astype(np.float32)),
        rho_i2f(nri.astype(np.float32)),seqs)
    return{'delta':float(new-old),'type':pt,
           'new_mask_flat':nm.flatten() if new>old else None,
           'new_theta':nt if pt=='theta' else None,
           'new_freq_i':nfi if pt=='c19' else None,
           'new_phase_i':npi if pt=='c19' else None,
           'new_rho_i':nri if pt=='c19' else None}

def print_uint8_hist(arr, name, i2f_func, bins=16):
    """Print histogram of uint8 array with float mapping."""
    vals = arr.astype(float)
    fvals = i2f_func(vals)
    counts, edges = np.histogram(vals, bins=bins, range=(0, 256))
    mx = max(counts.max(), 1)
    print(f"\n  {name} (uint8):  mean={vals.mean():.1f} std={vals.std():.1f} [{int(vals.min())},{int(vals.max())}]")
    print(f"  {name} (float):  mean={fvals.mean():.4f} std={fvals.std():.4f} [{fvals.min():.4f},{fvals.max():.4f}]")
    step = 256 // bins
    for i in range(bins):
        lo = i * step; hi = (i+1) * step
        flo = i2f_func(np.float32(lo)); fhi = i2f_func(np.float32(hi))
        bar = '#' * int(counts[i] / mx * 35)
        print(f"    [{lo:3d}-{hi:3d}) -> [{flo:.3f},{fhi:.3f}): {counts[i]:3d} |{bar}")
    # Top values
    uv, uc = np.unique(arr, return_counts=True)
    top = sorted(zip(uc, uv), reverse=True)[:8]
    print(f"    Top values: {', '.join(f'{int(v)}({c})' for c,v in top)}")

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

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
    pol_f32 = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)
    irng = np.random.RandomState(42)
    mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask, False)
    theta = np.full(H, 1.0, np.float32)

    # Init int8 arrays — random uniform
    frng = np.random.RandomState(42)
    freq_i = frng.randint(0, 256, size=H).astype(np.uint8)
    phase_i = frng.randint(0, 256, size=H).astype(np.uint8)
    rho_i = np.full(H, 77, dtype=np.uint8)  # 77/255 = 0.302 ~ 0.3 start

    print(f"\n{'='*60}")
    print(f"  C19 INT8 SWEEP: freq+phase+rho all learnable uint8")
    print(f"  schedule={SCHEDULE}")
    print(f"  freq:  uint8 [0-255] -> [0.5, 2.0]")
    print(f"  phase: uint8 [0-255] -> [0, 2pi]")
    print(f"  rho:   uint8 [0-255] -> [0.0, 1.0]")
    print(f"{'='*60}")
    sys.stdout.flush()

    pool = Pool(N_WORKERS, initializer=init_w, initargs=(bp_out, ALL_DATA, bigram, pol_f32))

    best=0;acc=0;c19_acc=0;t0=time.time()
    for step in range(1, 2001):
        pt=SCHEDULE[(step-1)%len(SCHEDULE)]
        if pt in('flip','remove','theta','c19') and mask.sum()==0:pt='add'
        args=[(mask.flatten(),theta.copy(),freq_i.copy(),phase_i.copy(),rho_i.copy(),
               1000+step*50+w,pt) for w in range(N_WORKERS)]
        res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])
        if br['delta']>THRESHOLD:
            if br['type'] in('add','remove','flip') and br['new_mask_flat'] is not None:
                mask[:]=br['new_mask_flat'].reshape(H,H);acc+=1
            elif br['type']=='theta' and br['new_theta'] is not None:
                theta[:]=br['new_theta'];acc+=1
            elif br['type']=='c19':
                if br['new_freq_i'] is not None: freq_i[:]=br['new_freq_i']
                if br['new_phase_i'] is not None: phase_i[:]=br['new_phase_i']
                if br['new_rho_i'] is not None: rho_i[:]=br['new_rho_i']
                c19_acc+=1;acc+=1
        if step%EVAL_EVERY==0:
            ea_list=[]
            freq_f=freq_i2f(freq_i.astype(np.float32))
            phase_f=phase_i2f(phase_i.astype(np.float32))
            rho_f=rho_i2f(rho_i.astype(np.float32))
            for s in eval_seqs:
                sc=SelfWiringGraph.build_sparse_cache(mask)
                st=np.zeros(H,np.float32);ch=np.zeros(H,np.float32);cor=0;tot=0
                for i in range(len(s)-1):
                    inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
                    st,ch=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                        decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                        state=st,charge=ch,sparse_cache=sc,polarity=pol_f32,
                        freq=freq_f,phase=phase_f,rho=rho_f)
                    logits=np.dot(bp_out,ch[H-OUT_DIM:])
                    if np.argmax(logits)==s[i+1]:cor+=1
                    tot+=1
                ea_list.append(cor/tot if tot else 0)
            ea=np.mean(ea_list)
            if ea>best:best=ea
            if step%200==0:
                print(f"  [{step:4d}] eval={ea*100:.1f}% best={best*100:.1f}% th={theta.mean():.1f} "
                      f"freq_i={freq_i.mean():.0f} phase_i={phase_i.mean():.0f} rho_i={rho_i.mean():.0f} "
                      f"c19={c19_acc} acc={acc} {time.time()-t0:.0f}s")
                sys.stdout.flush()

        # Snapshot every 500
        if step in (500, 1000, 1500, 2000):
            print(f"\n  --- SNAPSHOT step={step} best={best*100:.1f}% ---")
            print_uint8_hist(freq_i, "FREQ", freq_i2f)
            print_uint8_hist(phase_i, "PHASE", phase_i2f)
            print_uint8_hist(rho_i, "RHO", rho_i2f)
            sys.stdout.flush()

    pool.terminate();pool.join()
    elapsed=time.time()-t0

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS: best={best*100:.1f}% c19_acc={c19_acc} {elapsed:.0f}s")
    print(f"{'='*60}")

    # Theta final
    th=theta.astype(int)
    counts=np.bincount(th,minlength=16)[1:16]
    total=counts.sum()
    print(f"\n  THETA final: mean={theta.mean():.2f}")
    for v in range(1,16):
        bar='#'*int(counts[v-1]/max(counts.max(),1)*30)
        print(f"    theta={v:2d}: {counts[v-1]:3d} ({counts[v-1]/total*100:4.1f}%) |{bar}")

    # Correlations
    fi=freq_i.astype(float);pi=phase_i.astype(float);ri=rho_i.astype(float);ti=theta
    print(f"\n  Correlations:")
    print(f"    freq-phase: r={np.corrcoef(fi,pi)[0,1]:.4f}")
    print(f"    freq-rho:   r={np.corrcoef(fi,ri)[0,1]:.4f}")
    print(f"    phase-rho:  r={np.corrcoef(pi,ri)[0,1]:.4f}")
    print(f"    freq-theta: r={np.corrcoef(fi,ti)[0,1]:.4f}")
    print(f"    rho-theta:  r={np.corrcoef(ri,ti)[0,1]:.4f}")

    np.savez_compressed(os.path.join(BASE_DIR,"data/c19_int8_final.npz"),
        freq_i=freq_i,phase_i=phase_i,rho_i=rho_i,theta=theta,
        mask_rows=np.where(mask)[0].astype(np.uint16),
        mask_cols=np.where(mask)[1].astype(np.uint16))
    print(f"\n  Saved c19_int8_final.npz")
