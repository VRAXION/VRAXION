"""
Quick C mode rerun: freq+phase learnable convergence analysis
=============================================================
Rerun mode C (1600 steps) and print full distribution at end.
Focus: do freq/phase/rho converge to clusters or stay random?
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
SCHEDULE = ['add','flip','theta','wave','wave','theta','flip','remove']

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
_bp_out=None;_all_data=None;_bigram=None;_pol=None;_freq=None;_phase=None;_rho=None

def init_w(bpo,data,bg,pol,fr,ph,rh):
    global _bp_out,_all_data,_bigram,_pol,_freq,_phase,_rho
    _bp_out=bpo;_all_data=data;_bigram=bg;_pol=pol;_freq=fr;_phase=ph;_rho=rh

def _eval_bigram(mask, theta, freq_a, phase_a, seqs):
    sc = SelfWiringGraph.build_sparse_cache(mask); total = 0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32);s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                state=state,charge=charge,sparse_cache=sc,polarity=_pol,
                freq=freq_a,phase=phase_a,rho=_rho)
            logits=np.dot(_bp_out,charge[H-OUT_DIM:])
            e=np.exp(logits-logits.max());pred=e/e.sum()
            tgt=_bigram[tb[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            s+=cos;n+=1
        total+=s/n if n else 0
    return total/len(seqs)

def worker_eval(args):
    mf,theta,freq_a,phase_a,seed,pt=args
    rng=random.Random(seed);nrng=np.random.RandomState(seed)
    mask=mf.reshape(H,H);nm=mask;nt=theta;nf=freq_a;nph=phase_a
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
    elif pt=='wave':
        idx=rng.randint(0,H-1)
        if rng.random()<0.5:
            nf=freq_a.copy();nf[idx]=np.float32(rng.uniform(0.5,2.0))
        else:
            nph=phase_a.copy();nph[idx]=np.float32(rng.uniform(0,2*3.14159265))
    seqs=[]
    for _ in range(2):
        off=nrng.randint(0,len(_all_data)-100);seqs.append(_all_data[off:off+100])
    old=_eval_bigram(mask,theta,freq_a,phase_a,seqs)
    new=_eval_bigram(nm,nt,nf,nph,seqs)
    return{'delta':float(new-old),'type':pt,
           'new_mask_flat':nm.flatten() if new>old else None,
           'new_theta':nt if pt=='theta' else None,
           'new_freq':nf if pt=='wave' else None,
           'new_phase':nph if pt=='wave' else None}

def ascii_hist(values, bins=15, width=40, label=""):
    counts, edges = np.histogram(values, bins=bins)
    mx = max(counts) if max(counts) > 0 else 1
    print(f"\n  {label}  (N={len(values)}, mean={np.mean(values):.4f}, std={np.std(values):.4f})")
    print(f"  range: [{values.min():.4f}, {values.max():.4f}]  median={np.median(values):.4f}")
    for i in range(bins):
        bar = '#' * int(counts[i] / mx * width)
        print(f"    [{edges[i]:7.3f},{edges[i+1]:7.3f}) {counts[i]:4d} |{bar}")

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
    rho = np.full(H, 0.3, np.float32)
    irng = np.random.RandomState(42)
    mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask, False)
    theta = np.full(H, 1.0, np.float32)

    frng = np.random.RandomState(42)
    freq_arr = (frng.rand(H).astype(np.float32) * 1.5 + 0.5)
    phase_arr = (frng.rand(H).astype(np.float32) * 2.0 * np.pi)

    # Save init for comparison
    freq_init = freq_arr.copy()
    phase_init = phase_arr.copy()

    print(f"\n{'='*60}")
    print(f"  MODE C: wave LEARNABLE — convergence analysis")
    print(f"  1600 steps, schedule={SCHEDULE}")
    print(f"{'='*60}")

    pool = Pool(N_WORKERS, initializer=init_w,
        initargs=(bp_out, ALL_DATA, bigram, pol_f32, freq_arr, phase_arr, rho))

    best=0;acc=0;wave_acc=0;freq_acc=0;phase_acc=0;t0=time.time()
    for step in range(1, 1601):
        pt=SCHEDULE[(step-1)%len(SCHEDULE)]
        if pt in('flip','remove','theta','wave') and mask.sum()==0:pt='add'
        args=[(mask.flatten(),theta.copy(),freq_arr.copy(),phase_arr.copy(),
               1000+step*50+w,pt) for w in range(N_WORKERS)]
        res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])
        if br['delta']>THRESHOLD:
            if br['type'] in('add','remove','flip') and br['new_mask_flat'] is not None:
                mask[:]=br['new_mask_flat'].reshape(H,H);acc+=1
            elif br['type']=='theta' and br['new_theta'] is not None:
                theta[:]=br['new_theta'];acc+=1
            elif br['type']=='wave':
                if br['new_freq'] is not None:
                    # Check which neuron changed to count freq vs phase
                    diff = np.abs(br['new_freq'] - freq_arr)
                    if diff.max() > 0.001:
                        freq_acc += 1
                    freq_arr[:]=br['new_freq']
                if br['new_phase'] is not None:
                    diff = np.abs(br['new_phase'] - phase_arr)
                    if diff.max() > 0.001:
                        phase_acc += 1
                    phase_arr[:]=br['new_phase']
                wave_acc+=1;acc+=1
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
                        freq=freq_arr,phase=phase_arr,rho=rho)
                    logits=np.dot(bp_out,ch[H-OUT_DIM:])
                    if np.argmax(logits)==s[i+1]:cor+=1
                    tot+=1
                ea_list.append(cor/tot if tot else 0)
            ea=np.mean(ea_list)
            if ea>best:best=ea
            if step%400==0:
                print(f"  [{step:4d}] eval={ea*100:.1f}% best={best*100:.1f}% th={theta.mean():.1f} "
                      f"f={freq_arr.mean():.3f}[{freq_arr.std():.3f}] p_std={phase_arr.std():.2f} "
                      f"wacc={wave_acc}(f={freq_acc},p={phase_acc}) {time.time()-t0:.0f}s")
                sys.stdout.flush()

    pool.terminate();pool.join()
    elapsed=time.time()-t0

    # === FULL CONVERGENCE ANALYSIS ===
    print(f"\n{'='*60}")
    print(f"  CONVERGENCE ANALYSIS (step 1600, best={best*100:.1f}%)")
    print(f"  wave_acc={wave_acc} (freq={freq_acc}, phase={phase_acc})")
    print(f"{'='*60}")

    # --- FREQ ---
    ascii_hist(freq_arr, bins=15, label="FREQ (trained)")
    ascii_hist(freq_init, bins=15, label="FREQ (init for comparison)")

    # Per-neuron delta from init
    freq_delta = freq_arr - freq_init
    print(f"\n  FREQ delta from init:")
    print(f"    mean_delta={freq_delta.mean():.4f} std_delta={freq_delta.std():.4f}")
    print(f"    neurons unchanged (|d|<0.01): {(np.abs(freq_delta)<0.01).sum()}/{H}")
    print(f"    neurons changed >0.5: {(np.abs(freq_delta)>0.5).sum()}/{H}")
    ascii_hist(freq_delta, bins=15, label="FREQ delta (trained - init)")

    # --- PHASE ---
    ascii_hist(phase_arr, bins=16, label="PHASE (trained)")
    ascii_hist(phase_init, bins=16, label="PHASE (init for comparison)")

    phase_delta = phase_arr - phase_init
    # Wrap to [-pi, pi]
    phase_delta = (phase_delta + np.pi) % (2*np.pi) - np.pi
    print(f"\n  PHASE delta from init (wrapped [-pi,pi]):")
    print(f"    mean_delta={phase_delta.mean():.4f} std_delta={phase_delta.std():.4f}")
    print(f"    neurons unchanged (|d|<0.01): {(np.abs(phase_delta)<0.01).sum()}/{H}")
    print(f"    neurons changed >pi/4: {(np.abs(phase_delta)>np.pi/4).sum()}/{H}")
    ascii_hist(phase_delta, bins=15, label="PHASE delta (wrapped)")

    # --- FREQ clusters ---
    fr_rd = np.round(freq_arr, 1)
    uv, uc = np.unique(fr_rd, return_counts=True)
    top = sorted(zip(uc, uv), reverse=True)[:10]
    print(f"\n  FREQ clusters (rounded to 0.1):")
    for cnt, fv in top:
        pct = cnt / H * 100
        bar = '#' * int(cnt / max(top[0][0],1) * 30)
        print(f"    freq={fv:.1f}: {cnt:3d} ({pct:.1f}%) |{bar}")

    # --- PHASE clusters ---
    ph_rd = np.round(phase_arr / (np.pi/4)) * (np.pi/4)
    uv, uc = np.unique(ph_rd, return_counts=True)
    top = sorted(zip(uc, uv), reverse=True)[:8]
    print(f"\n  PHASE clusters (rounded to pi/4):")
    for cnt, pv in top:
        pct = cnt / H * 100
        label_p = f"{pv/np.pi:.2f}pi"
        bar = '#' * int(cnt / max(top[0][0],1) * 30)
        print(f"    phase={label_p:>8s}: {cnt:3d} ({pct:.1f}%) |{bar}")

    # --- CORRELATION ---
    corr = np.corrcoef(freq_arr, phase_arr)[0,1]
    print(f"\n  freq-phase correlation: r={corr:.4f}")

    # --- THETA final ---
    th = theta.astype(int)
    counts = np.bincount(th, minlength=16)[1:16]
    total = counts.sum()
    print(f"\n  THETA final (for comparison):")
    print(f"    mean={theta.mean():.2f} median={np.median(theta):.0f}")
    for v in range(1, 16):
        bar = '#' * int(counts[v-1] / max(counts.max(),1) * 30)
        print(f"    theta={v:2d}: {counts[v-1]:3d} ({counts[v-1]/total*100:4.1f}%) |{bar}")

    # --- EFFECTIVE WAVE at each tick ---
    print(f"\n  Effective wave modulation (rho=0.3):")
    print(f"  {'tick':>4s}  {'wave_mean':>9s}  {'wave_std':>8s}  {'eff_theta_mult':>14s}")
    for t in range(8):
        w = np.sin(t * freq_arr + phase_arr)
        mod = 1.0 + 0.3 * w
        print(f"  {t:4d}  {w.mean():9.4f}  {w.std():8.4f}  x{mod.mean():.3f} [{mod.min():.3f},{mod.max():.3f}]")

    print(f"\n  TOTAL TIME: {elapsed:.0f}s")
    sys.stdout.flush()

    # Save
    np.savez_compressed(os.path.join(BASE_DIR,"data/freq_phase_trained_C.npz"),
        freq=freq_arr, phase=phase_arr, freq_init=freq_init, phase_init=phase_init,
        theta=theta, mask_rows=np.where(mask)[0].astype(np.uint16),
        mask_cols=np.where(mask)[1].astype(np.uint16))
    print(f"  Saved freq_phase_trained_C.npz")
