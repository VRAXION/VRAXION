"""
Freq + Phase convergence sweep
===============================
freq and phase were NEVER mutated in prior recipes (not in schedule).
This test adds 'wave' to the schedule: each wave slot randomly mutates
one neuron's freq OR phase (50-50).

Modes:
  A: NO wave gating (freq=None, phase=None) -- does it even help?
  B: wave gating, freq+phase FROZEN at init (diversity but no learning)
  C: wave gating, freq+phase LEARNABLE float32 (full learning)

After run: check convergence distribution of freq/phase in mode C.
"""
import sys, os, time, random, json
import numpy as np
from multiprocessing import Pool
from pathlib import Path

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

_bp_out=None;_all_data=None;_bigram=None;_pol=None
_freq=None;_phase=None;_rho=None;_wave_mode=None

def init_w(bpo,data,bg,pol,fr,ph,rh,wm):
    global _bp_out,_all_data,_bigram,_pol,_freq,_phase,_rho,_wave_mode
    _bp_out=bpo;_all_data=data;_bigram=bg;_pol=pol
    _freq=fr;_phase=ph;_rho=rh;_wave_mode=wm

def _eval_bigram(mask, theta, freq_arr, phase_arr, seqs):
    sc = SelfWiringGraph.build_sparse_cache(mask); total = 0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32);s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                state=state,charge=charge,sparse_cache=sc,polarity=_pol,
                freq=freq_arr,phase=phase_arr,rho=_rho)
            logits=np.dot(_bp_out,charge[H-OUT_DIM:])
            e=np.exp(logits-logits.max());pred=e/e.sum()
            tgt=_bigram[tb[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            s+=cos;n+=1
        total+=s/n if n else 0
    return total/len(seqs)

def worker_eval(args):
    mf,theta,freq_arr,phase_arr,seed,pt=args
    rng=random.Random(seed);nrng=np.random.RandomState(seed)
    mask=mf.reshape(H,H);nm=mask;nt=theta;nf=freq_arr;np_=phase_arr
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
            nf=freq_arr.copy();nf[idx]=np.float32(rng.uniform(0.5,2.0))
        else:
            np_=phase_arr.copy();np_[idx]=np.float32(rng.uniform(0,2*3.14159265))
    seqs=[]
    for _ in range(2):
        off=nrng.randint(0,len(_all_data)-100);seqs.append(_all_data[off:off+100])
    old=_eval_bigram(mask,theta,freq_arr,phase_arr,seqs)
    new=_eval_bigram(nm,nt,nf,np_,seqs)
    return{'delta':float(new-old),'type':pt,
           'new_mask_flat':nm.flatten() if new>old else None,
           'new_theta':nt if pt=='theta' else None,
           'new_freq':nf if pt=='wave' else None,
           'new_phase':np_ if pt=='wave' else None}

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

    # Schedule: 2 add, 1 flip, 3 theta, wave slots, 1 flip, remove at end
    MODES = [
        ('A', 'NO wave gating',       'none',
         ['add','flip','theta','theta','theta','flip','flip','remove']),
        ('B', 'wave FROZEN at init',   'frozen',
         ['add','flip','theta','theta','theta','flip','flip','remove']),
        ('C', 'wave LEARNABLE float32','learnable',
         ['add','flip','theta','wave','wave','theta','flip','remove']),
    ]

    print(f"\n{'='*70}")
    print(f"  FREQ+PHASE SWEEP: none vs frozen vs learnable")
    print(f"{'='*70}")

    results = []
    for mk, label, wave_mode, schedule in MODES:
        print(f"\n  {mk}: {label}")
        print(f"      schedule={schedule}")

        random.seed(42); np.random.seed(42)
        ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
        pol_f32 = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)
        rho = np.full(H, 0.3, np.float32)
        irng = np.random.RandomState(42)
        mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask, False)
        theta = np.full(H, 1.0, np.float32)

        # freq/phase init
        frng = np.random.RandomState(42)
        freq_arr = (frng.rand(H).astype(np.float32) * 1.5 + 0.5)
        phase_arr = (frng.rand(H).astype(np.float32) * 2.0 * np.pi)

        # For mode A: pass None to rollout_token (no wave gating)
        if wave_mode == 'none':
            w_freq = None; w_phase = None
        else:
            w_freq = freq_arr.copy(); w_phase = phase_arr.copy()

        pool = Pool(N_WORKERS, initializer=init_w,
            initargs=(bp_out, ALL_DATA, bigram, pol_f32, w_freq, w_phase, rho, wave_mode))

        best=0;acc=0;wave_acc=0;t0=time.time()
        history=[]
        for step in range(1, 2001):
            pt = schedule[(step-1) % len(schedule)]
            if pt in ('flip','remove','theta','wave') and mask.sum()==0: pt='add'
            if pt == 'wave' and wave_mode != 'learnable': pt = 'theta'  # frozen/none: skip wave

            args=[(mask.flatten(),theta.copy(),
                   w_freq.copy() if w_freq is not None else freq_arr.copy(),
                   w_phase.copy() if w_phase is not None else phase_arr.copy(),
                   1000+step*50+w,pt) for w in range(N_WORKERS)]
            res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])
            if br['delta']>THRESHOLD:
                if br['type'] in('add','remove','flip') and br['new_mask_flat'] is not None:
                    mask[:]=br['new_mask_flat'].reshape(H,H);acc+=1
                elif br['type']=='theta' and br['new_theta'] is not None:
                    theta[:]=br['new_theta'];acc+=1
                elif br['type']=='wave':
                    if br['new_freq'] is not None and w_freq is not None:
                        w_freq[:]=br['new_freq']
                    if br['new_phase'] is not None and w_phase is not None:
                        w_phase[:]=br['new_phase']
                    wave_acc+=1;acc+=1

            if step%EVAL_EVERY==0:
                el=time.time()-t0;ea_list=[]
                for s in eval_seqs:
                    sc=SelfWiringGraph.build_sparse_cache(mask)
                    st=np.zeros(H,np.float32);ch=np.zeros(H,np.float32);cor=0;tot=0
                    for i in range(len(s)-1):
                        inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
                        st,ch=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                            decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                            state=st,charge=ch,sparse_cache=sc,polarity=pol_f32,
                            freq=w_freq,phase=w_phase,rho=rho)
                        logits=np.dot(bp_out,ch[H-OUT_DIM:])
                        if np.argmax(logits)==s[i+1]:cor+=1
                        tot+=1
                    ea_list.append(cor/tot if tot else 0)
                ea=np.mean(ea_list)
                if ea>best:best=ea
                history.append({'step':step,'eval':float(ea),'best':float(best)})
                if step%200==0:
                    extra=""
                    if wave_mode=='learnable' and w_freq is not None:
                        extra=f" freq={w_freq.mean():.3f}[{w_freq.min():.2f},{w_freq.max():.2f}] phase_std={w_phase.std():.2f} wacc={wave_acc}"
                    print(f"    [{step:4d}] eval={ea*100:.1f}% best={best*100:.1f}% "
                          f"theta={theta.mean():.1f} edges={mask.sum()} acc={acc}{extra} {el:.0f}s")
                sys.stdout.flush()
        pool.terminate();pool.join()

        elapsed=time.time()-t0
        res_entry={'mode':mk,'label':label,'best':float(best),'acc':acc,'wave_acc':wave_acc,'time':elapsed}

        # Save freq/phase convergence data for learnable mode
        if wave_mode=='learnable' and w_freq is not None:
            res_entry['freq_mean']=float(w_freq.mean())
            res_entry['freq_std']=float(w_freq.std())
            res_entry['freq_min']=float(w_freq.min())
            res_entry['freq_max']=float(w_freq.max())
            res_entry['phase_mean']=float(w_phase.mean())
            res_entry['phase_std']=float(w_phase.std())

            # Detailed distribution
            print(f"\n    FREQ convergence (mode C):")
            print(f"      mean={w_freq.mean():.4f} std={w_freq.std():.4f} [{w_freq.min():.3f}, {w_freq.max():.3f}]")
            fh,fb=np.histogram(w_freq,bins=15,range=(0.5,2.0))
            for i in range(15):
                bar='#'*int(fh[i]/max(fh.max(),1)*40)
                print(f"      [{fb[i]:.2f},{fb[i+1]:.2f}) {fh[i]:3d} |{bar}")

            print(f"\n    PHASE convergence (mode C):")
            print(f"      mean={w_phase.mean():.4f} std={w_phase.std():.4f} [{w_phase.min():.3f}, {w_phase.max():.3f}]")
            ph,pb=np.histogram(w_phase,bins=16,range=(0,2*np.pi))
            for i in range(16):
                bar='#'*int(ph[i]/max(ph.max(),1)*40)
                lo_pi=pb[i]/np.pi;hi_pi=pb[i+1]/np.pi
                print(f"      [{lo_pi:.2f}pi,{hi_pi:.2f}pi) {ph[i]:3d} |{bar}")

            # Top clusters
            fr_rd=np.round(w_freq,1)
            uv,uc=np.unique(fr_rd,return_counts=True)
            top=sorted(zip(uc,uv),reverse=True)[:5]
            print(f"\n    Top freq clusters: {', '.join(f'{v:.1f}={c}' for c,v in top)}")

            # freq-phase correlation
            corr=np.corrcoef(w_freq,w_phase)[0,1]
            print(f"    freq-phase correlation: r={corr:.4f}")

            # Save checkpoint
            np.savez_compressed(os.path.join(BASE_DIR,f"data/freq_phase_trained_C.npz"),
                freq=w_freq,phase=w_phase,theta=theta,
                mask_rows=np.where(mask)[0].astype(np.uint16),
                mask_cols=np.where(mask)[1].astype(np.uint16))
            print(f"    Saved freq_phase_trained_C.npz")

        results.append(res_entry)
        print(f"    DONE: {label} best={best*100:.1f}% {elapsed:.0f}s")

        # Archive
        try:
            save_experiment(f"freq_phase_{mk}",
                mask=mask,theta=theta,
                metadata={'mode':mk,'label':label,'best':float(best),'wave_acc':wave_acc,
                          'schedule':schedule,'wave_mode':wave_mode},
                extra_arrays={'freq':w_freq,'phase':w_phase} if w_freq is not None else {})
        except Exception as e:
            print(f"    Archive: {e}")

    print(f"\n{'='*70}")
    print(f"  FREQ+PHASE RESULTS")
    print(f"{'='*70}")
    for r in results:
        extra=""
        if 'freq_mean' in r:
            extra=f" freq={r['freq_mean']:.3f}+-{r['freq_std']:.3f}"
        print(f"  {r['mode']}: {r['label']:30s} best={r['best']*100:5.1f}% wacc={r['wave_acc']:3d}{extra}")
    print(f"{'='*70}")

    # Save results JSON
    with open(os.path.join(BASE_DIR,"data/freq_phase_results.json"),"w") as f:
        json.dump(results,f,indent=2)
    print(f"  Saved freq_phase_results.json")
