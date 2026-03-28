"""
A/B: Claude's graph.py vs Gemini's graph.py
============================================
Same test, same params, same data. Only graph.py differs.
Uses SDR input, random output, learnable theta — our best proven setup.

A: main (Claude's graph.py) — with clip, refractory, SDR support
B: Gemini's branch — no clip, no batch refractory, int8 mask
"""
import sys, os, time, random, importlib, shutil
import numpy as np
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# We'll swap graph.py before each run
MAIN_GRAPH = ROOT / "model" / "graph.py"
CLAUDE_GRAPH = ROOT / "model" / "graph_claude_backup.py"
GEMINI_GRAPH = ROOT / "model" / "graph_gemini.py"

H = 256; IN_DIM = 64; IN_K = 13; OUT_DIM = 64
N_WORKERS = 18; TICKS = 8; INPUT_DURATION = 2
THRESHOLD = 0.00005; EVAL_EVERY = 20
SCHEDULE = ['add', 'remove', 'flip', 'theta', 'theta', 'theta', 'decay', 'decay']
INIT_DENSITY = 0.05; THETA_INIT = 1.0

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

_SWG = None  # will hold the SelfWiringGraph class
_all_data=None;_bigram=None;_pol=None;_freq=None;_phase=None;_rho=None

def init_w(data,bg,pol,fr,ph,rh):
    global _all_data,_bigram,_pol,_freq,_phase,_rho,_SWG
    _all_data=data;_bigram=bg;_pol=pol;_freq=fr;_phase=ph;_rho=rh
    # Import whatever graph.py is currently in place
    sys.path.insert(0, str(ROOT / "model"))
    if 'graph' in sys.modules:
        del sys.modules['graph']
    from graph import SelfWiringGraph
    _SWG = SelfWiringGraph

def _eval_bigram(mask, theta, decay, seqs):
    sc = _SWG.build_sparse_cache(mask)
    total = 0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32)
        s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=_SWG.rollout_token(inj,mask=mask,theta=theta,decay=decay,
                ticks=TICKS,input_duration=INPUT_DURATION,state=state,charge=charge,
                sparse_cache=sc,polarity=_pol,freq=_freq,phase=_phase,rho=_rho)
            logits=np.dot(BP_OUT, charge[H-OUT_DIM:])
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
        nt[idx]=rng.uniform(0.0, 16.0)
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

def run_test(label, ALL_DATA, bigram, eval_seqs):
    # Re-import graph module fresh
    sys.path.insert(0, str(ROOT / "model"))
    if 'graph' in sys.modules:
        del sys.modules['graph']
    from graph import SelfWiringGraph

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
    pol = ref.polarity.astype(np.float32)

    irng = np.random.RandomState(42)
    mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool)
    np.fill_diagonal(mask, False)
    theta = np.full(H, THETA_INIT, np.float32)
    decay = np.random.RandomState(99).uniform(0.08, 0.24, H).astype(np.float32)
    print(f"  edges={int(mask.sum())}")

    pool = Pool(N_WORKERS, initializer=init_w,
        initargs=(ALL_DATA, bigram, pol, ref.freq, ref.phase, ref.rho))

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
                        inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
                        st,ch=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,decay=decay,
                            ticks=TICKS,input_duration=INPUT_DURATION,state=st,charge=ch,
                            sparse_cache=sc,polarity=pol,freq=ref.freq,phase=ref.phase,rho=ref.rho)
                        logits=np.dot(BP_OUT,ch[H-OUT_DIM:])
                        if np.argmax(logits)==s[i+1]:cor+=1
                        tot+=1
                    ea_list.append(cor/tot if tot else 0)
                ea=np.mean(ea_list)
                if ea>best_eval: best_eval=ea; stall=0
                else: stall+=EVAL_EVERY

                if step % 200 == 0:
                    print(f"  [{step:4d}] eval={ea*100:.1f}% best={best_eval*100:.1f}% "
                          f"theta={theta.mean():.2f}[{theta.min():.1f},{theta.max():.1f}] "
                          f"t_acc={t_acc} acc={accepts} {el:.0f}s")
                sys.stdout.flush()
                if stall >= 800 and step >= 600:
                    print(f"  ** STALL at step {step}"); break
    finally:
        pool.terminate(); pool.join()

    elapsed=time.time()-t0
    print(f"\n  DONE: best={best_eval*100:.1f}% theta={theta.mean():.2f} acc={accepts} {elapsed:.0f}s")
    return {'label': label, 'best': best_eval, 'theta': float(theta.mean()),
            'acc': accepts, 'time': elapsed}


if __name__ == "__main__":
    # Prepare: extract Gemini's graph.py from their branch
    import subprocess
    print("Extracting graph.py versions...")

    # Save Claude's (current main) graph.py
    shutil.copy2(MAIN_GRAPH, CLAUDE_GRAPH)
    print(f"  Saved Claude's: {CLAUDE_GRAPH}")

    # Extract Gemini's graph.py from their branch
    result = subprocess.run(
        ['git', 'show', 'origin/feature/axonal-delay-v5.0:instnct/model/graph.py'],
        capture_output=True, text=True, cwd=str(ROOT.parent))
    GEMINI_GRAPH.write_text(result.stdout, encoding='utf-8')
    print(f"  Extracted Gemini's: {GEMINI_GRAPH}")

    sys.path.insert(0, str(ROOT))
    from lib.data import load_fineweb_bytes, resolve_fineweb_path
    resolve_fineweb_path(); ALL_DATA = load_fineweb_bytes()
    print(f"Loaded {len(ALL_DATA)/1e6:.1f} MB")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    bigram = np.load(os.path.join(BASE_DIR, "data", "bigram_table.npy"))
    eval_rng = np.random.RandomState(9999)
    eval_seqs = [ALL_DATA[o:o+100] for o in [eval_rng.randint(0,len(ALL_DATA)-100) for _ in range(5)]]

    results = []

    # Run A: Claude's graph.py
    shutil.copy2(CLAUDE_GRAPH, MAIN_GRAPH)
    r = run_test("A: CLAUDE (main) — clip, refractory, SDR", ALL_DATA, bigram, eval_seqs)
    results.append(r)

    # Run B: Gemini's graph.py
    shutil.copy2(GEMINI_GRAPH, MAIN_GRAPH)
    r = run_test("B: GEMINI (axonal-delay) — no clip, no batch refrac", ALL_DATA, bigram, eval_seqs)
    results.append(r)

    # Restore Claude's graph.py
    shutil.copy2(CLAUDE_GRAPH, MAIN_GRAPH)
    # Cleanup
    CLAUDE_GRAPH.unlink(missing_ok=True)
    GEMINI_GRAPH.unlink(missing_ok=True)

    print(f"\n{'='*70}")
    print(f"  CLAUDE vs GEMINI graph.py")
    print(f"{'='*70}")
    for r in results:
        print(f"  {r['label']:55s} best={r['best']*100:5.1f}% theta={r['theta']:.2f} {r['time']:.0f}s")
    best = max(results, key=lambda x: x['best'])
    print(f"\n  WINNER: {best['label']}")
    print(f"{'='*70}")
