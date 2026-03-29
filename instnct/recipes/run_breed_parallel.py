"""
Parallel breeding: 2 networks evolve independently, breed periodically
========================================================================
Every 500 steps: breed the two, child replaces the weaker parent.
Both use same fix schedule (theta×2, channel×2 = 23.8% baseline).
"""
import sys, os, time, random, copy
import numpy as np
from multiprocessing import Pool
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR)); sys.path.insert(0, str(ROOT_DIR / "model"))
from graph import SelfWiringGraph

H = 256; N_WORKERS = 9; TICKS = 8; INPUT_DURATION = 2  # 9 workers per network (18 total / 2)
THRESHOLD = 0.00005; EVAL_EVERY = 20
INIT_DENSITY = 0.05
PHI = (1 + 5**0.5) / 2
IN_DIM = int(round(H / PHI))
OUT_DIM = int(round(H / PHI))
SDR_K = int(round(IN_DIM * 0.20))
WAVE_LUT = SelfWiringGraph.WAVE_LUT
SCHEDULE = ['add','flip','theta','channel','theta','channel','flip','remove']

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
_bp_out=None;_all_data=None;_bigram=None

def init_w(bpo,data,bg):
    global _bp_out,_all_data,_bigram
    _bp_out=bpo;_all_data=data;_bigram=bg

def _eval_bigram(mask, theta, channel, pol_f, seqs):
    sc = SelfWiringGraph.build_sparse_cache(mask); total = 0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32);s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                state=state,charge=charge,sparse_cache=sc,polarity=pol_f,
                channel=channel)
            logits=np.dot(_bp_out,charge[H-OUT_DIM:])
            e=np.exp(logits-logits.max());pred=e/e.sum()
            tgt=_bigram[tb[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            s+=cos;n+=1
        total+=s/n if n else 0
    return total/len(seqs)

def worker_eval(args):
    mf,theta,channel,pol_f,seed,pt=args
    rng=random.Random(seed);nrng=np.random.RandomState(seed)
    mask=mf.reshape(H,H);nm=mask;nt=theta;nc=channel;npf=pol_f
    if pt=='add':
        r=rng.randint(0,H-1);c=rng.randint(0,H-1)
        if r==c or mask[r,c]:return{'delta':-1e9,'type':'add'}
        nm=mask.copy();nm[r,c]=True
    elif pt=='flip':
        idx=rng.randint(0,H-1)
        pb=(pol_f>0);pb[idx]=not pb[idx]
        npf=np.where(pb,1.0,-1.0).astype(np.float32)
    elif pt=='theta':
        idx=rng.randint(0,H-1);nt=theta.copy();nt[idx]=float(rng.randint(1,15))
    elif pt=='channel':
        idx=rng.randint(0,H-1);nc=channel.copy();nc[idx]=np.uint8(rng.randint(1,8))
    elif pt=='remove':
        alive=list(zip(*np.where(mask)))
        if len(alive)<1:return{'delta':-1e9,'type':'remove'}
        r,c=alive[rng.randint(0,len(alive)-1)];nm=mask.copy();nm[r,c]=False
    seqs=[]
    for _ in range(2):
        off=nrng.randint(0,len(_all_data)-100);seqs.append(_all_data[off:off+100])
    old=_eval_bigram(mask,theta,channel,pol_f,seqs)
    new=_eval_bigram(nm,nt,nc,npf,seqs)
    return{'delta':float(new-old),'type':pt,
           'new_mask_flat':nm.flatten() if new>old else None,
           'new_theta':nt if pt=='theta' else None,
           'new_channel':nc if pt=='channel' else None,
           'new_pol_f':npf if pt=='flip' else None}

class Individual:
    def __init__(self, seed):
        rng = np.random.RandomState(seed)
        ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
        self.pol_f32 = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)
        self.pol_bool = ref.polarity.copy()
        self.mask = (rng.rand(H,H) < INIT_DENSITY).astype(bool)
        np.fill_diagonal(self.mask, False)
        self.theta = np.full(H, 1.0, np.float32)
        self.channel = rng.randint(1, 9, size=H).astype(np.uint8)
        self.best = 0.0
        self.name = f"ind_{seed}"

    @staticmethod
    def breed(parent_a, parent_b, score_a, score_b, eval_fn):
        """Consensus + better parent wins + crystallize pruning."""
        child = Individual.__new__(Individual)
        child.name = f"child_{random.randint(0,999)}"
        better = parent_a if score_a >= score_b else parent_b
        worse = parent_b if score_a >= score_b else parent_a

        # Mask: consensus (AND) + better parent's unique edges
        consensus = parent_a.mask & parent_b.mask  # both agree = keep
        only_better = better.mask & ~worse.mask     # only better has = keep
        child.mask = consensus | only_better
        np.fill_diagonal(child.mask, False)

        # Params: from better parent (proven winner)
        child.theta = better.theta.copy()
        child.channel = better.channel.copy()
        child.pol_bool = better.pol_bool.copy()
        child.pol_f32 = better.pol_f32.copy()
        child.best = 0.0

        n_consensus = consensus.sum()
        n_better_only = only_better.sum()
        n_total = child.mask.sum()
        print(f"    Breed: consensus={n_consensus} + better_only={n_better_only} = {n_total} edges")

        # Fast crystallize: sample 300 random edges, remove if not helpful
        print(f"    Fast crystallize (sample 300)...", end="")
        sys.stdout.flush()
        alive = list(zip(*np.where(child.mask)))
        np.random.shuffle(alive)
        sample = alive[:min(300, len(alive))]
        base_score = eval_fn(child)
        removed = 0
        for r, c in sample:
            child.mask[r, c] = False
            score = eval_fn(child)
            if score >= base_score - 0.0001:
                removed += 1  # keep removed
                base_score = score  # update baseline
            else:
                child.mask[r, c] = True  # restore
        print(f" removed {removed}/300, final={child.mask.sum()} edges")

        return child

def eval_individual(ind, eval_seqs):
    ea_list=[]
    for s in eval_seqs:
        sc=SelfWiringGraph.build_sparse_cache(ind.mask)
        st=np.zeros(H,np.float32);ch=np.zeros(H,np.float32);cor=0;tot=0
        for i in range(len(s)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
            st,ch=SelfWiringGraph.rollout_token(inj,mask=ind.mask,theta=ind.theta,
                decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                state=st,charge=ch,sparse_cache=sc,polarity=ind.pol_f32,
                channel=ind.channel)
            logits=np.dot(bp_out,ch[H-OUT_DIM:])
            if np.argmax(logits)==s[i+1]:cor+=1
            tot+=1
        ea_list.append(cor/tot if tot else 0)
    return np.mean(ea_list)

def train_individual(ind, pool, steps, seed_base):
    """Train one individual for N steps."""
    for step in range(steps):
        pt=SCHEDULE[step%len(SCHEDULE)]
        if ind.mask.sum()==0:pt='add'
        args=[(ind.mask.flatten(),ind.theta.copy(),ind.channel.copy(),ind.pol_f32.copy(),
               seed_base+step*50+w,pt) for w in range(N_WORKERS)]
        res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])
        if br['delta']>THRESHOLD:
            if br['new_mask_flat'] is not None:ind.mask[:]=br['new_mask_flat'].reshape(H,H)
            if br['new_theta'] is not None:ind.theta[:]=br['new_theta']
            if br['new_channel'] is not None:ind.channel[:]=br['new_channel']
            if br['new_pol_f'] is not None:ind.pol_f32[:]=br['new_pol_f'];ind.pol_bool[:]=(ind.pol_f32>0)

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

    BREED_EVERY = 500
    TOTAL_STEPS = 3000
    N_BREEDS = TOTAL_STEPS // BREED_EVERY

    print(f"\n{'='*60}")
    print(f"  PARALLEL BREEDING: 2 individuals, breed every {BREED_EVERY} steps")
    print(f"  Total: {TOTAL_STEPS} steps, {N_BREEDS} breeds")
    print(f"  baseline: single individual = 23.8%")
    print(f"{'='*60}")

    # Two individuals from different seeds
    ind_a = Individual(seed=42)
    ind_b = Individual(seed=77)

    init_w(bp_out, ALL_DATA, bigram)
    pool = Pool(N_WORKERS, initializer=init_w, initargs=(bp_out, ALL_DATA, bigram))

    global_best = 0; t0 = time.time()

    for breed_round in range(N_BREEDS):
        step_base = breed_round * BREED_EVERY

        # Train both for BREED_EVERY steps
        print(f"\n  Round {breed_round+1}/{N_BREEDS}: training both for {BREED_EVERY} steps...")
        sys.stdout.flush()

        train_individual(ind_a, pool, BREED_EVERY, seed_base=1000+step_base*100)
        train_individual(ind_b, pool, BREED_EVERY, seed_base=5000+step_base*100)

        # Eval both
        score_a = eval_individual(ind_a, eval_seqs)
        score_b = eval_individual(ind_b, eval_seqs)
        ind_a.best = max(ind_a.best, score_a)
        ind_b.best = max(ind_b.best, score_b)
        global_best = max(global_best, ind_a.best, ind_b.best)

        # Breed: consensus + better parent + crystallize
        child = Individual.breed(ind_a, ind_b, score_a, score_b,
                                  lambda ind: eval_individual(ind, eval_seqs))
        score_child = eval_individual(child, eval_seqs)

        # Child replaces weaker parent
        if score_a <= score_b:
            weaker = "A"
            if score_child > score_a:
                ind_a = child
                verdict = f"child({score_child*100:.1f}%) replaces A({score_a*100:.1f}%)"
            else:
                verdict = f"child({score_child*100:.1f}%) < A({score_a*100:.1f}%), kept"
        else:
            weaker = "B"
            if score_child > score_b:
                ind_b = child
                verdict = f"child({score_child*100:.1f}%) replaces B({score_b*100:.1f}%)"
            else:
                verdict = f"child({score_child*100:.1f}%) < B({score_b*100:.1f}%), kept"

        global_best = max(global_best, score_child)
        elapsed = time.time() - t0

        print(f"  Step {(breed_round+1)*BREED_EVERY}: A={score_a*100:.1f}% B={score_b*100:.1f}% "
              f"child={score_child*100:.1f}% global_best={global_best*100:.1f}%")
        print(f"    Breed: {verdict}")
        print(f"    A: edges={ind_a.mask.sum()} th={ind_a.theta.mean():.1f}")
        print(f"    B: edges={ind_b.mask.sum()} th={ind_b.theta.mean():.1f}")
        print(f"    {elapsed:.0f}s")
        sys.stdout.flush()

    pool.terminate();pool.join()
    elapsed = time.time()-t0

    print(f"\n{'='*60}")
    print(f"  BREEDING RESULTS")
    print(f"{'='*60}")
    print(f"  global_best={global_best*100:.1f}%")
    print(f"  A best={ind_a.best*100:.1f}% B best={ind_b.best*100:.1f}%")
    print(f"  baseline: single = 23.8%")
    print(f"  {elapsed:.0f}s")
