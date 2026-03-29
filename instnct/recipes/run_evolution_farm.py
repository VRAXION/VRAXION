"""
Continuous evolution farm: 2 individuals, breed on death, crystallize-as-training
=================================================================================
- 2 individuals evolve in parallel (alternating steps)
- Each lives for MAX_LIFE steps then dies
- On death: score saved to DB, breed top 2 from DB + crystallize schedule
- Crystallize schedule: NO ADD, heavy remove + flip/theta/channel/reverse
- Runs forever (or until stopped)
"""
import sys, os, time, random, json
import numpy as np
from multiprocessing import Pool
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR)); sys.path.insert(0, str(ROOT_DIR / "model"))
from graph import SelfWiringGraph

H = 256; N_WORKERS = 9; TICKS = 8; INPUT_DURATION = 2
THRESHOLD = 0.00005; EVAL_EVERY = 20
INIT_DENSITY = 0.05
PHI = (1 + 5**0.5) / 2
IN_DIM = int(round(H / PHI))
OUT_DIM = int(round(H / PHI))
SDR_K = int(round(IN_DIM * 0.20))
WAVE_LUT = SelfWiringGraph.WAVE_LUT

ALL_OPS = ['add','flip','theta','channel','reverse','remove']
# Normal schedule (building phase)
SCHEDULE_BUILD = ['add','flip','theta','channel','theta','channel','flip','remove']
# Crystallize schedule (pruning phase — NO ADD, heavy remove)
SCHEDULE_CRYSTAL = ['remove','flip','theta','channel','reverse','remove','remove','remove']

MAX_LIFE = 2000  # steps per individual per generation
CRYSTAL_STEPS = 500  # crystallize steps after breed

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
                state=state,charge=charge,sparse_cache=sc,polarity=pol_f,channel=channel)
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
    elif pt=='reverse':
        alive=list(zip(*np.where(mask)))
        if len(alive)<1:return{'delta':-1e9,'type':'reverse'}
        r,c=alive[rng.randint(0,len(alive)-1)]
        if r==c or mask[c,r]:return{'delta':-1e9,'type':'reverse'}
        nm=mask.copy();nm[r,c]=False;nm[c,r]=True
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
    def __init__(self, name, mask, theta, channel, pol_f32):
        self.name = name
        self.mask = mask.copy()
        self.theta = theta.copy()
        self.channel = channel.copy()
        self.pol_f32 = pol_f32.copy()
        self.age = 0
        self.best = 0.0

    @staticmethod
    def from_random(name, seed):
        rng = np.random.RandomState(seed)
        ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
        mask = (rng.rand(H,H) < INIT_DENSITY).astype(bool)
        np.fill_diagonal(mask, False)
        return Individual(name, mask, np.full(H, 1.0, np.float32),
                         rng.randint(1,9,size=H).astype(np.uint8),
                         np.where(ref.polarity, 1.0, -1.0).astype(np.float32))

    @staticmethod
    def breed_consensus(name, parent_a, parent_b, score_a, score_b):
        better = parent_a if score_a >= score_b else parent_b
        worse = parent_b if score_a >= score_b else parent_a
        consensus = parent_a.mask & parent_b.mask
        only_better = better.mask & ~worse.mask
        mask = consensus | only_better
        np.fill_diagonal(mask, False)
        n_con = consensus.sum(); n_bet = only_better.sum()
        print(f"    Breed: consensus={n_con} + better_only={n_bet} = {mask.sum()} edges")
        return Individual(name, mask, better.theta, better.channel, better.pol_f32)

    def save(self, path):
        np.savez_compressed(path, mask=self.mask, theta=self.theta,
                           channel=self.channel, pol_f32=self.pol_f32)

    @staticmethod
    def load(name, path):
        d = np.load(path)
        return Individual(name, d['mask'], d['theta'], d['channel'], d['pol_f32'])

def eval_individual(ind, eval_seqs):
    ea_list=[]
    for s in eval_seqs:
        sc=SelfWiringGraph.build_sparse_cache(ind.mask)
        st=np.zeros(H,np.float32);ch=np.zeros(H,np.float32);cor=0;tot=0
        for i in range(len(s)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
            st,ch=SelfWiringGraph.rollout_token(inj,mask=ind.mask,theta=ind.theta,
                decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                state=st,charge=ch,sparse_cache=sc,polarity=ind.pol_f32,channel=ind.channel)
            logits=np.dot(bp_out,ch[H-OUT_DIM:])
            if np.argmax(logits)==s[i+1]:cor+=1
            tot+=1
        ea_list.append(cor/tot if tot else 0)
    return np.mean(ea_list)

def train_steps(ind, pool, n_steps, schedule, seed_base):
    for step in range(n_steps):
        pt = schedule[step % len(schedule)]
        if pt in ('remove','reverse','flip','theta','channel') and ind.mask.sum() < 100: pt='add'
        if ind.mask.sum() == 0: pt='add'
        args=[(ind.mask.flatten(),ind.theta.copy(),ind.channel.copy(),ind.pol_f32.copy(),
               seed_base+step*50+w,pt) for w in range(N_WORKERS)]
        res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])
        if br['delta']>THRESHOLD:
            if br['new_mask_flat'] is not None:ind.mask[:]=br['new_mask_flat'].reshape(H,H)
            if br['new_theta'] is not None:ind.theta[:]=br['new_theta']
            if br['new_channel'] is not None:ind.channel[:]=br['new_channel']
            if br['new_pol_f'] is not None:ind.pol_f32[:]=br['new_pol_f']
        ind.age += 1

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

    DB_PATH = os.path.join(BASE_DIR, "data", "evolution_db.json")
    CKPT_DIR = os.path.join(BASE_DIR, "data", "evo_checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)

    # Load or init DB
    if os.path.exists(DB_PATH):
        with open(DB_PATH) as f: db = json.load(f)
    else:
        db = {'individuals': [], 'generation': 0, 'global_best': 0.0}

    print(f"\n{'='*60}")
    print(f"  EVOLUTION FARM: continuous breeding")
    print(f"  MAX_LIFE={MAX_LIFE} CRYSTAL={CRYSTAL_STEPS}")
    print(f"  Build: {SCHEDULE_BUILD}")
    print(f"  Crystal: {SCHEDULE_CRYSTAL}")
    print(f"  DB: {len(db['individuals'])} individuals, gen={db['generation']}")
    print(f"{'='*60}")
    sys.stdout.flush()

    init_w(bp_out, ALL_DATA, bigram)
    pool = Pool(N_WORKERS, initializer=init_w, initargs=(bp_out, ALL_DATA, bigram))

    # Init or breed first 2 individuals
    def spawn_individual(gen):
        if len(db['individuals']) >= 2:
            # Breed from top 2
            sorted_db = sorted(db['individuals'], key=lambda x: x['score'], reverse=True)
            p1_path = os.path.join(CKPT_DIR, sorted_db[0]['file'])
            p2_path = os.path.join(CKPT_DIR, sorted_db[1]['file'])
            if os.path.exists(p1_path) and os.path.exists(p2_path):
                parent_a = Individual.load("pa", p1_path)
                parent_b = Individual.load("pb", p2_path)
                name = f"gen{gen:03d}_{random.randint(0,999)}"
                child = Individual.breed_consensus(name, parent_a, parent_b,
                                                    sorted_db[0]['score'], sorted_db[1]['score'])
                # Crystallize phase
                print(f"    Crystallize {CRYSTAL_STEPS} steps...")
                sys.stdout.flush()
                train_steps(child, pool, CRYSTAL_STEPS, SCHEDULE_CRYSTAL, seed_base=gen*10000)
                return child
        # Fallback: random init
        name = f"gen{gen:03d}_rnd"
        return Individual.from_random(name, seed=gen*42+7)

    t0 = time.time()
    MAX_GENERATIONS = 20

    for gen in range(db['generation'], db['generation'] + MAX_GENERATIONS):
        db['generation'] = gen + 1
        print(f"\n  === GENERATION {gen+1} ===")

        # Spawn 2 individuals
        ind_a = spawn_individual(gen*2)
        ind_b = spawn_individual(gen*2+1)

        # Train both
        print(f"  Training {ind_a.name}...")
        sys.stdout.flush()
        train_steps(ind_a, pool, MAX_LIFE, SCHEDULE_BUILD, seed_base=gen*100000)
        score_a = eval_individual(ind_a, eval_seqs)
        ind_a.best = score_a

        print(f"  Training {ind_b.name}...")
        sys.stdout.flush()
        train_steps(ind_b, pool, MAX_LIFE, SCHEDULE_BUILD, seed_base=gen*100000+50000)
        score_b = eval_individual(ind_b, eval_seqs)
        ind_b.best = score_b

        # Save both to DB
        for ind, score in [(ind_a, score_a), (ind_b, score_b)]:
            fname = f"{ind.name}.npz"
            ind.save(os.path.join(CKPT_DIR, fname))
            db['individuals'].append({
                'name': ind.name, 'file': fname, 'score': float(score),
                'edges': int(ind.mask.sum()), 'theta_mean': float(ind.theta.mean()),
                'generation': gen+1, 'age': ind.age
            })
            if score > db['global_best']:
                db['global_best'] = float(score)

        # Save DB
        with open(DB_PATH, 'w') as f: json.dump(db, f, indent=2)

        # Report
        elapsed = time.time() - t0
        top3 = sorted(db['individuals'], key=lambda x: x['score'], reverse=True)[:3]
        print(f"\n  Gen {gen+1}: A={score_a*100:.1f}% B={score_b*100:.1f}% "
              f"global_best={db['global_best']*100:.1f}% {elapsed:.0f}s")
        print(f"  Top 3: {', '.join(f'{t['name']}={t['score']*100:.1f}%' for t in top3)}")
        print(f"  DB: {len(db['individuals'])} individuals")
        sys.stdout.flush()

    pool.terminate();pool.join()
    print(f"\n  FARM DONE: {db['generation']} generations, global_best={db['global_best']*100:.1f}%")
