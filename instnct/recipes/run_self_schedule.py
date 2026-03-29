"""
Self-scheduling: network decides its own mutations
====================================================
Extra inputs: accuracy + pain/reward feedback
Extra outputs: 6 op neurons, highest charge wins

Feed cycle:
  1. Normal byte prediction (8 ticks)
  2. Read 6 "op neurons" charge -> pick op
  3. Try mutation -> accept/reject
  4. Feed pain(reject) or reward(accept) into next forward pass
"""
import sys, os, time, random
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
WAVE_LUT = SelfWiringGraph.WAVE_LUT

ALL_OPS = ['add','flip','theta','channel','reverse','remove']
N_OPS = len(ALL_OPS)

# Designate 6 "op neurons" in the output zone (last 6 before readout)
# These neurons' charge after forward pass = "vote" for each op
OP_NEURONS = list(range(H - OUT_DIM - N_OPS, H - OUT_DIM))  # 6 neurons just before output zone

# Designate 2 "feedback neurons" in the input zone
PAIN_NEURON = 0       # fires on reject
REWARD_NEURON = 1     # fires on accept

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
_feedback_pain=None;_feedback_reward=None

def init_w(bpo,data,bg,pol,fp,fr):
    global _bp_out,_all_data,_bigram,_pol,_feedback_pain,_feedback_reward
    _bp_out=bpo;_all_data=data;_bigram=bg;_pol=pol;_feedback_pain=fp;_feedback_reward=fr

def _eval_and_vote(mask, theta, channel, polarity_f, seqs, feedback_pain, feedback_reward):
    """Run forward pass, return accuracy AND op neuron votes."""
    sc = SelfWiringGraph.build_sparse_cache(mask)
    total = 0.0
    op_charges = np.zeros(N_OPS, np.float32)
    n_tokens = 0

    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32);s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            # Inject pain/reward feedback
            inj[PAIN_NEURON] += feedback_pain
            inj[REWARD_NEURON] += feedback_reward
            state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                state=state,charge=charge,sparse_cache=sc,polarity=polarity_f,
                channel=channel)
            logits=np.dot(_bp_out,charge[H-OUT_DIM:])
            e=np.exp(logits-logits.max());pred=e/e.sum()
            tgt=_bigram[tb[i]]
            cos=np.dot(pred,tgt)/(np.linalg.norm(pred)*np.linalg.norm(tgt)+1e-8)
            s+=cos;n+=1
            # Accumulate op neuron charges
            op_charges += charge[OP_NEURONS]
            n_tokens += 1
        total+=s/n if n else 0

    accuracy = total/len(seqs)
    avg_op_charges = op_charges / max(n_tokens, 1)
    return accuracy, avg_op_charges

def worker_eval(args):
    mf,theta,channel,pol_f,pain,reward,seed,pt=args
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
    old_acc, _ = _eval_and_vote(mask,theta,channel,pol_f,seqs,pain,reward)
    new_acc, _ = _eval_and_vote(nm,nt,nc,npf,seqs,pain,reward)
    return{'delta':float(new_acc-old_acc),'type':pt,
           'new_mask_flat':nm.flatten() if new_acc>old_acc else None,
           'new_theta':nt if pt=='theta' else None,
           'new_channel':nc if pt=='channel' else None,
           'new_pol_f':npf if pt=='flip' else None}

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

    print(f"\n{'='*60}")
    print(f"  SELF-SCHEDULING NETWORK at H={H}")
    print(f"  Op neurons: {OP_NEURONS}")
    print(f"  Pain neuron: {PAIN_NEURON}, Reward neuron: {REWARD_NEURON}")
    print(f"  ops: {ALL_OPS}")
    print(f"  baselines: fixed=23.8%, MLP=13.7%")
    print(f"{'='*60}")
    sys.stdout.flush()

    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
    pol_f32 = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)
    irng = np.random.RandomState(42)
    mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask, False)
    theta = np.full(H, 1.0, np.float32)
    nrng = np.random.RandomState(42)
    channel = nrng.randint(1, 9, size=H).astype(np.uint8)

    # Set globals for main thread too
    init_w(bp_out, ALL_DATA, bigram, pol_f32, 0.0, 0.0)

    # Feedback state
    pain = np.float32(0.0)
    reward = np.float32(0.0)
    PAIN_STRENGTH = 5.0
    REWARD_STRENGTH = 5.0
    DECAY_RATE = 0.8

    init_w(bp_out, ALL_DATA, bigram, pol_f32, pain, reward)
    pool = Pool(N_WORKERS, initializer=init_w,
        initargs=(bp_out, ALL_DATA, bigram, pol_f32, pain, reward))

    best=0;acc=0;t0=time.time()
    op_counts = {op: 0 for op in ALL_OPS}
    op_accepts = {op: 0 for op in ALL_OPS}
    vote_history = []

    # First: 50 steps with round-robin (bootstrap before network can vote)
    BOOTSTRAP = 50
    for step in range(1, BOOTSTRAP+1):
        pt = ALL_OPS[(step-1) % N_OPS]
        if mask.sum()==0: pt='add'
        args=[(mask.flatten(),theta.copy(),channel.copy(),pol_f32.copy(),
               pain,reward,1000+step*50+w,pt) for w in range(N_WORKERS)]
        res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])
        if br['delta']>THRESHOLD:
            if br['new_mask_flat'] is not None: mask[:]=br['new_mask_flat'].reshape(H,H)
            if br['new_theta'] is not None: theta[:]=br['new_theta']
            if br['new_channel'] is not None: channel[:]=br['new_channel']
            if br['new_pol_f'] is not None: pol_f32[:]=br['new_pol_f']
            reward = REWARD_STRENGTH; pain = 0.0
            op_accepts[br['type']] += 1
        else:
            pain = PAIN_STRENGTH; reward = 0.0
        op_counts[pt] += 1
    print(f"  Bootstrap done ({BOOTSTRAP} steps)")
    sys.stdout.flush()

    # Main loop: network votes
    for step in range(BOOTSTRAP+1, 2001):
        # Get network's vote via forward pass
        off = random.randint(0,len(ALL_DATA)-15)
        vote_seqs = [ALL_DATA[off:off+15]]  # short seq for speed
        try:
            _, op_charges = _eval_and_vote(mask, theta, channel, pol_f32, vote_seqs, pain, reward)
        except Exception as e:
            print(f"  VOTE ERROR: {e}")
            op_charges = np.ones(N_OPS, np.float32)  # fallback equal

        # Softmax over op charges (with temperature)
        temp = 2.0
        logits = op_charges / temp
        logits -= logits.max()
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum()

        # 20% exploration
        if random.random() < 0.2:
            pt = random.choice(ALL_OPS)
        else:
            pt = ALL_OPS[np.random.choice(N_OPS, p=probs)]

        if mask.sum()==0: pt='add'
        op_counts[pt] += 1

        # Re-init workers with current feedback
        args=[(mask.flatten(),theta.copy(),channel.copy(),pol_f32.copy(),
               pain,reward,1000+step*50+w,pt) for w in range(N_WORKERS)]
        res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])

        accepted = br['delta'] > THRESHOLD
        if accepted:
            if br['new_mask_flat'] is not None: mask[:]=br['new_mask_flat'].reshape(H,H)
            if br['new_theta'] is not None: theta[:]=br['new_theta']
            if br['new_channel'] is not None: channel[:]=br['new_channel']
            if br['new_pol_f'] is not None: pol_f32[:]=br['new_pol_f']
            acc+=1
            op_accepts[br['type']] += 1
            reward = REWARD_STRENGTH; pain = 0.0
        else:
            pain = PAIN_STRENGTH; reward = 0.0

        # Decay feedback
        pain *= DECAY_RATE
        reward *= DECAY_RATE

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
                        channel=channel)
                    logits_eval=np.dot(bp_out,ch[H-OUT_DIM:])
                    if np.argmax(logits_eval)==s[i+1]:cor+=1
                    tot+=1
                ea_list.append(cor/tot if tot else 0)
            ea=np.mean(ea_list)
            if ea>best:best=ea
            if step%200==0:
                probs_str = ' '.join(f"{ALL_OPS[i]}={probs[i]:.0%}" for i in range(N_OPS))
                chrg_str = ' '.join(f"{ALL_OPS[i]}={op_charges[i]:.2f}" for i in range(N_OPS))
                print(f"  [{step:4d}] eval={ea*100:.1f}% best={best*100:.1f}% th={theta.mean():.1f}")
                print(f"    Charges: {chrg_str}")
                print(f"    Probs: {probs_str}")
                print(f"    Pain={pain:.1f} Reward={reward:.1f}")
                print(f"    Ops: {dict(op_counts)}  Accepts: {dict(op_accepts)}")
                print(f"    {time.time()-t0:.0f}s")
                sys.stdout.flush()

    pool.terminate();pool.join()
    elapsed=time.time()-t0

    print(f"\n{'='*60}")
    print(f"  SELF-SCHEDULING RESULTS")
    print(f"{'='*60}")
    print(f"  best={best*100:.1f}% theta={theta.mean():.1f} {elapsed:.0f}s")
    print(f"  Ops: {dict(op_counts)}")
    print(f"  Accepts: {dict(op_accepts)}")
    rates = ' '.join(f"{op}={op_accepts[op]/max(op_counts[op],1)*100:.0f}%" for op in ALL_OPS)
    print(f"  Accept rates: {rates}")
    print(f"  baselines: fixed=23.8%, MLP=13.7%, adaptive=12.5%")
