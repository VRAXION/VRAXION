"""
MLP-controlled adaptive schedule
==================================
A tiny MLP (8->16->6) learns which mutation op to use based on
current training state. Online learning with SGD after each step.

Features: theta_mean, edge_count, accuracy, progress, recent_accept_per_op
Output: softmax probability over 6 ops
Reward: accepted delta (positive) or small penalty (negative)
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

# === TINY MLP: 8 -> 16 -> 6 ===
class TinyMLP:
    def __init__(self, lr=0.01):
        self.lr = lr
        # Xavier init
        self.W1 = np.random.randn(8, 16).astype(np.float32) * 0.35
        self.b1 = np.zeros(16, np.float32)
        self.W2 = np.random.randn(16, N_OPS).astype(np.float32) * 0.25
        self.b2 = np.zeros(N_OPS, np.float32)

    def forward(self, x):
        """Forward pass, return probs and cache for backprop."""
        self.x = x.astype(np.float32)
        self.h = np.maximum(0, self.x @ self.W1 + self.b1)  # ReLU
        logits = self.h @ self.W2 + self.b2
        # Softmax with temperature
        logits = logits - logits.max()
        exp_l = np.exp(logits)
        self.probs = exp_l / exp_l.sum()
        return self.probs

    def update(self, action_idx, reward):
        """REINFORCE-style update: increase prob of good actions."""
        # Policy gradient: d_log_prob * reward
        grad_logits = -self.probs.copy()
        grad_logits[action_idx] += 1.0  # d_log_softmax
        grad_logits *= reward  # scale by reward

        # Backprop through W2
        dW2 = np.outer(self.h, grad_logits)
        db2 = grad_logits

        # Backprop through ReLU + W1
        dh = grad_logits @ self.W2.T
        dh[self.h <= 0] = 0  # ReLU grad
        dW1 = np.outer(self.x, dh)
        db1 = dh

        # SGD update
        self.W2 += self.lr * dW2
        self.b2 += self.lr * db2
        self.W1 += self.lr * dW1
        self.b1 += self.lr * db1

    def get_features(self, theta, mask, accuracy, step, recent_accepts):
        """Build 8-dim feature vector."""
        return np.array([
            theta.mean() / 15.0,            # theta progress [0-1]
            mask.sum() / (H*H*0.1),         # edge density normalized
            accuracy,                         # current accuracy [0-1]
            step / 2000.0,                   # training progress [0-1]
            recent_accepts.get('add', 0),     # recent accept rates
            recent_accepts.get('flip', 0),
            recent_accepts.get('theta', 0),
            recent_accepts.get('channel', 0),
        ], dtype=np.float32)

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

def _eval_bigram(mask, theta, channel, polarity_f, seqs):
    sc = SelfWiringGraph.build_sparse_cache(mask); total = 0.0
    for tb in seqs:
        state=np.zeros(H,np.float32);charge=np.zeros(H,np.float32);s=0.0;n=0
        for i in range(len(tb)-1):
            inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[tb[i]]
            state,charge=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                state=state,charge=charge,sparse_cache=sc,polarity=polarity_f,
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
    print(f"  MLP SCHEDULE CONTROLLER at H={H}")
    print(f"  MLP: 8 -> 16 (ReLU) -> 6 (softmax)")
    print(f"  ops: {ALL_OPS}")
    print(f"  baselines: fixed=23.8%, adaptive=12.5%")
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

    mlp = TinyMLP(lr=0.005)
    recent_accepts = {op: 0.5 for op in ALL_OPS}  # start with 50% prior
    current_accuracy = 0.0

    init_w(bp_out, ALL_DATA, bigram, pol_f32)
    pool = Pool(N_WORKERS, initializer=init_w,
        initargs=(bp_out, ALL_DATA, bigram, pol_f32))

    best=0;acc=0;t0=time.time()
    op_counts = {op: 0 for op in ALL_OPS}
    op_accepts = {op: 0 for op in ALL_OPS}
    # Rolling window for recent accepts (last 20 per op)
    op_history = {op: [] for op in ALL_OPS}

    for step in range(1, 2001):
        # MLP chooses op
        features = mlp.get_features(theta, mask, current_accuracy, step, recent_accepts)
        probs = mlp.forward(features)

        # Epsilon-greedy: 10% random exploration
        if random.random() < 0.1:
            pt = random.choice(ALL_OPS)
        else:
            pt = ALL_OPS[np.random.choice(N_OPS, p=probs)]

        # Force add first if empty
        if mask.sum()==0: pt='add'
        op_counts[pt] += 1

        args=[(mask.flatten(),theta.copy(),channel.copy(),pol_f32.copy(),
               1000+step*50+w,pt) for w in range(N_WORKERS)]
        res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])

        # Apply mutation if accepted
        accepted = br['delta'] > THRESHOLD
        if accepted:
            if br['new_mask_flat'] is not None: mask[:]=br['new_mask_flat'].reshape(H,H)
            if br['new_theta'] is not None: theta[:]=br['new_theta']
            if br['new_channel'] is not None: channel[:]=br['new_channel']
            if br['new_pol_f'] is not None: pol_f32[:]=br['new_pol_f']
            acc+=1
            op_accepts[br['type']] += 1

        # Update MLP with reward
        action_idx = ALL_OPS.index(pt)
        reward = br['delta'] * 100 if accepted else -0.01  # scale up positive, small negative
        mlp.update(action_idx, reward)

        # Update rolling accept rate
        op_history[pt].append(1.0 if accepted else 0.0)
        if len(op_history[pt]) > 20:
            op_history[pt] = op_history[pt][-20:]
        recent_accepts[pt] = np.mean(op_history[pt]) if op_history[pt] else 0.5

        # Eval
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
                    logits=np.dot(bp_out,ch[H-OUT_DIM:])
                    if np.argmax(logits)==s[i+1]:cor+=1
                    tot+=1
                ea_list.append(cor/tot if tot else 0)
            ea=np.mean(ea_list)
            current_accuracy = ea
            if ea>best:best=ea
            if step%200==0:
                probs_str = ' '.join(f"{ALL_OPS[i]}={probs[i]:.0%}" for i in range(N_OPS))
                rates_str = ' '.join(f"{op}={recent_accepts[op]:.0%}" for op in ALL_OPS)
                cnt_str = ' '.join(f"{op}={op_counts[op]}" for op in ALL_OPS)
                print(f"  [{step:4d}] eval={ea*100:.1f}% best={best*100:.1f}% th={theta.mean():.1f}")
                print(f"    MLP probs: {probs_str}")
                print(f"    Accept rates: {rates_str}")
                print(f"    Op counts: {cnt_str}")
                print(f"    {time.time()-t0:.0f}s")
                sys.stdout.flush()

    pool.terminate();pool.join()
    elapsed=time.time()-t0

    print(f"\n{'='*60}")
    print(f"  MLP SCHEDULE RESULTS")
    print(f"{'='*60}")
    print(f"  best={best*100:.1f}% theta={theta.mean():.1f} {elapsed:.0f}s")
    print(f"  Op counts: {dict(op_counts)}")
    print(f"  Op accepts: {dict(op_accepts)}")
    print(f"  Accept rates: {' '.join(f'{op}={op_accepts[op]/max(op_counts[op],1)*100:.0f}%' for op in ALL_OPS)}")
    print(f"  Final MLP probs: {' '.join(f'{ALL_OPS[i]}={probs[i]:.0%}' for i in range(N_OPS))}")
    print(f"  baselines: fixed=23.8%, adaptive=12.5%")
