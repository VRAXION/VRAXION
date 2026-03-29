"""
Big ReLU controller: significantly larger network
==================================================
10 -> 128 -> 128 -> 128 -> 6 = ~33K params
REINFORCE with baseline, proper deterministic eval.
No A/B on controller (too noisy) — just gradient-based learning.
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

class BigReLUController:
    """10->128->128->128->6, ReLU, REINFORCE+baseline."""
    def __init__(self, seed=42, lr=0.001):
        rng = np.random.RandomState(seed)
        self.lr = lr
        # Xavier init scaled for ReLU
        self.W1 = rng.randn(10, 128).astype(np.float32) * np.sqrt(2.0/10)
        self.b1 = np.zeros(128, np.float32)
        self.W2 = rng.randn(128, 128).astype(np.float32) * np.sqrt(2.0/128)
        self.b2 = np.zeros(128, np.float32)
        self.W3 = rng.randn(128, 128).astype(np.float32) * np.sqrt(2.0/128)
        self.b3 = np.zeros(128, np.float32)
        self.Wo = rng.randn(128, N_OPS).astype(np.float32) * 0.1
        self.bo = np.zeros(N_OPS, np.float32)
        self.reward_baseline = 0.0
        n_params = 10*128+128 + 128*128+128 + 128*128+128 + 128*6+6
        print(f"  BigReLU: {n_params} params")

    def forward(self, features):
        self.x = features.astype(np.float32)
        self.z1 = self.x @ self.W1 + self.b1
        self.h1 = np.maximum(0, self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2
        self.h2 = np.maximum(0, self.z2)
        self.z3 = self.h2 @ self.W3 + self.b3
        self.h3 = np.maximum(0, self.z3)
        logits = self.h3 @ self.Wo + self.bo
        # Temperature scaling — start warm (explore), cool down
        logits -= logits.max()
        exp_l = np.exp(logits)
        self.probs = exp_l / (exp_l.sum() + 1e-8)
        return self.probs

    def update(self, action_idx, reward):
        advantage = reward - self.reward_baseline
        self.reward_baseline = 0.95 * self.reward_baseline + 0.05 * reward

        grad_logits = -self.probs.copy()
        grad_logits[action_idx] += 1.0
        grad_logits *= advantage

        # Backprop output
        dWo = np.outer(self.h3, grad_logits)
        dbo = grad_logits
        dh3 = grad_logits @ self.Wo.T

        # Layer 3
        dz3 = dh3 * (self.z3 > 0).astype(np.float32)
        dW3 = np.outer(self.h2, dz3)
        db3 = dz3
        dh2 = dz3 @ self.W3.T

        # Layer 2
        dz2 = dh2 * (self.z2 > 0).astype(np.float32)
        dW2 = np.outer(self.h1, dz2)
        db2 = dz2
        dh1 = dz2 @ self.W2.T

        # Layer 1
        dz1 = dh1 * (self.z1 > 0).astype(np.float32)
        dW1 = np.outer(self.x, dz1)
        db1 = dz1

        # SGD with gradient clipping
        for param, grad in [(self.Wo,dWo),(self.bo,dbo),
                            (self.W3,dW3),(self.b3,db3),
                            (self.W2,dW2),(self.b2,db2),
                            (self.W1,dW1),(self.b1,db1)]:
            np.clip(grad, -0.5, 0.5, out=grad)
            param += self.lr * grad

    def get_features(self, accuracy, pain, reward_sig, tool_rates, progress):
        return np.array([
            accuracy, pain, reward_sig,
            tool_rates.get('add', 0.5),
            tool_rates.get('flip', 0.5),
            tool_rates.get('theta', 0.5),
            tool_rates.get('channel', 0.5),
            tool_rates.get('reverse', 0.5),
            tool_rates.get('remove', 0.5),
            progress,
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
    print(f"  BIG ReLU CONTROLLER (10->128->128->128->6)")
    print(f"  baselines: fixed=23.8%, Swish=18.8%")
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

    ctrl = BigReLUController(seed=99, lr=0.001)
    pain = 0.0; reward_sig = 0.0; current_accuracy = 0.0
    tool_rates = {op: 0.5 for op in ALL_OPS}
    op_history = {op: [] for op in ALL_OPS}

    init_w(bp_out, ALL_DATA, bigram, pol_f32)
    pool = Pool(N_WORKERS, initializer=init_w,
        initargs=(bp_out, ALL_DATA, bigram, pol_f32))

    best=0;acc=0;t0=time.time()
    op_counts = {op: 0 for op in ALL_OPS}
    op_accepts = {op: 0 for op in ALL_OPS}

    for step in range(1, 5001):
        progress = step / 5000.0
        features = ctrl.get_features(current_accuracy, pain, reward_sig, tool_rates, progress)
        probs = ctrl.forward(features)

        eps = 0.10 if step < 500 else 0.05
        if random.random() < eps:
            pt = random.choice(ALL_OPS)
        else:
            pt = ALL_OPS[np.random.choice(N_OPS, p=probs)]

        if mask.sum()==0: pt='add'
        op_counts[pt] += 1

        args=[(mask.flatten(),theta.copy(),channel.copy(),pol_f32.copy(),
               1000+step*50+w,pt) for w in range(N_WORKERS)]
        res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])

        accepted = br['delta'] > THRESHOLD
        if accepted:
            if br['new_mask_flat'] is not None: mask[:]=br['new_mask_flat'].reshape(H,H)
            if br['new_theta'] is not None: theta[:]=br['new_theta']
            if br['new_channel'] is not None: channel[:]=br['new_channel']
            if br['new_pol_f'] is not None: pol_f32[:]=br['new_pol_f']
            acc+=1; op_accepts[br['type']]+=1
            reward = min(br['delta'] * 50, 1.0)
            pain = 0.0; reward_sig = reward
        else:
            pain = 0.3; reward_sig = 0.0
            reward = -0.05

        action_idx = ALL_OPS.index(pt)
        ctrl.update(action_idx, reward)

        op_history[pt].append(1.0 if accepted else 0.0)
        if len(op_history[pt]) > 100: op_history[pt] = op_history[pt][-100:]
        tool_rates[pt] = np.mean(op_history[pt]) if op_history[pt] else 0.5

        pain *= 0.5; reward_sig *= 0.5

        if step%EVAL_EVERY==0:
            ea_list=[]
            for s in eval_seqs:
                sc=SelfWiringGraph.build_sparse_cache(mask)
                st=np.zeros(H,np.float32);ch=np.zeros(H,np.float32);cor=0;tot=0
                for i in range(len(s)-1):
                    inj=np.zeros(H,np.float32);inj[0:IN_DIM]=BP_IN[s[i]]
                    st,ch=SelfWiringGraph.rollout_token(inj,mask=mask,theta=theta,
                        decay=np.float32(0.16),ticks=TICKS,input_duration=INPUT_DURATION,
                        state=st,charge=ch,sparse_cache=sc,polarity=pol_f32,channel=channel)
                    logits_e=np.dot(bp_out,ch[H-OUT_DIM:])
                    if np.argmax(logits_e)==s[i+1]:cor+=1
                    tot+=1
                ea_list.append(cor/tot if tot else 0)
            ea=np.mean(ea_list)
            current_accuracy = ea
            if ea>best:best=ea
            if step%500==0:
                probs_str = ' '.join(f"{ALL_OPS[i]}={probs[i]:.0%}" for i in range(N_OPS))
                rates_str = ' '.join(f"{op}={tool_rates[op]:.0%}" for op in ALL_OPS)
                top_op = ALL_OPS[np.argmax(probs)]
                top_pct = probs.max()*100
                print(f"  [{step:5d}] eval={ea*100:.1f}% best={best*100:.1f}% th={theta.mean():.1f}")
                print(f"    Probs: {probs_str}  TOP: {top_op}({top_pct:.0f}%)")
                print(f"    Rates: {rates_str}")
                print(f"    Baseline: {ctrl.reward_baseline:.4f}")
                print(f"    {time.time()-t0:.0f}s")
                sys.stdout.flush()

    pool.terminate();pool.join()
    elapsed=time.time()-t0

    print(f"\n{'='*60}")
    print(f"  BIG ReLU RESULTS")
    print(f"{'='*60}")
    print(f"  best={best*100:.1f}% theta={theta.mean():.1f} {elapsed:.0f}s")
    print(f"  Ops: {dict(op_counts)}")
    print(f"  Accepts: {dict(op_accepts)}")
    rates = ' '.join(f"{op}={op_accepts[op]/max(op_counts[op],1)*100:.0f}%" for op in ALL_OPS)
    print(f"  Accept rates: {rates}")
    print(f"  Final probs: {' '.join(f'{ALL_OPS[i]}={probs[i]:.0%}' for i in range(N_OPS))}")

    # Save controller weights + sparsity analysis
    ckpt_dir = os.path.join(BASE_DIR, "data")
    np.savez_compressed(os.path.join(ckpt_dir, "big_relu_ctrl_weights.npz"),
        W1=ctrl.W1, b1=ctrl.b1, W2=ctrl.W2, b2=ctrl.b2,
        W3=ctrl.W3, b3=ctrl.b3, Wo=ctrl.Wo, bo=ctrl.bo)
    np.savez_compressed(os.path.join(ckpt_dir, "big_relu_main_network.npz"),
        mask=mask, theta=theta, channel=channel, pol_f32=pol_f32)

    # Sparsity analysis
    print(f"\n  SPARSITY ANALYSIS:")
    for name, W in [('W1',ctrl.W1),('W2',ctrl.W2),('W3',ctrl.W3),('Wo',ctrl.Wo)]:
        total = W.size
        near_zero = (np.abs(W) < 0.01).sum()
        small = (np.abs(W) < 0.1).sum()
        large = (np.abs(W) > 0.5).sum()
        print(f"    {name} {W.shape}: zero(<0.01)={near_zero}/{total}({near_zero/total*100:.0f}%) "
              f"small(<0.1)={small/total*100:.0f}% large(>0.5)={large/total*100:.0f}%")

    # Dead ReLU analysis
    print(f"\n  DEAD RELU ANALYSIS:")
    test_features = np.random.rand(100, 10).astype(np.float32)
    for i in range(100):
        ctrl.forward(test_features[i])
    # Check which neurons never activated
    activations = np.zeros((3, 128))
    for i in range(100):
        ctrl.forward(test_features[i])
        activations[0] += (ctrl.h1 > 0).astype(float)
        activations[1] += (ctrl.h2 > 0).astype(float)
        activations[2] += (ctrl.h3 > 0).astype(float)
    for layer, name in enumerate(['h1','h2','h3']):
        dead = (activations[layer] == 0).sum()
        active = (activations[layer] > 0).sum()
        print(f"    {name}: {dead}/128 dead ({dead/128*100:.0f}%), {active} active")

    print(f"\n  Saved: big_relu_ctrl_weights.npz + big_relu_main_network.npz")
    print(f"  baselines: fixed=23.8%, Swish=18.8%")
