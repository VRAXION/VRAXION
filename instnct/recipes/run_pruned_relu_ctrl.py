"""
Pruned ReLU controller: 128->64 hidden + warm start
=====================================================
Based on sparsity analysis: 58% small weights, 16% dead ReLU.
Halve the network. Warm start output bias toward reverse.

A: Big (10->128->128->128->6, 33K) = 23.6% baseline
B: Pruned (10->64->64->64->6, 8K) = ???
C: Pruned + warm start (reverse bias) = ???
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

class PrunedReLU:
    def __init__(self, hidden=64, seed=42, lr=0.001, warm_start=False):
        rng = np.random.RandomState(seed)
        self.lr = lr; self.hidden = hidden
        self.W1 = rng.randn(10, hidden).astype(np.float32) * np.sqrt(2.0/10)
        self.b1 = np.zeros(hidden, np.float32)
        self.W2 = rng.randn(hidden, hidden).astype(np.float32) * np.sqrt(2.0/hidden)
        self.b2 = np.zeros(hidden, np.float32)
        self.W3 = rng.randn(hidden, hidden).astype(np.float32) * np.sqrt(2.0/hidden)
        self.b3 = np.zeros(hidden, np.float32)
        self.Wo = rng.randn(hidden, N_OPS).astype(np.float32) * 0.1
        self.bo = np.zeros(N_OPS, np.float32)
        if warm_start:
            # Bias output toward reverse (index 4) initially
            self.bo[4] = 1.0  # reverse gets head start
            self.bo[2] = 0.3  # theta gets small boost
        self.reward_baseline = 0.0
        n = 10*hidden+hidden + hidden*hidden+hidden + hidden*hidden+hidden + hidden*6+6
        print(f"  PrunedReLU: hidden={hidden}, {n} params, warm={warm_start}")

    def forward(self, features):
        self.x = features.astype(np.float32)
        self.z1 = self.x @ self.W1 + self.b1
        self.h1 = np.maximum(0, self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2
        self.h2 = np.maximum(0, self.z2)
        self.z3 = self.h2 @ self.W3 + self.b3
        self.h3 = np.maximum(0, self.z3)
        logits = self.h3 @ self.Wo + self.bo
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
        dWo = np.outer(self.h3, grad_logits); dbo = grad_logits
        dh3 = grad_logits @ self.Wo.T
        dz3 = dh3 * (self.z3 > 0).astype(np.float32)
        dW3 = np.outer(self.h2, dz3); db3 = dz3; dh2 = dz3 @ self.W3.T
        dz2 = dh2 * (self.z2 > 0).astype(np.float32)
        dW2 = np.outer(self.h1, dz2); db2 = dz2; dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (self.z1 > 0).astype(np.float32)
        dW1 = np.outer(self.x, dz1); db1 = dz1
        for p, g in [(self.Wo,dWo),(self.bo,dbo),(self.W3,dW3),(self.b3,db3),
                      (self.W2,dW2),(self.b2,db2),(self.W1,dW1),(self.b1,db1)]:
            np.clip(g, -0.5, 0.5, out=g); p += self.lr * g

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

def run_controller(label, ctrl, steps=5000):
    random.seed(42); np.random.seed(42)
    ref = SelfWiringGraph(max(IN_DIM,16), hidden=H, projection_scale=1.0)
    pol_f32 = np.where(ref.polarity, 1.0, -1.0).astype(np.float32)
    irng = np.random.RandomState(42)
    mask = (irng.rand(H,H) < INIT_DENSITY).astype(bool); np.fill_diagonal(mask, False)
    theta = np.full(H, 1.0, np.float32)
    nrng = np.random.RandomState(42)
    channel = nrng.randint(1, 9, size=H).astype(np.uint8)

    tool_rates = {op: 0.5 for op in ALL_OPS}
    op_history = {op: [] for op in ALL_OPS}
    current_accuracy = 0.0
    pain=0.0;reward_sig=0.0

    pool = Pool(N_WORKERS, initializer=init_w, initargs=(bp_out, ALL_DATA, bigram, pol_f32))
    best=0;t0=time.time()

    for step in range(1, steps+1):
        progress = step / steps
        features = np.array([current_accuracy, pain, reward_sig,
            tool_rates.get('add',0.5), tool_rates.get('flip',0.5),
            tool_rates.get('theta',0.5), tool_rates.get('channel',0.5),
            tool_rates.get('reverse',0.5), tool_rates.get('remove',0.5),
            progress], np.float32)
        probs = ctrl.forward(features)
        if random.random() < (0.10 if step < 500 else 0.05):
            pt = random.choice(ALL_OPS)
        else:
            pt = ALL_OPS[np.random.choice(N_OPS, p=probs)]
        if mask.sum()==0: pt='add'

        args=[(mask.flatten(),theta.copy(),channel.copy(),pol_f32.copy(),
               1000+step*50+w,pt) for w in range(N_WORKERS)]
        res=pool.map(worker_eval,args);br=max(res,key=lambda x:x['delta'])
        accepted = br['delta'] > THRESHOLD
        if accepted:
            if br['new_mask_flat'] is not None: mask[:]=br['new_mask_flat'].reshape(H,H)
            if br['new_theta'] is not None: theta[:]=br['new_theta']
            if br['new_channel'] is not None: channel[:]=br['new_channel']
            if br['new_pol_f'] is not None: pol_f32[:]=br['new_pol_f']
            reward = min(br['delta'] * 50, 1.0)
            pain=0.0;reward_sig=reward
        else:
            pain=0.3;reward_sig=0.0;reward=-0.05
        ctrl.update(ALL_OPS.index(pt), reward)
        op_history[pt].append(1.0 if accepted else 0.0)
        if len(op_history[pt]) > 100: op_history[pt] = op_history[pt][-100:]
        tool_rates[pt] = np.mean(op_history[pt]) if op_history[pt] else 0.5
        pain*=0.5;reward_sig*=0.5

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
            if step%1000==0:
                top=ALL_OPS[np.argmax(probs)]
                print(f"  [{label}:{step:5d}] best={best*100:.1f}% top={top}({probs.max()*100:.0f}%) "
                      f"{time.time()-t0:.0f}s")
                sys.stdout.flush()

    pool.terminate();pool.join()
    return best, time.time()-t0

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
    init_w(bp_out, ALL_DATA, bigram, None)

    print(f"\n{'='*60}")
    print(f"  PRUNED ReLU: 64 vs 128 hidden + warm start")
    print(f"  baseline: big ReLU (128) = 23.6%")
    print(f"{'='*60}")

    results = []
    for label, hidden, warm in [('B_pruned64', 64, False), ('C_warm64', 64, True)]:
        print(f"\n>> {label}:")
        ctrl = PrunedReLU(hidden=hidden, seed=99, lr=0.001, warm_start=warm)
        best, elapsed = run_controller(label, ctrl, steps=5000)
        results.append({'label':label,'best':best,'time':elapsed,'hidden':hidden,'warm':warm})
        print(f"  >> DONE {label}: best={best*100:.1f}% {elapsed:.0f}s")

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Big ReLU (128): 23.6% (33K params)")
    for r in results:
        w = "warm" if r['warm'] else "cold"
        print(f"  {r['label']:15s}: {r['best']*100:.1f}% (h={r['hidden']}, {w})")
