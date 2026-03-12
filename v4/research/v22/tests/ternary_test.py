"""
TERNARY ACTIVATION TEST — (-1, 0, +1)
======================================
Like the brain: excite (+1), inhibit (-1), or stay silent (0).
Three states, nothing in between.

Previously binary failed at large scale because threshold was wrong.
Ternary adds inhibition which failed as learnable per-neuron property.
But maybe as a FIXED activation it works differently — every neuron
CAN inhibit, the topology decides WHEN.

Test multiple threshold configs for the ternary mapping.
"""

import numpy as np
import time, math, random

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

class SelfWiringNet:
    def __init__(self, n_neurons, n_in, n_out, addr_dim=4, density=0.06,
                 activation="leaky_relu", pos_thresh=0.5, neg_thresh=-0.5):
        self.N = n_neurons; self.n_in = n_in; self.n_out = n_out
        self.addr_dim = addr_dim; self.activation = activation
        self.pos_thresh = pos_thresh; self.neg_thresh = neg_thresh

        s = math.sqrt(2.0 / n_neurons)
        self.W = np.random.randn(n_neurons, n_neurons) * s
        self.mask = (np.random.rand(n_neurons, n_neurons) < density).astype(np.float64)
        np.fill_diagonal(self.mask, 0)
        self.addresses = np.random.randn(n_neurons, addr_dim)
        self.target_W = np.random.randn(n_neurons, addr_dim) * 0.1
        self.state = np.zeros(n_neurons); self.decay = 0.5

    def _activate(self, x, tick, ticks):
        if self.activation == "leaky_relu":
            return np.where(x > 0, x, 0.01 * x)
        elif self.activation == "ternary":
            # +1 if above pos threshold, -1 if below neg threshold, 0 otherwise
            result = np.zeros_like(x)
            result[x > self.pos_thresh] = 1.0
            result[x < self.neg_thresh] = -1.0
            # Output neurons keep raw values on last tick for softmax
            if tick == ticks - 1:
                result[-self.n_out:] = x[-self.n_out:]
            return result
        elif self.activation == "ternary_scaled":
            # Same but magnitude preserved: sign(x) * 1 if |x| > thresh, else 0
            result = np.zeros_like(x)
            result[x > self.pos_thresh] = 1.0
            result[x < self.neg_thresh] = -1.0
            if tick == ticks - 1:
                result[-self.n_out:] = x[-self.n_out:]
            return result
        elif self.activation == "ternary_soft":
            # Ternary for internal, leaky_relu for output neurons always
            internal = slice(self.n_in, self.N - self.n_out)
            result = x.copy()
            int_vals = x[internal]
            ternary = np.zeros_like(int_vals)
            ternary[int_vals > self.pos_thresh] = 1.0
            ternary[int_vals < self.neg_thresh] = -1.0
            result[internal] = ternary
            # Output neurons always leaky relu
            out_vals = x[-self.n_out:]
            result[-self.n_out:] = np.where(out_vals > 0, out_vals, 0.01 * out_vals)
            return result

    def reset_state(self): self.state = np.zeros(self.N)

    def forward(self, world, diff, ticks=6):
        inp = np.concatenate([world, diff])
        act = self.state.copy(); Weff = self.W * self.mask
        for t in range(ticks):
            act = act * self.decay; act[:self.n_in] = inp
            raw = act @ Weff + act * 0.1
            act = self._activate(raw, t, ticks)
            act[:self.n_in] = inp
        self.state = act.copy()
        # self-wire using absolute activation
        a2 = np.abs(act[self.n_in:]).copy()
        if a2.sum() > 0.01:
            n_c = min(3, len(a2))
            top = np.argpartition(a2, -n_c)[-n_c:] + self.n_in
            new = 0
            for ni in top:
                ni = int(ni)
                if np.abs(act[ni]) < 0.1: continue
                tgt = self.addresses[ni] + np.abs(act[ni]) * self.target_W[ni]
                d = ((self.addresses - tgt)**2).sum(axis=1); d[ni] = float('inf')
                near = int(np.argmin(d))
                if self.mask[ni, near] == 0:
                    self.mask[ni, near] = 1
                    self.W[ni, near] = random.gauss(0, math.sqrt(2.0/self.N))
                    new += 1
                if new >= 2: break
        return act[-self.n_out:]

    def count_connections(self): return int(self.mask.sum())
    def save_state(self):
        return (self.W.copy(), self.mask.copy(), self.state.copy(),
                self.addresses.copy(), self.target_W.copy())
    def restore_state(self, s):
        self.W, self.mask, self.state = s[0].copy(), s[1].copy(), s[2].copy()
        self.addresses, self.target_W = s[3].copy(), s[4].copy()

    def mutate_structure(self, rate=0.05):
        action = random.choice(["add", "remove", "rewire"])
        if action == "add":
            dead = np.argwhere(self.mask == 0); dead = dead[dead[:,0]!=dead[:,1]]
            if len(dead) > 0:
                n = max(1, int(len(dead)*rate))
                idx = dead[np.random.choice(len(dead), min(n,len(dead)), replace=False)]
                for j in range(len(idx)):
                    self.mask[int(idx[j][0]),int(idx[j][1])] = 1
                    self.W[int(idx[j][0]),int(idx[j][1])] = random.gauss(0, math.sqrt(2.0/self.N))
        elif action == "remove":
            alive = np.argwhere(self.mask == 1)
            if len(alive) > 3:
                n = max(1, int(len(alive)*rate))
                idx = alive[np.random.choice(len(alive), min(n,len(alive)), replace=False)]
                for j in range(len(idx)):
                    self.mask[int(idx[j][0]),int(idx[j][1])] = 0
        else:
            alive = np.argwhere(self.mask == 1)
            if len(alive) > 0:
                n = max(1, int(len(alive)*rate))
                idx = alive[np.random.choice(len(alive), min(n,len(alive)), replace=False)]
                for j in range(len(idx)):
                    r,c = int(idx[j][0]),int(idx[j][1]); self.mask[r,c]=0
                    nc = random.randint(0,self.N-1)
                    while nc==r: nc=random.randint(0,self.N-1)
                    self.mask[r,nc]=1; self.W[r,nc]=self.W[r,c]

    def mutate_weights(self, scale=0.05):
        self.W += np.random.randn(*self.W.shape)*scale*self.mask
        self.target_W += np.random.randn(*self.target_W.shape)*scale*0.5
        self.addresses += np.random.randn(*self.addresses.shape)*scale*0.2


def evaluate(net, inputs, targets, vocab, ticks=6, n_passes=2):
    correct=0; total_score=0.0; n=len(inputs)
    net.reset_state(); prev_diff=np.zeros(vocab)
    for p in range(n_passes):
        for idx in range(n):
            world=np.zeros(vocab); world[inputs[idx]]=1.0
            logits=net.forward(world, prev_diff, ticks=ticks)
            probs=softmax(logits); pred=int(np.argmax(probs)); tgt=int(targets[idx])
            if p==n_passes-1:
                total_score+=probs[tgt]
                if pred==tgt: correct+=1
            tv=np.zeros(vocab); tv[tgt]=1.0; prev_diff=tv-probs
    return 0.5*(correct/n)+0.5*(total_score/n), correct/n


def run_test(label, activation, pos_t=0.5, neg_t=-0.5,
             v=16, n_neurons=80, max_attempts=4000, seed=42):
    np.random.seed(seed); random.seed(seed)
    perm=np.random.permutation(v); inputs=list(range(v))
    n_in,n_out,ticks = v*2,v,6
    net = SelfWiringNet(n_neurons, n_in, n_out, 4, 0.06,
                        activation, pos_t, neg_t)
    score,acc = evaluate(net,inputs,perm,v,ticks); best=score
    phase,kept,stale="STRUCT",0,0; switched=False; t0=time.time(); curve=[]

    for att in range(max_attempts):
        state=net.save_state()
        if phase=="STRUCT": net.mutate_structure(0.05)
        else:
            if random.random()<0.3: net.mutate_structure(0.02)
            else: net.mutate_weights(0.05)
        ns,na=evaluate(net,inputs,perm,v,ticks)
        if ns>score: score=ns;kept+=1;stale=0;best=max(best,score)
        else: net.restore_state(state);stale+=1
        if phase=="STRUCT" and stale>2000 and not switched:
            phase="BOTH";switched=True;stale=0
        if (att+1)%500==0:
            _,a=evaluate(net,inputs,perm,v,ticks)
            # Count ternary distribution
            net.reset_state()
            tw=np.zeros(v);tw[0]=1.0;td=np.zeros(v)
            net.forward(tw,td,ticks=ticks)
            internal = net.state[n_in:-n_out]
            n_pos = (internal > 0.5).sum()
            n_neg = (internal < -0.5).sum()
            n_zero = len(internal) - n_pos - n_neg
            curve.append((att+1, a, net.count_connections(), n_pos, n_neg, n_zero))
        if best>0.99: break
        if stale>=8000: break

    elapsed=time.time()-t0; _,final=evaluate(net,inputs,perm,v,ticks)
    return {"label":label,"acc":final,"att":att+1,"time":elapsed,
            "conns":net.count_connections(),"curve":curve,"kept":kept}


print("="*70)
print("TERNARY (-1, 0, +1) vs BINARY vs LEAKY_RELU")
print("="*70)
print("16-class, 80 neurons, 4k max attempts\n")

configs = [
    # label, activation, pos_thresh, neg_thresh
    ("leaky_relu (baseline)", "leaky_relu", 0, 0),
    ("ternary t=0.3",        "ternary", 0.3, -0.3),
    ("ternary t=0.5",        "ternary", 0.5, -0.5),
    ("ternary t=1.0",        "ternary", 1.0, -1.0),
    ("ternary asymm hi+/lo-","ternary", 0.5, -0.2),  # easier to inhibit
    ("ternary_soft t=0.3",   "ternary_soft", 0.3, -0.3),
    ("ternary_soft t=0.5",   "ternary_soft", 0.5, -0.5),
]

results = []
for label, act, pt, nt in configs:
    print(f"  {label:28s}...", end=" ", flush=True)
    r = run_test(label, act, pt, nt)
    results.append(r)
    f100="never"
    for s,a,c,_,_,_ in r['curve']:
        if a>=1.0: f100=str(s); break
    # Show ternary distribution at last checkpoint
    if r['curve']:
        _,_,_,np_,nn_,nz_ = r['curve'][-1]
        dist = f"+:{np_} -:{nn_} 0:{nz_}"
    else:
        dist = ""
    print(f"{r['acc']*100:.0f}% | 100%@{f100:>5} | "
          f"Conns:{r['conns']:5d} | {dist} | {r['time']:.0f}s")

print(f"\n{'='*70}")
print(f"RANKED")
print(f"{'='*70}")
print(f"{'Config':<28} {'Acc':>5} {'100%@':>6} {'Conns':>6} {'Distribution':>18}")
print(f"{'-'*28} {'-'*5} {'-'*6} {'-'*6} {'-'*18}")
def sk(r):
    for s,a,c,_,_,_ in r['curve']:
        if a>=1.0: return (0,s)
    return (1,-r['acc'])
for r in sorted(results,key=sk):
    f100="never"
    for s,a,c,_,_,_ in r['curve']:
        if a>=1.0: f100=str(s); break
    if r['curve']:
        _,_,_,np_,nn_,nz_ = r['curve'][-1]
        dist = f"+:{np_} -:{nn_} 0:{nz_}"
    else:
        dist = ""
    print(f"{r['label']:<28} {r['acc']*100:4.0f}% {f100:>6} "
          f"{r['conns']:>6} {dist:>18}")

print(f"\n--- Learning curves ---")
for r in sorted(results,key=sk):
    accs=" → ".join(f"{a*100:.0f}%" for _,a,_,_,_,_ in r['curve'])
    print(f"  {r['label']:<28}: {accs}")

print(f"\n--- Ternary distribution evolution (+/- /0) ---")
for r in sorted(results,key=sk):
    if r['activation'] != 'leaky_relu' if hasattr(r,'activation') else True:
        dists=" → ".join(f"+{p}-{n}o{z}" for _,_,_,p,n,z in r['curve'][:6])
        print(f"  {r['label']:<28}: {dists}")
