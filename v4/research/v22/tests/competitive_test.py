"""
COMPETITIVE INHIBITION — Force orthogonal patterns
====================================================
The 64-class wall is interference: patterns overlap.
Solution: when a neuron activates, it suppresses nearby neurons.
This forces distinct, non-overlapping patterns per input.

Test:
1. Baseline (no inhibition)
2. Global top-k: only k strongest neurons survive per tick
3. Local inhibition: each active neuron suppresses its neighbors
4. Soft WTA: softmax-like competition within groups
"""

import numpy as np
import time, math, random

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

class CompetitiveNet:
    def __init__(self, n_neurons, n_in, n_out, density=0.06,
                 inhibition="none", top_k=5, inhib_radius=3, inhib_strength=0.8):
        self.N = n_neurons; self.n_in = n_in; self.n_out = n_out
        self.inhibition = inhibition
        self.top_k = top_k
        self.inhib_radius = inhib_radius
        self.inhib_strength = inhib_strength
        self.last_acc = 0.0

        s = math.sqrt(2.0 / n_neurons)
        self.W = np.random.randn(n_neurons, n_neurons) * s
        self.mask = (np.random.rand(n_neurons, n_neurons) < density).astype(float)
        np.fill_diagonal(self.mask, 0)
        self.addr = np.random.randn(n_neurons, 4)
        self.tw = np.random.randn(n_neurons, 4) * 0.1
        self.state = np.zeros(n_neurons); self.decay = 0.5

    def _inhibit(self, act):
        """Apply competitive inhibition to internal neurons."""
        internal = slice(self.n_in, self.N - self.n_out)
        vals = act[internal].copy()
        n_internal = len(vals)

        if self.inhibition == "none":
            return act

        elif self.inhibition == "global_topk":
            # Only top-k internal neurons survive, rest zeroed
            if n_internal <= self.top_k:
                return act
            threshold_idx = np.argpartition(np.abs(vals), -self.top_k)[-self.top_k:]
            mask = np.zeros(n_internal)
            mask[threshold_idx] = 1.0
            act[internal] = vals * mask
            return act

        elif self.inhibition == "soft_topk":
            # Top-k survive fully, rest get dampened (not zeroed)
            if n_internal <= self.top_k:
                return act
            abs_vals = np.abs(vals)
            threshold_idx = np.argpartition(abs_vals, -self.top_k)[-self.top_k:]
            threshold_val = abs_vals[threshold_idx].min()
            # Winners keep full value, losers get dampened
            dampen = np.where(abs_vals >= threshold_val, 1.0, 0.1)
            act[internal] = vals * dampen
            return act

        elif self.inhibition == "local_addr":
            # Neurons suppress nearby neurons in 4D address space
            # Strongest neuron in local neighborhood wins
            abs_vals = np.abs(vals)
            int_addrs = self.addr[internal]
            suppressed = np.ones(n_internal)

            # Find strongly active neurons
            active_idx = np.where(abs_vals > 0.1)[0]
            if len(active_idx) == 0:
                return act

            # Sort by strength (strongest inhibits first)
            sorted_active = active_idx[np.argsort(-abs_vals[active_idx])]

            for idx in sorted_active:
                if suppressed[idx] < 0.5:
                    continue  # already suppressed
                # Suppress neighbors
                dists = np.sqrt(((int_addrs - int_addrs[idx])**2).sum(axis=1) + 1e-8)
                neighbors = (dists < self.inhib_radius) & (np.arange(n_internal) != idx)
                suppressed[neighbors] *= (1 - self.inhib_strength)

            act[internal] = vals * suppressed
            return act

        elif self.inhibition == "group_wta":
            # Divide neurons into groups, winner-takes-all within each group
            group_size = max(2, n_internal // self.top_k)
            for g in range(0, n_internal, group_size):
                end = min(g + group_size, n_internal)
                group = vals[g:end]
                if len(group) == 0: continue
                winner = np.argmax(np.abs(group))
                mask_g = np.zeros(len(group))
                mask_g[winner] = 1.0
                # Winner keeps value, others get small fraction
                vals[g:end] = group * (mask_g * 0.9 + 0.1)
            act[internal] = vals
            return act

        return act

    def reset(self): self.state = np.zeros(self.N)

    def forward(self, world, diff, ticks=6):
        inp = np.concatenate([world, diff])
        act = self.state.copy(); Weff = self.W * self.mask
        for t in range(ticks):
            act = act * self.decay; act[:self.n_in] = inp
            raw = act @ Weff + act * 0.1
            act = np.where(raw > 0, raw, 0.01 * raw)  # leaky relu
            act = self._inhibit(act)  # competitive inhibition
            act[:self.n_in] = inp
        self.state = act.copy()
        # Inverse arousal self-wire
        if self.last_acc < 0.3: tk,mn = 2,1
        elif self.last_acc < 0.7: tk,mn = 3,2
        else: tk,mn = 5,3
        a2 = np.abs(act[self.n_in:])
        if a2.sum() > 0.01:
            nc = min(tk, len(a2))
            top = np.argpartition(a2, -nc)[-nc:] + self.n_in; new = 0
            for ni in top:
                ni = int(ni)
                if np.abs(act[ni]) < 0.1: continue
                tgt = self.addr[ni] + np.abs(act[ni]) * self.tw[ni]
                d = ((self.addr-tgt)**2).sum(axis=1); d[ni] = 1e9
                near = int(np.argmin(d))
                if self.mask[ni, near] == 0:
                    self.mask[ni, near] = 1
                    self.W[ni, near] = random.gauss(0, math.sqrt(2.0/self.N))
                    new += 1
                if new >= mn: break
        return act[-self.n_out:]

    def conns(self): return int(self.mask.sum())
    def save(self):
        return (self.W.copy(), self.mask.copy(), self.state.copy(),
                self.addr.copy(), self.tw.copy())
    def restore(self, s):
        self.W,self.mask,self.state = s[0].copy(),s[1].copy(),s[2].copy()
        self.addr,self.tw = s[3].copy(),s[4].copy()

    def mutate_struct(self, rate=0.05):
        a = random.choice(["add","remove","rewire"])
        if a == "add":
            dead = np.argwhere(self.mask==0); dead = dead[dead[:,0]!=dead[:,1]]
            if len(dead) > 0:
                n = max(1,int(len(dead)*rate))
                idx = dead[np.random.choice(len(dead),min(n,len(dead)),replace=False)]
                for j in range(len(idx)):
                    self.mask[int(idx[j][0]),int(idx[j][1])] = 1
                    self.W[int(idx[j][0]),int(idx[j][1])] = random.gauss(0,math.sqrt(2.0/self.N))
        elif a == "remove":
            alive = np.argwhere(self.mask==1)
            if len(alive) > 3:
                n = max(1,int(len(alive)*rate))
                idx = alive[np.random.choice(len(alive),min(n,len(alive)),replace=False)]
                for j in range(len(idx)):
                    self.mask[int(idx[j][0]),int(idx[j][1])] = 0
        else:
            alive = np.argwhere(self.mask==1)
            if len(alive) > 0:
                n = max(1,int(len(alive)*rate))
                idx = alive[np.random.choice(len(alive),min(n,len(alive)),replace=False)]
                for j in range(len(idx)):
                    r,c=int(idx[j][0]),int(idx[j][1]);self.mask[r,c]=0
                    nc=random.randint(0,self.N-1)
                    while nc==r:nc=random.randint(0,self.N-1)
                    self.mask[r,nc]=1;self.W[r,nc]=self.W[r,c]

    def mutate_weights(self, scale=0.05):
        self.W += np.random.randn(*self.W.shape)*scale*self.mask
        self.tw += np.random.randn(*self.tw.shape)*scale*0.5
        self.addr += np.random.randn(*self.addr.shape)*scale*0.2


def run_test(label, inhibition, top_k=5, v=16, n_neurons=80,
             max_att=4000, seed=42, inhib_r=2.0, inhib_s=0.7):
    np.random.seed(seed); random.seed(seed)
    perm = np.random.permutation(v); inputs = list(range(v))
    n_in = v*2; n_out = v; ticks = 6
    net = CompetitiveNet(n_neurons, n_in, n_out, 0.06,
                         inhibition, top_k, inhib_r, inhib_s)

    def ev():
        net.reset(); pd = np.zeros(v); c = 0
        for p in range(2):
            for i in range(v):
                w = np.zeros(v); w[inputs[i]] = 1.0
                lo = net.forward(w, pd); pr = softmax(lo[:v])
                if p == 1 and np.argmax(pr) == perm[i]: c += 1
                tv = np.zeros(v); tv[int(perm[i])] = 1.0; pd = tv - pr
        return c / v

    score = ev(); best = score
    phase, kept, stale = "S", 0, 0; switched = False
    t0 = time.time(); curve = []

    for att in range(max_att):
        s = net.save()
        if phase == "S": net.mutate_struct()
        else:
            if random.random() < 0.3: net.mutate_struct()
            else: net.mutate_weights()
        ns = ev(); net.last_acc = ns
        if ns > score: score=ns;kept+=1;stale=0;best=max(best,score)
        else: net.restore(s);stale+=1
        if phase=="S" and stale>2000 and not switched:
            phase="B";switched=True;stale=0
        if (att+1)%500==0:
            # Count active neurons for pattern analysis
            net.reset()
            tw=np.zeros(v);tw[0]=1.0;td=np.zeros(v)
            net.forward(tw,td,ticks=ticks)
            internal = net.state[n_in:-n_out]
            n_active = (np.abs(internal)>0.01).sum()
            curve.append((att+1, score, n_active))
        if best>=0.99:break
        if stale>=6000:break

    elapsed = time.time()-t0
    return {"label":label,"acc":best,"att":att+1,"time":elapsed,
            "conns":net.conns(),"kept":kept,"curve":curve}


print("="*65)
print("COMPETITIVE INHIBITION — Force orthogonal patterns")
print("="*65)
print("16-class, 80 neurons\n")

configs = [
    ("No inhibition (baseline)", "none",        5),
    ("Global top-3",             "global_topk", 3),
    ("Global top-5",             "global_topk", 5),
    ("Global top-8",             "global_topk", 8),
    ("Soft top-5",               "soft_topk",   5),
    ("Local addr r=2.0",         "local_addr",  5),
    ("Group WTA (k=5 groups)",   "group_wta",   5),
]

results = []
for label, inhib, k in configs:
    print(f"  {label:28s}...", end=" ", flush=True)
    r = run_test(label, inhib, k)
    results.append(r)
    f100 = "never"
    for s,a,n in r['curve']:
        if a >= 0.99: f100 = str(s); break
    last_active = r['curve'][-1][2] if r['curve'] else 0
    print(f"{r['acc']*100:.0f}% | 100%@{f100:>5} | "
          f"Conns:{r['conns']:5d} | Active:{last_active:2d} | {r['time']:.0f}s")

print(f"\n{'='*65}")
print(f"RANKED")
print(f"{'='*65}")
print(f"{'Config':<28} {'Acc':>5} {'100%@':>6} {'Conns':>6} {'Active':>7}")
print(f"{'-'*28} {'-'*5} {'-'*6} {'-'*6} {'-'*7}")
def sk(r):
    for s,a,n in r['curve']:
        if a>=0.99:return(0,s)
    return(1,-r['acc'])
for r in sorted(results,key=sk):
    f100="never"
    for s,a,n in r['curve']:
        if a>=0.99:f100=str(s);break
    la = r['curve'][-1][2] if r['curve'] else 0
    print(f"{r['label']:<28} {r['acc']*100:4.0f}% {f100:>6} "
          f"{r['conns']:>6} {la:>7}")

print(f"\n--- Learning curves ---")
for r in sorted(results,key=sk):
    accs=" -> ".join(f"{a*100:.0f}%" for _,a,_ in r['curve'])
    print(f"  {r['label'][:26]:<26}: {accs}")

print(f"\n--- Active neuron count evolution ---")
for r in sorted(results,key=sk):
    acts=" -> ".join(f"{n}" for _,_,n in r['curve'])
    print(f"  {r['label'][:26]:<26}: {acts}")
