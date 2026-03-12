"""
IS SELF-WIRING NEEDED? — Ablation of components
=================================================
Test every combination:
1. No self-wire, no arousal (pure mutation)
2. Self-wire, no arousal (v18 original)
3. No self-wire, inverse arousal (arousal without self-wire — doesn't make sense?)
4. Self-wire + inverse arousal (the winner)
5. No self-wire, but mutation rate varies inversely with accuracy
   (arousal applied to MUTATION instead of self-wire)
6. Self-wire only, no mutation (v19 style but with inverse arousal)

This tells us exactly which component matters.
"""

import numpy as np
import time, math, random

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

class FlexNet:
    def __init__(self, n_neurons, n_in, n_out, addr_dim=4, density=0.06,
                 use_selfwire=True, use_mutation=True,
                 arousal_target="none"):
        self.N = n_neurons; self.n_in = n_in; self.n_out = n_out
        self.addr_dim = addr_dim
        self.use_selfwire = use_selfwire
        self.use_mutation = use_mutation
        self.arousal_target = arousal_target  # "none", "selfwire", "mutation", "both"
        self.last_acc = 0.0

        s = math.sqrt(2.0 / n_neurons)
        self.W = np.random.randn(n_neurons, n_neurons) * s
        self.mask = (np.random.rand(n_neurons, n_neurons) < density).astype(np.float64)
        np.fill_diagonal(self.mask, 0)
        self.addresses = np.random.randn(n_neurons, addr_dim)
        self.target_W = np.random.randn(n_neurons, addr_dim) * 0.1
        self.state = np.zeros(n_neurons); self.decay = 0.5

    def reset_state(self): self.state = np.zeros(self.N)

    def _get_sw_params(self):
        if not self.use_selfwire:
            return 0, 0
        if self.arousal_target in ("selfwire", "both"):
            # Inverse: more self-wire when good
            if self.last_acc < 0.3: return 2, 1
            elif self.last_acc < 0.7: return 3, 2
            else: return 5, 3
        return 3, 2  # default

    def _get_mutation_rate(self):
        if not self.use_mutation:
            return 0.0
        if self.arousal_target in ("mutation", "both"):
            # Inverse for mutation too: more mutation when bad (explore),
            # less when good (let self-wire refine)
            if self.last_acc < 0.3: return 0.08
            elif self.last_acc < 0.7: return 0.05
            else: return 0.02
        return 0.05  # default

    def forward(self, world, diff, ticks=6):
        inp = np.concatenate([world, diff])
        act = self.state.copy(); Weff = self.W * self.mask
        for t in range(ticks):
            act = act * self.decay; act[:self.n_in] = inp
            raw = act @ Weff + act * 0.1
            act = np.where(raw > 0, raw, 0.01 * raw)
            act[:self.n_in] = inp
        self.state = act.copy()
        top_k, max_new = self._get_sw_params()
        if top_k > 0 and max_new > 0:
            self._self_wire(act, top_k, max_new)
        return act[-self.n_out:]

    def _self_wire(self, activations, top_k, max_new):
        a2 = np.abs(activations[self.n_in:]).copy()
        if a2.sum() < 0.01: return
        n_c = min(top_k, len(a2))
        top = np.argpartition(a2, -n_c)[-n_c:] + self.n_in
        new = 0
        for ni in top:
            ni = int(ni)
            if np.abs(activations[ni]) < 0.1: continue
            tgt = self.addresses[ni] + np.abs(activations[ni]) * self.target_W[ni]
            d = ((self.addresses - tgt)**2).sum(axis=1); d[ni]=float('inf')
            near = int(np.argmin(d))
            if self.mask[ni, near] == 0:
                self.mask[ni, near] = 1
                self.W[ni, near] = random.gauss(0, math.sqrt(2.0/self.N))
                new += 1
            if new >= max_new: break

    def count_connections(self): return int(self.mask.sum())
    def save_state(self):
        return (self.W.copy(), self.mask.copy(), self.state.copy(),
                self.addresses.copy(), self.target_W.copy())
    def restore_state(self, s):
        self.W, self.mask, self.state = s[0].copy(), s[1].copy(), s[2].copy()
        self.addresses, self.target_W = s[3].copy(), s[4].copy()

    def mutate_structure(self):
        rate = self._get_mutation_rate()
        if rate <= 0: return
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
        if not self.use_mutation: return
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
    acc=correct/n; net.last_acc=acc
    return 0.5*acc+0.5*(total_score/n), acc


def run_test(label, use_sw, use_mut, arousal, v=16, n_neurons=80,
             max_attempts=4000, seed=42):
    np.random.seed(seed); random.seed(seed)
    perm=np.random.permutation(v); inputs=list(range(v))
    n_in,n_out,ticks = v*2,v,6
    net = FlexNet(n_neurons, n_in, n_out, 4, 0.06, use_sw, use_mut, arousal)
    score,acc = evaluate(net,inputs,perm,v,ticks); best=score
    phase,kept,stale="STRUCT",0,0; switched=False; t0=time.time(); curve=[]

    for att in range(max_attempts):
        state=net.save_state()
        if phase=="STRUCT": net.mutate_structure()
        else:
            if random.random()<0.3: net.mutate_structure()
            else: net.mutate_weights()
        ns,na=evaluate(net,inputs,perm,v,ticks)
        if ns>score: score=ns;kept+=1;stale=0;best=max(best,score)
        else: net.restore_state(state);stale+=1
        if phase=="STRUCT" and stale>2000 and not switched:
            phase="BOTH";switched=True;stale=0
        if (att+1)%500==0:
            _,a=evaluate(net,inputs,perm,v,ticks)
            curve.append((att+1, a, net.count_connections()))
        if best>0.99: break
        if stale>=8000: break

    elapsed=time.time()-t0; _,final=evaluate(net,inputs,perm,v,ticks)
    return {"label":label,"acc":final,"att":att+1,"time":elapsed,
            "conns":net.count_connections(),"curve":curve,"kept":kept}


print("="*70)
print("COMPONENT ABLATION — What actually matters?")
print("="*70)
print("16-class, 80 neurons, 4k max attempts\n")

configs = [
    # label, self_wire, mutation, arousal_target
    ("Mut only (no SW)",           False, True,  "none"),
    ("Mut + SW (v18 base)",        True,  True,  "none"),
    ("Mut + inverse SW arousal",   True,  True,  "selfwire"),
    ("Mut + inverse mut arousal",  False, True,  "mutation"),
    ("Mut + inverse both arousal", True,  True,  "both"),
    ("SW only + inverse (no mut)", True,  False, "selfwire"),
]

results = []
for label, sw, mut, ar in configs:
    print(f"  {label:32s}...", end=" ", flush=True)
    r = run_test(label, sw, mut, ar)
    results.append(r)
    f100="never"
    for s,a,c in r['curve']:
        if a>=1.0: f100=str(s); break
    print(f"{r['acc']*100:.0f}% | 100%@{f100:>5} | "
          f"Conns:{r['conns']:5d} | Kept:{r['kept']:3d} | {r['time']:.0f}s")

print(f"\n{'='*70}")
print(f"RANKED")
print(f"{'='*70}")
print(f"{'Config':<32} {'Acc':>5} {'100%@':>6} {'Conns':>6} {'Kept':>5}")
print(f"{'-'*32} {'-'*5} {'-'*6} {'-'*6} {'-'*5}")
def sk(r):
    for s,a,c in r['curve']:
        if a>=1.0: return (0,s)
    return (1,-r['acc'])
for r in sorted(results,key=sk):
    f100="never"
    for s,a,c in r['curve']:
        if a>=1.0: f100=str(s); break
    print(f"{r['label']:<32} {r['acc']*100:4.0f}% {f100:>6} "
          f"{r['conns']:>6} {r['kept']:>5}")

print(f"\n--- Learning curves ---")
for r in sorted(results,key=sk):
    accs=" → ".join(f"{a*100:.0f}%" for _,a,_ in r['curve'])
    print(f"  {r['label'][:30]:<30}: {accs}")

print(f"\n--- Connection curves ---")
for r in sorted(results,key=sk):
    conns=" → ".join(f"{c}" for _,_,c in r['curve'])
    print(f"  {r['label'][:30]:<30}: {conns}")

# Analysis
print(f"\n{'='*70}")
print(f"ANALYSIS")
print(f"{'='*70}")
print(f"\nSelf-wiring contribution:")
for r in results:
    if "no SW" in r['label'] or "SW only" in r['label']:
        continue
# Compare with and without SW at same arousal level
sw_results = {r['label']: r for r in results}
