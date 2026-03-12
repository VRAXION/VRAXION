"""
DOES THIS WORK IN STANDARD NETWORKS?
======================================
Test the inverse arousal + self-wiring concept in:
1. Standard MLP (layered, no graph, no ticks)
2. Standard MLP + self-wiring bolted on
3. Simple recurrent net (like Elman)
4. Original graph net (control)

If it only works in the graph net, it's architecture-specific.
If it works in standard nets too, it's a general learning principle.
"""

import numpy as np
import time, math, random

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

# =========================================================================
# ARCHITECTURE 1: Standard MLP (2 hidden layers)
# =========================================================================
class StandardMLP:
    def __init__(self, n_in, n_hidden, n_out):
        self.n_in = n_in; self.n_hidden = n_hidden; self.n_out = n_out
        s1 = math.sqrt(2.0/n_in)
        s2 = math.sqrt(2.0/n_hidden)
        self.W1 = np.random.randn(n_in, n_hidden) * s1
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.randn(n_hidden, n_hidden) * s2
        self.b2 = np.zeros(n_hidden)
        self.W3 = np.random.randn(n_hidden, n_out) * s2
        self.b3 = np.zeros(n_out)

    def forward(self, x):
        h1 = np.where((x @ self.W1 + self.b1) > 0,
                       x @ self.W1 + self.b1,
                       0.01 * (x @ self.W1 + self.b1))
        h2 = np.where((h1 @ self.W2 + self.b2) > 0,
                       h1 @ self.W2 + self.b2,
                       0.01 * (h1 @ self.W2 + self.b2))
        return h2 @ self.W3 + self.b3

    def save_state(self):
        return (self.W1.copy(), self.b1.copy(), self.W2.copy(),
                self.b2.copy(), self.W3.copy(), self.b3.copy())

    def restore_state(self, s):
        self.W1, self.b1 = s[0].copy(), s[1].copy()
        self.W2, self.b2 = s[2].copy(), s[3].copy()
        self.W3, self.b3 = s[4].copy(), s[5].copy()

    def mutate(self, scale=0.05):
        self.W1 += np.random.randn(*self.W1.shape) * scale
        self.b1 += np.random.randn(*self.b1.shape) * scale
        self.W2 += np.random.randn(*self.W2.shape) * scale
        self.b2 += np.random.randn(*self.b2.shape) * scale
        self.W3 += np.random.randn(*self.W3.shape) * scale
        self.b3 += np.random.randn(*self.b3.shape) * scale

    def count_params(self):
        return (self.W1.size + self.b1.size + self.W2.size +
                self.b2.size + self.W3.size + self.b3.size)


# =========================================================================
# ARCHITECTURE 2: MLP + Self-Wiring (bolted on)
# Self-wiring adds skip connections based on activation
# =========================================================================
class MLPWithSelfWire:
    def __init__(self, n_in, n_hidden, n_out, addr_dim=4):
        self.n_in = n_in; self.n_hidden = n_hidden; self.n_out = n_out
        s1 = math.sqrt(2.0/n_in)
        s2 = math.sqrt(2.0/n_hidden)
        self.W1 = np.random.randn(n_in, n_hidden) * s1
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.randn(n_hidden, n_hidden) * s2
        self.b2 = np.zeros(n_hidden)
        self.W3 = np.random.randn(n_hidden, n_out) * s2
        self.b3 = np.zeros(n_out)

        # Skip connections (sparse, self-wired)
        self.skip_mask = np.zeros((n_in, n_out))  # input → output direct
        self.skip_W = np.random.randn(n_in, n_out) * 0.01
        self.skip_internal = np.zeros((n_hidden, n_out))  # h1 → output
        self.skip_W2 = np.random.randn(n_hidden, n_out) * 0.01

        # Addresses for self-wiring
        total = n_in + n_hidden + n_out
        self.addresses = np.random.randn(total, addr_dim)
        self.target_W = np.random.randn(total, addr_dim) * 0.1
        self.last_acc = 0.0

    def forward(self, x):
        h1 = np.where((x @ self.W1 + self.b1) > 0,
                       x @ self.W1 + self.b1,
                       0.01 * (x @ self.W1 + self.b1))
        h2 = np.where((h1 @ self.W2 + self.b2) > 0,
                       h1 @ self.W2 + self.b2,
                       0.01 * (h1 @ self.W2 + self.b2))
        out = h2 @ self.W3 + self.b3

        # Add skip connections
        out += x @ (self.skip_W * self.skip_mask)
        out += h1 @ (self.skip_W2 * self.skip_internal)

        # Self-wire based on activation (inverse arousal)
        if self.last_acc < 0.3:
            top_k, max_new = 2, 1
        elif self.last_acc < 0.7:
            top_k, max_new = 3, 2
        else:
            top_k, max_new = 4, 3

        # Active hidden neurons propose skip connections
        active = np.abs(h1)
        if active.sum() > 0.01 and max_new > 0:
            n_c = min(top_k, len(active))
            top_idx = np.argpartition(active, -n_c)[-n_c:]
            new = 0
            for ni in top_idx:
                ni = int(ni)
                # Connect to random output
                oi = random.randint(0, self.n_out - 1)
                if self.skip_internal[ni, oi] == 0:
                    self.skip_internal[ni, oi] = 1
                    self.skip_W2[ni, oi] = random.gauss(0, 0.01)
                    new += 1
                if new >= max_new: break

        return out

    def save_state(self):
        return (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy(),
                self.W3.copy(), self.b3.copy(),
                self.skip_mask.copy(), self.skip_W.copy(),
                self.skip_internal.copy(), self.skip_W2.copy(),
                self.addresses.copy(), self.target_W.copy())

    def restore_state(self, s):
        (self.W1, self.b1, self.W2, self.b2,
         self.W3, self.b3,
         self.skip_mask, self.skip_W,
         self.skip_internal, self.skip_W2,
         self.addresses, self.target_W) = [x.copy() for x in s]

    def mutate(self, scale=0.05):
        self.W1 += np.random.randn(*self.W1.shape) * scale
        self.b1 += np.random.randn(*self.b1.shape) * scale
        self.W2 += np.random.randn(*self.W2.shape) * scale
        self.b2 += np.random.randn(*self.b2.shape) * scale
        self.W3 += np.random.randn(*self.W3.shape) * scale
        self.b3 += np.random.randn(*self.b3.shape) * scale
        # Also mutate skip weights
        self.skip_W += np.random.randn(*self.skip_W.shape) * scale * 0.5
        self.skip_W2 += np.random.randn(*self.skip_W2.shape) * scale * 0.5

    def count_skip_conns(self):
        return int(self.skip_mask.sum() + self.skip_internal.sum())


# =========================================================================
# ARCHITECTURE 3: Elman Recurrent Net
# =========================================================================
class ElmanRNN:
    def __init__(self, n_in, n_hidden, n_out):
        self.n_in = n_in; self.n_hidden = n_hidden; self.n_out = n_out
        s1 = math.sqrt(2.0/(n_in + n_hidden))
        s2 = math.sqrt(2.0/n_hidden)
        self.W_ih = np.random.randn(n_in, n_hidden) * s1
        self.W_hh = np.random.randn(n_hidden, n_hidden) * s2
        self.b_h = np.zeros(n_hidden)
        self.W_ho = np.random.randn(n_hidden, n_out) * s2
        self.b_o = np.zeros(n_out)
        self.hidden = np.zeros(n_hidden)

    def reset_state(self): self.hidden = np.zeros(self.n_hidden)

    def forward(self, x):
        raw = x @ self.W_ih + self.hidden @ self.W_hh + self.b_h
        self.hidden = np.where(raw > 0, raw, 0.01 * raw)
        return self.hidden @ self.W_ho + self.b_o

    def save_state(self):
        return (self.W_ih.copy(), self.W_hh.copy(), self.b_h.copy(),
                self.W_ho.copy(), self.b_o.copy(), self.hidden.copy())

    def restore_state(self, s):
        (self.W_ih, self.W_hh, self.b_h,
         self.W_ho, self.b_o, self.hidden) = [x.copy() for x in s]

    def mutate(self, scale=0.05):
        self.W_ih += np.random.randn(*self.W_ih.shape) * scale
        self.W_hh += np.random.randn(*self.W_hh.shape) * scale
        self.b_h += np.random.randn(*self.b_h.shape) * scale
        self.W_ho += np.random.randn(*self.W_ho.shape) * scale
        self.b_o += np.random.randn(*self.b_o.shape) * scale


# =========================================================================
# ARCHITECTURE 4: Graph net (original v18 simplified)
# =========================================================================
class GraphNet:
    def __init__(self, n_neurons, n_in, n_out, density=0.06,
                 use_sw=True, inverse_arousal=True):
        self.N = n_neurons; self.n_in = n_in; self.n_out = n_out
        self.use_sw = use_sw; self.inverse = inverse_arousal
        self.last_acc = 0.0
        s = math.sqrt(2.0/n_neurons)
        self.W = np.random.randn(n_neurons, n_neurons) * s
        self.mask = (np.random.rand(n_neurons, n_neurons) < density).astype(np.float64)
        np.fill_diagonal(self.mask, 0)
        self.addresses = np.random.randn(n_neurons, 4)
        self.target_W = np.random.randn(n_neurons, 4) * 0.1
        self.state = np.zeros(n_neurons); self.decay = 0.5

    def reset_state(self): self.state = np.zeros(self.N)

    def forward(self, world, diff, ticks=6):
        inp = np.concatenate([world, diff])
        act = self.state.copy(); Weff = self.W * self.mask
        for t in range(ticks):
            act = act*self.decay; act[:self.n_in]=inp
            raw = act @ Weff + act*0.1
            act = np.where(raw>0, raw, 0.01*raw)
            act[:self.n_in]=inp
        self.state = act.copy()
        if self.use_sw:
            if self.inverse:
                if self.last_acc < 0.3: tk,mn = 2,1
                elif self.last_acc < 0.7: tk,mn = 3,2
                else: tk,mn = 5,3
            else:
                tk,mn = 3,2
            a2 = np.abs(act[self.n_in:]).copy()
            if a2.sum() > 0.01:
                nc = min(tk, len(a2))
                top = np.argpartition(a2, -nc)[-nc:]+self.n_in
                new=0
                for ni in top:
                    ni=int(ni)
                    if np.abs(act[ni])<0.1: continue
                    tgt=self.addresses[ni]+np.abs(act[ni])*self.target_W[ni]
                    d=((self.addresses-tgt)**2).sum(axis=1);d[ni]=float('inf')
                    near=int(np.argmin(d))
                    if self.mask[ni,near]==0:
                        self.mask[ni,near]=1
                        self.W[ni,near]=random.gauss(0,math.sqrt(2.0/self.N))
                        new+=1
                    if new>=mn: break
        return act[-self.n_out:]

    def count_connections(self): return int(self.mask.sum())
    def save_state(self):
        return (self.W.copy(),self.mask.copy(),self.state.copy(),
                self.addresses.copy(),self.target_W.copy())
    def restore_state(self,s):
        self.W,self.mask,self.state=s[0].copy(),s[1].copy(),s[2].copy()
        self.addresses,self.target_W=s[3].copy(),s[4].copy()

    def mutate_structure(self, rate=0.05):
        action=random.choice(["add","remove","rewire"])
        if action=="add":
            dead=np.argwhere(self.mask==0);dead=dead[dead[:,0]!=dead[:,1]]
            if len(dead)>0:
                n=max(1,int(len(dead)*rate))
                idx=dead[np.random.choice(len(dead),min(n,len(dead)),replace=False)]
                for j in range(len(idx)):
                    self.mask[int(idx[j][0]),int(idx[j][1])]=1
                    self.W[int(idx[j][0]),int(idx[j][1])]=random.gauss(0,math.sqrt(2.0/self.N))
        elif action=="remove":
            alive=np.argwhere(self.mask==1)
            if len(alive)>3:
                n=max(1,int(len(alive)*rate))
                idx=alive[np.random.choice(len(alive),min(n,len(alive)),replace=False)]
                for j in range(len(idx)):
                    self.mask[int(idx[j][0]),int(idx[j][1])]=0
        else:
            alive=np.argwhere(self.mask==1)
            if len(alive)>0:
                n=max(1,int(len(alive)*rate))
                idx=alive[np.random.choice(len(alive),min(n,len(alive)),replace=False)]
                for j in range(len(idx)):
                    r,c=int(idx[j][0]),int(idx[j][1]);self.mask[r,c]=0
                    nc=random.randint(0,self.N-1)
                    while nc==r:nc=random.randint(0,self.N-1)
                    self.mask[r,nc]=1;self.W[r,nc]=self.W[r,c]

    def mutate_weights(self, scale=0.05):
        self.W+=np.random.randn(*self.W.shape)*scale*self.mask
        self.target_W+=np.random.randn(*self.target_W.shape)*scale*0.5
        self.addresses+=np.random.randn(*self.addresses.shape)*scale*0.2


# =========================================================================
# UNIFIED EVAL
# =========================================================================
def run_experiment(label, net_type, v=16, n_hidden=32, max_attempts=4000, seed=42):
    np.random.seed(seed); random.seed(seed)
    perm = np.random.permutation(v); inputs = list(range(v))
    n_in_raw = v

    if net_type == "mlp":
        net = StandardMLP(n_in_raw*2, n_hidden, v)
    elif net_type == "mlp_sw":
        net = MLPWithSelfWire(n_in_raw*2, n_hidden, v)
    elif net_type == "rnn":
        net = ElmanRNN(n_in_raw*2, n_hidden, v)
    elif net_type == "graph":
        n_neurons = n_in_raw*2 + n_hidden + v
        net = GraphNet(n_neurons, n_in_raw*2, v, 0.06, False, False)
    elif net_type == "graph_sw":
        n_neurons = n_in_raw*2 + n_hidden + v
        net = GraphNet(n_neurons, n_in_raw*2, v, 0.06, True, False)
    elif net_type == "graph_sw_inverse":
        n_neurons = n_in_raw*2 + n_hidden + v
        net = GraphNet(n_neurons, n_in_raw*2, v, 0.06, True, True)

    best = -1; kept = 0; stale = 0; t0 = time.time(); curve = []
    is_graph = net_type.startswith("graph")
    is_rnn = net_type == "rnn"

    # Initial eval
    if is_rnn: net.reset_state()
    if is_graph: net.reset_state()
    prev_diff = np.zeros(v)
    correct = 0
    for idx in range(v):
        world = np.zeros(v); world[inputs[idx]] = 1.0
        inp = np.concatenate([world, prev_diff])
        if is_graph:
            logits = net.forward(world, prev_diff)
        else:
            logits = net.forward(inp) if not is_rnn else net.forward(inp)
        probs = softmax(logits[:v])
        if np.argmax(probs) == perm[idx]: correct += 1
        tv = np.zeros(v); tv[perm[idx]] = 1.0
        prev_diff = tv - probs
    current_score = correct / v
    best = current_score

    for att in range(max_attempts):
        state = net.save_state()

        if is_graph:
            if att < 2000: net.mutate_structure(0.05)
            else:
                if random.random()<0.3: net.mutate_structure(0.05)
                else: net.mutate_weights(0.05)
        else:
            net.mutate(0.05)

        # Eval
        if is_rnn: net.reset_state()
        if is_graph: net.reset_state()
        prev_diff = np.zeros(v)
        correct = 0
        for idx in range(v):
            world = np.zeros(v); world[inputs[idx]] = 1.0
            inp = np.concatenate([world, prev_diff])
            if is_graph:
                logits = net.forward(world, prev_diff)
            else:
                logits = net.forward(inp)
            probs = softmax(logits[:v])
            if np.argmax(probs) == perm[idx]: correct += 1
            tv = np.zeros(v); tv[perm[idx]] = 1.0
            prev_diff = tv - probs
        new_score = correct / v

        if hasattr(net, 'last_acc'):
            net.last_acc = new_score

        if new_score > current_score:
            current_score = new_score; kept += 1; stale = 0
            best = max(best, new_score)
        else:
            net.restore_state(state); stale += 1

        if (att+1)%500==0:
            curve.append((att+1, current_score))

        if best >= 1.0: break
        if stale >= 6000: break

    elapsed = time.time() - t0
    return {"label":label, "acc":best, "att":att+1, "time":elapsed,
            "kept":kept, "curve":curve}


# =========================================================================
print("="*70)
print("CROSS-ARCHITECTURE TEST — Does inverse arousal generalize?")
print("="*70)
print("16-class, hidden=32, 4k max attempts\n")

configs = [
    ("MLP (mutation only)",         "mlp"),
    ("MLP + self-wire",             "mlp_sw"),
    ("Elman RNN",                   "rnn"),
    ("Graph (no SW)",               "graph"),
    ("Graph + SW",                  "graph_sw"),
    ("Graph + SW + inverse",        "graph_sw_inverse"),
]

results = []
for label, ntype in configs:
    print(f"  {label:30s}...", end=" ", flush=True)
    r = run_experiment(label, ntype)
    results.append(r)
    f100="never"
    for s,a in r['curve']:
        if a>=1.0: f100=str(s); break
    print(f"{r['acc']*100:.0f}% | 100%@{f100:>5} | "
          f"Kept:{r['kept']:3d} | {r['time']:.0f}s")

print(f"\n{'='*70}")
print(f"RANKED")
print(f"{'='*70}")
print(f"{'Config':<30} {'Acc':>5} {'100%@':>6} {'Kept':>5}")
print(f"{'-'*30} {'-'*5} {'-'*6} {'-'*5}")
def sk(r):
    for s,a in r['curve']:
        if a>=1.0: return (0,s)
    return (1,-r['acc'])
for r in sorted(results,key=sk):
    f100="never"
    for s,a in r['curve']:
        if a>=1.0: f100=str(s); break
    print(f"{r['label']:<30} {r['acc']*100:4.0f}% {f100:>6} {r['kept']:>5}")

print(f"\n--- Learning curves ---")
for r in sorted(results,key=sk):
    accs=" → ".join(f"{a*100:.0f}%" for _,a in r['curve'])
    print(f"  {r['label'][:28]:<28}: {accs}")
