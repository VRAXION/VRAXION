"""
REAL DATA TEST — Character prediction on English text
======================================================
Not synthetic A→B lookup, but actual language patterns.
Can the self-wiring graph learn bigram statistics?

Task: given current character, predict next character.
This requires learning real statistical patterns of English.
"""

import numpy as np
import time, math, random

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

# Real English text for training
TEXT = """the quick brown fox jumps over the lazy dog. 
the cat sat on the mat. the dog chased the cat around the garden.
she sells sea shells by the sea shore. peter piper picked a peck of pickled peppers.
to be or not to be that is the question. all that glitters is not gold.
a stitch in time saves nine. the early bird catches the worm.
actions speak louder than words. practice makes perfect.
knowledge is power. time is money. better late than never.
the pen is mightier than the sword. where there is a will there is a way.
an apple a day keeps the doctor away. birds of a feather flock together.
every cloud has a silver lining. fortune favors the bold.
the best things in life are free. honesty is the best policy.
if at first you do not succeed try try again. rome was not built in a day.
the grass is always greener on the other side. curiosity killed the cat.
do not count your chickens before they hatch. a penny saved is a penny earned.
two wrongs do not make a right. when in rome do as the romans do.
the squeaky wheel gets the grease. you can not judge a book by its cover."""

# Build character vocabulary from text
chars = sorted(set(TEXT.lower()))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}
VOCAB = len(chars)

# Build bigram training data
def build_bigrams(text, char_to_idx):
    text = text.lower()
    pairs = []
    for i in range(len(text) - 1):
        if text[i] in char_to_idx and text[i+1] in char_to_idx:
            pairs.append((char_to_idx[text[i]], char_to_idx[text[i+1]]))
    return pairs

# Build training sequences
def build_sequences(text, char_to_idx, seq_len=32):
    text = text.lower()
    seqs = []
    for i in range(0, len(text) - seq_len, seq_len // 2):
        seq = []
        for j in range(seq_len):
            if text[i+j] in char_to_idx:
                seq.append(char_to_idx[text[i+j]])
        if len(seq) == seq_len:
            seqs.append(seq)
    return seqs


class GraphNet:
    def __init__(self, n_neurons, n_in, n_out, density=0.06,
                 inhibition="none", group_k=5):
        self.N = n_neurons; self.n_in = n_in; self.n_out = n_out
        self.inhibition = inhibition; self.group_k = group_k
        self.last_acc = 0.0

        s = math.sqrt(2.0 / n_neurons)
        self.W = np.random.randn(n_neurons, n_neurons).astype(np.float32) * s
        self.mask = (np.random.rand(n_neurons, n_neurons) < density).astype(np.float32)
        np.fill_diagonal(self.mask, 0)
        self.addr = np.random.randn(n_neurons, 4).astype(np.float32)
        self.tw = np.random.randn(n_neurons, 4).astype(np.float32) * 0.1
        self.state = np.zeros(n_neurons, dtype=np.float32)
        self.decay = 0.5

    def _inhibit(self, act):
        if self.inhibition != "group_wta": return act
        internal = slice(self.n_in, self.N - self.n_out)
        vals = act[internal].copy()
        n_int = len(vals)
        group_size = max(2, n_int // self.group_k)
        for g in range(0, n_int, group_size):
            end = min(g + group_size, n_int)
            group = vals[g:end]
            if len(group) == 0: continue
            winner = np.argmax(np.abs(group))
            mask_g = np.full(len(group), np.float32(0.1))
            mask_g[winner] = np.float32(1.0)
            vals[g:end] = group * mask_g
        act[internal] = vals
        return act

    def reset(self): self.state = np.zeros(self.N, dtype=np.float32)

    def forward(self, world, diff, ticks=6):
        inp = np.concatenate([world, diff])
        act = self.state.copy(); Weff = self.W * self.mask
        for t in range(ticks):
            act = act * self.decay; act[:self.n_in] = inp
            raw = act @ Weff + act * 0.1
            act = np.where(raw > 0, raw, np.float32(0.01) * raw)
            act = self._inhibit(act)
            act[:self.n_in] = inp
        self.state = act.copy()
        # Inverse arousal self-wire
        if self.last_acc < 0.15: tk,mn = 2,1
        elif self.last_acc < 0.4: tk,mn = 3,2
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
                    self.W[ni, near] = np.float32(random.gauss(0, math.sqrt(2.0/self.N)))
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
            dead = np.argwhere(self.mask == 0); dead = dead[dead[:,0]!=dead[:,1]]
            if len(dead) > 0:
                n = max(1, int(len(dead)*rate))
                idx = dead[np.random.choice(len(dead), min(n,len(dead)), replace=False)]
                for j in range(len(idx)):
                    self.mask[int(idx[j][0]),int(idx[j][1])] = 1
                    self.W[int(idx[j][0]),int(idx[j][1])] = np.float32(random.gauss(0, math.sqrt(2.0/self.N)))
        elif a == "remove":
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
                    r,c = int(idx[j][0]),int(idx[j][1]); self.mask[r,c] = 0
                    nc = random.randint(0, self.N-1)
                    while nc == r: nc = random.randint(0, self.N-1)
                    self.mask[r,nc] = 1; self.W[r,nc] = self.W[r,c]

    def mutate_weights(self, scale=0.05):
        self.W += np.random.randn(*self.W.shape).astype(np.float32)*scale*self.mask
        self.tw += np.random.randn(*self.tw.shape).astype(np.float32)*scale*0.5
        self.addr += np.random.randn(*self.addr.shape).astype(np.float32)*scale*0.2


def eval_bigram(net, pairs, vocab, sample_n=100):
    """Evaluate on bigram prediction — given char, predict next."""
    net.reset()
    prev_diff = np.zeros(vocab, dtype=np.float32)
    correct = 0
    total = 0
    total_prob = 0.0

    # Sample subset for speed
    sample = random.sample(pairs, min(sample_n, len(pairs)))

    for inp_idx, tgt_idx in sample:
        world = np.zeros(vocab, dtype=np.float32)
        world[inp_idx] = 1.0
        logits = net.forward(world, prev_diff, ticks=6)
        probs = softmax(logits[:vocab])

        if np.argmax(probs) == tgt_idx:
            correct += 1
        total_prob += probs[tgt_idx]
        total += 1

        target_vec = np.zeros(vocab, dtype=np.float32)
        target_vec[tgt_idx] = 1.0
        prev_diff = target_vec - probs

    acc = correct / total
    avg_prob = total_prob / total
    return acc, avg_prob


def eval_sequence(net, seqs, vocab, sample_n=4):
    """Evaluate on sequence prediction — running text."""
    sample = random.sample(seqs, min(sample_n, len(seqs)))
    correct = 0; total = 0; total_prob = 0.0

    for seq in sample:
        net.reset()
        prev_diff = np.zeros(vocab, dtype=np.float32)
        for i in range(len(seq) - 1):
            world = np.zeros(vocab, dtype=np.float32)
            world[seq[i]] = 1.0
            logits = net.forward(world, prev_diff, ticks=6)
            probs = softmax(logits[:vocab])
            tgt = seq[i+1]
            if np.argmax(probs) == tgt: correct += 1
            total_prob += probs[tgt]
            total += 1
            tv = np.zeros(vocab, dtype=np.float32)
            tv[tgt] = 1.0
            prev_diff = tv - probs

    return correct/max(1,total), total_prob/max(1,total)


def run_test(label, inhibition, group_k, eval_mode, max_att=5000, seed=42):
    np.random.seed(seed); random.seed(seed)
    n_in = VOCAB * 2; n_out = VOCAB
    n_neurons = n_in + 48 + n_out  # 48 internal neurons
    ticks = 6

    net = GraphNet(n_neurons, n_in, n_out, 0.08, inhibition, group_k)

    pairs = build_bigrams(TEXT, char_to_idx)
    seqs = build_sequences(TEXT, char_to_idx, 24)

    if eval_mode == "bigram":
        ev = lambda: eval_bigram(net, pairs, VOCAB, 80)
    else:
        ev = lambda: eval_sequence(net, seqs, VOCAB, 3)

    acc, prob = ev()
    score = 0.5*acc + 0.5*prob
    best_score = score; best_acc = acc
    phase = "S"; kept = 0; stale = 0; switched = False
    t0 = time.time(); curve = []

    for att in range(max_att):
        s = net.save()
        if phase == "S": net.mutate_struct(0.05)
        else:
            if random.random() < 0.3: net.mutate_struct(0.05)
            else: net.mutate_weights(0.05)

        new_acc, new_prob = ev()
        new_score = 0.5*new_acc + 0.5*new_prob
        net.last_acc = new_acc

        if new_score > score:
            score = new_score; kept += 1; stale = 0
            best_score = max(best_score, score)
            best_acc = max(best_acc, new_acc)
        else:
            net.restore(s); stale += 1

        if phase == "S" and stale > 2000 and not switched:
            phase = "B"; switched = True; stale = 0

        if (att+1) % 500 == 0:
            elapsed = time.time() - t0
            curve.append((att+1, best_acc))
            print(f"    [{label[:20]:20s}] {att+1:5d} | "
                  f"Acc:{best_acc*100:5.1f}% | Prob:{score:.3f} | "
                  f"Conns:{net.conns():5d} | Kept:{kept:3d} | "
                  f"Phase:{phase} | {elapsed:.0f}s")

        if stale >= 5000: break

    elapsed = time.time() - t0

    # Show some predictions
    net.reset()
    prev_diff = np.zeros(VOCAB, dtype=np.float32)
    print(f"\n    Sample predictions:")
    test_chars = "the qu"
    for ch in test_chars:
        if ch in char_to_idx:
            world = np.zeros(VOCAB, dtype=np.float32)
            world[char_to_idx[ch]] = 1.0
            logits = net.forward(world, prev_diff, ticks=6)
            probs = softmax(logits[:VOCAB])
            top3_idx = np.argsort(probs)[-3:][::-1]
            top3 = [(idx_to_char[i], probs[i]) for i in top3_idx]
            print(f"    '{ch}' -> {top3[0][0]}({top3[0][1]:.2f}) "
                  f"{top3[1][0]}({top3[1][1]:.2f}) "
                  f"{top3[2][0]}({top3[2][1]:.2f})")
            tv = np.zeros(VOCAB, dtype=np.float32)
            prev_diff = -probs  # no target, just negative

    return {"label": label, "acc": best_acc, "score": best_score,
            "time": elapsed, "conns": net.conns(), "curve": curve}


# =========================================================================
print("="*65)
print("REAL DATA TEST — English character prediction")
print("="*65)
print(f"Vocabulary: {VOCAB} chars: {''.join(chars)}")
print(f"Text length: {len(TEXT)} chars")
print(f"Random baseline: {100/VOCAB:.1f}%\n")

# Test bigram prediction
print("--- BIGRAM PREDICTION (given char, predict next) ---\n")

configs = [
    ("Baseline",    "none",      5, "bigram"),
    ("Group WTA k=5","group_wta", 5, "bigram"),
    ("Group WTA k=8","group_wta", 8, "bigram"),
]

results = []
for label, inhib, k, mode in configs:
    print(f"  {label}:")
    r = run_test(label, inhib, k, mode)
    results.append(r)
    print(f"  => Acc:{r['acc']*100:.1f}% (random:{100/VOCAB:.1f}%)\n")

# Test sequence prediction
print("\n--- SEQUENCE PREDICTION (running text) ---\n")

seq_configs = [
    ("Seq baseline",    "none",      5, "sequence"),
    ("Seq Group WTA",   "group_wta", 5, "sequence"),
]

for label, inhib, k, mode in seq_configs:
    print(f"  {label}:")
    r = run_test(label, inhib, k, mode)
    results.append(r)
    print(f"  => Acc:{r['acc']*100:.1f}%\n")

# Summary
print(f"\n{'='*65}")
print(f"SUMMARY")
print(f"{'='*65}")
print(f"Random baseline: {100/VOCAB:.1f}%")
print(f"{'Config':<20} {'Acc':>6} {'vs random':>10}")
print(f"{'-'*20} {'-'*6} {'-'*10}")
for r in results:
    ratio = r['acc'] / (1/VOCAB)
    print(f"{r['label']:<20} {r['acc']*100:5.1f}% {ratio:>9.1f}x")
