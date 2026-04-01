"""
No-Backprop v7 — DIAGNOSTIC / ADVERSARIAL TESTING
====================================================
Sanity checks to find exactly what's broken.
"""

import torch
import torch.nn.functional as F
import math


class BrainNet:
    def __init__(self, sizes, oversize=4):
        self.sizes = sizes.copy()
        self.target = sizes.copy()
        for i in range(1, len(self.sizes) - 1):
            self.sizes[i] *= oversize

        self.W = []
        self.b = []
        for i in range(len(self.sizes) - 1):
            s = math.sqrt(2.0 / self.sizes[i])
            self.W.append(torch.randn(self.sizes[i], self.sizes[i+1]) * s)
            self.b.append(torch.zeros(self.sizes[i+1]))

        self.alive = [torch.ones(self.sizes[i]) for i in range(1, len(self.sizes)-1)]
        self.survival = [torch.zeros(self.sizes[i]) for i in range(1, len(self.sizes)-1)]
        self.acts = []

    def forward(self, x):
        self.acts = [x]
        h = x
        for i in range(len(self.W)):
            h = h @ self.W[i] + self.b[i]
            if i < len(self.W) - 1:
                h = torch.relu(h)
                h = h * self.alive[i]
            self.acts.append(h)
        return h

    def learn(self, dopamine, pain, lr=0.02):
        for i in range(len(self.W)):
            pre = self.acts[i].detach()
            post = self.acts[i+1].detach()
            if pre.dim() > 1:
                pre = pre.mean(0)
                post = post.mean(0)
            pre_norm = pre / (pre.norm() + 1e-8)
            post_norm = post / (post.norm() + 1e-8)

            if dopamine > 0:
                delta = lr * dopamine * torch.outer(pre_norm, post_norm)
                delta = delta.clamp(-0.1, 0.1)
                self.W[i] += delta
                self.b[i] += (lr * 0.1 * dopamine * post_norm).clamp(-0.01, 0.01)

            if pain > 0:
                post_strength = post.abs()
                strong_mask = (post_strength > post_strength.median()).float()
                anti_delta = -lr * pain * torch.outer(pre_norm, post_norm * strong_mask)
                anti_delta = anti_delta.clamp(-0.1, 0.1)
                self.W[i] += anti_delta

                noise = torch.randn_like(self.W[i]) * (lr * pain * 0.05)
                self.W[i] += noise

                if i == len(self.W) - 1:
                    dominant = post.argmax()
                    self.W[i][:, dominant] -= lr * pain * pre_norm * 0.5
                    boost = torch.ones(post.shape[0]) * 0.1
                    boost[dominant] = 0
                    self.b[i] += lr * pain * boost * 0.01

            if i < len(self.alive):
                active = (post.abs() > 0.01).float()
                self.survival[i] += (dopamine - pain * 0.5) * active * 0.1


def compute_signals(pred, target, vocab):
    """Same signal computation as in run()."""
    prob_correct = pred[target].item()
    predicted = pred.argmax().item()
    dopamine = max(0, (prob_correct - 1.0/vocab) * 8)
    if predicted != target:
        pain = pred[predicted].item() * 4
    else:
        pain = 0.0
    return dopamine, pain


def test_sanity_1_single_input():
    """Can it learn ONE single mapping? 0 → 3"""
    print("\n" + "="*60)
    print("  SANITY 1: Can it learn ONE mapping? (0 → 3)")
    print("="*60)

    torch.manual_seed(42)
    vocab = 8
    net = BrainNet([vocab, 16, 16, vocab], oversize=1)  # no oversize

    target = 3
    for epoch in range(500):
        x = torch.zeros(vocab)
        x[0] = 1.0
        logits = net.forward(x)
        pred = torch.softmax(logits, dim=-1)

        dopamine, pain = compute_signals(pred, target, vocab)
        net.learn(dopamine, pain, lr=0.02)

        if (epoch+1) % 100 == 0 or epoch == 0:
            predicted = pred.argmax().item()
            print(f"  Epoch {epoch+1:4d} | pred: {predicted} (exp: {target}) | "
                  f"p(correct): {pred[target].item():.3f} | "
                  f"dop: {dopamine:.3f} pain: {pain:.3f}")

    result = "PASS" if pred.argmax().item() == target else "FAIL"
    print(f"  → {result}")
    return result == "PASS"


def test_sanity_2_two_inputs():
    """Can it learn TWO mappings without interference? 0→3, 1→5"""
    print("\n" + "="*60)
    print("  SANITY 2: Two mappings (0→3, 1→5) — interference?")
    print("="*60)

    torch.manual_seed(42)
    vocab = 8
    net = BrainNet([vocab, 16, 16, vocab], oversize=1)
    mappings = {0: 3, 1: 5}

    for epoch in range(1000):
        for inp, tgt in mappings.items():
            x = torch.zeros(vocab)
            x[inp] = 1.0
            logits = net.forward(x)
            pred = torch.softmax(logits, dim=-1)
            dopamine, pain = compute_signals(pred, tgt, vocab)
            net.learn(dopamine, pain, lr=0.02)

        if (epoch+1) % 200 == 0 or epoch == 0:
            results = []
            for inp, tgt in mappings.items():
                x = torch.zeros(vocab)
                x[inp] = 1.0
                logits = net.forward(x)
                pred = torch.softmax(logits, dim=-1)
                predicted = pred.argmax().item()
                results.append(f"{inp}→{predicted}(exp:{tgt}) p={pred[tgt].item():.3f}")
            print(f"  Epoch {epoch+1:4d} | {' | '.join(results)}")

    correct = 0
    for inp, tgt in mappings.items():
        x = torch.zeros(vocab)
        x[inp] = 1.0
        if net.forward(x).argmax().item() == tgt:
            correct += 1
    result = "PASS" if correct == 2 else f"FAIL ({correct}/2)"
    print(f"  → {result}")
    return correct == 2


def test_sanity_3_weight_dynamics():
    """Track what actually happens to weights during learning."""
    print("\n" + "="*60)
    print("  SANITY 3: Weight dynamics — what's actually changing?")
    print("="*60)

    torch.manual_seed(42)
    vocab = 4
    net = BrainNet([vocab, 8, vocab], oversize=1)  # tiny, 2 layers

    target = 2
    x = torch.zeros(vocab)
    x[0] = 1.0

    W0_start = net.W[0].clone()
    W1_start = net.W[1].clone()

    for epoch in range(200):
        logits = net.forward(x)
        pred = torch.softmax(logits, dim=-1)
        dopamine, pain = compute_signals(pred, target, vocab)
        net.learn(dopamine, pain, lr=0.02)

        if (epoch+1) % 50 == 0:
            W0_diff = (net.W[0] - W0_start).abs().mean().item()
            W1_diff = (net.W[1] - W1_start).abs().mean().item()
            W0_norm = net.W[0].norm().item()
            W1_norm = net.W[1].norm().item()
            predicted = pred.argmax().item()
            print(f"  Epoch {epoch+1:4d} | pred: {predicted}(exp:{target}) | "
                  f"W0 Δ={W0_diff:.4f} norm={W0_norm:.2f} | "
                  f"W1 Δ={W1_diff:.4f} norm={W1_norm:.2f} | "
                  f"dop={dopamine:.3f} pain={pain:.3f}")

    # Check: are output logits for target growing?
    logits = net.forward(x)
    print(f"\n  Final logits: {[f'{l:.3f}' for l in logits.tolist()]}")
    print(f"  Final probs:  {[f'{p:.3f}' for p in torch.softmax(logits, dim=-1).tolist()]}")
    print(f"  Target={target}, predicted={logits.argmax().item()}")


def test_adversarial_1_dopamine_only():
    """Disable pain — does dopamine alone do anything?"""
    print("\n" + "="*60)
    print("  ADVERSARIAL 1: Dopamine ONLY (no pain) — does it learn?")
    print("="*60)

    torch.manual_seed(42)
    vocab = 8
    net = BrainNet([vocab, 16, 16, vocab], oversize=1)
    perm = torch.tensor([6, 3, 0, 7, 2, 1, 4, 5])

    for epoch in range(2000):
        correct = 0
        for val in range(vocab):
            x = torch.zeros(vocab)
            x[val] = 1.0
            target = perm[val].item()
            logits = net.forward(x)
            pred = torch.softmax(logits, dim=-1)
            dopamine, _ = compute_signals(pred, target, vocab)
            net.learn(dopamine, 0.0, lr=0.02)  # NO PAIN
            if pred.argmax().item() == target:
                correct += 1

        if (epoch+1) % 500 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d} | Acc: {correct/vocab*100:.1f}%")

    # Check collapse
    outputs = set()
    for val in range(vocab):
        x = torch.zeros(vocab)
        x[val] = 1.0
        outputs.add(net.forward(x).argmax().item())
    print(f"  Unique outputs: {outputs} ({len(outputs)} different)")
    print(f"  → {'COLLAPSED' if len(outputs) <= 2 else 'DIVERSE'}")


def test_adversarial_2_pain_only():
    """Disable dopamine — does pain alone do anything useful?"""
    print("\n" + "="*60)
    print("  ADVERSARIAL 2: Pain ONLY (no dopamine) — does it diversify?")
    print("="*60)

    torch.manual_seed(42)
    vocab = 8
    net = BrainNet([vocab, 16, 16, vocab], oversize=1)
    perm = torch.tensor([6, 3, 0, 7, 2, 1, 4, 5])

    for epoch in range(2000):
        correct = 0
        for val in range(vocab):
            x = torch.zeros(vocab)
            x[val] = 1.0
            target = perm[val].item()
            logits = net.forward(x)
            pred = torch.softmax(logits, dim=-1)
            _, pain = compute_signals(pred, target, vocab)
            net.learn(0.0, pain, lr=0.02)  # NO DOPAMINE
            if pred.argmax().item() == target:
                correct += 1

        if (epoch+1) % 500 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d} | Acc: {correct/vocab*100:.1f}%")

    outputs = set()
    for val in range(vocab):
        x = torch.zeros(vocab)
        x[val] = 1.0
        outputs.add(net.forward(x).argmax().item())
    print(f"  Unique outputs: {outputs} ({len(outputs)} different)")
    print(f"  → Pain {'DIVERSIFIES' if len(outputs) >= 4 else 'DOES NOT DIVERSIFY'}")


def test_adversarial_3_signal_magnitude():
    """How big are dopamine vs pain signals? Is one drowning the other?"""
    print("\n" + "="*60)
    print("  ADVERSARIAL 3: Signal magnitudes — who's louder?")
    print("="*60)

    torch.manual_seed(42)
    vocab = 8
    net = BrainNet([vocab, 16, 16, vocab], oversize=1)
    perm = torch.tensor([6, 3, 0, 7, 2, 1, 4, 5])

    dop_total = 0; pain_total = 0; dop_count = 0; pain_count = 0
    n_steps = 0

    for epoch in range(500):
        for val in range(vocab):
            x = torch.zeros(vocab)
            x[val] = 1.0
            target = perm[val].item()
            logits = net.forward(x)
            pred = torch.softmax(logits, dim=-1)
            dopamine, pain = compute_signals(pred, target, vocab)

            if dopamine > 0:
                dop_total += dopamine
                dop_count += 1
            if pain > 0:
                pain_total += pain
                pain_count += 1
            n_steps += 1

            net.learn(dopamine, pain, lr=0.02)

        if (epoch+1) % 100 == 0:
            avg_dop = dop_total / max(dop_count, 1)
            avg_pain = pain_total / max(pain_count, 1)
            dop_rate = dop_count / n_steps * 100
            pain_rate = pain_count / n_steps * 100
            print(f"  Epoch {epoch+1:4d} | "
                  f"Dop: avg={avg_dop:.3f} ({dop_rate:.0f}% of steps) | "
                  f"Pain: avg={avg_pain:.3f} ({pain_rate:.0f}% of steps) | "
                  f"Ratio: {pain_total/(dop_total+1e-8):.1f}x")
            dop_total = 0; pain_total = 0; dop_count = 0; pain_count = 0; n_steps = 0


def test_adversarial_4_hebbian_direction():
    """Does outer(pre, post) even point in the right direction?"""
    print("\n" + "="*60)
    print("  ADVERSARIAL 4: Does Hebbian update point right?")
    print("  (Is outer(pre,post)*reward moving toward target?)")
    print("="*60)

    torch.manual_seed(42)
    vocab = 4
    # Tiny: 1 layer, direct input→output
    net = BrainNet([vocab, vocab], oversize=1)

    x = torch.zeros(vocab)
    x[0] = 1.0
    target = 2

    for epoch in range(20):
        logits = net.forward(x)
        pred = torch.softmax(logits, dim=-1)
        dopamine, pain = compute_signals(pred, target, vocab)

        # Manual check: what does outer(pre_norm, post_norm) look like?
        pre = net.acts[0]
        post = net.acts[1]
        pre_norm = pre / (pre.norm() + 1e-8)
        post_norm = post / (post.norm() + 1e-8)
        hebbian = torch.outer(pre_norm, post_norm)

        # What we WANT: W[0][0, target] to increase
        # What Hebbian does: increases W[0][0, j] proportional to post_norm[j]
        # Problem: post_norm[j] is highest for the CURRENT prediction, not target!
        print(f"  Epoch {epoch+1:2d} | pred: {pred.argmax().item()} | "
              f"post_norm: [{', '.join(f'{v:.3f}' for v in post_norm.tolist())}] | "
              f"target col={target}: hebbian[0,{target}]={hebbian[0,target]:.4f} | "
              f"pred col={pred.argmax().item()}: hebbian[0,{pred.argmax().item()}]={hebbian[0,pred.argmax().item()]:.4f}")

        net.learn(dopamine, pain, lr=0.02)

    print(f"\n  ⚠ KEY INSIGHT: Hebbian strengthens columns proportional to post_norm.")
    print(f"    post_norm is highest for CURRENT OUTPUT (whatever net predicts).")
    print(f"    So dopamine reinforces WHAT THE NET ALREADY DOES,")
    print(f"    not what it SHOULD do. This is the fundamental problem.")


def test_adversarial_5_noise_vs_learning():
    """How much of pain is noise vs structured change?"""
    print("\n" + "="*60)
    print("  ADVERSARIAL 5: Pain noise magnitude vs weight magnitude")
    print("="*60)

    torch.manual_seed(42)
    vocab = 8
    net = BrainNet([vocab, 16, 16, vocab], oversize=1)
    perm = torch.tensor([6, 3, 0, 7, 2, 1, 4, 5])

    for epoch in range(500):
        for val in range(vocab):
            x = torch.zeros(vocab)
            x[val] = 1.0
            target = perm[val].item()
            logits = net.forward(x)
            pred = torch.softmax(logits, dim=-1)
            dopamine, pain = compute_signals(pred, target, vocab)

            # Snapshot before
            W_before = [w.clone() for w in net.W]
            net.learn(dopamine, pain, lr=0.02)
            # Measure change
            if (epoch+1) % 100 == 0 and val == 0:
                for i in range(len(net.W)):
                    delta = (net.W[i] - W_before[i]).abs().mean().item()
                    wnorm = net.W[i].abs().mean().item()
                    ratio = delta / (wnorm + 1e-8)
                    print(f"  Epoch {epoch+1:4d} W[{i}] | "
                          f"Δ={delta:.6f} | |W|={wnorm:.4f} | "
                          f"Δ/|W|={ratio:.4f} ({ratio*100:.2f}%)")


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  NO-BACKPROP v7 — DIAGNOSTIC SUITE                     ║")
    print("║  Finding exactly what's broken                         ║")
    print("╚══════════════════════════════════════════════════════════╝")

    test_sanity_1_single_input()
    test_sanity_2_two_inputs()
    test_sanity_3_weight_dynamics()
    test_adversarial_1_dopamine_only()
    test_adversarial_2_pain_only()
    test_adversarial_3_signal_magnitude()
    test_adversarial_4_hebbian_direction()
    test_adversarial_5_noise_vs_learning()

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
