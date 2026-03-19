"""
probe_undetach.py — LCX write value gradient probe (#97)

Question: does un-detaching write values (allowing gradient to flow through
what the model stores) improve Phase B content integration?

Configs:
  A — current:  topk_values = values.detach()[topk_idx]  (baseline)
  B — proposed: topk_values = values[topk_idx]           (un-detached)

Keys stay detached in both configs (addressing only, no gradient needed).

Setup: D=128, depth=4, batch=32, tt=1, slots=1000, top_k=2
       delay_echo gap=8, seeds [42, 137], 1000 steps, CPU fp32
Win:   delta > 0.010 AND seed_gap < 0.010 AND no divergence
"""

import sys, os, time, math, copy
sys.path.insert(0, r'S:\AI\work\VRAXION_DEV\Diamond Code')
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU only

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Config ──────────────────────────────────────────────────────────────────
SEEDS       = [42, 137]
STEPS       = 1000
BATCH       = 32
D           = 128
DEPTH       = 4
SLOTS       = 1000
TOP_K       = 2
GAP         = 8
SEQ_LEN     = 16
LR          = 1e-3
STEP_TIMEOUT = 60  # seconds

LOG_PATH    = r'S:\AI\work\VRAXION_DEV\Diamond Code\logs\probe\probe_undetach_console.log'
RESULTS_PATH= r'S:\AI\work\VRAXION_DEV\Diamond Code\logs\probe\probe_undetach_results.txt'
LIVE_LOG    = r'S:\AI\work\VRAXION_DEV\Diamond Code\logs\probe\probe_live.log'

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, path):
        self.f = open(path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, msg):
        self.stdout.write(msg)
        self.f.write(msg)
        self.f.flush()
    def flush(self):
        self.stdout.flush()
        self.f.flush()

sys.stdout = Tee(LOG_PATH)

# ── Task: delay_echo ─────────────────────────────────────────────────────────
def make_batch(batch, seq_len, gap, device):
    """Generate delay_echo batch. Target[t] = input[t-gap] if t>=gap else 0."""
    x = torch.randint(0, 2, (batch, seq_len), dtype=torch.float32, device=device)
    y = torch.zeros_like(x)
    if gap < seq_len:
        y[:, gap:] = x[:, :seq_len - gap]
    return x, y

# ── Mini model ───────────────────────────────────────────────────────────────
class SimHash(nn.Module):
    def __init__(self, key_dim, n_planes=9):
        super().__init__()
        self.register_buffer('planes', torch.randn(n_planes, key_dim))

    def bucket(self, q):  # q: [B, key_dim] -> [B] int
        bits = (q @ self.planes.T > 0).int()  # [B, n_planes]
        w = 2 ** torch.arange(bits.shape[1], device=q.device)
        return (bits * w).sum(dim=1)  # [B]


class MiniLCX(nn.Module):
    def __init__(self, d, key_dim, slots, top_k, un_detach_values=False):
        super().__init__()
        self.d = d
        self.key_dim = key_dim
        self.slots = slots
        self.top_k = top_k
        self.un_detach_values = un_detach_values

        self.query_proj = nn.Linear(d, key_dim)
        self.write_gate = nn.Linear(d, 1)

        self.register_buffer('keys',   torch.zeros(slots, key_dim))
        self.register_buffer('values', torch.zeros(slots, d))
        self.register_buffer('valid',  torch.zeros(slots, dtype=torch.bool))
        self.register_buffer('heat',   torch.zeros(slots))

        self.simhash = SimHash(key_dim)

    def read(self, hidden):  # hidden: [B, D]
        q = F.normalize(self.query_proj(hidden), dim=-1)  # [B, key_dim]
        k = F.normalize(self.keys, dim=-1)                # [S, key_dim]

        scores = q @ k.detach().T                         # [B, S] — keys always detached
        if self.valid.any() and not self.valid.all():
            scores = scores.masked_fill(~self.valid.unsqueeze(0), float('-inf'))

        eff_k = min(self.top_k, scores.shape[-1])
        topk_scores, topk_idx = scores.topk(eff_k, dim=-1)   # [B, K]
        weights = F.softmax(topk_scores, dim=-1)              # [B, K]

        # THE KEY DIFFERENCE:
        if self.un_detach_values:
            topk_values = self.values[topk_idx]               # [B, K, D] — gradient flows
        else:
            topk_values = self.values.detach()[topk_idx]      # [B, K, D] — gradient blocked

        context = (weights.unsqueeze(-1) * topk_values).sum(dim=1)  # [B, D]

        # score margin telemetry (guard against -inf scores at cold start)
        if eff_k >= 2 and not torch.isinf(topk_scores).any():
            margin = (topk_scores[:, 0] - topk_scores[:, 1]).mean().item()
        else:
            margin = 0.0

        return context, margin, topk_idx[:, 0]

    def write(self, hidden, top_slot):  # EMA write — always no_grad on buffer
        with torch.no_grad():
            gate = torch.sigmoid(self.write_gate(hidden)).squeeze(-1).mean()  # scalar
            write_val = hidden.detach().mean(dim=0)  # [D]
            write_key = F.normalize(self.query_proj(hidden.detach()).mean(dim=0), dim=-1)  # [key_dim]

            slot = top_slot[0].item()
            self.keys[slot]   = (1 - gate) * self.keys[slot]   + gate * write_key
            self.values[slot] = (1 - gate) * self.values[slot] + gate * write_val
            self.valid[slot]  = True
            self.heat[slot]   += 1.0


class MiniModel(nn.Module):
    def __init__(self, d, depth, lcx):
        super().__init__()
        self.input_proj = nn.Linear(1, d)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            for _ in range(depth)
        ])
        self.lcx = lcx

        # bottleneck projection (2x, locked design)
        bn_dim = max(d // 10, 8)
        self.bn = nn.Sequential(
            nn.Linear(d, bn_dim),
            nn.GELU(),
            nn.Linear(bn_dim, bn_dim),
            nn.GELU(),
            nn.Linear(bn_dim, d),
        )
        self.zoom_gate = nn.Parameter(torch.zeros(1))
        self.output_proj = nn.Linear(d, 1)

    def forward(self, x):  # x: [B, T]
        B, T = x.shape
        outputs = []
        hidden = torch.zeros(B, self.input_proj.out_features, device=x.device)
        margins = []

        for t in range(T):
            inp = self.input_proj(x[:, t:t+1])   # [B, D]
            hidden = hidden + inp
            for layer in self.layers:
                hidden = hidden + layer(hidden)

            # LCX read + integrate
            context, margin, top_slot = self.lcx.read(hidden)
            gate = torch.sigmoid(self.zoom_gate)
            hidden = hidden + self.bn(context) * gate

            self.lcx.write(hidden, top_slot)

            out = self.output_proj(hidden)        # [B, 1]
            outputs.append(out)
            margins.append(margin)

        return torch.stack(outputs, dim=1).squeeze(-1), sum(margins) / len(margins)


# ── Eval ─────────────────────────────────────────────────────────────────────
def eval_config(name, un_detach_values, seed):
    torch.manual_seed(seed)
    device = torch.device('cpu')

    lcx = MiniLCX(D, D // 2, SLOTS, TOP_K, un_detach_values=un_detach_values)
    model = MiniModel(D, DEPTH, lcx).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    tail_accs = []
    margins = []
    had_nan = False
    had_div = False

    for step in range(STEPS):
        if step % 100 == 0:
            print(f'    step {step}...', flush=True)

        t0 = time.time()
        x, y = make_batch(BATCH, SEQ_LEN, GAP, device)

        opt.zero_grad()
        pred, margin = model(x)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        loss.backward()

        # gradient clip
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        elapsed = time.time() - t0
        if elapsed > STEP_TIMEOUT:
            print(f'TIMEOUT: step {step} took {elapsed:.0f}s, aborting')
            sys.exit(1)

        # metrics
        with torch.no_grad():
            acc = ((pred > 0).float() == y).float().mean().item()

        if math.isnan(loss.item()):
            had_nan = True
            break
        if loss.item() > 3.0 and step > 100:
            had_div = True
            break

        if step >= 900:
            tail_accs.append(acc)
        margins.append(margin)

        # live log
        s_per_step = elapsed
        with open(LIVE_LOG, 'a', encoding='utf-8') as lf:
            lf.write(f'step {step} | loss {loss.item():.6f} | acc={acc:.4f} RD:{s_per_step:.4f} traction={acc:.4f} shard=0/0\n')

    tail_median = sorted(tail_accs)[len(tail_accs)//2] if tail_accs else 0.0
    score_margin = sum(margins[-100:]) / max(len(margins[-100:]), 1)

    return tail_median, score_margin, had_nan, had_div


# ── Main ─────────────────────────────────────────────────────────────────────
print('=' * 60)
print('probe_undetach -- LCX value write gradient probe (#97)')
print('CPU ONLY -- quarantined from GPU training')
print('=' * 60)
print(f'  D={D}, depth={DEPTH}, batch={BATCH}, slots={SLOTS}, top_k={TOP_K}')
print(f'  task: delay_echo gap={GAP}, seq_len={SEQ_LEN}')
print(f'  steps: {STEPS}, seeds: {SEEDS}')
print(f'  win condition: delta > 0.010, seed_gap < 0.010, no divergence')
print()

results = {}

for config_name, un_detach in [('A_detached', False), ('B_undetached', True)]:
    print('=' * 60)
    print(f'  CONFIG {config_name}  (un_detach_values={un_detach})')
    print('=' * 60)

    seed_tails = []
    seed_margins = []

    for seed in SEEDS:
        print(f'  seed={seed}:')
        tail, margin, had_nan, had_div = eval_config(config_name, un_detach, seed)
        seed_tails.append(tail)
        seed_margins.append(margin)
        status = 'nan=True' if had_nan else ('div=True' if had_div else 'nan=False div=False')
        print(f'    DONE | tail={tail:.3f} | margin={margin:.4f} | {status}')

    mean_tail = sum(seed_tails) / len(seed_tails)
    seed_gap  = max(seed_tails) - min(seed_tails)
    mean_margin = sum(seed_margins) / len(seed_margins)
    results[config_name] = dict(mean_tail=mean_tail, seed_gap=seed_gap, mean_margin=mean_margin)

    print(f'  {config_name} | mean_tail={mean_tail:.4f} | seed_gap={seed_gap:.4f} | margin={mean_margin:.4f}')
    print()

# ── Decision ─────────────────────────────────────────────────────────────────
A = results['A_detached']
B = results['B_undetached']
delta = B['mean_tail'] - A['mean_tail']

print('=' * 60)
print('  RESULTS SUMMARY')
print('=' * 60)
print(f'  Config A (detached):   mean_tail={A["mean_tail"]:.4f}  seed_gap={A["seed_gap"]:.4f}')
print(f'  Config B (undetached): mean_tail={B["mean_tail"]:.4f}  seed_gap={B["seed_gap"]:.4f}')
print(f'  Delta (B - A): {delta:+.4f}')
print()

win = delta > 0.010 and B['seed_gap'] < 0.010
if win:
    verdict = 'WIN -- un-detach values. Apply to swarm_model.py lines 2339 + 2377.'
elif delta > 0.005:
    verdict = 'MARGINAL -- delta above noise but below threshold. Consider re-running with more steps.'
elif delta < -0.005:
    verdict = 'DETACH IS BETTER -- un-detaching hurts. Keep values.detach().'
else:
    verdict = 'FLAT -- no difference. Phase B blocker is elsewhere.'

print(f'  VERDICT: {verdict}')
print()

# Write results file
with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
    f.write(f'probe_undetach | {time.strftime("%Y-%m-%d %H:%M:%S")} | slots={SLOTS} top_k={TOP_K}\n\n')
    f.write(f'{"config":<20} {"mean_tail":>10} {"seed_gap":>10} {"margin":>10} {"vs_A":>10}\n')
    f.write('-' * 62 + '\n')
    f.write(f'{"A_detached":<20} {A["mean_tail"]:>10.4f} {A["seed_gap"]:>10.4f} {A["mean_margin"]:>10.4f} {"baseline":>10}\n')
    f.write(f'{"B_undetached":<20} {B["mean_tail"]:>10.4f} {B["seed_gap"]:>10.4f} {B["mean_margin"]:>10.4f} {delta:>+10.4f}\n')
    f.write(f'\nVERDICT: {verdict}\n')

print(f'Results written to: {RESULTS_PATH}')
print('Done.')
