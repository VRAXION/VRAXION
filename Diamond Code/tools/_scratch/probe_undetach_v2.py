"""
probe_undetach_v2.py -- LCX value write gradient probe, v2 (#97)

Uses real swarm_model.py via monkey-patch (no file edits, GPU-safe).
Separate log file -- does NOT touch probe_live.log (GPU training uses that).

Key fixes vs v1:
  - Real swarm_model.py, not a simplified MiniLCX
  - Monkey-patches lines 2339+2377 at runtime via subclass override
  - Task: delay_echo256 at seq_len=32 (gap=16, forces LCX retrieval)
  - 500 steps -- enough to see routing + content phases
  - Checks Score Margin trajectory (routing) AND bit_acc (content)

Win condition: delta > 0.010 AND seed_gap < 0.010 AND no divergence
"""

import sys, os, time, math
sys.path.insert(0, r'S:\AI\work\VRAXION_DEV\Diamond Code')
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU only -- do NOT touch GPU

import torch
import torch.nn.functional as F

# ── Paths ────────────────────────────────────────────────────────────────────
LOG_PATH     = r'S:\AI\work\VRAXION_DEV\Diamond Code\logs\probe\probe_undetach_v2_console.log'
RESULTS_PATH = r'S:\AI\work\VRAXION_DEV\Diamond Code\logs\probe\probe_undetach_v2_results.txt'
LIVE_LOG     = r'S:\AI\work\VRAXION_DEV\Diamond Code\logs\probe\probe_undetach_v2_live.log'
# NOTE: deliberately NOT writing to probe_live.log -- GPU training owns that file

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# ── Tee logger ───────────────────────────────────────────────────────────────
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

# ── Import real model ─────────────────────────────────────────────────────────
print('Loading swarm_model.py...', flush=True)
from swarm_model import SwarmByteRingModel
print('OK', flush=True)

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS   = [42, 137]
STEPS   = 500
BATCH   = 16
SEQ_LEN = 32      # longer sequence -- forces LCX retrieval for delay_echo
LR      = 1e-4
STEP_TIMEOUT = 120

# Mini model config (CPU-safe)
MODEL_CFG = dict(
    embedding_dim        = 128,
    depth                = 4,
    num_beings           = 1,
    num_bits             = 8,
    num_memory_positions = SEQ_LEN,
    use_lcx              = True,
    lcx_mode             = 'hash',
    lcx_num_levels       = 1,
    lcx_level_slots      = [500],
    lcx_top_k            = 2,
    lcx_key_dim          = 13,
    attention_radius     = 3,
    think_ticks          = 1,
    byte_token_mode      = True,
)

# ── Task: delay_echo (gap = seq_len // 2) ────────────────────────────────────
GAP = SEQ_LEN // 2   # 16 -- answer is outside naive attention window

def make_batch(batch, seq_len, gap, device):
    x = torch.randint(0, 256, (batch, seq_len), dtype=torch.long, device=device)
    y = torch.zeros_like(x)
    if gap < seq_len:
        y[:, gap:] = x[:, :seq_len - gap]
    return x, y

# ── Monkey-patch: un_detach_values flag ──────────────────────────────────────
# Overrides _lcx_flat_read to skip .detach() on values when flag is set.
# Does NOT modify swarm_model.py on disk -- GPU training is safe.

_ORIG_LCX_READ = SwarmByteRingModel._lcx_flat_read

def _patched_lcx_read_level(self, state, level):
    """Drop-in replacement that skips values.detach() when un_detach_values=True."""
    squeeze = state.dim() == 1
    if squeeze:
        state = state.unsqueeze(0)
    B = state.shape[0]
    k = self.lcx_top_k

    query = self.lcx_route_query(state)
    query = F.normalize(query, dim=-1)

    keys, values = self._lcx_level_bufs(level)
    valid   = getattr(self, f'lcx_valid_{level}', None)
    temp    = self._lcx_route_temps[min(level, len(self._lcx_route_temps) - 1)]
    is_bucketed = getattr(self, f'_lcx_bucketed_{level}', False)

    if not is_bucketed:
        scores = query @ keys.detach().clone().T
        if valid is not None and valid.any() and not valid.all():
            scores = scores.masked_fill(~valid.unsqueeze(0), float('-inf'))
        if temp != 1.0:
            scores = scores / temp
        eff_k = min(k, scores.shape[-1])
        topk_scores, topk_idx = scores.topk(eff_k, dim=-1)
        weights = F.softmax(topk_scores, dim=-1)

        # THE KEY DIFFERENCE
        if getattr(self, '_un_detach_values', False):
            topk_values = values[topk_idx]             # gradient FLOWS
        else:
            topk_values = values.detach()[topk_idx]    # gradient BLOCKED (default)

    else:
        # bucketed path
        bucket_index  = getattr(self, f'_lcx_bucket_index_{level}')
        bucket_counts = getattr(self, f'_lcx_bucket_counts_{level}')
        max_bkt       = getattr(self, f'_lcx_max_bucket_size_{level}')
        query_bucket  = self._lcx_compute_bucket(query, level)
        sel_indices   = bucket_index[query_bucket]
        sel_counts    = bucket_counts[query_bucket]
        flat_idx      = sel_indices.clamp(min=0)
        sel_keys      = keys.detach()[flat_idx]
        q_norm        = F.normalize(query, dim=-1).unsqueeze(2)
        k_norm        = F.normalize(sel_keys, dim=-1)
        scores        = (k_norm @ q_norm).squeeze(2)

        mask = torch.arange(max_bkt, device=query.device).unsqueeze(0) >= sel_counts.unsqueeze(1)
        if valid is not None:
            sel_valid = valid[flat_idx]
            mask = mask | ~sel_valid
        scores = scores.masked_fill(mask, float('-inf'))
        if temp != 1.0:
            scores = scores / temp

        eff_k = min(k, max_bkt)
        topk_scores, topk_local = scores.topk(eff_k, dim=-1)
        topk_idx = flat_idx.gather(1, topk_local)
        weights  = F.softmax(topk_scores, dim=-1)

        if getattr(self, '_un_detach_values', False):
            topk_values = values[topk_idx]
        else:
            topk_values = values.detach()[topk_idx]

    context = (weights.unsqueeze(-1) * topk_values).sum(dim=1)

    # score margin telemetry
    if topk_scores.shape[-1] >= 2 and not torch.isinf(topk_scores).any():
        margin = (topk_scores[:, 0] - topk_scores[:, 1]).mean().item()
    else:
        margin = 0.0

    # store margin for retrieval
    self._last_lcx_margin = margin
    return context

# Apply patch
SwarmByteRingModel._lcx_flat_read = _patched_lcx_read_level
print('Monkey-patch applied to _lcx_flat_read', flush=True)

# ── Run one config ────────────────────────────────────────────────────────────
def run_config(label, un_detach, seed):
    torch.manual_seed(seed)
    device = torch.device('cpu')

    model = SwarmByteRingModel(**MODEL_CFG).to(device)
    model._un_detach_values = un_detach   # flag read by patched method
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    tail_accs   = []
    margins_log = []
    had_nan = had_div = False

    for step in range(STEPS):
        if step % 100 == 0:
            print(f'    step {step}...', flush=True)

        t0 = time.time()
        x, y = make_batch(BATCH, SEQ_LEN, GAP, device)

        opt.zero_grad()
        out, stats = model(x, return_stats=True)

        # byte_token_mode: out is [B, T, 256] -- CrossEntropy over byte classes
        loss = F.cross_entropy(out.float().reshape(-1, 256), y.reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # detach LCX buffers between steps (prevent cross-step graph growth)
        with torch.no_grad():
            for lvl in range(model._lcx_num_levels):
                k_buf, v_buf = model._lcx_level_bufs(lvl)
                if v_buf.requires_grad:
                    setattr(model, f'lcx_values_{lvl}', v_buf.detach())
                if k_buf.requires_grad:
                    setattr(model, f'lcx_keys_{lvl}', k_buf.detach())

        elapsed = time.time() - t0
        if elapsed > STEP_TIMEOUT:
            print(f'TIMEOUT: step {step} took {elapsed:.0f}s, aborting')
            sys.exit(1)

        with torch.no_grad():
            pred_byte = out.argmax(dim=-1)           # [B, T]
            acc = (pred_byte == y).float().mean().item()  # byte accuracy

        if math.isnan(loss.item()):
            had_nan = True; break
        if loss.item() > 5.0 and step > 100:
            had_div = True; break

        margin = getattr(model, '_last_lcx_margin', 0.0)
        margins_log.append(margin)

        if step >= 400:
            tail_accs.append(acc)

        # live log (separate file -- not probe_live.log)
        with open(LIVE_LOG, 'a', encoding='utf-8') as lf:
            lf.write(f'[{label}] step {step} | loss {loss.item():.6f} | acc={acc:.4f} margin={margin:.4f} RD:{elapsed:.4f}\n')

    tail_median  = sorted(tail_accs)[len(tail_accs)//2] if tail_accs else 0.0
    margin_early = sum(margins_log[:50])  / max(len(margins_log[:50]),  1)
    margin_late  = sum(margins_log[-50:]) / max(len(margins_log[-50:]), 1)
    status = 'nan=True' if had_nan else ('div=True' if had_div else 'clean')

    print(f'    DONE | tail={tail_median:.3f} | margin {margin_early:.4f}->{margin_late:.4f} | {status}')
    return tail_median, margin_early, margin_late, had_nan, had_div

# ── Main ──────────────────────────────────────────────────────────────────────
print()
print('=' * 62)
print('probe_undetach_v2 -- real swarm_model.py, harder task (#97)')
print('CPU ONLY -- separate log, GPU training unaffected')
print('=' * 62)
print(f'  D=128, depth=4, batch={BATCH}, seq_len={SEQ_LEN}, gap={GAP}')
print(f'  slots=500, top_k=2, steps={STEPS}, seeds={SEEDS}')
print(f'  win: delta > 0.010, seed_gap < 0.010, no divergence')
print()

results = {}
for label, un_detach in [('A_detached', False), ('B_undetached', True)]:
    print('=' * 62)
    print(f'  CONFIG {label}  (un_detach={un_detach})')
    print('=' * 62)

    seed_tails = []
    seed_m_early = []
    seed_m_late  = []

    for seed in SEEDS:
        print(f'  seed={seed}:')
        tail, me, ml, nan, div = run_config(label, un_detach, seed)
        seed_tails.append(tail)
        seed_m_early.append(me)
        seed_m_late.append(ml)

    mean_tail   = sum(seed_tails)   / len(seed_tails)
    seed_gap    = max(seed_tails)   - min(seed_tails)
    mean_m_early= sum(seed_m_early) / len(seed_m_early)
    mean_m_late = sum(seed_m_late)  / len(seed_m_late)
    results[label] = dict(mean_tail=mean_tail, seed_gap=seed_gap,
                          m_early=mean_m_early, m_late=mean_m_late)
    print(f'  {label}: mean_tail={mean_tail:.4f} seed_gap={seed_gap:.4f} margin {mean_m_early:.4f}->{mean_m_late:.4f}')
    print()

# ── Decision ──────────────────────────────────────────────────────────────────
A = results['A_detached']
B = results['B_undetached']
delta = B['mean_tail'] - A['mean_tail']

print('=' * 62)
print('  FINAL RESULTS')
print('=' * 62)
print(f'  A detached:   mean_tail={A["mean_tail"]:.4f}  seed_gap={A["seed_gap"]:.4f}  margin {A["m_early"]:.4f}->{A["m_late"]:.4f}')
print(f'  B undetached: mean_tail={B["mean_tail"]:.4f}  seed_gap={B["seed_gap"]:.4f}  margin {B["m_early"]:.4f}->{B["m_late"]:.4f}')
print(f'  Delta (B-A):  {delta:+.4f}')
print()

if delta > 0.010 and B['seed_gap'] < 0.010:
    verdict = 'WIN -- apply un-detach to swarm_model.py lines 2339+2377'
elif delta > 0.005:
    verdict = 'MARGINAL -- weak signal, re-run with more steps'
elif delta < -0.005:
    verdict = 'DETACH IS BETTER -- keep values.detach()'
else:
    verdict = 'FLAT -- Phase B blocker is elsewhere'

print(f'  VERDICT: {verdict}')

with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
    f.write(f'probe_undetach_v2 | {time.strftime("%Y-%m-%d %H:%M:%S")}\n\n')
    f.write(f'{"config":<16} {"mean_tail":>10} {"seed_gap":>10} {"m_early":>10} {"m_late":>10} {"vs_A":>8}\n')
    f.write('-' * 66 + '\n')
    f.write(f'{"A_detached":<16} {A["mean_tail"]:>10.4f} {A["seed_gap"]:>10.4f} {A["m_early"]:>10.4f} {A["m_late"]:>10.4f} {"base":>8}\n')
    f.write(f'{"B_undetached":<16} {B["mean_tail"]:>10.4f} {B["seed_gap"]:>10.4f} {B["m_early"]:>10.4f} {B["m_late"]:>10.4f} {delta:>+8.4f}\n')
    f.write(f'\nVERDICT: {verdict}\n')

print(f'\nResults: {RESULTS_PATH}')
print('Done.')
