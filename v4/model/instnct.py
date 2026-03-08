import math                # sqrt for golden ratio constant
from contextlib import nullcontext
import torch               # tensor operations and autograd
from torch.utils.checkpoint import checkpoint as grad_checkpoint  # memory-efficient backward
import torch.nn as nn      # neural network layers (Linear, Module, ModuleList)
import torch.nn.functional as F  # softmax for dot-product attention kernel
from torch.profiler import record_function
import yaml                # YAML config parser
from pathlib import Path   # cross-platform file path resolution

# ── Constants ───────────────────────────────────────────────────
# inverse golden ratio — used in phi_destinations() to compute maximally-spaced pointer jumps
PHI_INV = (math.sqrt(5) - 1) / 2  # 1/φ ≈ 0.6180339887
PHI = (1 + math.sqrt(5)) / 2      # φ ≈ 1.6180339887 — golden ratio

# C19 period constant
_C19_C = math.pi
_SOURCE_MAP_ENABLED = False
_NO_SOURCE_SCOPE = nullcontext()
_RING_TRACE_ENABLED = False


def set_source_map_enabled(enabled: bool):
    global _SOURCE_MAP_ENABLED
    _SOURCE_MAP_ENABLED = bool(enabled)


def set_ring_trace_enabled(enabled: bool):
    global _RING_TRACE_ENABLED
    _RING_TRACE_ENABLED = bool(enabled)


def _source_scope(name: str):
    if not _SOURCE_MAP_ENABLED:
        return _NO_SOURCE_SCOPE
    return record_function(f'scope::{name}')


def _c19_activation(x, rho=4.0, C=None):
    """C19 periodic parabolic wave activation. C defaults to _C19_C (π)."""
    if C is None:
        C = _C19_C
    l = 6.0 * C
    inv_c = 1.0 / C
    scaled = x * inv_c
    n = torch.floor(scaled)
    t = scaled - n
    h = t * (1.0 - t)
    is_even = torch.remainder(n, 2.0) < 1.0
    sgn = torch.where(is_even, 1.0, -1.0)
    core = C * (sgn * h + (rho * h * h))
    # Linear tails: shift by ±l outside periodic region
    result = torch.where(x <= -l, x + l, core)
    return torch.where(x >= l, x - l, result)


def _c19_dualphi_activation(x, rho=4.0, C=None):
    """C19 with dual-phi asymmetric gain and linear tails.

    Validated +1.5% acc, 2× lower max gradient norm vs baseline C19.
    - Positive arches scaled by 1/φ (compress, stabilize)
    - Negative arches scaled by φ (amplify, signal-carrying)
    - Linear tails beyond ±6C (not periodic wrap) for gradient stability
    - rho parameter accepted but always uses 4.0 (optimal for zero-touch at midpoint)
    """
    if C is None:
        C = _C19_C
    inv_c = 1.0 / C
    scaled = x * inv_c
    n = torch.floor(scaled)
    t = scaled - n
    h = t - t * t                               # same as t*(1-t)
    odd = torch.remainder(n, 2.0)
    sgn = 1.0 - 2.0 * odd
    gain = odd * (PHI - PHI_INV) + PHI_INV      # neg→φ, pos→1/φ
    core = C * h * (sgn + 4.0 * h) * gain       # rho=4.0 hardcoded

    tail_k = 6.0
    limit = tail_k * C
    return torch.where(x.abs() > limit, x - x.sign() * limit, core)


_C19_RHO_MIN = 0.5
_C19_RHO_MAX = 8.0
_C19_C_MIN = 1.0
_C19_C_MAX = 50.0

def _sigmoid_bounded(raw, lo, hi):
    """Sigmoid-bounded parameter: raw float → [lo, hi]."""
    return lo + (hi - lo) * torch.sigmoid(raw)

def _rho_from_raw(raw_rho):
    """Sigmoid-bounded rho: raw float → [0.5, 8.0]."""
    return _sigmoid_bounded(raw_rho, _C19_RHO_MIN, _C19_RHO_MAX)

def _C_from_raw(raw_C):
    """Sigmoid-bounded C: raw float → [1.0, 10.0]."""
    return _sigmoid_bounded(raw_C, _C19_C_MIN, _C19_C_MAX)

def _init_raw(val, lo, hi):
    """Inverse sigmoid so that sigmoid(raw) maps to val within [lo, hi]."""
    p = (val - lo) / (hi - lo)
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))

def _rho_init_raw(rho_init=4.0):
    """Inverse sigmoid so that sigmoid(raw) maps to rho_init."""
    return _init_raw(rho_init, _C19_RHO_MIN, _C19_RHO_MAX)

def _C_init_raw(C_init=None):
    """Inverse sigmoid so that sigmoid(raw) maps to C_init (default π)."""
    if C_init is None:
        C_init = math.pi
    return _init_raw(C_init, _C19_C_MIN, _C19_C_MAX)


class _C19Module(nn.Module):
    """Wraps c19 as nn.Module for use in nn.Sequential (fixed rho=4.0)."""
    def forward(self, x):
        return _c19_activation(x)


class _C19LearnableModule(nn.Module):
    """C19 with per-neuron learnable rho and C, bounded via sigmoid."""
    def __init__(self, width, rho_init=4.0, learn_C=True):
        super().__init__()
        self.raw_rho = nn.Parameter(torch.full((width,), _rho_init_raw(rho_init)))
        self.learn_C = learn_C
        if learn_C:
            self.raw_C = nn.Parameter(torch.full((width,), _C_init_raw()))

    def forward(self, x):
        rho = _rho_from_raw(self.raw_rho)
        C = _C_from_raw(self.raw_C) if self.learn_C else None
        return _c19_activation(x, rho=rho, C=C)

# ── Config ──────────────────────────────────────────────────────
# All hyperparameters live in vraxion_config.yaml — single source of truth.
# Change values in the YAML, not here — these lines just bind them to Python names.

def _load_yaml(section: str) -> dict:
    """Load one section from vraxion_config.yaml. Fails loud if missing."""
    cfg_path = Path(__file__).parent.parent / 'config' / 'vraxion_config.yaml'  # model/ → v4/ → config/
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    try:
        with open(cfg_path, encoding='utf-8') as f:
            root = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise RuntimeError(f"Corrupted YAML in {cfg_path}: {e}")
    if not isinstance(root, dict):
        raise RuntimeError(f"Expected dict in {cfg_path}, got {type(root).__name__}")
    if section not in root:
        raise KeyError(f"Missing '{section}' in {cfg_path} (have: {list(root.keys())})")
    return root[section]

_cfg = _load_yaml('model')

# ─ architecture ─
CNFG_RINGSLOTS_INT = _cfg['M']                            # 256  memory slots on the ring buffer
CNFG_SLOTDIMS_INT  = _cfg.get('slot_dim', _cfg['D'])      # ring buffer slot width (small = less VRAM)
CNFG_HIDDNDIMS_INT = _cfg.get('hidden_dim', _cfg['D'])    # hidden state width (large = more capacity)
CNFG_BEINGSNUM_INT = _cfg['N']                            # 6    parallel experts sharing the ring
CNFG_ATNRADIUS_INT = _cfg['R']                            # 2    attention window half-width (window = 2R+1)

# ─ attention kernel ─
CNFG_KERNELMODE_STR = _cfg.get('kernel_mode', 'vshape')  # 'uniform'|'vshape'|'gaussian'|'dotprod'|'topk'

# ─ memory optimization ─
CNFG_CKPTCHUNK_INT = _cfg.get('checkpoint_chunks', 0)  # 0 = disabled, >0 = chunk size for gradient checkpointing

# ─ expert write weighting ─
CNFG_XPRTWGHT_BOOL = _cfg.get('expert_weighting', False)  # gradient-based expert write confidence

# ─ embedding encoding ─
CNFG_EMBEDENC_STR = _cfg.get('embed_encoding', 'learned')  # 'learned' | 'hadamard' | 'sincos'

# ─ output encoding ─
CNFG_OUTPUTENC_STR = _cfg.get('output_encoding', 'learned')  # 'learned' | 'hadamard' | 'sincos'

# ─ pointer movement mode ─
CNFG_PTRMODE_STR = _cfg.get('pointer_mode', 'sequential')  # 'sequential' | 'learned' | 'pilot'

# ─ pilot pulse pointer ─
CNFG_PILOTJUMP_INT = int(_cfg.get('pilot_max_jump', 0))     # max extra jump distance (0 = M//4 default)
CNFG_PILOTIDDIM_INT = int(_cfg.get('pilot_id_dim', 32))     # identity vector size per slot

# ─ bulletin board temporal cache ─
CNFG_BBENABLED_BOOL = _cfg.get('bb_enabled', False)  # short-term temporal cache
CNFG_BBGATEBIAS_FLT = float(_cfg.get('bb_gate_bias', 0.0))
CNFG_BBSCALE_FLT = float(_cfg.get('bb_scale', 0.1))
CNFG_BBTAU_FLT = float(_cfg.get('bb_tau', 4.0))
CNFG_BBGATEMODE_STR = _cfg.get('bb_gate_mode', 'learned')  # 'learned' | 'fixed'


# ─ topK ring read ─
CNFG_TOPK_INT = int(_cfg.get('topk_K', 8))  # number of ring slots for content-based read

# ─ hourglass I/O split ─
CNFG_IOSPLIT_STR = _cfg.get('io_split_mode', 'off')       # 'off' | 'strict'
CNFG_IOWRITERCNT_INT = int(_cfg.get('io_writer_count', 1))  # how many experts are writers
CNFG_IOREADOUT_BOOL = bool(_cfg.get('io_output_from_readers_only', True))  # output from readers only
CNFG_REPLACEIMPL_STR = _cfg.get('replace_impl', 'dense')  # 'dense' | 'proxy_overlay' (nightly-only fast path)

del _cfg

# ── Fixed Encoding Tables ────────────────────────────────────────
def _make_hadamard_table(vocab: int, dim: int) -> torch.Tensor:
    """Rows from Hadamard matrix, normalized to unit norm.
    When dim >= vocab: perfect orthogonality (cosine sim = 0 between ALL pairs).
    When dim < vocab: deterministic near-orthogonal fallback (random unit vectors)."""
    if dim >= vocab:
        from scipy.linalg import hadamard
        H = hadamard(dim)                                     # (dim, dim) entries +/-1
        table = torch.tensor(H[:vocab], dtype=torch.float32)  # (vocab, dim)
    else:
        # dim < vocab: not enough Hadamard rows; deterministic random fallback
        # (only hits test configs — production uses dim=8192 >> vocab=256)
        gen = torch.Generator().manual_seed(42)
        table = torch.randn(vocab, dim, generator=gen)
    table = table / table.norm(dim=1, keepdim=True)           # unit norm rows
    return table

def _make_sincos_table(vocab: int, dim: int, base: float = 10000.0) -> torch.Tensor:
    """Standard sinusoidal encoding. Adjacent tokens have ~0.97 cosine similarity."""
    table = torch.zeros(vocab, dim)
    pos = torch.arange(vocab, dtype=torch.float32).unsqueeze(1)   # (V, 1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(base) / dim))
    table[:, 0::2] = torch.sin(pos * div)
    table[:, 1::2] = torch.cos(pos * div)
    table = table / table.norm(dim=1, keepdim=True)       # unit norm (consistent with Hadamard)
    return table

# ── Helpers ─────────────────────────────────────────────────────
def phi_destinations(n_int, m_int):
    """Builds the jump destination table for all experts on the ring.
    Each expert at slot j looks up where to jump: dests[i, j].
    Uses the golden ratio to space jumps maximally apart — φ guarantees
    the most uniform coverage of the ring with zero clustering.
    Called once in __init__ and stored as a buffer — never recomputed."""

    # golden-ratio step: floor(M × 1/φ) = 39 when M=64 — maximally irrational stride
    step_int = int(m_int * PHI_INV)

    # all M slot indices as a row vector for broadcasting; shape (M,)
    slots_tns = torch.arange(m_int, dtype=torch.long)

    # N expert IDs as a column vector — unsqueeze(1) makes (N,1) so
    # slots (M,) + beings (N,1) broadcasts to the full (N, M) table
    beings_tns = torch.arange(n_int, dtype=torch.long).unsqueeze(1)

    # destination[i, j] = (j + step + i) % M — the slot expert i jumps to from slot j; shape (N, M)
    return (slots_tns + step_int + beings_tns) % m_int

# ── func_ operations ───────────────────────────────────────────
def func_ringstart_tns(  # func_ = standalone op, _tns = returns a tensor
    batch_int,    # parallel sequences in batch
    slots_int,    # circular buffer capacity
    dims_int,     # embedding width per slot
    device_str    # torch device (cpu/cuda)
):
    """Creates the ring buffer — a (B, M, D) zero tensor that serves as
    shared memory for all experts. Every expert reads from and writes
    to this buffer each timestep. Starts empty so the network learns
    what to store, not what to forget."""

    # shape (B, M, D) — batch-first so PyTorch gather/scatter work along dim=1 (slots)
    return torch.zeros(
        batch_int,   # B — one ring per sequence in the batch
        slots_int,   # M — each slot is an addressable memory cell
        dims_int,    # embed_dim — each cell holds an embedding vector of this width
        device=device_str
    )

# · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
def func_softread_tns(  # func_ = standalone op, _tns = returns a tensor
    ring_tns,      # the shared ring buffer; shape (B, M, D)
    indices_tns,   # window slot indices from func_attnwndow_tpl; shape (B, 2R+1)
    weights_tns,   # window weights from func_attnwndow_tpl; shape (B, 2R+1)
    dims_int       # embedding dimension (D) — needed to expand indices for gather
):
    """Differentiable read from the ring buffer. Gathers 2R+1 neighbor
    slot vectors at the window positions, then collapses them into a
    single embed_dim-wide read vector via uniform weighted sum.
    The caller applies the per-expert projection and scaling separately."""

    # expand window indices from (B, 2R+1) to (B, 2R+1, embed_dim) so gather can pull full vectors
    expanded_idx_tns = indices_tns.unsqueeze(-1).expand(-1, -1, dims_int)

    # pull 2R+1 neighbor slot vectors from the ring; shape (B, 2R+1, D)
    neighbors_tns = ring_tns.gather(1, expanded_idx_tns)

    # uniform weighted sum collapses the window into one read vector; shape (B, D)
    read_vec_tns = (weights_tns.unsqueeze(-1) * neighbors_tns).sum(1)

    return read_vec_tns, expanded_idx_tns  # also return expanded indices — soft_write reuses them

# · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
def func_linear_pointer_window_tns(
    ptr_tns,
    offsets_long,
    base_weights_tns,
    slots_int,
):
    """Blend two adjacent local windows around a fractional pointer.

    For ptr = i + alpha we merge the windows centered at i and i+1 into a
    unique support of width (2R+2). Integer ptr values fall back exactly to
    the original discrete support.
    """
    ptr_floor_tns = torch.floor(ptr_tns)
    alpha_tns = (ptr_tns - ptr_floor_tns).unsqueeze(1)  # (B, 1)
    center0_tns = ptr_floor_tns.long().clamp(0, slots_int - 1)
    center1_tns = (center0_tns + 1) % slots_int

    idx0_tns = (center0_tns.unsqueeze(1) + offsets_long) % slots_int
    idx1_tns = (center1_tns.unsqueeze(1) + offsets_long) % slots_int
    w0_tns = (1.0 - alpha_tns) * base_weights_tns
    w1_tns = alpha_tns * base_weights_tns

    merged_idx_tns = torch.cat([idx0_tns[:, :1], idx1_tns], dim=1)
    merged_weights_tns = torch.cat(
        [w0_tns[:, :1], w0_tns[:, 1:] + w1_tns[:, :-1], w1_tns[:, -1:]],
        dim=1,
    )
    return center0_tns, alpha_tns.squeeze(1), merged_idx_tns, merged_weights_tns

# · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
def func_shortest_arc_delta_tns(
    current_tns,
    target_tns,
    slots_int,
):
    """Shortest signed circular delta from current to target on a ring."""
    slots_flt = float(slots_int)
    half_flt = slots_flt / 2.0
    return torch.remainder((target_tns - current_tns) + half_flt, slots_flt) - half_flt


# · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
def func_softwrit_tns(  # func_ = standalone op, _tns = returns a tensor (the updated ring)
    ring_tns,            # the shared ring buffer to write into; shape (B, M, D)
    hidden_tns,          # hidden state of the current expert; shape (B, D)
    expanded_idx_tns,    # expanded window indices from func_softread_tns; shape (B, 2R+1, D)
    weights_tns          # window weights from func_attnwndow_tpl; shape (B, 2R+1)
):
    """Differentiable write back into the ring buffer. Broadcasts the
    expert's hidden state to all window positions, then scatter_adds it
    weighted by the attention window. Additive — accumulates information,
    never erases existing slot content. No write projection in v4."""

    # broadcast hidden state to match window size — same value written to all window slots; shape (B, 2R+1, D)
    write_val_tns = hidden_tns.unsqueeze(1).expand(-1, weights_tns.size(1), -1)

    # clone ring so autograd can track gradients through learnable weights across
    # multiple read→write cycles in the expert loop (avoids in-place version conflict).
    # scatter_add writes weighted hidden into ring at window positions — additive, never erases; shape (B, M, D)
    return ring_tns.clone().scatter_add(1, expanded_idx_tns, weights_tns.unsqueeze(-1) * write_val_tns)

# · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
def func_gated_write_tns(
    ring_tns,            # the shared ring buffer; shape (B, M, D)
    write_vec_tns,       # compressed hidden state to write; shape (B, D)
    expanded_idx_tns,    # expanded window indices; shape (B, 2R+1, D)
    weights_tns,         # window weights; shape (B, 2R+1)
    erase,               # (B,) tensor [0,1] — how much to decay existing content
    write_gate           # (B,) tensor [0,1] — how strong the new write is
):
    """Gated write: erase old content then add new. Fixes ring blob.

    slot_new = slot_old * (1 - erase * w) + write_gate * w * write_vec
    where w = attention window weight at each position.

    erase and write_gate are (B,) tensors — context-dependent per batch element.
    When erase=0, write_gate=1: equivalent to scatter_add (legacy behavior).
    When erase>0: old content decays, preventing accumulation/blob."""

    w = weights_tns.unsqueeze(-1)                       # (B, 2R+1, 1)
    erase_b = erase.unsqueeze(-1).unsqueeze(-1)         # (B, 1, 1) — broadcasts over window and dim
    wgate_b = write_gate.unsqueeze(-1).unsqueeze(-1)    # (B, 1, 1)

    # Gather current slot values at write positions
    current = ring_tns.gather(1, expanded_idx_tns)      # (B, 2R+1, D)

    # Erase: decay existing content
    erased = current * (1 - erase_b * w)                # (B, 2R+1, D)

    # Write: add new content (broadcast write_vec to all window positions)
    write_val = write_vec_tns.unsqueeze(1).expand(-1, weights_tns.size(1), -1)  # (B, 2R+1, D)
    updated = erased + wgate_b * w * write_val           # (B, 2R+1, D)

    # Scatter back (replace slots with updated values)
    ring_new = ring_tns.clone()
    ring_new.scatter_(1, expanded_idx_tns, updated)
    return ring_new

# · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
def func_hdd_write_tns(
    ring_tns,            # the shared ring buffer; shape (B, M, D)
    write_vec_tns,       # content to write; shape (B, D)
    expanded_idx_tns,    # expanded window indices; shape (B, 2R+1, D)
    weights_tns,         # window weights; shape (B, 2R+1)
    write_strength=None, # per-sample write intensity; shape (B, 1) or None
):
    """HDD-style write: REPLACE slot content instead of accumulating.

    Like a hard disk: old sector content is overwritten, not blended.
    Weighted by attention window — center slot gets full replacement,
    edge slots get partial replacement (lerp with old content).

    If write_strength is provided, it modulates the overall replacement:
      slot_new = (w * s) * write_vec + (1 - w * s) * slot_old
    Otherwise:
      slot_new = w * write_vec + (1 - w) * slot_old
    where w = attention weight, s = per-sample write strength in [0,1]."""

    w = weights_tns.unsqueeze(-1)  # (B, 2R+1, 1)
    if write_strength is not None:
        w = w * write_strength.unsqueeze(1)  # (B, 1, 1) broadcast → (B, 2R+1, 1)

    # Current slot values
    current = ring_tns.gather(1, expanded_idx_tns)  # (B, 2R+1, D)

    # Broadcast write_vec to all window positions
    write_val = write_vec_tns.unsqueeze(1).expand(-1, weights_tns.size(1), -1)  # (B, 2R+1, D)

    # Lerp: full replacement at center (w=1), partial at edges
    updated = w * write_val + (1 - w) * current  # (B, 2R+1, D)

    # Scatter back (replace)
    ring_new = ring_tns.clone()
    ring_new.scatter_(1, expanded_idx_tns, updated)
    return ring_new


# ── Batched expert helpers ─────────────────────────────────────
def func_batched_linear(x_tns, weight_list, bias_list):
    """Batched linear: apply N different Linear layers to N inputs simultaneously.

    Args:
        x_tns:       (N, B, in_dim)   — inputs per expert
        weight_list: list of N nn.Linear modules
        bias_list:   ignored (biases come from weight_list)

    Returns:
        (N, B, out_dim)
    """
    N = x_tns.shape[0]
    # Stack weights: (N, out_dim, in_dim) and biases: (N, out_dim)
    W = torch.stack([weight_list[i].weight for i in range(N)])   # (N, out, in)
    b = torch.stack([weight_list[i].bias for i in range(N)])     # (N, out)
    # bmm: (N, B, in) @ (N, in, out) → (N, B, out)
    return torch.bmm(x_tns, W.transpose(1, 2)) + b.unsqueeze(1)


def func_proxy_overlay_read_tns(
    ring_tns,
    indices_tns,
    weights_tns,
    dims_int,
    overlay_start_int,
    overlay_vals_tns,
    overlay_len_int,
    slots_int,
):
    """Read from dense ring, then overlay any recently-written contiguous slots."""
    expanded_idx_tns = indices_tns.unsqueeze(-1).expand(-1, -1, dims_int)
    neighbors_tns = ring_tns.gather(1, expanded_idx_tns)

    if overlay_vals_tns is not None and overlay_len_int > 0:
        rel_idx_tns = (indices_tns - overlay_start_int) % slots_int
        mask_tns = rel_idx_tns < overlay_len_int
        if bool(mask_tns.any().item()):
            overlay_expanded = rel_idx_tns.clamp_max(overlay_len_int - 1).unsqueeze(-1).expand(-1, -1, dims_int)
            overlay_neighbors = overlay_vals_tns.gather(1, overlay_expanded)
            neighbors_tns = torch.where(mask_tns.unsqueeze(-1), overlay_neighbors, neighbors_tns)

    read_vec_tns = (weights_tns.unsqueeze(-1) * neighbors_tns).sum(1)
    return read_vec_tns, expanded_idx_tns, neighbors_tns


def func_proxy_overlay_flush_tns(
    ring_tns,
    overlay_start_int,
    overlay_vals_tns,
    overlay_len_int,
    slots_int,
):
    """Materialize the contiguous overlay back into the dense ring tensor."""
    if overlay_vals_tns is None or overlay_len_int <= 0:
        return ring_tns

    slot_idx_tns = (torch.arange(overlay_len_int, device=ring_tns.device) + overlay_start_int) % slots_int
    expanded_idx_tns = slot_idx_tns.view(1, -1, 1).expand(ring_tns.size(0), -1, ring_tns.size(2))
    ring_new = ring_tns.clone()
    ring_new.scatter_(1, expanded_idx_tns, overlay_vals_tns)
    return ring_new


def func_proxy_overlay_write_tns(
    ring_tns,
    updated_window_tns,
    window_start_int,
    overlay_start_int,
    overlay_vals_tns,
    overlay_len_int,
    overlay_steps_int,
    slots_int,
    flush_interval_int,
    overlay_cap_int,
):
    """Maintain a small contiguous overlay instead of cloning the full ring each write."""
    win_int = int(updated_window_tns.shape[1])

    if overlay_vals_tns is None or overlay_len_int <= 0:
        overlay_start_int = window_start_int
        overlay_vals_tns = updated_window_tns
        overlay_len_int = win_int
        overlay_steps_int = 1
    else:
        rel0_int = (window_start_int - overlay_start_int) % slots_int

        if rel0_int + win_int <= overlay_len_int:
            prefix_tns = overlay_vals_tns[:, :rel0_int]
            suffix_tns = overlay_vals_tns[:, rel0_int + win_int:overlay_len_int]
            overlay_vals_tns = torch.cat([prefix_tns, updated_window_tns, suffix_tns], dim=1)
        elif rel0_int < overlay_len_int and rel0_int + win_int <= overlay_cap_int:
            prefix_tns = overlay_vals_tns[:, :rel0_int]
            overlay_vals_tns = torch.cat([prefix_tns, updated_window_tns], dim=1)
            overlay_len_int = rel0_int + win_int
        else:
            ring_tns = func_proxy_overlay_flush_tns(
                ring_tns,
                overlay_start_int,
                overlay_vals_tns,
                overlay_len_int,
                slots_int,
            )
            overlay_start_int = window_start_int
            overlay_vals_tns = updated_window_tns
            overlay_len_int = win_int
            overlay_steps_int = 0

        overlay_steps_int += 1

    if overlay_len_int >= overlay_cap_int or overlay_steps_int >= flush_interval_int:
        ring_tns = func_proxy_overlay_flush_tns(
            ring_tns,
            overlay_start_int,
            overlay_vals_tns,
            overlay_len_int,
            slots_int,
        )
        overlay_start_int = 0
        overlay_vals_tns = None
        overlay_len_int = 0
        overlay_steps_int = 0

    return ring_tns, overlay_start_int, overlay_vals_tns, overlay_len_int, overlay_steps_int

# · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
def func_movepntr_tns(  # func_ = standalone op, _tns = returns a tensor (new pointer positions)
    ptr_tns,         # current pointer position for this expert; shape (B,)
    dests_tns,       # φ-destination row for this expert; shape (M,) — from phi_destinations table
    prob_flt,        # jump probability for this expert — e.g. 0.9, 0.1, or 0.5
    slots_int,       # total ring slots (M) — needed for mod wrap on walk
    seam_mode='mod'  # 'mod' | 'shortest_arc' — nightly-only wrap semantics
):
    """Moves one expert's pointer on the ring. Two options soft-blended:
    φ-jump (long leap to golden-ratio destination) or +1 walk (short step).
    The blend is p·jump + (1-p)·walk — differentiable at this stage.
    The pointer stays fractional (float); note that attnwndow uses long()
    for indexing, which breaks the gradient path through the pointer."""

    # truncate float pointer to int slot, clamp to valid range [0, M-1]; shape (B,)
    current_slot_tns = ptr_tns.long().clamp(0, slots_int - 1)

    # look up φ-destination for this expert at current slot — the long jump target; shape (B,)
    jump_target_tns = dests_tns[current_slot_tns].float()

    # short step: just move +1, wrap around via mod M; shape (B,)
    walk_target_tns = (ptr_tns + 1) % slots_int

    # soft blend on the circle, not on the flattened [0, M) line.
    if seam_mode == 'shortest_arc':
        jump_delta_tns = func_shortest_arc_delta_tns(ptr_tns, jump_target_tns, slots_int)
        walk_delta_tns = func_shortest_arc_delta_tns(ptr_tns, walk_target_tns, slots_int)
        blended_delta_tns = prob_flt * jump_delta_tns + (1 - prob_flt) * walk_delta_tns
        return torch.remainder(ptr_tns + blended_delta_tns, float(slots_int))

    # legacy flat blend: p·jump + (1-p)·walk; shape (B,)
    return prob_flt * jump_target_tns + (1 - prob_flt) * walk_target_tns

# ── Model ───────────────────────────────────────────────────────
class INSTNCT(nn.Module):
    """Ring-buffer pointer network — v4 reference implementation.
    N experts share one ring of M slots (each embed_dim-wide). Each expert reads,
    writes, and moves its pointer every timestep. tanh activation + sin/cos
    phase embeddings on hidden state update. Phase encodes pointer position
    on the ring (2π period for smooth wrap). Complexity lives in the ring
    topology and nonlinear hidden dynamics."""

    def __init__(self,
                 M=CNFG_RINGSLOTS_INT,          # ring slots (from YAML, default 256)
                 embed_dim=None,                 # BACKWARD COMPAT alias — sets both dims if given alone
                 hidden_dim=CNFG_HIDDNDIMS_INT,  # hidden state width (from YAML)
                 slot_dim=CNFG_SLOTDIMS_INT,     # ring buffer slot width (from YAML)
                 N=CNFG_BEINGSNUM_INT,           # expert count (from YAML, default 6)
                 R=CNFG_ATNRADIUS_INT,           # attention radius (from YAML, default 2)
                 B=8,                             # bits per input position (default: 1 byte = 8 bits)
                 embed_mode=False,                # True = byte embedding (256->hidden_dim), False = binary bits (B->hidden_dim)
                 kernel_mode=CNFG_KERNELMODE_STR,  # attention kernel: 'uniform', 'vshape', 'gaussian', 'dotprod', 'topk'
                 read_kernel_mode=None,            # nightly-only proof split: defaults to kernel_mode
                 checkpoint_chunks=CNFG_CKPTCHUNK_INT,   # gradient checkpointing chunk size (0 = off)
                 expert_weighting=CNFG_XPRTWGHT_BOOL,    # gradient-based expert write confidence
                 embed_encoding=CNFG_EMBEDENC_STR,       # 'learned' | 'hadamard' | 'sincos' | 'bitlift'
                 output_encoding=CNFG_OUTPUTENC_STR,      # 'learned' | 'hadamard' | 'sincos'
                 pointer_mode=CNFG_PTRMODE_STR,            # 'sequential' | 'learned' | 'pilot'
                 bb_enabled=CNFG_BBENABLED_BOOL,              # bulletin board temporal cache
                 bb_gate_bias=CNFG_BBGATEBIAS_FLT,              # gate init bias (sigmoid(x))
                 bb_scale=CNFG_BBSCALE_FLT,                     # output scale multiplier
                 bb_tau=CNFG_BBTAU_FLT,                         # attention temperature
                 bb_gate_mode=CNFG_BBGATEMODE_STR,              # 'learned' | 'fixed'
                 topk_K=CNFG_TOPK_INT,                            # topK ring read: K slots
                 read_topk_K=None,                                # nightly-only proof split: defaults to topk_K
                 write_address_mode='pointer',                    # 'pointer' | 'content_topk'
                 write_topk_K=2,                                  # nightly-only proof split: K slots for content_topk write
                 io_split_mode=CNFG_IOSPLIT_STR,                     # 'off' | 'strict' — hourglass I/O split
                 io_writer_count=CNFG_IOWRITERCNT_INT,               # experts 0..count-1 are writers
                 io_output_from_readers_only=CNFG_IOREADOUT_BOOL,    # output from readers only (strict mode)
                 gated_write=False,                        # True = erase+gate write (anti-blob), False = scatter_add (legacy)
                 write_mode='accumulate',                  # 'accumulate' (scatter_add) | 'replace' (HDD-style overwrite)
                 replace_impl=CNFG_REPLACEIMPL_STR,        # 'dense' | 'proxy_overlay' (nightly-only proxy fast path)
                 pointer_interp_mode='off',                # 'off' | 'linear' — nightly-only fractional pointer center
                 pointer_seam_mode='mod',                  # 'mod' | 'shortest_arc' — nightly-only wrap-seam fix
                 mtaps_enabled=False,                      # nightly-only multi-timescale taps
                 mtaps_lags=(1, 2, 4, 8, 16, 32),          # lag taps, kept separate then mixed
                 mtaps_mixer_mode='current',               # 'current' | 'tap_scalar_gate' | 'residual_gated' | 'hybrid_heads_scalar_gate' | 'hybrid_heads_spaced_scalar_gate' | 'hybrid_heads_fixed_scalar_gate'
                 mtaps_aux_fixed_offsets=(),               # nightly-only fixed auxiliary read heads, pointer-relative
                 s_constraint='softplus',                  # 'softplus' (S>0) | 'raw' (unconstrained)
                 c19_mode='standard',                      # 'standard' | 'dualphi' — activation variant
                 jump_gate=False):                         # learned φ-jump gate for sequential pointer mode
        super().__init__()
        assert kernel_mode in ('uniform', 'vshape', 'gaussian', 'dotprod', 'topk'), \
            f"kernel_mode must be 'uniform', 'vshape', 'gaussian', 'dotprod', or 'topk', got '{kernel_mode}'"
        if read_kernel_mode is None:
            read_kernel_mode = kernel_mode
        if read_topk_K is None:
            read_topk_K = topk_K
        assert read_kernel_mode in ('uniform', 'vshape', 'gaussian', 'dotprod', 'topk'), \
            f"read_kernel_mode must be 'uniform', 'vshape', 'gaussian', 'dotprod', or 'topk', got '{read_kernel_mode}'"
        assert write_mode in ('accumulate', 'replace'), \
            f"write_mode must be 'accumulate' or 'replace', got '{write_mode}'"
        assert pointer_mode in ('sequential', 'learned', 'pilot'), \
            f"pointer_mode must be 'sequential', 'learned', or 'pilot', got '{pointer_mode}'"
        assert replace_impl in ('dense', 'proxy_overlay'), \
            f"replace_impl must be 'dense' or 'proxy_overlay', got '{replace_impl}'"
        assert pointer_interp_mode in ('off', 'linear'), \
            f"pointer_interp_mode must be 'off' or 'linear', got '{pointer_interp_mode}'"
        assert pointer_seam_mode in ('mod', 'shortest_arc'), \
            f"pointer_seam_mode must be 'mod' or 'shortest_arc', got '{pointer_seam_mode}'"
        assert write_address_mode in ('pointer', 'content_topk'), \
            f"write_address_mode must be 'pointer' or 'content_topk', got '{write_address_mode}'"
        assert c19_mode in ('standard', 'dualphi'), \
            f"c19_mode must be 'standard' or 'dualphi', got '{c19_mode}'"
        assert mtaps_mixer_mode in (
            'current',
            'tap_scalar_gate',
            'residual_gated',
            'hybrid_heads_scalar_gate',
            'hybrid_heads_spaced_scalar_gate',
            'hybrid_heads_fixed_scalar_gate',
        ), \
            f"mtaps_mixer_mode must be 'current', 'tap_scalar_gate', 'residual_gated', 'hybrid_heads_scalar_gate', 'hybrid_heads_spaced_scalar_gate', or 'hybrid_heads_fixed_scalar_gate', got '{mtaps_mixer_mode}'"
        if mtaps_enabled:
            if read_kernel_mode == 'topk':
                raise ValueError("mtaps_enabled requires a local read path, not pooled topk")
            if not mtaps_lags:
                raise ValueError("mtaps_enabled requires at least one lag")
            mtaps_lags = tuple(sorted({int(x) for x in mtaps_lags}))
            if any(x <= 0 for x in mtaps_lags):
                raise ValueError(f"mtaps_lags must be positive integers, got {mtaps_lags!r}")
        mtaps_aux_fixed_offsets = tuple(int(x) for x in mtaps_aux_fixed_offsets)
        if mtaps_mixer_mode == 'hybrid_heads_fixed_scalar_gate':
            if len(mtaps_aux_fixed_offsets) < 1:
                raise ValueError(
                    "hybrid_heads_fixed_scalar_gate requires at least 1 mtaps_aux_fixed_offsets entry"
                )
            if any(x <= 0 for x in mtaps_aux_fixed_offsets):
                raise ValueError(
                    f"mtaps_aux_fixed_offsets must be positive integers, got {mtaps_aux_fixed_offsets!r}"
                )
        self.write_mode = write_mode
        self.replace_impl = replace_impl
        self.pointer_interp_mode = pointer_interp_mode
        self.pointer_seam_mode = pointer_seam_mode
        self.c19_mode = c19_mode
        # Store the activation variant. Note: nightly runner monkey-patches
        # instnct._c19_activation, so we check the module-level name at call time
        # rather than capturing the function object at __init__ time.
        self._c19_dualphi = (c19_mode == 'dualphi')
        self.mtaps_enabled = bool(mtaps_enabled)
        self.mtaps_mixer_mode = mtaps_mixer_mode
        self._mtaps_aux_fixed_offsets = mtaps_aux_fixed_offsets

        # backward compat: embed_dim=X sets hidden_dim=slot_dim=X
        if embed_dim is not None:
            hidden_dim = embed_dim
            slot_dim = embed_dim

        self.M, self.hidden_dim, self.slot_dim, self.N, self.R = M, hidden_dim, slot_dim, N, R
        self.embed_dim = hidden_dim               # alias for backward compat (tests, prints)
        self.B = B
        self.embed_mode = embed_mode
        self.kernel_mode = kernel_mode
        self.read_kernel_mode = read_kernel_mode
        self.write_address_mode = write_address_mode
        self._mtaps_lags = list(mtaps_lags) if self.mtaps_enabled else []
        self.checkpoint_chunks = checkpoint_chunks  # not nn.Parameter — excluded from state_dict
        self.expert_weighting = expert_weighting
        self._proxy_overlay_flush_interval = 16
        self._proxy_overlay_cap = 18
        self._proxy_overlay_enabled = (
            replace_impl == 'proxy_overlay'
            and N == 1
            and R == 1
            and write_mode == 'replace'
            and read_kernel_mode == 'vshape'
            and write_address_mode == 'pointer'
            and pointer_mode == 'sequential'
            and pointer_interp_mode == 'off'
            and not bb_enabled
            and io_split_mode == 'off'
            and checkpoint_chunks == 0
        )

        # ── Diagnostics accumulator ──
        # Populated during forward(), read by train.py after each step.
        # Keys: alpha_mean/min/max (per expert), ring_norm, hidden_norm, input_norm, ptr_pos
        # Gate: _diag_enabled controls whether .item() calls run (CUDA sync points).
        # train.py sets True on log steps only → ~98% of steps skip all diagnostics.
        self._diag: dict = {}
        self._diag_enabled: bool = False
        if expert_weighting:
            self._write_grad_ema = torch.zeros(N)     # EMA of gradient magnitudes (CPU)
            self.register_buffer('_expert_conf', torch.ones(N), persistent=False)  # moves with device, no .item() needed
        else:
            self._write_grad_ema = None
            self._expert_conf = None

        # ── Hourglass I/O split ──
        self.io_split_mode = io_split_mode
        self.io_writer_count = io_writer_count
        self.io_output_from_readers_only = io_output_from_readers_only
        if io_split_mode == 'strict':
            assert N >= 2, f"strict io_split requires N>=2, got N={N}"
            wc = max(0, min(N, io_writer_count))
            assert wc < N, f"strict io_split needs at least 1 reader, got {wc} writers out of {N}"
            _wmask = torch.zeros(N, dtype=torch.bool)
            _wmask[:wc] = True
            self.register_buffer('_writer_mask', _wmask)
            self.register_buffer('_reader_mask', ~_wmask)
        else:
            self.register_buffer('_writer_mask', torch.zeros(N, dtype=torch.bool))
            self.register_buffer('_reader_mask', torch.ones(N, dtype=torch.bool))

        self.embed_encoding = embed_encoding
        self.output_encoding = output_encoding
        self._bitlift = False
        if embed_mode:
            # ── Input encoding ──
            if embed_encoding == 'learned':
                self.inp = nn.Embedding(256, hidden_dim)   # byte token → hidden_dim
            elif embed_encoding == 'bitlift':
                self.inp = nn.Linear(8, hidden_dim)        # 8 bits → hidden_dim (28× fewer params)
                self._bitlift = True
            else:
                if embed_encoding == 'hadamard':
                    table = _make_hadamard_table(256, hidden_dim)
                elif embed_encoding == 'sincos':
                    table = _make_sincos_table(256, hidden_dim)
                else:
                    raise ValueError(f"Unknown embed_encoding: {embed_encoding!r}")
                self.register_buffer('_fixed_table', table)  # (256, hidden_dim) — no grad, on device
                self.inp = None  # sentinel — _process_chunk uses _fixed_table[x] instead
            # ── Output encoding ──
            self._bitlift_out = False
            if output_encoding == 'learned':
                self.out = nn.Linear(hidden_dim, 256)      # hidden_dim → 256 logits
            elif output_encoding == 'lowrank_c19':
                # Low-rank factored head with c19 activation: H→64→c19→256
                # Sweep-validated: 47.2% acc vs 43.5% baseline (r=64, 1K steps)
                # 148K params vs 524K full learned — 72% fewer, better accuracy
                self.out = nn.Sequential(
                    nn.Linear(hidden_dim, 64),
                    _C19LearnableModule(width=64),
                    nn.Linear(64, 256),
                )
            elif output_encoding == 'bitlift':
                self.out = nn.Linear(hidden_dim, 8)        # hidden_dim → 8 bit scores (28× fewer params)
                self._bitlift_out = True
                # Fixed table: all 256 byte values as 8-bit float patterns
                _bit_shifts_out = torch.arange(7, -1, -1)
                _patterns = ((torch.arange(256).unsqueeze(1) >> _bit_shifts_out) & 1).float()  # (256, 8)
                self.register_buffer('_bit_patterns', _patterns)
            else:
                if output_encoding == 'hadamard':
                    out_table = _make_hadamard_table(256, hidden_dim)
                elif output_encoding == 'sincos':
                    out_table = _make_sincos_table(256, hidden_dim)
                else:
                    raise ValueError(f"Unknown output_encoding: {output_encoding!r}")
                self.register_buffer('_fixed_output_table', out_table)
                self.out = None  # sentinel — _process_chunk uses matmul instead
        else:
            self._bitlift_out = False
            self.inp = nn.Linear(B, hidden_dim)        # B bits → hidden_dim
            self.out = nn.Linear(hidden_dim, B)        # hidden_dim → B bits

        # ── Per-neuron learnable rho + C for c19 activations (both embed_mode) ──
        _raw0_rho = _rho_init_raw(4.0)
        _raw0_C = _C_init_raw()
        self.c19_rho_input = nn.Parameter(torch.full((hidden_dim,), _raw0_rho))
        self.c19_rho_hidden = nn.Parameter(torch.full((hidden_dim,), _raw0_rho))
        self.c19_C_input = nn.Parameter(torch.full((hidden_dim,), _raw0_C))
        self.c19_C_hidden = nn.Parameter(torch.full((hidden_dim,), _raw0_C))

        # read_proj: slot_dim → hidden_dim (decompress ring read into hidden space)
        self.read_proj = nn.ModuleList([nn.Linear(slot_dim, hidden_dim) for _ in range(N)])
        if self.mtaps_enabled:
            self.register_buffer('_mtaps_lags_tns', torch.tensor(self._mtaps_lags, dtype=torch.long))
            n_taps = len(self._mtaps_lags)
            if mtaps_mixer_mode in ('hybrid_heads_scalar_gate', 'hybrid_heads_spaced_scalar_gate'):
                aux_heads = 2
            elif mtaps_mixer_mode == 'hybrid_heads_fixed_scalar_gate':
                aux_heads = len(mtaps_aux_fixed_offsets)
            else:
                aux_heads = 0
            if mtaps_mixer_mode in ('current', 'tap_scalar_gate', 'hybrid_heads_scalar_gate', 'hybrid_heads_spaced_scalar_gate', 'hybrid_heads_fixed_scalar_gate'):
                self.read_tap_proj = nn.ModuleList([
                    nn.Linear(slot_dim * (1 + n_taps + aux_heads), hidden_dim) for _ in range(N)
                ])
            else:
                self.read_tap_proj = None
            if mtaps_mixer_mode == 'tap_scalar_gate':
                self.read_tap_gate = nn.ModuleList([
                    nn.Linear(hidden_dim, 1 + n_taps) for _ in range(N)
                ])
                for gate in self.read_tap_gate:
                    nn.init.zeros_(gate.weight)
                    nn.init.zeros_(gate.bias)
                self.read_tap_delta_proj = None
                self.read_tap_resid_gate = None
                self.read_tap_aux_offset = None
                self.register_buffer('_mtaps_aux_ranges_tns', torch.tensor([], dtype=torch.float32))
            elif mtaps_mixer_mode in ('hybrid_heads_scalar_gate', 'hybrid_heads_spaced_scalar_gate'):
                self.read_tap_gate = nn.ModuleList([
                    nn.Linear(hidden_dim, 1 + n_taps + 2) for _ in range(N)
                ])
                self.read_tap_aux_offset = nn.ModuleList([
                    nn.Linear(hidden_dim, 2) for _ in range(N)
                ])
                for gate in self.read_tap_gate:
                    nn.init.zeros_(gate.weight)
                    nn.init.zeros_(gate.bias)
                for head in self.read_tap_aux_offset:
                    nn.init.zeros_(head.weight)
                    head.bias.data.copy_(torch.tensor([-0.5, -0.75]))
                self.read_tap_delta_proj = None
                self.read_tap_resid_gate = None
                self.register_buffer('_mtaps_aux_ranges_tns', torch.tensor([16.0, 64.0], dtype=torch.float32))
                self.register_buffer('_mtaps_aux_fixed_offsets_tns', torch.tensor([], dtype=torch.float32))
            elif mtaps_mixer_mode == 'hybrid_heads_fixed_scalar_gate':
                self.read_tap_gate = nn.ModuleList([
                    nn.Linear(hidden_dim, 1 + n_taps + len(mtaps_aux_fixed_offsets)) for _ in range(N)
                ])
                for gate in self.read_tap_gate:
                    nn.init.zeros_(gate.weight)
                    nn.init.zeros_(gate.bias)
                self.read_tap_aux_offset = None
                self.read_tap_delta_proj = None
                self.read_tap_resid_gate = None
                self.register_buffer('_mtaps_aux_ranges_tns', torch.tensor([], dtype=torch.float32))
                self.register_buffer('_mtaps_aux_fixed_offsets_tns', torch.tensor(mtaps_aux_fixed_offsets, dtype=torch.float32))
            elif mtaps_mixer_mode == 'residual_gated':
                self.read_tap_gate = nn.ModuleList([
                    nn.Linear(hidden_dim, n_taps) for _ in range(N)
                ])
                self.read_tap_delta_proj = nn.ModuleList([
                    nn.Linear(slot_dim * n_taps, hidden_dim) for _ in range(N)
                ])
                self.read_tap_resid_gate = nn.ModuleList([
                    nn.Linear(hidden_dim, 1) for _ in range(N)
                ])
                for gate in self.read_tap_gate:
                    nn.init.zeros_(gate.weight)
                    nn.init.zeros_(gate.bias)
                for resid_gate in self.read_tap_resid_gate:
                    nn.init.zeros_(resid_gate.weight)
                    nn.init.zeros_(resid_gate.bias)
                self.read_tap_aux_offset = None
                self.register_buffer('_mtaps_aux_ranges_tns', torch.tensor([], dtype=torch.float32))
                self.register_buffer('_mtaps_aux_fixed_offsets_tns', torch.tensor([], dtype=torch.float32))
            else:
                self.read_tap_gate = None
                self.read_tap_delta_proj = None
                self.read_tap_resid_gate = None
                self.read_tap_aux_offset = None
                self.register_buffer('_mtaps_aux_ranges_tns', torch.tensor([], dtype=torch.float32))
                self.register_buffer('_mtaps_aux_fixed_offsets_tns', torch.tensor([], dtype=torch.float32))
        else:
            self.register_buffer('_mtaps_lags_tns', torch.tensor([], dtype=torch.long))
            self.read_tap_proj = None
            self.read_tap_gate = None
            self.read_tap_delta_proj = None
            self.read_tap_resid_gate = None
            self.read_tap_aux_offset = None
            self.register_buffer('_mtaps_aux_ranges_tns', torch.tensor([], dtype=torch.float32))
            self.register_buffer('_mtaps_aux_fixed_offsets_tns', torch.tensor([], dtype=torch.float32))

        # query_proj: hidden_dim → slot_dim (content-based attention query per expert)
        # created for dotprod/topk reads only — pilot uses its own ptr_query (smaller dim)
        if read_kernel_mode in ('dotprod', 'topk') or write_address_mode == 'content_topk':
            self.query_proj = nn.ModuleList([nn.Linear(hidden_dim, slot_dim) for _ in range(N)])
        self._topk_K = topk_K  # legacy alias for checkpoint compat
        self._read_topk_K = read_topk_K
        self._write_topk_K = write_topk_K

        # write_proj: hidden_dim → slot_dim (compress hidden before writing to ring)
        # only created when dims differ — when equal, write is identity (exact v4 behavior)
        if hidden_dim != slot_dim:
            self.write_proj = nn.ModuleList([nn.Linear(hidden_dim, slot_dim) for _ in range(N)])
        else:
            self.write_proj = None

        # ── Ring gate mode ──
        # Controls how ring read signal blends into hidden state.
        # 'dotprod': α = sigmoid(τ · cosine(input, ring_signal)) — scale-invariant, bounded [0,1]
        # 'fixed':   α = S (from config) — fixed scalar, backward compat
        self.s_constraint = s_constraint  # kept for checkpoint compat
        self.S_raw = nn.Parameter(torch.tensor(1.0))  # full ring signal — let backprop decide
        self.gate_tau = nn.Parameter(torch.tensor(4.0))  # learnable temperature for cosine gate
        self.ring_signal_norm = nn.LayerNorm(hidden_dim)  # normalizes ring signal before blend

        # ── Gated write (anti-blob) ──
        # Mini head per expert: Linear(hidden_dim → 2) → sigmoid → [erase, write_gate]
        # Context-dependent: each timestep/batch element gets its own erase/gate decision.
        # Output 0 = erase (how much old slot content decays), output 1 = write_gate (new write strength)
        # Bias init at 0 → sigmoid(0)=0.5 → let backprop decide the balance
        self.gated_write = gated_write
        if gated_write:
            self.write_head = nn.ModuleList([
                nn.Linear(hidden_dim, 2) for _ in range(N)
            ])

        # ── Adaptive write strength ──
        # Per-timestep write intensity from hidden state: S_write = sigmoid(linear(h))
        # Bias init so sigmoid(bias) ≈ 0.3 (matches old fixed S_raw), weight ≈ 0
        # Old checkpoints without this key → falls back to fixed S_raw behavior
        self.write_gate = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(N)
        ])
        for wg in self.write_gate:
            nn.init.zeros_(wg.weight)
            wg.bias.data.fill_(-0.847)  # sigmoid(-0.847) ≈ 0.3

        # ── Bulletin board temporal cache ──
        # FIFO buffer of past input_vecs with soft attention read at fixed time taps.
        # Experts attend to cached past inputs (t-1, t-5, t-10) via query·key softmax.
        self.bb_enabled = bb_enabled
        if bb_enabled:
            _bb_head_dim = slot_dim  # reuse slot_dim=64 as attention head dim
            self._bb_taps = [1, 5, 10]           # fixed time offsets
            self._bb_L = max(self._bb_taps) + 1  # buffer depth = 11
            self.bb_key_proj = nn.Linear(hidden_dim, _bb_head_dim)  # shared key projection
            self.bb_query_proj = nn.ModuleList([
                nn.Linear(hidden_dim, _bb_head_dim) for _ in range(N)
            ])
            self._bb_gate_mode = bb_gate_mode
            if bb_gate_mode == 'learned':
                self.bb_gate = nn.ModuleList([
                    nn.Linear(hidden_dim, 1) for _ in range(N)
                ])
                for g in self.bb_gate:
                    g.bias.data.fill_(bb_gate_bias)
            else:
                self.bb_gate = None  # fixed mode: no gate, just scale
            self._bb_scale = bb_scale
            self._bb_tau = bb_tau

        # phase embeddings: sin/cos position encoding — hidden_dim wide (adds to hidden state)
        self.phase_cos = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        self.phase_sin = nn.Parameter(torch.randn(hidden_dim) * 0.01)

        # fixed attention radius from config — R_eff = R + 0.5 gives window of exactly 2R+1 slots
        # R=0 → 1 slot (needle), R=1 → 3 slots, R=2 → 5 slots
        self.register_buffer('_R_eff', torch.full((N,), R + 0.5))
        self._max_R: float = R + 0.5  # precomputed scalar — avoids .item() graph break in forward

        # precomputed φ-jump table — shape (N, M), not trainable, moves with device
        self.dests: torch.Tensor
        self.register_buffer('dests', phi_destinations(N, M))

        # ── Pointer movement ──
        self.pointer_mode = pointer_mode
        if pointer_mode == 'learned':
            _ptr_Rmax = M // 16  # max jump distance per step (bounded)
            self._ptr_Rmax = _ptr_Rmax
            # direction head: stay / forward / backward (per expert)
            self.ptr_dir_head = nn.ModuleList([nn.Linear(hidden_dim, 3) for _ in range(N)])
            # magnitude head: how far to jump (per expert)
            self.ptr_mag_head = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(N)])
            # init bias: default to "forward +1" behavior (matches ptr+1 baseline)
            for head in self.ptr_dir_head:
                head.bias.data.copy_(torch.tensor([0.0, 2.0, 0.0]))  # softmax → ~[0.1, 0.8, 0.1]
            for head in self.ptr_mag_head:
                head.bias.data.fill_(-math.log(max(_ptr_Rmax - 1, 1)))  # sigmoid → ~1/Rmax → step≈1
        elif pointer_mode == 'pilot':
            # Pilot pulse: each slot has a learned identity vector.
            # The pulse (hidden state) compares to current slot — similar = stay, different = jump.
            _id_dim = CNFG_PILOTIDDIM_INT                              # identity vector width (default 32)
            _max_jump = CNFG_PILOTJUMP_INT if CNFG_PILOTJUMP_INT > 0 else M // 4  # default M/4
            self._ptr_id_dim = _id_dim
            self._ptr_max_jump = _max_jump
            self._ptr_tau = nn.Parameter(torch.tensor(5.0))            # learnable temperature (softplus → ~5.0)
            self.slot_identity = nn.Parameter(torch.randn(M, _id_dim) * 0.01)  # per-slot identity
            self.ptr_query = nn.ModuleList([nn.Linear(hidden_dim, _id_dim) for _ in range(N)])

        # ── Jump Gate (optional, for sequential pointer mode) ──
        # Learned gate: sigmoid(Linear(hidden)) → probability of φ-jump vs +1 walk.
        # Soft blend: ptr = gate * jump_dest + (1-gate) * (ptr+1).
        # ~hidden_dim params per expert. Init bias -3.0 → sigmoid ≈ 0.05 (mostly walk).
        self.jump_gate_enabled = bool(jump_gate) and pointer_mode == 'sequential'
        if self.jump_gate_enabled:
            self.jump_gate_head = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(N)])
            for head in self.jump_gate_head:
                head.bias.data.fill_(-3.0)  # sigmoid(-3) ≈ 0.047 → mostly walk at init

    @property
    def _can_batch_experts(self) -> bool:
        """Check if the hot-path batched expert loop can be used.

        Returns True only for the v2 default config: sequential pointer,
        vshape/uniform/gaussian kernel, replace write, no exotic features.
        All other configs fall back to the sequential expert loop.
        """
        return (
            self.pointer_mode == 'sequential'
            and not self.jump_gate_enabled
            and self.read_kernel_mode in ('uniform', 'vshape', 'gaussian')
            and self.write_mode == 'replace'
            and not self.mtaps_enabled
            and not self.bb_enabled
            and self.io_split_mode == 'off'
            and self.write_address_mode == 'pointer'
            and not self._proxy_overlay_enabled
        )

    def _process_chunk_batched(self, x_chunk, ring_tns, ptr_tns, hidden_tns,
                               bb_buf, bb_keys, bb_write_ptr, bb_steps,
                               S_flt, probs_lst, offsets_long, expert_weights,
                               topk_write_weights=None):
        """Batched expert variant of _process_chunk.

        All N experts read from the SAME (snapshot) ring state simultaneously,
        compute hidden updates in one fused bmm, then write back sequentially
        (to handle slot conflicts). Pointer moves are vectorized.

        Semantic difference from sequential: Expert₁ does NOT see Expert₀'s
        writes within the same timestep. Empirically equivalent ~73% of time
        (when experts read different slots). Faster due to batched matmuls.

        Only supports the v2 default hot path (see _can_batch_experts).
        """
        M, slot_dim, N, hidden_dim = self.M, self.slot_dim, self.N, self.hidden_dim

        ptr_tns = ptr_tns.clone()
        hidden_tns = hidden_tns.clone()  # (N, B, hidden_dim)

        C = x_chunk.shape[1]
        B = x_chunk.shape[0]
        outs_lst = []

        use_linear_pointer = (
            self.pointer_interp_mode == 'linear'
            and expert_weights is not None
        )

        if self._bitlift:
            _bit_shifts = torch.arange(7, -1, -1, device=x_chunk.device)

        for t in range(C):
            # ── Input encoding (shared across experts) ──
            if self.inp is not None:
                if self._bitlift:
                    byte_val = x_chunk[:, t]
                    bits = ((byte_val.unsqueeze(-1) >> _bit_shifts) & 1).float()
                    input_vec_tns = (_c19_dualphi_activation if self._c19_dualphi else _c19_activation)(
                        self.inp(bits),
                        rho=_rho_from_raw(self.c19_rho_input),
                        C=_C_from_raw(self.c19_C_input),
                    )
                else:
                    input_vec_tns = self.inp(x_chunk[:, t])
            else:
                input_vec_tns = self._fixed_table[x_chunk[:, t]]

            # ── BATCHED READ: all experts read from same ring snapshot ──
            if use_linear_pointer:
                # Linear pointer interp per expert — compute merged windows
                all_indices = []
                all_weights = []
                for i in range(N):
                    base_w = expert_weights[i].unsqueeze(0).expand(B, -1)
                    _, _, merged_idx, merged_w = func_linear_pointer_window_tns(
                        ptr_tns[i], offsets_long, base_w, M,
                    )
                    all_indices.append(merged_idx)      # (B, W) where W = 2R+2
                    all_weights.append(merged_w)         # (B, W)
                # Stack: (N, B, W)
                all_indices_tns = torch.stack(all_indices)   # (N, B, W)
                all_weights_tns = torch.stack(all_weights)   # (N, B, W)
            else:
                # Standard discrete center
                centers = ptr_tns.long().clamp(0, M - 1)                      # (N, B)
                all_indices_tns = (centers.unsqueeze(2) + offsets_long) % M    # (N, B, 2R+1)
                all_weights_tns = expert_weights.unsqueeze(1).expand(-1, B, -1)  # (N, B, 2R+1)

            W = all_indices_tns.shape[2]  # window width (2R+1 or 2R+2)

            # Gather windows: ring (B, M, slot_dim), indices (N, B, W) → (N, B, W, slot_dim)
            idx_exp = all_indices_tns.unsqueeze(-1).expand(-1, -1, -1, slot_dim)  # (N, B, W, slot_dim)
            # Expand ring for N experts
            ring_exp = ring_tns.unsqueeze(0).expand(N, -1, -1, -1)    # (N, B, M, slot_dim)
            windows = ring_exp.gather(2, idx_exp)                      # (N, B, W, slot_dim)

            # Weighted sum → read vectors: (N, B, slot_dim)
            read_vecs = (all_weights_tns.unsqueeze(-1) * windows).sum(2)

            # ── BATCHED RING SIGNAL: read_proj (N separate Linears, batched via bmm) ──
            ring_signals = func_batched_linear(read_vecs, self.read_proj, None)  # (N, B, hidden_dim)

            # ── BATCHED COSINE GATE ──
            if S_flt == 'dotprod':
                # Expand input_vec for N experts: (N, B, hidden_dim)
                inp_exp = input_vec_tns.unsqueeze(0).expand(N, -1, -1)
                cos_sim = F.cosine_similarity(
                    inp_exp, ring_signals, dim=-1
                ).unsqueeze(-1)  # (N, B, 1)
                alpha = torch.sigmoid(self.gate_tau * cos_sim)  # (N, B, 1)
                blended_ring = alpha * ring_signals  # (N, B, hidden_dim)
            else:
                blended_ring = S_flt * ring_signals

            # ── BATCHED PHASE SIGNAL ──
            theta_tns = (ptr_tns / M) * (2 * math.pi)  # (N, B)
            phase_tns = (torch.cos(theta_tns).unsqueeze(-1) * self.phase_cos
                       + torch.sin(theta_tns).unsqueeze(-1) * self.phase_sin)  # (N, B, hidden_dim)

            # ── BATCHED HIDDEN UPDATE (C19 activation) ──
            inp_exp = input_vec_tns.unsqueeze(0).expand(N, -1, -1)  # (N, B, hidden_dim)
            c19_fn = _c19_dualphi_activation if self._c19_dualphi else _c19_activation
            # Reshape to (N*B, hidden_dim) for C19, then back
            pre_act = inp_exp + blended_ring + phase_tns + hidden_tns  # (N, B, hidden_dim)
            hidden_tns = c19_fn(
                pre_act.reshape(N * B, hidden_dim),
                rho=_rho_from_raw(self.c19_rho_hidden),
                C=_C_from_raw(self.c19_C_hidden),
            ).reshape(N, B, hidden_dim)

            # ── WRITE: sequential per expert (scatter conflicts) ──
            if self.write_proj is not None:
                write_vecs = func_batched_linear(hidden_tns, self.write_proj, None)  # (N, B, slot_dim)
            else:
                write_vecs = hidden_tns  # identity when hidden_dim == slot_dim

            if self._expert_conf is not None:
                write_vecs = self._expert_conf.view(N, 1, 1) * write_vecs

            # Write gate: per expert
            write_strengths = []
            for i in range(N):
                ws = torch.sigmoid(self.write_gate[i](hidden_tns[i]))  # (B, 1)
                write_strengths.append(ws)

            # Write back to ring sequentially (N is small, scatter conflict safety)
            for i in range(N):
                w_idx = all_indices_tns[i].unsqueeze(-1).expand(-1, -1, slot_dim)  # (B, W, slot_dim)
                ring_tns = func_hdd_write_tns(
                    ring_tns, write_vecs[i], w_idx,
                    all_weights_tns[i], write_strength=write_strengths[i],
                )

            # ── BATCHED POINTER MOVE (sequential: +1) ──
            ptr_tns = (ptr_tns + 1) % M

            # ── OUTPUT HEAD ──
            mean_hidden = hidden_tns.mean(0)  # (B, hidden_dim)
            if self._bitlift_out:
                bit_scores = torch.tanh(self.out(mean_hidden))
                outs_lst.append(bit_scores @ self._bit_patterns.T)
            elif self.out is not None:
                outs_lst.append(self.out(mean_hidden))
            else:
                outs_lst.append(mean_hidden @ self._fixed_output_table.T)

        hidden_tns_out = hidden_tns  # already (N, B, hidden_dim)
        outs_tns = torch.stack(outs_lst, dim=1)  # (B, C, vocab_size)

        # Diagnostics (last timestep only, to match sequential behavior)
        if self._diag_enabled:
            d = self._diag
            d['ring_norm'] = ring_tns.detach().norm().item()
            d['ring_slot_mean'] = ring_tns.detach().norm(dim=-1).mean().item()
            for i in range(N):
                d[f'hidden_final_norm_{i}'] = hidden_tns[i].detach().norm(dim=-1).mean().item()
                d[f'ptr_pos_{i}'] = ptr_tns[i].detach().float().mean().item()
                if S_flt == 'dotprod':
                    # Report last-timestep alpha for compatibility
                    d[f'alpha_{i}_mean'] = alpha[i].detach().mean().item()

        return ring_tns, ptr_tns, hidden_tns_out, bb_buf, bb_keys, bb_write_ptr, bb_steps, outs_tns

    def _process_chunk(self, x_chunk, ring_tns, ptr_tns, hidden_tns,
                       bb_buf, bb_keys, bb_write_ptr, bb_steps,
                       S_flt, probs_lst, offsets_long, expert_weights,
                       topk_write_weights=None):
        """Process C timesteps of the T×N loop.

        Extracted from forward() so the same code serves both the
        checkpointed path (wrapped in grad_checkpoint) and the
        non-checkpointed path (called directly).  Zero duplication.

        Args:
            x_chunk:        (B, C) or (B, C, B_bits) — input slice for C timesteps
            ring_tns:       (B, M, slot_dim)          — ring buffer state
            ptr_tns:        (N, B)                    — pointer positions (float)
            hidden_tns:     (N, B, hidden_dim)         — packed hidden states
            S_flt:          str|float                  — 'dotprod' or fixed scale factor
            probs_lst:      list[float]                — jump/walk probabilities per expert role
            offsets_long:   (2·R_max+1,)               — precomputed window offsets
            expert_weights: (N, 2·R_max+1)             — precomputed attention kernel weights
            topk_write_weights: (N, 2·R_max+1) or None — vshape weights for topK write path

        Returns:
            tuple of (ring_tns, ptr_tns, hidden_tns, outs_tns):
                ring_tns:   (B, M, slot_dim)     — updated ring buffer
                ptr_tns:    (N, B)               — updated pointer positions
                hidden_tns: (N, B, hidden_dim)    — updated packed hidden states
                outs_tns:   (B, C, vocab_size)   — output logits for this chunk
        """
        # ── Dispatch to batched expert path when eligible ──
        if self._can_batch_experts:
            return self._process_chunk_batched(
                x_chunk, ring_tns, ptr_tns, hidden_tns,
                bb_buf, bb_keys, bb_write_ptr, bb_steps,
                S_flt, probs_lst, offsets_long, expert_weights,
                topk_write_weights,
            )

        M, slot_dim, N, hidden_dim = self.M, self.slot_dim, self.N, self.hidden_dim

        # Clone ptr_tns to protect the input from in-place modification (ptr_tns[i] = ...).
        # Without this, grad_checkpoint's recompute pass would see corrupted pointer state
        # because the original input tensor was mutated during the first forward pass.
        ptr_tns = ptr_tns.clone()

        # Unpack hidden states from (N, B, hidden_dim) tensor into per-expert list
        hidden_lst = [hidden_tns[i] for i in range(N)]

        C = x_chunk.shape[1]  # chunk length (may be < checkpoint_chunks for the last chunk)
        outs_lst = []
        proxy_overlay_enabled = self._proxy_overlay_enabled
        overlay_start_int = 0
        overlay_len_int = 0
        overlay_steps_int = 0
        overlay_vals_tns = None
        proxy_center_int = None
        topk_diag_sum_dist = 0.0
        topk_diag_sum_outside = 0.0
        topk_diag_sum_entropy = 0.0
        topk_diag_sum_unique = 0.0
        write_topk_diag_sum_dist = 0.0
        write_topk_diag_sum_outside = 0.0
        write_topk_diag_sum_unique = 0.0
        topk_diag_count = 0
        write_topk_diag_count = 0
        ring_trace = None
        if _RING_TRACE_ENABLED:
            ring_trace = {
                'ptr_trace': [],
                'read_idx_trace': [],
                'read_weight_trace': [],
                'tap_idx_trace': [],
                'write_idx_trace': [],
                'write_weight_trace': [],
                'read_write_overlap_trace': [],
                'center_hist': torch.zeros(M, dtype=torch.long),
                'read_hist': torch.zeros(M, dtype=torch.long),
                'tap_hist': torch.zeros(M, dtype=torch.long),
                'write_hist': torch.zeros(M, dtype=torch.long),
            }
        if proxy_overlay_enabled:
            ptr0_tns = ptr_tns[0].long().clamp(0, M - 1)
            if bool((ptr0_tns != ptr0_tns[:1]).any().item()):
                proxy_overlay_enabled = False
            else:
                proxy_center_int = int(ptr0_tns[0].item())

        if self._bitlift:
            _bit_shifts = torch.arange(7, -1, -1, device=x_chunk.device)  # [7,6,5,4,3,2,1,0]

        for t in range(C):
            if self.inp is not None:
                if self._bitlift:
                    # byte index → 8 float bits, GPU-native
                    byte_val = x_chunk[:, t]                               # (B,) int64
                    bits = ((byte_val.unsqueeze(-1) >> _bit_shifts) & 1).float()  # (B, 8)
                    input_vec_tns = (_c19_dualphi_activation if self._c19_dualphi else _c19_activation)(self.inp(bits), rho=_rho_from_raw(self.c19_rho_input), C=_C_from_raw(self.c19_C_input))  # Linear+c19: (B, 8) → (B, hidden_dim)
                else:
                    input_vec_tns = self.inp(x_chunk[:, t])                # Embedding: int64 → (B, hidden_dim)
            else:
                input_vec_tns = self._fixed_table[x_chunk[:, t]]           # (B, hidden_dim)

            for i in range(N):
                # ─ PRE-READ pointer seek (pilot: move BEFORE reading) ─
                if self.pointer_mode == 'pilot':
                    current_slot = ptr_tns[i].long().clamp(0, M - 1)                  # (B,)
                    slot_id = F.normalize(self.slot_identity[current_slot], dim=-1)    # (B, id_dim)
                    query = F.normalize(self.ptr_query[i](hidden_lst[i]), dim=-1)      # (B, id_dim)
                    sim = (query * slot_id).sum(-1)                                    # (B,) cosine sim
                    tau = F.softplus(self._ptr_tau)
                    jump = self._ptr_max_jump * torch.sigmoid(-sim * tau)              # (B,)
                    target_tns = torch.remainder(ptr_tns[i] + 1 + jump, float(M))
                    if self.pointer_seam_mode == 'shortest_arc':
                        delta_tns = func_shortest_arc_delta_tns(ptr_tns[i], target_tns, M)
                        ptr_tns[i] = torch.remainder(ptr_tns[i] + delta_tns, float(M))
                    else:
                        ptr_tns[i] = target_tns

                    if self._diag_enabled:
                        d = self._diag
                        d[f'jump_mean_{i}'] = jump.detach().mean().item()
                        d[f'sim_mean_{i}'] = sim.detach().mean().item()
                        d['ptr_tau'] = tau.detach().item()

                with _source_scope('window_prepare'):
                    if proxy_overlay_enabled:
                        center = torch.full(
                            (input_vec_tns.shape[0],),
                            proxy_center_int,
                            device=x_chunk.device,
                            dtype=torch.long,
                        )
                    else:
                        center = ptr_tns[i].long().clamp(0, M - 1)
                    indices_tns = (center.unsqueeze(1) + offsets_long) % M

                # ── Hourglass I/O split: role flags ──
                is_writer = bool(self._writer_mask[i]) if self.io_split_mode == 'strict' else False
                is_reader = bool(self._reader_mask[i]) if self.io_split_mode == 'strict' else True
                current_window_tns = None
                local_pointer_indices_tns = indices_tns
                local_pointer_weights_tns = None
                local_pointer_expanded_idx_tns = None
                content_topk_scores_tns = None
                content_topk_idx_tns = None
                max_content_topk = 0
                if self.read_kernel_mode == 'topk':
                    max_content_topk = max(max_content_topk, self._read_topk_K)
                if self.write_address_mode == 'content_topk':
                    max_content_topk = max(max_content_topk, self._write_topk_K)
                if max_content_topk > 0:
                    q = self.query_proj[i](hidden_lst[i])                           # (B, slot_dim)
                    scores = torch.bmm(ring_tns, q.unsqueeze(-1)).squeeze(-1)       # (B, M)
                    scores = scores * (slot_dim ** -0.5)                            # scale
                    content_topk_scores_tns, content_topk_idx_tns = scores.topk(max_content_topk, dim=-1)

                use_linear_pointer = (
                    self.pointer_interp_mode == 'linear'
                    and expert_weights is not None
                    and self.read_kernel_mode in ('uniform', 'vshape', 'gaussian')
                    and self.write_address_mode == 'pointer'
                )
                if use_linear_pointer:
                    with _source_scope('window_prepare'):
                        base_weights_tns = expert_weights[i].unsqueeze(0).expand(input_vec_tns.shape[0], -1)
                        center, ptr_alpha_tns, local_pointer_indices_tns, local_pointer_weights_tns = func_linear_pointer_window_tns(
                            ptr_tns[i],
                            offsets_long,
                            base_weights_tns,
                            M,
                        )
                    if self._diag_enabled:
                        self._diag[f'ptr_alpha_{i}'] = ptr_alpha_tns.detach().mean().item()

                tap_idx_src = None
                tap_neighbors_tns = None
                aux_head_reads_tns = None

                if is_writer and not is_reader:
                    # Writer-only: NO ring read, zero ring signal
                    with _source_scope('window_prepare'):
                        read_vec_tns = torch.zeros(input_vec_tns.shape[0], slot_dim, device=input_vec_tns.device)
                        local_pointer_expanded_idx_tns = local_pointer_indices_tns.unsqueeze(-1).expand(-1, -1, slot_dim)
                        if local_pointer_weights_tns is not None:
                            pass
                        elif expert_weights is not None:
                            local_pointer_weights_tns = expert_weights[i].unsqueeze(0).expand(input_vec_tns.shape[0], -1)
                        elif topk_write_weights is not None:
                            local_pointer_weights_tns = topk_write_weights[i].unsqueeze(0).expand(input_vec_tns.shape[0], -1)
                        else:
                            # dotprod fallback: use uniform weights for write
                            local_pointer_weights_tns = torch.ones(input_vec_tns.shape[0], local_pointer_indices_tns.shape[1], device=input_vec_tns.device)
                            local_pointer_weights_tns = local_pointer_weights_tns / local_pointer_weights_tns.sum(dim=1, keepdim=True)
                elif expert_weights is not None:
                    # ─ positional kernel (vshape/gaussian/uniform) ─
                    with _source_scope('window_prepare'):
                        if local_pointer_weights_tns is None:
                            local_pointer_weights_tns = expert_weights[i].unsqueeze(0).expand(input_vec_tns.shape[0], -1)
                    with _source_scope('softread'):
                        if proxy_overlay_enabled:
                            read_vec_tns, local_pointer_expanded_idx_tns, current_window_tns = func_proxy_overlay_read_tns(
                                ring_tns,
                                local_pointer_indices_tns,
                                local_pointer_weights_tns,
                                slot_dim,
                                overlay_start_int,
                                overlay_vals_tns,
                                overlay_len_int,
                                M,
                            )
                        else:
                            read_vec_tns, local_pointer_expanded_idx_tns = func_softread_tns(
                                ring_tns, local_pointer_indices_tns, local_pointer_weights_tns, slot_dim)
                elif self.read_kernel_mode == 'topk':
                    # ─ topK: content-based global search over entire ring ─
                    topk_scores = content_topk_scores_tns[:, :self._read_topk_K]
                    topk_idx = content_topk_idx_tns[:, :self._read_topk_K]
                    topk_attn = F.softmax(topk_scores, dim=-1)                      # (B, K)
                    if self._diag_enabled:
                        center_long = center.unsqueeze(1)
                        delta_fwd = torch.remainder(topk_idx - center_long, M).float()
                        delta_back = torch.remainder(center_long - topk_idx, M).float()
                        circ_dist = torch.minimum(delta_fwd, delta_back)
                        outside_frac = (circ_dist > float(self.R)).float().mean().item()
                        probs_safe = topk_attn.clamp_min(1e-12)
                        entropy = -(probs_safe * probs_safe.log()).sum(dim=-1).mean().item()
                        unique_vals = [
                            row.unique().numel() / max(int(row.numel()), 1)
                            for row in topk_idx.detach()
                        ]
                        unique_frac = float(sum(unique_vals) / max(len(unique_vals), 1))
                        topk_diag_sum_dist += circ_dist.mean().item()
                        topk_diag_sum_outside += outside_frac
                        topk_diag_sum_entropy += entropy
                        topk_diag_sum_unique += unique_frac
                        topk_diag_count += 1
                    topk_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, slot_dim) # (B, K, slot_dim)
                    topk_neighbors = ring_tns.gather(1, topk_expanded)              # (B, K, slot_dim)
                    read_vec_tns = (topk_attn.unsqueeze(-1) * topk_neighbors).sum(1)  # (B, slot_dim)
                else:
                    # ─ dot-product attention within local window ─
                    with _source_scope('window_prepare'):
                        local_pointer_expanded_idx_tns = indices_tns.unsqueeze(-1).expand(-1, -1, slot_dim)
                    with _source_scope('softread'):
                        neighbors_tns = ring_tns.gather(1, local_pointer_expanded_idx_tns)        # (B, 2R+1, slot_dim)
                    current_window_tns = neighbors_tns
                    q = self.query_proj[i](hidden_lst[i])                       # (B, slot_dim)
                    scores = (q.unsqueeze(1) * neighbors_tns).sum(-1)           # (B, 2R+1)
                    scores = scores * (slot_dim ** -0.5)                        # scale
                    local_pointer_weights_tns = F.softmax(scores, dim=-1)                     # (B, 2R+1)
                    read_vec_tns = (local_pointer_weights_tns.unsqueeze(-1) * neighbors_tns).sum(1)  # (B, slot_dim)

                if self.mtaps_enabled:
                    with _source_scope('softread'):
                        tap_idx_src = torch.remainder(
                            center.unsqueeze(1) - self._mtaps_lags_tns.unsqueeze(0),
                            M,
                        )
                        tap_expanded_idx_tns = tap_idx_src.unsqueeze(-1).expand(-1, -1, slot_dim)
                        tap_neighbors_tns = ring_tns.gather(1, tap_expanded_idx_tns)
                        if self.mtaps_mixer_mode in ('hybrid_heads_scalar_gate', 'hybrid_heads_spaced_scalar_gate', 'hybrid_heads_fixed_scalar_gate'):
                            base_weights_tns = expert_weights[i].unsqueeze(0).expand(input_vec_tns.shape[0], -1)
                            if self.mtaps_mixer_mode == 'hybrid_heads_fixed_scalar_gate':
                                aux_offsets_tns = -self._mtaps_aux_fixed_offsets_tns.unsqueeze(0).expand(input_vec_tns.shape[0], -1)
                            else:
                                aux_offset_logits_tns = self.read_tap_aux_offset[i](hidden_lst[i])
                                aux_offsets_tns = torch.tanh(aux_offset_logits_tns) * self._mtaps_aux_ranges_tns.unsqueeze(0)
                            if self.mtaps_mixer_mode == 'hybrid_heads_spaced_scalar_gate':
                                # Hard-space the two learned heads:
                                # head 0 stays near/medium, head 1 is forced to remain
                                # outside the local shortcut band and at least 8 slots away
                                # from head 0 on the ring.
                                near_off_tns = aux_offsets_tns[:, 0]
                                raw_far_off_tns = aux_offsets_tns[:, 1]
                                far_sign_tns = torch.where(raw_far_off_tns >= 0, 1.0, -1.0)
                                far_mag_tns = raw_far_off_tns.abs() / self._mtaps_aux_ranges_tns[1].clamp_min(1e-8)
                                far_mag_tns = 8.0 + far_mag_tns * (self._mtaps_aux_ranges_tns[1] - 8.0)
                                far_off_tns = far_sign_tns * far_mag_tns
                                near_center_tns = torch.remainder(ptr_tns[i] + near_off_tns, float(M))
                                far_center_tns = torch.remainder(ptr_tns[i] + far_off_tns, float(M))
                                signed_delta_tns = torch.remainder(
                                    far_center_tns - near_center_tns + (float(M) / 2.0),
                                    float(M),
                                ) - (float(M) / 2.0)
                                spacing_sign_tns = torch.where(
                                    signed_delta_tns.abs() < 1e-6,
                                    far_sign_tns,
                                    torch.sign(signed_delta_tns),
                                )
                                far_center_tns = torch.where(
                                    signed_delta_tns.abs() < 8.0,
                                    torch.remainder(near_center_tns + spacing_sign_tns * 8.0, float(M)),
                                    far_center_tns,
                                )
                                aux_centers_tns = torch.stack([near_center_tns, far_center_tns], dim=1)
                                far_off_rewrapped_tns = torch.remainder(
                                    far_center_tns - ptr_tns[i] + (float(M) / 2.0),
                                    float(M),
                                ) - (float(M) / 2.0)
                                aux_offsets_tns = torch.stack([near_off_tns, far_off_rewrapped_tns], dim=1)
                            aux_centers_tns = torch.remainder(ptr_tns[i].unsqueeze(1) + aux_offsets_tns, float(M))
                            aux_reads_lst = []
                            aux_center_dist_lst = []
                            aux_near_local_lst = []
                            aux_unique_frac_lst = []
                            for head_idx in range(aux_centers_tns.shape[1]):
                                head_center0_tns, _head_alpha_tns, head_indices_tns, head_weights_tns = func_linear_pointer_window_tns(
                                    aux_centers_tns[:, head_idx],
                                    offsets_long,
                                    base_weights_tns,
                                    M,
                                )
                                head_read_tns, _ = func_softread_tns(
                                    ring_tns,
                                    head_indices_tns,
                                    head_weights_tns,
                                    slot_dim,
                                )
                                aux_reads_lst.append(head_read_tns)
                                if self._diag_enabled:
                                    center_long = center.long()
                                    delta_fwd = torch.remainder(head_center0_tns - center_long, M).float()
                                    delta_back = torch.remainder(center_long - head_center0_tns, M).float()
                                    head_circ_dist = torch.minimum(delta_fwd, delta_back)
                                    aux_center_dist_lst.append(head_circ_dist.mean().item())
                                    aux_near_local_lst.append((head_circ_dist <= float(self.R)).float().mean().item())
                                    aux_unique_frac_lst.append(
                                        head_center0_tns.detach().unique().numel() / max(int(head_center0_tns.numel()), 1)
                                    )
                            aux_head_reads_tns = torch.stack(aux_reads_lst, dim=1)
                            if self._diag_enabled:
                                self._diag[f'head_offset_mean_abs_{i}'] = aux_offsets_tns.detach().abs().mean(dim=0).cpu().tolist()
                                self._diag[f'head_offset_std_{i}'] = aux_offsets_tns.detach().std(dim=0, unbiased=False).cpu().tolist()
                                self._diag[f'head_near_local_frac_{i}'] = aux_near_local_lst
                                self._diag[f'head_center_dist_mean_{i}'] = aux_center_dist_lst
                                self._diag[f'head_unique_frac_{i}'] = aux_unique_frac_lst
                                if aux_centers_tns.shape[1] >= 2:
                                    head0_tns = aux_centers_tns[:, 0]
                                    head1_tns = aux_centers_tns[:, 1]
                                    delta_fwd = torch.remainder(head1_tns - head0_tns, M).float()
                                    delta_back = torch.remainder(head0_tns - head1_tns, M).float()
                                    head_head_dist_tns = torch.minimum(delta_fwd, delta_back)
                                    self._diag[f'head_pair_dist_mean_{i}'] = head_head_dist_tns.mean().item()
                                    self._diag[f'head_pair_near_frac_{i}'] = (head_head_dist_tns < 8.0).float().mean().item()

                if self.write_address_mode == 'content_topk':
                    write_topk_scores = content_topk_scores_tns[:, :self._write_topk_K]
                    write_topk_idx = content_topk_idx_tns[:, :self._write_topk_K]
                    write_weights_tns = F.softmax(write_topk_scores, dim=-1)
                    write_expanded_idx_tns = write_topk_idx.unsqueeze(-1).expand(-1, -1, slot_dim)
                    if self._diag_enabled:
                        center_long = center.unsqueeze(1)
                        delta_fwd = torch.remainder(write_topk_idx - center_long, M).float()
                        delta_back = torch.remainder(center_long - write_topk_idx, M).float()
                        write_circ_dist = torch.minimum(delta_fwd, delta_back)
                        write_topk_diag_sum_dist += write_circ_dist.mean().item()
                        write_topk_diag_sum_outside += (write_circ_dist > float(self.R)).float().mean().item()
                        write_unique_vals = [
                            row.unique().numel() / max(int(row.numel()), 1)
                            for row in write_topk_idx.detach()
                        ]
                        write_topk_diag_sum_unique += float(sum(write_unique_vals) / max(len(write_unique_vals), 1))
                        write_topk_diag_count += 1
                else:
                    if local_pointer_expanded_idx_tns is None:
                        local_pointer_expanded_idx_tns = local_pointer_indices_tns.unsqueeze(-1).expand(-1, -1, slot_dim)
                    if local_pointer_weights_tns is None:
                        local_pointer_weights_tns = torch.ones(input_vec_tns.shape[0], local_pointer_indices_tns.shape[1], device=input_vec_tns.device)
                        local_pointer_weights_tns = local_pointer_weights_tns / local_pointer_weights_tns.sum(dim=1, keepdim=True)
                    write_expanded_idx_tns = local_pointer_expanded_idx_tns
                    write_weights_tns = local_pointer_weights_tns

                if ring_trace is not None and i == 0:
                    if self.read_kernel_mode == 'topk':
                        read_idx_src = topk_idx
                        read_w_src = topk_attn
                    else:
                        read_idx_src = local_pointer_indices_tns
                        read_w_src = local_pointer_weights_tns
                    if self.write_address_mode == 'content_topk':
                        write_idx_src = write_topk_idx
                        write_w_src = write_weights_tns
                    else:
                        write_idx_src = local_pointer_indices_tns
                        write_w_src = write_weights_tns

                    ptr_sample = int(center[0].detach().item())
                    read_idx_sample = [int(x) for x in read_idx_src[0].detach().tolist()]
                    read_weight_sample = [float(x) for x in read_w_src[0].detach().tolist()]
                    write_idx_sample = [int(x) for x in write_idx_src[0].detach().tolist()]
                    write_weight_sample = [float(x) for x in write_w_src[0].detach().tolist()]
                    overlap = len(set(read_idx_sample) & set(write_idx_sample)) / max(len(set(read_idx_sample) | set(write_idx_sample)), 1)

                    ring_trace['ptr_trace'].append(ptr_sample)
                    ring_trace['read_idx_trace'].append(read_idx_sample)
                    ring_trace['read_weight_trace'].append(read_weight_sample)
                    ring_trace['tap_idx_trace'].append(
                        [int(x) for x in tap_idx_src[0].detach().tolist()] if tap_idx_src is not None else []
                    )
                    ring_trace['write_idx_trace'].append(write_idx_sample)
                    ring_trace['write_weight_trace'].append(write_weight_sample)
                    ring_trace['read_write_overlap_trace'].append(float(overlap))
                    ring_trace['center_hist'] += torch.bincount(center.detach().cpu(), minlength=M)
                    ring_trace['read_hist'] += torch.bincount(read_idx_src.detach().reshape(-1).cpu(), minlength=M)
                    if tap_idx_src is not None:
                        ring_trace['tap_hist'] += torch.bincount(tap_idx_src.detach().reshape(-1).cpu(), minlength=M)
                    ring_trace['write_hist'] += torch.bincount(write_idx_src.detach().reshape(-1).cpu(), minlength=M)

                # ─ phase signal ─
                theta_tns = (ptr_tns[i] / M) * (2 * math.pi)
                phase_tns = (torch.cos(theta_tns).unsqueeze(-1) * self.phase_cos
                           + torch.sin(theta_tns).unsqueeze(-1) * self.phase_sin)

                # ─ hidden update (all hidden_dim-wide) ─
                if self.mtaps_enabled:
                    tap_flat_tns = tap_neighbors_tns.reshape(input_vec_tns.shape[0], -1)
                    if self.mtaps_mixer_mode == 'current':
                        mtap_input_tns = torch.cat([read_vec_tns, tap_flat_tns], dim=-1)
                        ring_signal = self.read_tap_proj[i](mtap_input_tns)
                    elif self.mtaps_mixer_mode == 'tap_scalar_gate':
                        gate_logits = self.read_tap_gate[i](hidden_lst[i])  # (B, 1+K)
                        gate_vals = torch.sigmoid(gate_logits)
                        main_gate = gate_vals[:, :1]
                        tap_gate = gate_vals[:, 1:]
                        tap_gate_norm = tap_gate / tap_gate.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                        mtap_input_tns = torch.cat(
                            [
                                main_gate * read_vec_tns,
                                (tap_gate.unsqueeze(-1) * tap_neighbors_tns).reshape(input_vec_tns.shape[0], -1),
                            ],
                            dim=-1,
                        )
                        ring_signal = self.read_tap_proj[i](mtap_input_tns)
                        if self._diag_enabled:
                            probs_safe = tap_gate_norm.clamp_min(1e-12)
                            self._diag[f'mtap_main_frac_{i}'] = (main_gate / gate_vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)).detach().mean().item()
                            self._diag[f'mtap_gate_mean_by_lag_{i}'] = tap_gate_norm.detach().mean(dim=0).cpu().tolist()
                            self._diag[f'mtap_gate_max_frac_{i}'] = tap_gate_norm.detach().max(dim=-1).values.mean().item()
                            self._diag[f'mtap_gate_entropy_{i}'] = (-(probs_safe * probs_safe.log()).sum(dim=-1).mean().item())
                    elif self.mtaps_mixer_mode in ('hybrid_heads_scalar_gate', 'hybrid_heads_spaced_scalar_gate', 'hybrid_heads_fixed_scalar_gate'):
                        gate_logits = self.read_tap_gate[i](hidden_lst[i])  # (B, 1+K+2)
                        gate_vals = torch.sigmoid(gate_logits)
                        main_gate = gate_vals[:, :1]
                        tap_gate = gate_vals[:, 1:1 + len(self._mtaps_lags)]
                        head_gate = gate_vals[:, 1 + len(self._mtaps_lags):]
                        channel_gate_norm = gate_vals / gate_vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                        tap_gate_norm = tap_gate / tap_gate.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                        head_gate_norm = head_gate / head_gate.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                        mtap_input_tns = torch.cat(
                            [
                                main_gate * read_vec_tns,
                                (tap_gate.unsqueeze(-1) * tap_neighbors_tns).reshape(input_vec_tns.shape[0], -1),
                                (head_gate.unsqueeze(-1) * aux_head_reads_tns).reshape(input_vec_tns.shape[0], -1),
                            ],
                            dim=-1,
                        )
                        ring_signal = self.read_tap_proj[i](mtap_input_tns)
                        if self._diag_enabled:
                            probs_safe = channel_gate_norm.clamp_min(1e-12)
                            self._diag[f'mtap_main_frac_{i}'] = channel_gate_norm[:, :1].detach().mean().item()
                            self._diag[f'mtap_gate_mean_by_lag_{i}'] = tap_gate_norm.detach().mean(dim=0).cpu().tolist()
                            self._diag[f'mtap_gate_max_frac_{i}'] = tap_gate_norm.detach().max(dim=-1).values.mean().item()
                            self._diag[f'mtap_gate_entropy_{i}'] = (-(tap_gate_norm.clamp_min(1e-12) * tap_gate_norm.clamp_min(1e-12).log()).sum(dim=-1).mean().item())
                            self._diag[f'head_gate_mean_{i}'] = channel_gate_norm[:, 1 + len(self._mtaps_lags):].detach().mean(dim=0).cpu().tolist()
                            self._diag[f'head_gate_max_frac_{i}'] = channel_gate_norm[:, 1 + len(self._mtaps_lags):].detach().max(dim=-1).values.mean().item()
                            self._diag[f'channel_gate_entropy_{i}'] = (-(probs_safe * probs_safe.log()).sum(dim=-1).mean().item())
                    else:
                        tap_gate_logits = self.read_tap_gate[i](hidden_lst[i])  # (B, K)
                        tap_gate = torch.sigmoid(tap_gate_logits)
                        tap_gate_norm = tap_gate / tap_gate.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                        tap_delta = self.read_tap_delta_proj[i](
                            (tap_gate.unsqueeze(-1) * tap_neighbors_tns).reshape(input_vec_tns.shape[0], -1)
                        )
                        resid_beta = torch.sigmoid(self.read_tap_resid_gate[i](hidden_lst[i]))
                        ring_signal = self.read_proj[i](read_vec_tns) + resid_beta * tap_delta
                        if self._diag_enabled:
                            probs_safe = tap_gate_norm.clamp_min(1e-12)
                            self._diag[f'mtap_gate_mean_by_lag_{i}'] = tap_gate_norm.detach().mean(dim=0).cpu().tolist()
                            self._diag[f'mtap_gate_max_frac_{i}'] = tap_gate_norm.detach().max(dim=-1).values.mean().item()
                            self._diag[f'mtap_gate_entropy_{i}'] = (-(probs_safe * probs_safe.log()).sum(dim=-1).mean().item())
                            self._diag[f'mtap_resid_beta_{i}'] = resid_beta.detach().mean().item()
                            self._diag[f'mtap_delta_norm_{i}'] = tap_delta.detach().norm(dim=-1).mean().item()
                    if self._diag_enabled:
                        self._diag[f'mtap_main_norm_{i}'] = read_vec_tns.detach().norm(dim=-1).mean().item()
                        self._diag[f'mtap_tap_norm_{i}'] = tap_neighbors_tns.detach().norm(dim=-1).mean().item()
                        self._diag[f'mtap_signal_norm_{i}'] = ring_signal.detach().norm(dim=-1).mean().item()
                else:
                    ring_signal = self.read_proj[i](read_vec_tns)  # slot_dim → hidden_dim
                if S_flt == 'dotprod':
                    # content-based gate: cosine similarity (scale-invariant)
                    cos_sim = F.cosine_similarity(
                        input_vec_tns, ring_signal, dim=-1
                    ).unsqueeze(-1)  # (B, 1) — always [-1, +1] regardless of norms
                    alpha = torch.sigmoid(self.gate_tau * cos_sim)  # (B, 1) — bounded [0, 1]
                    blended_ring = alpha * ring_signal  # no LayerNorm — keep magnitude info

                    # ── diagnostics: track alpha + signal norms ──
                    if self._diag_enabled:
                        d = self._diag
                        a = alpha.detach()
                        d[f'alpha_{i}_mean'] = a.mean().item()
                        d[f'alpha_{i}_min'] = a.min().item()
                        d[f'alpha_{i}_max'] = a.max().item()
                        d[f'input_norm_{i}'] = input_vec_tns.detach().norm(dim=-1).mean().item()
                        d[f'ring_signal_norm_{i}'] = ring_signal.detach().norm(dim=-1).mean().item()
                        d[f'blended_norm_{i}'] = blended_ring.detach().norm(dim=-1).mean().item()
                        d[f'hidden_norm_{i}'] = hidden_lst[i].detach().norm(dim=-1).mean().item()
                        d[f'ptr_pos_{i}'] = ptr_tns[i].detach().float().mean().item()
                else:
                    blended_ring = S_flt * ring_signal

                # ─ bulletin board cache read (before hidden update — causality) ─
                bb_ctx = 0
                if self.bb_enabled and bb_buf is not None:
                    taps = self._bb_taps  # [1, 5, 10]
                    if bb_steps >= taps[0]:  # at least 1 step written
                        tap_idx = [(bb_write_ptr - tap) % self._bb_L for tap in taps]
                        cached_vecs = bb_buf[:, tap_idx]     # (B, K, hidden_dim)
                        cached_keys = bb_keys[:, tap_idx]    # (B, K, head_dim)

                        q = self.bb_query_proj[i](hidden_lst[i])              # (B, head_dim)
                        # L2-normalize q and k → cosine attention (bounded scores)
                        q_norm = F.normalize(q, dim=-1)                        # (B, head_dim)
                        k_norm = F.normalize(cached_keys, dim=-1)              # (B, K, head_dim)
                        scores = (q_norm.unsqueeze(1) * k_norm).sum(-1)        # (B, K) in [-1, 1]
                        scores = scores * self._bb_tau                         # temperature scaling

                        # cold-start mask: -inf for taps not yet available
                        for k_idx, tap in enumerate(taps):
                            if bb_steps < tap:
                                scores[:, k_idx] = float('-inf')

                        weights_bb = F.softmax(scores, dim=-1)                # (B, K)
                        ctx = (weights_bb.unsqueeze(-1) * cached_vecs).sum(1) # (B, hidden_dim)

                        if self.bb_gate is not None:
                            beta = torch.sigmoid(self.bb_gate[i](hidden_lst[i]))  # (B, 1)
                            bb_ctx = self._bb_scale * beta * ctx
                        else:
                            beta = None
                            bb_ctx = self._bb_scale * ctx

                        # ── BB telemetry (every transform point) ──
                        if self._diag_enabled:
                            d = self._diag
                            ctx_raw_n = ctx.detach().norm(dim=-1).mean().item()
                            ctx_scaled_n = bb_ctx.detach().norm(dim=-1).mean().item()
                            inp_n = input_vec_tns.detach().norm(dim=-1).mean().item()
                            ring_n = blended_ring.detach().norm(dim=-1).mean().item()
                            # gate openness
                            d[f'bb_beta_{i}'] = beta.detach().mean().item() if beta is not None else 1.0
                            # cache read magnitudes
                            d[f'bb_ctx_raw_norm_{i}'] = ctx_raw_n
                            d[f'bb_ctx_scaled_norm_{i}'] = ctx_scaled_n
                            # attention sharpness: entropy of softmax weights
                            # max entropy = ln(K) = ln(3) = 1.099; 0 = hard one-hot
                            log_w = torch.log(weights_bb.detach().clamp(min=1e-8))
                            entropy = -(weights_bb.detach() * log_w).sum(-1).mean().item()
                            d[f'bb_attn_entropy_{i}'] = entropy
                            # query/key norms
                            d[f'bb_query_norm_{i}'] = q.detach().norm(dim=-1).mean().item()
                            # ratios vs other signals
                            d[f'bb_ctx_vs_input_{i}'] = ctx_scaled_n / max(inp_n, 1e-8)
                            d[f'bb_ctx_vs_ring_{i}'] = ctx_scaled_n / max(ring_n, 1e-8)

                hidden_lst[i] = (_c19_dualphi_activation if self._c19_dualphi else _c19_activation)(
                    input_vec_tns                              # (B, hidden_dim)
                    + blended_ring                             # (B, hidden_dim)
                    + bb_ctx                                   # (B, hidden_dim) — bulletin board
                    + phase_tns                                # (B, hidden_dim)
                    + hidden_lst[i],                           # (B, hidden_dim)
                    rho=_rho_from_raw(self.c19_rho_hidden),
                    C=_C_from_raw(self.c19_C_hidden),
                )

                # ─ soft_write (compress hidden → slot_dim if needed) ─
                # Hourglass strict: readers don't write
                if not (self.io_split_mode == 'strict' and is_reader and not is_writer):
                    with _source_scope('write_prepare'):
                        if self.write_proj is not None:
                            write_vec = self.write_proj[i](hidden_lst[i])  # hidden_dim → slot_dim
                        else:
                            write_vec = hidden_lst[i]                      # identity when dims match
                        if self._expert_conf is not None:
                            write_vec = self._expert_conf[i] * write_vec
                    if self.write_mode == 'replace':
                        with _source_scope('write_prepare'):
                            ws = torch.sigmoid(self.write_gate[i](hidden_lst[i]))  # (B, 1)
                        with _source_scope('write_replace'):
                            if proxy_overlay_enabled:
                                w = write_weights_tns.unsqueeze(-1) * ws.unsqueeze(1)
                                if current_window_tns is None:
                                    current_window_tns = ring_tns.gather(1, write_expanded_idx_tns)
                                write_val_tns = write_vec.unsqueeze(1).expand(-1, write_weights_tns.size(1), -1)
                                updated_window_tns = w * write_val_tns + (1.0 - w) * current_window_tns
                                window_start_int = (proxy_center_int - self.R) % M
                                ring_tns, overlay_start_int, overlay_vals_tns, overlay_len_int, overlay_steps_int = func_proxy_overlay_write_tns(
                                    ring_tns,
                                    updated_window_tns,
                                    window_start_int,
                                    overlay_start_int,
                                    overlay_vals_tns,
                                    overlay_len_int,
                                    overlay_steps_int,
                                    M,
                                    self._proxy_overlay_flush_interval,
                                    self._proxy_overlay_cap,
                                )
                            else:
                                ring_tns = func_hdd_write_tns(ring_tns, write_vec, write_expanded_idx_tns, write_weights_tns, write_strength=ws)
                    elif self.gated_write:
                        decisions = torch.sigmoid(self.write_head[i](hidden_lst[i]))  # (B, 2)
                        erase = decisions[:, 0]   # (B,) — per-sample erase strength
                        wgate = decisions[:, 1]   # (B,) — per-sample write gate
                        ring_tns = func_gated_write_tns(ring_tns, write_vec, write_expanded_idx_tns, write_weights_tns, erase, wgate)
                    else:
                        ring_tns = func_softwrit_tns(ring_tns, write_vec, write_expanded_idx_tns, write_weights_tns)

                # ─ POST-WRITE pointer move (sequential/learned only; pilot already moved pre-read) ─
                if self.pointer_mode == 'learned':
                    dir_logits = self.ptr_dir_head[i](hidden_lst[i])             # (B, 3)
                    a = F.softmax(dir_logits, dim=-1)                            # (B, 3): [stay, fwd, back]
                    raw_mag = self.ptr_mag_head[i](hidden_lst[i])                # (B, 1)
                    m = self._ptr_Rmax * torch.sigmoid(raw_mag.squeeze(-1))      # (B,) bounded [0, Rmax]
                    direction = a[:, 1] - a[:, 2]                                # (B,) in [-1, 1]
                    delta = (1 - a[:, 0]) * direction * m                        # (B,)
                    with _source_scope('pointer_update'):
                        if self.pointer_seam_mode == 'shortest_arc':
                            target_tns = torch.remainder(ptr_tns[i] + delta, float(M))
                            delta_tns = func_shortest_arc_delta_tns(ptr_tns[i], target_tns, M)
                            ptr_tns[i] = torch.remainder(ptr_tns[i] + delta_tns, float(M))
                        else:
                            ptr_tns[i] = torch.remainder(ptr_tns[i] + delta, float(M))
                elif self.pointer_mode == 'pilot':
                    pass  # already moved pre-read (seek-then-read)
                elif self.jump_gate_enabled:
                    # Learned jump gate: soft blend between walk (+1) and φ-jump
                    with _source_scope('pointer_update'):
                        gate = torch.sigmoid(self.jump_gate_head[i](hidden_lst[i])).squeeze(-1)  # (B,)
                        walk_target = (ptr_tns[i] + 1) % M
                        ptr_int = ptr_tns[i].long().clamp(0, M - 1)
                        jump_target = self.dests[i][ptr_int].float()           # (B,) φ-destination
                        if self.pointer_seam_mode == 'shortest_arc':
                            walk_delta = func_shortest_arc_delta_tns(ptr_tns[i], walk_target.float(), M)
                            jump_delta = func_shortest_arc_delta_tns(ptr_tns[i], jump_target, M)
                            blended_delta = gate * jump_delta + (1 - gate) * walk_delta
                            ptr_tns[i] = torch.remainder(ptr_tns[i] + blended_delta, float(M))
                        else:
                            ptr_tns[i] = torch.remainder(gate * jump_target + (1 - gate) * walk_target, float(M))
                        if self._diag_enabled:
                            self._diag[f'jump_gate_mean_{i}'] = gate.detach().mean().item()
                            self._diag[f'jump_gate_max_{i}'] = gate.detach().max().item()
                else:
                    with _source_scope('pointer_update'):
                        ptr_tns[i] = (ptr_tns[i] + 1) % M
                    if proxy_overlay_enabled:
                        proxy_center_int = (proxy_center_int + 1) % M

            # ─ bulletin board cache write (once per timestep, after all expert reads) ─
            if self.bb_enabled and bb_buf is not None:
                bb_buf = bb_buf.clone()   # avoid in-place on checkpointed tensor
                bb_buf[:, bb_write_ptr] = input_vec_tns
                bb_keys = bb_keys.clone()
                written_key = self.bb_key_proj(input_vec_tns)
                bb_keys[:, bb_write_ptr] = written_key
                bb_write_ptr = (bb_write_ptr + 1) % self._bb_L
                bb_steps += 1
                # key norm telemetry
                if self._diag_enabled:
                    self._diag['bb_key_norm'] = written_key.detach().norm(dim=-1).mean().item()

            # ── Hourglass: reader-only output ──
            with _source_scope('output_head'):
                if self.io_split_mode == 'strict' and self.io_output_from_readers_only:
                    reader_idx = self._reader_mask.nonzero(as_tuple=False).flatten().tolist()
                    mean_hidden = torch.stack([hidden_lst[j] for j in reader_idx]).mean(0)
                else:
                    mean_hidden = torch.stack(hidden_lst).mean(0)       # (B, hidden_dim)
                if self._bitlift_out:
                    bit_scores = torch.tanh(self.out(mean_hidden))   # (B, 8)
                    outs_lst.append(bit_scores @ self._bit_patterns.T)  # (B, 8) @ (8, 256) → (B, 256)
                elif self.out is not None:
                    outs_lst.append(self.out(mean_hidden))           # nn.Linear
                else:
                    outs_lst.append(mean_hidden @ self._fixed_output_table.T)  # (B, 256)

        # Repack hidden states and stack outputs
        hidden_tns = torch.stack(hidden_lst)       # (N, B, hidden_dim)
        with _source_scope('output_head'):
            outs_tns = torch.stack(outs_lst, dim=1)    # (B, C, vocab_size)

        if proxy_overlay_enabled and overlay_len_int > 0:
            with _source_scope('write_replace'):
                ring_tns = func_proxy_overlay_flush_tns(
                    ring_tns,
                    overlay_start_int,
                    overlay_vals_tns,
                    overlay_len_int,
                    M,
                )
            overlay_vals_tns = None
            overlay_len_int = 0

        # ── diagnostics: ring + final hidden norms ──
        if self._diag_enabled:
            d = self._diag
            d['ring_norm'] = ring_tns.detach().norm().item()
            d['ring_slot_mean'] = ring_tns.detach().norm(dim=-1).mean().item()
            for i in range(self.N):
                d[f'hidden_final_norm_{i}'] = hidden_tns[i].detach().norm(dim=-1).mean().item()
        if self._diag_enabled and topk_diag_count > 0:
            self._diag['topk_mean_abs_circ_dist'] = topk_diag_sum_dist / topk_diag_count
            self._diag['topk_outside_local_frac'] = topk_diag_sum_outside / topk_diag_count
            self._diag['topk_attn_entropy'] = topk_diag_sum_entropy / topk_diag_count
            self._diag['topk_unique_slot_frac'] = topk_diag_sum_unique / topk_diag_count
        if self._diag_enabled and write_topk_diag_count > 0:
            self._diag['write_topk_mean_abs_circ_dist'] = write_topk_diag_sum_dist / write_topk_diag_count
            self._diag['write_topk_outside_local_frac'] = write_topk_diag_sum_outside / write_topk_diag_count
            self._diag['write_topk_unique_slot_frac'] = write_topk_diag_sum_unique / write_topk_diag_count
        if ring_trace is not None:
            self._ring_trace = {
                'ptr_trace': ring_trace['ptr_trace'],
                'read_idx_trace': ring_trace['read_idx_trace'],
                'read_weight_trace': ring_trace['read_weight_trace'],
                'tap_idx_trace': ring_trace['tap_idx_trace'],
                'write_idx_trace': ring_trace['write_idx_trace'],
                'write_weight_trace': ring_trace['write_weight_trace'],
                'read_write_overlap_trace': ring_trace['read_write_overlap_trace'],
                'center_hist': ring_trace['center_hist'].tolist(),
                'read_hist': ring_trace['read_hist'].tolist(),
                'tap_hist': ring_trace['tap_hist'].tolist(),
                'write_hist': ring_trace['write_hist'].tolist(),
            }
        else:
            self._ring_trace = None

        return ring_tns, ptr_tns, hidden_tns, bb_buf, bb_keys, bb_write_ptr, bb_steps, outs_tns

    @torch.no_grad()
    def update_expert_conf(self, temperature=2.0, decay=0.95, floor=0.1):
        """Update expert write confidence from gradient magnitudes.

        Call AFTER loss.backward(), BEFORE optimizer.step().
        Uses write_proj parameter gradients as proxy for write quality.
        Small gradient magnitude = expert was close to optimal = higher confidence.
        Weights sum to N to preserve total ring write magnitude."""
        if self._write_grad_ema is None or self._expert_conf is None or self.write_proj is None:
            return
        N = self.N
        for i in range(N):
            g = self.write_proj[i].weight.grad  # type: ignore[union-attr]
            if g is None:
                continue
            mag = g.abs().mean().item()  # type: ignore[misc]  — CPU-side, OK outside forward
            w = self._expert_conf[i].item()  # CPU-side, OK outside forward
            mag = mag / max(w, 0.05)       # normalize out the weight's influence
            self._write_grad_ema[i] = decay * self._write_grad_ema[i] + (1.0 - decay) * mag
        # Normalize EMA by its mean so softmax sees relative differences,
        # not absolute magnitudes (which can be ~1e-5 for large models).
        ema_mean = self._write_grad_ema.mean().clamp(min=1e-10)
        normalized = self._write_grad_ema / ema_mean
        raw = (-normalized / temperature).softmax(dim=0)
        self._expert_conf.copy_(N * ((1.0 - floor) * raw + floor / N))

    def forward(self, x, S=None, probs=None, state=None):
        """Processes T timesteps over B sequences. Each timestep: every expert
        reads from ring, updates hidden, writes back, moves pointer.
        Output = uniform mean of all N experts, projected to token space.

        Args:
            state: optional dict {'ring': (B,M,slot_dim), 'ptr': (N,B), 'hidden': (N,B,hidden_dim)}
                   When provided, resumes from this state instead of zeros.
                   When None, starts fresh (identical to previous behavior).

        Returns:
            (output, final_state) tuple:
                output:      (B, T, vocab_size) — logits for each timestep
                final_state: dict with detached ring/ptr/hidden tensors for carry-over

        When checkpoint_chunks > 0 and gradients are enabled, the T timesteps
        are split into chunks and processed with gradient checkpointing —
        intermediate activations (ring clones) are discarded and recomputed
        during backward, trading ~50% slower backward for ~13× less VRAM."""

        # ── Resolve context scale S ──
        # S='dotprod': content-based gate (sigmoid of dot product), no scalar needed
        # S=float:     fixed scalar multiplier (backward compat / ablation)
        # S=None:      default to 'dotprod'
        if S is None:
            S = 'dotprod'

        if self.embed_mode:
            B, T = x.shape
        else:
            B, T, _ = x.shape
        M, hidden_dim, slot_dim, N = self.M, self.hidden_dim, self.slot_dim, self.N
        if probs is None:
            probs = [0.5]
        self._diag.clear()
        topk_diag_sum_dist = 0.0
        topk_diag_sum_outside = 0.0
        topk_diag_sum_entropy = 0.0
        topk_diag_sum_unique = 0.0
        topk_diag_count = 0

        # ─ state initialization ─
        # When state is provided (sequential/TBPTT mode), resume from previous
        # sequence's final state. Otherwise start fresh (zeros).
        if state is not None:
            ring_tns   = state['ring']                                     # (B, M, slot_dim)
            ptr_tns    = state['ptr']                                      # (N, B)
            hidden_tns = state['hidden']                                   # (N, B, hidden_dim)
        else:
            with _source_scope('state_init'):
                ring_tns = func_ringstart_tns(B, M, slot_dim, x.device)   # (B, M, slot_dim)
                ptr_tns = torch.zeros(N, B, device=x.device)              # (N, B)
                # staggered init: experts start evenly spaced around the ring
                for i in range(N):
                    ptr_tns[i] = (i * M // N) % M
                hidden_tns = torch.zeros(N, B, hidden_dim, device=x.device)   # (N, B, hidden_dim)

        # ── Bulletin board cache state ──
        if self.bb_enabled:
            _bb_L = self._bb_L
            _bb_head_dim = self.bb_key_proj.out_features
            if state is not None and 'bb_buf' in state:
                bb_buf       = state['bb_buf']         # (B, L, hidden_dim)
                bb_keys      = state['bb_keys']        # (B, L, head_dim)
                bb_write_ptr = state['bb_write_ptr']   # int
                bb_steps     = state['bb_steps']       # int
            else:
                bb_buf       = torch.zeros(B, _bb_L, hidden_dim, device=x.device)
                bb_keys      = torch.zeros(B, _bb_L, _bb_head_dim, device=x.device)
                bb_write_ptr = 0
                bb_steps     = 0
        else:
            bb_buf = bb_keys = None
            bb_write_ptr = bb_steps = 0

        # ── Precompute attention weights (constant during forward) ──
        R_effs = self._R_eff  # fixed from config: R_eff = R + 0.5

        # Dynamic window: only span offsets where kernel weights are non-zero.
        # vshape: weight = (1 - |offset|/R_eff).clamp(0) → zero at |offset| >= R_eff
        #   win = floor(R_eff) is the last offset with nonzero weight.
        #   R=0 → R_eff=0.5 → win=0 → 1 slot (needle, zero waste)
        #   R=1 → R_eff=1.5 → win=1 → 3 slots (exact)
        #   R=2 → R_eff=2.5 → win=2 → 5 slots (exact)
        # gaussian/uniform: tail extends further, use 2.5× + guard.
        with _source_scope('window_prepare'):
            max_R = self._max_R  # precomputed in __init__, no .item() graph break
            if self.read_kernel_mode in ('vshape', 'dotprod', 'topk'):
                win = int(math.floor(max_R))  # exact: no wasted gathers
            else:
                win = max(int(math.ceil(max_R * 2.5)) + 1, 1)
            offsets_long = torch.arange(-win, win + 1, device=x.device)
            abs_offsets  = offsets_long.float().abs()

            if self.read_kernel_mode in ('dotprod', 'topk'):
                expert_weights = None  # computed per-timestep in loop (content-based)
                # content-based read can still use local pointer write weights
                if self.write_address_mode == 'pointer':
                    _raw_w = (1.0 - abs_offsets.unsqueeze(0) / R_effs.unsqueeze(1).clamp(min=0.5)).clamp(min=0)
                    topk_write_weights = _raw_w / _raw_w.sum(dim=1, keepdim=True)  # (N, 2R+1) vshape
                else:
                    topk_write_weights = None
            elif self.read_kernel_mode == 'vshape':
                raw_w = (1.0 - abs_offsets.unsqueeze(0) / R_effs.unsqueeze(1).clamp(min=0.5)).clamp(min=0)
                expert_weights = raw_w / raw_w.sum(dim=1, keepdim=True)
                topk_write_weights = None
            elif self.read_kernel_mode == 'gaussian':
                sigma = (R_effs.unsqueeze(1) / 2.5).clamp(min=0.3)
                raw_w = torch.exp(-0.5 * (abs_offsets.unsqueeze(0) / sigma) ** 2)
                expert_weights = raw_w / raw_w.sum(dim=1, keepdim=True)
                topk_write_weights = None
            else:  # 'uniform'
                raw_w = torch.sigmoid(10.0 * (R_effs.unsqueeze(1) - abs_offsets.unsqueeze(0)))
                expert_weights = raw_w / raw_w.sum(dim=1, keepdim=True)
                topk_write_weights = None

        # ── Chunk-based processing ──
        # When checkpoint_chunks=0 or not training: C=T → single chunk, zero overhead.
        # When checkpoint_chunks>0 and training: T split into C-sized chunks,
        # each wrapped in grad_checkpoint to discard/recompute ring clone activations.
        C = self.checkpoint_chunks
        use_ckpt = C > 0 and torch.is_grad_enabled()
        if not use_ckpt:
            C = T  # single chunk = identical to original forward loop

        all_outs = []
        for t_start in range(0, T, C):
            t_end = min(t_start + C, T)
            x_chunk = x[:, t_start:t_end]

            if use_ckpt:
                # grad_checkpoint discards intermediate activations and recomputes
                # them during backward — saves ~13× VRAM on ring clones.
                ring_tns, ptr_tns, hidden_tns, bb_buf, bb_keys, bb_write_ptr, bb_steps, chunk_outs = grad_checkpoint(  # type: ignore[misc]
                    self._process_chunk,
                    x_chunk, ring_tns, ptr_tns, hidden_tns,
                    bb_buf, bb_keys, bb_write_ptr, bb_steps,
                    S, probs, offsets_long, expert_weights,
                    topk_write_weights,
                    use_reentrant=False,       # modern API, handles non-tensor args
                    preserve_rng_state=False,  # no random ops in the loop
                )
            else:
                ring_tns, ptr_tns, hidden_tns, bb_buf, bb_keys, bb_write_ptr, bb_steps, chunk_outs = self._process_chunk(
                    x_chunk, ring_tns, ptr_tns, hidden_tns,
                    bb_buf, bb_keys, bb_write_ptr, bb_steps,
                    S, probs, offsets_long, expert_weights,
                    topk_write_weights)

            all_outs.append(chunk_outs)

        output = torch.cat(all_outs, dim=1)  # (B, T, vocab_size)

        # Return final state for optional carry-over (sequential/TBPTT mode).
        # Always detached — gradients truncate at sequence boundaries.
        final_state = {
            'ring':         ring_tns.detach(),
            'ptr':          ptr_tns.detach(),
            'hidden':       hidden_tns.detach(),
        }
        if self.bb_enabled and bb_buf is not None:
            final_state['bb_buf']       = bb_buf.detach()
            final_state['bb_keys']      = bb_keys.detach()
            final_state['bb_write_ptr'] = bb_write_ptr
            final_state['bb_steps']     = bb_steps
        return output, final_state
