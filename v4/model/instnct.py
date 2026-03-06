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
_TOPK_READ_DIAG_ENABLED = False
_TOPK_READ_DIAG_KEYS = (
    'topk_mean_abs_circ_dist',
    'topk_outside_local_frac',
    'topk_attn_entropy',
    'topk_unique_slot_frac',
)


def set_source_map_enabled(enabled: bool):
    global _SOURCE_MAP_ENABLED
    _SOURCE_MAP_ENABLED = bool(enabled)


def set_topk_read_diag_enabled(enabled: bool):
    global _TOPK_READ_DIAG_ENABLED
    _TOPK_READ_DIAG_ENABLED = bool(enabled)


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
    sgn = torch.where(is_even, torch.ones_like(x), -torch.ones_like(x))
    core = C * (sgn * h + (rho * h * h))
    return torch.where(x >= l, x - l, torch.where(x <= -l, x + l, core))


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

# ─ pointer behavior ─
CNFG_CNTEXTSCL_FLT = _cfg['S']    # 0.05 how much the read signal scales into hidden state

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
    slots_int        # total ring slots (M) — needed for mod wrap on walk
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

    # soft blend: p·jump + (1-p)·walk — differentiable, no hard switching; shape (B,)
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
                 io_split_mode=CNFG_IOSPLIT_STR,                     # 'off' | 'strict' — hourglass I/O split
                 io_writer_count=CNFG_IOWRITERCNT_INT,               # experts 0..count-1 are writers
                 io_output_from_readers_only=CNFG_IOREADOUT_BOOL,    # output from readers only (strict mode)
                 gated_write=False,                        # True = erase+gate write (anti-blob), False = scatter_add (legacy)
                 write_mode='accumulate',                  # 'accumulate' (scatter_add) | 'replace' (HDD-style overwrite)
                 replace_impl=CNFG_REPLACEIMPL_STR,        # 'dense' | 'proxy_overlay' (nightly-only proxy fast path)
                 s_constraint='softplus'):                 # 'softplus' (S>0) | 'raw' (unconstrained)
        super().__init__()
        assert kernel_mode in ('uniform', 'vshape', 'gaussian', 'dotprod', 'topk'), \
            f"kernel_mode must be 'uniform', 'vshape', 'gaussian', 'dotprod', or 'topk', got '{kernel_mode}'"
        assert write_mode in ('accumulate', 'replace'), \
            f"write_mode must be 'accumulate' or 'replace', got '{write_mode}'"
        assert pointer_mode in ('sequential', 'learned', 'pilot'), \
            f"pointer_mode must be 'sequential', 'learned', or 'pilot', got '{pointer_mode}'"
        assert replace_impl in ('dense', 'proxy_overlay'), \
            f"replace_impl must be 'dense' or 'proxy_overlay', got '{replace_impl}'"
        self.write_mode = write_mode
        self.replace_impl = replace_impl

        # backward compat: embed_dim=X sets hidden_dim=slot_dim=X
        if embed_dim is not None:
            hidden_dim = embed_dim
            slot_dim = embed_dim

        self.M, self.hidden_dim, self.slot_dim, self.N, self.R = M, hidden_dim, slot_dim, N, R
        self.embed_dim = hidden_dim               # alias for backward compat (tests, prints)
        self.B = B
        self.embed_mode = embed_mode
        self.kernel_mode = kernel_mode
        self.checkpoint_chunks = checkpoint_chunks  # not nn.Parameter — excluded from state_dict
        self.expert_weighting = expert_weighting
        self._proxy_overlay_flush_interval = 16
        self._proxy_overlay_cap = 18
        self._proxy_overlay_enabled = (
            replace_impl == 'proxy_overlay'
            and N == 1
            and R == 1
            and write_mode == 'replace'
            and kernel_mode == 'vshape'
            and pointer_mode == 'sequential'
            and not bb_enabled
            and io_split_mode == 'off'
            and checkpoint_chunks == 0
        )

        # ── Diagnostics accumulator ──
        # Populated during forward(), read by train.py after each step.
        # Keys: alpha_mean/min/max (per expert), ring_norm, hidden_norm, input_norm, ptr_pos
        self._diag: dict = {}
        if expert_weighting:
            self._write_grad_ema = torch.zeros(N)     # EMA of gradient magnitudes (CPU)
            self._expert_conf = torch.ones(N)         # write weights, sum=N → each=1.0
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

        # query_proj: hidden_dim → slot_dim (content-based attention query per expert)
        # created for dotprod/topk reads only — pilot uses its own ptr_query (smaller dim)
        if kernel_mode in ('dotprod', 'topk'):
            self.query_proj = nn.ModuleList([nn.Linear(hidden_dim, slot_dim) for _ in range(N)])
        self._topk_K = topk_K  # number of ring slots for topK content-based read

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
        topk_diag_count = 0
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
                    input_vec_tns = _c19_activation(self.inp(bits), rho=_rho_from_raw(self.c19_rho_input), C=_C_from_raw(self.c19_C_input))  # Linear+c19: (B, 8) → (B, hidden_dim)
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
                    ptr_tns[i] = (ptr_tns[i] + 1 + jump) % M                          # seek to new pos

                    if torch.is_grad_enabled() and self.training:
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

                if is_writer and not is_reader:
                    # Writer-only: NO ring read, zero ring signal
                    with _source_scope('window_prepare'):
                        read_vec_tns = torch.zeros(input_vec_tns.shape[0], slot_dim, device=input_vec_tns.device)
                        expanded_idx_tns = indices_tns.unsqueeze(-1).expand(-1, -1, slot_dim)
                        if expert_weights is not None:
                            weights_tns = expert_weights[i].unsqueeze(0).expand(input_vec_tns.shape[0], -1)
                        elif topk_write_weights is not None:
                            weights_tns = topk_write_weights[i].unsqueeze(0).expand(input_vec_tns.shape[0], -1)
                        else:
                            # dotprod fallback: use uniform weights for write
                            weights_tns = torch.ones(input_vec_tns.shape[0], indices_tns.shape[1], device=input_vec_tns.device)
                            weights_tns = weights_tns / weights_tns.sum(dim=1, keepdim=True)
                elif expert_weights is not None:
                    # ─ positional kernel (vshape/gaussian/uniform) ─
                    with _source_scope('window_prepare'):
                        weights_tns = expert_weights[i].unsqueeze(0).expand(input_vec_tns.shape[0], -1)
                    with _source_scope('softread'):
                        if proxy_overlay_enabled:
                            read_vec_tns, expanded_idx_tns, current_window_tns = func_proxy_overlay_read_tns(
                                ring_tns,
                                indices_tns,
                                weights_tns,
                                slot_dim,
                                overlay_start_int,
                                overlay_vals_tns,
                                overlay_len_int,
                                M,
                            )
                        else:
                            read_vec_tns, expanded_idx_tns = func_softread_tns(
                                ring_tns, indices_tns, weights_tns, slot_dim)
                            current_window_tns = None
                elif self.kernel_mode == 'topk':
                    # ─ topK: content-based global search over entire ring ─
                    q = self.query_proj[i](hidden_lst[i])                           # (B, slot_dim)
                    scores = torch.bmm(ring_tns, q.unsqueeze(-1)).squeeze(-1)       # (B, M)
                    scores = scores * (slot_dim ** -0.5)                            # scale
                    topk_scores, topk_idx = scores.topk(self._topk_K, dim=-1)      # (B, K)
                    topk_attn = F.softmax(topk_scores, dim=-1)                      # (B, K)
                    if _TOPK_READ_DIAG_ENABLED:
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
                    # Write uses pointer-based indices (vshape), NOT topK read indices
                    with _source_scope('window_prepare'):
                        expanded_idx_tns = indices_tns.unsqueeze(-1).expand(-1, -1, slot_dim)
                        weights_tns = topk_write_weights[i].unsqueeze(0).expand(input_vec_tns.shape[0], -1)
                else:
                    # ─ dot-product attention within local window ─
                    with _source_scope('window_prepare'):
                        expanded_idx_tns = indices_tns.unsqueeze(-1).expand(-1, -1, slot_dim)
                    with _source_scope('softread'):
                        neighbors_tns = ring_tns.gather(1, expanded_idx_tns)        # (B, 2R+1, slot_dim)
                    current_window_tns = neighbors_tns
                    q = self.query_proj[i](hidden_lst[i])                       # (B, slot_dim)
                    scores = (q.unsqueeze(1) * neighbors_tns).sum(-1)           # (B, 2R+1)
                    scores = scores * (slot_dim ** -0.5)                        # scale
                    weights_tns = F.softmax(scores, dim=-1)                     # (B, 2R+1)
                    read_vec_tns = (weights_tns.unsqueeze(-1) * neighbors_tns).sum(1)  # (B, slot_dim)

                # ─ phase signal ─
                theta_tns = (ptr_tns[i] / M) * (2 * math.pi)
                phase_tns = (torch.cos(theta_tns).unsqueeze(-1) * self.phase_cos
                           + torch.sin(theta_tns).unsqueeze(-1) * self.phase_sin)

                # ─ hidden update (all hidden_dim-wide) ─
                ring_signal = self.read_proj[i](read_vec_tns)  # slot_dim → hidden_dim
                if S_flt == 'dotprod':
                    # content-based gate: cosine similarity (scale-invariant)
                    cos_sim = F.cosine_similarity(
                        input_vec_tns, ring_signal, dim=-1
                    ).unsqueeze(-1)  # (B, 1) — always [-1, +1] regardless of norms
                    alpha = torch.sigmoid(self.gate_tau * cos_sim)  # (B, 1) — bounded [0, 1]
                    blended_ring = alpha * ring_signal  # no LayerNorm — keep magnitude info

                    # ── diagnostics: track alpha + signal norms ──
                    if not torch.is_grad_enabled() or not self.training:
                        pass  # skip during eval to save compute
                    else:
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
                        if torch.is_grad_enabled() and self.training:
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

                hidden_lst[i] = _c19_activation(
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
                            write_vec = self._expert_conf[i].item() * write_vec
                    if self.write_mode == 'replace':
                        with _source_scope('write_prepare'):
                            ws = torch.sigmoid(self.write_gate[i](hidden_lst[i]))  # (B, 1)
                        with _source_scope('write_replace'):
                            if proxy_overlay_enabled:
                                w = weights_tns.unsqueeze(-1) * ws.unsqueeze(1)
                                if current_window_tns is None:
                                    current_window_tns = ring_tns.gather(1, expanded_idx_tns)
                                write_val_tns = write_vec.unsqueeze(1).expand(-1, weights_tns.size(1), -1)
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
                                ring_tns = func_hdd_write_tns(ring_tns, write_vec, expanded_idx_tns, weights_tns, write_strength=ws)
                    elif self.gated_write:
                        decisions = torch.sigmoid(self.write_head[i](hidden_lst[i]))  # (B, 2)
                        erase = decisions[:, 0]   # (B,) — per-sample erase strength
                        wgate = decisions[:, 1]   # (B,) — per-sample write gate
                        ring_tns = func_gated_write_tns(ring_tns, write_vec, expanded_idx_tns, weights_tns, erase, wgate)
                    else:
                        ring_tns = func_softwrit_tns(ring_tns, write_vec, expanded_idx_tns, weights_tns)

                # ─ POST-WRITE pointer move (sequential/learned only; pilot already moved pre-read) ─
                if self.pointer_mode == 'learned':
                    dir_logits = self.ptr_dir_head[i](hidden_lst[i])             # (B, 3)
                    a = F.softmax(dir_logits, dim=-1)                            # (B, 3): [stay, fwd, back]
                    raw_mag = self.ptr_mag_head[i](hidden_lst[i])                # (B, 1)
                    m = self._ptr_Rmax * torch.sigmoid(raw_mag.squeeze(-1))      # (B,) bounded [0, Rmax]
                    direction = a[:, 1] - a[:, 2]                                # (B,) in [-1, 1]
                    delta = (1 - a[:, 0]) * direction * m                        # (B,)
                    with _source_scope('pointer_update'):
                        ptr_tns[i] = (ptr_tns[i] + delta) % M
                elif self.pointer_mode == 'pilot':
                    pass  # already moved pre-read (seek-then-read)
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
                if torch.is_grad_enabled() and self.training:
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
        if torch.is_grad_enabled() and self.training:
            d = self._diag
            d['ring_norm'] = ring_tns.detach().norm().item()
            d['ring_slot_mean'] = ring_tns.detach().norm(dim=-1).mean().item()
            for i in range(self.N):
                d[f'hidden_final_norm_{i}'] = hidden_tns[i].detach().norm(dim=-1).mean().item()
        if _TOPK_READ_DIAG_ENABLED and topk_diag_count > 0:
            self._diag['topk_mean_abs_circ_dist'] = topk_diag_sum_dist / topk_diag_count
            self._diag['topk_outside_local_frac'] = topk_diag_sum_outside / topk_diag_count
            self._diag['topk_attn_entropy'] = topk_diag_sum_entropy / topk_diag_count
            self._diag['topk_unique_slot_frac'] = topk_diag_sum_unique / topk_diag_count

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
            mag = g.abs().mean().item()  # type: ignore[misc]
            w = self._expert_conf[i].item()
            mag = mag / max(w, 0.05)       # normalize out the weight's influence
            self._write_grad_ema[i] = decay * self._write_grad_ema[i] + (1.0 - decay) * mag
        # Normalize EMA by its mean so softmax sees relative differences,
        # not absolute magnitudes (which can be ~1e-5 for large models).
        ema_mean = self._write_grad_ema.mean().clamp(min=1e-10)
        normalized = self._write_grad_ema / ema_mean
        raw = (-normalized / temperature).softmax(dim=0)
        self._expert_conf = N * ((1.0 - floor) * raw + floor / N)

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
        for key in _TOPK_READ_DIAG_KEYS:
            self._diag.pop(key, None)
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
            max_R = R_effs.max().item()
            if self.kernel_mode in ('vshape', 'dotprod', 'topk'):
                win = int(math.floor(max_R))  # exact: no wasted gathers
            else:
                win = max(int(math.ceil(max_R * 2.5)) + 1, 1)
            offsets_long = torch.arange(-win, win + 1, device=x.device)
            abs_offsets  = offsets_long.float().abs()

            if self.kernel_mode in ('dotprod', 'topk'):
                expert_weights = None  # computed per-timestep in loop (content-based)
                # topK still needs positional weights for WRITING (scatter_add to pointer window)
                if self.kernel_mode == 'topk':
                    _raw_w = (1.0 - abs_offsets.unsqueeze(0) / R_effs.unsqueeze(1).clamp(min=0.5)).clamp(min=0)
                    topk_write_weights = _raw_w / _raw_w.sum(dim=1, keepdim=True)  # (N, 2R+1) vshape
                else:
                    topk_write_weights = None
            elif self.kernel_mode == 'vshape':
                raw_w = (1.0 - abs_offsets.unsqueeze(0) / R_effs.unsqueeze(1).clamp(min=0.5)).clamp(min=0)
                expert_weights = raw_w / raw_w.sum(dim=1, keepdim=True)
                topk_write_weights = None
            elif self.kernel_mode == 'gaussian':
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
