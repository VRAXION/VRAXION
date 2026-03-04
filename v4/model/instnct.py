import math                # sqrt for golden ratio constant
import torch               # tensor operations and autograd
from torch.utils.checkpoint import checkpoint as grad_checkpoint  # memory-efficient backward
import torch.nn as nn      # neural network layers (Linear, Module, ModuleList)
import torch.nn.functional as F  # softmax for dot-product attention kernel
import yaml                # YAML config parser
from pathlib import Path   # cross-platform file path resolution

# ── Constants ───────────────────────────────────────────────────
# inverse golden ratio — used in phi_destinations() to compute maximally-spaced pointer jumps
PHI_INV = (math.sqrt(5) - 1) / 2  # 1/φ ≈ 0.6180339887
PHI = (1 + math.sqrt(5)) / 2      # φ ≈ 1.6180339887 — golden ratio

# C19 period constant
_C19_C = math.pi


def _c19_activation(x, rho=4.0, C=None):
    """C19 periodic parabolic wave activation with dual-phi interference filter.

    Four key properties for recurrent stability:
    1. Bounded core (|x| < 6C): prevents hidden state explosion
    2. Linear tails (|x| > 6C): prevents gradient vanishing
    3. Periodic parabolic arches: natural hashing of input magnitudes
    4. Dual-phi asymmetry: neg×φ, pos×(1/φ) — two anti-resonance filters
       that prevent gradient standing waves in recurrent loops

    rho is fixed at 4.0 — the critical point where negative arches just
    touch zero at their midpoint. Combined with dual-phi, this gives:
    - Positive arches (even n): peak 0.5C, scaled by 1/φ → 0.309C
    - Negative arches (odd n): min -0.0625C, scaled by φ → -0.101C
    - Three-level anti-resonance: φ, 1/φ, and φ² ratio between them

    The dual-phi gain is folded into arch parity (odd=φ, even=1/φ)
    instead of testing core<0 — saves one comparison + one where.
    """
    if C is None:
        C = _C19_C
    l = 6.0 * C
    inv_c = 1.0 / C
    scaled = x * inv_c
    n = torch.floor(scaled)
    t = scaled - n                                      # [0, 1) fractional position
    h = t - t * t                                       # parabola: t*(1-t), peak 0.25 at t=0.5
    odd = torch.remainder(n, 2.0)                       # 0=even (pos arch), 1=odd (neg arch)
    sgn = 1.0 - 2.0 * odd                              # ±1 alternating sign
    gain = odd * (PHI - PHI_INV) + PHI_INV              # odd→φ, even→φ⁻¹ (branchless)
    core = C * h * (sgn + rho * h) * gain               # fused: arch × dual-phi
    return torch.where(x.abs() > l, x - x.sign() * l, core)


_C19_C_MIN = 1.0
_C19_C_MAX = 50.0

def _sigmoid_bounded(raw, lo, hi):
    """Sigmoid-bounded parameter: raw float → [lo, hi]."""
    return lo + (hi - lo) * torch.sigmoid(raw)

def _C_from_raw(raw_C):
    """Sigmoid-bounded C: raw float → [1.0, 50.0]."""
    return _sigmoid_bounded(raw_C, _C19_C_MIN, _C19_C_MAX)

def _init_raw(val, lo, hi):
    """Inverse sigmoid so that sigmoid(raw) maps to val within [lo, hi]."""
    p = (val - lo) / (hi - lo)
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))

def _C_init_raw(C_init=None):
    """Inverse sigmoid so that sigmoid(raw) maps to C_init (default π)."""
    if C_init is None:
        C_init = math.pi
    return _init_raw(C_init, _C19_C_MIN, _C19_C_MAX)


class _C19Module(nn.Module):
    """Wraps c19 as nn.Module for use in nn.Sequential (fixed rho=4.0)."""
    def forward(self, x):
        return _c19_activation(x)


class BatchedLinear(nn.Module):
    """N independent Linear layers fused into a single bmm call.

    Replaces nn.ModuleList([nn.Linear(in, out) for _ in range(N)])
    with a single (N, out, in) weight + (N, out) bias — one torch.bmm
    instead of N sequential matmuls.

    Input:  (N, B, in_features)
    Output: (N, B, out_features)
    """
    def __init__(self, N, in_features, out_features):
        super().__init__()
        self.N = N
        self.in_features = in_features
        self.out_features = out_features
        # Same init as nn.Linear (Kaiming uniform)
        self.weight = nn.Parameter(torch.empty(N, out_features, in_features))
        self.bias = nn.Parameter(torch.empty(N, out_features))
        self._reset_parameters()

    def _reset_parameters(self):
        for i in range(self.N):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, x):
        """x: (N, B, in_features) → (N, B, out_features)"""
        # bmm: (N, out, in) @ (N, in, B)^T → (N, B, out)
        return torch.bmm(x, self.weight.transpose(1, 2)) + self.bias.unsqueeze(1)

    @staticmethod
    def from_module_list(module_list):
        """Convert existing nn.ModuleList of nn.Linear to BatchedLinear.

        Copies weights/biases from individual Linear layers into the
        fused (N, out, in) format. For checkpoint migration."""
        N = len(module_list)
        in_f = module_list[0].in_features
        out_f = module_list[0].out_features
        bl = BatchedLinear(N, in_f, out_f)
        with torch.no_grad():
            for i, lin in enumerate(module_list):
                bl.weight[i].copy_(lin.weight)
                bl.bias[i].copy_(lin.bias)
        return bl


class _C19LearnableModule(nn.Module):
    """C19 with per-neuron learnable C (rho fixed at 4.0), bounded via sigmoid."""
    def __init__(self, width, learn_C=True):
        super().__init__()
        self.learn_C = learn_C
        if learn_C:
            self.raw_C = nn.Parameter(torch.full((width,), _C_init_raw()))

    def forward(self, x):
        C = _C_from_raw(self.raw_C) if self.learn_C else None
        return _c19_activation(x, C=C)

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

# ─ circuit fixes (gradient flow improvements) ─
CNFG_RINGDECAY_BOOL = _cfg.get('ring_decay', False)             # exponential ring forgetting
CNFG_GATEBIAS_BOOL = _cfg.get('gate_bias', False)               # learnable bias on cosine gate
CNFG_EXPERTOUTW_BOOL = _cfg.get('expert_output_weights', False) # learned expert output weighting
CNFG_PTRGRADIENT_BOOL = _cfg.get('ptr_gradient', False)         # fractional pointer interpolation

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

# · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
def func_additive_write_tns(
    ring_tns,            # the shared ring buffer; shape (B, M, D)
    write_vec_tns,       # content to write; shape (B, D)
    expanded_idx_tns,    # expanded window indices; shape (B, 2R+1, D)
    weights_tns,         # window weights; shape (B, 2R+1)
    write_strength=None, # per-sample write gate; shape (B, 1) or None
):
    """Pure additive write: ring_new = ring_old + σ(gate) * w * write_vec.

    Perfect skip-connection: ∂ring_new/∂ring_old = 1.0 (identity).
    The read_proj linear layer handles arbitrary ring magnitude."""

    w = weights_tns.unsqueeze(-1)  # (B, 2R+1, 1)
    if write_strength is not None:
        w = w * write_strength.unsqueeze(1)  # modulate by learned gate

    # Broadcast write_vec to all window positions
    write_val = write_vec_tns.unsqueeze(1).expand(-1, weights_tns.size(1), -1)  # (B, 2R+1, D)

    # Pure additive: old content preserved + gated new content
    current = ring_tns.gather(1, expanded_idx_tns)  # (B, 2R+1, D)
    updated = current + w * write_val                # (B, 2R+1, D)

    ring_new = ring_tns.clone()
    ring_new.scatter_(1, expanded_idx_tns, updated)
    return ring_new

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
                 s_constraint='softplus',                  # 'softplus' (S>0) | 'raw' (unconstrained)
                 parallel_experts=False,                   # False/True/'vectorized' — expert execution mode
                 ring_decay=CNFG_RINGDECAY_BOOL,          # exponential ring forgetting (prevents accumulation blob)
                 gate_bias=CNFG_GATEBIAS_BOOL,            # learnable bias on cosine gate (allows full close/open)
                 expert_output_weights=CNFG_EXPERTOUTW_BOOL,  # learned softmax weights for expert output mix
                 ptr_gradient=CNFG_PTRGRADIENT_BOOL):     # fractional pointer interpolation (gradient through read)
        super().__init__()
        assert kernel_mode in ('uniform', 'vshape', 'gaussian', 'dotprod', 'topk'), \
            f"kernel_mode must be 'uniform', 'vshape', 'gaussian', 'dotprod', or 'topk', got '{kernel_mode}'"
        assert write_mode in ('accumulate', 'replace', 'additive'), \
            f"write_mode must be 'accumulate', 'replace', or 'additive', got '{write_mode}'"
        assert pointer_mode in ('sequential', 'learned', 'pilot'), \
            f"pointer_mode must be 'sequential', 'learned', or 'pilot', got '{pointer_mode}'"
        self.write_mode = write_mode

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

        # ── Per-neuron learnable C for c19 activations (rho fixed at 4.0) ──
        _raw0_C = _C_init_raw()
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
        self.parallel_experts = parallel_experts  # parallel read/write mode
        self.S_raw = nn.Parameter(torch.tensor(1.0))  # full ring signal — let backprop decide
        self.gate_tau = nn.Parameter(torch.tensor(4.0))  # learnable temperature for cosine gate
        self.ring_signal_norm = nn.LayerNorm(hidden_dim)  # normalizes ring signal before blend

        # ── Ring read norm (additive write mode) ──
        # When write_mode='additive', ring slot magnitudes grow unbounded (no erasure).
        # LayerNorm on the raw ring read stabilizes the signal before read_proj.
        # Always created for checkpoint compat; only used in additive mode.
        self.ring_read_norm = nn.LayerNorm(slot_dim)

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

        # ── Circuit fix: Ring decay (exponential forgetting) ──
        # Prevents unbounded ring magnitude in accumulate/additive write modes.
        # ring_tns *= sigmoid(raw_decay) each timestep. sigmoid(4.6) ≈ 0.99.
        # Default off: decay=1.0 (identity). When enabled, starts at ~0.99.
        self.ring_decay_enabled = ring_decay
        if ring_decay:
            self._raw_ring_decay = nn.Parameter(torch.tensor(4.6))  # sigmoid(4.6) ≈ 0.99

        # ── Circuit fix: Gate bias (allows full close/open) ──
        # alpha = sigmoid(tau * cos_sim + bias). Default bias=0 → same as before.
        self.gate_bias_enabled = gate_bias
        if gate_bias:
            self.gate_bias = nn.Parameter(torch.tensor(0.0))

        # ── Circuit fix: Expert output weighting ──
        # mean_hidden = softmax(logits) · stack(hidden). Default logits=0 → uniform.
        self.expert_output_weights_enabled = expert_output_weights
        if expert_output_weights:
            self.expert_output_logits = nn.Parameter(torch.zeros(N))

        # ── Circuit fix: Pointer gradient (fractional interpolation) ──
        # Makes kernel weights differentiable w.r.t. pointer position.
        # Only active for learned/pilot modes (sequential is always integer).
        self.ptr_gradient = ptr_gradient

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

        if self._bitlift:
            _bit_shifts = torch.arange(7, -1, -1, device=x_chunk.device)  # [7,6,5,4,3,2,1,0]

        for t in range(C):
            if self.inp is not None:
                if self._bitlift:
                    # byte index → 8 float bits, GPU-native
                    byte_val = x_chunk[:, t]                               # (B,) int64
                    bits = ((byte_val.unsqueeze(-1) >> _bit_shifts) & 1).float()  # (B, 8)
                    input_vec_tns = _c19_activation(self.inp(bits), C=_C_from_raw(self.c19_C_input))  # Linear+c19: (B, 8) → (B, hidden_dim)
                else:
                    input_vec_tns = self.inp(x_chunk[:, t])                # Embedding: int64 → (B, hidden_dim)
            else:
                input_vec_tns = self._fixed_table[x_chunk[:, t]]           # (B, hidden_dim)

            # ── Circuit fix: ring decay (per-timestep exponential forgetting) ──
            if self.ring_decay_enabled:
                ring_tns = ring_tns * torch.sigmoid(self._raw_ring_decay)

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

                center = ptr_tns[i].long().clamp(0, M - 1)
                indices_tns = (center.unsqueeze(1) + offsets_long) % M

                # ── Hourglass I/O split: role flags ──
                is_writer = bool(self._writer_mask[i]) if self.io_split_mode == 'strict' else False
                is_reader = bool(self._reader_mask[i]) if self.io_split_mode == 'strict' else True

                if is_writer and not is_reader:
                    # Writer-only: NO ring read, zero ring signal
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
                    if self.ptr_gradient and self.pointer_mode != 'sequential':
                        # Circuit fix: fractional pointer interpolation
                        # Recompute kernel weights using differentiable fractional offset
                        ptr_float = ptr_tns[i] % M                                    # (B,) differentiable
                        frac = ptr_float - ptr_float.detach().floor()                  # (B,) has gradient
                        adj_offsets = offsets_long.float().unsqueeze(0) - frac.unsqueeze(1)  # (B, 2R+1)
                        R_eff_i = self._R_eff[i]
                        if self.kernel_mode == 'vshape':
                            raw_w = (1.0 - adj_offsets.abs() / R_eff_i.clamp(min=0.5)).clamp(min=0)
                        elif self.kernel_mode == 'gaussian':
                            sigma = (R_eff_i / 2.5).clamp(min=0.3)
                            raw_w = torch.exp(-0.5 * (adj_offsets / sigma) ** 2)
                        else:  # uniform
                            raw_w = torch.sigmoid(10.0 * (R_eff_i - adj_offsets.abs()))
                        weights_tns = raw_w / raw_w.sum(dim=1, keepdim=True)           # (B, 2R+1) normalized
                    else:
                        weights_tns = expert_weights[i].unsqueeze(0).expand(input_vec_tns.shape[0], -1)
                    read_vec_tns, expanded_idx_tns = func_softread_tns(
                        ring_tns, indices_tns, weights_tns, slot_dim)
                elif self.kernel_mode == 'topk':
                    # ─ topK: content-based global search over entire ring ─
                    q = self.query_proj[i](hidden_lst[i])                           # (B, slot_dim)
                    scores = torch.bmm(ring_tns, q.unsqueeze(-1)).squeeze(-1)       # (B, M)
                    scores = scores * (slot_dim ** -0.5)                            # scale
                    topk_scores, topk_idx = scores.topk(self._topk_K, dim=-1)      # (B, K)
                    topk_attn = F.softmax(topk_scores, dim=-1)                      # (B, K)
                    topk_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, slot_dim) # (B, K, slot_dim)
                    topk_neighbors = ring_tns.gather(1, topk_expanded)              # (B, K, slot_dim)
                    read_vec_tns = (topk_attn.unsqueeze(-1) * topk_neighbors).sum(1)  # (B, slot_dim)
                    # Write uses pointer-based indices (vshape), NOT topK read indices
                    expanded_idx_tns = indices_tns.unsqueeze(-1).expand(-1, -1, slot_dim)
                    weights_tns = topk_write_weights[i].unsqueeze(0).expand(input_vec_tns.shape[0], -1)
                else:
                    # ─ dot-product attention within local window ─
                    expanded_idx_tns = indices_tns.unsqueeze(-1).expand(-1, -1, slot_dim)
                    neighbors_tns = ring_tns.gather(1, expanded_idx_tns)        # (B, 2R+1, slot_dim)
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
                    gate_input = self.gate_tau * cos_sim
                    if self.gate_bias_enabled:
                        gate_input = gate_input + self.gate_bias
                    alpha = torch.sigmoid(gate_input)  # (B, 1) — bounded [0, 1]
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
                    C=_C_from_raw(self.c19_C_hidden),
                )

                # ─ soft_write (compress hidden → slot_dim if needed) ─
                # Hourglass strict: readers don't write
                if not (self.io_split_mode == 'strict' and is_reader and not is_writer):
                    if self.write_proj is not None:
                        write_vec = self.write_proj[i](hidden_lst[i])  # hidden_dim → slot_dim
                    else:
                        write_vec = hidden_lst[i]                      # identity when dims match
                    if self._expert_conf is not None:
                        write_vec = self._expert_conf[i].item() * write_vec
                    if self.write_mode == 'additive':
                        ws = torch.sigmoid(self.write_gate[i](hidden_lst[i]))  # (B, 1)
                        ring_tns = func_additive_write_tns(ring_tns, write_vec, expanded_idx_tns, weights_tns, write_strength=ws)
                    elif self.write_mode == 'replace':
                        ws = torch.sigmoid(self.write_gate[i](hidden_lst[i]))  # (B, 1)
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
                    ptr_tns[i] = (ptr_tns[i] + delta) % M
                elif self.pointer_mode == 'pilot':
                    pass  # already moved pre-read (seek-then-read)
                else:
                    ptr_tns[i] = (ptr_tns[i] + 1) % M

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
            if self.io_split_mode == 'strict' and self.io_output_from_readers_only:
                reader_idx = self._reader_mask.nonzero(as_tuple=False).flatten().tolist()
                stacked = torch.stack([hidden_lst[j] for j in reader_idx])  # (K, B, H)
            else:
                stacked = torch.stack(hidden_lst)                           # (N, B, H)
            if self.expert_output_weights_enabled:
                # Circuit fix: learned softmax weights over experts
                w = F.softmax(self.expert_output_logits, dim=0)             # (N,)
                mean_hidden = (stacked * w[:stacked.shape[0]].view(-1, 1, 1)).sum(0)
            else:
                mean_hidden = stacked.mean(0)                               # (B, hidden_dim)
            if self._bitlift_out:
                bit_scores = torch.tanh(self.out(mean_hidden))   # (B, 8)
                outs_lst.append(bit_scores @ self._bit_patterns.T)  # (B, 8) @ (8, 256) → (B, 256)
            elif self.out is not None:
                outs_lst.append(self.out(mean_hidden))           # nn.Linear
            else:
                outs_lst.append(mean_hidden @ self._fixed_output_table.T)  # (B, 256)

        # Repack hidden states and stack outputs
        hidden_tns = torch.stack(hidden_lst)       # (N, B, hidden_dim)
        outs_tns = torch.stack(outs_lst, dim=1)    # (B, C, vocab_size)

        # ── diagnostics: ring + final hidden norms ──
        if torch.is_grad_enabled() and self.training:
            d = self._diag
            d['ring_norm'] = ring_tns.detach().norm().item()
            d['ring_slot_mean'] = ring_tns.detach().norm(dim=-1).mean().item()
            for i in range(self.N):
                d[f'hidden_final_norm_{i}'] = hidden_tns[i].detach().norm(dim=-1).mean().item()

        return ring_tns, ptr_tns, hidden_tns, bb_buf, bb_keys, bb_write_ptr, bb_steps, outs_tns

    def _process_chunk_parallel(self, x_chunk, ring_tns, ptr_tns, hidden_tns,
                                bb_buf, bb_keys, bb_write_ptr, bb_steps,
                                S_flt, probs_lst, offsets_long, expert_weights,
                                topk_write_weights=None):
        """Parallel-expert variant of _process_chunk.

        Key difference from sequential version:
        - All N experts read from the SAME ring state (no intra-timestep crosstalk)
        - All N writes are coalesced into a SINGLE ring update per timestep
        - 1 ring.clone() per timestep instead of N → 6× less memory copy for N=6

        Trade-off: experts only see each other's writes at t+1 (not within t).
        With staggered pointers (expert i starts at i*M/N ≈ 10 slots apart),
        intra-timestep overlap is rare, so this is nearly equivalent.
        """
        M, slot_dim, N, hidden_dim = self.M, self.slot_dim, self.N, self.hidden_dim

        ptr_tns = ptr_tns.clone()
        hidden_lst = [hidden_tns[i] for i in range(N)]

        C = x_chunk.shape[1]
        outs_lst = []

        if self._bitlift:
            _bit_shifts = torch.arange(7, -1, -1, device=x_chunk.device)

        for t in range(C):
            # ── Embed input (same as sequential) ──
            if self.inp is not None:
                if self._bitlift:
                    byte_val = x_chunk[:, t]
                    bits = ((byte_val.unsqueeze(-1) >> _bit_shifts) & 1).float()
                    input_vec_tns = _c19_activation(self.inp(bits), C=_C_from_raw(self.c19_C_input))
                else:
                    input_vec_tns = self.inp(x_chunk[:, t])
            else:
                input_vec_tns = self._fixed_table[x_chunk[:, t]]

            B_size = input_vec_tns.shape[0]

            # ── Circuit fix: ring decay (per-timestep exponential forgetting) ──
            if self.ring_decay_enabled:
                ring_tns = ring_tns * torch.sigmoid(self._raw_ring_decay)

            # ════════════════════════════════════════════════════
            # PHASE 1: All reads from the SAME ring state
            # ════════════════════════════════════════════════════
            read_vecs = []           # (B, slot_dim) per expert
            expanded_idxs = []       # (B, 2R+1, slot_dim) per expert
            all_weights = []         # (B, 2R+1) per expert
            write_flags = []         # bool per expert — should this expert write?

            for i in range(N):
                # ─ PRE-READ pointer seek (pilot mode) ─
                if self.pointer_mode == 'pilot':
                    current_slot = ptr_tns[i].long().clamp(0, M - 1)
                    slot_id = F.normalize(self.slot_identity[current_slot], dim=-1)
                    query = F.normalize(self.ptr_query[i](hidden_lst[i]), dim=-1)
                    sim = (query * slot_id).sum(-1)
                    tau = F.softplus(self._ptr_tau)
                    jump = self._ptr_max_jump * torch.sigmoid(-sim * tau)
                    ptr_tns[i] = (ptr_tns[i] + 1 + jump) % M

                center = ptr_tns[i].long().clamp(0, M - 1)
                indices_tns = (center.unsqueeze(1) + offsets_long) % M

                # ── Hourglass I/O split ──
                is_writer = bool(self._writer_mask[i]) if self.io_split_mode == 'strict' else False
                is_reader = bool(self._reader_mask[i]) if self.io_split_mode == 'strict' else True
                should_write = not (self.io_split_mode == 'strict' and is_reader and not is_writer)
                write_flags.append(should_write)

                if is_writer and not is_reader:
                    read_vec_tns = torch.zeros(B_size, slot_dim, device=input_vec_tns.device)
                    expanded_idx_tns = indices_tns.unsqueeze(-1).expand(-1, -1, slot_dim)
                    if expert_weights is not None:
                        weights_tns = expert_weights[i].unsqueeze(0).expand(B_size, -1)
                    elif topk_write_weights is not None:
                        weights_tns = topk_write_weights[i].unsqueeze(0).expand(B_size, -1)
                    else:
                        weights_tns = torch.ones(B_size, indices_tns.shape[1], device=input_vec_tns.device)
                        weights_tns = weights_tns / weights_tns.sum(dim=1, keepdim=True)
                elif expert_weights is not None:
                    if self.ptr_gradient and self.pointer_mode != 'sequential':
                        ptr_float = ptr_tns[i] % M
                        frac = ptr_float - ptr_float.detach().floor()
                        adj_offsets = offsets_long.float().unsqueeze(0) - frac.unsqueeze(1)
                        R_eff_i = self._R_eff[i]
                        if self.kernel_mode == 'vshape':
                            raw_w = (1.0 - adj_offsets.abs() / R_eff_i.clamp(min=0.5)).clamp(min=0)
                        elif self.kernel_mode == 'gaussian':
                            sigma = (R_eff_i / 2.5).clamp(min=0.3)
                            raw_w = torch.exp(-0.5 * (adj_offsets / sigma) ** 2)
                        else:
                            raw_w = torch.sigmoid(10.0 * (R_eff_i - adj_offsets.abs()))
                        weights_tns = raw_w / raw_w.sum(dim=1, keepdim=True)
                    else:
                        weights_tns = expert_weights[i].unsqueeze(0).expand(B_size, -1)
                    read_vec_tns, expanded_idx_tns = func_softread_tns(
                        ring_tns, indices_tns, weights_tns, slot_dim)
                elif self.kernel_mode == 'topk':
                    q = self.query_proj[i](hidden_lst[i])
                    scores = torch.bmm(ring_tns, q.unsqueeze(-1)).squeeze(-1)
                    scores = scores * (slot_dim ** -0.5)
                    topk_scores, topk_idx = scores.topk(self._topk_K, dim=-1)
                    topk_attn = F.softmax(topk_scores, dim=-1)
                    topk_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, slot_dim)
                    topk_neighbors = ring_tns.gather(1, topk_expanded)
                    read_vec_tns = (topk_attn.unsqueeze(-1) * topk_neighbors).sum(1)
                    expanded_idx_tns = indices_tns.unsqueeze(-1).expand(-1, -1, slot_dim)
                    weights_tns = topk_write_weights[i].unsqueeze(0).expand(B_size, -1)
                else:
                    expanded_idx_tns = indices_tns.unsqueeze(-1).expand(-1, -1, slot_dim)
                    neighbors_tns = ring_tns.gather(1, expanded_idx_tns)
                    q = self.query_proj[i](hidden_lst[i])
                    scores = (q.unsqueeze(1) * neighbors_tns).sum(-1)
                    scores = scores * (slot_dim ** -0.5)
                    weights_tns = F.softmax(scores, dim=-1)
                    read_vec_tns = (weights_tns.unsqueeze(-1) * neighbors_tns).sum(1)

                read_vecs.append(read_vec_tns)
                expanded_idxs.append(expanded_idx_tns)
                all_weights.append(weights_tns)

            # ════════════════════════════════════════════════════
            # PHASE 2: All hidden updates (independent — no ring mutation)
            # ════════════════════════════════════════════════════
            write_vecs = []          # (B, slot_dim) per expert
            write_strengths = []     # (B, 1) per expert (for HDD write)
            write_erases = []        # (B,) per expert (for gated write)
            write_gates = []         # (B,) per expert (for gated write)

            for i in range(N):
                # ─ phase signal ─
                theta_tns = (ptr_tns[i] / M) * (2 * math.pi)
                phase_tns = (torch.cos(theta_tns).unsqueeze(-1) * self.phase_cos
                           + torch.sin(theta_tns).unsqueeze(-1) * self.phase_sin)

                # ─ ring signal blend ─
                ring_signal = self.read_proj[i](read_vecs[i])
                if S_flt == 'dotprod':
                    cos_sim = F.cosine_similarity(
                        input_vec_tns, ring_signal, dim=-1
                    ).unsqueeze(-1)
                    gate_input = self.gate_tau * cos_sim
                    if self.gate_bias_enabled:
                        gate_input = gate_input + self.gate_bias
                    alpha = torch.sigmoid(gate_input)
                    blended_ring = alpha * ring_signal
                else:
                    blended_ring = S_flt * ring_signal

                # ─ bulletin board cache read ─
                bb_ctx = 0
                if self.bb_enabled and bb_buf is not None:
                    taps = self._bb_taps
                    if bb_steps >= taps[0]:
                        tap_idx = [(bb_write_ptr - tap) % self._bb_L for tap in taps]
                        cached_vecs = bb_buf[:, tap_idx]
                        cached_keys = bb_keys[:, tap_idx]
                        q = self.bb_query_proj[i](hidden_lst[i])
                        q_norm = F.normalize(q, dim=-1)
                        k_norm = F.normalize(cached_keys, dim=-1)
                        scores = (q_norm.unsqueeze(1) * k_norm).sum(-1)
                        scores = scores * self._bb_tau
                        for k_idx, tap in enumerate(taps):
                            if bb_steps < tap:
                                scores[:, k_idx] = float('-inf')
                        weights_bb = F.softmax(scores, dim=-1)
                        ctx = (weights_bb.unsqueeze(-1) * cached_vecs).sum(1)
                        if self.bb_gate is not None:
                            beta = torch.sigmoid(self.bb_gate[i](hidden_lst[i]))
                            bb_ctx = self._bb_scale * beta * ctx
                        else:
                            bb_ctx = self._bb_scale * ctx

                # ─ hidden update ─
                hidden_lst[i] = _c19_activation(
                    input_vec_tns + blended_ring + bb_ctx + phase_tns + hidden_lst[i],
                    C=_C_from_raw(self.c19_C_hidden),
                )

                # ─ compute write vectors (but don't apply yet) ─
                if write_flags[i]:
                    if self.write_proj is not None:
                        wv = self.write_proj[i](hidden_lst[i])
                    else:
                        wv = hidden_lst[i]
                    if self._expert_conf is not None:
                        wv = self._expert_conf[i].item() * wv
                    write_vecs.append(wv)

                    if self.write_mode in ('replace', 'additive'):
                        ws = torch.sigmoid(self.write_gate[i](hidden_lst[i]))
                        write_strengths.append(ws)
                    elif self.gated_write:
                        decisions = torch.sigmoid(self.write_head[i](hidden_lst[i]))
                        write_erases.append(decisions[:, 0])
                        write_gates.append(decisions[:, 1])
                else:
                    write_vecs.append(None)
                    write_strengths.append(None)

                # ─ POST-WRITE pointer move ─
                if self.pointer_mode == 'learned':
                    dir_logits = self.ptr_dir_head[i](hidden_lst[i])
                    a = F.softmax(dir_logits, dim=-1)
                    raw_mag = self.ptr_mag_head[i](hidden_lst[i])
                    m = self._ptr_Rmax * torch.sigmoid(raw_mag.squeeze(-1))
                    direction = a[:, 1] - a[:, 2]
                    delta = (1 - a[:, 0]) * direction * m
                    ptr_tns[i] = (ptr_tns[i] + delta) % M
                elif self.pointer_mode == 'pilot':
                    pass
                else:
                    ptr_tns[i] = (ptr_tns[i] + 1) % M

            # ════════════════════════════════════════════════════
            # PHASE 3: Single coalesced ring write (1 clone, not N)
            # ════════════════════════════════════════════════════
            if self.write_mode == 'additive':
                # Additive: ring_old + gate*w*write_vec (gradient=1.0 on old path)
                ring_new = ring_tns.clone()
                wr_idx = 0
                for i in range(N):
                    if write_vecs[i] is not None:
                        w = all_weights[i].unsqueeze(-1)
                        ws = write_strengths[wr_idx]
                        wr_idx += 1
                        if ws is not None:
                            w = w * ws.unsqueeze(1)
                        current = ring_new.gather(1, expanded_idxs[i])
                        write_val = write_vecs[i].unsqueeze(1).expand(-1, all_weights[i].size(1), -1)
                        updated = current + w * write_val
                        ring_new.scatter_(1, expanded_idxs[i], updated)
                ring_tns = ring_new
            elif self.write_mode == 'replace':
                # HDD-style: apply all expert writes to a single ring clone
                ring_new = ring_tns.clone()
                wr_idx = 0
                for i in range(N):
                    if write_vecs[i] is not None:
                        w = all_weights[i].unsqueeze(-1)
                        ws = write_strengths[wr_idx]
                        wr_idx += 1
                        if ws is not None:
                            w = w * ws.unsqueeze(1)
                        current = ring_new.gather(1, expanded_idxs[i])
                        write_val = write_vecs[i].unsqueeze(1).expand(-1, all_weights[i].size(1), -1)
                        updated = w * write_val + (1 - w) * current
                        ring_new.scatter_(1, expanded_idxs[i], updated)
                ring_tns = ring_new
            elif self.gated_write:
                ring_new = ring_tns.clone()
                er_idx = 0
                for i in range(N):
                    if write_vecs[i] is not None:
                        w = all_weights[i].unsqueeze(-1)
                        erase_b = write_erases[er_idx].unsqueeze(-1).unsqueeze(-1)
                        wgate_b = write_gates[er_idx].unsqueeze(-1).unsqueeze(-1)
                        er_idx += 1
                        current = ring_new.gather(1, expanded_idxs[i])
                        erased = current * (1 - erase_b * w)
                        write_val = write_vecs[i].unsqueeze(1).expand(-1, all_weights[i].size(1), -1)
                        updated = erased + wgate_b * w * write_val
                        ring_new.scatter_(1, expanded_idxs[i], updated)
                ring_tns = ring_new
            else:
                # scatter_add: accumulate all writes into one clone
                ring_new = ring_tns.clone()
                for i in range(N):
                    if write_vecs[i] is not None:
                        write_val = write_vecs[i].unsqueeze(1).expand(-1, all_weights[i].size(1), -1)
                        ring_new.scatter_add_(1, expanded_idxs[i], all_weights[i].unsqueeze(-1) * write_val)
                ring_tns = ring_new

            # ── Bulletin board write ──
            if self.bb_enabled and bb_buf is not None:
                bb_buf = bb_buf.clone()
                bb_buf[:, bb_write_ptr] = input_vec_tns
                bb_keys = bb_keys.clone()
                written_key = self.bb_key_proj(input_vec_tns)
                bb_keys[:, bb_write_ptr] = written_key
                bb_write_ptr = (bb_write_ptr + 1) % self._bb_L
                bb_steps += 1

            # ── Output ──
            if self.io_split_mode == 'strict' and self.io_output_from_readers_only:
                reader_idx = self._reader_mask.nonzero(as_tuple=False).flatten().tolist()
                stacked = torch.stack([hidden_lst[j] for j in reader_idx])
            else:
                stacked = torch.stack(hidden_lst)
            if self.expert_output_weights_enabled:
                w = F.softmax(self.expert_output_logits, dim=0)
                mean_hidden = (stacked * w[:stacked.shape[0]].view(-1, 1, 1)).sum(0)
            else:
                mean_hidden = stacked.mean(0)
            if self._bitlift_out:
                bit_scores = torch.tanh(self.out(mean_hidden))
                outs_lst.append(bit_scores @ self._bit_patterns.T)
            elif self.out is not None:
                outs_lst.append(self.out(mean_hidden))
            else:
                outs_lst.append(mean_hidden @ self._fixed_output_table.T)

        hidden_tns = torch.stack(hidden_lst)
        outs_tns = torch.stack(outs_lst, dim=1)

        return ring_tns, ptr_tns, hidden_tns, bb_buf, bb_keys, bb_write_ptr, bb_steps, outs_tns

    def _ensure_batched_weights(self):
        """Stack per-expert weights into fused (N, out, in) tensors for bmm.

        Called once before the vectorized loop. Reuses existing nn.ModuleList
        weights (no extra parameters, fully checkpoint-compatible)."""
        if hasattr(self, '_bw_read') and self._bw_read is not None:
            return  # already cached
        N = self.N
        # read_proj: (N, hidden_dim, slot_dim) + (N, hidden_dim)
        self._bw_read = torch.stack([self.read_proj[i].weight for i in range(N)])
        self._bb_read = torch.stack([self.read_proj[i].bias for i in range(N)])
        # write_proj: (N, slot_dim, hidden_dim) + (N, slot_dim)
        if self.write_proj is not None:
            self._bw_write = torch.stack([self.write_proj[i].weight for i in range(N)])
            self._bb_write = torch.stack([self.write_proj[i].bias for i in range(N)])
        # write_gate: (N, 1, hidden_dim) + (N, 1)
        self._bw_wgate = torch.stack([self.write_gate[i].weight for i in range(N)])
        self._bb_wgate = torch.stack([self.write_gate[i].bias for i in range(N)])
        # ptr_query (pilot): (N, id_dim, hidden_dim) + (N, id_dim)
        if self.pointer_mode == 'pilot':
            self._bw_ptrq = torch.stack([self.ptr_query[i].weight for i in range(N)])
            self._bb_ptrq = torch.stack([self.ptr_query[i].bias for i in range(N)])

    def _invalidate_batched_weights(self):
        """Clear cached batched weights (call after optimizer.step())."""
        self._bw_read = None

    def _bmm_linear(self, x, W, b):
        """Batched linear: x (N,B,in) @ W^T (N,in,out) + b (N,out) → (N,B,out)"""
        return torch.bmm(x, W.transpose(1, 2)) + b.unsqueeze(1)

    def _process_chunk_vectorized(self, x_chunk, ring_tns, ptr_tns, hidden_tns,
                                  bb_buf, bb_keys, bb_write_ptr, bb_steps,
                                  S_flt, probs_lst, offsets_long, expert_weights,
                                  topk_write_weights=None):
        """Vectorized expert processing — replaces the for-i-in-range(N) loop.

        Instead of N sequential Linear calls, uses torch.bmm to process
        all experts in a single batched matmul. Combined with parallel ring
        access (all experts read same state, single coalesced write).

        Supports: vshape kernel + pilot pointer + replace write (production config).
        Falls back to _process_chunk_parallel for unsupported configs.
        """
        M, slot_dim, N, hidden_dim = self.M, self.slot_dim, self.N, self.hidden_dim

        # Validate: vectorized path only supports specific config
        if self.kernel_mode not in ('vshape',):
            return self._process_chunk_parallel(
                x_chunk, ring_tns, ptr_tns, hidden_tns,
                bb_buf, bb_keys, bb_write_ptr, bb_steps,
                S_flt, probs_lst, offsets_long, expert_weights,
                topk_write_weights)

        ptr_tns = ptr_tns.clone()
        # hidden_tns is already (N, B, hidden_dim) — keep as tensor, no unpacking

        C = x_chunk.shape[1]
        outs_lst = []
        W = offsets_long.shape[0]  # window size = 2R+1

        if self._bitlift:
            _bit_shifts = torch.arange(7, -1, -1, device=x_chunk.device)

        # Pre-stack expert weights for bmm (views into existing params, no copy)
        self._ensure_batched_weights()

        # Pre-expand vshape weights: (N, W) → (N, 1, W) for broadcasting
        # expert_weights is (N, 2R+1) — each expert's positional attention
        ew_N1W = expert_weights.unsqueeze(1)  # (N, 1, W)

        for t in range(C):
            # ── Embed input ──
            if self.inp is not None:
                if self._bitlift:
                    byte_val = x_chunk[:, t]
                    bits = ((byte_val.unsqueeze(-1) >> _bit_shifts) & 1).float()
                    input_vec = _c19_activation(self.inp(bits), C=_C_from_raw(self.c19_C_input))
                else:
                    input_vec = self.inp(x_chunk[:, t])
            else:
                input_vec = self._fixed_table[x_chunk[:, t]]

            B_size = input_vec.shape[0]

            # ── Circuit fix: ring decay ──
            if self.ring_decay_enabled:
                ring_tns = ring_tns * torch.sigmoid(self._raw_ring_decay)

            # ════════════════════════════════════════════════════
            # VECTORIZED PILOT POINTER (all N experts at once)
            # ════════════════════════════════════════════════════
            if self.pointer_mode == 'pilot':
                current_slots = ptr_tns.long().clamp(0, M - 1)       # (N, B)
                # Gather slot identities for all experts at once
                # slot_identity: (M, id_dim)  current_slots: (N, B)
                slot_ids = self.slot_identity[current_slots]          # (N, B, id_dim)
                slot_ids = F.normalize(slot_ids, dim=-1)              # (N, B, id_dim)

                # Batched query projection: hidden_tns (N,B,H) → queries (N,B,id_dim)
                queries = self._bmm_linear(hidden_tns, self._bw_ptrq, self._bb_ptrq)
                queries = F.normalize(queries, dim=-1)                # (N, B, id_dim)

                sim = (queries * slot_ids).sum(-1)                    # (N, B)
                tau = F.softplus(self._ptr_tau)
                jump = self._ptr_max_jump * torch.sigmoid(-sim * tau) # (N, B)
                ptr_tns = (ptr_tns + 1 + jump) % M                   # (N, B)

            # ════════════════════════════════════════════════════
            # VECTORIZED RING READ (all N experts, single batched gather)
            # ════════════════════════════════════════════════════
            centers = ptr_tns.long().clamp(0, M - 1)                  # (N, B)
            # Window indices: (N, B, W) — each expert's read positions
            indices = (centers.unsqueeze(2) + offsets_long) % M       # (N, B, W)

            # Expand for gather: (N, B, W, slot_dim)
            exp_idx = indices.unsqueeze(-1).expand(-1, -1, -1, slot_dim)  # (N, B, W, D)

            # Batched gather: ring_tns is (B, M, D), expand to (N, B, M, D) for N gathers
            ring_exp = ring_tns.unsqueeze(0).expand(N, -1, -1, -1)    # (N, B, M, D)
            neighbors = ring_exp.gather(2, exp_idx)                    # (N, B, W, D)

            # Weighted sum: (N, ?, W, 1) * (N, B, W, D) → sum over W → (N, B, D)
            if self.ptr_gradient and self.pointer_mode != 'sequential':
                # Circuit fix: fractional pointer interpolation (vectorized)
                ptr_float = ptr_tns % M                                       # (N, B)
                frac = ptr_float - ptr_float.detach().floor()                 # (N, B)
                adj_offsets = offsets_long.float().unsqueeze(0).unsqueeze(0) - frac.unsqueeze(2)  # (N, B, W)
                R_effs = self._R_eff.view(N, 1, 1)                           # (N, 1, 1)
                if self.kernel_mode == 'vshape':
                    raw_w = (1.0 - adj_offsets.abs() / R_effs.clamp(min=0.5)).clamp(min=0)
                elif self.kernel_mode == 'gaussian':
                    sigma = (R_effs / 2.5).clamp(min=0.3)
                    raw_w = torch.exp(-0.5 * (adj_offsets / sigma) ** 2)
                else:
                    raw_w = torch.sigmoid(10.0 * (R_effs - adj_offsets.abs()))
                weights_4d = (raw_w / raw_w.sum(dim=2, keepdim=True)).unsqueeze(-1)  # (N, B, W, 1)
            else:
                weights_4d = ew_N1W.unsqueeze(-1)                         # (N, 1, W, 1)
            read_vecs = (weights_4d * neighbors).sum(2)               # (N, B, D)

            # ════════════════════════════════════════════════════
            # VECTORIZED READ PROJECTION (all N: slot_dim → hidden_dim)
            # ════════════════════════════════════════════════════
            ring_signals = self._bmm_linear(read_vecs, self._bw_read, self._bb_read)  # (N, B, H)

            # ════════════════════════════════════════════════════
            # VECTORIZED GATE + BLEND
            # ════════════════════════════════════════════════════
            if S_flt == 'dotprod':
                # input_vec: (B, H) → broadcast to (N, B, H)
                input_exp = input_vec.unsqueeze(0).expand(N, -1, -1)  # (N, B, H)
                cos_sim = F.cosine_similarity(input_exp, ring_signals, dim=-1).unsqueeze(-1)  # (N, B, 1)
                gate_input = self.gate_tau * cos_sim
                if self.gate_bias_enabled:
                    gate_input = gate_input + self.gate_bias
                alpha = torch.sigmoid(gate_input)                     # (N, B, 1)
                blended_ring = alpha * ring_signals                   # (N, B, H)
            else:
                blended_ring = S_flt * ring_signals

            # ════════════════════════════════════════════════════
            # VECTORIZED PHASE SIGNAL
            # ════════════════════════════════════════════════════
            theta = (ptr_tns / M) * (2 * math.pi)                    # (N, B)
            phase = (torch.cos(theta).unsqueeze(-1) * self.phase_cos  # (N, B, H)
                   + torch.sin(theta).unsqueeze(-1) * self.phase_sin)

            # ════════════════════════════════════════════════════
            # VECTORIZED HIDDEN UPDATE (single batched c19)
            # ════════════════════════════════════════════════════
            input_exp = input_vec.unsqueeze(0).expand(N, -1, -1) if S_flt != 'dotprod' else input_exp
            hidden_tns = _c19_activation(
                input_exp + blended_ring + phase + hidden_tns,
                C=_C_from_raw(self.c19_C_hidden),
            )  # (N, B, H)

            # ════════════════════════════════════════════════════
            # VECTORIZED WRITE PROJECTION + GATE
            # ════════════════════════════════════════════════════
            if self.write_proj is not None:
                write_vecs = self._bmm_linear(hidden_tns, self._bw_write, self._bb_write)  # (N, B, D)
            else:
                write_vecs = hidden_tns  # identity when dims match

            if self._expert_conf is not None:
                conf = self._expert_conf.view(N, 1, 1)               # (N, 1, 1)
                write_vecs = conf * write_vecs

            # Write strength: (N, B, 1) — from batched write_gate
            ws = torch.sigmoid(self._bmm_linear(hidden_tns, self._bw_wgate, self._bb_wgate))  # (N, B, 1)

            # ════════════════════════════════════════════════════
            # COALESCED RING WRITE (single clone, all N experts)
            # ════════════════════════════════════════════════════
            ring_new = ring_tns.clone()                                # 1 clone per timestep

            # Write weights: (N, 1, W, 1) * ws: (N, B, 1, 1) → (N, B, W, 1)
            w = ew_N1W.unsqueeze(-1) * ws.unsqueeze(2)                # (N, B, W, 1)

            # write_vecs: (N, B, D) → expand to (N, B, W, D)
            wv_exp = write_vecs.unsqueeze(2).expand(-1, -1, W, -1)    # (N, B, W, D)

            # Current ring values at write positions
            ring_new_exp = ring_new.unsqueeze(0).expand(N, -1, -1, -1)  # (N, B, M, D)
            current = ring_new_exp.gather(2, exp_idx)                   # (N, B, W, D)

            # Write formula depends on mode
            if self.write_mode == 'additive':
                # Additive: ring_old + gate*w*write_vec (gradient=1.0 on old path)
                updated = current + w * wv_exp                        # (N, B, W, D)
            else:
                # HDD lerp: w * write_val + (1-w) * current
                updated = w * wv_exp + (1 - w) * current              # (N, B, W, D)

            # Scatter all experts back — sequential per expert (scatter_ doesn't support batched dim0)
            for i in range(N):
                ring_new.scatter_(1, exp_idx[i], updated[i])

            ring_tns = ring_new

            # ════════════════════════════════════════════════════
            # POINTER MOVEMENT (post-write, non-pilot modes)
            # ════════════════════════════════════════════════════
            if self.pointer_mode == 'pilot':
                pass  # already moved pre-read
            else:
                ptr_tns = (ptr_tns + 1) % M

            # ── Bulletin board write ──
            if self.bb_enabled and bb_buf is not None:
                bb_buf = bb_buf.clone()
                bb_buf[:, bb_write_ptr] = input_vec
                bb_keys = bb_keys.clone()
                written_key = self.bb_key_proj(input_vec)
                bb_keys[:, bb_write_ptr] = written_key
                bb_write_ptr = (bb_write_ptr + 1) % self._bb_L
                bb_steps += 1

            # ── Output ──
            if self.expert_output_weights_enabled:
                w = F.softmax(self.expert_output_logits, dim=0)       # (N,)
                mean_hidden = (hidden_tns * w.view(N, 1, 1)).sum(0)   # (B, H)
            else:
                mean_hidden = hidden_tns.mean(0)                      # (B, H)
            if self._bitlift_out:
                bit_scores = torch.tanh(self.out(mean_hidden))
                outs_lst.append(bit_scores @ self._bit_patterns.T)
            elif self.out is not None:
                outs_lst.append(self.out(mean_hidden))
            else:
                outs_lst.append(mean_hidden @ self._fixed_output_table.T)

        outs_tns = torch.stack(outs_lst, dim=1)

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

        # ─ state initialization ─
        # When state is provided (sequential/TBPTT mode), resume from previous
        # sequence's final state. Otherwise start fresh (zeros).
        if state is not None:
            ring_tns   = state['ring']                                     # (B, M, slot_dim)
            ptr_tns    = state['ptr']                                      # (N, B)
            hidden_tns = state['hidden']                                   # (N, B, hidden_dim)
        else:
            ring_tns   = func_ringstart_tns(B, M, slot_dim, x.device)     # (B, M, slot_dim)
            ptr_tns    = torch.zeros(N, B, device=x.device)               # (N, B)
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

        # Select chunk processor based on parallel_experts flag
        if self.parallel_experts == 'vectorized':
            _chunk_fn = self._process_chunk_vectorized
        elif self.parallel_experts:
            _chunk_fn = self._process_chunk_parallel
        else:
            _chunk_fn = self._process_chunk

        all_outs = []
        for t_start in range(0, T, C):
            t_end = min(t_start + C, T)
            x_chunk = x[:, t_start:t_end]

            if use_ckpt:
                # grad_checkpoint discards intermediate activations and recomputes
                # them during backward — saves ~13× VRAM on ring clones.
                ring_tns, ptr_tns, hidden_tns, bb_buf, bb_keys, bb_write_ptr, bb_steps, chunk_outs = grad_checkpoint(  # type: ignore[misc]
                    _chunk_fn,
                    x_chunk, ring_tns, ptr_tns, hidden_tns,
                    bb_buf, bb_keys, bb_write_ptr, bb_steps,
                    S, probs, offsets_long, expert_weights,
                    topk_write_weights,
                    use_reentrant=False,       # modern API, handles non-tensor args
                    preserve_rng_state=False,  # no random ops in the loop
                )
            else:
                ring_tns, ptr_tns, hidden_tns, bb_buf, bb_keys, bb_write_ptr, bb_steps, chunk_outs = _chunk_fn(
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
