import math                # sqrt for golden ratio constant
import torch               # tensor operations and autograd
import torch.nn as nn      # neural network layers (Linear, Module, ModuleList)
import yaml                # YAML config parser
from pathlib import Path   # cross-platform file path resolution

# ── Constants ───────────────────────────────────────────────────
# inverse golden ratio — used in phi_destinations() to compute maximally-spaced pointer jumps
PHI_INV = (math.sqrt(5) - 1) / 2  # 1/φ ≈ 0.6180339887

# ── Config ──────────────────────────────────────────────────────
# All hyperparameters live in vraxion_config.yaml — single source of truth.
# Change values in the YAML, not here — these lines just bind them to Python names.
_yaml_path = Path(__file__).parent / 'vraxion_config.yaml'
with open(_yaml_path, encoding='utf-8') as _f:
    _cfg = yaml.safe_load(_f)['model']  # only the model section

# ─ architecture ─
CNFG_RINGSLOTS_INT = _cfg['M']    # 256  memory slots on the ring buffer
CNFG_RSLOTDIMS_INT = _cfg['D']    # 256  embedding dimensions per slot
CNFG_BEINGSNUM_INT = _cfg['N']    # 6    parallel experts sharing the ring
CNFG_ATNRADIUS_INT = _cfg['R']    # 2    attention window half-width (window = 2R+1)

# ─ pointer behavior ─
CNFG_CNTEXTSCL_FLT = _cfg['S']    # 0.05 how much the read signal scales into hidden state
CNFG_JUMPIPROB_FLT = _cfg['Je']   # 0.9  probability of φ-jump (long leap)
CNFG_WALKIPROB_FLT = _cfg['Jw']   # 0.1  probability of +1 walk (short step)

del _yaml_path, _cfg, _f  # remove temporaries so they don't pollute the module namespace

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
        dims_int,    # D — each cell holds a D-dimensional embedding vector
        device=device_str
    )

# · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
def func_attnwndow_tpl(  # function: attention window | format: tuple (tpl)
    ptr_tns,       # pointer position per batch (B,)
    radius_int,    # attention radius — window covers 2R+1 slots
    slots_int,     # total ring slots (M)
    device_str     # torch device (cpu/cuda)
):
    """Builds a sliding window of 2R+1 ring slots centered on the current
    pointer position. Each slot gets equal weight (uniform mean-pool)
    so the read/write operations blend nearby memory smoothly. The
    window wraps around using modular arithmetic — no edge effects."""

    # total slots the window covers — e.g. 2*2+1 = 5 when R=2
    window_size_int = 2 * radius_int + 1

    # symmetric offsets around the pointer: [-2,-1,0,1,2] when R=2; shape (2R+1,)
    offsets_tns = torch.arange(-radius_int, radius_int + 1, device=device_str)

    # pointer arrives as float (blended position) — truncate to int slot,
    # clamp to valid ring range [0, M-1] so gather never goes out of bounds; shape (B,)
    center_tns = ptr_tns.long().clamp(0, slots_int - 1)

    # add offsets to each batch's center, mod M wraps around the ring
    # so slot 63+1 becomes slot 0 — no edge effects; shape (B, 2R+1)
    indices_tns = (center_tns.unsqueeze(1) + offsets_tns) % slots_int

    # every slot in the window gets equal weight 1/(2R+1) — uniform mean-pool
    # so the read blends 5 neighbors equally (no learned attention yet); shape (B, 2R+1)
    weights_tns = torch.full((ptr_tns.shape[0], window_size_int), 1.0 / window_size_int, device=device_str)

    return indices_tns, weights_tns

# · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
def func_softread_tns(  # func_ = standalone op, _tns = returns a tensor
    ring_tns,      # the shared ring buffer; shape (B, M, D)
    indices_tns,   # window slot indices from func_attnwndow_tpl; shape (B, 2R+1)
    weights_tns,   # window weights from func_attnwndow_tpl; shape (B, 2R+1)
    dims_int       # embedding dimension (D) — needed to expand indices for gather
):
    """Differentiable read from the ring buffer. Gathers 2R+1 neighbor
    slot vectors at the window positions, then collapses them into a
    single D-dimensional read vector via uniform weighted sum.
    The caller applies the per-expert projection and scaling separately."""

    # expand window indices from (B, 2R+1) to (B, 2R+1, D) so gather can pull full D-dim vectors
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

    # scatter_add writes weighted hidden into ring at window positions — additive, never erases; shape (B, M, D)
    return ring_tns.scatter_add(1, expanded_idx_tns, weights_tns.unsqueeze(-1) * write_val_tns)

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
    """Ring-buffer pointer network — v4 minimal reference implementation.
    N experts share one ring of M slots (each D-wide). Each expert reads,
    writes, and moves its pointer every timestep. No activation functions,
    no decay — pure linear accumulation. Complexity lives in the ring
    topology, not in the compute graph."""

    def __init__(self,
                 M=CNFG_RINGSLOTS_INT,   # ring slots (64)
                 D=CNFG_RSLOTDIMS_INT,   # embedding dim (64)
                 N=CNFG_BEINGSNUM_INT,   # expert count (6)
                 R=CNFG_ATNRADIUS_INT,   # attention radius (2)
                 B=8):                    # bits per input position (default: 1 byte = 8 bits)
        super().__init__()                         # initialize nn.Module — registers parameters and buffers
        self.M, self.D, self.N, self.R = M, D, N, R  # store arch dims for forward()
        self.B = B                                    # input/output token width

        # input projection: raw B-dim token → D-dim embedding space
        self.inp = nn.Linear(B, D)

        # output projection: D-dim hidden → B-dim prediction (mirrors inp)
        self.out = nn.Linear(D, B)

        # one learned D→D read projection per expert — transforms ring read before mixing into hidden
        self.read_proj = nn.ModuleList([nn.Linear(D, D) for _ in range(N)])

        # precomputed φ-jump table — shape (N, M), not trainable, moves with device
        self.dests: torch.Tensor                   # type hint for Pylance — register_buffer is dynamic
        self.register_buffer('dests', phi_destinations(N, M))

    def forward(self, x, S=CNFG_CNTEXTSCL_FLT, probs=None):
        """Processes T timesteps over B sequences. Each timestep: every expert
        reads from ring, updates hidden, writes back, moves pointer.
        Output = uniform mean of all N experts, projected to token space."""

        B, T, _ = x.shape                           # B=batch, T=time, _=self.B (token dim)
        M, D, N, R = self.M, self.D, self.N, self.R # unpack for readability

        # 3-role cycling: explorer (0.9 jump), walker (0.1 jump), neutral (0.5)
        # expert i gets role i % len(probs) — 6 experts = 2 of each role
        if probs is None:
            probs = [CNFG_JUMPIPROB_FLT, CNFG_WALKIPROB_FLT, 0.5]

        # ─ state initialization — all start at zero ─
        ring_tns   = func_ringstart_tns(B, M, D, x.device)  # shared memory; shape (B, M, D)
        ptr_tns    = torch.zeros(N, B, device=x.device)      # pointer positions; shape (N, B)
        hidden_tns = torch.zeros(N, B, D, device=x.device)   # hidden accumulators; shape (N, B, D)
        outs_lst   = []                                       # one output per timestep

        for t in range(T):
            # project token into embedding space — shared across all experts; shape (B, D)
            input_vec_tns = self.inp(x[:, t])

            for i in range(N):
                # accumulate input into this expert's hidden (additive, no decay)
                hidden_tns[i] += input_vec_tns

                # build attention window centered on pointer; returns (B, 2R+1) indices + weights
                indices_tns, weights_tns = func_attnwndow_tpl(ptr_tns[i], R, M, x.device)

                # ─ soft_read ─
                read_vec_tns, expanded_idx_tns = func_softread_tns(ring_tns, indices_tns, weights_tns, D)
                # learned D→D projection, scaled by S (0.05) — small so read nudges, doesn't dominate
                hidden_tns[i] += S * self.read_proj[i](read_vec_tns)

                # ─ soft_write ─
                ring_tns = func_softwrit_tns(ring_tns, hidden_tns[i], expanded_idx_tns, weights_tns)

                # ─ move_pointer — role cycles: explorer(0.9) → walker(0.1) → neutral(0.5) → ... ─
                ptr_tns[i] = func_movepntr_tns(ptr_tns[i], self.dests[i], probs[i % len(probs)], M)

            # collapse N experts via uniform mean, project to token space; shape (B, self.B)
            outs_lst.append(self.out(hidden_tns.mean(0)))

        return torch.stack(outs_lst, 1)  # stack along time axis; shape (B, T, self.B)
