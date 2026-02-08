"""Expert routing utilities.

This module contains the extracted *expert router* used by the INSTNCT
kernel:

- Each batch element routes to one of N output heads ("experts") based on a
  pointer/ring address.
- Optional "hibernation" support allows experts to be offloaded to disk and
  restored during the forward pass.

Behavior is locked by ``tests/verify_golden.py``. Keep semantics stable.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, MutableMapping
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


StateMap = Mapping[str, Any]
MetaMap = MutableMapping[str, Any]


def _parse_capacity_split_env() -> Optional[list[float]]:
    """Parse VRX_EXPERT_CAPACITY_SPLIT into positive floats.

    Format:
      VRX_EXPERT_CAPACITY_SPLIT="0.55,0.34,0.11"

    Returns None when unset/invalid.
    """

    raw = os.environ.get("VRX_EXPERT_CAPACITY_SPLIT", "").strip()
    if not raw:
        return None
    try:
        vals = [float(p.strip()) for p in raw.split(",") if p.strip()]
    except Exception:
        return None
    if not vals or any((not torch.isfinite(torch.tensor(v)) or float(v) <= 0.0) for v in vals):
        return None
    total = float(sum(vals))
    if total <= 0.0:
        return None
    return [float(v) / total for v in vals]


def _capacity_total_mult() -> float:
    raw = os.environ.get("VRX_EXPERT_CAPACITY_TOTAL_MULT", "").strip()
    if not raw:
        return 1.0
    try:
        val = float(raw)
    except Exception:
        return 1.0
    if not torch.isfinite(torch.tensor(val)) or val <= 0.0:
        return 1.0
    return float(val)


def _capacity_min_hidden() -> int:
    raw = os.environ.get("VRX_EXPERT_CAPACITY_MIN_HIDDEN", "").strip()
    if not raw:
        return 8
    try:
        val = int(float(raw))
    except Exception:
        return 8
    return max(1, int(val))


def _parse_capacity_hidden_dims_env(num_experts: int) -> Optional[list[int]]:
    """Parse explicit hidden dims for adapter experts.

    Format:
      VRX_EXPERT_CAPACITY_HIDDEN_DIMS="950,588,190"
    """

    raw = os.environ.get("VRX_EXPERT_CAPACITY_HIDDEN_DIMS", "").strip()
    if not raw:
        return None
    try:
        vals = [int(float(p.strip())) for p in raw.split(",") if p.strip()]
    except Exception:
        return None
    if len(vals) != int(num_experts):
        return None
    if any(int(v) <= 0 for v in vals):
        return None
    return [int(v) for v in vals]


def _allocate_hidden_dims(
    *,
    d_model: int,
    num_experts: int,
    split: list[float],
    total_mult: float,
    min_hidden: int,
) -> Optional[list[int]]:
    """Allocate per-expert hidden sizes that preserve total capacity budget.

    Baseline budget is `num_experts * d_model`, optionally scaled by
    VRX_EXPERT_CAPACITY_TOTAL_MULT. Hidden dims are integer-allocated by split.
    """

    if len(split) != int(num_experts):
        return None
    if int(d_model) <= 0 or int(num_experts) <= 0:
        return None
    base_total = int(round(float(num_experts * d_model) * float(total_mult)))
    base_total = max(int(min_hidden * num_experts), base_total)

    targets = [float(base_total) * float(w) for w in split]
    dims = [max(int(min_hidden), int(t)) for t in targets]
    cur = int(sum(dims))

    # Distribute residual to match exact target sum.
    if cur < base_total:
        frac = sorted(
            [(targets[i] - float(int(targets[i])), i) for i in range(len(targets))],
            reverse=True,
        )
        k = 0
        while cur < base_total:
            dims[frac[k % len(frac)][1]] += 1
            cur += 1
            k += 1
    elif cur > base_total:
        frac = sorted(
            [(targets[i] - float(int(targets[i])), i) for i in range(len(targets))],
            reverse=False,
        )
        k = 0
        while cur > base_total and k < 10_000:
            idx = frac[k % len(frac)][1]
            if dims[idx] > int(min_hidden):
                dims[idx] -= 1
                cur -= 1
            k += 1
    return dims


class AdapterExpert(nn.Module):
    """Small adapter expert: shared-width -> hidden -> logits."""

    def __init__(self, d_model: int, vocab_size: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(max(1, hidden_dim))
        self.down = nn.Linear(int(d_model), self.hidden_dim)
        self.up = nn.Linear(self.hidden_dim, int(vocab_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(x)))


def _hash_state_dict(state: Optional[StateMap]) -> Optional[str]:
    """Return a deterministic SHA256 hash for a state-dict-like mapping.

    Contract (legacy compatible):
    - Returns ``None`` if ``state`` is falsy.
    - Keys are processed in sorted order.
    - Tensor values contribute their contiguous CPU bytes. If a dtype cannot be
      converted to NumPy (e.g. bfloat16 on some builds), this function falls
      back to hashing a float32-cast view (behavior-preserving for previously
      supported dtypes; prevents crashes for unsupported ones).
    - Non-tensor values contribute ``repr(value)``.
    """

    if not state:
        return None

    hshobj = hashlib.sha256()
    for keystr in sorted(state.keys()):
        valobj = state[keystr]
        hshobj.update(keystr.encode("utf-8"))
        if torch.is_tensor(valobj):
            tenval = valobj.detach().contiguous().cpu()
            try:
                arrval = tenval.numpy()
            except Exception:
                # NumPy may not support some PyTorch dtypes (notably bfloat16).
                # Cast to float32 to produce a deterministic byte view.
                arrval = tenval.to(dtype=torch.float32).numpy()
            hshobj.update(arrval.tobytes())
        else:
            hshobj.update(repr(valobj).encode("utf-8"))

    return hshobj.hexdigest()


def _safe_torch_load(path: str) -> Any:
    """Load a torch payload onto CPU (best-effort, version compatible)."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older PyTorch does not accept weights_only.
        return torch.load(path, map_location="cpu")
    except Exception:
        # If weights_only is supported but the checkpoint contains unsupported
        # objects, fall back to a regular load to preserve legacy behavior.
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


def _load_expert_snapshot(path: Optional[str]) -> Tuple[Optional[StateMap], Optional[str]]:
    """Load an expert snapshot from disk and return (state, sha256).

    Missing/unreadable snapshots are treated as corruption and return
    ``(None, None)``.
    """

    if not path or not os.path.exists(path):
        return None, None

    try:
        loaded = _safe_torch_load(path)
    except Exception:
        return None, None

    if not isinstance(loaded, Mapping):
        return None, None

    digval = _hash_state_dict(loaded)
    if digval is None:
        return None, None

    # Cast to a plain dict to avoid surprises from OrderedDict subclasses.
    return dict(loaded), digval


def _restore_expert_state(expert: nn.Module, state: StateMap) -> None:
    """Restore parameters and buffers from ``state`` into ``expert``.

    This is intentionally lenient (matches the legacy extraction): only keys
    that exist in both module and state are copied.
    """

    if not state:
        return

    with torch.no_grad():
        for namstr, parval in expert.named_parameters():
            srcval = state.get(namstr)
            if torch.is_tensor(srcval):
                parval.copy_(srcval.to(device=parval.device, dtype=parval.dtype))

        for namstr, bufval in expert.named_buffers():
            srcval = state.get(namstr)
            if torch.is_tensor(srcval):
                bufval.copy_(srcval.to(device=bufval.device, dtype=bufval.dtype))


class LocationExpertRouter(nn.Module):
    """Route each batch element to one of N output heads.

    Routing rule (behavior-preserving):
      - If ``pointer_addresses`` is ``None``, all samples route to expert 0.
      - Else ``expert_index = pointer_addresses % num_experts``.

    Hibernation:
      External lifecycle code may set:
        - ``self.hibernation_enabled = True``
        - ``self.hibernation_state = {i: {"offloaded": True, "path": ..., "hash": ...}}``

      During ``forward``, offloaded experts are restored from disk.

      Counters (created on demand):
        - ``self.hibernation_fetched``: incremented for each attempted fetch.
        - ``self.hibernation_corrupt``: incremented when a snapshot is missing
          or its hash mismatches the saved hash.

    IMPORTANT: restoration is attempted for *each* expert in index order
    regardless of whether the current batch routes to it. This preserves legacy
    side effects relied upon by existing tooling.
    """

    def __init__(self, d_model: int, vocab_size: int, num_experts: int = 1) -> None:
        super().__init__()

        self.num_experts = max(1, int(num_experts))
        self.in_features = int(d_model)
        self.out_features = int(vocab_size)
        self.capacity_mode = "linear"
        self.capacity_hidden_dims: Optional[list[int]] = None

        if self.num_experts == 1:
            self.single: Optional[nn.Linear] = nn.Linear(d_model, vocab_size)
            self.experts: Optional[nn.ModuleList] = None
        else:
            self.single = None
            hidden_dims = _parse_capacity_hidden_dims_env(self.num_experts)
            if hidden_dims is None:
                split = _parse_capacity_split_env()
                if split is not None and len(split) == self.num_experts:
                    hidden_dims = _allocate_hidden_dims(
                        d_model=self.in_features,
                        num_experts=self.num_experts,
                        split=split,
                        total_mult=_capacity_total_mult(),
                        min_hidden=_capacity_min_hidden(),
                    )
            if hidden_dims is not None:
                self.experts = nn.ModuleList(
                    [AdapterExpert(d_model, vocab_size, h) for h in hidden_dims]
                )
                self.capacity_mode = "adapter"
                self.capacity_hidden_dims = [int(h) for h in hidden_dims]
            else:
                self.experts = nn.ModuleList(
                    [nn.Linear(d_model, vocab_size) for _ in range(self.num_experts)]
                )

    def reset_parameters(self) -> None:
        """Reset weights using Xavier init (behavior-preserving)."""

        def init6x(laysix: nn.Linear) -> None:
            nn.init.xavier_uniform_(laysix.weight)
            if laysix.bias is not None:
                nn.init.zeros_(laysix.bias)

        if self.single is not None:
            init6x(self.single)
            return

        explst = self.experts
        if explst is None:
            return
        for expsix in explst:
            if isinstance(expsix, nn.Linear):
                init6x(expsix)
            elif hasattr(expsix, "reset_parameters"):
                expsix.reset_parameters()

    def _maybe_restore_expert(self, expidx: int, expert: nn.Module) -> None:
        """Restore an offloaded expert from disk if hibernation is enabled."""

        hibena = bool(getattr(self, "hibernation_enabled", False))
        hibsta = getattr(self, "hibernation_state", None)
        if not hibena or not hibsta:
            return

        meta6x = hibsta.get(expidx) if isinstance(hibsta, dict) else None
        if not isinstance(meta6x, MutableMapping) or not meta6x.get("offloaded"):
            return

        staval, digval = _load_expert_snapshot(meta6x.get("path"))
        savhsh = meta6x.get("hash")

        if digval is None:
            # Missing/corrupt snapshot.
            self.hibernation_corrupt = getattr(self, "hibernation_corrupt", 0) + 1
            meta6x["offloaded"] = False
        else:
            if savhsh and digval != savhsh:
                self.hibernation_corrupt = getattr(self, "hibernation_corrupt", 0) + 1
            _restore_expert_state(expert, staval or {})
            meta6x["hash"] = digval
            meta6x["offloaded"] = False

        self.hibernation_fetched = getattr(self, "hibernation_fetched", 0) + 1

    def forward(self, x: torch.Tensor, pointer_addresses: torch.Tensor | None = None) -> torch.Tensor:
        if self.single is not None:
            return self.single(x)

        explst = self.experts
        if explst is None:
            raise RuntimeError("LocationExpertRouter misconfigured: missing experts")

        if pointer_addresses is None:
            expidx = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        else:
            expidx = pointer_addresses.to(torch.long, non_blocking=True) % self.num_experts

        first_par = next(explst[0].parameters())
        outdty = first_par.dtype
        outten = torch.zeros(x.shape[0], int(self.out_features), device=x.device, dtype=outdty)

        for idxsix, expsix in enumerate(explst):
            # Behavior-preserving: restoration is attempted regardless of routing.
            self._maybe_restore_expert(idxsix, expsix)

            mask6x = expidx == idxsix
            if mask6x.any():
                outten[mask6x] = expsix(x[mask6x]).to(outdty)

        return outten
