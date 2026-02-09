"""Expert routing — LocationExpertRouter + hibernation helpers.

Each batch element routes to one of N output heads ("experts") based on
a pointer/ring address. Optional hibernation support offloads experts to disk.

Behavior is locked by ``tests/verify_golden.py``. Keep semantics stable.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, MutableMapping
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


StateMap = Mapping[str, Any]
MetaMap = MutableMapping[str, Any]


def _hash_state_dict(state: Optional[StateMap]) -> Optional[str]:
    """Return a deterministic SHA256 hash for a state-dict-like mapping."""
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
        return torch.load(path, map_location="cpu")
    except Exception:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


def _load_expert_snapshot(path: Optional[str]) -> Tuple[Optional[StateMap], Optional[str]]:
    """Load an expert snapshot from disk and return (state, sha256)."""
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

    return dict(loaded), digval


def _restore_expert_state(expert: nn.Module, state: StateMap) -> None:
    """Restore parameters and buffers from ``state`` into ``expert``."""
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

    Routing rule:
      - If ``pointer_addresses`` is ``None``, all samples route to expert 0.
      - Else ``expert_index = pointer_addresses % num_experts``.

    Hibernation:
      External lifecycle code may set:
        - ``self.hibernation_enabled = True``
        - ``self.hibernation_state = {i: {"offloaded": True, "path": ..., "hash": ...}}``

      During ``forward``, offloaded experts are restored from disk.
    """

    def __init__(self, d_model: int, vocab_size: int, num_experts: int = 1) -> None:
        super().__init__()

        self.num_experts = max(1, int(num_experts))
        self.in_features = int(d_model)
        self.out_features = int(vocab_size)

        if self.num_experts == 1:
            self.single: Optional[nn.Linear] = nn.Linear(d_model, vocab_size)
            self.experts: Optional[nn.ModuleList] = None
        else:
            self.single = None
            self.experts = nn.ModuleList(
                [nn.Linear(d_model, vocab_size) for _ in range(self.num_experts)]
            )

    def reset_parameters(self) -> None:
        """Reset weights using Xavier init."""

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
            init6x(expsix)

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

        outdty = explst[0].weight.dtype
        outten = torch.zeros(x.shape[0], explst[0].out_features, device=x.device, dtype=outdty)

        for idxsix, expsix in enumerate(explst):
            self._maybe_restore_expert(idxsix, expsix)

            mask6x = expidx == idxsix
            if mask6x.any():
                outten[mask6x] = expsix(x[mask6x]).to(outdty)

        return outten
