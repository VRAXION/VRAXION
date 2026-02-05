# Static-Space Truth Audit (VRAXION)

Purpose: pin down **repo-truth** for “static keyspace + checkpoint-time refinement” so future architecture work stays grounded and doesn’t accidentally implement runtime paging.

This document is **descriptive** (what exists today), not prescriptive.

## 1) Static keyspace = ring address space (router_map index space)

VRAXION already has a stable discrete coordinate system:

- `AbsoluteHallway` registers `router_map` as a buffer that maps **address → expert_id**.
  - `map_len` is derived from `ring_range` (the configured ring size).
  - `router_map[i]` is the canonical owner of address `i`.
  - Source: `Golden Code/vraxion/instnct/absolute_hallway.py:584-590`.

Routing uses that keyspace directly:

- `_map_expert_ids(ptr_int)` clamps addresses into `[0, len(router_map)-1]`, looks up `expert_ids = router_map[idx]`, and guards against stale maps with `% head.num_experts`.
  - Source: `Golden Code/vraxion/instnct/absolute_hallway.py:716-727`.

### Why this matters
If “coordinate X must always mean coordinate X”, then the **meaningful invariant** in this repo is:

- the **address** (`ptr_int`) and
- the **address→expert mapping** (`router_map`)

…not a slice of a learned feature vector.

## 2) Pointer address telemetry: `last_ptr_int`

`AbsoluteHallway` writes a stable per-batch pointer address at the end of forward:

- `self.last_ptr_int = ptr_int.detach().cpu()`
  - Source: `Golden Code/vraxion/instnct/absolute_hallway.py:1655-1660`.

So tooling can read `model.last_ptr_int` as the “current address” without needing custom hooks.

## 3) Multi-expert head layout: `head.single.*` vs `head.experts.<id>.*`

The output head implementation is `LocationExpertRouter`:

- If `num_experts == 1`, it constructs `self.single: nn.Linear` and **does not** create `self.experts`.
- Else it constructs `self.experts: nn.ModuleList([nn.Linear(...) ...])` and `self.single=None`.
  - Source: `Golden Code/vraxion/instnct/experts.py:153-167`.

**Consequence:** checkpoints differ structurally:

- 1 expert ⇒ keys like `head.single.*`
- N experts ⇒ keys like `head.experts.<id>.*`

This also means “1→2 growth” is not just a `router_map` edit; it requires a checkpoint transform that creates `head.experts.*` keys.

## 4) Offline refinement tools already exist (checkpoint-time only)

### 4.1 Mitosis (split / fission)
`Golden Draft/vraxion_mitosis_split.py` performs checkpoint-only expert fission:

- clones tensors under `head.experts.<parent_id>.` into a new highest-id expert slot
- redirects selected addresses in `router_map` to the new expert id
  - Source: `Golden Draft/vraxion_mitosis_split.py:4-9` and `Golden Draft/vraxion_mitosis_split.py:104-131`.

### 4.2 Prune / merge
`Golden Draft/tools/vraxion_prune_merge.py` removes the highest-index expert:

- remaps `router_map` entries pointing to the removed expert into a kept expert
- deletes the removed expert’s tensors from the checkpoint state dict
  - Source: `Golden Draft/tools/vraxion_prune_merge.py:4-13` and `Golden Draft/tools/vraxion_prune_merge.py:154-170`.

### Interpretation
These tools imply the cleanest v0/v1 “paging” model in this repo is:

- **SSD / disk = checkpoint store**
- structural edits happen **offline** (checkpoint boundaries)
- online forward pass should not do disk I/O

## 5) Runtime paging / hibernation is unsafe to build on today

`LocationExpertRouter` supports “hibernation” (restore experts from disk during forward), but its semantics are explicitly dangerous for scale:

- “restoration is attempted for *each* expert in index order regardless of whether the current batch routes to it.”
  - Source: `Golden Code/vraxion/instnct/experts.py:148-151`.

And the forward loop does exactly that:

- inside `forward(...)`, it iterates `for idxsix, expsix in enumerate(explst): self._maybe_restore_expert(idxsix, expsix)` regardless of routing.
  - Source: `Golden Code/vraxion/instnct/experts.py:231-234`.

### Conclusion (hard guardrail)
Any “demand paging from disk during forward()” is forbidden in v0/v1. If we ever want true demand paging, we must redesign the restore contract to “restore only routed experts this step” and version that behavior explicitly.

## 6) Proven pattern: EXPERT_HEADS is an init-time global

`AbsoluteHallway` uses a module-global `EXPERT_HEADS` to decide `num_experts` at construction time:

- `EXPERT_HEADS = 1` (default)
  - Source: `Golden Code/vraxion/instnct/absolute_hallway.py:295-296`.
- `self.head = LocationExpertRouter(... num_experts=int(EXPERT_HEADS))`
  - Source: `Golden Code/vraxion/instnct/absolute_hallway.py:584-586`.

One existing tool already sets this correctly before building a model:

- `Golden Draft/tools/gpu_capacity_probe.py` sets `absolute_hallway.EXPERT_HEADS = int(args.out_dim)` before instantiating `AbsoluteHallway`.
  - Source: `Golden Draft/tools/gpu_capacity_probe.py:495-509`.

This is the “truth” we build on for reliable multi-expert checkpoints: ensure Golden Draft entrypoints set `absolute_hallway.EXPERT_HEADS` from `vraxion.instnct.seed.EXPERT_HEADS` before model construction.

