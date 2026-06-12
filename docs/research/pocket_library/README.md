# Pocket Library

This directory is the canonical registry for reusable VRAXION Pocket Operators.

The rule is intentionally strict:

```text
Archive frozen anchors under docs/research/pocket_archive/.
Record lifecycle selection under docs/research/pocket_ecology/.
Load reusable pockets only through docs/research/pocket_library/registry.json.
Do not overwrite a stable/core frozen anchor directly.
Write new pockets as candidate/staging artifacts, then promote only after checks.
```

## Current Status

The active registry is:

```text
docs/research/pocket_library/registry.json
```

The active training/ABI lock is:

```text
docs/research/POCKET_GENERATION_TRAINING_LOCK_V1.md
docs/research/pocket_library/training_lock_v1.json
```

The first stable pocket is:

```text
protocol_framing_ingress_v001
```

It is a protocol/framing ingress mechanism pocket, not a full binary-stream or
text-ingress solution. It still requires a target-world adapter when feature
codebooks differ, and bit-slip stream reassembly remains unresolved.

## Lifecycle Statuses

```text
candidate   new, not trusted
staging     useful but not stable/core
stable      load-allowed frozen anchor
core        load-allowed high-confidence frozen anchor
deprecated  stale/AFK/no-use pocket
banned      toxic/negative-transfer pocket
```

Only `stable` and `core` pockets may be loaded automatically by future probes.

Stable/core pockets must declare `abi_version = PocketABI-v1`.

The locked principle:

```text
Freeze the interface, not the internals.
```

Pocket internals may change across candidates, but loadable library pockets must
keep their external call contract, adapter boundary, known bottlenecks, and
frozen parameter digest explicit.

## Helper

Use:

```text
python scripts/probes/pocket_library.py --check
python scripts/probes/pocket_library.py --list-stable
python scripts/probes/pocket_library.py --load-pocket protocol_framing_ingress_v001
python scripts/probes/run_pocket_library_registry_check.py --write-summary
```

Future probes should import `scripts/probes/pocket_library.py` and call
`load_frozen_params()` instead of hardcoding archive paths.
