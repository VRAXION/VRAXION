# E37 Pocket ABI v1 And Registry Compatibility Lock Result

Status: complete.

Decision:

```text
e37_pocket_abi_v1_registry_lock_confirmed
```

## Summary

E37 locks the external Pocket ABI and registry compatibility rules before
longer pocket-generation curriculum runs accumulate more reusable artifacts.

This is infrastructure/versioning work, not a new reasoning capability probe.

## Locked Artifacts

```text
docs/research/POCKET_ABI_V1.md
docs/research/POCKET_GENERATION_TRAINING_LOCK_V1.md
docs/research/pocket_library/abi_schema_v1.json
docs/research/pocket_library/registry_schema_v1.json
docs/research/pocket_library/training_lock_v1.json
docs/research/pocket_library/registry.json
scripts/probes/pocket_library.py
scripts/probes/run_pocket_abi_v1_registry_check.py
```

The stable `protocol_framing_ingress_v001` archive now also includes:

```text
abi_compatibility.json
```

## Checks

```text
registry validates against schema v1
PocketABI-v1 lock is active
stable/core archive files exist
stable/core frozen params digest matches registry and manifest
candidate/staging/deprecated/banned cannot be auto-loaded
banned pocket load hard-fails
stable/core anchor overwrite attempt is blocked
unknown ABI major version hard-fails
adapter-required target import fails without adapter declaration
adapter-declared target import passes
protocol_framing_ingress_v001 passes ABI v1 compatibility
dirty/staging and banned pockets remain not load-allowed
```

Checker result:

```text
run_pocket_abi_v1_registry_check.py:
  decision = e37_pocket_abi_v1_registry_lock_confirmed
  passed = true
  checker_failure_count = 0

pocket_library.py --check:
  passed = true
  failure_count = 0
  stable_pocket_ids = protocol_framing_ingress_v001
```

## Boundary

E37 does not freeze pocket internals:

```text
internal matrix size
capacity K
mutation operator
quantization implementation
Flow/Grounding Field dimension
parser/protocol family
```

E37 freezes the external lifecycle boundary:

```text
ABI
registry schema
archive layout
load policy
adapter boundary
trace/mask/cost declarations
re-audit policy
```

Boundary: E37 is infrastructure/versioning work. It does not prove raw language
reasoning, AGI, consciousness, deployed-model behavior, or model-scale behavior.
