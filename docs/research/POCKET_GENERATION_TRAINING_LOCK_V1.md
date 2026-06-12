# Pocket Generation Training Lock v1

Status: active.

This lock defines what is stable enough to start curriculum-driven pocket
generation without repeatedly guessing the architecture boundary.

## Locked Architecture Boundary

```text
Grounding / Flow Field
  shared state substrate

Router / Controller
  chooses which pocket to call

Pocket Operator
  reusable state-transforming mechanism

Adapter
  local translation layer for world/codebook/layout differences

Pocket Library
  versioned archive + registry + lifecycle evaluator
```

The locked rule:

```text
Freeze the pocket interface, not the pocket internals.
```

Pocket internals may vary:

```text
matrix size
capacity K
quantization
mutation history
implementation details
```

Pocket ABI must remain stable and explicit.

## Pocket ABI v1

Every stable/core pocket must declare:

```text
abi_version = PocketABI-v1
pocket_id
version
pocket_type
input_contract
output_contract
allowed_side_effects
requires_adapter
compatible_families
known_bottlenecks
frozen_params_digest
status
load_allowed
promotion_source
```

Minimal call shape:

```text
CALL(pocket_id, flow_or_payload, context) -> proposal/event/reject
```

For E35/E36 ingress pockets, the concrete call shape is:

```text
CALL(raw_bits, requested_feature?) -> evidence_event or reject
```

## What May Become Core Later

A pocket may become stable/core only if it repeatedly shows:

```text
positive or safety-useful contribution across worlds
low wrong write / wrong commit rate
clean manifest and frozen param hash
non-toxic cross-world behavior
clear contract boundary
known adapter requirements
known failure modes
sample/checker/replay pass
```

Core should mean:

```text
measurably useful in our tested universe
low/no measured downside under adversarial controls
safe to load automatically
```

## What Must Not Be Auto-Locked

Do not promote these directly to stable/core:

```text
dirty monolithic decoders
world-specific fused pockets
pockets needing hidden oracle state
pockets with unknown write side effects
pockets that only win one seed/run
pockets that are useful only because another pocket did the real work
AFK/dormant pockets
toxic/wrong-codebook pockets
```

These may exist as candidate/staging diagnostics, but `load_allowed` must stay
false until they pass the registry and ecology gates.

## Canonical Storage

```text
registry:
  docs/research/pocket_library/registry.json

stable frozen archives:
  docs/research/pocket_archive/

ecology value reports:
  docs/research/pocket_ecology/

registry helper:
  scripts/probes/pocket_library.py
```

Future probes should not hardcode archive paths. They should load pockets through:

```text
pocket_library.load_frozen_params(pocket_id)
```

## Required Before Long Curriculum Runs

Before a long pocket-generation/curriculum run:

```text
python scripts/probes/pocket_library.py --check
python scripts/probes/run_pocket_library_registry_check.py --write-summary
```

Long runs must still obey the repository rule:

```text
write progress and partial outcomes on a heartbeat
use available CPU/GPU aggressively where safe
never run as a black box
```

## Current Locked Stable Pocket

```text
protocol_framing_ingress_v001
status = stable
load_allowed = true
abi_version = PocketABI-v1
```

Known boundary:

```text
Good:
  stable start/length/CRC/end framing families

Needs local adapter:
  changed feature codebook / target-world mapping

Not solved:
  bit-slip stream reassembly
```

Boundary: this lock is an engineering/research scaffold for controlled pocket
generation. It is not a raw language reasoning, AGI, consciousness,
deployed-model, or model-scale claim.
