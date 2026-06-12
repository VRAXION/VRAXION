# Pocket ABI v1

Status: active.

Pocket ABI v1 locks the external interface for reusable Pocket Operators.
It intentionally does not lock pocket internals.

## Call Shape

```text
CALL(pocket_id, flow_or_payload, context) -> proposal/event/reject
```

The caller must treat pocket output as a proposal/event unless the pocket
contract explicitly grants commit authority.

## Required Stable/Core Entry Fields

```text
abi_version
pocket_id
version
pocket_type
input_contract
output_contract
read_mask_contract
write_mask_contract
preserve_mask_contract
side_effect_policy
requires_adapter
compatible_flow_dims
compatible_protocols
compatible_families
quantization_format
cost_estimate
trace_contract
failure_modes
known_bottlenecks
frozen_params_digest
evaluator_version
reaudit_policy
status
load_allowed
promotion_source
```

## Adapter Boundary

Mechanism pocket and world adapter are separate objects.

```text
mechanism pocket != world adapter
```

If `requires_adapter = true`, target-world import must explicitly declare an
adapter. Import without adapter declaration is a hard failure.

## Mask Contract

Mask contracts may be:

```text
absolute
symbolic
adapter_mapped
region_relative
payload_native
none
```

If a pocket is Flow layout dependent, `compatible_flow_dims` must be explicit or
the pocket must require an adapter.

## Trace Contract

A stable/core pocket must define what its trace records:

```text
activation
read/payload summary
proposed output
commit/reject reason
evidence reference
rollback/guard result if applicable
```

## Re-Audit Policy

Stable/core status is valid only for the evaluator and ABI version that promoted
the pocket. Re-audit is required when:

```text
ABI major version changes
registry schema changes
evaluator version changes
Flow/Grounding Field layout changes
quantization format changes
safety rule changes
adapter contract changes
```

Boundary: Pocket ABI v1 is infrastructure for controlled pocket generation and
library lifecycle. It is not a raw language reasoning, AGI, consciousness,
deployed-model, or model-scale claim.
