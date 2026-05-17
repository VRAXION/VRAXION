# INSTNCT SDK Error Envelope

Status: 057 SDK release-candidate engineering artifact.

All SDK candidate failures return structured error envelopes. Success and error
responses both include the claim boundary.

## Envelope Shape

```json
{
  "schema_version": "instnct_sdk_candidate_v1",
  "code": "UNKNOWN_SCHEMA_VERSION",
  "message": "unknown SDK candidate schema version",
  "retryable": false,
  "details": {},
  "claim_boundary": {}
}
```

## Exact Error Codes

```text
UNKNOWN_SCHEMA_VERSION
INVALID_INPUT
CHECKPOINT_HASH_MISMATCH
POLICY_GUARD_REJECTED
PRODUCTION_FLAG_CONTAMINATION
IO_ERROR
VISUAL_EXPORT_ERROR
```

## Policy Guard Errors

Clinical and high-stakes education requests must return
`POLICY_GUARD_REJECTED` before checkpoint or visual side effects.

Production flag contamination must return `PRODUCTION_FLAG_CONTAMINATION`.

## Claim Boundary

057 supports SDK release-candidate engineering only. It does not support
production API readiness, public beta, clinical use, high-stakes education use,
full VRAXION, language grounding, consciousness, biological/FlyWire equivalence,
or physical quantum behavior.

Exact boundary tokens:

```text
no production API readiness
no public beta
no clinical use
no high-stakes education use
no full VRAXION
no language grounding
no consciousness
no biological/FlyWire equivalence
no physical quantum behavior
```
