# INSTNCT Service API Alpha Schema

Status: 062 API v1 alpha schema.

The schema version is:

```text
instnct_service_api_alpha_v1
```

Every route returns a structured envelope:

```json
{
  "schema_version": "instnct_service_api_alpha_v1",
  "ok": true,
  "request_id": "req_example",
  "claim_boundary": [],
  "rate_limit_policy": "alpha_static_local",
  "rate_limit_remaining": 127,
  "value": {}
}
```

Every error returns:

```json
{
  "schema_version": "instnct_service_api_alpha_v1",
  "ok": false,
  "request_id": "req_example",
  "claim_boundary": [],
  "rate_limit_policy": "alpha_static_local",
  "rate_limit_remaining": 126,
  "error": {
    "code": "POLICY_GUARD_REJECTED",
    "message": "policy guard rejected request",
    "retryable": false,
    "details": {}
  }
}
```

The envelope is alpha-only. It is no production API readiness, no production
deployment, no hosted SaaS, no public beta, no multi-tenant IAM, no clinical
use, no high-stakes education use, no commercial launch, no full VRAXION, no
language grounding, no consciousness, no biological/FlyWire equivalence, and no
physical quantum behavior.

## Boundary

Exact boundary tokens:

```text
no production API readiness
no production deployment
no hosted SaaS
no public beta
no multi-tenant IAM
no clinical use
no high-stakes education use
no commercial launch
no full VRAXION
no language grounding
no consciousness
no biological/FlyWire equivalence
no physical quantum behavior
```

