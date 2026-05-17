# INSTNCT Service Alpha Idempotency And Rate-Limit Boundary

Status: 062 idempotency and alpha rate-limit boundary.

Idempotency is deterministic:

- same `idempotency_key` plus same request body returns the same `job_id`
- same `idempotency_key` plus different request body returns conflict

Rate limiting is alpha metadata only. It is no production rate-limit claim.

Required response fields:

```text
rate_limit_policy = alpha_static_local
rate_limit_remaining
retry_after when exceeded
```

This boundary is no production API readiness, no production deployment, no
hosted SaaS, no public beta, no multi-tenant IAM, no clinical use, no
high-stakes education use, no commercial launch, no full VRAXION, no language
grounding, no consciousness, no biological/FlyWire equivalence, and no physical
quantum behavior.

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

