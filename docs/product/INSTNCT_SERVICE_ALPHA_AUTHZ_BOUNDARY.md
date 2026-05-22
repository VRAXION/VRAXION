# INSTNCT Service Alpha Authz Boundary

Status: 062 localhost alpha auth/authz boundary.

The service alpha requires a local static bearer token for every non-health
route. `GET /v1/health` is the only route that does not require the token.

Auth must run before side effects. Missing or invalid bearer tokens must reject
before:

- job creation
- subprocess launch
- checkpoint write
- visual export
- artifact access

The alpha has no multi-tenant IAM.

No production API readiness.
No production deployment.
No hosted SaaS.
No public beta.
No clinical use.
No high-stakes education use.
No commercial launch.
No full VRAXION.
No language grounding.
No consciousness.
No biological/FlyWire equivalence.
No physical quantum behavior.

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
