<!-- OWNER_LOCAL_CI_20261027 -->
## Owner CI and preservation policy

Read `.github/CI_COST_POLICY.md` and `.github/ci_policy.json` before changing or
dispatching workflows, registering runners, uploading artifacts or cleaning up.
Run complete feasible QA locally. Routine online CI, including self-hosted
Actions, has a 60-second limit. Longer essential transfer/deployment/readback
needs a concrete disclosed reason or explicit owner exception before execution.
Private jobs require trusted repository-scoped owner-local runners, never a paid
hosted fallback. Offline means QUEUED/NOT_RUN. Do not raise spending limits,
fake a pass, weaken required tests or publish private evidence. Old long/manual
workflow timeouts are not authorization: execute their full QA locally outside
Actions before a short source-bound gate. Preserve unique evidence through a
verified private Release before shortening retention or deleting copies.
Policy: `owner-local-ci-20261027-v1`.
<!-- /OWNER_LOCAL_CI_20261027 -->

