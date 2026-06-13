# E63 Pocket Observatory Visual Debug Dashboard Contract

## Summary

E63 adds a local visual/debug dashboard for Pocket ecology artifacts. It is
designed to make Pocket calls, proposal activity, Agency decisions, lifecycle
changes, and Flow Field commits visible during future curriculum runs.

This is visualization/debug infrastructure only. It is not a new training run,
model capability claim, production dashboard/API, AGI claim, consciousness claim,
or model-scale behavior claim.

## Required Artifacts

```text
backend_manifest.json
observatory_snapshot.json
pocket_state.json
pocket_events.jsonl
flow_snapshot.json
proposal_snapshot.json
agency_decisions.jsonl
aggregate_metrics.json
decision.json
summary.json
deterministic_replay.json
progress.jsonl
hardware_heartbeat.jsonl
report.md
index.html
```

## Viewer Requirements

```text
self-contained HTML
no external CDN/network dependency
Pocket list with lifecycle/status metrics
cycle-by-pocket activity heatmap
Flow Field commit grid
Proposal/Agency timeline
manual refresh
auto-refresh
embedded sample fallback
local file-picker fallback
relative artifact fetch when served over HTTP
```

## Data Contract

Pocket rows must include:

```text
pocket_id
kind
lifecycle
calls
accepted
rejected
false_commits
utility_score
safety_score
cost_score
last_active_cycle
```

Event rows must include:

```text
cycle
pocket_id
event_type
lifecycle
utility_delta
safety_delta
label
```

Agency decision rows must include:

```text
cycle
proposal_id
pocket_id
agency_action
reason
false_commit
```

## Positive Decision

```text
e63_pocket_observatory_dashboard_ready
```

if the dashboard and artifact contract are generated, checker passes, static
checks pass, deterministic replay hashes match, and browser/render validation
can load the dashboard.

## Boundary

E63 does not train or mutate Pocket behavior. It only visualizes the runtime
artifact shape that future curriculum runs can write into.
