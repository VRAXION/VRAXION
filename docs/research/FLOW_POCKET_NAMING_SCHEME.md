# Flow/Pocket Naming Scheme

Status: canonical for new E-series Flow/Pocket research notes and probe artifacts.

## Canonical Terms

```text
Ground Field
  Stable reference field / grounding layer.
  Holds persistent state, context anchors, and long-lived constraints.

Flow Field
  Active working field.
  Holds the current evolving state that pockets read, transform, and return.

Pocket Operator
  Callable local state-transform operation.
  Reads from fields, transforms state, and may propose or perform a gated write.

Lens Pocket
  Read-only Pocket Operator.
  Detects, measures, or interprets field state without writing.

Writer Pocket
  Writeback-specialized Pocket Operator.
  Proposes or performs field updates under Arbiter control.

Arbiter
  Scheduler + router + conflict resolver + commit gate.
  Decides which pocket runs, which write is allowed, when to halt, and when the state remains unresolved.

Trace Ledger
  Audit trail for pocket calls, evidence, conflicts, writebacks, rollbacks, and commits.

Ingress Codec
  Input stream -> field event/state translation.

Egress Codec
  Field state -> output/action translation.
```

## Deprecated / Avoid

```text
transformer matrix
```

Avoid this name because it implies a neural Transformer block. Use `Pocket Operator` or `Local Field Transform` instead.

```text
Self Matrix
```

Avoid this name because it is vague and too anthropomorphic. Use `Ground Field` when referring to stable grounding state.

```text
Flow Matrix
```

Use `Flow Field` unless the artifact is literally describing a concrete matrix tensor.

## System Definition

```text
Flow/Pocket system =
an auditable state-update architecture where Pocket Operators transform an active Flow Field,
the Ground Field anchors persistent state, the Arbiter controls routing and commits,
and the Trace Ledger records evidence, conflicts, writebacks, and rollbacks.
```

## Minimal Diagram

```text
Ingress Codec
  -> Flow Field
  -> Arbiter
  -> Pocket Operator / Lens Pocket / Writer Pocket
  -> Trace Ledger
  -> Flow Field update
  -> Ground Field anchoring
  -> Egress Codec
```

## Artifact Naming

New probes should prefer these artifact names where relevant:

```text
pocket_activation_map.json
field_writeback_map.json
arbiter_decision_trace.json
trace_ledger.jsonl
conflict_map.json
unresolved_state_map.json
ground_field_snapshot.json
flow_field_snapshot.json
```
