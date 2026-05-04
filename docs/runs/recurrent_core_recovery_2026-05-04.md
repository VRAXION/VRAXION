# Recurrent Core Recovery Finding - 2026-05-04

## Source Runs

- Core-recovery run: `target/context-cancellation-probe/20260504T185308Z/context_cancellation_report.json`
- Nuisance-influence run: `target/context-cancellation-probe/20260504T191155Z/context_cancellation_report.json`
- Pressure run: `target/context-cancellation-probe/20260504T191339Z/context_cancellation_report.json`
- Script: `scripts/run_context_cancellation_probe.py`

The `target/` reports are local run artifacts and are not committed by default. This note records the durable result in a tracked repo document.

## Original Hypothesis

The first probe targeted **Recurrent Context Cancellation**:

> A sparse recurrent edge+threshold-style system may learn an iterative cleanup dynamic where task-irrelevant context or nuisance features are suppressed over recurrent steps while the causal core stays active or becomes more stable.

## Updated Hypothesis

The v4 result supports a narrower mechanism:

> **Recurrent Core Recovery under Entangled Interference**

In this framing, the recurrent loop does not primarily erase nuisance context. Instead, it recovers and amplifies the task-causal core from an entangled core+nuisance representation until the core dominates the decision state.

## Key Result

On label-only entangled input:

- recurrent model accuracy: `0.99975`
- zero-recurrent baseline accuracy: `0.6395`
- recurrence gain: `+0.36025`

The recurrent update is therefore not decorative. The model needs the learned recurrent dynamics to solve the entangled task reliably.

## Mechanism Evidence

- core probe delta: `+0.1885`
- nuisance probe delta: `-0.042`
- core-to-nuisance ratio: `0.4608 -> 1.5534`
- output entropy: `0.3278 -> 0.0038`
- logit margin: `2.8643 -> 12.9283`
- freeze-after-step-1 accuracy: `0.69225`
- randomized recurrent matrix accuracy: `0.52375`
- random-label control accuracy: `0.505`

Interpretation:

- Core decodability rises strongly across recurrent steps.
- Nuisance remains weakly decodable, so this is not clean erasure.
- The relative dominance of core over nuisance rises by more than 3x.
- Output uncertainty collapses and decision margin grows over recurrent steps.
- Freezing recurrence early hurts accuracy.
- Randomizing the recurrent matrix destroys the gain.
- Random labels do not produce a meaningful learned result.

## Nuisance Influence

The follow-up intervention tested whether still-decodable nuisance retained decision authority.

| Intervention | Label Change | Output Change | Target Accuracy | Mean Abs Label-Prob Delta | Mean KL |
|---|---:|---:|---:|---:|---:|
| Same core, different nuisance | `0.0` | `0.00125` | `0.999` | `0.002187` | `0.005354` |
| Same nuisance, different core | `0.4935` | `0.4925` | `0.99825` | `0.492809` | `5.880728` |

This supports the sharper interpretation:

> Nuisance remains weakly decodable, but loses decision authority after recurrent core recovery.

Changing nuisance while preserving the core almost never changes the output. Changing the core while preserving nuisance changes the output at almost exactly the expected counterfactual label-change rate.

## Pressure Run

Under stronger bottleneck pressure (`hidden=16`, `update_rate=0.1`), the pattern weakened:

| Intervention | Label Change | Output Change | Target Accuracy | Mean Abs Label-Prob Delta | Mean KL |
|---|---:|---:|---:|---:|---:|
| Same core, different nuisance | `0.0` | `0.26475` | `0.771` | `0.212475` | `0.264992` |
| Same nuisance, different core | `0.4935` | `0.45` | `0.75975` | `0.343555` | `0.701513` |

This pressure setting remains `unclear`: recurrence helps, but the representation is too compressed to cleanly isolate core authority from nuisance influence.

## Interpretation

This toy experiment supports **recurrent core recovery**, not strong context cancellation.

The useful reading is:

> The recurrent loop does not delete nuisance. It recovers a core-dominant decision state where nuisance remains weakly decodable but has almost no decision authority.

## Claim Boundary

Safe claim:

> In a controlled toy setting, a sparse recurrent label-only model can recover task-causal core information from entangled core+nuisance inputs. In the full setting, that recovered core makes the decision almost invariant to nuisance changes while remaining highly sensitive to core changes.

Do not claim:

- consciousness,
- full VRAXION behavior,
- production architecture validation,
- clean nuisance cancellation,
- biological equivalence.
