# STABLE_LOOP_PHASE_LOCK_138I_REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_PROBE Result

## Status

138I implements a deterministic targeted rollout-aligned real-raw reasoning
repair/probe. The runner trains only a new target checkpoint under `target/` and
final-evaluates through `scripts/probes/shared_raw_generation_helper.py`.

The result is intentionally bounded. A positive result may partially restore
only reasoning subtrack real-raw evidence. Raw assistant capability remains
quarantined. Structured/tool capability remains invalidated as model evidence.
This is not GPT-like readiness, not open-domain assistant readiness, not
production chat, not public API, not deployment readiness, and not safety
alignment.

## Implemented Gates

- verifies 138H, 138GA, and 138R upstream artifacts
- confirms the 138H bottleneck: `train_namespace_rollout_alignment_failure`
- preserves source checkpoint immutability
- trains only a helper-compatible target byte-GRU checkpoint under `target/`
- writes train/eval split manifests and leakage audit
- records namespace metrics for `ANSWER=T...` leakage and `ANSWER=E...` emission
- rejects checkpoint-change-only, train-loss-only, and `ANSWER=`-prefix-only success
- reruns forbidden-input rejection and expected-output canary
- scans helper, runner, and checker for shortcut patterns
- records helper provenance and strict checkpoint key/shape evidence
- proves `generated_text` is produced before scoring
- uses deterministic scorer controls that do not call the helper/model
- includes `TRAIN_NAMESPACE_REPLAY_CONTROL`
- reruns final eval for deterministic replay identity
- keeps all broad capability and readiness claims false

## Decision Meaning

If 138I emits `REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_POSITIVE`, the target
checkpoint improved helper-only rollout, suppressed `ANSWER=T...` train
namespace leakage under the declared floor, improved `ANSWER=E...` eval
namespace emission, improved answer-value accuracy, and passed canary, AST,
provenance, leakage, controls, generated-before-scoring, and determinism gates.

If 138I emits `REAL_RAW_REASONING_ROLLOUT_ALIGNED_REPAIR_FAILS`, the clean
negative is valid. It must not trigger threshold weakening, helper changes,
namespace leakage acceptance, or loss-only success claims.

If 138I emits `ROLLOUT_ALIGNED_TRAINING_PATH_MISSING`, no safe helper-compatible
rollout-aligned training path was available and the correct next step is
`138IA_ROLLOUT_ALIGNED_TRAINING_HELPER_INTEGRATION_PLAN`.

If 138I emits `DETERMINISM_REPLAY_MISMATCH`, the repair/probe evidence is not
stable enough for capability claims.

Namespace leak persistence routes to
`138S_NAMESPACE_ROLLOUT_FAILURE_ANALYSIS`. No rollout improvement routes to
`138I_FAILURE_ANALYSIS`.

## Boundary Reminder

138I does not modify `shared_raw_generation_helper.py`, old runners, source
checkpoints, runtime/service/deploy surfaces, product or release docs, SDK
exports, docs/product, docs/releases, or root `LICENSE`.

Raw assistant capability remains quarantined. Structured/tool capability
remains invalidated. Not GPT-like readiness. Not open-domain assistant
readiness. Not production chat. Not public API. Not deployment readiness. Not
safety alignment.
