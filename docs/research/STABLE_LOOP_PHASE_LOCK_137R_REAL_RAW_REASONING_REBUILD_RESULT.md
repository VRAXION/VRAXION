# STABLE_LOOP_PHASE_LOCK_137R_REAL_RAW_REASONING_REBUILD Result

137R implements the eval-only real-raw reasoning rebuild after positive 136R.

The runner uses only `scripts/probes/shared_raw_generation_helper.py`, records helper requests and trace hashes for every generated row, reruns forbidden-input rejection, reruns the expected-output canary, scans helper/runner/checker code for shortcut paths, audits leakage, runs scorer-only controls, and scores deterministic reasoning rows only after generated text exists.

## Decision

If all reasoning gates pass:

- `verdict = REAL_RAW_REASONING_REBUILD_POSITIVE`
- `decision = real_raw_reasoning_evidence_rebuilt`
- `next = 138R_REAL_RAW_MULTI_TURN_STATE_REBUILD`

If real raw generation works but reasoning gates fail:

- `verdict = REAL_RAW_REASONING_REBUILD_FAILS`
- `decision = real_raw_reasoning_not_restored`
- `next = 137B_REAL_RAW_REASONING_REPAIR_PLAN`

A clean negative is valid and preferred over fake restoration.

## Boundary

137R rebuilds only the reasoning subtrack if positive. Raw assistant capability remains quarantined. Structured/tool capability remains invalidated as model evidence. No full raw assistant capability is restored. It is not GPT-like readiness, not open-domain assistant readiness, not production chat, not public API, not deployment readiness, and not safety alignment.
