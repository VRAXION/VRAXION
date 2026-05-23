# STABLE_LOOP_PHASE_LOCK_138YQ_INSTNCT_POCKET_GATED_VALUE_GROUNDING_PROBE

138YQ is the targeted helper/backend probe after 138YP.

The probe tests strict pocket-gated value grounding:

- the manifest sets `value_selection_requires_open_pocket = true`
- visible value bypass is forbidden
- the main arm opens `GATE:POCKET_OPEN`
- the ablation arm uses `GATE:NEVER_OPEN`
- closed-pocket and visible-value-bypass controls must fail
- value emission must come from `open_pocket_writeback`

Positive requires:

- main answer value accuracy at or above `0.25`
- main pocket writeback rate at or above `0.95`
- main phase transport success rate at or above `0.95`
- ablation answer value accuracy at or below `0.05`
- ablation delta at or above `0.20`
- deterministic replay
- helper request keys only
- no expected/scorer/oracle metadata in helper requests

138YQ may modify only strict backend dispatch semantics in
`scripts/probes/shared_raw_generation_helper.py`; it does not change public
request keys or public API surfaces.

This is constrained pocket-gated adapter evidence. It is not GPT-like readiness
and not broad architecture superiority.
