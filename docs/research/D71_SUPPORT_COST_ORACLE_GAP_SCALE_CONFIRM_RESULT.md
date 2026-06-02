# D71 Support Cost Oracle Gap Scale Confirm Result

D71 records the scale-confirm run for D70's oracle-gap support-cost reduction.

Authoritative artifacts are emitted under:

`target/pilot_wave/d71_support_cost_oracle_gap_scale_confirm/smoke/`

Validation:

- `python -m py_compile scripts/probes/run_d71_support_cost_oracle_gap_scale_confirm.py`
- `python -m py_compile scripts/probes/run_d71_support_cost_oracle_gap_scale_confirm_check.py`
- `python scripts/probes/run_d71_support_cost_oracle_gap_scale_confirm.py --out target/pilot_wave/d71_support_cost_oracle_gap_scale_confirm/smoke --seeds 13301,13302,13303,13304,13305,13306,13307,13308 --train-rows-per-seed 240 --test-rows-per-seed 240 --ood-rows-per-seed 240 --workers auto --cpu-target 50-75 --heartbeat-sec 20`
- `python scripts/probes/run_d71_support_cost_oracle_gap_scale_confirm_check.py --check-only --out target/pilot_wave/d71_support_cost_oracle_gap_scale_confirm/smoke`

Expected decision labels:

- `support_cost_oracle_gap_scale_confirmed` -> `D72_SUPPORT_COST_ORACLE_GAP_BOUND_ANALYSIS`
- `oracle_gap_scale_confirmed_safety_margin_watch` -> `D71S_SAFETY_MARGIN_REPAIR`
- `oracle_gap_reduction_not_scale_stable` -> `D71_REPAIR`
- `oracle_gap_routing_regression` -> `D68J_JOINT_COUNTER_RECALL_REPAIR`

Boundary: D71 only scale-confirms D70 support-cost oracle-gap reduction in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
