# D70 Support Cost Oracle Gap Reduction Result

D70 implements a bounded oracle-gap reduction probe after D69 scale-confirmed D68C support-cost optimization. It compares D69 replay, safe deescalation variants, targeted oracle-gap routers, controls, and reference-only oracle/sentinel arms while preserving routing and safety gates.

Validation command:

```bash
python scripts/probes/run_d70_support_cost_oracle_gap_reduction.py --out target/pilot_wave/d70_support_cost_oracle_gap_reduction/smoke --seeds 13201,13202,13203,13204,13205 --train-rows-per-seed 240 --test-rows-per-seed 240 --ood-rows-per-seed 240 --workers auto --cpu-target 50-75 --heartbeat-sec 20
python scripts/probes/run_d70_support_cost_oracle_gap_reduction_check.py --check-only --out target/pilot_wave/d70_support_cost_oracle_gap_reduction/smoke
```

Expected decision labels:

- `support_cost_oracle_gap_reduction_confirmed` -> `D71_SUPPORT_COST_ORACLE_GAP_SCALE_CONFIRM`
- `oracle_gap_reduction_not_found_high_recall_bound` -> `D70B_ORACLE_GAP_BOUND_ANALYSIS`
- `oracle_gap_reduction_regression` -> `D70R_ROUTING_SAFETY_REPAIR`
- `oracle_gap_reduction_safety_failure` -> `D70S_SAFETY_MARGIN_REPAIR`

The authoritative metrics are emitted under `target/pilot_wave/d70_support_cost_oracle_gap_reduction/smoke/`. Required final reporting includes D69 availability/bootstrap status, arm table, best arm metrics, support-cost/oracle frontier, D68 loss preservation, joint/external recall, safety margin watch, truth-leak audit, Rust invocation/fallback, decision/next, failed jobs, and compact JSON.

Boundary: D70 only reduces support-cost oracle gap after D69 scale-confirmed support-cost optimization in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
