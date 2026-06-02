# D69 Support Cost Optimization Scale Confirm Result

D69 implements a bounded scale-confirm probe for the D68C support-cost optimization result. It expands the D68C replay into a larger seed/OOD scale, compares D67/D68/D68R/D68C/variant/control arms, and keeps safety regression visible.

Validation command:

```bash
python scripts/probes/run_d69_support_cost_optimization_scale_confirm.py --out target/pilot_wave/d69_support_cost_optimization_scale_confirm/smoke --seeds 13101,13102,13103,13104,13105,13106,13107,13108 --train-rows-per-seed 240 --test-rows-per-seed 240 --ood-rows-per-seed 240 --workers auto --cpu-target 50-75 --heartbeat-sec 20
python scripts/probes/run_d69_support_cost_optimization_scale_confirm_check.py --check-only --out target/pilot_wave/d69_support_cost_optimization_scale_confirm/smoke
```

Expected decision labels:

- `support_cost_optimization_scale_confirmed` -> `D70_SUPPORT_COST_ORACLE_GAP_REDUCTION`
- `support_cost_scale_confirmed_safety_margin_watch` -> `D69S_SAFETY_MARGIN_REPAIR`
- `support_cost_optimization_not_scale_stable` -> `D69_REPAIR`
- `support_cost_optimization_routing_regression` -> `D68J_JOINT_COUNTER_RECALL_REPAIR`

The authoritative metrics are the emitted JSON artifacts under `target/pilot_wave/d69_support_cost_optimization_scale_confirm/smoke/`. The report must include scale mode, arm table, support-cost frontier, oracle distance, safety regression status, D68 loss preservation, concrete routing metrics, Rust invocation/fallback status, decision, failed jobs, and compact JSON.

Boundary: D69 only scale-confirms D68C support-cost optimization after concrete counter-action routing repair in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
