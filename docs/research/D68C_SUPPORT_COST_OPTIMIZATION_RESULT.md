# D68C Support Cost Optimization Result

D68C implements a bounded support-cost optimization probe that replays D67/D68/D68R provenance and evaluates cost-aware concrete counter-action routers against the D68R repair gates.

Smoke validation command:

```bash
python scripts/probes/run_d68c_support_cost_optimization.py --out target/pilot_wave/d68c_support_cost_optimization/smoke --seeds 13001,13002,13003,13004,13005 --train-rows-per-seed 240 --test-rows-per-seed 240 --ood-rows-per-seed 240 --workers auto --cpu-target 50-75 --heartbeat-sec 20
python scripts/probes/run_d68c_support_cost_optimization_check.py --check-only --out target/pilot_wave/d68c_support_cost_optimization/smoke
```

Expected decision labels:

- `support_cost_optimization_confirmed` -> `D69_SUPPORT_COST_OPTIMIZATION_SCALE_CONFIRM`
- `counter_action_routing_stable_high_cost` -> `D68C_SUPPORT_COST_SEARCH_EXPANSION`
- `support_cost_optimization_recall_failure` -> `D68J_JOINT_COUNTER_RECALL_REPAIR`
- `support_cost_optimization_safety_failure` -> `D68S_EXTERNAL_ABSTAIN_SAFETY_REPAIR`
- `support_cost_optimization_not_confirmed` -> `D68C_REPAIR`

The authoritative metrics are the JSON artifacts emitted under `target/pilot_wave/d68c_support_cost_optimization/smoke/` during the smoke run. The probe reports whether handoff artifacts were restored, which arm is the best fair support-cost arm, whether D68 loss rows remain repaired, whether D68's weak top1/top2 failure mode returned, and how far the best fair support cost remains from the concrete oracle.

Boundary: D68C only optimizes support cost after concrete counter-action routing repair in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven reasoning, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
