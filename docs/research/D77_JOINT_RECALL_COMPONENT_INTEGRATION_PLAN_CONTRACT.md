# D77 Joint-Recall Component Integration Plan Contract

## Purpose

D77 plans integration of the D76 scale-confirmed `JOINT_RECALL_COMPONENT_COST_AWARE` component into the main Rust sparse ECF/IPF support-routing stack. D77 is a planning/analysis milestone only: it selects an integration surface, defines the component interface, and specifies D78 proof gates. It does not add a new broad architecture claim.

## Phase 0 upstream audit

The runner must verify the current branch and HEAD, check whether D76 commit `b0a5f3986441ef566ed83e74163b5e34d212bed2` is present and an ancestor of `HEAD`, and write `d76_upstream_manifest.json`. If the requested D76 commit or D76 artifacts are missing, the runner must restore/rerun D76 before planning D77 and must record the restore/rerun status explicitly. D77 must not silently assume D76 was pushed.

## D76 handoff reference

D76 scale-confirmed the component with:

- `decision=joint_recall_component_scale_confirmed`
- `next=D77_JOINT_RECALL_COMPONENT_INTEGRATION_PLAN`
- `scaled_arm=D75_JOINT_RECALL_COST_AWARE_REPLAY`
- `average_total_support_used=6.6515`
- `distance_to_concrete_oracle_support=0.3315`
- `gap_reduction_vs_D73_bound=0.1605`
- `exact_joint_accuracy=0.99916`
- `correlated_echo_accuracy=0.9964`
- `adversarial_distractor_accuracy=0.9961`
- `external_test_required_accuracy=0.9959`
- `joint_counter_recall_on_joint_required_rows=0.9943`
- `external_recall_on_external_required_rows=0.9959`
- `wrong_concrete_counter_rate=0.0006`
- `weak_top1_top2_path_failure_rate=0.0005`
- `top1_top2_sufficient_false_joint_rate=0.0011`
- `false_confidence_rate=0.0043`
- `indistinguishable_abstain_rate=0.9949`
- `D68_loss_repair_preservation_rate=1.0`
- `routing_failure_rows=0`
- `rust_path_invoked=true`
- `fallback_rows=0`
- `failed_jobs=[]`

## Planning questions

1. Where should the joint-recall component live in the ECF/IPF controller stack?
2. Which inputs does it consume?
3. Which actions/gates does it influence?
4. How does it avoid D68 cheap-top1 regression?
5. What should D78 prove in an integrated run?
6. Should the next milestone be integration prototype, cost repair, or another component migration?

## Integration candidates

1. `PRE_GATE_JOINT_RECALL_SCORER`
2. `POLICY_GATE_JOINT_RECALL_FEATURE`
3. `COUNTER_ACTION_ROUTER_JOINT_RECALL_MODULE`
4. `POSTCHECK_JOINT_RECALL_ESCALATION`
5. `HYBRID_JOINT_RECALL_AND_EXTERNAL_ROUTING`
6. `JOINT_RECALL_AS_RUST_SPARSE_DIAGNOSTIC_COMPONENT`

## Required reports

Artifacts are written under `target/pilot_wave/d77_joint_recall_component_integration_plan/`:

- `d76_upstream_manifest.json`
- `integration_target_selection_report.json`
- `component_interface_report.json`
- `required_input_feature_report.json`
- `action_influence_report.json`
- `D68_regression_prevention_report.json`
- `Rust_sparse_integration_surface_report.json`
- `D78_proof_gate_report.json`
- `risk_register.json`
- `truth_leak_audit_report.json`
- `aggregate_metrics.json`
- `decision.json`
- `summary.json`
- `report.md`

## Planning metrics

D77 planning metrics must be labeled as estimates, not empirical D77 performance results:

- expected support effect
- expected oracle gap effect
- implementation complexity
- D68 regression risk
- safety margin risk
- dependency on symbolic stack
- dependency on Rust sparse path
- required truth-leak guards
- D78 measurable gates
- integration ROI estimate

## Decisions

- If a single best integration surface is clear: `decision=joint_recall_integration_plan_selected`, `next=D78_JOINT_RECALL_INTEGRATED_CONTROLLER_PROTOTYPE`.
- If Rust sparse diagnostic migration is required first: `decision=joint_recall_requires_sparse_diagnostic_integration`, `next=D78_JOINT_RECALL_RUST_DIAGNOSTIC_INTEGRATION`.
- If external routing must be integrated jointly: `decision=joint_recall_external_combo_plan_selected`, `next=D78_JOINT_EXTERNAL_INTEGRATED_CONTROLLER_PROTOTYPE`.
- If plan is not ready: `decision=joint_recall_integration_plan_not_ready`, `next=D77_REPAIR`.

## Hard gates

- No full brain, Raven, AGI, consciousness, DNA/genome, architecture-superiority, or production-readiness claims.
- No fake empirical metrics; D77 planning values are estimates and D78 gates are explicit measurable requirements.
- No label echo fair oracle.
- Truth hidden from fair arms.
- D68 cheap-top1 prevention must be explicit.
- D78 proof gates must be concrete and measurable.
- `failed_jobs` must be visible.

## Boundary

D77 only plans integration of the scale-confirmed joint-recall component in controlled symbolic ECF/IPF joint formula discovery. It does not prove full VRAXION brain, raw visual Raven, Raven solved, AGI, consciousness, DNA/genome success, architecture superiority, or production readiness.
