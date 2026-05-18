param(
  [string]$Out = "target/pilot_wave/stable_loop_phase_lock_064_observability_incident_backup_gate/smoke",
  [int]$HeartbeatSec = 20
)

python tools/ops_readiness/instnct_ops_gate.py --out $Out --heartbeat-sec $HeartbeatSec
