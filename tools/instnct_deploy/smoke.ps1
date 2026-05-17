param(
  [string]$Config = "tools/instnct_deploy/config/example.local.json",
  [string]$Out = "target/pilot_wave/stable_loop_phase_lock_058_deployment_harness/smoke"
)

$ErrorActionPreference = "Stop"
python tools/instnct_deploy/instnct_deploy.py smoke --config $Config --out $Out
