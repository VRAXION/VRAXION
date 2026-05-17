param(
  [string]$Out = "target/pilot_wave/stable_loop_phase_lock_063_security_supply_chain_gate/smoke",
  [int]$HeartbeatSec = 20
)

python tools/security_supply_chain/instnct_security_gate.py --out $Out --heartbeat-sec $HeartbeatSec
exit $LASTEXITCODE
