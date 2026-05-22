param(
  [string]$Config = "tools/instnct_service_alpha/config/example.local.json",
  [string]$Out = "target/pilot_wave/stable_loop_phase_lock_062_service_api_alpha/smoke"
)

python tools/instnct_service_alpha/instnct_service_alpha.py smoke --config $Config --out $Out
exit $LASTEXITCODE
