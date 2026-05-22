param(
  [string]$Config = "tools/instnct_service_alpha/config/example.local.json"
)

python tools/instnct_service_alpha/instnct_service_alpha.py healthcheck --config $Config
exit $LASTEXITCODE
