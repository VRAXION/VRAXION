param(
  [string]$Config = "tools/instnct_service_alpha/config/example.local.json"
)

python tools/instnct_service_alpha/instnct_service_alpha.py serve --config $Config
exit $LASTEXITCODE
