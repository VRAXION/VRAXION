param(
  [string]$Config = "tools/instnct_deploy/config/example.local.json",
  [string]$Out = ""
)

$ErrorActionPreference = "Stop"
$argsList = @("tools/instnct_deploy/instnct_deploy.py", "run-local", "--config", $Config)
if ($Out -ne "") {
  $argsList += @("--out", $Out)
}
python @argsList
