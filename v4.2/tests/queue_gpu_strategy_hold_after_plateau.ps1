param(
    [Parameter(Mandatory = $true)][int]$WaitPid,
    [Parameter(Mandatory = $true)][string]$QueueLog,
    [string]$RepoRoot = "S:/AI/work/VRAXION_DEV",
    [string]$Configs = "V128_N384,V256_N768",
    [int]$Attempts = 16000,
    [string]$Seeds = "42,77,123",
    [string]$Candidates = "two_bit_random35,two_bit_hold25,two_bit_hold50,two_bit_hold100,two_bit_hold200",
    [string]$LogName = "gpu_strategy_hold_sweep"
)

$ErrorActionPreference = "Stop"

function Write-QueueLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format o), $Message
    $line | Tee-Object -FilePath $QueueLog -Append
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $QueueLog) | Out-Null
Write-QueueLog "hold queue armed wait_pid=$WaitPid configs=$Configs attempts=$Attempts seeds=$Seeds candidates=$Candidates"

try {
    Wait-Process -Id $WaitPid -ErrorAction Stop
    Write-QueueLog "plateau queue pid=$WaitPid finished"
}
catch {
    Write-QueueLog "wait finished early for pid=$WaitPid : $($_.Exception.Message)"
}

Set-Location $RepoRoot

$cmd = @(
    "python",
    "v4.2/tests/gpu_strategy_hold_sweep.py",
    "--configs", $Configs,
    "--attempts", [string]$Attempts,
    "--seeds", $Seeds,
    "--candidates", $Candidates,
    "--log-name", $LogName
)

Write-QueueLog ("RUN " + ($cmd -join " "))
$exe = $cmd[0]
$argsList = $cmd[1..($cmd.Length - 1)]
& $exe @argsList 2>&1 | Tee-Object -FilePath $QueueLog -Append
$exitCode = $LASTEXITCODE
Write-QueueLog "hold queue done exit_code=$exitCode"
exit $exitCode
