param(
    [int]$WaitPid = 0,
    [string]$QueueLog = "S:/AI/work/VRAXION_DEV/v4.2/logs/gpu_backend_ab_queue.log",
    [string]$RepoRoot = "S:/AI/work/VRAXION_DEV",
    [string]$Configs = "V128_N384,V256_N768",
    [int]$Attempts = 16000,
    [string]$Seeds = "42,77,123",
    [string]$Backends = "current_guided_backend,edge_list_backend",
    [string]$LogName = "gpu_backend_ab",
    [string]$StopPids = ""
)

$ErrorActionPreference = "Stop"

function Write-QueueLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format o), $Message
    $line | Tee-Object -FilePath $QueueLog -Append
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $QueueLog) | Out-Null
Write-QueueLog "backend queue armed wait_pid=$WaitPid configs=$Configs attempts=$Attempts seeds=$Seeds backends=$Backends stop_pids=$StopPids"

if ($WaitPid -gt 0) {
    try {
        Wait-Process -Id $WaitPid -ErrorAction Stop
        Write-QueueLog "active pid=$WaitPid finished"
    }
    catch {
        Write-QueueLog "wait finished early for pid=$WaitPid : $($_.Exception.Message)"
    }
}

if ($StopPids -ne "") {
    foreach ($pidText in ($StopPids -split ",")) {
        $trimmed = $pidText.Trim()
        if ($trimmed -eq "") { continue }
        $pid = [int]$trimmed
        try {
            Stop-Process -Id $pid -Force -ErrorAction Stop
            Write-QueueLog "stopped stale pid=$pid"
        }
        catch {
            Write-QueueLog "stale pid=$pid already gone: $($_.Exception.Message)"
        }
    }
}

Set-Location $RepoRoot

$cmd = @(
    "python",
    "v4.2/tests/gpu_backend_ab.py",
    "--configs", $Configs,
    "--attempts", [string]$Attempts,
    "--seeds", $Seeds,
    "--backends", $Backends,
    "--log-name", $LogName
)

Write-QueueLog ("RUN " + ($cmd -join " "))
$exe = $cmd[0]
$argsList = $cmd[1..($cmd.Length - 1)]
& $exe @argsList 2>&1 | Tee-Object -FilePath $QueueLog -Append
$exitCode = $LASTEXITCODE
Write-QueueLog "backend queue done exit_code=$exitCode"
exit $exitCode
