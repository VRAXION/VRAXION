[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
$ErrorActionPreference = 'Continue'
function QLog([string]$m) {
  $line = "[{0}] {1}" -f (Get-Date -Format o), $m
  $line | Tee-Object -FilePath 'S:/AI/work/VRAXION_DEV/v4.2/logs/gpu_strategy_plateau_fastgate_20260315_1351.log' -Append
}
QLog 'fastgate armed: wait current V128 two_bit seed42, then stop long queues and run only seed77+seed123'
try { Wait-Process -Id 24340 -ErrorAction Stop; QLog 'current run finished' } catch { QLog ('wait current run ended with: ' + $_.Exception.Message) }
foreach ($pid in @(26328,7608)) {
  try {
    Stop-Process -Id $pid -Force -ErrorAction Stop
    QLog ("stopped queue pid=" + $pid)
  } catch {
    QLog ("queue pid=" + $pid + ' already gone')
  }
}
Set-Location 'S:/AI/work/VRAXION_DEV'
foreach ($seed in @(77,123)) {
  $args = @('v4.2/tests/gpu_strategy_plateau.py','--config','V128_N384','--variant','two_bit_decoupled','--seed',[string]$seed,'--safety-cap','160000')
  QLog ('RUN python ' + ($args -join ' '))
  & python @args 2>&1 | Tee-Object -FilePath 'S:/AI/work/VRAXION_DEV/v4.2/logs/gpu_strategy_plateau_fastgate_20260315_1351.log' -Append
}
QLog 'fastgate done'
