[Console]::OutputEncoding=[System.Text.Encoding]::UTF8
Continue = 'Continue'
function QLog([string]) {
   = "[{0}] {1}" -f (Get-Date -Format o), 
   | Tee-Object -FilePath 'S:/AI/work/VRAXION_DEV/v4.2/logs/gpu_strategy_plateau_fastgate_20260315_1339.log' -Append
}
QLog 'fastgate armed: wait current V128 two_bit seed42, then stop long queues and run only seed77+seed123'
try { Wait-Process -Id 24340 -ErrorAction Stop } catch { QLog ("wait current run ended with: " + .Exception.Message) }
foreach (51568 in @(26328,7608)) {
  try {
    if (Get-Process -Id 51568 -ErrorAction Stop) {
      Stop-Process -Id 51568 -Force -ErrorAction Stop
      QLog "stopped queue pid=51568"
    }
  } catch {
    QLog "queue pid=51568 already gone"
  }
}
Set-Location 'S:/AI/work/VRAXION_DEV'
 = @(
  @{config='V128_N384'; variant='two_bit_decoupled'; seed=77; safety=160000},
  @{config='V128_N384'; variant='two_bit_decoupled'; seed=123; safety=160000}
)
foreach ( in ) {
   = @('python','v4.2/tests/gpu_strategy_plateau.py','--config',.config,'--variant',.variant,'--seed',[string].seed,'--safety-cap',[string].safety)
  QLog ('RUN ' + ( -join ' '))
  & [0] @([1..(.Length-1)]) 2>&1 | Tee-Object -FilePath 'S:/AI/work/VRAXION_DEV/v4.2/logs/gpu_strategy_plateau_fastgate_20260315_1339.log' -Append
}
QLog 'fastgate done'
