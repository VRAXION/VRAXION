$logPath = "S:\AI\mirror\VRAXION\logs\tier2_8_4_surge.log"
Write-Host ">>> Tailing $logPath" -ForegroundColor Green
Get-Content -Path $logPath -Wait -Tail 20
