$ErrorActionPreference = "Stop"
$DemoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PidFile = Join-Path $DemoRoot ".runtime\server.pid"
$PortFile = Join-Path $DemoRoot ".runtime\server.port"

if (-not (Test-Path -LiteralPath $PidFile)) {
    Write-Host "[demo] No recorded server process."
    exit 0
}

$ServerPid = [int](Get-Content -LiteralPath $PidFile -Raw)
$process = Get-Process -Id $ServerPid -ErrorAction SilentlyContinue
if ($process) {
    Stop-Process -Id $ServerPid
    $process.WaitForExit(10000) | Out-Null
    Write-Host "[demo] Stopped server PID $ServerPid."
}
else {
    Write-Host "[demo] Recorded server PID $ServerPid is no longer running."
}
Remove-Item -LiteralPath $PidFile -Force
Remove-Item -LiteralPath $PortFile -Force -ErrorAction SilentlyContinue
