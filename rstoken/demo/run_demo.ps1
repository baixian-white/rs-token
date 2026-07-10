param(
    [int]$Port = 7860,
    [string]$Device = "cuda",
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$DemoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $DemoRoot

function Find-Python {
    $candidates = @(
        "C:\Users\Administrator\miniconda3\envs\rstoken\python.exe",
        "C:\Users\Windows11\.conda\envs\rstoken\python.exe"
    )
    if ($env:CONDA_DEFAULT_ENV -eq "rstoken" -and $env:CONDA_PREFIX) {
        $candidates = @((Join-Path $env:CONDA_PREFIX "python.exe")) + $candidates
    }
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) { return $candidate }
    }
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    throw "No Python runtime found. Activate the rstoken Conda environment first."
}

$Python = Find-Python
$Probe = Join-Path $DemoRoot "artifacts\h0_probe.joblib"
if (-not (Test-Path -LiteralPath $Probe)) {
    & (Join-Path $DemoRoot "setup.ps1") -Device $Device
}

$Runtime = Join-Path $DemoRoot ".runtime"
New-Item -ItemType Directory -Path $Runtime -Force | Out-Null
$Stdout = Join-Path $Runtime "server.stdout.log"
$Stderr = Join-Path $Runtime "server.stderr.log"
$PidFile = Join-Path $Runtime "server.pid"
$PortFile = Join-Path $Runtime "server.port"

if ((Test-Path -LiteralPath $PidFile) -and (Test-Path -LiteralPath $PortFile)) {
    $ExistingPid = [int](Get-Content -LiteralPath $PidFile -Raw)
    $ExistingPort = [int](Get-Content -LiteralPath $PortFile -Raw)
    $ExistingProcess = Get-Process -Id $ExistingPid -ErrorAction SilentlyContinue
    if ($ExistingProcess) {
        $ExistingUrl = "http://127.0.0.1:$ExistingPort"
        try {
            $ExistingHealth = Invoke-RestMethod -Uri "$ExistingUrl/api/health" -TimeoutSec 2
            if ($ExistingHealth.status -eq "ready") {
                Write-Host "[demo] Already running: $ExistingUrl (PID $ExistingPid)"
                if (-not $NoBrowser) { Start-Process $ExistingUrl }
                exit 0
            }
        }
        catch { }
    }
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $PortFile -Force -ErrorAction SilentlyContinue
}

$UsedPorts = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty LocalPort)
while ($UsedPorts -contains $Port) { $Port += 1 }

$env:RSTOKEN_DEMO_DEVICE = $Device
$Url = "http://127.0.0.1:$Port"

$arguments = @("-m", "uvicorn", "demo.backend.app:app", "--host", "127.0.0.1", "--port", "$Port")
$process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $ProjectRoot -PassThru -WindowStyle Hidden -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
$process.Id | Set-Content -Path $PidFile -Encoding ascii
$Port | Set-Content -Path $PortFile -Encoding ascii

Write-Host "[demo] Loading RS-Token on $Device (PID $($process.Id))..."
$ready = $false
for ($attempt = 0; $attempt -lt 120; $attempt++) {
    if ($process.HasExited) { break }
    try {
        $health = Invoke-RestMethod -Uri "$Url/api/health" -TimeoutSec 2
        if ($health.status -eq "ready") { $ready = $true; break }
    }
    catch { }
    Start-Sleep -Seconds 1
}

if (-not $ready) {
    Write-Host "[demo] Startup failed. See $Stderr"
    if (Test-Path $Stderr) { Get-Content $Stderr -Tail 30 }
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $PortFile -Force -ErrorAction SilentlyContinue
    exit 1
}

Write-Host "[demo] RS-Token adaptive link console: $Url"
if (-not $NoBrowser) { Start-Process $Url }
