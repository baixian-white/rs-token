param(
    [string]$Device = "cuda",
    [switch]$RebuildProbe
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
Write-Host "[setup] Python: $Python"
& $Python -m pip install -r (Join-Path $DemoRoot "requirements.txt")

$Probe = Join-Path $DemoRoot "artifacts\h0_probe.joblib"
if ($RebuildProbe -or -not (Test-Path -LiteralPath $Probe)) {
    Write-Host "[setup] Exporting the frozen L0 probe. This is a one-time GPU job."
    Push-Location $ProjectRoot
    try {
        & $Python scripts\17_export_demo_probe.py --device $Device
    }
    finally {
        Pop-Location
    }
}

$StaticIndex = Join-Path $DemoRoot "static\index.html"
if (-not (Test-Path -LiteralPath $StaticIndex)) {
    throw "Frontend build is missing at $StaticIndex. Build demo/frontend before running the server."
}

Write-Host "[setup] Demo assets are ready."
