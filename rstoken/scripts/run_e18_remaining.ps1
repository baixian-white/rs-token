$ErrorActionPreference = "Continue"

$py = "C:\Users\Windows11\.conda\envs\rstoken\python.exe"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

New-Item -ItemType Directory -Force -Path "logs/seed_sweep" | Out-Null

$statusPath = "logs/seed_sweep/e18_remaining_status.csv"
"timestamp,stage,run_name,command,exit_code,log_path,note" | Set-Content -Encoding UTF8 $statusPath

function Add-Status {
    param(
        [string]$Stage,
        [string]$RunName,
        [string]$Command,
        [int]$ExitCode,
        [string]$LogPath,
        [string]$Note
    )
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $safeCommand = $Command.Replace('"', '""')
    $safeNote = $Note.Replace('"', '""')
    "$ts,$Stage,$RunName,""$safeCommand"",$ExitCode,$LogPath,""$safeNote""" | Add-Content -Encoding UTF8 $statusPath
}

function Run-Logged {
    param(
        [string]$Stage,
        [string]$RunName,
        [string[]]$ArgsList,
        [string]$LogPath,
        [string]$Note = ""
    )
    $cmdText = "$py " + ($ArgsList -join " ")
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] START $Stage $RunName"
    & $py @ArgsList *> $LogPath
    $code = $LASTEXITCODE
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] END   $Stage $RunName exit=$code"
    Add-Status -Stage $Stage -RunName $RunName -Command $cmdText -ExitCode $code -LogPath $LogPath -Note $Note
    return $code
}

function Eval-Run {
    param([string]$RunName)
    $ckpt = "checkpoints/$RunName/best.pt"
    if (-not (Test-Path $ckpt)) {
        Add-Status -Stage "eval_skip" -RunName $RunName -Command "" -ExitCode 1 -LogPath "" -Note "missing checkpoint $ckpt"
        return
    }
    Run-Logged -Stage "l0_linear" -RunName $RunName -ArgsList @("-W", "ignore", "-X", "utf8", "scripts/04_eval_l0_linear.py", "--ckpt", $ckpt) -LogPath "logs/seed_sweep/${RunName}_l0.log"
    Run-Logged -Stage "channel_awgn" -RunName $RunName -ArgsList @("-W", "ignore", "-X", "utf8", "scripts/05_eval_channel.py", "--ckpt", $ckpt, "--snrs", "5,10,inf", "--ks", "1", "--channel_type", "awgn", "--channel_seed", "0", "--out_csv", "logs/seed_sweep/${RunName}_awgn_cs0.csv") -LogPath "logs/seed_sweep/${RunName}_awgn_cs0.log"
    Run-Logged -Stage "channel_rayleigh" -RunName $RunName -ArgsList @("-W", "ignore", "-X", "utf8", "scripts/05_eval_channel.py", "--ckpt", $ckpt, "--snrs", "5,10,inf", "--ks", "1", "--channel_type", "rayleigh", "--channel_seed", "0", "--out_csv", "logs/seed_sweep/${RunName}_rayleigh_cs0.csv") -LogPath "logs/seed_sweep/${RunName}_rayleigh_cs0.log"
}

Eval-Run -RunName "rvq_baseline_s43"

$trainRuns = @(
    @{RunName="rvq_distill_s41"; Config="configs/seed_sweep/rvq_distill_s41.yaml"},
    @{RunName="rvq_distill_s43"; Config="configs/seed_sweep/rvq_distill_s43.yaml"}
)

foreach ($run in $trainRuns) {
    $runName = $run.RunName
    $cfg = $run.Config
    $ckpt = "checkpoints/$runName/best.pt"
    if (Test-Path $ckpt) {
        Add-Status -Stage "train_skip" -RunName $runName -Command "" -ExitCode 0 -LogPath "" -Note "checkpoint already exists"
    } else {
        Run-Logged -Stage "train" -RunName $runName -ArgsList @("-X", "utf8", "scripts/03_train_vqvae.py", "--config", $cfg) -LogPath "logs/seed_sweep/${runName}_train.log"
    }
    Eval-Run -RunName $runName
}

Run-Logged -Stage "aggregate" -RunName "v04_tables" -ArgsList @("-X", "utf8", "scripts/11_make_v04_tables.py", "--all") -LogPath "logs/seed_sweep/v04_tables_aggregate.log"
