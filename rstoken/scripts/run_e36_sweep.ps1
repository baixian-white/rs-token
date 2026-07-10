# E36 — continuous SNR sweep, sequential over 6 checkpoints.
# Each checkpoint: 1 eval call -> task k=1 + recon k=1,4
# Channels: AWGN and Rayleigh at SNRs {-5,-2,0,2,5,7,10,12,15,20} dB
# (AWGN waterfall ends at 15 dB; Rayleigh extends to 20 dB; we keep
#  the union grid because condition_grid() pairs the same snr list
#  with both channels.)

$ErrorActionPreference = 'Stop'
$root = 'd:\CODE\遥感+通信\遥感+通信\rstoken'
Push-Location $root
try {
  $env:PYTHONUTF8 = '1'
  $env:PYTHONIOENCODING = 'utf-8'
  $env:OMP_NUM_THREADS = '1'
  $env:MKL_NUM_THREADS = '1'
  $env:OPENBLAS_NUM_THREADS = '1'
  $env:NUMEXPR_NUM_THREADS = '1'
  $env:RSTOKEN_LOGREG_MAX_ITER = '300'
  $env:RSTOKEN_LOGREG_TOL = '1e-3'

  $py = 'C:\Users\Administrator\miniconda3\envs\rstoken\python.exe'
  $snrs = '-5,-2,0,2,5,7,10,12,15,20'
  $ks = '1,4'
  $data_yaml = 'configs/paper_v05/eval_aid_local.yaml'

  # 6 checkpoints: 2 model families x 3 seeds.
  # Names mirror checkpoint dir structure.
  $models = @(
    @{ name = 'rvq_baseline_s41'; ckpt = 'checkpoints/rvq_baseline_s41/best.pt' },
    @{ name = 'rvq_baseline_s42'; ckpt = 'checkpoints/rvq_baseline/best.pt' },
    @{ name = 'rvq_baseline_s43'; ckpt = 'checkpoints/rvq_baseline_s43/best.pt' },
    @{ name = 'rvq_distill_s41';  ckpt = 'checkpoints/rvq_distill_s41/best.pt' },
    @{ name = 'rvq_distill_s42';  ckpt = 'checkpoints/rvq_distill/best.pt' },
    @{ name = 'rvq_distill_s43';  ckpt = 'checkpoints/rvq_distill_s43/best.pt' }
  )

  foreach ($m in $models) {
    $name = $m.name
    $ckpt = $m.ckpt
    $tcsv = "logs/paper_v05/e36_task_${name}.csv"
    $rcsv = "logs/paper_v05/e36_recon_${name}.csv"
    $tlog = "logs/paper_v05/run_logs/e36_${name}.log"
    Write-Host "=== E36 eval: $name === ($(Get-Date -Format HH:mm:ss))"

    & $py scripts\09_eval_rvqs_recon_task_split.py `
      --models "$name=$ckpt" `
      --recon_models "$name" `
      --task_out "$tcsv" `
      --recon_out "$rcsv" `
      --task_snrs $snrs `
      --recon_snrs $snrs `
      --ks $ks `
      --seed 42 `
      --device cuda `
      --batch_size 64 `
      --data_yaml $data_yaml `
      *> $tlog

    if ($LASTEXITCODE -ne 0) {
      Write-Host "FAILED on $name (exit $LASTEXITCODE); see $tlog"
      exit $LASTEXITCODE
    }
  }
  Write-Host "=== E36 done at $(Get-Date -Format HH:mm:ss) ==="
}
finally {
  Pop-Location
}
