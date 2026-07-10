# RS-Token P0/P1 补实验执行手册

版本：2026-06-08  
工作目录：`H:\H-CODE\遥感+通信\rstoken`  
目的：把当前 RS-Token v0.4 论文的 P0/P1 严谨性短板补成可引用、可追踪、可复现实验产物。  
原则：只使用本地真实存在的脚本、配置、checkpoint 和日志；缺失结果必须标注为缺失，不允许从旧 `experiments/results.md` 乱码内容复制数字。

---

## 0. 证据边界和优先级

### 0.1 当前最强证据

- `logs/v04_tables/table2_task_path_mean_std.csv`：任务路径 3 个模型 seed 的 mean±std。
- `logs/v04_tables/table3_reconstruction_path.csv`：重建路径主配置结果，但当前是单模型 seed。
- `logs/e23_ldpc_key5seeds_mean_std.csv`：自定义 rate-1/2 systematic sparse LDPC，5 个 channel seeds。
- `logs/v04_tables/table4_external_baseline.csv`：WebP/JPEG2000 未保护 bitstream stress test。

### 0.2 本轮必须补的 P0/P1

| 优先级 | 实验 | 论文中解决的问题 | 最终产物 |
|---|---|---|---|
| P0 | E24：3 seed reconstruction sweep | 重建路径目前单 seed，支撑“渐进式重建”不足 | `logs/paper_p0/e24_recon_3seed_raw.csv`、`logs/paper_p0/e24_recon_3seed_mean_std.csv` |
| P0 | E25：L0-only vs all-layer distillation | 证明“只蒸馏 L0”的设计必要性 | 新 checkpoint + `logs/paper_p0/e25_l0_vs_all_layers.csv` |
| P0 | E26：RemoteCLIP vs OpenAI CLIP teacher | 校准 novelty，避免夸大 RemoteCLIP 独有性 | `logs/paper_p0/e26_teacher_ablation.csv` |
| P1 | E27：strict codec+LDPC baseline | 避免 WebP/JPEG2000 baseline 不公平 | `logs/paper_p1/e27_codec_ldpc_raw.csv`、`logs/paper_p1/e27_codec_ldpc_mean_std.csv` |
| P1 | E28：Rayleigh +10 reconstruction full sweep | 补齐 fading 下 k=1..4 的重建曲线 | `logs/paper_p1/e28_rayleigh10_recon_sweep.csv` |
| P1 | E29：LDPC full appendix table | 避免 selected rows 被认为 cherry-picking | `logs/paper_p1/e29_ldpc_full_appendix.csv` |
| **P1**（新） | **E30：NWPU-RESISC45 zero-shot tokenizer transfer** | 跨数据集泛化证据缺失，主稿"remote-sensing communication"框架在单数据集上站不住 | `logs/paper_p1/e30_nwpu_zeroshot.csv` |
| **P2**（新） | **E31：counterfactual layer-target ablation (L1-only distill)** | §4.3 layered probe 当前是循环论证；蒸馏只在 L0 → L1-L3 没语义是构造决定，不是发现 | 新 checkpoint + `logs/paper_p2/e31_layer_target_counterfactual.csv` |
| **P2**（新） | **E32：rvq_baseline / rvq_distill init seed 控制** | 25-pp gap 中 init 噪声占比未被剔除，reviewer 可问 | `logs/paper_p2/e32_init_seed_control.csv` |
| **P1**（新） | **E33：编解码器系统成本 profile** | 论文是"通信"主题但没报参数量/FLOPs/teacher-only 训练成本，UAV 部署性无法评估 | `logs/paper_p1/e33_system_cost.md` |
| **P0**（新，纯文档） | **E34：AID split 完整文档化** | §4.1 只说"a fixed split"未给比例、per-split 样本数、是否 per-class 平衡 | 在 `logs/paper_p0/e34_aid_split_audit.md` 落盘 |

---

## 1. 运行环境准备

### 1.1 进入项目

```powershell
cd H:\H-CODE\遥感+通信\rstoken
```

### 1.2 激活环境

优先尝试：

```powershell
conda activate rstoken
```

如果当前 shell 不支持 `conda activate`，使用本机 Conda 的实际 Python 路径替代以下命令中的 `python`。不要改实验脚本逻辑。

### 1.3 环境检查

```powershell
python scripts/00_check_env.py
```

验收：能看到 PyTorch、CUDA、依赖库信息；如果 CUDA 不可用，P0/P1 全量实验不建议跑 CPU。

### 1.4 固定线程，减少 Windows/sklearn native 崩溃风险

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:RSTOKEN_LOGREG_MAX_ITER='300'
$env:RSTOKEN_LOGREG_TOL='1e-3'
```

### 1.5 创建输出目录

```powershell
New-Item -ItemType Directory -Force logs\paper_p0, logs\paper_p1, logs\paper_p0\run_logs, logs\paper_p1\run_logs
```

---

## 2. 实验前完整性检查

### 2.1 必需 checkpoint

执行：

```powershell
Test-Path checkpoints\rvq_distill_s41\best.pt
Test-Path checkpoints\rvq_distill\best.pt
Test-Path checkpoints\rvq_distill_s43\best.pt
Test-Path checkpoints\rvq_distill_openai\best.pt
Test-Path checkpoints\rvq_baseline\best.pt
Test-Path checkpoints\aid_classifier_resnet34\best.pt
```

全部应输出 `True`。若任一为 `False`，先停止并记录缺失项，不要继续编造结果。

### 2.2 必需脚本

执行：

```powershell
Test-Path scripts\09_eval_rvqs_recon_task_split.py
Test-Path scripts\04b_eval_layered_probe.py
Test-Path scripts\10_eval_classic_baselines.py
Test-Path scripts\13_eval_ldpc_protected.py
Test-Path scripts\11_make_v04_tables.py
Test-Path scripts\12_make_v04_figures.py
```

全部应输出 `True`。

---

## 3. P0 / E24：3 seed reconstruction sweep

### 3.1 目的

当前论文的重建路径 `table3_reconstruction_path.csv` 是单模型 seed。E24 要补 `rvq_distill_s41`、`rvq_distill`、`rvq_distill_s43` 在 No-channel、AWGN +5 dB、AWGN +10 dB、Rayleigh +10 dB 下的 k=1..4 重建路径结果。

### 3.2 输入

| seed | model name | checkpoint |
|---|---|---|
| 41 | `rvq_distill_s41` | `checkpoints/rvq_distill_s41/best.pt` |
| 42 | `rvq_distill_s42` | `checkpoints/rvq_distill/best.pt` |
| 43 | `rvq_distill_s43` | `checkpoints/rvq_distill_s43/best.pt` |

### 3.3 命令

注意：`scripts/09_eval_rvqs_recon_task_split.py` 会同时输出 task 和 recon CSV。本实验只使用 recon CSV，但 task CSV 也要保留作为审计痕迹。

```powershell
python scripts/09_eval_rvqs_recon_task_split.py `
  --models "rvq_distill_s41=checkpoints/rvq_distill_s41/best.pt" `
  --recon_models "rvq_distill_s41" `
  --task_out logs/paper_p0/e24_task_s41.csv `
  --recon_out logs/paper_p0/e24_recon_s41.csv `
  --task_snrs "5,10" `
  --recon_snrs "5,10" `
  --ks "1,2,3,4" `
  --seed 41 `
  --device cuda `
  --batch_size 64 *> logs/paper_p0/run_logs/e24_s41.log
```

```powershell
python scripts/09_eval_rvqs_recon_task_split.py `
  --models "rvq_distill_s42=checkpoints/rvq_distill/best.pt" `
  --recon_models "rvq_distill_s42" `
  --task_out logs/paper_p0/e24_task_s42.csv `
  --recon_out logs/paper_p0/e24_recon_s42.csv `
  --task_snrs "5,10" `
  --recon_snrs "5,10" `
  --ks "1,2,3,4" `
  --seed 42 `
  --device cuda `
  --batch_size 64 *> logs/paper_p0/run_logs/e24_s42.log
```

```powershell
python scripts/09_eval_rvqs_recon_task_split.py `
  --models "rvq_distill_s43=checkpoints/rvq_distill_s43/best.pt" `
  --recon_models "rvq_distill_s43" `
  --task_out logs/paper_p0/e24_task_s43.csv `
  --recon_out logs/paper_p0/e24_recon_s43.csv `
  --task_snrs "5,10" `
  --recon_snrs "5,10" `
  --ks "1,2,3,4" `
  --seed 43 `
  --device cuda `
  --batch_size 64 *> logs/paper_p0/run_logs/e24_s43.log
```

说明：`condition_grid()` 会自动加入 `none,inf`，所以 `--recon_snrs "5,10"` 实际覆盖 No-channel、AWGN +5/+10、Rayleigh +5/+10。论文主表至少使用 No-channel、AWGN +5、AWGN +10、Rayleigh +10；Rayleigh +5 可作为边界结果。

### 3.4 汇总命令

用 PowerShell 合并 raw CSV：

```powershell
$files = @('logs/paper_p0/e24_recon_s41.csv','logs/paper_p0/e24_recon_s42.csv','logs/paper_p0/e24_recon_s43.csv')
$rows = foreach ($f in $files) { Import-Csv $f }
$rows | Export-Csv logs/paper_p0/e24_recon_3seed_raw.csv -NoTypeInformation -Encoding UTF8
```

生成 mean/std 汇总：

```powershell
$rows = Import-Csv logs/paper_p0/e24_recon_3seed_raw.csv
$groups = $rows | Group-Object channel,snr,k,bits_per_img
$out = foreach ($g in $groups) {
  $first = $g.Group[0]
  $psnr = @($g.Group | ForEach-Object {[double]$_.psnr})
  $lpips = @($g.Group | ForEach-Object {[double]$_.lpips})
  $cls = @($g.Group | Where-Object {$_.recon_cls_acc -ne ''} | ForEach-Object {[double]$_.recon_cls_acc})
  [pscustomobject]@{
    model = 'rvq_distill'
    channel = $first.channel
    snr = $first.snr
    k = $first.k
    bits_per_img = $first.bits_per_img
    psnr_mean = ($psnr | Measure-Object -Average).Average
    psnr_std = if ($psnr.Count -gt 1) { [math]::Sqrt((($psnr | ForEach-Object {($_ - (($psnr | Measure-Object -Average).Average))*($_ - (($psnr | Measure-Object -Average).Average))}) | Measure-Object -Sum).Sum / ($psnr.Count-1)) } else { 0 }
    lpips_mean = ($lpips | Measure-Object -Average).Average
    lpips_std = if ($lpips.Count -gt 1) { [math]::Sqrt((($lpips | ForEach-Object {($_ - (($lpips | Measure-Object -Average).Average))*($_ - (($lpips | Measure-Object -Average).Average))}) | Measure-Object -Sum).Sum / ($lpips.Count-1)) } else { 0 }
    recon_cls_acc_mean = if ($cls.Count -gt 0) { ($cls | Measure-Object -Average).Average } else { '' }
    recon_cls_acc_std = if ($cls.Count -gt 1) { [math]::Sqrt((($cls | ForEach-Object {($_ - (($cls | Measure-Object -Average).Average))*($_ - (($cls | Measure-Object -Average).Average))}) | Measure-Object -Sum).Sum / ($cls.Count-1)) } else { 0 }
    n_model_seeds = $g.Group.Count
    source = 'E24 3 model seeds; generated by scripts/09_eval_rvqs_recon_task_split.py'
  }
}
$out | Export-Csv logs/paper_p0/e24_recon_3seed_mean_std.csv -NoTypeInformation -Encoding UTF8
```

### 3.5 验收标准

- `logs/paper_p0/e24_recon_3seed_raw.csv` 存在，且每个 seed 应覆盖 `none/inf`、`awgn 5/10`、`rayleigh 5/10` 与 `k=1..4`。
- `logs/paper_p0/e24_recon_3seed_mean_std.csv` 中 `n_model_seeds` 应为 3。
- 如果任一 seed 缺失，论文不能写 3 seed mean±std，只能写 completed seeds 的 n，并在手稿中说明。

### 3.6 论文使用方式

- 若 E24 完成：把重建路径主表从单 seed 改为 mean±std。
- 若 E24 未完成：保留当前降调写法 `main-seed evidence suggests / indicates`。

---

## 4. P0 / E25：L0-only vs all-layer distillation

### 4.1 目的

当前方法主张“只将 RemoteCLIP 蒸馏施加到 L0”。需要一个直接消融：同样架构、同样 teacher、同样 seed，把蒸馏信号施加到所有 RVQ 层的累加表示或 full quantized representation，比较 L0 task、layered probe 和 reconstruction。

### 4.2 当前代码状态

现有 `scripts/03_train_vqvae.py` 只实现 L0 蒸馏：训练时使用 `out["zq_l0_ste"]` 输入 `DistillHead`。因此 E25 需要最小代码扩展，不能只靠命令行完成。

### 4.3 必须新增的配置

复制 `configs/seed_sweep/rvq_distill_s41.yaml` 为：

```text
configs/paper_p0/rvq_distill_all_layers_s42.yaml
```

建议内容：

- `run_name: rvq_distill_all_layers_s42`
- `seed: 42`
- `logging.ckpt_dir: checkpoints/rvq_distill_all_layers_s42`
- `logging.log_dir: logs/rvq_distill_all_layers_s42`
- `loss.distill_weight: 0.5`
- 新增字段：`distill.target: all_layers`

### 4.4 必须新增的训练逻辑

在 `scripts/03_train_vqvae.py` 中增加可选分支：

- 如果 `cfg["distill"].get("target", "l0") == "l0"`：保持现有 `out["zq_l0_ste"]`，不得改变旧实验。
- 如果 `target == "all_layers"`：使用 full quantized representation，例如 `out["zq"]` 或模型输出中等价的 full RVQ quantized feature。若当前 `VQVAE.forward()` 没有暴露 full zq，需要先检查 `models/vqvae.py` 的输出键，选择真实存在的张量。
- 新 checkpoint 命名必须与旧 checkpoint 分开，不能覆盖 `checkpoints/rvq_distill`。

### 4.5 训练命令

```powershell
python scripts/03_train_vqvae.py --config configs/paper_p0/rvq_distill_all_layers_s42.yaml *> logs/paper_p0/run_logs/e25_train_all_layers_s42.log
```

### 4.6 评估命令

任务/重建拆分：

```powershell
python scripts/09_eval_rvqs_recon_task_split.py `
  --models "rvq_distill_l0=checkpoints/rvq_distill/best.pt,rvq_distill_all=checkpoints/rvq_distill_all_layers_s42/best.pt" `
  --recon_models "rvq_distill_l0,rvq_distill_all" `
  --task_out logs/paper_p0/e25_task_l0_vs_all.csv `
  --recon_out logs/paper_p0/e25_recon_l0_vs_all.csv `
  --task_snrs "5,10" `
  --recon_snrs "5,10" `
  --ks "1,2,3,4" `
  --seed 42 `
  --device cuda `
  --batch_size 64 *> logs/paper_p0/run_logs/e25_eval_l0_vs_all.log
```

Layered probe：

```powershell
python scripts/04b_eval_layered_probe.py --ckpt checkpoints/rvq_distill/best.pt --out logs/paper_p0/e25_layered_probe_l0.csv --device cuda *> logs/paper_p0/run_logs/e25_probe_l0.log
python scripts/04b_eval_layered_probe.py --ckpt checkpoints/rvq_distill_all_layers_s42/best.pt --out logs/paper_p0/e25_layered_probe_all.csv --device cuda *> logs/paper_p0/run_logs/e25_probe_all.log
```

合并输出：

```powershell
Import-Csv logs/paper_p0/e25_task_l0_vs_all.csv | Export-Csv logs/paper_p0/e25_l0_vs_all_layers.csv -NoTypeInformation -Encoding UTF8
```

### 4.7 验收标准

- all-layer checkpoint 存在：`checkpoints/rvq_distill_all_layers_s42/best.pt`。
- `e25_task_l0_vs_all.csv` 中必须同时有 `rvq_distill_l0` 和 `rvq_distill_all`。
- `e25_recon_l0_vs_all.csv` 中必须同时有两个模型的 `k=1..4`。
- 不能只报告 all-layer 更差或更好；必须同时报告 L0 task、layered probe、PSNR/LPIPS/recon cls。

### 4.8 论文使用方式

- 如果 all-layer 不优于 L0-only：可写“localized L0 distillation is sufficient and avoids disturbing residual reconstruction layers”。
- 如果 all-layer 更强：必须修改方法叙事，不能继续强称 L0-only 是必要设计，只能写“we use L0-only as a simple conservative choice”。
- 如果结果混合：写 trade-off，避免选择性叙述。

---

## 5. P0 / E26：RemoteCLIP vs OpenAI CLIP teacher

### 5.1 目的

校准 RemoteCLIP novelty。已有 `checkpoints/rvq_distill_openai/best.pt`，可以直接评估，不需要重训。

### 5.2 命令

```powershell
python scripts/09_eval_rvqs_recon_task_split.py `
  --models "remoteclip=checkpoints/rvq_distill/best.pt,openai_clip=checkpoints/rvq_distill_openai/best.pt" `
  --recon_models "remoteclip,openai_clip" `
  --task_out logs/paper_p0/e26_teacher_task.csv `
  --recon_out logs/paper_p0/e26_teacher_recon.csv `
  --task_snrs "5,10" `
  --recon_snrs "5,10" `
  --ks "1,2,3,4" `
  --seed 42 `
  --device cuda `
  --batch_size 64 *> logs/paper_p0/run_logs/e26_teacher_eval.log
```

Layered probe：

```powershell
python scripts/04b_eval_layered_probe.py --ckpt checkpoints/rvq_distill/best.pt --out logs/paper_p0/e26_layered_probe_remoteclip.csv --device cuda *> logs/paper_p0/run_logs/e26_probe_remoteclip.log
python scripts/04b_eval_layered_probe.py --ckpt checkpoints/rvq_distill_openai/best.pt --out logs/paper_p0/e26_layered_probe_openai.csv --device cuda *> logs/paper_p0/run_logs/e26_probe_openai.log
```

### 5.3 汇总建议

生成 `logs/paper_p0/e26_teacher_ablation.csv`，至少包含：

| teacher | channel | snr | h0_acc | k4_psnr | k4_lpips | k4_recon_cls_acc | l0_layered_probe_acc | l0_to_l3_probe_acc |
|---|---|---|---|---|---|---|---|---|

可以由后续 AI 用 `Import-Csv` 从 `e26_teacher_task.csv`、`e26_teacher_recon.csv`、两个 layered probe CSV 合并。

### 5.4 验收标准

- 必须同时比较 no-channel、AWGN +5/+10、Rayleigh +5/+10 的 h0 task。
- 若只做 single seed，表注必须写 `single model seed, teacher ablation`。
- 论文不能写 RemoteCLIP 是唯一有效 teacher；只能写 RemoteCLIP 是 domain-specific teacher，在本实验中相对 OpenAI CLIP 的收益是多少。

---

## 6. P1 / E27：strict codec+LDPC baseline

### 6.1 目的

当前 WebP/JPEG2000 是未保护 bitstream stress test，不是公平系统 baseline。E27 使用现有 `scripts/13_eval_ldpc_protected.py` 对 RS-Token、JPEG2000、WebP 同时做 rate-1/2 LDPC 保护，在相同 total transmitted bits 下比较。

### 6.2 重要边界

- 该 LDPC 是本地自定义 systematic sparse LDPC + min-sum BP，不是 5G NR LDPC。
- 该实验可写作 `strict transmitted-bit controlled codec+LDPC stress baseline`，但仍不是完整 HARQ/AMC/标准协议系统。
- WebP 可能无法严格达到很低 source bits，脚本会记录 `actual_source_bits_mean/std`；不能忽略该列。

### 6.3 快速 smoke test

先用少量样本验证脚本跑通：

```powershell
python scripts/13_eval_ldpc_protected.py `
  --methods "rstoken,jpeg2000,webp" `
  --total_bits "5120" `
  --channels "awgn" `
  --snrs "10" `
  --channel_seeds "0" `
  --max_samples 60 `
  --device cuda `
  --out_csv logs/paper_p1/e27_smoke_codec_ldpc.csv *> logs/paper_p1/run_logs/e27_smoke.log
```

验收：`logs/paper_p1/e27_smoke_codec_ldpc.csv` 存在，且包含三类 method。如果 JPEG2000 被跳过，必须在日志中记录 encoder unavailable，不能强行补数。

### 6.4 全量命令

```powershell
python scripts/13_eval_ldpc_protected.py `
  --methods "rstoken,jpeg2000,webp" `
  --total_bits "5120,10240,20480" `
  --channels "awgn,rayleigh" `
  --snrs "5,10" `
  --channel_seeds "0,1,2,3,4" `
  --ldpc_rate "1/2" `
  --ldpc_seed 2026 `
  --max_iter 30 `
  --device cuda `
  --rstoken_ckpt checkpoints/rvq_distill/best.pt `
  --classifier_ckpt checkpoints/aid_classifier_resnet34/best.pt `
  --out_csv logs/paper_p1/e27_codec_ldpc_raw.csv *> logs/paper_p1/run_logs/e27_codec_ldpc_full.log
```

### 6.5 汇总命令

```powershell
$rows = Import-Csv logs/paper_p1/e27_codec_ldpc_raw.csv
$groups = $rows | Group-Object method,total_bits,source_bits,k,channel,snr
$out = foreach ($g in $groups) {
  $first = $g.Group[0]
  $metrics = @('h0_acc','recon_cls_acc','psnr','lpips','decode_failure_rate','post_ldpc_ber','ldpc_success_rate','actual_source_bits_mean')
  $obj = [ordered]@{
    method = $first.method
    total_bits = $first.total_bits
    source_bits = $first.source_bits
    k = $first.k
    channel = $first.channel
    snr = $first.snr
    n_channel_seeds = $g.Group.Count
  }
  foreach ($m in $metrics) {
    $vals = @($g.Group | Where-Object { $_.$m -ne '' -and $_.$m -ne $null } | ForEach-Object {[double]$_.$m})
    if ($vals.Count -gt 0) {
      $mean = ($vals | Measure-Object -Average).Average
      $std = if ($vals.Count -gt 1) { [math]::Sqrt((($vals | ForEach-Object {($_-$mean)*($_-$mean)}) | Measure-Object -Sum).Sum / ($vals.Count-1)) } else { 0 }
      $obj["${m}_mean"] = $mean
      $obj["${m}_std"] = $std
    } else {
      $obj["${m}_mean"] = ''
      $obj["${m}_std"] = ''
    }
  }
  [pscustomobject]$obj
}
$out | Export-Csv logs/paper_p1/e27_codec_ldpc_mean_std.csv -NoTypeInformation -Encoding UTF8
```

### 6.6 验收标准

- `e27_codec_ldpc_raw.csv` 包含 `method=rstoken,jpeg2000,webp`，如果某方法缺失必须解释原因。
- 每个 method / total_bits / channel / snr 应有 5 个 channel seeds。
- 必须报告 `actual_source_bits_mean` 和 `decode_failure_rate`，不能只报告 accuracy。

### 6.7 论文使用方式

- 若 RS-Token 在同等 total bits + LDPC 下明显更稳，可写“under this custom LDPC-controlled setting”。
- 若 codec+LDPC 表现接近或更好，必须保留并诚实讨论，不能删除。
- 无论结果如何，都不能写成 5G NR LDPC 或完整商用系统比较。

---

## 7. P1 / E28：Rayleigh +10 reconstruction full sweep

### 7.1 目的

为 fading 场景补完整 k=1..4 reconstruction 曲线。E24 已经会产生 Rayleigh +10 的 3 seed reconstruction sweep；如果 E24 成功，E28 可直接从 E24 提取，不必重复跑。

### 7.2 如果 E24 已完成

提取 Rayleigh +10：

```powershell
Import-Csv logs/paper_p0/e24_recon_3seed_mean_std.csv |
  Where-Object { $_.channel -eq 'rayleigh' -and $_.snr -eq '10' } |
  Export-Csv logs/paper_p1/e28_rayleigh10_recon_sweep.csv -NoTypeInformation -Encoding UTF8
```

### 7.3 如果只想单独跑主 seed

```powershell
python scripts/09_eval_rvqs_recon_task_split.py `
  --models "rvq_distill=checkpoints/rvq_distill/best.pt" `
  --recon_models "rvq_distill" `
  --task_out logs/paper_p1/e28_task_unused.csv `
  --recon_out logs/paper_p1/e28_rayleigh10_recon_sweep_single_seed.csv `
  --task_snrs "10" `
  --recon_snrs "10" `
  --ks "1,2,3,4" `
  --seed 42 `
  --device cuda `
  --batch_size 64 *> logs/paper_p1/run_logs/e28_rayleigh10_single_seed.log
```

说明：该命令仍会包含 No-channel、AWGN +10、Rayleigh +10。只取 `channel=rayleigh,snr=10`。

### 7.4 验收标准

- 最好使用 E24 的 3 seed 结果。
- 如果只使用 single seed，图注和正文必须写 single main seed。

---

## 8. P1 / E29：LDPC full appendix table

### 8.1 目的

当前论文主文只放 selected LDPC rows。E29 生成完整 appendix table，防止审稿人认为 cherry-picking。

### 8.2 输入优先级

优先使用已有：

```text
logs/e23_ldpc_key5seeds.csv
logs/e23_ldpc_key5seeds_mean_std.csv
```

如果 E27 已完成，也可以附：

```text
logs/paper_p1/e27_codec_ldpc_raw.csv
logs/paper_p1/e27_codec_ldpc_mean_std.csv
```

### 8.3 生成 appendix CSV

```powershell
Copy-Item logs/e23_ldpc_key5seeds_mean_std.csv logs/paper_p1/e29_ldpc_full_appendix.csv -Force
```

如果要合并 E27：

```powershell
Copy-Item logs/paper_p1/e27_codec_ldpc_mean_std.csv logs/paper_p1/e29_codec_ldpc_full_appendix.csv -Force
```

### 8.4 验收标准

- appendix 表必须说明：5 channel seeds、custom rate-1/2 systematic sparse LDPC、min-sum BP、not 5G NR LDPC。
- 主文 selected rows 必须能在 appendix full table 中查到。

---

## 8.5 P1 / E30：NWPU-RESISC45 zero-shot tokenizer transfer

### 8.5.1 目的

主稿框架是"remote-sensing communication"，但所有证据来自 AID 30 类。外部复审（§12 N1 已并入主报告）指出：单数据集泛化是 letter 级以上 venue 的硬要求。E30 用现成 `rvq_distill` checkpoint 在 **NWPU-RESISC45** 上做 zero-shot tokenizer 迁移：tokenizer 不重训，只在新数据集的 train split 上用 `h_0`/L0 BoW 训一个新的线性 probe，验证 L0 RemoteCLIP-aligned 表示是否跨数据集仍然 carry scene semantics。

> 原则：**不重训 tokenizer**，否则就不是"transfer"了；只重训线性 probe 头。

### 8.5.2 数据准备前置

NWPU-RESISC45（45 类，每类 700 张，共 31,500 张）需要**手工下载一次**。常见来源：
- 官方页面：https://gcheng-nwpu.github.io/  → "NWPU-RESISC45" 数据集
- arXiv 论文：Cheng, Han, Lu (2017), *Remote Sensing Image Scene Classification: Benchmark and State of the Art*

下载后解压到：
```text
H:\H-CODE\遥感+通信\rstoken\data\NWPU-RESISC45\
```
应得到 45 个子文件夹（airplane、airport、…），每个 700 张 jpg。

**如果不下载**：把 E30 标记为"未做 / future work"，论文 §1 / 摘要必须明确"single-dataset evaluation"，不能写"general remote-sensing communication"。

### 8.5.3 准备 NWPU split CSV

复用现有的 `AIDDataset(csv)` 读取器（`models/datasets.py:38`），不需要改模型代码。新建脚本：

```text
scripts/14_prepare_nwpu_splits.py
```

建议逻辑（伪代码，由后续 AI 实现）：
```python
# 输入: data/NWPU-RESISC45/<class>/*.jpg
# 输出: data/NWPU_splits/{train,val,test}.csv  with columns path,class_id
# split: per-class 70/10/20，固定 random_state=2026，确保可复现
import random, csv, glob
from pathlib import Path

random.seed(2026)
root = Path("data/NWPU-RESISC45")
classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
out_dir = Path("data/NWPU_splits"); out_dir.mkdir(exist_ok=True)
(out_dir / "classes.txt").write_text("\n".join(classes), encoding="utf-8")

splits = {"train": [], "val": [], "test": []}
for cls_id, cls in enumerate(classes):
    files = sorted(glob.glob(str(root / cls / "*.jpg")))
    random.shuffle(files)
    n = len(files)
    n_tr = int(n * 0.70); n_va = int(n * 0.10)
    for f in files[:n_tr]:        splits["train"].append((f, cls_id))
    for f in files[n_tr:n_tr+n_va]: splits["val"].append((f, cls_id))
    for f in files[n_tr+n_va:]:   splits["test"].append((f, cls_id))

for k, items in splits.items():
    with open(out_dir / f"{k}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["path", "class_id"]); w.writerows(items)
```

运行：
```powershell
python scripts/14_prepare_nwpu_splits.py *> logs/paper_p1/run_logs/e30_prepare_split.log
```

### 8.5.4 评估命令

`scripts/09_eval_rvqs_recon_task_split.py` 当前从 `cfg["data"]` 读 `AIDConfig`。最低成本做法：**新建一个 NWPU 配置 yaml**，复用同一脚本。建议在 `configs/paper_p1/eval_nwpu.yaml` 写：

```yaml
data:
  splits_dir: data/NWPU_splits
  image_size: 256
  batch_size: 64
  num_workers: 4
  pin_memory: true

model:
  num_classes: 45        # 与 AID 的 30 不同
```

**关键边界**：tokenizer 在 AID 上训过；NWPU 类别**不在**蒸馏可见类列表里。这正是"zero-shot tokenizer transfer"的定义——L0 表示是否仍能让一个新训的线性 probe 区分 NWPU 的 45 类。

如果 09 脚本无法直接接受外部 yaml override `data` 路径，**最小改造**是给它加一个 `--data_yaml` 参数指向新 yaml；不要改主流程。

```powershell
python scripts/09_eval_rvqs_recon_task_split.py `
  --models "rvq_distill=checkpoints/rvq_distill/best.pt,rvq_baseline=checkpoints/rvq_baseline/best.pt" `
  --recon_models "" `
  --task_out logs/paper_p1/e30_nwpu_zeroshot.csv `
  --recon_out logs/paper_p1/e30_nwpu_recon_unused.csv `
  --task_snrs "5,10" `
  --recon_snrs "5,10" `
  --ks "1" `
  --seed 42 `
  --device cuda `
  --batch_size 64 `
  --data_yaml configs/paper_p1/eval_nwpu.yaml *> logs/paper_p1/run_logs/e30_nwpu_eval.log
```

> 注意：`--ks "1"` 是因为 E30 只评 task path（k=1），不评重建。`--recon_models ""` 跳过重建分支。

### 8.5.5 验收标准

- `data/NWPU_splits/{train,val,test}.csv` 三个文件存在；每个 CSV 行数和官方 31,500 数量一致。
- `e30_nwpu_zeroshot.csv` 至少包含 `rvq_distill` 与 `rvq_baseline` 两个 model 在 `none/inf`、`awgn 5/10`、`rayleigh 5/10` 下的 h0_acc。
- 比较 `rvq_distill` 在 NWPU 上 vs AID 上的 h0_acc gap：gap 越大说明 distillation 越数据集-specific（贡献被削弱）；gap 小说明 L0 表示真有泛化（贡献被强化）。

### 8.5.6 论文使用方式

- **若 NWPU h0_acc ≥ 70%（绝对值）且 distill > baseline gap ≥ 15 pp**：可以写"L0 representation transfers to an unseen RS scene benchmark; gain over plain RVQ persists"。强支持泛化主张。
- **若 NWPU h0_acc 显著低于 AID 但 distill > baseline 仍存在**：写"transfer is partial; absolute accuracy drops on NWPU's 45-way classification, but distillation gain is preserved"。中等支持。
- **若 distill 与 baseline 在 NWPU 上无显著 gap**：必须诚实报告"AID-trained distillation does not transfer to NWPU; we limit our claims to AID scene classification"。这种情况下论文 §1 / 摘要的 "remote-sensing" 必须改为 "AID scene-level"。

---

## 8.6 P2 / E31：counterfactual layer-target ablation（L1-only distill）

### 8.6.1 目的

外部复审（§12 N3 已并入主报告）指出 §4.3 layered probe 存在循环论证：蒸馏只在 L0 → L1-L3 当然不携带任务语义 → 这是"工程意图"而非"发现"。E31 训练一个把蒸馏施加到 **L1**（而不是 L0）的对照 checkpoint。预期：L1 那一层会因蒸馏而携带任务语义，L0/L2/L3 不携带；从而证明**蒸馏目标层决定语义所在层**，而不是 L0 本质上更适合语义。这才是真正的"layer specialization"证据。

### 8.6.2 与 E25 的关系

- E25 是"L0-only vs all-layer"的纵向对照（蒸馏强度从 1 层 → 4 层）。
- E31 是"L0-only vs L1-only"的横向对照（蒸馏作用层位置变化但只 1 层）。
两者互不替代：E25 回答"是否需要全部层"，E31 回答"L0 这个位置是否特殊"。

### 8.6.3 配置

复用 E25 中扩展过的 `distill.target` 字段。`scripts/03_train_vqvae.py` 在 E25 改造后已支持 `target=l0|all_layers`，本实验需要再加一个分支 `target=l1`：

- `distill.target: l1` → 训练时使用 `out["zq_l1_ste"]`（如果 `models/vqvae.py` 已暴露；否则需要确认）。
- 配置文件：`configs/paper_p2/rvq_distill_l1_only_s42.yaml`，内容除 `distill.target: l1` 和 ckpt 路径外与 E25 相同。

### 8.6.4 训练命令

```powershell
python scripts/03_train_vqvae.py --config configs/paper_p2/rvq_distill_l1_only_s42.yaml *> logs/paper_p2/run_logs/e31_train_l1.log
```

### 8.6.5 评估命令

```powershell
python scripts/04b_eval_layered_probe.py `
  --ckpt checkpoints/rvq_distill_l1_only_s42/best.pt `
  --out logs/paper_p2/e31_layered_probe_l1distill.csv `
  --device cuda *> logs/paper_p2/run_logs/e31_probe_l1.log

python scripts/09_eval_rvqs_recon_task_split.py `
  --models "rvq_distill_l1=checkpoints/rvq_distill_l1_only_s42/best.pt" `
  --recon_models "rvq_distill_l1" `
  --task_out logs/paper_p2/e31_l1_task.csv `
  --recon_out logs/paper_p2/e31_l1_recon.csv `
  --task_snrs "5,10" `
  --recon_snrs "5,10" `
  --ks "1,2,3,4" `
  --seed 42 `
  --device cuda `
  --batch_size 64 *> logs/paper_p2/run_logs/e31_eval_l1.log
```

汇总：
```powershell
$rows = @()
$rows += Import-Csv logs/paper_p0/e25_layered_probe_l0.csv | ForEach-Object { $_ | Add-Member -NotePropertyName distill_target -NotePropertyValue 'l0' -PassThru }
$rows += Import-Csv logs/paper_p2/e31_layered_probe_l1distill.csv | ForEach-Object { $_ | Add-Member -NotePropertyName distill_target -NotePropertyValue 'l1' -PassThru }
$rows | Export-Csv logs/paper_p2/e31_layer_target_counterfactual.csv -NoTypeInformation -Encoding UTF8
```

### 8.6.6 验收标准

- `checkpoints/rvq_distill_l1_only_s42/best.pt` 存在。
- 输出 CSV 中可比较：`distill_target=l0` 时 L0 layer 概率（probe acc）vs `distill_target=l1` 时 L1 layer 概率。
- 如果 L1-only 蒸馏后 **L0 probe acc 没显著下降，L1 probe acc 没显著上升** → §4.3 的"localization"叙事不成立，论文必须降调。

### 8.6.7 论文使用方式

- **预期结果**（蒸馏目标层决定语义层）：可在 §4.3 加一句"A counterfactual ablation that distills L1 instead of L0 shifts task semantics to L1, confirming that distillation localizes scene semantics to its target layer"。这条句子能把 §4.3 从循环论证升级为因果证据。
- **反预期结果**（语义无法被定位到 L1）：必须在 Discussion 加 limitation："L0-only distillation may benefit from properties beyond layer position; we cannot fully separate target-layer effect from L0's coarse-quantization role"。

---

## 8.7 P2 / E32：rvq_baseline / rvq_distill init seed 控制

### 8.7.1 目的

§4.2 报告 `rvq_distill` 比 `rvq_baseline` 在 no-channel h0_acc 高 25 pp（58.23 → 83.33）。但**两个模型是否使用同一 init seed** 在论文中没说明。如果 init 不同，25 pp 中可能有一部分来自 init 差异而非蒸馏本身。E32 用同一 init seed 对两个模型分别训一个对照副本，剔除 init 噪声。

### 8.7.2 检查现有训练脚本

第一步只是审计现有代码，不一定要新跑：

```powershell
python -c "import yaml; print(yaml.safe_load(open('configs/seed_sweep/rvq_distill_s41.yaml'))['seed'])"
python -c "import yaml; print(yaml.safe_load(open('configs/rvq_baseline.yaml'))['seed'])" 2>$null
```

> 如果 `configs/rvq_baseline.yaml` 不存在，找出实际用过的 baseline yaml 文件名（`grep -r 'rvq_baseline' configs/`）。

如果两者 seed 相同 → 直接在 audit 文档里说明 "init seed controlled by config"，不需要新实验。

### 8.7.3 如果 seed 不同

为 baseline 重新训练一个版本，使用与 `rvq_distill_s42` 相同的 seed=42：

新建 `configs/paper_p2/rvq_baseline_s42_matched.yaml`：复制现有 baseline yaml + 把 seed 改为 42 + ckpt 改为 `checkpoints/rvq_baseline_s42_matched`。

```powershell
python scripts/03_train_vqvae.py --config configs/paper_p2/rvq_baseline_s42_matched.yaml *> logs/paper_p2/run_logs/e32_train_baseline_matched.log
```

评估：
```powershell
python scripts/09_eval_rvqs_recon_task_split.py `
  --models "rvq_baseline_matched=checkpoints/rvq_baseline_s42_matched/best.pt,rvq_distill_s42=checkpoints/rvq_distill/best.pt" `
  --recon_models "" `
  --task_out logs/paper_p2/e32_init_seed_control.csv `
  --recon_out logs/paper_p2/e32_recon_unused.csv `
  --task_snrs "5,10" `
  --recon_snrs "5,10" `
  --ks "1" `
  --seed 42 `
  --device cuda `
  --batch_size 64 *> logs/paper_p2/run_logs/e32_eval.log
```

### 8.7.4 验收标准

- audit 文档 `logs/paper_p2/e32_init_seed_audit.md` 至少说明：
  - 原 baseline 与原 distill 是否同 seed
  - 如不同 seed 则给出 matched-seed baseline 的 h0_acc
  - 计算 `delta_distill = h0_acc(distill) - h0_acc(baseline_matched)`
- 如果 matched gap 仍 ≥ 20 pp → distillation 收益是稳定的；论文不用改。
- 如果 matched gap 显著小于 25 pp（如降到 15 pp）→ 论文必须更新 §4.2 数字并加一句"after controlling for init seed, distillation gain is X pp"。

---

## 8.8 P1 / E33：编解码器系统成本 profile

### 8.8.1 目的

论文是"通信"主题，但**没有任何参数量、FLOPs、推理延迟、teacher-only 训练成本**的报告。外部复审指出：UAV/低空遥感场景下，**部署可行性**是审稿核心关注点之一。E33 不跑新训练或评估，只 profile 现有 checkpoint。

### 8.8.2 命令

新建 `scripts/15_profile_system_cost.py`（伪代码，由后续 AI 实现）：

```python
# 输入: checkpoint 路径、image size 256
# 输出: encoder/decoder/distill_head 各自的参数量、FLOPs/image、单图推理延迟（GPU/CPU 各一次）
# 用 thop 或 torchinfo + torch.profiler
import torch, time
from thop import profile  # pip install thop
from models.vqvae import build_model_from_ckpt   # 假定已有，否则按实际 import 调整

model = build_model_from_ckpt("checkpoints/rvq_distill/best.pt", device="cuda").eval()
x = torch.randn(1, 3, 256, 256, device="cuda")

# 编码器单独
enc_flops, enc_params = profile(model.encoder, inputs=(x,), verbose=False)
# 解码器单独 (需要先得到 quantized latent)
with torch.no_grad():
    out = model(x)
    z = out["zq_sum"] if "zq_sum" in out else out["zq"]   # 按实际键
dec_flops, dec_params = profile(model.decoder, inputs=(z,), verbose=False)

# 全模型推理延迟（GPU 50 次平均）
torch.cuda.synchronize()
t0 = time.time()
for _ in range(50):
    with torch.no_grad():
        _ = model(x)
torch.cuda.synchronize()
gpu_latency_ms = (time.time() - t0) / 50 * 1000

# 写出
with open("logs/paper_p1/e33_system_cost.md", "w", encoding="utf-8") as f:
    f.write(f"# E33 System Cost Profile\n\n")
    f.write(f"| Component | Params | FLOPs/image |\n|---|---|---|\n")
    f.write(f"| Encoder | {enc_params/1e6:.2f} M | {enc_flops/1e9:.2f} G |\n")
    f.write(f"| Decoder | {dec_params/1e6:.2f} M | {dec_flops/1e9:.2f} G |\n")
    f.write(f"\nGPU latency (single image, 256x256, fp32): **{gpu_latency_ms:.2f} ms**\n")
    f.write(f"\n**RemoteCLIP teacher** is used only during training; it is NOT part of the deployed encoder/decoder.\n")
```

```powershell
python scripts/15_profile_system_cost.py *> logs/paper_p1/run_logs/e33_profile.log
```

### 8.8.3 验收标准

- `logs/paper_p1/e33_system_cost.md` 存在，至少包含 Encoder 参数量、Decoder 参数量、单图 GPU 延迟。
- 文档明确标注 RemoteCLIP teacher **training-only**，部署时不包含。

### 8.8.4 论文使用方式

在 §3 末尾或 §4.1 加一句话级别的 system-cost 描述：

> "The deployed encoder–decoder has X.X M / Y.Y M parameters and Z.Z G FLOPs per 256×256 image; single-image GPU inference latency is W.W ms. RemoteCLIP (≈150 M parameters) is used only during training and is not deployed at the transmitter or receiver."

这一句话能拆掉一整类"deployability"质疑。

---

## 8.9 P0（纯文档）/ E34：AID split 完整文档化

### 8.9.1 目的

主稿 §4.1 只说"a fixed train/validation/test split on AID"。比例是多少？per-class 还是 random？per-split 样本数？这是审稿人下载论文后**5 分钟内会问的问题**，但不需要任何新实验，是纯文档工作。

### 8.9.2 命令

```powershell
$splits = 'train','val','test'
$rows = foreach ($s in $splits) {
  $csv = Import-Csv "data/AID_splits/$s.csv"
  $byclass = $csv | Group-Object class_id | ForEach-Object { $_.Count }
  [pscustomobject]@{
    split = $s
    n_samples = $csv.Count
    n_classes = ($csv | Group-Object class_id).Count
    per_class_min = ($byclass | Measure-Object -Minimum).Minimum
    per_class_max = ($byclass | Measure-Object -Maximum).Maximum
    per_class_mean = [math]::Round(($byclass | Measure-Object -Average).Average, 1)
  }
}
$rows | Format-Table | Out-File logs/paper_p0/e34_aid_split_audit.md -Encoding UTF8
$rows | Export-Csv logs/paper_p0/e34_aid_split_audit.csv -NoTypeInformation -Encoding UTF8
```

### 8.9.3 验收标准

- `e34_aid_split_audit.md` 给出三 split 的样本数、类别数、per-class min/mean/max。
- 用这些数字反推 split ratio（例如 5500/1100/3400 → 55/11/34）。

### 8.9.4 论文使用方式

把现有的"a fixed train/validation/test split on AID"扩展为：

> "We use a fixed per-class stratified split of AID into train/val/test with X / Y / Z samples (≈A% / B% / C%); per-class counts range from N_min to N_max (mean N_mean). The split CSVs are bundled with the artifact."

---

## 9. 结果归档与论文表格更新

### 9.1 禁止覆盖旧 v04 表格

不要直接覆盖：

```text
logs/v04_tables/table2_task_path_mean_std.csv
logs/v04_tables/table3_reconstruction_path.csv
logs/v04_tables/table4_external_baseline.csv
```

除非已经完成完整审计并准备更新论文。P0/P1 新结果先放在：

```text
logs/paper_p0/
logs/paper_p1/
```

### 9.2 建议新增最终表格

完成全部实验后，后续 AI 应生成：

| 文件 | 内容 |
|---|---|
| `logs/paper_p0/final_table_recon_3seed.md` | 论文重建路径替代表 |
| `logs/paper_p0/final_table_l0_vs_all_layers.md` | L0-only vs all-layer 消融表 |
| `logs/paper_p0/final_table_teacher_ablation.md` | RemoteCLIP vs OpenAI CLIP |
| `logs/paper_p1/final_table_codec_ldpc.md` | codec+LDPC 公平 transmitted-bit 表 |
| `logs/paper_p1/final_appendix_ldpc_full.md` | LDPC 完整附录表 |

### 9.3 论文措辞更新规则

- E24 完成前：重建结论写 `single main-seed evidence suggests`。
- E24 完成后：可写 `across three model seeds`，并报告 mean±std。
- E25 完成前：不要强称 L0-only 是最优，只写 design choice。
- E26 完成后：RemoteCLIP 只能写 domain-specific teacher；如果 OpenAI CLIP 接近，必须承认 V-L distillation 是主要因素。
- E27 完成前：WebP/JPEG2000 只能是 unprotected stress test。
- E27 完成后：可以新增 codec+LDPC baseline，但仍限定为 custom LDPC，不是标准通信系统。

---

## 10. 最终完成检查清单

后续 AI 执行完必须逐项打勾：

- [ ] 所有命令从 `H:\H-CODE\遥感+通信\rstoken` 运行。
- [ ] 所有新 CSV 保存在 `logs/paper_p0/` 或 `logs/paper_p1/`。
- [ ] 没有从 `experiments/results.md` 复制任何数值。
- [ ] E24 的 mean/std 标明 `n_model_seeds`。
- [ ] E25 明确是否改过 `03_train_vqvae.py`，并记录 diff。
- [ ] E26 标明 single seed teacher ablation。
- [ ] E27 标明 `actual_source_bits_mean`、`decode_failure_rate`、`post_ldpc_ber`。
- [ ] E29 附录完整表能追溯到 raw CSV。
- [ ] 更新论文前先写一份 `logs/paper_p0_p1_completion_audit.md`，列出每个 claim 对应的新证据。
- [ ] 若任何实验失败，保留失败日志，并在 audit 中写明不能支持相应 claim。

---

## 11. 推荐执行顺序

按"成本 × 影响"排序，先做无需重训、能立刻提升论文严谨性的项：

**第 0 档（零训练，立刻做，~1 小时）**
1. **E34**：AID split 文档化。pure shell，5 分钟。审稿人 5 分钟内会问的问题。
2. **E33**：系统成本 profile。只需要 thop + 一个新脚本，30 分钟，拆掉一类"deployability"质疑。

**第 1 档（无新训练，只评估，~半天）**
3. **E24**：3 seed reconstruction sweep。**最重要**，且不改训练代码。补完后 Table 3 从单 seed 升级为 mean ± std，直接拆掉外部复审的 N6/N7 + 旧 MW1 三个攻击面。
4. **E26**：RemoteCLIP vs OpenAI CLIP teacher。已有 `checkpoints/rvq_distill_openai/best.pt`，无需重训。校准 RemoteCLIP novelty。
5. **E32**：init seed 控制 audit。先只做 §8.7.2 配置审计；如果 seed 已对齐，直接落 audit 文档，零成本剔除一个潜在质疑。

**第 2 档（新评估或轻微改造，~1 天）**
6. **E27 smoke test**：确认 codec+LDPC 三类方法能跑。
7. **E27 full**：严格 transmitted-bit codec+LDPC baseline。
8. **E29**：整理 LDPC full appendix（依赖 E27 或 E23）。
9. **E28**：如果 E24 已覆盖，直接从 E24 提取；否则单独补。
10. **E30**：NWPU-RESISC45 zero-shot tokenizer transfer。需要先**手工下载** NWPU 数据集，并写一个 split 准备脚本和 09 脚本的 yaml override 参数。如果跨数据集 gap 不大，论文可从"single-dataset"升级为"cross-dataset zero-shot"，**最高 ROI 的单项**。

**第 3 档（需要重训，~2-3 天，最后做）**
11. **E25**：L0-only vs all-layer distillation。需要改 `03_train_vqvae.py` 加 `distill.target=all_layers` 分支，并重训一个 checkpoint。
12. **E31**：counterfactual L1-only distill。E25 改造完之后顺手再加一个 `target=l1` 分支，复用同套训练脚手架。如果跑出预期结果，§4.3 从循环论证升级为因果证据。
13. **E32 实跑分支**：仅当 E32 §8.7.2 audit 发现 seed 不对齐时才需要。

**最低完成线**（如果算力非常有限）：
- 必须做：**E34 + E24 + E26 + E27 smoke/full**
- 强烈建议：**E33 + E30**（任何投稿都会被问到这两件事）
- E25/E31 若来不及：论文 §4.3 必须降调为"engineering choice"而非"localization discovery"，并把 L0-only 的"必要性"主张移到 future work。

### 11.1 与外部复审 P0/P1 的对应关系

| 外部复审优先级（review_report §12） | 对应实验 | 不需新实验，只需文字修改 |
|---|---|---|
| P0 N1（修 bib + 正文写 MOC-RVQ/DeepJSCC/SemCLIP） | — | ✓ 在 `rs_token_v0.4.tex` 直接改 |
| P0 N2（标题改 Channel-Aware/Adaptive） | — | ✓ 10 分钟 |
| P0 N4（摘要砍到 ≤200 词） | — | ✓ 30 分钟 |
| P0 §7 #4（英文稿同步图 2/图 3） | — | ✓ |
| P1 §7 #1（多 seed 重建） | **E24** | |
| P1 N3 低成本（§4.3 措辞重写） | — | ✓ |
| P1 N6（Table 4 加 unprotected 列） | — | ✓ 由 E24 + 现有 Table 3 数据合并即可 |
| P1（跨数据集泛化） | **E30** | |
| P1（系统成本） | **E33** | |
| P0/P1（split 文档化） | **E34** | |
| P2 N3 高成本（layer-target counterfactual） | **E31** | |
| P2 N5（contribution #2 措辞收紧） | — | ✓ |
| P2 N7（跨表 std 一致化） | — | ✓ 由 E24 后自动解决 |
| P2（init seed 控制） | **E32** | |

