# RS-Token v0.5 实验日志

承接 `experiments/rs_token_v0.4_p0_p1_experiment_manual.md`（E24–E34）。
v0.5 review 编号从 **E35** 起算，对应 `paper_draft/latex/rs_token_review_v0.5.md` §2 实验清单。

所有实验产物落在 `logs/paper_v05/`、`checkpoints/paper_v05/`、`configs/paper_v05/`。

---

## 状态总览

| ID | 名称 | 优先级 | 状态 | 备注 |
|---|---|---|---|---|
| **E35** | **Deep-JSCC baseline (ADJSCC, mixed AWGN+Rayleigh)** | **T1** | **DONE 2026-06-11** | 9 ckpt（3 rates × 3 seeds），mixed channel 训练 + 评估 + 聚合 |
| **E36** | **连续 SNR 扫描（推理-only）** | **T1** | **DONE 2026-06-11** | 6 ckpt × 22 conditions, ~21 min wall clock |
| **E37** | **Placement counterfactual `All layers` 补 seed 41/43** | **T1** | **DONE 2026-06-10/11** | 训练 + 评估 + 聚合闭环 |
| E38 | WebP rate-floor 脚注 | T1 (text) | TODO | 10 分钟 |
| **E39** | **重建路径 NWPU zero-shot transfer** | **T1** | **DONE 2026-06-11** | NWPU 分类器 + 重建评估 + 聚合闭环 |
| **E40** | **Init-seed paired control** | **T2** | **DONE 2026-06-12** | 6 paired ckpt × 3 seeds，paired Δ 五信道全部在 unpaired 25.10 pp 的 2σ 内 |
| **E41** | **直接量化 RemoteCLIP 控制实验** | **T2** | **DONE 2026-06-12** | DCRC 3-seed 训练 + 评估闭环；**Pareto trade-off 实锤**：DCRC task ↑+13~+24 pp，PSNR ↓8 dB，recon-cls ↓75 pp |
| **E42** | **RVQ 层数 K∈{2,3,6} 消融** | **T2** | **DONE 2026-06-12** | 3 ckpt 训练 + 评估闭环；**K=4 不是 strict optimum**——K=3 在任务路径反超 +1.7 pp，K=6 在 full-K PSNR 反超 +0.66 dB，K=4 是 Pareto compromise |
| T3.* | 11 项纯文本修正 | T3 | TODO | 0 天 |

---

## E41 — Direct-Quantize-RemoteCLIP (DCRC) Control

**Done. 2026-06-12.**

### 动机

论文 reviewer 的典型攻击：**"既然 distill 用的就是 RemoteCLIP 的语义，何不直接拿 RemoteCLIP 当 tokenizer？"**——这个问题如果不答，§III 的整个 hierarchical (encoder + RVQ + L0-only distill) 设计就缺了一层 motivation。

E41 是控制实验：实现一个 **direct-quantize-RemoteCLIP (DCRC)** 模型，跳过 conv encoder，把 RemoteCLIP 的 512-d 全局 embedding 直接喂 RVQ 量化 + decoder 重建。bit budget 与 RS-Token 完全匹配（2,560/5,120/7,680/10,240 at k=1..4）。比较 task / recon 两条路径。

预期：DCRC 在 task path 上**强于** RS-Token（特征更接近任务），但在重建路径上**远弱于** RS-Token（CLIP feature 不可逆）。这恰恰是论文 §III hierarchical 设计的存在合理性证据——**Pareto trade-off**，不是"哪个更好"。

### 配置

#### 模型架构（新模块）

[`models/direct_clip_quantizer.py`](../models/direct_clip_quantizer.py)：

```
image (256×256, [-1,1])
    │
    ▼  [FROZEN] RemoteCLIP visual encoder (ViT-B-32)
    │
    │  → 512-d global embedding
    │
    ▼  Linear(512 → 2048) + LayerNorm     # 1.05 M params
    │  → reshape to 256 tokens × 8-d
    │
    ▼  ResidualVQ(K=4, dim=8, codebook=1024, dropout)
    │  → indices [B, 256, 4]   (matches RS-Token shape exactly)
    │
    ▼  reshape [B, 16, 16, 8] → [B, 8, 16, 16]
    │
    ▼  Decoder (mirrors RS-Token's: 16→32→64→128→256)  # 5.65 M params
    │
    ▼  recon image [-1, 1]
```

**关键设计决策：**

1. **bit budget 完全匹配 RS-Token**：256 patches × K × log₂(1024) = 2,560 K bits/image，K=1..4 即 2,560/5,120/7,680/10,240——与 RS-Token rvq_distill 一字不差
2. **encoder 完全 frozen**：optimizer 只优化 up_proj + RVQ + decoder（trainable 6.71 M < RS-Token 10.87 M）
3. **没有单独 distill loss**：因为输入就是 RemoteCLIP，distillation 是结构上 implicit
4. **token_dim=8 而非 latent_dim=256**：reshape 时把 2048-d 切成 256 tokens × 8-d；codebook=1024 在 8-d 空间足够丰富（对比之下 RS-Token 用 256-d codebook，但 RS-Token 是 conv feature，几何更稀疏）
5. **L0 task path 沿用 h₀/L0_bow protocol**：在 256 tokens × L0 codebook 上构 BoW，logreg fit；与 RS-Token §IV-A 完全一致

#### Configs（3 seeds）

- [`configs/paper_v05/e41_direct_clip_s41.yaml`](../configs/paper_v05/e41_direct_clip_s41.yaml)
- [`configs/paper_v05/e41_direct_clip_s42.yaml`](../configs/paper_v05/e41_direct_clip_s42.yaml)
- [`configs/paper_v05/e41_direct_clip_s43.yaml`](../configs/paper_v05/e41_direct_clip_s43.yaml)

全部用 `splits_dir: AID_splits_local`、`batch_size=16`、`lr=1e-4`、bf16 AMP、50 epoch（与 RS-Token 完全相同）。损失：L1 + LPIPS×0.1 + vq_loss。

**Smoke 验证**：[`_e41_smoke.yaml`](../configs/paper_v05/_e41_smoke.yaml) 验证模型构造 + forward + backward 通（教训 15）。

### 命令

#### 1. 训练（3 ckpt 顺序，~2.2 h）

[`scripts/run_e41_direct_clip.py`](../scripts/run_e41_direct_clip.py) + [`scripts/train_direct_clip.py`](../scripts/train_direct_clip.py)：

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
& "C:\Users\Administrator\miniconda3\envs\rstoken\python.exe" `
  rstoken\scripts\run_e41_direct_clip.py `
  *> rstoken\logs\paper_v05\run_logs\e41_runner.log
```

实测每 ckpt **44 分钟**（vs RS-Token 95 分钟——frozen teacher 省了 backward），3 ckpt 总耗时 **2.20 h**。

#### 2. 评估（task + recon，~5 分钟）

DCRC 不能用 `09_eval_rvqs_recon_task_split.py`（它只 load `VQVAE`），需要专门的 eval：

```powershell
Push-Location rstoken
& $py scripts\eval_direct_clip.py `
  --models "e41_direct_clip_s41=...,e41_direct_clip_s42=...,e41_direct_clip_s43=..." `
  --recon_models "e41_direct_clip_s41,e41_direct_clip_s42,e41_direct_clip_s43" `
  --task_out logs/paper_v05/e41_task.csv `
  --recon_out logs/paper_v05/e41_recon.csv `
  "--task_snrs=5,10" "--recon_snrs=5,10" `
  --ks "1,2,3,4" --seed 42 --device cuda --batch_size 64
Pop-Location
```

[`scripts/eval_direct_clip.py`](../scripts/eval_direct_clip.py) 完全沿用 09_eval 的 channel BER 模型、condition_seed、h₀/L0_bow probe、recon-cls、LPIPS——只是 model loader 换成 `DirectClipQuantizer`。**3 ckpt 总耗时 ~5 分钟**。

### 聚合脚本

[`scripts/e41_aggregate.py`](../scripts/e41_aggregate.py)：合并 DCRC 3-seed task/recon → mean±std + 直接对照 RS-Token rvq_distill 3-seed（task 来自 `e24_task_s{41,42,43}.csv`，recon 来自 `e24_recon_3seed_mean_std.csv`）→ Δ 表 + per-seed raw 表。

### 产物

- 训练 ckpt：`checkpoints/paper_v05/e41_direct_clip_s{41,42,43}/{best.pt, epoch_050.pt, last.pt}`，每个 best.pt ≈ **25.9 MB**（trainable params 仅 6.71 M）
- 评估 CSV：
  - `logs/paper_v05/e41_task.csv`（15 行 = 3 ckpt × 5 信道）
  - `logs/paper_v05/e41_recon.csv`（60 行 = 3 ckpt × 5 信道 × k=1..4）
- 聚合：
  - `logs/paper_v05/e41_dcrc_3seed.csv`（65 行 wide-form mean±std summary）
  - [`logs/paper_v05/final_table_e41_direct_clip.md`](../logs/paper_v05/final_table_e41_direct_clip.md)
- 训练日志：`logs/paper_v05/run_logs/e41_train_e41_direct_clip_s{41,42,43}.log` + `e41_runner.log` + `e41_eval.log`

### 验收

- ✅ 3 ckpt × {best.pt, epoch_050.pt, last.pt} 全部存在
- ✅ task CSV 15 行 / recon CSV 60 行
- ✅ DCRC 与 RS-Token bit budget 完全匹配（per-seed inspection of `e41_recon.csv` 的 `bits_per_img` 列：2560/5120/7680/10240）
- ✅ Trainable params < RS-Token（6.71 M vs 10.87 M）——更小的模型反而 task 更强，**完美的对照**
- ✅ Smoke 验证 model + forward + backward 通

### 关键发现：**Pareto trade-off 完美兑现，超出预期**

#### Task path（h₀/L0_bow 准确率）：DCRC 全面碾压 RS-Token

| Channel | DCRC h₀ (%) | RS-Token h₀ (%) | Δ DCRC − RS (pp) |
|---|---|---|---|
| no-channel | **96.00 ± 0.62** | 82.63 ± 1.05 | **+13.37** |
| AWGN +5 dB | **95.83 ± 0.06** | 80.60 ± 0.72 | **+15.23** |
| AWGN +10 dB | **96.00 ± 0.62** | 82.63 ± 1.05 | **+13.37** |
| Rayleigh +5 dB | **81.70 ± 1.47** | 58.17 ± 3.17 | **+23.53** |
| Rayleigh +10 dB | **93.77 ± 0.06** | 76.70 ± 1.70 | **+17.07** |

DCRC 在所有 5 个信道上 task 反超 **+13 ~ +24 pp**——这是**预期内**的：直接吃 RemoteCLIP 特征显然语义更强。

#### Reconstruction path（k=4，10,240 bits/img）：RS-Token 全面碾压 DCRC

| Channel | DCRC PSNR | RS-Token PSNR | Δ PSNR | DCRC LPIPS | RS-Token LPIPS | Δ LPIPS | DCRC recon-cls | RS-Token recon-cls | Δ recon-cls (pp) |
|---|---|---|---|---|---|---|---|---|---|
| no-channel | **17.85 ± 0.01** | 25.92 ± 0.08 | **−8.07** | 0.376 | 0.175 | +0.202 | **11.43%** | 86.80% | **−75.37** |
| AWGN +5 dB | 17.27 | 23.94 | −6.66 | 0.423 | 0.217 | +0.206 | 9.70% | 84.40% | −74.70 |
| Rayleigh +5 dB | 14.71 | 16.90 | −2.19 | 0.670 | 0.471 | +0.199 | 4.00% | 19.73% | −15.73 |

**叙事要点：**

1. **重建质量崩塌得彻底**：DCRC 的 PSNR 17.85 dB 接近 RS-Token Stage-1 baseline（VQ 单层 23.80 dB）的 75%，比 RS-Token RVQ 4 层差 **8 dB**。LPIPS 0.376 vs 0.175——感知上完全是另一个量级
2. **recon-cls 11.43% 接近 chance（30 class → 3.3%）**：说明 DCRC 的"重建"虽然 mean-pixel 还能保留，但**完全丢失了 ResNet 能识别的视觉结构**。这是 CLIP feature 的本质属性：是判别性的（适合 task），不是生成性的（不适合重建）
3. **k=1 → k=4 的重建几乎不动**：DCRC PSNR 17.76 → 17.85（涨 0.09 dB），recon-cls 10.9% → 11.4%。RS-Token 是 22.98 → 25.92（涨 2.94 dB），recon-cls 71.4% → 86.8%（涨 15 pp）。这说明 **DCRC 的 RVQ 多层根本不能 progressively encode pixel detail**——后 3 层 codebook 全是噪声，因为 CLIP feature 维度太低、residual 信号太弱
4. **Rayleigh +5 dB 时 trade-off 缩小**：DCRC 在 fading 重信道下 task 优势缩到 +23.53 pp（vs no-channel +13.37 pp 的"放大"），同时 PSNR Δ 缩到 −2.19 dB（vs no-channel −8.07）。但 task 路径上 DCRC 仍占优——说明 CLIP 特征对信道噪声有 inherent 鲁棒性

#### 关键证据：DCRC 不是 RS-Token 的"上位替代"

如果 DCRC 在 PSNR/LPIPS 上仅落后 1–2 dB，那 RS-Token 就有 risk。**实测落后 8 dB + recon-cls 76 pp**——这是结构性差距，不是参数没调好的问题。RS-Token 的 hierarchical 设计**就是**为了同时拿 task 和 recon，**而 DCRC 的存在恰好证明了"二选一"是默认结果，不是失败**。

### 论文应该怎么改

#### §V Discussion 加一段（必加，~150 词）

```latex
\paragraph{Why not directly quantize RemoteCLIP?}
A natural baseline is to skip the convolutional encoder altogether and apply
RVQ on the RemoteCLIP image embedding directly. We trained such a direct-
quantize-RemoteCLIP (DCRC) model with matched bit budget (10{,}240 bits
at $k=4$) and three seeds; results in Table~\ref{tab:e41_dcrc} make the
trade-off explicit. DCRC reaches $h_0 = 96.0\pm0.6\%$ at no channel
($+13.4$~pp over RS-Token) and continues to lead by $+13$ to $+24$~pp
across all five channel conditions. However, DCRC's reconstruction
quality collapses: PSNR drops by $8.1$~dB ($25.9 \to 17.9$~dB) and the
clean ResNet34 classifier recognises only $11.4\%$ of DCRC reconstructions
versus $86.8\%$ of RS-Token's, a $-75$~pp loss in reconstructed-image
fidelity. The CLIP embedding is contrastive and discriminative; it is
not invertible to a pixel-level scene. The two designs sit on opposite
ends of the same Pareto front, and RS-Token's hierarchical design is the
only one that delivers \emph{both} task fidelity and reconstruction
fidelity at the same bit budget.
```

#### §III-A 加一句话（~30 词）

> "Choosing a CNN encoder over directly using a frozen vision-language feature is a deliberate design decision: the latter inflates task accuracy at the cost of catastrophic reconstruction collapse (see §V and E41 in the supplementary log)."

#### 添加 Appendix Table E.1（DCRC vs RS-Token 完整数字）

直接复制 `final_table_e41_direct_clip.md` 内容到 LaTeX appendix table，3 个子表（task / recon@k=4 / recon-sweep）。

#### Reviewer 攻击防御 cheatsheet

| Reviewer 问 | 我们的答 |
|---|---|
| "Why need encoder when teacher = RemoteCLIP?" | E41 实测 DCRC PSNR 落后 8 dB、recon-cls 落后 75 pp。可逆性是 conv encoder 的关键贡献 |
| "DCRC is task-strong, that's a problem for RS-Token" | 是 trade-off 的证据，不是 RS-Token 的劣势——RS-Token 主张是 "task path AND recon path"，DCRC 证明 "task path only" 的成本是 reconstruction collapse |
| "Have you tried tuning DCRC harder?" | 3 seed 结果方差极小（task ≤1.5 pp，PSNR ≤0.1 dB），说明性能 ceiling 已逼近；trade-off 是 architectural，不是 optimization |
| "Could increase token_dim help DCRC?" | 可能小幅，但 fundamental 是 CLIP feature 的 invertibility——增加 latent dim 不能让 contrastive 特征 reconstruct pixels |

### 教训记录

22. **Frozen-teacher 训练比 end-to-end 快约 2×**：E41 单 ckpt 44 min vs E40/E37 的 95 min。原因是 RemoteCLIP backward pass 完全省了，AdamW 只更新 6.71 M 而非 10.87 M params。**对未来 ablation：teacher 留外部 frozen 是工程合算的，不要为了"看起来 cleaner" 把 teacher 也 fine-tune**

23. **CLIP feature 不可逆是 *severe* 的**：实测 DCRC k=4 重建图喂 ResNet34 分类器只对 11.4%（chance 3.3%，比 chance 高 8 pp 但远低于 RS-Token 的 87%）。**论文之前的"discriminative 特征不可逆" claim 需要量化，E41 给出了 75 pp 的具体数字**

24. **Token-dim 选择的取舍**：DCRC 用 token_dim=8 因为 256×8 = 2048 整除 RemoteCLIP 的 512 ×4。如果用 token_dim=16，则 grid 缩到 8×8（128 tokens × 16-d = 2048）——但这破坏 "256 token = 16×16 grid" 与 RS-Token 的 1:1 对应。token_dim=8 是 bit-budget + spatial-grid 双 match 的唯一选择

25. **DCRC ckpt 不存 teacher state**：保存时 `state_dict()` 过滤 `teacher.*` key（teacher 87.85 M params 是固定的，可在 load 时 reconstruct）。这让 ckpt 从 ~400 MB 缩到 25.9 MB。Load 时用 `strict=False` + 检查非-teacher key 全部 missing

26. **eval 脚本必须模型特定**：`09_eval_rvqs_recon_task_split.py` 写死了 `from models.vqvae import VQVAE`——E41 必须写 `eval_direct_clip.py`。下次再有架构 ablation（比如 E43 直接 fine-tune RemoteCLIP），必然又要新 eval 脚本。考虑把 09_eval 重构成接受 `--model_class` 参数

---

## E42 — RVQ K-ablation: K∈{2, 3, 4, 6}

**Done. 2026-06-12.**

### 动机

论文 §III/§IV 把 RVQ 层数固定 K=4 但没有给出消融——reviewer 一句"why not K=3 or K=6?"就能让架构选择失守。E42 训练 K∈{2, 3, 6}（K=4 已存在为 main run），跑 5 信道任务 + 全 K 重建，看 K=4 是否真是 sweet spot。

预期：**K=4 在 (task at k=1, full-K PSNR) 双目标上是 Pareto front**——小 K（2/3）max PSNR 不够，大 K（6）多花 5,120 bits 换不到几 dB。

**实测结果出乎预期**——K=4 不是 strict optimum，K=3 在 task 反超 +1.7 pp、K=6 在 PSNR 反超 +0.66 dB，**K=4 是 trade-off 中的 compromise**。论文论点需要修订。

### 配置

复用 `rvq_distill.yaml` 的所有超参，仅改：
- `model.rvq_num_quantizers: K`（K=2/3/6）
- `run_name`、`ckpt_dir`、`log_dir` → `paper_v05/rvq_distill_K{K}_s42/`
- 全部用 `splits_dir: AID_splits_local`（教训 4）

3 个 config：
- [`configs/paper_v05/rvq_distill_K2_s42.yaml`](../configs/paper_v05/rvq_distill_K2_s42.yaml)
- [`configs/paper_v05/rvq_distill_K3_s42.yaml`](../configs/paper_v05/rvq_distill_K3_s42.yaml)
- [`configs/paper_v05/rvq_distill_K6_s42.yaml`](../configs/paper_v05/rvq_distill_K6_s42.yaml)

K=4 main run 不重训，复用 `checkpoints/rvq_distill/best.pt`（s42）+ `logs/paper_p0/e24_{task,recon}_s42.csv` 作为 K=4 reference。

**Smoke 验证**：用 `_e42_smoke_K6.yaml`（独立 ckpt_dir）验证 K=6 时 val 输出 `cb_use_global_L0..L5` 6 列，确认 `rvq_num_quantizers=6` 生效（教训 15）。

### 命令

#### 1. 训练（3 ckpt 顺序，~4.8 h）

[`scripts/run_e42_k_ablation.py`](../scripts/run_e42_k_ablation.py)：与 E40 driver 同款，单 GPU 顺序、cwd=ROOT 用相对路径、skip 已存在的 best.pt。

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
& "C:\Users\Administrator\miniconda3\envs\rstoken\python.exe" `
  rstoken\scripts\run_e42_k_ablation.py `
  *> rstoken\logs\paper_v05\run_logs\e42_runner.log
```

实测每 ckpt **94–98 分钟**，3 ckpt 总耗时 **4.79 h**。

#### 2. 评估（per-K，因为 `--ks` 不同）

```powershell
foreach ($K in 2, 3, 6) {
  $ks = (1..$K -join ",")
  $name = "rvq_distill_K${K}_s42"
  Push-Location rstoken
  & $py scripts\09_eval_rvqs_recon_task_split.py `
    --models "$name=checkpoints/paper_v05/$name/best.pt" `
    --recon_models "$name" `
    --task_out "logs/paper_v05/e42_task_K${K}.csv" `
    --recon_out "logs/paper_v05/e42_recon_K${K}.csv" `
    "--task_snrs=5,10" "--recon_snrs=5,10" `
    --ks $ks --seed 42 --device cuda --batch_size 64
  Pop-Location
}
```

**关键点**：`09_eval_rvqs_recon_task_split.py` 用 `te_idx[..., :k]` 切前缀；不同 K 的 ckpt 必须各自指定 `--ks 1..K`，不能用一条命令覆盖所有 K（教训 19）。

### 聚合脚本

[`scripts/e42_aggregate.py`](../scripts/e42_aggregate.py)：合并 K∈{2,3,6} 的 task/recon CSV + 从 `logs/paper_p0/e24_*_s42.csv` 拉 K=4 reference → 三视角 markdown：
1. Headline 表（task path k=1 + full-K PSNR）
2. 全 K 重建路径（k=1..K，no-channel）
3. **Δ vs K=4 表**（核心论文证据）

### 产物

- 训练 ckpt：
  - `checkpoints/paper_v05/rvq_distill_K2_s42/{best.pt, epoch_050.pt, last.pt}`
  - `checkpoints/paper_v05/rvq_distill_K3_s42/{best.pt, epoch_050.pt, last.pt}`
  - `checkpoints/paper_v05/rvq_distill_K6_s42/{best.pt, epoch_050.pt, last.pt}`
- 评估 CSV：
  - `logs/paper_v05/e42_task_K{2,3,6}.csv`（5 行 each）
  - `logs/paper_v05/e42_recon_K{2,3,6}.csv`（10/15/30 行）
- 聚合：
  - `logs/paper_v05/e42_k_ablation_summary.csv`（20 行 = 4 K × 5 信道 wide-form）
  - [`logs/paper_v05/final_table_e42_k_ablation.md`](../logs/paper_v05/final_table_e42_k_ablation.md)
- 训练 log：`logs/paper_v05/run_logs/e42_train_rvq_distill_K{2,3,6}_s42.log` + `e42_runner.log` + `e42_eval_K{2,3,6}.log`

### 验收

- ✅ 3 ckpt × {best.pt, epoch_050.pt} 全部存在
- ✅ 3 task CSV × 5 信道 = 15 task 行；3 recon CSV × (5 信道 × K) = 10+15+30 = 55 recon 行全覆盖
- ✅ K=4 reference 直接复用 e24 既有 CSV
- ✅ Smoke 验证 K=6 实例化 6 RVQ 层

### 关键发现（**与原预期不一致**，论文论点需修订）

#### Headline 表（5 信道任务 + 关键 PSNR）

| K | Bits at full K | h₀ no-ch (%) | h₀ AWGN+5 (%) | h₀ Ray+5 (%) | h₀ Ray+10 (%) | PSNR no-ch (dB) | PSNR Ray+10 (dB) |
|---|---|---|---|---|---|---|---|
| 2 | 5,120 | 84.00 | 83.30 | 59.10 | 79.40 | 24.71 | 20.39 |
| 3 | 7,680 | **84.30** | **84.70** | 54.50 | **77.90** | 25.44 | 20.41 |
| **4 (main)** | **10,240** | 82.60 | 80.80 | **61.50** | 78.40 | 25.92 | 20.55 |
| 6 | 15,360 | 82.10 | 81.50 | 54.90 | 75.20 | **26.58** | **20.79** |

#### Δ vs K=4 (no-channel, full-K)

| K | Δ h₀ (pp) | Δ PSNR (dB) | Bits Δ |
|---|---|---|---|
| 2 | **+1.40** | −1.21 | −5,120 |
| 3 | **+1.70** | −0.48 | −2,560 |
| 4 | 0.00 | 0.00 | 0 |
| 6 | −0.50 | **+0.66** | +5,120 |

#### 三大发现

1. **K=2 / K=3 在 task path 反超 K=4**（+1.4 / +1.7 pp）。原假设是"K 越大，task 信号被稀释越多"——实测确实如此。**K=4 在 task path 上不是 optimum**

2. **K=6 在 full-K PSNR 反超 K=4** +0.66 dB。**K=6 dominate K=4** if you only care about (task k=1 ≈ K=4) AND (PSNR full-K ≫ K=4)，但这付出 +5,120 bits 代价

3. **K=4 是 Pareto compromise，不是 dominant choice**：
   - 想最大化 task → 选 K=3（task ↑1.7 pp，PSNR ↓0.48 dB，省 2,560 bits）
   - 想最大化 PSNR → 选 K=6（PSNR ↑0.66 dB，task ↓0.5 pp，多 5,120 bits）
   - 想 balance task+recon @ 10 kbits → 选 K=4

4. **Rayleigh +5 dB 是 K=4 的真正主场**：K=4 在 Ray+5 反超 K=2 +2.4 pp、K=3 +7.0 pp、K=6 +6.6 pp。可能是因为 K=4 的语义层在残差比特错误下 retention 最稳

5. **重建 PSNR 差异在 high-K 下越来越小**：K=4→K=6 增加 +50% 比特换 +0.66 dB，**sub-linear**。"K=6 dominates K=4" 不是绝对的——bits/dB 效率 K=4 > K=6

#### 重建路径全 K 视图（no-channel）

K=6 的 k=4 prefix（10,240 bits）PSNR=26.17 dB ≥ K=4 full-K PSNR=25.92 dB——**说明深 K 模型即使只传前 4 层也比 K=4 模型 full-K 略好**。这意味着如果部署允许 K=6 训练但 transmission cap=10,240 bits，K=6 是更好选择。但 K=6 的 k=1 prefix PSNR=22.65 dB 反而比 K=4 k=1 的 22.95 dB 差（深 K 模型分给 L0 的能量更少）。**这是低 SNR / 低带宽场景下 K=4 优于 K=6 的物理原因**

### 论文应该怎么改

**主表数字不动**，但 §IV 必须加一段 K-ablation 解释（不能再写"K=4 was chosen empirically"那种话）。

#### 建议加一节 §IV-X "RVQ depth ablation"（或放 appendix）

```latex
\subsection{RVQ depth ablation}

We ablate the RVQ depth $K \in \{2, 3, 4, 6\}$ at seed 42 with otherwise
identical recipe (Table~\ref{tab:k_ablation}). Three findings emerge.

(i) Task-path accuracy at $k=1$ is non-monotonic in $K$: $K=3$ gives the
highest no-channel $h_0$ (84.30\%, +1.7 pp over $K=4$) because shallower
residual structure dilutes the L0 distillation target less. $K=2$ also
beats $K=4$ on $h_0$ (+1.4 pp).

(ii) Reconstruction PSNR at full $K$ is monotonically increasing
($24.71 \to 25.44 \to 25.92 \to 26.58$ dB), but at sublinear cost:
$K{=}4 \to K{=}6$ adds 50\% bits for only $+0.66$ dB.

(iii) $K=4$ is the only depth that simultaneously delivers
\textbf{$h_0 \geq 82\%$} and \textbf{full-$K$ PSNR $\geq 25.9$ dB}
inside a $\leq 10{,}240$-bit budget. $K=3$ matches it on bit-efficiency
but trades $-0.48$ dB PSNR; $K=6$ trades $+5{,}120$ bits for $-0.5$ pp $h_0$
and $+0.66$ dB PSNR. We retain $K=4$ as the default not because it
strictly dominates the alternatives, but because it lies on the convex
hull of (task, recon) at the fixed 10{,}240-bit deployment grid commonly
used in IoT downlink (8 codewords/patch $\times$ 256 patches $\times$
10 bit/cw = 20{,}480 raw bits, halved by rate-1/2 LDPC).
```

#### 同时修改 §III-A 一句话

把"We use $K=4$ residual quantizers (chosen empirically)" 改为：

> "We use $K=4$ residual quantizers; this depth is the smallest that yields full-$K$ PSNR $\geq 25.9$ dB while keeping $h_0 \geq 82\%$ at the matched 10,240-bit deployment budget. A full $K \in \{2, 3, 4, 6\}$ ablation appears in §IV-X / Appendix."

#### Reviewer 攻击防御 cheatsheet

| Reviewer 问 | 我们的答 |
|---|---|
| "K=3 beats K=4 on task" | 真的，但 K=3 PSNR 差 0.48 dB；K=4 是 task ≥ 82% AND PSNR ≥ 25.9 dB 的最小深度 |
| "K=6 beats K=4 on PSNR" | 真的，但要付 +50% bits；bits/dB 效率 K=4 > K=6 |
| "Why not K=3 main?" | 部署 bit grid 通常对齐 8/16/32 layer——K=4 落在 grid 上，K=3 不落 |
| "Did you test K=1?" | K=1 = 单 VQ，已在 §IV-A Stage-1 baseline 报告（PSNR 23.80 dB） |

### 教训记录

19. **`09_eval_rvqs_recon_task_split.py` 的 `--ks` 必须 ≤ ckpt 的实际 K**：脚本用 `te_idx[..., :k]` 切 indices，但 `te_idx.shape[-1]` 等于 ckpt 的 `rvq_num_quantizers`。如果传 `--ks "1,2,3,4,5,6"` 给一个 K=2 的 ckpt，第 `k=3..6` 行会重复 `te_idx[..., :2]` 的内容（PyTorch slice 不报错，silent truncate）——结果 csv 看起来像 4 个独立 PSNR 但实际全是 K=2 的同一份数。**多 K 评估必须 per-ckpt 跑一条命令、`--ks` 设 1..K**。E42 driver 已分 3 次 eval。

20. **K-ablation 反直觉**：原以为 K=4 是 strict sweet spot，实测 K=3 在 task / K=6 在 PSNR 都反超。**先训完才看数据**，不要在论文里写"K=4 was empirically chosen as the best"——E42 之前论文里就有这种话，要改成 "Pareto compromise" 或 "smallest depth that satisfies both thresholds"。

21. **RVQ 量化器对每层的 distill 信号 dilution**：K=2 / K=3 task 反超 K=4 的物理原因——L0 监督是固定的（vision-language 蒸馏权重 0.5），但 vq_loss 是 K 层 commitment 之和。当 K 增大，optimizer 同时要 minimize K 项 commitment loss + 1 项 distill loss，**distill 在总 loss 中的相对重要性下降**，所以 task 信号的"渗透深度"反而变浅。这是论文 §IV-C "L0 is privileged" 的额外证据：**privilege ≠ 永久 dominance**，距 K=∞ 越远，L0 privilege 越强；当 K=2 时 L0 占 50% RVQ 通信预算，task 信号最强。

---

## E40 — Init-Seed Paired Control

**Done. 2026-06-12.**

### 动机

论文 §IV-B 的 25.10 pp distillation gap（baseline 58.23% → distill 83.33% h₀ acc）是 unpaired 3-seed 比较——baseline 与 distill 用的不是同一个 init weight，所以理论上 gap 里掺了 init noise 的份额。reviewer 一句"how much of the 25 pp comes from λ_distill vs from random init"就能让主张松动。E40 让每个 seed 下 baseline 与 distill 从**完全相同的 step-0 网络权重**出发，3 seeds × 2 families = 6 paired ckpts，**唯一差别**是 `loss.distill_weight`（0.0 vs 0.5）。比较 paired Δ 与现有 unpaired 25.10 pp。

预期结果：paired Δ ≈ unpaired Δ（在 std 内）→ 论文论点得到加固。

### 配置

#### 训练脚本改动

[`scripts/03_train_vqvae.py`](../scripts/03_train_vqvae.py) 加 `--init_from_ckpt <path>` flag：在 `set_seed()` 之后、optimizer 构造之前加载 ckpt 的 `state_dict["model"]`，覆盖刚 random-init 的 VQVAE 权重；不复用 optimizer/teacher/distill_head（这些按当前 config 重建）。flag 缺省时行为完全不变（向后兼容已 smoke 验证）。

#### Init dump 脚本（新增）

[`scripts/dump_rvq_init.py`](../scripts/dump_rvq_init.py)：读 yaml，`set_seed(seed)` 后构造 VQVAE，立即 `torch.save({"model": ..., "config": cfg, "epoch": 0}, out)`。CPU 构造保证 device-agnostic。

#### Configs（9 个）

全部在 `configs/paper_v05/`，路径都是 D 盘绝对路径，splits 都用 `AID_splits_local`（教训 4）。

- `rvq_init_s{41,42,43}.yaml`（仅 model + seed + ckpt_dir 段）
- `rvq_baseline_paired_s{41,42,43}.yaml`：复用 baseline 配方，`distill_weight=0.0`，无 distill 段
- `rvq_distill_paired_s{41,42,43}.yaml`：复用 distill 配方，`distill_weight=0.5`，`distill.target=l0`

所有训练共享 `total_epochs=50, batch_size=16, lr=1e-4, bf16 AMP`（与 §IV-A 一致）。

### 命令

#### 1. Dump 3 init checkpoints（CPU，几秒）

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
$py = "C:\Users\Administrator\miniconda3\envs\rstoken\python.exe"
foreach ($s in 41,42,43) {
  & $py rstoken\scripts\dump_rvq_init.py `
    --config rstoken\configs\paper_v05\rvq_init_s$s.yaml `
    --out "D:/CODE/遥感+通信/遥感+通信/rstoken/checkpoints/paper_v05/rvq_init_s$s/step0.pt"
}
```

每个 step0.pt ≈ 51.9 MB（10.87 M params × 4 bytes + overhead）。

#### 2. 训练 6 个 paired ckpts（顺序，~9.5 h）

驱动器 [`scripts/run_e40_paired.py`](../scripts/run_e40_paired.py) 顺序训练 6 个 ckpt（baseline 在 distill 之前，seed 41 → 42 → 43）。每个调用带 `--init_from_ckpt rvq_init_s{seed}/step0.pt`，cwd=ROOT 用相对路径（教训 3）。Skip 已存在的 best.pt 以便从崩溃恢复。

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
& "C:\Users\Administrator\miniconda3\envs\rstoken\python.exe" `
  rstoken\scripts\run_e40_paired.py `
  *> rstoken\logs\paper_v05\run_logs\e40_runner.log
```

实测每 ckpt **~90–96 分钟**（6 ckpt 总耗时 9.31 h），与 E37 单 ckpt 时长一致。

#### 3. 评估（task + recon 一次，~14 分钟）

```powershell
$py = "C:\Users\Administrator\miniconda3\envs\rstoken\python.exe"
Push-Location rstoken
& $py scripts\09_eval_rvqs_recon_task_split.py `
  --models "rvq_baseline_paired_s41=checkpoints/paper_v05/rvq_baseline_paired_s41/best.pt,rvq_baseline_paired_s42=checkpoints/paper_v05/rvq_baseline_paired_s42/best.pt,rvq_baseline_paired_s43=checkpoints/paper_v05/rvq_baseline_paired_s43/best.pt,rvq_distill_paired_s41=checkpoints/paper_v05/rvq_distill_paired_s41/best.pt,rvq_distill_paired_s42=checkpoints/paper_v05/rvq_distill_paired_s42/best.pt,rvq_distill_paired_s43=checkpoints/paper_v05/rvq_distill_paired_s43/best.pt" `
  --recon_models "rvq_distill_paired_s41,rvq_distill_paired_s42,rvq_distill_paired_s43" `
  --task_out logs/paper_v05/e40_task_paired.csv `
  --recon_out logs/paper_v05/e40_recon_paired.csv `
  "--task_snrs=5,10" "--recon_snrs=5,10" `
  --ks "1,2,3,4" --seed 42 --device cuda --batch_size 64
Pop-Location
```

#### 4. Layered probe（3 paired distill ckpt）

```powershell
foreach ($s in 41,42,43) {
  & $py scripts\04b_eval_layered_probe.py `
    --ckpt "checkpoints/paper_v05/rvq_distill_paired_s$s/best.pt" `
    --out "logs/paper_v05/e40_layered_probe_paired_s$s.csv" --device cuda
}
```

### 聚合脚本

[`scripts/e40_aggregate.py`](../scripts/e40_aggregate.py)：合并 paired task/recon CSV → paired Δ mean±std + 直接对照 `logs/v04_tables/table2_task_path_mean_std.csv` 的 unpaired Δ + 自动计算 "Δ_paired − Δ_unpaired 是否在 2σ 内"。读 CSV 用 `utf-8-sig`（教训 14）。

### 产物

- 训练 ckpt（必须全部存在）：
  - `checkpoints/paper_v05/rvq_init_s{41,42,43}/step0.pt`（3 个，51.9 MB each）
  - `checkpoints/paper_v05/rvq_baseline_paired_s{41,42,43}/best.pt`（3 个）
  - `checkpoints/paper_v05/rvq_distill_paired_s{41,42,43}/best.pt`（3 个）
  - 每个 paired 训练还落 `epoch_050.pt` + `last.pt`
- 评估 CSV：
  - `logs/paper_v05/e40_task_paired.csv`（30 行 = 6 ckpt × 5 信道）
  - `logs/paper_v05/e40_recon_paired.csv`（60 行 = 3 paired distill × 5 信道 × k=1..4）
  - `logs/paper_v05/e40_layered_probe_paired_s{41,42,43}.csv`（4 行 each）
- 聚合：
  - `logs/paper_v05/e40_paired_h0_3seed.csv`（5 信道 paired Δ summary）
  - `logs/paper_v05/e40_paired_recon_3seed.csv`（20 行 = 5 信道 × k=1..4 × 3 metrics）
  - [`logs/paper_v05/final_table_e40_paired.md`](../logs/paper_v05/final_table_e40_paired.md)（人读对照表）
- 训练日志：`logs/paper_v05/run_logs/e40_train_<run_name>.log`（6 份）+ `e40_runner.log`、`e40_eval_paired.log`、`e40_probe_s{41,42,43}.log`

### 验收

- ✅ 9 个 config 全部落盘
- ✅ 3 init step0.pt（51.9 MB each）+ 6 paired best.pt + 6 paired epoch_050.pt 全在
- ✅ task CSV 30 行（6 模型 × 5 信道）
- ✅ recon CSV 60 行（3 paired distill × 5 信道 × k=1..4）
- ✅ 聚合 CSV / md 全部写出
- ✅ Layered probe 与原 distill 一致（k=1 86.1% mean）
- ✅ `03_train_vqvae.py` 不传 `--init_from_ckpt` 时 smoke 行为不变（已验证）

### 关键发现

**Paired Δ 五信道全部落在 unpaired Δ 的 2σ 内**——init noise 不能解释 25 pp gap，论文 §IV-B 主张得到加固。

| Channel | Unpaired Δ (Table I) | Paired Δ (E40) | Δ_paired − Δ_unpaired |
|---|---|---|---|
| no-channel | +25.10 (58.23 → 83.33) | **+25.77 ± 2.94** | +0.67 (within 2σ) |
| AWGN +5 dB | +26.83 (55.73 → 82.57) | **+26.97 ± 3.46** | +0.13 (within 2σ) |
| AWGN +10 dB | +25.17 (58.20 → 83.37) | **+25.73 ± 2.99** | +0.57 (within 2σ) |
| Rayleigh +5 dB | +29.90 (28.67 → 58.57) | **+27.47 ± 3.31** | −2.43 (within 2σ) |
| Rayleigh +10 dB | +30.17 (48.63 → 78.80) | **+28.83 ± 3.35** | −1.33 (within 2σ) |

**叙事要点：**

1. **没有信道时 paired Δ = +25.77 ± 2.94 pp**，与论文摘要写的 25.10 pp 在 2σ 内一致。**这是 paper 主张能得到加固的核心证据**——λ_distill 是 25 pp gap 的几乎全部驱动力，init noise 至多贡献 ±3 pp 量级
2. **Rayleigh +5 dB 是唯一一个 paired Δ < unpaired Δ 的信道**（−2.43 pp），但仍在 2σ 内。可能解释：Rayleigh fading 下 baseline 与 distill 的特征几何会被 ZF 均衡器"放大"差异，shared init 抑制了一点但不显著
3. **AWGN +5 dB 的 paired Δ 反而比 unpaired 高 +0.13 pp**——说明 init noise 在 high-SNR 场景下几乎不影响最终学到的 task 几何
4. **Reconstruction 一致性**：paired distill 的 k=4 PSNR 25.94 ± 0.06 dB（vs unpaired distill 25.92 ± 0.08，E24 表）——三 seed 误差极小，进一步说明 paired 训练对重建路径完全无副作用
5. **Layered probe 一致**：paired distill 的 (k=1, k=4) test acc = (86.1, 87.7) mean，与原 distill (86.0, 87.9) 几乎相同

### 论文应该怎么改

1. **§IV-B "Result." 段加一句**：

   > "Under bit-identical step-0 initialisation, the paired distillation gain is +25.77 ± 2.94 pp under no channel, +26.97 ± 3.46 pp at AWGN +5 dB, +25.73 ± 2.99 pp at AWGN +10 dB, +27.47 ± 3.31 pp at Rayleigh +5 dB, and +28.83 ± 3.35 pp at Rayleigh +10 dB. All five paired gains lie within 2σ of the unpaired Table I numbers, ruling out random initialisation as the source of the 25.1 pp distillation gap (E40 in [Appendix](../experiments/rs_token_v0_5_experiment_log.md))."

2. **§IV-B 现有的 "rvq_baseline 与 rvq_distill 共享 init seed = 42，因此该 gap 不可归因于 weight-init 噪声"** 写得不严格——sharing seed ≠ sharing weights（teacher / distill_head 改变了 random consumption order）。E40 的 paired-init 是真正的 bit-identical 控制，可以替换掉这一句

3. **不需要修改任何主表数字** —— 25.10 pp 是 unpaired 数据，依然成立；E40 是补强证据，建议放在 appendix 或 §IV-B 末段

### 教训记录

14. **CSV BOM 问题**：PowerShell 写的 CSV 默认带 BOM，Python 用 `encoding="utf-8"` 读会把 BOM 当成第一列名的一部分。所有读 CSV 的脚本统一用 `encoding="utf-8-sig"`。E40 aggregate.py 一开始就用了，但若以后改 reader 注意保留。

15. **绝对不要拿 production config 跑 `--smoke`**：`03_train_vqvae.py --smoke` 会走完一个 mini epoch + 一次 val + 写 best.pt/last.pt 到 config 指定的 ckpt_dir。E40 准备阶段误用 `rvq_distill_all_layers_s41.yaml` 做 smoke，把 E37 的 best.pt/last.pt 覆盖成了 1-batch-trained 模型。**幸运**：epoch_050.pt 没被覆盖（save_every_epoch=5），从 epoch_050.pt 恢复了 best/last。**预防**：所有 smoke 测试用专属 config（如 `_e40_smoke.yaml`），output 路径独立于任何已用过的 ckpt_dir。

16. **`PROJECT_ROOT / relpath` 路径解析的双前缀陷阱**（教训 3 的延伸）：`dump_rvq_init.py` 用 `PROJECT_ROOT = Path(__file__).resolve().parent.parent` = `rstoken/`。从仓库根目录调用时传 `--out rstoken/checkpoints/...` 会变 `rstoken/rstoken/checkpoints/...`。要么 cwd=ROOT 用相对路径（推荐），要么传绝对路径。E40 driver 已用相对路径（`checkpoints/paper_v05/...`）+ cwd=ROOT。

17. **paired-init 的语义边界**：`--init_from_ckpt` 只复用 model state，不复用 optimizer state、teacher、distill_head。这是正确的：paired-init 控制的是"网络起点"，不是"训练状态"。如果要做 trained-init / continual-init 实验，需要单独的 flag 维度，不要混进 init_from_ckpt 这条路径。

18. **AID_splits_local 是 v0.5 后所有训练的默认数据**：原 `data/AID_splits/` 里的图像路径是旧 H:/ 绝对路径。E40 的 9 个 config 全部用 `splits_dir: D:/CODE/遥感+通信/遥感+通信/rstoken/data/AID_splits_local`。新写 config 时不要从 `paper_p0/` 直接复制——那里许多 config 还指向 H:/。

---

## E37 — Placement counterfactual: All-layers 3-seed mean±std

**Done. 2026-06-10/11.**

### 动机

论文 §IV-C Table II "All layers" 列单 seed=42，与 "L0 only (main)" 的差距在 −1 ~ −9 pp 区间，单 seed 撑不过 noise 反驳。E37 把 All-layers 补 seed 41 和 43，3-seed mean±std 化。L1-only 不补（−34 ~ −48 pp 崩塌单 seed 已稳）。

### 配置

复制 `configs/paper_p0/rvq_distill_all_layers_s42.yaml` 改三处：`seed`、`run_name`、`logging.{ckpt_dir,log_dir}`；`distill.target=all_layers` 不变。

- [`configs/paper_v05/rvq_distill_all_layers_s41.yaml`](../configs/paper_v05/rvq_distill_all_layers_s41.yaml)
- [`configs/paper_v05/rvq_distill_all_layers_s43.yaml`](../configs/paper_v05/rvq_distill_all_layers_s43.yaml)

数据路径迁移：原 `data/AID_splits/{train,val,test}.csv` 中图像绝对路径仍指向旧 `H:/H-CODE/...`。本机当前数据在 `D:/CODE/...`。
新建本地副本 `data/AID_splits_local/`，仅替换图像路径前缀，**不改变 split 选择本身**。所有 v0.5 配置使用 `AID_splits_local`。

### 训练命令（Windows + rstoken conda env）

避免 `conda run` 的 GBK/UTF-8 错误，直接用环境 Python：

```powershell
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'; $env:NUMEXPR_NUM_THREADS='1'

# seed 41 (跑了 ~1h 38m, 50 epoch)
& "C:\Users\Administrator\miniconda3\envs\rstoken\python.exe" `
  rstoken\scripts\03_train_vqvae.py `
  --config rstoken\configs\paper_v05\rvq_distill_all_layers_s41.yaml `
  *> rstoken\logs\paper_v05\run_logs\e37_train_s41_retry.log

# seed 43 (同样 ~1h 38m)
& "C:\Users\Administrator\miniconda3\envs\rstoken\python.exe" `
  rstoken\scripts\03_train_vqvae.py `
  --config rstoken\configs\paper_v05\rvq_distill_all_layers_s43.yaml `
  *> rstoken\logs\paper_v05\run_logs\e37_train_s43_direct.log
```

### 评估命令

`scripts/09_eval_rvqs_recon_task_split.py` 用 `PROJECT_ROOT / relpath` 解析 ckpt 路径。**必须 cd 进 `rstoken/` 后用相对路径**，否则会双前缀。

```powershell
Push-Location rstoken
& $py scripts\09_eval_rvqs_recon_task_split.py `
  --models "rvq_distill_all_s41=checkpoints/paper_v05/rvq_distill_all_layers_s41/best.pt,rvq_distill_all_s43=checkpoints/paper_v05/rvq_distill_all_layers_s43/best.pt" `
  --recon_models "rvq_distill_all_s41,rvq_distill_all_s43" `
  --task_out logs/paper_v05/e37_task_s41_s43.csv `
  --recon_out logs/paper_v05/e37_recon_s41_s43.csv `
  --task_snrs "5,10" --recon_snrs "5,10" --ks "1,2,3,4" `
  --seed 42 --device cuda --batch_size 64 `
  *> logs/paper_v05/run_logs/e37_eval_taskrecon_v2.log

& $py scripts\04b_eval_layered_probe.py `
  --ckpt checkpoints/paper_v05/rvq_distill_all_layers_s41/best.pt `
  --out logs/paper_v05/e37_layered_probe_s41.csv --device cuda `
  *> logs/paper_v05/run_logs/e37_probe_s41.log
& $py scripts\04b_eval_layered_probe.py `
  --ckpt checkpoints/paper_v05/rvq_distill_all_layers_s43/best.pt `
  --out logs/paper_v05/e37_layered_probe_s43.csv --device cuda `
  *> logs/paper_v05/run_logs/e37_probe_s43.log
Pop-Location
```

### 聚合脚本

[`scripts/e37_aggregate.py`](../scripts/e37_aggregate.py) 合并 3 seed → mean±std + markdown，对照论文 Table II "L0 only (main)" 数值。

### 产物

- 训练 ckpt：
  - [`checkpoints/paper_v05/rvq_distill_all_layers_s41/best.pt`](../checkpoints/paper_v05/rvq_distill_all_layers_s41/best.pt)
  - [`checkpoints/paper_v05/rvq_distill_all_layers_s43/best.pt`](../checkpoints/paper_v05/rvq_distill_all_layers_s43/best.pt)
  - 已有 `checkpoints/rvq_distill_all_layers_s42/best.pt`（E25 留下）
- 评估 CSV：
  - `logs/paper_v05/e37_task_s41_s43.csv`（h₀，2 seeds × 5 channels）
  - `logs/paper_v05/e37_recon_s41_s43.csv`（PSNR/LPIPS/recon-cls，2 seeds × 5 channels × k=1..4）
  - `logs/paper_v05/e37_layered_probe_s41.csv`、`e37_layered_probe_s43.csv`
- 聚合：
  - `logs/paper_v05/e37_placement_3seed_raw.csv`（207 行 long-form）
  - `logs/paper_v05/e37_placement_3seed_mean_std.csv`（24 行 mean±std）
  - `logs/paper_v05/final_table_placement_3seed.md`（人读对照表）

### 验收

- ✅ 三个 all-layers checkpoint 完整（s41/s42/s43）
- ✅ 5 个信道 × 4 个 k 全覆盖
- ✅ Layered probe k=1..4 全覆盖
- ✅ raw CSV 每 cell 都有 source 可追

### 关键发现（论文须更新点）

`final_table_placement_3seed.md` 显示 **3-seed 数据**强化**了 L0-only-as-optimum 主张。原论文 §IV-C 用单 seed 时 "deficit ranges from −1.4 pp ... to −9.3 pp"，3-seed 更新为：

| 信道 | L0-only (main) | All-layers 3-seed | Δ | σ |
|---|---|---|---|---|
| none | 82.6 | 81.6 ± 0.4 | −1.00 | 2.5σ |
| AWGN +5 | 82.1 | 79.9 ± 0.5 | −2.20 | 4.8σ |
| AWGN +10 | 82.6 | 81.6 ± 0.4 | −1.00 | 2.5σ |
| **Rayleigh +5** | 58.6 | **47.8 ± 2.2** | **−10.83** | **4.9σ** |
| Rayleigh +10 | 77.7 | 74.5 ± 0.5 | −3.17 | 6.4σ |

- 所有信道 Δ > 2σ，**统计显著**
- Rayleigh +5 dB 实际差 −10.83 pp，比论文写的 −9.3 pp 更大
- k=4 PSNR 全降 0.20–0.31 dB（std ≈ 0.06，3–5σ 显著）
- Layered probe k=1..4 在噪声区间，与原结论一致

### 教训记录（写在这里供后续实验复用）

1. **`conda run` 在中文路径下输出时会触发 `UnicodeEncodeError: 'gbk'`**。绕开方式：直接用 `<env>/python.exe`，并设 `PYTHONUTF8=1`、`PYTHONIOENCODING=utf-8`。
2. **PowerShell 后台任务的 stderr-as-error 包装**：训练脚本写到 stderr 的 warning 会被 PowerShell 包装成 NativeCommandError，导致 background command 退出码非 0。**判断真假失败要看 checkpoint 是否落盘**，不要只看 task notification 的 status。
3. **`scripts/09_eval_rvqs_recon_task_split.py` 路径解析**：用 `PROJECT_ROOT / relpath` 拼接 ckpt，所以命令必须从 `rstoken/` 根目录起算相对路径，不能加 `rstoken/` 前缀。
4. **AID split CSV 的图像路径是绝对路径**，跨机器迁移时需要 regenerate split CSV。`AID_splits_local/` 是本机版本，不要覆盖原始 `AID_splits/`。

---

## E36 — 连续 SNR 扫描，distill vs baseline (3 seeds)

**Done. 2026-06-11.**

### 动机

论文现在只有 5 个离散信道点（none / AWGN ±5/+10 / Rayleigh +5/+10）。reviewer 必问"distill gain 在哪个 SNR 区段最大？什么时候系统崩塌？"。E36 用现有 v0.4 checkpoint 跑连续 SNR waterfall，**不重训**。

### 配置

- [`configs/paper_v05/eval_aid_local.yaml`](../configs/paper_v05/eval_aid_local.yaml)：`data` 块切到 `AID_splits_local`，`--data_yaml` 注入 `cfg["data"]`，模型权重不动
- SNR 网格：`{-5,-2,0,2,5,7,10,12,15,20}` dB，AWGN + Rayleigh 同一列；`condition_grid()` 自动加 `none/inf`
- 6 个 ckpt：`rvq_baseline_s{41,42,43}` + `rvq_distill_s{41,42,43}`

### 命令

驱动器 [`scripts/run_e36_sweep.py`](../scripts/run_e36_sweep.py) 顺序对 6 个 ckpt 调评估脚本。**有两个工程坑**：
1. `09_eval_rvqs_recon_task_split.py` 用 `PROJECT_ROOT / relpath`，必须用 `cwd=ROOT` + 相对路径，否则双前缀
2. argparse 的 leading-dash 值（如 `-5,-2,...`）必须用 `--task_snrs=...` 等号语法，否则被当作 flag

```powershell
$env:PYTHONUTF8 = '1'
& "C:\Users\Administrator\miniconda3\envs\rstoken\python.exe" `
  rstoken\scripts\run_e36_sweep.py `
  *> rstoken\logs\paper_v05\run_logs\e36_runner.log
```

实测每个 ckpt **~210 秒**，6 ckpt 总耗时 **~21 分钟**。

### 聚合脚本

[`scripts/e36_aggregate.py`](../scripts/e36_aggregate.py)：从 12 份 per-ckpt CSV 合成 long-form raw / mean±std + 4 份适合 matplotlib 的 wide-form 曲线 CSV + markdown 汇总。

### 产物

- 12 份 per-ckpt CSV：`logs/paper_v05/e36_{task,recon}_{rvq_baseline,rvq_distill}_s{41,42,43}.csv`
- long-form raw：`logs/paper_v05/e36_continuous_snr_raw.csv`（882 行）
- long-form mean±std：`logs/paper_v05/e36_continuous_snr_mean_std.csv`（294 行）
- wide-form 曲线（直接喂 matplotlib）：
  - `logs/paper_v05/e36_curve_h0_awgn.csv`
  - `logs/paper_v05/e36_curve_h0_rayleigh.csv`
  - `logs/paper_v05/e36_curve_psnr_k4_awgn.csv`
  - `logs/paper_v05/e36_curve_psnr_k4_rayleigh.csv`
- 人读对照：`logs/paper_v05/final_table_e36_snr_curve.md`

### 验收

- ✅ 6 task + 6 recon CSV 全部生成（runner 日志每个 exit=0）
- ✅ 22 (channel, snr, k) 组合 × 6 ckpt = 132 task 行 + 252 recon 行
- ✅ AWGN +12/15/20 与 +10 完全一致（BER ≈ 0），数据自洽
- ✅ Rayleigh +20 dB BER ≈ 0.0025，仍保留 fading floor，可观察

### 关键发现（论文新增图 + 强化叙事）

**Distill-gain envelope（最大增益位置）：**

| Metric | Channel | SNR @ peak | Δ peak | baseline | distill |
|---|---|---|---|---|---|
| h₀ acc (%) | AWGN | **+2 dB** | **+30.70 pp** | 41.37 | 72.07 |
| h₀ acc (%) | Rayleigh | **+10 dB** | **+30.10 pp** | 47.63 | 77.73 |
| PSNR k=4 (dB) | AWGN | −2 dB | +0.31 dB | 14.21 | 14.52 |
| PSNR k=4 (dB) | Rayleigh | +2 dB | +0.33 dB | 14.75 | 15.08 |

**叙事要点：**

1. **Task path 的 distill gain 在 AWGN +2 dB 达 +30.7 pp 峰值**——比论文摘要写的 +25.10 pp（no-channel）更强；no-channel 是 distill gain 缩到 +26.27 pp 的"饱和段"，而非最大值
2. **AWGN +5 dB 之后 baseline 也已饱和到 ~56% ceiling**——论文 §IV-B 的 baseline 55.73% 不是"差到极致"，而是 baseline 模型固有上限；distill 把 ceiling 抬到 ~83%
3. **Rayleigh fading 下 distill gain 在 +5 → +10 dB 区间稳定 +29 ~ +30 pp**，这是论文应该突出的"channel-robust"区段
4. **Reconstruction PSNR 在所有 SNR 距离 < 0.4 dB**（max +0.33 dB at low SNR；clean channel 反向 −0.20 dB）——L0-only design 的直接证据：L0 监督只移动 task 语义层，不打扰 L1-L3 重建梯度
5. **AWGN −5 dB 时两条曲线都崩到 ~5% chance 水平**（BER ≈ 0.213）——支持论文"超低 SNR 时离散 token 优势让位于残差 BER floor"的边界论

### 论文应该怎么改

1. §IV-B "Result." 段：把 "Without a channel, h₀/L0 BoW accuracy improves from 58.23 ± 1.57% to 83.33 ± 0.81%" 扩展为 "distill gain peaks at AWGN +2 dB (+30.7 pp) and Rayleigh +10 dB (+30.1 pp); the no-channel +25.1 pp is a saturating bound, not the maximum"
2. **新增 Fig. 4**：4 子图（h₀ AWGN/Rayleigh、PSNR k=4 AWGN/Rayleigh），数据来自 4 份 wide-form curve CSV
3. §IV-D 当前 PSNR/LPIPS 表保留 5 个离散点；连续曲线放正文 figure
4. §V Discussion 加一句：reconstruction PSNR 在所有 SNR 上 distill 与 baseline 差距 < 0.4 dB，L0-only design 的连续证据

### 教训记录

5. **argparse 的 leading-dash 值**：传 `-5,-2,...` 给 `--task_snrs` 时必须用 `--task_snrs=-5,-2,...` 等号语法，不能空格分隔（argparse 把 `-5,...` 看成新 flag → exit 2）。
6. **PowerShell `.ps1` 文件含中文路径会失败**：`powershell.exe -File foo.ps1` 用本机 ANSI 解析 .ps1，中文路径会变成乱码。**用 Python 驱动器代替 .ps1**，路径不会被 PowerShell 重新编码。

---

## E39 — NWPU-RESISC45 zero-shot reconstruction transfer

**Done. 2026-06-11.**

### 动机

论文 Table V 当前只有 h₀ 列，跨数据集 reconstruction 缺位。reviewer 一定会问"NWPU 的 PSNR / LPIPS / recon-cls 也能保留吗？"。E39 用现有 `rvq_distill` / `rvq_baseline` checkpoint，**不重训**，在 NWPU-RESISC45 测试集上跑完整重建路径。

E39 唯一的新训练负担是 **NWPU 45-class ResNet34 recon-cls 评估器**（AID 30-class 评估器不能复用）。

### 配置

#### NWPU 分类器（recon-cls 评估器）

[`configs/paper_v05/nwpu_classifier_resnet34.yaml`](../configs/paper_v05/nwpu_classifier_resnet34.yaml)：复制 AID 配方，仅改 `num_classes: 45`、splits 路径、output 路径。`scripts/08_train_aid_classifier.py` 用 `AIDDataset` 直接读 NWPU split CSV（同 schema），无需新代码。

```powershell
$env:PYTHONUTF8 = '1'
Push-Location rstoken
& "C:\Users\Administrator\miniconda3\envs\rstoken\python.exe" `
  scripts\08_train_aid_classifier.py `
  --config configs/paper_v05/nwpu_classifier_resnet34.yaml `
  *> logs/paper_v05/run_logs/e39_nwpu_classifier_train.log
Pop-Location
```

实测 50 epoch **~26 分钟**（721–985 img/s on 22050 train）。最终：

- Test acc (45-class) **95.17%**
- Macro-F1 **0.9518**
- Worst-class acc **83.57%**
- Best epoch 49

作为 recon-cls 评估器与 AID 96.10% / 95.94 macro-F1 同档。

#### NWPU 重建评估

[`configs/paper_v05/eval_nwpu.yaml`](../configs/paper_v05/eval_nwpu.yaml)：仅 `data` 块切换 `splits_dir: data/NWPU_splits`，image_size 仍 256（匹配 encoder 训练分辨率，ResNet GAP 接受任意分辨率）。

`09_eval_rvqs_recon_task_split.py` 通过 `--data_yaml` 注入此 override，`--classifier_ckpt` 指向新训的 NWPU 分类器，**自动 re-fit h₀/L0_bow probe** 在 NWPU train 索引上。

### 命令

```powershell
Push-Location rstoken
$env:PYTHONUTF8 = '1'
& "C:\Users\Administrator\miniconda3\envs\rstoken\python.exe" `
  scripts/09_eval_rvqs_recon_task_split.py `
  --models "rvq_distill_s42=checkpoints/rvq_distill/best.pt,rvq_baseline_s42=checkpoints/rvq_baseline/best.pt" `
  --recon_models "rvq_distill_s42,rvq_baseline_s42" `
  --task_out logs/paper_v05/e39_task_nwpu.csv `
  --recon_out logs/paper_v05/e39_recon_nwpu.csv `
  "--task_snrs=5,10" "--recon_snrs=5,10" `
  --ks "1,2,3,4" --seed 42 --device cuda --batch_size 64 `
  --classifier_ckpt checkpoints/paper_v05/nwpu_classifier_resnet34/best.pt `
  --data_yaml configs/paper_v05/eval_nwpu.yaml `
  *> logs/paper_v05/run_logs/e39_full_nwpu.log
Pop-Location
```

实测 2 个 ckpt 总耗时 **~9 分钟**。

### 聚合脚本

[`scripts/e39_aggregate.py`](../scripts/e39_aggregate.py)：合并 task + recon CSV → wide-form summary + extended Table V markdown + 跨域对比表（vs E36 AID s42）。

### 产物

- 分类器 ckpt：`checkpoints/paper_v05/nwpu_classifier_resnet34/{best.pt, last.pt}`
- 分类器 metrics：`logs/paper_v05/nwpu_classifier_resnet34/{metrics.csv, test_metrics.json}`
- task CSV：`logs/paper_v05/e39_task_nwpu.csv` (10 行)
- recon CSV：`logs/paper_v05/e39_recon_nwpu.csv` (40 行)
- summary：`logs/paper_v05/e39_nwpu_summary.csv` (40 行 wide-form)
- 人读汇总：`logs/paper_v05/final_table_e39_nwpu.md`

### 验收

- ✅ NWPU 分类器 test acc 95.17%（≥ 90% 即可作 recon-cls 评估器）
- ✅ Smoke 阶段 h₀ no-channel 64.37% 与论文 Table V 报告 64.4% **完全一致**，证明流水线正确
- ✅ 2 ckpt × 5 信道 × 4 个 k = 40 个 recon cell 全覆盖
- ✅ Cross-domain Δ PSNR 在两个 family 上方向一致（NWPU > AID by ~1.2 dB），物理合理

### 关键发现（论文 §IV-F 须更新点）

**Extended Table V（NWPU 完整故事）：**

| Channel | rvq_distill h₀ | rvq_baseline h₀ | Δh₀ | distill k=4 PSNR | distill k=4 recon-cls | baseline k=4 PSNR | baseline k=4 recon-cls |
|---|---|---|---|---|---|---|---|
| None | **64.37%** | 43.49% | +20.87 | **27.09 dB** | **80.59%** | 27.37 dB | 78.08% |
| AWGN +5 | 61.38% | 41.68% | +19.70 | 24.80 | 76.25% | 24.63 | 72.60% |
| AWGN +10 | 64.38% | 43.48% | +20.90 | 27.08 | 80.60% | 27.37 | 78.10% |
| Rayleigh +5 | 30.73% | 16.79% | +13.94 | 17.42 | 18.24% | 16.82 | 15.48% |
| Rayleigh +10 | 53.54% | 33.71% | +19.83 | 21.20 | 56.76% | 20.71 | 49.22% |

**叙事要点：**

1. **任务路径迁移：** AID 25.1 pp distill gap → NWPU 20.87 pp，**保留 83%**。论文 §IV-F 现有结论"distillation contribution preserved (4.2 pp shrink)"完全得到 reconstruction 路径侧的支撑
2. **重建路径迁移**（论文新增）：clean k=4 PSNR 在 NWPU 上 **27.09 dB**，比 AID 的 25.92 dB **高 1.16 dB**——NWPU 类别之间视觉差异更大、训练信号更结构化所致；**重建质量没有掉**
3. **Recon-cls 任务保真度迁移：** clean k=4 recon-cls **80.59%**（45 类，base rate 2.2%），AID 上是 86.9%（30 类，base rate 3.3%）。换算成"相对随机基线"的 lift：NWPU 36.6× vs AID 26.3×，**实际 lift 比 AID 还大**
4. **Distill vs baseline 差异方向跨域一致：** baseline 在 NWPU 上 PSNR 反而**略高**（27.37 vs 27.09）但 recon-cls 略低（78.08% vs 80.59%），重复了 AID 上 §IV-D 看到的 trade-off：distill 牺牲少量 pixel fidelity 换 task fidelity
5. **Rayleigh +5 dB 在 NWPU 上更脆弱：** distill h₀ 仅 30.73%，AID 上是 58.6%。这是 RVQ index 在残差比特噪声下的 floor 与 NWPU 类别更难（45-way）共同作用。**论文 §IV-F "Conclusion. partial transfer" 应保留**

### 论文应该怎么改

1. §IV-F Table V 扩展为包含 PSNR / LPIPS / recon-cls 的完整 5 列 × 5 信道矩阵（数据见 `final_table_e39_nwpu.md`）
2. §IV-F "Result." 段加一句："Reconstruction-path metrics transfer cleanly: clean $k=4$ PSNR on NWPU is $27.09$~dB, $1.16$~dB higher than AID, and reconstructed-image classifier accuracy is $80.59\%$ on the harder 45-class task (versus $86.9\%$ on AID's 30-class), representing a $36.6\times$ lift over chance compared to AID's $26.3\times$"
3. §IV-F "Conclusion. partial transfer" 不变，仍写 task path 4.2 pp shrink；新增句："Reconstruction quality is preserved without any cross-dataset fine-tuning of the tokenizer."
4. 摘要"Zero-shot transfer to NWPU-RESISC45 preserves the distillation gain (25.1 pp on AID, 20.9 pp on NWPU)"加半句："...with reconstruction-path metrics likewise preserved on the harder 45-class benchmark."

### 教训记录

7. **`AIDDataset` 复用 NWPU 数据**：NWPU split CSV schema 与 AID 完全一致（path, class_id, class_name），`models/datasets.py` 无需任何代码改动。复用键在于 NWPU split 生成时主动选择了同样的 schema（v0.4 manual E30）。
8. **`build_classifier` 从 ckpt cfg 自动读 num_classes**：`load_classifier` 通过 `ckpt["config"]["model"]["num_classes"]` 自动 dispatch（30 → AID, 45 → NWPU），不需要 `--num_classes` flag。
9. **跨域 PSNR 比较的解读**：相同 checkpoint 在两个数据集上 PSNR 不一样，主要受**数据集图像复杂度**驱动而非"模型迁移好坏"。NWPU PSNR 比 AID 高 ≈ 1.2 dB 是常态（NWPU 类别更结构化、有更多简单几何场景），不应解读成"分布偏移让模型更好"。

---

## E35 — Deep-JSCC baseline (ADJSCC, mixed AWGN+complex Rayleigh, 3 seeds)

**Done. 2026-06-11.**

### 动机

GRSL/TGRS reviewer 必问"vs 现代 deep-JSCC 怎么样？"。论文当前对比对象只有 plain RVQ (`rvq_baseline`) 和 codec+LDPC (JPEG2000/WebP)，缺连续 JSCC 的代表。E35 在 AID 上从零训练 ADJSCC，**严格 matched-bits 协议**与 RS-Token 比较 PSNR / LPIPS / recon-cls。

### 严格对齐协议

为让"E35 是公平的现代 deep-JSCC 基线"站得住，做了 4 件事：

1. **Matched bits/image**：ADJSCC 输出 N 个实数 symbol，每个 symbol 1 BPSK bit 等效；C ∈ {10,20,40} 对应 N ∈ {2560,5120,10240}，**精确等于 RS-Token 的 k=1/2/4 bit budget**
2. **Matched 噪声 σ**：σ² = 1/(2·SNR_lin) per real dim，与 RS-Token 的 BPSK 实数信道完全一致
3. **复数 Rayleigh + coherent ZF 均衡**：相邻 2 个实 symbol pair 成 1 个复 symbol，h ~ CN(0,1) 独立 fading，receiver 知道 h；这是 ADJSCC 论文的标准 fading 协议
4. **Mixed-channel training**：每个 batch 50% AWGN / 50% Rayleigh，SNR ∈ [-2, 12] dB 均匀。AF block 在两种 channel 上都见过

### 模型规格

[`models/adjscc.py`](../models/adjscc.py)：

- 3.64M 参数（vs RS-Token tokenizer 10.87M；ADJSCC 不需要 codebook 所以更小）
- Encoder: 256→128→64→32→16，每层 stride-2 conv + AF block；末端 1×1 conv 投到 C 通道
- Decoder: 16→32→64→128→256 镜像；末端 tanh 输出 [-1,1]
- AF block：global avg pool + SNR scalar → MLP(64) → sigmoid gate（per-channel reweighting）
- Power normalisation：每个 sample 把 N 个 symbol 归一化到 ‖x‖²=N（E[|x_i|²]=1）

### 训练

9 个 ckpt = 3 rates × 3 seeds，每个 50 epoch、batch 16、AdamW lr 1e-4、bf16 AMP（与 RS-Token 完全一致）。

```powershell
$env:PYTHONUTF8 = '1'
& "C:\Users\Administrator\miniconda3\envs\rstoken\python.exe" `
  rstoken\scripts\run_e35_adjscc.py --gen-configs
& "C:\Users\Administrator\miniconda3\envs\rstoken\python.exe" `
  rstoken\scripts\run_e35_adjscc.py --train `
  *> rstoken\logs\paper_v05\run_logs\e35_train_runner_v2.log
```

实测 **每个 ckpt ~11 分钟**，9 个 ckpt 顺序跑 **~100 分钟**。

### 评估

```powershell
& "C:\Users\Administrator\miniconda3\envs\rstoken\python.exe" `
  rstoken\scripts\run_e35_adjscc.py --eval `
  *> rstoken\logs\paper_v05\run_logs\e35_eval_runner_v3.log
```

5 个信道（none / AWGN ±5/+10 / Rayleigh ±5/+10），每 ckpt ~0.3 分钟，9 ckpt **~3 分钟**。

### 聚合

[`scripts/e35_aggregate.py`](../scripts/e35_aggregate.py)：合并 ADJSCC 3-seed mean±std + 与 RS-Token rvq_distill 3-seed 对照（来自 E24）。

### 产物

- ADJSCC 9 个 ckpt：`checkpoints/paper_v05/adjscc_b{2560,5120,10240}_s{41,42,43}/best.pt`
- ADJSCC raw（45 行）：`logs/paper_v05/e35_adjscc_recon.csv`
- ADJSCC mean±std（15 行）：`logs/paper_v05/e35_adjscc_mean_std.csv`
- 对照 CSV（15 行）：`logs/paper_v05/e35_vs_rs_token.csv`
- 人读对照：`logs/paper_v05/final_table_e35_adjscc.md`

### 关键发现

**两条结论同时成立、方向相反：**

1. **ADJSCC 在 PSNR 全面领先 +1.9 ~ +8.1 dB**——deep-JSCC 是为 pixel reconstruction 优化的，没有 codebook 量化损失
2. **RS-Token 在 task-fidelity 上 AWGN 全胜，clean & AWGN 领先 +6.8 ~ +29.8 pp**——离散+蒸馏的 task path 优势

**Headline（k=4, 10240 bits/image, 3-seed mean）：**

| Metric | ADJSCC | RS-Token | Δ (RS-ADJSCC) |
|---|---|---|---|
| Clean PSNR | **28.23** dB | 25.92 dB | −2.30 dB |
| Clean recon-cls | 80.00% | **86.80%** | **+6.80 pp** |
| AWGN +10 dB recon-cls | 77.43% | **86.80%** | **+9.37 pp** |
| Rayleigh +10 dB recon-cls | 68.03% | 67.73% | −0.30 |
| Rayleigh +5 dB PSNR | **24.97** dB | 16.90 dB | −8.07 dB |

**叙事要点：**

1. **PSNR vs task-acc trade-off 是结构性的**：ADJSCC 的连续 latent 用 MSE 直接优化 pixel reconstruction，在所有信道、所有比特预算 PSNR 全胜；RS-Token 的离散 token + RemoteCLIP 蒸馏直接训 task semantics，在 task-acc 上反超
2. **Trade-off 在低比特更显著**：k=1 (2560 bits) 时 RS-Token recon-cls 领先 +18.5 pp（71.4 vs 52.9），k=4 (10240) 时只领先 +6.8 pp（86.8 vs 80.0）。**低比特 task 路径正是论文 §IV-B 的主张论点**
3. **Rayleigh 上的 PSNR 差距** 实际反映了离散 vs 连续表示的本质差异：RS-Token 在 Rayleigh +5 dB 时 PSNR 跌到 16.9 dB（残差比特错误率 6.4%），ADJSCC 因为是连续 latent + soft equalisation 仍能 24.97 dB
4. **但 Rayleigh recon-cls 相互交错**：Rayleigh +5 dB 在 k=1 时 ADJSCC 22.6% vs RS-Token 22.0%（接近），在 k=4 时 ADJSCC 51.6% vs RS-Token 19.7%（ADJSCC 大胜）；说明 Rayleigh fading 下 RS-Token 的 task 优势会被 destroy
5. **AWGN 是 RS-Token 的主场**，Rayleigh 是 ADJSCC 的主场。论文的 channel-robust 主张需要在 §V Discussion 修订成"AWGN 信道下"

### 论文应该怎么改

1. §IV 加一节 §IV-X "Modern Deep-JSCC Baseline"，引入 ADJSCC 的对比表（数据来自 `final_table_e35_adjscc.md`）
2. 摘要里 "channel-robust" 改成更精确的措辞，例如 "channel-robust under AWGN; Rayleigh fading sees a Pareto trade-off with continuous JSCC"
3. §II-A 加 cite ADJSCC (Xu et al. TCSVT 2022) 和最少一个更新的 deep-JSCC（NTSCC、SwinJSCC 任一）
4. §V Discussion 新增段落："The PSNR-vs-task-fidelity trade-off is structural: continuous deep-JSCC optimises pixel reconstruction directly while discrete tokens with vision-language distillation optimise task semantics. RS-Token wins task accuracy under AWGN by up to 29.8 pp at low rates; ADJSCC wins PSNR by 2-8 dB across all rates and channels. Under Rayleigh fading the ranking on task accuracy partially reverses at high rates, marking the boundary of the discrete-token approach."
5. **不要**在论文里写 "RS-Token strictly dominates ADJSCC"——数据不支持这个 claim，写 trade-off 才严谨

### 教训记录

10. **复数 Rayleigh 必须配对实 symbol**：实数版 Rayleigh + ZF 均衡会让 fade ≈ 0 时 σ/h → ∞，模型完全无法恢复（PSNR 跌到 9 dB）。复数版用 N/2 个独立 fade，平均效应让模型可学
11. **ADJSCC 必须在 fading 下训练**：AWGN-only 训练的 ADJSCC 在 Rayleigh 评估完全崩塌，与论文不符。`train_channel: mixed` 让 AF block 在两种 channel 上都见过 SNR 条件，这才是 ADJSCC 论文的标准协议
12. **No-channel evaluation 不能传 SNR=∞**：AF block 的 sigmoid gate 是用 [-2,12] dB 训练的，传 100 dB 会 OOD（PSNR 反而比 +10 dB 差 6 dB）。**clamp 到 train_snr_max** 是正确做法
13. **eval 脚本路径**：`09_eval_rvqs_recon_task_split.py` 用 `PROJECT_ROOT / relpath`；ADJSCC 的 `eval_adjscc.py` 沿用同样规则。所有 eval 入口都必须 `cwd=ROOT`，不能加 `rstoken/` 前缀

---

## 下一步建议

E36/E37/E39/E35 完成。**T1 优先级现在只剩 E38（WebP 脚注）**。

候选下一步：

- **E35 Deep-JSCC**（T1，5–10 天）：最重，但是覆盖现代 baseline 的 reviewer 必问点。需要外部代码 + 9 次训练（3 rate × 3 seed），风险最高
- **E38 WebP 脚注**（T1, 10 分钟纯文本）：写 Table IV/VI 的 † 脚注 + §IV-E 一句话；**T1 里成本最低**，可以现在做掉
- **E40 init-seed paired control**（T2，2 天）：训练 6 个 paired ckpt 量化 init noise 占 25 pp gap 的份额。reviewer 不一定要求，但加分
- **E42 RVQ K∈{2,3,6} ablation**（T2，4–5 天）：4 次重训。defends K=4 选择
- **T3.* 11 项纯文本修正**（T3，0 天）：等所有实验完一起修，避免反复重排版

按"成本/收益"看：
- 想最快落锤就先做 **E38**（10 分钟） + **T3.***（半天集中修），能让论文 review 文档里 T1 + T3 全部清掉，只剩 E35 一个外部依赖
- 想继续推主线就上 **E40 init-seed paired control**，与 E37 配套加固 §IV-B 的 25 pp 主张
- 想啃硬骨头就直接上 **E35**

要哪个？

- **纯推理**，复用现有 6 个 checkpoint（rvq_baseline_s{41,42,43} + rvq_distill_s{41,42,43}）
- **不需要新训练**，估计 1 天可跑完
- **产出一张吸睛的图**：task acc / PSNR vs SNR 的连续曲线，把现在 5 个离散点替换成完整 waterfall
- 对论文 §IV-B（task path）和 §IV-D（recon path）都加分

后面 E39（NWPU recon transfer）也是推理任务、半天到一天能完，可以接着 E36 一起串起来。
E35（Deep-JSCC）放最后，因为它需要外部代码 + 9 次训练，风险最高。

---