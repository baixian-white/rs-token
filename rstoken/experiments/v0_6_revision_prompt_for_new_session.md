# 📄 Claude Code 新会话提词：完成 RS-Token 论文 v0.6 修订

将下面整段内容作为新会话的 **第一条用户消息** 粘贴。Claude Code 会从这条消息开始，按计划自主完成所有剩余工作。

---

## 提词正文（从这里开始复制）

你是一位严格执行实验的工程 AI。现在要在已有 RS-Token 论文 **v0.5** 的基础上，整合 **6 个新实验（E35, E36, E37, E39, E40, E41, E42）的所有数据**，产出一份完整、自洽、可投稿的 **v0.6** 论文。**完全自主、无需向用户请示**——遇到歧义按"对论文最严谨"的选择走，记到日志里。

# 项目背景

- 项目位置：`d:\CODE\遥感+通信\遥感+通信`（Windows 11，PowerShell）
- conda env：`C:\Users\Administrator\miniconda3\envs\rstoken\python.exe`（**直接用绝对 Python 路径，不要用 `conda run`**——后者在中文路径下会触发 GBK UnicodeEncodeError）
- v0.5 主稿：`paper_draft/latex/rs_token_v0.5.tex`（391 行，IEEEtran 模板）
- v0.5 中文版：`paper_draft/latex/rs_token_v0.5_zh.md`
- review 文档：`paper_draft/latex/rs_token_review_v0.5.md`（实验清单 + 修改 checklist 来源）
- bib 文件：`paper_draft/latex/rs_token_v05.bib`
- 实验日志（**先读这份！**）：`rstoken/experiments/rs_token_v0_5_experiment_log.md`（62 KB，包含 E35–E42 全部产物清单、命令、关键发现、26 条教训）

# 核心任务：把 v0.5 升级为 v0.6

## 任务一：所有实验数据已就绪

E35–E42 全部完成训练 + 评估 + 聚合，产物落盘：

| 实验 | 关键产物路径 | 用途 |
|---|---|---|
| E35 ADJSCC | `logs/paper_v05/final_table_e35_adjscc.md` + `e35_adjscc_mean_std.csv` | 论文新增 §IV-X "Modern Deep-JSCC Baseline" 节 |
| E36 SNR sweep | `logs/paper_v05/final_table_e36_snr_curve.md` + `e36_curve_*.csv` (4 份 wide-form) | 论文 §IV-B/§IV-D 新增连续曲线 figure |
| E37 All-layers 3-seed | `logs/paper_v05/final_table_placement_3seed.md` + `e37_placement_3seed_mean_std.csv` | 论文 §IV-C Table II "All layers" 升级为 3-seed mean±std |
| E39 NWPU transfer | `logs/paper_v05/final_table_e39_nwpu.md` + `e39_nwpu_summary.csv` | 论文 §IV-F Table V 扩展为含 PSNR/LPIPS/recon-cls 完整矩阵 |
| **E40 paired init control** | `logs/paper_v05/final_table_e40_paired.md` + `e40_paired_h0_3seed.csv` | 论文 §IV-B 加 paired Δ 段落，替换"shared seed"那句不严谨表述 |
| **E41 DCRC** | `logs/paper_v05/final_table_e41_direct_clip.md` + `e41_dcrc_3seed.csv` | 论文 §V Discussion 加 "Why not directly quantize RemoteCLIP?" 段，新增 Appendix Table |
| **E42 K-ablation** | `logs/paper_v05/final_table_e42_k_ablation.md` + `e42_k_ablation_summary.csv` | 论文 §IV 加 "RVQ depth ablation" 节，§III-A 改 K=4 justification |

**这些 markdown 文件的"论文应该怎么改"段已经写好了完整的 LaTeX 草稿和"建议加这一句"格式的成稿文字**——直接抄进 v0.6 即可。

## 任务二：3 项纯文本修改（T3 + E38）

review 文档 `paper_draft/latex/rs_token_review_v0.5.md` 第 §2 列出了：

- **E38 WebP rate-floor 脚注**（10 分钟）：给 v0.6 的 Table IV "WebP/k=1, k=2" 行加 `\textsuperscript{$\dagger$}` + 脚注；§IV-E "Result." 段加一句话。具体位置和措辞 review 文档第 "#### E38 — WebP rate-floor footnote (text-only)" 节有完整文字
- **T3.1–T3.11**（11 项）：见 review 文档 "### Tier-3 (text-only, no experiment)" 表，包括：typography 修复（半角冒号、Abstract 措辞）、Eq. 注释、§II-A 加 deep-JSCC 引用（NTSCC/ADJSCC/SwinJSCC 任一）、§V Discussion 三个新 bullet、§V 加 limitation 段、bib 清理未引用条目

## 任务三：必须严格遵循的工程纪律

1. **新文件命名**：所有新文件用 `rs_token_v0.6` 前缀，不要覆盖 v0.5 文件。具体要产出的：
   - `paper_draft/latex/rs_token_v0.6.tex`（基于 v0.5.tex 修改）
   - `paper_draft/latex/rs_token_v0.6_zh.md`（基于 v0.5_zh.md 同步修改的中文版，**必须保持与英文版完全对应**）
   - `paper_draft/latex/rs_token_v06.bib`（基于 rs_token_v05.bib 增补 ADJSCC + 至少 1 个新 deep-JSCC 引用）
   - `paper_draft/latex/rs_token_v0.6.pdf`（编译产物）

2. **不修改 v0.5 文件**：v0.5 是已锁定的版本，所有修改都在 v0.6 副本上进行。先 `Copy-Item` 然后 `Edit`

3. **CSV 读取必须用 `encoding="utf-8-sig"`**——PowerShell 写的 CSV 带 BOM（教训 14 in 实验日志）。所有要从 logs/ 读数字的脚本严格遵守

4. **绝对不要用 `--smoke` 跑 production config**——会覆盖 best.pt/last.pt（教训 15）。但本任务**不需要再跑任何训练**——所有实验都完成了。如果需要重新生成图，写独立 figure 脚本读 csv 即可

5. **路径全用 D 盘 `D:/CODE/遥感+通信/遥感+通信/`**，不要写 H 盘（教训 4）

6. **PDF 编译**：用 `pdflatex` + `bibtex` 三遍。编译命令模板：
   ```powershell
   Push-Location paper_draft\latex
   try {
     pdflatex -interaction=nonstopmode rs_token_v0.6.tex
     bibtex rs_token_v0.6
     pdflatex -interaction=nonstopmode rs_token_v0.6.tex
     pdflatex -interaction=nonstopmode rs_token_v0.6.tex
   } finally { Pop-Location }
   ```
   遇到错误必须 fix-then-recompile，不能留 ?? 引用或 missing references

# 详细执行步骤

## Step 0 — 环境与上下文确认（必做）

1. **完整阅读** `rstoken/experiments/rs_token_v0_5_experiment_log.md`——这是 26 条工程教训 + 7 个实验的完整索引，每一步都要避免重蹈覆辙
2. **完整阅读** `paper_draft/latex/rs_token_review_v0.5.md` 的 §2 实验清单（找到 E38 + T3 的完整文字）
3. **完整阅读** v0.5 主稿 `paper_draft/latex/rs_token_v0.5.tex` 与 v0.5_zh.md，标记每个 §section 的当前内容
4. **完整阅读以下 7 个 final_table markdown**（这些是数据真相来源，每份都包含"论文应该怎么改"段）：
   - `rstoken/logs/paper_v05/final_table_e35_adjscc.md`
   - `rstoken/logs/paper_v05/final_table_e36_snr_curve.md`
   - `rstoken/logs/paper_v05/final_table_placement_3seed.md`
   - `rstoken/logs/paper_v05/final_table_e39_nwpu.md`
   - `rstoken/logs/paper_v05/final_table_e40_paired.md`
   - `rstoken/logs/paper_v05/final_table_e41_direct_clip.md`
   - `rstoken/logs/paper_v05/final_table_e42_k_ablation.md`
5. 用 `TodoWrite` 建一份至少 12 步的清单，对应下面 Step 1–12

## Step 1 — 复制 v0.5 → v0.6 骨架

```powershell
Copy-Item paper_draft\latex\rs_token_v0.5.tex paper_draft\latex\rs_token_v0.6.tex
Copy-Item paper_draft\latex\rs_token_v0.5_zh.md paper_draft\latex\rs_token_v0.6_zh.md
Copy-Item paper_draft\latex\rs_token_v05.bib paper_draft\latex\rs_token_v06.bib
```

修改 `rs_token_v0.6.tex` 的 `\bibliography{...}` 改成 `\bibliography{rs_token_v06}`；header 注释的版本号改成 v0.6 + 当前日期；title 不变。

## Step 2 — bib 增补

向 `rs_token_v06.bib` 至少添加：
- ADJSCC 主引用（Xu, Wang, Gao, Zhao, "Wireless Image Transmission Using Deep Source Channel Coding With Attention Modules", IEEE TCSVT 2022）
- 1 个其他现代 deep-JSCC（NTSCC 或 SwinJSCC，任一即可，T3.7 要求）

bib 关键格式必须是 IEEEtran 兼容的 `@article{key, ...}`，key 用 lowercase 短名（例如 `xu2022adjscc`、`yang2024swinjscc`）。

清理 v0.4 → v0.5 中可能未引用的条目（T3.11）：先 `bibtex` 编译一次拿到 warning list，再删除 `references.bib` 里 unused 的项。

## Step 3 — §IV-B 修订（E40 paired init）

在 §IV-B "Result." 段：

1. 主表数字**不动**（83.33 ± 0.81% 等照旧 —— 来自 unpaired Table I）
2. 替换现有"`rvq_baseline` 与 `rvq_distill` 共享 init seed = 42，因此该 gap 不可归因于 weight-init 噪声"那句不严谨表述（sharing seed ≠ sharing weights，因为 teacher / distill_head 改变 RNG 消耗顺序）
3. 加新一句（中英文版都要加）：
   > Under bit-identical step-0 initialisation, the paired distillation gain is +25.77 ± 2.94 pp under no channel, +26.97 ± 3.46 pp at AWGN +5 dB, +25.73 ± 2.99 pp at AWGN +10 dB, +27.47 ± 3.31 pp at Rayleigh +5 dB, and +28.83 ± 3.35 pp at Rayleigh +10 dB. All five paired gains lie within 2σ of the unpaired Table I numbers, ruling out random initialisation as the source of the 25.1 pp distillation gap (see Appendix \ref{app:e40} or experiment log §E40).

数据来源：`logs/paper_v05/e40_paired_h0_3seed.csv`

## Step 4 — §IV-C Table II 升级（E37 3-seed）

- Table II 的 "All layers" 列从单 seed 升级为 3-seed mean±std
- 数据来源：`logs/paper_v05/e37_placement_3seed_mean_std.csv` 和 `final_table_placement_3seed.md`
- 5 个信道 × 3 metrics（h₀, k=4 PSNR, k=4 recon-cls）+ layered probe k=1..4
- 核心更新：Rayleigh +5 dB 的 deficit 实测 −10.83 pp，比论文原文写的 −9.3 pp 大；改 §IV-C "Result." 段的描述

## Step 5 — §IV 新增 "RVQ depth ablation" 节（E42）

用 `final_table_e42_k_ablation.md` 中已经写好的 LaTeX 草稿（"建议加一节 §IV-X 'RVQ depth ablation'"），完整复制到 v0.6.tex，编号统一为 §IV-G（或新章节按上下文调整）。**必须包含的核心叙事**：

1. K=4 不是 strict optimum
2. K=3 在 task 反超 +1.7 pp（task ≥ 82% AND PSNR ≥ 25.9 dB 的最小深度）
3. K=6 PSNR 反超 +0.66 dB 但多花 +50% 比特，bits/dB 效率 K=4 > K=6
4. K=4 是 Pareto compromise + 部署 bit grid 对齐理由

**§III-A 同步修改**：把"K=4 (chosen empirically)"改成 "smallest depth satisfying both task ≥ 82% and PSNR ≥ 25.9 dB at the 10,240-bit deployment grid"

数据：`logs/paper_v05/e42_k_ablation_summary.csv`

## Step 6 — §IV 新增 "Modern Deep-JSCC Baseline (ADJSCC)" 节（E35）

用 `final_table_e35_adjscc.md` 的"论文应该怎么改"段。新节插在 §IV-D 之后或 §IV-E 之前（按顺序）。**必须包含**：

1. ADJSCC 9 ckpt × 3 rates × 3 seeds 训练协议描述（mixed AWGN+Rayleigh，matched bits）
2. headline 表（k=4，10240 bits/image, 3-seed mean）：PSNR / clean recon-cls / AWGN+10 recon-cls / Rayleigh+5 PSNR / Rayleigh+10 recon-cls
3. **Pareto trade-off 叙事**：ADJSCC PSNR 全胜 +1.9~+8.1 dB，RS-Token task fidelity AWGN 全胜 +6.8~+29.8 pp。**不能写"RS-Token strictly dominates"**——数据不支持
4. §V Discussion 同步加一段（trade-off 是结构性的）

摘要里的 "channel-robust" 改成精确措辞 `"channel-robust under AWGN; Rayleigh fading sees a Pareto trade-off with continuous JSCC"`

数据：`logs/paper_v05/e35_vs_rs_token.csv` 和 `e35_adjscc_mean_std.csv`

## Step 7 — §IV-D 加连续 SNR figure（E36）

`logs/paper_v05/` 下有 4 份 wide-form curve CSV：
- `e36_curve_h0_awgn.csv`
- `e36_curve_h0_rayleigh.csv`
- `e36_curve_psnr_k4_awgn.csv`
- `e36_curve_psnr_k4_rayleigh.csv`

写一个 `paper_draft/build_fig4_snr_sweep.py` 脚本，读这 4 份 CSV，用 matplotlib 画 4 子图（h₀ AWGN/Rayleigh + PSNR k=4 AWGN/Rayleigh），baseline vs distill 双曲线 + 3-seed std 阴影。输出 `rstoken/figs/fig_v06_snr_sweep.pdf` + `.png`。

§IV-D "Result." 段加一句：连续曲线展示 distill gain 在 AWGN +2 dB 达 +30.7 pp 峰值、Rayleigh +10 dB 达 +30.1 pp，no-channel 25.1 pp 是饱和段非最大值。

`\includegraphics{...}` 引用新 figure，caption 用 `final_table_e36_snr_curve.md` 中的描述。

## Step 8 — §IV-F Table V 扩展（E39）

NWPU recon path 加入 Table V，扩展为完整 5 列 × 5 信道矩阵：rvq_distill h₀ / rvq_baseline h₀ / Δh₀ / distill k=4 PSNR / distill k=4 recon-cls。

§IV-F "Result." 段加一句：
> Reconstruction-path metrics transfer cleanly: clean k=4 PSNR on NWPU is 27.09 dB, 1.16 dB higher than AID, and reconstructed-image classifier accuracy is 80.59% on the harder 45-class task (versus 86.9% on AID's 30-class), representing a 36.6× lift over chance compared to AID's 26.3×.

§IV-F "Conclusion. partial transfer" 不变；新增句："Reconstruction quality is preserved without any cross-dataset fine-tuning of the tokenizer."

数据：`logs/paper_v05/e39_nwpu_summary.csv`

## Step 9 — §V Discussion 加 DCRC 段（E41）

把 `final_table_e41_direct_clip.md` 中已写好的 §V LaTeX 段（"Why not directly quantize RemoteCLIP?"，约 150 词）完整复制到 v0.6.tex 的 §V Discussion。这段必须写：

1. DCRC 的 trade-off 对比：DCRC h₀ +13~+24 pp，但 PSNR −8 dB、recon-cls −75 pp
2. CLIP feature 不可逆是结构原因
3. RS-Token 的 hierarchical 设计**不是 dominant**，是 Pareto front 的一端

**§III-A 同步修改**：加一句话 justify CNN encoder 选择（避免 reviewer 攻击）

新增 Appendix Table E.1 完整 DCRC vs RS-Token 数字（3 子表：task / recon@k=4 / recon-sweep）。

数据：`logs/paper_v05/e41_dcrc_3seed.csv`

## Step 10 — E38 + T3 全部纯文本修改

按 review 文档 §2 的 E38 和 Tier-3 表，逐项 fix：

- **E38**：Table IV 加 † 脚注 + §IV-E 一句话（review 文档有完整文字）
- **T3.1**：Abstract 半角"："改成 ASCII `:`
- **T3.2**：Abstract "receiver transmits" → "transmitter sends"
- **T3.3**：Fig.1 caption + §III-C 统一 φ vs g_ψ
- **T3.4**：Eq. (8) loss 展开或引用 §IV-A weights
- **T3.5**：Eq. (10) 加 "perfect CSI at receiver" 注解
- **T3.6**：§III-A 形式化定义 C_ℓ
- **T3.7**：§II-A 与 bib 加 deep-JSCC variant 引用（Step 2 已加 bib，这里加正文引用）
- **T3.8**：§V Discussion 加 "vision-language is necessary, RemoteCLIP increment is bounded" bullet
- **T3.9**：§V 加 "5G NR LDPC out of scope" bullet
- **T3.10**：§V 加 limitation "scene classification only; object detection / segmentation needs L0 grid > 16×16"
- **T3.11**：清理 bib 未引用条目（用 bibtex log 的 warning 定位）

每一项 fix 后，**英文版 v0.6.tex 与中文版 v0.6_zh.md 必须同步修改**。中文版翻译保持 v0.5_zh.md 的语调和术语（"L0 任务路径"、"无信道"、"任务保真"、"通信预算"等术语沿用）。

## Step 11 — 编译 v0.6.pdf

```powershell
Push-Location paper_draft\latex
try {
  pdflatex -interaction=nonstopmode rs_token_v0.6.tex *> compile_v06_pass1.log
  bibtex rs_token_v0.6 *> compile_v06_bibtex.log
  pdflatex -interaction=nonstopmode rs_token_v0.6.tex *> compile_v06_pass2.log
  pdflatex -interaction=nonstopmode rs_token_v0.6.tex *> compile_v06_pass3.log
} finally { Pop-Location }
```

**必须**确认：
- 没有 `?` 占位（说明引用未解析）
- 没有 LaTeX Error 或 Undefined control sequence
- 所有 `\ref{}` 都解析
- 所有 `\cite{}` 都在 bib 中
- 编译产物：`paper_draft/latex/rs_token_v0.6.pdf`

如果编译失败，**必须**修到通过——不允许留 broken state。

## Step 12 — 写一份 v0.6 修订对照清单

新建 `paper_draft/latex/v05_to_v06_changelog.md`，结构化列出：

1. **Abstract changes**（前后 diff）
2. **每个 §section 的 paragraph-level diff**
3. **Tables changes**（哪个 table 是新增、哪个是更新、哪个不动）
4. **Figures changes**（新增 figs + 不动 figs）
5. **Bib changes**（新增 + 删除条目）
6. **新增 Appendix sections**（E.1 DCRC table 等）
7. **关键叙事 deltas**：
   - Abstract: "channel-robust" → "channel-robust under AWGN; Pareto trade-off with continuous JSCC under Rayleigh"
   - §IV-B: 加 paired Δ 数据 + 删"shared seed"那句不严谨表述
   - §IV-C: All-layers 升级 3-seed
   - §IV: 新增 "RVQ depth ablation" 节 + "Modern Deep-JSCC Baseline" 节
   - §IV-D: 加连续曲线 figure
   - §IV-F: Table V 扩展为完整 PSNR/LPIPS/recon-cls 矩阵
   - §V: 新增 DCRC 段 + 3 个新 bullet + limitation
   - §III-A: K=4 justification 改写 + CNN encoder 解释
   - Appendix: 新增 E.1 DCRC table

# 验收标准（全部满足才算 DONE）

- [ ] `paper_draft/latex/rs_token_v0.6.tex` 存在且与 v0.5 不同
- [ ] `paper_draft/latex/rs_token_v0.6_zh.md` 存在，**与英文版 §section 编号、数据数字、新增段落 100% 对应**
- [ ] `paper_draft/latex/rs_token_v06.bib` 存在，至少含 ADJSCC + 1 个其他 deep-JSCC 引用
- [ ] `paper_draft/latex/rs_token_v0.6.pdf` 存在且 **pdflatex 三遍编译无 LaTeX Error / 无 ? 引用**
- [ ] `paper_draft/build_fig4_snr_sweep.py` 存在，读 4 份 e36 curve CSV，输出 `rstoken/figs/fig_v06_snr_sweep.pdf` 与 `.png`
- [ ] **所有 7 个实验**（E35/E36/E37/E39/E40/E41/E42）的核心数据都进入 v0.6.tex 主稿或 appendix
- [ ] **E38 + T3.1–T3.11 全部 11+1 项纯文本修改完成**
- [ ] `paper_draft/latex/v05_to_v06_changelog.md` 存在且条目完整
- [ ] **不修改任何 v0.5 文件**（v0.5.tex / v0.5_zh.md / rs_token_v05.bib / v0.5.pdf）
- [ ] **不修改任何 rstoken/checkpoints/、rstoken/logs/、rstoken/data/**（论文修订不需要再跑实验）
- [ ] 实验日志末尾 `rstoken/experiments/rs_token_v0_5_experiment_log.md` 加一行 v0.6 提交记录（或新建 `rs_token_v0_6_paper_revision_log.md`）

# 核心数字（必须用对，不能写错）

复制下面的对照表到工作记忆（也可贴在 changelog 顶部）：

| 量 | 数值 | 来源 |
|---|---|---|
| Unpaired no-channel h₀ baseline | 58.23 ± 1.57% | Table I (paper_v0.5 已有) |
| Unpaired no-channel h₀ distill | 83.33 ± 0.81% | Table I (paper_v0.5 已有) |
| **Paired no-channel Δ** | **+25.77 ± 2.94 pp** | E40 |
| Paired AWGN +5 Δ | +26.97 ± 3.46 pp | E40 |
| Paired AWGN +10 Δ | +25.73 ± 2.99 pp | E40 |
| Paired Rayleigh +5 Δ | +27.47 ± 3.31 pp | E40 |
| Paired Rayleigh +10 Δ | +28.83 ± 3.35 pp | E40 |
| **K=2 no-ch h₀** | 84.00% | E42 |
| **K=3 no-ch h₀** | 84.30% | E42 |
| **K=4 no-ch h₀ (main)** | 82.60% | E42 |
| **K=6 no-ch h₀** | 82.10% | E42 |
| K=2 full-K PSNR | 24.71 dB | E42 |
| K=3 full-K PSNR | 25.44 dB | E42 |
| K=4 full-K PSNR | 25.92 dB | E42 |
| K=6 full-K PSNR | 26.58 dB | E42 |
| **DCRC h₀ no-channel** | **96.00 ± 0.62%** | E41 |
| **DCRC PSNR k=4 no-ch** | 17.85 ± 0.01 dB | E41 |
| **DCRC recon-cls k=4 no-ch** | 11.43 ± 1.15% | E41 |
| RS-Token PSNR k=4 no-ch | 25.92 ± 0.08 dB | Table III (paper_v0.5 已有) |
| RS-Token recon-cls k=4 no-ch | 86.80 ± 0.36% | Table III (paper_v0.5 已有) |
| All-layers Rayleigh +5 Δ vs L0 | −10.83 pp (4.9σ) | E37 |
| Distill gain peak AWGN +2 dB | +30.70 pp | E36 |
| Distill gain peak Rayleigh +10 dB | +30.10 pp | E36 |
| NWPU clean k=4 PSNR | 27.09 dB | E39 |
| NWPU clean k=4 recon-cls | 80.59% | E39 |
| ADJSCC clean k=4 PSNR | 28.23 dB | E35 |
| ADJSCC clean k=4 recon-cls | 80.00% | E35 |

# 风险与边界

1. **不要主动新增 figures 或 tables 超出 review 文档要求**——只加 review §2 列出的（E36 figure、E42 K-ablation table、E41 appendix table、E39 Table V 扩展）
2. **不要重新跑任何实验**——所有数据已就绪，只读 csv 不写 csv
3. **不要修改 rstoken/ 下面除 figs/ 之外的任何东西**
4. **figs/ 只能 add 不能 delete**——v0.5 用过的 fig_v04_*.pdf 和 fig_layered_probe.pdf 等保持不动
5. **如果 LaTeX 编译卡住超过 3 次都 fix 不通**，先做最小可编译版本（删掉新加的 figure 引用 / 复杂 table 包），再逐项加回
6. **中英文版同步**是硬要求——任何只改英文不改中文（或反之）的状态都算未完成

# 完成标志

最终向用户输出**一段汇报**，包括：

1. v0.6.pdf 总页数 + 编译时间
2. v0.5 → v0.6 主要 deltas（5–8 条）
3. 所有 7 个实验的整合位置（每个实验在哪个 §section 出现）
4. 11+1 项 E38/T3 修改的完成情况（一行一项）
5. v05_to_v06_changelog.md 的路径
6. 任何遗留风险或未解决项（如有）

# 期望耗时

- Step 0–2（读 + 复制 + bib）: 30 分钟
- Step 3–9（实验整合 + 主稿修订）: 3–4 小时
- Step 10（E38 + T3）: 30 分钟
- Step 11（编译 + 修错）: 30 分钟到 1 小时
- Step 12（changelog）: 20 分钟

**总预计 5–6 小时**。这是一次性论文修订，没有训练、没有 GPU、没有等待。开始执行吧。

---

## 提词正文（结束）
