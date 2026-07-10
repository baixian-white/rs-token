# RS-Token v0.5 → v0.6 修订对照清单（中文 MD 版）

> **本次修订只针对中文 MD 版本** `paper_draft/latex/rs_token_v0.6_zh.md`（基于 v0.5_zh.md）。LaTeX 主稿、英文版与 PDF 编译产物本次未生成。Bib 增补条目（ADJSCC、SwinJSCC）将在下次同步英文版时一并落到 `rs_token_v06.bib`。

## 1. 文件改动

| 文件 | 状态 | 说明 |
|---|---|---|
| `paper_draft/latex/rs_token_v0.6_zh.md` | **新建** | 基于 v0.5_zh.md 复制后整合 E35–E42 七项实验数据 + E38 + T3.1–T3.11 全部修正 |
| `paper_draft/latex/v05_to_v06_changelog.md` | **新建** | 本对照清单 |
| `paper_draft/latex/rs_token_v0.5*` 全部 | **未修改** | v0.5 已锁定版本 |
| `paper_draft/latex/rs_token_v0.6.tex` | 暂未生成 | 用户要求只做中文 MD 版本 |
| `paper_draft/latex/rs_token_v0.6.pdf` | 暂未生成 | 用户要求只做中文 MD 版本 |
| `paper_draft/latex/rs_token_v06.bib` | 暂未生成 | 用户要求只做中文 MD 版本 |
| `paper_draft/build_fig4_snr_sweep.py` | 暂未生成 | 用户要求只做中文 MD 版本 |
| `rstoken/figs/fig_v06_snr_sweep.{pdf,png}` | 暂未生成 | 用户要求只做中文 MD 版本 |

## 2. 关键叙事 deltas（核心声明的精确化）

| 原 (v0.5) | 修订后 (v0.6) | 来源实验 |
|---|---|---|
| "channel-robust" | "channel-robust under AWGN; Rayleigh fading sees a Pareto trade-off with continuous JSCC" | E35 (摘要) |
| §4.2 "rvq_baseline 与 rvq_distill 共享 init seed = 42，因此该 gap 不可归因于 weight-init 噪声" | 替换为 paired-init 控制实验：5 信道 paired Δ 全部在 unpaired 25.10 pp 的 2σ 内 | E40 (§4.2 末段 + 附录 E.2) |
| §4.3 All-layers Rayleigh +5 deficit "−9.3 pp" (1 seed) | 升级为 "$-10.83\pm2.16$ pp" (3 seed, 4.9σ 显著) | E37 (§4.3 表 2) |
| §3.1 "K=4 (chosen empirically)" | 改写为 "smallest depth satisfying $h_0 \geq 82\%$ AND PSNR $\geq 25.9$ dB at 10,240-bit budget; aligns with IoT downlink bit grid" | E42 (§3.1 + §4.8) |
| §4.4 离散 5 SNR 点 | 新增连续 SNR 扫描叙事：AWGN +2 dB 峰值 +30.7 pp，Rayleigh +10 dB 峰值 +30.1 pp，no-channel 是饱和段 | E36 (§4.4 末段 + 图 4) |
| §4.6 Table 5 仅 $h_0$ 列 | 扩展为 5 列 × 5 信道完整矩阵（PSNR / recon-cls / Δ）；NWPU clean k=4 PSNR 27.09 dB（比 AID +1.16 dB） | E39 (§4.6 表 5) |

## 3. 摘要前后 diff

**v0.5 摘要**关键句：

> "RS-Token attains 0% decode-failure ... 信道鲁棒"

**v0.6 摘要**关键句变化：
- 加入 paired-init 控制实验数据：paired Δ = +25.77 ± 2.94 pp（无信道），与 25.10 pp 在 2σ 内一致 → 排除"权重初始化噪声驱动 25 pp"的解读
- 加入 NWPU 重建路径迁移：clean k=4 PSNR 27.09 dB / recon-cls 80.59% on 45 类
- 加入 ADJSCC trade-off：AWGN 下 RS-Token task 全胜（最高 +29.8 pp），Rayleigh 下 ADJSCC PSNR 全胜（+1.9~+8.1 dB）；trade-off 是结构性的
- "channel-robust" → "channel-robust under AWGN; Rayleigh fading sees a Pareto trade-off with continuous JSCC"
- 把"接收端选择前 k 层作为前缀传输" → "发射端选择前 k 层作为前缀进行传输"（T3.2）

## 4. 每个 §section 的 paragraph-level diff

### §1 引言
- 贡献项加入 v0.6 新增 5 条新实验：（i）ADJSCC Pareto 对照、（ii）连续 SNR 曲线、（iii）paired-init 控制、（iv）DCRC 控制、（v）K-ablation。

### §2.1 任务导向遥感通信
- 加入 ADJSCC \cite{xu2022adjscc}、SwinJSCC \cite{yang2024swinjscc} 两项现代 deep-JSCC 引用 (T3.7)
- 末句加入 §4.4a 引导："本文 §4.4a 在严格匹配比特预算下与 ADJSCC 直接对照"

### §3.1 问题设定
- 加入 $C_\ell$ 形式化定义（T3.6）："每层 codebook 大小 $C_\ell = 1024$（$\ell=1,\ldots,K$）"
- K=4 justification 改写为 Pareto compromise + IoT bit grid 对齐（E42）
- 加入 CNN encoder 选择的 justification（E41 / 反 reviewer 攻击）

### §3.3 蒸馏目标
- Fig.1 caption 把 φ → $g_\psi$ 统一（T3.3）
- Eq. (8) 后加补："$\mathcal{L}_{\rm rec}$ 本身是 $\ell_1$ 与 LPIPS 损失的加权和，具体权重见 §4.1"（T3.4）

### §3.4 信道模型
- Eq. (10) 前加注："**假设接收端拥有完美的信道状态信息（CSI）**"（T3.5）

### §4.2 RemoteCLIP 蒸馏是否使 L0 更具语义？
- 删除 v0.5 不严谨语句"`rvq_baseline` 与 `rvq_distill` **共享 init seed = 42**，因此该 gap 不可归因于 weight-init 噪声"（sharing seed ≠ sharing weights，因 teacher / distill_head 改变 RNG 消耗）
- 加 paired-init 控制实验段：5 信道 paired Δ 全部数据 + 2σ 比较（E40）
- 结论补一句"E40 paired-init 控制实验进一步将该 gap 归因于 $\lambda_{\rm distill}$ 而非初始化噪声"

### §4.3 蒸馏位置反事实
- 表 2 "All layers" 列从 1 seed 升级为 3 seed mean ± std，5 信道 × 3 metric × 4 probe 全覆盖（E37）
- "Result." 段重写：所有信道 deficit > 2σ 统计显著；Rayleigh +5 dB deficit 实测 −10.83 ± 2.16 pp（4.9σ），比 v0.5 的 −9.3 pp 更大

### §4.4 渐进式重建
- 加 Fig. 4 描述（连续 SNR 扫描）
- "Result." 段加连续 SNR 曲线小结（E36）：AWGN +2 dB 峰值 +30.70 pp、Rayleigh +10 dB 峰值 +30.10 pp、PSNR 差距全 SNR 段 < 0.4 dB

### §4.4a Modern Deep-JSCC Baseline (ADJSCC) — **新增节**（E35）
- 训练协议：9 ckpt（3 rates × 3 seeds），mixed AWGN+复数 Rayleigh + coherent ZF 均衡，bit budget 与 RS-Token 严格匹配
- 表 4a：headline 5×3 cell at $k=4$ / 10240 bits
- 双向 trade-off 叙事：ADJSCC PSNR 全胜 +1.9~+8.1 dB，RS-Token AWGN task 全胜 +6.8~+29.8 pp
- 不写 "RS-Token strictly dominates"，写 "structural Pareto trade-off"

### §4.5 严格同传输比特 codec+LDPC
- WebP $k=1, 2$ 行加 $^\dagger$ 脚注 + 表后说明：14,164–14,301 src-bit，超 target 2.8×~5.5×，未严格满足 matched-bits 前提（E38）
- 结果段补一句关于 WebP rate-floor

### §4.6 NWPU zero-shot 迁移
- 表 5 从 1 列 ($h_0$) 扩展为 5 列 × 5 信道矩阵：含 distill / baseline 的 $h_0$, $k=4$ PSNR, $k=4$ recon-cls + Δ（E39）
- 结果段加重建路径迁移叙事：clean PSNR 27.09 dB（比 AID +1.16 dB），recon-cls 80.59% on 45 类（lift 36.6× vs AID 26.3×）
- 结论加："重建路径在跨数据集 zero-shot 迁移下质量得以保留 ... 未对 tokenizer 做任何跨域 fine-tune"

### §4.8 RVQ depth ablation — **新增节**（E42）
- 表 9：headline 4 K × 5 信道 + Δ vs K=4 + 全 K 重建 sweep
- 三大叙事：K=3 task 反超 +1.7 pp、K=6 PSNR 反超 +0.66 dB、K=4 是 Pareto compromise
- 与 §3.1 的 K=4 justification 闭环

### §5 讨论
- **新增 ADJSCC trade-off 段**："与现代 deep-JSCC 的结构性 Pareto trade-off"（E35）
- **新增 DCRC 段**："为什么不直接量化 RemoteCLIP？"（E41）：DCRC h₀ +13~+24 pp 但 PSNR −8 dB / recon-cls −75 pp；CLIP feature 不可逆是结构原因；RS-Token 在 task / recon Pareto 前沿的折中位置
- **局限**段扩展为 4 个 bullet：（T3.8）vision-language 蒸馏是承载机制，RemoteCLIP 增益 bounded；（T3.9）5G NR LDPC out of scope；（T3.10）当前评估限于场景分类，object detection / segmentation 需要 L0 grid > 16×16；（v0.6 新增）现代 deep-JSCC 比较限于 ADJSCC

### §6 结论
- 加入 v0.6 七项实验整合的总结句
- "channel-robust" → "channel-robust（AWGN）；在 Rayleigh 衰落下与连续 deep-JSCC 存在 Pareto trade-off"

### 参考文献列表
- 新增 `xu2022adjscc`（IEEE TCSVT 2022, ADJSCC）
- 新增 `yang2024swinjscc`（SwinJSCC, T3.7 第二个 deep-JSCC 引用）
- 共 17 → 19 条 entry

## 5. 表格变化

| 表 | v0.5 状态 | v0.6 变化 |
|---|---|---|
| Table 1 (Task seed stability) | 不动 | 不动（unpaired 数据仍是 headline） |
| Table 2 (Placement counterfactual) | 1 seed | All-layers 列升级为 3 seed mean ± std；L1-only 保留 1 seed（崩塌量级 −34~−48 pp 远超种子方差） |
| Table 3 (Reconstruction sweep) | 不动 | 不动 |
| **Table 4a (新增)** | — | ADJSCC vs RS-Token at $k=4$ headline，5 信道 |
| Table 4 (codec+LDPC strict) | 不动 | 加 WebP $^\dagger$ 脚注 |
| Table 5 (NWPU transfer) | 1 列 ($h_0$) | 扩展为 5 列 × 5 信道矩阵 |
| Table 6 (unprotected codec) | 不动 | 不动 |
| Table 7 (Teacher ablation) | 不动 | 不动 |
| Table 8 (λ sweep) | 不动 | 不动 |
| **Table 9 (新增)** | — | RVQ K-ablation + Δ vs K=4 |
| **附录 D**（新增） | — | ADJSCC vs RS-Token 完整对照（PSNR / recon-cls / LPIPS × 3 rates × 5 信道） |
| **附录 E.1**（新增） | — | DCRC vs RS-Token 3 子表（task / recon@k=4 / recon-sweep） |
| **附录 E.2**（新增） | — | E40 paired-init 逐 seed 表 + 3-seed mean ± std 与 unpaired 比较 |

## 6. 图变化

| 图 | v0.5 状态 | v0.6 变化 |
|---|---|---|
| Fig. 1 method overall | 不动 | caption 把 φ → $g_\psi$ |
| Fig. 2 L0 task robustness | 不动 | 不动 |
| Fig. 3 progressive reconstruction | 不动 | 不动 |
| **Fig. 4 SNR sweep**（新增） | — | 4 子图 (h₀ AWGN/Rayleigh + PSNR k=4 AWGN/Rayleigh)，3-seed std 阴影；图本身待 LaTeX 同步时由 `paper_draft/build_fig4_snr_sweep.py` 生成 |

## 7. Bib 变化

| 操作 | 条目 | 节 |
|---|---|---|
| 新增 | `xu2022adjscc`：ADJSCC, IEEE TCSVT 2022 | §2.1 / §4.4a / §5 / 附录 D / 摘要 |
| 新增 | `yang2024swinjscc`：SwinJSCC | §2.1 / §5 |
| 删除 | (无未引用条目；v0.5 已经过 bib hygiene) | — |

## 8. 新增 Appendix sections

- **附录 D**：ADJSCC vs RS-Token 完整对照表（PSNR / recon-cls / LPIPS）
- **附录 E.1**：DCRC vs RS-Token 完整数字（3 子表：task / recon@k=4 / recon-sweep）
- **附录 E.2**：E40 paired-init 逐 seed 数据 + 3-seed 汇总

## 9. 11+1 项 E38/T3 修改完成情况

| 项 | 描述 | 状态 |
|---|---|---|
| E38 | WebP rate-floor 脚注 + §4.5 一句话 | ✅ 完成 |
| T3.1 | Abstract 半角"："改成 ASCII | ⚠️ 中文 MD 中"："是中文标点正确用法，保留；title 与各章节标题中的 "："均为中文全角，符合中文排版规范。该项主要适用于 LaTeX 英文版 |
| T3.2 | Abstract "receiver transmits" → "transmitter sends" | ✅ 完成（中文摘要："发射端选择前 $k$ 层作为前缀进行传输"） |
| T3.3 | Fig.1 caption + §3.3 统一 φ vs $g_\psi$ | ✅ 完成（Fig.1 caption 加 $g_\psi$） |
| T3.4 | Eq. (8) loss 展开或引用 §4.1 weights | ✅ 完成（在 Eq. (8) 后补一句） |
| T3.5 | Eq. (10) 加 "perfect CSI at receiver" 注解 | ✅ 完成 |
| T3.6 | §3.1 形式化定义 $C_\ell$ | ✅ 完成 |
| T3.7 | §2.1 与 bib 加 deep-JSCC variant 引用 | ✅ 完成（加 ADJSCC + SwinJSCC） |
| T3.8 | §5 Discussion "vision-language is necessary, RemoteCLIP increment is bounded" bullet | ✅ 完成（局限段第 1 个 bullet） |
| T3.9 | §5 "5G NR LDPC out of scope" bullet | ✅ 完成（局限段第 2 个 bullet） |
| T3.10 | §5 limitation "scene classification only; object detection / segmentation needs L0 grid > 16×16" | ✅ 完成（局限段第 3 个 bullet） |
| T3.11 | bib 清理未引用条目 | N/A（v0.5 bib 已无未引用条目；v0.6 仅增补 ADJSCC + SwinJSCC，未删除） |

## 10. 7 个实验整合位置一览

| 实验 | 位置 | 形式 |
|---|---|---|
| **E35** ADJSCC | §4.4a 新增节、表 4a、§5 trade-off 段、摘要、附录 D | 主稿 + 附录 + 摘要 |
| **E36** SNR sweep | §4.4 末段、图 4 | 主稿 + 新图 |
| **E37** All-layers 3-seed | §4.3 表 2、Result 段 | 主稿表升级 |
| **E38** WebP rate-floor | 表 4 $^\dagger$ 脚注、§4.5 Result 末句 | 主稿 |
| **E39** NWPU recon | §4.6 表 5 扩展、Result/Conclusion 加句、摘要 | 主稿 |
| **E40** paired init | §4.2 末段、附录 E.2、摘要 | 主稿 + 附录 + 摘要 |
| **E41** DCRC | §5 新段、§3.1 CNN encoder justification、附录 E.1 | 主稿 + 附录 |
| **E42** K-ablation | §4.8 新增节、§3.1 K=4 justification 改写、表 9 | 主稿表 |

## 11. 核心数字对照表（贴在工作记忆顶部）

| 量 | 数值 | 来源 |
|---|---|---|
| Unpaired no-channel h₀ baseline | 58.23 ± 1.57% | Table 1 (paper_v0.5) |
| Unpaired no-channel h₀ distill | 83.33 ± 0.81% | Table 1 (paper_v0.5) |
| Paired no-channel Δ | +25.77 ± 2.94 pp | E40 |
| Paired AWGN +5 Δ | +26.97 ± 3.46 pp | E40 |
| Paired AWGN +10 Δ | +25.73 ± 2.99 pp | E40 |
| Paired Rayleigh +5 Δ | +27.47 ± 3.31 pp | E40 |
| Paired Rayleigh +10 Δ | +28.83 ± 3.35 pp | E40 |
| K=2 no-ch h₀ | 84.00% | E42 |
| K=3 no-ch h₀ | 84.30% | E42 |
| K=4 no-ch h₀ (main) | 82.60% | E42 |
| K=6 no-ch h₀ | 82.10% | E42 |
| K=4 full-K PSNR | 25.92 dB | E42 |
| K=6 full-K PSNR | 26.58 dB | E42 |
| DCRC h₀ no-channel | 96.00 ± 0.62% | E41 |
| DCRC PSNR k=4 no-ch | 17.85 ± 0.01 dB | E41 |
| DCRC recon-cls k=4 no-ch | 11.43 ± 1.15% | E41 |
| RS-Token PSNR k=4 no-ch | 25.92 ± 0.08 dB | Table 3 |
| RS-Token recon-cls k=4 no-ch | 86.80 ± 0.36% | Table 3 |
| All-layers Rayleigh +5 Δ vs L0 | −10.83 pp (4.9σ) | E37 |
| Distill gain peak AWGN +2 dB | +30.70 pp | E36 |
| Distill gain peak Rayleigh +10 dB | +30.10 pp | E36 |
| NWPU clean k=4 PSNR | 27.09 dB | E39 |
| NWPU clean k=4 recon-cls | 80.59% | E39 |
| ADJSCC clean k=4 PSNR | 28.23 dB | E35 |
| ADJSCC clean k=4 recon-cls | 80.00% | E35 |

## 12. 遗留事项

1. **LaTeX 英文版 / PDF / bib 增补 / SNR sweep figure 脚本**：用户本次只要求中文 MD 版本，未生成英文 LaTeX、PDF 编译产物、`rs_token_v06.bib`、以及 `paper_draft/build_fig4_snr_sweep.py`。下次同步英文主稿时需要：
   - 复制 `rs_token_v0.5.tex` → `rs_token_v0.6.tex` 并按本 MD 同步全部修改
   - 复制 `rs_token_v05.bib` → `rs_token_v06.bib` 并增补 `xu2022adjscc`、`yang2024swinjscc` 两条 entry
   - 写 `paper_draft/build_fig4_snr_sweep.py` 读 4 份 e36 wide-form curve CSV 并输出 `rstoken/figs/fig_v06_snr_sweep.{pdf,png}`
   - 三遍 `pdflatex` + `bibtex` 编译，确认无 `?` 占位、`undefined references` 等
2. **T3.1 中文标点**：T3.1 原意是 LaTeX Abstract 中误用了中文全角"："；在中文 MD 中"："是中文标点的正确用法，保留。下次同步英文版时把英文 Abstract 中的全角"："改为半角 `:`。
3. **附录 D / E.1 / E.2 的 LaTeX 表格化**：本 MD 用 markdown table，下次同步英文版时改为 IEEEtran `tabular` + `\caption{}` + `\label{app:e35_full}` / `\label{app:e41}` / `\label{app:e40}`。

---

## 13. 自评 + 修复记录（2026-06-12 第二轮）

完成首次起草后，使用 `/review` 自评对照 7 份 final_table.md 真值文件，发现并修复以下 10 处问题：

| 优先级 | 问题 | 位置 | 修复 |
|---|---|---|---|
| 🔴 Critical | §4.7 章节编号错（K-ablation 实际为 §4.8） | §1 引言贡献 (v) | "§4.7" → "§4.8" |
| 🔴 Critical | §3.1 K-ablation 引用错号 (2 处) | §3.1 第 1 段、第 3 段 | 两处 "§4.7" → "§4.8" |
| 🔴 Critical | Table 7 错引为 §4.3（Table 7 实际在 §5） | §5 局限段第 1 个 bullet | "§4.3 表 7" → "表 7（§5 内）" |
| 🟡 Important | §5 DCRC 段 RS-Token h₀ 数字与 Table 1 不一致（83.33 vs 82.63） | §5 DCRC 段 | 加澄清说明：Table 1 是 channel-seed × model-seed 双重平均给出 83.33%，附录 E.1 与 DCRC 同口径给出 82.63%；两者是评估 averaging 方式差异 |
| 🟡 Important | §1 contribution list 漏掉 E39 NWPU 重建路径迁移 | §1 末段 | 新增 (vi) NWPU 重建路径 5 列 × 5 信道矩阵扩展 |
| 🟡 Important | §4 节首"五个问题"未涵盖 §4.4a / §4.8 新增节 | §4 节首 | 在五条之后加一句说明 §4.4a、§4.8 是 v0.6 新增对照与消融子节 |
| 🟢 Minor | 双 `---` 分隔线冗余（§5 局限段末 → 附录 D 起） | line 511–513 | 删除多余分隔线 |
| 🟢 Minor | "AWGN ≥7 dB 饱和段" 不严谨（应是 ≥10 dB 完全饱和） | §4.4 SNR sweep 段 | "≥7 dB" → "≥10 dB" |
| 🟢 Minor | "AWGN −5 dB 跌至 ~5% chance 水平" 不严谨（30 类 chance 是 3.33%） | §4.4 SNR sweep 段 | 改为 "跌至接近 chance（30 类 chance = 3.33%；实测 baseline 5.03%、distill 6.67%）" |
| 🟢 Minor | "残差比特错误率 ≈6.4%" 数字未给出推导 | §4.4a Result (iii) | 改为 "约 6%（参见 §3.4 信道模型与 §4.4 表 3 Rayleigh +5 dB 行的 PSNR 平台）" |
| 🟢 Minor | Table 4a 表头未明示只取 $k=4$ | 表 4a caption | caption 改为 "**表 4a（$k=4$ / 10,240 bits/image）**" |

**核查通过**（无需修复）：
- 七份 final_table 数据 100% 一致（E40 paired Δ 五信道、E41 DCRC、E42 K-ablation、E37 −10.83 pp、E36 +30.70/+30.10 pp、E39 NWPU 27.09 dB / 80.59%、E35 ADJSCC 28.23 dB）
- Markdown 表格列对齐 100% 匹配
- 摘要叙事 deltas（"channel-robust under AWGN + Pareto trade-off"、"+29.8 pp"、"−2.30 ~ −8.07 dB"）内部自洽
