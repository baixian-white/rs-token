# 同行评审报告 · RS-Token 短文 v0.3

> **审稿人**：自审，模拟 IEEE GRSL / TGRS / IGARSS 审稿视角  
> **稿件**：RemoteCLIP-Distilled Hierarchical Tokenizer for Channel-Robust Remote Sensing Communication  
> **依据**：`short_paper_v0.3.md` 与 `latex/rs_token.tex`  
> **日期**：2026-06-01  
> **建议**：**Major Revision**（较 v0.2 明显改进，但仍有 P0 blocker）

---

## 1. 总体评价

v0.3 相比 v0.2 是一次实质性修订。方法部分从“结果驱动叙述”变成了较完整的论文骨架：encoder/decoder 结构、bit budget、L0 语义增益公式、`h_0` 标准化、信道仿真协议、codebook 利用率、HARQ/ACM 关系和 limitation 都补上了。尤其是 Table 4 / Figure 4 的 codebook 几何分析，使 RemoteCLIP 相比 OpenAI CLIP 的优势不再只是一个经验现象，而有了初步机制支撑。

但从正式投稿角度看，v0.3 仍不能越过“大修”门槛。最大问题仍然是：核心强结论主要来自内部消融，而不是与外部传输或语义通信 baseline 的同台比较；其次，所有主结果仍是 single-seed，缺少模型种子和信道种子的统计报告；第三，`k` 与 `h_0` 评估口径仍可能被审稿人抓住，认为“传多层”和“任务分类特征”之间的因果链没有讲清。

**总体判断**：

- 对 IGARSS 短文：补一个外部 baseline + 多种子主表后，有较现实投稿价值。
- 对 GRSL：至少需要外部 baseline、多模型种子、评估口径拆清。
- 对 TGRS/JSTARS：当前仍偏早，需要 2-3 条外部 baseline、跨 seed 统计和更完整的重建任务评估。

---

## 2. v0.3 已解决或显著改善的问题

### R1. 方法可复现性明显增强

v0.3 在 §3.1 补充了 encoder / decoder 结构：VQ-VAE-2 / VQ-GAN 风格卷积主干、`base_channels=64`、通道倍率 `[1,2,4,4]`、每 stage 一个 residual block、16× 下采样、约 10.87M 参数、GroupNorm + SiLU、不使用 self-attention。这基本解决了 v0.2 中“主干是黑箱”的问题。

### R2. bit budget 已显式化

正文写出了：

$$T \cdot k \cdot \log_2 K$$

并明确 `L0 = 2560 bits/img`、`4 layers = 10240 bits/img`，脚注说明 `k ∈ {1,2,3,4}` 的 2-bit 元数据开销可忽略。这个修订到位。

### R3. “L0 覆盖 95% 语义增益”的定义已清楚

v0.3 明确区分了：

- `86.0 / 88.0 = 97.7%`：L0 占 4 层最终准确率的比例。
- `(86.0 - 47.7) / (88.0 - 47.7) = 95.0%`：L0 占蒸馏引入语义增益的比例。

这个公式化处理可以抵御审稿人的重新计算。

### R4. Rayleigh 0 dB 断点被纳入 limitation

v0.3 不再把 Rayleigh 0 dB 包装成 graceful degradation 成功案例，而是明确称为 breakdown point，并把 graceful degradation 主张限定到 `SNR >= +5 dB` 的 deployment 工作区间。这是必要的收缩。

### R5. RemoteCLIP 表述更克制，并加入几何证据

相比 v0.2 的“不可替代”风险，v0.3 将 RemoteCLIP 的贡献表述为中等 SNR 下额外 `4-7 pp` 稳健性收益，并加入了 inter/intra distance、silhouette、Calinski-Harabasz、Davies-Bouldin 与 t-SNE。这个方向正确。

需要注意：`short_paper_v0.3.md` 顶部仍写着“M6 几何分析未执行”，但正文已经包含 Table 4 / Figure 4。这里是作者侧记录不一致，投稿前应清掉。

### R6. HARQ/ACM 讨论位置更合理

v0.3 将长篇物理层反驳移至 §5.2，Introduction 只保留“正交并可叠加”的 framing。这个改动降低了通信方向审稿人的逆反风险。

### R7. 若干复现细节已经补齐

包括 random seed = 42、`StandardScaler(with_mean=False)` 口径、BPSK closed-form BER 与 waveform-level Monte Carlo 交叉验证、4 层 codebook 利用率 >99%、BPSK 作为 worst-case link condition 的解释。这些都能提升可信度。

---

## 3. 仍然阻碍投稿的主要问题

### M1. 缺少外部 baseline [Critical]

v0.3 自己在 limitation 中承认尚未与 JPEG2000+LDPC、DeepJSCC、Gao et al. 2022 等外部方法进行 same-SNR sweep 比较。这仍是最大硬伤。

当前最强结果“L0-only 比 no-distill 4-layer 高 25-32 pp”本质上是内部消融：它证明 RemoteCLIP 蒸馏有效，但还不能证明 RS-Token 比已有图像传输、任务感知传输或 semantic communication 方法更好。

**审稿人会追问**：

- 同样 `2560 / 10240 bits/img` 下，JPEG2000/BPG/WebP + channel corruption 表现如何？
- Gao et al. 2022 在同 AID + AWGN/Rayleigh 协议下是否更强？
- DeepJSCC 是否在低 SNR 下更平滑？
- 如果传统压缩流 bit flip 后无法解码，decode failure rate 是多少？

**最低修复标准**：

- IGARSS / GRSL 前至少补 1 条外部 baseline。
- 最建议先做 JPEG2000/BPG/WebP 同 bit budget + AID classifier，因为实现成本最低，也直接回应 Introduction 中的 cliff-effect claim。
- 如果能复现 Gao et al. 2022，则它是协议同源的最强对照。

### M2. 缺少模型种子与统计显著性 [Critical]

v0.3 所有主表仍写着 single-seed run。单一 `seed=42` 可以作为复现实验入口，但不能支撑论文主结论。

需要区分两类随机性：

- **model seed**：训练初始化、数据顺序、优化轨迹。
- **channel seed**：bit flip / Monte Carlo 信道实现。

目前真正缺的是 model seed。`+24.7 pp` 的主增益大概率稳定，但 RemoteCLIP vs OpenAI CLIP 的 `4-7 pp` 差距、低 SNR 下 `k=1` vs `k=4` 的小差距，都需要 mean ± std。

**最低修复标准**：

- `RVQ x4 no-distill`：3 个 model seeds。
- `RVQ x4 + RemoteCLIP`：3 个 model seeds。
- 每个 checkpoint 跑固定 no-channel probe、AWGN sweep、Rayleigh sweep。
- 主表报告 `mean ± std`。

**推荐统计口径**：

先在同一 model seed 内对 channel seeds 求均值，再跨 model seeds 报告 `mean ± std`。不要把 `3 model seeds × 5 channel seeds` 当成 15 个独立模型样本。

### M3. `k` 与 `h_0` 的评估口径仍有概念风险 [Critical / Major]

v0.3 的 Fig. 1 caption 已说明 task path 在 deployment 中是 `k=1`，但 Figure 2 / LaTeX caption 仍写“每个 SNR × k 单元格都基于 `h_0` 特征得到分类准确率”。这里会让审稿人困惑：

- `h_0` 只由 L0 indices 构造。
- 如果 task classifier 始终只看 L0，那么额外传 L1-L3 不应直接提供任务信息。
- 那么 Figure 2 的 `k` 轴究竟是在评估传输层数，还是在检验 L0 在不同总传输设置下的稳定性？

**建议拆成两条评估路径**：

- **Task path**：只报告 `k=1` 的 `h_0` 分类准确率，这是“最少索引保任务”的核心。
- **Reconstruction path**：报告 `k=1..4` 的 PSNR / LPIPS / reconstructed-image classifier accuracy，这是“链路变好时叠加细节层”的证据。

如果仍保留 SNR × k 的 `h_0` 图，需要明确说明它不是“L1-L3 提升任务准确率”的证据，而只是展示不同传输层数下 L0 任务分支的可用性或包络。

### M4. Rayleigh 0 dB 仍缺少外部语境 [Major]

v0.3 已经把 Rayleigh 0 dB 作为 breakdown point，这是正确的。但没有外部 baseline 时，读者仍无法判断 16.6% 是：

- 所有方法都难以工作的物理层极限；
- 还是 RS-Token 对 Rayleigh fading 特别脆弱。

**建议**：在 Rayleigh 0 dB 加入至少一个外部 baseline，哪怕只是 JPEG2000/BPG/WebP 同 bit budget 的 decode failure rate + classifier accuracy。

### M5. 信道模型仍偏理想化 [Major]

当前信道实验是 closed-form BER independent bit flip，等价于 uncoded BPSK hard-decision 的统计平均。它适合作为第一版，但不能支撑过强的真实 UAV / SmallSat deployment claim。

**建议正文措辞**：

- 将“真实部署”类表达改成 “deployment-motivated simulation” 或 “deployment-relevant operating regime”。
- 明确当前未包含 packetization、FEC、interleaving、OFDM、多径信道估计误差等现实链路因素。
- 若篇幅允许，加入一个 lightweight FEC on index bits 的小实验会大幅增强通信可信度。

### M6. 参考文献仍有占位 [Major]

`rs_token.bib` 中仍有多个 `[Author list TODO]` 和 `TODO`：

- `moc-rvq`
- `revqom`
- `gao2022task`
- `vilau`
- `semclip`

这是正式投稿前必须清掉的硬伤。尤其 `MOC-RVQ / ReVQom / SemCLIP / Gao 2022` 都参与 novelty framing 或 baseline framing，不能以未核验条目支撑关键论证。

### M7. claim 仍略满，需要等实验补齐后再恢复 [Major-Minor]

v0.3 已经明显克制，但摘要和 Introduction 中仍有一些容易被认为“提前兑现”的表述：

- “cliff-effect collapse” 批评 JPEG2000+LDPC，但当前没有外部 baseline 曲线。
- “25-32 pp higher” 是相对 no-distill internal baseline，不是相对已有方法。
- “deployment-relevant” 需要绑定到当前仿真范围，而不是暗示真实链路验证。

**建议**：在补 baseline 前，所有强比较都写成 “against the no-distillation RVQ baseline”。补完外部 baseline 后，再恢复更强的 framing。

### M8. LaTeX 元信息和版式仍需清理 [Minor / Major-Minor]

LaTeX 仍有一些投稿前会被一眼看到的问题：

- 作者单位仍是 `School of XXX`。
- 单作者稿件的 `markboth` 仍写 `et al.`。
- 当前 PDF log 显示输出为 7 pages，而 README 目标是 5-6 pages。
- LaTeX log 仍有多处 table overfull hbox，尤其 lines 130-137 与 157-166。

这些不是科学硬伤，但会影响投稿成熟度。

---

## 4. 建议实验补强顺序

### Step 1. 先修评估口径

在继续跑实验前，先把 task path / reconstruction path 拆清。否则新实验容易继续混在旧叙事里。

**建议新增两个主表**：

- Task table：`h_0` / L0-only 分类，主打语义兜底。
- Reconstruction table：`k=1..4` 的 PSNR、LPIPS、重建图分类准确率，主打链路改善后的细节恢复。

### Step 2. 跑核心模型种子

优先级最高的是：

- `rvq_baseline`：3 个 model seeds。
- `rvq_distill_remoteclip`：3 个 model seeds。

每个模型跑：

- no-channel `h_0` probe。
- AWGN `0/+5/+10 dB`。
- Rayleigh `0/+5/+10 dB`，其中 Rayleigh 0 dB 可作为 breakdown point。

### Step 3. 固定一个 clean AID classifier

为了做外部压缩基线和重建图分类，需要一个 clean-image classifier：

- 在 AID train 上训练。
- 在 clean test 上报告上限。
- 对 RS-Token reconstruction 和 external baseline reconstruction 统一评估。

### Step 4. 做经典压缩外部 baseline

最低可行 baseline：

- JPEG2000 或 BPG/WebP。
- bit budget 对齐 `2560 / 5120 / 10240 bits/img`。
- 对压缩流施加同样信道 corruption。
- 报告 decode failure rate、PSNR、LPIPS、classifier accuracy。

如果压缩 bitstream 在 bit flip 后无法解码，不要丢弃失败样本；应把失败率作为结果的一部分。

### Step 5. 可选补 Gao / DeepJSCC

若目标是 GRSL，建议至少尝试 Gao et al. 2022 protocol-matched baseline。若目标是 TGRS/JSTARS，再补 DeepJSCC 或 EO+JSCC 变体。

---

## 5. 投稿建议

### IGARSS 2026

**可行，但需要最小 P0 包**：

- 至少 1 条外部 baseline。
- 主方法与 no-distill baseline 的 3 model seeds。
- task / reconstruction 评估口径拆清。
- 清理 bib TODO 和 LaTeX 占位。

### IEEE GRSL

**有潜力，但当前还不稳**。GRSL 可以接受短文体例和有限实验范围，但不会轻易接受无外部 baseline + single-seed 的主张。建议完成外部 baseline 与 mean ± std 后再投。

### IEEE TGRS / JSTARS

**当前不建议直接投**。需要更完整的 baseline matrix、跨数据集或多传感器扩展、FEC / coded transmission 讨论，以及更强的统计报告。

---

## 6. v0.3 大修检查清单

### P0：投稿前必须解决

- [ ] 加入至少 1 条外部 baseline，同 bit budget、同 SNR sweep、同 AID task metric。
- [ ] `RVQ x4 no-distill` 跑 3 个 model seeds。
- [ ] `RVQ x4 + RemoteCLIP` 跑 3 个 model seeds。
- [ ] 主表报告 `mean ± std`，并说明 model seed 与 channel seed 的统计方式。
- [ ] 拆清 task path (`h_0`, `k=1`) 与 reconstruction path (`k=1..4`, PSNR/LPIPS/recon classifier)。
- [ ] 修正 Figure 2 / caption 中容易暗示 “k 层数直接提升 h_0 任务准确率” 的表述。

### P1：强烈建议解决

- [ ] Rayleigh 0 dB 加入至少一个 baseline 或更明确降级为 breakdown-only 结果。
- [ ] 将 JPEG2000/BPG/WebP 的 decode failure rate 纳入结果，而不是只报告成功解码样本。
- [ ] 对 RemoteCLIP vs OpenAI CLIP 教师消融至少补 channel seeds；资源允许则补 model seeds。
- [ ] 将 “cliff effect” 等外部方法批评绑定到实际 baseline 数字或改为更克制动机。
- [ ] 将当前信道设置明确称为 uncoded BPSK hard-decision / deployment-motivated simulation。
- [ ] 清理 `short_paper_v0.3.md` 顶部“M6 几何分析未执行”与正文 Table 4 的不一致。

### P2：投稿呈现清理

- [ ] 补齐 `School of XXX`。
- [ ] 单作者 `markboth` 去掉 `et al.`。
- [ ] 完成 `rs_token.bib` 中所有 TODO 文献核验。
- [ ] 将 PDF 从 7 页压到目标 5-6 页，或明确改为 journal/arXiv 版本。
- [ ] 处理主要 overfull hbox，尤其宽表。
- [ ] 检查标题：Markdown v0.3 标题与 LaTeX 标题当前不一致，投稿前统一。

---

## 7. 编辑部式决定信

本文提出的问题重要，方法有清晰创新点：把遥感专用视觉语言模型蒸馏到 RVQ 第一层，使离散索引形成任务保真层级。这一想法与遥感低 SNR 下行链路场景匹配，v0.3 的方法细节和机制分析也已经显著加强。

然而，当前证据仍不足以支撑正式接收。外部 baseline 缺失使核心贡献无法与已有图像传输 / 语义通信方案比较；single-seed 结果不足以支撑统计稳定性；`k` 与 `h_0` 的评估路径仍需澄清。建议作者完成上述 P0 项后重新送审。

**编辑建议**：Major Revision。若下一版补齐外部 baseline、模型种子统计，并清理评估口径，本稿可进入“短文可投”的状态。
