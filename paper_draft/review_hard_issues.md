# RS-Token 审稿硬伤清单

> 日期：2026-06-01  
> 稿件：`paper_draft/latex/rs_token.tex`  
> 当前判断：Major Revision。核心思路成立，但投稿前必须补齐外部基线、训练种子统计，并修正若干评估口径。

## P0 必须先解决

### H1. 外部基线不足

**严重级别**：Critical

**问题**：当前最强结论主要来自“RemoteCLIP distill vs no-distill RVQ”的内部消融。它能证明蒸馏有用，但不能证明 RS-Token 相对已有图像传输或语义通信方法有竞争力。

**审稿风险**：审稿人会问：“同样 bit budget 和 SNR 下，和 JPEG2000/BPG + channel coding、DeepJSCC、Gao 2022 或其他 task-aware RS transmission 相比如何？”

**最低修复标准**：

- 至少加入一个外部基线。
- 优先顺序：
  1. JPEG2000 或 BPG/WebP，按相同 bit budget 对齐。
  2. Gao et al. 2022 protocol-matched baseline，如果能低成本复现。
  3. DeepJSCC，作为完整版本或期刊版本增强项。

**建议先做的版本**：

- 经典压缩基线：
  - `2560 bits/img`，对齐 RS-Token `k=1`。
  - `5120 bits/img`，对齐 `k=2`。
  - `10240 bits/img`，对齐 `k=4`。
- 对解码图像跑同一个 AID classifier，报告 classification accuracy、PSNR、LPIPS。
- 若 bit-flip 后压缩流无法解码，报告 decode failure rate，不要丢弃失败样本。

**完成标志**：

- 论文中出现 “RS-Token vs external baseline” 表。
- 表中至少含 `bpp/bits per image`、channel condition、task accuracy、PSNR/LPIPS。

### H2. 缺少真正的模型训练种子统计

**严重级别**：Critical

**问题**：已有多种子主要是 channel seed 复测，不是 model seed。审稿人真正需要的是不同训练初始化下主结论是否稳定。

**审稿风险**：单个 seed 下的 `+24.7 pp`、`25-32 pp` 优势可能被质疑为偶然结果。尤其 RemoteCLIP vs OpenAI CLIP 的 `4-7 pp` 差距更需要统计支持。

**最低修复标准**：

- `RVQ x4 no-distill` 跑 3 个 model seeds。
- `RVQ x4 + RemoteCLIP` 跑 3 个 model seeds。
- 每个 checkpoint 跑一致的 no-channel probe、AWGN sweep、Rayleigh sweep。
- 主表报告 `mean ± std`。

**建议矩阵**：

| 模型 | model seeds | channel seeds | 用途 |
|---|---:|---:|---|
| RVQ x4 no-distill | 3 | 3 到 5 | 内部主基线 |
| RVQ x4 + RemoteCLIP | 3 | 3 到 5 | 主方法 |
| OpenAI CLIP distill | 1 到 3 | 3 | 教师消融，资源允许再扩 |
| lambda sweep | 1 | 3 | 机制分析，不作为主表统计 |

**统计口径**：

- 先对同一 model seed 下的 channel seeds 求均值。
- 再跨 model seeds 报告 `mean ± std`。
- 不要把 `3 model seeds × 5 channel seeds` 直接当作 15 个独立模型样本。

**完成标志**：

- Table 1 和主 channel table 不再是 single-seed。
- 文中明确写出 model seed 和 channel seed 的处理方式。

### H3. `k` 与 `L0_bow/h_0` 的评估口径不清

**严重级别**：Critical

**问题**：当前 channel table 中 `k=1..4` 都报告 `h_0` 或 `L0_bow` 分类准确率。但 `h_0` 只依赖 L0 index histogram，额外传 L1-L3 不应直接改变分类特征。若不澄清，审稿人会认为 `k` 轴和 task feature 轴混在了一起。

**最低修复标准**：

- 明确拆成两条路径：
  - Task path：`k=1`，只用 L0 indices 构造 `h_0`。
  - Reconstruction path：`k=1..4`，评估 PSNR/LPIPS，以及重建图像的 classifier accuracy。
- 如果仍保留 `k=1..4` 的 `L0_bow` 表，必须解释它只是在检验“额外传输层不会破坏 L0”，不能作为“多层带来任务增益”的证据。

**建议改法**：

- 主任务表只放：
  - no channel `L0_bow`
  - AWGN 0/+5/+10 dB `L0_bow`
  - Rayleigh +5/+10 dB `L0_bow`
- 另设重建表：
  - `k=1,2,3,4` 的 PSNR/LPIPS
  - 重建图 classifier accuracy

**完成标志**：

- 论文中不再让读者误以为 `k=4` 的 task accuracy 是由 L1-L3 提升出来的。

## P1 投稿前强烈建议解决

### H4. Rayleigh 0 dB breakdown 缺少对照语境

**严重级别**：Major

**问题**：Rayleigh 0 dB 下主方法准确率明显下跌。稿件承认 breakdown point，但没有外部基线说明这个失败是普遍链路困难，还是 RS-Token 特有弱点。

**修复标准**：

- 在 Rayleigh 0 dB 加入至少一个外部基线。
- 或将 graceful degradation 的主张严格限定到 `SNR >= +5 dB`。

**建议表述**：

> RS-Token provides a task-preserving operating envelope in the moderate-to-high SNR regime, while Rayleigh 0 dB marks a breakdown point requiring physical-layer protection such as FEC, HARQ, or equalization.

### H5. 信道模型偏理想化

**严重级别**：Major

**问题**：当前信道是闭式 BER 独立 bit flip，属于 uncoded BPSK hard-decision simulation。它适合做第一版，但不能支撑过强的真实 UAV/SmallSat deployment claim。

**最低修复标准**：

- 在方法中明确写成 uncoded hard-decision baseline。
- 避免暗示已经覆盖真实物理层链路。
- 若资源允许，加入一个简单 FEC 或 packet-level failure 实验。

**完成标志**：

- 文中所有“真实部署”表述都改为“deployment-motivated simulation”或类似克制表述。

### H6. 参考文献存在 TODO/占位

**严重级别**：Major

**问题**：`rs_token.bib` 中有多个 `[Author list TODO]`、`coauthors TODO`、venue/page/DOI 未核验项。正式投稿时这是明显扣分点。

**必须处理的条目**：

- `moc-rvq`
- `revqom`
- `gao2022task`
- `vilau`
- `semclip`

**修复标准**：

- 所有作者、题名、venue、年份、页码、DOI/arXiv ID 必须核验。
- 未能核验的文献不要作为 novelty claim 的核心支撑。

### H7. RemoteCLIP 不可替代性的表述偏强

**严重级别**：Major

**问题**：已有结果显示 OpenAI CLIP 也能带来大部分 L0 语义增益。RemoteCLIP 的优势主要体现在中等 SNR 或 Rayleigh 中高 SNR 下的额外稳健性，而不是绝对不可替代。

**建议改成的主张**：

- 层级蒸馏机制是主贡献。
- V-L teacher 能提供大部分语义化增益。
- RemoteCLIP 在遥感相关退化信道下提供额外 `4-7 pp` 稳健性收益。

**完成标志**：

- 摘要、引言、讨论中不再出现过强的 “cannot be replaced” 类表达。
- Teacher ablation 被解释为“额外收益”，不是“唯一可行性证明”。

### H8. 编码器/解码器可复现细节不足

**严重级别**：Major

**问题**：方法中有 RVQ 和 distillation 公式，但 encoder/decoder backbone 描述还不够完整。

**最低修复标准**：

- 写清：
  - base channels
  - channel multipliers
  - residual block 数量
  - downsampling ratio
  - normalization/activation
  - parameter count
  - 是否使用 attention

**完成标志**：

- 审稿人不需要读代码就能复现主干结构。

## P2 写作与呈现硬伤

### H9. 作者单位和投稿元信息仍有占位

**严重级别**：Major-Minor

**问题**：LaTeX 中仍有 `School of XXX`。单作者稿件中 `markboth` 用了 `et al.`。

**修复标准**：

- 补齐真实单位。
- 单作者时去掉 `et al.`。

### H10. 摘要和结论 claim 过满

**严重级别**：Major-Minor

**问题**：摘要中已经写出多个很强结论，但在外部基线和多 seed 还没补前，表述会显得超前。

**修复标准**：

- 在补实验前，所有强比较只限定为内部 baseline。
- 补实验后，再恢复和外部 baseline 对比有关的 claim。

### H11. 页数、表格和 overfull hbox

**严重级别**：Minor

**问题**：当前 PDF 约 7 页，README 目标是 5 到 6 页。LaTeX log 中存在多个 table overfull warning。

**修复标准**：

- 压缩摘要、相关工作、讨论。
- 宽表改为更紧凑列名或拆表。
- 最终 PDF 不应有明显 overfull table。

## 建议执行顺序

1. 修正评估口径：拆清 task path 和 reconstruction path。
2. 跑 3 个 model seeds：`rvq_baseline` 和 `rvq_distill`。
3. 训练或固定一个 clean AID classifier，用于重建图分类评估。
4. 做 JPEG2000/BPG/WebP 同 bit-budget 外部基线。
5. 汇总 `mean ± std` 主表。
6. 清理 bib TODO 和作者信息。
7. 再改论文正文 claim。

## 可作为完成定义的最小投稿包

- [ ] 主方法和 no-distill baseline 均有 3 model seeds。
- [ ] 主表均报告 `mean ± std`。
- [ ] 至少一个外部压缩/传输基线。
- [ ] `k` 与 `h_0` 评估逻辑被拆清。
- [ ] Rayleigh 0 dB 被放入明确语境。
- [ ] `rs_token.bib` 无 TODO 占位。
- [ ] LaTeX 作者单位和 `markboth` 无占位错误。
