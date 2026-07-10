# RS-Token 论文复审报告

复审日期：2026-06-08  
复审语言：中文  
复审范围：`paper_draft/latex/rs_token_v0.4_zh.md`、`paper_draft/latex/rs_token_v0.4.tex`、`paper_draft/latex/rs_token.bib`、`rstoken/logs/v04_tables/*`、`rstoken/logs/e23_ldpc_key5seeds_mean_std.csv`、`rstoken/logs/e19_classic_baselines_summary.md`、关键实现脚本与模型代码。  
使用的本地 skill：`airesearch-ml-paper-writing`、`airesearch-ara-compiler`、`airesearch-ara-rigor-reviewer`，并用 `airesearch-academic-plotting` 的图表审查原则检查现有图表/图表引用。  
重要约束：本报告不改原稿；无法从本地文件验证的引用或实验，一律标为“待核查/缺失”，不凭记忆补充。

---

## 1. 总体审稿意见：接收风险判断

**总体判断：当前稿件已经具备“可投递的技术短文/初稿”基础，但若按顶会/高水平期刊标准，接收风险为中高。**

正面来看，论文的核心叙事已经比早期版本稳健：

- 核心问题明确：遥感图像通信中存在“低码率优先任务语义，高码率逐步补重建细节”的层次化需求。
- 方法边界相对清楚：RS-Token 是 RVQ index 表示上的 L0 RemoteCLIP 蒸馏机制，而不是新物理层编码、不是 5G NR LDPC、也不是完整 codec+channel-code 系统。
- 证据链基本成形：L0 任务路径、layered probe、重建路径、LDPC 兼容性、WebP/JPEG2000 外部压力测试被分开叙述。
- 主稿中多处主动降调，例如明确 WebP/JPEG2000 是 unprotected stress test，LDPC 是 custom sparse LDPC，task path 不用于 k=2..4 重建结论。

主要风险在于：

- **统计严谨性不均衡**：任务路径有 3 个模型 seed 的 mean±std，但重建路径主表是 `single model seed; reconstruction path supports k=1..4 claims`，这会让 reviewer 质疑重建结论的稳定性。
- **外部 baseline 仍不足以支撑强“通信系统优越性”结论**：WebP/JPEG2000 的实际 bit 数、保护方式、失败模式与 RS-Token 不完全公平；稿件已降调，但仍需更明确地避免读者误解为完整系统比较。
- **引用存在 camera-ready 硬风险**：`rs_token.bib` 中 MOC-RVQ、ReVQom、Gao 2022、VILA-U、SemCLIP 等条目含 TODO 或作者占位，不能进入正式投稿版本。
- **泛化范围窄**：当前主证据来自 AID scene classification；对 object detection、segmentation、多数据集、真实信道/调制链路的泛化尚未验证。
- **方法可复现性尚可但仍需集中化**：代码里有训练/评估脚本和配置，但论文需更明确列出 split、模型 seed、channel seed、classifier 训练细节、LDPC 参数和生成表格命令。

**审稿式结论**：如果目标是 workshop、中文期刊初稿或遥感通信方向的短论文，当前版本经过引用修复和若干措辞降调后有希望；如果目标是 IEEE TGRS/TWC/TCOM/JSAC 或 ML 顶会，需要补强多 seed 重建、严格外部 baseline、更多数据集/任务泛化和引用真实性。

---

## 2. 核心贡献是否成立

| 贡献 | 当前证据 | 成立程度 | 主要风险 | 建议 |
|---|---|---:|---|---|
| C1：RemoteCLIP 引导的 L0 语义分配机制 | `table2_task_path_mean_std.csv` 显示 no-channel h0 acc 从 58.23±1.57% 到 83.33±0.81%；AWGN/Rayleigh 下同样提升 | 强 | “RemoteCLIP 独有性”被 OpenAI CLIP ablation 降低：OpenAI CLIP 也能到 80.80% | 表述为“V-L teacher distillation + RemoteCLIP domain prior”，不要写成 RemoteCLIP 是唯一原因 |
| C2：前缀式 RVQ index 传输机制 | 方法公式 `B(k)=k*T*log2K`，AID 配置下 2,560/5,120/7,680/10,240 bits/image | 中强 | 更像机制设计/工程组织，不是单独理论贡献 | 强调其价值在“可评估、可分层、可接入信道编码”的表示组织 |
| C3：任务路径与重建路径分离评估协议 | 主稿、脚本 `09_eval_rvqs_recon_task_split.py`、claim audit 都明确区分 h0/L0_bow 与 PSNR/LPIPS/recon cls | 强 | 这是评估协议贡献而非算法核心贡献 | 可保留，但不要把协议包装成过强 novelty |
| C4：AWGN/Rayleigh 下验证 L0 任务鲁棒性与多层重建 | 任务路径 3 seeds；重建路径单 seed；LDPC 5 channel seeds | 中 | 重建统计不充分；Rayleigh +5 dB 退化明显；无真实无线链路 | 用“验证/观察到”而非“证明鲁棒”；把 Rayleigh +5 dB 写成边界 |
| C5：可与 LDPC 叠加使用 | `e23_ldpc_key5seeds_mean_std.csv`，custom rate-1/2 systematic sparse LDPC | 中 | 不是标准 LDPC；不是最优 coding；只支持兼容性，不支持通信系统 SOTA | 保持 supplement/compatibility 定位 |
| C6：相对 WebP/JPEG2000 的外部压力测试 | E19/E23 表格，decode failure 清楚 | 弱到中 | 实际 bit 不严格匹配；未加 HARQ/标准 ECC；JPEG2000 清洁准确率低 | 只作为 failure-mode illustration，不作为主 baseline 胜出证据 |

---

## 3. Major Weaknesses

### MW1：重建路径只有单模型 seed，支撑不了高置信稳定性结论

`table3_reconstruction_path.csv` 的 scope 明确写着 `single model seed; reconstruction path supports k=1..4 claims`。主稿也在 limitations 中承认“Reconstruction-path sweeps are reported for the main model seed”。这说明 k=1..4 的 PSNR/LPIPS/recon classifier 改善虽然方向合理，但统计上弱于任务路径。

**为什么影响录用**：论文的一个核心卖点是“L1-L3 提供渐进式重建细节”。如果只有单 seed，reviewer 可能认为这是某个 checkpoint 的特性，而不是方法稳定性质。

**建议**：至少补 `rvq_distill_s41/s42/s43` 的 no-channel 和 AWGN +5 dB reconstruction sweep，并报告 mean±std。若算力不足，在主稿中把重建结论改为“main-seed evidence suggests”。

### MW2：外部 baseline 公平性不足，不能支撑强系统比较

WebP 在 target 2,560 bits 时 actual_bits_mean 约 14,164 bits，远高于目标；JPEG2000 更接近 bit budget，但 clean classifier accuracy 在低码率明显低。E19 是 unprotected bitstream；E23 是自定义 LDPC 而非标准强 ECC/HARQ 系统。

**为什么影响录用**：通信/遥感通信 reviewer 会非常敏感 baseline 公平性。若措辞稍强，容易被质疑“打弱 baseline”。

**建议**：继续保留外部 baseline，但标题和结论写成“stress-test / failure mode”，不要写“outperforms conventional compression systems”。如果要做强比较，需要：严格 rate control、同等 transmitted bits、相同 channel code、可能加 BPG/HEVC/VVC/BPG+LDPC 或 JPEG2000+标准 LDPC/HARQ。

### MW3：引用条目存在 TODO，占位引用会导致直接 camera-ready 风险

`rs_token.bib` 中存在 `[Author list TODO]`、`[coauthors TODO]`、`TODO: confirm authors and full title` 等。涉及 MOC-RVQ、ReVQom、Gao 2022、VILA-U、SemCLIP。

**为什么影响录用**：引用不完整可能被视为不严谨；若条目不存在或信息错误，属于学术诚信/格式硬伤。

**建议**：投稿前逐条用 DOI、IEEE Xplore、arXiv、Semantic Scholar 或 CrossRef 核验；无法核验的引用从正文移除或标为待补，不要进入正式版本。

### MW4：任务泛化范围较窄，当前只强支持 AID 场景分类

主证据是 AID 30 类 scene classification；任务路径使用 L0_bow 线性 probe。没有多数据集（如 NWPU-RESISC45、UCMerced、PatternNet）、没有检测/分割任务，也没有真实下游遥感业务指标。

**为什么影响录用**：标题和摘要若写“remote-sensing communication”容易被理解为更宽泛的遥感图像通信。当前证据实际是“scene-level land-cover semantics on AID”。

**建议**：短期在摘要、引言、limitations 降调为“AID scene-level evaluation”；中期补一个外部遥感场景数据集；长期补 detection/segmentation 或 human inspection 指标。

### MW5：RemoteCLIP novelty 容易被 OpenAI CLIP ablation 削弱

主稿 discussion 已诚实写明 OpenAI CLIP 也能把 L0_bow 提升到 80.80%，RemoteCLIP no-channel 仅高 1.60 pp；RemoteCLIP优势主要在 degraded-channel regimes。

**为什么影响录用**：如果标题/摘要过度强调 RemoteCLIP，reviewer 可能问：贡献到底是 RemoteCLIP，还是任意 V-L teacher distillation？

**建议**：贡献表述改为“domain V-L teacher guided L0 allocation”，RemoteCLIP 是本文选择的遥感教师；强调 degraded channel 下的额外收益，但不要声称 RemoteCLIP 是唯一可行教师。

### MW6：信道模型和 LDPC 实验仍偏工程模拟，真实通信结论需要降调

当前 index corruption 基于 BPSK AWGN/Rayleigh 的 BER/硬判决模拟，LDPC 是自定义 systematic sparse code + min-sum BP。没有软解调整体链路、AMC、HARQ、interleaving、burst fading、真实无线采集或标准码。

**为什么影响录用**：通信方向审稿人可能认为物理层过简化，无法支撑“channel-robust communication system”的强表述。

**建议**：把“channel-robust”限定为“under simulated BPSK AWGN/Rayleigh index corruption”；未来工作列出 soft demodulation、interleaving、HARQ、standard LDPC/Polar。

### MW7：图表在 LaTeX 主稿中使用不足，中文稿和英文稿图表不完全同步

中文稿引用了图 2、图 3：`fig_exp_l0_task_robustness_v3.png` 和 `fig_exp_progressive_reconstruction_v3.png`；英文 LaTeX 当前只 include 了方法图。图表文件确实存在，但英文稿未纳入实验图。

**为什么影响录用**：实验叙事过度依赖表格，缺少直观趋势图；如果中文稿作为主导，英文稿同步不足会降低可读性。

**建议**：在英文稿中加入 L0 robustness 和 progressive reconstruction 两张图，或至少在报告中说明最终版会加入；图注明确 metric、seed、channel、是否单 seed。

---

## 4. Minor Weaknesses

1. **标题可能略强**：`Channel-Robust Remote Sensing Communication` 容易暗示完整通信系统；可考虑 `for Simulated Noisy-Channel Remote-Sensing Image Transmission` 或在摘要首句限定模拟信道。
2. **符号 L0/q1 不完全直观**：英文 LaTeX 中 `g_\psi(q_1)` 表示 L0 quantized representation，可能引起层编号混淆；建议统一 L0/q0 或明确 “we use one-based notation in equations”。
3. **WebP/JPEG2000 clean baseline 叙述需谨慎**：WebP clean actual bits 明显高于 target，不能直接与 2,560-bit RS-Token 清洁点比较。
4. **ResNet34 evaluator 需要更多细节**：论文写 96.10% top-1 和 95.94% macro-F1，但应说明 train/test split、输入预处理、是否与 tokenizer 训练数据重叠。
5. **层级 probe 是辅助机制分析**：表 2 是 single main configuration，不应作为强统计主证据；当前写法基本谨慎，但图注/表注可再写“single checkpoint”。
6. **README/results 有编码损坏和旧叙事痕迹**：不影响主稿，但如果开源仓库随文提交，会影响复现体验。
7. **LDPC 表格只列 selected rows**：若论文只放 selected rows，应说明完整 CSV 在 artifact 中，避免 cherry-picking 印象。
8. **图 1 可能偏展示型**：`fig_method_aech_image2pro.png` 若为 AI 生成架构图，需检查所有标签是否准确、无装饰性误导；最好准备可编辑 PDF/SVG 版本。

---

## 5. 实验严谨性审查

### Baseline adequacy

- **内部 baseline 充分**：`rvq_baseline` 与 `rvq_distill` 架构和传输协议一致，差异聚焦于 L0 RemoteCLIP 蒸馏，是检验核心机制的合适 baseline。
- **教师 baseline 有价值但未进入主实验表**：OpenAI CLIP ablation 对 novelty calibration 很关键，建议保留在 Discussion 或 Appendix。
- **外部 baseline 不足以做强比较**：WebP/JPEG2000 是 stress test，不是完整公平系统 baseline；不应替代 internal rvq_baseline。
- **缺失强通信 baseline**：DeepJSCC、Gao 2022、BPG/HEVC/VVC + channel code、标准 LDPC/Polar/HARQ 均未完成，不能声称 SOTA。

### Fairness

- 内部 baseline 公平性较好。
- 外部 baseline 的 target bits 与 actual bits 不完全一致，尤其 WebP；JPEG2000 虽接近目标，但 clean task accuracy 与视觉质量关系复杂。
- LDPC 比较使用 fixed transmitted bits 是合理方向，但 custom LDPC 的性能较弱，不能代表成熟系统。

### Ablation

- 已有：无蒸馏 vs RemoteCLIP 蒸馏、layered probe、OpenAI CLIP teacher、distill weight trade-off、LDPC compatibility。
- 缺失：多 seed reconstruction ablation、L0-only distill vs all-layer distill、不同 codebook size/grid size、不同数据集、不同 teacher backbone。

### Statistical reporting

- 任务路径：3 model seeds，mean±std，较强。
- LDPC：5 channel seeds，mean±std，较好，但是单模型 checkpoint。
- 重建路径：主表单模型 seed，弱。
- 外部 baseline：部分结果未见多 seed/置信区间，且 decode failure 强依赖实现。

### Generalization

- 数据集：当前主要是 AID。
- 任务：当前强支持 scene classification，不支持 detection/segmentation/general visual intelligence。
- 信道：支持模拟 AWGN/Rayleigh BPSK，不支持真实链路或标准协议。

### Robustness

- AWGN +5/+10 dB 下任务路径表现强。
- Rayleigh +10 dB 较好；Rayleigh +5 dB 明显退化但仍有任务可用性；Rayleigh 0 dB 是 breakdown boundary。
- 低 SNR / 强衰落不能写成稳定可用。

### Reproducibility

- 代码层面：训练、评估、LDPC、表格生成脚本存在，模型/配置路径较清楚。
- 论文层面：还需集中列出复现实验命令、seed、split、classifier ckpt、LDPC 参数、完整 artifact 路径。

---

## 6. 证据-主张矩阵

| 关键 claim | 支持证据 | 当前强度 | 是否需降调/补实验 |
|---|---|---:|---|
| L0 RemoteCLIP 蒸馏显著提升场景级任务判别性 | `table2_task_path_mean_std.csv`：58.23±1.57% → 83.33±0.81%；AWGN/Rayleigh 下同向 | 强 | 可保留“significantly/substantially improves”，但限定 AID scene classification |
| L0 是主要语义层，L1-L3 增加很少任务语义 | layered probe：86.0→88.0%，增量小；无蒸馏约 47-48% | 中强 | 标为机制分析；最好补多 seed 或 appendix |
| L1-L3 主要补重建细节 | no-channel/AWGN +5 k sweep 中 PSNR/LPIPS/recon cls 改善 | 中 | 因单 seed，写“suggest / indicate”；补 3 seed 最佳 |
| RS-Token 支持前缀式低码率任务路径与高码率重建路径 | 方法公式 + k=1..4 表格 | 中强 | 保留，但 bit 数是配置相关，不是普适常数 |
| RS-Token 在 AWGN/Rayleigh 下 channel-robust | AWGN +5/+10 强，Rayleigh +10 较强，Rayleigh +5 退化 | 中 | 改为“under simulated BPSK AWGN/Rayleigh, with clear degradation boundaries” |
| RS-Token 可与 LDPC 叠加 | E23 5 channel seeds，自定义 rate-1/2 LDPC | 中 | 只写 compatibility，不写 coding advance 或 standard LDPC |
| WebP/JPEG2000 在 noisy channel 下易失败 | E19 unprotected decode failure；E23 LDPC 后仍弱 | 中 | 只写 stress-test，不写完整 conventional system comparison |
| RemoteCLIP 是核心独特来源 | OpenAI CLIP 也强，RemoteCLIP额外收益主要在退化信道 | 弱到中 | 降调为“RemoteCLIP is a useful domain teacher; much gain comes from V-L distillation” |
| 方法适用于遥感通信一般任务 | 当前 AID scene classification | 弱 | 降调；补 NWPU/PatternNet 或检测/分割 |
| 可复现 | 脚本、configs、logs 存在 | 中 | 需要复现清单与命令；清理 README 编码 |

---

## 7. 修改优先级

### P0：投稿前必须处理

1. 修复 `rs_token.bib` 所有 TODO/占位引用；无法核验的引用从正文移除。
2. 在摘要、贡献和结论中继续限制外部 baseline：WebP/JPEG2000 仅为 unprotected/coded stress test，不是完整系统比较。
3. 明确重建路径为单模型 seed；若不补实验，所有重建结论用 “main-seed evidence suggests / indicates”。
4. 在英文 LaTeX 中加入或删除与中文稿一致的图 2/图 3，避免图文不同步。
5. 检查所有数值是否来自 `v04_tables` 或 E23 mean/std；不要从旧 `results.md` 的乱码内容复制。

### P1：显著降低拒稿风险

1. 补 3 model seeds 的 reconstruction-path sweep，至少 no-channel 与 AWGN +5 dB 的 k=1..4。
2. 增加一个外部遥感场景分类数据集，验证 L0 task path 泛化。
3. 把 OpenAI CLIP ablation 放入 appendix 或主文小表，帮助校准 RemoteCLIP claim。
4. 增加复现 appendix：训练命令、评估命令、seed、split、checkpoint、表格生成脚本。
5. 准备严格 rate-matched codec+ECC baseline，或明确列为 future work 不作比较。

### P2：增强论文完成度

1. 清理 `rstoken/README.md` 和 `rstoken/experiments/results.md` 编码问题，方便开源。
2. 补 L0-only distill vs all-layer distill ablation，更直接证明“只蒸馏 L0”的设计必要性。
3. 补 codebook size / latent grid / bit budget sensitivity。
4. 为架构图提供可编辑 SVG/PDF，确保出版质量。
5. 添加 limitations 小表：数据集、任务、信道、baseline、统计 seed 边界。

---

## 8. 建议改写

### 8.1 摘要改写建议（中文）

建议把当前摘要的“channel-robust”进一步限定为模拟信道与 AID 场景分类，避免泛化过强：

> 遥感图像下行链路常受带宽、噪声和衰落限制，而接收端的信息需求具有层次性：低码率下优先获得场景级判断，链路改善时再补充图像细节。本文提出 RS-Token，一种基于四层残差向量量化的分层离散表示。该方法仅将 RemoteCLIP 语义蒸馏施加到第一层 token，使 L0 更偏向遥感场景语义，后续层继续服务于渐进式重建。在 AID 场景分类与模拟 BPSK AWGN/Rayleigh 信道下，我们分别评估 L0 任务路径和多层重建路径。结果显示，RemoteCLIP 蒸馏显著提升 L0 bag-of-words 任务准确率；在无信道和 AWGN +5 dB 下，增加 RVQ 层可逐步改善 PSNR、LPIPS 和重建图分类准确率。补充的 WebP/JPEG2000 与自定义 rate-1/2 sparse LDPC 实验用于界定传统 bitstream 压力测试和信道编码兼容性，而不构成标准 5G NR LDPC 或完整 codec+channel-code 系统比较。

### 8.2 引言贡献段改写建议（中文）

建议贡献列表写得更“证据绑定”：

> 本文贡献可以概括为三点。第一，我们提出一种面向遥感场景语义的 L0 分配策略：在四层 RVQ tokenizer 中仅对第一层施加 RemoteCLIP 蒸馏，使最低码率 index 更适合场景级任务判断。第二，我们给出前缀式 RVQ index 传输与分离评估协议，将 k=1 的 L0 任务路径与 k=1..4 的重建路径明确区分，避免用任务指标解释重建质量或用重建指标解释语义保真。第三，我们在 AID 数据集和模拟 AWGN/Rayleigh 信道下验证该分层表示的有效性，并通过自定义 LDPC 与传统压缩 bitstream 压力测试界定其工程兼容性和适用边界。

### 8.3 实验结论段改写建议（中文）

建议用“支持/表明/提示”替代“证明”：

> 实验结果支持三个有边界的结论。首先，在 AID 场景分类任务中，L0 RemoteCLIP 蒸馏显著提高了最低码率任务路径的判别性；这一结论由三组模型随机种子的 h0/L0_bow 结果支持。其次，层级 probe 表明后续 RVQ 层对任务准确率的额外贡献较小，符合 L0 偏语义、L1–L3 偏重建残差的设计预期。第三，在主模型 seed 的无信道和 AWGN +5 dB sweep 中，增加传输层数可逐步改善重建质量；该结论仍需更多模型 seed 支撑其统计稳定性。LDPC 和 WebP/JPEG2000 结果应理解为兼容性与压力边界分析，而不是完整通信系统的 SOTA 对比。

---

## 9. 需要作者补充的材料

### 缺失或待核查文件

1. `rs_token.bib` 中所有 TODO 引用的真实 BibTeX、DOI、作者、venue。
2. E16 clean AID ResNet34 classifier 的训练日志、split、macro-F1 计算脚本或结果文件。
3. 3 model seeds 的 reconstruction-path CSV，若已经存在，需要明确路径；当前 v04 table 标注为 single seed。
4. OpenAI CLIP ablation 的原始 CSV/表格路径，若要在主文或 appendix 使用。
5. distill weight trade-off 的原始 CSV/表格路径，若要支持 w=0.5 选择。
6. 完整表格生成命令：`11_make_v04_tables.py`、`12_make_v04_figures.py` 的调用参数。
7. 若投稿需开源，需清理 README/results 中的乱码版本或另写 clean artifact README。

### 缺失实验

1. 3 seed reconstruction sweep。
2. 至少一个外部遥感场景数据集。
3. 更严格的 codec+ECC baseline：BPG/HEVC/VVC/JPEG2000 + standard channel code 或成熟 LDPC/Polar/HARQ。
4. L0-only vs all-layer distillation ablation。
5. 不同 codebook size、grid size、bit budget 的 sensitivity。
6. 更真实信道或 burst error / interleaving 评估。

### 缺失引用或待核查引用

- MOC-RVQ：作者、正式题名、会议/页码/DOI 待核查。
- ReVQom：作者、年份、会议、是否真实发表待核查。
- Gao 2022 task-oriented UAV remote sensing semantic communication：作者列表、期刊/卷页/DOI 待核查。
- SemCLIP：作者与 arXiv 条目待核查。
- VILA-U：若正文未引用可移除；若引用需核验。

---

## 10. 最值得先改的 10 个问题

1. **修复所有 BibTeX TODO**：这是最硬的 camera-ready 风险。
2. **补或降调重建路径统计**：单 seed reconstruction 是当前最大严谨性短板。
3. **把外部 baseline 全部写成 stress test**：避免被 reviewer 认为不公平比较。
4. **限制标题/摘要中的 channel-robust 范围**：明确是模拟 BPSK AWGN/Rayleigh，不是真实完整通信系统。
5. **加入英文稿实验图 2/图 3**：中文稿已有引用，英文稿应同步。
6. **把 RemoteCLIP novelty 降调为 domain teacher advantage**：承认主要增益来自 V-L distillation，RemoteCLIP 在退化信道有额外收益。
7. **补复现 appendix**：seed、split、命令、checkpoint、表格生成流程。
8. **补一个外部遥感数据集或明确 future work**：否则泛化到 remote sensing communication general case 较弱。
9. **清理开源文档乱码**：仓库随文提交时会影响可信度。
10. **统一符号和层编号**：L0/q0/q1 表达要一致，避免方法节读者困惑。

---

## 11. 简短结论

当前 RS-Token 稿件最强的部分是“内部同架构 baseline 下，L0 RemoteCLIP/V-L 蒸馏显著提升 AID 场景任务路径”，这条主线证据充足且叙事清楚。最需要收缩的部分是“完整通信系统鲁棒性”和“传统压缩系统比较”；最需要补强的是“重建路径多 seed 统计”和“引用真实性”。如果按这个方向修改，论文会从“有想法但容易被质疑越界”变成“边界清楚、证据链完整的分层遥感 token 通信方法”。

---

## 12. 外部复审补充意见（2026-06-08，scholar-evaluation + peer-review skill）

> 本节由外部 skill (`scholar-evaluation` + `peer-review`，K-Dense Inc.) 在英文 LaTeX 主稿 `rs_token_v0.4.tex` 上独立做了一遍复审后追加。原始报告见同目录下：
> - `paper_draft/latex/SCHOLAR_EVALUATION.md`（ScholarEval 8 维评分）
> - `paper_draft/latex/PEER_REVIEW.md`（IEEE 风格 Major/Minor Comments）
>
> 本节只列出与 §1–§11 **不重复或证据更具体**的发现，旧报告已涉及的（重建路径单 seed、外部 baseline 公平性、RemoteCLIP novelty 降调、bib TODO、跨数据集泛化）此处不再重述。

### 12.1 总体打分

| 项 | 值 |
|---|---|
| 平均分（ScholarEval 8 维） | **3.56 / 5** |
| 推荐意见 | **Major Revision** |
| 最适合 venue | **IEEE GRSL**（letter，模板已对齐） |
| 阻塞问题数 | 2（D2 文献 / D8 引用，均与下面 12.2 同源） |

| 维度 | 分数 | 状态 |
|---|---|---|
| D1 Problem Formulation | 4 / 5 | ok |
| D2 Literature Review | **2 / 5** | ⚠ 阻塞 |
| D3 Methodology & Design | 4 / 5 | ok |
| D4 Data Sources | 4 / 5 | ok |
| D5 Analysis & Interpretation | 4.5 / 5 | ★ 亮点 |
| D6 Results & Findings | 3.5 / 5 | major（重建单 seed） |
| D7 Writing & Presentation | 4 / 5 | ok |
| D8 Citations & References | **2.5 / 5** | ⚠ 阻塞 |

### 12.2 旧报告未覆盖的新发现

#### N1（阻塞）：bib 中 9 条文献**完全未在正文出现**，不只是 TODO 占位

旧 §3 MW3 只指出 `rs_token.bib` 有 `TODO/作者占位`。但实际更严重的是：bib 定义 16 条，正文 `\cite` 实际只有 **7 条**。**9 条从未引用过**：

```
zhang2024speechtokenizer, moc-rvq, revqom, deepjscc, vilau, beitv2, semclip, proakis
```

其中至少 4 条是**直接相关**而非装饰性引用：
- **moc-rvq**（Zhou et al., arXiv:2401.01272）—— 同样 RVQ + 多码本 + 语义通信，**最直接的 concurrent work**，必须在 Related Work 做 positioning，不能只在 bib 里挂着
- **deepjscc**（Bourtsoulatze, Burth Kurka, Gündüz, 2019）—— JSCC 奠基论文，几乎所有语义通信论文都引；当前 §1 ¶2 讨论 compress-then-transmit 时缺这一引用是显眼的洞
- **semclip**（Hu et al., arXiv:2502.18200）—— 同样用 CLIP 系做语义通信，必须 acknowledge
- **revqom**（ICASSP 2026）—— RVQ + comm + 感知任务，spirit 很近

**为什么比 TODO 占位更严重**：IEEE 编辑预审会同时做 `bibtex` 输出检查和 `\cite` 覆盖检查。**unused entries 会触发 warning 并出现在编辑端的预审报告里**，这是和 TODO 占位独立的问题。

**修复建议（覆盖旧 P0 §1）**：
1. 把 `moc-rvq`、`deepjscc`、`semclip`、`revqom` **写进正文**——不是补 BibTeX 字段，是补 §1 / §2 中的 positioning 段落。
2. `vilau`、`beitv2`、`zhang2024speechtokenizer` 如果不打算在正文 acknowledge，从 bib 删除。
3. `proakis` 在 §3.4 BPSK channel model 处补一个 cite 即可。

#### N2（阻塞）：标题"Channel-Robust"和**定量结果**直接矛盾

旧 §4 Minor 1 只说"标题略强"。复审从结果表里找到了**具体的定量矛盾**：

- Table `tab:recon_path`：Rayleigh +5 dB k=4 unprotected 重建分类准确率 = **21.0%**
- Table `tab:ldpc_rstoken`：Rayleigh +5 dB k=1 即使**加 LDPC 保护**后还是 **47.2 ± 0.6%**
- 论文自己 §5 ¶4 也承认："Rayleigh +5 dB remains difficult ... channel coding improves robustness but does not fully remove the effect of strong fading"

→ 标题的 headline 词被论文自己的结论否定。

**修复建议**：标题改成 **"Channel-Aware"** 或 **"Channel-Adaptive"** ——后者更贴合 RS-Token "按信道选 k" 的语义。这是 10 分钟的改动，能换掉一个高概率被 reviewer 抓的攻击面。

#### N3（major）：§4.3 layered probe 结论存在**循环论证嫌疑**

旧报告只把 layered probe 列为"中强支持"。复审指出这个实验有逻辑问题：

> 蒸馏**只施加在 L0**（构造决定）→ L1-L3 当然没有任务语义增量
> ↓ 但论文写作语气是
> "supports the intended specialization" → 暗示这是**发现**了 L1-L3 不携带语义

实际上 probe 只能证明**工程意图被实现了**，不能证明 L1-L3 **本质上**不携带语义。一旦审稿人意识到这一点，§4.3 的推断力会大幅打折。

**修复建议**（可选两档）：
- **低成本**：把 §4.3 Conclusion 改写为"The probe **confirms** that the engineered specialization was achieved at training time, consistent with L1–L3's reconstruction-residual role"——把"支持发现"改成"确认工程实现"。
- **高成本**（消除循环）：训练一个 `rvq_distill_L1`(蒸馏只施加在 L1)的反事实 checkpoint，证明蒸馏目标层确实**决定**语义所在层。这条路在新实验手册的 E25 上稍作改造即可顺带做掉。

#### N4（minor）：摘要超长

旧报告未覆盖。

- 当前摘要 ~370 词
- IEEE GRSL 摘要上限 **200 词**

**修复建议**：保留"一句话 framing + 一句话方法 + 三个核心数字 + 一句 supplementary"。30 分钟改完。

#### N5（minor）：贡献 #2 措辞过强

旧报告未直接指出。当前 contribution #2 写"We **formulate** prefix-style RVQ index transmission"——但**前缀解码是 RVQ 自 SoundStream/EnCodec 起的定义性属性**，不是 RS-Token 提出的。

**修复建议**：改成 "We **adapt** prefix-style RVQ index transmission to remote-sensing communication and tie its operating points to scene-level task accuracy"。把 novelty 从"提出机制"改为"应用到新场景并绑定任务指标",这是更诚实的措辞,也更难被审稿人挑战。

#### N6（minor）：Table 4 vs Table 3 对照断裂

旧报告未覆盖。Rayleigh +5 dB k=1 的"unprotected 21% → LDPC 47%"是 **LDPC 实验最有价值的 26-pp delta**，但它**横跨两张表**（Table 3 unprotected vs Table 4 LDPC），读者必须自己拼。

**修复建议**：在 Table 4 加一个 `unprotected baseline` 列，或把 Table 3+4 合并成"with LDPC ✗/✓"列。读者一眼能看到 LDPC 的边际收益。

#### N7（minor）：跨表 std 报告不一致

- Table 1（任务路径，3 model seeds）：mean ± std
- Table 4（LDPC，5 channel seeds）：mean ± std
- Table 5（WebP/JPEG2000）：**mean only**
- Table 3（重建路径）：**point estimate**

→ 同一篇论文里有 4 种统计报告风格。reviewer 会注意到。

**修复建议**：完成 §13 E24 后，Table 3 转 mean ± std。Table 5 要么补 channel-seed 重复实验，要么明确标注"channel-seed variance not measured for conventional codecs"。

### 12.3 给作者的额外问题（peer-review 风格）

这些问题如果作者能在 cover letter 或 rebuttal 提前回答，可以拆掉一批审稿火力：

1. Table 3 当前是单 seed 还是 3 seed 但只取主 seed？如果数据已存在为何不报 std？
2. Table 4 k=4 的 $h_0$ 列写 `--`，是设计上 k=1 only 还是没跑？需要脚注。
3. 蒸馏教师作用于原图 $x$ 还是重建 $\hat{x}_1$？§3.3 暗示原图，需要明确。
4. `rvq_baseline` 和 `rvq_distill` 是否使用相同 init seed？如果不是，25 pp 的差距里有多少来自 init 噪声？
5. $h_0$ 是 1024 维 BoW，**是位置加权还是纯直方图**？§3.4 和 §4.1 没对齐。
6. AID split ratio 是多少？per-class 还是 random？

### 12.4 与旧报告 P0/P1/P2 的并轨

旧 §7 已经定义了 P0/P1/P2 优先级。本节新发现并入：

| 旧优先级 | 旧条目 | 新增内容 |
|---|---|---|
| **P0** | §7 P0 #1 修 bib TODO | + **N1**：把 9 条 unused 全部要么写进正文要么删除；MOC-RVQ/DeepJSCC/SemCLIP 必须正文 cite |
| **P0** | §7 P0 #4 英文稿图同步 | （独立） |
| **P0**（新） | — | **N2**：标题改 Channel-Aware/Adaptive |
| **P0**（新） | — | **N4**：摘要砍到 ≤200 词 |
| **P1** | §7 P1 #1 多 seed 重建 | （已对齐 E24） |
| **P1**（新） | — | **N3 低成本版**：§4.3 conclusion 改写 |
| **P1**（新） | — | **N6**：Table 4 加 unprotected 对照列 |
| **P2**（新） | — | **N3 高成本版**：训练 `rvq_distill_L1` 反事实 |
| **P2**（新） | — | **N5**：contribution #2 措辞收紧 |
| **P2**（新） | — | **N7**：跨表 std 一致化 |

执行这些 P0+P1 后，预计 ScholarEval 平均分能从 3.56 提升到 ~4.0，从 Weak Accept 升到 Accept（letter 档）。

---
