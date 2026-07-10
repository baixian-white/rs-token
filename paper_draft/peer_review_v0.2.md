# 同行评审报告 · RS-Token 短文 v0.2

> **审稿人**：自审，模拟 IEEE GRSL / TGRS / IGARSS 审稿视角  
> **稿件**：Hierarchical RemoteCLIP Distillation for Channel-Robust Semantic Communication of Remote Sensing Imagery  
> **类型**：短文 / Letter 风格  
> **日期**：2026-06-01  
> **建议**：**大修**（修订后偏向接收）

---

## 1. 总体评价

本文提出 **RS-Token**：一种由 RemoteCLIP 蒸馏监督的 4 阶残差向量量化（Residual Vector Quantization, RVQ）遥感图像语义通信框架。核心主张是：RemoteCLIP 监督会驱动第一层 RVQ 码本（L0）编码地物类别语义身份，而后续层主要承载像素细节；因此在信道恶化时，系统能够在保持任务保真度的前提下实现渐进式退化。接收端可根据信道质量和任务需求动态选择传输层数 $k \in \{1,2,3,4\}$。在 AID 数据集上，基于 AWGN 与 Rayleigh 衰落仿真的实验表明：仅传输 L0（2,560 bits/img）在中高 SNR 区间的分类准确率比无蒸馏的 4 层传输（10,240 bits/img）高 25-32 个百分点；同时，在实际部署更相关的中等 SNR 信道（AWGN 0 dB、Rayleigh +5 到 +10 dB）中，RemoteCLIP 教师比通用 OpenAI CLIP 教师高 4-7 个百分点。

这项工作**选题及时、在当前范围内方法扎实，并且与遥感部署场景高度契合**（无人机 / 应急 / 边缘端 / SmallSat）。其与 SpeechTokenizer 在音频中进行层级蒸馏的跨模态类比有恰当引用，也提供了可信的归纳支持。不过，稿件目前存在两个主要短板：第一，**缺少标准语义通信基线**（如 DeepJSCC、JPEG2000+LDPC、同一评测协议下的 MOC-RVQ），这是 TGRS/JSTARS 接收的最大障碍；第二，**随机信道实验只报告单种子结果**，不符合当前 IEEE 对统计报告的期待。此外，编码器 / 解码器结构、码本利用率、训练稳定性等方法细节也需要补充。完成这些修订后，本文会成为一篇有竞争力的工作。

**主要优点**
- 部署场景清晰且动机充分，SNR / 衰落区间与无人机、应急、对地观测链路等场景对应具体。
- 核心结果有冲击力：4 倍带宽降低同时带来 25-32 个百分点准确率提升，结果显著且可复现。
- 分层探针分析（Table 3）明确证明：L0 语义现象来自蒸馏，而不是 RVQ 多阶段结构本身。
- 与 Gao et al. (TGRS 2022) 的协议级对齐支持后续直接比较。

**主要不足**
- 没有与成熟 SemCom / 图像传输基线进行定量比较（DeepJSCC、JPEG2000+LDPC、MOC-RVQ）——这是最致命的缺口。
- 所有信道结果均来自单一随机种子，没有置信区间或方差估计。
- 编码器 / 解码器结构没有描述，只给出了 RVQ 与蒸馏头。
- Rayleigh 0 dB 下性能断崖（53.2% → 16.6%）虽有承认，但没有与替代方法比较；读者无法判断这是有竞争力还是比已有方法更差。
- 部分表述仍带有宣传口吻（如“唯一驱动因素”“不可被通用预训练所替代”），建议改为更克制的学术表达。

---

## 2. 主要意见

### M1. 缺少基线，从根本上削弱了核心比较 [Critical，必须解决]

“4× 带宽、25-32 个百分点准确率增益”这一标题性结果完全是相对于**无蒸馏 RVQ 基线（Stage 2）**计算得到的。这是一个*内部*消融，而不是与已有工作的比较。若论文声称提出一种新的任务保真传输范式，则必须在相同比特预算和 SNR 下至少与以下一种方法进行比较：

1. **JPEG2000（或 BPG）+ LDPC，在匹配比特预算下比较**（2,560 与 10,240 bits/img），并使用单独训练的下游分类器对重建图像进行分类。这是 Section 1 所批评的传统“先压缩、再信道编码”基线；缺少该比较会削弱“cliff effect”的动机论证。
2. **DeepJSCC**（Bourtsoulatze et al.，或 Luxembourg SnT 的 EO+JSCC 系列）在匹配复杂度下进行比较，可报告重建图像分类结果，或使用带分类头的 JSCC 变体。
3. **MOC-RVQ** 的遥感分类协议版本，即便原始工作针对自然图像。可以复现或实现一个接近变体（去掉 RemoteCLIP 蒸馏、保留相同 RVQ×4 主干），以判断 MOC-RVQ 的 64-QAM 对齐码本设计本身是否贡献部分增益。
4. **Gao et al. (TGRS 2022)**，采用同样的 AID 30 类分类 + Rayleigh/AWGN 协议，因为稿件明确声称与该工作“协议级对齐”。

Section 5.3 将其列为未来工作，但对于 TGRS/JSTARS 投稿这是必需项；对于 GRSL 或 IGARSS，要求可以稍低，但至少需要一个外部基线（最现实的是 Gao 2022，因为协议匹配）。**建议**：重投前至少加入 Gao et al. (TGRS 2022) 和一个 JSCC 风格基线（DeepJSCC 即可），并在相同 SNR sweep 下比较。

### M2. 单种子信道结果 [Critical，必须解决]

Section 4.2 报告为“single training run”。Section 4.4（Table 2）和 Section 4.6（Table 4）给出的信道结果没有任何跨种子的方差信息。对于 BPSK over AWGN 的高 SNR 场景（如 +10 dB，BER ≈ 4×10⁻⁶），单次 Monte Carlo 实现近似确定；但在低 SNR（-5 dB AWGN，BER ≈ 0.21；Rayleigh 0 dB，BER ≈ 0.15）下，不同信道实现之间的方差不可忽略。-5 dB 处的差距（k=1: 8.2% vs k=4: 5.7%，$\Delta=2.5$ 个百分点）非常接近噪声水平，平均后未必仍然成立。

内部记录（`pre_experiments_log.md`）显示 +10 dB PSNR 检查（E9）曾跑过 3 个信道种子且结果完全一致；这对高 SNR 区间是可以的，但不能回答低 SNR 方差问题。**建议**：
- 对所有 SNR $\le$ +5 dB 的点至少用 5 个信道种子重跑，并在 Tables 2、4、5（及对应图）中报告 mean ± std。
- Stage 3 训练至少用 3 个不同模型种子重跑，在 Table 1 中报告无信道和 AWGN 0 dB 条件下 L0_bow 的 mean ± std，以证明 +24.7 个百分点增益不是种子偶然性。

+24.7 个百分点增益足够大，几乎肯定能经受方差检验，但报告方差已经是当前社区期待。

### M3. 编码器 / 解码器结构未记录 [Major，应解决]

Section 3.1 给出了潜变量尺寸（16×16×256）和 RVQ 规格，但编码器 $E$ 与解码器 $D$ 本身仍然是黑箱。为了可复现性，TGRS/GRSL 读者会期待看到：
- 卷积结构（通道倍率、每阶段残差块数、归一化方式）
- 下采样倍数及其实现方式
- 参数量
- 设计是否沿用 VQ-VAE-2、VQ-GAN，或自定义主干

内部配置（`rvq_distill.yaml`）显示 `base_channels=64`、`channel_multipliers=[1,2,4,4]`、`num_res_blocks=1`、总参数约 10.87M，这些应明确写入正文。**建议**：增加一个约 1 段的小节（或扩展 §3.1），描述编码器 / 解码器结构和参数量，最好说明其参考的源模式（如“follows the VQ-VAE-2 backbone with...”）。

### M4. “L0 carries 95% of semantics” 需要精确定义 [Major]

Section 4.5 写道：“L0 单层即承担约 95% 的最终语义能力”。这个 95% 可以来自线性探针准确率的 $86.0 / 88.0 = 97.7\%$，也可以来自相对于无蒸馏增益的 $\Delta_{L0} / \Delta_{L0 \to L_3} = 38.3 / 40.3 = 95\%$。当前文本似乎指后者，但没有明确说明。这个定义很重要，因为审稿人会重新计算。**建议**：写出公式：“L0 captures $86.0 / 88.0 = 97.7\%$ of the full-stack accuracy, or $(86.0 - 47.7) / (88.0 - 47.7) = 95.0\%$ of the distillation-induced gain over the no-distillation L0.” 或者去掉“95%”说法，直接报告从 L0 到 L0+L1+L2+L3 的绝对增量仅为 +2.0 个百分点，这样最不含糊。

### M5. Rayleigh 0 dB 性能断崖被承认，但缺少语境 [Major]

Section 5.3 提到 Rayleigh 0 dB 断崖（k=1: 16.6%，而无信道上限为 82.4%）是一个限制。然而，这个断崖正出现在论文“graceful degradation”核心主张本应成立的区间。若没有比较数字（如 JPEG2000+LDPC 或 DeepJSCC 在 Rayleigh 0 dB 下表现如何），读者无法判断 16.6% 相对替代方法是好是坏。很可能**两个替代方法也会在该区间崩溃**（JPEG2000+LDPC 在 LDPC waterfall 以下会灾难性失败，DeepJSCC 可能相对平滑但也会显著下降），但缺少这些数字会给审稿人留下明显攻击点。**建议**：要么 (a) 扩展实验，在 Rayleigh 0 dB 下加入至少一个基线；要么 (b) 将贡献重新表述为“在 SNR ≥ +5 dB 区间实现渐进退化，并在 Rayleigh 0 dB 明确出现 breakdown point”，避免过度声明。

### M6. 教师消融的解释存在过度外推 [Major]

Section 4.6 / Section 5.1 依据中 SNR 下 4-7 个百分点优势，得出 RemoteCLIP “不可替代于”通用 CLIP 的结论。数据支持更克制的说法。报告数字显示：
- 无信道：1.6 个百分点（很小）
- SNR ≥ +5 dB AWGN：1.7-2.9 个百分点
- AWGN 0 dB：6.7 个百分点（最强证据）
- Rayleigh +10 dB：4.2 个百分点

Section 5.1 末句“真正能扛信道恶化的是带遥感领域归纳偏置的 RemoteCLIP 蒸馏”修辞力度较强，严格来说并不完全由数据支持——OpenAI CLIP 在 AWGN 0 dB 下达到 46.5%，显著高于 1/30 = 3.3% 随机水平，并且在多数运行指标上与 RemoteCLIP 的 53.2% 同量级。§5.1 中关于机制的解释（“簇间间距更大、簇内更紧凑”）是合理猜想，但若没有 t-SNE / silhouette / 最近邻分析支持，仍属推测。**建议**：
- 将“真正能扛信道恶化的是 ... RemoteCLIP”软化为“RemoteCLIP 在最具部署相关性的中 SNR 区间提供了稳定且可测量的 4-7 个百分点优势”等类似表述。
- 至少加入一个定量几何分析（例如两种教师下 L0 码本质心的类间平均距离 / 类内平均距离比值）以支撑机制主张。这可基于已有 checkpoint 做 5 行分析，不是新实验。

### M7. HARQ + ACM 反驳放在引言中略显突兀 [Major-Minor]

Section 1 第二段包含了对 HARQ + ACM 作为完整解决方案的较长反驳。技术点本身没问题，但放置位置打断了从问题到方案的叙事，并且显得防御性较强。审稿人（尤其是通信方向的 TGRS 审稿人）可能会强烈反驳这个 framing。当前稿件写到“物理层手段已经用尽”，但批判性审稿人会指出：实验本身并没有穷尽这些手段；它假设的是 hard-decision BPSK 且没有 LDPC。**建议**：将 HARQ + ACM 讨论整体移到 Section 5（§5.2 已经存在且与 §1 有重叠），并在 §1 中用正面方式简要表述正交性：“RS-Token operates at the application layer above whatever physical-layer techniques are deployed”。这样可以避免被认为是在树立稻草人。

### M8. 比特预算计算需要显式说明 [Major-Minor]

“2,560 bits/img”和“10,240 bits/img”来自 $T \times \ell \times \log_2 K = 256 \times 1 \times 10 = 2{,}560$，以及再乘以 4 得到 10,240。稿件应明确写出该计算（§3.1 目前只有片段性说明），并说明该数值不含 $k$ 选择的信令开销。动机文档中提到 $k \in \{1,2,3,4\}$ 每张图需要 2 bit 元数据，这会把数字变为 2,562 和 10,242——影响可忽略，但应主动说明以防审稿人挑刺。**建议**：在 Section 3.1 / Section 4.4 中加一句或脚注：“Excluding the 2-bit metadata field for receiver-side $k$ selection, which is negligible relative to the per-image budget.”

---

## 3. 次要意见

### 写作与结构

1. **标题长度**：当前工作标题约 95 个字符，接近 IEEE GRSL 标题上限（通常偏好 ≤ 70 字符）。备选标题“RS-Token: A Layered Discrete Tokenizer for Task-Preserving UAV Image Transmission Over Degraded Channels”也偏长。可考虑更紧凑的形式：“RemoteCLIP-Distilled Hierarchical Tokenizer for Channel-Robust Remote Sensing Communication”（88 字符）或进一步缩短。

2. **摘要结构**：摘要是一个 280 词的单段，超过典型 GRSL/IGARSS 摘要长度（150-200 词）。建议将四项结果枚举压缩为两句话。

3. **§1.1 表述**：“无人机巡检、灾害应急、边缘智能传感器、CubeSat / SmallSat 等遥感部署场景下行链路在距离、遮挡、多普勒、干扰与业余频段等因素影响下表现出剧烈的 SNR 波动”——并列结构较重，“等因素”和“等遥感部署场景”叠加后略显累赘。英文版中建议拆成两句。

4. **§3.2 末句**：“实测见 §4.6”——前向引用没有问题，但括号式表达会打断行文。可改为：“Section 4.6 provides empirical evidence for this geometric mechanism.”

5. **§4.5**：“在视觉/遥感域首次以定量证据呈现”带有隐含 first claim，审稿人会严格审视。可软化为“this layered semantic / detail decomposition has not been quantitatively demonstrated in remote sensing prior to this work”，或直接省略。

6. **§5.1 机制解释**：“通用 CLIP 在 4 亿对自然图-文本上学到的特征空间，对遥感数据的几何‘展平’是 incidental 的副产物”——英文中不宜混用“incidental”。“an unintended consequence”或“a byproduct of cross-domain transfer”更顺。

7. **§5.3**：三个 limitation 被列出但没有编号。IEEE 格式中建议显式编号或使用小节标签以提升清晰度。

8. **结论第一段**：与摘要重复。建议压缩为一句贡献说明和一句影响说明。

### 表格与图

9. **Table 1 列名**：“best PSNR”中“best”含义不清（验证集最佳 epoch？最终 epoch？跨种子最佳？）。建议改为“PSNR (dB)”，并加脚注：“Best validation PSNR over 50 epochs.”

10. **Table 1 缺项**：Stage 1（single VQ）的 L0_bow / L0_emb / zq_pool 为“—”。若 single-VQ 本身有单层，其 L0_bow 探针是有定义的，也可作为一个有用锚点。补充该项约需 1-2 小时，可以关闭一个小但显眼的缺口。

11. **Tables 2-5 caption**：所有表都使用“AID test”描述，但只有 Table 1 明确写了“无信道”。建议统一：每个表题都明确注明信道条件。

12. **Figure 1（架构图）**：caption 很好，但应在 §3.2 和 §3.5 描述对应模块时显式引用该图。

13. **Figure 2（信道热力图）**：确认 AWGN 与 Rayleigh 两个 panel 使用相同绝对色标，以保证视觉比较公平。若每个 panel 自动缩放，读者会低估 Rayleigh 退化程度。

14. **Figure 4（教师消融）**：带 $\Delta$ 标注的成对柱状图有信息量，但如果 y 轴断点更清晰，或 SNR 排列体现运行严重程度，信息表达会更强。

15. **Figure 5（trade-off）**：“paper choice”标注有创意但不够标准；审稿人可能更偏好干净的 marker（如用圆圈圈出所选点并加脚注）。

### 方法与可复现性

16. **§4.1 数据集划分**：“8000/1000/1000 train/val/test”——需要说明这是否与 Gao et al. (TGRS 2022) 的划分完全一致，还是仅协议结构一致。最好引用 AID 原始协议（Xia et al. 2017），并说明该划分是标准划分还是自定义重采样。

17. **§4.2 训练配置**：缺少使用的 seed。按 `rvq_distill.yaml`，应加入“random seed = 42”。

18. **§3.4 信道仿真**：说明 BER 值是为分析效率通过闭式公式计算得到的，并且与完整 waveform-level Monte Carlo 仿真在一两个参考 SNR 上验证一致（或引用标准教材结论）。一句话即可。

19. **§3.5 L0_bow 标准化**：“经标准化后”——需要说明是哪种标准化：z-score、L1、L2？实现中 BoW 使用 `StandardScaler(with_mean=False)`（因为它是频率向量）；建议明确写出。

20. **码本利用率**：训练日志（`pre_experiments_log.md`）显示 Stage 3 四层码本利用率均 > 99%。这是一个非平凡的训练成功，应报告（一句放在 §4.2 或 §4.3）："All four codebook layers maintain > 99% utilization throughout training, with no observed codebook collapse."

### 引用与相关工作

21. **§2 缺少引用**：VQ-VAE（van den Oord et al. 2017）是整个方法的基础，应引用。SoundStream（Zeghidour et al. 2021）和 EnCodec（Défossez et al. 2022）在音频中引入 RVQ，并且是 SpeechTokenizer 的基础，至少应引用一个。若加入基线，还应引用 DeepJSCC（Bourtsoulatze et al. 2019）。

22. **§2.1 MOC-RVQ 描述**：“用 multi-head octonary codebook 加 4 阶 RVQ 让 codebook 大小天然对齐 64-QAM 星座点”——技术上准确但过于具体。英文中可改为“MOC-RVQ aligns multi-head codebook indices to digital constellation points (64-QAM)”以便更广读者理解。

23. **§2.2 SemCLIP 引用**：Gunduz（Imperial）与 BUPT 的双重引用——投稿前确认真实作者列表和 venue（arXiv 预印本还是会议论文）。

24. **参考文献列表**：需要完整 IEEE 格式 bibliography，包括 DOI / arXiv ID / 页码。目前 placeholder 列表对已发表论文也缺少卷号、期号、页码。

### 命名与符号

25. **“L0_bow” 与数学模式中的 “L0\_bow”**：LaTeX 下划线渲染会不自然。可考虑 $\mathrm{L0}_{\mathrm{bow}}$，或直接用 $h_0$ 表示直方图特征。

26. **“$\hat{\mathbf{I}}_\ell$” 与 “$\mathbf{I}_\ell$”**：§3.1 对接收端 indices 使用 hat，§3.2 对 L0 quantized embedding 使用无 hat。建议统一修饰规则（例如接收端量一律加 hat）。

27. **“deployment 信道”**：纯中文稿中应尽量避免可替换的英文插入。“部署场景信道”更统一。

---

## 4. 审稿人可能追问的尖锐问题（需提前准备 rebuttal）

以下是敏锐审稿人可能提出的问题。即便答案不写进正文，作者也应在补充材料或 rebuttal letter 中准备好。

### Q1. “在相同总比特预算下，它与 JPEG2000+LDPC 相比如何？”

**建议提前准备的回答**：在每张 256×256 图像 2,560 bits（约 0.039 bpp）时，JPEG2000 无法以可接受质量编码图像；通常至少需要 ≥ 0.05-0.1 bpp 才能得到有意义重建。在更高预算（10,240 bits ≈ 0.16 bpp）下，JPEG2000+LDPC 在 +5 dB AWGN 且 hard decision 条件下会位于 LDPC waterfall 以下，并发生灾难性失败。这正是 §1 提到的 cliff effect，但**稿件没有展示曲线**，这是问题。作者应运行该实验（几小时工作量，JPEG2000 + 5G NR LDPC 参考实现可用），或从已有文献给出带明确引用的参考数值。

### Q2. “如果 Stage 3 PSNR 是 25.88 dB，而 Stage 2 是 26.10 dB，这不是在主要图像保真指标上退化了吗？为什么 RVQ-only RVQ 是相关基线而不是竞争方法？”

**建议回答**：Stage 2 不是**任务保真**比较中的竞争方法；它是仅靠 RVQ（无语义监督）在重建质量上能达到的上限，而 0.22 dB 的代价换来了 +24.7 个百分点的任务准确率提升。现有文本基本能防守这一点，但审稿人仍可能追问。

### Q3. “为什么比较对象是无蒸馏，而不是无 RVQ 多层（single VQ）或 MoC-RVQ 风格 64-QAM 对齐码本设计？”

**建议回答**：Single VQ（Stage 1）PSNR 为 23.80 dB，明显更差，因此不是公平的重建竞争方法；但补充其 L0_bow 探针数值可关闭这个漏洞（见 Minor 10）。MOC-RVQ 比较需要 M1 中所述的基线实验。

### Q4. “为什么说 k 由‘信道 + 任务’选择，但实验只报告固定 k 结果？实际信道状态估计开销和信令协议是什么？”

**建议回答**：这是一个结构性批评。目前稿件展示的是*每个 k 的性能包络*，并据此论证接收端可选择工作点；它没有展示完整闭环自适应策略。对于短文这可以接受，但措辞应更谨慎：“RS-Token provides the rate-distortion-task envelope from which a receiver can select operating points based on link quality estimates”，而不是暗示已有完整闭环系统。

### Q5. “如果对 index bits 使用纠错码，低 SNR 下会怎样？”

**建议回答**：这是前瞻问题；§5.3 已将 index bit 上轻量 LDPC 作为未来工作。若能加入一个 back-of-envelope 估计会更有力（例如对每个 10-bit index 使用 (15,11) Hamming 或 BCH，会给每个 index 增加 5 bit 开销，以 50% rate reduction 换取错误率下降；这会使 RT 表现接近 AWGN-equivalent，但代价是比特预算上升）。

### Q6. “是否做过跨数据集评测（例如 AID 训练、RESISC45 测试）？”

**建议回答**：没有。这对 TGRS 投稿会是一个有力补充；对于 GRSL/IGARSS 短文格式，一句 future work 承认即可。

### Q7. “单种子运行下，4-7 个百分点 RemoteCLIP 优势是否统计显著？”

**建议回答**：这回到 M2。作者需要多种子运行来支撑该数字。

### Q8. “为什么只用 BPSK？现代系统大多使用高阶 QAM 或 OFDM。”

**建议回答**：BPSK 是最坏情况调制；框架可自然扩展到高阶调制，只需替换对应 BER 曲线。对于无人机 / 应急场景，SNR 低且链路可靠性优先，BPSK 是现实选择。建议在 §3.4 或 §5.3 中加一句提前回应。

---

## 5. 大修检查清单（按优先级排序）

以下清单将评审意见转化为可执行任务。标记 **[CRITICAL]** 的项目必须完成，否则难以获得合理评审结果；**[MAJOR]** 项显著增强论文说服力；**[POLISH]** 项主要影响呈现质量。

### 重投前必须完成

- [ ] **[CRITICAL]** 在匹配比特预算和 SNR sweep 下加入至少一个外部基线。优先级从高到低：(1) Gao et al. (TGRS 2022)——协议一致，最直接；(2) JPEG2000 + 5G NR LDPC，0.04 / 0.16 bpp；(3) DeepJSCC。(M1)
- [ ] **[CRITICAL]** 对 SNR ≤ +5 dB 的信道实验至少跑 5 个信道种子；在 Tables 2、4、5 中报告 mean ± std。(M2)
- [ ] **[CRITICAL]** Stage 3 使用 3 个模型种子重跑；在 Table 1 和至少一个信道 SNR 下报告 L0_bow 的 mean ± std。(M2)

### 重投前建议完成

- [ ] **[MAJOR]** 在 §3.1 中增加编码器 / 解码器结构描述（约 10 行 + 参数量）。(M3)
- [ ] **[MAJOR]** 在 §4.5 中用明确公式量化“L0 carries 95% of semantics”。(M4)
- [ ] **[MAJOR]** 要么在 Rayleigh 0 dB 下加入基线实验，要么将 graceful degradation 主张限定到 SNR ≥ +5 dB 区间。(M5)
- [ ] **[MAJOR]** 软化 §5.1 末句关于 RemoteCLIP “不可替代”的表述；加入几何分析（码本质心的类间 / 类内距离比）支撑机制解释。(M6)
- [ ] **[MAJOR]** 将 HARQ + ACM 反驳从 §1 移到 §5.2；在 §1 中正面表述与物理层方案的正交性。(M7)
- [ ] **[MAJOR]** 显式写出比特预算计算，并用脚注说明元数据开销。(M8)

### 投稿前润色

- [ ] **[POLISH]** 在 Table 1 中补充 Stage 1（single VQ）的 L0_bow 探针项（约 1-2 小时工作）。(Minor 10)
- [ ] **[POLISH]** 在 §4.2 或 §4.3 中加入码本利用率 >99% 的说明。(Minor 20)
- [ ] **[POLISH]** 统一表格 caption，明确每张表的信道条件。(Minor 11)
- [ ] **[POLISH]** 确认 Figure 2 的 AWGN/Rayleigh 两个 panel 共用同一 colorbar。(Minor 13)
- [ ] **[POLISH]** 将摘要从 280 词压缩到 ≤ 200 词。(Minor 2)
- [ ] **[POLISH]** 尽可能将标题压缩到 ≤ 70 字符。(Minor 1)
- [ ] **[POLISH]** 补充 VQ-VAE / SoundStream / EnCodec / DeepJSCC 引用。(Minor 21)
- [ ] **[POLISH]** 调整语言：“incidental”“deployment 信道”“首次”“唯一驱动因素”“不可替代” → 更克制的等价表达。(Minor 5, 6, 27; M6)
- [ ] **[POLISH]** 统一 $\hat{\mathbf{I}}$ 与 $\mathrm{L0}_{\mathrm{bow}}$ 的符号记法。(Minor 25, 26)
- [ ] **[POLISH]** 补全 IEEE 格式参考文献，包括 DOI。(Minor 24)
- [ ] **[POLISH]** 在 §4.2 中加入 seed = 42。(Minor 17)
- [ ] **[POLISH]** 说明 L0_bow 标准化方式（StandardScaler with_mean=False）。(Minor 19)

### 前瞻工作（适合完整 TGRS 版本，短文非必需）

- [ ] 跨数据集评测（AID → RESISC45）。(Q6)
- [ ] 高阶调制扩展（16-QAM / 64-QAM 与对应 BER 曲线）。(Q8)
- [ ] 带信道状态估计的闭环自适应 k 选择。(Q4)
- [ ] 多光谱 / 高光谱扩展。(§5.3)
- [ ] 在 index bits 上加入轻量纠错，以缓解 Rayleigh 0 dB 断崖。(Q5)

---

## 6. 推荐意见细节

**对于 IEEE GRSL（5 页短文）**：当前稿件处于**大修后有希望接收的边界状态**。CRITICAL 项（M1、M2）必须解决；MAJOR 项会显著增强说服力，但在 GRSL 标准下未必全部强制。若加入 **Gao et al. 2022 基线 + 多种子报告 + 编码器描述**，这会是一篇有竞争力的 GRSL 投稿。

**对于 IEEE TGRS（完整论文）**：当前稿件**尚未准备好**投 TGRS。TGRS 审稿人通常期待 2-3 个强基线、多种子统计、编码器 / 解码器结构消融，最好还有跨数据集泛化。预计还需要 2-3 周实验与写作。

**对于 IEEE IGARSS（4 页会议论文）**：若完成**多种子报告（M2）+ 至少一个外部基线（M1，例如 Gao 2022）**，则可以接受。IGARSS 审稿周期更快，是该工作的首发良好 venue，后续可扩展为 TGRS 期刊版本。

**当前总体建议**：**大修**。工作具有清晰新颖性（首个面向遥感语义通信的层级 RemoteCLIP 蒸馏 RVQ），核心结果显著且可复现，方法基础扎实。主要缺口是**比较性**而非概念性：缺少外部基线和多种子统计。这两个问题都可以通过有限额外实验补齐，不是根本缺陷。

---

## 7. 复现实验数字核对

审稿人已将稿件中的全部数值与补充记录（`rstoken/experiments/results.md` 和 `rstoken/experiments/pre_experiments_log.md`）交叉核对：

- Table 1 数值：已核对，与 results.md §1、§3 一致
- Table 2 数值：已核对，与 results.md §4（蒸馏 vs 无蒸馏 信道下对比）一致
- Table 3 数值：已核对，与 pre_experiments_log.md E7 记录一致
- Table 4 数值：已核对，与 pre_experiments_log.md E8 记录一致
- Table 5 数值：已核对，与 pre_experiments_log.md E11 记录一致
- 比特预算计算：已核对，$256 \times 4 \times 10 = 10240$ 成立
- BER 闭式公式：已核对，为标准教材结果，且与 E10 实现一致
- AID 类别数 = 30，train/val/test 划分 = 8000/1000/1000：已通过 E12 核对

未发现数值不一致。

---

## 8. 给作者的最后建议

这项工作**具有科学趣味且适合目标 venue**。其核心思想——将遥感预训练视觉语言模型蒸馏到残差向量量化器的第一层，以实现面向信道鲁棒任务导向通信——新颖、动机充分，并且在当前范围内已有有力证明。主要弱点是**比较不足**，而非概念不足：缺少外部基线和多种子统计。这两个缺口都是机械性可补的；补齐后，本文可从边界接收的短文转变为明确可接收的投稿。

实际策略建议：先以 **IGARSS 2026** 为目标，完成 M1 + M2 修订（约 1 周工作）实现首发；随后为 **TGRS** 扩展 M3-M8 与跨数据集评测，形成更强的期刊版本。这个双轨策略可以最大化首发速度，同时保留更完整期刊贡献的空间。
