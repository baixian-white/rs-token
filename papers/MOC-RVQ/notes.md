# MOC-RVQ 精读笔记

> 文件用途: 写论文 related work 和 Stage 5 baseline 对比的参考
> 创建日期: 2026-05-31
> 来源: WebFetch arXiv HTML 全文 (2401.01272v1) + GitHub README

---

## 1. 论文身份证

| 项 | 内容 |
|---|---|
| **完整标题** | MOC-RVQ: Multilevel Codebook-assisted Digital Generative Semantic Communication |
| **作者** | Yingbin Zhou¹, Guanying Chen¹, Shuguang Cui¹², Yaping Sun²¹, Hao Chen², Binhong Huang², Xiaodong Xu³², Ping Zhang³² |
| **单位** | ¹ **港中文(深圳)FNii + SSE** · ² **鹏城实验室** · ³ **北京邮电大学** |
| **发表** | **GLOBECOM 2024**(IEEE 全球通信会议)/ arXiv 2401.01272v1 (2024-01-02) |
| **代码** | github.com/Albert2X/moc_rvq —— **占位仓库, 暂无代码**(README 写"will be released soon",至 2026-05 仍未发布) |

---

## 2. 它解决什么问题

VQ-based 语义通信系统有**两个矛盾**:

### 矛盾 1 · Incompatible 问题(modulation 错位)
传统 VQ codebook 用大索引范围(优化重建保真),但数字星座调制(16-QAM, 64-QAM)只有少量离散态。"重新分组比特以适配调制" 会破坏"索引-星座点"之间能保留局部语义的显式映射关系。

### 矛盾 2 · Mismatch 问题(语义距离与索引距离脱节)
原文引用:
> "while the difference between index '1' and index '2' is 1, the distance between the underlying code vectors of index '1' and index '2' can be substantial."

→ 信道噪声会让 1 → 2 的微小比特错误,在 codebook 几何上跳到完全不同的语义簇。

---

## 3. 完整架构

### 整体流水线

```
图像 I → Encoder E → latent z ∈ ℝ^(h×w×n_q)
       → MOC-RVQ 量化 → 索引 s
       → bits → 64-QAM 调制 → AWGN 信道 → 解调 → 索引恢复
       → codebook 查表 R(·) → 噪声特征 ẑ_q
       → Noise Reduction Block (NRB, Swin-Transformer 残差堆叠) → ẑ_r
       → Feature Requantization (用无噪 codebook 再量化) → z̃_r
       → Decoder G → 重建图像 Î
```

### Encoder/Decoder 主干

- 沿用 Chen et al. [13] 的 autoencoder
- 下采样因子 **f=8**(latent 分辨率为输入的 1/64)

### MOC(Multi-head Octonary Codebook)—— 核心创新 1

每一阶量化的具体做法:

1. 残差 r_d ∈ ℝ^(h×w×n_q) 沿 channel 维度切成 **P=4 头**:r_{d,i} ∈ ℝ^(h×w×n_q/P)
2. **每头用 8 状态 codebook 量化**(N=8,精确对应 64-QAM 的 "8×8" 结构,即 I/Q 各 8 状态或每头 3 比特)
3. 每头量化结果拼回:e^(d) = concat(e_{d,1}, ..., e_{d,P})
4. **每阶组合状态总数 = N^P = 8^4 = 4096**

原文引用:
> "our proposed MOC dramatically expands the feature matching space, generating N^P discrete states through the combination of each head state."

→ **核心思路**:在调制端只需 8 状态(对接 64-QAM),但语义端能表达 4096 种组合,既适配调制又保住表达力。

### MOC-RVQ(在 MOC 上做残差堆叠)

- **D=4 层**(默认)
- 递推: r_0 = z; s_d = MOC^(d)(r_{d-1}); r_d = r_{d-1} − e^(d)_{s_d}
- 总输出: z_q = Σ_{d=1..D} e^(d)_{s_d}
- 索引张量: s ∈ {0,...,N−1}^(h·w·P·D) → 每位置 P×D = **16 个 8 状态索引 = 48 比特**

### Noise Reduction Block (NRB) —— 核心创新 2

- 接在接收端 codebook 查表后
- 由若干 **residual Swin Transformer layers** 堆成(沿用 [13])
- 输入:噪声扰动后的 ẑ_q
- 输出:净化特征 ẑ_r

### Feature Requantization —— 核心创新 3

净化后的 ẑ_r **再次过 MOC-RVQ**(用 Stage 1 训出来的无噪 codebook)。

原文引用:
> "MOC is trained in the absence of any noise, establishing itself as a high-quality semantic knowledge base to inject high quality prior for better feature restoration."

→ **意图**:无噪 codebook 当"语义 anchor",把信道损坏后的连续特征重新拉回 codebook 几何空间。

### Codebook Reordering (CR) —— 核心创新 4

**Algorithm 1**:
1. 从 codebook 均值开始,贪心最近邻遍历得到排序序列
2. 用 Gray code 序列 g(例如 3-bit Gray:{0,1,3,2,6,7,5,4})对排序序列做置换
3. 结果:Gray-相邻索引 ⇔ 语义相邻向量

→ **用途**:配合 64-QAM 的 Gray 映射,让"星座点相邻 = 语义向量相邻",信道翻 1 比特只跳到语义最近的 codeword。

---

## 4. 训练细节

### 两阶段训练

**Stage 1:无信道噪声预训练**

VQ loss(常规 stop-gradient + straight-through):
```
L'_VQ = ||Î − I||_1 + Σ_{d,h} ||sg[r_{d,h}] − e_{d,h}||²₂
```

**语义引导 loss(关键)**——用 VGG19(ϕ)提取参考特征,1×1 卷积对齐维度:
```
L_VQ = L'_VQ + γ ||CONV(z) − ϕ(I)||²₂,  γ = 0.1
```

加上 VQGAN-style 的 perceptual loss + adversarial loss(沿用 [Esser et al. 12])。

**Stage 2:NRB 微调(模拟信道下)**

```
L_NR = ||ψ(ẑ_r) − ψ(sg[z_q])||²₂ + α ||ẑ_r − sg[z_q]||²₂
α = 0.25
```
其中 ψ 是 Gram 矩阵(style loss 形式)。Stage 1 的 encoder/decoder/codebook 全部冻结,**只微调 NRB**。

### 数据集

- **训练**:DIV2K + Flickr2K + DIV8K + 10000 张 FFHQ 人脸
- **预处理**:无重叠 512×512 patch,过滤低纹理 patch;FFHQ 随机 resize [0.5, 1.0] 后 crop
- **测试**:**仅 DIV2K validation 中的 25 张图**,平均分辨率 2012×1407

### 训练超参

| 项 | 值 |
|---|---|
| Optimizer | Adam, β₁=0.9, β₂=0.99 |
| Learning rate | 1e-4(两阶段都用) |
| Crop | 256×256 |
| Batch size | 16 |
| 硬件 | **2× RTX 3090** |
| 训练时长 | **每阶段 ~3 天** |

---

## 5. 信道与评测

### 信道模型

| 项 | 内容 |
|---|---|
| 调制 | **64-QAM**(8×8 = 64 = N²,精确对应 P=4 头 × 8 状态) |
| 信道 | **AWGN**(无衰落)y = x + w, w ~ N(0, σ²) |
| SNR 扫描 | **−5 dB 到 30 dB** |
| **MOC-RVQ 自身** | **不用任何信道编码** |
| **Baselines** (BPG/JPEG) | **LDPC 648/1/2** |

→ **关键差异**:它把"自己不需要 LDPC"作为卖点,baselines 反而要靠 LDPC 才能传输。

### 评测指标

**只看像素 + 感知质量**,**没做下游任务保真**:
- PSNR
- SSIM
- LPIPS([Zhang et al. 2018])

→ 与本研究关键差异:**它根本不评测分类准确率等任务保真指标**。

### 实验结果(从论文文本提取的)

⚠ **重要事实**:论文正文 **没有任何具体的 PSNR/SSIM/LPIPS 数字**,所有数字都嵌在图 3/4/5 里(WebFetch 的文本提取无法读图)。

文本里只有这些可直接引用的事实:
- SNR 范围 -5 到 30 dB
- 量化层级 L1/L2/L3/L4 对应不同传输比特数,可动态调整
- 测试集 25 张图,平均分辨率 2012×1407
- 视觉描述:"even under relatively poor channel quality (SNR=5), our full model impressively reconstructs images with clear semantic meaning."
- 关于 "CBR=1/6" 的声明:**论文文本里没有这个数字**(之前 abstract WebFetch 提到,但这版 HTML 全文里没出现,需要看原 PDF 图才能确认)

---

## 6. 它声明的创新

| # | 创新点 |
|---|---|
| 1 | 识别 VQ-SemCom 的两个矛盾(incompatible + mismatch),提出两阶段框架 |
| 2 | **MOC**(多头八进制 codebook)对接数字星座调制 |
| 3 | NRB(Swin-Transformer 噪声去除)+ Feature Requantization |
| 4 | Gray-code-inspired codebook reordering |

附带卖点:
- 用 64-QAM AWGN,**不需要 LDPC**就打败 BPG/JPEG(它们要 LDPC)
- 多级传输 L1-L4 支持速率自适应

---

## 7. 它没说但你应该挑出来的局限

论文**没有显式 Limitations 章节**,但隐式问题:

| 局限 | 解读 |
|---|---|
| 测试集仅 25 张图 | 太小,且只是 DIV2K validation,**没有遥感数据** |
| FFHQ 在训练但不报告 | 训练数据多样但测试不全 |
| 仅 AWGN, 无衰落 | "To simplify channel modeling without sacrificing representativeness" —— 实际是回避 |
| Codebook reordering 是"heuristic" | 作者自承不是最优解 |
| 没有任务保真评测 | **这是你的论文最大差异化点** |
| **没有基础模型蒸馏** | VGG19 是"语义引导"(像素特征级),不是"地物身份" |

---

## 8. 与本研究的精确差异化

| 维度 | MOC-RVQ | 本研究 |
|---|---|---|
| **应用域** | 自然图像 (DIV2K + Flickr2K + DIV8K + FFHQ) | **遥感图像 (AID 30 类)** |
| **训练教师** | VGG19 像素感知特征(γ=0.1)| **RemoteCLIP 语义 embedding(0.5)** |
| **教师作用** | 像素级感知监督,**辅助重建** | **语义对齐,显式让 L0 codebook 承载地物身份** |
| **量化器** | MOC-RVQ(每阶 P=4 头 × N=8 状态 = 4096 组合,D=4 阶) | 标准 RVQ(每阶 codebook 1024,D=4 阶) |
| **调制对接** | 显式对接 64-QAM(8×8)| BPSK(1 比特一符号),不刻意对接调制 |
| **信道编码** | 不用 LDPC | 不用 LDPC(同) |
| **接收端去噪** | NRB + Feature Requantization 两步 | 无 NRB,直接接 decoder |
| **评测维度** | PSNR + SSIM + LPIPS,**仅重建保真** | PSNR + LPIPS + **L0 线性分类 + 信道下任务保真**(论点 A 验证) |
| **第一层语义** | **没有** L0 特殊地位,L1-L4 仅精度递进 | **L0 蒸馏后承载语义身份**,信道差时只传 L0 仍可用 |
| **降级传输验证** | 演示 L1-L4 传输比特数变化对 PSNR 影响 | **k=1 vs k=4 的下游分类准确率对比**(任务保真意义) |

---

## 9. 对你论文 related work 段落的写法建议

### 一段直接对标 MOC-RVQ 的段落(中文草稿)

> Among prior VQ-based digital semantic communication systems, MOC-RVQ [Zhou et al., GLOBECOM 2024] is the closest in engineering framework to ours. They adopt a multilevel residual VQ structure paired with a multi-head octonary codebook tailored for 64-QAM modulation, and demonstrate effective reconstruction over AWGN channels without explicit LDPC. **However, MOC-RVQ targets natural image transmission (DIV2K-class) and uses VGG19 perceptual features as semantic guidance — its layer hierarchy reflects only progressive precision refinement, with no special semantic role assigned to the first layer.** In contrast, **our work explicitly distills RemoteCLIP semantics into the first-layer codebook of an RVQ tokenizer for remote-sensing imagery, enabling layer-1-only transmission to retain task-level fidelity (e.g., classification accuracy) under bad-channel conditions.** This addresses two gaps left open by MOC-RVQ: domain (natural → remote sensing) and hierarchical mechanism (precision refinement → semantic-vs-detail decomposition).

### 写作要点

- **正面引用,不诋毁**——它是正经的 GLOBECOM 工作
- **明确两个差异**:应用域 + 层级机制(蒸馏)
- **强调你的延伸**:任务保真评测是它没做的,这是你的论点 A 闭环

---

## 10. Stage 5 Baseline 对比的可行性

### 直接复刻它?

**目前不可行**:GitHub 仓库是空的,作者 1.5 年都没释出代码。要复刻必须自己实现:
- MOC-RVQ 量化器(4 头 × 8 状态)
- NRB(Swin-Transformer 残差块)
- Codebook Reordering 算法
- VGG19 语义引导 loss
- 64-QAM AWGN 信道仿真

### 工程量估计

如果你 Stage 5 要做这个 baseline:
- **量化器实现** ~2 天(基于 vector_quantize_pytorch 改)
- **NRB** ~1 天(套用现成 Swin Transformer)
- **Codebook reordering** ~半天
- **64-QAM 信道**(替换你现有 BPSK)~半天
- **训练 + 调参** ~3-5 天(因为要复现 PSNR 数字)
- **接 AID 数据集**(他们没用过遥感数据)~半天

→ **总共 ~1-2 周**

### 折中方案

| 选项 | 做法 |
|---|---|
| **A. 完整复刻** | 自己实现 MOC-RVQ + 训练 + AID 上跑,公平对比 | 1-2 周 |
| **B. 数字层对比** | 论文里**只引用**它的 PSNR vs 你的 PSNR,不复刻——但要明确"不同数据集,不可直接对比" |
| **C. 替代 baseline** | 用 ReVQom(若代码开源)或 DeepJSCC 做对比,绕开 MOC-RVQ |

→ **我建议 B**:论文里把它列在 prior work 章节而不是实验章节。Stage 5 主表用 JPEG2000+LDPC 和 DeepJSCC 这种工程上能站得住的 baseline。
→ 如果导师/审稿人坚持要 MOC-RVQ 数字对比,再做 A。

---

## 11. 一句话回顾

**MOC-RVQ 是 RVQ-SemCom 路线最完整的工程级工作,但它的"语义"停留在 VGG19 像素特征级、应用在自然图像、评测只看重建保真。本研究在这条路线上往前推三步:换成遥感域、把 VGG19 像素引导换成 RemoteCLIP 语义蒸馏、并用任务保真评测验证降级传输的实际可用性。**

---

## 附:引用 BibTeX(草稿,需 verify)

```bibtex
@inproceedings{zhou2024mocrvq,
  title={MOC-RVQ: Multilevel Codebook-assisted Digital Generative Semantic Communication},
  author={Zhou, Yingbin and Chen, Guanying and Cui, Shuguang and Sun, Yaping and Chen, Hao and Huang, Binhong and Xu, Xiaodong and Zhang, Ping},
  booktitle={IEEE Global Communications Conference (GLOBECOM)},
  year={2024},
  note={arXiv:2401.01272}
}
```
