# RS-Token：面向信道鲁棒遥感通信的层次化 RemoteCLIP 蒸馏 Token (v0.5)

作者：Baohui Zhang（安徽大学 XXX 学院，合肥 230601；y125211005@ahu.edu.cn）

期刊：拟投 IEEE Geoscience and Remote Sensing Letters

源文件：`paper_draft/latex/rs_token_v0.5.tex`（10 页 PDF, 1.75 MB）

> 本中文版与 v0.5.tex 严格逐段对应，公式、表格、图引用全部保留 LaTeX 原始符号以便交叉对照。

---

## 摘要

遥感图像下行链路常面临带宽受限、低信噪比和衰落信道，而接收端可能需要的是场景标签、低码率任务图像或高保真重建。本文提出 **RS-Token**，把四阶段残差向量量化器（RVQ）与"仅在第一层蒸馏 RemoteCLIP"的设计组合在一起：L0 承载地物语义，L1–L3 提供重建细化；接收端选择前 $k$ 层作为前缀传输（在 AID 配置下为 2,560–10,240 bits/image），任务路径与重建路径在评估中严格分离。在三个模型种子上，蒸馏将无信道下 L0 准确率从 $58.23\pm1.57\%$ 提升到 $83.33\pm0.81\%$，AWGN $+5$ dB 下保持 $82.57\pm0.31\%$，Rayleigh $+10$ dB 下保持 $78.80\pm0.72\%$；重建从 $k=1$ 的 $22.98\pm0.09$ dB / $71.4\pm1.3\%$ 提升到 $k=4$ 的 $25.92\pm0.08$ dB / $86.8\pm0.4\%$。一项蒸馏位置反事实实验把"L0-only"确立为实测最优：把蒸馏施加到全部层会稀释 L0，把蒸馏移到 L1 会让 $h_0$ 崩塌 48 pp。RemoteCLIP 相对于 OpenAI CLIP 仅有 1.4–6.0 pp 的 $h_0$ 优势，主要体现在衰落条件下。L0 tokenizer 跨数据集 zero-shot 迁移到 NWPU-RESISC45 时蒸馏增益基本保留（AID 上 25.1 pp，NWPU 上 20.9 pp）。在严格相同传输比特预算下，RS-Token 取得 $0\%$ 的解码失败率，而 rate-1/2 LDPC 保护的 codec pipeline（JPEG2000、WebP）在 60–100\% 的图像上失败。部署侧使用 10.87 M 参数、单图 GPU 推理 8.59 ms；RemoteCLIP teacher（约 150 M 参数）仅训练时使用。

**关键词：** 语义通信，遥感，残差向量量化，RemoteCLIP，任务导向通信，层次化 token，信道鲁棒性。

---

## 1. 引言

遥感图像通信越来越多地用于低空 UAV 巡检和应急场景筛查 \cite{gao2022task}。这类场景下，下行链路可能因测距变化、遮挡、多径衰落以及临时性的通信约束而不稳定。同时，接收端并不总是需要立即获得完整保真图像。某些实时告警任务只需可靠的场景级判断；现场研判可能需要可大致解读的图像；后续测绘、归档或人工复核才需要更高质量的重建。因此，遥感通信受益于**层次化表示**：在低码率下保留任务相关语义，在更多比特可传输时逐步恢复图像细节。

传统图像通信通常采用"先压缩、再传输"的流程：发送端将图像编码为压缩码流，接收端在恢复码流后再解码图像。这一范式工程成熟，但与上述层次化需求并不完全匹配。一方面，WebP 或 JPEG2000 这样的压缩码流对未纠正的 bit error 敏感，可能引发严重失真甚至解码失败。另一方面，传统压缩主要为视觉重建优化，并不显式保证最先传输的比特优先保留遥感场景语义。信道编码可以降低误码率，但本身不会改变源表示的语义排序。也有以联合源信道编码（JSCC）方式直接学习信道适应表示的工作 \cite{deepjscc}，但其输出通常是连续表示，不天然支持本文需要的离散、可分层 token 结构。

一种自然替代是直接传输神经 tokenizer 的离散 index，例如 VQ \cite{vqvae} 或 RVQ \cite{lee2022residual} index。相较于传统压缩文件，离散 index 更易于与数字信道、重传与差错保护机制结合，且 RVQ 天然支持按层传输 \cite{moc-rvq}。然而，标准 RVQ tokenizer 通常由重建损失驱动，其第一层 codebook 主要是对连续 latent 的粗粒度逼近，并不保证对遥感场景类别具有判别性。因此，朴素 RVQ index 传输虽然给出层次化码率结构，却不一定提供低码率下的任务语义保真。

**RS-Token** 在这个层次化 RVQ 表示中引入 RemoteCLIP \cite{liu2024remoteclip} 蒸馏，明确改变第一层 token 的角色。蒸馏损失只施加到 L0 派生特征上，使其对齐遥感视觉语言嵌入空间。L1–L3 仍然编码用于重建的残差信息。同一个 tokenizer 因此支持前缀式传输：接收端按信道与任务需求选择前 $k$ 层 RVQ index。比特开销由 token 网格、codebook size 和传输层数共同决定。在本文 AID 配置下，$k=1,2,3,4$ 分别对应 2,560、5,120、7,680 和 10,240 bits/image。

本文的证据链组织如下。**第一**，将 RS-Token 与"无蒸馏 plain RVQ index"内部基线 `rvq_baseline` 比较，检验蒸馏是否使 L0 更具语义。**第二**，用一项蒸馏位置反事实实验，检验 L0 特化是由蒸馏位置造成的，还是第一残差层的某种架构偶然。**第三**，重建路径结果评估增大 $k$ 是否提升 PSNR、LPIPS 和重建图分类器准确率。在此之外，本文还在严格相同传输比特预算下与 rate-1/2 systematic sparse LDPC 保护的 WebP/JPEG2000 pipeline 做 stress 比较，并报告 L0 tokenizer 到 NWPU-RESISC45 的 zero-shot 迁移结果。

**贡献：**

- 提出一种 RemoteCLIP 引导的 L0 语义分配机制，在四层 RVQ tokenizer 中只蒸馏第一层，使其承载遥感场景语义，后续层留作渐进式重建之用。
- 形式化定义遥感图像的 prefix 式 RVQ index 传输方案。具体比特开销与配置相关；在本文 AID 配置下，四个前缀分别为 2,560、5,120、7,680 和 10,240 bits/image。
- 建立"任务路径 / 重建路径分离"的评估协议：$h_0$/L0 bag-of-words 准确率用于 $k=1$ 任务路径声明；PSNR、LPIPS 加上一个 clean AID ResNet34 评估器用于重建路径声明。
- 在 AID 上（含 AWGN 与 Rayleigh 信道）证明：相对于 plain RVQ index 传输，RemoteCLIP 蒸馏显著改善 L0 任务准确率；当信道允许时，更多 RVQ 层进一步改善重建。

---

## 2. 相关工作

### 2.1 任务导向遥感通信

任务导向语义通信优化的是面向下游决策的传输，而不是单纯的像素保真。早期 Deep JSCC 工作 \cite{deepjscc} 表明端到端学习的图像编码可在含噪信道下显著优于"压缩 + 经典信道码"的串联流程。遥感应用是天然的目标场景，因为场景分类、目标监控、应急筛查在与类相关的语义被保留时通常容忍部分视觉信息缺失。已有 UAV 或遥感语义通信工作表明，在比传统图像重建系统更低的码率下任务准确率仍可保持较高 \cite{gao2022task}。最近也有把多模态基础模型作为语义先验、在零样本设置下做语义通信的尝试 \cite{semclip}，以及把多层 codebook 与生成式语义通信耦合的数字方案 \cite{moc-rvq}。RS-Token 的不同之处在于保持**离散、层次化**的表示——其第一层有意对齐到一个遥感基础模型，后续层留作重建细节。

### 2.2 离散视觉 Token 与残差量化

向量量化自编码器（VQ-VAE）将离散 latent 图像 token 引入为一种紧凑神经表示 \cite{vqvae}。残差向量量化（RVQ）扩展了这一思想，用多层 codebook 表示同一个 latent，后续 codebook 编码前一层留下的残差 \cite{lee2022residual}。RVQ 在 SoundStream、EnCodec 等神经音频 codec 中被广泛使用 \cite{soundstream,encodec}，逐层 codebook 自然定义了多个工作点；类似的"层次化 token"原则在语音侧也由 SpeechTokenizer 扩展为语义层与声学层显式分离的设计 \cite{zhang2024speechtokenizer}。在视觉侧，VILA-U 等近期工作探索了把同一离散 token 同时用于理解与生成的统一接口 \cite{vilau}，而通信侧也已有把 RVQ 用作多智能体感知传输瓶颈的尝试 \cite{revqom}。RS-Token 把这种"层次化 token"原则迁移到遥感通信，并在第一层加入 teacher 引导的语义约束。

### 2.3 基础模型蒸馏

视觉语言基础模型提供了语义结构良好的嵌入，可用于监督紧凑表示。在通用视觉侧，BEiT v2 等工作已经表明把基础模型的视觉特征（如 CLIP 视觉编码器输出）蒸馏到 VQ tokenizer 上能让离散 token 更具语义判别性 \cite{beitv2}。RemoteCLIP 针对遥感图文对齐训练，给出了面向地物语义的领域特定嵌入空间 \cite{liu2024remoteclip}。RS-Token 不是把通用视觉表示蒸馏到所有 latent 层，而是只蒸馏到 L0 派生特征上。这种"局部化蒸馏"很关键，因为通信目标本身就是层次化的：第一层应支持鲁棒任务推理，余下层应留作重建细化。

### 2.4 传统与编码基线

WebP 与 JPEG2000 这样的传统图像 codec 是清洁图像存储的强工程基线。但在含噪数字信道下，无保护的压缩码流可能因解码器期望"语法合法的文件"而灾难性失效。信道编码可以缓解这种失效模式 \cite{proakis}，但具体编码至关重要；联合源信道编码也是一条不同于"先压缩再纠错"的替代路径 \cite{deepjscc}。本文中：WebP/JPEG2000 结果只被描述为**无保护压缩码流经 BPSK 信道**的结果；LDPC 实验则单独描述为 rate-1/2 systematic sparse LDPC + min-sum BP 解码——它**不是 5G NR LDPC**，也不被用来对标准蜂窝码进行比较。

---

## 3. 方法

> **图 1**（`fig_method_aech_image2pro.png`）：RS-Token 总体框架。发射端将一幅遥感图像编码为四层 RVQ index。部署阶段只传输前缀的 index；RemoteCLIP teacher 监督和蒸馏头**只在训练时使用**。接收端可以基于 L0 派生特征做任务推理，也可以解码所收到的前缀做图像重建。

### 3.1 问题设定

设 $x \in \mathbb{R}^{H \times W \times 3}$ 为一幅遥感图像、$y$ 为其场景类别。发射端将 $x$ 映射为离散 index 的层次结构 $z_{1:K}=\{z_1,\ldots,z_K\}$，本文取 $K=4$。每层共享相同的空间 token 网格，在本实现中每层贡献 2,560 bits/image。接收端可按信道质量与应用需求请求前缀 $z_{1:k}$：

$$B(k) = 2560 k, \quad k \in \{1,2,3,4\},$$

即 2,560、5,120、7,680、10,240 bits/image。$k=1$ 是任务导向低码率模式；更大的 $k$ 是重建细化模式。

### 3.2 层次化 RVQ 编码–解码器

编码器 $E_\theta$ 把图像映射到连续 latent 张量 $u = E_\theta(x)$。残差向量量化器随后用多个 codebook 嵌入之和近似 $u$，其残差更新规则与 \cite{lee2022residual,soundstream} 一致：

$$
\begin{aligned}
r_0 &= u, \\
z_\ell &= \arg\min_j \|r_{\ell-1} - c_{\ell,j}\|_2^2, \\
q_\ell &= c_{\ell, z_\ell}, \\
r_\ell &= r_{\ell-1} - q_\ell,
\end{aligned}
$$

其中 $\ell=1,\ldots,4$。给定前 $k$ 层前缀，解码器重建：

$$\hat{x}_k = D_\phi\!\left(\sum_{\ell=1}^k q_\ell\right).$$

正是这种"前缀性质"使得**同一个训练好的模型可以支持多种传输码率**。

### 3.3 L0 上的 RemoteCLIP 蒸馏

核心设计选择是**只**把第一层量化与遥感语义 teacher 对齐。设 $f_T(x)$ 是归一化后的 RemoteCLIP 图像嵌入 \cite{liu2024remoteclip}，$g_\psi(q_1)$ 是从 L0 量化表示到 teacher 特征维度的投影，蒸馏项为：

$$\mathcal{L}_{\rm distill} = 1 - \frac{g_\psi(q_1)^\top f_T(x)}{\|g_\psi(q_1)\|_2 \, \|f_T(x)\|_2}.$$

总训练目标结合重建、感知/codebook 正则、L0 蒸馏：

$$\mathcal{L} = \mathcal{L}_{\rm rec} + \mathcal{L}_{\rm vq} + \lambda \mathcal{L}_{\rm distill}.$$

主模型取 $\lambda=0.5$，源自先前的 trade-off 实验，在"无信道下语义准确率"与"信道鲁棒性"之间平衡。无蒸馏内部基线令 $\lambda=0$，但保留完全相同的架构和训练协议。

### 3.4 信道模型与评估指标

**信道模型。** 设前 $k$ 层 RVQ 的 index 比特序列为 $\mathbf{b} \in \{0,1\}^{B(k)}$。BPSK 调制把每个比特映射到一个实数符号 $s_i = 1 - 2 b_i \in \{+1, -1\}$。接收端在第 $i$ 个符号时刻收到

$$y_i = h_i\, s_i + n_i, \qquad n_i \sim \mathcal{CN}(0,\sigma^2),$$

其中 AWGN 信道下 $h_i \equiv 1$，Rayleigh 衰落信道下 $h_i \sim \mathcal{CN}(0,1)$ 在每个符号时刻独立采样 \cite{proakis}。信噪比定义为 $\mathrm{SNR} = 1/\sigma^2$（dB 形式 $10\log_{10}(1/\sigma^2)$）。接收端做相干硬判决恢复比特：

$$\hat{b}_i = \tfrac{1}{2}\bigl(1 - \mathrm{sgn}\!\bigl(\mathrm{Re}\{h_i^{*} y_i\}\bigr)\bigr),$$

随后按 codebook size 重组成 indices $\hat{z}_{1:k}$。本节信道实验**不**插入纠错码；§4.5 在 RS-Token 与传统 codec 之间公平地附加同一 rate-1/2 LDPC 保护，独立报告。

**任务路径评估。** 设第 $\ell$ 层 codebook size 为 $C_\ell$，第 $\ell$ 层在第 $t$ 个空间 token 位置接收到的 index 为 $\hat{z}_\ell^{(t)} \in \{0, \ldots, C_\ell - 1\}$，token 网格大小为 $T = 16 \times 16 = 256$。L0 bag-of-words 特征是对 L0 codeword 使用频次的归一化直方图：

$$
h_{0,j} \;=\; \frac{1}{T} \sum_{t=1}^{T} \mathbf{1}\!\bigl[\hat{z}_1^{(t)} = j\bigr], \qquad j = 0, \ldots, C_1 - 1,
$$

得到 $\mathbf{h}_0 \in \mathbb{R}^{C_1}$（本文 $C_1 = 1024$）。在 $\mathbf{h}_0$ 上训练线性分类器 $W$ 做 30 类 AID 场景分类，$\hat{y} = \arg\max_{c} (W \mathbf{h}_0)_c$；任务路径准确率即 $\mathbb{E}[\mathbf{1}[\hat{y} = y]]$。该指标**只**针对 $k=1$，因为它衡量的是 L0 的语义内容，而非更高层次产生的重建图像视觉质量。

**重建路径评估。** 给定接收 indices $\hat{z}_{1:k}$，解码器输出重建 $\hat{x}_k = D_\phi\!\bigl(\sum_{\ell=1}^{k} c_{\ell, \hat{z}_\ell}\bigr)$。PSNR 在 $[0,1]$ 归一化像素域定义为

$$\mathrm{PSNR}(x, \hat{x}_k) \;=\; 10 \log_{10}\!\frac{1}{\mathrm{MSE}(x, \hat{x}_k)}, \qquad \mathrm{MSE}(x, \hat{x}_k) \;=\; \frac{1}{3HW}\,\|x - \hat{x}_k\|_2^2.$$

LPIPS 取 \cite{zhang2018lpips} 的 AlexNet 主干默认实现。重建图分类准确率使用一个**clean AID ResNet34 分类器** $f_{\mathrm{cls}}$：

$$\mathrm{Acc}_{\mathrm{recon}}(k) \;=\; \mathbb{E}\bigl[\mathbf{1}\!\bigl[\arg\max f_{\mathrm{cls}}(\hat{x}_k) = y\bigr]\bigr],$$

其中 $f_{\mathrm{cls}}$ 在干净 AID 图像上独立训练，达到 96.10\% test top-1 准确率与 95.94\% macro-F1，是"重建是否保留场景识别线索"的稳定代理。所有 $k=2\ldots 4$ 的声明只由这些**重建路径指标**支持，不由 $\mathbf{h}_0$ 支持。

---

## 4. 实验

本节按以下五个问题组织：

1. RemoteCLIP 蒸馏是否使 L0 更具语义？
2. L0 特化由蒸馏位置造成，还是第一残差层的架构偶然？
3. 传输更多 RVQ 层是否改善重建？
4. 在严格相同传输比特预算下，RS-Token 与 codec+LDPC pipeline 的失败模式有何差异？
5. AID 上训练的 L0 tokenizer 能否 zero-shot 迁移到另一遥感场景 benchmark？

每个子节遵循"问题–设置–结果–结论"结构。无保护 WebP/JPEG2000 在 BPSK 含噪信道下的失败模式作为外部 stress test 一并报告。

为避免指标混用，评估分为：

- **任务路径**：只用 $h_0$/L0 bag-of-words 准确率，仅支持 $k=1$ 低码率语义声明。
- **分层 probe**：用累积 codeword 嵌入分析 L0 与 L1–L3 的语义分工。
- **重建路径**：用 PSNR、LPIPS、clean AID ResNet34 重建图像分类器，支持 $k=1\ldots 4$ 重建声明。

也就是说：$h_0$/L0 bag-of-words 与 layered probe 不被用作多层重建质量证据；PSNR/LPIPS 单独不被用作"L0 是否语义"的证据。

### 4.1 实验设置

所有实验使用 30 类 AID 航空场景数据集 \cite{xia2017aid}，采用**固定的、按类分层的 train/val/test split**：8,000 / 1,000 / 1,000（80%/10%/10%）；30 类全部出现在每个 split 中，per-class 样本数：train 集每类 176–336 张（均值 266.7），val 与 test 集每类 22–42 张（均值 33.3）。

训练用 RandomResizedCrop(256)，评估用 Resize + CenterCrop(256)；图像归一化与训练脚本一致。

RS-Token 的四层 RVQ tokenizer 接收 $256\times 256$ RGB 输入；latent token 网格 $16\times 16$，latent 维度 256，codebook size 1024。每层包含 256 个离散符号，每个用 $\log_2(1024)=10$ 比特表示，因此每层 payload 为 2,560 bits/image，$k=1,2,3,4$ 分别对应 2,560、5,120、7,680、10,240 bits/image。

内部 tokenizer 比较使用两个模型族：`rvq_baseline`（无 RemoteCLIP 蒸馏的四层 RVQ tokenizer，仅用重建 + RVQ 量化损失训练）与 `rvq_distill`（架构与传输协议完全相同，但加入 §3.3 的 L0 RemoteCLIP 蒸馏项）。两者均使用 batch size 16、AdamW、学习率 $10^{-4}$、50 epochs、梯度裁剪、bf16 自动混合精度；L1、LPIPS \cite{zhang2018lpips}、RVQ、distillation 的损失权重分别为 1、0.1、1、0.5；LPIPS 用 AlexNet 主干。Teacher 是冻结的 RemoteCLIP-ViT-B/32 \cite{liu2024remoteclip}，蒸馏投影头 $g_\psi$ 隐藏维度为 512。

任务路径上，从收到的 L0 indices 构造 1024 维 bag-of-words 特征 $h_0$ 并训练线性分类器做 AID 30 类场景分类。重建路径上，从前 $k$ 层 RVQ 解码出重建图像，并报告 PSNR、LPIPS、clean AID ResNet34 在重建图上的准确率。除非另有说明，主任务路径结果报告 seed 41/42/43 三个模型种子的 mean ± std；重建路径 sweep 使用实验日志报告的主配置。信道实验把 RVQ index 比特转 BPSK 符号，经 AWGN 或 Rayleigh 信道传输 \cite{proakis}，接收端硬判决恢复 indices。

---

### 4.2 RemoteCLIP 蒸馏是否使 L0 更具语义？

**问题。** RS-Token 核心假设是：当只传输最低码率层时，L0 仍应可用于场景级判断。Plain RVQ tokenizer 也产生 L0 indices \cite{lee2022residual}，但因为它由重建损失驱动，其第一层 codebook 不一定对 AID 场景类别具有判别性。本实验在同一 RVQ-index 传输框架下检验"RemoteCLIP \cite{liu2024remoteclip} 蒸馏是否显著提升 L0 任务保真度"。

**设置。** 比较 `rvq_baseline` 与 `rvq_distill`。两者唯一不同是 L0 RemoteCLIP 蒸馏项；tokenizer 架构与 index 传输保持一致。任务路径只用收到的 L0 indices 构造 $h_0$/L0 bag-of-words 特征，再训一个线性 probe 做 30 类 AID 场景分类。该指标只对应 $k=1$ 低码率任务路径，不用于 $k=2\ldots 4$ 重建声明。表 1 报告三个模型种子的 mean ± std，表中 best PSNR/LPIPS 仅作为训练质量参考，不作为 L0 语义的主证据。

**表 1**：三模型种子下任务路径的稳定性。准确率为 $h_0$/L0 bag-of-words top-1 准确率（百分比），仅用于 $k=1$ 任务路径声明。

| 指标 | 无蒸馏 | RemoteCLIP 蒸馏 |
|---|---|---|
| Best PSNR (dB) | $26.10\pm0.02$ | $25.89\pm0.07$ |
| Best LPIPS | $0.172\pm0.001$ | $0.175\pm0.002$ |
| 无信道 $h_0$ 准确率 | $58.23\pm1.57$ | **$83.33\pm0.81$** |
| AWGN +5 dB | $55.73\pm0.72$ | **$82.57\pm0.31$** |
| AWGN +10 dB | $58.20\pm1.59$ | **$83.37\pm0.76$** |
| Rayleigh +5 dB | $28.67\pm0.65$ | **$58.57\pm0.47$** |
| Rayleigh +10 dB | $48.63\pm1.31$ | **$78.80\pm0.72$** |

> **图 2**（`fig_exp_l0_task_robustness_v3.png`）：L0 任务路径在无信道、AWGN、Rayleigh 下的鲁棒性。该图可视化与表 1 相同的 $h_0$/L0 bag-of-words 任务路径证据，仅用于 $k=1$ 任务路径声明。

**结果。** RemoteCLIP 蒸馏给出大且稳定的 L0 任务增益。无信道下 $h_0$ 从 $58.23\pm1.57\%$ 提升到 $83.33\pm0.81\%$，**+25.10 pp**。AWGN +5 / +10 dB 下蒸馏模型分别保持 $82.57\pm0.31\%$ 与 $83.37\pm0.76\%$，接近无信道表现。Rayleigh 衰落更困难，但蒸馏模型仍以 **+29.90 pp** 和 **+30.17 pp** 的优势分别在 +5/+10 dB 下击败基线。

 `rvq_baseline` 与 `rvq_distill` **共享 init seed = 42**，因此该 gap 不可归因于 weight-init 噪声。

**结论。** 表 1 支持"RemoteCLIP 蒸馏显著提升 L0 indices 的场景级判别能力"，使 $k=1$ 低码率任务路径更可用。该结论仅涉及 L0 任务保真度；它不证明所有 RVQ 层都变得有语义，也不证明 $k=2\ldots 4$ 的重建质量。

---

### 4.3 L0 语义来自哪里？蒸馏位置反事实

**问题。** 上一小节表明 `rvq_distill` 的 L0 indices 具有场景判别性。一个自然的进一步问题是：这种"L0 特化"是否**真的由 RemoteCLIP 监督的位置造成**，还是无论如何都会自然出现在第一残差层？仅在主模型上拟合的 layered linear probe 是暗示性的，但**不能单独区分这两种解释**：probe 是在 L0 被监督的同一 ckpt 的累积 codeword 上拟合的，因此无法排除"L0 只是恰好被先 probe"的循环解读。我们因此**用一项蒸馏位置反事实实验**直接回答这个问题。

**设置。** 训练三个**仅在 RemoteCLIP 蒸馏施加层上不同**的对齐 RVQ tokenizer：

- **L0（主模型）**：发表的 `rvq_distill` 主模型，RemoteCLIP teacher \cite{liu2024remoteclip} 对齐 L0 直通量化特征。
- **All layers**：将同一对齐损失施加到 RVQ 总和表示 $\mathbf{z}_q = \sum_{\ell=0}^{3}\mathbf{z}_q^{(\ell)}$。
- **L1**：将对齐损失移到 L1 直通量化特征。

三个 ckpt 共享 seed=42、相同 RemoteCLIP teacher、相同 50 epoch 训练、相同重建与 commitment 损失权重，以及与 \cite{lee2022residual} 残差顺序一致的相同 encoder/decoder/quantizer 架构。表 2 报告每个 ckpt 在 5 个信道条件下的 $h_0$/L0 bag-of-words 准确率、相同 5 个条件下的 $k=4$ PSNR 与重建图分类准确率，以及 4 条 L0 → L0+L1+L2+L3 累积线性 probe。**Probe 行被作为四个指标族之一保留**：probe 在三个反事实 ckpt 上一并评估，与"蒸馏位置干预"互为印证，避免单 ckpt probe 的循环解读。

**表 2**：蒸馏位置反事实实验。三个对齐 RVQ tokenizer 仅在"对齐损失施加位置"上不同：L0（主 `rvq_distill`）、整个 RVQ 表示 $\mathbf{z}_q$（"all layers"）、或仅 L1。同 seed=42、同 teacher、同 50 epoch 训练、同损失权重；每信道条件 $n=1000$ AID test 图像。Probe 行为在累积量化特征上拟合的线性分类器。

| 设置 | L0 only (主) | All layers | L1 only |
|---|---|---|---|
| **$h_0$ 准确率（$k=1$, %）** | | | |
| 无信道 | 82.6 | 81.2 | 34.2 |
| AWGN +5 dB | 82.1 | 79.8 | 32.1 |
| AWGN +10 dB | 82.6 | 81.2 | 34.2 |
| Rayleigh +5 dB | 58.6 | 49.3 | 10.4 |
| Rayleigh +10 dB | 77.7 | 74.2 | 24.0 |
| **$k=4$ PSNR (dB)** | | | |
| 无信道 | 25.92 | 25.64 | 22.32 |
| AWGN +5 dB | 23.94 | 23.67 | 14.36 |
| AWGN +10 dB | 25.92 | 25.64 | 22.29 |
| Rayleigh +5 dB | 16.93 | 16.57 | 12.45 |
| Rayleigh +10 dB | 20.63 | 20.28 | 13.08 |
| **$k=4$ recon-cls 准确率 (%)** | | | |
| 无信道 | 86.9 | 86.1 | 35.3 |
| AWGN +5 dB | 84.6 | 83.1 | 3.7 |
| AWGN +10 dB | 86.9 | 86.1 | 35.2 |
| Rayleigh +5 dB | 22.6 | 19.2 | 2.8 |
| Rayleigh +10 dB | 66.8 | 62.8 | 3.0 |
| **累积 codeword 上的分层 linear probe (%)** | | | |
| L0 | 86.0 | 86.2 | 30.5 |
| L0+L1 | 87.8 | 87.0 | 33.7 |
| L0+L1+L2 | 87.9 | 88.3 | 35.8 |
| L0+L1+L2+L3 | 87.9 | 88.3 | 36.5 |

**结果。** L1-only 反事实在**全部四个指标族上同时崩塌**。L0 layered probe 从 $86.0\%$ 跌到 $30.5\%$（**−55.5 pp**），无信道 $h_0$ 从 $82.6\%$ 跌到 $34.2\%$（**−48.4 pp**），无信道 $k=4$ PSNR 从 $25.92$ dB 跌到 $22.32$ dB（**−3.60 dB**）；AWGN +5 dB 下 $k=4$ PSNR 损失高达 **−9.58 dB**，$k=4$ recon-cls 仅 $3.7\%$，接近随机。**关键的是**：L1 probe **并没有反过来上升以补偿** —— 每条累积 probe 都停留在 30–37\% 区间，因此**缺失的语义并未被搬到 L1**，而是整个 RVQ 训练根本没有收敛好。

All-layer 反事实**比 L0-only 主模型一致地差**，但远没有 L1-only 那样灾难性。其 $h_0$ 损失在无信道与 AWGN +10 dB 条件下为 **−1.4 pp**，在 Rayleigh +5 dB 条件下达到 **−9.3 pp**；无信道 $k=4$ PSNR 仅落 **−0.28 dB**，$k=4$ recon-cls 仅落 **−0.8 pp**。All-layer 蒸馏下的 layered probe 几乎持平（与主模型相比 ±0.4 pp 内），说明**把对齐损失分散到所有层会轻微稀释 L0 的语义浓度，但并不在其他层补偿**。

**结论。** L0-only 蒸馏是**实测最优**且是**蒸馏位置带来的因果结果**，而非第一残差层的某种架构属性：把对齐损失分散到所有层会带来**轻微、部分**的稀释；把它移到 L1 则**灾难性地**同时退化所有指标族。三个 ckpt 都使用 seed = 42，但观测到的 deltas 远超我们在主重建结果中测得的三 seed std（PSNR std $\approx 0.08$ dB $\ll 0.28$ dB；$h_0$ std $\approx 0.8$ pp $\ll 9.3$ pp），因此初始化噪声无法解释这些差距。layered probe 在此**作为四个指标族中的一个**辅助证据，而不再独自承担层特化论证；更广义的"L1–L3 更适合视为重建残差细节层"由**蒸馏位置干预**与 probe **共同**支持。

---

### 4.4 额外的 RVQ 层如何影响重建？

**问题。** "L0 携带任务语义"并不意味着 L0 单独足以做图像重建。本实验问：**传输更多 RVQ 层是否在重建路径上改善图像质量与重建图分类准确率？**

**设置。** 表 3 报告 RemoteCLIP 蒸馏 tokenizer 的重建路径结果，在三个独立训练种子（s41/s42/s43）下评估，并报告三种子的 mean ± std。本小节**不使用** $h_0$/L0 指标。$k=2\ldots 4$ 的声明仅由 PSNR、LPIPS、clean AID ResNet34 重建图分类器准确率支持。每个 cell 在每信道条件下聚合 1,000 张 AID test 图像；信道噪声在三种子间保持一致（每个 cell 的方差只反映三个 RVQ-distill 模型种子）。完整 $k=1\ldots 4$ sweep 在五个信道配置下都报告。

**表 3**：RemoteCLIP 蒸馏 tokenizer 的重建路径，按三个独立训练种子（s41/s42/s43）报告 mean ± std。本表不使用 $h_0$/L0 指标；$k=2\ldots 4$ 的声明由重建图指标支持。

| 信道 | SNR | $k$ | Bits/image | PSNR (dB) | LPIPS ↓ | Recon. cls. acc. (%) |
|---|---|---|---|---|---|---|
| None | -- | 1 | 2,560 | $22.98\pm0.09$ | $0.284\pm0.002$ | $71.4\pm1.3$ |
| None | -- | 2 | 5,120 | $24.76\pm0.04$ | $0.205\pm0.001$ | $84.6\pm0.2$ |
| None | -- | 3 | 7,680 | $25.55\pm0.06$ | $0.183\pm0.002$ | $86.6\pm0.3$ |
| None | -- | 4 | 10,240 | $25.92\pm0.08$ | $0.175\pm0.002$ | $86.8\pm0.4$ |
| AWGN | +5 dB | 1 | 2,560 | $21.99\pm0.07$ | $0.313\pm0.002$ | $68.5\pm1.4$ |
| AWGN | +5 dB | 2 | 5,120 | $23.24\pm0.02$ | $0.245\pm0.001$ | $80.6\pm0.5$ |
| AWGN | +5 dB | 3 | 7,680 | $23.72\pm0.04$ | $0.225\pm0.002$ | $83.2\pm0.9$ |
| AWGN | +5 dB | 4 | 10,240 | $23.94\pm0.09$ | $0.217\pm0.002$ | $84.4\pm0.9$ |
| AWGN | +10 dB | 1 | 2,560 | $22.98\pm0.09$ | $0.284\pm0.002$ | $71.4\pm1.3$ |
| AWGN | +10 dB | 2 | 5,120 | $24.76\pm0.04$ | $0.205\pm0.001$ | $84.6\pm0.2$ |
| AWGN | +10 dB | 3 | 7,680 | $25.55\pm0.06$ | $0.183\pm0.001$ | $86.6\pm0.4$ |
| AWGN | +10 dB | 4 | 10,240 | $25.92\pm0.08$ | $0.175\pm0.002$ | $86.8\pm0.4$ |
| Rayleigh | +5 dB | 1 | 2,560 | $16.95\pm0.05$ | $0.481\pm0.002$ | $22.0\pm1.5$ |
| Rayleigh | +5 dB | 2 | 5,120 | $16.94\pm0.06$ | $0.470\pm0.003$ | $20.5\pm0.4$ |
| Rayleigh | +5 dB | 3 | 7,680 | $16.90\pm0.05$ | $0.470\pm0.003$ | $19.7\pm1.3$ |
| Rayleigh | +5 dB | 4 | 10,240 | $16.90\pm0.07$ | $0.471\pm0.003$ | $19.7\pm2.6$ |
| Rayleigh | +10 dB | 1 | 2,560 | $19.87\pm0.07$ | $0.383\pm0.002$ | $55.2\pm1.9$ |
| Rayleigh | +10 dB | 2 | 5,120 | $20.33\pm0.07$ | $0.342\pm0.003$ | $63.4\pm1.3$ |
| Rayleigh | +10 dB | 3 | 7,680 | $20.50\pm0.04$ | $0.330\pm0.003$ | $65.1\pm0.7$ |
| Rayleigh | +10 dB | 4 | 10,240 | $20.55\pm0.07$ | $0.326\pm0.003$ | $67.7\pm1.1$ |

> **图 3**（`fig_exp_progressive_reconstruction_v3.png`）：传输更多 RVQ 层时的渐进式重建。该图对应表 3 中的重建路径指标，不使用 $h_0$/L0 bag-of-words 任务指标。

**结果。** 无信道下增大 $k$ 从 1 到 4：PSNR 从 $22.98\pm0.09$ dB 升到 $25.92\pm0.08$ dB，LPIPS 从 $0.284\pm0.002$ 降到 $0.175\pm0.002$，重建图分类准确率从 $71.4\pm1.3$\% 升到 $86.8\pm0.4$\%。**$k=1\to 4$ 的 PSNR gap = 2.94 dB，远超每 cell 的 std（0.04–0.09 dB）**，因此渐进式重建在三个种子间统计稳健。AWGN +5 dB 同向：PSNR 从 $21.99\pm0.07$ dB 升到 $23.94\pm0.09$ dB，重建图分类准确率从 $68.5\pm1.4$\% 升到 $84.4\pm0.9$\%。AWGN +10 dB sweep 在每个 $k$ 下与无信道 sweep **统计上不可区分**，cell-wise 差异都在一个 std 之内。Rayleigh +5 dB 是更困难的衰落 stress：PSNR 在不同 $k$ 下都在 16.9 dB 附近，$k=4$ 重建图分类准确率在无保护重建路径 sweep 下仅有 $19.7\pm2.6$\%，**额外 RVQ 层在深衰落下不起作用**。Rayleigh +10 dB 部分恢复，$k=4$ 达 PSNR $20.55\pm0.07$ dB、重建图分类准确率 $67.7\pm1.1$\%；随 $k$ 的渐进性保留但幅度温和。

**结论。** 表 3 支持"额外 RVQ 层在无信道与 AWGN sweep 下渐进改善重建路径"的结论，且三种子间方差比 $k=1\to 4$ gap **小一个数量级**。在 Rayleigh +5 dB 强衰落下，渐进性大体丧失，重建路径在所有种子下都显著退化；+10 dB 下仅保留温和的部分恢复。**该结论仅由重建路径指标支持，不依赖 $h_0$/L0 bag-of-words 准确率**。

---

### 4.5 同传输比特严格 codec+LDPC stress baseline

**问题。** "RS-Token indices 可被 rate-1/2 LDPC 包裹并在含噪信道下解码"只是一个**兼容性声明**。一个更尖锐的问题是：**当两条 pipeline 被强制传输相同数量的信道比特时，传统 codec+LDPC 是否能赶上 RS-Token？** 本小节用一项**严格同传输比特 stress 比较**回答这个问题：RS-Token、JPEG2000 与 WebP 各自传输完全相同数量的信道比特，经过完全相同的 LDPC 保护，并在接收端报告**解码失败率**与**重建质量**。

**设置。** 评估三个传输比特预算：total bits ∈ {5,120, 10,240, 20,480}。每个方法都被相同的 rate-1/2 systematic sparse LDPC 保护，min-sum BP 解码；该 LDPC 实现是本仓库的**自定义稀疏码**，不是 5G NR LDPC。由于编码率 $1/2$，**源比特预算是传输比特预算的一半**。RS-Token 用 $k=1,2,4$ 层 RVQ，使源比特数分别匹配 2,560、5,120、10,240。JPEG2000 与 WebP 编码到相同的源比特预算，再经过与 RS-Token 相同的 rate-1/2 LDPC 保护。每个 cell 在 5 个 channel seeds 上平均。表 4 报告 SNR = +10 dB 下 AWGN 与 Rayleigh 信道的结果；完整 +5/+10 dB sweep 见 LDPC appendix。

**表 4**：SNR = +10 dB 下严格同传输比特 codec+LDPC stress baseline。RS-Token、JPEG2000 与 WebP 各被编码到匹配的**源比特**预算，使经过 rate-1/2 LDPC 后，三方 **总传输比特** 一致。"Src. bits (mean)" 为编码后实际生成的源比特数（WebP 编码器**不能总是精确达到**最低预算）。"Decode failure" 为 LDPC + 容器解码后没有可用输出的图像比例。"Valid PSNR" 仅对成功解码图像计算；"--" 表示无成功解码因而无 valid PSNR。数据在 5 个 channel seeds 上平均。

| 方法 | 信道 | $k$ | Total bits | Src. bits (mean) | Decode fail. (%) | Recon. cls. acc. (%) | Valid PSNR (dB) |
|---|---|---|---|---|---|---|---|
| RS-Token | AWGN | 1 | 5,120 | 2,560 | **0.0** | 70.3 | 22.95 |
| JPEG2000 | AWGN | -- | 5,120 | 2,663 | 99.9 | 0.0 | -- |
| WebP | AWGN | -- | 5,120 | 14,164 | 99.3 | 0.3 | -- |
| RS-Token | AWGN | 2 | 10,240 | 5,120 | **0.0** | 84.3 | 24.77 |
| JPEG2000 | AWGN | -- | 10,240 | 5,192 | 90.2 | 0.7 | -- |
| WebP | AWGN | -- | 10,240 | 14,301 | 94.7 | 3.2 | -- |
| RS-Token | AWGN | 4 | 20,480 | 10,240 | **0.0** | **86.9** | **25.92** |
| JPEG2000 | AWGN | -- | 20,480 | 10,246 | 62.1 | 14.3 | -- |
| WebP | AWGN | -- | 20,480 | 15,410 | 79.4 | 13.6 | -- |
| RS-Token | Rayleigh | 1 | 5,120 | 2,560 | **0.0** | 69.5 | 22.56 |
| JPEG2000 | Rayleigh | -- | 5,120 | 2,663 | 99.9 | 0.0 | -- |
| WebP | Rayleigh | -- | 5,120 | 14,164 | 100.0 | 0.0 | -- |
| RS-Token | Rayleigh | 2 | 10,240 | 5,120 | **0.0** | 83.0 | 24.13 |
| JPEG2000 | Rayleigh | -- | 10,240 | 5,192 | 92.8 | 0.5 | -- |
| WebP | Rayleigh | -- | 10,240 | 14,301 | 100.0 | 0.0 | -- |
| RS-Token | Rayleigh | 4 | 20,480 | 10,240 | **0.0** | **86.3** | **25.08** |
| JPEG2000 | Rayleigh | -- | 20,480 | 10,246 | 71.6 | 1.3 | -- |
| WebP | Rayleigh | -- | 20,480 | 15,410 | 100.0 | 0.0 | -- |

**结果。** 表 4 显示在同传输比特下，**RS-Token 解码失败率全部 = 0.0\%**（所有三个预算 × 两条信道），而 codec+LDPC 在最大预算下也大量失败：AWGN +10 dB total = 20,480 时，JPEG2000+LDPC 失败 62.1\%，WebP+LDPC 失败 79.4\%；在更小的 AWGN +10 dB 预算下，JPEG2000+LDPC 失败 90.2–99.9\%，WebP+LDPC 失败 94.7–99.3\%——因此 AWGN +10 dB 下"headline 范围"为 JPEG2000+LDPC 62.1–99.9\%，WebP+LDPC 79.4–99.3\%。Rayleigh +10 dB total = 20,480 下，RS-Token 仍 0.0\% 失败，而 JPEG2000+LDPC 71.6\%、**WebP+LDPC 100.0\% 全失败**。重建图分类准确率方面，RS-Token $k=4$ 在 AWGN +10 dB 下为 86.9\%、Rayleigh +10 dB 下为 86.3\%；codec+LDPC 在所有条件下都停留在 0.0–14.3\%（失败的解码计为分类错误）。JPEG2000+LDPC 在最大 AWGN 预算下失败率的下降是**温和的、部分性的改善，不是差距的弥合**。

**结论。** 在同传输比特下，**RS-Token 离散 token 传输与传统 codec+LDPC pipeline 并不等价**。Codec+LDPC pipeline 保留**结构性脆弱性**，因为压缩码流解码器在 rate-1/2 LDPC 无法完全纠正的残余 bit error 上失败，而 **RS-Token indices 在每个 token 上仍可逐字典查找**：index 内的残余 bit error 只扰乱该 token 的 codebook 查找，而**不会损坏全局熵编码容器**。这一定性差距在两条信道与三个传输比特预算下都被保留，Rayleigh +10 dB 在最大预算下尖锐总结了它：**RS-Token 0.0\% 解码失败 vs. JPEG2000+LDPC 71.6\% vs. WebP+LDPC 100.0\%**。本节 LDPC 仍是自定义 rate-1/2 systematic sparse 码 + min-sum BP，不是 5G NR LDPC；比较是在**固定传输比特**而非任何标准通信协议下进行的。

---

### 4.6 L0 tokenizer 是否能 zero-shot 迁移到 NWPU-RESISC45？

**问题。** §4.2–§4.4 的任务路径准确率全部在 AID \cite{xia2017aid} 上报告。一个合理的关切是：RemoteCLIP \cite{liu2024remoteclip} 蒸馏后的 L0 是承载**真实的遥感场景语义**，还是仅仅是恰好与 AID 标签集对齐的 AID-specific 特征？本节问：**AID 上训练好的 tokenizer 能否 zero-shot 迁移到另一个遥感场景 benchmark？**

**设置。** AID 上训练的 tokenizer（encoder、RVQ codebooks 与 decoder 权重**全部冻结**）直接施加到 NWPU-RESISC45 \cite{cheng2017nwpu}（45 类、31,500 张图像）。**只有线性 $h_0$ probe 在 NWPU 训练 split 上重新拟合**，使用规范的 60/20/20 split。`rvq_distill` 与 `rvq_baseline` 在匹配条件下评估，单 seed（42），每条件 $n=6,300$ NWPU test 图像。

**表 5**：AID 上训练的 L0 tokenizer 到 NWPU-RESISC45 的 zero-shot 迁移。Encoder、RVQ、decoder 权重全部冻结；只有线性 $h_0$ probe 在 NWPU 上重新拟合。单 seed，每条件 $n=6,300$ test 图像。

| 信道 | SNR | `rvq_distill` $h_0$ acc. (%) | `rvq_baseline` $h_0$ acc. (%) | Gap (pp) |
|---|---|---|---|---|
| None | -- | 64.4 | 43.5 | 20.9 |
| AWGN | +5 dB | 61.4 | 41.9 | 19.5 |
| AWGN | +10 dB | 64.4 | 43.5 | 20.9 |
| Rayleigh | +5 dB | 29.7 | 16.5 | 13.2 |
| Rayleigh | +10 dB | 53.3 | 33.6 | 19.7 |

**结果。** 表 5 报告 zero-shot 迁移数据。无信道下 `rvq_distill` 达 64.4\% 而 `rvq_baseline` 达 43.5\%，gap = **20.9 pp**。AWGN +10 dB 保持相同 gap = 20.9 pp。Rayleigh 衰落下 gap 收窄但仍可观：+10 dB 下 19.7 pp（53.3\% vs. 33.6\%）、+5 dB 下 13.2 pp（29.7\% vs. 16.5\%）。**对照 AID 上无信道蒸馏 gap 是 25.1 pp（表 1）**；NWPU 上**只缩小 4.2 pp 到 20.9 pp**——尽管面对的是更难的 45 类任务和不同的类分布。绝对 $h_0$ 准确率下降是因为 NWPU 有 45 类（chance 2.2\%）、AID 有 30 类（chance 3.3\%）。

**结论。** 我们将这次迁移描述为**部分迁移**：绝对 $h_0$ 准确率在更难的 45-way 上下降，但**蒸馏贡献被保留**（gap 仅缩 4.2 pp，跨数据集偏移并未抹除蒸馏增益）。Rayleigh +5 dB 的 gap 最小（13.2 pp），暗示**强衰落 + 数据集偏移**是蒸馏 L0 表示最难的组合。这一结果支持一个**相对温和的声明**：RemoteCLIP 蒸馏的 L0 编码的是**遥感领域场景语义**，而不仅仅是 AID-specific 特征。

---

### 4.7 无保护传统码流如何在 BPSK 信道下失败？

**问题。** RS-Token 的主内部对照是 `rvq_baseline`，因为它共享同一 RVQ-index 框架并直接检验 RemoteCLIP 蒸馏的效果。WebP 与 JPEG2000 是**外部传统压缩基线**——不是架构 ablation，也不应被视作完美公平的语义 tokenizer 比较。本节问一个**更窄的问题**：当传统压缩文件的码流以**无保护**方式经 BPSK 含噪信道传输时，**它们如何失败？**

**设置。** WebP 与 JPEG2000 编码到目标比特预算 2,560 / 5,120 / 10,240 bits/image，与 RS-Token $k=1$ / $k=2$ / $k=4$ 对齐。传统压缩实验**没有** 7,680 bits/image 设置，因此不覆盖 RS-Token $k=3$ 操作点。压缩码流**不经 LDPC 或任何纠错码**直接送入 BPSK 信道。"Actual bits" 表示生成文件的平均字节数（× 8）。"Decode failure" 是接收端无法解码图像的比例。"All-image classifier accuracy" 把失败解码计为分类错误，而 "Valid PSNR" 只对成功解码图像计算。

**表 6**：外部传统压缩 stress 测试。**这是无保护 WebP/JPEG2000 码流经 BPSK 信道**——**不**经 LDPC 保护。Target bits 仅与 RS-Token $k=1,2,4$ 对齐。

| 方法 | 信道/SNR | Target bits | Actual bits (mean) | Decode failure | All-image cls. acc. (%) | Valid PSNR (dB) |
|---|---|---|---|---|---|---|
| WebP | None | 2,560 | 14,164 | 0.000 | 61.8 | 26.56 |
| WebP | AWGN +10 dB | 2,560 | 14,164 | 0.039 | 58.8 | 26.51 |
| WebP | Rayleigh +10 dB | 2,560 | 14,164 | 0.999 | 0.0 | 10.52 |
| WebP | None | 5,120 | 14,301 | 0.000 | 62.6 | 26.65 |
| WebP | AWGN +10 dB | 5,120 | 14,301 | 0.045 | 59.9 | 26.63 |
| WebP | Rayleigh +10 dB | 5,120 | 14,301 | 1.000 | 0.0 | -- |
| WebP | None | 10,240 | 15,410 | 0.000 | 67.5 | 27.04 |
| WebP | AWGN +10 dB | 10,240 | 15,410 | 0.041 | 64.5 | 26.97 |
| WebP | Rayleigh +10 dB | 10,240 | 15,410 | 1.000 | 0.0 | -- |
| JPEG2000 | None | 2,560 | 2,663 | 0.000 | 6.5 | 20.19 |
| JPEG2000 | AWGN +10 dB | 2,560 | 2,663 | 0.003 | 6.6 | 20.11 |
| JPEG2000 | Rayleigh +10 dB | 2,560 | 2,663 | 1.000 | 0.0 | -- |
| JPEG2000 | None | 5,120 | 5,192 | 0.000 | 11.0 | 22.35 |
| JPEG2000 | AWGN +10 dB | 5,120 | 5,192 | 0.001 | 10.9 | 22.27 |
| JPEG2000 | Rayleigh +10 dB | 5,120 | 5,192 | 1.000 | 0.0 | -- |
| JPEG2000 | None | 10,240 | 10,246 | 0.000 | 39.6 | 23.97 |
| JPEG2000 | AWGN +10 dB | 10,240 | 10,246 | 0.004 | 38.9 | 23.90 |
| JPEG2000 | Rayleigh +10 dB | 10,240 | 10,246 | 1.000 | 0.0 | -- |

**结果。** 传统压缩码流是有用的**clean reference** baseline，但无保护码流在含噪信道下表现出**清晰的文件级脆弱性**。WebP 干净重建图分类准确率较高，但其 actual bits 远超请求的 target bits（**实现上不能严格按 rate 匹配**）。Rayleigh +10 dB 下 WebP 几乎全部解码失败。JPEG2000 更接近请求的 target bits，但小预算下 clean 分类准确率低，且在 AWGN +10 dB 与 Rayleigh +10 dB 下也出现大量解码失败。

**结论。** 本节支持一个**有限**结论：无保护传统压缩码流在 BPSK 含噪信道下容易出现文件级失败，而 RS-Token indices 更易被分离为可逐层使用的**任务表示与重建表示**。本实验**不是**针对一个工程化的"codec + 强信道码 + 重传"系统的完整比较，也**不替代** `rvq_baseline` 作为检验 RemoteCLIP 蒸馏贡献的核心 baseline。

---

## 5. 讨论

实验支持以下结论，每个绑定到一个独立的指标族。

**第一**，语义层结论是任务路径结论：在三个模型种子下，L0 bag-of-words 准确率从 $58.23\pm1.57\%$ 提升到 $83.33\pm0.81\%$。

**第二**，"L0-only 蒸馏是实测最优"由蒸馏位置反事实建立：**把 RemoteCLIP 信号移到 L1 同时崩塌任务准确率与重建**，而把它分散到全部四层 RVQ 一致地退化两条路径。因此层特化（L0 偏任务、L1–L3 残差细节）**是蒸馏位置的属性**，而不是某种架构上的偶然。

**第三**，渐进式重建结论是重建路径结论：在三个模型种子下，$k=4$ 无信道 PSNR = $25.92\pm0.08$ dB、重建图分类准确率 = $86.8\pm0.4\%$；在完整无信道与 AWGN +5 dB 的 $k=1\ldots 4$ sweep 中，PSNR、LPIPS、重建图分类准确率随 $k$ 增加普遍改善。

**严格同传输比特 codec+LDPC stress baseline** 给出比单纯"兼容性"更强的陈述：在同传输比特下，**RS-Token 解码失败率为 0.0\%**，而 rate-1/2 LDPC 保护的 JPEG2000 或 WebP pipeline 在 62–100\% 的图像上失败——**给传统 codec 加上纠错码并不能弥合这一定性差距**。无保护 WebP/JPEG2000 结果支持一个外部 stress test 结论：无保护传统压缩码流在 BPSK 含噪信道下可能解码失败。

RemoteCLIP \cite{liu2024remoteclip} 帮助第一层是因为它为遥感图像提供了**领域特定的语义几何**。把这种几何蒸馏到 L0 让第一层 codebook 比"仅由重建驱动的 RVQ 层" \cite{lee2022residual} 对航空场景类别更具判别性。这一思路与通用视觉中"把基础模型的视觉特征蒸馏到 VQ tokenizer 上以增强语义" \cite{beitv2} 一脉相承。这并不意味所有层都变得有语义、也不意味 L0 单独足以做高保真重建。**更精确的解读是层特化**：L0 偏向任务语义，L1–L3 改善像素与感知细节。

一项与 OpenAI CLIP 基线（架构与种子均匹配）的 **teacher ablation** 发现 RemoteCLIP 仅有 **1.4–6.0 pp** 的 h0 优势、且随衰落增强；**重建指标（PSNR、$k=4$ recon-cls）基本相同**，差距分别在 0.1 dB 与 0.3 pp 之内（表 7）。因此本文的贡献**不应被声称为 RemoteCLIP 独占性**：视觉语言蒸馏才是承载机制，RemoteCLIP 提供了**领域特定 teacher**，其优势主要在衰落信道下显现。这与"用通用基础模型嵌入直接做语义通信" \cite{semclip} 给出的观察一致——领域偏移决定了 teacher 的边际收益。

**表 7**：Teacher ablation：RemoteCLIP vs. OpenAI CLIP，架构与种子匹配。RemoteCLIP 的 h0 优势在衰落下增长，但 $k=4$ 重建在两种 teacher 间基本相同。

| 信道 | SNR | RemoteCLIP | OpenAI CLIP | $\Delta$ |
|---|---|---|---|---|
| **$h_0$ 准确率（$k=1$, %）** | | | | |
| None | -- | 82.6 | 80.8 | +1.8 |
| AWGN | +5 dB | 82.5 | 78.8 | +3.7 |
| AWGN | +10 dB | 82.6 | 80.8 | +1.8 |
| Rayleigh | +5 dB | 59.8 | 53.8 | +6.0 |
| Rayleigh | +10 dB | 79.2 | 74.4 | +4.8 |
| **$k=4$ 无信道重建** | | | | |
| PSNR (dB) | -- | 25.92 | 26.03 | −0.11 |
| recon-cls (%) | -- | 86.9 | 86.6 | +0.3 |

表 8 的蒸馏权重 sweep 印证**语义对齐与衰落鲁棒性之间的 trade-off**：λ 从 0.1 增至 1.0 时无信道 $h_0$ 单调上升到 84.5\%，但 Rayleigh +5 dB 下的 $h_0$ 在 $\lambda=0.5$ 取峰值 59.1\%、$\lambda=1.0$ 反而退到 49.8\%；同时 $k=4$ 无信道 PSNR 从 26.20 dB 单调降到 25.64 dB。本文取 $\lambda=0.5$ 是基于这一观察的折中工作点，并非声称为通用最优。

**表 8**：蒸馏权重 $\lambda$ sweep（同 seed=42、同架构、同 50 epoch 训练协议）。无信道与 AWGN 下大 $\lambda$ 收益单调；Rayleigh 衰落下 $\lambda=0.5$ 取峰值，$\lambda=1.0$ 反而退化——这是本文取 $\lambda=0.5$ 的依据。

| 条件 | $\lambda=0.1$ | $\lambda=0.5$（主） | $\lambda=1.0$ |
|---|---|---|---|
| 无信道 $h_0$ (%) | 71.2 | 82.4 | **84.5** |
| AWGN +10 dB $h_0$ (%) | 71.2 | 82.5 | **84.5** |
| AWGN +5 dB $h_0$ (%) | 69.2 | 82.3 | **83.3** |
| Rayleigh +10 dB $h_0$ (%) | 64.3 | **79.0** | 77.6 |
| Rayleigh +5 dB $h_0$ (%) | 42.1 | **59.1** | 49.8 |
| $k=4$ 无信道 PSNR (dB) | **26.20** | 25.92 | 25.64 |
| $k=4$ 无信道 LPIPS ↓ | **0.165** | 0.176 | 0.185 |

NWPU-RESISC45 \cite{cheng2017nwpu} 上的 zero-shot tokenizer 迁移（§4.6）显示**蒸馏增益在数据集间基本保留**：gap 仅缩 4.2 pp（AID 25.1 pp → NWPU 20.9 pp）。这支持"L0 承载真实遥感场景语义"，而非 AID-specific 特征。但绝对 h0 准确率在更难的 45-way 上下降，因此跨数据集声明限于**部分迁移、蒸馏贡献完整**。

从系统视角看，RS-Token **最好被理解为物理层适配之上的应用层表示**，而不是 HARQ、自适应调制编码或 LDPC \cite{proakis} 的替代品；它也不直接替代连续表示的端到端 JSCC \cite{deepjscc}，而是给出与离散信道接口天然兼容的另一条路径，与 \cite{moc-rvq} 中"多层 codebook + 数字语义通信"的思路相呼应。链路差或接收端只需告警时，系统传 L0 走任务路径；链路改善或需要人工复核时，请求 L1–L3 提升重建。LDPC 实验显示该 index 表示**可与纠错码组合**，但联合优化层选、重传、调制和信道编码是未来工作。

**部署侧 encoder–decoder 共 10.87 M 参数、$256\times 256$ 图像 38.31 GFLOPs；单图 GPU 推理延迟 8.59 ms，CPU 单线程 66.70 ms。RemoteCLIP teacher（约 150 M 参数）只在训练时使用，不部署。** 接收端 codebook 增加约 8 MB fp32 存储。该参数量与延迟符合 UAV / 边缘部署的工程约束。

**局限。** 实验用 AID 场景分类而非目标检测或语义分割，因此当前任务路径证据**最强支持图像级地物语义**。重建路径 sweep 报告了三个模型种子；蒸馏位置、teacher、跨数据集三项 ablation 只用了单 seed=42。观测到的 effect size 远超主重建实验中测得的 per-seed std（PSNR std ≈ 0.08 dB；h0 std ≈ 0.8 pp），因此初始化噪声不能解释这些差距，但**多 seed 复现仍留作未来工作**。我们没有提供针对完整传统 codec-plus-channel-code 系统的严格 rate 匹配比较。LDPC 实验使用自定义 sparse LDPC + min-sum 解码，**不应**被视作 5G NR LDPC 比较。

---

## 6. 结论

本文提出 **RS-Token**——面向遥感通信的层次化 RVQ tokenizer。通过**仅在第一层量化上蒸馏 RemoteCLIP**，RS-Token 让 L0 在任务上更具判别力，余下层渐进式细化重建。评估**严格分离** L0 任务路径证据与多层重建路径证据，并以 plain RVQ index 传输作为内部基线检验蒸馏的贡献。无保护 WebP/JPEG2000 与自定义 rate-1/2 systematic sparse LDPC 实验厘清了外部 stress test 与信道编码兼容性的边界。总体而言，结果支持一个**保守但有用**的声明：**RemoteCLIP 引导的层特化可以为信道鲁棒的遥感图像通信提供低码率任务保真与渐进式重建**。

---

## 参考文献

引用见 `paper_draft/latex/rs_token_v05.bib`，IEEEtran 格式。bib 共 17 条 entries，正文已全部引用：

- `liu2024remoteclip` — RemoteCLIP（IEEE TGRS 2024）：L0 蒸馏 teacher，§1 / §2.3 / §3.3 / §4.1–§4.6 / §5。
- `vqvae` — VQ-VAE（NeurIPS 2017）：离散视觉 token 起源，§1 / §2.2。
- `lee2022residual` — RVQ for autoregressive image generation（CVPR 2022）：RVQ 残差量化原方法，§1 / §2.2 / §3.2 / §4.2 / §4.3 / §5。
- `soundstream` — SoundStream（IEEE/ACM TASLP, vol. 30, 2022）：音频神经 codec 中的 RVQ，§2.2 / §3.2。
- `encodec` — EnCodec（TMLR 2023）：高保真神经音频压缩，§2.2。
- `zhang2024speechtokenizer` — SpeechTokenizer（ICLR 2024）：语义层与声学层显式分离的语音 tokenizer，§2.2。
- `vilau` — VILA-U（ICLR 2025）：统一视觉理解与生成的离散 token 接口，§2.2。
- `revqom` — Residual VQ for Multi-Agent Perception（arXiv:2509.21464, 2025）：通信侧 RVQ 应用，§2.2。
- `beitv2` — BEiT v2（arXiv:2208.06366, 2022）：把基础模型视觉特征蒸馏到 VQ tokenizer 上，§2.3 / §5。
- `gao2022task` — UAV Task-Oriented Image Transmission（IEEE TCOM, vol. 70, no. 8, 2022）：UAV 任务导向语义通信，§1 / §2.1。
- `moc-rvq` — MOC-RVQ（IEEE GLOBECOM 2024）：多层 codebook + 数字生成式语义通信，§1 / §2.1 / §5。
- `semclip` — Zero-Shot Semantic Communication with Multimodal Foundation Models（IEEE TVT, vol. 75, no. 5, 2026）：基础模型语义通信 zero-shot，§2.1 / §5。
- `deepjscc` — Deep JSCC（IEEE TCCN 2019）：连续表示的联合源信道编码，§1 / §2.1 / §2.4 / §5。
- `zhang2018lpips` — LPIPS（CVPR 2018）：感知度量，§3.4 / §4.1。
- `xia2017aid` — AID（IEEE TGRS 2017）：AID 数据集，§4.1 / §4.6。
- `cheng2017nwpu` — NWPU-RESISC45（Proc. IEEE 2017）：跨数据集迁移 benchmark，§4.6 / §5。
- `proakis` — Digital Communications（McGraw-Hill, 5th ed., 2007）：BPSK / Rayleigh 信道模型与信道编码概念引用，§2.4 / §3.4 / §4.1 / §5。

完整 bibtex 条目见 `rs_token_v05.bib`（已通过 `bibtex` + 三次 `pdflatex` 编译验证）。

---

> **PDF 编译状态**：3 次 pdflatex pass + bibtex 全部 exit 0；0 undefined references；最终输出 `rs_token_v0.5.pdf`，10 页 / 1.75 MB。
