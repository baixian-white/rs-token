# RS-Token：面向信道鲁棒遥感通信的层次化 RemoteCLIP 蒸馏 Token
作者：Baohui Zhang

## 摘要
遥感图像下行链路常工作在带宽受限、低信噪比和衰落信道条件下，而接收端对信息的需求具有明显层次性：应急告警可能只需要场景类别，现场研判需要能支持任务判断的低码率图像，测绘归档则更依赖高保真重建。针对这一特点，本文提出 RS-Token，一种面向遥感语义通信的层次化离散 token 表示。RS-Token 采用四阶段残差向量量化器 RVQ 编码图像，并将 RemoteCLIP 特征蒸馏到第一层量化 token，使 L0 层优先携带地物和场景语义，后续层逐步补充重建细节。
该表示允许接收端根据链路条件和任务需求选择传输层数：只传输 L0 时开销为 2,560 bits/image，可用于低码率任务识别；继续传输 L1、L2、L3 后，开销分别增加到 5,120、7,680 和 10,240 bits/image，用于逐步改善重建质量。本文评估严格区分两条路径：任务路径只使用 L0 bag-of-words 特征 h0 的分类准确率；重建路径使用 PSNR、LPIPS 和 clean AID ResNet34 重建图分类器准确率。
在 AID 数据集的三个模型随机种子上，RemoteCLIP 蒸馏将无信道条件下的 L0 任务准确率从 58.23±1.57% 提升到 83.33±0.81%，并在 AWGN +5 dB 下保持 82.57±0.31%，在 Rayleigh +10 dB 下保持 78.80±0.72%。重建路径显示，无信道条件下，随着传输层数从 k=1 增加到 k=4，PSNR / 重建图分类准确率从 22.95 dB / 70.3% 提升到 25.92 dB / 86.9%。额外的 rate-1/2 systematic sparse LDPC + min-sum BP decoding 实验表明 RS-Token 可与信道编码叠加使用；该实验是自定义稀疏 LDPC 工程扩展，不是 5G NR LDPC 对比。
关键词：语义通信；遥感；残差向量量化；RemoteCLIP；任务导向通信；层次化 token；信道鲁棒性。

## 1. 引言
以低空无人机巡检和应急场景筛查为例，回传链路常受到距离变化、遮挡、多径衰落和临时通信条件不足的影响，接收端无法总是假设可以稳定获得完整高质量图像。与此同时，接收端的第一需求往往不是立即重建每个像素，而是先完成场景级判断：图像是否属于机场、港口、居民区、农田或灾后疑似受影响区域等类别。只有在链路条件允许或需要人工复核时，系统才进一步请求更高质量的图像重建。因此，遥感图像通信需要一种层次化表示：低码率时优先传输任务相关语义，链路改善时逐步补充重建细节。
传统图像通信通常采用先压缩、再传输的流程：发送端将整幅图像编码为压缩码流，接收端在码流正确恢复后再解码图像。该范式工程成熟，但与上述层次化需求并不完全匹配。一方面，JPEG2000、WebP 等压缩码流对 bit error 较敏感，少量未纠正错误就可能导致局部严重失真甚至文件无法解码；另一方面，传统压缩码流主要面向视觉重建质量优化，其 bitstream 内部并不显式保证最先传输的部分优先保留遥感场景语义。信道编码可以降低误码率，却不能改变源表示本身的信息排序。因此，仅依赖传统压缩和信道编码，难以直接实现低码率下优先保任务、高码率下逐步补细节的传输机制。
一种自然替代方案是直接传输神经 tokenizer 的离散 index，例如 VQ 或 RVQ index。相比传统压缩码流，离散 index 更容易与数字信道、重传和错误保护机制结合，也天然支持按层传输。然而，普通 RVQ 的各层通常由重建损失驱动，第一层 codebook 主要服务于连续 latent 的粗粒度逼近，并不保证 L0 index 对遥感场景类别具有判别性。因此，直接传输无蒸馏 RVQ index 虽然提供了层次化码率结构，但不必然带来低码率下的任务语义保真。
本文提出的 RS-Token 在 RVQ 的分层离散表示上引入 RemoteCLIP 蒸馏，目标是显式改变第一层 token 的信息分工。具体而言，RS-Token 只对 L0 层施加 RemoteCLIP 语义蒸馏，使其对齐遥感视觉语言模型的语义空间；L1 到 L3 层继续编码残差信息并提升重建质量。这样，同一个 tokenizer 可支持前缀式分层传输：接收端可根据链路条件和任务需求选择传输前 k 层 RVQ index。具体 bit 开销由 token 网格大小、codebook size 和层数共同决定；在本文 AID 实验配置下，k=1、k=2、k=3、k=4 分别对应 2,560、5,120、7,680 和 10,240 bits/image。
基于这一设计，本文从三个层面验证 RS-Token：首先，与普通 RVQ index 传输基线 rvq_baseline 比较，检验 RemoteCLIP 蒸馏是否使 L0 获得更强场景语义；其次，评估 k=1 到 k=4 的重建结果，验证后续 RVQ 层是否提供渐进式细节恢复；最后，通过自定义 rate-1/2 systematic sparse LDPC 扩展实验，检验 RS-Token index 是否可以与常规信道编码组合使用。
本文贡献如下：
1. 提出一种 RemoteCLIP 引导的 L0 语义分配机制，在四层 RVQ token 表示中只蒸馏第一层，使最低码率 token 优先承载遥感场景语义，后续层用于渐进式重建。
2. 构建面向遥感图像通信的层次化 token 传输机制，使接收端可根据链路条件和任务需求选择传输前 k 层 RVQ index。具体 bit 开销由 token 网格、codebook size 和层数决定；本文实验配置下的四个前缀传输点分别为 2,560、5,120、7,680 和 10,240 bits/image。
3. 建立任务路径与重建路径分离的评估协议：任务路径使用 h0 / L0 bag-of-words 准确率评估低码率语义保真，重建路径使用 PSNR、LPIPS 和 clean AID ResNet34 重建图分类准确率评估多层重建质量。
4. 在 AID 30 类场景分类和 AWGN / Rayleigh 信道下验证：相较普通 RVQ index 传输基线，RemoteCLIP 蒸馏显著提升 L0 任务准确率；在信道允许时，额外 RVQ 层逐步改善重建质量。

## 2. 相关工作

### 2.1 任务导向语义通信与遥感图像传输
语义通信和任务导向通信关注接收端任务表现，而不只追求像素级重建质量。DeepJSCC 等神经联合信源信道编码工作表明，在图像传输中可以直接针对信道失真学习端到端表示，而不是完全依赖传统压缩加信道编码的分离设计 [deepjscc]。在遥感场景中，Gao 等任务导向遥感语义通信工作进一步说明，UAV / remote-sensing 图像回传可以围绕下游识别任务优化传输表示 [gao2022task]。这些研究支持本文的问题设定：遥感图像通信不一定总以完整重建为唯一目标，而可以优先保证场景级任务信息。与这些方法不同，RS-Token 保留显式离散 index 表示，并将最低码率层设计为可直接传输和评估的语义层。

### 2.2 离散视觉 Token 与残差向量量化
VQ-VAE 将连续图像特征映射为离散 codebook index，为神经离散表示学习和 token 化图像建模奠定了基础 [vqvae]。Residual quantization 进一步使用多个 codebook 逐层逼近同一个 latent，使前几层提供粗粒度近似，后续层补充 residual 信息 [lee2022residual]。类似的多层离散 codebook 思想也被广泛用于神经音频 codec，例如 SoundStream 和 EnCodec，它们通过逐层增加 codebook 改善重建质量并支持可变码率 [soundstream, encodec]。SpeechTokenizer 进一步展示了分层 RVQ 可以形成语义层与细节层的功能分工：其低层 token 更偏向语音内容，高层 token 补充声学细节 [zhang2024speechtokenizer]。本文借鉴 RVQ 的前缀式分层表示，但关注点不是通用压缩质量，而是第一层 index 是否能在遥感任务中优先承载场景语义。

### 2.3 VQ / RVQ 在语义通信中的应用
近期语义通信工作已经开始使用 VQ 或 RVQ 作为数字语义表示。MOC-RVQ 使用多级 codebook 和调制友好的索引设计，面向数字图像语义通信中的 codebook-index 传输问题 [moc-rvq]。ReVQom 在多智能体协同感知中使用多阶段 RVQ 压缩中间特征，说明 RVQ index 传输也可用于通信受限的感知系统 [revqom]。这些工作与本文共享离散 index 和多层量化的技术背景，但其 RVQ 层主要体现精度递进，并未显式将第一层塑造成遥感场景语义层。

### 2.4 Foundation Model 蒸馏与遥感语义对齐
视觉语言基础模型为图像表示提供了更强的语义结构。BEiT v2 等工作表明，视觉 tokenizer 的语义质量会显著影响下游视觉表示学习 [beitv2]。SemCLIP 等 CLIP-based semantic communication 工作也说明，CLIP 语义空间可以作为通信表示设计的重要先验 [semclip]。RemoteCLIP 针对遥感图文对齐训练，其特征空间更适合表达遥感场景、地物类别和遥感图像语义 [liu2024remoteclip]。因此，本文不直接使用普通 RVQ 的重建驱动 L0，而是将 RemoteCLIP 的遥感语义结构蒸馏到 L0 派生特征中。与把 teacher feature 蒸馏到整个 latent 不同，RS-Token 只约束第一层，使最低码率 token 更适合场景级任务保真，同时保留后续层用于重建细节。

### 2.5 本文与现有工作的区别
综上，已有工作分别从任务导向通信、离散 token 表示、VQ/RVQ 语义通信和视觉语言模型蒸馏等方向提供了基础 [deepjscc, gao2022task, vqvae, lee2022residual, moc-rvq, revqom, liu2024remoteclip]。然而，普通 RVQ index 传输虽然具有分层结构，却不保证第一层对遥感场景类别具有判别性；传统压缩码流虽然成熟，却不显式提供语义优先的前缀表示。本文的核心区别在于：在 RVQ index 传输框架中只对 L0 层施加 RemoteCLIP 蒸馏，使最低码率层承担任务语义，后续层承担重建细节，并通过任务路径与重建路径分离的实验协议验证这种层级分工。

## 3. 方法

本节先给出 RS-Token 的整体传输框架，再说明该框架如何通过 RemoteCLIP 引导的训练目标形成，最后分别定义 RVQ 表示、前缀式 index 传输、index 级信道模型和接收端两条使用路径。这样安排是为了区分三个概念：训练阶段的 RemoteCLIP 蒸馏、部署阶段实际传输的 RVQ index，以及实验阶段用于验证任务保真和重建质量的评价路径。

![图 1：RS-Token 方法整体框架](../../rstoken/figs/fig_method_aech_image2pro.png)

图 1 展示 RS-Token 的整体链路。发送端将输入遥感图像 x 编码为四层 RVQ index，即 L0 到 L3。部署时，系统不必发送全部 index，而是根据链路条件和接收端需求发送前 k 层 index。接收端收到 index 后有两种使用方式：如果只需要场景级任务判断，则使用 L0 派生的 h0 / bag-of-words 表示；如果需要图像内容，则将收到的前 k 层 index 反量化并解码为重建图像。训练阶段的 RemoteCLIP 蒸馏只直接监督 L0 派生表示，使最低码率层优先承载遥感场景语义，后续层继续补充重建细节。

图中的黑色实线表示部署时真实经过信道传输的数据流，红色虚线表示只在训练阶段使用的 teacher-student 蒸馏信号，橙色实线表示接收端用于验证 L0 任务保真的下游 probe。需要强调的是，发送端实际传输的是离散 index，而不是连续 latent 或传统压缩文件；RemoteCLIP teacher 和 DistillHead 只用于训练，不参与部署传输；linear probe 只用于评估 L0 表示是否具有任务判别性，不是传输系统的必要组成部分。

### 3.1 RemoteCLIP 引导的 L0 训练方法

普通 RVQ tokenizer 只由重建目标驱动时，第一层 codebook 通常学习的是对连续 latent 的粗粒度近似，而不一定形成适合遥感场景分类的语义划分。RS-Token 的训练目标不是把所有 RVQ 层都变成语义 token，而是显式改变第一层的功能分工：L0 优先承载场景级语义，L1 到 L3 继续补充重建所需的残差信息。

训练时，输入图像 x 同时进入两条分支。第一条是自编码重建分支：编码器 E 将 x 映射为 latent z，四层 RVQ 将 z 量化为多层 codeword，解码器 D 根据量化 latent 重建图像 x_hat。该分支提供图像重建损失和 RVQ 量化损失，用于保证 tokenizer 仍然具备图像压缩与重建能力。第二条是训练专用的 RemoteCLIP 蒸馏分支：冻结的 RemoteCLIP 图像编码器 f_T 从原图中提取 teacher embedding t；学生侧只从 L0 量化特征中得到表示 h0，经 DistillHead phi 投影为 student embedding s，并用余弦对齐损失约束 s 接近 t。

设一批训练图像为 {x_i}_{i=1}^B。编码器首先得到连续 latent：

```text
z_i = E(x_i)
```

重建分支输出 x_hat_i，并用像素级 L1 项和感知项组成图像损失：

```text
L_img = alpha * (1/B) sum_i || x_i - x_hat_i ||_1
      + beta  * (1/B) sum_i LPIPS(x_i, x_hat_i)
```

RVQ 量化损失写作各层量化约束的加权和：

```text
L_vq = gamma * sum_{l=0}^{L-1} L_vq^(l)
```

其中 L 是 RVQ 层数，L_vq^(l) 表示第 l 层 codebook 与 encoder latent / residual 表示之间的量化匹配损失，gamma 是量化损失权重。该项用于保持 encoder 输出、离散 codebook 和 straight-through 量化表示之间的训练稳定性。

RemoteCLIP 蒸馏只从 L0 量化特征构造学生表示。记 q_0 为第一层 codeword map，pool 为空间平均池化，phi 为 DistillHead，则：

```text
h0_i = pool(q_0,i)
s_i  = phi(h0_i)
t_i  = f_T(x_i)
```

其中 f_T 是冻结的 RemoteCLIP 图像编码器。蒸馏损失采用 batch-mean cosine distance：

```text
L_distill = (1/B) sum_i [1 - cos(s_i, t_i)]
          = (1/B) sum_i [1 - <s_i, t_i> / (||s_i||_2 ||t_i||_2)]
```

选择余弦距离的原因是，RemoteCLIP embedding 的主要作用是提供遥感语义空间中的方向性结构，而不是要求学生表示复现 teacher feature 的绝对尺度。通过最小化 L_distill，L0 聚合表示被鼓励接近 RemoteCLIP 对同一遥感图像的语义方向，从而使最低码率 index 更容易保留场景类别、地物组合和全局语义信息。

综合上述项，RS-Token 的训练目标为：

```text
L_total = L_img + L_vq + lambda * L_distill
        = alpha * L1(x, x_hat) + beta * LPIPS(x, x_hat)
          + gamma * sum_l L_vq^(l) + lambda * L_distill
```

其中 alpha、beta、gamma 和 lambda 分别控制像素重建、感知重建、RVQ 量化和 L0 语义蒸馏的相对权重。前三项保证 tokenizer 仍然服务于图像重建和稳定量化；最后一项直接监督 L0 派生表示，并主要引导 L0 的信息分配。lambda 是语义蒸馏与重建目标之间的权衡系数：lambda 过小，L0 仍可能主要服务于重建；lambda 过大，则可能牺牲 tokenizer 的重建能力。因此，本文将 RemoteCLIP 蒸馏作为附加约束，而不是替代重建目标。具体权重设置放在实验设置中报告。

需要强调的是，L_distill 只通过 L0 相关表示回传，不直接约束 L1、L2 和 L3。这样设计的原因是：如果所有层都被强制对齐 RemoteCLIP，后续层可能不再专注于补充细节；而本文需要的是一个层次化 token 表示，即最低码率层负责任务语义，额外层负责渐进式重建。因此，损失函数的设计本身就对应本文的层级分工假设：L0 承担任务语义，L1–L3 承担残差细节。

### 3.2 四层残差向量量化表示

RS-Token 的离散表示由四层 residual vector quantization tokenizer 产生。编码器 E 将图像 x 映射为连续 latent 表示 z = E(x)。第一层 codebook 量化原始 latent，后续层逐步量化上一层留下的残差。记第 l 层 codebook 为 C_l，第 l 层 index map 为 I_l，量化 codeword map 为 q_l。对 latent 网格中的空间位置 u，量化过程为：

```text
r_0(u) = z(u)
I_l(u) = argmin_j || r_l(u) - C_l[j] ||_2^2
q_l(u) = C_l[I_l(u)]
r_{l+1}(u) = r_l(u) - q_l(u)
```

当接收端获得前 k 层 index 时，对应 codeword 被相加形成前缀重建 latent：

```text
z_hat_k = sum_{l=0}^{k-1} q_l
x_hat_k = D(z_hat_k)
```

该结构自然支持渐进式重建：k=1 时只保留最粗粒度的量化结果；k 增加时，更多 residual codeword 被加入，重建图像应包含更多纹理和结构细节。本文将这一点放在重建路径中验证，而不是用 L0 任务指标替代重建指标。

### 3.3 前缀式 RVQ index 传输

RS-Token 的发送对象不是传统压缩文件，而是神经 tokenizer 产生的离散 index map。给定四层 index 集合 {I0, I1, I2, I3}，当链路只能支持最低开销传输时，发送端只发送 I0；当链路条件更好或接收端需要更高质量重建时，发送端发送前 k 层 index：

```text
I_{0:k-1} = {I0, I1, ..., I_{k-1}},    k in {1, 2, 3, 4}
```

传输开销由 token 网格大小、codebook size 和前缀层数共同决定。若 latent 网格包含 T 个空间位置，codebook size 为 K，则每层 index 需要 T log2 K bit，前 k 层的开销为：

```text
B(k) = k * T * log2 K  bits/image
```

这个公式说明，bits/image 不是方法本身的普适常数，而是由 tokenizer 配置决定。当前 AID 实验配置中，T=16×16=256，K=1024，因此每层 index 对应 2,560 bits/image，k=1 到 k=4 分别对应 2,560、5,120、7,680 和 10,240 bits/image。该具体数值将在实验设置中作为本文配置下的传输工作点使用。

### 3.4 Index 级信道模型

每个 RVQ index 先转换为二进制比特，再经过 BPSK 调制并通过 AWGN 或 Rayleigh fading 信道。接收端对信道输出进行硬判决，恢复比特并重新组合为 index map。整体传输过程可写为：

```text
index map -> bits -> BPSK symbols -> channel -> received bits -> recovered index map
```

这种设置保留了数字通信链路的可解释性：信道错误直接表现为 index 错误。本文的实验重点不是文件格式能否被完整解码，而是错误 index 下任务路径和重建路径还能保留多少可用信息。

### 3.5 接收端任务路径与重建路径

接收端根据任务需求选择不同路径。任务路径只使用 L0 index 派生的 h0 / bag-of-words 表示，并只用于评价 k=1 低码率语义保真。重建路径使用收到的前 k 层 index 反量化得到近似 latent，再由解码器生成重建图像 x_hat_k，并用图像质量和重建图分类指标评价 k=1 到 k=4 的渐进式重建。

| 阶段 / 路径 | 使用模块 | 输入 | 输出 | 作用 |
|---|---|---|---|---|
| 训练蒸馏 | RemoteCLIP teacher + DistillHead | 原图 x 与 L0 量化特征 q0 | L_distill | 只在训练中引导 L0 语义分配 |
| 部署传输 | RVQ index + BPSK 信道 | 前 k 层 index | 接收端 index | 实际通信载荷 |
| 任务路径 | L0 bag-of-words probe | L0 index | 场景分类准确率 | 验证 k=1 任务保真 |
| 重建路径 | Codebook lookup + Decoder | 前 k 层 index | 重建图像 x_hat_k | 验证 k=1..4 渐进式重建 |

这一分离对本文论证很重要：h0 / L0 bag-of-words accuracy 不能用来证明 k=2 到 k=4 的重建质量；PSNR、LPIPS 和重建图分类准确率也不能单独证明 L0 具有任务语义。后续实验将分别回答两个问题：RemoteCLIP 蒸馏是否让 L0 更适合低码率任务识别，以及额外 RVQ 层是否在不同信道下逐步补充重建细节。

## 4. 实验

本节围绕三类核心问题和两个补充问题组织实验，而不是按日志顺序堆叠结果。第一，L0 是否因为 RemoteCLIP 蒸馏获得更强的场景语义；第二，后续 RVQ 层是否继续增加任务语义，还是主要承担残差细节；第三，在已经获得前缀式 RVQ index 后，增加 L1 到 L3 是否逐步改善重建质量。两个补充实验分别回答工程边界问题：RS-Token index 是否可以接入常规信道编码，以及未保护传统压缩码流在相同 BPSK 噪声链路下会出现怎样的失效模式。每个核心小节均按照“问题—设置—结果—结论”的逻辑展开；传统压缩结果只作为外部压力测试，不作为本文核心创新证据。

为避免层次化表示中的指标混用，本文将评估分为任务路径、层级 probe 和重建路径。任务路径只使用 h0 / L0 bag-of-words 准确率，并只用于 k=1 的低码率语义层结论；层级 probe 使用逐层累加的 codeword embedding，用于分析 L0 与 L1–L3 的语义分工；重建路径使用 PSNR、LPIPS 和 clean AID ResNet34 重建图分类准确率，并用于支持 k=1 到 k=4 的重建结论。换言之，h0 / L0_bow 和层级 probe 不用于证明多层重建质量，PSNR / LPIPS 也不单独用于证明 L0 语义化。

实验使用 AID 航拍场景数据集，共 30 类。内部 tokenizer 对比包括两类模型：rvq_baseline 是普通 RVQ index 传输基线，不含 RemoteCLIP 蒸馏；rvq_distill 是相同 RVQ index 传输框架下加入 L0 RemoteCLIP 蒸馏的 RS-Token。二者共享相同的 RVQ 层数、codebook size 和传输协议，因此该对比用于检验 RemoteCLIP 蒸馏本身的作用。

### 4.1 实验设置

本文实验使用 AID 航拍场景数据集，共 30 个场景类别。所有实验使用固定的 train / validation / test split；训练阶段采用 RandomResizedCrop(256)，评测阶段采用 Resize + CenterCrop(256)，并将图像归一化到训练脚本使用的输入范围。RS-Token tokenizer 采用四层 RVQ，输入图像大小为 256×256，latent 网格大小为 16×16，latent channel 为 256，codebook size 为 1024。因此每层 index map 包含 256 个离散符号，每个符号需要 log2(1024)=10 bit，单层传输开销为 2,560 bits/image。

内部 tokenizer 对比包括 rvq_baseline 和 rvq_distill。rvq_baseline 是普通四层 RVQ tokenizer，仅使用图像重建损失和 RVQ 量化损失训练；rvq_distill 与其架构相同，但额外加入 L0 RemoteCLIP 蒸馏损失。两者使用相同的 encoder / decoder 结构、四层 RVQ quantizer、batch size 16、AdamW 优化器、学习率 1e-4、50 个训练 epoch、梯度裁剪和 bf16 automatic mixed precision。训练目标采用第 3 节定义的 L_total，其中 L1、LPIPS、RVQ 和蒸馏损失权重分别为 1、0.1、1 和 0.5；LPIPS 使用 AlexNet backbone。rvq_distill 的 teacher 为冻结的 RemoteCLIP-ViT-B/32，DistillHead hidden dimension 为 512。

任务路径评估使用 L0 index 构造 1024 维 bag-of-words 表示 h0，并训练线性分类器预测 AID 场景类别。重建路径评估使用接收端前 k 层 index 解码得到的重建图像，并报告 PSNR、LPIPS 和 clean AID ResNet34 重建图分类准确率。除特别说明外，任务路径主结果使用模型随机种子 41/42/43 报告 mean±std；重建路径使用对应实验日志中的主配置结果。信道实验将 RVQ index bit 经 BPSK 调制后置于 AWGN 或 Rayleigh 信道中，并在接收端硬判决恢复 index。

### 4.2 L0 是否被 RemoteCLIP 蒸馏语义化？

**问题。** RS-Token 的核心假设是：仅传输 L0 时，接收端仍应保留可用于场景级判断的语义信息。普通 RVQ 虽然也能产生 L0 index，但它只受重建目标驱动，不能保证第一层 index 对 AID 场景类别具有判别性。因此，本实验要回答的问题是：在相同 RVQ index 传输框架下，RemoteCLIP 蒸馏是否显著提高 L0 的任务保真度？

**设置。** 本实验比较 rvq_baseline 与 rvq_distill。两者差异在于是否加入 L0 RemoteCLIP 蒸馏，其余 tokenizer 架构和 index 传输方式保持一致。任务路径只使用接收端 L0 index 构造 h0 / L0 bag-of-words 表示，并训练线性 probe 进行 AID 30 类场景分类。该指标只对应 k=1 的低码率任务路径，不用于支持 k=2 到 k=4 的重建结论。表 1 报告三个模型随机种子的 mean ± std，并同时列出 best PSNR / LPIPS 作为训练质量参考，而非语义层结论的主要证据。

| 指标 | 无蒸馏 | RemoteCLIP 蒸馏 |
|---|---:|---:|
| Best PSNR (dB) | 26.10±0.02 | 25.89±0.07 |
| Best LPIPS | 0.172±0.001 | 0.175±0.002 |
| No channel h0 acc. | 58.23±1.57% | 83.33±0.81% |
| AWGN +5 dB h0 acc. | 55.73±0.72% | 82.57±0.31% |
| AWGN +10 dB h0 acc. | 58.20±1.59% | 83.37±0.76% |
| Rayleigh +5 dB h0 acc. | 28.67±0.65% | 58.57±0.47% |
| Rayleigh +10 dB h0 acc. | 48.63±1.31% | 78.80±0.72% |

![图 2：L0 任务路径在不同信道条件下的鲁棒性](../../rstoken/figs/fig_exp_l0_task_robustness_v3.png)

**结果。** RemoteCLIP 蒸馏带来显著且稳定的 L0 任务增益。无信道条件下，h0 / L0_bow 准确率从 58.23±1.57% 提升到 83.33±0.81%，提升 25.10 个百分点。在 AWGN +5 dB 和 +10 dB 下，蒸馏模型分别达到 82.57±0.31% 和 83.37±0.76%，接近其无信道表现。Rayleigh fading 更困难，但蒸馏模型在 +5 dB 和 +10 dB 下仍分别比无蒸馏模型高 29.90 和 30.17 个百分点。

**结论。** 表 1 支持一个明确的结论：RemoteCLIP 蒸馏显著提高了 L0 index 的场景级任务判别性，使 k=1 的低码率任务路径更可用。该结论只关于 L0 语义保真，不意味着所有 RVQ 层都被语义化，也不用于证明 k=2 到 k=4 的重建质量。

### 4.3 后续 RVQ 层是否继续增加任务语义？

**问题。** 第 4.2 节说明 RemoteCLIP 蒸馏显著提高了 L0 的任务判别性，但这还不足以证明 RS-Token 形成了清晰的层级分工。一个可能的替代解释是：后续 L1–L3 也继续携带大量任务语义，L0 只是其中一部分。因此，本实验进一步检查当逐层累加 RVQ codeword 后，任务路径分类准确率是否继续明显提升。

**设置。** 我们对 rvq_baseline 和 rvq_distill 分别构造 L0、L0+L1、L0+L1+L2、L0+L1+L2+L3 四种累加 codeword embedding，并在每种表示上训练线性 probe 进行 AID 场景分类。该实验使用主模型配置，属于辅助机制分析；它只用于分析层级语义分工，不用于评价重建质量，也不作为三随机种子任务路径主结果。

| 累加层 | 无蒸馏 acc. | 无蒸馏增量 | RemoteCLIP 蒸馏 acc. | 蒸馏增量 |
|---|---:|---:|---:|---:|
| L0 | 47.70% | -- | 86.00% | -- |
| L0+L1 | 47.20% | -0.50 | 87.70% | +1.70 |
| L0+L1+L2 | 47.70% | +0.50 | 87.90% | +0.20 |
| L0+L1+L2+L3 | 48.30% | +0.60 | 88.00% | +0.10 |

**结果。** 对蒸馏模型而言，从 L0 到 L0+L1 的任务准确率只增加 1.70 个百分点，继续加入 L2 和 L3 后总共只再增加 0.30 个百分点。这说明大部分任务语义已经集中在 L0 中。无蒸馏模型在四种累加表示下始终停留在约 47%–48%，说明普通 RVQ 的多层残差并不会自然形成场景语义层。

**结论。** 该实验直接支持 RS-Token 的层级分工假设：RemoteCLIP 蒸馏主要使 L0 获得场景级语义，后续 RVQ 层并没有显著增加任务语义，而更适合解释为重建路径中的残差细节层。因此，第 4.4 节讨论 k=1 到 k=4 的改善时，只使用重建路径指标，而不使用 L0_bow 指标支撑重建结论。

### 4.4 额外 RVQ 层如何影响重建？
**问题。** 前述实验说明 L0 已经承担主要任务语义，但这并不意味着只传 L0 就足以满足图像重建需求。本实验要回答的问题是：当接收端传输更多 RVQ 层时，重建路径的图像质量和重建图分类准确率是否逐步改善？

**设置。** 表 2 给出 RemoteCLIP-distilled tokenizer 的重建路径结果。这里不使用 h0 / L0 指标；k=2 到 k=4 的结论只由 PSNR、LPIPS 和 clean AID ResNet34 重建图分类准确率支持。完整的 k=1..4 sweep 覆盖无信道和 AWGN +5 dB；AWGN +10 dB 与 Rayleigh 条件只报告关键 k=4 工作点，用于展示高 SNR 或衰落场景下的重建表现。

| 信道 | SNR | k | bits/image | PSNR (dB) | LPIPS↓ | 重建图分类准确率 |
|---|---:|---:|---:|---:|---:|---:|
| None | -- | 1 | 2,560 | 22.95 | 0.284 | 70.3% |
| None | -- | 2 | 5,120 | 24.77 | 0.205 | 84.3% |
| None | -- | 3 | 7,680 | 25.57 | 0.183 | 86.4% |
| None | -- | 4 | 10,240 | 25.92 | 0.175 | 86.9% |
| AWGN | +5 dB | 1 | 2,560 | 21.97 | 0.312 | 67.3% |
| AWGN | +5 dB | 2 | 5,120 | 23.24 | 0.244 | 81.6% |
| AWGN | +5 dB | 3 | 7,680 | 23.76 | 0.223 | 84.1% |
| AWGN | +5 dB | 4 | 10,240 | 23.96 | 0.217 | 84.3% |
| AWGN | +10 dB | 4 | 10,240 | 25.92 | 0.175 | 86.9% |
| Rayleigh | +5 dB | 4 | 10,240 | 16.96 | 0.467 | 21.0% |
| Rayleigh | +10 dB | 4 | 10,240 | 20.60 | 0.323 | 66.6% |

![图 3：增加 RVQ 层带来的渐进式重建效果](../../rstoken/figs/fig_exp_progressive_reconstruction_v3.png)

**结果。** 无信道条件下，k 从 1 增加到 4 后，PSNR 从 22.95 dB 提升到 25.92 dB，LPIPS 从 0.284 降到 0.175，重建图分类准确率从 70.3% 提升到 86.9%。AWGN +5 dB 下也呈现同向趋势：k=1 到 k=4 的重建图分类准确率从 67.3% 提升到 84.3%。AWGN +10 dB 的 k=4 结果几乎与无信道 k=4 一致。Rayleigh +5 dB 是更困难的衰落压力场景，即使使用 k=4，未保护重建路径的重建图分类准确率也只有 21.0%。

**结论。** 表 2 支持的结论是：在无信道和 AWGN +5 dB 的完整 sweep 中，额外 RVQ 层能够逐步改善重建路径；在 Rayleigh 衰落较强时，重建质量仍会显著退化。该结论仅由重建路径指标支撑，不依赖 h0 / L0_bow。

### 4.5 补充：RS-Token index 能否接入信道编码？

**问题。** 前述实验直接将 RS-Token index bit 经过 BPSK 噪声信道，用于观察 index 错误下的任务路径和重建路径退化。实际数字通信系统通常还会使用纠错码。因此，本实验补充验证一个工程兼容性问题：RS-Token 的离散 index 是否可以像普通数字 bitstream 一样接入信道编码保护。

**设置。** 我们使用 rate-1/2 systematic sparse LDPC code 对 RS-Token bitstream 进行保护，并在接收端采用 min-sum belief-propagation decoding。需要强调的是，该 LDPC 实现是自定义 sparse LDPC 工程实现，不是 5G NR LDPC。由于编码率为 1/2，传输 bit 数相对未编码 index 翻倍：k=1 从 2,560 source bits/image 变为 5,120 transmitted bits/image，k=4 从 10,240 source bits/image 变为 20,480 transmitted bits/image。

| 信道 | SNR | k | transmitted bits | 重建图分类准确率 | h0 acc. |
|---|---:|---:|---:|---:|---:|
| AWGN | +5 dB | 1 | 5,120 | 70.1±0.2% | 82.4±0.2% |
| AWGN | +10 dB | 1 | 5,120 | 70.3±0.0% | 82.4±0.0% |
| Rayleigh | +5 dB | 1 | 5,120 | 47.2±0.6% | 75.0±0.8% |
| Rayleigh | +10 dB | 1 | 5,120 | 69.1±0.6% | 82.6±0.4% |
| AWGN | +10 dB | 4 | 20,480 | 86.9±0.0% | -- |
| Rayleigh | +10 dB | 4 | 20,480 | 86.1±0.1% | -- |

**结果。** 在 AWGN +5 dB 和 +10 dB 下，k=1 的 h0 accuracy 约保持在 82.4%，重建图分类准确率约为 70%。Rayleigh +10 dB 下，k=1 h0 accuracy 为 82.6%，k=4 重建图分类准确率为 86.1%。Rayleigh +5 dB 仍然更困难，k=1 重建图分类准确率下降到 47.2±0.6%，说明信道编码改善鲁棒性但不能完全消除强衰落影响。

**结论。** 该实验支持的是兼容性结论：RS-Token 的离散 index 可以作为普通数字 bitstream 接入纠错编码保护。它不是新的信道编码方法，也不构成 5G NR LDPC 对比；本文核心证据仍来自 4.2 的 L0 任务路径实验、4.3 的层级分工实验和 4.4 的多层重建路径实验。

### 4.6 补充：传统压缩码流在未保护 BPSK 链路下会怎样失效？

**问题。** RS-Token 的主要内部对比对象是 rvq_baseline，因为二者共享同一 RVQ index 传输框架，能够直接检验 RemoteCLIP 蒸馏的作用。WebP 和 JPEG2000 则属于另一类外部传统压缩 baseline。它们不是同架构消融，也不是严格公平的语义 tokenizer 对照；本节只用它们回答一个有限问题：当传统压缩码流不加纠错保护、直接经过 BPSK 噪声信道时，文件级 bitstream 的失效模式是否与 RS-Token index 传输不同？

**设置。** WebP 和 JPEG2000 分别以 target_bits=2,560、5,120 和 10,240 bits/image 进行压缩，对齐 RS-Token 的 k=1、k=2 和 k=4 三个传输开销；由于传统压缩实验没有 target_bits=7,680，因此不覆盖 RS-Token 的 k=3 工作点。压缩后的 bitstream 直接通过 BPSK 信道传输，不加入 LDPC 或其他纠错码。表中 actual bits 表示实际生成的平均码流长度；decode failure 表示接收端无法成功解码文件的比例；all-image cls. acc. 将解码失败样本计为分类错误；valid PSNR 只在可解码样本上计算。

| 方法 | 信道 | Target bits | Actual bits (mean) | Decode failure | All-image cls. acc. | Valid PSNR |
|---|---|---:|---:|---:|---:|---:|
| WebP | None | 2,560 | 14,164 | 0.000 | 61.8% | 26.56 |
| WebP | AWGN +10 dB | 2,560 | 14,164 | 0.039 | 58.8% | 26.51 |
| WebP | Rayleigh +10 dB | 2,560 | 14,164 | 0.999 | 0.0% | 10.52 |
| WebP | None | 5,120 | 14,301 | 0.000 | 62.6% | 26.65 |
| WebP | AWGN +10 dB | 5,120 | 14,301 | 0.045 | 59.9% | 26.63 |
| WebP | Rayleigh +10 dB | 5,120 | 14,301 | 1.000 | 0.0% | -- |
| WebP | None | 10,240 | 15,410 | 0.000 | 67.5% | 27.04 |
| WebP | AWGN +10 dB | 10,240 | 15,410 | 0.041 | 64.5% | 26.97 |
| WebP | Rayleigh +10 dB | 10,240 | 15,410 | 1.000 | 0.0% | -- |
| JPEG2000 | None | 2,560 | 2,663 | 0.000 | 6.5% | 20.19 |
| JPEG2000 | AWGN +10 dB | 2,560 | 2,663 | 0.003 | 6.6% | 20.11 |
| JPEG2000 | Rayleigh +10 dB | 2,560 | 2,663 | 1.000 | 0.0% | -- |
| JPEG2000 | None | 5,120 | 5,192 | 0.000 | 11.0% | 22.35 |
| JPEG2000 | AWGN +10 dB | 5,120 | 5,192 | 0.001 | 10.9% | 22.27 |
| JPEG2000 | Rayleigh +10 dB | 5,120 | 5,192 | 1.000 | 0.0% | -- |
| JPEG2000 | None | 10,240 | 10,246 | 0.000 | 39.6% | 23.97 |
| JPEG2000 | AWGN +10 dB | 10,240 | 10,246 | 0.004 | 38.9% | 23.90 |
| JPEG2000 | Rayleigh +10 dB | 10,240 | 10,246 | 1.000 | 0.0% | -- |

**结果。** 传统压缩码流在 clean 条件下可以作为有用的重建参考，但未保护 bitstream 在噪声信道下表现出明显的文件级脆弱性。WebP 的 clean 重建分类准确率较高，但当前实现的 actual bits 明显超过 target bits，因此不能视为严格 rate-matched 对比；在 Rayleigh +10 dB 下，WebP 几乎全部解码失败。JPEG2000 的 actual bits 更接近 target bits，但低 bit budget 下 clean 分类准确率较低，并且在 AWGN +10 dB 和 Rayleigh +10 dB 下也出现大量解码失败。

**结论。** 本节只支持一个有限结论：未保护传统压缩码流在 BPSK 噪声链路中容易发生文件级失效，而 RS-Token 的离散 index 传输更适合被拆分为可逐层使用的任务/重建表示。该实验不等价于与“传统压缩 + 强信道编码 + 重传协议”的完整工程系统比较，也不应替代 rvq_baseline 作为检验 RemoteCLIP 蒸馏贡献的核心 baseline。

## 5. 讨论
本文实验支持三类核心结论和两个补充结论，每类结论对应不同指标。第一，语义层结论属于任务路径结论：三个模型种子下，L0 bag-of-words accuracy 从 58.23±1.57% 提升到 83.33±0.81%。第二，层级 probe 显示后续 RVQ 层没有显著增加任务语义，支持 L0 偏任务语义、L1–L3 偏残差细节的分工。第三，渐进式重建结论属于重建路径结论：在无信道和 AWGN +5 dB 的完整 k=1..4 sweep 中，PSNR、LPIPS 和重建图分类准确率总体随 k 增加而改善。LDPC 扩展实验只支持兼容性结论：RS-Token index 可以与常规纠错编码组合使用。WebP/JPEG2000 结果只支持外部压力测试结论：未保护传统压缩 bitstream 在 BPSK 噪声信道下容易出现解码失败，不能被写成完整 codec-plus-channel-code 系统比较。
RemoteCLIP 有助于第一层，是因为它为遥感图像提供领域特定的语义几何结构。将这种几何结构蒸馏进 L0，使第一层 codebook 相比 reconstruction-only RVQ layer 更能区分航拍场景类别。但该结果并不意味着所有层都变成语义层，也不意味着仅靠 L0 就足以高保真重建。更准确地说，RS-Token 为不同层分配不同角色：L0 偏向任务语义，L1 到 L3 改善像素和感知细节。
教师消融可以进一步限定本文对 RemoteCLIP 的表述边界，但这些辅助数字不属于 v0.4 主表证据链。更安全的写法是：主表支持的是在 RemoteCLIP teacher 下进行 L0 视觉语言蒸馏能够显著改善任务路径；不同 teacher 之间的优劣应作为补充观察，而不是本文的主要 v0.4 结论。

蒸馏权重消融也提示语义对齐与信道鲁棒性之间可能存在权衡，但这些辅助数字不属于 v0.4 主表证据链。因此，本文采用的蒸馏权重应保守表述为 v0.4 实验中的主配置，而不是已经充分优化的普适最优选择。

从系统角度看，RS-Token 更适合作为物理层自适应之上的应用层表示，而不是替代 HARQ、自适应调制编码或 LDPC。链路较差或接收端只需要告警判断时，系统可以优先传输 L0 并走任务路径；链路改善或需要人工查看时，再请求 L1–L3 以提高重建质量。本文的 LDPC 实验说明这种 index 表示可以继续接入纠错保护，但如何联合优化层选择、重传、调制方式和信道编码，仍属于后续系统设计问题。

本文仍存在若干局限。实验使用 AID scene classification，而不是目标检测或语义分割，因此当前任务路径证据主要支持 image-level land-cover semantics。重建路径 sweep 只报告主模型种子，而任务路径结论由三个模型种子支持。本文没有进行严格 rate-matched 的传统 codec-plus-channel-code 系统比较；LDPC 实验使用 custom sparse LDPC code with min-sum decoding，不应描述为 5G NR LDPC comparison。

## 6. 结论
本文提出 RS-Token，一种面向遥感通信的层次化 RVQ tokenizer。通过 RemoteCLIP 蒸馏，第一层 token 获得更强任务判别性，后续层用于逐步细化重建。本文评估将 L0 任务路径证据与多层重建路径证据分开，并以普通 RVQ index 传输作为内部基线检验蒸馏贡献；同时用未保护 WebP/JPEG2000 码流和自定义 rate-1/2 systematic sparse LDPC 实验说明外部压力测试与信道编码兼容性边界。总体而言，结果支持一个保守但有用的结论：RemoteCLIP-guided layer specialization 能够为信道鲁棒遥感图像通信提供低码率任务保真与渐进式重建能力。




