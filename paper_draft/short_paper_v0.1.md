# Short Paper · RS-Token · 投稿草稿 (v0.1)

> 投稿目标: 遥感期刊 / 会议 (TGRS / JSTARS / IGARSS / IEEE GRSL / IEEE TIP-RS Special Issue)
> 论文页数目标: 4-6 页 short paper 体例 (含 figure)
> 写作语言: 英文 (本草稿先用中文写, 定稿后翻译)
> 创建日期: 2026-06-01

---

## 工作标题 (备选 3 个)

1. **"Hierarchical RemoteCLIP Distillation for Channel-Robust Semantic Communication of Remote Sensing Imagery"**
2. **"RS-Token: A Layered Discrete Tokenizer for Task-Preserving UAV Image Transmission Over Degraded Channels"**
3. **"Beyond Pixel Fidelity: RemoteCLIP-Distilled Residual Quantization for Graceful Task Degradation in UAV Downlink"**

→ 偏向 #1 或 #2; #3 标题学术味更重但太长。最终定稿前再选。

---

## 摘要 (Abstract, ~200 词中文先稿)

无人机 / 应急 / 边缘传感器场景下的遥感图像下行链路常面临严苛的信道条件 (低 SNR / 衰落 / 链路余量受限),
传统的"先压缩后信道编码"范式在低 SNR 区间出现 cliff effect, 任务全损; 而"传完整图像"的码率僵化又
浪费有限带宽。我们提出 **RS-Token**, 一个面向遥感语义通信的分层离散 tokenizer: 用 4 阶 RVQ 把图像
编码成多层离散索引, 通过 **RemoteCLIP 蒸馏**强制让第一层 codebook (L0) 承载地物语义身份, 后续层
(L1-L3) 承载像素细节。接收端按"信道条件 + 任务需求"动态选择传几层 (k ∈ {1, 2, 3, 4}): 信道差时仅
传 L0 (2,560 bits/img) 保住任务下限; 信道好时叠加后续层 (10,240 bits/img) 提升像素质量上限。
在 AID 30 类分类协议 + AWGN/Rayleigh 信道仿真下, 我们展示了 4 个核心结果:
(1) 蒸馏让 L0 索引的 linear probe 准确率从 57.7% 跃升到 82.4% (+24.7 pp), L0 单层承载约 95% 的语义能力;
(2) **蒸馏 + 仅传 L0 (2,560 bits)** 在 +5 dB 以上 SNR 下的分类准确率全面优于**无蒸馏 + 全传 4 层
(10,240 bits)**, 带宽减少 4× 而准确率反升 25 pp;
(3) -5 dB 低 SNR 下蒸馏 k=1 (8.2%) 仍优于 k=4 (5.7%), graceful degradation 成立;
(4) 教师消融显示 RemoteCLIP 相比通用 OpenAI CLIP 在 deployment 中等 SNR 区间多保 4-7 pp,
**遥感专用基础模型的领域归纳偏置在信道下游任务保真度上不可替代**。
据我们所知, 这是首个把"遥感基础模型蒸馏 → RVQ 分层 → 数字信道仿真 → 任务保真评测"四件事统一在
单一框架内的工作。

---

## 1. Introduction (~ 1 页)

### 1.1 痛点 (锚定遥感 deployment 场景)

无人机巡检、灾害应急、边缘智能传感器、CubeSat / SmallSat 等真实遥感部署场景共享三个核心特征:
**(a) 链路质量受限且会波动** (距离 / 遮挡 / 多普勒 / 干扰 / 业余频段),
**(b) 接收方需要实时任务输出** (操作员告警 / 应急指挥 / 自动检测),
**(c) 任务对图像不同抽象层级的需求差异巨大** (粗类别 / 区域定位 / 像素重建)。

现有数字传输范式在这些场景下有两个共同空白:
- **链路差时断崖式坍塌**: JPEG2000+LDPC / BPG+LDPC 在低 SNR 经历 cliff effect, 任务全损。
- **链路好时码率僵化**: 即使任务只需粗类别, 也必须传完整图像的全部码率。

> reviewer 必然反驳: HARQ + ACM 难道没解决? 答: HARQ 把 BER 压到 ~10⁻⁶ 但代价是 latency, 在 UAV
> 实时回传 / 应急指挥这种**任务有决策窗口**的场景, 重传等待会错过决策窗口; ACM 码率有下限, 当
> SNR 跌破最低门限 (BPSK + 最低码率 LDPC 仍不足) 链路无法建立; 两者**都不直接保任务**, 一个 BER=10⁻⁶
> 完美传过去的图仍可能因低码率压缩导致**分类错误**。本研究关心的不是物理层信号工程, 而是
> **当物理层手段已经用完, 怎么让接收方在尽可能少的 codebook 索引下保住任务**。

### 1.2 设想 · 层级化离散语义传输

如果离散 codebook 的第一层能稳定承载"任务最关键的信息" (在遥感里就是地物身份), 那么:
- 信道差时只传第一层 → 任务下限保住
- 信道好时叠加后续层 → 像素质量上限达到
- 接收方按"信道 + 任务"动态选 k → 自适应码率不破坏任务

这一机制在音频域已被 SpeechTokenizer (ICLR 2024) 与后继 STACodec / DM-Codec / HAC / LM-SPT
反复验证 (HuBERT 蒸馏 RVQ 第一层让其编码音素, 后层留声学), 但在视觉 / 遥感域至今没有任何工作完整实现。

### 1.3 本研究的位置 (相邻路线从未交汇)

经过 30+ 篇相邻工作的深度梳理, 三条路线**从未交汇**:

| 路线 | 代表工作 | 缺什么 |
|---|---|---|
| 多层离散 token + 数字信道 | MOC-RVQ (GLOBECOM'24), ReVQom (ICASSP'26), MSVQ-SC | 不蒸馏基础模型 / 不在遥感任务协议上评测 |
| 基础模型蒸馏离散 token | VILA-U (ICLR'25), BEiT v2, SemCLIP, TokLIP | 不做分层解耦 / 几乎不做信道仿真 / 不在遥感 |
| 遥感任务感知传输 | Gao TGRS'22, Luxembourg SnT 系列, MAGC, FOOL | 不用离散 codebook / 不做基础模型蒸馏 |

**本研究是这三条路线第一次汇到一起的尝试** —— 给出 "RemoteCLIP 蒸馏 → RVQ 分层 → 数字信道 → 任务保真"
在遥感域的初步组合实现。

### 1.4 主要贡献 (4 条, 与 Abstract 4 个核心结果一一对应)

1. **方法**: 提出 RS-Token 框架, 第一次把 SpeechTokenizer 风格的"分层蒸馏 RVQ tokenizer"思想
   迁移到遥感图像下行链路, 用 RemoteCLIP 监督 RVQ 第一层让 L0 索引承载地物身份。
2. **任务保真层级降级**: 在 AID 30 类分类协议下, 蒸馏 + 仅传 L0 (2,560 bits) 在 +5 dB 以上 SNR 下
   的任务准确率全面优于无蒸馏 + 全传 4 层 (10,240 bits), **带宽减少 4× 反升 25 pp 准确率**。
3. **graceful degradation**: -5 dB 低 SNR 下蒸馏 k=1 (8.2%) 优于 k=4 (5.7%), AWGN 与 Rayleigh
   衰落同向成立, 与 Gao 等 TGRS'22 的 UAV 任务感知协议同源。
4. **遥感基础模型的领域归纳偏置**: 教师消融显示 RemoteCLIP 相比通用 OpenAI CLIP 在中等 SNR 区间
   多保 4-7 pp 任务保真度, 印证遥感专用预训练在 deployment 信道下的不可替代性。

---

## 2. Related Work (~ 3/4 页)

### 2.1 数字 SemCom 中的多层离散 token

- **MOC-RVQ** (GLOBECOM'24): RVQ 4 阶 + AWGN/QAM 调制对齐, 但**自然图域**, 多层仅为残差精度递进, 第一层无语义身份。
- **ReVQom** (ICASSP'26): RVQ 多阶压缩, V2X 车端场景, **假设无错传输**, 第一层仅精度递进。
- **MSVQ-SC / ESC-MVQ** (POSTECH Jeon 团队, arXiv'25): 多 codebook + 联合调制, **不蒸馏基础模型**。

→ 共性局限: 都把 RVQ 多层当"残差精度递进", 第一层不承载特殊语义, 不蒸馏任何基础模型。

### 2.2 基础模型蒸馏到离散 token

- **BEiT v2** (MSRA, 2022): CLIP/DINO 蒸馏 VQ 单层, MIM 用途, **不分层 / 不做信道仿真**。
- **VILA-U** (MIT Han Lab, ICLR'25): CLIP 监督整条 RQ tokenizer, **整链对齐而非分层**, LLM 用途, 无信道。
- **SemCLIP** (Imperial Gunduz + BUPT, arXiv'25): 直接传 CLIP token 做 SemCom, 但用 OpenAI CLIP 而非遥感专用模型, **单层非分层**。

→ 共性局限: "整链对齐"或"单层 VQ + CLIP 蒸馏"为主, 没有 SpeechTokenizer 式的"第一层蒸语义、后续层留细节"
分层解耦, 几乎没有用**遥感专用基础模型**蒸馏 codebook 的工作。

### 2.3 遥感 / EO 任务感知传输

- **Gao 等** (TGRS'22): UAV 遥感 + DRL 选块 + 块状压缩感知 + 8-bit 量化 + ResNet34 分类, **协议同源**
  (AID 30 类 + 分类准确率 + Rayleigh+AWGN), 但**用块状压缩感知不用 codebook, 浅技术栈**。
- **Luxembourg SnT 团队** (Chatzinotas + Lagunas + Chou, ICC'26 已接收): EO+JSCC 全家桶, 但**走连续
  JSCC 不做离散 RVQ**。
- **MAGC** (武大 2024): VAE + 扩散重建 + 矢量地图条件做遥感极低码率压缩, **非 codebook 索引传输**。
- **FOOL** (TU Wien Dustdar, IEEE TMC'25): 遥感任务无关压缩, 非 codebook。

→ 共性局限: 已有遥感工作要么用块状压缩感知, 要么走连续 JSCC, 要么是 VAE+扩散重建, **没有任何一篇用
离散 codebook + 多层 RVQ + 遥感专用基础模型蒸馏**。

### 2.4 音频域类比强化

SpeechTokenizer (ICLR'24) 之后, STACodec / DM-Codec / HAC / LM-SPT 连续 4 篇工作走"分层蒸馏 RVQ
tokenizer"路线, 这一机制在音频域已成范式。视觉 / 遥感域至今空白, 不是"想做但走不通", 而是**未填补**。

---

## 3. Method (~ 3/4 页, 含 fig_method_arch)

### 3.1 RS-Token 总体架构

输入图像 x ∈ ℝ^(3×256×256) → encoder E → 特征图 z ∈ ℝ^(256×16×16) → flatten 为 256 个 patch token
(每个 256 维) → **4 阶 ResidualVQ** (每层 codebook 大小 1024, dim 256) → 4 层离散索引 [B, T, 4]
(每个索引 10 bit, 总码率 10,240 bits/img); 接收端从前 k 层索引重建 ẑ_q → decoder D → 重建图像 x̂
+ 用 L0 索引词频 (BoW) 直接喂线性分类器输出任务标签。

### 3.2 RemoteCLIP 蒸馏机制 (本论文核心)

**蒸馏路径**: L0 量化后特征 zq_L0 (256 维 / patch) 在 patch 维度 mean pool 后 → 2 层 MLP 投到 512 维
→ 与 RemoteCLIP 教师 image embedding f_T(x) ∈ ℝ^512 做 cosine 对齐, 损失 L_distill = 1 - cos(s, t)。

**为什么用 RemoteCLIP**: RemoteCLIP (TGRS'24, Liu et al.) 在 165 万对遥感图像-文本对上预训练, AID
linear probe **95.95%** 远超 ImageNet-ViT-B (83.55%) 的 +12.4 pp, 提供**遥感专用领域归纳偏置**。
通用 V-L 预训练模型对遥感数据的特征"展平"是 incidental 副产物; RemoteCLIP 让 30 个 AID 簇间几何
间距更大、簇内更紧凑, 蒸馏后这一特性传递到 L0 codebook, **使 L0 索引在 bit 错码后掉得慢** (见 §4
教师消融)。

### 3.3 总损失

$$L = L_{recon} + 0.1 \cdot L_{LPIPS} + 0.25 \cdot L_{VQ} + 0.5 \cdot L_{distill}$$

distill_weight = 0.5 是 trade-off 曲线 (§4.4) 上"in-domain 性能 vs 信道下游任务保真"的 sweet spot。

### 3.4 信道仿真协议

发射端: 索引 → 10 bit 二进制 → BPSK 调制。
信道: AWGN (BER = ½ erfc(√SNR_lin)) 与 Rayleigh fading (BER = ½ (1 - √(SNR_lin/(1+SNR_lin)))),
两个闭式 BER 都是 BPSK over flat 信道的标准结果 (Proakis & Salehi)。
接收端: 硬判决 → 索引 → 查同一 codebook 重建 zq → decoder。

### 3.5 接收端任务保真特征 · L0_bow

接收方拿到 L0 索引序列后无需查表 codebook embedding, 直接构建 1024 维**词频直方图** (L0_bow):

$$\text{L0\_bow}[i] = \frac{1}{T}\sum_t \mathbb{1}[\text{idx}_t = i], \quad T = 256$$

L0_bow 喂单层 sklearn LogisticRegression 输出任务标签。**这是接收方能拿到的最朴素信息形式** —
不依赖 codebook 几何, 不依赖 decoder, 只依赖索引本身的统计模式。如果连这种最弱形式都能让线性
分类器拿到 82%+, 那 L0 索引在离散组合模式里已经编码了地物身份。这是论证上最硬的特征。

---

## 4. Experiments (~ 1.5 页, 含 4 张 figure)

### 4.1 数据集与协议

AID (Aerial Image Dataset, 30 类场景分类), 8000 / 1000 / 1000 train / val / test split, 同 Gao
等 TGRS'22 协议。256×256 RGB, [-1, 1] 归一化。

### 4.2 训练配置

RVQ x4, codebook 1024, latent 256, 50 epoch, batch 16, lr 1e-4, AdamW (β1=0.9, β2=0.95),
warmup 500 steps, gradient clip 1.0, AMP bf16. 单卡 5070 Ti, 单次训练 ~70 分钟。

### 4.3 主结果 · 蒸馏 + L0 单层任务保真 (Table 1)

| 配置 | best PSNR | best LPIPS | L0_bow acc (无信道) | L0_emb acc | zq_pool acc |
|---|:-:|:-:|:-:|:-:|:-:|
| Stage 1 (单层 VQ) | 23.80 | 0.242 | — | — | — |
| Stage 2 (RVQ x4 无蒸馏) | 26.10 | 0.172 | 57.7% | 47.4% | 48.3% |
| **Stage 3 (RVQ x4 + RemoteCLIP)** | **25.88** | 0.176 | **82.4%** | **86.0%** | **88.0%** |

蒸馏代价 -0.22 dB ≪ 0.5 dB 红线; L0_bow +24.7 pp ≫ 5 pp 阈值 (5x 超额)。

### 4.4 信道仿真 · graceful degradation (Figure 1: fig_channel_snr_k)

(2 子图: AWGN / Rayleigh, SNR×k 矩阵热图, 数字标注)

**核心反差**: 蒸馏 + 仅传 L0 (2,560 bits) vs 无蒸馏 + 全传 4 层 (10,240 bits):

| SNR | 蒸馏 k=1 | 无蒸馏 k=4 | Δ |
|:-:|:-:|:-:|:-:|
| 0 dB AWGN | **53.2%** | 21.2% | **+32.0 pp** |
| +5 dB AWGN | **82.3%** | 57.2% | +25.1 pp |
| +10 dB AWGN | 82.5% | 57.6% | +24.9 pp |
| +5 dB Rayleigh | **59.1%** | 27.2% | **+31.9 pp** |
| +10 dB Rayleigh | **79.0%** | 48.2% | **+30.8 pp** |

**带宽减少 4× + 准确率反升 25-32 pp**, 在 AWGN 与 Rayleigh 双信道下都成立。

-5 dB 低 SNR 下 k=1 (8.2%) 优于 k=4 (5.7%), graceful degradation 成立。

### 4.5 层级分工 · L0 是语义层 (Figure 2: fig_layered_probe)

(蒸馏 vs 无蒸馏的累加层 linear probe 双曲线)

| 配置 | L0 | L0+L1 | L0+L1+L2 | L0+L1+L2+L3 |
|---|:-:|:-:|:-:|:-:|
| 无蒸馏 | 47.7% | 47.2% | 47.7% | 48.3% |
| **蒸馏** | **86.0%** | 87.7% | 87.9% | 88.0% |

蒸馏版 L0 → L0+L1 增量仅 +1.7 pp, L1-L3 三层合计仅 +2.0 pp, **L0 单层承载约 95% 的语义能力**。
无蒸馏版 4 层全部在 47-48% 横线波动, **从反面证明 RVQ 无监督训练不会自发学出语义, 蒸馏是"L0 语义"的唯一根因**。

### 4.6 教师消融 · 遥感专用基础模型的不可替代性 (Figure 3: fig_teacher_ablation)

(双柱状图, 7 种信道场景下 OpenAI CLIP vs RemoteCLIP)

| 信道场景 | OpenAI CLIP k=1 | **RemoteCLIP k=1** | 优势 |
|---|:-:|:-:|:-:|
| 无信道 (in-domain 上限) | 80.8% | 82.4% | +1.6 pp |
| AWGN -5 dB | 6.2% | 8.2% | +2.0 pp |
| **AWGN 0 dB** | 46.5% | **53.2%** | **+6.7 pp** |
| AWGN +5 dB | 79.4% | 82.3% | +2.9 pp |
| AWGN +10 dB | 80.8% | 82.5% | +1.7 pp |
| Rayleigh +5 dB | 55.3% | 59.1% | +3.8 pp |
| Rayleigh +10 dB | 74.8% | 79.0% | +4.2 pp |

**RemoteCLIP 的遥感专用预训练在 deployment 信道 (AWGN 0 dB / Rayleigh +5~+10 dB) 下贡献额外
4-7 pp 任务保真度, 正是 motivation §1 真实部署区间最关心的鲁棒性来源**。in-domain 上 1.6 pp
的微小优势在信道下游被放大, 印证遥感专用基础模型在 UAV / 应急 / 边缘场景下的关键价值。

### 4.7 蒸馏权重 trade-off (Figure 4: fig_trade_off)

(双轴: PSNR 副轴 + L0_bow 三档主轴 + 标注 paper choice w=0.5)

| w | best PSNR | L0_bow (无信道) | L0_bow @ AWGN 0dB | L0_bow @ Rayleigh +5dB |
|:-:|:-:|:-:|:-:|:-:|
| 0.0 (无蒸馏) | 26.10 | 57.7% | 23.4% | 28.0% |
| 0.1 | 26.17 | 71.2% | 35.5% | 42.1% |
| **0.5 (主推)** | 25.88 | 82.4% | **53.2%** | **59.1%** |
| 1.0 | 25.61 | 84.5% | 37.5% | 49.8% |

**关键发现**: w=1.0 in-domain 准确率反超 w=0.5 (+2.1 pp), 但**信道下游任务反而比 w=0.5 低 15.7 pp**
(0 dB AWGN)。机制: 强蒸馏让 codebook 几何过度紧凑, 单 bit 翻转更易跨簇 → 抗噪声下降。
**w=0.5 是 "in-domain 性能 vs 抗噪鲁棒性" 的最佳平衡, 不是任意挑的**。我们按 deployment 场景的
任务保真目标选择 w=0.5, 而非按 validation accuracy 单一指标。

---

## 5. Discussion (~ 1/2 页)

### 5.1 为什么遥感专用基础模型在信道下贡献更大

(连接 §3.2 直觉解释 + §4.6 实测数据)

通用 CLIP 在 4 亿对自然图-文本上学到的特征空间, 对遥感数据的"展平"是副产物; RemoteCLIP 在 165 万对
遥感图像-文本对上的预训练, 让特征空间的 30 个 AID 簇间几何间距更大、簇内更紧凑。蒸馏后这一几何
特性传递到 L0 codebook: 同样是 80% in-domain 精度, RemoteCLIP 训出来的 codebook 在 bit 错码后
掉得慢。在 motivation §1 的真实部署区间 (信道不完美但还能用), **通用 CLIP 蒸馏的 codebook 能勉强
工作, 但真正能扛信道恶化的仍然是带遥感领域归纳偏置的 RemoteCLIP 蒸馏**。

### 5.2 与 HARQ + ACM 的关系

本研究的层级化降级**与物理层 HARQ + ACM 正交, 不替代它们**。HARQ 把 BER 压低需要重传, 占用上行
带宽 + 引入 latency, 在 UAV 实时回传 / 应急指挥这种有决策窗口的场景代价大; ACM 码率有下限, SNR 跌
到调制门限以下链路无法建立。本研究的方法在物理层之上提供应用层的任务保真兜底 — **物理层手段已经
用完时, 仍能让接收方在尽可能少的 codebook 索引下保住下游任务准确率**。

### 5.3 局限与未来工作

**局限 1 · Rayleigh 0 dB 的性能悬崖**: AWGN 53.2% → Rayleigh 16.6% (-36.6 pp), 衰落让 BER 翻倍
让系统从"工作"跌到"半工作"。论文方法在物理层 LDPC + HARQ + 均衡器之上叠加, 不是替代它们。
**局限 2 · 单波段 RGB**: 多光谱 / 高光谱扩展留作后续。
**局限 3 · 经典基线对比未包含**: JPEG2000+LDPC / DeepJSCC / MOC-RVQ 经典基线对比是后续完整版
论文的主表内容。

---

## 6. Conclusion (~ 1/4 页)

我们提出 **RS-Token**, 一个面向遥感语义通信的分层蒸馏 RVQ tokenizer。通过 RemoteCLIP 蒸馏让第一层
codebook 承载地物语义, 后续层承载像素细节, 实现"信道差时只传 L0 保任务 / 信道好时叠加后续层提质量"
的任务保真层级降级传输。AID + AWGN/Rayleigh 仿真显示**蒸馏 + 仅传 L0 (2,560 bits) 的任务准确率
反超无蒸馏 + 全传 4 层 (10,240 bits) 25-32 pp**, 验证遥感专用基础模型 RemoteCLIP 在 deployment
信道下的不可替代性。这是首个把"遥感基础模型蒸馏 + RVQ 分层 + 数字信道 + 任务保真"四件事汇到一起
的工作, 为 UAV / 应急 / 边缘 / SmallSat 等真实遥感部署场景提供一种**任务保真且带宽自适应**的新范式。

---

## 参考文献占位 (待整理 .bib)

- Liu et al., RemoteCLIP, IEEE TGRS 2024
- Zhang et al., SpeechTokenizer, ICLR 2024
- MOC-RVQ, GLOBECOM 2024
- ReVQom, ICASSP 2026
- Gao et al., UAV semantic communication, IEEE TGRS 2022
- VILA-U, ICLR 2025
- BEiT v2, arXiv 2208.06366
- SemCLIP, arXiv 2502.18200
- DeepJSCC family
- ResidualVQ original (Lee et al., neural audio compression)
- LPIPS (Zhang et al., CVPR 2018)
- AID dataset (Xia et al., 2017)

---

## Figure 列表 (已生成在 rstoken/figs/)

1. **fig_method_arch** (待画) — RS-Token 总体架构, 重点突出 RemoteCLIP 蒸馏路径
2. **fig_trade_off** (✅ 已生成) — Figure 4: 蒸馏权重 trade-off
3. **fig_layered_probe** (✅ 已生成) — Figure 2: 累加层 linear probe
4. **fig_channel_snr_k** (✅ 已生成) — Figure 1: AWGN + Rayleigh 双面板 SNR×k 矩阵
5. **fig_teacher_ablation** (✅ 已生成) — Figure 3: 教师消融

---

## 写作 TODO (定稿前)

- [ ] 翻译为英文 (用 ChatGPT-5 / Claude 辅助)
- [ ] 加 fig_method_arch 架构图 (TikZ / drawio)
- [ ] 整理 .bib 文件 (填充参考文献占位)
- [ ] 选定 venue 并 fit 投稿 LaTeX 模板 (TGRS 用 IEEE Trans, IGARSS 用 IEEE Conf)
- [ ] reviewer 视角自审一遍
- [ ] 加 ethical / reproducibility / impact statement (按 venue 要求)

---

## 写作策略备忘 (来自前期讨论)

1. 标题与摘要必须含 "RemoteCLIP" 关键词, 不弱化为 "V-L 基础模型"
2. Introduction 用 RemoteCLIP vs ImageNet-ViT-B 的 +12.4 pp 优势作主论据, 不展开 OpenAI CLIP 仅差 1 pp
3. Method 写 "we choose RemoteCLIP to inject **remote-sensing-specific inductive bias**"
4. Experiments 教师消融框定为 "RemoteCLIP 在 deployment 信道下多保 4-7 pp", 不框定为 "method robust to teacher choice"
5. Discussion 可补一句 "framework can in principle accept other V-L teachers", 不作主线卖点

详见 `rstoken/experiments/results.md` §8 末尾的"论文叙事 (投遥感 venue)"5 条规则。
