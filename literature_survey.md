# 文献调研档案 · RemoteTokenizer 项目

> 本文件记录与本研究方向相关的所有已检索论文。
> 所有信息来源标注是 **「已 verify」** (经 WebFetch 抓取 abstract / 作者 / 发表渠道)
> 还是 **「仅 search snippet」** (只看到搜索结果摘要,详细信息需进一步 verify)。
> **不确定的字段统一标 "—" 或 "待 verify",绝不编造。**
>
> 最后更新: 2026-05-31

---

## 目录

1. [本研究的灵感来源(2 篇,机制层引用)](#一-本研究的灵感来源)
2. [核心 prior work 排查 · 5 篇候选(已 verify)](#二-核心-prior-work-排查--5-篇候选已-verify)
3. [深度检索新发现 · 强相关(应加进 prior work)](#三-深度检索新发现--强相关)
4. [深度检索新发现 · 中度相关](#四-深度检索新发现--中度相关)
5. [深度检索发现 · 机制类比(非通信场景)](#五-深度检索发现--机制类比非通信场景)
6. [DLR 系统性研究线(德国宇航中心)](#六-dlr-系统性研究线)
7. [次要参考与历史 baseline](#七-次要参考与历史-baseline)
8. [总览快查表](#八-总览快查表)
9. [风险提醒与后续行动](#九-风险提醒与后续行动)
10. [**第三轮深度多 agent 检索 (2026-05) · 新发现**](#十第三轮深度多-agent-检索-2026-05--新发现) ← **新增, ~20 篇**

---

## 一、本研究的灵感来源

### 1.1 SpeechTokenizer 【已 verify】

| 项 | 内容 |
|---|---|
| **完整标题** | SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models |
| **作者** | Xin Zhang, Dong Zhang, Shimin Li, Yaqian Zhou, Xipeng Qiu(复旦大学) |
| **发表** | **ICLR 2024** |
| **arXiv** | 2308.16692 |
| **核心机制** | RVQ + Encoder-Decoder + 用 HuBERT 蒸馏第一层 codebook,使 L1 承载音素语义,L2-L8 承载音色细节 |
| **与本研究的关系** | **直接灵感来源**——本研究将 SpeechTokenizer 的"蒸馏第一层 codebook"招式从语音域搬到遥感图像域 |
| **与本研究的差异** | 应用域不同(语音 → 遥感图像),教师不同(HuBERT → RemoteCLIP),信道仿真不同(无 → AWGN+BPSK) |
| **可信度** | ✓ 完整 verify |

### 1.2 RemoteCLIP 【已 verify】

| 项 | 内容 |
|---|---|
| **完整标题** | RemoteCLIP: A Vision Language Foundation Model for Remote Sensing |
| **作者** | Fan Liu (河海) ‡, Delong Chen (港科大 HKUST) ‡, Zhangqingyun Guan, Xiaocong Zhou, Jiale Zhu (河海), Qiaolin Ye (南京林业), Liyong Fu (中国林科院), Jun Zhou (Griffith Univ)。‡ 共同一作 + 通讯 |
| **发表** | **IEEE TGRS 2024**(SCI 一区遥感顶刊) |
| **arXiv** | 2306.11029 · IEEE Xplore 10504785 |
| **预训练数据** | 828,725 图文对(165,745 张 × 5 caption,~12× 现有遥感图文总和) |
| **关键数字 (Table IV linear probe top-1)** | RemoteCLIP ViT-B-32: AID 95.95% / RESISC45 94.27%<br>CLIP ViT-B-32: 94.95% / 92.60%<br>ImageNet ViT-Base: 83.55% / 86.89% |
| **开源 ckpt** | HuggingFace `chendelong/RemoteCLIP`(ViT-B-32 / B-16 / L-14 三档) |
| **与本研究的关系** | **唯一外部论据来源 + Stage 3 蒸馏教师**——双角色 |
| **可信度** | ✓ 完整 verify |

---

## 二、核心 prior work 排查 · 5 篇候选(已 verify)

### 2.1 MOC-RVQ 🔶 技术路线最接近 【已 verify】

| 项 | 内容 |
|---|---|
| **完整标题** | MOC-RVQ: Multilevel Codebook-Assisted Digital Generative Semantic Communication |
| **作者** | Yingbin Zhou, Yaping Sun, Guanying Chen, Xiaodong Xu, Hao Chen, Binhong Huang, Shuguang Cui, Ping Zhang(具体单位未抓到) |
| **发表** | **GLOBECOM 2024**(IEEE 全球通信会议,已接收) |
| **arXiv** | 2401.01272(2024-01-02) |
| **代码** | github.com/Albert2X/moc_rvq |
| **核心机制** | 多头八进制 codebook (MOC) 缩小索引范围 + RVQ 多层量化 + Swin-Transformer 去噪块 + 数字星座调制 + 模拟 JSCC 对照 |
| **数据集** | 2K 高分辨率图像测试集(自然图,**非遥感**) |
| **与本研究重合** | RVQ + codebook 索引传输 + AWGN 数字信道 + 多层量化整体框架 |
| **关键差异** | (1) **应用域** 自然图 → 遥感;(2) **层级机制** 仅残差精度递进,**无任何语义蒸馏** |
| **学术权重** | ⭐⭐⭐ 已会议接收,本研究最强对手 |
| **可信度** | ✓ 完整 verify |

### 2.2 ResiTok 【已 verify】

| 项 | 内容 |
|---|---|
| **完整标题** | ResiTok: A Resilient Tokenization-Enabled Framework for Ultra-Low-Rate and Robust Image Transmission |
| **作者** | Zhenyu Liu, Yi Ma, Rahim Tafazolli(从作者推测可能是英国萨里大学 Surrey,**待 verify**) |
| **发表** | arXiv preprint(2025-05-03,**未见正式接收**) |
| **arXiv** | 2505.01870 |
| **arXiv Comments 字段** | 无(没有"Accepted by XXX"标注) |
| **核心机制** | 1D tokenizer + 分层 key tokens / detail tokens + zero-out 训练模拟 token 丢失 + channel-adaptive coding/modulation |
| **数据集** | abstract 未明示具体数据集 |
| **信道** | "channel-adaptive coding and modulation",abstract 未明示具体信道模型 |
| **与本研究重合** | 分层 + 鲁棒传输 + 极低码率 + 截断传输概念 |
| **关键差异** | (1) **不是 RVQ**(用 1D tokenizer + zero-out);(2) **无基础模型蒸馏**;(3) 通用图像非遥感 |
| **学术权重** | ⭐⭐ 占坑级 preprint |
| **可信度** | ✓ 完整 verify |

### 2.3 VQ-VAE + OFDM 【已 verify】

| 项 | 内容 |
|---|---|
| **完整标题** | VQ-VAE Based Digital Semantic Communication with Importance-Aware OFDM Transmission |
| **作者** | Ming Lyu, Hao Chen, Dan Wang, Chen Qiu, Guangyin Feng, Nan Ma, Xiaodong Xu(具体单位未抓到) |
| **发表** | arXiv preprint(2025-08-12 提交,2025-12-29 修订,**未见正式接收**) |
| **arXiv** | 2508.08686 |
| **arXiv Comments 字段** | "6 pages, 5 figures, conference"(投会议格式,未指定) |
| **核心机制** | VQ-VAE 共享 codebook(收发两端共用)+ 重要性感知 OFDM(梯度估计重要性 → 关键特征近参考信号)+ 接收端 codebook 重匹配纠错 |
| **数据集** | abstract 未明示 |
| **信道** | OFDM + 重要性感知子载波分配 |
| **与本研究重合** | 离散 codebook + 数字信道 + 索引传输 |
| **关键差异** | (1) **单层 VQ-VAE**(没有 RVQ 多层概念);(2) **无第一层语义化机制**;(3) 通用图像非遥感专用 |
| **学术权重** | ⭐⭐ 占坑级 preprint |
| **可信度** | ✓ 完整 verify |

### 2.4 MAGC 【已 verify】

| 项 | 内容 |
|---|---|
| **完整标题** | Map-Assisted Remote-Sensing Image Compression at Extremely Low Bitrates |
| **作者** | Yixuan Ye, Ce Wang, Wanjie Sun, Zhenzhong Chen(GitHub 句柄 `WHUyyx` 暗示武汉大学,**未在 abstract 页明示**) |
| **发表** | arXiv preprint(2024-09-03,**未见正式接收**) |
| **arXiv** | 2409.01935 |
| **arXiv Comments 字段** | 无 |
| **代码** | github.com/WHUyyx/MAGC(数据集和代码计划公开) |
| **核心机制** | 两阶段框架:VAE 压缩 latent + 预训练扩散模型重建 + 矢量地图(vector maps)做语义/结构条件 |
| **数据集** | abstract 未明示具体名 |
| **信道** | **未指定信道模型**(只提"边缘设备存储 + 窄带宽传输"动机) |
| **与本研究重合** | **遥感图像**(同应用域) + 极低码率 |
| **关键差异** | (1) **VAE+扩散重建**,传连续 latent 不传 codebook 索引;(2) 用地图条件而非语义蒸馏;(3) 没做信道仿真 |
| **学术权重** | ⭐⭐ 占坑级 preprint |
| **可信度** | ✓ 完整 verify |

### 2.5 FM-SemCom 【已 verify】

| 项 | 内容 |
|---|---|
| **完整标题** | Foundation Model-Based Adaptive Semantic Image Transmission for Dynamic Wireless Environments |
| **作者** | Fangyu Liu, Peiwen Jiang, Wenjin Wang, Chao-Kai Wen, Shi Jin, Jun Zhang(具体单位未抓到) |
| **发表** | arXiv preprint(2025-09-28,**未见正式接收**) |
| **arXiv** | 2509.23590 |
| **arXiv Comments 字段** | 无 |
| **核心机制** | 图像分解为分割图 + 压缩表示 + 任务感知优先级 + 双扩散模型(信道估计知识图 CEKM + 接收端重建) |
| **数据集** | **BDD100K**(自动驾驶) |
| **信道** | QuaDRiGa 多场景无线信道 + CEKM 信道估计 |
| **与本研究重合** | 用基础模型 + 通信场景 + 任务感知 |
| **关键差异** | (1) 基础模型作**特征提取器/分割模型** ≠ 蒸馏教师(运行时 forward 大模型,推理成本高);(2) 自动驾驶域非遥感;(3) 不是 codebook 索引传输 |
| **学术权重** | ⭐⭐ 占坑级 preprint |
| **可信度** | ✓ 完整 verify |

---

## 三、深度检索新发现 · 强相关

### 3.1 ReVQom 🚨 新强敌 【已 verify】

| 项 | 内容 |
|---|---|
| **完整标题** | Residual Vector Quantization for Communication-Efficient Multi-Agent Perception (ReVQom) |
| **作者** | Dereje Shenkut, B.V.K Vijaya Kumar(B.V.K Vijaya Kumar 是 CMU 著名教授,**未在 abstract 页明示单位**,推测来自 **CMU**) |
| **发表** | **ICASSP 2026 接收**(Comments 字段明示"Accepted at ICASSP 2026, 5 pages") |
| **arXiv** | 2509.21464(2025-09) |
| **核心机制** | bottleneck 降维网络 + 多阶 RVQ + 仅传 per-pixel code indices,**用于多智能体感知** |
| **数据集** | **DAIR-V2X**(车路协同感知真实数据集) |
| **场景** | 多智能体协同感知(自动驾驶 / UAV / 机器人) |
| **关键数字** | 8192 bpp(原始 32-bit float 特征)→ 6-30 bpp,V2X 上达到最高 1365× 压缩,中等码率下匹配或超越 raw-feature 协同感知 |
| **与本研究重合** | **多阶 RVQ + 索引传输 + 任务保真**(都用 RVQ 做特征压缩) |
| **关键差异** | (1) 多智能体协同感知场景,**不是遥感卫星下行**;(2) **没用基础模型蒸馏**——只是端到端学习 RVQ;(3) 没有"第一层语义化"的设计;(4) RVQ 多阶仅是精度递进 |
| **学术权重** | ⭐⭐⭐ ICASSP 2026 已接收,RVQ-通信压缩的新强对手 |
| **可信度** | ✓ 完整 verify |
| **建议行动** | **强烈建议加进 prior work 排查页**——它是 RVQ 路线下最近被接收的工作 |

### 3.2 Luxembourg Semantic-Loss 【已 verify · 之前误标 DLR】

> ⚠ **重要修正**:之前我把这篇标成"DLR(德国宇航中心)"工作,**实际不是**。
> 真实作者团队是 **Luxembourg University 的 Symeon Chatzinotas 组**(SnT 卫星通信中心)。
> Chatzinotas 是卫星通信领域知名学者,这个组同时做了"LEO JSCC EO"(§4.4)那篇工作。

| 项 | 内容 |
|---|---|
| **完整标题** | A Semantic-Loss Function Modeling Framework With Task-Oriented Machine Learning Perspectives |
| **作者** | Ti Ti Nguyen, Thanh-Dung Le, Vu Nguyen Ha, Hong-fu Chou, Geoffrey Eappen, Duc-Dung Tran, Hung Nguyen-Kha, Prabhu Thiruvasagam, Luis M. Garces-Socarras, Jorge L. Gonzalez-Rios, Juan C. Merlano-Duncan, **Symeon Chatzinotas**(单位未在 abstract 页明示,但 Chatzinotas 组在 Luxembourg University SnT) |
| **发表** | arXiv preprint(2025-03,**未见正式接收**)Comments 字段仅"6 pages, 11 figures" |
| **arXiv** | 2503.09903 |
| **核心机制** | 数据拟合框架 + 经验性建模 EO 系统的"语义损失"+ 拆分 source-coding loss(数据质量)与 transmission loss(对照 Shannon 极限)+ 用下游任务准确率验证 |
| **任务模型** | EfficientViT, MobileViT, ResNet50-DINO, ResNet8-KD |
| **数据集** | abstract 仅泛说"real-world EO datasets / lossy image datasets",**未指定** |
| **压缩方案** | abstract 未列具体编解码器(无 JPEG2000、学习式压缩等具体名字) |
| **与本研究重合** | **EO 任务保真评测** 的范式建立工作 |
| **关键差异** | **不做 codebook**,只研究"压缩失真 → 任务损失"的关系建模 |
| **学术权重** | ⭐⭐ Luxembourg 研究线的一部分 |
| **可信度** | ✓ 完整 verify |
| **修正备注** | 同作者 Hung Nguyen-Kha / Symeon Chatzinotas 也出现在 **LEO JSCC EO**(§4.4)那篇里——两篇是 Luxembourg 同组的连续工作 |

---

## 四、深度检索新发现 · 中度相关

### 4.1 SFSC · Semantic Forwarding and Codebook-Enhanced Model Division Multiple Access 【已 verify】

| 项 | 内容 |
|---|---|
| **完整标题** | Semantic Forwarding and Codebook-Enhanced Model Division Multiple Access for Satellite-Terrestrial Networks |
| **作者** | Jinghong Huang, Mengying Sun, Xiaodong Xu, Jianchi Zhu, Zechuan Fang, Jingxuan Zhang, Ruichen Zhang, Chen Dong, Ping Zhang, Dusit Niyato(单位未在 abstract 页明示) |
| **发表** | arXiv preprint(2026-03-03 提交,**未见 Comments 字段、未见接收**) |
| **arXiv** | 2603.02536(预占编号) |
| **核心机制** | 联合语义编码/调制(VQ + jointly optimized semantic codebook)+ 卫星端语义转发 + FiLM 信道感知重建 + Codebook Split-enhanced MDMA(CS-MDMA)多用户接入 |
| **数据集** | abstract 未明示 |
| **信道** | 卫星-地面通信链路(高路径损耗、有限频谱、时变信道、低 SNR 工况)+ 中继-卫星(转发)拓扑 |
| **评测** | 仅报 PSNR(低 SNR 比 baseline 高 ~7.9 dB);**未提任务保真/分类准确率** |
| **量化** | abstract 描述"vector-quantized joint semantic coding";**未提 RVQ**(疑似单层 VQ) |
| **资助** | NSFC No. 62401074 / U24B20131 + 北京自然科学基金 L242012 |
| **与本研究重合** | 卫星 + codebook + 通信场景 |
| **关键差异** | (1) 多址接入侧重传输调度,不是任务保真;(2) 单层 VQ(非 RVQ);(3) 评测仅 PSNR |
| **可信度** | ✓ 完整 verify |

### 4.2 Scalable Data Transmission Framework for EO Satellites 【已 verify】

| 项 | 内容 |
|---|---|
| **完整标题** | Scalable Data Transmission Framework for Earth Observation Satellites with Channel Adaptation |
| **作者** | Van-Phuc Bui, Shashi Raj Pandey, Israel Leyva-Mayorga, Petar Popovski(单位未在 abstract 页明示;Petar Popovski 知名于 **Aalborg 大学**) |
| **发表** | arXiv preprint(2024-12,**未见 Comments 字段、未见接收**) |
| **arXiv** | 2412.11857 |
| **核心机制** | 多光谱 EO 图像下行框架 + 语义通信原则优先级 + 关键信息(变化的多光谱像素)优先 + 信道自适应 |
| **数据集** | abstract 仅说"real dataset",**未指定具体名** |
| **信道** | 卫星-地面下行,rate-limited,动态调整 + simulated link |
| **量化** | abstract 未明示是 codebook 索引 还是 连续值 JSCC |
| **与本研究重合** | EO + 信道自适应 + 优先级机制 |
| **关键差异** | (1) 优先级基于"变化的多光谱像素"而非基础模型语义;(2) 是否离散索引未明示 |
| **可信度** | ✓ 完整 verify |

### 4.3 FMSAT · Semantic Satellite Communications Based on Generative Foundation Model 【已 verify】

| 项 | 内容 |
|---|---|
| **完整标题** | Semantic Satellite Communications Based on Generative Foundation Model |
| **简称** | FMSAT(论文里自己用的简称) |
| **作者** | Peiwen Jiang, Chao-Kai Wen, Xiao Li, Shi Jin, Geoffrey Ye Li(单位未在 abstract 页明示;**Shi Jin / Peiwen Jiang 也出现在 §2.5 FM-SemCom 那篇里**——同团队连续工作) |
| **发表** | arXiv preprint(2024-04,Comments: "This work has been submitted to the IEEE for possible publication") |
| **arXiv** | 2404.11941 |
| **核心机制** | FM 驱动的分割 + 重建框架 + 自适应编码器/解码器 + 卫星端错误检测器(减少长传播延迟下的重传) |
| **数据集** | abstract 未指定 |
| **信道** | 卫星通信链路,含再生卫星 + 网关,处理雨衰 / 长传播延迟 / 同信道干扰 / 高速移动 |
| **基础模型角色** | **运行时特征提取器/分割/重建** ≠ 蒸馏教师(类似 FM-SemCom 模式) |
| **与本研究重合** | 卫星通信 + 基础模型 |
| **关键差异** | 基础模型作运行时分割/重建,**不是蒸馏教师**(同 FM-SemCom 路线) |
| **可信度** | ✓ 完整 verify |
| **重要发现** | **与 §2.5 FM-SemCom 是同一团队(Peiwen Jiang / Chao-Kai Wen / Shi Jin)的连续工作**——他们在 2024 年发了 FMSAT(卫星场景),2025 年发了 FM-SemCom(自动驾驶场景),都用基础模型作分割/重建 |

### 4.4 LEO JSCC EO · A Joint JSCC-Resource Allocation Framework 【已 verify】

| 项 | 内容 |
|---|---|
| **完整标题** | A Joint JSCC-Resource Allocation Framework for QoS-Aware Semantic Communication in LEO Satellite-based EO Missions |
| **作者** | Hung Nguyen-Kha, Ti Ti Nguyen, Vu Nguyen Ha, Eva Lagunas, Symeon Chatzinotas, Bjorn Ottersten(**Luxembourg University SnT 中心**——与 §3.2 同组) |
| **发表** | **IEEE ICC 2026 接收**(Comments 字段明示"Accepted for publishing in the proceeding IEEE ICC 2026") |
| **arXiv** | 2603.12027 |
| **核心机制** | LEO 卫星 EO 高分辨图像下行 + JSCC 联合编码 + 混合整数-连续优化最小化发射功率(满足图像质量 QoS 约束)+ 曲线拟合的 compression-SNR-quality 模型 + JCRRA 算法 |
| **量化方式** | 描述围绕 JSCC + 可调"compression ratio" + compression-SNR-quality 关系 → **更接近连续值 JSCC**,不是离散 codebook(待 PDF 进一步 verify) |
| **场景** | LEO 卫星 EO 任务 + 受限星上功率预算 + 卫星动态 |
| **信道** | abstract 未给具体衰落/路径损耗/Doppler 模型,只通过 SNR 表征链路质量 |
| **与本研究重合** | LEO + EO + JSCC |
| **关键差异** | **连续值 JSCC**(而非离散 codebook 索引);侧重资源分配/QoS,不是任务保真 |
| **学术权重** | ⭐⭐⭐ ICC 2026 已接收,Luxembourg 组的卫星 EO 工作 |
| **可信度** | ✓ 完整 verify |
| **重要发现** | **与 §3.2 Semantic-Loss 是 Luxembourg 同组连续工作**(共享 Hung Nguyen-Kha / Ti Ti Nguyen / Vu Nguyen Ha / Symeon Chatzinotas) |

### 4.5 Discernment · Discrete-Space Generative AI Pipeline 【已 verify】

| 项 | 内容 |
|---|---|
| **完整标题** | Discrete-Space Generative AI Pipeline for Semantic Transmission of Signals |
| **简称** | Discernment |
| **作者** | Silvija Kokalj-Filipovic, Yagna Kaasaragadda(单位未在 abstract 页明示) |
| **发表** | arXiv preprint(2026-02-14 提交,**未见 Comments 字段、未见接收**) |
| **arXiv** | 2602.13556 |
| **核心机制** | 离散空间生成 AI + 自适应在自回归模型与扩散模型之间切换 + 在不同 erasure pattern 下保持语义完整性 |
| **信号类型** | **基带射频(baseband radio)+ 音频** —— **不是图像** |
| **信道** | erasure 信道 |
| **任务保真评测** | ✓ 报告分类准确率 + 重建语义的统计保真度 |
| **与本研究重合** | 离散空间 + 信道下保任务 + 降级特性 |
| **关键差异** | (1) 信号类型 是 RF 信号 / 音频,**不是图像**;(2) 不是遥感场景 |
| **可信度** | ✓ 完整 verify |

---

## 五、深度检索发现 · 机制类比(非通信场景)

### 5.1 SemHiTok · Semantic-Guided Hierarchical Codebook 【已 verify】

| 项 | 内容 |
|---|---|
| **完整标题** | SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook for Multimodal Understanding and Generation |
| **作者** | Zisheng Chen, Chunwei Wang, Runhui Huang, Hongbin Xu, Xiuwei Chen, Jun Zhou, Jianhua Han, Hang Xu, Xiaodan Liang(单位未在 abstract 页明示;**Xiaodan Liang 在 中山大学**, Hang Xu 在 华为诺亚实验室——多机构合作) |
| **发表** | **ICLR 2026 接收**(Comments 字段明示) |
| **arXiv** | 2503.06764(2025-03) |
| **核心机制** | **语义引导的层级 codebook**——pixel 子 codebook 建在预训练 semantic codebook 之上 + 结构与训练策略上分离语义和像素 + 集成到 LLaVA-v1.5 MLLM |
| **数据集** | LLaVA-v1.5 setting 评测 |
| **场景** | **多模态图像理解与生成**,**没有通信/信道仿真** |
| **与本研究的关系** | **机制层最相似**——都用"语义引导多层 codebook" |
| **关键差异** | (1) 自然图像多模态生成场景,**非通信场景**;(2) 不是遥感;(3) 没做信道下任务保真评测 |
| **学术权重** | ⭐⭐⭐ ICLR 2026 已接收,**机制学的同期最重要工作** |
| **建议** | 论文方法学讨论时**应当引用**作为"层级 codebook + 语义引导"的同期工作,体现"层级语义 codebook"是 2025-2026 多个领域的共同热点 |
| **可信度** | ✓ 完整 verify |

### 5.2 LM-SPT · LM-Aligned Semantic Distillation for Speech Tokenization 【已 verify】

| 项 | 内容 |
|---|---|
| **完整标题** | LM-SPT: LM-Aligned Semantic Distillation for Speech Tokenization |
| **作者** | Daejin Jo, Jeeyoung Yun, Byungseok Roh, Sungwoong Kim(单位未在 abstract 页明示) |
| **发表** | arXiv preprint(2025-06-20,**未见 Comments 字段、未见接收**) |
| **arXiv** | 2506.16738 |
| **核心机制** | **从语义 token 重建音频 → 用 frozen ASR encoder 对齐重建波形特征与原波形特征** + 多帧率支持(25Hz/12.5Hz/6.25Hz) |
| **改进对象** | abstract 未明确点名 SpeechTokenizer,**只泛说"prior methods using SSL teachers such as HuBERT"** |
| **对齐机制** | 对齐 frozen **ASR encoder**(不是 LM,虽然名字叫 LM-SPT) |
| **RVQ 使用** | abstract **未明示是否用 RVQ**,只说"semantic quantizer" |
| **与本研究的关系** | 语音域同期工作,**机制类比**——也是用预训练教师做语义蒸馏 |
| **建议** | 写论文时可作为"我们的方法学不是孤例,语音域同期也在做"的旁证 |
| **可信度** | ✓ 完整 verify |

### 5.3 Hierarchical Codec Diffusion (语音生成) 【仅 search snippet · 弱相关】

| 项 | 内容 |
|---|---|
| **完整标题** | Hierarchical Codec Diffusion for Video-to-Speech Generation |
| **arXiv** | 2604.15923 |
| **与本研究的关系** | 语音生成,机制类比但场景不重合 |
| **可信度** | ⚠ 仅 snippet,弱相关 |

---

## 六、其他 EO 研究线

> ⚠ **修正记录**:之前曾把这一章标为"DLR 系统性研究线",**这是错的**。
> 经 WebFetch 验证, 这里收录的论文实际**不是统一团队的工作**:
> - §6.1 **Compressed Learning** 的作者是 Protim Bhattacharjee, Peter Jung,通过 SPIE proceedings 发表(DOI 10.1117/12.3031472),与 DLR 的关系待 verify
> - §6.2 **Rate-Distortion EO** 来自 elib.dlr.de(DLR 内部出版库),可能确实是 DLR 工作,但需进一步 verify
> - 之前误归在此章的 §6.3 **Semantic-Loss** 实际是 **Luxembourg University Chatzinotas 组**,**已移至 §3.2**
>
> 因此本章只是"其他 EO 语义压缩工作"的杂项分类,不代表统一研究线。

### 6.1 Compressed Learning 【已 verify】

| 项 | 内容 |
|---|---|
| **完整标题** | Compressed learning based onboard semantic compression for remote sensing platforms |
| **作者** | Protim Bhattacharjee, Peter Jung(单位未在 abstract 页明示) |
| **发表** | SPIE proceedings(Related DOI 10.1117/12.3031472)/ arXiv 2409.01988(2024-09) |
| **核心机制** | 学习式稀疏压缩矩阵在星上编码 + unrolled NA-ALISTA + wavelet 稀疏先验 + 联合 fine-tune 编码矩阵/解码器/分类器 |
| **数据集** | abstract 未指定具体数据集(只说下游任务是图像分类) |
| **信道** | 建模信道("camera noise + communication channel"),编码后 downlink |
| **量化** | abstract 未提任何量化方案(无 RVQ 无 VQ) |
| **与本研究的关系** | EO 星上压缩 + 信道下任务保真 |
| **关键差异** | (1) 压缩感知风格的稀疏矩阵编码 ≠ codebook 索引;(2) 无 RVQ;(3) 无基础模型蒸馏 |
| **可信度** | ✓ 完整 verify |

### 6.2 Rate-Distortion EO 【仅 search snippet · 待 verify】

| 项 | 内容 |
|---|---|
| **完整标题** | Rate-distortion trade-off for learned semantic compression for remote sensing platforms |
| **来源** | elib.dlr.de/217046(DLR 内部出版库) |
| **机构** | DLR |
| **核心机制** | EO 平台 + 学习式语义压缩 + 率失真权衡 |
| **可信度** | ⚠ 仅 snippet |

### 6.3 Semantic-Loss Function Modeling

(原误归在此,实际是 Luxembourg University Chatzinotas 组工作,**已移至 §3.2**)

---

## 七、次要参考与历史 baseline

> 以下论文与本研究方向部分相关或属于早期 baseline,**仅记录,不必加进 prior work 排查页**。

### 7.1 经典/历史方法

| 简称 | 标题(部分) | arXiv | 与本研究的关系 |
|---|---|---|---|
| Masked VQ-VAE SemCom | Robust Semantic Communications with Masked VQ-VAE Enabled Codebook | 2206.04011 | 早期 VQ-VAE SemCom |
| OFDM-SemCom | OFDM-Based Digital Semantic Communication with Importance Awareness | 2401.02178 | OFDM 语义通信先驱 |
| Robust VQ + OFDM | Robust Vector Quantized-Enabled Digital Semantic Communication With OFDM Transmission | 2602.15045 | VQ + OFDM 鲁棒传输 |
| Vector Quantized + Channel Adaptive | Vector Quantized-Enabled Digital Semantic Communication with Channel Adaptive Image Transmission | 2508.03740 | VQ + 信道自适应 |
| Multi-Codebook VQ | End-to-End Semantic Communication With Multi-Codebook Vector Quantization | 2504.11709 | 多 codebook VQ |
| ViT-Importance-Aware | Vision Transformer-based Semantic Communications With Importance-Aware Quantization | 2412.06038 | ViT + 重要性感知量化 |
| MIMO-OFDM SemCom | Importance-Aware Semantic Communication in MIMO-OFDM Systems Using Vision Transformer | 2508.07696 | MIMO-OFDM + ViT |

### 7.2 卫星/EO 通信场景

| 简称 | 标题(部分) | arXiv | 与本研究的关系 |
|---|---|---|---|
| Semantic Image-Adaptive Sat | Semantic Image-Adaptive Transmission in Satellite–Ground Scenario | MDPI Sensors 25(1)/269 | 卫星-地面场景语义传输 |
| LEO Satellite EO | Semantic Enabled 6G LEO Satellite Communication for Earth Observation | 2408.03959 | 6G LEO + EO |
| Cognitive Semantic LEO | Cognitive Semantic Augmentation LEO Satellite Networks for EO | 2410.21916 | LEO 认知语义增强 |
| LEO Adaptive Rate | Adaptive Rate Control for Semantic Communications over LEO Satellite-to-Ground Links | 2605.10095 | LEO 自适应速率 |
| SAGIN DRL | Deep Reinforcement Learning-Based Resource Allocation for Hybrid Bit and Generative Semantic Communications in Space-Air-Ground Integrated Networks | 2412.05647 | 空天地一体化 + DRL |
| On-board Change Detection | On-board Change Detection for Resource-efficient Earth Observation with LEO Satellites | 2305.10119 | 星上变化检测 |
| On-Air DL EO | On-Air Deep Learning Integrated Semantic Inference Models for Enhanced Earth Observation Satellite Networks | 2409.15246 | EO 卫星网络深度推理 |

### 7.3 LLM/VLM 驱动的图像传输

| 简称 | 标题(部分) | arXiv | 与本研究的关系 |
|---|---|---|---|
| Text-Guided Token | Text-Guided Token Communication for Wireless Image Transmission | 2507.05781 | 文本引导 token 传输 |
| LLM SemCom | Large Language Model-Based Semantic Communication System for Image Transmission | 2501.12988 | LLM 驱动 SemCom |
| Adaptive Bitrate Video | Adaptive Bitrate Video Semantic Communication over Wireless Networks | 2308.00531 | 视频自适应码率 |
| World Model SemCom | Semantic Communications with World Models | 2510.24785 | 世界模型 SemCom |
| Distributionally Robust LAM | Distributionally Robust Wireless Semantic Communication with Large AI Models | 2506.03167 | 大 AI 模型 + 鲁棒性 |
| Adaptive Wireless ASCViT | Adaptive Wireless Image Semantic Transmission (ASCViT-JSCC) | 2410.17536 | ViT-JSCC 自适应 |
| Position-Aided | Position-Aided Semantic Communication for Efficient Image Transmission | 2410.18364 | 位置辅助 SemCom |
| Adaptive Generative FM | Adaptive Semantic Image Transmission Using Generative Foundation Model | IEEE 10734720 | 生成式 FM 自适应 |
| Resi-VidTok | An Efficient and Decomposed Progressive Tokenization Framework for Ultra-Low-Rate and Lightweight Video Transmission | 2510.25002 | ResiTok 视频版扩展 |

### 7.4 与本研究有关的其他视觉 tokenizer 工作

| 简称 | 标题(部分) | arXiv | 与本研究的关系 |
|---|---|---|---|
| Unified VLM Discrete | Modeling Unified VLM through Semantic Discrete Encoding | 2411.17762 | 统一 VLM + 离散编码 |
| MLLM Tokenization | Towards Semantic Equivalence of Tokenization in Multimodal LLM | 2406.05127 | MLLM 视觉 tokenization |
| Generative Ultra-Low | Generative Semantic Coding for Ultra-Low Bitrate Visual Communication and Analysis | 2510.27324 | 生成式极低码率 |
| Implicit Codebooks | Residual Quantization with Implicit Neural Codebooks | 2401.14732 | RQ 隐式 codebook |
| RFSQ | Robust Residual Finite Scalar Quantization for Neural Compression | 2508.15860 | 鲁棒残差有限标量量化 |

### 7.5 EO 数据集 / 任务工具

| 简称 | 标题(部分) | arXiv | 与本研究的关系 |
|---|---|---|---|
| RESISC45 原数据集 | Remote Sensing Image Scene Classification | 1703.00121 | RESISC45 原论文,测试集来源之一 |
| ATMformer | An Adaptive Token Merging Vision Transformer for Remote Sensing Image Scene Classification | RG 389061885 | RESISC45 + AID + EuroSAT 上的 ViT 方法 |
| EarthDial | Turning Multi-sensory Earth Observations to Interactive Dialogues | 2412.15190 | 多感官 EO 对话(EarthDial) |
| RS-Foundation Genealogy | A Genealogy of Foundation Models in Remote Sensing | 2504.17177 | 遥感基础模型综述 |
| FlexiMo | A Flexible Remote Sensing Foundation Model | 2503.23844 | 遥感基础模型 |

---

## 八、总览快查表

> 按"与本研究重合度"排序。重合度高 = 必须在 prior work 中明确差异化。
> ✓ = 完整 verify(看过 arXiv abstract 页 + Comments 字段);⚠ = 仅 snippet。

| # | 简称 | 重合度 | 发表 | 关键差异 | 状态 |
|---|------|:------:|------|---------|:----:|
| 1 | **MOC-RVQ** | 🔥🔥🔥 | GLOBECOM 2024 | 域 + 蒸馏机制 | ✓ |
| 2 | **ReVQom** | 🔥🔥🔥 | **ICASSP 2026** | 多智能体场景 + 无蒸馏 | ✓ |
| 3 | **SemHiTok** | 🔥🔥 | **ICLR 2026** | 多模态生成,非通信 | ✓ |
| 4 | **LEO JSCC EO** | 🔥🔥 | **ICC 2026** | 连续值 JSCC + 资源分配 | ✓ |
| 5 | **ResiTok** | 🔥🔥 | arXiv 2025 | 非 RVQ + 无蒸馏 | ✓ |
| 6 | **VQ-VAE+OFDM** | 🔥🔥 | arXiv 2025 | 单层 VQ | ✓ |
| 7 | **MAGC** | 🔥🔥 | arXiv 2024 | 非离散索引 | ✓ |
| 8 | **FM-SemCom** | 🔥🔥 | arXiv 2025 | FM 用法不同 | ✓ |
| 9 | **FMSAT** | 🔥🔥 | arXiv 2024 | FM 用法不同(同 §8 团队) | ✓ |
| 10 | **Semantic-Loss (Luxembourg)** | 🔥🔥 | arXiv 2025 | 不做 codebook | ✓ |
| 11 | **SFSC** | 🔥 | arXiv 2026 | 单层 VQ + 多址 | ✓ |
| 12 | **Compressed Learning** | 🔥 | SPIE proc / 2024 | 压缩感知 | ✓ |
| 13 | **Scalable EO Transmission** | 🔥 | arXiv 2024 | 多光谱像素优先级 | ✓ |
| 14 | **Discernment** | 🟡 | arXiv 2026 | RF/音频信号,非图像 | ✓ |
| 15 | **LM-SPT** | 🟡 | arXiv 2025 | 语音域类比 | ✓ |
| 16 | Rate-Distortion EO | 🔥 | DLR 内部 | 无 codebook | ⚠ snippet |
| ... | (次要 baseline 见 §7) | — | — | — | — |

**新发现的关键事实**:
- ReVQom 已 ICASSP 2026 接收,SemHiTok 已 ICLR 2026 接收,LEO JSCC EO 已 ICC 2026 接收 —— **2026 会议接收 3 篇相关工作**,这条赛道竞争升温
- FMSAT (§4.3) 与 FM-SemCom (§2.5) 是 **同一团队**(Peiwen Jiang / Chao-Kai Wen / Shi Jin)的连续工作,2024-2025 年覆盖卫星 + 自动驾驶两个域
- Semantic-Loss (§3.2) 与 LEO JSCC EO (§4.4) 是 Luxembourg University Chatzinotas 组的 **连续工作**,共享多位作者

---

## 九、风险提醒与后续行动

### 9.1 时间线压力(更新)

```
2024-01    MOC-RVQ ↓ (GLOBECOM 2024 已接收)
2024-04    FMSAT ↓ (preprint, 提交 IEEE 待审)
2024-09    MAGC, Compressed Learning ↓
2024-12    Scalable EO Transmission ↓
2025-03    Semantic-Loss (Luxembourg), SemHiTok ↓
2025-05    ResiTok ↓
2025-06    LM-SPT ↓
2025-08    VQ-VAE+OFDM ↓
2025-09    ReVQom (ICASSP 2026 接收), FM-SemCom ↓
2026-02    Discernment ↓
2026-03    SFSC, LEO JSCC EO (ICC 2026 接收) ↓
2026-05-31 你做完 Stage 1-4 (今天)
                 ↑
            **3 篇 2026 会议接收的工作已存在: ReVQom (ICASSP) + SemHiTok (ICLR) + LEO JSCC EO (ICC)**
```

**密集出现期 + 接收升温**:2024 下半年到 2026 上半年——**3 篇 2026 会议接收的工作已经存在**。
**风险升级**:多篇 preprint 随时被接收,你的论文必须**在 2026 年下半年前投稿**,否则赛道会被这些 2026 会议工作堵住。

### 9.2 待补 verify 的论文

经过这一轮深度 verify,**只剩 1 篇** 仍然只有 snippet 信息:
- **§6.2 Rate-Distortion EO**(elib.dlr.de 内部出版库)—— 需 fetch DLR 内部页才能 verify

其余 16 篇核心论文全部完成 abstract 级 verify。

### 9.3 你方向的护城河(确认)

完整组合「**遥感 + RVQ + 离散索引 + RemoteCLIP 蒸馏 L0 + 数字信道仿真 + 任务保真评测**」**6 件事同时成立**——经过两轮检索 + 深度搜索 8 个维度,**未发现完全相同的工作**。

### 9.4 写论文 related work 章节的引用清单(更新版)

**必引(差异化论证 · 已会议接收的强对手)**:
- **MOC-RVQ**(GLOBECOM 2024)—— RVQ-SemCom 路线最近的会议工作
- **ReVQom**(ICASSP 2026)—— RVQ + 通信压缩的新强对手
- **LEO JSCC EO**(ICC 2026)—— LEO + EO + JSCC 的新强对手
- **SemHiTok**(ICLR 2026)—— "层级语义 codebook"的同期重要工作(机制类比)

**必引(差异化论证 · arXiv preprint)**:
- **MAGC**(同应用域,方法学不同)
- **Semantic-Loss (Luxembourg)**(任务保真评测范式)

**应引(完整性)**:
- ResiTok / VQ-VAE+OFDM / FM-SemCom / FMSAT(占坑级 preprint,但同期工作要点到)

**可引(灵感来源)**:
- SpeechTokenizer(招式来源)
- RemoteCLIP(论据 A 唯一外援 + 我们的蒸馏教师)
- LM-SPT(语音域同期工作,旁证不孤例)

**baseline 对比章节(Stage 5 做)**:
- JPEG2000 + LDPC(经典基线)
- DeepJSCC(连续值 JSCC 代表)
- MOC-RVQ(RVQ-SemCom 代表)
- ReVQom(可选,如果做多智能体感知扩展实验)

### 9.5 文件维护建议

- 每次写论文前 / 投稿前 / 中期答辩前,**重新 verify 这份文件中所有标 "⚠ 待 verify" 或 "snippet" 的论文**
- 新发现论文随时追加,标注 "已 verify / snippet" 状态
- 如果某篇 preprint 被接收,把 "arXiv preprint" 改成具体期刊/会议名

---

## 来源说明

- **18 篇核心论文**经 WebFetch 完整 verify(SpeechTokenizer / RemoteCLIP / MOC-RVQ / ResiTok / VQ-VAE+OFDM / MAGC / FM-SemCom / **ReVQom / Semantic-Loss / SFSC / Scalable EO / FMSAT / LEO JSCC EO / Discernment / SemHiTok / LM-SPT / Compressed Learning** + 部分次要论文)
- 仅 **1 篇** Rate-Distortion EO 还没 verify(elib.dlr.de 内部页)
- 已修正之前的两条错误归类:
  - **Semantic-Loss 之前误归 DLR,实际是 Luxembourg University Chatzinotas 组**
  - **§6 章原本叫"DLR 系统线",实际不是统一团队工作**
- **2026-05 第三轮深度检索追加 ~20 篇新论文**, 详见 §十

---

## 十、第三轮深度多 agent 检索 (2026-05) · 新发现

> 本节是第三轮检索的新增收录, 不替换前面 1-9 节的已有论文。该次检索由 4 个并发 agent 跨 6 个社区扫描, 共扫到 ~30 篇相关工作 (其中部分与前面已收录论文重叠), 本节只列**新发现的 ~20 篇**。
>
> 该轮检索的动机: motivation 场景从"商业卫星下行" 改为 "UAV / 应急 / 边缘 / SmallSat 链路余量受限场景" 后, 需要扩展相邻文献覆盖。
>
> 这一轮检索把同类工作按 6 个**轴**重新组织 (与 motivation.md §2 现状的轴划分一致):
> - 轴 A · 数字 SemCom 多层离散 token (§十.A)
> - 轴 B · 基础模型 → 离散 token 语义化 (§十.B)
> - 轴 C · 音频域同构机制 (§十.C)
> - 轴 D · 渐进式 / 可伸缩编码 (§十.D)
> - 轴 E · LEO / 遥感 EO 数字传输 (§十.E)
> - 轴 F · UAV/EO 任务感知传输 (§十.F)

### 十.A · 轴 A 新增 · 数字 SemCom 多层离散 token

#### 10.A.1 ESC-MVQ 🔴 极高威胁 (POSTECH Jeon 团队) 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | End-to-End Semantic Communication with Multi-Codebook Vector Quantization |
| 作者 | Shin / Jeon (POSTECH, 待 verify) |
| 来源 | arXiv 2504.11709 (2025-04 preprint) |
| 核心机制 | 多 codebook + 可学习 BSC + 联合调制功率优化 |
| 与本研究重合 | 多 codebook + 数字信道 + 联合优化 |
| 关键差异 | (1) 自然图非遥感; (2) **无 RemoteCLIP 蒸馏**; (3) **不分层语义 vs 细节** |
| 撞车级别 | 🔴 极高 (机制接近, 但不在遥感, 不蒸基础模型) |

#### 10.A.2 MSVQ-SC 🔴 极高威胁 (POSTECH Jeon 团队) 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | Rate-Adaptive Semantic Communication via Multi-Stage Vector Quantization |
| 作者 | Jinsung Park / Yo-Seb Jeon (POSTECH, 待 verify) |
| 来源 | arXiv 2510.02646 (2025-10 preprint) |
| 核心机制 | **多级 VQ + 动态 stage/module 激活 + 增量比特分配 + 熵编码** |
| 与本研究重合 | 多级 VQ + **k 选择思想已经像** + 速率自适应 |
| 关键差异 | (1) CIFAR-10 自然图非遥感; (2) **无基础模型蒸馏**; (3) **只解码 rate 不解 channel**; (4) 动态激活基于 rate budget 而非"信道+任务"组合 |
| 撞车级别 | 🔴 极高 (k=1..4 思想最像, 但缺蒸馏 + 缺 channel) |

#### 10.A.3 VQ-CA-DJSCC 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | VQ-Enabled Digital Semantic Communication with Channel Adaptive Image Transmission |
| 来源 | arXiv 2508.03740 (2025-08) |
| 核心机制 | Swin + 多 VQ + DJSCC 离散映射 |
| 关键差异 | 自然图; 无 RVQ 蒸馏路径 |
| 撞车级别 | 🟡 高 |

#### 10.A.4 DeepJSCC-CDSC 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | Channel-Capacity Codebook Design for Digital Task-Oriented Semantic Communication |
| 来源 | arXiv 2508.04291 (2025-08) |
| 核心机制 | Wasserstein 正则把 codebook 分布对齐 K-QAM, 任务导向分类 |
| 关键差异 | 单层 codebook; CIFAR-10 |
| 撞车级别 | 🟡 中 |

#### 10.A.5 TextTokenComm 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | Text-Guided Token Communication for Wireless Image Transmission |
| 来源 | arXiv 2507.05781 (2025-07) |
| 核心机制 | 离散 token + 5G NR polar + 文本条件生成补丢失 token |
| 关键差异 | 不分层 RVQ; 文本作为辅助 |
| 撞车级别 | 🟡 中 |

### 十.B · 轴 B 新增 · 基础模型 → 离散 token 语义化 (撞车风险最高)

#### 10.B.1 VILA-U 🔴 极高威胁 (MIT Han Lab) 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | VILA-U: A Unified Foundation Model Integrating Visual Understanding and Generation |
| 作者 | Yecheng Wu / Han Cai / Song Han (MIT Han Lab) |
| 来源 | **ICLR 2025** (已接收) · arXiv 2409.04429 |
| 核心机制 | **CLIP 文本-图像对齐损失监督 Residual Quantization 视觉塔**, 把 CLIP 语义编入离散 token |
| 与本研究重合 | **CLIP 蒸馏 + RQ 离散结构 + 视觉域** |
| 关键差异 | (1) **不做 SpeechTokenizer 式分层解耦** — 对整条 RQ 联合对齐, 不区分"第 1 层语义 / 后层细节"; (2) 多模态 LLM 用途, **无信道仿真**; (3) 自然图非遥感; (4) 评测是 LLM 任务而非分类准确率 |
| 撞车级别 | 🔴 **极高** — "CLIP 蒸馏 RVQ" 在视觉域被它占走整链版本 |
| 处理 | 必引必差异化, 强调"分层解耦 vs 整链对齐"是质变 |

#### 10.B.2 TokLIP 🔴 高威胁 (Tencent ARC) 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | TokLIP: Marry Visual Tokens to CLIP for Multimodal Comprehension and Generation |
| 来源 | arXiv 2505.05422 (Tencent ARC) |
| 核心机制 | SigLIP/CLIP 语义化 VQ token, 双路 (低层 VQ + 高层 ViT-CLIP) |
| 关键差异 | (1) 双路结构非"RVQ 第一层蒸馏"; (2) LLM 用途无信道 |
| 撞车级别 | 🔴 高 |

#### 10.B.3 BEiT v2 (VQ-KD) 🟡 高威胁 (MSRA) 【已知工作】

| 项 | 内容 |
|---|---|
| 完整标题 | Masked Image Modeling with Vector-Quantized Visual Tokenizers |
| 作者 | Zhiliang Peng / MSRA |
| 来源 | arXiv 2208.06366 (2022) |
| 核心机制 | **CLIP/DINO 蒸馏 VQ 单层 codebook** 用于 MIM 预训练 |
| 关键差异 | (1) 单层 VQ 而非 RVQ; (2) MIM 预训练不做通信 |
| 撞车级别 | 🟡 高 (是"CLIP 蒸馏 VQ"的开山祖, 必引) |

#### 10.B.4 SemCLIP 🔴 高威胁 (Imperial Gunduz + BUPT) 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | (Semantic Communication via CLIP tokens) |
| 作者 | Hu / 王峰 (BUPT) / Gunduz (Imperial College) |
| 来源 | arXiv 2502.18200 (2025-02) |
| 核心机制 | **直接传 CLIP token 做 SemCom**, 跨任务 |
| 与本研究重合 | **思路最近** — 都用 CLIP 做语义通信 |
| 关键差异 | (1) **OpenAI CLIP 非 RemoteCLIP**; (2) 自然图非遥感; (3) 不切 RVQ 层; (4) 无 AID 任务保真协议 |
| 撞车级别 | 🔴 高 (一旦换 RemoteCLIP + RS 数据集就追上) |

#### 10.B.5 Semantic Compression with MFM 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | Semantic Compression with Multimodal Foundation Models |
| 作者 | Ruiqi Shen / Deniz Gunduz (Imperial College) |
| 来源 | **IEEE MLSP 2025** (已接收) · arXiv 2509.05925 |
| 核心机制 | 直接压缩 CLIP embedding 到 2-3×10⁻³ bpp |
| 关键差异 | (1) 单层非渐进; (2) 非 RVQ; (3) 自然图 |
| 撞车级别 | 🟡 中 |

#### 10.B.6 CLIP-SemCom 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | CLIP-based Semantic Communication Performance Optimization |
| 来源 | arXiv 2507.08873 (2025-07) |
| 核心机制 | CLIP 作语义编码器 + 延迟/能耗优化 |
| 关键差异 | 不分层 RVQ; 仅作编码器不做蒸馏 |
| 撞车级别 | 🟡 中 |

#### 10.B.7 DINO-Tok / UniTok / MUSE-VL / Free Semantics for UVSC 【snippet 待 verify】

| 简称 | 来源 | 核心机制 | 撞车级别 |
|---|---|---|:-:|
| DINO-Tok | arXiv 2511.20565 | DINO 蒸 VQ tokenizer (单层) | 🟡 中 |
| UniTok | arXiv 2502.20321 (FoundationVision) | CLIP + VQ-VAE 联合训练 | 🟡 中 |
| MUSE-VL | arXiv 2411.17762 | text-aligned visual tokens (单层) | 🟡 中 |
| Free Semantics for UVSC | arXiv 2409.11718 | VFM 共享语义无监督视频压缩 (连续 latent) | 🟢 低 |

#### 10.B.8 PIC-SHC 🟡 中 (概念上最像"语义级渐进") 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | Progressive Image Compression for Semantically Hierarchical Classification |
| 来源 | arXiv (待 verify, 编号待确认) |
| 核心机制 | 单 bitstream 拆 prefix 通道 (128/224/320) 对应粗/中/细类别; **CLIP 仅做标签聚类**, ResNet50 多层 CE 监督 |
| 与本研究重合 | "语义级渐进" 概念上最像 |
| 关键差异 | (1) **连续 hyperprior 非 VQ token**; (2) **CLIP 仅用作标签聚类不蒸馏到 token**; (3) 自然图; (4) 无信道仿真 |
| 撞车级别 | 🟡 中 |

### 十.C · 轴 C 新增 · 音频域同构 (类比论据强化)

#### 10.C.1 STACodec 🔴 同构 (机制完全镜像) 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | Semantic Token Assignment for Balancing Acoustic Fidelity and Semantic Information |
| 来源 | arXiv 2602.06180 (待 verify, 编号疑 v2/索引误差) |
| 核心机制 | **机制完全同构 SpeechTokenizer** — RVQ 第一层语义 + 后层 acoustic |
| 与本研究关系 | 类比论据强化 — 证明音频域已成范式 |
| 关键差异 | 音频域非视觉/遥感 |

#### 10.C.2 LM-SPT 【已收录在 §5.2, 此处补编号】

#### 10.C.3 HAC (Factorized RVQ-GAN) 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | Factorized RVQ-GAN For Disentangled Speech Tokenization |
| 来源 | arXiv 2506.15456 (2025-06) |
| 蒸馏教师 | HuBERT + LaBSE (双教师) |
| 核心机制 | 因子化 RVQ |
| 与本研究关系 | 类比论据强化 |

#### 10.C.4 DM-Codec 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | Distilling Multimodal Representations for Speech Tokenization |
| 来源 | arXiv 2410.15017 |
| 蒸馏教师 | LM + Speech model (多模态) |
| 核心机制 | RVQ + 多模态蒸馏 |
| 与本研究关系 | 类比论据强化 |

### 十.D · 轴 D 新增 · 渐进式 / 可伸缩编码

#### 10.D.1 SIC-HM / SVC-HM (Scalable Image/Video Coding for Humans and Machines) 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 作者 | Hyomin Choi (SFU) |
| 来源 | arXiv 2107.08373 (TIP) / arXiv 2208.02512 |
| 核心机制 | **base 层服务机器视觉, enhancement 层服务人眼重建** |
| 与本研究关系 | 经典"机器/人眼分层"对照 |
| 关键差异 | 未用 VQ token; 无信道仿真 |

#### 10.D.2 StyleGAN-Scalable 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | Scalable Face Image Coding via StyleGAN Prior |
| 来源 | arXiv 2312.15622 (2023) |
| 核心机制 | StyleGAN 先验 + 三层 basic/middle/enhancement |
| 关键差异 | 限人脸; 单模态 GAN |

#### 10.D.3 Linear Progressive Coding 【已 verify · snippet】

| 项 | 内容 |
|---|---|
| 完整标题 | Linear Progressive Coding for Semantic Communication using DNNs |
| 作者 | Iowa University (待 verify 具体团队) |
| 来源 | arXiv 2309.15959 (2023) |
| 核心机制 | 线性投影构造 hierarchical 分层 measurement, coarse→fine 任务 |
| 与本研究关系 | 渐进语义早期工作 |
| 关键差异 | 非离散 token; 非基础模型蒸馏; 层数不随 SNR 动态选 |

#### 10.D.4 Resi-VidTok 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | An Efficient and Decomposed Progressive Tokenization Framework for Ultra-Low-Rate Video Transmission |
| 来源 | arXiv 2510.25002 (2025-10, ResiTok 视频版) |
| 核心机制 | key + refinement token, prefix-decodable |
| 关键差异 | 视频版, 同样无基础模型蒸馏 |

#### 10.D.5 Dynamic DJSCC 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | Dynamic Deep Joint Source-Channel Coding for Semantic Communications |
| 来源 | arXiv 2507.20467 (2025-07) |
| 核心机制 | hierarchical layer activation + sequential randomized training |
| 关键差异 | 连续 latent 非离散 RVQ; 无基础模型蒸馏 |

### 十.E · 轴 E 新增 · LEO / 遥感 EO 数字传输 (Luxembourg SnT 团队全家桶)

#### 10.E.1 CSA-LEO / OnAir-EO 🔴 高威胁 (Luxembourg SnT) 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 (1) | Cognitive Semantic Augmentation LEO Satellite Networks for Earth Observation (magazine) |
| 完整标题 (2) | On-Air Deep Learning Integrated Semantic Inference Models for EO (期刊版) |
| 作者 | Chou / Chatzinotas (Luxembourg SnT) |
| 来源 | arXiv 2410.21916 (10/2024) / arXiv 2409.15246 v3 (2024) |
| 核心机制 | DT-JSCC + 语义数据增强, 多光谱, ISL (跨星链路) |
| 关键差异 | 离散 JSCC 但**无 RVQ, 无 RemoteCLIP, 无层级 k 选择** |
| 撞车级别 | 🔴 高 (团队已占 EO+JSCC, 一旦加 RVQ 就直接撞车) |

#### 10.E.2 EO-KD (Semantic Knowledge Distillation Onboard) 🟡 中 (Luxembourg SnT) 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | Semantic Knowledge Distillation for Onboard Sat EO Image Classification |
| 来源 | arXiv 2411.00209 (2024-11, Luxembourg SnT) |
| 核心机制 | KD 把大模型蒸到 ResNet8/16 onboard |
| 关键差异 | KD 但**不蒸到 codebook**; 无信道 |

#### 10.E.3 ADJSCC-SAT (Sentinel-2) 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | Deep JSCC for Small Satellite Applications |
| 来源 | arXiv 2508.00715 (2025-08) |
| 核心机制 | DJSCC + ADJSCC, 多状态信道注意力 |
| 关键差异 | 模拟 JSCC; 无 codebook; 无 RVQ |

#### 10.E.4 SwinJSCC over LEO + DQN 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | Adaptive Rate Control for SemCom over LEO Sat-to-Ground |
| 来源 | arXiv 2605.10095 (待 verify 编号) |
| 核心机制 | RL 选 SwinJSCC channel dim, 过顶窗口 |
| 关键差异 | 连续 SwinJSCC; 自然图 Kodak24 |

#### 10.E.5 Visual Event AI-Edge LEO 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | Visual Event Detection over AI-Edge LEO Satellites with AoI Awareness |
| 来源 | arXiv 2512.19764 (待 verify 编号) |
| 核心机制 | DJSCC + AoI 阈值, 任务推理 |
| 关键差异 | 模拟 JSCC; 重在 AoI; 不做 codebook |

#### 10.E.6 FOOL 🟢 低 (TU Wien Dustdar) 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | Addressing the Downlink Bottleneck in Satellite Computing with Neural Feature Compression |
| 作者 | Furutanpey (TU Wien, Dustdar 组) |
| 来源 | **IEEE TMC 2025** (已接收) |
| 核心机制 | 任务无关特征压缩 + 切片 + tile context |
| 关键差异 | 任务 agnostic; 不针对信道; 无 RVQ |

#### 10.E.7 Compressed Learning Onboard 🟢 低 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | Compressed Learning Based Onboard Semantic Compression for RS Platforms |
| 作者 | Bhattacharjee / Jung (TU Berlin) |
| 来源 | arXiv 2409.01988 (2024-09) |
| 核心机制 | 稀疏矩阵编码 + NA-ALISTA 解 + 分类 fine-tune |
| 关键差异 | 非 codebook; 非 RVQ; 不做 SNR 扫描 |

#### 10.E.8 NEC (Neural Embedding Compression) 🟢 低 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | Neural Embedding Compression for Efficient Multi-Task EO Modelling |
| 来源 | arXiv 2403.17886 (IGARSS 系) |
| 核心机制 | 直接传压缩 embedding, 多任务 |
| 关键差异 | 无 RVQ 层级; 无信道仿真 |

### 十.F · 轴 F 新增 · UAV/EO 任务感知传输 (协议同源)

#### 10.F.1 Gao 等 UAS-TOC 🔴 协议同源 (TGRS 2022) 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | Task-Oriented Image Transmission for Scene Classification in Unmanned Aerial Systems |
| 作者 | Gao et al. |
| 来源 | **IEEE TGRS 2022** (已接收) · arXiv 2112.10948 |
| 核心机制 | UAV 端轻量策略网, DRL 选 4×4 块 → 块状压缩感知 + 8-bit 量化; 后端 ResNet34 分类 |
| 与本研究重合 | **🔴 同 AID 30 类、同分类准确率指标、同 Rayleigh+AWGN 信道协议** |
| 关键差异 | (1) 块状压缩感知**没有 codebook / RVQ**; (2) 单层量化无层级; (3) 无基础模型蒸馏 |
| 撞车级别 | 🔴 协议同源对手 — 必须在 method 章节直接对比 |
| 处理 | 必引必对比. 这是 motivation §1 痛点的"协议见证者" — 同协议下证明了"任务感知比像素感知好", 但局限于 2022 浅技术栈 |

#### 10.F.2 DSC-UAV 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | Context-Aware Information Transfer via Digital Semantic Communication in UAV-Based Networks |
| 来源 | arXiv 2601.01430 v2 (待 verify 编号) |
| 核心机制 | ViT + prompt 文本编码器 → 量化数字符号 → UAV 中继转发 + RL 轨迹优化 |
| 关键差异 | 单层学习量化非 RVQ; 无第一层蒸馏; 指标 SSI 与 AoI |
| 撞车级别 | 🟡 高 (UAV 域命中) |

#### 10.F.3 PCAS-GR (UAV downlink) 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | Predictive Channel-Aware Scheduling and Generative Reconstruction (UAV downlink) |
| 来源 | arXiv 2602.10482 (待 verify 编号) |
| 核心机制 | UAV 下行场景把语义拆为"结构 + 纹理"两路, 信道预测调度结构走稳; 生成重建纹理 |
| 关键差异 | 2 路而非 4 层 RVQ; 指标 PSNR; 无 codebook 蒸馏 |
| 撞车级别 | 🟡 中 (UAV + 双层语义) |

#### 10.F.4 TOAST 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | Task-Oriented Adaptive Semantic Transmission over Dynamic Wireless Environments |
| 来源 | arXiv 2506.21900 (2025-06) |
| 核心机制 | Swin-JSCC + DQN 自适应损失权重 + LoRA 信道适配 + Diffusion 去噪 |
| 关键差异 | 连续 latent 非 codebook; 不是 UAV/航空特化 |

#### 10.F.5 EO-JSCC Unified Semantic Loss 【snippet 待 verify】

| 项 | 内容 |
|---|---|
| 完整标题 | Toward a Unified Semantic Loss Model for Deep JSCC-based Transmission of EO Imagery |
| 来源 | arXiv 2602.00136 (待 verify 编号) |
| 核心机制 | EO 遥感图像 DeepJSCC, 统一语义损失分析 JSCC×SNR×语义质量 |
| 关键差异 | 纯 JSCC 连续传输; 无离散 codebook |

### 十.G · 第三轮检索的撞车判定与风险窗口

#### 10.G.1 撞车判定汇总

| 撞车情境 | 是否撞 | 备注 |
|---|:-:|---|
| RemoteCLIP / 遥感基础模型 → codebook | **未撞** | 用户在这一组合上仍独家 |
| 通用 CLIP → RVQ 第一层 (SpeechTokenizer 式分层蒸馏) | **半撞** | VILA-U 蒸了整链不分层; 视觉域 SpeechTokenizer 式分层蒸馏仍空白 |
| 多层 RVQ + 真信道仿真 | **半撞** | MOC-RVQ + MSVQ-SC + ESC-MVQ 占, 但都在自然图 |
| EO + JSCC 数字传输 | **半撞** | Luxembourg SnT 占 EO+JSCC, 但仍走连续 JSCC 非离散 RVQ |
| AID 30 类分类协议 | **半撞** | Gao 2022 (TGRS) 用同协议但块状压缩感知非 codebook |
| **三轴交集 (RVQ 分层蒸馏 × 视觉/遥感 × 任务保真协议)** | **未撞** | **空白, 用户独家位置** |

#### 10.G.2 潜在追赶者三梯队 (按威胁排序)

| 梯队 | 团队 | 威胁点 |
|:-:|---|---|
| 1 (最危险) | **Luxembourg SnT (Chatzinotas + Lagunas + Chou)** | 5+ 篇 EO+JSCC 含 ICC 2026 接收, 加 RVQ + 基础模型蒸馏是自然延伸 |
| 2 | **POSTECH Jeon 团队 (ESC-MVQ + MSVQ-SC)** | 多级 VQ + 信道联合调制功底极强, 还没做遥感是用户的护城河 |
| 3 | **Imperial Gunduz + BUPT (SemCLIP 作者)** | 直接传 CLIP token 思路在那条路上, 一旦换 RemoteCLIP + RS 就追上 |
| (同梯队) | **MIT Han Lab (VILA-U)** | LLM 视角扩展到压缩/通信不远 |
| (同梯队) | **Tencent ARC (TokLIP)** / **FoundationVision (UniTok)** | 持续做 visual tokenizer |

#### 10.G.3 抢占建议

> **来自 Agent 1 的判断** (UAV / 边缘 AI 检索 agent):
> *"建议尽快把'第一层 codebook 蒸馏'这一步用 RemoteCLIP 跑通并挂上 arXiv 占位"*
>
> **来自 Agent 4 的判断** (LEO / 遥感 EO 检索 agent):
> *"Luxembourg SnT 团队连续产出, 接下来很容易加 RVQ"*
> *"2026 年很可能有 1-2 篇 ICC/GLOBECOM/IoT-J 把它拼起来"*

→ Stage 3 已实测验证 L0 linear probe +38.6pp, 应整理 short paper 草稿挂 arXiv 占位首发地位。

#### 10.G.4 写论文 related work 章节的更新引用清单

按"必引 / 强引 / 选引 / 选不引" 分类:

**必引 (related work 必须出现)**:
- SpeechTokenizer (类比论据源头)
- RemoteCLIP (蒸馏教师)
- MOC-RVQ (轴 A 头号对手)
- ReVQom (轴 A 新强敌)
- VILA-U (轴 B 撞车风险最高)
- BEiT v2 (轴 B 开山祖)
- Gao 2022 TGRS (轴 F 协议同源)
- LEO JSCC EO (轴 E Luxembourg 接收作)
- ESC-MVQ / MSVQ-SC (轴 A POSTECH 强对手)

**强引 (related work 主线段落要提)**:
- TokLIP, SemCLIP (轴 B 高威胁)
- STACodec (轴 C 同构强化)
- CSA-LEO / OnAir-EO (轴 E Luxembourg 系统)
- Semantic-Loss (轴 E EO 任务损失)
- ResiTok / Resi-VidTok (轴 D 分层 token)
- PIC-SHC (轴 D "语义级渐进"概念对照)
- SIC-HM / SVC-HM (轴 D 经典对照)
- DSC-UAV (轴 F UAV 域)

**选引 (related work 末尾或脚注)**:
- DINO-Tok / UniTok / MUSE-VL (轴 B 中相似)
- DM-Codec / HAC / LM-SPT (轴 C 类比扩展)
- StyleGAN-Scalable / LPC (轴 D 渐进编码)
- TOAST / FM-SemCom (轴 F 任务导向)
- DeepJSCC-CDSC / VQ-CA-DJSCC / TextTokenComm (轴 A 中相似)
- ADJSCC-SAT / FOOL / NEC / EO-KD / Compressed Learning Onboard (轴 E 弱相关)
- Free Semantics for UVSC (轴 B 弱相关)
- CLIP-SemCom / Semantic Compression w/ MFM (轴 B 弱相关)
- Visual Event AI-Edge LEO (轴 E 弱相关)
- Dynamic DJSCC (轴 D 弱相关)

→ **共 ~40 篇引用清单**, related work 章节按 6 轴组织展开。

- 部分论文的作者单位仍未抓全(arXiv abstract 页有些不直接列单位)—— 这部分如需写论文需 fetch 全文 PDF
