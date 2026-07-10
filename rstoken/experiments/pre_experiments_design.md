# 前期实验设计 · 完整版

> 本文件定位: **完整的前期实验设计文档** —— 涵盖 (1) Stage 1-4 已完成的奠基性实验, (2) 投稿前必须补齐的预实验, 一并编号管理。
>
> 配套执行记录在 [pre_experiments_log.md](pre_experiments_log.md), 每个实验跑完往那里填实测结果。
>
> 上下文: motivation.md 已给出 4 条可证伪承诺, 已完成的 Stage 1-4 (E1-E6) 验证了主线 4 条, E7-E15 补齐了层级分工、教师消融、Rayleigh 与论文 figure 证据。v0.3 自审后新增 Stage 5 / 阶段 F (E16-E22), 专门补投稿前硬伤: 外部 baseline、模型种子统计、task/reconstruction 评估口径拆分。
>
> 关联: [results.md](results.md) (Stage 1-4 数字汇总) / [../motivation.md](../motivation.md) (论证主文) / [pre_experiments_log.md](pre_experiments_log.md) (执行记录)
>
> 创建日期: 2026-05-31

---

## 0. 实验体系总览

```
─── 阶段 A · 基础设施 (已完成) ──────────────────────────────
  E1  数据集与 splits 准备           AID → train/val/test
─── 阶段 B · 主线论证训练 (已完成) ───────────────────────────
  E2  单层 VQ baseline               vqvae_baseline (Stage 1)
  E3  RVQ x4 无蒸馏 baseline         rvq_baseline   (Stage 2)
  E4  RVQ x4 + RemoteCLIP 蒸馏       rvq_distill    (Stage 3, 主推)
─── 阶段 C · 主线评测 (已完成) ──────────────────────────────
  E5  L0 linear probe (4 端点特征)   分类提升 +24.7~38.6pp
  E6  Stage 4 AWGN 信道仿真          k=1 vs k=4 SNR 扫描
─── 阶段 D · 论据补齐 (已完成 E7-E12) ──────────────────────
  E7  分层 linear probe (中间档)     P0  验证层级分工干净
  E8  教师消融 (OpenAI CLIP)         P0  验证 RemoteCLIP 不可替代
  E9  +10dB 多种子复测               P0  §6.1-3 阈值稳定性
  E10 Rayleigh 衰落信道              P1  Gao 2022 协议同源
  E11 蒸馏权重 trade-off (w=0.1/1.0) P1  方法章节必备
  E12 AID 类数 29 vs 30 核对         P1  数据一致性
─── 阶段 E · 论文 figure (部分待跑) ─────────────────────────
  E13 per-class confusion matrix     P2  与 Gao 2022 比对
  E14 L0 codebook t-SNE 可视化       P2  几何证据
  E15 bit-budget 含 metadata 核算    P2  motivation §3.2 承诺
─── 阶段 F · v0.4 投稿前硬伤补齐 ───────────────────────────
  E16 clean AID classifier            P0  重建图 / 外部 baseline 统一任务评估器
  E17 task vs reconstruction split     P0  拆清 h0 任务路径与 k=1..4 重建路径
  E18 model-seed suite                 P0  rvq_baseline / rvq_distill 3 模型种子
  E19 classic compression baseline     P0  WebP/JPEG2000/BPG 同 bit budget 外部基线
  E20 Rayleigh 0dB stress slice        P1  breakdown point 外部语境
  E21 Gao/DeepJSCC feasibility         P1  GRSL/TGRS 扩展基线预研
  E22 v0.4 table aggregation           P0  mean±std 主表与 claim audit
```

**优先级标签**: P0 = 必做 (挂 arXiv 前) · P1 = 应做 (投稿前) · P2 = 加分 (figure 章节)

---

## 1. motivation 论断 → 实验对照矩阵

| motivation 论断 | 对应实验 | 状态 |
|---|---|:-:|
| §6.1-1 L0 蒸馏后分类 +5pp | E5 (Stage 3 L0_bow +24.7pp) | ✅ |
| §6.1-2 SNR=-5dB 下 k=1 ≥ k=4 | E6 (8.2% > 5.7%) | ✅ |
| §6.1-3 SNR=+10dB 下 k=4-k=1 PSNR ≥ 3dB | E6 (实测 2.98 dB) → **E9 复测** | ⚠️ 踩线 |
| §6.1-4 蒸馏代价 ≤ 0.5 dB | E4 vs E3 (-0.22 dB) | ✅ |
| §3.2 层级分工干净 | **E7** | ❌ 待做 |
| §5.1 RemoteCLIP 不可替代 | **E8** | ❌ 待做 |
| §6 (c) Rayleigh 协议同源 (Gao 2022) | **E10** | ❌ 待做 |
| 蒸馏权重 trade-off | **E11** | ❌ 待做 (w=0.5 单点) |
| 数据一致性 (29 vs 30 类) | **E12** | ❌ 待做 |

---

## 阶段 A · 基础设施 (已完成)

### E1 · 数据集与 splits 准备

**目的**: 确立 AID 数据集的 train/val/test 划分, 锚定 motivation §6 (c) 的 "AID 30 类 + Gao 2022 协议同源" 评测协议。

**做法**:
- 数据集: AID (Aerial Image Dataset, 30 类场景分类)
- 脚本: [scripts/01_prepare_aid.py](../scripts/01_prepare_aid.py)
- 输出: `data/splits/train.csv`, `val.csv`, `test.csv`
- transform: 训练用 `RandomResizedCrop(256)`, 评测用 `Resize(256) + CenterCrop(256)`, 归一化到 `[-1, 1]`

**判据**: 脚本能跑通, train/val/test csv 存在, 类标签覆盖完整。

**注**: 实际类数以 csv 里 `label.nunique()` 为准, 见 E12 的核对。

---

## 阶段 B · 主线论证训练 (已完成)

### E2 · Stage 1 · 单层 VQ baseline (`vqvae_baseline`)

**目的**: 最朴素 VQ-VAE, 验证基础设施跑得通; 给 RVQ 提供"单层"对照点。

**配置**: [configs/vqvae_baseline.yaml](../configs/vqvae_baseline.yaml)
- 量化器: VQ x1, codebook 1024 个 entries, dim=256
- 损失: `L_recon + L_VGG`
- 50 epoch, batch=32, lr=2e-4

**命令**:
```powershell
& $py -X utf8 scripts/03_train_vqvae.py --config configs/vqvae_baseline.yaml
```

**判据**: 训练能收敛, PSNR > 22 dB。

---

### E3 · Stage 2 · RVQ x4 无蒸馏 baseline (`rvq_baseline`)

**目的**: 残差量化(4 层) 把 PSNR 推到上限, 给 Stage 3 提供"加蒸馏前后"的纯净对照 —— **E3 vs E4 唯一差异是有无蒸馏 loss, 其它超参 100% 一致**。

**配置**: [configs/rvq_baseline.yaml](../configs/rvq_baseline.yaml)
- 量化器: RVQ x4, 每层 codebook 1024 个 entries, dim=256
- `quantize_dropout=True` (训练时随机用前 k 层重建, 让降级路径自然 fit)
- 其它同 E2

**命令**:
```powershell
& $py -X utf8 scripts/03_train_vqvae.py --config configs/rvq_baseline.yaml
```

**判据**: PSNR 比 E2 提升 ≥ 1.5 dB (RVQ 对单层 VQ 的常规增益)。

---

### E4 · Stage 3 · RVQ x4 + RemoteCLIP 蒸馏 (`rvq_distill`, **论文主推**)

**目的**: 这是论文核心方法 —— 用 RemoteCLIP 蒸馏 L0 让其承载地物身份。motivation §6 (a) 直接对应这个实验。

**配置**: [configs/rvq_distill.yaml](../configs/rvq_distill.yaml)
- 模型主体与 E3 完全一致
- **额外加蒸馏路径** ([models/distillation.py](../models/distillation.py)):
  - 教师: RemoteCLIP-ViT-B/32 (frozen, eval), embed_dim=512
  - student head: L0 量化后特征 mean pool → LayerNorm → Linear(256→512) → SiLU → Linear(512→512)
  - loss: `1 - cosine(student, teacher)`
- 总损失: `L = L_recon + L_VGG + 0.5 × L_distill`

**命令**:
```powershell
& $py -X utf8 scripts/03_train_vqvae.py --config configs/rvq_distill.yaml
```

**判据**: 与 E3 相比, PSNR 下降 ≤ 0.5 dB (motivation §6.1-4)。

---

## 阶段 C · 主线评测 (已完成)

### E5 · L0 linear probe (4 端点特征)

**目的**: 验证 motivation §6.1-1 "蒸馏后 L0 分类提升 ≥ 5pp"。

**做法**: 在 E3 (无蒸馏) 与 E4 (蒸馏) 两个 ckpt 上各提取 4 种特征, 跑 sklearn LogisticRegression。

| 特征 | 含义 |
|---|---|
| L0_bow | L0 索引在 1024 词表上的词频直方图 (1024 维), 代表"只传 L0 索引的下游能力" |
| L0_emb | L0 索引查表回 codebook embedding 后空间均池 (256 维) |
| zq_pool | 全 4 层量化输出 zq 空间均池 (256 维), 代表"传完整 RVQ"上限 |
| zpre_pool | encoder 输出无量化的空间均池 (256 维), 代表无量化天花板 |

#### 关于 L0_bow 的直观说明

L0_bow 是论文最关键的特征, 单独展开:

**字面拆解**: L0 = RVQ 第 0 层 (第一层); bow = Bag of Words (词袋)。合起来: **把 L0 codebook 的 1024 个 codeword 当成"词典", 把每张图的 L0 索引序列当成"一段文章", 数每个词出现的频率, 得到一个 1024 维向量。**

**Step 1 · 一张图过 encoder + L0 量化**:

```
输入图像: 256×256 RGB
   ↓ encoder (16 倍下采样, 通道 3 → 256)
特征图: [256 通道, 16, 16] = 256 个 patch, 每个 patch 256 维
   ↓ flatten 空间维
patch 序列: T = 256 个 token, 每个 256 维
   ↓ L0 量化 (在 1024 个 256 维 codeword 里找最近的)
L0 索引: [i_1, i_2, ..., i_256], 每个 i ∈ {0..1023}
```

例子 (假设一张机场图):

```
L0 索引 = [537, 12, 537, 805, 537, 12, 200, ..., 537]
                       ↑
            "537 号 codeword" 被选了好多次
```

直觉上, 机场图里大量是"跑道"patch → 都被量化到同一个 codeword (假设 537); "飞机"patch → 805; "草地"patch → 12。

**Step 2 · 数每个 codeword 出现了几次, 归一化**:

把长度 256 的索引序列**塌缩**成 1024 维向量:

```
L0_bow[i] = (codeword i 在这张图里出现的次数) / 256
```

| codeword id | 出现次数 | 频率 |
|:-:|:-:|:-:|
| 0 | 0 | 0.000 |
| 1 | 0 | 0.000 |
| 12 | 38 | 0.148 |
| ... | ... | ... |
| 537 | 102 | 0.398 |
| ... | ... | ... |
| 805 | 21 | 0.082 |
| 1023 | 0 | 0.000 |

→ 整张图被压成 **1024 维稀疏向量** (大部分位置是 0, 只有实际用到的 codeword 位置非零)。

**Step 3 · 这个向量是图的"语义指纹"**:

不同类别的图会有不同的 codeword 频率分布:

| 类别 | 高频 codeword | 解读 |
|---|---|---|
| Airport | 537 (跑道) / 805 (飞机) | 跑道占主体 |
| Forest | 12 (草地) / 89 (深绿) | 全是植被 |
| BaseballField | 233 (棒球场红土) / 12 (草地) | 红土+绿地 |
| Beach | 401 (沙) / 555 (海) | 沙+水 |

→ 即使**完全不看 codeword 的几何位置和具体值**, 只看"哪些 codeword 被用了多少次", 就能看出图的内容差异。

**为什么 L0_bow 是论证上最硬的特征**: 信道里实际传的就是索引本身 (10 bit 一个), 接收方拿到索引后**无需查表回 codebook embedding** 就能算 L0_bow。如果连这种最朴素的形式都能让单层 Linear 分类器拿到高准确率, 就证明 "L0 索引在离散组合模式里已经编码了地物身份"。这比"查表回 embedding 再分类" (= L0_emb) 是更弱的条件, 所以是更硬的证据 —— motivation §6.1-1 的 +5pp 判据正是用 L0_bow。

**脚本**: [scripts/04_eval_l0_linear.py](../scripts/04_eval_l0_linear.py)

**命令**:
```powershell
& $py -X utf8 scripts/04_eval_l0_linear.py --ckpt checkpoints/rvq_baseline/best.pt
& $py -X utf8 scripts/04_eval_l0_linear.py --ckpt checkpoints/rvq_distill/best.pt
```

**判据**: L0_bow Δ ≥ +5pp (motivation §6.1-1)。

---

### E6 · Stage 4 AWGN 信道仿真

**目的**: 同时验证 motivation §6.1-2 (k=1 ≥ k=4 @ -5dB) 和 §6.1-3 (k=4 - k=1 PSNR ≥ 3dB @ +10dB)。

**做法**: 拿 E3 和 E4 的 ckpt 跑 SNR × k 矩阵评测:
- SNR ∈ {-10, -5, 0, +5, +10, ∞} dB (∞ 即无信道)
- k ∈ {1, 2, 3, 4} (传前 k 层 RVQ 索引)
- 调制: BPSK
- 信道: AWGN
- 对每个组合算 (PSNR, LPIPS, L0_bow 分类准确率)

**脚本**: [scripts/05_eval_channel.py](../scripts/05_eval_channel.py)

**已知坑**: **不要在同一张 5070 Ti 并发跑两个评测**, 详见 [project_rstoken_channel_eval](../../../../.claude/projects/h--H-CODE------/memory/project_rstoken_channel_eval.md)。

**命令**:
```powershell
& $py -X utf8 scripts/05_eval_channel.py --ckpt checkpoints/rvq_distill/best.pt --out_csv logs/stage4_distill.csv
& $py -X utf8 scripts/05_eval_channel.py --ckpt checkpoints/rvq_baseline/best.pt --out_csv logs/stage4_rvq_baseline.csv
```

**判据**:
- §6.1-2: SNR=-5dB 下 蒸馏版 k=1 acc ≥ k=4 acc
- §6.1-3: SNR=+10dB 下 蒸馏版 k=4 PSNR - k=1 PSNR ≥ 3 dB

---

## 阶段 D · 论据补齐 (已完成, 历史设计)

### E7 · 分层 linear probe (P0)

**回答的问题**: L0 是不是"语义层"、L1-L3 是不是"细节层"? motivation §3.2 第 146 行明确承诺要做。

**实验设计**: 在 E3 与 E4 ckpt 上各提取 5 组分层累加的 embedding 特征, 跑 LogReg。

| 特征 | 含义 |
|---|---|
| L0_emb | 仅 L0 codebook embedding pool (E5 已有) |
| L0+L1_emb | 前 2 层 codebook embedding 求和后 pool **(新)** |
| L0+L1+L2_emb | 前 3 层 **(新)** |
| L0+L1+L2+L3_emb | 全 4 层 = 等价 zq_pool (E5 已有) |

**预期模式**:

| 配置 | L0 | L0+L1 | L0+L1+L2 | L0+L1+L2+L3 |
|---|:-:|:-:|:-:|:-:|
| E4 蒸馏 | 86% | ~86% | ~87% | 88% |
| E3 无蒸馏 | 47% | 60+% | 70+% | 88% |

**判据**:
- 蒸馏版 L0 → L0+L1 准确率提升 < 2pp → **分层解耦干净**, motivation §3.2 站住
- 提升 ≥ 5pp → L1 也吃了语义, motivation 改 "前 1-2 层语义层", k=1 承诺改 k ∈ {1, 2}

**实施**:
- 新建 [scripts/04b_eval_layered_probe.py](../scripts/04b_eval_layered_probe.py) (基于 E5 脚本扩展)
- 关键改动: `extract_features` 里增加按 `indices[..., :k]` 累加 codebook embedding 的循环
- 输出 `logs/layered_probe_distill.csv` 和 `logs/layered_probe_baseline.csv`

**耗时**: 代码 ~30 行, 跑 5 min × 2 ckpt = 10 min。

---

### E8 · 教师消融 OpenAI CLIP vs RemoteCLIP (P0)

**回答的问题**: 护城河是不是"遥感专用基础模型"? motivation §5.1 给出 RemoteCLIP-ViT-B (95.95%) vs OpenAI CLIP-ViT-B (94.95%), 但 linear probe 差距仅 1pp; 蒸馏后差距会扩大还是消失?

**实验设计**:

| 教师 | distill_weight | 来源 |
|---|:-:|---|
| 无 | — | E3 (rvq_baseline) |
| RemoteCLIP-ViT-B/32 | 0.5 | E4 (rvq_distill, 已有) |
| **OpenAI CLIP-ViT-B/32** | 0.5 | **E8 (本预实验)** |

唯一差异: 教师权重源。CLIP_MEAN / CLIP_STD 在两者间一致, 不需改预处理。

**判据**:
- Δ ≥ 7pp → 护城河成立, motivation §5 不动
- Δ < 3pp → 护城河窄, 卖点改成"分层蒸馏机制本身"
- 中间 → 诚实给出"边际收益 X pp"

**实施**:
- 改 [models/distillation.py](../models/distillation.py) `RemoteCLIPTeacher.__init__`: `if ckpt_path == "openai"` 时调 `open_clip.create_model_and_transforms(model_name, pretrained='openai')`
- 新建 [configs/rvq_distill_openai.yaml](../configs/rvq_distill_openai.yaml): 复制 `rvq_distill.yaml`, `teacher_ckpt: openai`, `run_name: rvq_distill_openai`
- 三连: 训练 → L0 probe → 信道评测

**耗时**: 代码 ~30 行, 训 70 min + L0 probe 2 min + 信道 5 min ≈ 80 min。

---

### E9 · +10dB 多种子复测 (P0)

**回答的问题**: E6 实测 k=4 - k=1 PSNR = 25.92 - 22.94 = **2.98 dB**, 距 motivation §6.1-3 红线 3 dB 仅 0.02 dB。这是真踩线还是单次信道随机噪声?

**实验设计**: 同一 E4 ckpt, SNR=+10dB, k ∈ {1,4}, 跑 3 个信道随机种子 (seed=0/1/2), 看均值与方差。

**判据**:
- mean ≥ 3.0 dB AND std ≤ 0.5 dB → 通过
- mean < 3.0 dB AND std 小 → 真踩线, 阈值放宽到 2.5 dB 或扩到 100 epoch 重训
- std > 0.5 dB → 后续所有信道实验改成 3 种子均值

**实施**:
- 扩 [scripts/05_eval_channel.py](../scripts/05_eval_channel.py) 加 `--channel_seed`, 信道前 `torch.manual_seed(seed)` + `np.random.seed(seed)`
- 跑 `--snr_list 10 --k_list 1 4 --channel_seed {0,1,2}` 三次

**耗时**: 代码 ~5 行, 运行 ~3 min × 3 = 10 min。

---

### E10 · Rayleigh 衰落信道复测 (P1)

**回答的问题**: motivation §A.6 声明 "Gao TGRS 2022 协议同源 (AID + Rayleigh+AWGN)", 但 E6 只跑了 AWGN, 不补 Rayleigh 等于声明同源却只跑了一半。

**实验设计**: 在 E3 与 E4 上各跑一遍完整 SNR×k 矩阵, 信道改成 Rayleigh+AWGN (每符号乘 |h|, h ~ CN(0, 1), 再加 AWGN, 接收端假设理想信道估计除以 |h|)。

**判据**: SNR ≤ 0 dB 区间, 蒸馏版 k=1 ≥ k=4 (与 AWGN 同向)。若不成立, 论文承认"降级机制在 AWGN 下成立, Rayleigh 需追加索引保护, 留作 future work"。

**实施**:
- 扩 [scripts/05_eval_channel.py](../scripts/05_eval_channel.py) 加 `--channel_type {awgn, rayleigh}`
- 输出 `logs/stage4_distill_rayleigh.csv`, `logs/stage4_baseline_rayleigh.csv`

**耗时**: 代码 ~20 行, 运行 ~5 min × 2 = 10 min。

---

### E11 · 蒸馏权重 trade-off 扫 (P1)

**回答的问题**: 论文方法章节"为什么 w=0.5"必备 trade-off 曲线 (PSNR vs L0_bow 准确率)。

**实验设计**: 3 个权重 w ∈ {0.1, 0.5, 1.0}, 其它超参与 E4 一致。w=0.5 = E4 已有结果。

yaml 已就绪: [configs/rvq_distill_w01.yaml](../configs/rvq_distill_w01.yaml), [configs/rvq_distill_w10.yaml](../configs/rvq_distill_w10.yaml)。

**判据**: 主要是描述性 —— 看曲线是否单调、w=0.5 是否在 frontier 上。

**实施** (顺序跑, 不要并发):
```powershell
& $py -X utf8 scripts/03_train_vqvae.py --config configs/rvq_distill_w01.yaml
& $py -X utf8 scripts/04_eval_l0_linear.py --ckpt checkpoints/rvq_distill_w01/best.pt
& $py -X utf8 scripts/05_eval_channel.py --ckpt checkpoints/rvq_distill_w01/best.pt --out_csv logs/stage4_distill_w01.csv

& $py -X utf8 scripts/03_train_vqvae.py --config configs/rvq_distill_w10.yaml
& $py -X utf8 scripts/04_eval_l0_linear.py --ckpt checkpoints/rvq_distill_w10/best.pt
& $py -X utf8 scripts/05_eval_channel.py --ckpt checkpoints/rvq_distill_w10/best.pt --out_csv logs/stage4_distill_w10.csv
```

**耗时**: 每个 weight 75 min × 2 = 150 min。

---

### E12 · AID 类数 29 vs 30 一致性核对 (P1)

**回答的问题**: [results.md](results.md) 第 30 行写 "AID 29 类", motivation 全文写 30 类 (Gao 2022 协议)。**这是一行字就能被审稿人抓到的不一致**。

**实施**:
- 读 `data/splits/{train,val,test}.csv`, `pd.read_csv(...).label.nunique()`
- 若实际 30: 改 results.md
- 若实际 29: 论文 §3 数据集小节加一句 "we exclude class XXX because YYY"

**耗时**: 5 min。

---

## 阶段 E · 论文 figure 章节 (P2 加分项)

不需重新训练, 在 E4 ckpt 上算几分钟出图, 投稿前最后一步。

### E13 · per-class confusion matrix
- SNR=-5dB 与 +5dB 下的混淆矩阵, 与 Gao 2022 per-class 图比对
- 工具: sklearn `confusion_matrix` + matplotlib

### E14 · L0 codebook t-SNE 可视化
- 把 L0 的 1024 个 codeword 投到 RemoteCLIP 特征空间, 跑 t-SNE
- 用 AID 30 类中心做颜色区分, 验证 codebook 是否聚成 30 个簇
- 数字证据 → 几何证据的转换

### E15 · bit-budget 含 metadata 协商开销显式核算
- motivation §3.2 第 144 行已承诺
- k 信令: ⌈log₂(4)⌉ = 2 bit / frame
- 把 2560 bits/img (k=1) 与 10240 bits/img (k=4) 加上 2 bit metadata, 重新核算
- 给出 trade-off 不变的诚实陈述

---

## 阶段 F · v0.4 投稿前硬伤补齐 (P0/P1)

> 来源: `paper_draft/peer_review_v0.3.md` 与 `paper_draft/review_hard_issues.md`。  
> 目标: 把 v0.3 仍被判 Major Revision 的 P0 blocker 变成可写进主文的表和图。  
> 原则: 先建立统一评估器和清晰口径, 再跑昂贵训练; 所有表默认保留 `single-seed` 字样直到 E18 聚合完成。

### E16 · clean AID classifier for reconstructed images (P0)

**回答的问题**: 外部压缩 baseline 与 RS-Token reconstruction path 都需要一个统一、冻结、与 tokenizer 无关的图像分类器。否则只能报告 `h_0` 分类, 无法比较“重建图像是否还能完成 AID 任务”。

**实验设计**:

- 数据: `data/AID_splits/{train,val,test}.csv`, 30 类, 与 tokenizer 完全同 split。
- 模型: ResNet34-AID evaluator。
  - 首选: `torchvision.models.resnet34` + ImageNet pretrained weights (若本机已有缓存)。
  - fallback: 从头训练 ResNet34, 但 log 中必须标明 `pretrained=false`。
  - 不建议用 RemoteCLIP 作为唯一 evaluator, 因为主方法本身蒸馏 RemoteCLIP, 容易被审稿人认为 evaluator 偏置。
- 输入: 256×256 RGB; train 用 random resized crop / flip, val/test 用 resize + center crop。
- 输出: clean-image top-1, per-class accuracy, confusion matrix。

**新增脚本**:

- `scripts/08_train_aid_classifier.py`
- `configs/aid_classifier_resnet34.yaml`

**建议命令**:

```powershell
$py = "C:\Users\Windows11\.conda\envs\rstoken\python.exe"
& $py -X utf8 scripts/08_train_aid_classifier.py --config configs/aid_classifier_resnet34.yaml
```

**产物**:

- `checkpoints/aid_classifier_resnet34/best.pt`
- `logs/aid_classifier_resnet34/train.log`
- `logs/aid_classifier_resnet34/metrics.csv`
- `figs/aid_classifier_clean_confusion.png`

**判据**:

- clean test top-1 ≥ 90%: 可作为主 evaluator。
- 85%-90%: 可用, 但论文需写 evaluator ceiling。
- <85%: 不建议用于主表; 改用更强 backbone 或 frozen RemoteCLIP evaluator 作为临时诊断。

**审稿对应**: 支撑 H1 外部 baseline、H3 reconstruction path、H4 Rayleigh 0 dB 语境。

---

### E17 · task path / reconstruction path 评估拆分 (P0)

**回答的问题**: 解决审稿硬伤 H3。`h_0` 只依赖 L0 indices, 所以主任务路径应固定为 `k=1`; `k=1..4` 的作用应主要体现在 reconstruction path 的 PSNR / LPIPS / reconstructed-image classifier accuracy 上。

**实验设计**:

在已有 `rvq_baseline` 与 `rvq_distill` seed=42 ckpt 上先跑一版口径拆分, 等 E18 多模型种子完成后再批量复跑。

**Task path 表**:

- 模型: no-distill / RemoteCLIP-distill。
- 特征: `h_0 = L0_bow`。
- 传输: `k=1` only。
- 信道: no-channel, AWGN `0/+5/+10 dB`, Rayleigh `0/+5/+10 dB`。
- 指标: AID top-1。

**Reconstruction path 表**:

- 模型: no-distill / RemoteCLIP-distill。
- 传输: `k=1,2,3,4`。
- 信道: no-channel, AWGN `0/+5/+10 dB`, Rayleigh `0/+5/+10 dB`。
- 指标: PSNR, LPIPS, reconstructed-image classifier accuracy (E16 evaluator), optional decode examples。

**新增脚本**:

- `scripts/09_eval_rvqs_recon_task_split.py`
  - 可从 `scripts/05_eval_channel.py` 复制主体。
  - 保留 `--channel_type`, `--channel_seed`, `--snrs`, `--ks`。
  - 新增 `--classifier_ckpt`。
  - 输出两张 csv: `task_path.csv` 与 `recon_path.csv`。

**建议命令**:

```powershell
$py = "C:\Users\Windows11\.conda\envs\rstoken\python.exe"
& $py -X utf8 scripts/09_eval_rvqs_recon_task_split.py `
  --ckpt checkpoints/rvq_distill/best.pt `
  --classifier_ckpt checkpoints/aid_classifier_resnet34/best.pt `
  --channel_type awgn `
  --out_prefix logs/e17_distill_awgn

& $py -X utf8 scripts/09_eval_rvqs_recon_task_split.py `
  --ckpt checkpoints/rvq_distill/best.pt `
  --classifier_ckpt checkpoints/aid_classifier_resnet34/best.pt `
  --channel_type rayleigh `
  --out_prefix logs/e17_distill_rayleigh
```

**判据**:

- 主文不再用 `SNR × k` 的 `h_0` 表暗示 L1-L3 提升任务准确率。
- 能形成两句清楚结论:
  1. `k=1` 的 L0 task path 是任务兜底。
  2. `k=1..4` 的 reconstruction path 随链路改善提升像素质量和重建图分类。

**失败预案**:

- 若 reconstructed-image classifier accuracy 在 `k=4` 下也很低, 论文应弱化“重建图可直接任务使用”, 改成“像素质量恢复”。
- 若 `k=2/3/4` 在低 SNR 下 PSNR/LPIPS 明显劣于 `k=1`, 继续保留“低 SNR 只传 L0”的策略, 不强推多层传输。

---

### E18 · 3 model seeds for RVQ baseline and RS-Token (P0)

**回答的问题**: 解决 H2。当前所有训练都是 seed=42, 审稿人会质疑 `+24.7 pp` 与 `25-32 pp` 是否偶然。

**实验设计**:

| 组别 | model seeds | 训练 | 评测 |
|---|---|---|---|
| `rvq_baseline` | 41 / 42 / 43 | 50 epoch | E5 + E17 |
| `rvq_distill` | 41 / 42 / 43 | 50 epoch | E5 + E17 |

seed=42 可复用现有 ckpt, 但聚合表中必须标明它来自既有 run; seed=41/43 需要新训。若为了路径整洁, 也可以把 seed=42 复制登记到 `checkpoints/seed_sweep/rvq_*_s42/`, 不改变权重。

**配置生成**:

- `configs/seed_sweep/rvq_baseline_s41.yaml`
- `configs/seed_sweep/rvq_baseline_s42.yaml`
- `configs/seed_sweep/rvq_baseline_s43.yaml`
- `configs/seed_sweep/rvq_distill_s41.yaml`
- `configs/seed_sweep/rvq_distill_s42.yaml`
- `configs/seed_sweep/rvq_distill_s43.yaml`

每个 yaml 只改:

- `run_name`
- `seed`
- `logging.ckpt_dir`
- `logging.log_dir`

**建议命令**:

```powershell
$py = "C:\Users\Windows11\.conda\envs\rstoken\python.exe"
foreach ($cfg in @(
  "configs/seed_sweep/rvq_baseline_s41.yaml",
  "configs/seed_sweep/rvq_baseline_s43.yaml",
  "configs/seed_sweep/rvq_distill_s41.yaml",
  "configs/seed_sweep/rvq_distill_s43.yaml"
)) {
  & $py -X utf8 scripts/03_train_vqvae.py --config $cfg
}
```

**评测矩阵**:

- no-channel: `scripts/04_eval_l0_linear.py`
- task/reconstruction split: `scripts/09_eval_rvqs_recon_task_split.py`
- channel seeds: 对 `SNR <= +5 dB` 至少跑 `channel_seed={0,1,2}`; `+10 dB` 可只跑 seed=42 或沿用 3 seeds 但预期方差接近 0。

**统计口径**:

1. 同一 model seed 内, 先对 channel seeds 求均值。
2. 再跨 model seeds 报告 `mean ± std`。
3. 不把 `3 model seeds × 3 channel seeds` 当作 9 个独立模型样本。

**主表最低产物**:

- Table A: train/val reconstruction metrics, no-channel `h_0` acc, `mean ± std`。
- Table B: task path, AWGN/Rayleigh selected SNR, `mean ± std`。
- Table C: reconstruction path, `k=1/4` selected SNR, `mean ± std`。

**判据**:

- RemoteCLIP-distill `h_0` no-channel 仍比 no-distill 高 ≥ 15 pp。
- `SNR >= +5 dB` 下 distill `k=1` task acc 仍高于 no-distill `k=4` 或 no-distill task baseline。
- PSNR 代价均值 ≤ 0.5 dB; 若某 seed 超过 0.5 dB, 需报告并解释。

---

### E19 · classic compression external baseline at matched bit budgets (P0)

**回答的问题**: 解决 H1。内部消融不能证明 RS-Token 相对传统 compress-then-channel-code / unprotected bitstream 更有竞争力。先做最低成本、可复现的经典压缩 baseline。

**baseline 选择**:

| baseline | 优先级 | 依赖 | 备注 |
|---|:-:|---|---|
| WebP | 必做 | Pillow 通常内置 | 最稳, 先保证外部 baseline 落地 |
| JPEG2000 | 应做 | Pillow + OpenJPEG | 若本机支持 `.jp2`, 加入主表 |
| BPG | 可选 | `bpgenc/bpgdec` CLI | 若无 CLI, 放到后续 |

**bit budget**:

256×256 图像共 65,536 pixels:

| RS-Token 对齐点 | bits/img | bpp |
|---|---:|---:|
| `k=1` | 2,560 | 0.0391 |
| `k=2` | 5,120 | 0.0781 |
| `k=4` | 10,240 | 0.1563 |

**压缩策略**:

- 对每张图在 val set 上或 per-image binary search quality, 找到不超过目标 bits 的最高质量压缩结果。
- test set 使用同一策略, 但必须记录实际 bits/img 的 mean ± std。
- 若 codec 无法达到 2,560 bits/img, 使用最低质量并报告 actual bits 与失败原因。

**信道策略**:

- 将压缩 bitstream 按同一 BER 做 bit flip。
- 信道: AWGN `0/+5/+10 dB`, Rayleigh `0/+5/+10 dB`。
- 解码失败不能丢弃:
  - `decode_failure = 1`
  - task accuracy 记为错误
  - PSNR/LPIPS 可同时报告 `valid_only` 与 `failure_as_black` 两个版本, 主文用前者 + failure rate, supplement 用后者。

**新增脚本**:

- `scripts/10_eval_classic_baselines.py`

**建议命令**:

```powershell
$py = "C:\Users\Windows11\.conda\envs\rstoken\python.exe"
& $py -X utf8 scripts/10_eval_classic_baselines.py `
  --methods webp,jpeg2000 `
  --budgets 2560,5120,10240 `
  --classifier_ckpt checkpoints/aid_classifier_resnet34/best.pt `
  --channel_types awgn,rayleigh `
  --snrs 0,5,10 `
  --channel_seeds 0,1,2 `
  --out_csv logs/e19_classic_baselines.csv
```

**输出列**:

`method, target_bits, actual_bits_mean, actual_bits_std, channel_type, snr_db, channel_seed, decode_failure_rate, cls_acc_all, cls_acc_valid, psnr_valid, lpips_valid`

**判据**:

- 至少 WebP 完整产出所有 budget × channel rows。
- 主文能放一张 “RS-Token vs WebP/JPEG2000” 表。
- 若外部 baseline 在低 SNR 大量 decode failure, 这是有效结果, 用于支撑 cliff-effect motivation。

**公平性备注**:

这个 baseline 是 unprotected compressed bitstream, 对传统 codec 很苛刻; 论文中必须写清这是“无额外 FEC 的 compress-then-transmit baseline”。若要主张对 JPEG2000+LDPC 的优势, 还需后续加入真正 channel coding。

---

### E20 · Rayleigh 0 dB stress slice with external context (P1)

**回答的问题**: H4。Rayleigh 0 dB 是 RS-Token 的 breakdown point, 但没有外部语境时不知道它是普遍困难还是方法弱点。

**实验设计**:

从 E17/E18/E19 直接抽取 Rayleigh 0 dB 行, 做一张 stress table:

| method | bits/img | task acc all | recon cls acc | PSNR | LPIPS | failure rate |
|---|---:|---:|---:|---:|---:|---:|
| RS-Token `k=1` | 2,560 | 待填 | — | 待填 | 待填 | 0 |
| RS-Token `k=4` | 10,240 | 不作为 task 主证据 | 待填 | 待填 | 待填 | 0 |
| WebP target 2,560 | 实测 | 待填 | 待填 | 待填 | 待填 | 待填 |
| WebP target 10,240 | 实测 | 待填 | 待填 | 待填 | 待填 | 待填 |
| JPEG2000 target 10,240 | 实测 | 待填 | 待填 | 待填 | 待填 | 待填 |

**新增脚本**:

- `scripts/11_make_v04_tables.py --table rayleigh0`

**判据**:

- 若所有 baseline 都接近随机或高 failure, 论文可写: Rayleigh 0 dB is a shared destructive boundary requiring FEC/HARQ/equalization。
- 若某 baseline 明显优于 RS-Token, 论文必须承认 RS-Token 在 severe fading 下不占优, 将贡献限定到 `SNR >= +5 dB`。

---

### E21 · Gao 2022 / DeepJSCC feasibility baseline (P1, optional for GRSL; required-ish for TGRS)

**回答的问题**: 如果目标从 IGARSS/arXiv 往 GRSL/TGRS 推进, 经典 codec 还不够, 需要至少一个 task-aware RS 或 JSCC baseline。

**两条路线**:

1. **Gao 2022 protocol-matched surrogate**
   - 复现难点: 原文 DRL block selection 细节、compressed sensing 参数、ResNet34 classifier。
   - 最低可行 surrogate: uniform block compressed sensing + 8-bit quantization + E16 classifier。
   - 风险: 不能声称完整复现 Gao, 只能写 “Gao-style protocol-matched shallow baseline”。

2. **DeepJSCC image transmission**
   - 训练 encoder/channel/decoder, 用 E16 classifier 评估重建图。
   - 工作量更大, 但对 TGRS 更有说服力。

**建议优先级**:

- IGARSS: 可不做, E19 classic baseline + E18 seeds 足够形成最小包。
- GRSL: 先做 Gao-style surrogate。
- TGRS/JSTARS: Gao-style + DeepJSCC 都应做。

**产物**:

- `experiments/baseline_feasibility_g16.md` 或 `logs/e21_feasibility.md`
- 明确记录哪些是完整复现, 哪些只是 surrogate。

---

### E22 · v0.4 table aggregation and claim audit (P0)

**回答的问题**: 新实验跑完后, 必须统一生成论文主表和 claim 边界, 避免正文继续写 single-seed 或 internal-only claim。

**新增脚本**:

- `scripts/11_make_v04_tables.py`

**输入**:

- E16 classifier clean metrics
- E17 task/reconstruction split csv
- E18 seed sweep csv
- E19 classic baseline csv
- E20 Rayleigh 0 dB stress table

**输出**:

- `logs/v04_tables/table1_seed_stats.csv`
- `logs/v04_tables/table2_task_path_mean_std.csv`
- `logs/v04_tables/table3_reconstruction_path.csv`
- `logs/v04_tables/table4_external_baseline.csv`
- `logs/v04_tables/table5_rayleigh0_context.csv`
- `logs/v04_tables/claim_audit.md`

**claim audit 规则**:

- 若某结果只有 seed=42, 表题必须写 `single-seed`。
- 若外部 baseline 只有 WebP, 正文只能写 “against a classic WebP compressed-bitstream baseline”, 不能泛化到 JPEG2000+LDPC。
- 若 channel coding 未做, 禁止写 “JPEG2000+LDPC cliff effect”; 只能写 “unprotected compressed bitstream cliff effect”。
- `SNR >= +5 dB` 是 graceful degradation 主张的默认边界; Rayleigh 0 dB 只作为 breakdown boundary。

---

## 4. 总耗时与建议执行顺序

### 已完成 (E1-E6)

| # | 实验 | 实际耗时 |
|---|---|:-:|
| E1 | 数据准备 | (一次性) |
| E2 | Stage 1 训练 | ~60 min |
| E3 | Stage 2 训练 | ~67 min |
| E4 | Stage 3 训练 | ~67 min |
| E5 | L0 probe (×2 ckpt) | ~10 min |
| E6 | Stage 4 信道评测 (×2 ckpt) | ~10 min |
| **合计** | | **~3.5 h** |

### 原计划 (E7-E15, 当前状态以 log 为准)

| # | 实验 | 优先级 | 代码改动 | 训练 | 评测 | 总时长 |
|---|---|:-:|:-:|:-:|:-:|:-:|
| E12 | 类数核对 | P1 | 0 | 0 | 5 min | 5 min |
| E9 | +10dB 多种子复测 | P0 | 5 行 | 0 | 10 min | 15 min |
| E7 | 分层 probe | P0 | 30 行 | 0 | 10 min | 30 min |
| E10 | Rayleigh 复测 | P1 | 20 行 | 0 | 10 min | 30 min |
| E8 | 教师消融 (OpenAI CLIP) | P0 | 30 行 | 70 min | 7 min | 80 min |
| E11 | 蒸馏权重扫 | P1 | 0 | 140 min | 10 min | 150 min |
| E13-15 | figure 章节 | P2 | ~50 行 | 0 | 30 min | 60 min |
| **合计** | | | **~135 行** | **~3.5 h** | **~80 min** | **~6 h** |

### 新增待跑 (E16-E22, v0.4 投稿前补强)

| # | 实验 | 优先级 | 代码改动 | 训练 | 评测 | 总时长 |
|---|---|:-:|:-:|:-:|:-:|:-:|
| E16 | clean AID classifier | P0 | 新脚本+配置 | 1-3 h | 5 min | 半天内 |
| E17 | task/reconstruction split | P0 | 新脚本 | 0 | 20-40 min | 1 h |
| E18 | 3 model seeds | P0 | 配置复制+聚合 | 4 个新训练 ≈ 5 h | 1-2 h | 1 天 |
| E19 | classic compression baseline | P0 | 新脚本 | 0 | 1-3 h | 半天 |
| E20 | Rayleigh 0dB stress table | P1 | 聚合脚本 | 0 | 10 min | 30 min |
| E21 | Gao/DeepJSCC feasibility | P1 | 文档/预研 | 待定 | 待定 | 1-7 天 |
| E22 | v0.4 table aggregation | P0 | 聚合脚本 | 0 | 10 min | 1 h |

### v0.4 建议执行顺序

```
Day 1 上午: E16 clean classifier
Day 1 下午: E17 口径拆分 seed=42 → E19 WebP/JPEG2000 外部 baseline 初版
Day 2:      E18 训练 rvq_baseline_s41/s43 + rvq_distill_s41/s43
Day 3 上午: E18 批量评测 + E20 Rayleigh 0dB stress table
Day 3 下午: E22 聚合 mean±std 主表 + claim audit
后续:       E21 Gao/DeepJSCC, 视投稿目标决定是否展开
```

### 建议执行顺序

```
Day 1 上午: E12 → E9 → E7 → E10
            (零训练, ~1.5h, 验证 motivation 是否需要修订)
Day 1 下午: E8 训练 (70 min) → 评测三连
Day 2 上午: E11 w=0.1 训练 → 评测 → w=1.0 训练 → 评测
Day 2 下午: E13-15 三张图 + arXiv short paper 草稿
```

---

## 5. 失败时的预案

每条预案都不需要全推倒, 这是 motivation 已经诚实标注边界条件的好处。

| 实验 | 不如预期 | 论文调整方向 |
|---|---|---|
| E7 分层 probe | L0+L1 比 L0 显著高 5pp+ | 改 motivation §3 "蒸馏让前 1-2 层都吃语义", k=1 承诺改 k ∈ {1,2} |
| E8 教师消融 | OpenAI CLIP 也到 80%+ | 卖点从"遥感专用基础模型"换成"分层蒸馏机制本身", §5 RemoteCLIP 单点改成"任何 V-L 基础模型" |
| E9 +10dB 复测 | mean < 3 dB | 阈值放宽到 2.5 dB(诚实)或扩到 100 epoch 重训 Stage 3 |
| E10 Rayleigh | k=1 在 Rayleigh 下劣于 k=4 | 承认"降级机制在 AWGN 下成立, Rayleigh 需追加误码控制", 留作 future work |
| E11 权重扫 | trade-off 不单调 | 在论文方法章节诚实展示曲线, 解释 "w 过小语义层信号不足、w 过大压重建", 选 w=0.5 是 frontier 上一点 |
| E12 类数 | 实际 29 类 | 论文 §3 数据集小节加一句排除理由 |

---

## 6. 一句话凝结

**E1-E15 解决的是“方法是否成立”的内部证据链; E16-E22 解决的是“能否投稿”的外部证据链。下一步不再继续堆内部消融, 而是先用 clean classifier 拆清 task/reconstruction 口径, 再用 3 个 model seeds 和至少 1 条 classic compression baseline 关闭审稿 P0。**

---

## 附录 G · 复盘笔记 (概念辨析)

> 这一节是 2026-06-01 与 Claude 对话时反复打磨的关键概念集合, 是"读懂 motivation + 实验"的最小基础。每条都是用户实际问过的疑点, 答完后整理回这里。

### G.1 PSNR / LPIPS 的方向

- **PSNR** 越**高**越好 (定义 = 10·log₁₀(MAX² / MSE), MSE 越小 PSNR 越大)。常见区间: < 20 差 / 25-30 一般 / > 35 很好
- **LPIPS** 越**低**越好 (感知距离, 距离 0 越近图越像)
- 两者方向相反, 都是衡量"和原图差多少"的指标
- 所以 E4 比 E3 的 PSNR -0.22 dB = 蒸馏让重建**变差**了 0.22 dB, 是**代价**, 不是改进

### G.2 为什么 E4 判据是 "PSNR 下降 ≤ 0.5 dB" 而不是 "上升"

加蒸馏后损失函数从 `L_recon + L_VGG` 变成 `L_recon + L_VGG + 0.5·L_distill`, 模型容量固定情况下两个目标抢参数, 重建目标必然被牺牲, **PSNR 下降是必然的工程代价**。0.5 dB 是 motivation §6.1-4 给的红线 — 在感知上几乎不可见, 用这点代价换 L0 分类 +24.7pp 是大赚。如果蒸馏代价超过 1 dB, trade-off 就不成立, 论文要修订。

### G.3 encoder 的下采样 + 升维

[motivation 链路] **256×256 RGB → 16×16 patch × 256 维 (= 256 个 token, 每个 256 维)**

| 维度 | 输入 | 输出 | 变化 |
|---|:-:|:-:|---|
| 空间 H × W | 256 × 256 | 16 × 16 | **下采样 16 倍** (不是 8 倍, 之前文档写错了) |
| 通道 C | 3 (RGB) | 256 (latent) | **升维** (每个 patch 用 256 维丰富特征描述) |

下采样发生在空间轴 (看更大块), 升维发生在通道轴 (用更多数描述每一块), 两件事在不同轴上发生, 不矛盾。

### G.4 flatten 不是线性层

flatten = **不带参数的形状重排**, 把多维张量摊平成更少维。这里把 [B, 256, 16, 16] 摊成 [B, 256 个 patch, 256 维], 方便 RVQ 逐 patch 量化。**它不学习, 不算东西, 只动数据形状**, 跟 nn.Linear 完全是两回事。

### G.5 codebook 维度 vs codebook 容量

| 数字 | 含义 | 配置项 |
|:-:|---|---|
| **1024** | codebook 容量 (词典大小, 有多少候选 codeword) | `codebook_size: 1024` |
| **256** | 每个 codeword 的维度 (与 patch 同维度, 才能算欧氏距离) | `latent_dim: 256` |

类比成词典: 词典里有 **1024 个词条**, 每个词条都用一个 **256 维向量**表示其语义。整个 L0 codebook 是 [1024, 256] 矩阵。

### G.6 L0_bow 是什么 (论文最关键的特征)

**字面**: L0 = RVQ 第 0 层, bow = Bag of Words 词袋。**把每张图的 L0 索引序列当成"一段文章", 数每个词出现的频率, 得到 1024 维向量**。

**步骤**:
1. 一张图 → encoder + L0 量化 → 256 个索引 [i₁, ..., i₂₅₆], 每个 ∈ {0..1023}
2. 数频率: `L0_bow[i] = (codeword i 出现次数) / 256`, 得到 1024 维稀疏向量
3. 这个向量是图的"语义指纹" — 不同类别的图有不同的 codeword 频率分布 (机场: 跑道+飞机的 codeword 高频; 森林: 草地+深绿 codeword 高频)

**为什么是论证上最硬的特征**: 信道里实际传的就是索引本身 (10 bit 一个), 接收方拿到索引后**无需查表**就能算 L0_bow。如果连这种最朴素的形式都能让单层 Linear 拿到 82.4%, 就证明"L0 索引在离散组合模式里已经编码了地物身份"。比 L0_emb (查表+池化) 是更弱的条件, 所以是更硬的证据。

### G.7 E5 的 linear probe 在做什么

```
一张 AID 图 → encoder + L0 量化 → L0 索引 → 数频率 → L0_bow (1024 维)
                                                      ↓ 单层 Linear (sklearn LogisticRegression)
                                                    30 类预测
```

**整条链路里真正训练的只有最后那一层 Linear**, encoder 和 codebook 都冻结。**双弱条件**:
- 特征弱: L0_bow 不查表, 只数频率
- 分类器弱: 单层 Linear 只学线性边界

蒸馏后从 57.7% → 82.4% (+24.7pp, motivation +5pp 红线的 5 倍)。在双弱条件下还能这么准, **数学上只能解释为 L0_bow 这个 1024 维空间里 30 类已经聚成 30 个线性可分的簇** — 即蒸馏把 RemoteCLIP 那套"30 类线性可分"的几何结构, 部分地灌进了离散索引选择模式里。

### G.8 AWGN 信道是什么

**A**dditive **W**hite **G**aussian **N**oise = 加性 / 白 / 高斯 / 噪声。最基础的信道模型: $y = x + n$, $n \sim \mathcal{N}(0, \sigma^2)$。

**为什么在我们项目里只用闭式 BER 公式而不真模拟波形**: BPSK over AWGN 等价于"每 bit 独立按 $\text{BER} = Q(\sqrt{2 \cdot \text{SNR}_{\text{lin}}})$ 概率翻转"。所以 [scripts/05_eval_channel.py](../scripts/05_eval_channel.py) 直接拆 bit + 翻转 + 重组, 跳过波形仿真。

**SNR vs BER 对照**:
| SNR (dB) | BER |
|:-:|:-:|
| -10 | 0.327 |
| -5 | 0.213 |
| 0 | 0.079 |
| +5 | 0.006 |
| +10 | 3.9e-6 |

**为什么默认 AWGN**: (1) 通信圈所有教材 / SemCom 论文的公共基线, 结果可横比; (2) 真实信道经过 LDPC + HARQ + 均衡器后, 残余误差近似 AWGN, 是合理的"理想物理层后剩余不确定性"近似。**AWGN 不模拟多径衰落 / 多普勒 / 干扰**, 这就是 E10 要补 Rayleigh 的原因。

### G.9 表名 "蒸馏版 · L0_bow 分类准确率" 的命名逻辑

E6 实验本质是**三维数据立方体**:
- 维度 1 · 模型版本 (蒸馏 / 无蒸馏)
- 维度 2 · 评测指标 (PSNR / LPIPS / L0_bow_acc / L0_emb_acc / zq_acc)
- 维度 3 · 实验条件 (SNR × k)

每张表必须**固定其中一个维度, 另两个做行 × 列**。表名 "蒸馏版 · L0_bow 分类准确率" 锁定了模型版本 + 评测指标, SNR 和 k 是表的内容轴 (行 × 列), 不需要写进标题。这跟 X-Y 坐标图的标题不会写 "X=1..100, Y=0..1" 是一样的逻辑。

**重要澄清**: "L0_bow" 是**评测分类时用的特征**, 不是说"只用 L0 一层"。表里 k=4 那列**信道里完整传 4 层**, 但分类时**仍然只取 L0 索引数频率**喂分类器。"k 传几层" 和 "分类用什么特征" 完全正交, 两件事。这么算是为了**纯粹观察 L0 的语义保真随 k 变化** — 如果 k=1 和 k=4 的 L0_bow 都接近 82%, 说明传更多层不破坏 L0 自己, 这是 E6 真正想验证的事。

### G.10 L1_bow 单独传输工程上不成立

RVQ 不是 4 份独立语义, 而是**残差递进**:
- L0: 量化 z → zq_L0, 残差 r1 = z - zq_L0
- L1: 量化 r1 → zq_L1, 残差 r2 = r1 - zq_L1
- L2 / L3 同理

所以 L1 不是"另一份语义", 它是**给定 L0 之后的残差** — 语义已被 L0 吃掉, L1 学到的天然是纹理 / 边缘 / 局部对比。

**只传 L1 不传 L0 在解码上不可用**: ResidualVQ 库的 `get_output_from_indices` 只支持"前 k 层"截取, 不支持跳过中间层。即使强行做, 接收端只拿 L1 残差, 主体 zq_L0 缺失, 重建会塌掉。

**但 L1_bow 单独跑 linear probe 是有意义的诊断工具** — 回答 "L1 离开 L0 之后还剩多少语义"。预测落在 60-75% 区间 (比 L0 低, 比无蒸馏 L0 高, 因为蒸馏让 encoder 整体吸收语义)。如果 L0_bow > L1_bow > L2_bow > L3_bow 单调下降, 就是层级分工干净的最直接证据。

### G.11 为什么要测重建质量 (PSNR/LPIPS) — 下游不是分类吗

论文卖的不是"只能分类", 是"**分类 + 重建可切换**"。motivation §3.1 两半边:

```
链路差时: 只传 L0  → 分类仍准确         ← 任务下限 (E5 / E6 分类列)
链路好时: 叠加 L1-L3 → 重建恢复像素质量  ← 质量上限 (E6 重建 PSNR 列)
```

**重建 PSNR 是论证后半句的唯一手段, 解决三件事**:

| 重建指标解决的问题 | 数据证据 | 没测会怎样 |
|---|---|---|
| 1. RVQ 多层是不是浪费? | +10dB 下 k=4-k=1 PSNR = +2.98 dB | 被问"为什么不用单层 VQ" |
| 2. 蒸馏代价是不是太大? | E4 vs E3 = -0.22 dB << 0.5 dB | 被问"为了语义牺牲了图像质量" |
| 3. 信道差时图能用吗? | 0dB 下 k=1 PSNR=16.4 (能粗看) | 不能解释"图差但能分类" |

**论文最重要的反差**: 分类 vs 重建**正好反着走** —
- k=1 vs k=4 在分类上**几乎相等** (传更多层不加分)
- k=1 vs k=4 在重建上**显著有别** (高 SNR 下 +3 dB)

这正是 motivation "**任务保真 ≠ 重建质量**" 的核心反差。同一个 RVQ 模型同时服务两类需求, 接收方按场景动态选 k:

| 接收方需要 | 选 k | 带宽 |
|---|:-:|:-:|
| 实时告警 / 分类 | 1 | 2560 bit/img |
| 应急指挥 / 区域识别 | 2 | 5120 |
| 测绘 / 高保真重建 | 4 | 10240 |

只测分类讲不出"动态切换"的故事, 论文卖点只剩半边。


## E23 amendment (2026-06-03)

The earlier E19 design deliberately scoped classic baselines as unprotected compressed bitstreams. E23 supersedes that limitation for classic codec baselines by adding a rate-1/2 systematic sparse LDPC + min-sum BP comparison under fixed transmitted-bit budgets. Use E19 only for unprotected-baseline claims and E23 for LDPC-protected WebP/JPEG2000 claims. Do not describe E23 as 5G NR LDPC.
