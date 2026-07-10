# RemoteTokenizer · 遥感语义通信项目

> 用 RemoteCLIP 蒸馏 RVQ 第一层 codebook,让"信道差时只传第一层"在任务保真意义上真正可用。
> Motivation 详见上级目录的 [motivation.md](../motivation.md)。

---

## 协作分工

| 角色 | 负责 |
|------|------|
| Claude | 写代码 / 设计实验 / 调超参 / 读 log 找问题 |
| 你的机器 | 执行长时训练 (一次 VQ-VAE 训练通常 3-12 小时) |
| 你 | 按回车启动训练 / 贴日志或 wandb 链接 / 审结果 / 把控大方向 |

**Claude 不会主动启动 GPU 长任务**。任何 `python train.py` 类命令都需你手动确认后再跑。

---

## 目录结构

```
rstoken/
├── configs/        训练超参 yaml
├── data/           数据集软链接 / 缓存索引
├── models/         模型定义 (encoder, RVQ, decoder, distillation head)
├── scripts/        可执行入口 (00_check_env.py, 01_train_vqvae.py, ...)
├── experiments/    实验脚本与对比基线
├── checkpoints/    训练保存的 ckpt
├── logs/           训练 log
├── notebooks/      探索性分析
├── environment.yml conda 环境定义
└── README.md       本文件
```

---

## 环境搭建 (一次性, 你来执行)

GPU: **RTX 5070 Ti / 16GB / Blackwell sm_120**。普通 stable PyTorch 不支持 sm_120, 必须用 cu128 wheel (PyTorch ≥ 2.7)。

```powershell
# 1. 建 conda 环境
conda env create -f environment.yml
conda activate rstoken

# 2. 装 Blackwell 兼容的 PyTorch (environment.yml 故意没装这个,
#    因为 conda 通道滞后, 必须从 pytorch 官方 cu128 channel 拉)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. sanity check (秒级, 几乎不占显存, 不影响别的训练)
python scripts/00_check_env.py
```

预期输出:三个 `[OK]`:PyTorch+CUDA / ResidualVQ / open_clip。任意一个 `[FAIL]` 截图给 Claude 看。

---

## RemoteCLIP 权重下载 (一次性, 你来执行)

已自动下载完成,位置 `checkpoints/remoteclip/RemoteCLIP-ViT-B-32.pt` (578 MB)。

如需重新获取: https://huggingface.co/chendelong/RemoteCLIP

---

## 数据集 AID (一次性)

下载: https://captain-whu.github.io/AID/ (~2.6 GB, 30 类, 10000 张 600×600 RGB)

解压后假设结构:
```
<某个目录>/AID/
  Airport/
    airport_1.jpg
    ...
  BareLand/
  ...
```

切分索引 (按类分层 8/1/1, 不复制图像):
```powershell
conda activate rstoken
python scripts/01_prepare_aid.py --aid-root "<某个目录>/AID"
```

输出 `data/AID_splits/{train,val,test}.csv` + `classes.txt`。后续 DataLoader 直接读 csv。

---

## 当前进度

- [x] Stage 0: 项目骨架 + sanity check 脚本
- [x] Stage 0: conda 环境 rstoken 已建
- [x] Stage 0: PyTorch cu128 已装
- [x] Stage 0: RemoteCLIP-ViT-B-32 权重已下载
- [ ] Stage 0: 跑 sanity check 验证环境
- [ ] Stage 0: 下载 AID 数据集 + 跑 prepare_aid
- [ ] Stage 1: 单层 VQ-VAE 在 AID 上跑通重建
- [ ] Stage 2: 换成 ResidualVQ 多层
- [ ] Stage 3: 加 RemoteCLIP 蒸馏 (Q1 关键节点 — 验证论据 A)
- [ ] Stage 4: AWGN 信道仿真 (Q2 — 验证优雅降级)
- [ ] Stage 5: 完整对比实验 + 出图 (Q3 — vs MOC-RVQ / DeepJSCC / JPEG2000)
