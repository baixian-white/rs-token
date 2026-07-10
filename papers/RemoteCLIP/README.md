# RemoteCLIP · 文件夹清单

## 资料

| 文件/目录 | 内容 | 状态 |
|---|---|---|
| `notes.md` | 精读笔记(身份证、数据构造、架构、实验、与本研究关系、复现细节、镜像 SpeechTokenizer、引用) | ✓ 已写 |
| `paper.pdf` | arXiv 2306.11029 完整 PDF(14 MB) | ✓ 已下载 |
| `code/` | GitHub 官方仓库 clone(github.com/ChenDelong1999/RemoteCLIP) | ✓ **真代码**(notebook + retrieval.py + README + assets) |
| `code/demo.ipynb` | 模型加载与基本推理 demo | ✓ |
| `code/RemoteCLIP_colab_demo.ipynb` | Colab 版 demo | ✓ |
| `code/retrieval.py` | RSITMD/RSICD/UCM 上的检索评测脚本 | ✓ |
| `code/LICENSE` | Apache-2.0 | ✓ |

## 关于代码

仓库**有真代码**(对比 MOC-RVQ 是空仓):
- demo notebook 演示如何用 OpenCLIP 加载 RemoteCLIP 权重
- `retrieval.py` 复刻论文的 retrieval 评测
- 无训练脚本(论文是用 ITRA 框架训的, 训练代码留在 ITRA 仓库)

**模型权重**在 HuggingFace:
- `chendelong/RemoteCLIP` 提供 RN50 / ViT-B-32 / ViT-L-14 三档
- 本研究 Stage 3 用 **ViT-B-32**

**训练数据**也已开源(2024-04-26 释出):
- `gzqy1026/RemoteCLIP` 包含 RET-3 + SEG-4 + DET-10 完整数据

## 在本研究中的角色

**双重角色**:

1. **论证命门**(motivation 章节)
   - 论据: AID linear probe 95.95%
   - 推论: 遥感地物语义在 RemoteCLIP 特征空间里是可分离的, 因此蒸馏到 RVQ L0 在原理上可行

2. **Stage 3 蒸馏教师**(实际训练)
   - 提取 patch-level token embeddings 作为 teacher signal
   - 通过 STE + 余弦/KL loss 把语义灌进 L0 codebook
   - 训练完成后丢弃, 推理时只保留 L0 codebook

## 引用

```bibtex
@article{liu2024remoteclip,
  title={RemoteCLIP: A Vision Language Foundation Model for Remote Sensing},
  author={Liu, Fan and Chen, Delong and Guan, Zhangqingyun and Zhou, Xiaocong and Zhu, Jiale and Ye, Qiaolin and Fu, Liyong and Zhou, Jun},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={62},
  pages={1--16},
  year={2024},
  publisher={IEEE},
  doi={10.1109/TGRS.2024.3390378}
}
```

## 相关链接

- arXiv: https://arxiv.org/abs/2306.11029
- GitHub: https://github.com/ChenDelong1999/RemoteCLIP
- HF 权重: https://huggingface.co/chendelong/RemoteCLIP
- HF 数据: https://huggingface.co/datasets/gzqy1026/RemoteCLIP
- TGRS DOI: https://doi.org/10.1109/TGRS.2024.3390378
