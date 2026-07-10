# Papers · 重点关注论文资料库

> 每篇论文一个独立文件夹, 内含:
> - `README.md` —— 文件夹清单 + 资源状态
> - `notes.md` —— 精读笔记(必有)
> - `paper.pdf` —— 论文 PDF(从 arXiv 下载)
> - `code/` —— 官方源码(git clone, 若开源)
> - `figures/` —— 论文里值得截下来引用的图(可选)
> - `bibtex.bib` —— 引用 entry(可选, 通常已嵌在 notes.md)

## 索引

| 文件夹 | 简称 | 角色 | notes | PDF | 代码 |
|---|---|---|:---:|:---:|:---:|
| `MOC-RVQ/` | MOC-RVQ | 头号对手 (GLOBECOM 2024) | ✓ | ✓ | ⚠ 占位仓 |
| `ReVQom/` | ReVQom | 新强敌 (ICASSP 2026) | ✓ | ✓ | ✗ 未释出 |
| `RemoteCLIP/` | RemoteCLIP | 论证命门 (TGRS 2024) + 蒸馏教师 | ✓ | ✓ | ✓ 真代码 |

## 命名规范

- 文件夹名用论文**简称**, 不用 arXiv ID
- 简称尽量与论文里自命名一致(MOC-RVQ / ReVQom / RemoteCLIP / SpeechTokenizer 都是论文自命名)
- 多词论文用连字符: `Foo-Bar` 而非 `Foo Bar`

## 处理流程(每加一篇)

1. WebFetch arXiv HTML / GitHub README → 写 `notes.md`
2. `curl -L -o paper.pdf https://arxiv.org/pdf/<arxiv_id>` 下载 PDF
3. 若有官方代码: `git clone <github_url> code/`(若仓库为空也克隆作占位)
4. 写 `README.md` 说明文件夹内容、代码状态

## 与 literature_survey.md 的关系

- `literature_survey.md` 是**全部**检索到的论文的汇总表(20+ 篇),保持精简
- `papers/<简称>/` 是**重点关注的少数论文**的完整资料(笔记 + PDF + 代码),可详细
- 升级标准: 与本研究存在直接竞争 / 需深度差异化 / 需复刻对比 / 是论证命门
