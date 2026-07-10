# RS-Token v0.6 论文修订日志

> 本日志记录 v0.5 → v0.6 的论文修订工作。所有实验数据来自 v0.5 实验日志（E35–E42 已全部完成训练 + 评估 + 聚合），本次只做论文集成，**无任何新训练 / 评估 / GPU 工作**。

## 任务范围

用户指令（2026-06-12）：**只产出中文 MD 版本** `rs_token_v0.6_zh.md`，暂不生成 LaTeX、PDF、bib、figure 脚本。

## 完成交付物

1. `paper_draft/latex/rs_token_v0.6_zh.md` —— 在 v0.5_zh.md 基础上整合 E35/E36/E37/E38/E39/E40/E41/E42 七项实验数据，外加 T3.1–T3.11 全部纯文本修正
2. `paper_draft/latex/v05_to_v06_changelog.md` —— 修订对照清单（节级 paragraph diff、表格 diff、bib diff、新增 appendix、关键叙事 deltas）

## 七项实验整合位置

| 实验 | 位置 | 备注 |
|---|---|---|
| E35 ADJSCC | §4.4a 新增节 + 表 4a + §5 trade-off 段 + 摘要 + 附录 D | "channel-robust" 措辞精确化 |
| E36 SNR sweep | §4.4 Result 末段 + 图 4 (Fig.4) | 4 子图等待 LaTeX 同步时由 build_fig4_snr_sweep.py 生成 |
| E37 All-layers 3-seed | §4.3 表 2 升级 + Result 段重写 | Rayleigh +5 deficit 实测 −10.83 pp（4.9σ），比 v0.5 的 −9.3 pp 更大 |
| E38 WebP 脚注 | 表 4 加 $^\dagger$ + §4.5 Result 末句 | k=1, k=2 WebP 行 |
| E39 NWPU recon | §4.6 表 5 扩展为 5 列 × 5 信道 + Result/Conclusion 加句 | 重建路径同样保留：clean k=4 PSNR 27.09 dB / recon-cls 80.59% on 45 类 |
| E40 paired init | §4.2 末段 + 附录 E.2 + 摘要 | 删除 v0.5 不严谨语句，加 paired Δ 五信道全数据 |
| E41 DCRC | §5 新段 + §3.1 CNN encoder justification + 附录 E.1 | 三子表（task / recon@k=4 / recon-sweep） |
| E42 K-ablation | §4.8 新增节 + §3.1 K=4 justification 改写 + 表 9 | K=4 是 Pareto compromise 而非 strict optimum |

## E38 + T3.1–T3.11 完成情况

11 项纯文本修正（详见 changelog §9）：
- ✅ E38 WebP 脚注
- ⚠️ T3.1（中文标点 "：" 正确，无需改；下次英文版同步时改 ASCII `:`）
- ✅ T3.2 transmitter wording
- ✅ T3.3 Fig.1 caption $g_\psi$ 统一
- ✅ T3.4 Eq. (8) loss 展开
- ✅ T3.5 Eq. (10) CSI 注解
- ✅ T3.6 §3.1 $C_\ell$ 形式化定义
- ✅ T3.7 §2.1 + bib 加 ADJSCC + SwinJSCC
- ✅ T3.8 §5 RemoteCLIP increment bounded bullet
- ✅ T3.9 §5 5G NR LDPC out of scope bullet
- ✅ T3.10 §5 scene-cls only / L0 grid limitation bullet
- N/A T3.11 bib 清理（v0.5 已通过 hygiene；v0.6 仅增补未删除）

## 不修改的文件

- v0.5 主稿 `rs_token_v0.5.tex` / `_zh.md` / `_v05.bib` / `.pdf` —— 已锁定版本
- `rstoken/checkpoints/` / `rstoken/logs/` / `rstoken/data/` —— 论文修订无需 GPU
- `rstoken/figs/` —— 仅 add（本次未 add，因为还没生成 SNR sweep figure；下次 LaTeX 同步时再加）

## 遗留事项（下次接英文 LaTeX 同步时执行）

1. 复制 `rs_token_v0.5.tex` → `rs_token_v0.6.tex`，按 `v05_to_v06_changelog.md` 同步全部修改
2. 复制 `rs_token_v05.bib` → `rs_token_v06.bib`，增补 `xu2022adjscc`、`yang2024swinjscc` 两条 entry
3. 编写 `paper_draft/build_fig4_snr_sweep.py`，读 `logs/paper_v05/e36_curve_*.csv` 4 份，输出 `rstoken/figs/fig_v06_snr_sweep.{pdf,png}`
4. 三遍 `pdflatex` + `bibtex` 编译；确认 0 undefined references / 0 LaTeX Error

## 时间记录

- 起始：2026-06-12
- 完成：2026-06-12
- 实际工作量：~1 小时（用户中途要求只做中文 MD，避免了 3–4 小时的 LaTeX / 编译 / figure 工作）
