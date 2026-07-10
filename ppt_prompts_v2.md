# PPT 图片生成提示词集 · v2

> 这一版由用户逐页定义内容, Claude 把每页的要点扩成 image2 pro 提示词。
> 工作流: 用户说"第 N 页讲 XXX, 要点是 A/B/C" → Claude 出版面草图 + 完整提示词 → 用户确认/调整 → 进入下一页。
>
> 本文件复用 v1 的视觉规范, 不再重新定义。新页直接写"## Page N — 标题"即可。

---

## 全局规则(每页都自动遵守, 提示词内不重复写)

### 视觉语言

- **深蓝 #1F3A68** = 结构 / 全局 / 通用机制 (主色)
- **暖橙 #E07B3C** = 语义 / 关键 / 降级故事线 (强调色)
- **一页只允许一条橙色故事线** —— 出现的所有橙色元素必须共属同一论点, 不撒胡椒面
- 字体: 中文思源黑体 / 鸿蒙黑体, 英文数字 Inter / Helvetica Neue
- 几何直线、SVG 友好、不要 stock 图标、不要 emoji、不要手绘抖动

### 每页固定结构

```
┌────────────────────────────────────────┐
│ ┌──┐ 主标题                  [logo 留白] │  ← 12% 高
│ │NN│ 副标题(中性灰)                     │
│ └──┘ ──────────────────────             │
│                                        │
│         主体内容(占 ~78% 高)             │
│                                        │
│ ─────────────                          │
│ → 底部脚注 / 主结论                      │  ← ~10% 高
└────────────────────────────────────────┘
```

- 编号方框: 深蓝细线 2px, Inter Medium 大字号, 内容随页定
- **右上角约 15% × 15% 区域必须完全留白** (用户后期加学校 logo)
- 标题区下方一条 1px 浅灰水平分隔线 (#D0D4DA)
- 底部一条主结论 + 一行小字补充

### 术语 / 用词

- 不出现 "Page X" / "上一页" / "下一页" 等工程标签 (回引用主题词)
- 用 **降级传输** , 不用 "优雅降级"
- 引用论文用 "RemoteCLIP (TGRS 2024)" 这类形式, 不写 "[1]"

### 通用风格提示词(每页拼接到具体内容前面)

```
风格: 现代学术幻灯片, Distill.pub / Anthropic 论文图美学, 科研组会汇报场景。
背景: 纯白 (#FFFFFF)。
主色: 深蓝 (#1F3A68)。
辅色: 中蓝 (#3D6FB5)。
浅色阶: #A8BBD8 / #C9D5E6 / #E4EAF3 用于层级递减的色阶填充。
强调色: 暖橙 (#E07B3C)。
       【全局规则】橙色仅用于"语义 / 关键 / 降级"这条故事线,
       一页里出现的所有橙色元素必须语义上属于同一论点,
       绝不在装饰性元素或次要图标上使用橙色。
中性灰: #666666 副标题与脚注; #D0D4DA 细分隔线。
高亮底: 极淡蓝 #EAF1FB 用于卡片填充; 极淡橙 #FCEFE5 只用于强调"语义层"那一行/块。
字体: 中文使用思源黑体 / 鸿蒙黑体, 正文 Regular、标题 Medium 或 Bold,
     字距正常, 绝不使用艺术字、手写体、书法体。
     英文 / 数字使用 Inter 或 Helvetica Neue。
线条: 严格几何直线, 锐利边角 (允许 4-6 px 圆角), 线宽统一, 绝不抖动。
排版: 网格对齐, 允许不对称布局以建立视觉节奏, 留白克制而干净。
画面比例: 16:9。
禁止项: 不要手绘风、不要墨水抖动、不要马卡龙色、不要纸张纹理、
       不要 3D、不要光泽渐变、不要霓虹色、不要 stock 风格图标 (图片框、标签、信号格等)、
       不要装饰性 emoji。
SVG 友好硬约束:
  - 只使用矩形、圆形、直线、单色填充、文字、简单箭头
  - 不使用渐变、阴影、模糊、半透明叠加、多路径复杂图标
特别要求:
  1. 所有中文必须清晰、印刷品质, 不要把汉字风格化。
  2. **右上角必须留白** —— 预留一块约占画面宽 15%、高 15% 的干净矩形区域,
     不放任何文字、图标、装饰、色块或边框线, 后期由用户添加学校 logo。
图标策略:
  - 整套 PPT 不使用 stock 图标, 用纯色矩形/圆形作为"信息载荷"的可视化代理
  - 例如表示"传输 N 层"就画 N 个并排小色块, 不画图片图标或标签图标
```

---

## 页面索引

(以下随用户逐页定义后填入)

| 编号 | 标题 | 主要论点 | 状态 |
|------|------|---------|------|
| 01   | 初步想法的来源 | 遥感 + 语义通信 + SpeechTokenizer 三种子交汇 | ✓ 已写 |
| 02   | 已有工作排查 · 搜索策略 | 八角度 + 四平台 + 漏斗筛选 18 → 7 | ✓ 已写 |
| 03   | 重点对手 · 2 篇已接收强对手详细版 | MOC-RVQ + ReVQom 各占半页 + 5 条以上差异 | ✓ 已写 |
| 04   | 参考工作 · 5 篇 preprint 候选 | ResiTok / VQ-VAE+OFDM / MAGC / FM-SemCom / Semantic-Loss(3+2 网格) | ✓ 已写 |
| 05   | 遥感中的"语义"到底是什么 | 4 层抽象阶梯 + 农田 vs 球场反例 + 95.95% 数字锚点 | ✓ 已写 |

---

<!-- 用户定义的页面从这里开始追加 -->

---

## Page 01 — 初步想法的来源 · 三个种子, 一个组合

### 用户定义的要点

- 整个汇报的开场, 讲"我这个研究怎么从三个独立概念碰出来的"
- 三个种子: 遥感场景 + 语义通信范式 + SpeechTokenizer 招式
- 三者交汇 → 初步想法: 把 SpeechTokenizer 的招式搬到遥感图像
- 强调这只是直觉, 为后面"已有工作排查"和"逻辑论证"埋伏笔

### 视觉骨架

- 上半 3 张等大种子卡片(深蓝)横排, 下半 1 张收口卡片(暖橙焦点)
- 三张种子卡向下汇聚到收口卡, 视觉传达"3 → 1 的收束 = 组合"
- 三种子卡都是深蓝中性, 只有底部收口卡用暖橙 → 故事在交汇处而非种子本身

### 完整提示词

```
[此处拼接通用风格提示词]

请生成一张 16:9 的现代学术风幻灯片, Distill.pub / Anthropic 论文图美学。
全图采用"三卡片横排 + 三向一收 + 底部收口卡"几何骨架。
本页是整个汇报的开场, 视觉上要传达"三个独立种子, 在交汇处碰出一个想法"。

【顶部标题区, 占画面约 12% 高度】
  - 左侧: 深蓝色细线方框(2px)内放编号"01",
    Inter Medium 大字号, 深蓝色 #1F3A68
  - 编号右侧: 主标题"初步想法的来源",
    思源黑体 Bold, 深蓝色
  - 主标题正下方副标题(中性灰):
    "三个种子, 一个组合"
  - 标题区下方一条 1px 浅灰水平分隔线
  - **右上角约 15% × 15% 区域完全留白**

【中区 — 三个种子卡, 占画面约 44% 高度】(横向并列):

  整体布局: 三张卡片等宽 (各占 30%) + 卡片间距 5%
  三张卡片高度严格相等, 顶部对齐、底部对齐。
  统一卡片样式:
    白底, 深蓝细边框 1.5px, 圆角 6px
    顶部一个深蓝实底色块横幅 + 白字"种子 N · 标题"
    内部纵向: 4 行关键词 (每行配深蓝细线"·" 起首)
    底部一行小字注解 (中性灰, 居中)

  【种子 1 — 遥感场景】(横幅"种子 1 · 遥感场景"):
    第 1 行(深蓝 Medium): "卫星下行 / 边缘遥感设备"
    第 2 行(深蓝 Medium): "信道质量随时间剧烈波动"
    第 3 行(深蓝 Medium): "带宽预算紧张"
    第 4 行(深蓝 Medium): "下游任务驱动: 地物分类 / 变化检测 / 灾害监测"
    底部小字: "应用域 · 真实需求"

  【种子 2 — 语义通信范式】(横幅"种子 2 · 语义通信范式"):
    第 1 行(深蓝 Medium): "不传像素, 传'含义'"
    第 2 行(深蓝 Medium): "离散 token + 索引传输"
    第 3 行(深蓝 Medium): "适配数字信道 (BPSK / OFDM)"
    第 4 行(深蓝 Medium): "MOC-RVQ / VQ-VAE+OFDM 已工程化"
    底部小字: "技术框架 · 当下范式"

  【种子 3 — SpeechTokenizer 招式】(横幅"种子 3 · SpeechTokenizer 招式"):
    第 1 行(深蓝 Medium): "RVQ 多层量化"
    第 2 行(深蓝 Medium): "HuBERT 蒸馏第一层 codebook"
    第 3 行(深蓝 Medium): "L1 = 音素语义 / L2-L4 = 音色细节"
    第 4 行(深蓝 Medium): "信道差时只传 L1 = 保语义丢音色"
    底部小字: "ICLR 2024 · 复旦 张鑫等"

【三向一收的箭头层, 占画面约 6% 高度】:
  在三张种子卡下方居中位置, 画一组向下汇聚的细线箭头:
    - 三根深蓝 1.5px 细实线分别从三张种子卡的底部中点出发
    - 三线汇聚到一个公共点 (位于画面水平中线)
    - 公共点处再向下画一个深蓝粗实心箭头"↓"
    - 公共点旁标注暖橙小字"三者交汇"

【底区 — 初步想法收口卡, 占画面约 26% 高度】:
  - 一个横跨整页 (左右各 5% 边距) 的横幅卡片
  - 极淡橙底色 #FCEFE5, 暖橙边框 2px, 圆角 6px
  - 卡片左侧 4px 厚的暖橙实心竖条作视觉锚边
  - 卡片左上角小标签: 暖橙实底色块 + 白字"★ 初步想法"
  - 卡片正文居中三行:
    第 1 行(深蓝 Bold 大字号):
      "把 SpeechTokenizer 的招式从语音搬到遥感图像"
    水平短分隔细线 (浅灰)
    第 2 行(深蓝 Medium, '遥感基础模型' 与 '第一层' 用暖橙加粗):
      "用遥感基础模型蒸馏 RVQ 第一层 codebook"
    第 3 行(深蓝 Medium, '任务保真' 用暖橙加粗):
      "让信道差时只传第一层在任务保真意义上真正可用"

【底部脚注, 占约 12% 高度】:
  - 上方 1px 浅灰水平分隔线
  - 主结论 (深蓝粗体大字, 左对齐):
    "→ 这只是初步直觉"
  - 紧接一行小字 (中性灰):
    "后续要回答: 别人做过没? 凭啥能走通? 真的能跑出结果吗?"

【全图视觉规则】
  - 三张种子卡严格等大、网格对齐, 体现"三个独立平等的来源"
  - **暖橙色仅用于**:
    (1) 三向一收箭头汇聚点旁的"三者交汇"标注
    (2) 底部"初步想法"收口卡 (背景 + 边框 + 锚边 + 标签 + 第 2/3 行的高亮词)
    —— 共属"交汇 → 想法"故事线
  - 三张种子卡严格使用深蓝 + 中性灰, 不使用任何橙色
  - 三向一收的视觉路径必须明显: 三根线汇聚 → 一根粗箭头 → 收口卡
  - 收口卡是全页视觉重心, 暖橙边框比种子卡的深蓝边框略粗 (2px vs 1.5px)
  - 收口卡正文第 1 行字号最大, 是整张图的"标题级"句子
  - 严格几何直线、无手绘抖动、无 stock 图标
  - 中文清晰、印刷级品质
  - 整张图传达"三个独立种子, 在交汇处碰出一个尚待验证的初步想法"
```

### 关键提醒

- 三张种子卡大小不一时:
  "All 3 seed cards must have IDENTICAL width and height, strict grid
   alignment. Each card has the same 4-line structure inside."

- 暖橙撒到种子卡上时:
  "The 3 seed cards are STRICTLY dark blue + neutral gray. NO orange
   anywhere inside them. Orange appears ONLY in the bottom orange
   focus card and the central '三者交汇' marker."

- 三向一收的视觉路径不清楚时:
  "Three thin dark-blue lines descend from the bottom-center of each
   seed card, converging at a single point on the horizontal centerline
   of the slide. From that point, a thicker dark-blue downward arrow
   '↓' continues to the orange focus card. This 3-to-1 convergence is
   the visual key of the slide."

- 底部收口卡不够醒目时:
  "The bottom orange card is the visual anchor — its border is 2px
   (thicker than the 1.5px of seed cards), it has a 4px solid orange
   left bar, and the first line ('把 SpeechTokenizer 的招式搬到遥感图像')
   must be the largest text on the entire slide."

- "三者交汇"标注被忽略时:
  "Next to the convergence point of the three thin lines, place a small
   orange label '三者交汇' — this is the only orange element in the
   middle band, marking the transition from neutral seeds to the
   orange conclusion below."

---

## Page 02 — 已有工作排查 · 搜索策略

### 用户定义的要点

- 这一页只讲方法论, 不讲找到的具体论文, 论文留给下一页
- 把"如何找"组织成: **八个检索角度 + 四个平台 + 漏斗筛选 18 → 7**
- 八角度 = 直接交集(4) + 邻域 / 已接收新论文(4), 合并成一轮呈现
- 四平台 = arXiv / Google Scholar / GitHub / paperswithcode (+ 会议官网接收列表)
- 漏斗收口数字 = 18 篇命中 → 7 篇精读, 唯一橙色用在最终"7"上
- 视觉故事: 系统性 + 多角度 + 多平台 → 收束到 7 篇高相关候选

### 视觉骨架

- 整页"上 12% 标题 + 中上 36% 八角度网格 + 中 16% 四平台横条 + 中下 24% 漏斗收束 + 下 12% 脚注"
- 八角度排成 2 行 × 4 列等大网格, 全深蓝, 不分类不染色
- 四平台横条放在角度网格下方, 视觉上像底座
- 漏斗用两条对称斜线 + 收口圆角矩形, 圆角矩形里写"7 篇"用唯一橙色
- 唯一橙色 = 漏斗收口的 "7 篇" 数字 + 底部脚注的"高相关候选"

### 完整提示词

```
[此处拼接通用风格提示词]

请生成一张 16:9 的现代学术风幻灯片, Distill.pub / Anthropic 论文图美学。
全图采用"八角度网格 + 平台底座 + 漏斗收束"几何骨架。
本页只讲检索方法论, 不出现任何论文名字。

【顶部标题区, 占画面约 12% 高度】
  - 左侧: 深蓝色细线方框(2px)内放编号"02",
    Inter Medium 大字号, 深蓝色 #1F3A68
  - 编号右侧: 主标题"已有工作排查 · 搜索策略",
    思源黑体 Bold, 深蓝色
  - 主标题正下方副标题(中性灰):
    "八个角度宽域检索 · 四个平台覆盖 · 漏斗筛选 18 → 7"
  - 标题区下方一条 1px 浅灰水平分隔线
  - **右上角约 15% × 15% 区域完全留白**

【中上区 — 八角度网格, 占画面约 36% 高度】
  - 段落顶部小标题: "▶ 八个检索角度"
    (深蓝 Medium, 左对齐)
  - 标题下方 2 行 × 4 列的等大网格, 共 8 张小卡片, 卡片间距均匀
  - 统一卡片样式:
      白底, 深蓝细边框 1.5px, 圆角 4px
      内部从上到下三段:
        段 1: 角度编号 (深蓝细边框小圆 + 数字 ① ~ ⑧)
        段 2: 角度名 (深蓝 Bold 中字号, 中文)
        段 3: 主搜索关键词 (中性灰小字, 英文)
      卡片高度严格相等, 顶/底对齐
  - 8 张卡片内容:
      ①  RVQ + 图像 + 通信         "RVQ semantic communication"
      ②  遥感压缩 + VQ              "RS image VQ codebook"
      ③  离散 token + 卫星          "discrete tokens satellite downlink"
      ④  遥感 + VQ-VAE              "RS SemCom VQ-VAE tokenizer"
      ⑤  基础模型 + SemCom          "foundation model semantic communication"
      ⑥  EO + 语义损失              "earth observation semantic loss"
      ⑦  tokenizer + 通信           "image tokenizer wireless transmission"
      ⑧  会议新接收(ICASSP/ICLR)    "ICASSP 2026 / ICLR 2026 accepted"

【中区 — 四平台横条, 占画面约 16% 高度】
  - 段落顶部小标题: "▶ 四个检索平台"
    (深蓝 Medium, 左对齐)
  - 标题下方一条横向长卡片, 内部 4 等分:
      白底, 深蓝细边框 1.5px, 圆角 4px
      4 个分区之间用 1px 浅灰垂直细线分隔
  - 4 个分区各放一行, 居中对齐, 内部从上到下:
      分区名 (深蓝 Bold 中字号)
      检索特异性 (中性灰小字)
  - 4 分区内容:
      arXiv               "preprint 主战场, 关键词宽域命中"
      Google Scholar      "引用反向追溯, 找祖论文与最新引用"
      GitHub              "验证是否真有开源代码"
      paperswithcode      "看 SOTA 排行 + 已接收新会议论文"

【中下区 — 漏斗收束, 占画面约 24% 高度】
  - 段落顶部小标题: "▶ 漏斗筛选"
    (深蓝 Medium, 左对齐)
  - 漏斗主体(居中):
      顶部宽 — 一条横向深蓝细线, 上方标注: "宽域命中 · 18 篇"
        (深蓝 Medium, 居中)
      两条对称斜线向下汇聚 (深蓝 1.5px 实线), 形成漏斗轮廓
      斜线上对称放两个中间过滤步骤的小标签 (深蓝细边框小圆角矩形 + 深蓝小字):
        左侧标签: "标题/摘要相关性"
        右侧标签: "全文精读差异化"
      底部窄 — 一个圆角矩形收口框:
        极淡橙底色 #FCEFE5, 暖橙边框 2px, 圆角 6px
        框内居中两行:
          第 1 行 (暖橙 Bold 大字): "7 篇"
          第 2 行 (深蓝 Medium 小字): "高相关候选"
  - 漏斗右侧空白区域可放一行小字注解 (中性灰小字, 左对齐):
      "—— 已接收 2 篇 (GLOBECOM 2024 / ICASSP 2026)"
      "—— preprint 5 篇 (arXiv 2024-2025)"

【底部脚注, 占约 12% 高度】
  - 上方 1px 浅灰水平分隔线
  - 主结论 (深蓝粗体大字, 左对齐):
    "→ 系统性检索, 收束到 7 篇高相关候选"
  - 紧接一行小字 (中性灰):
    "下一页对 7 篇逐一差异化, 已接收的 2 篇为重点对手"

【全图视觉规则】
  - 8 张角度卡严格等大、网格对齐, 体现"八个独立平等的角度"
  - 4 平台横条等分, 视觉上像角度网格的"底座"
  - 漏斗轮廓用两条简单斜线表达, 不用复杂图形
  - **暖橙色仅用于**:
    (1) 漏斗收口的"7 篇"圆角矩形 (背景 + 边框 + 数字)
    —— 共属"漏斗收束 · 7 篇高相关候选"故事线
  - 8 张角度卡 + 4 平台分区 严格使用深蓝 / 中性灰, 不使用任何橙色
  - 漏斗顶部"18 篇"用深蓝, 底部"7 篇"用暖橙, 形成"宽域 → 收束"的色彩递进
  - 严格几何直线、无手绘抖动、无 stock 图标
  - 中文清晰、印刷级品质
  - 整张图传达"系统化的检索方法 + 收束到精读候选的可信筛选过程"
```

### 关键提醒

- 8 张角度卡大小不一时:
  "All 8 search-angle cards must have IDENTICAL dimensions, strictly
   aligned to a 2-row × 4-column grid. Each card has the same 3-section
   internal structure (number circle / Chinese name / English keyword)."

- 任何角度卡被染上橙色时:
  "The 8 search-angle cards are STRICTLY dark blue + neutral gray. NO
   orange anywhere inside any of them. Orange appears ONLY in the funnel
   bottom rounded box marking '7 篇'."

- 漏斗被画成复杂图形时:
  "The funnel is just two symmetric thin lines converging downward, plus
   a rounded rectangle at the bottom. No fill, no decoration, no 3D effect."

- 平台横条不像底座时:
  "The 4-platform bar is a single horizontal long card with 4 equal-width
   sections separated by 1px vertical light-gray dividers. It sits
   directly under the 8-angle grid, visually anchoring it."

- 漏斗中间过滤步骤标签丢失时:
  "Two small rounded-rectangle labels sit symmetrically on the funnel's
   converging slopes, labeling the two filtering criteria
   ('标题/摘要相关性' on the left slope, '全文精读差异化' on the right slope).
   They are dark blue with thin border and small dark-blue text."

- '18 → 7' 的递进不明显时:
  "The funnel's TOP wide end shows '宽域命中 · 18 篇' in dark blue text.
   The funnel's BOTTOM narrow end shows '7 篇' in orange Bold large text
   inside a light-orange rounded rectangle. This dark-blue-to-orange
   color progression is the key visual signal of the filtering process."

---

## Page 03 — 重点对手 · 2 篇已会议接收的强对手详细版

### 用户定义的要点

- 整页只放 **MOC-RVQ + ReVQom** 2 张焦点卡, 各占半页, 详细版
- 焦点卡每张包含: 简称 / 完整标题 / 作者(带机构后缀, 不全列) / 来源 / 解决问题 / 核心架构关键数字 / 信道仿真状态 / 与本研究关键差异(多条)
- **不放**: 代码状态、数据集、主指标
- **必放**: 信道仿真状态(因为这是本研究的核心差异化点)
- 与本研究关键差异要写得多, 至少 4-5 条, 体现"已被接收但仍有清晰差异化"

### 视觉骨架

- 整页"上 12% 标题 + 中 78% 两张焦点大卡左右并列 + 下 10% 收束句"
- 两张焦点卡严格等大、左右对称, 各占画面宽度的 47%, 中间留 6% 间隙
- 两张卡共享相同的橙色样式 + 相同的 9 段内部结构
- 唯一橙色 = 两张焦点卡(背景 + 边框 + 简称 + 来源胶囊 + 差异化标签) + 底部脚注小字

### 完整提示词

```
[此处拼接通用风格提示词]

请生成一张 16:9 的现代学术风幻灯片, Distill.pub / Anthropic 论文图美学。
全图采用"两张大焦点卡左右对称 + 顶部回引 + 底部收束"几何骨架。
本页只展开 2 篇已会议接收的强对手, preprint 5 篇留给下一页。

【顶部标题区, 占画面约 12% 高度】
  - 左侧: 深蓝色细线方框(2px)内放编号"03",
    Inter Medium 大字号, 深蓝色 #1F3A68
  - 编号右侧: 主标题"重点对手 · 已会议接收",
    思源黑体 Bold, 深蓝色
  - 主标题正下方副标题(中性灰):
    "MOC-RVQ (GLOBECOM 2024) · ReVQom (ICASSP 2026) · 逐一深度差异化"
  - 标题区下方一条 1px 浅灰水平分隔线
  - **右上角约 15% × 15% 区域完全留白**

【中区 — 两张焦点大卡, 占画面约 78% 高度】(左右并列, 各占宽 47%, 中间 6% 间隙)
  - 两张卡片严格等高、顶/底对齐
  - 统一卡片样式:
      极淡橙底色 #FCEFE5, 暖橙边框 2px, 圆角 6px
      左上角小标签: 暖橙实底色块 + 白字 (内容见下)
      左侧 4px 厚的暖橙实心竖条作视觉锚边
      内部从上到下分 3 段, 段之间用 1px 浅灰水平细线分隔:

      ─── 段 1 · 身份(占卡片高 25%)───
        简称 (暖橙 Bold 巨字号)
        完整标题 (中性灰小字, 可两行)
        作者带机构后缀 (深蓝 Medium 小字, 一行, 用 / 分隔多个作者机构)
        来源胶囊 (暖橙细边框 + 暖橙字, 圆角 4px)

      ─── 段 2 · 内容(占卡片高 35%)───
        小标签 "▶ 解决什么问题" (深蓝 Medium 小字)
        一句话 (深蓝 Regular)
        小标签 "▶ 核心架构" (深蓝 Medium 小字)
        2-3 个关键数字配短语 (深蓝 Regular, 数字加粗)
        小标签 "▶ 信道仿真" (深蓝 Medium 小字)
        一句话 (深蓝 Regular, 关键状态用暖橙加粗)

      ─── 段 3 · 与本研究的关键差异(占卡片高 40%)───
        小标签 "✗ 与本研究的关键差异" (暖橙实底色块 + 白字)
        4-5 条差异 (深蓝 Regular, 每条以暖橙小方点 "▪" 起首):
          每条句子简洁, 一行内说完一个差异维度

  ─────────────────────────────────────────────
  左卡 · MOC-RVQ
  ─────────────────────────────────────────────
    左上角小标签: "★ 技术路线最接近"
    简称: "MOC-RVQ"
    完整标题(中性灰小字, 两行):
      "Multilevel Codebook-Assisted"
      "Digital Generative Semantic Communication"
    作者(深蓝 Medium 小字):
      "Yingbin Zhou · Yaping Sun (CUHK-Shenzhen / 鹏城实验室), et al."
    来源胶囊: "GLOBECOM 2024"

    ▶ 解决什么问题
      "数字星座下做生成式语义通信, 让 codebook 索引直接对齐调制星座点"

    ▶ 核心架构
      "MOC 头数 P=4 × 状态数 N=8 = 64-QAM 对齐"
      "RVQ 4 阶量化, 多层精度递进"
      "VGG19 语义引导损失 (γ=0.1)"

    ▶ 信道仿真
      "AWGN + 实数字星座调制 (BPSK / QPSK / 16-QAM / 64-QAM)"
      "—— 有信道仿真, 但限于自然图域"

    ✗ 与本研究的关键差异
      ▪ 应用域: 自然图(DIV2K / Flickr2K / FFHQ) ≠ 遥感卫星
      ▪ 蒸馏机制: 仅用 VGG19 感知损失, 无基础模型蒸馏
      ▪ 第一层语义: 4 阶 RVQ 仅精度递进, L0 不承载特殊语义
      ▪ 下游任务: 重建质量(PSNR / LPIPS) ≠ 任务保真(分类准确率)
      ▪ 降级传输: 没做 "只传 L0 仍可用" 的任务保真验证

  ─────────────────────────────────────────────
  右卡 · ReVQom
  ─────────────────────────────────────────────
    左上角小标签: "★ RVQ 路线新接收"
    简称: "ReVQom"
    完整标题(中性灰小字, 两行):
      "Residual Vector Quantization for"
      "Communication-Efficient Multi-Agent Perception"
    作者(深蓝 Medium 小字):
      "Dereje Shenkut · B.V.K. Vijaya Kumar (CMU ECE)"
    来源胶囊: "ICASSP 2026"

    ▶ 解决什么问题
      "V2X 多智能体协同感知, 把车-车之间传的特征图压缩 1000× 仍保检测精度"

    ▶ 核心架构
      "通道 bottleneck C_rr = 16 (256 → 16 通道)"
      "RVQ 阶数 n_q = 3, 跨 agent 共享 codebook"
      "压缩比 273× ~ 1365× (6-30 bpp)"

    ▶ 信道仿真
      "假设无错传输, 把信道损伤 / 丢包 / 自适应码率全列为 future work"
      "—— 完全没仿真信道"

    ✗ 与本研究的关键差异
      ▪ 应用域: V2X 车端 LiDAR BEV 特征 ≠ 卫星下行 RGB 遥感图
      ▪ 模态: LiDAR BEV ≠ 光学影像
      ▪ 信道: 无信道仿真 ≠ AWGN + BPSK + SNR 扫描
      ▪ 蒸馏机制: 无教师模型, 仅 commitment + 正交损失
      ▪ 第一层语义: 3 阶 RVQ 仅精度递进, L0 不承载语义身份
      ▪ 下游任务: 3D 目标检测 ≠ 遥感场景分类

【底部脚注, 占约 10% 高度】
  - 上方 1px 浅灰水平分隔线
  - 主结论 (深蓝粗体大字, 左对齐):
    "→ 已接收的 2 篇均有 4 个以上维度的差异化空间"
  - 紧接一行小字 (暖橙):
    "MOC-RVQ 缺基础模型蒸馏 · ReVQom 缺信道仿真 · 两者均缺第一层语义化"

【全图视觉规则】
  - 两张焦点卡严格等大、左右对称, 内部 3 段结构完全一致, 只在内容上区分
  - **暖橙色仅用于**:
    (1) 两张焦点卡 (背景 + 边框 + 锚边 + 简称 + 来源胶囊 + 差异化标签 + 信道仿真高亮)
    (2) 底部脚注第二行小字
    —— 共属"重点对手 · 已接收强对手 · 但仍有差异"故事线
  - 段 2 的"信道仿真"是关键差异化锚点, 文字内"有信道仿真但限于自然图域"
    和"完全没仿真信道"两句话用暖橙加粗高亮
  - 段 3 的差异化项目用暖橙小方点 "▪" 起首, 每条独立一行, 至少 5 条
  - 严格几何直线、无手绘抖动、无 stock 图标
  - 中文清晰、印刷级品质
  - 整张图传达"两个已被接收的强对手, 但每个都在 5 个以上维度上与本研究有清晰差异"
```

### 关键提醒

- 两张焦点卡大小不一时:
  "Both focus cards must have IDENTICAL width (47% of slide each), height,
   and internal 3-section structure. They are mirrors of each other in
   layout, only differing in content."

- 差异化条数不足时:
  "Each focus card MUST have AT LEAST 5 differentiation bullets in
   Section 3, each starting with a small orange '▪' marker. Do NOT merge
   bullets to save space — keep them as separate one-liners."

- 段间分隔不清晰时:
  "The 3 internal sections of each focus card are separated by 1px
   light-gray horizontal lines. Each section starts with a small dark-blue
   inline label ('▶ 解决什么问题', '▶ 核心架构', '▶ 信道仿真', '✗ 与本研究的关键差异')."

- 信道仿真状态不突出时:
  "In Section 2, under '▶ 信道仿真', the key status phrase
   ('有信道仿真, 但限于自然图域' for MOC-RVQ; '完全没仿真信道' for ReVQom)
   MUST be in orange Bold to visually anchor the differentiation."

- 作者机构后缀格式错误时:
  "Author line format: 'Name1 · Name2 (Institution), et al.' Use middle
   dot '·' between author names, parentheses for institution. Do NOT list
   all authors — top 2 + et al. is sufficient."

---

## Page 04 — 参考工作 · 5 篇 preprint 候选

### 用户定义的要点

- 这一页放 **ResiTok / VQ-VAE+OFDM / MAGC / FM-SemCom / Semantic-Loss** 5 张深蓝普通卡
- 排成 3+2 网格(第一行 3 张 / 第二行 2 张靠左)
- 每张普通卡 6 行: 简称 / 完整标题 / 作者(带机构) / 来源胶囊 / 核心做法 / 与本研究差异
- 全深蓝, 不出现橙色焦点(因为没有已接收 conference)
- 视觉重要性低于 Page 03, 用作"参考工作列表"

### 视觉骨架

- 整页"上 12% 标题 + 中 78% 5 卡 3+2 网格 + 下 10% 收束句"
- 5 卡严格等大、网格对齐, 第二行 2 张左对齐于上行第 1、2 列
- 全深蓝色系, 无橙色

### 完整提示词

```
[此处拼接通用风格提示词]

请生成一张 16:9 的现代学术风幻灯片, Distill.pub / Anthropic 论文图美学。
全图采用"3+2 候选卡网格 + 顶部回引 + 底部收束"几何骨架。

【顶部标题区, 占画面约 12% 高度】
  - 左侧: 深蓝色细线方框(2px)内放编号"04",
    Inter Medium 大字号, 深蓝色 #1F3A68
  - 编号右侧: 主标题"参考工作 · preprint 5 篇",
    思源黑体 Bold, 深蓝色
  - 主标题正下方副标题(中性灰):
    "未会议接收, 但思路相邻 · 简要差异化即可"
  - 标题区下方一条 1px 浅灰水平分隔线
  - **右上角约 15% × 15% 区域完全留白**

【中区 — 5 篇候选卡 3+2 网格, 占画面约 78% 高度】
  - 段落顶部一行小字回引 (深蓝 Medium 小字, 左对齐):
    "▶ ResiTok / VQ-VAE+OFDM / MAGC / FM-SemCom / Semantic-Loss"
  - 该行下方按 3 + 2 网格布局 5 张候选卡片:
    第 1 行 3 张, 第 2 行 2 张 (左对齐, 与上行第 1、2 列对齐, 第 3 列留空白)
  - 5 张卡片宽高严格相等, 间距均匀

  - 统一卡片样式:
      白底, 深蓝细边框 1.5px, 圆角 4px
      内部从上到下:
        第 1 行 (深蓝 Bold 中字号): 简称
        第 2 行 (中性灰小字, 较小字号): 完整标题(可两行)
        水平短分隔细线 (浅灰)
        第 3 行 (深蓝 Medium 小字): 作者带机构后缀 (前 2 + et al.)
        第 4 行 (深蓝细边框胶囊, 深蓝小字): 来源胶囊
        水平短分隔细线
        第 5 行 (深蓝 Medium 小字): "▶ 核心做法" + 一句话
        第 6 行 (深蓝实底圆角小标签 "✗" + 深蓝小字): 与本研究差异

  ─────────────────────────────────────────────
  位置 [1,1] — ResiTok
  ─────────────────────────────────────────────
    简称: "ResiTok"
    完整标题:
      "A Resilient Tokenization-Enabled Framework"
      "for Ultra-Low-Rate and Robust Image Transmission"
    作者: "Zhenyu Liu · Yi Ma (University of Surrey), et al."
    来源胶囊: "arXiv 2025"
    核心做法: "1D tokenizer + 分层 key/detail tokens + zero-out 训练"
    差异化("✗"): "非 RVQ + 无基础模型蒸馏"

  ─────────────────────────────────────────────
  位置 [1,2] — VQ-VAE + OFDM
  ─────────────────────────────────────────────
    简称: "VQ-VAE + OFDM"
    完整标题:
      "VQ-VAE Based Digital Semantic Communication"
      "with Importance-Aware OFDM Transmission"
    作者: "Ming Lyu · Hao Chen (BUPT), et al."
    来源胶囊: "arXiv 2025"
    核心做法: "VQ-VAE 共享 codebook + 重要性感知 OFDM 子载波分配"
    差异化("✗"): "单层 VQ, 没有'层级'概念"

  ─────────────────────────────────────────────
  位置 [1,3] — MAGC
  ─────────────────────────────────────────────
    简称: "MAGC"
    完整标题:
      "Map-Assisted Remote-Sensing Image Compression"
      "at Extremely Low Bitrates"
    作者: "Yixuan Ye · Ce Wang (Wuhan University), et al."
    来源胶囊: "arXiv 2024"
    核心做法: "VAE 压缩 latent + 扩散重建 + 矢量地图条件"
    差异化("✗"): "传连续 latent, 不传 codebook 索引"

  ─────────────────────────────────────────────
  位置 [2,1] — FM-SemCom
  ─────────────────────────────────────────────
    简称: "FM-SemCom"
    完整标题:
      "Foundation Model-Based Adaptive Semantic Image"
      "Transmission for Dynamic Wireless Environments"
    作者: "Fangyu Liu · Peiwen Jiang (SEU), et al."
    来源胶囊: "arXiv 2025"
    核心做法: "分割图 + 双扩散(信道估计 + 接收端重建)"
    差异化("✗"): "基础模型作特征提取器, 非蒸馏教师"

  ─────────────────────────────────────────────
  位置 [2,2] — Semantic-Loss
  ─────────────────────────────────────────────
    简称: "Semantic-Loss"
    完整标题:
      "A Semantic-Loss Function Modeling Framework"
      "With Task-Oriented Machine Learning Perspectives"
    作者: "Ti Ti Nguyen · Symeon Chatzinotas (Luxembourg SnT), et al."
    来源胶囊: "arXiv 2025"
    核心做法: "EO 任务保真评测范式 + 4 个任务模型(EfficientViT 等)"
    差异化("✗"): "不做 codebook, 做失真-任务损失建模"

【底部脚注, 占约 10% 高度】
  - 上方 1px 浅灰水平分隔线
  - 主结论 (深蓝粗体大字, 左对齐):
    "→ preprint 5 篇均与本方案存在本质差异"
  - 紧接一行小字 (中性灰):
    "重点对手集中在 Page 03 的 2 篇已接收会议论文"

【全图视觉规则】
  - 5 张候选卡严格等大、3+2 网格对齐, 第二行 2 张严格左对齐于上行第 1、2 列
  - 第二行第 3 列必须是干净空白(不放占位卡, 不放装饰)
  - **整页严格使用深蓝 + 中性灰, 不出现任何橙色** —— 因为这一页是参考工作, 不是重点对手
  - 5 张卡片的差异化标签视觉一致 (同位置、同 "✗" 图标、同字号、同深蓝色)
  - 来源胶囊样式统一 (深蓝细边框 + 深蓝字)
  - 严格几何直线、无手绘抖动、无 stock 图标
  - 中文清晰、印刷级品质
  - 整张图传达"5 篇相邻方向的参考工作, 视觉重要性弱于 Page 03 的 2 篇焦点"
```

### 关键提醒

- 5 张卡大小不一时:
  "All 5 candidate cards must have IDENTICAL dimensions, strictly aligned
   to a 3+2 grid (3 in row 1, 2 in row 2). Row 2 cards are LEFT-aligned
   with columns 1 and 2 of row 1 — column 3 of row 2 is empty whitespace."

- 任何一张卡被染上橙色时:
  "This page has NO orange focus cards. All 5 cards are STRICTLY white
   fill with dark blue borders. Orange is reserved for Page 03 only."

- 第二行第 3 列被填充时:
  "Position [2,3] MUST be empty — no placeholder card, no decorative
   element, no orange highlight, just clean white space."

- 来源胶囊不一致时:
  "All 5 source pills use IDENTICAL style: dark blue thin border + dark
   blue text. Content varies between 'arXiv 2024' and 'arXiv 2025'."

- 作者机构后缀格式不一致时:
  "All 5 author lines follow IDENTICAL format:
   'Name1 · Name2 (Institution), et al.'
   Use middle dot '·' between author names. Use parentheses for the
   institution. Do NOT list all authors."

---

## Page 05 — 遥感中的"语义"到底是什么

### 用户定义的要点

- 这一页要回答 motivation 链条上一个关键定义问题: **"遥感语义" 究竟指什么**
- 核心论点: **语义 ≠ 像素 / 边缘 / 纹理 / 颜色 / 频率 等低级视觉特征, 语义 = 地物身份 / 场景类别**
- 论证方式 = "层次阶梯"+ "反例对照" 双手段:
  - 左侧: 自底向上 4 层抽象阶梯 (像素 → 低级 → 中级 → 语义), 顶层用暖橙高亮
  - 右侧: 反例对照(农田 vs 足球场草坪), 低级特征几乎相同但语义完全不同
- 底部锚点: "语义可分性 = RemoteCLIP 在 AID 95.95% linear probe", 这是 motivation 量化指标
- 与 MOC-RVQ 划清界限: VGG 感知损失只到中级, 不算遥感语义

### 视觉骨架

- 整页"上 12% 标题 + 中 78% 左阶梯 60% + 右反例 40% + 下 10% 底部锚点"
- 左侧: 4 层抽象阶梯, 自底向上, 颜色递进 (灰 → 浅蓝 → 中蓝 → 暖橙)
- 右侧: 反例对照面板, 表格式呈现"低级特征相同 + 语义不同"
- 唯一橙色 = 阶梯顶层"语义层" + 反例右下"语义✗" + 底部"95.95%"数字

### 完整提示词

```
[此处拼接通用风格提示词]

请生成一张 16:9 的现代学术风幻灯片, Distill.pub / Anthropic 论文图美学。
全图采用"左侧 4 层抽象阶梯 + 右侧反例对照面板"几何骨架。
本页是 motivation 链条上的关键定义页, 回答"遥感语义到底是什么"。

【顶部标题区, 占画面约 12% 高度】
  - 左侧: 深蓝色细线方框(2px)内放编号"05",
    Inter Medium 大字号, 深蓝色 #1F3A68
  - 编号右侧: 主标题"遥感中的'语义'到底是什么",
    思源黑体 Bold, 深蓝色
  - 主标题正下方副标题(中性灰):
    "语义 ≠ 像素 / 边缘 / 纹理, 语义 = 地物身份"
  - 标题区下方一条 1px 浅灰水平分隔线
  - **右上角约 15% × 15% 区域完全留白**

【中区左侧 — 4 层抽象阶梯, 占画面宽 56%, 高 78%】
  - 段落顶部小标题: "▶ 4 层抽象阶梯"
    (深蓝 Medium, 左对齐)
  - 标题下方一个垂直堆叠的 4 层阶梯, 自底向上, 每层是一个等宽矩形条
  - 4 层从下到上颜色逐渐"语义化": 浅灰 → 极浅蓝 → 中浅蓝 → 极淡橙
  - 4 层等高, 之间用 4px 间距分隔, 每层左侧凸出 4px 厚色块作锚边
  - 每层内部从左到右分 3 段:
      段 A (宽 25%): 层级名称 (大字 Bold)
      段 B (宽 35%): 这一层是什么 (中性灰小字, 一句话)
      段 C (宽 40%): 遥感图里的具体例子 (深蓝 Regular 小字, 用 / 分隔多个例子)

  ─────────────────────────────────────────────
  阶梯第 1 层 (最底, 浅灰底)
  ─────────────────────────────────────────────
    段 A: "像素层"
          (深蓝 Bold)
          英文小字: "pixel level"
    段 B: "每个位置的 RGB 数值"
    段 C: "一张图 = 256×256×3 个浮点数"

  ─────────────────────────────────────────────
  阶梯第 2 层 (浅蓝底)
  ─────────────────────────────────────────────
    段 A: "低级视觉"
          (深蓝 Bold)
          英文小字: "low-level visual"
    段 B: "局部统计 / 几何性质"
    段 C: "边缘 / 纹理 / 颜色 / 梯度方向 / 空间频率"

  ─────────────────────────────────────────────
  阶梯第 3 层 (中蓝底)
  ─────────────────────────────────────────────
    段 A: "中级视觉"
          (深蓝 Bold)
          英文小字: "mid-level visual"
    段 B: "形状 / 物体部件 / 局部 pattern"
    段 C: "圆形屋顶 / 矩形耕地 / 平行跑道线"

  ─────────────────────────────────────────────
  阶梯第 4 层 (最顶, 极淡橙底色 #FCEFE5, 暖橙边框 2px, 圆角 4px)
  ─────────────────────────────────────────────
    段 A: "语义层"
          (暖橙 Bold 大字, 比下面 3 层字号略大)
          英文小字 (暖橙): "semantic level"
          右侧追加暖橙小标签 "★ 我们要的"
    段 B: "地物身份 / 场景类别"
    段 C: "机场 / 港口 / 农田 / 河流 / 居民区"
          (深蓝 Regular, 关键词不加粗, 让结构本身突出)

  阶梯右侧画一个垂直深蓝箭头 ↑, 长度贯穿 4 层, 旁注小字:
    底部 (中性灰): "看像素"
    中部 (中性灰): "看局部统计"
    顶部 (暖橙): "看是什么"

【中区右侧 — 反例对照面板, 占画面宽 38%, 高 78%】
  - 段落顶部小标题: "▶ 反例 · 低级特征几乎相同, 语义完全不同"
    (深蓝 Medium, 左对齐)
  - 标题下方一个白底深蓝边框的对照表, 圆角 6px
  - 顶部表头两栏 (深蓝实底色块 + 白字, 居中):
      左栏: "农田"
      右栏: "足球场草坪"
  - 表头下方 5 行对照, 每行格式:
      左侧 30%: 维度名 (深蓝 Medium)
      中间 35%: 农田值
      右侧 35%: 球场值
      行末 (与该行对齐): "✓" 或 "✗" 标识(具体见下)
  - 5 行内容:
      第 1 行(深蓝 Regular):
        颜色      |  绿色      |  绿色      |  ✓ 相同
      第 2 行(深蓝 Regular):
        纹理      |  规则周期   |  规则周期   |  ✓ 相同
      第 3 行(深蓝 Regular):
        梯度方向  |  单一方向   |  单一方向   |  ✓ 相同
      第 4 行(深蓝 Regular):
        空间频率  |  中频       |  中频       |  ✓ 相同
      水平分隔细线 (浅灰)
      第 5 行(暖橙 Bold, 整行高亮极淡橙底色):
        语义      |  农业用地   |  体育设施   |  ✗ 完全不同
  - 表格下方一行小字注解 (中性灰小字):
    "→ 低级特征看不出区别, 必须靠基础模型才能区分"

【底部锚点, 占约 10% 高度】
  - 上方 1px 浅灰水平分隔线
  - 主结论(深蓝粗体大字, 左对齐):
    "→ 语义可分性 = AID 30 类的 linear probe 准确率"
  - 紧接一行(暖橙 Bold 数字 + 深蓝小字):
    "RemoteCLIP 在 AID 上 = 95.95%   ↔   颜色直方图 ≈ 40-50%"

【全图视觉规则】
  - 左侧阶梯 4 层等宽等高, 严格垂直堆叠, 不出现倾斜或透视
  - 阶梯 4 层颜色递进必须明显: 浅灰 → 浅蓝 → 中蓝 → 极淡橙
  - **暖橙色仅用于**:
    (1) 阶梯第 4 层 (语义层) 的底色 + 边框 + 锚边 + 层级名 + "★ 我们要的"标签 + 段 C 图标
    (2) 阶梯右侧箭头顶部小字 "看是什么"
    (3) 反例表第 5 行 "语义" (整行高亮极淡橙底)
    (4) 底部锚点的 "95.95%" 数字
    —— 共属"语义层 = motivation 锚点"故事线
  - 阶梯第 1-3 层 + 反例表第 1-4 行严格使用深蓝 / 中性灰, 不用任何橙色
  - 反例表第 1-4 行的 "✓ 相同" 用深蓝色
  - 反例表第 5 行的 "✗ 完全不同" 用暖橙色, 整行底色用极淡橙
  - 阶梯右侧箭头 ↑ 是单一深蓝直线, 不画装饰性分叉或渐变
  - 严格几何直线、无手绘抖动、无 stock 图标
  - 中文清晰、印刷级品质
  - 整张图传达"4 层抽象 + 反例对照, 共同锁定'遥感语义 = 地物身份'这个定义"
```

### 关键提醒

- 阶梯各层高度不一时:
  "All 4 ladder levels must have IDENTICAL height. They are stacked
   vertically with 4px gaps between layers. Only the colors and the top
   level's border thickness differ."

- 顶层语义层不够突出时:
  "The top level (语义层) is the visual focus. It uses a 2px orange border
   (other levels have no border or thin gray border), light orange fill
   (#FCEFE5), and the level name '语义层' is in BOLD orange with a slightly
   larger font than the lower 3 levels. A small '★ 我们要的' label appears
   to the right of '语义层'."

- 颜色递进不明显时:
  "The 4 levels MUST show a clear color progression bottom-up:
     Level 1 (像素层): light gray fill #F0F0F0
     Level 2 (低级视觉): very light blue #E4EAF3
     Level 3 (中级视觉): medium light blue #C9D5E6
     Level 4 (语义层): very light orange #FCEFE5
   This visual gradient signals the 'abstraction goes up'."

- 反例表"✓"和"✗"位置不对应时:
  "The 5 comparison rows of the right-side table all have a ✓/✗ marker
   at the row's far right. Rows 1-4 use dark blue '✓ 相同' for low-level
   features. Row 5 uses orange '✗ 完全不同' with the WHOLE ROW shaded in
   light orange #FCEFE5 to anchor the contrast."

- 底部锚点数字不显眼时:
  "The bottom anchor line places '95.95%' in BOLD orange large font,
   contrasted against '40-50%' in dark blue smaller font. This 95.95% is
   THE quantitative anchor for the entire motivation chain — visually
   it must pop."

- 把阶梯画成"金字塔"形状时:
  "The 4 levels are STACKED RECTANGLES of EQUAL WIDTH, NOT a pyramid.
   They are aligned to the same left and right edges. Do NOT taper the
   width going up — that would mislead the audience into thinking 'higher
   = less'. The point is 'higher = more abstract', not 'higher = smaller'."

- 把"语义"和"低级特征"用同一颜色时:
  "Levels 1-3 use blue/gray cool tones, level 4 (语义层) uses orange warm
   tone. The cool→warm shift at the top is essential to communicate
   'this is qualitatively different from below — this is what we want'."
