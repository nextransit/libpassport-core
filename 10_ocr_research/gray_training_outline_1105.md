# MRZ OCR 灰度训练大纲（v1.2）

| 项 | 内容 |
|---|---|
| 版本 | v1.2 / 2026-11-05（v1.1 依据 GUI 1380 样本集像素实测，更正 §0 结论 1、§2 现状表，新增 §2.1；v1.2 新增 §7 训练目标/Loss/指标、§8 Exact 率瓶颈突破、附录 C 强制 Skill，重排 §9–§11） |
| 状态 | **目标态规范（Target State）**，非当前实现描述 |
| 适用引擎 | `6_mrz_ocr` C 侧 CNN / CRNN 字符分类头；TD3 (2×44) 与 CS/CD (30) 双布局 |
| 唯一验收真值 | `10_ocr_research/gt_review_loop.xlsx`（"GT对比" sheet 的 GT line1 / GT line2） |
| 约束基线 | `AGENTS.md`（TD3 零退化红线、98% / 95%、fuse ≥ 2.0 张/秒） |
| 强制流程 | `AGENTS.md`「模型训练与识别 Skill 强制使用」→ 入口 `.workbuddy-ai/skills/ai-research-skills-mrz/SKILL.md`；路由见附录 C。**本文 §7 / §8 及后续任何训练方案修订均受其约束** |
| 关联实测 | `eval_corpus_v2_1105.md`、`eval_by_algorithm_1104.md`、`6_mrz_ocr/docs/CNN-OPTIMIZATION.md`、`6_mrz_ocr/docs/deep-beam.md`、`band_candidates_product_port.md`（2026-11-06 落地验证：候选自选预算闸门与 Exact/P95 同源归因）、`c_prereq_recalibration_1107.md`（2026-11-07：C 前置重标定立项与五要素） |
| 文档定位 | 本文整合灰度训练主张、补充建议与代码评审结论。凡标注「评审修订」的条款，均为基于本仓库实测数据对原主张的修正，**优先级高于原表述** |

---

## 0. 评审结论摘要

**方向判定**：以灰度端到端训练替代二值渲染 + 推理端硬阈值，在成像物理与域偏移机理上成立，应作为训练语料的演进方向。但本次评审得出**六点**必须前置的修正（1–4 为 v1.1 的图像层与工程红线，5–6 为 v1.2 的训练目标与流程约束），直接决定改造的收益与风险：

1. **必须区分三个图像层次，否则会得出相反结论**（v1.1 修订，见 §2.1 实测）：
   - **A 语料图像**（GUI「MRZ OCR测试」展示的 1380 张 PPM，1518×126）：**准二值**。前景恒为 0、背景恒为 255，仅笔画边缘有 1px 人工灰阶；`img_0001_b0_v0` 实测 84.45% 像素是纯 0/255，93.22% 落在端点 ±64 内，只有 6.78% 属真正中间调，全图仅 12 个灰度级。`aa:true` 并非字体抗锯齿，而是对 **1-bit 字模**做 `scale` 倍展开后一次 `blur3` 得到的 16 级量化边缘（`gen_mrz_image.c:119-124,150-189`）。
   - **B 真实照片 band_gray**：真正的光学灰度，来自 `tmp/PassPortPhoto` / `assets/pic`。
   - **C 分类器输入 12×16 float**：`1 - gray/255`，`include/cnn.h:14-16` 的 "INPUT IS SM00TH GRAYSCALE" 与 `CNN-OPTIMIZATION.md:53` 说的是**这一层**；其灰阶来自面积重采样 + 3×3 box 平滑，是**降采样副产物**，不是成像产生的。
   - 因此**原主张在语料层（A）基本成立**：A 层的灰阶是二值掩膜边缘覆盖率的确定性函数，不携带油墨浓淡、底纹、光照、局部对比度信息，浅层卷积只能学到边缘位置。`v1.0` 中「本仓库并非纯黑白训练」的表述混淆了 A 与 C，本版予以更正。改造对象仍是 A 层渲染源头与依赖 Otsu 的几何链路。
2. **「远优于」缺乏本项目的单变量证据**。仓库中 `aa=0 / aa=1` 同条件对照不存在；`eval_by_algorithm_1104.md:94` 的 CRNN l1 22.17%→100.00% 同时更换了字体（bitmap→OCR-B）、语料规模与训练策略，属多变量混合收益，不能归因于灰度单一因素。必须按 §6.4 补做单变量 A/B。
3. **改造优先级需让位于已测得的更高收益项**。`eval_corpus_v2_1105.md:178-180` 与 `eval_by_algorithm_1104.md:98` 一致指出剩余瓶颈在 **band 定位层而非字符分类层**：`assets/pic` 上 restore pair + pad 8..25 得 74.17%，改用 `mrz_band` 候选自选可达 97.15%（+23pp）。字体渲染改造的边际收益预计显著低于此。建议先完成 band 候选自选，再推进本大纲。
4. **改动会立即使已部署权重失效**。`6_mrz_ocr/src/mrz_ocr_cnn.c:3261` 注释："the CNN weights were trained on Otsu binaries"；`:869-873` 明确「窗口宽度、训练权重与 ink gate 是同一套标定，分阶段上线严格劣于不上线」；`docs/CNN-OPTIMIZATION.md:265-268` 记录局部归一化重训后 full-match 45.0%→37.9% 的实测退化。**任何预处理/渲染变更必须整体重训 + 全量批测 + TD3 零退化验证，禁止灰度权重与旧切分链路混用。**
5. **「Exact 率卡在 95%」缺少口径，且「约束解码可拉到 99%+」被本仓库实测驳回**（v1.2 新增，见 §7.2 / §8.3）。`AGENTS.md` 的硬门槛是**逐字符** passport ≥98% / pic ≥95%，并无「Exact 95%」这一指标；现有报告里 `line exact`、`both exact`、`full match`、`cksum valid` 四种口径并存且分母不同。约束解码（`fill_allow` syntax mask + ICAO 校验位 DFS）**已上线**，`docs/deep-beam.md:91-98` 的 1380 语料对照显示 beam off → depth4 只把 full match 从 59.57% 提到 **64.42%**（+4.85pp），而 cksum valid 从 72.68% 提到 **96.09%**（+23.4pp）——收益主要在校验通过率，不是 Exact；`:82-85` 更明确记录剩余失败是分类器把真值 cell 概率给到 0.000，**真值不在候选集时任何约束搜索都无法恢复**。因此 Exact 的上界由定位与分类环节决定。
6. **通用训练经验必须先过架构过滤**（v1.2 新增，见 §7 / 附录 C）。本引擎是固定 44/30 网格 + 37 类逐字符 Cross-Entropy，无 CTC blank、无序列长度漂移、无 decoder，故 CTC Loss / Sequence Attention Loss / Greedy→Beam Search / LM / 冻结浅层微调等常见建议多数**不适用或已实现**。`AGENTS.md` 已规定所有模型训练与识别任务必须先加载 `ai-research-skills-mrz`，用其 §3 架构事实过滤后再给方案。

**流程前置（v1.2）**：本文 §7、§8、附录 C 及后续所有训练/识别方案，均须按 `AGENTS.md`「模型训练与识别 Skill 强制使用」条款，先加载入口 skill、登记上游方法论路由，并按其 §7 输出五要素（skill 适用性、架构兼容判定、A/B 设计与门禁、翻车清单核对、红线影响）后方可提交。

---

## 1. 目标与适用范围

- **目标**：使训练语料的像素分布逼近真实摄像头成像（连续亚像素灰度过渡、底纹、光照、模糊、噪声、几何失真），从而降低训练—推理域偏移，提升离焦/反光/底纹场景下的字符与序列准确率。
- **范围**：仅约束训练样本合成与训推输入契约。不改动网络结构（`cnn.h` 定义的 12×16 / conv1 8 / conv2 16 / fc 64 / 37 类）、不改动 MRZ 解码与校验位逻辑、不改动 TD3 / CS-CD 布局判定。
- **不在范围**：STN 网络结构改造（见 §4.4）、推理端 C 代码重写、端侧新增依赖库。
- **强制方法论前置（v1.2）**：`AGENTS.md` 规定的 `ai-research-skills-mrz` 是本大纲所有训练/识别条文的生效前提；上游 AI-Research-SKILLs 库按附录 C 路由，其 LLM/框架特定配方（CTC、LoRA、分布式训练、GPU serving）在本项目一律标注「不适用」，只复用其研究编排、实验追踪、评估严谨性与误差审查方法论。
- **v1.2 扩展范围**：新增训练目标与 Loss 口径（§7）、Exact 率瓶颈排查与突破（§8）、字段级指标补全（§7.3）与强制研究流程（附录 C）；这些属于规范与流程改造，不改网络结构、不改 C 侧解码逻辑。

---

## 2. 现状核验（代码实测，2026-11-05）

| # | 原主张 | 代码事实 | 证据位置 | 判定 |
|---|---|---|---|---|
| 1 | 纯黑白训练 | 语料源头为二值位图（`bg=255`、`put_ink(...,0)`），`scale` 倍 bit 展开；1380 张实测为准二值，详见 §2.1 | `6_mrz_ocr/tools/gen_mrz_image.c:251,159`；`tools/train_cnn_v2.py:151-158` | **成立（准二值）** |
| 2 | 分类输入为二值 | 12×16 灰度软边缘 float，`1.0f - ink/255.0f`，逐像素 `in_mean/in_std`（当前全 0/1 恒等） | `src/mrz_ocr_cnn.c:900,936-937`；`src/cnn.c:108-113`；`include/cnn.h:14-16` | **不成立** |
| 3 | 抗锯齿 | `--aa` / `aa:true` 传给 `gen_mrz_image` 的 `aa=1`，实现为「1-bit 字模 → `scale` 倍展开 → 一次 `blur3` → 16 级覆盖率 fringe 混合」，**非字体级 TrueType 抗锯齿**；`gen_ocrb_glyphs.py:57,85` 用 `INK_THRESHOLD=128` 硬阈值生成 1-bit 字模 | `tools/gen_mrz_image.c:113-124,150-189`；`tests/gen_corpus.py:290-293,309-311`；`tools/gen_ocrb_glyphs.py:57,85` | 名实不符 |
| 4 | 前景 20–110 / 背景 180–245 | 主链路固定 0 / 255；仅 CS/CD 脚本随机化 `ink_val 0.05–0.20`、`bg_val 0.80–0.95`（≈13–51 / 204–242） | `tools/gen_cscd_synthetic.py:146-151` | **不符**，但已有可复用原型 |
| 5 | 真实护照底纹融合 | 未找到真实切片读取；仅 `sin/cos` 程序化 guilloche，融合为非标准加权式 | `tools/gen_cscd_synthetic.py:78-88,122` | **不存在** |
| 6 | 物理退化顺序 | 现状：光照渐变 → 二值字模 → blur3 → **最近邻旋转** → 椒盐 → RGB；缺运动模糊、传感器暗电流、下采样、JPEG；且旋转在模糊之后、用最近邻，会重新引入锯齿 | `tools/gen_mrz_image.c:252-298` | 顺序与完整性均需修正 |
| 7 | 彻底剔除 Otsu | 全局 Otsu(:3056)、band 局部二次 Otsu(:3200)、Sauvola 布局(:3271)、ink gate 用 `band_bin`(:3712) 仍为必经 | `src/mrz_ocr_cnn.c`；`docs/CNN-OPTIMIZATION.md:26,28` | **不成立** |
| 8 | 训推统一 8-bit 单通道 + 均值方差归一化 | 训练与推理均为 float 0..1（`/255.0`）；归一化是 192 个逐像素参数且当前恒等；PPM 为单通道复制成 RGB | `tools/train_cnn.py:188-190`；`src/cnn.c:108-113` | **不成立** |
| 9 | 轻量级 CLAHE | 生产 CNN 链路未找到 CLAHE；唯一实现是 `eval_all_engines.py:735` 给 Tesseract 用的 `clipLimit=3.0, tileGridSize=(8,2)`；`mrz_normalize_local_contrast()` 默认 OFF | `6_mrz_ocr/tools/eval_all_engines.py:735`；`src/mrz_geom.c:280-297`；`src/mrz_ocr_cnn.c:3311` | **不存在** |
| 10 | 灰度 vs 二值 A/B | 未找到 `aa=0/aa=1` 同条件并存对照代码或结果 | 全仓 | **缺失** |
| 11 | real-frac 0.70 | 口径不一致：`train_cnn_mixed.py:31`=0.30、`tests/mrz_train_loop.py:86`=0.50、`1104` 报告 pic=0.70、`1105` 重训用 0.5 | 见附录 A | 需统一 |
| 12 | 应用 CTC Loss / Sequence Attention Loss（v1.2） | 输出为每格 37 维 softmax，恒 44×2 / 30×1，无 blank、无长度漂移；训练侧全部是逐字符 CE | `include/cnn.h:29`；`src/template.h:20`；`tools/train_cnn.py:360`；`tools/train_crnn_seq.py:161,170-171` | **不适用**（见 §7.1） |
| 13 | 引入约束解码 / Beam Search 提升 Exact（v1.2） | 已上线：ICAO syntax mask + line2 校验位约束 DFS（`MRZ_BEAM_COLS 14` / `MRZ_BEAM_DEPTH 4`），并有 A/B 数据 | `mrz_ocr_cnn.c:372,1675-1676,2022`；`docs/deep-beam.md:63-67,91-98` | **已实现，非待办**（见 §8.3） |
| 14 | 「Exact 率 95%」为现有门槛（v1.2） | `AGENTS.md` 只有逐字符 98% / 95%；工程有 line exact / both exact / full-match / cksum 四种口径，字段级准确率未实现 | `AGENTS.md:13-18`；`tools/eval_weight_sets.py:147-188`；`tests/bench.py:92-129` | **口径缺失**（见 §7.2 / §7.3） |
| 15 | 字段加权 Loss 待新增（v1.2） | 已实现 `--pn-weight`（line2 cols 28..41，默认 1.0 关闭），注释内含实测残差 14.2% vs 5.6% | `tools/train_cnn_omtest.py:69-77,186-189,545` | **已实现，默认关闭**（见 §7.1） |

### 2.1 GUI「MRZ OCR测试」1380 样本集像素实测（v1.1 新增）

来源链路：`tests/gui.py:2874,4800-4801` → `tests/ocr_bench_runner.py:60` 的 `DATA = 6_mrz_ocr/data/corpus_eval`。该目录 `corpus.json` 为 `seed=2002, count=1380, font=ocrb, aa=true`。GUI 默认展示 `img_0001_b0_v0.ppm`（1518×126，P6，`clean` 档，`blur=0 / illum=0 / noise=0 / rot=0`）。

| 观测量 | 实测值 |
|---|---|
| `img_0001_b0_v0` 全图灰度级 | **仅 12 个**：0, 15, 47, 63, 95, 111, 143, 159, 191, 207, 239, 255（16 级量化覆盖率的子集） |
| 纯 0 / 纯 255 像素占比 | **84.45%**（ink 9.05% + paper 75.39%） |
| 距端点 ≤64 的像素占比 | **93.22%**；真正中间调 65–190 仅 **6.78%** |
| 全局对比度 | 前景恒 0、背景恒 255，**满量程 255**，无油墨浓淡与背景起伏 |
| 按 `blur` 分组 | blur=0：830 张（60.1%），中间灰阶 13.3%、12 个灰度级；blur=1：330 张；blur=2：220 张（此两组才出现连续灰阶 249/255 级） |
| 其它退化 | `illum=0` 940/1380（68.1%）、`rot=0` 775/1380（56.2%）、`profile=clean` 720/1380 |

**为什么肉眼看成黑白**：三条叠加原因。① 主体是满对比度 0/255，灰阶只占 1–2 px 的笔画边缘；② 这些边缘值集中在 15/47/63 与 191/207/239 等贴近端点的档位；③ `tests/gui.py:3374` 用 `disp.thumbnail((cw,ch))` 把 1518 px 宽的条带缩到面板宽度（约 3× 降采样），1–2 px 灰边被平均掉，视觉上直接坍缩为黑白。

**结论**：`aa:true` 提供的是「二值掩膜 → 模糊 → 边缘覆盖率」，不是 §4.1 要求的字体级抗锯齿，更不是 §4.3 的物理成像灰度。用户所见与原主张 §3.1 / §3.4 / §3.5 的批评一致；同时也说明 **`corpus_eval` 这一准二值冻结集不能充当 §6.1 的四级 Test Bench**，其上的准确率（`eval_corpus_v2_1105.md` 的 corpus_eval_v2 full 73.47%/74.93%）不度量真实反光/底纹/模糊鲁棒性。

**验收基线冲突（评审发现，2026-11-05/07 已裁决）**：`gt_review_loop.xlsx` 生成于 2026-10-29 14:23，各引擎逐字符仅 40.66%~74.24%，全部低于 98%/95%；而 `eval_corpus_v2_1105.md:169-176` 记载 passport l1 41/41（100.00%）、both 40/41 达标，pic l1 57/63、l2 52/59、both 49/59 **未达标**，速率 17.9 张/秒达标。二者口径与批次不同，不可直接比较。旧 `AGENTS.md` 曾规定生产权重 `0ab2131`，1105 发布配置为 `40951c5`；**人工签字裁决以 `gt_review_loop.xlsx` 与 `40951c5` 为唯一有效基线，`AGENTS.md` 已于 2026-11-07 同步。**

---

## 3. 灰度与黑白训练的核心差距（机理，含证据分级）

| # | 机理 | 证据等级 | 措辞修订 |
|---|---|---|---|
| 3.1 | 边缘梯度与抗模糊：真实成像为连续亚像素过渡，阶跃边缘使浅层卷积学到与物理成像不符的统计；离焦/运动模糊时域偏移放大 | 机理共识 + 本项目实测 | 「过拟合」宜表述为**协变量偏移 / 训练—推理域间隙**。本项目 `eval_corpus_v2_1105.md:44-49` 实测 blur=2 时 cnn full 从 80.4% 跌至 **12.9%**，为该机理提供了直接量化支撑；§2.1 的准二值语料是该域间隙的直接来源 |
| 3.2 | 抗底纹与覆膜反光：硬阈值在强光/深色底纹下造成笔画断裂与粘连 | 机理共识；混淆对需修正 | 「8→B、0→C、<→K」为假设，实测 top 混淆见 §4.5，`<→K` 未进前列 |
| 3.3 | 避免不可逆信息丢失：灰度端到端让浅层自主学习局部对比度特征，保留弱梯度 | 机理共识 + 本仓库设计共识 | 成立。但 C 层的灰阶是**二值掩膜重采样得出的边缘覆盖率**（§2.1），是几何量的确定性函数，不含局部对比度信息；A 层改造仍是必要条件 |
| 3.4 | 消除锯齿伪影：二值渲染在 `<`、`W`、`M`、`N` 斜线边缘产生阶梯像素 | 机理共识 + 代码证据 | 成立且本仓库有具体缺陷：`blur3` 之后的最近邻 `rotate_gray` 会重新产生锯齿，见 §2 #6 |
| 3.5 | 油墨深浅泛化：Gamma 变换 / 高斯扩散模拟墨色浓淡与晕染 | 机理共识 | 成立；`tools/train_cnn.py:213` 已有 15% 概率 `gamma_contrast` 旧实现可复用 |
| 3.6 | 训推预处理对齐 | 工程原则 | 成立，但实施受 §0 #4 的权重失效约束；且局部归一化在本项目有 45.0%→37.9% 的反例 |

**结论**：上述六条均方向正确，可作为设计依据；但「灰度模型识别率与现场鲁棒性远优于纯黑白」在**本项目范围内尚无定量证据**，须由 §6.4 的单变量 A/B 给出结论后方可写入验收结论。

---

## 4. 灰度训练样本合成规范

### 4.1 抗锯齿与动态灰度（目标参数）

| 参数 | 目标取值 | 说明 |
|---|---|---|
| 渲染 | TrueType OCR-B，**字体级抗锯齿开启**，禁用 `INK_THRESHOLD` 硬阈值 | 替换 1-bit 字模展开；`tools/ocrb_glyphs.h` 需重新生成 |
| 前景灰度 | 随机 `[20, 110]`（8-bit） | 按字符/笔画块采样，非整图单值；模拟墨色浓淡 |
| 背景灰度 | 随机 `[180, 245]` | 与前景独立采样，保证对比度下限 ≥ 70 |
| 笔画内部不均匀 | 每笔画乘性系数 `[0.85, 1.15]` + 局部脏污/断墨概率 ≤ 5% | 对应补充建议「随机笔画不均匀油墨」 |
| Gamma | `γ ∈ [0.8, 1.25]`，概率 30% | 复用 `train_cnn.py:213` |
| 边缘晕染 | 高斯扩散 `σ ∈ [0.3, 0.8] px`，概率 30% | 必须在几何变换之后（见 §4.3） |

**评审修订（实现差距）**：当前 `corpus_eval` 的 `aa:true` **不满足**本表第一行——字模来自 `INK_THRESHOLD=128` 的 1-bit 位图，灰边由 `blur3` 生成（§2.1）；前景/背景是固定 0/255，而 `gen_cscd_synthetic.py:146-151` 的 `ink_val 0.05–0.20` / `bg_val 0.80–0.95` 只在 CS/CD 专用脚本中随机化。因此 §4.1 是**待实现**目标，不是现状描述。建议将 CS/CD 脚本的随机灰度配置**上提为主链路共用配置**，并与字体级 AA 一同落地；区间差异（13–51 / 204–242 vs 20–110 / 180–245）需以 §6.4 的 A/B 裁决，不得两套并存。

### 4.2 真实护照底纹融合

- 从 `tmp/PassPortPhoto`、`assets/pic` 的**无字区域**裁切 Guilloche / 防伪底纹切片，建立带光照与膜反光分层的背景切片库；按证件类型（TD3 / CS-CD）分别建库。
- 融合方式：`out = bg * (1 - α·m) + ink * (α·m)` 的 Alpha 形式，或 `out = bg * ink / 255` 的正片叠底（Multiply）。**必须显式记录所用公式并固化**，替换 `gen_cscd_synthetic.py:122` 的自定义三项加权式。
- 背景切片需覆盖亮/暗/高光三类，且**与 §6.1 的 Reflective 子集同源但不同切片**，避免数据泄漏。
- 评审修订：程序化 sin/cos guilloche 保留为**补充**（低成本、可控频率），不替代真实切片；`passport_photo_restore/src/restore.py:31` 的 `remove_guilloche` 是推理侧去底纹，若训练引入真实底纹，须同步评估该模块是否造成训推不一致。

### 4.3 物理光学退化流水线（补充建议 1：顺序 + 量化）

**处置：采纳，并按实测弱轴重标定参数。** 顺序直接决定梯度真实性，必须严格按以下物理成像顺序执行，禁止并行或乱序：

```
S1 抗锯齿灰度字体渲染（§4.1）
S2 笔画级油墨不均匀 / 透视浓淡 / 脏污
S3 与真实底纹融合（§4.2）
S4 乘性光照场 + 暗角 + 局部高光（二维高斯表面）
S5 空间几何变换（仿射 / 透视 / 弹性形变，§4.4）  ← 评审插入
S6 光学模糊（高斯离焦 + 运动模糊）
S7 传感器噪声（加性高斯 + 暗电流固定图样 + 椒盐）
S8 下采样（双线性，filter=true）+ JPEG 压缩
```

**评审插入说明**：几何变换必须置于光学模糊之前。现有 `gen_mrz_image.c:274-276` 是「先 blur3 后最近邻 rotate」，物理顺序错误且旋转会重新引入阶梯锯齿，正是 §3.4 所述伪影的直接来源。

| 阶段 | 参数 | 目标范围 | 依据 |
|---|---|---|---|
| S4 光照 | 线性渐变幅度 `illum` | 0–80，其中 ≥40 占比 ≥ 30% | `eval_corpus_v2_1105.md:44-49`：illum≥40 时 full 9.6% |
| S4 高光 | 二维高斯：`σx/σy ∈ [0.1, 0.4]·W`、峰值增益 1.2–2.0、角度 0–360° | 叠加后 **clip 至 245–255** | 纯乘性场无法产生饱和截断，必须显式 clip |
| S5 几何 | 旋转 ±3°、剪切 ±0.03、透视四点偏移 ≤ 2% W | `gen_corpus.py` 现用 rot ±3° | 见 §4.4 幅度上限 |
| S6 高斯模糊 | kernel 3×3 / 5×5 / 7×7 随机，**绑定 σ ∈ [0.4, 1.5] px** | 仅写 kernel size 无意义：OpenCV 由 ksize 反推 σ，须显式给 σ | blur=2 → full 12.9% |
| S6 运动模糊 | 长度 3–9 px、角度 ±15°、概率 20% | 现有增强缺失项 | 原主张要求 |
| S7 噪声 | 加性高斯 σ ∈ [1, 6]/255；暗电流 FPN 幅度 ≤ 3/255；椒盐 p ∈ [0, 0.05] | `gen_corpus.py` noise 0–0.05 | 现有为纯 0/255 翻转 |
| S8 下采样 | 比例 0.6–1.0，**双线性 filter=true** | 与 `fastPair` 红线一致；用 area/双三次会引入新训推失配 | `AGENTS.md` |
| S8 JPEG | 质量因子 Q ∈ [60, 95]，概率 50%；灰度 4:0:0 无色度子采样 | 现有增强缺失项 | 原主张要求 |

**评审修订**：`eval_corpus_v2_1105.md:44-49` 的分轴实测（blur 0/1/2 → 80.4/56.7/12.9%、illum≥40 → 9.6%、rot≠0 → 42.6%、scale=3 → 50.8%）已是本项目最可靠的量化弱轴，参数区间应**优先锚定这些数值**而非通用经验值。`1105:63-66` 的「下一步」第 2 条同样要求 blur=2 / illum≥40 / rot±3 / scale=3 的退化鲁棒性，本表与之对齐。

### 4.4 空间几何退化（补充建议 4）

**处置：条件采纳。**

- **优先采用离线数据增强**：随机仿射、透视、弹性形变作为 S5 阶段注入训练语料，收益/成本比最高，不增加端侧算子。
- **STN 列为后续研究项，不纳入本期**：本引擎为纯 C 端侧实现（`mrz_ocr_cnn.c` / `cnn.c`），新增 STN 会引入算子与耗时，直接威胁 fuse ≥ 2.0 张/秒红线，且与「生产使用 `40951c5` 权重」冲突。上线前须先提交独立耗时评估。
- **弹性形变幅度上限**：`α ≤ 字符宽度的 5–10%`。MRZ 解码依赖固定 44×2 / 30×1 的等宽几何切分（`segment_line_grid`），过度形变会破坏切片定位，使训练与推理的字符网格不一致。
- **同步面**：任何几何退化必须同步验证 `mrz_ocr_split_lines`、`mrz_classify_layout`、`decode_row` 的 ink gate 与 TD3 回退；`mrz_ocr_cnn.c:869-873` 已明确窗口宽度、权重、ink gate 是同一标定，几何改动不可单独上线。

### 4.5 难例字符重度退化（补充建议 2）

**处置：修订采纳——以实测混淆集替代假设混淆集。**

`eval_corpus_v2_1105.md:55-60` 的真实照片逐格统计明确指出：

> 「用户关注的混淆对在真实照片上均非主力（O→0 = 5 格、8→B = 7 格）；语料上曾占大头的 5→2 在真实照片几乎不出现 —— **语料混淆集 ≠ 真实混淆集**。」

| 真实照片 top 混淆 | 次数 | 原主张是否覆盖 |
|---|---|---|
| `<` ↔ `C` | 14 | 否 |
| `O` → `<`（pureclf） | 13 | 否 |
| `7` → `8` | 9 | 否 |
| `M` → `N` | 8 | 部分（斜线类） |
| `8` → `B` | 7 | 是 |
| `9` → `R` | 6 | 否 |
| `O` → `0` | 5 | 是 |
| `<` → `K` / `X` | 未进 top | **需验证，不可作为既定假设** |

据此修订实施条款：

1. 难例集以**每轮批测输出的混淆矩阵动态生成**（混淆驱动 hard example mining），不硬编码固定字符表；保留 `O/0`、`I/1`、`B/8`、`<` 类作为冷启动种子。
2. 定向重度退化（高强度边缘模糊 + 局部高光遮挡）施加于上述动态难例，目标是迫使浅层提取**孔洞/骨架拓扑特征**。施加比例上限 15%，避免过拟合到合成伪影。
3. **验证集不施加同类重度退化**，否则指标虚高；难例增强只作用于训练集，另设独立「难例压力子集」单独报告。
4. 补充原主张未覆盖的一项：MRZ 含校验位与严格字段结构，**约束解码（checksum beam / 字段长度与字符类先验）比纯视觉特征更可靠**地抑制 `O/0`、`I/1` 混淆，应与视觉增强并行推进。
5. 与 `1105:63-66` 已排序的第 1 条（line2 字母类定向补样：G/D/K/M/L/R，line2 pos28-43 字母 cell 错误率 14.2%）合并执行，避免重复建集。

### 4.6 训推预处理对齐与 CLAHE 风险（补充建议 3）

**处置：风险判断采纳；缓解策略修订。**

- 风险确认：CLAHE 在平坦/暗区放大高频噪声，CTC 可能将噪点解码为 `<` 或 `.`。本仓库有更直接的负面证据：`mrz_ocr_cnn.c:3298-3304` 的 A/B 实测显示局部对比度归一化虽让字符级准确率略升（95.6→97.6），却使 **realistic full-match 45.0% → 37.9%** 且增加约 0.5 ms；`docs/CNN-OPTIMIZATION.md:265-268` 同载此结论。因此 **CLAHE 在本项目不宜作为默认开启项**，`MRZ_OCR_LOCAL_NORM` 保持默认 OFF。
- 对「训练时以 50% 概率在线触发 CLAHE」的修订：若推理端确定性启用 CLAHE，而训练端仅 50% 施加，会造成新的训推分布失配；若推理端也随机化，则端侧行为不确定，不可接受。正确拆分为两条独立措施：
  1. **确定性对齐**：一旦决定启用 CLAHE，训推两端使用**同一份实现、同一组参数**（`clipLimit ∈ [2.0, 4.0]`、`tileGridSize` 固定 8×8，灰度单通道），训练语料 100% 施加，禁止跨语言实现差异（Python/C 的 CLAHE 边界处理不同）。
  2. **概率化的是噪声增强，不是 CLAHE**：以 30–50% 概率叠加 §4.3 S7 的传感器噪声，使模型对放大后的噪声具备免疫力。
- 替代方案（副作用更小，建议优先评估）：逐图 mean/std 归一化 + 轻量局部对比度归一化，且必须先通过 §6.4 的 A/B 才可进入生产。
- **目标输入契约**：8-bit 单通道灰度 → 统一 `1 - gray/255` → 逐像素 `in_mean/in_std`（当前为 0/1 恒等，若启用标量归一化需同步改写 `cnn_weights_*.h` 中的 192 个参数并重训）。**禁止**在分类支路引入 Otsu / 自适应二值化。
- **「彻底剔除 Otsu」的边界澄清**：Otsu/Sauvola 当前承担的是定位、行切分、布局判定与 ink gate，不是分类输入。移除它们属于**几何链路重构**，影响 `split_lines` / `classify_layout` / `segment_line_grid` / `decode_row` 及全部权重，须作为独立立项，不与字体灰度改造合并上线。

---

## 5. 训推一致性契约（新增门禁）

每次训练产出权重时，随权重一同归档以下四项，并在批测前自动校验：

| 项 | 训练侧 | 推理侧 | 校验点 |
|---|---|---|---|
| 输入位深/极性 | 8-bit 灰度 → `1 - gray/255` | `resample_char_gray` 同式 | `src/mrz_ocr_cnn.c:936-937` |
| 归一化 | `in_mean/in_std` 192 参数 | 同参数数组 | `src/cnn.c:108-113`、各 `cnn_weights_*.h` |
| 降采样 | 双线性 filter=true | `fastPair` 双线性 filter=true | `8_android/app/src/main/java/com/nextransit/mrzbench/photo/PhotoMrzPipeline.kt:836,2093-2095` |
| 局部增强 | CLAHE / 局部归一化开关与参数 | 完全一致 | `MRZ_OCR_LOCAL_NORM` 默认 OFF |

`mrz_ocr_cnn.c:3214-3218` 的双线性 deskew 注释已给出同类结论："it changes the input distribution and only pays off after a distribution-matched retrain"。**四项中任一项不一致，视为该权重集不可发布。**

---

## 6. 验收与监控 KPI（补充建议 KPI）

### 6.1 四级 Test Bench

| 子集 | 定义 | 数据来源 | 用途 |
|---|---|---|---|
| Clean | 标准扫描、无模糊无反光 | 真实照片分层标注 + 合成可控档 | 基线，防止灰度改造回退干净样本 |
| Blurry | 运动/离焦模糊 | 真实照片为主；合成按 blur=2、运动核 3–9px | 验证 §4.3 S6 |
| Reflective | 强反光 / 暗角 / 深底纹 | 真实照片 + §4.2 底纹切片（不同切片） | 验证 §4.2 / §4.3 S4 |
| Distorted | 大角度 / 透视 / 页面未展平 | 真实照片 + §4.4 几何增强 | 验证 §4.4 |

**评审修订**：分层标注必须来自**真实照片**（`tmp/PassPortPhoto`、`assets/pic`），合成子集仅用于可控消融，不得作为达标依据——`1105:55-60` 已证明「语料混淆集 ≠ 真实混淆集」。§2.1 进一步给出像素级证据：`corpus_eval` 的 1380 张是前景 0 / 背景 255 的准二值合成图，60.1% 无模糊、68.1% 无光照退化，其上的准确率不度量反光/底纹/油墨浓淡鲁棒性。**`Blurry` / `Reflective` 两档不得以该语料打分。**

### 6.2 指标定义与门槛（与 AGENTS.md 合并）

| 指标 | 定义 | 门槛 |
|---|---|---|
| CRR | 逐字符正确率（Character Recognition Rate），按 GT line1/line2 逐字符比对 | passport ≥ 98%、pic ≥ 95% |
| SRR | 序列识别率：整行 exact + both 双行 exact，分引擎报告 | 双行 exact 不做回退 |
| Exact Match Rate（v1.2） | 口径需先按 §7.2 裁决（line / both / full match 三选一 + 分母定义） | **当前无门槛**，裁决后书面写入 |
| Field-Level Accuracy（v1.2） | §7.3 的 ICAO 字段切分逐字段准确率 | 报告项，P2 起必须输出 |
| cksum valid（v1.2） | 校验位通过率，**与 Exact 分开报告，不得互相替代** | 不回退（1380 语料基线 96.09%） |
| 产品路径速率 | fuse 平均 / P95 耗时换算张/秒 | ≥ 2.0 张/秒 |
| TD3 零退化 | 分类器不确定时默认 2×44；`fixed=0 / broke=0` | 硬红线 |
| CS/CD | 30 字符单行格式与 `mrz_cscd_decode` | 不回退 |
| 单元测试 | `test_postproc` 147/147、`test_mrz` 89/89 | 全通过 |
| 训推一致性 | §5 四项契约 | 全一致方可发布 |

### 6.3 混淆矩阵门禁

- 每个 Test Bench 子集单独输出 37 类混淆矩阵，重点跟踪 `<→C`、`O→<`、`7→8`、`M→N`、`8→B`、`9→R`。
- **量化通过条件（建议）**：相对二值基线，`8→B` 与 `<→C` 误判次数下降 ≥ 50%，且其余 top-5 混淆对无一上升 > 10%。
- `<→K` / `<→X` 先建立基线计数，再决定是否纳入考核；当前无证据表明其为主力混淆。
- 分层报告 line2 pos28-43 字母 cell 错误率（基线 14.2%），防止整体指标被数字类稀释。

### 6.4 灰度 vs 二值单变量 A/B（必须补做）

| 控制项 | 要求 |
|---|---|
| 唯一变量 | 渲染方式（AA 灰度 vs 1-bit 二值 + 硬阈值） |
| 固定项 | 同一网络结构、同一语料文本与规模、同一 seed、同一 epoch、同一 real-frac、同一增强参数（除渲染）、同一评估脚本与 GT |
| 评估集 | §6.1 四级 Test Bench + `corpus_eval_v2` |
| 输出 | CRR / SRR / 分轴退化曲线 / 混淆矩阵 / 速率；附显著性说明 |
| 判定 | 四级子集均无回退，且 Blurry + Reflective 子集 SRR 有统计显著提升，方可确认「灰度优于二值」 |
| 反例警戒 | 不重复已翻车方案：`--balance-real`（37/41→28/41）、`--class-weight`（40/41→32/41），见 `1105:68` |

---

## 7. 训练目标、Loss 与评估指标（v1.2 新增）

> **本章受强制流程约束**：按 `AGENTS.md` 先加载 `ai-research-skills-mrz`，用其 §3 架构事实过滤通用建议；上游 skill 路由见附录 C。凡「不适用 / 已实现」项不得写入待办。

### 7.1 目标函数：现状、适用性与目标态

| 候选项 | 本仓库代码事实 | 判定 |
|---|---|---|
| **CTC Loss** | 输出是**每格 37 维 softmax**（`include/cnn.h:29` 的 `fc2 : 64 -> 37`；`src/template.h:20` `MRZ_OCR_GLYPH_COUNT 37`），长度恒 44×2 / 30×1，**无 CTC blank、无空格类、无冗余类、无对齐问题** | **不适用**。引入 blank 会改变 37 类索引布局，连带破坏全部 `cnn_weights_*.h` 与 `decode_row` 的 argmax 语义 |
| **Sequence Attention Loss / Seq2Seq** | 无 decoder；`train_crnn_seq.py:170-171` 对 `(N,44,37)` 逐位置算 CE；`src/crnn.c` 每步输出 37 维 softmax。所谓 CRNN 分支是「CNN 前端特征 `(N,2,44,96)` → 线性投影 + 位置 embedding → BiGRU → 逐位置分类」 | **不适用** |
| **Cross-Entropy** | 已在用：`tools/train_cnn.py:360` `F.cross_entropy(logits, yb)`；`tools/train_crnn_seq.py:161,171` `nn.CrossEntropyLoss()` + `logits.reshape(-1, CLASSES)` | **已实现，且是当前唯一有效主损失** |
| **字段/位置加权 Loss** | 已实现 `--pn-weight`，对 line2 cols 28..41 加权，默认 `1.0`（关闭）；参数注释即写明「实测残差集中于该区：字母 cell 14.2% vs 数字 5.6%」，并注明 `--class-boost` 已实测为负 | **已实现，默认关闭**。只能作为 §8 的单变量 A/B 候选，不得当新增项 |
| **训练期语法约束 Loss** | 未找到实现。语法约束只在**推理期**：`fill_allow` / ICAO 9303 syntax mask / line-2 校验位 DFS | **可研究但需防重复约束**：推理期已有强先验，训练期再加同类约束须独立 A/B 验证是否过约束 |

**目标态**：保持逐字符 / 逐位置 Cross-Entropy 为唯一主损失；若启用加权，只允许以 `--pn-weight` 为**唯一变量**做 §8.5 的 A/B。**禁止**引入 CTC blank、扩充字符集或改 softmax 维度——这会与 §0 #4 的「输入分布—权重—ink gate 同一套标定」叠加成不可控的多变量改动。

### 7.2 评估指标口径：先统一，再谈 95%

| 指标 | 现有实现 | 口径风险 |
|---|---|---|
| CRR 逐字符 | `eval_weight_sets.py:147-159`：仅 `len(GT)==44` 计入，另有 `len(sel)*44` 的 strict 分母 | 与 `gt_review_loop.xlsx` 全表口径不同；§2.1 已记录两套基线冲突 |
| 行级 Exact | `eval_weight_sets.py:161-164` `exact(l1/l2)` | 分母按 tier 过滤（`l1_sel` / `l2_sel`），不是全部图片 |
| Both Exact | `eval_weight_sets.py:183-188` | 只统计两行均 tier A/B 且 GT 均 44 字符的样本 |
| full-match / cksum valid | `tests/bench.py:92-129`；`docs/deep-beam.md:91-98` | **cksum valid ≠ Exact**：同批 1380 语料上 96.09% vs 64.42%，相差 31.67pp |
| Field-Level Accuracy | 全仓未找到实现 | 国籍、证件号、姓名、出生日期、有效期、各校验位均无独立统计 |
| Exact Match Rate | 无统一命名 | 报告中 `line exact` / `both exact` / `full match` 三种含义混用 |

**「Exact 率 95%」的裁决要求（v1.2 新增）**：`AGENTS.md:13-18` 的硬门槛是**逐字符** passport ≥98% / pic ≥95%，**不存在「Exact 95%」这一门槛**，仓库也没有把 95% 记为 Exact 卡点的批次。启动 §8 之前必须书面确定四项：① Exact 指 line / both / full match 哪一个；② 分母是全量图片还是 tier A/B 子集、是否排除 band-fail 与空输出；③ 真值用 `gt_review_loop.xlsx` 还是 `tests/gt_*_official.json`；④ 以哪份权重与产品路径为基线。**未裁决前，任何「95% → 99%+」的结论不可验收、不可写进报告。**

### 7.3 指标补全（P2 交付项）

- **字段级准确率**：按本仓库 `check_digit_of` 实际使用的 0-based 边界切分，与校验位 DFS 保持同一套定义，避免另立标准：
  - line1：`[0]` 证件类型、`[2..4]` 签发国、`[5..43]` 姓名（含 `<` 分隔符）；
  - line2：`[0..8]` 证件号 + `[9]` CD、`[10..12]` 国籍、`[13..18]` 出生日期 + `[19]` CD、`[20]` 性别、`[21..26]` 有效期 + `[27]` CD、`[28..41]` 个人号/可选区 + `[42]` CD、`[43]` 复合 CD。
- **四维误差归因面板**：按「字段 × 网格列位置 × 字符类别（字母 / 数字 / `<`） × 校验位是否通过」输出错误计数，作为 §8.1 的唯一输入。
- **分层报告强制项**：延续 §6.3，单独跟踪 line2 pos28-43 字母 cell 错误率（基线 14.2%），防止整体 CRR 被数字类稀释。
- 字段级指标是**观测项**，不得直接替换 `AGENTS.md` 的逐字符门槛；须与 CRR / Exact / cksum 并列输出。

---

## 8. Exact 率瓶颈排查与突破策略（v1.2 新增）

> 前置条件：① 按 §7.2 裁决口径；② 按附录 C 加载强制 skill；③ §0 #3 的 band 定位收益（+23pp）优先于本章任何调参。

### 8.1 深度误差归因（先行，禁止直接改 Loss / lr）

1. 用现有工具链输出真实照片的错位与切片：`tools/collect_confusions.py`、`gen_confusion_tables.py`、`mine_failures.py`、`export_hard_slices.py`；以 §7.3 的四维面板拆账。
2. **必须先把错误归入三类之一**，三者混算时调 Loss / lr 不会提升 Exact：
   - **A 定位/切分失败**：`eval_by_algorithm_1104.md:98` 实测 `assets/pic` 上 restore pair + pad 8..25 得 74.17%，改用 `mrz_band` 候选自选可达 97.15%（**+23pp，当前最大单点收益**）；`deep-beam.md:102-107` 的 4 个设备端失败里有 3 个是「MRZ 行根本没定位到」。
   - **B 字符分类错误**：37 类混淆，主力为 `<→C`(14)、`O→<`(13)、`7→8`(9)、`M→N`(8)、`8→B`(7)、`9→R`(6)（§4.5）。
   - **C 解码/后处理错误**：校验位、syntax mask、trailing filler、ink gate；用 `cksum valid` 与 `false accept` 计数度量。
3. **门禁**：每类占比书面记录，占比最大的一类定为本轮唯一变量；占比未拆清前不得进入 8.2–8.4。

### 8.2 Data Curing / Hard Mining

- 以**每轮批测输出的实测混淆矩阵**动态生成难例集（§4.5 #1），不硬编码形近字表；`10_ocr_research/confusions/photo_official.json` 已证明 `O→0 = 1`、`I→1 = 4`、`S→5 = 0`、`Z→2 = 0`，假设表与真实分布差距极大。
- **已实测翻车清单（禁止重复，`eval_corpus_v2_1105.md:68`、`docs/deep-beam.md:177-189`）**：`--balance-real` 37/41→**28/41**（false accepts 2→11）、`--class-weight` 40/41→**32/41**（0→7）、`--class-boost=750` line2 exact 40/41→**39/41**、`--mix-hard-slices` 单字准确率升但 **full-match 不动**。共同点是**改变 37 类先验或过采样**，而推理期 syntax mask 已隐含强先验，重复加先验会把 false accept 推高。
- **允许尝试的替代（各自独立单变量 A/B）**：
  1. 真值清洗 / Data Curing（`gt_review_issuer_fix/README.md:16` 记录统一真值本身有 3 张错误，属零风险收益）；
  2. 按**位置**补样而非按类别补样（`--pn-weight`，不动类别分布）；
  3. 按**退化轴**补样：`blur=2` / `illum≥40` / `rot±3°` / `scale=3` 四个已测弱轴（§4.3）。
- 验证集不施加同类重度退化；另设独立「难例压力子集」单独报告（§4.5 #3）。

### 8.3 架构与解码策略：含「95% → 99%+」主张的实测驳回

| 建议 | 本仓库事实 | 判定 |
|---|---|---|
| Greedy 换标准 Beam Search | 现有的是「最低 margin 列 top-3 DFS + 校验位硬约束」，`MRZ_BEAM_COLS 14` / `MRZ_BEAM_DEPTH 4`，**不是** beam-width beam search；`deep-beam.md:63-67` 实测 depth2 37/41 → depth4 always-run **40/41**、FALSE 2→0 | **已实现且已量化**，不得作为待新增项 |
| 引入 LM / 语言模型重排 | MRZ 字段是随机证件号、姓名与校验位，无固定词表；端侧为纯 C 实现且受 2.0 张/秒红线约束 | **不适用**；ICAO syntax mask 已是本场景的语言先验 |
| Constrained Beam / ICAO 校验位硬约束 | `fill_allow` 位置白名单 + `check_digit_of` 7/3/1 + line-2 校验位 DFS 已上线；校验位权重见 `1_mrz_decode/src/mrz.c:36-45` | **已实现** |
| **「约束解码可把 Exact 从 95% 拉到 99%+」** | 1380 语料 A/B（`deep-beam.md:91-98`）：beam off → depth4，line1 94.16% 不变、line2 96.01%→96.32%、**full match 59.57%→64.42%（+4.85pp）**、**cksum valid 72.68%→96.09%（+23.4pp）**，代价 +0.9 ms/行。真实照片 37/41→40/41 | **驳回**。收益主体是校验通过率而非 Exact；且 `deep-beam.md:82-85` 明确剩余失败是分类器给真值 cell 概率 **0.000**，真值根本不在候选集内，搜索无从恢复 |
| 加深搜索到 depth 5/6 | `deep-beam.md:137-148` 的 Python replica 显示更深会引入歧义且增加耗时 | 不作首选；如需验证必须独立 A/B + P95 与 false-accept 双门禁 |

**结论**：Exact 的上界由「**真值字符是否进入每格 top-k 候选**」决定。突破点应放在 §8.1 归因出的定位/分类环节，而不是换 decoder。解码侧仅余两项可研究增量：① 训练期语法约束（§7.1，须防与推理期重复约束）；② 扩大 DFS 候选列数/深度（须同时报告 P95 与 false accept）。二者都必须走 §8.5 的单变量流程。

### 8.4 学习率与冻结浅层

- **现状**：Adam / AdamW + **CosineAnnealingLR 已实现**，`tools/train_cnn.py:304,347-348`（lr 0.02）、`train_cnn_v2.py:675,717-718`（lr 0.02 / 60ep）、`train_crnn_seq.py:159-160`（AdamW + Cosine）、`train_from_slices.py:149-150`。lr 实际量级为 **0.02 / 0.01 / 3e-3**。
- **「把 lr 从 1e-4 降到 1e-5」不适用**：本项目 lr 高两个量级，且余弦退火已覆盖衰减；照搬会直接欠拟合。若确需调 lr，可用 `train_from_slices.py` 的 `--lr` 做单变量扫描，并固定 epoch 与 seed。
- **「冻结浅层微调」判为高风险低收益**：网络极小（12×16 输入、conv1 8 / conv2 16 / fc 64），浅层参数占比很低；且 §0 #4 已证明输入分布变更必须整体重训。冻结只可能用于「分布完全不变 + 仅补少数类」的场景，与本章的灰度语料改造互斥。若坚持验证，须作为独立 A/B，并核对 `in_mean/in_std` 192 参数是否随权重同步导出（§5）。
- **已翻车超参**：`wd 0.3` / `40ep` 得 both 31/41、34/41，低于 60ep 合并的 37/41（`eval_corpus_v2_1105.md:75-78`）；字段级 voting 仲裁已判定 "NOT safe to adopt"（`tests/mrz_arbitration.py:39-49`）。

### 8.5 迭代顺序与门禁（v1.2）

| 步 | 动作 | 出口 |
|---|---|---|
| 0 | 按附录 C 加载 `ai-research-skills-mrz`，登记上游方法论路由；按 §7.2 书面裁决 Exact 口径与唯一基线 | 口径与基线签字确认 |
| 1 | 先做 band 候选自选（§0 #3，+23pp 最大单点） | pic 改善且 passport / TD3 不回退 |
| 2 | §7.3 面板 + §8.1 三类归因 | 主因类别占比书面记录 |
| 3 | 单变量 A/B：每轮只改一个变量，固定 seed / 语料 / real-frac / epoch / GT / 评估脚本；对照 §8.2 与 skill §5 翻车清单 | 有数值、有显著性说明、含翻车结论 |
| 4 | 门禁全验：CRR ≥98% / ≥95%、Exact 与 cksum valid 均不回退、false accept 不升、≥2.0 张/秒、TD3 零退化（`fixed=0/broke=0`）、CS/CD 不回退、`test_postproc` 147/147 + `test_mrz` 89/89、§5 四项契约一致 | 全绿方可发布权重 |
| 5 | 归档：A/B 配置、数值与结论（含翻车）写入 `10_ocr_research/` 报告与当日 `.workbuddy-ai/memory/YYYY-MM-DD.md`，并回写本文「评审修订」条款 | 按 skill §7 五要素交付 |

---

## 9. 风险与红线冲突清单

| 风险 | 证据 | 缓解 |
|---|---|---|
| 输入分布改变使 `0ab2131` / `40951c5` 权重失效 | `mrz_ocr_cnn.c:3261`；`CNN-OPTIMIZATION.md:265-268` | 整体重训 + 全量批测；禁止新旧混用 |
| 「窗口宽度 + 权重 + ink gate 是一套标定」，分阶段上线更差 | `mrz_ocr_cnn.c:869-873` | 灰度改造原子化发布，禁止灰度开关灰度上线 |
| 局部归一化历史退化 45.0%→37.9% | `mrz_ocr_cnn.c:3298-3304` | CLAHE 默认 OFF，先 A/B |
| 几何形变破坏 44/30 固定网格切分 | `segment_line_grid:1198` | 幅度上限 + 同步验证切分与 TD3 回退 |
| 新增算子冲击 2.0 张/秒 | `AGENTS.md` | STN 不纳入本期；任何新增预处理先测 P95 |
| `FAST_NATIVE_MAX_WIDTH` / `fastPair` 双线性 | `PhotoMrzPipeline.kt:195,836,2093-2095` | 保持 1600 与 filter=true；训练侧下采样同滤波器 |
| pic 集未达标且权重集不含 CRNN 头 | `1105:174`、`1104:96` | pic 单独重训，real-frac 按红线 0.70；先解决 band 候选自选 |
| 真值表与最新报告口径冲突、生产权重版本冲突 | §2 末 | 实施前人工裁决唯一基线 |
| 真实底纹与 `remove_guilloche` 训推不一致 | `passport_photo_restore/src/restore.py:31` | 端到端验证，必要时同步调整 |
| 「约束解码 95%→99%+」过度承诺，用 cksum valid 冒充 Exact | `docs/deep-beam.md:91-98`（96.09% vs 64.42%） | §7.2 口径裁决 + §6.2 两指标分列；不得以校验通过率宣称 Exact 达标 |
| Exact 口径未统一、字段级准确率缺失，导致 95% 无法复现 | `eval_weight_sets.py:147-188` | P2 先交付 §7.2 / §7.3 |
| 绕过强制 skill，把 CTC / LoRA / 分布式 / serving 配方套到本引擎 | 上游库以 LLM 基础设施为中心，无 OCR skill | 附录 C 门禁；方案必须含架构兼容判定 |
| 类别重采样 / 重加权反复翻车并推高 false accept | `1105:68`；`deep-beam.md:177-189` | §8.2 列为禁试清单；改用位置/退化轴补样与真值清洗 |
| 冻结浅层或降 lr 与整体重训要求冲突 | `mrz_ocr_cnn.c:869-873,3261`；§8.4 | 不纳入本期；如验证须独立 A/B |

---

## 10. 分阶段实施计划

| 阶段 | 内容 | 出口条件 |
|---|---|---|
| P0 基线与流程裁决 | 确认唯一真值口径与生产权重版本；冻结四级 Test Bench 标注；按附录 C 加载 skill、登记实验追踪；书面裁决 §7.2 的 Exact 口径 | 基线数字与口径书面确认 |
| P1 前置收益项 | band 候选自选（`mrz_band`）落地；line2 字母类补样 | pic 指标改善且不回退 passport / TD3 |
| P2 指标补全 | 交付 §7.2 口径定义、§7.3 字段级准确率与四维归因面板；Exact / cksum valid 分列 | 归因面板可复现，口径单一 |
| P3 合成器改造 | 按 §4.1–§4.3 实现 AA 灰度、真实底纹、物理顺序与参数；复用 `gen_cscd_synthetic.py` 随机灰度原型 | 合成样本人工目视 + 灰度直方图分布检查通过 |
| P4 单变量 A/B | 按 §6.4 完成灰度/二值对照；按 §8.5 启动 Exact 瓶颈单变量迭代 | 四级 Test Bench 无回退 + 退化子集显著提升 |
| P5 难例、几何与瓶颈突破 | §4.4 离线几何增强、§4.5 混淆驱动难例；§8.2–§8.4 的候选变量逐个 A/B | 混淆矩阵门禁与 §8.5 门禁全绿 |
| P6 整体重训与验收 | passport / pic 权重重训；全量批测生成 `gt_review_loop.xlsx` | 98% / 95%、≥2.0 张/秒、TD3 零退化、CS/CD 不回退、单测全过、§5 契约一致 |
| P7 可选研究 | 分类链路去二值化、CLAHE、STN、训练期语法约束 Loss、DFS 深度 | 各自独立 A/B + 耗时评估，不与 P3–P6 合并 |

> **流程门禁（v1.2）**：P0–P7 每个阶段启动前须确认 `ai-research-skills-mrz` 已加载并产出附录 C 要求的五要素；未产出视为阶段未启动。

---

## 11. 未决事项

1. ~~`gt_review_loop.xlsx`（2026-10-29，40~74%）与 `eval_corpus_v2_1105.md`（passport 100%、pic 未达标）何者为验收口径？~~ **已裁决（2026-11-05，见 #11）**：唯一真值 `gt_review_loop.xlsx` 的 `GT对比` sheet；10-29 批次读数被取代。
2. ~~生产权重基线是 `AGENTS.md` 规定的 `0ab2131`，还是 1105 报告发布的 `40951c5`？~~ **已裁决并同步（2026-11-05 / 2026-11-07）**：生产权重基准 = `40951c5`；`AGENTS.md` TD3 红线已于 2026-11-07 改为 `40951c5`，旧 `0ab2131` 不再作为现行基线。
3. `real-frac` 以 `AGENTS.md` 的 pic 0.70 为准，还是 1105 重训实际使用的 0.5？主链路 0.30 是否需要废弃？
4. 前景/背景灰度区间采用原主张 `[20,110]/[180,245]`，还是 CS/CD 现有 `[13,51]/[204,242]`？须由 P4 的 A/B 裁决。
5. `<→K` / `<→X` 是否纳入难例与考核，取决于基线混淆矩阵计数。
6. 分类链路是否移除 Otsu/Sauvola，需独立立项，本期不做。
7. ~~「Exact 率 95%」指 line / both / full match 哪一个？分母是否含 band-fail 与空输出？真值用 `gt_review_loop.xlsx` 还是官方 GT json？~~ **已书面裁决（2026-11-07，见 `AGENTS.md`「计时、Exact 与权重口径裁决」）**：项目不存在「Exact 95%」硬门槛；内部 `exact ≥45/47` 采用 both exact（分母 47，3 张 GT 非 44/44 记 `bad3`）；line exact / full match / cksum valid 分列；唯一真值 `gt_review_loop.xlsx`；产品计时以整链 `algorithm_ms` 为准，<100 ms 看 P95，速率看 mean。（§7.2）
8. 是否引入训练期语法约束 Loss？与已上线的推理期 syntax mask + 校验位 DFS 是否构成重复约束、会否推高 false accept？（§7.1 / §8.3）
9. 字段级准确率以 ICAO 9303 标准边界还是本仓库 `check_digit_of` 的 0-based 边界为准？本节暂按后者，需人工确认。（§7.3）
10. ~~`--pn-weight` 是否从默认关闭改为启用、取何值？~~ **已裁决（2026-11-05）**：
    单变量 A/B（1.0 vs 2.0，唯一变量，同种子/同语料/同 epoch）→ **驳回，保持默认关闭**。
    实测：both exact 持平（37/47、43/61、官方 40/41 与 49/59），但**目标区
    （line2 pos28-41）字母错误率 7.16% → 9.81%**（其余 41 张 1/299 → 6/299）、
    **false accept 9 → 10**；唯一上升的是合成语料（+1.3pp），而合成语料的关联已被
    同日 v3 实验证伪。机理：推理期 syntax mask + 校验位 DFS 已隐含强先验，训练期再加权
    等于与解码先验对抗。与 `--balance-real` / `--class-weight` / `--class-boost` /
    `--mix-hard-slices` 同属一类净负干预。详见
    `10_ocr_research/ab_protocol_pn_weight_1105.md`。
11. **基线裁决（2026-11-05，人工签字）**：真值 = `gt_review_loop.xlsx`；口径 = **逐字符**
    （AGENTS.md 98%/95%；不存在「Exact 95%」门槛）；权重基准 = **40951c5**；
    `gt_review_loop.xlsx` 10-29 批次（各引擎逐字符 40.66%~74.24%）与 `AGENTS.md` 的
    `0ab2131` 读数**被取代**。今日基线：passport l1 100.00%、both 40/41（达标）；
    pic both 49/59（**未达标**）；真实照片 50 张 PASS 37/verified 49、pic 64 张 PASS 45/verified 58；
    17.9 张/秒。剩余缺口 = pic/真实照片的 both exact，主导失败类是
    §8.1 的 **C 类「校验通过但读错」20/114**。
12. **A/B 协议必须写明 CRNN 的特征一致性步骤**：改 line2 权重后 CRNN 通道直接失效
    （实测两臂都崩到 0/47，line2 l2_acc 5.35%）—— CRNN 头绑定 line2 CNN 的 conv 特征，
    换权重必须重导特征 + 重训 CRNN，否则该通道不能当门禁。
13. （2026-11-06 追记）**渲染 A/B（§6.4）实测被前置阻塞**：物理流水线已落地
    （`mrz_phys.h` + `phys=1` + `--physical`，像素合规 12→108 灰级），但发布前置
    （band 定位+行切分+二值化）在物理语料上 exact 仅 **1/300**（OK 率 97%、垃圾 67/300、
    dump cell 整字空白）—— §0 #4「同一套标定」的实测证明。**"语料 exact ≥90%" 非训练可解**；
    训练侧杠杆全部实测完毕（先验类 5 负 + `--pn-weight` 驳回 + 语料 v3 中性 + 渲染 A/B 阻塞）。
    唯一真实杠杆 = **C 前置对真实灰度的重标定**（独立立项，整体上线）。
    **2026-11-07 已立项**：见 `10_ocr_research/c_prereq_recalibration_1107.md`；
    本轮只做只读协调与协议冻结，未改 C 源码、未启动重训。
    切片导出两代失败与最终方案见 `ab_protocol_render_1106.md`：
    几何切片违反 §5 契约（自训语料上 1.25%）；dump+位置标签（`export_corpus_slices_pos.py`）
    为正解。1380 语料合规 review：14 条款 8 不符 3 部分 0 达标
    （`corpus_compliance_review_1106.md`）。
14. **§7.3 字段级面板已落地**：`6_mrz_ocr/tools/field_accuracy.py`（用本仓库
    `check_digit_of` 的 0-based 边界，与 C 端校验位 DFS 同一套定义）。
    114 张真实照片实测（cnn，逐格错 2.73%）：个人号/可选区 4.61%（最差多数字段区间）、
    个人号校验位 11.21%、签发国 6.36%。**但该区 37 个字母错有 30 个来自 Photo/ 的
    6 张机位照（同一证件，该区错 36.90%；其余 41 张仅 1.74%）** —— 任何以该区为靶心的
    干预都必须按证件分开报告，否则等于对一份证件做记忆。
15. **评审修订（2026-11-06）：§0 结论 3 的 band 层优先级，须与「预算闸门」捆绑**。
    桌面 GUI 产品路径把 `mrz_band` 候选自选全量上线（无闸门）后，正式 50 张
    bench 实测 official 整链 `mean/median/p95 = 197.8/179.8/377.3 ms`、
    **0/50 满足 <100 ms**，而 fuse 引擎本身仅 ~15 ms，候选搜索独占 185~237 ms；
    三引擎 raw GT 在开关候选搜索前后**完全相同**（cnn 37/47、crnn 28/47、
    fuse 37/47）⇒ 候选选择只吃时间、不改变识别结果。补上两阶段一致性早停
    （`tests/gui.py` 的 `MRZ_BAND_PICK_GATE=1`，单变量 A/B）后
    `99.3/64.8/215.8 ms`、<100 ms 升到 35/50，准确率零退化、fallback=0、
    `test_postproc` 147/147、`test_mrz` 89/89。
    **但 P95 仍未达标**：剩余 15 张 ≥100 ms 里 9 张是 FAIL，10 张 FAIL 里仅
    1 张走快路径；`Photo/3..8` 六张全部 FAIL 且 129.5~445.8 ms，正是 §8.1 的
    宽 band / 暗区黑边行分割错位样本。⇒ **耗时缺口与 Exact 缺口同源，治本在
    `6_mrz_ocr/src/mrz_ocr.c:273` 的 `mrz_ocr_split_lines` 与
    `tests/gui.py:3723` 的 `_rst_native_mrz_band` 垂直裁剪交互，
    而非继续压缩候选预算**（压缩只会把失败藏起来）。早前把 splitter 记在
    `mrz_geom.c` 属路径误记，2026-11-07 已纠正。
    另记负结果：`MRZ_BAND_PICK_PAR=8` 使 P95 215.8→270.6 ms 并出现 528 ms 离群，
    禁止以增大并发度当作无损优化。详见
    `10_ocr_research/band_candidates_product_port.md`「2026-11-06 落地验证」与
    `.workbuddy-ai/memory/2026-11-06.md` §7 补记（原 §24/§25 曾被覆盖）。
    **对本文排期的影响**：P0 的 band 定位改造仍优先于 P4 灰度 A/B，但必须自带
    预算闸门；本文 §6.2 的 fuse ≥2.0 张/秒红线在「整链墙钟」口径下尚未达成
    （单引擎口径 ~10-15 ms 已达）。计时/Exact 口径已于 2026-11-07 书面裁决，
    见 `AGENTS.md`「计时、Exact 与权重口径裁决」及本文 §11 第 7 条。
16. **立项与交接（2026-11-07，用户批准 1+2+3 之 ①）**：C 前置重标定已独立立项，
    见 `10_ocr_research/c_prereq_recalibration_1107.md`。只读协调确认
    `6_mrz_ocr/src` / `include` 工作区干净；旧记录中的「并行会话在改 `mrz_ocr_cnn.c`」
    仍需实施前复核。方案按强制 skill 五要素给出上游方法论、架构兼容判定、
    单变量 A/B（前置 + 对应整体重训权重为原子臂）、翻车清单核对与红线影响评估。
    **本轮不改 C 源码、不重训、不跑新 A/B**；正式实施任务保持 pending。
    同批书面裁决（1+2+3 之 ②）已写入 `AGENTS.md`，权重红线同步为 `40951c5`。

---

## 附录 A：证据索引

| 主题 | 路径:行 |
|---|---|
| 语料二值渲染 / 光照 / blur3 / 最近邻旋转 / 椒盐 | `6_mrz_ocr/tools/gen_mrz_image.c:245-298` |
| `--aa` 与 `--font` 语义 | `6_mrz_ocr/tests/gen_corpus.py:305-315` |
| 二值 ink box 展开 | `6_mrz_ocr/tools/train_cnn_v2.py:151-158` |
| 随机灰度 / 程序化 guilloche / 融合式 | `6_mrz_ocr/tools/gen_cscd_synthetic.py:78-88,118-126,144-155` |
| 分类输入为灰度软边缘 | `6_mrz_ocr/include/cnn.h:14-16,31-32`；`6_mrz_ocr/src/mrz_ocr_cnn.c:900,936-937` |
| 逐像素归一化 | `6_mrz_ocr/src/cnn.c:108-113` |
| Otsu / Sauvola / 权重绑定 / ink gate 标定 | `6_mrz_ocr/src/mrz_ocr_cnn.c:3056,3200,3257-3261,3271,3311,3693-3698,3712,869-873` |
| 设计共识「不做硬二值化」 | `6_mrz_ocr/docs/CNN-OPTIMIZATION.md:24-38,48-58,259-268` |
| 局部归一化实测退化 | `6_mrz_ocr/src/mrz_ocr_cnn.c:3298-3304` |
| 真实照片混淆对 / 分轴退化 / 下一步排序 | `10_ocr_research/eval_corpus_v2_1105.md:44-68` |
| 1105 主树最终状态 | `10_ocr_research/eval_corpus_v2_1105.md:165-180` |
| 1104 结论与 band 收益 | `10_ocr_research/eval_by_algorithm_1104.md:88-98` |
| GUI 1380 样本集来源 | `tests/gui.py:2874,4800-4801`；`tests/ocr_bench_runner.py:60`；`6_mrz_ocr/data/corpus_eval/corpus.json`（seed 2002, ocrb, aa=true, 1380） |
| `aa=1` 的 1-bit 字模 + blur3 fringe | `6_mrz_ocr/tools/gen_mrz_image.c:113-124,150-189`；`6_mrz_ocr/tests/gen_corpus.py:290-293`；`6_mrz_ocr/tools/gen_ocrb_glyphs.py:57,85` |
| GUI 显示降采样 | `tests/gui.py:3369-3377`（`disp.thumbnail`） |
| TD3 基线 / 红线 | `6_mrz_ocr/src/mrz_ocr.c:388`；`AGENTS.md` |
| fastPair 双线性 / 1600 | `8_android/app/src/main/java/com/nextransit/mrzbench/photo/PhotoMrzPipeline.kt:195,836,2093-2095` |
| 37 类输出 / 无 blank | `6_mrz_ocr/include/cnn.h:29`；`6_mrz_ocr/src/template.h:20` |
| 逐字符 CE（CNN / CRNN） | `6_mrz_ocr/tools/train_cnn.py:360`；`tools/train_crnn_seq.py:159-161,170-171` |
| 字段位置加权 `--pn-weight` | `6_mrz_ocr/tools/train_cnn_omtest.py:69-77,186-189,506-507,545` |
| Adam / CosineAnnealingLR / 实际 lr | `tools/train_cnn.py:304,347-348`；`tools/train_cnn_v2.py:675,717-718`；`tools/train_crnn_seq.py:159-160` |
| 指标实现（exact / char / both / cksum） | `6_mrz_ocr/tools/eval_weight_sets.py:147-188`；`6_mrz_ocr/tests/bench.py:92-129` |
| 约束解码与 beam A/B 实测 | `6_mrz_ocr/docs/deep-beam.md:55-98,137-148`；`6_mrz_ocr/src/mrz_ocr_cnn.c:85-96,372,1675-1676,2022` |
| 校验位权重 7/3/1 | `1_mrz_decode/src/mrz.c:36-45` |
| 类别重采样 / 超参翻车 | `6_mrz_ocr/docs/deep-beam.md:177-189`；`10_ocr_research/eval_corpus_v2_1105.md:68,75-78`；`tests/mrz_arbitration.py:39-49` |
| band 定位 +23pp | `10_ocr_research/eval_by_algorithm_1104.md:98`；`10_ocr_research/band_candidates_product_port.md:3` |
| 强制 skill 与上游 pin | `AGENTS.md:46-73`；`.workbuddy-ai/skills/ai-research-skills-mrz/SKILL.md`；`skills/AI-Research-SKILLs/VENDOR_PIN.md` |

## 附录 B：补充建议逐条处置

| # | 建议 | 处置 | 关键修订 |
|---|---|---|---|
| 1 | 物理退化顺序与量化参数 | **采纳** | 插入 S5 几何阶段于模糊之前；σ 绑定 kernel；参数锚定 blur=2/illum≥40 实测弱轴；高光显式 clip；下采样锁定双线性 |
| 2 | OCR-B 易混淆字符对重度退化 | **修订采纳** | 实测 top 混淆为 `<→C`(14)、`O→<`(13)、`7→8`(9)、`M→N`(8)、`8→B`(7)、`9→R`(6)；`<→K/X` 待验证；改为混淆驱动动态难例 + 校验位约束解码 |
| 3 | CLAHE 噪声放大警示 | **风险采纳、策略修订** | 生产链路本无 CLAHE 且有 45.0%→37.9% 反例；50% 概率改为「CLAHE 确定性对齐 + 噪声概率化」 |
| 4 | STN / 仿射 / 弹性形变 | **条件采纳** | 本期只做离线几何增强；STN 因端侧 C 实现与 2.0 张/秒红线列为 P7；弹性形变 α ≤ 字符宽 5–10% |
| KPI | 四级 Test Bench / CRR / SRR / 混淆矩阵 | **采纳并扩展** | 分层须基于真实照片；加入量化混淆门禁、TD3/CS-CD/速率/单测/训推一致性契约 |
| 5（v1.2） | CTC Loss / Sequence Attention Loss / Cross-Entropy | **部分驳回** | CTC 与 Attention Loss 因固定 44/30 网格 + 37 类逐格分类而不适用；CE 已在使用；字段加权 Loss (`--pn-weight`) 已实现但默认关闭；仅训练期语法约束 Loss 可作研究项（§7.1） |
| 6（v1.2） | Exact Match Rate 95% 目标、Field-Level Accuracy | **条件采纳** | `AGENTS.md` 无 Exact 门槛，须先按 §7.2 裁决四种口径；字段级准确率按本仓库 `check_digit_of` 边界补全（§7.3） |
| 7（v1.2） | Exact 卡在 95% 的四维排查（归因 / 数据 / 架构解码 / lr 冻结） | **修订采纳** | 归因先行且 band 定位优先；类别重采样列入禁试清单；Beam/约束解码已上线并有 A/B；lr 实际 0.02 而非 1e-4，冻结浅层与整体重训互斥（§8） |
| 8（v1.2） | 「约束解码把 Exact 95% 拉到 99%+」 | **驳回** | `deep-beam.md:91-98` 实测 full match 仅 +4.85pp（59.57→64.42%），cksum valid +23.4pp；`:82-85` 证明真值不在候选集时搜索无法恢复（§8.3） |
| 9（v1.2） | 安装 AI-Research-SKILLs 并在项目规则中强制使用 | **采纳并落地** | `skills/AI-Research-SKILLs/`（pin `773a529` / v1.7.2，98 skill）+ 入口 `.workbuddy-ai/skills/ai-research-skills-mrz/SKILL.md` + `AGENTS.md` 强制条款 + 本文附录 C |

---

## 附录 C：AI-Research-SKILLs 强制使用与路由（v1.2 新增）

### C.1 规则与产物

| 项 | 内容 |
|---|---|
| 规则来源 | `AGENTS.md`「模型训练与识别 Skill 强制使用」：本工程内**所有模型训练与识别相关任务**必须先加载项目级 skill 并按其工作流执行 |
| 入口 skill（项目级） | `.workbuddy-ai/skills/ai-research-skills-mrz/SKILL.md` |
| Vendored 上游库（只读） | `skills/AI-Research-SKILLs/`：上游 `https://github.com/Orchestra-Research/AI-Research-SKILLs.git`，pin `773a52944ba4747a18bd4ae9ade53fff041adcbc` / v1.7.2，98 个 `SKILL.md` / 23 分类，MIT |
| 版本与裁剪说明 | `skills/AI-Research-SKILLs/VENDOR_PIN.md`；升级须按该流程并重新验收，**不得直接修改 vendored 目录内文件** |
| 触发范围 | 训练/重训 CNN 与 CRNN 权重、Loss 与 lr 调参、识别链路改造、评估指标与批测、误差归因、难例挖掘、合成语料分布调整、训练大纲与评审文档修订 |

上游库按 `<category>/<skill>/SKILL.md` 两级组织，与扁平 skill 发现路径不同，**不要直接把分类目录当 skill 加载**；按 C.2 相对路径显式读取对应 `SKILL.md` 及其 `references/`。

**适用性警示**：该库以 LLM / 大模型基础设施为中心（Megatron、vLLM、TRL、PEFT 等），**没有任何 OCR / 小 CNN / MRZ 专用 skill**。本项目只复用其**方法论层**（研究编排、实验追踪、评估严谨性、误差审查），框架特定配方（CTC、LoRA、分布式训练、GPU serving）须标注「不适用」或仅供未来技术选型参考——这正是 §7.1 / §8.3 判定多条通用建议不适用的原因。

### C.2 上游方法论路由（按需加载，禁止全库加载）

| 本大纲任务 | 上游 skill 路径（相对 `skills/AI-Research-SKILLs/`） |
|---|---|
| §8.5 多假设训练实验编排与迭代循环 | `0-autoresearch-skill/SKILL.md` |
| §6 / §7.3 实验记录、指标曲线、超参追踪 | `13-mlops/tensorboard/SKILL.md`、`13-mlops/mlflow/SKILL.md`、`13-mlops/weights-and-biases/SKILL.md` |
| §3 机理证据分级、§8.3 主张审查 | `22-agent-native-research-artifact/rigor-reviewer/SKILL.md` |
| §11 未决事项、决策与死胡同归档 | `22-agent-native-research-artifact/research-manager/SKILL.md` |
| §8.2 Data Curing / 语料去重与筛选（仅借鉴设计） | `05-data-processing/ray-data/SKILL.md`、`05-data-processing/nemo-curator/SKILL.md` |
| §9 端侧 2.0 张/秒红线下的压缩 | `19-emerging-techniques/knowledge-distillation/SKILL.md`、`19-emerging-techniques/model-pruning/SKILL.md` |
| 分轴退化曲线与报告图表 | `20-ml-paper-writing/academic-plotting/SKILL.md` |
| §7.2 评估口径严谨性（仅借鉴设计） | `11-evaluation/lm-evaluation-harness/SKILL.md` |

### C.3 交付五要素（入口 skill §7）

任何训练/识别方案与本文后续修订，必须同时给出，缺一视为不完整：

1. 适用的上游 skill 及其方法论要点（C.2 路由）；
2. 与入口 skill §3 架构事实的兼容性判定（**适用 / 不适用 / 已实现**）；
3. 单变量 A/B 设计与门禁指标（§6.4 / §8.5）；
4. 是否触及入口 skill §5 已实测翻车清单（§8.2 同步列出）；
5. 对 `AGENTS.md` 技术约束红线的影响评估（§9 / 入口 skill §6）。

### C.4 与本文章节的对应

| 大纲章节 | skill 依据 |
|---|---|
| §0 结论 5 / 6、§2 第 12–15 行 | 入口 skill §3 架构硬过滤 |
| §7 训练目标 / Loss / 指标 | 入口 skill §3 Loss 行、指标行；C.3 五要素 |
| §8 Exact 瓶颈排查 | 入口 skill §4 工作流（基线裁决 → A/B → 归因 → 门禁 → 归档）、§5 翻车清单 |
| §9 风险清单、§10 阶段门禁 | 入口 skill §6 红线 |
| §6.2 / §6.3 KPI | 入口 skill §6 与 `AGENTS.md` 验收标准 |
