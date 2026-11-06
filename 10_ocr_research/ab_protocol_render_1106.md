# A/B 协议：渲染方式（物理 §4.1/§4.3 vs 遗留二值）—— 大纲 §6.4 规定的缺失实验

> 按 `ai-research-skills-mrz` §4/§8.5、`gray_training_outline_1105.md` §6.4 建立。
> 上游方法论：`skills/AI-Research-SKILLs/0-autoresearch-skill/SKILL.md`
> （可检验假设 + 跑前锁定基线与协议 + 负面结论同样归档）。

## H2（假设）

按大纲 §4.1/§4.3 的**物理渲染**生成的语料（真实灰度 + 正确退化顺序 + 传感器噪声 +
下采样/JPEG）训练出的模型，在同一物理语料上的 **full exact** 显著高于二值渲染语料训练的模型。

**可检验预测**：arm P（physical）在 `corpus_eval_phys` 上的 full exact ≥ arm L（legacy）+10pp，
且 blur/illum/scale=3 三个已测弱轴（现状 35.4% / 32.5% / 49.2%）均不低于 arm L。

## 为什么是这一个变量（§8.5 step 3 要求的单变量）

大纲 §2 表第 10 行：「灰度 vs 二值 A/B：未找到 aa=0/aa=1 同条件并存对照代码或结果」→ **缺失**。
且今天的实测已经排除了其它训练侧杠杆：

| 已试杠杆 | 结果 |
|---|---|
| `--pn-weight=2.0`（位置加权） | **驳回**：目标区 7.16% → 9.81%，false accept 9 → 10 |
| 字母类过采样 + 退化变体（语料 v3） | 逐格 −6.6%，但产品指标 0 变化，语料 full 71.94%（≈ 基线 73.47%）|
| class-prior 类（`--balance-real` / `--class-weight` / `--class-boost` / `--mix-hard-slices`）| 已实测全部为负 |

即：**同样的训练换语料内容不动产品；换渲染方式才是 §6.4 指定的那条唯一变量。**

## 实现（本轮新增，均默认关闭，旧语料/旧基线逐位不变）

| 组件 | 文件 | 内容 |
|---|---|---|
| 8-bit coverage 字形库 | `tools/gen_ocrb_glyphs_cov.py` → `tools/ocrb_glyphs_aa.h` | 4× 渲染 + BOX 降采样 → 每像素油墨覆盖率（§4.1 字体级 AA 的前置）|
| 物理流水线 | `tools/mrz_phys.h` | S2 墨色场 / S3 底纹 / S4 光照场+高光(clip 245) / S6 高斯(显式 σ)+运动模糊 / S7 传感器噪声(高斯+暗电流+椒盐) / S8 下采样(双线性 filter=true) |
| 渲染器接入 | `tools/gen_mrz_image.c` | `phys=1` 分支；**几何变换在模糊之前**（遗留路径是先 blur3 后最近邻 rotate，顺序错）|
| 语料开关 | `tests/gen_corpus.py --physical` | 按 §4.3 的参数区间逐图确定性生成 |

**像素级合规（同条件 clean 对照，outline §2.1 的量化指标）**：

| | 灰阶级数 | 纯 0/255 占比 | 中间调占比 |
|---|---|---|---|
| 遗留（1-bit + blur3） | **12** | **84.8%** | **1.3%** |
| 物理（S1–S8） | **108** | **0.0%** | **10.0%** |
| 物理语料实测（随机退化） | 111 | 57.7% | **42.3%** |

## 固定项（两臂完全一致）

| 项 | 值 |
|---|---|
| 架构 / Loss | 固定 44/30 网格 + 37 类逐字符 CE（不加 CTC/Attention、不改 softmax 维度）|
| 文本 | 同一 seed（eval 2002 / train 1001）→ 两臂**文本完全相同**，只有渲染不同 |
| 变体表 | 同一 `VARIANTS_CLEAN` / `VARIANTS_REALISTIC` |
| 语料内容 | plausible（k_iso3 池 + 日历合法日期），两臂一致 |
| 训练脚本 / real-frac / epochs / seed / batch | `train_cnn_omtest.py --line 1`、0.5、40、1、512（两臂一致）|
| 切片导出 | 两臂都用同一导出流程（`export_corpus_slices.py`），标签来自 corpus.json |
| GT / 评估 | `gt_review_loop.xlsx`；`bench_crnn.py`（corpus）+ `photo_direct_bench.py --set`（真实照片）+ `eval_weight_sets.py`（官方集）|
| 门禁 | corpus full exact（用户目标 ≥90%）、真实照片两集、官方集、单测 147/89、速率 ≥2.0 张/秒 |

## 变量

| arm | 训练语料 | 渲染 |
|---|---|---|
| L（对照） | `corpus_train_v2`（既有，legacy 渲染） | 1-bit 字模 + blur3 + 最近邻旋转 |
| P（处理） | `corpus_train_phys`（同文本同变体） | §4.1/§4.3 物理流水线 |

## 判定规则（跑前锁定）

- **采纳**：arm P 在 `corpus_eval_phys` 上 full exact ≥ arm L 在同一语料上 +5pp（或达到
  ≥90%），且真实照片两集与官方集**均不回退**，false accept 不升，单测全过，速率 ≥2.0。
- **驳回**：arm P 在真实照片/官方集回退，或语料增益 < 5pp → 记入负面结论，
  并说明「语料物理化本身不带来产品收益」。
- 附带报告：两臂在**两套语料**（legacy / physical）上的完整矩阵（4 格），
  以区分「模型更强」与「语料更容易」。

## 归档（§8.5 step 5）

协议 / 数值 / 结论（含翻车）→ `10_ocr_research/` + 当日 `.workbuddy-ai/memory/` +
本文「评审修订」条款。
