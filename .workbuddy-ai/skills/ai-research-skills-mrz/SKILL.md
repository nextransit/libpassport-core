---
name: ai-research-skills-mrz
description: 在 passport 项目进行模型训练、识别链路改造、重训调参、评估指标设计、误差归因、难例挖掘或训练大纲修订时，必须首先加载本 skill。它把工程内 vendored 的 Orchestra Research AI-Research-SKILLs（98 个 skill）路由到 MRZ OCR 任务，并叠加本项目「固定 44/30 网格 + 37 类逐字符分类」架构的真实约束、验收红线与已实测翻车清单，防止把通用 LLM/CTC 训练经验误用到本项目。触发词：模型训练、训练大纲、重训、识别优化、准确率、Exact、误差归因、难例挖掘、评估指标、CRNN、CNN 权重、批测、gt_review_loop。
version: 1.0.0
author: passport project
license: MIT
tags: [MRZ OCR, Model Training, Recognition, Evaluation, Error Attribution, Research Rigor]
---

# AI Research Skills — passport 项目强制入口

## 1. 何时必须使用

`AGENTS.md` 的「模型训练与识别 Skill 强制使用」条款规定：**本工程内所有模型训练与识别相关任务必须先加载本 skill**。覆盖范围包括但不限于：

- 训练 / 重训 CNN、CRNN 权重，调整 Loss、学习率、epoch、数据配比、类别权重；
- 修改识别链路（band 定位、切分、解码、校验位、后处理、CLAHE、二值化）；
- 设计或修改评估指标、批测脚本、`gt_review_loop.xlsx` 口径；
- 误差归因、混淆矩阵分析、难例挖掘、合成语料分布调整；
- 编写或修订 `10_ocr_research/gray_training_outline_1105.md` 等训练大纲与评审文档。

加载后按 §4 工作流执行；**任何训练或识别方案在提出前，必须先用 §3 的架构事实做适用性过滤**。

## 2. 库的位置与版本

| 项 | 路径 / 值 |
|---|---|
| Vendored 库（只读） | `skills/AI-Research-SKILLs/` |
| 版本 pin 与裁剪说明 | `skills/AI-Research-SKILLs/VENDOR_PIN.md` |
| 上游 | https://github.com/Orchestra-Research/AI-Research-SKILLs.git |
| Pin 提交 | `773a529` (v1.7.2) |
| 规模 | 98 个 `SKILL.md`，23 个分类目录 |

上游库按 `<category>/<skill>/SKILL.md` 两级组织，与 WorkBuddy 的扁平 skill 发现路径不同，**不要直接把分类目录当 skill 加载**；按 §4 的相对路径显式读取对应 `SKILL.md` 及其 `references/`。

**适用性警示**：该库以 LLM / 大模型基础设施为中心（Megatron、vLLM、TRL、PEFT 等），**没有任何 OCR / 小 CNN / MRZ 专用 skill**。在本项目应复用其**方法论层**（研究编排、实验追踪、评估严谨性、压缩、误差审查），**不得**直接套用其框架特定配方（CTC、LoRA、分布式训练、GPU  serving）。框架类 skill 仅供未来技术选型参考。

## 3. 本项目架构事实（通用建议的硬过滤条件）

提出任何方案前必须先确认它与以下事实兼容；不兼容的应明确标注「不适用」而不是照搬。

| 维度 | 事实 | 证据 |
|---|---|---|
| 架构 | **固定网格切分 + 逐字符 37 类分类 + 规则后处理**，非端到端序列识别 | `6_mrz_ocr/src/mrz_ocr_cnn.c:11-24` |
| 输出 | 每格 37 维 softmax（`0-9`, `A-Z`, `<`），**无 CTC blank、无空格类、无冗余类** | `6_mrz_ocr/include/cnn.h:29`；`src/template.h:20` |
| 长度 | 输出恒 44×2（TD3）或 30×1（CS/CD），**不存在序列长度漂移** | `mrz_ocr_cnn.c:3739,3748,4194-4206` |
| Loss | 逐字符 / 逐位置 **Cross-Entropy**；无 CTC、MSE、Attention、Seq2Seq | `tools/train_cnn.py:360`；`tools/train_crnn_seq.py:161,171` |
| 字段加权 | 已实现（`--pn-weight`，line2 cols 28–41），默认关闭 | `tools/train_cnn_omtest.py:69,186-189,545` |
| 解码 | ICAO 9303 syntax mask + ink gate + 3-neighbour `<` + trailing-filler lock + **line-2 校验位约束 DFS**（`MRZ_BEAM_COLS 14 / MRZ_BEAM_DEPTH 4`），**不是**标准 beam-width Beam Search | `mrz_ocr_cnn.c:19-24,372-408,1650-1651,1825-1858,2086-2198` |
| 校验位 | 权重 7/3/1；C 侧自建 `check_digit_of`，未复用 `1_mrz_decode` | `1_mrz_decode/src/mrz.c:36-45`；`mrz_ocr_cnn.c:85-96` |
| 优化器 | Adam / AdamW + **CosineAnnealingLR** 已实现；lr 实际为 0.02 / 0.01 / 3e-3 | 各 `tools/train_*.py`；`6_mrz_ocr/docs/ARCHITECTURE.md:178` |
| 训推输入 | 12×16 灰度 float `1 - gray/255`，逐像素 `in_mean/in_std`；Otsu/Sauvola 用于定位与切分 | `mrz_ocr_cnn.c:936-937`；`src/cnn.c:108-113` |
| 指标 | 逐字符准确率 / line exact / both exact / full-match / cksum；**字段级准确率未实现** | `tools/eval_weight_sets.py:147-188`；`tests/bench.py:92-129` |

**因此以下常见建议在本项目不适用**：改用 CTC Loss 或 Sequence Attention Loss、Greedy 改 Beam Search、靠字符集扩充修 `<` 对齐、把「约束解码」当作未实现的新增项（它已上线且有 A/B 数据）、把 lr 从 1e-4 降到 1e-5（实际 lr 高两个量级）、用 CTC 空白符解释 `<` 漂移（应归因 band 定位/分割）。

## 4. 强制工作流

1. **加载本 skill**，读 `AGENTS.md` 与 `10_ocr_research/gray_training_outline_1105.md`（v1.2）的 §0 / §2 / §7 / §8 / §9 / 附录 C。其中 §7 是 Loss 与指标口径、§8 是 Exact 率瓶颈流程、§9 是红线清单、附录 C 是上游路由；章节已在 v1.2 重排，旧版「§7 = 风险清单」的引用失效。
2. **读上游路由 skill（按需）**，从下表选取，不要全库加载：

   | 任务 | 上游 skill 路径（相对 `skills/AI-Research-SKILLs/`） |
   |---|---|
   | 多假设训练实验编排、迭代循环 | `0-autoresearch-skill/SKILL.md` |
   | 实验记录、指标曲线、超参追踪 | `13-mlops/tensorboard/SKILL.md`、`13-mlops/mlflow/SKILL.md`、`13-mlops/weights-and-biases/SKILL.md` |
   | 评估严谨性、证据等级审查 | `22-agent-native-research-artifact/rigor-reviewer/SKILL.md` |
   | 会话级研究记录、决策与死胡同归档 | `22-agent-native-research-artifact/research-manager/SKILL.md` |
   | 数据清洗 / 语料去重与筛选 | `05-data-processing/ray-data/SKILL.md`、`05-data-processing/nemo-curator/SKILL.md` |
   | 模型压缩（端侧 2.0 张/秒红线） | `19-emerging-techniques/knowledge-distillation/SKILL.md`、`19-emerging-techniques/model-pruning/SKILL.md` |
   | 训练报告图表 | `20-ml-paper-writing/academic-plotting/SKILL.md` |
   | 评估框架方法论（仅借鉴设计） | `11-evaluation/lm-evaluation-harness/SKILL.md` |

3. **先裁决基线**：`gt_review_loop.xlsx`（2026-10-29，逐字符 40.66%~74.24%）与 `eval_corpus_v2_1105.md`（passport 100%、pic 未达标、17.9 张/秒）口径冲突；生产权重 `0ab2131` vs `40951c5` 冲突。未书面确认前不得宣称达标或回退。
4. **单变量 A/B**：任何训练/识别改动只允许一个变量，固定 seed、语料、real-frac、epoch、评估脚本与 GT。对照 §5 翻车清单，已失败方案不得重复。
5. **误差归因先行**：使用 `6_mrz_ocr/tools/collect_confusions.py`、`gen_confusion_tables.py`、`mine_failures.py`、`export_hard_slices.py` 输出真实照片混淆与分轴退化；**以实测混淆集为准，不用假设的形近字表**。
6. **门禁验证**：`AGENTS.md` 的构建与测试命令全跑，并核对 §6 红线；`test_postproc` 147/147、`test_mrz` 89/89 全通过。
7. **归档**：把实验配置、A/B 数值、结论（含翻车）写入 `10_ocr_research/` 报告与当日 `.workbuddy-ai/memory/YYYY-MM-DD.md`，并更新训练大纲的「评审修订」条款。

## 5. 已实测翻车清单（禁止重复）

| 方案 | 实测结果 | 来源 |
|---|---|---|
| `--balance-real` | both 37/41 → **28/41**，false accepts 2→11 | `6_mrz_ocr/docs/deep-beam.md:177-189` |
| `--class-weight` | 40/41 → **32/41**，false accepts 0→7 | 同上 |
| `--class-boost=750` | line2 exact 40/41 → 39/41，char 99.89%→99.56% | `tools/train_cnn_v2.py:536-545` |
| `--mix-hard-slices`（30% 真实难题切片） | 单字准确率升，**full-match 不动** | `6_mrz_ocr/docs/ARCHITECTURE.md:250-252` |
| 局部对比度归一化 | realistic full-match **45.0% → 37.9%**，+0.5 ms | `mrz_ocr_cnn.c:3298-3304` |
| wd 0.3 / 40ep 等超参 | both 31/41、34/41（低于 60ep 合并的 37/41） | `eval_corpus_v2_1105.md:75-78` |
| 字段级 voting 仲裁 | 判定 "NOT safe to adopt" | `tests/mrz_arbitration.py:39-49` |

**根本约束**：`mrz_ocr_cnn.c:869-873` 明确「窗口宽度、训练权重、ink gate 是同一套标定，分阶段上线严格劣于不上线」；`:3261` 注明权重训练于 Otsu binaries。**任何预处理/渲染变更必须整体重训 + 全量批测，禁止新旧混用。**

## 6. 不可回退红线（来自 `AGENTS.md`）

- passport 逐字符准确率 ≥ 98%，pic ≥ 95%；fuse 产品路径 ≥ 2.0 张/秒；
- TD3 零退化（不确定时默认 2×44），`fixed=0 / broke=0`；
- `FAST_NATIVE_MAX_WIDTH` 保持 1600；`fastPair` 降采样保持双线性 `filter=true`；
- pic L1 训练 real-frac 0.70；CS/CD 30 字符单行解码不回退；
- 唯一验收真值：`10_ocr_research/gt_review_loop.xlsx` 的 `GT line1` / `GT line2`。

## 7. 输出要求

交付训练/识别方案时必须同时给出：① 适用的上游 skill 及其方法论要点；② 与 §3 架构事实的兼容性判定（适用 / 不适用 / 已实现）；③ 单变量 A/B 设计与门禁指标；④ 是否触及 §5 翻车清单；⑤ 对 §6 红线的影响评估。缺少任一项视为方案不完整。
