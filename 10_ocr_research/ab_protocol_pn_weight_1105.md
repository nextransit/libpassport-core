# A/B 协议：位置加权 Loss `--pn-weight`（唯一变量）

> 按 `ai-research-skills-mrz` §4 / §8.5 与 `gray_training_outline_1105.md` §8.5 建立。
> 上游方法论：`skills/AI-Research-SKILLs/0-autoresearch-skill/SKILL.md`
> （可检验假设 + 跑前锁定基线与协议 + 负面结论同样归档）。

## H1（假设）

按**位置**加权（`--pn-weight`，line2 cols 28–41 的个人号/可选区）能降低该区字母 cell 的
错误率，从而提高 **both exact**，且不产生新的 false accept。

**可检验预测**：arm B 的 line2 pos28–41 字母 cell 错误率低于 arm A；both exact 不下降。

## 为什么是这一个变量

- §8.1 三类归因（114 张真实照片，实测 2026-11-05）：正确 82 (74.5%)、
  **C 类「校验通过但读错」20 (18.2%)**、B 类「解码已拦住的分类错」5 (4.5%)、
  A 类「定位/切分无输出」3 (2.7%) → **主导失败类是 C 类**，而 §8.3 已证
  「约束解码不可把 Exact 拉到 99%+」，故突破口只能是「真值进入每格 top-k 候选」，
  即分类器侧。
- 分类侧剩余空间由 **line2 pos28–41 字母 cell 14.2% vs 数字 5.6%**（实测，
  106 张产品 band）定位，与 `--pn-weight` 的作用区间**完全重合**。
- §8.2 明确允许「按位置补样而非按类别补样」，并把它列为 class-prior 调平失败后的
  替代方案之一。

## 固定项（两臂完全一致）

| 项 | 值 |
|---|---|
| 架构 | 固定 44/30 网格 + 37 类逐字符 CE（**不改** loss 形式、不加 CTC/Attention）|
| 语料 | `data/corpus_train_v3/cnn_slices.npz`（339k 切片，含退化变体）+ 照片切片 |
| real-frac | 0.5（护照集；pic 红线 0.70 不动）|
| epochs / seed / batch | 40 / 1 / 512（与 40951c5 同）|
| 训练脚本 | `6_mrz_ocr/tools/train_cnn_omtest.py`（worktree 内 HEAD 快照）|
| 训练目标 | **仅 line2**（`--line 1`）；line1 与 CRNN 不动 |
| GT / 评估脚本 | `gt_review_loop.xlsx`；`photo_direct_bench.py --set ...`；`eval_weight_sets.py` |
| 门禁 | 语料 eval_v2、官方 passport/pic 集、真实照片两集、单测 147/89 |

## 变量

| arm | `--pn-weight` |
|---|---|
| A（对照） | 1.0（= 默认，关闭） |
| B（处理） | 2.0 |

取值理由：位置加权只改该区 loss 质量，2.0 是最小非平凡加权；`--class-boost`
（按类别加权重）已在 §5 翻车清单里（line2 exact 40/41→39/41），**不得同时开启**。

## 判定规则（跑前锁定）

- **采纳**：both exact 不下降，且至少一项（both exact ↑ 或 pos28–41 字母 cell 错误率 ↓）
  有改善，且 false accept 不升、`test_postproc`/`test_mrz` 全过、速率 ≥2.0 张/秒。
- **驳回**：both exact 下降，或 false accept 升，或无任何改善（→ 记入负面结论）。
- 任一红线被触及 → 回退到 40951c5，不发布。

## 归档（§8.5 step 5）

配置 / 数值 / 结论（含翻车）→ `10_ocr_research/` 报告 +
`.workbuddy-ai/memory/2026-11-05.md` + 本文「评审修订」条款。
