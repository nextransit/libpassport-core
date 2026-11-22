# Passport Layout Detection — Research Notes

> 这是配合 `docs/passport_layout/research.md` 的工作笔记。doc 是完整调研报告（206 行），
> 这里只放**当前痛点量化**和**推荐路径速览**——下次回到这个 task 时先读这里。

## TL;DR（结论速览）

当前 `tests/passport_locator.py` 在 84 张人手审核样本上**只得到 45/84 = 53.6% 三区齐全**。
最普遍的单一 bug 是 **"MRZ 只识别到第 2 行，第 1 行被 DATA 吞"**（25/84 = 29.8%）。
根因是 `_locate_mrz` 的"底部 8% 行密度 + 双行签名"启发式，在"被签字 / data 文字密度高"的横图上漏掉第 1 行。
**推荐路径**：12 种算法对比后选 **A2 (连通域+card) + B2 (DBSCAN) + B3 (OCR-B 模板相关) + D2 (SeetaFace2 兜底)** 的纯 CV 混合栈，预期 mrz 正确率从 ~15/84 提升到 ≥ 40/84，photo 从 39/84 提升到 ≥ 55/84，单图定位 ≤ 40 ms（A53）。
**不要立刻上 DL**（YOLO/LayoutLM）：84 张标注远不够训练鲁棒模型；先做经典栈 + OCR-B 模板验证。

## 现状痛点量化（84 张，0f9780b8e1e3cc21_0008 类图）

| 问题 | 频次 | 根因 |
|---|---|---|
| mrz 只找到第 2 行 | 25/84 = 29.8% | `_locate_mrz` 双行签名启发式被 data 行密度覆盖 |
| mrz 整体缺失 / 错位 | 7/84 = 8.3% | 旋转 / 裁切 / 非标准版式 |
| photo 缺失 | 26/84 = 31.0% | `_largest_skin_bbox` 肤色阈值过严 |
| data 缺失 / 被覆盖 | 15/84 = 17.9% | `_locate_data` 依赖 photo+ mrz 存在，photo 一缺就 fallback 到整图 |
| 三区齐全 | 45/84 = 53.6% | — |

横图 56 张、竖图 28 张；宽度 563~960、高度 421~1372（**版式不固定**——任何"固定模板"启发式都脆弱）。

## 推荐实现路径（详见 `docs/passport_layout/research.md` §3-§4）

**Phase 1（1-2 天）**：纯 CV 混合栈
- 文件：`tests/passport_layout/classic.py` (~250 LOC)
- 核心：连通域卡 → 肤色列找 photo → DBSCAN 找文字行 → 底部两行做 OCR-B 模板相关
- 验收：mrz 正确率 ≥ 40/84、photo 正确率 ≥ 55/84，45/84 三区齐全**不退**

**Phase 2（2-3 天）**：边缘 case
- OCR-B 模板相关作为置信度闸门
- 90° 旋转检测 + 旋转后重定位
- 部分 MRZ 恢复（仅 1 行时向上找 line-pitch 内的第 2 行）

**Phase 3（仅在 Phase 2 < 75% 时）**：DL 兜底
- 手标 200~300 张训练 YOLOv5n-320 int8
- 仅在 classical 置信度 < 0.5 时调用
- A53 p50 80-120 ms

## 已完成 / 待办

- [x] `assets/pic/annotations_review.xlsx` 解析（84 行 + 3 sheet：审核表/区域明细/缺失清单）
- [x] 现状根因定性分析（连通性、灰度投影、肤色阈值、双行启发）
- [x] 12 种算法 census（cv/区域建议/文档 DL/baseline 四组）
- [x] 推荐栈与三阶段实施计划
- [x] 风险登记册与开放问题
- [ ] Phase 1 实现 + 84 张重生成 review
- [ ] 与 `tests/passport_locator.py` 并行 A/B（不删旧，留作回退）
- [ ] 把 review 后的 locator 合并到 master（要 user sign-off）

## 关键代码指针

- `tests/passport_locator.py:300-450` — 当前 `_locate_mrz` 双行启发
- `tests/passport_locator.py:460-500` — `_largest_skin_bbox` 肤色阈值
- `tests/passport_locator.py:505-560` — `_locate_data` 依赖 photo+mrz
- `tests/passport_pipeline.py:55-80` — `locate()` 入口（downscaled, coords rescaled back）
- `3_1_seetaface2/` — D2 的现成 C 库（face detector）
- `6_mrz_ocr/src/template.c` — D1 OCR-B 模板的真相

## 用户已确认的方向

- "全面的研究" → 调研报告 + census + 推荐路径（已出）
- "当前结果差强人意" → 待 Phase 1 实施后给 user 看
- "annotations_review.xlsx 已经审核并注视过了" → 用作 ground-truth 信号源（弱监督：只有对/错标签，无 GT 框）
