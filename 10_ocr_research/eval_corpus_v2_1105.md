# 语料基准 v2（plausible 化）+ 真实照片混淆统计 — 2026-11-05

## 背景与根因（对 1104 语料的 review 结论）

旧语料 `data/corpus_eval`（1380 张，random-soup 文本）full exact 只有 13.55%（cnn），
review 定位为**基准设计问题而非模型问题**：

1. **issuer-fix 冲突**（`src/mrz_ocr_cnn.c:1852-1884`）：语料国籍码是随机 3 字母，
   分类器逐格全对（conf 1.000）仍被改写成"合理国家码"（MLQ→PLW）。
   `NOISSUERFIX=1` 全量重跑：l1 错误 5048→3050（−40%）。
2. **泄漏**：eval 的 b 系列 = 5 个 BASE_DOCS（在训练集内），其中 12 张与训练集字节级相同；
   seen 文本 clean full 88.3% vs unseen 12.6%。
3. **字段非法**：出生/有效期 90% 非日历日期、国籍码 ~全部不在 ISO-3166 池，
   触发 strict-ICAO/日期合法性先验（再贡献 ~12% 错误）。
4. 退化轴与真实照片错位（椒盐噪声、纯线性光照）。

## 变更

- `6_mrz_ocr/tests/gen_corpus.py`：
  - `random_pair()` plausible 化（默认）：国籍/签发国从 `k_iso3` 池采样（与工具同源，80% 相同 20% 不同）、
    日历合法日期（bd 1940-2009 / ex 2015-2039）、docnr 数字为主（20% 短号 `<` 填充）、
    pers 字段按真实分布混合（字母/数字/`<`）。`--legacy-random` 保留旧行为。
  - eval 语料不再包含 BASE_DOCS（消除 seen/unseen 泄漏）；eval 用 seed*31、train 用 seed*131，文本零交集。
- `6_mrz_ocr/tools/bench_crnn.py`：新增
  - `--dump`（per-record JSON，含 GT/参数/预测，供失败归因）
  - 先验三档 A/B（pipeline / nofix / pureclf）
  - 按 blur/illum/rot/scale/invert 分轴报告

## 新基线（data/corpus_eval_v2，1440 张 = 60 文本 × 12clean+12realistic，ocrb+aa，当前权重）

| engine | l1 | l2 | full | cksum |
|---|---|---|---|---|
| cnn | 95.18% | 90.00% | **63.19%** | 86.81% |
| crnn | 95.55% | **95.61%** | 62.36% | **95.49%** |
| fuse | 95.38% | 90.30% | **64.10%** | 87.64% |

对照旧语料：cnn full 13.55% → **63.19%**，clean 档 18.89% → **87.08%**。
（同一权重，只改了语料内容 → 差值 = 旧基准里"先验打架 + 泄漏"的水分。）

先验 A/B（cnn）：pipeline 63.19% > nofix 58.75% > pureclf 55.83% —— 与真实照片方向一致
（先验在 plausible 内容上是净收益）。**新基准终于和生产行为对齐。**

### 残余错误归因（cnn full，分轴）

| 轴 | 取值 | full | 结论 |
|---|---|---|---|
| blur | 0/1/2 | 80.4 / 56.7 / **12.9%** | 最大单项 |
| illum | 0 / ≥40 | 80.9 / **9.6%** | 与 blur1+rot 共现 |
| rot | 0 / ≠0 | 80.6 / **42.6%** | 旋转鲁棒性 |
| scale | 3/4/5/6 | **50.8** / 59.5 / 64.7 / 79.6 | scale=3 + l2 66.3% |

## 真实照片逐格 top-3 统计（106 张产品 band，`MRZ_OCR_DIAG TOPK=3`）

- 逐格 top1 错误率 4.90%；GT 不在 top-3 的占 2.57%（top-3 改写可达上限）。
- **最难 GT 类**：G 18.9%、D 15.6%、K 14.8%、7 14.4%、M 14.0%、L 13.6%、O 12.4%、R 12.1%。
- 位置集中：**line2 pos28-43（个人号字段）**，其中字母 cell 错误率 **14.2%**（数字 5.6%、`<` 6.8%）
  → line2 模型字母类饥饿（line2 训练切片里字母极少）。
- 用户关注的混淆对在真实照片上均非主力（O→0 = 5 格、8→B = 7 格）；
  语料上曾占大头的 5→2 在真实照片几乎不出现 —— **语料混淆集 ≠ 真实混淆集**。
- 真实照片 top 混淆：`<->C`(14)、7→8(9)、M→N(8)、8→B(7)、9→R(6)、O→<(pureclf 13)。

## 下一步（按收益排序）

1. **line2 字母类定向补样**：G/D/K/M/L/R 在 line2 语境（个人号字段混合串）的语料占比提高 ——
   直接对应真实照片 14.2% 的字母 cell 错误。
2. blur=2 / illum≥40 / rot±3 / scale=3 的退化鲁棒性（新基准上的真实瓶颈）。
3. 用 corpus_train_v2 切片重训 CNN/CRNN（新语料已生成，`data/corpus_train_v2`），
   以 eval_v2 + 真实照片双 gate 验收（AGENTS.md 红线不变）。
4. 之前两次实测翻车的方案不要重复：`--balance-real`（37/41→28/41）、`--class-weight`（40/41→32/41）。
