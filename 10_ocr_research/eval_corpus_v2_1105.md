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

---

## 重训闭环（2026-11-05 凌晨，commit 40951c5）

训练: CNN line1/line2 = corpus_train_v2 切片 248k(weak 3.0%) + photo_train_passport，real-frac 0.5，40ep；
CRNN = **新 CNN 前端重导特征**（特征一致性红线: 旧前端特征配新前端 → l1 95.5%→44.6% 崩溃实测），
train = corpus_v2 + corpus_v1 特征合并 (6000) + 41 照片×20，60ep，wd 1e-4。
（中间尝试: wd 0.3 → 真实 both 31/41；40ep → 34/41；60ep+合并 → 37/41。）

### 真实照片 gate（44 张 PassPortPhoto，TD3 子集 n=41，产品路径 band）

| engine | 重训前 both | 重训后 both | l1 char |
|---|---|---|---|
| cnn | 33/41 | **36/41** | 95.95 → **97.51%** |
| crnn | 37/41 | **37/41**（零退化） | 99.89 → 99.9x% |
| fuse | 33/41 | **37/41** | 98.50 → **99.94%** |

### 语料 gate（corpus_eval_v2，1440 张）

| engine | 重训前 full | 重训后 full |
|---|---|---|
| cnn | 63.19% | **73.47%** |
| crnn | 62.36% | **74.93%** |
| fuse | 64.10% | **75.21%** |

单元测试: test_postproc 147/147、test_mrz 89/89；FAST_NATIVE_MAX_WIDTH=1600 完好。

### 环境发现（重要）

- 仓库存在第二个克隆 `/Users/zhouyong/gitlab/passport`；主树 `6_mrz_ocr/build/` 的 CMake 缓存绑定在那个克隆上
  → 主树里 `cmake --build build` 会编译另一棵树的源码。已为主树配置 `6_mrz_ocr/build_v2`（验证可用）。
- 2026-11-05 01:21 起有并行会话在改 `src/mrz_ocr_cnn.c`（amb_margin/decode_row 未完成的编辑，当前编译不过）；
  本轮全部构建/验收在隔离 worktree（6c9f8ab + 新权重）完成，未触碰这些编辑。
- 验收用 gate 二进制: `/tmp/passport_gate/6_mrz_ocr/build_gate/`（worktree 保留至重启）。

---

## v3 语料实验（2026-11-05，字母增强 + 退化变体）— 实测结论

### 做了什么

1. **字母类定向采样**（真实照片诊断驱动）：line2 个人号字段（pos28-43）的字母 cell
   错误率 14.2%（数字 5.6%），最难类 G 18.9% / D 15.6% / K 14.8% / M 14.0% →
   `gen_corpus.py` 新增 `PERS_ALNUM_POOL`，G/D/K/M/L/R 以 3× 权重采样
   （该池占 pers 字段字母的 46.2%，均匀分布为 16.7%）。
2. **退化轴增强**：`VARIANTS_DEGRADED`（8 个训练专用变体：scale3+blur2+illum40、
   blur3、illum80、rot3、heavy noise 等），仅训练语料使用，eval 不含 → eval 的
   blur/illum/rot/scale 分轴桶仍然诚实地度量迁移。

语料 `corpus_train_v3`：4000 张（1500 clean / 1500 realistic / 1000 degraded），
与 eval_v2 文本零交集。切片 339k（weak 2.2%）。

### 分类器层面：明显改善

同一脚本、同一 106 张真实照片 band（逐格 top-1，9064 cells）：

| 指标 | 40951c5 | v3 |
|---|---|---|
| 逐格错误 | 1057 (11.66%) | **987 (10.89%)** |
| 字母→'B'（退化解吸器） | 95 | **60** |
| 字母→'<' | 90 | **69** |
| M / N / O / 5 / 7 错误 | 46/55/40/28/32 | **25/36/23/11/21** |

**关键因果**：改善来自**退化变体**，不是字母过采样 —— G/D/K/M/L/R 硬类错误率
29.39% → 29.94%（基本未变），而"字母→B"这类**退化解吸器**下降了 37%。
即：真实照片上这些字母的错误是**退化驱动**（模糊/低对比），不是**类先验驱动**
（类先验再均衡对此无效）。

### 产品层面：完全中性（不发布）

全引擎仲裁（44 张 PassPortPhoto，photo_direct_bench 全方法）：

| 配置 | official | verified | cnn | crnn | fuse | tesseract | paddle |
|---|---|---|---|---|---|---|---|
| 40951c5（发布中） | PASS 32 / FAIL 9 | 38 | 36/41 | 36/41 | 37/41 | 37/41 | 37/41 |
| v3 | PASS 32 / FAIL 9 | 38 | 36/41 | 36/41 | 37/41 | 37/41 | 37/41 |

语料基准（corpus_eval_v2 full）：cnn 73.47→72.99、crnn 74.93→**77.50**、
fuse 75.21→**76.74**。

CRNN 配方扫描（真实照片 crnn both）：v3mix×20 → 34、×60 → 36、×120 → 36；
v2+v1 mix ×20 → 36。**无一回��� 40951c5 的 crnn 37**（该值在 3 引擎口径下测得，
全引擎口径下基线同为 36/41 —— 两者口径不可混用）。

### 结论与决策

- **发布权重保持 40951c5 不变**（产品指标中性 → 无理由承担替换风险）。
- v3 的生成器能力（字母增强 + 退化变体）与语料落地为工具，留在仓库供下一轮使用。
- **下一杠杆不在采样**：逐格分类已改善 6.6% 但端到端不变，说明剩余失败在
  band 定位 / 分割 / beam 层，不在字符分类层。真实照片错误位置集中在
  line1 name 字段（col 3-14，35 errors at col4）。

---

## 主树最终状态（2026-11-05 11:00，含并行会话的 band-framing + 仲裁 + p95 优化）

发布配置 = `40951c5` 权重 + `perf/exact-2026-11-05` 的产品路径代码。

| 指标 | 结果 |
|---|---|
| 真实照片 44 张（官方仲裁） | **PASS 37 / FAIL 4，verified 41/41** |
| 速度 | mean 55.9 ms、p50 51.5、p95 67.3 → **17.9 张/秒**（门槛 ≥2.0） |
| 官方 GT passport | line1 41/41（char **100.00%**）、line2 40/41、both 40/41 |
| 官方 GT pic | line1 57/63、line2 52/59、both 49/59（pic 权重未动） |
| 语料 corpus_eval_v2 full | cnn 73.47% / crnn 74.93% / fuse 73.54% |
| 单元测试 | test_postproc 147/147、test_mrz 89/89 |

并行会话的两笔提交打中了 v3 实验指出的剩余瓶颈（band 定位层，非字符分类层）：
`2cffcd3` p95 路径优化（去掉每帧的第二次 Sauvola、ink-indexed skew、单次 beam）、
`c287979` band-framing 候选（restore pair 以 1.35×/0.45× pad 重框，取首个通过
decode + shape + 五个 ICAO 校验位的候选）+ 歧义解码时的 CRNN 仲裁。

### 环境更正（2026-11-05）

先前记录的「仓库有两个克隆」是错的：`/Users/zhouyong/gitlab` 是指向
`/Users/zhouyong/Desktop/work/Decard/gitlab` 的**符号链接**，只有一棵树。
`6_mrz_ocr/build/` 的 CMake 缓存仍指向旧路径（移动前生成），因此主树里
`cmake --build build` 会报错 —— 用 `6_mrz_ocr/build_v2`（已配置并验证）。
