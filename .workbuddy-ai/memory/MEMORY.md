# passport 项目长期约定（curated）

## 强制流程：模型训练与识别 Skill

- `AGENTS.md`「模型训练与识别 Skill 强制使用」是项目硬规则：**所有模型训练与识别相关任务**
  （重训 CNN/CRNN、Loss/lr 调参、识别链路改造、评估指标与批测、误差归因、难例挖掘、
  合成语料分布、训练大纲与评审文档修订）必须先加载入口 skill
  `.workbuddy-ai/skills/ai-research-skills-mrz/SKILL.md`。
- 上游库 vendored 在 `skills/AI-Research-SKILLs/`（pin `773a529` / v1.7.2，98 个 `SKILL.md`，
  MIT），只读；版本与升级流程见 `VENDOR_PIN.md`。该库以 LLM 基础设施为中心，**无 OCR / 小 CNN
  skill**，只借方法论层（编排、实验追踪、评估严谨性、误差审查）。
- 交付训练/识别方案必须含五要素：上游 skill 适用性、架构兼容判定、单变量 A/B 与门禁、
  翻车清单核对、红线影响评估。

## 架构事实（通用训练经验的硬过滤条件）

- 固定 44×2（TD3）/ 30×1（CS/CD）网格 + **37 类逐字符 Cross-Entropy**，非 CTC / Seq2Seq；
  无 blank、无空格类、无序列长度漂移。CTC Loss、Sequence Attention Loss、LM 重排、
  标准 Beam Search **不适用或已实现**。
- 约束解码已上线：`fill_allow` ICAO syntax mask + `check_digit_of` 7/3/1 + line-2 校验位 DFS
  （`MRZ_BEAM_COLS 14` / `MRZ_BEAM_DEPTH 4`）。字段位置加权 `--pn-weight` 已实现但默认关闭。
  Adam + CosineAnnealingLR 已实现，实际 lr 0.02 / 0.01 / 3e-3（**不是 1e-4**）。
- `AGENTS.md` 的硬门槛是**逐字符** passport ≥98% / pic ≥95%，fuse ≥2.0 张/秒，TD3 零退化；
  **没有「Exact 95%」门槛**。line exact / both exact / full match / cksum valid 四种口径分母不同，
  字段级准确率未实现；口径已于 **2026-11-07 书面裁决**，见下节。
- 「约束解码把 Exact 95%→99%+」已被 `6_mrz_ocr/docs/deep-beam.md:91-98` 驳回：
  full match 仅 59.57%→64.42%，cksum valid 72.68%→96.09%；`:82-85` 证明真值不在 top-k
  候选集时搜索无法恢复。Exact 上界由定位/分类决定。

## 口径与基线裁决（2026-11-07，`AGENTS.md`）

- 生产权重基准 **`40951c5`**（人工签字），取代旧 `0ab2131`；三个权重头整体一致，禁止新旧混用。
- 唯一验收真值 `10_ocr_research/gt_review_loop.xlsx` 的 `GT对比` sheet；`tests/*.json` 与
  `gt_review_loop.json` 均为缓存/派生，不作验收依据。
- 内部 `exact ≥45/47` = **both exact**，分母 47（50 张中 3 张 GT 非 44/44 记 `bad3`）；
  line exact / full match / cksum valid 分列，`cksum valid` 不是 Exact。
- 产品计时以 `_ocr_photo_engines` 的整链 `algorithm_ms` 为验收口径；`≥2.0 张/秒` 按 mean，
  用户硬约束「单次 <100 ms」按 P95；单引擎 ~10–15 ms 仅诊断。
- HEAD `e0fe137` 的 `pic 4.7%→42%` 已归档于
  `10_ocr_research/device_pic_42pct_evidence_1107.md`：新值 `both exact 21/50`，旧值 `3/64`，
  **不同分母/图集/配置，不能相减或宣称达标**；pic ≥95% 仍未达成。
- C 前置重标定立项见 `10_ocr_research/c_prereq_recalibration_1107.md`；真实 splitter 在
  `6_mrz_ocr/src/mrz_ocr.c:273`（不是 `mrz_geom.c`）。本轮只立项/交接，未改 C 源码、未重训。

## 图像层次（勿再混淆）

- A 语料 PPM（`6_mrz_ocr/data/corpus_eval`，GUI 1380 样本）= **准二值**：前景 0 / 背景 255，
  `aa:true` 是 1-bit 字模 → scale 展开 → `blur3` 的 16 级边缘覆盖率，非字体抗锯齿。
- B 真实照片 `band_gray` = 真光学灰度。C 12×16 分类输入 `1 - gray/255` = 降采样副产物。
- `mrz_ocr_cnn.c:869-873,3261`：窗口宽度、权重、ink gate 是同一套标定；**任何预处理/渲染
  变更必须整体重训 + 全量批测，禁止分阶段上线或新旧权重混用**。

## 已知翻车清单（禁止重复）

`--balance-real` 37/41→28/41、`--class-weight` 40/41→32/41、`--class-boost=750` 40/41→39/41、
`--mix-hard-slices` full-match 不动、局部对比度归一化 45.0%→37.9%、字段级 voting 仲裁
"NOT safe to adopt"。类别重采样/重加权会推高 false accept；改用真值清洗、按位置补样、
按退化轴（blur=2 / illum≥40 / rot±3 / scale=3）补样。

## 训练大纲

`10_ocr_research/gray_training_outline_1105.md` 是灰度训练与 Exact 瓶颈的**目标态规范**。
当前 v1.2：§0–§6 灰度合成与 KPI，§7 Loss/指标，§8 Exact 瓶颈，§9 红线，§10 实施计划 P0–P7，
§11 未决事项，附录 A 证据 / B 建议处置 / C Skill 路由。修订该文件前先加载强制 skill。
