# AGENTS.md — passport 项目验收规则

## 验收标准

### 真值数据源

**`10_ocr_research/gt_review_loop.xlsx`** 是唯一验收标准文件。

- **真值**: "GT对比" sheet 的 `GT line1` + `GT line2` 列
- 真值可由人工调整（"需人工核对" sheet 列出待核对条目）
- 各引擎识别结果与 GT 逐字符对比计算准确率

### 准确率门槛

| 训练集 | 照片目录 | 目标准确率 |
|--------|----------|-----------|
| passport | tmp/PassPortPhoto | ≥ 98% |
| pic | assets/pic | ≥ 95% |

### 性能指标

"全算法验收" sheet 包含各引擎性能指标：

| 指标 | 说明 |
|------|------|
| `平均耗时(ms)` | 各引擎单次识别平均耗时 |
| `P95耗时(ms)` | 95 分位耗时 |
| `速率(张/秒)` | 吞吐速率 |

产品路径（fuse）速率门槛: ≥ 2.0 张/秒

### 工作流程

```
人工修改 gt_review_loop.xlsx (GT line1/GT line2)
    ↓
重新训练模型 (passport/pic 权重)
    ↓
优化模型参数
    ↓
运行批测生成 gt_review_loop.xlsx
    ↓
验证准确率达标 + TD3 零退化
```

## 模型训练与识别 Skill 强制使用

本工程内**所有模型训练与识别相关任务**必须先加载项目级 skill `ai-research-skills-mrz`
并按其工作流执行。覆盖范围：训练/重训 CNN 与 CRNN 权重、Loss 与学习率调参、识别链路
改造（定位/切分/解码/后处理）、评估指标与批测、误差归因、难例挖掘、合成语料分布调整、
训练大纲与评审文档修订。

- **入口 skill**: `.workbuddy-ai/skills/ai-research-skills-mrz/SKILL.md`
- **只读 vendored 库**: `skills/AI-Research-SKILLs/`
  （上游 `https://github.com/Orchestra-Research/AI-Research-SKILLs.git`，pin `773a529` / v1.7.2，
  98 个 `SKILL.md` / 23 分类，MIT；版本与裁剪说明见 `skills/AI-Research-SKILLs/VENDOR_PIN.md`）
- **触发词**: 模型训练、重训、识别优化、准确率、Exact、误差归因、难例挖掘、评估指标、
  CRNN、CNN 权重、批测、训练大纲、`gt_review_loop`

执行要求：

1. 先用入口 skill §3 的架构事实过滤通用建议——本项目是**固定 44/30 网格 + 37 类逐字符
   Cross-Entropy** 架构，非 CTC/Seq2Seq；约束解码（ICAO syntax mask + line-2 校验位 DFS）、
   字段位置加权 Loss、CosineAnnealingLR 均已实现，不得当作待新增项。Loss / 评估指标口径见
   `10_ocr_research/gray_training_outline_1105.md` §7，Exact 率瓶颈流程见 §8；其中「Exact 95%」
   尚无对应门槛，须先按 §7.2 书面裁决口径才可作为验收目标。
2. 训练/识别改动必须做**单变量 A/B**，并对照入口 skill §5 的已实测翻车清单
   （`--balance-real` 37/41→28/41、`--class-weight` 40/41→32/41、`--class-boost` 40/41→39/41、
   局部归一化 45.0%→37.9%），已失败方案不得重复。
3. 交付方案必须同时给出：适用的上游 skill 及其方法论要点、架构兼容性判定、A/B 设计与门禁、
   对下方技术约束红线的影响评估。
4. 不得直接修改 `skills/AI-Research-SKILLs/` 内文件；升级按 `VENDOR_PIN.md` 流程并重新验收。

## 技术约束

### TD3 零退化红线
- 分类器不确定时默认走 TD3 路径 (2×44)
- `FAST_NATIVE_MAX_WIDTH` 必须保持 1600
- `fastPair` 降采样必须用双线性 (filter=true)
- 生产使用 `0ab2131` 版本权重
- pic L1 模型训练应保持 real-frac 0.70

### CS/CD 港澳台通行证支持
- 单行 30 字符 MRZ 格式
- 格式: `{CS|CD}{国家2}{证件号8}<{出生YYMMDD}{出生CK}<{有效期YYMMDD}{有效期CK}<{复合CK}`
- mrz_tool decode 自动识别 30 字符输入并走 CS/CD 解码路径
- gui.py 支持单行 GT 对比和结果显示

### 构建与测试
```bash
# 构建所有工具
cmake --build 6_mrz_ocr/build --target mrz_ocr_cnn_tool --target mrz_ocr_cnn_tool_pic --target test_postproc -j4
# 构建解码工具
cmake --build 1_mrz_decode/build --target mrz_tool --target test_mrz -j4
# 单元测试
./6_mrz_ocr/build/test_postproc
./1_mrz_decode/build/test_mrz
# 评估 passport
python3 6_mrz_ocr/tools/eval_weight_sets.py --gt tests/gt_photo_passport_official.json --tool 6_mrz_ocr/build/mrz_ocr_cnn_tool
# 评估 pic
python3 6_mrz_ocr/tools/eval_weight_sets.py --gt tests/gt_assets_pic_official.json --tool 6_mrz_ocr/build/mrz_ocr_cnn_tool_pic
```
