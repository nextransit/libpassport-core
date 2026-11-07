# A/B 结果：渲染方式（物理 vs 遗留二值）—— 2026-11-06

> 协议：`ab_protocol_render_1106.md`
> 基线 commit：40951c5（passport 40/41, pic 49/61, 速率 17.9 张/秒）

## 训练

| arm | 训练语料 | 切片数 | line | best_val |
|-----|---------|--------|------|----------|
| L（对照） | corpus_train_v2（legacy） | 248068 | l1 | 0.9992 |
| L（对照） | corpus_train_v2（legacy） | 248068 | l2 | 0.9987 |
| P（处理） | corpus_train_phys（物理） | 105519 | l1 | 0.9998 |
| P（处理） | corpus_train_phys（物理） | 105519 | l2 | 0.9997 |

训练参数：`train_cnn_omtest.py --line {0,1} --real-frac 0.5 --epochs 40 --seed 1`，
`--synth-extra-npz` 分别指向 legacy / phys 切片，`--real-npz data/photo_train_passport/cnn_slices.npz`。

物理切片导出：3000 张图片中 tool_fail=1273（42%），align_drop=985（33%），
成功导出 1394 张图片的 105519 个切片。

## 4 格矩阵（corpus full exact）

| 模型 | 语料 | full exact | l1_acc | l2_acc | ok | clean full | realistic full |
|------|------|-----------|--------|--------|-----|------------|----------------|
| L | legacy (corpus_eval_v2) | **67.08%** | 95.03% | 94.89% | 100% | 87.36% | 46.81% |
| L | phys (corpus_eval_phys) | 0.35% | 43.44% | 32.69% | 55.83% | 0.69% | 0.00% |
| P | legacy (corpus_eval_v2) | 5.00% | 88.68% | 77.30% | 100% | 8.89% | 1.11% |
| P | phys (corpus_eval_phys) | **24.58%** | 52.77% | 43.59% | 55.83% | 35.00% | 14.17% |

## 官方集（eval_weight_sets.py）

| 模型 | passport both | 备注 |
|------|--------------|------|
| 生产基线 | 40/41 | |
| P (phys) | 40/41 | 无回退 |

pic 官方集不适用（用 `mrz_ocr_cnn_tool_pic` + pic 权重，不在本次 A/B 范围）。

## 真实照片（photo_direct_bench.py --set passport_union）

| 模型 | cnn GT | cnn l1/l2 acc | fuse GT | fuse l1/l2 acc |
|------|--------|---------------|---------|----------------|
| 生产基线 | 37/47 | 99.3/96.8 | 37/47 | 99.3/96.8 |
| P (phys) | 37/47 | 99.2/97.6 | 37/47 | 99.2/97.6 |

真实照片无回退。

## 判定：**驳回**

### 协议判定规则核对

| 条件 | 结果 | |
|------|------|-|
| arm P 在 corpus_eval_phys 上 full exact ≥ arm L +5pp | 24.58% vs 0.35% = +24.23pp | ✓ |
| arm P 在 corpus_eval_phys 上 full exact ≥ 90% | 24.58% | ✗ |
| 真实照片不回退 | 37/47 = 37/47 | ✓ |
| 官方集不回退 | 40/41 = 40/41 | ✓ |
| false accept 不升 | 未升 | ✓ |
| 速率 ≥ 2.0 | ~17.9 张/秒 | ✓ |

### 驳回理由

1. **P×phys = 24.58% 远低于 90% 目标**：物理语料本身质量有问题。
   `ok=55.83%` 意味着 44% 的物理渲染图片工具无法定位 MRZ 区域——
   物理渲染的图片与工具的 MRZ 定位器不兼容。
2. **P×legacy = 5.00% 严重回退**（L×legacy = 67.08%，-62.08pp）：
   phys 权重丧失了识别 legacy 渲染图片的能力。
   4 格矩阵表明 P 只在 phys 语料上有效，在 legacy 语料上崩溃——
   这是"语料过拟合"而非"模型更强"。
3. **物理切片导出 tool_fail=42%**：物理渲染的图片质量与工具预期差异太大，
   导致切片导出成功率低，训练信号不足。

### 根因

物理渲染流水线（S2-S8）生成的图片与工具的 MRZ 定位器不兼容。
定位器在 legacy 渲染（1-bit + blur3）上训练，无法定位物理渲染
（灰度 + 底纹 + 光照 + 噪声）的 MRZ 区域。这导致：
- 物理语料 ok=55.83%（定位失败率 44%）
- 切片导出 tool_fail=42%
- 训练信号不足且分布偏移

### 结论

**语料物理化本身不带来产品收益**。物理渲染训练的模型：
- 在真实照片和官方集上无回退（因为真实照片分布介于 legacy 和 phys 之间）
- 但在 corpus 评估上严重回退（P×legacy=5.00%）
- 物理语料绝对值太低（P×phys=24.58% << 90%），无法作为验收目标

要使物理渲染有效，需先修复工具的 MRZ 定位器，使其能定位物理渲染的图片。
但这属于定位器改造，不在本次 A/B 范围内。

## 翻车清单更新

| 已试杠杆 | 结果 |
|---|---|
| 物理渲染 A/B（§4.1/§4.3） | **驳回**：P×phys=24.58% << 90%，P×legacy=5.00% << 67.08%，物理语料 ok=55.83% |

## 文件

- 权重：`/tmp/ab_render/{legacy,phys}_{l1,l2}`
- 日志：`/tmp/ab_render/{legacy,phys}_{l1,l2}.log`
- 评估 JSON：`/tmp/ab_render/{phys_passport,phys_pic,phys_photo,prod_photo}.json`
- 生产权重已恢复，工具已重新编译
