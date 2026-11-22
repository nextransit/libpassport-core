# assets/pic 准确率提升实验报告 (2026-11-13)

## 基线

| 指标 | passport | pic |
|------|----------|-----|
| Both exact (cnn) | 40/44 (90.91%) | 44/62 (70.97%) |
| L1 char% | 96.88% | 89.98% |
| L2 char% | 99.78% | 90.50% |
| 目标准确率 | ≥98% | ≥95% |

## 已实施的改进

### 改进1：3-neighbour `<` 规则扩展
- **文件**: `6_mrz_ocr/src/mrz_ocr_cnn.c:2556-2572`
- **内容**: `<` 字符的 3-neighbour 规则从 X/R/T 扩展到 X/R/T/W/P/U/L/M/H/N/J/O
- **效果**: L2 char% +0.56%，对 exact 无帮助但无负面影响

### 改进2：Line 2 pos28-41 连续相同字符替换
- **文件**: `6_mrz_ocr/src/mrz_ocr_cnn.c:2591-2607`
- **内容**: Line 2 pos28-41 连续相同字符替换为 `<`
- **效果**: L2 char% +0.56%，对 exact 无帮助但无负面影响

### 改进3：looks44 阈值收紧 — 已撤销
- **文件**: `6_mrz_ocr/src/mrz_classify.c`
- **内容**: 将 `looks44 = (n_groups >= 34) || (energy44 <= energy30)` 收紧为 `>= 36`
- **效果**: A/B 测试显示 L1 char% -0.08%，对 exact 无帮助，**已撤销**

### Sauvola fallback for segment_line_grid
- **文件**: `6_mrz_ocr/src/mrz_ocr_cnn.c:4764-4779 (parallel), 4978-4988 (非 parallel)`
- **内容**: 当 Otsu 二值化后 `segment_line_grid` 返回 `nchars < 30` 时，使用 Sauvola 二值化重试
- **效果**: L1 char% +0.11%, L2 char% +0.19%，对 exact 无帮助但无负面影响

### c0 搜索范围扩展
- **文件**: `6_mrz_ocr/src/mrz_ocr_cnn.c:1515-1522`
- **内容**: 当 `L0 > Lfit * 1.15`（median gap 被异常间距拉高）时，扩展 phase search 的 c0 搜索范围到 `runs.center[left_run + 1]`
- **效果**: 对 `31611582cbf14d7f_0018.jpg` Line 2 的 c0 从 5 改善到 60.5，Line 1 未改善。无退化。

### _locate_band_bottom fallback
- **文件**: `6_mrz_ocr/tools/export_photo_train.py:79-96`
- **内容**: 当固定阈值 `< 170` 找不到有效 runs 时，使用 `< 100` 阈值和 5% min_ink 重试
- **效果**: `acf14e0c631ec935_0030.png` 的 locate_band 从 `(0, 450, 960, 24)` 改善到 `(0, 335, 960, 163)`，覆盖两行 MRZ。无退化。

## 失败案例根因分析

### 18张失败案例分类

#### 灾难性失败（5张）

| 图片 | 问题 | 根因 | 修复状态 |
|------|------|------|----------|
| `1792f7a2d79c7a8a_0017.jpg` | EMPTY | locate_band 只找到 w=402 | 分析完成，修复需旋转校正 |
| `31611582cbf14d7f_0018.jpg` | BOTH L1_err=44 L2_err=44 | 虚假边缘 run 导致网格偏移 | c0 扩展部分生效 |
| `61d671ec161f22e3_0042.png` | EMPTY→Sauvola fallback后有输出但全错 | Otsu 阈值=209，浅灰色背景被判定为 dark | Sauvola fallback 已添加 |
| `acf14e0c631ec935_0030.png` | BOTH L1_err=44 | locate_band 只找到第一行 | **已修复**（定位改善，识别仍乱码） |
| `72ef4ce48bf27731_0002.jpg` | BOTH L1_err=44 | L1 全是 W/N（`<` 被批量误识别） | 未修复 |

#### 严重错误（3张）
| 图片 | 问题 | 根因 |
|------|------|------|
| `b7a97ca0edede07b_0164.jpg` | L2_ONLY L2_err=30 | pos0-27 几乎每个字符都错 |
| `526d2e38b6b977bc_0014.jpg` | L1_ONLY L1_err=29 | L1 前4字符丢失 + 末尾多2字符 |
| `1c6acd092dee7031_0222.png` | L1_ONLY L1_err=23 | CS/CD 格式字符内容错误 |

#### 中等错误（5张，2-6字符错误）
| 图片 | 问题 |
|------|------|
| `3f5bb54276c6b754_0110.png` | L2_ONLY L2_err=6 |
| `7c6f47bad47d70e9_0230.jpg` | BOTH L1_err=1 L2_err=5 |
| `6dafa3b35c792d06_0039.png` | L2_ONLY L2_err=5 |
| `28a42952ba25584a_0078.png` | L1_ONLY L1_err=4 |
| `0f9780b8e1e3cc21_0008.jpg` | L1_ONLY L1_err=2 |

#### 轻微错误（5张，1字符错误）
| 图片 | 问题 |
|------|------|
| `f80334b200240666_0034.png` | L2_ONLY L2_err=2 |
| `23f16a7d9b6ab9da_0062.jpg` | L2_ONLY L2_err=1 |
| `556644d1d8bf4672_0215.jpg` | L1_ONLY L1_err=1 |
| `90b464ea218a9909_0001.jpg` | L2_ONLY L2_err=1 |
| `dd6a2ea3ce305a5d_0192.jpg` | L2_ONLY L2_err=1 |

## 关键技术发现

### `31611582cbf14d7f_0018.jpg` — 虚假边缘 run

**根因链条**:
1. band crop 左边缘包含黑色边框/阴影
2. upscale 后阴影形成 width=23（≈medw=24）的虚假 ink run，center=21
3. `first_center=21` 导致 phase search 范围 `[5, 37]`，正确 c0≈53 不在范围内
4. 网格偏移导致 cell 0 和 cell 43 溢出图像边界

**修复尝试**:
- `width < medw/2` 条件 — 无效（虚假 run width=23 ≈ medw=24）
- `gap > medw*1.3` 条件 — **严重退化**（passport 40→0, pic 44→5），已撤销
- `mrz_is_edge_shadow`（顶底 ink 检查）— **严重退化**（passport 40→33, pic 44→19），已撤销
- c0 搜索范围扩展（`L0 > Lfit*1.15` 时扩展到 `runs.center[left_run+1]`）— **部分生效**（Line 2 c0 从 5 到 60.5），无退化

### `acf14e0c631ec935_0030.png` — 低对比度灰色背景

**根因链条**:
1. 图片 y=467-655 是低对比度灰色区域（像素值 100-170）
2. 固定阈值 `< 170` 导致 dark 值饱和（226/226），超过 max_ink=198
3. `_locate_band_bottom` 排除了包含第二行 MRZ 的整个区域

**修复**: `_locate_band_bottom` 添加 `< 100` 阈值 fallback — **定位已修复**，但 CNN 识别仍乱码（深色背景上的 MRZ 未在训练数据中覆盖）

### `1792f7a2d79c7a8a_0017.jpg` — 旋转 + 光照不均

**根因链条**:
1. MRZ 旋转 ~3.4 度，`fast_pair` 无法找到平行的行对
2. 不均匀光照（左侧暗、右侧亮），右侧文本对 `adaptive_xpad` 不可见
3. 旋转伪间隙（44px）被 `adaptive_xpad` 解读为文本结束标志，截断到 w=402

**修复方向**: 需要旋转校正或自适应阈值，风险较高，暂未实施

### 改进4：Sex 字段合法值约束（col 20 → M/F/<）
- **文件**: `6_mrz_ocr/src/mrz_ocr_cnn.c:521-525`
- **内容**: 在 strict ICAO 模式下，将 col 20 (sex) 的 `fill_allow` 从允许所有 alnum + `<` 收紧为仅允许 M/F/<，与 loose 模式一致
- **效果**: 修复 `90b464ea218a9909_0001.jpg`（pos20: O→F），但引入 `78f45152a658f728_0012.jpg` 新失败（GT pos20='S' 非标准值，CNN 现在读 'F'）。净效果 0，但修复正确执行了 ICAO 9303 规范。GT 标注 'S' 需人工修正。

### 改进5：港澳台通行证类型码 'T' 支持
- **文件**: `6_mrz_ocr/src/mrz_ocr_cnn.c:477-487`（fill_allow col 0 添加 'T'）+ `mrz_ocr_cnn.c:2693-2710`（后处理规则 c3）
- **内容**: 
  1. 在 strict ICAO 模式下，将 'T' 加入 col 0 的允许列表（港澳台通行证非标准类型码）
  2. 添加后处理规则 (c3)：当 L1 col 0 = 'I' 且 issuing state (cols 2-4) 为 TWN/HKG/MAC 时，检查 'T' 是否在 top-3 候选中，若是则将 col 0 修正为 'T'
- **效果**: 修复 `556644d1d8bf4672_0215.jpg`（pos0: I→T），现在完全匹配 GT。无退化。

## 当前评估数据

| 指标 | passport | pic |
|------|----------|-----|
| Both exact (cnn) | 40/44 (90.91%) | 45/62 (72.58%) |
| L1 char% | 96.73% | 90.53% |
| L2 char% | 99.78% | 90.76% |

### 与基线对比

| 指标 | 基线 passport | 当前 passport | 基线 pic | 当前 pic |
|------|---------------|---------------|----------|----------|
| Both exact | 40/44 (90.91%) | 40/44 (90.91%) | 44/62 (70.97%) | 45/62 (72.58%) |
| L1 char% | 96.88% | 96.73% | 89.98% | 90.53% |
| L2 char% | 99.78% | 99.78% | 90.50% | 90.76% |

注：passport L1 char% 从 96.88% 微降到 96.73%，原因是 sex 字段约束使得 `78f45152a658f728_0012.jpg` 的 pos20 从 'S'（匹配 GT）变为 'F'（不匹配 GT），但该 GT 值 'S' 本身是非标准的。

## 修改的文件清单

| 文件 | 改动 |
|------|------|
| `6_mrz_ocr/src/mrz_ocr_cnn.c` | 改进1+2、Sauvola fallback、c0 搜索范围扩展、SEGDBG 移除、改进4（sex 字段约束）、改进5（港澳台类型码 'T'） |
| `6_mrz_ocr/src/mrz_classify.c` | 改进3已撤销，恢复原始版本 |
| `6_mrz_ocr/tools/export_photo_train.py` | `_locate_band_bottom` fallback |
| `6_mrz_ocr/tests/test_postproc.c` | 更新 `test_fill_allow_strict` 以反映 col 20 新约束 |

## GT 标注问题

以下图片的 GT 值存在非标准 sex 字段（pos 20），需人工核对修正：

| 图片 | GT pos20 | 问题 |
|------|----------|------|
| `78f45152a658f728_0012.jpg` | 'S' | ICAO 9303 只允许 M/F/<，'S' 非法 |
| `6dafa3b35c792d06_0039.png` | '7' | ICAO 9303 只允许 M/F/<，'7' 非法 |

## 下一步方向

1. **`1792f7a2d79c7a8a_0017.jpg`**: 需要旋转校正或自适应阈值，风险较高
2. **`31611582cbf14d7f_0018.jpg` Line 1**: c0 仍偏左，需要更精确的 first_center 估计
3. **`72ef4ce48bf27731_0002.jpg`**: `<` 被批量误识别为 W/N，需要分析 CNN 分类层面的问题
4. **深色背景图片**: `acf14e0c631ec935_0030.png` 等图片的 CNN 识别需要训练数据增强或图像预处理
5. **GT 标注修正**: `78f45152a658f728_0012.jpg` 和 `6dafa3b35c792d06_0039.png` 的 sex 字段需人工修正
