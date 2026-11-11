# 207ms 耗时分解分析报告

## 测试图片
`tmp/PassPortPhoto/2026-9-30-AG.jpg` (1440×960, fuse 引擎)

## 耗时分解（极速模式 MRZ_FAST_STRICT=1，非常驻）

| 步骤 | 非常驻 | 常驻(第2次) | 节省 |
|------|--------|-------------|------|
| cv2.imread | 3.2 ms | 2.9 ms | — |
| Sobel 翻转检测 | 1.5 ms | 1.5 ms | — |
| restore.load (resize) | 0.3 ms | 0.1 ms | — |
| detect_structure | 7.2 ms | 4.4 ms | — |
| drop_reflection | 0.0 ms | 0.0 ms | — |
| **_rst_native_mrz_band (2次)** | **21.3 ms** | **19.1 ms** | — |
|   — rgb.mean(axis=2) ×2 | **16 ms** | **16 ms** | **可优化到 0.24ms** |
|   — L1 扩展循环 ×2 | 0.02 ms | 0.02 ms | — |
|   — 其他 | 5.3 ms | 3.1 ms | — |
| **_ocr_pick_band** | **37.8 ms** | **13.0 ms** | **24.8 ms** |
|   — primary check (native_band+PPM+引擎) | ~12 ms | ~12 ms | — |
|   — gate check (native_band+PPM+引擎) | ~12 ms | ~2 ms | 常驻省进程启动 |
|   — band_candidates 构建 | ~14 ms | ~0 ms | gate 通过时跳过 |
| **_rst_run_ocr_methods** | **15.2 ms** | **8.8 ms** | **6.4 ms** |
|   — subprocess 启动开销 | ~7 ms | 0 ms | 常驻省进程启动 |
|   — fuse 引擎推理 | ~8 ms | ~8.8 ms | — |
| PPM 保存 | ~11 ms | ~0.5 ms | 固定路径 vs tempfile |
| **总计** | **80 ms** | **42 ms** | **37.6 ms (47%)** |

## 207ms 来源

用户报告的 207ms 对应 **非极速模式** (FAST_STRICT=0)：
- 7 引擎完整运行（tesseract 333ms, paddle 447ms, indra 767ms 首次冷启动）
- 第 2/3 次运行稳定在 ~216ms（含 A/B 假设重试）
- 极速模式下首次运行 126ms（含冷启动），后续 43ms

## 优化方案

### 方案 1: 修复 rgb.mean(axis=2) → cv2.cvtColor（最高优先级）

**问题**: `_rst_native_mrz_band` 中 `rgb.mean(axis=2)` 对 1440×960 图像耗时 8ms，
而 `cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)` 仅需 0.12ms — **快 69 倍**。
每次 `_ocr_photo_engines` 调用 `_rst_native_mrz_band` 2 次（primary + gate），
浪费 16ms。

**修复**: 将 `rgb.mean(axis=2)` 替换为 `cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)`。
注意：`cv2.cvtColor` 使用加权平均 (0.299R + 0.587G + 0.114B)，而 `mean` 是简单平均。
对于 L1 扩展的阈值判断（`< 128`），两种方法的差异不影响结果——墨量检测只关心
像素是否暗，加权 vs 简单平均的数值差异在阈值边界上几乎无区别。

**预期收益**: 每次 `_ocr_photo_engines` 省 ~16ms（从 80ms → 64ms，非常驻模式）

**风险**: 极低。灰度转换方法差异对暗像素检测无实质影响。

### 方案 2: 启用常驻引擎池（中等优先级）

**问题**: GUI 单张识别默认 `resident=False`，每次 `subprocess.run` 启动新进程 ~7ms。
`_ocr_pick_band` 中的 `_check` 也用非常驻方式，每个候选检查多付 ~5ms 进程启动。

**修复**: 将 `_ocr_photo_engines` 中的 `resident` 默认改为 `True`，并在 GUI 启动时
调用 `warmup_all()` 预热常驻池。

**预期收益**: 从 80ms → 42ms（省 37.6ms，47%）

**风险**: 中等。常驻池需要正确管理生命周期（切换权重集时清理旧进程）。
批测模式已使用常驻池验证过，但单张识别路径未测试。

### 方案 3: 只计算需要的行的灰度（低优先级）

**问题**: `_rst_native_mrz_band` 中 L1 扩展只需要 `[first_top - _max_exp, first_top)`
区间的灰度，但 `rgb.mean(axis=2)` 计算整张图。

**修复**: 只对需要的行做灰度转换：
```python
rows = rgb[max(0, first_top - _max_exp):first_top]
_gray_rows = rows.mean(axis=2) if rows.ndim == 3 else rows
```

**预期收益**: 0.06ms（vs 8ms），但与方案 1 互斥（方案 1 已将 8ms → 0.12ms）

### 方案 4: PPM 保存优化（低优先级）

**问题**: 非 `_fast_strict` 模式下使用 `tempfile.NamedTemporaryFile`，每次创建/删除
临时文件约 11ms。`_fast_strict` 模式下使用固定路径覆盖写，仅 0.5ms。

**修复**: 在非极速模式下也使用固定路径覆盖写（按线程隔离）。

**预期收益**: ~10ms（非极速模式）

## 推荐实施顺序

1. **方案 1**（rgb.mean → cv2.cvtColor）— 最低风险，立即收益 16ms
2. **方案 2**（常驻引擎池）— 中等风险，收益 37ms，需要更多测试
3. **方案 4**（PPM 固定路径）— 低风险，非极速模式收益 10ms

方案 1 + 2 合计可将非常驻模式从 80ms → 42ms（省 47%）。
