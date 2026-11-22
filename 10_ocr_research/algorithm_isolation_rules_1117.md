# 算法修改隔离规则 — Android 与桌面端互不影响

> 2026-11-17 制定。根因：commit `59daace`（Android 移植）修改共享 C 源文件，
> 导致桌面端 gui.py CRNN exact 从 50/50 退化到 14/50。

## 1. 共享代码边界

### 1.1 共享源文件清单

以下 8 个 C 源文件被桌面端（`6_mrz_ocr/CMakeLists.txt`）和 Android 端
（`8_android/app/src/main/cpp/CMakeLists.txt`）**同时引用**，修改任一行都会
影响两个平台：

| 文件 | 行数 | 职责 |
|------|------|------|
| `6_mrz_ocr/src/cnn.c` | ~450 | CNN 推理（conv/pool/fc） |
| `6_mrz_ocr/src/crnn.c` | ~420 | CRNN 推理（GRU/projection/output） |
| `6_mrz_ocr/src/mrz_ocr.c` | ~300 | OCR 核心入口/公共接口 |
| `6_mrz_ocr/src/mrz_ocr_cnn.c` | ~6128 | CNN 识别管线（TD3+CS/CD+ICAO mask） |
| `6_mrz_ocr/src/mrz_ocr_crnn.c` | ~200 | CRNN 识别管线 |
| `6_mrz_ocr/src/mrz_classify.c` | ~288 | 文档布局分类 |
| `6_mrz_ocr/src/template.c` | ~200 | 传统模板匹配 |
| `6_mrz_ocr/src/mrz_geom.c` | ~150 | 几何变换/倾斜校正 |

### 1.2 平台特定文件（不共享）

| 文件 | 平台 | 说明 |
|------|------|------|
| `8_android/app/src/main/cpp/mrz_jni.cpp` | Android | JNI 接口 |
| `8_android/app/src/main/cpp/mrz_run.cpp` | Android | JNI 执行包装 |
| `8_android/.../PhotoMrzPipeline.kt` | Android | Kotlin 识别管线 |
| `tests/gui.py` | 桌面端 | GUI + 识别管线 |
| `tests/photo_direct_bench.py` | 桌面端 | 基准测试 |
| `tests/mrz_arbitration.py` | 桌面端 | 多引擎仲裁 |

### 1.3 条件编译宏

| 宏 | 用途 | 安全性 |
|----|------|--------|
| `__ARM_NEON` / `__ARM_NEON__` | NEON SIMD 向量化 | **安全**：桌面端 x86 不触发 |
| `__ANDROID__` | Android logcat | **安全**：桌面端不触发 |
| `__linux__` | Linux 线程亲和性 | **不安全**：macOS 不触发但 Linux CI 会触发 |
| `MRZ_WEIGHTS_SET_PIC` | 编译时权重集选择 | **安全**：桌面端用独立二进制 |
| `_WIN32` | Windows 兼容 | **安全** |

## 2. 修改规则

### 规则 R1：共享文件中的算法修改必须通过桌面端门禁

**任何**对共享 C 源文件的修改（包括 NEON 优化、CS/CD 支持、ICAO mask 调整），
在提交前必须通过以下桌面端门禁：

```bash
# 构建
cmake --build 6_mrz_ocr/build --target mrz_ocr_cnn_tool --target mrz_ocr_cnn_tool_pic --target test_postproc -j4
cmake --build 1_mrz_decode/build --target mrz_tool --target test_mrz -j4

# 单元测试（不得退化）
./6_mrz_ocr/build/test_postproc    # 必须 147/147
./1_mrz_decode/build/test_mrz      # 必须 89/89

# 护照照片基准（不得退化）
MRZ_FAST_STRICT=0 python3 tests/photo_direct_bench.py --set passport_union --methods crnn
# 必须 ≥ 46/50 PASS（official path）

# assets/pic 基准（不得退化）
MRZ_FAST_STRICT=0 python3 6_mrz_ocr/tools/eval_all_engines.py
# pic CNN both exact 必须 ≥ 41/62
```

### 规则 R2：NEON 优化不得改变 `#else` 分支的数值结果

NEON 重构允许改变循环顺序和累加方式，但**桌面端 `#else` 分支的输出必须与
修改前逐字节一致**。验证方法：

```bash
# 修改前保存基线
MRZ_FAST_STRICT=0 python3 tests/photo_direct_bench.py --set passport_union --methods crnn --json /tmp/baseline_before.json

# 修改后对比
MRZ_FAST_STRICT=0 python3 tests/photo_direct_bench.py --set passport_union --methods crnn --json /tmp/baseline_after.json

# diff 两个 JSON 的 line1/line2 字段，必须完全一致
```

**禁止**：为了 NEON 方便而改变 `#else` 分支的循环顺序或浮点累加方式。
如果 NEON 需要不同的循环顺序，必须在 `#if defined(__ARM_NEON)` 分支内完成，
`#else` 分支保持原样。

### 规则 R3：ICAO mask 修改必须单独评估对桌面端的影响

`mrz_ocr_cnn.c` 中的 `fill_allow()` 函数定义了每个位置允许的字符集合。
对它的任何修改（添加/删除/收紧字符）都会同时影响桌面端和 Android 端。

**必须**：
1. 修改前记录当前 50 张照片的 exact 基线
2. 修改后重新运行 50 张照片基准
3. 如果桌面端 exact 下降，修改必须回退或改为 Android 专用条件编译

**示例**：commit `59daace` 中 col 0 添加 `'T'`（港澳台通行证）和 col 20
收紧为 `M/F/<` 是为了 CS/CD 支持，但这些修改同时影响了桌面端的 TD3 识别。
正确的做法是在 `fill_allow()` 中根据 `g_doc_layout` 区分 TD3 和 CS/CD 的
mask，而不是全局修改。

### 规则 R4：CS/CD 功能添加不得改变 TD3 识别路径

CS/CD（港澳台通行证）支持添加了大量代码到 `mrz_ocr_cnn.c`（约 500+ 行）。
这些代码在桌面端也会被编译和执行。

**必须**：
1. CS/CD 代码路径必须通过 `g_doc_layout == MRZ_LAYOUT_SINGLE_LINE_30` 守卫
2. TD3 识别路径（`g_doc_layout == MRZ_LAYOUT_TD3_DOUBLE_44`）不得因 CS/CD
   代码的添加而改变行为
3. 分类器（`mrz_classify.c`）的 TD3 判定逻辑不得因 CS/CD 支持而放宽

**禁止**：
- 在 TD3 路径中调用 CS/CD 函数（即使是无害的）
- 修改 TD3 的 `fill_allow` mask 来兼容 CS/CD 字符
- 修改 TD3 的分割/解码逻辑来处理 30 字符格式

### 规则 R5：底部 trim 逻辑修改必须验证不误剪 MRZ 行

`mrz_ocr.c` 的底部 trim 逻辑是导致 CRNN exact 从 50/50 退化到 14/50 的根因。
trim 逻辑从底部向上找 ≥5 行低墨水间隙就修剪，但 MRZ 两行之间也有这样的间隙，
导致第二行被误剪。

**必须**：
1. 找到间隙后，必须验证间隙下方是否有黑边（`row_dark >= max_text_dark` 的行
   占 ≥50%），只有确认是黑边才 trim
2. 修改后必须运行 50 张照片基准，确认 L2 不被误剪

### 规则 R6：功能解耦——独立功能必须在独立 commit 中提交

**禁止**：在一个 commit 中同时提交多个独立功能（如 CS/CD 支持 + NEON 优化 +
ICAO mask 修改）。每个独立功能必须单独提交，以便：
1. 独立测试每个功能的影响
2. 出现退化时可以精确回退单个功能
3. 代码审查时可以聚焦单个功能的正确性

### 规则 R7：Kotlin/C 重复实现必须同步参数

`PhotoMrzPipeline.kt` 中的 CS/CD 单行定位逻辑与 `mrz_ocr_cnn.c` 中的
`cscd_locate_text_row`、`cscd_rescue_band` 是同一算法的两份实现。

**必须**：
1. 修改 C 版本的参数/阈值时，必须同步修改 Kotlin 版本
2. 修改 Kotlin 版本的参数/阈值时，必须同步修改 C 版本
3. 两份实现的输出必须在一组测试照片上完全一致

## 3. 验证门禁流程

### 3.1 提交前必跑

```
┌─────────────────────────────────────────┐
│ 1. 构建（两个 target + 单元测试）        │
│    cmake --build ... -j4                │
│    ./test_postproc  (147/147)           │
│    ./test_mrz       (89/89)             │
├─────────────────────────────────────────┤
│ 2. 护照照片基准（不得退化）              │
│    photo_direct_bench.py --set passport_union │
│    official path PASS ≥ 46/50           │
│    CRNN alone exact ≥ 43/47             │
├─────────────────────────────────────────┤
│ 3. assets/pic 基准（不得退化）           │
│    eval_all_engines.py                  │
│    pic CNN both exact ≥ 41/62           │
├─────────────────────────────────────────┤
│ 4. 如果修改了 NEON 代码：               │
│    #else 分支输出必须与修改前逐字节一致  │
├─────────────────────────────────────────┤
│ 5. 如果修改了 fill_allow：              │
│    TD3 路径 exact 不得下降              │
│    CS/CD 路径 exact 不得下降            │
├─────────────────────────────────────────┤
│ 6. 如果修改了 trim 逻辑：               │
│    50 张照片 L2 不被误剪                │
└─────────────────────────────────────────┘
```

### 3.2 退化判定

- **硬退化**：门禁测试数值下降 → 必须回退修改
- **软退化**：门禁测试数值不变但某些照片的识别结果变化 → 需要人工审核
- **无影响**：门禁测试数值不变且识别结果不变 → 可以提交

## 4. 代码组织改进建议

### 4.1 拆分 `mrz_ocr_cnn.c`

当前 6128 行的 `mrz_ocr_cnn.c` 承载了过多职责。建议拆分：

| 新文件 | 从 mrz_ocr_cnn.c 提取 | 行数 |
|--------|----------------------|------|
| `mrz_cscd.c` | CS/CD 定位/分割/解码/校验修复 | ~500 |
| `mrz_icao_mask.c` | `fill_allow()` + ICAO syntax mask | ~200 |
| `mrz_ocr_cnn.c` | TD3 识别管线（保留） | ~5400 |

### 4.2 统一 CMakeLists.txt 源文件列表

将 8 个共享源文件列表提取到一个 `6_mrz_ocr/src/sources.cmake` 文件中，
两个 CMakeLists.txt 都 include 它，避免手动同步。

### 4.3 消除 Kotlin/C 重复实现

将 CS/CD 定位逻辑统一到 C 端，Kotlin 端通过 JNI 调用 C 函数，
而不是自己重新实现一份。

### 4.4 全局状态线程安全

`g_doc_layout` 全局变量改为函数参数传递，消除竞态条件风险。

## 5. 历史教训记录

| 事件 | 根因 | 影响 | 修复 |
|------|------|------|------|
| CRNN exact 50/50→14/50 | `mrz_ocr.c` 底部 trim 误剪 L2 | gui.py 退化 | 添加黑边验证条件 |
| CRNN exact 46/50→43/50 | commit `59daace`/`634b278` 修改共享代码 | 桌面端退化 | 恢复未提交的改进修改 |
| 仲裁器选错引擎 | crnn 权重 0.3 过低（已过时） | 6 张照片 FAIL | 提高 crnn 权重到 1.0 |
| NEON 重构影响桌面端 | `#else` 分支循环顺序被改变 | 潜在数值差异 | 规则 R2 禁止此行为 |
| ICAO mask 全局修改 | col 0 添加 'T'、col 20 收紧 | TD3 识别行为改变 | 规则 R3 要求按 layout 区分 |
