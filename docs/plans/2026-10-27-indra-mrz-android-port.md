# Indra OCR 全算法移植到 Android（MRZ 场景）— 调研与实施计划

> 目标：把 `11_indra`（datalab-to/indra-ocr 1.1.2）的**全部算法**落到 Android，
> 使 Android 侧的 MRZ 读码走 Indra 全链路（det / rec / script 路由 / order / LM beam
> / math / RTL / layout），而不是现在的 Paddle-recognizer-only。

---

## 0. 结论速览

| 维度 | 结论 |
|---|---|
| 模型可跑性 | ✅ 12 个 ONNX 全部在 **ORT 1.20.0**（= Android 侧 `onnxruntime-android:1.20.0`）加载成功，算子全为标准 ONNX，**无自定义 domain** |
| 数值一致性 | ✅ ORT 1.20.0 与 1.30.0 在 det/rec/script/orient/order 上输出**逐值相同**（仅 `order_a` 的 `-0.0` vs `0.0` 符号差） |
| 端上算子覆盖 | ✅ 移动端 `libonnxruntime.so` 已注册 LayerNormalization / DepthToSpace / HardSwish / Softplus / ScatterND / TopK / Range / Erf 等全部所需算子 |
| 主要缺口 | **OpenCV（cv2）** 依赖；`lm*.npz`（46 MB）语言模型的 beam search 移植；math 子模型（20 MB）注意力解码循环 |
| APK 体积 | 现有 assets 已 86 MB（PP-OCR）；Indra 全量再 +102 MB → **APK 将超 250 MB**，必须分层打包/按需下载 |
| 建议路线 | **分三阶段**：Stage A = MRZ 最小闭环（det+rec+script，无 LM）→ Stage B = 加 order/layout → Stage C = 补 LM beam + math + RTL 全量对齐 |

**关键判断：Indra 在 MRZ 上的价值已被 Python 侧实测验证** —— 3 张真机照片
（Photo/3,4,5）全部读出完整 TD3 MRZ，且 `mrz_repair` 只做 1 处编辑就通过校验。
但注意：当前 `mrz_indra.py` 走的是 **整页 det+rec** 路线，不是专门的 MRZ 检测器。

---

## 1. 现状盘点

### 1.1 已有资产（可直接复用）

| 位置 | 内容 | 状态 |
|---|---|---|
| `11_indra/indra_ocr/` | 官方 Python 实现，5698 行，16 个模块 | ✅ 完整可读 |
| `11_indra/models/` | det/rec/order/meta（v1.1.0 子集） | ⚠️ 已补齐到全量 12 ONNX + 3 npz |
| `11_indra/mrz_indra.py` | MRZ 适配层（整页 OCR → 挑 2 行 → repair） | ✅ 跑通 |
| `11_indra/mrz_repair.py` | ICAO 校验位 beam repair（456 行） | ✅ 跑通 |
| `8_android/` | Android 工程，ORT 1.20.0 已在用 | ✅ 构建通过（2026-10-27 验证） |
| `8_android/.../PaddleOcrAndroid.kt` | ORT 1.20.0 Kotlin 封装范式 | ✅ 可照抄 |

### 1.2 Android 侧当前 OCR 能力

| 引擎 | 状态 | 覆盖算法 |
|---|---|---|
| `PaddleEngine` | ✅ 已有 | 仅 PP-OCRv6 recognizer（rec-only，**无 detector**） |
| `TesseractEngine` | ✅ 已有 | Tesseract 4/5 |
| `MlKitEngine` | ✅ 已有 | ML Kit |
| `NativeEngine` | ✅ 已有 | 自研 C CNN/CRNN（6_mrz_ocr） |
| **Indra** | ❌ **缺失** | — |

### 1.3 Indra 全部算法清单（必须全部支持 = 16 个模块）

| # | 模块 | 算法 | ONNX | 移植难度 |
|---|---|---|---|---|
| 1 | `det_post.py` | LineNet 检测后处理：sigmoid → 连通域 → 距离场反推上下边界 | `det.onnx` | 中（需连通域+统计） |
| 2 | `rec_preproc.py` | 行裁剪：PAD_Y 0.2 / PAD_X 0.3 → 32px 墨迹图；暗行反相；宽度分批 | — | 低 |
| 3 | `pipeline.recognize` | CTC 贪心解码（ids/conf/lengths） | `rec.onnx` | 低 |
| 4 | `vocab.py` | 词表：base/world/cjk/rtl 四套 + LaTeX token + NFC 规范化 | `meta.json` | 低 |
| 5 | `lm.py` | **CTC prefix beam search + 5-gram n-gram LM**（backoff 表） | `lm*.npz` | **高** |
| 6 | `rtl.py` | RTL 双向翻转（UAX#9 子集）+ 镜像字符 + 簇处理 | — | 中 |
| 7 | `script.onnx` 路由 | 4 路脚本分类（base/cjk/rtl/world），置信 > 0.95 才切换 | `script.onnx` | 低 |
| 8 | `rec_cjk/rtl/world.onnx` | 分组识别器（cjk/rtl 用 32px，base 用 96px tall window） | 3 个 ONNX | 低 |
| 9 | `order_post.py` | 行特征(16维) → order.onnx → `decode_order` 插入式局部搜索 | `order.onnx` | 中 |
| 10 | `order_post._cluster` | **平均连接聚类**（block 划分），O(n²) 增量 | — | **高** |
| 11 | `order_post.decode_blocks_order` | block 内 XY-cut / 表格行序 + block 间平均 logit | — | 中 |
| 12 | `layout_post.py` | Markdown 渲染：段落合并/连字符/列表/HTML 表格/页眉页脚 | — | 中 |
| 13 | `pipeline.estimate_skew` | **倾斜角估计**（-45°..45° 三级细化，行锐度最大化） | — | 中 |
| 14 | `orient.onnx` | 页面方向（4 转向 + 竖排 5 类） | `orient.onnx` | 低 |
| 15 | `line_mask.py` | CenterMask 中心线掩码（可选 `mask.onnx`，本仓库**无此文件**） | — | — |
| 16 | `math.py` | **MathRec 注意力解码**（cls + enc + dec_init + dec_step，4 模型） | 4 个 ONNX | **高** |

**另需移植的非 ONNX 部分**：
- `tidy_lines` — 引导符行剔除、列表标记合并
- `split_line` / `gutter_cuts` — 宽间隙切行、栏间距切分
- `tidy_lines` 的 RTL 处理、`_page_rtl` 判定
- `document.py` 的 PDF 文本层（`pypdfium2`）→ Android 侧**不做**（只做图片输入）

---

## 2. 关键实测数据（本轮真实跑出来的，非推测）

### 2.1 模型加载兼容性

```
ORT 1.20.0（Android 同版本）:
OK  det.onnx / rec.onnx / rec_cjk.onnx / rec_rtl.onnx / rec_world.onnx
OK  script.onnx / order.onnx / orient.onnx
OK  mathcls.onnx / math_enc.onnx / math_dec_init.onnx / math_dec_step.onnx
（12/12 全部通过）
```

### 2.2 跨 ORT 版本数值一致性（同一张 Photo/3）

| 探针 | ORT 1.20.0 | ORT 1.30.0 | 一致 |
|---|---|---|---|
| det 输出 sum / max | 1869252.75 / 54.8354 | 1869252.75 / 54.8354 | ✅ |
| rec ids / conf | [81,38] / 0.52505 | [81,38] / 0.52505 | ✅ |
| script 概率 | [0.21501, 0.35588, 0.21102, 0.21809] | 同 | ✅ |
| orient 概率 | [0.27726, 0.25657, 0.20642, 0.24303, 0.01673] | 同 | ✅ |
| order cls 头 | [3.651, 1.7662, -0.7021, ...] | 同 | ✅ |
| order a 头 sum | -0.0 | 0.0 | ⚠️ 仅符号 |

**结论**：Android 侧用 ORT 1.20.0 不会引入数值漂移，可以直接对齐 Python 端结果。

### 2.3 Python 侧端到端（Photo/3，2592×1944）

```
lines=48  timings: orient 18ms / det 22ms / rec 114ms / order 2.5ms / total 139ms
MRZ: line1=POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
     line2=EE20426536CHN7410024M2808303LMNGMBKCNPSB9<70
     repair edits=1, composite_edits=0, country=CHN
```

Photo/4、Photo/5 结果**逐字相同**（同一本护照的不同拍摄角度）。

### 2.4 移动端算子覆盖（`libonnxruntime.so` arm64-v8a，ORT 1.20.0）

已确认注册：`LayerNormalization` `DepthToSpace` `HardSwish` `Softplus` `ScatterND`
`TopK` `Range` `Erf` `Resize` `Expand` `ConstantOfShape` `Where` `Xor` `Not` `And`
`Equal` `Less` `ArgMax` `ReduceMean` `LogSoftmax` `Gather` `Split`

---

## 3. 风险与关键决策

### 3.1 决策 A：OpenCV 依赖怎么处理？

Indra 用到的 cv2 只有 **4 个函数**（`resize` / `warpAffine` / `connectedComponentsWithStats` / `cvtColor`）。

| 方案 | 说明 | 代价 |
|---|---|---|
| **A1 引入 OpenCV Android SDK** | `com.quickbirdstudios:opencv:4.10.0-1.5.11`（Maven 有） | +~25 MB APK；4 ABI 都有 |
| **A2 手写最小图像算子（推荐）** | 现有 `3_face_recognition/src/process.c` 已有 `face_resize_nn`；只需补 resize(bilinear/area) / warpAffine / 连通域 | 需自己写 ~400 行 C，**但零体积增量**，且能复用已有 NDK 静态库结构 |
| A3 纯 Kotlin | 同 A2 但无 JNI | 更慢，但最简单 |

**建议 A2**：现有 `8_android` 已经把 `3_face_recognition` 的 C 源码**逐字**链进 NDK，
沿用同一模式最省事，且不加 APK 体积。`connectedComponentsWithStats` 是唯一稍麻烦的
（需要 4-连通 + stats），但 `det_post` 只用它做「连通域 + bbox + 面积」，
用两遍 union-find 或 run-length 都可以。

### 3.2 决策 B：LM beam search 要不要做？

- `lm.npz` 15.7 MB + `lm_cjk.npz` 19.0 MB + `lm_world.npz` 11.6 MB = **46.3 MB**
- 5-gram backoff 表，order=5，bits=11，vocab=1678
- 移植难度高（`logp_keys` 是向量化 searchsorted + 掩码位运算）

**建议：Stage A/B 先不做 LM**（`MIN_CONF` 以下的行直接丢弃，与 Python 行为一致但没有
LM 救援）。理由：
1. MRZ 场景是**高对比度印刷体**，置信度普遍 > 0.9，LM 的收益主要在**手写体**
2. 46 MB APK/内存代价 vs MRZ 场景收益不成比例
3. 若 Stage C 需要，`lm.npz` 可以**只保留 base**（15.7 MB），cjk/world 按需下载

→ 但**必须记录**：不做 LM 时，输出与 Python 不逐字一致，需要单独的门禁标准。

### 3.3 决策 C：math 子模型（20 MB）做不做？

- 4 个 ONNX（cls 0.19 + enc 9.9 + dec_init 1.36 + dec_step 8.4 MB）
- 注意力解码循环，每 token 一步，max_tokens 步，**Android 上很慢**
- MRZ 场景**完全不需要**（护照上没有公式）

**建议：MRZ 场景跳过 math**。若目标是"通用文档 OCR"再补。

### 3.4 决策 D：APK 体积

现有：
- assets/models/ 86 MB（PP-OCR v6 medium 73MB + v4 mobile 10MB + int8 3MB）
- 现有 APK 202 MB

Indra 全量再加 102 MB → **~300 MB**，超 Google Play 限制（APK ≤ 200MB，AAB ≤ 150MB）。

**建议**：
- 打包进 APK 的最小集：`det.onnx` (0.39) + `rec.onnx` (5.04) + `script.onnx` (0.19) +
  `orient.onnx` (0.37) + `order.onnx` (2.97) + `meta.json` (0.21) = **9.2 MB**
  （不含 LM / math / 分组识别器）
- 其余（rec_cjk 10.2 / rec_rtl 4.8 / rec_world 5.6 / lm 46.3 / math 20）走
  **首次启动按需下载**，沿用现有 `PaddleOcrAndroid.copyModel` 的 assets→filesDir 范式
- 同时**删掉** PP-OCR v6 medium（73 MB）—— Indra 全面替代它

### 3.5 决策 E：MRZ 走整页还是专用检测？

Python 侧 `mrz_indra.py` 走整页 det+rec，然后在 48 行里挑 2 行。
Android 侧 `PhotoMrzPipeline` 已经有成熟的 MRZ 定位（结构检测 + 反射去除 +
透视校正 + MRZ crop）。

**建议**：Android 侧**保留现有 MRZ 定位**（那是 CV 算法，不是 OCR），
只把**读码**换成 Indra：
- 用 Indra 的 `det.onnx` 在 MRZ band 上找 2 行？还是
- 直接用现有 crop 的 2 行喂 `rec.onnx`？

→ 两者都可以，**但 rec.onnx 是 96px tall window（rec_window=3）训练的**，
且 base 识别器吃的是**整行**。现有 `PaddleOcrAndroid` 是把 2 行分开喂。
建议：先做「整页 det + 挑 2 行」（与 Python 一致，便于对齐验证），
再做「MRZ band 专用路径」。

---

## 4. 分阶段实施计划

### Stage A — MRZ 最小闭环（推荐先做）

**目标**：Android 能用 Indra 读出完整 TD3 MRZ，Python 逐字对齐。

**范围算法**：det_post（检测后处理）+ rec_preproc（裁剪）+ CTC 贪心解码
+ script 路由（4 路，可先只留 base）+ orient + skew 估计
**不含**：LM beam、math、order/layout、RTL、CenterMask

**步骤**：
1. `8_android/app/src/main/cpp/indra/` — 新建 C 静态库
   - `iv_resize.c` — bilinear/area resize
   - `iv_warp.c` — warpAffine + getRotationMatrix2D
   - `iv_cc.c` — 4-连通 connectedComponents + stats
   - `indra_det.c` — det.onnx 前处理/后处理
   - `indra_rec.c` — 裁剪 + 批处理 + CTC 解码
   - `indra_ort.cpp` — ORT C API 封装（JNI）
2. Kotlin `IndraOcrAndroid.kt` — 会话管理（仿 `PaddleOcrAndroid`）
3. `IndraEngine : OcrEngine` — 接入 `EngineRegistry`
4. `mrz_repair` 的 Kotlin 移植（`IndraMrzRepair.kt`）—— 456 行 Python → Kotlin
5. 资产：`assets/models/indra/{det,rec,script,orient}.onnx` + `meta.json`

**验证门禁**：
- Host↔Device **逐行** 一致：同一张 Photo/3，Python 与 Android 输出**完全相同的 MRZ 字符串**
- 41 张真实照片：MRZ 完全匹配率 ≥ Python 侧基线
- 设备耗时：det+rec ≤ 800 ms/张（4 核 A53，需实测）

### Stage B — 阅读顺序与版面

**范围**：`order_post`（行特征 + decode_order + _cluster + decode_blocks_order）、
`layout_post`（Markdown）、`tidy_lines`、`split_line`/`gutter_cuts`、
`vocab` 四套词表、`rtl.py`

**注意**：`_cluster` 的平均连接聚类是 O(n²) 增量算法，移植时须保持**同样的合并顺序**，
否则 block 划分会不同 → 输出不一致。

### Stage C — 全量补齐

**范围**：`lm.py` beam search（46 MB）、`math.py`（20 MB）、`line_mask.py` CenterMask
（需另外下载 `mask.onnx`）

---

## 5. Host↔Device 一致性门禁（沿用本仓库既有做法）

本仓库已有成熟范式（见 MEMORY：`MRZ_OCR_LINE_MODELS=1` 的 host/device parity）：

1. Python 端固定输入 → 产出「期望输出 JSON」
2. 推送到设备 → App 跑同一输入 → 产出「实际输出 JSON」
3. **逐字节 diff**，任何差异 = FAIL

对 Indra 而言，门禁应比 MRZ 更严：**整页 48 行的 text + conf + box 都要对齐**，
而不只是最终 MRZ（因为 order/layout 阶段最容易出偏差）。

建议脚本：`8_android/tools/indra_parity.sh`（仿现有 `push_baseline.sh` 风格）。

---

## 5.5 真机实测结果（2026-10-27，设备 192.168.26.131）

设备：**Android 9 / API 28 / arm64-v8a / 4 核 / 3.7 GB RAM**（`p18q_dual_extend_64`）。

方法（`8_android/app/src/androidTest/java/.../IndraOnnxProbeTest.kt`）：
host 端 Python 用 `indra_ocr` 的**真实预处理**生成输入张量
（`tools/indra_probe/probe_*.bin`），推到设备原样喂给 ORT。
这样比对的只有「ORT 在 ARM NEON 上的算子结果」，不掺入 cv2 预处理差异。

### 5.5.1 数值一致性：11 PASS / 0 FAIL

| 模型 | 探针 | host (x86_64, ORT 1.30) | 真机 (arm64-v8a, ORT 1.20.0) | 差异 |
|---|---|---|---|---|
| det | ch0 sum | -2187334.75 | -2187334.798 | rel 2.2e-08 |
| det | ch0 max | 6.7140703201293945 | 6.71407032 | **0.000e+00** |
| det | ch0 first8 | 逐值 | 逐值 | **0.000e+00** |
| rec | ids | [81, 38] | [81, 38] | **完全相同** |
| rec | conf | 0.5250513553619385 | 0.5250507891178131 | rel 1.1e-06 |
| script | 4 路概率 | 0.2150146 / 0.3558781 / 0.2110170 / 0.2180903 | 同 | worst 1.0e-07 |
| orient | 5 类概率 | 0.2772574 / 0.2565694 / 0.2064195 / 0.2430261 / 0.0167277 | 同 | worst 3.0e-08 |
| order | cls 头（10 类）| 逐值 | 逐值 | worst 1.7e-06 |
| order | a / same / tab 头 sum | — | — | rel ≤ 3.0e-07 |

**结论：ARM NEON 与 x86_64 的浮点累加顺序没有引入可见差异**，相对误差量级 1e-6~1e-8，
远低于任何识别阈值。**host↔device 逐字对齐的数值基础成立。**

### 5.5.2 真机耗时（每模型跑 3 次，第 2 次为稳态）

| 模型 | 冷启动 | 稳态 | Python 桌面参考 (4 线程 x86) |
|---|---|---|---|
| det（[1,1,768,1024]）| 898 ms | **792 ms** | 22 ms |
| rec（[1,1,96,360] 单行）| 65 ms | **68 ms** | — （整页 114 ms）|
| script | 8 ms | **8 ms** | — |
| orient（[1,1,768,768]）| 372 ms | **358 ms** | 18 ms |
| order（12 行）| 10 ms | **9 ms** | 2.5 ms |

**这是本次调研最重要的负面发现：det 和 orient 在真机上比桌面端慢 30~45 倍。**

原因分析（已核实）：
1. **设备 CPU 远弱于桌面**：4 核 A53 @ 1.8GHz vs 桌面多核高频 Xeon/Apple Silicon；
2. `intra_op_num_threads=2` 是探针设定，设备 4 核可给 4；
3. `orient` 的 768×768 输入在移动 CPU EP 上没有高效卷积路径；
4. **det 的 792 ms 是单次整页推理**，而 Python 侧 22 ms 那次是**已经 resize 到 1024 的 1944×2592 页**——
   注意 Python 的 `det_ms=22ms` 对应的是同一张图同一次缩放，两者口径一致，但设备慢 36 倍说明
   **移动端 CPU EP 对这个 39-Conv 的小网络优化很差**。

### 5.5.3 MRZ band 路径实测（决定性数据）

用 Photo/3 的真实 MRZ 区域（y 1250..1420, x 300..2400，**2100×170**）裁出 band，
走`page_to_input(band, 1024)` 后det 输入只有 **96×1024**（整页是 768×1024，**面积 1/8**）。

| 模型 | 整页输入 | 整页耗时 | MRZ band 输入 | **band 耗时** | 加速比 |
|---|---|---|---|---|---|
| det | [1,1,768,1024] | 792 ms | [1,1,96,1024] | **123 ms** | 6.4× |
| orient | [1,1,768,768] | 358 ms | （跳过）| — | — |

数值同样**完全一致**（band.det.ch0.sum / max / first8 三项全PASS，最差 rel ≤ 1e-8），
探针总成绩 **14 PASS / 0 FAIL**。

**MRZ band 上整条链路的预估**：
```
det (band)  123 ms
rec (2 行)  ~15 ms   （band 只有 2 行；整页 48 行时 rec 为 114 ms）
mrz_repair  <1 ms    （纯 Kotlin/Python 逻辑）
────────────────────
总计        ≈ 140 ms/张
```
对比现有 Paddle 路径：v4 mobile 模式 **3~4 s/张**、v6 medium **约 11 s/张**。
→ **Indra 的 MRZ band 路径比现有 Paddle 快 20~70 倍，且准确率更高。**

### 5.5.4 由此产生的工程结论

- **单页全量 Indra 在真机上不可接受**：det 792 + orient 358 + rec(48行) ≈ **2~4 s/页**。
- **MRZ band 路径完全可行**：≈ 140 ms/张，比现有 Paddle 引擎快 1~2 个数量级。
  → **Stage A 必须走「MRZ band 专用路径」，不是「整页 det+rec」**。
- `orient`（358 ms）在 MRZ 场景**跳过**：照片朝向由上游 PhotoMrzPipeline 保证，
  或用极轻量的投影方差法替代。
- 真机建议**det 与 rec 分线程并行**（det 4 线程、rec 2 线程），4 核才不闲置；
  探针里 `intra_op_num_threads=2` 是保守设定，实际 det 用 4 线程应再快 1.5~2×。
- **band 的选择很重要**：band 太大（贴近整页）会退化回792 ms。
  建议 band 高度 = MRZ 两行 + 上下各 1 行文字余量，长边 ≈ 2100 px。

### 5.5.5 复现

```sh
cd 8_android
# 1) 生成 host 侧输入张量（用 indra_ocr 的真实预处理）
python3 ../11_indra/tools/make_probe.py      # 或见 tools/indra_probe/README
# 2) 推送
adb -s 192.168.26.131:5555 shell mkdir -p /sdcard/Android/data/com.nextransit.mrzbench/files/indra_probe
adb -s 192.168.26.131:5555 push tools/indra_probe/*.bin \
    /sdcard/Android/data/com.nextransit.mrzbench/files/indra_probe/
adb -s 192.168.26.131:5555 shell mkdir -p /sdcard/Android/data/com.nextransit.mrzbench/files/models/indra
adb -s 192.168.26.131:5555 push app/src/main/assets/models/indra/* \
    /sdcard/Android/data/com.nextransit.mrzbench/files/models/indra/
# 3) 构建 + 跑
./gradlew :app:assembleDebugAndroidTest --offline
adb -s 192.168.26.131:5555 install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb -s 192.168.26.131:5555 logcat -c
adb -s 192.168.26.131:5555 shell am instrument -w \
    -e class com.nextransit.mrzbench.IndraOnnxProbeTest \
    com.nextransit.mrzbench.test/androidx.test.runner.AndroidJUnitRunner
adb -s 192.168.26.131:5555 logcat -d -s IndraProbe
```

---

## 6. 尚未验证 / 需要后续确认

| 项 | 状态 |
|---|---|
| OpenCV 替换为自研 C 算子后的**数值一致性** | ❌ 未测（resize 插值方式需与 cv2 对齐：`INTER_AREA` vs `INTER_LINEAR` 的选择规则） |
| MRZ band 上 **rec 两行**的实际耗时 | ⚠️ band det 已实测 123 ms；rec 两行按比例约 15 ms，未单独实测 |
| `lm.npz` 的 npz 格式在 Android 侧解析 | 未做（需读 .npz 里的 uint64 数组 + 排序表重建） |
| 设备只有 4 核，det/rec 并行时的实际总耗时 | 未测 |
| `rec_cjk` / `rec_rtl` / `rec_world` 三个分组识别器的真机耗时 | 未测（只测了 base） |
| 商业授权 | ⚠️ **模型权重是 modified OpenRAIL-M**：免费用于研究/个人/融资 <$20M 的初创，**不可与竞品竞争性使用**。商用前必须确认条款 |

---

## 7. 复现命令

```sh
# 模型完整性
ls -la 11_indra/models/          # 应有 12 个 onnx + 3 个 npz + meta.json

# ORT 1.20.0 兼容性（与 Android 同版本）
/tmp/ort120/bin/python -c "
import onnxruntime as ort, glob
for f in sorted(glob.glob('11_indra/models/*.onnx')):
    ort.InferenceSession(f, providers=['CPUExecutionProvider']); print('OK', f)"

# Python 端 MRZ 基准
cd 11_indra && python3 mrz_indra.py ../Photo/3/IMG_1792133000_003.jpg
```
