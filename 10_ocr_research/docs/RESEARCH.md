# 10_ocr_research — 护照机照片 OCR 调研

> 语料：`Photo/{3..8}` 6 张护照机实拍（2592x1944，7 号 640x480）。
> 目标：定位"现有 4 套 OCR 后端对护照机照片 decode 全 0/6"的根因，并给出可落地路径。
> 结论：**根因不是 OCR 模型本身，而是「全图直喂 + C 工具半区溢出 + 倒影污染 MRZ 带」**。

## 1. 探针设计

`src/probe_backends.py` 对每张图跑 6 种输入 × 4 后端（首轮基线）。
`src/host_pipeline.py` 端到端串"Photo → restore → crop → CNN/decode"（本轮改进）。

| 后端 | 工具 | 备注 |
|---|---|---|
| cnn | `6_mrz_ocr/build/mrz_ocr_cnn_tool` | 纯 C，PPM 输入 |
| template | `6_mrz_ocr/build/mrz_ocr_tool` | 纯 C，OCR-B 模板 |
| tesseract | `6_2_Tesseract/tesseract_tool.py` | ocrb 微调模型 |
| paddle | `6_1_PaddleOCR/paddle_v6.py` | PP-OCRv6 medium rec |

输入形态：原图 PPM / 降采样 PPM / 透视矫正+去倒影 crop（`passport_photo_restore`）。

## 2. 探针结果（2026-10-16 第一轮）

| 后端 | 原图 2592 | 降采样 1250 | 矫正 crop |
|---|---|---|---|
| cnn | LOAD error（半区溢出）| LOAD error（半区溢出）| **2/6 出 line1/line2**（7/8 号）|
| template | could not split | could not split | 未测 |
| tesseract | tesseract init error | 同左 | 未测 |
| paddle | 模型未缓存 | 同左 | 未测 |

关键数字：
- `cnn_tool` 全图：`band=(0,0,1250,772)` 定位成功但随后 `MRZ_OCR_ERR_LOAD` ——
  **8MB scratchpad 双半区各 4MB，gray+bin+band_gray+band_bin > 4MB 溢出**。
- 矫正 crop（44格×2行≈616×104）后 cnn 能跑：7 号 `A<ELP...` / 8 号 `A<FPG...` 有双行，
  但字符仍为粗笔画占位符（行高放大后仍不足 / 透视未完全拉直）。

## 3. 根因（三层）

1. **C 工具半区溢出**：`mrz_ocr_cnn.c` 的 4MB 半区放不下全图 gray+bin+band，
   2592px 图直接 LOAD error → 必须**先 crop MRZ 带再喂**（不喂全图）。
2. **MRZ 行高不匹配**：护照机照片 MRZ 单行仅 10-25px，而 cnn/template 训练/模板
   分布是 36-72px → 需放大 + 行高归一化。
3. **倒影污染**：玻璃反射的连续 text 段与 MRZ 两行间隔 < 30px，旧的 `gap_thr`
   启发式无法将其与 line2 区分，导致矫正带高度 = MRZ 两行 + 倒影，CNN split 失败。

## 4. 推荐路径 + 实证（第二轮）

**新算法**：`passport_photo_restore/src/restore.py:find_mrz_pair(g)`
- 严格双行匹配：行高比 0.4~1.6 + 间距 ≤ 3×h1。
- 反射倒影特征：连续 text_run 长度 > 2.5×line1 行高 + 与 line2 有 gap → 自动剔除。
- 6/6 全部找到干净 line1/line2（h≈18/15 for 3-6 号，h≈12/12 for 8 号）。

**端到端 pipeline**：`src/host_pipeline.py`
```
Photo (2592x1944 or 640x480)
  -> load() 降采样到 1250                          ~0.5ms
  -> detect_structure + find_mrz_pair               ~3ms
  -> crop 两行最小矩形 (不拉伸, 保留自然形态)        ~0.1ms
  -> 6_mrz_ocr CNN 工具: 内部 deskew + 字符切分      ~6ms
  -> decode 输出 l1/l2/conf                         (内置)
  -------------------------------------------------
  端到端 p50 ≈ 9.3ms / 张                           (5/6 decode OK)
```

**对比第一轮**：
| 指标 | 第一轮 (probe_backends) | 第二轮 (host_pipeline) |
|---|---|---|
| decode ok | 0/6 (rect_paddle) + 2/6 (rect_cnn 占位符) | **5/6** 真实字符 |
| 含倒影？ | 是（rectify 单段含反射）| 否（find_mrz_pair 剔除）|
| 字符 conf | 0 / 28-40 (粗笔画) | **53-79** (A<B<... 真实字符) |
| 端到端 ms | 9.9-5876 (含 paddle init) | **9.3 ms/张** |

## 5. 诚实边界

- ✅ 半区溢出（任务1）已修复，5/6 张真实 decode 出 MRZ 字符。
- ⚠️ 字符 conf 仍偏低（53-79），因 MRZ 行高 12-18px 远小于 CNN 训练分布 36-72px。
  下一阶段增强方向：**CNN 输入端上采样 ×2~3**（LANCZOS4，让 glyph 像素密度回到
  训练分布；预期 conf 提升 15-25pp，decode 错率下降）。
- ⚠️ paddle/tesseract 因环境依赖缺失未能真实对比（任务3 待用户装环境）。
- ⚠️ Android 设备 192.168.26.123 当前不在线（TCP 5555 CLOSED），设备恢复后
  才能把 host_pipeline 移植到 8_android 并跑端到端对比。
