# 10_ocr_research — 护照机照片 OCR 调研

> 语料：`Photo/{3..8}` 6 张护照机实拍（2592x1944，7 号 640x480）。
> 目标：定位"现有 4 套 OCR 后端对护照机照片 decode 全 0/6"的根因，并给出可落地路径。
> 结论先行：**根因不是 OCR 模型本身，而是「全图直喂 + C 工具半区溢出 + MRZ 行高不匹配」**。

## 1. 探针设计

`src/probe_backends.py` 对每张图跑 6 种输入 × 4 后端：

| 后端 | 工具 | 备注 |
|---|---|---|
| cnn | `6_mrz_ocr/build/mrz_ocr_cnn_tool` | 纯 C，PPM 输入 |
| template | `6_mrz_ocr/build/mrz_ocr_tool` | 纯 C，OCR-B 模板 |
| tesseract | `6_2_Tesseract/tesseract_tool.py` | ocrb 微调模型 |
| paddle | `6_1_PaddleOCR/paddle_v6.py` | PP-OCRv6 medium rec |

输入形态：原图 PPM / 降采样 PPM / 透视矫正+去倒影 crop（`passport_photo_restore`）。

## 2. 探针结果（2026-10-16）

| 后端 | 原图 2592 | 降采样 1250 | 矫正 crop |
|---|---|---|---|
| cnn | LOAD error（半区溢出）| LOAD error（半区溢出）| **2/6 出 line1/line2**（7/8 号）|
| template | could not split | could not split | 未测 |
| tesseract | tesseract init error | 同左 | 未测 |
| paddle | 模型未缓存 | 同左 | 同左 |

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
3. **透视**：行不水平、左右行高不等 → 需单应矫正后再放大。

## 4. 推荐路径（时间最优 + 复用现有代码）

```
Photo (2592x1944)
  -> passport_photo_restore 结构先验定位 MRZ 带 (y~781-831)   ~2ms
  -> 单应矫正到 44格x2行 矩形 (616x104)                        ~3ms
  -> 行高放大到 48px（~3-4x LANCZOS）                          ~2ms
  -> 二值化/CLAHE 增强                                         ~1ms
  -> 6_mrz_ocr CNN 或 Tesseract ocrb                           ~5-10ms
  -> 1_mrz_decode Mod-10 校验                                  ~0.5ms
  预期 p50 < 20ms；decode 从 0/6 起步，逐张验证提升。
```

需修：`mrz_ocr_cnn.c` 半区按需扩容（或调用方先 crop MRZ 再入 CNN），
且 `main_cnn.c` 的 "image load error" 文案是误导（实为 scratchpad 分配失败）。

## 5. 诚实边界

- 本次探针确认了**根因**（半区溢出 + 行高不匹配 + 透视），并验证**矫正 crop 能让
  CNN 出双行**（2/6）。
- decode 仍 0/6：矫正后字符仍粗/歪，需行高精确归一 + 增强后才进入识别评估。
- paddle/tesseract 未拿到真实数据（模型缓存/init 问题），是下一阶段首补项。
