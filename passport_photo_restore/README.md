# passport_photo_restore — 护照机拍照图：固定矫正结构先验 + 去倒影算法

独立自包含目录（仅依赖 OpenCV + NumPy，不依赖仓库 tests/ 或 6_mrz_ocr）。

## 结构
```
passport_photo_restore/
├── README.md
├── src/
│   ├── restore.py        # 核心算法：结构检测 + MRZ 定位 + 去倒影 crop
│   ├── eval.py           # 评估脚本：对 Photo/ 图集输出 EVAL.md
│   └── EVAL.md           # 最近一次评估结果
└── docs/
    ├── RESEARCH.md       # 问题量化 + 方案选型（12 候选）
    ├── OCR_STATUS.md     # OCR 端到端诚实现状 + 卡点
    └── README.md         # 研究包说明（源自 docs/passport_photo_restore）
```

## 运行
```sh
cd passport_photo_restore/src
python3 eval.py ../../Photo            # 结构定位评估 -> src/EVAL.md
python3 -c "from restore import restore; r=restore('../../Photo/3/IMG_1792133000_003.jpg'); print(r['mrz'])"
```

## 核心算法（src/restore.py）
1. **结构先验**：`detect_structure(g)` — 识别护照机固定取景的最后一条 solid 黑带，
   并取其上方紧邻的两段等高 text 作为 MRZ，避免把页面下方倒影当成真实 MRZ。
2. **去倒影**：只保留到真实 MRZ 第二行末尾，裁掉其后的黑色边界、模糊 MRZ
   和其他数据倒影。
3. **MRZ 区**：`crop_pair_clean()` — 擦掉 line1/line2 紧邻的反射行，保留
   自然形态不拉伸。**这是识别真正吃的那条带**（OCR 输入）。
4. **单应矫正**：`rectify_pair()` — line1/line2 分别单应到标准 44格×2行矩形（LANCZOS4）。
   ⚠️ 这是**展示/调研**用的几何变换，**不是**识别路径：它的页面四边形估计在整体
   偏亮的照片上会把整幅图当成页面，矫正后整张照片被压成 44 格×2 行。实测把矫正图
   喂给同一个 `mrz_ocr_cnn_tool`，`assets/pic` 上 ICAO 通过 2/72，而自然 crop 是
   3/72。native 识别流水线自己已经做了 deskew + 44 格 pitch-grid + 逐格面积重采样。
5. **自动识别**：GUI 在原图分辨率逐行调用 Tesseract OCR-B，提取 44 字符候选，
   用 TD3 checksum 修复后必须通过 `mrz_tool decode` 才显示成功。

## 评估结果（10-17）
- MRZ 双行定位 6/6、solid 黑带 6/6；去影后只保留到真实 line2 末尾。
- Photo/3-8 自动 OCR 与 `mrz_tool decode` 全部通过，回归测试覆盖 MRZ 区上下两行。
