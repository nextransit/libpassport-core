# 10_ocr_research

护照机实拍照片（2592x1944）端到端 OCR 调研。

## 目录
- `src/probe_backends.py` — 4 后端 × 多输入形态探针
- `probe_results.json` — 最近一次探针输出（运行时生成）
- `docs/RESEARCH.md` — 根因分析与推荐路径

## 运行
```sh
cd 10_ocr_research/src
python3 probe_backends.py        # 需要 passport_photo_restore/src 在 sys.path（已处理）
```

## 结论速览
1. 全图直喂 C 工具必失败：`mrz_ocr_cnn.c` 4MB 半区溢出（gray+bin+band>4MB）。
2. 矫正+去倒影 crop 后 CNN 能出双行（7/8 号），但行高/透视仍需归一。
3. decode 0/6 根因 = 半区溢出 + 行高不匹配 + 透视，非 OCR 模型本身。
