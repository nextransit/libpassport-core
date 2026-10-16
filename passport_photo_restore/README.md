# passport_photo_restore — 护照机拍照图：固定矫正结构先验 + 去倒影算法

独立自包含目录（仅依赖 OpenCV + NumPy，不依赖仓库 tests/ 或 6_mrz_ocr）。

## 结构
```
passport_photo_restore/
├── README.md
├── src/
│   ├── restore.py        # 核心算法：结构检测 + MRZ 定位 + 去倒影 crop + 单应矫正
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
1. **结构先验**：`detect_structure(g)` — solid 黑带（护照机固定取景 y~725-770）
   下方连续 text 段 = MRZ 两行。
2. **去倒影**：MRZ 带与下一段 text gap>15px 判定进入倒影区 → crop 天然不含倒影。
3. **单应矫正**：`rectify()` — MRZ 带四边形 → 标准 44格×2行 矩形（LANCZOS4）。

## 评估结果（10-16）
- MRZ 结构定位 6/6、solid 黑带 6/6，单图定位 1-2ms。
- OCR 端到端 decode 0/6（受照片 MRZ 单行仅 10-25px 与训练分布 36-72px 差距、透视限制）——
  详见 `docs/OCR_STATUS.md`。
