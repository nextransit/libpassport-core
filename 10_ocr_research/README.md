# 10_ocr_research

护照机实拍照片（2592x1944，Photo/7 例外 640x480）端到端 OCR 调研。

## 目录
- `src/probe_backends.py` — 4 后端 × 多输入形态探针（第一轮基线）
- `src/host_pipeline.py` — Photo → restore → crop → CNN/decode 端到端（本轮）
- `archive/host_pipeline_results.json` — 6 张实测结果
- `archive/probe_results.json` — 第一轮 4 后端探针输出
- `docs/RESEARCH.md` — 根因分析与推荐路径
- `docs/TASKS.md` — 任务 1+2+3+4 状态

## 当前轮闭环产物（2026-10-29）
- `archive/photo_auto_gt_all.json` — **统一真值**（115 条：B=36 / C=70 / no_record=9），唯一权威
- `photo_direct_bench_final.json` — 基线权重全量批测（115 张）
- `photo_direct_bench_cnn_candidate.json` / `photo_direct_bench_crnn_candidate.json` — 候选权重批测
- `gt_review_loop.md` / `gt_review_loop.xlsx` — GT vs PRED 逐引擎对比 + **验收汇总**（速率 + 通过率 + 门槛 + 退化清单）

### 评测口径（2026-10-29 起）
1. **不做 fallback**：`allow_tesseract_fallback=False`，引擎失败就是失败。Tesseract 只作为独立引擎参与评估。
2. **预加载 + 常驻引擎**（`--resident`）：模型先加载完再计时，测的是单次完整识别时间。
   - C 工具走 `--serve` 常驻模式（`tests/mrz_engine_pool.py`）；CRNN ONNX 进程内常驻。
   - 实测：单次进程 8 ms → 常驻 4 ms；CRNN 137 ms → 15 ms。
3. **速率三口径**：产品路径（全引擎并发）/ 单引擎 / 批测墙钟吞吐（仅参考）。

### 基线验收结果（`gt_review_loop.xlsx` → 验收汇总 + 全算法验收）
| 指标 | 数值 | 门槛 | 判定 |
|---|---|---|---|
| 官方 PASS | 105 / 115（91.3%） | ≥ 105 | PASS |
| raw fuse decode | 84 | ≥ 84 | PASS |
| 产品路径速率 | **3.32 张/秒**（平均 301.4 ms / 中位 235.1 ms / P95 591.6 ms） | ≥ 2.0 | PASS |
| 退化张数 | 0 | = 0 | PASS |

去fallback + 常驻化 前后对比：官方 PASS 105→105，raw fuse 84→84，各引擎 decode 数全部不变，
产品路径 385.9 ms → **301.4 ms**（-22%）。速率 2.59 → **3.32 张/秒**。

单引擎实测提速（常驻 vs 单次进程）：
- C 工具（traditional/cnn/fuse）：8 ms → 4 ms
- CRNN（ONNX 进程内常驻）：137 ms → 18 ms
- Paddle（ONNX 进程内常驻）：258 ms → 219 ms（官方路径口径）

### 全算法验收（8 个引擎全部评估，见 xlsx「全算法验收」sheet）
| engine | decode 率 | GT 精确率 | 平均耗时 | 速率(张/秒) |
|---|---|---|---|---|
| traditional | 0% | 0% | 14.3 ms | 69.8 |
| cnn | 58.3% | 0% | 17.1 ms | 58.6 |
| crnn | 11.3% | 22.2% | 17.8 ms | 56.2 |
| fuse | 73.0% | 0% | 17.6 ms | 57.0 |
| tesseract | 84.3% | 22.2% | 192.5 ms | 5.2 |
| paddle | 86.1% | 33.3% | 111.6 ms | 9.0 |
| indra | 76.5% | 22.2% | 46.9 ms | 21.3 |
| indra_fullpage | 89.6% | 11.1% | 167.3 ms | 6.0 |

### 候选权重结论
- **CRNN 候选（统一真值训练）**：真实生效后 raw fuse 84 → **83**（退化 1 张），未达门槛，**已回滚**。
- **CNN 候选**：训练产物只写 `DEFAULT_*` 符号，而产线启用 `MRZ_OCR_LINE_MODELS=1` 只读
  `cnn_weights_line1.h` / `cnn_weights_line2.h` 的 `LINE1_*` / `LINE2_*` 符号 →
  候选对产线**零影响**。要评估 CNN 候选必须先产出 line1/line2 双头文件。

## 运行
```sh
cd 10_ocr_research/src
python3 host_pipeline.py   # 端到端（推荐）
python3 probe_backends.py  # 4 后端横向（基线对比）
```

## 结论速览
1. 半区溢出已修复：`mrz_ocr_cnn.c` 4MB → 32MB，全图不再 LOAD error。
2. **倒影分离**：新 `find_mrz_pair`（严格双行匹配）天然剔除反射，6/6 定位干净。
3. **端到端 5/6 decode 出真实字符**（conf 53-79），p50 ≈ 9.3ms/张。
4. 字符 conf 偏低 → 下一步 CNN 输入端上采样 ×2~3（行高对齐训练分布 36-72px）。

## 设备状态
- Android 192.168.26.123 当前不在线（TCP 5555 CLOSED）。设备恢复后：
  - host_pipeline → 8_android (Kotlin/JNI)
  - 上 6_mrz_ocr 的 NEON 优化（vmulq_f32 + vaddq_f32）
  - host vs on-device 耗时对比 + byte-exact 一致性门禁
