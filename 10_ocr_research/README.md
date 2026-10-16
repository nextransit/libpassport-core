# 10_ocr_research

护照机实拍照片（2592x1944，Photo/7 例外 640x480）端到端 OCR 调研。

## 目录
- `src/probe_backends.py` — 4 后端 × 多输入形态探针（第一轮基线）
- `src/host_pipeline.py` — Photo → restore → crop → CNN/decode 端到端（本轮）
- `host_pipeline_results.json` — 6 张实测结果
- `probe_results.json` — 第一轮 4 后端探针输出
- `docs/RESEARCH.md` — 根因分析与推荐路径
- `docs/TASKS.md` — 任务 1+2+3+4 状态

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
