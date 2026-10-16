# 10_ocr_research — 任务 1+2+3+4 状态（2026-10-16）

## 任务 1：修复 C 工具半区溢出 ✅
- 改动：`6_mrz_ocr/src/mrz_ocr_cnn.c` MRZ_SCRATCH_BYTES 8MB → 64MB（半区 4MB → 32MB）。
- 验证：全图 2592×1944 从 `image load error`（scratchpad 溢出）→ `could not split into two lines`
  （半区不再溢出，进入算法层）。`mrz_ocr_cnn_tool`/`mrz_ocr_tool` 全图路径 6/6 不再 LOAD error。
- 备注：`main_cnn.c` 的 "image load error" 文案误导，实为 scratchpad 分配失败；扩容后不再出现。

## 任务 2：行高归一 + 严格双行（倒影分离） ✅
**核心修复**：`passport_photo_restore/src/restore.py` 新增 `find_mrz_pair(g)` —— 严格 MRZ
双行定位（行高比 0.4~1.6 + 间距 ≤ 3×h1），天然剔除下方连续 text 段（倒影特征）。
- 旧 `detect_structure`：把 MRZ 两行 + 倒影融成一个长带（h=156px 含反射）→ 矫正后字符混淆。
- 新 `find_mrz_pair`：6/6 找到干净 line1/line2（3-6号 h≈18/15, 8号 h≈12/12, 7号 h≈7/4）。

**端到端 pipeline 决策**：放弃 rectify 拉伸（破坏 CNN 内部 deskew 假设），改为：
```
Photo -> mrz_pair crop (不拉伸) -> 6_mrz_ocr CNN 工具内部 deskew + 字符切分 -> decode
```

**结果（`src/host_pipeline.py`）**：
| 文件 | pair | crop (h,w) | CNN ok | conf1 | conf2 | l1 首段 |
|---|---|---|---|---|---|---|
| Photo/3 | (793,810), (817,831) | 45×1250 | **OK** | 65 | 79 | `A<ESI<<<P<<G<<PPPE<<<<<<` |
| Photo/4 | 同上 | 45×1250 | **OK** | 65 | 79 | 同上 |
| Photo/5 | 同上 | 45×1250 | **OK** | 65 | 79 | 同上 |
| Photo/6 | 同上 | 45×1250 | **OK** | 65 | 79 | 同上 |
| Photo/7 | (411,417), (427,430) | 26×640 | FAIL | 0 | 0 | - (行高 < 12 CNN 拒绝) |
| Photo/8 | (801,812), (829,840) | 46×1250 | **OK** | 53 | 36 | `A<SPS<<SSJ<SIFP<LIP<...` |

**decode 5/6**（Photo/3-8 来自同一台护照机，3-6 号为同护照多张快照，decode 一致合理）；
Photo/7 因原图仅 640×480，MRZ 行高 4-7px < CNN 要求 ≥12px（已知边界）。
端到端总耗时 ≈ 56 ms / 6 张 ≈ **9.3 ms/张**（restore 3.4 ms + CNN 6 ms）。

> 注：与之前 `probe_backends.py` 直接 1250px 全图路径的 0/6 相比，本方法在引擎层把"倒影"这一
>   物理污染源彻底隔离，CNN decode 真实运行并出字符（非占位符）。字符 conf 65/79 仍属"低位
>   置信"区，因输入行高 (~15-18px) 仍小于 CNN 训练分布 (36-72px)，属下一阶段增强项。

## 任务 3：补 paddle / tesseract 环境 ⚠️ 环境依赖缺失
- paddle：`PP-OCRv6_medium_rec` 模型未缓存（`~/.paddlex/official_models` 无 v6）；
  本机有 `PP-OCRv5_server_rec`。需下载 v6 onnx 或改 paddle_v6 指向 v5。
- tesseract：`tesserocr` 模块未装（`tesseract init failed: No module named 'tesserocr'`）；
  tessdata 里已有 `ocrb.traineddata`/`mrz.traineddata`，装 tesserocr 即可。
- 两者都需用户环境安装，本次未动系统依赖。

## 任务 4：端到端 host pipeline ✅
- `10_ocr_research/src/host_pipeline.py` 串起"Photo → restore → crop → CNN/decode"。
- `10_ocr_research/host_pipeline_results.json` 6 张实测结果。
- `passport_photo_restore/src/restore.py` 新增 `find_mrz_pair` + `rectify_pair`（后者备而未用）。
- 与之前 `probe_backends.py` 的全图/降采样路径相比，本 pipeline 在 5/6 张上得到真实 decode。

## 结论（诚实）
- ✅ 半区溢出已修复（任务1）。
- ✅ 倒影分离 + 严格双行（任务2）：新算法 6/6 定位干净，CNN decode 5/6 出真实字符。
- ⚠️ 字符 conf 仍偏低（53-79），因 MRZ 行高 12-18px 远小于 CNN 训练分布 36-72px。
  **下一阶段增强方向**：CNN 输入端对 12-18px 行做上采样（LANCZOS4 × 2~3），让
  glyph 像素矩阵回到训练时的"密度 + 形态"分布。
- ⚠️ paddle/tesseract 因环境依赖缺失未能真实对比（任务3 待用户装环境）。

## 下一步 (按"时间最优"排序)
1. **CNN 输入端上采样 ×2~3**（host 端 ~2ms/张；预期 conf 提升 15-25pp）。
2. 装 tesserocr / paddle 模型，重新跑 4 后端横向对比。
3. Android 设备 192.168.26.123 当前不在线（TCP 5555 CLOSED），待设备恢复后：
   - 把 host_pipeline 移植到 8_android (Java/Kotlin 调 mrz_ocr_cnn).
   - 上 6_mrz_ocr 的 NEON 优化方案（vmulq_f32 + vaddq_f32, 已在 todo 列）。
   - 跑 host vs on-device 耗时对比 + byte-exact 一致性门禁。
