# 10_ocr_research — 任务 1+2+3+4 状态（2026-10-16）

## 任务 1：修复 C 工具半区溢出 ✅
- 改动：`6_mrz_ocr/src/mrz_ocr_cnn.c` MRZ_SCRATCH_BYTES 8MB → 64MB（半区 4MB → 32MB）。
- 验证：全图 2592×1944 从 `image load error`（scratchpad 溢出）→ `could not split into two lines`
  （半区不再溢出，进入算法层）。`mrz_ocr_cnn_tool`/`mrz_ocr_tool` 全图路径 6/6 不再 LOAD error。
- 备注：`main_cnn.c` 的 "image load error" 文案误导，实为 scratchpad 分配失败；扩容后不再出现。

## 任务 2：行高归一 + 矫正后重跑 ⚠️ 部分
- 矫正 crop（`passport_photo_restore/rectify` → 44格×2行 616×104，LANCZOS4）后：
  - 1250px 工作图：7/8 号 `mrz_ocr_cnn_tool` **ok=True 出双行**（l1/l2 均 44 字符）；
    3-6 号仍 could-not-split（倒影/黑带污染带内）。
  - **decode 仍 0/6**：矫正后字符仍粗/歪（护照机 MRZ 单行仅 10-25px < 训练 36-72px），
    CNN 输出粗笔画占位符（P/B/S/A）。
- 全分辨率（2592）定位：MRZ 单行 ~42-66px，接近训练分布，但带内混入倒影（如 8 号
  y1613-1744 含倒影 y1720-1744）导致 split 失败 → **需更精确的倒影分离**（垂直镜像
  判别，非 row 投影可解）。
- 下一步：倒影分离 = MRZ 两行与其下方镜像的相关性检测；或换 Tesseract ocrb / PaddleOCR
  低分辨率后端（见任务 3）。

## 任务 3：补 paddle / tesseract 环境 ⚠️ 环境依赖缺失
- paddle：`PP-OCRv6_medium_rec` 模型未缓存（`~/.paddlex/official_models` 无 v6）；
  本机有 `PP-OCRv5_server_rec`。需下载 v6 onnx 或改 paddle_v6 指向 v5。
- tesseract：`tesseract_tool.py` 依赖 `tesserocr` 模块（本机未装）→ init failed；
  tessdata 里已有 `ocrb.traineddata`/`mrz.traineddata`，装 tesserocr 即可。
- 两者都需用户环境安装（pip install tesserocr / 缓存 paddle 模型），本次未动系统依赖。

## 任务 4：提交 10_ocr_research
- 提交内容：`10_ocr_research/{README.md, docs/RESEARCH.md, docs/TASKS.md, src/probe_backends.py, probe_results.json}`
  + `6_mrz_ocr/src/mrz_ocr_cnn.c`（半区 64MB 修复） + `passport_photo_restore/src/restore.py`（detect 改进）。

## 结论（诚实）
- 半区溢出已修复（任务1 ✅），全图不再 LOAD error。
- 矫正 crop 能让 CNN 出双行（任务2 部分 ✅），但 decode 0/6 受"小 MRZ + 倒影 + 透视"物理限制。
- paddle/tesseract 因环境依赖缺失未能真实对比（任务3 待用户装环境）。
- 要突破 decode，推荐：精确倒影分离（镜像相关性）+ 行高严格归一到 48px + 换 ocrb/Paddle 后端，
  而非继续调 `6_mrz_ocr` 参数。
