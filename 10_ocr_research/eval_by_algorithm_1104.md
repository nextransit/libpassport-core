# 单算法评估（cnn / crnn / fuse）+ l1 / l2 / exact

生成：2026-11-04（末次更新：OCR-B + 抗锯齿(AA) 语料 + 语料域训练 + CRNN 用新语料重训之后）。
GT = `10_ocr_research/gt_review_loop.xlsx`（sheet「GT对比」）→ 由 `tests/build_gt_from_xlsx.py` 派生出官方 GT JSON。

口径：只统计 tier A/B 且 GT 长度为 44 字符的行（tier C = 截断/单行，不计分）；l1 exact = line1 与 GT 完全一致；l2 exact = line2 完全一致；exact = 两行同时完全一致；char = 逐字符准确率。

权重状态（当前 src/）：

| 组件 | 训练数据 | 说明 |
|---|---|---|
| passport CNN line1/line2 | 真实照片切片 + **语料切片**（`--synth-extra-npz`） | 照片半边 real-frac 0.5，语料切片并入合成半边 → 语料变成 in-distribution |
| pic CNN line1/line2 | 同上（真实照片切片 + 语料切片） | 照片半边 real-frac 0.70（红线） |
| CRNN | `data/corpus_train`（OCR-B + AA，3000 张）导特征 + 照片 CRNN 特征 | 必须来自与推理相同的 binary（CNN 前端已换） |

## A. 冻结 GT + 与训练同源的 band（`6_mrz_ocr/tools/eval_engines.py`）

每个引擎各自跑同一套 band 候选并按「line2 校验位 + 置信度」自选（与训练切片同源同 framing）。

### A1. 护照集 `tests/gt_photo_passport_official.json`（44 条，mrz_ocr_cnn_tool）

| engine | l1 exact | l1 char | l2 exact | l2 char | exact (两行) | decode_ok |
|---|---|---|---|---|---|---|
| cnn | 39/41 | 98.61% | 38/41 | 99.17% | 37/41 | 43/44 |
| **crnn** | **41/41** | **100.00%** | **39/41** | **99.56%** | **39/41** | 43/44 |
| fuse | 39/41 | 98.73% | 38/41 | 99.11% | 37/41 | 43/44 |

> **CRNN 单独用现在是最强的一档**：换 OCR-B+AA 语料重训后 l1 逐字 22.17% → **100.00%**（41/41 全 exact）、exact 0/41 → 39/41。
> 旧 CRNN 头是拿 legacy 简化点阵语料训的，与真实照片字形差太远；这是「语料字体/渲染与训练分布一致」带来的最大收益。

### A2. assets/pic 集 `tests/gt_assets_pic_official.json`（64 条，mrz_ocr_cnn_tool_pic）

| engine | l1 exact | l1 char | l2 exact | l2 char | exact (两行) | decode_ok |
|---|---|---|---|---|---|---|
| cnn | 57/63 | 97.15% | 54/59 | 98.19% | 51/59 | 64/64 |
| crnn | 57/63 | 97.15% | 54/59 | 98.19% | 51/59 | 64/64 |
| fuse | 57/63 | 97.15% | 54/59 | 98.19% | 51/59 | 64/64 |

> pic 构建（MRZ_WEIGHTS_SET_PIC）**不含 CRNN 权重**（`crnn_weights_available()` 返回 0：CRNN 头绑在同一个权重集的 CNN 特征上训练），因此 crnn / fuse 在 pic 上退化为 cnn，上表三行数值相同。想让 pic 用上 fuse/crnn 需给 pic 权重集单独训 CRNN 头。

## B. 官方产品路径的单引擎逐条读数（bench raw 通道，114 张）

数据源 `10_ocr_research/photo_direct_bench_corpusfix2_1104.json` 的 `records[].raw[method]`，GT 用同一份 xlsx。band = 产品路径 native band（restore 文字行 pair + pad），工具 = GUI 当前权重选择（本次为 passport 工具；`ocr_weights_pic` 默认 False，故 assets/pic 也走护照权重）。

```
bench = photo_direct_bench_corpusfix2_1104.json   gt = gt_review_loop.xlsx (GT对比)
engine                 l1 exact      l1 char     l2 exact      l2 char   exact(both)    decode_ok
-------------------------------------------------------------------------------------------------
cnn                   79/110          82.79%    62/106          78.07%     59/106       82/114
crnn                  88/110          97.13%    79/106          94.30%     74/106      104/114
fuse                  91/110          96.76%    73/106          92.28%     68/106       96/114
indra                 55/110          82.29%    48/106          81.22%     33/106       98/114
indra_fullpage        38/110          70.29%    31/106          66.47%     16/106      103/114
paddle               108/110          99.94%    76/106          96.48%     75/106      103/114
tesseract             84/110          94.44%    54/106          89.24%     48/106      104/114
traditional            0/110          60.21%     0/106          28.82%      0/106        0/114
```

assets/pic 子集（64 张，同一批工具=护照权重）：

| engine | l1 exact | l1 char | l2 exact | l2 char | exact (both) | decode_ok |
|---|---|---|---|---|---|---|
| cnn | 38/63 | 74.17% | 28/59 | 67.80% | 26/59 | 37/64 |
| crnn | 47/63 | 95.53% | 42/59 | 95.22% | 37/59 | 57/64 |
| fuse | 49/63 | 95.74% | 39/59 | 91.64% | 35/59 | 51/64 |
| paddle | 63/63 | 100.00% | 37/59 | 96.22% | 37/59 | 57/64 |
| tesseract | 55/63 | 94.91% | 49/59 | 94.34% | 45/59 | 58/64 |

官方多引擎仲裁：**PASS 99/114、SUSPECT 0、REJECT 0、verified 113、ms mean/median/p95 = 307.3/229.5/672.3（≈3.25 张/秒）**。
official 通道：cnn 114/82、**crnn 114/105**、fuse 111/97、tesseract 114/104、paddle 111/103、indra 111/98。

## C. 渲染语料（`data/corpus_eval`，OCR-B + AA，1380 张）

`6_mrz_ocr/tools/bench_crnn.py --corpus data/corpus_eval`

| engine | l1 | l2 | full(两行 exact) | 校验通过率 | p50/p95 (ms) |
|---|---|---|---|---|---|
| cnn | 91.69% | 90.10% | **13.55%** | 87.54% | 10.6/18.4 |
| crnn | **92.28%** | **95.16%** | **18.12%** | **95.65%** | 10.0/15.5 |
| fuse | 91.83% | 90.45% | 11.38% | 87.25% | 10.5/18.3 |

前 100 条（GUI 里 0001-0100 的默认范围）：

| engine | l1 | l2 | full | 校验通过率 |
|---|---|---|---|---|
| cnn | 95.36%（原 87.95） | 95.89%（原 74.02） | **61.00%**（原 0.00） | 93%（原 69） |
| crnn | 95.23% | 98.11% | 57.00% | 99% |
| fuse | 95.27% | 96.05% | 56.00% | 94% |

按 profile（前 100）：clean 档 cnn 95.59/98.13/full 71.43/cksum 97.62；realistic 档 cnn 94.18/84.09/full 6.25/cksum 68.75（上一版 realistic l2 只有 59.80 → **+24pp**，cksum 43.75 → 68.75）。
全量 1380 张：clean 档 cnn 91.91/97.79/full 18.89；realistic 档 cnn 91.44/81.70/full 7.73/cksum 76.06。

## D. 结论
1. **CRNN 是被语料「救活」的引擎**：换 OCR-B+AA 语料 + 语料域训练后，真实照片 l1 逐字 22.17%→100%（41/41 全 exact）、both 0/41→39/41，产品路径 decode 22/114→104/114、l1 char 39.67%→97.13%，与 paddle 同档。
2. **CNN 的瓶颈是域外，不是模型容量**：同一语料上 CRNN 92/95/fuse-full 18%，而 CNN 之前只有 87.9/74.0/full 0；把语料切片（含弱标记难样本）并入训练后语料 full 0→13.55%（前 100 条 0→**61%**），真实照片 both 不降（passport 37/41），官方 PASS 97→**99/114**。
3. **pic 上没有 fuse/crnn**：pic 权重集不含 CRNN 头（`crnn_weights_available()` 在 MRZ_WEIGHTS_SET_PIC 分支返回 0），需要单独训。
4. **单引擎最强是 paddle**（l1 99.94%、exact 75/106）；C 系里 CRNN 已与 paddle 持平（both 74/106 vs 75/106），fuse 68/106。
5. 剩余瓶颈：① 语料 realistic 档（rot/blur/illum 强退化）cnn l2 81.70% / full 7.73%；② 产品路径 band 仍是 restore pair + pad 8..25（assets/pic 上 cnn 74.17% vs 候选自选 97.15%）→ 改用 `mrz_band` 候选自选收益最大；③ `mrz_golden` scale=3 E→F（8×12 字模在 3× 下的分辨率上限，HEAD 预存失败）。
