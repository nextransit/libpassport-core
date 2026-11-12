# MRZ OCR benchmark

Corpus: 1380 images (660 realistic). Metrics: char accuracy over all lines; full = both lines exact; cksum = line2 passes all 5 ICAO check digits.

> **口径标识（2026-11-07 补）**：本文件当前数值对应 `6_mrz_ocr/metrics_history.md`
> 的 **2026-11-05 23:24** 一次 `data/corpus_eval` 跑批（旧 1380 张准二值语料，含 660
> `realistic`），**不是** `10_ocr_research/gt_review_loop.xlsx` 验收集，也不是 plausible v2
> 语料。该旧语料含随机国家码 / 非法日期，与 ICAO syntax mask + 校验位 DFS 冲突，
> 见 `10_ocr_research/corpus_eval_data_curing_1106.md` 与
> `10_ocr_research/corpus_compliance_review_1106.md`；因此表里 cnn `full match 0.00%`
> 主要反映**语料内容 / 解码先验冲突**，不是生产权重崩坏，不得用于宣称达标或回退。
> `metrics_history.md` 的 23:24 两行**未记录 `w=` 权重指纹**，不能倒推当时权重集；
> 现行生产基线按 `AGENTS.md` 为 `40951c5`，当前工作区三个权重头 sha256 前缀为
> `crnn_weights.h d8c6624e…` / `cnn_weights_line1.h 49f9fb50…` /
> `cnn_weights_line2.h 4c6df06f…`（仅标识当前工作区，非 23:24 运行指纹）。
> 分口径报告须同时列出 line exact / both exact / full match / cksum valid；
> `cksum valid` 不是 Exact，产品路径计时以整链 `algorithm_ms` 为准（见 `AGENTS.md`）。

## Overall

| method | pipeline OK | ms p50 | line1 acc | line2 acc | full match | cksum valid |
|--------|-------------|--------|-----------|-----------|------------|-------------|
| traditional | 1380/1380 (100.0%) | 5.6 | 54.44% | 16.46% | 0.00% | 0.00% |
| cnn | 1380/1380 (100.0%) | 10.4 | 76.32% | 59.00% | 0.00% | 29.64% |

## Breakdown by profile
| method | pipeline OK | ms p50 | line1 acc | line2 acc | full match | cksum valid |
|--------|-------------|--------|-----------|-----------|------------|-------------|
| clean | 100.0% | 5.5 | 62.30% | 25.53% | 0.00% | 0.00% |
| clean | 100.0% | 10.2 | 81.62% | 71.11% | 0.00% | 37.64% |
| realistic | 100.0% | 5.8 | 45.87% | 6.56% | 0.00% | 0.00% |
| realistic | 100.0% | 10.6 | 70.55% | 45.79% | 0.00% | 20.91% |

## Top confusions (traditional)
| gt->got | count |
|---------|-------|
| K->< | 1531 |
| P->< | 1525 |
| 5->S | 1517 |
| 1->4 | 1311 |
| 8->B | 1259 |
| 0->2 | 1155 |
| M->Y | 1124 |
| U->L | 1099 |
| 9->V | 1081 |
| 6->A | 1061 |
| Z->2 | 948 |
| <->L | 897 |
| J->3 | 870 |
| B->3 | 859 |
| 6->4 | 847 |

## Top confusions (cnn)
| gt->got | count |
|---------|-------|
| 6->A | 1190 |
| 5->9 | 961 |
| 3->5 | 879 |
| Q->O | 827 |
| U->L | 760 |
| T->Y | 756 |
| 5->7 | 625 |
| K->< | 526 |
| G->C | 503 |
| B->0 | 463 |
| O->G | 427 |
| <->P | 419 |
| 6->9 | 414 |
| D->M | 402 |
| X->< | 401 |

## Worst columns (traditional)
| line1 col | errors | line2 col | errors |
|-----------|--------|-----------|--------|
| 0 | 1285 | 12 | 1253 |
| 2 | 1206 | 34 | 1249 |
| 3 | 1153 | 7 | 1242 |
| 5 | 1151 | 32 | 1231 |
| 4 | 1119 | 11 | 1228 |

## Worst columns (cnn)
| line1 col | errors | line2 col | errors |
|-----------|--------|-----------|--------|
| 2 | 1040 | 2 | 777 |
| 3 | 1001 | 8 | 748 |
| 4 | 981 | 40 | 729 |
| 6 | 641 | 7 | 716 |
| 5 | 628 | 1 | 710 |
