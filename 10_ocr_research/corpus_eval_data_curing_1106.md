# corpus_eval Data Curing 报告

> 日期：2026-11-06
> 触发：1380 样本集 fuse both exact = 17.9%，用户要求处理

## 1. 问题

`6_mrz_ocr/data/corpus_eval/` 1380 样本集 fuse both exact = 247/1380 = **17.9%**，
line1 exact = 19.6%（瓶颈），line2 exact = 77.8%。

## 2. 误差归因（按 skill §8.1 三类）

### 2.1 A 类（定位/切分/约束）— 主因

**根因**：corpus_eval 用 `--legacy-random` 生成，GT 国家码全部为随机 3 字母
（60 个 issuer，0 个真实 ISO-3166 码）。C 工具的 issuer fix / ICAO 约束解码
（`mrz_ocr_cnn.c:198` k_iso3 表 + `mrz_code3_plausible`）会把随机国家码
"纠正"成接近的真实国家码，导致 line1 col 2-4（国家码位置）全部错误。

**证据**：
- v0（最干净样本，noise=0）line1 错误集中在 col 2/3/4：col 2=69错, col 3=64错, col 4=71错，其他列几乎没错
- 混淆模式不是单字符混淆（如 N→M），而是整个 3 字母国家码完全识别错误（HCF→MLT, FRX→ERI）
- `gen_corpus.py:163-166` 注释明确说："the benchmark then measured that collision, not recognition"

### 2.2 B 类（字符分类）— 次因

noise≥0.03 时字符图像退化致分类错误。noise=0.05 时 both=7.2%。

### 2.3 C 类（解码/后处理）— 微量

校验位 / syntax mask 正常工作，cksum valid 与 exact 差距正常。

## 3. 修复（Data Curing，零风险）

重新生成 corpus_eval（`plausible=True`，真实 ISO-3166 国家码）：

```bash
python3 6_mrz_ocr/tests/gen_corpus.py --eval --out data/corpus_eval --seed 2002 --font ocrb --aa
```

- 旧 corpus_eval：`plausible` 字段不存在（旧版生成），60 个 issuer 全部非真实 ISO-3166
- 新 corpus_eval：`plausible=True`，19 个 issuer 全部真实 ISO-3166（AFG, AGO, ALB...）
- 样本数：1380 → 1440（60 docs × 24 variants）

## 4. A/B 结果

| 指标 | 旧 (legacy-random, 1380) | 新 (plausible, 1440) | 提升 |
|------|--------------------------|---------------------|------|
| l1_exact | 19.6% | **78.9%** | +59.3pp |
| l2_exact | 77.8% | **84.2%** | +6.4pp |
| both | 17.9% | **72.0%** | +54.1pp |
| l1_char | 91.94% | 95.45% | +3.51pp |
| l2_char | 94.62% | 95.35% | +0.73pp |

### 按变体分布（新 corpus_eval）

| v-index | both | 说明 |
|---------|------|------|
| v0 (clean) | 95.0% | noise=0, skew=0 |
| v9 (scale=3) | 100.0% | 小字符 |
| v11 (noise=0.05) | **9.2%** | 主要失败源 |
| v10 (noise=0.03) | 48.3% | |
| v4 (scale=5) | 50.8% | 大字符 |

### 按 profile 分布

| profile | both |
|---------|------|
| clean | 87.5% |
| realistic | 56.5% |

### 按 noise 分布

| noise | both |
|-------|------|
| 0.00 | **97.1%** |
| 0.01 | 78.7% |
| 0.05 | **7.2%** |

## 5. 门禁

- `test_postproc`：147/147 全通过
- `test_mrz`：89/89 全通过
- `corpus_manifest.py generate`：重新签名成功（2821 文件）
- TD3 零退化：未改 C 工具，未改权重，零风险

## 6. Skill §7 五要素

1. **上游 skill**：Data Curing / 真值清洗（§8.2 #1），方法论"以实测混淆集为准"
2. **架构兼容性**：不改架构/Loss/解码，与 §3 完全兼容
3. **A/B**：单变量 = GT 国家码分布（legacy-random vs plausible）
4. **翻车清单**：不触及 §5（未改 Loss/lr/类别权重/预处理）
5. **红线**：不触及 §6（TD3 零退化、FAST_NATIVE_MAX_WIDTH、real-frac 均不变）

## 7. 剩余瓶颈（B 类字符分类）

- noise=0.05: 7.2%（重度噪声致字符图像退化）
- realistic: 56.5%（旋转/模糊/光照梯度）
- 注意：全量重训练均退化（2026-11-04），不宜盲目重训
- noise=0.00 已达 97.1%，接近 95% 目标
