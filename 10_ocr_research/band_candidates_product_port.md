# band 候选自选 → 产品路径移植设计(待设备验证)— 2026-11-05

## 证据(为什么这是当前最大的单点 exact 收益)

同一批 assets/pic 照片(官方 63 张 TD3),同一套权重,只差 band framing:

| framing | l1 char | 出处 |
|---|---|---|
| 候选自选(mrz_band.py: fast/res/sens 三路 + line2 校验位选优) | **97.15%** | eval_weight_sets(官方 GT) |
| 产品 restore pair + pad 8..25 | **74.17%** | eval_by_algorithm_1104 B 节 |

差距是 framing 而非字形。10-24 的"训练/评测 band 分布锁"教训在**产品侧**的复发。

与被否决方案的区别:`mrz_ocr_cnn.c:2740` 注释记录的 bottom-up 救援否决是
**页面级**(整页二值化上做 row-span/ink 分离,花纹底 3 次原型均失败)。
本设计不触碰页面定位,只在**已裁 band** 的 2-3 个候选 framing 里按 line2
校验位选优——与 1104 文档 D-② 的结论一致。

## 产品路径补丁(PhotoMrzPipeline.kt,需设备验证)

挂接点:`processFast` 内,现 fastNativeInput 裁剪处(fastPair restore pair 之后)。

```kotlin
// 1) 生成 2 个备选 framing:在 fastPair 现有 restore pair 的 y 上,
//    各外扩/内缩一行高(与 mrz_band.py 的 sens/res 等价的廉价变体):
val candidates = listOf(bandRect, bandRect.expandTop(bandRect.height/8),
                        bandRect.expandBottom(bandRect.height/8))
// 2) 预算内的循环:逐候选调 NativeEngine MODE_CNN;首个 line2 ICAO
//    五段校验全过 且 structOk 的候选即胜出;全失败则保留第一个结果
//    (与现状一致,回退语义不变)。
// 3) 预算闸门:每候选开始前检查 remaining >= 单次引擎成本(FALLBACK
//    闸门同款逻辑);候选数 ≤ 3,p50 不变(首候选命中率 ~98%),p95
//    上界 +1 次引擎 ≈ 65ms → 与现有 fallback 预算合并计算,不叠加。
```

注意:
- 只在**首候选未过校验**时才尝试次候选——不改变成功帧的任何行为;
- 与 MODE_FUSE fallback 的关系:先 framing 选优(便宜、同引擎),后
  fuse(贵),两层各有预算闸门;
- `isMrzShape` + `IcaoChecksum.line2Ok` Kotlin 侧现成,无需新增校验代码。

## 验收口径(设备上)

1. assets 官方 63 张 GUI 直跑:目标 both exact 49/59 → **≥55/59**;
2. passport 官方 44 张:不得回退(40/41 保持);
3. 耗时:p50 不变;p95 在预算闸门下仍 ≤100ms(次候选只在失败帧触发);
4. 回归:`test_postproc` / `test_golden` / corpus_eval_v2 bench 三件套。

## 备选(若设备验证发现次候选命中率高但预算吃紧)

把"次候选尝试"挪进 C 侧 JNI 一次调用里(传 band 列表,C 内循环选优),
省 2 次 JNI/位图拷贝开销——C 侧签名扩展 `recognise_bands(bands[], n)`。

---

## 2026-11-06 落地验证：桌面 GUI 侧两阶段一致性门控（非 Kotlin 路径）

> 本节记录 `tests/gui.py` 产品路径（`_ocr_pick_band`）的同源实现与**桌面端实测**，
> 与上文 Kotlin/设备侧设计是**两条独立链路**：本节只验证「候选自选的
> 准确率收益 vs 预算冲突」这一共性结论，不构成设备侧验收。

### 背景：本文 §3 的 p95 预算在桌面侧被实测推翻

桌面 GUI 早已把候选自选全量上线（`MRZ_BAND_PICK_ALL=1` +
`MRZ_BAND_PICK_VOTE=1` + padding 20/30/40），但**没有实现 §3 的预算闸门**
（「只在首候选未过校验时才尝试次候选」），变成每张照片都评分约 12 个候选。
正式 50 张 bench 实测 official 整链 `mean/median/p95 = 197.8/179.8/377.3 ms`，
**0/50 满足 <100 ms**；其中 `_ocr_pick_band` 占 185~237 ms，而 fuse 引擎本身只
~15 ms。这直接证伪了本文 §3 「p50 不变、p95 ≤100 ms」在无闸门时的成立前提，
也说明该设计里**预算闸门不是可选项而是必需项**。

### 落地：两阶段一致性早停（`MRZ_BAND_PICK_GATE`，默认 1）

`tests/gui.py:5547, 5672-5688`。primary band 读数已付过一次 C 推理；
若 line2 校验合法且非 CS/CD，再用 `pad_override=20` 重建一个独立裁剪读一次，
两行逐字节一致即早停（等价 §3 的「首个合法候选即胜出」+ 一次独立复核）。
primary 不合法 / CS/CD / 读数不一致 / 异常 → 落回完整候选投票。
`GATE=0` 逐字恢复未门控路径，用于单变量 A/B。**不新增进程、不改权重、
不引入跨引擎 fallback。**

### 单变量 A/B（唯一变量 = GATE，50 张，--resident，同权重/同 GT/同三引擎）

| 配置 | PASS | verified | fallback | mean | median | P95 | <100ms |
|---|---|---|---|---|---|---|---|
| GATE=0（未门控 ALL=1） | 40/50 | 48 | 0 | 193.5 | 174.3 | 375.1 | 0/50 |
| **GATE=1（新默认）** | **40/50** | 48 | 0 | **99.3** | **64.8** | **215.8** | **35/50** |
| GATE=1 + `MRZ_BAND_PICK_PAR=8` | 40/50 | 48 | 0 | 105.3 | 65.1 | 270.6 | 35/50 |
| 参考：HEAD 默认 ALL=0 VOTE=0 | 38/50 | 48 | 0 | 58.0 | 47.7 | 106.5 | 47/50 |

所有配置三引擎 raw GT 恒为 `cnn 37/47 / crnn 28/47 / fuse 37/47`，
单引擎耗时 ~10.4-11.3 mean ⇒ **候选选择只吃时间，不改变识别结果**；
门控换来 mean -48.7% / median -62.8% / P95 -42.5% 且准确率零退化。
`PAR=8` 为**负结果**（P95 反而升到 270.6 ms，并出现 528 ms 离群），
并发度保持 4，禁止用「调大 PAR」压 P95。

### 结论：剩余 P95 缺口与 Exact 缺口同源，治本在行分割

GATE=1 的 15 张 ≥100 ms 中 **9 张是 FAIL**；10 张 FAIL 里仅
`2026-9-30-H.jpg`（64.1 ms）走快路径。`Photo/3..8` 六张全部 FAIL 且
129.5~445.8 ms，`Photo/8` 单张 `_ocr_pick_band` 就占 456.7 ms。
这些正是早前定位的**宽 band + 顶部暗区/黑边导致行分割把 line1 定到暗区**的样本。
⇒ 继续压缩候选预算只会把失败藏起来；正确路径是
`6_mrz_ocr/src/mrz_geom.c::mrz_ocr_split_lines` 与 `_rst_native_mrz_band`
的垂直裁剪交互。**本文 §3 的 Kotlin 设计仍待设备验证，且必须带上预算闸门。**

### 门禁与红线（按项目强制 skill 五要素核对）

- 上游 skill：仅借 `rigor-reviewer`（评估严谨性）/ `research-manager`（死胡同归档）
  方法论层；vendored 库无 OCR / 小 CNN skill，CTC / LoRA / 分布式配方标注「不适用」。
- 架构兼容：改动在 Python 侧候选**调度**，不触碰 C 侧 44×2 网格、37 类逐字符 CE、
  syntax mask + 校验位 DFS、任何权重 ⇒ 属「已实现链路优化」，
  不触发 `mrz_ocr_cnn.c:869-873,3261` 的整体重训红线。
- 门禁：`py_compile` OK；`test_postproc` **147/147**、`test_mrz` **89/89**；
  单张 smoke GATE=1 `[114.7, 97.6, 70.7] ms` vs GATE=0 `[223.2, 205.3, 180.5] ms`，
  均 PASS / GTmatch / fallback=0。官方 PASS gate `assets/pic/90b464ea218a9909_0001.jpg`
  两臂均 grade=FAIL（既有 line2 pos22 `F`→`O`），**非本次回归**。
- 翻车清单：未触及类别重采样/重加权、`class-boost`、局部对比度归一化、字段级 voting 任一条。
- 红线：`fallback=0`、TD3 零退化、98%/95% 逐字符口径、唯一 GT `gt_review_loop.xlsx` 均未变。

**仍未达标**：产品路径 P95 215.8 ms（<100 ms 仅 35/50）；exact 仍 37/47（目标 ≥45/47）；
`Photo/3–8` 全 FAIL。原始数据 `/tmp/bench_G1.json`、`/tmp/bench_G0.json`、
`/tmp/bench_G1_PAR8.json`；完整日志见 `.workbuddy-ai/memory/2026-11-06.md` §24–§25。
