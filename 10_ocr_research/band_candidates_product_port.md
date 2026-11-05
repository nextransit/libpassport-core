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
