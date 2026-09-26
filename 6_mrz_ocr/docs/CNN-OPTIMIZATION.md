# CNN MRZ OCR 优化技术架构

> 覆盖 2026-09-26 对 `6_mrz_ocr` CNN 识别链路的完整优化。
> 本文描述**最终架构**与**演进脉络**;数字以 `data/corpus_eval`(1380 张,
> 冻结、不参与训练)实测为准,见 `bench_results.md` 与 `metrics_history.md`。

## 1. 总体结论

| 阶段 | 结果 | 说明 |
|---|---|---|
| 初始基线(合成 500 张,脏评估) | line1 77.9% / line2 71.1% | 错误主要是数字↔字母混淆 |
| 后处理约束修复(fill_allow) | line1 96.4% / line2 90.0% | 白名单不再与 GT 冲突 |
| 评测协议修正(ICAO-valid GT + train/eval 分离) | CNN 反超传统 | 暴露真实水平 |
| 分布匹配重训(train_cnn_real.py) | clean 子集 ~100% | 消除合成渲染器与推理分布差 |
| realistic 语料(rot/blur/illum/invert) | 见 bench_results.md | 尚未饱和,是当前主要短板 |

核心判断:**CNN 精度的主要瓶颈从来不是网络容量,而是三件事**——
(a) 后处理语法约束与真实数据冲突;(b) 训练数据分布与 C 端推理
输入不一致;(c) 评估协议把训练集当测试集。三者逐一修复后,网络
本身(2 卷积块 + 1x1 conv + 两层全连接,约 41 KB 权重)已足够。

## 2. 识别流水线(最终架构,src/mrz_ocr_cnn.c)

```
输入灰度页
  1 全局 Otsu -> 粗二值图                 (定位用,容忍全页光照)
  2 行密度投影定位 MRZ band
  3 band 局部二次 Otsu                    (全页阈值被照片/背景主导时仍准)
  4 band 倾斜估计 + deskew(>=0.4° 才旋转) (墨点旋转投影能量最大化)
  5 两行 TD3 切分
  6 pitch-grid 切字符(44 定长格)          (主),run 窗口(回退)
  7 每个 cell: 灰度 area-resample -> 16x12 float
     3x3 box smooth + 质心对齐(±2px)
  8 整行批量 conv+fc 推理                  (可选 MRZ_OCR_TTA=1: ±1px 三视图 logits 平均)
  9 ICAO 9303 严格语法掩码                 (可选 MRZ_OCR_STRICT_ICAO=0 回退宽松掩码)
 10 确定性后处理: 相对墨量门 + 3邻'<'规则 + line1 尾部填充锁定
 11 line2 校验和 beam-search(top-3,最低置信优先,单/双翻转)
```

### 2.1 网络结构(include/cnn.h, src/cnn.c)

```
input  12 x 16 灰度 float(墨迹高亮, 软边缘)
conv1  3x3, 8 filters, pad=1 -> ReLU
pool1  2x1 max(仅高度)        -> 6 x 16 x 8
conv2  3x3, 16 filters, pad=1 -> ReLU
pool2  2x2 max                -> 3 x 8 x 16
conv3  1x1, 4 filters         -> 3 x 8 x 4 = 96
fc1    96 -> 64 (ReLU)
fc2    64 -> 37 (softmax)
```

- 输入保留**灰度软边缘**,不做硬二值化——子像素笔画信息进入 conv1。
- pool1 只在高度方向降采样,**宽度空间分辨率受保护**,±1px 位移不会
  塌缩成 2px 宽特征图(历史 8x12 二值化版本的坑)。
- 整行 44 字符一次性批量推理,无逐字符 malloc。
- 大缓冲(权重、glyph 批、概率)全部堆分配,栈帧从 ~110 KB 降下来
  (嵌入式线程栈友好)。
- 性能优化(`src/cnn.c`):conv1/conv2 的 3x3 感受野 load 提出通道循环,
  浮点求和顺序保持不变;删除一段死循环。同一 500 张合成语料下 CNN
  p50 从 ~6.0ms 降到 ~4.5ms;当前 eval 语料(含 deskew、realistic)
  p50 7.4ms、p95 10.3ms。

### 2.2 切字符:pitch-grid 是最大单项改进

TD3 行恰为 44 个等距 cell。旧 run 分段"每个墨迹 run 一个字符",
一个噪声 run 会把后续所有窗口推歪、一个粘连 run 会丢字符。
pitch-grid 改而**拟合全局网格**:

- pitch L = 相邻 run 中心差的中位数(合法区间内),自相关回退;
- phase c0 = 在 [0,L) 上穷举,按 88 个窗口边缘附近的墨量加权打分
  (边缘必须落在字符间空白,权重 1-2-3-2-1 打破零平台);
- 窗口位置来自拟合网格,宽度仍是 medw+8(与训练分布一致);
- 拟合失败时回退 run 分段。

注:此前"43 字符错位"曾被误判为 segmenter bug;实测是评估语料本身
就是 43 字符 free-form,与 segmenter 无关。

### 2.3 后处理:语法掩码(严格 ICAO 模式,默认)

`fill_allow` 按 ICAO 9303 TD3 的每个槽位给 37 类白名单:

- line1: col0 任意;col1 `P` 后填充 `<`;col2-4 发行国 3 字母;
  col5+ 姓名字段 A-Z + `<`(ICAO 规定名字无数字);
- line2: col0-8 证件号(字母/数字/`<`);col9/19/27/42/43 校验位
  (数字,尾部容忍 `<`);col10-12 国籍字母;col13-18/21-26 日期数字
  (+`<` 表示未知);col20 性别 M/F/`<`;col28-41 个人号字母/数字/`<`。

宽松模式(`MRZ_OCR_STRICT_ICAO=0`)保留历史行为,用于 free-form
语料/调试。

**教训**:上一版把 country/name/date 全部放开成 any-printable 曾把
line1 从 77.9% 拉到 96.4%,但那是因为评估语料把数字混进字母字段、
白名单与 GT 冲突。真实护照这些位置必须严格,**放开规则等于用脏语料
换分数**。因此现在的协议改为:GT 全部 ICAO-valid,掩码默认严格,
白名单永远包含真值、只删不可能类。

### 2.4 line2 校验和 beam-search

五个 ICAO mod-7/3/1 校验位(证件号、出生、有效期、个人号、复合)全
部通过才算数。beam 从 top-3 候选中、按置信差(p0-p1)从小到大(最
不确定的先试),单位置翻转后双位置翻转;复合校验让**每一列**都参与
搜索,无先验排除列。合成 clean 语料上 cksum 有效率达到 ~99.9%。

### 2.5 deskew 与光照

`estimate_band_skew`:只访问墨点像素,对 band 内容按 -alpha 旋转后
的行投影能量打分,-4..+4°、0.5° 步进;需比无旋转得分高 >5% 才采用。
`rotate_band` 反向映射最近邻。光照梯度由 **band 局部二次 Otsu** 抵消
(全页阈值会被照片/阴影主导)。合成 realistic 语料(rot ±3°、blur、
illum、invert、noise≤0.05)下 CNN 仍明显优于传统。

## 3. 训练与评估协议(数据完整性问题)

### 3.1 根因

早先报告"重训后 inference 恶化到 76%",根因是**训练分布与 C 端
推理输入不一致**:

- C 端 `segment_line_cnn` 的窗口是 `medw+8 = 6*scale+8`;
- 旧合成渲染器给窗口 8*scale+8 或裁剪 ink bbox,训练字形比推理字形
  **胖 10-25%**;
- 于是 val_acc 自洽(98.9%)、真实推理反而退化。

且历史 100.0%/99.5% 是在**训练集本身**(85% 的图就是训练数据)上测得,
是无效数字。

### 3.2 现在的协议

- **两个不相交、可位级复现的语料**(`tests/gen_corpus.py`):
  - `data/corpus_eval`(seed 2002,冻结):只供 bench + gate,永不训练;
  - `data/corpus_train`(seed 1001):只供训练。
  - 每条记录用固定 noise-seed 渲染,语料目录可逐位重建、结果可比。
- **GT 全部 ICAO-valid**:line1 名字字段仅 A-Z/`<`;line2 带真实
  mod-7/3/1 校验位、数字日期、M/F/< 性别。
- **realistic 剖面**:rot ±3°、blur、光照梯度、反白、noise≤0.05,
  两个语料都含(eval 660/1380)。
- **分布匹配训练**(`tools/train_cnn_real.py`):以 `MRZ_OCR_DUMP=1`
  跑 C 流水线,直接采集网络真实看到的 16x12 glyph 作为训练数据,
  按图像划分(无泄漏),每次运行重建 workdir 并清理 /tmp dump(防
  陈旧 glyph 配错标签);可选 `--mix-synthetic N` 混入 v2 渲染样本
  补覆盖。
- **回归门**(`tests/bench.py --gate data/baseline.json`):任何关键
  指标回退 >0.3pp 或低于绝对下限则失败;通过后刷新 baseline。
  baseline 在 `data/baseline.json`。
- `loop.sh` 一键闭环:build → 语料 → 单测 → bench → gate → 失败挖掘
  → 示例识别。`--retrain` 先重训。

### 3.3 失败驱动迭代

`bench.py` 输出 `failures_{method}.json`;`tools/mine_failures.py`
按渲染条件(profile/scale/noise/rot 桶)、MRZ 列、混淆对聚类,并给
出下一轮动作(扩展语料桶、加权 hard-pair)。每次 bench 追加
`metrics_history.md`。

## 4. 测试与质量

- `ctest` 4 项全过:`mrz_ocr_unit`、`mrz_postproc`(直接 include
  mrz_ocr_cnn.c 测静态后处理)、`mrz_golden`(进程内渲染 ICAO-valid
  记录,双后端必须逐字符精确复现)、`face_unit`。
- `run.sh` 冒烟;`loop.sh` 完整闭环。

## 5. 当前数字与未完成

- 当前 bench(eval 1380,含 660 realistic,2026-09-26 12:57):见
  `bench_results.md` 与 `metrics_history.md`。
  - traditional: pipeline OK 56.4%,line1 44.6% / line2 36.0%,
    full-match 0.0%,cksum 2.8%,p50 4.1ms;
  - cnn: pipeline OK **100.0%**,line1 **95.6%** / line2 **98.9%**,
    full-match 71.5%,cksum 94.6%,p50 7.4ms。
  - clean 子集(720)CNN full-match 95.7%、cksum 100%;realistic 子集
    (660)CNN full-match 45.0%、cksum 88.8%。
  - **回归门当前 FAIL**:`cnn.full_match_rate 71.45% < floor 85%`
    —— realistic 子集仍未饱和,是当前主要短板,也是后续迭代主攻方向
    (`loop.sh` 的失败挖掘/语料扩展闭环)。这是如实状态,不是已达标。
- **未交付**:真实护照扫描样本验收;realistic full-match 达到 85%
  floor;CNN 相对传统的速度差距(7.4ms vs 4.1ms);int8/定点量化评估。

## 6. 关键文件地图

| 文件 | 职责 |
|---|---|
| `src/mrz_ocr_cnn.c` | CNN 流水线:deskew/切分/重采样/解码/后处理 |
| `src/cnn.c` + `include/cnn.h` | 卷积网络推理(批量、堆分配、conv unroll) |
| `src/cnn_weights.h` | float32 权重(train_cnn_real.py 生成) |
| `src/mrz_ocr.c` | 共享 band/lines/Otsu 定位 |
| `tools/train_cnn_real.py` | 真实分布训练(C 端 dump 采集) |
| `tools/train_cnn_v2.py` | 向量化合成渲染器(窗口已对齐 6s+8) |
| `tools/gen_mrz_image.c` | 渲染器:rot/blur/illum/invert/noise-seed |
| `tests/gen_corpus.py` | 双语料确定性生成(ICAO-valid GT) |
| `tests/bench.py` | bench + 回归 gate + failures 导出 |
| `tools/mine_failures.py` | 失败聚类 + 下轮建议 |
| `tests/test_golden.c` / `test_postproc.c` | 进程内回归测试 |
| `loop.sh` / `run.sh` | 闭环 / 冒烟 |


## 7. 架构决策:几何解耦 + 产品路径定调(2026-09-26)

**战略:执行「最小预处理解耦」+ 敲定「CNN 独占生产管线」,坚决不做「重写传统
特征匹配引擎」。**

背景:传统后端在 1380 张 eval 语料上只有 44.57%/36.04%,曾被误读为"需要
升级"。拆解后真实根因:
- 602/660 realistic 图直接 pipeline FAIL,**全部 rot!=0**——传统后端没有
  deskew,行投影在几度倾斜下直接失效(连字符识别都到不了);
- clean 720/720 全部跑通但只有 75.7%——8x12 逐比特 XOR 匹配器本身弱。

### 7.1 共享几何预处理层(src/mrz_geom.c)

`estimate_band_skew` / `rotate_band` 本是 CNN 模块私有的 static,但它们属于
光学前处理几何层。已提取为独立公共 API:
- `mrz_estimate_band_skew(bin, W, H)`:墨点行投影能量,±4° 0.5° 步进,
  >5% 才采用;
- `mrz_rotate_band(src, W, H, deg, fill, dst)`:反向映射最近邻旋转,
  fill 参数区分灰度(255)/二值(0)。

两个后端现在共用:
```
            [ 灰度/多光谱图 ]
                    │
 ┌──────────────────▼─────────────────────┐
 │ 共享预处理层 (mrz_geom.c)               │
 │  locate_band → estimate_skew → rotate  │
 └──────────────────┬─────────────────────┘
          ┌─────────┴─────────┐
          ▼                   ▼
   [ CNN 生产主线 ]      [ 传统 LEGACY 对照 ]
```
- CNN:band-local 二次 Otsu + deskew(与之前一致,行为不变);
- 传统:全局 Otsu + deskew(新接入),行切分前无条件校正旋转。

**实测(1380 eval):传统 pipeline OK 从 56.4% → 100%,realistic 从 8.8% →
100%;传统 line1 44.57% → 63.58%,line2 36.04% → 42.60%;realistic 传统
line1 2.95% → 42.70%。CNN 数字不变(95.63/98.87),4/4 ctest 通过。**

### 7.2 传统后端定调:LEGACY BASELINE(冻结)

`src/mrz_ocr.c` 已加 banner:`LEGACY BASELINE - DO NOT ADD NEW FEATURES`。
其 8x12 逐比特 XOR、run 分割、无语法掩码/无校验和 beam 的行为**永久冻结**,
只作为 A/B 测试的对照组存在,证明"无学习能力的浅层分类器"天花板
(clean 75.7% vs CNN 98.5%)。回归门 FLOORS 只绑定 CNN。

### 7.3 已满足项:Line-2 双字联合校验回溯

检查清单第 3 项"Top-2 → 2x2 最小裕度联合翻转"**已实现于 decode_row**:
top-3 候选 + 置信裕度(p0-p1)升序 + 单位置翻转 + 双位置 2x2 联合翻转
(取裕度最小的 6 个位置对)。无需重复实现;当前 cksum 有效率为 94.64%。

### 7.4 下一步(聚焦 CNN realistic full-match 45% → 85%)

- 前端:单行 X 轴滑动窗口局部背景/前景归一化(替代全局 Otsu/1-ink/255),
  抗光照梯度与反光;
- 训练:把 realistic 中置信 <0.8 或 Top-2 纠错的 16x12 字符 patch 落盘,
  按 30% 权重与 clean 混合重训;
- 验收:以 1380 eval 的 realistic full-match 与 cksum 为唯一指标。

### 7.5 光照梯度归一化 A/B(2026-09-26,默认 OFF)

针对 realistic illum>=30 桶(失败率 55~100%)实现了共享层局部对比度归一化
`mrz_normalize_local_contrast`(src/mrz_geom.c):逐列滑动窗口 p95 背景 +
band 全局 p01 墨参考,滑动直方图 O(W*H),~0.5ms;输出极性 bg=255/ink=0,
gate:`MRZ_OCR_LOCAL_NORM=1`。

**A/B 结论(1380 eval):**
- NORM ON 但**权重重训前**:illum>=40 桶 line1 78.37 vs 78.44、line2 96.58
  vs 96.24——无差异(权重未适配归一化输入分布);
- NORM ON + hard-slices 重训后:字符精度升(line1 95.63->97.60,line2
  98.87->99.05,cksum 94.64->97.10),但 **realistic full-match 45.0%->
  37.9% 反而降**,clean full-match 也从 95.7% 微降;
- 因此 NORM 默认 OFF(产品验收只看 full-match),代码与 gate 保留供现场 A/B。

**教训**:full-match 是占了"双行同时全对"的联合指标,字符精度升不等于
整行可交付。前端归一化 + 重训的方向与 hard-slices 30% 权重一样,都提高
单字、压低整行——真正能推 full-match 的路径还没找到,优先怀疑切分/deskew
残留误差与低置信双字,而非单字分类器。

### 7.6 realistic full-match 45%->85% 瓶颈定位(2026-09-26,证据)

对 660 realistic 全量逐字符复测,full-match 失败 363 张(55%)构成:

- 失败形态:仅 line1 错 152、仅 line2 错 93、两行都错 118;
- 主因条件:rot>=2 占 234,illum>=30 占 68,rot<1 但 illum 轻 55;
- **line1 行首 col0-4 是 dominant**:col0(文档类型位 P)错 95 次
  (P->F 57、P->< 26、P->B 8),col2-4 国码区其次(U->O、E->F、
  I->T 等)。rot<1 也存在行首错 53 次 -> 不是纯旋转残差。

结论(推 full-match 的正确抓手):
- 单字分类器已不是瓶颈(NORM/hard-slices 重训把字符精度推到
  97.6/99.1 但 full-match 反而降),line1 行首 P/F 混淆 + 高位 rot
  是联合失真主因;
- 下一步两个候选,只能选产品收益可证的那个:
  a) line1 col0/col2-4 的段级后处理(文档类型位掩码收窄到 P/I/A/
     V+<、国码 A-Z,已含于 strict mask;改为对 col0 用 top-3 强制
     P/I/A/V 候选,不依赖 beam);
  b) 高位 rot 的 deskew 残差补偿(双线分离后按行再估一次斜)。
- 验收口径不变:realistic full-match,45%->85%。
