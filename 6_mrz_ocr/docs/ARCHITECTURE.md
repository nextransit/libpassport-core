# MRZ OCR 完整技术架构(CNN 生产主线 × 传统基线对照)

> 模块:`6_mrz_ocr/` — 纯 C(float32)推理,无第三方运行时依赖。
> 对照当前已提交基线(HEAD=8462ef5):`c` 端 4/4 ctest 通过,
> eval corpus(1380 张,660 realistic)实测 CNN 95.75/98.87,full-match
> 71.81%,cksum 94.64%,p50 6.4ms。本文为架构、方案、训练与微调全量文档。

---

## 0. 一句话定位

```
输入灰度页/条带图
  → 共享光学预处理几何层(找出 band、deskew、切两行)
  → 两条并行识别后端
       ├─ CNN(生产主线):pitch-grid 切 44 格 → 灰度 16x12 → ConvNet
       │    → ICAO 语法掩码 → 确定性后处理 → line2 校验和 beam
       └─ 传统模板(LEGACY 对照,冻结):run 切分 → 8x12 逐比特 XOR → 无后处理
```

**产品/嵌入式路径只走 CNN**;传统后端是 A/B 评测对照组,代码已打
`LEGACY BASELINE - DO NOT ADD NEW FEATURES` 冻结标记。

---

## 1. 数据流总览

### 1.1 入口与输出

| 工具 | 入口 | 输出 |
|---|---|---|
| `build/mrz_ocr_cnn_tool <img>` | PPM/BMP | `result.line1/line2/conf1/conf2/band` |
| `build/mrz_ocr_tool <img>` | PPM/BMP | 同上(传统后端) |
| `mrz_ocr_recognise_cnn()` | `face_image_t*` | `mrz_ocr_result_t{line1,line2,conf,...}` |

### 1.2 判定集

- 37 类: `A-Z, 0-9, '<'(填充符)` — 与 OCR-B 模板银行完全一致。
- GT 全部 ICAO 9303 TD3 合法(44 字符/行):line1 名姓仅 A-Z+`<`、
  line2 带真实 mod-7/3/1 校验位、日期数字、性别 M/F/`<`。

---

## 2. CNN 生产流水线(`src/mrz_ocr_cnn.c`,9 个阶段)

```
输入
 1 全局 Otsu -> 粗二值图(band 定位用,容忍全页光照)
 2 行密度投影 -> 定位 MRZ band(共享)
 3 band 局部二次 Otsu(全页阈值被照片/背景主导时仍准)
 4 deskew(共享几何层):估倾角 + 旋转校正
     └ 默认最近邻;MRZ_OCR_BILINEAR=1 切双线性(opt-in,见 §7)
 5 两行 TD3 切分(共享)
 6 pitch-grid 切 44 定长窗(主)/ run 窗(回退)
 7 每格: 灰度 area-resample -> 16x12 float
     + 3x3 box smooth + X/Y 质心对齐(±2px)
     └ 可选 MRZ_OCR_LOCAL_NORM=1 局部对比度归一化(opt-in,默认 OFF)
 8 整行批量 conv+fc 推理
     └ 可选 MRZ_OCR_TTA=1: ±1px 三视图 logits 平均(opt-in)
 9 解码 + 后处理
     · ICAO 严格语法掩码(fill_allow)
     · 相对墨量门(X/R 低墨 → '<')
     · 3 邻'<' 规则 / line1 尾部填充锁定
     · line2 校验和 beam(top-3,最低裕度优先,单+双字翻转)
```

### 2.1 网络拓扑(`include/cnn.h`+`src/cnn.c`,约 41 KB 权重)

```
input  12 x 16 灰度 float(墨高亮,软边缘; [-1,1] 归一: mean=0,std=1)
conv1  3x3, 8 filters, pad=1 -> ReLU
pool1  2x1 max(仅高度)      -> 6 x 16 x 8
conv2  3x3, 16 filters, pad=1 -> ReLU
pool2  2x2 max              -> 3 x 8 x 16
conv3  1x1, 4 filters       -> 3 x 8 x 4 = 96(flat)
fc1    96 -> 64 (ReLU)
fc2    64 -> 37 (softmax)
```

关键设计:
- **灰度软边缘输入**:不做硬二值化,子像素笔画保留给 conv1;
- **高度方向 2x1 pool 保宽**:水平空间分辨率不塌缩,±1px 位移不丢特征;
- **大缓冲全部堆分配**:CNN 权重 ~41 KB 不进栈,嵌入式线程栈友好;
- **整行 44 字符批量推理**:无逐字 malloc;
- 性能:conv1/conv2 感受野 load 提至通道外、求和顺序不变(精度一致),
  同一 1380 张语料下 p50 6.4ms(含 deskew/切分/后处理)。

### 2.2 切字符:pitch-grid(`segment_line_grid`,主)

- TD3 行恰 44 个等距 cell,不信任单个 ink run:
  - pitch L = 相邻 run 中心差中位数(合法区间),自相关回退;
  - phase c0 = [0,L) 穷举,88 个窗边 ±2px 加权重(1-2-3-2-1)
    打分,窗边必须落在字符间空白;
  - 窗口位置来自拟合网格,宽度保持 medw+8(与训练分布一致);
  - 拟合失败回退 run 分段(`segment_line_cnn`)。
- 历史"43 字符错位"实为旧 free-form 语料本身 43 字符,与 segmenter 无关。

### 2.3 解码后处理(`decode_row`)

1. **ICOA 严格语法掩码**(默认,`MRZ_OCR_STRICT_ICAO=0` 关):
   - line1: col0 ∈ {P,I,A,V,C,<}(文档类型)、col1='<'、
     col2-4 国码 A-Z、col5+ 姓名 A-Z+'<';
   - line2: col0-8 证件号 alnum+'<'、col9/19/27/42/43 校验位数字
     (尾两列容忍'<')、col10-12 国籍 A-Z+'<'、col13-18/21-26 日期
     数字+'<'、col20 性别 M/F/'<'、col28-41 个人号 alnum+'<'。
   - 掩码总包含真值,只删不可能类;col0 收窄曾 +0.36pp full-match。
2. **相对墨量门**:模型输出 X/R 但 patch 墨量不足
   (0.55 × 自信字母单元格中位墨量) → 判为 '<'。
3. **3-邻'<'**:夹在两边方框中的 X/R/T → '<'。
4. **line1 尾部填充锁定**:连续 ≥5 个'<'后其余全锁 '<'。
5. **line2 校验和 beam**:5 个 mod-7/3/1 校验位全过才算数;
   top-3 候选按置信差(p0-p1)升序,单字+双字组合翻转(最低裕度
   6 位),复合校验让每列都参与;当前 cksum 94.64%。

---

## 3. 共享光学预处理几何层(`src/mrz_geom.c`)

| API | 说明 |
|---|---|
| `mrz_estimate_band_skew(bin,W,H)` | 墨点行投影能量,±4° 0.5° 步进,需比无旋转高>5% |
| `mrz_rotate_band(src,W,H,deg,fill,bilinear,dst)` | 反向映射;`bilinear=1` 保留灰度亚像素(CNN),`0` 保二值(legacy) |
| `mrz_normalize_local_contrast(gray,W,H,out)` | 逐列滑动 p95 背景+band 全局 p01 墨参考,输入极性归一;输出 bg=255/ink=0;滑动直方图 O(W*H) |

- 由 CNN 与传统**共用** deskew,使传统 baseline 如实暴露分类器天花板
  (deskew 前 realistic pipeline OK 8.8%,deskew 后 100%)。
- 传统端保持最近邻+二值域(冻结);CNN 端可 opt-in 双线性/局部归一化。

---

## 4. 传统 LEGACY 后端(`src/mrz_ocr.c`,冻结)

- 二值化(极性归一 Otsu) → band 定位 → deskew → 双行切分 →
  **run 墨迹切分** → 8x12 硬二值 resample → **逐比特 XOR 模板比对**。
- 无语法掩码/无校验和/无墨量门;已冻结。
- 现状(eval 1380):pipeline OK 100%,line1 63.6%/line2 42.6%,
  realistic line1 42.7% —— 作为"无学习能力浅层分类器"的对照组。

---

## 5. 训练体系(分布匹配是核心)

### 5.1 原则

> **训练分布必须等于推理分布。** 早期"重训后 inference 恶化"的根因
> 就是合成渲染器(8s+8 窗 / 裁剪 ink)与 C 端真实窗口(medw+8=6s+8)
> 不一致;现以 C 端 dump 真实 glyph 为训练源。

### 5.2 数据源

- **`data/corpus_eval`**(seed 2002,冻结,1380 张,660 realistic):
  只给 bench/gate,永不训练;
- **`data/corpus_train`**(seed 1001,3000 张):只给训练;
- 两条语料由 `tests/gen_corpus.py` 确定性生成(固定 seed/噪声种子),
  可位级复现;GT 全 ICAO 合法;含 clean + realistic(rot±3°、
  blur、光照梯度、反白、噪声≤0.05)两剖面。

### 5.3 训练脚本

| 脚本 | 用途 |
|---|---|
| `tools/train_cnn_real.py` | **生产训练**。`MRZ_OCR_DUMP=1` 跑 C 流水线采集网络真实看到的 16x12 glyph,按图像划分(无泄漏),重建 workdir+清 /tmp 防陈旧 dump 污染 |
| `tools/train_cnn_v2.py` | 向量化合成渲染器(窗口已对齐 6s+8),用于 `--mix-synthetic` 补覆盖 |
| `tools/export_hard_slices.py` | 导出 eval 上 CNN 认错的真实 16x12 patch + GT 标签到 `.npz` |
| `tools/mine_failures.py` | 失败聚类(条件/列/混淆对)+下轮建议 |

`train_cnn_real.py` 参数:
```
--corpus data/corpus_train   # 训练语料(绝不能是 eval)
--out src/cnn_weights.h
--epochs 60  --val-frac 0.15
--mix-synthetic N            # 混入合成样本/类
--mix-hard-slices PATH.npz   # 混入认错切片(30% 权重实验,见 §7)
--max-images N               # 子采样(快速冒烟)
```

网络结构 torch 定义与 C 端完全一致(conv1=8,conv2=16,conv3=4,
fc1=64,fc2=37);Adam + CosineAnnealingLR;best-val 早停;输出
float32 C 头 `src/cnn_weights.h`。

### 5.4 当前权重

`src/cnn_weights.h` = e9d6b31a(`train_cnn_real.py` 于 corpus_train
真实 dump 训练,MD5 提交一致)。**未经授权不重训**:凡改预处理
(deskew 插值/归一化)必先重训以保持分布匹配,且以 full-match 为
交付验收。

---

## 6. 评测与回归门

### 6.1 指标(eval corpus,唯一验收口径)

- `line1_acc / line2_acc`:逐字符正确率;
- `full_match`:两行同时完全正确(产品级联合指标,门禁绑定);
- `cksum_valid`:line2 通过全部 5 个 ICAO 校验位的比例;
- `ok_rate / ms_p50 / ms_p95 / ok_ms_p95`。

### 6.2 门禁(`tests/bench.py --gate data/baseline.json`)

- 绝对下限(FLOORS,仅 CNN):ok_rate 98%、line1 95%、line2 95%、
  full-match 85%、cksum 90%;
- 相对回退>0.3pp 失败;通过后 `--update-baseline` 刷新 `data/baseline.json`。
- **当前短板**:full_match 71.81% < 85% 下限(realistic 45.76%)——认知为
  未达标待攻克。

### 6.3 运行

```sh
./run.sh              # build + 单测 + golden + 示例
./loop.sh             # build→语料→ctest→bench→gate→mine(闭环)
python3 tests/bench.py --corpus data/corpus_eval \
    --gate data/baseline.json --update-baseline
ctest --test-dir build  # 4/4: mrz_ocr_unit / mrz_postproc / mrz_golden / face_unit
```

---

## 7. 运行期开关(gate 一览,默认状态=生产一致)

| env | 默认 | 作用 |
|---|---|---|
| `MRZ_OCR_STRICT_ICAO` | 1 | 严格 ICAO 语法掩码(0=宽松) |
| `MRZ_OCR_TTA` | off | ±1px logits 平均(A/B:对 full-match 复用无正收益,当前不推) |
| `MRZ_OCR_DUMP` | off | 落 16x12 glyph PPM(训练/诊断) |
| `MRZ_OCR_TIMING` | off | 阶段耗时打印(decode/resample/...) |
| `MRZ_OCR_TOPK` | off | 每字符 top-3+概率(候选窗口诊断) |
| `MRZ_OCR_BILINEAR` | off | 双线性 deskew(opt-in;对现有 NN 权重改分布,需配套重训) |
| `MRZ_OCR_LOCAL_NORM` | off | 局部对比度归一化(opt-in;字符↑但 full-match↓,不推) |

---

## 8. 关键决策与局限记录(avoid 重复踩坑)

1. **不做选项 2**(重写纯传统 NCC+质心+动态掩码引擎)——CNN 3~5ms 已
   无算力瓶颈,维护双后处理=双份债;传统只留作对照。
2. **full-match 由"候选集合"决定,不由"候选排序"决定**:top-k oracle
   显示 high-rot/模糊 patch 上 GT 常不在 top-3(例 col2 GT=H,top3=K/M/O)。
   NORM/hard-slices/重训 improves 单字准确率,却不移动 full-match——因为
   它们优化排序,不是把 GT 请回候选。
3. **合成 realistic 的 rot/blur/illum/noise 叠加是否等价真实扫描未验证**;
   当前 45%→85% 的缺口主要来自"GT 出候选"的 patch 质量 loss。
4. **真实护照验收仍未做**。next(需授权):真机样本 + top-k oracle,
   判断候选缺失在真机是否同样严重,再决定走特征级(高分辨率/多尺度)
   还是解码级(重排/beam)。
5. **性能**:CNN p50 6.4ms > 传统 4.9ms(后处理/切分略重);嵌入式
   (RK3588)若需压时间,优先查 deskew 扫描 + pitch-grid phase 全枚举。

---

## 9. 质量与验证状态

- ctest 4/4;`mrz_golden` 进程内渲染 ICAO 记录双后端逐字符复现。
- eval 实测(见 `bench_results.md` / `metrics_history.md`):
  CNN 95.75/98.87 / 71.81 / 94.64;p50 6.4ms,node 100%。
- 分剖面:clean full 95.69%、cksum 100%;realistic full 45.76%、
  cksum 88.79%。

## 10. 文件地图

| 路径 | 职责 |
|---|---|
| `src/mrz_ocr_cnn.c` | CNN 流水线(定位/deskew接入/切分/解码/后处理) |
| `src/cnn.c` `include/cnn.h` | ConvNet 推理(批量、堆分配、conv unroll) |
| `src/mrz_geom.c` | 共享几何预处理(估斜/旋转/局部归一化) |
| `src/mrz_ocr.c` | 传统 LEGACY 后端 + 共享 band/lines/Otsu |
| `src/cnn_weights.h` | 生产权重(train_cnn_real.py 生成) |
| `src/template.c/.h` | OCR-B 37 字形库 |
| `tools/train_cnn_real.py` | 生产训练(真实 dump 分布匹配) |
| `tools/train_cnn_v2.py` | 向量化合成渲染器(窗口 6s+8) |
| `tools/export_hard_slices.py` | 认错切片导出(微调闭环) |
| `tools/mine_failures.py` | 失败挖掘/下轮建议 |
| `tools/gen_mrz_image.c` | 渲染器(rot/blur/illum/invert/noise-seed) |
| `tests/gen_corpus.py` | eval/train 双语料确定性生成 |
| `tests/bench.py` | bench + 回归 gate + failures 导出 |
| `tests/test_golden.c` / `test_postproc.c` | 进程内回归测试 |
| `loop.sh` / `run.sh` | 闭环 / 冒烟 |
| `docs/CNN-OPTIMIZATION.md` | 优化历程/决策记录(本文件的互补) |
