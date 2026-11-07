# H0 只读诊断 dump — Photo/3–8 三引擎失败归因（2026-11-07）

| 项 | 内容 |
|---|---|
| 状态 | **H0 已执行（只读，未改 C 源码/权重/生产代码）**；结论触发立项 H0 证伪条件（对 exact 目标） |
| 依据 | `c_prereq_recalibration_1107.md` §4.3 H0、Task #13；强制 skill 已加载 |
| 复现入口 | `/opt/homebrew/bin/python3 tests/photo_direct_bench.py --methods cnn,crnn,fuse --dirs Photo --resident --json h0_dump_1107/bench_photo_1107.json` |
| 样本事实 | `Photo/3、4、5、6` 四个目录为**同一文件**（md5 `efeaf0b6…75c2`），实际独立样本 = 3 张（_003、_007、_008） |

## 1. 复现结果（产品路径，GATE=1，resident）

- 官方路径 6/6 **FAIL**；`loc=6/6`（band 定位全部命中）；逐引擎字符准确率 cnn `94.7/81.4`、fuse 同 cnn、crnn `82.6/69.3`；GT both exact **0/6**。
- 每张 reason 均为「融合CNN+CRNN 识别并通过校验」而 grade=FAIL：reason 是引擎态（`tests/gui.py:6202`），FAIL 来自 GT 对比（`tests/gui.py:6224`）。

## 2. 关键对照实验（同一 band，两个头）

从产品链截获 `_ocr_pick_band` 后实际喂引擎的 native band（`h0_dump_1107/h0_band_003.png`，191×2392），直接喂 `mrz_ocr_cnn_tool`：

| 运行 | line1 | line2 | conf(l1/l2) |
|---|---|---|---|
| GT | `POCHNJIA<<LIMIN<<<…` | `EE20426536CHN7410024M2808303LMNGMBKCMDPBA970` | — |
| **v2 line heads**（`MRZ_OCR_LINE_MODELS=1`，产品默认） | `PBCHNHIA<<LIMIN…` | `EE20H26535CHN…LMNOMBKE80EBA910` | 94/98 |
| 旧整体头（无该 env） | `POCHNJIA<<KIHIN…K…` | `EE20426536CHN…GMBKENG9BA970` | 96/96 |

- v2 运行与 bench 产品读数**逐字一致**（003 的 `PBCHNHIA/EE20H26535…` 即 `raw.cnn/raw.fuse`）→ 复现闭环成立。
- **两头的 cell 窗口几何完全相同**：diag JSON（`h0_band_diag_003.json` vs `_003l.json`）l2 均 `y72 h38 pitch46.25 cell42 win_w42 L0 47 seg_score93`；44 个窗口 x/w/h 逐一对齐（0 差异）。输入相同、输出不同 → 失败在**头部权重对退化字形的泛化**，不在 band/行框/二值化/cell。
- v2 在错误读数上置信度**更高**（98 > 96）。
- 错误读数**校验位自洽**：`EE20H2653→ck5`、`LMNOMBKE80EBA9→ck1`、composite 全过（7/3/1 复算）。即无 GT 时该错误读数可被当作合法 MRZ 接受 —— false-accept 形态。
- 旧头在这些样本上也**不 exact**（_003 l1 3 错 + l2 尾 4 错），仅失败形态更温和；故换回旧头不是解。

## 3. 字形证据

- `h0_dump_1107/h0_zoom_cells.png`：错位 cell（l2 pos4 `4`、pos9 `6`、pos31 `G`、pos35 `C`、pos38 `P`）放大 20×——字形退化（低对比、笔画发虚）但人眼可辨；v2 头分别读成 `H/5/O/E/E`。
- `h0_cells_l2_v2vsold.png` / `h0_cells_l1_v2vsold.png`：44 格全列对照（上=GT/预测，`*`=错位）。
- 各引擎失败分布：v2 头错误集中于**证件号位（4→H、6→5）与尾部 personalize 区**；_007（640×480 低分辨率）l1 姓名区崩坏（`POCHC<FA<<UCMIN`）；_008 证件号 `EE20426536→E220626536/EE20626336`。crnn 全部远差于 cnn/fuse（l2 conf 70–73）。

## 4. 整图直喂 C 工具的附加发现（非产品路径）

整张照片直接喂工具时，内置定位对 **PASS 样本同样失败**（`2026-9-29-A.jpg`：band 高 662/960、ERR_BAD_LINES；_003 误判 layout=CS_CD_30、行框高 822px）。说明 C 内置定位本就不吃整页照片，产品路径的 Python band 层（`_ocr_pick_band`）才是定位主力；该层在 6 张上 loc 全中。→ 整图直喂的失败不构成 exact 缺口的证据。

## 5. H0 结论与对立项的影响

1. **证伪条件触发**：立项 §4.3 H0 证伪条件为「若失败不集中在行框/二值化，则整个立项前提不成立」。实测 Photo/3–8 的 exact 失败**不集中在行框/二值化**（行框健康、cell 两头一致且可辨），而在 **v2 line heads 对该证件退化字形的分类域**。→ **C 前置重标定（H1–H3）不得以「救 Photo/3–8 exact」为由进入实施**。
2. 立项中仍然成立的部分：**P95/耗时段**（`_ocr_pick_band` 全候选搜索慢、`Photo/8` 456.7ms）与 exact 无关，该半边前提不受影响。
3. 「P95 缺口与 Exact 缺口同源」的表述应按本 dump 修正为：同因（定位搜索慢）仅解释耗时；exact 缺口在这六张上是识别域问题。

## 6. 后续候选方向（均未启动，需单独立项）

- **候选 A（新立项）**：v2 line heads vs 旧整体头在真实退化字形上的域失配研究——单变量 A/B 需覆盖官方 41/47 全集（v2 当时 38/41 优于旧 37/41，全局换回会丢既有收益；按域门控选择头部属仲裁类，触「不引入 fallback/仲裁」红线，须用户书面批准后另行立项）。
- **候选 B**：归档本报告后回写 `c_prereq_recalibration_1107.md` §3/§4.3 与大纲 §11（H0 已执行+证伪记录）。
- 不重复已驳回路线：矫正图喂 OCR（2/72 vs 自然 crop 3/72，`tests/gui.py:3304-3311`）、字段 voting、多引擎仲裁、候选预算压缩等。

## 7. 产物清单（`10_ocr_research/h0_dump_1107/`）

| 文件 | 内容 |
|---|---|
| `bench_photo_1107.json` | 产品路径 6 张逐引擎原始记录 |
| `h0_band_003.png` / `h0_band_008.png` | 产品链实际喂引擎的 native band（191×2392） |
| `h0_band_diag_003.json` / `_003l.json` / `_008.json` | band 直喂 C 工具的诊断 JSON（旧头 / v2 / 008） |
| `h0_diag_003/007/008.json` | 整图直喂 C 工具的诊断 JSON（附录 4 证据） |
| `h0_cells_l1_v2vsold.png` / `h0_cells_l2_v2vsold.png` | 44 格 cell 全列对照（v2 vs 旧头，GT/预测标注） |
| `h0_zoom_cells.png` | 错位 cell 20× 放大 |
