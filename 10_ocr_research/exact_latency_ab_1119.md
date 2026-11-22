# 50 张照片 exact / 耗时 A/B 与根因裁决（2026-11-19）

## 结论先行

在 **HEAD `634b278`**（当前工作区，含并行会话 11-17 的 `crnn_weights_cscd.h`
改动与 11-19 09:18 重编的 C 工具）上，50 张照片正式 bench 的基线为：

- 官方路径 PASS **46/50**，verified 50，整链 **mean 82.0 / median 38.1 / P95 395.2 ms**；
- 三引擎 raw GT both-exact：**cnn 36/47、crnn 43/47、fuse 36/47**（目标 ≥45/47）；
- 单引擎直出耗时 ~17 ms（诊断口径，非验收口径）。

**P95 395.2 ms 的唯一来源**是 3 张 CS/CD 港澳台通行证（单行 30 字符）走了
`crnn -> tesseract-fallback`（300–888 ms）。这同时违反用户「严禁 fallback」硬约束。
**exact 的 4 张缺口**（U / AG / H / Photo-8）是逐字符分类错误（0↔O、6↔G、
姓名音译字符），非 band 错位。

三臂单变量 A/B 证明：**在不修改 C 源码 / 权重（Task #13 范畴，且并行会话正在改
`crnn_weights_cscd.h`）的前提下，Python 侧没有同时保住 exact 与消除 fallback 的无损解。**
继续试开关会重复 `gray_training_outline_1105.md` §5 翻车清单。

## 验收入口

```
/opt/homebrew/bin/python3 tests/photo_direct_bench.py \
  --methods cnn,crnn,fuse --dirs tmp/PassPortPhoto,Photo \
  --resident --json /tmp/bench_<arm>_1119.json
```

口径按 `AGENTS.md`「计时、Exact 与权重口径裁决」：产品路径耗时取
`_ocr_photo_engines` 整链 `algorithm_ms`，P95 判 <100 ms，exact 取 both-exact
分母 47（50 张中 3 张 GT 为 30/0 单行记 bad3）。

## A/B 结果（单变量，同机同批同工具）

| 臂 | 变量 | PASS | fallback 事件 | 整链 mean/median/P95 (ms) | crnn raw exact |
|---|---|---|---|---|---|
| A 基线 | 无 | 46 | 3 | 82.0 / 38.1 / **395.2** | 43/47 |
| B | `MRZ_OCR_CSCD_FLD=1` | **45** ↓ | 2 | 71.1 / 37.1 / **108.7** | 43/47 |
| C | `MRZ_FAST_FUSION=1` + `CSCD_FLD=1` | **39** ↓↓ | 6 | 79.7 / 44.6 / 134.3 | 43/47 |

- 三臂的 `each engine alone`（fallback OFF）GT exact 完全一致（cnn 36、crnn 43、fuse 36 /47），
  说明开关只改 rescue / 引擎调度，不改逐字符分类能力。
- 进程退出时偶发 `recursive_mutex lock failed`（resident 池析构竞态），发生在 JSON 落盘之后，
  不影响数据；A 臂 EXIT=134、B 臂 EXIT=0，数据完整。

## 根因（误差归因先行，skill §8）

### 耗时：3 张 CS/CD 单行 fallback

| 图 | GT(30) | crnn 直读 | 后端 | 官方 ms |
|---|---|---|---|---|
| 2026-9-30-I | CSC733210936<2711209<8612261<8 | CSC733210936<8711209<3512251<2（CK 不匹配） | crnn→tesseract-fallback | 729.9 |
| 2026-9-30-J | CSC482624408<2610106<7410024<0 | 空（OCR 未输出任何行） | crnn→tesseract-fallback | 1004.1 |
| 2026-9-30-K | CSC495643597<2610265<8706216<0 | CSZZ04474597<2610165<3709219<9（CK 不匹配） | crnn→tesseract-fallback | 395.2 |

链条：`is_cscd=True` → `allow_tesseract_fallback=True`（`tests/gui.py:6186`）→
crnn 把单行当 TD3 读失败 → `needs_rescue`（`:4516`）→ `tesseract_rescue()`
（`:4108`，CS/CD 时先 `_rst_tesseract_cscd` PSM 7/6/3，无 candidate 再 `_rst_run_indra_inprocess`
整页 ONNX，`:4118`）→ 300–888 ms。fallback 后的读数与 GT 完全一致，即**结果对、路径违规**。

三引擎（cnn/crnn/fuse）对这 3 张直读均失败，非简单引擎路由可解：
I/K 是字符读错且 top-K 不含真值，J 是整行无输出。属 C 前置/CS-CD 权重问题。

### exact：4 张逐字符分类错误

| 图 | 差异 |
|---|---|
| 2026-9-29-U | line2 5 字符（6↔8、G↔0、B↔8、O↔0、1↔8） |
| 2026-9-30-AG | line1 `GX`→`OA`，line2 2 字符 |
| 2026-9-30-H | line1 10 字符（整行错位读取） |
| Photo/8 | line1 1 字符（I↔Z） |

均为 37 类逐字符分类边界错误，归因到 band/切分/权重，非解码策略。

## 上游 skill 适用性（skill §7 五要素）

- 适用：`22-agent-native-research-artifact/rigor-reviewer`（证据等级与「只报有利项」审查）、
  `research-manager`（负结果与死胡同归档）、`0-autoresearch-skill`（单变量迭代纪律）。
- 不适用：CTC / LoRA / 分布式 / GPU serving 类框架 skill —— 本项目是固定网格 + 逐字符
  Cross-Entropy（skill §3 硬过滤）。
- 架构兼容：本轮只改 Python 侧 GUI 交互与环境变量 A/B，未触碰 C 分类头 / Loss / 解码，
  与 §3 架构事实无冲突。
- A/B 设计：单变量（CSCD_FLD、FAST_FUSION 各自独立），固定工具/语料/GT/方法集/分母。
- 翻车清单：未重复 `--balance-real` / `--class-weight` / 局部归一化等已失败方案；
  `CSCD_FLD` 的 false accept（臂 B）与融合臂 PASS 崩塌（臂 C）作为新负结果登记。
- 红线：fallback 未移除（移除即 3 张 FAIL）；`FAST_NATIVE_MAX_WIDTH`、TD3、CS/CD 解码
  未改；权重未动。当前 HEAD **违反禁 fallback 红线**，须由 C 前置修复（Task #13）。

## 下一步（不在本轮授权内）

1. **Task #13（C 前置重标定）**：让 C 引擎在 CS/CD 单行上直读通过校验，方可移除
   `allow_tesseract_fallback=is_cscd` 并消除 P95 尖峰。并行会话已在改
   `crnn_weights_cscd.h`，须先协调避免冲突，再整体重训 + 全量批测（禁止新旧权重混用）。
2. exact 4 张走 `collect_confusions.py` / `mine_failures.py` 误差归因后补样重训，
   同属 Task #13 的整体重训臂。
3. 重训后复验门禁：both-exact ≥45/47、Photo/3–8 全过、整链 P95 <100 ms、<100 ms 50/50、
   fallback=0、TD3 零退化、`test_postproc` 147/147、`test_mrz` 89/89。

## 产物

- `/tmp/bench_baseline_1119.json` / `.txt`（臂 A）
- `/tmp/bench_cscd_fld_1119.json` / `.txt`（臂 B）
- `/tmp/bench_fusion_fld_1119.json` / `.txt`（臂 C）
