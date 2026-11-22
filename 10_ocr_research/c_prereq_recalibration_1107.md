# C 前置重标定立项与并行会话交接（2026-11-07）

| 项 | 内容 |
|---|---|
| 状态 | **已立项；本轮只做只读协调与方案冻结，不修改 C 源码、不启动重训、不跑新 A/B** |
| 用户指令 | 2026-11-07「好的。继续 1+2+3」之 ① |
| 项目强制流程 | 已加载 `.workbuddy-ai/skills/ai-research-skills-mrz/SKILL.md`，并按附录 C 读 `0-autoresearch-skill` / `rigor-reviewer` / `research-manager`；`AGENTS.md` 与训练大纲 v1.2 §0/§2/§7/§8/§9/附录 C 已核对 |
| 唯一真值 | `10_ocr_research/gt_review_loop.xlsx` 的 `GT对比` sheet |
| 唯一生产权重基线 | `40951c5`（2026-11-05 人工签字；见 `gray_training_outline_1105.md` §11 #11） |
| 现行计时/Exact 裁决 | `AGENTS.md`「计时、Exact 与权重口径裁决（2026-11-07）」：整链 `algorithm_ms`；`exact ≥45/47` 为 both exact；`<100 ms` 看整链 P95 |
| 关联任务 | Task #10（本立项）、Task #11（口径/红线同步）、Task #12（42% 证据与台账）；正式实施任务另立并保持 pending |

---

## 1. 立项理由：P95 缺口与 Exact 缺口同源

桌面 50 张正式 bench 已证明两阶段 `MRZ_BAND_PICK_GATE=1` 在不改权重、不改识别结果的前提下把整链
`mean/median/P95` 从 `193.5/174.3/375.1 ms` 压到 `99.3/64.8/215.8 ms`，`<100 ms` 从 `0/50` 升到
`35/50`；三引擎 raw GT 恒为 `cnn 37/47 / crnn 28/47 / fuse 37/47`，`PASS=40/50`、`verified=48`、
`fallback=0`（`band_candidates_product_port.md`「2026-11-06 落地验证」）。

但剩余 15 张 ≥100 ms 中 9 张是 FAIL；10 张 FAIL 里只有 `2026-9-30-H.jpg`（64.1 ms）走快路径；
`Photo/3–8` 六张全 FAIL 且 129.5–445.8 ms。`Photo/8` 单张 `_ocr_pick_band` 占 456.7 ms，
其中 `ResidentTool.run` 17 次合计 207.8 ms、`mb.band_candidates` 72.0 ms。慢分支不是「候选太多」
这一独立问题，而是**宽 band / 顶部暗区 / 黑边导致行分割错位，触发全候选搜索仍读不对**。
继续压缩候选预算只会把失败隐藏，不会把 exact 从 37/47 提到 45/47。

训练侧已判定收敛：8 项杠杆全负或中性，渲染 A/B 被 C 前置阻塞（物理语料 exact 1/300），
逐列 oracle 上界 39/47 < 45/47。因此唯一有证据支持的方向是 **C 前置（band 定位 → 行切分 →
二值化 → 网格输入）对真实光学灰度的整体重标定**，而不是再训一轮 CNN/CRNN。

---

## 2. 并行会话协调结论（只读检查，2026-11-07 09:32）

- `git status --porcelain -- 6_mrz_ocr/src 6_mrz_ocr/include` 为空：当前**没有未提交的 C 源码改动**。
- 11-06 日志 §5 曾记录「`mrz_ocr_cnn.c` 有并行会话在改，立项前需协调」；工作区干净不等于并行会话已结束，
  实施前仍须重新 `git status`、认领任务、按文件粒度登记改动范围。
- 工作区存在未跟踪的 `8_android/.kotlin/`、`9_face_app_android/` 等 Android 侧目录，但它们**不是 C 源码冲突**；
  本立项不触碰这些目录，也不把设备侧 `e0fe137` 的 pad-alternate 改动当成已验收基线。
- `tests/gui.py` 当前干净且已包含 `MRZ_BAND_PICK_GATE`（`tests/gui.py:5534-5547`、门控逻辑 `:5672-5688`），
  说明门控已入库；本立项不回退、不重写该实现。
- **本轮未修改任何 `6_mrz_ocr/src/**` 或 `6_mrz_ocr/include/**` 文件。**

### 2.1 纠正路径误记

早前文档写成 `6_mrz_ocr/src/mrz_geom.c::mrz_ocr_split_lines`，全库检索确认 `mrz_geom.c` 无此符号。
真实实现与调用点：

| 角色 | 位置 |
|---|---|
| 两行 splitter 定义 | `6_mrz_ocr/src/mrz_ocr.c:273` `int mrz_ocr_split_lines(const uint8_t *bin, int W, int H, mrz_ocr_rect_t *out)` |
| 头文件声明 | `6_mrz_ocr/include/mrz_ocr.h:260` |
| CNN 主链路调用（Otsu / Sauvola / 单行回退） | `6_mrz_ocr/src/mrz_ocr_cnn.c:4047-4080` |
| Python 侧 band 垂直裁剪 | `tests/gui.py:3723` `_rst_native_mrz_band` |
| Python 侧候选选择与两阶段门控 | `tests/gui.py:5487` `_ocr_pick_band` |

`mrz_ocr.c:283-290` 已经用 `max_text_dark = 0.75*W` 排除近全宽黑边；`:300-325` 用
`gap_thr = peak/6` 与两段高度比筛选分裂点；`:326-355` 明确记录「line1 少一行」缺陷**必须和整体重训一起发布**。
`mrz_ocr_cnn.c:4053-4066` 则记录了 Otsu 行高偏短导致半格错位，以及「按行高早停 Sauvola 重切」
已被驳回（official both 40/41→36/41），并指出正确方向是**在网格对齐得分已知后再重切**。
这两处是立项的硬证据，不是待猜测的假设。

---

## 3. 研究问题与边界

**主问题**：在不引入 fallback、不新增进程、不破坏 TD3/CS-CD 的前提下，能否把 C 前置重标定为
「真实灰度 → 正确两行裁剪 → 与 40951c5 同分布的 12×16 cell」的原子发布单元，使
`Photo/3–8` 与其余难例同时满足 both exact 与整链 P95？

**范围内**：

1. 用只读 dump 量化 `row_dark`、黑边、gap、Otsu/Sauvola 行框、垂直 pad、网格对齐得分与 12×16 cell；
2. 每次只改变一个前置假设，并与**对应整体重训的权重集**作为同一原子臂评估；
3. 全量门禁、误差归因、负结果归档、大纲 §11 回写。

**范围外（禁止夹带）**：

- 不再做类别重采样 / 重加权 / `--class-boost` / `mix-hard-slices` / `--pn-weight`；
- 不改 37 类、44×2 或 30×1 网格、syntax mask、校验位 DFS、标准 Beam / LM / CTC；
- 不启用 Tesseract / Paddle / Indra / TD3 / 多假设 rescue 等 fallback；
- 不把 `MRZ_BAND_PICK_PAR=8`、压缩候选预算、字段级 voting、局部对比度归一化当作优化项；
- 不在 C 前置尚未定稿前启动新一轮 CNN/CRNN 重训。

---

## 4. 强制 skill 五要素（入口 skill §7 / 大纲附录 C.3）

### 4.1 上游 skill 与方法论

| 上游 skill | 借用要点 | 本项目落地方式 |
|---|---|---|
| `0-autoresearch-skill/SKILL.md` | 双循环、实验前锁定协议、确认性 vs 探索性结果、负结果也是进展、协议先于结果提交 | 每个 Hn 先写 `protocol.md` 并落库，再跑 A/B；结果无论正负都归档 |
| `22-agent-native-research-artifact/rigor-reviewer/SKILL.md` | 证据相关性、可证伪性、范围校准、基线充分性、单变量隔离 | 宣称「P95 与 exact 同源」必须同时列出 FAIL/慢样本重叠；不得用 cksum valid 冒充 exact |
| `22-agent-native-research-artifact/research-manager/SKILL.md` | 决策、死胡同、pivot、用户/AI 来源标记、append-only | 当日日志追加；`PAR=8`、行高早停、渲染 A/B 等负结果登记为 dead end |

上游库以 LLM 基础设施为中心，**无 OCR / 小 CNN / MRZ 专用 skill**；CTC、LoRA、分布式训练、
GPU serving、标准 Beam / LM 重排一律标注「不适用」。本项目只借研究流程与证据审查方法。

### 4.2 架构兼容性判定

| 维度 | 判定 |
|---|---|
| 固定 44×2 / 30×1 网格 + 37 类逐字符 Cross-Entropy | **适用**：C 前置只决定送进网格的 cell，不改输出维度；但改动会改变输入分布 |
| 约束解码（ICAO syntax mask + line2 校验位 DFS） | **已实现**：不作为待新增项，也不得用来掩盖定位错误 |
| 字段位置加权 Loss / Adam + CosineAnnealingLR | **已实现 / 默认关闭**：不纳入本立项 |
| CTC / Seq2Seq / blank / LM / 标准 Beam | **不适用**：固定网格无长度漂移、无 blank、端侧纯 C |
| 整体重训契约 `mrz_ocr_cnn.c:869-873,3261` | **高风险、必须遵守**：窗口宽度、训练权重、ink gate 是同一套标定；前置改动必须与权重整体上线，禁止分阶段混用 |

### 4.3 单变量 A/B 设计与门禁

**基线臂 A**：当前 C 前置 + `40951c5` 权重 + `tests/gui.py` `GATE=1 / PAR=4`；
正式入口固定为：

```bash
/opt/homebrew/bin/python3 tests/photo_direct_bench.py \
  --methods cnn,crnn,fuse \
  --dirs tmp/PassPortPhoto,Photo \
  --resident \
  --json /tmp/bench_<arm>.json
```

**实验臂 B**：一个且仅一个前置假设 + 该假设对应**整体重训的 CNN/CRNN 权重集**。
「前置 + 权重」在本项目是一个**原子变量**，不允许「只改 C 前置不换权重」或「只换权重不改前置」；
固定项：GT、50 张图集、`real-frac`、seed、epoch、评估脚本、三引擎方法集、`resident` 模式、
`GATE=1`、`PAR=4`、`fallback=0`。设备侧另跑同 APK / 同权重集 / 同 pad 配置对照，避免把
`e0fe137` 的 pad-alternate 收益混入。

建议的假设顺序（**须逐个独立立项，不得合并**）：

| Hn | 唯一假设 | 主要观测 | 证伪条件 |
|---|---|---|---|
| H0 | 只做诊断 dump，不改生产代码 | `Photo/3–8` 的 row_dark / 黑边 / 行框 / cell 与 GT 字符错位图 | 若失败不集中在行框/二值化，则整个立项前提不成立 |
| H1 | `_rst_native_mrz_band` 垂直 pad 与 C 行框交互导致暗区入框 | 行框 y/h、grid alignment score、both exact | 行框修正后 `Photo/3–8` 仍全 FAIL，或 passport / TD3 回退 |
| H2 | `mrz_ocr_split_lines` 的 `peak / gap_thr / 高度比` 在宽 band 黑边下选错分裂点 | 分裂点分布、Otsu vs Sauvola 行高、line1 exact | 与 H1 同时改动（违反单变量）、或 official both 40/41 回退 |
| H3 | 在网格对齐得分已知后选择 Otsu / Sauvola 重切（对应 `mrz_ocr_cnn.c:4060-4066` 的方向） | 对齐得分、char/exact、P95 | 重切增加候选/进程、触发 fallback，或重复已驳回的「按行高早停」 |

每个 Hn 都必须先记录协议，再跑实验；结果分「确认性 / 探索性」标记。

**发布门禁（全绿才可发布，任何一项失败即驳回该臂）**：

| 门禁 | 阈值 / 证据 |
|---|---|
| 真实照片 Exact | `both exact ≥45/47`，`Photo/3–8` 六张全 PASS；line exact / full match / cksum valid 分列 |
| 三引擎 | `cnn,crnn,fuse` 均报 mean / P95 / exact；`cnn/fuse` 不低于基线 37/47，`crnn` 不低于 28/47 |
| 逐字符硬门槛 | passport ≥98%、pic ≥95%；`gt_review_loop.xlsx` 唯一真值 |
| 延迟 | 整链 P95 <100 ms，`<100 ms` 覆盖 50/50；速率按整链 mean ≥2.0 张/秒 |
| 无回退 | `fallback=0`、TD3 零退化 `fixed=0/broke=0`、CS/CD 30 字符不回退、官方 PASS gate `assets/pic/90b464ea218a9909_0001.jpg` 不丢失 |
| 单测 | `test_postproc` 147/147、`test_mrz` 89/89 |
| 误差与 false accept | 混淆矩阵、false accept 不升；不得以 cksum valid 上升代替 exact |
| 归档 | 原始 JSON / dump、A/B 配置、数值、结论（含负结果）写入 `10_ocr_research/` 与当日 memory；大纲 §11 回写 |

**当前已知基线（2026-11-06/07）**：`PASS 40/50`、`verified=48`、`fallback=0`、
`cnn/fuse 37/47`、`crnn 28/47`、`mean/median/P95=99.3/64.8/215.8 ms`、`<100 ms=35/50`、
`test_postproc 147/147`、`test_mrz 89/89`。官方 PASS gate 已有 line2 pos22 `F→O` 风险，
两臂都须复核，不能因「门控前后都 FAIL」而当作已解决。

### 4.4 翻车清单核对

明确**不重复**：`--balance-real` 37/41→28/41、`--class-weight` 40/41→32/41、
`--class-boost=750` 40/41→39/41、`--mix-hard-slices` full-match 不动、
局部对比度归一化 45.0%→37.9%、字段级 voting "NOT safe to adopt"、
按行高早停 Sauvola 重切（40/41→36/41）、`MRZ_BAND_PICK_PAR=8`（P95 215.8→270.6 ms）、
物理渲染 A/B（P×phys 24.58%、前置 exact 1/300）、多引擎仲裁（oracle 39/47 < 45/47）。
本立项不引入类别先验、不新增候选预算压缩、不改解码器。

### 4.5 红线影响评估

- **可能受益**：真实照片 both exact、整链 P95、pic 逐字符（当前均未达标）。
- **高风险**：C 前置改输入分布会使 `40951c5` 权重与 ink gate 失配；必须整体重训 + 全量批测，
  禁止新旧权重混用、禁止分阶段上线。
- **必须保持**：`FAST_NATIVE_MAX_WIDTH=1600`、`fastPair` 双线性 `filter=true`、TD3 默认 2×44、
  pic L1 `real-frac=0.70`、CS/CD 30 字符、单引擎直出无 fallback、唯一真值 `gt_review_loop.xlsx`。
- **计时口径**：按 `AGENTS.md` 2026-11-07 裁决，只用整链 `algorithm_ms` 宣称达标；
  单引擎 ~10–15 ms 仅诊断。速率按 mean，单次 <100 ms 按 P95，二者并列。

---

## 5. 交接与实施约束

1. 实施者启动前重新执行 `git status --porcelain -- 6_mrz_ocr/src 6_mrz_ocr/include`；
   若有并行会话改动，先协商文件边界，禁止整文件覆盖式提交（11-06 memory 已被覆盖过一次）。
2. 在共享任务列表认领正式实施任务；每日 memory **append-only**，不整文件 Write。
3. 先做 H0 只读 dump；没有出现「行框/二值化错位为主因」的证据，不得进入 H1–H3。
4. 每个 Hn 的协议、原始结果、门禁表、负结果都必须落库；不得只报有利指标。
5. 本立项不修改 C 源码、不训练权重、不宣称达标。产物仅为本文、任务交接与口径/证据归档。

## 6. 未决事项

- 官方 PASS gate 的原始定义与本轮 `F→O` 单字符错如何裁决；
- 设备侧 `e0fe137` pad-alternate expansion 的确切参数与 42% 口径（见 `device_pic_42pct_evidence_1107.md`）；
- pic 权重集仍缺 CRNN 头，「融合」名不副实，但按训练侧结论不先补头；
- `Photo/3–8` 六张为同一证件，误差归因须按证件分组，避免对一份证件做记忆；
- H1–H3 的具体重训成本与 APK/桌面权重同步方式，由实施者在协议阶段锁定。
