# gui.py→fuse 50 张 exact / 耗时深度优化（2026-11-20）

## 结论先行

目标：exact 50/50、整链 p50 ≤110 ms、p95 <200 ms、严禁 fallback、优化尽量落
C 侧（可移植、与板端一致）。

**达成**：
- **exact 50/50**（both-exact，含 3 张 CS/CD bad3；官方路径 PASS 50/50）
- **p50 36.3 ms** ✓（目标 ≤110）
- mean 88.3 ms（产品速率门槛 ≥2.0 张/秒 ✓，折合 11.3 张/秒）
- fuse raw（引擎 alone）**36/47 → 44/47**；crnn raw 43/47 不变（TD3 零退化）
- 门禁：`test_postproc` 147/147、`test_mrz` 89/89、pic 集逐字节零回归

**未达成**：p95 539.8 ms（目标 <200）。尾部 7 张全部是**C 层真值缺失**类
（见「p95 结构性墙」），只有 Task #13（C 前置重标定 + CS/CD 头重训）能打开。
本轮尝试了全部不重训的可行性方案（9 个臂），负结果一并归档。

## 修改清单

| # | 文件 | 内容 | 类型 |
|---|---|---|---|
| C1 | `6_mrz_ocr/src/mrz_ocr_cnn.c` | **双头合并候选池**（`MRZ_OCR_DUAL`，默认关）：fuse 模式下逐格合并 CNN/CRNN top-K（product-of-experts `max(pc,ε)·max(ps,ε)^α` 归一化，ε=1e-4，α=`MRZ_OCR_DUAL_ALPHA` 默认 1.0），束搜索/字段 DP/复合校验全部走合并池；dual 开启时跳过旧 runner-up arb（避免重复计票） | C |
| C2 | `tests/mrz_engine_pool.py` | `get_bandpick_tools` 增加 `fuse` 参数（bandpick_fuse 槽位，与生产 fuse 引擎同头） | Python |
| P1 | `tests/gui.py` | bandpick CS/CD 候选改用 **CNN 头**评分（`crnn=not _is_cscd`，原硬编码 crnn=True 与 docstring 意图相悖） | Python |
| P2 | `tests/gui.py` | DUAL bandpick：primary 须过**纯 CRNN 佐证**（dual==crnn）才提前返回；冲突→全候选投票，候选入选资格=双头一致 | Python |
| P3 | `tests/gui.py` | cond_agree（两处）加**空读防线**：两引擎同读空串不再视为「一致直出」 | Python |
| P4 | `tests/gui.py` | phase-2 移除 easyocr（50 张中它从未成为最终读数来源，却是墙钟 647-1320ms） | Python |
| P5 | `tests/gui.py` | GUI 默认启用 dual（`os.environ.setdefault("MRZ_OCR_DUAL","1")`、`ALPHA=2.0`）；C 侧默认保持关闭，Android 待设备 bench 再翻转 | Python |

## A/B 轨（单变量，同机同批同 GT，`photo_direct_bench.py --resident`）

| 臂 | 变量 | PASS | exact(50) | mean/median/p95 (ms) | fuse raw | 备注 |
|---|---|---|---|---|---|---|
| A 基线 | 11-19 工作区 | 49 | 50* | 233.8/42.9/1022.1 | 36/47 | *exact 统计修正后（见下） |
| B | C1 α=1.0 | 50 | 50 | 151.9/37.5/562.4 | 40/47 | Photo/3-6 直出 |
| C | C1 α=1.5 | 50 | 50 | 131.7/34.3/539.1 | 42/47 | U 全对（α≥1.5） |
| D | C1 α=2.0 + P1/P2(无一致性)/P3 | 50 | 50 | 118.3/35.4/538.4 | 44/47 | W/AJ/P7 直出；K 仍 335 |
| E | +P2 一致性投票 + phase-2 门 | **48** ↓↓ | **48** ↓↓ | 75.7/35.6/322.4 | 44/47 | **翻车**：门被自洽假收骗过 |
| F | E 撤 phase-2 门 | 50 | 50 | 102.6/37.2/546.6 | 44/47 | P8 166ms（投票修复） |
| **G 终态** | F + P4 | **50** | **50** | **88.3/36.3/539.8** | **44/47** | H 274 / AG 328 |

*注：本轮前报（exact_latency_ab_1119.md）的「exact 46/50」是统计脚本把 CS/CD
（GT line2 为空）误判为非 exact 所致；修正口径后 A 基线即 50/50——**exact 的
真实起点是 50/50**，本轮的真实战场是 p95 与 fallback 红线。*

## 根因与机制

### exact（起点已 50/50，防线加固）
- **fuse(36/47) 弱于 crnn(43/47) 的根源**：束搜索/CK-DFS 只跑 CNN probs，
  CRNN 只能在 CNN beam 的 best/runner-up 间投票，无法注入自己的候选。
  Photo/3-8（cnn conf 0.98 错 vs crnn conf 0.99 对）、H L2 尾部（真值
  `<<<72` 只在 CRNN 池）、U L2（CNN/CRNN 前后段互补）全部属于此类。
  C1 合并池 + 复合校验过滤后，4 张 class 全部直读。
- **α=2.0 甜点**：α=1.0 时 H c39（cnn 3:1.0 vs crnn <:0.678）翻不动；
  α≥1.5 U 全对；α=2.0 时 W/AJ/P7 全对且 P8 直出。α 继续增大开始破坏
  AG L0（c15 X 需 cnn 0.433 胜 crnn 0.955）。

### 耗时（p95 1022→540）
1. **13→4 张 8 引擎 rescue**：dual 让 fuse 与 crnn 在 Photo/3-8、W、AJ、
   P7、P8（投票修复带）上一致 → cond_agree 直出（30-176ms）。
2. **bandpick 三 bug**：
   - CS/CD 候选用 CSCD-CRNN 头评分（差头，K 的 fast* 候选 GT conf 89 被
     判无效）→ P1 改 CNN 头（与 C 侧 `MRZ_OCR_CSCD_NET=line2` 默认一致）；
   - DUAL 下候选评分头与生产引擎不一致 → C2/P2（bandpick_fuse 槽位）；
   - primary ck 有效即锁带 → P2 佐证门（P8 的 213px 带 dual 假收
     conf 197 而纯 crnn EXACT；假收家族 dual 9 票 vs GT 3 票，普通投票
     会锁死假收，双头一致性过滤后 GT 候选 pad30/40/res10 胜出）。
3. **phase-2 墙钟**：easyocr（mean 647/p95 1320ms）从未提供最终读数 → P4 摘除。

### 翻车归档（禁止重复）
- **phase-2「fuse 已验证即跳过」门**（臂 E）：AG/H 的 dual 假收校验闭合
  （composite 尾部救援让任何字段自洽读数都 ck 有效）且与带选择获胜读数
  自洽 → 跳过 paddle 裁决 → PASS 50→48。C 层不存在区分 U 型真读与
  AG/H 型假收的信号，phase-2 多引擎仲裁是唯一防线。
- **MRZ_BAND_PICK_ALL 普通投票对 P8 无效**：假收家族 9 票 > GT 3 票，
  必须用双头一致性过滤而非票数。

## p95 结构性墙（<200ms 需 Task #13）

| 照片 | 现状 ms | C 层证据 | 归类 |
|---|---|---|---|
| 2026-9-30-I | 551 | **完整假收**：c26 4:0.919(真值 6:0.003)、c27 3:0.671(1:0.328)，doc/birth/expiry/复合四层校验全部自洽闭合；CSCD_FLD DP 锚定误读 CK 无法到达真值；crnn-cscd 头真值 0.004 | 重训 |
| 2026-9-30-J | 540 | **C 盲**：单行 16px（DPI 185），12 个候选带全部 conf 0，行切分器在 34px 带上切错行框；lanczos×3/4 上采样、白底填充均无效 | 重训+前置 |
| 2026-9-29-U | 587 | 82px 带上 dual 直读 GT，但纯 crnn 头在所有一致候选上与 dual 冲突 → 佐证门拒绝 → phase-2 重确认 | 墙内最优 |
| 2026-9-30-K | 324 | fuse 已直读 GT；324ms 是 crnn 方法 conf 74<80 触发的 tesseract 复核（rescue 已 per-photo 去重） | fallback 红线 |
| AG/H | 328/274 | phase-2 由 paddle 裁决（真值不在任何 C 头候选池）；AG c40 S、H L0 双头皆盲 | 重训 |
| I/J/K 的 fallback | — | `allow_tesseract_fallback=is_cscd`（commit 3410514 引入）在 I/J 上仍是 exact 的唯一来源；**移除即 48/50** | 与红线冲突待裁决 |

**裁决请求**：I/J 当前 exact 依赖 tesseract fallback，与「严禁 fallback」
红线冲突。三条路：(a) 维持现状（2 张违反红线，exact 50/50）；(b) 立即移除
（红线合规，exact 48/50，I 变哑假收风险）；(c) Task #13 重训 CS/CD 头后
移除（两者兼得，需与改 `crnn_weights_cscd.h` 的并行会话协调）。

## 复验入口

```
# 官方口径（不导出任何 env，GUI 默认已启用 dual）
/opt/homebrew/bin/python3 tests/photo_direct_bench.py \
  --methods cnn,crnn,fuse --dirs tmp/PassPortPhoto,Photo \
  --resident --json /tmp/bench_final_1120.json
```

## 上游 skill 适用性（skill §7 五要素）

- 适用：`0-autoresearch-skill`（9 臂单变量迭代纪律）、
  `rigor-reviewer`（exact 统计口径纠错、只报有利项审查）、
  `research-manager`（臂 E 翻车与 P8 投票陷阱归档）。
- 不适用：CTC/LoRA/分布式/GPU serving（skill §3：固定网格+逐字符 CE）。
- 架构兼容：C1 只换束搜索候选源（机制/语法掩码/校验不变）；与
  「窗口宽度、训练权重、ink gate 同一套标定」红线无冲突（未动预处理）。
- 红线核查：TD3 零退化（crnn raw 43/47 三臂不变）、
  `FAST_NATIVE_MAX_WIDTH` 1600 未动、fastPair 双线性未动、CS/CD 30 字符
  解码路径未回退、真值唯一来源 gt_review_loop.xlsx。

## 产物

- `/tmp/bench_baseline_1120.json/.txt`（臂 A）… `/tmp/bench_final_1120.json`（终验）
- `10_ocr_research/_diag7_1120.py` / `_diag_topk_1120.py`（band 截获与逐格 top-K 审计）
- `10_ocr_research/_diag1120_*.ppm`（7+ 难照产品带/clamp 带/official 带存证）
