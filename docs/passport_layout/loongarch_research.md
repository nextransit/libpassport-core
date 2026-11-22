# LoongArch（龙芯）架构下的 Passport 算法优化研究

> 状态：research / proposal。不直接修改 master。
> 目标硬件：LoongArch64 龙芯 Linux 目标机（用户已有 `loongarch64-linux-gnu`
> 交叉配置，见 `/Users/zhouyong/wy/dcrf32/config_loongarch64.py`）。
> 本机验证链：Homebrew LLVM 21（`/opt/homebrew/opt/llvm/bin/clang`）支持
> `loongarch64-unknown-linux-gnu` 目标、`la464/la664` CPU、LSX/LASX 向量化，
> 可交叉编译 + `llvm-objdump` 汇编级验证（无龙芯模拟器/目标机可运行）。

## 0. 一句话结论

把我们在 AArch64 上验证过的三件套（**bit-exact 循环重排 + scratchpad + 双核并行**）
原样移植到龙芯，约等于白拿（架构无关，纯 cache/访存优化）；
要**进一步**超过 AArch64 表现，靠龙芯的 **LSX/LASX 向量化 + 硬加速**（`fmadd.s`
单精度 FMA、`vfmadd`/`xvfmadd` 向量 FMA）+ **LA664 顺序双发射**特性。关键约束：
**不能破坏 bit-exact 门禁**——LSX 累加顺序 ≠ 标量串行累加，需逐算子 host 验证
argmax 0 翻转 + 1380 图 5 项指标 0 回退（与 AArch64 时完全相同的纪律）。

## 1. 龙芯硬件特性（推断 + 官方规格）

| 属性 | 值 | 对优化的影响 |
|---|---|---|
| ISA | LoongArch64（64-bit RISC，类似 MIPS/ARM 混合）| 寄存器多：32 GPR + 32 FPR/128-bit 向量 |
| 微架构 | la464（3A5000）→ la664（3A6000）| 3A5000 顺序双发射，3A6000 3-4 发射乱序，IPC 明显更高 |
| SIMD | **LSX**（128-bit 向量，2 条指令=128b）\| **LASX**（256-bit，4 条指令=256b）| 类似 ARM NEON / AVX2；LASX 为 3A6000 标配 |
| 标量 FP | `fmadd.s` / `fmadd.d`（单条 FMA）| 编译器默认会 FMA-fuse（`-ffp-contract=fast`）|
| 内存序 | little-endian，Cacheline 通常 64B | 对齐 64B 与 AArch64 相同 |
| 特性 | 无 big.LITTLE（同类核集群，龙芯 3 系多核同构）| 双核并行 + 绑核策略与 A53 一致 |

## 2. 可直接白拿的架构无关优化（先做，零风险）

这些与 ISA 无关，是纯 cache/访存/并发层面的改进，已在 AArch64 端全部验证过
bit-exact，移植到龙芯不需要重新推倒：

| # | 优化 | AArch64 端实测 | 龙芯预期 |
|---|---|---|---|
| G1 | **GEMM 循环序**（`cnn_fc_batch`：权重外层、row 内层，权重读一次）| p50 -9.9% | 相近（访存主导，架构无关）|
| G2 | **scratchpad bump allocator + 双半区**（消除逐图 malloc/free，半区隔离供双核）| host p50 ~8ms 稳定 | 相近 |
| G3 | **双行并行 CNN forward**（line0/line1 各一核，半区无竞争）| 冷态 -18%，热态 -10% | 3A5000/6000 同构多核 → 稳定 -20% 以上（无 big.LITTLE 节流问题）|
| G4 | **常驻 worker + condvar 握手**（消除每图 pthread_create/join 抖动）| p95 -9.4% | 相近（p95 收益同样成立）|
| G5 | **worker 绑定核 + 主线程不绑**（防迁移抖动；A53 实测 pin main 反优化）| p50 -4.2ms | 相近，需同 A/B 纪律重测 |

> 风险最低、收益确定性最高：**先把 G1-G5 在龙芯上按 AArch64 相同的代码路径交叉编译、
> 目标机跑 1380 图门禁**，确认 5 项指标 0 回退即可落地上线。

## 3. 龙芯特有的新优化（收益 > AArch64 的点）

### 3.1 LSX/LASX 向量化 `cnn_fc_batch` / `conv` 内层（重点 + 已实测）

**本机 LLVM 21 交叉编译实测证实的结论（见 §附录A）**：

- **CNN（cnn.c）在 `-mlasx` 下自动向量化极强**：`-O3 -march=loongarch64 -mlasx`
  直接生成 `xvld(316)/xvst(228)/xvfcmp.clt.s(152)+xvbitsel.v(144)/xvfmadd.s(48)`
  等 256-bit LASX 指令；纯标量版本则是 `fld.s(923)/fmadd.s(112)/fsel(429)`。
  **即：龙芯 LLVM 21 的 auto-vectorizer 已能把 fc/conv 内层向量化到 LASX，
  不需要手写 SIMD 就能拿到主要收益**。这与 AArch64/A53 的处境类似——我们
  在 A53 上放弃 NEON 就是因为 `-O3` 已自动向量化、手写无边际收益。龙芯上
  **先用 `-mlasx` 跑基线，再决定要不要手写**，手写仅在自动向量化漏掉的
  寄存器分块（如 fc2 4-输出并行 + 权重行复用）时才值得。
- **注意 FMA 引入**：自动向量化会把 `x=a*b; acc+=x` 熔成 `xvfmadd.s`（向量 FMA）。
  这与标量 `-ffp-contract=fast`（`fmadd.s`）行为一致 —— **门禁只要求"龙芯编译
  的 FMA 语义与 host 基线一致"**。bet：确保两端 `-ffp-contract` 相同，否则必须
  用分离式 `vfmul+vfadd` 复现两次舍入。
- **geom 自动向量化失败**：`mrz_geom.c` 在 LASX 下几乎无向量指令（`fld.d/st.b`
  为主）——因为 skew 投影是动态分支（`if(!row[x]) continue`）、deskew 依赖
  `sin/cos`，**LLVM 无法自动向量化 → geom 需要手写 LSX/LASX 或算法级改造**
  （见 §3.2）。这是龙芯上唯一必须手写 SIMD 的热点。

**具体方案（保持 bit-exact 纪律）**：
- 方案 A（零手写，先测）：`-march=loongarch64 -mlasx -O3` 编译 → 目标机跑
  1380 图门禁。若 0 回退，直接收下自动向量化收益（白拿）。
- 方案 B（手写）：`#include <lsxintrin.h>/<lasxintrin.h>`，对 fc2 做
  LOSX 分离式 `__lsx_vfmul_s + __lsx_vfadd_s`（若 A 引入 FMA 漂移则改回
  分离式复现标量两次舍入；对应 AArch64 的 vmulq+vaddq 纪律）。
- 评估：1380 图 l1/l2/full/cksum/ok 0 回退 + argmax 0 翻转，才切默认。

### 3.2 几何前处理（deskew / skew 投影）——龙芯上的新机会

AArch64 上 band+deskew 占 p50 52%，是最大痛点，但**stride/减角/并行三次尝试全部
回退**（0/1 稀疏图 stride 破坏投影排序、corpus 含 >±2° 偏斜）。

龙芯上用 **LASX 直接向量化现有标量循环**（不动算法逻辑、不改 winner 选择）：

- `mrz_skew_score_at` 内层（每角度对 band_bin 做投影累加）：band 高 ~50 行，
  每行宽 ~900px。内层是 `if (!row[x]) continue; proj[yi]++`——**动态索引投影**，
  向量化需要先把 proj[yi] 累积改成"每行先判 ink 再按位移投影"的向量友好形式。
  **重点不是矢量宽度而是减少分支**：用 `vld` 一次读 16/32 字节 ink 掩码，
  提前知道哪些像素是 ink（AArch64 上同样的改动没做成，因为 NEON 收益被 A53
  顺序核吃掉；龙芯 3A6000 LASX + 更强乱序 → 预期可 +30~50%）。
- `mrz_rotate_band` 双线性 deskew：**数学上是对称的**（dst 每个像素采样 src），
  是**纯读 src 写 dst 的无依赖循环** → LASX 可 8-lane 平滑；且 AArch64 我们
  没动它是因为 CNN 输入分布不能变（`MRZ_OCR_BILINEAR` 默认关）。**若保持
  bilinear=0（最近邻）则纯拷贝语义，可 64B 对齐 SIMD memcpy 化**。

### 3.3 硬 FMA 吞吐（3A6000 双 128b FMA/周期）

3A6000 的 `fmadd.s` 是单周期双发射 FPU。CNN fc 内层 `s += a*b` 若保持标量，
编译器 `-ffp-contract=fast` 会 fma-fuse 成单指令 → 每周期 2 个 FMA。**在确认
host 基线同样 FMA 的前提下，龙芯标量 `-O3` 已能把 fc 内层压到接近向量水平**，
手写 SIMD 的边际收益主要来自 LASX 宽度（8 float/指令 vs 1）。

## 4. 编译器 / 构建参数（LoongArch 特有）

```sh
# 本机交叉编译验证（无目标机也能做汇编级检查）
/opt/homebrew/opt/llvm/bin/clang -target loongarch64-unknown-linux-gnu \
    -mcpu=la664 -march=la64v1.1 \
    -O3 -ffp-contract=fast -fno-fast-math \
    -funroll-loops -ftree-vectorize -mllvm -enable-constraint-elimination \
    -I6_mrz_ocr/include -I6_mrz_ocr/src -I3_face_recognition/include \
    -c 6_mrz_ocr/src/cnn.c -o /tmp/cnn_loong.o

# 汇编级检查：确认 LSX/LASX 指令真的生成
/opt/homebrew/opt/llvm/bin/llvm-objdump -d /tmp/cnn_loong.o | rg -i "vfmul|vfadd|fmadd"
```

关键 flag：
- `-mcpu=la464`（3A5000）或 `-mcpu=la664`（3A6000）：选择微架构调度表。
- `-march=la64v1.1`：显式开 LSX/LASX（等价 `lsx`/`lasx` 扩展）。
- `-ffp-contract=fast`：默认让标量 fma-fuse（**需确认与 host 基线一致，否则
  破坏 bit-exact**）。若 host 是 `-O3 -fno-fast-math`（我们 passport 主 build 就是），
  龙芯端也必须保持一致，SIMD 手写用分离式 `vfmul+vfadd` 复现标量两次舍入。

## 5. 针对 passport 三个热点的逐点实施清单

### 5.1 CNN forward（`6_mrz_ocr/src/cnn.c`）
| 热点 | 现状 | 龙芯方案 | 门禁 |
|---|---|---|---|
| fc1/fc2 内层点积 | GEMM 循环序（G1，已 bit-exact）| 加 LASX 8-lane 分离式 FMA 变体，env 开关 A/B | 1380 图 5 项 0 回退 + argmax 0 翻转 |
| conv1/conv2/conv3 | 标量 `-O3` | LSX 向量化（3x3 卷积 8/16 通道可分 lane）| 同上 |
| softmax | 标量 expf | 可不变（占比小）| — |

### 5.2 几何前处理（`6_mrz_ocr/src/mrz_geom.c`）
| 热点 | 现状 | 龙芯方案 | 门禁 |
|---|---|---|---|
| skew 投影 | 标量逐像素分支投影 | LASX 向量 ink 掩码 + 行投影（不动 winner）| coarse winner 不偏移 |
| rotate_band（bilinear=0）| 标量最近邻 | 64B 对齐 SIMD 拷贝 | 字节一致 |
| rotate_band（bilinear=1）| 标量双线性 | LASX 8-lane 无依赖采样 | 浮点位一致（难），需 A/B |

### 5.3 并发/调度（`6_mrz_ocr/src/mrz_ocr_cnn.c`）
| 热点 | 现状 | 龙芯方案 |
|---|---|---|
| 双行并行 | 常驻 worker + 绑核（G3-G5）| 原样；龙芯同构多核无节流 → 收益更稳 |
| scratchpad | 8MB 双半区 bump allocator | 原样；64B 对齐 cacheline 兼容 |

## 6. 验证链（无龙芯目标机也能推进）

1. **本机交叉编译**：LLVM 21 `-target loongarch64` 编译全部 C 源 → `.o`。
2. **汇编级确认**：`llvm-objdump -d` 检查 SIMD/FMA/对齐。
3. **数值等价（host 侧可跑）**：龙芯端与 AArch64 端**用同一份 C 源码**，
  但龙芯编译器的浮点行为（是否 FMA-fuse、舍入）可能不同。**真正确认 bit-exact
  只能靠目标机**跑 1380 图门禁。本机只能做到：
   - 同一源码在 host（AArch64）跑出基线 5 项指标；
   - 龙芯 `.o` 反汇编与 host `.o` 反汇编对比，确认未引入 FMA 语义漂移。
4. **目标机回归（用户侧）**：部署到 3A5000/3A6000 机器，跑 `tests/bench.py`，
   比对 5 项指标 + p50/p95。**这是唯一权威门禁**。

## 7. 风险登记

| 风险 | 缓解 |
|---|---|
| 龙芯编译器 FMA 行为与 host 不同 → bit-exact 被破坏 | 先对比两边反汇编；`-ffp-contract` 两端强制一致；SIMD 用分离式 |
| LSX/LASX 收益取决于目标函数：CNN 自动向量化已生效（`-mlasx` 白拿），geom 自动向量化失败（须手写）| 分模块评估：CNN 先测 `-mlasx` 基线，geom 另走手写 SIMD 路径 |
| 无龙芯目标机 → 无法跑权威门禁 | 本机做汇编级验证 + host 数值等价；门禁权交用户目标机 |
| 3A5000（la464）与 3A6000（la664）性能差异大 | `-mcpu` 分开调优；默认 la664 兼容 la464 |

## 8. 实施顺序（建议）

1. **Phase A（1 天，零风险）**：G1-G5 原样交叉编译到龙芯 + 汇编级检查 +
   host 数值等价确认 → 白拿 AArch64 已验证的收益。
2. **Phase B（1-2 天）**：CNN 先用 `-mlasx` 自动向量化基线跑门禁（A 方案，
   已实测 LLVM 自动生成 LASX，先收货）；仅在门禁被 FMA 漂移卡住或想要更多
   时，手写分离式 LSX/LASX（B 方案，`MRZ_CNN_LOONG_SIMD=1` env 开关）。
3. **Phase C（2-3 天）**：几何前处理手写向量化（skew 投影 ink-mask + rotate
   双线性——自动向量化已实测够不着，必须手写 LSX/LASX），
   目标 3A6000 上 band+deskew 压 30-50%。
4. **Phase D（门禁）**：用户目标机 1380 图回归 + p50/p95 记录（与 AArch64 相同
   双指标纪律）。

## 9. 结论

- **白拿层（G1-G5）**：龙芯上直接复用 AArch64 已验证的 bit-exact 优化，零风险，
  收益约同 AArch64（冷态并行 -18%）。
- **超额层（LSX/LASX + FMA，已实测）**：分两半看——
  - **CNN**：LLVM 21 `-mlasx` 自动向量化**已成功**（实测 `xvld/xvst/xvfmadd.s`
    大量生成），比标量版大幅减少 FP 指令。**先用 `-mlasx` 白拿，手写边际收益小**。
  - **几何前处理**：自动向量化**够不着**（动态分支 skew 投影 + `sin/cos` deskew，
    实测几乎零向量指令）→ 这里手写 LSX/LASX 才是**真正的差异化收益点**，
    预期 3A6000 上 band+deskew 压 30-50%。
- **唯一硬门禁**：bit-exact 5 项 0 回退 + argmax 0 翻转，必须目标机跑。
  本机 LLVM 21 支持交叉编译 + objdump 验证，可先把"编译/汇编/数值等价"
  全部提前在本机做完，把目标机只留最终回归。


---

## 附录A：本机 LLVM 21 龙芯交叉编译实测（证明 §3.1）

命令（Homebrew LLVM 21.1.6，`loongarch64-unknown-linux-gnu`，无系统 libc，
用最小 stub 头编译合法 C 源，仅作指令级观测）：

```sh
/opt/homebrew/opt/llvm/bin/clang -target loongarch64-unknown-linux-gnu \
    -march=loongarch64 -mlsx -mlasx -O3 -ffp-contract=fast -fno-fast-math \
    -nostdinc -I/tmp/la_stub -I6_mrz_ocr/include -I6_mrz_ocr/src \
    -I3_face_recognition/include -I3_face_recognition/src \
    -c 6_mrz_ocr/src/cnn.c -o /tmp/loong_build/cnn.o
/opt/homebrew/opt/llvm/bin/llvm-objdump -d cnn.o   # 观察指令
```

`cnn.o`（`-mlasx`）指令直方图（top）：
```
 544 ori     316 xvld     309 add.d     228 xvst      201 fld.s
 152 xvfcmp.clt.s  144 xvbitsel.v  133 xvldx   109 lu12i.w
  80 xvldrepl.w     73 fcmp.clt.s    64 fmadd.s   48 xvfmadd.s
```
`cnn_scalar.o`（无 `-mlasx`）直方图（top）：
```
 923 fld.s    465 fcmp.clt.s   429 fsel     221 fst.s
 112 fmadd.s    61 fmul.s
```
`mrz_geom.o`（`-mlasx`）直方图（top）—— 几乎无向量指令：
```
  94 move     79 addi.d     55 ld.d      46 st.d      39 ori
  33 nop      24 b         21 pcalau12i  21 fld.d     19 st.b
```

解读：
1. CNN 在 LASX 下自动向量化成功（`xv*` 256-bit 指令大量出现），标量 FMA 也被
   向量 FMA 取代 → **`-mlasx` 是 cnn.c 的白拿优化，先测门禁**。
2. geom 几乎没有向量指令 → 自动向量化够不着动态分支/三角函数的循环；
   **geom 是龙芯上唯一需要手写 SIMD 或算法重构的热点**。
3. `-mlasx` 需要目标机 3A6000 才完全发挥；3A5000 只支持 LSX（128-bit），
   用 `-mlsx`（不带 `-mlasx`）以兼容。

## 附录B：工具链可用性核对（2026-10-10 本机）

- `clang`（Apple 17）：`--print-targets` 无 loongarch，但 `-target
  loongarch64-unknown-linux-gnu -fsyntax-only` 通过（可做语法/类型检查）。
- `/opt/homebrew/opt/llvm/bin/clang`（LLVM 21.1.6）：完整支持 loongarch64
  codegen（可交叉汇编 cnn/geom/template 全部 C 源）。
- `-march=loongarch64 -mlsx -mlasx` 可正常生成 LSX/LASX 指令；`-mcpu=` 不做为
  LoongArch 目标识别（用 `-march`+`-mlsx/-mlasx` 控制特征）。
- 无龙芯模拟器（qemu-loongarch64）无法本机运行；权威门禁须用户侧 3A5xxx/3A6xxx
  目标机跑 `tests/bench.py`。
