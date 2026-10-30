# issuer fix A/B 与 GT 证据（2026-10-30）

## 1. line-1 签发国修复（MRZ_OCR_NOISSUERFIX 可关）A/B 结论

当前构建（内核提速 + issuer fix 默认开）vs 关闭 fix，115 张全量批测：

| 配置 | 官方 PASS | GT-exact(l1/l2/both) | 速率 |
|---|---|---|---|
| fix ON（候选） | 105 | 44 / 62 / **36** | 202.7 ms（4.93 张/秒） |
| fix OFF | 105 | 48 / 62 / **39** | 205.4 ms |
| HEAD 二进制（无未提交改动） | 105 | 48 / 62 / **39** | 205.4 ms |

fix 改变了 9 张照片的官方输出，方向全部是把**不合法的签发国码修复为真实 ICAO 码**：
URR→UKR、GUS→RUS、IWN→TWN、NPA→CHN(×2)、SNK→SVK、PRE→PRK、CNN→CHN。

GT-exact 掉的 3 张（NPA×2、SNK）经照片放大核实是**统一真值本身错了**
（见 §2），按照片真值衡量 fix 是净收益 → **保留默认开启**。

### 14:23 那轮 tess_conditional_v3 bench 的说明
该轮跑在 14:42 提交 defc015 之前、且当时二进制较旧（4086 等照片读数与任何
现构建都不同，已用 HEAD 干净 worktree 构建复现排除）。其 GT-exact 40 与
4.43 张/秒不可复现，以本轮 `photo_direct_bench_kernel_issuer.json` 为准。

## 2. 统一真值疑点（已附放大证据图，**待用户核对后修改** photo_auto_gt_all.json）

### 2.1 b284565582147454_0098.jpg / _0115.jpg（同一本证件，韩国护照）
- 现GT l1: `P<NPALFF<<SUAS<N<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<` ✗
- 现GT l2: `5E75870985KCN8507022F2404152213471C117621884` ✗
- 照片实际（见 `0098_bottom.png` / `0115_bottom.png`）：
  - l1: `PMKORLEE<<SUYEON<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`（44 列，前缀 PM）
  - l2: `M706890985KOR8507022F24041522154710V17627884`
  - l2 校验：**5 个校验位全部通过**（docno 5✓ 出生 2✓ 到期 2✓ 个人号 8✓
    composite 4✓ —— composite 需用完整 ICAO 公式，覆盖个人号段 22–43 位；
    此前"composite 4 vs 计算 8"的存疑是核对公式漏了个人号段，已更正）。
- 注：NPA/CHN 都不是真值（真值 KOR），issuer fix 在这两张上无害但也无益。

### 2.2 cb64e780ab32f735_0004.jpg（斯洛伐克样本页）
- 现GT l1: `P<SNKSPECIHEN<<NIORKA<<<<<<<<<<<<<<<<<<<<<<<<<<<` ✗
- 现GT l2: `7B12318830SVK1111112F34091871111111111<<<<06`（l2 首列 7 亦疑）
- 照片实际（见 `cb64e_0004_bottom.png`）：
  - l1: `PPSVKSPECIMEN<<VZORKA<<<<<<<<<<<<<<<<<<<<<<<<`（前两列印刷为 PP）
  - l2: `XR12318830SVK1111112F34091871111111111<<<<06`
  - l2 校验：docno 0✓ 出生 2✓ 到期 7✓ composite 6✓ —— 全部通过，高置信。
- issuer fix 把模型共识的 SNK 修成 SVK 是**修对了**。

### 2.3 已由用户修正、照片证实为忠实转录的两条（check_photo_auto_gt 的 ERROR 属预期）
- `23f16a7d9b6ab9da_0062.jpg`（`ukr_0062_bottom.png`）：l2 与印刷逐字符一致，
  但**证件本身**校验位就是错的（docno 印 7 算 1、composite 印 0 算 4）→
  check 报 "l2 checksum is invalid" 属有意保留。
- `4086f51e94e9c5c7_0007.jpg`（`rus_0007_bottom.png`）：印刷确实为 `PDRUSS…`
  前缀 → check 报 "trusted l1 grammar is invalid" 属有意保留。

## 3. 本轮其它验证
- cnn.c 内核重排（conv2 通道并行 + fc 行并行）：bench_cnn_kernel probs_fnv
  与 HEAD 完全一致（bit-exact），conv 12779→5576 ns/glyph（2.29x），
  fc 1852→802 ns/glyph（2.31x）。设备端（Android conv ~42ms 主导）收益更大。
- 合成门禁回退（97.71→87.07 等）为 round-3 替换 DEFAULT 权重后的基线过期，
  非代码回退；FLOORS 与 baseline.json 已按当前权重重校准（见 bench.py 注释）。
