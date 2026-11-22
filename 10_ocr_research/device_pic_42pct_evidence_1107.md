# HEAD `e0fe137` 的 `pic 4.7%→42%` 证据归档与口径裁决（2026-11-07）

> 本文用于补齐 `e0fe137` 提交信息 `device pad-alternate expansion (pic 4.7%→42%)`
> 缺失的结果文件。结论：**42% 可复现，但不是与旧 4.7% 同分母、同图集、同配置的验收提升，
> 不得据此宣称 pic 达标或计入验收进度。**

---

## 1. 原始证据

| 项 | 值 |
|---|---|
| 归档文件 | `10_ocr_research/device_pic_result_1107.txt`（从 `/tmp/device_pic_result.txt` 原样复制） |
| 原始文件 sha256 | `243897cc5e0da3be748d0d07f9f8f1b167d0e0baf9e621cf3bb8f7c3312aa902` |
| 行数 / 字节 | 872 / 114440 |
| 采集时间 | 2026-11-07 08:55:43–08:55:55 CST（logcat 时间戳） |
| 设备命令 | `am instrument -w -e class com.nextransit.mrzbench.PhotoMrzDeviceTest -e methods auto -e parallel 1 -e weightSet pic -e imageFilter pic com.nextransit.mrzbench.test/androidx.test.runner.AndroidJUnitRunner` |
| 测试自检 | `TEST start images=65 filter='pic'`、`TEST methods=[auto]`、`TEST weightSet=pic`、`MRZ_WEIGHTS_SET=pic`、`TEST parallel=true`、`MRZ_OCR_PAR ON` |
| 结果行 | 65 条 `RESULT external/pic/...` |

原始日志中可见 `framing candidates: primary rejected, trying 5 alternates` 与
`framing candidate 0 (pad=1.35)` 等记录，对应提交信息中的 pad-alternate expansion；
本轮没有重跑设备、没有修改 APK，只归档与复算。

## 2. 65 条 RESULT 的直接统计

| 指标 | 数值 |
|---|---|
| 结果行 | 65 |
| `verdict` | PASS 39 / FAIL 23 / SUSPECT 3 |
| `decode=true` | 42 / 65 |
| `wall` | p50 77 ms / p95 207 ms / max 268 ms |

**`verdict PASS` 与 `decode=true` 都不是 both exact**，不能用来替代准确率。

## 3. `42%` 是怎么算出来的

用 `tests/gt_assets_pic_official.json` 的 `gt` 字典按图片 basename 比对 `L1` / `L2`：

```python
import json, re
from pathlib import Path
root = Path('.')
g = json.loads((root/'tests/gt_assets_pic_official.json').read_text())['gt']
pat = re.compile(
    r'RESULT external/pic/([^ |]+).*?\| L1=(.*?) \| L2=(.*?) \| err=')
rows = []
for line in (root/'10_ocr_research/device_pic_result_1107.txt').read_text(
        errors='replace').splitlines():
    if 'RESULT external/pic/' not in line:
        continue
    m = pat.search(line)
    if m:
        name, l1, l2 = m.group(1), m.group(2).strip(), m.group(3).strip()
        if name in g:
            rows.append((l1 == g[name]['l1'], l2 == g[name]['l2']))
print(len(rows),
      sum(a for a,b in rows), sum(b for a,b in rows),
      sum(a and b for a,b in rows))
# 50 25 37 21
```

| GT 来源 | 可比对张数 | line1 exact | line2 exact | both exact |
|---|---:|---:|---:|---:|
| `tests/gt_assets_pic_official.json` | **50/65** | 25/50 = 50.0% | 37/50 = 74.0% | **21/50 = 42.0%** |
| 同上，仅两行 GT 均 44 字符 | 49 | 24/49 | 37/49 | 21/49 = 42.9% |
| `tests/gt_assets_pic.json` | 50 | 8/50 | 19/50 | 5/50 = 10.0% |
| `tests/photo_auto_gt.json` | 50 | 24/50 | 28/50 | 20/50 = 40.0% |

65 条结果里只有 **50** 个 basename 能匹配 `tests/gt_assets_pic_official.json`；
另外 15 张（14 FAIL / 1 SUSPECT）不在该 GT 子集，未进入 42% 分母。
可比对 50 张的 `verdict` 是 PASS 39 / FAIL 9 / SUSPECT 2。

## 4. 与旧 `4.7%` 为什么不能直接相比

旧证据 `10_ocr_research/device_bench_1105_result.txt`：

| 引擎 | pic 分母 | line1 exact | line2 exact | both exact | verdict PASS |
|---|---:|---:|---:|---:|---:|
| CNN | 64 | 3/64 | 13/64 | **3/64 = 4.7%** | 21/64 |
| FUSE | 64 | 3/64 | 13/64 | **3/64 = 4.7%** | 21/64 |
| AUTO（产品路径） | 64 | 3/64 | 16/64 | **3/64 = 4.7%** | 24/64 |

差异至少有三重：

1. **分母与图集不同**：旧读数 64 张；新日志跑 65 张，仅 50 张可比对官方 GT 子集。
   若把今早的 21 张 both exact 强行放到 64 张分母上，是 21/64 = **32.8%**，不是 42%。
2. **配置不同**：旧文件头写 `产品路径(PhotoMrzPipeline, PAR=1)`、`HEAD 40951c5(gated)`；
   新日志是 `methods=[auto]`、`weightSet=pic`、`parallel=true`，并启用 pad-alternate framing
   candidates，提交 `e0fe137` 明确把它记为一次设备侧改动。
3. **口径不同**：`42%` 是 both exact；`4.7%` 也是 both exact，但旧表同时列出
   `verdict PASS 21/64` / `24/64`、`decode 28/64` / `31/64` 与 `cksum 39~44/64`。
   评审报告必须分列，不能把 `verdict` / `decode` / `cksum valid` 当成 Exact。

因此 `4.7%→42%` 只能改写成：

> 旧 2026-11-05 设备产品路径：`pic both exact 3/64 = 4.7%`；
> 今早 2026-11-07 设备 `auto/pic` + pad-alternate 路径：在有官方 GT 的 50 张上
> `both exact 21/50 = 42.0%`。**两者不是同分母、同图集、同配置的 A/B，不能相减为 +37.3pp，
> 也不能作为 pic ≥95% 的验收证据。**

## 5. 验收结论

- `AGENTS.md` 唯一硬门槛仍是**逐字符** passport ≥98% / pic ≥95%，唯一验收真值是
  `10_ocr_research/gt_review_loop.xlsx` 的 `GT对比` sheet；本文的官方 GT json 仅用于诊断复算。
- `42%` 远低于 pic ≥95%；按 64 张分母换算的 `32.8%` 同样未达标。
- HEAD 提交信息 `pic 4.7%→42%` 不应计入验收进度；`e0fe137` 的设备 pad-alternate 改动
  仍需按 `10_ocr_research/c_prereq_recalibration_1107.md` 的单变量 A/B 与全量门禁重新评估。
- 本轮只归档原始 logcat、复算口径并写入台账标识，**没有重跑设备、没有发布新权重、没有改 C 源码**。
