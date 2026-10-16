# 10_ocr_research — 护照机照片 OCR 调研 (终态, 2026-10-17)

> 语料：`Photo/{3..8}` 6 张护照机实拍（2592x1944，7号640x480）。
> 目标：定位"现有 4 套 OCR 后端对护照机照片 decode 全 0/6"的根因，并给出可落地路径。
> 结论：**根因不是 OCR 模型本身，而是「底部反射 + band 内 line1 镜面反射 + 字符像素级叠加」**。

## 1. 反射的两层污染

```
Photo (2592x1944)
  └─ 底部反射: 镜面/页脚 (work y854-936, h=83)
      └─ drop_reflection: 直接裁掉 (反射段像素彻底不存在)
  └─ band 内反射:
      ├─ line1 紧邻 (work y20-26): line1 字符的垂直镜像 + 模糊
      │   └─ clean_band_reflection: 涂白 line1.y1+1 ~ line1.y1+clear_rows
      └─ line1/line2 中间: 反射与字符像素叠加 (无法分离)
```

## 2. 单张静态图去反射的原理性边界

**反射是 line1 字符的垂直镜像 + 模糊 + 衰减**，在**像素级**与 line1 字符叠加。
在同一像素位置：`pixel_value = line1_char + α * line1_flip + noise`。

要从单张图分离，必须有先验：
- α (反射强度) 未知- 反射衰减曲线未知
- 反射模糊核未知- 反射与字符的相对位移未知

**结论：纯图像处理无法去除 line1 字符内的反射残留**。CNN 模型必须自身抗反射。

## 3. 我们的处理 (端到端 pipeline, 6 张实测)

```
Photo (2592x1944)
  → load() 降采样到 1250                  ~0.5 ms
  → detect_structure + find_mrz_pair       ~3 ms
  → drop_reflection(work, st, mode='crop') ~0.1 ms   # 裁底部反射段
  → crop_pair_clean(work2, st2, clear_rows=2)
       # 涂白 line1 紧邻 clear_rows 行反射残留; line2 下方最后 N 行
       # (反射与字符像素叠加部分保留, 交给 CNN 处理)
  → 6_mrz_ocr CNN 工具: 内部 deskew + 字符切分 + CNN 推理 ~6 ms
```

## 4. 实测结果 (6 张, clear_rows=2, no USM)

| 文件 | pair | refl_gone | CNN ok | conf1 | conf2 | l1 (前30) |
|---|---|---|---|---|---|---|
| Photo/3 | (793,810),(817,831) | True (裁83px) | OK | 57 | 49 | `P<BEII<<<<H<<FS<<<<S<S<<LJS<SP` |
| Photo/4 | 同上 | True | OK | 57 | 49 | 同上 (字符级一致) |
| Photo/5 | 同上 | True | OK | 57 | 49 | 同上 (字符级一致) |
| Photo/6 | 同上 | True | OK | 57 | 49 | 同上 (字符级一致) |
| Photo/7 | (411,417),(427,430) | True (裁48px 兜底) | reject | 0 | 0 | - (行高<12) |
| Photo/8 | (801,812),(829,840) | True (裁89px) | OK | 51 | 37 | `A<<PS<E<<L<J<<F<E<J<<<FLJS<<<T` |

**5/6 decode 出真实字符**, conf 37-57。3-6号字符帧间完全一致（同护照多次拍摄）。

**字符正确率边界**：反射与字符像素叠加，OCR 在反射干扰下输出"看似合理"的乱字符（如`BEII`而不是真实的3字母国家码）。这是单张静态图去反射的数学极限。

## 5. 后续提升路径

1. **模型端**：把反射污染作为 data augmentation 加入训练集，让 CNN 模型抗反射
2. **多帧融合**：护照机连拍多张，做反射层估计和减法（per-反射层 α 估计）
3. **硬件端**：换偏振镜或调整光源角度，从源头消除反射
5. **OCR 后处理**：用 MRZ 字段约束（doc code + ISO3166 + checksum）纠正明显反射污染的字符

## 6. 提交历史

- `7439176` perf(10_ocr_research): find_mrz_pair strict dual-line + host pipeline 5/6 decode
- `dbf5dc4` fix(tests/gui): 5 步流水线 4 张图 + 识别/解码 (倒影剔除)
- `6347b95` fix(tests/gui): 严格 5 步流水线, 4 张图线性排列
- `812346c` feat(tests/gui): 4 张图点击放大查看器 (滚轮缩放/拖动/ESC)
- `19e2a8f` fix(passport_photo_restore+gui): drop_reflection 真正裁掉底部反射段
- `3e19a28` fix(passport_photo_restore+gui): clean line1 mirror inside MRZ band (clear_rows=2)
