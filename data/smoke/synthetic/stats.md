# 合成冒烟数据统计（不是真实实验结果）

- 样本数：72
- 图片数：24
- 场景组数：9
- 成对组数：9
- 已复核 / 未复核：0 / 72

## 按任务

| 项 | 数量 |
| --- | --- |
| checklist | 48 |
| presence | 24 |

## 按来源

| 项 | 数量 |
| --- | --- |
| synthetic_smoke | 72 |

## 按 split

| split | 样本 | 图片 | 场景组 | 成对组 |
| --- | --- | --- | --- | --- |
| train | 48 | 16 | 6 | 6 |
| val | 15 | 5 | 2 | 2 |
| test | 9 | 3 | 1 | 1 |

## 类别 × 状态（按图片，每张图计一次）

| 类别 | present | absent | unknown |
| --- | --- | --- | --- |
| screwdriver（螺丝刀） | 18 | 5 | 1 |
| pliers（钳子） | 16 | 7 | 1 |
| cable_bundle（线缆） | 13 | 10 | 1 |
| multimeter（万用表） | 20 | 3 | 1 |
| hammer（锤子） | 19 | 4 | 1 |

## 类别 × 状态（按样本，同一张图的多条问法会重复计数）

| 类别 | present | absent | unknown |
| --- | --- | --- | --- |
| screwdriver（螺丝刀） | 54 | 15 | 3 |
| pliers（钳子） | 48 | 21 | 3 |
| cable_bundle（线缆） | 39 | 30 | 3 |
| multimeter（万用表） | 60 | 9 | 3 |
| hammer（锤子） | 57 | 12 | 3 |

## 条件标签

| 标签 | 样本 | 图片 |
| --- | --- | --- |
| background_dark | 27 | 9 |
| background_gray | 24 | 8 |
| background_wood | 21 | 7 |
| baseline | 27 | 9 |
| defocus | 3 | 1 |
| lighting_cool | 24 | 8 |
| lighting_dim | 18 | 6 |
| lighting_mixed | 9 | 3 |
| lighting_warm | 21 | 7 |
| occlusion | 9 | 3 |
| partial_frame | 3 | 1 |
| position_shift | 3 | 1 |

## 困难负样本标签

| 标签 | 样本 | 图片 |
| --- | --- | --- |
| pliers_vs_metal_tool | 12 | 4 |
| probe_leads_vs_cable | 9 | 3 |

## 清单结论（checklist 样本，由冻结的 derive 规则推出，不是模型输出）

| 项 | 数量 |
| --- | --- |
| complete | 16 |
| incomplete | 29 |
| unknown | 3 |
