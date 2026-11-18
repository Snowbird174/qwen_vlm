# 采集步骤（简版）

目标：先拍约 **100 张试采照片**（20 个布置 × 5 张），标注后上传到服务器。状态判定的细节见 `docs/annotation_guide.md`。

## 第 1 步：架相机、贴检查区域

1. 相机装在三脚架上，**镜头垂直朝下**，横向拍摄，分辨率不低于 1280×960。
2. 用不反光的胶带在台面上贴一圈，作为**检查区域**，整圈胶带都要在画面内。
3. **锁定对焦和曝光**。拍完一组之前不要动三脚架；如果碰了，记下来。

## 第 2 步：每个布置拍 5 张（相机不动）

| 照片 | 怎么摆 | 成对编号 |
| --- | --- | --- |
| v01 | 五件器材都放在检查区域内 | 与 v02 相同，如 `pilotb01_l001_p01` |
| v02 | 在 v01 基础上**只拿走一件**（见下表） | 与 v01 相同 |
| v03 | 重新摆齐，再拿走**两件**（见下表） | 留空 |
| v04 | 五件都在，但每件换个位置 | 留空 |
| v05 | 制造"看不清"：遮挡、失焦或检查区域出画（见第 4 步） | 留空 |

每一张都要**等手离开画面再拍**（v05 故意用手遮挡的除外）。

拿走哪件按下表轮换，避免总是缺同一件。l011–l020 重复 l001–l010 这一轮：

| 布置 | v02 拿走 | v03 缺两件 | 布置 | v02 拿走 | v03 缺两件 |
| --- | --- | --- | --- | --- | --- |
| l001 | 螺丝刀 | 螺丝刀、钳子 | l006 | 螺丝刀 | 钳子、万用表 |
| l002 | 钳子 | 螺丝刀、线缆 | l007 | 钳子 | 钳子、锤子 |
| l003 | 线缆 | 螺丝刀、万用表 | l008 | 线缆 | 线缆、万用表 |
| l004 | 万用表 | 螺丝刀、锤子 | l009 | 万用表 | 线缆、锤子 |
| l005 | 锤子 | 钳子、线缆 | l010 | 锤子 | 万用表、锤子 |

每 5 个布置换一次光照和垫板，作为一个批次（session）：

| 批次 | 布置 | 光照 |
| --- | --- | --- |
| `pilotb01` | l001–l005 | 暖色日光 |
| `pilotb02` | l006–l010 | 冷色顶灯 |
| `pilotb03` | l011–l015 | 侧光 |
| `pilotb04` | l016–l020 | 偏暗 |

## 第 3 步：顺带拍两类容易混淆的情况

- **表笔线 vs 线缆**：万用表的红黑表笔展开放着（表笔算万用表的一部分）。约 4 张只有表笔、没有独立线缆，再约 4 张表笔和线缆都在。
- **钳子 vs 其他金属工具**：放一把扳手当干扰物。在 l002、l007 中，v01 有钳子和扳手，v02 只拿走钳子，扳手不动。

这些照片在标注表的 `hard_negative_tags` 一列分别填 `probe_leads_vs_cable`、`pliers_vs_metal_tool`。

## 第 4 步：v05 怎么制造"看不清"

轮换使用，不要全用手挡：

| 布置 | 做法 | 标注 |
| --- | --- | --- |
| l001 l009 l013 l017 | 用手挡住一件 | 被挡的那件标 `unknown`，其余按实际标 |
| l002 l010 l014 l018 | 用纸或盒子挡住一件 | 同上 |
| l003 l011 l015 l019 | 局部失焦 | 认不出的标 `unknown` |
| l005 l007 l020 | 检查区域一侧出画 | 出画那一侧的器材标 `unknown` |
| l004 l008 l012 l016 | 手入画，但**什么都没挡住** | 全部按实际标，不能出现 `unknown` |

## 第 5 步：存文件

```text
data/raw/<批次>/<布置>/<照片>.jpg     例：data/raw/pilotb01/l001/v02.jpg
```

- 文件名里**不要出现答案**（`missing_pliers.jpg` 之类）。
- 批次、布置、照片编号里**不要有下划线**。

## 第 6 步：填标注表

复制 `data/annotations/templates/annotation_template.csv`，另存为 `data/annotations/pilot.csv`，**一张照片一行**：

| 列 | 填什么 |
| --- | --- |
| session_id / layout_id / variant_id | 如 `pilotb01` / `l001` / `v02` |
| image_path | `raw/pilotb01/l001/v02.jpg` |
| pair_group_id | 只有 v01 和 v02 填，两者相同；其余留空 |
| screwdriver … hammer（5 列） | `present` 在 / `absent` 不在 / `unknown` 看不清 |
| unknown_reason | 有 `unknown` 时必填，如 `pliers=被手挡住` |
| condition_tags | 可选，用 `\|` 分隔，如 `lighting_warm\|occlusion` |
| hard_negative_tags | 第 3 步那两类照片才填 |
| annotation_verified | 复核过填 `true` |

判定口诀：**看得见才填 `present`；检查区域完整且确实没有才填 `absent`；看不清就填 `unknown`，不要猜。**

## 第 7 步：上传并检查

把 `data/raw/` 和 `pilot.csv` 拖进服务器的 `/root/gpufree-data/qwen_vlm/` 对应目录，然后运行：

```bash
cd /root/gpufree-data/qwen_vlm && source env/activate.sh
P=envs/qwen_vlm/bin/python
$P scripts/data_csv_to_jsonl.py --csv data/annotations/pilot.csv --out data/annotations/pilot.jsonl --source real_lab --seed 0
$P scripts/data_validate.py data/annotations/pilot.jsonl          # 报错就按提示修改标注表
$P scripts/data_split.py --input data/annotations/pilot.jsonl --out data/splits/v1 --seed 0 --fail-on-leak
```

## 之后：正式数据

试采能顺利跑通后，按同样的方法扩到 **160–240 个布置、800–1200 张照片、8–12 个批次**，最好分不同的天拍。正式批次用日期命名，如 `20261120b01`。
