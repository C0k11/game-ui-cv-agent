# 走格子答案库（我们的格式）

由 `scripts/convert_baah_grid.py` 从 `data/baah_grid_solution/`（BAAH 原始, MIT）
转换生成。**运行时只读这里**（`routing_v2/flow/grid.load_answer`），原始件只读不动。
重新转换: `py -X utf8 scripts/convert_baah_grid.py`（覆盖式, 手改无效）。

## 格式

    {"stage": "2-5", "type": "normal", "source": "BAAH (MIT), sol=0",
     "teams": [{"name": "A", "attr": "any", "pos": "center"}],
     "rounds": [[{"team": "A", "do": "move", "dir": "right-down"}], ...],
     "needs": {"teams": 1, "portal": false, "exchange": false, "attrs": []},
     "alts": [ ...同形备选解法（可换路线）... ]}

- `attr` 是队伍属性要求（blue=神秘 red=爆发 yellow=贯穿 purple=振动 any=任意），
  和 `type` 的 normal/hard 是两回事。`pos` 是部署方位（8 向），`dir` 是移动
  方向（六边形 6 向 + center），两套词汇不通用。
- `do` 属 move / exchange / portal。exchange=点完格子再点交換按钮确认换位;
  portal=点完格子等传送确认弹窗（BAAH GridQuest.py 的实测行为）。
- `needs` 给 flow 进关前预检: 能力不够 BLOCKED 在花 AP 之前。

## 语义勘误（2026-08-13）

BAAH 原始文件的数字键是**按目标(3star/challenge/gift)区分的备选解法**
（官方 DATA/grid_solution/grid_solution_format.json），不是多区域顺序打。
转换取主解法（属性全 any 里键最小的, 没有就键最小的），其余进 `alts`。
真正的中途重新部署（H1-2 双区域地图）是游戏自己弹部署屏，flow 部署后
继续同一份 rounds。

## 能力版图（转换器每次运行都会打印）

单队纯 move 33 关（1-3 章 + H1-H3）；portal 最早 4-1 / H4-1；
多队与属性队最早 6-1；exchange 最早 6-4。
