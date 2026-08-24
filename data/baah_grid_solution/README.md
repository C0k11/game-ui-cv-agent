# BAAH 集中指挥走关解法（第三方数据，MIT）

来源: https://github.com/BlueArchiveArisHelper/BAAH  `DATA/grid_solution/quest/`
许可: MIT (Copyright (c) Sanmusen Wu)，全文见 `LICENSE.BAAH`。
取得: 2026-08-13，共 240 份（normal 150 / hard 90）。

禁本目录是**只读原始件**，运行时不再直接读这里 --
`scripts/convert_baah_grid.py` 转换成 `data/grid_answers/`（我们的格式），
`grid.load_answer` 只认那边。数字键的真实语义是**按目标区分的备选解法**
（官方 grid_solution_format.json），不是多区域，详见 grid_answers/README.md。

## 我们用它的哪一部分

只用 `fight_plan` 的**方向序列**和 `initial_teams` 的**方位**，这些是分辨率无关的
语义（`right-down` 说的是"往右下那一格"，不是某个像素）。

    {"task_location":1, "task_level":2, "task_type":"normal",
     "requires": {"0": ["any"]},
     "0": {"initial_teams":[{"name":"A","type":"any","position":"center"}],
           "fight_plan":[[{"team":"A","action":"move","target":"right-up"}],
                         [{"team":"A","action":"move","target":"right"}]]}}

action 三种: move / exchange / portal。target 是 6 个方向 + center
（格子是**六边形**，所以没有 up/down）；`initial_teams[].position` 另有 8 个方位，
那是部署槽的方位，和移动方向不是一套。

## 不用的部分

`initial_teams[].click` 的硬编码像素（只出现在 11 个文件、19 处）一律忽略。
BAAH 自己的走格子识别是 HSV 阈值 + 手写 kmeans + `WALK_MAP` 写死 ±115px 偏移，
换服/换光照就要重调，我们不抄那一层 —— 方向到落点的映射改成:
**检出屏上所有格子，从当前队伍出发按相对角度挑出目标方向那一格，落点取它的框心。**
另外每一步都验证到达态，BAAH 是盲走。
