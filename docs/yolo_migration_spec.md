# 纯 YOLO 迁移规范 (sub-skill OCR → YOLO cls)

## 背景
BA 自动化 bot。YOLO `ui` 模型能识别 156 个 UI cls（见 `brain/skills/ui_classes.py`）。
用户要求 **所有点击靠 YOLO cls，OCR 只用于纯数字识别**（计数 / AP / 票数）。

## 铁律
1. **找不到 cls 就 `log + action_wait`**，把 gap 暴露出来。**绝不** OCR fallback、绝不 hardcoded 盲点坐标。
2. OCR 仅在两种情况保留：
   - 读**纯数字**（正则如 `\d+/\d+`、AP 数、票数）
   - YOLO 没有对应 cls 的**全屏过场文字守卫**（如战斗中 `AUTO`/`戰鬥時間`）— 这类极少，能用 cls 就用 cls
3. 保留：状态机结构、`should_run`+`dot_on_entry`、`_handle_common_popups`、`max_ticks`、`reset()`
4. 删除：用于找按钮的 `find_any_text`/`find_text_one`/`find_text`/`has_text`/`find_clickable_text`；`detect_current_screen`（改 `detect_screen_yolo`）；`find_ui`（legacy，改 `find_cls`/`click_cls`）；`find_claim_all_button`/`find_single_claim_button`（改 `find_cls`+CLAIM cls）

## base.py 已有的新接口（直接用，签名固定）
```python
self.find_cls(screen, cls_names, *, conf=0.30, region=None) -> YoloBox | None   # exact-match, 最高 conf
self.find_all_cls(screen, cls_names, *, conf=0.30, region=None) -> list[YoloBox] # 全部, conf 降序
self.click_cls(screen, cls_names, reason, *, conf=0.30, region=None) -> action | None  # 找到即点
self.detect_screen_yolo(screen) -> "Lobby"|"Mail"|"Schedule"|"Cafe"|"Craft"|"MomoTalk"|"Story"|"Battle"|None
```
- `cls_names` 可以是单个 str 或 list（多候选）
- `region` = (x1,y1,x2,y2) 归一化 0-1，按 box 中心过滤
- YoloBox 有 `.cx/.cy/.x1/.y1/.x2/.y2/.confidence/.cls_name`

## cls 常量来源
`from brain.skills import ui_classes as UC`，所有 cls 名是 UC 的命名常量（如 `UC.NAV_CAFE`、`UC.BTN_CONFIRM`、`UC.CLAIM_ALL_YELLOW`）。**先读 ui_classes.py 全文**，用里面的常量，不要写裸字符串。

## 范式参考
**先读 `brain/skills/mail.py`**——它是已迁移好的纯 YOLO skill 标准范式：
- `_on_mail_page()` 用 cls 判页面，不用 OCR header
- enter: `find_cls(NAV_MAIL)` 点入口；带 `_enter_click_cooldown` 防重复点
- claim: `find_cls(CLAIM_*)` 多候选 + region 约束；OCR 只读 X/200 计数验证真实进展
- 关键：找不到 claim cls 且 count>0 → `log("YOLO gap") + wait`，不盲点
- exit: `detect_screen_yolo()=="Lobby"` 判完成；退出优先 `find_cls(BTN_HOME/BTN_BACK)` 再 ESC

## 迁移后自检
```bash
py -3 -c "import ast; ast.parse(open('brain/skills/<X>.py',encoding='utf-8').read())"
```

## 报告格式（迁移完返回给我）
1. 改了哪些 OCR→cls（列表）
2. **遇到的 cls gap**：需要但 ui_classes 没有 / 该 cls frame 数太低（<20）可能识别不稳的 — 这些是后续补训练数据的清单
3. 哪些 OCR 保留了（数字/守卫）+ 原因
