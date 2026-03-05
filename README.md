# 私人碧蓝档案助手

**碧蓝档案 (Blue Archive)** 全自动日常助手 — 纯本地运行，OCR + YOLO 驱动，无需云端 API。

在 MuMu 模拟器上通过屏幕捕获 + ADB 实现零侵入自动化：每日咖啡厅、课程表、社团、商店、制造、活动刷体力、悬赏通缉、战术大赛、邮件、每日任务、Hard 关卡等全流程一键执行。

---

## 快速开始

### 环境要求

- Windows 10/11
- Python 3.11+
- [MuMu Player 12](https://mumu.163.com/) 运行碧蓝档案
- NVIDIA GPU（推荐 RTX 3060+，用于 YOLO 推理）

### 安装

```powershell
git clone https://github.com/C0k11/blue-archive-assistant.git
cd blue-archive-assistant
pip install -r requirements.txt
pip install rapidocr_onnxruntime   # OCR 引擎
```

### 运行

```powershell
# MuMu 模拟器自动化（推荐）
py mumu_runner.py

# 可选参数
py mumu_runner.py --title "MuMu"      # 自定义窗口标题
py mumu_runner.py --adb-port 7555     # 自定义 ADB 端口
py mumu_runner.py --dry-run           # 仅预览，不执行点击
py mumu_runner.py --fps 60            # 降低捕获帧率
```

### Web Dashboard（可选）

```powershell
py scripts\run_backend.py
# 打开 http://127.0.0.1:8000/dashboard.html
```

## 模型 & 缓存

| 用途 | 路径 |
|---|---|
| YOLO 模型 | `D:\Project\ml_cache\models\yolo\full.pt` / `full.engine` |
| HuggingFace 缓存 | `D:\Project\ml_cache\huggingface` |
| 轨迹记录 | `data\trajectories\run_YYYYMMDD_HHMMSS\` |

## 训练工具

```powershell
# 数据采集
py scripts\collect_window_dataset.py --title "MuMu" --out data\captures

# 头像数据增强（合成毒药覆盖）
py scripts\augment_avatar_dataset.py

# YOLO 训练
py scripts\train_expanded_yolo.py

# TensorRT 导出
py scripts\export_tensorrt.py
```

## 轨迹系统

每次运行自动记录完整轨迹到 `data/trajectories/`，每个 tick 保存：
- 当前技能 & 子状态
- 执行的动作（点击坐标/等待/返回）
- 完整 OCR 结果
- YOLO 检测结果
- 截图时间戳

用于事后分析和调试技能逻辑。

## 项目结构

```
ai-game-secretary/
├── brain/
│   ├── pipeline.py          # 技能调度器 + 全局拦截器
│   └── skills/
│       ├── base.py          # ScreenState, BaseSkill, 动作函数
│       ├── lobby.py         # 登录 & 大厅检测
│       ├── cafe.py          # 咖啡厅（收益/邀请/摸头 1F+2F）
│       ├── schedule.py      # 课程表
│       ├── club.py          # 社团
│       ├── shop.py          # 商店
│       ├── craft.py         # 制造
│       ├── event_farming.py # 活动刷体力
│       ├── bounty.py        # 悬赏通缉
│       ├── arena.py         # 战术大赛
│       ├── mail.py          # 邮件
│       ├── daily_tasks.py   # 每日任务
│       └── farming.py       # Hard 关卡
├── vision/
│   ├── yolo_detector.py     # YOLOv8 / TensorRT 检测
│   ├── avatar_matcher.py    # 头像特征匹配
│   ├── engine.py            # 推理引擎抽象
│   └── window.py            # 窗口捕获
├── scripts/
│   ├── yolo_overlay.py      # 实时 YOLO 覆盖窗口
│   ├── win_capture.py       # BitBlt 窗口截图
│   ├── augment_avatar_dataset.py
│   ├── train_expanded_yolo.py
│   └── export_tensorrt.py
├── server/
│   └── app.py               # FastAPI 后端 + Dashboard
├── mumu_runner.py           # MuMu 主运行入口
├── dashboard.html           # Web 控制面板
└── requirements.txt
```

## 未来计划

- [ ] **总力战 (Raid)** 自动化
- [ ] **大决战 (Grand Assault)** 自动化
- [ ] **火力演习 (Joint Firing Drill)** 自动化
- [ ] 智能体力分配：根据活动/Hard 需求自动规划 AP 消耗
- [ ] 多账号支持
- [ ] 运行结果推送通知（Discord / Telegram）
- [ ] GUI 配置面板（技能开关、优先级调整）
- [ ] 更精确的 OCR 模型微调（繁体中文 + 游戏字体）

## License

MIT
