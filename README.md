# Game-UI CV Agent

A computer-vision agent that plays a mobile game's daily routine by looking at the screen
and clicking. No game files are modified, no packets are touched, nothing leaves the machine.

The target game is Blue Archive running in an Android emulator, but nothing in the perception
or control stack is game-specific. It reads pixels and sends taps.

---

## 它是什么

一个在本机运行的《蔚蓝档案》自动助手。它像人一样看模拟器画面、像人一样点击：
每天自动收咖啡厅、扫荡关卡、打竞技场、领邮件和任务奖励；新活动能自动推图、
按加成规划体力。不改游戏文件、不走网络、纯本地。

**铁律：绝不花青辉石或真钱。** 所有涉及付费货币的操作都有 fail-closed 拦截：
读不出余额就什么都不做，而不是猜一个。

---

## 它怎么"看"

三个专用 YOLO 检测器加一个数字 OCR。**所有"这是什么页面 / 这个按钮能不能点"的判断
都由检测器给出**，OCR 只用来读数字（体力、票数、货币），从不参与判断。

| 用途 | 模型 | 干什么 | 延迟 |
|---|---|---|---|
| UI（主力） | YOLO26m | 每个按钮 / 页签 / 弹窗 / 角标，驱动全部导航和点击 | ~6 ms |
| 角色识别 | YOLO26x | 一次推理出框加角色身份，含战斗中技能卡（灰置 / 充能态） | ~10 ms |
| 战斗单位 | YOLO26s | 我方 / 敌方 / 5 种 Boss / 胜利横幅 / HUD | ~12 ms |
| 数字 | PP-OCRv4（BA 字形微调） | 只读数字字段 | ~50 ms |

现役版本从 `data/model_registry.json` 运行时解析，上线一个模型就是改一行 `active`，
回滚同样一行。

**为什么放弃 OCR 导航**：早期用 OCR 文字加模板匹配找按钮，字体渲染、本地化、分辨率
一变就崩。改成每条点击路径都必须命中一个已训练的类之后，导航变得与分辨率无关，
而且失败是诚实的「没检测到这个类」，不是一次无声的误点。

---

## 数据是怎么来的

模型全部自己标。素材来自 bot 真实跑线时录的干净帧
（Android 内部 `screencap`，所以不会把桌面层的调试框烧进画面）。

- 素材池 **82,590 帧**，经过内容去重、同源泄漏筛查（按时间戳前缀判，不是按文件名，
  改过名的池会漏网）、逐类标注审计
- 已标注 **52,750 帧 / 749,286 框**
- 类表 528 行（含历史废案占位），其中 **492 个在用**

当前 UI 数据集：train 25,375 帧 / val 10,830 帧。

**已知短板（写出来而不是藏进一个 mAP 数字里）**：249 个 UI 类里
**179 个（72%）** 达到「train 至少 30 框且 val 至少 10 框」的门槛。
其余要么是刚加的状态对类（同一控件的另一个状态，样本还只有个位数），
要么是老类完全没有 val 样本。它们的退化目前测不出来，只能靠继续采集补上。

---

## 安全设计

每一个可能花掉付费货币的动作都要过多道闸：

- **购买对话框判据是结构性的**（识别数量步进器），不依赖单个图标检出
- **读不出余额就不动**（fail-closed），而不是按默认值猜
- **按游戏日记账**，防止重复派发
- 上线前每道闸的误报率都在完整的 798 帧确认框语料上实测过，不是推理出来的

---

## 快速开始

**环境**：Windows 10/11 · Python 3.11+ · MuMu Player 12 运行着游戏 ·
NVIDIA GPU（战斗锁定需要 RTX 3060 以上；日常链路是 CPU 密集）

```bash
git clone https://github.com/C0k11/game-ui-cv-agent.git
cd game-ui-cv-agent
pip install -r requirements.txt
```

后端加控制台：

```bash
py -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000/dashboard.html`。

跑日常：

```bash
py -m routing_v2 run
```

跑战斗（scrcpy 视频流加行为树控牌）：

```bash
py -u scripts/bot_play_quest.py 10 11 12
```

训练或迭代模型：在控制台里采集和标注，然后

```bash
py scripts/build_ui_v2.py
py scripts/train_yolo26.py ui_yolo26m_v16
```

上线就是冻结权重、改 `data/model_registry.json` 里的 `active`。

---

## 仓库结构

    routing_v2/          当前架构: 感知 / 状态 / 动作 / flow 四层
      percept/           抓帧、检测、读数
      state/             页面识别、类表词汇
      act/               动作、推进闸、金钱闸、记账
      flow/              每个玩法一个 flow
    brain/               上一代架构, 仍在跑战斗链路
    server/              FastAPI 后端 + 标注控制台
    scripts/             训练 / 建集 / 评估 / 采集工具
    data/
      model_registry.json   现役模型版本, 单一真相源
      raw_images/           标注帧 + _classes.txt 主类表

模型权重、数据集和 HF 缓存都在仓库之外的独立缓存目录（已 gitignore），
路径通过 `HF_HOME` 和 `model_registry.json` 配置，所以 checkout 很小、不提交二进制。

---

## License and Disclaimer

仅供个人和学习使用。不分发任何游戏文件，素材留在你自己的模拟器安装里。
与 Yostar / Nexon / Bilibili / NetEase 无关联。
