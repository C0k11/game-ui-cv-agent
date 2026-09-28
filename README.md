# Game-UI CV Agent

A computer-vision agent that plays a mobile game's daily routine by looking at the
screen and clicking. No game files are modified, no packets are touched, nothing
leaves the machine.

The target game is Blue Archive (Taiwan server) running in an Android emulator.
Scripts that read the game state from memory instead of pixels are in progress.

## 概述

本机运行的《蔚蓝档案》台服日常自动化。YOLO 检测器逐帧识别页面和按钮，OCR 只读数字（体力、票数、货币），
动作通过 adb 点击模拟器。每个玩法是 `routing_v2/flow/` 下的一个 flow：咖啡厅、课程表、悬赏、战术大赛、
商店、邮件、任务、活动、任務推图等。青辉石购买键程序不点。

读游戏内存的版本（内存版脚本）在推进中。

## 本地配置

本机实测环境：

- Windows 11，Python 3.13.5
- NVIDIA 显卡，torch 2.6.0 + CUDA 12.4；没有可用的 CUDA 时检测自动改用 CPU，会慢
- MuMu 模拟器 6.8，16:9 横屏（本机 3840x2160），游戏为台服 `com.nexon.bluearchive`

安装：

```bash
git clone https://github.com/C0k11/game-ui-cv-agent.git
cd game-ui-cv-agent
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install --no-deps scrcpy-client==0.4.1
```

`scrcpy-client` 的依赖声明要求 `av<10`，本机实际用 av 18.0.0，所以单独用 `--no-deps` 装。
想让 OCR 的文字检测走 GPU，最后再装 `onnxruntime-gpu`（本机 1.24.4）。

模型权重不在仓库里。`data/model_registry.json` 里各模型 `active` 版本的 `path` 指向本机权重
（ui v21、fused_avatar v6、battle v11），路径无效时检测直接报错，不会回落到别的模型。

adb 和 MuMuManager 按 MuMu 默认安装路径查找，端口向 MuMuManager 查询；装在别处时用环境变量
`MUMU_ADB`、`MUMU_MANAGER`、`ADB_SERIAL` 指定。

运行：

```bash
py -X utf8 -m uvicorn server.app:app --host 127.0.0.1 --port 8000   # 控制台 http://127.0.0.1:8000/v2/
py -X utf8 -m routing_v2 probe                 # 看一帧：页面身份和全部检出
py -X utf8 -m routing_v2 step --go             # 单步：--go 才真点并验到达
py -X utf8 -m routing_v2 run --auto            # 自主跑；不带 --auto 每一发都要人放行
py -X utf8 -m routing_v2.tests.test_offline    # 离线回归，不连模拟器
```

## License and Disclaimer

仅供个人学习使用。仓库不分发任何游戏文件，素材保留在你自己的模拟器安装里。
本项目与 Yostar、Nexon、Bilibili、NetEase 均无关联。
