# -*- coding: utf-8 -*-
"""scrcpy 视频流感知线程 (combat 2.0 v7 地基).

为什么不用 ADB screencap: RAW 4K 一张 0.85s = 1fps 极限, 战斗感知废.
为什么不用 DXcam: 抓 Windows 桌面, MuMu 被遮挡/最小化就废(用户点破).
scrcpy = Android 内部 H.264 流, 不怕遮挡, 帧龄 <150ms, 且天然无 overlay
烧录(干净帧, 可直接进飞轮).

版本地狱(playbook 4.8 实录): pip install av scrcpy-client --no-deps
+ adbutils==0.14.1 (0.16/2.x 没有 _AdbStreamConnection).

⚠MuMu12 多 display 陷阱(2026-07-15 实锤): display 0 = Android 桌面
launcher, BA 等 app 跑在独立 EXTERNAL display(实测 2)。scrcpy-client
硬编码 display_id=0 → 抓到桌面。find_app_display() 自动定位。

⚠流断裂陷阱(2026-07-15 live 实锤): 出击进战斗时 MuMu 重置 encoder,
流 frame_num 跳变+PPS 丢失 → pyav InvalidDataError 杀掉原版 stream
线程; codec 原地重建也没用(丢 SPS/PPS 上下文, "non-existing PPS 0"
死循环) → 唯一正解 = watchdog 断流超时整个 client 重启(新 socket 让
server 重发 SPS/PPS+IDR)。

坐标系: MuMu 显示 4K(3840x2160), scrcpy 默认缩到 2560x1440;
input tap 用 4K 系 → 本模块只出归一化坐标/原始帧, 换算归动作层.
"""
import os
import re
import subprocess
import threading
import time

import numpy as np

_ADB = r"C:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe"


def _adb_io_lock():
    """AdbInput 类级 _IO_LOCK — feed 的所有 adb 子进程必须与 input tap
    串行(2026-06-15 实锤: 同 transport 并发多MB screencap 丢 MotionEvent;
    2026-07-16 审计: watchdog screencap 绕锁 = 同根因复发路径)."""
    from mumu_runner import AdbInput
    return AdbInput._IO_LOCK


def find_app_display(serial: str = "127.0.0.1:7555",
                     pkg: str = "com.nexon.bluearchive"):
    """dumpsys window 按 display 分段找 pkg 焦点窗口所在 displayId.
    找不到(BA 没起) → None: 调用方绝不能拿 display 0 凑数 — 0 是
    Android 桌面 launcher, feed 会永远盯着桌面且 watchdog 不报错."""
    with _adb_io_lock():
        out = subprocess.run(
            [_ADB, "-s", serial, "shell", "dumpsys", "window"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=15).stdout
    cur = 0
    for line in out.splitlines():
        m = re.search(r"Display: mDisplayId=(\d+)", line)
        if m:
            cur = int(m.group(1))
        if "mCurrentFocus" in line and pkg in line:
            return cur
    return None


def _make_client(device, max_fps: int, display_id: int):
    """工厂: 子类覆盖 name-mangled 私有方法(display_id 参数化 + 解码韧性)."""
    import scrcpy

    class _DisplayClient(scrcpy.Client):
        _target_display = display_id

        def _Client__stream_loop(self):
            import av
            from av.codec import CodecContext
            codec = CodecContext.create("h264", "r")
            while self.alive:
                try:
                    raw_h264 = self._Client__video_socket.recv(0x10000)
                    packets = codec.parse(raw_h264)
                    for packet in packets:
                        frames = codec.decode(packet)
                        for frame in frames:
                            frame = frame.to_ndarray(format="bgr24")
                            if self.flip:
                                import cv2 as _cv
                                frame = _cv.flip(frame, 1)
                            self.last_frame = frame
                            self.resolution = (frame.shape[1],
                                               frame.shape[0])
                            self._Client__send_to_listeners(
                                scrcpy.EVENT_FRAME, frame)
                except BlockingIOError:
                    time.sleep(0.01)
                    if not self.block_frame:
                        self._Client__send_to_listeners(
                            scrcpy.EVENT_FRAME, None)
                except av.error.InvalidDataError:
                    # 丢帧不丢线程; SPS/PPS 彻底丢失时由 Feed watchdog
                    # 重启整个 client 兜底
                    codec = CodecContext.create("h264", "r")
                except OSError as e:
                    if self.alive:
                        raise e

        def _Client__deploy_server(self):
            jar_name = "scrcpy-server-v1.24.jar"
            server_file_path = os.path.join(
                os.path.abspath(os.path.dirname(scrcpy.__file__)), jar_name)
            self.device.push(server_file_path, "/data/local/tmp/")
            commands = [
                f"CLASSPATH=/data/local/tmp/{jar_name}",
                "app_process", "/", "com.genymobile.scrcpy.Server",
                "1.24", "log_level=info",
                f"bit_rate={self.bitrate}",
                f"max_size={self.max_width}",
                f"max_fps={self.max_fps}",
                f"lock_video_orientation={self.lock_screen_orientation}",
                "tunnel_forward=true", "control=true",
                f"display_id={self._target_display}",
                "show_touches=false",
                f"stay_awake={str(self.stay_awake).lower()}",
                "clipboard_autosync=false",
            ]
            self._Client__server_stream = self.device.shell(
                commands, stream=True)
            self._Client__server_stream.read(10)

    c = _DisplayClient(device=device, max_fps=max_fps)
    return c


class ScrcpyFeed:
    """后台线程持帧: latest() 返回 (frame_bgr, age_s, seq); 线程安全.

    ⭐流寿命 17.0s 定律(2026-07-28 三组对照实测): MuMu12 对每个 scrcpy
    镜像流有内在 **17.0s** 寿命上限 — 30fps/8M、10fps/8M、30fps/4M 全部
    死在 t+17.0(误差 <0.1s), 与帧数/码率/输入无关; 死后 server 进程仍活、
    socket 不断, 只是永不再出帧。这就是"每 ~20s 断流 3.5s"(task#17,
    2026-07-21 埋点)的全部真相 — 旧架构等死+重启 = 每 17s 一个 3-5s 盲窗。

    修法 = 预热轮换(双缓冲): 双流并行实测 per-instance 定时器(A 死 t+17.3,
    B 起于 t+8.6 死于 t+25.6 = 自己的 17.0s), 且并存 8s+ 互不干扰。
    ⇒ 流活到 _ROTATE_AT 时预热新流(首帧实测稳定 0.20s) → 原子交接 → 旧流
    收尸, 全程零盲窗。断流 watchdog 保留作兜底(轮换失败/流早死)。
    150s 验证: 11 次轮换全成, 活跃画面零 >0.5s 间隙, 兜底重启 0, 无进程残留;
    静止画面(H.264 天然不出帧)仍由 _is_static() 独立链证实内容=当前屏。

    ⚠换手后的流实测最短只活过 13.5s(对照实验 V1, 可能受静止期影响),
    比裸流 17.0s 短 — _ROTATE_AT 必须给最坏 13.5s 留余量, 别调回 12。"""

    _ROTATE_AT = 10.0    # 实际节奏 ~10.5-11s(+watchdog 1s 粒度), 对 13.5s 余量>2s

    def __init__(self, serial: str = "127.0.0.1:7555", max_fps: int = 30,
                 display_id: int | None = None,
                 stale_restart_s: float = 3.0, log=None):
        self._serial = serial
        self._max_fps = max_fps
        self._display_id = display_id      # None = 自动定位 BA
        self._stale_restart_s = stale_restart_s
        self._log = log or (lambda m: None)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._ts = 0.0
        self._seq = 0
        self._client = None
        self._client_born = 0.0
        self._stopping = False
        self._watchdog = None
        self._restart_lock = threading.Lock()   # stop()×watchdog 重启互斥
        self.restarts = 0
        self.rotations = 0
        self._fail_streak = 0

    def _start_client(self):
        """起一个新 client 并返回 (client, first_frame_holder).
        ⚠不赋值 self._client — 调用方决定何时交接(轮换需要新旧并存窗口)."""
        from adbutils import adb
        import scrcpy
        did = self._display_id
        if did is None:
            did = find_app_display(self._serial)
            if did is None:
                raise RuntimeError("BA 焦点窗口不在任何 display(没起?)"
                                   " — 拒绝回退 display 0(桌面)")
            self._display_id = did
        dev = adb.device(serial=self._serial)
        client = _make_client(dev, self._max_fps, did)
        holder = {"got": False}

        def _on(frame, _h=holder):
            if frame is None:
                return
            _h["got"] = True
            self._on_frame(frame)

        client.add_listener(scrcpy.EVENT_FRAME, _on)
        client.start(threaded=True)
        return client, holder

    def start(self, timeout_s: float = 10.0) -> bool:
        self._client, _ = self._start_client()
        self._client_born = time.time()
        t0 = time.time()
        ok = False
        while time.time() - t0 < timeout_s:
            with self._lock:
                if self._frame is not None:
                    ok = True
                    break
            time.sleep(0.1)
        if ok and self._watchdog is None:
            self._watchdog = threading.Thread(target=self._watchdog_loop,
                                              daemon=True)
            self._watchdog.start()
        return ok

    def _is_static(self) -> bool:
        """age 大时区分「画面静止」vs「真断流」: ADB screencap(独立链)
        与 feed 最后帧对比, 一致=静止(H.264 静止页天然不出帧, 不该重启
        — 2026-07-15 实锤: lobby/hub 静止页 watchdog 误判断流疯狂重启,
        重启风暴反把流打烂)。一致也意味着 feed 帧内容=当前真实屏幕,
        刷新帧龄是语义正确的。对比失败(ADB 挂)按断流处理。
        判据用 12x8 分块 max diff(2026-07-16 审计: 全图均值差看不见
        按钮级变化 — 4K 按钮缩到 64x36 只占 ~10px, 贡献 ~1 << 阈值)."""
        try:
            import cv2
            with _adb_io_lock():     # ⛔与 input tap 串行(丢 tap 同根因)
                raw = subprocess.run(
                    [_ADB, "-s", self._serial, "exec-out", "screencap",
                     "-p"], capture_output=True, timeout=10).stdout
            adb_fr = cv2.imdecode(np.frombuffer(raw, np.uint8),
                                  cv2.IMREAD_COLOR)
            with self._lock:
                feed_fr = self._frame
            if adb_fr is None or feed_fr is None:
                return False
            a = cv2.cvtColor(cv2.resize(adb_fr, (96, 64)),
                             cv2.COLOR_BGR2GRAY).astype(float)
            b = cv2.cvtColor(cv2.resize(feed_fr, (96, 64)),
                             cv2.COLOR_BGR2GRAY).astype(float)
            diff = abs(a - b)
            # 12x8 块, 任一块均值差 >14 = 有局部变化(按钮/弹窗级可见)
            blocks = diff.reshape(8, 8, 12, 8).mean(axis=(1, 3))
            return float(blocks.max()) < 14.0
        except Exception:
            return False

    def _rotate(self):
        """预热新流 → 首帧到达即接管 → 旧流收尸. 交接窗口两流并写
        latest(内容都是当前屏, 无害), 全程不断流. 失败=沿用旧流,
        把 born 后移一轮退避, 旧流真死时由断流兜底接住."""
        with self._restart_lock:
            if self._stopping:
                return
            old = self._client
            try:
                new, holder = self._start_client()
            except Exception as e:
                self._log(f"    [feed] rotate 预热失败({e}) → 退避, 交断流兜底")
                self._client_born = time.time()
                return
            t0 = time.time()
            while time.time() - t0 < 3.0 and not holder["got"]:
                time.sleep(0.05)
            if not holder["got"]:
                self._log("    [feed] rotate 新流 3s 无首帧 → 弃, 沿用旧流")
                try:
                    new.stop()
                except Exception:
                    pass
                self._client_born = time.time()
                return
            self._client = new
            self._client_born = time.time()
            self.rotations += 1
            self._log(f"    [feed] rotate #{self.rotations} ok "
                      f"(首帧 {time.time()-t0:.2f}s)")
            try:
                if old is not None:
                    old.stop()
            except Exception:
                pass

    def _watchdog_loop(self):
        static_streak = 0
        while not self._stopping:
            time.sleep(1.0)
            # ⭐预热轮换: 流寿命 17.0s 定律(见类 docstring), 12s 处无缝换新
            if (not self._stopping and self._client is not None
                    and time.time() - self._client_born >= self._ROTATE_AT):
                self._rotate()
                continue
            with self._lock:
                age = time.time() - self._ts if self._ts else 0.0
            if age <= self._stale_restart_s or self._stopping:
                static_streak = 0
                continue
            # 连续 N 次判静止仍无新帧 → 强制重启一次验流活性
            # (死 socket 停在安静页会被静止判定无限续命 — 审计实锤)
            # ⛔2026-07-28 实测: 旧值 20 且本循环 `time.sleep(1.0)` ⇒ 静止 **~20s**
            # 就强制重启一次(注释里写的 "~90s" 对不上代码)。step_walk 逐帧门控时
            # 画面静止几十秒是常态(人在审帧/改代码), 实录 **每 21-36s 重启一次,
            # 一段日志里 10 次** —— 每次都要在 Android 侧重建 MediaCodec +
            # VirtualDisplay, 正是这段注释自己警告的"重启风暴"。
            # 提到 120(~2min): 代价可控 —— `_is_static()` 每轮都用**独立 ADB 链**
            # 与真实屏幕比过, 判静止就意味着"feed 最后一帧 == 当前屏幕", 拿到的
            # 画面是对的; 而死 socket 一旦屏幕真变化, 下一轮 `_is_static()` 立刻
            # 返回 False → 马上重启。所以"无限续命"的窗口只存在于屏幕也没变的
            # 时候, 那时重不重启对决策毫无影响。
            # ⚠ 这**不是**"游戏崩溃的原因"的证明(今天崩了 3 次, 未复现不断因果),
            #   只是消除一个有量化证据的无谓压力源。
            if static_streak < 120 and self._is_static():
                static_streak += 1
                with self._lock:      # ADB 对比一致=帧内容就是当前屏幕
                    self._ts = time.time()
                continue
            static_streak = 0
            self._log(f"    [feed] 断流{age:.1f}s → 重启 scrcpy client")
            with self._restart_lock:
                if self._stopping:    # stop() 竞态: 拿锁后二次确认
                    break             # (否则孤儿 server+非daemon线程挂死)
                try:
                    if self._client is not None:
                        self._client.stop()
                except Exception:
                    pass
                time.sleep(0.5)
                try:
                    # 连续失败≥2 → display id 可能变了(MuMu/BA 重启会换
                    # EXTERNAL display), 强制重新定位。⚠计数必须数"失败"
                    # 而非"成功"(审计: 成功计数在失败路径冻结, %3 相位
                    # 永不变 → 重定位永不触发)
                    if self._fail_streak >= 2:
                        self._display_id = None
                    self._client, _ = self._start_client()
                    self._client_born = time.time()
                    self.restarts += 1
                    self._fail_streak = 0
                    with self._lock:  # 宽限: 给新 client 出帧窗口
                        self._ts = time.time()
                except Exception as e:
                    self._fail_streak += 1
                    self._log(f"    [feed] 重启失败x{self._fail_streak}"
                              f"({e}), 下轮再试")

    def _on_frame(self, frame):
        if frame is None:
            return
        with self._lock:
            self._frame = frame
            self._ts = time.time()
            self._seq += 1

    def latest(self):
        """(frame, age_s, seq); frame 为 None = 还没出帧."""
        with self._lock:
            if self._frame is None:
                return None, None, 0
            return self._frame, time.time() - self._ts, self._seq

    def stop(self):
        self._stopping = True
        with self._restart_lock:      # 等 watchdog 重启块完成再收尸
            if self._client is not None:
                try:
                    self._client.stop()
                except Exception:
                    pass
                self._client = None
