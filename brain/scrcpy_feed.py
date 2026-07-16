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
    watchdog: 断流 > stale_restart_s 自动重启 client(流断裂唯一正解)."""

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
        self._stopping = False
        self._watchdog = None
        self._restart_lock = threading.Lock()   # stop()×watchdog 重启互斥
        self.restarts = 0
        self._fail_streak = 0

    def _start_client(self):
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
        self._client = _make_client(dev, self._max_fps, did)
        self._client.add_listener(scrcpy.EVENT_FRAME, self._on_frame)
        self._client.start(threaded=True)

    def start(self, timeout_s: float = 10.0) -> bool:
        self._start_client()
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

    def _watchdog_loop(self):
        static_streak = 0
        while not self._stopping:
            time.sleep(1.0)
            with self._lock:
                age = time.time() - self._ts if self._ts else 0.0
            if age <= self._stale_restart_s or self._stopping:
                static_streak = 0
                continue
            # 连续 20 次判静止(~90s)仍无新帧 → 强制重启一次验流活性
            # (死 socket 停在安静页会被静止判定无限续命 — 审计实锤)
            if static_streak < 20 and self._is_static():
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
                    self._start_client()
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
