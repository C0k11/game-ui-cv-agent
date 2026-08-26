# -*- coding: utf-8 -*-
"""scrcpy 帧源 —— 唯一的帧来路。`latest()` 实测 **0ms**。

从 `brain/scrcpy_feed.py` 移植（那是老代码里少数几块真金），改动:
  · 换用 routing_v2 自己的 device.IO_LOCK / resolve_serial
  · `_is_static()` 的 ADB 对比走 device.screencap 并显式标注为冷路径
  · 只出 (frame, age, seq)，seq 是主循环"对齐 fps"的唯一依据

**流寿命 17.0s 定律**（2026-07-28 三组对照实测）: MuMu12 对每个 scrcpy 镜像流
有内在 17.0s 寿命 —— 30fps/8M、10fps/8M、30fps/4M 全部死在 t+17.0（误差 <0.1s），
与帧数/码率/输入无关；死后 server 进程仍活、socket 不断，只是永不再出帧。
修法 = **预热轮换（双缓冲）**: 活到 _ROTATE_AT 就预热新流（首帧稳定 0.20s）
原子交接  旧流收尸，全程零盲窗。150s 验证：11 次轮换全成，零 >0.5s 间隙。
换手后的流实测最短只活过 13.5s，_ROTATE_AT 必须留余量，别调回 12。

MuMu12 多 display 陷阱: display 0 = Android 桌面 launcher，BA 跑在独立
EXTERNAL display（实测 2）。scrcpy-client 硬编码 display_id=0 会抓到桌面 ——
`find_app_display()` 自动定位，**找不到就 raise，绝不拿 0 凑数**（拿 0 的后果是
feed 永远盯着桌面而 watchdog 一声不吭）。
"""
from __future__ import annotations

import os
import re
import threading
import time
from typing import Optional, Tuple

import numpy as np

from routing_v2.percept.device import _ADB, IO_LOCK, _run, resolve_serial

_PKG = "com.nexon.bluearchive"


_last_good_display: Optional[int] = None


def find_app_display(serial: str, pkg: str = _PKG) -> Optional[int]:
    """找 BA 所在的 displayId。

    `mCurrentFocus` **不可信**（memory §C 早就记过）：MuMu + Unity 全屏不注册
       window focus，实测过它报前台是 `app.lawnchair`（Android 桌面）而游戏
       好好在大厅。2026-08-08 又撞了一次 —— 游戏进程活着，焦点查不到，
       scrcpy 直接起不来。
     三级找法，**任何一级都不许回退 display 0**（0 是桌面，回退了 feed 会
       永远盯着桌面而 watchdog 一声不吭）:
          mCurrentFocus 命中（最准，但常常查不到）
          任何一段里出现过 pkg 的窗口（不要求是"当前焦点"）
          上一次成功用过的 display（进程内记忆）
    """
    global _last_good_display
    with IO_LOCK:
        out = _run([_ADB, "-s", serial, "shell", "dumpsys", "window"], 15)
    cur = 0
    seen_pkg_on: Optional[int] = None
    for line in out.splitlines():
        m = re.search(r"Display: mDisplayId=(\d+)", line)
        if m:
            cur = int(m.group(1))
        if pkg in line:
            if "mCurrentFocus" in line:
                _last_good_display = cur
                return cur
            if cur != 0 and seen_pkg_on is None:
                seen_pkg_on = cur           #  有它的窗口就够了
    if seen_pkg_on is not None:
        _last_good_display = seen_pkg_on
        return seen_pkg_on
    if _last_good_display is not None:      #  用上次成功的
        return _last_good_display
    return None


def _make_client(device, max_fps: int, display_id: int):
    """工厂: 子类覆盖 name-mangled 私有方法（display_id 参数化 + 解码韧性）。"""
    import scrcpy
    try:
        # libav 把解码告警直接刷 stderr(non-existing PPS / no frame!), 坏流时
        #    一秒几十行, 淹掉全部业务日志(08-26 的 event 日志 3052 行里 ~2600
        #    行是它, 还把断流统计污染成 1696)。解码健康度由 watchdog 用
        #    结构化日志报, libav 闭嘴。
        import av
        av.logging.set_level(av.logging.PANIC)
    except Exception:
        pass

    class _DisplayClient(scrcpy.Client):
        _target_display = display_id
        # InvalidDataError 重建 codec 且未再解出帧的时刻(0=健康)
        codec_reset_ts = 0.0

        def _Client__stream_loop(self):
            import av
            from av.codec import CodecContext
            codec = CodecContext.create("h264", "r")
            while self.alive:
                try:
                    raw = self._Client__video_socket.recv(0x10000)
                    for packet in codec.parse(raw):
                        for frame in codec.decode(packet):
                            fr = frame.to_ndarray(format="bgr24")
                            self.last_frame = fr
                            self.resolution = (fr.shape[1], fr.shape[0])
                            self.codec_reset_ts = 0.0
                            self._Client__send_to_listeners(scrcpy.EVENT_FRAME, fr)
                except BlockingIOError:
                    time.sleep(0.01)
                    if not self.block_frame:
                        self._Client__send_to_listeners(scrcpy.EVENT_FRAME, None)
                except av.error.InvalidDataError:
                    # 出击进战斗时 MuMu 重置 encoder  frame_num 跳变 + PPS 丢失。
                    # 丢帧不丢线程；SPS/PPS 彻底丢失由 watchdog 重启整个 client。
                    codec = CodecContext.create("h264", "r")
                    # 重建的 codec **没有 SPS/PPS** -- scrcpy 裸流只在开流时
                    #    发一次参数集, 静止页连 IDR 都不来 -> 重建后若再无一帧
                    #    解出, 这条流就是废的(08-26 实锤: 有效帧率掉到每 4s
                    #    一两帧, event flow 全程无帧可用)。记下时刻, watchdog
                    #    1.5s 内看不到新帧就立刻换流, 不等 3s 断流线。
                    if not self.codec_reset_ts:
                        self.codec_reset_ts = time.time()
                except OSError as e:
                    if self.alive:
                        raise e

        def _Client__deploy_server(self):
            jar = "scrcpy-server-v1.24.jar"
            path = os.path.join(os.path.abspath(os.path.dirname(scrcpy.__file__)), jar)
            self.device.push(path, "/data/local/tmp/")
            cmds = [
                f"CLASSPATH=/data/local/tmp/{jar}",
                "app_process", "/", "com.genymobile.scrcpy.Server",
                "1.24", "log_level=info",
                f"bit_rate={self.bitrate}", f"max_size={self.max_width}",
                f"max_fps={self.max_fps}",
                f"lock_video_orientation={self.lock_screen_orientation}",
                "tunnel_forward=true", "control=true",
                f"display_id={self._target_display}",
                "show_touches=false",
                f"stay_awake={str(self.stay_awake).lower()}",
                "clipboard_autosync=false",
            ]
            self._Client__server_stream = self.device.shell(cmds, stream=True)
            self._Client__server_stream.read(10)

    return _DisplayClient(device=device, max_fps=max_fps)


class Feed:
    """后台线程持帧。`latest()`  (frame_bgr, age_s, seq)，线程安全。"""

    _ROTATE_AT = 10.0        # 对最坏 13.5s 寿命留 >2s 余量，别上调

    def __init__(self, serial: Optional[str] = None, max_fps: int = 30,
                 display_id: Optional[int] = None,
                 stale_restart_s: float = 3.0, log=None):
        self._serial = serial or resolve_serial()
        self._max_fps = max_fps
        self._display_id = display_id
        self._stale_restart_s = stale_restart_s
        self._log = log or (lambda m: print(m, flush=True))
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ts = 0.0
        self._seq = 0
        # 死机探针（2026-08-08 game freeze 实锤）：死机后 scrcpy/screencap
        #    都给**冻结帧**，YOLO 照样 0.97  感知层完全无感，只会表现成
        #    "连点 N 发全丢"。scrcpy 静止屏不推新帧，但流轮换会重发 IDR，
        #    所以要按**内容**判：稀疏采样 + 容差比对，记"内容最后变化时刻"。
        self._content_ref: Optional[np.ndarray] = None
        self._change_ts = 0.0
        self._client = None
        self._client_born = 0.0
        self._stopping = False
        self._watchdog = None
        self._restart_lock = threading.Lock()
        self.restarts = 0
        self.rotations = 0
        self._fail_streak = 0
        # 连续"重启成功但 3s 零首帧"次数 -- purge 升级判据
        self._dead_streak = 0

    #  生命周期
    def _start_client(self):
        from adbutils import adb
        import scrcpy
        did = self._display_id
        if did is None:
            did = find_app_display(self._serial)
            if did is None:
                raise RuntimeError("BA 焦点窗口不在任何 display（游戏没起？）"
                                   " — 拒绝回退 display 0（那是桌面）")
            self._display_id = did
        client = _make_client(adb.device(serial=self._serial), self._max_fps, did)
        holder = {"got": False}

        def _on(frame, _h=holder):
            if frame is None:
                return
            _h["got"] = True
            self._on_frame(frame)

        client.add_listener(scrcpy.EVENT_FRAME, _on)
        client.start(threaded=True)
        return client, holder

    def start(self, timeout_s: float = 12.0) -> bool:
        ok = self._try_start(timeout_s)
        if not ok:
            # **scrcpy server 挂死自愈**（2026-08-10 实测，不修就是每次人工救场）:
            #    症状 = client 连得上但一帧都出不来，日志刷 `non-existing PPS 0
            #    referenced` + `no frame!`，**而游戏完全正常**（adb 秒回、
            #    find_app_display 也对）。根因是设备上残留一个挂死的 scrcpy
            #    server（`ps -A | grep app_process` 状态 `futex_wait_queue_me`），
            #    新 client 握上去永远等不到 SPS/PPS。
            #     杀掉它 + 删掉 server jar（让 client 重新 push 一份）再试一次。
            #    只做**一次**：真的是游戏/模拟器挂了的话，重试再多也没用，
            #      让它 return False 交给上层（runner 有 halt、cli 有报错）。
            self._log("[feed] 起不来  清理挂死的 scrcpy server 后重试一次")
            self._purge_dead_server()
            ok = self._try_start(timeout_s)
            self._log(f"[feed] 自愈{'成功' if ok else '失败'}")
        if ok and self._watchdog is None:
            self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
            self._watchdog.start()
        return ok

    def _try_start(self, timeout_s: float) -> bool:
        try:
            self._client, _ = self._start_client()
        except Exception as e:
            self._log(f"[feed] client 起不来: {e}")
            return False
        self._client_born = time.time()
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            with self._lock:
                if self._frame is not None:
                    return True
            time.sleep(0.05)
        return False

    def _purge_dead_server(self) -> None:
        """杀掉设备上挂死的 scrcpy server 并删掉它的 jar。

        `pkill -f scrcpy` 在 MuMu 上会**挂住**（08-10 实测超时被杀），
           只能先 `ps -A` 拿 pid 再 `kill -9`。
        jar 文件名带版本号（`scrcpy-server-v1.24.jar`），别删错名字。
        """
        try:
            if self._client is not None:
                try:
                    self._client.stop()
                except Exception:
                    pass
                self._client = None
            with self._lock:
                self._frame = None
            out = _run([_ADB, "-s", self._serial, "shell",
                        "ps -A 2>/dev/null | grep app_process"], 15)
            for line in (out or "").splitlines():
                parts = line.split()
                if len(parts) > 1 and parts[1].isdigit():
                    self._log(f"[feed] kill 挂死的 scrcpy server pid={parts[1]}")
                    _run([_ADB, "-s", self._serial, "shell", "kill", "-9",
                          parts[1]], 10)
            _run([_ADB, "-s", self._serial, "shell",
                  "rm -f /data/local/tmp/scrcpy-server*.jar"], 10)
            time.sleep(1.0)
        except Exception as e:
            self._log(f"[feed] 清理挂死 server 失败（继续重试）: {e}")

    def stop(self):
        self._stopping = True
        with self._restart_lock:
            if self._client is not None:
                try:
                    self._client.stop()
                except Exception:
                    pass
                self._client = None

    #  取帧
    def _on_frame(self, frame):
        if frame is None:
            return
        # 稀疏采样（~30x40x3 像素，微秒级）。容差 >2.0 才算"内容变了"——
        # 流轮换后同一画面重解码会有 ±1 级噪声，精确比对会把轮换误判成变化。
        small = frame[::73, ::97].astype(np.int16)
        with self._lock:
            self._frame = frame
            self._ts = time.time()
            self._seq += 1
            if (self._content_ref is None
                    or self._content_ref.shape != small.shape
                    or float(np.abs(small - self._content_ref).mean()) > 2.0):
                self._content_ref = small
                self._change_ts = self._ts

    def frozen_s(self) -> float:
        """屏幕内容已经多久没变过（秒）。>30s 且 tap 无响应  疑似游戏死机。"""
        with self._lock:
            if self._change_ts <= 0.0:
                return 0.0
            return time.time() - self._change_ts

    def latest(self) -> Tuple[Optional[np.ndarray], float, int]:
        with self._lock:
            if self._frame is None:
                return None, 0.0, 0
            return self._frame, time.time() - self._ts, self._seq

    def wait_new(self, last_seq: int, timeout: float = 2.0):
        """阻塞到出现比 last_seq 新的帧。主循环用它对齐 fps（而不是 sleep）。"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            fr, age, seq = self.latest()
            if fr is not None and seq != last_seq:
                return fr, age, seq
            time.sleep(0.005)
        return self.latest()

    #  watchdog：轮换 + 断流兜底
    def _is_static(self) -> bool:
        """age 大时区分「画面静止」vs「真断流」。

        H.264 静止页天然不出帧 —— 老架构在 lobby/hub 静止页疯狂重启，重启风暴
        反把流打烂（2026-07-15 实锤）。用**独立 ADB 链**抓一张比对：一致 = 静止，
        且意味着 feed 最后一帧就是当前真实屏幕，刷新帧龄是语义正确的。
        判据用 12x8 分块 max diff —— 全图均值差看不见按钮级变化。
        """
        try:
            import cv2
            with IO_LOCK:
                import subprocess
                raw = subprocess.run(
                    [_ADB, "-s", self._serial, "exec-out", "screencap", "-p"],
                    capture_output=True, timeout=12).stdout
            adb_fr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            with self._lock:
                feed_fr = self._frame
            if adb_fr is None or feed_fr is None:
                return False
            a = cv2.cvtColor(cv2.resize(adb_fr, (96, 64)), cv2.COLOR_BGR2GRAY).astype(float)
            b = cv2.cvtColor(cv2.resize(feed_fr, (96, 64)), cv2.COLOR_BGR2GRAY).astype(float)
            blocks = abs(a - b).reshape(8, 8, 12, 8).mean(axis=(1, 3))
            return float(blocks.max()) < 14.0
        except Exception:
            return False

    def _rotate(self):
        """预热新流  首帧到达即接管  旧流收尸。失败=沿用旧流，交断流兜底。"""
        with self._restart_lock:
            if self._stopping:
                return
            old = self._client
            try:
                new, holder = self._start_client()
            except Exception as e:
                self._log(f"    [feed] rotate 预热失败({e})  退避")
                self._client_born = time.time()
                return
            t0 = time.time()
            while time.time() - t0 < 3.0 and not holder["got"]:
                time.sleep(0.03)
            if not holder["got"]:
                self._log("    [feed] rotate 新流 3s 无首帧  弃, 沿用旧流")
                try:
                    new.stop()
                except Exception:
                    pass
                self._client_born = time.time()
                return
            self._client = new
            self._client_born = time.time()
            self.rotations += 1
            try:
                if old is not None:
                    old.stop()
            except Exception:
                pass

    def _watchdog_loop(self):
        static_streak = 0
        revive = 0                 # 靠 _is_static 续命的累计轮数(真帧才清零)
        seen_seq = -1
        while not self._stopping:
            time.sleep(1.0)
            if (not self._stopping and self._client is not None
                    and time.time() - self._client_born >= self._ROTATE_AT):
                self._rotate()
                continue
            # codec 孤儿: 数据在流、解码全废。_is_static 会把它当"画面静止"
            #    无限续命(比对用的是最后一张**解出的**帧, 静止页恰好一致),
            #    所以必须在静止判据**之前**拦。
            _c = self._client
            _reset = float(getattr(_c, "codec_reset_ts", 0.0) or 0.0) if _c else 0.0
            dead_decode = _reset > 0 and time.time() - _reset > 1.5
            with self._lock:
                age = time.time() - self._ts if self._ts else 0.0
                seq_now = self._seq
            if seq_now != seen_seq:
                seen_seq = seq_now
                revive = 0         # **只有真帧到达才算活** -- 数事实
            if not dead_decode and (age <= self._stale_restart_s or self._stopping):
                static_streak = 0
                # 2026-08-13 尸体复活实锤: 部署屏(静止画面)上流死 ->
                #    _is_static 每秒刷新 _ts -> age 永远小 -> 这里把
                #    static_streak 清零 -> 120 轮强制验活**永远攒不满** ->
                #    死流被无限续命, runner 静默饿死 25 分钟。
                #    修 = 续命轮数 revive 单独记账, 只有 seq 前进(真解出新帧)
                #    才清零; 续命满 40 次(每次隔 stale阈值~3-4s, 约 2.5min)
                #    强制走重启分支。
                if revive < 40:
                    continue
                self._log("    [feed] 靠静止判据续命 40 次但没有任何新帧"
                          " -- 强制重启验活")
            # 逐帧门控时画面静止几十秒是常态（人在审帧）。静止判据每轮都用独立
            # ADB 链跟真屏比过，所以"无限续命"的窗口只存在于屏幕也没变的时候，
            # 那时重不重启对决策毫无影响。上限 120 轮（~2min）后强制验一次活性。
            elif (not dead_decode and static_streak < 120 and self._is_static()):
                static_streak += 1
                revive += 1
                with self._lock:
                    self._ts = time.time()
                continue
            static_streak = 0
            revive = 0
            if dead_decode:
                self._log(f"    [feed] codec 孤儿(重建后 {time.time()-_reset:.1f}s"
                          f" 零解码) -- 重启 scrcpy client 拿新参数集")
            else:
                self._log(f"    [feed] 断流{age:.1f}s -- 重启 scrcpy client")
            with self._restart_lock:
                if self._stopping:
                    break
                try:
                    if self._client is not None:
                        self._client.stop()
                except Exception:
                    pass
                time.sleep(0.5)
                try:
                    # 连续失败≥2  display id 可能变了（MuMu/BA 重启会换 EXTERNAL
                    # display）。计数必须数"失败"而非"成功"：成功计数在失败路径
                    # 冻结，相位永不变  重定位永不触发（老代码审计实锤）。
                    if self._fail_streak >= 2:
                        self._display_id = None
                    # 连续两次重启拿不到首帧 = 设备上多半挂着一个吃连接的死
                    #    server(08-10 结案那族: 连得上、永远等不到 SPS/PPS)。
                    #    start() 里的 purge 自愈这里永远轮不到 -- 零星帧会把
                    #    runner 的 40 口饥饿计数一直清零 -- 只能在这里清。
                    if self._dead_streak >= 2:
                        self._log(f"    [feed] 连续 {self._dead_streak} 次重启"
                                  f"零首帧 -- 清理设备上挂死的 scrcpy server")
                        self._purge_dead_server()
                    self._client, holder = self._start_client()
                    self._client_born = time.time()
                    self._fail_streak = 0
                    # 首帧验活。原来这里无条件把 _ts 刷成 now -- 一帧都没有的
                    #    重启也被当成活了 3s, 死流按 3-4s 一轮无限空转。
                    _t0 = time.time()
                    while time.time() - _t0 < 3.0 and not holder["got"]:
                        time.sleep(0.05)
                    if holder["got"]:
                        self.restarts += 1
                        self._dead_streak = 0
                    else:
                        self._dead_streak += 1
                        self._log(f"    [feed] 重启后 3s 无首帧"
                                  f" x{self._dead_streak}")
                except Exception as e:
                    self._fail_streak += 1
                    self._log(f"    [feed] 重启失败x{self._fail_streak}({e})")
