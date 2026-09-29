"""Vision helpers left from the old daily pipeline: OCR engine, YOLO model loading and
inference, the clean-frame cache, top-bar and digit reading. server/app.py (labeling
dashboard) and a few scripts use them.

The DailyPipeline class and its skills were archived on 2026-09-28 (archive/2026-09-28/);
daily runs go through routing_v2.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from brain.skills.base import OcrBox, YoloBox, ScreenState


#  OCR Engine (singleton)

_ocr_engine = None
_ocr_lock = None

def _try_enable_cuda_dlls() -> bool:
    """Load CUDA/cuDNN + TensorRT DLLs so onnxruntime-gpu can register both
    the CUDA and TensorRT execution providers without a separate CUDA/TRT
    Toolkit install. Pip packages supply all needed DLLs:

      - torch (cu124) bundles cublasLt / cudart / cudnn under torch/lib/
      - tensorrt-cu12-libs (pip) bundles nvinfer_10.dll under tensorrt_libs/

    Returns True if at least one DLL dir was added.
    """
    added = False
    try:
        import os
        if not hasattr(os, "add_dll_directory"):
            return False
        try:
            import torch  # type: ignore
            lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
            if os.path.isdir(lib_dir):
                os.add_dll_directory(lib_dir)
                added = True
        except Exception:
            pass
        try:
            import tensorrt_libs  # type: ignore
            trt_dir = os.path.dirname(tensorrt_libs.__file__)
            if os.path.isdir(trt_dir):
                os.add_dll_directory(trt_dir)
                added = True
        except Exception:
            pass
    except Exception:
        pass
    return added


def _get_ocr():
    """Get or create RapidOCR engine (thread-safe singleton).

    Automatically:
    - Loads fine-tuned Blue Archive rec model from data/ocr_model/ba_rec.onnx
      if present, otherwise uses default PP-OCRv3.
    - Tries to enable CUDA provider (det + rec). Falls back to CPU if CUDA
      runtime DLLs are unavailable. Measured on RTX 4090: full-frame 1262x2243
      CPU 2.2 FPS  CUDA 3.3 FPS; ROI-sized (~45% screen) CUDA 9 FPS.
    """
    global _ocr_engine, _ocr_lock
    import threading
    if _ocr_lock is None:
        _ocr_lock = threading.Lock()
    with _ocr_lock:
        if _ocr_engine is None:
            _try_enable_cuda_dlls()
            from rapidocr_onnxruntime import RapidOCR
            custom_rec = Path(__file__).resolve().parent.parent / "data" / "ocr_model" / "ba_rec.onnx"
            kw = dict(
                det_use_cuda=True, det_model_path=None,
                rec_use_cuda=True,
                cls_use_cuda=True, cls_model_path=None,
            )
            kw["rec_model_path"] = str(custom_rec) if custom_rec.exists() else None
            try:
                _ocr_engine = RapidOCR(**kw)
                # Inspect what provider det actually got - if CPU, we know
                # CUDA load failed silently.
                try:
                    det_providers = _ocr_engine.text_detector.infer.session.get_providers()
                    if "CUDAExecutionProvider" in det_providers:
                        print(f"[OCR] CUDA provider active (det={det_providers})")
                    else:
                        print(f"[OCR] CUDA requested but fell back to CPU "
                              f"(det={det_providers}). Install nvidia-cuda-runtime-cu12 "
                              f"or add CUDA Toolkit to PATH.")
                except Exception:
                    pass
                if custom_rec.exists():
                    print(f"[OCR] Using fine-tuned BA rec model: {custom_rec.name}")
            except Exception as e:
                # Fall back to pure CPU with default kwargs
                print(f"[OCR] CUDA init failed ({e!r}); using CPU")
                _ocr_engine = (
                    RapidOCR(rec_model_path=str(custom_rec))
                    if custom_rec.exists()
                    else RapidOCR()
                )
        return _ocr_engine


#  YOLO Detector (singleton)

_yolo_models = []   # list of (model, conf_threshold, model_tag) tuples
_yolo_lock = None
# Only two purpose-built models: battle character heads + cafe emoticon bubbles.
# emoticon migrated to YOLO26n (2026-05-17) - same architecture family as the
# previous v8n but NMS-free, 122 layers / 2.4M params / 5.2 GFLOPs.  Validation
# on emoticon_v2 dataset: P=0.994 R=1.000 mAP50=0.995 mAP50-95=0.994, inference
# 0.4ms/frame.  Drop-in replacement - same single class "Emoticon_Action".

#  Model registry - single source of truth for active model paths
# data/model_registry.json drives which version is "live".  Hardcoded paths
# below are fallbacks for back-compat / when registry is unreachable.

def _load_model_registry() -> dict:
    """Read data/model_registry.json. Returns {} on any error."""
    try:
        reg_path = Path(__file__).resolve().parent.parent / "data" / "model_registry.json"
        if reg_path.is_file():
            import json
            return json.loads(reg_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[Pipeline] model_registry load failed: {e}")
    return {}

def _resolve_path(model_key: str, fallback: Path) -> Path:
    """Resolve an active model's weights path via registry.

    fail-closed(2026-07-16 审计): registry 存在但该 key 解析不出
    raise, 绝不静默回落硬编码老路径 - 旧行为是"registry JSON 手编出错
    /active 指错"时带着 5 月 v1(145类, 缺全部商店/confirm 钱防线类)照常
    开跑, 正是"改了 registry 没效果、旧模型还在跑"的复现路径。
    fallback 仅在 registry 文件整体缺失(全新环境)时使用。"""
    reg = _load_model_registry()
    if not reg:                     # registry 文件缺失/坏  见下
        if fallback.is_file():
            print(f"[Pipeline] registry 缺失, {model_key} 回落 {fallback}")
            return fallback
        raise RuntimeError(f"model registry 缺失且 {model_key} fallback 不存在")
    section = reg.get(model_key)
    if not section:
        raise RuntimeError(f"registry 无 '{model_key}' 节 - 修 registry, 不回落老模型")
    active = section.get("active")
    versions = section.get("versions", {})
    info = versions.get(active, {})
    p = info.get("path")
    if p and Path(p).is_file():
        return Path(p)
    raise RuntimeError(
        f"registry {model_key}.active='{active}' 路径无效({p}) - "
        f"修 registry, 绝不静默回落老模型")


_YOLO_BATTLE_HEADS = Path(r"D:\Project\ml_cache\models\yolo\battle_heads.pt")
_YOLO_EMOTICON_V26 = Path(r"D:\Project\ml_cache\models\yolo\runs\emoticon_yolo26n\weights\best.pt")
_YOLO_EMOTICON = _YOLO_EMOTICON_V26

#  Fused avatar v4 (251-class student head detector)
# Manual mAP50 = 0.9657 on hand-curated 29-frame val (vs v3 baseline 0.683).
# best_manual.pt = ep11 weights (true best, vs best.pt = ep15 nominal-best
# that was synth-fitness-biased). See yolo_migration memory for details.
_YOLO_FUSED_AVATAR_V4 = _resolve_path("fused_avatar", Path(
    r"D:\Project\ml_cache\models\yolo\runs\fused_avatar_yolo26x_v4\weights\best_manual.pt"
))

#  UI v1 (~145-class static UI detector)
# Replaces OCR-driven button finding in most skills. Trained 2026-05-27 from
# COCO yolo26m + 1220 frames (oversampled minority classes target=12) + 51
# hand-curated val frames. Target mAP50 ≥ 0.85.
_YOLO_UI_V1 = _resolve_path("ui", Path(
    r"D:\Project\ml_cache\models\yolo\runs\ui_yolo26m_v1\weights\best.pt"
))

# Context-aware YOLO: controls which models run each tick.
# Value semantics:
#   'none'                   skip YOLO entirely (default)
#   'all'                    run every loaded model
#   single tag e.g. 'cafe'   run only that detector
#   '+'-joined e.g. 'ui+avatar' or 'cafe+ui'   run multiple detectors
# Legacy callers passing 'cafe' / 'battle' still work unchanged.
_yolo_context = "none"
_yolo_context_lock = None

def set_yolo_context(ctx: str) -> None:
    """Set which YOLO models should run. Called by pipeline on skill change."""
    global _yolo_context, _yolo_context_lock
    import threading
    if _yolo_context_lock is None:
        _yolo_context_lock = threading.Lock()
    with _yolo_context_lock:
        if _yolo_context != ctx:
            print(f"[Pipeline] YOLO context: {_yolo_context}  {ctx}")
            _yolo_context = ctx

def get_yolo_context() -> str:
    """Get current YOLO context (thread-safe)."""
    global _yolo_context, _yolo_context_lock
    import threading
    if _yolo_context_lock is None:
        _yolo_context_lock = threading.Lock()
    with _yolo_context_lock:
        return _yolo_context


#  数字 strip 几何: 一律用「锚点图标自身尺寸」当单位
# 2026-07-27 定死, 起因是 bounty 票数在关卡列表页 **0/211 帧**读得出。
#
# 为什么不能用屏幕比例(0.078 这种):
#   屏幕比例只在**标定它的那个分辨率+宽高比**上成立。这个系统的帧至少有三条
#   来路 -- scrcpy(设备分辨率) / ADB screencap(设备分辨率) / DXcam(**窗口**大小,
#   窗口随便拖多大都行, 还可能带边框) -- 语料里实测出 **19 种分辨率、宽高比
#   1.4812~1.7927**。UI 是整体等比缩放的, 所以"数字串相对图标的位置"是**布局
#   常数**, 而"数字串占屏幕宽度的百分之几"不是。
#
# 9 种分辨率(2363x1331 ~ 3840x2160)实测数字串右界, 单位=图标宽:
#   青辉石 med 4.11 iw (各分辨率 3.43~4.31) / 信用点 4.99 (4.63~5.13)
#   体力 5.44 (4.68~5.72, 含 "/240" 尾巴)
# 同一批数据换算成屏幕比例则随分辨率漂 -- 这就是差别。
#
# y 留白同样关键, 而且这条**教训在仓库里躺了 10 天没传导**:
# `arena.py:188` 2026-07-17 就写过"±0.4bh 是临界高度, icon 框轻微抖动就把整串
# 裁没(live 12 连 None 实锤), ±0.8bh 同帧完整读出" -- 但 ticket_sweep(0.4)
# 和这里的顶栏(0.25)都没跟着改。DB 文本检测器需要行外留白才肯出框。
def icon_strip(box, x_from: float, x_to: float, y_pad: float):
    """锚点图标右侧的数字 strip, 单位 = 图标自身宽/高(分辨率与宽高比无关)。

    x_from / x_to: 从 `box.x2` 起算, 单位为图标宽度 iw。
    y_pad:         图标框上下各外扩多少个图标高度 bh。
    """
    iw = max(1e-6, box.x2 - box.x1)
    bh = max(1e-6, box.y2 - box.y1)
    return (min(1.0, box.x2 + x_from * iw),
            max(0.0, box.y1 - y_pad * bh),
            min(1.0, box.x2 + x_to * iw),
            min(1.0, box.y2 + y_pad * bh))


# Per-currency digit strip in ICON units: (x_from, x_to, y_pad).
# 语义保留自 2026-06-11 的标定意图:
#   AP     只要 "999" 不碰 "/240"(旧 0.078 屏宽吃到斜杠  读成 9999);
#           parse_count 取分子, 所以右界宁短勿长。
#   credit 9 位长数, 需要最宽的窗口。
#   pyrox  中等。
def _topbar_strip_map():
    """2026-07-27 网格标定(scratchpad/grid_cal.py)。

    真值代理 = **同一 run 内货币值近乎恒定**(只有买卖才动)  对每个
    (run, 货币) 取全部 (窗口 x 帧) 读数的共识值, 共识≥55% 才当真值; 34 run /
    246 帧 / 35 组参数 / 58 个有强真值的组。

    | 货币   | 最优 (x_to, y_pad) | 对    | 错   | 空   | n/组 |
    | 青辉石 | (6.0, **1.2**)     | 84.2% | 7.9% | 7.9% | 165 |
    | 信用点 | (5.5, **0.5**)     |100.0% | 0.0% | 0.0% |  40 |
    | 体力   | (5.0, **1.6**)     | 88.1% | 9.4% | 2.5% | 202 |

    **y 留白必须逐币标定, 方向还相反**: 信用点要**紧**(0.5), AP 要**松**(1.6)。
    我先前用单一 0.80 是折中 -- 那也是为什么"tight vs loose"二选一的对比里
    信用点反而变差。别再拿一个常数套所有货币。
    信用点那 100% 的 n 只有 40(强真值组少), 可信度低于另外两个, 别过度解读。
    这些数字是**离线共识**不是绝对真值; 顶栏读数上层仍有 5 帧众数投票 +
      ≥2 票才信 + 位数收缩闸兜底(`_read_topbar_clean`), 本表只提高单帧命中率。
    """
    from brain.skills.ui_classes import TOPBAR_AP, TOPBAR_CREDIT, TOPBAR_PYROXENE
    return {
        TOPBAR_AP: (0.10, 5.0, 1.60),
        # 信用点**没用网格那个 0.50** -- 网格的真值代理是"多参数共识", 而 8 位
        # 长数的**系统性截断可以自己成为共识**: 它推荐的 (5.5,0.50) 在今天的
        # 基线帧上读出 `390459`(吞掉前三位), 而屏幕上白纸黑字是 23,390,459。
        # 改用**人眼读出的真值**重标(今天 walk 的 10 张 4K 帧, 真值 23,390,459):
        #   y_pad=0.65  **10/10 全对**(x_to 4.5~6.5 都行), 其余 y_pad 只有 9/10,
        #   且失败一律是 None 不是错数(零错读)。
        #  教训: 短数字(14176/999)能自证, 长数不能; 给长数标定必须挂人审真值。
        # n=10 且只有一个信用点数值, 样本小 -- 但真值锚定优于大样本的共识锚定。
        TOPBAR_CREDIT: (0.10, 5.5, 0.65),
        # 2026-08-07 live 事故 + 重标: kill-switch 报
        # `青辉石 1803618003 MONEY BREACH` 把整条 pipeline 急停, 而人眼看帧
        # 余额**一分没少**。复现: 同一张 4K 帧 OCR 4 次里 3 次读成 18003
        # (逗号被读成 0 + 末位 6 被切) -- 是**裁切窗口**的锅, 不是模型。
        # 病根: 信用点当初已按人眼真值重标(y_pad 0.65), 而青辉石**留着网格
        # 共识值 (6.0, 1.20)** -- 正是上面那条注释警告过的"共识≠真值", 这一格
        # 当时漏做了。27 帧 × 42 组参数实测(真值 18,036, 人眼读):
        #   y_pad ≤0.35  裁太紧, 纵向切掉数字, 读成 `36`
        #   y_pad 0.40~0.50  **27/27 零错读**(x_to 5.0~6.0 全通)
        #   y_pad ≥0.55  裁太松吃进邻居, 读成 `18003`
        #   现役的 1.20 = 20/27 对 / **7 次错读**, 整张网格最烂的区之一
        # 取安全块中心, 对两种失效模式都留最大余量。
        # 单一数值(18,036)标定, 样本值不够多 -- 但失效是**几何性**的(裁切高度),
        #   且旧值已被实锤打脸。危害双向: 会读低就会读高, **读高会掩盖真实掉钱**。
        TOPBAR_PYROXENE: (0.10, 5.5, 0.45),
    }


try:
    _TOPBAR_STRIP = _topbar_strip_map()
except Exception:
    _TOPBAR_STRIP = {}
_TOPBAR_STRIP_DEFAULT = (0.10, 4.8, 0.80)


def _read_topbar_count(screen, cls_name: str):
    """DIGIT-only read of the number right of a top-bar icon (cy<0.10)."""
    best = None
    for b in (screen.yolo_boxes or []):
        if b.cls_name != cls_name or b.confidence < 0.25:
            continue
        if b.cy >= 0.10:
            continue
        if best is None or b.confidence > best.confidence:
            best = b
    if best is None or screen.frame is None:
        return None
    # Right edge = a per-currency span from the icon, in ICON WIDTHS. History:
    # the neighbour-clip (clip at the next 加号/icon) was the first bug - the
    # neighbour flickers frame-to-frame, and when AP's 加号 dropped the span
    # over-reached into credit and read 9999999 (12× live 2026-06-11). It was
    # replaced by a fixed SCREEN FRACTION, which killed the flicker but pinned
    # the read to one resolution - see `icon_strip` above for why that breaks.
    x_from, x_to, y_pad = _TOPBAR_STRIP.get(cls_name, _TOPBAR_STRIP_DEFAULT)
    raw = run_digit_ocr(screen.frame, icon_strip(best, x_from, x_to, y_pad))
    res = parse_count(raw)
    return res[0] if (res is not None and res[0] is not None) else None


class _FrameShim:
    """Minimal screen-like holder for _read_topbar_count on a raw frame."""
    __slots__ = ("yolo_boxes", "frame")

    def __init__(self, boxes, frame):
        self.yolo_boxes = boxes
        self.frame = frame


def _read_topbar_clean(cls_name, samples: int = 5):
    """Top-bar count from fresh overlay-free ADB frames (2026-06-10 money rule),
    made robust to the digit OCR's BOTH-WAY instability (live 2026-06-11:
    leading-digit DROP 6587587 AND right-edge OVER-read 9999999 / credit
    reaching into the neighbour). Neither "fewest" nor "most" digits is right,
    so VOTE: read up to `samples` clean frames, return the MODE (the value the
    OCR agrees on most often - correct more often than any single error mode).
    Returns int or None.

     KNOWN GAP (task#5): AP/credit still mis-crop on many frames; pyroxene is
    reliable. Until per-currency right-edge crop is calibrated, callers that
    spend on a balance (shop) must treat a low-confidence read as unverifiable
    and skip - never over-trust an inflated read."""
    from collections import Counter
    reads = []
    for _ in range(max(1, samples)):
        frame = get_clean_frame()
        if frame is None:
            continue
        try:
            h, w = frame.shape[:2]
            boxes = _run_yolo_on_image(frame, w, h)
            v = _read_topbar_count(_FrameShim(boxes, frame), cls_name)
        except Exception:
            v = None
        if v is not None:
            reads.append(v)
            if reads.count(v) >= 3:   # strong agreement  done early
                return v
    if not reads:
        return None
    cnt = Counter(reads)
    top, n = cnt.most_common(1)[0]
    # Require a real majority (≥2 agreeing) before trusting; a single noisy
    # read is not authoritative for money decisions.
    return top if n >= 2 else None


def _read_topbar_clean_multi(cls_names, samples: int = 5):
    """快照专用: 一批 clean 帧共享给多个货币读数 (2026-07-11 链路审计).
    语义与 _read_topbar_clean 完全一致(≤samples 帧 / 每 cls mode 投票 /
    3 票强共识早退 / ≥2 票才信), 但 3 货币共享同批帧  captures 155、
    YOLO 155(旧版 lobby 快照单 tick 16 连拍阻塞主循环 7-20s 的元凶)。
    _read_topbar_clean 本体保持原样 - shop/ticket_sweep 等金钱敏感调用方
    的单币种投票语义不动。Returns {cls_name: int|None}."""
    from collections import Counter
    reads = {c: [] for c in cls_names}
    done = set()
    for _ in range(max(1, samples)):
        if len(done) == len(cls_names):
            break
        frame = get_clean_frame()
        if frame is None:
            continue
        try:
            h, w = frame.shape[:2]
            boxes = _run_yolo_on_image(frame, w, h)
            shim = _FrameShim(boxes, frame)
        except Exception:
            continue
        for c in cls_names:
            if c in done:
                continue
            try:
                v = _read_topbar_count(shim, c)
            except Exception:
                v = None
            if v is not None:
                reads[c].append(v)
                if reads[c].count(v) >= 3:
                    done.add(c)
    out = {}
    for c in cls_names:
        r = reads[c]
        if not r:
            out[c] = None
            continue
        top, n = Counter(r).most_common(1)[0]
        out[c] = top if (n >= 2 or r.count(top) >= 3) else None
    return out


_yolo_load_attempts = 0
_MAX_YOLO_LOAD_ATTEMPTS = 3
_yolo_status = "not_attempted"
# ui 模型加载失败旗标 (2026-07-07): True = 导航之眼缺失, tick 循环立刻 abort,
# 绝不 blind/wake-tap (假 no-UI 下 wake-tap 曾反复戳开購買AP框)。
_UI_LOAD_FAILED = False
# No class-name filter - purpose-built models (battle_heads, emoticon) only
# contain relevant classes.  The old _YOLO_ALLOWED_SUBSTRINGS gate silently
# dropped every detection from battle_heads whose classes are c0-c3.

def _get_yolo():
    """Get or create YOLO model(s) (lazy singleton). Only loads on first call.

    Loads available model files (battle_heads + emoticon).
    Each entry is a (model, conf_threshold) tuple.
    Returns the first model for backward compat; _yolo_models holds all.
    """
    global _yolo_models, _yolo_lock, _yolo_load_attempts, _yolo_status
    import threading
    if _yolo_lock is None:
        _yolo_lock = threading.Lock()
    with _yolo_lock:
        if _yolo_models:
            return _yolo_models[0]
        if _yolo_load_attempts >= _MAX_YOLO_LOAD_ATTEMPTS:
            return None
        _yolo_load_attempts += 1
        # Per-model confidence thresholds:
        # battle_heads: 0.45 (well-defined targets; 0.15 causes false positives
        #   on cafe sprites at conf 0.25-0.47)
        # emoticon: 0.15 (headpat bubbles on 2F score as low as 0.18)
        candidates = []  # (path, conf_threshold, tag)
        # battle 走 registry 最新 vN(2026-07-16 历史遗留修复: 旧代码硬编码
        # legacy battle_heads.pt 无视 registry - server 侧 Bounty/Arena/JFD
        # 一直用老模型, v9(0.989 nc18) 只有战斗脚本在用)
        # active 优先(2026-07-16 审计: max vN 会在"先登记 v10 条目未验收"
        # 时静默上未验收模型; active 是验收后才 bump 的正式指针), 无 active
        # 才回退 max vN。
        _battle_path = _YOLO_BATTLE_HEADS
        try:
            import re as _re
            _bh = _load_model_registry().get("battle_heads", {})
            _vers = _bh.get("versions", {})
            _pick = _bh.get("active")
            if _pick not in _vers or not _re.fullmatch(r"v\d+", str(_pick)):
                _pick = max((v for v in _vers if _re.fullmatch(r"v\d+", v)),
                            key=lambda x: int(x[1:]), default=None)
            if _pick:
                _p = Path(_vers[_pick]["path"])
                if not _p.is_absolute():
                    _p = Path("D:/Project") / str(_p).lstrip("/\\")
                if _p.is_file():
                    _battle_path = _p
                    print(f"[Pipeline] battle_heads  registry {_pick}")
        except Exception as _e:
            print(f"[Pipeline] battle registry resolve failed({_e}), legacy")
        if _battle_path.is_file():
            candidates.append((_battle_path, 0.45, "battle"))
        # standalone emoticon (tag "cafe") 不在这里 append - fold-in 判定提前
        # (2026-07-17): ui v6+ 自带 Emoticon_Action, 仅当 ui 类表缺该类时才在
        # 加载环节末尾补载 v26n(见 load loop 之后), 省一次白加载即丢的模型。
        # Fused avatar (251 BA student heads).  conf 0.35 = balanced
        # precision/recall on manual val.  Tagged "avatar" - opt-in per skill.
        if _YOLO_FUSED_AVATAR_V4.is_file():
            candidates.append((_YOLO_FUSED_AVATAR_V4, 0.35, "avatar"))
        # UI (registry active - buttons, dots, banners, etc).  Tagged "ui" -
        # most skills need this. (unified v6b 接线已拆除 2026-07-17: registry
        # unified.active 恒 PENDING 从未通电, v6b nc=455 缺后续金钱防线类,
        # 见 registry unified._deprecated 注。)
        _ui_path = _YOLO_UI_V1
        if _ui_path is not None and _ui_path.is_file():
            # 0.20 - within the dashboard's own prefill range (server/app.py:
            # single-frame suggest 0.15, batch prefill 0.25), the settings the
            # user verifies cls against. Live at 0.30 dropped weak cls the
            # dashboard catches (免费 14f live ~0.18-0.30). Strong cls (0.9+)
            # unaffected. Skills still gate money paths structurally (2-button
            # confirm + 免费/币种 checks), so a lower floor doesn't risk spend.
            candidates.append((_ui_path, 0.20, "ui"))
        if not candidates:
            _yolo_status = "model_not_found"
            print(f"[Pipeline] YOLO model NOT found")
            return None
        from ultralytics import YOLO
        import numpy as np
        loaded_names = []
        global _UI_LOAD_FAILED
        _UI_LOAD_FAILED = False

        def _load_model(model_path, model_conf, model_tag) -> bool:
            # 加载失败(fresh-server CUDA dtype 瞬态, 2026-07-07 实锤
            # "float != c10::Half")重试一次 - dtype 病随机咬任意模型。
            for _t in range(2):
                try:
                    m = YOLO(str(model_path))
                    m(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
                    _yolo_models.append((m, model_conf, model_tag))
                    loaded_names.append(f"{model_path.stem}({len(m.names)}cls)")
                    print(f"[Pipeline] YOLO loaded from {model_path} (conf={model_conf}, tag={model_tag})")
                    return True
                except Exception as e:
                    print(f"[Pipeline] YOLO load failed for {model_path} (try {_t+1}/2): {e}")
            return False

        for model_path, model_conf, model_tag in candidates:
            # ui 模型是导航的眼睛 - 加载失败(重试后仍败)打 _UI_LOAD_FAILED 旗标,
            # tick 循环见旗标立刻 abort(绝不带着假 no-UI 去 wake-tap/blind-tap -
            # 那次假 no-UI 让 wake-tap 反复戳开「購買AP」框, 差点碰钱)。
            if not _load_model(model_path, model_conf, model_tag) and model_tag == "ui":
                _UI_LOAD_FAILED = True
                print("[Pipeline]  ui model FAILED to load after retry - pipeline will "
                      "abort immediately (fail-closed: no taps without UI eyes)")
        #  emoticon fold-in (ui v6+, 判定提前 2026-07-17)
        # ui 类表含 Emoticon_Action  摸头泡泡由 ui forward pass 提供,
        # standalone v26n 不再加载(旧代码先完整加载再 fold-in 丢弃 = 每次
        # 启动白加载一个模型)。ui 缺该类(pre-v6)才补载 v26n 保 cafe headpat。
        # cafe.py + 摸头过滤按 cls_name 匹配, box 来自哪个模型无感。
        # (2026-06-11 用户决策) ui 接管摸头, v26n 退役出 live 管线, 只留
        # dashboard 预标注 teacher (server prefill 走 registry, 不受影响)。
        _FOLD_IN_EMOTICON = True
        ui_has_emoticon = _FOLD_IN_EMOTICON and any(
            "emoticon" in str(n).lower()
            for m, _c, t in _yolo_models if t == "ui"
            for n in m.names.values()
        )
        if ui_has_emoticon:
            print("[Pipeline] ui model carries Emoticon_Action  standalone "
                  "emoticon model not loaded (one fewer inference per cafe tick)")
        elif _YOLO_EMOTICON.is_file():
            # 0.50 conf (was 0.150.30 same day): live 2026-06-09 credit-card
            # icons kept firing as Emoticon_Action (0.36-0.75). Business gate is
            # 0.55 (cafe.py _EMOTICON_CONF), real v26n bubbles score 0.9+, so
            # 0.50 costs nothing and kills the remaining mid-conf FPs.
            _load_model(_YOLO_EMOTICON, 0.50, "cafe")
        if _yolo_models:
            _yolo_status = f"loaded_ok: {', '.join(loaded_names)}"
            return _yolo_models[0][0]
        _yolo_status = "all_candidates_failed"
        return None


_OCR_WORK_W = 1280  # Downscale wide frames for faster OCR

#  PURE-YOLO MODE (user spec 2026-05-29)
# OCR is fully disabled to force every skill's navigation + click logic
# through YOLO cls - NO OCR fallback. This surfaces every place still
# secretly relying on OCR (they go blind  log+wait  we migrate them).
# Once the YOLO pipeline is verified end-to-end, flip this back on and
# scope OCR to DIGIT-ONLY scanning (AP / ticket / mail counts).
_OCR_ENABLED = False



def _run_ocr_on_image(img, w: int, h: int) -> List[OcrBox]:
    """Run OCR on a BGR numpy array and return normalized OcrBox list.

    Downscales frames wider than _OCR_WORK_W for speed (4K1280px ≈ 9x faster).
    Coordinates are normalized 0-1 so the caller is resolution-independent.
    """
    if not _OCR_ENABLED:
        return []  # pure-YOLO mode - see _OCR_ENABLED note above
    import cv2
    ocr = _get_ocr()
    # Downscale for speed if frame is very wide (e.g. 3840px 4K)
    ocr_img = img
    ocr_w, ocr_h = w, h
    if w > _OCR_WORK_W:
        ratio = _OCR_WORK_W / w
        ocr_h = max(1, int(h * ratio))
        ocr_w = _OCR_WORK_W
        ocr_img = cv2.resize(img, (ocr_w, ocr_h), interpolation=cv2.INTER_AREA)
    result, _ = ocr(ocr_img)
    boxes: List[OcrBox] = []
    if result:
        for line in result:
            pts, text, conf = line
            conf = float(conf)
            if conf < 0.4:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            boxes.append(OcrBox(
                text=text,
                confidence=conf,
                x1=min(xs) / ocr_w,
                y1=min(ys) / ocr_h,
                x2=max(xs) / ocr_w,
                y2=max(ys) / ocr_h,
            ))
    return boxes


#  Clean-frame source (money-read defense)
# The Win32 YoloOverlay burns boxes/labels into every DXcam frame, which can
# KILL detection of small icons (live 2026-06-09: arena 战术大赛票 icon got a
# tight green box + label burned over it  ui_v7 detected NOTHING  ticket
# read None every tick  fail-closed exit with tickets unspent). ADB screencap
# runs inside Android where the overlay physically doesn't exist. Money-
# critical reads should prefer this source. The server registers the ADB
# capture function here once the pipeline's ADB connection is up.
_CLEAN_FRAME_SOURCE = None

#  同 tick 干净帧复用 (2026-07-25)
# 实测: run_digit_ocr 每次调用都重抓一张 ADB 4K 帧 ~770ms, 而 OCR 本体只要
# 19.6ms(scratchpad/bench_digit.py)。一个 tick 内 skill 不可能改变屏幕 -
# skill.tick() 只返回 action dict, 点击在 tick 返回之后才由 server 落屏 -
# 所以同一 tick 内的多次数字读数看到的必然是同一屏, 抓一次就够。
# ticket_sweep._read_tickets 一个 tick 最多读 3 次 = 白等 1.5s。
#
# 默认 max_age=0 不缓存, 语义与旧版逐字节相同。原因: _read_topbar_clean /
# _read_topbar_clean_multi 的多帧投票**靠每次拿到不同帧**才有意义, 一旦缓存
# 就塌成"同一帧读 5 遍"的复读, 直接废掉金钱读数的投票防线。只有显式传
# max_age 的调用方(run_digit_ocr)才走缓存。
_CF_LOCK = threading.Lock()
_CF_CACHE = {"frame": None, "ts": 0.0, "epoch": -1}
_CF_EPOCH = 0                      # 每 tick +1, 跨 tick 复用物理上不可能
_CF_STATS = {"grab": 0, "hit": 0}  # 只统计走缓存的调用方(digit 读数)
_DIGIT_FRAME_MAX_AGE = 1.5         # 兜底 TTL(epoch 已挡跨 tick, 这道防 tick 外调用)


def set_clean_frame_source(fn) -> None:
    """Register a zero-arg callable returning a clean BGR frame (or None)."""
    global _CLEAN_FRAME_SOURCE
    _CLEAN_FRAME_SOURCE = fn


def invalidate_clean_frame_cache() -> None:
    """新 tick 开始 / 动作落屏后调用: 让上一 tick 的干净帧彻底失效。"""
    global _CF_EPOCH
    with _CF_LOCK:
        _CF_EPOCH += 1
        _CF_CACHE["frame"] = None
        _CF_CACHE["ts"] = 0.0
        _CF_CACHE["epoch"] = -1


def clean_frame_cache_stats() -> Dict[str, int]:
    """{'grab': 真抓次数, 'hit': 同 tick 复用次数} - 用于实测收益。"""
    with _CF_LOCK:
        return dict(_CF_STATS)


def get_clean_frame(max_age: float = 0.0):
    """A fresh overlay-free frame via the registered source, or None.

    max_age > 0  允许复用 **本 tick 内** max_age 秒以内抓过的帧。
    max_age = 0(默认)  每次真抓, 老调用方语义不变。
    """
    if _CLEAN_FRAME_SOURCE is None:
        return None
    if max_age > 0:
        now = time.time()
        with _CF_LOCK:
            if (_CF_CACHE["frame"] is not None
                    and _CF_CACHE["epoch"] == _CF_EPOCH
                    and (now - _CF_CACHE["ts"]) <= max_age):
                _CF_STATS["hit"] += 1
                return _CF_CACHE["frame"]
    # 时间戳取抓帧**开始**时刻(保守: ADB screencap 本身要 0.77s, 按结束
    # 时刻记会低估帧龄)。
    t0 = time.time()
    try:
        fr = _CLEAN_FRAME_SOURCE()
    except Exception:
        return None
    if fr is not None and max_age > 0:
        with _CF_LOCK:
            _CF_CACHE["frame"] = fr
            _CF_CACHE["ts"] = t0
            _CF_CACHE["epoch"] = _CF_EPOCH
            _CF_STATS["grab"] += 1
    return fr


def run_digit_ocr(frame, region_norm) -> Optional[str]:
    """DIGIT-ONLY OCR on a normalized sub-region of a BGR frame.

    The pure-YOLO design (user spec): YOLO locates an icon/region, OCR reads
    ONLY the digits inside a crop next to it. This is INDEPENDENT of the global
    `_OCR_ENABLED` flag (that gates full-screen text OCR for navigation, which
    stays off) - digit reads are always allowed because that's OCR's one job.

    Args:
        frame: BGR numpy array (ScreenState.frame).
        region_norm: (x1, y1, x2, y2) normalized 0-1 crop to read.
    Returns:
        The raw recognized string filtered to digits/separators ("240/240",
        "25117", "7/8"), or None if nothing digit-like was read. Caller parses.
    """
    if frame is None:
        return None
    # 数字读数需要 4K 细节(1080p 实测伤金钱读数): 主 tick 帧换 scrcpy
    # 1440p 后(2026-07-16 Phase2), 凡传入低于 4K 的帧自动升级 ADB 干净帧
    # 重抓 - 一处兜底, 9 个 skill 调用点零改动。抓帧失败用原帧(降级可用)。
    # 同 tick 复用(2026-07-25): 一个 tick 内屏幕不会变(点击在 tick 返回后
    # 才落屏), 所以第 2..N 次读数直接吃缓存, 每次省 ~770ms。
    try:
        if frame.shape[1] < 3200:
            _cf = get_clean_frame(max_age=_DIGIT_FRAME_MAX_AGE)
            if _cf is not None:
                frame = _cf
    except Exception:
        pass
    import cv2
    import re as _re
    try:
        h, w = frame.shape[:2]
        x1 = max(0, int(region_norm[0] * w)); y1 = max(0, int(region_norm[1] * h))
        x2 = min(w, int(region_norm[2] * w)); y2 = min(h, int(region_norm[3] * h))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        crop = frame[y1:y2, x1:x2]
        # upscale small crops - OCR is far more accurate on larger glyphs
        ch, cw = crop.shape[:2]
        if ch < 40:
            sc = 40.0 / ch
            crop = cv2.resize(crop, (int(cw * sc), 40), interpolation=cv2.INTER_CUBIC)
        ocr = _get_ocr()
        result, _ = ocr(crop)
        if not result:
            return None
        # Sort fragments LEFTRIGHT before joining - the detector returns text
        # boxes in arbitrary order, which scrambles comma-grouped numbers
        # (live 2026-06-09: "179,958,141" came back as '9581179414').
        try:
            result = sorted(result, key=lambda ln: min(p[0] for p in ln[0]))
        except Exception:
            pass
        # concat all recognized text on the strip, keep only digits + / and ,
        raw = "".join(line[1] for line in result)
        # Comma-grouped big numbers (credit 25,583,379 etc): blind strip-and-
        # join DUPLICATES digits when OCR fragments overlap (live 2026-06-12:
        # '25,583,379'  '255833379' = 10x over-read  shop budget chaos).
        # The comma grouping VALIDATES digit structure - when present, trust
        # only a clean single group; several disjoint groups = fragment mess
        # fail-closed None (multi-sample voting retries).
        # 千位分隔符归一(2026-07-28 live 实锤, 原始片段为证):
        # 青辉石 15,426 那一条, ocr 返回**两个片段** `'15.'`(score .66) 与
        # `',426'`(score .80) -- **两边各自把分隔符包了进去**, 拼接成 `'15.,426'`。
        # 后果: 旧正则只认 `,` 不匹配; 落到下面"保留小数点"的通用清洗
        # `'15.426'`  `parse_count` 既非纯数字也无 '/'  **None**, 这一帧整个
        # 读不出(fail-closed 安全, 但钱闸拿不到数)。
        # 我第一版只把正则的 `,` 放宽成 `[.,]` -- **不管用**, 因为真正的形态是
        # 「点+逗号连在一起」。先折叠连续分隔符, 再做三位分组匹配。
        # 为什么可以放心把 `.` 当分隔符: 判据是**三位一组**(`[.,]\d{3}`)。
        # 本域里真正的小数只有百分比且都是**一位**小数(cafe 收益 '58.3' / '0.0%',
        # 见下面那段 2026-06-09 的注释) -- 一位小数永远匹配不上 \d{3}。
        raw_n = _re.sub(r"[.,]{2,}", ",", raw.replace("，", ","))
        groups = _re.findall(r"\d{1,3}(?:[.,]\d{3})+", raw_n)
        if groups:
            longest = max(groups, key=len)
            others = [g for g in groups if g != longest and g not in longest]
            if others:
                return None   # ambiguous overlapping fragments
            return longest.replace(",", "").replace(".", "")
        # Keep the decimal point too (deep-dive r2 C1, 2026-06-09): stripping it
        # turned "0.0%" into "00" and "58.3" into "583" - consumers that parse
        # floats (cafe earnings % gate) need the dot. parse_count() is dot-free
        # by domain (counts/AP/tickets never render decimals) so this is safe.
        kept = _re.sub(r"[^0-9/.]", "", raw_n.replace(",", ""))
        return kept or None
    except Exception as e:
        print(f"[digit-OCR] error: {e}")
        return None


def parse_count(s: Optional[str]):
    """Parse a digit-OCR string into useful numbers.

    "7/8"  (7, 8); "25117"  (25117, None); "240/240"  (240, 240).
    Returns (current, total) where total may be None. None on unparseable.
    """
    if not s:
        return None
    try:
        if "/" in s:
            a, _, b = s.partition("/")
            cur = int(a) if a.isdigit() else None
            tot = int(b) if b.isdigit() else None
            if cur is None:
                return None
            return (cur, tot)
        if s.isdigit():
            return (int(s), None)
    except Exception:
        pass
    return None


def _run_yolo_on_image(img, w: int, h: int, context: str = "") -> List[YoloBox]:
    """Run YOLO on a BGR numpy array and return normalized YoloBox list.

    Only runs models matching the current context:
      'cafe'    emoticon model only
      'battle'  battle_heads model only
      'all'     all models
      'none'/''  skip entirely (returns empty)

    Non-blocking: if YOLO is still loading in the pre-warm thread, returns
    empty immediately instead of blocking the pipeline worker.
    """
    if not context:
        context = get_yolo_context()
    yolo_boxes: List[YoloBox] = []
    if context == "none":
        return yolo_boxes
    # Non-blocking check: if lock is held (pre-warm loading), skip this tick
    if _yolo_lock is not None and not _yolo_models:
        acquired = _yolo_lock.acquire(blocking=False)
        if not acquired:
            return yolo_boxes
        _yolo_lock.release()
    _get_yolo()  # ensure models are loaded
    if not _yolo_models:
        if not getattr(_run_yolo_on_image, '_warned', False):
            print(f"[Pipeline] YOLO unavailable: {_yolo_status}")
            _run_yolo_on_image._warned = True
        return yolo_boxes
    # Parse context: 'all' = run everything, 'a+b' = run tags a or b,
    # single tag = run only that tag.
    if context == "all":
        wanted_tags = None  # None = no filter
    else:
        wanted_tags = set(t.strip() for t in context.split("+") if t.strip())
    # Per-detector inference imgsz. ui_v1 was trained on 2255×1268 frames
    # at imgsz=960, but pipeline captures at 3840×2160 (4K MuMu). Default
    # imgsz=640 loses small UI elements completely (verified: 0 detections
    # on lobby tick 1). 1920 brings detection back to expected mAP. Other
    # detectors were trained at smaller native frame sizes - 960 is fine.
    _IMGSZ_BY_TAG = {
        # ui_v2 trained at imgsz=960 on 2475×1392 frames. MUST infer at 960:
        # verified 2026-05-28 that v2 @ imgsz=1920  0 detections, but @ 960
        # 一次领取黄色 conf 0.936 etc. (The earlier 1920 "4K fix" was wrong -
        # production frames are 2475×1392, not 4K; the occasional 4K frame was
        # a capture-path glitch, not the norm.)
        "ui": 960,
        "avatar": 960,     # fused_avatar trained at 960
        "battle": 960,
        "cafe": 640,       # emoticon - 1 class, simple, default ok
    }
    # Standalone emoticon (tag "cafe") actually running in THIS call? The
    # ui-emoticon yield rule below must only fire when v26n is really there -
    # after fold-in the model is unloaded and ui's 451 IS the headpat source.
    _standalone_emo_active = any(
        t == "cafe" and (wanted_tags is None or t in wanted_tags)
        for _y, _c, t in _yolo_models)
    for yolo, model_conf, model_tag in _yolo_models:
        if wanted_tags is not None and model_tag not in wanted_tags:
            continue
        try:
            ifsz = _IMGSZ_BY_TAG.get(model_tag, 960)
            yolo_results = yolo(img, conf=model_conf, imgsz=ifsz, verbose=False)
            for r in yolo_results:
                for box in r.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    cls_name = yolo.names.get(cls_id, str(cls_id))
                    cls_low = str(cls_name).lower()
                    # ui carries a folded Emoticon_Action (cls451). When the
                    # standalone v26n (tag "cafe", 0.995) is ALSO running it is
                    # the emoticon AUTHORITY - drop the ui copy, else the two
                    # models double-box every bubble (offset boxes, IoU<0.6
                    # dedup can't catch) = ghosting + "emoticon 和 ui 抢信用点"
                    # (live 2026-06-09). After fold-in (2026-06-11) v26n is
                    # unloaded  _standalone_emo_active False  ui 451 passes.
                    if (model_tag == "ui" and "emoticon" in cls_low
                            and _standalone_emo_active):
                        continue
                    nx1, ny1, nx2, ny2 = bx1/w, by1/h, bx2/w, by2/h
                    # Filter headpat/emoticon: only accept in cafe play area
                    if "headpat" in cls_low or "emoticon" in cls_low:
                        bcx = (nx1 + nx2) / 2
                        bcy = (ny1 + ny2) / 2
                        if bcy < 0.15 or bcy > 0.85:
                            continue
                        bw = nx2 - nx1
                        bh = ny2 - ny1
                        if bw < 0.02 or bh < 0.02:
                            continue
                    yolo_boxes.append(YoloBox(
                        cls_id=cls_id,
                        cls_name=cls_name,
                        confidence=float(box.conf[0]),
                        x1=nx1,
                        y1=ny1,
                        x2=nx2,
                        y2=ny2,
                        model_tag=model_tag,
                    ))
        except Exception as e:
            print(f"[Pipeline] YOLO detect error: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
    #  Cross-class / cross-model region dedup (user rule 2026-06-09:
    # "一个框的区域不能重复然后检测出另外的东西"). One screen region = ONE
    # detection: when boxes from different classes/models overlap heavily
    # (IoU>0.6), keep only the highest-confidence one. Kills e.g. the
    # earnings-popup 信用点 icon (ui 0.9+) ALSO firing as Emoticon_Action
    # (cafe model 0.38-0.58) = ghosted double box. Small-on-big overlaps
    # (红点 on an entry icon) have tiny IoU and are never deduped.
    if len(yolo_boxes) > 1:
        # Pre-pass - DOMAIN AUTHORITY, not confidence: the emoticon model's
        # (tag "cafe") only legit target is a headpat bubble, which never
        # overlaps a UI element. Any emoticon box overlapping (IoU>0.3) a box
        # from another model is an FP on that element  drop it EVEN IF its
        # conf is higher (live 2026-06-09: emoticon 0.75 on the 2號店 credit
        # icon outranked ui and "won" the conf-desc dedup - wrong winner).
        _others = [b for b in yolo_boxes if b.model_tag != "cafe"]
        if _others:
            def _emo_on_ui(e: YoloBox) -> bool:
                ea = max((e.x2 - e.x1) * (e.y2 - e.y1), 1e-9)
                for o in _others:
                    ix1, iy1 = max(e.x1, o.x1), max(e.y1, o.y1)
                    ix2, iy2 = min(e.x2, o.x2), min(e.y2, o.y2)
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    oa = max((o.x2 - o.x1) * (o.y2 - o.y1), 1e-9)
                    if inter / (ea + oa - inter) > 0.3:
                        return True
                return False
            yolo_boxes = [b for b in yolo_boxes
                          if b.model_tag != "cafe" or not _emo_on_ui(b)]
        kept: List[YoloBox] = []
        for b in sorted(yolo_boxes, key=lambda x: -x.confidence):
            bw, bh = b.x2 - b.x1, b.y2 - b.y1
            area_b = max(bw * bh, 1e-9)
            dup = False
            for k in kept:
                ix1, iy1 = max(b.x1, k.x1), max(b.y1, k.y1)
                ix2, iy2 = min(b.x2, k.x2), min(b.y2, k.y2)
                iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                inter = iw * ih
                area_k = max((k.x2 - k.x1) * (k.y2 - k.y1), 1e-9)
                iou = inter / (area_b + area_k - inter)
                # emoticon (tag "cafe") vs a higher-conf box from another
                # model: suppress at the LOOSER 0.3 - its FPs sit ON ui icons
                # (信用点 card 0.9 ui vs 0.38-0.58 emoticon) but with offset
                # boxes that rarely clear 0.6. Real bubbles overlap nothing.
                thr = 0.3 if ("cafe" in (b.model_tag, k.model_tag)
                              and b.model_tag != k.model_tag) else 0.6
                if iou > thr:
                    dup = True
                    break
            if not dup:
                kept.append(b)
        yolo_boxes = kept
    return yolo_boxes


#  Top-level detector helpers for skills
# Skills should NOT call _run_yolo_on_image directly - use these.  They
# operate on the current ScreenState's yolo_boxes (already populated each
# tick by the pipeline observation step) so there's no extra inference cost.

def find_yolo_box(screen: ScreenState, class_names: List[str],
                   min_conf: float = 0.3) -> Optional[YoloBox]:
    """Return the highest-confidence YoloBox matching any of class_names,
    or None.  Use this in skills to replace OCR-driven button finding:

        # OLD:
        btn = screen.find_any_text(["一次領取", "一次领取"], min_conf=0.6)
        # NEW:
        btn = find_yolo_box(screen, ["一次领取_黄", "一次领取"], min_conf=0.5) \\
              or screen.find_any_text(["一次領取", "一次领取"], min_conf=0.6)

    Matches by class_name exact-equal first, then case-insensitive substring
    fallback.  Returns the box with highest confidence among matches.
    """
    if not screen.yolo_boxes:
        return None
    name_set = set(class_names)
    name_lower = [n.lower() for n in class_names]
    hits: List[YoloBox] = []
    for b in screen.yolo_boxes:
        if b.confidence < min_conf:
            continue
        # exact match
        if b.cls_name in name_set:
            hits.append(b)
            continue
        # substring fallback
        bn = (b.cls_name or "").lower()
        if any(q in bn or bn in q for q in name_lower):
            hits.append(b)
    if not hits:
        return None
    return max(hits, key=lambda x: x.confidence)


def find_all_yolo_boxes(screen: ScreenState, class_names: List[str],
                         min_conf: float = 0.3) -> List[YoloBox]:
    """Like find_yolo_box but returns ALL matches sorted by confidence."""
    if not screen.yolo_boxes:
        return []
    name_set = set(class_names)
    name_lower = [n.lower() for n in class_names]
    hits: List[YoloBox] = []
    for b in screen.yolo_boxes:
        if b.confidence < min_conf:
            continue
        if b.cls_name in name_set:
            hits.append(b); continue
        bn = (b.cls_name or "").lower()
        if any(q in bn or bn in q for q in name_lower):
            hits.append(b)
    return sorted(hits, key=lambda x: -x.confidence)


def read_screen_from_frame(frame_bgr, *, screenshot_path: str = "",
                           skip_ocr: bool = False,
                           prev_ocr_boxes=None,
                           injected_yolo_boxes=None) -> ScreenState:
    """Build ScreenState from an in-memory BGR numpy array (no file I/O).

    Used by the MuMu runner for zero-copy capture  detect pipeline.

    Args:
        skip_ocr: if True, skip OCR (expensive ~50ms) and reuse prev_ocr_boxes.
        prev_ocr_boxes: OCR boxes from a previous tick to reuse when skip_ocr=True.
        injected_yolo_boxes: pre-computed YOLO boxes from high-FPS thread.
            If provided, skip running YOLO here (already done at high FPS).
    """
    if frame_bgr is None:
        return ScreenState(screenshot_path=screenshot_path)
    h, w = frame_bgr.shape[:2]
    if injected_yolo_boxes is not None:
        yolo_boxes = injected_yolo_boxes
    else:
        yolo_boxes = _run_yolo_on_image(frame_bgr, w, h)
    # OCR on-demand (2026-07-11 用户铁律: OCR 只在读数字/盲区兜底时跑):
    # YOLO ≥3 框 = 已知屏, cls 主导, 整帧 OCR 纯浪费(实测每次 1-1.5s, 旧策略
    # 每 3 tick 一跑拖慢全链)。YOLO <3 框 = 未知屏(羁绊升级/通知弹窗/TOUCH
    # START 等 OCR 拦截器兜底场景)  才跑整帧 OCR。数字读取(票数/AP/总价)
    # 各 skill 本来就走 screen.frame 裁剪 digit-OCR, 不依赖这里。
    # 2026-07-11 用户二次收紧("OCR只有花钱/用票时启动"): 门从 <3框 收到
    # **完全零检出的亮屏**才跑 - 加载中转场(加载中 cls ≥1框)/普通页一律
    # 零 OCR(旧 <3框 门在每个转场帧白跑 1-1.5s); 钱/票数字=skill 内裁剪
    # digit-OCR 天然按需, 与整帧 OCR 无关。
    if skip_ocr and len(yolo_boxes) >= 1:
        ocr_boxes = prev_ocr_boxes if prev_ocr_boxes is not None else []
    else:
        ocr_boxes = _run_ocr_on_image(frame_bgr, w, h)
    return ScreenState(
        ocr_boxes=ocr_boxes,
        yolo_boxes=yolo_boxes,
        image_w=w,
        image_h=h,
        screenshot_path=screenshot_path,
        frame=frame_bgr,   # kept for on-demand digit-OCR cropping
    )
