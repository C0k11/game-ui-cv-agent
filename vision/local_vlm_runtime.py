import os
import queue
import threading
import time
from multiprocessing import Process, Queue
from typing import Any, Dict, Optional

from vision.local_vlm_ocr import LocalVlmConfig, LocalVlmOcr


_LOCAL_VLM_LOCK = threading.Lock()
_LOCAL_VLM: Optional[LocalVlmOcr] = None
_LOCAL_VLM_KEY: str = ""


def _vlm_worker_main(cfg_dict: Dict[str, Any], req_q: Queue, resp_q: Queue) -> None:
    cfg = LocalVlmConfig(
        model=str(cfg_dict.get("model") or ""),
        models_dir=str(cfg_dict.get("models_dir") or ""),
        hf_home=str(cfg_dict.get("hf_home") or ""),
        device=str(cfg_dict.get("device") or "cuda"),
        max_new_tokens=int(cfg_dict.get("max_new_tokens") or 2048),
        lora_path=str(cfg_dict.get("lora_path") or "") if cfg_dict.get("lora_path") else None,
    )
    engine = LocalVlmOcr(cfg)
    ready_err = ""
    try:
        engine.ensure_loaded()
    except Exception as e:
        ready_err = str(e)
    try:
        resp_q.put({"type": "ready", "error": ready_err})
    except Exception:
        pass
    while True:
        req = req_q.get()
        if req is None:
            break
        rid = req.get("id")
        try:
            out = engine.ocr(
                image_path=str(req.get("image_path") or ""),
                prompt=str(req.get("prompt") or ""),
                max_new_tokens=req.get("max_new_tokens"),
            )
            resp_q.put({"id": rid, "raw": out.get("raw", "")})
        except Exception as e:
            resp_q.put({"id": rid, "raw": "", "error": str(e)})


class _SubprocessVlm:
    def __init__(self, cfg: LocalVlmConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._proc: Optional[Process] = None
        self._req_q: Optional[Queue] = None
        self._resp_q: Optional[Queue] = None
        self._pending: Dict[int, Dict[str, Any]] = {}
        self._ready: bool = False
        self._ready_error: str = ""
        self._cooldown_until: float = 0.0
        self._cooldown_error: str = ""

    def _ensure_proc(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.is_alive():
                return
            self._pending = {}
            self._ready = False
            self._ready_error = ""
            self._req_q = Queue()
            self._resp_q = Queue()
            cfg_dict = {
                "model": self.cfg.model,
                "models_dir": self.cfg.models_dir,
                "hf_home": self.cfg.hf_home,
                "device": self.cfg.device,
                "max_new_tokens": self.cfg.max_new_tokens,
                "lora_path": self.cfg.lora_path,
            }
            p = Process(target=_vlm_worker_main, args=(cfg_dict, self._req_q, self._resp_q), daemon=True)
            p.start()
            self._proc = p

        self._wait_ready()

    def _wait_ready(self) -> None:
        try:
            startup_to = float(os.environ.get("LOCAL_VLM_STARTUP_TIMEOUT_S", "180"))
        except Exception:
            startup_to = 180.0
        deadline = time.time() + max(5.0, startup_to)

        while True:
            if self._ready_error:
                raise RuntimeError(self._ready_error)
            if self._ready:
                return
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("vlm_worker_startup_timeout")
            try:
                assert self._resp_q is not None
                msg = self._resp_q.get(timeout=min(0.5, remaining))
            except queue.Empty:
                continue
            self._handle_resp_msg(msg)

    def _shutdown_proc(self) -> None:
        with self._lock:
            try:
                if self._proc is not None and self._proc.is_alive():
                    self._proc.terminate()
            except Exception:
                pass
            try:
                if self._proc is not None:
                    self._proc.join(timeout=1.0)
            except Exception:
                pass
            self._proc = None
            self._req_q = None
            self._resp_q = None
            self._pending = {}
            self._ready = False
            self._ready_error = ""

    def _handle_resp_msg(self, msg: Dict[str, Any]) -> None:
        if not isinstance(msg, dict):
            return
        mtype = str(msg.get("type") or "")
        if mtype == "ready":
            self._ready = True
            self._ready_error = str(msg.get("error") or "")
            return
        try:
            mid = int(msg.get("id"))
        except Exception:
            return
        self._pending[mid] = msg

    def _restart_proc(self) -> None:
        with self._lock:
            try:
                if self._proc is not None and self._proc.is_alive():
                    self._proc.terminate()
            except Exception:
                pass
            try:
                if self._proc is not None:
                    self._proc.join(timeout=1.0)
            except Exception:
                pass
            self._proc = None
            self._req_q = None
            self._resp_q = None
            self._pending = {}
            self._ready = False
            self._ready_error = ""
        self._ensure_proc()

    def ensure_loaded(self) -> None:
        self._ensure_proc()
        if not self._ready:
            self._wait_ready()

    def ocr(self, *, image_path: str, prompt: str, max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        try:
            if self._cooldown_until and time.time() < float(self._cooldown_until):
                return {"raw": "", "error": str(self._cooldown_error or "startup_error:cooldown")}
        except Exception:
            pass
        self._ensure_proc()
        if not self._ready:
            try:
                self._wait_ready()
            except Exception as e:
                err = str(e)
                low = err.lower()
                if "paging file" in low or "os error 1455" in low or " 1455" in low or low.endswith("1455"):
                    try:
                        cd = float(os.environ.get("LOCAL_VLM_STARTUP_COOLDOWN_S", "60"))
                    except Exception:
                        cd = 60.0
                    try:
                        self._cooldown_error = f"startup_error:{err}"
                        self._cooldown_until = time.time() + max(5.0, float(cd))
                    except Exception:
                        pass
                    try:
                        self._shutdown_proc()
                    except Exception:
                        pass
                    return {"raw": "", "error": f"startup_error:{err}"}
                self._restart_proc()
                return {"raw": "", "error": f"startup_error:{err}"}
        assert self._req_q is not None
        assert self._resp_q is not None

        rid = int(time.time_ns() & 0x7FFFFFFF)
        self._req_q.put({"id": rid, "image_path": image_path, "prompt": prompt, "max_new_tokens": max_new_tokens})

        try:
            to_s = float(os.environ.get("LOCAL_VLM_HARD_TIMEOUT_S", "35"))
        except Exception:
            to_s = 35.0
        deadline = time.time() + max(1.0, to_s)

        while True:
            if rid in self._pending:
                return self._pending.pop(rid)
            remaining = deadline - time.time()
            if remaining <= 0:
                self._restart_proc()
                return {"raw": "", "error": "hard_timeout"}
            try:
                msg = self._resp_q.get(timeout=min(0.5, remaining))
            except queue.Empty:
                continue
            self._handle_resp_msg(msg)
            if rid in self._pending:
                return self._pending.pop(rid)


def get_local_vlm(*, model: str, models_dir: str, hf_home: str, device: str):
    global _LOCAL_VLM, _LOCAL_VLM_KEY
    try:
        use_sub = (os.environ.get("LOCAL_VLM_SUBPROCESS") or "1").strip() not in ("0", "false", "False")
    except Exception:
        use_sub = True
    key = f"{model}|{models_dir}|{hf_home}|{device}|sub={int(use_sub)}"
    with _LOCAL_VLM_LOCK:
        if _LOCAL_VLM is not None and _LOCAL_VLM_KEY == key:
            return _LOCAL_VLM

        cfg = LocalVlmConfig(
            model=model,
            models_dir=models_dir,
            hf_home=hf_home,
            device=device,
            max_new_tokens=2048,
        )
        if use_sub:
            _LOCAL_VLM = _SubprocessVlm(cfg)  # type: ignore[assignment]
        else:
            _LOCAL_VLM = LocalVlmOcr(cfg)
        _LOCAL_VLM_KEY = key
        return _LOCAL_VLM
