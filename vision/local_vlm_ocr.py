import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor


try:
    from transformers import AutoModelForVision2Seq  # type: ignore
except Exception:  # pragma: no cover
    AutoModelForVision2Seq = None  # type: ignore


try:
    from transformers import AutoModelForImageTextToText  # type: ignore
except Exception:  # pragma: no cover
    AutoModelForImageTextToText = None  # type: ignore


try:
    from transformers import AutoModelForCausalLM  # type: ignore
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore


try:
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover
    snapshot_download = None  # type: ignore


@dataclass
class LocalVlmConfig:
    model: str
    models_dir: str
    hf_home: str
    device: str = "cuda"
    max_new_tokens: int = 2048


class LocalVlmOcr:
    def __init__(self, cfg: LocalVlmConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._model = None
        self._processor = None

    def ensure_loaded(self) -> None:
        self._ensure_loaded()

    def _set_hf_cache(self) -> None:
        os.environ.setdefault("HF_HOME", self.cfg.hf_home)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", self.cfg.hf_home)

    def _resolve_model_path(self) -> Tuple[str, Optional[str]]:
        m = (self.cfg.model or "").strip()
        if not m:
            raise ValueError("model is required")

        p = Path(m)
        if p.exists() and p.is_dir():
            return str(p), None

        models_dir = Path(self.cfg.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        local_dir = models_dir / m.replace("/", "__").replace(":", "__")

        if local_dir.exists() and local_dir.is_dir() and any(local_dir.iterdir()):
            return str(local_dir), m

        if snapshot_download is None:
            raise ImportError("huggingface_hub is required to download models")

        snapshot_download(
            repo_id=m,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        return str(local_dir), m

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._model is not None and self._processor is not None:
                return

            self._set_hf_cache()
            model_path, _ = self._resolve_model_path()

            self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

            def _load_model(*, use_cuda: bool):
                model2 = None
                last_err2: Optional[Exception] = None
                if AutoModelForImageTextToText is not None:
                    try:
                        model2 = AutoModelForImageTextToText.from_pretrained(
                            model_path,
                            torch_dtype="auto",
                            device_map="auto" if use_cuda else None,
                            trust_remote_code=True,
                        )
                    except Exception as e:
                        last_err2 = e
                        model2 = None

                if model2 is None and AutoModelForVision2Seq is not None:
                    try:
                        model2 = AutoModelForVision2Seq.from_pretrained(
                            model_path,
                            torch_dtype="auto",
                            device_map="auto" if use_cuda else None,
                            trust_remote_code=True,
                        )
                    except Exception as e:
                        last_err2 = e
                        model2 = None

                if model2 is None:
                    allow_text_fb = (os.environ.get("LOCAL_VLM_ALLOW_TEXT_FALLBACK") or "").strip() == "1"
                    if allow_text_fb and AutoModelForCausalLM is not None:
                        model2 = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            torch_dtype="auto",
                            device_map="auto" if use_cuda else None,
                            trust_remote_code=True,
                        )
                    else:
                        if last_err2 is None:
                            raise RuntimeError("No compatible vision model loader found in transformers")
                        raise RuntimeError(f"Failed to load vision model: {last_err2}")
                return model2

            is_cuda = str(self.cfg.device or "").lower().startswith("cuda")
            try:
                model = _load_model(use_cuda=bool(is_cuda))
            except Exception as e:
                msg = str(e).lower()
                fb = (os.environ.get("LOCAL_VLM_FALLBACK_CPU_ON_OOM") or "1").strip()
                allow_fb = fb not in ("0", "false", "False")
                if allow_fb and is_cuda and ("out of memory" in msg or "cuda" in msg and "memory" in msg):
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    self.cfg.device = "cpu"
                    model = _load_model(use_cuda=False)
                else:
                    raise

            self._model = model.eval()

    def ocr(self, *, image_path: str, prompt: str, max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        self._ensure_loaded()
        assert self._model is not None
        assert self._processor is not None

        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        if hasattr(self._processor, "apply_chat_template"):
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt

        inputs = self._processor(text=[text], images=[image], return_tensors="pt")

        if hasattr(self._model, "device"):
            device = getattr(self._model, "device")
            try:
                inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
            except Exception:
                pass

        with torch.inference_mode():
            mt = None
            try:
                s = (os.environ.get("LOCAL_VLM_MAX_TIME_S") or "").strip()
                if s:
                    mt = float(s)
            except Exception:
                mt = None
            if mt is None:
                mt = 25.0
            generated = self._model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens) if max_new_tokens else int(self.cfg.max_new_tokens),
                do_sample=False,
                max_time=float(mt),
            )

        if isinstance(inputs, dict) and "input_ids" in inputs:
            try:
                generated = generated[:, inputs["input_ids"].shape[1] :]
            except Exception:
                pass

        out = self._processor.batch_decode(generated, skip_special_tokens=True)
        text_out = out[0] if out else ""
        return {"raw": text_out}
