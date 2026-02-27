from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union


@dataclass
class TemplateMatch:
    template: str
    score: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]


class Cerebellum:
    def __init__(self, *, assets_dir: Union[str, Path], confidence: float = 0.20) -> None:
        self.assets_dir = Path(assets_dir).expanduser().resolve()
        self.confidence = float(confidence)
        self._tmpl_cache: Dict[str, Any] = {}

    def _imread(self, p: Path, flags: int):
        import cv2  # type: ignore

        try:
            import numpy as np  # type: ignore
        except Exception:
            np = None  # type: ignore

        if np is None:
            return cv2.imread(str(p), flags)

        try:
            data = p.read_bytes()
        except Exception:
            return cv2.imread(str(p), flags)
        try:
            buf = np.frombuffer(data, dtype=np.uint8)
            return cv2.imdecode(buf, flags)
        except Exception:
            return cv2.imread(str(p), flags)

    def _load_template(self, template_name: str):
        if template_name in self._tmpl_cache:
            return self._tmpl_cache[template_name]
        p = (self.assets_dir / template_name).resolve()
        if not p.exists() or not p.is_file():
            self._tmpl_cache[template_name] = None
            return None
        import cv2  # type: ignore

        img = self._imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            self._tmpl_cache[template_name] = None
            return None
        mask = None
        bgr = None
        try:
            if len(getattr(img, "shape", ())) == 3 and int(img.shape[2]) == 4:
                bgr = img[:, :, :3]
                alpha = img[:, :, 3]
                try:
                    mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)[1]
                except Exception:
                    mask = None
                # Discard fully-opaque masks (no transparent pixels).
                # A useless mask triggers TM_CCORR_NORMED which gives
                # high scores at WRONG positions for many templates.
                if mask is not None:
                    try:
                        import numpy as _np  # type: ignore
                        if int(_np.count_nonzero(mask == 0)) == 0:
                            mask = None
                    except Exception:
                        pass
            elif len(getattr(img, "shape", ())) == 2:
                bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                bgr = img
        except Exception:
            bgr = None
            mask = None

        if bgr is None:
            self._tmpl_cache[template_name] = None
            return None

        try:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            self._tmpl_cache[template_name] = None
            return None

        info = {
            "bgr": bgr,
            "gray": gray,
            "mask": mask,
        }
        self._tmpl_cache[template_name] = info
        return info

    def best_match(
        self,
        *,
        screenshot_path: Union[str, Path],
        template_name: str,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[TemplateMatch]:
        import cv2  # type: ignore

        info = self._load_template(str(template_name))
        if info is None:
            return None
        try:
            tmpl_g = info.get("gray")
            tmpl_mask = info.get("mask")
        except Exception:
            tmpl_g = None
            tmpl_mask = None
        if tmpl_g is None:
            return None

        sp = Path(screenshot_path).resolve()
        scr = self._imread(sp, cv2.IMREAD_COLOR)
        if scr is None:
            return None

        th, tw = tmpl_g.shape[:2]
        sh, sw = scr.shape[:2]
        if tw <= 1 or th <= 1 or sw <= 1 or sh <= 1:
            return None
        if tw > sw or th > sh:
            return None

        scr_g = cv2.cvtColor(scr, cv2.COLOR_BGR2GRAY)

        x_off = 0
        y_off = 0
        if roi is not None:
            try:
                x0, y0, x1, y1 = [int(v) for v in roi]
                x0 = max(0, min(int(sw) - 1, int(x0)))
                y0 = max(0, min(int(sh) - 1, int(y0)))
                x1 = max(int(x0) + 1, min(int(sw), int(x1)))
                y1 = max(int(y0) + 1, min(int(sh), int(y1)))
                if (x1 - x0) >= int(tw) and (y1 - y0) >= int(th):
                    scr_g = scr_g[int(y0) : int(y1), int(x0) : int(x1)]
                    x_off = int(x0)
                    y_off = int(y0)
            except Exception:
                x_off = 0
                y_off = 0

        method = cv2.TM_CCOEFF_NORMED
        use_mask = tmpl_mask is not None
        if use_mask:
            method = cv2.TM_CCORR_NORMED

        if use_mask:
            res = cv2.matchTemplate(scr_g, tmpl_g, method, mask=tmpl_mask)
        else:
            res = cv2.matchTemplate(scr_g, tmpl_g, method)
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
        score = float(max_val)

        x1 = int(max_loc[0]) + int(x_off)
        y1 = int(max_loc[1]) + int(y_off)
        x2 = int(x1 + tw)
        y2 = int(y1 + th)
        cx = int(x1 + tw // 2)
        cy = int(y1 + th // 2)
        return TemplateMatch(
            template=str(template_name),
            score=float(score),
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            center=(int(cx), int(cy)),
        )

    def match(
        self,
        *,
        screenshot_path: Union[str, Path],
        template_name: str,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[TemplateMatch]:
        m = self.best_match(screenshot_path=screenshot_path, template_name=template_name, roi=roi)
        if m is None:
            return None
        if float(m.score) < float(self.confidence):
            return None
        return m

    def click_action(
        self,
        *,
        screenshot_path: Union[str, Path],
        template_name: str,
        reason_prefix: str,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[Dict[str, Any]]:
        m = self.match(screenshot_path=screenshot_path, template_name=template_name, roi=roi)
        if m is None:
            return None
        x, y = m.center
        return {
            "action": "click",
            "target": [int(x), int(y)],
            "reason": f"{reason_prefix} template={m.template} score={m.score:.3f}",
            "_cerebellum": {
                "template": m.template,
                "score": float(m.score),
                "bbox": [int(m.bbox[0]), int(m.bbox[1]), int(m.bbox[2]), int(m.bbox[3])],
                "center": [int(x), int(y)],
            },
        }
