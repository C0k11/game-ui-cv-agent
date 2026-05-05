"""Unicode-safe image IO helpers.

On Windows, `cv2.imread` silently returns `None` when the path contains
non-ASCII characters (CJK folder names like `角色头像`). The fix is to
read bytes with `numpy.fromfile` (which opens via the Python-level
Unicode path) and decode via `cv2.imdecode`.

All new code should use `imread_any` instead of `cv2.imread`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

_PathLike = Union[str, Path]


def imread_any(path: _PathLike, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Read an image file regardless of path encoding.

    Returns `None` on any failure (missing file, corrupt data, unsupported
    format) — mirroring the `cv2.imread` contract.
    """
    try:
        p = str(path)
        data = np.fromfile(p, dtype=np.uint8)
    except Exception:
        return None
    if data is None or data.size == 0:
        return None
    try:
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_any(
    path: _PathLike,
    img: np.ndarray,
    params: Optional[list] = None,
) -> bool:
    """Write an image file regardless of path encoding. Returns success bool."""
    p = str(path)
    # pick encoder from extension
    ext = "." + p.rsplit(".", 1)[-1].lower() if "." in p else ".png"
    try:
        ok, buf = cv2.imencode(ext, img, params or [])
        if not ok:
            return False
        buf.tofile(p)
        return True
    except Exception:
        return False
