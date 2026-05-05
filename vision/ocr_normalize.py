"""OCR text normalization for Blue Archive UI recognition.

Two-stage pipeline applied to both OCR output (`box.text`) and match
patterns before substring comparison:

  1. Apply a learned CORRECTIONS dictionary — maps known misreads back
     to the canonical form (e.g. "Duest" → "Quest", "辨中！" → "辦中！").
  2. Fold Traditional ↔ Simplified CJK via a small char table covering
     every char that appears in the BA vocabulary. This eliminates the
     "任務資讯" (mixed Trad/Simp) failure class without needing to list
     every permutation in keyword literals.

`data/ocr_corrections.json` is the runtime-mutable corrections store;
`scripts/mine_ocr_corrections.py` populates it from trajectory mining.
`scripts/ocr_training/ba_vocab.py::CORRECTIONS` is merged in as the
hand-curated baseline.

All functions are pure and cached — call `normalize(text)` repeatedly
with no IO cost after first invocation.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict

_REPO = Path(__file__).resolve().parents[1]
_CORR_JSON = _REPO / "data" / "ocr_corrections.json"


# ── Traditional → Simplified char folding table ─────────────────────
# Covers every TC char that appears in brain/skills/ keyword literals.
# Kept inline (no external dep) because we only need ~200 pairs.
# Build this by inspecting TC ba_vocab entries; SC is the canonical form
# chosen here so OCR output in either form collapses to a single key.
_TC_TO_SC: Dict[str, str] = {
    # Common UI / buttons
    "務": "务", "資": "资", "訊": "讯", "確": "确", "關": "关",
    "閉": "闭", "開": "开", "領": "领", "繼": "继", "續": "续",
    "選": "选", "擇": "择", "過": "过", "鍾": "钟", "爾": "尔",
    "與": "与", "內": "内", "當": "当", "節": "节", "點": "点",
    "時": "时", "間": "间", "達": "达", "場": "场", "種": "种",
    "類": "类", "劃": "划", "則": "则", "個": "个", "優": "优",
    "萬": "万", "電": "电", "話": "话", "語": "语", "應": "应",
    "準": "准", "備": "备", "處": "处", "這": "这", "裡": "里",
    "後": "后", "實": "实", "際": "际", "總": "总", "結": "结",
    "為": "为", "無": "无", "對": "对", "異": "异", "頻": "频",
    "現": "现", "況": "况", "雖": "虽", "權": "权", "條": "条",
    "會": "会", "團": "团", "園": "园", "區": "区", "職": "职",
    "業": "业", "統": "统", "級": "级", "戰": "战", "鬥": "斗",
    "術": "术", "傳": "传", "聞": "闻", "飛": "飞", "遊": "游",
    "戲": "戏", "網": "网", "頁": "页", "樣": "样", "單": "单",
    "顯": "显", "誌": "志", "認": "认", "讓": "让", "訪": "访",
    "問": "问", "請": "请", "邀": "邀", "隨": "随", "機": "机",
    "號": "号", "動": "动", "進": "进", "還": "还", "剩": "剩",
    "獎": "奖", "勵": "励", "獲": "获", "得": "得", "贈": "赠",
    "購": "购", "買": "买", "賣": "卖", "賞": "赏", "緝": "缉",
    "懸": "悬", "總": "总", "戰": "战", "學": "学", "師": "师",
    "競": "竞", "賽": "赛", "課": "课", "務": "务", "劇": "剧",
    "離": "离", "結": "结", "獎": "奖", "活": "活", "動": "动",
    # Cafe / schedule specific
    "舒": "舒", "適": "适", "度": "度", "廳": "厅", "禮": "礼",
    "預": "预", "設": "设", "體": "体", "收": "收", "納": "纳",
    "說": "说", "聯": "联", "絡": "络", "記": "记", "錄": "录",
    # Battle / quest
    "戰": "战", "鬥": "斗", "擊": "击", "撃": "击", "勝": "胜",
    "敗": "败", "勁": "劲", "難": "难", "卡": "卡", "輪": "轮",
    "排": "排", "鐵": "铁", "銀": "银", "號": "号", "點": "点",
    "燒": "烧", "發": "发", "驗": "验", "檢": "检", "辦": "办",
    "舉": "举", "壓": "压", "擴": "扩", "張": "张", "稱": "称",
    "項": "项", "萊": "莱", "雜": "杂", "給": "给", "禍": "祸",
    "務": "务", "誌": "志", "讀": "读",
    # Schools / places
    "黑": "黑", "娜": "娜", "格": "格", "千": "千", "年": "年",
    "聖": "圣", "三": "三", "埃": "埃", "德": "德", "卡": "卡",
    "補": "补", "償": "偿", "訂": "订", "項": "项",
}


def _tc_to_sc(text: str) -> str:
    """Fold Traditional chars to Simplified where we have a mapping."""
    if not text:
        return text
    out = []
    for ch in text:
        out.append(_TC_TO_SC.get(ch, ch))
    return "".join(out)


# ── Corrections loading ─────────────────────────────────────────────
# Merges hand-curated ba_vocab.CORRECTIONS with auto-mined
# data/ocr_corrections.json. Cached at import.

def _load_base_corrections() -> Dict[str, str]:
    """Load curated corrections from ba_vocab.py if importable."""
    try:
        import sys
        path = str(_REPO / "scripts" / "ocr_training")
        if path not in sys.path:
            sys.path.insert(0, path)
        from ba_vocab import CORRECTIONS as _BC  # type: ignore
        return dict(_BC)
    except Exception:
        return {}


def _load_mined_corrections() -> Dict[str, str]:
    """Load auto-mined corrections from data/ocr_corrections.json."""
    try:
        if _CORR_JSON.exists():
            raw = json.loads(_CORR_JSON.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return {str(k): str(v) for k, v in raw.items() if k and v}
    except Exception:
        pass
    return {}


_CORRECTIONS: Dict[str, str] = {**_load_base_corrections(), **_load_mined_corrections()}
# Pre-sort by descending length so longer, more-specific misreads replace first.
_SORTED_CORR_KEYS = sorted(_CORRECTIONS.keys(), key=len, reverse=True)


def apply_corrections(text: str) -> str:
    """Replace known misreads in-place (longest match first)."""
    if not text or not _CORRECTIONS:
        return text
    # Exact full-string match first (fastest).
    if text in _CORRECTIONS:
        return _CORRECTIONS[text]
    out = text
    for k in _SORTED_CORR_KEYS:
        if k in out:
            out = out.replace(k, _CORRECTIONS[k])
    return out


@lru_cache(maxsize=4096)
def normalize(text: str) -> str:
    """Run corrections → Trad→Simp fold. Safe for both patterns and OCR text."""
    if not text:
        return text
    return _tc_to_sc(apply_corrections(text))


def reload() -> None:
    """Reload corrections from disk. For use after mining scripts update JSON."""
    global _CORRECTIONS, _SORTED_CORR_KEYS
    _CORRECTIONS = {**_load_base_corrections(), **_load_mined_corrections()}
    _SORTED_CORR_KEYS = sorted(_CORRECTIONS.keys(), key=len, reverse=True)
    normalize.cache_clear()


def stats() -> Dict[str, int]:
    return {
        "tc_sc_pairs": len(_TC_TO_SC),
        "corrections": len(_CORRECTIONS),
    }
