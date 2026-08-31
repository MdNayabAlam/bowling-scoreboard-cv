"""
OCR engine wrapper with bowling-specific post-processing.

Uses EasyOCR for text recognition. Applies symbol corrections:
  • 'O' → '0', 'I'/'l' → '1', 'S' → '5' (digit context)
  • Recognises bowling marks: X (strike), / (spare), - (miss)
  • Evaluates multiple preprocessing variants and applies bowling-domain rules.
"""

import re
import os
import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-init to avoid loading torch at import time
_reader = None


def _get_reader():
    """Lazy-initialise the EasyOCR Reader (downloads models on first run)."""
    global _reader
    if _reader is None:
        import torch
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        import easyocr
        from src import config
        logger.info("OCR model initialized.")
        print("OCR model initialized.")
        _reader = easyocr.Reader(
            config.OCR_LANGUAGES,
            gpu=config.OCR_GPU,
            verbose=False,
        )
    return _reader


# ── post-processing & domain validation rules ────────────────────────────────

_DIGIT_MAP = str.maketrans({
    "O": "0", "o": "0",
    "I": "1", "l": "1", "i": "1", "|": "1",
    "S": "5", "s": "5",
    "B": "8",
    "Z": "2", "z": "2",
    "G": "6",
    "q": "9",
    "b": "6",
})


def _clean_text(raw: str, cell_type: str) -> str:
    """Normalise OCR string specifically for the given cell_type."""
    text = raw.strip()
    if not text:
        return ""

    if cell_type == "name":
        # Restrict player initials to J, V, P, T
        upper = text.upper()
        for char in upper:
            if char in ("J", "V", "P", "T"):
                return char
        return ""

    if cell_type in ("shot", "shot_1", "shot_2", "shot_3"):
        # Shot symbols: digits, X, /, -
        upper = text.upper()
        cleaned = []
        for ch in text:
            ch_u = ch.upper()
            if ch_u in ("X", "/", "-"):
                cleaned.append(ch_u)
            elif ch_u in ("\\", "|"):
                cleaned.append("/")
            elif ch_u in ("_", "–", "—", "*"):
                cleaned.append("-")
            elif ch in "0123456789":
                cleaned.append(ch)
            elif ch in _DIGIT_MAP:
                cleaned.append(_DIGIT_MAP[ch])
        out = "".join(cleaned)

        if not out:
            return ""

        # Single sub-box shot should be at most 1 character
        if cell_type in ("shot_1", "shot_2", "shot_3"):
            if len(out) == 1:
                return out
            return out[0] if out[0] in "0123456789X/-" else ""

        # Full shot box normalization & spare arithmetic correction (Frames 1-9)
        if len(out) > 2:
            return ""  # Discard invalid noise (>2 chars in standard frames)

        if len(out) == 1:
            return out if out in "0123456789X/-" else ""

        if len(out) == 2:
            c1, c2 = out[0], out[1]
            if c1 == "X" and c2 != "X":
                return "X"
            if c1 != "X" and c2 == "X":
                return ""  # 1X is invalid in frames 1-9

            if c1 == "-" and c2 == "-":
                return "--"
            if c1.isdigit() and c2 == "-":
                return f"{c1}-"
            if c1 == "-" and c2.isdigit():
                return f"-{c2}"
            if c1.isdigit() and c2 == "/":
                return f"{c1}/"

            # 2 digits: check bowling arithmetic!
            if c1.isdigit() and c2.isdigit():
                n1, n2 = int(c1), int(c2)
                if n1 + n2 < 10:
                    return f"{c1}{c2}"  # Valid open frame (e.g. 15, 34, 61)
                else:
                    # n1 + n2 >= 10 is mathematically a spare / (e.g. 74 -> 7/, 91 -> 9/, 18 -> 1/, 28 -> 2/)
                    return f"{c1}/"

        return out

    if cell_type in ("score", "total"):
        # Cumulative score & total: digits only
        cleaned = text.translate(_DIGIT_MAP)
        digits_only = re.sub(r"[^0-9]", "", cleaned)
        if digits_only and is_valid_cumulative_score(digits_only):
            return digits_only
        return ""

    return text


def is_valid_bowling_shot(s: str, is_subbox: bool = False) -> bool:
    """Validate whether string *s* represents valid bowling shot notation."""
    if not s:
        return True
    s = s.strip().upper()
    valid_chars = set("0123456789X/-")
    if not set(s).issubset(valid_chars):
        return False

    if is_subbox:
        return len(s) == 1 and s in valid_chars

    if len(s) == 1:
        return s in valid_chars
    elif len(s) == 2:
        if s in ("--", "-/"):
            return True
        c1, c2 = s[0], s[1]
        if c1 == "X":
            return False  # Strike should be single character in frames 1-9
        if c1.isdigit() and c2 == "/":
            return True
        if c1.isdigit() and c2 == "-":
            return True
        if c1 == "-" and c2.isdigit():
            return True
        if c1.isdigit() and c2.isdigit():
            return True  # Spare conversion handled in _clean_text
        return False
    elif len(s) == 3:
        return True
    return False


def is_valid_cumulative_score(s: str) -> bool:
    """Validate cumulative or total score integer value (0–300)."""
    if not s:
        return True
    if not s.isdigit():
        return False
    val = int(s)
    return 0 <= val <= 300


# ── public API ───────────────────────────────────────────────────────────────

def _get_allowlist(cell_type: str) -> str:
    """Return the character allowlist for EasyOCR based on cell type."""
    from src import config
    if cell_type == "name":
        return config.ALLOWLIST_NAME
    elif cell_type in ("shot", "shot_1", "shot_2", "shot_3"):
        return config.ALLOWLIST_SHOT
    elif cell_type in ("score", "total"):
        return config.ALLOWLIST_SCORE
    return None


def read_cell_image(image: np.ndarray,
                    cell_type: str = "score",
                    min_confidence: float = 0.0) -> list[dict]:
    """
    Run recognition-only EasyOCR on a single pre-processed image array.
    """
    if image is None or image.size == 0:
        return []

    reader = _get_reader()
    allowlist = _get_allowlist(cell_type)
    h, w = image.shape[:2]
    horizontal_box = [[0, w, 0, h]]

    import torch
    with torch.no_grad():
        try:
            results = reader.recognize(
                image,
                horizontal_list=horizontal_box,
                free_list=[],
                allowlist=allowlist,
            )
        except Exception:
            try:
                results = reader.recognize(
                    image,
                    horizontal_list=horizontal_box,
                    free_list=[],
                )
            except Exception:
                results = []

    out = []
    for item in results:
        if len(item) == 3:
            bbox, text, conf = item
        elif len(item) == 2:
            bbox, text = item
            conf = 0.5
        else:
            continue

        if conf < min_confidence:
            continue
        cleaned = _clean_text(text, cell_type)
        if cleaned:
            out.append({"text": cleaned, "confidence": float(conf), "bbox": bbox})
    return out


def read_cell_smart(variants: list[tuple[str, np.ndarray]],
                    cell_type: str = "score",
                    min_confidence: float = 0.15) -> tuple[str, float]:
    """
    Evaluate multiple preprocessing variants and return (best_text, best_confidence).
    Filters candidates using bowling-aware domain rules.
    """
    candidates = []

    for var_name, img in variants:
        items = read_cell_image(img, cell_type=cell_type, min_confidence=min_confidence)
        for item in items:
            txt = item["text"]
            conf = item["confidence"]

            # Domain validation
            is_sub = cell_type in ("shot_1", "shot_2", "shot_3")
            if cell_type in ("shot", "shot_1", "shot_2", "shot_3"):
                if not is_valid_bowling_shot(txt, is_subbox=is_sub):
                    continue
            elif cell_type in ("score", "total"):
                if not is_valid_cumulative_score(txt):
                    continue
            elif cell_type == "name":
                if txt not in ("J", "V", "P", "T"):
                    continue

            candidates.append((txt, conf, var_name))

    if not candidates:
        return "", 0.0

    # For score/total cells, prefer longer valid digit strings if conf is similar
    if cell_type in ("score", "total"):
        candidates.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
    else:
        candidates.sort(key=lambda x: x[1], reverse=True)

    best_txt, best_conf, _ = candidates[0]
    return best_txt, best_conf


def read_cell_simple(image: np.ndarray,
                      cell_type: str = "score",
                      min_confidence: float = 0.0) -> str:
    """Convenience single-image wrapper."""
    items = read_cell_image(image, cell_type, min_confidence)
    if not items:
        return ""
    return items[0]["text"]
