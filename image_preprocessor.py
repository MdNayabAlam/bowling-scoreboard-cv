"""
Image pre-processing pipeline tuned for the bowling scoreboard video.

Steps applied to each cell crop before OCR:
  1. BGR → Grayscale
  2. CLAHE contrast enhancement
  3. Cubic-interpolation upscaling (3×)
  4. Optional Otsu / adaptive thresholding
  5. Optional sharpening
"""

import cv2
import numpy as np
import logging
from src import config

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Apply a series of image enhancements to improve OCR accuracy."""

    def __init__(self,
                 upscale: int = None,
                 clahe_clip: float = None,
                 clahe_tile: tuple = None):
        self.upscale = upscale or config.UPSCALE_FACTOR
        self.clahe_clip = clahe_clip or config.CLAHE_CLIP_LIMIT
        self.clahe_tile = clahe_tile or config.CLAHE_TILE_SIZE
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip, tileGridSize=self.clahe_tile
        )

    # ── public API ───────────────────────────────────────────────────────────
    def preprocess_for_ocr(self, crop: np.ndarray) -> np.ndarray:
        """
        Standard pipeline: grayscale → CLAHE → upscale.
        Returns a single-channel image suitable for EasyOCR.
        """
        gray = self._to_gray(crop)
        enhanced = self._clahe.apply(gray)
        scaled = self._upscale(enhanced)
        return scaled

    def preprocess_cell_variants(self, crop: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """
        Generate multiple preprocessing variants for robust OCR:
          - Variant A: Standard CLAHE + Upscale + White Border
          - Variant B: Inverted CLAHE + Upscale + Black Border
          - Variant C: Otsu Binary Threshold + Upscale + White Border
          - Variant D: Inverted Otsu Binary Threshold + Upscale + White Border
        """
        if crop is None or crop.size == 0:
            return []

        gray = self._to_gray(crop)
        mean_bright = np.mean(gray)

        # Standard CLAHE + upscale
        enhanced = self._clahe.apply(gray)
        scaled = self._upscale(enhanced)

        # Variant A: Standard (good for dark text on light background or standard)
        var_a = self.add_border_padding(scaled, pad=12, value=255)

        # Variant B: Inverted (good for light text on dark background or active row highlights)
        inv_gray = 255 - gray
        enhanced_inv = self._clahe.apply(inv_gray)
        scaled_inv = self._upscale(enhanced_inv)
        var_b = self.add_border_padding(scaled_inv, pad=12, value=0)

        # Variant C: Otsu threshold
        _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        var_c = self.add_border_padding(thresh, pad=12, value=255)

        # Variant D: Inverted Otsu threshold
        _, thresh_inv = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        var_d = self.add_border_padding(thresh_inv, pad=12, value=255)

        # Return variants ordered according to background brightness heuristic
        if mean_bright > 140:
            # Active yellow row: inverted variants may work better
            return [("var_b", var_b), ("var_a", var_a), ("var_d", var_d), ("var_c", var_c)]
        else:
            # Dark/blue background
            return [("var_a", var_a), ("var_c", var_c), ("var_b", var_b), ("var_d", var_d)]

    def add_border_padding(self, img: np.ndarray, pad: int = 12, value: int = 255) -> np.ndarray:
        """Add margin padding around image so characters do not touch borders."""
        if len(img.shape) == 3:
            border_val = [value, value, value]
        else:
            border_val = value
        return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=border_val)

    def preprocess_with_threshold(self, crop: np.ndarray) -> np.ndarray:
        """Like preprocess_for_ocr but adds Otsu thresholding."""
        processed = self.preprocess_for_ocr(crop)
        _, thresh = cv2.threshold(processed, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def preprocess_full_scoreboard(self, scoreboard_crop: np.ndarray) -> dict:
        """
        Return a dict of intermediate images for debugging / screenshots:
          "original", "grayscale", "enhanced", "thresholded"
        """
        gray = self._to_gray(scoreboard_crop)
        enhanced = self._clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return {
            "original": scoreboard_crop.copy(),
            "grayscale": gray,
            "enhanced": enhanced,
            "thresholded": thresh,
        }

    # ── internals ────────────────────────────────────────────────────────────
    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _upscale(self, img: np.ndarray) -> np.ndarray:
        if self.upscale <= 1:
            return img
        return cv2.resize(img, (0, 0),
                          fx=self.upscale, fy=self.upscale,
                          interpolation=cv2.INTER_CUBIC)
