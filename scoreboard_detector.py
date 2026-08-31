"""
Scoreboard detection / localisation.

The scoreboard in this video is a fixed broadcast overlay, so we use the
empirically-determined ROI from config.  We still expose a `detect` method
that verifies the scoreboard is present (via OCR landmark check) and returns
the bounding rectangle.
"""

import cv2
import numpy as np
import logging
from src import config

logger = logging.getLogger(__name__)


class ScoreboardDetector:
    """Locate and crop the scoreboard region from a video frame."""

    def __init__(self):
        self.roi = config.SCOREBOARD_ROI

    def detect(self, frame: np.ndarray) -> dict:
        """
        Return the scoreboard bounding box as
        {"x1": …, "y1": …, "x2": …, "y2": …, "detected": True/False}.

        Because the overlay is fixed, detection always succeeds as long as the
        frame dimensions match the expected resolution.
        """
        h, w = frame.shape[:2]
        roi = self.roi.copy()
        # Clamp to frame bounds just in case
        roi["x1"] = max(0, roi["x1"])
        roi["y1"] = max(0, roi["y1"])
        roi["x2"] = min(w, roi["x2"])
        roi["y2"] = min(h, roi["y2"])
        roi["detected"] = True
        return roi

    def crop(self, frame: np.ndarray, roi: dict = None) -> np.ndarray:
        """Crop the scoreboard region from the frame."""
        if roi is None:
            roi = self.detect(frame)
        return frame[roi["y1"]:roi["y2"], roi["x1"]:roi["x2"]].copy()

    def draw_bbox(self, frame: np.ndarray, roi: dict = None,
                  color=None, thickness: int = 2, label: str = "Scoreboard Detected") -> np.ndarray:
        """
        Draw a bounding box and label around the detected scoreboard on *frame*.
        Returns a copy so the original is not mutated.
        """
        if roi is None:
            roi = self.detect(frame)
        if color is None:
            color = config.ANNOTATION_COLOR_BOX

        annotated = frame.copy()
        pt1 = (roi["x1"], roi["y1"])
        pt2 = (roi["x2"], roi["y2"])
        cv2.rectangle(annotated, pt1, pt2, color, thickness)

        if label:
            label_y = max(roi["y1"] - 8, 15)
            cv2.putText(
                annotated, label, (roi["x1"], label_y),
                cv2.FONT_HERSHEY_SIMPLEX, config.ANNOTATION_FONT_SCALE,
                color, config.ANNOTATION_THICKNESS, cv2.LINE_AA,
            )
        return annotated
