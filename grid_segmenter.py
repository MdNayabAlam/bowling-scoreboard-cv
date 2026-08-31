"""
Grid / cell segmentation for the bowling scoreboard.

The scoreboard grid is defined by the pixel coordinates discovered during
video analysis (see config.py).  This module slices a full video frame into
individual cell crops ready for OCR.
"""

import cv2
import numpy as np
import logging
from src import config

logger = logging.getLogger(__name__)


class CellInfo:
    """Metadata + image crop for one grid cell."""
    __slots__ = ("player_idx", "frame_idx", "cell_type", "crop",
                 "x1", "y1", "x2", "y2")

    def __init__(self, player_idx, frame_idx, cell_type, crop,
                 x1, y1, x2, y2):
        self.player_idx = player_idx   # 0 or 1
        self.frame_idx = frame_idx     # 0–9 for frames 1–10, 10 for TTL
        self.cell_type = cell_type     # "shot" | "score" | "total" | "name"
        self.crop = crop               # numpy BGR image
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def __repr__(self):
        return (f"CellInfo(P{self.player_idx+1}, "
                f"F{self.frame_idx+1 if self.frame_idx < 10 else 'TTL'}, "
                f"{self.cell_type})")


class GridSegmenter:
    """Slice a video frame into individual scoreboard cells."""

    def __init__(self):
        self.col_edges = config.FRAME_COL_EDGES
        self.player_rows = config.PLAYER_ROWS
        self.name_col = config.NAME_COL

    def segment(self, frame: np.ndarray) -> list[CellInfo]:
        """
        Extract all cells from *frame* and return a list of CellInfo objects.

        Inner margin padding (insets) is applied to remove horizontal/vertical
        grid lines before returning crops to OCR.
        """
        cells: list[CellInfo] = []
        pad_y = config.CELL_PADDING["pad_y"]
        pad_x = config.CELL_PADDING["pad_x"]

        for p_idx, row in enumerate(self.player_rows):
            y_top = row["y_top"]
            y_mid = row["y_mid"]
            y_bot = row["y_bot"]
            name_x1 = self.name_col["x1"]
            name_x2 = self.name_col["x2"]

            # Player name cell (padded crop from top sub-cell)
            name_crop = frame[y_top + pad_y : y_mid - pad_y, name_x1 + pad_x : name_x2 - pad_x]
            cells.append(CellInfo(
                p_idx, -1, "name", name_crop,
                name_x1, y_top, name_x2, y_mid,
            ))

            # Frame columns 1–10 + Total
            for f_idx in range(len(self.col_edges) - 1):
                x1 = self.col_edges[f_idx]
                x2 = self.col_edges[f_idx + 1]
                is_total = (f_idx == len(self.col_edges) - 2)

                if is_total:
                    # Total column score lives in bottom sub-row (y_mid to y_bot)
                    total_crop = frame[y_mid + pad_y : y_bot - pad_y, x1 + pad_x : x2 - pad_x]
                    cells.append(CellInfo(
                        p_idx, f_idx, "total", total_crop,
                        x1, y_mid, x2, y_bot,
                    ))
                else:
                    # Full shot crop (top sub-row)
                    shot_full_crop = frame[y_top + pad_y : y_mid - pad_y, x1 + pad_x : x2 - pad_x]
                    cells.append(CellInfo(
                        p_idx, f_idx, "shot", shot_full_crop,
                        x1, y_top, x2, y_mid,
                    ))

                    # Shot sub-boxes (Shot 1, Shot 2, and Shot 3 for 10th frame)
                    w = x2 - x1
                    if f_idx == 9:  # 10th frame has 3 shot slots
                        w3 = w / 3.0
                        xm1 = x1 + int(round(w3))
                        xm2 = x1 + int(round(2 * w3))

                        s1_crop = frame[y_top + pad_y : y_mid - pad_y, x1 + pad_x : xm1 - pad_x]
                        s2_crop = frame[y_top + pad_y : y_mid - pad_y, xm1 + pad_x : xm2 - pad_x]
                        s3_crop = frame[y_top + pad_y : y_mid - pad_y, xm2 + pad_x : x2 - pad_x]

                        cells.append(CellInfo(p_idx, f_idx, "shot_1", s1_crop, x1, y_top, xm1, y_mid))
                        cells.append(CellInfo(p_idx, f_idx, "shot_2", s2_crop, xm1, y_top, xm2, y_mid))
                        cells.append(CellInfo(p_idx, f_idx, "shot_3", s3_crop, xm2, y_top, x2, y_mid))
                    else:
                        x_mid = x1 + w // 2
                        s1_crop = frame[y_top + pad_y : y_mid - pad_y, x1 + pad_x : x_mid - pad_x]
                        s2_crop = frame[y_top + pad_y : y_mid - pad_y, x_mid + pad_x : x2 - pad_x]

                        cells.append(CellInfo(p_idx, f_idx, "shot_1", s1_crop, x1, y_top, x_mid, y_mid))
                        cells.append(CellInfo(p_idx, f_idx, "shot_2", s2_crop, x_mid, y_top, x2, y_mid))

                    # Cumulative score sub-cell (bottom sub-row)
                    score_crop = frame[y_mid + pad_y : y_bot - pad_y, x1 + pad_x : x2 - pad_x]
                    cells.append(CellInfo(
                        p_idx, f_idx, "score", score_crop,
                        x1, y_mid, x2, y_bot,
                    ))

        return cells

    def draw_grid(self, frame: np.ndarray,
                  color=(0, 255, 255), thickness: int = 1) -> np.ndarray:
        """
        Draw the segmentation grid on a copy of *frame* for visual debugging.
        """
        vis = frame.copy()
        for row in self.player_rows:
            for y in (row["y_top"], row["y_mid"], row["y_bot"]):
                cv2.line(vis, (self.name_col["x1"], y),
                         (self.col_edges[-1], y), color, thickness)

        # Vertical column lines
        for x in self.col_edges:
            cv2.line(vis, (x, self.player_rows[0]["y_top"]),
                     (x, self.player_rows[-1]["y_bot"]), color, thickness)

        # Name column
        cv2.line(vis, (self.name_col["x1"], self.player_rows[0]["y_top"]),
                 (self.name_col["x1"], self.player_rows[-1]["y_bot"]),
                 color, thickness)
        cv2.line(vis, (self.name_col["x2"], self.player_rows[0]["y_top"]),
                 (self.name_col["x2"], self.player_rows[-1]["y_bot"]),
                 color, thickness)

        # Labels
        for i, row in enumerate(self.player_rows):
            mid_y = (row["y_top"] + row["y_bot"]) // 2
            cv2.putText(vis, f"P{i+1}",
                        (self.name_col["x1"] + 2, mid_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        for f_idx in range(len(self.col_edges) - 1):
            x_mid = (self.col_edges[f_idx] + self.col_edges[f_idx + 1]) // 2
            label = str(f_idx + 1) if f_idx < 10 else "TTL"
            cv2.putText(vis, label,
                        (x_mid - 8, self.player_rows[0]["y_top"] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        return vis
