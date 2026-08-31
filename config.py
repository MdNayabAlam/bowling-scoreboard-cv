"""
Configuration constants for the Bowling Scoreboard Extraction pipeline.
All ROI coordinates were determined empirically from the actual video.
"""

import os

# Prevent OpenBLAS / OpenMP thread memory allocation failure on Windows CPU
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VIDEO = os.path.join(PROJECT_ROOT, "data", "bowling_scoreboard.mp4")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
SCREENSHOT_DIR = os.path.join(PROJECT_ROOT, "screenshots")

OUTPUT_JSON = os.path.join(OUTPUT_DIR, "extracted_scoreboard.json")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "extracted_scoreboard.csv")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "annotated_scoreboard.mp4")
OCR_CELLS_DIR = os.path.join(SCREENSHOT_DIR, "ocr_cells")

# ── Video Sampling ───────────────────────────────────────────────────────────
# Sample one frame every SAMPLE_INTERVAL seconds for temporal aggregation.
SAMPLE_INTERVAL_SEC = 1.0

# ── Scoreboard Region-of-Interest (pixels in the 848×478 frame) ─────────────
# The scoreboard is a fixed broadcast overlay; spans all 4 player rows.
SCOREBOARD_ROI = {
    "x1": 30,
    "y1": 60,
    "x2": 825,
    "y2": 355,
}

# ── Grid Layout (absolute pixel coordinates in the full frame) ───────────────
# Column boundaries for Frames 1–10 and the Total (TTL) column.
FRAME_COL_EDGES = [135, 197, 259, 321, 383, 445, 507, 569, 631, 693, 755, 825]

# Each of the 4 player rows (J, V, P, T) is split into a *shot* sub-row (top)
# and a *cumulative score* sub-row (bottom).
PLAYER_ROWS = [
    {"name_label": "J", "y_top": 60, "y_mid": 95, "y_bot": 130},
    {"name_label": "V", "y_top": 130, "y_mid": 165, "y_bot": 205},
    {"name_label": "P", "y_top": 205, "y_mid": 240, "y_bot": 280},
    {"name_label": "T", "y_top": 280, "y_mid": 315, "y_bot": 355},
]

# Player initials / name column
NAME_COL = {"x1": 30, "x2": 135}

# Header row for frame numbers (1..10, TTL)
HEADER_ROW = {"y1": 20, "y2": 60}

# Inner cell padding (insets) to exclude cyan/white grid border lines before OCR
CELL_PADDING = {
    "pad_y": 2,
    "pad_x": 3,
}

# ── Image Pre-processing ────────────────────────────────────────────────────
UPSCALE_FACTOR = 3          # cubic-interpolation scale factor for cell crops
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)

# ── OCR Allow-lists ─────────────────────────────────────────────────────────
OCR_LANGUAGES = ["en"]
OCR_GPU = False
MIN_OCR_CONFIDENCE = 0.15   # low floor; temporal voting cleans noise

# Character whitelists per cell type
ALLOWLIST_NAME = "JVPT"
ALLOWLIST_SHOT = "0123456789X/-"
ALLOWLIST_SCORE = "0123456789"
ALLOWLIST_TOTAL = "0123456789"

# ── Temporal Aggregation ─────────────────────────────────────────────────────
# Minimum number of frames that must agree on a value to accept it.
MIN_VOTE_COUNT = 2

# ── Annotated Video ─────────────────────────────────────────────────────────
ANNOTATION_COLOR_BOX = (0, 255, 0)       # green bounding box
ANNOTATION_COLOR_TEXT = (255, 255, 255)   # white overlay text
ANNOTATION_FONT_SCALE = 0.5
ANNOTATION_THICKNESS = 1

