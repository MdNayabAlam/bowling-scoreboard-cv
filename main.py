#!/usr/bin/env python3
"""
Bowling Scoreboard Data Extraction – main entry point.

Usage:
    python src/main.py --video data/bowling_scoreboard.mp4
"""

import os
import sys

# Set OpenBLAS/MKL thread limits BEFORE cv2, numpy, or torch are imported
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

import argparse
import cv2
import logging
import time

# ── ensure project root is on sys.path ───────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import config
from src.video_processor import VideoProcessor
from src.scoreboard_detector import ScoreboardDetector
from src.image_preprocessor import ImagePreprocessor
from src.grid_segmenter import GridSegmenter
from src.ocr_processor import read_cell_smart
from src.temporal_aggregator import TemporalAggregator
from src.data_extractor import build_structured_output, save_json, save_csv
from src.visualizer import (
    save_screenshot, create_annotated_video, save_final_output_image,
)


def setup_logging():
    """Configure console logging with a clean format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def banner(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def main(video_path: str):
    setup_logging()
    cv2.setNumThreads(1)
    logger = logging.getLogger(__name__)

    banner("BOWLING SCOREBOARD DATA EXTRACTION")

    # ── 1. Load video ────────────────────────────────────────────────────────
    print("Loading video…")
    if not os.path.isfile(video_path):
        print(f"ERROR: Video file not found → {video_path}")
        sys.exit(1)

    vp = VideoProcessor(video_path)
    meta = vp.get_metadata()
    print(f"  Video resolution : {meta['resolution']}")
    print(f"  FPS              : {meta['fps']}")
    print(f"  Duration         : {meta['duration_seconds']}s")
    print(f"  Frame count      : {meta['frame_count']}")

    # ── 2. Initialise components ─────────────────────────────────────────────
    detector = ScoreboardDetector()
    preprocessor = ImagePreprocessor()
    segmenter = GridSegmenter()
    aggregator = TemporalAggregator()

    # ── 3. Grab one reference frame for screenshots ──────────────────────────
    print("\nAnalysing scoreboard layout…")
    ref_frame = vp.read_frame_at(5.0)
    if ref_frame is None:
        ref_frame = vp.read_frame_at(1.0)

    # Screenshot: input frame
    save_screenshot(ref_frame, "input_frame.png")

    # Screenshot: detected scoreboard
    roi = detector.detect(ref_frame)
    detected_vis = detector.draw_bbox(ref_frame, roi)
    save_screenshot(detected_vis, "detected_scoreboard.png")
    print("  Scoreboard detected successfully.")

    # Screenshot: preprocessed scoreboard
    sb_crop = detector.crop(ref_frame, roi)
    stages = preprocessor.preprocess_full_scoreboard(sb_crop)
    save_screenshot(stages["enhanced"], "preprocessed_scoreboard.png")

    # Screenshot: segmented grid
    grid_vis = segmenter.draw_grid(ref_frame)
    save_screenshot(grid_vis, "segmented_grid.png")

    # Save representative OCR cell debug outputs in screenshots/ocr_cells/
    ocr_cell_dir = config.OCR_CELLS_DIR
    os.makedirs(ocr_cell_dir, exist_ok=True)

    # Grab clean frame at 20.0s for representative OCR cell outputs
    sample_frame_20s = vp.read_frame_at(20.0)
    if sample_frame_20s is not None:
        sample_cells = segmenter.segment(sample_frame_20s)
        for c in sample_cells:
            p_name = config.PLAYER_ROWS[c.player_idx]["name_label"]
            f_num = c.frame_idx + 1
            variants = preprocessor.preprocess_cell_variants(c.crop)
            if not variants:
                continue
            best_prep = variants[0][1]

            if c.cell_type == "name" and c.frame_idx == -1:
                cv2.imwrite(os.path.join(ocr_cell_dir, f"player_{p_name}.png"), best_prep)
            elif c.cell_type == "total" and c.frame_idx == 10:
                cv2.imwrite(os.path.join(ocr_cell_dir, f"total_{p_name}.png"), best_prep)
            elif c.cell_type in ("shot", "score") and f_num in (1, 2, 3, 4):
                # Save both standard naming and frame-specific naming
                cv2.imwrite(os.path.join(ocr_cell_dir, f"{p_name}_frame{f_num}_{c.cell_type}.png"), best_prep)
                cv2.imwrite(os.path.join(ocr_cell_dir, f"{p_name}_frame{f_num:02d}_{c.cell_type}.png"), best_prep)

    print(f"  Saved representative OCR debug cell images -> screenshots/ocr_cells/")

    # ── 4. Process video frames (temporal sampling + OCR) ────────────────────
    print("\nProcessing video frames…")
    from src.ocr_processor import _get_reader
    _get_reader()

    import math
    total_sampled_frames = math.ceil(meta['duration_seconds'] / config.SAMPLE_INTERVAL_SEC)

    t0 = time.time()
    frame_count = 0
    valid_shot_count = 0
    valid_score_count = 0
    rejected_count = 0

    for ts, frame in vp.sample_frames(config.SAMPLE_INTERVAL_SEC):
        frame_count += 1
        print(f"Processing sampled frame {frame_count}/{total_sampled_frames}")

        cells = segmenter.segment(frame)
        snapshot_state = {}

        for cell in cells:
            variants = preprocessor.preprocess_cell_variants(cell.crop)
            text, conf = read_cell_smart(
                variants, cell_type=cell.cell_type, min_confidence=config.MIN_OCR_CONFIDENCE
            )

            if text:
                if cell.cell_type in ("shot", "shot_1", "shot_2", "shot_3"):
                    valid_shot_count += 1
                elif cell.cell_type in ("score", "total"):
                    valid_score_count += 1
            else:
                rejected_count += 1

            aggregator.record(ts, cell.player_idx,
                              cell.frame_idx, cell.cell_type, text)

            p_label = config.PLAYER_ROWS[cell.player_idx]["name_label"]
            f_label = f"F{cell.frame_idx+1}" if cell.frame_idx >= 0 and cell.frame_idx < 10 else ("TTL" if cell.frame_idx == 10 else "NAME")
            key = f"{p_label}_{f_label}_{cell.cell_type}"
            snapshot_state[key] = text

        aggregator.record_snapshot(ts, snapshot_state)

        if frame_count % 10 == 0:
            import gc
            gc.collect()

    elapsed = time.time() - t0
    print(f"  Processed {frame_count} sampled frames in {elapsed:.1f}s")
    print(f"  Total OCR observations: {aggregator.get_observation_count()}")

    # ── 5. Temporal aggregation ──────────────────────────────────────────────
    print("\nPerforming temporal validation…")
    aggregated = aggregator.aggregate()
    snapshots = aggregator.get_snapshots()
    print(f"  Distinct scoreboard states detected: {len(snapshots)}")

    # ── 6. Build structured output ───────────────────────────────────────────
    print("\nSaving results…")
    final_data = build_structured_output(meta, aggregated, snapshots)

    save_json(final_data)
    save_csv(final_data)

    # Screenshot: final output
    save_final_output_image(ref_frame, detector, segmenter, final_data)

    # ── 7. Annotated video ───────────────────────────────────────────────────
    print("\nGenerating annotated video…")
    create_annotated_video(vp, detector, segmenter, final_data)

    # ── 8. Summary & Execution Quality Report ─────────────────────────────────
    vp.release()
    banner("PROCESSING COMPLETE")

    print(f"  Players detected: {len(config.PLAYER_ROWS)}")
    print(f"  Frames mapped: 10")
    print(f"  Sampled video frames processed: {frame_count}")
    print(f"  Total OCR observations processed: {aggregator.get_observation_count()}")
    print(f"  Valid shot cell observations: {valid_shot_count}")
    print(f"  Valid cumulative-score observations: {valid_score_count}")
    print(f"  Rejected low-confidence OCR observations: {rejected_count}")
    print()
    print("  [OK] JSON saved       ->", config.OUTPUT_JSON)
    print("  [OK] CSV saved        ->", config.OUTPUT_CSV)
    print("  [OK] Annotated video  ->", config.OUTPUT_VIDEO)
    print("  [OK] Screenshots      ->", config.SCREENSHOT_DIR)
    print("  [OK] OCR Cell Debugs  ->", config.OCR_CELLS_DIR)

    # Print extracted scoreboard summary
    print("\n--- Extracted Scoreboard ---")
    for p in final_data["scoreboard_final"]["players"]:
        name = p["name"]
        total = p["total"]
        shots = []
        for i in range(1, 11):
            fr = p["frames"].get(str(i), {})
            s = fr.get("shot", "")
            sc = fr.get("cumulative_score", "")
            shots.append(f"{s if s else '_'}({sc if sc else '_'})")
        print(f"  {name}: {' | '.join(shots)}  TTL={total}")

    print()


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bowling Scoreboard Data Extraction using Computer Vision & OCR",
    )
    parser.add_argument(
        "--video", type=str, default=config.DEFAULT_VIDEO,
        help="Path to the input bowling scoreboard video",
    )
    args = parser.parse_args()
    main(args.video)
