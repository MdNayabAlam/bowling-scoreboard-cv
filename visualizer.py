"""
Visualisation utilities.

• Save pipeline-stage debugging screenshots.
• Render an annotated output video with bounding boxes and extracted data.
"""

import cv2
import os
import logging
import numpy as np
from src import config

logger = logging.getLogger(__name__)


def save_screenshot(image: np.ndarray, name: str):
    """Save *image* to the screenshots directory."""
    os.makedirs(config.SCREENSHOT_DIR, exist_ok=True)
    path = os.path.join(config.SCREENSHOT_DIR, name)
    cv2.imwrite(path, image)
    logger.info("Screenshot saved -> %s", path)


def create_annotated_video(video_processor,
                           scoreboard_detector,
                           grid_segmenter,
                           final_data: dict,
                           output_path: str = None):
    """
    Re-read the source video and write an annotated MP4 showing:
      • Scoreboard bounding box
      • Grid overlay
      • Current extracted data as text overlay
      • Timestamp
    """
    output_path = output_path or config.OUTPUT_VIDEO
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    vp = video_processor
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path, fourcc, vp.fps, (vp.width, vp.height)
    )

    if not writer.isOpened():
        logger.error("Cannot open VideoWriter for %s", output_path)
        return

    # Build a compact text summary of extracted scores
    summary_lines = _build_summary_lines(final_data)

    frame_idx = 0
    vp.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    total = vp.frame_count
    log_interval = max(1, total // 10)

    while True:
        ret, frame = vp.cap.read()
        if not ret:
            break

        timestamp = frame_idx / vp.fps

        # Draw scoreboard bounding box
        roi = scoreboard_detector.detect(frame)
        annotated = scoreboard_detector.draw_bbox(frame, roi)

        # Draw grid lines
        annotated = grid_segmenter.draw_grid(annotated)

        # Timestamp label
        ts_text = f"Time: {timestamp:.1f}s"
        cv2.putText(annotated, ts_text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # Extracted data overlay (bottom-left with clean semi-transparent background box)
        line_height = 20
        box_h = len(summary_lines) * line_height + 15
        box_w = 480
        overlay_y1 = vp.height - box_h - 10
        overlay_y2 = vp.height - 10
        overlay_x1 = 10
        overlay_x2 = overlay_x1 + box_w

        # Semi-transparent dark background for readability
        overlay_bg = annotated[overlay_y1:overlay_y2, overlay_x1:overlay_x2].copy()
        cv2.rectangle(overlay_bg, (0, 0), (box_w, box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay_bg, 0.65, annotated[overlay_y1:overlay_y2, overlay_x1:overlay_x2], 0.35, 0,
                        annotated[overlay_y1:overlay_y2, overlay_x1:overlay_x2])
        cv2.rectangle(annotated, (overlay_x1, overlay_y1), (overlay_x2, overlay_y2), (0, 255, 255), 1)

        y_curr = overlay_y1 + 16
        for i, line in enumerate(summary_lines):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(annotated, line, (overlay_x1 + 10, y_curr),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        color, 1, cv2.LINE_AA)
            y_curr += line_height

        writer.write(annotated)
        frame_idx += 1
        if frame_idx % log_interval == 0:
            pct = int(100 * frame_idx / total)
            logger.info("  Annotated video progress: %d%%", pct)

    writer.release()
    logger.info("Annotated video saved -> %s", output_path)


def save_final_output_image(frame: np.ndarray,
                            scoreboard_detector,
                            grid_segmenter,
                            final_data: dict):
    """
    Create and save a single composite screenshot showing detection +
    grid + extracted text overlay.
    """
    roi = scoreboard_detector.detect(frame)
    vis = scoreboard_detector.draw_bbox(frame, roi)
    vis = grid_segmenter.draw_grid(vis)

    summary_lines = _build_summary_lines(final_data)
    y_offset = frame.shape[0] - 15 - len(summary_lines) * 16
    for line in summary_lines:
        y_offset += 16
        cv2.putText(vis, line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (0, 255, 255), 1, cv2.LINE_AA)

    save_screenshot(vis, "final_output.png")


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_summary_lines(final_data: dict) -> list[str]:
    """Build compact text lines summarising the extracted scoreboard."""
    lines = ["--- Extracted Scoreboard ---"]
    players = final_data.get("scoreboard_final", {}).get("players", [])
    for p in players:
        name = p.get("name", "?")
        total = p.get("total", "?")
        shots = []
        for i in range(1, 11):
            fr = p.get("frames", {}).get(str(i), {})
            shot_val = fr.get("shot", "")
            shots.append(shot_val if shot_val else "_")
        shots_str = " | ".join(shots)
        lines.append(f"{name}: {shots_str}  TTL={total}")
    return lines
