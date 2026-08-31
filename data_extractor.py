"""
Data extraction and formatting.

Takes the aggregated OCR results and formats them into the final JSON and
CSV structures.
"""

import json
import csv
import os
import logging
from src import config

logger = logging.getLogger(__name__)


def build_structured_output(video_meta: dict,
                            aggregated: dict,
                            snapshots: list[dict]) -> dict:
    """
    Build the final JSON-serialisable output dictionary.

    Parameters
    ----------
    video_meta : dict
        From VideoProcessor.get_metadata().
    aggregated : dict
        { (player_idx, frame_idx, cell_type) : value }
    snapshots : list[dict]
        Temporal snapshots with timestamps.

    Returns
    -------
    dict  – ready for json.dump()
    """
    num_players = len(config.PLAYER_ROWS)
    num_frames = 10

    players = []
    for p in range(num_players):
        default_name = config.PLAYER_ROWS[p]["name_label"]
        name = aggregated.get((p, -1, "name"), default_name)
        if not name or name.isspace() or name not in ("J", "V", "P", "T"):
            name = default_name

        frames = {}
        for f in range(num_frames):
            shot = aggregated.get((p, f, "shot"), "")
            score = aggregated.get((p, f, "score"), "")
            frames[str(f + 1)] = {"shot": shot, "cumulative_score": score}

        # Total is in index 10 (the TTL column)
        total = aggregated.get((p, 10, "total"), "")

        players.append({
            "name": name,
            "frames": frames,
            "total": total,
        })

    output = {
        "video_metadata": video_meta,
        "scoreboard_final": {"players": players},
        "scoreboard_states": snapshots,
    }
    return output


def save_json(data: dict, path: str = None):
    """Write the structured data to a JSON file cleanly."""
    path = path or config.OUTPUT_JSON
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("JSON saved -> %s", path)


def save_csv(data: dict, path: str = None):
    """
    Flatten the structured data into a CSV with one row per player-frame.
    Columns: Player, Frame, Shot, CumulativeScore, Total
    """
    path = path or config.OUTPUT_CSV
    os.makedirs(os.path.dirname(path), exist_ok=True)

    rows = []
    for player in data["scoreboard_final"]["players"]:
        name = player["name"]
        total = player["total"]
        for frame_num, cell in player["frames"].items():
            rows.append({
                "Player": name,
                "Frame": str(frame_num),
                "Shot": str(cell["shot"]),
                "CumulativeScore": str(cell["cumulative_score"]),
                "Total": str(total),
            })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Player", "Frame", "Shot", "CumulativeScore", "Total"]
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("CSV saved -> %s", path)
