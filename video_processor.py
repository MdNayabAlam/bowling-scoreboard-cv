"""
Video I/O utilities: open a video, read metadata, iterate frames.
"""

import cv2
import logging

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Wraps cv2.VideoCapture with convenience helpers."""

    def __init__(self, video_path: str):
        self.path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0.0

    # ── public helpers ───────────────────────────────────────────────────────
    def get_metadata(self) -> dict:
        """Return video metadata as a plain dict (JSON-friendly)."""
        return {
            "filename": self.path,
            "resolution": f"{self.width}x{self.height}",
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_seconds": round(self.duration, 2),
        }

    def read_frame_at(self, timestamp_sec: float):
        """Seek to *timestamp_sec* and return the BGR frame (or None)."""
        frame_idx = int(timestamp_sec * self.fps)
        if frame_idx >= self.frame_count:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        return frame if ret else None

    def sample_frames(self, interval_sec: float):
        """
        Generator that yields (timestamp_sec, frame) tuples at the given
        interval across the entire video.
        """
        ts = 0.0
        while ts < self.duration:
            frame = self.read_frame_at(ts)
            if frame is not None:
                yield round(ts, 2), frame
            ts += interval_sec

    def iterate_all(self):
        """Generator yielding (frame_index, frame) for every frame."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield idx, frame
            idx += 1

    def release(self):
        self.cap.release()

    def __del__(self):
        try:
            self.cap.release()
        except Exception:
            pass
