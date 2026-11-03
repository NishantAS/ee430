"""
The flange camera is plugged into the laptop over USB and read here with OpenCV.
A background thread keeps only the newest frame, so slow processing never sees stale, queued images.
"""

import atexit
import threading
import time
from typing import Optional

import cv2
import numpy as np

import config


class Camera:
    def __init__(self, index: int):
        self.index = index
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {index}. Run: python calibrate.py camera")
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_SIZE[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_SIZE[1])
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._frame, self._frame_id, self._last_id = None, 0, -1
        self._cond = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        # Always stop the reader thread before Python exits (also after an exception): killing it while it is
        # inside OpenCV's capture code aborts the interpreter ("FATAL: exception not rethrown").
        atexit.register(self.close)

    def _reader(self):
        while self._running:
            ok, img = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._cond:
                self._frame, self._frame_id = img, self._frame_id + 1
                self._cond.notify_all()

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Newest frame not returned before (waits up to timeout); None if the camera delivers nothing."""
        with self._cond:
            self._cond.wait_for(lambda: self._frame_id != self._last_id, timeout=timeout)
            if self._frame is None:
                return None
            self._last_id = self._frame_id
            return self._frame.copy()

    def close(self):
        if not self._running:
            return
        self._running = False
        self._thread.join(timeout=2.0)
        self._cap.release()


def show(frame: np.ndarray, title: str = "EE430", wait_ms: int = 1) -> int:
    """imshow + waitKey; returns the pressed key code (-1 if none)."""
    cv2.imshow(title, frame)
    return cv2.waitKey(wait_ms) & 0xFF if wait_ms >= 0 else -1


def put_lines(frame: np.ndarray, lines, origin=(10, 25), color=(0, 255, 255)) -> np.ndarray:
    """Draws several text lines with a dark outline so they are readable on any background."""
    x, y = origin
    for i, text in enumerate(lines):
        pos = (x, y + 24 * i)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    return frame
