"""
Deep-learning finger counter (bonus E): Google MediaPipe Hand Landmarker (palm-detection CNN + 21-keypoint
hand-landmark regression network, pretrained on ~30k annotated real hand images plus synthetic renders).

The network gives 21 3-D keypoints; the finger count is read from their geometry, which is invariant to
hand rotation and scale (the camera looks down at the hand from any yaw):
  - index..pinky are extended if the tip is farther from the wrist than the PIP joint (by 10 %),
  - the thumb is extended if its tip is far from the index-finger base relative to the palm size.

Model file (7.8 MB) is downloaded once to assets/ (config.HAND_MODEL).
"""

import time
import urllib.request
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import config

WRIST, THUMB_IP, THUMB_TIP, INDEX_MCP, MIDDLE_MCP = 0, 3, 4, 5, 9
FINGERS = [(6, 8), (10, 12), (14, 16), (18, 20)]  # (PIP, TIP) of index, middle, ring, pinky
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
         (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def ensure_model():
    if not config.HAND_MODEL.exists():
        print(f"Downloading MediaPipe hand model to {config.HAND_MODEL} ...")
        config.HAND_MODEL.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(config.HAND_MODEL_URL, config.HAND_MODEL)
    return str(config.HAND_MODEL)


def count_from_landmarks(pts: np.ndarray) -> int:
    """pts: (21, 3) normalised landmarks. Returns the number of extended digits (0-5)."""
    d = lambda i, j: float(np.linalg.norm(pts[i, :2] - pts[j, :2]))
    palm = d(WRIST, MIDDLE_MCP) + 1e-9
    n = sum(d(WRIST, tip) > 1.1 * d(WRIST, pip) for pip, tip in FINGERS)
    thumb_out = d(THUMB_TIP, INDEX_MCP) > 0.6 * palm and d(THUMB_TIP, INDEX_MCP) > d(THUMB_IP, INDEX_MCP)
    return n + int(thumb_out)


class MediaPipeGestureCounter:
    name = "mediapipe"

    def __init__(self, min_confidence: float = 0.5):
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.IMAGE, num_hands=1,
            min_hand_detection_confidence=min_confidence, min_hand_presence_confidence=min_confidence)
        self._landmarker = vision.HandLandmarker.create_from_options(opts)
        self.last_points = None
        self.latency_ms = 0.0

    def count(self, frame: np.ndarray) -> Optional[int]:
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        self.latency_ms = (time.perf_counter() - t0) * 1e3
        if not res.hand_landmarks:
            self.last_points = None
            return None
        self.last_points = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]])
        return count_from_landmarks(self.last_points)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if self.last_points is not None:
            h, w = frame.shape[:2]
            px = (self.last_points[:, :2] * [w, h]).astype(int)
            for i, j in BONES:
                cv2.line(frame, tuple(px[i]), tuple(px[j]), (0, 255, 0), 2)
            for p in px:
                cv2.circle(frame, tuple(p), 3, (0, 0, 255), -1)
        return frame
