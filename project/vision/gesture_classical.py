"""
Classical finger counter (component B baseline): skin segmentation + convexity defects.

1. Skin mask in YCrCb (Cr 135-175, Cb 80-130 is robust to brightness because Y is ignored), blur + morphology.
2. Largest contour = hand (must cover > 2 % of the image).
3. Convexity defects of the hand contour: a deep defect with an angle < 90 deg is the valley between two
   extended fingers, so fingers = valleys + 1. With no valleys: one finger if the contour reaches more than
   2x the palm radius (largest inscribed circle) from the palm centre, otherwise a fist (0).
"""

import time
from typing import Optional

import cv2
import numpy as np


class ClassicalGestureCounter:
    name = "classical"

    def __init__(self, cr=(135, 175), cb=(80, 130)):
        self.lo = np.array([0, cr[0], cb[0]], np.uint8)
        self.hi = np.array([255, cr[1], cb[1]], np.uint8)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.last_contour = None
        self.last_valleys = []
        self.latency_ms = 0.0

    def count(self, frame: np.ndarray) -> Optional[int]:
        """Number of extended fingers (0-5), or None if no hand-sized skin region is found."""
        t0 = time.perf_counter()
        self.last_contour, self.last_valleys = None, []
        result = self._count(frame)
        self.latency_ms = (time.perf_counter() - t0) * 1e3
        return result

    def _count(self, frame):
        ycrcb = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, self.lo, self.hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        hand = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(hand)
        if area < 0.02 * frame.shape[0] * frame.shape[1]:
            return None
        self.last_contour = hand

        hull_idx = cv2.convexHull(hand, returnPoints=False)
        if hull_idx is None or len(hull_idx) < 4:
            return None
        try:
            defects = cv2.convexityDefects(hand, hull_idx)
        except cv2.error:  # self-intersecting contour
            return None
        hand_size = np.sqrt(area)
        if defects is not None:
            for s, e, f, depth in defects.reshape(-1, 4):  # (N,1,4) in OpenCV 4, (N,4) in OpenCV 5
                a, b, c = hand[s][0], hand[e][0], hand[f][0]
                v1, v2 = a - c, b - c
                cos = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
                if depth / 256.0 > 0.12 * hand_size and cos > 0.0:   # deep and sharper than 90 deg
                    self.last_valleys.append(tuple(c))
        if self.last_valleys:
            return min(len(self.last_valleys) + 1, 5)
        # No valleys: one finger or a fist. The palm is the largest inscribed circle (distance transform peak);
        # a single extended finger reaches far beyond it, a fist does not.
        hand_mask = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(hand_mask, [hand], -1, 255, -1)
        dist = cv2.distanceTransform(hand_mask, cv2.DIST_L2, 5)
        _, palm_r, _, (px, py) = cv2.minMaxLoc(dist)
        reach = np.max(np.hypot(hand[:, 0, 0] - px, hand[:, 0, 1] - py))
        return 1 if reach > 2.0 * palm_r else 0

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if self.last_contour is not None:
            cv2.drawContours(frame, [self.last_contour], -1, (0, 255, 0), 2)
            cv2.drawContours(frame, [cv2.convexHull(self.last_contour)], -1, (255, 0, 0), 1)
        for p in self.last_valleys:
            cv2.circle(frame, p, 6, (0, 0, 255), -1)
        return frame
